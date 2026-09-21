"""Build the reference eval set and the baseline training set from MVTec AD.

Methodology:
- eval set: the full MVTec `test` split (optionally capped per class);
- baseline training set: all `train/good` images (normal) plus a fixed fraction
  of test defect images, disjoint from the eval set (supervised baseline).

Outputs (JSONL in the artifacts dir):
- eval_manifest.jsonl           -> used by run_benchmark.py for every layer
- baseline_train_manifest.jsonl -> used by train_baseline.py
- vlm_finetune_train.jsonl      -> chat-format export for the QLoRA notebook
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from defectbench.config import get_settings
from defectbench.dataset import GOOD_LABEL, Sample, iter_split, save_manifest
from defectbench.vlm import SYSTEM_PROMPT, USER_PROMPT

SEVERITY_HINTS = {
    "bottle": "major",
    "screw": "major",
}


def split_defect_images(root: Path, categories: list[str], fraction_for_baseline: float, seed: int) -> dict[str, list[Sample]]:
    """Deterministically split each defect class between baseline-train and eval."""
    rng = random.Random(seed)
    assignment: dict[str, list[Sample]] = {"train": [], "eval": []}
    for category in categories:
        by_class: dict[str, list[Sample]] = defaultdict(list)
        for sample in iter_split(root, category, "test"):
            if sample.is_defect:
                by_class[sample.defect_type].append(sample)
        for group in by_class.values():
            shuffled = list(group)
            rng.shuffle(shuffled)
            cut = max(1, int(len(shuffled) * fraction_for_baseline))
            assignment["train"].extend(shuffled[:cut])
            assignment["eval"].extend(shuffled[cut:])
        # all good test images go to eval; baseline normals come from train/good
        assignment["eval"].extend(s for s in iter_split(root, category, "test") if not s.is_defect)
    return assignment


def make_finetune_records(train_samples: list[Sample], data_dir: Path) -> list[dict]:
    """Export baseline-train samples as chat records for VLM fine-tuning.

    Image paths are stored relative to the data dir so that the dataset can be
    zipped and consumed verbatim in Colab.
    """
    records = []
    for sample in train_samples:
        target = {
            "defect_type": sample.defect_type,
            "is_defect": sample.is_defect,
            "location": "unknown",
            "severity": "none" if not sample.is_defect else SEVERITY_HINTS.get(sample.category, "minor"),
            "confidence": 1.0,
        }
        relative_image = str(Path(sample.path).resolve().relative_to(data_dir.resolve()))
        records.append(
            {
                "messages": [
                    # Same prompts the benchmark sends at inference: train == eval distribution.
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT},
                    {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
                ],
                "image": relative_image,
            }
        )
    return records


def save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--max-good-per-category", type=int, default=50, help="Cap test/good images in the eval set")
    parser.add_argument("--defect-fraction-for-baseline", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    settings = get_settings()
    categories = args.categories or settings.category_list
    artifacts = settings.artifacts_dir

    # 1. Eval set: all test defect images not used for baseline training + capped test/good.
    split = split_defect_images(settings.mvtec_root, categories, args.defect_fraction_for_baseline, args.seed)
    eval_samples = split["eval"]

    good_cap = args.max_good_per_category
    if good_cap is not None:
        by_cat: dict[str, list[Sample]] = defaultdict(list)
        for sample in eval_samples:
            if not sample.is_defect:
                by_cat[sample.category].append(sample)
        for category, good_samples in by_cat.items():
            if len(good_samples) > good_cap:
                to_drop = set(id(s) for s in good_samples[good_cap:])
                eval_samples = [s for s in eval_samples if not (not s.is_defect and s.category == category and id(s) in to_drop)]

    save_manifest(eval_samples, artifacts / "eval_manifest.jsonl")

    # 2. Baseline training set: train/good + the defect half of the test split.
    baseline_train = []
    good_train: list[Sample] = []
    for category in categories:
        good_train.extend(s for s in iter_split(settings.mvtec_root, category, "train") if s.defect_type == GOOD_LABEL)
    baseline_train.extend(good_train)
    baseline_train.extend(split["train"])
    save_manifest(baseline_train, artifacts / "baseline_train_manifest.jsonl")

    # 3. VLM fine-tune export: defect half + good examples sampled to match its size.
    # Without good examples the model learns "always report a defect" and fails on normal images.
    rng = random.Random(args.seed)
    good_pool = list(good_train)
    rng.shuffle(good_pool)
    n_defect = len(split["train"])
    finetune_samples = split["train"] + good_pool[: min(n_defect, len(good_pool))]
    save_jsonl(make_finetune_records(finetune_samples, settings.data_dir), artifacts / "vlm_finetune_train.jsonl")

    n_eval_defect = sum(1 for s in eval_samples if s.is_defect)
    n_train_defect = sum(1 for s in baseline_train if s.is_defect)
    print(f"Categories: {', '.join(categories)}")
    print(f"Eval set:        {len(eval_samples)} images ({n_eval_defect} defective)")
    print(f"Baseline train:  {len(baseline_train)} images ({n_train_defect} defective)")
    print(f"VLM finetune:    {len(finetune_samples)} records ({n_defect} defective, {len(finetune_samples) - n_defect} good)")
    print(f"Written to {artifacts}")


if __name__ == "__main__":
    main()
