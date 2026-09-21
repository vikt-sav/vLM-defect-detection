"""Select a validation split from the fine-tune manifest and re-emit train/val JSONLs.

Deterministic (seed). Used by the v5 notebook: checkpoints are compared on val,
the best one is merged and exported.

Usage: python scripts/split_finetune_val.py [--n-val 24] [--seed 42]
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--n-val", type=int, default=24, help="Validation records (balanced across classes)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    src = args.artifacts / "vlm_finetune_train.jsonl"
    records = [json.loads(line) for line in src.open(encoding="utf-8") if line.strip()]

    # stratified by binary is_defect: val must contain both normal and defective
    by_class: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        key = "good" if not json.loads(r["messages"][2]["content"])["is_defect"] else "defect"
        by_class[key].append(r)

    rng = random.Random(args.seed)
    val, train = [], []
    half = args.n_val // 2
    for cls, group in sorted(by_class.items()):
        rng.shuffle(group)
        take = min(half if cls == "good" else args.n_val - half, len(group) - 1)
        val.extend(group[:take])
        train.extend(group[take:])

    rng.shuffle(train)
    train_out = args.artifacts / "vlm_finetune_train_split.jsonl"
    val_out = args.artifacts / "vlm_finetune_val.jsonl"
    for path, rows in [(train_out, train), (val_out, val)]:
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def n_defect(rows):
        return sum(1 for r in rows if json.loads(r["messages"][2]["content"])["is_defect"])

    print(f"train: {len(train)} records ({n_defect(train)} defect)")
    print(f"val:   {len(val)} records ({n_defect(val)} defect)")
    print(f"written: {train_out.name}, {val_out.name}")


if __name__ == "__main__":
    main()
