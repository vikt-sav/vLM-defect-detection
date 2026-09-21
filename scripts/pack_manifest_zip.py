"""Pack ONLY the images referenced by the manifests into a POSIX zip for Kaggle.

Inputs:  artifacts/eval_manifest.jsonl + artifacts/vlm_finetune_train.jsonl
Output:  artifacts/mvtec_images_v5.zip (forward-slash paths, per ZIP spec)

Usage: python scripts/pack_manifest_zip.py [--data-dir data] [--out artifacts/mvtec_images_v5.zip]
"""

import argparse
import json
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/mvtec_images_v5.zip"))
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    referenced: set[str] = set()

    eval_manifest = args.artifacts / "eval_manifest.jsonl"
    for line in eval_manifest.open(encoding="utf-8"):
        if line.strip():
            referenced.add(str(Path(json.loads(line)["path"]).resolve()))

    finetune = args.artifacts / "vlm_finetune_train.jsonl"
    for line in finetune.open(encoding="utf-8"):
        if line.strip():
            referenced.add(str((data_dir / json.loads(line)["image"]).resolve()))

    assert referenced, "manifests are empty - run build_eval_set.py first"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_STORED) as zf:
        for abs_path in sorted(referenced):
            arc = Path(abs_path).relative_to(data_dir).as_posix()
            zf.write(abs_path, arc)
        # manifests travel with the images so the notebook needs a single upload
        for manifest in sorted(args.artifacts.glob("*.jsonl")):
            zf.write(manifest, manifest.name)

    n = len(referenced)
    size_mb = args.out.stat().st_size / 1e6
    print(f"Packed {n} images + {len(list(args.artifacts.glob('*.jsonl')))} manifests ({size_mb:.0f} MB) -> {args.out}")


if __name__ == "__main__":
    main()
