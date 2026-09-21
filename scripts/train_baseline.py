"""Train the layer-1 baseline: ResNet18 features + logistic regression.

Requires the `baseline` extra: pip install -e ".[baseline]"
"""

import argparse
from pathlib import Path

from defectbench.config import get_settings
from defectbench.dataset import load_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=None, help="Path to baseline_train_manifest.jsonl")
    args = parser.parse_args()

    settings = get_settings()
    manifest_path = args.manifest or settings.artifacts_dir / "baseline_train_manifest.jsonl"
    samples = load_manifest(manifest_path)
    if not samples:
        raise SystemExit(f"Empty manifest: {manifest_path}. Run scripts/build_eval_set.py first.")

    from defectbench.baseline import BaselineClassifier

    paths = [s.path for s in samples]
    labels = [s.is_defect for s in samples]

    model = BaselineClassifier()
    print(f"Extracting features for {len(paths)} images (CPU, ResNet18)...")
    stats = model.train(paths, labels)
    print(f"Train accuracy: {stats['train_accuracy']:.4f} on {stats['n_samples']} images")

    out_path = settings.artifacts_dir / "baseline.pkl"
    model.save(out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
