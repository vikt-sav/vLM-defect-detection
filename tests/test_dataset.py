import json

from defectbench.dataset import (
    build_eval_manifest,
    iter_split,
    list_categories,
    load_manifest,
    save_manifest,
)


def make_mvtec_tree(root):
    base = root / "mvtec_anomaly_detection" / "bottle"
    for name, files in {
        "train/good": ["000.png", "001.png"],
        "test/good": ["100.png"],
        "test/broken_large": ["200.png", "201.png"],
        "test/broken_small": ["300.png"],
        "ground_truth/broken_large": ["200_mask.png"],
    }.items():
        d = base / name
        d.mkdir(parents=True, exist_ok=True)
        for f in files:
            (d / f).write_bytes(b"png")
    # noise that must be ignored
    (base / "test" / "notes.txt").write_text("readme", encoding="utf-8")
    return base


def test_iter_split_yields_samples(tmp_path):
    make_mvtec_tree(tmp_path)
    samples = list(iter_split(tmp_path / "mvtec_anomaly_detection", "bottle", "test"))
    assert len(samples) == 4  # 1 good + 2 broken_large + 1 broken_small
    good = [s for s in samples if not s.is_defect]
    assert len(good) == 1 and good[0].defect_type == "good"
    assert all(s.category == "bottle" and s.split == "test" for s in samples)


def test_iter_split_missing_dir(tmp_path):
    assert list(iter_split(tmp_path, "nope", "test")) == []


def test_list_categories(tmp_path):
    make_mvtec_tree(tmp_path)
    assert list_categories(tmp_path / "mvtec_anomaly_detection") == ["bottle"]


def test_build_eval_manifest(tmp_path):
    make_mvtec_tree(tmp_path)
    root = tmp_path / "mvtec_anomaly_detection"
    samples = build_eval_manifest(root, ["bottle"], max_per_class=1)
    defect_types = [s.defect_type for s in samples]
    assert defect_types.count("broken_large") == 1
    assert defect_types.count("good") == 1


def test_manifest_roundtrip(tmp_path):
    make_mvtec_tree(tmp_path)
    root = tmp_path / "mvtec_anomaly_detection"
    samples = build_eval_manifest(root, ["bottle"])
    out = tmp_path / "manifest.jsonl"
    save_manifest(samples, out)
    loaded = load_manifest(out)
    assert loaded == samples
    first = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert {"path", "category", "split", "defect_type", "is_defect"} <= set(first)
