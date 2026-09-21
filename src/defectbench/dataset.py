"""MVTec AD dataset parsing and eval-set construction.

Expected layout (standard MVTec AD):

    <root>/<category>/train/good/*.png
    <root>/<category>/test/<defect_type|good>/*.png
    <root>/<category>/ground_truth/<defect_type>/*_mask.png
"""

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

GOOD_LABEL = "good"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class Sample:
    path: str
    category: str
    split: str
    defect_type: str  # "good" or the MVTec defect class name
    is_defect: bool


def iter_split(root: Path, category: str, split: str) -> Iterator[Sample]:
    split_dir = root / category / split
    if not split_dir.is_dir():
        return
    for label_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        is_defect = label_dir.name != GOOD_LABEL
        for img in sorted(label_dir.iterdir()):
            if img.suffix.lower() in IMAGE_EXTS:
                yield Sample(
                    path=str(img.resolve()),
                    category=category,
                    split=split,
                    defect_type=label_dir.name,
                    is_defect=is_defect,
                )


def list_categories(root: Path) -> list[str]:
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if (p / "test").is_dir())


def build_eval_manifest(
    root: Path,
    categories: list[str],
    split: str = "test",
    max_per_class: int | None = None,
) -> list[Sample]:
    samples: list[Sample] = []
    for category in categories:
        by_class: dict[str, list[Sample]] = {}
        for sample in iter_split(root, category, split):
            by_class.setdefault(sample.defect_type, []).append(sample)
        for group in by_class.values():
            if max_per_class is not None:
                group = group[:max_per_class]
            samples.extend(group)
    return samples


def save_manifest(samples: list[Sample], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(asdict(sample), ensure_ascii=False) + "\n")
    return out_path


def load_manifest(path: Path) -> list[Sample]:
    samples = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(Sample(**json.loads(line)))
    return samples
