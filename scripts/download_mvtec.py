"""Download MVTec AD categories from the official MVTec mirrors and extract them.

Usage:
    python scripts/download_mvtec.py --categories bottle screw
    python scripts/download_mvtec.py --list
"""

import argparse
import sys
import tarfile
from pathlib import Path

import httpx
from tqdm import tqdm

OFFICIAL_URLS: dict[str, str] = {
    "bottle": "https://www.mydrive.ch/shares/150452/132a93367fb17cdf968dfb5c4013f6e7/download/420937370-1629958698/bottle.tar.xz",
    "cable": "https://www.mydrive.ch/shares/150453/10f960e07fec2838b2cd512586633a32/download/420937413-1629958794/cable.tar.xz",
    "capsule": "https://www.mydrive.ch/shares/150454/e0ce6dd74eb150f46c0d98131b3703f2/download/420937454-1629958872/capsule.tar.xz",
    "carpet": "https://www.mydrive.ch/shares/150455/eac7fbce84d93a5094e13f391170eca4/download/420937484-1629959013/carpet.tar.xz",
    "grid": "https://www.mydrive.ch/shares/150456/bb0b2e3dc804ccb8b4485e01f8e4493b/download/420937487-1629959044/grid.tar.xz",
    "hazelnut": "https://www.mydrive.ch/shares/150457/51d49f65e84bdc100ff8035d7dd783ea/download/420937545-1629959162/hazelnut.tar.xz",
    "leather": "https://www.mydrive.ch/shares/150458/923030ce14e10a7d147e95d0f8885f6b/download/420937607-1629959262/leather.tar.xz",
    "metal_nut": "https://www.mydrive.ch/shares/150459/c68856a21dca589b0f8ff6d4ee0f18f4/download/420937637-1629959294/metal_nut.tar.xz",
    "pill": "https://www.mydrive.ch/shares/150460/d4f1c04da67034ccc8d8b5fa3e73f244/download/420938129-1629960351/pill.tar.xz",
    "screw": "https://www.mydrive.ch/shares/150461/242f454cc6385e5693c4fd4b94567d1e/download/420938130-1629960389/screw.tar.xz",
    "tile": "https://www.mydrive.ch/shares/150462/5479f0fdc97bc6fa16eab0cb0cf0109f/download/420938133-1629960456/tile.tar.xz",
    "toothbrush": "https://www.mydrive.ch/shares/150463/895a1a5fb84b958f07417784b060434b/download/420938134-1629960477/toothbrush.tar.xz",
    "transistor": "https://www.mydrive.ch/shares/150464/99b8ea0332438d64fcc2475ee73f9c29/download/420938166-1629960554/transistor.tar.xz",
    "wood": "https://www.mydrive.ch/shares/150465/d5b4115b720cdb54d217e75636e6e374/download/420938383-1629960649/wood.tar.xz",
    "zipper": "https://www.mydrive.ch/shares/150466/bd155b557520edaf692d9bfdb915c24a/download/420938385-1629960680/zipper.tar.xz",
}


def download_category(category: str, root: Path, keep_archive: bool = False) -> Path:
    if category not in OFFICIAL_URLS:
        raise SystemExit(f"Unknown category '{category}'. Available: {', '.join(OFFICIAL_URLS)}")

    archives_dir = root / "archives"
    archives_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archives_dir / f"{category}.tar.xz"
    extract_to = root / "mvtec_anomaly_detection"

    if (extract_to / category / "test").is_dir():
        print(f"[skip] {category} already extracted")
        return extract_to / category

    url = OFFICIAL_URLS[category]
    print(f"[get] {category}: {url}")
    with httpx.stream("GET", url, timeout=60.0, follow_redirects=True) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with archive_path.open("wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=category) as bar:
            for chunk in response.iter_bytes(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))

    print(f"[extract] {archive_path.name} -> {extract_to}")
    with tarfile.open(archive_path, mode="r:xz") as tar:
        tar.extractall(extract_to, filter="data")

    if not keep_archive:
        archive_path.unlink(missing_ok=True)
    return extract_to / category


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--categories", nargs="+", default=["bottle", "screw"])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--list", action="store_true", help="List available categories and exit")
    parser.add_argument("--keep-archive", action="store_true")
    args = parser.parse_args()

    if args.list:
        print("Available categories:", ", ".join(OFFICIAL_URLS))
        return

    data_dir = args.data_dir.resolve()
    print(f"Data dir: {data_dir}")
    failures = []
    for category in args.categories:
        try:
            download_category(category, data_dir, keep_archive=args.keep_archive)
        except Exception as exc:
            failures.append((category, str(exc)))
    if failures:
        for category, error in failures:
            print(f"[fail] {category}: {error}", file=sys.stderr)
        sys.exit(1)
    print("Done.")


if __name__ == "__main__":
    main()
