"""Pack the MVTec images into a POSIX-compliant zip for Kaggle/Colab uploads.

PowerShell's Compress-Archive writes backslash separators inside the archive,
which violates the ZIP spec and gets rejected by web uploaders (e.g. Kaggle).

Usage: python scripts/pack_dataset_zip.py [--data-dir data] [--out artifacts/mvtec_images.zip]
"""

import argparse
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/mvtec_images.zip"))
    args = parser.parse_args()

    src = (args.data_dir / "mvtec_anomaly_detection").resolve()
    if not src.is_dir():
        raise SystemExit(f"Not found: {src}. Run scripts/download_mvtec.py first.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # PNG files are already compressed -> ZIP_STORED is fast and big enough.
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_STORED) as zf:
        for path in sorted(src.rglob("*")):
            if path.is_file():
                arcname = path.relative_to(src.parent).as_posix()  # forward slashes, per ZIP spec
                zf.write(path, arcname)

    size_mb = args.out.stat().st_size / 1e6
    print(f"Packed {size_mb:.0f} MB -> {args.out}")


if __name__ == "__main__":
    main()
