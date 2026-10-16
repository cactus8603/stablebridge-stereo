#!/usr/bin/env python3
"""Download, verify, and extract only the RoCo-Spring stereo assets."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path


SCRIPT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_ROOT
sys.path.insert(0, str(PROJECT_ROOT))

from stablebridge.paths import writable_output
from stablebridge.archives import extract_one, md5sum  # noqa: E402


MANIFEST_PATH = PROJECT_ROOT / "configs" / "datasets.json"
SPRING_REQUIRED = (
    "test_frame_left.zip",
    "test_frame_right.zip",
    "train_frame_left.zip",
    "train_frame_right.zip",
    "train_disp1_left.zip",
)
SAFETY_RESERVE_BYTES = 20 * 1024**3


def log(message: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S"), message, flush=True)


def load_items() -> list[tuple[str, dict]]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    spring = {item["filename"]: item for item in manifest["spring"]["files"]}
    items = [("spring", spring[name]) for name in SPRING_REQUIRED]
    items.extend(
        ("robustspring", item)
        for item in sorted(
            manifest["robustspring"]["files"],
            key=lambda value: value["filename"],
        )
    )
    return items


def marker_is_ready(dataset_root: Path, item: dict) -> bool:
    marker = dataset_root / ".extracted" / f"{item['filename']}.json"
    if not marker.is_file():
        return False
    try:
        saved = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return saved.get("md5") == item["md5"]


def download(item: dict, archive: Path) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    partial = archive.with_suffix(archive.suffix + ".part")
    expected_size = int(item["size_bytes"])

    if archive.is_file():
        if archive.stat().st_size == expected_size and md5sum(archive) == item["md5"]:
            log(f"already verified: {archive.name}")
            return
        raise RuntimeError(f"Existing archive failed verification: {archive}")

    url = f"https://darus.uni-stuttgart.de/api/access/datafile/{item['datafile_id']}"
    log(f"downloading: {archive.name}")
    subprocess.run(
        [
            "curl",
            "--location",
            "--fail",
            "--show-error",
            "--retry",
            "12",
            "--retry-delay",
            "10",
            "--retry-all-errors",
            "--silent",
            "--continue-at",
            "-",
            "--output",
            str(partial),
            url,
        ],
        check=True,
    )
    if partial.stat().st_size != expected_size:
        raise RuntimeError(
            f"Size mismatch for {partial}: {partial.stat().st_size} != {expected_size}"
        )
    actual_md5 = md5sum(partial)
    if actual_md5 != item["md5"]:
        raise RuntimeError(f"MD5 mismatch for {partial}: {actual_md5} != {item['md5']}")
    partial.replace(archive)
    log(f"verified: {archive.name}")


def require_extract_space(target: Path, archive: Path) -> None:
    with zipfile.ZipFile(archive) as handle:
        uncompressed_bytes = sum(member.file_size for member in handle.infolist())
    available = shutil.disk_usage(target).free
    required = uncompressed_bytes + SAFETY_RESERVE_BYTES
    if available < required:
        raise RuntimeError(
            "Insufficient free space to extract "
            f"{archive.name}: need {required / 1024**3:.1f} GiB including reserve, "
            f"have {available / 1024**3:.1f} GiB"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--dataset",choices=("spring","spring-train","spring-test","robustspring","all"),default="all")
    args = parser.parse_args()
    root = writable_output(args.root)
    root.mkdir(parents=True, exist_ok=True)

    items = [(name,item) for name,item in load_items() if
             args.dataset == "all" or name == args.dataset or
             (name == "spring" and args.dataset.startswith("spring-") and
              item["filename"].startswith(args.dataset.split("-", 1)[1] + "_"))]
    log(f"stereo-only download: {len(items)} archives")
    def process(index: int, dataset: str, item: dict) -> None:
        dataset_root = root / "datasets" / dataset
        archive = dataset_root / "archives" / item["filename"]
        if marker_is_ready(dataset_root, item):
            log(f"[{index}/{len(items)}] already extracted: {item['filename']}")
            archive.unlink(missing_ok=True)
            return

        log(f"[{index}/{len(items)}] starting {item['filename']}")
        download(item, archive)
        require_extract_space(root, archive)
        extract_one(dataset, archive, item, dataset_root)
        archive.unlink()
        log(f"removed archive after extraction: {item['filename']}")

    workers = max(1, min(args.workers, 8))
    log(f"parallel workers: {workers}")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(process, index, dataset, item)
            for index, (dataset, item) in enumerate(items, start=1)
        ]
        for future in as_completed(futures):
            future.result()

    complete = root / "STEREO_DOWNLOAD_COMPLETE.json"
    complete.write_text(
        json.dumps(
            {
                "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "archives": len(items),
                "scope": args.dataset,
                "archives_deleted_after_verified_extraction": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    log(f"complete: {complete}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
