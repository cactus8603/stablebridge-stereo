from __future__ import annotations
import hashlib,json,shutil,time,zipfile
from pathlib import Path,PurePosixPath

def md5sum(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()

def normalized_member_path(dataset: str, archive_stem: str, name: str) -> Path | None:
    pure = PurePosixPath(name)
    if pure.is_absolute() or ".." in pure.parts:
        raise RuntimeError(f"Unsafe archive member: {name}")
    parts = [part for part in pure.parts if part not in ("", ".")]
    if not parts:
        return None

    if dataset == "spring":
        if parts[0].lower() != "spring":
            raise RuntimeError(f"Unexpected Spring archive root in {name}")
        return Path(*parts)

    if parts[0].lower() in {"robustspring", "robust_spring"}:
        parts = parts[1:]
    if not parts:
        return None
    if parts[0].lower() == "test":
        parts.insert(0, archive_stem)
    return Path("robust_spring", *parts)

def extract_one(
    dataset: str,
    archive: Path,
    item: dict,
    dataset_root: Path,
) -> None:
    marker_dir = dataset_root / ".extracted"
    marker = marker_dir / f"{archive.name}.json"
    if marker.is_file():
        saved = json.loads(marker.read_text(encoding="utf-8"))
        if saved.get("md5") == item["md5"]:
            print(f"already extracted: {archive.name}", flush=True)
            return

    extracted_root = dataset_root / "extracted"
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            relative = normalized_member_path(dataset, archive.stem, member.filename)
            if relative is None:
                continue
            target = extracted_root / relative
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with handle.open(member) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=16 * 1024 * 1024)

    marker_dir.mkdir(parents=True, exist_ok=True)
    temporary = marker.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            {
                "archive": archive.name,
                "md5": item["md5"],
                "extracted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(marker)
    print(f"extracted: {archive.name}", flush=True)
