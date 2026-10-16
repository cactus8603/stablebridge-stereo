"""Bind resumed predictions and official packages to one immutable model contract."""
from pathlib import Path
import json
import torch
from stablebridge.backbone import EXPECTED_CHECKPOINT_SHA256, sha256_file, write_json
from stablebridge.paths import ROOT


def source_identity():
    return {p.relative_to(ROOT).as_posix(): sha256_file(p)
            for p in sorted((ROOT / "stablebridge").rglob("*.py"))}


def model_identity(checkpoint, bundle):
    checkpoint, bundle = Path(checkpoint), Path(bundle)
    digest = sha256_file(checkpoint)
    if digest != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("CroCo checkpoint SHA256 mismatch")
    model = json.loads(bundle.read_text())
    artifacts = [model["evidence"], *model["heads"]]
    for row in artifacts:
        path = Path(row["path"])
        path = path if path.is_absolute() else bundle.parent / path
        if sha256_file(path) != row["sha256"]:
            raise ValueError(f"Model artifact SHA256 mismatch: {path}")
    return {"checkpoint_sha256": digest, "bundle_sha256": sha256_file(bundle),
            "artifacts": [{"path": r["path"], "sha256": r["sha256"]} for r in artifacts]}


def prediction_contract(args):
    return {"schema_version": 1, "model": model_identity(args.checkpoint, args.bundle),
            "source_sha256": source_identity(),
            "clean_test_root": str(args.clean_test_root.resolve()),
            "robust_test_root": str(args.robust_test_root.resolve()),
            "tile_overlap": 0.9, "disparity_precision": "float32", "router_precision": "bfloat16",
            "batch_pairs": args.batch_pairs, "route_tile_batch_size": 1,
            "clean_identity_required": True, "device": args.device,
            "torch": torch.__version__, "cuda_build": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(args.device)}


def bind_output(root, contract):
    """Never adopt existing maps whose model or configuration is unknown."""
    root = Path(root)
    path = root / "prediction-contract.json"
    if path.exists():
        if json.loads(path.read_text()) != contract:
            raise ValueError("Prediction contract changed; use a new output directory")
    else:
        if root.exists() and any(root.rglob("*.dsp5")):
            raise ValueError("Existing predictions have no contract; use a new output directory")
        root.mkdir(parents=True, exist_ok=True)
        write_json(path, contract)


def validate_contract(root, checkpoint, bundle, clean_root, robust_root):
    path = Path(root) / "prediction-contract.json"
    contract = json.loads(path.read_text())
    if contract.get("schema_version") != 1 or contract.get("model") != model_identity(checkpoint, bundle):
        raise ValueError("Prediction model provenance mismatch")
    if contract.get("source_sha256") != source_identity():
        raise ValueError("Prediction source differs from this release")
    if contract.get("clean_test_root") != str(Path(clean_root).resolve()) or contract.get("robust_test_root") != str(Path(robust_root).resolve()):
        raise ValueError("Prediction input roots differ from packaging inputs")
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "content": contract}
