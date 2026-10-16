from pathlib import Path
import os
ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = Path(os.environ.get("STABLEBRIDGE_WEIGHTS", ROOT / "weights"))
def checkpoint_path(*parts):
    return WEIGHTS.joinpath(*parts)


def writable_output(path):
    """Keep generated files outside immutable release inputs."""
    path = Path(path).expanduser().resolve()
    if path.is_relative_to(ROOT):
        relative = path.relative_to(ROOT).parts
        if not relative or relative[0] != "runs":
            raise ValueError("Use runs/ or an external output to preserve the release")
    return path
