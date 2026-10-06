from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from tracks.stereo import package_submission
from tracks.stereo import submission_contract


def _write_dsp5(path: Path, *, nan: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.zeros((1080, 1920, 1), dtype=np.float32)
    if nan:
        values[0, 0, 0] = np.nan
    with h5py.File(path, "w") as handle:
        handle.create_dataset("disparity", data=values, compression="gzip")


def test_official_test_contract_has_1000_pairs() -> None:
    assert sum(submission_contract.TEST_SEQUENCE_LENGTHS.values()) == 1000
    assert submission_contract.EXPECTED_FILES_PER_VARIANT == 2000


def test_inspect_dsp5_checks_shape_dtype_and_values(tmp_path: Path) -> None:
    valid = tmp_path / "valid.dsp5"
    _write_dsp5(valid)
    result = submission_contract.inspect_dsp5(valid, check_values=True)
    assert result["shape"] == [1080, 1920, 1]
    assert result["dtype"] == "float32"
    assert result["finite"] is True

    invalid = tmp_path / "nan.dsp5"
    _write_dsp5(invalid, nan=True)
    with pytest.raises(ValueError, match="NaN/Inf"):
        submission_contract.inspect_dsp5(invalid, check_values=True)


def test_clean_package_contract(tmp_path: Path) -> None:
    clean = tmp_path / "disp1_submission.hdf5"
    with h5py.File(clean, "w") as handle:
        handle.create_dataset("disparity", data=np.array([1, 2], dtype=np.float16))
    result = submission_contract.inspect_submission_hdf5(
        clean, mode="clean", enforce_expected_size=False
    )
    assert result["datasets"]["disparity"]["elements"] == 2
    assert len(result["sha256"]) == 64


def test_mode_selection() -> None:
    assert package_submission.selected_modes("clean") == ("clean",)
    assert package_submission.selected_modes("robust") == ("robust",)
    assert package_submission.selected_modes("all") == ("clean", "robust")
