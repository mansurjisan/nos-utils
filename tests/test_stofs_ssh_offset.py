"""Tests for STOFS SSH offset application in the Fortran path."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("netCDF4")
from netCDF4 import Dataset  # noqa: E402

from nos_utils.forcing.rtofs import RTOFSProcessor  # noqa: E402


def _write_fake_elev2d(path: Path, nt: int = 5, n_bnd: int = 10) -> None:
    """Create a minimal elev2D.th.nc with a writable time_series variable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with Dataset(str(path), "w") as ds:
        ds.createDimension("time", nt)
        ds.createDimension("nOpenBndNodes", n_bnd)
        ds.createDimension("nLevels", 1)
        ds.createDimension("nComponents", 1)
        v = ds.createVariable(
            "time_series", "f4",
            ("time", "nOpenBndNodes", "nLevels", "nComponents"),
            fill_value=-9999.0,
        )
        v[:] = 0.0


class TestApplySSHOffset:
    def test_adds_offset_to_all_records(self, tmp_path):
        nc = tmp_path / "elev2D.th.nc"
        _write_fake_elev2d(nc, nt=4, n_bnd=8)
        offset = 0.04

        RTOFSProcessor._apply_ssh_offset(nc, offset)

        with Dataset(str(nc)) as ds:
            vals = ds.variables["time_series"][:]
        # Every entry should be exactly the offset.
        assert np.allclose(vals, offset, atol=1e-6)

    def test_preserves_existing_values(self, tmp_path):
        nc = tmp_path / "elev2D.th.nc"
        _write_fake_elev2d(nc, nt=3, n_bnd=5)
        offset = 1.25

        # Populate with a known pattern.
        with Dataset(str(nc), "r+") as ds:
            ds.variables["time_series"][:] = np.arange(
                3 * 5 * 1 * 1, dtype=np.float32
            ).reshape(3, 5, 1, 1)

        RTOFSProcessor._apply_ssh_offset(nc, offset)

        with Dataset(str(nc)) as ds:
            vals = ds.variables["time_series"][:]
        expected = np.arange(3 * 5, dtype=np.float32).reshape(3, 5, 1, 1) + offset
        assert np.allclose(vals, expected, atol=1e-5)

    def test_zero_offset_is_noop(self, tmp_path):
        nc = tmp_path / "elev2D.th.nc"
        _write_fake_elev2d(nc, nt=2, n_bnd=3)
        with Dataset(str(nc), "r+") as ds:
            ds.variables["time_series"][:] = 0.5

        RTOFSProcessor._apply_ssh_offset(nc, 0.0)

        with Dataset(str(nc)) as ds:
            vals = ds.variables["time_series"][:]
        assert np.allclose(vals, 0.5)

    def test_missing_file_does_not_raise(self, tmp_path):
        """The helper logs and returns silently when the file is missing."""
        # Should not raise even though the file doesn't exist.
        RTOFSProcessor._apply_ssh_offset(tmp_path / "nope.nc", 0.04)
