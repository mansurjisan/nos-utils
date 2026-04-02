"""Tests for HotstartProcessor."""

from pathlib import Path

import pytest

from nos_utils.config import ForcingConfig
from nos_utils.forcing.hotstart import HotstartProcessor, HotstartInfo

netCDF4 = pytest.importorskip("netCDF4")


@pytest.fixture
def mock_hotstart(tmp_path):
    """Create a mock hotstart.nc file."""
    hs_dir = tmp_path / "restart"
    hs_dir.mkdir()
    hs_file = hs_dir / "hotstart.nc"

    ds = netCDF4.Dataset(str(hs_file), "w")
    ds.createDimension("node", 100)
    ds.createDimension("nVert", 51)

    time_var = ds.createVariable("time", "f8")
    time_var[:] = 21600.0  # 6 hours in seconds

    iths_var = ds.createVariable("iths", "i4")
    iths_var[:] = 180  # 180 time steps

    ds.close()
    return hs_dir


class TestHotstartProcessor:
    def test_find_hotstart(self, mock_config, mock_hotstart, tmp_path):
        proc = HotstartProcessor(
            mock_config, mock_hotstart, tmp_path / "out",
        )
        result = proc.process()

        assert result.success
        assert result.metadata["ihot"] == 1
        assert result.metadata["time_seconds"] == 21600.0
        assert result.metadata["iths"] == 180

    def test_no_hotstart_cold_start(self, mock_config, tmp_path):
        """Missing hotstart -> cold start (ihot=0), still success."""
        proc = HotstartProcessor(
            mock_config, tmp_path / "empty", tmp_path / "out",
        )
        result = proc.process()

        assert result.success  # Non-fatal
        assert result.metadata["ihot"] == 0
        assert "cold start" in result.warnings[0].lower()

    def test_links_hotstart(self, mock_config, mock_hotstart, tmp_path):
        """Should create symlink to hotstart.nc in output dir."""
        out_dir = tmp_path / "out"
        proc = HotstartProcessor(mock_config, mock_hotstart, out_dir)
        proc.process()

        assert (out_dir / "hotstart.nc").exists()


class TestHotstartInfo:
    def test_time_days(self):
        info = HotstartInfo(
            filepath=Path("/test"), time_seconds=86400.0,
            iths=100, n_nodes=1000, n_levels=51,
        )
        assert info.time_days == 1.0

    def test_repr(self):
        info = HotstartInfo(
            filepath=Path("/test/hotstart.nc"), time_seconds=21600.0,
            iths=180, n_nodes=1684786, n_levels=63,
        )
        s = repr(info)
        assert "21600" in s
        assert "180" in s
