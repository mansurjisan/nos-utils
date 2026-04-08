"""Tests for HRRRProcessor."""

from pathlib import Path

import pytest

from nos_utils.config import ForcingConfig
from nos_utils.forcing.hrrr import HRRRProcessor


class TestHRRRFileDiscovery:
    def test_find_files_in_conus_path(self, mock_config, mock_hrrr_dir):
        proc = HRRRProcessor(mock_config, mock_hrrr_dir, Path("/tmp/out"))
        files = proc.find_input_files()
        assert len(files) > 0

    def test_nowcast_plus_forecast_files(self, mock_config, mock_hrrr_dir):
        """Should find nowcast analysis files + forecast files."""
        proc = HRRRProcessor(mock_config, mock_hrrr_dir, Path("/tmp/out"))
        files = proc.find_input_files()

        # 6h nowcast = ~6 analysis files + up to 48 forecast files
        assert len(files) >= 6

    def test_max_forecast_48h(self, mock_config, mock_hrrr_dir):
        """HRRR max forecast is 48 hours regardless of config."""
        cfg = ForcingConfig(
            lon_min=-80, lon_max=-70, lat_min=25, lat_max=35,
            pdy="20260401", cyc=12, forecast_hours=120,
        )
        proc = HRRRProcessor(cfg, mock_hrrr_dir, Path("/tmp/out"))
        assert proc.MAX_FORECAST_HOURS == 48


class TestHRRRVariables:
    def test_mslma_mapping(self):
        """HRRR uses MSLMA for pressure, not PRMSL."""
        grib_var, level = HRRRProcessor.GRIB2_VARIABLES["prmsl"]
        assert grib_var == "MSLMA"

    def test_default_variables(self, mock_config):
        proc = HRRRProcessor(mock_config, Path("/tmp"), Path("/tmp/out"))
        assert len(proc.variables) == 8


class TestHRRRFailureHandling:
    def test_missing_input_is_nonfatal(self, mock_config, tmp_path):
        """HRRR failure returns success=True with warnings."""
        proc = HRRRProcessor(mock_config, tmp_path / "nonexistent", tmp_path / "out")
        result = proc.process()

        assert result.success  # Non-fatal
        assert len(result.warnings) > 0

    def test_no_files_is_nonfatal(self, mock_config, tmp_path):
        """Empty HRRR directory returns success=True with warnings."""
        empty = tmp_path / "empty_hrrr"
        empty.mkdir()

        proc = HRRRProcessor(mock_config, empty, tmp_path / "out")
        result = proc.process()

        assert result.success
        assert len(result.warnings) > 0

    def test_min_file_size(self):
        """HRRR enforces 100 MB minimum (matching shell FILESIZE=100000000)."""
        assert HRRRProcessor.MIN_FILE_SIZE == 100_000_000
