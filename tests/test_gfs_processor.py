"""Tests for GFSProcessor."""

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from nos_utils.config import ForcingConfig
from nos_utils.forcing.gfs import GFSProcessor


class TestGFSFileDiscovery:
    def test_find_files_in_standard_path(self, mock_config, mock_gfs_dir):
        """Find GFS files in gfs.YYYYMMDD/HH/atmos/ structure."""
        # Disable file size check for mock files
        proc = GFSProcessor(mock_config, mock_gfs_dir, Path("/tmp/out"))
        proc.MIN_FILE_SIZE = 0  # Mock files are tiny

        files = proc.find_input_files()
        assert len(files) > 0

    def test_compute_search_cycles(self, mock_config):
        """Verify cycle computation covers nowcast window."""
        proc = GFSProcessor(mock_config, Path("/tmp"), Path("/tmp/out"))
        cycles = proc._compute_search_cycles()

        # For 12z with 6h nowcast, should include at least 06z and 12z
        cycle_hours = [c[1] for c in cycles]
        assert 12 in cycle_hours
        assert 6 in cycle_hours

    def test_compute_search_cycles_24h_nowcast(self):
        """STOFS-style 24h nowcast should search more cycles."""
        cfg = ForcingConfig.for_stofs_3d_atl(pdy="20260401", cyc=12)
        proc = GFSProcessor(cfg, Path("/tmp"), Path("/tmp/out"))
        cycles = proc._compute_search_cycles()

        # 24h nowcast from 12z goes back to yesterday 12z
        assert len(cycles) >= 5  # Should span ~24h of 6h cycles

    def test_backup_list(self, mock_config, mock_gfs_dir):
        proc = GFSProcessor(mock_config, mock_gfs_dir, Path("/tmp/out"))
        proc.MIN_FILE_SIZE = 0

        backup = proc._build_backup_list()
        assert len(backup) > 0

    def test_base_date_computation(self, mock_config):
        proc = GFSProcessor(mock_config, Path("/tmp"), Path("/tmp/out"))
        base = proc._compute_base_date()

        # base_date is always day-start (00Z), matching Fortran convention
        # 12z - 6h = 06z → truncated to 00Z on same day
        expected = datetime(2026, 4, 1, 0, 0, 0)
        assert base == expected


class TestGFSVariables:
    def test_default_variables(self, mock_config):
        proc = GFSProcessor(mock_config, Path("/tmp"), Path("/tmp/out"))
        assert len(proc.variables) == 8
        assert "uwind" in proc.variables
        assert "prate" in proc.variables

    def test_custom_variables(self, mock_config):
        proc = GFSProcessor(mock_config, Path("/tmp"), Path("/tmp/out"),
                           variables=["uwind", "vwind"])
        assert proc.variables == ["uwind", "vwind"]

    def test_all_18_variables_mapped(self):
        """Ensure all 18 GRIB2 variables have valid mappings."""
        assert len(GFSProcessor.GRIB2_VARIABLES) == 18
        for name, (grib_var, level) in GFSProcessor.GRIB2_VARIABLES.items():
            assert isinstance(grib_var, str)
            assert isinstance(level, str)


class TestGFSProcess:
    def test_process_no_input_returns_failure(self, mock_config, tmp_path):
        """Empty input path -> failure result."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        proc = GFSProcessor(mock_config, empty_dir, tmp_path / "out")
        proc.MIN_FILE_SIZE = 0
        result = proc.process()

        assert not result.success
        assert "No GFS" in result.errors[0]

    def test_process_with_mock_extractor(self, mock_config, mock_gfs_dir, tmp_path):
        """Full pipeline with mocked GRIB extraction."""
        mock_extractor = MagicMock()
        mock_extractor.get_grid.return_value = (
            np.linspace(-80, -70, 5),
            np.linspace(25, 35, 5),
        )
        mock_extractor.extract.return_value = np.random.rand(5, 5).astype(np.float32)

        out_dir = tmp_path / "out"
        proc = GFSProcessor(mock_config, mock_gfs_dir, out_dir,
                           extractor=mock_extractor)
        proc.MIN_FILE_SIZE = 0

        result = proc.process()

        assert result.success
        assert len(result.output_files) > 0
        assert result.metadata["num_input_files"] > 0

    def test_min_file_size(self):
        # Class-level fallback is 40 MB; resolution-specific set in __init__
        assert GFSProcessor.MIN_FILE_SIZE == 40_000_000
        assert GFSProcessor.MIN_FILE_SIZE_BY_RES["0p25"] == 400_000_000
        assert GFSProcessor.MIN_FILE_SIZE_BY_RES["0p50"] == 40_000_000
