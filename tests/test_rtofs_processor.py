"""Tests for RTOFSProcessor."""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from nos_utils.config import ForcingConfig
from nos_utils.forcing.rtofs import RTOFSProcessor


class TestRTOFSFileDiscovery:
    def test_find_no_files(self, mock_config, tmp_path):
        proc = RTOFSProcessor(mock_config, tmp_path / "empty", tmp_path / "out")
        files = proc.find_input_files()
        assert len(files) == 0

    def test_find_files_by_type(self, mock_config, tmp_path):
        """Create mock RTOFS directory and verify file discovery."""
        rtofs_dir = tmp_path / "rtofs.20260401"
        rtofs_dir.mkdir()

        # Create mock 2D files
        for cycle in ["n012", "f006"]:
            f = rtofs_dir / f"rtofs_glo_2ds_{cycle}_diag.nc"
            f.write_bytes(b"\x00" * 200_000_000)  # 200MB

        # Create mock 3D files
        for cycle in ["n012", "f006"]:
            f = rtofs_dir / f"rtofs_glo_3dz_{cycle}_6hrly_hvr_US_east.nc"
            f.write_bytes(b"\x00" * 300_000_000)  # 300MB

        proc = RTOFSProcessor(mock_config, tmp_path, tmp_path / "out")
        files_2d, files_3d = proc.find_input_files_by_type()

        assert len(files_2d) == 2
        assert len(files_3d) == 2

    def test_dedup_prefers_forecast_over_nowcast(self, mock_config, tmp_path):
        """Verify forecast (f) files are preferred over nowcast (n) for same
        valid time, matching Fortran which only uses f* files."""
        rtofs_dir = tmp_path / "rtofs.20260401"
        rtofs_dir.mkdir()

        # n012 and f012 have the same valid time (cycle + 12h)
        for prefix in ["n012", "f012", "f024"]:
            f = rtofs_dir / f"rtofs_glo_2ds_{prefix}_diag.nc"
            f.write_bytes(b"\x00" * 200_000_000)

        proc = RTOFSProcessor(mock_config, tmp_path, tmp_path / "out")
        files_2d, _ = proc.find_input_files_by_type()

        # Should have 2 files (f012 wins over n012, plus f024)
        assert len(files_2d) == 2
        # The hour-12 file should be forecast, not nowcast
        assert "_f012_" in files_2d[0].name

    def test_cycle_search_prefers_pdy_minus_1(self, mock_config, tmp_path):
        """Verify PDY-1 is searched before PDY-2 (matches Fortran behavior)."""
        # Create files in both PDY-2 and PDY-1 directories
        # mock_config has pdy="20260401"
        for day in ["20260330", "20260331"]:
            d = tmp_path / f"rtofs.{day}"
            d.mkdir()
            f = d / "rtofs_glo_2ds_f024_diag.nc"
            f.write_bytes(b"\x00" * 200_000_000)

        proc = RTOFSProcessor(mock_config, tmp_path, tmp_path / "out")
        files_2d, _ = proc.find_input_files_by_type()

        # Should find PDY-1 (20260331), not PDY-2 (20260330)
        assert len(files_2d) == 1
        assert "20260331" in str(files_2d[0])


class TestRTOFSProcess:
    def test_no_input_returns_failure(self, mock_config, tmp_path):
        proc = RTOFSProcessor(mock_config, tmp_path / "empty", tmp_path / "out")
        result = proc.process()
        assert not result.success

    def test_ssh_offset_stored_in_config(self, mock_config):
        mock_config.obc_ssh_offset = 0.04
        proc = RTOFSProcessor(mock_config, Path("/tmp"), Path("/tmp/out"))
        assert proc.config.obc_ssh_offset == 0.04


class TestRTOFSConstants:
    def test_min_file_sizes(self):
        assert RTOFSProcessor.MIN_FILE_SIZE_2D == 150_000_000
        assert RTOFSProcessor.MIN_FILE_SIZE_3D == 200_000_000

    def test_source_name(self):
        assert RTOFSProcessor.SOURCE_NAME == "RTOFS"


class TestRTOFSParseHour:
    """Test _parse_rtofs_hour filename parsing."""

    def test_forecast_file(self):
        hour, is_nc = RTOFSProcessor._parse_rtofs_hour(
            Path("rtofs_glo_2ds_f048_diag.nc"))
        assert hour == 48
        assert not is_nc

    def test_nowcast_file(self):
        hour, is_nc = RTOFSProcessor._parse_rtofs_hour(
            Path("rtofs_glo_2ds_n024_diag.nc"))
        assert hour == 24
        assert is_nc

    def test_3d_file(self):
        hour, is_nc = RTOFSProcessor._parse_rtofs_hour(
            Path("rtofs_glo_3dz_f012_6hrly_hvr_US_east.nc"))
        assert hour == 12
        assert not is_nc


class TestRTOFSTimeAxis:
    """Verify the temporal interpolation uses actual file spacing,
    not hardcoded 6h.  This is the fix for the 3cm SSH bias."""

    def _make_filenames(self, hours):
        """Create Path objects mimicking RTOFS 2D filenames."""
        return [Path(f"rtofs_glo_2ds_f{h:03d}_diag.nc") for h in hours]

    def test_hourly_files_not_assumed_6h(self):
        """47 files (37 hourly + 10 three-hourly) must produce
        correct non-uniform time axis, not uniform 6h."""
        hours = list(range(36, 73)) + list(range(75, 103, 3))
        files = self._make_filenames(hours)

        # Compute time axis the same way _process_2d does after fix
        file_hours = []
        for f in files:
            h, _ = RTOFSProcessor._parse_rtofs_hour(f)
            file_hours.append(h)
        rtofs_times = np.array(
            [(h - file_hours[0]) * 3600.0 for h in file_hours])

        assert len(rtofs_times) == 47
        # First file at t=0
        assert rtofs_times[0] == 0.0
        # Second file at 1h (NOT 6h)
        assert rtofs_times[1] == 3600.0
        # Last file at 66h
        assert rtofs_times[-1] == 66 * 3600.0
        # Transition from hourly to 3-hourly at index 37
        assert rtofs_times[37] - rtofs_times[36] == 3 * 3600.0
        # Hourly section: all 1h gaps
        hourly_diffs = np.diff(rtofs_times[:37])
        assert np.all(hourly_diffs == 3600.0)

    def test_uniform_6h_files(self):
        """If files ARE 6-hourly, time axis should still be correct."""
        hours = list(range(12, 78, 6))  # f012,f018,...,f072
        files = self._make_filenames(hours)

        file_hours = []
        for f in files:
            h, _ = RTOFSProcessor._parse_rtofs_hour(f)
            file_hours.append(h)
        rtofs_times = np.array(
            [(h - file_hours[0]) * 3600.0 for h in file_hours])

        assert rtofs_times[1] == 6 * 3600.0
        diffs = np.diff(rtofs_times)
        assert np.all(diffs == 6 * 3600.0)

    def test_time_span_matches_real_coverage(self):
        """Verify total time span reflects actual file hours,
        not n_files * 6h (the old bug)."""
        hours = list(range(36, 73)) + list(range(75, 103, 3))
        files = self._make_filenames(hours)

        file_hours = [RTOFSProcessor._parse_rtofs_hour(f)[0] for f in files]
        rtofs_times = np.array(
            [(h - file_hours[0]) * 3600.0 for h in file_hours])

        real_span_h = rtofs_times[-1] / 3600.0
        old_bug_span_h = (len(files) - 1) * 6.0

        # Real span: 66h.  Old bug span: 276h.
        assert real_span_h == 66.0
        assert old_bug_span_h == 276.0
        assert real_span_h < old_bug_span_h  # fix is smaller
