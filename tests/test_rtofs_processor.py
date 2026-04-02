"""Tests for RTOFSProcessor."""

from pathlib import Path

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
