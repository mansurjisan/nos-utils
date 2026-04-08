"""Tests for STOFS-3D-ATL RTOFS and ADT extensions."""

import pytest
from pathlib import Path

import numpy as np

from nos_utils.config import ForcingConfig
from nos_utils.forcing.rtofs import RTOFSProcessor


class TestSTOFSModeDetection:
    def test_stofs_mode(self):
        cfg = ForcingConfig.for_stofs_3d_atl(pdy="20260401", cyc=12)
        proc = RTOFSProcessor(cfg, Path("/tmp"), Path("/tmp"))
        assert proc.is_stofs_mode is True

    def test_secofs_mode(self):
        cfg = ForcingConfig.for_secofs(pdy="20260401", cyc=12)
        proc = RTOFSProcessor(cfg, Path("/tmp"), Path("/tmp"))
        assert proc.is_stofs_mode is False


class TestROISubsetting:
    def test_subset_2d(self):
        """Test _stofs_subset_roi with a mock dataset."""
        cfg = ForcingConfig.for_stofs_3d_atl(pdy="20260401", cyc=12)
        proc = RTOFSProcessor(cfg, Path("/tmp"), Path("/tmp"))

        # Create a mock dataset-like object
        class MockDS:
            def __init__(self):
                ny, nx = 100, 200
                self.variables = {
                    "ssh": MockVar(np.random.rand(2, ny, nx).astype(np.float32),
                                   dims=("time", "Y", "X")),
                    "Longitude": MockVar(np.random.rand(ny, nx).astype(np.float32),
                                         dims=("Y", "X")),
                    "Latitude": MockVar(np.random.rand(ny, nx).astype(np.float32),
                                        dims=("Y", "X")),
                }

        class MockVar:
            def __init__(self, data, dims):
                self._data = data
                self.dimensions = dims
            def __getitem__(self, key):
                return self._data[key]

        ds = MockDS()
        roi = {"x1": 10, "x2": 20, "y1": 5, "y2": 15}
        result = proc._stofs_subset_roi(ds, roi, ["ssh", "Longitude", "Latitude"])

        assert "ssh" in result
        assert result["ssh"].shape == (2, 11, 11)  # y2-y1+1=11, x2-x1+1=11
        assert result["Longitude"].shape == (11, 11)

    def test_subset_3d(self):
        """Test _stofs_subset_roi with 4D data."""
        cfg = ForcingConfig.for_stofs_3d_atl(pdy="20260401", cyc=12)
        proc = RTOFSProcessor(cfg, Path("/tmp"), Path("/tmp"))

        class MockDS:
            def __init__(self):
                nt, nz, ny, nx = 2, 40, 100, 200
                self.variables = {
                    "temperature": MockVar(np.random.rand(nt, nz, ny, nx).astype(np.float32),
                                           dims=("time", "depth", "Y", "X")),
                }

        class MockVar:
            def __init__(self, data, dims):
                self._data = data
                self.dimensions = dims
            def __getitem__(self, key):
                return self._data[key]

        ds = MockDS()
        roi = {"x1": 10, "x2": 20, "y1": 5, "y2": 15}
        result = proc._stofs_subset_roi(ds, roi, ["temperature"])

        assert result["temperature"].shape == (2, 40, 11, 11)


class TestFortranWrapper:
    def test_no_exe_returns_false(self, tmp_path):
        """Without Fortran exe, _call_fortran_gen_3dth should return False."""
        cfg = ForcingConfig.for_stofs_3d_atl(pdy="20260401", cyc=12)
        proc = RTOFSProcessor(cfg, tmp_path, tmp_path)
        result = proc._call_fortran_gen_3dth(tmp_path, None, None)
        assert result is False


class TestSSHOffset:
    def test_offset_value(self):
        cfg = ForcingConfig.for_stofs_3d_atl(pdy="20260401", cyc=12)
        assert cfg.obc_ssh_offset == 0.04

    def test_secofs_offset(self):
        cfg = ForcingConfig.for_secofs(pdy="20260401", cyc=12)
        assert cfg.obc_ssh_offset == 1.25


class TestADTBlender:
    def test_import(self):
        from nos_utils.forcing.adt import ADTBlender
        cfg = ForcingConfig.for_stofs_3d_atl(pdy="20260401", cyc=12)
        blender = ADTBlender(cfg, Path("/tmp"))
        assert blender is not None

    def test_no_adt_returns_none(self, tmp_path):
        """Without ADT data files, blend_ssh should return None."""
        from nos_utils.forcing.adt import ADTBlender
        cfg = ForcingConfig.for_stofs_3d_atl(pdy="20260401", cyc=12)
        blender = ADTBlender(cfg, tmp_path)
        result = blender.blend_ssh(tmp_path / "SSH_1.nc", tmp_path)
        assert result is None
