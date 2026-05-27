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


class TestElev2DModelDt:
    """The elev2D.th.nc ``time_step`` must equal config.model_dt.

    SCHISM aborts (misc_subs.F90:563 ``MISC: elev2D.th dt wrong``) if the
    written ``time_step`` is smaller than the run's ``dt`` in param.nml.
    SECOFS runs at dt=120, STOFS-3D-ATL at dt=150.
    """

    def _write_2d_files(self, tmp_path, bnd_lons, bnd_lats, hours):
        """Build synthetic RTOFS 2D diag files covering the boundary nodes."""
        pytest.importorskip("netCDF4")
        from netCDF4 import Dataset

        rtofs_dir = tmp_path / "rtofs_2d"
        rtofs_dir.mkdir()

        # Curvilinear-ish grid spanning the boundary with a margin so every
        # boundary node falls inside the interpolation hull.
        lo = np.linspace(bnd_lons.min() - 2.0, bnd_lons.max() + 2.0, 30)
        la = np.linspace(bnd_lats.min() - 2.0, bnd_lats.max() + 2.0, 30)
        lon2d, lat2d = np.meshgrid(lo, la)

        files = []
        for h in hours:
            f = rtofs_dir / f"rtofs_glo_2ds_f{h:03d}_diag.nc"
            with Dataset(str(f), "w") as ds:
                ds.createDimension("time", 1)
                ds.createDimension("Y", lat2d.shape[0])
                ds.createDimension("X", lat2d.shape[1])
                lon_v = ds.createVariable("Longitude", "f4", ("Y", "X"))
                lat_v = ds.createVariable("Latitude", "f4", ("Y", "X"))
                ssh_v = ds.createVariable("ssh", "f4", ("time", "Y", "X"))
                lon_v[:] = lon2d.astype(np.float32)
                lat_v[:] = lat2d.astype(np.float32)
                # Mild spatial signal so interpolation is well-defined.
                ssh_v[0, :, :] = (0.1 * np.sin(np.radians(lon2d))
                                  + 0.05 * h).astype(np.float32)
            files.append(f)
        return files

    def _drive_elev2d(self, cfg, tmp_path):
        from nos_utils.forcing.rtofs import RTOFSProcessor
        from netCDF4 import Dataset

        out = tmp_path / "out"
        out.mkdir()
        proc = RTOFSProcessor(cfg, tmp_path, out)

        # Inject boundary nodes directly (skip hgrid parsing).
        proc._bnd_lons = np.linspace(cfg.lon_min + 1.0, cfg.lon_min + 3.0, 8)
        proc._bnd_lats = np.linspace(cfg.lat_min + 1.0, cfg.lat_min + 3.0, 8)
        # RTOFS cycle anchor used by the time-axis math.
        from datetime import datetime
        proc._rtofs_cycle_date = datetime.strptime(cfg.pdy, "%Y%m%d")

        files = self._write_2d_files(tmp_path, proc._bnd_lons, proc._bnd_lats,
                                     hours=[0, 6, 12, 18, 24])
        result = proc._process_2d(files)
        assert result is not None and result.exists(), "elev2D.th.nc not written"

        with Dataset(str(result)) as ds:
            return float(ds.variables["time_step"][0])

    def test_stofs_elev2d_time_step_is_150(self, tmp_path):
        pytest.importorskip("scipy")
        cfg = ForcingConfig.for_stofs_3d_atl(pdy="20260401", cyc=12)
        time_step = self._drive_elev2d(cfg, tmp_path)
        assert time_step == 150.0

    def test_secofs_elev2d_time_step_is_120(self, tmp_path):
        pytest.importorskip("scipy")
        cfg = ForcingConfig.for_secofs(pdy="20260401", cyc=12)
        time_step = self._drive_elev2d(cfg, tmp_path)
        assert time_step == 120.0

    def test_elev2d_honors_config_override(self, tmp_path):
        """A bare config with an explicit model_dt drives the written value."""
        pytest.importorskip("scipy")
        cfg = ForcingConfig(
            lon_min=-80.0, lon_max=-70.0,
            lat_min=25.0, lat_max=35.0,
            pdy="20260401", cyc=12,
            model_dt=200.0,
        )
        time_step = self._drive_elev2d(cfg, tmp_path)
        assert time_step == 200.0


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
