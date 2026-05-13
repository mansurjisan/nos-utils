"""Tests for BlenderProcessor."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import pytest
from nos_utils.config import ForcingConfig
from nos_utils.forcing.blender import BlenderProcessor, HRRR_LOV, HRRR_LAD
from nos_utils.forcing.forcing_writer import SFLUX_TO_DATM, ForcingNcWriter

netCDF4 = pytest.importorskip("netCDF4")
pytest.importorskip("scipy")

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _epoch(dt: datetime) -> float:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (dt - _EPOCH).total_seconds()


def _make_gfs_forcing(path: Path, times, lons, lats) -> None:
    """Write a minimal gfs_forcing.nc with all 8 DATM vars on a regular grid."""
    ny, nx = len(lats), len(lons)
    data = {
        "uwind": [np.full((ny, nx), 5.0, dtype=np.float32) for _ in times],
        "vwind": [np.full((ny, nx), -1.0, dtype=np.float32) for _ in times],
        "stmp":  [np.full((ny, nx), 290.0, dtype=np.float32) for _ in times],
        "spfh":  [np.full((ny, nx), 0.01, dtype=np.float32) for _ in times],
        "prmsl": [np.full((ny, nx), 101325.0, dtype=np.float32) for _ in times],
        "prate": [np.zeros((ny, nx), dtype=np.float32) for _ in times],
        "dswrf": [np.full((ny, nx), 100.0, dtype=np.float32) for _ in times],
        "dlwrf": [np.full((ny, nx), 350.0, dtype=np.float32) for _ in times],
    }
    ForcingNcWriter().write_1d(
        data, list(times), np.asarray(lons), np.asarray(lats),
        path, source_name="GFS",
    )


def _rotate_winds_lcc(u, v, lon):
    """Test helper: Lambert Conformal wind rotation (grid -> earth-relative).

    Mirrors the inline rotation step in BlenderProcessor.process(). Kept as
    a standalone function in this test module so the unit tests don't have
    to drive the full process() pipeline.
    """
    D2R = np.pi / 180.0
    rotcon = np.sin(HRRR_LAD * D2R)
    angle = rotcon * (np.asarray(lon) - HRRR_LOV) * D2R
    cos_r = np.cos(angle)
    sin_r = np.sin(angle)
    u_e = cos_r * u + sin_r * v
    v_e = -sin_r * u + cos_r * v
    return u_e, v_e


class TestRotateWinds:
    def test_zero_rotation_at_lov(self):
        """At LoV longitude, rotation angle should be 0."""
        u = np.array([[10.0]])
        v = np.array([[0.0]])
        lon = np.array([[-97.5]])  # LoV
        u_e, v_e = _rotate_winds_lcc(u, v, lon)
        assert u_e[0, 0] == pytest.approx(10.0, abs=0.001)
        assert v_e[0, 0] == pytest.approx(0.0, abs=0.001)

    def test_rotation_away_from_lov(self):
        """Away from LoV, winds should be rotated."""
        u = np.array([[10.0]])
        v = np.array([[0.0]])
        lon = np.array([[-80.0]])  # 17.5° east of LoV
        u_e, v_e = _rotate_winds_lcc(u, v, lon)
        # Should have some v component after rotation
        assert abs(v_e[0, 0]) > 0.1

    def test_preserves_wind_speed(self):
        """Rotation should preserve wind magnitude."""
        u = np.array([[3.0]])
        v = np.array([[4.0]])
        lon = np.array([[-75.0]])
        u_e, v_e = _rotate_winds_lcc(u, v, lon)
        speed_orig = np.sqrt(u**2 + v**2)
        speed_rot = np.sqrt(u_e**2 + v_e**2)
        assert speed_rot[0, 0] == pytest.approx(speed_orig[0, 0], rel=1e-5)


class TestBlenderProcessor:
    def test_no_gfs_fails(self, mock_config, tmp_path):
        # Empty input dir → gfs_forcing.nc missing → blender returns failure
        empty = tmp_path / "empty"
        empty.mkdir()
        proc = BlenderProcessor(mock_config, empty, tmp_path / "out")
        result = proc.process()
        assert not result.success
        assert "GFS forcing file not found" in result.errors[0]

    def test_variable_mapping_complete(self):
        """All 8 sflux variable names should have DATM mappings."""
        expected = ["uwind", "vwind", "stmp", "spfh", "prmsl", "prate", "dswrf", "dlwrf"]
        for var in expected:
            assert var in SFLUX_TO_DATM, f"Missing DATM mapping for {var}"


class TestNetCDFUtils:
    def test_validate_monotonic_ok(self):
        from nos_utils.io.netcdf_utils import validate_monotonic
        assert validate_monotonic([1.0, 2.0, 3.0])

    def test_validate_monotonic_fail(self):
        from nos_utils.io.netcdf_utils import validate_monotonic
        with pytest.raises(ValueError, match="Non-monotonic"):
            validate_monotonic([1.0, 3.0, 2.0])

    def test_replace_fill_values(self):
        from nos_utils.io.netcdf_utils import replace_fill_values
        data = np.array([1.0, 99999.0, -30000.0, 5.0])
        result = replace_fill_values(data, threshold=10000.0, fill_value=-9999.0)
        assert result[0] == 1.0
        assert result[1] == -9999.0
        assert result[2] == -9999.0
        assert result[3] == 5.0


class TestRouteBTimeAnchor:
    """Route B: DATM output time axis is anchored at model_t0 = cycle - nowcast_hours.

    The output must cover [model_t0, model_t0 + (nowcast_hours +
    forecast_hours)*3600] regardless of which prep phase generated
    the input GFS/HRRR forcing files.
    """

    def _build_config(self, tmp_path: Path) -> ForcingConfig:
        # Small DATM domain so the Delaunay/RegularGridInterp run fast.
        return ForcingConfig(
            lon_min=-80.0, lon_max=-70.0,
            lat_min=25.0, lat_max=35.0,
            pdy="20260401", cyc=12,
            nowcast_hours=6, forecast_hours=48,
            nws=4,
            datm_lon_min=-80.0, datm_lon_max=-70.0,
            datm_lat_min=25.0, datm_lat_max=35.0,
            datm_dx=1.0,  # coarse so it's quick
        )

    def _cycle_dt(self, cfg: ForcingConfig) -> datetime:
        return datetime.strptime(cfg.pdy, "%Y%m%d") + timedelta(hours=cfg.cyc)

    def test_forecast_phase_input_spans_full_window(self, tmp_path):
        """GFS forecast-phase output (cycle-6h..cycle+51h) produces 55-step DATM."""
        cfg = self._build_config(tmp_path)
        cycle_dt = self._cycle_dt(cfg)

        in_dir = tmp_path / "in"
        out_dir = tmp_path / "out"
        in_dir.mkdir()

        # Mimic GFSProcessor forecast-phase output: cycle-6h to cycle+51h.
        gfs_times = [
            cycle_dt - timedelta(hours=cfg.nowcast_hours) + timedelta(hours=h)
            for h in range(cfg.nowcast_hours + cfg.forecast_hours + 3 + 1)
        ]
        gfs_lons = np.linspace(-82.0, -68.0, 15, dtype=np.float32)
        gfs_lats = np.linspace(23.0, 37.0, 15, dtype=np.float32)
        _make_gfs_forcing(in_dir / "gfs_forcing.nc", gfs_times, gfs_lons, gfs_lats)

        proc = BlenderProcessor(cfg, in_dir, out_dir, target_dx=cfg.datm_dx)
        result = proc.process()
        assert result.success, result.errors

        ds = netCDF4.Dataset(str(out_dir / "datm_forcing.nc"))
        try:
            t = ds.variables["time"][:]
            # 55 hourly records for SECOFS (nowcast=6, forecast=48).
            n_expected = cfg.nowcast_hours + cfg.forecast_hours + 1
            assert len(t) == n_expected, (
                f"expected {n_expected} steps, got {len(t)}"
            )
            # First step is model_t0 = cycle - nowcast_hours.
            model_t0 = cycle_dt - timedelta(hours=cfg.nowcast_hours)
            assert float(t[0]) == pytest.approx(_epoch(model_t0), abs=1.0)
            # Last step is model_t0 + sim_duration_hours.
            sim_end = model_t0 + timedelta(
                hours=cfg.nowcast_hours + cfg.forecast_hours,
            )
            assert float(t[-1]) == pytest.approx(_epoch(sim_end), abs=1.0)
            # Cadence is hourly.
            assert np.allclose(np.diff(t), 3600.0)
        finally:
            ds.close()

    def test_nowcast_phase_short_input_extends_to_full_window(self, tmp_path):
        """GFS nowcast-phase input (cycle-9h..cycle+3h) still produces 55 steps.

        Route B requires both prep phases to write a DATM file covering
        the full coupled-run window. The nowcast prep's GFS input ends
        ~3h past cycle, so the blender must extend the time axis
        forward and hold the last GFS slab constant for the gap.
        """
        cfg = self._build_config(tmp_path)
        cycle_dt = self._cycle_dt(cfg)

        in_dir = tmp_path / "in"
        out_dir = tmp_path / "out"
        in_dir.mkdir()

        # Mimic GFSProcessor nowcast-phase output: cycle-9h to cycle+3h.
        gfs_times = [
            cycle_dt - timedelta(hours=cfg.nowcast_hours + 3) + timedelta(hours=h)
            for h in range(cfg.nowcast_hours + 3 + 3 + 1)
        ]
        gfs_lons = np.linspace(-82.0, -68.0, 15, dtype=np.float32)
        gfs_lats = np.linspace(23.0, 37.0, 15, dtype=np.float32)
        _make_gfs_forcing(in_dir / "gfs_forcing.nc", gfs_times, gfs_lons, gfs_lats)

        proc = BlenderProcessor(cfg, in_dir, out_dir, target_dx=cfg.datm_dx)
        result = proc.process()
        assert result.success, result.errors

        ds = netCDF4.Dataset(str(out_dir / "datm_forcing.nc"))
        try:
            t = ds.variables["time"][:]
            # 55 hourly records regardless of input span.
            n_expected = cfg.nowcast_hours + cfg.forecast_hours + 1
            assert len(t) == n_expected
            model_t0 = cycle_dt - timedelta(hours=cfg.nowcast_hours)
            assert float(t[0]) == pytest.approx(_epoch(model_t0), abs=1.0)
            # The held-constant tail: the last record exists and equals
            # model_t0 + sim_duration, not the GFS input end.
            sim_end = model_t0 + timedelta(
                hours=cfg.nowcast_hours + cfg.forecast_hours,
            )
            assert float(t[-1]) == pytest.approx(_epoch(sim_end), abs=1.0)
            # Variable values at every step are finite (held-constant tail
            # samples reuse the edge GFS slab, no NaNs / fill values).
            u = ds.variables["UGRD_10maboveground"][:]
            assert np.all(np.isfinite(u))
            assert u.shape[0] == n_expected
        finally:
            ds.close()

    def test_metadata_reflects_full_window(self, tmp_path):
        """ForcingResult.metadata['ntime'] matches the anchored 55-step axis."""
        cfg = self._build_config(tmp_path)
        cycle_dt = self._cycle_dt(cfg)

        in_dir = tmp_path / "in"
        out_dir = tmp_path / "out"
        in_dir.mkdir()

        # Use a non-trivial input span so the metadata check is meaningful.
        gfs_times = [
            cycle_dt - timedelta(hours=cfg.nowcast_hours + 3) + timedelta(hours=h)
            for h in range(20)
        ]
        gfs_lons = np.linspace(-82.0, -68.0, 15, dtype=np.float32)
        gfs_lats = np.linspace(23.0, 37.0, 15, dtype=np.float32)
        _make_gfs_forcing(in_dir / "gfs_forcing.nc", gfs_times, gfs_lons, gfs_lats)

        proc = BlenderProcessor(cfg, in_dir, out_dir, target_dx=cfg.datm_dx)
        result = proc.process()
        assert result.success
        assert result.metadata["ntime"] == (
            cfg.nowcast_hours + cfg.forecast_hours + 1
        )
