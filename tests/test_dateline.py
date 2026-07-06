"""Dateline / 0-360 regression tests (STOFS-3D-Pacific).

Every existing STOFS/SECOFS domain is western-hemisphere (-180..180). These
tests exercise a domain that SPANS the dateline in a 0-360 frame, which used to
silently drop the eastern half of every source grid because the modules folded
sources to -180..180 and masked against 0-360 target bounds. They fail against
the pre-fix code and pass once source + target share the domain-centered frame
(nos_utils.geo.unwrap_to_domain).

The pure-math contract for the helper itself lives in test_geo.py.
"""
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from nos_utils.config import ForcingConfig
from nos_utils.forcing.rtofs import RTOFSProcessor

pytest.importorskip("scipy")

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


class TestConfigDateline:
    def test_pacific_flags(self):
        cfg = ForcingConfig.for_stofs_3d_atl(
            pdy="20260401", cyc=12,
            lon_min=92.5, lon_max=290.0, lat_min=-33.5, lat_max=67.0,
        )
        assert cfg.crosses_dateline is True
        assert cfg.domain_center_lon == pytest.approx(191.25)

    def test_atlantic_flags(self):
        cfg = ForcingConfig.for_stofs_3d_atl(pdy="20260401", cyc=12)
        assert cfg.crosses_dateline is False
        assert cfg.domain_center_lon == pytest.approx(-75.495, abs=1e-3)


class TestRTOFSBoundaryDateline:
    """RTOFS SSH/T-S interpolation to boundary nodes across the dateline."""

    def _pacific_cfg(self):
        # Small dateline-crossing box (lon 170..200 -> center 185).
        return ForcingConfig.for_stofs_3d_atl(
            pdy="20260401", cyc=12,
            lon_min=170.0, lon_max=200.0, lat_min=-5.0, lat_max=5.0,
        )

    def test_eastern_boundary_nodes_get_source_data(self):
        cfg = self._pacific_cfg()
        proc = RTOFSProcessor(cfg, Path("/tmp"), Path("/tmp"))

        # Boundary nodes straddling 180 (0-360). Nodes >180 are the ones the
        # old -180..180 fold dropped.
        proc._bnd_lons = np.array([172.0, 178.0, 183.0, 190.0, 197.0])
        proc._bnd_lats = np.array([0.0, 1.0, -1.0, 2.0, -2.0])

        # Synthetic RTOFS grid in 0-360 spanning the box (includes lon > 180).
        rlon = np.linspace(160.0, 210.0, 51)
        rlat = np.linspace(-8.0, 8.0, 17)
        lon2d, lat2d = np.meshgrid(rlon, rlat)
        # Field encodes longitude (kept < 99 so it is not treated as a land
        # fill): a node that fell off the eastern half would recover a wrong,
        # western-neighbour value instead of its own.
        data = ((lon2d - 185.0) / 10.0).astype(np.float32)

        result = proc._interpolate_2d_to_boundary(lon2d, lat2d, data)

        assert result is not None
        assert np.all(np.isfinite(result)), "some boundary node got no source data"
        expected = (proc._bnd_lons - 185.0) / 10.0
        np.testing.assert_allclose(result, expected, atol=0.2)


class TestPrecomputedWeightsCenterLon:
    """The frame used to build a weight .npz must survive round-trip so
    validate_grid re-hashes in the same frame (regression: build_nudge_npz /
    build_3d_npz once omitted center_lon from savez -> every Pacific weight
    file failed validation and silently fell back to Delaunay)."""

    def _grid(self):
        import hashlib
        from nos_utils.interp import precomputed_weights as pw

        rlon = np.linspace(150.0, 210.0, 20)  # 0-360, includes lon > 180
        rlat = np.linspace(-5.0, 5.0, 8)
        lon2d, lat2d = np.meshgrid(rlon, rlat)
        center = 185.0
        lon_u = np.ascontiguousarray(pw.unwrap_to_domain(lon2d, center), dtype=np.float64)
        lat_c = np.ascontiguousarray(lat2d, dtype=np.float64)
        grid_hash = hashlib.md5(lon_u.tobytes() + lat_c.tobytes()).hexdigest()
        return pw, lon2d, lat2d, center, grid_hash

    def test_persisted_center_lon_validates(self):
        pw, lon2d, lat2d, center, grid_hash = self._grid()
        npz = {
            "grid_shape": np.array(lon2d.shape, dtype=np.int32),
            "grid_hash": np.array([grid_hash]),
            "center_lon": np.float64(center),
        }
        pw.validate_grid(npz, lon2d, lat2d)  # must NOT raise

    def test_missing_center_lon_would_mismatch(self):
        # Reproduces the bug: without center_lon, validate_grid re-hashes with
        # the legacy -180..180 fold, which differs for a grid with lon > 180.
        pw, lon2d, lat2d, center, grid_hash = self._grid()
        npz = {
            "grid_shape": np.array(lon2d.shape, dtype=np.int32),
            "grid_hash": np.array([grid_hash]),
        }
        with pytest.raises(ValueError):
            pw.validate_grid(npz, lon2d, lat2d)


# --- Blender (atmospheric DATM) end-to-end across the dateline ---------------

netCDF4 = pytest.importorskip("netCDF4")
from nos_utils.forcing.blender import BlenderProcessor  # noqa: E402
from nos_utils.forcing.forcing_writer import ForcingNcWriter  # noqa: E402


def _epoch(dt):
    return (dt.replace(tzinfo=timezone.utc) - _EPOCH).total_seconds()


def _make_gfs_forcing_lonvar(path, times, lons, lats):
    """GFS forcing whose uwind == longitude, so a dropped eastern-hemisphere
    column shows up as a wrong (mean-filled) value rather than passing silently."""
    ny, nx = len(lats), len(lons)
    lon2d = np.broadcast_to(np.asarray(lons, dtype=np.float32), (ny, nx))
    data = {
        "uwind": [lon2d.astype(np.float32).copy() for _ in times],
        "vwind": [np.full((ny, nx), -1.0, dtype=np.float32) for _ in times],
        "stmp": [np.full((ny, nx), 290.0, dtype=np.float32) for _ in times],
        "spfh": [np.full((ny, nx), 0.01, dtype=np.float32) for _ in times],
        "prmsl": [np.full((ny, nx), 101325.0, dtype=np.float32) for _ in times],
        "prate": [np.zeros((ny, nx), dtype=np.float32) for _ in times],
        "dswrf": [np.full((ny, nx), 100.0, dtype=np.float32) for _ in times],
        "dlwrf": [np.full((ny, nx), 350.0, dtype=np.float32) for _ in times],
    }
    ForcingNcWriter().write_1d(
        data, list(times), np.asarray(lons), np.asarray(lats), path, source_name="GFS",
    )


class TestBlenderDateline:
    def _pacific_cfg(self):
        return ForcingConfig(
            lon_min=170.0, lon_max=200.0, lat_min=-5.0, lat_max=5.0,
            pdy="20260401", cyc=12, nowcast_hours=6, forecast_hours=12,
            nws=4,
            datm_lon_min=170.0, datm_lon_max=200.0,
            datm_lat_min=-5.0, datm_lat_max=5.0,
            datm_dx=1.0,
        )

    def test_eastern_half_of_datm_grid_is_populated(self, tmp_path):
        cfg = self._pacific_cfg()
        cycle_dt = datetime.strptime(cfg.pdy, "%Y%m%d") + timedelta(hours=cfg.cyc)

        in_dir = tmp_path / "in"
        out_dir = tmp_path / "out"
        in_dir.mkdir()

        gfs_times = [
            cycle_dt - timedelta(hours=cfg.nowcast_hours) + timedelta(hours=h)
            for h in range(cfg.nowcast_hours + cfg.forecast_hours + 3 + 1)
        ]
        # GFS lon in 0-360, spanning the box incl. lon > 180 (the dropped half).
        gfs_lons = np.linspace(165.0, 205.0, 41, dtype=np.float32)
        gfs_lats = np.linspace(-8.0, 8.0, 17, dtype=np.float32)
        _make_gfs_forcing_lonvar(in_dir / "gfs_forcing.nc", gfs_times, gfs_lons, gfs_lats)

        proc = BlenderProcessor(cfg, in_dir, out_dir, target_dx=cfg.datm_dx)
        result = proc.process()
        assert result.success, result.errors

        ds = netCDF4.Dataset(str(out_dir / "datm_forcing.nc"))
        try:
            u = np.asarray(ds.variables["UGRD_10maboveground"][:])
            lon2d = np.asarray(ds.variables["longitude"][:])
            if lon2d.ndim == 1:
                lon2d = np.broadcast_to(lon2d, u.shape[1:])
            assert np.all(np.isfinite(u)), "non-finite DATM cells"
            # There ARE target columns east of the dateline.
            assert np.any(lon2d > 180.0)
            # Every cell recovered its lon-encoded value (uwind == lon), so the
            # eastern half is real interpolated data, not a mean-fill.
            np.testing.assert_allclose(u[0], lon2d, atol=1.5)
        finally:
            ds.close()
