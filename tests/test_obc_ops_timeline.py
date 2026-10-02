"""ops OBC timeline for STOFS-3D-ATL: 6-hourly phase slices, non-zero uv3D, T/S limits."""

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("netCDF4")
pytest.importorskip("scipy")
from netCDF4 import Dataset  # noqa: E402

from nos_utils.forcing import obc_ops_interp as oi  # noqa: E402
from nos_utils.forcing.rtofs import RTOFSProcessor  # noqa: E402

from .test_adt_blend import _blend, _cfg, _elev, _write_2d  # noqa: E402,F401
from .test_stofs_obc_bilinear import SIGMA, _tsuv_proc  # noqa: E402

DT = 21600.0


class TestResample:
    def test_grid_aligned_records_are_exact_copies(self):
        a = np.array([[1.1], [2.2], [3.3]], np.float32)
        out = oi.resample_records(a, [0, DT, 2 * DT], DT, 3)
        np.testing.assert_array_equal(out, a)

    def test_slice_starts_at_the_phase_start_and_holds_the_ends(self):
        a = np.arange(5, dtype=np.float32)[:, None]
        out = oi.resample_records(a, [k * DT - 2 * DT for k in range(5)], DT, 6)  # phase starts at record 2
        np.testing.assert_array_equal(out[:, 0], [2, 3, 4, 4, 4, 4])
        before = oi.resample_records(a, [k * DT + DT for k in range(5)], DT, 3)
        np.testing.assert_array_equal(before[:, 0], [0, 0, 1])

    def test_off_grid_target_is_linear_and_single_record_repeats(self):
        a = np.array([[0.0], [6.0]])
        np.testing.assert_allclose(oi.resample_records(a, [0, DT], DT / 2, 3)[:, 0], [0, 3, 6])
        np.testing.assert_array_equal(oi.resample_records(a[:1], [0], DT, 3)[:, 0], [0, 0, 0])


def _ops_proc(tmp_path, phase, **cfg_kw):
    cfg = _cfg()
    cfg.obc_ops_timeline = True
    for k, v in cfg_kw.items():
        setattr(cfg, k, v)
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    proc = RTOFSProcessor(cfg, tmp_path, out, phase=phase)
    proc._bnd_lons = np.linspace(cfg.lon_min + 3.0, cfg.lon_min + 5.0, 8)
    proc._bnd_lats = np.linspace(cfg.lat_min + 3.0, cfg.lat_min + 5.0, 8)
    proc._rtofs_cycle_date = datetime.strptime(cfg.pdy, "%Y%m%d")
    files = _write_2d(tmp_path, cfg, proc._bnd_lons, proc._bnd_lats)
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    ssh_1 = proc._stofs_prepare_ssh(files, work)
    proc._ssh1_path = ssh_1
    return cfg, proc, files


def _meta(path):
    with Dataset(str(path)) as ds:
        return np.array(ds["time"][:]), float(ds["time_step"][0])


class TestElev2dTimeline:
    @pytest.mark.parametrize("phase, n", [("nowcast", 6), ("forecast", 20)])
    def test_held_record_count_axis_and_dt(self, tmp_path, phase, n):
        cfg, proc, files = _ops_proc(tmp_path, phase)
        out = proc._process_2d(files)
        t, dt = _meta(out)
        assert dt == DT and len(t) == n
        np.testing.assert_array_equal(t, np.arange(n) * DT)
        e = _elev(out)
        assert e.shape == (n, 8) and (e == e[0]).all()
        np.testing.assert_allclose(e, 0.04, atol=1e-6)

    def test_unheld_series_is_sliced_at_the_phase_start(self, tmp_path):
        cfg, proc, files = _ops_proc(tmp_path, "nowcast", obc_ssh_hold_first_record=False, nowcast_hours=12)
        e = _elev(proc._process_2d(files))  # nowcast starts 00z; files 00z..24z carry 0.05 m per hour
        np.testing.assert_allclose(e[:, 0], [0.0, 0.3, 0.6, 0.9, 1.2, 1.2][:e.shape[0]] + 0.04 * np.ones(e.shape[0]),
                                   atol=1e-5)
        assert e.shape[0] == 4  # 12 h + 3 h buffer -> ceil(15/6) + 1 records

    def test_record_count_follows_phase_length(self, tmp_path):
        cfg, proc, files = _ops_proc(tmp_path, "forecast", forecast_hours=96)
        t, _ = _meta(proc._process_2d(files))
        assert len(t) == 18  # ceil((96 + 3) / 6) + 1


def _t_by_record(path, bump=1000.0):
    """Raise temperature record t by t degrees (raw packing 0.001) so records are distinguishable."""
    with Dataset(str(path), "r+") as ds:
        v = ds["temperature"]
        v.set_auto_maskandscale(False)
        for t in range(v.shape[0]):
            v[t] = v[t] + bump * t


class TestThreeDTimeline:
    def _run(self, tmp_path, phase, hours=(6, 12, 18), **cfg_kw):
        p, path, nt = _tsuv_proc(tmp_path, [-59.5, -59.2], [30.5, 30.9], [40.0, 40.0], SIGMA, nt=len(hours))
        for k, v in {"obc_ops_timeline": True, **cfg_kw}.items():
            setattr(p.config, k, v)
        p.phase = phase
        p.output_path = tmp_path / "o"
        p.output_path.mkdir()
        p._rtofs_cycle_date = datetime(2026, 4, 1)
        p._tsuv1_path = path
        _t_by_record(path)
        files = [Path(f"rtofs_glo_3dz_f{h:03d}_6hrly_hvr_US_east.nc") for h in hours]
        names = {f.name: f for f in p._process_3d(files)}
        return p, names

    @staticmethod
    def _ts(path):
        with Dataset(str(path)) as ds:
            return np.array(ds["time_series"][:]), np.array(ds["time"][:]), float(ds["time_step"][0])

    def test_forecast_slice_starts_at_the_cycle_on_the_6h_grid(self, tmp_path):
        # cycle 12z; files at 06z, 12z, 18z -> forecast record 0 is the 12z file (second record)
        p, names = self._run(tmp_path, "forecast")
        ts, t, dt = self._ts(names["TEM_3D.th.nc"])
        assert dt == DT and len(t) == 20 and t[1] - t[0] == DT and t[0] == 0
        base = ts[0, 0, :, 0]
        np.testing.assert_allclose(ts[1, 0, :, 0], base + 1.0, atol=1e-4)  # 18z file
        np.testing.assert_allclose(ts[5, 0, :, 0], base + 1.0, atol=1e-4)  # last record held
        _, t2, _ = self._ts(names["uv3D.th.nc"])
        np.testing.assert_array_equal(t2, t)
        _, tn, _ = self._ts(names["SAL_3D.th.nc"])
        np.testing.assert_array_equal(tn, t)

    def test_nowcast_has_six_records_from_the_nowcast_start(self, tmp_path):
        p, names = self._run(tmp_path, "nowcast", hours=(0, 6, 12))
        # nowcast starts 03-31 12z (cycle 12z - 24 h): all files are later, record 0 is the first file
        ts, t, _ = self._ts(names["TEM_3D.th.nc"])
        assert len(t) == 6 and ts.shape[1:] == (2, 4, 1)

    def test_uv3d_is_interpolated_not_zero(self, tmp_path):
        p, path, nt = _tsuv_proc(tmp_path, [-59.5], [30.5], [40.0], SIGMA, nt=1)
        with Dataset(str(path), "r+") as ds:
            for name, val in (("water_u", 0.5), ("water_v", -0.25)):
                v = ds[name]
                v.set_auto_maskandscale(False)
                v[:] = val / 0.001
        T, S, U, V = p._ops_ts_profiles(path, nt, uv=True)
        np.testing.assert_allclose(U[0], 0.5, atol=1e-5)
        np.testing.assert_allclose(V[0], -0.25, atol=1e-5)  # negative, unclamped
        assert U[0].dtype == np.float32

    def test_uv_junk_in_middle_is_zero_and_below_bottom_copies(self, tmp_path):
        p, path, nt = _tsuv_proc(tmp_path, [-59.0], [31.0], [40.0], SIGMA, nt=1)
        with Dataset(str(path), "r+") as ds:
            u = ds["water_u"]
            u.set_auto_maskandscale(False)
            u[:] = 1000.0  # 1 m/s
            u[0, 1, 1, 1] = -30000.0  # 10 m level at the node's column
        _, _, U, _ = p._ops_ts_profiles(path, nt, uv=True)
        # z = [-40, -20, -10, 0]: the 10 m level (f90: junk -> 0) lies exactly at z=-10
        np.testing.assert_allclose(U[0][0], [1.0, 1.0, 0.0, 1.0], atol=1e-5)

    def test_uv_zero_outside_the_grid_and_dry_corner_takes_the_parent(self, tmp_path):
        p, path, nt = _tsuv_proc(tmp_path, [-50.0], [31.0], [40.0], SIGMA, nt=1)
        _, _, U, V = p._ops_ts_profiles(path, nt, uv=True)
        assert (U[0] == 0).all() and (V[0] == 0).all()

    def test_salinity_cap_40_and_temperature_floor_0(self, tmp_path):
        p, path, nt = _tsuv_proc(tmp_path, [-59.5], [30.5], [40.0], SIGMA, nt=1)
        with Dataset(str(path), "r+") as ds:
            for name, real in (("salinity", 45.0), ("temperature", -3.0)):
                v = ds[name]
                v.set_auto_maskandscale(False)
                v[:] = (real - 20.0) / 0.001
        T, S = p._ops_ts_profiles(path, nt)
        assert S[0].max() == 40.0 and T[0].max() == 0.0

    def test_flag_off_keeps_3h_records_and_zero_uv(self, tmp_path):
        p, names = self._run(tmp_path, "forecast", obc_ops_timeline=False)
        ts, t, dt = self._ts(names["TEM_3D.th.nc"])
        assert dt == 10800.0
        uv, _, _ = self._ts(names["uv3D.th.nc"])
        assert (uv == 0).all()


def _adt_stage_proc(tmp_path, monkeypatch, ops):
    from nos_utils.config import ForcingConfig
    from nos_utils.forcing import adt

    cfg = ForcingConfig.for_stofs_3d_atl(pdy="20260401", cyc=12)
    cfg.obc_ops_timeline = ops
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    proc = RTOFSProcessor(cfg, tmp_path, out, phase="nowcast")
    cfg.rtofs_3d_region = None
    proc.find_input_files_by_type = lambda: ([tmp_path / "f.nc"], [])
    proc._stofs_prepare_ssh = lambda files, work: tmp_path / "ssh.nc"
    seen = {}

    class Blender:
        regrid = None
        warnings = []

        def __init__(self, *a, ops_numerics=False, **k):
            seen["ops"] = ops_numerics

        def blend_ssh(self, ssh, work):
            if seen["ops"]:
                raise adt.ADTUnavailableError("no ADT; paths tried: /a; /b")
            return None

    monkeypatch.setattr(adt, "ADTBlender", Blender)
    return proc, seen


def test_missing_adt_fails_rtofs_under_ops_timeline(tmp_path, monkeypatch):
    proc, seen = _adt_stage_proc(tmp_path, monkeypatch, True)
    res = proc._process_stofs()
    assert seen["ops"] is True and not res.success and "paths tried" in res.errors[0]


def test_non_ops_timeline_keeps_old_adt_path(tmp_path, monkeypatch):
    proc, seen = _adt_stage_proc(tmp_path, monkeypatch, False)
    proc._process_stofs()
    assert seen["ops"] is False
