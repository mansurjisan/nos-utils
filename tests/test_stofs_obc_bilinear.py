"""ops gen_3Dth_from_hycom-equivalent OBC interpolation (STOFS-3D-ATL Python path)."""

import time
from datetime import datetime

import numpy as np
import pytest

pytest.importorskip("netCDF4")
pytest.importorskip("scipy")
from netCDF4 import Dataset  # noqa: E402

from nos_utils.config import ForcingConfig  # noqa: E402
from nos_utils.forcing import obc_ops_interp as oi  # noqa: E402
from nos_utils.forcing.adt import ADTBlender  # noqa: E402
from nos_utils.forcing.rtofs import RTOFSProcessor  # noqa: E402
from nos_utils.io.schism_vgrid import SchismVgrid  # noqa: E402

from .test_adt_blend import _cfg, _write_2d, _write_adt  # noqa: E402

XAX = np.array([0.0, 1.0, 2.0])
YAX = np.array([0.0, 2.0, 4.0])


def _proc(tmp_path, lons, lats, depths=None, **cfg_kw):
    cfg = ForcingConfig.for_stofs_3d_atl(pdy="20260401", cyc=12)
    for k, v in cfg_kw.items():
        setattr(cfg, k, v)
    p = RTOFSProcessor(cfg, tmp_path, tmp_path)
    p._bnd_lons = np.asarray(lons, float)
    p._bnd_lats = np.asarray(lats, float)
    p._bnd_depths = np.asarray(depths if depths is not None else np.full(len(lons), 100.0))
    return p


def _write_ssh1(path, lons, lats, fields):
    LO, LA = np.meshgrid(lons, lats)
    with Dataset(str(path), "w") as ds:
        ds.createDimension("time", len(fields))
        ds.createDimension("ylat", len(lats))
        ds.createDimension("xlon", len(lons))
        ds.createVariable("xlon", "f4", ("ylat", "xlon"))[:] = LO
        ds.createVariable("ylat", "f4", ("ylat", "xlon"))[:] = LA
        v = ds.createVariable("ssh", "f4", ("time", "ylat", "xlon"), fill_value=-30000.0)
        for t, f in enumerate(fields):
            v[t] = f
    return path


class TestWeights:
    def test_bilinear_interior_hand_values(self):
        # cell [0,1]x[0,2]; point (0.25, 0.5) -> xr=0.25, yr=0.25
        ix, iy, w, inside = oi.parent_weights(XAX, YAX, [0.25], [0.5])
        assert inside[0] and ix[0] == 0 and iy[0] == 0
        np.testing.assert_allclose(w[0], [0.5625, 0.1875, 0.0625, 0.1875])
        f = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])  # f[iy, ix]
        cj, ci = oi.corner_cells(ix, iy)
        got = (w * f[cj, ci].T).sum(1)
        np.testing.assert_allclose(got, 0.5625 * 1 + 0.1875 * 2 + 0.0625 * 5 + 0.1875 * 4)

    def test_node_on_cell_edge_goes_to_lower_cell_same_value(self):
        # x=1.0 is the edge shared by cells 0 and 1; ops first match is cell 0 with xr=1
        ix, iy, w, inside = oi.parent_weights(XAX, YAX, [1.0], [1.0])
        assert ix[0] == 0
        np.testing.assert_allclose(w[0], [0.0, 0.5, 0.5, 0.0])

    def test_node_on_grid_corner_gets_exact_value(self):
        for x, y, ixe, iye in [(0.0, 0.0, 0, 0), (2.0, 4.0, 1, 1)]:
            ix, iy, w, inside = oi.parent_weights(XAX, YAX, [x], [y])
            assert inside[0] and ix[0] == ixe and iy[0] == iye
            assert np.isclose(w[0].sum(), 1.0) and np.isclose(w[0].max(), 1.0)

    def test_outside_node_flagged_zero_weight(self):
        ix, iy, w, inside = oi.parent_weights(XAX, YAX, [2.5, -0.1], [1.0, 1.0])
        assert not inside.any() and (w == 0).all()

    def test_mode1_triangle_split_hand_values(self):
        # xr=0.25 < yr=0.75 -> triangle (1,3,4): (1-yr, 0, xr, yr-xr)
        ix, iy, w, _ = oi.parent_weights(XAX, YAX, [0.25], [1.5], mode=1)
        np.testing.assert_allclose(w[0], [0.25, 0.0, 0.25, 0.5])
        # xr=0.75 > yr=0.25 -> triangle (1,2,3): (1-xr, xr-yr, yr, 0)
        _, _, w2, _ = oi.parent_weights(XAX, YAX, [0.75], [0.5], mode=1)
        np.testing.assert_allclose(w2[0], [0.25, 0.5, 0.25, 0.0])

    def test_rect_axes_from_separable_2d_and_rejects_curvilinear(self):
        LO, LA = np.meshgrid(XAX, YAX)
        ax = oi.rect_axes(LO, LA)
        np.testing.assert_array_equal(ax[0], XAX)
        np.testing.assert_array_equal(ax[1], YAX)
        assert oi.rect_axes(LO + 0.1 * LA, LA) is None


class TestSshFill:
    def test_dry_cell_takes_first_wet_in_x_major_scan(self):
        wet = np.ones((3, 3), bool)
        wet[0, 0] = False
        # (j=0,i=0): block m=1 scanned i-major -> (i=0,j=1) comes before (i=1,j=0)
        assert oi.nearest_wet(wet, 0, 0) == (1, 0)

    def test_dry_ssh_corner_filled_from_parent(self, tmp_path):
        f = np.array([[-30000.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
        path = _write_ssh1(tmp_path / "s.nc", XAX, YAX, [f])
        p = _proc(tmp_path, [0.5], [1.0])
        out = p._ops_ssh_boundary(path, 1)
        # corners LL(0,0) dry -> parent (j=1,i=0)=3, LR=1, UR=4, UL=3 ; w=0.25 each
        np.testing.assert_allclose(out[0, 0], (3.0 + 1.0 + 4.0 + 3.0) / 4, atol=1e-6)

    def test_later_record_junk_uses_forward_neighbour_else_zero(self):
        s = np.full((4, 4), 1.0)
        s[0, 0] = -30000.0
        s[1, 1] = 2.0
        wet = np.ones((4, 4), bool)
        out, n = oi.fix_ssh(s, wet)
        assert n == 1
        # ii=2 (x+2) overrides ii=1: (j+1, i+2) = 1.0, not (j+1,i+1)=2.0
        assert out[0, 0] == 1.0
        s2 = np.full((2, 2), -30000.0)
        s2[1, 1] = 5.0
        out2, _ = oi.fix_ssh(s2, np.array([[True, True], [True, False]]))
        assert out2[0, 0] == 5.0 and out2[0, 1] == 0.0

    def test_outside_node_gets_zero_ssh(self, tmp_path):
        f = np.full((3, 3), 1.0)
        path = _write_ssh1(tmp_path / "s.nc", XAX, YAX, [f])
        p = _proc(tmp_path, [0.5, 9.0], [1.0, 1.0])
        out = p._ops_ssh_boundary(path, 1)
        np.testing.assert_allclose(out[0], [1.0, 0.0])

    def test_step_count_mismatch_falls_back(self, tmp_path):
        path = _write_ssh1(tmp_path / "s.nc", XAX, YAX, [np.zeros((3, 3))] * 2)
        assert _proc(tmp_path, [0.5], [1.0])._ops_ssh_boundary(path, 3) is None


class TestColumns:
    def test_junk_in_middle_takes_bottom_value_and_below_extends(self):
        # bottom-first columns; level 0 junk (below bottom), level 2 junk in the middle
        T = np.array([[-10.0, 5.0, -10.0, 7.0]])
        S = np.array([[-10.0, 30.0, 31.0, 32.0]])
        U = np.zeros((1, 4))
        U[0, 0] = -10.0
        V = np.zeros((1, 4))
        V[0, 0] = -10.0
        Tf, Sf, klev0, n_mid = oi.fill_columns(T, S, U, V)
        np.testing.assert_allclose(Tf[0], [5.0, 5.0, 5.0, 7.0])
        np.testing.assert_allclose(Sf[0], [30.0, 30.0, 31.0, 32.0])
        assert klev0[0] == 1 and n_mid == 1

    def test_all_junk_column_is_fatal(self):
        z = np.full((1, 3), -10.0)
        with pytest.raises(ValueError):
            oi.fill_columns(z, z, z, z)

    def test_vertical_above_top_and_below_bottom(self):
        zm = np.array([-50.0, -20.0, -10.0, 0.0])
        z = np.array([[-80.0, -50.0, -35.0, -10.0, 0.0, 1.0]])
        lev, vrat = oi.vertical_index(z, zm, np.array([0]), np.array([False]))
        vals = np.array([5.0, 6.0, 7.0, 8.0])  # at zm
        got = vals[lev] * (1 - vrat) + vals[lev + 1] * vrat
        # below bottom -> bottom value; -50 -> bottom; -35 halfway 5..6; -10 -> 7... 
        np.testing.assert_allclose(got[0], [5.0, 5.0, 5.5, 7.0, 8.0, 8.0])

    def test_shallow_lower_left_kbp_clamps_to_level_one(self):
        zm = np.array([-50.0, -20.0, -10.0, 0.0])
        # LL column valid from level 2 (zm=-10): z at/below -10 -> level 0 value
        lev, vrat = oi.vertical_index(np.array([[-15.0, -5.0]]), zm, np.array([2]), np.array([False]))
        assert lev[0, 0] == 0 and vrat[0, 0] == 0.0
        assert lev[0, 1] == 2 and np.isclose(vrat[0, 1], 0.5)

    def test_dry_lower_left_uses_surface_everywhere(self):
        zm = np.array([-50.0, -20.0, -10.0, 0.0])
        lev, vrat = oi.vertical_index(np.array([[-45.0, -5.0]]), zm, np.array([0]), np.array([True]))
        vals = np.array([5.0, 6.0, 7.0, 8.0])
        np.testing.assert_allclose((vals[lev] * (1 - vrat) + vals[lev + 1] * vrat)[0], [8.0, 8.0])


def _tsuv_proc(tmp_path, lons, lats, node_depths, sigma, dry=False, nt=2, mid_junk=False):
    lo = np.array([-60.0, -59.0, -58.0])
    la = np.array([30.0, 31.0, 32.0])
    depth = np.array([0.0, 10.0, 20.0, 50.0])
    LO, LA = np.meshgrid(lo, la)
    base = 10.0 + 2.0 * (LO + 60.0) + 1.0 * (LA - 30.0)
    T = base[None] + 0.1 * depth[:, None, None]
    S = 30.0 + 0.5 * (LO + 60.0) + 0.01 * depth[:, None, None] + 0 * base[None]
    U = np.zeros_like(T)
    V = np.zeros_like(T)
    if dry:
        for a in (T, S, U, V):
            a[:, 0, 0] = -30000.0
    if mid_junk:
        T[1, 1, 1] = -30000.0
    p = _proc(tmp_path, lons, lats, node_depths)
    sig = np.asarray(sigma, float)
    n = len(lons)
    p._vgrid = SchismVgrid(
        nvrt=sig.shape[0], kz=0, h_s=100.0, z_levels=np.array([]),
        sigma_levels=np.linspace(-1, 0, sig.shape[0]),
        node_sigma=np.tile(sig[:, None], (1, n)), node_kbp=np.ones(n, int),
    )
    work = tmp_path / "w"
    work.mkdir()
    path = p._write_tsuv_nc(work, [T] * nt, [S] * nt, [U] * nt, [V] * nt, LO, LA, depth)
    return p, path, nt


SIGMA = [-1.0, -0.5, -0.25, 0.0]


class TestTS:
    def test_horizontal_and_vertical_exact_for_linear_fields(self, tmp_path):
        p, path, nt = _tsuv_proc(tmp_path, [-59.5], [30.5], [40.0], SIGMA)
        temp, salt = p._ops_ts_profiles(path, nt)
        assert len(temp) == nt and temp[0].shape == (1, 4)
        z = np.array([-40.0, -20.0, -10.0, 0.0])
        want_t = 10.0 + 2.0 * 0.5 + 0.5 + 0.1 * (-z)
        want_s = 30.0 + 0.5 * 0.5 + 0.01 * (-z)
        np.testing.assert_allclose(temp[0][0], want_t, atol=1e-4)
        np.testing.assert_allclose(salt[1][0], want_s, atol=1e-4)

    def test_below_bottom_gets_bottom_value(self, tmp_path):
        p, path, nt = _tsuv_proc(tmp_path, [-59.5], [30.5], [200.0], SIGMA)
        temp, _ = p._ops_ts_profiles(path, nt)
        # sigma -1 -> z=-200, below the 50 m deepest RTOFS level
        np.testing.assert_allclose(temp[0][0, 0], 10.0 + 1.0 + 0.5 + 0.1 * 50.0, atol=1e-4)

    def test_outside_node_uses_tem_sal_outside(self, tmp_path):
        p, path, nt = _tsuv_proc(tmp_path, [-59.5, -50.0], [30.5, 31.0], [40.0, 40.0], SIGMA)
        temp, salt = p._ops_ts_profiles(path, nt)
        np.testing.assert_array_equal(temp[0][1], 20.0)
        np.testing.assert_array_equal(salt[0][1], 33.0)

    def test_dry_lower_left_corner_uses_parent_surface_values(self, tmp_path):
        p, path, nt = _tsuv_proc(tmp_path, [-59.5], [30.5], [40.0], SIGMA, dry=True)
        temp, _ = p._ops_ts_profiles(path, nt)
        # corners LL(0,0) dry -> parent (j=1,i=0); surface T per column: 10+2*i+j
        want = (11.0 + 12.0 + 13.0 + 11.0) / 4
        np.testing.assert_allclose(temp[0][0], want, atol=1e-4)

    def test_junk_in_middle_filled_with_bottom_value(self, tmp_path):
        p, path, nt = _tsuv_proc(tmp_path, [-59.0], [31.0], [40.0], SIGMA, mid_junk=True)
        temp, _ = p._ops_ts_profiles(path, nt)
        # node is the (j=1,i=1) column: 13 + 0.1*depth at 0/10/20/50 m, 10 m junk -> bottom 18
        # bottom-first column [18, 15, 18, 13] at z=[-50,-20,-10,0]; SCHISM z=[-40,-20,-10,0]
        np.testing.assert_allclose(temp[0][0], [17.0, 15.0, 18.0, 13.0], atol=1e-4)

    def test_mode1_uses_triangle_split(self, tmp_path):
        p, path, nt = _tsuv_proc(tmp_path, [-59.75], [30.75], [40.0], SIGMA)
        p.config.obc_interp_mode = 1
        t1, _ = p._ops_ts_profiles(path, nt)
        p.config.obc_interp_mode = 0
        t0, _ = p._ops_ts_profiles(path, nt)
        # T is linear in i and j so bilinear and triangle split agree; both exact
        np.testing.assert_allclose(t1[0], t0[0], atol=1e-4)

    def test_clamps_temperature_floor(self, tmp_path):
        p, path, nt = _tsuv_proc(tmp_path, [-59.5], [30.5], [40.0], SIGMA)
        with Dataset(str(path), "r+") as ds:
            ds["temperature"].set_auto_maskandscale(False)
            ds["temperature"][:] = (-5.0 - 20.0) / 0.001  # real -5 C everywhere
        temp, _ = p._ops_ts_profiles(path, nt)
        assert temp[0].max() == 0.0


class TestWiring:
    def _setup(self, tmp_path):
        cfg = _cfg()
        out = tmp_path / "out"
        out.mkdir()
        proc = RTOFSProcessor(cfg, tmp_path, out)
        proc._bnd_lons = np.linspace(cfg.lon_min + 3.0, cfg.lon_min + 5.0, 8)
        proc._bnd_lats = np.linspace(cfg.lat_min + 3.0, cfg.lat_min + 5.0, 8)
        proc._rtofs_cycle_date = datetime.strptime(cfg.pdy, "%Y%m%d")
        files = _write_2d(tmp_path, cfg, proc._bnd_lons, proc._bnd_lats)
        work = tmp_path / "work"
        work.mkdir()
        return cfg, proc, files, proc._stofs_prepare_ssh(files, work), work

    @staticmethod
    def _elev(path):
        with Dataset(str(path)) as ds:
            return np.array(ds.variables["time_series"][:])[:, :, 0, 0]

    def test_blended_ssh_1_adt_is_the_source_when_present(self, tmp_path, monkeypatch):
        monkeypatch.delenv("COMINadt", raising=False)
        monkeypatch.delenv("DCOMROOT", raising=False)
        cfg, proc, files, ssh_1, work = self._setup(tmp_path)
        _write_adt(tmp_path / "adt_20260401.nc", lambda lo, la: np.full_like(lo, 0.90))
        blended = ADTBlender(cfg, tmp_path).blend_ssh(ssh_1, work)
        assert blended is not None
        proc._ssh1_path = ssh_1
        raw = self._elev(proc._process_2d(files))
        got = self._elev(proc._process_2d(files, ssh_source=blended))
        assert proc._ssh_blended_used is True
        np.testing.assert_allclose(got - raw, 0.90 - 0.45, atol=2e-5)

    def test_raw_ssh_1_is_used_without_blend_and_not_reported_blended(self, tmp_path):
        cfg, proc, files, ssh_1, work = self._setup(tmp_path)
        proc._ssh1_path = ssh_1
        seen = []
        orig = RTOFSProcessor._ops_ssh_boundary
        proc._ops_ssh_boundary = lambda p, n: seen.append(p) or orig(proc, p, n)
        proc._process_2d(files)
        assert seen == [ssh_1] and proc._ssh_blended_used is False

    def test_secofs_never_touches_the_ops_path(self, tmp_path, monkeypatch):
        cfg = ForcingConfig.for_secofs(pdy="20260401", cyc=12)
        out = tmp_path / "out"
        out.mkdir()
        proc = RTOFSProcessor(cfg, tmp_path, out)
        assert proc.is_stofs_mode is False
        proc._bnd_lons = np.linspace(-80.0, -79.0, 5)
        proc._bnd_lats = np.linspace(30.0, 31.0, 5)
        proc._rtofs_cycle_date = datetime.strptime(cfg.pdy, "%Y%m%d")
        files = _write_2d(tmp_path, cfg, proc._bnd_lons, proc._bnd_lats)
        ref = self._elev(proc._process_2d(files))

        def boom(*a, **k):
            raise AssertionError("ops path used in SECOFS mode")

        monkeypatch.setattr(RTOFSProcessor, "_ops_ssh_boundary", boom)
        monkeypatch.setattr(RTOFSProcessor, "_ops_ts_profiles", boom)
        proc._ssh1_path = tmp_path / "SSH_1.nc"
        proc._tsuv1_path = tmp_path / "TSUV_1.nc"
        np.testing.assert_array_equal(self._elev(proc._process_2d(files)), ref)

    @pytest.mark.parametrize("opt_in", [False, True])
    def test_fortran_exe_only_called_when_opted_in(self, tmp_path, monkeypatch, opt_in):
        cfg, proc, files, ssh_1, work = self._setup(tmp_path)
        cfg.rtofs_3d_region = None
        cfg.adt_enabled = False
        cfg.obc_use_fortran_gen3dth = opt_in
        called = []
        monkeypatch.setattr(RTOFSProcessor, "find_input_files_by_type", lambda self: (files, []))
        monkeypatch.setattr(RTOFSProcessor, "_call_fortran_gen_3dth",
                            lambda *a, **k: called.append(1) or False)
        monkeypatch.setattr(RTOFSProcessor, "_load_grid", lambda self: True)
        res = proc._process_stofs()
        assert res.success
        assert bool(called) is opt_in
        assert res.metadata["fortran_used"] is False


def test_defaults_keep_fortran_off_and_ops_constants():
    cfg = ForcingConfig.for_stofs_3d_atl(pdy="20260401", cyc=12)
    assert cfg.obc_use_fortran_gen3dth is False
    assert (cfg.obc_tem_outside, cfg.obc_sal_outside, cfg.obc_interp_mode) == (20.0, 33.0, 0)


def test_160k_nodes_interpolate_in_seconds(tmp_path):
    ny, nx = 120, 160
    lons = np.linspace(-98.0, -52.0, nx)
    lats = np.linspace(7.5, 52.0, ny)
    rng = np.random.default_rng(0)
    path = _write_ssh1(tmp_path / "s.nc", lons, lats, [rng.random((ny, nx)) for _ in range(4)])
    n = 160_000
    p = _proc(tmp_path, rng.uniform(-97, -53, n), rng.uniform(8, 51, n))
    t0 = time.perf_counter()
    out = p._ops_ssh_boundary(path, 4)
    assert out.shape == (4, n)
    assert time.perf_counter() - t0 < 5.0
