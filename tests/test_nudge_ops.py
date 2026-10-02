"""ops gen_nudge port for STOFS-3D-ATL: parents/weights, fill and vertical rules, node set, layout, phases."""

import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("netCDF4")
pytest.importorskip("scipy")
from netCDF4 import Dataset  # noqa: E402

from nos_utils.config import ForcingConfig  # noqa: E402
from nos_utils.forcing import nudge_ops as no  # noqa: E402
from nos_utils.forcing.base import ForcingResult  # noqa: E402
from nos_utils.forcing.nudging import NudgingProcessor  # noqa: E402
from nos_utils.forcing.obc_ops_interp import parent_weights_2d  # noqa: E402
from nos_utils.forcing.rtofs import RTOFSProcessor  # noqa: E402

F = np.float32


def _curvi(ny, nx, x0=-75.0, y0=30.0, dx=0.2, dy=0.2, shear=0.03):
    jj, ii = np.mgrid[0:ny, 0:nx]
    lon = x0 + dx * ii + shear * jj + 0.004 * np.sin(jj)
    lat = y0 + dy * jj + 0.5 * shear * ii + 0.004 * np.cos(ii)
    return lon.astype(F), lat.astype(F)


# ---- literal f90 loops (1 -> 0-based), the references the vectorised port is compared against ----

def _signa(x1, x2, x3, y1, y2, y3):
    return ((x1 - x3) * (y2 - y3) - (x2 - x3) * (y1 - y3)) / F(2)


def ref_parent(lon, lat, px, py):
    """f90:544-594 for one node: (ix, iy, arco[4]) or None."""
    ny, nx = lon.shape
    small1 = F(1e-2)
    for ix in range(nx - 1):
        for iy in range(ny - 1):
            x1, x2, x3, x4 = lon[iy, ix], lon[iy, ix + 1], lon[iy + 1, ix + 1], lon[iy + 1, ix]
            y1, y2, y3, y4 = lat[iy, ix], lat[iy, ix + 1], lat[iy + 1, ix + 1], lat[iy + 1, ix]
            a1 = abs(_signa(px, x1, x2, py, y1, y2))
            a2 = abs(_signa(px, x2, x3, py, y2, y3))
            a3 = abs(_signa(px, x3, x4, py, y3, y4))
            a4 = abs(_signa(px, x4, x1, py, y4, y1))
            b1 = abs(_signa(x1, x2, x3, y1, y2, y3))
            b2 = abs(_signa(x1, x3, x4, y1, y3, y4))
            if abs(a1 + a2 + a3 + a4 - b1 - b2) / (b1 + b2) < small1:
                ap = abs(_signa(px, x1, x3, py, y1, y3))
                arco = np.zeros(4, F)
                bb = abs(_signa(x1, x2, x3, y1, y2, y3))
                if abs(a1 + a2 + ap - bb) / bb < small1 * F(5):
                    arco[0] = max(F(0), min(F(1), a2 / bb))
                    arco[1] = max(F(0), min(F(1), ap / bb))
                    arco[2] = max(F(0), min(F(1), F(1) - arco[0] - arco[1]))
                    return ix, iy, arco
                bb = abs(_signa(x1, x3, x4, y1, y3, y4))
                assert abs(a3 + a4 + ap - bb) / bb < small1 * F(5)
                arco[0] = max(F(0), min(F(1), a3 / bb))
                arco[2] = max(F(0), min(F(1), a4 / bb))
                arco[3] = max(F(0), min(F(1), F(1) - arco[0] - arco[2]))
                return ix, iy, arco
    return None


def ref_first(s_raw, nz, ny, nx):
    """f90:385-485: kbp (0-based, -1 dry) and the parent of every dry cell, on [i, j] arrays."""
    rjunk = F(-3e4) * F(1e-3) + F(20)
    salt = np.empty((nx, ny, nz), F)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                salt[i, j, k] = F(s_raw[nz - 1 - k, j, i]) * F(1e-3) + F(20)
    kbp = np.zeros((nx, ny), int)
    for i in range(nx):
        for j in range(ny):
            if salt[i, j, nz - 1] < rjunk + F(0.1):
                kbp[i, j] = -1
            else:
                kbp[i, j] = next(k for k in range(nz) if salt[i, j, k] > rjunk)
    par = {}
    for i in range(nx):
        for j in range(ny):
            if kbp[i, j] != -1:
                continue
            m = 0
            while (i, j) not in par:
                m += 1
                for ii in range(max(-m, -i), min(m, nx - 1 - i) + 1):
                    for jj in range(max(-m, -j), min(m, ny - 1 - j) + 1):
                        if kbp[i + ii, j + jj] >= 0 and (i, j) not in par:
                            par[(i, j)] = (i + ii, j + jj)
    return kbp, par, rjunk


def ref_record(kbp, par, rjunk, t_raw, s_raw, depth, nodes, z):
    """f90:601-736 for one record. nodes = [(ix, iy, arco)], z (n, nvrt); returns T, S (n, nvrt)."""
    nz, ny, nx = s_raw.shape
    zm = np.array([-F(depth[nz - 1 - k]) for k in range(nz)], F)
    out = []
    salt = np.empty((nx, ny, nz), F)
    temp = np.empty((nx, ny, nz), F)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                salt[i, j, k] = F(s_raw[nz - 1 - k, j, i]) * F(1e-3) + F(20)
                temp[i, j, k] = F(t_raw[nz - 1 - k, j, i]) * F(1e-3) + F(20)
    for i in range(nx):
        for j in range(ny):
            if kbp[i, j] >= 0:
                k0 = next(k for k in range(nz) if salt[i, j, k] > rjunk)
                salt[i, j, :k0] = salt[i, j, k0]
                temp[i, j, :k0] = temp[i, j, k0]
    for (i, j), (i1, j1) in par.items():
        salt[i, j, :] = salt[i1, j1, :]
        temp[i, j, :] = temp[i1, j1, :]
    for (ix, iy, arco), zz in zip(nodes, z):
        tt, ss = [], []
        for k in range(len(zz)):
            kb = kbp[ix, iy]
            if kb == -1:
                lev, vrat = nz - 2, F(1)
            elif zz[k] <= zm[kb]:
                lev, vrat = kb, F(0)
            elif zz[k] >= zm[nz - 1]:
                lev, vrat = nz - 2, F(1)
            else:
                lev = next(kk for kk in range(nz - 1) if zm[kk] <= zz[k] <= zm[kk + 1])
                vrat = (zm[lev] - zz[k]) / (zm[lev] - zm[lev + 1])
            lev2 = min(lev + 1, nz - 1)
            row = []
            for arr in (temp, salt):
                w2 = [arr[ix, iy, lev] * (F(1) - vrat) + arr[ix, iy, lev2] * vrat,
                      arr[ix + 1, iy, lev] * (F(1) - vrat) + arr[ix + 1, iy, lev2] * vrat,
                      arr[ix + 1, iy + 1, lev] * (F(1) - vrat) + arr[ix + 1, iy + 1, lev2] * vrat,
                      arr[ix, iy + 1, lev] * (F(1) - vrat) + arr[ix, iy + 1, lev2] * vrat]
                row.append(((w2[0] * arco[0] + w2[1] * arco[1]) + w2[2] * arco[2]) + w2[3] * arco[3])
            tt.append(max(F(0), row[0]))
            ss.append(row[1])
        out.append((tt, ss))
    return (np.array([o[0] for o in out], F), np.array([o[1] for o in out], F))


def _fields(nz, ny, nx, seed=1):
    rng = np.random.default_rng(seed)
    base = 10.0 + rng.normal(0, 0.5, (ny, nx))
    t = np.stack([base - 0.4 * k for k in range(nz)])
    s = np.stack([35.0 - 0.05 * k + 0.01 * rng.normal(size=(ny, nx)) for k in range(nz)])
    return no.pack_ts(t), no.pack_ts(s)


class TestParents:
    def test_curvilinear_parent_and_weights_equal_the_f90_loop(self):
        lon, lat = _curvi(14, 18)
        rng = np.random.default_rng(3)
        px = rng.uniform(lon.min() - 0.2, lon.max() + 0.2, 70).astype(F)
        py = rng.uniform(lat.min() - 0.2, lat.max() + 0.2, 70).astype(F)
        ix, iy, w, found = parent_weights_2d(lon, lat, px, py, single=True)
        n_found = 0
        for k in range(px.size):
            ref = ref_parent(lon, lat, px[k], py[k])
            assert found[k] == (ref is not None)
            if ref is not None:
                n_found += 1
                assert (ix[k], iy[k]) == ref[:2]
                np.testing.assert_array_equal(w[k], ref[2])
        assert 20 < n_found < 70

    def test_float32_weights(self):
        lon, lat = _curvi(6, 6)
        w = parent_weights_2d(lon, lat, [lon[2, 2] + 0.01], [lat[2, 2] + 0.01], single=True)[2]
        assert w.dtype == np.float32 and abs(float(w.sum()) - 1.0) < 1e-6


def _scene(dry=True, nz=5, ny=9, nx=10, nodes=24, seed=5):
    lon, lat = _curvi(ny, nx)
    depth = np.array([0.0, 5.0, 15.0, 40.0, 100.0][:nz], F)
    t_raw, s_raw = _fields(nz, ny, nx)
    t_raw, s_raw = t_raw.copy(), s_raw.copy()
    if dry:
        for (j, i) in [(2, 3), (2, 4), (3, 3), (6, 7), (0, 0)]:
            t_raw[:, j, i] = F(-30000)
            s_raw[:, j, i] = F(-30000)
    # shallow wet columns: junk at the deepest levels (file order: last index is the deepest)
    for (j, i, nb) in [(4, 5, 2), (5, 5, 3), (1, 8, 1)]:
        t_raw[nz - nb:, j, i] = F(-30000)
        s_raw[nz - nb:, j, i] = F(-30000)
    rng = np.random.default_rng(seed)
    px = rng.uniform(lon[1, 1], lon[ny - 2, nx - 2], nodes).astype(F)
    py = rng.uniform(lat[1, 1], lat[ny - 2, nx - 2], nodes).astype(F)
    sig = np.array([-1.0, -0.7, -0.3, -0.1, 0.0], F)
    dp = rng.uniform(3.0, 140.0, nodes).astype(F)
    z = no.node_z(dp, np.tile(sig[:, None], (1, nodes)), np.ones(nodes, int))
    return lon, lat, depth, t_raw, s_raw, px, py, z


class TestFillAndVertical:
    def _run(self, dry):
        lon, lat, depth, t_raw, s_raw, px, py, z = _scene(dry=dry)
        nz, ny, nx = s_raw.shape
        state, found = no.build_state(lon, lat, depth, px, py, np.ones(px.size, bool), z, s_raw)
        T, S = no.interpolate_record(state, t_raw, s_raw)
        kbp, par, rj = ref_first(s_raw, nz, ny, nx)
        nodes = []
        keep = []
        for k in range(px.size):
            r = ref_parent(lon, lat, px[k], py[k])
            if r is not None:
                nodes.append(r)
                keep.append(k)
        assert list(state.sel) == keep
        rT, rS = ref_record(kbp, par, rj, t_raw, s_raw, depth, nodes, z[keep])
        return state, T, S, rT, rS, (lon, lat, depth, t_raw, s_raw, px, py, z, kbp, par, rj, nodes, keep)

    @pytest.mark.parametrize("dry", [False, True])
    def test_record_equals_the_f90_loop(self, dry):
        state, T, S, rT, rS, _ = self._run(dry)
        assert T.dtype == np.float32 and len(T) > 10
        np.testing.assert_array_equal(T, rT)
        np.testing.assert_array_equal(S, rS)

    def test_later_record_uses_first_record_kbp_and_parents(self):
        state, T, S, rT, rS, ctx = self._run(True)
        lon, lat, depth, t_raw, s_raw, px, py, z, kbp, par, rj, nodes, keep = ctx
        t2, s2 = t_raw.copy(), s_raw.copy()
        t2[-1, 4, 5] = F(-30000)
        s2[-1, 4, 5] = F(-30000)
        t2[-1, 2, 6] = F(-30000)
        s2[-1, 2, 6] = F(-30000)
        T2, S2 = no.interpolate_record(state, t2, s2)
        rT2, rS2 = ref_record(kbp, par, rj, t2, s2, depth, nodes, z[keep])
        np.testing.assert_array_equal(T2, rT2)
        np.testing.assert_array_equal(S2, rS2)

    def test_bottom_extension_and_below_bottom_level_rule(self):
        # one shallow wet column with 2 junk bottom levels: values below klev0 repeat klev0
        t = np.zeros((4, 3, 3), F)
        s = np.zeros((4, 3, 3), F)
        for k in range(4):
            t[k] = no.pack_ts(np.full((3, 3), 20.0 - k))
            s[k] = no.pack_ts(np.full((3, 3), 35.0))
        t[2:], s[2:] = F(-30000), F(-30000)
        lon, lat = _curvi(3, 3)
        depth = np.array([0.0, 10.0, 20.0, 30.0], F)
        z = np.array([[-90.0, -25.0, -9.0, -1.0]], F)  # below the deepest RTOFS level, then inside
        state, _ = no.build_state(lon, lat, depth, [lon[0, 0] + 0.05], [lat[0, 0] + 0.05],
                                  np.ones(1, bool), z, s)
        assert (state.kbp == 2).all()
        T, S = no.interpolate_record(state, t, s)
        # klev0 = 10 m (19 C); z <= zm(kbp) takes that level, then 0.1 and 0.9 of the way to the 20 C surface
        np.testing.assert_allclose(T[0], [19.0, 19.0, 19.1, 19.9], atol=1e-4)

    def test_dry_cell_uses_surface_value_and_parent_columns(self):
        lon, lat, depth, t_raw, s_raw, px, py, z = _scene(dry=True)
        state, _ = no.build_state(lon, lat, depth, px, py, np.ones(px.size, bool), z, s_raw)
        dry_read = [c for c in np.unique(state.cells) if state.kbp[c] < 0]
        assert dry_read, "the scene must have a node reading a dry cell"
        for c in dry_read:
            assert state.parent[c] != c and state.kbp[state.parent[c]] >= 0

    def test_t_floored_s_not_capped(self):
        lon, lat = _curvi(4, 4)
        t = no.pack_ts(np.full((2, 4, 4), -3.0))
        s = no.pack_ts(np.full((2, 4, 4), 44.0))
        state, _ = no.build_state(lon, lat, np.array([0.0, 10.0], F), [lon[1, 1] + 0.02],
                                  [lat[1, 1] + 0.02], np.ones(1, bool), np.array([[-5.0, 0.0]], F), s)
        T, S = no.interpolate_record(state, t, s)
        assert T.max() == 0.0 and S.min() == pytest.approx(44.0, abs=1e-4)

    def test_sanity_limits_raise(self):
        lon, lat = _curvi(4, 4)
        s = no.pack_ts(np.full((2, 4, 4), 46.0))
        t = no.pack_ts(np.full((2, 4, 4), 10.0))
        state, _ = no.build_state(lon, lat, np.array([0.0, 10.0], F), [lon[1, 1] + 0.02],
                                  [lat[1, 1] + 0.02], np.ones(1, bool), np.array([[-5.0, 0.0]], F),
                                  no.pack_ts(np.full((2, 4, 4), 35.0)))
        with pytest.raises(ValueError, match="sanity"):
            no.interpolate_record(state, t, s)


class TestNodeSet:
    def test_zone_is_value_plus_one_element_ring(self):
        values = np.array([0.5, 0, 0, 0.3, 0, 0, 0, 0, 0, 1e-15])
        el = np.array([[0, 1, 2, -1], [3, 4, 5, 6], [7, 8, 9, -1], [2, 7, 8, -1]])
        z = no.nudge_zone(values, el)
        assert z.tolist() == [True] * 7 + [False, False, False]  # element 4 has no included node

    def test_unit_conversion_is_float32_ops(self):
        assert no.RJUNK == F(-3e4) * F(1e-3) + F(20) and no.unpack(F(-30000)) == no.RJUNK
        assert no.unpack(no.pack_ts(np.array([12.5]))).dtype == np.float32

    def test_read_grid_streams_nodes_and_elements(self, tmp_path):
        p = tmp_path / "g.ll"
        p.write_text("x\n 2 5\n1 -70.5 30.0 10.0\n2 -70.0 30.5 20.0\n3 -69.5 31.0 30.0\n"
                     "4 -69.0 31.5 40.0\n5 -68.5 32.0 50.0\n1 3 1 2 3\n2 4 2 3 4 5\n0 = open\n")
        lon, lat, dp, el = no.read_grid(p)
        assert lon.dtype == np.float32 and el.tolist() == [[0, 1, 2, -1], [1, 2, 3, 4]]
        assert dp.tolist() == [10, 20, 30, 40, 50]


class TestPhasePlan:
    @pytest.mark.parametrize("offset, dur, n, need", [(0.0, 27 * 3600.0, 6, 2), (86400.0, 99 * 3600.0, 18, 4)])
    def test_counts_and_needed_records(self, offset, dur, n, need):
        n_out, ratio, off, aligned, k, _ = no.phase_plan(offset, dur)
        assert (n_out, ratio, aligned, k) == (n, 10, True, need) and off == offset / 21600

    def test_forecast_run_hours_do_not_need_the_buffer_record_r3(self):
        n_out, ratio, off, aligned, need, last_j = no.phase_plan(86400.0, 99 * 3600.0, run_s=96 * 3600.0)
        assert (n_out, need, last_j) == (18, 3, 16)
        assert no.phase_plan(0.0, 27 * 3600.0, run_s=24 * 3600.0)[4:] == (2, 4)

    def test_unaligned_offset_is_flagged(self):
        assert not no.phase_plan(3 * 3600.0, 27 * 3600.0)[3]

    @pytest.mark.parametrize("offset, dur", [(0.0, 27 * 3600.0), (86400.0, 99 * 3600.0)])
    def test_every_ops_breakpoint_is_a_phase_knot_and_schism_reads_the_ops_field(self, offset, dur):
        n_out, ratio, off, aligned, need, _ = no.phase_plan(offset, dur)
        rng = np.random.default_rng(0)
        rec = [rng.normal(size=(3, 2)).astype(F) for _ in range(need)]
        for j in range(n_out):
            k, num = divmod(off + j, ratio)
            tau = offset + j * no.NU_DT
            if tau % no.OPS_NU_STEP == 0:
                assert num == 0
                np.testing.assert_array_equal(no.effective(rec, k, num, ratio), rec[int(tau // no.OPS_NU_STEP)])
        phase = np.stack([no.effective(rec, *divmod(off + j, ratio), ratio) for j in range(n_out)])

        def schism(t):  # misc_subs.F90:447-449, schism_step.F90:1092-1164 with step_nu_tr = 21600
            n1 = int(t / no.NU_DT)
            rat = ((n1 + 1) * no.NU_DT - t) / no.NU_DT
            return rat * phase[n1].astype(np.float64) + (1 - rat) * phase[n1 + 1].astype(np.float64)

        def ops(t_abs):  # same records at step_nu_tr = 216000 on the nowcast clock
            n1 = int(t_abs / no.OPS_NU_STEP)
            rat = ((n1 + 1) * no.OPS_NU_STEP - t_abs) / no.OPS_NU_STEP
            return rat * rec[n1].astype(np.float64) + (1 - rat) * rec[n1 + 1].astype(np.float64)

        for t in np.arange(0.0, (n_out - 1) * no.NU_DT, 1350.0):
            np.testing.assert_allclose(schism(t), ops(offset + t), atol=2e-6, rtol=0)


# ---- end to end on a small curvilinear ROI --------------------------------------------------

PDY = "20260927"
NZ, NY, NX = 4, 10, 12
DEPTH = np.array([0.0, 10.0, 20.0, 50.0], np.float32)


def _tfield(lon, lat, depth, k):
    return 10.0 + k + 2.0 * (lon + 75.0) + (lat - 30.0) + 0.05 * depth


def _sfield(lon, lat, depth, k):
    return 30.0 + 0.2 * k + 0.5 * (lon + 75.0) + 0.01 * depth


def _stage(tmp_path):
    lon, lat = _curvi(NY, NX)
    d = tmp_path / "rtofs" / f"rtofs.{PDY}"
    d.mkdir(parents=True)
    for k, tag in enumerate(["n012", "n018", "n024", "f006"]):
        with Dataset(str(d / f"rtofs_glo_3dz_{tag}_6hrly_hvr_US_east.nc"), "w") as ds:
            ds.createDimension("MT", 1)
            ds.createDimension("Depth", NZ)
            ds.createDimension("Y", NY)
            ds.createDimension("X", NX)
            ds.createVariable("Longitude", "f4", ("Y", "X"))[:] = lon
            ds.createVariable("Latitude", "f4", ("Y", "X"))[:] = lat
            ds.createVariable("Depth", "f4", ("Depth",))[:] = DEPTH
            tv = ds.createVariable("temperature", "f4", ("MT", "Depth", "Y", "X"))
            sv = ds.createVariable("salinity", "f4", ("MT", "Depth", "Y", "X"))
            tv[0] = _tfield(lon[None], lat[None], DEPTH[:, None, None], k)
            sv[0] = _sfield(lon[None], lat[None], DEPTH[:, None, None], k)
    # 10 nodes: 0-6 in the zone (0, 3 carry values; 1, 2 and 4-6 via the ring), 7-9 outside it
    nodes = [(-74.2, 30.7), (-74.0, 30.9), (-73.8, 30.6), (-73.6, 31.0), (-73.4, 31.2),
             (-73.2, 30.8), (-72.9, 31.1), (-72.6, 30.9), (-72.4, 31.3), (-61.0, 31.0)]
    with open(tmp_path / "hgrid.ll", "w") as f:
        f.write("t\n 3 10\n")
        for i, (x, y) in enumerate(nodes):
            f.write(f"{i + 1} {x} {y} 40.0\n")
        f.write("1 3 1 2 3\n2 4 4 5 6 7\n3 3 8 9 10\n0 = open\n")
    with open(tmp_path / "nudge.gr3", "w") as f:
        f.write("t\n 3 10\n")
        for i, (x, y) in enumerate(nodes):
            f.write(f"{i + 1} {x} {y} {0.5 if i in (0, 3) else 0.0}\n")
    with open(tmp_path / "vgrid.in", "w") as f:
        f.write("1 !ivcor\n4 !nvrt\n" + " ".join(["1"] * 10) + "\n")
        for k, sg in enumerate([-1.0, -0.5, -0.25, 0.0]):
            f.write(f"{k + 1} " + " ".join([str(sg)] * 10) + "\n")
    return nodes


def _proc(tmp_path, phase, ops_tl=True):
    cfg = ForcingConfig.for_stofs_3d_atl(pdy=PDY, cyc=12)
    cfg.obc_ops_timeline = ops_tl
    cfg.nudge_roi_3d = {"x1": 0, "x2": NX - 1, "y1": 0, "y2": NY - 1}
    cfg.grid_file = tmp_path / "hgrid.ll"
    cfg.nudging_enabled = True
    out = tmp_path / f"out_{phase}"
    out.mkdir(parents=True, exist_ok=True)
    return NudgingProcessor(cfg, tmp_path, out, nudge_weight_file=tmp_path / "nudge.gr3",
                            rtofs_input_path=tmp_path / "rtofs", phase=phase), out


@pytest.fixture(autouse=True)
def _no_size_floor(monkeypatch):
    monkeypatch.setattr(RTOFSProcessor, "MIN_FILE_SIZE_3D", 0)
    monkeypatch.setattr(NudgingProcessor, "_cached_vgrid", None)


class TestEndToEnd:
    @pytest.mark.parametrize("phase, n, off", [("nowcast", 6, 0), ("forecast", 18, 4)])
    def test_layout_nodes_and_ops_effective_values(self, tmp_path, phase, n, off):
        nodes = _stage(tmp_path)
        proc, out = _proc(tmp_path, phase)
        res = proc.process()
        assert res.success and res.metadata["nudge_interp"] == "ops", res.warnings
        assert res.metadata["n_timesteps"] == n and res.metadata["dt_seconds"] == 21600.0
        with Dataset(str(out / "TEM_nu.nc")) as ds:
            ids = np.array(ds["map_to_global_node"][:])
            assert ids.tolist() == list(range(1, 8))  # zone + ring, ascending, 8-10 outside
            assert ds["tracer_concentration"].dtype == np.float32
            assert ds["tracer_concentration"].dimensions == ("time", "node", "nLevels", "one")
            assert ds["time"].dtype == np.float64 and ds["map_to_global_node"].dtype == np.int32
            np.testing.assert_allclose(np.array(ds["time"][:]), np.arange(n) * 0.25)
            tr = np.array(ds["tracer_concentration"][:])
        assert tr.shape == (n, 7, 4, 1)
        zc = np.array([-40.0, -20.0, -10.0, 0.0])
        for j in range(n):
            keff = (off + min(j, 16 if phase == "forecast" else 4)) / 10.0
            for i in range(7):
                want = _tfield(nodes[i][0], nodes[i][1], -zc, keff)
                np.testing.assert_allclose(tr[j, i, :, 0], want, atol=2e-4)
        with Dataset(str(out / "SAL_nu.nc")) as ds:
            sal = np.array(ds["tracer_concentration"][:])
        np.testing.assert_allclose(sal[2, 0, :, 0], _sfield(nodes[0][0], nodes[0][1], -zc, (off + 2) / 10.0),
                                   atol=2e-4)

    def test_ops_path_is_not_used_without_the_ops_timeline(self, tmp_path, monkeypatch):
        _stage(tmp_path)
        proc, out = _proc(tmp_path, "nowcast", ops_tl=False)
        called = []
        monkeypatch.setattr(NudgingProcessor, "_process_ops", lambda self: called.append(1))
        monkeypatch.setattr(NudgingProcessor, "_process_stofs_legacy",
                            lambda self: ForcingResult(True, "NUDGING", metadata={"nudge_interp": "delaunay"}))
        res = proc._process_stofs()
        assert not called and res.metadata["nudge_interp"] == "delaunay" and not res.warnings

    def test_fallback_is_labelled_and_warned(self, tmp_path, monkeypatch):
        _stage(tmp_path)
        proc, out = _proc(tmp_path, "nowcast")

        def boom(self):
            raise ValueError("no vgrid")

        monkeypatch.setattr(NudgingProcessor, "_process_ops", boom)
        monkeypatch.setattr(NudgingProcessor, "_process_stofs_legacy",
                            lambda self: ForcingResult(True, "NUDGING", metadata={"nudge_interp": "delaunay"}))
        res = proc._process_stofs()
        assert res.metadata["nudge_interp"] == "delaunay"
        assert any("fell back to delaunay" in w and "no vgrid" in w for w in res.warnings)

    def test_missing_ops_record_falls_back_with_a_reason(self, tmp_path, monkeypatch):
        _stage(tmp_path)
        for tag in ("n024", "f006"):
            (tmp_path / "rtofs" / f"rtofs.{PDY}" / f"rtofs_glo_3dz_{tag}_6hrly_hvr_US_east.nc").unlink()
        proc, out = _proc(tmp_path, "forecast")
        monkeypatch.setattr(NudgingProcessor, "_process_stofs_legacy",
                            lambda self: ForcingResult(True, "NUDGING", metadata={"nudge_interp": "delaunay"}))
        res = proc._process_stofs()
        assert res.metadata["nudge_interp"] == "delaunay"
        assert any("RTOFS 3D files, found" in w for w in res.warnings)

    def test_forecast_without_f006_needs_only_three_records_and_holds_the_tail(self, tmp_path):
        nodes = _stage(tmp_path)
        (tmp_path / "rtofs" / f"rtofs.{PDY}" / "rtofs_glo_3dz_f006_6hrly_hvr_US_east.nc").unlink()
        proc, out = _proc(tmp_path, "forecast")
        res = proc._process_stofs()
        assert res.metadata["nudge_interp"] == "ops" and res.metadata["ops_records"] == 3, res.warnings
        with Dataset(str(out / "TEM_nu.nc")) as ds:
            tr = np.array(ds["tracer_concentration"][:])
        assert tr.shape[0] == 18
        np.testing.assert_array_equal(tr[17], tr[16])
        zc = np.array([-40.0, -20.0, -10.0, 0.0])
        np.testing.assert_allclose(tr[16, 0, :, 0], _tfield(nodes[0][0], nodes[0][1], -zc, 2.0), atol=2e-4)

    def test_valid_range_does_not_mask_real_values(self, tmp_path):
        nodes = _stage(tmp_path)
        for f in (tmp_path / "rtofs" / f"rtofs.{PDY}").glob("*.nc"):
            with Dataset(str(f), "a") as ds:
                ds["temperature"].valid_range = np.array([0.0, 1.0], np.float32)
        proc, out = _proc(tmp_path, "nowcast")
        res = proc._process_stofs()
        assert res.metadata["nudge_interp"] == "ops", res.warnings
        with Dataset(str(out / "TEM_nu.nc")) as ds:
            tr = np.array(ds["tracer_concentration"][:])
        zc = np.array([-40.0, -20.0, -10.0, 0.0])
        np.testing.assert_allclose(tr[0, 0, :, 0], _tfield(nodes[0][0], nodes[0][1], -zc, 0.0), atol=2e-4)

    def test_tem_nudge_gr3_preferred_under_ops_timeline(self, tmp_path, monkeypatch):
        _stage(tmp_path)
        fix = tmp_path / "fix"
        fix.mkdir()
        for n in ("x.nudge.gr3", "x.TEM_nudge.gr3"):
            (fix / n).write_text("t\n")
        monkeypatch.setenv("FIXofs", str(fix))
        monkeypatch.delenv("FIXstofs3d", raising=False)
        proc, _ = _proc(tmp_path, "nowcast")
        proc.nudge_weight_file = None
        assert proc._nudge_file().name == "x.TEM_nudge.gr3"
        proc.config.obc_ops_timeline = False
        assert proc._nudge_file().name == "x.nudge.gr3"

    def test_chosen_gr3_is_in_metadata(self, tmp_path):
        _stage(tmp_path)
        proc, _ = _proc(tmp_path, "nowcast")
        assert proc._process_stofs().metadata["nudge_gr3"] == "nudge.gr3"


class TestOpsFallbackAndFortranOptIn:
    @pytest.mark.parametrize("phase, n", [("nowcast", 6), ("forecast", 18)])
    def test_legacy_fallback_writes_phase_anchored_21600(self, tmp_path, monkeypatch, phase, n):
        _stage(tmp_path)
        proc, out = _proc(tmp_path, phase)
        monkeypatch.setattr(NudgingProcessor, "_process_ops",
                            lambda self: (_ for _ in ()).throw(ValueError("boom")))
        called = []
        monkeypatch.setattr(NudgingProcessor, "_call_fortran_gen_nudge",
                            lambda self, w: called.append(1) or False)
        res = proc._process_stofs()
        assert not called
        assert res.success and res.metadata["nudge_interp"] in ("delaunay", "precomputed"), res.warnings
        assert res.metadata["dt_seconds"] == 21600.0
        assert not any("wrong speed" in w for w in res.warnings)
        with Dataset(str(out / "TEM_nu.nc")) as ds:
            t = np.asarray(ds["time"][:], float)
        assert len(t) == n
        np.testing.assert_array_equal(t, np.arange(n) * 21600.0)
        assert any("fell back to" in w for w in res.warnings)

    @staticmethod
    def _fallback(tmp_path, monkeypatch, phase, keep):
        _stage(tmp_path)
        d = tmp_path / "rtofs" / f"rtofs.{PDY}"
        for f in d.glob("*.nc"):
            if not any(tag in f.name for tag in keep):
                f.unlink()
        proc, out = _proc(tmp_path, phase)
        monkeypatch.setattr(NudgingProcessor, "_process_ops",
                            lambda self: (_ for _ in ()).throw(ValueError("boom")))
        monkeypatch.setattr(NudgingProcessor, "_call_fortran_gen_nudge", lambda self, w: False)
        res = proc._process_stofs()
        with Dataset(str(out / "TEM_nu.nc")) as ds:
            t = np.asarray(ds["time"][:], float)
            tr = np.array(ds["tracer_concentration"][:])
        return res, t, tr

    @pytest.mark.parametrize("phase, n", [("nowcast", 6), ("forecast", 18)])
    def test_partial_coverage_holds_the_last_state(self, tmp_path, monkeypatch, phase, n):
        res, t, tr = self._fallback(tmp_path, monkeypatch, phase, ("n012", "n018"))
        assert res.success, res.errors
        assert len(t) == n
        np.testing.assert_array_equal(t, np.arange(n) * 21600.0)
        assert any("held" in w for w in res.warnings)
        np.testing.assert_allclose(tr[-1], tr[-2])

    @pytest.mark.parametrize("phase, n", [("nowcast", 6), ("forecast", 18)])
    def test_single_file_is_replicated_over_the_full_axis(self, tmp_path, monkeypatch, phase, n):
        res, t, tr = self._fallback(tmp_path, monkeypatch, phase, ("n024",))
        assert res.success, res.errors
        assert len(t) == n
        np.testing.assert_array_equal(t, np.arange(n) * 21600.0)
        assert any("held" in w for w in res.warnings)
        np.testing.assert_allclose(tr[0], tr[-1])

    def test_fortran_path_raises_when_nudge_tsuv_prep_fails(self, tmp_path, monkeypatch):
        _stage(tmp_path)
        proc, _ = _proc(tmp_path, "nowcast")
        monkeypatch.setattr(NudgingProcessor, "_prepare_nudge_tsuv", lambda self, w: None)
        called = []
        monkeypatch.setattr(NudgingProcessor, "_call_fortran_gen_nudge", lambda self, w: called.append(1) or True)
        with pytest.raises(RuntimeError, match="nudge ROI"):
            proc._process_ops_fortran()
        assert not called

    def test_wrong_dt_is_warned_under_ops_timeline(self, tmp_path):
        proc, _ = _proc(tmp_path, "nowcast")
        res = proc._check_ops_dt(ForcingResult(True, "NUDGING", metadata={"dt_seconds": 10800.0}))
        assert any("wrong speed" in w for w in res.warnings)

    def test_fortran_is_not_used_unless_opted_in(self, tmp_path, monkeypatch):
        _stage(tmp_path)
        proc, _ = _proc(tmp_path, "nowcast")
        called = []
        monkeypatch.setattr(NudgingProcessor, "_process_ops_fortran", lambda self: called.append(1))
        assert proc._process_stofs().metadata["nudge_interp"] == "ops" and not called

    def test_fortran_opt_in_records_are_resampled_with_E(self, tmp_path, monkeypatch):
        nodes = _stage(tmp_path)
        proc, out = _proc(tmp_path, "forecast")
        proc.config.obc_use_fortran_gen_nudge = True
        rng = np.random.default_rng(3)
        raw = {n: [rng.normal(size=(5, 4)).astype(np.float32) for _ in range(23)] for n in ("TEM_nu.nc", "SAL_nu.nc")}

        monkeypatch.setattr(NudgingProcessor, "_prepare_nudge_tsuv", lambda self, w: w / "TSUV_1.nc")

        def fake_exe(self, work):
            for name, recs in raw.items():
                with Dataset(str(work / name), "w") as ds:
                    ds.createDimension("node", 5)
                    ds.createDimension("nLevels", 4)
                    ds.createDimension("one", 1)
                    ds.createDimension("time", None)
                    ds.createVariable("time", "f8", ("time",))[:] = np.arange(23) * 0.25
                    ds.createVariable("map_to_global_node", "i4", ("node",))[:] = np.arange(1, 6)
                    v = ds.createVariable("tracer_concentration", "f4", ("time", "node", "nLevels", "one"))
                    for k, r in enumerate(recs):
                        v[k, :, :, 0] = r
            return True

        monkeypatch.setattr(NudgingProcessor, "_call_fortran_gen_nudge", fake_exe)
        res = proc._process_stofs()
        assert res.metadata["nudge_interp"] == "fortran_ops", res.warnings
        assert res.metadata["dt_seconds"] == 21600.0 and res.metadata["n_timesteps"] == 18
        with Dataset(str(out / "TEM_nu.nc")) as ds:
            tr = np.array(ds["tracer_concentration"][:])
        r = raw["TEM_nu.nc"]
        for j in (0, 3, 6, 16, 17):
            k, num = divmod(4 + min(j, 16), 10)
            want = r[k] if num == 0 else (r[k].astype(np.float64) * (1 - num / 10) + r[k + 1].astype(np.float64) * num / 10)
            np.testing.assert_allclose(tr[j, :, :, 0], want, atol=1e-6)

    def test_non_ops_config_is_unchanged(self, tmp_path, monkeypatch):
        proc, _ = _proc(tmp_path, "nowcast", ops_tl=False)
        proc.config.obc_use_fortran_gen_nudge = True
        called = []
        monkeypatch.setattr(NudgingProcessor, "_process_ops_fortran", lambda self: called.append(1))
        monkeypatch.setattr(NudgingProcessor, "_process_stofs_legacy",
                            lambda self: ForcingResult(True, "NUDGING", metadata={"dt_seconds": 10800.0}))
        res = proc._process_stofs()
        assert not called and not res.warnings


class TestTiming:
    def test_160k_nodes_parent_search_and_state_are_fast(self):
        lon, lat = _curvi(742, 179, dx=0.075, dy=0.05, shear=0.02)
        rng = np.random.default_rng(1)
        n = 160_000
        px = rng.uniform(lon.min(), lon.max(), n).astype(F)
        py = rng.uniform(lat.min(), lat.max(), n).astype(F)
        nz = 6
        s = no.pack_ts(np.full((nz, 742, 179), 35.0))
        z = np.tile(np.linspace(-80, 0, 12, dtype=F), (n, 1))
        t0 = time.time()
        state, found = no.build_state(lon, lat, np.linspace(0, 100, nz).astype(F), px, py,
                                      np.ones(n, bool), z, s)
        dt = time.time() - t0
        assert found.sum() > 0.3 * n and dt < 90.0, dt
