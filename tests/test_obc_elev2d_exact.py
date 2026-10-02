"""float32 f90-order elev2D and the held ADT field."""

from datetime import datetime

import numpy as np
import pytest

pytest.importorskip("netCDF4")
pytest.importorskip("scipy")
from netCDF4 import Dataset  # noqa: E402

from nos_utils.forcing import obc_ops_interp as oi  # noqa: E402
from nos_utils.forcing.adt import ADTBlender  # noqa: E402
from nos_utils.forcing.rtofs import RTOFSProcessor  # noqa: E402

from .test_adt_blend import _blend, _cfg, _elev, _write_2d, setup  # noqa: E402,F401
from .test_stofs_obc_bilinear import _proc, _write_ssh1  # noqa: E402

f32 = np.float32


def test_single_weights_are_float32_and_match_double():
    lo, la = np.meshgrid(np.linspace(0, 3, 7), np.linspace(10, 14, 9))
    lo = lo + 0.05 * la
    x = np.array([0.7, 1.31, 2.2])
    y = np.array([10.9, 12.2, 13.3])
    ix, iy, w, f = oi.parent_weights_2d(lo, la, x, y)
    ix1, iy1, w1, f1 = oi.parent_weights_2d(lo, la, x, y, single=True)
    assert w1.dtype == np.float32 and f1.all()
    np.testing.assert_array_equal((ix, iy), (ix1, iy1))
    np.testing.assert_allclose(w1, w, atol=2e-6)


def test_interpolate4_sums_left_to_right_in_float32():
    s = np.array([[1e8, 1.0, -1e8, 1.0]], f32)
    w = np.ones((1, 4), f32)
    assert oi.interpolate4(s, w)[0] == f32(1.0)
    assert (s[:, 0] + s[:, 1] + s[:, 2] + s[:, 3])[0] == f32(1.0)
    assert float(s[0, 0] + (s[0, 1] + (s[0, 2] + s[0, 3]))) == 0.0


def _ssh1_packed(path, lons, lats, surf):
    _write_ssh1(path, lons, lats, [np.zeros_like(surf)])
    with Dataset(str(path), "r+") as ds:
        v = ds.createVariable("surf_el", "f4", ("time", "ylat", "xlon"), fill_value=-30000.0)
        v.set_auto_maskandscale(False)
        v[0] = surf
    return path


def test_ops_ssh_boundary_uses_packed_surf_el_as_float32(tmp_path):
    lons = np.array([0.0, 1.0, 2.0, 3.0])
    lats = np.array([0.0, 1.0, 2.0, 3.0])
    rng = np.random.default_rng(3)
    surf = (rng.uniform(100, 900, (4, 4)) / 7).astype(f32)
    p = _ssh1_packed(tmp_path / "SSH_1.nc", lons, lats, surf)
    proc = _proc(tmp_path, [1.3, 2.1], [0.4, 1.7], obc_interp_mode=1)
    got = proc._ops_ssh_boundary(p, 1)
    assert got.dtype == np.float32
    LO, LA = np.meshgrid(lons, lats)
    ix, iy, w, _ = oi.parent_weights_2d(LO, LA, [1.3, 2.1], [0.4, 1.7], single=True)
    cj, ci = oi.corner_cells(ix, iy)
    ssh = (surf * f32(1e-3)).astype(f32)
    want = oi.interpolate4(ssh[cj, ci].T, w)
    np.testing.assert_array_equal(got[0], want)


def test_blend_record_0_is_the_adt_field_bit_exact(setup):
    cfg, proc, files, ssh_1, work, tmp = setup
    with Dataset(str(ssh_1)) as ds:
        ny, nx = ds["ssh"].shape[1:]
    adt = (np.random.default_rng(1).uniform(-0.3, 0.9, (ny, nx))).astype(f32)
    adt[0, :3] = np.nan
    out = ADTBlender(cfg, tmp, ops_numerics=True)._apply_adt_blend(ssh_1, None, work, adt_dst=adt)
    with Dataset(str(out)) as ds:
        ds.set_auto_maskandscale(False)
        s0, e0, e3 = ds["ssh"][0], ds["surf_el"][0], ds["surf_el"][3]
        np.testing.assert_array_equal(s0[1:], adt[1:])
        np.testing.assert_array_equal(s0[0, :3], f32(-30000))
        np.testing.assert_array_equal(e0[1:], (adt[1:] * f32(1000)).astype(f32))
        np.testing.assert_array_equal(e0[0, :3], f32(-30000))
        d = (ds["ssh"][3] - ds["ssh"][0])[1:]
        np.testing.assert_allclose(e3[1:], (d + adt[1:]) * f32(1000), atol=1e-3)


def test_held_row_is_the_adt_record_not_the_nowcast_start_value(setup):
    cfg, proc, files, ssh_1, work, tmp = setup
    cfg.nowcast_hours = 6  # the old hold took the ADT blend at 06z, i.e. ADT + (ssh(06z) - ssh(00z))
    proc._ssh1_path = ssh_1
    blended = _blend(cfg, tmp, ssh_1, work, lambda lo, la: np.full_like(lo, 0.90))
    held = _elev(proc._process_2d(files, ssh_source=blended))
    assert (held == held[0]).all()
    np.testing.assert_allclose(held, 0.90 - 0.45 + 0.04, atol=2e-5)
