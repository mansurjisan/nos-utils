"""Tests for nos_utils.post.slab2d."""
from __future__ import annotations

import pytest

netCDF4 = pytest.importorskip("netCDF4")
import numpy as np  # noqa: E402

from nos_utils.post.slab2d import FILL_VALUE, write_slab2d  # noqa: E402
from tests.post_fixtures import (  # noqa: E402
    MINI_DEPTH,
    MINI_ELEMS_1BASED,
    MINI_X,
    MINI_Y,
    write_out2d_stack,
    write_var3d_stack,
)

BASE_DATE = "2026-07-10 00:00:00 UTC"
HOURS = [1, 2]
KBP1 = np.array([1, 2, 1, 1])  # 1-based bottom_index_node
KBP0 = KBP1 - 1

# z columns (bottom->surface) chosen so with depth 5 m and the elevs
# below every interpolation regime is hand-computable: node 0 interior
# brackets, node 1 bottom-clamps at 4.5 m (below-bottom level repeats
# the bottom z), node 2 surface-clamps (column top below both targets),
# node 3 interior brackets while wet.
Z_COLS = np.array([
    [-5.0, -2.0, 0.0],
    [-3.0, -3.0, 0.0],
    [-5.0, -4.0, -3.75],
    [-5.0, -2.5, 0.0],
])
ELEV = [0.0, 0.0, 2.0, 0.0]  # node 3 goes dry (-5.0) at the 2nd record


def elev_fn(hour, node):
    if node == 3 and hour >= 2:
        return -5.0  # elev + depth = 0 <= 1e-6 -> dry
    return ELEV[node]


def z_fn(hour, node, k):
    return Z_COLS[node][k]


def t_fn(hour, node, k):
    if hour == 1 and node == 0 and k == 2:
        return 20000.0  # > 10000 junk-magnitude threshold
    return (node + 1) * 10.0 + k + 0.5 * hour


def s_fn(hour, node, k):
    return 30.0 + node + 0.25 * k + 0.5 * hour


def u_fn(hour, node, k):
    return node + 0.25 * k + 0.5 * hour


def v_fn(hour, node, k):
    return -(node + 0.25 * k) + 0.5 * hour


def build_stacks(tmp_path):
    return (
        write_out2d_stack(tmp_path / "out2d_1.nc", HOURS, elev_fn=elev_fn),
        write_var3d_stack(
            tmp_path / "zCoordinates_1.nc", "zCoordinates", HOURS,
            value_fn=z_fn,
        ),
        write_var3d_stack(
            tmp_path / "temperature_1.nc", "temperature", HOURS,
            value_fn=t_fn,
        ),
        write_var3d_stack(
            tmp_path / "salinity_1.nc", "salinity", HOURS, value_fn=s_fn
        ),
        write_var3d_stack(
            tmp_path / "horizontalVelX_1.nc", "horizontalVelX", HOURS,
            value_fn=u_fn,
        ),
        write_var3d_stack(
            tmp_path / "horizontalVelY_1.nc", "horizontalVelY", HOURS,
            value_fn=v_fn,
        ),
    )


def run_slab2d(tmp_path, **kwargs):
    paths = build_stacks(tmp_path)
    out = tmp_path / "field2d.nc"
    kwargs.setdefault("elements", MINI_ELEMS_1BASED)
    kwargs.setdefault("bottom_index_node", KBP1)
    write_slab2d(*paths, out, BASE_DATE, **kwargs)
    return out


def open_raw(path):
    ds = netCDF4.Dataset(path)
    ds.set_auto_mask(False)
    return ds


def expected_native(fn, kind):
    rows = []
    for h in HOURS:
        row = []
        for n in range(4):
            if n == 3 and h >= 2:
                row.append(FILL_VALUE)
                continue
            k = {"surface": 2,
                 "bottom": KBP0[n],
                 "near_bottom": KBP0[n] + 1}[kind]
            val = fn(h, n, k)
            row.append(FILL_VALUE if val > 10000 else val)
        rows.append(row)
    return np.array(rows)


def interp_expect(fn, h, n, d):
    """Mirror of the ops column interpolation for the crafted columns."""
    if n == 3 and h >= 2:
        return FILL_VALUE
    zinter = -d + elev_fn(h, n)
    z = Z_COLS[n]
    if zinter >= z[-1]:  # surface clamp: all weight on top level
        return fn(h, n, 2)
    if zinter < z[KBP0[n]]:  # bottom clamp: all weight on bottom level
        return fn(h, n, KBP0[n])
    for k in range(2):
        if z[k] <= zinter < z[k + 1]:
            w = (z[k + 1] - zinter) / (z[k + 1] - z[k])
            return fn(h, n, k) * w + fn(h, n, k + 1) * (1.0 - w)
    raise AssertionError("target depth not bracketed")


def expected_interp(fn, d):
    return np.array(
        [[interp_expect(fn, h, n, d) for n in range(4)] for h in HOURS]
    )


def test_slab2d_dims_time_and_mesh(tmp_path):
    out = run_slab2d(tmp_path)
    with open_raw(out) as ds:
        assert len(ds.dimensions["node"]) == 4
        assert len(ds.dimensions["nele"]) == 2
        assert len(ds.dimensions["nvertex"]) == 3
        assert ds.dimensions["time"].isunlimited()
        np.testing.assert_allclose(ds["time"][:], [3600.0, 7200.0])
        assert ds["time"].units == f"seconds since {BASE_DATE}"
        assert ds["time"].base_date == BASE_DATE
        np.testing.assert_allclose(ds["x"][:], MINI_X)
        np.testing.assert_allclose(ds["y"][:], MINI_Y)
        np.testing.assert_allclose(ds["depth"][:], MINI_DEPTH)
        np.testing.assert_array_equal(
            ds["element"][:], [[1, 2, 3], [2, 4, 3]]
        )


def test_surface_bottom_near_bottom_values(tmp_path):
    out = run_slab2d(tmp_path)
    cases = [
        ("temp_surface", t_fn, "surface"),
        ("temp_bottom", t_fn, "bottom"),
        ("salt_surface", s_fn, "surface"),
        ("salt_bottom", s_fn, "bottom"),
        ("uvel_surface", u_fn, "surface"),
        ("uvel_bottom", u_fn, "near_bottom"),
        ("vvel_surface", v_fn, "surface"),
        ("vvel_bottom", v_fn, "near_bottom"),
    ]
    with open_raw(out) as ds:
        for name, fn, kind in cases:
            np.testing.assert_allclose(
                ds[name][:], expected_native(fn, kind), rtol=1e-6,
                err_msg=name,
            )
        # T/S bottom is level kbp; velocity bottom is one level above.
        assert ds["temp_bottom"][0, 0] == t_fn(1, 0, 0)
        np.testing.assert_allclose(
            ds["uvel_bottom"][0, 0], u_fn(1, 0, 1), rtol=1e-6
        )


def test_fixed_depth_interpolation(tmp_path):
    out = run_slab2d(tmp_path)
    with open_raw(out) as ds:
        for d in (0.5, 4.5):
            for prefix, fn in (("uvel", u_fn), ("vvel", v_fn)):
                np.testing.assert_allclose(
                    ds[f"{prefix}{d:g}"][:], expected_interp(fn, d),
                    rtol=1e-6, err_msg=f"{prefix}{d:g}",
                )
        # Regime spot checks: node 0 brackets levels 0-1 at 4.5 m
        # (weight 2.5/3) and levels 1-2 at 0.5 m (weight 0.25).
        w = 2.5 / 3.0
        np.testing.assert_allclose(
            ds["uvel4.5"][0, 0],
            u_fn(1, 0, 0) * w + u_fn(1, 0, 1) * (1 - w), rtol=1e-6,
        )
        np.testing.assert_allclose(
            ds["uvel0.5"][0, 0],
            u_fn(1, 0, 1) * 0.25 + u_fn(1, 0, 2) * 0.75, rtol=1e-6,
        )
        # node 1: 4.5 m is below its bottom -> clamps to bottom value.
        np.testing.assert_allclose(
            ds["vvel4.5"][0, 1], v_fn(1, 1, KBP0[1]), rtol=1e-6
        )
        # node 2: column top sits below both targets -> surface value.
        np.testing.assert_allclose(
            ds["uvel0.5"][0, 2], u_fn(1, 2, 2), rtol=1e-6
        )
        np.testing.assert_allclose(
            ds["uvel4.5"][0, 2], u_fn(1, 2, 2), rtol=1e-6
        )


def test_dry_and_large_value_masking(tmp_path):
    out = run_slab2d(tmp_path)
    slab_names = [
        "temp_surface", "temp_bottom", "salt_surface", "salt_bottom",
        "uvel_surface", "uvel_bottom", "vvel_surface", "vvel_bottom",
        "uvel0.5", "vvel0.5", "uvel4.5", "vvel4.5",
    ]
    with open_raw(out) as ds:
        np.testing.assert_allclose(
            ds["zeta"][:],
            [[0.0, 0.0, 2.0, 0.0], [0.0, 0.0, 2.0, FILL_VALUE]],
        )
        assert ds["zeta"]._FillValue == FILL_VALUE
        for name in slab_names:
            data = ds[name][:]
            assert data[1, 3] == FILL_VALUE, name  # dry at 2nd record
            assert data[0, 3] != FILL_VALUE, name  # wet at 1st record
        # junk magnitude (20000) at the surface -> filled, that record only
        assert ds["temp_surface"][0, 0] == FILL_VALUE
        assert ds["temp_surface"][1, 0] == t_fn(2, 0, 2)


def test_quad_split_element_table(tmp_path):
    elems = np.array([[1, 2, 3, -1], [1, 2, 4, 3]])
    out = run_slab2d(tmp_path, elements=elems)
    with open_raw(out) as ds:
        assert len(ds.dimensions["nele"]) == 3
        np.testing.assert_array_equal(
            ds["element"][:], [[1, 2, 3], [1, 2, 4], [1, 4, 3]]
        )
        assert ds["element"].start_index == 1
        assert ds["element"].standard_name == "face_node_connectivity"


def test_variable_set_and_metadata(tmp_path):
    out = run_slab2d(tmp_path)
    with open_raw(out) as ds:
        assert set(ds.variables) == {
            "time", "x", "y", "element", "depth", "zeta",
            "temp_surface", "temp_bottom", "salt_surface", "salt_bottom",
            "uvel_surface", "uvel_bottom", "vvel_surface", "vvel_bottom",
            "uvel0.5", "vvel0.5", "uvel4.5", "vvel4.5",
        }
        assert ds["uvel4.5"].long_name == (
            "U-component at 4.5m below free surface"
        )
        assert ds["vvel0.5"].long_name == (
            "V-component at 0.5m below free surface"
        )
        assert ds["uvel0.5"].units == "m/s"
        assert ds["temp_bottom"].long_name == "Bottom temperature"
        assert ds["temp_surface"].units == "deg C"
        assert ds["salt_surface"].units == "psu"
        assert ds["temp_surface"]._FillValue == FILL_VALUE
        assert ds["zeta"].standard_name == (
            "sea_surface_height_above_xgeoid20b"
        )
        assert ds["depth"].long_name == "distance below XGEOID20B"
        assert ds["depth"].standard_name == "depth below XGEOID20B"
        assert ds["x"].standard_name == "longitude"
        assert ds["y"].units == "degrees_north"
        assert ds.title == "SCHISM Model output"
        assert ds.references == "http://ccrm.vims.edu/schismweb/"


def test_mesh_vars_fallback_from_out2d(tmp_path):
    paths = build_stacks(tmp_path)
    with netCDF4.Dataset(paths[0], "a") as ds:
        ds.createDimension("nSCHISM_hgrid_face", 2)
        ds.createDimension("nMaxSCHISM_hgrid_face_nodes", 4)
        ev = ds.createVariable(
            "SCHISM_hgrid_face_nodes", "i4",
            ("nSCHISM_hgrid_face", "nMaxSCHISM_hgrid_face_nodes"),
        )
        ev[:] = MINI_ELEMS_1BASED
        bv = ds.createVariable(
            "bottom_index_node", "i4", ("nSCHISM_hgrid_node",)
        )
        bv[:] = KBP1
    out = tmp_path / "field2d.nc"
    write_slab2d(*paths, out, BASE_DATE)
    with open_raw(out) as ds:
        np.testing.assert_array_equal(
            ds["element"][:], [[1, 2, 3], [2, 4, 3]]
        )
        assert ds["temp_bottom"][0, 1] == t_fn(1, 1, KBP0[1])


def test_missing_mesh_vars_raise(tmp_path):
    paths = build_stacks(tmp_path)
    out = tmp_path / "field2d.nc"
    with pytest.raises(ValueError, match="bottom_index_node"):
        write_slab2d(
            *paths, out, BASE_DATE, elements=MINI_ELEMS_1BASED
        )
    with pytest.raises(ValueError, match="SCHISM_hgrid_face_nodes"):
        write_slab2d(*paths, out, BASE_DATE, bottom_index_node=KBP1)


def test_depths_and_datum_parameterization(tmp_path):
    out = run_slab2d(tmp_path, depths=(4.5,), datum="navd88")
    with open_raw(out) as ds:
        assert "uvel4.5" in ds.variables
        assert "vvel4.5" in ds.variables
        assert "uvel0.5" not in ds.variables
        assert ds["zeta"].standard_name == (
            "sea_surface_height_above_navd88"
        )
        assert ds["depth"].long_name == "distance below NAVD88"
    with pytest.raises(ValueError, match="positive"):
        run_slab2d(tmp_path, depths=(-4.5,))
