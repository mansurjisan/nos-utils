"""Tests for nos_utils.post.slab2d's STOFS-3D-ATL (``2ds``) output path.

Reuses the mesh/stack fixtures from ``test_post_slab2d`` -- the six
scribe-shaped stacks, the hand-computable interpolation columns, and
the expected-value helpers -- so the per-hour STOFS-named output is
checked against the same numbers the ADCIRC-named ``field2d`` writer
is, rather than a second hand-derived expectation.
"""
from __future__ import annotations

import pytest

netCDF4 = pytest.importorskip("netCDF4")
import numpy as np  # noqa: E402

from nos_utils.post.slab2d import (  # noqa: E402
    FILL_VALUE,
    compute_slab_fields,
    write_2ds_record,
)
from tests.post_fixtures import (  # noqa: E402
    MINI_DEPTH,
    MINI_ELEMS_1BASED,
    MINI_X,
    MINI_Y,
)
from tests.test_post_slab2d import (  # noqa: E402
    HOURS,
    KBP1,
    build_stacks,
    expected_interp,
    expected_native,
    s_fn,
    t_fn,
    u_fn,
    v_fn,
)

BASE_DATE = "2026-07-10 00:00:00 UTC"

_STOFS_SLAB_VARS = {
    "temp_surface", "temp_bottom", "salt_surface", "salt_bottom",
    "uvel_surface", "uvel_bottom", "vvel_surface", "vvel_bottom",
    "uvel0.5", "vvel0.5", "uvel4.5", "vvel4.5",
}


def compute(tmp_path):
    paths = build_stacks(tmp_path)
    return compute_slab_fields(
        *paths, elements=MINI_ELEMS_1BASED, bottom_index_node=KBP1,
    )


def write_record(tmp_path, record, name="2ds.nc"):
    fields = compute(tmp_path)
    out = tmp_path / name
    write_2ds_record(fields, record, out, BASE_DATE)
    return out, fields


def open_raw(path):
    ds = netCDF4.Dataset(path)
    ds.set_auto_mask(False)
    return ds


def test_2ds_dims_time_equals_one(tmp_path):
    out, _fields = write_record(tmp_path, 0)
    with open_raw(out) as ds:
        assert len(ds.dimensions["time"]) == 1
        assert len(ds.dimensions["nSCHISM_hgrid_node"]) == 4
        assert len(ds.dimensions["nSCHISM_hgrid_face"]) == 2
        assert len(ds.dimensions["nMaxSCHISM_hgrid_face_nodes"]) == 3


def test_2ds_mesh_vars_and_face_table_is_tri_split(tmp_path):
    out, _fields = write_record(tmp_path, 0)
    with open_raw(out) as ds:
        np.testing.assert_allclose(ds["SCHISM_hgrid_node_x"][:], MINI_X)
        np.testing.assert_allclose(ds["SCHISM_hgrid_node_y"][:], MINI_Y)
        np.testing.assert_allclose(ds["depth"][:], MINI_DEPTH)
        assert ds["SCHISM_hgrid_face_nodes"].shape == (2, 3)
        np.testing.assert_array_equal(
            ds["SCHISM_hgrid_face_nodes"][:], [[1, 2, 3], [2, 4, 3]]
        )
        assert ds["SCHISM_hgrid_face_nodes"].start_index == 1
        assert ds["SCHISM_hgrid_node_x"].standard_name == "longitude"
        assert ds["SCHISM_hgrid_node_y"].standard_name == "latitude"
        assert ds["depth"].long_name == "bathymetry"


def test_2ds_variable_set_and_fill_value(tmp_path):
    out, _fields = write_record(tmp_path, 0)
    with open_raw(out) as ds:
        assert set(ds.variables) == {
            "time", "SCHISM_hgrid_node_x", "SCHISM_hgrid_node_y",
            "SCHISM_hgrid_face_nodes", "depth", "elev",
        } | _STOFS_SLAB_VARS
        assert ds["elev"]._FillValue == FILL_VALUE
        for name in _STOFS_SLAB_VARS:
            assert ds[name]._FillValue == FILL_VALUE
        assert ds["uvel4.5"].long_name == (
            "U-component at 4.5m below free surface"
        )
        assert ds["temp_bottom"].long_name == "Bottom temperature"
        assert ds.title == "SCHISM Model output"
        assert ds.references == "http://ccrm.vims.edu/schismweb/"


def test_2ds_one_file_per_hour_matches_field2d_values(tmp_path):
    """Each hour's file carries exactly that record's slab values -- the
    same numbers the multi-record ADCIRC writer would put in that row."""
    for it, hour in enumerate(HOURS):
        out, _fields = write_record(tmp_path, it, name=f"2ds_{hour}.nc")
        with open_raw(out) as ds:
            np.testing.assert_allclose(ds["time"][:], [hour * 3600.0])
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
            for name, fn, kind in cases:
                np.testing.assert_allclose(
                    ds[name][0, :], expected_native(fn, kind)[it, :],
                    rtol=1e-6, err_msg=name,
                )
            for d in (0.5, 4.5):
                for prefix, fn in (("uvel", u_fn), ("vvel", v_fn)):
                    np.testing.assert_allclose(
                        ds[f"{prefix}{d:g}"][0, :],
                        expected_interp(fn, d)[it, :],
                        rtol=1e-6, err_msg=f"{prefix}{d:g}",
                    )


def test_2ds_dry_node_masked(tmp_path):
    # HOURS[1] == 2, where node 3 (0-based) goes dry in the fixture.
    out, _fields = write_record(tmp_path, 1)
    with open_raw(out) as ds:
        assert ds["elev"][0, 3] == FILL_VALUE
        for name in _STOFS_SLAB_VARS:
            assert ds[name][0, 3] == FILL_VALUE, name


def test_2ds_record_out_of_range_raises(tmp_path):
    fields = compute(tmp_path)
    with pytest.raises(IndexError, match="out of range"):
        write_2ds_record(fields, fields.ntimes, tmp_path / "x.nc", BASE_DATE)


def test_2ds_reuses_computed_fields_across_records(tmp_path):
    """One compute_slab_fields() call serves every hour's write -- the
    per-hour product must not recompute the science per file."""
    fields = compute(tmp_path)
    outs = []
    for it in range(fields.ntimes):
        out = tmp_path / f"hr_{it}.nc"
        write_2ds_record(fields, it, out, BASE_DATE)
        outs.append(out)
    assert len(outs) == len(HOURS)
    for out in outs:
        with open_raw(out) as ds:
            assert len(ds.dimensions["time"]) == 1
