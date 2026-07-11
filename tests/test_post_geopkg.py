"""Tests for nos_utils.post.geopkg."""
from __future__ import annotations

import pytest

gpd = pytest.importorskip("geopandas")
pytest.importorskip("matplotlib")
netCDF4 = pytest.importorskip("netCDF4")
import numpy as np  # noqa: E402

from nos_utils.post.geopkg import (  # noqa: E402
    DEFAULT_LEVELS,
    FILL_VALUE,
    SMALL_DISTURBANCE_THRESHOLD,
    disturbance_field,
    nowcast_forecast_namer,
    write_disturbance_gpkg,
    write_disturbance_series,
)
from tests.post_fixtures import (  # noqa: E402
    MINI_DEPTH,
    MINI_ELEMS_1BASED,
    MINI_X,
    MINI_Y,
    write_out2d_stack,
)


def _jet():
    import matplotlib

    try:
        return matplotlib.colormaps["jet"]
    except AttributeError:
        from matplotlib import cm

        return cm.get_cmap("jet")


def _add_mesh(path):
    """Append the mini-mesh element table to a fixture stack file."""
    with netCDF4.Dataset(path, "a") as ds:
        ds.createDimension("nSCHISM_hgrid_face", len(MINI_ELEMS_1BASED))
        ds.createDimension("nMaxSCHISM_hgrid_face_nodes", 4)
        fv = ds.createVariable(
            "SCHISM_hgrid_face_nodes", "i4",
            ("nSCHISM_hgrid_face", "nMaxSCHISM_hgrid_face_nodes"),
            fill_value=-1,
        )
        fv[:] = MINI_ELEMS_1BASED
    return path


def test_default_levels_ladder():
    assert DEFAULT_LEVELS[0] == -5.0
    assert DEFAULT_LEVELS[-1] == 20.0
    inner = np.array(DEFAULT_LEVELS[1:-1])
    assert inner[0] == -0.5
    assert inner[-1] == 2.0
    assert np.allclose(np.diff(inner), 0.1)
    # Current ops small-on-land threshold matches the bottom level.
    assert SMALL_DISTURBANCE_THRESHOLD == -5.0


def test_disturbance_field_masking():
    depth = np.array([5.0, 5.0, -2.0, -2.0, -2.0])
    elev = np.array([1.2, -5.0, 3.0, 2.2, 1.0])
    elev_orig = elev.copy()

    d = disturbance_field(elev, depth)

    assert d[0] == pytest.approx(1.2)  # wet water: plain elevation
    assert d[1] == FILL_VALUE  # dry water: elev + depth <= 1e-6
    assert d[2] == pytest.approx(1.0)  # land: elevation + depth
    # Current ops threshold -5 never fires on land (clamped >= 0).
    assert d[3] == pytest.approx(0.2)
    assert d[4] == FILL_VALUE  # land dry: elev + depth < 0
    assert np.array_equal(elev, elev_orig)  # ops deepcopies


def test_legacy_small_disturbance_threshold(tmp_path):
    # Legacy ops masked land nodes with disturbance < 0.3 m.
    depth = np.array([5.0, 5.0, 5.0, -2.0])
    elev = np.array([0.6, 0.8, 1.0, 2.2])  # node 3: land, 2.2 - 2 = 0.2

    d = disturbance_field(elev, depth, small_disturbance_threshold=0.3)
    assert d[3] == FILL_VALUE
    assert np.allclose(d[:3], elev[:3])

    # Passed through the writer: the triangle touching the masked land
    # node drops out (area 0.5), while the current default keeps it.
    legacy = tmp_path / "legacy.gpkg"
    write_disturbance_gpkg(
        elev, depth, MINI_X, MINI_Y, MINI_ELEMS_1BASED, legacy,
        small_disturbance_threshold=0.3,
    )
    current = tmp_path / "current.gpkg"
    write_disturbance_gpkg(
        elev, depth, MINI_X, MINI_Y, MINI_ELEMS_1BASED, current
    )
    area_legacy = gpd.read_file(legacy).geometry.union_all().area
    area_current = gpd.read_file(current).geometry.union_all().area
    assert area_legacy == pytest.approx(0.5, rel=1e-6)
    assert area_current == pytest.approx(1.0, rel=1e-6)


def test_write_gpkg_wet_field(tmp_path):
    elev = np.array([0.2, 0.6, 1.0, 1.4])
    out = tmp_path / "disturbance.gpkg"

    res = write_disturbance_gpkg(
        elev, MINI_DEPTH, MINI_X, MINI_Y, MINI_ELEMS_1BASED, out
    )

    assert res == out
    assert out.exists()
    layers = gpd.list_layers(out)
    assert list(layers["name"]) == ["disturbance"]

    gdf = gpd.read_file(out, layer="disturbance")
    assert gdf.crs.to_epsg() == 4326
    assert list(gdf.columns) == [
        "id", "minWaterLevel", "maxWaterLevel",
        "verticalDatum", "units", "rgba", "geometry",
    ]
    assert len(gdf) >= 1
    assert set(gdf["verticalDatum"]) == {"XGEOID20B"}
    assert set(gdf["units"]) == {"meters"}
    assert (~gdf.geometry.is_empty).all()
    assert set(gdf.geom_type) <= {"Polygon", "MultiPolygon"}

    # Bands partition the all-wet mesh: union is the full unit square.
    assert gdf.geometry.union_all().area == pytest.approx(1.0, rel=1e-6)

    # Band bounds ascend along the mandated level ladder.
    assert gdf["id"].is_monotonic_increasing
    assert (gdf["minWaterLevel"] < gdf["maxWaterLevel"]).all()
    ladder = {round(level, 1) for level in DEFAULT_LEVELS}
    assert set(np.round(gdf["maxWaterLevel"], 1)) <= ladder
    assert gdf["minWaterLevel"].min() >= 0.1 - 1e-9
    assert gdf["maxWaterLevel"].max() <= 1.5 + 1e-9

    # rgba = alpha-less jet(i / nrows) hex, exactly as ops derives it.
    from matplotlib import colors as mcolors

    cmap = _jet()
    expected = [
        mcolors.to_hex(cmap(i / len(gdf))) for i in range(len(gdf))
    ]
    assert list(gdf["rgba"]) == expected
    assert gdf["rgba"].iloc[0] == "#000080"


def test_write_gpkg_all_dry_returns_none(tmp_path):
    elev = -MINI_DEPTH  # elevation + depth == 0 everywhere -> dry
    out = tmp_path / "dry.gpkg"

    res = write_disturbance_gpkg(
        elev, MINI_DEPTH, MINI_X, MINI_Y, MINI_ELEMS_1BASED, out
    )

    assert res is None
    assert not out.exists()


def test_extend_band_min_is_field_min_including_fill(tmp_path):
    # Node 3 dry -> FILL_VALUE; the wet triangle sits below levels[0],
    # so the extend='min' band emits with the ops' raw field minimum.
    depth = np.full(4, 10.0)
    elev = np.array([-6.0, -5.5, -7.0, -10.0])
    out = tmp_path / "extend.gpkg"

    res = write_disturbance_gpkg(
        elev, depth, MINI_X, MINI_Y, MINI_ELEMS_1BASED, out
    )

    assert res == out
    gdf = gpd.read_file(out, layer="disturbance")
    assert len(gdf) == 1
    assert gdf["id"].iloc[0] == 1
    assert gdf["minWaterLevel"].iloc[0] == pytest.approx(FILL_VALUE)
    assert gdf["maxWaterLevel"].iloc[0] == pytest.approx(-5.0)
    # Only the unmasked triangle (half the unit square) has geometry.
    assert gdf.geometry.union_all().area == pytest.approx(0.5, rel=1e-6)


def test_series_names_and_order(tmp_path):
    s1 = _add_mesh(write_out2d_stack(tmp_path / "out2d_1.nc", hours=[1, 2]))
    s2 = _add_mesh(write_out2d_stack(tmp_path / "out2d_2.nc", hours=[3, 4]))
    out_dir = tmp_path / "gpkg"

    written = write_disturbance_series(
        [s1, s2], out_dir, nowcast_forecast_namer("mini_ofs", 6, n_nowcast=2)
    )

    assert [p.name for p in written] == [
        "mini_ofs.t06z.disturbance.n001.gpkg",
        "mini_ofs.t06z.disturbance.n000.gpkg",
        "mini_ofs.t06z.disturbance.f001.gpkg",
        "mini_ofs.t06z.disturbance.f002.gpkg",
    ]
    for p in written:
        assert p.parent == out_dir
        assert p.exists()
    gdf = gpd.read_file(written[-1], layer="disturbance")
    assert len(gdf) >= 1


def test_series_skips_all_dry_timestep(tmp_path):
    def elev_fn(hour, node):
        return -5.0 if hour == 1 else float(node + 1)

    s1 = _add_mesh(
        write_out2d_stack(tmp_path / "out2d_1.nc", hours=[1, 2],
                          elev_fn=elev_fn)
    )
    out_dir = tmp_path / "gpkg"

    written = write_disturbance_series(
        [s1], out_dir, nowcast_forecast_namer("mini_ofs", 0, n_nowcast=0)
    )

    assert [p.name for p in written] == [
        "mini_ofs.t00z.disturbance.f002.gpkg"
    ]
    assert not (out_dir / "mini_ofs.t00z.disturbance.f001.gpkg").exists()


def test_namer_matches_ops_hardcoded_list():
    ops = [
        f"stofs_3d_atl.t12z.disturbance.n{i - 1:03d}.gpkg"
        for i in range(24, 0, -1)
    ]
    ops += [
        f"stofs_3d_atl.t12z.disturbance.f{i + 1:03d}.gpkg"
        for i in range(96)
    ]

    name_fn = nowcast_forecast_namer("stofs_3d_atl", 12)

    assert [name_fn(i) for i in range(120)] == ops


def test_series_parallel_matches_serial(tmp_path):
    s1 = _add_mesh(write_out2d_stack(tmp_path / "out2d_1.nc", hours=[1, 2]))
    namer = nowcast_forecast_namer("mini_ofs", 0, n_nowcast=0)

    serial = write_disturbance_series([s1], tmp_path / "ser", namer)
    parallel = write_disturbance_series(
        [s1], tmp_path / "par", namer, max_workers=2
    )

    assert [p.name for p in serial] == [p.name for p in parallel]
    for a, b in zip(serial, parallel):
        ga = gpd.read_file(a, layer="disturbance")
        gb = gpd.read_file(b, layer="disturbance")
        assert len(ga) == len(gb)
        assert list(ga["rgba"]) == list(gb["rgba"])
        assert list(ga["maxWaterLevel"]) == list(gb["maxWaterLevel"])


def test_series_rejects_empty_inputs(tmp_path):
    with pytest.raises(ValueError):
        write_disturbance_series(
            [], tmp_path, nowcast_forecast_namer("mini_ofs", 0)
        )
