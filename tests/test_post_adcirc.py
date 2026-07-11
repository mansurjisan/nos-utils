"""Tests for nos_utils.post.adcirc."""
from __future__ import annotations

import pytest

netCDF4 = pytest.importorskip("netCDF4")
import numpy as np  # noqa: E402

from nos_utils.post.adcirc import (  # noqa: E402
    FILL_VALUE,
    city_mask_from_polygons,
    write_adcirc,
)
from tests.post_fixtures import (  # noqa: E402
    MINI_ELEMS_1BASED,
    MINI_X,
    MINI_Y,
    default_elev,
    write_out2d_stack,
)

BASE = "2026-07-10 00:00:00"

# Hand-built masking scenario on the mini mesh, hours [1, 2], with
# dryFlagNode (the current-ops path):
#   node 0: city node just below geoid (depth 0.05), wet both records;
#           inundation max(0, 0.2 + 0.05) = 0.25 m -> small -> filled
#   node 1: land node (ground 1 m above geoid), inundated 0.5 m at h2
#   node 2: ocean node, dry at h1 and wet at h2
#   node 3: land node (ground 3 m up), dry the whole run
SCEN_DEPTH = np.array([0.05, -1.0, 5.0, -3.0])
SCEN_ELEV = {0: (0.1, 0.2), 1: (0.5, 1.5),
             2: (-5.0, -4.9), 3: (-2.9, -2.8)}
SCEN_DRY = {0: (0, 0), 1: (0, 0), 2: (1, 0), 3: (1, 1)}
SCEN_CITY = np.array([True, False, False, False])


def _scen_elev(hour, node):
    return SCEN_ELEV[node][int(hour) - 1]


def _scen_dry(hour, node):
    return SCEN_DRY[node][int(hour) - 1]


def _default_wind(hour, node):
    return 10.0 * (node + 1) + hour


def _adcirc_stack(
    path,
    hours,
    elev_fn=default_elev,
    depth=None,
    dry_fn=None,
    wind=True,
    wind_fn=_default_wind,
    faces=True,
):
    """out2d stack extended with the vars generate_adcirc consumes."""
    hours = list(hours)
    write_out2d_stack(path, hours, elev_fn=elev_fn)
    with netCDF4.Dataset(path, "a") as ds:
        if depth is not None:
            ds["depth"][:] = depth
        if faces:
            ds.createDimension(
                "nSCHISM_hgrid_face", len(MINI_ELEMS_1BASED)
            )
            ds.createDimension("nMaxSCHISM_hgrid_face_nodes", 4)
            fv = ds.createVariable(
                "SCHISM_hgrid_face_nodes", "i4",
                ("nSCHISM_hgrid_face", "nMaxSCHISM_hgrid_face_nodes"),
            )
            fv[:] = MINI_ELEMS_1BASED
        if dry_fn is not None:
            dv = ds.createVariable(
                "dryFlagNode", "i4", ("time", "nSCHISM_hgrid_node")
            )
            for it, h in enumerate(hours):
                dv[it, :] = [dry_fn(h, n) for n in range(4)]
        if wind:
            for var, sign in (("windSpeedX", 1.0), ("windSpeedY", -1.0)):
                wv = ds.createVariable(
                    var, "f4", ("time", "nSCHISM_hgrid_node")
                )
                for it, h in enumerate(hours):
                    wv[it, :] = [sign * wind_fn(h, n) for n in range(4)]
    return path


def test_adcirc_masking_semantics_dryflag(tmp_path):
    s = _adcirc_stack(tmp_path / "out2d_1.nc", [1, 2],
                      elev_fn=_scen_elev, depth=SCEN_DEPTH,
                      dry_fn=_scen_dry)
    out = tmp_path / "adcirc.nc"

    write_adcirc([s], out, base_date=BASE, city_mask=SCEN_CITY)

    with netCDF4.Dataset(out) as ds:
        ds.set_auto_mask(False)
        zmax = ds["zeta_max"][:]
        tmax = ds["time_of_zeta_max"][:]
        dmax = ds["disturbance_max"][:]

        # Dry records are filled in zeta itself ...
        assert ds["zeta"][0, 2] == FILL_VALUE
        assert np.isclose(ds["zeta"][1, 2], -4.9)
        assert np.all(ds["zeta"][:, 3] == FILL_VALUE)
        # ... so the reductions skip them: node 2's max is its wet
        # record, and always-dry node 3's max is the fill itself.
        assert np.allclose(zmax[:3], [0.2, 1.5, -4.9])
        assert zmax[3] == FILL_VALUE
        # time_of_zeta_max is never filled (all-dry tie -> 1st record).
        assert np.allclose(tmax, [7200.0, 7200.0, 7200.0, 3600.0])
        # City node converted to inundation depth (0.25) then filled as
        # small; land inundation kept; ocean keeps its raw (negative)
        # max; always-dry land clamps to 0 then fills as small.
        assert dmax[0] == FILL_VALUE
        assert np.isclose(dmax[1], 0.5)
        assert np.isclose(dmax[2], -4.9)
        assert dmax[3] == FILL_VALUE


def test_adcirc_legacy_never_wet_fallback_without_dryflag(tmp_path):
    # No dryFlagNode in the stack -> older-lineage never-wet fill.
    depth = np.array([2.0, -1.0, 5.0, -3.0])
    elev = {0: (0.1, 0.2), 1: (0.5, 1.5),
            2: (-5.0, -5.0), 3: (-2.9, -2.8)}
    s = _adcirc_stack(
        tmp_path / "out2d_1.nc", [1, 2],
        elev_fn=lambda h, n: elev[n][int(h) - 1], depth=depth,
    )
    out = tmp_path / "adcirc.nc"

    write_adcirc([s], out, base_date=BASE, city_mask=SCEN_CITY)

    with netCDF4.Dataset(out) as ds:
        ds.set_auto_mask(False)
        zmax = ds["zeta_max"][:]
        dmax = ds["disturbance_max"][:]

        # zeta keeps the bed-pinned dry elevations (no dryFlagNode) ...
        assert np.allclose(ds["zeta"][:, 2], [-5.0, -5.0])
        # ... and never-wet nodes are filled via zeta_max + depth <= 0.
        assert np.allclose(zmax[:2], [0.2, 1.5])
        assert zmax[2] == FILL_VALUE and zmax[3] == FILL_VALUE
        assert np.allclose(ds["time_of_zeta_max"][:],
                           [7200.0, 7200.0, 3600.0, 7200.0])
        # Conversion covers city nodes too: 0.2 + 2.0 = 2.2 kept.
        assert np.isclose(dmax[0], 2.2)
        assert np.isclose(dmax[1], 0.5)
        assert np.isclose(dmax[2], -5.0)
        assert dmax[3] == FILL_VALUE


def test_adcirc_ocean_small_disturbance_kept_without_city_mask(tmp_path):
    s = _adcirc_stack(tmp_path / "out2d_1.nc", [1, 2],
                      elev_fn=_scen_elev, depth=SCEN_DEPTH,
                      dry_fn=_scen_dry)
    out = tmp_path / "adcirc.nc"

    write_adcirc([s], out, base_date=BASE)

    with netCDF4.Dataset(out) as ds:
        ds.set_auto_mask(False)
        # Without the city mask node 0 is plain ocean (depth > 0): no
        # inundation conversion, no small-disturbance fill.
        assert np.isclose(ds["disturbance_max"][0], 0.2)


def test_adcirc_min_disturbance_parameterized(tmp_path):
    s = _adcirc_stack(tmp_path / "out2d_1.nc", [1, 2],
                      elev_fn=_scen_elev, depth=SCEN_DEPTH,
                      dry_fn=_scen_dry)
    out = tmp_path / "adcirc.nc"

    write_adcirc([s], out, base_date=BASE, city_mask=SCEN_CITY,
                 min_disturbance=0.1)

    with netCDF4.Dataset(out) as ds:
        ds.set_auto_mask(False)
        dmax = ds["disturbance_max"][:]
        assert np.isclose(dmax[0], 0.25)  # city node above threshold
        assert dmax[3] == FILL_VALUE      # dry land still below it


def test_adcirc_multi_stack_reduces_full_window(tmp_path):
    # default_elev grows with the hour -> global max in the 2nd stack;
    # node 0 flagged dry in the first record only.
    def dry_fn(hour, node):
        return 1 if (hour == 1 and node == 0) else 0

    s1 = _adcirc_stack(tmp_path / "out2d_1.nc", [1, 2], dry_fn=dry_fn)
    s2 = _adcirc_stack(tmp_path / "out2d_2.nc", [3, 4], dry_fn=dry_fn)
    out = tmp_path / "adcirc.nc"

    write_adcirc([s1, s2], out, base_date=BASE)

    with netCDF4.Dataset(out) as ds:
        ds.set_auto_mask(False)
        assert list(ds["time"][:]) == [3600.0, 7200.0, 10800.0, 14400.0]
        expected = [n + 1 + 4 / 100.0 for n in range(4)]
        assert np.allclose(ds["zeta_max"][:], expected)
        assert np.allclose(ds["time_of_zeta_max"][:], 14400.0)
        # All wet ocean nodes -> disturbance is the raw max elevation.
        assert np.allclose(ds["disturbance_max"][:], expected)
        assert ds["zeta"].shape == (4, 4)
        # dryFlagNode concatenates across stacks like the data vars.
        assert ds["zeta"][0, 0] == FILL_VALUE
        assert np.isclose(ds["uwind"][2, 0], _default_wind(3, 0))
        assert np.isclose(ds["vwind"][3, 1], -_default_wind(4, 1))


def test_adcirc_junk_values_filled(tmp_path):
    def elev_fn(hour, node):
        if node == 0 and hour == 1:
            return 9.0e5
        return default_elev(hour, node)

    def wind_fn(hour, node):
        if node == 1 and hour == 1:
            return 2.0e5
        return 3.0

    s = _adcirc_stack(tmp_path / "out2d_1.nc", [1, 2],
                      elev_fn=elev_fn, wind_fn=wind_fn)
    out = tmp_path / "adcirc.nc"

    write_adcirc([s], out, base_date=BASE)

    with netCDF4.Dataset(out) as ds:
        ds.set_auto_mask(False)
        assert ds["zeta"][0, 0] == FILL_VALUE
        # Junk record ignored by the reductions.
        assert np.isclose(ds["zeta_max"][0], default_elev(2, 0))
        assert ds["time_of_zeta_max"][0] == 7200.0
        # Ops quirk: BOTH components filled where uwind is junk, even
        # though vwind (= -2e5 here) is not junk itself.
        assert ds["uwind"][0, 1] == FILL_VALUE
        assert ds["vwind"][0, 1] == FILL_VALUE
        assert np.isclose(ds["vwind"][1, 1], -3.0)


def test_adcirc_schema_and_attrs(tmp_path):
    s = _adcirc_stack(tmp_path / "out2d_1.nc", [1, 2])
    out = write_adcirc([s], tmp_path / "adcirc.nc", base_date=BASE)

    with netCDF4.Dataset(out) as ds:
        assert ds.dimensions["time"].isunlimited()
        assert len(ds.dimensions["node"]) == 4
        assert len(ds.dimensions["nele"]) == 2
        assert len(ds.dimensions["nvertex"]) == 3

        assert ds["time"].long_name == "Time"
        assert ds["time"].base_date == BASE
        assert ds["time"].units == f"seconds since {BASE}"

        assert ds["x"].standard_name == "longitude"
        assert ds["y"].positive == "north"
        assert np.allclose(ds["x"][:], MINI_X)

        el = ds["element"]
        assert el.standard_name == "face_node_connectivity"
        assert el.start_index == 1
        assert el.units == "nondimensional"
        assert el[:].tolist() == [[1, 2, 3], [2, 4, 3]]

        # Datum spliced verbatim (mixed case, like current ops).
        assert ds["depth"].long_name == "distance below xGEOID20B"
        assert ds["depth"].standard_name == "depth below xGEOID20B"
        assert ds["depth"].coordinates == "y x"  # ops fixed "time y x"
        assert np.allclose(ds["depth"][:], 5.0)

        assert (ds["zeta"].standard_name
                == "sea_surface_height_above_xGEOID20B")
        assert ds["zeta"].coordinates == "time y x"
        assert (ds["zeta_max"].standard_name
                == "maximum_sea_surface_height_above_xGEOID20B")
        assert (ds["time_of_zeta_max"].standard_name
                == "time_of_maximum_sea_surface_height_above_xGEOID20B")
        assert ds["time_of_zeta_max"].units == "sec"
        # Ops-verbatim standard_name, "depature" typo included.
        assert (ds["disturbance_max"].standard_name
                == "maximum_depature_from_initial_condition")
        assert ds["disturbance_max"].coordinates == "y x"
        for var in ("zeta", "zeta_max", "time_of_zeta_max",
                    "disturbance_max", "uwind", "vwind"):
            assert ds[var]._FillValue == FILL_VALUE
        assert ds["uwind"].long_name == "10m_above_ground/UGRD"
        assert ds["uwind"].units == "ms-1"
        assert ds["vwind"].standard_name == "northward_wind"

        assert ds.title == "SCHISM Model output"
        assert ds.source == "SCHISM model output version v10"
        assert ds.references == "http://ccrm.vims.edu/schismweb/"


def test_adcirc_datum_parameterized(tmp_path):
    s = _adcirc_stack(tmp_path / "out2d_1.nc", [1])
    out = tmp_path / "adcirc.nc"

    write_adcirc([s], out, base_date=BASE, datum="NAVD88")

    with netCDF4.Dataset(out) as ds:
        assert ds["depth"].long_name == "distance below NAVD88"
        assert (ds["zeta_max"].standard_name
                == "maximum_sea_surface_height_above_NAVD88")


def test_adcirc_without_wind_vars(tmp_path):
    s = _adcirc_stack(tmp_path / "out2d_1.nc", [1, 2], wind=False)
    out = tmp_path / "adcirc.nc"

    write_adcirc([s], out, base_date=BASE)

    with netCDF4.Dataset(out) as ds:
        assert "uwind" not in ds.variables
        assert "vwind" not in ds.variables
        assert "zeta" in ds.variables


def test_adcirc_elnode_override_splits_quads(tmp_path):
    s = _adcirc_stack(tmp_path / "out2d_1.nc", [1], faces=False)
    out = tmp_path / "adcirc.nc"

    write_adcirc([s], out, base_date=BASE,
                 elnode=np.array([[1, 2, 4, 3]]))

    with netCDF4.Dataset(out) as ds:
        assert len(ds.dimensions["nele"]) == 2
        assert ds["element"][:].tolist() == [[1, 2, 4], [1, 4, 3]]


def test_adcirc_input_validation(tmp_path):
    with pytest.raises(ValueError, match="no out2d"):
        write_adcirc([], tmp_path / "o.nc", base_date=BASE)

    s = _adcirc_stack(tmp_path / "out2d_1.nc", [1], faces=False)
    with pytest.raises(ValueError, match="elnode"):
        write_adcirc([s], tmp_path / "o.nc", base_date=BASE)

    s2 = _adcirc_stack(tmp_path / "out2d_2.nc", [1])
    with pytest.raises(ValueError, match="city_mask"):
        write_adcirc([s2], tmp_path / "o.nc", base_date=BASE,
                     city_mask=np.array([True, False]))


def test_city_mask_from_polygons(tmp_path):
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import box

    # Unit square around node 0 only (nodes at (0,0),(1,0),(0,1),(1,1)).
    polys = gpd.GeoDataFrame(geometry=[box(-0.5, -0.5, 0.5, 0.5)])

    mask = city_mask_from_polygons(MINI_X, MINI_Y, polys)
    assert mask.dtype == np.dtype(bool)
    assert mask.tolist() == [True, False, False, False]

    # GeoSeries input takes the same path.
    mask2 = city_mask_from_polygons(MINI_X, MINI_Y, polys.geometry)
    assert mask2.tolist() == [True, False, False, False]
