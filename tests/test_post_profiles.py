"""Tests for nos_utils.post.profiles."""
from __future__ import annotations

import pytest

netCDF4 = pytest.importorskip("netCDF4")
import numpy as np  # noqa: E402

from nos_utils.post.profiles import (  # noqa: E402
    FILL_VALUE,
    VARS_3D,
    compute_area_coords,
    read_station_in,
    stack_inputs,
    write_station_profiles,
)
from tests.post_fixtures import (  # noqa: E402
    MINI_ELEMS_1BASED,
    MINI_NVRT,
    MINI_X,
    MINI_Y,
    write_mini_hgrid,
    write_mini_vgrid,
    write_out2d_stack,
    write_var3d_stack,
)

MINI_ELNODE = np.where(MINI_ELEMS_1BASED > 0, MINI_ELEMS_1BASED - 1, -1)

# Fixture value laws (see tests/post_fixtures.py):
#   elevation(h, n) = (n + 1) + h/100
#   var3d(h, n, k) = (n + 1)*10 + k + h/100
# Station A sits on node 0 -> weights (1,0,0); station B sits at the
# centroid of element 2 (nodes 2,4,3 1-based -> mean node factor 3).
STA_LON = [0.0, 2.0 / 3.0]
STA_LAT = [0.0, 2.0 / 3.0]
STA_FACTOR = [1.0, 3.0]


def _make_stack(tmp_path, stack, hours):
    files = {
        "out2d": write_out2d_stack(tmp_path / f"out2d_{stack}.nc", hours)
    }
    for var in VARS_3D:
        files[var] = write_var3d_stack(
            tmp_path / f"{var}_{stack}.nc", var, hours
        )
    return files


def _mesh(tmp_path):
    return (
        write_mini_hgrid(tmp_path / "hgrid.gr3"),
        write_mini_vgrid(tmp_path / "vgrid.in"),
    )


def _station_names(ds):
    """chartostring returns bytes or str depending on the version."""
    out = []
    for x in np.atleast_1d(netCDF4.chartostring(ds["station_name"][:])):
        if isinstance(x, bytes):
            x = x.decode()
        out.append(str(x).rstrip("\x00"))
    return out


def test_area_coords_node_centroid_edge_outside():
    ie, ip, acor = compute_area_coords(
        MINI_X, MINI_Y, MINI_ELNODE,
        [0.0, 2.0 / 3.0, 0.5, 5.0],
        [0.0, 2.0 / 3.0, 0.5, 5.0],
    )
    # station at node 0 -> element 0, weights (1,0,0)
    assert ie[0] == 0
    assert list(ip[0]) == [0, 1, 2]
    assert np.allclose(acor[0], [1.0, 0.0, 0.0])
    # element-2 centroid -> equal thirds on nodes (1,3,2)
    assert ie[1] == 1
    assert list(ip[1]) == [1, 3, 2]
    assert np.allclose(acor[1], [1 / 3, 1 / 3, 1 / 3])
    # on the shared diagonal: inclusive >=0 edges, first element wins
    assert ie[2] == 0
    assert np.allclose(acor[2], [0.0, 0.5, 0.5])
    # outside the mesh: pylib fallback -- nearest node (3), acor (1,0,0)
    assert ie[3] == -1
    assert list(ip[3]) == [3, 3, 3]
    assert np.allclose(acor[3], [1.0, 0.0, 0.0])
    # weights reproduce the point coordinates (partition of unity)
    for i in range(3):
        assert np.isclose((MINI_X[ip[i]] * acor[i]).sum(), STA_LON[i]
                          if i < 2 else 0.5)


def test_area_coords_quad_second_triangle():
    # one CCW quad (0,1,2,3): tris (0,1,2) then (0,2,3)
    x = np.array([0.0, 1.0, 1.0, 0.0])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    elnode = np.array([[0, 1, 2, 3]])
    ie, ip, acor = compute_area_coords(
        x, y, elnode, [0.9, 0.1], [0.2, 0.9]
    )
    assert list(ie) == [0, 0]
    assert list(ip[0]) == [0, 1, 2]
    assert np.allclose(acor[0], [0.1, 0.7, 0.2])
    assert list(ip[1]) == [0, 2, 3]
    assert np.allclose(acor[1], [0.1, 0.1, 0.8])


def test_profile_values_single_stack(tmp_path):
    hgrid, vgrid = _mesh(tmp_path)
    stack = _make_stack(tmp_path, 1, hours=[1, 2])
    out = tmp_path / "profile.nc"

    result = write_station_profiles(
        [stack], hgrid, vgrid, out, base_date="2026-07-10-00",
        lons=STA_LON, lats=STA_LAT, names=["STA_A", "STA_B"],
    )
    assert result == out

    with netCDF4.Dataset(out) as ds:
        assert ds.data_model == "NETCDF3_CLASSIC"
        assert len(ds.dimensions["station"]) == 2
        assert len(ds.dimensions["siglay"]) == MINI_NVRT
        assert len(ds.dimensions["namelen"]) == 100
        assert ds.dimensions["time"].isunlimited()

        assert list(ds["time"][:]) == [3600.0, 7200.0]
        assert ds["time"].units == "seconds since 2026-07-10 00:00:00 UTC"
        assert ds["time"].base_date == "2026-07-10 00:00:00 UTC"

        assert np.allclose(ds["lon"][:], STA_LON)
        assert np.allclose(ds["lat"][:], STA_LAT)
        assert np.allclose(ds["depth"][:], [5.0, 5.0])

        # elevation: factor + h/100
        expect_zeta = [[1.01, 3.01], [1.02, 3.02]]
        assert np.allclose(ds["zeta"][:], expect_zeta)
        assert ds["zeta"].long_name == (
            "water surface elevation above navd88"
        )
        assert ds["zeta"].standard_name == (
            "sea_surface_height_above_navd88"
        )
        assert ds["zeta"]._FillValue == np.float32(FILL_VALUE)

        # 3D vars: factor*10 + k + h/100, identical law for all five
        expect = np.array(
            [[[f * 10 + k + h / 100.0 for k in range(MINI_NVRT)]
              for f in STA_FACTOR]
             for h in (1, 2)]
        )
        for var in ("salinity", "temperature", "u", "v", "zCoordinates"):
            assert np.allclose(ds[var][:], expect), var
        assert ds["u"].long_name == "Eastward Water Velocity"
        assert ds["temperature"].units == "degree_C"
        assert ds["zCoordinates"].long_name == (
            "vertical coordinate, positive upward"
        )

        # fixture out2d has no wind vars -> fill (masked on read-back)
        assert np.ma.getmaskarray(ds["uwind_speed"][:]).all()
        assert np.ma.getmaskarray(ds["vwind_speed"][:]).all()

        assert _station_names(ds) == ["STA_A", "STA_B"]
        assert ds.title == "SCHISM Model output"
        assert ds.references == "http://ccrm.vims.edu/schismweb/"


def test_wind_interpolated_when_present(tmp_path):
    hgrid, vgrid = _mesh(tmp_path)
    stack = _make_stack(tmp_path, 1, hours=[1])
    with netCDF4.Dataset(stack["out2d"], "a") as ds:
        for var, base in (("windSpeedX", 100.0), ("windSpeedY", 200.0)):
            v = ds.createVariable(
                var, "f4", ("time", "nSCHISM_hgrid_node")
            )
            v[0, :] = [base + n for n in range(4)]
    out = tmp_path / "profile.nc"
    write_station_profiles(
        [stack], hgrid, vgrid, out, base_date="2026-07-10-00",
        lons=STA_LON, lats=STA_LAT,
    )
    with netCDF4.Dataset(out) as ds:
        # node 0 -> base + 0; centroid of nodes (1,3,2) -> base + 2
        assert np.allclose(ds["uwind_speed"][:], [[100.0, 102.0]])
        assert np.allclose(ds["vwind_speed"][:], [[200.0, 202.0]])


def test_outside_station_fallback_and_error(tmp_path):
    hgrid, vgrid = _mesh(tmp_path)
    stack = _make_stack(tmp_path, 1, hours=[1])
    out = tmp_path / "profile.nc"

    write_station_profiles(
        [stack], hgrid, vgrid, out, base_date="2026-07-10-00",
        lons=[5.0], lats=[5.0],
    )
    with netCDF4.Dataset(out) as ds:
        # nearest node is 3 (factor 4): value taken from that node
        assert np.allclose(ds["zeta"][:], [[4.01]])
        assert np.allclose(
            ds["salinity"][:],
            [[[40.01, 41.01, 42.01]]],
        )
        assert np.allclose(ds["depth"][:], [5.0])

    # ops driver parity: abort on out-of-mesh stations
    with pytest.raises(ValueError, match="outside of domain"):
        write_station_profiles(
            [stack], hgrid, vgrid, out, base_date="2026-07-10-00",
            lons=[5.0], lats=[5.0], outside="error",
        )


def test_multi_stack_time_concatenation(tmp_path):
    hgrid, vgrid = _mesh(tmp_path)
    stacks = [
        _make_stack(tmp_path, 1, hours=[1, 2]),
        _make_stack(tmp_path, 2, hours=[3, 4]),
    ]
    out = tmp_path / "profile.nc"
    write_station_profiles(
        stacks, hgrid, vgrid, out, base_date="2026-07-10 00:00:00",
        lons=STA_LON, lats=STA_LAT,
    )
    with netCDF4.Dataset(out) as ds:
        assert list(ds["time"][:]) == [
            3600.0, 7200.0, 10800.0, 14400.0
        ]
        assert ds["time"].units == "seconds since 2026-07-10 00:00:00 UTC"
        assert np.allclose(
            ds["zeta"][:, 0], [1.01, 1.02, 1.03, 1.04]
        )
        # last record, centroid station, top layer: 30 + 2 + 0.04
        assert np.isclose(ds["salinity"][3, 1, 2], 32.04)


def test_station_in_parsing(tmp_path):
    # ops flavour: single-space tokens, trailing sentinel comma on names
    sta = tmp_path / "station.in"
    sta.write_text(
        "1 1 1 1 1 1 1 1 0 !on (1)|off(0) flags\n"
        "3 !# of stations\n"
        "1 0.0 0.0 0 !STA_A,\n"
        "2 0.666666666667 0.666666666667 0 !STA_B,\n"
        "3 5.0 5.0 0\n"  # no '!name' -> not a station (ops skips)
    )
    lons, lats, names = read_station_in(sta)
    assert np.allclose(lons, [0.0, 0.666666666667])
    assert np.allclose(lats, [0.0, 0.666666666667])
    assert list(names) == ["STA_A", "STA_B"]

    hgrid, vgrid = _mesh(tmp_path)
    stack = _make_stack(tmp_path, 1, hours=[1])
    out = tmp_path / "profile.nc"
    write_station_profiles(
        [stack], hgrid, vgrid, out, base_date="2026-07-10-00",
        station_file=sta,
    )
    with netCDF4.Dataset(out) as ds:
        assert _station_names(ds) == ["STA_A", "STA_B"]
        assert np.allclose(ds["zeta"][:], [[1.01, 3.01]])


def test_stack_inputs_helper_and_validation(tmp_path):
    files = stack_inputs(tmp_path / "outputs", 3)
    assert files["out2d"] == tmp_path / "outputs" / "out2d_3.nc"
    assert files["salinity"] == tmp_path / "outputs" / "salinity_3.nc"
    assert set(files) == {"out2d", *VARS_3D}

    with pytest.raises(ValueError, match="no stack inputs"):
        write_station_profiles(
            [], tmp_path / "h", tmp_path / "v", tmp_path / "o.nc",
            base_date="2026-07-10-00", lons=[0.0], lats=[0.0],
        )
    with pytest.raises(ValueError, match="not both"):
        write_station_profiles(
            [{}], tmp_path / "h", tmp_path / "v", tmp_path / "o.nc",
            base_date="2026-07-10-00", station_file=tmp_path / "s",
            lons=[0.0], lats=[0.0],
        )
    with pytest.raises(ValueError, match="stations required"):
        write_station_profiles(
            [{}], tmp_path / "h", tmp_path / "v", tmp_path / "o.nc",
            base_date="2026-07-10-00",
        )
    with pytest.raises(ValueError, match="outside"):
        write_station_profiles(
            [{}], tmp_path / "h", tmp_path / "v", tmp_path / "o.nc",
            base_date="2026-07-10-00", lons=[0.0], lats=[0.0],
            outside="bogus",
        )
