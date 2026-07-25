"""Tests for nos_utils.post.stations."""
from __future__ import annotations

import json

import pytest

netCDF4 = pytest.importorskip("netCDF4")
pytest.importorskip("scipy")
import numpy as np  # noqa: E402

from nos_utils.post.stations import (  # noqa: E402
    FILL_VALUE,
    load_station_csv,
    write_station_timeseries,
)

BASE_DATE = "2026-02-13 12:00:00 UTC"

# Mirrors the ops stofs_3d_atl_staout_nc.json structure, including the
# 'stardard_name' typo key (the 'u' entry checks the sane spelling too).
VAR_DEFS = {
    "elev": {
        "staout_fname": "staout_1",
        "name": "zeta",
        "long_name": "water surface elevation above navd88",
        "stardard_name": "sea_surface_height_above_navd88",
        "units": "m",
    },
    "temperature": {
        "staout_fname": "staout_5",
        "name": "temperature",
        "long_name": "temperature at water surface",
        "stardard_name": "sea_surface_temperature",
        "units": "degree_C",
    },
    "u": {
        "staout_fname": "staout_7",
        "name": "u",
        "long_name": "u component of surface velocity",
        "standard_name": "eastward_surface_velocity",
        "units": "meters s-1",
    },
}

# (station_info, lon, lat); second name is longer than namelen=50.
STATIONS = [
    ("PSBM1 SOUS41 8410140 ME Eastport", -66.9829, 44.9046),
    ("Q" * 60, -70.2442, 43.6581),
]

# Linear in t so interp1d reproduces them exactly at any t.
VALUE_FNS = {
    1: lambda t, s: s * 100.0 + t / 100.0,
    5: lambda t, s: 20.0 + s + t / 1000.0,
    7: lambda t, s: 0.5 * s - t / 500.0,
}
RAW_TIMES = [180.0 * k for k in range(1, 7)]  # 180..1080, model dt=180


def write_staout(path, times, value_fn, n_station=2):
    lines = []
    for t in times:
        vals = " ".join(f"{value_fn(t, s):.8e}" for s in range(n_station))
        lines.append(f"{t:.8e} {vals}")
    path.write_text("\n".join(lines) + "\n")
    return path


def make_staout_files(tmp_path, indices=(1, 5, 7), times=RAW_TIMES):
    return {
        i: write_staout(tmp_path / f"staout_{i}", times, VALUE_FNS[i])
        for i in indices
    }


def test_full_product(tmp_path):
    staout = make_staout_files(tmp_path)
    out = tmp_path / "staout_timeseries.nc"

    result = write_station_timeseries(
        staout, VAR_DEFS, STATIONS, out, base_date=BASE_DATE
    )
    assert result == out

    t_expect = np.array([360.0, 720.0, 1080.0])
    with netCDF4.Dataset(out) as ds:
        assert len(ds.dimensions["station"]) == 2
        assert len(ds.dimensions["namelen"]) == 50
        assert len(ds.dimensions["time"]) == 3
        assert ds.dimensions["time"].isunlimited()

        tv = ds["time"]
        assert np.allclose(tv[:], t_expect)
        assert tv.units == f"seconds since {BASE_DATE}"
        assert tv.base_date == BASE_DATE
        assert tv.long_name == "Time"
        assert tv.standard_name == "time"

        names = netCDF4.chartostring(ds["station_name"][:])
        assert names[0] == STATIONS[0][0]
        assert names[1] == "Q" * 50  # truncated like ops
        assert ds["station_name"].long_name == "station name"

        assert np.allclose(ds["x"][:], [-66.9829, -70.2442])
        assert np.allclose(ds["y"][:], [44.9046, 43.6581])
        assert ds["x"].units == "degrees_east"
        assert ds["x"].positive == "east"
        assert ds["y"].standard_name == "latitude"
        assert ds["y"].positive == "north"

        for idx, name in ((1, "zeta"), (5, "temperature"), (7, "u")):
            v = ds[name]
            assert v.dimensions == ("time", "station")
            assert v.dtype == np.float64
            assert v._FillValue == FILL_VALUE
            expect = np.array(
                [[VALUE_FNS[idx](t, s) for s in range(2)] for t in t_expect]
            )
            assert np.allclose(v[:], expect)

        assert ds["zeta"].standard_name == "sea_surface_height_above_navd88"
        assert ds["zeta"].units == "m"
        assert ds["u"].standard_name == "eastward_surface_velocity"

        assert ds.title == "SCHISM Model output"
        assert ds.source == "SCHISM model output version v10"
        assert ds.references == "http://ccrm.vims.edu/schismweb/"


def test_metadata_accepted_as_paths(tmp_path):
    """JSON/CSV paths work; CSV columns are located by header name.

    The CSV replicates the production quirk: header order is
    ``station_info;lat;lon`` while the data is lon-first, so the
    name-based read must land the 'lon'-labelled column in x.
    """
    staout = make_staout_files(tmp_path, indices=(1,))
    json_path = tmp_path / "staout_nc.json"
    json_path.write_text(json.dumps({"elev": VAR_DEFS["elev"]}))
    csv_path = tmp_path / "staout_nc.csv"
    csv_path.write_text(
        ";station_info;lat;lon\n"
        "0;PSBM1 SOUS41 8410140 ME Eastport;-66.9829;44.9046\n"
        "1;CFWM1 SOUS41 8411060 ME Cutler;-67.2047;44.6570\n"
    )

    rows = load_station_csv(csv_path)
    assert rows[0] == ("PSBM1 SOUS41 8410140 ME Eastport", 44.9046, -66.9829)

    out = tmp_path / "out.nc"
    write_station_timeseries(
        staout, json_path, csv_path, out, base_date=BASE_DATE
    )
    with netCDF4.Dataset(out) as ds:
        assert np.allclose(ds["x"][:], [44.9046, 44.6570])
        assert np.allclose(ds["y"][:], [-66.9829, -67.2047])
        names = netCDF4.chartostring(ds["station_name"][:])
        assert names[1] == "CFWM1 SOUS41 8411060 ME Cutler"


def test_datum_offsets_shift_zeta_only(tmp_path):
    staout = make_staout_files(tmp_path, indices=(1, 5))
    var_defs = {k: VAR_DEFS[k] for k in ("elev", "temperature")}
    plain = tmp_path / "plain.nc"
    shifted = tmp_path / "shifted.nc"
    offsets = np.array([0.30516, -0.08734])  # negated .nco constants

    write_station_timeseries(
        staout, var_defs, STATIONS, plain, base_date=BASE_DATE
    )
    write_station_timeseries(
        staout, var_defs, STATIONS, shifted, base_date=BASE_DATE,
        datum_offsets=offsets,
    )

    with netCDF4.Dataset(plain) as d0, netCDF4.Dataset(shifted) as d1:
        assert np.allclose(d1["zeta"][:], d0["zeta"][:] + offsets)
        assert np.allclose(d1["temperature"][:], d0["temperature"][:])


def test_datum_offsets_wrong_length(tmp_path):
    staout = make_staout_files(tmp_path, indices=(1,))
    with pytest.raises(ValueError, match="datum_offsets"):
        write_station_timeseries(
            staout, {"elev": VAR_DEFS["elev"]}, STATIONS,
            tmp_path / "o.nc", base_date=BASE_DATE,
            datum_offsets=np.array([0.1]),
        )


def test_short_staout_extrapolates(tmp_path):
    # Two records ending before the first tick: single interpolated
    # time at 360 s, linearly extrapolated (ops fill_value=extrapolate).
    staout = {
        1: write_staout(
            tmp_path / "staout_1", [100.0, 200.0], lambda t, s: t + s
        )
    }
    out = tmp_path / "o.nc"
    write_station_timeseries(
        staout, {"elev": VAR_DEFS["elev"]}, STATIONS, out,
        base_date=BASE_DATE,
    )
    with netCDF4.Dataset(out) as ds:
        assert list(ds["time"][:]) == [360.0]
        assert np.allclose(ds["zeta"][:], [[360.0, 361.0]])


def test_custom_interval(tmp_path):
    staout = make_staout_files(tmp_path, indices=(1,))
    out = tmp_path / "o.nc"
    write_station_timeseries(
        staout, {"elev": VAR_DEFS["elev"]}, STATIONS, out,
        base_date=BASE_DATE, interval_seconds=180.0,
    )
    with netCDF4.Dataset(out) as ds:
        assert np.allclose(ds["time"][:], RAW_TIMES)


def test_ragged_staout_raises(tmp_path):
    p = tmp_path / "staout_1"
    p.write_text("180.0 1.0 2.0\n360.0 1.0\n")
    with pytest.raises(ValueError):
        write_station_timeseries(
            {1: p}, {"elev": VAR_DEFS["elev"]}, STATIONS,
            tmp_path / "o.nc", base_date=BASE_DATE,
        )


def test_station_column_mismatch_raises(tmp_path):
    # 3 value columns but 2 stations: ops would broadcast-error.
    p = write_staout(
        tmp_path / "staout_1", RAW_TIMES, VALUE_FNS[1], n_station=3
    )
    with pytest.raises(ValueError, match="station columns"):
        write_station_timeseries(
            {1: p}, {"elev": VAR_DEFS["elev"]}, STATIONS,
            tmp_path / "o.nc", base_date=BASE_DATE,
        )


def test_mismatched_time_axes_raise(tmp_path):
    staout = make_staout_files(tmp_path, indices=(1,))
    staout[5] = write_staout(
        tmp_path / "staout_5", RAW_TIMES[:4], VALUE_FNS[5]
    )
    with pytest.raises(ValueError, match="time axis"):
        write_station_timeseries(
            staout, {k: VAR_DEFS[k] for k in ("elev", "temperature")},
            STATIONS, tmp_path / "o.nc", base_date=BASE_DATE,
        )


def test_missing_staout_entry_raises(tmp_path):
    staout = make_staout_files(tmp_path, indices=(1,))
    with pytest.raises(ValueError, match="staout"):
        write_station_timeseries(
            staout, {k: VAR_DEFS[k] for k in ("elev", "temperature")},
            STATIONS, tmp_path / "o.nc", base_date=BASE_DATE,
        )


def test_empty_inputs_rejected(tmp_path):
    staout = make_staout_files(tmp_path, indices=(1,))
    with pytest.raises(ValueError, match="variable definitions"):
        write_station_timeseries(
            staout, {}, STATIONS, tmp_path / "o.nc", base_date=BASE_DATE
        )
    with pytest.raises(ValueError, match="no stations"):
        write_station_timeseries(
            staout, {"elev": VAR_DEFS["elev"]}, [],
            tmp_path / "o.nc", base_date=BASE_DATE,
        )
