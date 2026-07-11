"""Station time-series product (``points.cwl.temp.salt.vel.nc``).

Port of IT-STOFS ``ush/stofs_3d_atl/pysh/generate_station_timeseries
.py`` as driven by ``stofs_3d_atl_create_awips_shef.sh``: each SCHISM
``staout_N`` text file (rows of time + one value per station -- 3D
variables are already sampled by SCHISM at the station.in z-level,
surface in ops, so there is no vertical slicing here) is interpolated
onto a fixed-interval time axis (6-minute in ops:
``arange(dt, t_end + dt/2, dt)``, ends extrapolated) and written to a
single netCDF with the ops schema: ``time``/``station``/``namelen``
dims, char ``station_name``, ``x``/``y`` station coords, per-variable
f8 data with fill -99999 and attrs taken from the staout-nc JSON
variable definitions (whose ``stardard_name`` typo key is honoured).

Ops then shifts ``zeta`` from xGEOID20B to the output datum with ncap2
(``stofs_3d_atl_sta_cwl_xgeoid_to_navd.nco``, per-station constants
*subtracted*). That step is folded in as ``datum_offsets``: per-station
values *added* to the staout_1 (elevation) variable, so pass the
negated .nco constants to reproduce ops; None applies no shift (the
raw pre-ncap2 product).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Tuple, Union

import numpy as np

FILL_VALUE = -99999.0
#: SCHISM station-output index carrying water level (staout_1);
#: ``datum_offsets`` applies to variables sourced from it.
ELEVATION_STAOUT_INDEX = 1

StationRow = Tuple[str, float, float]


def load_station_csv(path: Union[str, Path]) -> List[StationRow]:
    """Parse the ops staout-nc station CSV into (name, lon, lat) rows.

    Format (``stofs_3d_atl_staout_nc.csv``): ``;``-separated, first
    column an unused row index, remaining columns located BY HEADER
    NAME (pandas ``read_csv`` semantics): ``station_info``, ``lon``,
    ``lat``. NB the production fix file's ``lat``/``lon`` headers are
    swapped relative to its data, so ops ``x``/``y`` actually carry
    lat/lon values; reading by name reproduces that faithfully when
    given the same file.
    """
    with open(path, newline="") as f:
        rows = [
            r for r in csv.reader(f, delimiter=";")
            if any(cell.strip() for cell in r)
        ]
    if not rows:
        raise ValueError(f"{path}: empty station csv")
    col = {name: i for i, name in enumerate(rows[0])}
    missing = [k for k in ("station_info", "lon", "lat") if k not in col]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    return [
        (
            row[col["station_info"]],
            float(row[col["lon"]]),
            float(row[col["lat"]]),
        )
        for row in rows[1:]
    ]


def _staout_index(staout_fname: str) -> int:
    """``'staout_5' -> 5`` (trailing integer of the ops JSON name)."""
    try:
        return int(str(staout_fname).rsplit("_", 1)[-1])
    except ValueError:
        raise ValueError(
            f"cannot parse staout index from "
            f"staout_fname={staout_fname!r}"
        )


def _interp_staout(
    path: Union[str, Path], nstation: int, interval_seconds: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Load one staout text file and interpolate to the fixed interval.

    Returns ``(t_interp, values)`` with ``values`` shaped
    ``(nt, nstation)``. Ops assigns ``data[:, 1:]`` into an
    ``(nt, nstation)`` array, so exactly one column per station is
    required; a mismatch raises instead of numpy's broadcast error.
    """
    from scipy import interpolate

    data = np.loadtxt(path, ndmin=2)
    if data.shape[1] != nstation + 1:
        raise ValueError(
            f"{path}: expected time + {nstation} station columns, "
            f"found {data.shape[1]}"
        )
    t_raw = data[:, 0]
    dt = float(interval_seconds)
    t_interp = np.arange(dt, t_raw[-1] + dt / 2.0, dt)
    f_interp = interpolate.interp1d(
        t_raw, data[:, 1:], axis=0, fill_value="extrapolate"
    )
    return t_interp, f_interp(t_interp)


def write_station_timeseries(
    staout_files: Mapping[int, Union[str, Path]],
    var_defs: Union[str, Path, Mapping[str, Mapping[str, str]]],
    station_meta: Union[str, Path, Iterable[StationRow]],
    out_path: Union[str, Path],
    base_date: str,
    interval_seconds: float = 360.0,
    datum_offsets: Optional[np.ndarray] = None,
    name_length: int = 50,
) -> Path:
    """Interpolate staout files to a fixed interval and write the nc.

    ``staout_files`` maps SCHISM staout indices to paths (ops feeds
    ``{1: .../staout_1, 5: ..., 6: ..., 7: ..., 8: ...}``); each
    variable definition selects its file via the trailing integer of
    its ``staout_fname``. ``var_defs`` is the ops staout-nc JSON (path
    or parsed mapping): ``{key: {staout_fname, name, long_name,
    stardard_name|standard_name, units}}``, written in mapping order
    with the first entry (ops: elevation/zeta) defining the time axis;
    every other staout must produce the same axis. ``station_meta`` is
    the ops station CSV path or (name, lon, lat) rows; lon -> ``x``,
    lat -> ``y``, names null-padded/truncated to ``name_length``.
    ``base_date`` is stamped verbatim into the time units/attr (ops
    format ``YYYY-MM-DD HH:00:00 UTC``, the nowcast begin).
    ``datum_offsets`` (shape ``(nstation,)``) is added to the staout_1
    variable -- the fold-in of the ops xGEOID20B->NAVD88 ncap2 shift,
    whose .nco constants are subtracted, so negate them here.
    """
    from netCDF4 import Dataset

    if isinstance(var_defs, (str, Path)):
        with open(var_defs) as f:
            var_defs = json.load(f)
    if isinstance(station_meta, (str, Path)):
        station_meta = load_station_csv(station_meta)
    stations = [
        (str(name), float(lon), float(lat))
        for name, lon, lat in station_meta
    ]
    if not var_defs:
        raise ValueError("write_station_timeseries: no variable definitions")
    if not stations:
        raise ValueError("write_station_timeseries: no stations")
    nstation = len(stations)
    if datum_offsets is not None:
        datum_offsets = np.asarray(datum_offsets, dtype=float)
        if datum_offsets.shape != (nstation,):
            raise ValueError(
                f"datum_offsets shape {datum_offsets.shape} != "
                f"({nstation},)"
            )

    t_axis: Optional[np.ndarray] = None
    with Dataset(out_path, "w", format="NETCDF4") as fout:
        for ivar, (var_key, spec) in enumerate(var_defs.items()):
            idx = _staout_index(spec["staout_fname"])
            if idx not in staout_files:
                raise ValueError(
                    f"no staout path given for index {idx} "
                    f"(variable {var_key!r})"
                )
            t_interp, values = _interp_staout(
                staout_files[idx], nstation, interval_seconds
            )

            if ivar == 0:
                t_axis = t_interp
                fout.createDimension("station", nstation)
                fout.createDimension("namelen", int(name_length))
                fout.createDimension("time", None)

                tv = fout.createVariable("time", "f8", ("time",))
                tv.long_name = "Time"
                tv.units = f"seconds since {base_date}"
                tv.base_date = base_date
                tv.standard_name = "time"
                tv[:] = t_interp

                sv = fout.createVariable(
                    "station_name", "c", ("station", "namelen")
                )
                sv.long_name = "station name"
                names = np.empty((nstation,), f"S{int(name_length)}")
                for i, (sname, _, _) in enumerate(stations):
                    names[i] = sname
                # ops uses netCDF4.stringtochar, broken for S-dtype
                # input under numpy 2; this view is its byte-identical
                # bytes-array path.
                sv[:] = names.view("S1").reshape(
                    nstation, int(name_length)
                )

                xv = fout.createVariable("x", "f8", ("station",))
                xv.long_name = "longitude"
                xv.standard_name = "longitude"
                xv.units = "degrees_east"
                xv.positive = "east"
                xv[:] = [lon for _, lon, _ in stations]

                yv = fout.createVariable("y", "f8", ("station",))
                yv.long_name = "latitude"
                yv.standard_name = "latitude"
                yv.units = "degrees_north"
                yv.positive = "north"
                yv[:] = [lat for _, _, lat in stations]

                fout.title = "SCHISM Model output"
                fout.source = "SCHISM model output version v10"
                fout.references = "http://ccrm.vims.edu/schismweb/"
            elif not np.array_equal(t_interp, t_axis):
                raise ValueError(
                    f"variable {var_key!r} (staout_{idx}) time axis "
                    "differs from the first variable's"
                )

            if datum_offsets is not None and idx == ELEVATION_STAOUT_INDEX:
                values = values + datum_offsets[np.newaxis, :]

            dv = fout.createVariable(
                spec["name"], "f8", ("time", "station"),
                fill_value=FILL_VALUE,
            )
            dv.long_name = spec["long_name"]
            dv.standard_name = (
                spec["stardard_name"] if "stardard_name" in spec
                else spec["standard_name"]
            )
            dv.units = spec["units"]
            dv[:, :] = values

    return Path(out_path)


__all__ = [
    "ELEVATION_STAOUT_INDEX",
    "FILL_VALUE",
    "load_station_csv",
    "write_station_timeseries",
]
