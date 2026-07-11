"""Station vertical-profile product (``*.station.profile.nc``).

Port of IT-STOFS ``pysh/get_stations_profile.py`` (driven by
``stofs_3d_atl_create_station_profile_nc.sh``): interpolate the scribe
per-stack outputs -- ``out2d`` (elevation + wind) and the 3D
``salinity``/``temperature``/``horizontalVelX``/``horizontalVelY``/
``zCoordinates`` files -- to station points with SCHISM area
coordinates, concatenating all stacks in-code (the ops python loops
stack_start..stack_end the same way), and publish the NETCDF3_CLASSIC
station-profile file.

The pylib dependency of the ops script is replaced:

* ``read_schism_hgrid``/``save_schism_grid`` -> ``SchismGrid`` for the
  node table plus a local element-connectivity pass over hgrid.gr3.
* ``hgrid.compute_acor`` -> :func:`compute_area_coords`, an exact
  re-implementation of pylib's semantics: signed-area containment with
  inclusive (``>= 0``) edges on the CCW mesh, quads split as
  (a,b,c) + (a,c,d), first containing triangle in file order wins.
  Stations outside the mesh take pylib's fallback -- nearest node in
  all three slots with weights (1, 0, 0) and ``ie = -1``; the
  operational driver then aborts on any ``ie == -1``, available here
  as ``outside="error"`` (default ``"nearest"`` keeps the fallback).
* ``read_schism_vgrid`` -> a local two-line header read; the product
  only consumes ``nvrt`` for the ``siglay`` dimension (zCoordinates
  come from the scribe stacks), matching ops usage.
* ``read_schism_bpfile``/``read_station_file`` -> :func:`read_station_in`.

The zeta datum labeling ("water surface elevation above navd88") is
replicated as-is from ops: the actual xGEOID -> NAVD88 shift is applied
downstream by ``ncap2 -S ..._sta_cwl_xgeoid_to_navd.nco`` in the shell
driver, never in the extractor.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from nos_utils.io.schism_grid import SchismGrid
from nos_utils.post.mesh import split_quads

FILL_VALUE = -99999.0
NAMELEN = 100

#: scribe out2d variable -> profile-file variable (time, station)
VARS_2D = {
    "elevation": "zeta",
    "windSpeedX": "uwind_speed",
    "windSpeedY": "vwind_speed",
}
#: per-variable 3D stack key -> profile-file variable (time, station, siglay)
VARS_3D = {
    "salinity": "salinity",
    "temperature": "temperature",
    "horizontalVelX": "u",
    "horizontalVelY": "v",
    "zCoordinates": "zCoordinates",
}


def stack_inputs(outputs_dir: Path, stack: int) -> Dict[str, Path]:
    """Per-stack input files in the ops ``outputs/`` layout.

    The ops script derives them as ``outputs/out2d_{stack}.nc`` and
    ``outputs/{var}_{stack}.nc``; callers build the
    ``write_station_profiles`` stack list with
    ``[stack_inputs(d, i) for i in range(start, end + 1)]``.
    """
    outputs_dir = Path(outputs_dir)
    files = {"out2d": outputs_dir / f"out2d_{stack}.nc"}
    for var in VARS_3D:
        files[var] = outputs_dir / f"{var}_{stack}.nc"
    return files


def read_station_in(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse a SCHISM ``station.in`` the way the ops extractor consumes it.

    Format (see the operational ``stofs_3d_atl_station.in``)::

        1 1 1 1 1 1 1 1 0 !output flags
        108                !station count (ignored; '!' lines rule)
        1 -66.962027 44.915967 0 !PSBM1 SOUS41 8410140 ME Eastpor,

    The two header lines are skipped; every remaining line containing
    ``!`` is a station (rows without a name comment are ignored, like
    ops). Longitude/latitude are the 2nd/3rd whitespace tokens -- the
    ops reader splits on single spaces, which only parses
    single-space-separated files, so whitespace splitting accepts a
    strict superset. The name is everything after the last ``!`` minus
    its final character: the operational file pads each name with a
    trailing sentinel comma which the ops writer chops with
    ``str[:-1]``; the strip is replicated here at parse time so the
    writer stays truncation-free.

    Returns ``(lons, lats, names)``.
    """
    lons: List[float] = []
    lats: List[float] = []
    names: List[str] = []
    with open(path) as f:
        f.readline()
        f.readline()
        for line in f.read().splitlines():
            if "!" not in line:
                continue
            names.append(line.split("!")[-1][:-1])
            tok = line.split()
            lons.append(float(tok[1]))
            lats.append(float(tok[2]))
    return (
        np.asarray(lons, dtype=float),
        np.asarray(lats, dtype=float),
        np.asarray(names),
    )


def compute_area_coords(
    node_x: np.ndarray,
    node_y: np.ndarray,
    elnode: np.ndarray,
    px: np.ndarray,
    py: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Containing element + area coordinates (pylib ``compute_acor`` port).

    ``elnode`` is the (ne, 3|4) 0-based connectivity with -1 padding in
    the 4th column for triangles. Returns ``(ie, ip, acor)``:
    ``ie[npt]`` containing-element index (-1 = outside mesh),
    ``ip[npt, 3]`` node indices and ``acor[npt, 3]`` barycentric
    weights of the interpolation triangle.

    Semantics replicated from pylib: a point is inside a triangle iff
    all three signed sub-areas are >= 0 (edges and vertices inclusive;
    elements are CCW), the weights are ``[A1/A0, A2/A0, 1 - w1 - w2]``,
    quads are tested as (a,b,c) then (a,c,d), and ties on shared edges
    go to the first containing triangle in file order (pylib fmt=0's
    neighbor search may pick the other side of an edge, but the
    interpolated values are identical there since the off-edge vertex
    weight is 0). Points in no element fall back to their nearest node:
    ``ie = -1``, ``ip`` = nearest node in all slots, ``acor = (1,0,0)``.
    """
    node_x = np.asarray(node_x, dtype=float)
    node_y = np.asarray(node_y, dtype=float)
    elnode = np.asarray(elnode)
    px = np.atleast_1d(np.asarray(px, dtype=float))
    py = np.atleast_1d(np.asarray(py, dtype=float))

    ne = elnode.shape[0]
    tris = split_quads(elnode)
    parent = np.arange(ne)
    if tris.shape[0] > ne:
        parent = np.concatenate(
            [parent, np.nonzero(elnode[:, 3] >= 0)[0]]
        )

    x1, x2, x3 = (node_x[tris[:, k]] for k in range(3))
    y1, y2, y3 = (node_y[tris[:, k]] for k in range(3))
    a0 = ((x1 - x3) * (y2 - y3) - (x2 - x3) * (y1 - y3)) / 2.0

    npt = px.size
    ie = np.full(npt, -1, dtype=int)
    ip = np.full((npt, 3), -1, dtype=int)
    acor = np.zeros((npt, 3))
    for i in range(npt):
        xi, yi = px[i], py[i]
        a1 = ((xi - x3) * (y2 - y3) - (x2 - x3) * (yi - y3)) / 2.0
        a2 = ((x1 - x3) * (yi - y3) - (xi - x3) * (y1 - y3)) / 2.0
        a3 = ((x1 - xi) * (y2 - yi) - (x2 - xi) * (y1 - yi)) / 2.0
        hits = np.nonzero((a1 >= 0) & (a2 >= 0) & (a3 >= 0))[0]
        if hits.size:
            t = int(hits[0])
            ie[i] = parent[t]
            ip[i] = tris[t]
            w1 = a1[t] / a0[t]
            w2 = a2[t] / a0[t]
            acor[i] = (w1, w2, 1.0 - w1 - w2)
        else:
            n = int(np.argmin((node_x - xi) ** 2 + (node_y - yi) ** 2))
            ip[i] = n
            acor[i] = (1.0, 0.0, 0.0)
    return ie, ip, acor


def _read_elements(
    hgrid_path: Path, n_nodes: int, n_elements: int
) -> np.ndarray:
    """0-based (ne, 4) connectivity with -1 padding.

    ``SchismGrid.read`` skips the element block, so re-walk the file
    past the header and node table.
    """
    elnode = np.full((n_elements, 4), -1, dtype=int)
    with open(hgrid_path) as f:
        f.readline()
        f.readline()
        for _ in range(n_nodes):
            f.readline()
        for i in range(n_elements):
            tok = f.readline().split()
            i34 = int(tok[1])
            elnode[i, :i34] = [int(t) - 1 for t in tok[2:2 + i34]]
    return elnode


def _read_nvrt(vgrid_path: Path) -> int:
    """``nvrt`` from vgrid.in -- mirrors pylib ``read_vgrid``'s header.

    Both formats (ivcor=1 LSC2 and ivcor=2 SZ) carry ivcor on line 1
    and nvrt as the first token of line 2; that is all this product
    consumes (the ``siglay`` dimension).
    """
    with open(vgrid_path) as f:
        ivcor = int(f.readline().split()[0])
        nvrt = int(f.readline().split()[0])
    if ivcor not in (1, 2):
        raise ValueError(
            f"unrecognized vgrid.in (ivcor={ivcor}): {vgrid_path}"
        )
    return nvrt


def _format_base_date(base_date: str) -> str:
    """Ops time-units string from a date string.

    Accepts the ops CLI form ``YYYY-MM-DD-HH`` plus ISO-ish variants
    (``YYYY-MM-DD HH[:MM[:SS]]``, ``T`` separator); minutes/seconds are
    dropped exactly like ops, which formats only year/month/day/hour.
    """
    m = re.match(
        r"\s*(\d{4})-(\d{1,2})-(\d{1,2})[-T ](\d{1,2})", base_date
    )
    if not m:
        raise ValueError(
            f"base_date {base_date!r} not of the form YYYY-MM-DD-HH"
        )
    y, mo, d, h = (int(g) for g in m.groups())
    return f"{y}-{mo:02d}-{d:02d} {h:02d}:00:00 UTC"


def write_station_profiles(
    stacks: Sequence[Mapping[str, Path]],
    hgrid_path: Path,
    vgrid_path: Path,
    out_path: Path,
    base_date: str,
    station_file: Optional[Path] = None,
    lons: Optional[Sequence[float]] = None,
    lats: Optional[Sequence[float]] = None,
    names: Optional[Sequence[str]] = None,
    outside: str = "nearest",
) -> Path:
    """Interpolate stack outputs to stations and write the profile nc.

    ``stacks`` is a chronological list of per-stack file mappings with
    keys ``out2d`` + the :data:`VARS_3D` names (see
    :func:`stack_inputs`); their times are concatenated in-code.
    Stations come from ``station_file`` (ops ``station.in``, see
    :func:`read_station_in`) or explicit ``lons``/``lats`` (+ optional
    ``names``, defaulting to 1-based station numbers). ``base_date``
    stamps the time units (ops passes the nowcast/forecast begin as
    ``YYYY-MM-DD-HH``). ``outside="nearest"`` keeps pylib's
    nearest-node fallback for out-of-mesh stations; ``"error"``
    replicates the ops driver's abort.

    Deviations from ops, all documented here: ``windSpeedX/Y`` missing
    from an out2d stack write fill values instead of crashing; masked
    inputs are filled with :data:`FILL_VALUE` (ops silently writes the
    raw bytes under the mask); a 3D stack shorter than its out2d raises
    ``ValueError`` instead of ``IndexError``; the ops ``ndims == 4``
    branch (broken upstream: references an undefined variable) is not
    ported. Element-centered (``nSCHISM_hgrid_face``) variables follow
    ops in taking the containing element's value un-interpolated and
    are only meaningful for in-mesh stations.
    """
    from netCDF4 import Dataset

    if not stacks:
        raise ValueError("write_station_profiles: no stack inputs given")
    if outside not in ("nearest", "error"):
        raise ValueError(f"outside must be 'nearest' or 'error': {outside}")

    if station_file is not None:
        if lons is not None or lats is not None:
            raise ValueError(
                "pass station_file OR lons/lats arrays, not both"
            )
        lons, lats, file_names = read_station_in(station_file)
        if names is None:
            names = file_names
    else:
        if lons is None or lats is None:
            raise ValueError(
                "stations required: station_file or lons + lats"
            )
        lons = np.asarray(lons, dtype=float)
        lats = np.asarray(lats, dtype=float)
        if names is None:
            names = np.array([str(i + 1) for i in range(lons.size)])
    nsta = lons.size
    if lats.size != nsta or len(names) != nsta:
        raise ValueError("lons, lats and names lengths differ")

    grid = SchismGrid.read(hgrid_path, read_boundaries=False)
    elnode = _read_elements(hgrid_path, grid.n_nodes, grid.n_elements)
    ie, ip, acor = compute_area_coords(
        grid.node_lons, grid.node_lats, elnode, lons, lats
    )
    out_of_mesh = np.nonzero(ie == -1)[0]
    if outside == "error" and out_of_mesh.size:
        raise ValueError(
            "points outside of domain: "
            f"{np.c_[lons[out_of_mesh], lats[out_of_mesh]]}"
        )
    depth0 = (grid.node_depths[ip] * acor).sum(axis=1)
    nvrt = _read_nvrt(vgrid_path)

    times: List[float] = []
    series2d: Dict[str, List[np.ndarray]] = {v: [] for v in VARS_2D}
    series3d: Dict[str, List[np.ndarray]] = {v: [] for v in VARS_3D}

    for stack in stacks:
        with Dataset(stack["out2d"]) as ds2d:
            t = np.asarray(ds2d.variables["time"][:], dtype=float)
            ntimes = t.size
            times.extend(t.tolist())
            for var in VARS_2D:
                if var not in ds2d.variables:
                    series2d[var].extend(
                        [np.full(nsta, FILL_VALUE)] * ntimes
                    )
                    continue
                for it in range(ntimes):
                    row = (ds2d.variables[var][it][ip] * acor).sum(axis=1)
                    series2d[var].append(np.ma.filled(row, FILL_VALUE))

        for var in VARS_3D:
            with Dataset(stack[var]) as ds3d:
                v = ds3d.variables[var]
                dims = v.dimensions
                if v.shape[0] < ntimes:
                    raise ValueError(
                        f"{stack[var]}: {v.shape[0]} times < "
                        f"{ntimes} in {stack['out2d']}"
                    )
                for it in range(ntimes):
                    if "nSCHISM_hgrid_node" in dims:
                        if v.ndim != 3:
                            raise ValueError(
                                f"unsupported rank {v.ndim} for {var}"
                            )
                        row = (v[it][ip] * acor[..., None]).sum(axis=1)
                    elif "nSCHISM_hgrid_face" in dims:
                        row = v[it][ie]
                    else:
                        raise ValueError(
                            f"unknown variable format: {var} {dims}"
                        )
                    series3d[var].append(np.ma.filled(row, FILL_VALUE))

    if not times:
        raise ValueError("write_station_profiles: all stacks were empty")

    base = _format_base_date(base_date)
    with Dataset(out_path, "w", format="NETCDF3_CLASSIC") as fout:
        fout.createDimension("station", nsta)
        fout.createDimension("namelen", NAMELEN)
        fout.createDimension("siglay", nvrt)
        fout.createDimension("time", None)

        tv = fout.createVariable("time", "f4", ("time",))
        tv.long_name = "Time"
        tv.units = f"seconds since {base}"
        tv.base_date = base
        tv.standard_name = "time"
        tv[:] = times

        nv = fout.createVariable("station_name", "c", ("station", "namelen"))
        nv.long_name = "station name"
        # null-padded char matrix; same bytes the ops stringtochar(S100)
        # path writes, but built with plain numpy (netCDF4 1.7's
        # stringtochar breaks on S-dtype input under numpy 2)
        name_arr = np.zeros((nsta, NAMELEN), dtype="S1")
        for i in range(nsta):
            b = str(names[i]).encode("ascii")[:NAMELEN]
            name_arr[i, :len(b)] = np.frombuffer(b, dtype="S1")
        nv[:] = name_arr

        lonv = fout.createVariable("lon", "f4", ("station",))
        lonv.long_name = "longitude"
        lonv.standard_name = "longitude"
        lonv.units = "degrees_east"
        lonv.positive = "east"
        lonv[:] = lons

        latv = fout.createVariable("lat", "f4", ("station",))
        latv.long_name = "latitude"
        latv.standard_name = "latitude"
        latv.units = "degrees_north"
        latv.positive = "north"
        latv[:] = lats

        dv = fout.createVariable("depth", "f4", ("station",))
        dv.long_name = "Bathymetry"
        dv.standard_name = "depth"
        dv.units = "meters"
        dv[:] = depth0

        zv = fout.createVariable(
            "zeta", "f4", ("time", "station"), fill_value=FILL_VALUE
        )
        zv.long_name = "water surface elevation above navd88"
        zv.standard_name = "sea_surface_height_above_navd88"
        zv.units = "m"
        zv[:, :] = np.asarray(series2d["elevation"])

        zc = fout.createVariable(
            "zCoordinates", "f4", ("time", "station", "siglay"),
            fill_value=FILL_VALUE,
        )
        zc.long_name = "vertical coordinate, positive upward"
        zc.standard_name = "vertical coordinate"
        zc.units = "m"
        zc[:, :, :] = np.asarray(series3d["zCoordinates"])

        sv = fout.createVariable(
            "salinity", "f4", ("time", "station", "siglay"),
            fill_value=FILL_VALUE,
        )
        sv.long_name = "salinity"
        sv.standard_name = "sea_water_salinity"
        sv.units = "psu"
        sv[:, :, :] = np.asarray(series3d["salinity"])

        tev = fout.createVariable(
            "temperature", "f4", ("time", "station", "siglay"),
            fill_value=FILL_VALUE,
        )
        tev.long_name = "temperature"
        tev.standard_name = "sea_water_temperature"
        tev.units = "degree_C"
        tev[:, :, :] = np.asarray(series3d["temperature"])

        uv = fout.createVariable(
            "u", "f4", ("time", "station", "siglay"), fill_value=FILL_VALUE
        )
        uv.long_name = "Eastward Water Velocity"
        uv.standard_name = "eastward_sea_water_velocity"
        uv.units = "meters s-1"
        uv[:, :, :] = np.asarray(series3d["horizontalVelX"])

        vv = fout.createVariable(
            "v", "f4", ("time", "station", "siglay"), fill_value=FILL_VALUE
        )
        vv.long_name = "Northward Water Velocity"
        vv.standard_name = "northward_sea_water_velocity"
        vv.units = "meters s-1"
        vv[:, :, :] = np.asarray(series3d["horizontalVelY"])

        uw = fout.createVariable(
            "uwind_speed", "f4", ("time", "station"), fill_value=FILL_VALUE
        )
        uw.long_name = "Eastward Wind Velocity"
        uw.standard_name = "eastward_wind"
        uw.units = "meters s-1"
        uw[:, :] = np.asarray(series2d["windSpeedX"])

        vw = fout.createVariable(
            "vwind_speed", "f4", ("time", "station"), fill_value=FILL_VALUE
        )
        vw.long_name = "Northward Wind Velocity"
        vw.standard_name = "northward_wind"
        vw.units = "meters s-1"
        vw[:, :] = np.asarray(series2d["windSpeedY"])

        fout.title = "SCHISM Model output"
        fout.references = "http://ccrm.vims.edu/schismweb/"

    return Path(out_path)


__all__ = [
    "FILL_VALUE",
    "NAMELEN",
    "VARS_2D",
    "VARS_3D",
    "compute_area_coords",
    "read_station_in",
    "stack_inputs",
    "write_station_profiles",
]
