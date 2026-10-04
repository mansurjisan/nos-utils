"""2D-slab field products (``field2d`` ADCIRC-named, ``2ds`` STOFS-named).

Port of the operational STOFS-3D-ATL slab extractor
``extract_slab_fcst_netcdf4.py`` (IT-STOFS ``ush/stofs_3d_atl/pysh``,
run once per output stack by ``stofs_3d_atl_create_2d_field_nc.sh``):
surface and bottom slices of temperature/salinity/velocity plus
velocity interpolated at fixed depths below the free surface. Semantics
follow ops exactly: per-record dry nodes (inundation <= 1e-6 m) and
magnitudes > 10000 carry ``FILL_VALUE``; T/S "bottom" is the bottom
level ``kbp`` while velocity "bottom" is one level above it (no-slip
zero at ``kbp``); fixed-depth values interpolate the ``zCoordinates``
column with surface/bottom clamping. Domain knobs (paths, depths,
datum, element table, bottom indices, base date) are explicit
arguments -- no CWD reads, no hardcoded mesh dims or cycle strings.

:func:`compute_slab_fields` does the shared computation once per stack;
:func:`write_slab2d` writes it on the quad-split triangle mesh in the
ADCIRC-flavoured schema (``zeta``/``x``/``y``/``element``, one file per
stack, ops calls these ``field2d_*``); :func:`write_2ds_record` writes
one record of the same computed fields in the STOFS-3D-ATL schema
(``elev``/``SCHISM_hgrid_node_x``/``SCHISM_hgrid_face_nodes``, one file
per hour, ops calls these ``schout_2d_*``) -- both read off the same
:class:`SlabFields`, so the physics/masking is never duplicated between
the two output shapes.

Note: the ``element``/``SCHISM_hgrid_face_nodes`` table uses the shared
``mesh.split_quads`` diagonal ((a,b,c)+(a,c,d)); the ops adcirc/slab
sources split quads as (a,b,d)+(b,c,d) -- an equivalent triangulation
that differs byte-wise on quad meshes (node-based values are
unaffected).
"""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from nos_utils.post.mesh import split_quads

FILL_VALUE = -99999.0
LARGE_VALUE_THRESHOLD = 10000.0
DRY_THRESHOLD = 1e-6

_SOURCE_VARS = {
    "temperature": "temperature",
    "salinity": "salinity",
    "u": "horizontalVelX",
    "v": "horizontalVelY",
}

# (output name, source key, long_name, units, level kind) in ops order.
_NATIVE_SPECS = (
    ("temp_surface", "temperature", "sea surface temperature", "deg C",
     "surface"),
    ("temp_bottom", "temperature", "Bottom temperature", "deg C", "bottom"),
    ("salt_surface", "salinity", "sea surface salinity", "psu", "surface"),
    ("salt_bottom", "salinity", "Bottom salinity", "psu", "bottom"),
    ("uvel_surface", "u", "U-component at the surface", "m/s", "surface"),
    ("uvel_bottom", "u", "U-component at the bottom", "m/s", "near_bottom"),
    ("vvel_surface", "v", "V-component at the surface", "m/s", "surface"),
    ("vvel_bottom", "v", "V-component at the bottom", "m/s", "near_bottom"),
)


@dataclass
class SlabFields:
    """The full-stack slab computation, independent of output schema.

    ``elev2d``/``idry`` are (ntimes, n_points); ``native``/``interp``
    values are (ntimes, n_points) arrays already dry- and junk-masked.
    ``tris`` carries whatever node numbering ``elements`` used (ops: 1
    based). Both writers in this module read off one instance of this
    rather than recomputing.
    """

    x: np.ndarray
    y: np.ndarray
    depth: np.ndarray
    elev2d: np.ndarray
    idry: np.ndarray
    tris: np.ndarray
    times: np.ndarray
    ntimes: int
    n_points: int
    depths: Tuple[float, ...]
    native: Dict[str, np.ndarray]
    interp: Dict[str, np.ndarray]


def _interp_weights(zcor, zinter, bottom_index):
    """Lower-level index and lower-level weight per node column.

    Faithful to ops ``vertical_interp``: surface clamp, bottom clamp,
    then the bracketing-interval scan (which deliberately overrides the
    clamps where an interval matches, as in ops).
    """
    n_points, nvrt = zcor.shape
    rows = np.arange(n_points)
    level = np.zeros(n_points, dtype=int)
    weight = np.zeros(n_points, dtype=float)

    above = zinter >= zcor[:, -1]
    level[above] = nvrt - 2
    weight[above] = 0.0

    below = zinter < zcor[rows, bottom_index]
    level[below] = bottom_index[below]
    weight[below] = 1.0

    for k in range(nvrt - 1):
        idx = (zinter >= zcor[:, k]) & (zinter < zcor[:, k + 1])
        level[idx] = k
        weight[idx] = (
            (zcor[idx, k + 1] - zinter[idx])
            / (zcor[idx, k + 1] - zcor[idx, k])
        )

    if np.isnan(weight).any():
        raise ValueError("NaN values in vertical interpolation weights")
    return level, weight


def _mask_var(data, idry):
    """Ops ``mask_var``: fill dry nodes, then junk magnitudes."""
    data[idry] = FILL_VALUE
    data[data > LARGE_VALUE_THRESHOLD] = FILL_VALUE
    return data


def _slab_var_specs(depths: Sequence[float]):
    """(name, long_name, units) for every slab var, native then interpolated.

    Shared by both writers so a variable's name/long_name/units cannot
    drift between the ADCIRC-named and STOFS-named outputs -- ops uses
    identical field names in both (only the mesh/coordinate names and
    dims differ).
    """
    specs = [(n, ln, u) for n, _s, ln, u, _k in _NATIVE_SPECS]
    for d in depths:
        tag = f"{d:g}"
        specs.append(
            (f"uvel{tag}", f"U-component at {tag}m below free surface",
             "m/s")
        )
        specs.append(
            (f"vvel{tag}", f"V-component at {tag}m below free surface",
             "m/s")
        )
    return specs


def compute_slab_fields(
    out2d_path: Path,
    zcor_path: Path,
    temperature_path: Path,
    salinity_path: Path,
    uvel_path: Path,
    vvel_path: Path,
    depths: Sequence[float] = (0.5, 4.5),
    elements: Optional[np.ndarray] = None,
    bottom_index_node: Optional[np.ndarray] = None,
) -> SlabFields:
    """Compute the 2D slab fields for one stack (no I/O beyond reading).

    The six input paths are one stack of canonical scribe-shaped files
    (``out2d`` with ``time``/``elevation``/coords/``depth``, and
    ``(time, node, layer)`` stacks for zCoordinates, temperature,
    salinity, horizontalVelX/Y). ``depths`` are positive below-free-
    surface distances; each yields ``uvel{d}``/``vvel{d}`` in
    ``.interp``. ``elements`` (1-based, -1/masked padding) and
    ``bottom_index_node`` (1-based bottom level per node) default to
    the out2d variables ``SCHISM_hgrid_face_nodes`` and
    ``bottom_index_node`` as in ops.

    The dry mask reproduces the ``field2d``/``2ds`` production
    convention: per-record ``elev + depth <= 1e-6``, not the per-stack
    max-elevation mask ``extract_slab_fcst_netcdf4.py`` itself computes
    -- the two only differ on nodes that dry partway through a
    multi-record stack, and per-record is what is already deployed and
    tested here.
    """
    from netCDF4 import Dataset

    depths = tuple(float(d) for d in depths)
    if any(d <= 0 for d in depths):
        raise ValueError(
            "slab2d: depths are positive distances below the free surface"
        )

    with Dataset(out2d_path, "r") as ds:
        x = ds.variables["SCHISM_hgrid_node_x"][:]
        y = ds.variables["SCHISM_hgrid_node_y"][:]
        depth = np.array(ds.variables["depth"][:])
        elev2d = np.array(ds.variables["elevation"][:])
        times = ds.variables["time"][:]
        if elements is None:
            if "SCHISM_hgrid_face_nodes" not in ds.variables:
                raise ValueError(
                    "slab2d: pass elements= or provide "
                    "SCHISM_hgrid_face_nodes in the out2d stack"
                )
            elements = ds.variables["SCHISM_hgrid_face_nodes"][:, :]
        if bottom_index_node is None:
            if "bottom_index_node" not in ds.variables:
                raise ValueError(
                    "slab2d: pass bottom_index_node= or provide "
                    "bottom_index_node in the out2d stack"
                )
            bottom_index_node = ds.variables["bottom_index_node"][:]

    n_points = len(x)
    ntimes = len(times)
    rows = np.arange(n_points)
    bottom_index = np.asarray(bottom_index_node).astype(int) - 1

    idry = elev2d + depth.reshape(1, -1) <= DRY_THRESHOLD
    elev2d[idry] = FILL_VALUE

    tris = split_quads(elements)

    native: Dict[str, np.ndarray] = {}
    interp: Dict[str, np.ndarray] = {
        f"{p}{d:g}": np.empty((ntimes, n_points))
        for d in depths for p in ("uvel", "vvel")
    }

    with ExitStack() as ctx:
        src = {
            key: ctx.enter_context(Dataset(path, "r"))
            for key, path in (
                ("temperature", temperature_path),
                ("salinity", salinity_path),
                ("u", uvel_path),
                ("v", vvel_path),
            )
        }

        # Native slabs: whole-stack reads per source, like ops
        # (non-mem-save mode; slices copied so the dry/junk fills of one
        # slab never alias into the shared 3D buffer).
        for src_key in ("temperature", "salinity", "u", "v"):
            data_3d = np.array(
                src[src_key].variables[_SOURCE_VARS[src_key]][:]
            )
            for name, spec_src, _long, _units, kind in _NATIVE_SPECS:
                if spec_src != src_key:
                    continue
                if kind == "surface":
                    data = np.array(data_3d[:, :, -1])
                elif kind == "bottom":
                    data = data_3d[:, rows, bottom_index]
                else:
                    data = data_3d[:, rows, bottom_index + 1]
                native[name] = _mask_var(data, idry)
            del data_3d

        # Fixed-depth slabs: per-record reads, weights computed once per
        # (record, depth) and shared by u/v, like the ops weight cache.
        zvar = ctx.enter_context(
            Dataset(zcor_path, "r")
        ).variables["zCoordinates"]
        uvar = src["u"].variables["horizontalVelX"]
        vvar = src["v"].variables["horizontalVelY"]
        for it in range(ntimes):
            zcor = np.array(zvar[it, :, :])
            u_it = np.array(uvar[it, :, :])
            v_it = np.array(vvar[it, :, :])
            for d in depths:
                # ops: zinter = level + elev, elev already fill-masked
                zinter = -d + elev2d[it, :]
                level, weight = _interp_weights(zcor, zinter, bottom_index)
                for prefix, var_it in (("uvel", u_it), ("vvel", v_it)):
                    data = (
                        var_it[rows, level] * weight
                        + var_it[rows, level + 1] * (1.0 - weight)
                    )
                    interp[f"{prefix}{d:g}"][it, :] = _mask_var(
                        data, idry[it, :]
                    )

    return SlabFields(
        x=x, y=y, depth=depth, elev2d=elev2d, idry=idry, tris=tris,
        times=times, ntimes=ntimes, n_points=n_points, depths=depths,
        native=native, interp=interp,
    )


def write_slab2d(
    out2d_path: Path,
    zcor_path: Path,
    temperature_path: Path,
    salinity_path: Path,
    uvel_path: Path,
    vvel_path: Path,
    out_path: Path,
    base_date: str,
    depths: Sequence[float] = (0.5, 4.5),
    elements: Optional[np.ndarray] = None,
    bottom_index_node: Optional[np.ndarray] = None,
    datum: str = "xgeoid20b",
) -> Path:
    """Write the field2d nc for one stack, ADCIRC-flavoured schema.

    ``base_date`` stamps the time units verbatim (ops derives the same
    string from param.nml-driven attrs). ``datum`` brands the ``depth``
    and ``zeta`` datum attributes (ops: xgeoid20b). See
    :func:`compute_slab_fields` for the shared field computation this
    delegates to.
    """
    from netCDF4 import Dataset

    fields = compute_slab_fields(
        out2d_path, zcor_path, temperature_path, salinity_path,
        uvel_path, vvel_path, depths=depths, elements=elements,
        bottom_index_node=bottom_index_node,
    )

    with Dataset(out_path, "w", format="NETCDF4") as fout:
        fout.createDimension("time", None)
        fout.createDimension("node", fields.n_points)
        fout.createDimension("nele", len(fields.tris))
        fout.createDimension("nvertex", 3)

        tv = fout.createVariable("time", "f4", ("time",))
        tv.long_name = "Time"
        tv.units = f"seconds since {base_date}"
        tv.base_date = base_date
        tv.standard_name = "time"
        tv[:] = fields.times

        xv = fout.createVariable("x", "f8", ("node",))
        xv.long_name = "node x-coordinate"
        xv.standard_name = "longitude"
        xv.units = "degrees_east"
        xv.positive = "east"
        xv[:] = fields.x

        yv = fout.createVariable("y", "f8", ("node",))
        yv.long_name = "node y-coordinate"
        yv.standard_name = "latitude"
        yv.units = "degrees_north"
        yv.positive = "north"
        yv[:] = fields.y

        ev = fout.createVariable("element", "i4", ("nele", "nvertex"))
        ev.long_name = "element"
        ev.standard_name = "face_node_connectivity"
        ev.start_index = 1
        ev.units = "nondimensional"
        ev[:] = np.array(fields.tris)

        dv = fout.createVariable("depth", "f8", ("node",))
        dv.long_name = f"distance below {datum.upper()}"
        dv.standard_name = f"depth below {datum.upper()}"
        dv.coordinates = "time y x"
        dv.location = "node"
        dv.units = "m"
        dv[:] = fields.depth

        zv = fout.createVariable(
            "zeta", "f8", ("time", "node"), fill_value=FILL_VALUE
        )
        zv.standard_name = f"sea_surface_height_above_{datum.lower()}"
        zv.coordinates = "time y x"
        zv.location = "node"
        zv.units = "m"
        zv[:, :] = fields.elev2d

        for name, long_name, units in _slab_var_specs(fields.depths):
            var = fout.createVariable(
                name, "f8", ("time", "node"), fill_value=FILL_VALUE
            )
            var.long_name = long_name
            var.units = units
            var[:, :] = fields.native.get(name, fields.interp.get(name))

        fout.title = "SCHISM Model output"
        fout.source = "SCHISM model output version v10"
        fout.references = "http://ccrm.vims.edu/schismweb/"

    return Path(out_path)


def write_2ds_record(
    fields: SlabFields,
    record: int,
    out_path: Path,
    base_date: str,
) -> Path:
    """Write one hour of ``fields`` in the STOFS-3D-ATL ``2ds`` schema.

    Mirrors ``extract_slab_fcst_netcdf4.py``'s output exactly (dims
    ``nSCHISM_hgrid_node``/``nSCHISM_hgrid_face``/
    ``nMaxSCHISM_hgrid_face_nodes``, vars ``SCHISM_hgrid_node_x/y``,
    ``SCHISM_hgrid_face_nodes``, ``elev``) but with ``time`` holding a
    single record instead of the whole stack, since this product
    publishes one file per forecast/nowcast hour rather than one per
    stack. Field values (temp/salt/uvel/vvel slabs) come straight off
    ``fields`` -- see :func:`compute_slab_fields` for how they, the dry
    mask and the fixed-depth interpolation are computed once and shared
    with :func:`write_slab2d`.
    """
    from netCDF4 import Dataset

    if not 0 <= record < fields.ntimes:
        raise IndexError(
            f"write_2ds_record: record {record} out of range "
            f"(0..{fields.ntimes - 1})"
        )

    with Dataset(out_path, "w", format="NETCDF4") as fout:
        fout.createDimension("time", None)
        fout.createDimension("nSCHISM_hgrid_node", fields.n_points)
        fout.createDimension("nSCHISM_hgrid_face", len(fields.tris))
        fout.createDimension("nMaxSCHISM_hgrid_face_nodes", 3)

        tv = fout.createVariable("time", "f4", ("time",))
        tv.long_name = "Time"
        tv.units = f"seconds since {base_date}"
        tv.base_date = base_date
        tv.standard_name = "time"
        tv[:] = [fields.times[record]]

        xv = fout.createVariable(
            "SCHISM_hgrid_node_x", "f8", ("nSCHISM_hgrid_node",)
        )
        xv.long_name = "node x-coordinate"
        xv.standard_name = "longitude"
        xv.units = "degrees_east"
        xv.mesh = "SCHISM_hgrid"
        xv[:] = fields.x

        yv = fout.createVariable(
            "SCHISM_hgrid_node_y", "f8", ("nSCHISM_hgrid_node",)
        )
        yv.long_name = "node y-coordinate"
        yv.standard_name = "latitude"
        yv.units = "degrees_north"
        yv.mesh = "SCHISM_hgrid"
        yv[:] = fields.y

        ev = fout.createVariable(
            "SCHISM_hgrid_face_nodes", "i4",
            ("nSCHISM_hgrid_face", "nMaxSCHISM_hgrid_face_nodes"),
        )
        ev.long_name = "element"
        ev.standard_name = "face_node_connectivity"
        ev.start_index = 1
        ev.units = "nondimensional"
        ev[:] = np.array(fields.tris)

        dv = fout.createVariable("depth", "f4", ("nSCHISM_hgrid_node",))
        dv.long_name = "bathymetry"
        dv.units = "m"
        dv.mesh = "SCHISM_hgrid"
        dv[:] = fields.depth

        zv = fout.createVariable(
            "elev", "f8", ("time", "nSCHISM_hgrid_node"),
            fill_value=FILL_VALUE,
        )
        zv.long_name = "water elevation"
        zv.units = "m"
        zv.mesh = "SCHISM_hgrid"
        zv[:, :] = fields.elev2d[record:record + 1, :]

        for name, long_name, units in _slab_var_specs(fields.depths):
            var = fout.createVariable(
                name, "f8", ("time", "nSCHISM_hgrid_node"),
                fill_value=FILL_VALUE,
            )
            var.long_name = long_name
            var.units = units
            data = fields.native.get(name, fields.interp.get(name))
            var[:, :] = data[record:record + 1, :]

        fout.title = "SCHISM Model output"
        fout.source = "SCHISM model output version v10"
        fout.references = "http://ccrm.vims.edu/schismweb/"

    return Path(out_path)


__all__ = [
    "DRY_THRESHOLD", "FILL_VALUE", "LARGE_VALUE_THRESHOLD",
    "SlabFields", "compute_slab_fields", "write_2ds_record", "write_slab2d",
]
