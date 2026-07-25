"""ADCIRC-format water-level field product (``schout_adcirc_*.nc``).

Port of the current operational STOFS-3D-ATL ``pysh/generate_adcirc.py``
(STOFS-operational-main lineage; the IT-STOFS copy is an older lineage
of the same script). Driven in ops by
``stofs_3d_atl_create_adcirc_nc.sh``, feeding CERA: reads scribe-shaped
``out2d_*`` stacks and writes the ADCIRC-format file -- ``zeta(time,
node)`` plus per-node reductions ``zeta_max`` / ``time_of_zeta_max`` /
``disturbance_max``, ``uwind``/``vwind`` when the stacks carry
``windSpeedX/Y``, the triangulated element table, and coords/depth.

Ops-faithful masking: junk values (above ``JUNK_THRESHOLD``) and dry
records (``dryFlagNode == 1``) become ``FILL_VALUE`` in ``zeta`` before
any reduction, so an always-dry node carries a filled ``zeta_max``;
``disturbance_max`` starts as the max elevation and on normally-dry
nodes (land, ``depth < 0``, or city) becomes the clamped inundation
depth ``max(0, zeta_max + depth)`` (an always-dry land/city node clamps
to 0), then is filled where below ``min_disturbance`` on land or city
nodes (small ocean disturbances are kept -- ops has that branch
commented out); ``time_of_zeta_max`` is never filled; both wind
components are filled where ``uwind`` is junk (ops derives the mask
from uwind only).

Fallback: when the stacks carry no ``dryFlagNode`` (e.g. re-split
canonical stacks), the older-lineage never-wet fill is applied to
``zeta_max`` instead (``zeta_max + depth <= DRY_TOLERANCE`` ->
``FILL_VALUE``); current ops comments that fill out as redundant once
dry records are filled via ``dryFlagNode``.

Deviations from ops: the city mask is a precomputed boolean argument
(see :func:`city_mask_from_polygons`) instead of a shapefile /
node-id-file read; quads split via the shared
:func:`nos_utils.post.mesh.split_quads` convention ((a,b,c)+(a,c,d)
appended at the end) whereas the current-ops vectorized split replaces
each quad with (a,b,d) in place and appends (b,c,d) at the end (both
valid triangulations, different diagonal and row order); and with
several stacks the max reductions span the whole window, unlike the ops
``ncrcat`` pair merge which keeps only the first stack's non-record
variables.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from nos_utils.post.mesh import split_quads

FILL_VALUE = -99999.0
JUNK_THRESHOLD = 100000.0
DRY_TOLERANCE = 1.0e-6


def write_adcirc(
    out2d_files: Sequence[Path],
    out_path: Path,
    base_date: str,
    datum: str = "xGEOID20B",
    city_mask: Optional[np.ndarray] = None,
    elnode: Optional[np.ndarray] = None,
    min_disturbance: float = 0.3,
    time_units: Optional[str] = None,
) -> Path:
    """Reduce out2d stacks into one ADCIRC-format ``schout_adcirc`` nc.

    ``out2d_files`` are scribe-shaped 2D stacks (``time``,
    ``elevation(time, node)``, node coords + ``depth``, optionally
    ``SCHISM_hgrid_face_nodes``, ``dryFlagNode`` and ``windSpeedX/Y``)
    in chronological order -- ops processes one half-day stack per call
    and pair-merges; here the pair is passed directly. ``base_date``
    stamps the time attrs (``time_units`` overrides the default
    ``seconds since {base_date}``, e.g. to carry the input stack's units
    verbatim). ``datum`` is spliced verbatim into the vertical-datum
    attr strings (ops default ``xGEOID20B``). ``city_mask`` is an
    optional (node,) boolean array, True inside urban polygons, used in
    the inundation conversion and the small-disturbance fill; ops loads
    the equivalent from ``stofs_3d_atl_node_id_city_poly_adcirc.txt``
    and uses an all-False mask when no city file is given. ``elnode``
    overrides the element table ((ne, 3|4), 1-based, -1 padding) when
    the stacks do not carry ``SCHISM_hgrid_face_nodes``.
    """
    from netCDF4 import Dataset

    if not out2d_files:
        raise ValueError("write_adcirc: no out2d files given")
    if time_units is None:
        time_units = f"seconds since {base_date}"

    time_parts = []
    elev_parts = []
    dry_parts = []
    uwind_parts = []
    vwind_parts = []
    x = y = depth = None
    face_nodes = None
    have_dry: Optional[bool] = None
    have_wind: Optional[bool] = None

    for path in out2d_files:
        with Dataset(path, "r") as ds:
            t = np.asarray(ds.variables["time"][:], dtype="f8")
            if t.size == 0:
                continue
            time_parts.append(t)
            elev_parts.append(
                np.ma.filled(
                    ds.variables["elevation"][:], FILL_VALUE
                ).astype("f8")
            )
            if have_dry is None:
                have_dry = "dryFlagNode" in ds.variables
            if have_dry:
                dry_parts.append(
                    np.ma.filled(ds.variables["dryFlagNode"][:], 0)
                )
            if have_wind is None:
                have_wind = (
                    "windSpeedX" in ds.variables
                    and "windSpeedY" in ds.variables
                )
            if have_wind:
                uwind_parts.append(
                    np.ma.filled(
                        ds.variables["windSpeedX"][:], FILL_VALUE
                    ).astype("f8")
                )
                vwind_parts.append(
                    np.ma.filled(
                        ds.variables["windSpeedY"][:], FILL_VALUE
                    ).astype("f8")
                )
            if x is None and "SCHISM_hgrid_node_x" in ds.variables:
                x = np.asarray(
                    ds.variables["SCHISM_hgrid_node_x"][:], dtype="f8"
                )
                y = np.asarray(
                    ds.variables["SCHISM_hgrid_node_y"][:], dtype="f8"
                )
            if depth is None and "depth" in ds.variables:
                depth = np.ma.filled(
                    ds.variables["depth"][:], FILL_VALUE
                ).astype("f8")
            if (
                elnode is None
                and face_nodes is None
                and "SCHISM_hgrid_face_nodes" in ds.variables
            ):
                face_nodes = np.ma.filled(
                    ds.variables["SCHISM_hgrid_face_nodes"][:], -1
                )

    if not time_parts:
        raise ValueError("write_adcirc: all out2d stacks were empty")
    if x is None or depth is None:
        raise ValueError(
            "write_adcirc: out2d stacks carry no node coords/depth"
        )
    table = np.asarray(elnode) if elnode is not None else face_nodes
    if table is None:
        raise ValueError(
            "write_adcirc: no SCHISM_hgrid_face_nodes in the stacks;"
            " pass elnode"
        )
    tris = split_quads(table)

    times = np.concatenate(time_parts)
    elev = np.concatenate(elev_parts, axis=0)
    elev[elev > JUNK_THRESHOLD] = FILL_VALUE
    if have_dry:
        # Current ops: fill dry records before any reduction.
        dry_flag = np.concatenate(dry_parts, axis=0)
        elev[dry_flag == 1] = FILL_VALUE
    n_nodes = elev.shape[1]

    if city_mask is None:
        city = np.zeros(n_nodes, dtype=bool)
    else:
        city = np.asarray(city_mask, dtype=bool)
        if city.shape != (n_nodes,):
            raise ValueError(
                f"write_adcirc: city_mask shape {city.shape} does not"
                f" match ({n_nodes},)"
            )

    zeta_max = elev.max(axis=0)
    time_of_max = times[elev.argmax(axis=0)]

    # Disturbance: max elevation, converted on normally-dry (land or
    # city) nodes to the clamped inundation depth above ground; an
    # always-dry node there (zeta_max == FILL_VALUE) clamps to 0.
    dist_max = zeta_max.copy()
    land = depth < 0
    normally_dry = land | city
    dist_max[normally_dry] = np.maximum(
        0.0, zeta_max[normally_dry] + depth[normally_dry]
    )

    if not have_dry:
        # Older-lineage fallback for stacks without dryFlagNode: fill
        # never-wet nodes explicitly (current ops comments this out as
        # redundant with the per-record dry fill).
        zeta_max[(zeta_max + depth) <= DRY_TOLERANCE] = FILL_VALUE

    small = dist_max < min_disturbance
    dist_max[small & normally_dry] = FILL_VALUE

    if have_wind:
        uwind = np.concatenate(uwind_parts, axis=0)
        vwind = np.concatenate(vwind_parts, axis=0)
        # Ops fills BOTH components where uwind is junk.
        bad = uwind > JUNK_THRESHOLD
        uwind[bad] = FILL_VALUE
        vwind[bad] = FILL_VALUE

    with Dataset(out_path, "w", format="NETCDF4") as ds:
        ds.createDimension("time", None)
        ds.createDimension("node", n_nodes)
        ds.createDimension("nele", tris.shape[0])
        ds.createDimension("nvertex", 3)

        tv = ds.createVariable("time", "f8", ("time",))
        tv.long_name = "Time"
        tv.base_date = base_date
        tv.standard_name = "time"
        tv.units = time_units
        tv[:] = times

        xv = ds.createVariable("x", "f8", ("node",))
        xv.long_name = "node x-coordinate"
        xv.standard_name = "longitude"
        xv.units = "degrees_east"
        xv.positive = "east"
        xv[:] = x

        yv = ds.createVariable("y", "f8", ("node",))
        yv.long_name = "node y-coordinate"
        yv.standard_name = "latitude"
        yv.units = "degrees_north"
        yv.positive = "north"
        yv[:] = y

        ev = ds.createVariable("element", "i4", ("nele", "nvertex"))
        ev.long_name = "element"
        ev.standard_name = "face_node_connectivity"
        ev.start_index = 1
        ev.units = "nondimensional"
        ev[:] = tris

        dv = ds.createVariable("depth", "f8", ("node",))
        dv.long_name = f"distance below {datum}"
        dv.standard_name = f"depth below {datum}"
        dv.coordinates = "y x"
        dv.location = "node"
        dv.units = "m"
        dv[:] = depth

        zm = ds.createVariable(
            "zeta_max", "f8", ("node",), fill_value=FILL_VALUE
        )
        zm.standard_name = f"maximum_sea_surface_height_above_{datum}"
        zm.coordinates = "y x"
        zm.location = "node"
        zm.units = "m"
        zm[:] = zeta_max

        tm = ds.createVariable(
            "time_of_zeta_max", "f8", ("node",), fill_value=FILL_VALUE
        )
        tm.standard_name = (
            f"time_of_maximum_sea_surface_height_above_{datum}"
        )
        tm.coordinates = "y x"
        tm.location = "node"
        tm.units = "sec"
        tm[:] = time_of_max

        dm = ds.createVariable(
            "disturbance_max", "f8", ("node",), fill_value=FILL_VALUE
        )
        # Ops-verbatim standard_name, "depature" typo included.
        dm.standard_name = "maximum_depature_from_initial_condition"
        dm.coordinates = "y x"
        dm.location = "node"
        dm.units = "m"
        dm[:] = dist_max

        zv = ds.createVariable(
            "zeta", "f8", ("time", "node"), fill_value=FILL_VALUE
        )
        zv.standard_name = f"sea_surface_height_above_{datum}"
        zv.coordinates = "time y x"
        zv.location = "node"
        zv.units = "m"
        zv[:, :] = elev

        if have_wind:
            uv = ds.createVariable(
                "uwind", "f8", ("time", "node"), fill_value=FILL_VALUE
            )
            uv.long_name = "10m_above_ground/UGRD"
            uv.standard_name = "eastward_wind"
            uv.coordinates = "time y x"
            uv.location = "node"
            uv.units = "ms-1"
            uv[:, :] = uwind

            vv = ds.createVariable(
                "vwind", "f8", ("time", "node"), fill_value=FILL_VALUE
            )
            vv.long_name = "10m_above_ground/VGRD"
            vv.standard_name = "northward_wind"
            vv.coordinates = "time y x"
            vv.location = "node"
            vv.units = "ms-1"
            vv[:, :] = vwind

        ds.title = "SCHISM Model output"
        ds.source = "SCHISM model output version v10"
        ds.references = "http://ccrm.vims.edu/schismweb/"

    return Path(out_path)


def city_mask_from_polygons(x, y, polygon_source) -> np.ndarray:
    """Boolean (node,) mask: True for nodes inside any polygon.

    Optional-dependency replacement for the ops shapefile search
    (``find_points_in_polyshp``); the result is what ops caches in
    ``stofs_3d_atl_node_id_city_poly_adcirc.txt`` and feeds to
    ``write_adcirc`` as ``city_mask``. ``polygon_source`` is a geopandas
    GeoDataFrame/GeoSeries or anything ``geopandas.read_file`` accepts
    (e.g. the ops ``city_poly.shp``). geopandas/shapely are imported
    lazily -- they are not core deps of nos-utils.
    """
    import geopandas as gpd

    if isinstance(polygon_source, gpd.GeoSeries):
        geoms = polygon_source
    elif isinstance(polygon_source, gpd.GeoDataFrame):
        geoms = polygon_source.geometry
    else:
        geoms = gpd.read_file(polygon_source).geometry

    union = (
        geoms.union_all()
        if hasattr(geoms, "union_all")
        else geoms.unary_union
    )
    pts = gpd.GeoSeries(
        gpd.points_from_xy(np.asarray(x, dtype="f8"),
                           np.asarray(y, dtype="f8"))
    )
    return np.asarray(pts.within(union), dtype=bool)


__all__ = ["FILL_VALUE", "city_mask_from_polygons", "write_adcirc"]
