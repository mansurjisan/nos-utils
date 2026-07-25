"""Hourly disturbance GeoPackage product (``*.disturbance.*.gpkg``).

Port of the current operational ``pysh/gen_geojson.py`` generator
(STOFS-operational ``stofs_3d_atl``; the IT-STOFS copy is an older
lineage) driven by ``stofs_3d_atl_create_geopackage.sh`` (nowCOAST
feed). Per timestep the node disturbance -- ``max(0, elevation +
depth)`` on land nodes (``depth < 0``), plain ``elevation`` on water
nodes -- is masked (dry where ``elevation + depth <= 1e-6``; land
where disturbance < ``small_disturbance_threshold``, -5 in current ops
so it effectively never fires), tri-contoured over the split-quad
mesh, and each filled contour band becomes one row of an EPSG:4326
GeoPackage layer ``disturbance`` carrying minWaterLevel/maxWaterLevel/
verticalDatum/units/rgba. File naming is a caller-provided callback
instead of the ops' hardcoded
``stofs_3d_atl.t12z.disturbance.{n|f}NNN.gpkg`` (cyc=12) list.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from nos_utils.post.mesh import split_quads

FILL_VALUE = -99999.0
DRY_TOLERANCE = 1.0e-6
# Land nodes with disturbance below this are masked. Current ops uses
# -5 (never fires: land disturbance is clamped >= 0, matching the -5
# bottom contour level); the legacy operational value was 0.3.
SMALL_DISTURBANCE_THRESHOLD = -5.0
VERTICAL_DATUM = "XGEOID20B"
UNITS = "meters"
DEFAULT_LEVELS: Tuple[float, ...] = (
    -5.0,
    *(round(-0.5 + 0.1 * i, 1) for i in range(26)),
    20.0,
)

# Ops masks triangles via `disturbance < -90000`, so fill_value must
# stay below this threshold; junk elevations (> 1e5) are pre-filled.
_MASK_THRESHOLD = -90000.0
_JUNK_ELEVATION = 100000.0


def disturbance_field(
    elevation: np.ndarray,
    depth: np.ndarray,
    fill_value: float = FILL_VALUE,
    small_disturbance_threshold: float = SMALL_DISTURBANCE_THRESHOLD,
) -> np.ndarray:
    """Ops disturbance + masking for one timestep of node elevations.

    Land nodes (``depth < 0``) carry ``max(0, elevation + depth)``,
    water nodes plain ``elevation``. Dry nodes (``elevation + depth <=
    1e-6``) and land nodes with disturbance below
    ``small_disturbance_threshold`` are set to ``fill_value`` (keep it
    below -90000 for the triangle mask). The default -5 matches current
    ops, where the small-on-land mask is a no-op; the legacy
    operational threshold was 0.3.
    """
    elevation = np.asarray(elevation, dtype=float)
    depth = np.asarray(depth, dtype=float)
    disturbance = elevation.copy()
    land = depth < 0
    disturbance[land] = np.maximum(0.0, elevation[land] + depth[land])
    disturbance[elevation + depth <= DRY_TOLERANCE] = fill_value
    disturbance[
        (disturbance < small_disturbance_threshold) & land
    ] = fill_value
    return disturbance


def _band_paths(contour) -> List[list]:
    """Per-band path lists across the matplotlib contour API drift.

    matplotlib < 3.8 exposes one PathCollection per filled band via
    ``ContourSet.collections``; since 3.8 the ContourSet is itself a
    single Collection holding one (possibly vertex-less) compound path
    per band, and ``collections`` was removed in 3.10.
    """
    import matplotlib

    version = tuple(
        int(part) for part in matplotlib.__version__.split(".")[:2]
    )
    if version < (3, 8):
        return [list(coll.get_paths()) for coll in contour.collections]
    return [[path] for path in contour.get_paths()]


def _jet_cmap():
    import matplotlib

    try:
        return matplotlib.colormaps["jet"]
    except AttributeError:  # matplotlib < 3.5
        from matplotlib import cm

        return cm.get_cmap("jet")


def write_disturbance_gpkg(
    elevation: np.ndarray,
    depth: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    elnode: np.ndarray,
    out_path: Path,
    levels: Sequence[float] = DEFAULT_LEVELS,
    fill_value: float = FILL_VALUE,
    small_disturbance_threshold: float = SMALL_DISTURBANCE_THRESHOLD,
    vertical_datum: str = VERTICAL_DATUM,
    units: str = UNITS,
    layer: str = "disturbance",
) -> Optional[Path]:
    """Contour one timestep's disturbance into a GeoPackage.

    ``elevation``/``depth``/``x``/``y`` are (node,) arrays and
    ``elnode`` the (ne, 3|4) element table (1- or 0-based, -1/masked
    padding). Filled bands span ``(field_min, levels[0])`` (via
    ``extend='min'``) then each ``(levels[i], levels[i+1])``; bands
    without geometry emit no row, matching the ops PathCollections.
    Returns None without writing when nothing is contourable (all-dry
    field or no band polygons) -- the ops script would error there
    rather than write a file.
    """
    from matplotlib import colors as mcolors
    from matplotlib.figure import Figure
    from matplotlib.tri import Triangulation
    from shapely.geometry import MultiPolygon, Polygon
    from shapely.validation import make_valid

    disturbance = disturbance_field(
        elevation, depth, fill_value=fill_value,
        small_disturbance_threshold=small_disturbance_threshold,
    )

    tris = split_quads(elnode)
    if tris.max() >= len(x):  # ops utils.triangulation: 1-based table
        tris = tris - 1
    tri = Triangulation(x, y, tris)
    dry_tri = np.any(disturbance[tri.triangles] < _MASK_THRESHOLD, axis=1)
    if dry_tri.all():  # tricontourf rejects a fully masked triangulation
        return None
    tri.set_mask(dry_tri)

    levels = [float(level) for level in levels]
    # Band facecolors are discarded by ops too; rgba is re-derived from
    # jet over the emitted rows below, so no cmap/vmin/vmax here.
    fig = Figure()
    ax = fig.add_subplot()
    contour = ax.tricontourf(tri, disturbance, levels=levels, extend="min")

    minmax = [(float(disturbance.min()), levels[0])]
    minmax += [(levels[i], levels[i + 1]) for i in range(len(levels) - 1)]

    rows = []
    for iband, paths in enumerate(_band_paths(contour)):
        polys = []
        for path in paths:
            path.should_simplify = False
            rings = path.to_polygons()
            if not len(rings) or len(rings[0]) <= 3:
                continue
            holes = [ring for ring in rings[1:] if len(ring) > 3]
            polys.append(make_valid(Polygon(rings[0], holes)))
        if not polys:
            continue
        geometry = polys[0] if len(polys) == 1 else MultiPolygon(polys)
        rows.append(
            {
                "id": iband + 1,
                "minWaterLevel": minmax[iband][0],
                "maxWaterLevel": minmax[iband][1],
                "verticalDatum": vertical_datum,
                "units": units,
                "geometry": geometry,
            }
        )
    if not rows:
        return None

    cmap = _jet_cmap()
    for irow, row in enumerate(rows):
        row["rgba"] = mcolors.to_hex(cmap(irow / len(rows)))

    from geopandas import GeoDataFrame

    gdf = GeoDataFrame(rows).set_crs(4326)
    gdf.to_file(out_path, driver="GPKG", layer=layer)
    return Path(out_path)


def write_disturbance_series(
    out2d_files: Sequence[Path],
    out_dir: Path,
    name_fn: Callable[[int], str],
    levels: Sequence[float] = DEFAULT_LEVELS,
    fill_value: float = FILL_VALUE,
    small_disturbance_threshold: float = SMALL_DISTURBANCE_THRESHOLD,
    vertical_datum: str = VERTICAL_DATUM,
    units: str = UNITS,
    max_workers: Optional[int] = None,
) -> List[Path]:
    """Write one disturbance GeoPackage per timestep across stacks.

    ``out2d_files`` are chronological scribe-shaped 2D stacks carrying
    ``elevation`` plus node coordinates/depth/face_nodes (the mesh is
    read from the first stack). Timesteps are numbered globally across
    the concatenation, like the ops merged ``ncrcat`` file, and each
    output is named ``name_fn(istep)`` under ``out_dir``.
    ``max_workers > 1`` fans timesteps out to a process pool (ops used
    a fork pool over the merged array). Returns the written paths in
    timestep order; uncontourable timesteps are skipped.
    """
    from netCDF4 import Dataset

    if not out2d_files:
        raise ValueError("write_disturbance_series: no stack files given")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with Dataset(out2d_files[0], "r") as ds:
        x = np.asarray(ds.variables["SCHISM_hgrid_node_x"][:], dtype=float)
        y = np.asarray(ds.variables["SCHISM_hgrid_node_y"][:], dtype=float)
        depth = np.asarray(ds.variables["depth"][:], dtype=float)
        elnode = np.ma.filled(
            ds.variables["SCHISM_hgrid_face_nodes"][:], -1
        ).astype(int)

    jobs: List[Tuple[np.ndarray, Path]] = []
    istep = 0
    for stack in out2d_files:
        with Dataset(stack, "r") as ds:
            elev = np.ma.filled(
                ds.variables["elevation"][:], fill_value
            ).astype(float)
        elev[elev > _JUNK_ELEVATION] = fill_value
        for irec in range(elev.shape[0]):
            jobs.append((elev[irec], out_dir / name_fn(istep)))
            istep += 1

    kwargs = dict(
        levels=levels,
        fill_value=fill_value,
        small_disturbance_threshold=small_disturbance_threshold,
        vertical_datum=vertical_datum,
        units=units,
    )
    if max_workers and max_workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = [
                pool.submit(
                    write_disturbance_gpkg,
                    elev_i, depth, x, y, elnode, path, **kwargs,
                )
                for elev_i, path in jobs
            ]
            written = [future.result() for future in futures]
    else:
        written = [
            write_disturbance_gpkg(
                elev_i, depth, x, y, elnode, path, **kwargs
            )
            for elev_i, path in jobs
        ]
    return [path for path in written if path is not None]


def nowcast_forecast_namer(
    prefix: str, cyc, n_nowcast: int = 24
) -> Callable[[int], str]:
    """Ops file-name convention with the domain/cycle parameterized.

    Nowcast hours count down to ``n000`` at the cycle time, forecast
    hours count up from ``f001``: timestep 0 of a 24 h nowcast maps to
    ``{prefix}.t{cyc}z.disturbance.n023.gpkg`` and timestep 24 to
    ``...f001.gpkg`` -- the list gen_geojson.py hardcoded for
    stofs_3d_atl at t12z.
    """
    tag = f"t{int(cyc):02d}z"

    def name_fn(istep: int) -> str:
        if istep < n_nowcast:
            hour = n_nowcast - 1 - istep
            return f"{prefix}.{tag}.disturbance.n{hour:03d}.gpkg"
        fhour = istep - n_nowcast + 1
        return f"{prefix}.{tag}.disturbance.f{fhour:03d}.gpkg"

    return name_fn


__all__ = [
    "DEFAULT_LEVELS",
    "FILL_VALUE",
    "SMALL_DISTURBANCE_THRESHOLD",
    "disturbance_field",
    "nowcast_forecast_namer",
    "write_disturbance_gpkg",
    "write_disturbance_series",
]
