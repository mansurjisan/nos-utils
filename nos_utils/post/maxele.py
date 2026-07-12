"""Maximum-water-level field product (``fields.cwl.maxele.nc``).

Port of the NCO chain in IT-STOFS
``stofs_3d_atl_create_AWS_autoval_nc.sh`` (ncks/ncra/ncrcat/ncwa/
ncrename/ncap2/ncatted): per-stack max of ``elevation`` over the
forecast stacks, reduced to one global max per node, published in the
autoval/ADCIRC-style schema -- ``zeta_max(node)`` with ``x``/``y``/
``depth`` companions and a 2-point ``time`` coordinate spanning the
product window. Raw max like ops: dry nodes carry their bed elevation;
masking is a downstream consumer concern.

Caller contract for ops parity: pass exactly the FORECAST stacks (ops
reduces records 1-12 of stacks 3-10 only) and override
``window_seconds=(90000, 432000)`` to reproduce the hardcoded ops time
coordinate; the default derives the window from the stacks' own time
axes (identical for the standard run).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

FILL_VALUE = -99999.0


def write_maxele(
    stack_files: Sequence[Path],
    out_path: Path,
    base_date: str,
    window_seconds: Optional[Tuple[float, float]] = None,
) -> Path:
    """Reduce ``elevation`` over ``stack_files`` and write the maxele nc.

    ``stack_files`` are canonical scribe-shaped 2D stacks (``time`` +
    ``elevation(time, node)``, optionally node coords + depth) in
    chronological order. ``base_date`` (``YYYY-MM-DD HH:MM`` style)
    stamps the time units, matching the ops param.nml-derived attrs.
    ``window_seconds`` overrides the (start, end) time coordinate pair;
    by default it is read from the first/last records of the stacks.
    """
    from netCDF4 import Dataset

    if not stack_files:
        raise ValueError("write_maxele: no stack files given")

    zeta_max: Optional[np.ndarray] = None
    x = y = depth = None
    x_attrs: dict = {}
    y_attrs: dict = {}
    t_first: Optional[float] = None
    t_last: Optional[float] = None

    for path in stack_files:
        with Dataset(path, "r") as ds:
            t = ds.variables["time"][:]
            if t.size == 0:
                continue
            if t_first is None:
                t_first = float(t[0])
            t_last = float(t[-1])
            elev = np.ma.filled(
                ds.variables["elevation"][:], FILL_VALUE
            ).astype("f4")
            stack_max = elev.max(axis=0)
            zeta_max = (
                stack_max if zeta_max is None
                else np.maximum(zeta_max, stack_max)
            )
            if x is None and "SCHISM_hgrid_node_x" in ds.variables:
                xv_src = ds.variables["SCHISM_hgrid_node_x"]
                yv_src = ds.variables["SCHISM_hgrid_node_y"]
                x = xv_src[:]
                y = yv_src[:]
                # ops renames the coord vars, which keeps their attrs
                x_attrs = {a: xv_src.getncattr(a) for a in xv_src.ncattrs()}
                y_attrs = {a: yv_src.getncattr(a) for a in yv_src.ncattrs()}
            if depth is None and "depth" in ds.variables:
                depth = ds.variables["depth"][:]

    if zeta_max is None:
        raise ValueError("write_maxele: all stacks were empty")

    if window_seconds is None:
        window_seconds = (t_first, t_last)

    with Dataset(out_path, "w", format="NETCDF4") as ds:
        ds.createDimension("node", zeta_max.size)
        ds.createDimension("time", 2)

        tv = ds.createVariable("time", "f8", ("time",))
        tv.units = f"seconds since {base_date}"
        tv.base_date = base_date
        tv[:] = list(window_seconds)

        zv = ds.createVariable(
            "zeta_max", "f4", ("node",), fill_value=FILL_VALUE
        )
        zv[:] = zeta_max

        if x is not None:
            xv = ds.createVariable("x", "f8", ("node",))
            yv = ds.createVariable("y", "f8", ("node",))
            for key, val in x_attrs.items():
                xv.setncattr(key, val)
            for key, val in y_attrs.items():
                yv.setncattr(key, val)
            xv[:] = x
            yv[:] = y
        if depth is not None:
            dv = ds.createVariable("depth", "f4", ("node",))
            dv.units = "m"
            dv.long_name = "distance below geoid"
            dv.standard_name = "depth below geoid"
            dv.mesh = "adcirc_mesh"
            dv.coordinates = "time y x"
            dv[:] = depth

        # ops: ncatted -a _FillValue,global,o,c,"-99999."
        ds.setncattr("_FillValue", "-99999.")

    return Path(out_path)


__all__ = ["FILL_VALUE", "write_maxele"]
