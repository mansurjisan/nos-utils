"""Shared synthetic fixtures for the post-product writer tests.

A 4-node / 2-triangle mini mesh (nodes: 1(0,0) 2(1,0) 3(0,1) 4(1,1),
elems (1,2,3) and (2,4,3), depth 5 m) plus builders for canonical
scribe-shaped stack files and SCHISM text inputs. Every post test
builds from these so the mesh assumptions stay in one place.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np

MINI_X = np.array([0.0, 1.0, 0.0, 1.0])
MINI_Y = np.array([0.0, 0.0, 1.0, 1.0])
MINI_DEPTH = np.full(4, 5.0)
MINI_ELEMS_1BASED = np.array([[1, 2, 3, -1], [2, 4, 3, -1]])
MINI_NVRT = 3
MINI_SIGMA = np.array([-1.0, -0.5, 0.0])


def default_elev(hour: float, node: int) -> float:
    """Deterministic, verifiable node value: node_id + hour/100."""
    return (node + 1) + hour / 100.0


def write_out2d_stack(
    path: Path,
    hours: Iterable[float],
    elev_fn: Callable[[float, int], float] = default_elev,
    n_nodes: int = 4,
    with_coords: bool = True,
    base_date: str = "2026-07-10 00:00:00",
) -> Path:
    """Canonical scribe-shaped 2D stack: time + elevation (+ coords/depth)."""
    from netCDF4 import Dataset

    hours = list(hours)
    with Dataset(path, "w", format="NETCDF4") as ds:
        ds.createDimension("time", None)
        ds.createDimension("nSCHISM_hgrid_node", n_nodes)
        tv = ds.createVariable("time", "f8", ("time",))
        tv.units = f"seconds since {base_date}"
        tv[:] = [h * 3600.0 for h in hours]
        ev = ds.createVariable(
            "elevation", "f4", ("time", "nSCHISM_hgrid_node")
        )
        for it, h in enumerate(hours):
            ev[it, :] = [elev_fn(h, n) for n in range(n_nodes)]
        if with_coords:
            xv = ds.createVariable(
                "SCHISM_hgrid_node_x", "f8", ("nSCHISM_hgrid_node",)
            )
            yv = ds.createVariable(
                "SCHISM_hgrid_node_y", "f8", ("nSCHISM_hgrid_node",)
            )
            dv = ds.createVariable("depth", "f4", ("nSCHISM_hgrid_node",))
            xv[:] = MINI_X[:n_nodes]
            yv[:] = MINI_Y[:n_nodes]
            dv[:] = MINI_DEPTH[:n_nodes]
    return path


def write_var3d_stack(
    path: Path,
    var: str,
    hours: Iterable[float],
    value_fn: Optional[Callable[[float, int, int], float]] = None,
    n_nodes: int = 4,
    nvrt: int = MINI_NVRT,
    base_date: str = "2026-07-10 00:00:00",
) -> Path:
    """Canonical scribe-shaped 3D stack: (time, node, layer) variable."""
    from netCDF4 import Dataset

    hours = list(hours)
    if value_fn is None:
        def value_fn(h, n, k):  # noqa: E306
            return (n + 1) * 10.0 + k + h / 100.0

    with Dataset(path, "w", format="NETCDF4") as ds:
        ds.createDimension("time", None)
        ds.createDimension("nSCHISM_hgrid_node", n_nodes)
        ds.createDimension("nSCHISM_vgrid_layers", nvrt)
        tv = ds.createVariable("time", "f8", ("time",))
        tv.units = f"seconds since {base_date}"
        tv[:] = [h * 3600.0 for h in hours]
        vv = ds.createVariable(
            var, "f4",
            ("time", "nSCHISM_hgrid_node", "nSCHISM_vgrid_layers"),
        )
        for it, h in enumerate(hours):
            for n in range(n_nodes):
                vv[it, n, :] = [value_fn(h, n, k) for k in range(nvrt)]
    return path


def write_mini_hgrid(path: Path) -> Path:
    """SCHISM hgrid.gr3 for the mini mesh (no boundary blocks)."""
    lines = ["mini mesh", "2 4"]
    for i in range(4):
        lines.append(f"{i + 1} {MINI_X[i]} {MINI_Y[i]} {MINI_DEPTH[i]}")
    for ie, row in enumerate(MINI_ELEMS_1BASED):
        nodes = [n for n in row if n > 0]
        lines.append(
            f"{ie + 1} {len(nodes)} " + " ".join(str(n) for n in nodes)
        )
    path.write_text("\n".join(lines) + "\n")
    return path


def write_mini_vgrid(path: Path) -> Path:
    """Pure-sigma SCHISM vgrid.in (ivcor=1 style) for the mini mesh."""
    lines = ["1 !ivcor", f"{MINI_NVRT}"]
    for i in range(4):
        sig = " ".join(f"{s}" for s in MINI_SIGMA)
        lines.append(f"{i + 1} 1 {sig}")
    path.write_text("\n".join(lines) + "\n")
    return path


__all__ = [
    "MINI_DEPTH",
    "MINI_ELEMS_1BASED",
    "MINI_NVRT",
    "MINI_SIGMA",
    "MINI_X",
    "MINI_Y",
    "default_elev",
    "write_mini_hgrid",
    "write_mini_vgrid",
    "write_out2d_stack",
    "write_var3d_stack",
]
