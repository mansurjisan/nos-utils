"""
Precomputed INTERP_REMESH weight replay for SSH, 3D T/S, and nudging interpolation.

Uses weights exported from the Fortran INTERP_REMESH subroutine to
produce numerically equivalent interpolation without Delaunay
triangulation. The weights are computed once and stored in .npz
files in the FIX directory.

SSH weights (obc_ssh_weights.npz):
    1488 boundary nodes for elev2D.th.nc
    Source grid: global RTOFS 2D (3298x4500)

3D weights (obc_3d_weights.npz):
    1488 boundary nodes for TEM_3D.th.nc / SAL_3D.th.nc
    Source grid: regional RTOFS 3D (e.g., US_east 1710x742)

Nudge weights (obc_nudge_weights.npz):
    ~32K interior nodes for TEM_nu.nc / SAL_nu.nc
    Source grid: global RTOFS 2D (3298x4500)

Runtime:
1. First call: KDTree maps source coordinates to RTOFS grid cells (cached)
2. Per timestep: gather data at source cells, apply weights (pure indexed ops)
"""

import hashlib
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def load_remesh_export(filepath: Path) -> dict:
    """
    Load the one-time Fortran INTERP_REMESH export text file.

    Returns dict with source coordinates, target mapping, weights, modes, donors.
    """
    metadata = {}
    src = []
    tgt_idx, tgt_w, tgt_mode, tgt_donor = [], [], [], []
    section = None

    with open(filepath) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("## ") and "=" in line and section is None:
                k, v = line[3:].split("=", 1)
                metadata[k.strip()] = v.strip()
                continue
            if line == "## SOURCE_POINTS":
                section = "src"
                continue
            if line == "## TARGET_MAPPING":
                section = "tgt"
                continue
            if line.startswith("##"):
                continue

            p = line.split()
            if section == "src" and len(p) == 4:
                src.append((float(p[1]), float(p[2]), int(p[3])))
            elif section == "tgt" and len(p) == 9:
                tgt_idx.append([int(p[1]), int(p[2]), int(p[3])])
                tgt_w.append([float(p[4]), float(p[5]), float(p[6])])
                tgt_mode.append(int(p[7]))
                tgt_donor.append(int(p[8]))

    src = np.array(src)
    return {
        "source_lon": src[:, 0].astype(np.float64),
        "source_lat": src[:, 1].astype(np.float64),
        "source_is_corner": src[:, 2].astype(bool),
        "target_idx": np.array(tgt_idx, dtype=np.int32),
        "weights": np.array(tgt_w, dtype=np.float64),
        "mode": np.array(tgt_mode, dtype=np.int32),
        "donor": np.array(tgt_donor, dtype=np.int32),
        "metadata": metadata,
    }


def build_npz(
    export_txt: Path,
    rtofs_lon_2d: np.ndarray,
    rtofs_lat_2d: np.ndarray,
    out_npz: Path,
    tol: float = 0.01,
) -> None:
    """
    One-time conversion: export text → production .npz with grid indices.

    Args:
        export_txt: Path to obc_ssh_remesh_export.txt
        rtofs_lon_2d: RTOFS 2D longitude array (ny, nx)
        rtofs_lat_2d: RTOFS 2D latitude array (ny, nx)
        out_npz: Output NPZ path
        tol: Max allowed distance (degrees) for source-to-grid matching
    """
    exp = load_remesh_export(export_txt)

    lon = np.where(rtofs_lon_2d > 180, rtofs_lon_2d - 360, rtofs_lon_2d)
    lat = rtofs_lat_2d
    flat_lon = lon.ravel()
    flat_lat = lat.ravel()

    src_lon = exp["source_lon"]
    src_lat = exp["source_lat"]
    is_corner = exp["source_is_corner"]

    # Map non-corner source points to nearest RTOFS grid cell
    tree = cKDTree(np.column_stack([flat_lon, flat_lat]))
    ocean_src = np.column_stack([src_lon[~is_corner], src_lat[~is_corner]])
    dist, flat_idx = tree.query(ocean_src)

    # Validate: every source must map unambiguously
    if np.any(dist > tol):
        bad = np.where(dist > tol)[0]
        raise ValueError(
            f"{len(bad)} source points exceed tol={tol}; "
            f"max dist={dist.max():.6f} at source idx {bad[0]}"
        )

    # Verify uniqueness
    unique_idx = np.unique(flat_idx)
    if len(unique_idx) < len(flat_idx):
        n_dup = len(flat_idx) - len(unique_idx)
        log.warning(f"{n_dup} source points map to duplicate grid cells "
                    f"(expected for ocean-filtered subset)")

    # Build canonical flat index for all source points
    source_flat_idx = np.full(len(src_lon), -1, dtype=np.int32)
    source_flat_idx[~is_corner] = flat_idx.astype(np.int32)

    # Map target vertices to flat grid indices
    vertex_src = exp["target_idx"] - 1  # 1-based to 0-based
    vertex_flat_idx = source_flat_idx[vertex_src]
    vertex_is_corner = vertex_flat_idx < 0

    # Donor: 1-based to 0-based
    donor0 = np.where(exp["donor"] > 0, exp["donor"] - 1, -1).astype(np.int32)

    # Grid identity checksum (canonical float64 contiguous)
    lon_canon = np.ascontiguousarray(lon, dtype=np.float64)
    lat_canon = np.ascontiguousarray(lat, dtype=np.float64)
    grid_hash = hashlib.md5(lon_canon.tobytes() + lat_canon.tobytes()).hexdigest()

    # Source data flat indices (for computing corner mean at runtime)
    source_data_flat_idx = source_flat_idx[~is_corner].astype(np.int32)

    np.savez(
        str(out_npz),
        vertex_flat_idx=vertex_flat_idx.astype(np.int32),
        vertex_is_corner=vertex_is_corner,
        source_data_flat_idx=source_data_flat_idx,
        weights=exp["weights"],
        mode=exp["mode"],
        donor=donor0,
        grid_shape=np.array(lon.shape, dtype=np.int32),
        grid_hash=np.array([grid_hash]),
        corner_mean_export=np.float64(float(exp["metadata"].get("corner_mean", "0"))),
        n_target=np.int32(len(exp["mode"])),
        n_source_data=np.int32(int(exp["metadata"].get("n_source_data", "0"))),
    )

    log.info(f"Built SSH weight NPZ: {len(exp['mode'])} targets, "
             f"{len(source_data_flat_idx)} source cells, "
             f"max dist={dist.max():.2e}, hash={grid_hash}")


def validate_grid(npz: dict, rtofs_lon_2d: np.ndarray, rtofs_lat_2d: np.ndarray) -> None:
    """Verify the RTOFS grid matches the stored weight mapping."""
    if tuple(rtofs_lon_2d.shape) != tuple(npz["grid_shape"]):
        raise ValueError(
            f"Grid shape mismatch: weights for {tuple(npz['grid_shape'])}, "
            f"data is {rtofs_lon_2d.shape}")

    lon = np.ascontiguousarray(
        np.where(rtofs_lon_2d > 180, rtofs_lon_2d - 360, rtofs_lon_2d),
        dtype=np.float64)
    lat = np.ascontiguousarray(rtofs_lat_2d, dtype=np.float64)
    current_hash = hashlib.md5(lon.tobytes() + lat.tobytes()).hexdigest()
    stored_hash = str(npz["grid_hash"][0])
    if current_hash != stored_hash:
        raise ValueError(
            f"RTOFS grid changed: stored={stored_hash}, current={current_hash}")


def apply_precomputed_ssh(npz: dict, ssh_2d: np.ndarray) -> np.ndarray:
    """
    Apply precomputed Fortran REMESH weights to RTOFS SSH field.

    Numerically equivalent to Fortran, with residual differences on the
    order of floating-point roundoff (REAL*4 weights, float64 arithmetic).

    Args:
        npz: Dict from np.load('secofs.obc_ssh_weights.npz')
        ssh_2d: Current cycle RTOFS SSH field, shape matching grid_shape

    Returns:
        SSH at boundary nodes, shape (n_target,). Does NOT include SSH offset.
    """
    if tuple(ssh_2d.shape) != tuple(npz["grid_shape"]):
        raise ValueError(
            f"Grid shape mismatch: weights for {tuple(npz['grid_shape'])}, "
            f"data is {ssh_2d.shape}")

    flat = np.asarray(ssh_2d, dtype=np.float64).ravel()

    # Corner mean from current field's source points
    source_data = flat[npz["source_data_flat_idx"]]
    corner_mean = float(source_data.mean())

    # Gather vertex values: corner → mean, grid → flat[idx]
    # Use np.clip to avoid negative index wrapping for corner vertices
    safe_idx = np.clip(npz["vertex_flat_idx"], 0, len(flat) - 1)
    vals = np.where(
        npz["vertex_is_corner"],
        corner_mean,
        flat[safe_idx],
    )

    # Weighted sum
    out = np.sum(npz["weights"] * vals, axis=1, dtype=np.float64)

    # Fallback: mode=2 nodes copy from donor
    for k in np.where(npz["mode"] == 2)[0]:
        d = int(npz["donor"][k])
        if d < 0 or d >= len(out):
            raise ValueError(f"Invalid donor at node {k}: donor={d}")
        if npz["mode"][d] == 2 and d >= k:
            raise ValueError(f"Circular fallback at node {k}: donor={d}")
        out[k] = out[d]

    return out.astype(np.float32)


# ======================================================================
# Nudge weight functions (interior T/S nudging, ~32K target nodes)
# ======================================================================


def build_nudge_npz(
    export_txt: Path,
    rtofs_lon_2d: np.ndarray,
    rtofs_lat_2d: np.ndarray,
    out_npz: Path,
    tol: float = 0.01,
) -> None:
    """
    One-time conversion: nudge export text -> production .npz with grid indices.

    The export text has the same format as the SSH export
    (produced by INTERP_REMESH with NUDGE_REMESH_ACTIVE), but for the
    ~32K interior nudging nodes instead of ~1488 boundary nodes.

    Args:
        export_txt: Path to obc_nudge_remesh_export.txt
        rtofs_lon_2d: RTOFS 2D longitude array (ny, nx)
        rtofs_lat_2d: RTOFS 2D latitude array (ny, nx)
        out_npz: Output NPZ path
        tol: Max allowed distance (degrees) for source-to-grid matching
    """
    exp = load_remesh_export(export_txt)

    lon = np.where(rtofs_lon_2d > 180, rtofs_lon_2d - 360, rtofs_lon_2d)
    lat = rtofs_lat_2d
    flat_lon = lon.ravel()
    flat_lat = lat.ravel()

    src_lon = exp["source_lon"]
    src_lat = exp["source_lat"]
    is_corner = exp["source_is_corner"]

    # Map non-corner source points to nearest RTOFS grid cell
    tree = cKDTree(np.column_stack([flat_lon, flat_lat]))
    ocean_src = np.column_stack([src_lon[~is_corner], src_lat[~is_corner]])
    dist, flat_idx = tree.query(ocean_src)

    if np.any(dist > tol):
        bad = np.where(dist > tol)[0]
        raise ValueError(
            f"{len(bad)} source points exceed tol={tol}; "
            f"max dist={dist.max():.6f} at source idx {bad[0]}"
        )

    if len(np.unique(flat_idx)) < len(flat_idx):
        n_dup = len(flat_idx) - len(np.unique(flat_idx))
        log.warning(f"{n_dup} source points map to duplicate grid cells "
                    f"(expected for ocean-filtered subset)")

    # Build canonical flat index for all source points
    source_flat_idx = np.full(len(src_lon), -1, dtype=np.int32)
    source_flat_idx[~is_corner] = flat_idx.astype(np.int32)

    # Map target vertices to flat grid indices
    vertex_src = exp["target_idx"] - 1  # 1-based to 0-based
    vertex_flat_idx = source_flat_idx[vertex_src]
    vertex_is_corner = vertex_flat_idx < 0

    # Donor: 1-based to 0-based
    donor0 = np.where(exp["donor"] > 0, exp["donor"] - 1, -1).astype(np.int32)

    # Grid identity checksum
    lon_canon = np.ascontiguousarray(lon, dtype=np.float64)
    lat_canon = np.ascontiguousarray(lat, dtype=np.float64)
    grid_hash = hashlib.md5(lon_canon.tobytes() + lat_canon.tobytes()).hexdigest()

    # Source data flat indices
    source_data_flat_idx = source_flat_idx[~is_corner].astype(np.int32)

    np.savez(
        str(out_npz),
        vertex_flat_idx=vertex_flat_idx.astype(np.int32),
        vertex_is_corner=vertex_is_corner,
        source_data_flat_idx=source_data_flat_idx,
        weights=exp["weights"],
        mode=exp["mode"],
        donor=donor0,
        grid_shape=np.array(lon.shape, dtype=np.int32),
        grid_hash=np.array([grid_hash]),
        corner_mean_export=np.float64(float(exp["metadata"].get("corner_mean", "0"))),
        n_target=np.int32(len(exp["mode"])),
        n_source_data=np.int32(int(exp["metadata"].get("n_source_data", "0"))),
    )

    log.info(f"Built NUDGE weight NPZ: {len(exp['mode'])} targets, "
             f"{len(source_data_flat_idx)} source cells, "
             f"max dist={dist.max():.2e}, hash={grid_hash}")


def build_3d_npz(
    export_txt: Path,
    rtofs_lon_2d: np.ndarray,
    rtofs_lat_2d: np.ndarray,
    out_npz: Path,
    tol: float = 0.01,
) -> None:
    """
    One-time conversion: 3D T/S export text -> production .npz with grid indices.

    The export text has the same format as the SSH and nudge exports
    (produced by INTERP_REMESH with TS3D_REMESH_ACTIVE), but uses the
    regional RTOFS 3D grid (e.g., US_east 1710x742) instead of the
    global 2D grid (3298x4500). Target nodes are the same 1488 boundary
    nodes as SSH.

    Args:
        export_txt: Path to obc_3d_remesh_export.txt
        rtofs_lon_2d: RTOFS 3D regional longitude array (ny, nx)
        rtofs_lat_2d: RTOFS 3D regional latitude array (ny, nx)
        out_npz: Output NPZ path
        tol: Max allowed distance (degrees) for source-to-grid matching
    """
    exp = load_remesh_export(export_txt)

    lon = np.where(rtofs_lon_2d > 180, rtofs_lon_2d - 360, rtofs_lon_2d)
    lat = rtofs_lat_2d
    flat_lon = lon.ravel()
    flat_lat = lat.ravel()

    src_lon = exp["source_lon"]
    src_lat = exp["source_lat"]
    is_corner = exp["source_is_corner"]

    # Map non-corner source points to nearest RTOFS grid cell
    tree = cKDTree(np.column_stack([flat_lon, flat_lat]))
    ocean_src = np.column_stack([src_lon[~is_corner], src_lat[~is_corner]])
    dist, flat_idx = tree.query(ocean_src)

    if np.any(dist > tol):
        bad = np.where(dist > tol)[0]
        raise ValueError(
            f"{len(bad)} source points exceed tol={tol}; "
            f"max dist={dist.max():.6f} at source idx {bad[0]}"
        )

    if len(np.unique(flat_idx)) < len(flat_idx):
        n_dup = len(flat_idx) - len(np.unique(flat_idx))
        log.warning(f"{n_dup} source points map to duplicate grid cells "
                    f"(expected for ocean-filtered subset)")

    # Build canonical flat index for all source points
    source_flat_idx = np.full(len(src_lon), -1, dtype=np.int32)
    source_flat_idx[~is_corner] = flat_idx.astype(np.int32)

    # Map target vertices to flat grid indices
    vertex_src = exp["target_idx"] - 1  # 1-based to 0-based
    vertex_flat_idx = source_flat_idx[vertex_src]
    vertex_is_corner = vertex_flat_idx < 0

    # Donor: 1-based to 0-based
    donor0 = np.where(exp["donor"] > 0, exp["donor"] - 1, -1).astype(np.int32)

    # Grid identity checksum
    lon_canon = np.ascontiguousarray(lon, dtype=np.float64)
    lat_canon = np.ascontiguousarray(lat, dtype=np.float64)
    grid_hash = hashlib.md5(lon_canon.tobytes() + lat_canon.tobytes()).hexdigest()

    # Source data flat indices
    source_data_flat_idx = source_flat_idx[~is_corner].astype(np.int32)

    np.savez(
        str(out_npz),
        vertex_flat_idx=vertex_flat_idx.astype(np.int32),
        vertex_is_corner=vertex_is_corner,
        source_data_flat_idx=source_data_flat_idx,
        weights=exp["weights"],
        mode=exp["mode"],
        donor=donor0,
        grid_shape=np.array(lon.shape, dtype=np.int32),
        grid_hash=np.array([grid_hash]),
        corner_mean_export=np.float64(float(exp["metadata"].get("corner_mean", "0"))),
        n_target=np.int32(len(exp["mode"])),
        n_source_data=np.int32(int(exp["metadata"].get("n_source_data", "0"))),
    )

    log.info(f"Built 3D weight NPZ: {len(exp['mode'])} targets, "
             f"{len(source_data_flat_idx)} source cells, "
             f"max dist={dist.max():.2e}, hash={grid_hash}")


def apply_precomputed_nudge(
    npz: dict,
    field_2d: np.ndarray,
    fill_value: float = np.nan,
) -> np.ndarray:
    """
    Apply precomputed Fortran REMESH weights to a 2D RTOFS field for nudging.

    Works identically to ``apply_precomputed_ssh`` but for the ~32K interior
    nudging target nodes. Each call interpolates ONE 2D slice (one depth
    level, one timestep) from the RTOFS grid to the nudge target nodes.

    Args:
        npz: Dict from np.load('secofs.obc_nudge_weights.npz')
        field_2d: Single 2D RTOFS field, shape matching grid_shape.
                  Land/fill values should be set to NaN or abs >= 99.
        fill_value: Value to use for corner-mean when all data is NaN.
                    Defaults to NaN (let caller handle fill).

    Returns:
        Interpolated values at nudge nodes, shape (n_target,), float32.
    """
    if tuple(field_2d.shape) != tuple(npz["grid_shape"]):
        raise ValueError(
            f"Grid shape mismatch: weights for {tuple(npz['grid_shape'])}, "
            f"data is {field_2d.shape}")

    flat = np.asarray(field_2d, dtype=np.float64).ravel()

    # Corner mean from current field's source points
    source_data = flat[npz["source_data_flat_idx"]]
    finite_mask = np.isfinite(source_data) & (np.abs(source_data) < 99.0)

    if np.any(finite_mask):
        corner_mean = float(source_data[finite_mask].mean())
    else:
        corner_mean = float(fill_value)

    # Gather vertex values: corner -> mean, grid -> flat[idx]
    # Use np.clip to avoid negative index wrapping for corner vertices
    safe_idx = np.clip(npz["vertex_flat_idx"], 0, len(flat) - 1)
    vals = np.where(
        npz["vertex_is_corner"],
        corner_mean,
        flat[safe_idx],
    )

    # Replace NaN source values with corner_mean to avoid NaN propagation
    nan_vals = ~np.isfinite(vals)
    if np.any(nan_vals):
        vals = np.where(nan_vals, corner_mean, vals)

    # Weighted sum
    out = np.sum(npz["weights"] * vals, axis=1, dtype=np.float64)

    # Fallback: mode=2 nodes copy from donor
    for k in np.where(npz["mode"] == 2)[0]:
        d = int(npz["donor"][k])
        if d < 0 or d >= len(out):
            # Invalid donor — use corner mean as final fallback
            out[k] = corner_mean
            continue
        if npz["mode"][d] == 2 and d >= k:
            out[k] = corner_mean
            continue
        out[k] = out[d]

    return out.astype(np.float32)
