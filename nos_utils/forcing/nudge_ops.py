"""numpy port of the ops stofs_3d_atl_gen_nudge_from_hycom (TEM_nu / SAL_nu records)."""

from dataclasses import dataclass

import numpy as np

from .obc_ops_interp import corner_cells, dry_parents, parent_weights_2d

F32 = np.float32
# f90:377, 391, 204-207 in real*4; the -30000 fill unpacks to exactly RJUNK. MJ (10/02/26)
RJUNK = F32(-3.0e4) * F32(1.0e-3) + F32(20.0)
DRY_SALT = RJUNK + F32(0.1)
TEMP_MIN, TEMP_MAX = F32(-15.0), F32(50.0)
SALT_MIN, SALT_MAX = F32(0.0), F32(45.2)
FILL = -30000.0
# ops param.nml step_nu_tr and the port's phase-file spacing (s)
OPS_NU_STEP = 216000.0
NU_DT = 21600.0


def read_grid(path):
    """Stream hgrid.ll/gr3: lon, lat, depth (float32) and 0-based element nodes (ne, 4), -1 padded."""
    with open(path) as f:
        f.readline()
        ne, nn = (int(v) for v in f.readline().split()[:2])
        xyz = np.empty((nn, 3), np.float64)
        for i in range(nn):
            xyz[i] = f.readline().split()[1:4]
        el = np.full((ne, 4), -1, np.int64)
        for i in range(ne):
            p = f.readline().split()
            n = int(p[1])
            el[i, :n] = np.asarray(p[2:2 + n], np.int64) - 1
    xyz = xyz.astype(F32)
    return xyz[:, 0], xyz[:, 1], xyz[:, 2], el


def read_vgrid_nodes(path, idx):
    """LSC2 vgrid.in columns for 0-based nodes idx: (kbp (n,), sigma (nvrt, n) float32), streamed."""
    idx = np.asarray(idx, np.int64)
    with open(path) as f:
        ivcor = int(f.readline().split("!")[0].split()[0])
        if ivcor != 1:
            raise ValueError(f"vgrid ivcor={ivcor}, only LSC2 (1) is supported")
        nvrt = int(f.readline().split("!")[0].split()[0])
        kbp = np.asarray(f.readline().split("!")[0].split(), np.int64)[idx]
        sig = np.full((nvrt, idx.size), -9.0, np.float32)
        for _ in range(nvrt):
            p = f.readline().split("!")[0].split()
            sig[int(p[0]) - 1] = np.asarray(p[1:], np.float64)[idx]
    return kbp, sig


def nudge_zone(values, elnode):
    """Nodes to nudge: |value| > 1e-14 plus one ring of element neighbours (f90:129-160)."""
    inc = np.abs(np.asarray(values, np.float64)) > 1e-14
    valid = elnode >= 0
    hit = (inc[np.where(valid, elnode, 0)] & valid).any(axis=1)
    out = inc.copy()
    out[elnode[hit][valid[hit]]] = True
    return out


def node_z(dp, sigma, kbp):
    """SCHISM z (n, nvrt) float32: max(0.11, dp)*sigma, held at z(kbp) below it (f90:188-200)."""
    h = np.maximum(F32(0.11), np.asarray(dp, F32))
    z = h[:, None] * np.asarray(sigma, F32).T
    nvrt = z.shape[1]
    ix = np.maximum(np.arange(nvrt)[None, :], np.asarray(kbp)[:, None] - 1)
    return np.take_along_axis(z, ix, axis=1)


def pack_ts(real):
    """ops ncap2 packing: float((x-20)/0.001) in double, missing/junk -> -30000 (cvt_tsuv.nco)."""
    real = np.ma.filled(np.ma.asarray(real), FILL).astype(np.float64)
    bad = ~np.isfinite(real) | (np.abs(real) > 1e20) | (np.abs(real - FILL) < 1e-3)
    return np.where(bad, F32(FILL), ((real - 20.0) / 0.001).astype(F32))


def unpack(raw):
    """f90:373-374, real*4."""
    return np.asarray(raw, F32) * F32(1.0e-3) + F32(20.0)


def vertical_index(z, zm, kbp, nz):
    """Level pair and ratio per node/level from the lower-left cell's kbp (0-based, -1 dry), f90:678-704."""
    lev = np.clip(np.searchsorted(zm, z, side="left") - 1, 0, nz - 2)
    vrat = (zm[lev] - z) / (zm[lev] - zm[lev + 1])
    top = z >= zm[-1]
    lev = np.where(top, nz - 2, lev)
    vrat = np.where(top, F32(1), vrat)
    # below the RTOFS bottom the ops uses level kbp itself (gen_3Dth uses level 1)
    below = z <= zm[np.maximum(kbp, 0)][:, None]
    lev = np.where(below, kbp[:, None], lev)
    vrat = np.where(below, F32(0), vrat)
    dry = (kbp < 0)[:, None]
    lev = np.where(dry, nz - 2, lev)
    vrat = np.where(dry, F32(1), vrat)
    lev2 = np.minimum(lev + 1, nz - 1)
    return lev.astype(np.int32), lev2.astype(np.int32), vrat.astype(F32)


def _columns(arr_raw):
    """(nz, ny, nx) top-first packed field -> (ncell, nz) bottom-first float32 real units."""
    a = unpack(arr_raw)[::-1]
    return np.ascontiguousarray(a.reshape(a.shape[0], -1).T)


def _extend(cols_t, cols_s, kbp_cell):
    """Bottom extension of wet columns from the first level with salinity > rjunk (f90:396-408, 606-617).

    kbp_cell is the 0-based klev0 per cell, updated in place for wet cells; dry cells are -1.
    """
    wet = np.flatnonzero(kbp_cell >= 0)
    s = cols_s[wet]
    ok = s > RJUNK
    if not ok.any(axis=1).all():
        raise ValueError("wet RTOFS column with no valid salinity level")
    k0 = ok.argmax(axis=1)
    below = np.arange(s.shape[1])[None, :] < k0[:, None]
    row = np.arange(wet.size)
    for c in (cols_t, cols_s):
        a = c[wet]
        c[wet] = np.where(below, a[row, k0][:, None], a)
    return wet, k0


def _check(t, s, what):
    if not (np.isfinite(t).all() and np.isfinite(s).all()):
        raise ValueError(f"non-finite T/S in {what}")
    if (s < SALT_MIN).any() or (s > SALT_MAX).any() or (t < TEMP_MIN).any() or (t > TEMP_MAX).any():
        raise ValueError(f"T/S outside the ops sanity limits in {what}")


@dataclass
class NudgeState:
    """Everything fixed by the first record: nodes, weights, kbp, parents, vertical indices."""
    sel: np.ndarray        # 0-based global node ids written, ascending
    cells: np.ndarray      # (4, n) flat ROI cell index of the corners LL, LR, UR, UL
    w: np.ndarray          # (n, 4) float32 weights
    kbp: np.ndarray        # (ncell,) 0-based klev0 of the first record, -1 dry
    parent: np.ndarray     # (ncell,) flat index of the cell whose columns a referenced dry cell copies
    check: np.ndarray      # (ncell,) bool: wet cells and the dry corner cells the nodes read
    lev: np.ndarray
    lev2: np.ndarray
    vrat: np.ndarray
    nz: int
    outside: int           # candidates that found no parent (not written)


def build_state(lon, lat, depth, xl, yl, candidate, z, first_salt_raw, chunk=4000):
    """First-record work of the f90: dry mask, parents, weights, vertical indices.

    lon/lat (ny, nx) float32 RTOFS grid, depth (nz,) increasing, xl/yl hgrid.ll float32 over
    all nodes, candidate bool over nodes (zone and inside the grid box), z (n_candidate, nvrt)
    float32 SCHISM levels of the candidates, first_salt_raw (nz, ny, nx) packed salinity.
    Returns (state, found) where found marks the candidates that got a parent.
    """
    ny, nx = lon.shape
    nz = depth.size
    zm = (-np.asarray(depth, F32))[::-1].copy()
    xl, yl = np.asarray(xl, F32), np.asarray(yl, F32)
    cand = np.flatnonzero(candidate)
    ix, iy, w, found = parent_weights_2d(lon, lat, xl[cand], yl[cand], single=True, chunk=chunk)
    sel = cand[found]
    ix, iy, w = ix[found], iy[found], w[found]
    cj, ci = corner_cells(ix, iy)
    sal = unpack(first_salt_raw)[::-1]
    kbp = np.where(sal[-1] < DRY_SALT, -1, 0).reshape(-1).astype(np.int64)
    cols_t = np.zeros((kbp.size, nz), F32)
    cols_s = np.ascontiguousarray(sal.reshape(nz, -1).T)
    wet_cells, k0 = _extend(cols_t, cols_s, kbp)
    kbp[wet_cells] = k0
    parent = np.arange(kbp.size)
    pj, pi, _ = dry_parents((kbp >= 0).reshape(ny, nx), cj, ci)
    parent[(cj * nx + ci).ravel()] = (pj * nx + pi).ravel()
    cells = cj * nx + ci
    # Only the dry cells some node reads are given parents; the f90 fills them all, but the rest
    # never reach the output. MJ (10/02/26)
    check = kbp >= 0
    check[cells.ravel()] = True
    lev, lev2, vrat = vertical_index(z[found], zm, kbp[cells[0]], nz)
    return NudgeState(sel, cells, w.astype(F32), kbp, parent, check, lev, lev2, vrat, nz,
                      int((~found).sum())), found


def interpolate_record(state, t_raw, s_raw, chunk=20000):
    """One record: packed (nz, ny, nx) T and S -> (n, nvrt) float32 T (floored at 0) and S, as f90:601-736."""
    cols_t, cols_s = _columns(t_raw), _columns(s_raw)
    kbp = state.kbp.copy()
    wet, _ = _extend(cols_t, cols_s, np.where(kbp >= 0, 0, -1))
    dry = np.flatnonzero((kbp < 0) & state.check)
    cols_t[dry] = cols_t[state.parent[dry]]
    cols_s[dry] = cols_s[state.parent[dry]]
    _check(cols_t[state.check], cols_s[state.check], "the RTOFS columns")
    n, nvrt = state.lev.shape
    out_t = np.empty((n, nvrt), F32)
    out_s = np.empty((n, nvrt), F32)
    one = F32(1)
    for s0 in range(0, n, chunk):
        sl = slice(s0, s0 + chunk)
        lev, lev2, vrat, w = state.lev[sl], state.lev2[sl], state.vrat[sl], state.w[sl]
        for cols, out in ((cols_t, out_t), (cols_s, out_s)):
            acc = None
            for c in range(4):
                ci = state.cells[c, sl][:, None]
                v = cols[ci, lev] * (one - vrat) + cols[ci, lev2] * vrat
                term = v * w[:, c:c + 1]
                acc = term if acc is None else acc + term
            out[sl] = acc
    _check(out_t, out_s, "the interpolated nudging field")
    np.maximum(out_t, F32(0), out=out_t)
    return out_t, out_s


def effective(rec, n, num, ratio):
    """Ops-effective field at index n + num/ratio of the ops records (float64 blend, float32 result)."""
    if num == 0:
        return rec[n]
    f = num / ratio
    return (rec[n].astype(np.float64) * (1.0 - f) + rec[n + 1].astype(np.float64) * f).astype(F32)


def phase_plan(offset_s, duration_s, dt=NU_DT, step=OPS_NU_STEP, run_s=None):
    """Record plan of one phase file: (n_records, ratio, offset_steps, aligned, n_ops_records_needed, last_j).

    Record j sits at offset_s + j*dt of the nowcast clock and holds the ops-effective field there.
    SCHISM never reads past the run end, so only records up to run_s (default duration_s) need real
    values (last_j); the trailing buffer records hold the last one and need no extra ops record.
    """
    ratio = int(round(step / dt))
    aligned = abs(ratio * dt - step) < 1e-6 and abs(offset_s / dt - round(offset_s / dt)) < 1e-9
    n_out = int(np.ceil(duration_s / dt - 1e-9)) + 1
    off = int(round(offset_s / dt))
    last_j = n_out - 1
    if run_s is not None:
        last_j = min(last_j, int(np.ceil(run_s / dt - 1e-9)))
    last_n, last_num = divmod(off + last_j, ratio)
    return n_out, ratio, off, aligned, last_n + 1 + (1 if last_num else 0), last_j
