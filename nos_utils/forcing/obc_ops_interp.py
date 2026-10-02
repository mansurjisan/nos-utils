"""numpy reproduction of the ops stofs_3d_atl_gen_3Dth_from_hycom interpolation."""

import numpy as np

# ops thresholds, after the *1e-3 (+20 for T/S) unpacking (f90:478, 488)
RJUNK = -10.0
DRY_SSH = RJUNK + 0.1
# float robustness: a -30000 fill unpacks to RJUNK +- rounding. MJ (10/01/26)
JUNK_EPS = 1e-3


def rect_axes(lon, lat, tol=1e-3):
    """1-D increasing (xax, yax) from 1-D or separable 2-D lon/lat, else None."""
    lon = np.ma.filled(lon, np.nan).astype(np.float64)
    lat = np.ma.filled(lat, np.nan).astype(np.float64)
    if lon.ndim == 2 and lat.ndim == 2 and lon.shape == lat.shape:
        if not (np.isfinite(lon).all() and np.isfinite(lat).all()):
            return None
        if np.ptp(lon, axis=0).max() > tol or np.ptp(lat, axis=1).max() > tol:
            return None
        lon, lat = lon[0, :], lat[:, 0]
    if lon.ndim != 1 or lat.ndim != 1 or lon.size < 2 or lat.size < 2:
        return None
    if not (np.all(np.diff(lon) > 0) and np.all(np.diff(lat) > 0)):
        return None
    return lon, lat


def parent_weights(xax, yax, x, y, mode=0):
    """Parent cell and 4 weights [LL, LR, UR, UL] per point; mode 1 splits the cell along LL-UR."""
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    inside = (x >= xax[0]) & (x <= xax[-1]) & (y >= yax[0]) & (y <= yax[-1])
    # side="left" - 1 picks the lowest cell on a shared edge, as the ops first-match loop does
    ix = np.clip(np.searchsorted(xax, x, side="left") - 1, 0, xax.size - 2)
    iy = np.clip(np.searchsorted(yax, y, side="left") - 1, 0, yax.size - 2)
    xr = np.clip((x - xax[ix]) / (xax[ix + 1] - xax[ix]), 0.0, 1.0)
    yr = np.clip((y - yax[iy]) / (yax[iy + 1] - yax[iy]), 0.0, 1.0)
    w = np.zeros(x.shape + (4,))
    if mode == 0:
        w[..., 0] = (1 - xr) * (1 - yr)
        w[..., 1] = xr * (1 - yr)
        w[..., 2] = xr * yr
        w[..., 3] = (1 - xr) * yr
    else:
        # f90:726-749; triangle (1,2,3) is accepted up to 0.025 past the diagonal
        tri1 = (yr - xr) < 0.025
        w1 = np.clip(1 - xr, 0, 1)
        w2 = np.clip(np.abs(xr - yr), 0, 1)
        w[..., 0] = np.where(tri1, w1, np.clip(1 - yr, 0, 1))
        w[..., 1] = np.where(tri1, w2, 0.0)
        w[..., 2] = np.where(tri1, np.clip(1 - w1 - w2, 0, 1), np.clip(xr, 0, 1))
        w[..., 3] = np.where(tri1, 0.0, np.clip(1 - (1 - yr) - xr, 0, 1))
    w[~inside] = 0.0
    return ix, iy, w, inside


def corner_cells(ix, iy):
    """(4, n) row/col index arrays of the [LL, LR, UR, UL] corners."""
    return (np.stack([iy, iy, iy + 1, iy + 1]), np.stack([ix, ix + 1, ix + 1, ix]))


def nearest_wet(wet, j, i):
    """First wet cell in the growing block around (j, i), scanned x-major as f90:591-614; None if none."""
    ny, nx = wet.shape
    for m in range(1, max(i, nx - 1 - i, j, ny - 1 - j) + 1):
        i0, i1 = max(i - m, 0), min(i + m, nx - 1)
        j0, j1 = max(j - m, 0), min(j + m, ny - 1)
        sub = wet[j0:j1 + 1, i0:i1 + 1]
        if sub.any():
            k = int(np.argmax(sub.T.ravel()))
            return j0 + k % sub.shape[0], i0 + k // sub.shape[0]
    return None


def dry_parents(wet, cj, ci):
    """Replace dry (cj, ci) cells by their nearest wet cell; returns new index arrays and the dry count."""
    cj, ci = cj.copy(), ci.copy()
    dry = ~wet[cj, ci]
    cache = {}
    for k in np.flatnonzero(dry.ravel()):
        key = (int(cj.flat[k]), int(ci.flat[k]))
        if key not in cache:
            p = nearest_wet(wet, *key)
            if p is None:
                raise ValueError("no wet cell on the background grid")
            cache[key] = p
        cj.flat[k], ci.flat[k] = cache[key]
    return cj, ci, len(cache)


def fix_ssh(s, wet):
    """Later-record SSH repair for wet cells (f90:774-786); returns (field, n_fixed)."""
    bad = wet & ~(s >= DRY_SSH)
    n = int(bad.sum())
    if n == 0:
        return s, 0
    ny, nx = s.shape
    out = s.copy()
    for ii in (1, 2):
        nb = np.full((ny, nx), np.nan)
        for jj in (2, 1):
            cand = np.full((ny, nx), np.nan)
            if jj < ny and ii < nx:
                cand[:ny - jj, :nx - ii] = s[jj:, ii:]
            nb = np.where(cand > RJUNK, cand, nb)
        # the exit in the ops loop only leaves the jj loop, so ii=2 overwrites ii=1
        out = np.where(bad & np.isfinite(nb), nb, out)
    out[bad & ~(out >= DRY_SSH)] = 0.0
    return out, n


def fill_columns(T, S, U, V):
    """Bottom-first (n, nz) columns: bottom extension and junk-in-middle fill (f90:493-558); returns T, S, klev0, n_mid."""
    nz = T.shape[1]
    thr = RJUNK + JUNK_EPS
    valid = [a > thr for a in (T, S, U, V)]
    first = np.stack([np.where(v.any(1), v.argmax(1), -1) for v in valid])
    if (first < 0).any():
        raise ValueError("wet column with no valid level")
    klev0 = first.max(0)
    row = np.arange(T.shape[0])
    below = np.arange(nz)[None, :] < klev0[:, None]
    out, n_mid = [], 0
    for a, v in ((T, valid[0]), (S, valid[1])):
        mid = ~v & ~below
        if mid[row, klev0].any():
            raise ValueError("bottom T/S is junk")
        n_mid += int(mid.sum())
        out.append(np.where(mid | below, a[row, klev0][:, None], a))
    return out[0], out[1], klev0, n_mid


def vertical_index(z, zm, kbp0, dry):
    """Lower level and ratio per node/level from the lower-left corner only (f90:932-964)."""
    nz = zm.size
    lev = np.clip(np.searchsorted(zm, z, side="left") - 1, 0, nz - 2)
    vrat = (z - zm[lev]) / (zm[lev + 1] - zm[lev])
    top = z >= zm[-1]
    lev = np.where(top, nz - 2, lev)
    vrat = np.where(top, 1.0, vrat)
    below = z <= zm[np.maximum(kbp0, 0)][:, None]
    lev = np.where(below, 0, lev)
    vrat = np.where(below, 0.0, vrat)
    lev = np.where(dry[:, None], nz - 2, lev)
    vrat = np.where(dry[:, None], 1.0, vrat)
    return lev, np.clip(vrat, 0.0, 1.0)
