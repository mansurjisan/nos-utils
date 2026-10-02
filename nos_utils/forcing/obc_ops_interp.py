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
        return _mode1(xax, yax, x, y)
    w[~inside] = 0.0
    return ix, iy, w, inside


def _signa(x1, x2, x3, y1, y2, y3):
    two = np.result_type(x1, y1).type(2)
    return np.abs(((x1 - x3) * (y2 - y3) - (x2 - x3) * (y1 - y3)) / two)


def _accepted(X, Y, px, py, ci, small1):
    """Area-test acceptance (f90:709-722) of points (px, py) against cells ci."""
    c = [(X[m][ci], Y[m][ci]) for m in range(4)]
    a1 = _signa(px, c[0][0], c[1][0], py, c[0][1], c[1][1])
    a2 = _signa(px, c[1][0], c[2][0], py, c[1][1], c[2][1])
    a3 = _signa(px, c[2][0], c[3][0], py, c[2][1], c[3][1])
    a4 = _signa(px, c[3][0], c[0][0], py, c[3][1], c[0][1])
    b1 = _signa(c[0][0], c[1][0], c[2][0], c[0][1], c[1][1], c[2][1])
    b2 = _signa(c[0][0], c[2][0], c[3][0], c[0][1], c[2][1], c[3][1])
    return np.abs(a1 + a2 + a3 + a4 - b1 - b2) / (b1 + b2) < small1


def parent_weights_2d(lon, lat, x, y, small1=1e-2, chunk=4000, single=False):
    """ops interp_mode=1 on a general 2-D quad grid (f90:709-760); lon/lat are [j, i].

    Returns ix, iy, w [LL, LR, UR, UL], found. The winner is the first accepted cell in the
    f90 scan order (ix outer, iy inner); w is clamped as in the f90. single=True evaluates
    the area tests and weights in float32 on float32 coordinates, as the f90's real*4.
    """
    from scipy.spatial import cKDTree

    dt = np.float32 if single else np.float64
    lon = np.asarray(lon, dt)
    lat = np.asarray(lat, dt)
    x = np.atleast_1d(np.asarray(x, dt))
    y = np.atleast_1d(np.asarray(y, dt))
    small1 = dt(small1)
    if not (np.diff(lon, axis=1) > 0).all() or not (np.diff(lat, axis=0) > 0).all():
        raise ValueError("lon must increase along i and lat along j")
    ny, nx = lon.shape
    X = [lon[:-1, :-1], lon[:-1, 1:], lon[1:, 1:], lon[1:, :-1]]
    Y = [lat[:-1, :-1], lat[:-1, 1:], lat[1:, 1:], lat[1:, :-1]]
    ncx = nx - 1
    X = [a.ravel() for a in X]
    Y = [a.ravel() for a in Y]
    diag = np.maximum(np.hypot(X[0] - X[2], Y[0] - Y[2]), np.hypot(X[1] - X[3], Y[1] - Y[3]))
    cen = np.column_stack([sum(X) / 4, sum(Y) / 4])
    tree = cKDTree(cen)
    k = min(25, len(diag))
    n = x.size
    key = np.full(n, np.iinfo(np.int64).max)
    for s0 in range(0, n, chunk):
        pts = np.column_stack([x[s0:s0 + chunk], y[s0:s0 + chunk]])
        # A point passing the 1% area test lies within ~0.02 diag of the cell (area <= diag *
        # min width), so it is within ~1.02 diag of the centre. The radius is 3x the largest
        # diagonal among the 25 nearest cells, which covers it with margin. MJ (10/02/26)
        r = 3.0 * diag[tree.query(pts, k=k)[1].reshape(len(pts), -1)].max(axis=1)
        lists = tree.query_ball_point(pts, r)
        lens = np.fromiter((len(a) for a in lists), int, len(lists))
        if lens.sum() == 0:
            continue
        pi = np.repeat(np.arange(len(pts)), lens)
        ci = np.concatenate([np.asarray(a, int) for a in lists if len(a)])
        acc = _accepted(X, Y, pts[pi, 0], pts[pi, 1], ci, small1)
        if acc.any():
            kk = (ci[acc] % ncx) * (ny - 1) + ci[acc] // ncx
            np.minimum.at(key, s0 + pi[acc], kk)
    found = key < np.iinfo(np.int64).max
    # A node inside the grid's bounding box with no KD candidate accepted: scan every cell in the
    # f90 order. MJ (10/02/26)
    allc = np.arange(len(diag))
    for k_ in np.flatnonzero(~found & (x >= lon.min()) & (x <= lon.max())
                             & (y >= lat.min()) & (y <= lat.max())):
        ok = np.flatnonzero(_accepted(X, Y, np.full(allc.size, x[k_]), np.full(allc.size, y[k_]),
                                      allc, small1))
        if ok.size:
            key[k_] = ((ok % ncx) * (ny - 1) + ok // ncx).min()
    found = key < np.iinfo(np.int64).max
    kf = np.where(found, key, 0)
    ix, iy = kf // (ny - 1), kf % (ny - 1)
    ci = iy * ncx + ix
    x1, x2, x3, x4 = (X[m][ci] for m in range(4))
    y1, y2, y3, y4 = (Y[m][ci] for m in range(4))
    a1 = _signa(x, x1, x2, y, y1, y2)
    a2 = _signa(x, x2, x3, y, y2, y3)
    a3 = _signa(x, x3, x4, y, y3, y4)
    a4 = _signa(x, x4, x1, y, y4, y1)
    ap = _signa(x, x1, x3, y, y1, y3)
    bb1 = _signa(x1, x2, x3, y1, y2, y3)
    bb2 = _signa(x1, x3, x4, y1, y3, y4)
    five = dt(5)
    tri1 = np.abs(a1 + a2 + ap - bb1) / bb1 < five * small1
    tri2 = np.abs(a3 + a4 + ap - bb2) / bb2 < five * small1
    if (found & ~tri1 & ~tri2).any():
        raise ValueError("cannot find a triangle")
    zero, one = dt(0), dt(1)
    cl = lambda v: np.clip(v, zero, one)  # noqa: E731
    w = np.zeros((n, 4), dt)
    w[:, 0] = np.where(tri1, cl(a2 / bb1), cl(a3 / bb2))
    w[:, 1] = np.where(tri1, cl(ap / bb1), zero)
    w[:, 2] = np.where(tri1, cl(one - w[:, 0] - w[:, 1]), cl(a4 / bb2))
    w[:, 3] = np.where(tri1, zero, cl(one - w[:, 0] - w[:, 2]))
    w[~found] = 0.0
    return ix, iy, w, found


def _mode1(xax, yax, x, y):
    LO, LA = np.meshgrid(xax, yax)
    return parent_weights_2d(LO, LA, x, y)


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


def interpolate4(s, w):
    """Corner values s (n, 4) combined with weights w (n, 4), summed left to right as the f90's eout."""
    return ((s[:, 0] * w[:, 0] + s[:, 1] * w[:, 1]) + s[:, 2] * w[:, 2]) + s[:, 3] * w[:, 3]


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
    # Deliberate deviation: in float32 the -30000 fill unpacks to exactly RJUNK, so the f90's
    # `< rjunk` mid-column test never fires and ops would abort at its sanity stop; here such
    # levels are filled with the bottom value instead. MJ (10/02/26)
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
