"""
Structured grid bilinear interpolation for curvilinear grids (e.g., RTOFS).

Replaces Delaunay-based interpolation with cell-search + bilinear on the
native grid structure. This eliminates triangulation algorithm differences
and convex-hull edge effects that cause ~3cm SSH bias.

Algorithm:
1. Build KDTree from grid cell centers for fast lookup
2. For each target point, find the nearest cell center
3. Check if the target is inside that cell (or its neighbors)
4. Compute bilinear weights within the enclosing quadrilateral
5. Fall back to nearest-neighbor for points outside the grid
"""

import logging
from typing import Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class StructuredGridInterpolator:
    """
    Bilinear interpolation on a 2D curvilinear structured grid.

    The grid has shape (ny, nx) with 2D lon/lat arrays.
    Grid cells are quadrilaterals defined by 4 corner nodes.
    """

    def __init__(self, lon_2d: np.ndarray, lat_2d: np.ndarray,
                 mask: Optional[np.ndarray] = None):
        """
        Build interpolator from a structured grid.

        Args:
            lon_2d: Longitude array, shape (ny, nx). Will be converted to -180/180.
            lat_2d: Latitude array, shape (ny, nx).
            mask: Boolean ocean mask, shape (ny, nx). True = ocean. If None, all ocean.
        """
        self.ny, self.nx = lon_2d.shape
        self.lon = np.where(lon_2d > 180, lon_2d - 360, lon_2d).astype(np.float64)
        self.lat = lat_2d.astype(np.float64)
        self.mask = mask if mask is not None else np.ones((self.ny, self.nx), dtype=bool)

        # Cell centers for KDTree lookup
        # Cell (j, i) has corners at (j, i), (j, i+1), (j+1, i+1), (j+1, i)
        self.nc_y = self.ny - 1
        self.nc_x = self.nx - 1
        center_lon = 0.25 * (self.lon[:-1, :-1] + self.lon[:-1, 1:] +
                              self.lon[1:, 1:] + self.lon[1:, :-1])
        center_lat = 0.25 * (self.lat[:-1, :-1] + self.lat[:-1, 1:] +
                              self.lat[1:, 1:] + self.lat[1:, :-1])

        # Ocean cell mask: all 4 corners must be ocean
        cell_ocean = (self.mask[:-1, :-1] & self.mask[:-1, 1:] &
                      self.mask[1:, 1:] & self.mask[1:, :-1])

        # Flatten for KDTree
        self._cell_centers = np.column_stack([
            center_lon.ravel(), center_lat.ravel()
        ])
        self._cell_ocean = cell_ocean.ravel()

        # Apply cos(lat) correction for more accurate NN search
        mean_lat = np.radians(np.mean(center_lat))
        corrected = self._cell_centers.copy()
        corrected[:, 0] *= np.cos(mean_lat)
        self._tree = cKDTree(corrected)
        self._cos_lat = np.cos(mean_lat)

        n_ocean = int(np.sum(cell_ocean))
        log.info(f"StructuredGridInterpolator: {self.ny}x{self.nx} grid, "
                 f"{n_ocean}/{self.nc_y * self.nc_x} ocean cells")

    def interpolate(self, target_lon: np.ndarray, target_lat: np.ndarray,
                    data: np.ndarray) -> np.ndarray:
        """
        Interpolate data field to target points using bilinear interpolation.

        Args:
            target_lon: Target longitudes (-180/180), shape (n_targets,)
            target_lat: Target latitudes, shape (n_targets,)
            data: Field values on the grid, shape (ny, nx). NaN/fill for land.

        Returns:
            Interpolated values at target points, shape (n_targets,)
        """
        n_targets = len(target_lon)
        result = np.full(n_targets, np.nan, dtype=np.float32)

        # Find nearest cell for each target
        target_corrected = np.column_stack([
            target_lon * self._cos_lat, target_lat
        ])
        _, nearest_cell = self._tree.query(target_corrected)

        for k in range(n_targets):
            cell_idx = nearest_cell[k]
            j0 = cell_idx // self.nc_x
            i0 = cell_idx % self.nc_x

            # Search this cell and its neighbors for the enclosing cell
            found = False
            for dj in range(0, 3):
                for di in range(0, 3):
                    j = j0 + dj - 1
                    i = i0 + di - 1
                    if j < 0 or j >= self.nc_y or i < 0 or i >= self.nc_x:
                        continue

                    # Get cell corner values
                    d00 = data[j, i]
                    d01 = data[j, i + 1]
                    d11 = data[j + 1, i + 1]
                    d10 = data[j + 1, i]

                    # Skip if any corner is land/NaN
                    if (np.isnan(d00) or np.isnan(d01) or
                        np.isnan(d11) or np.isnan(d10) or
                        abs(d00) >= 99 or abs(d01) >= 99 or
                        abs(d11) >= 99 or abs(d10) >= 99):
                        continue

                    # Check if target is inside this cell using bilinear coords
                    weights = self._bilinear_weights(
                        self.lon[j, i], self.lat[j, i],
                        self.lon[j, i + 1], self.lat[j, i + 1],
                        self.lon[j + 1, i + 1], self.lat[j + 1, i + 1],
                        self.lon[j + 1, i], self.lat[j + 1, i],
                        target_lon[k], target_lat[k],
                    )

                    if weights is not None:
                        s, t = weights
                        result[k] = ((1 - s) * (1 - t) * d00 +
                                     s * (1 - t) * d01 +
                                     s * t * d11 +
                                     (1 - s) * t * d10)
                        found = True
                        break
                if found:
                    break

        # Fill remaining NaN from nearest valid target (matching Fortran REMESH fallback)
        nan_mask = np.isnan(result)
        if np.any(nan_mask) and not np.all(nan_mask):
            valid_idx = np.where(~nan_mask)[0]
            nan_idx = np.where(nan_mask)[0]
            valid_pts = np.column_stack([target_lon[valid_idx] * self._cos_lat,
                                         target_lat[valid_idx]])
            nan_pts = np.column_stack([target_lon[nan_idx] * self._cos_lat,
                                       target_lat[nan_idx]])
            valid_tree = cKDTree(valid_pts)
            _, nn = valid_tree.query(nan_pts)
            result[nan_idx] = result[valid_idx[nn]]
            if np.sum(nan_mask) > 0:
                log.debug(f"Filled {np.sum(nan_mask)} NaN points from nearest valid target")

        return result

    @staticmethod
    def _bilinear_weights(
        x0: float, y0: float,  # corner (0,0)
        x1: float, y1: float,  # corner (1,0)
        x2: float, y2: float,  # corner (1,1)
        x3: float, y3: float,  # corner (0,1)
        xp: float, yp: float,  # target point
    ) -> Optional[Tuple[float, float]]:
        """
        Compute bilinear (s, t) coordinates for point (xp, yp) inside
        the quadrilateral defined by corners (x0,y0), (x1,y1), (x2,y2), (x3,y3).

        Returns (s, t) in [0,1]x[0,1] if inside, None if outside.
        Uses iterative Newton's method for curvilinear quads.
        """
        # Initial guess: center of cell
        s, t = 0.5, 0.5

        for _ in range(20):  # Newton iterations
            # Bilinear mapping: P(s,t) = (1-s)(1-t)*P0 + s(1-t)*P1 + st*P2 + (1-s)t*P3
            px = (1 - s) * (1 - t) * x0 + s * (1 - t) * x1 + s * t * x2 + (1 - s) * t * x3
            py = (1 - s) * (1 - t) * y0 + s * (1 - t) * y1 + s * t * y2 + (1 - s) * t * y3

            # Residual
            fx = px - xp
            fy = py - yp

            if abs(fx) < 1e-10 and abs(fy) < 1e-10:
                break

            # Jacobian
            dxds = -(1 - t) * x0 + (1 - t) * x1 + t * x2 - t * x3
            dxdt = -(1 - s) * x0 - s * x1 + s * x2 + (1 - s) * x3
            dyds = -(1 - t) * y0 + (1 - t) * y1 + t * y2 - t * y3
            dydt = -(1 - s) * y0 - s * y1 + s * y2 + (1 - s) * y3

            det = dxds * dydt - dxdt * dyds
            if abs(det) < 1e-20:
                return None

            # Newton step
            ds = (dydt * fx - dxdt * fy) / det
            dt = (-dyds * fx + dxds * fy) / det
            s -= ds
            t -= dt

        # Check if inside [0,1] x [0,1] with small tolerance
        tol = 0.01
        if -tol <= s <= 1 + tol and -tol <= t <= 1 + tol:
            return (max(0, min(1, s)), max(0, min(1, t)))

        return None
