"""
DATM (Data Atmosphere) NetCDF writer for UFS-Coastal.

Creates datm_forcing.nc for CDEPS DATM in-memory coupling with SCHISM (nws=4).
Supports both direct write from extracted arrays and blended GFS+HRRR output.

Output dimensions: (time, latitude, longitude) on a regular lat/lon grid.
CRITICAL: nx_global/ny_global in datm_in must match actual file dimensions (lesson #19).
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

try:
    from netCDF4 import Dataset
    HAS_NETCDF4 = True
except ImportError:
    HAS_NETCDF4 = False


# DATM variable naming: DATM field name -> internal sflux variable name
VAR_MAP = {
    "UGRD_10maboveground": "uwind",
    "VGRD_10maboveground": "vwind",
    "TMP_2maboveground": "stmp",
    "SPFH_2maboveground": "spfh",
    "MSLMA_meansealevel": "prmsl",
    "PRATE_surface": "prate",
    "DSWRF_surface": "dswrf",
    "DLWRF_surface": "dlwrf",
}

# Reverse map: sflux name -> DATM name
SFLUX_TO_DATM = {v: k for k, v in VAR_MAP.items()}


class DATMWriter:
    """
    Creates datm_forcing.nc for UFS-Coastal CDEPS DATM component.

    Usage:
        writer = DATMWriter()
        path = writer.write(data, times, lons, lats, output_path)

        # With GFS+HRRR blending:
        path = writer.write_blended(gfs_data, hrrr_data, ..., output_path)
    """

    def __init__(self):
        if not HAS_NETCDF4:
            raise ImportError("netCDF4 required for DATM output. Install with: pip install netCDF4")

    def write(
        self,
        data: Dict[str, List[np.ndarray]],
        times: List[datetime],
        lons: np.ndarray,
        lats: np.ndarray,
        output_path: Path,
        epoch: Optional[datetime] = None,
    ) -> Path:
        """
        Write datm_forcing.nc from extracted arrays (single source, no blending).

        Args:
            data: Dict mapping sflux variable names to lists of 2D arrays
            times: List of datetime objects
            lons: 1D longitude array
            lats: 1D latitude array
            output_path: Path for output file
            epoch: Reference datetime for time axis (default: 1970-01-01)

        Returns:
            Path to created file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if epoch is None:
            epoch = datetime(1970, 1, 1)

        nx = len(lons)
        ny = len(lats)

        nc = Dataset(str(output_path), "w", format="NETCDF4")

        # Dimensions
        nc.createDimension("time", None)  # unlimited
        nc.createDimension("latitude", ny)
        nc.createDimension("longitude", nx)

        # Time variable (seconds since epoch)
        time_var = nc.createVariable("time", "f8", ("time",))
        time_var.units = f"seconds since {epoch.strftime('%Y-%m-%d')} 00:00:00"
        time_var.calendar = "standard"
        time_var.long_name = "Time"
        time_var[:] = [(t - epoch).total_seconds() for t in times]

        # Coordinate variables
        lon_var = nc.createVariable("longitude", "f4", ("longitude",))
        lon_var.units = "degrees_east"
        lon_var.long_name = "Longitude"
        lon_var[:] = lons

        lat_var = nc.createVariable("latitude", "f4", ("latitude",))
        lat_var.units = "degrees_north"
        lat_var.long_name = "Latitude"
        lat_var[:] = lats

        # Data variables with DATM naming convention
        for sflux_name, arrays in data.items():
            if sflux_name not in SFLUX_TO_DATM:
                continue
            if not arrays:
                continue

            datm_name = SFLUX_TO_DATM[sflux_name]
            var = nc.createVariable(
                datm_name, "f4", ("time", "latitude", "longitude"),
                fill_value=-9999.0,
            )
            try:
                var[:] = np.stack(arrays, axis=0)
            except Exception as e:
                log.warning(f"Could not write {datm_name}: {e}")

        nc.title = "DATM forcing for UFS-Coastal"
        nc.conventions = "CF-1.6"
        nc.close()

        log.info(f"Created DATM forcing: {output_path} (nx={nx}, ny={ny}, nt={len(times)})")
        return output_path

    def write_blended(
        self,
        gfs_data: Dict[str, List[np.ndarray]],
        hrrr_data: Optional[Dict[str, List[np.ndarray]]],
        gfs_times: List[datetime],
        hrrr_times: Optional[List[datetime]],
        target_lons: np.ndarray,
        target_lats: np.ndarray,
        hrrr_lons_2d: Optional[np.ndarray],
        hrrr_lats_2d: Optional[np.ndarray],
        output_path: Path,
        epoch: Optional[datetime] = None,
    ) -> Path:
        """
        Write datm_forcing.nc with GFS+HRRR blending.

        HRRR overrides GFS where HRRR has spatial coverage.
        Uses Delaunay triangulation for HRRR LCC -> regular grid interpolation.

        Args:
            gfs_data: GFS extracted data (on regular lat/lon grid)
            hrrr_data: HRRR extracted data (on LCC grid, or None)
            gfs_times: GFS time steps
            hrrr_times: HRRR time steps (or None)
            target_lons: 1D target longitude array
            target_lats: 1D target latitude array
            hrrr_lons_2d: 2D HRRR longitude array (or None)
            hrrr_lats_2d: 2D HRRR latitude array (or None)
            output_path: Path for output file
            epoch: Reference datetime

        Returns:
            Path to created file
        """
        if hrrr_data is None or hrrr_times is None:
            # No HRRR — fall through to GFS-only write
            return self.write(gfs_data, gfs_times, target_lons, target_lats, output_path, epoch)

        try:
            from scipy.spatial import Delaunay
        except ImportError:
            log.warning("scipy not available — writing GFS-only DATM (no HRRR blending)")
            return self.write(gfs_data, gfs_times, target_lons, target_lats, output_path, epoch)

        # Build HRRR interpolation index
        hrrr_interp = self._build_hrrr_interpolator(
            hrrr_lons_2d, hrrr_lats_2d, target_lons, target_lats
        )

        # Merge time axes (use GFS as base, override with HRRR where available)
        blended_data = {}
        target_lon_2d, target_lat_2d = np.meshgrid(target_lons, target_lats)
        ny, nx = target_lon_2d.shape

        for sflux_name in SFLUX_TO_DATM:
            if sflux_name not in gfs_data:
                continue

            blended_arrays = []
            for t_idx, t in enumerate(gfs_times):
                # Start with GFS
                if t_idx < len(gfs_data[sflux_name]):
                    field = gfs_data[sflux_name][t_idx].copy()
                else:
                    continue

                # Ensure field matches target grid shape
                if field.shape != (ny, nx):
                    # GFS may need interpolation to target grid
                    field = np.resize(field, (ny, nx))

                # Override with HRRR where available
                if sflux_name in hrrr_data and hrrr_interp is not None:
                    hrrr_field = self._interpolate_hrrr(
                        hrrr_data, sflux_name, t, hrrr_times, hrrr_interp, ny, nx
                    )
                    if hrrr_field is not None:
                        coverage = hrrr_interp["coverage_2d"]
                        field[coverage] = hrrr_field[coverage]

                blended_arrays.append(field)

            blended_data[sflux_name] = blended_arrays

        return self.write(blended_data, gfs_times, target_lons, target_lats, output_path, epoch)

    def _build_hrrr_interpolator(
        self,
        hrrr_lons_2d: np.ndarray,
        hrrr_lats_2d: np.ndarray,
        target_lons: np.ndarray,
        target_lats: np.ndarray,
    ) -> Optional[dict]:
        """Build Delaunay triangulation for HRRR LCC -> regular grid."""
        try:
            from scipy.spatial import Delaunay
        except ImportError:
            return None

        target_lon_2d, target_lat_2d = np.meshgrid(target_lons, target_lats)
        ny, nx = target_lon_2d.shape

        hrrr_pts = np.column_stack([hrrr_lons_2d.ravel(), hrrr_lats_2d.ravel()])
        tri = Delaunay(hrrr_pts)

        target_pts = np.column_stack([target_lon_2d.ravel(), target_lat_2d.ravel()])
        simplices = tri.find_simplex(target_pts)
        coverage = simplices >= 0

        # Precompute barycentric coordinates for covered points
        valid_s = simplices[coverage]
        valid_t = target_pts[coverage]
        tri_vi = tri.simplices[valid_s]
        v0 = hrrr_pts[tri_vi[:, 0]]
        v1 = hrrr_pts[tri_vi[:, 1]]
        v2 = hrrr_pts[tri_vi[:, 2]]
        det = ((v1[:, 1] - v2[:, 1]) * (v0[:, 0] - v2[:, 0]) +
               (v2[:, 0] - v1[:, 0]) * (v0[:, 1] - v2[:, 1]))
        lam0 = ((v1[:, 1] - v2[:, 1]) * (valid_t[:, 0] - v2[:, 0]) +
                (v2[:, 0] - v1[:, 0]) * (valid_t[:, 1] - v2[:, 1])) / det
        lam1 = ((v2[:, 1] - v0[:, 1]) * (valid_t[:, 0] - v2[:, 0]) +
                (v0[:, 0] - v2[:, 0]) * (valid_t[:, 1] - v2[:, 1])) / det
        bary = np.column_stack([lam0, lam1, 1 - lam0 - lam1]).astype(np.float32)

        n_cov = int(np.sum(coverage))
        log.info(f"HRRR coverage: {n_cov}/{len(target_pts)} target points "
                 f"({100 * n_cov / len(target_pts):.1f}%)")

        return {
            "tri_vi": tri_vi,
            "bary": bary,
            "valid_idx": np.where(coverage)[0],
            "coverage_2d": coverage.reshape(ny, nx),
        }

    def _interpolate_hrrr(
        self,
        hrrr_data: Dict[str, List[np.ndarray]],
        var_name: str,
        target_time: datetime,
        hrrr_times: List[datetime],
        interp_info: dict,
        ny: int, nx: int,
    ) -> Optional[np.ndarray]:
        """Interpolate a single HRRR field to the target grid at a specific time."""
        if var_name not in hrrr_data or not hrrr_data[var_name]:
            return None

        # Find nearest HRRR time
        time_diffs = [abs((t - target_time).total_seconds()) for t in hrrr_times]
        nearest_idx = int(np.argmin(time_diffs))

        # Only use if within 1.5 hours
        if time_diffs[nearest_idx] > 5400:
            return None

        if nearest_idx >= len(hrrr_data[var_name]):
            return None

        hrrr_field = hrrr_data[var_name][nearest_idx].ravel()
        tri_vi = interp_info["tri_vi"]
        bary = interp_info["bary"]
        valid_idx = interp_info["valid_idx"]

        # Barycentric interpolation
        vals = (hrrr_field[tri_vi[:, 0]] * bary[:, 0] +
                hrrr_field[tri_vi[:, 1]] * bary[:, 1] +
                hrrr_field[tri_vi[:, 2]] * bary[:, 2])

        result = np.full(ny * nx, np.nan, dtype=np.float32)
        result[valid_idx] = vals
        return result.reshape(ny, nx)

    @staticmethod
    def get_grid_dims(lons: np.ndarray, lats: np.ndarray) -> Tuple[int, int]:
        """
        Return (nx_global, ny_global) for datm_in configuration.

        CRITICAL: These must match the actual file dimensions (lesson #19).
        """
        return len(lons), len(lats)
