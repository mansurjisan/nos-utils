"""
HRRR + GFS atmospheric forcing blender.

Blends high-resolution HRRR (3km, CONUS) with global GFS (0.25°) on a
regular lat/lon grid via Delaunay triangulation. HRRR overrides GFS
where HRRR has spatial coverage.

Input:
  - GFS sflux files (sflux_air_1.*.nc) — primary, global coverage
  - HRRR sflux files (sflux_air_2.*.nc) — secondary, CONUS only

Output:
  - datm_forcing.nc — blended forcing on regular grid for UFS-Coastal DATM

Wind rotation: HRRR Lambert Conformal winds are grid-relative and must
be rotated to earth-relative. Formula uses LoV and LaD from the LC projection.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import ForcingConfig
from .base import ForcingProcessor, ForcingResult

log = logging.getLogger(__name__)

try:
    from netCDF4 import Dataset
    HAS_NETCDF4 = True
except ImportError:
    HAS_NETCDF4 = False

# HRRR Lambert Conformal projection parameters
HRRR_LOV = -97.5   # Longitude of vertical (degrees)
HRRR_LAD = 38.5    # Latitude of tangency (degrees)

# DATM variable mapping
BLEND_VARIABLES = [
    "uwind", "vwind", "stmp", "spfh", "prmsl", "prate", "dswrf", "dlwrf",
]

# DATM naming convention
SFLUX_TO_DATM = {
    "uwind": "UGRD_10maboveground",
    "vwind": "VGRD_10maboveground",
    "stmp": "TMP_2maboveground",
    "spfh": "SPFH_2maboveground",
    "prmsl": "MSLMA_meansealevel",
    "prate": "PRATE_surface",
    "dswrf": "DSWRF_surface",
    "dlwrf": "DLWRF_surface",
}


class BlenderProcessor(ForcingProcessor):
    """
    Blend HRRR + GFS sflux files into a single DATM forcing file.

    Uses Delaunay triangulation for HRRR (Lambert Conformal) to regular
    lat/lon interpolation. HRRR overrides GFS where coverage exists.
    """

    SOURCE_NAME = "BLENDER"

    def __init__(
        self,
        config: ForcingConfig,
        input_path: Path,
        output_path: Path,
        target_dx: float = 0.025,
    ):
        """
        Args:
            config: ForcingConfig with domain bounds
            input_path: Directory containing sflux_air_1.*.nc (GFS) and sflux_air_2.*.nc (HRRR)
            output_path: Output directory for datm_forcing.nc
            target_dx: Output grid resolution in degrees (default 0.025°)
        """
        super().__init__(config, input_path, output_path)
        self.target_dx = target_dx

    def process(self) -> ForcingResult:
        """Blend GFS + HRRR sflux files into datm_forcing.nc."""
        if not HAS_NETCDF4:
            return ForcingResult(
                success=False, source=self.SOURCE_NAME,
                errors=["netCDF4 required for blending"],
            )

        log.info(f"Blender: dx={self.target_dx}°")
        self.create_output_dir()

        # Find sflux files
        gfs_files = sorted(self.input_path.glob("sflux_air_1.*.nc"))
        hrrr_files = sorted(self.input_path.glob("sflux_air_2.*.nc"))

        if not gfs_files:
            return ForcingResult(
                success=False, source=self.SOURCE_NAME,
                errors=["No GFS sflux files found for blending"],
            )

        log.info(f"GFS files: {len(gfs_files)}, HRRR files: {len(hrrr_files)}")

        # Build target grid
        lon_min, lon_max, lat_min, lat_max = self.config.domain
        target_lons = np.arange(lon_min, lon_max + self.target_dx, self.target_dx)
        target_lats = np.arange(lat_min, lat_max + self.target_dx, self.target_dx)

        # Load GFS data
        gfs_data, gfs_times, gfs_lons, gfs_lats = self._load_sflux_stack(gfs_files)

        # Load HRRR data (if available)
        hrrr_data = None
        hrrr_interp = None
        if hrrr_files:
            hrrr_data, hrrr_times, hrrr_lons_2d, hrrr_lats_2d = self._load_sflux_stack(
                hrrr_files, return_2d_coords=True
            )
            if hrrr_data and hrrr_lons_2d is not None:
                hrrr_interp = self._build_interpolator(
                    hrrr_lons_2d, hrrr_lats_2d, target_lons, target_lats
                )

        # Blend onto target grid
        output_file = self.output_path / "datm_forcing.nc"
        self._write_blended(
            gfs_data, gfs_times, gfs_lons, gfs_lats,
            hrrr_data, hrrr_interp,
            target_lons, target_lats, output_file,
        )

        return ForcingResult(
            success=True, source=self.SOURCE_NAME,
            output_files=[output_file],
            metadata={
                "gfs_files": len(gfs_files),
                "hrrr_files": len(hrrr_files),
                "target_dx": self.target_dx,
                "nx": len(target_lons),
                "ny": len(target_lats),
                "has_hrrr_blend": hrrr_interp is not None,
            },
        )

    def find_input_files(self) -> List[Path]:
        return sorted(self.input_path.glob("sflux_air_*.*.nc"))

    def _load_sflux_stack(
        self, files: List[Path], return_2d_coords: bool = False,
    ) -> tuple:
        """Load and concatenate sflux files."""
        all_data = {var: [] for var in BLEND_VARIABLES}
        all_times = []
        lons = lats = None

        for f in files:
            ds = Dataset(str(f))
            time_vals = ds.variables["time"][:]
            lon_arr = ds.variables["lon"][:]
            lat_arr = ds.variables["lat"][:]

            if lons is None:
                lons = lon_arr
                lats = lat_arr

            for var in BLEND_VARIABLES:
                if var in ds.variables:
                    all_data[var].append(ds.variables[var][:])

            # Also load from rad/prc files (same directory, same numbering)
            base = f.name.replace("air", "{type}")
            for ftype, fvars in [("rad", ["dlwrf", "dswrf"]), ("prc", ["prate"])]:
                companion = f.parent / f.name.replace("air", ftype)
                if companion.exists():
                    ds_c = Dataset(str(companion))
                    for var in fvars:
                        if var in ds_c.variables and var not in ds.variables:
                            all_data[var].append(ds_c.variables[var][:])
                    ds_c.close()

            all_times.extend(time_vals.tolist())
            ds.close()

        # Concatenate time axis
        for var in all_data:
            if all_data[var]:
                all_data[var] = np.concatenate(all_data[var], axis=0)
            else:
                del all_data[var]

        if return_2d_coords:
            return all_data, all_times, lons, lats
        else:
            # Extract 1D coords
            if lons is not None and lons.ndim == 2:
                lons_1d = lons[0, :]
                lats_1d = lats[:, 0]
            else:
                lons_1d = lons
                lats_1d = lats
            return all_data, all_times, lons_1d, lats_1d

    def _build_interpolator(
        self, hrrr_lons_2d, hrrr_lats_2d, target_lons, target_lats,
    ) -> Optional[dict]:
        """Build Delaunay triangulation for HRRR -> target grid."""
        try:
            from scipy.spatial import Delaunay
        except ImportError:
            log.warning("scipy not available — no HRRR blending")
            return None

        target_lon_2d, target_lat_2d = np.meshgrid(target_lons, target_lats)
        ny, nx = target_lon_2d.shape

        hrrr_pts = np.column_stack([hrrr_lons_2d.ravel(), hrrr_lats_2d.ravel()])
        tri = Delaunay(hrrr_pts)

        target_pts = np.column_stack([target_lon_2d.ravel(), target_lat_2d.ravel()])
        simplices = tri.find_simplex(target_pts)
        coverage = simplices >= 0

        # Barycentric coordinates
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
        log.info(f"HRRR coverage: {n_cov}/{len(target_pts)} points ({100*n_cov/len(target_pts):.1f}%)")

        return {
            "tri_vi": tri_vi,
            "bary": bary,
            "valid_idx": np.where(coverage)[0],
            "coverage_2d": coverage.reshape(ny, nx),
            "hrrr_shape": hrrr_lons_2d.shape,
        }

    def _write_blended(
        self, gfs_data, gfs_times, gfs_lons, gfs_lats,
        hrrr_data, hrrr_interp,
        target_lons, target_lats, output_path,
    ):
        """Write blended datm_forcing.nc."""
        from scipy.interpolate import RegularGridInterpolator

        ny = len(target_lats)
        nx = len(target_lons)
        nt = len(gfs_times)
        target_lon_2d, target_lat_2d = np.meshgrid(target_lons, target_lats)

        nc = Dataset(str(output_path), "w", format="NETCDF4")
        nc.createDimension("time", None)
        nc.createDimension("latitude", ny)
        nc.createDimension("longitude", nx)

        time_var = nc.createVariable("time", "f8", ("time",))
        time_var.units = "seconds since 1970-01-01 00:00:00"
        time_var[:] = gfs_times

        lon_var = nc.createVariable("longitude", "f4", ("longitude",))
        lon_var.units = "degrees_east"
        lon_var[:] = target_lons

        lat_var = nc.createVariable("latitude", "f4", ("latitude",))
        lat_var.units = "degrees_north"
        lat_var[:] = target_lats

        for sflux_name in BLEND_VARIABLES:
            if sflux_name not in gfs_data:
                continue

            datm_name = SFLUX_TO_DATM[sflux_name]
            var = nc.createVariable(datm_name, "f4", ("time", "latitude", "longitude"))

            gfs_field = gfs_data[sflux_name]

            for t in range(min(nt, gfs_field.shape[0])):
                # Interpolate GFS to target grid
                try:
                    interp = RegularGridInterpolator(
                        (gfs_lats, gfs_lons), gfs_field[t],
                        method="linear", bounds_error=False, fill_value=np.nan,
                    )
                    field = interp(np.column_stack([
                        target_lat_2d.ravel(), target_lon_2d.ravel()
                    ])).reshape(ny, nx).astype(np.float32)
                except Exception:
                    field = np.full((ny, nx), np.nan, dtype=np.float32)

                # Override with HRRR where coverage exists
                if hrrr_interp and hrrr_data and sflux_name in hrrr_data:
                    hrrr_field = hrrr_data[sflux_name]
                    if t < hrrr_field.shape[0]:
                        hrrr_flat = hrrr_field[t].ravel()
                        tri_vi = hrrr_interp["tri_vi"]
                        bary = hrrr_interp["bary"]
                        valid_idx = hrrr_interp["valid_idx"]
                        coverage = hrrr_interp["coverage_2d"]

                        vals = (hrrr_flat[tri_vi[:, 0]] * bary[:, 0] +
                                hrrr_flat[tri_vi[:, 1]] * bary[:, 1] +
                                hrrr_flat[tri_vi[:, 2]] * bary[:, 2])

                        blended = field.ravel()
                        blended[valid_idx] = vals
                        field = blended.reshape(ny, nx)

                var[t, :, :] = field

        # Data source mask
        if hrrr_interp:
            src = nc.createVariable("data_source", "i4", ("latitude", "longitude"))
            src.long_name = "1=HRRR coverage, 0=GFS only"
            src[:] = hrrr_interp["coverage_2d"].astype(np.int32)

        nc.title = "Blended HRRR+GFS forcing for UFS-Coastal DATM"
        nc.close()
        log.info(f"Created {output_path.name}: nx={nx}, ny={ny}, nt={nt}")


def rotate_winds_lcc(u_grid, v_grid, lon_2d, lov=HRRR_LOV, lad=HRRR_LAD):
    """
    Rotate grid-relative winds to earth-relative for Lambert Conformal.

    HRRR GRIB2 winds are relative to the LC grid. DATM/SCHISM need
    earth-relative (true north) winds.

    Args:
        u_grid, v_grid: Grid-relative wind components
        lon_2d: 2D longitude array (degrees)
        lov: Longitude of vertical (degrees)
        lad: Latitude of tangency (degrees)

    Returns:
        (u_earth, v_earth) rotated to true north
    """
    angle = np.radians(np.sin(np.radians(lad)) * (lon_2d - lov))
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    u_earth = u_grid * cos_a - v_grid * sin_a
    v_earth = u_grid * sin_a + v_grid * cos_a

    return u_earth, v_earth
