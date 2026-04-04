"""
RTOFS ocean boundary condition processor.

Creates SCHISM boundary time-history files by interpolating RTOFS global
ocean model output to SCHISM open boundary nodes.

Output:
  - elev2D.th.nc  — SSH at boundary nodes (time, nOpenBndNodes)
  - TEM_3D.th.nc  — Temperature at boundary nodes (time, nOpenBndNodes, nLevels)
  - SAL_3D.th.nc  — Salinity at boundary nodes (time, nOpenBndNodes, nLevels)
  - uv3D.th.nc    — Velocity at boundary nodes (time, nOpenBndNodes, nLevels, 2)

Reads SCHISM boundary node locations from hgrid.ll and interpolates RTOFS
data to those specific nodes — matching Fortran gen_3Dth_from_hycom behavior.

Works with any SCHISM-based OFS (SECOFS, STOFS-3D-ATL, CREOFS, etc.)
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import ForcingConfig
from ..io.schism_grid import SchismGrid
from .base import ForcingProcessor, ForcingResult

log = logging.getLogger(__name__)

try:
    from netCDF4 import Dataset
    HAS_NETCDF4 = True
except ImportError:
    HAS_NETCDF4 = False


class RTOFSProcessor(ForcingProcessor):
    """
    RTOFS ocean boundary condition processor for SCHISM.

    Interpolates RTOFS data to SCHISM open boundary nodes from hgrid.ll.
    """

    SOURCE_NAME = "RTOFS"
    MIN_FILE_SIZE_2D = 150_000_000
    MIN_FILE_SIZE_3D = 200_000_000

    def __init__(
        self,
        config: ForcingConfig,
        input_path: Path,
        output_path: Path,
        grid_file: Optional[Path] = None,
    ):
        """
        Args:
            config: ForcingConfig with domain and OBC settings
            input_path: Root RTOFS data directory (COMINrtofs)
            output_path: Output directory for boundary files
            grid_file: SCHISM grid file (hgrid.ll) for boundary node extraction
        """
        super().__init__(config, input_path, output_path)
        self.grid_file = grid_file or config.grid_file
        self._grid = None
        self._bnd_lons = None
        self._bnd_lats = None
        self._bnd_depths = None

    def _load_grid(self) -> bool:
        """Load SCHISM grid and extract boundary node coordinates."""
        if self._bnd_lons is not None:
            return True

        if self.grid_file is None or not Path(self.grid_file).exists():
            log.warning(f"Grid file not found: {self.grid_file}")
            return False

        self._grid = SchismGrid.read(self.grid_file)
        self._bnd_lons, self._bnd_lats, self._bnd_depths, self._bnd_ids = \
            self._grid.open_boundary_nodes()

        log.info(f"Loaded {len(self._bnd_lons)} boundary nodes from {Path(self.grid_file).name}")
        return len(self._bnd_lons) > 0

    def process(self) -> ForcingResult:
        if not HAS_NETCDF4:
            return ForcingResult(
                success=False, source=self.SOURCE_NAME,
                errors=["netCDF4 required for RTOFS processing"],
            )

        log.info(f"RTOFS processor: pdy={self.config.pdy} cyc={self.config.cyc:02d}z")
        self.create_output_dir()

        # Load SCHISM grid for boundary nodes
        has_grid = self._load_grid()
        if not has_grid:
            return ForcingResult(
                success=False, source=self.SOURCE_NAME,
                errors=[f"Cannot load boundary nodes from grid: {self.grid_file}"],
            )

        # Find RTOFS files
        files_2d, files_3d = self.find_input_files_by_type()
        if not files_2d and not files_3d:
            return ForcingResult(
                success=False, source=self.SOURCE_NAME,
                errors=["No RTOFS input files found"],
            )

        output_files = []
        warnings = []

        # Process 2D (SSH → elev2D.th.nc)
        if files_2d:
            log.info(f"Processing {len(files_2d)} RTOFS 2D files")
            f = self._process_2d(files_2d)
            if f:
                output_files.append(f)
            else:
                warnings.append("Failed to create elev2D.th.nc")

        # Process 3D (T,S,U,V → TEM_3D.th.nc, SAL_3D.th.nc, uv3D.th.nc)
        if files_3d:
            log.info(f"Processing {len(files_3d)} RTOFS 3D files")
            obc_files = self._process_3d(files_3d)
            output_files.extend(obc_files)

        return ForcingResult(
            success=len(output_files) > 0,
            source=self.SOURCE_NAME,
            output_files=output_files,
            warnings=warnings,
            metadata={
                "n_boundary_nodes": len(self._bnd_lons),
                "n_2d_files": len(files_2d),
                "n_3d_files": len(files_3d),
                "n_levels": self.config.n_levels,
            },
        )

    def find_input_files(self) -> List[Path]:
        files_2d, files_3d = self.find_input_files_by_type()
        return files_2d + files_3d

    def find_input_files_by_type(self) -> Tuple[List[Path], List[Path]]:
        """Find RTOFS 2D and 3D files."""
        base_date = datetime.strptime(self.config.pdy, "%Y%m%d")
        files_2d = []
        files_3d = []

        for date in [base_date, base_date - timedelta(days=1)]:
            date_str = date.strftime("%Y%m%d")

            for rtofs_dir in [
                self.input_path / f"rtofs.{date_str}",
                self.input_path / date_str,
                self.input_path,
            ]:
                if not rtofs_dir.exists():
                    continue

                found_2d = sorted(rtofs_dir.glob("rtofs_glo_2ds_*_diag.nc"))
                found_3d = sorted(rtofs_dir.glob("rtofs_glo_3dz_*_6hrly_hvr_*.nc"))
                found_3d.extend(sorted(rtofs_dir.glob("rtofs_glo_3dz_*_6hrly_hvr_*.nc4")))

                for f in found_2d:
                    if self.validate_file_size(f, self.MIN_FILE_SIZE_2D):
                        files_2d.append(f)
                for f in found_3d:
                    if self.validate_file_size(f, self.MIN_FILE_SIZE_3D):
                        files_3d.append(f)

                if files_2d or files_3d:
                    log.info(f"Found RTOFS files in {rtofs_dir}: {len(files_2d)} 2D, {len(files_3d)} 3D")
                    break

            if files_2d or files_3d:
                break

        return files_2d, files_3d

    def _interpolate_2d_to_boundary(
        self, rtofs_lon: np.ndarray, rtofs_lat: np.ndarray, rtofs_data: np.ndarray,
    ) -> np.ndarray:
        """
        Interpolate a 2D RTOFS field to boundary node locations.

        Uses scipy RegularGridInterpolator for speed on regular grids,
        falls back to nearest-neighbor if scipy unavailable.
        """
        n_bnd = len(self._bnd_lons)

        # RTOFS uses 0-360 longitude; boundary nodes use -180 to 180
        bnd_lons_360 = np.where(self._bnd_lons < 0, self._bnd_lons + 360, self._bnd_lons)

        try:
            from scipy.interpolate import RegularGridInterpolator

            # Extract 1D axes (assume regular grid)
            if rtofs_lon.ndim == 2:
                lon_1d = rtofs_lon[0, :]
                lat_1d = rtofs_lat[:, 0]
            else:
                lon_1d = rtofs_lon
                lat_1d = rtofs_lat

            interp = RegularGridInterpolator(
                (lat_1d, lon_1d), rtofs_data,
                method="linear", bounds_error=False, fill_value=np.nan,
            )

            points = np.column_stack([self._bnd_lats, bnd_lons_360])
            result = interp(points).astype(np.float32)

            # Fill NaN with nearest valid value
            nan_mask = np.isnan(result)
            if np.any(nan_mask):
                from scipy.interpolate import NearestNDInterpolator
                valid = ~np.isnan(rtofs_data)
                if np.any(valid):
                    lat_grid, lon_grid = np.meshgrid(lat_1d, lon_1d, indexing="ij")
                    nn = NearestNDInterpolator(
                        np.column_stack([lat_grid[valid], lon_grid[valid]]),
                        rtofs_data[valid],
                    )
                    result[nan_mask] = nn(points[nan_mask])

            return result

        except ImportError:
            # Nearest-neighbor fallback
            result = np.zeros(n_bnd, dtype=np.float32)
            if rtofs_lon.ndim == 2:
                lon_1d = rtofs_lon[0, :]
                lat_1d = rtofs_lat[:, 0]
            else:
                lon_1d = rtofs_lon
                lat_1d = rtofs_lat

            for k in range(n_bnd):
                j = np.argmin(np.abs(lat_1d - self._bnd_lats[k]))
                i = np.argmin(np.abs(lon_1d - bnd_lons_360[k]))
                result[k] = rtofs_data[j, i]

            return result

    def _process_2d(self, files_2d: List[Path]) -> Optional[Path]:
        """Extract SSH from RTOFS 2D files, interpolate to boundary nodes."""
        output_file = self.output_path / "elev2D.th.nc"
        n_bnd = len(self._bnd_lons)

        try:
            all_ssh = []

            for f in files_2d:
                ds = Dataset(str(f))
                ssh_raw = ds.variables["ssh"][:]
                lon = ds.variables["Longitude"][:]
                lat = ds.variables["Latitude"][:]

                ssh_raw = np.ma.filled(ssh_raw, fill_value=np.nan)

                # Interpolate each time step to boundary nodes
                for t in range(ssh_raw.shape[0]):
                    ssh_bnd = self._interpolate_2d_to_boundary(lon, lat, ssh_raw[t])
                    ssh_bnd += self.config.obc_ssh_offset
                    all_ssh.append(ssh_bnd)

                ds.close()

            if not all_ssh:
                return None

            ssh_array = np.stack(all_ssh, axis=0)  # (time, nOpenBndNodes)

            # Write SCHISM format
            nc = Dataset(str(output_file), "w", format="NETCDF4")
            nt = ssh_array.shape[0]
            dt = 21600.0  # 6-hourly

            nc.createDimension("time", nt)
            nc.createDimension("nOpenBndNodes", n_bnd)
            nc.createDimension("nLevels", 1)
            nc.createDimension("nComponents", 1)
            nc.createDimension("one", 1)

            time_var = nc.createVariable("time", "f8", ("time",))
            time_var[:] = [i * dt for i in range(nt)]

            ts = nc.createVariable("time_series", "f4",
                                   ("time", "nOpenBndNodes", "nLevels", "nComponents"),
                                   fill_value=-30000.0)
            ts[:, :, 0, 0] = ssh_array

            nc.close()
            log.info(f"Created elev2D.th.nc: ({nt}, {n_bnd}) boundary nodes")
            return output_file

        except Exception as e:
            log.error(f"Failed to process RTOFS 2D: {e}")
            import traceback
            log.error(traceback.format_exc())
            return None

    def _process_3d(self, files_3d: List[Path]) -> List[Path]:
        """Extract T,S,U,V from RTOFS 3D files, interpolate to boundary nodes."""
        output_files = []
        n_bnd = len(self._bnd_lons)
        n_levels = self.config.n_levels

        try:
            all_temp = []
            all_salt = []
            all_u = []
            all_v = []

            for f in files_3d:
                ds = Dataset(str(f))

                # Get RTOFS grid
                lon = ds.variables.get("Longitude") or ds.variables.get("lon")
                lat = ds.variables.get("Latitude") or ds.variables.get("lat")
                depth = ds.variables.get("Depth") or ds.variables.get("lev")

                if lon is None or lat is None:
                    ds.close()
                    continue

                lon_arr = lon[:]
                lat_arr = lat[:]
                depth_arr = depth[:] if depth is not None else np.arange(n_levels)
                n_rtofs_levels = len(depth_arr)

                for var_name, target_list in [
                    ("temperature", all_temp),
                    ("salinity", all_salt),
                    ("u", all_u),
                    ("v", all_v),
                ]:
                    if var_name not in ds.variables:
                        continue

                    data = ds.variables[var_name][:]
                    data = np.ma.filled(data, fill_value=np.nan)

                    # Take first time step if 4D
                    if data.ndim == 4:
                        data = data[0]  # (depth, y, x)

                    # Interpolate each depth level to boundary nodes
                    bnd_profile = np.full((n_bnd, n_rtofs_levels), np.nan, dtype=np.float32)
                    for lev in range(min(n_rtofs_levels, data.shape[0])):
                        bnd_profile[:, lev] = self._interpolate_2d_to_boundary(
                            lon_arr, lat_arr, data[lev]
                        )

                    target_list.append(bnd_profile)

                ds.close()

            # Write boundary files
            dt = 21600.0  # 6-hourly

            if all_temp:
                fpath = self.output_path / "TEM_3D.th.nc"
                merged = np.stack(all_temp, axis=0)  # (time, nBnd, nLevels)
                self._write_3d_th(fpath, merged, "temperature", "degC", dt, n_bnd)
                output_files.append(fpath)
                log.info(f"Created TEM_3D.th.nc: {merged.shape}")

            if all_salt:
                fpath = self.output_path / "SAL_3D.th.nc"
                merged = np.stack(all_salt, axis=0)
                self._write_3d_th(fpath, merged, "salinity", "PSU", dt, n_bnd)
                output_files.append(fpath)
                log.info(f"Created SAL_3D.th.nc: {merged.shape}")

            if all_u and all_v:
                fpath = self.output_path / "uv3D.th.nc"
                u_merged = np.stack(all_u, axis=0)
                v_merged = np.stack(all_v, axis=0)
                self._write_uv3d_th(fpath, u_merged, v_merged, dt, n_bnd)
                output_files.append(fpath)
                log.info(f"Created uv3D.th.nc: u={u_merged.shape}")

        except Exception as e:
            log.error(f"Failed to process RTOFS 3D: {e}")
            import traceback
            log.error(traceback.format_exc())

        return output_files

    def _write_3d_th(self, output_path: Path, data: np.ndarray,
                     var_name: str, units: str, dt: float, n_bnd: int) -> None:
        """Write TEM_3D.th.nc or SAL_3D.th.nc in SCHISM format."""
        nc = Dataset(str(output_path), "w", format="NETCDF4")
        nt = data.shape[0]
        n_levels = data.shape[2]

        nc.createDimension("time", nt)
        nc.createDimension("nOpenBndNodes", n_bnd)
        nc.createDimension("nLevels", n_levels)
        nc.createDimension("nComponents", 1)
        nc.createDimension("one", 1)

        time_var = nc.createVariable("time", "f8", ("time",))
        time_var[:] = [i * dt for i in range(nt)]

        ts = nc.createVariable("time_series", "f4",
                               ("time", "nOpenBndNodes", "nLevels", "nComponents"),
                               fill_value=-30000.0)
        ts[:, :, :, 0] = data  # (time, nBnd, nLevels)

        nc.close()

    def _write_uv3d_th(self, output_path: Path, u: np.ndarray,
                       v: np.ndarray, dt: float, n_bnd: int) -> None:
        """Write uv3D.th.nc in SCHISM format."""
        nc = Dataset(str(output_path), "w", format="NETCDF4")
        nt = u.shape[0]
        n_levels = u.shape[2]

        nc.createDimension("time", nt)
        nc.createDimension("nOpenBndNodes", n_bnd)
        nc.createDimension("nLevels", n_levels)
        nc.createDimension("nComponents", 2)
        nc.createDimension("one", 1)

        time_var = nc.createVariable("time", "f8", ("time",))
        time_var[:] = [i * dt for i in range(nt)]

        ts = nc.createVariable("time_series", "f4",
                               ("time", "nOpenBndNodes", "nLevels", "nComponents"),
                               fill_value=-30000.0)
        ts[:, :, :, 0] = u
        ts[:, :, :, 1] = v

        nc.close()
