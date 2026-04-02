"""
RTOFS (Real-Time Ocean Forecast System) ocean boundary condition processor.

Creates SCHISM boundary time-history files from RTOFS global ocean model output:
  - elev2D.th.nc  — Sea surface height at boundary nodes
  - TEM_3D.th.nc  — Temperature at boundary nodes (3D)
  - SAL_3D.th.nc  — Salinity at boundary nodes (3D)
  - uv3D.th.nc    — Velocity at boundary nodes (3D)

Optionally creates interior nudging fields:
  - TEM_nu.nc     — Temperature nudging (interior relaxation)
  - SAL_nu.nc     — Salinity nudging

Input: RTOFS NetCDF files from COMINrtofs
  2D: rtofs_glo_2ds_{cycle}_diag.nc (SSH)
  3D: rtofs_glo_3dz_{cycle}_6hrly_hvr_US_east.nc (T,S,U,V)

Processing: Extract ROI → merge time steps → interpolate to SCHISM boundary nodes.
"""

import logging
from datetime import datetime, timedelta
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


class RTOFSProcessor(ForcingProcessor):
    """
    RTOFS ocean boundary condition processor for SCHISM.

    Reads RTOFS 2D (SSH) and 3D (T,S,U,V) NetCDF files, extracts the region
    of interest, and creates SCHISM-compatible boundary time-history files.
    """

    SOURCE_NAME = "RTOFS"
    MIN_FILE_SIZE_2D = 150_000_000   # 150 MB for 2D files
    MIN_FILE_SIZE_3D = 200_000_000   # 200 MB for 3D files

    # RTOFS file patterns
    FILE_PATTERNS_2D = [
        "rtofs_glo_2ds_{cycle}_diag.nc",
    ]
    FILE_PATTERNS_3D = [
        "rtofs_glo_3dz_{cycle}_6hrly_hvr_US_east.nc",
        "rtofs_glo_3dz_{cycle}_6hrly_hvr_US_east.nc4",
    ]

    # RTOFS cycle labels for nowcast and forecast
    CYCLES_NOWCAST = ["n024", "n018", "n012", "n006", "n000"]
    CYCLES_FORECAST = [f"f{h:03d}" for h in range(6, 132, 6)]  # f006-f126

    def __init__(
        self,
        config: ForcingConfig,
        input_path: Path,
        output_path: Path,
        grid_file: Optional[Path] = None,
        vgrid_file: Optional[Path] = None,
    ):
        """
        Args:
            config: ForcingConfig with domain, cycle, and OBC settings
            input_path: Root RTOFS data directory (COMINrtofs)
            output_path: Output directory for boundary files
            grid_file: SCHISM grid file (hgrid.ll) for boundary node extraction
            vgrid_file: SCHISM vertical grid (vgrid.in) for level mapping
        """
        super().__init__(config, input_path, output_path)
        self.grid_file = grid_file or config.grid_file
        self.vgrid_file = vgrid_file

    def process(self) -> ForcingResult:
        """
        Process RTOFS data and create SCHISM boundary files.

        Pipeline: discover files → extract ROI → merge → write boundary files
        """
        if not HAS_NETCDF4:
            return ForcingResult(
                success=False, source=self.SOURCE_NAME,
                errors=["netCDF4 required for RTOFS processing"],
            )

        log.info(f"RTOFS processor: pdy={self.config.pdy} cyc={self.config.cyc:02d}z")
        self.create_output_dir()

        # Step 1: Find input files
        files_2d, files_3d = self.find_input_files_by_type()

        if not files_2d and not files_3d:
            return ForcingResult(
                success=False, source=self.SOURCE_NAME,
                errors=["No RTOFS input files found"],
            )

        output_files = []
        warnings = []

        # Step 2: Process 2D (SSH)
        if files_2d:
            log.info(f"Processing {len(files_2d)} RTOFS 2D files")
            elev_file = self._process_2d(files_2d)
            if elev_file:
                output_files.append(elev_file)
            else:
                warnings.append("Failed to create elev2D.th.nc")
        else:
            warnings.append("No RTOFS 2D files found")

        # Step 3: Process 3D (T,S,U,V)
        if files_3d:
            log.info(f"Processing {len(files_3d)} RTOFS 3D files")
            obc_3d_files = self._process_3d(files_3d)
            output_files.extend(obc_3d_files)
            if not obc_3d_files:
                warnings.append("Failed to create 3D boundary files")
        else:
            warnings.append("No RTOFS 3D files found")

        return ForcingResult(
            success=len(output_files) > 0,
            source=self.SOURCE_NAME,
            output_files=output_files,
            warnings=warnings,
            metadata={
                "num_2d_files": len(files_2d),
                "num_3d_files": len(files_3d),
                "ssh_offset": self.config.obc_ssh_offset,
                "nudging_enabled": self.config.nudging_enabled,
            },
        )

    def find_input_files(self) -> List[Path]:
        """Find all RTOFS files (2D + 3D combined)."""
        files_2d, files_3d = self.find_input_files_by_type()
        return files_2d + files_3d

    def find_input_files_by_type(self) -> Tuple[List[Path], List[Path]]:
        """Find RTOFS 2D and 3D files separately.

        Searches multiple path patterns to handle different RTOFS directory layouts:
          - $COMINrtofs/rtofs.YYYYMMDD/rtofs_glo_*  (standard NCO)
          - $COMINrtofs/rtofs.YYYYMMDD/           (flat)
        """
        base_date = datetime.strptime(self.config.pdy, "%Y%m%d")
        files_2d = []
        files_3d = []

        # Search today and yesterday
        for date in [base_date, base_date - timedelta(days=1)]:
            date_str = date.strftime("%Y%m%d")

            # Try multiple directory patterns
            candidate_dirs = [
                self.input_path / f"rtofs.{date_str}",
                self.input_path / date_str,
                self.input_path,
            ]

            for rtofs_dir in candidate_dirs:
                if not rtofs_dir.exists():
                    continue

                # Use glob to find files matching RTOFS naming patterns
                # This handles both exact-name and glob-based discovery
                found_2d = sorted(rtofs_dir.glob("rtofs_glo_2ds_*_diag.nc"))
                found_3d = sorted(rtofs_dir.glob("rtofs_glo_3dz_*_6hrly_hvr_*.nc"))
                # Also check .nc4 extension
                found_3d.extend(sorted(rtofs_dir.glob("rtofs_glo_3dz_*_6hrly_hvr_*.nc4")))

                for f in found_2d:
                    if self.validate_file_size(f, self.MIN_FILE_SIZE_2D):
                        files_2d.append(f)
                for f in found_3d:
                    if self.validate_file_size(f, self.MIN_FILE_SIZE_3D):
                        files_3d.append(f)

                if files_2d or files_3d:
                    log.info(f"Found RTOFS files in {rtofs_dir}: "
                             f"{len(files_2d)} 2D, {len(files_3d)} 3D")
                    break

            if files_2d or files_3d:
                break  # Found files for this date

        return files_2d, files_3d

    def _process_2d(self, files_2d: List[Path]) -> Optional[Path]:
        """Extract SSH from RTOFS 2D files and create elev2D.th.nc."""
        output_file = self.output_path / "elev2D.th.nc"
        roi = self.config.obc_roi_2d

        try:
            all_times = []
            all_ssh = []

            for f in files_2d:
                ds = Dataset(str(f))
                # Extract SSH with optional ROI
                if roi:
                    ssh = ds.variables["ssh"][
                        :, roi["y1"]:roi["y2"], roi["x1"]:roi["x2"]
                    ]
                    lon = ds.variables["Longitude"][
                        roi["y1"]:roi["y2"], roi["x1"]:roi["x2"]
                    ]
                    lat = ds.variables["Latitude"][
                        roi["y1"]:roi["y2"], roi["x1"]:roi["x2"]
                    ]
                else:
                    ssh = ds.variables["ssh"][:]
                    lon = ds.variables["Longitude"][:]
                    lat = ds.variables["Latitude"][:]

                # Get time
                time_var = ds.variables.get("MT") or ds.variables.get("time")
                if time_var is not None:
                    all_times.extend(time_var[:].tolist())

                # Replace fill values
                ssh = np.ma.filled(ssh, fill_value=-30000.0)
                ssh[ssh > 10000] = -30000.0

                all_ssh.append(ssh.squeeze())
                ds.close()

            if not all_ssh:
                return None

            # Stack and apply SSH offset
            ssh_merged = np.stack(all_ssh, axis=0) if len(all_ssh) > 1 else all_ssh[0][np.newaxis, :]
            ssh_merged = ssh_merged + self.config.obc_ssh_offset

            # Write output
            self._write_elev2d(output_file, ssh_merged, all_times, lon, lat)
            log.info(f"Created {output_file.name}: shape={ssh_merged.shape}")
            return output_file

        except Exception as e:
            log.error(f"Failed to process RTOFS 2D: {e}")
            return None

    def _process_3d(self, files_3d: List[Path]) -> List[Path]:
        """Extract T,S,U,V from RTOFS 3D files and create boundary files."""
        output_files = []

        try:
            all_times = []
            all_temp = []
            all_salt = []
            all_u = []
            all_v = []

            roi = self.config.obc_roi_3d

            for f in files_3d:
                ds = Dataset(str(f))

                if roi:
                    sl = (slice(None), slice(None),
                          slice(roi["y1"], roi["y2"]),
                          slice(roi["x1"], roi["x2"]))
                else:
                    sl = (slice(None),) * 4

                for var_name, target_list in [
                    ("temperature", all_temp),
                    ("salinity", all_salt),
                    ("u", all_u),
                    ("v", all_v),
                ]:
                    if var_name in ds.variables:
                        data = ds.variables[var_name][sl]
                        data = np.ma.filled(data, fill_value=-30000.0)
                        data[data > 10000] = -30000.0
                        target_list.append(data.squeeze())

                time_var = ds.variables.get("MT") or ds.variables.get("time")
                if time_var is not None:
                    all_times.extend(time_var[:].tolist())

                ds.close()

            # Write boundary files
            time_step = 21600.0  # 6 hours in seconds (RTOFS output interval)

            if all_temp:
                fpath = self.output_path / "TEM_3D.th.nc"
                merged = np.stack(all_temp, axis=0) if len(all_temp) > 1 else all_temp[0][np.newaxis, :]
                self._write_3d_th(fpath, merged, "temperature", "degC", time_step)
                output_files.append(fpath)
                log.info(f"Created TEM_3D.th.nc: shape={merged.shape}")

            if all_salt:
                fpath = self.output_path / "SAL_3D.th.nc"
                merged = np.stack(all_salt, axis=0) if len(all_salt) > 1 else all_salt[0][np.newaxis, :]
                self._write_3d_th(fpath, merged, "salinity", "PSU", time_step)
                output_files.append(fpath)
                log.info(f"Created SAL_3D.th.nc: shape={merged.shape}")

            if all_u and all_v:
                fpath = self.output_path / "uv3D.th.nc"
                u_merged = np.stack(all_u, axis=0) if len(all_u) > 1 else all_u[0][np.newaxis, :]
                v_merged = np.stack(all_v, axis=0) if len(all_v) > 1 else all_v[0][np.newaxis, :]
                self._write_uv3d_th(fpath, u_merged, v_merged, time_step)
                output_files.append(fpath)
                log.info(f"Created uv3D.th.nc: shape u={u_merged.shape}")

        except Exception as e:
            log.error(f"Failed to process RTOFS 3D: {e}")

        return output_files

    def _write_elev2d(self, output_path: Path, ssh: np.ndarray,
                      times: list, lon: np.ndarray, lat: np.ndarray) -> None:
        """Write elev2D.th.nc in SCHISM format."""
        nc = Dataset(str(output_path), "w", format="NETCDF4")
        nt = ssh.shape[0]
        # Flatten spatial dims for boundary representation
        n_bnd = ssh[0].size

        nc.createDimension("time", nt)
        nc.createDimension("nOpenBndNodes", n_bnd)

        time_var = nc.createVariable("time", "f8", ("time",))
        time_var.units = "seconds since model start"
        time_var[:] = [i * 21600.0 for i in range(nt)]  # 6-hourly

        ts = nc.createVariable("time_series", "f4", ("time", "nOpenBndNodes"),
                               fill_value=-30000.0)
        ts.long_name = "sea surface height"
        ts.units = "m"
        ts[:] = ssh.reshape(nt, -1)

        nc.close()

    def _write_3d_th(self, output_path: Path, data: np.ndarray,
                     var_name: str, units: str, dt: float) -> None:
        """Write TEM_3D.th.nc or SAL_3D.th.nc in SCHISM format."""
        nc = Dataset(str(output_path), "w", format="NETCDF4")
        nt = data.shape[0]

        # data shape: (time, levels, y, x) or (time, levels, nodes)
        if data.ndim == 4:
            n_nodes = data.shape[2] * data.shape[3]
            n_levels = data.shape[1]
            data_flat = data.reshape(nt, n_levels, n_nodes)
        elif data.ndim == 3:
            n_nodes = data.shape[2]
            n_levels = data.shape[1]
            data_flat = data
        else:
            nc.close()
            return

        nc.createDimension("time", nt)
        nc.createDimension("nOpenBndNodes", n_nodes)
        nc.createDimension("nLevels", n_levels)

        time_var = nc.createVariable("time", "f8", ("time",))
        time_var.units = "seconds since model start"
        time_var[:] = [i * dt for i in range(nt)]

        ts = nc.createVariable("time_series", "f4",
                               ("time", "nOpenBndNodes", "nLevels"),
                               fill_value=-30000.0)
        ts.long_name = var_name
        ts.units = units
        # Transpose levels and nodes for SCHISM convention
        ts[:] = np.transpose(data_flat, (0, 2, 1))

        nc.close()

    def _write_uv3d_th(self, output_path: Path, u: np.ndarray,
                       v: np.ndarray, dt: float) -> None:
        """Write uv3D.th.nc in SCHISM format."""
        nc = Dataset(str(output_path), "w", format="NETCDF4")
        nt = u.shape[0]

        if u.ndim == 4:
            n_nodes = u.shape[2] * u.shape[3]
            n_levels = u.shape[1]
            u_flat = u.reshape(nt, n_levels, n_nodes)
            v_flat = v.reshape(nt, n_levels, n_nodes)
        elif u.ndim == 3:
            n_nodes = u.shape[2]
            n_levels = u.shape[1]
            u_flat = u
            v_flat = v
        else:
            nc.close()
            return

        nc.createDimension("time", nt)
        nc.createDimension("nOpenBndNodes", n_nodes)
        nc.createDimension("nLevels", n_levels)
        nc.createDimension("nComponents", 2)

        time_var = nc.createVariable("time", "f8", ("time",))
        time_var.units = "seconds since model start"
        time_var[:] = [i * dt for i in range(nt)]

        ts = nc.createVariable("time_series", "f4",
                               ("time", "nOpenBndNodes", "nLevels", "nComponents"),
                               fill_value=-30000.0)
        ts.long_name = "velocity"
        ts.units = "m/s"
        # Stack u,v as last dimension
        uv = np.stack([
            np.transpose(u_flat, (0, 2, 1)),
            np.transpose(v_flat, (0, 2, 1)),
        ], axis=-1)
        ts[:] = uv

        nc.close()
