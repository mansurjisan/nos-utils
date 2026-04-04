"""
RTOFS ocean boundary condition processor.

Creates SCHISM boundary time-history files by interpolating RTOFS global
ocean model output to SCHISM open boundary nodes.

Output:
  - elev2D.th.nc  — SSH at boundary nodes (time, nOpenBndNodes)
  - TEM_3D.th.nc  — Temperature at boundary nodes (time, nOpenBndNodes, nLevels)
  - SAL_3D.th.nc  — Salinity at boundary nodes (time, nOpenBndNodes, nLevels)
  - uv3D.th.nc    — Velocity at boundary nodes (time, nOpenBndNodes, nLevels, 2)

Reads SCHISM boundary node locations from hgrid.ll (or obc.ctl for exact
Fortran-matching node list) and interpolates RTOFS data to those nodes.

IMPORTANT: SSH bias correction
  The Fortran gen_3Dth_from_hycom applies a station-based bias correction:
    1. Reads real-time tide gauge observations (NOSBUFR)
    2. Computes AVGERR = mean(obs - RTOFS) per station (~1.25m for SECOFS)
    3. Applies WLOBC += weight * (AVGERR + obs_subtidal) per boundary node
  This Python processor does NOT apply this correction — the ~1.25m SSH
  offset relative to Fortran output is expected. For production runs,
  use hybrid mode (Fortran OBC) until Python has access to real-time
  tide gauge data.

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
        obc_ctl_file: Optional[Path] = None,
        vgrid_file: Optional[Path] = None,
    ):
        """
        Args:
            config: ForcingConfig with domain and OBC settings
            input_path: Root RTOFS data directory (COMINrtofs)
            output_path: Output directory for boundary files
            grid_file: SCHISM grid file (hgrid.ll) for boundary node extraction
            obc_ctl_file: OBC control file ({ofs}.obc.ctl) for exact node list
            vgrid_file: Vertical grid file (vgrid.in) for depth interpolation
        """
        super().__init__(config, input_path, output_path)
        self.grid_file = grid_file or config.grid_file
        self.obc_ctl_file = obc_ctl_file
        self.vgrid_file = vgrid_file
        self._grid = None
        self._bnd_lons = None
        self._bnd_lats = None
        self._bnd_depths = None
        self._interp_cache = {}
        self._vgrid = None

    def _load_grid(self) -> bool:
        """Load SCHISM grid and extract boundary node coordinates."""
        if self._bnd_lons is not None:
            return True

        if self.grid_file is None or not Path(self.grid_file).exists():
            log.warning(f"Grid file not found: {self.grid_file}")
            return False

        self._grid = SchismGrid.read(self.grid_file)

        # Use obc.ctl for exact node list if available (matches Fortran 1,488 nodes)
        if self.obc_ctl_file and Path(self.obc_ctl_file).exists():
            self._bnd_lons, self._bnd_lats, self._bnd_depths, self._bnd_ids = \
                self._grid.obc_nodes_from_ctl(self.obc_ctl_file)
        else:
            self._bnd_lons, self._bnd_lats, self._bnd_depths, self._bnd_ids = \
                self._grid.open_boundary_nodes()

        log.info(f"Loaded {len(self._bnd_lons)} boundary nodes from "
                 f"{Path(self.obc_ctl_file).name if self.obc_ctl_file else Path(self.grid_file).name}")

        # Load vertical grid for depth interpolation
        if self.vgrid_file and Path(self.vgrid_file).exists():
            from ..io.schism_vgrid import SchismVgrid
            self._vgrid = SchismVgrid.read(self.vgrid_file)

        return len(self._bnd_lons) > 0

    def process(self) -> ForcingResult:
        if not HAS_NETCDF4:
            return ForcingResult(
                success=False, source=self.SOURCE_NAME,
                errors=["netCDF4 required for RTOFS processing"],
            )

        log.info(f"RTOFS processor: pdy={self.config.pdy} cyc={self.config.cyc:02d}z")
        self.create_output_dir()

        has_grid = self._load_grid()
        if not has_grid:
            return ForcingResult(
                success=False, source=self.SOURCE_NAME,
                errors=[f"Cannot load boundary nodes from grid: {self.grid_file}"],
            )

        files_2d, files_3d = self.find_input_files_by_type()
        if not files_2d and not files_3d:
            return ForcingResult(
                success=False, source=self.SOURCE_NAME,
                errors=["No RTOFS input files found"],
            )

        output_files = []
        warnings = []

        if files_2d:
            log.info(f"Processing {len(files_2d)} RTOFS 2D files")
            f = self._process_2d(files_2d)
            if f:
                output_files.append(f)
            else:
                warnings.append("Failed to create elev2D.th.nc")

        if files_3d:
            log.info(f"Processing {len(files_3d)} RTOFS 3D files")
            obc_files = self._process_3d(files_3d)
            output_files.extend(obc_files)

        # Note about SSH bias correction
        if not warnings:
            warnings.append(
                "SSH not bias-corrected (no tide gauge data). "
                "Use hybrid mode for production OBC."
            )

        return ForcingResult(
            success=len(output_files) > 0,
            source=self.SOURCE_NAME,
            output_files=output_files,
            warnings=warnings,
            metadata={
                "n_boundary_nodes": len(self._bnd_lons),
                "n_2d_files": len(files_2d),
                "n_3d_files": len(files_3d),
                "n_levels": self._vgrid.nvrt if self._vgrid else self.config.n_levels,
                "ssh_bias_corrected": False,
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

    def _build_rtofs_interpolator(
        self, rtofs_lon: np.ndarray, rtofs_lat: np.ndarray,
    ) -> Optional[dict]:
        """
        Build nearest-neighbor interpolation index from RTOFS grid to boundary nodes.
        Cached per grid shape (global 2D vs regional 3D have different grids).
        """
        if rtofs_lon.ndim == 2:
            cache_key = rtofs_lon.shape
        else:
            cache_key = (len(rtofs_lat), len(rtofs_lon))

        if cache_key in self._interp_cache:
            return self._interp_cache[cache_key]

        n_bnd = len(self._bnd_lons)
        bnd_lons_360 = np.where(self._bnd_lons < 0, self._bnd_lons + 360, self._bnd_lons)

        if rtofs_lon.ndim == 2:
            lon_flat = rtofs_lon.ravel()
            lat_flat = rtofs_lat.ravel()
        else:
            lon_2d, lat_2d = np.meshgrid(rtofs_lon, rtofs_lat)
            lon_flat = lon_2d.ravel()
            lat_flat = lat_2d.ravel()

        # Use scipy cKDTree if available (much faster than brute force)
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(np.column_stack([lon_flat, lat_flat]))
            _, indices = tree.query(np.column_stack([bnd_lons_360, self._bnd_lats]))
            indices = indices.astype(np.int64)
        except ImportError:
            # Brute force fallback
            indices = np.zeros(n_bnd, dtype=np.int64)
            for k in range(n_bnd):
                dist = (lon_flat - bnd_lons_360[k])**2 + (lat_flat - self._bnd_lats[k])**2
                indices[k] = np.argmin(dist)

        result = {
            "indices": indices,
            "n_points": len(lon_flat),
        }
        self._interp_cache[cache_key] = result
        log.info(f"Built RTOFS interpolation index for {n_bnd} boundary nodes "
                 f"(grid {cache_key}, {len(lon_flat)} points)")
        return result

    def _interpolate_2d_to_boundary(
        self, rtofs_lon: np.ndarray, rtofs_lat: np.ndarray, rtofs_data: np.ndarray,
    ) -> np.ndarray:
        """Interpolate a 2D RTOFS field to boundary node locations."""
        n_bnd = len(self._bnd_lons)

        interp = self._build_rtofs_interpolator(rtofs_lon, rtofs_lat)
        if interp is None:
            return np.full(n_bnd, np.nan, dtype=np.float32)

        data_flat = rtofs_data.ravel()
        result = data_flat[interp["indices"]].astype(np.float32)
        result[np.abs(result) > 1e10] = np.nan

        return result

    def _process_2d(self, files_2d: List[Path]) -> Optional[Path]:
        """Extract SSH from RTOFS 2D files, interpolate to boundary nodes."""
        output_file = self.output_path / "elev2D.th.nc"
        n_bnd = len(self._bnd_lons)
        model_dt = 120.0  # SCHISM model timestep (seconds)

        try:
            all_ssh = []

            for f in files_2d:
                ds = Dataset(str(f))
                ssh_raw = ds.variables["ssh"][:]
                lon = ds.variables["Longitude"][:]
                lat = ds.variables["Latitude"][:]

                ssh_raw = np.ma.filled(ssh_raw, fill_value=np.nan)

                for t in range(ssh_raw.shape[0]):
                    ssh_bnd = self._interpolate_2d_to_boundary(lon, lat, ssh_raw[t])
                    ssh_bnd += self.config.obc_ssh_offset
                    all_ssh.append(ssh_bnd)

                ds.close()

            if not all_ssh:
                return None

            ssh_array = np.stack(all_ssh, axis=0)

            # Temporally interpolate to model dt (120s)
            n_rtofs = ssh_array.shape[0]
            rtofs_dt = 21600.0
            total_time = (n_rtofs - 1) * rtofs_dt
            n_model_steps = int(total_time / model_dt) + 1

            if n_model_steps > n_rtofs and n_rtofs > 1:
                try:
                    from scipy.interpolate import interp1d
                    rtofs_times = np.arange(n_rtofs) * rtofs_dt
                    model_times = np.arange(n_model_steps) * model_dt
                    model_times = model_times[model_times <= rtofs_times[-1]]

                    ssh_interp = np.zeros((len(model_times), n_bnd), dtype=np.float32)
                    for node in range(n_bnd):
                        f_interp = interp1d(rtofs_times, ssh_array[:, node],
                                           kind="linear", fill_value="extrapolate")
                        ssh_interp[:, node] = f_interp(model_times)

                    ssh_array = ssh_interp
                    dt_out = model_dt
                    log.info(f"Temporally interpolated SSH: {n_rtofs} steps at 6h → "
                             f"{len(model_times)} steps at dt={model_dt}s")
                except ImportError:
                    dt_out = rtofs_dt
            else:
                dt_out = rtofs_dt

            # Write SCHISM format
            nc = Dataset(str(output_file), "w", format="NETCDF4")
            nt = ssh_array.shape[0]

            nc.createDimension("time", nt)
            nc.createDimension("nOpenBndNodes", n_bnd)
            nc.createDimension("nLevels", 1)
            nc.createDimension("nComponents", 1)
            nc.createDimension("one", 1)

            time_var = nc.createVariable("time", "f8", ("time",))
            time_var[:] = [i * dt_out for i in range(nt)]

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
        n_levels = self._vgrid.nvrt if self._vgrid else self.config.n_levels

        try:
            all_temp = []
            all_salt = []
            all_u = []
            all_v = []

            for f in files_3d:
                ds = Dataset(str(f))

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

                    if data.ndim == 4:
                        data = data[0]

                    # Interpolate each RTOFS depth level to boundary nodes
                    bnd_profile_rtofs = np.full((n_bnd, n_rtofs_levels), np.nan, dtype=np.float32)
                    for lev in range(min(n_rtofs_levels, data.shape[0])):
                        bnd_profile_rtofs[:, lev] = self._interpolate_2d_to_boundary(
                            lon_arr, lat_arr, data[lev]
                        )

                    # Vertically interpolate from RTOFS levels to SCHISM levels
                    if self._vgrid and n_levels != n_rtofs_levels:
                        bnd_profile = self._interpolate_vertical(
                            bnd_profile_rtofs, depth_arr, var_name, n_levels
                        )
                    else:
                        bnd_profile = bnd_profile_rtofs

                    target_list.append(bnd_profile)

                ds.close()

            dt = 21600.0

            if all_temp:
                fpath = self.output_path / "TEM_3D.th.nc"
                merged = np.stack(all_temp, axis=0)
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

    def _interpolate_vertical(
        self, bnd_profile: np.ndarray, rtofs_depths: np.ndarray,
        var_name: str, target_levels: int,
    ) -> np.ndarray:
        """Interpolate from RTOFS depth levels to SCHISM vertical levels."""
        n_bnd = bnd_profile.shape[0]
        rtofs_z = -np.abs(rtofs_depths)

        defaults = {"temperature": 15.0, "salinity": 35.0, "u": 0.0, "v": 0.0}
        fill_val = defaults.get(var_name, 0.0)

        result = np.full((n_bnd, target_levels), fill_val, dtype=np.float32)

        for node in range(n_bnd):
            profile = bnd_profile[node, :]
            valid = ~np.isnan(profile)

            if np.sum(valid) < 2:
                continue

            node_depth = abs(self._bnd_depths[node])
            if self._vgrid:
                schism_z = self._vgrid.get_depths(node_depth)
                if len(schism_z) > target_levels:
                    schism_z = schism_z[:target_levels]
                elif len(schism_z) < target_levels:
                    extra = np.linspace(schism_z[-1], 0, target_levels - len(schism_z) + 1)[1:]
                    schism_z = np.concatenate([schism_z, extra])
            else:
                schism_z = np.linspace(-node_depth, 0, target_levels)

            from scipy.interpolate import interp1d
            f_interp = interp1d(
                rtofs_z[valid], profile[valid],
                kind="linear", bounds_error=False,
                fill_value=(profile[valid][0], profile[valid][-1]),
            )
            n_out = min(len(schism_z), target_levels)
            result[node, :n_out] = f_interp(schism_z[:n_out])

        return result

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
        ts[:, :, :, 0] = data

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
