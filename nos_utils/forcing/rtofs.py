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
        phase: Optional[str] = None,
        time_hotstart: Optional[datetime] = None,
    ):
        """
        Args:
            config: ForcingConfig with domain and OBC settings
            input_path: Root RTOFS data directory (COMINrtofs)
            output_path: Output directory for boundary files
            grid_file: SCHISM grid file (hgrid.ll) for boundary node extraction
            obc_ctl_file: OBC control file ({ofs}.obc.ctl) for exact node list
            vgrid_file: Vertical grid file (vgrid.in) for depth interpolation
            phase: "nowcast" or "forecast" — determines time window filter
            time_hotstart: Hotstart datetime (nowcast starts from here)
        """
        super().__init__(config, input_path, output_path)
        self.grid_file = grid_file or config.grid_file
        self.obc_ctl_file = obc_ctl_file
        self.vgrid_file = vgrid_file
        self.phase = phase
        self.time_hotstart = time_hotstart
        self._grid = None
        self._bnd_lons = None
        self._bnd_lats = None
        self._bnd_depths = None
        self._vgrid = None
        self._lin_interp = {}  # Cached LinearNDInterpolator per grid
        self._nn_tree = {}     # Cached cKDTree for nearest-neighbor fallback
        self._interp_points = {}   # Cached wet points used to build triangulation
        self._interp_indices = {}  # Flat-grid indices corresponding to cached points

    def _get_time_window(self) -> Tuple[datetime, datetime]:
        """Compute the time window for RTOFS file filtering.

        OBC files cover the full simulation (nowcast + forecast), so the
        window always spans from nowcast start to forecast end, regardless
        of the current phase. A 6h buffer is added on each side.
        """
        cycle_dt = datetime.strptime(self.config.pdy, "%Y%m%d") + \
                   timedelta(hours=self.config.cyc)

        if self.time_hotstart:
            t_start = self.time_hotstart - timedelta(hours=6)
        else:
            t_start = cycle_dt - timedelta(hours=self.config.nowcast_hours) - timedelta(hours=6)
        t_end = cycle_dt + timedelta(hours=self.config.forecast_hours) + timedelta(hours=6)

        return t_start, t_end

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

            # Load per-node sigma values for boundary nodes (LSC2)
            if self._vgrid._filepath is not None:
                self._vgrid.load_boundary_sigma(self._bnd_ids)

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

    @staticmethod
    def _parse_rtofs_hour(filepath: Path) -> Tuple[int, bool]:
        """Parse hour offset and nowcast/forecast flag from RTOFS filename.

        Returns (hour_offset, is_nowcast).
        Examples:
            rtofs_glo_2ds_n024_diag.nc -> (24, True)
            rtofs_glo_3dz_f048_6hrly_hvr_US_east.nc -> (48, False)
        """
        import re
        match = re.search(r'_([nf])(\d{3})[_.]', filepath.name)
        if match:
            return int(match.group(2)), match.group(1) == 'n'
        return 0, False

    def _sort_and_dedup(self, files: List[Path], cycle_date: datetime) -> List[Path]:
        """Sort RTOFS files by valid time and dedup n/f overlap.

        When nowcast (n) and forecast (f) files have the same valid time,
        prefer the nowcast file (analysis-based).
        """
        # Parse valid times
        file_info = []
        for f in files:
            hour, is_nowcast = self._parse_rtofs_hour(f)
            valid_time = cycle_date + timedelta(hours=hour)
            file_info.append((valid_time, is_nowcast, f))

        # Sort by valid time; for ties, nowcast first (True > False)
        file_info.sort(key=lambda x: (x[0], not x[1]))

        # Dedup: keep first per valid time (nowcast preferred due to sort)
        seen = {}
        for vt, is_nc, f in file_info:
            if vt not in seen:
                seen[vt] = f

        # Filter by phase time window if set
        if self.phase is not None:
            t_start, t_end = self._get_time_window()
            seen = {vt: f for vt, f in seen.items()
                    if t_start <= vt <= t_end}

        result = [f for _, f in sorted(seen.items())]

        n_removed = len(files) - len(result)
        if n_removed > 0:
            log.info(f"RTOFS dedup: {len(files)} -> {len(result)} files "
                     f"({n_removed} duplicates/out-of-window removed)")

        return result

    def find_input_files_by_type(self) -> Tuple[List[Path], List[Path]]:
        """Find RTOFS 2D and 3D files, sorted by valid time and deduplicated."""
        base_date = datetime.strptime(self.config.pdy, "%Y%m%d")
        files_2d = []
        files_3d = []
        rtofs_cycle_date = None

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
                    rtofs_cycle_date = date
                    log.info(f"Found RTOFS files in {rtofs_dir}: {len(files_2d)} 2D, {len(files_3d)} 3D")
                    break

            if files_2d or files_3d:
                break

        # Sort by valid time and deduplicate n/f overlap
        if rtofs_cycle_date is None:
            rtofs_cycle_date = base_date
        if files_2d:
            files_2d = self._sort_and_dedup(files_2d, rtofs_cycle_date)
        if files_3d:
            files_3d = self._sort_and_dedup(files_3d, rtofs_cycle_date)

        return files_2d, files_3d

    def _interpolate_2d_to_boundary(
        self, rtofs_lon: np.ndarray, rtofs_lat: np.ndarray, rtofs_data: np.ndarray,
    ) -> np.ndarray:
        """
        Interpolate a 2D RTOFS field to boundary node locations.

        Uses cached Delaunay triangulation built from the first call's ocean
        mask (typically surface level with most ocean points). Subsequent calls
        reuse the same triangulation — deep levels with fewer ocean points get
        their land values filled from nearest valid ocean before interpolation.
        This matches Fortran INTERP_REMESH behavior.

        For 3D fields where the ocean mask changes per depth level, a separate
        interpolator is built per unique mask.
        """
        n_bnd = len(self._bnd_lons)
        bnd_lons_360 = np.where(self._bnd_lons < 0, self._bnd_lons + 360, self._bnd_lons)
        target_pts = np.column_stack([bnd_lons_360, self._bnd_lats])

        # Flatten RTOFS grid
        if rtofs_lon.ndim == 2:
            lon_flat = rtofs_lon.ravel()
            lat_flat = rtofs_lat.ravel()
        else:
            lon_2d, lat_2d = np.meshgrid(rtofs_lon, rtofs_lat)
            lon_flat = lon_2d.ravel()
            lat_flat = lat_2d.ravel()

        data_flat = rtofs_data.ravel()

        # Mask land/fill values
        ocean_mask = np.abs(data_flat) < 1e10
        n_ocean = int(np.sum(ocean_mask))

        if n_ocean == 0:
            log.warning("No valid ocean data in field")
            return np.full(n_bnd, np.nan, dtype=np.float32)

        ocean_pts = np.column_stack([lon_flat[ocean_mask], lat_flat[ocean_mask]])
        ocean_val = data_flat[ocean_mask].astype(np.float32)

        # Cache key: grid shape only (NOT n_ocean) — reuse surface triangulation
        # for deeper levels where ocean mask shrinks. Fill values at land points
        # get interpolated from surrounding ocean, matching Fortran remesh behavior.
        cache_key = rtofs_lon.shape

        try:
            from scipy.interpolate import LinearNDInterpolator
            from scipy.spatial import cKDTree

            # Check if cached indices are compatible with current data
            cached_ok = (
                cache_key in self._lin_interp
                and cache_key in self._interp_indices
                and len(data_flat) >= int(np.max(self._interp_indices[cache_key])) + 1
            )

            if cache_key not in self._lin_interp or not cached_ok:
                # First call (or grid changed) — build triangulation from ocean points
                self._lin_interp[cache_key] = LinearNDInterpolator(
                    ocean_pts, ocean_val,
                )
                self._nn_tree[cache_key] = cKDTree(ocean_pts)
                self._interp_points[cache_key] = ocean_pts
                self._interp_indices[cache_key] = np.where(ocean_mask)[0]
                log.info(f"Built Delaunay interpolator: {n_ocean} ocean pts "
                         f"(grid {rtofs_lon.shape})")
                interp_vals = ocean_val
                result = self._lin_interp[cache_key](target_pts).astype(np.float32)
            else:
                # Reuse triangulation with values expressed on the original
                # wet-point geometry. Deeper levels may turn some of those
                # points into land, so fill them from the nearest still-wet
                # cached point before evaluating the interpolator.
                cached_pts = self._interp_points[cache_key]
                orig_idx = self._interp_indices[cache_key]
                vals = data_flat[orig_idx].astype(np.float32, copy=True)
                valid_v = np.abs(vals) < 1e10
                vals[~valid_v] = np.nan

                if np.any(~valid_v):
                    if np.any(valid_v):
                        from scipy.spatial import cKDTree as _cKDTree
                        valid_tree = _cKDTree(cached_pts[valid_v])
                        _, nn = valid_tree.query(cached_pts[~valid_v])
                        vals[~valid_v] = vals[valid_v][nn]
                    else:
                        vals[:] = np.float32(np.nanmean(ocean_val))

                self._lin_interp[cache_key].values[:, 0] = vals
                interp_vals = vals
                result = self._lin_interp[cache_key](target_pts).astype(np.float32)

            # Fill NaN (outside convex hull) with nearest-neighbor
            nan_mask = np.isnan(result)
            if np.any(nan_mask):
                _, nn_idx = self._nn_tree[cache_key].query(target_pts[nan_mask])
                result[nan_mask] = interp_vals[nn_idx]

        except ImportError:
            result = np.zeros(n_bnd, dtype=np.float32)
            for k in range(n_bnd):
                dist = (ocean_pts[:, 0] - bnd_lons_360[k])**2 + \
                       (ocean_pts[:, 1] - self._bnd_lats[k])**2
                result[k] = ocean_val[np.argmin(dist)]

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
                    # Geoid-to-MSL datum offset (config-driven, per OFS):
                    #   SECOFS: +1.25m (nos_ofs_create_forcing_obc_schism.f line 3133)
                    #   STOFS-3D-ATL: 0.0 (uses ADT blending, not constant offset)
                    #   Other OFS: verify offset from Fortran source before setting
                    ssh_bnd += self.config.obc_ssh_offset
                    all_ssh.append(ssh_bnd)

                ds.close()

            if not all_ssh:
                return None

            ssh_array = np.stack(all_ssh, axis=0)

            # Fill NaN nodes (e.g., nodes 0-3 that fall outside RTOFS domain)
            # by propagating from nearest valid boundary node
            nan_mask = np.isnan(ssh_array[0, :])
            n_nan = np.sum(nan_mask)
            if n_nan > 0 and n_nan < n_bnd:
                valid_indices = np.where(~nan_mask)[0]
                nan_indices = np.where(nan_mask)[0]
                for ni in nan_indices:
                    nearest_valid = valid_indices[np.argmin(np.abs(valid_indices - ni))]
                    ssh_array[:, ni] = ssh_array[:, nearest_valid]
                log.info(f"Filled {n_nan} NaN boundary nodes from nearest valid nodes")

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
        """Extract T/S from RTOFS 3D files and write SCHISM boundary files."""
        output_files = []
        n_bnd = len(self._bnd_lons)
        n_levels = self._vgrid.nvrt if self._vgrid else self.config.n_levels

        try:
            all_temp = []
            all_salt = []
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
                ]:
                    if var_name not in ds.variables:
                        continue

                    data = ds.variables[var_name][:]
                    data = np.ma.filled(data, fill_value=np.nan)

                    # Handle time dimension: iterate over all time steps
                    if data.ndim == 4:
                        n_times = data.shape[0]
                    else:
                        n_times = 1
                        data = data[np.newaxis, ...]

                    for t in range(n_times):
                        data_t = data[t]

                        # Interpolate each RTOFS depth level to boundary nodes
                        bnd_profile_rtofs = np.full((n_bnd, n_rtofs_levels), np.nan, dtype=np.float32)
                        for lev in range(min(n_rtofs_levels, data_t.shape[0])):
                            bnd_profile_rtofs[:, lev] = self._interpolate_2d_to_boundary(
                                lon_arr, lat_arr, data_t[lev]
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

            rtofs_dt_3d = 21600.0  # 6-hourly RTOFS input
            target_dt_3d = 10800.0  # 3-hourly output (matches Fortran DELT_TS)

            # Temporally interpolate 3D fields from 6h to 3h
            for var_list in [all_temp, all_salt]:
                if len(var_list) > 1:
                    try:
                        from scipy.interpolate import interp1d
                        n_in = len(var_list)
                        rtofs_times = np.arange(n_in) * rtofs_dt_3d
                        n_out = int((n_in - 1) * rtofs_dt_3d / target_dt_3d) + 1
                        target_times = np.arange(n_out) * target_dt_3d
                        target_times = target_times[target_times <= rtofs_times[-1]]

                        stacked = np.stack(var_list, axis=0)  # (n_in, n_bnd, n_levels)
                        interp_out = np.zeros((len(target_times),) + stacked.shape[1:],
                                              dtype=np.float32)
                        for node in range(stacked.shape[1]):
                            for lev in range(stacked.shape[2]):
                                f_i = interp1d(rtofs_times, stacked[:, node, lev],
                                               kind="linear", fill_value="extrapolate")
                                interp_out[:, node, lev] = f_i(target_times)
                        var_list.clear()
                        var_list.extend([interp_out[t] for t in range(len(target_times))])
                    except ImportError:
                        pass

            dt = target_dt_3d if len(all_temp) > 1 else rtofs_dt_3d

            if all_temp:
                fpath = self.output_path / "TEM_3D.th.nc"
                merged = np.stack(all_temp, axis=0)
                self._write_3d_th(fpath, merged, "temperature", "degC", dt, n_bnd)
                output_files.append(fpath)
                log.info(f"Created TEM_3D.th.nc: {merged.shape} at dt={dt}s")

            if all_salt:
                fpath = self.output_path / "SAL_3D.th.nc"
                merged = np.stack(all_salt, axis=0)
                self._write_3d_th(fpath, merged, "salinity", "PSU", dt, n_bnd)
                output_files.append(fpath)
                log.info(f"Created SAL_3D.th.nc: {merged.shape} at dt={dt}s")

            # uv3D: write ALL ZEROS — COMF SCHISM computes boundary velocities
            # from SSH gradients. Prescribing RTOFS velocities is wrong physics.
            if all_temp:
                # Use same time dimension as T/S
                nt_3d = len(all_temp)
                fpath = self.output_path / "uv3D.th.nc"
                u_zeros = np.zeros((nt_3d, n_bnd, n_levels), dtype=np.float32)
                v_zeros = np.zeros((nt_3d, n_bnd, n_levels), dtype=np.float32)
                self._write_uv3d_th(fpath, u_zeros, v_zeros, dt, n_bnd)
                output_files.append(fpath)
                log.info(f"Created uv3D.th.nc: zeros ({nt_3d}, {n_bnd}, {n_levels}) "
                         f"— COMF uses SSH-derived boundary velocities")

        except Exception as e:
            log.error(f"Failed to process RTOFS 3D: {e}")
            import traceback
            log.error(traceback.format_exc())

        return output_files

    def _interpolate_vertical(
        self, bnd_profile: np.ndarray, rtofs_depths: np.ndarray,
        var_name: str, target_levels: int,
    ) -> np.ndarray:
        """
        Interpolate from RTOFS depth levels to SCHISM vertical levels.

        Uses per-node sigma values from LSC2 vgrid when available,
        giving node-specific vertical structure that matches the Fortran output.
        """
        from scipy.interpolate import interp1d

        n_bnd = bnd_profile.shape[0]
        rtofs_z = -np.abs(rtofs_depths)  # negative down

        defaults = {"temperature": 15.0, "salinity": 35.0, "u": 0.0, "v": 0.0}
        fill_val = defaults.get(var_name, 0.0)

        result = np.full((n_bnd, target_levels), fill_val, dtype=np.float32)

        for node in range(n_bnd):
            profile = bnd_profile[node, :]
            valid = ~np.isnan(profile)

            if np.sum(valid) < 2:
                continue

            node_depth = abs(self._bnd_depths[node])
            if node_depth < 0.1:
                continue  # Skip dry nodes

            # Get SCHISM target depths for this node
            if self._vgrid and self._vgrid.node_sigma is not None:
                # Per-node sigma from LSC2 (best match to Fortran)
                schism_z = self._vgrid.get_node_depths(node, node_depth)
            elif self._vgrid:
                schism_z = self._vgrid.get_depths(node_depth)
            else:
                schism_z = np.linspace(-node_depth, 0, target_levels)

            # Pad or trim to target_levels
            if len(schism_z) > target_levels:
                schism_z = schism_z[-target_levels:]  # Keep surface levels
            elif len(schism_z) < target_levels:
                # Pad deep levels
                n_pad = target_levels - len(schism_z)
                pad_z = np.full(n_pad, schism_z[0])
                schism_z = np.concatenate([pad_z, schism_z])

            # Interpolate from RTOFS depths to SCHISM depths
            f_interp = interp1d(
                rtofs_z[valid], profile[valid],
                kind="linear", bounds_error=False,
                fill_value=(profile[valid][0], profile[valid][-1]),
            )
            result[node, :] = f_interp(schism_z[:target_levels])

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
