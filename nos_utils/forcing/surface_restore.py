"""
Surface Restore processor for SCHISM.

Interpolates RTOFS surface temperature and salinity to boundary/restore nodes
and applies a 2-step temporal moving average, replicating the ops pipeline:
    ncks -O -d time,0,3 ... ncks -O -d time,1,4 ... ncecat ... ncwa

Output:
  - surface_restore.nc
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np

from ..config import ForcingConfig
from .base import ForcingProcessor, ForcingResult

log = logging.getLogger(__name__)

try:
    from netCDF4 import Dataset
    HAS_NETCDF4 = True
except ImportError:
    HAS_NETCDF4 = False


class SurfaceRestoreProcessor(ForcingProcessor):
    """Generate surface_restore.nc from RTOFS data."""

    SOURCE_NAME = "SURFACE_RESTORE"

    def __init__(
        self,
        config: ForcingConfig,
        input_path: Path,
        output_path: Path,
        weight_file: Optional[Path] = None,
        phase: Optional[str] = None,
    ):
        super().__init__(config, input_path, output_path)
        self.weight_file = weight_file
        self.phase = phase

    def process(self) -> ForcingResult:
        if not HAS_NETCDF4:
            return ForcingResult(
                success=False, source=self.SOURCE_NAME,
                errors=["netCDF4 required for surface restore"],
            )

        if not getattr(self.config, "surface_restore_enabled", False):
            return ForcingResult(
                success=True, source=self.SOURCE_NAME,
                warnings=["Surface restore disabled in config"],
            )

        self.create_output_dir()
        
        # Reuse RTOFS 3D discovery to get the raw daily/6-hourly files
        from .rtofs import RTOFSProcessor
        rtofs_proc = RTOFSProcessor(
            self.config, self.input_path, self.output_path, phase=self.phase
        )
        _, files_3d = rtofs_proc.find_input_files_by_type()

        if not files_3d:
            return ForcingResult(
                success=False, source=self.SOURCE_NAME,
                errors=["No RTOFS 3D files found for surface restore"],
            )

        # Load target nodes from the gr3 weight file (same layout as nudging)
        node_ids, lons, lats = self._load_restore_nodes()
        if node_ids is None:
            return ForcingResult(
                success=False, source=self.SOURCE_NAME,
                errors=["Cannot identify restore nodes: weight_file missing"],
            )

        n_nodes = len(node_ids)
        all_temp, all_salt, times = [], [], []

        # RTOFS Extraction (Surface Only)
        for f in files_3d:
            try:
                ds = Dataset(str(f))
                lon_arr = ds.variables.get("Longitude") or ds.variables.get("lon")
                lat_arr = ds.variables.get("Latitude") or ds.variables.get("lat")
                
                if lon_arr is None or lat_arr is None:
                    ds.close()
                    continue
                
                # Fetch surface layer (index 0)
                temp = ds.variables.get("temperature")
                salt = ds.variables.get("salinity")
                
                if temp is not None:
                    # [time, depth, lat, lon] -> grab surface depth (0)
                    t_surf = np.ma.filled(temp[:, 0, :, :], fill_value=np.nan)
                    t_surf[np.abs(t_surf) >= 99.0] = np.nan
                    
                    s_surf = np.ma.filled(salt[:, 0, :, :], fill_value=np.nan)
                    s_surf[np.abs(s_surf) >= 99.0] = np.nan
                    
                    for t in range(t_surf.shape[0]):
                        t_bnd = rtofs_proc._interpolate_2d_to_boundary(lon_arr[:], lat_arr[:], t_surf[t])
                        s_bnd = rtofs_proc._interpolate_2d_to_boundary(lon_arr[:], lat_arr[:], s_surf[t])
                        all_temp.append(t_bnd)
                        all_salt.append(s_bnd)
                        times.append(float(t)) # Placeholder for actual valid time
                        
                ds.close()
            except Exception as e:
                log.warning(f"Failed to read 3D file for restore {f.name}: {e}")

        if not all_temp:
            return ForcingResult(
                success=False, source=self.SOURCE_NAME,
                errors=["Failed to extract surface T/S"],
            )

        # Convert to numpy arrays: shape (n_times, n_nodes)
        temp_arr = np.stack(all_temp, axis=0)
        salt_arr = np.stack(all_salt, axis=0)

        dt_out = getattr(self.config, "obc_dt", 86400.0)

        # Apply 2-step temporal moving average if configured
        if getattr(self.config, "surface_restore_time_smoothing", False) and temp_arr.shape[0] > 1:
            temp_avg = (temp_arr[:-1] + temp_arr[1:]) / 2.0
            salt_avg = (salt_arr[:-1] + salt_arr[1:]) / 2.0
            time_axis = np.arange(temp_avg.shape[0]) * dt_out
            log.info(f"Surface restore: applied 2-step temporal smoothing, resulting in {temp_avg.shape[0]} steps")
        else:
            temp_avg, salt_avg = temp_arr, salt_arr
            time_axis = np.arange(temp_avg.shape[0]) * dt_out

        # Replace remaining NaNs with typical oceanic fill values
        temp_avg = np.where(np.isnan(temp_avg), 15.0, temp_avg)
        salt_avg = np.where(np.isnan(salt_avg), 35.0, salt_avg)

        # Write output file
        out_path = self.output_path / "surface_restore.nc"
        self._write_restore_nc(out_path, temp_avg, salt_avg, node_ids, time_axis)

        return ForcingResult(
            success=True,
            source=self.SOURCE_NAME,
            output_files=[out_path],
            metadata={
                "n_restore_nodes": n_nodes,
                "n_timesteps": len(time_axis),
            }
        )

    def _load_restore_nodes(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load target nodes from the gr3 weight file."""
        from ..io.schism_grid import SchismGrid
        weight_file = self.weight_file or self._find_restore_file()
        
        if not weight_file or not Path(weight_file).exists():
            return None, None, None
            
        node_ids, lons, lats, values = SchismGrid.read_gr3_values(weight_file)
        mask = values > 0
        return node_ids[mask], lons[mask], lats[mask]
        
    def _find_restore_file(self) -> Optional[Path]:
        import os
        patterns = ["*.nudge.gr3", "*.TEM_nudge.gr3"]
        for env_var in ["FIXofs", "FIXstofs3d"]:
            fix_dir = os.environ.get(env_var)
            if fix_dir:
                for pattern in patterns:
                    matches = sorted(Path(fix_dir).glob(pattern))
                    if matches: return matches[0]
        return None

    def _write_restore_nc(self, output_path: Path, temp: np.ndarray, salt: np.ndarray, 
                          node_ids: np.ndarray, time_axis: np.ndarray) -> None:
        """Write surface_restore.nc using standard SCHISM formats."""
        nt, n_nodes = temp.shape
        
        nc = Dataset(str(output_path), "w", format="NETCDF4_CLASSIC")
        nc.createDimension("time", nt)
        nc.createDimension("node", n_nodes)
        nc.createDimension("one", 1)

        time_var = nc.createVariable("time", "f8", ("time",))
        time_var[:] = time_axis

        map_var = nc.createVariable("map_to_global_node", "i4", ("node",))
        map_var[:] = node_ids.astype(np.int32)

        t_var = nc.createVariable("temperature", "f8", ("time", "node", "one"))
        t_var[:, :, 0] = temp

        s_var = nc.createVariable("salinity", "f8", ("time", "node", "one"))
        s_var[:, :, 0] = salt

        nc.close()
        log.info(f"Created surface_restore.nc: {nt} times, {n_nodes} nodes.")
