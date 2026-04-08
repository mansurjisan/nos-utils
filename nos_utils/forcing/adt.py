"""
ADT (Absolute Dynamic Topography) satellite SSH blender.

Blends CMEMS satellite ADT observations with RTOFS SSH to improve
boundary condition accuracy for STOFS-3D-ATL.

The core formula (from stofs_3d_atl_create_obc_3d_th.sh lines 580-610):
    SSH_final = SSH_rtofs - SSH_rtofs(t=0) + ADT(t=0)

This removes the RTOFS bias at t=0 and replaces it with the satellite-observed
absolute dynamic topography, preserving RTOFS temporal variability.

Input:
  - SSH_1.nc — RTOFS SSH prepared by RTOFSProcessor._stofs_prepare_ssh()
  - CMEMS ADT: nrt_global_allsat_phy_l4_YYYYMMDD_YYYYMMDD.nc
  - Weight file: stofs_3d_atl_adt_weight.nc (regridding weights)

Output:
  - SSH_1.nc updated with ADT-blended surf_el values

Graceful fallback: returns None if ADT data is unavailable (RTOFS-only SSH used).
"""

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np

from ..config import ForcingConfig

log = logging.getLogger(__name__)

try:
    from netCDF4 import Dataset
    HAS_NETCDF4 = True
except ImportError:
    HAS_NETCDF4 = False

# ADT subset domain (eastern boundary region)
ADT_LON_MIN = -62.5
ADT_LON_MAX = -51.5
ADT_LAT_MIN = 7.0
ADT_LAT_MAX = 54.0


class ADTBlender:
    """Blend CMEMS ADT satellite SSH with RTOFS SSH."""

    def __init__(self, config: ForcingConfig, input_path: Path):
        """
        Args:
            config: ForcingConfig with ADT settings
            input_path: Root data path (COMINrtofs parent or COMINadt)
        """
        self.config = config
        self.input_path = input_path

    def blend_ssh(self, ssh_path: Path, work_dir: Path) -> Optional[Path]:
        """Blend ADT into RTOFS SSH_1.nc.

        Args:
            ssh_path: Path to SSH_1.nc (RTOFS-only)
            work_dir: Working directory for intermediate files

        Returns:
            Path to updated SSH_1.nc with ADT blending, or None if ADT unavailable.
        """
        if not HAS_NETCDF4:
            log.warning("netCDF4 required for ADT blending")
            return None

        # Find ADT data files
        adt_data = self._find_adt_data()
        if adt_data is None:
            log.info("No ADT satellite data available — using RTOFS-only SSH")
            return None

        # Find weight file (for weighted blending if available)
        weight_path = self._find_weight_file()

        try:
            # Read ADT and subset to domain
            adt_ssh = self._read_adt(adt_data)
            if adt_ssh is None:
                return None

            # Apply ADT blending formula to SSH_1.nc
            output = self._apply_adt_blend(ssh_path, adt_ssh, work_dir,
                                           weight_path=weight_path)
            return output

        except Exception as e:
            log.warning(f"ADT blending failed: {e}")
            return None

    def _find_adt_data(self) -> Optional[Path]:
        """Find CMEMS ADT satellite data file.

        Searches COMINadt directory structure:
            {COMINadt}/{date}/validation_data/marine/cmems/ssh/
                nrt_global_allsat_phy_l4_{date}_{date}.nc
        """
        base_date = datetime.strptime(self.config.pdy, "%Y%m%d")

        # Check environment variable first
        comin_adt = os.environ.get("COMINadt", "")

        # Search today and previous day
        for offset in [0, -1]:
            date = base_date + timedelta(days=offset)
            date_str = date.strftime("%Y%m%d")

            # Standard WCOSS path
            if comin_adt:
                adt_file = (Path(comin_adt) / date_str /
                           "validation_data" / "marine" / "cmems" / "ssh" /
                           f"nrt_global_allsat_phy_l4_{date_str}_{date_str}.nc")
                if adt_file.exists():
                    log.info(f"Found ADT data: {adt_file.name}")
                    return adt_file

            # Local path fallback
            for parent in [self.input_path, self.input_path.parent]:
                adt_file = parent / f"adt_{date_str}.nc"
                if adt_file.exists():
                    return adt_file

        return None

    def _find_weight_file(self) -> Optional[Path]:
        """Find ADT regridding weight file."""
        fix_dir = os.environ.get("FIXstofs3d", "")
        if fix_dir:
            wt = Path(fix_dir) / "stofs_3d_atl_adt_weight.nc"
            if wt.exists():
                return wt
        return None

    def _read_adt(self, adt_path: Path) -> Optional[np.ndarray]:
        """Read and subset ADT data to the eastern boundary domain."""
        try:
            ds = Dataset(str(adt_path))

            # Find coordinate variables
            lon_name = "longitude" if "longitude" in ds.variables else "lon"
            lat_name = "latitude" if "latitude" in ds.variables else "lat"

            lons = np.array(ds.variables[lon_name][:])
            lats = np.array(ds.variables[lat_name][:])

            # Subset to ADT domain
            lon_mask = (lons >= ADT_LON_MIN) & (lons <= ADT_LON_MAX)
            lat_mask = (lats >= ADT_LAT_MIN) & (lats <= ADT_LAT_MAX)

            lon_idx = np.where(lon_mask)[0]
            lat_idx = np.where(lat_mask)[0]

            if len(lon_idx) == 0 or len(lat_idx) == 0:
                ds.close()
                log.warning("ADT data doesn't cover target domain")
                return None

            # Read ADT variable
            adt_var = "adt" if "adt" in ds.variables else "surf_el"
            adt_data = ds.variables[adt_var]
            if adt_data.ndim == 3:
                # (time, lat, lon) — take mean across time if multiple
                subset = np.ma.filled(
                    adt_data[:, lat_idx[0]:lat_idx[-1]+1, lon_idx[0]:lon_idx[-1]+1],
                    fill_value=np.nan,
                )
                adt_mean = np.nanmean(subset, axis=0)
            else:
                adt_mean = np.ma.filled(
                    adt_data[lat_idx[0]:lat_idx[-1]+1, lon_idx[0]:lon_idx[-1]+1],
                    fill_value=np.nan,
                )

            ds.close()
            log.info(f"Read ADT: shape={adt_mean.shape}, "
                     f"range=[{np.nanmin(adt_mean):.3f}, {np.nanmax(adt_mean):.3f}]m")
            return adt_mean

        except Exception as e:
            log.warning(f"Failed to read ADT: {e}")
            return None

    def _apply_adt_blend(
        self, ssh_path: Path, adt_ssh: np.ndarray, work_dir: Path,
        weight_path: Optional[Path] = None,
    ) -> Optional[Path]:
        """Apply ADT blending formula to SSH_1.nc.

        Formula: SSH_final(t) = SSH_rtofs(t) - SSH_rtofs(t=0) + ADT
        This removes RTOFS bias at t=0 and replaces with satellite observation.
        """
        try:
            import shutil
            output = work_dir / "SSH_1_adt.nc"
            shutil.copy2(ssh_path, output)

            ds = Dataset(str(output), "r+")
            ssh = ds.variables["ssh"]

            # Extract first timestep reference
            ssh_t0 = ssh[0, :, :].copy()
            # Fill extreme values with 0 (matching NCO: where(abs>1000) = 0)
            ssh_t0 = np.where(np.abs(ssh_t0) > 1000, 0.0, ssh_t0)

            nt = ssh.shape[0]
            ny_ssh, nx_ssh = ssh.shape[1], ssh.shape[2]
            ny_adt, nx_adt = adt_ssh.shape

            # If ADT grid doesn't match SSH grid, we need interpolation
            # For now, apply ADT only where grids overlap (eastern boundary)
            # The ADT correction is typically small (~0.05-0.10m)
            if ny_adt == ny_ssh and nx_adt == nx_ssh:
                # Same grid — direct application
                for t in range(nt):
                    ssh_t = ssh[t, :, :]
                    ssh_corrected = ssh_t - ssh_t0 + adt_ssh
                    ssh_corrected = np.where(np.abs(ssh_corrected) > 1000,
                                             -30000.0, ssh_corrected)
                    ssh[t, :, :] = ssh_corrected
            else:
                # Different grid size — apply as scalar correction
                # Use mean ADT as uniform offset (simplified)
                adt_mean = float(np.nanmean(adt_ssh))
                log.info(f"ADT grid mismatch ({ny_adt}x{nx_adt} vs {ny_ssh}x{nx_ssh}), "
                         f"applying mean ADT correction: {adt_mean:.4f}m")
                for t in range(nt):
                    ssh_t = ssh[t, :, :]
                    ssh_corrected = ssh_t - ssh_t0 + adt_mean
                    ssh_corrected = np.where(np.abs(ssh_corrected) > 1000,
                                             -30000.0, ssh_corrected)
                    ssh[t, :, :] = ssh_corrected

            # Update surf_el with scaled values
            if "surf_el" in ds.variables:
                for t in range(nt):
                    data = ssh[t, :, :].copy()
                    data = np.where(np.abs(data) < 1000, data * 1000.0, -3000.0)
                    ds.variables["surf_el"][t, :, :] = data

            ds.close()
            log.info(f"ADT blending applied to {output.name}")
            return output

        except Exception as e:
            log.warning(f"ADT blend failed: {e}")
            return None
