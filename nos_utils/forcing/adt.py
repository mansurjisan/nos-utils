"""
ADT (Absolute Dynamic Topography) satellite SSH blender.

Blends CMEMS satellite ADT observations with RTOFS SSH to improve
boundary condition accuracy for STOFS-3D-ATL.

The core formula (ops stofs_3d_atl_create_obc_3d_th_non_adjust.sh, SSH_1.nc block):
    SSH_final = SSH_rtofs - SSH_rtofs(t=0) + (ADT - 0.45)

ADT is the mean of the valid-day and previous-day CMEMS files (ncra), regridded
bilinearly onto the SSH_1 grid (ops: ncremap with stofs_3d_atl_adt_weight.nc,
ESMF bilinear), then offset by -0.45 m (stofs_3d_atl_adt_cvtz.nco).

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
from ..coords import normalize_lon, lon_convention

log = logging.getLogger(__name__)

try:
    from netCDF4 import Dataset
    HAS_NETCDF4 = True
except ImportError:
    HAS_NETCDF4 = False

# ADT subset domain — computed from config at runtime, not hardcoded here.
# Atlantic defaults kept as module-level fallbacks for callers that do not
# pass a config (e.g. legacy shell tests).
ADT_LON_MIN_DEFAULT = -62.5
ADT_LON_MAX_DEFAULT = -51.5
ADT_LAT_MIN = 7.0
ADT_LAT_MAX = 54.0

# Ops fix/stofs_3d_atl_adt_cvtz.nco: surf_el = adt - 0.45 (ADT to the MSL-like
# datum of the model boundary). Applied exactly once, here. MJ (10/01/26)
ADT_MSL_OFFSET = 0.45


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

        # ADT subset domain: derive from the config's lon/lat extent.
        # Pacific (0-360) needs the domain bounds converted to the ADT
        # product convention (-180/+180) for subsetting the CMEMS NetCDF.
        self._adt_lon_min = ADT_LON_MIN_DEFAULT
        self._adt_lon_max = ADT_LON_MAX_DEFAULT
        if hasattr(config, "lon_min") and hasattr(config, "lon_max"):
            _conv = lon_convention(config)
            if _conv == "0360":
                # Pacific: convert the domain lon_min/max from 0-360 to -180/180
                # so we can subset the CMEMS global ADT file (which uses -180/180)
                self._adt_lon_min = float(normalize_lon(
                    np.array([config.lon_min]), "pm180")[0])
                self._adt_lon_max = float(normalize_lon(
                    np.array([config.lon_max]), "pm180")[0])
            else:
                self._adt_lon_min = float(config.lon_min)
                self._adt_lon_max = float(config.lon_max)

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

        adt_files = self._find_adt_files()
        if not adt_files:
            log.info("No ADT satellite data available — using RTOFS-only SSH")
            return None

        try:
            fields = [self._read_adt(f) for f in adt_files]
            fields = [f for f in fields if f is not None]
            if not fields:
                return None
            lons, lats = fields[0][1], fields[0][2]
            fields = [f for f in fields if f[0].shape == fields[0][0].shape]
            with np.errstate(all="ignore"):
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    adt = np.nanmean(np.stack([f[0] for f in fields]), axis=0)
            log.info(f"ADT: averaged {len(fields)} daily file(s)")
            return self._apply_adt_blend(ssh_path, (adt, lons, lats), work_dir)

        except Exception as e:
            log.warning(f"ADT blending failed: {e}")
            return None

    def _find_adt_data(self) -> Optional[Path]:
        """Newest available CMEMS ADT file (valid day, else previous day)."""
        files = self._find_adt_files()
        return files[0] if files else None

    def _find_adt_files(self) -> list:
        """CMEMS ADT files for the valid day and the previous day, in that order.

        Searches COMINadt (or DCOMROOT) directory structure:
            {COMINadt}/{date}/validation_data/marine/cmems/ssh/
                nrt_global_allsat_phy_l4_{date}_{date}.nc
        Ops averages both days when both exist (ncra), so all hits are returned.
        """
        base_date = datetime.strptime(self.config.pdy, "%Y%m%d")

        # Operational WCOSS2 J-jobs don't always export COMINadt; the CMEMS ADT
        # files live under the same DCOM root ($DCOMROOT=/lfs/h1/ops/prod/dcom).
        comin_adt = os.environ.get("COMINadt") or os.environ.get("DCOMROOT", "")

        found = []
        for offset in [0, -1]:
            date_str = (base_date + timedelta(days=offset)).strftime("%Y%m%d")
            hit = None
            if comin_adt:
                adt_file = (Path(comin_adt) / date_str /
                           "validation_data" / "marine" / "cmems" / "ssh" /
                           f"nrt_global_allsat_phy_l4_{date_str}_{date_str}.nc")
                if adt_file.exists():
                    hit = adt_file
            if hit is None:
                for parent in [self.input_path, self.input_path.parent]:
                    adt_file = parent / f"adt_{date_str}.nc"
                    if adt_file.exists():
                        hit = adt_file
                        break
            if hit is not None:
                log.info(f"Found ADT data: {hit.name}")
                found.append(hit)
        return found

    def _find_weight_file(self) -> Optional[Path]:
        """Find ADT regridding weight file."""
        fix_dir = os.environ.get("FIXstofs3d", "")
        if fix_dir:
            wt = Path(fix_dir) / "stofs_3d_atl_adt_weight.nc"
            if wt.exists():
                return wt
        return None

    def _read_adt(self, adt_path: Path):
        """Read and subset ADT to the domain.

        Returns (adt[lat, lon], lons, lats) or None. Fill values become NaN.
        """
        try:
            ds = Dataset(str(adt_path))

            lon_name = "longitude" if "longitude" in ds.variables else "lon"
            lat_name = "latitude" if "latitude" in ds.variables else "lat"

            lons = np.array(ds.variables[lon_name][:], dtype=np.float64)
            lats = np.array(ds.variables[lat_name][:], dtype=np.float64)

            # self._adt_lon_min/max are always in -180/+180 to match CMEMS files
            lon_mask = (lons >= self._adt_lon_min) & (lons <= self._adt_lon_max)
            lat_mask = (lats >= ADT_LAT_MIN) & (lats <= ADT_LAT_MAX)

            lon_idx = np.where(lon_mask)[0]
            lat_idx = np.where(lat_mask)[0]

            if len(lon_idx) == 0 or len(lat_idx) == 0:
                ds.close()
                log.warning("ADT data doesn't cover target domain")
                return None

            adt_var = "adt" if "adt" in ds.variables else "surf_el"
            adt_data = ds.variables[adt_var]
            sl_y = slice(lat_idx[0], lat_idx[-1] + 1)
            sl_x = slice(lon_idx[0], lon_idx[-1] + 1)
            if adt_data.ndim == 3:
                subset = np.ma.filled(adt_data[:, sl_y, sl_x], fill_value=np.nan)
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    adt_2d = np.nanmean(subset, axis=0)
            else:
                adt_2d = np.ma.filled(adt_data[sl_y, sl_x], fill_value=np.nan)

            ds.close()
            adt_2d = np.asarray(adt_2d, dtype=np.float64)
            adt_2d[np.abs(adt_2d) > 1000] = np.nan
            log.info(f"Read ADT {adt_path.name}: shape={adt_2d.shape}, "
                     f"range=[{np.nanmin(adt_2d):.3f}, {np.nanmax(adt_2d):.3f}]m")
            return adt_2d, lons[sl_x], lats[sl_y]

        except Exception as e:
            log.warning(f"Failed to read ADT: {e}")
            return None

    @staticmethod
    def _regrid_bilinear(adt, adt_lons, adt_lats, dst_lon, dst_lat) -> np.ndarray:
        """Bilinear regrid of a regular lon/lat field onto 2D destination points.

        Equivalent in intent to the ops ncremap ESMF bilinear weights: source
        NaNs (land) are excluded and the remaining weights renormalized, and
        destinations with no valid source stay NaN. MJ (10/01/26)
        """
        from scipy.interpolate import RegularGridInterpolator
        valid = np.isfinite(adt)
        pts = np.column_stack([dst_lat.ravel(), dst_lon.ravel()])
        kw = dict(method="linear", bounds_error=False, fill_value=np.nan)
        num = RegularGridInterpolator((adt_lats, adt_lons), np.where(valid, adt, 0.0), **kw)(pts)
        den = RegularGridInterpolator((adt_lats, adt_lons), valid.astype(np.float64), **kw)(pts)
        with np.errstate(all="ignore"):
            out = np.where(den > 1e-6, num / den, np.nan)
        return out.reshape(dst_lon.shape)

    def _apply_adt_blend(self, ssh_path: Path, adt_field, work_dir: Path) -> Optional[Path]:
        """Apply the ops ADT blend to SSH_1.nc.

        Formula: SSH_final(t) = SSH_rtofs(t) - SSH_rtofs(t=0) + (ADT - 0.45)
        with ADT regridded (bilinear) onto the SSH_1 grid. Where either side is
        invalid the point is left as the -30000 fill.
        """
        try:
            import shutil
            adt, adt_lons, adt_lats = adt_field
            output = work_dir / "SSH_1_adt.nc"
            shutil.copy2(ssh_path, output)

            ds = Dataset(str(output), "r+")
            ssh = ds.variables["ssh"]
            nt, ny, nx = ssh.shape

            dst_lon = np.array(ds.variables["xlon"][:], dtype=np.float64)
            dst_lat = np.array(ds.variables["ylat"][:], dtype=np.float64)
            if dst_lon.ndim == 1:
                dst_lon, dst_lat = np.meshgrid(dst_lon, dst_lat)
            # SSH_1 keeps the RTOFS (0-360) longitudes; ADT is -180/180
            dst_lon_adt = np.where(dst_lon > 180.0, dst_lon - 360.0, dst_lon)

            adt_dst = self._regrid_bilinear(adt, adt_lons, adt_lats, dst_lon_adt, dst_lat)
            n_ok = int(np.isfinite(adt_dst).sum())
            if n_ok == 0:
                ds.close()
                log.warning("ADT regrid produced no valid points on the SSH grid — RTOFS-only SSH")
                return None
            adt_dst = adt_dst - ADT_MSL_OFFSET
            log.info(f"ADT blend: bilinear regrid {adt.shape} -> {ny}x{nx} "
                     f"({n_ok}/{ny * nx} valid), mean(ADT-{ADT_MSL_OFFSET})={np.nanmean(adt_dst):.4f}m")

            def _raw(t):
                return np.ma.filled(ssh[t, :, :], fill_value=-30000.0).astype(np.float64)

            ssh_t0 = _raw(0)
            ssh_t0 = np.where(np.abs(ssh_t0) > 1000, 0.0, ssh_t0)

            # surf_el is stored raw in mm with a scale_factor attribute (as ops/the
            # Fortran reader expects); disable netCDF4 auto-scaling to avoid
            # packing twice (x1e6). MJ (10/01/26)
            surf = ds.variables["surf_el"] if "surf_el" in ds.variables else None
            if surf is not None:
                surf.set_auto_maskandscale(False)

            for t in range(nt):
                ssh_t = _raw(t)
                corrected = ssh_t - ssh_t0 + adt_dst
                bad = ~np.isfinite(corrected) | (np.abs(ssh_t) > 1000) | (np.abs(corrected) > 1000)
                corrected = np.where(bad, -30000.0, corrected)
                ssh[t, :, :] = corrected
                if surf is not None:
                    surf[t, :, :] = np.where(bad, -3000.0, corrected * 1000.0)

            ds.close()
            log.info(f"ADT blending applied to {output.name}")
            return output

        except Exception as e:
            log.warning(f"ADT blend failed: {e}")
            return None
