"""
ADT (Absolute Dynamic Topography) satellite SSH blender.

Blends CMEMS satellite ADT observations with RTOFS SSH to improve
boundary condition accuracy for STOFS-3D-ATL.

The core formula (ops stofs_3d_atl_create_obc_3d_th_non_adjust.sh, SSH_1.nc block):
    SSH_final = SSH_rtofs - SSH_rtofs(t=0) + (ADT - 0.45)

ADT is the mean of the valid-day and previous-day CMEMS files (ncra), regridded
bilinearly onto the SSH_1 grid (ops: ncremap with stofs_3d_atl_adt_weight.nc,
ESMF bilinear). The -0.45 m datum offset (stofs_3d_atl_adt_cvtz.nco) is applied
once, in _read_adt.

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
from .rtofs import _pack_for_fortran

log = logging.getLogger(__name__)

try:
    from netCDF4 import Dataset
    HAS_NETCDF4 = True
except ImportError:
    HAS_NETCDF4 = False

# Ops ADT subset ROI, used only when the SSH_1 destination grid cannot be read.
# MJ (10/01/26)
ADT_LON_MIN = -62.5
ADT_LON_MAX = -51.5
ADT_LAT_MIN = 7.0
ADT_LAT_MAX = 54.0
ADT_PAD = 1.0


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

        adt_files = self._find_adt_files()
        if not adt_files:
            log.warning("No ADT satellite data available — using RTOFS-only SSH")
            return None

        try:
            bounds = self._dest_bounds(ssh_path)
            fields = [self._read_adt(f, bounds) for f in adt_files]
            fields = [f for f in fields if f is not None]
            if not fields:
                return None
            lons, lats = fields[0][1], fields[0][2]
            fields = [f for f in fields if f[0].shape == fields[0][0].shape]
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                adt = np.nanmean(np.stack([f[0] for f in fields]), axis=0)
            log.info(f"ADT: averaged {len(fields)} daily file(s)")
            return self._apply_adt_blend(ssh_path, (adt, lons, lats), work_dir)

        except Exception as e:
            log.warning(f"ADT blending failed: {e}")
            return None

    @staticmethod
    def _dest_bounds(ssh_path: Path):
        """ADT subset (lon_min, lon_max, lat_min, lat_max): SSH_1 grid extent padded
        by ADT_PAD (same -180/180 convention as ADT), else the ops ROI. MJ (10/01/26)
        """
        try:
            with Dataset(str(ssh_path)) as ds:
                lon = np.array(ds.variables["xlon"][:], dtype=np.float64)
                lat = np.array(ds.variables["ylat"][:], dtype=np.float64)
            return (float(np.nanmin(lon)) - ADT_PAD, float(np.nanmax(lon)) + ADT_PAD,
                    float(np.nanmin(lat)) - ADT_PAD, float(np.nanmax(lat)) + ADT_PAD)
        except Exception as e:
            log.warning(f"Cannot read SSH_1 grid extent ({e}) — using ops ADT ROI")
            return ADT_LON_MIN, ADT_LON_MAX, ADT_LAT_MIN, ADT_LAT_MAX

    def _find_adt_data(self) -> Optional[Path]:
        """Newest available CMEMS ADT file (valid day, else previous day)."""
        files = self._find_adt_files()
        return files[0] if files else None

    def _find_adt_files(self) -> list:
        """CMEMS ADT files for the valid day and the previous day, in that order.

        Searches COMINadt directory structure:
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

    def _read_adt(self, adt_path: Path, bounds=None):
        """Read and subset ADT to bounds, offset by -0.45 m.

        bounds is (lon_min, lon_max, lat_min, lat_max) in -180/180; default is the
        ops ROI. Returns (adt[lat, lon], lons, lats) or None. Fill values become NaN.
        """
        lon_min, lon_max, lat_min, lat_max = bounds or (
            ADT_LON_MIN, ADT_LON_MAX, ADT_LAT_MIN, ADT_LAT_MAX)
        try:
            ds = Dataset(str(adt_path))
            try:
                lon_name = "longitude" if "longitude" in ds.variables else "lon"
                lat_name = "latitude" if "latitude" in ds.variables else "lat"

                lons = np.array(ds.variables[lon_name][:], dtype=np.float64)
                lats = np.array(ds.variables[lat_name][:], dtype=np.float64)

                lon_idx = np.where((lons >= lon_min) & (lons <= lon_max))[0]
                lat_idx = np.where((lats >= lat_min) & (lats <= lat_max))[0]

                if len(lon_idx) == 0 or len(lat_idx) == 0:
                    log.warning("ADT data doesn't cover target domain")
                    return None

                adt_var = "adt" if "adt" in ds.variables else "surf_el"
                # 0.45 from NCO file
                # TODO: Should be different for different systems, e.g. PAC
                adt_data = ds.variables[adt_var][...] - 0.45
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
            finally:
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

    @staticmethod
    def _fill_nearest(adt, adt_lons, adt_lats, dst_lon, dst_lat, adt_dst, need):
        """Fill adt_dst where `need` and still NaN with the nearest valid source cell
        (ESMF nearest-source for unmapped points; plain lon/lat distance). MJ (10/01/26)
        """
        from scipy.spatial import cKDTree
        miss = need & ~np.isfinite(adt_dst)
        valid = np.isfinite(adt)
        if not miss.any() or not valid.any():
            return adt_dst
        LA, LO = np.meshgrid(adt_lats, adt_lons, indexing="ij")
        tree = cKDTree(np.column_stack([LO[valid], LA[valid]]))
        _, idx = tree.query(np.column_stack([dst_lon[miss], dst_lat[miss]]))
        out = adt_dst.copy()
        out[miss] = adt[valid][idx]
        return out

    def _apply_adt_blend(self, ssh_path: Path, adt_field, work_dir: Path) -> Optional[Path]:
        """Apply the ops ADT blend to SSH_1.nc.

        Formula: SSH_final(t) = SSH_rtofs(t) - SSH_rtofs(t=0) + ADT
        (ADT already carries the -0.45 offset) with ADT regridded bilinearly
        onto the SSH_1 grid. Where either side is invalid the point is left
        as the -30000 fill.
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
            # SSH_1 longitudes are RTOFS lon-360 (_stofs_prepare_ssh), same as ADT. MJ (10/01/26)
            dst_lon_adt = dst_lon

            adt_dst = self._regrid_bilinear(adt, adt_lons, adt_lats, dst_lon_adt, dst_lat)
            rtofs_wet = np.abs(np.ma.filled(ssh[0, :, :], fill_value=-30000.0)) < 1000
            adt_dst = self._fill_nearest(adt, adt_lons, adt_lats, dst_lon_adt, dst_lat,
                                         adt_dst, rtofs_wet)
            n_ok = int(np.isfinite(adt_dst).sum())
            if n_ok == 0:
                ds.close()
                log.warning("ADT regrid produced no valid points on the SSH grid — RTOFS-only SSH")
                return None
            log.info(f"ADT blend: bilinear regrid {adt.shape} -> {ny}x{nx} "
                     f"({n_ok}/{ny * nx} valid), mean(ADT)={np.nanmean(adt_dst):.4f}m")

            def _raw(t):
                return np.ma.filled(ssh[t, :, :], fill_value=-30000.0).astype(np.float64)

            # Fill extreme values with 0 (matching NCO: where(abs>1000) = 0)
            ssh_t0 = _raw(0)
            ssh_t0 = np.where(np.abs(ssh_t0) > 1000, 0.0, ssh_t0)

            for t in range(nt):
                ssh_t = _raw(t)
                corrected = ssh_t - ssh_t0 + adt_dst
                bad = ~np.isfinite(corrected) | (np.abs(ssh_t) > 1000) | (np.abs(corrected) > 1000)
                ssh[t, :, :] = np.where(bad, -30000.0, corrected)

            # Update surf_el: pack to match _stofs_prepare_ssh.
            # auto-maskandscale is ON for this r+ handle, so disable it on the
            # variable and pack manually with its own declared attrs; otherwise
            # netCDF re-applies scale_factor and double-packs.
            # Masked/extreme cells use the variable's -30000 fill.
            if "surf_el" in ds.variables:
                surf_el = ds.variables["surf_el"]
                surf_el.set_auto_maskandscale(False)
                fill = getattr(surf_el, "missing_value", -30000)
                for t in range(nt):
                    real = np.ma.filled(ssh[t, :, :], -30000.0).astype(np.float32)
                    ds.variables["surf_el"][t, :, :] = _pack_for_fortran(
                        real, surf_el.scale_factor, surf_el.add_offset,
                        fill, fill_mask=np.abs(real) >= 10000,
                    )

            ds.close()
            log.info(f"ADT blending applied to {output.name}")
            return output

        except Exception as e:
            log.warning(f"ADT blend failed: {e}")
            return None
