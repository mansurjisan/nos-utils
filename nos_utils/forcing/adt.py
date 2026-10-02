"""
ADT (Absolute Dynamic Topography) satellite SSH blender.

Blends CMEMS satellite ADT observations with RTOFS SSH to improve
boundary condition accuracy for STOFS-3D-ATL.

The core formula (ops stofs_3d_atl_create_obc_3d_th_non_adjust.sh, SSH_1.nc block):
    SSH_final = SSH_rtofs - SSH_rtofs(t=0) + (ADT - 0.45)

ADT is the mean of the valid-day and previous-day CMEMS files (ncra), mapped onto
the SSH_1 grid with the ops ESMF map (ncremap with stofs_3d_atl_adt_weight.nc,
which is a nearest-source map with S=1 despite its "bilinear" global attribute).
The -0.45 m datum offset (stofs_3d_atl_adt_cvtz.nco) is applied per day and rounded
to float32 before the mean, as ops does. Without the map, a bilinear regrid is used
and the result is not ops-exact.

This removes the RTOFS bias at t=0 and replaces it with the satellite-observed
absolute dynamic topography, preserving RTOFS temporal variability.

Input:
  - SSH_1.nc — RTOFS SSH prepared by RTOFSProcessor._stofs_prepare_ssh()
  - CMEMS ADT: nrt_global_allsat_phy_l4_YYYYMMDD_YYYYMMDD.nc
  - Weight file: stofs_3d_atl_adt_weight.nc (regridding weights)

Output:
  - SSH_1.nc updated with ADT-blended surf_el values

Fallback as ops (non_adjust.sh:525-537): with no ADT file, the previous cycle's archived
adt_aft_cvtz_cln.nc is reused with a warning. Only when that is missing too does
blend_ssh return None (RTOFS-only SSH).
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

# Ops holds ADT-missing cells as the CMEMS int fill carried through float32. MJ (10/02/26)
ADT_OPS_FILL = -2147483647.0
ADT_MISSING_BELOW = -1.0e4
ADT_COORD_TOL = 1e-3
ADT_DST_TOL = 1e-2


class ADTBlender:
    """Blend CMEMS ADT satellite SSH with RTOFS SSH."""

    def __init__(self, config: ForcingConfig, input_path: Path, keep: bool = False,
                 archive_path: Optional[Path] = None, prev_dirs=()):
        """
        Args:
            config: ForcingConfig with ADT settings
            input_path: Root data path (COMINrtofs parent or COMINadt)
            keep: also write adt_on_rtofs.nc (the ops adt_aft_cvtz_cln.nc analogue)
            archive_path: write the ADT field here each cycle for the next cycle's fallback
            prev_dirs: previous-cycle directories searched for the archived field
        """
        self.config = config
        self.input_path = input_path
        self.keep = keep
        self.archive_path = Path(archive_path) if archive_path else None
        self.prev_dirs = [Path(d) for d in prev_dirs]
        self.regrid = None  # "esmf" | "bilinear" once blend_ssh has run
        self.warnings = []

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
        from ._log import log_input_files
        log_input_files(
            "ADT", adt_files, source="ADT", category="ocean",
            note=f"pdy={self.config.pdy} n={len(adt_files)}",
        )
        self.regrid = None
        self.warnings = []
        if not adt_files:
            prev = self._load_previous(ssh_path)
            if prev is None:
                log.warning("No ADT satellite data and no archived previous field — "
                            "using RTOFS-only SSH")
                return None
            self.regrid = "previous"
            return self._apply_adt_blend(ssh_path, None, work_dir, adt_dst=prev)
        try:
            adt_dst = self._regrid_esmf(adt_files, ssh_path)
            if adt_dst is not None:
                self.regrid = "esmf"
                return self._apply_adt_blend(ssh_path, None, work_dir, adt_dst=adt_dst)
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
            self.regrid = "bilinear"
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

    def _weight_file(self) -> Optional[Path]:
        """Configured ADT ESMF map, or None (warns when configured but missing)."""
        wt = getattr(self.config, "adt_weight_file", None)
        if not wt:
            return None
        wt = Path(wt)
        if not wt.is_file():
            self._warn(f"ADT weight file {wt} not found — bilinear ADT regrid, result is "
                       f"not ops-exact")
            return None
        return wt

    def _warn(self, msg):
        log.warning(msg)
        self.warnings.append(msg)

    @staticmethod
    def _subset_ops_roi(adt_path: Path):
        """ncks -d longitude,-62.5,-51.5 -d latitude,7.0,54.0: inclusive coordinate bounds.

        Returns (adt[lat, lon] in m with NaN for missing, lons, lats, n_time). MJ (10/02/26)
        """
        with Dataset(str(adt_path)) as ds:
            lon_name = "longitude" if "longitude" in ds.variables else "lon"
            lat_name = "latitude" if "latitude" in ds.variables else "lat"
            lons = np.array(ds.variables[lon_name][:], dtype=np.float64)
            lats = np.array(ds.variables[lat_name][:], dtype=np.float64)
            li = np.where((lons >= ADT_LON_MIN) & (lons <= ADT_LON_MAX))[0]
            la = np.where((lats >= ADT_LAT_MIN) & (lats <= ADT_LAT_MAX))[0]
            if li.size == 0 or la.size == 0:
                raise ValueError("ADT file does not cover the ops ROI")
            var = ds.variables["adt" if "adt" in ds.variables else "surf_el"]
            a = np.ma.filled(var[..., la[0]:la[-1] + 1, li[0]:li[-1] + 1], np.nan)
            a = np.asarray(a, dtype=np.float64)
            if a.ndim == 3:
                a = a[0]
        a[np.abs(a) > 1000] = np.nan
        return a, lons[li[0]:li[-1] + 1], lats[la[0]:la[-1] + 1]

    @staticmethod
    def _read_esmf_map(wt: Path):
        """(S, row0, col0, n_a, n_b, xc_a, yc_a, xc_b, yc_b, dst_dims, src_dims); row/col made 0-based."""
        with Dataset(str(wt)) as ds:
            for v in ds.variables.values():
                v.set_auto_maskandscale(False)

            def deg(name):
                v = ds.variables[name]
                x = np.asarray(v[:], dtype=np.float64)
                return np.rad2deg(x) if "rad" in str(getattr(v, "units", "")).lower() else x

            n_a = ds.dimensions["n_a"].size
            n_b = ds.dimensions["n_b"].size
            S = np.asarray(ds.variables["S"][:], dtype=np.float64)
            row = np.asarray(ds.variables["row"][:], dtype=np.int64) - 1
            col = np.asarray(ds.variables["col"][:], dtype=np.int64) - 1
            dims = [np.asarray(ds.variables[k][:]).astype(int).tolist()
                    if k in ds.variables else None for k in ("dst_grid_dims", "src_grid_dims")]
            return (S, row, col, n_a, n_b, deg("xc_a"), deg("yc_a"), deg("xc_b"), deg("yc_b"),
                    dims[0], dims[1])

    @staticmethod
    def _apply_esmf_map(S, row, col, n_b, src):
        """dst = sum(S * src) over valid sources, as `ncremap -m`; NaN where none is valid."""
        v = np.isfinite(src[col])
        num = np.bincount(row, weights=np.where(v, S * np.where(v, src[col], 0.0), 0.0),
                          minlength=n_b)
        tally = np.bincount(row, weights=v.astype(np.float64), minlength=n_b)
        return np.where(tally > 0, num, np.nan)

    def _regrid_esmf(self, adt_files, ssh_path: Path):
        """Ops ADT step with the ESMF map: ROI subset, ncremap, float32(adt-0.45), two-day mean.

        Each day is rounded to float32 after the offset and the valid days are averaged
        (float64 accumulation, float32 result), as ncap2 and ncra do. Returns the float32
        field on the SSH_1 grid with NaN for missing cells, or None after recording a warning.
        """
        wt = self._weight_file()
        if wt is None:
            if not self.warnings:
                self._warn("ADT: no ESMF map configured — bilinear ADT regrid, result is "
                           "not ops-exact")
            return None
        try:
            S, row, col, n_a, n_b, xca, yca, xcb, ycb, dst_dims, src_dims = self._read_esmf_map(wt)
            with Dataset(str(ssh_path)) as ds:
                dlon = np.ma.filled(ds.variables["xlon"][:], np.nan).astype(np.float64)
                dlat = np.ma.filled(ds.variables["ylat"][:], np.nan).astype(np.float64)
            if dlon.ndim == 1:
                dlon, dlat = np.meshgrid(dlon, dlat)
            if dlon.size != n_b:
                raise ValueError(f"map n_b={n_b} != SSH_1 grid size {dlon.size}")
            if dst_dims and tuple(dst_dims) != (dlon.shape[1], dlon.shape[0]):
                raise ValueError(f"map dst_grid_dims {dst_dims} != SSH_1 (nx, ny) "
                                 f"{(dlon.shape[1], dlon.shape[0])}")

            def dlon_diff(a, b):
                return np.abs((a - b + 180.0) % 360.0 - 180.0).max()

            if (dlon_diff(xcb, dlon.ravel()) > ADT_DST_TOL
                    or np.abs(ycb - dlat.ravel()).max() > ADT_DST_TOL):
                raise ValueError("map xc_b/yc_b do not match the SSH_1 grid")
            if row.min() < 0 or row.max() >= n_b or col.min() < 0 or col.max() >= n_a:
                raise ValueError("map row/col out of range")

            days = []
            for f in adt_files:
                a, lons, lats = self._subset_ops_roi(f)
                if a.size != n_a:
                    raise ValueError(f"ADT ROI subset has {a.size} cells, map n_a={n_a}")
                if src_dims and tuple(src_dims) != (a.shape[1], a.shape[0]):
                    raise ValueError(f"map src_grid_dims {src_dims} != ADT (nlon, nlat) "
                                     f"{(a.shape[1], a.shape[0])}")
                LO, LA = np.meshgrid(lons, lats)
                if (dlon_diff(xca, LO.ravel()) > ADT_COORD_TOL
                        or np.abs(yca - LA.ravel()).max() > ADT_COORD_TOL):
                    raise ValueError("ADT ROI coordinates do not match map xc_a/yc_a")
                out = self._apply_esmf_map(S, row, col, n_b, a.ravel())
                days.append((out - 0.45).astype(np.float32))
            stack = np.stack(days)
            ok = np.isfinite(stack)
            cnt = ok.sum(axis=0)
            tot = np.where(ok, stack, 0.0).astype(np.float64).sum(axis=0)
            with np.errstate(all="ignore"):
                field = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
            field = field.astype(np.float32).reshape(dlon.shape)
        except Exception as e:
            self._warn(f"ADT ESMF regrid unavailable ({e}) — bilinear ADT regrid, result is "
                       f"not ops-exact")
            return None
        log.info(f"ADT: ESMF map {wt.name} applied to {len(days)} day(s), "
                 f"{int(np.isfinite(field).sum())}/{field.size} valid")
        return field

    def _write_adt_on_rtofs(self, adt_dst, dst_lon, dst_lat, out: Path):
        """adt_aft_cvtz_cln.nc analogue: float32 surf_el(time=1, ylat, xlon) in m after -0.45.

        Missing cells carry the ops fill (the CMEMS int fill held as float32).
        """
        ny, nx = adt_dst.shape
        with Dataset(str(out), "w", format="NETCDF4") as nc:
            nc.createDimension("time", 1)
            nc.createDimension("ylat", ny)
            nc.createDimension("xlon", nx)
            nc.createVariable("lon", "f8", ("ylat", "xlon"))[:] = dst_lon
            nc.createVariable("lat", "f8", ("ylat", "xlon"))[:] = dst_lat
            v = nc.createVariable("surf_el", "f4", ("time", "ylat", "xlon"),
                                  fill_value=np.float32(ADT_OPS_FILL))
            v.units = "m"
            v[0] = np.where(np.isfinite(adt_dst), adt_dst, np.float32(ADT_OPS_FILL))

    def _load_previous(self, ssh_path: Path):
        """Previous cycle's archived ADT field on the SSH_1 grid (NaN = missing), else None.

        Ops reuses yesterday's adt_aft_cvtz_cln.nc when no ADT file exists (non_adjust.sh:525-537).
        """
        name = self.archive_path.name if self.archive_path else None
        cands = []
        for d in self.prev_dirs:
            if name:
                cands.append(d / name)
            cands.extend(sorted(d.glob("*.adt_aft_cvtz_cln.nc"), reverse=True))
        try:
            with Dataset(str(ssh_path)) as ds:
                shape = ds.variables["ssh"].shape[1:]
        except Exception as e:
            log.warning(f"Cannot read SSH_1 grid for the archived ADT fallback ({e})")
            return None
        for c in dict.fromkeys(cands):
            if not c.is_file():
                continue
            try:
                with Dataset(str(c)) as ds:
                    v = ds.variables["surf_el"]
                    v.set_auto_maskandscale(False)
                    f = np.asarray(v[0], dtype=np.float32)
            except Exception as e:
                log.warning(f"Cannot read archived ADT {c}: {e}")
                continue
            if f.shape != tuple(shape):
                log.warning(f"Archived ADT {c} has shape {f.shape}, SSH_1 is {tuple(shape)} — skipped")
                continue
            f = np.where(f < ADT_MISSING_BELOW, np.float32(np.nan), f)
            self._warn(f"No ADT satellite data — reusing the previous cycle's ADT field {c}, "
                       f"as ops does")
            return f
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

        Fallback when the ops ESMF map is unavailable (the ops map is nearest-source, so
        this is not ops-exact): source NaNs (land) are excluded, the remaining weights
        renormalized, and destinations with no valid source stay NaN. MJ (10/02/26)
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

    def _apply_adt_blend(self, ssh_path: Path, adt_field, work_dir: Path,
                         adt_dst=None) -> Optional[Path]:
        """Apply the ops ADT blend to SSH_1.nc.

        Formula: SSH_final(t) = SSH_rtofs(t) - SSH_rtofs(t=0) + ADT
        (ADT already carries the -0.45 offset) with ADT regridded bilinearly
        onto the SSH_1 grid. Where either side is invalid the point is left
        as the -30000 fill.
        """
        try:
            import shutil
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

            if adt_dst is None:
                adt, adt_lons, adt_lats = adt_field
                adt_dst = self._regrid_bilinear(adt, adt_lons, adt_lats, dst_lon_adt, dst_lat)
                rtofs_wet = np.abs(np.ma.filled(ssh[0, :, :], fill_value=-30000.0)) < 1000
                adt_dst = self._fill_nearest(adt, adt_lons, adt_lats, dst_lon_adt, dst_lat,
                                             adt_dst, rtofs_wet)
            n_ok = int(np.isfinite(adt_dst).sum())
            if n_ok == 0:
                ds.close()
                log.warning("ADT regrid produced no valid points on the SSH grid — RTOFS-only SSH")
                return None
            if self.keep:
                self._write_adt_on_rtofs(adt_dst, dst_lon, dst_lat, work_dir / "adt_on_rtofs.nc")
            if self.archive_path:
                self._write_adt_on_rtofs(adt_dst, dst_lon, dst_lat, self.archive_path)
            log.info(f"ADT blend: {self.regrid or 'bilinear'} regrid -> {ny}x{nx} "
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
