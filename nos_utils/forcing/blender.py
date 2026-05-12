"""
HRRR + GFS atmospheric forcing blender for UFS-Coastal DATM.

Reads ``gfs_forcing.nc`` (global, 1D coords) and ``hrrr_forcing.nc``
(CONUS, 2D LCC coords) produced by GFS/HRRR processors via the
ForcingNcWriter, then blends them onto a regular lat/lon target grid.

Algorithm mirrors ``ush/python/nos_ofs/datm/blend_hrrr_gfs.py`` (the
operational shell pipeline blender) so output is byte-equivalent up
to floating-point ordering:

  1. Build ATLANTIC target grid from config.datm_domain at target_dx.
  2. Subset HRRR to the target box (with 1° buffer); precompute Delaunay
     triangulation + barycentric coordinates for HRRR LCC -> target.
  3. Subset GFS to the target box; flip latitude axis if descending.
  4. Build a unified hourly time grid covering the union of HRRR and
     GFS time ranges. HRRR-covered timesteps blend HRRR-over-GFS;
     timesteps beyond HRRR range fall back to GFS-only.
  5. For each output timestep:
     - Linear time-interpolate GFS values from the bracketing GFS times.
     - Bilinear spatial-interpolate GFS to the target grid.
     - If HRRR has a matching time and HRRR has spatial coverage at a
       cell, override GFS with HRRR at that cell.
  6. Apply Lambert Conformal wind rotation to HRRR-sourced winds
     (HRRR U/V are grid-relative; rotate to earth-relative).
  7. Write a single ``datm_forcing.nc`` with 2D ``latitude(y,x)`` and
     ``longitude(y,x)``, time in seconds since 1970, calendar=standard.

Output format matches CDEPS/ESMF expectations for the DATM component.
"""

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from ..config import ForcingConfig
from .base import ForcingProcessor, ForcingResult

log = logging.getLogger(__name__)

EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _epoch_seconds(dt: datetime) -> float:
    """Return seconds since 1970-01-01 UTC for a naive-or-aware datetime."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (dt - EPOCH).total_seconds()

try:
    from netCDF4 import Dataset
    HAS_NETCDF4 = True
except ImportError:
    HAS_NETCDF4 = False

# HRRR Lambert Conformal projection (CONUS)
HRRR_LOV = -97.5   # Longitude of vertical (degrees)
HRRR_LAD = 38.5    # Latitude of tangency (degrees)

# Variable mapping: output (HRRR) name -> (GFS name candidates)
# Some variables differ between sources (MSLMA vs PRMSL); we try the
# first that exists in the GFS file.
BLEND_VARIABLES = [
    ("UGRD_10maboveground", ["UGRD_10maboveground"]),
    ("VGRD_10maboveground", ["VGRD_10maboveground"]),
    ("TMP_2maboveground",   ["TMP_2maboveground"]),
    ("SPFH_2maboveground",  ["SPFH_2maboveground"]),
    ("PRATE_surface",       ["PRATE_surface"]),
    ("DSWRF_surface",       ["DSWRF_surface"]),
    ("DLWRF_surface",       ["DLWRF_surface"]),
    ("MSLMA_meansealevel",  ["MSLMA_meansealevel", "PRMSL_meansealevel"]),
]

FILL_VALUE = np.float32(9.999e20)


class BlenderProcessor(ForcingProcessor):
    """
    Blend HRRR + GFS forcing.nc files into a single datm_forcing.nc.

    Mirrors ``ush/python/nos_ofs/datm/blend_hrrr_gfs.py`` so output is
    semantically equivalent to the shell pipeline.
    """

    SOURCE_NAME = "BLENDER"

    # Default buffer (hours) appended past the forecast end so CDEPS
    # taxmode=limit has interpolation headroom on its pre-load look-ahead.
    # Matches the legacy COMF DATM window, which emits forcing through
    # ``forecast_end + 3h``. CDEPS aborts with
    # ``(shr_stream_findBounds) ERROR: rDateIn gt rDategvd limit true``
    # if the requested model time falls outside the forcing time bounds,
    # which happens near the end of the run when the DATM file stops
    # exactly at ``forecast_end``.
    DEFAULT_BUFFER_HOURS = 3

    def __init__(
        self,
        config: ForcingConfig,
        input_path: Path,
        output_path: Path,
        target_dx: float = 0.025,
        buffer_hours: Optional[int] = None,
    ):
        """
        Args:
            config: ForcingConfig with datm_domain bounds
            input_path: Directory containing gfs_forcing.nc and (optionally)
                hrrr_forcing.nc. Created by the ForcingNcWriter from the
                GFS/HRRR processors when nws=4.
            output_path: Output directory for datm_forcing.nc
            target_dx: Output grid resolution in degrees (default 0.025°)
            buffer_hours: Extra hourly records appended past the forecast
                end so CDEPS can interpolate without overrunning the
                forcing time axis. Defaults to
                ``config.datm_buffer_hours`` when set on the
                ``ForcingConfig``, otherwise to
                ``DEFAULT_BUFFER_HOURS`` (3h, matching legacy COMF).
        """
        super().__init__(config, input_path, output_path)
        self.target_dx = target_dx
        if buffer_hours is None:
            buffer_hours = getattr(
                config, "datm_buffer_hours", self.DEFAULT_BUFFER_HOURS,
            )
        self.buffer_hours = int(buffer_hours)

    def process(self) -> ForcingResult:
        if not HAS_NETCDF4:
            return ForcingResult(
                success=False, source=self.SOURCE_NAME,
                errors=["netCDF4 required for blending"],
            )

        try:
            from scipy.spatial import Delaunay
            from scipy.interpolate import RegularGridInterpolator, interp1d
        except ImportError:
            return ForcingResult(
                success=False, source=self.SOURCE_NAME,
                errors=["scipy required for blending (Delaunay + interp1d)"],
            )

        self.create_output_dir()

        gfs_path = self.input_path / "gfs_forcing.nc"
        hrrr_path = self.input_path / "hrrr_forcing.nc"

        if not gfs_path.exists():
            return ForcingResult(
                success=False, source=self.SOURCE_NAME,
                errors=[f"GFS forcing file not found: {gfs_path}"],
            )

        # Log the intermediate files we're about to consume.  These are
        # the GFS + HRRR forcing NetCDFs produced upstream by GFSProcessor
        # and HRRRProcessor in nws=4 mode -- without this line the prep
        # log shows NO indication of what the blender actually read.
        from ._log import log_input_files
        _datm_inputs = [gfs_path]
        if hrrr_path.exists():
            _datm_inputs.append(hrrr_path)
        log_input_files(
            "DATM", _datm_inputs,
            note=(
                f"target_dx={self.target_dx} "
                f"domain={self.config.datm_domain} "
                f"hrrr_present={hrrr_path.exists()}"
            ),
        )

        # Target grid (ATLANTIC for UFS-Coastal SECOFS)
        lon_min, lon_max, lat_min, lat_max = self.config.datm_domain
        target_lon = np.arange(
            lon_min, lon_max + self.target_dx / 2,
            self.target_dx, dtype=np.float32,
        )
        target_lat = np.arange(
            lat_min, lat_max + self.target_dx / 2,
            self.target_dx, dtype=np.float32,
        )
        target_lon2d, target_lat2d = np.meshgrid(target_lon, target_lat)
        ny, nx = target_lat2d.shape
        log.info(
            f"Blender target grid: {nx}x{ny} @ {self.target_dx}° "
            f"bounds=({lon_min:.4f},{lon_max:.4f},{lat_min:.4f},{lat_max:.4f})"
        )

        # ---- Open inputs ----
        gfs = Dataset(str(gfs_path), "r")
        gfs_lat_full = np.asarray(gfs.variables["latitude"][:], dtype=np.float32)
        gfs_lon_full = np.asarray(gfs.variables["longitude"][:], dtype=np.float32)
        gfs_time = np.asarray(gfs.variables["time"][:], dtype=np.float64)

        # GFS lon may be 0..360; convert to -180..180
        gfs_lon_180 = np.where(gfs_lon_full > 180, gfs_lon_full - 360, gfs_lon_full)

        # Subset GFS to target box (1° buffer)
        BUFFER = 1.0
        lat_mask = (gfs_lat_full >= lat_min - BUFFER) & (gfs_lat_full <= lat_max + BUFFER)
        lon_mask = (gfs_lon_180 >= lon_min - BUFFER) & (gfs_lon_180 <= lon_max + BUFFER)
        gfs_lat_idx = np.where(lat_mask)[0]
        gfs_lon_idx = np.where(lon_mask)[0]
        gfs_lat = gfs_lat_full[lat_mask]
        gfs_lon = gfs_lon_180[lon_mask]
        log.info(f"GFS subset: {len(gfs_lat)} x {len(gfs_lon)}, {len(gfs_time)} timesteps")

        if gfs_lat[0] > gfs_lat[-1]:
            gfs_lat_asc = gfs_lat[::-1]
            gfs_flip = True
        else:
            gfs_lat_asc = gfs_lat
            gfs_flip = False

        # ---- HRRR (optional) ----
        hrrr = None
        hrrr_time_raw = None
        hrrr_valid_mask = None
        bary_coords = None
        tri_vert_idx = None
        valid_flat_indices = None
        hrrr_row_slice = None
        hrrr_col_slice = None
        cos_rot = None
        sin_rot = None

        if hrrr_path.exists():
            hrrr = Dataset(str(hrrr_path), "r")
            hrrr_time_raw = np.asarray(hrrr.variables["time"][:], dtype=np.float64)
            hrrr_lon2d_full = np.asarray(hrrr.variables["longitude"][:], dtype=np.float32)
            hrrr_lat2d_full = np.asarray(hrrr.variables["latitude"][:], dtype=np.float32)

            # Subset HRRR to target box (1° buffer)
            hrrr_mask = (
                (hrrr_lon2d_full >= lon_min - BUFFER) &
                (hrrr_lon2d_full <= lon_max + BUFFER) &
                (hrrr_lat2d_full >= lat_min - BUFFER) &
                (hrrr_lat2d_full <= lat_max + BUFFER)
            )
            rows_with = np.any(hrrr_mask, axis=1)
            cols_with = np.any(hrrr_mask, axis=0)

            if rows_with.any() and cols_with.any():
                r0, r1 = np.where(rows_with)[0][[0, -1]]
                c0, c1 = np.where(cols_with)[0][[0, -1]]
                hrrr_row_slice = slice(r0, r1 + 1)
                hrrr_col_slice = slice(c0, c1 + 1)
                hrrr_lon2d = hrrr_lon2d_full[hrrr_row_slice, hrrr_col_slice]
                hrrr_lat2d = hrrr_lat2d_full[hrrr_row_slice, hrrr_col_slice]
                log.info(f"HRRR subset: {hrrr_lon2d.shape}, {len(hrrr_time_raw)} timesteps")

                # Delaunay triangulation
                hrrr_pts = np.column_stack([hrrr_lon2d.ravel(), hrrr_lat2d.ravel()])
                tri = Delaunay(hrrr_pts)
                target_pts_flat = np.column_stack([
                    target_lon2d.ravel(), target_lat2d.ravel(),
                ])
                simplices = tri.find_simplex(target_pts_flat)
                hrrr_valid_mask = (simplices >= 0).reshape(ny, nx)
                n_covered = int(hrrr_valid_mask.sum())
                log.info(f"HRRR covers {n_covered}/{ny*nx} target points "
                         f"({100*n_covered/(ny*nx):.1f}%)")

                # Barycentric coords for valid (covered) points
                valid_flat = simplices >= 0
                valid_flat_indices = np.where(valid_flat)[0]
                valid_simplices = simplices[valid_flat]
                valid_targets = target_pts_flat[valid_flat]
                tri_vert_idx = tri.simplices[valid_simplices]
                tri_verts = hrrr_pts[tri_vert_idx]
                T0 = tri_verts[:, 0, :]
                T1 = tri_verts[:, 1, :]
                T2 = tri_verts[:, 2, :]
                v0 = T1 - T0
                v1 = T2 - T0
                v2 = valid_targets - T0
                den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
                u = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / den
                v = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / den
                w = 1.0 - u - v
                bary_coords = np.column_stack([w, u, v]).astype(np.float32)

                # Wind rotation parameters (LC)
                D2R = np.pi / 180.0
                ROTCON = np.sin(HRRR_LAD * D2R)
                rot_angle = ROTCON * (target_lon2d - HRRR_LOV) * D2R
                cos_rot = np.cos(rot_angle).astype(np.float32)
                sin_rot = np.sin(rot_angle).astype(np.float32)
            else:
                log.warning("HRRR has no overlap with target domain — using GFS only")
                hrrr.close()
                hrrr = None
                hrrr_time_raw = None

        # ---- Build unified time grid anchored on model_t0 ----
        # DATM forcing must cover the full simulation window of the
        # coupled SCHISM run regardless of which prep phase generated
        # it (nowcast / forecast). CDEPS at runtime aborts if the
        # requested model time falls outside the forcing time bounds,
        # and the forecast prep overwrites the nowcast's DATM file, so
        # both phases write a file spanning the same model window:
        #
        #   model_t0     = cycle - nowcast_hours
        #   model_t_end  = cycle + forecast_hours + buffer_hours
        #   duration     = nowcast_hours + forecast_hours + buffer_hours
        #   n_times      = duration + 1                (hourly cadence)
        #
        # ``buffer_hours`` (default 3) appends extra hourly records
        # past the forecast end so CDEPS taxmode=limit has interpolation
        # headroom for its pre-load look-ahead. Without it CDEPS aborts
        # near the end of the run with
        # ``(shr_stream_findBounds) ERROR: rDateIn gt rDategvd limit true``
        # when the requested model time falls outside the forcing time
        # bounds. Matches legacy COMF, which emits DATM through
        # forecast_end + 3h.
        #
        # For SECOFS (nowcast=6h, forecast=48h, buffer=3h) this is 58
        # hourly records anchored at cycle-6h, covering
        # [cycle-6h, cycle+51h].
        #
        # Inputs (gfs_forcing.nc, hrrr_forcing.nc) may not span the
        # full window — e.g. nowcast-phase GFS only covers ~cycle-9h
        # to cycle+3h, so beyond cycle+3h we must extend forward. The
        # existing per-timestep GFS sampler clips ``t_low`` / ``t_high``
        # into ``[0, len(gfs_time)-1]`` and reuses the edge slab when
        # the target falls outside the input range, which is a
        # held-constant extrapolation in time. The HRRR time-match
        # mask (``hrrr_time_has``) only fires on exact hourly matches,
        # so points outside HRRR coverage automatically fall through
        # to the GFS path. Net result: target steps before the first
        # input or after the last input reuse the nearest edge slab,
        # which also covers the post-forecast buffer when GFS does not
        # extend that far.
        if hrrr_time_raw is not None and len(hrrr_time_raw) >= 2:
            dt = float(hrrr_time_raw[1] - hrrr_time_raw[0])
        elif len(gfs_time) >= 2:
            dt = float(gfs_time[1] - gfs_time[0])
        else:
            dt = 3600.0

        cycle_dt = datetime.strptime(self.config.pdy, "%Y%m%d") + \
                   timedelta(hours=self.config.cyc)
        model_t0 = cycle_dt - timedelta(hours=self.config.nowcast_hours)
        sim_duration_hours = (
            self.config.nowcast_hours
            + self.config.forecast_hours
            + self.buffer_hours
        )
        t_start = _epoch_seconds(model_t0)
        t_end = t_start + sim_duration_hours * 3600.0

        out_time = np.arange(t_start, t_end + dt / 2, dt)
        n_times = len(out_time)
        log.info(
            f"DATM time grid: anchored at model_t0={model_t0.isoformat()} "
            f"(cycle-{self.config.nowcast_hours}h), "
            f"covering {sim_duration_hours}h "
            f"(nowcast={self.config.nowcast_hours}h + "
            f"forecast={self.config.forecast_hours}h + "
            f"buffer={self.buffer_hours}h) = {n_times} hourly steps"
        )

        # Diagnose input vs. target time coverage so silent
        # held-constant edges are visible in the prep log.
        gfs_t0, gfs_t1 = float(gfs_time[0]), float(gfs_time[-1])
        if gfs_t0 > t_start or gfs_t1 < t_end:
            gap_start_h = max(0.0, (gfs_t0 - t_start) / 3600.0)
            gap_end_h = max(0.0, (t_end - gfs_t1) / 3600.0)
            log.warning(
                f"GFS input span [{gfs_t0:.0f}..{gfs_t1:.0f}] does not "
                f"cover target [{t_start:.0f}..{t_end:.0f}]: "
                f"{gap_start_h:.1f}h held-constant at start, "
                f"{gap_end_h:.1f}h at end"
            )

        if hrrr_time_raw is not None:
            hrrr_time_has = np.array([
                np.any(np.abs(hrrr_time_raw - ot) < dt / 2) for ot in out_time
            ])
            n_hrrr_covered_t = int(hrrr_time_has.sum())
            log.info(f"Time grid: {n_times} steps, HRRR covers {n_hrrr_covered_t}, "
                     f"GFS-only fill for {n_times - n_hrrr_covered_t}")
        else:
            hrrr_time_has = np.zeros(n_times, dtype=bool)
            log.info(f"Time grid: {n_times} steps, GFS-only (no HRRR)")

        gfs_time_to_idx = interp1d(
            gfs_time, np.arange(len(gfs_time)),
            kind="linear", bounds_error=False, fill_value="extrapolate",
        )
        target_to_gfs_idx = gfs_time_to_idx(out_time)

        # ---- Open output ----
        output_file = self.output_path / "datm_forcing.nc"
        ncout = Dataset(str(output_file), "w", format="NETCDF4_CLASSIC")
        ncout.createDimension("time", None)
        ncout.createDimension("y", ny)
        ncout.createDimension("x", nx)

        time_var = ncout.createVariable("time", "f8", ("time",))
        time_var.units = "seconds since 1970-01-01 00:00:00"
        time_var.calendar = "standard"
        time_var.standard_name = "time"
        time_var.axis = "T"
        time_var[:] = out_time

        lat_var = ncout.createVariable("latitude", "f4", ("y", "x"))
        lat_var.units = "degrees_north"
        lat_var.standard_name = "latitude"
        lat_var.long_name = "latitude"
        lat_var.axis = "Y"
        lat_var[:] = target_lat2d

        lon_var = ncout.createVariable("longitude", "f4", ("y", "x"))
        lon_var.units = "degrees_east"
        lon_var.standard_name = "longitude"
        lon_var.long_name = "longitude"
        lon_var.axis = "X"
        lon_var[:] = target_lon2d

        source_var = ncout.createVariable("data_source", "i1", ("y", "x"))
        source_var.long_name = "Data source (1=HRRR, 0=GFS)"
        if hrrr_valid_mask is not None:
            source_var[:] = hrrr_valid_mask.astype(np.int8)
        else:
            source_var[:] = np.zeros((ny, nx), dtype=np.int8)

        ncout.title = "Blended HRRR+GFS Forcing for CDEPS/DATM"
        ncout.source = "HRRR (CONUS) + GFS (gap fill)"
        ncout.history = f"Created {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')} by nos-utils BlenderProcessor"
        ncout.Conventions = "CF-1.6"

        # ---- Process each variable ----
        gfs_vars = list(gfs.variables.keys())
        hrrr_vars = list(hrrr.variables.keys()) if hrrr is not None else []

        for hrrr_name, gfs_candidates in BLEND_VARIABLES:
            gfs_name = next((n for n in gfs_candidates if n in gfs_vars), None)
            if gfs_name is None:
                log.warning(f"Skipping {hrrr_name}: not in GFS")
                continue

            hrrr_has_var = (hrrr is not None) and (hrrr_name in hrrr_vars)
            gfs_var = gfs.variables[gfs_name]
            hrrr_var = hrrr.variables[hrrr_name] if hrrr_has_var else None

            units = getattr(gfs_var, "units", "")
            long_name = getattr(gfs_var, "long_name", hrrr_name)

            out_var = ncout.createVariable(
                hrrr_name, "f4", ("time", "y", "x"),
                fill_value=FILL_VALUE,
            )
            out_var.short_name = hrrr_name
            out_var.units = units
            out_var.long_name = long_name

            for t in range(n_times):
                use_hrrr = hrrr_time_has[t] and hrrr_has_var

                # GFS time interpolation
                gfs_t_real = float(target_to_gfs_idx[t])
                t_low = int(np.floor(gfs_t_real))
                t_high = int(np.ceil(gfs_t_real))
                t_frac = gfs_t_real - t_low
                t_low = max(0, min(t_low, len(gfs_time) - 1))
                t_high = max(0, min(t_high, len(gfs_time) - 1))

                gfs_low = np.asarray(
                    gfs_var[t_low,
                            gfs_lat_idx[0]:gfs_lat_idx[-1] + 1,
                            gfs_lon_idx[0]:gfs_lon_idx[-1] + 1],
                    dtype=np.float32,
                )
                if t_low == t_high:
                    gfs_data = gfs_low
                else:
                    gfs_high = np.asarray(
                        gfs_var[t_high,
                                gfs_lat_idx[0]:gfs_lat_idx[-1] + 1,
                                gfs_lon_idx[0]:gfs_lon_idx[-1] + 1],
                        dtype=np.float32,
                    )
                    gfs_data = (1.0 - t_frac) * gfs_low + t_frac * gfs_high

                if gfs_flip:
                    gfs_data = gfs_data[::-1, :]

                gfs_interp = RegularGridInterpolator(
                    (gfs_lat_asc, gfs_lon), gfs_data,
                    method="linear", bounds_error=False, fill_value=np.nan,
                )
                gfs_regrid = gfs_interp(np.column_stack([
                    target_lat2d.ravel(), target_lon2d.ravel(),
                ])).reshape(ny, nx).astype(np.float32)

                if use_hrrr:
                    # Find matching HRRR time
                    hrrr_t = int(np.argmin(np.abs(hrrr_time_raw - out_time[t])))
                    if abs(hrrr_time_raw[hrrr_t] - out_time[t]) < dt / 2:
                        hrrr_slab = np.asarray(
                            hrrr_var[hrrr_t, hrrr_row_slice, hrrr_col_slice],
                            dtype=np.float32,
                        ).ravel()
                        hrrr_slab = np.where(hrrr_slab > 1e10, np.nan, hrrr_slab)

                        vals_at_verts = hrrr_slab[tri_vert_idx]
                        hrrr_interp_valid = np.sum(vals_at_verts * bary_coords, axis=1)
                        hrrr_regrid = np.full(ny * nx, np.nan, dtype=np.float32)
                        hrrr_regrid[valid_flat_indices] = hrrr_interp_valid
                        hrrr_regrid = hrrr_regrid.reshape(ny, nx)
                        combined = np.where(
                            hrrr_valid_mask & ~np.isnan(hrrr_regrid),
                            hrrr_regrid, gfs_regrid,
                        )
                    else:
                        combined = gfs_regrid
                else:
                    combined = gfs_regrid

                # CDEPS/CMEPS bilinear regrid does NOT filter _FillValue or
                # NaN, so any non-finite cell here will poison neighboring
                # SCHISM nodes after regrid (manifests as wind speed > 100
                # m/s and SCHISM aborts at step 2). Fill any remaining
                # non-finite cells with the slab mean — bounded data is
                # always preferable to fills here.
                bad = ~np.isfinite(combined)
                if bad.any():
                    n_bad = int(bad.sum())
                    finite_vals = combined[~bad]
                    if finite_vals.size > 0:
                        fill = float(finite_vals.mean())
                    else:
                        fill = 0.0
                    combined = np.where(bad, fill, combined)
                    log.warning(
                        f"  {hrrr_name} t={t}: filled {n_bad} non-finite "
                        f"cells with mean={fill:.3f}"
                    )

                out_var[t, :, :] = combined

        # ---- Lambert Conformal wind rotation (HRRR-sourced cells only) ----
        if (hrrr is not None and "UGRD_10maboveground" in ncout.variables
                and "VGRD_10maboveground" in ncout.variables and cos_rot is not None):
            log.info("Applying Lambert Conformal wind rotation to HRRR cells...")
            u_v = ncout.variables["UGRD_10maboveground"]
            v_v = ncout.variables["VGRD_10maboveground"]
            for t in range(n_times):
                if not hrrr_time_has[t]:
                    continue
                u_data = np.asarray(u_v[t, :, :], dtype=np.float32)
                v_data = np.asarray(v_v[t, :, :], dtype=np.float32)
                u_rot = np.where(
                    hrrr_valid_mask,
                    cos_rot * u_data + sin_rot * v_data, u_data,
                )
                v_rot = np.where(
                    hrrr_valid_mask,
                    -sin_rot * u_data + cos_rot * v_data, v_data,
                )
                u_v[t, :, :] = u_rot
                v_v[t, :, :] = v_rot

        ncout.close()
        gfs.close()
        if hrrr is not None:
            hrrr.close()

        log.info(f"Wrote {output_file.name}: {nx}x{ny} grid, {n_times} hourly steps")

        return ForcingResult(
            success=True, source=self.SOURCE_NAME,
            output_files=[output_file],
            metadata={
                "nx": nx, "ny": ny, "ntime": n_times,
                "target_dx": self.target_dx,
                "has_hrrr_blend": hrrr is not None,
            },
        )

    def find_input_files(self) -> List[Path]:
        """Discover input forcing files."""
        files = []
        gfs = self.input_path / "gfs_forcing.nc"
        hrrr = self.input_path / "hrrr_forcing.nc"
        if gfs.exists():
            files.append(gfs)
        if hrrr.exists():
            files.append(hrrr)
        return files
