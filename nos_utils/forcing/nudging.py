"""
T/S interior nudging field generator.

Creates SAL_nu.nc and TEM_nu.nc from RTOFS 3D temperature/salinity fields.
These files apply interior relaxation (nudging) toward observed T/S values
at specified nodes with a configurable timescale.

Input: RTOFS 3D NetCDF files (temperature, salinity on depth levels)

Output:

- TEM_nu.nc — temperature nudging field
- SAL_nu.nc — salinity nudging field

Replaces: nudging portion of nos_ofs_create_forcing_obc / gen_hycom_3Dth_nudge.py
"""

import logging
import os
import shutil
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..config import ForcingConfig
from .base import ForcingProcessor, ForcingResult

log = logging.getLogger(__name__)

try:
    from netCDF4 import Dataset
    HAS_NETCDF4 = True
except ImportError:
    HAS_NETCDF4 = False


class NudgingProcessor(ForcingProcessor):
    """
    Generate T/S interior nudging fields from RTOFS data.

    Nudging relaxes interior model T/S toward observed values with a
    configurable timescale. Only nodes identified in the nudging weight
    files (TEM_nudge.gr3, SAL_nudge.gr3) receive nudging.
    """

    SOURCE_NAME = "NUDGING"

    def __init__(
        self,
        config: ForcingConfig,
        input_path: Path,
        output_path: Path,
        nudge_weight_file: Optional[Path] = None,
        rtofs_input_path: Optional[Path] = None,
    ):
        """
        Args:
            config: ForcingConfig with nudging settings
            input_path: Directory with RTOFS 3D files (or pre-extracted T/S)
            output_path: Output directory for TEM_nu.nc, SAL_nu.nc
            nudge_weight_file: Path to nudging weight gr3 file (optional)
            rtofs_input_path: COMINrtofs root for STOFS nudge-specific data prep
                (uses nudge_roi_3d instead of obc_roi_3d)
        """
        super().__init__(config, input_path, output_path)
        self.nudge_weight_file = nudge_weight_file
        self.rtofs_input_path = rtofs_input_path

    @property
    def is_stofs_mode(self) -> bool:
        """True if using STOFS-style ROI-based nudging with Fortran exe."""
        return self.config.nudge_roi_3d is not None

    def process(self) -> ForcingResult:
        """Generate nudging fields."""
        if not HAS_NETCDF4:
            return ForcingResult(
                success=False, source=self.SOURCE_NAME,
                errors=["netCDF4 required for nudging"],
            )

        if not self.config.nudging_enabled:
            return ForcingResult(
                success=True, source=self.SOURCE_NAME,
                warnings=["Nudging disabled in config"],
            )

        log.info(f"Nudging processor: timescale={self.config.nudging_timescale_seconds}s "
                 f"mode={'STOFS' if self.is_stofs_mode else 'SECOFS'}")
        self.create_output_dir()

        # STOFS mode: try Fortran gen_nudge_from_hycom
        if self.is_stofs_mode:
            return self._process_stofs()

        return self._process_python()

    def _process_stofs(self) -> ForcingResult:
        """STOFS mode: prepare data with nudge-specific ROI, call Fortran exe.

        IMPORTANT: The nudge ROI (422,600,94,835) is wider than the OBC 3D ROI
        (482,600,94,821). We must prepare TS_1.nc with the nudge ROI, NOT reuse
        the OBC TS_1.nc which has a smaller domain.
        """
        import tempfile
        work_dir = Path(tempfile.mkdtemp(prefix="nudge_stofs_"))

        try:
            # Prepare TS_1.nc with nudge-specific ROI if we have RTOFS access
            if self.rtofs_input_path:
                self._prepare_nudge_tsuv(work_dir)

            # Try Fortran nudge executable
            fortran_ok = self._call_fortran_gen_nudge(work_dir)

            output_files = []
            warnings = []

            if fortran_ok:
                for fname in ["TEM_nu.nc", "SAL_nu.nc"]:
                    src = work_dir / fname
                    if src.exists():
                        dst = self.output_path / fname
                        shutil.copy2(src, dst)
                        output_files.append(dst)
            else:
                warnings.append("Fortran gen_nudge not available, using Python fallback")
                result = self._process_python()
                return result

            return ForcingResult(
                success=len(output_files) > 0,
                source=self.SOURCE_NAME,
                output_files=output_files,
                warnings=warnings,
                metadata={
                    "timescale_seconds": self.config.nudging_timescale_seconds,
                    "n_levels": self.config.n_levels,
                    "fortran_used": fortran_ok,
                },
            )
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    def _prepare_nudge_tsuv(self, work_dir: Path) -> Optional[Path]:
        """Prepare TS_1.nc with nudge-specific ROI (wider than OBC ROI).

        Shell script stofs_3d_atl_create_obc_nudge.sh does its own complete
        RTOFS data prep with nudge ROI {422,600,94,835} which is wider than
        OBC 3D ROI {482,600,94,821}.
        """
        from .rtofs import RTOFSProcessor

        roi = self.config.nudge_roi_3d
        if not roi:
            return None

        # Use RTOFSProcessor's subsetting with nudge ROI
        rtofs_proc = RTOFSProcessor(
            self.config, self.rtofs_input_path, work_dir,
        )
        _, files_3d = rtofs_proc.find_input_files_by_type()
        if not files_3d:
            log.warning("No RTOFS 3D files found for nudge data prep")
            return None

        # Use nudge-specific ROI (wider than OBC ROI)
        tsuv_path = rtofs_proc._stofs_prepare_tsuv(
            files_3d, work_dir, roi_override=roi,
        )
        if tsuv_path:
            # Symlink as TS_1.nc for Fortran
            ts_link = work_dir / "TS_1.nc"
            if not ts_link.exists():
                ts_link.symlink_to(tsuv_path)
            log.info(f"Prepared nudge TS_1.nc with ROI {roi}")
        return tsuv_path

    def _call_fortran_gen_nudge(self, work_dir: Path) -> bool:
        """Call Fortran stofs_3d_atl_gen_nudge_from_hycom."""
        exe_names = ["stofs_3d_atl_gen_nudge_from_hycom"]
        env_dirs = ["EXECstofs3d", "EXECnos", "EXECofs"]

        exe = None
        for env_var in env_dirs:
            exec_dir = os.environ.get(env_var)
            if not exec_dir:
                continue
            for name in exe_names:
                candidate = Path(exec_dir) / name
                if candidate.exists():
                    exe = candidate
                    break
            if exe:
                break

        if exe is None:
            log.debug("No Fortran gen_nudge_from_hycom executable found")
            return False

        # Symlink required files
        fix_dir = os.environ.get("FIXstofs3d", "")
        fix_files = {
            "hgrid.ll": "stofs_3d_atl_hgrid.ll",
            "hgrid.gr3": "stofs_3d_atl_hgrid.gr3",
            "vgrid.in": "stofs_3d_atl_vgrid.in",
            "estuary.gr3": "stofs_3d_atl_estuary.gr3",
            "TEM_nudge.gr3": "stofs_3d_atl_tem_nudge.gr3",
            "gen_nudge_from_nc.in": "stofs_3d_atl_obc_nudge_nc.in",
        }

        if fix_dir:
            for link_name, fix_name in fix_files.items():
                src = Path(fix_dir) / fix_name
                if src.exists():
                    target = work_dir / link_name
                    if not target.exists():
                        target.symlink_to(src)

        # Look for TS_1.nc in input_path (prepared by RTOFS processor)
        for ts_name in ["TS_1.nc", "TSUV_1.nc"]:
            ts_src = self.input_path / ts_name
            if ts_src.exists():
                target = work_dir / "TS_1.nc"
                if not target.exists():
                    target.symlink_to(ts_src)
                break

        try:
            result = subprocess.run(
                [str(exe)],
                cwd=str(work_dir),
                capture_output=True,
                text=True,
                timeout=600,
            )

            if result.returncode != 0:
                log.warning(f"gen_nudge returned {result.returncode}: {result.stderr[:300]}")
                return False

            found = sum(1 for f in ["TEM_nu.nc", "SAL_nu.nc"]
                       if (work_dir / f).exists())
            log.info(f"Fortran gen_nudge produced {found}/2 files")
            return found >= 1

        except subprocess.TimeoutExpired:
            log.warning("Fortran gen_nudge timed out")
            return False
        except Exception as e:
            log.warning(f"Error calling Fortran gen_nudge: {e}")
            return False

    def _process_python(self) -> ForcingResult:
        """Python implementation: interpolate RTOFS T/S to interior nudging nodes.

        Optimized pipeline:
        1. Read nudge weight file to identify nodes with weight > 0
        2. Find RTOFS 3D files with simulation time-window filtering
           (reduces ~32 raw files to ~10, matching OBC behavior)
        3. Precompute Delaunay interpolation weights ONCE from surface level,
           then apply via gather+multiply for all depth levels and files
        4. Vertically interpolate from RTOFS depth levels to SCHISM sigma levels
        5. Vectorized temporal interpolation (numpy, no per-node loop)
        6. Filter to nodes with valid RTOFS coverage
        7. Write TEM_nu.nc and SAL_nu.nc in COMF format
        """
        from .rtofs import RTOFSProcessor

        # --- 1. Identify nudging nodes ---
        nudge_node_ids, nudge_lons, nudge_lats, nudge_depths = \
            self._load_nudge_nodes()
        if nudge_node_ids is None:
            return ForcingResult(
                success=False, source=self.SOURCE_NAME,
                errors=["Cannot identify nudging nodes: need nudge_weight_file "
                        "and grid_file in config"],
            )

        n_nudge = len(nudge_node_ids)
        log.info(f"Nudging candidate nodes: {n_nudge:,}")

        # --- 2. Find RTOFS 3D files (with time-window filtering) ---
        # Nudging doesn't need more temporal resolution than OBC — it's a
        # slowly-varying relaxation field. We filter files to the simulation
        # window (nowcast_start to forecast_end) plus a 6h buffer, matching
        # the OBC time-window approach. This typically reduces ~32 raw files
        # to ~10, cutting I/O and interpolation time proportionally.
        rtofs_root = self.rtofs_input_path or self.input_path
        rtofs_proc = RTOFSProcessor(
            self.config, rtofs_root, self.output_path,
            phase="nowcast",   # enables time-window filtering
        )
        _, files_3d = rtofs_proc.find_input_files_by_type()

        if not files_3d:
            # Retry without phase filtering as fallback (e.g., when RTOFS
            # cycle date doesn't align with simulation window in test data)
            rtofs_proc_nophase = RTOFSProcessor(
                self.config, rtofs_root, self.output_path,
            )
            _, files_3d = rtofs_proc_nophase.find_input_files_by_type()
            # Transfer the cycle date for valid-time calculation
            if hasattr(rtofs_proc_nophase, '_rtofs_cycle_date'):
                rtofs_proc._rtofs_cycle_date = rtofs_proc_nophase._rtofs_cycle_date

        if not files_3d:
            return ForcingResult(
                success=False, source=self.SOURCE_NAME,
                errors=["No RTOFS 3D files found for nudging"],
            )

        log.info(f"Found {len(files_3d)} RTOFS 3D files for nudging")

        # --- 3. Load vertical grid ---
        vgrid = self._load_vgrid()
        n_levels = vgrid.nvrt if vgrid else self.config.n_levels

        # Load per-node sigma for nudge nodes (LSC2)
        if vgrid and vgrid._filepath is not None:
            vgrid.load_boundary_sigma(nudge_node_ids.tolist())

        # --- 4. Interpolate RTOFS to nudge nodes (optimized) ---
        # Instead of calling _interpolate_2d_to_boundary per depth level
        # (which rebuilds/evaluates the Delaunay triangulation each time),
        # we precompute interpolation weights once from the surface level
        # and reuse them for all levels and files via simple indexed ops.
        rtofs_proc._bnd_lons = nudge_lons
        rtofs_proc._bnd_lats = nudge_lats
        rtofs_proc._bnd_depths = nudge_depths
        rtofs_proc._bnd_ids = nudge_node_ids.tolist()
        rtofs_proc._vgrid = vgrid

        all_temp = []
        all_salt = []

        # Precomputed weight state: computed once from first file's surface
        interp_weights = None  # Will hold _NudgeInterpWeights

        for fi, f in enumerate(files_3d):
            try:
                ds = Dataset(str(f))
                lon = ds.variables.get("Longitude") or ds.variables.get("lon")
                lat = ds.variables.get("Latitude") or ds.variables.get("lat")
                depth_var = ds.variables.get("Depth") or ds.variables.get("lev")

                if lon is None or lat is None:
                    ds.close()
                    continue

                lon_arr = lon[:]
                lat_arr = lat[:]
                depth_arr = depth_var[:] if depth_var is not None else np.arange(n_levels)
                n_rtofs_levels = len(depth_arr)

                for var_name, target_list in [
                    ("temperature", all_temp),
                    ("salinity", all_salt),
                ]:
                    if var_name not in ds.variables:
                        continue

                    data = ds.variables[var_name][:]
                    data = np.ma.filled(data, fill_value=np.nan)
                    data[np.abs(data) >= 99.0] = np.nan

                    if data.ndim == 4:
                        n_times = data.shape[0]
                    else:
                        n_times = 1
                        data = data[np.newaxis, ...]

                    for t in range(n_times):
                        data_t = data[t]

                        # Build interpolation weights from the first
                        # valid surface slice, then reuse for everything.
                        if interp_weights is None:
                            interp_weights = self._build_interp_weights(
                                lon_arr, lat_arr, data_t[0],
                                nudge_lons, nudge_lats,
                            )
                            if interp_weights is not None:
                                log.info(
                                    f"Precomputed nudge interp weights: "
                                    f"{interp_weights.n_source} ocean pts "
                                    f"-> {n_nudge} target pts "
                                    f"(grid {lon_arr.shape})"
                                )

                        if interp_weights is not None:
                            bnd_profile_rtofs = self._apply_weights_all_levels(
                                interp_weights, data_t,
                                n_nudge, n_rtofs_levels,
                            )
                        else:
                            # Fallback: use RTOFSProcessor per-level interp
                            bnd_profile_rtofs = np.full(
                                (n_nudge, n_rtofs_levels), np.nan,
                                dtype=np.float32,
                            )
                            for lev in range(min(n_rtofs_levels, data_t.shape[0])):
                                bnd_profile_rtofs[:, lev] = \
                                    rtofs_proc._interpolate_2d_to_boundary(
                                        lon_arr, lat_arr, data_t[lev],
                                    )

                        # Vertically interpolate to SCHISM levels
                        if vgrid and n_levels != n_rtofs_levels:
                            bnd_profile = rtofs_proc._interpolate_vertical(
                                bnd_profile_rtofs, depth_arr, var_name, n_levels,
                            )
                        else:
                            bnd_profile = bnd_profile_rtofs

                        target_list.append(bnd_profile)

                ds.close()
            except Exception as e:
                log.warning(f"Failed to read RTOFS 3D file {f.name}: {e}")

        if not all_temp and not all_salt:
            return ForcingResult(
                success=False, source=self.SOURCE_NAME,
                errors=["Failed to interpolate RTOFS data to nudge nodes"],
            )

        # --- 5. Temporal interpolation: 6-hourly RTOFS -> 3-hourly output ---
        # Clip to simulation duration so output timesteps match COMF Fortran.
        rtofs_dt = 21600.0   # 6-hourly input
        target_dt = 10800.0  # 3-hourly output (matches COMF Fortran)

        cycle_dt = datetime.strptime(self.config.pdy, "%Y%m%d") + \
                   timedelta(hours=self.config.cyc)
        sim_start = cycle_dt - timedelta(hours=self.config.nowcast_hours)
        sim_end = cycle_dt + timedelta(hours=self.config.forecast_hours)
        sim_duration = (sim_end - sim_start).total_seconds()

        for var_list in [all_temp, all_salt]:
            if len(var_list) > 1:
                n_in = len(var_list)
                rtofs_times = np.arange(n_in) * rtofs_dt
                n_out = int((n_in - 1) * rtofs_dt / target_dt) + 1
                target_times = np.arange(n_out) * target_dt
                target_times = target_times[
                    target_times <= min(rtofs_times[-1], sim_duration)
                ]

                stacked = np.stack(var_list, axis=0)  # (n_in, n_nodes, n_levels)

                # Vectorized temporal interpolation using numpy.
                # Reshape to (n_in, n_nodes*n_levels), interp along axis 0,
                # then reshape back. np.interp is 1-D, so we transpose and
                # use a single apply_along_axis-free loop over the flat dim.
                orig_shape = stacked.shape[1:]  # (n_nodes, n_levels)
                flat = stacked.reshape(n_in, -1)  # (n_in, N)
                n_flat = flat.shape[1]
                interp_out = np.empty(
                    (len(target_times), n_flat), dtype=np.float32,
                )
                # np.interp is fast in C — one call per flat column is still
                # much faster than scipy.interp1d per (node, level) pair,
                # because we eliminated the Python overhead of 32K*63 interp1d
                # constructor calls. For further speed, we vectorize by
                # computing fractional indices and using linear combination.
                if len(target_times) > 0 and n_in > 1:
                    # Compute fractional indices into rtofs_times for each
                    # target time. This avoids any scipy overhead.
                    frac_idx = np.clip(
                        target_times / rtofs_dt, 0, n_in - 1,
                    )
                    idx_lo = np.floor(frac_idx).astype(np.intp)
                    idx_hi = np.minimum(idx_lo + 1, n_in - 1)
                    alpha = (frac_idx - idx_lo).astype(np.float32)

                    # Fully vectorized: gather + lerp
                    # flat[idx_lo] has shape (n_target_times, N)
                    val_lo = flat[idx_lo]  # (n_out, N)
                    val_hi = flat[idx_hi]  # (n_out, N)
                    interp_out[:] = val_lo + alpha[:, np.newaxis] * (val_hi - val_lo)
                elif len(target_times) == 1:
                    interp_out[0] = flat[0]

                interp_out = interp_out.reshape(
                    (len(target_times),) + orig_shape,
                )
                var_list.clear()
                var_list.extend(
                    [interp_out[t] for t in range(len(target_times))],
                )
                log.info(f"Temporally interpolated nudge field: "
                         f"{n_in} -> {len(target_times)} steps "
                         f"(sim_duration={sim_duration:.0f}s)")

        # --- 6. Filter to nodes with valid RTOFS coverage ---
        # A node is valid if it has non-NaN data at the surface level for the
        # first timestep. This matches the Fortran behavior where land nodes
        # in the RTOFS domain are excluded.
        if all_temp:
            reference = all_temp[0]  # (n_nudge, n_levels)
        else:
            reference = all_salt[0]

        # Check surface level (last column = shallowest in bottom-to-surface ordering)
        surface_valid = ~np.isnan(reference[:, -1])
        valid_indices = np.where(surface_valid)[0]
        n_valid = len(valid_indices)

        if n_valid == 0:
            return ForcingResult(
                success=False, source=self.SOURCE_NAME,
                errors=["No nudge nodes have valid RTOFS coverage"],
            )

        log.info(f"Nudge nodes with valid RTOFS: {n_valid:,} / {n_nudge:,}")

        # Subset to valid nodes
        valid_node_ids = nudge_node_ids[valid_indices]

        # --- 7. Write output files ---
        output_files = []
        dt_out = target_dt if len(all_temp) > 1 else rtofs_dt

        for var_list, label, out_name, units in [
            (all_temp, "TEM", "TEM_nu.nc", "degC"),
            (all_salt, "SAL", "SAL_nu.nc", "PSU"),
        ]:
            if not var_list:
                continue

            out_path = self.output_path / out_name
            stacked = np.stack(var_list, axis=0)  # (n_time, n_nudge, n_levels)
            # Subset to valid nodes
            stacked = stacked[:, valid_indices, :]

            self._write_nudge_nc(
                out_path, stacked, valid_node_ids, dt_out, label, units,
            )
            output_files.append(out_path)

        return ForcingResult(
            success=len(output_files) > 0,
            source=self.SOURCE_NAME,
            output_files=output_files,
            metadata={
                "timescale_seconds": self.config.nudging_timescale_seconds,
                "n_levels": n_levels,
                "n_nudge_nodes": n_valid,
                "n_timesteps": len(all_temp) if all_temp else len(all_salt),
                "dt_seconds": dt_out,
                "fortran_used": False,
            },
        )

    # ------------------------------------------------------------------
    # Precomputed-weight interpolation helpers
    # ------------------------------------------------------------------

    class _NudgeInterpWeights:
        """Stores precomputed Delaunay interpolation weights.

        Holds triangle vertex indices and barycentric weights for every
        target node.  Applying these to a new data field is a simple
        gather + weighted sum — no Delaunay evaluation needed.
        """
        __slots__ = (
            "tri_indices",     # (n_target, 3) int — simplex vertex indices into aug_pts
            "bary_weights",    # (n_target, 3) float32 — barycentric weights
            "n_corners",       # int — number of prepended corner points
            "n_source",        # int — number of ocean source points (excl. corners)
            "domain_mask",     # (n_flat,) bool — mask selecting domain bbox pts
            "ocean_mask_sub",  # (n_domain,) bool — ocean mask within domain subset
            "nan_fill_idx",    # (n_nan,) int — indices of target pts that got NaN
            "nan_donor_idx",   # (n_nan,) int — valid-target donor for each NaN pt
        )

        def __init__(self):
            pass

    def _build_interp_weights(
        self,
        rtofs_lon: np.ndarray,
        rtofs_lat: np.ndarray,
        surface_data: np.ndarray,
        target_lons: np.ndarray,
        target_lats: np.ndarray,
    ) -> "_NudgeInterpWeights":
        """Build Delaunay interpolation weights from a single 2D field.

        This is called ONCE using the surface level of the first RTOFS
        file.  It builds the Delaunay triangulation, locates every target
        node's enclosing simplex, and stores the barycentric weights.

        Subsequent calls to ``_apply_weights_all_levels`` use these
        weights to interpolate any 2D field on the same grid in O(N)
        time (indexed gather + weighted sum) instead of O(N log N)
        Delaunay evaluation.

        Returns None if scipy is unavailable or all data is masked.
        """
        try:
            from scipy.spatial import Delaunay, cKDTree
        except ImportError:
            return None

        target_pts = np.column_stack([target_lons, target_lats])

        # Flatten RTOFS grid (same logic as _interpolate_2d_to_boundary)
        # Ensure plain ndarray (not masked) for Delaunay
        rtofs_lon_arr = np.ma.filled(np.asarray(rtofs_lon), fill_value=np.nan)
        rtofs_lat_arr = np.ma.filled(np.asarray(rtofs_lat), fill_value=np.nan)
        surface_arr = np.ma.filled(np.asarray(surface_data), fill_value=np.nan)

        if rtofs_lon_arr.ndim == 2:
            lon_flat = rtofs_lon_arr.ravel()
            lat_flat = rtofs_lat_arr.ravel()
        else:
            lon_2d, lat_2d = np.meshgrid(rtofs_lon_arr, rtofs_lat_arr)
            lon_flat = lon_2d.ravel()
            lat_flat = lat_2d.ravel()

        data_flat = surface_arr.ravel()
        lon_flat = np.where(lon_flat > 180, lon_flat - 360, lon_flat)

        # Domain bounding-box subset
        buf = 1.0
        lon_min = target_lons.min() - buf
        lon_max = target_lons.max() + buf
        lat_min = target_lats.min() - buf
        lat_max = target_lats.max() + buf
        domain_mask = (
            (lon_flat >= lon_min) & (lon_flat <= lon_max) &
            (lat_flat >= lat_min) & (lat_flat <= lat_max)
        )

        if int(np.sum(domain_mask)) == 0:
            domain_mask = np.ones(len(lon_flat), dtype=bool)

        lon_sub = lon_flat[domain_mask]
        lat_sub = lat_flat[domain_mask]
        data_sub = data_flat[domain_mask]

        ocean_mask_sub = (np.abs(data_sub) < 99.0) & np.isfinite(data_sub)
        n_ocean = int(np.sum(ocean_mask_sub))
        if n_ocean == 0:
            return None

        ocean_lon = lon_sub[ocean_mask_sub]
        ocean_lat = lat_sub[ocean_mask_sub]
        ocean_val = data_sub[ocean_mask_sub].astype(np.float32)

        # Corner points (matching Fortran REMESH mode=0)
        avg_val = float(np.mean(ocean_val))
        tgt_lon_min, tgt_lon_max = target_pts[:, 0].min(), target_pts[:, 0].max()
        tgt_lat_min, tgt_lat_max = target_pts[:, 1].min(), target_pts[:, 1].max()
        lon_pad = 0.01 * (tgt_lon_max - tgt_lon_min)
        lat_pad = 0.01 * (tgt_lat_max - tgt_lat_min)
        corner_pts = np.array([
            [tgt_lon_min - lon_pad, tgt_lat_min - lat_pad],
            [tgt_lon_min - lon_pad, tgt_lat_max + lat_pad],
            [tgt_lon_max + lon_pad, tgt_lat_max + lat_pad],
            [tgt_lon_max + lon_pad, tgt_lat_min - lat_pad],
        ], dtype=np.float64)
        n_corners = 4

        aug_pts = np.vstack([corner_pts, np.column_stack([ocean_lon, ocean_lat])])
        aug_vals = np.concatenate([
            np.full(n_corners, avg_val, dtype=np.float32), ocean_val,
        ])

        # Build Delaunay and find simplices
        tri = Delaunay(aug_pts)
        simplex_ids = tri.find_simplex(target_pts)

        # Compute barycentric coordinates for each target point
        n_target = len(target_pts)
        tri_indices = np.zeros((n_target, 3), dtype=np.intp)
        bary_weights = np.zeros((n_target, 3), dtype=np.float32)

        valid_simplex = simplex_ids >= 0
        if np.any(valid_simplex):
            vs = valid_simplex
            s_ids = simplex_ids[vs]
            verts = tri.simplices[s_ids]  # (n_valid, 3) indices into aug_pts
            tri_indices[vs] = verts

            # Barycentric weights via transform
            # For each target point p in simplex s:
            #   bary[0:2] = T_inv @ (p - r3)
            #   bary[2]   = 1 - bary[0] - bary[1]
            T_inv = tri.transform[s_ids, :2]     # (n_valid, 2, 2)
            r3 = tri.transform[s_ids, 2]         # (n_valid, 2)
            dp = target_pts[vs] - r3             # (n_valid, 2)
            b = np.einsum('ijk,ik->ij', T_inv, dp)  # (n_valid, 2)
            bary_weights[vs, 0] = b[:, 0]
            bary_weights[vs, 1] = b[:, 1]
            bary_weights[vs, 2] = 1.0 - b[:, 0] - b[:, 1]

        # Handle NaN targets (outside convex hull) — fill from nearest
        # valid boundary node (same as _interpolate_2d_to_boundary)
        nan_fill_idx = np.array([], dtype=np.intp)
        nan_donor_idx = np.array([], dtype=np.intp)
        nan_mask = ~valid_simplex
        if np.any(nan_mask):
            valid_tgt = np.where(valid_simplex)[0]
            if len(valid_tgt) > 0:
                nan_tgt = np.where(nan_mask)[0]
                mean_lat = np.radians(np.mean(target_lats))
                corrected = target_pts.copy()
                corrected[:, 0] *= np.cos(mean_lat)
                bnd_tree = cKDTree(corrected[valid_tgt])
                _, nn = bnd_tree.query(corrected[nan_tgt])
                nan_fill_idx = nan_tgt
                nan_donor_idx = valid_tgt[nn]

        w = NudgingProcessor._NudgeInterpWeights()
        w.tri_indices = tri_indices
        w.bary_weights = bary_weights
        w.n_corners = n_corners
        w.n_source = n_ocean
        w.domain_mask = domain_mask
        w.ocean_mask_sub = ocean_mask_sub
        w.nan_fill_idx = nan_fill_idx
        w.nan_donor_idx = nan_donor_idx
        return w

    def _apply_weights_all_levels(
        self,
        weights: "_NudgeInterpWeights",
        data_3d: np.ndarray,
        n_target: int,
        n_levels: int,
    ) -> np.ndarray:
        """Apply precomputed interpolation weights to all depth levels.

        For each level:
        1. Flatten + domain subset + ocean mask the RTOFS field
        2. Prepend corner values (field mean)
        3. Gather vertex values using tri_indices
        4. Weighted sum using bary_weights

        This is O(n_levels * n_target) with very small constants —
        no Delaunay evaluation, no scipy calls.

        Args:
            weights: Precomputed from ``_build_interp_weights``
            data_3d: (n_levels, ny, nx) or (n_levels, n_flat) RTOFS field
            n_target: Number of nudge target nodes
            n_levels: Number of RTOFS depth levels

        Returns:
            (n_target, n_levels) float32 array
        """
        result = np.full((n_target, n_levels), np.nan, dtype=np.float32)
        tri_idx = weights.tri_indices     # (n_target, 3)
        bary_w = weights.bary_weights     # (n_target, 3)
        n_corners = weights.n_corners
        domain_mask = weights.domain_mask
        ocean_mask_sub = weights.ocean_mask_sub

        for lev in range(min(n_levels, data_3d.shape[0])):
            data_2d = data_3d[lev]
            data_flat = data_2d.ravel()
            data_sub = data_flat[domain_mask]

            # Ocean values within domain subset
            ocean_val = data_sub[ocean_mask_sub].astype(np.float32)
            # Mask land/fill
            bad = (np.abs(ocean_val) >= 99.0) | ~np.isfinite(ocean_val)
            ocean_val[bad] = np.nan

            # Augmented values: corners (mean) + ocean
            finite_mask = np.isfinite(ocean_val)
            if not np.any(finite_mask):
                # Entire level is land — leave as NaN
                continue
            avg_val = float(np.nanmean(ocean_val))
            aug_vals = np.empty(n_corners + len(ocean_val), dtype=np.float32)
            aug_vals[:n_corners] = avg_val
            aug_vals[n_corners:] = np.where(finite_mask, ocean_val, avg_val)

            # Gather + weighted sum
            v0 = aug_vals[tri_idx[:, 0]]
            v1 = aug_vals[tri_idx[:, 1]]
            v2 = aug_vals[tri_idx[:, 2]]
            interp_vals = bary_w[:, 0] * v0 + bary_w[:, 1] * v1 + bary_w[:, 2] * v2

            # Fill NaN targets from nearest valid target
            if len(weights.nan_fill_idx) > 0:
                interp_vals[weights.nan_fill_idx] = interp_vals[weights.nan_donor_idx]

            result[:, lev] = interp_vals

        return result

    def _load_nudge_nodes(self):
        """Load nudging node IDs, coordinates, and depths.

        Reads the nudge weight gr3 file (e.g., secofs.nudge.gr3) and returns
        all nodes with non-zero weight. If a grid_file is provided separately,
        depths come from there; otherwise from the gr3 file itself.

        Returns:
            (node_ids_1based, lons, lats, depths) or (None, None, None, None)
        """
        from ..io.schism_grid import SchismGrid

        nudge_file = self.nudge_weight_file
        if nudge_file is None:
            # Try to find it from FIX environment
            for env_var in ["FIXofs", "FIXstofs3d"]:
                fix_dir = os.environ.get(env_var)
                if not fix_dir:
                    continue
                for pattern in ["*.nudge.gr3", "*.TEM_nudge.gr3"]:
                    matches = sorted(Path(fix_dir).glob(pattern))
                    if matches:
                        nudge_file = matches[0]
                        break
                if nudge_file:
                    break

        if nudge_file is None or not Path(nudge_file).exists():
            log.warning(f"Nudge weight file not found: {nudge_file}")
            return None, None, None, None

        log.info(f"Reading nudge weight file: {nudge_file}")
        node_ids, lons, lats, values = SchismGrid.read_gr3_values(nudge_file)

        # Select nodes with positive weight
        mask = values > 0
        sel_ids = node_ids[mask]
        sel_lons = lons[mask]
        sel_lats = lats[mask]

        # Get depths from the grid file (gr3 value column is weight, not depth)
        grid_file = self.config.grid_file
        if grid_file and Path(grid_file).exists():
            grid = SchismGrid.read(grid_file, read_boundaries=False)
            sel_depths = grid.node_depths[sel_ids - 1]  # 1-based to 0-based
        else:
            # Fall back to depth=0 (vertical interpolation will use default sigma)
            log.warning("No grid_file for nudge node depths, using depth=0")
            sel_depths = np.zeros(len(sel_ids), dtype=np.float64)

        log.info(f"Nudge nodes: {len(sel_ids):,} with weight > 0")
        return sel_ids, sel_lons, sel_lats, sel_depths

    def _load_vgrid(self):
        """Load vertical grid if available."""
        # Try vgrid from FIX
        for env_var in ["FIXofs", "FIXstofs3d"]:
            fix_dir = os.environ.get(env_var)
            if not fix_dir:
                continue
            for name in ["*.vgrid.in", "vgrid.in"]:
                matches = sorted(Path(fix_dir).glob(name))
                if matches:
                    from ..io.schism_vgrid import SchismVgrid
                    return SchismVgrid.read(matches[0])

        # Try from config grid_file parent directory
        if self.config.grid_file:
            vgrid_path = Path(self.config.grid_file).parent / "vgrid.in"
            if not vgrid_path.exists():
                # Try OFS-prefixed
                grid_name = Path(self.config.grid_file).stem.split(".")[0]
                vgrid_path = Path(self.config.grid_file).parent / f"{grid_name}.vgrid.in"
            if vgrid_path.exists():
                from ..io.schism_vgrid import SchismVgrid
                return SchismVgrid.read(vgrid_path)

        log.warning("No vgrid.in found for nudging vertical interpolation")
        return None

    def _write_nudge_nc(
        self,
        output_path: Path,
        data: np.ndarray,
        node_ids: np.ndarray,
        dt: float,
        label: str,
        units: str,
    ) -> None:
        """Write TEM_nu.nc or SAL_nu.nc matching COMF Fortran format.

        Output format:
            time               (n_time,)          float64  — seconds from model start
            map_to_global_node (n_nodes,)          int32    — 1-based SCHISM node IDs
            tracer_concentration (n_time, n_nodes, n_levels, 1) float64

        Args:
            output_path: Path to write
            data: Array of shape (n_time, n_nodes, n_levels)
            node_ids: 1-based global node IDs
            dt: Time step in seconds
            label: "TEM" or "SAL"
            units: Variable units ("degC" or "PSU")
        """
        nt, n_nodes, n_levels = data.shape

        # Replace any remaining NaN with fill value
        fill_val = 15.0 if label == "TEM" else 35.0
        nan_mask = np.isnan(data)
        if np.any(nan_mask):
            n_nan = int(np.sum(nan_mask))
            log.info(f"Filling {n_nan} NaN values in {label}_nu with {fill_val}")
            data = np.where(nan_mask, fill_val, data)

        nc = Dataset(str(output_path), "w", format="NETCDF4")

        nc.createDimension("time", nt)
        nc.createDimension("node", n_nodes)
        nc.createDimension("nLevels", n_levels)
        nc.createDimension("one", 1)

        time_var = nc.createVariable("time", "f8", ("time",))
        time_var[:] = np.arange(nt, dtype=np.float64) * dt

        map_var = nc.createVariable("map_to_global_node", "i4", ("node",))
        map_var[:] = node_ids.astype(np.int32)

        tracer = nc.createVariable(
            "tracer_concentration", "f8",
            ("time", "node", "nLevels", "one"),
        )
        tracer[:, :, :, 0] = data.astype(np.float64)

        nc.close()
        log.info(f"Created {output_path.name}: {nt} times, {n_nodes:,} nodes, "
                 f"{n_levels} levels at dt={dt}s")

    def find_input_files(self) -> List[Path]:
        return sorted(self.input_path.glob("*temperature*.nc")) + \
               sorted(self.input_path.glob("*salinity*.nc"))
