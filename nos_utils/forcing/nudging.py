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

        Steps:
        1. Read nudge weight file to identify nodes with weight > 0
        2. Find RTOFS 3D files using same discovery as RTOFSProcessor
        3. For each RTOFS timestep and depth level, interpolate T/S to nudge nodes
        4. Vertically interpolate from RTOFS depth levels to SCHISM sigma levels
        5. Filter to nodes with valid RTOFS coverage
        6. Write TEM_nu.nc and SAL_nu.nc in COMF format
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

        # --- 2. Find RTOFS 3D files ---
        rtofs_root = self.rtofs_input_path or self.input_path
        rtofs_proc = RTOFSProcessor(
            self.config, rtofs_root, self.output_path,
        )
        _, files_3d = rtofs_proc.find_input_files_by_type()

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

        # --- 4. Interpolate RTOFS to nudge nodes ---
        # Set up the interpolator state on rtofs_proc so we can reuse
        # _interpolate_2d_to_boundary. We temporarily point its boundary
        # arrays at our nudge nodes.
        rtofs_proc._bnd_lons = nudge_lons
        rtofs_proc._bnd_lats = nudge_lats
        rtofs_proc._bnd_depths = nudge_depths
        rtofs_proc._bnd_ids = nudge_node_ids.tolist()
        rtofs_proc._vgrid = vgrid

        all_temp = []
        all_salt = []

        for f in files_3d:
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

                        # Interpolate each RTOFS depth level to nudge nodes
                        bnd_profile_rtofs = np.full(
                            (n_nudge, n_rtofs_levels), np.nan, dtype=np.float32,
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
        # Without clipping, raw RTOFS file count (e.g. 63) would produce far
        # more timesteps than the actual simulation needs (e.g. 19 for 54h).
        rtofs_dt = 21600.0   # 6-hourly input
        target_dt = 10800.0  # 3-hourly output (matches COMF Fortran)

        cycle_dt = datetime.strptime(self.config.pdy, "%Y%m%d") + \
                   timedelta(hours=self.config.cyc)
        sim_start = cycle_dt - timedelta(hours=self.config.nowcast_hours)
        sim_end = cycle_dt + timedelta(hours=self.config.forecast_hours)
        sim_duration = (sim_end - sim_start).total_seconds()

        for var_list in [all_temp, all_salt]:
            if len(var_list) > 1:
                try:
                    from scipy.interpolate import interp1d
                    n_in = len(var_list)
                    rtofs_times = np.arange(n_in) * rtofs_dt
                    n_out = int((n_in - 1) * rtofs_dt / target_dt) + 1
                    target_times = np.arange(n_out) * target_dt
                    target_times = target_times[
                        target_times <= min(rtofs_times[-1], sim_duration)
                    ]

                    stacked = np.stack(var_list, axis=0)
                    interp_out = np.zeros(
                        (len(target_times),) + stacked.shape[1:], dtype=np.float32,
                    )
                    for node in range(stacked.shape[1]):
                        for lev in range(stacked.shape[2]):
                            f_i = interp1d(
                                rtofs_times, stacked[:, node, lev],
                                kind="linear", fill_value="extrapolate",
                            )
                            interp_out[:, node, lev] = f_i(target_times)

                    var_list.clear()
                    var_list.extend([interp_out[t] for t in range(len(target_times))])
                    log.info(f"Temporally interpolated nudge field: "
                             f"{n_in} -> {len(target_times)} steps "
                             f"(sim_duration={sim_duration:.0f}s)")
                except ImportError:
                    pass

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
