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
import os
import shutil
import subprocess
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
        self._nn_tree = {}      # Cached cKDTree per grid shape
        self._interp = {}       # Cached LinearNDInterpolator per grid shape
        self._ocean_pts = {}    # Cached ocean point coordinates per grid shape
        self._n_ocean_surface = {}  # Cached ocean point count per grid shape
        self._struct_interp = {}  # Cached StructuredGridInterpolator per grid shape

    def _get_time_window(self) -> Tuple[datetime, datetime]:
        """Compute the time window for RTOFS file filtering.

        OBC files cover the full simulation (nowcast + forecast).
        6h buffer on each side for temporal interpolation at boundaries.
        The SSH temporal interpolation clips output to the exact simulation
        window, so extra files just provide interpolation endpoints.
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

    @property
    def is_stofs_mode(self) -> bool:
        """True if using STOFS-style ROI-based processing."""
        return self.config.obc_roi_2d is not None

    def process(self) -> ForcingResult:
        if not HAS_NETCDF4:
            return ForcingResult(
                success=False, source=self.SOURCE_NAME,
                errors=["netCDF4 required for RTOFS processing"],
            )

        log.info(f"RTOFS processor: pdy={self.config.pdy} cyc={self.config.cyc:02d}z "
                 f"mode={'STOFS' if self.is_stofs_mode else 'SECOFS'}")
        self.create_output_dir()

        if self.is_stofs_mode:
            return self._process_stofs()
        return self._process_secofs()

    def _process_secofs(self) -> ForcingResult:
        """SECOFS mode: Python Delaunay interpolation to boundary nodes."""
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

    def _process_stofs(self) -> ForcingResult:
        """STOFS mode: ROI subsetting → data prep → Fortran exe (or Python fallback)."""
        files_2d, files_3d = self.find_input_files_by_type()
        if not files_2d and not files_3d:
            return ForcingResult(
                success=False, source=self.SOURCE_NAME,
                errors=["No RTOFS input files found"],
            )

        import tempfile
        work_dir = Path(tempfile.mkdtemp(prefix="rtofs_stofs_"))

        try:
            output_files = []
            warnings = []

            # Step 1: Subset and merge 2D files (SSH)
            ssh_path = None
            if files_2d:
                ssh_path = self._stofs_prepare_ssh(files_2d, work_dir)
                if ssh_path:
                    log.info(f"Prepared SSH_1.nc: {ssh_path}")

            # Step 2: Subset and merge 3D files (T/S/U/V)
            tsuv_path = None
            if files_3d:
                tsuv_path = self._stofs_prepare_tsuv(files_3d, work_dir)
                if tsuv_path:
                    log.info(f"Prepared TSUV_1.nc: {tsuv_path}")

            if not ssh_path and not tsuv_path:
                return ForcingResult(
                    success=False, source=self.SOURCE_NAME,
                    errors=["Failed to prepare RTOFS data for STOFS OBC"],
                )

            # Step 3: ADT SSH blending (if enabled and data available)
            if ssh_path and self.config.adt_enabled:
                from .adt import ADTBlender
                blender = ADTBlender(self.config, self.input_path)
                blended = blender.blend_ssh(ssh_path, work_dir)
                if blended:
                    ssh_path = blended
                    log.info("ADT SSH blending applied")
                else:
                    warnings.append("ADT data unavailable, using RTOFS-only SSH")

            # Step 4: Try Fortran gen_3Dth_from_hycom
            fortran_ok = self._call_fortran_gen_3dth(work_dir, ssh_path, tsuv_path)

            if fortran_ok:
                # Copy Fortran outputs to final output directory
                for fname in ["elev2D.th.nc", "TEM_3D.th.nc", "SAL_3D.th.nc", "uv3D.th.nc"]:
                    src = work_dir / fname
                    if src.exists():
                        dst = self.output_path / fname
                        shutil.copy2(src, dst)
                        output_files.append(dst)
            else:
                warnings.append("Fortran gen_3Dth not available, using Python interpolation")
                # Fall back to Python Delaunay — load grid and use existing _process_2d/_process_3d
                has_grid = self._load_grid()
                if has_grid:
                    if files_2d:
                        f = self._process_2d(files_2d)
                        if f:
                            output_files.append(f)
                    if files_3d:
                        obc_files = self._process_3d(files_3d)
                        output_files.extend(obc_files)

            return ForcingResult(
                success=len(output_files) > 0,
                source=self.SOURCE_NAME,
                output_files=output_files,
                warnings=warnings,
                metadata={
                    "n_2d_files": len(files_2d),
                    "n_3d_files": len(files_3d),
                    "stofs_mode": True,
                    "fortran_used": fortran_ok,
                    "adt_blended": self.config.adt_enabled and not any("ADT" in w for w in warnings),
                },
            )
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    def find_input_files(self) -> List[Path]:
        files_2d, files_3d = self.find_input_files_by_type()
        return files_2d + files_3d

    # ---- STOFS data preparation methods ----

    def _stofs_subset_roi(self, ds, roi: dict, variables: List[str]) -> dict:
        """Subset RTOFS NetCDF by ROI indices (replaces NCO ncks -d X,x1,x2 -d Y,y1,y2)."""
        x1, x2 = roi["x1"], roi["x2"]
        y1, y2 = roi["y1"], roi["y2"]
        result = {}
        for var in variables:
            if var in ds.variables:
                data = ds.variables[var]
                ndim = len(data.dimensions)
                if ndim == 3:  # (time, Y, X)
                    result[var] = data[:, y1:y2+1, x1:x2+1]
                elif ndim == 4:  # (time, depth, Y, X)
                    result[var] = data[:, :, y1:y2+1, x1:x2+1]
                elif ndim == 2:  # (Y, X)
                    result[var] = data[y1:y2+1, x1:x2+1]
                else:
                    result[var] = data[:]
        return result

    def _stofs_prepare_ssh(self, files_2d: List[Path], work_dir: Path) -> Optional[Path]:
        """Subset 2D files by ROI, merge, and prepare SSH_1.nc for Fortran.

        Replaces the NCO pipeline: ncks subset → ncrcat merge → ncap2/ncatted cleanup.
        """
        roi = self.config.obc_roi_2d
        if not roi:
            return None

        all_ssh = []
        all_lon = None
        all_lat = None
        all_times = []

        for f in files_2d:
            try:
                ds = Dataset(str(f))
                subset = self._stofs_subset_roi(ds, roi, ["ssh", "Longitude", "Latitude"])

                if "ssh" not in subset:
                    ds.close()
                    continue

                ssh = np.ma.filled(subset["ssh"], fill_value=-30000.0)
                # Mask extreme values (matching NCO: where(ssh>10000) ssh=-30000)
                ssh = np.where(np.abs(ssh) > 10000, -30000.0, ssh)

                for t in range(ssh.shape[0]):
                    all_ssh.append(ssh[t])

                if all_lon is None:
                    all_lon = np.array(subset["Longitude"])
                    all_lat = np.array(subset["Latitude"])

                # Read time
                if "MT" in ds.variables:
                    for t in range(ds.variables["MT"].shape[0]):
                        all_times.append(float(ds.variables["MT"][t]))

                ds.close()
            except Exception as e:
                log.warning(f"Failed to read 2D file {f.name}: {e}")

        if not all_ssh:
            return None

        # Write SSH_1.nc (matching Fortran input format)
        output = work_dir / "SSH_1.nc"
        nc = Dataset(str(output), "w", format="NETCDF4")

        ny, nx = all_ssh[0].shape
        nt = len(all_ssh)

        nc.createDimension("time", nt)
        nc.createDimension("ylat", ny)
        nc.createDimension("xlon", nx)

        time_var = nc.createVariable("time", "f8", ("time",))
        if all_times:
            time_var[:] = all_times[:nt]
        else:
            time_var[:] = np.arange(nt) * 21600.0  # 6-hourly default

        lon_var = nc.createVariable("xlon", "f4", ("ylat", "xlon"))
        lon_var[:] = all_lon

        lat_var = nc.createVariable("ylat", "f4", ("ylat", "xlon"))
        lat_var[:] = all_lat

        ssh_var = nc.createVariable("ssh", "f4", ("time", "ylat", "xlon"),
                                    fill_value=-30000.0)
        ssh_var.missing_value = -30000.0
        for t in range(nt):
            ssh_var[t] = all_ssh[t]

        # Also write surf_el (scaled format expected by Fortran)
        surf_el = nc.createVariable("surf_el", "f4", ("time", "ylat", "xlon"),
                                    fill_value=-30000.0)
        surf_el.scale_factor = 0.001
        for t in range(nt):
            data = all_ssh[t].copy()
            data = np.where(np.abs(data) < 10000, data * 1000.0, -3000.0)
            surf_el[t] = data

        nc.close()
        log.info(f"Created SSH_1.nc: {nt} times, {ny}x{nx} grid")
        return output

    def _stofs_prepare_tsuv(
        self, files_3d: List[Path], work_dir: Path,
        roi_override: Optional[dict] = None,
    ) -> Optional[Path]:
        """Subset 3D files by ROI, merge, and prepare TSUV_1.nc for Fortran.

        Replaces: ncrcat merge → ncrename dims/vars → ncap2 NCO script.

        Args:
            roi_override: Use this ROI instead of config.obc_roi_3d.
                Nudging needs a wider ROI than OBC (422 vs 482 for x1).
        """
        roi = roi_override or self.config.obc_roi_3d
        if not roi:
            return None

        all_temp = []
        all_salt = []
        all_u = []
        all_v = []
        all_lon = None
        all_lat = None
        all_depth = None

        for f in files_3d:
            try:
                ds = Dataset(str(f))
                subset = self._stofs_subset_roi(
                    ds, roi,
                    ["temperature", "salinity", "u", "v", "Longitude", "Latitude"],
                )

                if "temperature" not in subset:
                    ds.close()
                    continue

                for t in range(subset["temperature"].shape[0]):
                    all_temp.append(np.ma.filled(subset["temperature"][t], fill_value=-30000.0))
                    all_salt.append(np.ma.filled(subset["salinity"][t], fill_value=-30000.0))
                    all_u.append(np.ma.filled(subset["u"][t], fill_value=0.0))
                    all_v.append(np.ma.filled(subset["v"][t], fill_value=0.0))

                if all_lon is None:
                    all_lon = np.array(subset["Longitude"])
                    all_lat = np.array(subset["Latitude"])
                if all_depth is None and "Depth" in ds.variables:
                    all_depth = np.array(ds.variables["Depth"][:])

                ds.close()
            except Exception as e:
                log.warning(f"Failed to read 3D file {f.name}: {e}")

        if not all_temp:
            return None

        # Write TSUV_1.nc (matching Fortran input format)
        output = work_dir / "TSUV_1.nc"
        nc = Dataset(str(output), "w", format="NETCDF4")

        nz, ny, nx = all_temp[0].shape
        nt = len(all_temp)

        nc.createDimension("time", nt)
        nc.createDimension("lev", nz)
        nc.createDimension("ylat", ny)
        nc.createDimension("xlon", nx)

        time_var = nc.createVariable("time", "f8", ("time",))
        time_var[:] = np.arange(nt) * 21600.0

        if all_depth is not None:
            lev_var = nc.createVariable("lev", "f4", ("lev",))
            lev_var[:] = all_depth

        lon_var = nc.createVariable("xlon", "f4", ("ylat", "xlon"))
        lon_var[:] = all_lon
        lat_var = nc.createVariable("ylat", "f4", ("ylat", "xlon"))
        lat_var[:] = all_lat

        for name, data_list in [("temperature", all_temp), ("salinity", all_salt),
                                ("water_u", all_u), ("water_v", all_v)]:
            var = nc.createVariable(name, "f4", ("time", "lev", "ylat", "xlon"))
            for t in range(nt):
                var[t] = data_list[t]

        nc.close()
        log.info(f"Created TSUV_1.nc: {nt} times, {nz} levels, {ny}x{nx} grid")
        return output

    def _call_fortran_gen_3dth(
        self, work_dir: Path,
        ssh_path: Optional[Path],
        tsuv_path: Optional[Path],
    ) -> bool:
        """Call Fortran stofs_3d_atl_gen_3Dth_from_hycom for 3D OBC interpolation.

        Pattern follows TidalProcessor._call_fortran_tide_fac().
        """
        exe_names = ["stofs_3d_atl_gen_3Dth_from_hycom"]
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
            log.debug("No Fortran gen_3Dth_from_hycom executable found")
            return False

        # Symlink required input files into work_dir
        fix_dir = os.environ.get("FIXstofs3d", "")
        required_links = {
            "SSH_1.nc": ssh_path,
            "TS_1.nc": tsuv_path,
            "UV_1.nc": tsuv_path,
        }

        # Grid files from FIX
        fix_files = {
            "hgrid.ll": "stofs_3d_atl_hgrid.ll",
            "hgrid.gr3": "stofs_3d_atl_hgrid.gr3",
            "vgrid.in": "stofs_3d_atl_vgrid.in",
            "estuary.gr3": "stofs_3d_atl_estuary.gr3",
            "TEM_nudge.gr3": "stofs_3d_atl_tem_nudge.gr3",
            "gen_3Dth_from_nc.in": "stofs_3d_atl_obc_3dth_nc.in",
        }

        for link_name, source in required_links.items():
            if source and source.exists():
                target = work_dir / link_name
                if not target.exists():
                    target.symlink_to(source)

        if fix_dir:
            for link_name, fix_name in fix_files.items():
                src = Path(fix_dir) / fix_name
                if src.exists():
                    target = work_dir / link_name
                    if not target.exists():
                        target.symlink_to(src)

        # Also try grid_file from config
        if self.grid_file and Path(self.grid_file).exists():
            for name in ["hgrid.ll"]:
                target = work_dir / name
                if not target.exists():
                    target.symlink_to(self.grid_file)

        try:
            result = subprocess.run(
                [str(exe)],
                cwd=str(work_dir),
                capture_output=True,
                text=True,
                timeout=600,
            )

            if result.returncode != 0:
                log.warning(f"gen_3Dth returned {result.returncode}: {result.stderr[:300]}")
                return False

            # NOTE: Do NOT apply SSH offset here — the Fortran executable
            # (nos_ofs_create_forcing_obc_schism / gen_3Dth_from_hycom)
            # already applies the offset internally (WL += 1.25 at line ~3133).
            # Applying it again would double the offset to +2.5m.

            # Verify outputs exist
            expected = ["elev2D.th.nc", "TEM_3D.th.nc", "SAL_3D.th.nc", "uv3D.th.nc"]
            found = [f for f in expected if (work_dir / f).exists()]
            log.info(f"Fortran gen_3Dth produced {len(found)}/{len(expected)} files")
            return len(found) >= 3  # Allow uv3D to be optional

        except subprocess.TimeoutExpired:
            log.warning("Fortran gen_3Dth timed out (600s)")
            return False
        except Exception as e:
            log.warning(f"Error calling Fortran gen_3Dth: {e}")
            return False

    @staticmethod
    def _apply_ssh_offset(elev_path: Path, offset: float) -> None:
        """Apply SSH offset to elev2D.th.nc (post-Fortran correction)."""
        try:
            ds = Dataset(str(elev_path), "r+")
            if "time_series" in ds.variables:
                ds.variables["time_series"][:] += offset
            ds.close()
            log.info(f"Applied SSH offset +{offset}m to {elev_path.name}")
        except Exception as e:
            log.warning(f"Failed to apply SSH offset: {e}")

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
        prefer the forecast file to match Fortran OBC behavior.
        The Fortran shell script only collects f* files (no n* files),
        so using forecast here ensures identical input data.
        """
        # Parse valid times
        file_info = []
        for f in files:
            hour, is_nowcast = self._parse_rtofs_hour(f)
            valid_time = cycle_date + timedelta(hours=hour)
            file_info.append((valid_time, is_nowcast, f))

        # Sort by valid time; for ties, forecast first (is_nowcast=False < True)
        file_info.sort(key=lambda x: (x[0], x[1]))

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

        # Search newest RTOFS cycle first to match Fortran shell behavior.
        # The Fortran prep (nos_ofs_create_forcing_obc.sh) uses the latest
        # available cycle. PDY itself rarely has RTOFS ready at 00Z, so
        # PDY-1 is the typical production hit.
        for date in [base_date - timedelta(days=1), base_date - timedelta(days=2), base_date]:
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
        self._rtofs_cycle_date = rtofs_cycle_date
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

        Primary: Structured grid bilinear interpolation on the native RTOFS
        curvilinear grid. Eliminates Delaunay triangulation differences.

        Fallback: Delaunay + corner points (matching Fortran INTERP_REMESH)
        if structured grid interpolation fails or is unavailable.

        For 3D fields, the surface-level triangulation is cached and reused
        for deeper levels where the ocean mask shrinks. Land values at cached
        ocean points are filled from the nearest still-valid point.
        """
        n_bnd = len(self._bnd_lons)

        # Delaunay interpolation with corner points (matching Fortran INTERP_REMESH)
        # Use -180/180 convention (matching Fortran: if lon>180, lon=lon-360)
        target_pts = np.column_stack([self._bnd_lons, self._bnd_lats])

        # Flatten RTOFS grid
        if rtofs_lon.ndim == 2:
            lon_flat = rtofs_lon.ravel()
            lat_flat = rtofs_lat.ravel()
        else:
            lon_2d, lat_2d = np.meshgrid(rtofs_lon, rtofs_lat)
            lon_flat = lon_2d.ravel()
            lat_flat = lat_2d.ravel()

        data_flat = rtofs_data.ravel()

        # Convert RTOFS lons to -180/180 (matching Fortran convention)
        lon_flat = np.where(lon_flat > 180, lon_flat - 360, lon_flat)

        # Subset to domain bounding box (matching Fortran minlon/maxlon from CTL)
        buf = 1.0  # 1-degree buffer (Fortran uses CTL bounds + 1 grid cell)
        lon_min = self._bnd_lons.min() - buf
        lon_max = self._bnd_lons.max() + buf
        lat_min = self._bnd_lats.min() - buf
        lat_max = self._bnd_lats.max() + buf
        domain_mask = ((lon_flat >= lon_min) & (lon_flat <= lon_max) &
                       (lat_flat >= lat_min) & (lat_flat <= lat_max))

        n_in_domain = int(np.sum(domain_mask))
        if n_in_domain == 0:
            # Subsetting removed everything — skip subsetting for this grid
            log.warning(f"Domain subset empty ({lon_min:.1f}-{lon_max:.1f}, "
                        f"{lat_min:.1f}-{lat_max:.1f}), using full grid")
        else:
            lon_flat = lon_flat[domain_mask]
            lat_flat = lat_flat[domain_mask]
            data_flat = data_flat[domain_mask]

        # Mask land/fill values (Fortran uses abs >= 99)
        ocean_mask = (np.abs(data_flat) < 99.0) & np.isfinite(data_flat)
        n_ocean = int(np.sum(ocean_mask))

        if n_ocean == 0:
            log.warning("No valid ocean data in field")
            return np.full(n_bnd, np.nan, dtype=np.float32)

        ocean_pts = np.column_stack([lon_flat[ocean_mask], lat_flat[ocean_mask]])
        ocean_val = data_flat[ocean_mask].astype(np.float32)

        # Cache key: grid shape — reuse surface triangulation for deeper levels
        cache_key = rtofs_lon.shape

        try:
            from scipy.interpolate import LinearNDInterpolator
            from scipy.spatial import cKDTree

            if cache_key not in self._nn_tree:
                # Fortran REMESH mode=0: add 4 corner points at 1% outside
                # target bounding box, valued at field mean. This extends the
                # convex hull to cover all boundary nodes and matches Fortran behavior.
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
                corner_vals = np.full(4, avg_val, dtype=np.float32)

                aug_pts = np.vstack([corner_pts, ocean_pts])
                aug_vals = np.concatenate([corner_vals, ocean_val])

                self._interp[cache_key] = LinearNDInterpolator(aug_pts, aug_vals)
                self._nn_tree[cache_key] = cKDTree(ocean_pts)
                self._ocean_pts[cache_key] = ocean_pts
                self._n_ocean_surface[cache_key] = n_ocean
                self._n_corner = 4  # number of prepended corner points
                log.info(f"Built Delaunay interpolator: {n_ocean}+4 corner pts "
                         f"(grid {rtofs_lon.shape}, avg={avg_val:.4f})")
                result = self._interp[cache_key](target_pts).astype(np.float32)
            else:
                # Reuse surface triangulation for deeper levels
                # Corner values get the new field mean
                avg_val = float(np.mean(ocean_val))
                n_corners = getattr(self, '_n_corner', 4)
                if n_ocean == self._n_ocean_surface[cache_key]:
                    new_vals = np.concatenate([
                        np.full(n_corners, avg_val, dtype=np.float32),
                        ocean_val,
                    ])
                    self._interp[cache_key].values[:, 0] = new_vals
                else:
                    current_tree = cKDTree(ocean_pts)
                    _, nn_idx = current_tree.query(self._ocean_pts[cache_key])
                    mapped_vals = ocean_val[nn_idx]
                    new_vals = np.concatenate([
                        np.full(n_corners, avg_val, dtype=np.float32),
                        mapped_vals,
                    ])
                    self._interp[cache_key].values[:, 0] = new_vals
                result = self._interp[cache_key](target_pts).astype(np.float32)

            # Fortran REMESH fallback: points with negative weights or |w|>3
            # are filled from nearest valid BOUNDARY node (not nearest data point).
            # With corner points, fewer nodes should be NaN, but handle remaining.
            nan_mask = np.isnan(result)
            if np.any(nan_mask):
                valid_bnd = np.where(~nan_mask)[0]
                if len(valid_bnd) > 0:
                    nan_bnd = np.where(nan_mask)[0]
                    # Use cos(lat)-corrected distance (approximates haversine)
                    mean_lat = np.radians(np.mean(self._bnd_lats))
                    corrected_pts = target_pts.copy()
                    corrected_pts[:, 0] *= np.cos(mean_lat)
                    bnd_tree = cKDTree(corrected_pts[valid_bnd])
                    _, nn_idx = bnd_tree.query(corrected_pts[nan_bnd])
                    result[nan_bnd] = result[valid_bnd[nn_idx]]
                else:
                    _, nn_idx = self._nn_tree[cache_key].query(target_pts[nan_mask])
                    result[nan_mask] = ocean_val[nn_idx]

        except ImportError:
            result = np.zeros(n_bnd, dtype=np.float32)
            for k in range(n_bnd):
                dist = (ocean_pts[:, 0] - self._bnd_lons[k])**2 + \
                       (ocean_pts[:, 1] - self._bnd_lats[k])**2
                result[k] = ocean_val[np.argmin(dist)]

        return result

    def _find_ssh_weights(self) -> Optional[dict]:
        """Find and load precomputed SSH REMESH weights from FIX."""
        if hasattr(self, '_ssh_weights_cache'):
            return self._ssh_weights_cache

        import os
        for env_var in ["FIXofs", "FIXstofs3d"]:
            fix_dir = os.environ.get(env_var)
            if not fix_dir:
                continue
            for name in ["secofs.obc_ssh_weights.npz", "obc_ssh_weights.npz"]:
                path = Path(fix_dir) / name
                if path.exists():
                    self._ssh_weights_cache = dict(np.load(str(path), allow_pickle=True))
                    log.info(f"Loaded precomputed SSH weights from {path}")
                    return self._ssh_weights_cache

        # Also check input_path (for testing)
        for name in ["secofs.obc_ssh_weights.npz", "obc_ssh_weights.npz"]:
            path = self.input_path / name
            if path.exists():
                self._ssh_weights_cache = dict(np.load(str(path), allow_pickle=True))
                log.info(f"Loaded precomputed SSH weights from {path}")
                return self._ssh_weights_cache

        self._ssh_weights_cache = None
        return None

    def _find_3d_weights(self) -> Optional[dict]:
        """Find and load precomputed 3D REMESH weights for regional RTOFS grid.

        The 3D RTOFS files (TEM_3D, SAL_3D) use a regional subset grid
        (e.g., US_east: 1710x742) which differs from the global 2D grid
        (3298x4500) used for SSH. This method searches for a separate
        NPZ with weights matching the regional grid.
        """
        if hasattr(self, '_3d_weights_cache'):
            return self._3d_weights_cache

        import os
        search_names = [
            "secofs.obc_3d_weights.npz",
            "obc_3d_weights.npz",
        ]
        for env_var in ["FIXofs", "FIXstofs3d"]:
            fix_dir = os.environ.get(env_var)
            if not fix_dir:
                continue
            for name in search_names:
                path = Path(fix_dir) / name
                if path.exists():
                    self._3d_weights_cache = dict(np.load(str(path), allow_pickle=True))
                    log.info(f"Loaded precomputed 3D weights from {path}")
                    return self._3d_weights_cache

        # Also check input_path and parent directories (for testing)
        for search_dir in [self.input_path, self.input_path.parent]:
            for name in search_names:
                path = search_dir / name
                if path.exists():
                    self._3d_weights_cache = dict(np.load(str(path), allow_pickle=True))
                    log.info(f"Loaded precomputed 3D weights from {path}")
                    return self._3d_weights_cache

        self._3d_weights_cache = None
        return None

    def _process_2d(self, files_2d: List[Path]) -> Optional[Path]:
        """Extract SSH from RTOFS 2D files, interpolate to boundary nodes."""
        output_file = self.output_path / "elev2D.th.nc"
        n_bnd = len(self._bnd_lons)
        model_dt = 120.0  # SCHISM model timestep (seconds)

        # Check for precomputed REMESH weights
        ssh_weights = self._find_ssh_weights()
        if ssh_weights:
            from ..interp.precomputed_weights import apply_precomputed_ssh
            log.info("Using precomputed REMESH weights for SSH (Fortran-equivalent)")

        try:
            all_ssh = []

            for f in files_2d:
                ds = Dataset(str(f))
                ssh_raw = ds.variables["ssh"][:]
                lon = ds.variables["Longitude"][:]
                lat = ds.variables["Latitude"][:]

                ssh_raw = np.ma.filled(ssh_raw, fill_value=np.nan)

                for t in range(ssh_raw.shape[0]):
                    if ssh_weights:
                        ssh_bnd = apply_precomputed_ssh(ssh_weights, ssh_raw[t])
                    else:
                        ssh_bnd = self._interpolate_2d_to_boundary(lon, lat, ssh_raw[t])
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
            # Clip to the actual simulation window (nowcast + forecast)
            n_rtofs = ssh_array.shape[0]

            # Build actual time axis from file valid times (NOT uniform 6h).
            # RTOFS 2D diag files are hourly (f001-f048) then 3-hourly
            # (f051+), so assuming uniform spacing is wrong.
            # Time axis is relative to sim_start (t=0 = model start),
            # matching the Fortran convention where day_start=0.
            file_hours = []
            for f in files_2d:
                hour, _ = self._parse_rtofs_hour(f)
                file_hours.append(hour)

            cycle_dt = datetime.strptime(self.config.pdy, "%Y%m%d") + \
                       timedelta(hours=self.config.cyc)
            sim_start = self.time_hotstart if self.time_hotstart else \
                        cycle_dt - timedelta(hours=self.config.nowcast_hours)
            sim_end = cycle_dt + timedelta(hours=self.config.forecast_hours)
            sim_duration = (sim_end - sim_start).total_seconds()

            # Compute time of each file relative to sim_start (model t=0).
            # Files before sim_start have negative times (buffer).
            rtofs_cycle = getattr(self, '_rtofs_cycle_date', None)
            if rtofs_cycle is None:
                rtofs_cycle = datetime.strptime(self.config.pdy, "%Y%m%d")
            rtofs_times = np.array(
                [(rtofs_cycle + timedelta(hours=h) - sim_start).total_seconds()
                 for h in file_hours],
                dtype=np.float64,
            )

            # Model output covers [0, sim_duration] at model_dt intervals
            n_model_steps = int(sim_duration / model_dt) + 1

            if n_model_steps > n_rtofs and n_rtofs > 1:
                try:
                    from scipy.interpolate import interp1d
                    model_times = np.arange(n_model_steps) * model_dt
                    model_times = model_times[model_times <= rtofs_times[-1]]

                    ssh_interp = np.zeros((len(model_times), n_bnd), dtype=np.float32)
                    for node in range(n_bnd):
                        f_interp = interp1d(rtofs_times, ssh_array[:, node],
                                           kind="linear", fill_value="extrapolate")
                        ssh_interp[:, node] = f_interp(model_times)

                    ssh_array = ssh_interp
                    dt_out = model_dt
                    avg_dt_h = (rtofs_times[-1] - rtofs_times[0]) / 3600.0 / max(n_rtofs - 1, 1)
                    log.info(f"Temporally interpolated SSH: {n_rtofs} steps "
                             f"(avg {avg_dt_h:.1f}h, t0={rtofs_times[0]/3600:.1f}h) → "
                             f"{len(model_times)} steps at dt={model_dt}s")
                except ImportError:
                    dt_out = (rtofs_times[1] - rtofs_times[0]) if n_rtofs > 1 else 3600.0
            else:
                dt_out = (rtofs_times[1] - rtofs_times[0]) if n_rtofs > 1 else 3600.0

            # Write SCHISM format
            nc = Dataset(str(output_file), "w", format="NETCDF4")
            nt = ssh_array.shape[0]

            nc.createDimension("time", nt)
            nc.createDimension("nOpenBndNodes", n_bnd)
            nc.createDimension("nLevels", 1)
            nc.createDimension("nComponents", 1)
            nc.createDimension("one", 1)

            time_var = nc.createVariable("time", "f8", ("time",))
            if dt_out == model_dt and nt > n_rtofs:
                # Temporally interpolated: uniform 120s steps
                time_var[:] = [i * dt_out for i in range(nt)]
            else:
                # No interpolation: use actual file valid times
                time_var[:] = rtofs_times[:nt]

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

        # Weight discovery for 3D T/S interpolation.
        # The RTOFS 3D files use a regional grid (e.g., US_east: 1710x742)
        # which differs from the global 2D grid (3298x4500) used for SSH.
        # Fallback chain:
        #   1. Dedicated 3D weights (obc_3d_weights.npz) for regional grid
        #   2. SSH weights (obc_ssh_weights.npz) if grid shape happens to match
        #   3. Delaunay triangulation (slowest, no precomputed weights)
        weights_3d = self._find_3d_weights()
        ssh_weights = self._find_ssh_weights()
        resolved_weights = None  # set after first file validates grid shape
        from ..interp.precomputed_weights import apply_precomputed_ssh

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
                    # Match Fortran: mask values >= 99 as land/fill
                    data[np.abs(data) >= 99.0] = np.nan

                    # Handle time dimension: iterate over all time steps
                    if data.ndim == 4:
                        n_times = data.shape[0]
                    else:
                        n_times = 1
                        data = data[np.newaxis, ...]

                    for t in range(n_times):
                        data_t = data[t]

                        # Resolve which weights to use (first time only)
                        if resolved_weights is None:
                            actual = data_t.shape[1:]  # (levels, Y, X) -> (Y, X)
                            if len(actual) >= 2:
                                grid_2d = (actual[-2], actual[-1])
                            else:
                                grid_2d = tuple(data_t.shape)

                            # Try dedicated 3D weights first
                            if weights_3d is not None:
                                expected_3d = tuple(weights_3d["grid_shape"])
                                if grid_2d == expected_3d:
                                    resolved_weights = weights_3d
                                    log.info(
                                        f"Using precomputed 3D weights for T/S "
                                        f"(regional grid {grid_2d})"
                                    )
                                else:
                                    log.info(
                                        f"3D weights grid {expected_3d} != data "
                                        f"grid {grid_2d}, trying SSH weights"
                                    )

                            # Fall back to SSH weights if 3D didn't match
                            if resolved_weights is None and ssh_weights is not None:
                                expected_ssh = tuple(ssh_weights["grid_shape"])
                                if grid_2d == expected_ssh:
                                    resolved_weights = ssh_weights
                                    log.info(
                                        "Using precomputed SSH weights for "
                                        "3D T/S (same grid as SSH)"
                                    )
                                else:
                                    log.info(
                                        f"SSH weights grid {expected_ssh} != "
                                        f"data grid {grid_2d}"
                                    )

                            # Final fallback: Delaunay (resolved_weights stays None)
                            if resolved_weights is None:
                                resolved_weights = False  # sentinel: use Delaunay
                                log.info(
                                    f"No precomputed weights match 3D grid "
                                    f"{grid_2d} — using Delaunay for T/S"
                                )

                        # Interpolate each RTOFS depth level to boundary nodes
                        bnd_profile_rtofs = np.full((n_bnd, n_rtofs_levels), np.nan, dtype=np.float32)
                        for lev in range(min(n_rtofs_levels, data_t.shape[0])):
                            if resolved_weights and resolved_weights is not False:
                                bnd_profile_rtofs[:, lev] = apply_precomputed_ssh(
                                    resolved_weights, data_t[lev])
                            else:
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

            # Compute simulation duration for clipping
            cycle_dt = datetime.strptime(self.config.pdy, "%Y%m%d") + \
                       timedelta(hours=self.config.cyc)
            sim_start = self.time_hotstart if self.time_hotstart else \
                        cycle_dt - timedelta(hours=self.config.nowcast_hours)
            sim_end = cycle_dt + timedelta(hours=self.config.forecast_hours)
            sim_duration_3d = (sim_end - sim_start).total_seconds()

            # Temporally interpolate 3D fields from 6h to 3h, clipped to simulation
            for var_list in [all_temp, all_salt]:
                if len(var_list) > 1:
                    try:
                        from scipy.interpolate import interp1d
                        n_in = len(var_list)
                        rtofs_times = np.arange(n_in) * rtofs_dt_3d
                        n_out = int((n_in - 1) * rtofs_dt_3d / target_dt_3d) + 1
                        target_times = np.arange(n_out) * target_dt_3d
                        target_times = target_times[target_times <= min(rtofs_times[-1], sim_duration_3d)]

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

        RTOFS depths are positive downward (0, 5, 10, ... 5000m).
        SCHISM sigma depths are negative (bottom=-depth to surface=0).
        The output array shape is (n_bnd, target_levels) with level ordering
        bottom-to-surface matching SCHISM convention.
        """
        from scipy.interpolate import interp1d

        n_bnd = bnd_profile.shape[0]
        # RTOFS depths: positive down → convert to negative (SCHISM convention)
        rtofs_z = -np.abs(rtofs_depths)  # e.g., [0, -5, -10, ... -5000]

        defaults = {"temperature": 15.0, "salinity": 35.0, "u": 0.0, "v": 0.0}
        fill_val = defaults.get(var_name, 0.0)

        result = np.full((n_bnd, target_levels), fill_val, dtype=np.float32)

        for node in range(n_bnd):
            profile = bnd_profile[node, :]
            valid = ~np.isnan(profile)

            if np.sum(valid) < 2:
                # Only 1 or 0 valid levels — fill with that value or default
                if np.sum(valid) == 1:
                    result[node, :] = profile[valid][0]
                continue

            node_depth = abs(self._bnd_depths[node])
            if node_depth < 0.1:
                continue  # Skip dry nodes

            # Get SCHISM target depths for this node (negative, bottom to surface)
            if self._vgrid and self._vgrid.node_sigma is not None:
                # Per-node sigma from LSC2 (best match to Fortran)
                schism_z = self._vgrid.get_node_depths(node, node_depth)
            elif self._vgrid:
                schism_z = self._vgrid.get_depths(node_depth)
            else:
                schism_z = np.linspace(-node_depth, 0, target_levels)

            # Ensure schism_z is sorted bottom to surface (most negative first)
            schism_z = np.sort(schism_z)

            # Pad or trim to target_levels
            if len(schism_z) > target_levels:
                schism_z = schism_z[-target_levels:]  # Keep surface levels
            elif len(schism_z) < target_levels:
                # Pad deep levels with the deepest available depth
                n_pad = target_levels - len(schism_z)
                pad_z = np.full(n_pad, schism_z[0])
                schism_z = np.concatenate([pad_z, schism_z])

            # RTOFS profile: valid depths sorted from surface to deep
            # For interpolation, ensure both arrays go in the same direction
            valid_z = rtofs_z[valid]
            valid_prof = profile[valid]
            # Sort by depth (most negative = deepest first)
            sort_idx = np.argsort(valid_z)
            valid_z = valid_z[sort_idx]
            valid_prof = valid_prof[sort_idx]

            # Interpolate from RTOFS depths to SCHISM depths
            # fill_value: use deepest value for below-bottom, surface value for above-surface
            f_interp = interp1d(
                valid_z, valid_prof,
                kind="linear", bounds_error=False,
                fill_value=(valid_prof[0], valid_prof[-1]),  # (deep, surface)
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
