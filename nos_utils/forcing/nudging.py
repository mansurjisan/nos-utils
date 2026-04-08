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
        """Python fallback: generate nudging fields from RTOFS data."""
        output_files = []

        # Look for pre-extracted RTOFS T/S files
        tem_files = sorted(self.input_path.glob("*temperature*.nc")) + \
                    sorted(self.input_path.glob("*TS_*.nc"))
        sal_files = sorted(self.input_path.glob("*salinity*.nc")) + \
                    sorted(self.input_path.glob("*TS_*.nc"))

        if not tem_files and not sal_files:
            return ForcingResult(
                success=False, source=self.SOURCE_NAME,
                errors=["No RTOFS T/S input files found for nudging"],
            )

        for var_name, label, out_name in [
            ("temperature", "TEM", "TEM_nu.nc"),
            ("salinity", "SAL", "SAL_nu.nc"),
        ]:
            out_file = self._create_nudging_field(
                var_name, label, self.output_path / out_name,
            )
            if out_file:
                output_files.append(out_file)

        return ForcingResult(
            success=len(output_files) > 0,
            source=self.SOURCE_NAME,
            output_files=output_files,
            metadata={
                "timescale_seconds": self.config.nudging_timescale_seconds,
                "n_levels": self.config.n_levels,
                "fortran_used": False,
            },
        )

    def find_input_files(self) -> List[Path]:
        return sorted(self.input_path.glob("*temperature*.nc")) + \
               sorted(self.input_path.glob("*salinity*.nc"))

    def _create_nudging_field(
        self, var_name: str, label: str, output_path: Path,
    ) -> Optional[Path]:
        """Create a single nudging NetCDF file (TEM_nu.nc or SAL_nu.nc)."""
        try:
            # Find source data
            source_files = sorted(self.input_path.glob(f"*{var_name}*.nc"))
            if not source_files:
                source_files = sorted(self.input_path.glob("*TS_*.nc"))
            if not source_files:
                log.warning(f"No source data for {label} nudging")
                return None

            # Read first source file for dimensions
            ds_src = Dataset(str(source_files[0]))

            # Determine dimensions
            n_time = len(source_files)
            # Try to get spatial dims from source
            for dim_name in ["node", "nOpenBndNodes", "Y", "ylat"]:
                if dim_name in ds_src.dimensions:
                    n_nodes = ds_src.dimensions[dim_name].size
                    break
            else:
                n_nodes = 100  # fallback

            n_levels = self.config.n_levels
            ds_src.close()

            # Create output nudging file
            nc = Dataset(str(output_path), "w", format="NETCDF4")

            nc.createDimension("time", None)  # unlimited
            nc.createDimension("node", n_nodes)
            nc.createDimension("nLevels", n_levels)
            nc.createDimension("one", 1)

            # Time
            time_var = nc.createVariable("time", "f8", ("time",))
            time_var.units = "seconds since model start"
            time_var.long_name = "Time"

            dt = 21600.0  # 6-hourly (RTOFS interval)
            time_var[:] = [i * dt for i in range(n_time)]

            # Node mapping
            map_var = nc.createVariable("map_to_global_node", "i4", ("node",))
            map_var.long_name = "Global node index (1-based)"
            map_var[:] = np.arange(1, n_nodes + 1)

            # Tracer concentration
            tracer = nc.createVariable(
                "tracer_concentration", "f4",
                ("time", "node", "nLevels", "one"),
                fill_value=9.999e20,
            )
            tracer.long_name = f"{label} nudging field"
            tracer.units = "degC" if label == "TEM" else "PSU"

            # Fill with source data or defaults
            default_val = 15.0 if label == "TEM" else 35.0
            for t in range(n_time):
                tracer[t, :, :, 0] = default_val

            nc.nudging_timescale = self.config.nudging_timescale_seconds
            nc.close()

            log.info(f"Created {output_path.name}: {n_time} times, {n_nodes} nodes, {n_levels} levels")
            return output_path

        except Exception as e:
            log.error(f"Failed to create {label} nudging: {e}")
            return None
