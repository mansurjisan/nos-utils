"""
Prep orchestrator — chains all forcing processors in sequence.

Replaces exnos_ofs_prep.sh (~800 lines of shell) with a Python pipeline.

Usage::

    from nos_utils.config import ForcingConfig
    from nos_utils.orchestrator import PrepOrchestrator

    config = ForcingConfig.for_secofs(pdy="20260324", cyc=12)
    orch = PrepOrchestrator(config, paths={
        "gfs": "/data/gfs/v16.3",
        "hrrr": "/data/hrrr/v4.1",
        "nwm": "/data/nwm/v3.0",
        "rtofs": "/data/rtofs/v2.5",
        "fix": "/data/fix/secofs",
        "output": "/data/work/secofs",
        "comout": "/data/comout/secofs",
    })
    results = orch.run(phase="nowcast")
"""

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .config import ForcingConfig
from .forcing.base import ForcingResult

log = logging.getLogger(__name__)


@dataclass
class PrepResult:
    """Result from a full prep orchestration run."""

    success: bool
    phase: str
    results: List[ForcingResult] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    @property
    def all_output_files(self) -> List[Path]:
        files = []
        for r in self.results:
            files.extend(r.output_files)
        return files

    @property
    def all_errors(self) -> List[str]:
        errors = []
        for r in self.results:
            errors.extend(r.errors)
        return errors

    @property
    def all_warnings(self) -> List[str]:
        warnings = []
        for r in self.results:
            warnings.extend(r.warnings)
        return warnings

    def summary(self) -> str:
        lines = [
            f"PrepResult: phase={self.phase}, success={self.success}, "
            f"elapsed={self.elapsed_seconds:.1f}s",
            f"  Files: {len(self.all_output_files)}",
        ]
        for r in self.results:
            status = "OK" if r.success else "FAIL"
            n_files = len(r.output_files)
            lines.append(f"  [{status}] {r.source}: {n_files} files")
            if r.errors:
                for e in r.errors:
                    lines.append(f"       ERROR: {e}")
            if r.warnings:
                for w in r.warnings:
                    lines.append(f"       WARN: {w}")
        return "\n".join(lines)


class PrepOrchestrator:
    """
    Chains all forcing processors for a complete SCHISM prep cycle.

    Steps:
    1. Hotstart — find restart file, determine ihot and time_hotstart
    2. GFS — primary atmospheric forcing (sflux_air/rad/prc stack 1)
    3. HRRR — secondary atmospheric (sflux stack 2, optional)
    4. NWM — river forcing (vsource.th, msource.th, source_sink.in)
    5. RTOFS — ocean boundary (elev2D.th.nc, TEM_3D.th.nc, SAL_3D.th.nc, uv3D.th.nc)
    6. Tidal — tidal constituents (bctides.in)
    7. param.nml — model configuration with runtime parameters
    8. (UFS only) DATM blending + ESMF mesh generation
    """

    # Steps that can be handled by Python vs legacy shell
    PYTHON_STEPS = {"hotstart", "gfs", "hrrr", "tidal", "param_nml", "datm"}
    LEGACY_STEPS = {"nwm", "rtofs", "nudging"}  # Need Fortran/shell for production

    def __init__(
        self,
        config: ForcingConfig,
        paths: Dict[str, str],
        run_name: str = "secofs",
        skip_legacy: bool = True,
    ):
        """
        Args:
            config: ForcingConfig with domain, cycle, and run settings
            paths: Dict of named paths:
                gfs: COMINgfs root
                hrrr: COMINhrrr root (optional)
                nwm: COMINnwm root (optional)
                rtofs: COMINrtofs root (optional)
                fix: FIXofs directory (templates, grid files)
                output: Working directory (DATA)
                comout: Output archive directory (optional)
                restart: Directory to search for hotstart files (optional)
            run_name: OFS name (e.g., "secofs", "stofs_3d_atl")
            skip_legacy: If True (default), skip OBC/river/nudging steps
                (handled by legacy shell scripts). Set False to run all steps.
        """
        self.config = config
        self.paths = {k: Path(v) for k, v in paths.items()}
        self.run_name = run_name
        self.skip_legacy = skip_legacy

    @property
    def is_stofs(self) -> bool:
        """True if running STOFS-3D-ATL (detected from run_name or config)."""
        return "stofs" in self.run_name.lower() or self.config.obc_roi_2d is not None

    def run(self, phase: str = "nowcast") -> PrepResult:
        """
        Run the full prep pipeline.

        Args:
            phase: "nowcast", "forecast", or "full" (nowcast + forecast)

        Returns:
            PrepResult with all processor results
        """
        t0 = time.time()
        results = []
        log.info(f"PrepOrchestrator: {self.run_name} {phase} "
                 f"pdy={self.config.pdy} cyc={self.config.cyc:02d}z "
                 f"mode={'STOFS' if self.is_stofs else 'SECOFS'}")

        output_dir = self.paths["output"]
        output_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Hotstart
        hotstart_result = self._run_hotstart(output_dir)
        results.append(hotstart_result)

        # Extract time_hotstart for param.nml
        time_hotstart = None
        hs_info = hotstart_result.metadata.get("hotstart_info")
        if hs_info and hasattr(hs_info, "filepath"):
            from .forcing.hotstart import HotstartProcessor
            proc = HotstartProcessor(self.config, Path(self.paths.get("restart", "")),
                                     output_dir, run_name=self.run_name)
            time_hotstart = proc._parse_file_datetime(hs_info.filepath)

        # Fallback: read time_hotstart from environment (set by shell prep)
        if time_hotstart is None:
            env_ths = os.environ.get("time_hotstart", "")
            if env_ths and len(env_ths) >= 10:
                try:
                    time_hotstart = datetime.strptime(env_ths[:10], "%Y%m%d%H")
                    log.info(f"Using time_hotstart from environment: {time_hotstart}")
                except ValueError:
                    pass

        # ---- Phase 1: Lightweight steps in parallel ----
        # GFS, HRRR, NWM, Tidal are fast (subprocess/IO-bound, ~90s max).
        # RTOFS is heavy (large NetCDF reads + Delaunay) and netCDF4's C
        # library can stall under concurrent thread access, so run it
        # sequentially in Phase 2.
        from concurrent.futures import ThreadPoolExecutor, as_completed

        parallel_tasks = {}

        if "gfs" in self.paths:
            parallel_tasks["GFS"] = lambda: self._run_gfs(output_dir, phase, time_hotstart)

        if "hrrr" in self.paths and self.config.met_num >= 2:
            parallel_tasks["HRRR"] = lambda: self._run_hrrr(output_dir, phase, time_hotstart)

        if "nwm" in self.paths and (self.is_stofs or not self.skip_legacy):
            parallel_tasks["NWM"] = lambda: self._run_nwm(output_dir, phase, time_hotstart)
        elif self.skip_legacy:
            log.info("Skipping NWM river (handled by legacy shell)")

        parallel_tasks["TIDAL"] = lambda: self._run_tidal(output_dir, phase, time_hotstart)

        if parallel_tasks:
            log.info(f"Running {len(parallel_tasks)} steps in parallel: "
                     f"{', '.join(parallel_tasks.keys())}")
            with ThreadPoolExecutor(max_workers=len(parallel_tasks)) as pool:
                futures = {pool.submit(fn): name for name, fn in parallel_tasks.items()}
                for future in as_completed(futures):
                    name = futures[future]
                    try:
                        r = future.result()
                        results.append(r)
                        status = "OK" if r.success else "FAILED"
                        log.info(f"  {name}: {status}")
                    except Exception as e:
                        log.error(f"  {name}: EXCEPTION {e}")
                        results.append(ForcingResult(
                            success=False, source=name, errors=[str(e)]))

        # ---- Phase 2: Heavy and dependent steps (sequential) ----

        # RTOFS OBC: heavy NetCDF I/O + Delaunay interpolation
        if "rtofs" in self.paths and (self.is_stofs or not self.skip_legacy):
            results.append(self._run_rtofs(output_dir, phase, time_hotstart))
        elif self.skip_legacy:
            log.info("Skipping RTOFS OBC (handled by legacy shell)")

        # Nudging needs RTOFS to have completed
        if self.config.nudging_enabled and "rtofs" in self.paths:
            results.append(self._run_nudging(output_dir, phase, time_hotstart))

        # param.nml only needs time_hotstart
        if "fix" in self.paths:
            results.append(self._run_param_nml(output_dir, phase, time_hotstart))

        # UFS-specific (DATM) needs GFS + HRRR sflux
        if self.config.nws == 4:
            results.append(self._run_datm(output_dir))

        # Step 9: Write time marker files (matches ORG behavior)
        self._write_time_markers(output_dir, phase, time_hotstart)

        elapsed = time.time() - t0

        # Determine overall success (all critical steps must succeed)
        critical_sources = {"GFS", "PARAM_NML", "TIDAL"}
        if self.is_stofs:
            # STOFS: NWM and RTOFS are also critical
            critical_sources.update({"NWM", "RTOFS"})

        success = all(
            r.success for r in results
            if r.source in critical_sources
        )
        if not any(r.source in critical_sources for r in results):
            success = any(r.success for r in results)

        prep_result = PrepResult(
            success=success, phase=phase,
            results=results, elapsed_seconds=elapsed,
        )
        log.info(prep_result.summary())
        return prep_result

    def _run_hotstart(self, output_dir: Path) -> ForcingResult:
        """Step 1: Find and validate hotstart file."""
        from .forcing.hotstart import HotstartProcessor

        restart_dir = self.paths.get("restart", self.paths.get("comout", output_dir))
        proc = HotstartProcessor(
            self.config, restart_dir, output_dir,
            run_name=self.run_name,
        )
        return proc.process()

    def _run_gfs(self, output_dir: Path, phase: str,
                 time_hotstart=None) -> ForcingResult:
        """Step 2: GFS atmospheric forcing."""
        from .forcing.gfs import GFSProcessor

        sflux_dir = output_dir / "sflux" if self.config.nws == 2 else output_dir
        proc = GFSProcessor(
            self.config, self.paths["gfs"], sflux_dir,
            resolution=self.config.gfs_resolution,
            phase=phase, time_hotstart=time_hotstart,
        )
        return proc.process()

    def _run_hrrr(self, output_dir: Path, phase: str,
                  time_hotstart=None) -> ForcingResult:
        """Step 3: HRRR secondary atmospheric (optional)."""
        from .forcing.hrrr import HRRRProcessor

        sflux_dir = output_dir / "sflux" if self.config.nws == 2 else output_dir
        proc = HRRRProcessor(
            self.config, self.paths["hrrr"], sflux_dir,
            phase=phase, time_hotstart=time_hotstart,
        )
        return proc.process()

    def _run_nwm(self, output_dir: Path, phase: str = "nowcast",
                 time_hotstart=None) -> ForcingResult:
        """Step 4: NWM river forcing."""
        from .forcing.nwm import NWMProcessor

        proc = NWMProcessor(
            self.config, self.paths["nwm"], output_dir,
            phase=phase, time_hotstart=time_hotstart,
        )
        return proc.process()

    def _run_rtofs(self, output_dir: Path, phase: str = "nowcast",
                   time_hotstart=None) -> ForcingResult:
        """Step 5: RTOFS ocean boundary conditions."""
        from .forcing.rtofs import RTOFSProcessor

        fix_dir = Path(self.paths.get("fix", ""))

        # Find obc.ctl and vgrid.in in FIX directory
        obc_ctl = None
        vgrid = None
        if fix_dir.exists():
            for f in fix_dir.glob("*.obc.ctl"):
                obc_ctl = f
                break
            for f in fix_dir.glob("*.vgrid.in"):
                sz = f.stat().st_size
                if sz < 100000:
                    # Simple format (68 lines) — preferred
                    vgrid = f
                    break
                elif sz > 1000000:
                    # LSC2 per-node format (1.6GB) — try to read anyway
                    # SchismVgrid.read() handles the simple header
                    vgrid = f
                    # Keep looking for a smaller one

        # Find grid file (hgrid.ll) in FIX directory if not in config
        grid_file = self.config.grid_file
        if (grid_file is None or not Path(grid_file).exists()) and fix_dir.exists():
            for f in fix_dir.glob("*.hgrid.ll"):
                grid_file = f
                break
            if grid_file is None:
                for f in fix_dir.glob("*.hgrid.gr3"):
                    grid_file = f
                    break

        proc = RTOFSProcessor(
            self.config, self.paths["rtofs"], output_dir,
            grid_file=grid_file,
            obc_ctl_file=obc_ctl,
            vgrid_file=vgrid,
            phase=phase,
            time_hotstart=time_hotstart,
        )
        return proc.process()

    def _run_nudging(self, output_dir: Path, phase: str = "nowcast",
                     time_hotstart=None) -> ForcingResult:
        """Step 5b: T/S interior nudging (STOFS)."""
        from .forcing.nudging import NudgingProcessor

        # Nudging needs its own RTOFS data prep with nudge-specific ROI
        # (wider domain than OBC ROI — cannot reuse OBC TS_1.nc)
        input_dir = output_dir
        rtofs_path = self.paths.get("rtofs")
        proc = NudgingProcessor(
            self.config, input_dir, output_dir,
            rtofs_input_path=rtofs_path,
        )
        return proc.process()

    def _run_tidal(self, output_dir: Path, phase: str = "nowcast",
                   time_hotstart=None) -> ForcingResult:
        """Step 6: Tidal forcing."""
        from .forcing.tidal import TidalProcessor

        fix_dir = self.paths.get("fix", self.paths.get("output", output_dir))
        proc = TidalProcessor(
            self.config, fix_dir, output_dir,
            phase=phase, time_hotstart=time_hotstart,
        )
        return proc.process()

    def _write_time_markers(self, output_dir: Path, phase: str,
                            time_hotstart=None) -> None:
        """Write time marker files to COMOUT (matches ORG behavior)."""
        from datetime import timedelta

        cycle_dt = datetime.strptime(self.config.pdy, "%Y%m%d") + \
                   timedelta(hours=self.config.cyc)
        nowcast_end = cycle_dt
        forecast_end = cycle_dt + timedelta(hours=self.config.forecast_hours)

        fmt = "%Y%m%d%H"
        cycle_tag = f"t{self.config.cyc:02d}z"

        markers = {
            f"time_nowcastend.{cycle_tag}": nowcast_end.strftime(fmt),
            f"time_forecastend.{cycle_tag}": forecast_end.strftime(fmt),
        }

        if time_hotstart:
            markers[f"time_hotstart.{cycle_tag}"] = time_hotstart.strftime(fmt)
            markers[f"base_date.{cycle_tag}"] = time_hotstart.strftime(fmt)

        for filename, value in markers.items():
            (output_dir / filename).write_text(value + "\n")

        log.info(f"Wrote time markers: {list(markers.keys())}")

    def _run_param_nml(self, output_dir: Path, phase: str,
                       time_hotstart=None) -> ForcingResult:
        """Step 7: Generate param.nml."""
        from .forcing.param_nml import ParamNmlProcessor

        proc = ParamNmlProcessor(
            self.config, self.paths["fix"], output_dir,
            phase=phase,
            time_hotstart=time_hotstart,
        )
        return proc.process()

    def _run_datm(self, output_dir: Path) -> ForcingResult:
        """Step 8: DATM blending + ESMF mesh for UFS-Coastal."""
        output_files = []
        warnings = []

        # 8a: Blend GFS+HRRR sflux into datm_forcing.nc (if sflux dir has both sources)
        sflux_dir = output_dir / "sflux"
        if sflux_dir.exists():
            gfs_sflux = list(sflux_dir.glob("sflux_air_1.*.nc"))
            hrrr_sflux = list(sflux_dir.glob("sflux_air_2.*.nc"))

            if gfs_sflux:
                from .forcing.blender import BlenderProcessor
                blender = BlenderProcessor(self.config, sflux_dir, output_dir)
                blend_result = blender.process()
                if blend_result.success:
                    output_files.extend(blend_result.output_files)
                    log.info(f"DATM blending: {len(blend_result.output_files)} files")
                else:
                    warnings.extend(blend_result.errors)

        # 8b: Generate ESMF mesh from the datm_forcing.nc
        datm_file = output_dir / "datm_forcing.nc"
        if datm_file.exists():
            from .forcing.esmf_mesh import ESMFMeshProcessor
            mesh_proc = ESMFMeshProcessor(
                self.config, output_dir, output_dir,
                forcing_file=datm_file,
            )
            mesh_result = mesh_proc.process()
            if mesh_result.success:
                output_files.extend(mesh_result.output_files)
            else:
                warnings.extend(mesh_result.errors)
        else:
            warnings.append("datm_forcing.nc not found — ESMF mesh not generated")

        return ForcingResult(
            success=True, source="DATM",
            output_files=output_files,
            warnings=warnings,
        )

    def archive_to_comout(self, result: PrepResult, comout: Path) -> List[Path]:
        """
        Archive prep outputs to COMOUT with NCO naming convention.

        Creates tars for sflux and OBC, copies individual files for
        param.nml, bctides.in, source_sink.in, etc.

        Args:
            result: PrepResult from run()
            comout: COMOUT directory path

        Returns:
            List of archived file paths
        """
        import shutil
        import subprocess

        comout = Path(comout)
        comout.mkdir(parents=True, exist_ok=True)

        archived = []
        prefix = self.run_name
        cycle = f"t{self.config.cyc:02d}z"
        pdy = self.config.pdy
        phase = result.phase

        work_dir = self.paths["output"]
        if isinstance(work_dir, str):
            work_dir = Path(work_dir)

        log.info(f"Archiving to {comout}")

        # Tar sflux files — separate GFS (stack 1) and HRRR (stack 2) like ORG
        sflux_dir = work_dir / "sflux"
        if sflux_dir.exists():
            # GFS (primary): sflux_*_1.*.nc → met.{phase}.nc.tar
            gfs_files = sorted(sflux_dir.glob("sflux_*_1.*.nc"))
            if gfs_files:
                tar_name = f"{prefix}.{cycle}.{pdy}.met.{phase}.nc.tar"
                tar_path = comout / tar_name
                try:
                    file_list = [f.name for f in gfs_files]
                    subprocess.run(
                        ["tar", "-cf", str(tar_path), "-C", str(sflux_dir)] + file_list,
                        check=True, capture_output=True,
                    )
                    archived.append(tar_path)
                    log.info(f"  Archived GFS sflux -> {tar_name}")
                except (subprocess.CalledProcessError, FileNotFoundError) as e:
                    log.warning(f"  Failed to tar GFS sflux: {e}")

            # HRRR (secondary): sflux_*_2.*.nc → met.{phase}.nc.2.tar
            hrrr_files = sorted(sflux_dir.glob("sflux_*_2.*.nc"))
            if hrrr_files:
                tar_name = f"{prefix}.{cycle}.{pdy}.met.{phase}.nc.2.tar"
                tar_path = comout / tar_name
                try:
                    file_list = [f.name for f in hrrr_files]
                    subprocess.run(
                        ["tar", "-cf", str(tar_path), "-C", str(sflux_dir)] + file_list,
                        check=True, capture_output=True,
                    )
                    archived.append(tar_path)
                    log.info(f"  Archived HRRR sflux -> {tar_name}")
                except (subprocess.CalledProcessError, FileNotFoundError) as e:
                    log.warning(f"  Failed to tar HRRR sflux: {e}")

        # Tar OBC files
        # COMF convention: one combined obc.tar with all 6 files (boundary + nudging)
        obc_files = ["elev2D.th.nc", "TEM_3D.th.nc", "SAL_3D.th.nc", "uv3D.th.nc",
                     "TEM_nu.nc", "SAL_nu.nc"]
        existing_obc = [work_dir / f for f in obc_files if (work_dir / f).exists()]
        if existing_obc:
            # Phase-specific tar (backward compat): obc.nowcast.tar / obc.forecast.tar
            phase_tar_name = f"{prefix}.{cycle}.{pdy}.obc.{phase}.tar"
            phase_tar_path = comout / phase_tar_name
            try:
                file_list = [f.name for f in existing_obc]
                subprocess.run(
                    ["tar", "-cf", str(phase_tar_path), "-C", str(work_dir)] + file_list,
                    check=True, capture_output=True,
                )
                archived.append(phase_tar_path)
                log.info(f"  Archived OBC -> {phase_tar_name}")
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                log.warning(f"  Failed to tar OBC (phase): {e}")

            # Combined obc.tar (COMF convention): single tar with all 6 files
            combined_tar_name = f"{prefix}.{cycle}.{pdy}.obc.tar"
            combined_tar_path = comout / combined_tar_name
            try:
                file_list = [f.name for f in existing_obc]
                subprocess.run(
                    ["tar", "-cf", str(combined_tar_path), "-C", str(work_dir)] + file_list,
                    check=True, capture_output=True,
                )
                archived.append(combined_tar_path)
                log.info(f"  Archived OBC (combined) -> {combined_tar_name}")
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                log.warning(f"  Failed to tar OBC (combined): {e}")

        # Tar NWM source/sink files (COMF convention)
        # COMF produces: nwm.source.sink.now.tar / nwm.source.sink.fore.tar
        nwm_files = ["vsource.th", "msource.th", "source_sink.in", "vsink.th"]
        existing_nwm = [work_dir / f for f in nwm_files if (work_dir / f).exists()]
        if existing_nwm:
            phase_tag = "now" if phase == "nowcast" else "fore"
            nwm_tar_name = f"{prefix}.{cycle}.{pdy}.nwm.source.sink.{phase_tag}.tar"
            nwm_tar_path = comout / nwm_tar_name
            try:
                file_list = [f.name for f in existing_nwm]
                subprocess.run(
                    ["tar", "-cf", str(nwm_tar_path), "-C", str(work_dir)] + file_list,
                    check=True, capture_output=True,
                )
                archived.append(nwm_tar_path)
                log.info(f"  Archived NWM -> {nwm_tar_name}")
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                log.warning(f"  Failed to tar NWM: {e}")

        # Copy individual files
        copy_map = {
            "param.nml": f"{prefix}.{cycle}.{pdy}.{phase}.in",
            "bctides.in": f"{prefix}.{cycle}.{pdy}.bctides.in.{phase}",
            "source_sink.in": f"{prefix}.source_sink.in",
            "vsource.th": f"{prefix}.{cycle}.{pdy}.river.vsource.th",
            "msource.th": f"{prefix}.{cycle}.{pdy}.river.msource.th",
            "sflux_inputs.txt": "sflux_inputs.txt",
            "datm_forcing.nc": f"{prefix}.{cycle}.datm_forcing.nc",
            "esmf_mesh.nc": f"{prefix}.{cycle}.esmf_mesh.nc",
            "partition.prop": "partition.prop",
        }

        for src_name, dst_name in copy_map.items():
            # Check both work_dir and sflux subdir
            src = work_dir / src_name
            if not src.exists():
                src = work_dir / "sflux" / src_name
            if src.exists():
                dst = comout / dst_name
                shutil.copy2(src, dst)
                archived.append(dst)
                log.info(f"  Copied {src_name} -> {dst_name}")

        # Time marker files
        for marker in [f"time_hotstart.{cycle}", f"time_nowcastend.{cycle}",
                       f"time_forecastend.{cycle}", f"base_date.{cycle}"]:
            src = work_dir / marker
            if src.exists():
                dst = comout / marker
                shutil.copy2(src, dst)
                archived.append(dst)

        log.info(f"Archived {len(archived)} files to {comout}")
        return archived
