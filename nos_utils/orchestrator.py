"""
Prep orchestrator — chains all forcing processors in sequence.

Replaces exnos_ofs_prep.sh (~800 lines of shell) with a Python pipeline.

Usage:
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

    def __init__(
        self,
        config: ForcingConfig,
        paths: Dict[str, str],
        run_name: str = "secofs",
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
        """
        self.config = config
        self.paths = {k: Path(v) for k, v in paths.items()}
        self.run_name = run_name

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
                 f"pdy={self.config.pdy} cyc={self.config.cyc:02d}z")

        output_dir = self.paths["output"]
        output_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Hotstart
        results.append(self._run_hotstart(output_dir))

        # Step 2: GFS atmospheric
        if "gfs" in self.paths:
            results.append(self._run_gfs(output_dir, phase))

        # Step 3: HRRR secondary (optional, non-fatal)
        if "hrrr" in self.paths and self.config.met_num >= 2:
            results.append(self._run_hrrr(output_dir, phase))

        # Step 4: NWM river (optional)
        if "nwm" in self.paths:
            results.append(self._run_nwm(output_dir))

        # Step 5: RTOFS OBC (optional)
        if "rtofs" in self.paths:
            results.append(self._run_rtofs(output_dir))

        # Step 6: Tidal
        results.append(self._run_tidal(output_dir))

        # Step 7: param.nml
        if "fix" in self.paths:
            results.append(self._run_param_nml(output_dir, phase))

        # Step 8: UFS-specific (DATM)
        if self.config.nws == 4:
            results.append(self._run_datm(output_dir))

        elapsed = time.time() - t0

        # Determine overall success (all critical steps must succeed)
        critical_sources = {"GFS", "PARAM_NML"}
        success = all(
            r.success for r in results
            if r.source in critical_sources
        )
        # If no critical steps ran, check if at least something succeeded
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

    def _run_gfs(self, output_dir: Path, phase: str) -> ForcingResult:
        """Step 2: GFS atmospheric forcing."""
        from .forcing.gfs import GFSProcessor

        sflux_dir = output_dir / "sflux" if self.config.nws == 2 else output_dir
        proc = GFSProcessor(
            self.config, self.paths["gfs"], sflux_dir,
        )
        return proc.process()

    def _run_hrrr(self, output_dir: Path, phase: str) -> ForcingResult:
        """Step 3: HRRR secondary atmospheric (optional)."""
        from .forcing.hrrr import HRRRProcessor

        sflux_dir = output_dir / "sflux" if self.config.nws == 2 else output_dir
        proc = HRRRProcessor(
            self.config, self.paths["hrrr"], sflux_dir,
        )
        return proc.process()

    def _run_nwm(self, output_dir: Path) -> ForcingResult:
        """Step 4: NWM river forcing."""
        from .forcing.nwm import NWMProcessor

        proc = NWMProcessor(
            self.config, self.paths["nwm"], output_dir,
        )
        return proc.process()

    def _run_rtofs(self, output_dir: Path) -> ForcingResult:
        """Step 5: RTOFS ocean boundary conditions."""
        from .forcing.rtofs import RTOFSProcessor

        proc = RTOFSProcessor(
            self.config, self.paths["rtofs"], output_dir,
        )
        return proc.process()

    def _run_tidal(self, output_dir: Path) -> ForcingResult:
        """Step 6: Tidal forcing."""
        from .forcing.tidal import TidalProcessor

        fix_dir = self.paths.get("fix", self.paths.get("output", output_dir))
        proc = TidalProcessor(
            self.config, fix_dir, output_dir,
        )
        return proc.process()

    def _run_param_nml(self, output_dir: Path, phase: str) -> ForcingResult:
        """Step 7: Generate param.nml."""
        from .forcing.param_nml import ParamNmlProcessor

        proc = ParamNmlProcessor(
            self.config, self.paths["fix"], output_dir,
            phase=phase,
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
