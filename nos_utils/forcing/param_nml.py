"""
SCHISM param.nml generator.

Reads a param.nml template and substitutes runtime parameters:
  - rnday: run duration in days
  - start_year/month/day/hour: model start time
  - ihot: hotstart mode (0=cold, 1=hotstart with time reset)
  - nws: atmospheric forcing mode (0=none, 2=sflux, 4=DATM/UFS)

Replaces: nos_ofs_prep_schism_ctl.sh (sed-based substitution)

Template placeholders (Fortran namelist format):
  rnday = rnday_value
  start_year = start_year_value
  start_month = start_month_value
  start_day = start_day_value
  start_hour = start_hour_value
"""

import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

from ..config import ForcingConfig
from .base import ForcingProcessor, ForcingResult

log = logging.getLogger(__name__)


class ParamNmlProcessor(ForcingProcessor):
    """
    Generate SCHISM param.nml from template with runtime substitutions.

    Usage:
        proc = ParamNmlProcessor(config, template_path, output_path)
        result = proc.process()
    """

    SOURCE_NAME = "PARAM_NML"

    def __init__(
        self,
        config: ForcingConfig,
        input_path: Path,
        output_path: Path,
        template_name: str = "param.nml",
        phase: str = "nowcast",
        time_hotstart: Optional[datetime] = None,
    ):
        """
        Args:
            config: ForcingConfig with run window and model settings
            input_path: Directory containing param.nml template (FIXofs)
            output_path: Directory to write generated param.nml
            template_name: Template filename (default: param.nml)
            phase: "nowcast" or "forecast" — determines rnday and start time
            time_hotstart: Hotstart datetime from restart file. If provided,
                nowcast starts from this time (not cycle-nowcast_hours).
                This matches the legacy shell behavior where rnday spans
                from time_hotstart to nowcastend.
        """
        super().__init__(config, input_path, output_path)
        self.template_name = template_name
        self.phase = phase
        self.time_hotstart = time_hotstart

    def process(self) -> ForcingResult:
        """Generate param.nml from template."""
        log.info(f"param.nml processor: phase={self.phase}")
        self.create_output_dir()

        # Find template — try multiple naming conventions
        template_path = self.input_path / self.template_name
        if not template_path.exists():
            # Common alternative names: {ofs}.param.nml, {ofs}_param.nml, etc.
            alternatives = [
                f"{self.template_name}_template",
                "param.nml.template",
            ]
            # Also glob for any file containing "param.nml" or "param_nml"
            found = sorted(self.input_path.glob("*param.nml*")) + \
                    sorted(self.input_path.glob("*param_nml*"))
            alternatives.extend([f.name for f in found])

            for alt in alternatives:
                alt_path = self.input_path / alt
                if alt_path.exists():
                    template_path = alt_path
                    log.info(f"Using template: {alt_path.name}")
                    break
            else:
                return ForcingResult(
                    success=False, source=self.SOURCE_NAME,
                    errors=[f"Template not found: {template_path}"],
                )

        # Compute substitution values
        subs = self._compute_substitutions()

        # Read template and substitute
        content = template_path.read_text()
        patched = self._apply_substitutions(content, subs)

        # Write output
        output_file = self.output_path / "param.nml"
        output_file.write_text(patched)

        log.info(f"Created param.nml: rnday={subs['rnday_value']}, "
                 f"ihot={subs.get('ihot', 1)}, start={subs['start_year_value']}-"
                 f"{subs['start_month_value']}-{subs['start_day_value']} "
                 f"{subs['start_hour_value']}:00")

        return ForcingResult(
            success=True, source=self.SOURCE_NAME,
            output_files=[output_file],
            metadata={
                "phase": self.phase,
                "rnday": subs["rnday_value"],
                "start_time": f"{subs['start_year_value']}-{subs['start_month_value']:>02s}-"
                              f"{subs['start_day_value']:>02s}T{subs['start_hour_value']}:00",
                "ihot": subs.get("ihot", 1),
                "template": str(template_path),
            },
        )

    def find_input_files(self):
        template = self.input_path / self.template_name
        return [template] if template.exists() else []

    def _compute_substitutions(self) -> Dict[str, str]:
        """
        Compute parameter values based on config, phase, and hotstart time.

        Legacy shell behavior:
          Nowcast:  start=time_hotstart, rnday=(nowcastend-hotstart)/24, ihot=1
          Forecast: start=nowcastend,    rnday=forecast_hours/24,        ihot=2
        """
        cycle_dt = datetime.strptime(self.config.pdy, "%Y%m%d") + \
                   timedelta(hours=self.config.cyc)
        nowcast_end = cycle_dt  # nowcastend = PDY + cyc

        if self.phase == "nowcast":
            # Start from hotstart time if available, otherwise cycle - nowcast_hours
            if self.time_hotstart:
                start_dt = self.time_hotstart
            else:
                start_dt = cycle_dt - timedelta(hours=self.config.nowcast_hours)
            end_dt = nowcast_end
            run_hours = (end_dt - start_dt).total_seconds() / 3600.0
            ihot = 1
        elif self.phase == "forecast":
            start_dt = nowcast_end
            end_dt = nowcast_end + timedelta(hours=self.config.forecast_hours)
            run_hours = self.config.forecast_hours
            ihot = 2  # Forecast continues from nowcast hotstart (don't reset clock)
        else:
            # Full run (nowcast + forecast)
            if self.time_hotstart:
                start_dt = self.time_hotstart
            else:
                start_dt = cycle_dt - timedelta(hours=self.config.nowcast_hours)
            end_dt = nowcast_end + timedelta(hours=self.config.forecast_hours)
            run_hours = (end_dt - start_dt).total_seconds() / 3600.0
            ihot = 1

        rnday = run_hours / 24.0

        return {
            "rnday_value": f"{rnday:.4f}",
            "start_year_value": str(start_dt.year),
            "start_month_value": f"{start_dt.month:02d}",
            "start_day_value": f"{start_dt.day:02d}",
            "start_hour_value": f"{start_dt.hour:.1f}",
            "ihot_value": str(ihot),
        }

    def _apply_substitutions(self, content: str, subs: Dict[str, str]) -> str:
        """Replace placeholder values in param.nml content."""
        result = content
        for placeholder, value in subs.items():
            result = result.replace(placeholder, value)

        # Remove lines containing "DUMMY" (placeholder cleanup)
        lines = result.split("\n")
        result = "\n".join(line for line in lines if "DUMMY" not in line)

        return result

    @staticmethod
    def patch_param(param_path: Path, **kwargs) -> None:
        """
        Patch specific values in an existing param.nml file.

        Usage:
            ParamNmlProcessor.patch_param(path, rnday=2.0, ihot=1, nws=4)
        """
        content = param_path.read_text()

        for key, value in kwargs.items():
            # Match Fortran namelist pattern: "  key = old_value"
            if isinstance(value, float):
                val_str = f"{value}"
            elif isinstance(value, int):
                val_str = str(value)
            else:
                val_str = str(value)

            pattern = rf"(\s*{key}\s*=\s*)([^\s!]+)"
            replacement = rf"\g<1>{val_str}"
            content = re.sub(pattern, replacement, content)

        param_path.write_text(content)
