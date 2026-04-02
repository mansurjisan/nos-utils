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
    ):
        """
        Args:
            config: ForcingConfig with run window and model settings
            input_path: Directory containing param.nml template (FIXofs)
            output_path: Directory to write generated param.nml
            template_name: Template filename (default: param.nml)
            phase: "nowcast" or "forecast" — determines rnday and start time
        """
        super().__init__(config, input_path, output_path)
        self.template_name = template_name
        self.phase = phase

    def process(self) -> ForcingResult:
        """Generate param.nml from template."""
        log.info(f"param.nml processor: phase={self.phase}")
        self.create_output_dir()

        # Find template
        template_path = self.input_path / self.template_name
        if not template_path.exists():
            # Try common alternative names
            for alt in [f"{self.template_name}_template", "param.nml.template"]:
                alt_path = self.input_path / alt
                if alt_path.exists():
                    template_path = alt_path
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
        """Compute parameter values based on config and phase."""
        cycle_dt = datetime.strptime(self.config.pdy, "%Y%m%d") + \
                   timedelta(hours=self.config.cyc)

        if self.phase == "nowcast":
            start_dt = cycle_dt - timedelta(hours=self.config.nowcast_hours)
            end_dt = cycle_dt
            run_hours = self.config.nowcast_hours
        elif self.phase == "forecast":
            start_dt = cycle_dt
            end_dt = cycle_dt + timedelta(hours=self.config.forecast_hours)
            run_hours = self.config.forecast_hours
        else:
            # Full run (nowcast + forecast)
            start_dt = cycle_dt - timedelta(hours=self.config.nowcast_hours)
            end_dt = cycle_dt + timedelta(hours=self.config.forecast_hours)
            run_hours = self.config.nowcast_hours + self.config.forecast_hours

        rnday = run_hours / 24.0

        return {
            "rnday_value": f"{rnday:.4f}",
            "start_year_value": str(start_dt.year),
            "start_month_value": str(start_dt.month),
            "start_day_value": str(start_dt.day),
            "start_hour_value": f"{start_dt.hour:.1f}",
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
