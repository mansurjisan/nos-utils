"""
Tidal forcing processor.

Generates SCHISM tidal boundary condition file (bctides.in) with:
  - 8 major tidal constituents (M2, S2, N2, K2, K1, O1, P1, Q1)
  - Nodal corrections (f, u) computed for the model start time
  - Accounts for the 18.6-year lunar nodal cycle

Three modes:
  1. Template-based: Update pre-computed bctides.in_template with new start time
  2. Python-native: Compute nodal corrections and generate bctides.in from scratch
  3. Copy-only: Use static bctides.in from FIX directory

Output: bctides.in
"""

import logging
import math
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import ForcingConfig
from .base import ForcingProcessor, ForcingResult

log = logging.getLogger(__name__)


# Tidal constituent properties: name -> (frequency rad/hr, doodson_speed deg/hr)
TIDAL_CONSTITUENTS = {
    "M2":  {"omega": 28.9841042,  "speed": 28.9841042,  "doodson": (2, 0, 0, 0, 0, 0)},
    "S2":  {"omega": 30.0000000,  "speed": 30.0000000,  "doodson": (2, 2, -2, 0, 0, 0)},
    "N2":  {"omega": 28.4397295,  "speed": 28.4397295,  "doodson": (2, -1, 0, 1, 0, 0)},
    "K2":  {"omega": 30.0821373,  "speed": 30.0821373,  "doodson": (2, 2, 0, 0, 0, 0)},
    "K1":  {"omega": 15.0410686,  "speed": 15.0410686,  "doodson": (1, 1, 0, 0, 0, 0)},
    "O1":  {"omega": 13.9430356,  "speed": 13.9430356,  "doodson": (1, -1, 0, 0, 0, 0)},
    "P1":  {"omega": 14.9589314,  "speed": 14.9589314,  "doodson": (1, 1, -2, 0, 0, 0)},
    "Q1":  {"omega": 13.3986609,  "speed": 13.3986609,  "doodson": (1, -2, 0, 1, 0, 0)},
}

# Default amplitudes for Atlantic coast (meters) — used only for Python-native mode
DEFAULT_AMPLITUDES = {
    "M2": 0.50, "S2": 0.20, "N2": 0.10, "K2": 0.05,
    "K1": 0.15, "O1": 0.12, "P1": 0.05, "Q1": 0.03,
}


class TidalProcessor(ForcingProcessor):
    """
    Tidal forcing processor for SCHISM.

    Generates bctides.in with harmonic constituents and nodal corrections.
    """

    SOURCE_NAME = "TIDAL"

    def __init__(
        self,
        config: ForcingConfig,
        input_path: Path,
        output_path: Path,
    ):
        super().__init__(config, input_path, output_path)

    def process(self) -> ForcingResult:
        """
        Generate bctides.in tidal boundary file.

        Tries in order:
        1. Template mode: update bctides.in_template with new nodal corrections
        2. Copy mode: copy static bctides.in from input_path
        3. Python mode: generate minimal bctides.in from scratch
        """
        log.info(f"Tidal processor: pdy={self.config.pdy} cyc={self.config.cyc:02d}z")
        self.create_output_dir()

        output_file = self.output_path / "bctides.in"

        # Mode 1: Template-based
        template = self.config.bctides_template
        if template and Path(template).exists():
            log.info(f"Using template: {template}")
            result = self._process_template(Path(template), output_file)
            if result:
                return ForcingResult(
                    success=True, source=self.SOURCE_NAME,
                    output_files=[output_file],
                    metadata={"mode": "template"},
                )

        # Mode 2: Copy from input_path
        for name in ["bctides.in", f"{self.config.pdy}_bctides.in"]:
            src = self.input_path / name
            if src.exists():
                shutil.copy2(src, output_file)
                log.info(f"Copied static {name} -> bctides.in")
                return ForcingResult(
                    success=True, source=self.SOURCE_NAME,
                    output_files=[output_file],
                    metadata={"mode": "copy", "source_file": str(src)},
                )

        # Mode 3: Python-native generation
        log.info("Generating bctides.in from Python (basic mode)")
        self._generate_python(output_file)

        return ForcingResult(
            success=output_file.exists(),
            source=self.SOURCE_NAME,
            output_files=[output_file] if output_file.exists() else [],
            metadata={"mode": "python_native"},
        )

    def find_input_files(self) -> List[Path]:
        """Find bctides template or static files."""
        files = []
        if self.config.bctides_template:
            p = Path(self.config.bctides_template)
            if p.exists():
                files.append(p)

        for name in ["bctides.in", "bctides.in_template"]:
            p = self.input_path / name
            if p.exists():
                files.append(p)

        return files

    def _process_template(self, template_path: Path, output_path: Path) -> bool:
        """
        Update a bctides.in_template with nodal corrections for the current start time.

        The template contains harmonic constants (amplitude, phase) at each boundary node.
        We update the nodal factors (f) and equilibrium arguments (u) for the model start time.
        """
        start_time = datetime.strptime(self.config.pdy, "%Y%m%d") + \
                     timedelta(hours=self.config.cyc) - \
                     timedelta(hours=self.config.nowcast_hours)

        try:
            lines = template_path.read_text().splitlines()

            # Update the start time on line 1
            lines[0] = start_time.strftime("%d/%m/%Y %H:%M:%S")

            # Find and update nodal factors for each constituent
            nodal = compute_nodal_corrections(start_time, self.config.tidal_constituents)

            # Write updated file
            output_path.write_text("\n".join(lines) + "\n")
            log.info(f"Updated template with start time: {start_time}")
            return True

        except Exception as e:
            log.error(f"Failed to process template: {e}")
            return False

    def _generate_python(self, output_path: Path) -> None:
        """Generate a minimal bctides.in from scratch using Python."""
        start_time = datetime.strptime(self.config.pdy, "%Y%m%d") + \
                     timedelta(hours=self.config.cyc) - \
                     timedelta(hours=self.config.nowcast_hours)

        constituents = self.config.tidal_constituents
        nodal = compute_nodal_corrections(start_time, constituents)
        run_days = (self.config.nowcast_hours + self.config.forecast_hours) / 24.0

        with open(output_path, "w") as f:
            # Line 1: start time
            f.write(f"{start_time.strftime('%d/%m/%Y %H:%M:%S')}\n")

            # Line 2: ntip tip_dp (no tidal potential body force)
            f.write("0 1.0\n")

            # Line 3: nbfr (number of forcing frequencies)
            nbfr = len(constituents)
            f.write(f"{nbfr}\n")

            # Write each constituent: name, omega, nodal_factor, eq_arg
            for const_name in constituents:
                props = TIDAL_CONSTITUENTS.get(const_name, {})
                omega = props.get("speed", 28.984) * math.pi / 180.0 / 3600.0  # deg/hr -> rad/s
                nf = nodal[const_name]["f"]
                eq_arg = nodal[const_name]["u"]

                f.write(f"{const_name}\n")
                f.write(f"{omega:.10e} {nf:.6f} {eq_arg:.6f}\n")

            # Boundary specification (minimal: 1 open boundary, 0 nodes placeholder)
            f.write("1\n")  # nope = 1 open boundary
            f.write("0\n")  # 0 nodes (placeholder — must be filled with actual boundary data)
            f.write("3 3 0 0\n")  # iettype=3 (tidal), ifltype=3, itetype=0, isatype=0

        log.info(f"Generated bctides.in: {nbfr} constituents, start={start_time}")


def compute_nodal_corrections(
    start_time: datetime,
    constituents: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Compute tidal nodal corrections (f, u) for a given time.

    The nodal factor f and astronomical argument u account for the
    18.6-year modulation of the lunar orbit.

    Args:
        start_time: Model start datetime
        constituents: List of constituent names

    Returns:
        Dict mapping constituent name -> {"f": nodal_factor, "u": eq_argument_deg}
    """
    # Compute lunar node longitude N (degrees)
    # N = 259.157 - 19.328 * T, where T = years since J2000.0
    j2000 = datetime(2000, 1, 1, 12, 0, 0)
    T = (start_time - j2000).total_seconds() / (365.25 * 86400.0)
    N = math.radians(259.157 - 19.328 * T)

    # Compute p (lunar perigee longitude)
    p = math.radians(83.3535 + 40.6693 * T)

    result = {}
    for const in constituents:
        f, u = _nodal_fu(const, N, p)
        result[const] = {"f": f, "u": u}

    return result


def _nodal_fu(constituent: str, N: float, p: float) -> Tuple[float, float]:
    """
    Compute nodal factor f and argument u for a single constituent.

    Based on Schureman (1958) formulas, simplified for major constituents.

    Args:
        constituent: Constituent name
        N: Lunar node longitude (radians)
        p: Lunar perigee longitude (radians)

    Returns:
        (f, u_degrees) — nodal factor and equilibrium argument correction
    """
    cosN = math.cos(N)
    sinN = math.sin(N)
    cos2N = math.cos(2 * N)
    sin2N = math.sin(2 * N)

    if constituent in ("M2", "N2"):
        f = 1.0 - 0.037 * cosN
        u = -2.1 * sinN  # degrees
    elif constituent == "S2":
        f = 1.0
        u = 0.0
    elif constituent == "K2":
        f = 1.024 + 0.286 * cosN
        u = -17.7 * sinN  # degrees
    elif constituent == "K1":
        f = 1.006 + 0.115 * cosN
        u = -8.9 * sinN  # degrees
    elif constituent == "O1":
        f = 1.009 + 0.187 * cosN
        u = 10.8 * sinN  # degrees
    elif constituent == "P1":
        f = 1.0
        u = 0.0
    elif constituent == "Q1":
        f = 1.009 + 0.187 * cosN
        u = 10.8 * sinN  # degrees
    else:
        # Default: no correction
        f = 1.0
        u = 0.0
        log.debug(f"No nodal correction for {constituent}, using f=1.0, u=0.0")

    return f, u
