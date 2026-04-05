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
        phase: str = "nowcast",
        time_hotstart: Optional[datetime] = None,
    ):
        """
        Args:
            config: ForcingConfig
            input_path: FIX directory with bctides template
            output_path: Output directory
            phase: "nowcast" or "forecast" — determines start time for nodal corrections
            time_hotstart: Hotstart datetime (nowcast starts from here)
        """
        super().__init__(config, input_path, output_path)
        self.phase = phase
        self.time_hotstart = time_hotstart

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
        # Auto-discover template if not explicitly set
        if not template or not Path(template).exists():
            for f in sorted(self.input_path.glob("*bctides*template*")):
                template = f
                log.info(f"Auto-discovered bctides template: {f.name}")
                break
        if template and Path(template).exists():
            log.info(f"Using template: {template}")
            result = self._process_template(Path(template), output_file)
            if result:
                return ForcingResult(
                    success=True, source=self.SOURCE_NAME,
                    output_files=[output_file],
                    metadata={"mode": "template"},
                )

        # Mode 2: Copy from input_path (try multiple naming patterns)
        search_names = ["bctides.in", f"{self.config.pdy}_bctides.in"]
        # Also glob for {ofs}.bctides.in patterns
        search_names.extend([f.name for f in sorted(self.input_path.glob("*bctides.in"))
                             if "template" not in f.name])
        for name in search_names:
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

    def _compute_start_time(self) -> datetime:
        """Compute the model start time for this phase."""
        cycle_dt = datetime.strptime(self.config.pdy, "%Y%m%d") + \
                   timedelta(hours=self.config.cyc)

        if self.phase == "nowcast":
            if self.time_hotstart:
                return self.time_hotstart
            return cycle_dt - timedelta(hours=self.config.nowcast_hours)
        elif self.phase == "forecast":
            return cycle_dt
        else:
            if self.time_hotstart:
                return self.time_hotstart
            return cycle_dt - timedelta(hours=self.config.nowcast_hours)

    def _process_template(self, template_path: Path, output_path: Path) -> bool:
        """
        Update a bctides.in_template with start time and nodal corrections.

        The template contains tidal potential and boundary forcing sections.
        We update:
          1. Line 1: start time in MM/DD/YYYY HH:MM:SS UTC format
          2. Nodal factor (f) and equilibrium argument for each constituent
             in the tidal potential section (lines after constituent names)
        """
        start_time = self._compute_start_time()

        try:
            lines = template_path.read_text().splitlines()

            # Update line 1: start time in MM/DD/YYYY format (matching ORG Fortran)
            lines[0] = start_time.strftime("%m/%d/%Y %H:%M:%S") + " UTC"

            # Compute nodal corrections for this start time
            nodal = compute_nodal_corrections(start_time, self.config.tidal_constituents)

            # Update nodal factors in the tidal potential section
            # Format: after each constituent name line, the next line has:
            #   n_doodson amplitude omega nodal_factor equilibrium_arg
            # We update nodal_factor and equilibrium_arg
            i = 2  # Start after line 0 (date) and line 1 (ntip tip_dp)
            while i < len(lines) - 1:
                line = lines[i].strip()
                # Check if this line is a constituent name
                if line in nodal:
                    # Next line has the nodal parameters
                    i += 1
                    parts = lines[i].split()
                    if len(parts) >= 4:
                        f_val = nodal[line]["f"]
                        v0_plus_u = (nodal[line]["v0"] + nodal[line]["u"]) % 360.0
                        parts[2] = f"{f_val:.5f}"
                        parts[3] = f"{v0_plus_u:.5f}"
                        lines[i] = " ".join(parts)
                i += 1

            # Write updated file
            output_path.write_text("\n".join(lines) + "\n")
            log.info(f"Updated template: phase={self.phase}, start={start_time}, "
                     f"nodal corrections applied for {len(nodal)} constituents")
            return True

        except Exception as e:
            log.error(f"Failed to process template: {e}")
            import traceback
            log.error(traceback.format_exc())
            return False

    def _generate_python(self, output_path: Path) -> None:
        """Generate a minimal bctides.in from scratch using Python."""
        start_time = self._compute_start_time()

        constituents = self.config.tidal_constituents
        nodal = compute_nodal_corrections(start_time, constituents)

        with open(output_path, "w") as f:
            # Line 1: start time in MM/DD/YYYY format
            f.write(f"{start_time.strftime('%m/%d/%Y %H:%M:%S')} UTC\n")

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
                eq_arg = (nodal[const_name]["v0"] + nodal[const_name]["u"]) % 360.0

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
    Compute tidal nodal corrections (f, u) and astronomical argument V0.

    The nodal factor f and argument u account for the 18.6-year modulation
    of the lunar orbit. V0 is the equilibrium tide argument at the given time.

    Args:
        start_time: Model start datetime
        constituents: List of constituent names

    Returns:
        Dict mapping constituent name -> {"f": nodal_factor, "u": deg, "v0": deg}
    """
    # Julian centuries from J2000.0
    j2000 = datetime(2000, 1, 1, 12, 0, 0)
    T = (start_time - j2000).total_seconds() / (365.25 * 86400.0)
    T_century = T / 100.0  # Julian centuries

    N = math.radians(259.157 - 19.328 * T)

    # Compute p (lunar perigee longitude) for nodal corrections
    p = math.radians(83.3535 + 40.6693 * T)

    # Astronomical longitudes for V0 (degrees, Schureman conventions)
    s_deg = (218.3165 + 481267.8813 * T_century) % 360.0   # Moon mean longitude
    h_deg = (280.4662 + 36000.7698 * T_century) % 360.0    # Sun mean longitude
    p_deg = (83.3535 + 4069.0137 * T_century) % 360.0      # Lunar perigee
    N_deg = (125.0445 - 1934.1363 * T_century) % 360.0     # Ascending node
    pp_deg = (282.9384 + 1.7195 * T_century) % 360.0       # Solar perihelion

    # Mean lunar time (tau): hour angle of mean Moon + 180°
    hour = start_time.hour + start_time.minute / 60.0 + start_time.second / 3600.0
    tau_deg = (15.0 * hour + h_deg - s_deg + 180.0) % 360.0

    # Astronomical arguments: (tau, s, h, p, N', p1)
    astro_args = [tau_deg, s_deg, h_deg, p_deg, -N_deg, pp_deg]

    result = {}
    for const in constituents:
        f, u = _nodal_fu(const, N, p)

        # Compute V0 from Doodson numbers
        props = TIDAL_CONSTITUENTS.get(const)
        if props and "doodson" in props:
            doodson = props["doodson"]
            v0 = sum(d * a for d, a in zip(doodson, astro_args)) % 360.0
        else:
            v0 = 0.0

        result[const] = {"f": f, "u": u, "v0": v0}

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
