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
import os
import shutil
import subprocess
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
        0. Fortran mode: call tide_fac executable with template (most accurate)
        1. Template mode: update bctides.in_template with Python nodal corrections
        2. Copy mode: copy static bctides.in from input_path
        3. Python mode: generate minimal bctides.in from scratch
        """
        log.info(f"Tidal processor: pdy={self.config.pdy} cyc={self.config.cyc:02d}z")
        self.create_output_dir()

        output_file = self.output_path / "bctides.in"

        # Find template (needed for both Fortran and template modes)
        template = self.config.bctides_template
        if not template or not Path(template).exists():
            for f in sorted(self.input_path.glob("*bctides*template*")):
                template = f
                log.info(f"Auto-discovered bctides template: {f.name}")
                break

        # Mode 0: Fortran tide_fac executable (production, most accurate)
        if template and Path(template).exists():
            result = self._call_fortran_tide_fac(Path(template), output_file)
            if result:
                return ForcingResult(
                    success=True, source=self.SOURCE_NAME,
                    output_files=[output_file],
                    metadata={"mode": "fortran_tide_fac"},
                )

        # Mode 1: Template-based with Python nodal corrections
        if template and Path(template).exists():
            log.warning(
                "Fortran tide_fac not available — using Python nodal corrections. "
                "For production, ensure nos_ofs_create_tide_fac_schism is in "
                "EXECnos or EXECofs."
            )
            result = self._process_template(Path(template), output_file)
            if result:
                return ForcingResult(
                    success=True, source=self.SOURCE_NAME,
                    output_files=[output_file],
                    metadata={"mode": "template"},
                )

        # Mode 2: Copy from input_path (try multiple naming patterns)
        search_names = ["bctides.in", f"{self.config.pdy}_bctides.in"]
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
        log.warning("Generating bctides.in from Python (basic mode) — approximate nodal corrections")
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

    def _call_fortran_tide_fac(self, template_path: Path, output_path: Path) -> bool:
        """
        Call Fortran tide_fac executable for accurate nodal corrections.

        Searches for the executable under multiple naming conventions:
        - COMF SCHISM: nos_ofs_create_tide_fac_schism (in EXECnos or EXECofs)
        - STOFS: stofs_3d_atl_tide_fac (in EXECstofs3d)

        The executable reads bctides.in_template and writes bctides.in with
        accurate nodal factors for the specified start time.
        """
        exe_names = ["nos_ofs_create_tide_fac_schism", "stofs_3d_atl_tide_fac"]
        env_dirs = ["EXECnos", "EXECofs", "EXECstofs3d"]

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
            log.debug("No Fortran tide_fac executable found in EXECnos/EXECofs/EXECstofs3d")
            return False

        start_time = self._compute_start_time()
        run_days = (self.config.nowcast_hours + self.config.forecast_hours) / 24.0

        try:
            work_template = output_path.parent / "bctides.in_template"
            shutil.copy2(template_path, work_template)

            # Fortran input: N_days, hh,dd,mm,yyyy, y (confirmation)
            input_text = (
                f"{int(run_days)}\n"
                f"{start_time.strftime('%H,%d,%m,%Y')}\n"
                "y\n"
            )

            log.info(f"Running Fortran tide_fac: {exe}")
            log.info(f"  phase={self.phase}, start_time={start_time}, run_days={int(run_days)}")

            result = subprocess.run(
                [str(exe)],
                input=input_text,
                cwd=str(output_path.parent),
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode != 0:
                log.warning(f"tide_fac returned {result.returncode}: {result.stderr[:200]}")
                return False

            if output_path.exists():
                log.info("Created bctides.in using Fortran tide_fac (accurate nodal corrections)")
                return True

            log.warning("tide_fac completed but bctides.in not found")
            return False

        except subprocess.TimeoutExpired:
            log.warning("Fortran tide_fac timed out")
            return False
        except Exception as e:
            log.warning(f"Error calling Fortran tide_fac: {e}")
            return False

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
                    if len(parts) >= 5:
                        # parts: species amplitude frequency nodefactor equil_arg
                        f_val = nodal[line]["f"]
                        v0_plus_u = nodal[line]["v0_plus_u"]
                        parts[3] = f"{f_val:.5f}"
                        parts[4] = f"{v0_plus_u:.5f}"
                    elif len(parts) >= 4:
                        # Shorter format without frequency column
                        f_val = nodal[line]["f"]
                        v0_plus_u = nodal[line]["v0_plus_u"]
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
                eq_arg = nodal[const_name]["v0_plus_u"]

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
    run_days: float = 0.25,
) -> Dict[str, Dict[str, float]]:
    """
    Compute tidal nodal factors and equilibrium arguments (V0+u).

    Exact port of the NOAA/SCHISM Fortran tide_fac program (Schureman 1958).
    Nodal factors computed at mid-run, equilibrium arguments at start.

    Args:
        start_time: Model start datetime
        constituents: List of constituent names
        run_days: Run length in days (nodal factors computed at midpoint)

    Returns:
        Dict mapping constituent name -> {"f": nodal_factor, "v0_plus_u": deg}
    """
    yr = float(start_time.year)
    month = float(start_time.month)
    day = float(start_time.day)
    bhr = start_time.hour + start_time.minute / 60.0 + start_time.second / 3600.0
    hrm = bhr + run_days * 24.0 / 2.0  # mid-run hour

    dayj = _dayjul(yr, month, day)

    # Nodal factors at mid-run
    f_all = _nfacs(yr, dayj, hrm)

    # Equilibrium arguments (V0+u) at start, u at mid-run
    eq_all = _gterms(yr, dayj, bhr, dayj, hrm)

    # Map constituent names to Fortran indices
    CNAME_MAP = {
        "M2": 0, "S2": 1, "N2": 2, "K1": 3, "M4": 4, "O1": 5,
        "M6": 6, "MK3": 7, "S4": 8, "MN4": 9, "NU2": 10, "S6": 11,
        "MU2": 12, "2N2": 13, "OO1": 14, "LAMBDA2": 15, "S1": 16,
        "M1": 17, "J1": 18, "MM": 19, "SSA": 20, "SA": 21,
        "MSF": 22, "MF": 23, "RHO1": 24, "Q1": 25, "T2": 26,
        "R2": 27, "2Q1": 28, "P1": 29, "2SM2": 30, "M3": 31,
        "L2": 32, "2MK3": 33, "K2": 34, "M8": 35, "MS4": 36,
    }

    result = {}
    for const in constituents:
        idx = CNAME_MAP.get(const.upper())
        if idx is not None:
            result[const] = {
                "f": f_all[idx],
                "v0_plus_u": eq_all[idx],
                # Keep separate u and v0 for backward compatibility
                "u": 0.0,
                "v0": eq_all[idx],
            }
        else:
            result[const] = {"f": 1.0, "v0_plus_u": 0.0, "u": 0.0, "v0": 0.0}
            log.debug(f"No tidal data for {const}, using f=1.0, V0+u=0.0")

    return result


def _angle(arg: float) -> float:
    """Place angle in 0-360 degrees."""
    result = arg % 360.0
    if result < 0:
        result += 360.0
    return result


def _dayjul(yr: float, xmonth: float, day: float) -> float:
    """Compute Julian day number within year."""
    dayt = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    dinc = 1.0 if ((yr - 1900) % 4 == 0) else 0.0
    days = list(dayt)
    for i in range(2, 12):
        days[i] = dayt[i] + dinc
    return days[int(xmonth) - 1] + day


def _orbit(yr: float, dayj: float, hr: float) -> dict:
    """Compute primary orbital elements (exact port of Fortran orbit())."""
    pi180 = math.pi / 180.0
    x = math.floor((yr - 1901.0) / 4.0)
    dyr = yr - 1900.0
    dday = dayj + x - 1.0

    # Moon's node (N)
    dn = _angle(259.1560564 - 19.328185764 * dyr - 0.0529539336 * dday - 0.0022064139 * hr)
    n = dn * pi180

    # Lunar perigee (p)
    dp = _angle(334.3837214 + 40.66246584 * dyr + 0.111404016 * dday + 0.004641834 * hr)
    p = dp * pi180

    # Inclination, nu, xi
    i = math.acos(0.9136949 - 0.0356926 * math.cos(n))
    di = _angle(i / pi180)
    nu = math.asin(0.0897056 * math.sin(n) / math.sin(i))
    dnu = nu / pi180
    xi = n - 2.0 * math.atan(0.64412 * math.tan(n / 2.0)) - nu
    dxi = xi / pi180
    dpc = _angle(dp - dxi)

    # Sun mean longitude (h)
    dh = _angle(280.1895014 - 0.238724988 * dyr + 0.9856473288 * dday + 0.0410686387 * hr)

    # Solar perigee (p1)
    dp1 = _angle(281.2208569 + 0.01717836 * dyr + 0.000047064 * dday + 0.000001961 * hr)

    # Moon mean longitude (s)
    ds = _angle(277.0256206 + 129.38482032 * dyr + 13.176396768 * dday + 0.549016532 * hr)

    nup = math.atan(math.sin(nu) / (math.cos(nu) + 0.334766 / math.sin(2.0 * i)))
    dnup = nup / pi180
    nup2 = math.atan(math.sin(2.0 * nu) / (math.cos(2.0 * nu) + 0.0726184 / math.sin(i)**2)) / 2.0
    dnup2 = nup2 / pi180

    return {
        "ds": ds, "dp": dp, "dh": dh, "dp1": dp1, "dn": dn,
        "di": di, "dnu": dnu, "dxi": dxi, "dnup": dnup, "dnup2": dnup2, "dpc": dpc,
    }


def _nfacs(yr: float, dayj: float, hr: float) -> List[float]:
    """Compute nodal factors for 37 constituents (exact port of Fortran nfacs())."""
    pi180 = math.pi / 180.0
    orb = _orbit(yr, dayj, hr)

    i = orb["di"] * pi180
    nu = orb["dnu"] * pi180
    xi = orb["dxi"] * pi180
    pc = orb["dpc"] * pi180

    sini = math.sin(i)
    sini2 = math.sin(i / 2.0)
    sin2i = math.sin(2.0 * i)
    cosi2 = math.cos(i / 2.0)
    tani2 = math.tan(i / 2.0)

    # Schureman equations
    qainv = math.sqrt(2.310 + 1.435 * math.cos(2.0 * pc))
    rainv = math.sqrt(1.0 - 12.0 * tani2**2 * math.cos(2.0 * pc) + 36.0 * tani2**4)

    eq73 = (2.0 / 3.0 - sini**2) / 0.5021
    eq74 = sini**2 / 0.1578
    eq75 = sini * cosi2**2 / 0.37988
    eq76 = sin2i / 0.7214
    eq77 = sini * sini2**2 / 0.0164
    eq78 = cosi2**4 / 0.91544
    eq149 = cosi2**6 / 0.8758
    eq227 = math.sqrt(0.8965 * sin2i**2 + 0.6001 * sin2i * math.cos(nu) + 0.1006)
    eq235 = 0.001 + math.sqrt(19.0444 * sini**4 + 2.7702 * sini**2 * math.cos(2.0 * nu) + 0.0981)

    f = [0.0] * 37
    f[0] = eq78       # M2
    f[1] = 1.0        # S2
    f[2] = eq78       # N2
    f[3] = eq227      # K1
    f[4] = f[0]**2    # M4
    f[5] = eq75       # O1
    f[6] = f[0]**3    # M6
    f[7] = f[0] * f[3]  # MK3
    f[8] = 1.0        # S4
    f[9] = f[0]**2    # MN4
    f[10] = eq78      # nu2
    f[11] = 1.0       # S6
    f[12] = eq78      # mu2
    f[13] = eq78      # 2N2
    f[14] = eq77      # OO1
    f[15] = eq78      # lambda2
    f[16] = 1.0       # S1
    f[17] = 0.0       # M1 (set to 0 per Fortran comment)
    f[18] = eq76      # J1
    f[19] = eq73      # Mm
    f[20] = 1.0       # Ssa
    f[21] = 1.0       # Sa
    f[22] = eq78      # MSf
    f[23] = eq74      # Mf
    f[24] = eq75      # rho1
    f[25] = eq75      # Q1
    f[26] = 1.0       # T2
    f[27] = 1.0       # R2
    f[28] = eq75      # 2Q1
    f[29] = 1.0       # P1
    f[30] = eq78      # 2SM2
    f[31] = eq149     # M3
    f[32] = 0.0       # L2 (set to 0 per Fortran comment)
    f[33] = f[0]**2 * f[3]  # 2MK3
    f[34] = eq235     # K2
    f[35] = f[0]**4   # M8
    f[36] = eq78      # MS4
    return f


def _gterms(yr: float, dayj: float, hr: float, daym: float, hrm: float) -> List[float]:
    """Compute equilibrium arguments V0+u for 37 constituents (exact Fortran port)."""
    pi180 = math.pi / 180.0

    # Orbital values at START for V0
    orb_start = _orbit(yr, dayj, hr)
    s = orb_start["ds"]
    p = orb_start["dp"]
    h = orb_start["dh"]
    p1 = orb_start["dp1"]
    t = _angle(180.0 + hr * (360.0 / 24.0))

    # Orbital values at MID-RUN for u
    orb_mid = _orbit(yr, daym, hrm)
    nu = orb_mid["dnu"]
    xi = orb_mid["dxi"]
    nup = orb_mid["dnup"]
    nup2 = orb_mid["dnup2"]

    eq = [0.0] * 37
    eq[0] = 2.0 * (t - s + h) + 2.0 * (xi - nu)                    # M2
    eq[1] = 2.0 * t                                                   # S2
    eq[2] = 2.0 * (t + h) - 3.0 * s + p + 2.0 * (xi - nu)          # N2
    eq[3] = t + h - 90.0 - nup                                       # K1
    eq[4] = 4.0 * (t - s + h) + 4.0 * (xi - nu)                    # M4
    eq[5] = t - 2.0 * s + h + 90.0 + 2.0 * xi - nu                 # O1
    eq[6] = 6.0 * (t - s + h) + 6.0 * (xi - nu)                    # M6
    eq[7] = 3.0 * (t + h) - 2.0 * s - 90.0 + 2.0 * (xi - nu) - nup  # MK3
    eq[8] = 4.0 * t                                                   # S4
    eq[9] = 4.0 * (t + h) - 5.0 * s + p + 4.0 * (xi - nu)          # MN4
    eq[10] = 2.0 * t - 3.0 * s + 4.0 * h - p + 2.0 * (xi - nu)    # nu2
    eq[11] = 6.0 * t                                                  # S6
    eq[12] = 2.0 * (t + 2.0 * (h - s)) + 2.0 * (xi - nu)           # mu2
    eq[13] = 2.0 * (t - 2.0 * s + h + p) + 2.0 * (xi - nu)         # 2N2
    eq[14] = t + 2.0 * s + h - 90.0 - 2.0 * xi - nu                # OO1
    eq[15] = 2.0 * t - s + p + 180.0 + 2.0 * (xi - nu)             # lambda2
    eq[16] = t                                                         # S1
    eq[17] = t - s + h - 90.0 + xi - nu                              # M1 (simplified)
    eq[18] = t + s + h - p - 90.0 - nu                               # J1
    eq[19] = s - p                                                     # Mm
    eq[20] = 2.0 * h                                                   # Ssa
    eq[21] = h                                                         # Sa
    eq[22] = 2.0 * (s - h)                                            # MSf
    eq[23] = 2.0 * s - 2.0 * xi                                      # Mf
    eq[24] = t + 3.0 * (h - s) - p + 90.0 + 2.0 * xi - nu          # rho1
    eq[25] = t - 3.0 * s + h + p + 90.0 + 2.0 * xi - nu            # Q1
    eq[26] = 2.0 * t - h + p1                                        # T2
    eq[27] = 2.0 * t + h - p1 + 180.0                               # R2
    eq[28] = t - 4.0 * s + h + 2.0 * p + 90.0 + 2.0 * xi - nu     # 2Q1
    eq[29] = t - h + 90.0                                             # P1
    eq[30] = 2.0 * (t + s - h) + 2.0 * (nu - xi)                   # 2SM2
    eq[31] = 3.0 * (t - s + h) + 3.0 * (xi - nu)                   # M3
    # L2: uses r term
    i_rad = orb_mid["di"] * pi180
    pc_rad = orb_mid["dpc"] * pi180
    r = math.sin(2.0 * pc_rad) / ((1.0 / 6.0) * (1.0 / math.tan(0.5 * i_rad))**2 - math.cos(2.0 * pc_rad))
    r = math.atan(r) / pi180
    eq[32] = 2.0 * (t + h) - s - p + 180.0 + 2.0 * (xi - nu) - r  # L2
    eq[33] = 3.0 * (t + h) - 4.0 * s + 90.0 + 4.0 * (xi - nu) + nup  # 2MK3
    eq[34] = 2.0 * (t + h) - 2.0 * nup2                             # K2
    eq[35] = 8.0 * (t - s + h) + 8.0 * (xi - nu)                   # M8
    eq[36] = 2.0 * (2.0 * t - s + h) + 2.0 * (xi - nu)             # MS4

    return [_angle(e) for e in eq]


def _nodal_fu(constituent: str, N: float, p: float) -> Tuple[float, float]:
    """Legacy simplified nodal corrections — kept for backward compatibility."""
    cosN = math.cos(N)
    sinN = math.sin(N)

    if constituent in ("M2", "N2"):
        f = 1.0 - 0.037 * cosN
        u = -2.1 * sinN
    elif constituent == "S2":
        f = 1.0
        u = 0.0
    elif constituent == "K2":
        f = 1.024 + 0.286 * cosN
        u = -17.7 * sinN
    elif constituent == "K1":
        f = 1.006 + 0.115 * cosN
        u = -8.9 * sinN
    elif constituent == "O1":
        f = 1.009 + 0.187 * cosN
        u = 10.8 * sinN
    elif constituent == "P1":
        f = 1.0
        u = 0.0
    elif constituent == "Q1":
        f = 1.009 + 0.187 * cosN
        u = 10.8 * sinN
    else:
        f = 1.0
        u = 0.0

    return f, u
