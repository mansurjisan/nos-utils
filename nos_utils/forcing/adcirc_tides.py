"""
ADCIRC tidal parameters (nodal factors, equilibrium arguments, constituent table).

Adapted from Zach Cobell's StofsWorkflow (oceanmodeling/nos-workflow, branch
zcobell/stofs_2d_global): models/adcirc/tidefac.py and models/adcirc/tide.py.
The TideFac numerics below are his Python port of Schureman (1958) as used by
stofs_2d_glo_tide_fac.f, copied unchanged except for Python 3.8 typing.

Public API::

    consts = compute_adcirc_tides(["M2", "K1"], start, run_days)

``start`` is the tidal reference time (ADCIRC cold-start time) and ``run_days``
the length from ``start`` to the end of the run. With the default
``nodal_reference="midrun"`` the nodal factors are evaluated at the middle of that
span and the equilibrium arguments at ``start``, as in his Tide class.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar, List

log = logging.getLogger(__name__)

@dataclass
class _OrbitalParameters:
    """Orbital parameters computed from date/time."""

    s: float  # Mean longitude of the moon (degrees)
    p: float  # Lunar perigee (degrees)
    h: float  # Mean longitude of the sun (degrees)
    p1: float  # Solar perigee (degrees)
    n: float  # Moon's node (degrees)
    i: float  # Inclination (degrees)
    nu: float  # Nu parameter (degrees)
    xi: float  # Xi parameter (degrees)
    nup: float  # Nup parameter (degrees)
    nup2: float  # Nup2 parameter (degrees)
    pc: float  # PC parameter (degrees)


@dataclass
class TidalConstituent:
    """Represents a tidal constituent with its computed parameters."""

    name: str
    node_factor: float
    equilibrium_argument: float
    speed: float  # degrees/hour
    amplitude: float  # equilibrium amplitude (meters)
    earth_tide_reduction_factor: float  # Love number reduction

    @property
    def frequency(self) -> float:
        """
        Return angular frequency in radians per second.

        Converts speed (deg/hr) to angular frequency (rad/s).
        """
        return self.speed * math.pi / 648000.0


class TideFac:
    """
    Computes nodal factors and equilibrium arguments for tidal constituents.

    This class implements the algorithms from Schureman (1958) for computing
    the time-varying corrections needed for tidal prediction.
    """

    # Constituent names (37 constituents)
    CONSTITUENT_NAMES: ClassVar[list[str]] = [
        "M2",
        "S2",
        "N2",
        "K1",
        "M4",
        "O1",
        "M6",
        "MK3",
        "S4",
        "MN4",
        "NU2",
        "S6",
        "MU2",
        "2N2",
        "OO1",
        "LAMBDA2",
        "S1",
        "M1",
        "J1",
        "MM",
        "SSA",
        "SA",
        "MSF",
        "MF",
        "RHO1",
        "Q1",
        "T2",
        "R2",
        "2Q1",
        "P1",
        "2SM2",
        "M3",
        "L2",
        "2MK3",
        "K2",
        "M8",
        "MS4",
    ]

    # Orbital speeds (degrees/hour) for each constituent
    SPEEDS: ClassVar[list[float]] = [
        28.9841042,
        30.0,
        28.4397295,
        15.0410686,
        57.9682084,
        13.9430356,
        86.9523127,
        44.0251729,
        60.0,
        57.4238337,
        28.5125831,
        90.0,
        27.9682084,
        27.8953548,
        16.1391017,
        29.4556253,
        15.0,
        14.4966939,
        15.5854433,
        0.5443747,
        0.0821373,
        0.0410686,
        1.0158958,
        1.0980331,
        13.4715145,
        13.3986609,
        29.9589333,
        30.0410667,
        12.8542862,
        14.9589314,
        31.0158958,
        43.4761563,
        29.5284789,
        42.9271398,
        30.0821373,
        115.9364169,
        58.9841042,
    ]

    # Number of tide cycles per day per constituent
    CYCLES_PER_DAY: ClassVar[list[float]] = [
        2.0,
        2.0,
        2.0,
        1.0,
        4.0,
        1.0,
        6.0,
        3.0,
        4.0,
        4.0,
        2.0,
        6.0,
        2.0,
        2.0,
        1.0,
        2.0,
        1.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        2.0,
        2.0,
        1.0,
        1.0,
        2.0,
        3.0,
        2.0,
        3.0,
        2.0,
        8.0,
        4.0,
    ]

    # Equilibrium amplitudes (meters) for each constituent
    # These are derived from Doodson's tidal potential coefficients
    # Values from standard tidal theory (Schureman, Cartwright & Tayler, etc.)
    AMPLITUDES: ClassVar[list[float]] = [
        0.242334,  # M2
        0.112841,  # S2
        0.046398,  # N2
        0.141565,  # K1
        0.0,  # M4 (shallow water, no direct potential)
        0.100514,  # O1
        0.0,  # M6 (shallow water)
        0.0,  # MK3 (shallow water)
        0.0,  # S4 (shallow water)
        0.0,  # MN4 (shallow water)
        0.008794,  # NU2
        0.0,  # S6 (shallow water)
        0.007408,  # MU2
        0.006112,  # 2N2
        0.004239,  # OO1
        0.006614,  # LAMBDA2
        0.004200,  # S1
        0.008200,  # M1
        0.007899,  # J1
        0.022026,  # MM
        0.019446,  # SSA
        0.004945,  # SA (meteorological, varies)
        0.0,  # MSF
        0.041742,  # MF
        0.006537,  # RHO1
        0.019256,  # Q1
        0.006614,  # T2
        0.000938,  # R2
        0.002567,  # 2Q1
        0.046843,  # P1
        0.0,  # 2SM2
        0.0,  # M3
        0.006694,  # L2
        0.0,  # 2MK3 (shallow water)
        0.030704,  # K2
        0.0,  # M8 (shallow water)
        0.0,  # MS4 (shallow water)
    ]

    # Earth tide reduction factors (Love number corrections)
    # These account for the elastic response of the solid Earth
    EARTH_TIDE_REDUCTION_FACTORS: ClassVar[list[float]] = [
        0.693,  # M2
        0.693,  # S2
        0.693,  # N2
        0.736,  # K1
        0.693,  # M4
        0.695,  # O1
        0.693,  # M6
        0.693,  # MK3
        0.693,  # S4
        0.693,  # MN4
        0.693,  # NU2
        0.693,  # S6
        0.693,  # MU2
        0.693,  # 2N2
        0.693,  # OO1
        0.693,  # LAMBDA2
        0.693,  # S1
        0.693,  # M1
        0.693,  # J1
        0.693,  # MM
        0.693,  # SSA
        0.693,  # SA
        0.693,  # MSF
        0.693,  # MF
        0.695,  # RHO1
        0.695,  # Q1
        0.693,  # T2
        0.693,  # R2
        0.695,  # 2Q1
        0.706,  # P1
        0.693,  # 2SM2
        0.693,  # M3
        0.693,  # L2
        0.693,  # 2MK3
        0.693,  # K2
        0.693,  # M8
        0.693,  # MS4
    ]

    # Default constituents used in STOFS-2D-GLO
    # Mapping: constituent name -> 0-based index
    DEFAULT_CONSTITUENTS: ClassVar[dict[str, int]] = {
        "K1": 3,
        "O1": 5,
        "P1": 29,
        "Q1": 25,
        "M2": 0,
        "S2": 1,
        "N2": 2,
        "K2": 34,
        "MF": 23,
        "MM": 19,
        "M4": 4,
        "MS4": 36,
        "MN4": 9,
        "SA": 21,
        "SSA": 20,
    }

    def __init__(
        self,
        start_date: datetime,
        run_length_days: float,
        constituents: list[str] | None = None,
    ) -> None:
        """
        Initialize the TideFac calculator.

        Args:
            start_date: Start date and time of the simulation
            run_length_days: Length of the simulation in days
            constituents: List of constituent names to compute. If None,
                         uses the default STOFS-2D-GLO constituents.
        """
        self._start_date = start_date
        self._run_length_days = run_length_days
        self._constituents = (
            constituents if constituents else list(self.DEFAULT_CONSTITUENTS.keys())
        )
        self._results: dict[str, TidalConstituent] = {}
        self._compute()

    @staticmethod
    def _compute_orbital_parameters(
        year: int, day_of_year: float, hour: float
    ) -> _OrbitalParameters:
        """
        Compute primary and secondary orbital functions.

        The equations derive from NOAA code. Tabular values of the orbital
        functions can be found in Table 1 of Schureman.

        Args:
            year: Year (e.g., 2024)
            day_of_year: Day of year
            hour: Hour of day (can be fractional)

        Returns:
            _OrbitalParameters containing all orbital values
        """
        x = int((year - 1901) / 4)
        dyr = year - 1900
        dday = day_of_year + x - 1

        # DN is the Moon's node (capital N, Table 1, Schureman)
        dn = (
            259.1560564 - 19.328185764 * dyr - 0.0529539336 * dday - 0.0022064139 * hour
        ) % 360.0
        n_rad = math.radians(dn)

        # DP is the lunar perigee (small p, Table 1)
        dp = (
            334.3837214 + 40.66246584 * dyr + 0.111404016 * dday + 0.004641834 * hour
        ) % 360.0

        # Compute inclination
        i_rad = math.acos(0.9136949 - 0.0356926 * math.cos(n_rad))
        di = math.degrees(i_rad) % 360.0

        # Compute nu
        nu_rad = math.asin(0.0897056 * math.sin(n_rad) / math.sin(i_rad))
        dnu = math.degrees(nu_rad)

        # Compute xi
        xi_rad = n_rad - 2.0 * math.atan(0.64412 * math.tan(n_rad / 2.0)) - nu_rad
        dxi = math.degrees(xi_rad)

        # PC = DP - XI
        dpc = (dp - dxi) % 360.0

        # DH is the mean longitude of the sun (small h, Table 1)
        dh = (
            280.1895014 - 0.238724988 * dyr + 0.9856473288 * dday + 0.0410686387 * hour
        ) % 360.0

        # DP1 is the solar perigee (small p1, Table 1)
        dp1 = (
            281.2208569 + 0.01717836 * dyr + 0.000047064 * dday + 0.000001961 * hour
        ) % 360.0

        # DS is the mean longitude of the moon (small s, Table 1)
        ds = (
            277.0256206 + 129.38482032 * dyr + 13.176396768 * dday + 0.549016532 * hour
        ) % 360.0

        # Compute nup
        nup_rad = math.atan(
            math.sin(nu_rad) / (math.cos(nu_rad) + 0.334766 / math.sin(2.0 * i_rad))
        )
        dnup = math.degrees(nup_rad)

        # Compute nup2
        nup2_rad = (
            math.atan(
                math.sin(2.0 * nu_rad)
                / (math.cos(2.0 * nu_rad) + 0.0726184 / math.sin(i_rad) ** 2)
            )
            / 2.0
        )
        dnup2 = math.degrees(nup2_rad)

        return _OrbitalParameters(
            s=ds,
            p=dp,
            h=dh,
            p1=dp1,
            n=dn,
            i=di,
            nu=dnu,
            xi=dxi,
            nup=dnup,
            nup2=dnup2,
            pc=dpc,
        )

    @staticmethod
    def _compute_node_factors(orbit: _OrbitalParameters) -> list[float]:  # noqa: PLR0915
        """
        Calculate node factors for all 37 constituent tidal signals.

        The equations come from Schureman (1958).

        Args:
            orbit: Orbital parameters at middle of record

        Returns:
            List of 37 node factors
        """
        i = math.radians(orbit.i)
        nu = math.radians(orbit.nu)

        sin_i = math.sin(i)
        sin_i2 = math.sin(i / 2.0)
        sin_2i = math.sin(2.0 * i)
        cos_i2 = math.cos(i / 2.0)

        # Variable names refer to equation numbers in Schureman
        eq73 = (2.0 / 3.0 - sin_i**2) / 0.5021
        eq74 = sin_i**2 / 0.1578
        eq75 = sin_i * cos_i2**2 / 0.37988
        eq76 = sin_2i / 0.7214
        eq77 = sin_i * sin_i2**2 / 0.0164
        eq78 = cos_i2**4 / 0.91544
        eq149 = cos_i2**6 / 0.8758

        # Note: eq207 and eq215 are computed in Fortran but produce incorrect
        # results for M1 and L2 respectively. Those constituents have node
        # factors set to 0 per the original code. The equations are:
        # pc = orbit.pc * pi180; tan_i2 = math.tan(i / 2.0)
        # qainv = math.sqrt(2.310 + 1.435 * math.cos(2.0 * pc))  # Eq 197
        # rainv = sqrt(1.0 - 12.0*tan_i2**2*cos(2.0*pc) + 36.0*tan_i2**4)  # Eq 213
        # eq207 = eq75 * qainv; eq215 = eq78 * rainv

        eq227 = math.sqrt(0.8965 * sin_2i**2 + 0.6001 * sin_2i * math.cos(nu) + 0.1006)
        eq235 = 0.001 + math.sqrt(
            19.0444 * sin_i**4 + 2.7702 * sin_i**2 * math.cos(2.0 * nu) + 0.0981
        )

        # Node factors for 37 constituents (0-indexed)
        fndcst: list[float] = [0.0] * 37

        fndcst[0] = eq78  # M2
        fndcst[1] = 1.0  # S2
        fndcst[2] = eq78  # N2
        fndcst[3] = eq227  # K1
        fndcst[4] = fndcst[0] ** 2  # M4
        fndcst[5] = eq75  # O1
        fndcst[6] = fndcst[0] ** 3  # M6
        fndcst[7] = fndcst[0] * fndcst[3]  # MK3
        fndcst[8] = 1.0  # S4
        fndcst[9] = fndcst[0] ** 2  # MN4
        fndcst[10] = eq78  # NU2
        fndcst[11] = 1.0  # S6
        fndcst[12] = eq78  # MU2
        fndcst[13] = eq78  # 2N2
        fndcst[14] = eq77  # OO1
        fndcst[15] = eq78  # LAMBDA2
        fndcst[16] = 1.0  # S1
        # Equation 207 not producing correct answer for M1
        # Set node factor for M1 = 0 until further research
        fndcst[17] = 0.0  # M1
        fndcst[18] = eq76  # J1
        fndcst[19] = eq73  # MM
        fndcst[20] = 1.0  # SSA
        fndcst[21] = 1.0  # SA
        fndcst[22] = eq78  # MSF
        fndcst[23] = eq74  # MF
        fndcst[24] = eq75  # RHO1
        fndcst[25] = eq75  # Q1
        fndcst[26] = 1.0  # T2
        fndcst[27] = 1.0  # R2
        fndcst[28] = eq75  # 2Q1
        fndcst[29] = 1.0  # P1
        fndcst[30] = eq78  # 2SM2
        fndcst[31] = eq149  # M3
        # Equation 215 not producing correct answer for L2
        # Set node factor for L2 = 0 until further research
        fndcst[32] = 0.0  # L2
        fndcst[33] = fndcst[0] ** 2 * fndcst[3]  # 2MK3
        fndcst[34] = eq235  # K2
        fndcst[35] = fndcst[0] ** 4  # M8
        fndcst[36] = eq78  # MS4

        return fndcst

    @staticmethod
    def _compute_equilibrium_arguments(  # noqa: PLR0915
        orbit_start: _OrbitalParameters,
        orbit_mid: _OrbitalParameters,
        hour: float,
    ) -> list[float]:
        """
        Calculate equilibrium arguments (V0+U) for all 37 constituent tides.

        Uses orbital values at beginning of series for V0 and at middle of
        series for U, following Schureman (1958).

        Args:
            orbit_start: Orbital parameters at beginning of record
            orbit_mid: Orbital parameters at middle of record
            hour: Starting hour

        Returns:
            List of 37 equilibrium arguments in degrees
        """
        # Values at beginning of series for V0
        s = orbit_start.s
        p = orbit_start.p
        h = orbit_start.h
        p1 = orbit_start.p1
        t = (180.0 + hour * 15.0) % 360.0  # 360/24 = 15 deg/hr

        # Values at middle of series for U
        nu = orbit_mid.nu
        xi = orbit_mid.xi
        nup = orbit_mid.nup
        nup2 = orbit_mid.nup2
        i_mid = math.radians(orbit_mid.i)
        pc_mid = math.radians(orbit_mid.pc)

        # Equilibrium arguments for 37 constituents
        eqcst: list[float] = [0.0] * 37

        eqcst[0] = 2.0 * (t - s + h) + 2.0 * (xi - nu)  # M2
        eqcst[1] = 2.0 * t  # S2
        eqcst[2] = 2.0 * (t + h) - 3.0 * s + p + 2.0 * (xi - nu)  # N2
        eqcst[3] = t + h - 90.0 - nup  # K1
        eqcst[4] = 4.0 * (t - s + h) + 4.0 * (xi - nu)  # M4
        eqcst[5] = t - 2.0 * s + h + 90.0 + 2.0 * xi - nu  # O1
        eqcst[6] = 6.0 * (t - s + h) + 6.0 * (xi - nu)  # M6
        eqcst[7] = 3.0 * (t + h) - 2.0 * s - 90.0 + 2.0 * (xi - nu) - nup  # MK3
        eqcst[8] = 4.0 * t  # S4
        eqcst[9] = 4.0 * (t + h) - 5.0 * s + p + 4.0 * (xi - nu)  # MN4
        eqcst[10] = 2.0 * t - 3.0 * s + 4.0 * h - p + 2.0 * (xi - nu)  # NU2
        eqcst[11] = 6.0 * t  # S6
        eqcst[12] = 2.0 * (t + 2.0 * (h - s)) + 2.0 * (xi - nu)  # MU2
        eqcst[13] = 2.0 * (t - 2.0 * s + h + p) + 2.0 * (xi - nu)  # 2N2
        eqcst[14] = t + 2.0 * s + h - 90.0 - 2.0 * xi - nu  # OO1
        eqcst[15] = 2.0 * t - s + p + 180.0 + 2.0 * (xi - nu)  # LAMBDA2
        eqcst[16] = t  # S1

        # M1 - requires special calculation
        q = math.degrees(
            math.atan2(
                (5.0 * math.cos(i_mid) - 1.0) * math.sin(pc_mid),
                (7.0 * math.cos(i_mid) + 1.0) * math.cos(pc_mid),
            )
        )
        if q < 0.0:
            q += 360.0
        eqcst[17] = t - s + h - 90.0 + xi - nu + q  # M1

        eqcst[18] = t + s + h - p - 90.0 - nu  # J1
        eqcst[19] = s - p  # MM
        eqcst[20] = 2.0 * h  # SSA
        eqcst[21] = h  # SA
        eqcst[22] = 2.0 * (s - h)  # MSF
        eqcst[23] = 2.0 * s - 2.0 * xi  # MF
        eqcst[24] = t + 3.0 * (h - s) - p + 90.0 + 2.0 * xi - nu  # RHO1
        eqcst[25] = t - 3.0 * s + h + p + 90.0 + 2.0 * xi - nu  # Q1
        eqcst[26] = 2.0 * t - h + p1  # T2
        eqcst[27] = 2.0 * t + h - p1 + 180.0  # R2
        eqcst[28] = t - 4.0 * s + h + 2.0 * p + 90.0 + 2.0 * xi - nu  # 2Q1
        eqcst[29] = t - h + 90.0  # P1
        eqcst[30] = 2.0 * (t + s - h) + 2.0 * (nu - xi)  # 2SM2
        eqcst[31] = 3.0 * (t - s + h) + 3.0 * (xi - nu)  # M3

        # L2 - requires special calculation
        r = math.sin(2.0 * pc_mid) / (
            (1.0 / 6.0) * (1.0 / math.tan(0.5 * i_mid)) ** 2 - math.cos(2.0 * pc_mid)
        )
        r = math.degrees(math.atan(r))
        eqcst[32] = 2.0 * (t + h) - s - p + 180.0 + 2.0 * (xi - nu) - r  # L2

        eqcst[33] = 3.0 * (t + h) - 4.0 * s + 90.0 + 4.0 * (xi - nu) + nup  # 2MK3
        eqcst[34] = 2.0 * (t + h) - 2.0 * nup2  # K2
        eqcst[35] = 8.0 * (t - s + h) + 8.0 * (xi - nu)  # M8
        eqcst[36] = 2.0 * (2.0 * t - s + h) + 2.0 * (xi - nu)  # MS4

        # Normalize all arguments to 0-360 range
        return [arg % 360.0 for arg in eqcst]

    def _compute(self) -> None:
        """Compute nodal factors and equilibrium arguments for requested constituents."""
        year = self._start_date.year
        hour = (
            self._start_date.hour
            + self._start_date.minute / 60.0
            + self._start_date.second / 3600.0
        )

        run_hours = self._run_length_days * 24.0
        hour_mid = hour + run_hours / 2.0

        day_of_year = float(self._start_date.timetuple().tm_yday)

        # Compute orbital parameters at beginning and middle of record
        orbit_start = self._compute_orbital_parameters(year, day_of_year, hour)

        # Adjust for middle of run - handle day rollover
        mid_day_of_year = day_of_year
        mid_hour = hour_mid
        while mid_hour >= 24.0:
            mid_hour -= 24.0
            mid_day_of_year += 1.0

        orbit_mid = self._compute_orbital_parameters(year, mid_day_of_year, mid_hour)

        # Compute node factors at middle of record
        node_factors = self._compute_node_factors(orbit_mid)

        # Compute equilibrium arguments
        eq_args = self._compute_equilibrium_arguments(orbit_start, orbit_mid, hour)

        # Store results for requested constituents
        for name in self._constituents:
            name_upper = name.upper()
            if name_upper in self.CONSTITUENT_NAMES:
                idx = self.CONSTITUENT_NAMES.index(name_upper)
                self._results[name_upper] = TidalConstituent(
                    name=name_upper,
                    node_factor=node_factors[idx],
                    equilibrium_argument=eq_args[idx],
                    speed=self.SPEEDS[idx],
                    amplitude=self.AMPLITUDES[idx],
                    earth_tide_reduction_factor=self.EARTH_TIDE_REDUCTION_FACTORS[idx],
                )

    def get_constituent(self, name: str) -> TidalConstituent | None:
        """
        Get the computed parameters for a specific constituent.

        Args:
            name: Name of the constituent (case-insensitive)

        Returns:
            TidalConstituent with computed parameters, or None if not found
        """
        return self._results.get(name.upper())

    def get_all_constituents(self) -> dict[str, TidalConstituent]:
        """
        Get all computed constituent parameters.

        Returns:
            Dictionary mapping constituent names to TidalConstituent objects
        """
        return self._results.copy()

    @property
    def start_date(self) -> datetime:
        """Return the start date of the simulation."""
        return self._start_date

    @property
    def run_length_days(self) -> float:
        """Return the simulation length in days."""
        return self._run_length_days

    @classmethod
    def available_constituents(cls) -> list[str]:
        """
        Return list of all available constituent names.

        Returns:
            List of 37 constituent names
        """
        return cls.CONSTITUENT_NAMES.copy()

    @classmethod
    def default_constituents(cls) -> list[str]:
        """
        Return list of default STOFS-2D-GLO constituent names.

        Returns:
            List of 15 default constituent names
        """
        return list(cls.DEFAULT_CONSTITUENTS.keys())


NODAL_REFERENCES = ("midrun", "start")


@dataclass
class AdcircConstituent:
    """One constituent as the ADCIRC fort.15 tide blocks need it.

    ``tpk`` is the equilibrium tidal potential amplitude (m), ``etrf`` the earth
    tide reduction factor, ``frequency`` rad/s, ``equilibrium_arg_deg`` degrees.
    """

    name: str
    frequency: float
    etrf: float
    tpk: float
    nodal_factor: float
    equilibrium_arg_deg: float
    speed_deg_per_hr: float = 0.0

    def potential_lines(self) -> List[str]:
        """Tidal potential entry (NTIF block): name line, then amp/freq/etrf/nf/eq-arg."""
        return [
            self.name,
            "{:0.5f} {:0.15f} {:0.3f} {:0.5f} {:0.2f}".format(
                self.tpk, self.frequency, self.etrf, self.nodal_factor,
                self.equilibrium_arg_deg),
        ]

    def boundary_freq_lines(self) -> List[str]:
        """Boundary forcing frequency entry (NBFR block): name, freq/nf/eq-arg."""
        return [
            self.name,
            "{:0.15f} {:0.5f} {:0.2f}".format(
                self.frequency, self.nodal_factor, self.equilibrium_arg_deg),
        ]


def available_adcirc_constituents() -> List[str]:
    return TideFac.available_constituents()


def default_adcirc_constituents() -> List[str]:
    return TideFac.default_constituents()


def compute_adcirc_tides(
    constituents: List[str],
    start: datetime,
    run_days: float,
    nodal_reference: str = "midrun",
) -> List[AdcircConstituent]:
    """Compute ADCIRC tidal parameters for ``constituents``.

    Unknown names are skipped with a warning, as in his Tide class; duplicates keep
    the first position. ``nodal_reference="start"`` evaluates the nodal factors at
    ``start`` (the same as ``run_days=0``). MJ (10/05/26)
    """
    if nodal_reference not in NODAL_REFERENCES:
        raise ValueError(
            f"nodal_reference must be one of {NODAL_REFERENCES}, got {nodal_reference!r}")
    length = 0.0 if nodal_reference == "start" else float(run_days)
    fac = TideFac(start_date=start, run_length_days=length,
                  constituents=list(constituents))
    out = {}  # type: dict
    for name in constituents:
        c = fac.get_constituent(name)
        if c is None:
            log.warning("Unknown constituent %s will be ignored", name)
            continue
        out[c.name] = AdcircConstituent(
            name=c.name,
            frequency=c.frequency,
            etrf=c.earth_tide_reduction_factor,
            tpk=c.amplitude,
            nodal_factor=c.node_factor,
            equilibrium_arg_deg=c.equilibrium_argument,
            speed_deg_per_hr=c.speed,
        )
    return list(out.values())
