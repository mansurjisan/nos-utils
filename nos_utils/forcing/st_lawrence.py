"""
St. Lawrence River forcing processor (STOFS-3D-ATL).

Generates the two climatology-free river files for the St. Lawrence:

  - ``flux.th``  — daily discharge (m^3/s, negative = inflow) from Canadian
    hydrometric observations (``02OA016_hydrometric.csv`` under
    ``$COMINlaw/<yyyymmdd>/can_streamgauge/``).
  - ``TEM_1.th`` — daily water temperature derived from GFS air temperature
    at the river mouth via a linear regression ``T_water = 0.83 * T_air + 2.817``
    (negative values clamped to 0). When a GFS sflux radiation file is not
    available (e.g., legacy CSV-only runs), temperature observations from
    the same CSV are used.

The operational ex-script sources
``gen_fluxth_st_lawrence_riv.py`` and ``gen_temp_1_st_lawrence_riv.py`` with a
previous-day CSV fallback and a previous-cycle archive fallback (re-shifted in
time). This module ports both algorithms plus both fallbacks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from ..config import ForcingConfig
from .base import ForcingProcessor, ForcingResult

log = logging.getLogger(__name__)

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    from netCDF4 import Dataset
    HAS_NETCDF4 = True
except ImportError:
    HAS_NETCDF4 = False


# Environment and Climate Change Canada hydrometric CSV parameter codes.
# See https://wateroffice.ec.gc.ca for the full dictionary.
PARAM_TEMPERATURE = 5
PARAM_DISCHARGE = 47

# Water-temperature linear regression at the St. Lawrence mouth; derived
# operationally from multi-year GFS-rad-to-observed river-temperature fits.
AIR_TO_WATER_SLOPE = 0.83
AIR_TO_WATER_INTERCEPT = 2.817

# River mouth coordinates (lat, lon in degrees) used for air-temp sampling.
RIVER_MOUTH = (45.415, -73.623056)

# Operational filename default.
DEFAULT_CSV_NAME = "02OA016_hydrometric.csv"


@dataclass
class _StLawrenceSeries:
    """Daily flow + temperature series for a single river."""
    # Days 0..N-1 relative to nowcast start, where N = nowcast_days + forecast_days + 1
    seconds_from_start: List[int]
    flow_cms: List[float]
    temp_c: List[float]


class StLawrenceProcessor(ForcingProcessor):
    """Produce flux.th and TEM_1.th for the St. Lawrence River.

    The processor expects a Canadian hydrometric CSV at
    ``<input_path>/<pdy>/can_streamgauge/<csv_name>`` (mirroring
    ``$COMINlaw``). If that file is missing, it falls back to the previous
    day's directory. If both are missing, the caller can provide
    ``prev_rerun_dir`` to reuse yesterday's archived
    ``<run>.<cycle>.riv.obs.flux.th`` and ``...tem_1.th``.

    The GFS sflux radiation file (``stofs_3d_atl.tHHz.gfs.rad.nc``) is read
    for temperature if available. When it is not, temperature observations
    from the CSV (parameter code 5) are used; otherwise a constant -9999
    sentinel is written, matching the operational default.
    """

    SOURCE_NAME = "ST_LAWRENCE"

    def __init__(
        self,
        config: ForcingConfig,
        input_path: Path,
        output_path: Path,
        *,
        csv_name: str = DEFAULT_CSV_NAME,
        sflux_rad_file: Optional[Path] = None,
        prev_rerun_dir: Optional[Path] = None,
        archive_prefix: Optional[str] = None,
    ) -> None:
        super().__init__(config, input_path, output_path)
        self.csv_name = csv_name
        self.sflux_rad_file = Path(sflux_rad_file) if sflux_rad_file else None
        self.prev_rerun_dir = Path(prev_rerun_dir) if prev_rerun_dir else None
        # Archive prefix like "stofs_3d_atl.t12z" — determines the fallback
        # archive filenames (…riv.obs.flux.th / …riv.obs.tem_1.th).
        self.archive_prefix = archive_prefix

    # ------------------------------------------------------------------ API

    def process(self) -> ForcingResult:
        if not HAS_PANDAS:
            return ForcingResult(
                success=False, source=self.SOURCE_NAME,
                errors=["pandas is required for StLawrenceProcessor"],
            )

        self.create_output_dir()

        start = self._cycle_datetime()
        n_days_total = self._n_days_total()
        datevectors_hindcast = self._daily_range(start, days=1)
        datevectors_full = self._daily_range(start, days=n_days_total)

        warnings: List[str] = []
        output_files: List[Path] = []

        csv_path = self._find_csv(start)
        series: Optional[_StLawrenceSeries] = None

        if csv_path is not None:
            log.info(f"St. Lawrence CSV: {csv_path}")
            try:
                series = self._read_hydrometric_csv(
                    csv_path, datevectors_hindcast, datevectors_full
                )
            except Exception as exc:
                warnings.append(f"Failed to parse CSV {csv_path}: {exc}")
                series = None

        # If we have series data, overwrite temperature with the sflux-based
        # regression when a rad file is available (matches operational flow:
        # `rm -f TEM_1.th` between the flux script and the temp script).
        if series is not None and self.sflux_rad_file and self.sflux_rad_file.exists():
            try:
                temp_from_sflux = self._temp_from_sflux(
                    self.sflux_rad_file, datevectors_full
                )
                if temp_from_sflux is not None:
                    series.temp_c = temp_from_sflux
            except Exception as exc:
                warnings.append(f"Failed to read sflux rad {self.sflux_rad_file}: {exc}")

        if series is not None:
            flux_path = self._write_flux_th(series)
            temp_path = self._write_tem_1_th(series)
            if flux_path:
                output_files.append(flux_path)
            if temp_path:
                output_files.append(temp_path)

        # Fallback: previous cycle's archive if either file is missing.
        if len(output_files) < 2 and self.prev_rerun_dir is not None:
            missing = {"flux.th", "TEM_1.th"} - {p.name for p in output_files}
            for name in sorted(missing):
                archived = self._fallback_from_archive(name)
                if archived is not None:
                    output_files.append(archived)
                    warnings.append(
                        f"Using previous-cycle archive for {name}"
                    )

        if not output_files:
            return ForcingResult(
                success=False, source=self.SOURCE_NAME,
                errors=[
                    "No St. Lawrence data available: CSV missing and no "
                    "previous-cycle archive found",
                ],
                warnings=warnings,
            )

        return ForcingResult(
            success=len(output_files) >= 1,
            source=self.SOURCE_NAME,
            output_files=output_files,
            warnings=warnings,
            metadata={
                "csv_used": str(csv_path) if csv_path else None,
                "sflux_used": str(self.sflux_rad_file) if self.sflux_rad_file else None,
                "n_timesteps": len(series.seconds_from_start) if series else 0,
            },
        )

    def find_input_files(self) -> List[Path]:
        """CSV candidates in priority order (today, then yesterday)."""
        start = self._cycle_datetime()
        found: List[Path] = []
        for days_back in (0, 1):
            day = start - timedelta(days=days_back)
            p = self._csv_path_for(day)
            if p.exists():
                found.append(p)
        return found

    # -------------------------------------------------------------- CSV read

    def _read_hydrometric_csv(
        self,
        csv_path: Path,
        datevectors_hindcast,
        datevectors_full,
    ) -> _StLawrenceSeries:
        """Return a _StLawrenceSeries with daily flow & temperature.

        Mirrors ``gen_fluxth_st_lawrence_riv.py``:
          * parameter 5  -> water temperature (°C)
          * parameter 47 -> discharge (m^3/s)
        For each day in the hindcast window we look up the exact timestamp
        in the CSV. If missing:
          * day 0: raise (flow) or default to -9999 (temp)
          * day >0: carry forward the previous day's value.
        Days in the forecast window are padded with the last available value.
        """
        df = pd.read_csv(csv_path, sep=",", na_values="")
        # CSV columns: STATION, date_local, parameter, value, and several
        # QA columns we don't care about. Drop what we don't need, rename
        # the surviving columns to match the operational script.
        drop_cols = [df.columns[i] for i in (0, 4, 5, 6, 7, 8) if i < len(df.columns)]
        df = df.drop(columns=drop_cols)
        df = df.rename(
            columns={
                df.columns[0]: "date_local",
                df.columns[1]: "parameter",
                df.columns[2]: "value",
            }
        )
        df["date_utc"] = pd.to_datetime(df["date_local"])

        df_temp = df[df["parameter"] == PARAM_TEMPERATURE].copy().set_index("date_utc")
        df_flow = df[df["parameter"] == PARAM_DISCHARGE].copy().set_index("date_utc")

        data_flow: List[float] = []
        last_flow_idx = -1
        for i, dt in enumerate(datevectors_hindcast):
            try:
                value = float(df_flow.loc[dt]["value"])
                data_flow.append(round(value, 3))
                last_flow_idx = i
            except KeyError:
                if i == 0:
                    raise KeyError(
                        f"No discharge data for {dt} in {csv_path}; "
                        "fallback to archived CSV or previous-cycle rerun "
                        "should be used by the caller"
                    )
                data_flow.append(data_flow[-1])
                last_flow_idx = i

        # Pad forecast days with last valid value.
        tail_start = last_flow_idx + 1
        for _ in datevectors_full[tail_start:]:
            data_flow.append(data_flow[-1])

        data_temp: List[float] = []
        for i, dt in enumerate(datevectors_hindcast):
            try:
                value = float(df_temp.loc[dt]["value"])
                data_temp.append(round(value, 3))
            except KeyError:
                if i == 0:
                    log.info(
                        "No temperature observation for %s; "
                        "using sentinel -9999 (caller will overwrite with "
                        "sflux regression if available)", dt,
                    )
                    data_temp.append(-9999.0)
                else:
                    data_temp.append(data_temp[-1])

        tail_start = min(len(datevectors_hindcast), len(data_temp))
        for _ in datevectors_full[tail_start:]:
            data_temp.append(data_temp[-1])

        seconds_from_start = [
            int((dt - datevectors_hindcast[0]).total_seconds())
            for dt in datevectors_full
        ]

        # Sanity: all lists must be the same length.
        n = len(seconds_from_start)
        if len(data_flow) != n or len(data_temp) != n:
            raise ValueError(
                f"St. Lawrence series length mismatch: "
                f"seconds={n}, flow={len(data_flow)}, temp={len(data_temp)}"
            )

        return _StLawrenceSeries(
            seconds_from_start=seconds_from_start,
            flow_cms=data_flow,
            temp_c=data_temp,
        )

    # ----------------------------------------------------- sflux temperature

    def _temp_from_sflux(
        self,
        sflux_rad_file: Path,
        datevectors_full,
    ) -> Optional[List[float]]:
        """Derive daily river-mouth temperature from GFS air temp (sflux rad).

        Returns a list of daily temperatures aligned with *datevectors_full*
        (length = nowcast_days + forecast_days + 1), or None if the sflux
        data doesn't span the required window.
        """
        if not HAS_NETCDF4:
            log.warning("netCDF4 missing; cannot derive St. Lawrence temp from sflux")
            return None

        with Dataset(str(sflux_rad_file)) as ds:
            # sflux convention: lon is (y, x) with lon[0,:] giving the x-axis
            # and lat is (y, x) with lat[:,0] giving the y-axis.
            lon = ds["lon"][0, :]
            lat = ds["lat"][:, 0]
            stmp = ds["stmp"]

            # Find a 0.2°-wide box around the mouth (matches operational script).
            lat_idx_candidates = np.where(
                (lat - RIVER_MOUTH[0] > 0) & (lat - RIVER_MOUTH[0] < 0.2)
            )[0]
            lon_idx_candidates = np.where(
                (lon - RIVER_MOUTH[1] > 0) & (lon - RIVER_MOUTH[1] < 0.2)
            )[0]
            if lat_idx_candidates.size == 0 or lon_idx_candidates.size == 0:
                log.warning(
                    "St. Lawrence mouth not within sflux rad domain; "
                    "falling back to CSV temperature"
                )
                return None

            # sflux times are days since a reference timestamp embedded in
            # the ``units`` attribute.
            time_var = ds["time"]
            times_days = time_var[:]
            units = time_var.units
            if "since" not in units:
                log.warning(f"sflux time units missing 'since': {units}")
                return None
            ref_str = units.split("since", 1)[1].strip()
            # Tolerate trailing 'UTC' or timezone descriptors.
            ref_str = ref_str.split(" UTC")[0].split("+")[0].strip()
            try:
                ref_dt = datetime.strptime(ref_str, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                try:
                    ref_dt = datetime.strptime(ref_str, "%Y-%m-%d %H:%M")
                except ValueError:
                    ref_dt = datetime.strptime(ref_str, "%Y-%m-%d")

            # Extract air temp, convert K->°C, squeeze to 1D.
            t_kelvin = stmp[:, lat_idx_candidates, lon_idx_candidates]
            t_celsius = np.squeeze(np.asarray(t_kelvin) - 273.15)
            if t_celsius.ndim != 1:
                # If more than one grid cell landed in the box, average them.
                t_celsius = t_celsius.reshape(t_celsius.shape[0], -1).mean(axis=1)

        # Apply regression and clip negatives.
        water_t = AIR_TO_WATER_SLOPE * t_celsius + AIR_TO_WATER_INTERCEPT
        water_t = np.where(water_t < 0.0, 0.0, water_t)

        timestamps = [
            ref_dt + timedelta(seconds=int(round(dt * 86400.0)))
            for dt in times_days
        ]

        ref_tz = pd.Timestamp(ref_dt, tz="UTC")
        df = pd.DataFrame(
            water_t,
            index=[pd.Timestamp(ts, tz="UTC") for ts in timestamps],
        )
        df_hourly = df.resample("h").mean().bfill()

        daily: List[float] = []
        hourly_index = df_hourly.index
        for dt in datevectors_full:
            # datevectors_full are tz-aware UTC pandas Timestamps.
            if dt in hourly_index:
                daily.append(float(df_hourly.loc[dt, 0]))
            else:
                # Find nearest hour (within the sflux window).
                diffs = np.abs((hourly_index - dt).total_seconds())
                if diffs.size == 0 or diffs.min() > 3600 * 3:
                    log.warning(
                        "sflux does not cover %s for St. Lawrence temp", dt,
                    )
                    return None
                daily.append(float(df_hourly.iloc[int(np.argmin(diffs)), 0]))
        return daily

    # --------------------------------------------------------------- Writers

    def _write_flux_th(self, series: _StLawrenceSeries) -> Optional[Path]:
        output_file = self.output_path / "flux.th"
        try:
            data = np.array(
                [
                    [t, -flow]  # negative sign = inflow in SCHISM convention
                    for t, flow in zip(series.seconds_from_start, series.flow_cms)
                ]
            )
            np.savetxt(output_file, data, fmt=["%d", "%.3f"])
            log.info(
                f"Wrote St. Lawrence flux.th: {len(series.seconds_from_start)} "
                f"timesteps -> {output_file}"
            )
            return output_file
        except Exception as exc:
            log.error(f"Failed to write flux.th: {exc}")
            return None

    def _write_tem_1_th(self, series: _StLawrenceSeries) -> Optional[Path]:
        output_file = self.output_path / "TEM_1.th"
        try:
            data = np.array(
                [
                    [t, temp]
                    for t, temp in zip(series.seconds_from_start, series.temp_c)
                ]
            )
            np.savetxt(output_file, data, fmt=["%d", "%.3f"])
            log.info(
                f"Wrote St. Lawrence TEM_1.th: {len(series.seconds_from_start)} "
                f"timesteps -> {output_file}"
            )
            return output_file
        except Exception as exc:
            log.error(f"Failed to write TEM_1.th: {exc}")
            return None

    # --------------------------------------------------------- Archive fallback

    def _fallback_from_archive(self, output_name: str) -> Optional[Path]:
        """Copy the previous cycle's archive and re-stamp its time axis.

        Operational shell reads the previous cycle's archive
        ``<prefix>.riv.obs.flux.th`` / ``…tem_1.th`` and rewrites the time
        column by stepping one slot forward (time_k <- time_{k+1}), keeping
        the value column. That effectively advances the timeline by one
        day, leaving the trailing value duplicated.
        """
        if self.archive_prefix is None or self.prev_rerun_dir is None:
            return None
        archive_name = (
            f"{self.archive_prefix}.riv.obs."
            f"{'flux' if output_name == 'flux.th' else 'tem_1'}.th"
        )
        src = self.prev_rerun_dir / archive_name
        if not src.exists():
            return None
        try:
            raw = np.loadtxt(src, dtype=float)
            if raw.ndim != 2 or raw.shape[1] < 2:
                log.warning(f"Archive {src} has unexpected shape {raw.shape}")
                return None
            # Shift the value column one step forward (operational idx_2 logic).
            shifted_times = raw[:, 0].copy()
            shifted_vals = raw[:, 1].copy()
            if len(shifted_vals) >= 2:
                shifted_vals[:-1] = raw[1:, 1]
                # Last row keeps the second-to-last new value (matches
                # the idx_2 = (N-2)*2+3 branch in the shell).
                shifted_vals[-1] = raw[-1, 1]
            out = np.column_stack([shifted_times.astype(int), shifted_vals])
            output_file = self.output_path / output_name
            np.savetxt(output_file, out, fmt=["%d", "%.3f"])
            log.info(
                f"Used previous-cycle archive for {output_name} <- {src}"
            )
            return output_file
        except Exception as exc:
            log.warning(f"Failed to restage archive {src}: {exc}")
            return None

    # ---------------------------------------------------------------- Helpers

    def _cycle_datetime(self) -> datetime:
        base = datetime.strptime(self.config.pdy, "%Y%m%d")
        return base + timedelta(hours=self.config.cyc)

    def _n_days_total(self) -> int:
        """Span (in days) covered by the output, for use with _daily_range.

        Operational STOFS configures nowcast=24h + forecast=108h = 5.5 days,
        producing a 7-row flux.th (days 0..6 inclusive). _daily_range
        takes the span and adds one entry, so we pass the ceiling span.
        With nowcast=24h/forecast=108h this returns 6 (span) → 7 rows.
        """
        total_hours = self.config.nowcast_hours + self.config.forecast_hours
        return int(np.ceil(total_hours / 24.0))

    @staticmethod
    def _daily_range(start: datetime, days: int):
        """Return a pandas UTC DatetimeIndex spanning ``days`` days, inclusive.

        Mirrors ``pd.date_range(start, start + timedelta(days=days))`` which
        yields ``days + 1`` entries (both endpoints).
        """
        return pd.date_range(
            start=start.strftime("%Y-%m-%d %H:00:00"),
            periods=days + 1,
            freq="D",
            tz="UTC",
        )

    def _csv_path_for(self, day: datetime) -> Path:
        return (
            self.input_path
            / day.strftime("%Y%m%d")
            / "can_streamgauge"
            / self.csv_name
        )

    def _find_csv(self, start: datetime) -> Optional[Path]:
        for days_back in (0, 1):
            p = self._csv_path_for(start - timedelta(days=days_back))
            if p.exists():
                return p
        # Also check flat input_path/<csv_name> for tests & single-dir layouts.
        flat = self.input_path / self.csv_name
        if flat.exists():
            return flat
        return None
