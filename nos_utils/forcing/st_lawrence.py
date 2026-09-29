"""
St. Lawrence River forcing processor (STOFS-3D-ATL).

Generates the two St. Lawrence river files for the flow-only open boundary:

  - ``flux.th``  — daily discharge (m^3/s, negative = inflow). Sources, in the
    ops v3.1 order (``stofs_3d_atl_create_river_st_lawrence.sh``):
      1. the first gauge CSV that exists, today's then yesterday's, under
         ``$COMINlaw/<yyyymmdd>/<subdir>/``. Two layouts are read, chosen by
         the file's contents: the v3.1 parameter-coded table
         (``can_streamgauge/02OA016_hydrometric.csv``, discharge = parameter
         47) and the v2.1 wide export
         (``canadian_water/QC_02OA016_hourly_hydrometric.csv``, discharge in
         column 6);
      2. the day-of-year climatology (``stofs_3d_atl_StLawrence_clim.txt``),
         one row per run day;
      3. the previous cycle's archived ``flux.th``, copied unchanged.
  - ``TEM_1.th`` — daily water temperature from GFS air temperature at the
    river mouth, ``T_water = 0.83 * T_air + 2.817`` (negative values clamped
    to 0), built from the current sflux radiation file whatever the flux
    source. Without it the previous cycle's archive is used, else a -9999
    sentinel.

Ports ``gen_fluxth_st_lawrence_riv.py`` and ``gen_temp_1_st_lawrence_riv.py``
plus the ex-script's fallbacks. MJ (09/28/26)
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


# Water-temperature linear regression at the St. Lawrence mouth; derived
# operationally from multi-year GFS-rad-to-observed river-temperature fits.
AIR_TO_WATER_SLOPE = 0.83
AIR_TO_WATER_INTERCEPT = 2.817

# River mouth coordinates (lat, lon in degrees) used for air-temp sampling.
RIVER_MOUTH = (45.415, -73.623056)

# Operational filename default.
DEFAULT_CSV_NAME = "02OA016_hydrometric.csv"

# Subdirectory under $COMINlaw/<pdy>/ holding the hydrometric CSV.
# Legacy default; operational WCOSS2 uses "canadian_water".
DEFAULT_SUBDIR = "can_streamgauge"

# Day-of-year climatology (fix file) used when no observation file yields data.  MJ (09/28/26)
DEFAULT_CLIM_NAME = "stofs_3d_atl_StLawrence_clim.txt"

# v3.1 long-format parameter codes (gen_fluxth_st_lawrence_riv.py).  MJ (09/28/26)
PARAM_DISCHARGE = 47


@dataclass
class _StLawrenceSeries:
    """Daily flow + temperature series for a single river."""
    # Days 0..N-1 relative to nowcast start, where N = nowcast_days + forecast_days + 1
    seconds_from_start: List[int]
    flow_cms: List[float]
    temp_c: List[float]
    # Pre-formatted flux.th rows (climatology path); None = derive from flow_cms.  MJ (09/28/26)
    flux_lines: Optional[List[str]] = None
    temp_from_sflux: bool = False


class StLawrenceProcessor(ForcingProcessor):
    """Produce flux.th and TEM_1.th for the St. Lawrence River.

    Flux source order: the first gauge CSV found at
    ``<input_path>/<pdy>/<subdir>/<csv_name>`` (today, else yesterday), then
    ``clim_file``, then ``prev_rerun_dir``'s archived
    ``<run>.<cycle>.riv.obs.flux.th``. See the module docstring for the two
    CSV layouts. Temperature comes from ``sflux_rad_file`` when present,
    otherwise the archived ``...tem_1.th``, otherwise a -9999 sentinel.
    Returns ``success=False`` when no flux.th can be produced. MJ (09/28/26)
    """

    SOURCE_NAME = "ST_LAWRENCE"

    def __init__(
        self,
        config: ForcingConfig,
        input_path: Path,
        output_path: Path,
        *,
        csv_name: str = DEFAULT_CSV_NAME,
        subdir: str = DEFAULT_SUBDIR,
        sflux_rad_file: Optional[Path] = None,
        prev_rerun_dir: Optional[Path] = None,
        archive_prefix: Optional[str] = None,
        clim_file: Optional[Path] = None,
    ) -> None:
        super().__init__(config, input_path, output_path)
        self.csv_name = csv_name
        # Subdirectory under <input_path>/<pdy>/ holding the CSV
        # (operational WCOSS2: "canadian_water"; legacy: "can_streamgauge").
        self.subdir = subdir
        self.sflux_rad_file = Path(sflux_rad_file) if sflux_rad_file else None
        self.prev_rerun_dir = Path(prev_rerun_dir) if prev_rerun_dir else None
        # Archive prefix like "stofs_3d_atl.t12z" — determines the fallback
        # archive filenames (…riv.obs.flux.th / …riv.obs.tem_1.th).
        self.archive_prefix = archive_prefix
        self.clim_file = Path(clim_file) if clim_file else None

    # ------------------------------------------------------------------ API

    def process(self) -> ForcingResult:
        if not HAS_PANDAS:
            return ForcingResult(
                success=False, source=self.SOURCE_NAME,
                errors=["pandas is required for StLawrenceProcessor"],
            )

        self.create_output_dir()

        from ._log import log_input_files
        log_input_files(
            self.SOURCE_NAME, self.find_input_files(),
            source="ST_LAWRENCE", category="river",
            note=f"pdy={self.config.pdy} cyc={self.config.cyc:02d}",
        )

        # ``start`` anchors the in-file time axis at SCHISM's ``model_t0``
        # (= ``cycle - nowcast_hours``). CSV/sflux discovery uses the cycle's
        # PDY directory, not model_t0, to match the operational layout.
        start = self._cycle_datetime()
        pdy_dt = self._pdy_datetime()
        n_days_total = self._n_days_total()
        datevectors_hindcast = self._daily_range(start, days=1)
        datevectors_full = self._daily_range(start, days=n_days_total)

        warnings: List[str] = []
        output_files: List[Path] = []

        # Ops v3.1 picks the FIRST existing obs file (today, then yesterday)
        # and, if it yields no usable data, goes straight to climatology.  MJ (09/28/26)
        csv_path = self._find_csv(pdy_dt)
        series: Optional[_StLawrenceSeries] = None
        tried: List[str] = []
        source_used = None

        if csv_path is not None:
            log.info(f"St. Lawrence CSV: {csv_path}")
            try:
                series = self._read_hydrometric_csv(
                    csv_path, datevectors_hindcast, datevectors_full
                )
                source_used = f"obs file {csv_path}"
            except Exception as exc:
                warnings.append(f"Failed to parse CSV {csv_path}: {exc}")
                tried.append(f"obs {csv_path}: {exc}")
                series = None
        else:
            tried.append(
                "obs: no file at "
                + " or ".join(str(self._csv_path_for(pdy_dt - timedelta(days=d)))
                              for d in (0, 1))
            )

        if series is None:
            try:
                series = self._read_climatology(start, datevectors_full)
                source_used = f"climatology {self.clim_file}"
                warnings.append(
                    "St. Lawrence obs unavailable; using climatology "
                    f"{self.clim_file}"
                )
            except Exception as exc:
                tried.append(f"climatology {self.clim_file}: {exc}")

        # Temperature is built from the current sflux rad file whatever the flux
        # source was, as ops runs gen_temp_1 separately from the flux step
        # (`rm -f TEM_1.th` in between). MJ (09/28/26)
        temp_from_sflux: Optional[List[float]] = None
        if self.sflux_rad_file and self.sflux_rad_file.exists():
            try:
                temp_from_sflux = self._temp_from_sflux(
                    self.sflux_rad_file, datevectors_full
                )
            except Exception as exc:
                warnings.append(f"Failed to read sflux rad {self.sflux_rad_file}: {exc}")
        if series is not None and temp_from_sflux is not None:
            series.temp_c = temp_from_sflux
            series.temp_from_sflux = True

        if series is not None:
            flux_path = self._write_flux_th(series)
            if flux_path:
                output_files.append(flux_path)
            # No sflux temperature: ops reuses the previous cycle's TEM_1.th.  MJ (09/28/26)
            if not series.temp_from_sflux and self.prev_rerun_dir is not None:
                archived = self._fallback_from_archive("TEM_1.th")
                if archived is not None:
                    output_files.append(archived)
                    warnings.append("Using previous-cycle archive for TEM_1.th")
            if not any(p.name == "TEM_1.th" for p in output_files):
                temp_path = self._write_tem_1_th(series)
                if temp_path:
                    output_files.append(temp_path)
        else:
            # Last resort: previous cycle's archive (ops copies it unchanged).  MJ (09/28/26)
            tried.append(f"previous-cycle archive in {self.prev_rerun_dir}")
            archived = self._fallback_from_archive("flux.th")
            if archived is not None:
                output_files.append(archived)
                warnings.append("Using previous-cycle archive for flux.th")
            if temp_from_sflux is not None:
                secs = [int((dt - datevectors_full[0]).total_seconds())
                        for dt in datevectors_full]
                temp_path = self._write_tem_1_th(_StLawrenceSeries(
                    seconds_from_start=secs, flow_cms=[0.0] * len(secs),
                    temp_c=temp_from_sflux, temp_from_sflux=True,
                ))
                if temp_path:
                    output_files.append(temp_path)
            else:
                archived = self._fallback_from_archive("TEM_1.th")
                if archived is not None:
                    output_files.append(archived)
                    warnings.append("Using previous-cycle archive for TEM_1.th")
            if any(p.name == "flux.th" for p in output_files):
                source_used = f"previous-cycle archive {self.prev_rerun_dir}"

        if not any(p.name == "flux.th" for p in output_files):
            msg = (
                "No St. Lawrence flux.th could be produced; paths tried: "
                + "; ".join(tried)
            )
            log.error(msg)
            return ForcingResult(
                success=False, source=self.SOURCE_NAME,
                errors=[msg], warnings=warnings,
            )

        log.info(f"St. Lawrence flux.th source: {source_used}")
        return ForcingResult(
            success=True,
            source=self.SOURCE_NAME,
            output_files=output_files,
            warnings=warnings,
            metadata={
                "flux_source": source_used,
                "csv_used": str(csv_path) if csv_path and source_used
                and source_used.startswith("obs") else None,
                "sflux_used": str(self.sflux_rad_file) if self.sflux_rad_file else None,
                "n_timesteps": len(series.seconds_from_start) if series else 0,
            },
        )

    def find_input_files(self) -> List[Path]:
        """CSV candidates in priority order (today, then yesterday)."""
        pdy_dt = self._pdy_datetime()
        found: List[Path] = []
        for days_back in (0, 1):
            day = pdy_dt - timedelta(days=days_back)
            p = self._csv_path_for(day)
            if p.exists():
                found.append(p)
        return found

    # -------------------------------------------------------------- CSV read

    def _load_flow_frame(self, csv_path: Path):
        """Return a UTC-indexed frame with a ``flow`` column.

        The layout is chosen from the content: the v3.1 long file has 9
        columns with a small set of integer parameter codes in column 2;
        anything else is treated as the v2.1 wide export.
        """
        df = pd.read_csv(csv_path, sep=",", na_values="")
        if self._is_long_layout(df):
            log.info("St. Lawrence CSV layout: v3.1 long (parameter-coded)")
            return self._flow_frame_long(df)
        log.info("St. Lawrence CSV layout: v2.1 wide")
        return self._flow_frame_wide(df, csv_path)

    @staticmethod
    def _is_long_layout(df) -> bool:
        if df.shape[1] != 9:
            return False
        code = pd.to_numeric(df.iloc[:, 2], errors="coerce")
        if code.isna().all():
            return False
        vals = code.dropna()
        return bool((vals == vals.round()).all() and vals.nunique() <= 20)

    @staticmethod
    def _to_utc_index(frame, date_col: str):
        ts = pd.to_datetime(frame[date_col], errors="coerce")
        if getattr(ts.dt, "tz", None) is None:
            ts = ts.dt.tz_localize("UTC")
        else:
            ts = ts.dt.tz_convert("UTC")
        frame = frame.assign(date_utc=ts).dropna(subset=["date_utc"])
        frame = frame[~frame["date_utc"].duplicated(keep="first")]
        return frame.set_index("date_utc")

    def _flow_frame_long(self, df):
        """Ops v3.1: drop cols [0,4..8]; rows with parameter == 47 are discharge."""
        sub = pd.DataFrame(
            {
                "date_local": df.iloc[:, 1].values,
                "parameter": pd.to_numeric(df.iloc[:, 2], errors="coerce").values,
                "flow": pd.to_numeric(df.iloc[:, 3], errors="coerce").values,
            }
        )
        sub = sub[sub["parameter"] == PARAM_DISCHARGE]
        if sub.empty:
            raise ValueError(f"no parameter {PARAM_DISCHARGE} (discharge) rows")
        return self._to_utc_index(sub[["date_local", "flow"]], "date_local")

    def _flow_frame_wide(self, df, csv_path: Path):
        DATE_COL = 1
        FLOW_COL = 6
        if df.shape[1] <= FLOW_COL:
            raise ValueError(
                f"St. Lawrence CSV {csv_path} has only {df.shape[1]} columns; "
                f"expected the wide ECCC hydrometric layout with discharge at "
                f"column {FLOW_COL} (>= {FLOW_COL + 1} columns), or the v3.1 "
                "9-column parameter-coded layout."
            )
        sub = pd.DataFrame(
            {
                "date_local": df.iloc[:, DATE_COL].values,
                "flow": pd.to_numeric(df.iloc[:, FLOW_COL], errors="coerce").values,
            }
        )
        return self._to_utc_index(sub, "date_local")

    def _read_climatology(self, start: datetime, datevectors_full) -> _StLawrenceSeries:
        """Ops v3.1 clim fallback: one day-of-year row per day from model_t0.

        Values are written as ``%.3f`` of the file value with no sign change
        (the clim file already stores negative inflow). Ops writes a fixed 6
        rows (0..120 h), which ends 12 h short of a 24 h + 108 h run and SCHISM
        aborts at the missing record; write one row per entry of
        ``datevectors_full``, the same span the observation path covers. MJ (09/28/26)
        """
        if self.clim_file is None or not self.clim_file.is_file():
            raise FileNotFoundError("climatology file not found")
        by_doy = {}
        for line in self.clim_file.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 2:
                try:
                    by_doy.setdefault(int(float(parts[0])), float(parts[1]))
                except ValueError:
                    continue
        base = datetime(start.year, start.month, start.day)
        n_rows = len(datevectors_full)
        lines: List[str] = []
        for i in range(n_rows):
            doy = (base + timedelta(days=i)).timetuple().tm_yday
            if doy in by_doy:
                lines.append(f"{i * 86400} {by_doy[doy]:.3f}")
        if len(lines) < n_rows:
            raise ValueError(
                f"climatology {self.clim_file} lacks day-of-year rows "
                f"(got {len(lines)} of {n_rows})"
            )
        secs = [int((dt - datevectors_full[0]).total_seconds())
                for dt in datevectors_full]
        return _StLawrenceSeries(
            seconds_from_start=secs,
            flow_cms=[0.0] * len(secs),
            temp_c=[-9999.0] * len(secs),
            flux_lines=lines,
        )

    def _read_hydrometric_csv(
        self,
        csv_path: Path,
        datevectors_hindcast,
        datevectors_full,
    ) -> _StLawrenceSeries:
        """Return a _StLawrenceSeries with daily flow & temperature.

        Reads either gauge layout via ``_load_flow_frame``, which picks the
        layout from the file's contents and reads columns by position (the
        real header is bilingual):

          * v3.1 ``can_streamgauge/02OA016_hydrometric.csv``: parameter-coded
            rows ``ID, Date (UTC, trailing Z), Parameter, Value, ...``;
            discharge is ``Parameter == 47`` (46 is water level), as in ops
            v3.1 ``gen_fluxth_st_lawrence_riv.py``.
          * v2.1 ``canadian_water/QC_02OA016_hourly_hydrometric.csv``: wide
            export with the date in column 1 (local offset) and discharge in
            column 6.

        Values are looked up at exact timestamps (the nowcast start and each
        following day), not averaged. MJ (09/28/26)

        Discharge windowing (also from the operational script):
          * day 0 miss -> raise (caller falls back to the climatology, then
            the previous-cycle archive);
          * day >0 miss -> carry forward the previous day's value;
          * forecast days are padded with the last available value.

        Neither layout is used for temperature, so the temperature series defaults to the -9999 sentinel here; ``process()``
        overwrites it with the GFS-sflux air-temp regression
        (``_temp_from_sflux``) when a radiation file is available, matching
        operational ``gen_temp_1_st_lawrence_riv.py`` (which never reads the
        CSV for temperature).
        """
        df_flow = self._load_flow_frame(csv_path)

        data_flow: List[float] = []
        last_flow_idx = -1
        for i, dt in enumerate(datevectors_hindcast):
            try:
                value = float(df_flow.loc[dt]["flow"])
                if np.isnan(value):
                    raise KeyError(dt)
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

        # The wide ECCC export has no river-temperature column (operational
        # temperature comes from the GFS-sflux regression, never the CSV), so
        # seed the hindcast window with the -9999 sentinel. ``process()``
        # overwrites this with ``_temp_from_sflux`` when a rad file is present.
        log.info(
            "St. Lawrence CSV carries discharge only; seeding temperature "
            "with -9999 sentinel (caller overwrites with sflux regression "
            "if a radiation file is available)"
        )
        data_temp: List[float] = [-9999.0 for _ in datevectors_hindcast]

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
            if series.flux_lines is not None:
                output_file.write_text("\n".join(series.flux_lines) + "\n")
                return output_file
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

        The exact ``{archive_prefix}.riv.obs.<kind>.th`` is tried first
        (self-sustaining cycles, where the prev rerun was written by an
        earlier nos-workflow run with the same run_name/cyc). When that
        is absent — e.g. bootstrapping from an operational STOFS rerun
        dir whose files carry the production ``stofs_3d_atl.t12z.*``
        prefix, not the UFS run_name/cyc — fall back to the unique
        ``*.riv.obs.<kind>.th`` in the rerun dir.
        """
        if self.prev_rerun_dir is None:
            return None
        kind = "flux" if output_name == "flux.th" else "tem_1"
        src = None
        if self.archive_prefix is not None:
            cand = self.prev_rerun_dir / f"{self.archive_prefix}.riv.obs.{kind}.th"
            if cand.exists():
                src = cand
        if src is None:
            matches = sorted(self.prev_rerun_dir.glob(f"*.riv.obs.{kind}.th"))
            if not matches:
                return None
            src = matches[0]
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
            if kind == "flux":
                # Ops copies the previous flux.th unchanged.  MJ (09/28/26)
                shifted_vals = raw[:, 1].copy()
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
        """SCHISM ``model_t0`` (= ``cycle - nowcast_hours``) for time-axis anchoring.

        Matches the operational shell that sets
        ``PDYHH_NCAST_BEGIN=$($NDATE -<nowcast_hours> $PDYHH)`` and passes
        that as ``startdate`` to ``gen_fluxth_st_lawrence_riv.py`` /
        ``gen_temp_1_st_lawrence_riv.py``. SCHISM reads ``flux.th`` /
        ``TEM_1.th`` with relative seconds-from-model-start, so row ``t=0``
        must correspond to ``model_t0`` — not the cycle hour.

        Note: input CSV/sflux discovery uses ``_pdy_datetime`` (the cycle's
        own date) so the operational ``$COMINlaw/<PDY>/can_streamgauge/``
        layout still resolves; only the in-file time axis is shifted.
        """
        return self._pdy_datetime() - timedelta(hours=self.config.nowcast_hours)

    def _pdy_datetime(self) -> datetime:
        """Cycle hour ``pdy + cyc`` — used for file discovery only."""
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
            / self.subdir
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
