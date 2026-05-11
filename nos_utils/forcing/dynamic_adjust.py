"""
Dynamic SSH boundary adjustment (STOFS-3D-ATL).

Corrects the RTOFS-derived SSH boundary (``elev2D.th.nc``) using real-time
NOAA tide-gauge observations. The operational pipeline (see
``stofs_3d_atl_create_obc_3d_th_dynamic_adjust.sh``) runs three stages:

    1. Observations: pull XML water-level files from
       ``$DCOMROOT/<yyyymmdd>/coops_waterlvlobs/<staID>.xml`` for 11
       reference stations, convert to CSV then NPZ.
    2. Bias: compare the model ``staout_1`` timeseries (previous cycle) to
       the NOAA observations over a 2-day window. Compute the mean
       per-station bias, then average across valid stations (stations with
       >20% NaN are excluded).
    3. Apply: subtract a time-varying offset from ``elev2D.th.nc``:
         * t=0: previous cycle's average bias (``adj0``)
         * t=1: mean of adj0 and today's bias (``avg_adj_0_1``)
         * t>=2: today's average bias (``adj1`` == ``adj_p``)
       NaN biases are treated as 0.0.

This module ports the algorithm to pure Python (no NCO/CDO/awk). When a
required input is missing it degrades gracefully — most commonly to a
zero-bias correction, which leaves ``elev2D.th.nc`` unchanged.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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
    from scipy import interpolate, stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from netCDF4 import Dataset
    HAS_NETCDF4 = True
except ImportError:
    HAS_NETCDF4 = False


# Operational reference station list (Fort Pulaski -> Newport, 11 stations).
# Excludes stations with zero geoid-NAVD88 offset (Cape Henry, Nantucket)
# and low-quality gauges (Beaufort 8656483, Wachapreague 8631044).
DEFAULT_STATIONS: List[str] = [
    "8670870", "8665530", "8661070", "8658163", "8651370", "8632200",
    "8557380", "8536110", "8534720", "8531680", "8452660",
]

DEFAULT_STATION_LONS: List[float] = [
    -80.90303, -79.923615, -78.9183, -77.7867, -75.7467, -75.9884,
    -75.11928, -74.96, -74.41805, -74.0094, -71.32614,
]
DEFAULT_STATION_LATS: List[float] = [
    32.034695, 32.780834, 33.655, 34.213306, 36.1833, 37.1652,
    38.782833, 38.9683, 39.356667, 40.4669, 41.504333,
]

# Stations with more than this NaN ratio in their resampled hourly bias
# timeseries are dropped before averaging.
DEFAULT_NAN_THRESHOLD = 0.20

# "Gap" threshold for observation validity: multiples of the modal sample
# interval. Matches operational `gap_idx=10`.
GAP_MULTIPLIER = 10


# =============================================================================
# Data containers
# =============================================================================

@dataclass
class NOAAStation:
    """Metadata + observation timeseries for a single NOAA tide gauge."""
    station_id: str
    lon: float
    lat: float
    # Observation samples in pandas UTC Timestamps and water-level meters.
    times: List["pd.Timestamp"] = field(default_factory=list)
    elev_m: List[float] = field(default_factory=list)


@dataclass
class ObsBundle:
    """Concatenated observation data across all stations (matches npz_data)."""
    station_ids: np.ndarray  # shape (N,) string
    times: np.ndarray        # shape (N,) datetime64[ns]
    elev: np.ndarray         # shape (N,) float
    station_lons: Dict[str, float]
    station_lats: Dict[str, float]


# =============================================================================
# XML / CSV parsing
# =============================================================================

_XML_OBS_RE = re.compile(
    r"t=\"(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2})(:\d{2})?\""
    r"\s+v=\"(-?\d+(?:\.\d+)?)\"",
    re.IGNORECASE,
)


def parse_noaa_xml(xml_path: Path) -> Tuple[List["pd.Timestamp"], List[float]]:
    """Parse a NOAA CO-OPS water-level XML file.

    The operational shell uses awk on fields 4/5/6 which targets the
    observations block. We parse the same content with a regex that
    recognises either `<wl t="..." v="..."/>` or `<obs t="..." v="..."/>`
    and any surrounding whitespace. Missing or non-numeric ``v`` entries
    are dropped.
    """
    text = xml_path.read_text(errors="replace")
    ts_list: List["pd.Timestamp"] = []
    vals: List[float] = []
    for match in _XML_OBS_RE.finditer(text):
        date_str, hhmm, ss, v_str = match.groups()
        try:
            v = float(v_str)
        except ValueError:
            continue
        if not math.isfinite(v):
            continue
        timestamp = pd.Timestamp(
            f"{date_str} {hhmm}{ss or ':00'}", tz="UTC"
        )
        ts_list.append(timestamp)
        vals.append(v)
    return ts_list, vals


def load_observations(
    src_dir: Path,
    stations: Sequence[str],
    station_lons: Sequence[float],
    station_lats: Sequence[float],
) -> ObsBundle:
    """Load NOAA observations from a directory of ``<stationID>.xml`` files.

    Missing files are silently skipped. The result bundles timeseries
    across all stations in a single 1D vector (matching the NPZ layout the
    operational ``derive_bias.py`` expects).
    """
    if not HAS_PANDAS:
        raise RuntimeError("pandas is required for NOAA observation loading")

    lon_map = dict(zip(stations, station_lons))
    lat_map = dict(zip(stations, station_lats))

    all_ids: List[str] = []
    all_times: List["pd.Timestamp"] = []
    all_elev: List[float] = []
    for sta in stations:
        xml = src_dir / f"{sta}.xml"
        if not xml.exists():
            log.info(f"NOAA XML missing for station {sta}: {xml}")
            continue
        try:
            ts, vals = parse_noaa_xml(xml)
        except Exception as exc:
            log.warning(f"Failed to parse {xml}: {exc}")
            continue
        if not ts:
            log.info(f"No observations found in {xml}")
            continue
        all_ids.extend([sta] * len(ts))
        all_times.extend(ts)
        all_elev.extend(vals)

    if not all_ids:
        return ObsBundle(
            station_ids=np.array([], dtype=object),
            times=np.array([], dtype="datetime64[ns]"),
            elev=np.array([], dtype=float),
            station_lons=lon_map,
            station_lats=lat_map,
        )

    times_np = np.array(
        [pd.Timestamp(t).tz_convert(None).to_datetime64() for t in all_times]
    )
    return ObsBundle(
        station_ids=np.array(all_ids, dtype=object),
        times=times_np,
        elev=np.array(all_elev, dtype=float),
        station_lons=lon_map,
        station_lats=lat_map,
    )


# =============================================================================
# param.nml + staout_1 helpers
# =============================================================================

def read_model_start(param_nml_path: Path) -> Optional[datetime]:
    """Extract ``start_year``/``month``/``day``/``hour`` from a SCHISM param.nml.

    Returns None if the file is missing or unparseable.
    """
    try:
        text = param_nml_path.read_text(errors="replace")
    except OSError:
        return None
    def _pick(name: str) -> Optional[int]:
        m = re.search(rf"{name}\s*=\s*(-?\d+)", text)
        return int(m.group(1)) if m else None
    y = _pick("start_year")
    mo = _pick("start_month")
    d = _pick("start_day")
    h = _pick("start_hour")
    if None in (y, mo, d):
        return None
    return datetime(y, mo, d, h or 0)


def read_staout_1(
    staout_path: Path, n_stations: Optional[int] = None
) -> Optional[np.ndarray]:
    """Load staout_1 (model water level timeseries at station rows).

    Format: whitespace-separated columns, first column = seconds from model
    start, remaining columns = SSH at each station row.
    """
    try:
        data = np.loadtxt(staout_path)
    except Exception as exc:
        log.warning(f"Failed to read staout_1 {staout_path}: {exc}")
        return None
    if data.ndim != 2 or data.shape[0] == 0:
        log.warning(f"staout_1 {staout_path} has unexpected shape {data.shape}")
        return None
    if n_stations is not None and data.shape[1] < n_stations + 1:
        log.warning(
            f"staout_1 {staout_path} has {data.shape[1] - 1} station columns, "
            f"expected >= {n_stations}"
        )
    return data


def read_bp_stations(bp_path: Path) -> Tuple[List[str], List[float], List[float]]:
    """Parse a SCHISM station.bp file.

    Format::
        <comment>
        N
        1 lon lat depth ! stationID
        2 lon lat depth ! stationID
        ...

    Returns (station_ids, lons, lats).
    """
    ids: List[str] = []
    lons: List[float] = []
    lats: List[float] = []
    with open(bp_path) as fh:
        lines = fh.readlines()
    if len(lines) < 2:
        return ids, lons, lats
    # Line 1 = comment, line 2 = count
    try:
        n = int(lines[1].split()[0])
    except (IndexError, ValueError):
        n = 0
    for ln in lines[2:2 + n]:
        parts = ln.split("!", 1)
        tokens = parts[0].split()
        if len(tokens) < 3:
            continue
        try:
            lon = float(tokens[1])
            lat = float(tokens[2])
        except ValueError:
            continue
        # Station ID is either the inline comment after '!' or the 5th token.
        if len(parts) > 1:
            sid = parts[1].strip().split()[0] if parts[1].strip() else ""
        else:
            sid = tokens[4] if len(tokens) > 4 else f"station_{len(ids)+1}"
        ids.append(sid)
        lons.append(lon)
        lats.append(lat)
    return ids, lons, lats


def read_diff_bp(bp_path: Path) -> Dict[str, float]:
    """Parse diff.bp (xGEOID - NAVD88 per station), returning {sid: offset}."""
    ids, _, _ = read_bp_stations(bp_path)
    offsets: Dict[str, float] = {}
    with open(bp_path) as fh:
        lines = fh.readlines()
    if len(lines) < 2:
        return offsets
    try:
        n = int(lines[1].split()[0])
    except (IndexError, ValueError):
        return offsets
    for i, ln in enumerate(lines[2:2 + n]):
        parts = ln.split("!", 1)
        tokens = parts[0].split()
        if len(tokens) < 4 or i >= len(ids):
            continue
        try:
            z = float(tokens[3])
        except ValueError:
            continue
        offsets[ids[i]] = z
    return offsets


# =============================================================================
# Bias computation
# =============================================================================

def _valid_station_mask(
    obs_times: np.ndarray,
    obs_elev: np.ndarray,
    target_start: "pd.Timestamp",
    target_end: "pd.Timestamp",
) -> Tuple[bool, np.ndarray]:
    """Return (has_data, sort_order) for a single station's obs window."""
    mask = (obs_times >= target_start.to_datetime64()) & (
        obs_times <= target_end.to_datetime64()
    )
    if not mask.any():
        return False, np.array([], dtype=int)
    sel_times = obs_times[mask]
    sel_elev = obs_elev[mask]
    if np.all(np.isnan(sel_elev)):
        return False, np.array([], dtype=int)
    # Sort by time and keep only ascending order.
    order = np.argsort(sel_times)
    return True, (np.where(mask)[0][order])


def _interpolate_obs_to_model(
    obs_times: np.ndarray, obs_elev: np.ndarray, model_times: np.ndarray,
) -> np.ndarray:
    """Linearly interpolate obs onto model times, leaving NaN inside gaps.

    Mirrors ``derive_bias.interpolate_observations`` — which splits the
    record at gaps larger than 10x the modal sampling interval and
    interpolates within each segment.
    """
    if not HAS_SCIPY:
        raise RuntimeError("scipy is required for bias interpolation")
    if obs_times.size < 2:
        return np.full_like(model_times, np.nan, dtype=float)

    # Convert datetimes to fractional days (pandas ns → days).
    t_days = obs_times.astype("datetime64[ns]").astype("int64") / 8.64e13
    m_days = model_times.astype("datetime64[ns]").astype("int64") / 8.64e13

    dt = np.diff(t_days)
    if dt.size == 0:
        return np.full_like(m_days, np.nan, dtype=float)
    mode_dt = float(stats.mode(dt, keepdims=False).mode)
    gap_thresh = mode_dt * GAP_MULTIPLIER
    gap_idx = np.where(dt > gap_thresh)[0]

    result = np.full_like(m_days, np.nan, dtype=float)

    start = 0
    for g in gap_idx:
        if g > start:
            seg_t = t_days[start:g + 1]
            seg_y = obs_elev[start:g + 1]
            f = interpolate.interp1d(
                seg_t, seg_y, bounds_error=False, fill_value=np.nan,
            )
            seg_mask = (m_days >= seg_t[0]) & (m_days <= seg_t[-1])
            result[seg_mask] = f(m_days[seg_mask])
        start = g + 1
    if start < len(t_days):
        seg_t = t_days[start:]
        seg_y = obs_elev[start:]
        if seg_t.size >= 2:
            f = interpolate.interp1d(
                seg_t, seg_y, bounds_error=False, fill_value=np.nan,
            )
            seg_mask = (m_days >= seg_t[0]) & (m_days <= seg_t[-1])
            result[seg_mask] = f(m_days[seg_mask])
    return result


def compute_bias(
    obs: ObsBundle,
    model_staout: np.ndarray,
    model_start: datetime,
    station_ids_in: Sequence[str],
    datum_offsets: Dict[str, float],
    target_start: datetime,
    target_end: datetime,
    nan_threshold: float = DEFAULT_NAN_THRESHOLD,
) -> Tuple[float, Dict[str, float]]:
    """Compute the scalar bias to subtract from SSH boundary.

    Returns (average_bias, per_station_bias). When no stations have enough
    data to compute a bias, returns (``float('nan')``, {}).
    """
    if not HAS_PANDAS:
        raise RuntimeError("pandas is required for bias computation")

    ts = pd.Timestamp(target_start, tz=None)
    te = pd.Timestamp(target_end, tz=None)

    # Model time vector (seconds since model_start -> datetime64).
    t_sec = model_staout[:, 0]
    m_times = np.array(
        [np.datetime64(model_start + timedelta(seconds=float(s)), "ns")
         for s in t_sec]
    )
    m_mask = (m_times >= np.datetime64(ts)) & (m_times < np.datetime64(te))
    if not m_mask.any():
        log.warning(
            "Model staout timeseries does not cover bias window "
            f"[{target_start} .. {target_end}]"
        )
        return float("nan"), {}

    m_times_sel = m_times[m_mask]

    per_station_bias: Dict[str, np.ndarray] = {}

    for idx_in, sid in enumerate(station_ids_in):
        if sid not in obs.station_lons:
            continue
        sta_mask = obs.station_ids == sid
        if not sta_mask.any():
            continue
        has_data, order = _valid_station_mask(
            obs.times[sta_mask], obs.elev[sta_mask], ts, te,
        )
        if not has_data:
            continue
        sub_times = obs.times[sta_mask][order]
        sub_elev = obs.elev[sta_mask][order]

        # Convert observations from NAVD88 → xGEOID (add datum offset).
        offset = datum_offsets.get(sid, 0.0)
        sub_elev_adj = sub_elev + offset

        # Interpolate obs onto the selected model time vector.
        try:
            soyi = _interpolate_obs_to_model(sub_times, sub_elev_adj, m_times_sel)
        except Exception as exc:
            log.warning(f"Interpolation failed for station {sid}: {exc}")
            continue

        # Model column for this station.
        model_col = model_staout[m_mask, idx_in + 1]
        bias_series = model_col - soyi
        per_station_bias[sid] = bias_series

    if not per_station_bias:
        return float("nan"), {}

    df = pd.DataFrame(per_station_bias, index=pd.to_datetime(m_times_sel))
    df_hourly = df.resample("h").mean()

    valid_cols = []
    for col in df_hourly.columns:
        if len(df_hourly) == 0:
            continue
        nan_ratio = df_hourly[col].isna().sum() / len(df_hourly)
        if nan_ratio <= nan_threshold:
            valid_cols.append(col)
        else:
            log.info(
                f"Excluding station {col}: NaN ratio {nan_ratio:.0%}"
            )

    if not valid_cols:
        return float("nan"), {}

    filtered = df_hourly[valid_cols].copy()
    filtered["Average"] = filtered.mean(axis=1)
    avg_bias = float(filtered["Average"].mean())

    per_station_mean = {
        sid: float(np.nanmean(vals)) for sid, vals in per_station_bias.items()
    }
    return avg_bias, per_station_mean


# =============================================================================
# NetCDF SSH adjustment
# =============================================================================

def apply_ssh_time_varying_adjust(
    elev_nc: Path,
    adj0: float,
    adj1: float,
    var_name: str = "time_series",
) -> bool:
    """Subtract a time-varying bias from ``time_series`` in ``elev_nc``.

    The operational ex-script applies:
        * t=0  -> value - adj0
        * t=1  -> value - (adj0 + adj1) / 2
        * t>=2 -> value - adj1
    NaN arguments are treated as 0.0 (which leaves that time slice
    unchanged, matching the shell's ``flag_adj0=1; adj0=0.0`` fallback).
    """
    if not HAS_NETCDF4:
        raise RuntimeError("netCDF4 is required for SSH adjustment")
    if not elev_nc.exists():
        log.error(f"Cannot apply SSH adjust: {elev_nc} not found")
        return False

    adj0_f = 0.0 if (adj0 is None or math.isnan(adj0)) else float(adj0)
    adj1_f = 0.0 if (adj1 is None or math.isnan(adj1)) else float(adj1)
    avg = 0.5 * (adj0_f + adj1_f)

    try:
        with Dataset(str(elev_nc), "r+") as ds:
            if var_name not in ds.variables:
                log.error(f"Variable {var_name} not in {elev_nc}")
                return False
            var = ds.variables[var_name]
            n_t = var.shape[0]
            if n_t == 0:
                log.warning(f"{elev_nc} {var_name} has no time records")
                return True
            data = var[...]
            # Handle masked arrays gracefully.
            if np.ma.isMaskedArray(data):
                data = data.filled(np.nan)
            data = np.asarray(data, dtype=np.float64)
            data[0] = data[0] - adj0_f
            if n_t > 1:
                data[1] = data[1] - avg
            if n_t > 2:
                data[2:] = data[2:] - adj1_f
            var[:] = data.astype(var.dtype, copy=False)
        log.info(
            f"Applied SSH dynamic adjust to {elev_nc.name}: "
            f"adj0={adj0_f:.4f} adj1={adj1_f:.4f} avg={avg:.4f}"
        )
        return True
    except Exception as exc:
        log.error(f"Failed to apply SSH adjustment to {elev_nc}: {exc}")
        return False


# =============================================================================
# Main processor
# =============================================================================

class DynamicAdjustProcessor(ForcingProcessor):
    """Compute and apply a dynamic SSH bias correction for STOFS-3D-ATL.

    Inputs (all resolved from constructor args or FIX/COMOUT defaults):
        - ``obs_dir``: ``<DCOMROOT>/<pdy>/coops_waterlvlobs/`` with
          ``<staID>.xml`` files.
        - ``prev_staout_1``: Previous cycle's model SSH timeseries
          (``staout_1``).
        - ``prev_param_nml``: Previous cycle's ``param.nml`` (for model
          start time).
        - ``station_bp`` + ``diff_bp``: from FIX.
        - ``prev_avg_bias_file``: Scalar file produced by the previous
          cycle's run.
        - ``elev2d_th_nc``: ``elev2D.th.nc`` to adjust in-place.

    Writes:
        - ``average_bias_today``: Scalar bias (float on a single line)
          into ``output_path`` for next cycle.
        - Adjusted ``elev2D.th.nc`` at ``elev2d_th_nc`` path.
    """

    SOURCE_NAME = "DYNAMIC_SSH_ADJUST"

    def __init__(
        self,
        config: ForcingConfig,
        input_path: Path,
        output_path: Path,
        *,
        obs_dir: Optional[Path] = None,
        prev_staout_1: Optional[Path] = None,
        prev_param_nml: Optional[Path] = None,
        station_bp: Optional[Path] = None,
        diff_bp: Optional[Path] = None,
        prev_avg_bias_file: Optional[Path] = None,
        elev2d_th_nc: Optional[Path] = None,
        stations: Sequence[str] = DEFAULT_STATIONS,
        station_lons: Sequence[float] = DEFAULT_STATION_LONS,
        station_lats: Sequence[float] = DEFAULT_STATION_LATS,
        bias_window_days: int = 2,
        nan_threshold: float = DEFAULT_NAN_THRESHOLD,
    ) -> None:
        super().__init__(config, input_path, output_path)
        self.obs_dir = Path(obs_dir) if obs_dir else None
        self.prev_staout_1 = Path(prev_staout_1) if prev_staout_1 else None
        self.prev_param_nml = Path(prev_param_nml) if prev_param_nml else None
        self.station_bp = Path(station_bp) if station_bp else None
        self.diff_bp = Path(diff_bp) if diff_bp else None
        self.prev_avg_bias_file = (
            Path(prev_avg_bias_file) if prev_avg_bias_file else None
        )
        self.elev2d_th_nc = (
            Path(elev2d_th_nc) if elev2d_th_nc
            else (output_path / "elev2D.th.nc")
        )
        self.stations = list(stations)
        self.station_lons = list(station_lons)
        self.station_lats = list(station_lats)
        self.bias_window_days = bias_window_days
        self.nan_threshold = nan_threshold

    # ------------------------------------------------------------------ API

    def process(self) -> ForcingResult:
        errors: List[str] = []
        warnings: List[str] = []
        metadata: Dict[str, object] = {"stations_configured": len(self.stations)}

        if not HAS_PANDAS or not HAS_SCIPY:
            return ForcingResult(
                success=False, source=self.SOURCE_NAME,
                errors=["pandas + scipy required for dynamic SSH adjustment"],
            )

        self.create_output_dir()

        if not self.elev2d_th_nc.exists():
            return ForcingResult(
                success=False, source=self.SOURCE_NAME,
                errors=[
                    f"elev2D.th.nc not found at {self.elev2d_th_nc}; "
                    "run RTOFS OBC first"
                ],
            )

        # --- Step 1: today's bias (requires observations + prev cycle model) --
        adj_today = self._compute_today_bias(warnings, metadata)

        # --- Step 2: previous cycle's bias (from a scalar file) ----------
        adj_prev = self._read_prev_bias(warnings)
        metadata["adj0"] = adj_prev if adj_prev is not None else "NaN"
        metadata["adj1"] = adj_today if adj_today is not None else "NaN"

        # --- Step 3: write today's bias for the next cycle ---------------
        out_bias_file = self.output_path / "average_bias_today"
        self._write_scalar(out_bias_file, adj_today)

        # --- Step 4: apply bias to elev2D.th.nc --------------------------
        applied = apply_ssh_time_varying_adjust(
            self.elev2d_th_nc,
            adj0=adj_prev if adj_prev is not None else float("nan"),
            adj1=adj_today if adj_today is not None else float("nan"),
        )
        if not applied:
            errors.append(f"Failed to apply SSH adjust to {self.elev2d_th_nc}")

        output_files: List[Path] = []
        if out_bias_file.exists():
            output_files.append(out_bias_file)
        # Always record elev2D.th.nc as an output so downstream reporting sees
        # the adjusted file, whether or not we changed it.
        output_files.append(self.elev2d_th_nc)

        return ForcingResult(
            success=len(errors) == 0,
            source=self.SOURCE_NAME,
            output_files=output_files,
            errors=errors,
            warnings=warnings,
            metadata=metadata,
        )

    def find_input_files(self) -> List[Path]:
        """Return the NOAA XML files discovered for today's stations."""
        if self.obs_dir is None or not self.obs_dir.exists():
            return []
        return [
            self.obs_dir / f"{sid}.xml"
            for sid in self.stations
            if (self.obs_dir / f"{sid}.xml").exists()
        ]

    # -------------------------------------------------------------- Helpers

    def _compute_today_bias(
        self,
        warnings: List[str],
        metadata: Dict[str, object],
    ) -> Optional[float]:
        """Load observations + prev staout_1 and return today's avg bias."""
        # Required inputs — without any of these we can't compute today's bias.
        if self.obs_dir is None:
            warnings.append("obs_dir not provided; today's bias = NaN")
            return None
        if not self.obs_dir.exists():
            warnings.append(
                f"NOAA obs directory not found: {self.obs_dir}; today's bias = NaN"
            )
            return None
        if self.prev_staout_1 is None or not self.prev_staout_1.exists():
            warnings.append(
                "Previous cycle staout_1 missing; today's bias = NaN"
            )
            return None
        if self.prev_param_nml is None or not self.prev_param_nml.exists():
            warnings.append(
                "Previous cycle param.nml missing; today's bias = NaN"
            )
            return None

        # Override default station metadata with station.bp when provided.
        station_ids = list(self.stations)
        if self.station_bp and self.station_bp.exists():
            ids, lons, lats = read_bp_stations(self.station_bp)
            if ids:
                station_ids = ids
                # If bp has different stations than the hardcoded list,
                # rebuild the coord lists from bp.
                if station_ids != list(self.stations):
                    self.stations = station_ids
                    self.station_lons = lons
                    self.station_lats = lats

        obs = load_observations(
            self.obs_dir, self.stations, self.station_lons, self.station_lats,
        )
        metadata["obs_stations_found"] = int(np.unique(obs.station_ids).size)
        if obs.station_ids.size == 0:
            warnings.append("No NOAA observations loaded; today's bias = NaN")
            return None

        model_staout = read_staout_1(
            self.prev_staout_1, n_stations=len(station_ids),
        )
        if model_staout is None or model_staout.shape[0] == 0:
            warnings.append(
                f"staout_1 empty / unreadable ({self.prev_staout_1}); "
                "today's bias = NaN"
            )
            return None

        model_start = read_model_start(self.prev_param_nml)
        if model_start is None:
            warnings.append(
                f"Could not read start time from {self.prev_param_nml}; "
                "today's bias = NaN"
            )
            return None

        datum_offsets = {}
        if self.diff_bp and self.diff_bp.exists():
            datum_offsets = read_diff_bp(self.diff_bp)

        # Bias window: operational derive_bias.py is called with
        # ``yyyymmdd_yesterday_Ncast_dash_fmt`` (the day two cycles back
        # = PDY - 48h for a 24h-nowcast) and duration=2 days, with the
        # hour defaulted to 12 inside derive_bias.py. That yields a
        # window of (cycle - 48h @ 12:00) .. (cycle @ 12:00) which
        # overlaps the previous cycle's staout_1 coverage. Our port
        # anchors bias_end at the current cycle's start and walks back
        # ``bias_window_days``. For STOFS (cyc=12z) this matches
        # operational exactly; for non-12z cycles we still anchor at
        # 12:00 UTC of the cycle day to preserve the operational semantic.
        cycle_day = datetime.strptime(self.config.pdy, "%Y%m%d")
        bias_end = cycle_day.replace(hour=12, minute=0, second=0)
        bias_start = bias_end - timedelta(days=self.bias_window_days)

        try:
            avg_bias, per_sta = compute_bias(
                obs,
                model_staout,
                model_start,
                station_ids,
                datum_offsets,
                bias_start,
                bias_end,
                nan_threshold=self.nan_threshold,
            )
        except Exception as exc:
            warnings.append(f"Bias computation failed: {exc}")
            return None

        if math.isnan(avg_bias):
            warnings.append("Bias computation returned NaN")
            metadata["per_station_bias"] = {}
            return None

        metadata["per_station_bias"] = {k: round(v, 4) for k, v in per_sta.items()}
        metadata["n_valid_stations"] = len(per_sta)
        return avg_bias

    def _read_prev_bias(self, warnings: List[str]) -> Optional[float]:
        if self.prev_avg_bias_file is None:
            warnings.append(
                "No prev_avg_bias_file supplied; adj0 defaulting to 0.0"
            )
            return 0.0
        if not self.prev_avg_bias_file.exists():
            warnings.append(
                f"Prev avg bias not found ({self.prev_avg_bias_file}); "
                "adj0 defaulting to 0.0"
            )
            return 0.0
        try:
            text = self.prev_avg_bias_file.read_text().strip()
        except OSError as exc:
            warnings.append(
                f"Failed to read {self.prev_avg_bias_file}: {exc}; adj0=0.0"
            )
            return 0.0
        if not text or re.search(r"nan", text, re.IGNORECASE):
            warnings.append(
                f"Prev avg bias file {self.prev_avg_bias_file} is "
                "empty/NaN; adj0 defaulting to 0.0"
            )
            return 0.0
        try:
            return float(text.split()[0])
        except ValueError:
            warnings.append(
                f"Could not parse prev bias from {self.prev_avg_bias_file} "
                f"({text!r}); adj0=0.0"
            )
            return 0.0

    def _write_scalar(self, path: Path, value: Optional[float]) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if value is None or (isinstance(value, float) and math.isnan(value)):
                path.write_text("nan\n")
            else:
                path.write_text(f"{value:.3f}\n")
        except OSError as exc:
            log.warning(f"Failed to write bias scalar to {path}: {exc}")
