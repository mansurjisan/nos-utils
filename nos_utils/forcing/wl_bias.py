"""
Water-level OBC bias correction (SECOFS / NOS-OFS SCHISM).

Line-anchored Python port of the water-level correction algorithm inside the
operational Fortran ``nos_ofs_create_forcing_obc_schism.f`` (verified against
the patched-but-algorithm-identical copy
``obc_verify/patched_fortran/obc_schism_patched.f``).

The correction replaces the RTOFS/ETSS-derived open-boundary water level
(``WLOBC``) with a value nudged toward real-time NOAA tide-gauge observations.
For every correction station the algorithm computes

    SWL_OBS(t)  = WL_OBS(t) - tide_prediction(t)          # subtidal residual
    AVGERR      = mean( SWL_OBS - WLOBC_source )           # constant bias
    err'(t)     = (SWL_OBS - WLOBC_source) - AVGERR        # time-varying part

and then, per open-boundary node,

    WLOBC(node,t) += WL_S_1*(AVGERR[sid1] + err'[sid1](t))     (WL_STA==1)
                  += WL_S_2*(AVGERR[sid2] + err'[sid2](t))     (also, WL_STA==2)

Fortran anchor map (line numbers in obc_schism_patched.f)
--------------------------------------------------------
  * OBC control-file reader ................ 595-715
  * base_date / day_start / day_end ........ 414-438
  * Station HC NetCDF reader ............... 442-519
  * tide prediction grid + NOS_PRD ......... 3652-3740
  * detide (SWL = OBS - PRED) .............. 4347-4405
  * backup-station substitution ............ 4411-4474
  * gap handling / QC pass ................. 4480-4535
  * AVGERR + residual + 6h ramp ............ 4536-4621
  * application to WLOBC ................... 4977-4995

OBC control-file (``OBC_CTL_FILE``) format
------------------------------------------
List-directed (whitespace-delimited) records.  Blank lines are transparent to
Fortran list-directed reads, so every section is preceded by exactly TWO
non-blank header lines that the reader discards (``obc_schism_patched.f``
670-689 / 699-708).  Ignoring blank lines the layout is::

    line 0            : NSTA  NOBC  NEOBC  DELT              (DELT in seconds)
    line 1,2          : <two header lines - discarded>
    NSTA station rows : SID  NOS_ID  NWS_ID  AGENCY_ID  DATUM
                        WL_FLAG  TS_FLAG  BACKUP_SID  GRIDID_STA  AS
    <two header lines - discarded>
    NOBC node rows    : GRIDID  IOBC  WL_STA  WL_SID_1  WL_S_1
                        WL_SID_2  WL_S_2  TS_STA  TS_SID_1  TS_S_1
                        TS_SID_2  TS_S_2
    <two header lines - discarded>
    NEOBC element rows: IDUMMY  JOBC

Station-block fields (``obc_schism_patched.f`` 679-680)::

    SID        int    running station id
    NOS_ID     str    CO-OPS/NOS id, matched against the HC NetCDF stationID
    NWS_ID     str    5-char NWS id (used to match BUFR RPID in production)
    AGENCY_ID  str    "NOS" / "USGS" ...
    DATUM      float  datum offset subtracted from raw gauge elevation
    WL_FLAG    int    0 => station IS a water-level correction source
                      (non-zero => skipped: AVGERR=0, err'=0)
    TS_FLAG    int    temperature/salinity flag (T/S port intentionally omitted)
    BACKUP_SID int    1-based station index used when this station has too few
                      obs (0 => no backup)
    GRIDID_STA int    1-based open-boundary node index whose WLOBC "source"
                      series this station is compared against
    AS         float  scale factor applied to a *backup* station's series

Node-block fields (``obc_schism_patched.f`` 691-693)::

    GRIDID     int    running node id
    IOBC       int    global grid-node index (only used for lon/lat/depth)
    WL_STA     int    0 => node untouched
                      1 => corrected by station WL_SID_1
                      2 => corrected by WL_SID_1 and WL_SID_2
    WL_SID_1   int    1-based station index (primary), weight WL_S_1
    WL_SID_2   int    1-based station index (secondary), weight WL_S_2
    TS_STA / TS_SID_* / TS_S_*   T/S analogues (parsed, not applied)

Station HC NetCDF schema (``nosofs.HC_NWLON.nc``, ``obc_schism_patched.f`` 442-519)
----------------------------------------------------------------------------------
Dimensions ``Station`` (NWLON_STA), ``Constituents`` (37), ``staID`` (char len).
Variables::

    stationID(Station, staID)          char   CO-OPS station ids
    constituentName(Constituents, ...)  char   constituent labels (unused here)
    amplitude(Constituents, Station)    float  harmonic amplitudes  -> tide_amp(N,K)
    phase(Constituents, Station)        float  harmonic phases (deg) -> tide_epoc(N,K)

The 37 constituent slots are the standard NOS/CO-OPS order (see ``SPEED_37``),
identical to the constituent order in ``tidal.py``.  The reader is orientation
robust: whichever axis matches the station count is treated as ``Station``.

Approximation ledger
--------------------
Everything above is a line-exact port EXCEPT the tide-prediction internals.
The operational ``NOS_PRD`` (``nos_ofs_tideprediction.f``) is not available
locally, so :func:`predict_station_tide` implements the standard Schureman
prediction ``amp * f * cos(speed*t + (V0+u) - phase)`` reusing the nodal-factor
machinery (``_nfacs`` / ``_gterms``) from :mod:`nos_utils.forcing.tidal`.  Nodal
corrections are evaluated once at ``base_date`` (the prediction reference
epoch).  This is the ONE documented approximation boundary.

Second (minor) deviation: the backup substitution reads the *raw* detided
subtidal of the backup station regardless of station ordering.  The Fortran
overwrites ``SWL_OBS`` in place while looping, so a backup whose index is lower
than the primary would (unintentionally) be read after correction; operational
ctl files do not rely on that state artifact, and this port preserves the raw
subtidal (``obc_schism_patched.f`` 4440-4454 vs 4606-4609).

Observation providers
---------------------
Production reads NCEP BUFR data tanks (``obc_schism_patched.f`` 3742-3940); that
ingest is the WCOSS2 path and is wired separately.  This module accepts pluggable
providers: :class:`ArrayObsProvider` (in-memory, for tests / offline),
:class:`FileObsProvider` (two-column ``days wl`` text files), and
:class:`CoopsApiProvider` (live CO-OPS datagetter; network only when called).
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

log = logging.getLogger(__name__)

# Missing-value sentinels used by the Fortran (obc_schism_patched.f 3695-3702, 4398).
_MISSING = -99999.9        # no observation
_OUT_OF_WINDOW = -9999.0   # obs falls outside the tide-prediction window
_QC_KEEP = -10.0           # gap pass keeps SWL greater than this (4493)
_QC_ABS = 3.0              # bias pass keeps |SWL| <= this (4556)
_BACKUP_MIN = 20           # NTR must exceed this else a backup is sought (4434/4480)
_RAMP_HOURS = 6.0          # ramp-to-zero point appended 6 h after last obs (4593)

# Standard NOS/CO-OPS 37-constituent speeds (deg / solar hour), in the SAME order
# as tidal.CNAME_MAP (M2,S2,N2,K1,M4,O1,...,K2,M8,MS4).  Part of the documented
# tide-prediction approximation.
SPEED_37 = np.array([
    28.9841042, 30.0000000, 28.4397295, 15.0410686, 57.9682084,
    13.9430356, 86.9523127, 44.0251729, 60.0000000, 57.4238337,
    28.5125831, 90.0000000, 27.9682084, 27.8953548, 16.1391017,
    29.4556253, 15.0000000, 14.4966939, 15.5854433,  0.5443747,
     0.0821373,  0.0410686,  1.0158958,  1.0980331, 13.4715145,
    13.3986609, 29.9589333, 30.0410667, 12.8542862, 14.9589314,
    31.0158958, 43.4761563, 29.5284789, 42.9271398, 30.0821373,
   115.9364169, 58.9841042,
], dtype=float)

NCON = 37


# ---------------------------------------------------------------------------
# OBC control-file data model + reader (obc_schism_patched.f 595-715)
# ---------------------------------------------------------------------------
@dataclass
class CorrectionStation:
    """One row of the station block (obc_schism_patched.f 679-680)."""

    sid: int
    nos_id: str
    nws_id: str
    agency_id: str
    datum: float
    wl_flag: int
    ts_flag: int
    backup_sid: int      # 1-based index into ObcCtl.stations, 0 = none
    gridid_sta: int      # 1-based index into ObcCtl.nodes (source WLOBC node)
    as_scale: float      # scale applied to this station when used as a backup


@dataclass
class BoundaryNode:
    """One row of the open-boundary-node block (obc_schism_patched.f 691-693)."""

    gridid: int
    iobc: int
    wl_sta: int          # 0 untouched / 1 single station / 2 two stations
    wl_sid_1: int        # 1-based station index (primary)
    wl_s_1: float
    wl_sid_2: int        # 1-based station index (secondary)
    wl_s_2: float
    ts_sta: int
    ts_sid_1: int
    ts_s_1: float
    ts_sid_2: int
    ts_s_2: float


@dataclass
class ObcCtl:
    """Parsed OBC control file."""

    nsta: int
    nobc: int
    neobc: int
    delt: float                                   # prediction/output step (seconds)
    stations: List[CorrectionStation] = field(default_factory=list)
    nodes: List[BoundaryNode] = field(default_factory=list)
    elements: List[Tuple[int, int]] = field(default_factory=list)


@dataclass
class WlCorrections:
    """Per-station water-level corrections on the ``zeta_time`` grid."""

    avgerr: np.ndarray          # (NSTA,) constant bias per station
    resid: np.ndarray           # (NSTA, NTMAX_WL) time-varying err'(t)
    zeta_time_days: np.ndarray  # (NTMAX_WL,) days since base_date


def _strip_token(tok: str) -> str:
    """Strip surrounding quotes from a list-directed string token."""
    tok = tok.strip()
    if len(tok) >= 2 and tok[0] in "'\"" and tok[-1] == tok[0]:
        tok = tok[1:-1]
    return tok


def parse_obc_ctl(path: Union[str, Path]) -> ObcCtl:
    """
    Parse an OBC control file into an :class:`ObcCtl`.

    Faithful to the Fortran list-directed reader (obc_schism_patched.f 598-715):
    blank lines are ignored and two header lines precede each data section.
    """
    text = Path(path).read_text()
    # Fortran list-directed reads skip blank records, so drop blank lines up-front
    # (obc_schism_patched.f 670-676 / 682-688 / 699-705).
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"OBC ctl file is empty: {path}")

    it = iter(range(len(lines)))

    def _next() -> str:
        try:
            return lines[next(it)]
        except StopIteration:
            raise ValueError(f"OBC ctl file ended prematurely: {path}")

    header = _next().split()
    if len(header) < 4:
        raise ValueError(
            f"OBC ctl header must be 'NSTA NOBC NEOBC DELT', got: {header!r}"
        )
    try:
        nsta, nobc, neobc = (int(header[0]), int(header[1]), int(header[2]))
        delt = float(header[3])
    except ValueError as exc:
        raise ValueError(f"Malformed OBC ctl header line: {header!r}") from exc

    _next(); _next()  # two discarded header lines before the station block
    stations: List[CorrectionStation] = []
    for n in range(nsta):
        f = _next().split()
        if len(f) < 10:
            raise ValueError(
                f"Station row {n + 1} needs 10 fields, got {len(f)}: {f!r}"
            )
        try:
            stations.append(CorrectionStation(
                sid=int(f[0]), nos_id=_strip_token(f[1]), nws_id=_strip_token(f[2]),
                agency_id=_strip_token(f[3]), datum=float(f[4]),
                wl_flag=int(f[5]), ts_flag=int(f[6]), backup_sid=int(f[7]),
                gridid_sta=int(f[8]), as_scale=float(f[9]),
            ))
        except ValueError as exc:
            raise ValueError(f"Malformed station row {n + 1}: {f!r}") from exc

    _next(); _next()  # two discarded header lines before the node block
    nodes: List[BoundaryNode] = []
    for n in range(nobc):
        f = _next().split()
        if len(f) < 12:
            raise ValueError(
                f"Node row {n + 1} needs 12 fields, got {len(f)}: {f!r}"
            )
        try:
            nodes.append(BoundaryNode(
                gridid=int(f[0]), iobc=int(f[1]), wl_sta=int(f[2]),
                wl_sid_1=int(f[3]), wl_s_1=float(f[4]),
                wl_sid_2=int(f[5]), wl_s_2=float(f[6]),
                ts_sta=int(f[7]), ts_sid_1=int(f[8]), ts_s_1=float(f[9]),
                ts_sid_2=int(f[10]), ts_s_2=float(f[11]),
            ))
        except ValueError as exc:
            raise ValueError(f"Malformed node row {n + 1}: {f!r}") from exc

    elements: List[Tuple[int, int]] = []
    try:
        _next(); _next()  # two discarded header lines before the element block
        for _ in range(neobc):
            f = _next().split()
            if len(f) >= 2:
                elements.append((int(f[0]), int(f[1])))
    except ValueError:
        # Element block is optional for the WL correction; tolerate absence.
        elements = []

    return ObcCtl(nsta=nsta, nobc=nobc, neobc=neobc, delt=delt,
                  stations=stations, nodes=nodes, elements=elements)


# ---------------------------------------------------------------------------
# Station HC reader + tide prediction (obc_schism_patched.f 442-519, 3652-3740)
# ---------------------------------------------------------------------------
def read_station_hc(hc_path: Union[str, Path], station_id: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read the 37-constituent amplitude/phase for ``station_id`` from a
    ``nosofs.HC_NWLON.nc`` file (obc_schism_patched.f 481-518, 3706-3732).

    Matching follows the Fortran prefix comparison (3707-3713): the file id
    (stripped) must start with the trimmed ``station_id``.

    Returns
    -------
    (amp, epoc) : two float arrays of length 37 (deg for ``epoc``).
    """
    from netCDF4 import Dataset, chartostring  # lazy: netCDF only when reading

    target = str(station_id).strip()
    with Dataset(str(hc_path), "r") as ds:
        ids = [str(s).strip() for s in np.atleast_1d(chartostring(ds.variables["stationID"][:]))]
        idx = None
        for k, fid in enumerate(ids):
            if fid[:len(target)] == target:  # Fortran BUFFER(1:L1) .EQ. BUFFER1(1:L1)
                idx = k
                break
        if idx is None:
            raise KeyError(
                f"Station {target!r} not found in HC file {hc_path} "
                f"({len(ids)} stations)"
            )
        amp = _orient_station_first(np.asarray(ds.variables["amplitude"][:], float), len(ids))
        pha = _orient_station_first(np.asarray(ds.variables["phase"][:], float), len(ids))

    return _pad37(amp[idx]), _pad37(pha[idx])


def _orient_station_first(arr: np.ndarray, nstation: int) -> np.ndarray:
    """Return ``arr`` shaped (nstation, nconstituents) regardless of file axis order."""
    if arr.ndim != 2:
        raise ValueError(f"HC amplitude/phase must be 2-D, got shape {arr.shape}")
    if arr.shape[0] == nstation:
        return arr
    if arr.shape[1] == nstation:
        return arr.T
    raise ValueError(
        f"Neither axis of HC array {arr.shape} matches station count {nstation}"
    )


def _pad37(row: np.ndarray) -> np.ndarray:
    """Take (or zero-pad) the first 37 constituent slots (obc_schism_patched.f 3729)."""
    out = np.zeros(NCON, dtype=float)
    n = min(NCON, row.shape[0])
    out[:n] = row[:n]
    return out


def predict_station_tide(
    hc_path: Union[str, Path],
    station_id: str,
    times_days: Sequence[float],
    base_date: datetime,
) -> np.ndarray:
    """
    Predict the tidal water level at a gauge from its 37 harmonic constants.

    Reproduces ``NOS_PRD`` (nos_ofs_tideprediction.f 196-500)::

        pred(t) = sum_j  amp_j * f_j * cos( speed_j * (t - Jan1) + VPU_j - kappa_j )

    with the ops epoch convention taken from the call site: ops calls
    ``equarg(37, IYRS, 1, 1, 365, ...)`` (line ~377) and sets
    ``jbase_date`` to January 1 of the start year (``yearb=IYRS; monthb=1``),
    so ``f`` and ``V0+u`` are evaluated at **Jan 1 00:00 of the prediction
    year** and the phase argument advances from that same origin
    (``FIRST=(jday0-jbase_date)*24``, line ~417).  The commented-out
    ``equarg(...,IMMS,IDDS,length,...)`` variant shows ops deliberately does
    not use the actual start date.

    APPROXIMATION BOUNDARY (see module docstring), now narrowed to two items:
    (1) ``V0+u`` / ``f`` come from ``tidal.py``'s Schureman implementation
    rather than ops' ``equarg.f`` (same theory, independent code); (2) ops
    accumulates through a 1024-bin quarter-wave cosine lookup
    (``XCOS``, ~0.09 deg bin, round-to-nearest) while this uses exact
    ``cos`` -- strictly more accurate.  Measured effect of the epoch
    convention alone before this was aligned: 6 mm peak, 3 mm RMS, and
    0.2 mm on the window mean (the only quantity ``AVGERR`` consumes).

    Parameters
    ----------
    times_days : days since ``base_date``.
    """
    amp, epoc = read_station_hc(hc_path, station_id)
    return _predict_from_hc(amp, epoc, times_days, base_date)


def _predict_from_hc(
    amp: np.ndarray, epoc: np.ndarray, times_days: Sequence[float], base_date: datetime
) -> np.ndarray:
    from .tidal import _dayjul, _gterms, _nfacs  # reuse Schureman nodal machinery

    # Ops epoch: equarg(37,IYRS,1,1,365) + jbase_date = Jan 1 00:00 of the
    # prediction year (nos_ofs_tideprediction.f ~300, ~377).  f and V0+u are
    # frozen there and the phase advances from that same origin, so the
    # nodal terms must NOT be re-evaluated at the cycle time.
    yr = float(base_date.year)
    dayj0 = _dayjul(yr, 1.0, 1.0)

    f37 = np.asarray(_nfacs(yr, dayj0, 0.0), dtype=float)          # nodal factors
    vpu37 = np.asarray(_gterms(yr, dayj0, 0.0, dayj0, 0.0), dtype=float)  # V0+u (deg)

    # FIRST=(jday0-jbase_date)*24 (line ~417): hours measured from Jan 1.
    jan1 = datetime(base_date.year, 1, 1)
    hours_base_from_jan1 = (base_date - jan1).total_seconds() / 3600.0

    t = np.asarray(times_days, dtype=float)
    hours = hours_base_from_jan1 + t * 24.0
    pred = np.zeros_like(t)
    for k in range(NCON):
        if amp[k] == 0.0:
            continue
        arg = np.deg2rad(SPEED_37[k] * hours + vpu37[k] - epoc[k])
        pred += amp[k] * f37[k] * np.cos(arg)
    return pred


# ---------------------------------------------------------------------------
# Interpolation helpers (utility.f LINEAR / LINEARARRAY)
# ---------------------------------------------------------------------------
def _lineararray(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    """
    Port of ``LINEARARRAY`` (utility.f 378-423): 1-D linear interpolation with
    constant-value extrapolation at both ends.  ``np.interp`` reproduces this
    exactly once ``xp`` is ascending (the Fortran reverses a descending input).
    """
    xp = np.asarray(xp, float)
    fp = np.asarray(fp, float)
    if xp.size >= 2 and xp[-1] < xp[0]:
        xp = xp[::-1]
        fp = fp[::-1]
    return np.interp(np.asarray(x, float), xp, fp)


# ---------------------------------------------------------------------------
# Observation providers
# ---------------------------------------------------------------------------
class ArrayObsProvider:
    """
    In-memory provider: ``{nos_id: (times_days, wl_meters)}``.

    Used for offline / WCOSS2 runs where observations are pre-fetched, and for
    tests.  Times are days since ``base_date``; missing rows return empty arrays.
    """

    def __init__(self, data: Dict[str, Tuple[Sequence[float], Sequence[float]]]):
        self._data = {str(k).strip(): v for k, v in data.items()}

    def get(self, station: CorrectionStation, base_date=None,
            day_start=None, day_end=None) -> Tuple[np.ndarray, np.ndarray]:
        t, w = self._data.get(str(station.nos_id).strip(), ([], []))
        return np.asarray(t, float), np.asarray(w, float)


class FileObsProvider:
    """
    Simple file provider: reads ``<obs_dir>/<nos_id><suffix>`` two-column text
    (``time_days  water_level``).  Offline WCOSS2-friendly, no network.
    """

    def __init__(self, obs_dir: Union[str, Path], suffix: str = ".obs"):
        self.obs_dir = Path(obs_dir)
        self.suffix = suffix

    def get(self, station: CorrectionStation, base_date=None,
            day_start=None, day_end=None) -> Tuple[np.ndarray, np.ndarray]:
        path = self.obs_dir / f"{str(station.nos_id).strip()}{self.suffix}"
        if not path.exists():
            return np.array([], float), np.array([], float)
        arr = np.loadtxt(path, ndmin=2)
        if arr.size == 0:
            return np.array([], float), np.array([], float)
        return arr[:, 0].astype(float), arr[:, 1].astype(float)


class CoopsApiProvider:
    """
    Live NOAA CO-OPS provider (6-minute water level, MSL datum, GMT, metric).

    Fetches from https://api.tidesandcurrents.noaa.gov/api/prod/datagetter using
    the standard library only.  Network access happens solely inside :meth:`get`;
    this class is never exercised by the test-suite.  In production the
    equivalent series comes from the NCEP BUFR data tanks
    (obc_schism_patched.f 3742-3940).
    """

    BASE_URL = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"

    def __init__(self, product: str = "water_level", datum: str = "MSL",
                 units: str = "metric", time_zone: str = "gmt",
                 application: str = "nos_utils", timeout: float = 60.0):
        self.product = product
        self.datum = datum
        self.units = units
        self.time_zone = time_zone
        self.application = application
        self.timeout = timeout

    def get(self, station: CorrectionStation, base_date: datetime,
            day_start: float, day_end: float) -> Tuple[np.ndarray, np.ndarray]:
        import json
        import urllib.parse
        import urllib.request

        begin = base_date + timedelta(days=float(day_start))
        end = base_date + timedelta(days=float(day_end))
        params = {
            "begin_date": begin.strftime("%Y%m%d %H:%M"),
            "end_date": end.strftime("%Y%m%d %H:%M"),
            "station": str(station.nos_id).strip(),
            "product": self.product,
            "datum": self.datum,
            "units": self.units,
            "time_zone": self.time_zone,
            "format": "json",
            "application": self.application,
        }
        url = f"{self.BASE_URL}?{urllib.parse.urlencode(params)}"
        with urllib.request.urlopen(url, timeout=self.timeout) as resp:
            payload = json.load(resp)

        rows = payload.get("data", []) or []
        times, wl = [], []
        for row in rows:
            try:
                value = float(row["v"])
            except (KeyError, ValueError, TypeError):
                continue  # CO-OPS flags gaps with an empty "v"
            ts = datetime.strptime(row["t"], "%Y-%m-%d %H:%M")
            times.append((ts - base_date).total_seconds() / 86400.0)
            wl.append(value)
        return np.asarray(times, float), np.asarray(wl, float)


# ---------------------------------------------------------------------------
# Core correction algorithm (obc_schism_patched.f 4347-4621)
# ---------------------------------------------------------------------------
def _detide(t_obs: np.ndarray, wl_obs: np.ndarray,
            time_prd: np.ndarray, wl_prd: np.ndarray) -> np.ndarray:
    """
    SWL = OBS - tidal prediction (obc_schism_patched.f 4352-4398).

    The prediction is linearly interpolated (2-point, subroutine LINEAR) from
    the prediction grid to each obs time; obs outside the window get the
    ``-9999`` sentinel, missing obs stay ``-99999.9``.
    """
    swl = np.full(t_obs.shape, _MISSING, dtype=float)
    t0, t1 = time_prd[0], time_prd[-1]
    for n in range(t_obs.size):
        if wl_obs[n] <= _MISSING:                        # 4353: skip missing obs
            continue
        t = t_obs[n]
        if t < t0 or t >= t1:                            # 4398: outside window
            swl[n] = _OUT_OF_WINDOW
            continue
        j = int(np.searchsorted(time_prd, t, side="right")) - 1
        x1, x2 = time_prd[j], time_prd[j + 1]
        y1, y2 = wl_prd[j], wl_prd[j + 1]
        y = y1 + (y2 - y1) / (x2 - x1) * (t - x1)        # 4390: LINEAR
        swl[n] = wl_obs[n] - y                           # 4391
    return swl


def _avgerr_ramp(rtime: np.ndarray, swl: np.ndarray,
                 zeta_time: np.ndarray, source: np.ndarray,
                 day_end: float) -> Tuple[float, np.ndarray]:
    """
    AVGERR + time-varying residual + 6-hour ramp for one station
    (obc_schism_patched.f 4480-4609).

    ``source`` is the WLOBC "source" series (RTOFS/ETSS) sampled at this
    station's ``GRIDID_STA`` node over ``zeta_time``.  Returns ``(avgerr,
    err'_on_zeta_time)``.
    """
    ntmax = zeta_time.size

    # -- 4491-4503: gap pass, keep only SWL > -10, compact -----------------
    keep = swl > _QC_KEEP
    r = rtime[keep]
    s = swl[keep]
    if r.size == 0:
        return 0.0, np.zeros(ntmax)
    time1, time2 = r[0], r[-1]

    # -- 4507-4533: edge handling (prepend/append to bracket zeta_time) ----
    o1: List[float] = []
    o2: List[float] = []
    if time1 > zeta_time[0]:                             # 4508
        o1.append(zeta_time[0]); o2.append(s[0])
    o1.extend(r.tolist()); o2.extend(s.tolist())         # 4513-4517
    for n1 in range(1, ntmax):                           # 4519-4526
        if zeta_time[n1] >= time2:
            o1.append(zeta_time[n1]); o2.append(s[-1])
            break
    o1 = np.asarray(o1, float)
    o2 = np.asarray(o2, float)
    time1, time2 = o1[0], o1[-1]

    # -- 4538-4545: zeta_time subset within [time1, time2] -> ONED3 --------
    oned3 = zeta_time[(zeta_time >= time1) & (zeta_time <= time2)]
    n0 = oned3.size
    if n0 < 1:                                           # 4546: Fortran STOP
        raise ValueError("real-time water-level data is insufficient")

    # -- 4554-4571: bias QC (|SWL| <= 3); regularize onto ONED3 ------------
    qc = np.abs(o2) <= _QC_ABS
    ntmp = int(qc.sum())
    if ntmp > 2:
        oned4 = _lineararray(oned3, o1[qc], o2[qc])
    else:
        oned4 = np.zeros(n0)                             # 4567-4570

    # -- 4573-4587: AVG = mean(regularized SWL - source) over matched grid -
    oned1 = np.zeros(n0)
    avg = 0.0
    for n in range(n0):
        match = np.where(np.abs(oned3[n] - zeta_time) <= 1.0e-10)[0]
        if match.size:                                   # 4576
            n1 = int(match[0])
            oned1[n] = oned4[n] - source[n1]             # 4577
            avg += oned1[n]
    avg = avg / n0 if n0 > 0 else 0.0                    # 4586
    avgerr = avg

    # -- 4588-4595: residual err' = (SWL - source) - AVG, then 6h ramp -----
    oned3b = oned3.astype(float).copy()
    oned4b = oned1 - avg
    if oned3b[-1] < day_end:                             # 4591
        oned3b = np.append(oned3b, oned3b[-1] + _RAMP_HOURS / 24.0)
        oned4b = np.append(oned4b, 0.0)

    # -- 4596-4609: interpolate err' onto zeta_time, zero beyond ramp ------
    resid = _lineararray(zeta_time, oned3b, oned4b)
    resid = np.where(zeta_time > oned3b[-1], 0.0, resid)
    return float(avgerr), resid


def compute_wl_corrections(
    ctl: ObcCtl,
    obs,
    wlobc_at_station_nodes: np.ndarray,
    zeta_time_days: Sequence[float],
    base_date: datetime,
    day_end: float,
    hc_path: Union[str, Path],
    day_start: Optional[float] = None,
) -> WlCorrections:
    """
    Compute per-station water-level corrections (obc_schism_patched.f 3652-4621).

    Parameters
    ----------
    ctl : parsed :class:`ObcCtl`.
    obs : observation provider exposing
        ``get(station, base_date, day_start, day_end) -> (times_days, wl)``
        (see :class:`ArrayObsProvider`).  A plain ``{nos_id: (times, wl)}`` dict
        is also accepted.
    wlobc_at_station_nodes : (NSTA, NTMAX_WL) source WLOBC sampled at each
        station's ``GRIDID_STA`` node (caller supplies
        ``WLOBC[station.gridid_sta - 1]``).
    zeta_time_days : (NTMAX_WL,) model boundary output times (days since base_date).
    day_end : model end time (days since base_date); controls the ramp point.
    hc_path : path to ``nosofs.HC_NWLON.nc``.
    day_start : model start (defaults to ``zeta_time_days[0]``); sets the
        prediction grid origin.
    """
    if isinstance(obs, dict):
        obs = ArrayObsProvider(obs)

    zt = np.asarray(zeta_time_days, dtype=float)
    ntmax = zt.size
    nsta = ctl.nsta
    delt = ctl.delt
    if day_start is None:
        day_start = float(zt[0])

    source = np.asarray(wlobc_at_station_nodes, dtype=float)
    if source.shape != (nsta, ntmax):
        raise ValueError(
            f"wlobc_at_station_nodes must be (NSTA={nsta}, NTMAX={ntmax}), "
            f"got {source.shape}"
        )

    # -- prediction time grid: day_start-2 .. day_end, step DELT (3655-3658) --
    nrec = int(round((day_end - day_start + 2.0) * 86400.0 / delt)) + 1
    time_prd = day_start - 2.0 + np.arange(nrec) * delt / 86400.0

    # -- ingest obs + detide every station (3690-3740, 4347-4405) ----------
    rtime: List[np.ndarray] = [np.array([], float)] * nsta
    wl_obs: List[np.ndarray] = [np.array([], float)] * nsta
    swl_obs: List[np.ndarray] = [np.array([], float)] * nsta
    ntr = np.zeros(nsta, dtype=int)

    for i, sta in enumerate(ctl.stations):
        t_obs, w_obs = obs.get(sta, base_date, day_start, day_end)
        t_obs = np.asarray(t_obs, float)
        w_obs = np.asarray(w_obs, float)
        ntr[i] = t_obs.size
        rtime[i] = t_obs
        wl_obs[i] = w_obs
        swl = np.full(t_obs.shape, _MISSING, dtype=float)
        if t_obs.size > 0:
            try:
                wl_prd = predict_station_tide(hc_path, sta.nos_id, time_prd, base_date)
                swl = _detide(t_obs, w_obs, time_prd, wl_prd)
            except KeyError:
                if sta.wl_flag == 0:                     # 3719-3723: Fortran STOP
                    raise
        swl_obs[i] = swl

    # -- backup substitution + AVGERR/ramp, sequential (4428-4621) ---------
    avgerr = np.zeros(nsta)
    resid = np.zeros((nsta, ntmax))
    stations = ctl.stations
    for i, sta in enumerate(stations):
        if sta.wl_flag != 0:                             # 4616-4620
            continue

        if ntr[i] <= _BACKUP_MIN:                        # 4434
            ibkp = sta.backup_sid
            if ibkp > 0:
                b = ibkp - 1
                if ntr[b] > _BACKUP_MIN:                 # 4437: first backup
                    scale = stations[b].as_scale
                    ntr[i] = ntr[b]
                    rtime[i] = rtime[b].copy()
                    wl_obs[i] = scale * wl_obs[b]
                    swl_obs[i] = scale * swl_obs[b]
                else:                                    # 4446: second backup hop
                    ibkp1 = stations[b].backup_sid
                    if ibkp1 > 0 and ntr[ibkp1 - 1] > _BACKUP_MIN:
                        b1 = ibkp1 - 1
                        scale = stations[b1].as_scale
                        ntr[i] = ntr[b1]
                        rtime[i] = rtime[b1].copy()
                        wl_obs[i] = scale * wl_obs[b1]
                        swl_obs[i] = scale * swl_obs[b1]

        if ntr[i] > _BACKUP_MIN:                         # 4480
            a, r = _avgerr_ramp(rtime[i], swl_obs[i], zt, source[i], day_end)
            avgerr[i] = a
            resid[i] = r
        # else: leave avgerr=0, resid=0 (4610-4614)

    return WlCorrections(avgerr=avgerr, resid=resid, zeta_time_days=zt)


def apply_wl_corrections(
    wlobc: np.ndarray, ctl: ObcCtl, corrections: WlCorrections
) -> np.ndarray:
    """
    Apply per-station corrections to the boundary WLOBC (obc_schism_patched.f 4977-4995).

    ``wlobc`` is (NOBC, NTMAX_WL).  For each node::

        WL_STA==1: WLOBC[i] += WL_S_1*(AVGERR[sid1] + err'[sid1])
        WL_STA==2: ... plus WL_S_2*(AVGERR[sid2] + err'[sid2])
        WL_STA==0: untouched

    Returns a corrected copy (input is not mutated).  The T/S analogues are
    commented out in the operational Fortran and are intentionally NOT applied.
    """
    out = np.array(wlobc, dtype=float, copy=True)
    avgerr = corrections.avgerr
    resid = corrections.resid
    if out.shape[0] != len(ctl.nodes):
        raise ValueError(
            f"wlobc first dim {out.shape[0]} != NOBC {len(ctl.nodes)}"
        )

    for i, node in enumerate(ctl.nodes):
        id1, id2 = node.wl_sid_1, node.wl_sid_2
        sc1, sc2 = node.wl_s_1, node.wl_s_2
        if node.wl_sta == 1:
            if id1 > 0:
                j = id1 - 1
                out[i] += sc1 * (avgerr[j] + resid[j])
        elif node.wl_sta == 2:
            if id1 > 0 and id2 > 0:
                j1, j2 = id1 - 1, id2 - 1
                out[i] += (sc1 * (avgerr[j1] + resid[j1])
                           + sc2 * (avgerr[j2] + resid[j2]))
        # wl_sta == 0 -> untouched
    return out
