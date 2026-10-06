"""
GFS 0.25 deg GRIB2 to ADCIRC OWI NetCDF meteorology (NWS=14) for STOFS-2D-Global.

Adapted from Zach Cobell's StofsWorkflow (oceanmodeling/nos-workflow, branch
zcobell/stofs_2d_global): met/gfs_file_selector.py, met/local_source.py,
met/met_forcing.py (local mode only), met/manifest.py and
adcircmodel._prepare_nws14_netcdf / _write_nws14_nc, with his June fixes 4 and 5
(COMINgfs passed through; an absent variable no longer crashes the conversion
and fort.22 lists icec only when ice is present).

Flow: select GFS files (nowcast: minimum lead time per valid hour across cycles;
forecast: one cycle, hourly to f120 then 3-hourly), copy them from
``<COMINgfs>/gfs.YYYYMMDD/HH/atmos/gfs.tHHz.pgrb2.0p25.fFFF`` with an optional wgrib2
variable subset, read with xarray/cfgrib, interpolate linearly to hourly steps and
write fort.221.nc (pressure), fort.222.nc (winds), fort.225.nc (ice) and the
fort.22 descriptor.

xarray, cfgrib and eccodes are imported when process() starts (a missing one fails
before any GRIB file is copied).

Public API::

    proc = AdcircMetProcessor(comin_gfs, variables)
    result = proc.process(start, end, out_dir, phase="nowcast")
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Set, Tuple

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_VARIABLES = [
    "PRMSL:mean sea level",
    "UGRD:10 m above ground",
    "VGRD:10 m above ground",
    "ICEC:surface",
]

# GRIB shortName filters per output key, in the order he reads them. MJ (10/05/26)
_VAR_MAP = {
    "prmsl": {"shortName": "prmsl"},
    "10u": {"shortName": "10u"},
    "10v": {"shortName": "10v"},
    "ci": {"shortName": "ci"},
}

PHASES = ("nowcast", "forecast")


class MetForcingError(Exception):
    """Base exception for ADCIRC meteorological forcing errors."""


class FileNotAvailableError(MetForcingError):
    """A required GRIB2 file was not found."""


class InsufficientDataError(MetForcingError):
    """Not enough GFS cycles or forecast hours to cover the window."""


class MetDependencyError(MetForcingError):
    """xarray, cfgrib or eccodes is missing or unusable."""


@dataclass(frozen=True)
class GfsFileRequest:
    cycle_time: datetime
    forecast_hour: int
    valid_time: datetime
    source_path: str


@dataclass(frozen=True)
class AcquiredFile:
    request: GfsFileRequest
    local_path: Path
    source_location: str
    checksum: str
    file_size: int
    variables_extracted: List[str]


def compute_file_checksum(filepath: Path, algorithm: str = "md5") -> str:
    h = hashlib.new(algorithm)
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


class GfsFileSelector:
    """Selects GFS GRIB2 files for nowcast and forecast windows.

    Nowcast uses minimum-tau selection across cycles, forecast one cycle.
    Selection is constrained by availability when it is given.
    """

    GFS_CYCLE_HOURS: ClassVar[List[int]] = [0, 6, 12, 18]
    GFS_CYCLE_INTERVAL = timedelta(hours=6)
    GFS_HOURLY_LIMIT: ClassVar[int] = 120  # hourly through f120, 3-hourly after

    @staticmethod
    def next_gfs_hour(tau: int) -> int:
        if tau + 1 <= GfsFileSelector.GFS_HOURLY_LIMIT:
            return tau + 1
        return tau + 3 - (tau % 3) if tau % 3 else tau + 3

    @staticmethod
    def is_valid_gfs_hour(tau: int) -> bool:
        if tau <= GfsFileSelector.GFS_HOURLY_LIMIT:
            return True
        return tau % 3 == 0

    @staticmethod
    def build_gfs_path(cycle_time: datetime, forecast_hour: int) -> str:
        date_str = cycle_time.strftime("%Y%m%d")
        hour_str = "{:02d}".format(cycle_time.hour)
        fhr_str = "{:03d}".format(forecast_hour)
        return "gfs.{}/{}/atmos/gfs.t{}z.pgrb2.0p25.f{}".format(
            date_str, hour_str, hour_str, fhr_str)

    @staticmethod
    def most_recent_cycle(dt: datetime) -> datetime:
        cycle_hour = (
            max(h for h in GfsFileSelector.GFS_CYCLE_HOURS if h <= dt.hour)
            if dt.hour >= GfsFileSelector.GFS_CYCLE_HOURS[0]
            else GfsFileSelector.GFS_CYCLE_HOURS[-1]
        )
        if cycle_hour > dt.hour:
            dt = dt - timedelta(days=1)
        return dt.replace(hour=cycle_hour, minute=0, second=0, microsecond=0)

    @staticmethod
    def get_candidate_cycles(start: datetime, end: datetime) -> List[datetime]:
        """Cycles from one before the window start through the last one at or before end."""
        first_cycle = GfsFileSelector.most_recent_cycle(start)
        prev_cycle = first_cycle - GfsFileSelector.GFS_CYCLE_INTERVAL
        last_cycle = GfsFileSelector.most_recent_cycle(end)
        cycles = []
        current = prev_cycle
        while current <= last_cycle:
            cycles.append(current)
            current += GfsFileSelector.GFS_CYCLE_INTERVAL
        return cycles

    def select_nowcast_files(
        self,
        nowcast_start: datetime,
        nowcast_end: datetime,
        time_step_hours: int = 1,
        availability: Optional[Dict[datetime, Set[int]]] = None,
    ) -> List[GfsFileRequest]:
        requests = []
        current_time = nowcast_start
        while current_time <= nowcast_end:
            request = self._select_best_file(current_time, availability)
            if request is None:
                raise InsufficientDataError(
                    "No GFS data available for valid time {}. Cannot proceed with "
                    "nowcast.".format(current_time.isoformat()))
            requests.append(request)
            current_time += timedelta(hours=time_step_hours)
        return requests

    def select_forecast_files(
        self,
        forecast_start: datetime,
        forecast_end: datetime,
        availability: Optional[Dict[datetime, Set[int]]] = None,
    ) -> List[GfsFileRequest]:
        cycle = self.most_recent_cycle(forecast_start)
        if availability is not None and (
            cycle not in availability or not availability[cycle]
        ):
            raise InsufficientDataError(
                "GFS cycle {} has no data available in the data source. Cannot "
                "proceed with forecast.".format(cycle.isoformat()))

        requests = []
        tau = int((forecast_start - cycle).total_seconds() / 3600)
        while True:
            current_time = cycle + timedelta(hours=tau)
            if current_time > forecast_end:
                break
            if availability is not None and tau not in availability.get(cycle, set()):
                raise InsufficientDataError(
                    "GFS forecast hour f{:03d} from cycle {} is not available. "
                    "Cannot proceed with forecast.".format(tau, cycle.isoformat()))
            requests.append(GfsFileRequest(
                cycle_time=cycle,
                forecast_hour=tau,
                valid_time=current_time,
                source_path=self.build_gfs_path(cycle, tau),
            ))
            tau = self.next_gfs_hour(tau)

        if not requests:
            raise InsufficientDataError(
                "No GFS forecast files selected for {} to {}".format(
                    forecast_start.isoformat(), forecast_end.isoformat()))
        log.info("Forecast: cycle=%s, %d files from f%03d to f%03d",
                 cycle.isoformat(), len(requests),
                 requests[0].forecast_hour, requests[-1].forecast_hour)
        return requests

    def _select_best_file(
        self,
        valid_time: datetime,
        availability: Optional[Dict[datetime, Set[int]]],
    ) -> Optional[GfsFileRequest]:
        best_cycle = self.most_recent_cycle(valid_time)
        best_tau = int((valid_time - best_cycle).total_seconds() / 3600)

        if availability is None:
            return GfsFileRequest(
                cycle_time=best_cycle,
                forecast_hour=best_tau,
                valid_time=valid_time,
                source_path=self.build_gfs_path(best_cycle, best_tau),
            )

        candidate_cycle = best_cycle
        max_lookback = 4  # up to 4 prior cycles (24 hours)
        for _ in range(max_lookback + 1):
            tau = int((valid_time - candidate_cycle).total_seconds() / 3600)
            if tau < 0:
                break
            if tau in availability.get(candidate_cycle, set()):
                return GfsFileRequest(
                    cycle_time=candidate_cycle,
                    forecast_hour=tau,
                    valid_time=valid_time,
                    source_path=self.build_gfs_path(candidate_cycle, tau),
                )
            candidate_cycle -= self.GFS_CYCLE_INTERVAL
        return None


def adcirc_met_windows(
    cycle_time: datetime,
    nowcast_hours: float = 6.0,
    forecast_hours: float = 180.0,
    spinup_days: Optional[float] = None,
) -> Dict[str, datetime]:
    """Window bounds as his StofsConfig.get_forecast_times, plus an optional spin-up start.

    ``nowcast_start`` is the nowcast window start; with ``spinup_days`` set it is
    instead ``cycle_time - spinup_days`` (the cold-start nowcast his
    _resolve_hotstart builds). MJ (10/05/26)
    """
    if spinup_days is not None:
        nowcast_start = cycle_time - timedelta(days=spinup_days)
    else:
        nowcast_start = cycle_time - timedelta(hours=nowcast_hours)
    return {
        "cycle_time": cycle_time,
        "nowcast_start": nowcast_start,
        "forecast_end": cycle_time + timedelta(hours=forecast_hours),
    }


def match_inventory(lines: List[str], variables: List[str]) -> List[str]:
    """wgrib2 -s inventory lines containing any variable string (substring match, as his code)."""
    return [line for line in lines if any(var in line for var in variables)]


class LocalGfsSource:
    """Copies GFS GRIB2 files from COMINgfs, optionally subsetting with wgrib2."""

    def __init__(self, comin_gfs, variables: List[str],
                 subset: bool = True, wgrib2_path: Optional[str] = None):
        if not str(comin_gfs):
            raise ValueError("comin_gfs is empty: local GFS source needs the COMINgfs path")
        self._comin_gfs = Path(comin_gfs)
        self._subset = subset
        self._wgrib2 = wgrib2_path or shutil.which("wgrib2")
        self._variables = list(variables)

    def check_availability(self, cycle_times: List[datetime],
                           max_forecast_hour: int) -> Dict[datetime, Set[int]]:
        availability = {}  # type: Dict[datetime, Set[int]]
        for cycle in cycle_times:
            hours = set()  # type: Set[int]
            for fhr in range(max_forecast_hour + 1):
                if (self._comin_gfs / GfsFileSelector.build_gfs_path(cycle, fhr)).exists():
                    hours.add(fhr)
            availability[cycle] = hours
        return availability

    def acquire(self, file_requests: List[GfsFileRequest],
                output_dir: Path) -> List[AcquiredFile]:
        output_dir.mkdir(parents=True, exist_ok=True)
        results = []
        for request in file_requests:
            src = self._comin_gfs / request.source_path
            dst = output_dir / "gfs_{:%Y%m%d%H}.grb2".format(request.valid_time)
            if not src.exists():
                raise FileNotAvailableError("GFS file not found: {}".format(src))
            if self._subset and self._wgrib2:
                self._copy_with_subset(src, dst)
                extracted = self._variables
            else:
                shutil.copy2(src, dst)
                extracted = ["all"]
            results.append(AcquiredFile(
                request=request,
                local_path=dst,
                source_location=str(src),
                checksum=compute_file_checksum(dst),
                file_size=dst.stat().st_size,
                variables_extracted=list(extracted),
            ))
            log.info("Acquired local file: %s -> %s", src, dst)
        return results

    def _copy_with_subset(self, src: Path, dst: Path) -> None:
        inv = subprocess.run([self._wgrib2, str(src), "-s"],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             universal_newlines=True, check=True)
        matching = match_inventory(inv.stdout.splitlines(), self._variables)
        if not matching:
            log.warning("No matching variables found in %s, copying full file", src)
            shutil.copy2(src, dst)
            return
        res = subprocess.run([self._wgrib2, str(src), "-i", "-grib", str(dst)],
                             input="\n".join(matching) + "\n",
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             universal_newlines=True, check=False)
        if res.returncode != 0:
            raise RuntimeError("wgrib2 subsetting failed for {}: {}".format(src, res.stderr))


def check_dependencies():
    """Fail fast (before any GRIB copy) when xarray, cfgrib or ecCodes is unusable."""
    return _import_xarray()


def _import_xarray():
    """Import xarray and cfgrib (with eccodes) or fail with a clear message."""
    try:
        import xarray as xr
    except ImportError as exc:
        raise MetDependencyError(
            "xarray is required for ADCIRC OWI conversion (pip install xarray netCDF4)"
        ) from exc
    try:
        import cfgrib  # noqa: F401
    except Exception as exc:
        raise MetDependencyError(
            "cfgrib with a working ecCodes library is required to read GFS GRIB2 "
            "(pip install cfgrib eccodes; on a node set ECCODES_DIR or load the "
            "eccodes module): {}".format(exc)
        ) from exc
    try:
        import eccodes  # noqa: F401
    except Exception as exc:
        raise MetDependencyError(
            "the eccodes Python module / ecCodes library is not usable "
            "(pip install eccodes; set ECCODES_DIR or load the eccodes module): {}".format(exc)
        ) from exc
    return xr


def _write_nws14_nc(output_path: Path, var_data: dict, steps: np.ndarray,
                    lats: np.ndarray, lons: np.ndarray) -> None:
    """Write one NWS=14 NetCDF file with G-STOFS dimension/variable names."""
    import xarray as xr

    out_ds = xr.Dataset()
    out_ds.coords["record"] = ("record", np.arange(len(steps)))
    out_ds.coords["grid_yt"] = ("grid_yt", lats)
    out_ds.coords["grid_xt"] = ("grid_xt", lons)
    for var_name, src_ds in var_data.items():
        src_var = next(iter(src_ds.data_vars))
        data = src_ds[src_var].values
        out_ds[var_name] = (("record", "grid_yt", "grid_xt"), data)
    out_ds.to_netcdf(output_path)
    out_ds.close()
    log.info("Wrote %s (%d bytes)", output_path.name, output_path.stat().st_size)


def write_owi_netcdf(grib2_files: List[Path], out_dir: Path) -> dict:
    """Convert per-timestep GRIB2 files to fort.221/222/225.nc and fort.22.

    Returns ``{"files": {name: Path}, "fort22": Path, "hourly_times": ndarray,
    "variables": [keys present]}``. Pressure and both wind components are required;
    ice is optional.
    """
    xr = _import_xarray()
    out_dir = Path(out_dir)
    grib2_files = sorted(Path(p) for p in grib2_files)
    if not grib2_files:
        raise InsufficientDataError("No GRIB2 files to convert")

    datasets = {}
    for var_key, filter_keys in _VAR_MAP.items():
        file_datasets = []
        for grib_path in grib2_files:
            ds = xr.open_dataset(
                grib_path,
                engine="cfgrib",
                backend_kwargs={"indexpath": "", "filter_by_keys": filter_keys},
            )
            # one timestep per file, possibly from different cycles: use valid_time only. MJ (10/05/26)
            ds = ds.expand_dims("valid_time")
            ds = ds.drop_vars([c for c in ("step", "time") if c in ds.coords])
            file_datasets.append(ds)
        if file_datasets:
            datasets[var_key] = xr.concat(file_datasets, dim="valid_time", coords="minimal")

    # June fix 5: drop variables with no data (e.g. ICEC absent from the source files)
    datasets = {k: v for k, v in datasets.items() if v.data_vars}  # MJ (10/05/26)

    missing = [k for k in ("prmsl", "10u", "10v") if k not in datasets]
    if missing:
        raise InsufficientDataError(
            "GRIB2 files have no data for required variable(s): {}".format(", ".join(missing)))

    one_hour = np.timedelta64(1, "h")
    ref_ds = next(iter(datasets.values()))
    times = ref_ds.coords["valid_time"].values
    hourly_times = np.arange(times[0], times[-1] + one_hour, one_hour)

    for var_key, var_ds in datasets.items():
        datasets[var_key] = var_ds.interp(valid_time=hourly_times, method="linear")

    lats = ref_ds.coords["latitude"].values
    lons = ref_ds.coords["longitude"].values
    log.info("Interpolated forcing to %d hourly steps (%d x %d grid)",
             len(hourly_times), len(lats), len(lons))

    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    p221 = out_dir / "fort.221.nc"
    _write_nws14_nc(p221, {"pressfc": datasets["prmsl"]}, hourly_times, lats, lons)
    files["fort.221.nc"] = p221
    p222 = out_dir / "fort.222.nc"
    _write_nws14_nc(p222, {"ugrd10m": datasets["10u"], "vgrd10m": datasets["10v"]},
                    hourly_times, lats, lons)
    files["fort.222.nc"] = p222
    if "ci" in datasets:
        p225 = out_dir / "fort.225.nc"
        _write_nws14_nc(p225, {"icec": datasets["ci"]}, hourly_times, lats, lons)
        files["fort.225.nc"] = p225

    present = list(datasets.keys())
    for ds in datasets.values():
        ds.close()

    fort22 = out_dir / "fort.22"
    write_fort22(fort22, ice="ci" in present)
    return {"files": files, "fort22": fort22, "hourly_times": hourly_times,
            "variables": present}


def fort22_lines(ice: bool = True) -> List[str]:
    lines = ["record", "none", "none", "grid_xt", "grid_xt", "grid_yt", "grid_yt",
             "pressfc", "ugrd10m", "vgrd10m"]
    if ice:
        lines.append("icec")
    return lines


def write_fort22(path: Path, ice: bool = True) -> Path:
    """Write the fort.22 descriptor; icec is listed only when ice is present."""
    Path(path).write_text("\n".join(fort22_lines(ice)) + "\n")
    log.info("Generated fort.22 descriptor")
    return Path(path)


@dataclass
class AdcircMetResult:
    files: Dict[str, Path]
    fort22: Path
    manifest: Optional[Path] = None
    grib_files: List[Path] = field(default_factory=list)
    hourly_times: Optional[np.ndarray] = None
    variables: List[str] = field(default_factory=list)
    phase: str = "nowcast"
    cycle_time: Optional[datetime] = None
    n_records: int = 0
    warnings: List[str] = field(default_factory=list)


class AdcircMetProcessor:
    """GFS to ADCIRC NWS=14 forcing from a local COMINgfs tank.

    Args:
        comin_gfs: root of the GFS tank (the directory holding ``gfs.YYYYMMDD``).
        variables: wgrib2 match strings for the subset (default PRMSL, 10 m winds, ICEC).
        subset: subset with wgrib2 when it is on PATH (else copy whole files).
        wgrib2_path: explicit wgrib2 binary.
    """

    def __init__(self, comin_gfs, variables: Optional[List[str]] = None,
                 subset: bool = True, wgrib2_path: Optional[str] = None,
                 meteo_subdir: str = "forcing_data/meteo"):
        self.comin_gfs = comin_gfs
        self.variables = list(variables) if variables else list(DEFAULT_VARIABLES)
        self._source = LocalGfsSource(comin_gfs, self.variables, subset, wgrib2_path)
        self._selector = GfsFileSelector()
        self.meteo_subdir = meteo_subdir

    def select_files(self, start: datetime, end: datetime,
                     phase: str = "nowcast") -> List[GfsFileRequest]:
        if phase not in PHASES:
            raise ValueError("phase must be one of {}, got {!r}".format(PHASES, phase))
        max_tau = int((end - start).total_seconds() / 3600) + 6  # margin for fallback cycles
        cycles = GfsFileSelector.get_candidate_cycles(start, end)
        availability = self._source.check_availability(cycles, max_tau)
        if phase == "nowcast":
            return self._selector.select_nowcast_files(start, end, availability=availability)
        return self._selector.select_forecast_files(start, end, availability=availability)

    def process(self, start: datetime, end: datetime, out_dir: Path,
                phase: str = "nowcast") -> AdcircMetResult:
        check_dependencies()  # before the GRIB copies, so a missing package fails in seconds. MJ (10/05/26)
        out_dir = Path(out_dir)
        meteo_dir = out_dir / self.meteo_subdir
        meteo_dir.mkdir(parents=True, exist_ok=True)

        requests = self.select_files(start, end, phase)
        acquired = self._source.acquire(requests, meteo_dir)

        if phase == "nowcast":
            cycle_time = acquired[-1].request.cycle_time if acquired else end
        else:
            cycle_time = acquired[0].request.cycle_time if acquired else start

        manifest_path = meteo_dir / "met_forcing_manifest.json"
        self._write_manifest(manifest_path, acquired, phase, cycle_time)

        conv = write_owi_netcdf([a.local_path for a in acquired], out_dir)
        return AdcircMetResult(
            files=conv["files"],
            fort22=conv["fort22"],
            manifest=manifest_path,
            grib_files=sorted(a.local_path for a in acquired),
            hourly_times=conv["hourly_times"],
            variables=conv["variables"],
            phase=phase,
            cycle_time=cycle_time,
            n_records=len(conv["hourly_times"]),
        )

    @staticmethod
    def _write_manifest(path: Path, acquired: List[AcquiredFile], phase: str,
                        cycle_time: datetime) -> None:
        data = {
            "schema_version": "0.1.0",
            "source_mode": "local",
            "source_model": "gfs",
            "acquisition_time": datetime.now().isoformat(),
            "cycle_time": cycle_time.isoformat(),
            "simulation_phase": phase,
            "files_requested": len(acquired),
            "files_acquired": len(acquired),
            "forcing": {"atmospheric": {
                "source_model": "gfs",
                "source_cycle": cycle_time.isoformat(),
                "nfiles": len(acquired),
                "files": [{
                    "filename": str(f.local_path),
                    "source_location": f.source_location,
                    "valid_time": f.request.valid_time.isoformat(),
                    "forecast_hour": f.request.forecast_hour,
                    "checksum": f.checksum,
                    "file_size": f.file_size,
                    "variables_extracted": f.variables_extracted,
                } for f in acquired],
            }},
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(path), "w") as fh:
            json.dump(data, fh, indent=2)


# --- Operational (stofs.v3.1.5) surface forcing from the GFS native-grid sfcf files ---------------
# Replicates exstofs_2d_glo_gfs.sh / stofs_2d_glo_surface_forcing.sh / stofs_2d_glo_getges.sh:
# one pass per record, ncks -v + ncecat + ncwa -a time as netCDF4 calls. MJ (10/06/26)

SFC_SEGMENTS = ("ncst", "fcst1", "fcst2")
SFC_OUTPUTS = (("221", ("pressfc",)), ("222", ("ugrd10m", "vgrd10m")), ("225", ("icec",)))
_SFC_TAGS = {"221": "pressfc", "222": "uvgrd10m", "225": "icec"}  # the ops intermediate file names
_GETGES_FHEND = 384


def sfcf_path(comin_gfs, cycle: datetime, fhr: int) -> Path:
    return Path(comin_gfs) / "gfs.{:%Y%m%d}".format(cycle) / "{:02d}".format(cycle.hour) / "atmos" / \
        "gfs.t{:02d}z.sfcf{:03d}.nc".format(cycle.hour, fhr)


def getges_sfcf(comin_gfs, valid: datetime) -> Path:
    """getges.sh -t sfgges -n gfs: the first readable file walking back one hour of lead at a time.

    The script starts at fhbeg=max(NHOUR valid, 0); a valid time in the past, as in ops, gives 0.
    A missing f000 therefore falls back to the previous cycle's f006. MJ (10/06/26)
    """
    for fh in range(_GETGES_FHEND + 1):
        cand = sfcf_path(comin_gfs, valid - timedelta(hours=fh), fh)
        if os.access(str(cand), os.R_OK):
            return cand
    raise FileNotAvailableError(
        "FATAL ERROR: no GFS sfcf file for valid {:%Y%m%d%H} under {} (searched f000-f{})".format(
            valid, comin_gfs, _GETGES_FHEND))


def sfcf_valid_times(cycle: datetime, segment: str, start: Optional[datetime] = None) -> List[datetime]:
    """Record times of the ops calls: surface1 ncst (start..cycle) and fcst1 (cycle..+120 h) hourly,
    surface3 fcst2 (+120 h..+180 h) every 3 h. ``start`` is the ncst time_beg (multistart)."""
    if segment not in SFC_SEGMENTS:
        raise ValueError("segment must be one of {}, got {!r}".format(SFC_SEGMENTS, segment))
    if segment == "ncst":
        beg, end, step = start or cycle - timedelta(hours=6), cycle, 1
    elif segment == "fcst1":
        beg, end, step = cycle, cycle + timedelta(hours=120), 1
    else:
        beg, end, step = cycle + timedelta(hours=120), cycle + timedelta(hours=180), 3
    if beg > end:
        raise MetForcingError("sfcf window starts after it ends: {} > {}".format(beg, end))
    out = []
    while beg <= end:
        out.append(beg)
        beg += timedelta(hours=step)
    return out


def _nco_stamp(t: datetime) -> str:
    return "{:%a %b} {:>2} {:%H:%M:%S %Y}".format(t, t.day, t)


def _history(valids: List[datetime], names: Tuple[str, ...], num: str, tag: str, now: datetime) -> str:
    ymdh = ["{:%Y%m%d%H}".format(v) for v in valids]
    ncks = "ncks -v time,grid_xt,lon,grid_yt,lat,{} swnd.{} {}.{}.nc".format(",".join(names), ymdh[0], tag, ymdh[0])
    ncecat = "ncecat {} tmp.{}.nc".format(" ".join("{}.{}.nc".format(tag, y) for y in ymdh), num)
    ncwa = "ncwa -a time tmp.{0}.nc fort.{0}.nc".format(num)
    return "{0}: {1}\n{0}: {2}\n{0}: {3}".format(_nco_stamp(now), ncwa, ncecat, ncks)


def _wait_for(path: Path, deadline: float, poll_s: float, sleep, clock) -> None:
    """The ops ``until [ -s file ]`` loop, bounded by a shared deadline."""
    while not (path.is_file() and path.stat().st_size > 0):
        left = deadline - clock()
        if left <= 0:
            raise FileNotAvailableError("GFS file did not appear before the wait expired: {}".format(path))
        log.info("%s not there yet, waiting", path)
        sleep(min(poll_s, left))


def build_sfcf_forcing(comin_gfs, cycle: datetime, segment: str, out_dir, prefix: str = "stofs_2d_glo",
                       start: Optional[datetime] = None, wait_s: float = 0.0, poll_s: float = 10.0,
                       sleep=time.sleep, clock=time.monotonic) -> Dict[str, Path]:
    """Write ``<prefix>_<segment>.{221,222,225}.nc`` from the GFS sfcf files, as the ops GFS_NCST/FCST1/FCST2 jobs.

    Valid times at or before ``cycle`` use getges; later ones the current cycle's own sfcf file, waited for
    up to ``wait_s`` seconds in total. Output files appear under their final names only when all three are
    complete, 221 last (the ops skip test is on 221). Records are copied one at a time.
    """
    import netCDF4 as nc

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    valids = sfcf_valid_times(cycle, segment, start)
    deadline = clock() + wait_s
    final = {n: out_dir / "{}_{}.{}.nc".format(prefix, segment, n) for n, _ in SFC_OUTPUTS}
    part = {n: p.with_name(p.name + ".partial") for n, p in final.items()}
    outs = {}  # type: Dict[str, object]
    used = []  # type: List[Tuple[datetime, Path]]
    try:
        for rec, valid in enumerate(valids):
            if valid <= cycle:
                src = getges_sfcf(comin_gfs, valid)
            else:
                src = sfcf_path(comin_gfs, cycle, int((valid - cycle).total_seconds() // 3600))
                _wait_for(src, deadline, poll_s, sleep, clock)
            used.append((valid, src))
            log.info("record %d valid %s <- %s", rec, valid, src)
            with nc.Dataset(str(src)) as ds:
                ds.set_auto_maskandscale(False)
                if rec == 0:
                    for n, names in SFC_OUTPUTS:
                        outs[n] = _open_sfcf_output(nc, part[n], ds, names)
                for n, names in SFC_OUTPUTS:
                    for v in ("lat", "lon") + names:
                        if ds.variables[v].shape[-2:] != (len(outs[n].dimensions["grid_yt"]),
                                                          len(outs[n].dimensions["grid_xt"])):  # ncecat needs equal shapes
                            raise MetForcingError("{}: {} has a different grid from the first record".format(src, v))
                        outs[n].variables[v][rec] = ds.variables[v][0] if v in names else ds.variables[v][:]
        now = datetime.now()
        for n, names in SFC_OUTPUTS:
            outs[n].setncattr("history", _history(valids, names, n, _SFC_TAGS[n], now))
            outs[n].close()
        outs = {}
        for n in ("225", "222", "221"):
            os.replace(str(part[n]), str(final[n]))
    finally:
        for o in outs.values():
            o.close()
    _write_sfcf_manifest(out_dir / "{}_{}.sfcf_manifest.json".format(prefix, segment), cycle, segment, used)
    return final


def _open_sfcf_output(nc, path: Path, src, names: Tuple[str, ...]):
    """One output file with the ncecat+ncwa layout: record unlimited, lat/lon/data (record, y, x), scalar time."""
    out = nc.Dataset(str(path), "w", format="NETCDF4_CLASSIC")
    ny, nx = len(src.dimensions["grid_yt"]), len(src.dimensions["grid_xt"])
    for k, v in src.__dict__.items():
        out.setncattr(k, v)
    out.createDimension("grid_xt", nx)
    out.createDimension("grid_yt", ny)
    out.createDimension("record", None)
    chunks = (1, ny, nx)

    def clone(name, dims, **kw):
        sv = src.variables[name]
        fill = sv.getncattr("_FillValue") if "_FillValue" in sv.ncattrs() else None
        ov = out.createVariable(name, sv.dtype, dims, fill_value=fill, **kw)
        for a in sv.ncattrs():
            if a != "_FillValue":
                ov.setncattr(a, sv.getncattr(a))
        return ov

    for c in ("grid_xt", "grid_yt"):
        clone(c, (c,))[:] = src.variables[c][:]
    for v in sorted(names + ("lat", "lon", "time")):  # ncecat orders the non-coordinate variables alphabetically
        scalar, data = v == "time", v not in ("lat", "lon", "time")
        kw = {} if scalar else {"chunksizes": chunks}
        if data:
            kw.update(zlib=True, complevel=1, shuffle=True)
        ov = clone(v, () if scalar else ("record", "grid_yt", "grid_xt"), **kw)
        if v not in ("lat", "lon"):
            cm = getattr(src.variables[v], "cell_methods", "")
            ov.setncattr("cell_methods", (cm + " " if cm else "") + "time: mean")  # what ncwa -a time appends
        if scalar:
            ov[...] = src.variables["time"][0]
    return out


def _write_sfcf_manifest(path: Path, cycle: datetime, segment: str, used: List[Tuple[datetime, Path]]) -> None:
    data = {"source_model": "gfs", "product": "sfcf", "cycle_time": cycle.isoformat(), "segment": segment,
            "nfiles": len(used),
            "files": [{"valid_time": v.isoformat(), "source_location": str(p), "file_size": p.stat().st_size}
                      for v, p in used]}
    with open(str(path), "w") as fh:
        json.dump(data, fh, indent=2)
