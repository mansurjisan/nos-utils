"""
USGS climatology-based river forcing processor.

For OFS systems (like SECOFS) where NWM has no matching reaches and
real-time USGS BUFR data is unavailable, river forcing is generated
entirely from the USGS daily climatology file (nosofs.river.clim.usgs.nc).

Reads:
  - {OFS}.river.ctl          — River node mappings, station IDs, scaling factors
  - nosofs.river.clim.usgs.nc — Daily climatological discharge, temperature, salinity

Writes:
  - vsource.th       — Volume source (hourly, positive m³/s per node)
  - msource.th       — Temperature + salinity per node (hourly)
  - source_sink.in   — SCHISM source/sink node configuration
  - schism_flux.th   — Volume flux at model dt (negative = inflow)
  - schism_temp.th   — River temperature at model dt
  - schism_salt.th   — River salinity at model dt
  - Tar archive of schism_flux/temp/salt.th
"""

import logging
import tarfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import ForcingConfig
from .base import ForcingProcessor, ForcingResult

log = logging.getLogger(__name__)

try:
    from netCDF4 import Dataset
    HAS_NETCDF4 = True
except ImportError:
    HAS_NETCDF4 = False

# Default salinity when clim has zero (SCHISM needs nonzero for stability)
DEFAULT_RIVER_SALT = 0.005


@dataclass
class RiverCTL:
    """Parsed river control file."""
    n_nodes: int           # Number of model grid nodes with river input
    n_stations: int        # Number of USGS stations
    dt_hours: float        # Output interval in hours
    # Per-station info
    station_ids: List[str]
    q_flags: List[int]     # 0=clim, 1=realtime
    q_means: List[float]   # Mean discharge (fallback)
    t_means: List[float]   # Mean temperature (fallback)
    # Per-node info
    node_ids: List[int]       # SCHISM node indices
    river_id_q: List[int]     # Which station provides Q for this node
    q_scales: List[float]     # Discharge scaling factor per node
    river_id_t: List[int]     # Which station provides T for this node
    t_scales: List[float]     # Temperature scaling factor per node
    river_flags: List[int]    # River flag (3 = both T and S active)
    river_names: List[str]


def parse_river_ctl(filepath: Path) -> RiverCTL:
    """
    Parse COMF river control file.

    Format:
        Section 1: USGS station definitions
            N_NODES  N_STATIONS  DELT
            Header line
            StationID ... Q_Flag TS_Flag "Name"

        Section 2: Grid node mappings
            Header line
            GRID_ID NODE_ID ELE_ID DIR FLAG RiverID_Q Q_Scale RiverID_T T_Scale "Name"
    """
    lines = filepath.read_text().splitlines()

    # Find Section 1 header
    sec1_start = None
    for i, line in enumerate(lines):
        if "Section 1" in line or "NRIVERS" in line.upper():
            sec1_start = i
            break

    if sec1_start is None:
        raise ValueError(f"Cannot find Section 1 in {filepath}")

    # Parse N_NODES, N_STATIONS, DELT from line after Section 1 header
    parts = lines[sec1_start + 1].split("!")[0].split()
    n_nodes = int(parts[0])
    n_stations = int(parts[1])
    dt_hours = float(parts[2])

    # Parse station entries (skip header line)
    station_ids, q_flags, q_means, t_means = [], [], [], []
    station_line_start = sec1_start + 3  # skip header
    for i in range(n_stations):
        line = lines[station_line_start + i]
        parts = line.split()
        station_ids.append(parts[1])       # STATION_ID
        q_means.append(float(parts[6]))    # Q_mean
        t_means.append(float(parts[9]))    # T_mean
        q_flags.append(int(parts[10]))     # Q_Flag

    # Find Section 2
    sec2_start = None
    for i, line in enumerate(lines):
        if "Section 2" in line or "GRID_ID" in line.upper():
            sec2_start = i
            break

    if sec2_start is None:
        raise ValueError(f"Cannot find Section 2 in {filepath}")

    # Parse node entries (skip header line)
    node_ids, river_id_q, q_scales = [], [], []
    river_id_t, t_scales, river_flags, river_names = [], [], [], []
    node_line_start = sec2_start + 2  # skip "Section 2" + header
    for i in range(n_nodes):
        line = lines[node_line_start + i]
        parts = line.split('"')
        fields = parts[0].split()
        node_ids.append(int(fields[1]))       # NODE_ID
        river_flags.append(int(fields[4]))    # FLAG
        river_id_q.append(int(fields[5]))     # RiverID_Q
        q_scales.append(float(fields[6]))     # Q_Scale
        river_id_t.append(int(fields[7]))     # RiverID_T
        t_scales.append(float(fields[8]))     # T_Scale
        river_names.append(parts[1].strip() if len(parts) > 1 else f"River {i+1}")

    return RiverCTL(
        n_nodes=n_nodes, n_stations=n_stations, dt_hours=dt_hours,
        station_ids=station_ids, q_flags=q_flags, q_means=q_means, t_means=t_means,
        node_ids=node_ids, river_id_q=river_id_q, q_scales=q_scales,
        river_id_t=river_id_t, t_scales=t_scales,
        river_flags=river_flags, river_names=river_names,
    )


def load_usgs_climatology(
    clim_path: Path,
    station_id: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load daily climatological Q, T, S for a USGS station.

    Returns:
        (months, days, discharge, temperature, salinity) — each shape (366,)
    """
    ds = Dataset(str(clim_path))

    # Read station IDs
    raw = ds.variables["stationID"][:]
    sids = []
    for row in raw:
        chars = []
        for c in row:
            if hasattr(c, "mask") and c.mask:
                break
            chars.append(c.decode() if isinstance(c, bytes) else str(c))
        sids.append("".join(chars).strip())

    idx = None
    for i, sid in enumerate(sids):
        if sid == station_id:
            idx = i
            break

    if idx is None:
        ds.close()
        raise ValueError(f"Station {station_id} not found in {clim_path}")

    months = np.array(ds.variables["month"][:], dtype=np.float64)
    days = np.array(ds.variables["day"][:], dtype=np.float64)
    discharge = np.array(ds.variables["discharge"][idx], dtype=np.float64)
    temperature = np.array(ds.variables["temperature"][idx], dtype=np.float64)
    salinity = np.array(ds.variables["salinity"][idx], dtype=np.float64)
    ds.close()

    log.info(f"Loaded clim for {station_id}: Q=[{discharge.min():.1f},{discharge.max():.1f}] "
             f"T=[{temperature.min():.1f},{temperature.max():.1f}]")

    return months, days, discharge, temperature, salinity


def _day_of_year(dt: datetime) -> int:
    """1-based day of year."""
    return dt.timetuple().tm_yday


def _find_clim_index(months: np.ndarray, days: np.ndarray, dt: datetime) -> int:
    """Find clim array index matching a datetime's month/day."""
    m, d = dt.month, dt.day
    for i in range(len(months)):
        if int(months[i]) == m and int(days[i]) == d:
            return i
    # Fallback: closest match
    for i in range(len(months)):
        if int(months[i]) == m and int(days[i]) == d - 1:
            return i + 1
    return _day_of_year(dt) - 1


def interp_clim_to_times(
    months: np.ndarray,
    days: np.ndarray,
    values: np.ndarray,
    times_dt: List[datetime],
) -> np.ndarray:
    """
    Interpolate daily climatology to arbitrary datetimes.

    Linear interpolation between the two surrounding daily clim values.
    Wraps around Dec 31 -> Jan 1.
    """
    result = np.empty(len(times_dt), dtype=np.float64)

    for i, dt in enumerate(times_dt):
        doy = _day_of_year(dt) - 1  # 0-based index
        frac = (dt.hour + dt.minute / 60.0 + dt.second / 3600.0) / 24.0
        idx0 = doy % len(values)
        idx1 = (doy + 1) % len(values)
        result[i] = values[idx0] * (1 - frac) + values[idx1] * frac

    return result


class RiverClimProcessor(ForcingProcessor):
    """
    USGS climatology-based river forcing processor.

    For OFS systems where NWM and real-time USGS data are unavailable,
    generates river forcing from the daily climatology file.
    """

    SOURCE_NAME = "RIVER_CLIM"

    def __init__(
        self,
        config: ForcingConfig,
        input_path: Path,
        output_path: Path,
        river_ctl_path: Optional[Path] = None,
        clim_path: Optional[Path] = None,
        phase: str = "nowcast",
        time_hotstart: Optional[datetime] = None,
    ):
        """
        Args:
            config: ForcingConfig
            input_path: FIX directory containing river.ctl and clim file
            output_path: Output directory
            river_ctl_path: Path to {OFS}.river.ctl (auto-discovered if None)
            clim_path: Path to nosofs.river.clim.usgs.nc (auto-discovered if None)
            phase: "nowcast" or "forecast"
            time_hotstart: Hotstart datetime
        """
        super().__init__(config, input_path, output_path)
        self._river_ctl_path = river_ctl_path
        self._clim_path = clim_path
        self.phase = phase
        self.time_hotstart = time_hotstart

    def _find_river_ctl(self) -> Optional[Path]:
        if self._river_ctl_path and self._river_ctl_path.exists():
            return self._river_ctl_path
        # Auto-discover
        ofs = getattr(self.config, "ofs_name", None) or "secofs"
        for name in [f"{ofs}.river.ctl", "river.ctl"]:
            p = self.input_path / name
            if p.exists():
                return p
        return None

    def _find_clim_file(self) -> Optional[Path]:
        if self._clim_path and self._clim_path.exists():
            return self._clim_path
        for name in ["nosofs.river.clim.usgs.nc", "river.clim.usgs.nc"]:
            p = self.input_path / name
            if p.exists():
                return p
            # Check parent (shared fix)
            p2 = self.input_path.parent / "shared" / name
            if p2.exists():
                return p2
        return None

    def _compute_time_window(self) -> Tuple[datetime, datetime, datetime]:
        """
        Compute start/end for the full simulation (nowcast + forecast combined).

        Returns (start, end_hourly, end_model):
          - end_hourly: cycle + forecast_hours (for vsource/msource)
          - end_model:  cycle + forecast_hours + 1h buffer (for schism_th)

        The Fortran writes vsource/msource to forecast end, but extends
        schism_flux/temp/salt.th by 1 additional hour.
        """
        cycle_dt = datetime.strptime(self.config.pdy, "%Y%m%d") + \
                   timedelta(hours=self.config.cyc)

        start = self.time_hotstart if self.time_hotstart else \
                cycle_dt - timedelta(hours=self.config.nowcast_hours)
        end_hourly = cycle_dt + timedelta(hours=self.config.forecast_hours)
        # Fortran END_TIME includes a 1-hour buffer beyond the forecast period
        end_model = cycle_dt + timedelta(hours=self.config.forecast_hours + 1)

        return start, end_hourly, end_model

    def find_input_files(self) -> List[Path]:
        files = []
        ctl = self._find_river_ctl()
        if ctl:
            files.append(ctl)
        clim = self._find_clim_file()
        if clim:
            files.append(clim)
        return files

    def process(self) -> ForcingResult:
        """Generate river forcing from USGS climatology."""
        log.info(f"River clim processor: pdy={self.config.pdy} cyc={self.config.cyc:02d}z "
                 f"phase={self.phase}")

        if not HAS_NETCDF4:
            return ForcingResult(
                success=False, source=self.SOURCE_NAME,
                errors=["netCDF4 not available"],
            )

        # Find inputs
        ctl_path = self._find_river_ctl()
        clim_path = self._find_clim_file()

        if not ctl_path:
            return ForcingResult(
                success=False, source=self.SOURCE_NAME,
                errors=[f"River CTL file not found in {self.input_path}"],
            )
        if not clim_path:
            return ForcingResult(
                success=False, source=self.SOURCE_NAME,
                errors=[f"Climatology file not found in {self.input_path}"],
            )

        self.create_output_dir()

        try:
            # Parse control file
            ctl = parse_river_ctl(ctl_path)
            log.info(f"River CTL: {ctl.n_nodes} nodes, {ctl.n_stations} stations, "
                     f"dt={ctl.dt_hours}h")
            for i, name in enumerate(ctl.river_names):
                log.info(f"  Node {ctl.node_ids[i]}: {name} "
                         f"(Q_scale={ctl.q_scales[i]}, station={ctl.station_ids[min(i, len(ctl.station_ids)-1)]})")

            # Load climatology for each unique station
            clim_data = {}
            for sid in set(ctl.station_ids):
                clim_data[sid] = load_usgs_climatology(clim_path, sid)

            # Compute time windows
            start, end_hourly, end_model = self._compute_time_window()
            model_dt = getattr(self.config, "dt", 120.0)  # model timestep in seconds

            log.info(f"Time window: {start} to {end_hourly} (hourly), "
                     f"{start} to {end_model} (model dt={model_dt}s)")

            # Generate hourly times for vsource/msource (always 1-hour interval)
            hourly_times = []
            t = start
            while t <= end_hourly:
                hourly_times.append(t)
                t += timedelta(hours=1.0)

            # Generate model-dt times for schism_flux/temp/salt (includes +1h buffer)
            model_times = []
            t = start
            while t <= end_model:
                model_times.append(t)
                t += timedelta(seconds=model_dt)

            # Compute climatological averages matching Fortran behavior:
            # Average 3 daily clim values centered on the hotstart day
            # (day before, hotstart day, day after)
            # Index by month/day (not day-of-year) to handle leap year clim file
            primary_sid = ctl.station_ids[0]
            months, days_arr, q_clim, t_clim, s_clim = clim_data[primary_sid]

            center_idx = _find_clim_index(months, days_arr, start)
            run_day_indices = [
                (center_idx - 1) % len(q_clim),
                center_idx,
                (center_idx + 1) % len(q_clim),
            ]

            # Average clim over these days (matching Fortran behavior)
            # Fortran rounds Q to nearest integer (NINT) before applying Q_scale
            q_avg = round(float(np.mean([q_clim[d % len(q_clim)] for d in run_day_indices])))
            t_avg = np.mean([t_clim[d % len(t_clim)] for d in run_day_indices])
            s_avg = np.mean([s_clim[d % len(s_clim)] for d in run_day_indices])

            # For msource.th: use config defaults for T/S (Fortran uses different
            # values for msource vs schism_temp — msource gets a simpler default)
            msource_temp = getattr(self.config, "river_default_temp", 10.0)
            msource_salt = getattr(self.config, "river_default_salt", 0.0)

            log.info(f"Clim average over days {run_day_indices}: "
                     f"Q={q_avg:.1f} T={t_avg:.1f} S={s_avg:.4f}")

            # Apply minimum salinity
            s_avg = max(s_avg, DEFAULT_RIVER_SALT)

            # Constant values for all timesteps (matching Fortran)
            q_hourly = np.full(len(hourly_times), q_avg)
            t_hourly = np.full(len(hourly_times), t_avg)
            s_hourly = np.full(len(hourly_times), s_avg)

            q_model = np.full(len(model_times), q_avg)
            t_model = np.full(len(model_times), t_avg)
            s_model = np.full(len(model_times), s_avg)

            output_files = []

            # Write vsource.th (hourly, positive discharge per node)
            vsource = self._write_vsource(q_hourly, hourly_times, start, ctl)
            if vsource:
                output_files.append(vsource)

            # Write msource.th (hourly, T + S per node — uses config defaults)
            t_ms = np.full(len(hourly_times), msource_temp)
            s_ms = np.full(len(hourly_times), msource_salt)
            msource = self._write_msource(t_ms, s_ms, hourly_times, start, ctl)
            if msource:
                output_files.append(msource)

            # Write source_sink.in
            ss = self._write_source_sink(ctl)
            if ss:
                output_files.append(ss)

            # Write schism_flux/temp/salt.th (at model dt, negative flux)
            th_files = self._write_schism_th(
                q_model, t_model, s_model, model_times, start, model_dt, ctl,
            )
            output_files.extend(th_files)

            return ForcingResult(
                success=True, source=self.SOURCE_NAME,
                output_files=output_files,
                metadata={
                    "n_nodes": ctl.n_nodes,
                    "n_stations": ctl.n_stations,
                    "clim_station": primary_sid,
                    "phase": self.phase,
                    "n_hourly_steps": len(hourly_times),
                    "n_model_steps": len(model_times),
                },
            )

        except Exception as e:
            log.error(f"River clim processing failed: {e}")
            import traceback
            log.error(traceback.format_exc())
            return ForcingResult(
                success=False, source=self.SOURCE_NAME,
                errors=[str(e)],
            )

    def _write_vsource(
        self, q_interp: np.ndarray, times: List[datetime],
        start: datetime, ctl: RiverCTL,
    ) -> Optional[Path]:
        """Write vsource.th — positive discharge per node, hourly."""
        output = self.output_path / "vsource.th"
        try:
            with open(output, "w") as f:
                for i, dt in enumerate(times):
                    t_sec = (dt - start).total_seconds()
                    vals = []
                    for n in range(ctl.n_nodes):
                        rid = ctl.river_id_q[n] - 1  # 1-based to 0-based
                        vals.append(q_interp[i] * ctl.q_scales[n])
                    line = " ".join(f"{v:.4f}" for v in vals)
                    f.write(f"{t_sec:.1f} {line}\n")
            log.info(f"Created {output.name}: {len(times)} steps, {ctl.n_nodes} nodes")
            return output
        except Exception as e:
            log.error(f"Failed to write vsource.th: {e}")
            return None

    def _write_msource(
        self, t_interp: np.ndarray, s_interp: np.ndarray,
        times: List[datetime], start: datetime, ctl: RiverCTL,
    ) -> Optional[Path]:
        """Write msource.th — temperature + salinity per node, hourly."""
        output = self.output_path / "msource.th"
        try:
            with open(output, "w") as f:
                for i, dt in enumerate(times):
                    t_sec = (dt - start).total_seconds()
                    vals = []
                    for n in range(ctl.n_nodes):
                        vals.append(f"{t_interp[i] * ctl.t_scales[n]:.1f}")
                        vals.append(f"{s_interp[i]:.1f}")
                    f.write(f"{t_sec:.1f} {' '.join(vals)}\n")
            log.info(f"Created {output.name}: {len(times)} steps")
            return output
        except Exception as e:
            log.error(f"Failed to write msource.th: {e}")
            return None

    def _write_source_sink(self, ctl: RiverCTL) -> Optional[Path]:
        """Write source_sink.in."""
        output = self.output_path / "source_sink.in"
        try:
            with open(output, "w") as f:
                f.write(f"{ctl.n_nodes}\n")
                for nid in ctl.node_ids:
                    f.write(f"{nid} 1\n")
                f.write("0\n")
            log.info(f"Created {output.name}: {ctl.n_nodes} sources")
            return output
        except Exception as e:
            log.error(f"Failed to write source_sink.in: {e}")
            return None

    def _write_schism_th(
        self, q_model: np.ndarray, t_model: np.ndarray, s_model: np.ndarray,
        times: List[datetime], start: datetime, dt: float, ctl: RiverCTL,
    ) -> List[Path]:
        """
        Write schism_flux/temp/salt.th at model dt and create tar archive.

        schism_flux.th uses NEGATIVE values (SCHISM convention: negative = inflow).
        """
        output_files = []
        data_dir = self.output_path / "data"
        data_dir.mkdir(exist_ok=True)

        for fname, values, sign, fmt in [
            ("schism_flux.th", q_model, -1.0, "{:12.2f}"),
            ("schism_temp.th", t_model, 1.0, "{:12.4f}"),
            ("schism_salt.th", s_model, 1.0, "{:12.4f}"),
        ]:
            fpath = data_dir / fname
            try:
                with open(fpath, "w") as f:
                    for i, dt_val in enumerate(times):
                        t_sec = (dt_val - start).total_seconds()
                        vals = []
                        for n in range(ctl.n_nodes):
                            scale = ctl.q_scales[n] if "flux" in fname else ctl.t_scales[n]
                            vals.append(fmt.format(sign * values[i] * scale))
                        f.write(f"{t_sec:11.0f}.{''.join(vals)}\n")
                output_files.append(fpath)
                log.info(f"Created {fname}: {len(times)} steps")
            except Exception as e:
                log.error(f"Failed to write {fname}: {e}")

        # Create tar archive
        tar_path = self.output_path / "river.th.tar"
        try:
            with tarfile.open(str(tar_path), "w") as tf:
                for fname in ["schism_flux.th", "schism_temp.th", "schism_salt.th"]:
                    fpath = data_dir / fname
                    if fpath.exists():
                        tf.add(str(fpath), arcname=fname)
            output_files.append(tar_path)
            log.info(f"Created {tar_path.name}")
        except Exception as e:
            log.error(f"Failed to create tar: {e}")

        return output_files
