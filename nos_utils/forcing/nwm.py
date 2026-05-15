"""
NWM (National Water Model) river forcing processor.

Creates SCHISM river forcing files from NWM streamflow data:
  - vsource.th     — Volume source time history (m³/s per river)
  - msource.th     — Mass source (temperature, salinity per river)
  - source_sink.in — SCHISM source/sink node configuration

Input: NWM NetCDF channel_rt files from COMINnwm
  Pattern: nwm.YYYYMMDD/nwm.tHHz.{product}.channel_rt.f*.conus.nc
  Products: analysis_assim, short_range, medium_range

Fallback: Monthly climatology when NWM data is unavailable.
"""

import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import ForcingConfig
from .base import ForcingProcessor, ForcingResult

log = logging.getLogger(__name__)


# NWM channel_rt filename: cycle hour HH and lead-time tag (tmNN lookback or
# fNNN forecast). Optional `_<member>` suffix on channel_rt for medium_range
# ensemble members (`channel_rt_1` etc.). The parent dir is ``nwm.YYYYMMDD``.
_NWM_FNAME_RE = re.compile(
    r"nwm\.t(?P<cyc>\d{2})z\.\w+\.channel_rt(?:_\d+)?\."
    r"(?P<lead_kind>tm|f)(?P<lead>\d+)\.conus\.nc$"
)


def _nwm_valid_time(path: Path) -> datetime:
    """Compute valid time for an NWM channel_rt file from its name + parent dir.

    Files within a single analysis_assim cycle are named by *lookback hours*
    (``tm00`` is the cycle hour, ``tm01`` is one hour earlier, …), so a
    plain lexicographic sort of `nwm.t00z.…tm00`, `…tm01`, `…tm02` walks
    valid time *backwards* and yields a non-monotonic time axis when fed
    to SCHISM. Sorting by this key fixes that for both the lookback (`tm`)
    and forecast (`f`) products without opening the NetCDF.

    Returns ``datetime.min`` for unparseable names so they sort to the start
    rather than crashing the pipeline.
    """
    m = _NWM_FNAME_RE.search(path.name)
    if not m:
        return datetime.min

    # Walk up to find the ``nwm.YYYYMMDD`` ancestor (parent or grandparent
    # depending on whether the file lives under a per-product subdir).
    parent = path.parent
    date_str: Optional[str] = None
    for _ in range(3):
        if parent.name.startswith("nwm.") and len(parent.name) == 12:
            date_str = parent.name[4:]
            break
        parent = parent.parent
    if date_str is None:
        return datetime.min

    try:
        cycle = datetime.strptime(date_str, "%Y%m%d") + \
                timedelta(hours=int(m.group("cyc")))
    except ValueError:
        return datetime.min

    lead = int(m.group("lead"))
    if m.group("lead_kind") == "tm":
        return cycle - timedelta(hours=lead)
    return cycle + timedelta(hours=lead)

try:
    from netCDF4 import Dataset
    HAS_NETCDF4 = True
except ImportError:
    HAS_NETCDF4 = False

# Monthly seasonal multipliers for climatology fallback
MONTHLY_FLOW_FACTOR = {
    1: 0.90, 2: 1.00, 3: 1.20, 4: 1.50, 5: 1.30, 6: 1.00,
    7: 0.80, 8: 0.70, 9: 0.75, 10: 0.85, 11: 0.95, 12: 0.90,
}

# Monthly river temperature climatology (°C)
MONTHLY_RIVER_TEMP = {
    1: 4.0, 2: 4.0, 3: 6.0, 4: 10.0, 5: 14.0, 6: 18.0,
    7: 22.0, 8: 24.0, 9: 20.0, 10: 14.0, 11: 8.0, 12: 5.0,
}


class RiverConfig:
    """River configuration: maps NWM reach IDs to SCHISM mesh nodes."""

    def __init__(self, feature_ids: List[int], node_indices: List[int],
                 clim_flows: List[float], names: Optional[List[str]] = None,
                 feature_id_groups: Optional[List[List[int]]] = None,
                 sink_node_indices: Optional[List[int]] = None,
                 sink_feature_id_groups: Optional[List[List[int]]] = None,
                 # COMF river.ctl per-grid-point fields — used by
                 # `_write_river_th_files` to apply the legacy Fortran
                 # formula: flux = Q_obs(RiverID_Q[i]) * Q_Scale[i],
                 # temp = max(1.0, T_obs(RiverID_T[i]) * T_Scale[i]),
                 # salt = 0.005 (constant per nos_ofs_create_forcing_river.f:1898)
                 river_id_q: Optional[List[int]] = None,
                 q_scale: Optional[List[float]] = None,
                 river_id_t: Optional[List[int]] = None,
                 t_scale: Optional[List[float]] = None,
                 stations_q_mean: Optional[Dict[int, float]] = None,
                 stations_t_mean: Optional[Dict[int, float]] = None,
                 station_names: Optional[Dict[int, str]] = None,
                 station_usgs_ids: Optional[Dict[int, str]] = None):
        self.feature_ids = feature_ids
        self.node_indices = node_indices
        self.clim_flows = clim_flows
        self.names = names or [f"river_{i}" for i in range(len(feature_ids))]
        self.n_rivers = len(feature_ids)
        # STOFS: groups of feature_ids per source (flow summed within group)
        self.feature_id_groups = feature_id_groups
        # STOFS sinks (e.g. diversions) — element IDs SCHISM should treat as
        # negative-flux sources. The volume timeseries comes from a static
        # vsink.th in FIX (Python doesn't generate it per-cycle).
        self.sink_node_indices = sink_node_indices or []
        self.sink_feature_id_groups = sink_feature_id_groups
        self.n_sinks = len(self.sink_node_indices)
        # COMF per-grid scaling — when populated, _write_river_th_files
        # uses station-driven Q/T (with Q_Scale / T_Scale per grid point)
        # to match production's nos_ofs_create_forcing_river output.
        self.river_id_q = river_id_q or []
        self.q_scale = q_scale or []
        self.river_id_t = river_id_t or []
        self.t_scale = t_scale or []
        self.stations_q_mean = stations_q_mean or {}
        self.stations_t_mean = stations_t_mean or {}
        self.station_names = station_names or {}
        self.station_usgs_ids = station_usgs_ids or {}

    @classmethod
    def from_text(cls, filepath: Path) -> "RiverConfig":
        """Load river config from text file.

        Supports three formats:

        Format 1 (NWM reach file — secofs.nwm.reach.dat):
            REACH_ID FLAG (header)
            2                     (count)
            20104159 1            (feature_id flag)
            9643431  1

        Format 2 (full river config):
            feature_id  node_index  river_name  clim_flow

        Format 3 (Fortran river.ctl — secofs.river.ctl):
            Section 1: USGS station info (Q_mean per station)
            Section 2: Grid node mappings (NODE_ID, Q_Scale, RiverID)
        """
        with open(filepath) as f:
            text = f.read()

        # Detect Fortran river.ctl format by "Section 1:" marker
        if "Section 1:" in text and "Section 2:" in text:
            return cls._parse_river_ctl(text)

        return cls._parse_simple(text)

    @classmethod
    def _parse_river_ctl(cls, text: str) -> "RiverConfig":
        """Parse Fortran river.ctl format (secofs.river.ctl).

        Section 1 defines USGS stations with Q_mean.
        Section 2 defines SCHISM grid nodes with Q_Scale and RiverID linkage.
        Each grid node gets: clim_flow = station.Q_mean * node.Q_Scale
        """
        lines = text.splitlines()

        # Parse Section 1: USGS stations
        stations = {}  # river_id -> {q_mean, t_mean, name, usgs_id}
        in_section1 = False
        nij = 0

        for line in lines:
            stripped = line.strip()
            if "Section 1:" in line:
                in_section1 = True
                continue
            if "Section 2:" in line:
                in_section1 = False
                continue
            if not in_section1 or not stripped:
                continue

            parts = stripped.split()
            # Header line: NIJ NRIVERS DELT
            if parts[0].isdigit() and "!!" in stripped:
                nij = int(parts[0])
                continue
            # Column header line
            if parts[0] in ("RiverID", "GRID_ID"):
                continue
            # Station data line: RiverID STATION_ID NWS_ID AGENCY Q_min Q_max Q_mean T_min T_max T_mean ...
            try:
                rid = int(parts[0])
                usgs_id = parts[1]
                q_mean = float(parts[6])
                t_mean = float(parts[9]) if len(parts) > 9 else 15.0
                # Extract quoted name
                name = stripped.split('"')[1] if '"' in stripped else f"station_{parts[1]}"
                stations[rid] = {"q_mean": q_mean, "t_mean": t_mean,
                                 "name": name, "usgs_id": usgs_id}
            except (ValueError, IndexError):
                continue

        # Parse Section 2: grid node mappings
        feature_ids: List[int] = []
        node_indices: List[int] = []
        clim_flows: List[float] = []
        names: List[str] = []
        river_id_q: List[int] = []
        q_scale: List[float] = []
        river_id_t: List[int] = []
        t_scale: List[float] = []
        in_section2 = False

        for line in lines:
            stripped = line.strip()
            if "Section 2:" in line:
                in_section2 = True
                continue
            if "PARAMETER DEFINITION:" in line:
                break
            if not in_section2 or not stripped:
                continue

            parts = stripped.split()
            if parts[0] in ("GRID_ID",):
                continue

            # Grid data: GRID_ID NODE_ID ELE_ID DIR FLAG RiverID_Q Q_Scale RiverID_T T_Scale "Name"
            try:
                grid_id = int(parts[0])
                node_id = int(parts[1])
                rid_q = int(parts[5])
                qs = float(parts[6])
                rid_t = int(parts[7]) if len(parts) > 7 else rid_q
                ts = float(parts[8]) if len(parts) > 8 else 1.0
                name = stripped.split('"')[1] if '"' in stripped else f"river_{grid_id}"

                # Compute climatological flow for this node (legacy fallback)
                if rid_q in stations:
                    clim_flow = stations[rid_q]["q_mean"] * qs
                else:
                    clim_flow = 50.0 * qs

                feature_ids.append(grid_id)
                node_indices.append(node_id)
                clim_flows.append(clim_flow)
                names.append(name.strip())
                river_id_q.append(rid_q)
                q_scale.append(qs)
                river_id_t.append(rid_t)
                t_scale.append(ts)
            except (ValueError, IndexError):
                continue

        # Build station lookups by RiverID for O(1) Q_mean / T_mean access
        stations_q_mean = {rid: s["q_mean"] for rid, s in stations.items()}
        stations_t_mean = {rid: s["t_mean"] for rid, s in stations.items()}
        station_names = {rid: s["name"] for rid, s in stations.items()}
        station_usgs_ids = {rid: s["usgs_id"] for rid, s in stations.items()}

        log.info(f"Parsed river.ctl: {len(stations)} USGS stations, "
                 f"{len(feature_ids)} grid nodes")
        return cls(
            feature_ids, node_indices, clim_flows, names,
            river_id_q=river_id_q,
            q_scale=q_scale,
            river_id_t=river_id_t,
            t_scale=t_scale,
            stations_q_mean=stations_q_mean,
            stations_t_mean=stations_t_mean,
            station_names=station_names,
            station_usgs_ids=station_usgs_ids,
        )

    @classmethod
    def _parse_simple(cls, text: str) -> "RiverConfig":
        """Parse simple NWM reach or feature_id format."""
        feature_ids = []
        node_indices = []
        clim_flows = []
        names = []

        data_lines = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("!"):
                continue
            data_lines.append(line)

        if not data_lines:
            return cls([], [], [], [])

        first_field = data_lines[0].split()[0]
        try:
            int(first_field)
        except ValueError:
            data_lines = data_lines[1:]

        if data_lines and len(data_lines[0].split()) == 1:
            try:
                int(data_lines[0])
                data_lines = data_lines[1:]
            except ValueError:
                pass

        for i, line in enumerate(data_lines):
            parts = line.split()
            try:
                if len(parts) >= 4:
                    feature_ids.append(int(parts[0]))
                    node_indices.append(int(parts[1]))
                    names.append(parts[2])
                    clim_flows.append(float(parts[3]))
                elif len(parts) >= 2:
                    fid = int(parts[0])
                    flag = int(parts[1])
                    if flag == 1:
                        feature_ids.append(fid)
                        node_indices.append(i + 1)
                        names.append(f"reach_{fid}")
                        clim_flows.append(50.0)
                elif len(parts) == 1:
                    feature_ids.append(int(parts[0]))
                    node_indices.append(i + 1)
                    names.append(f"reach_{parts[0]}")
                    clim_flows.append(50.0)
            except ValueError:
                continue

        return cls(feature_ids, node_indices, clim_flows, names)

    @classmethod
    def from_json(cls, filepath: Path) -> "RiverConfig":
        """Load river config from JSON file."""
        import json
        with open(filepath) as f:
            data = json.load(f)

        if isinstance(data, list):
            feature_ids = [r.get("feature_id", 0) for r in data]
            node_indices = [r.get("node_index", 0) for r in data]
            clim_flows = [r.get("clim_flow", 0.0) for r in data]
            names = [r.get("name", f"river_{i}") for i, r in enumerate(data)]
        else:
            feature_ids = data.get("feature_ids", [])
            node_indices = data.get("node_indices", [])
            clim_flows = data.get("clim_flows", [0.0] * len(feature_ids))
            names = data.get("names", [])

        return cls(feature_ids, node_indices, clim_flows, names)

    @classmethod
    def from_sources_json(cls, filepath: Path,
                          sinks_path: Optional[Path] = None) -> "RiverConfig":
        """Load STOFS sources.json: {element_id: [fid1, fid2, ...]}.

        STOFS-3D-ATL uses VIMS gen_sourcesink.py format where each source
        element maps to one or more NWM feature_ids. Flow is summed across
        all feature_ids belonging to each source element.

        Args:
            filepath: Path to sources.json (e.g. stofs_3d_atl_river_sources_conus.json)
            sinks_path: Optional path to sinks.json — same {element_id: [fid, ...]}
                schema. If provided, the resulting RiverConfig also lists the
                sink elements so source_sink.in includes them. Sink volume
                timeseries comes from a static vsink.th in FIX, not generated
                here.
        """
        import json
        with open(filepath) as f:
            data = json.load(f)

        # data: {str(element_id): [int(feature_id), ...]}
        element_ids = sorted(data.keys(), key=int)
        feature_id_groups = [data[eid] for eid in element_ids]
        node_indices = [int(eid) for eid in element_ids]
        # First feature_id as representative (for backward compat)
        feature_ids = [groups[0] if groups else 0 for groups in feature_id_groups]
        n_sources = len(element_ids)
        names = [f"src_{eid}" for eid in element_ids]
        clim_flows = [0.0] * n_sources  # No climatology for STOFS NWM

        sink_node_indices: List[int] = []
        sink_feature_id_groups: Optional[List[List[int]]] = None
        if sinks_path is not None and Path(sinks_path).exists():
            with open(sinks_path) as f:
                sink_data = json.load(f)
            sink_keys = sorted(sink_data.keys(), key=int)
            sink_node_indices = [int(eid) for eid in sink_keys]
            sink_feature_id_groups = [sink_data[eid] for eid in sink_keys]
            log.info(f"Loaded STOFS sinks.json: {len(sink_node_indices)} sink elements "
                     f"from {Path(sinks_path).name}")

        log.info(f"Loaded STOFS sources.json: {n_sources} source elements "
                 f"from {filepath.name}")

        return cls(feature_ids, node_indices, clim_flows, names,
                   feature_id_groups=feature_id_groups,
                   sink_node_indices=sink_node_indices,
                   sink_feature_id_groups=sink_feature_id_groups)


class NWMProcessor(ForcingProcessor):
    """
    NWM river forcing processor for SCHISM.

    Extracts streamflow for configured river reaches from NWM output,
    maps to SCHISM mesh nodes, and creates vsource.th/msource.th files.
    """

    SOURCE_NAME = "NWM"
    MIN_FILE_SIZE = 0  # NWM files vary in size

    # NWM products in priority order
    PRODUCTS = ["analysis_assim", "short_range", "medium_range"]

    # Default buffer (hours) appended past the phase-window end so SCHISM's
    # source/sink time-series reader has interpolation headroom past the
    # simulation tail. Shared knob with RTOFS/Nudging via
    # ``config.obc_buffer_hours``. SCHISM aborts with
    # ``ABORT: step: time_series in vsource.th`` if the requested model
    # time falls outside the source-term time bounds, which happens near
    # the end of the run when the phase file stops exactly at sim_end.
    #
    # Independent of ``config.river_hourly_extra_hours``: the legacy
    # ``extra`` knob is the production-COMF ``time_hotstart + 72h``
    # compatibility lever and is added on top of the buffer.
    DEFAULT_BUFFER_HOURS = 3

    def __init__(
        self,
        config: ForcingConfig,
        input_path: Path,
        output_path: Path,
        river_config: Optional[RiverConfig] = None,
        phase: Optional[str] = None,
        time_hotstart: Optional[datetime] = None,
        buffer_hours: Optional[int] = None,
    ):
        """
        Args:
            config: ForcingConfig with river settings
            input_path: Root NWM data directory (COMINnwm)
            output_path: Output directory for river forcing files
            river_config: Pre-loaded river configuration (or loaded from config.river_config_file)
            phase: "nowcast" or "forecast" — determines time window for NWM files
            time_hotstart: Hotstart datetime (nowcast starts from here)
            buffer_hours: Extra hours appended past the phase window end so
                SCHISM's source/sink time-series reader has interpolation
                headroom past the simulation tail. Defaults to
                ``config.obc_buffer_hours`` when set on the
                ``ForcingConfig``, otherwise to ``DEFAULT_BUFFER_HOURS``
                (3h, matching legacy COMF). Applied to phase != None only.
        """
        super().__init__(config, input_path, output_path)
        self._river_config = river_config
        self.phase = phase
        self.time_hotstart = time_hotstart
        if buffer_hours is None:
            buffer_hours = getattr(
                config, "obc_buffer_hours", self.DEFAULT_BUFFER_HOURS,
            )
        self.buffer_hours = int(buffer_hours)

    @property
    def river_config(self) -> Optional[RiverConfig]:
        if self._river_config is None and self.config.river_config_file:
            path = Path(self.config.river_config_file)
            if not path.exists():
                log.warning(f"River config file not found: {path}")
                return None
            try:
                if path.suffix == ".json":
                    # Detect STOFS sources.json format: {element: [fid, ...]}
                    import json
                    with open(path) as f:
                        data = json.load(f)
                    if isinstance(data, dict) and data:
                        first_val = next(iter(data.values()))
                        if isinstance(first_val, list):
                            sinks_path = (Path(self.config.sinks_config_file)
                                          if self.config.sinks_config_file else None)
                            self._river_config = RiverConfig.from_sources_json(
                                path, sinks_path=sinks_path)
                        else:
                            self._river_config = RiverConfig.from_json(path)
                    else:
                        self._river_config = RiverConfig.from_json(path)
                else:
                    self._river_config = RiverConfig.from_text(path)
            except Exception as e:
                log.warning(f"Failed to load river config: {e}")
                return None
        return self._river_config

    @property
    def is_stofs_mode(self) -> bool:
        """True if using STOFS-style aggregated sources (from_sources_json)."""
        return (self.river_config is not None and
                self.river_config.feature_id_groups is not None)

    def process(self) -> ForcingResult:
        """
        Process NWM river forcing data.

        Pipeline: load config → find NWM files → extract streamflow → write output
        Falls back to climatology if NWM data is unavailable.
        """
        log.info(f"NWM processor: pdy={self.config.pdy} cyc={self.config.cyc:02d}z")

        if self.river_config is None:
            return ForcingResult(
                success=False, source=self.SOURCE_NAME,
                errors=["No river configuration provided (river_config_file or RiverConfig)"],
            )

        self.create_output_dir()
        n_rivers = self.river_config.n_rivers
        log.info(f"Processing {n_rivers} rivers")

        # Target time steps (hourly grid for vsource/vsink/msource).
        #
        # Production reference: ``nos_ofs_create_forcing_river.sh`` writes
        # the NWM source/sink time window as
        # ``time_end = NDATE 72 ${time_hotstart}`` (line 86), regardless of
        # the simulation length. For SECOFS-UFS (nowcast=6, forecast=48,
        # sim=54h) that is 18h past the forecast end → 73 hourly rows.
        # The extension matters because SCHISM uses these rows for source
        # interpolation; a too-short grid aborts the run on
        # ``rnday > nowcast_hours``.
        #
        # ``river_hourly_extra_hours`` is set per-OFS in the factories
        # (SECOFS / SECOFS-UFS = 18) and via the ``river.hourly_extra_hours``
        # YAML knob; default 0 preserves prior behavior for STOFS, which
        # uses the medium-range 121-file path with its own length.
        #
        # Phase semantics for the OUTPUT axis:
        #   * phase="nowcast":  total_hours = nowcast_hours + buffer + extra
        #     (output covers [0, total_hours] anchored at
        #     ``cycle - nowcast_hours``; buffer past sim_end gives
        #     SCHISM time_series interpolation headroom)
        #   * phase="forecast": total_hours = forecast_hours + buffer + extra
        #     (output covers [0, total_hours] anchored at ``cycle``)
        #   * phase=None (default, backward compat): total_hours =
        #     nowcast_hours + forecast_hours + extra (combined 54h
        #     without buffer)
        total_hours = self._phase_total_hours()
        n_target = total_hours + 1  # hourly

        # Try NWM data first
        nwm_files = self.find_input_files()
        from ._log import log_input_files
        n_reaches = len(self.river_config.feature_ids) if self.river_config else 0
        log_input_files(
            "NWM", nwm_files,
            note=f"pdy={self.config.pdy} cyc={self.config.cyc} "
                 f"stofs_mode={self.is_stofs_mode} reaches={n_reaches} "
                 f"target_hours={n_target}",
        )
        if nwm_files:
            log.info(f"Found {len(nwm_files)} NWM files")
            if self.is_stofs_mode:
                flows, times = self._extract_streamflow_aggregated(nwm_files)
            else:
                flows, times = self._extract_streamflow(nwm_files)
        else:
            log.warning("No NWM files found — using climatology")
            flows, times = self._generate_climatology(n_target)

        # Snap to a uniform hourly grid covering [0, n_target) hours from
        # start_time. Drops pre-nowcast lookback (analysis_assim cycles
        # commonly span -14h..+0h around the cycle hour, but only the
        # nowcast window is part of the simulation), keeps the first
        # (earliest-cycle) flow when two products report the same hour
        # (analysis preferred over forecast for past hours), and
        # forward-fills any missing hours so the time axis is dense
        # and monotonic — SCHISM aborts otherwise.
        flows, times = self._normalize_to_simulation_grid(flows, times, n_target)

        # Pad to target length if needed (covers the case where NWM data
        # ends before n_target — pads with the last value, hourly grid).
        if len(times) < n_target:
            flows, times = self._pad_to_target(flows, times, n_target)

        # Write output files
        output_files = []

        vsource = self._write_vsource(flows, times)
        if vsource:
            output_files.append(vsource)

        # msource.th must have the SAME timestep count as vsource.th — SCHISM
        # allocates buffers based on the file dimensions and reads them
        # together. A single-timestep msource.th alongside a 55-row vsource.th
        # caused heap corruption at `partition_hgrid_` (V11 nowcast crash).
        # Always generate at the same time grid as vsource/vsink.
        msource = self._write_msource(times)
        if msource:
            output_files.append(msource)

        # STOFS sink volumes: SCHISM expects vsink.th columns to match the
        # n_sinks declared in source_sink.in. Static FIX files from older
        # baselines carry a different sink count and would crash SCHISM
        # during partition with a heap overflow. Generate the time series
        # in-place with zeros (no diversion flow) so the count is always
        # consistent. NWM doesn't provide sink data anyway.
        if self.is_stofs_mode and self.river_config.n_sinks > 0:
            vsink = self._write_vsink(times)
            if vsink:
                output_files.append(vsink)

        # SECOFS-UFS open-boundary rivers (river.ctl). Loaded separately
        # from sources.json; the two mechanisms coexist. Climatological Q
        # for the smoke test; per-cycle USGS observations are a TODO.
        ctl_cfg = self._load_ctl_river_config()
        if ctl_cfg is not None and ctl_cfg.n_rivers > 0:
            river_th_files = self._write_river_th_files(times, ctl_cfg)
            output_files.extend(river_th_files)

        source_sink = self._write_source_sink()
        if source_sink:
            output_files.append(source_sink)

        return ForcingResult(
            success=len(output_files) > 0,
            source=self.SOURCE_NAME,
            output_files=output_files,
            metadata={
                "n_rivers": n_rivers,
                "n_timesteps": len(times),
                "nwm_files_used": len(nwm_files),
                "used_climatology": len(nwm_files) == 0,
                "stofs_mode": self.is_stofs_mode,
                "phase": self.phase,
            },
        )

    # ------------------------------------------------------------------
    # Phase helpers
    # ------------------------------------------------------------------

    def _phase_start_time(self) -> datetime:
        """Return the absolute datetime of the row ``t=0`` in the output.

        * phase="nowcast":  ``cycle - nowcast_hours`` (SCHISM hotstart anchor)
        * phase="forecast": ``cycle`` (operator's forecast leg start)
        * phase=None (default, backward compat): ``cycle - nowcast_hours``
        """
        cycle_dt = datetime.strptime(self.config.pdy, "%Y%m%d") + \
                   timedelta(hours=self.config.cyc)
        if self.phase == "forecast":
            return cycle_dt
        # nowcast or None: anchor at cycle - nowcast_hours (combined window
        # starts here too, matching the pre-phase Route B behavior).
        return cycle_dt - timedelta(hours=int(self.config.nowcast_hours))

    def _phase_total_hours(self) -> int:
        """Return the number of hours covered by the output window.

        Hourly grid length is ``_phase_total_hours() + 1`` (0..N inclusive).

        * phase="nowcast":  ``nowcast_hours + buffer + extra``
        * phase="forecast": ``forecast_hours + buffer + extra``
        * phase=None (default, backward compat):
          ``nowcast_hours + forecast_hours + extra`` (no buffer)

        ``extra`` is ``max(0, river_hourly_extra_hours)`` — production
        COMF extends the source axis past the simulation end so SCHISM
        always has interpolation room at the tail.

        ``buffer`` is ``self.buffer_hours`` (default 3h, override via
        ``buffer_hours`` kwarg or ``config.obc_buffer_hours``) and is
        appended past ``sim_end`` for phase != None so the SCHISM
        time_series reader has interpolation headroom even when
        ``river_hourly_extra_hours`` is the default 0 (e.g. STOFS).
        The combined ``phase=None`` window omits the buffer so existing
        callers that depend on the exact 54h duration are unaffected.
        """
        extra = max(0, int(self.config.river_hourly_extra_hours))
        if self.phase == "nowcast":
            return int(self.config.nowcast_hours) + self.buffer_hours + extra
        if self.phase == "forecast":
            return int(self.config.forecast_hours) + self.buffer_hours + extra
        return (int(self.config.nowcast_hours)
                + int(self.config.forecast_hours)
                + extra)

    def find_input_files(self) -> List[Path]:
        """Find NWM channel_rt files for the run window.

        Dispatch is *product*-driven, not is_stofs_mode-driven: SECOFS-UFS
        loads STOFS-style sources.json (so is_stofs_mode=True) but its
        nwm_product is "analysis_assim" — routing it through the medium_range
        multi-cycle assembler returned 0 files and silently fell back to
        climatology (1 m³/s × 3522 rivers replacing real streamflow).
        Aggregation per source group is independent and is correctly
        triggered by is_stofs_mode at extract time.

        Returned list is sorted by valid time, not filename. Within an
        analysis_assim cycle the ``tmHH`` lookback runs backwards in
        valid time, so a lexicographic sort produces a non-monotonic
        time axis once multiple cycles are staged. SCHISM rejects that.
        """
        if self.config.nwm_product == "medium_range_mem1":
            files = self._find_stofs_nwm_files()
        else:
            files = self._find_secofs_nwm_files()
        # Secondary key on path string keeps the order deterministic for
        # files that share a valid time (same cycle revisited, etc.).
        return sorted(files, key=lambda p: (_nwm_valid_time(p), str(p)))

    def _find_secofs_nwm_files(self) -> List[Path]:
        """Find NWM files using SECOFS product search.

        NWM v3.0 production layout (`/lfs/h1/ops/prod/com/nwm/v3.0/`):
            nwm.YYYYMMDD/<product>/nwm.tHHz.<product>.channel_rt.<lead>.conus.nc

        Lead-time tag varies by product:
            - analysis_assim: ``tmHH`` (lookback hours; tm00 is on-the-hour)
            - short_range / medium_range: ``fHHH`` (forecast hours)

        If `nwm_product` is set explicitly (e.g. "analysis_assim"), only
        that product is searched. Otherwise the priority order is
        ``analysis_assim → short_range → medium_range`` — first non-empty
        product wins.
        """
        base_date = datetime.strptime(self.config.pdy, "%Y%m%d")
        nwm_files: List[Path] = []

        # Honor explicit product config; otherwise fall back to priority list.
        configured = self.config.nwm_product
        if configured and configured in self.PRODUCTS:
            search_products = [configured]
        else:
            search_products = list(self.PRODUCTS)

        for date in [base_date, base_date - timedelta(days=1)]:
            date_str = date.strftime("%Y%m%d")
            nwm_dir = self.input_path / f"nwm.{date_str}"
            if not nwm_dir.exists():
                continue

            for product in search_products:
                # analysis_assim uses tmHH lookback; forecast products use fHHH
                lead_glob = "tm*" if product == "analysis_assim" else "f*"
                pattern = (
                    f"nwm.t*z.{product}.channel_rt.{lead_glob}.conus.nc"
                )
                # Production layout has a per-product subdirectory; older
                # caches may have flat files. Try both, dedup by path.
                found = sorted(
                    set(
                        list((nwm_dir / product).glob(pattern))
                        + list(nwm_dir.glob(pattern))
                    )
                )
                if found:
                    nwm_files.extend(found)
                    # Stop at first product that returns files so we don't
                    # mix analysis with forecast on the same time grid.
                    break

            if nwm_files:
                break

        return nwm_files

    def _find_stofs_nwm_files(self) -> List[Path]:
        """Find NWM files using STOFS multi-cycle assembly (medium_range_mem1).

        Primary list (from shell script):
          yesterday t06z f006 (1 file)
          yesterday t12z f001-f006 (6 files)
          yesterday t18z f001-f006 (6 files)
          today     t00z f001-f006 (6 files)
          today     t06z f001-f110 (102 files → 121 total target)

        Backup: yesterday t06z f006 + yesterday t12z f001-f120
        """
        base_date = datetime.strptime(self.config.pdy, "%Y%m%d")
        prev_date = base_date - timedelta(days=1)
        today_str = base_date.strftime("%Y%m%d")
        prev_str = prev_date.strftime("%Y%m%d")

        product = self.config.nwm_product  # "medium_range_mem1"
        n_target = self.config.nwm_n_list_target
        n_min = self.config.nwm_n_list_min
        min_size = 10_000_000  # 10MB

        def _resolve_nwm_file(date_str, cyc, fhr):
            """Resolve a single NWM file path."""
            nwm_dir = self.input_path / f"nwm.{date_str}" / product
            pattern = f"nwm.t{cyc:02d}z.medium_range.channel_rt_1.f{fhr:03d}.conus.nc"
            path = nwm_dir / pattern
            if path.exists() and path.stat().st_size >= min_size:
                return path
            return None

        # Primary list assembly
        primary = []
        # Yesterday t06z f006
        f = _resolve_nwm_file(prev_str, 6, 6)
        if f:
            primary.append(f)
        # Yesterday t12z f001-f006
        for fhr in range(1, 7):
            f = _resolve_nwm_file(prev_str, 12, fhr)
            if f:
                primary.append(f)
        # Yesterday t18z f001-f006
        for fhr in range(1, 7):
            f = _resolve_nwm_file(prev_str, 18, fhr)
            if f:
                primary.append(f)
        # Today t00z f001-f006
        for fhr in range(1, 7):
            f = _resolve_nwm_file(today_str, 0, fhr)
            if f:
                primary.append(f)
        # Today t06z f001-f120 (shell: f0{0-9}? + f1{0-2}? = f000-f120)
        for fhr in range(1, 121):
            f = _resolve_nwm_file(today_str, 6, fhr)
            if f:
                primary.append(f)

        # Backup list (shell: yesterday t06z f006 + t12z f001-f129)
        backup = []
        f = _resolve_nwm_file(prev_str, 6, 6)
        if f:
            backup.append(f)
        for fhr in range(1, 130):
            f = _resolve_nwm_file(prev_str, 12, fhr)
            if f:
                backup.append(f)

        # Merge if primary incomplete (deduplicate by filepath)
        if len(primary) >= 2:
            result = primary
            if len(primary) < n_target and backup:
                primary_set = set(str(f) for f in primary)
                supplement = [f for f in backup if str(f) not in primary_set]
                n_need = min(n_target - len(primary), len(supplement))
                result = primary + supplement[:n_need]
                if n_need > 0:
                    log.info(f"NWM: merged {n_need} backup files "
                             f"(primary={len(primary)}, total={len(result)})")
        elif len(backup) >= 2:
            result = backup
            log.info(f"NWM: using backup list ({len(backup)} files)")
        else:
            result = primary or backup

        log.info(f"NWM STOFS file discovery: {len(result)} files "
                 f"(target={n_target}, min={n_min})")
        return result

    def _extract_streamflow(self, nwm_files: List[Path]) -> Tuple[np.ndarray, List[float]]:
        """Extract streamflow for configured rivers from NWM files.

        Time axis anchor: ``t=0`` is the phase start time (see
        :meth:`_phase_start_time`). For each file we resolve the actual
        valid time -- from the ``model_output_valid_time`` NetCDF
        attribute when present, otherwise from the filename via
        ``_nwm_valid_time`` -- and convert to ``hours from start_time``.

        The earlier file-index fallback (``len(all_times) * 1.0``) silently
        labelled file 0 as ``t=0`` regardless of its actual valid time. For
        analysis_assim runs that include pre-cycle lookback (tmHH > 0) this
        placed the cycle-hour flow at a positive offset and shifted every
        downstream row by the lookback depth, producing a vsource.th
        misaligned with SCHISM's ``model_t0``. Using actual valid times
        anchors the axis at the phase ``start_time`` and lets
        ``_normalize_to_simulation_grid`` drop / back-fill correctly.
        """
        n_rivers = self.river_config.n_rivers
        feature_ids = set(self.river_config.feature_ids)

        start_time = self._phase_start_time()

        all_flows = []
        all_times = []

        for nwm_file in nwm_files:
            try:
                ds = Dataset(str(nwm_file))
                file_features = ds.variables["feature_id"][:]
                streamflow = ds.variables["streamflow"][:]

                # Build feature_id -> index mapping for this file
                fid_to_idx = {}
                for i, fid in enumerate(file_features):
                    if int(fid) in feature_ids:
                        fid_to_idx[int(fid)] = i

                # Extract flow for each configured river
                flows = np.zeros(n_rivers, dtype=np.float32)
                for r_idx, fid in enumerate(self.river_config.feature_ids):
                    if fid in fid_to_idx:
                        flows[r_idx] = streamflow[fid_to_idx[fid]]
                    else:
                        flows[r_idx] = self.river_config.clim_flows[r_idx]

                # Resolve valid time -> hours from model_t0
                model_time: Optional[datetime] = None
                if hasattr(ds, "model_output_valid_time"):
                    try:
                        model_time = datetime.strptime(
                            ds.model_output_valid_time, "%Y-%m-%d_%H:%M:%S"
                        )
                    except (TypeError, ValueError):
                        model_time = None
                if model_time is None:
                    fname_time = _nwm_valid_time(nwm_file)
                    if fname_time != datetime.min:
                        model_time = fname_time

                if model_time is not None:
                    t_hours = (model_time - start_time).total_seconds() / 3600.0
                else:
                    # Last-resort sequential fallback (matches prior behavior
                    # for tests / fixtures that don't carry valid-time info).
                    t_hours = float(len(all_times))

                all_flows.append(flows)
                all_times.append(t_hours)

                ds.close()
            except Exception as e:
                log.warning(f"Failed to read {nwm_file.name}: {e}")
                continue

        if not all_flows:
            return np.array([]), []

        return np.stack(all_flows, axis=0), all_times

    def _extract_streamflow_aggregated(
        self, nwm_files: List[Path],
    ) -> Tuple[np.ndarray, List[float]]:
        """Extract streamflow summed per source element (STOFS gen_sourcesink logic).

        Ports the core logic from VIMS gen_sourcesink.py:
        - Each source element has a list of NWM feature_ids
        - Flow is summed across all feature_ids per source element
        - Negative flows are clamped to zero
        """
        groups = self.river_config.feature_id_groups
        n_sources = len(groups)

        all_flows = []
        all_times = []
        ref_feature_ids = None  # feature_id array from first file (for index lookup)
        src_ncidxs = None  # index groups mapping into feature_id array

        start_time = self._phase_start_time()

        for nwm_file in nwm_files:
            try:
                ds = Dataset(str(nwm_file))
                file_features = np.array(ds.variables["feature_id"][:])
                streamflow = np.array(ds.variables["streamflow"][:])

                # Clamp negative flows to zero (matching gen_sourcesink.py)
                streamflow[streamflow < -1e-5] = 0.0
                # Fill masked values with zero
                if hasattr(streamflow, 'mask'):
                    streamflow = np.where(streamflow.mask, 0.0, streamflow.data)

                # Build index mapping (rebuild if feature_id array changes)
                if ref_feature_ids is None or not np.array_equal(file_features, ref_feature_ids):
                    ref_feature_ids = file_features
                    src_ncidxs = []
                    for group in groups:
                        idxs = []
                        for fid in group:
                            matches = np.where(ref_feature_ids == int(fid))[0]
                            if len(matches) > 0:
                                idxs.append(matches[0])
                        src_ncidxs.append(idxs)

                # Sum flow per source element
                flows = np.zeros(n_sources, dtype=np.float32)
                for s_idx, idxs in enumerate(src_ncidxs):
                    if idxs:
                        flows[s_idx] = np.sum(streamflow[idxs])

                all_flows.append(flows)

                # Parse valid time from NetCDF attribute
                if hasattr(ds, 'model_output_valid_time'):
                    model_time = datetime.strptime(
                        ds.model_output_valid_time, "%Y-%m-%d_%H:%M:%S")
                    t_seconds = (model_time - start_time).total_seconds()
                else:
                    t_seconds = len(all_times) * 3600.0  # hourly fallback

                all_times.append(t_seconds / 3600.0)  # hours from start
                ds.close()

            except Exception as e:
                log.warning(f"Failed to read {nwm_file.name}: {e}")
                continue

        if not all_flows:
            return np.array([]), []

        log.info(f"Extracted aggregated flow for {n_sources} sources "
                 f"from {len(all_flows)} NWM files")
        return np.stack(all_flows, axis=0), all_times

    def _fix_search_dirs(self) -> List[Path]:
        """FIX directories to search for static river files (in priority order)."""
        import os
        dirs: List[Path] = []
        if self.config.river_config_file:
            dirs.append(Path(self.config.river_config_file).parent)
        for env_var in ("FIXstofs3d", "FIXofs"):
            env_val = os.environ.get(env_var)
            if env_val:
                dirs.append(Path(env_val))
        dirs.append(self.input_path)
        return dirs

    def _copy_static_river_file(self, output_name: str,
                                candidates: List[str]) -> Optional[Path]:
        """Search FIX dirs for any of *candidates* and copy to output/{output_name}."""
        import shutil
        output_file = self.output_path / output_name
        for fix_dir in self._fix_search_dirs():
            for name in candidates:
                src = fix_dir / name
                if src.exists():
                    shutil.copy2(src, output_file)
                    log.info(f"Copied static {name} from {fix_dir} -> {output_name}")
                    return output_file
        return None

    def _copy_static_msource(self) -> Optional[Path]:
        """Copy static msource.th from FIX directory (STOFS convention)."""
        path = self._copy_static_river_file(
            "msource.th",
            [
                "stofs_3d_atl_river_msource.th",
                "secofs_ufs.msource.th",
                "secofs.msource.th",
                "msource.th",
            ],
        )
        if path is not None:
            return path

        # Fallback: generate msource.th with default T/S in v3.9.1 PACKED
        # format (all temps then all salts, NOT interleaved). Format matches
        # `pyschism`-derived ``nwm_base.py:Msource.__str__``:
        #   {rel_time:G} {T_1:.4e} ... {T_N:.4e} {S_1:.4e} ... {S_N:.4e}
        # Single timestep at t=0 with constant T/S (SCHISM repeats last).
        log.warning("Static msource.th not found, generating with defaults (packed)")
        n_rivers = self.river_config.n_rivers
        temp = self.config.river_default_temp
        salt = self.config.river_default_salt
        output_file = self.output_path / "msource.th"
        try:
            with open(output_file, "w") as f:
                temps = " ".join(f"{temp:.4e}" for _ in range(n_rivers))
                salts = " ".join(f"{salt:.4e}" for _ in range(n_rivers))
                f.write(f"{0:G} {temps} {salts}\n")
            return output_file
        except Exception as e:
            log.error(f"Failed to write msource.th: {e}")
            return None

    def _copy_static_vsink(self) -> Optional[Path]:
        """Copy static vsink.th from FIX directory (STOFS convention).

        Search order tries the most specific names first so a SECOFS-UFS
        deploy picks up its own symlinked file rather than a STOFS-3D-ATL
        sibling that may live in a shared FIX root.
        """
        candidates = [
            "stofs_3d_atl_river_vsink.th",
            "secofs_ufs.vsink.th",
            "secofs.vsink.th",
            "vsink.th",
        ]
        path = self._copy_static_river_file("vsink.th", candidates)
        if path is None:
            # No fallback generation — STOFS relies on the FIX file being
            # provisioned; without it, vsink simply isn't staged and SCHISM
            # skips volume sinks.
            log.warning(
                "Static vsink.th not found in FIX dirs; skipping volume-sink file"
            )
        return path

    def _generate_climatology(self, n_steps: int) -> Tuple[np.ndarray, List[float]]:
        """Generate climatology-based river forcing.

        For STOFS-mode (sources.json), ``clim_flows`` is filled with zeros at
        load time because the JSON only carries element->feature_id maps with
        no flow estimate. Returning all-zeros for every source produced the
        V18 SECOFS-UFS bug where ``vsource.th`` / ``vsink.th`` were entirely
        0.0 when NWM file discovery failed (and SCHISM ran with no river
        forcing at all). Fall back to ``config.river_default_flow`` for any
        source whose per-river climatology is non-positive so the run still
        gets a sensible (small) inflow signal.
        """
        n_rivers = self.river_config.n_rivers
        month = int(self.config.pdy[4:6])
        factor = MONTHLY_FLOW_FACTOR.get(month, 1.0)
        default_flow = float(getattr(self.config, "river_default_flow", 1.0))

        clim = self.river_config.clim_flows
        flows = np.zeros((n_steps, n_rivers), dtype=np.float32)
        n_default = 0
        for r_idx in range(n_rivers):
            base = clim[r_idx] if r_idx < len(clim) else 0.0
            if base is None or base <= 0.0:
                base = default_flow
                n_default += 1
            flows[:, r_idx] = base * factor

        if n_default:
            log.info(
                f"Climatology fallback: {n_default}/{n_rivers} rivers used "
                f"river_default_flow={default_flow:.2f} m^3/s "
                f"(× monthly factor {factor:.2f})"
            )

        times = [float(i) for i in range(n_steps)]  # hourly
        return flows, times

    def _normalize_to_simulation_grid(
        self, flows: np.ndarray, times: List[float], n_target: int,
    ) -> Tuple[np.ndarray, List[float]]:
        """Snap raw extracted flows onto a dense hourly simulation grid.

        ``_extract_streamflow*`` returns ``times`` in *hours from start_time*
        (= cycle - nowcast_hours). Two issues if used directly:

        1. **Pre-nowcast lookback.** Analysis_assim cycles include lookback
           hours (tm00..tm02) that span ~14h before the latest cycle. For
           a 6h SECOFS-UFS nowcast that's 8h of pre-window data. SCHISM
           expects a time axis starting at 0.
        2. **Cycle gaps.** Files exist at irregular intervals (within-cycle
           1h steps + 4h gaps between cycles). The output must be dense
           hourly so the SCHISM source-interp doesn't stall.

        Strategy: snap each (time, flow) pair to its integer hour, drop
        anything outside ``[0, n_target)``, keep the first occurrence of
        any duplicate hour (analysis_assim sorted before short_range when
        they share a valid hour, which is the right preference for
        observed past), then forward-fill missing hours from the most
        recent prior flow. Returns at most ``n_target`` rows; the caller
        still pads to ``n_target`` if NWM data ends short of the window.
        """
        if flows.size == 0 or not times:
            return flows, times

        hour_to_flow: Dict[int, np.ndarray] = {}
        for i, t_h in enumerate(times):
            h = int(round(float(t_h)))
            if h < 0 or h >= n_target:
                continue
            if h not in hour_to_flow:
                hour_to_flow[h] = flows[i]

        if not hour_to_flow:
            log.warning(
                "All extracted NWM times fell outside the simulation window "
                f"[0, {n_target}h); falling through to climatology pad"
            )
            return flows[:0], []

        sorted_hours = sorted(hour_to_flow)
        first_hour = sorted_hours[0]

        # SCHISM source-input readers require the time axis to start at 0.
        # If the earliest available data is at hour first_hour > 0 (e.g.,
        # the cycle's analysis_assim window didn't include tm00 or the
        # earliest available product was after start_time), back-fill
        # hours [0, first_hour) with the flow from hour first_hour. The
        # forward-fill loop below does this automatically: for h <
        # first_hour, `h not in hour_to_flow` is True so `last_flow`
        # stays at its initial value (= hour_to_flow[first_hour]) and
        # gets appended verbatim.
        out_flows: List[np.ndarray] = []
        out_times: List[float] = []
        last_flow = hour_to_flow[first_hour]
        for h in range(0, n_target):
            if h in hour_to_flow:
                last_flow = hour_to_flow[h]
            out_flows.append(last_flow)
            out_times.append(float(h))

        n_back_filled = first_hour  # hours [0, first_hour) all use hour_to_flow[first_hour]
        n_forward_filled = sum(
            1 for h in range(first_hour, n_target) if h not in hour_to_flow
        )
        n_filled = n_back_filled + n_forward_filled
        if n_filled:
            log.info(
                f"NWM grid normalize: {len(hour_to_flow)} unique-hour rows from "
                f"{len(times)} files, back-filled {n_back_filled} pre-window "
                f"hours [0, {first_hour}), forward-filled {n_forward_filled} "
                f"intra-window gap hours, output starts at h=0"
            )
        else:
            log.info(
                f"NWM grid normalize: {len(hour_to_flow)} unique-hour rows, "
                f"no gaps, output starts at h=0"
            )

        return np.stack(out_flows, axis=0), out_times

    def _pad_to_target(self, flows: np.ndarray, times: List[float],
                       n_target: int) -> Tuple[np.ndarray, List[float]]:
        """Pad flow data to target number of time steps by repeating last row."""
        if flows.size == 0:
            return flows, times

        n_current = flows.shape[0]
        if n_current >= n_target:
            return flows, times

        n_pad = n_target - n_current
        last_row = flows[-1:, :]
        pad_data = np.repeat(last_row, n_pad, axis=0)
        flows = np.vstack([flows, pad_data])

        last_time = times[-1] if times else 0.0
        times.extend([last_time + (i + 1) for i in range(n_pad)])

        log.info(f"Padded river forcing from {n_current} to {n_target} time steps")
        return flows, times

    def _write_vsource(self, flows: np.ndarray, times: List[float]) -> Optional[Path]:
        """Write vsource.th — volume source time history.

        Format matches v3.9.1 production (`pyschism`-derived
        `nwm_base.py:TimeHistoryFile.__str__`):

            f"{relative_time_seconds:G} {flow_1:.4e} {flow_2:.4e} ..."

        i.e., general-scientific time + 4-digit scientific values per
        source. ``times`` arrives already conditioned to a dense hourly
        grid starting at 0 by ``_normalize_to_simulation_grid``; the
        writer just emits ``t_hours * 3600`` per row.

        Time-axis anchor: row ``t=0`` corresponds to SCHISM's ``model_t0``
        (= ``cycle - nowcast_hours``). The extractors and the climatology
        fallback both produce ``times`` in *hours from model_t0*, so the
        relative seconds written here align with SCHISM's run start. For
        SECOFS-UFS that means t=0 is ``cycle - 6h``.
        """
        output_file = self.output_path / "vsource.th"
        try:
            with open(output_file, "w") as f:
                for t_idx, t_hours in enumerate(times):
                    t_seconds = t_hours * 3600.0
                    values = " ".join(f"{flows[t_idx, r]:.4e}"
                                      for r in range(flows.shape[1]))
                    f.write(f"{t_seconds:G} {values}\n")

            log.info(f"Created {output_file.name}: {flows.shape[0]} steps, {flows.shape[1]} rivers")
            return output_file
        except Exception as e:
            log.error(f"Failed to write vsource.th: {e}")
            return None

    def _write_vsink(self, times: List[float]) -> Optional[Path]:
        """Write vsink.th — volume sink time history (zero flow).

        SCHISM requires vsink.th columns to match ``nsink`` from
        source_sink.in. NWM doesn't provide sink flow data and the static
        FIX vsink.th is keyed to an older sinks.json with a different sink
        count, so we generate a zero-flow time series sized to the current
        ``n_sinks``. This declares the sink elements without imposing any
        diversion volume — equivalent to running without active diversions.

        Format matches v3.9.1 production (`pyschism`-derived
        `nwm_base.py:TimeHistoryFile.__str__`):
        ``{rel_time:G} {value:.4e} ...`` per row.

        Time-axis anchor: same as vsource.th. Row ``t=0`` corresponds to
        ``model_t0 = cycle - nowcast_hours`` so the sink/source axes are
        co-aligned (SCHISM reads both inside the same source-term step).
        """
        output_file = self.output_path / "vsink.th"
        n_sinks = self.river_config.n_sinks
        if n_sinks == 0:
            return None
        try:
            with open(output_file, "w") as f:
                zeros = " ".join([f"{0.0:.4e}"] * n_sinks)
                for t_idx, t_hours in enumerate(times):
                    t_seconds = t_hours * 3600.0
                    if self.is_stofs_mode:
                        if t_idx == 0:
                            t_seconds = 0.0
                        elif t_idx == 1:
                            t_seconds = 3600.0
                    f.write(f"{t_seconds:G} {zeros}\n")
            log.info(f"Created vsink.th: {len(times)} steps, {n_sinks} sinks (zero volume)")
            return output_file
        except Exception as e:
            log.error(f"Failed to write vsink.th: {e}")
            return None

    def _load_ctl_river_config(self) -> Optional[RiverConfig]:
        """Load the COMF-style river.ctl as a separate RiverConfig.

        SECOFS-UFS uses BOTH a STOFS sources.json (interior source/sink) and
        a COMF river.ctl (open-boundary USGS rivers like Savannah, Fraser,
        Columbia). They drive different SCHISM forcing files: source_sink.in
        family vs. flux.th / temp.th / salt.th. This loader returns the
        boundary-flux side. Returns None when river_ctl_file isn't set.
        """
        ctl_path = self.config.river_ctl_file
        if not ctl_path:
            return None
        ctl_path = Path(ctl_path)
        if not ctl_path.exists():
            log.warning(f"river_ctl_file not found: {ctl_path}")
            return None
        try:
            return RiverConfig.from_text(ctl_path)
        except Exception as e:
            log.warning(f"Failed to load river.ctl: {e}")
            return None

    def _write_river_th_files(self, times: List[float],
                              ctl_cfg: RiverConfig) -> List[Path]:
        """Write SCHISM open-boundary forcing: schism_flux.th / temp.th / salt.th.

        Mirrors legacy Fortran ``nos_ofs_create_forcing_river.f`` formula
        (lines 1893-1898) per grid point ``i`` at time ``N``:

            flux  = -river_q(RiverID_Q[i], N) * Q_Scale[i]   ! negate for inflow
            temp  =  max(1.0, river_T(RiverID_T[i], N) * T_Scale[i])
            salt  =  0.005                                    ! constant per F90:1898

        Format matches production Fortran (``nos_ofs_create_forcing_river.f``
        formats 600/601):
            - flux.th: ``F12.0,50F12.2``  (time secs, F12.2 values)
            - temp/salt: ``F12.0,50F12.4``
            - First row at t=0 (forced; legacy F90:2016 convention)
            - Later rows skipped if ``t_sec <= 0.5`` (legacy F90:2022)
            - Time written as Fortran-style ``F12.0`` with trailing dot
              (``"        120."`` not ``"         120"``)

        Q/T source priority (highest first):
            1. USGS observation timeseries (TODO: BUFR or NWIS fetcher)
            2. Daily climatology from ``river.clim.usgs.nc`` (TODO once
               file is available)
            3. Annual ``Q_mean`` / ``T_mean`` from river.ctl Section 1
               (current path — smoke-test fallback)

        Salt is hardcoded 0.005 PSU per the Fortran reference — the legacy
        does NOT read salt observations.
        """
        n_riv = ctl_cfg.n_rivers
        if n_riv == 0:
            return []

        # If per-grid scaling is populated (parsed from river.ctl), use the
        # production formula. Otherwise fall back to the simpler clim_flows
        # array (used by ad-hoc RiverConfig instantiation in tests).
        use_production_formula = bool(ctl_cfg.river_id_q and ctl_cfg.q_scale)

        if use_production_formula:
            # Resolve per-grid Q and T from station_id mapping × scale.
            # Source priority for the per-station Q/T value:
            #   1. Daily climatology from river.clim.usgs.nc (3-day
            #      centered average, matching the Fortran behavior at
            #      `nos_ofs_create_forcing_river.f` lines 1499-1551).
            #   2. Annual Q_mean / T_mean from river.ctl Section 1.
            station_q: Dict[int, float] = {}
            station_t: Dict[int, float] = {}
            clim_label = "annual Q_mean/T_mean"
            clim_path = self.config.river_clim_file
            if clim_path is not None and Path(clim_path).exists():
                try:
                    from datetime import datetime
                    from .river_clim import load_usgs_climatology, _find_clim_index
                    pdy_dt = datetime.strptime(self.config.pdy, "%Y%m%d")
                    for sid_int, usgs_id in ctl_cfg.station_usgs_ids.items():
                        try:
                            months, days, dis, tem, sal = load_usgs_climatology(
                                Path(clim_path), str(usgs_id))
                            ci = _find_clim_index(months, days, pdy_dt)
                            n = len(dis)
                            window = [(ci - 1) % n, ci, (ci + 1) % n]
                            q_avg = float(np.mean([dis[d] for d in window]))
                            t_avg = float(np.mean([tem[d] for d in window]))
                            station_q[sid_int] = q_avg
                            station_t[sid_int] = t_avg
                        except (ValueError, KeyError):
                            # Station not in climatology netCDF — keep ctl Q_mean
                            pass
                    if station_q:
                        clim_label = (f"daily climatology 3-day avg "
                                      f"(pdy={self.config.pdy}, {len(station_q)} stations)")
                except Exception as e:
                    log.warning(f"Failed to read river climatology, falling "
                                f"back to ctl Q_mean: {e}")

            q_per_river: List[float] = []
            t_per_river: List[float] = []
            for i in range(n_riv):
                sid_q = ctl_cfg.river_id_q[i]
                sid_t = ctl_cfg.river_id_t[i]
                q_obs = station_q.get(sid_q, ctl_cfg.stations_q_mean.get(sid_q, 50.0))
                t_obs = station_t.get(sid_t, ctl_cfg.stations_t_mean.get(sid_t, 15.0))
                q_per_river.append(q_obs * ctl_cfg.q_scale[i])
                t_per_river.append(max(1.0, t_obs * ctl_cfg.t_scale[i]))
        else:
            q_per_river = list(ctl_cfg.clim_flows)
            month = int(self.config.pdy[4:6])
            t_default = MONTHLY_RIVER_TEMP.get(month, self.config.river_default_temp)
            t_per_river = [t_default] * n_riv
            clim_label = "legacy clim_flows (no per-station data)"

        salt_const = 0.005  # F90:1898 hardcoded

        flux_path = self.output_path / "schism_flux.th"
        temp_path = self.output_path / "schism_temp.th"
        salt_path = self.output_path / "schism_salt.th"

        # Fortran F12.0 emits "0." not "0" — match that.
        def _fmt_f12_0(t_sec: float) -> str:
            return f"{t_sec:11.0f}."

        # Generate time grid at the SCHISM model dt (production: 120 sec).
        # This produces a 1651-row file for a 55-hour run (matches
        # production cadence) instead of an hourly 55-row one. Q/T values
        # are constant across rows (climatology over the run window) so
        # the only effect is to give SCHISM denser interpolation points.
        #
        # Endpoint rule: legacy Fortran extends ``schism_flux.th`` by 1
        # hour past the forecast end (``end = forecast_hours + 1``), giving
        # SCHISM a safe interpolation buffer at the simulation tail. The
        # hourly ``times`` grid above carries a different buffer
        # (``time_hotstart + 72h`` for SECOFS) so we anchor the schism_*.th
        # end on the phase window + ``river_th_extra_hours`` rather than
        # ``times[-1]`` to avoid overshooting.
        extra = max(0, int(self.config.river_th_extra_hours))
        if self.phase == "nowcast":
            sim_hours = int(self.config.nowcast_hours) + extra
        elif self.phase == "forecast":
            sim_hours = int(self.config.forecast_hours) + extra
        else:
            sim_hours = (int(self.config.nowcast_hours)
                         + int(self.config.forecast_hours)
                         + extra)
        end_sec = max(0.0, float(sim_hours) * 3600.0)
        dt = max(1.0, float(self.config.schism_dt))
        # Build [0, dt, 2dt, ..., end_sec] inclusive
        n_steps = int(round(end_sec / dt)) + 1
        time_grid_sec = [k * dt for k in range(n_steps)]

        try:
            with open(flux_path, "w") as ff, \
                 open(temp_path, "w") as tf, \
                 open(salt_path, "w") as sf:
                for t_sec in time_grid_sec:
                    t_str = _fmt_f12_0(t_sec)
                    ff.write(t_str
                             + "".join(f"{-q:12.2f}" for q in q_per_river)
                             + "\n")
                    tf.write(t_str
                             + "".join(f"{t:12.4f}" for t in t_per_river)
                             + "\n")
                    sf.write(t_str
                             + "".join(f"{salt_const:12.4f}" for _ in q_per_river)
                             + "\n")

            mode = "production formula (per-grid station × Q_Scale)" \
                if use_production_formula else "legacy clim_flows fallback"
            log.info(f"Created schism_flux/temp/salt.th: {n_riv} grid points, "
                     f"{len(time_grid_sec)} rows at {dt:.0f}-sec cadence, "
                     f"{mode}, Q/T source = {clim_label}; "
                     f"avg Q={sum(q_per_river)/n_riv:.1f} m^3/s, "
                     f"avg T={sum(t_per_river)/n_riv:.2f}C, S={salt_const:.4f}")
            return [flux_path, temp_path, salt_path]
        except Exception as e:
            log.error(f"Failed to write schism_*.th boundary forcing: {e}")
            return []

    def _write_msource(self, times: List[float]) -> Optional[Path]:
        """Write msource.th — mass source (temperature, salinity).

        Format matches v3.9.1 production ``Msource.__str__``: PACKED
        (all temps then all salts, NOT interleaved):

            {rel_time:G} {T_1:.4e} ... {T_N:.4e} {S_1:.4e} ... {S_N:.4e}

        Values: production uses ``-9999.0`` as a sentinel meaning
        "use ambient T/S at the source element" (not a forced value).
        Production COMOUT confirms all-``-9999`` content; matching it
        avoids forcing artificial source-water properties that don't
        match the surrounding ambient cells.

        Time-axis anchor: shares the ``times`` array with ``vsource.th``
        so row ``t=0`` corresponds to ``model_t0 = cycle - nowcast_hours``.
        Keeping the three source files (vsource/vsink/msource) on the same
        axis is a SCHISM requirement -- the source-term reader allocates
        one buffer for all of them.
        """
        output_file = self.output_path / "msource.th"
        n_rivers = self.river_config.n_rivers
        # Production sentinel -- SCHISM treats this as "ambient water properties"
        SENTINEL = -9999.0

        try:
            with open(output_file, "w") as f:
                vals = " ".join(f"{SENTINEL:.4e}" for _ in range(2 * n_rivers))
                for t_hours in times:
                    t_seconds = t_hours * 3600.0
                    f.write(f"{t_seconds:G} {vals}\n")

            log.info(f"Created {output_file.name}: ambient sentinel "
                     f"({SENTINEL}), packed format, {n_rivers} rivers, "
                     f"{len(times)} steps")
            return output_file
        except Exception as e:
            log.error(f"Failed to write msource.th: {e}")
            return None

    def _write_source_sink(self) -> Optional[Path]:
        """Write source_sink.in — SCHISM source/sink configuration.

        Format (matches the static FIX file shipped with deterministic SECOFS
        production, e.g. ``$FIXofs/secofs/secofs.source_sink.in``):

            <nsource>
            <source_elem_id_1>
            ...
            <source_elem_id_n>
            <blank>
            <nsink>
            <sink_elem_id_1>
            ...

        Single-column throughout — earlier output of ``<elem_id> 1`` plus a
        hardcoded ``0`` sinks made SCHISM read past the source list and hit
        EOF on unit 31 (`schism_init.F90:2918`).
        """
        output_file = self.output_path / "source_sink.in"
        try:
            with open(output_file, "w") as f:
                n_src = self.river_config.n_rivers
                f.write(f"{n_src}\n")
                for node_idx in self.river_config.node_indices:
                    f.write(f"{node_idx}\n")
                f.write("\n")  # blank separator (matches FIX baseline)

                n_sink = self.river_config.n_sinks
                f.write(f"{n_sink}\n")
                for node_idx in self.river_config.sink_node_indices:
                    f.write(f"{node_idx}\n")

            log.info(
                f"Created {output_file.name}: {n_src} sources, {n_sink} sinks"
            )
            return output_file
        except Exception as e:
            log.error(f"Failed to write source_sink.in: {e}")
            return None
