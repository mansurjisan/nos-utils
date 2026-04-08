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
                 feature_id_groups: Optional[List[List[int]]] = None):
        self.feature_ids = feature_ids
        self.node_indices = node_indices
        self.clim_flows = clim_flows
        self.names = names or [f"river_{i}" for i in range(len(feature_ids))]
        self.n_rivers = len(feature_ids)
        # STOFS: groups of feature_ids per source (flow summed within group)
        self.feature_id_groups = feature_id_groups

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
        stations = {}  # river_id -> {q_mean, t_mean, name}
        in_section1 = False
        sec1_header = None
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
                q_mean = float(parts[6])
                t_mean = float(parts[9]) if len(parts) > 9 else 15.0
                # Extract quoted name
                name = stripped.split('"')[1] if '"' in stripped else f"station_{parts[1]}"
                stations[rid] = {"q_mean": q_mean, "t_mean": t_mean, "name": name}
            except (ValueError, IndexError):
                continue

        # Parse Section 2: grid node mappings
        feature_ids = []
        node_indices = []
        clim_flows = []
        names = []
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
                river_id_q = int(parts[5])
                q_scale = float(parts[6])
                name = stripped.split('"')[1] if '"' in stripped else f"river_{grid_id}"

                # Compute climatological flow for this node
                if river_id_q in stations:
                    clim_flow = stations[river_id_q]["q_mean"] * q_scale
                else:
                    clim_flow = 50.0 * q_scale

                feature_ids.append(grid_id)
                node_indices.append(node_id)
                clim_flows.append(clim_flow)
                names.append(name.strip())
            except (ValueError, IndexError):
                continue

        log.info(f"Parsed river.ctl: {len(stations)} USGS stations, "
                 f"{len(feature_ids)} grid nodes")
        return cls(feature_ids, node_indices, clim_flows, names)

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
    def from_sources_json(cls, filepath: Path) -> "RiverConfig":
        """Load STOFS sources.json: {element_id: [fid1, fid2, ...]}.

        STOFS-3D-ATL uses VIMS gen_sourcesink.py format where each source
        element maps to one or more NWM feature_ids. Flow is summed across
        all feature_ids belonging to each source element.

        Args:
            filepath: Path to sources.json (e.g. stofs_3d_atl_river_sources_conus.json)
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

        log.info(f"Loaded STOFS sources.json: {n_sources} source elements "
                 f"from {filepath.name}")

        return cls(feature_ids, node_indices, clim_flows, names,
                   feature_id_groups=feature_id_groups)


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

    def __init__(
        self,
        config: ForcingConfig,
        input_path: Path,
        output_path: Path,
        river_config: Optional[RiverConfig] = None,
        phase: Optional[str] = None,
        time_hotstart: Optional[datetime] = None,
    ):
        """
        Args:
            config: ForcingConfig with river settings
            input_path: Root NWM data directory (COMINnwm)
            output_path: Output directory for river forcing files
            river_config: Pre-loaded river configuration (or loaded from config.river_config_file)
            phase: "nowcast" or "forecast" — determines time window for NWM files
            time_hotstart: Hotstart datetime (nowcast starts from here)
        """
        super().__init__(config, input_path, output_path)
        self._river_config = river_config
        self.phase = phase
        self.time_hotstart = time_hotstart

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
                            self._river_config = RiverConfig.from_sources_json(path)
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

        # Target time steps
        total_hours = self.config.nowcast_hours + self.config.forecast_hours
        n_target = total_hours + 1  # hourly

        # Try NWM data first
        nwm_files = self.find_input_files()
        if nwm_files:
            log.info(f"Found {len(nwm_files)} NWM files")
            if self.is_stofs_mode:
                flows, times = self._extract_streamflow_aggregated(nwm_files)
            else:
                flows, times = self._extract_streamflow(nwm_files)
        else:
            log.warning("No NWM files found — using climatology")
            flows, times = self._generate_climatology(n_target)

        # Pad to target length if needed
        if len(times) < n_target:
            flows, times = self._pad_to_target(flows, times, n_target)

        # Write output files
        output_files = []

        vsource = self._write_vsource(flows, times)
        if vsource:
            output_files.append(vsource)

        # STOFS: msource.th is a static FIX file, not generated
        if self.is_stofs_mode:
            msource = self._copy_static_msource()
        else:
            msource = self._write_msource(times)
        if msource:
            output_files.append(msource)

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
            },
        )

    def find_input_files(self) -> List[Path]:
        """Find NWM channel_rt files for the run window."""
        if self.is_stofs_mode or self.config.nwm_product == "medium_range_mem1":
            return self._find_stofs_nwm_files()
        return self._find_secofs_nwm_files()

    def _find_secofs_nwm_files(self) -> List[Path]:
        """Find NWM files using SECOFS product search (analysis_assim, short/medium_range)."""
        base_date = datetime.strptime(self.config.pdy, "%Y%m%d")
        nwm_files = []

        for date in [base_date, base_date - timedelta(days=1)]:
            date_str = date.strftime("%Y%m%d")
            nwm_dir = self.input_path / f"nwm.{date_str}"
            if not nwm_dir.exists():
                continue

            for product in self.PRODUCTS:
                pattern = f"nwm.t*z.{product}.channel_rt.f*.conus.nc"
                found = sorted(nwm_dir.glob(pattern))
                nwm_files.extend(found)

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
        # Today t06z f001-f119 (shell: f0{0-9}? + f1{0,1}? = f000-f119)
        for fhr in range(1, 120):
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

        # Merge if primary incomplete
        if len(primary) >= 2:
            result = primary
            if len(primary) < n_target and len(backup) > len(primary):
                n_supplement = len(backup) - len(primary)
                result = primary + backup[len(primary):len(primary) + n_supplement]
                log.info(f"NWM: merged {n_supplement} backup files "
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
        """Extract streamflow for configured rivers from NWM files."""
        n_rivers = self.river_config.n_rivers
        feature_ids = set(self.river_config.feature_ids)

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

                all_flows.append(flows)
                # Time in hours from start (approximate)
                all_times.append(len(all_times) * 1.0)  # hourly assumption

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

        cycle_dt = datetime.strptime(self.config.pdy, "%Y%m%d") + \
                   timedelta(hours=self.config.cyc)
        start_time = cycle_dt - timedelta(hours=self.config.nowcast_hours)

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

    def _copy_static_msource(self) -> Optional[Path]:
        """Copy static msource.th from FIX directory (STOFS convention)."""
        import shutil
        output_file = self.output_path / "msource.th"

        # Search for static msource.th in input_path (FIX directory)
        for name in ["stofs_3d_atl_river_msource.th", "msource.th"]:
            src = self.input_path / name
            if src.exists():
                shutil.copy2(src, output_file)
                log.info(f"Copied static {name} -> msource.th")
                return output_file

        # Fallback: generate msource.th with default T/S
        log.warning("Static msource.th not found, generating with defaults")
        n_rivers = self.river_config.n_rivers
        temp = self.config.river_default_temp
        salt = self.config.river_default_salt
        try:
            with open(output_file, "w") as f:
                # Single timestep at t=0 with constant T/S (SCHISM repeats last)
                values = " ".join(f"{temp:.1f} {salt:.1f}" for _ in range(n_rivers))
                f.write(f"0.0 {values}\n")
            return output_file
        except Exception as e:
            log.error(f"Failed to write msource.th: {e}")
            return None

    def _generate_climatology(self, n_steps: int) -> Tuple[np.ndarray, List[float]]:
        """Generate climatology-based river forcing."""
        n_rivers = self.river_config.n_rivers
        month = int(self.config.pdy[4:6])
        factor = MONTHLY_FLOW_FACTOR.get(month, 1.0)

        flows = np.zeros((n_steps, n_rivers), dtype=np.float32)
        for r_idx in range(n_rivers):
            flows[:, r_idx] = self.river_config.clim_flows[r_idx] * factor

        times = [float(i) for i in range(n_steps)]  # hourly
        return flows, times

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

        STOFS convention: first two time stamps forced to 0.0 and 3600.0
        to ensure SCHISM starts cleanly at t=0.
        """
        output_file = self.output_path / "vsource.th"
        try:
            with open(output_file, "w") as f:
                for t_idx, t_hours in enumerate(times):
                    t_seconds = t_hours * 3600.0
                    # STOFS: force first two time tags (shell script post-processing)
                    if self.is_stofs_mode:
                        if t_idx == 0:
                            t_seconds = 0.0
                        elif t_idx == 1:
                            t_seconds = 3600.0
                    values = " ".join(f"{flows[t_idx, r]:.4f}"
                                      for r in range(flows.shape[1]))
                    f.write(f"{t_seconds:.1f} {values}\n")

            log.info(f"Created {output_file.name}: {flows.shape[0]} steps, {flows.shape[1]} rivers")
            return output_file
        except Exception as e:
            log.error(f"Failed to write vsource.th: {e}")
            return None

    def _write_msource(self, times: List[float]) -> Optional[Path]:
        """Write msource.th — mass source (temperature, salinity)."""
        output_file = self.output_path / "msource.th"
        n_rivers = self.river_config.n_rivers
        month = int(self.config.pdy[4:6])
        temp = MONTHLY_RIVER_TEMP.get(month, self.config.river_default_temp)
        salt = self.config.river_default_salt

        try:
            with open(output_file, "w") as f:
                for t_hours in times:
                    t_seconds = t_hours * 3600.0
                    # Each river has temp, salt pair
                    values = " ".join(f"{temp:.1f} {salt:.1f}" for _ in range(n_rivers))
                    f.write(f"{t_seconds:.1f} {values}\n")

            log.info(f"Created {output_file.name}: temp={temp:.1f}°C, salt={salt:.1f} PSU")
            return output_file
        except Exception as e:
            log.error(f"Failed to write msource.th: {e}")
            return None

    def _write_source_sink(self) -> Optional[Path]:
        """Write source_sink.in — SCHISM source/sink configuration."""
        output_file = self.output_path / "source_sink.in"
        try:
            with open(output_file, "w") as f:
                n = self.river_config.n_rivers
                f.write(f"{n}\n")
                for node_idx in self.river_config.node_indices:
                    f.write(f"{node_idx} 1\n")
                f.write("0\n")  # num_sinks = 0

            log.info(f"Created {output_file.name}: {self.river_config.n_rivers} sources")
            return output_file
        except Exception as e:
            log.error(f"Failed to write source_sink.in: {e}")
            return None
