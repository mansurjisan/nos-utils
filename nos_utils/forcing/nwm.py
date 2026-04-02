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
                 clim_flows: List[float], names: Optional[List[str]] = None):
        self.feature_ids = feature_ids
        self.node_indices = node_indices
        self.clim_flows = clim_flows
        self.names = names or [f"river_{i}" for i in range(len(feature_ids))]
        self.n_rivers = len(feature_ids)

    @classmethod
    def from_text(cls, filepath: Path) -> "RiverConfig":
        """Load river config from text file.

        Expected format (space-separated):
            feature_id  node_index  river_name  clim_flow  [clim_temp  clim_salt]
        """
        feature_ids = []
        node_indices = []
        clim_flows = []
        names = []

        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("!"):
                    continue
                parts = line.split()
                if len(parts) >= 4:
                    feature_ids.append(int(parts[0]))
                    node_indices.append(int(parts[1]))
                    names.append(parts[2])
                    clim_flows.append(float(parts[3]))

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
    ):
        """
        Args:
            config: ForcingConfig with river settings
            input_path: Root NWM data directory (COMINnwm)
            output_path: Output directory for river forcing files
            river_config: Pre-loaded river configuration (or loaded from config.river_config_file)
        """
        super().__init__(config, input_path, output_path)
        self._river_config = river_config

    @property
    def river_config(self) -> Optional[RiverConfig]:
        if self._river_config is None and self.config.river_config_file:
            path = Path(self.config.river_config_file)
            if path.suffix == ".json":
                self._river_config = RiverConfig.from_json(path)
            else:
                self._river_config = RiverConfig.from_text(path)
        return self._river_config

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
            },
        )

    def find_input_files(self) -> List[Path]:
        """Find NWM channel_rt files for the run window."""
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
        """Write vsource.th — volume source time history."""
        output_file = self.output_path / "vsource.th"
        try:
            with open(output_file, "w") as f:
                for t_idx, t_hours in enumerate(times):
                    t_seconds = t_hours * 3600.0
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
