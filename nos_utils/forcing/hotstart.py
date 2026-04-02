"""
SCHISM hotstart/restart file processor.

Reads SCHISM hotstart.nc to extract:
  - Model time (for computing rnday and time_hotstart)
  - Time step counter (iths)
  - Basic validation (file size, key variables present)

Searches for hotstart files from previous cycles with automatic date fallback.

Replaces: nos_ofs_read_restart (Fortran executable)

Key SCHISM hotstart.nc variables:
  time   — scalar: model time in seconds
  iths   — scalar: time step counter
  eta2   — [node]: surface elevation
  tr_nd  — [node, nVert, ntracers]: tracers at nodes
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

from ..config import ForcingConfig
from .base import ForcingProcessor, ForcingResult

log = logging.getLogger(__name__)

try:
    from netCDF4 import Dataset
    HAS_NETCDF4 = True
except ImportError:
    HAS_NETCDF4 = False


class HotstartInfo:
    """Information extracted from a SCHISM hotstart.nc file."""

    def __init__(
        self,
        filepath: Path,
        time_seconds: float,
        iths: int,
        n_nodes: int,
        n_levels: int,
    ):
        self.filepath = filepath
        self.time_seconds = time_seconds
        self.iths = iths
        self.n_nodes = n_nodes
        self.n_levels = n_levels

    @property
    def time_days(self) -> float:
        return self.time_seconds / 86400.0

    def __repr__(self):
        return (f"HotstartInfo(file={self.filepath.name}, "
                f"time={self.time_seconds:.0f}s ({self.time_days:.3f}d), "
                f"iths={self.iths}, nodes={self.n_nodes}, levels={self.n_levels})")


class HotstartProcessor(ForcingProcessor):
    """
    Find and validate SCHISM hotstart files.

    Searches for hotstart.nc from previous cycles, extracts timing info,
    and copies/links to the working directory.
    """

    SOURCE_NAME = "HOTSTART"

    # Minimum file size for a valid hotstart (SECOFS ~20GB, but small test files OK)
    MIN_HOTSTART_SIZE = 1000  # 1KB minimum (catches empty files)

    # Common hotstart file naming patterns
    HOTSTART_PATTERNS = [
        "hotstart.nc",
        "hotstart_it=*.nc",
        "{run}.hotstart.nc",
        "{run}.t{cyc:02d}z.hotstart.nc",
    ]

    def __init__(
        self,
        config: ForcingConfig,
        input_path: Path,
        output_path: Path,
        run_name: str = "secofs",
        max_lookback_days: int = 3,
    ):
        """
        Args:
            config: ForcingConfig
            input_path: Directory to search for hotstart files (COMOUT or restart archive)
            output_path: Working directory where hotstart.nc should be placed
            run_name: OFS run name for filename patterns (e.g., "secofs")
            max_lookback_days: Maximum days to search backward for hotstart
        """
        super().__init__(config, input_path, output_path)
        self.run_name = run_name
        self.max_lookback_days = max_lookback_days

    def process(self) -> ForcingResult:
        """
        Find and validate hotstart file.

        Returns HotstartInfo in result.metadata["hotstart_info"].
        """
        log.info(f"Hotstart processor: searching in {self.input_path}")
        self.create_output_dir()

        # Search for hotstart files
        hotstart_file = self._find_hotstart()
        if hotstart_file is None:
            log.warning("No hotstart file found — cold start will be used (ihot=0)")
            return ForcingResult(
                success=True, source=self.SOURCE_NAME,
                warnings=["No hotstart file found — cold start"],
                metadata={"ihot": 0, "hotstart_info": None},
            )

        # Read hotstart info
        info = self._read_hotstart(hotstart_file)
        if info is None:
            return ForcingResult(
                success=False, source=self.SOURCE_NAME,
                errors=[f"Failed to read hotstart: {hotstart_file}"],
            )

        log.info(f"Found hotstart: {info}")

        # Link/copy to output directory
        output_file = self.output_path / "hotstart.nc"
        if not output_file.exists() or output_file.resolve() != hotstart_file.resolve():
            try:
                if output_file.exists():
                    output_file.unlink()
                output_file.symlink_to(hotstart_file)
                log.info(f"Linked hotstart.nc -> {hotstart_file}")
            except OSError:
                import shutil
                shutil.copy2(hotstart_file, output_file)
                log.info(f"Copied hotstart.nc from {hotstart_file}")

        return ForcingResult(
            success=True, source=self.SOURCE_NAME,
            output_files=[output_file],
            metadata={
                "ihot": 1,
                "hotstart_info": info,
                "time_seconds": info.time_seconds,
                "time_days": info.time_days,
                "iths": info.iths,
                "source_file": str(hotstart_file),
            },
        )

    def find_input_files(self) -> List[Path]:
        """Find all candidate hotstart files."""
        candidates = []
        base_date = datetime.strptime(self.config.pdy, "%Y%m%d")

        for days_back in range(self.max_lookback_days + 1):
            date = base_date - timedelta(days=days_back)
            date_str = date.strftime("%Y%m%d")

            # Search in date-specific directories
            for dir_pattern in [
                self.input_path / f"{self.run_name}.{date_str}",
                self.input_path / date_str,
                self.input_path / f"{self.run_name}.{date_str}" / "restart_outputs",
                self.input_path,
            ]:
                if not dir_pattern.exists():
                    continue

                for file_pattern in ["hotstart*.nc", f"{self.run_name}*hotstart*.nc"]:
                    candidates.extend(sorted(dir_pattern.glob(file_pattern)))

        return candidates

    def _find_hotstart(self) -> Optional[Path]:
        """Find the most recent valid hotstart file."""
        candidates = self.find_input_files()

        # Filter by size and sort by modification time (newest first)
        valid = []
        for f in candidates:
            try:
                if f.stat().st_size >= self.MIN_HOTSTART_SIZE:
                    valid.append(f)
            except OSError:
                continue

        if not valid:
            return None

        # Sort by mtime, newest first
        valid.sort(key=lambda f: f.stat().st_mtime, reverse=True)

        log.info(f"Found {len(valid)} hotstart candidates, using newest: {valid[0].name}")
        return valid[0]

    def _read_hotstart(self, filepath: Path) -> Optional[HotstartInfo]:
        """Read timing and grid info from hotstart.nc."""
        if not HAS_NETCDF4:
            log.warning("netCDF4 not available — returning basic hotstart info")
            return HotstartInfo(
                filepath=filepath, time_seconds=0.0, iths=0,
                n_nodes=0, n_levels=0,
            )

        try:
            ds = Dataset(str(filepath))

            # Time: scalar or 1D array
            time_var = ds.variables.get("time")
            if time_var is not None:
                time_val = float(time_var[:].flat[0])
            else:
                time_val = 0.0

            # Time step counter
            iths_var = ds.variables.get("iths")
            iths = int(iths_var[:].flat[0]) if iths_var is not None else 0

            # Grid dimensions
            n_nodes = ds.dimensions.get("node", ds.dimensions.get("nSCHISM_hgrid_node"))
            n_nodes = n_nodes.size if n_nodes else 0

            n_levels = ds.dimensions.get("nVert", ds.dimensions.get("nSCHISM_vgrid_layers"))
            n_levels = n_levels.size if n_levels else 0

            ds.close()

            return HotstartInfo(
                filepath=filepath,
                time_seconds=time_val,
                iths=iths,
                n_nodes=n_nodes,
                n_levels=n_levels,
            )

        except Exception as e:
            log.error(f"Failed to read hotstart {filepath}: {e}")
            return None
