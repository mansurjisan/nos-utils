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
import shutil
import subprocess
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

    def stage_init_to_comout(
        self,
        comout_dir: Path,
        init_filename: str,
    ) -> Optional[Path]:
        """Stage the previous-cycle restart as today's COMOUT init file.

        Removes the manual ``nccopy`` hand-off operators have been doing each
        cycle: walks back through previous SECOFS cycles (the existing
        ``_find_hotstart`` already orders by datetime, so we just consume
        its result) and lands a NETCDF4_CLASSIC copy at
        ``comout_dir/init_filename``. SECOFS production cycles every 6h
        (00z, 06z, 12z, 18z), so the natural pick is the cycle 6h prior;
        when that's missing the next-most-recent valid restart is used.

        Format conversion is mandatory because every restart produced by
        ``combine_hotstart7.exe`` is HDF5 (NF90_NETCDF4) and parallel-IO
        collective open at the SCHISM init scale (2794 ranks for SECOFS-UFS)
        segfaults inside libpnetcdf on HDF5 files. Conversion uses ``nccopy``
        when available (operational standard), with a netCDF4-Python
        fallback for environments that don't ship it.

        Args:
            comout_dir: Target ``$COMOUT`` directory (created if missing).
            init_filename: Operational init name, typically
                ``f"{prefix}.t{cyc:02d}z.{pdy}.init.nowcast.nc"``.

        Returns:
            Path to the staged init file, or ``None`` if no valid restart
            was found within the lookback window.
        """
        source = self._find_hotstart()
        if source is None:
            log.warning(
                "No previous-cycle restart found for COMOUT init staging; "
                "downstream nowcast will need to cold-start"
            )
            return None

        comout_dir = Path(comout_dir)
        comout_dir.mkdir(parents=True, exist_ok=True)
        target = comout_dir / init_filename

        # Skip work if the target already exists and points at the same
        # source — operators sometimes pre-stage manually.
        if target.exists():
            try:
                if target.resolve() == source.resolve():
                    log.info(f"Init already staged at {target} (matches source)")
                    return target
            except OSError:
                pass

        src_format = self._netcdf_format(source)
        if src_format == "NETCDF4_CLASSIC":
            shutil.copy2(source, target)
            log.info(f"Staged init.nowcast.nc (already classic): {source.name} → {target}")
            return target

        if not self._nccopy_to_classic(source, target):
            log.error(f"Failed to convert {source} to NETCDF4_CLASSIC; init not staged")
            return None

        log.info(
            f"Staged init.nowcast.nc (converted {src_format or '?'} "
            f"→ NETCDF4_CLASSIC): {source.name} → {target}"
        )
        return target

    @staticmethod
    def _netcdf_format(path: Path) -> Optional[str]:
        """Best-effort NetCDF format probe — returns None if unreadable."""
        if not HAS_NETCDF4:
            return None
        try:
            ds = Dataset(str(path), "r")
            try:
                return ds.file_format  # e.g. "NETCDF4", "NETCDF4_CLASSIC"
            finally:
                ds.close()
        except Exception as e:
            log.warning(f"Could not probe NetCDF format of {path}: {e}")
            return None

    @staticmethod
    def _nccopy_to_classic(src: Path, dst: Path) -> bool:
        """Convert ``src`` to NETCDF4_CLASSIC at ``dst``.

        Tries the operational ``nccopy -k 'netCDF-4 classic model'`` first;
        falls back to a pure netCDF4-Python rewrite when the binary isn't
        on PATH. Both produce a byte-identical SCHISM-readable file as far
        as the model's pnetcdf reader is concerned.
        """
        if shutil.which("nccopy"):
            try:
                subprocess.run(
                    ["nccopy", "-k", "netCDF-4 classic model", str(src), str(dst)],
                    check=True, capture_output=True,
                )
                return True
            except subprocess.CalledProcessError as e:
                log.warning(
                    f"nccopy failed (rc={e.returncode}); falling back to "
                    f"Python conversion. stderr: {e.stderr.decode(errors='replace')[:200]}"
                )

        if not HAS_NETCDF4:
            log.error("Neither nccopy nor netCDF4-Python available for format conversion")
            return False

        try:
            with Dataset(str(src), "r") as src_ds, \
                 Dataset(str(dst), "w", format="NETCDF4_CLASSIC") as dst_ds:
                # Global attrs
                dst_ds.setncatts({k: src_ds.getncattr(k) for k in src_ds.ncattrs()})
                # Dims
                for name, dim in src_ds.dimensions.items():
                    dst_ds.createDimension(name, len(dim) if not dim.isunlimited() else None)
                # Vars
                for name, var in src_ds.variables.items():
                    new_var = dst_ds.createVariable(
                        name, var.dtype, var.dimensions,
                    )
                    new_var.setncatts({k: var.getncattr(k) for k in var.ncattrs()})
                    new_var[:] = var[:]
            return True
        except Exception as e:
            log.error(f"Python NetCDF format conversion failed: {e}")
            return False

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

                for file_pattern in [
                    "hotstart*.nc",
                    f"{self.run_name}*hotstart*.nc",
                    f"{self.run_name}*.rst.nowcast.nc",   # COMF restart naming
                    f"{self.run_name}*.init.nowcast.nc",   # COMF init naming
                    f"{self.run_name}*restart*.nc",
                ]:
                    candidates.extend(sorted(dir_pattern.glob(file_pattern)))

        return candidates

    def _find_hotstart(self) -> Optional[Path]:
        """
        Find the correct hotstart file for the current cycle.

        For a 00z cycle, the hotstart should be from the previous cycle's
        nowcast (e.g., t18z yesterday). Selects by matching cycle time
        in filename, not by filesystem modification time.

        COMF naming: secofs.t{cyc}z.YYYYMMDD.rst.nowcast.nc
        """
        candidates = self.find_input_files()

        # Filter by size
        valid = []
        for f in candidates:
            try:
                if f.stat().st_size >= self.MIN_HOTSTART_SIZE:
                    valid.append(f)
            except OSError:
                continue

        if not valid:
            return None

        # Parse cycle time from filenames and find the one just before current cycle
        cycle_dt = datetime.strptime(self.config.pdy, "%Y%m%d") + \
                   timedelta(hours=self.config.cyc)

        scored = []
        for f in valid:
            file_dt = self._parse_file_datetime(f)
            if file_dt and file_dt < cycle_dt:
                # Prefer most recent file that's BEFORE current cycle
                scored.append((file_dt, f))

        if scored:
            scored.sort(key=lambda x: x[0], reverse=True)  # newest valid first
            best = scored[0][1]
            log.info(f"Selected hotstart by cycle time: {best.name} "
                     f"(valid {scored[0][0]}, current cycle {cycle_dt})")
            return best

        # Fallback: sort by mtime if no cycle time could be parsed
        valid.sort(key=lambda f: f.stat().st_mtime, reverse=True)

        log.info(f"Found {len(valid)} hotstart candidates, using newest: {valid[0].name}")
        return valid[0]

    def _parse_file_datetime(self, filepath: Path) -> Optional[datetime]:
        """
        Parse datetime from COMF restart filename.

        Patterns:
          secofs.t00z.20260402.rst.nowcast.nc → 2026-04-02 00:00
          secofs.t18z.20260401.rst.nowcast.nc → 2026-04-01 18:00
        """
        import re
        name = filepath.name
        # Match: {ofs}.t{HH}z.{YYYYMMDD}.rst or .init
        m = re.search(r"\.t(\d{2})z\.(\d{8})\.", name)
        if m:
            cyc_hour = int(m.group(1))
            date_str = m.group(2)
            try:
                return datetime.strptime(date_str, "%Y%m%d") + timedelta(hours=cyc_hour)
            except ValueError:
                pass
        return None

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
