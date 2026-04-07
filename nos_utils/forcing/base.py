"""
Base classes for forcing data processors.

All forcing processors (GFS, HRRR, GEFS) inherit from ForcingProcessor
and implement the process() and find_input_files() methods.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from ..config import ForcingConfig

log = logging.getLogger(__name__)


@dataclass
class ForcingResult:
    """Result from forcing data processing."""

    success: bool
    source: str
    output_files: List[Path] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.success


class ForcingProcessor(ABC):
    """
    Abstract base class for atmospheric forcing processors.

    Subclasses implement source-specific file discovery, GRIB2 extraction,
    variable conversion, and output writing.
    """

    # Subclasses set this
    SOURCE_NAME: str = "Unknown"

    # Minimum file size in bytes for QC (0 = no check)
    MIN_FILE_SIZE: int = 0

    def __init__(self, config: ForcingConfig, input_path: Path, output_path: Path):
        self.config = config
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)

    @abstractmethod
    def process(self) -> ForcingResult:
        """Process forcing data and generate output files."""
        ...

    @abstractmethod
    def find_input_files(self) -> List[Path]:
        """Discover input GRIB2 files for the run window."""
        ...

    def validate_file_size(self, path: Path, min_bytes: int = 0) -> bool:
        """QC: reject files smaller than threshold."""
        threshold = min_bytes or self.MIN_FILE_SIZE
        if threshold <= 0:
            return True
        try:
            return path.stat().st_size >= threshold
        except OSError:
            return False

    def create_output_dir(self) -> None:
        """Create output directory if needed."""
        self.output_path.mkdir(parents=True, exist_ok=True)

    def write_files_used(
        self, files: List[Path], output_dir: Path, source: str, phase: str,
    ) -> Path:
        """
        Write met_files_used log matching Fortran format.

        Format: filepath on one line, then "YYYY MM DD CYC FHR" on the next.
        Output filename: met_files_used_{phase}_{cyc:02d}_{source}.dat
        """
        import re

        cyc = self.config.cyc
        outfile = output_dir / f"met_files_used_{phase}_{cyc:02d}_{source}.dat"

        with open(outfile, "w") as f:
            for filepath in files:
                f.write(f"{filepath}\n")
                name = filepath.name
                # Extract cycle date/hour and forecast hour from filename
                # GFS: gfs.tHHz.pgrb2.0p50.fHHH  in dir gfs.YYYYMMDD/HH/
                # HRRR: hrrr.tHHz.wrfsfcfHH.grib2 in dir hrrr.YYYYMMDD/
                try:
                    # Get cycle hour from tHHz
                    m_cyc = re.search(r"\.t(\d{2})z\.", name)
                    cyc_hr = int(m_cyc.group(1)) if m_cyc else 0
                    # Get forecast hour
                    m_fhr = re.search(r"[\.f](\d{2,3})(?:\.|\Z)", name)
                    if not m_fhr:
                        m_fhr = re.search(r"wrfsfcf(\d{2})", name)
                    fhr = int(m_fhr.group(1)) if m_fhr else 0
                    # Get date from parent directory
                    for part in filepath.parts:
                        m_date = re.match(r"(?:gfs|hrrr)\.(\d{8})", part)
                        if m_date:
                            date_str = m_date.group(1)
                            break
                    else:
                        date_str = self.config.pdy
                    yyyy = date_str[:4]
                    mm = date_str[4:6]
                    dd = date_str[6:8]
                    f.write(f"{yyyy} {mm} {dd} {cyc_hr:02d} {fhr:03d}\n")
                except Exception:
                    f.write(f"{self.config.pdy[:4]} {self.config.pdy[4:6]} "
                            f"{self.config.pdy[6:8]} 00 000\n")

        log.info(f"Wrote {outfile.name}: {len(files)} files")
        return outfile

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(source={self.SOURCE_NAME})"
