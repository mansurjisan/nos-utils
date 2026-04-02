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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(source={self.SOURCE_NAME})"
