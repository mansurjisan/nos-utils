"""
SCHISM domain partition generator.

Creates partition.prop assigning mesh elements to MPI compute ranks.
Two methods:
  1. Round-robin (default): Simple element_id % nprocs assignment
  2. Contiguous blocks: Sequential block assignment

Input: hgrid.gr3 (SCHISM grid file) — reads n_elements from line 2
Output: partition.prop (one rank per line, n_elements lines)

Note: Production uses ParMETIS for load-balanced partitioning.
Round-robin is adequate for testing and containers.
"""

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

from ..config import ForcingConfig
from .base import ForcingProcessor, ForcingResult

log = logging.getLogger(__name__)


class PartitionProcessor(ForcingProcessor):
    """Generate partition.prop for SCHISM MPI domain decomposition."""

    SOURCE_NAME = "PARTITION"

    def __init__(
        self,
        config: ForcingConfig,
        input_path: Path,
        output_path: Path,
        nprocs: int = 1,
        grid_file: Optional[Path] = None,
        method: str = "round_robin",
    ):
        """
        Args:
            config: ForcingConfig
            input_path: Directory containing hgrid.gr3
            output_path: Output directory for partition.prop
            nprocs: Number of MPI compute processes
            grid_file: Explicit path to hgrid.gr3 (overrides input_path search)
            method: "round_robin" or "contiguous"
        """
        super().__init__(config, input_path, output_path)
        self.nprocs = nprocs
        self._grid_file = grid_file
        self.method = method

    def process(self) -> ForcingResult:
        """Generate partition.prop."""
        log.info(f"Partition processor: nprocs={self.nprocs}, method={self.method}")
        self.create_output_dir()

        # Find grid file
        grid_file = self._grid_file or self._find_grid()
        if grid_file is None:
            return ForcingResult(
                success=False, source=self.SOURCE_NAME,
                errors=["hgrid.gr3 not found"],
            )

        # Parse n_elements from grid
        n_elements = self._read_n_elements(grid_file)
        if n_elements <= 0:
            return ForcingResult(
                success=False, source=self.SOURCE_NAME,
                errors=[f"Invalid n_elements={n_elements} from {grid_file}"],
            )

        # Generate partition
        output_file = self.output_path / "partition.prop"
        self._write_partition(output_file, n_elements)

        log.info(f"Created partition.prop: {n_elements} elements -> {self.nprocs} ranks")
        return ForcingResult(
            success=True, source=self.SOURCE_NAME,
            output_files=[output_file],
            metadata={
                "n_elements": n_elements,
                "nprocs": self.nprocs,
                "method": self.method,
            },
        )

    def find_input_files(self) -> List[Path]:
        grid = self._find_grid()
        return [grid] if grid else []

    def _find_grid(self) -> Optional[Path]:
        """Find hgrid.gr3 in input_path."""
        for name in ["hgrid.gr3", "hgrid.ll", f"{self.config.pdy}_hgrid.gr3"]:
            p = self.input_path / name
            if p.exists():
                return p
        # Glob
        found = sorted(self.input_path.glob("*hgrid*"))
        return found[0] if found else None

    def _read_n_elements(self, grid_file: Path) -> int:
        """Read number of elements from hgrid.gr3 header (line 2)."""
        try:
            with open(grid_file) as f:
                f.readline()  # Line 1: comment
                line2 = f.readline().strip()
                parts = line2.split()
                n_elements = int(parts[0])
                return n_elements
        except (ValueError, IndexError, IOError) as e:
            log.error(f"Failed to parse hgrid header: {e}")
            return 0

    def _write_partition(self, output_path: Path, n_elements: int) -> None:
        """Write partition.prop file."""
        if self.method == "contiguous":
            ranks = _contiguous_partition(n_elements, self.nprocs)
        else:
            ranks = _round_robin_partition(n_elements, self.nprocs)

        with open(output_path, "w") as f:
            for rank in ranks:
                f.write(f"{rank}\n")


def _round_robin_partition(n_elements: int, nprocs: int) -> np.ndarray:
    """Assign elements to ranks in round-robin order."""
    return np.arange(n_elements) % nprocs


def _contiguous_partition(n_elements: int, nprocs: int) -> np.ndarray:
    """Assign contiguous blocks of elements to each rank."""
    ranks = np.zeros(n_elements, dtype=int)
    block_size = n_elements // nprocs
    remainder = n_elements % nprocs

    start = 0
    for rank in range(nprocs):
        size = block_size + (1 if rank < remainder else 0)
        ranks[start:start + size] = rank
        start += size

    return ranks
