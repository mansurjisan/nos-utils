"""
SCHISM grid reader (hgrid.gr3 / hgrid.ll).

Parses the SCHISM unstructured grid file to extract:
  - Node coordinates (lon, lat, depth)
  - Open boundary node indices and coordinates
  - Land boundary information

Works with any SCHISM-based OFS (SECOFS, STOFS-3D-ATL, CREOFS, etc.)

hgrid format:
  Line 1: comment
  Line 2: n_elements n_nodes
  Lines 3 to n_nodes+2: node_id lon lat depth
  Lines n_nodes+3 to n_nodes+n_elements+2: element connectivity
  After elements: open boundary section, then land boundary section
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class OpenBoundary:
    """A single open boundary segment."""
    index: int                    # Boundary segment index (0-based)
    node_ids: List[int]           # 1-based node IDs
    lons: np.ndarray              # Longitude of each node
    lats: np.ndarray              # Latitude of each node
    depths: np.ndarray            # Depth of each node (positive down)

    @property
    def n_nodes(self) -> int:
        return len(self.node_ids)


@dataclass
class SchismGrid:
    """
    Parsed SCHISM grid with boundary information.

    Usage:
        grid = SchismGrid.read("secofs.hgrid.ll")
        bnd_nodes = grid.open_boundary_nodes()  # all open boundary node coords
    """
    filepath: Path
    n_nodes: int
    n_elements: int
    node_lons: np.ndarray          # All node longitudes
    node_lats: np.ndarray          # All node latitudes
    node_depths: np.ndarray        # All node depths
    open_boundaries: List[OpenBoundary] = field(default_factory=list)
    n_open_boundary_nodes: int = 0

    @classmethod
    def read(cls, filepath, read_boundaries: bool = True) -> "SchismGrid":
        """
        Read SCHISM grid file (hgrid.gr3 or hgrid.ll).

        Args:
            filepath: Path to grid file
            read_boundaries: If True, parse open boundary sections

        Returns:
            SchismGrid instance
        """
        filepath = Path(filepath)
        log.info(f"Reading SCHISM grid: {filepath.name} ({filepath.stat().st_size / 1e6:.0f} MB)")

        with open(filepath) as f:
            # Line 1: comment
            comment = f.readline().strip()

            # Line 2: n_elements n_nodes
            parts = f.readline().split()
            n_elements = int(parts[0])
            n_nodes = int(parts[1])

            log.info(f"  Nodes: {n_nodes:,}, Elements: {n_elements:,}")

            # Read all node coordinates
            node_lons = np.zeros(n_nodes, dtype=np.float64)
            node_lats = np.zeros(n_nodes, dtype=np.float64)
            node_depths = np.zeros(n_nodes, dtype=np.float64)

            for i in range(n_nodes):
                parts = f.readline().split()
                # Format: node_id lon lat depth
                node_lons[i] = float(parts[1])
                node_lats[i] = float(parts[2])
                node_depths[i] = float(parts[3])

            log.info(f"  Lon range: [{node_lons.min():.2f}, {node_lons.max():.2f}]")
            log.info(f"  Lat range: [{node_lats.min():.2f}, {node_lats.max():.2f}]")
            log.info(f"  Depth range: [{node_depths.min():.2f}, {node_depths.max():.2f}]")

            # Skip element connectivity
            for _ in range(n_elements):
                f.readline()

            # Parse open boundaries
            open_boundaries = []
            n_open_boundary_nodes = 0

            if read_boundaries:
                try:
                    # Number of open boundaries
                    line = f.readline().strip()
                    n_open_bnd = int(line.split("=")[0].strip().split()[0])

                    # Total number of open boundary nodes
                    line = f.readline().strip()
                    n_open_boundary_nodes = int(line.split("=")[0].strip().split()[0])

                    log.info(f"  Open boundaries: {n_open_bnd}, total nodes: {n_open_boundary_nodes}")

                    for bnd_idx in range(n_open_bnd):
                        # Number of nodes in this boundary
                        line = f.readline().strip()
                        n_bnd_nodes = int(line.split("=")[0].strip().split()[0])

                        # Read node IDs
                        node_ids = []
                        for _ in range(n_bnd_nodes):
                            node_id = int(f.readline().strip())
                            node_ids.append(node_id)

                        # Look up coordinates (node_ids are 1-based)
                        indices = [nid - 1 for nid in node_ids]
                        bnd_lons = node_lons[indices]
                        bnd_lats = node_lats[indices]
                        bnd_depths = node_depths[indices]

                        bnd = OpenBoundary(
                            index=bnd_idx,
                            node_ids=node_ids,
                            lons=bnd_lons,
                            lats=bnd_lats,
                            depths=bnd_depths,
                        )
                        open_boundaries.append(bnd)
                        log.info(f"    Boundary {bnd_idx}: {n_bnd_nodes} nodes, "
                                 f"depth [{bnd_depths.min():.1f}, {bnd_depths.max():.1f}]m")

                except (ValueError, IndexError) as e:
                    log.warning(f"  Failed to parse boundaries: {e}")

        return cls(
            filepath=filepath,
            n_nodes=n_nodes,
            n_elements=n_elements,
            node_lons=node_lons,
            node_lats=node_lats,
            node_depths=node_depths,
            open_boundaries=open_boundaries,
            n_open_boundary_nodes=n_open_boundary_nodes,
        )

    def open_boundary_nodes(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
        """
        Get all open boundary node coordinates concatenated.

        Returns:
            (lons, lats, depths, node_ids) — all open boundary nodes across all segments
        """
        if not self.open_boundaries:
            return np.array([]), np.array([]), np.array([]), []

        all_lons = np.concatenate([b.lons for b in self.open_boundaries])
        all_lats = np.concatenate([b.lats for b in self.open_boundaries])
        all_depths = np.concatenate([b.depths for b in self.open_boundaries])
        all_ids = []
        for b in self.open_boundaries:
            all_ids.extend(b.node_ids)

        return all_lons, all_lats, all_depths, all_ids

    def obc_nodes_from_ctl(self, ctl_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
        """
        Read boundary node IDs from obc.ctl file (exact match with Fortran).

        The obc.ctl has the precise 1,488 nodes used by gen_3Dth_from_hycom,
        which may differ slightly from hgrid.ll boundary section.

        Args:
            ctl_path: Path to {ofs}.obc.ctl

        Returns:
            (lons, lats, depths, node_ids) — from obc.ctl SECTION 2
        """
        ctl_path = Path(ctl_path)
        node_ids = []

        with open(ctl_path) as f:
            lines = f.readlines()

        # Find SECTION 2
        sec2_start = None
        for i, line in enumerate(lines):
            if "SECTION 2" in line:
                sec2_start = i
                break

        if sec2_start is None:
            log.warning(f"SECTION 2 not found in {ctl_path}")
            return self.open_boundary_nodes()

        # Parse NODE_ID from column 1 (0-indexed), skip header line
        for i in range(sec2_start + 2, len(lines)):
            parts = lines[i].split()
            if len(parts) >= 2:
                try:
                    node_id = int(parts[1])
                    node_ids.append(node_id)
                except ValueError:
                    break

        # Look up coordinates (node_ids are 1-based)
        indices = [nid - 1 for nid in node_ids]
        lons = self.node_lons[indices]
        lats = self.node_lats[indices]
        depths = self.node_depths[indices]

        log.info(f"Read {len(node_ids)} boundary nodes from {ctl_path.name}")
        return lons, lats, depths, node_ids

    def __repr__(self):
        return (f"SchismGrid(nodes={self.n_nodes:,}, elements={self.n_elements:,}, "
                f"open_bnd={len(self.open_boundaries)}, "
                f"open_bnd_nodes={self.n_open_boundary_nodes})")
