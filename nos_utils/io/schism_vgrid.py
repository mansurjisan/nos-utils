"""
SCHISM vertical grid reader (vgrid.in).

Parses the simple vgrid.in format (from FIXofs work directory) to extract
Z-levels and S-levels (sigma coordinates) for vertical interpolation.

Format:
  Line 1: nvrt kz h_s    (total levels, Z-level count, S-Z transition depth)
  Line 2: "Z levels"
  Lines 3 to kz+2: level_index depth_m
  Line kz+3: "S levels" header
  Remaining: level_index sigma_value
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class SchismVgrid:
    """Parsed SCHISM vertical grid."""
    nvrt: int               # Total number of vertical levels
    kz: int                 # Number of Z-levels
    h_s: float              # Depth of S-Z transition (meters)
    z_levels: np.ndarray    # Z-level depths (meters, negative down)
    sigma_levels: np.ndarray  # Sigma values (-1 = bottom, 0 = surface)
    node_sigma: Optional[np.ndarray] = None  # (nvrt, n_boundary_nodes) per-node sigma
    node_kbp: Optional[np.ndarray] = None    # Bottom level index per boundary node
    _filepath: Optional[str] = None          # For lazy loading

    def load_boundary_sigma(self, boundary_node_ids: List[int]) -> None:
        """
        Load per-node sigma values for boundary nodes from LSC2 vgrid.in.

        This reads the full file but only extracts columns for boundary nodes.
        Takes ~60-80 seconds for 1.68M nodes × 63 levels.

        Args:
            boundary_node_ids: List of 1-based node IDs
        """
        if self._filepath is None or self.node_sigma is not None:
            return

        filepath = Path(self._filepath)
        if not filepath.exists():
            return

        bnd_indices = [nid - 1 for nid in boundary_node_ids]  # 0-based
        n_bnd = len(bnd_indices)

        log.info(f"Loading per-node sigma for {n_bnd} boundary nodes from {filepath.name}...")

        with open(filepath) as f:
            f.readline()  # ivcor
            f.readline()  # nvrt

            # kbp line
            kbp_line = f.readline()
            kbp_all = kbp_line.split()
            self.node_kbp = np.array([int(kbp_all[i]) for i in bnd_indices])

            # Read level lines, extract boundary columns
            self.node_sigma = np.full((self.nvrt, n_bnd), -9.0, dtype=np.float64)
            for lev in range(self.nvrt):
                line = f.readline()
                parts = line.split()
                level_idx = int(parts[0]) - 1
                for k, ni in enumerate(bnd_indices):
                    self.node_sigma[level_idx, k] = float(parts[ni + 1])

        log.info(f"Loaded boundary sigma: kbp range [{self.node_kbp.min()}, {self.node_kbp.max()}]")

    def get_node_depths(self, node_idx: int, bottom_depth: float) -> np.ndarray:
        """
        Get actual depths for a specific node using per-node sigma values.

        For LSC2 format where each node has its own sigma distribution.
        Returns depths in meters (negative = below surface).

        Args:
            node_idx: Index into the boundary node sigma array (0-based)
            bottom_depth: Positive bottom depth in meters

        Returns:
            Array of depth values for valid levels (sigma > -8)
        """
        if self.node_sigma is not None and node_idx < self.node_sigma.shape[1]:
            sigma_col = self.node_sigma[:, node_idx]
            valid = sigma_col > -8.0  # -9 means below bottom / invalid
            valid_sigma = sigma_col[valid]
            # sigma: -1 = bottom, 0 = surface
            # depth = bottom_depth * sigma
            return bottom_depth * valid_sigma
        else:
            return self.get_depths(bottom_depth)

    def get_depths(self, bottom_depth: float) -> np.ndarray:
        """
        Compute actual depths for a node with given bottom depth.

        For deep nodes (depth > h_s): Z-levels + S-levels mapped to [h_s, 0]
        For shallow nodes (depth <= h_s): only S-levels mapped to [depth, 0]

        Args:
            bottom_depth: Positive bottom depth in meters

        Returns:
            Array of depth values (negative, from deep to surface)
        """
        depths = []

        if bottom_depth > self.h_s:
            # Deep node: use Z-levels that are deeper than h_s
            for z in self.z_levels:
                if z <= -self.h_s:
                    depths.append(z)

            # Then S-levels mapped from -h_s to 0
            for s in self.sigma_levels:
                if s <= 0:
                    depths.append(self.h_s * s)
        else:
            # Shallow node: S-levels only, mapped from -bottom_depth to 0
            for s in self.sigma_levels:
                if s <= 0:
                    depths.append(bottom_depth * s)

        return np.array(depths)

    @classmethod
    def read(cls, filepath) -> "SchismVgrid":
        """
        Read vgrid.in file. Supports two formats:

        Simple format (68 lines):
          Line 0: nvrt kz h_s
          Lines 2-kz+1: Z-levels
          Remaining: S-levels

        LSC2 format (1.6GB, per-node):
          Line 0: ivcor (=1)
          Line 1: nvrt
          Line 2: per-node kbp values
          ... (per-node sigma levels)
        """
        filepath = Path(filepath)

        with open(filepath) as f:
            line0 = f.readline()
            line1 = f.readline()

        parts0 = line0.split()

        # Detect format: simple has 3+ values on line 0 (nvrt kz h_s)
        # LSC2 has just 1 value on line 0 (ivcor)
        if len(parts0) >= 3:
            # Simple format
            return cls._read_simple(filepath)
        else:
            # LSC2 format — extract nvrt, store filepath for lazy boundary sigma loading
            nvrt = int(line1.strip().split()[0])
            log.info(f"Read LSC2 vgrid.in: nvrt={nvrt} (per-node sigma available via load_boundary_sigma)")
            return cls(
                nvrt=nvrt, kz=0, h_s=100.0,
                z_levels=np.array([]),
                sigma_levels=np.linspace(-1, 0, nvrt),
                _filepath=str(filepath),
            )

    @classmethod
    def _read_simple(cls, filepath) -> "SchismVgrid":
        """Read simple vgrid.in format (68 lines)."""
        filepath = Path(filepath)

        with open(filepath) as f:
            lines = f.readlines()

        parts = lines[0].split()
        nvrt = int(parts[0])
        kz = int(parts[1])
        h_s = float(parts[2])

        # Z-levels (lines 2 to kz+1)
        z_levels = []
        for i in range(2, 2 + kz):
            parts = lines[i].split()
            z_levels.append(float(parts[1]))

        # Find S-levels section
        s_start = 2 + kz
        while s_start < len(lines) and "S" in lines[s_start]:
            s_start += 1  # skip "S levels" header

        sigma_levels = []
        for i in range(s_start, len(lines)):
            parts = lines[i].split()
            if len(parts) >= 2:
                try:
                    sigma_levels.append(float(parts[1]))
                except ValueError:
                    break

        log.info(f"Read vgrid.in: nvrt={nvrt}, kz={kz} Z-levels, "
                 f"{len(sigma_levels)} S-levels, h_s={h_s}m")

        return cls(
            nvrt=nvrt, kz=kz, h_s=h_s,
            z_levels=np.array(z_levels),
            sigma_levels=np.array(sigma_levels),
        )
