"""
ESMF mesh file generator for UFS-Coastal DATM coupling.

Creates an ESMF unstructured mesh file from a regular lat/lon forcing grid.
Used by CDEPS DATM to interpolate atmospheric forcing to the ocean mesh.

CRITICAL: elementMask must be set to 1 (active), NOT 0 (masked).
  elementMask=0 causes CMEPS to mask out ALL elements, resulting in
  zero atmospheric forcing passed to SCHISM (lesson #18).

Input: Regular lat/lon grid (from datm_forcing.nc or sflux files)
Output: esmf_mesh.nc (ESMF unstructured mesh format)
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from ..config import ForcingConfig
from .base import ForcingProcessor, ForcingResult

log = logging.getLogger(__name__)

try:
    from netCDF4 import Dataset
    HAS_NETCDF4 = True
except ImportError:
    HAS_NETCDF4 = False


class ESMFMeshProcessor(ForcingProcessor):
    """
    Generate ESMF mesh file from a regular lat/lon forcing grid.

    The mesh file defines the spatial grid for CDEPS DATM component
    in UFS-Coastal coupled simulations.
    """

    SOURCE_NAME = "ESMF_MESH"

    def __init__(
        self,
        config: ForcingConfig,
        input_path: Path,
        output_path: Path,
        forcing_file: Optional[Path] = None,
    ):
        """
        Args:
            config: ForcingConfig with domain bounds
            input_path: Directory containing datm_forcing.nc
            output_path: Output directory for esmf_mesh.nc
            forcing_file: Explicit path to forcing file to read grid from
        """
        super().__init__(config, input_path, output_path)
        self.forcing_file = forcing_file

    def process(self) -> ForcingResult:
        """Generate ESMF mesh from forcing grid or config domain."""
        if not HAS_NETCDF4:
            return ForcingResult(
                success=False, source=self.SOURCE_NAME,
                errors=["netCDF4 required for ESMF mesh generation"],
            )

        log.info("ESMF mesh processor")
        self.create_output_dir()

        # Get grid from forcing file or config
        lons, lats = self._get_grid()
        if lons is None:
            return ForcingResult(
                success=False, source=self.SOURCE_NAME,
                errors=["Cannot determine grid for ESMF mesh"],
            )

        output_file = self.output_path / "esmf_mesh.nc"
        self._create_mesh(lons, lats, output_file)

        nx, ny = len(lons), len(lats)
        log.info(f"Created esmf_mesh.nc: nx={nx}, ny={ny}, "
                 f"elements={(nx-1)*(ny-1)}, nodes={nx*ny}")

        return ForcingResult(
            success=True, source=self.SOURCE_NAME,
            output_files=[output_file],
            metadata={
                "nx": nx, "ny": ny,
                "n_elements": (nx - 1) * (ny - 1),
                "n_nodes": nx * ny,
            },
        )

    def find_input_files(self) -> List[Path]:
        if self.forcing_file and Path(self.forcing_file).exists():
            return [Path(self.forcing_file)]
        found = sorted(self.input_path.glob("datm_forcing*.nc"))
        return found

    def _get_grid(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get 1D lon/lat arrays from forcing file or config."""
        # Try reading from forcing file
        src = self.forcing_file
        if src is None:
            candidates = sorted(self.input_path.glob("datm_forcing*.nc"))
            if candidates:
                src = candidates[0]

        if src and Path(src).exists():
            try:
                ds = Dataset(str(src))
                lons = ds.variables.get("longitude", ds.variables.get("lon"))[:]
                lats = ds.variables.get("latitude", ds.variables.get("lat"))[:]
                ds.close()
                # Ensure 1D
                if lons.ndim > 1:
                    lons = lons[0, :]
                if lats.ndim > 1:
                    lats = lats[:, 0]
                return np.asarray(lons), np.asarray(lats)
            except Exception as e:
                log.warning(f"Cannot read grid from {src}: {e}")

        # Fallback: construct from config domain
        lon_min, lon_max, lat_min, lat_max = self.config.domain
        dx = 0.25  # default GFS resolution
        lons = np.arange(lon_min, lon_max + dx, dx)
        lats = np.arange(lat_min, lat_max + dx, dx)
        return lons, lats

    def _create_mesh(self, lons: np.ndarray, lats: np.ndarray, output_path: Path) -> None:
        """
        Create ESMF unstructured mesh file from regular lat/lon grid.

        ESMF mesh format:
          - nodeCoords: [n_nodes, 2] (lon, lat pairs)
          - elementConn: [n_elements, 4] (quad connectivity, 1-based)
          - elementMask: [n_elements] — MUST be 1 (active), NOT 0 (lesson #18)
          - numElementConn: [n_elements] — always 4 for quads
          - centerCoords: [n_elements, 2] (element center lon/lat)
        """
        nx = len(lons)
        ny = len(lats)
        n_nodes = nx * ny
        n_elements = (nx - 1) * (ny - 1)

        nc = Dataset(str(output_path), "w", format="NETCDF4")

        # Dimensions
        nc.createDimension("nodeCount", n_nodes)
        nc.createDimension("elementCount", n_elements)
        nc.createDimension("maxNodePElement", 4)  # quadrilateral
        nc.createDimension("coordDim", 2)

        # Node coordinates (lon, lat pairs)
        node_coords = nc.createVariable("nodeCoords", "f8", ("nodeCount", "coordDim"))
        node_coords.units = "degrees"

        lon_2d, lat_2d = np.meshgrid(lons, lats)
        coords = np.column_stack([lon_2d.ravel(), lat_2d.ravel()])
        node_coords[:] = coords

        # Element connectivity (1-based, counter-clockwise quad vertices)
        elem_conn = nc.createVariable("elementConn", "i4", ("elementCount", "maxNodePElement"))
        elem_conn.long_name = "Node indices that define the element connectivity"
        elem_conn.start_index = 1  # 1-based indexing

        connectivity = np.zeros((n_elements, 4), dtype=np.int32)
        idx = 0
        for j in range(ny - 1):
            for i in range(nx - 1):
                # Counter-clockwise: SW, SE, NE, NW (1-based)
                n0 = j * nx + i + 1
                n1 = j * nx + (i + 1) + 1
                n2 = (j + 1) * nx + (i + 1) + 1
                n3 = (j + 1) * nx + i + 1
                connectivity[idx] = [n0, n1, n2, n3]
                idx += 1
        elem_conn[:] = connectivity

        # Number of nodes per element (always 4 for quads)
        num_conn = nc.createVariable("numElementConn", "i4", ("elementCount",))
        num_conn[:] = 4

        # Element mask — CRITICAL: must be 1 (active), NOT 0 (lesson #18)
        # CMEPS uses srcMaskValues=(/0/), so elementMask=0 means MASKED OUT
        elem_mask = nc.createVariable("elementMask", "i4", ("elementCount",))
        elem_mask[:] = np.ones(n_elements, dtype=np.int32)  # ALL ACTIVE

        # Element center coordinates
        center_coords = nc.createVariable("centerCoords", "f8", ("elementCount", "coordDim"))
        center_coords.units = "degrees"

        centers = np.zeros((n_elements, 2))
        idx = 0
        for j in range(ny - 1):
            for i in range(nx - 1):
                centers[idx, 0] = (lons[i] + lons[i + 1]) / 2.0
                centers[idx, 1] = (lats[j] + lats[j + 1]) / 2.0
                idx += 1
        center_coords[:] = centers

        # Global attributes
        nc.gridType = "unstructured"
        nc.title = "ESMF mesh for DATM atmospheric forcing"
        nc.close()
