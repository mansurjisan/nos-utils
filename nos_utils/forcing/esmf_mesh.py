"""
ESMF mesh file generator for UFS-Coastal DATM coupling.

Creates an ESMF unstructured mesh file from a regular lat/lon forcing grid.
Used by CDEPS DATM to interpolate atmospheric forcing to the ocean mesh.

CRITICAL: elementMask must be set to 1 (active), NOT 0 (masked).
  elementMask=0 causes CMEPS to mask out ALL elements, resulting in
  zero atmospheric forcing passed to SCHISM (lesson #18).

Input: Regular lat/lon grid (from datm_forcing.nc or sflux files)
Output: datm_esmf_mesh.nc (ESMF unstructured mesh format — name matches
        the legacy COMF default DATM_MESH_FILE in nos_ofs_gen_ufs_config.sh
        and the @[DATM_MESH_FILE] reference in datm_in.template).
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
            output_path: Output directory for datm_esmf_mesh.nc
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

        output_file = self.output_path / "datm_esmf_mesh.nc"
        self._create_mesh(lons, lats, output_file)

        nx, ny = len(lons), len(lats)
        log.info(f"Created datm_esmf_mesh.nc: nx={nx}, ny={ny}, "
                 f"elements={nx*ny}, nodes={(nx+1)*(ny+1)}")

        return ForcingResult(
            success=True, source=self.SOURCE_NAME,
            output_files=[output_file],
            metadata={
                "nx": nx, "ny": ny,
                # One element per forcing value (CDEPS reads at
                # ESMF_MESHLOC_ELEMENT); nodes are the surrounding corners.
                "n_elements": nx * ny,
                "n_nodes": (nx + 1) * (ny + 1),
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

    @staticmethod
    def _cell_edges(centers: np.ndarray) -> np.ndarray:
        """Cell boundaries for ``centers``: midpoints, extrapolated at the ends.

        Works for non-uniform spacing, so a stretched or subset grid is
        handled the same way as the regular 0.025 deg DATM grid.
        """
        centers = np.asarray(centers, dtype=float)
        if centers.size < 2:
            raise ValueError("need at least 2 grid points to build cell edges")
        edges = np.empty(centers.size + 1, dtype=float)
        edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
        edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
        edges[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])
        return edges

    @staticmethod
    def _create_mesh(lons: np.ndarray, lats: np.ndarray, output_path: Path) -> None:
        """Write an ESMF unstructured mesh whose ELEMENTS are the data points.

        CDEPS reads every stream field at ``ESMF_MESHLOC_ELEMENT``
        (``dshr_strdata_mod.F90`` creates all stream fields that way; there
        is no node-based read path), so a forcing file with ``nx*ny`` values
        needs a mesh with exactly ``nx*ny`` elements, centred ON the data
        points.

        This previously treated the forcing coordinates as cell CORNERS --
        ``n_elements = (nx-1)*(ny-1)`` with centres half a cell to the
        north-east of each data point. CDEPS does not check element count
        against the file's dimensions, and because (nx-1)*(ny-1) < nx*ny,
        PIO silently read the first (nx-1)*(ny-1) values in flat order. The
        data is row-major with nx per row while elements run nx-1 per row,
        so the mapping slipped one cell west per row and sheared with
        latitude: on the 1721x1721 SECOFS grid, stations were forced with
        wind from a median 1431 km away (Chesapeake Bay mouth was reading
        the open Atlantic, 1427 km east). Confirmed by reproducing station
        output exactly (r = 1.000 at 430/430 stations) from the predicted
        sheared indices.

        Layout, with k = j*nx + i so element order matches the forcing
        file's flattened ``(y, x)``:

          - elementCount = nx*ny, centerCoords[k] = (lons[i], lats[j])
          - nodeCount = (nx+1)*(ny+1) corner nodes on the staggered grid
          - elementConn = the 4 surrounding corners, counter-clockwise
          - elementMask = 1 everywhere; CMEPS uses srcMaskValues=(/0/), so
            0 would mask the cell OUT
        """
        nx = len(lons)
        ny = len(lats)
        n_elements = nx * ny
        lon_edges = ESMFMeshProcessor._cell_edges(lons)
        lat_edges = ESMFMeshProcessor._cell_edges(lats)
        nxe, nye = nx + 1, ny + 1
        n_nodes = nxe * nye

        nc = Dataset(str(output_path), "w", format="NETCDF4")
        nc.createDimension("nodeCount", n_nodes)
        nc.createDimension("elementCount", n_elements)
        nc.createDimension("maxNodePElement", 4)
        nc.createDimension("coordDim", 2)

        # Corner nodes, x-fastest.
        node_coords = nc.createVariable("nodeCoords", "f8", ("nodeCount", "coordDim"))
        node_coords.units = "degrees"
        # tile/repeat rather than meshgrid+ravel: it says x-fastest in the
        # call itself, and ordering is exactly what was wrong here. (Peak
        # memory is the same either way -- measured.)
        node_coords[:] = np.column_stack(
            [np.tile(lon_edges, nye), np.repeat(lat_edges, nxe)]
        )

        # Connectivity into the (ny+1) x (nx+1) node grid, 1-based CCW.
        elem_conn = nc.createVariable(
            "elementConn", "i4", ("elementCount", "maxNodePElement")
        )
        elem_conn.long_name = "Node indices that define the element connectivity"
        elem_conn.start_index = 1
        jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
        sw = (jj * nxe + ii + 1).ravel()
        conn = np.empty((n_elements, 4), dtype=np.int32)
        conn[:, 0] = sw                 # SW
        conn[:, 1] = sw + 1             # SE
        conn[:, 2] = sw + nxe + 1       # NE
        conn[:, 3] = sw + nxe           # NW
        elem_conn[:] = conn

        num_conn = nc.createVariable("numElementConn", "i4", ("elementCount",))
        num_conn[:] = 4

        elem_mask = nc.createVariable("elementMask", "i4", ("elementCount",))
        elem_mask[:] = np.ones(n_elements, dtype=np.int32)

        # Element centres ARE the data points -- this is the whole fix.
        center_coords = nc.createVariable(
            "centerCoords", "f8", ("elementCount", "coordDim")
        )
        center_coords.units = "degrees"
        center_coords[:] = np.column_stack(
            [np.tile(lons, ny), np.repeat(lats, nx)]
        )

        nc.gridType = "unstructured"
        nc.title = "ESMF mesh for DATM atmospheric forcing"
        nc.close()
