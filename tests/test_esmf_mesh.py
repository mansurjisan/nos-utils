"""Tests for ESMFMeshProcessor."""

from pathlib import Path
import numpy as np
import pytest
from nos_utils.forcing.esmf_mesh import ESMFMeshProcessor

netCDF4 = pytest.importorskip("netCDF4")


class TestESMFMeshProcessor:
    def test_basic_generation(self, mock_config, tmp_path):
        out_dir = tmp_path / "mesh_out"
        proc = ESMFMeshProcessor(mock_config, tmp_path, out_dir)
        result = proc.process()

        assert result.success
        mesh_file = out_dir / "datm_esmf_mesh.nc"
        assert mesh_file.exists()

    def test_element_mask_is_one(self, mock_config, tmp_path):
        """CRITICAL: elementMask must be 1 (active), NOT 0 (lesson #18)."""
        out_dir = tmp_path / "mesh_out"
        proc = ESMFMeshProcessor(mock_config, tmp_path, out_dir)
        proc.process()

        ds = netCDF4.Dataset(str(out_dir / "datm_esmf_mesh.nc"))
        mask = ds.variables["elementMask"][:]
        assert np.all(mask == 1), f"elementMask has zeros! This will mask ALL elements."
        ds.close()

    def test_mesh_dimensions(self, mock_config, tmp_path):
        out_dir = tmp_path / "mesh_out"
        proc = ESMFMeshProcessor(mock_config, tmp_path, out_dir)
        result = proc.process()

        ds = netCDF4.Dataset(str(out_dir / "datm_esmf_mesh.nc"))

        n_nodes = ds.dimensions["nodeCount"].size
        n_elements = ds.dimensions["elementCount"].size

        # For a 41x41 forcing grid: elements=41*41 (one per data value),
        # nodes=42*42 (the surrounding corners).
        assert n_nodes > 0
        assert n_elements > 0
        assert n_elements == result.metadata["n_elements"]

        # Connectivity should be quads (4 nodes per element)
        conn = ds.variables["elementConn"][:]
        assert conn.shape[1] == 4
        # 1-based indexing
        assert ds.variables["elementConn"].start_index == 1

        ds.close()

    def test_from_forcing_file(self, mock_config, tmp_path):
        """Should read grid from datm_forcing.nc if available."""
        # Create a mock forcing file
        forcing_file = tmp_path / "datm_forcing.nc"
        ds = netCDF4.Dataset(str(forcing_file), "w")
        ds.createDimension("longitude", 5)
        ds.createDimension("latitude", 4)
        lon_var = ds.createVariable("longitude", "f4", ("longitude",))
        lat_var = ds.createVariable("latitude", "f4", ("latitude",))
        lon_var[:] = np.linspace(-80, -70, 5)
        lat_var[:] = np.linspace(25, 35, 4)
        ds.close()

        out_dir = tmp_path / "mesh_out"
        proc = ESMFMeshProcessor(
            mock_config, tmp_path, out_dir, forcing_file=forcing_file,
        )
        result = proc.process()

        assert result.success
        assert result.metadata["nx"] == 5
        assert result.metadata["ny"] == 4
        # One element per forcing value: CDEPS reads stream fields at
        # ESMF_MESHLOC_ELEMENT, so elementCount must equal nx*ny. This
        # asserted (5-1)*(4-1) until 2026-07-28, which is what let the
        # cell-cornered mesh ship.
        assert result.metadata["n_elements"] == 5 * 4
        assert result.metadata["n_nodes"] == 6 * 5


class TestCDEPSElementMapping:
    """The mesh must survive CDEPS's flat element read without shearing.

    CDEPS creates every stream field at ESMF_MESHLOC_ELEMENT and does not
    validate elementCount against the forcing file's dimensions. With a
    cell-cornered mesh, (nx-1)*(ny-1) < nx*ny, so PIO silently read the
    first (nx-1)*(ny-1) values in flat order: element (r,c) received file
    value r*(nx-1)+c instead of r*nx+c, slipping one cell west per row.

    On the real 1721x1721 SECOFS grid that displaced forcing by a median
    1431 km -- the Chesapeake Bay mouth was driven by open-Atlantic wind
    from 1427 km east. Reproduced from station output at r = 1.000 across
    all 430 stations before the fix.
    """

    def _mesh(self, tmp_path, lons, lats):
        from nos_utils.forcing.esmf_mesh import ESMFMeshProcessor
        out = tmp_path / "m.nc"
        ESMFMeshProcessor._create_mesh(lons, lats, out)
        return netCDF4.Dataset(str(out))

    def test_element_k_is_the_data_point_at_flat_index_k(self, tmp_path):
        """The invariant that makes the flat read correct."""
        nx, ny = 7, 5
        lons = -98.0 + 0.025 * np.arange(nx)
        lats = 10.0 + 0.025 * np.arange(ny)
        ds = self._mesh(tmp_path, lons, lats)

        assert ds.dimensions["elementCount"].size == nx * ny
        assert ds.dimensions["nodeCount"].size == (nx + 1) * (ny + 1)

        centers = np.asarray(ds.variables["centerCoords"][:])
        k = np.arange(nx * ny)
        expected = np.column_stack([lons[k % nx], lats[k // nx]])
        assert np.allclose(centers, expected), (
            "element k must carry the forcing value at flat index k; any "
            "offset here IS the shear"
        )
        ds.close()

    def test_nodes_are_half_cell_corners_not_the_data_points(self, tmp_path):
        """Centres on the data points, corners staggered around them."""
        nx, ny = 7, 5
        dx = 0.025
        lons = -98.0 + dx * np.arange(nx)
        lats = 10.0 + dx * np.arange(ny)
        ds = self._mesh(tmp_path, lons, lats)
        nodes = np.asarray(ds.variables["nodeCoords"][:])
        assert np.allclose(nodes[0], [lons[0] - dx / 2, lats[0] - dx / 2])
        assert np.allclose(nodes[-1], [lons[-1] + dx / 2, lats[-1] + dx / 2])
        ds.close()

    def test_connectivity_indexes_the_corner_grid(self, tmp_path):
        nx, ny = 7, 5
        ds = self._mesh(tmp_path, -98.0 + 0.025 * np.arange(nx),
                        10.0 + 0.025 * np.arange(ny))
        conn = np.asarray(ds.variables["elementConn"][:])
        n_nodes = ds.dimensions["nodeCount"].size
        assert conn.shape == (nx * ny, 4)
        assert conn.min() >= 1 and conn.max() <= n_nodes
        # First element: SW, SE, NE, NW on a (nx+1)-wide node grid.
        assert conn[0].tolist() == [1, 2, nx + 3, nx + 2]
        assert np.all(np.asarray(ds.variables["elementMask"][:]) == 1)
        ds.close()

    def test_non_uniform_spacing_uses_midpoint_edges(self, tmp_path):
        lons = np.array([-80.0, -79.5, -78.0, -77.9])
        lats = np.array([25.0, 26.0, 28.5])
        ds = self._mesh(tmp_path, lons, lats)
        assert ds.dimensions["elementCount"].size == lons.size * lats.size
        centers = np.asarray(ds.variables["centerCoords"][:])
        assert np.allclose(centers[: lons.size, 0], lons)
        nodes = np.asarray(ds.variables["nodeCoords"][:])
        # Interior edge sits at the midpoint of neighbouring centres.
        assert np.isclose(nodes[1, 0], (lons[0] + lons[1]) / 2)
        ds.close()
