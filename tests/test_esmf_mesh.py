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
        mesh_file = out_dir / "esmf_mesh.nc"
        assert mesh_file.exists()

    def test_element_mask_is_one(self, mock_config, tmp_path):
        """CRITICAL: elementMask must be 1 (active), NOT 0 (lesson #18)."""
        out_dir = tmp_path / "mesh_out"
        proc = ESMFMeshProcessor(mock_config, tmp_path, out_dir)
        proc.process()

        ds = netCDF4.Dataset(str(out_dir / "esmf_mesh.nc"))
        mask = ds.variables["elementMask"][:]
        assert np.all(mask == 1), f"elementMask has zeros! This will mask ALL elements."
        ds.close()

    def test_mesh_dimensions(self, mock_config, tmp_path):
        out_dir = tmp_path / "mesh_out"
        proc = ESMFMeshProcessor(mock_config, tmp_path, out_dir)
        result = proc.process()

        ds = netCDF4.Dataset(str(out_dir / "esmf_mesh.nc"))

        n_nodes = ds.dimensions["nodeCount"].size
        n_elements = ds.dimensions["elementCount"].size

        # For a 41x41 grid (10° lon × 10° lat at 0.25°): nodes=41*41=1681, elements=40*40=1600
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
        assert result.metadata["n_elements"] == 4 * 3  # (5-1)*(4-1)
