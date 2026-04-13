"""Tests for NudgingProcessor."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from nos_utils.config import ForcingConfig
from nos_utils.forcing.nudging import NudgingProcessor

# netCDF4 is required for nudging output — skip gracefully if unavailable
netCDF4 = pytest.importorskip("netCDF4")


class TestNudgingProcessor:
    def test_disabled_returns_success(self, mock_config, tmp_path):
        mock_config.nudging_enabled = False
        proc = NudgingProcessor(mock_config, tmp_path, tmp_path / "out")
        result = proc.process()
        assert result.success
        assert "disabled" in result.warnings[0].lower()

    def test_no_input_returns_failure(self, mock_config, tmp_path):
        mock_config.nudging_enabled = True
        proc = NudgingProcessor(mock_config, tmp_path / "empty", tmp_path / "out")
        result = proc.process()
        assert not result.success

    def test_source_name(self):
        assert NudgingProcessor.SOURCE_NAME == "NUDGING"

    def test_stofs_mode_detection(self, stofs_config, tmp_path):
        """STOFS config has nudge_roi_3d, so is_stofs_mode is True."""
        proc = NudgingProcessor(stofs_config, tmp_path, tmp_path / "out")
        assert proc.is_stofs_mode is True

    def test_secofs_mode_detection(self, secofs_config, tmp_path):
        """SECOFS config has no nudge_roi_3d, so is_stofs_mode is False."""
        proc = NudgingProcessor(secofs_config, tmp_path, tmp_path / "out")
        assert proc.is_stofs_mode is False


class TestWriteNudgeNC:
    """Test the _write_nudge_nc output format."""

    def test_output_format_matches_comf(self, secofs_config, tmp_path):
        """Verify output dimensions, variables, and dtypes match COMF Fortran."""
        secofs_config.nudging_enabled = True
        proc = NudgingProcessor(secofs_config, tmp_path, tmp_path / "out")
        proc.create_output_dir()

        n_time, n_nodes, n_levels = 5, 100, 63
        data = np.random.rand(n_time, n_nodes, n_levels).astype(np.float32)
        node_ids = np.arange(1000, 1000 + n_nodes, dtype=np.int32)

        out_path = tmp_path / "out" / "TEM_nu.nc"
        proc._write_nudge_nc(out_path, data, node_ids, 10800.0, "TEM", "degC")

        ds = netCDF4.Dataset(str(out_path))

        # Check dimensions
        assert ds.dimensions["time"].size == n_time
        assert ds.dimensions["node"].size == n_nodes
        assert ds.dimensions["nLevels"].size == n_levels
        assert ds.dimensions["one"].size == 1

        # Check time variable
        time_var = ds.variables["time"]
        assert time_var.dtype == np.float64
        np.testing.assert_array_almost_equal(
            time_var[:], [0, 10800, 21600, 32400, 43200]
        )

        # Check map_to_global_node
        map_var = ds.variables["map_to_global_node"]
        assert map_var.dtype == np.int32
        np.testing.assert_array_equal(map_var[:], node_ids)

        # Check tracer_concentration
        tracer = ds.variables["tracer_concentration"]
        assert tracer.shape == (n_time, n_nodes, n_levels, 1)
        assert tracer.dtype == np.float64  # COMF uses float64

        ds.close()

    def test_nan_fill(self, secofs_config, tmp_path):
        """NaN values should be filled with defaults (15 for T, 35 for S)."""
        secofs_config.nudging_enabled = True
        proc = NudgingProcessor(secofs_config, tmp_path, tmp_path / "out")
        proc.create_output_dir()

        data = np.full((3, 10, 5), np.nan, dtype=np.float32)
        node_ids = np.arange(1, 11, dtype=np.int32)

        # TEM defaults to 15.0
        tem_path = tmp_path / "out" / "TEM_test.nc"
        proc._write_nudge_nc(tem_path, data, node_ids, 10800.0, "TEM", "degC")

        ds = netCDF4.Dataset(str(tem_path))
        np.testing.assert_array_almost_equal(
            ds.variables["tracer_concentration"][:], 15.0
        )
        ds.close()

        # SAL defaults to 35.0
        sal_path = tmp_path / "out" / "SAL_test.nc"
        proc._write_nudge_nc(sal_path, data.copy(), node_ids, 10800.0, "SAL", "PSU")

        ds = netCDF4.Dataset(str(sal_path))
        np.testing.assert_array_almost_equal(
            ds.variables["tracer_concentration"][:], 35.0
        )
        ds.close()


class TestLoadNudgeNodes:
    """Test nudge node identification from gr3 files."""

    def _make_gr3(self, path: Path, n_nodes: int, weights: np.ndarray):
        """Create a minimal gr3 file for testing."""
        with open(path, "w") as f:
            f.write("test nudge weight file\n")
            f.write(f"  0  {n_nodes}\n")  # 0 elements, n_nodes nodes
            for i in range(n_nodes):
                # node_id lon lat value
                lon = -80.0 + i * 0.1
                lat = 30.0 + i * 0.1
                f.write(f"  {i+1}  {lon:.5f}  {lat:.5f}  {weights[i]:.6e}\n")

    def test_reads_nonzero_nodes(self, mock_config, tmp_path):
        """Only nodes with weight > 0 should be returned."""
        mock_config.nudging_enabled = True

        n_nodes = 20
        weights = np.zeros(n_nodes)
        weights[5] = 0.001
        weights[10] = 0.5
        weights[15] = 1.0

        gr3_path = tmp_path / "test.nudge.gr3"
        self._make_gr3(gr3_path, n_nodes, weights)

        proc = NudgingProcessor(
            mock_config, tmp_path, tmp_path / "out",
            nudge_weight_file=gr3_path,
        )
        ids, lons, lats, depths = proc._load_nudge_nodes()

        assert ids is not None
        assert len(ids) == 3
        np.testing.assert_array_equal(ids, [6, 11, 16])  # 1-based

    def test_no_file_returns_none(self, mock_config, tmp_path):
        """Should return None tuple if no nudge file found."""
        mock_config.nudging_enabled = True

        proc = NudgingProcessor(
            mock_config, tmp_path, tmp_path / "out",
            nudge_weight_file=tmp_path / "nonexistent.gr3",
        )
        ids, lons, lats, depths = proc._load_nudge_nodes()
        assert ids is None

    def test_depths_from_grid(self, mock_config, tmp_path):
        """When grid_file is available, depths come from the grid."""
        mock_config.nudging_enabled = True

        # Create a tiny gr3 with 5 nodes, 2 with weight > 0
        n_nodes = 5
        weights = np.array([0.0, 0.5, 0.0, 1.0, 0.0])
        gr3_path = tmp_path / "test.nudge.gr3"
        self._make_gr3(gr3_path, n_nodes, weights)

        # Create a matching grid file
        grid_path = tmp_path / "hgrid.ll"
        with open(grid_path, "w") as f:
            f.write("test grid\n")
            f.write("  0  5\n")
            for i in range(5):
                f.write(f"  {i+1}  {-80.0 + i*0.1:.5f}  {30.0 + i*0.1:.5f}  {(i+1)*10.0:.1f}\n")
            f.write("0 = number of open boundaries\n")
            f.write("0 = total open boundary nodes\n")

        mock_config.grid_file = grid_path
        proc = NudgingProcessor(
            mock_config, tmp_path, tmp_path / "out",
            nudge_weight_file=gr3_path,
        )
        ids, lons, lats, depths = proc._load_nudge_nodes()

        assert len(ids) == 2
        # Nodes 2 and 4 have weight > 0; depths = 20.0, 40.0
        np.testing.assert_array_almost_equal(depths, [20.0, 40.0])


class TestGr3Reader:
    """Test SchismGrid.read_gr3_values."""

    def test_basic_read(self, tmp_path):
        from nos_utils.io.schism_grid import SchismGrid

        gr3_path = tmp_path / "test.gr3"
        with open(gr3_path, "w") as f:
            f.write("header line\n")
            f.write("  0  3\n")
            f.write("  1  -80.0  30.0  0.0\n")
            f.write("  2  -79.0  31.0  1.5\n")
            f.write("  3  -78.0  32.0  0.0\n")

        ids, lons, lats, vals = SchismGrid.read_gr3_values(gr3_path)

        assert len(ids) == 3
        np.testing.assert_array_equal(ids, [1, 2, 3])
        np.testing.assert_array_almost_equal(lons, [-80.0, -79.0, -78.0])
        np.testing.assert_array_almost_equal(lats, [30.0, 31.0, 32.0])
        np.testing.assert_array_almost_equal(vals, [0.0, 1.5, 0.0])

    def test_extra_columns(self, tmp_path):
        """gr3 files may have extra columns (e.g., secofs.nudge.gr3 has 5 cols)."""
        from nos_utils.io.schism_grid import SchismGrid

        gr3_path = tmp_path / "test.gr3"
        with open(gr3_path, "w") as f:
            f.write("  0.130  0.500\n")
            f.write("  0  4\n")
            f.write("  1  -82.996  30.009  0.0  227792\n")
            f.write("  2  -82.998  30.006  0.0  227792\n")
            f.write("  3  -82.980  30.009  5.5  989533\n")
            f.write("  4  -82.978  30.004  0.0  989533\n")

        ids, lons, lats, vals = SchismGrid.read_gr3_values(gr3_path)
        assert len(ids) == 4
        np.testing.assert_array_almost_equal(vals, [0.0, 0.0, 5.5, 0.0])


class TestNudgingWithSyntheticRTOFS:
    """Integration test with synthetic RTOFS data and small grid."""

    def _make_gr3(self, path: Path, n_nodes: int, lons, lats, weights):
        """Create a gr3 file with given coordinates and weights."""
        with open(path, "w") as f:
            f.write("test nudge\n")
            f.write(f"  0  {n_nodes}\n")
            for i in range(n_nodes):
                f.write(f"  {i+1}  {lons[i]:.5f}  {lats[i]:.5f}  {weights[i]:.6e}\n")

    def _make_grid(self, path: Path, n_nodes: int, lons, lats, depths):
        """Create a minimal grid file with given coordinates and depths."""
        with open(path, "w") as f:
            f.write("test grid\n")
            f.write(f"  0  {n_nodes}\n")
            for i in range(n_nodes):
                f.write(f"  {i+1}  {lons[i]:.5f}  {lats[i]:.5f}  {depths[i]:.1f}\n")
            f.write("0 = number of open boundaries\n")
            f.write("0 = total open boundary nodes\n")

    def _make_vgrid_simple(self, path: Path, nvrt: int = 10):
        """Create a simple (non-LSC2) vgrid.in."""
        with open(path, "w") as f:
            f.write(f"  {nvrt}  1  100.0\n")
            # 1 Z-level
            f.write("Z levels\n")
            f.write("  1  -100.0\n")
            # S-levels
            f.write("S levels\n")
            for i in range(nvrt - 1):
                sigma = -1.0 + i * (1.0 / (nvrt - 2))
                f.write(f"  {i+2}  {sigma:.4f}\n")

    def _make_rtofs_3d(self, path: Path, n_times: int, n_levels: int,
                       lon_range, lat_range, grid_size: int = 10):
        """Create a synthetic RTOFS 3D file."""
        ds = netCDF4.Dataset(str(path), "w", format="NETCDF4")

        lons_1d = np.linspace(lon_range[0], lon_range[1], grid_size)
        lats_1d = np.linspace(lat_range[0], lat_range[1], grid_size)
        lons_2d, lats_2d = np.meshgrid(lons_1d, lats_1d)
        depths = np.array([0, 5, 10, 25, 50, 100, 200, 500, 1000, 2000][:n_levels],
                          dtype=np.float32)

        ny, nx = lons_2d.shape

        ds.createDimension("MT", n_times)
        ds.createDimension("Depth", n_levels)
        ds.createDimension("Y", ny)
        ds.createDimension("X", nx)

        lon_var = ds.createVariable("Longitude", "f4", ("Y", "X"))
        lon_var[:] = lons_2d
        lat_var = ds.createVariable("Latitude", "f4", ("Y", "X"))
        lat_var[:] = lats_2d
        depth_var = ds.createVariable("Depth", "f4", ("Depth",))
        depth_var[:] = depths

        # Temperature: ~20C, varying slightly with depth
        temp = ds.createVariable("temperature", "f4", ("MT", "Depth", "Y", "X"))
        for t in range(n_times):
            for z in range(n_levels):
                temp[t, z, :, :] = 25.0 - z * 0.5 + np.random.randn(ny, nx) * 0.1

        # Salinity: ~35 PSU
        salt = ds.createVariable("salinity", "f4", ("MT", "Depth", "Y", "X"))
        for t in range(n_times):
            for z in range(n_levels):
                salt[t, z, :, :] = 35.0 + z * 0.01 + np.random.randn(ny, nx) * 0.01

        ds.close()

    def test_end_to_end_synthetic(self, tmp_path):
        """Full nudging pipeline with synthetic data."""
        # Create synthetic nudge nodes (5 nodes, 3 with weight > 0)
        n_nodes = 5
        lons = np.array([-77.0, -76.0, -75.0, -74.0, -73.0])
        lats = np.array([30.0, 31.0, 32.0, 33.0, 34.0])
        weights = np.array([0.0, 0.5, 1.0, 0.5, 0.0])
        depths = np.array([10.0, 50.0, 200.0, 100.0, 5.0])

        gr3_path = tmp_path / "test.nudge.gr3"
        self._make_gr3(gr3_path, n_nodes, lons, lats, weights)

        grid_path = tmp_path / "hgrid.ll"
        self._make_grid(grid_path, n_nodes, lons, lats, depths)

        vgrid_path = tmp_path / "vgrid.in"
        self._make_vgrid_simple(vgrid_path, nvrt=10)

        # Create 2 synthetic RTOFS 3D files (6-hourly)
        # Use rtofs.{date-1} since RTOFSProcessor searches PDY-1 first
        rtofs_dir = tmp_path / "rtofs.20260331"
        rtofs_dir.mkdir()

        for fhr in [0, 6]:
            rtofs_file = rtofs_dir / f"rtofs_glo_3dz_f{fhr:03d}_6hrly_hvr_US_east.nc"
            self._make_rtofs_3d(
                rtofs_file,
                n_times=1, n_levels=5,
                lon_range=(-80, -70), lat_range=(28, 36),
                grid_size=20,
            )

        config = ForcingConfig(
            lon_min=-80.0, lon_max=-70.0,
            lat_min=28.0, lat_max=36.0,
            pdy="20260401", cyc=12,
            nowcast_hours=6, forecast_hours=48,
            nudging_enabled=True,
            n_levels=10,
            grid_file=grid_path,
        )

        out_dir = tmp_path / "output"
        proc = NudgingProcessor(
            config, tmp_path, out_dir,
            nudge_weight_file=gr3_path,
            rtofs_input_path=tmp_path,
        )

        # Patch vgrid loading to use our simple vgrid
        # Patch validate_file_size to skip size check on tiny synthetic files
        with patch.object(proc, "_load_vgrid") as mock_vgrid, \
             patch("nos_utils.forcing.rtofs.RTOFSProcessor.validate_file_size",
                   return_value=True):
            from nos_utils.io.schism_vgrid import SchismVgrid
            mock_vgrid.return_value = SchismVgrid.read(vgrid_path)

            result = proc.process()

        assert result.success, f"Nudging failed: {result.errors}"
        assert len(result.output_files) == 2

        # Check output files
        for fname in ["TEM_nu.nc", "SAL_nu.nc"]:
            fpath = out_dir / fname
            assert fpath.exists(), f"{fname} not created"

            ds = netCDF4.Dataset(str(fpath))

            # Should have the 3 nodes with weight > 0 (or subset with valid RTOFS)
            n_out_nodes = ds.dimensions["node"].size
            assert n_out_nodes > 0
            assert n_out_nodes <= 3  # max 3 nudge nodes

            # Check time dimension
            assert ds.dimensions["time"].size >= 2

            # Check tracer has valid values (not all NaN)
            tracer = ds.variables["tracer_concentration"][:]
            assert np.all(np.isfinite(tracer))

            # map_to_global_node should be sorted 1-based IDs
            node_map = ds.variables["map_to_global_node"][:]
            assert np.all(node_map > 0)

            ds.close()

        # Check metadata
        assert result.metadata["fortran_used"] is False
        assert result.metadata["n_nudge_nodes"] > 0

    def test_no_rtofs_returns_failure(self, tmp_path):
        """If no RTOFS files found, should fail gracefully."""
        n_nodes = 5
        lons = np.linspace(-78, -73, n_nodes)
        lats = np.linspace(30, 34, n_nodes)
        weights = np.ones(n_nodes)

        gr3_path = tmp_path / "test.nudge.gr3"
        self._make_gr3(gr3_path, n_nodes, lons, lats, weights)

        config = ForcingConfig(
            lon_min=-80.0, lon_max=-70.0,
            lat_min=28.0, lat_max=36.0,
            pdy="20260401", cyc=12,
            nudging_enabled=True,
            n_levels=10,
        )

        proc = NudgingProcessor(
            config, tmp_path / "empty_rtofs", tmp_path / "out",
            nudge_weight_file=gr3_path,
            rtofs_input_path=tmp_path / "empty_rtofs",
        )
        result = proc.process()
        assert not result.success
        assert any("RTOFS" in e for e in result.errors)


class TestPrecomputedNudgeWeights:
    """Tests for precomputed Fortran REMESH nudge weight replay."""

    def test_load_remesh_export(self, tmp_path):
        """Test parsing of Fortran nudge export text file."""
        from nos_utils.interp.precomputed_weights import load_remesh_export

        export_path = tmp_path / "obc_nudge_remesh_export.txt"
        with open(export_path, "w") as f:
            f.write("## OBC_NUDGE_REMESH_EXPORT_V1\n")
            f.write("## n_target=       3\n")
            f.write("## n_source_total=       7\n")
            f.write("## n_source_data=       3\n")
            f.write("## corner_mean=  1.5000000000000000E+01\n")
            f.write("## SOURCE_POINTS\n")
            f.write("## idx lon lat is_corner\n")
            # 4 corners + 3 data points
            f.write("       1  -8.0800000000000000E+01   2.8900000000000000E+01  1\n")
            f.write("       2  -8.0800000000000000E+01   3.5100000000000000E+01  1\n")
            f.write("       3  -6.9200000000000000E+01   3.5100000000000000E+01  1\n")
            f.write("       4  -6.9200000000000000E+01   2.8900000000000000E+01  1\n")
            f.write("       5  -7.8000000000000000E+01   3.0000000000000000E+01  0\n")
            f.write("       6  -7.6000000000000000E+01   3.2000000000000000E+01  0\n")
            f.write("       7  -7.4000000000000000E+01   3.4000000000000000E+01  0\n")
            f.write("## TARGET_MAPPING\n")
            f.write("## target idx1 idx2 idx3 w1 w2 w3 mode donor\n")
            f.write("       1       5       6       7   3.3333333333333331E-01   3.3333333333333331E-01   3.3333333333333337E-01  0      -1\n")
            f.write("       2       5       6       3   5.0000000000000000E-01   2.5000000000000000E-01   2.5000000000000000E-01  1      -1\n")
            f.write("       3       5       6       7   0.0000000000000000E+00   0.0000000000000000E+00   0.0000000000000000E+00  2       1\n")

        result = load_remesh_export(export_path)

        assert len(result["source_lon"]) == 7
        assert len(result["source_lat"]) == 7
        assert result["source_is_corner"].sum() == 4
        assert result["target_idx"].shape == (3, 3)
        assert result["weights"].shape == (3, 3)
        assert result["mode"].tolist() == [0, 1, 2]
        assert result["donor"].tolist() == [-1, -1, 1]

    def test_apply_precomputed_nudge_basic(self, tmp_path):
        """Test apply_precomputed_nudge with a synthetic NPZ."""
        from nos_utils.interp.precomputed_weights import apply_precomputed_nudge

        # Create a simple 10x10 grid
        ny, nx = 10, 10
        lon_1d = np.linspace(-80, -70, nx)
        lat_1d = np.linspace(28, 36, ny)
        lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)

        # Create a field with known values
        field = np.arange(ny * nx, dtype=np.float64).reshape(ny, nx)

        # Synthetic NPZ: 5 target nodes, each maps to 3 grid cells
        n_target = 5
        npz = {
            "vertex_flat_idx": np.array([
                [11, 12, 21],
                [22, 23, 32],
                [33, 34, 43],
                [44, 45, 54],
                [55, 56, 65],
            ], dtype=np.int32),
            "vertex_is_corner": np.zeros((n_target, 3), dtype=bool),
            "source_data_flat_idx": np.arange(100, dtype=np.int32),
            "weights": np.array([
                [0.5, 0.3, 0.2],
                [0.4, 0.4, 0.2],
                [0.33, 0.34, 0.33],
                [0.25, 0.50, 0.25],
                [0.6, 0.2, 0.2],
            ], dtype=np.float64),
            "mode": np.array([0, 0, 0, 0, 0], dtype=np.int32),
            "donor": np.array([-1, -1, -1, -1, -1], dtype=np.int32),
            "grid_shape": np.array([ny, nx], dtype=np.int32),
            "grid_hash": np.array(["test_hash"]),
            "corner_mean_export": np.float64(0.0),
            "n_target": np.int32(n_target),
            "n_source_data": np.int32(100),
        }

        result = apply_precomputed_nudge(npz, field)

        assert result.shape == (n_target,)
        assert result.dtype == np.float32

        # Verify first target: 0.5*field[1,1] + 0.3*field[1,2] + 0.2*field[2,1]
        expected_0 = 0.5 * field.ravel()[11] + 0.3 * field.ravel()[12] + 0.2 * field.ravel()[21]
        np.testing.assert_almost_equal(result[0], expected_0, decimal=5)

    def test_apply_precomputed_nudge_with_fallback(self, tmp_path):
        """Test that mode=2 nodes copy from their donor."""
        from nos_utils.interp.precomputed_weights import apply_precomputed_nudge

        ny, nx = 5, 5
        field = np.full((ny, nx), 20.0, dtype=np.float64)

        npz = {
            "vertex_flat_idx": np.array([
                [5, 6, 10],
                [5, 6, 10],  # Donor for node 2 (mode=2)
            ], dtype=np.int32),
            "vertex_is_corner": np.zeros((2, 3), dtype=bool),
            "source_data_flat_idx": np.arange(25, dtype=np.int32),
            "weights": np.array([
                [0.5, 0.3, 0.2],
                [0.0, 0.0, 0.0],  # All zero weights — mode=2 copies from donor
            ], dtype=np.float64),
            "mode": np.array([0, 2], dtype=np.int32),
            "donor": np.array([-1, 0], dtype=np.int32),  # node 1 copies from node 0
            "grid_shape": np.array([ny, nx], dtype=np.int32),
            "grid_hash": np.array(["test"]),
            "corner_mean_export": np.float64(0.0),
            "n_target": np.int32(2),
            "n_source_data": np.int32(25),
        }

        result = apply_precomputed_nudge(npz, field)

        # Node 1 (mode=2) should equal node 0
        np.testing.assert_almost_equal(result[1], result[0], decimal=5)

    def test_apply_precomputed_nudge_nan_handling(self, tmp_path):
        """NaN source values should not propagate to output."""
        from nos_utils.interp.precomputed_weights import apply_precomputed_nudge

        ny, nx = 5, 5
        field = np.full((ny, nx), 20.0, dtype=np.float64)
        field[0, :] = np.nan  # First row is NaN (land)

        npz = {
            "vertex_flat_idx": np.array([
                [6, 7, 11],  # All ocean (row 1+)
                [1, 2, 6],   # idx 1,2 are NaN (row 0)
            ], dtype=np.int32),
            "vertex_is_corner": np.zeros((2, 3), dtype=bool),
            "source_data_flat_idx": np.arange(5, 25, dtype=np.int32),  # skip row 0
            "weights": np.array([
                [0.5, 0.3, 0.2],
                [0.3, 0.3, 0.4],
            ], dtype=np.float64),
            "mode": np.array([0, 0], dtype=np.int32),
            "donor": np.array([-1, -1], dtype=np.int32),
            "grid_shape": np.array([ny, nx], dtype=np.int32),
            "grid_hash": np.array(["test"]),
            "corner_mean_export": np.float64(0.0),
            "n_target": np.int32(2),
            "n_source_data": np.int32(20),
        }

        result = apply_precomputed_nudge(npz, field)

        # All results should be finite (NaN replaced with corner_mean=20.0)
        assert np.all(np.isfinite(result))

    def test_apply_precomputed_nudge_shape_mismatch(self):
        """Should raise ValueError on grid shape mismatch."""
        from nos_utils.interp.precomputed_weights import apply_precomputed_nudge

        npz = {
            "grid_shape": np.array([10, 20], dtype=np.int32),
        }
        field = np.zeros((5, 5))

        with pytest.raises(ValueError, match="Grid shape mismatch"):
            apply_precomputed_nudge(npz, field)

    def test_build_nudge_npz(self, tmp_path):
        """Test building NPZ from an export text file."""
        scipy = pytest.importorskip("scipy")
        from nos_utils.interp.precomputed_weights import build_nudge_npz

        # Create synthetic RTOFS grid
        ny, nx = 20, 30
        lon_1d = np.linspace(-80, -70, nx)
        lat_1d = np.linspace(28, 36, ny)
        rtofs_lon, rtofs_lat = np.meshgrid(lon_1d, lat_1d)

        # Create a synthetic export file
        # Pick 3 ocean source points that exist on the grid
        src_lons = [lon_1d[5], lon_1d[10], lon_1d[15]]
        src_lats = [lat_1d[5], lat_1d[10], lat_1d[15]]

        export_path = tmp_path / "obc_nudge_remesh_export.txt"
        with open(export_path, "w") as f:
            f.write("## OBC_NUDGE_REMESH_EXPORT_V1\n")
            f.write("## n_target=       2\n")
            f.write("## n_source_total=       7\n")
            f.write("## n_source_data=       3\n")
            f.write("## corner_mean=  2.0000000000000000E+01\n")
            f.write("## SOURCE_POINTS\n")
            f.write("## idx lon lat is_corner\n")
            # 4 corners (target bounding box + padding)
            f.write(f"       1  {-78.1:25.16E}  {29.9:25.16E}  1\n")
            f.write(f"       2  {-78.1:25.16E}  {34.1:25.16E}  1\n")
            f.write(f"       3  {-71.9:25.16E}  {34.1:25.16E}  1\n")
            f.write(f"       4  {-71.9:25.16E}  {29.9:25.16E}  1\n")
            # 3 ocean points
            for i, (slon, slat) in enumerate(zip(src_lons, src_lats)):
                f.write(f"       {i+5}  {slon:25.16E}  {slat:25.16E}  0\n")
            f.write("## TARGET_MAPPING\n")
            f.write("## target idx1 idx2 idx3 w1 w2 w3 mode donor\n")
            f.write("       1       5       6       7   4.0000000000000000E-01   3.0000000000000000E-01   3.0000000000000000E-01  0      -1\n")
            f.write("       2       5       6       7   2.0000000000000000E-01   5.0000000000000000E-01   3.0000000000000000E-01  0      -1\n")

        out_npz = tmp_path / "test.obc_nudge_weights.npz"
        build_nudge_npz(export_path, rtofs_lon, rtofs_lat, out_npz)

        assert out_npz.exists()

        data = dict(np.load(str(out_npz)))
        assert int(data["n_target"]) == 2
        assert int(data["n_source_data"]) == 3
        assert data["weights"].shape == (2, 3)
        assert data["vertex_flat_idx"].shape == (2, 3)
        assert data["mode"].shape == (2,)

    def test_find_nudge_weights_from_env(self, secofs_config, tmp_path):
        """Test _find_nudge_weights discovers NPZ from FIXofs."""
        # Create a dummy NPZ
        npz_path = tmp_path / "secofs.obc_nudge_weights.npz"
        np.savez(str(npz_path), n_target=np.int32(100))

        proc = NudgingProcessor(secofs_config, tmp_path, tmp_path / "out")

        import os
        old_fix = os.environ.get("FIXofs")
        try:
            os.environ["FIXofs"] = str(tmp_path)
            result = proc._find_nudge_weights()
            assert result is not None
            assert int(result["n_target"]) == 100
        finally:
            if old_fix is not None:
                os.environ["FIXofs"] = old_fix
            else:
                os.environ.pop("FIXofs", None)

    def test_find_nudge_weights_not_found(self, secofs_config, tmp_path):
        """Returns None when no NPZ exists."""
        proc = NudgingProcessor(secofs_config, tmp_path / "empty", tmp_path / "out")
        # Clear any cached value
        if hasattr(proc, '_nudge_weights_cache'):
            delattr(proc, '_nudge_weights_cache')

        import os
        old_fix = os.environ.get("FIXofs")
        try:
            os.environ.pop("FIXofs", None)
            os.environ.pop("FIXstofs3d", None)
            result = proc._find_nudge_weights()
            assert result is None
        finally:
            if old_fix is not None:
                os.environ["FIXofs"] = old_fix


class TestPrecomputed3DWeights:
    """Tests for precomputed 3D T/S boundary weight build and apply."""

    def test_build_3d_npz(self, tmp_path):
        """Test building 3D NPZ from an export text file."""
        scipy = pytest.importorskip("scipy")
        from nos_utils.interp.precomputed_weights import build_3d_npz

        # Create synthetic regional RTOFS 3D grid (smaller than global)
        ny, nx = 20, 30
        lon_1d = np.linspace(-80, -70, nx)
        lat_1d = np.linspace(28, 36, ny)
        rtofs_lon, rtofs_lat = np.meshgrid(lon_1d, lat_1d)

        # Create a synthetic export file
        src_lons = [lon_1d[5], lon_1d[10], lon_1d[15]]
        src_lats = [lat_1d[5], lat_1d[10], lat_1d[15]]

        export_path = tmp_path / "obc_3d_remesh_export.txt"
        with open(export_path, "w") as f:
            f.write("## OBC_3D_REMESH_EXPORT_V1\n")
            f.write("## n_target=       2\n")
            f.write("## n_source_total=       7\n")
            f.write("## n_source_data=       3\n")
            f.write("## corner_mean=  2.0000000000000000E+01\n")
            f.write("## SOURCE_POINTS\n")
            f.write("## idx lon lat is_corner\n")
            # 4 corners
            f.write(f"       1  {-78.1:25.16E}  {29.9:25.16E}  1\n")
            f.write(f"       2  {-78.1:25.16E}  {34.1:25.16E}  1\n")
            f.write(f"       3  {-71.9:25.16E}  {34.1:25.16E}  1\n")
            f.write(f"       4  {-71.9:25.16E}  {29.9:25.16E}  1\n")
            # 3 ocean points
            for i, (slon, slat) in enumerate(zip(src_lons, src_lats)):
                f.write(f"       {i+5}  {slon:25.16E}  {slat:25.16E}  0\n")
            f.write("## TARGET_MAPPING\n")
            f.write("## target idx1 idx2 idx3 w1 w2 w3 mode donor\n")
            f.write("       1       5       6       7"
                    "   4.0000000000000000E-01"
                    "   3.0000000000000000E-01"
                    "   3.0000000000000000E-01  0      -1\n")
            f.write("       2       5       6       7"
                    "   2.0000000000000000E-01"
                    "   5.0000000000000000E-01"
                    "   3.0000000000000000E-01  0      -1\n")

        out_npz = tmp_path / "test.obc_3d_weights.npz"
        build_3d_npz(export_path, rtofs_lon, rtofs_lat, out_npz)

        assert out_npz.exists()

        data = dict(np.load(str(out_npz)))
        assert int(data["n_target"]) == 2
        assert int(data["n_source_data"]) == 3
        assert data["weights"].shape == (2, 3)
        assert data["vertex_flat_idx"].shape == (2, 3)
        assert data["mode"].shape == (2,)
        assert tuple(data["grid_shape"]) == (20, 30)

    def test_apply_precomputed_ssh_with_3d_grid(self, tmp_path):
        """apply_precomputed_ssh works with any grid shape (3D regional)."""
        from nos_utils.interp.precomputed_weights import apply_precomputed_ssh

        # Simulate a regional 3D grid (10x15)
        ny, nx = 10, 15
        n_target = 3
        n_source = ny * nx - 4  # some ocean cells (minus 4 corners)

        # Build a minimal NPZ-like dict
        npz = {
            "grid_shape": np.array([ny, nx], dtype=np.int32),
            "vertex_flat_idx": np.array(
                [[4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=np.int32),
            "vertex_is_corner": np.array(
                [[False, False, False],
                 [False, False, False],
                 [False, False, False]]),
            "source_data_flat_idx": np.arange(4, 4 + 20, dtype=np.int32),
            "weights": np.array(
                [[0.5, 0.3, 0.2],
                 [0.4, 0.4, 0.2],
                 [0.3, 0.3, 0.4]], dtype=np.float64),
            "mode": np.array([0, 0, 0], dtype=np.int32),
            "donor": np.array([-1, -1, -1], dtype=np.int32),
        }

        field = np.random.rand(ny, nx).astype(np.float32) * 30.0
        result = apply_precomputed_ssh(npz, field)

        assert result.shape == (n_target,)
        assert result.dtype == np.float32
        # Verify it's a weighted sum
        flat = field.astype(np.float64).ravel()
        expected_0 = (0.5 * flat[4] + 0.3 * flat[5] + 0.2 * flat[6])
        np.testing.assert_allclose(result[0], expected_0, atol=1e-5)

    def test_apply_precomputed_ssh_rejects_wrong_shape(self):
        """apply_precomputed_ssh raises ValueError on grid shape mismatch."""
        from nos_utils.interp.precomputed_weights import apply_precomputed_ssh

        npz = {
            "grid_shape": np.array([1710, 742], dtype=np.int32),
            "vertex_flat_idx": np.array([[0, 1, 2]], dtype=np.int32),
            "vertex_is_corner": np.array([[False, False, False]]),
            "source_data_flat_idx": np.array([0, 1, 2], dtype=np.int32),
            "weights": np.array([[0.5, 0.3, 0.2]], dtype=np.float64),
            "mode": np.array([0], dtype=np.int32),
            "donor": np.array([-1], dtype=np.int32),
        }

        # Pass a 3298x4500 field — should fail because NPZ expects 1710x742
        wrong_field = np.zeros((3298, 4500), dtype=np.float32)
        with pytest.raises(ValueError, match="Grid shape mismatch"):
            apply_precomputed_ssh(npz, wrong_field)
