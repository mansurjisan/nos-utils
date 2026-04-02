"""Tests for SfluxWriter."""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from nos_utils.forcing.sflux_writer import SfluxWriter

# Skip all tests if netCDF4 not available
netCDF4 = pytest.importorskip("netCDF4")


class TestSfluxWriter:
    def test_write_day(self, synthetic_data, synthetic_grid, tmp_output_dir):
        lons, lats = synthetic_grid
        data, times = synthetic_data
        base_date = datetime(2026, 3, 31, 6, 0, 0)

        writer = SfluxWriter(tmp_output_dir, source_index=1)
        files = writer.write_day(data, times, lons, lats, base_date, day_num=1)

        assert len(files) == 3  # air, rad, prc
        assert all(f.exists() for f in files)

    def test_file_naming_uses_single_digit(self, synthetic_data, synthetic_grid, tmp_output_dir):
        """Verify .{N}.nc naming, NOT .{NNNN}.nc (lesson #11)."""
        lons, lats = synthetic_grid
        data, times = synthetic_data
        base_date = datetime(2026, 3, 31, 6, 0, 0)

        writer = SfluxWriter(tmp_output_dir, source_index=1)
        files = writer.write_day(data, times, lons, lats, base_date, day_num=1)

        filenames = [f.name for f in files]
        assert "sflux_air_1.1.nc" in filenames
        assert "sflux_rad_1.1.nc" in filenames
        assert "sflux_prc_1.1.nc" in filenames

        # Ensure NNNN format is NOT used
        for name in filenames:
            assert ".0001." not in name

    def test_source_index_2(self, synthetic_data, synthetic_grid, tmp_output_dir):
        """HRRR files use source_index=2."""
        lons, lats = synthetic_grid
        data, times = synthetic_data
        base_date = datetime(2026, 3, 31, 6, 0, 0)

        writer = SfluxWriter(tmp_output_dir, source_index=2)
        files = writer.write_day(data, times, lons, lats, base_date, day_num=1)

        filenames = [f.name for f in files]
        assert "sflux_air_2.1.nc" in filenames

    def test_write_all_splits_by_day(self, synthetic_grid, tmp_output_dir):
        """Verify write_all groups time steps into separate day files."""
        lons, lats = synthetic_grid
        ny, nx = len(lats), len(lons)

        base_date = datetime(2026, 3, 31, 6, 0, 0)
        # 6 time steps spanning 2 days
        times = [base_date + timedelta(hours=i * 8) for i in range(6)]
        data = {"uwind": [np.ones((ny, nx), dtype=np.float32) for _ in range(6)],
                "vwind": [np.ones((ny, nx), dtype=np.float32) for _ in range(6)]}

        writer = SfluxWriter(tmp_output_dir, source_index=1)
        files = writer.write_all(data, times, lons, lats, base_date)

        # Should have files for multiple days
        assert len(files) >= 2

    def test_monotonic_time_validation(self, synthetic_grid, tmp_output_dir):
        """Non-monotonic time axis raises ValueError (lesson #12)."""
        lons, lats = synthetic_grid
        ny, nx = len(lats), len(lons)

        base_date = datetime(2026, 3, 31, 6, 0, 0)
        # Non-monotonic: t1 > t2
        times = [
            base_date + timedelta(hours=6),
            base_date + timedelta(hours=3),  # backward!
            base_date + timedelta(hours=9),
        ]
        data = {"uwind": [np.ones((ny, nx), dtype=np.float32) for _ in range(3)]}

        writer = SfluxWriter(tmp_output_dir, source_index=1)
        with pytest.raises(ValueError, match="Non-monotonic"):
            writer.write_all(data, times, lons, lats, base_date)

    def test_netcdf_dimensions(self, synthetic_data, synthetic_grid, tmp_output_dir):
        """Verify SCHISM-expected dimension names and shapes."""
        lons, lats = synthetic_grid
        data, times = synthetic_data
        base_date = datetime(2026, 3, 31, 6, 0, 0)

        writer = SfluxWriter(tmp_output_dir, source_index=1)
        files = writer.write_day(data, times, lons, lats, base_date, day_num=1)

        air_file = [f for f in files if "air" in f.name][0]
        ds = netCDF4.Dataset(str(air_file))

        assert "ntime" in ds.dimensions
        assert "ny_grid" in ds.dimensions
        assert "nx_grid" in ds.dimensions
        assert ds.dimensions["nx_grid"].size == len(lons)
        assert ds.dimensions["ny_grid"].size == len(lats)
        assert ds.dimensions["ntime"].size == len(times)

        # Check variable names
        assert "uwind" in ds.variables
        assert "vwind" in ds.variables
        assert "prmsl" in ds.variables
        assert "stmp" in ds.variables
        assert "spfh" in ds.variables
        assert "time" in ds.variables
        assert "lon" in ds.variables
        assert "lat" in ds.variables

        # Check time is monotonic
        time_vals = ds.variables["time"][:]
        assert all(time_vals[i] < time_vals[i + 1] for i in range(len(time_vals) - 1))

        ds.close()

    def test_sflux_inputs(self, tmp_output_dir):
        """Verify sflux_inputs.txt namelist."""
        writer = SfluxWriter(tmp_output_dir, source_index=1)
        inputs_file = writer.write_sflux_inputs(met_num=2)

        assert inputs_file.exists()
        content = inputs_file.read_text()
        assert "&sflux_inputs" in content
        assert "air_1_relative_weight=1.0" in content
        assert "air_2_relative_weight=1.0" in content  # met_num=2
