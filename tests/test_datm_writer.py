"""Tests for DATMWriter."""

from datetime import datetime, timedelta

import numpy as np
import pytest

from nos_utils.forcing.datm_writer import DATMWriter, SFLUX_TO_DATM

netCDF4 = pytest.importorskip("netCDF4")


class TestDATMWriter:
    def test_write_basic(self, synthetic_data, synthetic_grid, tmp_output_dir):
        lons, lats = synthetic_grid
        data, times = synthetic_data

        writer = DATMWriter()
        output_path = tmp_output_dir / "datm_forcing.nc"
        result = writer.write(data, times, lons, lats, output_path)

        assert result.exists()

    def test_datm_variable_names(self, synthetic_data, synthetic_grid, tmp_output_dir):
        """Verify DATM naming convention (e.g., UGRD_10maboveground)."""
        lons, lats = synthetic_grid
        data, times = synthetic_data

        writer = DATMWriter()
        output_path = tmp_output_dir / "datm_forcing.nc"
        writer.write(data, times, lons, lats, output_path)

        ds = netCDF4.Dataset(str(output_path))
        for sflux_name in ["uwind", "vwind", "stmp", "spfh", "prmsl", "prate", "dswrf", "dlwrf"]:
            datm_name = SFLUX_TO_DATM[sflux_name]
            assert datm_name in ds.variables, f"Missing DATM variable: {datm_name}"
        ds.close()

    def test_datm_dimensions(self, synthetic_data, synthetic_grid, tmp_output_dir):
        """Verify dimensions match expected DATM format (lesson #19)."""
        lons, lats = synthetic_grid
        data, times = synthetic_data

        writer = DATMWriter()
        output_path = tmp_output_dir / "datm_forcing.nc"
        writer.write(data, times, lons, lats, output_path)

        ds = netCDF4.Dataset(str(output_path))
        assert "time" in ds.dimensions
        assert "latitude" in ds.dimensions
        assert "longitude" in ds.dimensions
        assert ds.dimensions["longitude"].size == len(lons)
        assert ds.dimensions["latitude"].size == len(lats)
        assert ds.dimensions["time"].size == len(times)
        ds.close()

    def test_get_grid_dims(self, synthetic_grid):
        """Verify grid dims helper matches array lengths."""
        lons, lats = synthetic_grid
        nx, ny = DATMWriter.get_grid_dims(lons, lats)
        assert nx == len(lons)
        assert ny == len(lats)

    def test_write_no_hrrr_falls_through(self, synthetic_data, synthetic_grid, tmp_output_dir):
        """write_blended with no HRRR data should produce GFS-only output."""
        lons, lats = synthetic_grid
        data, times = synthetic_data

        writer = DATMWriter()
        output_path = tmp_output_dir / "datm_forcing.nc"
        result = writer.write_blended(
            gfs_data=data, hrrr_data=None,
            gfs_times=times, hrrr_times=None,
            target_lons=lons, target_lats=lats,
            hrrr_lons_2d=None, hrrr_lats_2d=None,
            output_path=output_path,
        )
        assert result.exists()
