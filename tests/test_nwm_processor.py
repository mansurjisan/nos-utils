"""Tests for NWMProcessor."""

import json
from pathlib import Path

import numpy as np
import pytest

from nos_utils.config import ForcingConfig
from nos_utils.forcing.nwm import NWMProcessor, RiverConfig, MONTHLY_FLOW_FACTOR


class TestRiverConfig:
    def test_from_text(self, tmp_path):
        cfg_file = tmp_path / "rivers.txt"
        cfg_file.write_text(
            "# feature_id node_index name clim_flow\n"
            "12345 100 TestRiver1 50.0\n"
            "67890 200 TestRiver2 30.0\n"
        )
        rc = RiverConfig.from_text(cfg_file)
        assert rc.n_rivers == 2
        assert rc.feature_ids == [12345, 67890]
        assert rc.node_indices == [100, 200]
        assert rc.clim_flows == [50.0, 30.0]

    def test_from_json_list(self, tmp_path):
        cfg_file = tmp_path / "rivers.json"
        data = [
            {"feature_id": 111, "node_index": 10, "name": "R1", "clim_flow": 25.0},
            {"feature_id": 222, "node_index": 20, "name": "R2", "clim_flow": 15.0},
        ]
        cfg_file.write_text(json.dumps(data))
        rc = RiverConfig.from_json(cfg_file)
        assert rc.n_rivers == 2
        assert rc.names == ["R1", "R2"]


class TestNWMClimatology:
    def test_generate_climatology(self, tmp_path):
        """Verify climatology fallback produces reasonable output."""
        rc = RiverConfig(
            feature_ids=[111, 222],
            node_indices=[10, 20],
            clim_flows=[100.0, 50.0],
        )
        cfg = ForcingConfig(
            lon_min=-80, lon_max=-70, lat_min=25, lat_max=35,
            pdy="20260401", cyc=12,  # April
        )
        proc = NWMProcessor(cfg, tmp_path, tmp_path / "out", river_config=rc)
        flows, times = proc._generate_climatology(55)

        assert flows.shape == (55, 2)
        assert len(times) == 55

        # April factor = 1.5
        expected_r1 = 100.0 * 1.5
        assert flows[0, 0] == pytest.approx(expected_r1)

    def test_monthly_factors_sum(self):
        """Factors should average ~1.0 over the year."""
        avg = sum(MONTHLY_FLOW_FACTOR.values()) / 12
        assert 0.9 < avg < 1.1


class TestNWMProcess:
    def test_no_config_returns_failure(self, mock_config, tmp_path):
        proc = NWMProcessor(mock_config, tmp_path, tmp_path / "out")
        result = proc.process()
        assert not result.success
        assert "river configuration" in result.errors[0].lower()

    def test_climatology_fallback(self, mock_config, tmp_path):
        """With no NWM files, should fall back to climatology and succeed."""
        rc = RiverConfig(
            feature_ids=[111],
            node_indices=[10],
            clim_flows=[75.0],
        )
        out_dir = tmp_path / "out"
        proc = NWMProcessor(mock_config, tmp_path / "empty_nwm", out_dir, river_config=rc)
        result = proc.process()

        assert result.success
        assert result.metadata["used_climatology"] is True

        # Check output files exist
        assert (out_dir / "vsource.th").exists()
        assert (out_dir / "msource.th").exists()
        assert (out_dir / "source_sink.in").exists()

    def test_vsource_format(self, mock_config, tmp_path):
        """Verify vsource.th has correct format: time flow1 flow2 ..."""
        rc = RiverConfig(
            feature_ids=[111, 222],
            node_indices=[10, 20],
            clim_flows=[100.0, 50.0],
        )
        out_dir = tmp_path / "out"
        proc = NWMProcessor(mock_config, tmp_path, out_dir, river_config=rc)
        proc.process()

        vsource = (out_dir / "vsource.th").read_text().strip().split("\n")
        # Each line: time_seconds flow1 flow2
        parts = vsource[0].split()
        assert len(parts) == 3  # time + 2 rivers
        assert float(parts[0]) == 0.0  # First time = 0

    def test_source_sink_format(self, mock_config, tmp_path):
        """Verify source_sink.in lists all river nodes."""
        rc = RiverConfig(
            feature_ids=[111, 222, 333],
            node_indices=[10, 20, 30],
            clim_flows=[50.0, 30.0, 20.0],
        )
        out_dir = tmp_path / "out"
        proc = NWMProcessor(mock_config, tmp_path, out_dir, river_config=rc)
        proc.process()

        lines = (out_dir / "source_sink.in").read_text().strip().split("\n")
        assert lines[0] == "3"  # 3 sources
        assert lines[-1] == "0"  # 0 sinks
