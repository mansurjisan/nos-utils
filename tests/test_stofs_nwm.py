"""Tests for STOFS-3D-ATL NWM extensions."""

import json
import pytest
from pathlib import Path

import numpy as np

from nos_utils.config import ForcingConfig
from nos_utils.forcing.nwm import NWMProcessor, RiverConfig


class TestSourcesJSON:
    """Test RiverConfig.from_sources_json() for STOFS gen_sourcesink format."""

    def test_basic_parse(self, tmp_path):
        sources = {
            "100": [20104159, 9643431],
            "200": [12345678],
            "300": [111, 222, 333],
        }
        path = tmp_path / "sources.json"
        path.write_text(json.dumps(sources))

        rc = RiverConfig.from_sources_json(path)
        assert rc.n_rivers == 3
        assert rc.node_indices == [100, 200, 300]
        assert rc.feature_id_groups == [[20104159, 9643431], [12345678], [111, 222, 333]]
        assert rc.feature_ids == [20104159, 12345678, 111]  # first of each group

    def test_empty_groups(self, tmp_path):
        sources = {"100": [], "200": [12345]}
        path = tmp_path / "sources.json"
        path.write_text(json.dumps(sources))

        rc = RiverConfig.from_sources_json(path)
        assert rc.n_rivers == 2
        assert rc.feature_id_groups[0] == []
        assert rc.feature_id_groups[1] == [12345]

    def test_sorted_by_element_id(self, tmp_path):
        sources = {"300": [3], "100": [1], "200": [2]}
        path = tmp_path / "sources.json"
        path.write_text(json.dumps(sources))

        rc = RiverConfig.from_sources_json(path)
        assert rc.node_indices == [100, 200, 300]


class TestSTOFSMode:
    """Test is_stofs_mode detection."""

    def test_stofs_config_with_sources_json(self, tmp_path):
        sources = {"100": [123], "200": [456]}
        path = tmp_path / "sources.json"
        path.write_text(json.dumps(sources))

        cfg = ForcingConfig.for_stofs_3d_atl(pdy="20260401", cyc=12)
        rc = RiverConfig.from_sources_json(path)
        proc = NWMProcessor(cfg, tmp_path, tmp_path, river_config=rc)
        assert proc.is_stofs_mode is True

    def test_secofs_mode(self, tmp_path):
        cfg = ForcingConfig.for_secofs(pdy="20260401", cyc=12)
        rc = RiverConfig([1], [1], [50.0])
        proc = NWMProcessor(cfg, tmp_path, tmp_path, river_config=rc)
        assert proc.is_stofs_mode is False


class TestSTOFSFileDiscovery:
    """Test STOFS NWM file discovery patterns."""

    def test_empty_directory(self, tmp_path):
        cfg = ForcingConfig.for_stofs_3d_atl(pdy="20260401", cyc=12)
        rc = RiverConfig.from_sources_json(self._write_sources(tmp_path))
        proc = NWMProcessor(cfg, tmp_path, tmp_path, river_config=rc)
        files = proc.find_input_files()
        assert files == []

    def test_finds_medium_range_files(self, tmp_path):
        """Create a minimal NWM directory structure with medium_range_mem1 files."""
        cfg = ForcingConfig.for_stofs_3d_atl(pdy="20260402", cyc=12)
        rc = RiverConfig.from_sources_json(self._write_sources(tmp_path))

        # Create yesterday t12z f001 file
        nwm_dir = tmp_path / "nwm.20260401" / "medium_range_mem1"
        nwm_dir.mkdir(parents=True)
        fname = "nwm.t12z.medium_range.channel_rt_1.f001.conus.nc"
        (nwm_dir / fname).write_bytes(b"\x00" * 15_000_000)  # 15MB > 10MB threshold

        proc = NWMProcessor(cfg, tmp_path, tmp_path, river_config=rc)
        files = proc.find_input_files()
        assert len(files) == 1
        assert "t12z" in files[0].name

    @staticmethod
    def _write_sources(tmp_path):
        sources = {"100": [123]}
        path = tmp_path / "sources.json"
        path.write_text(json.dumps(sources))
        return path


class TestAutoDetectSourcesJSON:
    """Test that NWMProcessor auto-detects sources.json vs regular JSON."""

    def test_auto_detect_stofs_format(self, tmp_path):
        sources = {"100": [123, 456], "200": [789]}
        path = tmp_path / "river.json"
        path.write_text(json.dumps(sources))

        cfg = ForcingConfig.for_stofs_3d_atl(pdy="20260401", cyc=12,
                                              river_config_file=path)
        proc = NWMProcessor(cfg, tmp_path, tmp_path)
        rc = proc.river_config
        assert rc is not None
        assert rc.feature_id_groups is not None
        assert rc.n_rivers == 2

    def test_auto_detect_regular_format(self, tmp_path):
        data = [{"feature_id": 123, "node_index": 1, "clim_flow": 50.0}]
        path = tmp_path / "river.json"
        path.write_text(json.dumps(data))

        cfg = ForcingConfig.for_secofs(pdy="20260401", cyc=12,
                                       river_config_file=path)
        proc = NWMProcessor(cfg, tmp_path, tmp_path)
        rc = proc.river_config
        assert rc is not None
        assert rc.feature_id_groups is None  # regular format
