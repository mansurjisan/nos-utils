"""Tests for STOFS-3D-ATL static vsink.th copy from FIX."""

import json
from pathlib import Path

import pytest

from nos_utils.config import ForcingConfig
from nos_utils.forcing.nwm import NWMProcessor, RiverConfig


def _write_sources(tmp_path: Path) -> Path:
    sources = {"100": [123], "200": [456]}
    path = tmp_path / "sources.json"
    path.write_text(json.dumps(sources))
    return path


class TestVsinkCopy:
    """_copy_static_vsink should find and copy vsink.th from FIX dirs."""

    def test_copies_from_river_config_parent(self, tmp_path):
        """FIX dir is derived from river_config_file parent (SECOFS pattern)."""
        fix_dir = tmp_path / "fix"
        fix_dir.mkdir()
        sources_json = fix_dir / "sources.json"
        sources_json.write_text(json.dumps({"100": [123]}))
        (fix_dir / "stofs_3d_atl_river_vsink.th").write_text("0.0 -5.0\n3600.0 -5.0\n")

        output_dir = tmp_path / "out"
        output_dir.mkdir()

        cfg = ForcingConfig.for_stofs_3d_atl(
            pdy="20260401", cyc=12, river_config_file=sources_json,
        )
        proc = NWMProcessor(cfg, tmp_path, output_dir)

        result = proc._copy_static_vsink()
        assert result is not None
        assert result.name == "vsink.th"
        assert result.exists()
        assert "0.0 -5.0" in result.read_text()

    def test_falls_back_to_env_var(self, tmp_path, monkeypatch):
        """FIXstofs3d env var should be consulted when config path missing."""
        fix_dir = tmp_path / "env_fix"
        fix_dir.mkdir()
        (fix_dir / "stofs_3d_atl_river_vsink.th").write_text("0.0 -1.0\n")

        output_dir = tmp_path / "out"
        output_dir.mkdir()

        monkeypatch.setenv("FIXstofs3d", str(fix_dir))
        monkeypatch.delenv("FIXofs", raising=False)

        sources = _write_sources(tmp_path)
        cfg = ForcingConfig.for_stofs_3d_atl(pdy="20260401", cyc=12)
        cfg.river_config_file = None  # force env var fallback
        proc = NWMProcessor(cfg, tmp_path, output_dir, river_config=RiverConfig.from_sources_json(sources))

        result = proc._copy_static_vsink()
        assert result is not None
        assert "0.0 -1.0" in result.read_text()

    def test_missing_returns_none(self, tmp_path):
        """With no vsink.th anywhere, the helper returns None (graceful)."""
        sources = _write_sources(tmp_path)
        cfg = ForcingConfig.for_stofs_3d_atl(pdy="20260401", cyc=12)
        cfg.river_config_file = None
        proc = NWMProcessor(
            cfg, tmp_path, tmp_path,
            river_config=RiverConfig.from_sources_json(sources),
        )
        assert proc._copy_static_vsink() is None

    def test_prefers_prefixed_name_over_bare(self, tmp_path):
        """When both filenames exist, prefixed (`stofs_3d_atl_river_vsink.th`)
        wins over bare `vsink.th`."""
        fix_dir = tmp_path / "fix"
        fix_dir.mkdir()
        (fix_dir / "stofs_3d_atl_river_vsink.th").write_text("0.0 PREFIX\n")
        (fix_dir / "vsink.th").write_text("0.0 BARE\n")
        sources = fix_dir / "sources.json"
        sources.write_text(json.dumps({"1": [1]}))

        cfg = ForcingConfig.for_stofs_3d_atl(
            pdy="20260401", cyc=12, river_config_file=sources,
        )
        proc = NWMProcessor(cfg, tmp_path, tmp_path / "out")
        (tmp_path / "out").mkdir()

        result = proc._copy_static_vsink()
        assert result is not None
        text = result.read_text()
        assert "PREFIX" in text
        assert "BARE" not in text
