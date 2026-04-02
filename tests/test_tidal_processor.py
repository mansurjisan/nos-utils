"""Tests for TidalProcessor."""

import math
from datetime import datetime
from pathlib import Path

import pytest

from nos_utils.config import ForcingConfig
from nos_utils.forcing.tidal import (
    TidalProcessor,
    compute_nodal_corrections,
    TIDAL_CONSTITUENTS,
)


class TestNodalCorrections:
    def test_s2_no_correction(self):
        """S2 and P1 should have f=1.0, u=0.0 (no nodal dependence)."""
        nodal = compute_nodal_corrections(datetime(2026, 4, 1), ["S2", "P1"])
        assert nodal["S2"]["f"] == 1.0
        assert nodal["S2"]["u"] == 0.0
        assert nodal["P1"]["f"] == 1.0
        assert nodal["P1"]["u"] == 0.0

    def test_m2_near_unity(self):
        """M2 nodal factor should be close to 1.0 (within 0.04)."""
        nodal = compute_nodal_corrections(datetime(2026, 4, 1), ["M2"])
        assert 0.96 < nodal["M2"]["f"] < 1.04

    def test_k2_larger_variation(self):
        """K2 has the largest nodal variation (~0.286 amplitude)."""
        nodal = compute_nodal_corrections(datetime(2026, 4, 1), ["K2"])
        assert 0.7 < nodal["K2"]["f"] < 1.4

    def test_all_8_constituents(self):
        """All 8 standard constituents should compute without error."""
        consts = ["M2", "S2", "N2", "K2", "K1", "O1", "P1", "Q1"]
        nodal = compute_nodal_corrections(datetime(2026, 4, 1), consts)

        for c in consts:
            assert c in nodal
            assert 0.5 < nodal[c]["f"] < 2.0  # Reasonable range
            assert -30 < nodal[c]["u"] < 30  # Degrees

    def test_different_dates_give_different_results(self):
        """Nodal corrections should change over the 18.6-year cycle."""
        n1 = compute_nodal_corrections(datetime(2020, 1, 1), ["M2"])
        n2 = compute_nodal_corrections(datetime(2029, 1, 1), ["M2"])  # ~half cycle later
        # Should be different (but both near 1.0)
        assert n1["M2"]["f"] != pytest.approx(n2["M2"]["f"], abs=0.001)


class TestTidalProcessor:
    def test_python_native_generation(self, mock_config, tmp_path):
        """Python-native mode should create bctides.in."""
        out_dir = tmp_path / "tidal_out"
        proc = TidalProcessor(mock_config, tmp_path / "no_templates", out_dir)
        result = proc.process()

        assert result.success
        assert result.metadata["mode"] == "python_native"
        bctides = out_dir / "bctides.in"
        assert bctides.exists()

        content = bctides.read_text()
        # Should contain all 8 constituents
        for c in ["M2", "S2", "N2", "K2", "K1", "O1", "P1", "Q1"]:
            assert c in content

    def test_copy_mode(self, mock_config, tmp_path):
        """Should copy static bctides.in from input_path."""
        input_dir = tmp_path / "fix"
        input_dir.mkdir()
        (input_dir / "bctides.in").write_text("! static bctides\ntest content\n")

        out_dir = tmp_path / "tidal_out"
        proc = TidalProcessor(mock_config, input_dir, out_dir)
        result = proc.process()

        assert result.success
        assert result.metadata["mode"] == "copy"
        assert "test content" in (out_dir / "bctides.in").read_text()

    def test_template_mode(self, mock_config, tmp_path):
        """Template mode should update start time."""
        template = tmp_path / "bctides.in_template"
        template.write_text("01/01/2025 00:00:00\n0 1.0\n8\nM2\n0.1 1.0 0.0\n")

        mock_config.bctides_template = template
        out_dir = tmp_path / "tidal_out"
        proc = TidalProcessor(mock_config, tmp_path, out_dir)
        result = proc.process()

        assert result.success
        assert result.metadata["mode"] == "template"
        content = (out_dir / "bctides.in").read_text()
        # Start time should be updated to config date
        assert "2026" in content.split("\n")[0]

    def test_bctides_format(self, mock_config, tmp_path):
        """Verify bctides.in structure: date, ntip, nbfr, constituents."""
        out_dir = tmp_path / "tidal_out"
        proc = TidalProcessor(mock_config, tmp_path / "empty", out_dir)
        proc.process()

        lines = (out_dir / "bctides.in").read_text().strip().split("\n")

        # Line 0: date
        assert "/" in lines[0]  # DD/MM/YYYY format

        # Line 1: ntip tip_dp
        assert lines[1].strip().startswith("0")

        # Line 2: nbfr
        nbfr = int(lines[2].strip())
        assert nbfr == 8  # 8 constituents by default


class TestConstituentData:
    def test_all_constituents_have_omega(self):
        for name, props in TIDAL_CONSTITUENTS.items():
            assert "omega" in props
            assert props["omega"] > 0

    def test_m2_frequency(self):
        """M2 frequency should be ~28.984 deg/hr."""
        assert TIDAL_CONSTITUENTS["M2"]["omega"] == pytest.approx(28.984, abs=0.01)

    def test_s2_frequency(self):
        """S2 frequency should be exactly 30.0 deg/hr."""
        assert TIDAL_CONSTITUENTS["S2"]["omega"] == 30.0
