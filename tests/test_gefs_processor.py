"""Tests for GEFSProcessor."""

from pathlib import Path

import numpy as np
import pytest

from nos_utils.config import ForcingConfig
from nos_utils.forcing.gefs import GEFSProcessor


class TestGEFSFileDiscovery:
    def test_find_files_for_member(self, stofs_config, mock_gefs_dir):
        proc = GEFSProcessor(stofs_config, mock_gefs_dir, Path("/tmp/out"),
                            member="01")
        proc.MIN_FILE_SIZE = 0  # Mock files are tiny
        files = proc.find_input_files()
        assert len(files) > 0

    def test_find_files_for_control(self, stofs_config, mock_gefs_dir):
        proc = GEFSProcessor(stofs_config, mock_gefs_dir, Path("/tmp/out"),
                            member="c00")
        proc.MIN_FILE_SIZE = 0  # Mock files are tiny
        files = proc.find_input_files()
        assert len(files) > 0

    def test_file_prefix_perturbation(self):
        cfg = ForcingConfig.for_stofs_3d_atl(pdy="20260401", cyc=12)
        proc = GEFSProcessor(cfg, Path("/tmp"), Path("/tmp/out"), member="05")
        assert proc.file_prefix == "gep05"

    def test_file_prefix_control(self):
        cfg = ForcingConfig.for_stofs_3d_atl(pdy="20260401", cyc=12)
        proc = GEFSProcessor(cfg, Path("/tmp"), Path("/tmp/out"), member="c00")
        assert proc.file_prefix == "gec00"

    def test_file_product_derivation(self):
        cfg = ForcingConfig.for_stofs_3d_atl(pdy="20260401", cyc=12)

        proc1 = GEFSProcessor(cfg, Path("/tmp"), Path("/tmp/out"),
                              product="pgrb2sp25")
        assert proc1.file_product == "pgrb2s"

        proc2 = GEFSProcessor(cfg, Path("/tmp"), Path("/tmp/out"),
                              product="pgrb2ap5")
        assert proc2.file_product == "pgrb2a"


class TestGEFSConversions:
    def test_rh_to_spfh_known_values(self):
        """Verify RH->SPFH against known meteorological values."""
        # At T=293.15K (20°C), P=101325 Pa, RH=50%:
        # es ≈ 2338 Pa, e ≈ 1169 Pa, SPFH ≈ 0.0072 kg/kg
        rh = np.array([50.0], dtype=np.float32)
        temp = np.array([293.15], dtype=np.float32)
        pres = np.array([101325.0], dtype=np.float32)

        spfh = GEFSProcessor.convert_rh_to_spfh(rh, temp, pres)

        assert spfh.shape == (1,)
        assert 0.005 < spfh[0] < 0.010  # ~0.0072

    def test_rh_to_spfh_100_percent(self):
        """Saturated air should give higher SPFH."""
        rh_50 = np.array([50.0])
        rh_100 = np.array([100.0])
        temp = np.array([293.15])
        pres = np.array([101325.0])

        spfh_50 = GEFSProcessor.convert_rh_to_spfh(rh_50, temp, pres)
        spfh_100 = GEFSProcessor.convert_rh_to_spfh(rh_100, temp, pres)

        assert spfh_100[0] > spfh_50[0]
        assert spfh_100[0] == pytest.approx(2 * spfh_50[0], rel=0.01)

    def test_rh_to_spfh_clamps(self):
        """SPFH should be clamped to [0, 0.1]."""
        rh = np.array([0.0, 200.0])  # impossible values
        temp = np.array([293.15, 293.15])
        pres = np.array([101325.0, 101325.0])

        spfh = GEFSProcessor.convert_rh_to_spfh(rh, temp, pres)
        assert spfh[0] == 0.0
        assert spfh[1] <= 0.1

    def test_apcp_to_prate(self):
        """10mm accumulated over 3 hours = 10/10800 kg/m²/s."""
        apcp = np.array([10.0], dtype=np.float32)
        prate = GEFSProcessor.convert_apcp_to_prate(apcp)

        expected = 10.0 / 10800.0
        assert prate[0] == pytest.approx(expected, rel=1e-5)

    def test_apcp_to_prate_no_negative(self):
        """Negative precip should be clamped to 0."""
        apcp = np.array([-1.0], dtype=np.float32)
        prate = GEFSProcessor.convert_apcp_to_prate(apcp)
        assert prate[0] == 0.0

    def test_apcp_custom_dt(self):
        """Custom accumulation period."""
        apcp = np.array([10.0])
        prate = GEFSProcessor.convert_apcp_to_prate(apcp, dt_seconds=3600.0)
        assert prate[0] == pytest.approx(10.0 / 3600.0)


class TestGEFSMiscellaneous:
    def test_min_file_size(self):
        """GEFS files are ~15MB; threshold is 5MB."""
        assert GEFSProcessor.MIN_FILE_SIZE == 5_000_000

    def test_extract_variables_include_conversion_inputs(self):
        """EXTRACT_VARIABLES must include rh, apcp, pres for conversion."""
        assert "rh" in GEFSProcessor.EXTRACT_VARIABLES
        assert "apcp" in GEFSProcessor.EXTRACT_VARIABLES
        assert "pres" in GEFSProcessor.EXTRACT_VARIABLES

    def test_output_variables_have_converted_names(self):
        """OUTPUT_VARIABLES should have spfh and prate (not rh and apcp)."""
        assert "spfh" in GEFSProcessor.OUTPUT_VARIABLES
        assert "prate" in GEFSProcessor.OUTPUT_VARIABLES
        assert "rh" not in GEFSProcessor.OUTPUT_VARIABLES
        assert "apcp" not in GEFSProcessor.OUTPUT_VARIABLES
