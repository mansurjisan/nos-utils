"""Tests for ParamNmlProcessor."""

from pathlib import Path

import pytest

from nos_utils.config import ForcingConfig
from nos_utils.forcing.param_nml import ParamNmlProcessor


@pytest.fixture
def param_template(tmp_path):
    """Create a mock param.nml template."""
    template = tmp_path / "fix" / "param.nml"
    template.parent.mkdir()
    template.write_text(
        "&CORE\n"
        "  rnday = rnday_value  !total run time in days\n"
        "  dt = 120.\n"
        "  ihot = 1\n"
        "  nws = 4\n"
        "/\n"
        "&OPT\n"
        "  start_year = start_year_value\n"
        "  start_month = start_month_value\n"
        "  start_day = start_day_value\n"
        "  start_hour = start_hour_value\n"
        "/\n"
    )
    return template


class TestParamNmlProcessor:
    def test_nowcast_substitution(self, mock_config, param_template, tmp_path):
        out_dir = tmp_path / "out"
        proc = ParamNmlProcessor(
            mock_config, param_template.parent, out_dir, phase="nowcast",
        )
        result = proc.process()

        assert result.success
        content = (out_dir / "param.nml").read_text()

        # 6h nowcast -> rnday = 0.25
        assert "0.2500" in content
        # Start time: 2026-04-01 12z - 6h = 06z
        assert "start_year = 2026" in content
        assert "start_month = 04" in content
        assert "start_day = 01" in content
        assert "start_hour = 6.0" in content
        # Placeholders should be gone
        assert "rnday_value" not in content
        assert "start_year_value" not in content

    def test_forecast_substitution(self, mock_config, param_template, tmp_path):
        out_dir = tmp_path / "out"
        proc = ParamNmlProcessor(
            mock_config, param_template.parent, out_dir, phase="forecast",
        )
        result = proc.process()

        assert result.success
        content = (out_dir / "param.nml").read_text()

        # 48h forecast -> rnday = 2.0
        assert "2.0000" in content
        # Start time: 2026-04-01 12z
        assert "start_hour = 12.0" in content

    def test_missing_template(self, mock_config, tmp_path):
        proc = ParamNmlProcessor(
            mock_config, tmp_path / "nonexistent", tmp_path / "out",
        )
        result = proc.process()
        assert not result.success
        assert "Template not found" in result.errors[0]

    def test_metadata(self, mock_config, param_template, tmp_path):
        out_dir = tmp_path / "out"
        proc = ParamNmlProcessor(
            mock_config, param_template.parent, out_dir, phase="nowcast",
        )
        result = proc.process()

        assert result.metadata["phase"] == "nowcast"
        assert result.metadata["rnday"] == "0.2500"
        assert result.metadata["ihot"] == 1


class TestPatchParam:
    def test_patch_rnday(self, tmp_path):
        param = tmp_path / "param.nml"
        param.write_text("  rnday = 0.25  !run time\n  dt = 120.\n")

        ParamNmlProcessor.patch_param(param, rnday=2.0)
        content = param.read_text()
        assert "rnday = 2.0" in content
        assert "dt = 120." in content  # unchanged

    def test_patch_multiple(self, tmp_path):
        param = tmp_path / "param.nml"
        param.write_text("  rnday = 0.25\n  ihot = 0\n  nws = 2\n")

        ParamNmlProcessor.patch_param(param, rnday=5.0, ihot=1, nws=4)
        content = param.read_text()
        assert "rnday = 5.0" in content
        assert "ihot = 1" in content
        assert "nws = 4" in content
