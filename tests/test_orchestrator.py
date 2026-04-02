"""Tests for PrepOrchestrator."""

from pathlib import Path

import pytest

from nos_utils.config import ForcingConfig
from nos_utils.orchestrator import PrepOrchestrator, PrepResult


@pytest.fixture
def orch_paths(tmp_path):
    """Create minimal directory structure for orchestrator."""
    paths = {
        "output": str(tmp_path / "work"),
        "fix": str(tmp_path / "fix"),
    }
    # Create fix dir with a param.nml template
    fix_dir = tmp_path / "fix"
    fix_dir.mkdir()
    (fix_dir / "param.nml").write_text(
        "&CORE\n"
        "  rnday = rnday_value\n"
        "  dt = 120.\n"
        "/\n"
        "&OPT\n"
        "  start_year = start_year_value\n"
        "  start_month = start_month_value\n"
        "  start_day = start_day_value\n"
        "  start_hour = start_hour_value\n"
        "/\n"
    )
    return paths


class TestPrepOrchestrator:
    def test_minimal_run(self, mock_config, orch_paths):
        """Orchestrator with only fix dir should produce param.nml + tidal."""
        orch = PrepOrchestrator(mock_config, orch_paths)
        result = orch.run(phase="nowcast")

        assert isinstance(result, PrepResult)
        assert result.phase == "nowcast"
        assert result.elapsed_seconds > 0

        # Should have at least hotstart + tidal + param_nml results
        sources = [r.source for r in result.results]
        assert "HOTSTART" in sources
        assert "TIDAL" in sources
        assert "PARAM_NML" in sources

    def test_param_nml_created(self, mock_config, orch_paths, tmp_path):
        orch = PrepOrchestrator(mock_config, orch_paths)
        result = orch.run(phase="nowcast")

        param_file = Path(orch_paths["output"]) / "param.nml"
        assert param_file.exists()
        content = param_file.read_text()
        assert "rnday_value" not in content  # should be substituted

    def test_summary(self, mock_config, orch_paths):
        orch = PrepOrchestrator(mock_config, orch_paths)
        result = orch.run(phase="nowcast")

        summary = result.summary()
        assert "PrepResult" in summary
        assert "nowcast" in summary


class TestPrepResult:
    def test_all_output_files(self):
        from nos_utils.forcing.base import ForcingResult
        r = PrepResult(
            success=True, phase="nowcast",
            results=[
                ForcingResult(success=True, source="GFS",
                             output_files=[Path("/a.nc"), Path("/b.nc")]),
                ForcingResult(success=True, source="TIDAL",
                             output_files=[Path("/c.in")]),
            ],
        )
        assert len(r.all_output_files) == 3

    def test_all_errors(self):
        from nos_utils.forcing.base import ForcingResult
        r = PrepResult(
            success=False, phase="nowcast",
            results=[
                ForcingResult(success=False, source="GFS", errors=["no files"]),
                ForcingResult(success=True, source="TIDAL"),
            ],
        )
        assert r.all_errors == ["no files"]
