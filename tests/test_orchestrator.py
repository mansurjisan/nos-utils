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


class TestTimeHotstartAnchor:
    """time_hotstart must equal cycle time (Route A), regardless of whether
    _run_hotstart selected a restart file or not.

    The OBC NetCDFs, DATM forcing, sflux stack, and model_configure all anchor
    t=0 at cycle time.  SCHISM's param.nml start_*, bctides.in line 1, and
    the $COMOUT time markers must agree with that anchor or the model crashes
    at partition_hgrid:534 with ParMETIS heap corruption.

    The previous anchor (cycle - nowcast_hours, commit c0e232c) only migrated
    the SCHISM-side files and left the OBC/forcing side on cycle time,
    producing the exact misalignment Route A is designed to avoid.
    """

    def test_time_hotstart_equals_cycle_warm(self, mock_config, orch_paths):
        """time_hotstart marker must anchor to cycle time.

        mock_config has pdy=20260401, cyc=12 ->
        expected time_hotstart = 2026040112.
        """
        orch = PrepOrchestrator(mock_config, orch_paths)
        result = orch.run(phase="nowcast")
        assert result.success

        marker = Path(orch_paths["output"]) / "time_hotstart.t12z"
        assert marker.is_file(), "_write_time_markers must emit time_hotstart"
        assert marker.read_text().strip() == "2026040112"

    def test_base_date_matches_time_hotstart(self, mock_config, orch_paths):
        """base_date.${cycle} must byte-match time_hotstart.${cycle} -- the
        existing _write_time_markers contract."""
        orch = PrepOrchestrator(mock_config, orch_paths)
        orch.run(phase="nowcast")

        th = (Path(orch_paths["output"]) / "time_hotstart.t12z").read_text()
        bd = (Path(orch_paths["output"]) / "base_date.t12z").read_text()
        assert th == bd

    def test_time_nowcastend_is_cycle(self, mock_config, orch_paths):
        """Sanity-check the other marker: time_nowcastend == cycle time.
        mock_config pdy=20260401, cyc=12 -> 2026040112.
        """
        orch = PrepOrchestrator(mock_config, orch_paths)
        orch.run(phase="nowcast")

        marker = Path(orch_paths["output"]) / "time_nowcastend.t12z"
        assert marker.read_text().strip() == "2026040112"

    def test_env_time_hotstart_ignored(self, mock_config, orch_paths,
                                       monkeypatch):
        """An environment override of $time_hotstart must NOT change the
        marker value -- the YAML-derived cycle anchor is authoritative.

        Upstream J-jobs sometimes export a 24h-back convention; tolerating
        that here would re-introduce SCHISM/forcing time-axis misalignment.
        """
        monkeypatch.setenv("time_hotstart", "2026033112")  # ALL wrong
        orch = PrepOrchestrator(mock_config, orch_paths)
        orch.run(phase="nowcast")

        marker = Path(orch_paths["output"]) / "time_hotstart.t12z"
        # Still cycle time, not the env value.
        assert marker.read_text().strip() == "2026040112"


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
