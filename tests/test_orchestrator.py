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
    """time_hotstart must be derived as cycle - nowcast_hours, regardless of
    whether _run_hotstart selected a restart file or not.

    Regression suite for the bug where PrepOrchestrator.run() was parsing
    time_hotstart from the selected hotstart filename's ``tHHz.YYYYMMDD``
    tag.  That tag encodes the cycle that produced the restart, not the
    restart's time origin; when today's own pre-staged init file gets
    selected, the parse returns cycle time and the launcher's sim_start
    misaligns with the OBC time axis by LEN_NOWCAST hours -- surfacing as
    SCHISM partition_hgrid:534 ParMETIS heap corruption at 2914-rank scale.
    """

    def test_time_hotstart_equals_cycle_minus_nowcast_warm(self, mock_config,
                                                            orch_paths):
        """Even if _run_hotstart selected today's pre-staged init file,
        the marker must anchor to cycle - nowcast_hours.

        mock_config has pdy=20260401, cyc=12, nowcast_hours=6 ->
        expected time_hotstart = 2026040106.
        """
        orch = PrepOrchestrator(mock_config, orch_paths)
        result = orch.run(phase="nowcast")
        assert result.success

        marker = Path(orch_paths["output"]) / "time_hotstart.t12z"
        assert marker.is_file(), "_write_time_markers must emit time_hotstart"
        assert marker.read_text().strip() == "2026040106"

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
        marker value -- the YAML nowcast_hours anchor is authoritative.

        Upstream J-jobs sometimes export a 24h-back convention; tolerating
        that here would re-introduce the OBC/sim_start misalignment.
        """
        monkeypatch.setenv("time_hotstart", "2026033112")  # 24h-back ALL wrong
        orch = PrepOrchestrator(mock_config, orch_paths)
        orch.run(phase="nowcast")

        marker = Path(orch_paths["output"]) / "time_hotstart.t12z"
        # Still cycle - nowcast_hours, not the env value.
        assert marker.read_text().strip() == "2026040106"


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
