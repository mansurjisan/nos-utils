"""Declarative `prep.critical_sources` -- which forcing failures fail a prep.

Before this was declarable, criticality came from a substring test on the run
name (``"stofs" in run_name``), under which a SECOFS cycle with no RTOFS and no
NWM reported SUCCESS while writing climatology rivers and no OBC.
"""
import textwrap

import pytest

from nos_utils.config import ForcingConfig
from nos_utils.orchestrator import PrepOrchestrator


def _write_yaml(tmp_path, body: str):
    p = tmp_path / "sys.yaml"
    p.write_text(textwrap.dedent(body))
    return p


class TestConfigParsing:
    def test_absent_leaves_none(self, tmp_path):
        """No declaration => None => orchestrator keeps the legacy heuristic."""
        p = _write_yaml(tmp_path, """
            system:
              name: demo
            prep:
              extras:
                st_lawrence: false
            """)
        cfg = ForcingConfig.from_yaml(str(p), pdy="20260724", cyc=0)
        assert cfg.critical_sources is None

    def test_declared_list_is_parsed(self, tmp_path):
        p = _write_yaml(tmp_path, """
            system:
              name: demo
            prep:
              critical_sources: [GFS, PARAM_NML, TIDAL, RTOFS, NWM]
            """)
        cfg = ForcingConfig.from_yaml(str(p), pdy="20260724", cyc=0)
        assert cfg.critical_sources == [
            "GFS", "PARAM_NML", "TIDAL", "RTOFS", "NWM",
        ]

    def test_names_are_normalised_to_upper(self, tmp_path):
        p = _write_yaml(tmp_path, """
            system:
              name: demo
            prep:
              critical_sources: [gfs, rtofs]
            """)
        cfg = ForcingConfig.from_yaml(str(p), pdy="20260724", cyc=0)
        assert cfg.critical_sources == ["GFS", "RTOFS"]

    def test_empty_list_is_preserved(self, tmp_path):
        """An explicit empty list means 'nothing is critical' -- distinct from
        an absent key, which means 'use the heuristic'."""
        p = _write_yaml(tmp_path, """
            system:
              name: demo
            prep:
              critical_sources: []
            """)
        cfg = ForcingConfig.from_yaml(str(p), pdy="20260724", cyc=0)
        assert cfg.critical_sources == []


@pytest.fixture
def prep_paths(tmp_path):
    """Minimal paths for a real PrepOrchestrator.run(): no gfs/rtofs/nwm, so
    those stages are never created at all."""
    fix = tmp_path / "fix"
    fix.mkdir()
    (fix / "param.nml").write_text(
        "&CORE\n  rnday = rnday_value\n  dt = 120.\n/\n"
        "&OPT\n  start_year = start_year_value\n"
        "  start_month = start_month_value\n"
        "  start_day = start_day_value\n"
        "  start_hour = start_hour_value\n/\n"
    )
    return {"output": str(tmp_path / "work"), "fix": str(fix)}


class TestOrchestratorRun:
    """Drives the real PrepOrchestrator.run().

    These deliberately do NOT reimplement the selection logic: doing so is how
    the never-ran hole below survived its first round of tests.
    """

    def _run(self, config, paths, run_name="secofs_ufs"):
        return PrepOrchestrator(config, paths, run_name=run_name).run(phase="nowcast")

    def test_declared_source_that_never_ran_fails_the_prep(
        self, mock_config, prep_paths
    ):
        """The dominant failure on a new platform: COMINrtofs/COMINnwm unset,
        so no stage is created and there is no failed result to notice."""
        mock_config.critical_sources = ["TIDAL", "RTOFS", "NWM"]
        result = self._run(mock_config, prep_paths)

        ran = {r.source for r in result.results}
        assert "RTOFS" not in ran and "NWM" not in ran
        assert result.success is False

    def test_declared_sources_all_present_succeeds(self, mock_config, prep_paths):
        """Only sources that actually run are declared -> prep passes."""
        mock_config.critical_sources = ["TIDAL", "PARAM_NML"]
        result = self._run(mock_config, prep_paths)

        ran = {r.source for r in result.results}
        assert {"TIDAL", "PARAM_NML"} <= ran
        assert result.success is True

    def test_empty_declaration_means_nothing_is_critical(self, mock_config, prep_paths):
        mock_config.critical_sources = []
        assert self._run(mock_config, prep_paths).success is True

    def test_undeclared_keeps_legacy_heuristic(self, mock_config, prep_paths):
        """Configs that have not opted in must not start failing: the legacy
        heuristic keeps its never-ran blind spot on purpose, so declaring
        prep.critical_sources is the single switch that tightens it."""
        mock_config.critical_sources = None
        result = self._run(mock_config, prep_paths, run_name="secofs_ufs")

        ran = {r.source for r in result.results}
        assert "GFS" not in ran           # GFS is 'critical' under the heuristic
        assert result.success is True     # ...but never-ran is not checked there

    def test_declared_beats_heuristic_on_a_stofs_name(self, mock_config, prep_paths):
        """A STOFS-named run with a narrow declaration must use the
        declaration, not the run-name heuristic's wider set."""
        mock_config.critical_sources = ["TIDAL"]
        assert self._run(mock_config, prep_paths, run_name="stofs_3d_atl").success is True


# NOTE: the guard asserting that the *shipped* nos-workflow YAMLs declare
# RTOFS/NWM critical lives in nos_workflow's suite (test_system_yaml_contract),
# not here -- nos-utils is a separate repo and must not depend on the parent
# checkout's parm/ tree.
