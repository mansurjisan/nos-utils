"""Tests for ForcingConfig."""

import pytest
from nos_utils.config import ForcingConfig


class TestForcingConfig:
    def test_basic_creation(self):
        cfg = ForcingConfig(
            lon_min=-80, lon_max=-70, lat_min=25, lat_max=35,
            pdy="20260401", cyc=12,
        )
        assert cfg.domain == (-80, -70, 25, 35)
        assert cfg.pdy == "20260401"
        assert cfg.cyc == 12
        assert cfg.nowcast_hours == 6
        assert cfg.forecast_hours == 48

    def test_secofs_factory(self):
        cfg = ForcingConfig.for_secofs(pdy="20260401", cyc=12)
        assert cfg.lon_min == -88.0
        assert cfg.lon_max == -63.0
        assert cfg.lat_min == 17.0
        assert cfg.lat_max == 40.0
        assert cfg.met_num == 2
        assert cfg.nowcast_hours == 6
        assert cfg.forecast_hours == 48

    def test_stofs_factory(self):
        cfg = ForcingConfig.for_stofs_3d_atl(pdy="20260401", cyc=12)
        assert cfg.lon_min == pytest.approx(-98.5035)
        assert cfg.nowcast_hours == 24
        assert cfg.forecast_hours == 108

    def test_factory_overrides(self):
        cfg = ForcingConfig.for_secofs(pdy="20260401", cyc=12, forecast_hours=120)
        assert cfg.forecast_hours == 120

    def test_invalid_domain_raises(self):
        with pytest.raises(ValueError, match="lon_min"):
            ForcingConfig(lon_min=-70, lon_max=-80, lat_min=25, lat_max=35,
                         pdy="20260401", cyc=12)

    def test_invalid_lat_raises(self):
        with pytest.raises(ValueError, match="lat_min"):
            ForcingConfig(lon_min=-80, lon_max=-70, lat_min=35, lat_max=25,
                         pdy="20260401", cyc=12)

    def test_grid_file_optional(self):
        """grid_file can be None — resolved later by nco_bridge from FIXofs."""
        cfg = ForcingConfig(lon_min=-80, lon_max=-70, lat_min=25, lat_max=35,
                           pdy="20260401", cyc=12, igrd_met=1)
        assert cfg.grid_file is None

    def test_defaults(self):
        cfg = ForcingConfig(lon_min=-80, lon_max=-70, lat_min=25, lat_max=35,
                           pdy="20260401", cyc=12)
        assert cfg.igrd_met == 0
        assert cfg.met_num == 1
        assert cfg.scale_hflux == 1.0
        assert cfg.nws == 2
        assert cfg.grid_file is None
        assert cfg.variables == []


# --- prep.extras declarative override (P1 prep-unification) ---------------
#
# `prep.extras` decouples "is this a STOFS-style YAML" (inferred from OBC
# ROI indices) from "should prep run St-Lawrence / dynamic SSH adjust" (an
# explicit operational choice). These tests guard:
#   1. present  -> flags equal the declared values
#   2. absent   -> identical to the pre-change is_stofs_like heuristic
#                  (regression guard: SECOFS-derived stays False/False/0)
#   3. drift    -> factory values == a YAML carrying the matching
#                  prep.extras (factory/YAML can't silently diverge)

# A SECOFS-shaped YAML: no OBC ROI indices -> is_stofs_like is False.
_SECOFS_SHAPED_YAML = """\
system:
  name: secofs_shaped
grid:
  domain: {lon_min: -88.0, lon_max: -63.0, lat_min: 17.0, lat_max: 40.0}
forcing:
  atmospheric: {met_num: 2}
model:
  run: {nowcast_hours: 6, forecast_hours: 48}
"""

# A STOFS-shaped YAML: a faithful miniature of parm/systems/
# stofs_3d_atl_ufs.yaml for the keys that feed the prep optionals.
# Pre-change, this yields st_lawrence_enabled=True (river.st_lawrence.
# enabled), dynamic_adjust_enabled=True (obc.obc_mode == dynamic_adjust),
# obc_min_timesteps=21 (is_stofs_like via OBC ROI indices) -> the
# True/True/21 STOFS baseline asserted by the regression guard.
_STOFS_SHAPED_YAML = """\
system:
  name: stofs_shaped
grid:
  domain: {lon_min: -98.5035, lon_max: -52.4867, lat_min: 7.347, lat_max: 52.5904}
forcing:
  atmospheric: {met_num: 2}
  river:
    st_lawrence: {enabled: true}
  ocean:
    obc:
      obc_mode: dynamic_adjust
      roi_2ds: {x1: 2805, x2: 2923, y1: 1598, y2: 2325}
      roi_3dz: {x1: 482, x2: 600, y1: 94, y2: 821}
model:
  run: {nowcast_hours: 24, forecast_hours: 108}
"""

# A bare STOFS-SHAPED YAML carrying ONLY the OBC ROI indices (no
# river.st_lawrence, no obc_mode). This isolates the is_stofs_like
# heuristic itself: pre-change it yields st_lawrence_enabled=False
# (the heuristic never sets St-Lawrence), dynamic_adjust_enabled=True
# and obc_min_timesteps=21 (both driven by is_stofs_like). Used to
# document/guard the exact heuristic contract `prep.extras` overrides.
_STOFS_ROI_ONLY_YAML = """\
system:
  name: stofs_roi_only
grid:
  domain: {lon_min: -98.5035, lon_max: -52.4867, lat_min: 7.347, lat_max: 52.5904}
forcing:
  atmospheric: {met_num: 2}
  ocean:
    obc:
      roi_2ds: {x1: 2805, x2: 2923, y1: 1598, y2: 2325}
      roi_3dz: {x1: 482, x2: 600, y1: 94, y2: 821}
model:
  run: {nowcast_hours: 24, forecast_hours: 108}
"""

_PREP_EXTRAS_BLOCK = """\
prep:
  extras:
    st_lawrence: {st_lawrence}
    obc_dynamic_adjust: {obc_dynamic_adjust}
"""


def _write_yaml(tmp_path, body, name="cfg.yaml"):
    p = tmp_path / name
    p.write_text(body)
    return p


class TestPrepExtras:
    # 1. prep.extras present -> flags equal the declared values, regardless
    #    of whether the YAML shape is SECOFS-like or STOFS-like.

    def test_present_forces_on_for_secofs_shaped(self, tmp_path):
        body = _SECOFS_SHAPED_YAML + _PREP_EXTRAS_BLOCK.format(
            st_lawrence="true", obc_dynamic_adjust="true")
        cfg = ForcingConfig.from_yaml(_write_yaml(tmp_path, body))
        assert cfg.st_lawrence_enabled is True
        assert cfg.dynamic_adjust_enabled is True
        assert cfg.obc_min_timesteps == 21

    def test_present_forces_off_for_stofs_shaped(self, tmp_path):
        # prep.extras must override the is_stofs_like heuristic, which
        # would otherwise have produced True/True/21 for this YAML shape.
        body = _STOFS_SHAPED_YAML + _PREP_EXTRAS_BLOCK.format(
            st_lawrence="false", obc_dynamic_adjust="false")
        cfg = ForcingConfig.from_yaml(_write_yaml(tmp_path, body))
        assert cfg.st_lawrence_enabled is False
        assert cfg.dynamic_adjust_enabled is False
        assert cfg.obc_min_timesteps == 0

    def test_present_mixed_flags(self, tmp_path):
        body = _STOFS_SHAPED_YAML + _PREP_EXTRAS_BLOCK.format(
            st_lawrence="false", obc_dynamic_adjust="true")
        cfg = ForcingConfig.from_yaml(_write_yaml(tmp_path, body))
        assert cfg.st_lawrence_enabled is False
        assert cfg.dynamic_adjust_enabled is True
        assert cfg.obc_min_timesteps == 21

    # 2. prep.extras absent -> identical to the pre-change heuristic.
    #    This is the zero-behavior-change regression guard.

    def test_absent_secofs_shaped_regression_guard(self, tmp_path):
        cfg = ForcingConfig.from_yaml(
            _write_yaml(tmp_path, _SECOFS_SHAPED_YAML))
        assert cfg.st_lawrence_enabled is False
        assert cfg.dynamic_adjust_enabled is False
        assert cfg.obc_min_timesteps == 0

    def test_absent_stofs_shaped_regression_guard(self, tmp_path):
        cfg = ForcingConfig.from_yaml(
            _write_yaml(tmp_path, _STOFS_SHAPED_YAML))
        assert cfg.st_lawrence_enabled is True
        assert cfg.dynamic_adjust_enabled is True
        assert cfg.obc_min_timesteps == 21

    def test_absent_roi_only_heuristic_baseline(self, tmp_path):
        # Documents the exact pre-change is_stofs_like contract that
        # `prep.extras` overrides: the heuristic drives dynamic_adjust
        # and obc_min_timesteps from the OBC ROI indices, but NEVER
        # touches st_lawrence_enabled (that comes only from
        # river.st_lawrence.enabled or a factory). A ROI-only YAML
        # therefore yields False/True/21, not True/True/21.
        cfg = ForcingConfig.from_yaml(
            _write_yaml(tmp_path, _STOFS_ROI_ONLY_YAML))
        assert cfg.st_lawrence_enabled is False
        assert cfg.dynamic_adjust_enabled is True
        assert cfg.obc_min_timesteps == 21

    def test_empty_extras_block_falls_back_to_heuristic(self, tmp_path):
        # `prep:` with an empty `extras:` must not override the heuristic.
        body = _STOFS_SHAPED_YAML + "prep:\n  extras: {}\n"
        cfg = ForcingConfig.from_yaml(_write_yaml(tmp_path, body))
        assert cfg.st_lawrence_enabled is True
        assert cfg.dynamic_adjust_enabled is True
        assert cfg.obc_min_timesteps == 21

    # 3. Factory <-> YAML drift guard: the UFS factories hardcode these
    #    flags; assert a YAML carrying the matching prep.extras yields the
    #    same three values so the two sources can't silently diverge.

    def test_secofs_ufs_factory_matches_yaml(self, tmp_path):
        factory = ForcingConfig.for_secofs_ufs(pdy="20260401", cyc=12)
        body = _SECOFS_SHAPED_YAML + _PREP_EXTRAS_BLOCK.format(
            st_lawrence="false", obc_dynamic_adjust="false")
        from_yaml = ForcingConfig.from_yaml(_write_yaml(tmp_path, body))
        assert factory.st_lawrence_enabled is False
        assert factory.dynamic_adjust_enabled is False
        assert factory.obc_min_timesteps == 0
        assert (
            from_yaml.st_lawrence_enabled,
            from_yaml.dynamic_adjust_enabled,
            from_yaml.obc_min_timesteps,
        ) == (
            factory.st_lawrence_enabled,
            factory.dynamic_adjust_enabled,
            factory.obc_min_timesteps,
        )

    def test_stofs_3d_atl_ufs_factory_matches_yaml(self, tmp_path):
        factory = ForcingConfig.for_stofs_3d_atl_ufs(pdy="20260401", cyc=12)
        body = _STOFS_SHAPED_YAML + _PREP_EXTRAS_BLOCK.format(
            st_lawrence="true", obc_dynamic_adjust="true")
        from_yaml = ForcingConfig.from_yaml(_write_yaml(tmp_path, body))
        assert factory.st_lawrence_enabled is True
        assert factory.dynamic_adjust_enabled is True
        assert factory.obc_min_timesteps == 21
        assert (
            from_yaml.st_lawrence_enabled,
            from_yaml.dynamic_adjust_enabled,
            from_yaml.obc_min_timesteps,
        ) == (
            factory.st_lawrence_enabled,
            factory.dynamic_adjust_enabled,
            factory.obc_min_timesteps,
        )


class TestNCOBridgePaths:
    """config_from_env() maps NCO env vars into the paths dict.

    Regression guard for the STOFS-3D-ATL bug where COMINlaw / COMINadt
    were never threaded into paths, so the orchestrator silently skipped
    St. Lawrence forcing and reported ADT missing.
    """

    def _base_env(self, monkeypatch, tmp_path):
        monkeypatch.setenv("PDY", "20260401")
        monkeypatch.setenv("cyc", "12")
        monkeypatch.setenv("RUN", "stofs_3d_atl")
        monkeypatch.delenv("OFS_CONFIG", raising=False)
        # Required for hotstart/output resolution; harmless here.
        monkeypatch.setenv("DATA", str(tmp_path / "data"))

    def test_cominlaw_maps_to_law_path(self, monkeypatch, tmp_path):
        from nos_utils.nco_bridge import config_from_env

        self._base_env(monkeypatch, tmp_path)
        law_dir = tmp_path / "dcom_law"
        monkeypatch.setenv("COMINlaw", str(law_dir))

        _, paths = config_from_env()
        assert paths.get("law") == str(law_dir)

    def test_cominadt_maps_to_adt_path(self, monkeypatch, tmp_path):
        from nos_utils.nco_bridge import config_from_env

        self._base_env(monkeypatch, tmp_path)
        adt_dir = tmp_path / "dcom_adt"
        monkeypatch.setenv("COMINadt", str(adt_dir))

        _, paths = config_from_env()
        assert paths.get("adt") == str(adt_dir)

    def test_law_adt_absent_when_unset(self, monkeypatch, tmp_path):
        from nos_utils.nco_bridge import config_from_env

        self._base_env(monkeypatch, tmp_path)
        monkeypatch.delenv("COMINlaw", raising=False)
        monkeypatch.delenv("COMINadt", raising=False)

        _, paths = config_from_env()
        assert "law" not in paths
        assert "adt" not in paths


# A STOFS-shaped YAML that, like the real stofs_3d_atl_ufs.yaml, declares
# model.physics.nws=4 (UFS NUOPC coupling). Used to prove execution.mode
# is the single source of truth: standalone forces nws=2 over this value,
# while ufs/absent leaves model.physics.nws untouched.
_NWS4_YAML = """\
system:
  name: execmode_nws
grid:
  domain: {lon_min: -98.5035, lon_max: -52.4867, lat_min: 7.347, lat_max: 52.5904}
forcing:
  atmospheric: {met_num: 2}
model:
  physics: {nws: 4}
  run: {nowcast_hours: 24, forecast_hours: 108}
"""

_EXEC_BLOCK = "execution:\n  mode: {mode}\n"


class TestExecutionModeNws:
    """``execution.mode`` is the single source of truth for ``nws``.

    Phase 1 parity contract on the nos-utils side: standalone forces
    nws=2 regardless of model.physics.nws; ufs/absent must leave the
    existing ``int(physics.get("nws", 2))`` path exactly as-is.
    """

    def test_standalone_forces_nws_2_over_physics(self, tmp_path):
        body = _NWS4_YAML + _EXEC_BLOCK.format(mode="standalone")
        cfg = ForcingConfig.from_yaml(_write_yaml(tmp_path, body))
        assert cfg.nws == 2

    def test_mode_ufs_leaves_physics_nws_untouched(self, tmp_path):
        body = _NWS4_YAML + _EXEC_BLOCK.format(mode="ufs")
        cfg = ForcingConfig.from_yaml(_write_yaml(tmp_path, body))
        assert cfg.nws == 4

    def test_mode_absent_leaves_physics_nws_untouched(self, tmp_path):
        # Zero-behavior-change regression guard for the default path.
        cfg = ForcingConfig.from_yaml(_write_yaml(tmp_path, _NWS4_YAML))
        assert cfg.nws == 4

    def test_real_stofs_ufs_yaml_nws_unchanged_in_ufs_mode(self):
        # The shipped stofs_3d_atl_ufs.yaml carries execution.mode: ufs
        # and model.physics.nws: 4 -> from_yaml must still yield nws=4.
        from pathlib import Path

        yaml_path = (
            Path(__file__).resolve().parents[4]
            / "parm" / "systems" / "stofs_3d_atl_ufs.yaml"
        )
        if not yaml_path.exists():
            pytest.skip("stofs_3d_atl_ufs.yaml not found")
        cfg = ForcingConfig.from_yaml(yaml_path, pdy="20260401", cyc=12)
        assert cfg.nws == 4
