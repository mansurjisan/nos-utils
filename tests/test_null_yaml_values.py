"""An explicitly-null YAML value must not take the config load down.

Systems author `key: null` to say "not applicable here" -- Pacific and Alaska
both do it for `ssh_offset` (no datum shift for those domains) and for
`nudging.timescale_days` (relaxation strength is a spatial field in the
*_nudge.gr3 files, not a scalar), and Pacific does it for the UFS rank layout
because its validated engine is standalone.

That is a *present* key holding None, which neither `d.get(key, default)` nor
`key in d` protects against -- so these used to raise TypeError out of
`from_yaml` and fail the whole system, not just the one field.
"""
import pytest

from nos_utils.config import ForcingConfig

yaml = pytest.importorskip("yaml")


_BASE = """\
system:
  name: {name}
  prefix: {name}
grid:
  domain: {{lon_min: 156.0, lon_max: 204.0, lat_min: 48.5, lat_max: 67.0}}
  files: {{horizontal: {name}.hgrid.gr3}}
model:
  physics: {{dt: 45.0}}
  run: {{nowcast_hours: 6, forecast_hours: 48}}
"""


def _cfg(tmp_path, extra, name="nulltest"):
    path = tmp_path / "cfg.yaml"
    path.write_text(_BASE.format(name=name) + extra)
    return ForcingConfig.from_yaml(path)


class TestNullScalarsAreTolerated:
    def test_null_ssh_offset_falls_back_to_zero(self, tmp_path):
        cfg = _cfg(tmp_path, "forcing:\n  ocean:\n    obc:\n      ssh_offset: null\n")
        assert cfg.obc_ssh_offset == 0.0

    def test_real_ssh_offset_still_read(self, tmp_path):
        cfg = _cfg(tmp_path, "forcing:\n  ocean:\n    obc:\n      ssh_offset: 0.04\n")
        assert cfg.obc_ssh_offset == 0.04

    def test_null_nudging_timescale_falls_back_to_default(self, tmp_path):
        cfg = _cfg(
            tmp_path,
            "forcing:\n  ocean:\n    nudging:\n      enabled: true\n"
            "      timescale_days: null\n",
        )
        assert cfg.nudging_timescale_seconds == 86400.0

    def test_timescale_days_still_converted(self, tmp_path):
        cfg = _cfg(
            tmp_path,
            "forcing:\n  ocean:\n    nudging:\n      enabled: true\n"
            "      timescale_days: 0.125\n",
        )
        assert cfg.nudging_timescale_seconds == 10800.0

    def test_timescale_seconds_wins_over_days(self, tmp_path):
        cfg = _cfg(
            tmp_path,
            "forcing:\n  ocean:\n    nudging:\n      enabled: true\n"
            "      timescale_seconds: 3600\n      timescale_days: 5\n",
        )
        assert cfg.nudging_timescale_seconds == 3600.0


class TestNullUfsRankLayoutIsTolerated:
    # A system with no UFS layout yet (standalone-first, e.g. STOFS-3D-PAC)
    # authors these as null rather than omitting them, to record that the
    # decision is pending rather than forgotten.

    def test_null_task_counts_are_skipped_not_converted(self, tmp_path):
        cfg = _cfg(
            tmp_path,
            "ufs_coastal:\n  enabled: true\n  datm_tasks: 120\n"
            "  schism_tasks: null\n  total_tasks: null\n",
        )
        assert cfg.ufs_datm_tasks == 120
        # Left at the dataclass defaults rather than crashing on int(None).
        assert cfg.ufs_schism_tasks == 1080
        assert cfg.ufs_total_tasks == 1200

    def test_pinned_task_counts_are_read(self, tmp_path):
        cfg = _cfg(
            tmp_path,
            "ufs_coastal:\n  enabled: true\n  datm_tasks: 120\n"
            "  schism_tasks: 2393\n  total_tasks: 2513\n",
        )
        assert cfg.ufs_schism_tasks == 2393
        assert cfg.ufs_total_tasks == 2513

    def test_null_blend_resolution_is_skipped(self, tmp_path):
        cfg = _cfg(tmp_path, "ufs_coastal:\n  enabled: true\n  blend_resolution: null\n")
        assert cfg.datm_dx == 0.025
