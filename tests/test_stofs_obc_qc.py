"""Tests for the STOFS OBC dimension QC + COMOUT_PREV fallback."""

from pathlib import Path

import numpy as np
import pytest

nc_mod = pytest.importorskip("netCDF4")

from nos_utils.config import ForcingConfig  # noqa: E402
from nos_utils.orchestrator import PrepOrchestrator  # noqa: E402


def _write_obc_file(path: Path, nt: int, n_bnd: int = 4, var: str = "time_series",
                    is_3d: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with nc_mod.Dataset(str(path), "w") as ds:
        ds.createDimension("time", nt)
        ds.createDimension("nOpenBndNodes", n_bnd)
        ds.createDimension("nLevels", 3 if is_3d else 1)
        ds.createDimension("nComponents", 1)
        v = ds.createVariable(
            var, "f4",
            ("time", "nOpenBndNodes", "nLevels", "nComponents"),
        )
        v[:] = 0.0


def _make_orchestrator(tmp_path: Path, **paths) -> PrepOrchestrator:
    cfg = ForcingConfig.for_stofs_3d_atl(pdy="20260401", cyc=12)
    full_paths = {"output": str(tmp_path)}
    full_paths.update({k: str(v) for k, v in paths.items()})
    return PrepOrchestrator(cfg, paths=full_paths, run_name="stofs_3d_atl",
                             skip_legacy=True)


class TestOBCQC:
    def test_returns_none_when_all_files_satisfy_min(self, tmp_path):
        """When every file meets obc_min_timesteps (21), QC returns None."""
        output_dir = tmp_path / "out"
        # Default obc_min_timesteps=21 matches operational N_dim_cr_max.
        for name in ("elev2D.th.nc", "TEM_3D.th.nc", "SAL_3D.th.nc", "uv3D.th.nc"):
            _write_obc_file(output_dir / name, nt=25)

        orch = _make_orchestrator(tmp_path)
        result = orch._qc_obc_dimensions(output_dir)
        assert result is None

    def test_reports_short_files(self, tmp_path):
        """Short files without a fallback path -> failure ForcingResult."""
        output_dir = tmp_path / "out"
        # TEM_3D.th.nc is short (only 5 records); others meet the 21 threshold.
        _write_obc_file(output_dir / "elev2D.th.nc", nt=25)
        _write_obc_file(output_dir / "TEM_3D.th.nc", nt=5, is_3d=True)
        _write_obc_file(output_dir / "SAL_3D.th.nc", nt=25, is_3d=True)
        _write_obc_file(output_dir / "uv3D.th.nc", nt=25, is_3d=True)

        orch = _make_orchestrator(tmp_path)  # no prev_rerun
        result = orch._qc_obc_dimensions(output_dir)
        assert result is not None
        assert not result.success
        assert "TEM_3D.th.nc" in result.metadata["short_files"]
        assert any("5" in w for w in result.warnings)

    def test_reports_20_records_as_short(self, tmp_path):
        """Operational gate is >=21, so 20 should be flagged short."""
        output_dir = tmp_path / "out"
        _write_obc_file(output_dir / "elev2D.th.nc", nt=20)  # one below threshold
        _write_obc_file(output_dir / "TEM_3D.th.nc", nt=25, is_3d=True)
        _write_obc_file(output_dir / "SAL_3D.th.nc", nt=25, is_3d=True)
        _write_obc_file(output_dir / "uv3D.th.nc", nt=25, is_3d=True)

        orch = _make_orchestrator(tmp_path)
        result = orch._qc_obc_dimensions(output_dir)
        assert result is not None
        assert "elev2D.th.nc" in result.metadata["short_files"]

    def test_fallback_replaces_short_file(self, tmp_path):
        """When prev_rerun has a fallback, short files get replaced."""
        output_dir = tmp_path / "out"
        prev_dir = tmp_path / "prev_rerun"
        # Short current file:
        _write_obc_file(output_dir / "elev2D.th.nc", nt=5)
        _write_obc_file(output_dir / "TEM_3D.th.nc", nt=25, is_3d=True)
        _write_obc_file(output_dir / "SAL_3D.th.nc", nt=25, is_3d=True)
        _write_obc_file(output_dir / "uv3D.th.nc", nt=25, is_3d=True)
        # Well-sized fallback at the operational archive path.
        _write_obc_file(
            prev_dir / "stofs_3d_atl.t12z.elev2dth_non_adj.nc", nt=25,
        )

        orch = _make_orchestrator(tmp_path, prev_rerun=prev_dir)
        result = orch._qc_obc_dimensions(output_dir)
        assert result is not None
        assert result.success, result.errors
        assert len(result.output_files) == 1
        # The replaced file should now be the fallback's size (nt=25).
        with nc_mod.Dataset(str(output_dir / "elev2D.th.nc")) as ds:
            assert len(ds.dimensions["time"]) == 25

    def test_fallback_accepts_unprefixed_name(self, tmp_path):
        """Back-compat: QC also accepts the active filename as fallback."""
        output_dir = tmp_path / "out"
        prev_dir = tmp_path / "prev_rerun"
        _write_obc_file(output_dir / "elev2D.th.nc", nt=5)
        _write_obc_file(output_dir / "TEM_3D.th.nc", nt=25, is_3d=True)
        _write_obc_file(output_dir / "SAL_3D.th.nc", nt=25, is_3d=True)
        _write_obc_file(output_dir / "uv3D.th.nc", nt=25, is_3d=True)
        # Fallback under the active filename (not the operational std name).
        _write_obc_file(prev_dir / "elev2D.th.nc", nt=25)

        orch = _make_orchestrator(tmp_path, prev_rerun=prev_dir)
        result = orch._qc_obc_dimensions(output_dir)
        assert result is not None
        assert result.success
