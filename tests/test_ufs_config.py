"""Tests for UFSConfigProcessor (Stage 2b)."""

from datetime import datetime
from pathlib import Path

import pytest

from nos_utils.config import ForcingConfig
from nos_utils.forcing.ufs_config import UFSConfigProcessor


# ---------- shared template fixtures ---------- #

_MODEL_CONFIGURE_TEMPLATE = (
    "start_year:              @[YYYY]\n"
    "start_month:             @[MM]\n"
    "start_day:               @[DD]\n"
    "start_hour:              @[HH]\n"
    "nhours_fcst:             @[NHOURS]\n"
    "dt_atmos:                @[DT_ATMOS]\n"
)

_DATM_IN_TEMPLATE = (
    "&datm_nml\n"
    '  model_maskfile = "@[DATM_INPUT_DIR]/@[DATM_MESH_FILE]"\n'
    '  model_meshfile = "@[DATM_INPUT_DIR]/@[DATM_MESH_FILE]"\n'
    "  nx_global = @[NX_GLOBAL]\n"
    "  ny_global = @[NY_GLOBAL]\n"
    "/\n"
)

_DATM_STREAMS_TEMPLATE = (
    "stream_info:               atm.01\n"
    "yearFirst01:               @[YYYY]\n"
    "yearLast01:                @[YYYY]\n"
    "yearAlign01:               @[YYYY]\n"
    'stream_mesh_file01:        "@[DATM_INPUT_DIR]/@[DATM_MESH_FILE]"\n'
    'stream_data_files01:       "@[DATM_INPUT_DIR]/@[DATM_FORCING_FILE]"\n'
)

_UFS_CONFIGURE = (
    "# MED #\n"
    "MED_model:                      cmeps\n"
    "MED_petlist_bounds:             0 119\n"
    "MED_omp_num_threads:            1\n"
    "\n"
    "# ATM #\n"
    "ATM_model:                      datm\n"
    "ATM_petlist_bounds:             0 119\n"
    "ATM_omp_num_threads:            1\n"
    "\n"
    "# OCN #\n"
    "OCN_model:                      schism\n"
    "OCN_petlist_bounds:             120 1199\n"
    "OCN_omp_num_threads:            1\n"
)


def _write_full_fix(fix_dir: Path, *, include_optional: bool = True) -> None:
    fix_dir.mkdir(parents=True, exist_ok=True)
    (fix_dir / "model_configure.template").write_text(_MODEL_CONFIGURE_TEMPLATE)
    (fix_dir / "datm_in.template").write_text(_DATM_IN_TEMPLATE)
    (fix_dir / "datm.streams.template").write_text(_DATM_STREAMS_TEMPLATE)
    (fix_dir / "ufs.configure").write_text(_UFS_CONFIGURE)
    if include_optional:
        (fix_dir / "fd_ufs.yaml").write_text("# fake fd_ufs\n")
        (fix_dir / "noahmptable.tbl").write_text("# fake noahmptable\n")


@pytest.fixture
def ufs_config():
    """SECOFS-UFS ForcingConfig with the resource layout we care about."""
    return ForcingConfig.for_secofs_ufs(pdy="20260401", cyc=12)


# ---------- tests ---------- #


def test_basic_substitution(ufs_config, tmp_path):
    """All @[TOKEN] markers must be replaced with real values.

    Start time is anchored at ``model_t0 = cycle - nowcast_hours``.  For
    SECOFS-UFS (cyc=12, nowcast_hours=6) that is 2026-04-01 06:00, not the
    raw cycle time.  ``nhours_fcst`` correspondingly covers the full
    nowcast + forecast = 54h window from model_t0.
    """
    fix = tmp_path / "fix"
    out = tmp_path / "out"
    _write_full_fix(fix)

    proc = UFSConfigProcessor(ufs_config, fix, out)
    result = proc.process()

    assert result.success, result.errors
    # Every required output file should exist.
    for name in (
        "model_configure", "datm_in", "datm.streams",
        "ufs.configure", "fd_ufs.yaml", "noahmptable.tbl",
    ):
        assert (out / name).exists(), f"Missing output: {name}"

    # model_configure substitutions.  model_t0 = cycle 12z - 6h = 06z same day.
    mc = (out / "model_configure").read_text()
    assert "start_year:              2026" in mc
    assert "start_month:             04" in mc
    assert "start_day:               01" in mc
    assert "start_hour:              06" in mc
    # nhours_fcst covers nowcast_hours + forecast_hours = 6 + 48 = 54.
    assert "nhours_fcst:             54" in mc
    assert "dt_atmos:                720" in mc
    assert "@[" not in mc, "Unsubstituted token in model_configure"

    # datm_in substitutions.
    di = (out / "datm_in").read_text()
    assert "INPUT/datm_esmf_mesh.nc" in di
    assert "@[" not in di, "Unsubstituted token in datm_in"

    # datm.streams substitutions.
    ds = (out / "datm.streams").read_text()
    assert "yearFirst01:               2026" in ds
    assert "yearLast01:                2026" in ds
    assert "INPUT/datm_forcing.nc" in ds
    assert "@[" not in ds, "Unsubstituted token in datm.streams"


def test_pet_patching(tmp_path):
    """ufs.configure PET bounds must be patched per resource layout."""
    fix = tmp_path / "fix"
    out = tmp_path / "out"
    _write_full_fix(fix)

    # Use SECOFS V15 layout (datm=120, total=1200) for a deterministic check.
    cfg = ForcingConfig.for_secofs_ufs(
        pdy="20260401", cyc=12,
        ufs_datm_tasks=120, ufs_schism_tasks=1080, ufs_total_tasks=1200,
    )

    proc = UFSConfigProcessor(cfg, fix, out)
    result = proc.process()
    assert result.success, result.errors

    uc = (out / "ufs.configure").read_text()
    # MED + ATM cover 0..119; OCN covers 120..1199.
    assert "MED_petlist_bounds:             0 119" in uc
    assert "ATM_petlist_bounds:             0 119" in uc
    assert "OCN_petlist_bounds:             120 1199" in uc

    # Now exercise the v3.9 mesh layout: OCN must extend to 2913.
    cfg_v39 = ForcingConfig.for_secofs_ufs(
        pdy="20260401", cyc=12,
        ufs_datm_tasks=120, ufs_schism_tasks=2794, ufs_total_tasks=2914,
    )
    out2 = tmp_path / "out2"
    proc2 = UFSConfigProcessor(cfg_v39, fix, out2)
    result2 = proc2.process()
    assert result2.success, result2.errors

    uc2 = (out2 / "ufs.configure").read_text()
    assert "MED_petlist_bounds:             0 119" in uc2
    assert "ATM_petlist_bounds:             0 119" in uc2
    assert "OCN_petlist_bounds:             120 2913" in uc2
    # Old hardcoded value must be gone.
    assert "OCN_petlist_bounds:             120 1199" not in uc2


def test_missing_template_fails(ufs_config, tmp_path):
    """Missing required templates must yield a failure result, not a crash."""
    fix = tmp_path / "fix"
    out = tmp_path / "out"
    fix.mkdir()
    # Intentionally do NOT write model_configure.template.
    (fix / "datm_in.template").write_text(_DATM_IN_TEMPLATE)
    (fix / "datm.streams.template").write_text(_DATM_STREAMS_TEMPLATE)
    (fix / "ufs.configure").write_text(_UFS_CONFIGURE)

    proc = UFSConfigProcessor(ufs_config, fix, out)
    result = proc.process()

    assert not result.success
    assert any("model_configure.template" in e for e in result.errors)
    # No output files should have been emitted.
    assert not (out / "model_configure").exists()
    assert not (out / "datm_in").exists()


def test_auto_resolve_sibling_ufs_dir(ufs_config, tmp_path):
    """Templates in <name>_ufs sibling should be discovered automatically.

    Real layout on WCOSS2:
      fix/secofs/        <- SCHISM mesh files (caller's --fix)
      fix/secofs_ufs/    <- UFS templates (where we need to look)
    """
    # Caller passes fix/secofs (no templates here)
    fix_root = tmp_path / "fix"
    fix_secofs = fix_root / "secofs"
    fix_secofs.mkdir(parents=True)
    (fix_secofs / "secofs_hgrid.gr3").write_text("# stub mesh\n")  # SCHISM stuff

    # Sibling fix/secofs_ufs has the templates
    fix_secofs_ufs = fix_root / "secofs_ufs"
    _write_full_fix(fix_secofs_ufs)

    out = tmp_path / "out"
    proc = UFSConfigProcessor(ufs_config, fix_secofs, out)
    result = proc.process()

    assert result.success, result.errors
    assert (out / "model_configure").exists()
    assert (out / "ufs.configure").exists()


def test_model_t0_day_rollback(tmp_path):
    """model_t0 = cycle - nowcast_hours must roll back across midnight.

    SECOFS 00z cycle with nowcast_hours=6 anchors at 18z the previous day.
    The model_configure start_* fields and the datm.streams year tokens
    must reflect that rolled-back date, otherwise CMEPS/DATM stream
    indexing would start 6h ahead of SCHISM.
    """
    fix = tmp_path / "fix"
    out = tmp_path / "out"
    _write_full_fix(fix)

    cfg = ForcingConfig.for_secofs_ufs(pdy="20260510", cyc=0)
    proc = UFSConfigProcessor(cfg, fix, out)
    result = proc.process()
    assert result.success, result.errors

    mc = (out / "model_configure").read_text()
    # cycle 2026-05-10 00z - 6h = 2026-05-09 18z
    assert "start_year:              2026" in mc
    assert "start_month:             05" in mc
    assert "start_day:               09" in mc
    assert "start_hour:              18" in mc
    # nhours_fcst still covers the full 6 + 48 = 54h window.
    assert "nhours_fcst:             54" in mc

    # datm.streams year tokens must come from the rolled-back model_t0
    # (still 2026, but if the cycle were on Jan 1 we would need 2025).
    ds = (out / "datm.streams").read_text()
    assert "yearFirst01:               2026" in ds


def test_model_t0_year_rollback(tmp_path):
    """model_t0 must roll back across a year boundary cleanly.

    Cycle on 2026-01-01 00z with nowcast_hours=6 anchors at 2025-12-31 18z.
    The datm.streams ``yearFirst/Last/Align`` tokens must reflect 2025,
    not 2026, otherwise stream timestamps would mismatch the actual DATM
    forcing time axis.
    """
    fix = tmp_path / "fix"
    out = tmp_path / "out"
    _write_full_fix(fix)

    cfg = ForcingConfig.for_secofs_ufs(pdy="20260101", cyc=0)
    proc = UFSConfigProcessor(cfg, fix, out)
    result = proc.process()
    assert result.success, result.errors

    mc = (out / "model_configure").read_text()
    assert "start_year:              2025" in mc
    assert "start_month:             12" in mc
    assert "start_day:               31" in mc
    assert "start_hour:              18" in mc

    ds = (out / "datm.streams").read_text()
    assert "yearFirst01:               2025" in ds
    assert "yearLast01:                2025" in ds
    assert "yearAlign01:               2025" in ds


def test_time_hotstart_override(tmp_path):
    """Caller-provided time_hotstart wins over the derived cycle - nowcast.

    Orchestrators that have already pinned the model_t0 anchor (e.g. from
    a ``time_hotstart`` marker file or a hotstart NetCDF) can pass that
    datetime in to keep every component of the prep bundle in lockstep.
    """
    fix = tmp_path / "fix"
    out = tmp_path / "out"
    _write_full_fix(fix)

    cfg = ForcingConfig.for_secofs_ufs(pdy="20260510", cyc=12)
    # Pin model_t0 to 2026-05-10 09z (3h back from cycle, not the default 6h).
    pinned = datetime(2026, 5, 10, 9, 0)
    proc = UFSConfigProcessor(cfg, fix, out, time_hotstart=pinned)
    result = proc.process()
    assert result.success, result.errors

    mc = (out / "model_configure").read_text()
    assert "start_year:              2026" in mc
    assert "start_month:             05" in mc
    assert "start_day:               10" in mc
    assert "start_hour:              09" in mc


def test_stofs_ufs_nhours_covers_full_run(tmp_path):
    """STOFS-UFS factory (132 = 24+108) anchors at model_t0 too.

    Verifies that the long-window STOFS-3D-ATL UFS layout (nowcast=24,
    forecast=108) keeps producing nhours_fcst=132 with the new anchor;
    132 already covers the full coupled run from model_t0 so the explicit
    factory value passes through unchanged.
    """
    fix = tmp_path / "fix"
    out = tmp_path / "out"
    _write_full_fix(fix)

    cfg = ForcingConfig.for_stofs_3d_atl_ufs(pdy="20260510", cyc=12)
    proc = UFSConfigProcessor(cfg, fix, out)
    result = proc.process()
    assert result.success, result.errors

    mc = (out / "model_configure").read_text()
    # cycle 2026-05-10 12z - 24h = 2026-05-09 12z
    assert "start_year:              2026" in mc
    assert "start_month:             05" in mc
    assert "start_day:               09" in mc
    assert "start_hour:              12" in mc
    # STOFS-UFS factory sets ufs_nhours_fcst=132 (= 24 + 108).
    assert "nhours_fcst:             132" in mc


def test_nhours_fcst_bumped_when_factory_value_too_short(tmp_path):
    """SECOFS factory's legacy ufs_nhours_fcst=48 must be bumped to 54.

    The factory still emits ``ufs_nhours_fcst=48`` (matches forecast_hours
    only) for backward compatibility.  With the model_t0 anchor we MUST
    extend that to ``nowcast_hours + forecast_hours = 54`` so SCHISM does
    not stop short of the forecast end.
    """
    fix = tmp_path / "fix"
    out = tmp_path / "out"
    _write_full_fix(fix)

    cfg = ForcingConfig.for_secofs_ufs(pdy="20260510", cyc=12)
    # Factory default is 48 (forecast-only).
    assert cfg.ufs_nhours_fcst == 48

    proc = UFSConfigProcessor(cfg, fix, out)
    result = proc.process()
    assert result.success, result.errors

    mc = (out / "model_configure").read_text()
    # Must be bumped up to 54 (full run from model_t0).
    assert "nhours_fcst:             54" in mc
    assert "nhours_fcst:             48" not in mc


def test_nx_ny_from_datm(ufs_config, tmp_path):
    """When a datm_forcing.nc is provided, NX/NY come from its dimensions."""
    pytest.importorskip("netCDF4")
    from netCDF4 import Dataset
    import numpy as np

    fix = tmp_path / "fix"
    out = tmp_path / "out"
    _write_full_fix(fix)

    # Build a minimal fake datm_forcing.nc with x=100, y=80.
    datm_path = tmp_path / "datm_forcing.nc"
    with Dataset(str(datm_path), "w") as ds:
        ds.createDimension("x", 100)
        ds.createDimension("y", 80)
        ds.createDimension("time", 1)
        lon = ds.createVariable("longitude", "f4", ("y", "x"))
        lat = ds.createVariable("latitude", "f4", ("y", "x"))
        lon[:] = np.zeros((80, 100), dtype=np.float32)
        lat[:] = np.zeros((80, 100), dtype=np.float32)

    proc = UFSConfigProcessor(
        ufs_config, fix, out, datm_forcing_path=datm_path,
    )
    result = proc.process()
    assert result.success, result.errors

    di = (out / "datm_in").read_text()
    assert "nx_global = 100" in di
    assert "ny_global = 80" in di

    # Metadata should reflect the same.
    assert result.metadata["nx_global"] == 100
    assert result.metadata["ny_global"] == 80
