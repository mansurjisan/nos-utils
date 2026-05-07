"""Tests for UFSConfigProcessor (Stage 2b)."""

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
    """All @[TOKEN] markers must be replaced with real values."""
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

    # model_configure substitutions.
    mc = (out / "model_configure").read_text()
    assert "start_year:              2026" in mc
    assert "start_month:             04" in mc
    assert "start_day:               01" in mc
    assert "start_hour:              12" in mc
    # Default ufs_nhours_fcst for SECOFS UFS is 48.
    assert "nhours_fcst:             48" in mc
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
