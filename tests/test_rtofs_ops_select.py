"""Ops RTOFS file selection: same-day cycle first, previous cycle fills gaps, n-file dating."""

from datetime import datetime

import pytest

from nos_utils.config import ForcingConfig
from nos_utils.forcing.rtofs import RTOFSProcessor

PDY = "20260927"
H = 3600


@pytest.fixture(autouse=True)
def _no_size_floor(monkeypatch):
    monkeypatch.setattr(RTOFSProcessor, "MIN_FILE_SIZE_2D", 0)
    monkeypatch.setattr(RTOFSProcessor, "MIN_FILE_SIZE_3D", 0)


def _stage(root, date, n2, f2, n3, f3):
    d = root / f"rtofs.{date}"
    d.mkdir(parents=True, exist_ok=True)
    for h in n2:
        (d / f"rtofs_glo_2ds_n{h:03d}_diag.nc").touch()
    for h in f2:
        (d / f"rtofs_glo_2ds_f{h:03d}_diag.nc").touch()
    for h in n3:
        (d / f"rtofs_glo_3dz_n{h:03d}_6hrly_hvr_US_east.nc").touch()
    for h in f3:
        (d / f"rtofs_glo_3dz_f{h:03d}_6hrly_hvr_US_east.nc").touch()


def _ops_day(root, date=PDY):
    _stage(root, date, (12, 18), (0, *range(6, 121, 6)), (12, 18, 24), range(6, 121, 6))


def _proc(root, **kw):
    cfg = ForcingConfig.for_stofs_3d_atl(pdy=PDY, cyc=12)
    for k, v in kw.items():
        setattr(cfg, k, v)
    return RTOFSProcessor(cfg, root, root, phase="forecast")


def _hours(p, files):
    t0 = datetime(2026, 9, 26, 12)
    return [(p.valid_time(f) - t0).total_seconds() / H for f in files]


def test_same_day_cycle_gives_the_ops_series_in_6h_steps(tmp_path):
    _ops_day(tmp_path)
    _stage(tmp_path, "20260926", (), range(12, 145, 6), (), range(12, 145, 6))
    p = _proc(tmp_path)
    f2, f3 = p.find_input_files_by_type()
    assert [f.name for f in f2[:4]] == ["rtofs_glo_2ds_n012_diag.nc", "rtofs_glo_2ds_n018_diag.nc",
                                        "rtofs_glo_2ds_f000_diag.nc", "rtofs_glo_2ds_f006_diag.nc"]
    assert [f.name.split("_")[3] for f in f3[:4]] == ["n012", "n018", "n024", "f006"]
    assert len(f2) == len(f3) == 23
    assert _hours(p, f2) == _hours(p, f3) == [6.0 * k for k in range(23)]
    assert all(f.parent.name == f"rtofs.{PDY}" for f in f2 + f3)


def test_previous_cycle_only_fills_missing_slots(tmp_path):
    _ops_day(tmp_path)
    (tmp_path / f"rtofs.{PDY}" / "rtofs_glo_2ds_f030_diag.nc").unlink()
    _stage(tmp_path, "20260926", (), range(12, 145, 6), (), range(12, 145, 6))
    p = _proc(tmp_path)
    f2, f3 = p.find_input_files_by_type()
    assert len(f2) == 23 and _hours(p, f2) == [6.0 * k for k in range(23)]
    fill = [f for f in f2 if f.parent.name == "rtofs.20260926"]
    assert [f.name for f in fill] == ["rtofs_glo_2ds_f054_diag.nc"]  # 09-26 00z + 54 h = 09-28 06z = slot 11
    assert all(f.parent.name == f"rtofs.{PDY}" for f in f3)
    assert any("filled from the previous cycle" in m for m in p._select_notes)


def test_previous_cycle_alone_covers_the_series(tmp_path):
    _stage(tmp_path, "20260926", (), range(12, 145, 6), (), range(12, 145, 6))
    p = _proc(tmp_path)
    f2, f3 = p.find_input_files_by_type()
    assert _hours(p, f2)[0] == 0.0 and _hours(p, f2)[:23] == [6.0 * k for k in range(23)]
    assert f2[0].name == "rtofs_glo_2ds_f012_diag.nc"


def test_slot_count_covers_nowcast_plus_forecast_plus_buffer(tmp_path):
    assert len(_proc(tmp_path)._ops_slot_times()) == 23
    p = _proc(tmp_path, nowcast_hours=6, forecast_hours=48)
    assert len(p._ops_slot_times()) == 23


def test_n_files_are_dated_from_cycle_minus_24h():
    cfg = ForcingConfig.for_stofs_3d_atl(pdy=PDY, cyc=12)
    p = RTOFSProcessor(cfg, "/nonexistent", "/nonexistent")
    from pathlib import Path
    n12, f12 = Path("rtofs_glo_2ds_n012_diag.nc"), Path("rtofs_glo_2ds_f012_diag.nc")
    assert p.valid_time(n12) == datetime(2026, 9, 26, 12) and p.valid_time(f12) == datetime(2026, 9, 27, 12)


def test_secofs_dating_and_selection_are_unchanged(tmp_path):
    from pathlib import Path
    cfg = ForcingConfig.for_secofs(pdy=PDY, cyc=12)
    p = RTOFSProcessor(cfg, tmp_path, tmp_path)
    n12, f12 = Path("rtofs_glo_2ds_n012_diag.nc"), Path("rtofs_glo_2ds_f012_diag.nc")
    assert p.valid_time(n12) == datetime(2026, 9, 27, 12)  # cycle + 12 h, as before
    assert p._sort_and_dedup([n12, f12], datetime(2026, 9, 27)) == [f12]  # duplicate, forecast kept
    _stage(tmp_path, "20260926", (12,), (6, 12), (12,), (6, 12))
    _stage(tmp_path, PDY, (12,), (6, 12), (12,), (6, 12))
    f2, _ = p.find_input_files_by_type()
    assert all(f.parent.name == "rtofs.20260926" for f in f2)  # previous cycle searched first


def test_ops_timeline_flag_is_atl_only(tmp_path):
    from .test_stofs_obc_bilinear import _yaml_cfg
    assert ForcingConfig.for_stofs_3d_atl(PDY, 12).obc_ops_timeline is True
    assert ForcingConfig.for_stofs_3d_atl_ufs(PDY, 12).obc_ops_timeline is True
    assert ForcingConfig.for_stofs_3d_pac(PDY, 12).obc_ops_timeline is False
    assert ForcingConfig.for_secofs(PDY, 12).obc_ops_timeline is False
    assert _yaml_cfg(tmp_path, "stofs_3d_atl_ufs").obc_ops_timeline is True
    assert _yaml_cfg(tmp_path, "stofs_3d_pac_ufs").obc_ops_timeline is False
    assert _yaml_cfg(tmp_path, "stofs_3d_atl_ufs", "      ops_timeline: false\n").obc_ops_timeline is False
