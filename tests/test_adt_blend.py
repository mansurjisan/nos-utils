"""ADT-blended SSH must reach elev2D.th.nc in the Python fallback."""

from datetime import datetime

import numpy as np
import pytest

from nos_utils.config import ForcingConfig

pytest.importorskip("netCDF4")
pytest.importorskip("scipy")
from netCDF4 import Dataset  # noqa: E402

from nos_utils.forcing.adt import ADTBlender  # noqa: E402
from nos_utils.forcing.rtofs import RTOFSProcessor  # noqa: E402

HOURS = [0, 6, 12, 18, 24]


def _cfg():
    cfg = ForcingConfig.for_stofs_3d_atl(pdy="20260401", cyc=12)
    cfg.obc_roi_2d = {"x1": 0, "x2": 29, "y1": 0, "y2": 29}
    return cfg


def _write_2d(tmp_path, cfg, bnd_lons, bnd_lats):
    d = tmp_path / "rtofs_2d"
    d.mkdir()
    lo = np.linspace(bnd_lons.min() - 2.0, bnd_lons.max() + 2.0, 30)
    la = np.linspace(bnd_lats.min() - 2.0, bnd_lats.max() + 2.0, 30)
    lon2d, lat2d = np.meshgrid(lo, la)
    files = []
    for h in HOURS:
        f = d / f"rtofs_glo_2ds_f{h:03d}_diag.nc"
        with Dataset(str(f), "w") as ds:
            ds.createDimension("time", 1)
            ds.createDimension("Y", 30)
            ds.createDimension("X", 30)
            ds.createVariable("Longitude", "f4", ("Y", "X"))[:] = lon2d % 360.0  # RTOFS is 0-360
            ds.createVariable("Latitude", "f4", ("Y", "X"))[:] = lat2d
            # spatially uniform so blend - raw is an exact constant
            ds.createVariable("ssh", "f4", ("time", "Y", "X"))[0] = 0.05 * h
        files.append(f)
    return files


def _write_adt(path, fn):
    lon = np.arange(-100.0, -49.9, 0.5)
    lat = np.arange(5.0, 55.1, 0.5)
    with Dataset(str(path), "w") as ds:
        ds.createDimension("time", 1)
        ds.createDimension("latitude", lat.size)
        ds.createDimension("longitude", lon.size)
        ds.createVariable("longitude", "f8", ("longitude",))[:] = lon
        ds.createVariable("latitude", "f8", ("latitude",))[:] = lat
        LA, LO = np.meshgrid(lat, lon, indexing="ij")
        ds.createVariable("adt", "f4", ("time", "latitude", "longitude"))[0] = fn(LO, LA)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.delenv("COMINadt", raising=False)
    monkeypatch.delenv("DCOMROOT", raising=False)
    cfg = _cfg()
    out = tmp_path / "out"
    out.mkdir()
    proc = RTOFSProcessor(cfg, tmp_path, out)
    proc._bnd_lons = np.linspace(cfg.lon_min + 3.0, cfg.lon_min + 5.0, 8)
    proc._bnd_lats = np.linspace(cfg.lat_min + 3.0, cfg.lat_min + 5.0, 8)
    proc._rtofs_cycle_date = datetime.strptime(cfg.pdy, "%Y%m%d")
    files = _write_2d(tmp_path, cfg, proc._bnd_lons, proc._bnd_lats)
    work = tmp_path / "work"
    work.mkdir()
    ssh_1 = proc._stofs_prepare_ssh(files, work)
    return cfg, proc, files, ssh_1, work, tmp_path


def _elev(path):
    with Dataset(str(path)) as ds:
        return np.array(ds.variables["time_series"][:])[:, :, 0, 0]


def _blend(cfg, tmp_path, ssh_1, work, fn):
    _write_adt(tmp_path / "adt_20260401.nc", fn)
    return ADTBlender(cfg, tmp_path).blend_ssh(ssh_1, work)


def test_blended_ssh_reaches_elev2d_with_offsets_once(setup):
    cfg, proc, files, ssh_1, work, tmp = setup
    raw = _elev(proc._process_2d(files))
    blended = _blend(cfg, tmp, ssh_1, work, lambda lo, la: np.full_like(lo, 0.90))
    assert blended is not None
    got = _elev(proc._process_2d(files, ssh_source=blended))
    assert got.shape == raw.shape
    # (ADT - 0.45) replaces the t0 SSH: SSH(t) - SSH(t0) + ADT - 0.45, vs raw SSH(t)
    # obc_ssh_offset (0.04) is common to both, so the difference is exactly 0.90-0.45
    np.testing.assert_allclose(got - raw, 0.90 - 0.45, atol=2e-5)


def test_regrid_path_varies_in_space(setup):
    cfg, proc, files, ssh_1, work, tmp = setup
    blended = _blend(cfg, tmp, ssh_1, work, lambda lo, la: 0.9 + 0.02 * (lo + 80.0))
    got = _elev(proc._process_2d(files, ssh_source=blended))
    assert np.ptp(got[0]) > 1e-3  # a mean-only correction would be uniform
    with Dataset(str(blended)) as ds:
        lon = np.array(ds["xlon"][:])
        ssh0 = np.array(ds["ssh"][0])
    np.testing.assert_allclose(ssh0, 0.9 + 0.02 * (lon + 80.0) - 0.45, atol=1e-5)


def test_surf_el_packed_once(setup):
    cfg, proc, files, ssh_1, work, tmp = setup
    blended = _blend(cfg, tmp, ssh_1, work, lambda lo, la: np.full_like(lo, 0.90))
    with Dataset(str(blended)) as ds:
        ds.set_auto_maskandscale(False)
        np.testing.assert_allclose(np.array(ds["surf_el"][0]), 450.0, atol=0.1)
    with Dataset(str(ssh_1)) as ds:
        ds.set_auto_maskandscale(False)
        np.testing.assert_allclose(np.array(ds["surf_el"][2]), 600.0, atol=0.1)


def test_regrid_bilinear_exact_and_nan_renormalized():
    lons = np.arange(0.0, 10.0, 1.0)
    lats = np.arange(0.0, 10.0, 1.0)
    LA, LO = np.meshgrid(lats, lons, indexing="ij")
    f = 1.0 + 2.0 * LO - 0.5 * LA
    f[0, :] = np.nan  # a land row
    dl, da = np.array([[3.3, 7.7]]), np.array([[4.1, 5.9]])
    out = ADTBlender._regrid_bilinear(f, lons, lats, dl, da)
    np.testing.assert_allclose(out, 1.0 + 2.0 * dl - 0.5 * da, atol=1e-9)
    near_land = ADTBlender._regrid_bilinear(f, lons, lats, np.array([[3.3]]), np.array([[0.4]]))
    assert np.isfinite(near_land).all()
    outside = ADTBlender._regrid_bilinear(f, lons, lats, np.array([[50.0]]), np.array([[5.0]]))
    assert np.isnan(outside).all()


def test_two_day_adt_is_averaged(setup):
    cfg, proc, files, ssh_1, work, tmp = setup
    _write_adt(tmp / "adt_20260331.nc", lambda lo, la: np.full_like(lo, 1.10))
    blended = _blend(cfg, tmp, ssh_1, work, lambda lo, la: np.full_like(lo, 0.90))
    with Dataset(str(blended)) as ds:
        np.testing.assert_allclose(np.array(ds["ssh"][0]), 1.00 - 0.45, atol=1e-5)


def test_adt_inputs_reach_manifest(setup):
    from nos_utils.forcing._log import (
        drain_input_capture, reset_input_capture, start_input_capture,
    )
    cfg, proc, files, ssh_1, work, tmp = setup
    _write_adt(tmp / "adt_20260331.nc", lambda lo, la: np.full_like(lo, 1.10))
    reset_input_capture()
    start_input_capture()
    try:
        _blend(cfg, tmp, ssh_1, work, lambda lo, la: np.full_like(lo, 0.90))
        entries = {(e["category"], e["source"]): e for e in drain_input_capture()}
    finally:
        reset_input_capture()
    adt = entries[("ocean", "ADT")]
    assert adt["count"] == 2
    assert sorted(f.rsplit("/", 1)[-1] for f in adt["files"]) == [
        "adt_20260331.nc", "adt_20260401.nc"]


def test_adt_unavailable_equals_raw(setup):
    cfg, proc, files, ssh_1, work, tmp = setup
    assert ADTBlender(cfg, tmp).blend_ssh(ssh_1, work) is None
    raw = _elev(proc._process_2d(files))
    np.testing.assert_allclose(raw[0], raw[0, 0], atol=1e-6)
    # unchanged raw path: elev = ssh(file hour) + obc_ssh_offset
    assert np.isclose(raw.min(), 0.04, atol=1e-5) or raw.min() >= 0.04 - 1e-5
    again = _elev(proc._process_2d(files, ssh_source=None))
    np.testing.assert_array_equal(raw, again)


def test_bad_blended_length_falls_back_to_raw(setup):
    cfg, proc, files, ssh_1, work, tmp = setup
    blended = _blend(cfg, tmp, ssh_1, work, lambda lo, la: np.full_like(lo, 0.90))
    raw = _elev(proc._process_2d(files))
    got = _elev(proc._process_2d(files[:-1], ssh_source=blended))
    assert got.shape[1] == raw.shape[1]  # fell back (no exception), raw SSH from 4 files


@pytest.mark.parametrize("adt_on", [True, False])
def test_process_stofs_wiring(setup, monkeypatch, adt_on):
    cfg, proc, files, ssh_1, work, tmp = setup
    cfg.adt_enabled = adt_on
    cfg.rtofs_3d_region = None
    seen = {}
    monkeypatch.setattr(RTOFSProcessor, "find_input_files_by_type", lambda self: (files, []))
    monkeypatch.setattr(RTOFSProcessor, "_call_fortran_gen_3dth", lambda *a, **k: False)
    monkeypatch.setattr(RTOFSProcessor, "_load_grid", lambda self: True)
    if adt_on:
        _write_adt(tmp / "adt_20260401.nc", lambda lo, la: np.full_like(lo, 0.90))

    orig = RTOFSProcessor._process_2d

    def spy(self, f, ssh_source=None):
        seen["src"] = ssh_source
        return orig(self, f, ssh_source=ssh_source)

    monkeypatch.setattr(RTOFSProcessor, "_process_2d", spy)
    res = proc._process_stofs()
    assert res.success
    assert res.metadata["adt_blended"] is adt_on
    assert (seen["src"] is not None) is adt_on


def _write_ssh1(path, lons, lats, nt=2):
    LO, LA = np.meshgrid(lons, lats)
    with Dataset(str(path), "w") as ds:
        ds.createDimension("time", nt)
        ds.createDimension("ylat", len(lats))
        ds.createDimension("xlon", len(lons))
        ds.createVariable("xlon", "f4", ("ylat", "xlon"))[:] = LO
        ds.createVariable("ylat", "f4", ("ylat", "xlon"))[:] = LA
        v = ds.createVariable("ssh", "f4", ("time", "ylat", "xlon"), fill_value=-30000.0)
        for t in range(nt):
            v[t] = 0.1 * t
    return path


def test_adt_subset_follows_destination_grid_east_of_config_bounds(setup):
    cfg, proc, files, ssh_1, work, tmp = setup
    lons = np.linspace(-56.0, -51.6, 12)  # config lon_max is -52.4867
    assert lons.max() > cfg.lon_max
    dst = _write_ssh1(tmp / "ssh_dst.nc", lons, np.linspace(30.0, 32.0, 4))
    blended = _blend(cfg, tmp, dst, work, lambda lo, la: np.full_like(lo, 0.90))
    assert blended is not None
    with Dataset(str(blended)) as ds:
        ssh0 = np.array(ds["ssh"][0])
    assert np.isfinite(ssh0[:, lons > cfg.lon_max]).all()
    np.testing.assert_allclose(ssh0, 0.90 - 0.45, atol=1e-5)


def test_unmapped_wet_cell_gets_nearest_valid_adt(setup):
    cfg, proc, files, ssh_1, work, tmp = setup
    lons = np.array([-60.0, -58.0, -56.0])
    lats = np.array([30.0, 31.0, 32.0])
    dst = _write_ssh1(tmp / "ssh_dst.nc", lons, lats)

    def fn(lo, la):
        out = 0.9 + 0.05 * (lo + 60.0)
        out[(np.abs(lo + 58.0) < 2.1) & (np.abs(la - 31.0) < 2.1)] = np.nan  # 4x land
        return out

    blended = _blend(cfg, tmp, dst, work, fn)
    with Dataset(str(blended)) as ds:
        got = float(np.array(ds["ssh"][0])[1, 1])
    # nearest valid ADT cell to (-58, 31) is at lon -60.5 or -55.5 (2.5/3.0 away) or
    # lat 28.5/33.5; the lon -60.5, lat 31 cell is closest
    assert np.isfinite(got)
    np.testing.assert_allclose(got, 0.9 + 0.05 * (-60.5 + 60.0) - 0.45, atol=1e-5)


def test_step_count_mismatch_reports_raw_and_warns(setup, monkeypatch):
    cfg, proc, files, ssh_1, work, tmp = setup
    cfg.adt_enabled = True
    cfg.rtofs_3d_region = None
    monkeypatch.setattr(RTOFSProcessor, "find_input_files_by_type", lambda self: (files, []))
    monkeypatch.setattr(RTOFSProcessor, "_call_fortran_gen_3dth", lambda *a, **k: False)
    monkeypatch.setattr(RTOFSProcessor, "_load_grid", lambda self: True)
    _write_adt(tmp / "adt_20260401.nc", lambda lo, la: np.full_like(lo, 0.90))
    orig = RTOFSProcessor._process_2d
    monkeypatch.setattr(RTOFSProcessor, "_process_2d",
                        lambda self, f, ssh_source=None: orig(self, f[:-1], ssh_source=ssh_source))
    res = proc._process_stofs()
    assert res.success
    assert res.metadata["adt_blended"] is False
    assert any("raw RTOFS SSH" in w for w in res.warnings)


@pytest.mark.parametrize("src_idx,expect_blend", [(0, False), (5, True)])
def test_precomputed_weights_nan_guard(setup, monkeypatch, src_idx, expect_blend):
    cfg, proc, files, ssh_1, work, tmp = setup
    blended = _blend(cfg, tmp, ssh_1, work, lambda lo, la: np.full_like(lo, 0.90))
    with Dataset(str(blended), "r+") as ds:
        ds["ssh"][:, 0, 0] = -30000.0  # becomes NaN at flat index 0
    seen = []
    n_bnd = len(proc._bnd_lons)

    def stub(w, field):
        seen.append(bool(np.isnan(field.ravel()[w["source_data_flat_idx"]]).any()))
        return np.zeros(n_bnd)

    monkeypatch.setattr("nos_utils.interp.precomputed_weights.apply_precomputed_ssh", stub)
    monkeypatch.setattr(proc, "_find_ssh_weights",
                        lambda: {"source_data_flat_idx": np.array([src_idx])})
    proc._process_2d(files, ssh_source=blended)
    assert proc._ssh_blended_used is expect_blend
    assert not any(seen)


def test_fortran_exe_reads_blended_ssh(setup, monkeypatch):
    cfg, proc, files, ssh_1, work, tmp = setup
    blended = _blend(cfg, tmp, ssh_1, work, lambda lo, la: np.full_like(lo, 0.90))
    assert blended is not None and ssh_1.exists() and not ssh_1.is_symlink()
    exe_dir = tmp / "exec"
    exe_dir.mkdir()
    exe = exe_dir / "stofs_3d_atl_gen_3Dth_from_hycom"
    exe.write_text("#!/bin/sh\ncp SSH_1.nc seen.nc\n")
    exe.chmod(0o755)
    for v in ("EXECstofs3d", "EXECofs", "FIXstofs3d"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("EXECnos", str(exe_dir))
    proc._call_fortran_gen_3dth(work, blended, None)
    assert (work / "SSH_1.nc").resolve() == blended.resolve()
    with Dataset(str(work / "seen.nc")) as a, Dataset(str(blended)) as b:
        np.testing.assert_array_equal(np.array(a["ssh"][:]), np.array(b["ssh"][:]))


def test_fortran_link_keeps_raw_ssh_without_adt(setup, monkeypatch):
    cfg, proc, files, ssh_1, work, tmp = setup
    exe_dir = tmp / "exec"
    exe_dir.mkdir()
    exe = exe_dir / "stofs_3d_atl_gen_3Dth_from_hycom"
    exe.write_text("#!/bin/sh\nexit 0\n")
    exe.chmod(0o755)
    for v in ("EXECstofs3d", "EXECofs", "FIXstofs3d"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("EXECnos", str(exe_dir))
    proc._call_fortran_gen_3dth(work, ssh_1, None)
    assert ssh_1.exists() and not ssh_1.is_symlink()
