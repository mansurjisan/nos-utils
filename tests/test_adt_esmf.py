"""ADT regrid with the ops ESMF map, KEEPDATA intermediates."""

import numpy as np
import pytest

pytest.importorskip("netCDF4")
from netCDF4 import Dataset  # noqa: E402

from nos_utils.config import ForcingConfig  # noqa: E402
from nos_utils.forcing.adt import ADTBlender  # noqa: E402
from nos_utils.forcing.rtofs import RTOFSProcessor  # noqa: E402

SLON = np.array([-63.0, -62.5, -62.0, -61.5, -61.0])  # ROI keeps -62.5..-61.0 (inclusive)
SLAT = np.array([6.5, 7.0, 7.5, 8.0])  # ROI keeps 7.0..8.0
DLON = np.array([-62.4, -61.6])
DLAT = np.array([7.2, 7.8])
# ROI cells flatten (lat, lon): 3 lats x 4 lons = 12
ROW = np.array([1, 1, 2, 3, 3, 4, 4])
COL = np.array([1, 2, 3, 4, 5, 6, 7])
S = np.array([0.5, 0.5, 1.0, 0.25, 0.75, 0.5, 0.5])


def _adt_file(path, vals, lon=SLON, lat=SLAT):
    """vals: dict {(ilat_in_roi, ilon_in_roi): value}; the rest 0.9 m, NaN = missing."""
    a = np.full((lat.size, lon.size), 0.9)
    for (j, i), v in vals.items():
        a[j + 1, i + 1] = v
    with Dataset(str(path), "w") as ds:
        ds.createDimension("time", 1)
        ds.createDimension("latitude", lat.size)
        ds.createDimension("longitude", lon.size)
        ds.createVariable("longitude", "f8", ("longitude",))[:] = lon
        ds.createVariable("latitude", "f8", ("latitude",))[:] = lat
        v = ds.createVariable("adt", "f4", ("time", "latitude", "longitude"), fill_value=-32767.0)
        v[0] = np.ma.masked_invalid(a)


def _map(path, n_a=12, n_b=4, dst_lon=DLON, dst_lat=DLAT, xc_b=None):
    LO, LA = np.meshgrid(SLON[1:], SLAT[1:])
    LB, LT = np.meshgrid(dst_lon, dst_lat)
    with Dataset(str(path), "w") as ds:
        ds.createDimension("n_a", n_a)
        ds.createDimension("n_b", n_b)
        ds.createDimension("n_s", S.size)
        ds.createVariable("S", "f8", ("n_s",))[:] = S
        ds.createVariable("row", "i4", ("n_s",))[:] = ROW
        ds.createVariable("col", "i4", ("n_s",))[:] = COL
        ds.createVariable("xc_a", "f8", ("n_a",))[:] = np.resize(LO.ravel(), n_a)
        ds.createVariable("yc_a", "f8", ("n_a",))[:] = np.resize(LA.ravel(), n_a)
        ds.createVariable("xc_b", "f8", ("n_b",))[:] = np.resize(LB.ravel() if xc_b is None else xc_b, n_b)
        ds.createVariable("yc_b", "f8", ("n_b",))[:] = np.resize(LT.ravel(), n_b)
    return path


def _ssh1(path):
    LO, LA = np.meshgrid(DLON, DLAT)
    with Dataset(str(path), "w") as ds:
        ds.createDimension("time", 2)
        ds.createDimension("ylat", 2)
        ds.createDimension("xlon", 2)
        ds.createVariable("xlon", "f4", ("ylat", "xlon"))[:] = LO
        ds.createVariable("ylat", "f4", ("ylat", "xlon"))[:] = LA
        v = ds.createVariable("ssh", "f4", ("time", "ylat", "xlon"), fill_value=-30000.0)
        v[0] = 0.0
        v[1] = 0.1
    return path


def _cfg(wt):
    cfg = ForcingConfig.for_stofs_3d_atl(pdy="20260401", cyc=12)
    cfg.adt_weight_file = wt
    return cfg


def _run(tmp_path, cfg, day0, day1=None, keep=False):
    _adt_file(tmp_path / "adt_20260401.nc", day0)
    if day1 is not None:
        _adt_file(tmp_path / "adt_20260331.nc", day1)
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    ssh = _ssh1(tmp_path / "SSH_1.nc")
    b = ADTBlender(cfg, tmp_path, keep=keep)
    out = b.blend_ssh(ssh, work)
    return b, out, work


def _adt_of(out):
    with Dataset(str(out)) as ds:
        return np.ma.filled(ds["ssh"][0], np.nan)


def test_esmf_hand_computed_with_missing_and_two_day_mean(tmp_path, monkeypatch):
    monkeypatch.delenv("COMINadt", raising=False)
    monkeypatch.delenv("DCOMROOT", raising=False)
    cfg = _cfg(_map(tmp_path / "wt.nc"))
    # ROI cell k = j*4 + i (0-based); the map's col is 1-based. Unset cells are 0.9.
    d0 = {(0, 0): 1.0, (0, 1): np.nan, (0, 2): 0.5, (0, 3): 0.7}
    d1 = {(0, 0): 0.2, (0, 1): 0.4, (0, 2): np.nan, (0, 3): np.nan, (1, 0): 0.6}
    b, out, work = _run(tmp_path, cfg, d0, d1)
    assert b.regrid == "esmf" and out is not None
    # dst0 = .5*c0 + .5*c1 ; dst1 = c2 ; dst2 = .25*c3 + .75*c4 ; dst3 = .5*c5 + .5*c6
    # day0 dst0: c1 missing is skipped without renormalization -> 0.5*1.0
    # day1 dst1: only source missing -> day0 alone (ncra per-element mean)
    want = np.array([
        ((0.5 * 1.0 - 0.45) + (0.5 * 0.2 + 0.5 * 0.4 - 0.45)) / 2,
        0.5 - 0.45,
        ((0.25 * 0.7 + 0.75 * 0.9 - 0.45) + (0.75 * 0.6 - 0.45)) / 2,  # c3 missing on day1, no renormalization
        0.9 - 0.45,
    ])
    np.testing.assert_allclose(_adt_of(out).ravel(), want, atol=1e-6)


def test_esmf_dst_without_valid_source_is_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("COMINadt", raising=False)
    monkeypatch.delenv("DCOMROOT", raising=False)
    cfg = _cfg(_map(tmp_path / "wt.nc"))
    b, out, _ = _run(tmp_path, cfg, {(0, 2): np.nan})
    assert np.isnan(_adt_of(out).ravel()[1])


def test_apply_esmf_map_skips_missing_without_renormalization():
    src = np.array([np.nan, 1.0])
    out = ADTBlender._apply_esmf_map(np.array([0.5, 0.5]), np.array([0, 0]), np.array([0, 1]), 1, src)
    assert out[0] == pytest.approx(0.5)
    assert np.isnan(ADTBlender._apply_esmf_map(
        np.array([1.0]), np.array([0]), np.array([0]), 1, np.array([np.nan]))[0])


def test_nearest_map_copies_the_source_cell():
    src = np.array([0.1234567891, 2.0, 3.0])
    out = ADTBlender._apply_esmf_map(np.ones(3), np.array([2, 0, 1]), np.array([0, 1, 2]), 3, src)
    np.testing.assert_array_equal(out, [2.0, 3.0, 0.1234567891])


def test_days_are_rounded_to_float32_before_the_mean(tmp_path, monkeypatch):
    monkeypatch.delenv("COMINadt", raising=False)
    monkeypatch.delenv("DCOMROOT", raising=False)
    cfg = _cfg(_map(tmp_path / "wt.nc"))
    d0 = {(0, 2): 0.3333333333}
    d1 = {(0, 2): 0.7777777777}
    _adt_file(tmp_path / "adt_20260401.nc", d0)
    _adt_file(tmp_path / "adt_20260331.nc", d1)
    field = ADTBlender(cfg, tmp_path)._regrid_esmf(
        [tmp_path / "adt_20260401.nc", tmp_path / "adt_20260331.nc"], _ssh1(tmp_path / "SSH_1.nc"))
    assert field.dtype == np.float32
    f32 = np.float32
    a = f32(f32(0.3333333333).astype(np.float64) - 0.45)
    b = f32(f32(0.7777777777).astype(np.float64) - 0.45)
    want = f32((float(a) + float(b)) / 2)
    assert field.ravel()[1] == want


@pytest.mark.parametrize("bad", ["n_a", "n_b", "xc_b", "missing"])
def test_esmf_falls_back_to_bilinear_with_warning(tmp_path, monkeypatch, bad):
    monkeypatch.delenv("COMINadt", raising=False)
    monkeypatch.delenv("DCOMROOT", raising=False)
    if bad == "missing":
        wt = tmp_path / "absent.nc"
    elif bad == "n_a":
        wt = _map(tmp_path / "wt.nc", n_a=13)
    elif bad == "n_b":
        wt = _map(tmp_path / "wt.nc", n_b=5)
    else:
        wt = _map(tmp_path / "wt.nc", xc_b=np.array([-50.0, -61.6, -62.4, -61.6]))
    cfg = _cfg(wt)
    b, out, _ = _run(tmp_path, cfg, {})
    assert out is not None and b.regrid == "bilinear"
    assert any("not ops-exact" in w for w in b.warnings)
    np.testing.assert_allclose(_adt_of(out), 0.9 - 0.45, atol=1e-5)


def test_unconfigured_weight_warns_not_ops_exact(tmp_path, monkeypatch):
    monkeypatch.delenv("COMINadt", raising=False)
    monkeypatch.delenv("DCOMROOT", raising=False)
    b, out, _ = _run(tmp_path, _cfg(None), {})
    assert b.regrid == "bilinear" and any("not ops-exact" in w for w in b.warnings)


def test_keep_writes_adt_on_rtofs(tmp_path, monkeypatch):
    monkeypatch.delenv("COMINadt", raising=False)
    monkeypatch.delenv("DCOMROOT", raising=False)
    cfg = _cfg(_map(tmp_path / "wt.nc"))
    b, out, work = _run(tmp_path, cfg, {}, keep=True)
    with Dataset(str(work / "adt_on_rtofs.nc")) as ds:
        assert ds.dimensions["time"].size == 1
        assert ds["surf_el"].dimensions == ("time", "ylat", "xlon")
        assert ds["lon"].shape == (2, 2) and ds["lat"].shape == (2, 2)
        assert ds["surf_el"].units == "m"
        assert ds["surf_el"][0, 0, 0] == pytest.approx(0.5 * 0.9 + 0.5 * 0.9 - 0.45, abs=1e-6)
    b2, _, work2 = _run(tmp_path, cfg, {}, keep=False)
    assert b2.regrid == "esmf"


def test_default_no_adt_on_rtofs(tmp_path, monkeypatch):
    monkeypatch.delenv("COMINadt", raising=False)
    monkeypatch.delenv("DCOMROOT", raising=False)
    _, _, work = _run(tmp_path, _cfg(_map(tmp_path / "wt.nc")), {})
    assert not (work / "adt_on_rtofs.nc").exists()


def test_config_weight_defaults_and_yaml(tmp_path):
    assert str(ForcingConfig.for_stofs_3d_atl("20260401", 12).adt_weight_file) == \
        "stofs_3d_atl_ufs.adt_weight.nc"
    assert str(ForcingConfig.for_stofs_3d_atl_ufs("20260401", 12).adt_weight_file) == \
        "stofs_3d_atl_ufs.adt_weight.nc"
    y = tmp_path / "x.yaml"
    y.write_text("system: {name: foo}\nforcing:\n  ocean:\n    adt: {enabled: true, weight_file: my.nc}\n")
    assert str(ForcingConfig.from_yaml(y).adt_weight_file) == "my.nc"
