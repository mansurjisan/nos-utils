"""ADCIRC GFS to OWI (NWS=14) meteorology: equivalence with Zach Cobell's code (in-process).

The GRIB2 inputs are tiny files made with ecCodes samples. Tests that read GRIB skip
when cfgrib/ecCodes are not usable; wgrib2 is not needed (a fake script covers the subset).
"""

import importlib
import json
import os
import stat
import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from nos_utils.forcing import adcirc_met as am

ZACH_USH = Path(os.environ.get(
    "ZACH_STOFS_USH",
    "/mnt/d/NOS-Workflow-Project/STOFS_2DGLO_P0/zach_repo/ush"))


def _zach_path_ok():
    return (ZACH_USH / "StofsWorkflow" / "met" / "gfs_file_selector.py").exists()


@pytest.fixture(scope="module")
def zach_selector():
    if not _zach_path_ok():
        pytest.skip("Zach's clone not available (set ZACH_STOFS_USH)")
    sys.path.insert(0, str(ZACH_USH))
    try:
        mod = importlib.import_module("StofsWorkflow.met.gfs_file_selector")
    except Exception as exc:
        sys.path.remove(str(ZACH_USH))
        pytest.skip("cannot import his selector: {}".format(exc))
    yield mod
    sys.path.remove(str(ZACH_USH))


@pytest.fixture(scope="module")
def zach_model():
    """His AdcircModel; deps unrelated to met (haversine, schema) are stubbed if absent."""
    if not _zach_path_ok():
        pytest.skip("Zach's clone not available (set ZACH_STOFS_USH)")
    sys.path.insert(0, str(ZACH_USH))
    stubbed = []
    mod = None
    for _ in range(10):
        try:
            mod = importlib.import_module("StofsWorkflow.adcircmodel")
            break
        except ModuleNotFoundError as exc:
            if exc.name in ("xarray", "cfgrib"):
                break
            sys.modules[exc.name] = mock.MagicMock()
            stubbed.append(exc.name)
        except Exception:
            break
    if mod is None:
        sys.path.remove(str(ZACH_USH))
        for n in stubbed:
            sys.modules.pop(n, None)
        pytest.skip("cannot import his AdcircModel")
    yield mod.AdcircModel
    sys.path.remove(str(ZACH_USH))
    for n in stubbed:
        sys.modules.pop(n, None)


def _cfgrib_ok():
    try:
        import cfgrib  # noqa: F401
        import eccodes  # noqa: F401
        import xarray  # noqa: F401
    except Exception:
        return False
    return True


needs_grib = pytest.mark.skipif(not _cfgrib_ok(), reason="cfgrib/ecCodes/xarray not usable")

NJ, NI = 5, 7


def _make_grib(path, cycle, fhr, seed, with_ice=True):
    import eccodes as ec
    rng = np.random.RandomState(seed)
    keys = [("prmsl", 101000.0, 1500.0), ("10u", 0.0, 8.0), ("10v", 0.0, 8.0)]
    if with_ice:
        keys.append(("ci", 0.3, 0.3))
    with open(str(path), "wb") as out:
        for name, mean, sd in keys:
            h = ec.codes_grib_new_from_samples("regular_ll_sfc_grib2")
            ec.codes_set(h, "Ni", NI)
            ec.codes_set(h, "Nj", NJ)
            ec.codes_set(h, "latitudeOfFirstGridPointInDegrees", 10.0)
            ec.codes_set(h, "latitudeOfLastGridPointInDegrees", 6.0)
            ec.codes_set(h, "longitudeOfFirstGridPointInDegrees", 280.0)
            ec.codes_set(h, "longitudeOfLastGridPointInDegrees", 286.0)
            ec.codes_set(h, "iDirectionIncrementInDegrees", 1.0)
            ec.codes_set(h, "jDirectionIncrementInDegrees", 1.0)
            ec.codes_set(h, "dataDate", int(cycle.strftime("%Y%m%d")))
            ec.codes_set(h, "dataTime", cycle.hour * 100)
            ec.codes_set(h, "stepRange", str(fhr))
            ec.codes_set(h, "shortName", name)
            vals = mean + sd * rng.standard_normal(NI * NJ)
            if name == "ci":
                vals = np.clip(vals, 0.0, 1.0)
            ec.codes_set_values(h, vals)
            ec.codes_write(h, out)
            ec.codes_release(h)


def _make_tank(root, start, end, with_ice=True):
    """Every cycle and hour that either phase could ask for, hourly to f120."""
    sel = am.GfsFileSelector()
    cyc = sel.most_recent_cycle(start) - timedelta(hours=12)
    last = sel.most_recent_cycle(end)
    n = 0
    while cyc <= last:
        for fhr in range(0, 13):
            p = Path(root) / sel.build_gfs_path(cyc, fhr)
            p.parent.mkdir(parents=True, exist_ok=True)
            _make_grib(p, cyc, fhr, seed=int(cyc.strftime("%d%H")) * 100 + fhr,
                       with_ice=with_ice)
            n += 1
        cyc += timedelta(hours=6)
    return n


# ---- selection logic: identical to his selector ------------------------------------

def _avail(rng, cycles, hi):
    return {c: set(int(h) for h in range(hi + 1) if rng.rand() > 0.15) for c in cycles}


def _req_tuple(r):
    return (r.cycle_time, r.forecast_hour, r.valid_time, r.source_path)


def test_selector_matches_zach_nowcast_and_forecast(zach_selector):
    mine, his = am.GfsFileSelector(), zach_selector.GfsFileSelector()
    rng = np.random.RandomState(7)
    for i in range(40):
        start = datetime(2026, 10, 1) + timedelta(hours=int(rng.randint(0, 24 * 20)))
        span = int(rng.choice([6, 12, 30, 180, 300]))
        end = start + timedelta(hours=span)
        cycles = am.GfsFileSelector.get_candidate_cycles(start, end)
        assert cycles == zach_selector.GfsFileSelector.get_candidate_cycles(start, end)
        av = _avail(rng, cycles, span + 6)
        for fn in ("select_nowcast_files", "select_forecast_files"):
            try:
                exp = [_req_tuple(r) for r in getattr(his, fn)(start, end, availability=av)]
            except Exception as exc:
                with pytest.raises(am.InsufficientDataError):
                    getattr(mine, fn)(start, end, availability=av)
                assert type(exc).__name__ == "InsufficientDataError"
                continue
            got = [_req_tuple(r) for r in getattr(mine, fn)(start, end, availability=av)]
            assert got == exp
        # no availability map
        assert [_req_tuple(r) for r in mine.select_nowcast_files(start, end)] == \
            [_req_tuple(r) for r in his.select_nowcast_files(start, end)]


def test_selector_hourly_to_f120_then_3hourly(zach_selector):
    start = datetime(2026, 10, 5, 12)
    end = start + timedelta(hours=180)
    got = [r.forecast_hour for r in am.GfsFileSelector().select_forecast_files(start, end)]
    assert got[:3] == [0, 1, 2] and 120 in got and 121 not in got and 123 in got
    assert got[-1] == 180
    exp = [r.forecast_hour for r in zach_selector.GfsFileSelector().select_forecast_files(start, end)]
    assert got == exp


def test_paths_and_windows():
    assert am.GfsFileSelector.build_gfs_path(datetime(2026, 10, 5, 6), 3) == \
        "gfs.20261005/06/atmos/gfs.t06z.pgrb2.0p25.f003"
    w = am.adcirc_met_windows(datetime(2026, 10, 5, 12))
    assert w["nowcast_start"] == datetime(2026, 10, 5, 6)
    assert w["forecast_end"] == datetime(2026, 10, 13, 0)
    w = am.adcirc_met_windows(datetime(2026, 10, 5, 12), spinup_days=18.0)
    assert w["nowcast_start"] == datetime(2026, 9, 17, 12)


def test_fort22_ice_only_when_present(tmp_path):
    full = am.fort22_lines(ice=True)
    assert full[-1] == "icec" and len(full) == 11
    assert am.fort22_lines(ice=False) == full[:-1]
    p = am.write_fort22(tmp_path / "fort.22", ice=False)
    assert "icec" not in p.read_text() and p.read_text().endswith("vgrd10m\n")


def test_empty_comin_gfs_is_rejected():
    with pytest.raises(ValueError, match="comin_gfs"):
        am.AdcircMetProcessor("")


def test_missing_dependency_message():
    with mock.patch.dict(sys.modules, {"cfgrib": None}):
        with pytest.raises(am.MetDependencyError, match="cfgrib"):
            am._import_xarray()


_INVENTORY = [
    "1:0:d=2026100512:PRMSL:mean sea level:anl:",
    "2:1:d=2026100512:UGRD:10 m above ground:anl:",
    "3:2:d=2026100512:VGRD:10 m above ground:anl:",
    "4:3:d=2026100512:UGRD:850 mb:anl:",
    "5:4:d=2026100512:VGRD:500 mb:anl:",
    "6:5:d=2026100512:UGRD:planetary boundary layer:anl:",
    "7:6:d=2026100512:ICEC:surface:anl:",
    "8:7:d=2026100512:TMP:2 m above ground:anl:",
]


def test_default_subset_matches_only_10m_winds():
    got = am.match_inventory(_INVENTORY, am.DEFAULT_VARIABLES)
    assert [g.split(":")[0] for g in got] == ["1", "2", "3", "7"]


def test_bare_names_would_pull_every_level():
    got = am.match_inventory(_INVENTORY, ["PRMSL", "UGRD", "VGRD", "ICEC"])
    assert len(got) == 7  # the failure mode the 10 m strings avoid


def test_dependencies_checked_before_any_copy(tmp_path):
    proc = am.AdcircMetProcessor(tmp_path / "tank", subset=False)
    with mock.patch.dict(sys.modules, {"eccodes": None}):
        with pytest.raises(am.MetDependencyError, match="ecCodes|eccodes"):
            proc.process(datetime(2026, 10, 5, 0), datetime(2026, 10, 5, 6),
                         tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_wgrib2_subset_with_fake_binary(tmp_path):
    fake = tmp_path / "wgrib2"
    fake.write_text(
        "#!/bin/bash\n"
        "if [ \"$2\" = \"-s\" ]; then\n"
        "  echo '1:0:d=2026100512:PRMSL:mean sea level:anl:'\n"
        "  echo '2:9:d=2026100512:TMP:2 m above ground:anl:'\n"
        "  echo '3:19:d=2026100512:UGRD:10 m above ground:anl:'\n"
        "else\n"
        "  cat > /dev/null; echo \"$@\" > \"$(dirname $0)/args.txt\"; cp \"$1\" \"$4\"\n"
        "fi\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    src = tmp_path / "in.grib2"
    src.write_bytes(b"GRIB")
    src_obj = am.LocalGfsSource(tmp_path, ["PRMSL", "UGRD:10 m above ground"],
                                subset=True, wgrib2_path=str(fake))
    dst = tmp_path / "out.grb2"
    src_obj._copy_with_subset(src, dst)
    assert dst.read_bytes() == b"GRIB"
    assert (tmp_path / "args.txt").read_text().split()[1:3] == ["-i", "-grib"]


# ---- GRIB read, interpolation and writers: identical to his ------------------------

def _compare_outputs(mine_dir, his_dir, names):
    import xarray as xr
    for name in names:
        a = xr.open_dataset(str(mine_dir / name))
        b = xr.open_dataset(str(his_dir / name))
        assert a.attrs == b.attrs
        assert set(a.variables) == set(b.variables)
        assert dict(a.sizes) == dict(b.sizes)
        for v in a.variables:
            assert a[v].dtype == b[v].dtype
            assert a[v].attrs == b[v].attrs
            assert np.array_equal(a[v].values, b[v].values, equal_nan=True), (name, v)
        a.close()
        b.close()


@needs_grib
@pytest.mark.parametrize("with_ice", [True, False])
def test_process_nowcast_equals_zach(tmp_path, zach_model, with_ice):
    start, end = datetime(2026, 10, 5, 0), datetime(2026, 10, 5, 12)
    tank = tmp_path / "gfs_tank"
    _make_tank(tank, start, end, with_ice=with_ice)
    proc = am.AdcircMetProcessor(tank, subset=False)
    out = tmp_path / "mine"
    res = proc.process(start, end, out, phase="nowcast")

    assert res.n_records == 13 and res.phase == "nowcast"
    assert set(res.files) == ({"fort.221.nc", "fort.222.nc", "fort.225.nc"} if with_ice
                              else {"fort.221.nc", "fort.222.nc"})
    assert res.fort22.read_text().splitlines()[-1] == ("icec" if with_ice else "vgrd10m")
    man = json.loads(res.manifest.read_text())
    assert man["forcing"]["atmospheric"]["nfiles"] == 13

    his_files = [SimpleNamespace(local_path=p) for p in res.grib_files]
    his_dir = tmp_path / "his"
    his_dir.mkdir()
    if with_ice:
        zach_model._prepare_nws14_netcdf(SimpleNamespace(files=his_files), his_dir)
        _compare_outputs(out, his_dir, ["fort.221.nc", "fort.222.nc", "fort.225.nc"])
        assert (out / "fort.22").read_text() == (his_dir / "fort.22").read_text()
        for n in ("fort.221.nc", "fort.222.nc", "fort.225.nc"):
            assert (out / n).read_bytes() == (his_dir / n).read_bytes()
    else:
        # his unpatched code fails on the absent variable (June fix 5); the fixed
        # behaviour is checked by the assertions above
        with pytest.raises(Exception):
            zach_model._prepare_nws14_netcdf(SimpleNamespace(files=his_files), his_dir)


@needs_grib
def test_process_forecast_equals_zach(tmp_path, zach_model):
    start, end = datetime(2026, 10, 5, 12), datetime(2026, 10, 5, 23)
    tank = tmp_path / "gfs_tank"
    _make_tank(tank, start, end)
    res = am.AdcircMetProcessor(tank, subset=False).process(
        start, end, tmp_path / "mine", phase="forecast")
    assert res.cycle_time == start and res.n_records == 12
    assert [p.name for p in res.grib_files][0] == "gfs_2026100512.grb2"
    his_dir = tmp_path / "his"
    his_dir.mkdir()
    zach_model._prepare_nws14_netcdf(
        SimpleNamespace(files=[SimpleNamespace(local_path=p) for p in res.grib_files]), his_dir)
    _compare_outputs(tmp_path / "mine", his_dir,
                     ["fort.221.nc", "fort.222.nc", "fort.225.nc"])


@needs_grib
def test_time_interpolation_from_3hourly(tmp_path):
    cyc = datetime(2026, 10, 5, 0)
    files = []
    for fhr, val in ((0, 1.0), (3, 4.0)):
        p = tmp_path / "gfs_{:%Y%m%d%H}.grb2".format(cyc + timedelta(hours=fhr))
        _make_grib(p, cyc, fhr, seed=1)
        files.append(p)
    res = am.write_owi_netcdf(files, tmp_path / "o")
    assert len(res["hourly_times"]) == 4
    import xarray as xr
    ds = xr.open_dataset(str(tmp_path / "o" / "fort.221.nc"))
    p = ds["pressfc"].values
    assert np.allclose(p[1], p[0] + (p[3] - p[0]) / 3.0, rtol=1e-6)
    assert list(ds["record"].values) == [0, 1, 2, 3]
    ds.close()


@needs_grib
def test_missing_required_variable_raises_clear_error(tmp_path):
    import eccodes as ec
    p = tmp_path / "gfs_2026100500.grb2"
    h = ec.codes_grib_new_from_samples("regular_ll_sfc_grib2")
    ec.codes_set(h, "shortName", "ci")
    with open(str(p), "wb") as f:
        ec.codes_write(h, f)
    with pytest.raises(am.InsufficientDataError, match="prmsl"):
        am.write_owi_netcdf([p], tmp_path / "o")


# --- ops surface forcing from the GFS sfcf files ---------------------------------------------

NY, NX = 3, 4
CYC = datetime(2026, 10, 4, 0)


def _mk_sfcf(comin, cycle, fhr, seed, logf=True):
    import netCDF4 as nc
    p = am.sfcf_path(comin, cycle, fhr)
    p.parent.mkdir(parents=True, exist_ok=True)
    ds = nc.Dataset(str(p), "w", format="NETCDF4_CLASSIC")
    ds.fhzero = 6
    ds.dtp = np.float32(150.0)
    ds.source = "FV3GFS"
    ds.createDimension("grid_xt", NX)
    ds.createDimension("grid_yt", NY)
    ds.createDimension("time", 1)
    x = ds.createVariable("grid_xt", "f8", ("grid_xt",))
    x[:] = np.arange(NX)
    lon = ds.createVariable("lon", "f8", ("grid_yt", "grid_xt"))
    lon[:] = np.tile(np.arange(NX, dtype="f8"), (NY, 1))
    y = ds.createVariable("grid_yt", "f8", ("grid_yt",))
    y[:] = np.arange(NY)
    lat = ds.createVariable("lat", "f8", ("grid_yt", "grid_xt"))
    lat[:] = np.tile(np.arange(NY, dtype="f8")[:, None], (1, NX))
    t = ds.createVariable("time", "f8", ("time",))
    t.units = "hours since {:%Y-%m-%d %H:%M:%S}".format(cycle)
    t[:] = fhr
    for k, name in enumerate(("ugrd10m", "vgrd10m", "pressfc", "icec")):
        v = ds.createVariable(name, "f4", ("time", "grid_yt", "grid_xt"), fill_value=np.float32(9.99e20))
        v.long_name = name
        v.cell_methods = "time: point"
        v[0] = np.full((NY, NX), seed * 10 + k, dtype="f4") + np.arange(NY * NX, dtype="f4").reshape(NY, NX)
    ds.close()
    if logf:
        p.with_name(p.name.replace("sfcf", "logf").replace(".nc", ".txt")).write_text("done")
    return p


def _valid_seed(cycle, fhr):
    return int((cycle + timedelta(hours=fhr) - datetime(2026, 10, 1)).total_seconds() // 3600)


def _fill_tank(comin, pairs):
    for cycle, fhr in pairs:
        _mk_sfcf(comin, cycle, fhr, _valid_seed(cycle, fhr))


def _read(path, var):
    import netCDF4 as nc
    with nc.Dataset(str(path)) as d:
        d.set_auto_maskandscale(False)
        return d.variables[var][:]


def test_getges_picks_smallest_lead_and_falls_back(tmp_path):
    c18, c12 = CYC - timedelta(hours=6), CYC - timedelta(hours=12)
    _fill_tank(tmp_path, [(CYC, 0), (c18, 1), (c18, 6), (c12, 6), (c12 - timedelta(hours=6), 12)])
    assert am.getges_sfcf(tmp_path, CYC) == am.sfcf_path(tmp_path, CYC, 0)
    assert am.getges_sfcf(tmp_path, CYC - timedelta(hours=5)) == am.sfcf_path(tmp_path, c18, 1)
    # f000 absent -> the previous cycle's f006; that absent too -> the cycle before, f012
    assert am.getges_sfcf(tmp_path, c18) == am.sfcf_path(tmp_path, c12, 6)
    am.sfcf_path(tmp_path, c12, 6).unlink()
    assert am.getges_sfcf(tmp_path, c18) == am.sfcf_path(tmp_path, c12 - timedelta(hours=6), 12)
    with pytest.raises(am.FileNotAvailableError, match="no GFS sfcf"):
        am.getges_sfcf(tmp_path, CYC + timedelta(hours=40))


def test_sfcf_windows():
    ncst = am.sfcf_valid_times(CYC, "ncst")
    assert (len(ncst), ncst[0], ncst[-1]) == (7, CYC - timedelta(hours=6), CYC)
    assert len(am.sfcf_valid_times(CYC, "ncst", CYC - timedelta(hours=12))) == 13
    f = am.sfcf_valid_times(CYC, "fcst")
    assert (len(f), f[0], f[-1]) == (181, CYC, CYC + timedelta(hours=180))
    for old in ("spinup", "fcst1", "fcst2"):
        with pytest.raises(ValueError):
            am.sfcf_valid_times(CYC, old)
    with pytest.raises(am.MetForcingError):
        am.sfcf_valid_times(CYC, "ncst", CYC + timedelta(hours=1))


def test_ncst_layout_values_and_manifest(tmp_path):
    import netCDF4 as nc
    tank, out = tmp_path / "gfs", tmp_path / "rerun"
    c18 = CYC - timedelta(hours=6)
    # valid 18 -> f001 of 17z does not exist; f000 absent so it falls back to 12z f006
    _fill_tank(tank, [(CYC - timedelta(hours=12), 6)] + [(c18, h) for h in range(1, 6)] + [(CYC, 0)])
    files = am.build_sfcf_forcing(tank, CYC, "ncst", out)
    assert sorted(files) == ["221", "222", "225"]
    assert sorted(p.name for p in out.iterdir()) == sorted(
        ["stofs_2d_glo_ncst.{}.nc".format(n) for n in files] + ["stofs_2d_glo_ncst.sfcf_manifest.json"])
    with nc.Dataset(str(files["222"])) as d:
        assert d.data_model == "NETCDF4_CLASSIC"
        assert [k for k in d.dimensions] == ["grid_xt", "grid_yt", "record"]
        assert d.dimensions["record"].isunlimited() and len(d.dimensions["record"]) == 7
        assert list(d.variables) == ["grid_xt", "grid_yt", "lat", "lon", "time", "ugrd10m", "vgrd10m"]
        assert d.variables["lat"].dimensions == ("record", "grid_yt", "grid_xt")
        assert d.variables["time"].dimensions == () and d.variables["time"].cell_methods == "time: mean"
        assert d.variables["ugrd10m"].cell_methods == "time: point time: mean"
        assert d.variables["ugrd10m"].filters()["zlib"] and d.variables["ugrd10m"].filters()["shuffle"]
        assert d.variables["ugrd10m"].chunking() == [1, NY, NX]
        assert d.variables["lat"].dtype == np.dtype("f8") and d.variables["ugrd10m"].dtype == np.dtype("f4")
        assert d.getncattr("fhzero") == 6 and d.getncattr("dtp") == np.float32(150.0)
        assert d.variables["ugrd10m"].getncattr("_FillValue") == np.float32(9.99e20)
    # record 0 is 12z f006 (time 6, units from that file), records 1-5 are 18z f001-f005, record 6 is 00z f000
    src = [am.sfcf_path(tank, CYC - timedelta(hours=12), 6)] + [am.sfcf_path(tank, c18, h) for h in range(1, 6)] \
        + [am.sfcf_path(tank, CYC, 0)]
    for n, names in am.SFC_OUTPUTS:
        for v in names:
            got = _read(files[n], v)
            assert got.shape == (7, NY, NX)
            for r, s in enumerate(src):
                assert np.array_equal(got[r], _read(s, v)[0])
    with nc.Dataset(str(files["225"])) as d:
        assert list(d.variables) == ["grid_xt", "grid_yt", "icec", "lat", "lon", "time"]
    with nc.Dataset(str(files["221"])) as d:
        assert d.variables["time"].units == "hours since 2026-10-03 12:00:00" and d.variables["time"][...] == 6
    man = json.loads((out / "stofs_2d_glo_ncst.sfcf_manifest.json").read_text())
    assert man["nfiles"] == 7 and man["files"][0]["source_location"] == str(src[0])


def test_ncst_start_override_gives_fewer_records(tmp_path):
    c18 = CYC - timedelta(hours=6)
    _fill_tank(tmp_path / "g", [(c18, h) for h in range(1, 6)] + [(CYC, 0)])
    files = am.build_sfcf_forcing(tmp_path / "g", CYC, "ncst", tmp_path / "o", start=CYC - timedelta(hours=5))
    assert _read(files["225"], "icec").shape[0] == 6


def _fcst_tank(tank, last=180):
    _fill_tank(tank, [(CYC, h) for h in range(0, 121)] + [(CYC, h) for h in range(123, last + 1, 3)])


def test_fcst_is_hourly_with_interpolation_after_f120(tmp_path):
    tank = tmp_path / "g"
    _fcst_tank(tank)
    f = am.build_sfcf_forcing(tank, CYC, "fcst", tmp_path / "o")
    p = _read(f["221"], "pressfc")
    assert p.shape[0] == 181
    s = lambda h, k: _valid_seed(CYC, h) * 10 + k
    for h in (0, 7, 120, 123, 126, 180):  # native records are exact copies. MJ (10/11/26)
        assert p[h][0, 0] == s(h, 2)
    # 121 and 122 are 1/3 and 2/3 of the way from f120 to f123; the seeds step 10 per hour. MJ (10/11/26)
    assert p[121][0, 0] == pytest.approx(s(120, 2) + 10.0)
    assert p[122][0, 0] == pytest.approx(s(120, 2) + 20.0)
    q = _read(f["222"], "ugrd10m")
    assert q[121][1, 2] == pytest.approx(s(120, 0) + 10.0 + 6)
    assert _read(f["225"], "icec").shape[0] == 181
    man = json.loads((tmp_path / "o" / "stofs_2d_glo_fcst.sfcf_manifest.json").read_text())["files"]
    assert len(man) == 181 + 40 and [e["source_location"] for e in man[121:123]] == [
        str(am.sfcf_path(tank, CYC, 120)), str(am.sfcf_path(tank, CYC, 123))]


def test_fcst_interpolation_keeps_fill_values(tmp_path):
    tank = tmp_path / "g"
    _fcst_tank(tank)
    import netCDF4 as nc
    with nc.Dataset(str(am.sfcf_path(tank, CYC, 123)), "a") as d:
        d.variables["icec"][0, 0, 0] = np.float32(9.99e20)
    f = am.build_sfcf_forcing(tank, CYC, "fcst", tmp_path / "o")
    i = _read(f["225"], "icec")
    assert i[121][0, 0] == np.float32(9.99e20) and i[120][0, 0] < 1e6 and i[121][0, 1] < 1e6


def test_fcst_rejects_a_bracket_on_another_grid(tmp_path):
    import netCDF4 as nc
    tank = tmp_path / "g"
    _fcst_tank(tank)
    with nc.Dataset(str(am.sfcf_path(tank, CYC, 123)), "a") as d:
        d.variables["lon"][:] = d.variables["lon"][:] + 0.5
    with pytest.raises(am.MetForcingError, match="grid differs"):
        am.build_sfcf_forcing(tank, CYC, "fcst", tmp_path / "o")
    assert list((tmp_path / "o").glob("*.nc*")) == []


def test_forecast_waits_then_fails_clearly(tmp_path):
    tank = tmp_path / "g"
    _fcst_tank(tank, last=177)  # f180 is missing. MJ (10/11/26)
    clock = {"t": 0.0}
    naps = []

    def sleep(s):
        naps.append(s)
        clock["t"] += s

    with pytest.raises(am.FileNotAvailableError, match="sfcf180"):
        am.build_sfcf_forcing(tank, CYC, "fcst", tmp_path / "o", wait_s=25, poll_s=10,
                              sleep=sleep, clock=lambda: clock["t"])
    assert naps == [10, 10, 5]
    assert list((tmp_path / "o").glob("*.nc*")) == []  # no final or .partial files from a failed run

    def sleep_and_land(s):
        _mk_sfcf(tank, CYC, 180, 1)
        sleep(s)

    files = am.build_sfcf_forcing(tank, CYC, "fcst", tmp_path / "o", wait_s=25, poll_s=10,
                                  sleep=sleep_and_land, clock=lambda: clock["t"])
    assert _read(files["221"], "pressfc").shape[0] == 181


def test_current_cycle_f000_is_not_waited_for(tmp_path):
    # fcst record 0 (valid == cycle) goes through getges, so it never waits. MJ (10/11/26)
    _fill_tank(tmp_path / "g", [(CYC - timedelta(hours=6), 6)] + [(CYC, h) for h in range(1, 121)]
                + [(CYC, h) for h in range(123, 181, 3)])
    files = am.build_sfcf_forcing(tmp_path / "g", CYC, "fcst", tmp_path / "o", wait_s=0,
                                  sleep=lambda s: pytest.fail("waited"))
    assert _read(files["221"], "pressfc").shape[0] == 181


def _age(p, seconds):
    t = p.stat().st_mtime - seconds
    os.utime(str(p), (t, t))


def test_fresh_sfcf_needs_its_logf_marker(tmp_path):
    p = _mk_sfcf(tmp_path, CYC, 1, 1, logf=False)
    assert not am._sfcf_ready(p)  # just written, no logf: still being copied
    _age(p, 3600)
    assert am._sfcf_ready(p)  # old: a retrospective file
    q = _mk_sfcf(tmp_path, CYC, 2, 1, logf=False)
    assert not am._sfcf_ready(q)
    q.with_name("gfs.t00z.logf002.txt").write_text("done")
    assert am._sfcf_ready(q)


def test_realtime_cycle_record_waits_for_f000_before_falling_back(tmp_path):
    c18 = CYC - timedelta(hours=6)
    _fill_tank(tmp_path, [(c18, h) for h in range(1, 7)])
    for h in range(1, 7):
        _age(am.sfcf_path(tmp_path, c18, h), 3600)
    epoch = __import__("calendar").timegm(CYC.timetuple())
    naps = []

    def sleep(s):
        naps.append(s)
        p = _mk_sfcf(tmp_path, CYC, 0, 5)
        _age(p, 3600)

    files = am.build_sfcf_forcing(tmp_path, CYC, "ncst", tmp_path / "o", start=CYC - timedelta(hours=1), wait_s=30,
                                  sleep=sleep, clock=lambda: 0.0, now=lambda: epoch + 600)
    assert naps == [10]
    assert _read(files["221"], "pressfc")[1][0, 0] == 52
    assert str(am.sfcf_path(tmp_path, CYC, 0)) in (tmp_path / "o" / "stofs_2d_glo_ncst.sfcf_manifest.json").read_text()


def test_retrospective_cycle_record_does_not_wait(tmp_path):
    c18 = CYC - timedelta(hours=6)
    _fill_tank(tmp_path, [(c18, h) for h in range(1, 7)])
    epoch = __import__("calendar").timegm(CYC.timetuple())
    am.build_sfcf_forcing(tmp_path, CYC, "ncst", tmp_path / "o", start=CYC - timedelta(hours=1), wait_s=30,
                          sleep=lambda s: pytest.fail("waited"), now=lambda: epoch + 3 * 86400)


# --- the same forcing from the sfluxgrbf GRIB2 files (decoder mocked) -------------------------

_SF_NAMES = {"ugrd10m": 0, "vgrd10m": 1, "pressfc": 2, "icec": 3}


def _mk_sflux(comin, cycle, fhr, seed=None):
    p = am.sflux_path(comin, cycle, fhr)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"GRIB")
    p.with_name(p.name + ".idx").write_text("1:0")  # the downloader keeps the idx next to the grib. MJ (10/14/26)
    return p


def _fake_decode(descending=False, undef=False):
    import re

    def dec(path):
        m = re.search(r"gfs\.(\d{8})/(\d{2})/atmos/.*sfluxgrbf(\d{3})", str(path))
        cyc, fhr = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H"), int(m.group(3))
        seed = _valid_seed(cyc, fhr)
        lat, lon = np.arange(NY, dtype="f8"), np.arange(NX, dtype="f8")
        ramp = np.arange(NY * NX, dtype="f4").reshape(NY, NX)
        f = {k: np.full((NY, NX), seed * 10 + i, dtype="f4") + ramp for k, i in _SF_NAMES.items()}
        if undef:
            f["icec"][0, 0] = np.float32(9.999e20)
        if descending:
            lat, f = lat[::-1], {k: v[::-1] for k, v in f.items()}
        return lat, lon, f
    return dec


def _fill_sflux(tank, pairs):
    for cycle, fhr in pairs:
        _mk_sflux(tank, cycle, fhr)


def _same_structure(a, b):
    import netCDF4 as nc
    with nc.Dataset(str(a)) as x, nc.Dataset(str(b)) as y:
        assert list(x.dimensions) == list(y.dimensions)
        assert list(x.variables) == list(y.variables)
        for k, v in x.variables.items():
            w = y.variables[k]
            assert (v.dtype, v.dimensions, v.shape) == (w.dtype, w.dimensions, w.shape), k
            assert v.filters() == w.filters() and v.chunking() == w.chunking(), k
            assert getattr(v, "_FillValue", None) == getattr(w, "_FillValue", None), k
            assert getattr(v, "cell_methods", None) == getattr(w, "cell_methods", None), k
            assert np.array_equal(v[:], w[:]), k
        assert x.variables["time"].units == y.variables["time"].units


def test_sflux_matches_sfcf_path_ncst_and_fcst(tmp_path, monkeypatch):
    monkeypatch.setattr(am, "_SFCF_LAT_ASCENDING", True)  # the fixture sfcf grid and fake decoder are both south-north. MJ (10/14/26)
    monkeypatch.setattr(am, "_decode_sflux", _fake_decode())
    c18 = CYC - timedelta(hours=6)
    ncst = [(CYC - timedelta(hours=12), 6)] + [(c18, h) for h in range(1, 6)] + [(CYC, 0)]
    fc = [(CYC, h) for h in range(0, 121)] + [(CYC, h) for h in range(123, 181, 3)]
    _fill_tank(tmp_path / "a", ncst + fc)
    _fill_sflux(tmp_path / "b", ncst + fc)
    for seg in ("ncst", "fcst"):
        fa = am.build_sfcf_forcing(tmp_path / "a", CYC, seg, tmp_path / "oa")
        fb = am.build_sfcf_forcing(tmp_path / "b", CYC, seg, tmp_path / "ob", source="sflux")
        for n in fa:
            _same_structure(fa[n], fb[n])
    man = json.loads((tmp_path / "ob" / "stofs_2d_glo_fcst.sfcf_manifest.json").read_text())
    assert man["product"] == "sflux" and len(man["files"]) == 181 + 40
    assert man["files"][121]["source_location"] == str(am.sflux_path(tmp_path / "b", CYC, 120))
    assert man["files"][122]["source_location"] == str(am.sflux_path(tmp_path / "b", CYC, 123))
    assert json.loads((tmp_path / "oa" / "stofs_2d_glo_fcst.sfcf_manifest.json").read_text())["product"] == "sfcf"


def test_sflux_orientation_is_one_decision(tmp_path, monkeypatch):
    _fill_sflux(tmp_path, [(CYC, 0)])
    monkeypatch.setattr(am, "_decode_sflux", _fake_decode(descending=True))
    f = am.build_sfcf_forcing(tmp_path, CYC, "ncst", tmp_path / "o1", start=CYC, source="sflux")  # no flip: N to S kept. MJ (10/14/26)
    assert np.array_equal(_read(f["221"], "grid_yt"), np.arange(NY)[::-1])
    assert _read(f["221"], "pressfc")[0][0, 0] == _valid_seed(CYC, 0) * 10 + 2 + (NY - 1) * NX
    monkeypatch.setattr(am, "_SFCF_LAT_ASCENDING", True)
    f = am.build_sfcf_forcing(tmp_path, CYC, "ncst", tmp_path / "o2", start=CYC, source="sflux")
    assert np.array_equal(_read(f["221"], "grid_yt"), np.arange(NY))
    assert _read(f["221"], "pressfc")[0][0, 0] == _valid_seed(CYC, 0) * 10 + 2


def test_sflux_bitmap_points_icec_zero_others_raise(tmp_path, monkeypatch):
    monkeypatch.setattr(am, "_SFCF_LAT_ASCENDING", True)
    _fill_sflux(tmp_path, [(CYC, 0)])
    monkeypatch.setattr(am, "_decode_sflux", _fake_decode(undef=True))
    f = am.build_sfcf_forcing(tmp_path, CYC, "ncst", tmp_path / "o", start=CYC, source="sflux")
    i = _read(f["225"], "icec")[0]
    assert i[0, 0] == 0.0 and i[0, 1] > 0
    with pytest.raises(am.MetForcingError, match="undefined"):
        am._resolve_undefined("pressfc", np.full((2, 2), 9.999e20, dtype="f4"))


def test_sflux_attributes_match_sfcf(tmp_path, monkeypatch):
    import netCDF4 as nc
    _fill_sflux(tmp_path, [(CYC, 0)])
    monkeypatch.setattr(am, "_decode_sflux", _fake_decode())
    f = am.build_sfcf_forcing(tmp_path, CYC, "ncst", tmp_path / "o", start=CYC, source="sflux")
    with nc.Dataset(str(f["222"])) as d:
        g = {k: d.getncattr(k) for k in d.ncattrs()}
        assert g == {"fhzero": 6, "ncld": 5, "nsoil": 4, "imp_physics": 11, "dtp": 150.0, "source": "FV3GFS",
                     "grid": "gaussian", "im": NX, "jm": NY}
        assert type(g["fhzero"]) is np.int32 and type(g["dtp"]) is np.float32
        u = d.variables["ugrd10m"]
        assert (u.long_name, u.units, u.output_file) == ("10 meter u wind", "m/s", "sfc")
        assert u.getncattr("missing_value") == np.float32(9.99e20) == u.getncattr("_FillValue")
        assert d.variables["grid_yt"].long_name == "T-cell latiitude"
        assert d.variables["lon"].dtype == np.dtype("f8") and d.variables["lat"].dimensions == ("record", "grid_yt", "grid_xt")
        assert d.variables["time"].units == "hours since 2026-10-04 00:00:00" and d.variables["time"].calendar == "JULIAN"
    with nc.Dataset(str(f["225"])) as d:
        assert (d.variables["icec"].units, d.variables["icec"].cell_methods) == ("fraction", "time: point time: mean")


def test_gfs_input_files(tmp_path):
    c18, c12 = CYC - timedelta(hours=6), CYC - timedelta(hours=12)
    n = am.gfs_input_files(tmp_path, CYC, "ncst")
    assert n == [am.sfcf_path(tmp_path, c18, h) for h in range(0, 6)] + [am.sfcf_path(tmp_path, CYC, 0)]
    assert len(am.gfs_input_files(tmp_path, CYC, "ncst", start=CYC - timedelta(hours=12))) == 13
    sf = am.gfs_input_files(tmp_path, CYC, "ncst", source="sflux")  # includes the cycle's own f000 analysis. MJ (10/14/26)
    assert sf == [am.sflux_path(tmp_path, c18, h) for h in range(0, 6)] + [am.sflux_path(tmp_path, CYC, 0)]
    f = am.gfs_input_files(tmp_path, CYC, "fcst", source="sflux")
    assert f == [am.sflux_path(tmp_path, CYC, h) for h in list(range(0, 121)) + list(range(123, 181, 3))]
    assert len(f) == len(set(f))
    assert c18 != c12


def test_sflux_missing_file_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(am, "_decode_sflux", _fake_decode())
    with pytest.raises(am.FileNotAvailableError, match="no GFS sflux file"):
        am.build_sfcf_forcing(tmp_path, CYC, "ncst", tmp_path / "o", start=CYC, source="sflux")
    _fill_sflux(tmp_path, [(CYC, h) for h in range(0, 121)])
    with pytest.raises(am.FileNotAvailableError, match="did not appear"):
        am.build_sfcf_forcing(tmp_path, CYC, "fcst", tmp_path / "o", source="sflux")


def test_sflux_without_wgrib2_is_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.setattr(am.shutil, "which", lambda _: None)
    with pytest.raises(am.MetDependencyError, match="wgrib2"):
        am._decode_sflux(am.sflux_path(tmp_path, CYC, 0))


@pytest.mark.parametrize("source", ["sfcf", "sflux"])
@pytest.mark.parametrize("segment", ["ncst", "fcst"])
def test_manifest_files_equal_gfs_input_files(tmp_path, monkeypatch, source, segment):
    import re
    monkeypatch.setattr(am, "_SFCF_LAT_ASCENDING", True)
    monkeypatch.setattr(am, "_decode_sflux", _fake_decode())
    tank = tmp_path / "t"
    want = am.gfs_input_files(tank, CYC, segment, source=source)
    for p in want:  # a tank holding exactly the listed files, nothing else. MJ (10/14/26)
        h = int(re.search(r"f(\d{3})\.grib2$|sfcf(\d{3})\.nc$", p.name).group(1, 2)[0 if source == "sflux" else 1])
        cyc = datetime.strptime(p.parts[-4][4:] + p.parts[-3], "%Y%m%d%H")
        if source == "sfcf":
            _mk_sfcf(tank, cyc, h, _valid_seed(cyc, h))
        else:
            _mk_sflux(tank, cyc, h)
    am.build_sfcf_forcing(tank, CYC, segment, tmp_path / "o", source=source)
    man = json.loads((tmp_path / "o" / "stofs_2d_glo_{}.sfcf_manifest.json".format(segment)).read_text())
    assert list(dict.fromkeys(e["source_location"] for e in man["files"])) == [str(p) for p in want]


def test_sflux_ready_needs_idx_or_age(tmp_path):
    import os
    p = _mk_sflux(tmp_path, CYC, 0)
    assert am._sfcf_ready(p)
    p.with_name(p.name + ".idx").unlink()
    assert not am._sfcf_ready(p)
    q = _mk_sflux(tmp_path, CYC, 1)
    q.with_name(q.name + ".idx").unlink()
    old = q.stat().st_mtime - 3600
    os.utime(str(q), (old, old))
    assert am._sfcf_ready(q)


def test_sflux_decoder_rejects_bad_scan_and_times_out(tmp_path, monkeypatch):
    monkeypatch.setattr(am.shutil, "which", lambda _: "/bin/wgrib2")
    out = "gaussian grid: (4 x 3) units 1e-06 input {} output WE:SN lat 1 to -1 lon 0 to 270"
    monkeypatch.setattr(am.subprocess, "run", lambda *a, **k: type(
        "R", (), {"returncode": 0, "stdout": out.format("WE:EW"), "stderr": ""})())
    with pytest.raises(am.MetForcingError, match="scan order"):
        am._decode_sflux(tmp_path / "x.grib2")

    def boom(*a, **k):
        raise am.subprocess.TimeoutExpired("wgrib2", 1)
    monkeypatch.setattr(am.subprocess, "run", boom)
    with pytest.raises(am.MetForcingError, match="timed out"):
        am._decode_sflux(tmp_path / "x.grib2")


def test_sflux_bracket_files_decoded_once_per_build(tmp_path, monkeypatch):
    _fill_sflux(tmp_path, [(CYC, h) for h in range(0, 121)] + [(CYC, h) for h in range(123, 181, 3)])
    inner, calls = _fake_decode(), []
    monkeypatch.setattr(am, "_decode_sflux", lambda p: (calls.append(str(p)), inner(p))[1])
    monkeypatch.setattr(am, "_SFCF_LAT_ASCENDING", True)
    am.build_sfcf_forcing(tmp_path, CYC, "fcst", tmp_path / "o", source="sflux")
    assert len(calls) == len(set(calls)) == 121 + 20


def test_getges_sflux_skips_a_file_that_is_not_ready(tmp_path):
    c18 = CYC - timedelta(hours=6)
    _mk_sflux(tmp_path, c18, 6)
    half = _mk_sflux(tmp_path, CYC, 0)
    half.with_name(half.name + ".idx").unlink()  # fresh and no idx: still downloading. MJ (10/14/26)
    assert am.getges_sfcf(tmp_path, CYC, "sflux") == am.sflux_path(tmp_path, c18, 6)
    half.with_name(half.name + ".idx").write_text("1:0")
    assert am.getges_sfcf(tmp_path, CYC, "sflux") == half


def test_getges_sfcf_needs_only_a_readable_file(tmp_path):
    p = _mk_sfcf(tmp_path, CYC, 0, 1, logf=False)  # no logf marker and fresh: sfcf still takes it. MJ (10/14/26)
    assert am.getges_sfcf(tmp_path, CYC) == p


@pytest.mark.parametrize("source", ["sfcf", "sflux"])
def test_build_logs_the_used_input_files(tmp_path, monkeypatch, source):
    from nos_utils.forcing import _log
    monkeypatch.setattr(am, "_SFCF_LAT_ASCENDING", True)
    monkeypatch.setattr(am, "_decode_sflux", _fake_decode())
    tank = tmp_path / "t"
    want = am.gfs_input_files(tank, CYC, "ncst", source=source)
    for p in want:
        cyc = datetime.strptime(p.parts[-4][4:] + p.parts[-3], "%Y%m%d%H")
        h = int(p.name.rsplit("sfcf" if source == "sfcf" else "sfluxgrbf", 1)[1][:3])
        (_mk_sfcf(tank, cyc, h, _valid_seed(cyc, h)) if source == "sfcf" else _mk_sflux(tank, cyc, h))
    calls = []
    monkeypatch.setattr(_log, "log_input_files", lambda *a, **k: calls.append((a, k)))
    am.build_sfcf_forcing(tank, CYC, "ncst", tmp_path / "o", source=source)
    assert len(calls) == 1
    (name, files), kw = calls[0]
    assert name == "GFS_" + source.upper() and kw["category"] == "atmospheric"
    assert list(files) == want
