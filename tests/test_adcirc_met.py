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


def _mk_sfcf(comin, cycle, fhr, seed):
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
    f1 = am.sfcf_valid_times(CYC, "fcst1")
    assert (len(f1), f1[0], f1[-1]) == (121, CYC, CYC + timedelta(hours=120))
    f2 = am.sfcf_valid_times(CYC, "fcst2")
    assert (len(f2), f2[0], f2[1], f2[-1]) == (
        21, CYC + timedelta(hours=120), CYC + timedelta(hours=123), CYC + timedelta(hours=180))
    with pytest.raises(ValueError):
        am.sfcf_valid_times(CYC, "spinup")
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
        assert "ncwa -a time tmp.222.nc fort.222.nc" in d.history.splitlines()[0]
        assert "ncks -v time,grid_xt,lon,grid_yt,lat,ugrd10m,vgrd10m swnd.2026100318" in d.history
    # record 0 is 12z f006 (time 6, units from that file), records 1-5 are 18z f001-f005, record 6 is 00z f000
    src = [am.sfcf_path(tank, CYC - timedelta(hours=12), 6)] + [am.sfcf_path(tank, c18, h) for h in range(1, 6)] \
        + [am.sfcf_path(tank, CYC, 0)]
    for n, names in am.SFC_OUTPUTS:
        for v in names:
            got = _read(files[n], v)
            assert got.shape == (7, NY, NX)
            for r, s in enumerate(src):
                assert np.array_equal(got[r], _read(s, v)[0])
    with nc.Dataset(str(files["221"])) as d:
        assert d.variables["time"].units == "hours since 2026-10-03 12:00:00" and d.variables["time"][...] == 6
    man = json.loads((out / "stofs_2d_glo_ncst.sfcf_manifest.json").read_text())
    assert man["nfiles"] == 7 and man["files"][0]["source_location"] == str(src[0])


def test_ncst_start_override_gives_fewer_records(tmp_path):
    c18 = CYC - timedelta(hours=6)
    _fill_tank(tmp_path / "g", [(c18, h) for h in range(1, 6)] + [(CYC, 0)])
    files = am.build_sfcf_forcing(tmp_path / "g", CYC, "ncst", tmp_path / "o", start=CYC - timedelta(hours=5))
    assert _read(files["225"], "icec").shape[0] == 6


def test_fcst1_and_fcst2_record_counts_and_stride(tmp_path):
    tank = tmp_path / "g"
    _fill_tank(tank, [(CYC, 0)] + [(CYC, h) for h in range(1, 121)])
    f1 = am.build_sfcf_forcing(tank, CYC, "fcst1", tmp_path / "o")
    p = _read(f1["221"], "pressfc")
    assert p.shape[0] == 121
    assert p[7][0, 0] == _valid_seed(CYC, 7) * 10 + 2
    _fill_tank(tank, [(CYC, h) for h in range(120, 181, 3)])
    f2 = am.build_sfcf_forcing(tank, CYC, "fcst2", tmp_path / "o")
    q = _read(f2["222"], "ugrd10m")
    assert q.shape[0] == 21
    assert q[1][0, 0] == _valid_seed(CYC, 123) * 10 + 0


def test_forecast_waits_then_fails_clearly(tmp_path):
    tank = tmp_path / "g"
    _fill_tank(tank, [(CYC, h) for h in range(120, 178, 3)])  # f180 is missing
    clock = {"t": 0.0}
    naps = []

    def sleep(s):
        naps.append(s)
        clock["t"] += s

    with pytest.raises(am.FileNotAvailableError, match="sfcf180"):
        am.build_sfcf_forcing(tank, CYC, "fcst2", tmp_path / "o", wait_s=25, poll_s=10,
                              sleep=sleep, clock=lambda: clock["t"])
    assert naps == [10, 10, 5]
    assert list((tmp_path / "o").glob("*.nc")) == []  # no final files from a failed run

    def sleep_and_land(s):
        _mk_sfcf(tank, CYC, 180, 1)
        sleep(s)

    files = am.build_sfcf_forcing(tank, CYC, "fcst2", tmp_path / "o", wait_s=25, poll_s=10,
                                  sleep=sleep_and_land, clock=lambda: clock["t"])
    assert _read(files["221"], "pressfc").shape[0] == 21


def test_current_cycle_f000_is_not_waited_for(tmp_path):
    # fcst1 record 0 (valid == cycle) goes through getges, so it never waits
    _fill_tank(tmp_path / "g", [(CYC - timedelta(hours=6), 6)] + [(CYC, h) for h in range(1, 121)])
    files = am.build_sfcf_forcing(tmp_path / "g", CYC, "fcst1", tmp_path / "o", wait_s=0,
                                  sleep=lambda s: pytest.fail("waited"))
    assert _read(files["221"], "pressfc").shape[0] == 121
