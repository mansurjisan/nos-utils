"""
Synthetic tests for the water-level OBC bias correction port
(:mod:`nos_utils.forcing.wl_bias`).

All fixtures are synthetic and network-free.  The tide prediction is used to
BUILD the synthetic observations (obs = prediction + known subtidal), so the
subtidal, AVGERR and residual are recovered EXACTLY despite the tide-prediction
internals being the documented approximation boundary.

Grid convention (shared by every test):
    base_date = 2026-04-01 00:00, DELT = 3600 s, day_start = 0, day_end = 2.
    prediction grid  time_prd = day_start-2 .. day_end  (step 1/24 day, 97 pts)
    zeta_time (zt)   = time_prd[48:97]                  (0 .. 2 day, 49 pts)
    obs live on time_prd nodes so the 2-point detide interpolation is exact.
"""

from datetime import datetime

import numpy as np
import pytest

from nos_utils.forcing.wl_bias import (
    ArrayObsProvider,
    BoundaryNode,
    CorrectionStation,
    ObcCtl,
    WlCorrections,
    _avgerr_ramp,
    _detide,
    apply_wl_corrections,
    compute_wl_corrections,
    parse_obc_ctl,
    predict_station_tide,
    read_station_hc,
)

BASE_DATE = datetime(2026, 4, 1, 0, 0)
DELT = 3600.0
DAY_START = 0.0
DAY_END = 2.0


def _grid():
    """Return (time_prd, zeta_time) exactly as the module builds them."""
    nrec = int(round((DAY_END - DAY_START + 2.0) * 86400.0 / DELT)) + 1
    time_prd = DAY_START - 2.0 + np.arange(nrec) * DELT / 86400.0
    zt = time_prd[48:97]  # 0 .. 2 day, 49 points
    return time_prd, zt


def _chararray(strings, nchar):
    """Build an (n, nchar) 'S1' char array (stringtochar-free; numpy-2 safe)."""
    arr = np.zeros((len(strings), nchar), dtype="S1")
    for i, s in enumerate(strings):
        b = s.encode("ascii")[:nchar]
        for j in range(len(b)):
            arr[i, j] = b[j : j + 1]
    return arr


def _write_hc(path, station_ids, m2=(0.5, 30.0), s2=(0.2, 60.0)):
    """
    Write a minimal ``nosofs.HC_NWLON.nc`` with the exact ported schema:
    stationID(Station, staID), amplitude/phase(Constituents=37, Station).
    Only M2 (slot 0) and S2 (slot 1) are non-zero; the other 35 slots are 0.
    """
    from netCDF4 import Dataset

    nsta = len(station_ids)
    nchar = 12
    with Dataset(path, "w") as ds:
        ds.createDimension("Station", nsta)
        ds.createDimension("Constituents", 37)
        ds.createDimension("staID", nchar)
        ds.createDimension("conLen", 4)

        ds.createVariable("stationID", "S1", ("Station", "staID"))[:] = _chararray(
            station_ids, nchar)

        names = ["M2", "S2"] + [f"C{i:02d}" for i in range(2, 37)]
        ds.createVariable("constituentName", "S1", ("Constituents", "conLen"))[:] = \
            _chararray(names, 4)

        amp = np.zeros((37, nsta), dtype="f8")
        pha = np.zeros((37, nsta), dtype="f8")
        amp[0, :] = m2[0]; pha[0, :] = m2[1]
        amp[1, :] = s2[0]; pha[1, :] = s2[1]
        ds.createVariable("amplitude", "f8", ("Constituents", "Station"))[:] = amp
        ds.createVariable("phase", "f8", ("Constituents", "Station"))[:] = pha


# ---------------------------------------------------------------------------
# The shared 3-station / 4-node control file
# ---------------------------------------------------------------------------
CTL_TEXT = """\
3 4 2 3600.0

# ---- correction station block ----
SID NOS_ID NWS_ID AGENCY DATUM WL_FLAG TS_FLAG BACKUP GRIDID_STA AS
1 STA_A NA NOS 0.0 0 0 2 1 1.0
2 STA_B NB NOS 0.0 0 0 3 2 2.0
3 STA_C NC NOS 0.0 0 0 0 3 0.5

# ---- open-boundary node block ----
GRIDID IOBC WL_STA WL_SID_1 WL_S_1 WL_SID_2 WL_S_2 TS_STA TS_SID_1 TS_S_1 TS_SID_2 TS_S_2
1 101 0 0 0.0 0 0.0 0 0 0.0 0 0.0
2 102 1 3 1.0 0 0.0 0 0 0.0 0 0.0
3 103 2 3 0.5 1 0.5 0 0 0.0 0 0.0
4 104 1 2 1.0 0 0.0 0 0 0.0 0 0.0

# ---- open-boundary element block ----
IDUMMY JOBC
1 201
2 202
"""


def _single_ctl(nos_id="STA_C", backup=0, as_scale=1.0):
    """A 1-station / 1-node control file for the focused numeric tests."""
    return (
        "1 1 0 3600.0\n\n"
        "h1\nh2\n"
        f"1 {nos_id} NC NOS 0.0 0 0 {backup} 1 {as_scale}\n\n"
        "h3\nh4\n"
        "1 101 1 1 1.0 0 0.0 0 0 0.0 0 0.0\n"
    )


# ---------------------------------------------------------------------------
# 1. ctl parse round-trip
# ---------------------------------------------------------------------------
def test_parse_obc_ctl_roundtrip(tmp_path):
    p = tmp_path / "obc.ctl"
    p.write_text(CTL_TEXT)
    ctl = parse_obc_ctl(p)

    assert (ctl.nsta, ctl.nobc, ctl.neobc, ctl.delt) == (3, 4, 2, 3600.0)
    assert len(ctl.stations) == 3 and len(ctl.nodes) == 4

    a, b, c = ctl.stations
    assert a.nos_id == "STA_A" and a.backup_sid == 2 and a.gridid_sta == 1
    assert a.wl_flag == 0 and a.as_scale == 1.0 and a.agency_id == "NOS"
    assert b.nos_id == "STA_B" and b.backup_sid == 3 and b.as_scale == 2.0
    assert c.nos_id == "STA_C" and c.backup_sid == 0 and c.as_scale == 0.5

    assert ctl.nodes[0].wl_sta == 0
    assert (ctl.nodes[1].wl_sta, ctl.nodes[1].wl_sid_1, ctl.nodes[1].wl_s_1) == (1, 3, 1.0)
    n2 = ctl.nodes[2]
    assert (n2.wl_sta, n2.wl_sid_1, n2.wl_s_1, n2.wl_sid_2, n2.wl_s_2) == (2, 3, 0.5, 1, 0.5)
    assert (ctl.nodes[3].wl_sta, ctl.nodes[3].wl_sid_1) == (1, 2)
    assert ctl.elements == [(1, 201), (2, 202)]


def test_parse_obc_ctl_malformed_header(tmp_path):
    p = tmp_path / "bad.ctl"
    p.write_text("3 4\nh1\nh2\n")
    with pytest.raises(ValueError):
        parse_obc_ctl(p)


# ---------------------------------------------------------------------------
# 2. subtidal computation SWL = OBS - tide_prediction (+ out-of-window sentinel)
# ---------------------------------------------------------------------------
def test_subtidal_and_sentinel(tmp_path):
    pytest.importorskip("netCDF4")
    hc = tmp_path / "hc.nc"
    _write_hc(hc, ["STA_C"])
    time_prd, _ = _grid()

    wl_prd = predict_station_tide(hc, "STA_C", time_prd, BASE_DATE)
    # obs sit on prediction-grid nodes -> exact 2-point interpolation
    idx = np.arange(48, 60)
    t_obs = np.append(time_prd[idx], 5.0)             # last one is outside window
    wl_obs = np.append(wl_prd[idx] + 0.25, 1.0)

    swl = _detide(t_obs, wl_obs, time_prd, wl_prd)
    assert np.allclose(swl[:-1], 0.25, atol=1e-9)     # OBS - PRED == known offset
    assert swl[-1] == -9999.0                          # obs outside prediction window


def test_read_station_hc_orientation(tmp_path):
    pytest.importorskip("netCDF4")
    hc = tmp_path / "hc.nc"
    _write_hc(hc, ["STA_A", "STA_B", "STA_C"], m2=(0.5, 30.0), s2=(0.2, 60.0))
    amp, pha = read_station_hc(hc, "STA_B")
    assert amp.shape == (37,) and pha.shape == (37,)
    assert amp[0] == 0.5 and pha[0] == 30.0
    assert amp[1] == 0.2 and pha[1] == 60.0
    assert np.all(amp[2:] == 0.0)
    with pytest.raises(KeyError):
        read_station_hc(hc, "NOPE")


# ---------------------------------------------------------------------------
# 3. AVGERR recovery + 6-hour ramp + residual interpolation onto zeta_time
# ---------------------------------------------------------------------------
def test_avgerr_ramp_and_residual(tmp_path):
    pytest.importorskip("netCDF4")
    hc = tmp_path / "hc.nc"
    _write_hc(hc, ["STA_C"])
    time_prd, zt = _grid()
    ntmax = zt.size

    # 25 hourly obs over [0, 1] day; subtidal designed as swl(t) = t (linear)
    idx = np.arange(48, 73)                            # t = 0 .. 1.0 day
    t_obs = time_prd[idx]
    wl_prd = predict_station_tide(hc, "STA_C", time_prd, BASE_DATE)
    swl_design = t_obs.copy()                          # linear residual
    wl_obs = wl_prd[idx] + swl_design

    ctl = parse_obc_ctl_from_text(_single_ctl("STA_C"))
    source = np.zeros((1, ntmax))                      # zero ETSS/RTOFS source
    corr = compute_wl_corrections(
        ctl, ArrayObsProvider({"STA_C": (t_obs, wl_obs)}),
        source, zt, BASE_DATE, DAY_END, hc_path=hc, day_start=DAY_START,
    )

    # mean of swl(t)=t over the 0..1 grid = 0.5  ->  AVGERR == 0.5
    assert np.isclose(corr.avgerr[0], 0.5, atol=1e-9)

    r = corr.resid[0]
    assert np.isclose(r[12], 0.0, atol=1e-9)           # zt=0.5  : t-0.5 = 0
    assert np.isclose(r[24], 0.5, atol=1e-9)           # zt=1.0  : t-0.5 = 0.5
    assert np.isclose(r[27], 0.25, atol=1e-9)          # zt=1.125: on the 6h ramp
    assert np.isclose(r[30], 0.0, atol=1e-9)           # zt=1.25 : ramp endpoint
    # everything past the ramp point (last obs + 6 h) is zeroed
    assert np.allclose(r[31:], 0.0, atol=1e-9)


# ---------------------------------------------------------------------------
# 4. NTMP <= 2 zeroing (too few obs survive the |SWL| <= 3 QC, zero source)
# ---------------------------------------------------------------------------
def test_ntmp_le2_zeroing(tmp_path):
    pytest.importorskip("netCDF4")
    hc = tmp_path / "hc.nc"
    _write_hc(hc, ["STA_C"])
    time_prd, zt = _grid()
    ntmax = zt.size

    idx = np.arange(48, 73)                            # 25 obs (> 20, no backup)
    t_obs = time_prd[idx]
    wl_prd = predict_station_tide(hc, "STA_C", time_prd, BASE_DATE)
    swl_design = np.full(idx.size, 10.0)               # |10| > 3 -> excluded by QC
    swl_design[5] = 0.0                                # only two survive |SWL|<=3
    swl_design[6] = 0.0
    wl_obs = wl_prd[idx] + swl_design

    ctl = parse_obc_ctl_from_text(_single_ctl("STA_C"))
    corr = compute_wl_corrections(
        ctl, {"STA_C": (t_obs, wl_obs)},
        np.zeros((1, ntmax)), zt, BASE_DATE, DAY_END, hc_path=hc, day_start=DAY_START,
    )
    assert corr.avgerr[0] == 0.0
    assert np.allclose(corr.resid[0], 0.0)


# ---------------------------------------------------------------------------
# 5. backup substitution (first hop + second hop) with AS scaling, and
#    weighted application on the 4-node toy boundary (WL_STA 0/1/2)
# ---------------------------------------------------------------------------
def test_backup_chain_and_application(tmp_path):
    pytest.importorskip("netCDF4")
    hc = tmp_path / "hc.nc"
    _write_hc(hc, ["STA_A", "STA_B", "STA_C"])
    time_prd, zt = _grid()
    ntmax = zt.size

    wl_prd = predict_station_tide(hc, "STA_C", time_prd, BASE_DATE)

    # STA_C: 48 good obs, constant subtidal offset 0.30
    idx_c = np.arange(48, 96)
    tc = time_prd[idx_c]
    wc = wl_prd[idx_c] + 0.30
    # STA_A / STA_B: only 5 obs each -> both fall back (A via 2nd hop, B via 1st hop)
    idx_ab = np.arange(48, 53)
    tab = time_prd[idx_ab]
    wab = wl_prd[idx_ab]                                # value irrelevant (replaced)

    ctl = parse_obc_ctl_from_text(CTL_TEXT)
    obs = ArrayObsProvider({"STA_A": (tab, wab), "STA_B": (tab, wab), "STA_C": (tc, wc)})
    corr = compute_wl_corrections(
        ctl, obs, np.zeros((3, ntmax)), zt, BASE_DATE, DAY_END,
        hc_path=hc, day_start=DAY_START,
    )

    # AVGERR: C = 0.30 ; A and B = AS(C) * 0.30 = 0.5 * 0.30 = 0.15
    assert np.isclose(corr.avgerr[2], 0.30, atol=1e-9)   # STA_C
    assert np.isclose(corr.avgerr[1], 0.15, atol=1e-9)   # STA_B via first backup
    assert np.isclose(corr.avgerr[0], 0.15, atol=1e-9)   # STA_A via second backup hop
    assert np.allclose(corr.resid, 0.0, atol=1e-9)       # constant offset -> no residual

    # apply to a zero WLOBC over the 4 boundary nodes
    wlobc = np.zeros((4, ntmax))
    out = apply_wl_corrections(wlobc, ctl, corr)
    assert np.allclose(out[0], 0.0)                       # WL_STA=0 untouched
    assert np.allclose(out[1], 0.30)                      # 1.0*(0.30)
    assert np.allclose(out[2], 0.225)                     # 0.5*0.30 + 0.5*0.15
    assert np.allclose(out[3], 0.15)                      # 1.0*(0.15) via STA_B
    # original input not mutated
    assert np.allclose(wlobc, 0.0)


def test_wl_sta_zero_is_untouched(tmp_path):
    """A WL_STA==0 node is never modified even with non-trivial corrections."""
    corr = WlCorrections(
        avgerr=np.array([1.0]),
        resid=np.zeros((1, 3)),
        zeta_time_days=np.arange(3.0),
    )
    ctl = ObcCtl(
        nsta=1, nobc=1, neobc=0, delt=3600.0,
        stations=[CorrectionStation(1, "S", "S", "NOS", 0.0, 0, 0, 0, 1, 1.0)],
        nodes=[BoundaryNode(1, 1, 0, 1, 1.0, 0, 0.0, 0, 0, 0.0, 0, 0.0)],
    )
    wlobc = np.full((1, 3), 7.0)
    out = apply_wl_corrections(wlobc, ctl, corr)
    assert np.allclose(out, 7.0)


# ---------------------------------------------------------------------------
# helper: parse a ctl held in a string
# ---------------------------------------------------------------------------
def parse_obc_ctl_from_text(text):
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".ctl", delete=False) as fh:
        fh.write(text)
        name = fh.name
    return parse_obc_ctl(name)
