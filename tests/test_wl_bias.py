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
    apply_wl_bias_to_ssh,
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


# ---------------------------------------------------------------------------
# 6. Wiring: apply_wl_bias_to_ssh (what rtofs.py::_process_2d calls)
#
# The elev2D array is (nt, n_bnd) on the model output time grid with the axis
# in SECONDS since model_t0 — the same axis written to elev2D.th.nc.  Here
# model_t0 = BASE_DATE and the axis is hourly over 0..2 days, so the internal
# zeta_time reproduces the `_grid()` convention used by the tests above.
# ---------------------------------------------------------------------------
TIMES_SECONDS = np.arange(49) * 3600.0     # 0 .. 2 days, matches _grid()'s zt

# NSTA=0 / every WL_STA=0 — the operational SECOFS configuration.
CTL_TEXT_NSTA0 = """\
0 2 0 120.0
SECTION 1: WATER LEVEL INFORMATION
SID NOS_ID NWS_ID AGENCY DATUM WL_FLAG TS_FLAG BACKUP GRIDID_STA AS

SECTION 2: CONFIGURATION OF LATERAL OPEN BOUNDARY
GRIDID IOBC WL_STA WL_SID_1 WL_S_1 WL_SID_2 WL_S_2 TS_STA TS_SID_1 TS_S_1 TS_SID_2 TS_S_2
1 101 0 0 0.0 0 0.0 0 0 0.0 0 0.0
2 102 0 0 0.0 0 0.0 0 0 0.0 0 0.0
"""

# Station 1 points at a non-existent boundary node; station 2 is valid.
CTL_TEXT_BAD_GRIDID = """\
2 2 0 3600.0

h1
h2
1 STA_A NA NOS 0.0 0 0 0 99 1.0
2 STA_C NC NOS 0.0 0 0 0 2 1.0

h3
h4
1 101 1 1 1.0 0 0.0 0 0 0.0 0 0.0
2 102 1 2 1.0 0 0.0 0 0 0.0 0 0.0
"""


def _constant_offset_obs(hc, offset=0.30, station="STA_C"):
    """48 hourly obs whose subtidal is a constant ``offset`` (AVGERR recovery)."""
    time_prd, _ = _grid()
    idx = np.arange(48, 96)
    wl_prd = predict_station_tide(hc, station, time_prd, BASE_DATE)
    return time_prd[idx], wl_prd[idx] + offset


def test_wiring_nsta0_ctl_is_a_true_noop(tmp_path):
    """Operational config (NSTA=0): array untouched, no HC/obs even consulted."""
    ctl = tmp_path / "secofs.obc.ctl"
    ctl.write_text(CTL_TEXT_NSTA0)
    ssh = np.arange(49 * 2, dtype=np.float32).reshape(49, 2)

    out, info = apply_wl_bias_to_ssh(
        ssh, TIMES_SECONDS, BASE_DATE, ctl, hc_file=None, obs=None,
    )

    assert out is ssh                       # same object — nothing copied
    assert info["applied"] is False
    assert "no correction stations" in info["reason"]
    assert info["n_nodes_corrected"] == 0


def test_wiring_applies_correction_from_station_node(tmp_path):
    """Hand-computed: AVGERR = obs_subtidal - WLOBC(GRIDID_STA)."""
    pytest.importorskip("netCDF4")
    hc = tmp_path / "hc.nc"
    _write_hc(hc, ["STA_C"])
    ctl = tmp_path / "one.ctl"
    ctl.write_text(_single_ctl("STA_C"))

    t_obs, wl_obs = _constant_offset_obs(hc, offset=0.30)
    ssh = np.full((49, 1), 0.10, dtype=np.float32)   # source series = 0.10 m

    out, info = apply_wl_bias_to_ssh(
        ssh, TIMES_SECONDS, BASE_DATE, ctl, hc,
        obs={"STA_C": (t_obs, wl_obs)},
    )

    # AVGERR = 0.30 - 0.10 = 0.20  ->  corrected = 0.10 + 0.20 = 0.30
    assert info["applied"] is True
    assert np.isclose(info["avgerr"]["STA_C"], 0.20, atol=1e-6)
    assert np.allclose(out, 0.30, atol=1e-6)
    assert out.dtype == ssh.dtype
    assert info["n_nodes_corrected"] == 1
    assert np.allclose(ssh, 0.10)                    # input not mutated


def test_wiring_weighted_nodes_on_the_toy_boundary(tmp_path):
    """The 3-station / 4-node ctl: WL_STA 0/1/2 weighting through the wrapper."""
    pytest.importorskip("netCDF4")
    hc = tmp_path / "hc.nc"
    _write_hc(hc, ["STA_A", "STA_B", "STA_C"])
    ctl = tmp_path / "toy.ctl"
    ctl.write_text(CTL_TEXT)

    tc, wc = _constant_offset_obs(hc, offset=0.30)
    time_prd, _ = _grid()
    idx_ab = np.arange(48, 53)                        # 5 obs -> both back up to C
    tab = time_prd[idx_ab]
    wab = predict_station_tide(hc, "STA_C", time_prd, BASE_DATE)[idx_ab]

    ssh = np.zeros((49, 4), dtype=np.float32)
    out, info = apply_wl_bias_to_ssh(
        ssh, TIMES_SECONDS, BASE_DATE, ctl, hc,
        obs=ArrayObsProvider({"STA_A": (tab, wab), "STA_B": (tab, wab),
                              "STA_C": (tc, wc)}),
    )

    assert info["applied"] is True
    assert np.allclose(out[:, 0], 0.0)                # WL_STA=0 untouched
    assert np.allclose(out[:, 1], 0.30, atol=1e-6)    # 1.0*AVGERR(C)
    assert np.allclose(out[:, 2], 0.225, atol=1e-6)   # 0.5*0.30 + 0.5*0.15
    assert np.allclose(out[:, 3], 0.15, atol=1e-6)    # 1.0*AVGERR(B via backup)
    assert info["n_nodes_corrected"] == 3
    assert info["n_stations"] == 3


def test_wiring_missing_hc_file_is_a_warning_not_a_failure(tmp_path):
    pytest.importorskip("netCDF4")
    hc = tmp_path / "hc.nc"
    _write_hc(hc, ["STA_C"])
    ctl = tmp_path / "one.ctl"
    ctl.write_text(_single_ctl("STA_C"))
    t_obs, wl_obs = _constant_offset_obs(hc, offset=0.30)
    ssh = np.zeros((49, 1), dtype=np.float32)

    out, info = apply_wl_bias_to_ssh(
        ssh, TIMES_SECONDS, BASE_DATE, ctl, tmp_path / "absent.nc",
        obs={"STA_C": (t_obs, wl_obs)},
    )
    assert out is ssh
    assert info["applied"] is False
    assert "harmonic-constants file not found" in info["reason"]


def test_wiring_provider_exception_is_swallowed(tmp_path):
    pytest.importorskip("netCDF4")
    hc = tmp_path / "hc.nc"
    _write_hc(hc, ["STA_C"])
    ctl = tmp_path / "one.ctl"
    ctl.write_text(_single_ctl("STA_C"))
    ssh = np.zeros((49, 1), dtype=np.float32)

    class _Boom:
        def get(self, station, base_date=None, day_start=None, day_end=None):
            raise RuntimeError("gauge tank late")

    out, info = apply_wl_bias_to_ssh(
        ssh, TIMES_SECONDS, BASE_DATE, ctl, hc, obs=_Boom(),
    )
    assert out is ssh
    assert info["applied"] is False
    assert "gauge tank late" in info["reason"]


def test_wiring_missing_ctl_and_node_count_mismatch(tmp_path):
    pytest.importorskip("netCDF4")
    hc = tmp_path / "hc.nc"
    _write_hc(hc, ["STA_C"])
    ssh = np.zeros((49, 1), dtype=np.float32)

    out, info = apply_wl_bias_to_ssh(
        ssh, TIMES_SECONDS, BASE_DATE, tmp_path / "nope.ctl", hc, obs={},
    )
    assert out is ssh and "OBC control file not found" in info["reason"]

    ctl = tmp_path / "toy.ctl"
    ctl.write_text(CTL_TEXT)                          # NOBC=4 vs 1 SSH column
    out, info = apply_wl_bias_to_ssh(
        ssh, TIMES_SECONDS, BASE_DATE, ctl, hc, obs={},
    )
    assert out is ssh and "NOBC=4" in info["reason"]


def test_wiring_out_of_range_gridid_skips_only_that_station(tmp_path):
    """STA_A's GRIDID_STA is out of range; STA_C is still applied.

    The HC file deliberately holds ONLY STA_C: if the skipped station were
    still processed the (wl_flag=0) HC lookup would raise and the whole
    correction would be dropped.
    """
    pytest.importorskip("netCDF4")
    hc = tmp_path / "hc.nc"
    _write_hc(hc, ["STA_C"])
    ctl = tmp_path / "bad_gridid.ctl"
    ctl.write_text(CTL_TEXT_BAD_GRIDID)

    tc, wc = _constant_offset_obs(hc, offset=0.30)
    ssh = np.zeros((49, 2), dtype=np.float32)

    out, info = apply_wl_bias_to_ssh(
        ssh, TIMES_SECONDS, BASE_DATE, ctl, hc,
        obs={"STA_A": (tc, wc), "STA_C": (tc, wc)},
    )

    assert info["applied"] is True
    assert info["skipped_stations"] == ["STA_A"]
    assert np.allclose(out[:, 0], 0.0)                 # node fed by STA_A: no-op
    assert np.allclose(out[:, 1], 0.30, atol=1e-6)     # node fed by STA_C
    assert info["n_nodes_corrected"] == 1
    assert set(info["avgerr"]) == {"STA_C"}


# ---------------------------------------------------------------------------
# 7. Wiring: the RTOFSProcessor config surface + FIX lookup
# ---------------------------------------------------------------------------
def _rtofs_proc(tmp_path, **cfg_overrides):
    from nos_utils.config import ForcingConfig
    from nos_utils.forcing.rtofs import RTOFSProcessor

    cfg = ForcingConfig(
        lon_min=-80.0, lon_max=-70.0, lat_min=25.0, lat_max=35.0,
        pdy="20260401", cyc=12, **cfg_overrides,
    )
    return RTOFSProcessor(cfg, tmp_path / "in", tmp_path / "out")


def test_processor_wl_bias_disabled_by_default(tmp_path):
    """Feature off => hook returns immediately, array untouched."""
    proc = _rtofs_proc(tmp_path)
    ssh = np.zeros((49, 1), dtype=np.float32)
    out, info = proc._apply_wl_bias(ssh, TIMES_SECONDS, BASE_DATE)
    assert out is ssh
    assert info == {"applied": False, "reason": "wl_bias_enabled=False"}


def test_processor_resolves_ctl_hc_and_file_obs_from_fix(tmp_path, monkeypatch):
    """Enabled + FIX layout + file observations: end-to-end, no network."""
    pytest.importorskip("netCDF4")
    fix = tmp_path / "fix"
    fix.mkdir()
    _write_hc(fix / "nosofs.HC_NWLON.nc", ["STA_C"])
    (fix / "secofs.obc.ctl").write_text(_single_ctl("STA_C"))

    obs_dir = tmp_path / "obs"
    obs_dir.mkdir()
    t_obs, wl_obs = _constant_offset_obs(fix / "nosofs.HC_NWLON.nc", offset=0.30)
    np.savetxt(obs_dir / "STA_C.obs", np.column_stack([t_obs, wl_obs]))

    monkeypatch.setenv("FIXofs", str(fix))
    proc = _rtofs_proc(
        tmp_path,
        wl_bias_enabled=True,
        wl_bias_obs_source="file",
        wl_bias_obs_dir=obs_dir,
    )
    assert proc._resolve_wl_bias_ctl() == fix / "secofs.obc.ctl"
    assert proc._resolve_wl_bias_hc() == fix / "nosofs.HC_NWLON.nc"

    ssh = np.full((49, 1), 0.10, dtype=np.float32)
    out, info = proc._apply_wl_bias(ssh, TIMES_SECONDS, BASE_DATE)

    assert info["applied"] is True
    assert np.isclose(info["avgerr"]["STA_C"], 0.20, atol=1e-6)
    assert np.allclose(out, 0.30, atol=1e-6)


def test_processor_obs_source_none_never_reads_gauges(tmp_path, monkeypatch):
    """Default obs source is "none": nothing is fetched, correction skipped."""
    pytest.importorskip("netCDF4")
    fix = tmp_path / "fix"
    fix.mkdir()
    _write_hc(fix / "nosofs.HC_NWLON.nc", ["STA_C"])
    (fix / "secofs.obc.ctl").write_text(_single_ctl("STA_C"))
    monkeypatch.setenv("FIXofs", str(fix))

    proc = _rtofs_proc(tmp_path, wl_bias_enabled=True)
    assert proc._build_wl_obs_provider() is None

    ssh = np.zeros((49, 1), dtype=np.float32)
    out, info = proc._apply_wl_bias(ssh, TIMES_SECONDS, BASE_DATE)
    assert out is ssh
    assert info["applied"] is False
    assert "no observation provider" in info["reason"]


def test_prediction_epoch_is_jan1_not_base_date(tmp_path):
    """Nodal terms are frozen at Jan 1 of the year (ops equarg(IYRS,1,1) +
    jbase_date convention), NOT re-evaluated at the cycle time.

    Implementation-independent check: two calls whose (base_date, times_days)
    pairs denote the SAME absolute instants must agree exactly.  That holds
    only when f/V0+u are anchored to a fixed yearly epoch; anchoring them to
    base_date makes the two disagree (~mm, measurable at this tolerance).
    """
    hc = tmp_path / "hc.nc"
    _write_hc(hc, ["STA_C"])

    a_base = datetime(2026, 7, 22, 6, 0)
    b_base = datetime(2026, 7, 20, 6, 0)          # 2 days earlier
    t_a = np.array([0.0, 0.25, 0.5, 1.0])
    t_b = t_a + 2.0                                # same absolute instants

    pred_a = predict_station_tide(hc, "STA_C", t_a, a_base)
    pred_b = predict_station_tide(hc, "STA_C", t_b, b_base)

    np.testing.assert_allclose(pred_a, pred_b, atol=1e-12)
