"""ADCIRC tide parameters: equivalence with Zach Cobell's TideFac / Tide (in-process)."""

import importlib
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from nos_utils.forcing import adcirc_tides as at

ZACH_USH = Path(os.environ.get(
    "ZACH_STOFS_USH",
    "/mnt/d/NOS-Workflow-Project/STOFS_2DGLO_P0/zach_repo/ush"))

CONSTS_15 = ["K1", "O1", "P1", "Q1", "M2", "S2", "N2", "K2",
             "MF", "MM", "M4", "MS4", "MN4", "SA", "SSA"]

DATES = [
    datetime(2025, 1, 2, 3),
    datetime(2026, 2, 28, 18, 30),
    datetime(2026, 10, 5, 12),
    datetime(2024, 12, 31, 23),
    datetime(2027, 6, 15, 0, 0, 7),
]


@pytest.fixture(scope="module")
def zach_tide():
    if not (ZACH_USH / "StofsWorkflow" / "models" / "adcirc" / "tide.py").exists():
        pytest.skip("Zach's clone not available (set ZACH_STOFS_USH)")
    sys.path.insert(0, str(ZACH_USH))
    try:
        mod = importlib.import_module("StofsWorkflow.models.adcirc.tide")
    except Exception as exc:  # his code needs a newer Python than 3.8
        sys.path.remove(str(ZACH_USH))
        pytest.skip("cannot import his Tide: {}".format(exc))
    yield mod
    sys.path.remove(str(ZACH_USH))


@pytest.mark.parametrize("start", DATES)
@pytest.mark.parametrize("days", [0.25, 6.0, 18.5, 24.0, 365.0])
def test_identical_to_zach(zach_tide, start, days):
    end = start + timedelta(days=days)
    ref = zach_tide.Tide(CONSTS_15, start, end, 0.0).tides()
    got = at.compute_adcirc_tides(CONSTS_15, start, (end - start).total_seconds() / 86400.0)
    assert [c.name for c in got] == list(ref.keys())
    for c in got:
        r = ref[c.name]
        assert c.tpk == r["amplitude"]
        assert c.frequency == r["frequency"]
        assert c.etrf == r["earth_tide_reduction_factor"]
        assert c.nodal_factor == r["node_factor"]
        assert c.equilibrium_arg_deg == r["phase"]


def test_all_37_identical(zach_tide):
    names = zach_tide.Tide.available_tide_constituents()
    start = datetime(2026, 10, 5, 6)
    ref = zach_tide.Tide(names, start, start + timedelta(days=7), 0.0).tides()
    got = at.compute_adcirc_tides(names, start, 7.0)
    assert len(got) == 37
    for c in got:
        assert (c.nodal_factor, c.equilibrium_arg_deg) == (
            ref[c.name]["node_factor"], ref[c.name]["phase"])


def test_start_reference_equals_zero_length():
    start = datetime(2026, 10, 5, 12)
    a = at.compute_adcirc_tides(["M2", "K1"], start, 30.0, nodal_reference="start")
    b = at.compute_adcirc_tides(["M2", "K1"], start, 0.0)
    assert a == b


def test_bad_reference_and_unknown_constituent(caplog):
    with pytest.raises(ValueError, match="nodal_reference"):
        at.compute_adcirc_tides(["M2"], datetime(2026, 1, 1), 1.0, nodal_reference="x")
    got = at.compute_adcirc_tides(["m2", "ZZ9", "M2"], datetime(2026, 1, 1), 1.0)
    assert [c.name for c in got] == ["M2"]


def test_fort15_line_format_matches_his_writer():
    c = at.compute_adcirc_tides(["M2"], datetime(2026, 1, 1), 2.0)[0]
    pot = c.potential_lines()
    assert pot[0] == "M2"
    assert pot[1] == "{:0.5f} {:0.15f} {:0.3f} {:0.5f} {:0.2f}".format(
        c.tpk, c.frequency, c.etrf, c.nodal_factor, c.equilibrium_arg_deg)
    assert c.boundary_freq_lines()[1] == "{:0.15f} {:0.5f} {:0.2f}".format(
        c.frequency, c.nodal_factor, c.equilibrium_arg_deg)
