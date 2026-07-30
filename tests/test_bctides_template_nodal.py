"""Nodal-factor rewriting in a bctides.in template.

A bctides.in names each constituent once per section, and only the column
count distinguishes what follows it:

    tidal potential    5 cols  species amp freq f V0+u
    boundary forcing   3 cols  freq f V0+u
    per-node elevation 2 cols  amp pha
    per-node velocity  4 cols  u_amp u_pha v_amp v_pha

Only the first two carry cycle-dependent nodal parameters. The per-node
blocks are harmonic constants of the mesh and must survive untouched.
"""
from pathlib import Path

import pytest

from nos_utils.forcing.tidal import TidalProcessor


# One elevation-forced boundary with 2 nodes, iettype/ifltype 5, so the
# template exercises all four line shapes for a single constituent.
_TEMPLATE = """\
!2019-07-01 12:00:00 UTC
2 50.0 !ntip, cutoff depth
M2
 2 0.242334 0.000140519 1.00958 21.5386
S2
 2 0.112841 0.000145444 1.0 0.0
2 !nbfr
M2
 0.000140519 1.00958 21.5386
S2
 0.000145444 1.0 0.0
1 !nope
2 5 5 4 4 !test boundary
M2
 0.307520  168.103381
 0.302088  166.298557
S2
 0.100000  100.000000
 0.110000  110.000000
M2
 0.385391  306.426197  0.455913  151.518474
 0.380000  300.000000  0.450000  150.000000
S2
 0.108138  339.219848  0.146274  189.256472
 0.100000  330.000000  0.140000  180.000000
"""


def _run(tmp_path, mock_config, text=_TEMPLATE):
    tpl = tmp_path / "x.bctides.in_template"
    tpl.write_text(text)
    out = tmp_path / "bctides.in"
    proc = TidalProcessor(mock_config, tmp_path, tmp_path)
    assert proc._process_template(tpl, out) is True
    return text.splitlines(), out.read_text().splitlines()


def _field(lines, idx, col):
    return float(lines[idx].split()[col])


class TestNodalRewrite:
    def test_line_count_is_preserved(self, tmp_path, mock_config):
        before, after = _run(tmp_path, mock_config)
        assert len(before) == len(after)

    def test_potential_section_is_rewritten(self, tmp_path, mock_config):
        """5-column line: f and V0+u are columns 3 and 4.

        These used to be computed and then dropped -- the assignment back
        into the line was missing, so the output kept the template's own
        factors under a rewritten date.
        """
        before, after = _run(tmp_path, mock_config)
        assert after[3].split()[:3] == before[3].split()[:3]   # species/amp/freq intact
        assert _field(after, 3, 3) != _field(before, 3, 3)     # f updated
        assert _field(after, 3, 4) != _field(before, 3, 4)     # V0+u updated

    def test_boundary_forcing_section_is_rewritten(self, tmp_path, mock_config):
        """3-column line: f and V0+u are columns 1 and 2.

        This section drives the elevation boundary and was not handled at
        all -- neither the 5- nor the 4-column branch matched it.
        """
        before, after = _run(tmp_path, mock_config)
        assert _field(after, 8, 0) == _field(before, 8, 0)     # frequency intact
        assert _field(after, 8, 1) != _field(before, 8, 1)
        assert _field(after, 8, 2) != _field(before, 8, 2)

    def test_potential_and_boundary_agree_for_a_constituent(self, tmp_path, mock_config):
        """Same constituent, same f and V0+u, wherever it appears."""
        _before, after = _run(tmp_path, mock_config)
        assert _field(after, 3, 3) == _field(after, 8, 1)
        assert _field(after, 3, 4) == _field(after, 8, 2)


class TestPerNodeDataIsNeverTouched:
    # The regression that matters: per-node harmonics belong to the mesh,
    # not the cycle.

    def test_two_column_elevation_nodes_untouched(self, tmp_path, mock_config):
        before, after = _run(tmp_path, mock_config)
        for idx in (14, 15, 17, 18):
            assert after[idx] == before[idx], f"elevation node line {idx} rewritten"

    def test_four_column_velocity_nodes_untouched(self, tmp_path, mock_config):
        """4 columns is u_amp u_pha v_amp v_pha, not a short potential line.

        Treating it as the latter overwrote the v-component amplitude and
        phase of the first node of every constituent on every ifltype 4/5
        boundary with a nodal factor.
        """
        before, after = _run(tmp_path, mock_config)
        for idx in (20, 21, 23, 24):
            assert after[idx] == before[idx], f"velocity node line {idx} rewritten"

    def test_no_line_gains_or_loses_columns(self, tmp_path, mock_config):
        before, after = _run(tmp_path, mock_config)
        for i, (b, a) in enumerate(zip(before, after)):
            if i == 0:
                continue  # the date line is reformatted by design
            assert len(b.split()) == len(a.split()), f"column count changed on line {i}"


class TestUnmatchedTemplateIsReported:
    def test_warns_when_nothing_matched(self, tmp_path, mock_config, caplog):
        """A template whose constituents never match must not pass silently
        with the template's own (stale) factors."""
        text = "!2019-07-01 12:00:00 UTC\n0 50.0\n0 !nbfr\n0 !nope\n"
        with caplog.at_level("WARNING"):
            _run(tmp_path, mock_config, text=text)
        assert any("no nodal parameter line matched" in r.message for r in caplog.records)
