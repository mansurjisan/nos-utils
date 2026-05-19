"""Parity tests for the GFS pre-extraction pruning + single-pass extract.

These lock the invariant behind the performance fix: pruning the GFS file
list *before* decoding (and one wgrib2 pass per file for all variables) must
produce the exact same (valid_time -> source file) set, variable/level set,
and ordering as the old "decode everything, then dedup + window-filter"
pipeline. No real wgrib2 is required (mirrors the other GFS tests).
"""

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from nos_utils.config import ForcingConfig
from nos_utils.forcing.gfs import GFSProcessor
from nos_utils.io.grib_extract import GRIBExtractor, Wgrib2Extractor


def _make_multicycle_gfs_dir(root: Path, pdy: str) -> Path:
    """Create a multi-cycle GFS tree with heavily overlapping leads.

    Several 6-hourly cycles across the run window, each carrying long
    forecast leads, so valid times are produced by multiple cycles — the
    exact multi-cycle duplicate situation the prune must collapse.
    """
    gfs_root = root / "gfs_data"
    base = datetime.strptime(pdy, "%Y%m%d")
    # Cover the previous day + current day, 6-hourly cycles.
    for day_off in (-1, 0):
        day = base + timedelta(days=day_off)
        ds = day.strftime("%Y%m%d")
        for cyc in (0, 6, 12, 18):
            atmos = gfs_root / f"gfs.{ds}" / f"{cyc:02d}" / "atmos"
            atmos.mkdir(parents=True, exist_ok=True)
            # Long leads so cycles overlap one another substantially.
            for fhr in range(0, 60):
                f = atmos / f"gfs.t{cyc:02d}z.pgrb2.0p25.f{fhr:03d}"
                f.write_bytes(b"\x00" * 1024)
    return gfs_root


def _old_kept_set(proc: GFSProcessor, gfs_files):
    """Reference reimplementation of the OLD post-extraction selection.

    Mirrors exactly:
      * _extract_all: stable sort by parsed valid time, then keep-first
        dedup per valid time (earliest in discovery order wins);
      * _filter_to_time_window: drop valid times outside _get_time_window().

    Returns an ordered list of (valid_time, Path) — what the old pipeline
    would have written, derived independently of the new prune method.
    """
    parsed = []
    for f in gfs_files:
        # Same parser the old _extract_all used inline.
        try:
            fhr = int(f.name.split(".f")[-1])
            cyc_hour = int(f.name.split(".t")[1][:2])
            date_str = proc.config.pdy
            for parent in [f.parent, f.parent.parent, f.parent.parent.parent]:
                if parent.name.startswith("gfs."):
                    date_str = parent.name.split("gfs.")[1][:8]
                    break
            vt = datetime.strptime(date_str, "%Y%m%d") + timedelta(
                hours=cyc_hour
            ) + timedelta(hours=fhr)
        except (ValueError, IndexError):
            continue
        parsed.append((vt, f))

    order = sorted(range(len(parsed)), key=lambda i: parsed[i][0])
    seen = {}
    for i in order:
        vt = parsed[i][0]
        if vt not in seen:
            seen[vt] = i
    unique_idx = sorted(seen.values())

    t_start, t_end = proc._get_time_window()
    return [
        parsed[i] for i in unique_idx if t_start <= parsed[i][0] <= t_end
    ]


@pytest.fixture
def stofs_proc(tmp_path):
    """GFSProcessor over a realistic multi-cycle STOFS-3D-ATL nowcast."""
    pdy = "20260401"
    gfs_root = _make_multicycle_gfs_dir(tmp_path, pdy)
    cfg = ForcingConfig.for_stofs_3d_atl(pdy=pdy, cyc=12)
    proc = GFSProcessor(cfg, gfs_root, tmp_path / "out", phase="nowcast")
    proc.MIN_FILE_SIZE = 0  # mock files are tiny
    return proc


class TestPruneParity:
    def test_discovery_has_multicycle_overlap(self, stofs_proc):
        """Sanity: discovery really does over-collect (overlap exists)."""
        discovered = stofs_proc.find_input_files()
        old_final = _old_kept_set(stofs_proc, discovered)
        # The whole point of the fix: far more decoded than kept.
        assert len(discovered) > len(old_final)
        assert len(old_final) > 0

    def test_pruned_set_equals_old_post_dedup_window_set(self, stofs_proc):
        """NEW pre-extraction kept set == OLD post-dedup/post-window set.

        Same valid times, same chosen file per valid time, same order.
        """
        discovered = stofs_proc.find_input_files()
        old_final = _old_kept_set(stofs_proc, discovered)

        new_files = stofs_proc._select_files_for_window(discovered)
        new_final = [
            (stofs_proc._parse_valid_time(f), f) for f in new_files
        ]

        # Same number of kept entries.
        assert len(new_final) == len(old_final)
        # Same ordered (valid_time, file) sequence — file chosen per
        # valid time and emission order both match exactly.
        assert new_final == old_final
        # Same set of valid times, strictly increasing (deduped).
        new_times = [vt for vt, _ in new_final]
        assert new_times == sorted(new_times)
        assert len(set(new_times)) == len(new_times)
        assert new_times == [vt for vt, _ in old_final]

    def test_every_dropped_file_is_dup_or_out_of_window(self, stofs_proc):
        """Pruned-out files are exactly superseded dups / out-of-window.

        Nothing with a unique in-window valid time may be dropped.
        """
        discovered = stofs_proc.find_input_files()
        kept = set(stofs_proc._select_files_for_window(discovered))
        t_start, t_end = stofs_proc._get_time_window()

        kept_times = {
            stofs_proc._parse_valid_time(f) for f in kept
        }
        for f in discovered:
            if f in kept:
                continue
            vt = stofs_proc._parse_valid_time(f)
            in_window = vt is not None and t_start <= vt <= t_end
            # A dropped file is acceptable only if it is out of window OR
            # its valid time is still represented by the kept winner.
            assert (not in_window) or (vt in kept_times)

    def test_extract_all_only_sees_pruned_files(self, stofs_proc):
        """_extract_all decodes only files that survive to the kept set.

        Uses a recording mock extractor (no wgrib2): assert every file
        handed to extract_many is in the pruned set and the per-file
        valid times equal the pruned valid times.
        """
        discovered = stofs_proc.find_input_files()
        kept = stofs_proc._select_files_for_window(discovered)
        kept_set = set(kept)

        seen_files = []

        mock = MagicMock(spec=GRIBExtractor)
        mock.get_grid.return_value = (
            np.linspace(-80, -70, 4), np.linspace(25, 35, 4)
        )

        def _fake_extract_many(grib_file, var_levels, domain):
            seen_files.append(Path(grib_file))
            return {vl: np.zeros((4, 4), np.float32) for vl in var_levels}

        mock.extract_many.side_effect = _fake_extract_many
        stofs_proc._extractor = mock

        result = stofs_proc._extract_all(kept)

        # Every decoded file is in the pruned set, nothing extra.
        assert seen_files == kept
        assert all(f in kept_set for f in seen_files)
        # Times produced equal the pruned valid times, in order.
        assert result["times"] == [
            stofs_proc._parse_valid_time(f) for f in kept
        ]


class TestCombinedMatchParity:
    def test_combined_match_selects_same_var_set(self):
        """P3: combined -match regex == union of per-variable patterns.

        The set of VAR:LEVEL tokens in the combined regex must equal the
        set the old per-(file x variable) loop matched (one
        ``:VAR:LEVEL:`` per variable), no more and no fewer.
        """
        # Old behavior: one match string per variable.
        old_tokens = {
            f"{grib_var}:{level}"
            for grib_var, level in GFSProcessor.GRIB2_VARIABLES.values()
        }

        var_levels = list(GFSProcessor.GRIB2_VARIABLES.values())
        combined = Wgrib2Extractor._build_combined_match(var_levels)

        # Shape: ":(A:LA|B:LB|...):"
        assert combined.startswith(":(")
        assert combined.endswith("):")
        new_tokens = set(combined[2:-2].split("|"))

        assert new_tokens == old_tokens

    def test_combined_match_default_variable_subset(self):
        """Default 8-var sflux set: combined regex tokens match exactly."""
        proc_vars = GFSProcessor.DEFAULT_VARIABLES
        var_levels = [
            GFSProcessor.GRIB2_VARIABLES[v] for v in proc_vars
        ]
        old_tokens = {f"{gv}:{lv}" for gv, lv in var_levels}

        combined = Wgrib2Extractor._build_combined_match(var_levels)
        new_tokens = set(combined[2:-2].split("|"))

        assert new_tokens == old_tokens
        assert len(new_tokens) == len(proc_vars)

    def test_combined_match_dedups_repeated_pairs(self):
        """Repeated (var, level) pairs collapse (e.g. TMP at two levels).

        GRIB2_VARIABLES maps both 'stmp' and 'wtmp' to TMP but at
        different levels, and 'wtmp' shares ('TMP','surface') with none —
        ensure identical pairs are emitted once while distinct
        (var, level) pairs are all preserved.
        """
        var_levels = [
            ("UGRD", "10 m above ground"),
            ("UGRD", "10 m above ground"),  # exact duplicate
            ("TMP", "2 m above ground"),
            ("TMP", "surface"),  # same var, different level -> kept
        ]
        combined = Wgrib2Extractor._build_combined_match(var_levels)
        tokens = combined[2:-2].split("|")

        assert tokens == [
            "UGRD:10 m above ground",
            "TMP:2 m above ground",
            "TMP:surface",
        ]


class TestExtractManyFallbackParity:
    def test_base_extract_many_matches_per_variable_extract(self):
        """Default extract_many == per-variable extract (fallback parity).

        Backends that do not override extract_many (e.g. CfgribExtractor)
        must behave exactly like the old per-variable loop: one extract()
        call per (var, level) with identical arguments, same result map.
        """

        class _Rec(GRIBExtractor):
            def __init__(self):
                self.calls = []

            def extract(self, grib_file, variable, level, domain):
                self.calls.append((Path(grib_file), variable, level, domain))
                return np.full((2, 2), len(self.calls), np.float32)

            def get_grid(self, grib_file, domain):  # pragma: no cover
                return np.zeros(2), np.zeros(2)

        rec = _Rec()
        domain = (-80.0, -70.0, 25.0, 35.0)
        var_levels = [
            ("UGRD", "10 m above ground"),
            ("VGRD", "10 m above ground"),
            ("PRMSL", "mean sea level"),
        ]
        out = rec.extract_many(Path("/x/gfs.t12z.pgrb2.0p25.f001"),
                               var_levels, domain)

        # Exactly one extract() per pair, same args, same order.
        assert [(c[1], c[2]) for c in rec.calls] == var_levels
        assert all(c[3] == domain for c in rec.calls)
        # Result keyed by (var, level), values are what extract returned.
        assert set(out.keys()) == set(var_levels)
        for i, vl in enumerate(var_levels, start=1):
            assert np.array_equal(out[vl], np.full((2, 2), i, np.float32))
