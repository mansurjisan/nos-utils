"""Tests for the per-stage input-file manifest (prep)."""

import json
import threading

import pytest

from nos_utils.forcing._log import (
    drain_input_capture,
    log_input_files,
    reset_input_capture,
    start_input_capture,
)


@pytest.fixture(autouse=True)
def _clean_capture():
    """Each test starts and ends with a disarmed, empty collector."""
    reset_input_capture()
    yield
    reset_input_capture()


class TestCaptureGrouping:
    def test_grouped_by_category_and_source(self):
        start_input_capture()
        log_input_files(
            "GFS", ["/a/gfs.f000.nc", "/a/gfs.f001.nc", "/a/gfs.f002.nc"],
            source="GFS", category="atmospheric",
        )
        log_input_files("RTOFS", ["/b/rtofs_3d.nc"], source="RTOFS", category="ocean")
        log_input_files("NWM", ["/c/nwm.tm00.nc"], source="NWM", category="river")
        log_input_files("TIDAL", ["/d/bctides.in_template"],
                        source="TIDAL", category="tidal")
        log_input_files("HOTSTART", ["/e/rst.nowcast.nc"],
                        source="HOTSTART", category="hotstart")
        entries = drain_input_capture()

        keyed = {(e["category"], e["source"]): e for e in entries}
        assert ("atmospheric", "GFS") in keyed
        assert ("ocean", "RTOFS") in keyed
        assert ("river", "NWM") in keyed
        assert ("tidal", "TIDAL") in keyed
        assert ("hotstart", "HOTSTART") in keyed

        gfs = keyed[("atmospheric", "GFS")]
        assert gfs["count"] == 3
        assert gfs["files"] == [
            "/a/gfs.f000.nc", "/a/gfs.f001.nc", "/a/gfs.f002.nc",
        ]

    def test_files_are_strings_full_paths(self):
        from pathlib import Path

        start_input_capture()
        log_input_files("GFS", [Path("/a/gfs.f000.nc"), Path("/a/gfs.f001.nc")],
                        source="GFS", category="atmospheric")
        entries = drain_input_capture()
        files = entries[0]["files"]
        assert all(isinstance(f, str) for f in files)
        assert files == ["/a/gfs.f000.nc", "/a/gfs.f001.nc"]

    def test_no_checksum_size_or_mtime_keys(self):
        start_input_capture()
        log_input_files("GFS", ["/a/gfs.f000.nc"], source="GFS", category="atmospheric")
        entries = drain_input_capture()
        for e in entries:
            assert set(e.keys()) == {"category", "source", "count", "files"}
            assert "checksum" not in e
            assert "size" not in e
            assert "mtime" not in e

    def test_same_source_merges_files(self):
        start_input_capture()
        log_input_files("RTOFS", ["/b/2d.nc"], source="RTOFS", category="ocean")
        log_input_files("RTOFS", ["/b/3d.nc"], source="RTOFS", category="ocean")
        entries = drain_input_capture()
        rtofs = [e for e in entries if e["source"] == "RTOFS"]
        assert len(rtofs) == 1
        assert rtofs[0]["count"] == 2
        assert rtofs[0]["files"] == ["/b/2d.nc", "/b/3d.nc"]

    def test_category_defaults_from_processor_map(self):
        # No explicit category/source: filled from _PROCESSOR_CATEGORY.
        start_input_capture()
        log_input_files("HRRR", ["/a/hrrr.nc"])
        entries = drain_input_capture()
        assert entries[0]["category"] == "atmospheric"
        assert entries[0]["source"] == "HRRR"

    def test_stable_order(self):
        start_input_capture()
        log_input_files("TIDAL", ["/d/b.in"], source="TIDAL", category="tidal")
        log_input_files("GFS", ["/a/g.nc"], source="GFS", category="atmospheric")
        log_input_files("NWM", ["/c/n.nc"], source="NWM", category="river")
        entries = drain_input_capture()
        keys = [(e["category"], e["source"]) for e in entries]
        assert keys == sorted(keys)


class TestBackwardCompat:
    def test_two_arg_call_still_works(self):
        # Old 2-arg signature must not error.
        log_input_files("GFS", ["/a/gfs.f000.nc", "/a/gfs.f001.nc"])

    def test_unarmed_collects_nothing(self):
        # Capture not armed: nothing collected, no error.
        log_input_files("GFS", ["/a/gfs.f000.nc"], source="GFS")
        # Draining a disarmed collector returns an empty list.
        assert drain_input_capture() == []

    def test_empty_file_list(self):
        start_input_capture()
        log_input_files("GFS", [], source="GFS", category="atmospheric")
        entries = drain_input_capture()
        assert len(entries) == 1
        assert entries[0]["count"] == 0
        assert entries[0]["files"] == []


class TestThreadSafety:
    def test_concurrent_logging_no_lost_entries(self):
        start_input_capture()

        def worker(idx):
            log_input_files(
                "GFS", [f"/a/gfs.{idx}.nc"],
                source=f"SRC{idx}", category="atmospheric",
            )

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        entries = drain_input_capture()
        # One group per source, all 16 present, no exception raised.
        assert len(entries) == 16
        total_files = sum(e["count"] for e in entries)
        assert total_files == 16


class TestManifestWrite:
    def test_write_inputs_manifest_valid_json(self, tmp_path):
        from nos_utils.config import ForcingConfig
        from nos_utils.orchestrator import PrepOrchestrator, PrepResult

        config = ForcingConfig.for_secofs(pdy="20260226", cyc=12)
        orch = PrepOrchestrator(
            config,
            paths={"output": str(tmp_path / "work")},
            run_name="stofs_3d_atl_ufs",
        )

        start_input_capture()
        log_input_files(
            "GFS", ["/a/gfs.f000.nc", "/a/gfs.f001.nc"],
            source="GFS", category="atmospheric",
        )
        log_input_files("RTOFS", ["/b/3d.nc"], source="RTOFS", category="ocean")

        result = PrepResult(success=True, phase="forecast")
        comout = tmp_path / "comout"
        comout.mkdir()
        manifest_path = orch._write_inputs_manifest(result, comout)

        assert manifest_path is not None
        assert manifest_path.name == "stofs_3d_atl_ufs.t12z.20260226.inputs.prep.json"

        data = json.loads(manifest_path.read_text())
        assert data["ofs"] == "stofs_3d_atl_ufs"
        assert data["pdy"] == "20260226"
        assert data["cyc"] == "12"
        assert data["stage"] == "prep"
        assert data["phase"] == "forecast"
        assert data["schema_version"] == 1
        assert "generated_at" in data
        assert isinstance(data["inputs"], list)

        keyed = {(e["category"], e["source"]): e for e in data["inputs"]}
        assert keyed[("atmospheric", "GFS")]["count"] == 2
        assert keyed[("ocean", "RTOFS")]["count"] == 1
        # No per-file metadata leaked into the manifest.
        for e in data["inputs"]:
            assert set(e.keys()) == {"category", "source", "count", "files"}
