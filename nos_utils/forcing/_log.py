"""Structured input-file logging for forcing processors.

For data-tank traceability: when a forcing processor consumes files
from a remote/cache data tank, log a structured one-line summary so
operators can correlate input availability to output artifacts when
debugging issues. The NWM back-fill bug fixed in commit 1a925dd was an
example — if the per-cycle file list had been visible in the prep log,
the analysis_assim coverage gap would have been obvious from the start
instead of requiring a full diagnostic agent pass.

Each ``[INPUTS]`` line carries enough information to:
  - confirm WHICH cycle's data was used (path embeds PDY/CYC for most
    NOAA products)
  - confirm HOW MANY files were ingested (count)
  - GREP for the processor name (NWM / GFS / HRRR / RTOFS / ...)
  - identify the file-set extremes (first/last paths) without dumping
    every path at INFO level

The full list is emitted at DEBUG (``--verbose``) so noisy operational
prod logs stay clean while still being available for post-mortem.

Format::

    [INPUTS] processor=NWM count=72 first=/lfs/h1/cache/com/nwm/v3.0/nwm.20260507/analysis_assim/nwm.t00z.analysis_assim.channel_rt.tm02.conus.nc last=/lfs/h1/cache/com/nwm/v3.0/nwm.20260507/short_range/nwm.t00z.short_range.channel_rt.f067.conus.nc note=pdy=20260507 cyc=00 product=mixed reaches=3522

Capture collector
-----------------
``log_input_files`` doubles as the data source for the per-stage input
manifest the prep orchestrator writes to ``$COMOUT``. When capture is
armed (:func:`start_input_capture`) every call also records the file
list, grouped by ``(category, source)``, behind a lock so the
ThreadPoolExecutor-driven GFS/HRRR/NWM/tidal processors can log
concurrently without losing entries. :func:`drain_input_capture`
returns the merged groups and disarms the collector.
"""
import logging
import threading
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

_log = logging.getLogger("nos_utils.forcing.inputs")

PathLike = Union[str, Path]

# Map a processor / source name to the manifest category it belongs to.
# Used to fill ``category`` when a caller doesn't pass one explicitly.
_PROCESSOR_CATEGORY = {
    "GFS": "atmospheric",
    "HRRR": "atmospheric",
    "RTOFS": "ocean",
    "ADT": "ocean",
    "DYNAMIC_ADJUST": "ocean",
    "NWM": "river",
    "ST_LAWRENCE": "river",
    "TIDAL": "tidal",
    "HOTSTART": "hotstart",
    "NUDGING": "nudging",
}

# Capture state. ``_capture_armed`` gates the collection so existing
# callers outside an armed orchestrator run pay nothing. The lock guards
# both the flag and the buffer because forcing processors log from worker
# threads.
_capture_lock = threading.Lock()
_capture_armed = False
# (category, source) -> ordered list of path strings.
_capture: Dict[Tuple[str, str], List[str]] = {}


def log_input_files(
    processor: str,
    files: Iterable[PathLike],
    *,
    note: Optional[str] = None,
    category: Optional[str] = None,
    source: Optional[str] = None,
) -> None:
    """Emit an ``[INPUTS]`` log line for a processor's discovered files.

    Always emits the summary (count + first + last) at INFO so the
    operational log shows the data-tank consumption summary inline.
    When the file list is non-empty, also emits each path at DEBUG
    so ``--verbose`` runs get full per-file traceability.

    When input capture is armed (see :func:`start_input_capture`) the
    file list is additionally recorded for the prep input manifest,
    grouped by ``(category, source)``.

    Args:
        processor: Short name (e.g., ``"NWM"``, ``"GFS"``, ``"RTOFS"``,
            ``"HRRR"``, ``"HOTSTART"``). Becomes the grep target.
        files: Iterable of paths the processor consumed. ``None`` /
            empty iterable is OK — emits ``count=0`` and
            ``first=<none>`` / ``last=<none>``.
        note: Free-form key=value pairs separated by spaces, appended
            verbatim. Use to surface cycle / product / count of
            downstream consumers — anything that helps locate the
            issue when the log gets grep'd later.
        category: Manifest category for these files. Defaults to the
            ``_PROCESSOR_CATEGORY`` mapping for ``processor`` (or
            ``"other"`` if unmapped). Only affects the captured manifest.
        source: Manifest source label. Defaults to ``processor``. Only
            affects the captured manifest.
    """
    paths = [str(p) for p in files]
    n = len(paths)
    first = paths[0] if paths else "<none>"
    last = paths[-1] if paths else "<none>"
    parts = [
        "[INPUTS]",
        f"processor={processor}",
        f"count={n}",
        f"first={first}",
        f"last={last}",
    ]
    if note:
        parts.append(f"note={note}")
    _log.info(" ".join(parts))
    for p in paths:
        _log.debug("  [INPUT] %s %s", processor, p)

    with _capture_lock:
        if not _capture_armed:
            return
        cat = category or _PROCESSOR_CATEGORY.get(processor, "other")
        src = source or processor
        bucket = _capture.setdefault((cat, src), [])
        bucket.extend(paths)


def start_input_capture() -> None:
    """Arm the capture collector and clear any prior state."""
    global _capture_armed
    with _capture_lock:
        _capture.clear()
        _capture_armed = True


def reset_input_capture() -> None:
    """Clear the capture collector and disarm it."""
    global _capture_armed
    with _capture_lock:
        _capture.clear()
        _capture_armed = False


def drain_input_capture() -> List[dict]:
    """Return grouped capture entries and disarm the collector.

    Each entry is ``{"category", "source", "count", "files"}`` for one
    ``(category, source)`` pair, with files merged in the order they were
    logged. Groups are returned in a stable order (by category then
    source) so the manifest is deterministic.
    """
    global _capture_armed
    with _capture_lock:
        entries = [
            {
                "category": cat,
                "source": src,
                "count": len(files),
                "files": list(files),
            }
            for (cat, src), files in sorted(_capture.items())
        ]
        _capture.clear()
        _capture_armed = False
    return entries


__all__ = [
    "log_input_files",
    "start_input_capture",
    "reset_input_capture",
    "drain_input_capture",
]
