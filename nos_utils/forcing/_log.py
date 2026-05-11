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
"""
import logging
from pathlib import Path
from typing import Iterable, Optional, Union

_log = logging.getLogger("nos_utils.forcing.inputs")

PathLike = Union[str, Path]


def log_input_files(
    processor: str,
    files: Iterable[PathLike],
    *,
    note: Optional[str] = None,
) -> None:
    """Emit an ``[INPUTS]`` log line for a processor's discovered files.

    Always emits the summary (count + first + last) at INFO so the
    operational log shows the data-tank consumption summary inline.
    When the file list is non-empty, also emits each path at DEBUG
    so ``--verbose`` runs get full per-file traceability.

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


__all__ = ["log_input_files"]
