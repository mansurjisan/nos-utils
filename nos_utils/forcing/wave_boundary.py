"""
GFS-Wave boundary spectra forcing processor.

Builds WW3 boundary conditions (``nest.ww3``) for coupled SCHISM+WW3 runs
(STOFS-3D-AK) from NCEP's operational GFS-Wave point-spectra product.

Product layout (``$COMINgfswave``)::

    gfs.<YYYYMMDD>/<CC>/wave/station/
        gfswave.t<CC>z.ibp_tar        -- tar of ``./gfswave.<NAME>.spec``
                                          for every NCEP boundary point
                                          (TYPE == IBP in the points file)
        gfswave.t<CC>z.spec_tar.gz    -- gzipped tar of spectra for the
                                          full station list, including
                                          points that are NOT in ibp_tar
                                          (e.g. NDBC DAT buoys)

Each ``.spec`` file carries the full 384h hourly series from its GFS cycle
time, so a cycle older than the OFS's own doesn't need to be re-picked once
found -- it still covers the nowcast+forecast window.

Pipeline:
  1. Discovery -- newest gfs cycle (00/06/12/18) at or before the OFS
     nowcast start with BOTH tar files present, walking back up to
     ``max_cycle_fallback`` cycles.
  2. Point selection -- TYPE == IBP points inside a configurable lon/lat
     window (dateline-safe), plus an explicit extra-points name list for
     points that live only in ``spec_tar.gz`` (STOFS-3D-AK: the NDBC DAT
     buoys 46070/46071/46035, which sit in the Bering Sea / Aleutians and
     are not carried in the operational IBP list).
  3. Extraction -- pulls only the needed members from each tar.
  4. Emission -- writes ``ww3_bound.inp`` (READ mode) listing the
     extracted spectra, then runs ``ww3_bound`` to produce ``nest.ww3``.
     If the executable isn't found, the inputs are still emitted and the
     processor reports failure with an actionable message; the
     orchestrator's ``critical_sources`` gating decides whether that's
     fatal for the overall prep.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import ForcingConfig
from ..coords import normalize_lon
from .base import ForcingProcessor, ForcingResult

log = logging.getLogger(__name__)

# Fortran tool search, mirrored from TidalProcessor._call_fortran_tide_fac.
_WW3_BOUND_EXE_NAMES = ["ww3_bound", "stofs_3d_ak_ww3_bound"]
_EXEC_ENV_DIRS = ["EXECnos", "EXECofs", "EXECstofs3d"]

_CYCLE_HOURS = (0, 6, 12, 18)

# wave_gfs.buoys line:
#   lon lat 'NAME      '  depth  TYPE  SOURCE  interval
_POINT_LINE_RE = re.compile(
    r"^\s*([+-]?\d+(?:\.\d+)?)\s+([+-]?\d+(?:\.\d+)?)\s+'([^']*)'\s+"
    r"(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*$"
)


@dataclass
class WaveBoundaryPoint:
    """One entry from a WW3 points file (``wave_gfs.buoys`` format)."""

    name: str
    lon: float
    lat: float
    point_type: str
    depth: Optional[float] = None
    source: str = ""
    interval: Optional[float] = None


def parse_ww3_points_file(path: "Path | str") -> List[WaveBoundaryPoint]:
    """Parse a WW3 points file in ``wave_gfs.buoys`` format.

    Lines are::

        lon  lat  'NAME      '  depth  TYPE  SOURCE  interval

    Blank lines and comment lines (starting with ``$``, ignoring leading
    whitespace) are skipped. Lines that don't match the expected shape are
    skipped rather than raising -- the real product file carries section
    marker comments (e.g. ``$AGPN48``) that aren't always prefixed cleanly.
    """
    points: List[WaveBoundaryPoint] = []
    text = Path(path).read_text()
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("$"):
            continue
        m = _POINT_LINE_RE.match(line)
        if not m:
            continue
        lon_s, lat_s, name_s, depth_s, type_s, source_s, interval_s = m.groups()
        try:
            depth = float(depth_s)
        except ValueError:
            depth = None
        try:
            interval = float(interval_s)
        except ValueError:
            interval = None
        points.append(
            WaveBoundaryPoint(
                name=name_s.strip(),
                lon=float(lon_s),
                lat=float(lat_s),
                point_type=type_s.strip().upper(),
                depth=depth,
                source=source_s.strip(),
                interval=interval,
            )
        )
    return points


def _point_in_window(
    lon: float, lat: float,
    lon_min: float, lon_max: float, lat_min: float, lat_max: float,
) -> bool:
    """Dateline-safe box test.

    Longitudes are normalized to 0-360 (via ``nos_utils.coords``) before
    comparing, so a window authored either as -180/180 or 0-360 works, and
    a window that crosses the international dateline (e.g. lon_min=170,
    lon_max=-155 in -180/180 terms) is handled correctly: normalizing to
    0-360 makes that span contiguous (170 to 205) rather than wrapping.
    The wraparound branch below only matters for the (unusual, for this
    package's domains) case of a window that itself straddles the 0/360
    seam, i.e. crosses the prime meridian.
    """
    if lat < lat_min or lat > lat_max:
        return False
    lon360 = float(normalize_lon(np.array([lon]), "0360")[0])
    lo360 = float(normalize_lon(np.array([lon_min]), "0360")[0])
    hi360 = float(normalize_lon(np.array([lon_max]), "0360")[0])
    if lo360 <= hi360:
        return lo360 <= lon360 <= hi360
    return lon360 >= lo360 or lon360 <= hi360


def _snap_cycle(dt: datetime) -> Tuple[datetime, int]:
    """Floor *dt* to the most recent 00/06/12/18Z GFS-Wave cycle."""
    hour = dt.hour - (dt.hour % 6)
    date = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    return date, hour


class WaveBoundaryProcessor(ForcingProcessor):
    """Build WW3 ``nest.ww3`` boundary conditions from GFS-Wave spectra."""

    SOURCE_NAME = "WAVE_BC"

    IBP_TAR_TEMPLATE = "gfswave.t{cyc:02d}z.ibp_tar"
    SPEC_TARGZ_TEMPLATE = "gfswave.t{cyc:02d}z.spec_tar.gz"

    def __init__(
        self,
        config: ForcingConfig,
        input_path: Path,
        output_path: Path,
        *,
        points_file: Optional[Path] = None,
        window: Optional[dict] = None,
        extra_points: Optional[List[str]] = None,
        max_cycle_fallback: int = 4,
        mod_def: Optional[Path] = None,
    ) -> None:
        """
        Args:
            config: ForcingConfig (uses pdy, cyc, nowcast_hours).
            input_path: COMINgfswave root (parent of ``gfs.<YYYYMMDD>/``).
            output_path: Working directory for extracted specs, inp, and
                nest.ww3.
            points_file: WW3 points file (``wave_gfs.buoys`` format).
            window: ``{lon_min, lon_max, lat_min, lat_max}`` selection box
                for TYPE == IBP points. All four keys must be present and
                non-None for window selection to run; otherwise it's
                skipped (extra_points still apply).
            extra_points: Point names to include regardless of TYPE/window
                (pulled from ``spec_tar.gz``, not ``ibp_tar``).
            max_cycle_fallback: Extra 6h cycles to try walking backward if
                the newest cycle at/before the nowcast start is missing
                either tar.
            mod_def: WW3 model definition file (``mod_def.ww3``). ww3_bound
                reads this from its working directory, not a command-line
                argument, so it's staged into ``output_path`` before
                ww3_bound is invoked.
        """
        super().__init__(config, input_path, output_path)
        self.points_file = Path(points_file) if points_file else None
        self.window = dict(window) if window else {}
        self.extra_points = [
            str(p).strip() for p in (extra_points or []) if str(p).strip()
        ]
        self.max_cycle_fallback = max(0, int(max_cycle_fallback))
        self.mod_def = Path(mod_def) if mod_def else None
        # None = not yet searched; False = searched, nothing found;
        # tuple = (cycle_date, cycle_hour, ibp_tar, spec_targz).
        self._cycle_cache = None
        self._cycles_checked = 0

    # ------------------------------------------------------------------ API

    def process(self) -> ForcingResult:
        log.info(
            f"Wave boundary processor: pdy={self.config.pdy} "
            f"cyc={self.config.cyc:02d}z"
        )
        self.create_output_dir()

        warnings: List[str] = []
        errors: List[str] = []
        metadata: Dict = {}

        found = self._find_cycle()
        if found is None:
            msg = (
                f"No GFS-Wave cycle with both ibp_tar and spec_tar.gz found "
                f"within {self.max_cycle_fallback + 1} cycle(s) back from "
                f"{self._nowcast_start():%Y%m%d%H} under {self.input_path}"
            )
            log.warning(msg)
            return ForcingResult(
                success=False, source=self.SOURCE_NAME, errors=[msg],
                metadata={"cycles_checked": self._cycles_checked},
            )

        cyc_date, cyc_hour, ibp_tar, spec_targz = found
        metadata.update({
            "cycle_pdy": cyc_date.strftime("%Y%m%d"),
            "cycle_hour": cyc_hour,
            "cycles_walked_back": self._cycles_checked - 1,
        })

        from ._log import log_input_files
        log_input_files(
            self.SOURCE_NAME, [ibp_tar, spec_targz],
            source="WAVE_BC", category="waves",
            note=(
                f"pdy={self.config.pdy} cyc={self.config.cyc:02d} "
                f"chosen_cycle={cyc_date:%Y%m%d}{cyc_hour:02d}"
            ),
        )

        ibp_names, extra_names, warnings = self._select_points(warnings)
        metadata["n_points_ibp"] = len(ibp_names)
        metadata["n_points_extra"] = len(extra_names)

        if not ibp_names and not extra_names:
            errors.append(
                "No wave boundary points selected (empty window match and "
                "no extra_points)"
            )
            return ForcingResult(
                success=False, source=self.SOURCE_NAME, errors=errors,
                warnings=warnings, metadata=metadata,
            )

        # ww3_bound reads mod_def.ww3 from its working directory (no
        # command-line argument for it), so it must be staged before
        # ww3_bound runs. Checked here -- before the tar extraction below --
        # so a missing mod_def fails immediately instead of after minutes
        # spent extracting spectra that ww3_bound will never get to use.
        mod_def_error = self._stage_mod_def()
        if mod_def_error:
            log.warning(mod_def_error)
            errors.append(mod_def_error)
            return ForcingResult(
                success=False, source=self.SOURCE_NAME, errors=errors,
                warnings=warnings, metadata=metadata,
            )

        extracted, missing_points = self._extract_points(
            ibp_tar, spec_targz, ibp_names, extra_names,
        )
        if missing_points:
            warnings.append(
                f"points not found in tar listings: {sorted(missing_points)}"
            )
        metadata["n_extracted"] = len(extracted)
        metadata["spec_files"] = [p.name for p in extracted]

        if not extracted:
            errors.append(
                "No spec files could be extracted from ibp_tar/spec_tar.gz"
            )
            return ForcingResult(
                success=False, source=self.SOURCE_NAME, errors=errors,
                warnings=warnings, metadata=metadata,
            )

        inp_path = self._write_ww3_bound_inp(extracted)
        output_files = [inp_path] + extracted

        exe = self._find_ww3_bound_exe()
        if exe is None:
            msg = (
                "ww3_bound executable not found in EXECnos/EXECofs/"
                "EXECstofs3d -- emitted ww3_bound.inp and extracted "
                f"{len(extracted)} spec file(s), but nest.ww3 was not "
                "generated. Build/stage ww3_bound to complete WW3 "
                "boundary prep."
            )
            log.warning(msg)
            errors.append(msg)
            metadata["mode"] = "missing_executable"
            return ForcingResult(
                success=False, source=self.SOURCE_NAME,
                output_files=output_files, errors=errors,
                warnings=warnings, metadata=metadata,
            )

        nest_path = self.output_path / "nest.ww3"
        try:
            result = subprocess.run(
                [str(exe)], cwd=str(self.output_path),
                capture_output=True, text=True, timeout=300,
            )
        except (OSError, subprocess.TimeoutExpired) as e:
            errors.append(f"Error running ww3_bound: {e}")
            metadata["mode"] = "execution_error"
            return ForcingResult(
                success=False, source=self.SOURCE_NAME,
                output_files=output_files, errors=errors,
                warnings=warnings, metadata=metadata,
            )

        if result.returncode != 0 or not nest_path.exists():
            errors.append(
                f"ww3_bound returned {result.returncode} or did not "
                f"produce nest.ww3: {result.stderr[:300]}"
            )
            metadata["mode"] = "ww3_bound_failed"
            return ForcingResult(
                success=False, source=self.SOURCE_NAME,
                output_files=output_files, errors=errors,
                warnings=warnings, metadata=metadata,
            )

        output_files.append(nest_path)
        metadata["mode"] = "ww3_bound"
        log.info(f"Wrote nest.ww3 from {len(extracted)} boundary spectra")
        return ForcingResult(
            success=True, source=self.SOURCE_NAME,
            output_files=output_files, warnings=warnings, metadata=metadata,
        )

    def find_input_files(self) -> List[Path]:
        """The ibp_tar / spec_tar.gz for the chosen cycle (empty if none found)."""
        found = self._find_cycle()
        if found is None:
            return []
        _, _, ibp_tar, spec_targz = found
        return [ibp_tar, spec_targz]

    # ------------------------------------------------------------ discovery

    def _nowcast_start(self) -> datetime:
        cycle_dt = datetime.strptime(self.config.pdy, "%Y%m%d") + \
            timedelta(hours=self.config.cyc)
        return cycle_dt - timedelta(hours=self.config.nowcast_hours)

    def _cycle_dir(self, date: datetime, cyc: int) -> Path:
        return (
            self.input_path / f"gfs.{date.strftime('%Y%m%d')}"
            / f"{cyc:02d}" / "wave" / "station"
        )

    def _find_cycle(self):
        """Newest gfs-wave cycle at/before the nowcast start with both tars.

        Walks backward in 6h steps up to ``max_cycle_fallback`` additional
        cycles. Result is cached on the instance so repeated calls (e.g.
        ``process()`` then ``find_input_files()``) don't re-walk the tree.
        """
        if self._cycle_cache is not None:
            return self._cycle_cache if self._cycle_cache is not False else None

        start_date, start_cyc = _snap_cycle(self._nowcast_start())
        t = start_date + timedelta(hours=start_cyc)

        for i in range(self.max_cycle_fallback + 1):
            date, cyc = _snap_cycle(t)
            self._cycles_checked = i + 1
            cyc_dir = self._cycle_dir(date, cyc)
            ibp = cyc_dir / self.IBP_TAR_TEMPLATE.format(cyc=cyc)
            spec = cyc_dir / self.SPEC_TARGZ_TEMPLATE.format(cyc=cyc)
            if ibp.exists() and spec.exists():
                self._cycle_cache = (date, cyc, ibp, spec)
                return self._cycle_cache
            t -= timedelta(hours=6)

        self._cycle_cache = False
        return None

    # -------------------------------------------------------- point select

    def _select_points(
        self, warnings: List[str],
    ) -> Tuple[List[str], List[str], List[str]]:
        """Return (ibp_names, extra_names, warnings)."""
        ibp_names: List[str] = []

        if self.points_file and self.points_file.exists():
            points = parse_ww3_points_file(self.points_file)
            win = self.window
            required = ("lon_min", "lon_max", "lat_min", "lat_max")
            if all(win.get(k) is not None for k in required):
                for p in points:
                    if p.point_type != "IBP":
                        continue
                    if _point_in_window(
                        p.lon, p.lat,
                        win["lon_min"], win["lon_max"],
                        win["lat_min"], win["lat_max"],
                    ):
                        ibp_names.append(p.name)
            else:
                warnings.append(
                    "wave boundary window incomplete -- skipping IBP "
                    "window selection"
                )
        elif self.points_file:
            warnings.append(f"points_file not found: {self.points_file}")
        else:
            warnings.append(
                "no points_file configured -- relying on extra_points only"
            )

        extra_names = [n for n in self.extra_points if n not in ibp_names]
        return ibp_names, extra_names, warnings

    # --------------------------------------------------------- extraction

    def _extract_points(
        self,
        ibp_tar: Path,
        spec_targz: Path,
        ibp_names: List[str],
        extra_names: List[str],
    ) -> Tuple[List[Path], List[str]]:
        extracted: List[Path] = []
        missing: List[str] = []

        if ibp_names:
            members = self._list_tar_members(ibp_tar)
            matched = []
            for name in ibp_names:
                m = self._match_member(members, name)
                if m is None:
                    missing.append(name)
                else:
                    matched.append(m)
            extracted.extend(
                self._extract_members(ibp_tar, matched, self.output_path)
            )

        if extra_names:
            members = self._list_tar_members(spec_targz, gz=True)
            matched = []
            for name in extra_names:
                m = self._match_member(members, name)
                if m is None:
                    missing.append(name)
                else:
                    matched.append(m)
            extracted.extend(
                self._extract_members(
                    spec_targz, matched, self.output_path, gz=True,
                )
            )

        return extracted, missing

    @staticmethod
    def _match_member(members: List[str], name: str) -> Optional[str]:
        """Find *name*'s tar-root member, tolerating a ``./`` prefix.

        Only an exact match against ``target`` or ``./target`` is accepted.
        Nested-directory members (``subdir/target``) and anything reached
        via ``..`` segments (``../target``) are rejected: _extract_members
        assumes extracted files land directly in dest_dir, and a member
        spelled with ``..`` must never be treated as equivalent to the
        bare target name.
        """
        target = f"gfswave.{name}.spec"
        member_set = set(members)
        for candidate in (target, f"./{target}"):
            if candidate in member_set:
                return candidate
        return None

    @staticmethod
    def _list_tar_members(tar_path: Path, gz: bool = False) -> List[str]:
        flag = "-tzf" if gz else "-tf"
        try:
            result = subprocess.run(
                ["tar", flag, str(tar_path)],
                capture_output=True, text=True, timeout=60,
            )
        except (OSError, subprocess.TimeoutExpired) as e:
            log.warning(f"Failed to list {tar_path}: {e}")
            return []
        if result.returncode != 0:
            log.warning(f"tar list failed for {tar_path}: {result.stderr[:200]}")
            return []
        return [ln.strip() for ln in result.stdout.splitlines() if ln.strip()]

    @staticmethod
    def _extract_members(
        tar_path: Path, members: List[str], dest_dir: Path, gz: bool = False,
    ) -> List[Path]:
        """Extract *members* from *tar_path* in a single tar invocation.

        *members* must already be exact archive-listed spellings (as
        returned by ``_match_member`` against ``_list_tar_members``), so
        every name is expected to exist -- a nonzero exit is logged as an
        actionable warning rather than raised, and whichever members did
        land on disk are still collected and returned.
        """
        if not members:
            return []
        flag = "-xzf" if gz else "-xf"
        try:
            result = subprocess.run(
                ["tar", flag, str(tar_path), "-C", str(dest_dir), *members],
                capture_output=True, text=True, timeout=120,
            )
        except (OSError, subprocess.TimeoutExpired) as e:
            log.warning(f"Failed to extract {members} from {tar_path}: {e}")
            return []
        if result.returncode != 0:
            log.warning(
                f"tar extract failed for {members} from {tar_path}: "
                f"{result.stderr[:200]}"
            )
        extracted: List[Path] = []
        for member in members:
            out_path = dest_dir / Path(member).name
            if out_path.exists():
                extracted.append(out_path)
            else:
                log.warning(f"Extracted {member} but {out_path} not found")
        return extracted

    # ----------------------------------------------------------- mod_def

    def _stage_mod_def(self) -> Optional[str]:
        """Stage mod_def.ww3 into ``output_path`` for ww3_bound to find.

        ww3_bound reads ``mod_def.ww3`` from its current working directory
        -- there's no command-line argument for it -- and ``process()`` runs
        it with ``cwd=self.output_path``. Symlinks by default (the file is
        typically hundreds of MB); falls back to a copy if symlinking isn't
        available (e.g. some Windows/test filesystems).

        Returns an error message if ``self.mod_def`` isn't set or doesn't
        exist on disk, None on success.
        """
        if self.mod_def is None or not self.mod_def.exists():
            return (
                f"WW3 mod_def not found: {self.mod_def or '<not configured>'} "
                "-- ww3_bound reads mod_def.ww3 from its working directory "
                "and cannot run without it. Set forcing.waves.mod_def (or "
                "stage the default '{RUN}.mod_def.ww3' under FIXofs)."
            )
        dest = self.output_path / "mod_def.ww3"
        if dest.exists() or dest.is_symlink():
            return None
        try:
            dest.symlink_to(self.mod_def)
        except OSError:
            shutil.copy2(self.mod_def, dest)
        return None

    # ------------------------------------------------------------ emission

    def _write_ww3_bound_inp(self, spec_files: List[Path]) -> Path:
        """Write ww3_bound.inp in READ mode listing the extracted spectra.

        Format follows the classic single-grid WAVEWATCH III ww3_bound
        input: a quoted 'READ'/'WRITE' mode line, followed by one quoted
        spectra filename per line, terminated by a '$' line. ww3_bound is
        run with cwd set to ``self.output_path`` (which holds this file and
        the extracted spec files), so filenames are written bare (no path).
        """
        inp_path = self.output_path / "ww3_bound.inp"
        lines = [
            "$ WAVEWATCH III Boundary input file",
            "$ -------------------------------------------------",
            "$ Boundary option: 'READ' reads existing spectra for boundary input.",
            "$",
            "   'READ'",
            "$",
            "$ List of spectra files, terminated by a line starting with '$'.",
            "$",
        ]
        for f in spec_files:
            lines.append(f"'{f.name}'")
        lines.append("$")
        lines.append("$ -------------------------------------------------")
        lines.append("$ End of input file")
        lines.append("$ -------------------------------------------------")
        inp_path.write_text("\n".join(lines) + "\n")
        return inp_path

    # --------------------------------------------------------- executable

    @staticmethod
    def _find_ww3_bound_exe() -> Optional[Path]:
        """Search EXECnos/EXECofs/EXECstofs3d for the ww3_bound executable.

        Mirrors TidalProcessor._call_fortran_tide_fac's search order.
        """
        for env_var in _EXEC_ENV_DIRS:
            exec_dir = os.environ.get(env_var)
            if not exec_dir:
                continue
            for name in _WW3_BOUND_EXE_NAMES:
                candidate = Path(exec_dir) / name
                if candidate.exists():
                    return candidate
        return None
