"""
SCHISM hotstart/restart file processor.

Reads SCHISM hotstart.nc to extract:
  - Model time (for computing rnday and time_hotstart)
  - Time step counter (iths)
  - Basic validation (file size, key variables present)

Searches for hotstart files from previous cycles with automatic date fallback.

Replaces: nos_ofs_read_restart (Fortran executable)

Key SCHISM hotstart.nc variables:
  time   — scalar: model time in seconds
  iths   — scalar: time step counter
  eta2   — [node]: surface elevation
  tr_nd  — [node, nVert, ntracers]: tracers at nodes
"""

import json
import logging
import os
import re
import shutil
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

from ..config import ForcingConfig
from .base import ForcingProcessor, ForcingResult

log = logging.getLogger(__name__)

try:
    from netCDF4 import Dataset
    HAS_NETCDF4 = True
except ImportError:
    HAS_NETCDF4 = False


class HotstartStagingError(RuntimeError):
    """Raised by :meth:`HotstartProcessor.stage_init_to_comout` when the
    seed it must PRESERVE at the staged-init target is not
    NETCDF4_CLASSIC.

    The seed is left untouched -- this is never raised in a way that
    destroys an operator-provided file. It exists so that condition turns
    into a hard prep failure instead of a swallowed ``log.error()`` line:
    a non-CLASSIC seed is guaranteed to fail or segfault SCHISM's
    parallel-IO staging at rank scale, so prep must not report success
    with such a seed in place.
    """


class HotstartInfo:
    """Information extracted from a SCHISM hotstart.nc file."""

    def __init__(
        self,
        filepath: Path,
        time_seconds: float,
        iths: int,
        n_nodes: int,
        n_levels: int,
    ):
        self.filepath = filepath
        self.time_seconds = time_seconds
        self.iths = iths
        self.n_nodes = n_nodes
        self.n_levels = n_levels

    @property
    def time_days(self) -> float:
        return self.time_seconds / 86400.0

    def __repr__(self):
        return (f"HotstartInfo(file={self.filepath.name}, "
                f"time={self.time_seconds:.0f}s ({self.time_days:.3f}d), "
                f"iths={self.iths}, nodes={self.n_nodes}, levels={self.n_levels})")


class HotstartProcessor(ForcingProcessor):
    """
    Find and validate SCHISM hotstart files.

    Searches for hotstart.nc from previous cycles, extracts timing info,
    and copies/links to the working directory.
    """

    SOURCE_NAME = "HOTSTART"

    # Minimum file size for a valid hotstart (SECOFS ~20GB, but small test files OK)
    MIN_HOTSTART_SIZE = 1000  # 1KB minimum (catches empty files)

    # Common hotstart file naming patterns
    HOTSTART_PATTERNS = [
        "hotstart.nc",
        "hotstart_it=*.nc",
        "{run}.hotstart.nc",
        "{run}.t{cyc:02d}z.hotstart.nc",
    ]

    # Sidecar recording the provenance (source + target identity) of a
    # stage_init_to_comout write. Suffix is deliberately NOT `.nc` so it
    # can never be picked up by find_input_files' `*.nc`-suffixed globs.
    _PROVENANCE_SUFFIX = ".provenance.json"

    def __init__(
        self,
        config: ForcingConfig,
        input_path: Path,
        output_path: Path,
        run_name: str = "secofs",
        max_lookback_days: int = 3,
    ):
        """
        Args:
            config: ForcingConfig
            input_path: Directory to search for hotstart files (COMOUT or restart archive)
            output_path: Working directory where hotstart.nc should be placed
            run_name: OFS run name for filename patterns (e.g., "secofs")
            max_lookback_days: Maximum days to search backward for hotstart
        """
        super().__init__(config, input_path, output_path)
        self.run_name = run_name
        self.max_lookback_days = max_lookback_days

    def process(self) -> ForcingResult:
        """
        Find and validate hotstart file.

        Returns HotstartInfo in result.metadata["hotstart_info"].
        """
        log.info(f"Hotstart processor: searching in {self.input_path}")
        self.create_output_dir()

        # Search for hotstart files
        warnings: List[str] = []
        hotstart_file = self._find_hotstart(warnings)
        if hotstart_file is None:
            log.warning("No hotstart file found — cold start will be used (ihot=0)")
            return ForcingResult(
                success=True, source=self.SOURCE_NAME,
                warnings=["No hotstart file found — cold start"],
                metadata={"ihot": 0, "hotstart_info": None},
            )

        # Read hotstart info
        info = self._read_hotstart(hotstart_file)
        if info is None:
            return ForcingResult(
                success=False, source=self.SOURCE_NAME,
                errors=[f"Failed to read hotstart: {hotstart_file}"],
            )

        log.info(f"Found hotstart: {info}")

        # Link/copy to output directory
        output_file = self.output_path / "hotstart.nc"
        if not output_file.exists() or output_file.resolve() != hotstart_file.resolve():
            try:
                if output_file.exists():
                    output_file.unlink()
                output_file.symlink_to(hotstart_file)
                log.info(f"Linked hotstart.nc -> {hotstart_file}")
            except OSError:
                import shutil
                shutil.copy2(hotstart_file, output_file)
                log.info(f"Copied hotstart.nc from {hotstart_file}")

        return ForcingResult(
            success=True, source=self.SOURCE_NAME,
            output_files=[output_file],
            warnings=warnings,
            metadata={
                "ihot": 1,
                "hotstart_info": info,
                "time_seconds": info.time_seconds,
                "time_days": info.time_days,
                "iths": info.iths,
                "source_file": str(hotstart_file),
            },
        )

    def stage_init_to_comout(
        self,
        comout_dir: Path,
        init_filename: str,
        warnings: Optional[List[str]] = None,
    ) -> Optional[Path]:
        """Stage the previous-cycle restart as today's COMOUT init file.

        Removes the manual ``nccopy`` hand-off operators have been doing each
        cycle: walks back through previous SECOFS cycles (the existing
        ``_find_hotstart`` already orders by datetime, so we just consume
        its result) and lands a NETCDF4_CLASSIC copy at
        ``comout_dir/init_filename``. SECOFS production cycles every 6h
        (00z, 06z, 12z, 18z), so the natural pick is the cycle 6h prior;
        when that's missing the next-most-recent valid restart is used.

        Format conversion is mandatory because every restart produced by
        ``combine_hotstart7.exe`` is HDF5 (NF90_NETCDF4) and parallel-IO
        collective open at the SCHISM init scale (2794 ranks for SECOFS-UFS)
        segfaults inside libpnetcdf on HDF5 files. Conversion uses ``nccopy``
        when available (operational standard), with a netCDF4-Python
        fallback for environments that don't ship it.

        Args:
            comout_dir: Target ``$COMOUT`` directory (created if missing).
            init_filename: Operational init name, typically
                ``f"{prefix}.t{cyc:02d}z.{pdy}.init.nowcast.nc"``.
            warnings: Optional list to append surfaced warning messages to
                (typically the caller's ``ForcingResult.warnings``).

        Returns:
            Path to the staged init file, or ``None`` if no valid restart
            was found within the lookback window.

        Raises:
            HotstartStagingError: An existing target is being PRESERVED
                (not overwritten) but is not NETCDF4_CLASSIC, so it cannot
                be used by SCHISM's parallel-IO staging as-is. The seed
                itself is left untouched; the caller must treat this as a
                hard prep failure, not a warning.
        """
        source = self._find_hotstart()
        if source is None:
            log.warning(
                "No previous-cycle restart found for COMOUT init staging; "
                "downstream nowcast will need to cold-start"
            )
            return None

        comout_dir = Path(comout_dir)
        comout_dir.mkdir(parents=True, exist_ok=True)
        target = comout_dir / init_filename

        # Never overwrite an existing target with a DIFFERENT state. On a
        # normal cycle nothing else creates this file -- stage_init_to_comout
        # is its only producer -- so a pre-existing target usually means
        # either (a) a same-cycle prep re-run already staged it from this
        # very restart (nowcast and forecast phases both call this, so this
        # is the ordinary path on EVERY warm cycle -- must be silent, not a
        # warning), or (b) an operator hand-seeded a cold-start/rest-state
        # file at the documented name (the WCOSS2 20260802/12z incident: a
        # 60h-stale unrelated restart from earlier bring-up testing silently
        # overwrote a hand-seeded rest-state init because the old code only
        # skipped when target and source resolved to the identical file).
        #
        # Identity is decided by PROVENANCE, not by comparing time/iths
        # scalars: every daily ihot=1 restart is relabeled to the same
        # time_hotstart anchor, so every restart on a given cadence carries
        # IDENTICAL time/iths scalars regardless of the actual ocean state
        # (round-2 review finding: a target with eta2 from one state and a
        # source restart with different eta2 but matching time/iths used to
        # be misclassified as "identical restage"). The sidecar written by
        # this method's own writes below instead records the literal source
        # (path/size/mtime_ns) and target (size/mtime_ns) identity at the
        # moment of the write; only an exact match on BOTH is "identical".
        if target.exists():
            target_format = self._netcdf_format(target)
            if target_format is None:
                # Target exists but can't even be opened as netCDF -- a
                # corrupt/truncated leftover from an interrupted prep, not
                # a valid seed. Self-heal by re-staging over it rather than
                # protecting it forever.
                msg = (
                    f"Existing init file {target} could not be read (corrupt "
                    f"or truncated, likely from an interrupted prep); it is "
                    f"not a valid seed and is being re-staged from "
                    f"{source.name}"
                )
                log.warning(msg)
                if warnings is not None:
                    warnings.append(msg)
                # fall through to the normal copy/convert path below
            else:
                # The format must be verified BEFORE any identical-restage
                # early return: a sidecar match alone is not sufficient,
                # since stage_init_to_comout only ever writes CLASSIC, so a
                # target that matches provenance but isn't CLASSIC means the
                # file was tampered with or corrupted after the write.
                if (
                    target_format == "NETCDF4_CLASSIC"
                    and self._is_identical_restage(target, source)
                ):
                    log.info(f"Init already staged at {target} from {source.name}")
                    return target

                msg = (
                    f"Existing init file {target} is being PRESERVED, not "
                    f"overwritten by restart {source.name}. stage_init_to_comout "
                    f"is the only normal producer of this file, so its prior "
                    f"existence without matching provenance almost always "
                    f"means it was hand-seeded by an operator (e.g. a "
                    f"cold-start rest-state), or that the previously staged "
                    f"file was touched/replaced after staging; if a refresh "
                    f"from {source.name} was actually wanted, remove "
                    f"{target} manually and re-run prep."
                )
                log.warning(msg)
                if warnings is not None:
                    warnings.append(msg)

                if target_format != "NETCDF4_CLASSIC":
                    fmt_msg = (
                        f"Preserved seed {target} is {target_format} format, "
                        f"not NETCDF4_CLASSIC; SCHISM's parallel-IO staging "
                        f"requires the classic model format and this seed "
                        f"will fail or segfault at rank scale as-is "
                        f"(ush/nos_run.sh archives rst.nowcast.nc as HDF5, so "
                        f"`cp rst -> init` is a realistic way to end up here). "
                        f"The seed was preserved -- never destroying an "
                        f"operator file -- but it MUST be converted (e.g. "
                        f"nccopy -k 'netCDF-4 classic model') before the next "
                        f"run. Failing prep now instead of continuing with a "
                        f"seed guaranteed to fail at launch."
                    )
                    log.error(fmt_msg)
                    if warnings is not None:
                        warnings.append(fmt_msg)
                    raise HotstartStagingError(fmt_msg)
                return target

        src_format = self._netcdf_format(source)
        if src_format == "NETCDF4_CLASSIC":
            tmp_target = target.with_name(target.name + ".tmp")
            shutil.copy2(source, tmp_target)
            os.replace(tmp_target, target)
            self._write_provenance(target, source)
            log.info(f"Staged init.nowcast.nc (already classic): {source.name} → {target}")
            return target

        if not self._nccopy_to_classic(source, target):
            log.error(f"Failed to convert {source} to NETCDF4_CLASSIC; init not staged")
            return None

        self._write_provenance(target, source)
        log.info(
            f"Staged init.nowcast.nc (converted {src_format or '?'} "
            f"→ NETCDF4_CLASSIC): {source.name} → {target}"
        )
        return target

    @classmethod
    def _provenance_path(cls, target: Path) -> Path:
        """Sidecar path for a staged ``target``."""
        return target.with_name(target.name + cls._PROVENANCE_SUFFIX)

    @staticmethod
    def _stat_stamp(path: Path) -> dict:
        st = path.stat()
        return {"size": st.st_size, "mtime_ns": st.st_mtime_ns}

    def _write_provenance(self, target: Path, source: Path) -> None:
        """Atomically record ``source``'s and ``target``'s identity next to
        a freshly staged ``target``.

        Written via temp+``os.replace`` so an interrupted write never
        leaves a partial sidecar for a later call to misread. The source
        stamp is captured from ``source`` as staged; the target stamp is
        captured from ``target`` AFTER the write, so a later call can
        detect an operator having touched/replaced the staged file even if
        the original source restart is untouched.
        """
        prov_path = self._provenance_path(target)
        tmp_path = prov_path.with_name(prov_path.name + ".tmp")
        # The stat calls stay inside the try: the sidecar is bookkeeping,
        # and a source purged between staging and here must degrade to
        # "no sidecar" (WARN path next cycle), not abort a prep whose
        # staging work already succeeded.
        try:
            record = {
                "source": {
                    "path": str(source.resolve()),
                    **self._stat_stamp(source),
                },
                "target": self._stat_stamp(target),
            }
            with open(tmp_path, "w") as fh:
                json.dump(record, fh)
            os.replace(tmp_path, prov_path)
        except OSError as e:
            log.warning(f"Could not write provenance sidecar {prov_path}: {e}")
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

    def _is_identical_restage(self, target: Path, source: Path) -> bool:
        """True only when a provenance sidecar exists next to ``target``,
        parses, and its recorded source AND target stamps both match the
        current ``source``/``target`` stat exactly.

        This is the ONLY condition under which a pre-existing target is
        treated as an identical restage. Anything else -- no sidecar, an
        unreadable/corrupt sidecar, a source that moved or changed, or a
        target touched/replaced since it was staged -- returns False and
        the caller falls into the preserve/WARNING path.
        """
        prov_path = self._provenance_path(target)
        if not prov_path.exists():
            return False
        try:
            with open(prov_path) as fh:
                record = json.load(fh)
            recorded_source = record["source"]
            recorded_target = record["target"]
            source_stat = self._stat_stamp(source)
            target_stat = self._stat_stamp(target)
            return (
                recorded_source.get("path") == str(source.resolve())
                and recorded_source.get("size") == source_stat["size"]
                and recorded_source.get("mtime_ns") == source_stat["mtime_ns"]
                and recorded_target.get("size") == target_stat["size"]
                and recorded_target.get("mtime_ns") == target_stat["mtime_ns"]
            )
        except (OSError, ValueError, KeyError, TypeError) as e:
            log.warning(f"Could not read provenance sidecar {prov_path}: {e}")
            return False

    @staticmethod
    def _netcdf_format(path: Path) -> Optional[str]:
        """Best-effort NetCDF format probe — returns None if unreadable."""
        if not HAS_NETCDF4:
            return None
        try:
            ds = Dataset(str(path), "r")
            try:
                return ds.file_format  # e.g. "NETCDF4", "NETCDF4_CLASSIC"
            finally:
                ds.close()
        except Exception as e:
            log.warning(f"Could not probe NetCDF format of {path}: {e}")
            return None

    @staticmethod
    def _nccopy_to_classic(src: Path, dst: Path) -> bool:
        """Convert ``src`` to NETCDF4_CLASSIC at ``dst``.

        Tries the operational ``nccopy -k 'netCDF-4 classic model'`` first;
        falls back to a pure netCDF4-Python rewrite when the binary isn't
        on PATH. Both produce a byte-identical SCHISM-readable file as far
        as the model's pnetcdf reader is concerned. Both paths write to a
        temp name in the same directory and ``os.replace`` onto ``dst``
        only after a full successful write, so an interrupted prep (killed
        job, OOM, node failure) never leaves a truncated file at the final
        ``dst`` name for a later cycle to mistake for a valid seed.
        """
        tmp_dst = dst.with_name(dst.name + ".tmp")

        if shutil.which("nccopy"):
            try:
                subprocess.run(
                    ["nccopy", "-k", "netCDF-4 classic model", str(src), str(tmp_dst)],
                    check=True, capture_output=True,
                )
                os.replace(tmp_dst, dst)
                return True
            except subprocess.CalledProcessError as e:
                log.warning(
                    f"nccopy failed (rc={e.returncode}); falling back to "
                    f"Python conversion. stderr: {e.stderr.decode(errors='replace')[:200]}"
                )
                if tmp_dst.exists():
                    tmp_dst.unlink()

        if not HAS_NETCDF4:
            log.error("Neither nccopy nor netCDF4-Python available for format conversion")
            return False

        try:
            with Dataset(str(src), "r") as src_ds, \
                 Dataset(str(tmp_dst), "w", format="NETCDF4_CLASSIC") as dst_ds:
                # Global attrs
                dst_ds.setncatts({k: src_ds.getncattr(k) for k in src_ds.ncattrs()})
                # Dims
                for name, dim in src_ds.dimensions.items():
                    dst_ds.createDimension(name, len(dim) if not dim.isunlimited() else None)
                # Vars
                for name, var in src_ds.variables.items():
                    new_var = dst_ds.createVariable(
                        name, var.dtype, var.dimensions,
                    )
                    new_var.setncatts({k: var.getncattr(k) for k in var.ncattrs()})
                    new_var[:] = var[:]
            os.replace(tmp_dst, dst)
            return True
        except Exception as e:
            log.error(f"Python NetCDF format conversion failed: {e}")
            if tmp_dst.exists():
                tmp_dst.unlink()
            return False

    def _search_roots(self) -> List[Path]:
        """Directories to anchor the per-day hotstart walk.

        ``find_input_files`` builds ``{run}.{YYYYMMDD}`` subdirs back
        ``max_lookback_days`` days, so it must be anchored at the
        directory that *contains* the dated cycle dirs (the ``$COMOUT``
        root). ``nco_bridge``, however, sets ``paths["restart"]`` from
        ``$COMIN``, which in the SECOFS/STOFS-UFS J-jobs is the
        *current* cycle's dated leaf (``$COMROOT/$NET/$RUN.$PDY``) —
        empty at prep time. When handed such a leaf, also anchor at its
        parent so the walk can cross the day boundary into the sibling
        ``$RUN.$(PDY-1)/…rst.nowcast.nc`` produced by the prior cycle.
        """
        roots = [self.input_path]
        name = self.input_path.name
        if re.fullmatch(r".+\.\d{8}", name) or re.fullmatch(r"\d{8}", name):
            parent = self.input_path.parent
            if parent != self.input_path and parent not in roots:
                roots.append(parent)
        return roots

    def find_input_files(self) -> List[Path]:
        """Find all candidate hotstart files."""
        candidates: List[Path] = []
        base_date = datetime.strptime(self.config.pdy, "%Y%m%d")
        search_roots = self._search_roots()

        for days_back in range(self.max_lookback_days + 1):
            date = base_date - timedelta(days=days_back)
            date_str = date.strftime("%Y%m%d")

            for root in search_roots:
                # Search in date-specific directories
                for dir_pattern in [
                    root / f"{self.run_name}.{date_str}",
                    root / date_str,
                    root / f"{self.run_name}.{date_str}" / "restart_outputs",
                    root,
                ]:
                    if not dir_pattern.exists():
                        continue

                    for file_pattern in [
                        "hotstart*.nc",
                        f"{self.run_name}*hotstart*.nc",
                        f"{self.run_name}*.rst.nowcast.nc",   # COMF restart naming
                        f"{self.run_name}*.init.nowcast.nc",   # COMF init naming
                        f"{self.run_name}*restart*.nc",
                    ]:
                        candidates.extend(sorted(dir_pattern.glob(file_pattern)))

        # Parent + leaf roots can glob the current-cycle dir twice; dedupe
        # while preserving discovery order.
        seen = set()
        deduped: List[Path] = []
        for c in candidates:
            key = str(c)
            if key not in seen:
                seen.add(key)
                deduped.append(c)
        return deduped

    def _find_hotstart(self, warnings: Optional[List[str]] = None) -> Optional[Path]:
        """
        Find the correct hotstart file for the current cycle.

        For a 00z cycle, the hotstart should be from the previous cycle's
        nowcast (e.g., t18z yesterday). Selects by matching cycle time
        in filename, not by filesystem modification time.

        COMF naming: secofs.t{cyc}z.YYYYMMDD.rst.nowcast.nc
        """
        candidates = self.find_input_files()
        from ._log import log_input_files
        log_input_files(
            "HOTSTART", candidates,
            source="HOTSTART", category="hotstart",
            note=f"pdy={self.config.pdy} cyc={self.config.cyc:02d} "
                 f"input_path={self.input_path} run={self.run_name} "
                 f"lookback_days={self.max_lookback_days}",
        )

        # Filter by size
        valid = []
        for f in candidates:
            try:
                if f.stat().st_size >= self.MIN_HOTSTART_SIZE:
                    valid.append(f)
            except OSError:
                continue

        if not valid:
            return None

        # Parse cycle time from filenames and find the one just before current cycle
        cycle_dt = datetime.strptime(self.config.pdy, "%Y%m%d") + \
                   timedelta(hours=self.config.cyc)

        scored = []
        for f in valid:
            file_dt = self._parse_file_datetime(f)
            if file_dt and file_dt < cycle_dt:
                # Prefer most recent file that's BEFORE current cycle
                scored.append((file_dt, f))

        if scored:
            scored.sort(key=lambda x: x[0], reverse=True)  # newest valid first
            best = scored[0][1]
            log.info(f"Selected hotstart by cycle time: {best.name} "
                     f"(valid {scored[0][0]}, current cycle {cycle_dt})")
            self._check_staleness(best, scored[0][0], cycle_dt, warnings)
            return best

        # Fallback: mtime sort over files whose filename did NOT parse to a
        # datetime.  Explicitly EXCLUDE files whose filename parsed to a
        # cycle-time-or-later date (e.g. today's own pre-staged init file
        # secofs.t00z.YYYYMMDD.init.nowcast.nc which gets dropped into
        # $COMOUT by stage_init_to_comout earlier in the same prep run).
        # Returning that future-dated file would let the upstream caller
        # derive a wrong time_hotstart from its filename tag and produce
        # a sim_start misaligned with OBC[t=0].
        unparsable = [
            f for f in valid
            if self._parse_file_datetime(f) is None
        ]
        if not unparsable:
            log.info(
                f"No usable hotstart in {len(valid)} candidates "
                f"(all parsed to cycle-time-or-later); caller should "
                f"derive cold-start anchor from cycle - nowcast_hours."
            )
            return None

        unparsable.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        log.info(
            f"Found {len(unparsable)} unparsable-filename hotstart "
            f"candidates, using newest by mtime: {unparsable[0].name}"
        )
        self._check_staleness(unparsable[0], None, cycle_dt, warnings)
        return unparsable[0]

    @staticmethod
    def _is_init_staged(filepath: Path) -> bool:
        """True if ``filepath`` follows the COMF init-staging convention
        written by :meth:`stage_init_to_comout` and matched by the
        ``*.init.nowcast.nc`` glob in :meth:`find_input_files`.

        ``stage_init_to_comout`` copies whatever restart it finds for
        cycle C (valid at C - nowcast_hours) to a name tagged with C
        itself, so an init-staged file's name encodes the CONSUMING
        cycle, not the content's valid time.
        """
        return filepath.name.endswith(".init.nowcast.nc")

    def _check_staleness(
        self,
        hotstart_file: Path,
        file_dt: Optional[datetime],
        cycle_dt: datetime,
        warnings: Optional[List[str]],
    ) -> None:
        """Warn when the selected restart's valid time differs from the
        orchestrator's time_hotstart anchor (cycle - nowcast_hours): ihot=1
        relabels the restart's state to the anchor regardless, so any gap
        between them is silently skipped or re-simulated unless flagged here.
        """
        expected = cycle_dt - timedelta(hours=self.config.nowcast_hours)

        if self._is_init_staged(hotstart_file):
            msg = (
                f"Hotstart {hotstart_file.name} is an init-staged copy "
                f"(stage_init_to_comout renames whatever restart it found "
                f"to a name tagged with the CONSUMING cycle, not the "
                f"state's valid time); alignment with the time_hotstart "
                f"anchor {expected:%Y-%m-%d %H:%M} cannot be verified "
                f"from the filename"
            )
            log.warning(msg)
            if warnings is not None:
                warnings.append(msg)
            return

        if file_dt is None:
            msg = (
                f"Hotstart {hotstart_file.name} has no parseable valid time "
                f"(selected by mtime fallback); expected anchor is "
                f"{expected:%Y-%m-%d %H:%M} — cannot verify SCHISM state alignment"
            )
            log.warning(msg)
            if warnings is not None:
                warnings.append(msg)
            return

        if file_dt == expected:
            return

        gap_hours = abs((expected - file_dt).total_seconds()) / 3600.0
        if file_dt < expected:
            msg = (
                f"Hotstart {hotstart_file.name} valid at {file_dt:%Y-%m-%d %H:%M} "
                f"is {gap_hours:.0f}h before the time_hotstart anchor "
                f"{expected:%Y-%m-%d %H:%M} (cycle {cycle_dt:%Y-%m-%d %H:%M} - "
                f"nowcast_hours={self.config.nowcast_hours}h); SCHISM will relabel "
                f"this state to {expected:%Y-%m-%d %H:%M}; {gap_hours:.0f}h of ocean "
                f"evolution will be skipped — run intermediate cycles or increase "
                f"nowcast_hours"
            )
        else:
            msg = (
                f"Hotstart {hotstart_file.name} valid at {file_dt:%Y-%m-%d %H:%M} "
                f"is {gap_hours:.0f}h after the time_hotstart anchor "
                f"{expected:%Y-%m-%d %H:%M} (cycle {cycle_dt:%Y-%m-%d %H:%M} - "
                f"nowcast_hours={self.config.nowcast_hours}h); SCHISM will relabel "
                f"this state to {expected:%Y-%m-%d %H:%M}; {gap_hours:.0f}h will be "
                f"re-simulated"
            )
        log.warning(msg)
        if warnings is not None:
            warnings.append(msg)

    def _parse_file_datetime(self, filepath: Path) -> Optional[datetime]:
        """
        Parse datetime from COMF restart filename.

        Patterns:
          secofs.t00z.20260402.rst.nowcast.nc → 2026-04-02 00:00
          secofs.t18z.20260401.rst.nowcast.nc → 2026-04-01 18:00
        """
        name = filepath.name
        # Match: {ofs}.t{HH}z.{YYYYMMDD}.rst or .init
        m = re.search(r"\.t(\d{2})z\.(\d{8})\.", name)
        if m:
            cyc_hour = int(m.group(1))
            date_str = m.group(2)
            try:
                return datetime.strptime(date_str, "%Y%m%d") + timedelta(hours=cyc_hour)
            except ValueError:
                pass
        return None

    def _read_hotstart(self, filepath: Path) -> Optional[HotstartInfo]:
        """Read timing and grid info from hotstart.nc."""
        if not HAS_NETCDF4:
            log.warning("netCDF4 not available — returning basic hotstart info")
            return HotstartInfo(
                filepath=filepath, time_seconds=0.0, iths=0,
                n_nodes=0, n_levels=0,
            )

        try:
            ds = Dataset(str(filepath))

            # Time: scalar or 1D array
            time_var = ds.variables.get("time")
            if time_var is not None:
                time_val = float(time_var[:].flat[0])
            else:
                time_val = 0.0

            # Time step counter
            iths_var = ds.variables.get("iths")
            iths = int(iths_var[:].flat[0]) if iths_var is not None else 0

            # Grid dimensions
            n_nodes = ds.dimensions.get("node", ds.dimensions.get("nSCHISM_hgrid_node"))
            n_nodes = n_nodes.size if n_nodes else 0

            n_levels = ds.dimensions.get("nVert", ds.dimensions.get("nSCHISM_vgrid_layers"))
            n_levels = n_levels.size if n_levels else 0

            ds.close()

            return HotstartInfo(
                filepath=filepath,
                time_seconds=time_val,
                iths=iths,
                n_nodes=n_nodes,
                n_levels=n_levels,
            )

        except Exception as e:
            log.error(f"Failed to read hotstart {filepath}: {e}")
            return None
