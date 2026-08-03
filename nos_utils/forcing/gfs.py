"""
GFS (Global Forecast System) forcing processor.

Processes GFS 0.25° GRIB2 data and creates SCHISM sflux or DATM forcing files.

Input: GFS GRIB2 files from COMINgfs
  Pattern: gfs.YYYYMMDD/HH/atmos/gfs.tHHz.pgrb2.0p25.fHHH
           gfs.YYYYMMDD/HH/atmos/gfs.tHHz.sfluxgrbfHHH.grib2
  Resolution: "0p25" (hourly, ~500MB), "0p50" (3-hourly, ~60MB),
              "sflux" (hourly, ~30MB — surface fields only, native ~0.25°)

Output:
  sflux mode (nws=2): sflux_air_1.NNNN.nc, sflux_rad_1.NNNN.nc, sflux_prc_1.NNNN.nc
  DATM mode (nws=4):  datm_forcing.nc
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import ForcingConfig
from ..io.grib_extract import GRIBExtractor, get_extractor
from .base import ForcingProcessor, ForcingResult
from .sflux_writer import SfluxWriter
from .datm_writer import DATMWriter
from .forcing_writer import ForcingNcWriter

log = logging.getLogger(__name__)


class GFSProcessor(ForcingProcessor):
    """
    GFS atmospheric forcing processor.

    Extracts meteorological variables from GFS GRIB2 files and writes
    SCHISM-compatible sflux NetCDF files or DATM forcing for UFS-Coastal.
    """

    SOURCE_NAME = "GFS"
    # Minimum file size by resolution (bytes).
    # GFS 0.25°: ~500 MB, GFS 0.50°: ~60 MB per file.
    MIN_FILE_SIZE_BY_RES = {
        "0p25": 400_000_000,  # 400 MB
        "0p50": 40_000_000,   # 40 MB
        "sflux": 10_000_000,  # 10 MB (surface-only, ~30 MB typical)
    }
    MIN_FILE_SIZE = 40_000_000  # fallback: 40 MB

    # GRIB2 variable mapping: internal name -> (GRIB2 name, level)
    GRIB2_VARIABLES = {
        # Core 8 variables for sflux
        "uwind": ("UGRD", "10 m above ground"),
        "vwind": ("VGRD", "10 m above ground"),
        "prmsl": ("PRMSL", "mean sea level"),
        "stmp": ("TMP", "2 m above ground"),
        "spfh": ("SPFH", "2 m above ground"),
        "dlwrf": ("DLWRF", "surface"),
        "dswrf": ("DSWRF", "surface"),
        "prate": ("PRATE", "surface"),
        # Extended variables (for COMF Fortran parity)
        "tdair": ("DPT", "2 m above ground"),
        "rh": ("RH", "2 m above ground"),
        "ulwrf": ("ULWRF", "surface"),
        "uswrf": ("USWRF", "surface"),
        "lhtfl": ("LHTFL", "surface"),
        "shtfl": ("SHTFL", "surface"),
        "tcdc": ("TCDC", "entire atmosphere"),
        "apcp": ("APCP", "surface"),
        "evp": ("EVP", "surface"),
        "wtmp": ("TMP", "surface"),
    }

    # Default: extract only the 8 core sflux variables
    DEFAULT_VARIABLES = ["uwind", "vwind", "prmsl", "stmp", "spfh", "dlwrf", "dswrf", "prate"]

    def __init__(
        self,
        config: ForcingConfig,
        input_path: Path,
        output_path: Path,
        variables: Optional[List[str]] = None,
        resolution: str = "0p25",
        extractor: Optional[GRIBExtractor] = None,
        phase: str = "nowcast",
        time_hotstart: Optional[datetime] = None,
        direct_datm: bool = False,
    ):
        """
        Args:
            config: ForcingConfig with domain, cycle, and run window
            input_path: Root GFS data directory (COMINgfs)
            output_path: Output directory for sflux/DATM files
            variables: Variables to extract (default: 8 core sflux vars)
            resolution: GFS resolution ("0p25" or "0p50")
            extractor: GRIB2 extractor (auto-detected if None)
            phase: "nowcast" or "forecast" — determines time window
            time_hotstart: Hotstart datetime (nowcast starts from here)
            direct_datm: Write datm_forcing.nc directly via DATMWriter
                instead of sflux files. Standalone-DATM only — for
                UFS-Coastal coupled runs leave False so the orchestrator
                can blend GFS+HRRR sflux into a wide DATM grid.
        """
        super().__init__(config, input_path, output_path)
        self.variables = variables or (config.variables if config.variables else self.DEFAULT_VARIABLES)
        self.resolution = resolution
        self._extractor = extractor
        self.phase = phase
        self.time_hotstart = time_hotstart
        self.direct_datm = direct_datm
        # Set resolution-appropriate file size threshold
        self.MIN_FILE_SIZE = self.MIN_FILE_SIZE_BY_RES.get(resolution, 40_000_000)

    @property
    def extractor(self) -> GRIBExtractor:
        if self._extractor is None:
            self._extractor = get_extractor()
        return self._extractor

    def _get_time_window(self) -> Tuple[datetime, datetime]:
        """
        Compute the time window for this phase.

        Nowcast:  time_hotstart (or cycle-nowcast_hours) → cycle + 3h buffer
        Forecast: cycle - 3h buffer → cycle + forecast_hours + 3h buffer

        The 3h buffer ensures overlap between nowcast and forecast,
        matching ORG Fortran behavior.
        """
        cycle_dt = datetime.strptime(self.config.pdy, "%Y%m%d") + \
                   timedelta(hours=self.config.cyc)

        if self.phase == "nowcast":
            if self.time_hotstart:
                t_start = self.time_hotstart - timedelta(hours=3)  # buffer before hotstart
            else:
                t_start = cycle_dt - timedelta(hours=self.config.nowcast_hours) - timedelta(hours=3)
            t_end = cycle_dt + timedelta(hours=3)  # buffer past nowcast end
        elif self.phase == "forecast":
            # For UFS-coupled runs (nws=4), the SAME datm_forcing.nc is read
            # by both the nowcast and forecast SCHISM executions, because
            # the forecast prep overwrites the nowcast prep's archived file.
            # Extend t_start back to cover the nowcast window so CDEPS DATM
            # has data at the nowcast SCHISM start (cycle - nowcast_hours).
            # Otherwise CDEPS aborts with
            #   (shr_stream_findBounds) ERROR: rDateIn lt rDatelvd limit true
            #
            # No additional 3h buffer here: _compute_search_cycles now walks
            # back to cycle - nowcast_hours, so the earliest record needed is
            # already covered by the oldest cycle searched. The CDEPS check
            # is strict `<`, so first record at exactly cycle-nowcast_hours
            # equals SCHISM start time and the check passes (rDateIn <
            # rDatelvd is false on equality).
            if self.config.nws == 4:
                t_start = cycle_dt - timedelta(hours=self.config.nowcast_hours)
            else:
                t_start = cycle_dt - timedelta(hours=3)
            t_end = cycle_dt + timedelta(hours=self.config.forecast_hours) + timedelta(hours=3)
        else:
            # Full
            if self.time_hotstart:
                t_start = self.time_hotstart - timedelta(hours=3)
            else:
                t_start = cycle_dt - timedelta(hours=self.config.nowcast_hours) - timedelta(hours=3)
            t_end = cycle_dt + timedelta(hours=self.config.forecast_hours) + timedelta(hours=3)

        return t_start, t_end

    def _filter_to_time_window(self, extracted: dict) -> dict:
        """Filter extracted data to the phase-specific time window."""
        t_start, t_end = self._get_time_window()
        times = extracted["times"]

        # Find indices within window
        keep = [i for i, t in enumerate(times) if t_start <= t <= t_end]

        if len(keep) == len(times):
            log.info(f"Time window [{t_start} to {t_end}]: all {len(times)} steps kept")
            return extracted

        n_dropped = len(times) - len(keep)
        log.info(f"Time window [{t_start} to {t_end}]: kept {len(keep)}/{len(times)} "
                 f"(dropped {n_dropped} outside window)")

        filtered = {
            "times": [times[i] for i in keep],
            "lons": extracted["lons"],
            "lats": extracted["lats"],
            "data": {},
        }
        for var, arrays in extracted["data"].items():
            if len(arrays) != len(times):
                log.warning(f"Filter: {var} has {len(arrays)} entries but "
                            f"expected {len(times)}")
            filtered["data"][var] = [arrays[i] for i in keep if i < len(arrays)]

        return filtered

    def _parse_valid_time(self, gfs_file: Path) -> Optional[datetime]:
        """Parse the valid time of a GFS file from its name and parent path.

        This is the exact parsing previously inlined in ``_extract_all``;
        factored out so the pre-extraction file selection and the extractor
        derive valid times identically (parity by construction).
        """
        try:
            fhr = int(gfs_file.name.split(".f")[-1])
            # Determine cycle from filename: gfs.tHHz...
            cyc_str = gfs_file.name.split(".t")[1][:2]
            cyc_hour = int(cyc_str)
            # Determine date from parent path
            for parent in [gfs_file.parent, gfs_file.parent.parent,
                           gfs_file.parent.parent.parent]:
                if parent.name.startswith("gfs."):
                    date_str = parent.name.split("gfs.")[1][:8]
                    break
            else:
                date_str = self.config.pdy

            base_time = datetime.strptime(date_str, "%Y%m%d") + \
                timedelta(hours=cyc_hour)
            return base_time + timedelta(hours=fhr)
        except (ValueError, IndexError):
            return None

    def _parse_fhr(self, gfs_file: Path) -> Optional[int]:
        """Parse the forecast lead hour of a GFS file from its name.

        Same fhr extraction ``_parse_valid_time`` performs internally
        (``gfs_file.name.split(".f")[-1]``), factored out so the dedup
        preference below can be computed without re-deriving the valid
        time. Used to never let an f000 analysis -- which the GFS
        extractor NaN-fills DSWRF/DLWRF for, since f000 is an analysis
        record and carries no radiation fields (see ``_extract_all``) --
        win a valid time some other searched cycle also covers with a
        real (fhr > 0) forecast lead.
        """
        try:
            return int(gfs_file.name.split(".f")[-1])
        except (ValueError, IndexError):
            return None

    def _select_files_for_window(self, gfs_files: List[Path]) -> List[Path]:
        """Prune the discovered file list to exactly what reaches output.

        Applies, on the *file list* (keyed by parsed valid time), the same
        stable-sort + keep-first dedup + phase time-window filter that
        ``_extract_all`` / ``_filter_to_time_window`` previously applied to
        already-extracted arrays. Net effect: multi-cycle duplicate valid
        times (an earlier cycle's leads superseded by a later cycle) and
        out-of-window valid times are dropped *before* any GRIB2 decode,
        instead of being fully decoded and then discarded.

        Parity: input order is preserved from ``find_input_files``; Python's
        ``sorted`` is stable, so for a repeated valid time the candidates
        keep their original relative order. The window bounds come from the
        same ``_get_time_window``.

        Dedup preference: a file with a real forecast lead (fhr > 0) always
        wins a valid time over an f000 analysis, because f000 lacks
        DSWRF/DLWRF and the extractor NaN-fills them (see ``_extract_all``,
        which then get zero/mean-substituted downstream by the blender). An
        f000 file is selected for a valid time only when it is the SOLE
        candidate there. Within each preference tier, first occurrence in
        list order still wins -- oldest cycle first for nowcast, newest
        cycle first for forecast; see the ordering _compute_search_cycles
        returns for each phase. Keep this in sync with _extract_all's
        mirror dedup below.
        """
        parsed = []
        for f in gfs_files:
            vt = self._parse_valid_time(f)
            if vt is None:
                log.warning(f"Cannot parse time from {f.name}, skipping")
                continue
            parsed.append((vt, f))

        if not parsed:
            return list(gfs_files)

        # Stable sort by valid time (matches _extract_all's sort).
        order = sorted(range(len(parsed)), key=lambda i: parsed[i][0])

        # Two-pass keep-first dedup per valid time. Pass 1: only consider
        # candidates with a real forecast lead (fhr > 0); first occurrence
        # in list order wins, same tie-break as before. Pass 2: for any
        # valid time still unresolved (every candidate is f000 -- it is the
        # sole candidate there), fall back to first occurrence overall.
        seen: Dict[datetime, int] = {}
        for i in order:
            vt, f = parsed[i]
            if vt in seen:
                continue
            fhr = self._parse_fhr(f)
            if fhr is not None and fhr > 0:
                seen[vt] = i
        for i in order:
            vt = parsed[i][0]
            if vt not in seen:
                seen[vt] = i  # sole candidate for this valid time -- f000 accepted as last resort
        unique_idx = sorted(seen.values())

        # Phase time-window filter (matches _filter_to_time_window bounds).
        t_start, t_end = self._get_time_window()
        selected = [
            parsed[i][1]
            for i in unique_idx
            if t_start <= parsed[i][0] <= t_end
        ]

        n_drop = len(gfs_files) - len(selected)
        if n_drop > 0:
            log.info(
                f"Pruned {n_drop} GFS files before extraction "
                f"({len(gfs_files)} discovered -> {len(selected)} kept; "
                f"removed multi-cycle duplicates / out-of-window valid times)"
            )
        return selected

    def process(self) -> ForcingResult:
        """
        Process GFS forcing data.

        Pipeline: discover files -> extract GRIB2 -> filter to time window -> write sflux or DATM
        """
        log.info(f"GFS processor: pdy={self.config.pdy} cyc={self.config.cyc:02d}z "
                 f"phase={self.phase} domain={self.config.forcing_domain} res={self.resolution}")

        self.create_output_dir()

        # Step 1: Find input files
        gfs_files = self.find_input_files()
        from ._log import log_input_files
        log_input_files(
            "GFS", gfs_files or [],
            source="GFS", category="atmospheric",
            note=f"pdy={self.config.pdy} cyc={self.config.cyc} phase={self.phase}",
        )
        if not gfs_files:
            return ForcingResult(
                success=False, source=self.SOURCE_NAME,
                errors=["No GFS input files found"],
            )
        log.info(f"Found {len(gfs_files)} GFS files")

        # Step 1b: Prune to the exact (valid_time -> file) set that reaches
        # output BEFORE decoding. This drops multi-cycle duplicate valid
        # times and out-of-window valid times that the old pipeline only
        # discarded *after* fully decoding them (~700 wasted GRIB2 decodes
        # for a 24h nowcast). Parity-preserving: identical dedup/window rule
        # as the old post-extraction path, just applied to the file list.
        gfs_files = self._select_files_for_window(gfs_files)
        if not gfs_files:
            return ForcingResult(
                success=False, source=self.SOURCE_NAME,
                errors=["No GFS files within the simulation time window"],
            )

        # Write met_files_used log for traceability inside the per-job DATA dir.
        # Earlier code used `self.output_path.parent`, which dropped this file
        # one directory above the working dir (e.g. /work/secofs_ufs/ instead
        # of /work/secofs_ufs/secofs_ufs_prep_00_dev.<jobid>/), polluting the
        # parent dir with stale logs from every cycle.
        self.write_files_used(gfs_files, self.output_path, "GFS", self.phase)

        # Step 2: Extract variables from GRIB2
        extracted = self._extract_all(gfs_files)
        if not extracted["times"]:
            return ForcingResult(
                success=False, source=self.SOURCE_NAME,
                errors=["Failed to extract data from GFS files"],
            )

        # Step 3: Filter to phase-specific time window
        extracted = self._filter_to_time_window(extracted)

        # Step 4: Write output
        output_files = []
        warnings = []

        # Output mode:
        #   direct_datm=True  → write datm_forcing.nc directly via DATMWriter.
        #     Produces narrow-domain DATM (no HRRR blend). Used by tests
        #     and standalone-DATM callers.
        #   nws=4 (UFS-Coastal) and not direct_datm → write a single
        #     gfs_forcing.nc via ForcingNcWriter. The orchestrator's
        #     BlenderProcessor merges this with hrrr_forcing.nc into a
        #     wide DATM grid (matches the shell pipeline architecture).
        #   nws=2 (standalone SCHISM) → write 3 sflux files via SfluxWriter.
        if self.direct_datm:
            writer = DATMWriter()
            datm_path = self.output_path / "datm_forcing.nc"
            writer.write(
                extracted["data"], extracted["times"],
                extracted["lons"], extracted["lats"], datm_path,
            )
            output_files.append(datm_path)
        elif self.config.nws == 4:
            writer = ForcingNcWriter()
            forcing_path = self.output_path / "gfs_forcing.nc"
            writer.write_1d(
                extracted["data"], extracted["times"],
                extracted["lons"], extracted["lats"], forcing_path,
                source_name="GFS",
            )
            output_files.append(forcing_path)
        else:
            writer = SfluxWriter(self.output_path, source_index=1)
            base_date = self._compute_base_date()
            files = writer.write_all(
                extracted["data"], extracted["times"],
                extracted["lons"], extracted["lats"], base_date,
            )
            output_files.extend(files)

            # Write sflux_inputs.txt
            inputs_file = writer.write_sflux_inputs(met_num=self.config.met_num)
            output_files.append(inputs_file)

        return ForcingResult(
            success=True, source=self.SOURCE_NAME,
            output_files=output_files, warnings=warnings,
            metadata={
                "num_input_files": len(gfs_files),
                "num_timesteps": len(extracted["times"]),
                "variables": self.variables,
                "resolution": self.resolution,
                "grid_shape": (len(extracted["lats"]), len(extracted["lons"])),
            },
        )

    def find_input_files(self) -> List[Path]:
        """
        Config-driven GFS file discovery with multi-cycle fallback.

        Searches multiple GFS cycles to cover the full nowcast+forecast window.
        If the primary list is incomplete, supplements with backup files.
        """
        primary = self._build_file_list()

        # Target: enough files for nowcast + forecast (hourly)
        n_target = self.config.nowcast_hours + self.config.forecast_hours + 1

        if len(primary) >= n_target:
            return primary

        log.warning(f"Primary GFS list incomplete ({len(primary)}/{n_target}), checking backup")
        backup = self._build_backup_list()

        if backup and len(backup) > len(primary):
            n_supplement = min(n_target - len(primary), len(backup) - len(primary))
            merged = primary + backup[len(primary):len(primary) + n_supplement]
            log.info(f"Merged {n_supplement} backup files (total: {len(merged)})")
            return merged

        return primary

    def _build_file_list(self) -> List[Path]:
        """
        Build primary file list from GFS cycles covering the run window.

        For forecast phase: uses single cycle with extended leads (matching COMF).
        For nowcast phase: uses multiple cycles with short leads.
        """
        cycles = self._compute_search_cycles()
        gfs_files = []

        # Compute max forecast hour needed from a single cycle.
        # forecast_end includes the +3h buffer so files reaching past the
        # nominal forecast endpoint are included (CDEPS interpolation needs
        # forcing values slightly past the model stop time).
        base_date = datetime.strptime(self.config.pdy, "%Y%m%d")
        cycle_dt = base_date + timedelta(hours=self.config.cyc)
        forecast_end = (
            cycle_dt
            + timedelta(hours=self.config.forecast_hours)
            + timedelta(hours=3)  # buffer
        )

        for date, cyc in cycles:
            date_str = date.strftime("%Y%m%d")
            cycle_start = date + timedelta(hours=cyc)

            # Max lead = hours from this cycle to the buffered forecast end
            max_fhr = int((forecast_end - cycle_start).total_seconds() / 3600)
            max_fhr = max(max_fhr, self.config.forecast_hours + 3)

            # Try standard path structures
            for path_fmt in [
                self.input_path / f"gfs.{date_str}" / f"{cyc:02d}" / "atmos",
                self.input_path / f"gfs.{date_str}" / f"{cyc:02d}",
                self.input_path,
            ]:
                if not path_fmt.exists():
                    continue

                if self.resolution == "sflux":
                    pattern = f"gfs.t{cyc:02d}z.sfluxgrbf*.grib2"
                else:
                    pattern = f"gfs.t{cyc:02d}z.pgrb2.{self.resolution}.f*"
                found = sorted(path_fmt.glob(pattern))

                for f in found:
                    try:
                        if self.resolution == "sflux":
                            # gfs.t00z.sfluxgrbf006.grib2 -> 006
                            fhr = int(f.stem.replace(f"gfs.t{cyc:02d}z.sfluxgrbf", ""))
                        else:
                            fhr = int(f.name.split(".f")[-1])
                    except (ValueError, IndexError):
                        continue

                    if fhr > max_fhr:
                        continue

                    if self.MIN_FILE_SIZE and not self.validate_file_size(f, self.MIN_FILE_SIZE):
                        log.warning(f"Skipping undersized file: {f.name}")
                        continue

                    gfs_files.append(f)

                if found:
                    break  # Found files at this path level

        # Deduplicate preserving order
        seen = set()
        unique = []
        for f in gfs_files:
            if f not in seen:
                seen.add(f)
                unique.append(f)

        return unique

    def _build_backup_list(self) -> List[Path]:
        """Build backup file list from previous day's t12z cycle."""
        base_date = datetime.strptime(self.config.pdy, "%Y%m%d")
        prev_date = base_date - timedelta(days=1)
        date_str = prev_date.strftime("%Y%m%d")

        for path_fmt in [
            self.input_path / f"gfs.{date_str}" / "12" / "atmos",
            self.input_path / f"gfs.{date_str}" / "12",
        ]:
            if not path_fmt.exists():
                continue

            if self.resolution == "sflux":
                pattern = "gfs.t12z.sfluxgrbf*.grib2"
            else:
                pattern = f"gfs.t12z.pgrb2.{self.resolution}.f*"
            found = sorted(path_fmt.glob(pattern))
            files = []
            for f in found:
                try:
                    if self.resolution == "sflux":
                        fhr = int(f.stem.replace("gfs.t12z.sfluxgrbf", ""))
                    else:
                        fhr = int(f.name.split(".f")[-1])
                    if fhr <= self.config.forecast_hours:
                        files.append(f)
                except (ValueError, IndexError):
                    continue
            return files

        return []

    def _compute_search_cycles(self) -> List[Tuple[datetime, int]]:
        """
        Determine which GFS cycles to search.

        Production-realistic strategy matching COMF behavior:
        - Nowcast: walk backward from current cycle to cover the nowcast window
          (past cycles are available at runtime)
        - Forecast: use ONLY the latest available cycle before cycle time
          (future cycles don't exist yet at runtime — extend with longer leads)

        Cycle order matters downstream: _build_file_list preserves this
        order when it assembles the file list, and the keep-first dedup on
        duplicate valid times (_select_files_for_window, _extract_all)
        keeps whichever file appears earliest in that order. Nowcast/full
        return cycles oldest-first (chronological); forecast returns
        cycles newest-first (current cycle, then progressively older
        fallback cycles) so the keep-first dedup prefers the freshest
        cycle covering each valid time -- see the forecast branch below.
        """
        base_date = datetime.strptime(self.config.pdy, "%Y%m%d")
        cycle_dt = base_date + timedelta(hours=self.config.cyc)

        if self.phase == "nowcast" or self.phase == "full":
            # Nowcast (or full nowcast+forecast): multi-cycle, walk backward
            # to cover nowcast window with +3h buffer (matches CDEPS DATM
            # interpolation requirements; missing buffer hours show up as
            # 6 missing timesteps in datm_forcing.nc — 3 at each end).
            #
            # No separate f000-radiation margin cycle is needed here (unlike
            # the forecast branch below): the `- timedelta(hours=6)` slack
            # already baked into the loop condition below walks one 6h
            # cycle past nowcast_start on its own, and because this branch
            # emits oldest-first, that margin cycle's non-zero lead already
            # sits ahead of the window-start cycle's own f000 in dedup
            # priority -- the sole-candidate f000 case the forecast branch's
            # margin cycle guards against cannot arise here.
            nowcast_start = (
                cycle_dt
                - timedelta(hours=self.config.nowcast_hours)
                - timedelta(hours=3)  # buffer
            )
            cycles = []
            t = cycle_dt
            while t >= nowcast_start - timedelta(hours=6):
                cyc_hour = t.hour - (t.hour % 6)  # Snap to 0, 6, 12, 18
                cyc_date = t.replace(hour=0, minute=0, second=0, microsecond=0)
                cycles.append((cyc_date, cyc_hour))
                t -= timedelta(hours=6)
            cycles.reverse()
        else:
            # Forecast: the latest GFS cycle at or before cycle_dt, plus
            # enough earlier fallback cycles to reach back to the
            # forecast-phase window start.
            #
            # For nws=4 (UFS-Coastal/DATM), that start is
            # cycle_dt - nowcast_hours: the same datm_forcing.nc this build
            # produces is also read by the nowcast SCHISM execution (see
            # _get_time_window), so its earliest record must cover
            # cycle - nowcast_hours or CDEPS aborts with
            #   (shr_stream_findBounds) ERROR: rDateIn lt rDatelvd limit true
            # A fixed one-cycle (6h) fallback only covers nowcast_hours <= 6;
            # for larger nowcast_hours (e.g. STOFS-3D-ATL/PAC at 24h) it left
            # up to 18h of the nowcast window held-constant at start.
            # For nws=2 (standalone sflux) the forecast window only needs
            # the 3h pre-cycle buffer, same as before.
            if self.config.nws == 4:
                lookback_start = cycle_dt - timedelta(hours=self.config.nowcast_hours)
            else:
                lookback_start = cycle_dt - timedelta(hours=3)

            cyc_hour = cycle_dt.hour - (cycle_dt.hour % 6)
            cyc_date = cycle_dt.replace(hour=0, minute=0, second=0, microsecond=0)
            # Current (newest) cycle goes first. _build_file_list walks
            # `cycles` in order to assemble the discovered file list, and
            # every searched cycle gets leads out to the same forecast
            # end (see max_fhr below), so an older cycle's files cover the
            # same valid times as the current cycle's -- including times
            # at and after cycle_dt. With the current cycle first, the
            # keep-first dedup in _select_files_for_window / _extract_all
            # ("keep first occurrence for each valid time (shortest lead
            # preferred)") picks the current cycle for every valid time it
            # has, and falls back to progressively older cycles only for
            # the pre-cycle nowcast-overlap hours the current cycle can't
            # reach. Emitting older cycles first (as before) let a stale
            # cycle's long lead win valid times the current cycle already
            # covers -- e.g. the entire forecast window being served from
            # yesterday's data instead of today's.
            cycles = [(cyc_date, cyc_hour)]

            # Walk backward in 6h steps -- mirroring the nowcast branch's
            # cycle-snapping -- until a cycle's own start time reaches back
            # to (or past) lookback_start.
            #
            # nws=4 only: once that point is reached, take ONE further 6h
            # step back (margin_cycles below). lookback_start is itself a
            # GFS cycle boundary (cycle - nowcast_hours, both always
            # multiples of 6h), and _select_files_for_window /
            # _extract_all's dedup now prefers a file with a real forecast
            # lead (fhr > 0) over an f000 analysis -- f000 lacks
            # DSWRF/DLWRF (see _extract_all). Without a cycle older than
            # lookback_start's own cycle, that cycle's f000 is the SOLE
            # candidate at lookback_start and gets selected as the
            # last-resort f000 case, feeding NaN-filled (then
            # zero/mean-substituted by the blender) radiation into the
            # window's own left edge. The margin cycle's non-zero lead at
            # that valid time gives the dedup a real alternative. This is
            # a strict requirement now, not just a nice-to-have: previously
            # (oldest-first order, no fhr preference) a margin cycle here
            # would have let a stale cycle's long lead win the *entire*
            # window, which is why one was deliberately not added; under
            # newest-first plus the fhr-aware dedup that risk is gone --
            # the margin cycle can only ever win the single valid time at
            # lookback_start, nothing past it.
            #
            # nws=2 (standalone sflux): no margin cycle. Its window start
            # (cycle - 3h buffer) is never a GFS cycle boundary (cyc is
            # always a multiple of 6h), so no searched cycle's f000 lands
            # exactly on it and the sole-candidate case above cannot arise.
            # This keeps the nws=2 / minimum-nowcast_hours=6 (nws=4) cases'
            # cycle count matching the previous fixed "one cycle back"
            # fallback except where noted above.
            margin_cycles = 1 if self.config.nws == 4 else 0
            t = cycle_dt - timedelta(hours=6)
            while True:
                prev_hour = t.hour - (t.hour % 6)
                prev_date = t.replace(hour=0, minute=0, second=0, microsecond=0)
                # Append progressively older cycles after the current one,
                # keeping the list newest-first (see comment above `cycles
                # = [(cyc_date, cyc_hour)]`).
                cycles.append((prev_date, prev_hour))
                if prev_date + timedelta(hours=prev_hour) <= lookback_start:
                    if margin_cycles > 0:
                        margin_cycles -= 1
                        t -= timedelta(hours=6)
                        continue
                    break
                t -= timedelta(hours=6)

        # Deduplicate
        seen = set()
        unique = []
        for entry in cycles:
            if entry not in seen:
                seen.add(entry)
                unique.append(entry)

        return unique

    def _extract_all(self, gfs_files: List[Path]) -> dict:
        """Extract all variables from GFS GRIB2 files."""
        result = {"times": [], "lons": None, "lats": None, "data": {}}
        for var in self.variables:
            result["data"][var] = []

        # Use forcing_domain so nws=4 (UFS-Coastal) extracts over the
        # wide DATM grid; nws=2 still extracts over model domain.
        domain = self.config.forcing_domain

        # Get grid coordinates from first file
        result["lons"], result["lats"] = self.extractor.get_grid(gfs_files[0], domain)

        # (var, level) pairs to pull from each file, in self.variables order.
        var_levels = [
            (var, self.GRIB2_VARIABLES[var][0], self.GRIB2_VARIABLES[var][1])
            for var in self.variables
            if var in self.GRIB2_VARIABLES
        ]

        # Forecast lead hour per kept file, aligned 1:1 with result["times"]
        # (appended only when valid_time parses, same guard as below).
        # Feeds the fhr-aware dedup preference after the sort below --
        # keep in sync with _select_files_for_window's mirror dedup.
        fhrs: List[Optional[int]] = []

        for gfs_file in gfs_files:
            # Compute valid time from filename (same parser as the
            # pre-extraction selection — parity by construction).
            valid_time = self._parse_valid_time(gfs_file)
            if valid_time is None:
                log.warning(f"Cannot parse time from {gfs_file.name}, skipping")
                continue

            result["times"].append(valid_time)
            fhrs.append(self._parse_fhr(gfs_file))

            # One wgrib2 pass extracts every variable for this file. The
            # records, levels and arrays are identical to the previous
            # per-variable extraction; only the (large) file decode is
            # shared instead of repeated per variable.
            extracted_recs = self.extractor.extract_many(
                gfs_file,
                [(grib_var, level) for _, grib_var, level in var_levels],
                domain,
            )

            # Append per variable — fill array if missing to keep aligned
            # with times (e.g. dlwrf/dswrf absent in f000 analysis).
            for var, grib_var, level in var_levels:
                data = extracted_recs.get((grib_var, level))
                if data is not None:
                    result["data"][var].append(data)
                else:
                    if result["lons"] is not None and result["lats"] is not None:
                        ny, nx = len(result["lats"]), len(result["lons"])
                        result["data"][var].append(
                            np.full((ny, nx), np.nan, dtype=np.float32)
                        )
                    log.debug(f"Missing {var} in {gfs_file.name}, filled with NaN")

        # Sort by time and deduplicate (multi-cycle overlap produces duplicate valid times)
        if result["times"]:
            sorted_idx = sorted(range(len(result["times"])), key=lambda i: result["times"][i])
            result["times"] = [result["times"][i] for i in sorted_idx]
            fhrs = [fhrs[i] for i in sorted_idx]
            for var in result["data"]:
                if result["data"][var]:
                    # All data arrays must be same length as times (NaN-filled for missing)
                    n_data = len(result["data"][var])
                    n_times = len(sorted_idx)
                    if n_data == n_times:
                        result["data"][var] = [result["data"][var][i] for i in sorted_idx]
                    else:
                        log.warning(f"{var}: data length {n_data} != times {n_times}, skipping sort")

            # Deduplicate: keep first occurrence for each valid time (shortest
            # lead preferred), but never let an f000 analysis win a valid
            # time some other cycle also covers with a real (fhr > 0) lead
            # -- f000 lacks DSWRF/DLWRF and was NaN-filled above. Mirrors
            # _select_files_for_window's dedup; keep both in sync. Pass 1:
            # only fhr > 0 candidates, first occurrence wins. Pass 2: fill
            # in any valid time left over (every candidate there is f000 --
            # it is the sole candidate) with its first occurrence.
            seen_times: Dict[datetime, int] = {}
            for i, t in enumerate(result["times"]):
                if t in seen_times:
                    continue
                fhr = fhrs[i]
                if fhr is not None and fhr > 0:
                    seen_times[t] = i
            for i, t in enumerate(result["times"]):
                if t not in seen_times:
                    seen_times[t] = i  # sole candidate for this valid time -- f000 accepted as last resort
            unique_idx = sorted(seen_times.values())

            if len(unique_idx) < len(result["times"]):
                n_dups = len(result["times"]) - len(unique_idx)
                log.info(f"Removed {n_dups} duplicate valid times from multi-cycle overlap")
                result["times"] = [result["times"][i] for i in unique_idx]
                for var in result["data"]:
                    if result["data"][var]:
                        n_data = len(result["data"][var])
                        if n_data < max(unique_idx) + 1:
                            log.warning(f"Dedup: {var} has {n_data} entries but "
                                        f"max index is {max(unique_idx)}, truncating")
                        result["data"][var] = [result["data"][var][i]
                                               for i in unique_idx if i < n_data]

        return result

    def _compute_base_date(self) -> datetime:
        """Compute sflux base date (start of model simulation).

        Returns the START OF DAY (00Z) of the hotstart date.
        Fortran convention: base_date is always day-start, and the sflux
        time axis is "days since YYYY-MM-DD 00:00:00". The base_date
        attribute [Y,M,D,0] must match the actual reference used for
        computing time values, otherwise SCHISM reads wrong absolute times.
        """
        if self.time_hotstart:
            return self.time_hotstart.replace(hour=0, minute=0, second=0, microsecond=0)
        cycle_dt = datetime.strptime(self.config.pdy, "%Y%m%d") + timedelta(hours=self.config.cyc)
        base = cycle_dt - timedelta(hours=self.config.nowcast_hours)
        return base.replace(hour=0, minute=0, second=0, microsecond=0)
