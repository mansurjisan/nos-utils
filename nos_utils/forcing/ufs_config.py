"""
UFS-Coastal configuration file generator.

Generates the six runtime configuration files needed by a UFS-Coastal
nws=4 cycle from templates in ``$FIXofs``:

    model_configure   <- model_configure.template (date/cycle/forecast length)
    datm_in           <- datm_in.template (DATM grid + mesh paths)
    datm.streams      <- datm.streams.template (DATM stream definition)
    ufs.configure     <- ufs.configure (PET bounds patched per resource layout)
    fd_ufs.yaml       <- fd_ufs.yaml (verbatim)
    noahmptable.tbl   <- noahmptable.tbl (verbatim)

Replaces ``ush/nosofs/nos_ofs_gen_ufs_config.sh`` (~250 lines of bash sed).

Token substitution mirrors the shell ``sed -e "s/@\\[TOKEN\\]/value/g"``:

    @[YYYY], @[MM], @[DD], @[HH]      -> model_t0 = cycle - nowcast_hours
                                          (full coupled run anchor); pulled
                                          from ``time_hotstart`` when caller
                                          passes it, otherwise derived from
                                          ``config.pdy + cyc - nowcast_hours``.
    @[NHOURS]                          -> nowcast_hours + forecast_hours
                                          (covers nowcast + forecast from
                                          model_t0).
    @[DT_ATMOS]                        -> config.ufs_dt_atmos
    @[DATM_INPUT_DIR]                  -> "INPUT" (default)
    @[DATM_MESH_FILE]                  -> "datm_esmf_mesh.nc" (default)
    @[DATM_FORCING_FILE]               -> "datm_forcing.nc" (default)
    @[NX_GLOBAL], @[NY_GLOBAL]         -> from datm_forcing.nc dims, or from
                                          config.datm_* fallback
    @[YYYY_FIRST], @[YYYY_LAST]        -> datm.streams only. YYYY_FIRST is
                                          model_t0's year; YYYY_LAST is
                                          max(YYYY_FIRST, year of cycle +
                                          forecast_hours), so the one
                                          rendered file spans both the
                                          nowcast and forecast legs.

ufs.configure additionally has its petlist_bounds lines replaced based on
``config.ufs_datm_tasks`` and ``config.ufs_total_tasks`` so the v3.9 SECOFS
mesh (compute=2794) gets correct OCN PETs (120-2913) instead of the
hardcoded template value (120-1199). When ``config.ufs_wav_tasks`` is set
(> 0), a fourth WAV_petlist_bounds line is also patched for a DATM+SCHISM+
WW3 layout (see ``_patch_pet_bounds``). A wave system supplies its own
``ufs.configure`` fix file carrying the WAV component stanza and the wave
runSeq phases -- this module only patches PET bounds and the coupling
interval in whatever ufs.configure it is handed; it never adds WW3 runtime
files (ww3_shel.nml, mod_def.ww3, ww3_grid.nml) to ``_REQUIRED_TEMPLATES``.

The model_t0 anchor matches the operational COMF convention used by
``param_nml.py`` and ``tidal.py``: every component of the coupled run --
SCHISM start, DATM stream alignment, OBC time axis, hotstart -- shares the
same cycle - nowcast_hours origin, which is required so CMEPS' ATM->OCN
clock stays in sync.

Style mirrors ``param_nml.py`` and ``tidal.py``.
"""

import logging
import re
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..config import ForcingConfig
from .base import ForcingProcessor, ForcingResult

log = logging.getLogger(__name__)


# Templates that must exist (substituted at runtime).
_REQUIRED_TEMPLATES = (
    "model_configure.template",
    "datm_in.template",
    "datm.streams.template",
    "ufs.configure",
)

# Files that are copied as-is (warn if missing, don't fail).
_OPTIONAL_COPIES = ("fd_ufs.yaml", "noahmptable.tbl")


class UFSConfigProcessor(ForcingProcessor):
    """
    Generate UFS-Coastal runtime configuration files from templates.

    Usage::

        proc = UFSConfigProcessor(
            config, fix_dir, output_dir,
            datm_forcing_path=output_dir / "datm_forcing.nc",
        )
        result = proc.process()
    """

    SOURCE_NAME = "UFS_CONFIG"

    def __init__(
        self,
        config: ForcingConfig,
        fix_dir: Path,
        output_dir: Path,
        datm_forcing_path: Optional[Path] = None,
        datm_input_dir: str = "INPUT",
        datm_mesh_file: str = "datm_esmf_mesh.nc",
        datm_forcing_file: str = "datm_forcing.nc",
        time_hotstart: Optional[datetime] = None,
        phase: Optional[str] = None,
    ):
        """
        Args:
            config: ForcingConfig with pdy, cyc, run hours, and ufs_*
                resource fields.
            fix_dir: Directory containing the template files.
            output_dir: Directory to write generated config files.
            datm_forcing_path: Path to the just-written ``datm_forcing.nc``.
                When provided and readable the processor reads ``nx`` / ``ny``
                from its dimensions to drive ``@[NX_GLOBAL]`` / ``@[NY_GLOBAL]``.
                When None or unreadable, falls back to grid sizes derived from
                ``config.datm_*`` (mirrors the legacy shell behaviour).
            datm_input_dir: Substituted for ``@[DATM_INPUT_DIR]``.
            datm_mesh_file: Substituted for ``@[DATM_MESH_FILE]``.
            datm_forcing_file: Substituted for ``@[DATM_FORCING_FILE]``.
            time_hotstart: Datetime origin for the coupled run.  When None,
                the processor derives it as ``cycle - nowcast_hours`` (the
                operational COMF anchor).  Provide this when the caller has
                already pinned the anchor (e.g. from a hotstart file or a
                ``time_hotstart`` marker) so every component of the prep
                bundle agrees on the same model_t0.
            phase: "nowcast", "forecast", or None.
                * ``phase="nowcast"``: ``start_*`` tokens anchor at
                  ``cycle - nowcast_hours`` and ``NHOURS`` covers the
                  6h nowcast leg only.
                * ``phase="forecast"``: ``start_*`` tokens anchor at
                  ``cycle`` (the forecast leg start) and ``NHOURS``
                  covers ``forecast_hours``.
                * ``phase=None`` (default, backward compat): the combined
                  54h coupled-run anchor used by Route B Phase 1
                  (``start_*`` at ``cycle - nowcast_hours``,
                  ``NHOURS = nowcast_hours + forecast_hours``).

                Operationally the forecast call wins via second-write
                semantics in ``$COMOUT``, so on-disk
                ``model_configure`` ends up forecast-shaped. The
                stage-time patcher (``nos-workflow`` ``configure.patch_model_configure``)
                rewrites the nowcast leg back to its own anchor when
                the PBS nowcast job stages forcing into ``$DATA``.
        """
        super().__init__(config, fix_dir, output_dir)
        self.fix_dir = Path(fix_dir) if fix_dir is not None else None
        self.datm_forcing_path = (
            Path(datm_forcing_path) if datm_forcing_path is not None else None
        )
        self.datm_input_dir = datm_input_dir
        self.datm_mesh_file = datm_mesh_file
        self.datm_forcing_file = datm_forcing_file
        self.time_hotstart = time_hotstart
        self.phase = phase

    def find_input_files(self) -> List[Path]:
        if self.fix_dir is None or not self.fix_dir.exists():
            return []
        files = []
        for name in _REQUIRED_TEMPLATES + _OPTIONAL_COPIES:
            p = self.fix_dir / name
            if p.exists():
                files.append(p)
        return files

    def _resolve_template_dir(self) -> Optional[Path]:
        """Find the directory containing UFS templates.

        UFS templates live in ``fix/<ofs>_ufs/`` (e.g. ``fix/secofs_ufs/``)
        but callers commonly pass ``fix/<ofs>/`` (the SCHISM mesh FIX dir).
        Search order:
          1. ``self.fix_dir`` itself
          2. Sibling ``<name>_ufs`` directory (e.g. fix/secofs -> fix/secofs_ufs)
          3. Inside ``self.fix_dir`` — look for a ``<name>_ufs`` subdir
        Returns the first directory containing all required templates,
        or None if none of the candidates have them.
        """
        if self.fix_dir is None or not self.fix_dir.exists():
            return None

        def _has_all(d: Path) -> bool:
            return all((d / name).exists() for name in _REQUIRED_TEMPLATES)

        candidates: List[Path] = [self.fix_dir]
        # Sibling: fix/secofs -> fix/secofs_ufs
        if not self.fix_dir.name.endswith("_ufs"):
            sibling = self.fix_dir.parent / f"{self.fix_dir.name}_ufs"
            if sibling.exists():
                candidates.append(sibling)
        # Subdir: in case caller passed FIX root (e.g. fix/) and templates
        # live in fix/<ofs>_ufs/
        for child in self.fix_dir.iterdir() if self.fix_dir.is_dir() else []:
            if child.is_dir() and child.name.endswith("_ufs"):
                candidates.append(child)

        for c in candidates:
            if _has_all(c):
                if c != self.fix_dir:
                    log.info(
                        f"Auto-resolved UFS template dir: {self.fix_dir} -> {c}"
                    )
                return c
        return None

    def process(self) -> ForcingResult:
        """Generate all six UFS-Coastal configuration files."""
        log.info(
            f"UFS config processor: fix={self.fix_dir} out={self.output_path}"
        )

        if self.fix_dir is None or not self.fix_dir.exists():
            return ForcingResult(
                success=False,
                source=self.SOURCE_NAME,
                errors=[f"fix_dir does not exist: {self.fix_dir}"],
            )

        # Auto-resolve to a sibling/subdir if templates aren't directly here
        # (e.g. caller passed fix/secofs/ but templates live in fix/secofs_ufs/).
        resolved = self._resolve_template_dir()
        if resolved is None:
            # Build helpful error: list which templates were missing in fix_dir.
            missing = [
                name for name in _REQUIRED_TEMPLATES
                if not (self.fix_dir / name).exists()
            ]
            return ForcingResult(
                success=False,
                source=self.SOURCE_NAME,
                errors=[
                    f"Required template missing: {self.fix_dir / m}"
                    for m in missing
                ] + [
                    f"Hint: looked for templates in {self.fix_dir} and "
                    f"{self.fix_dir.parent}/{self.fix_dir.name}_ufs but "
                    f"none had all of {list(_REQUIRED_TEMPLATES)}"
                ],
            )
        self.fix_dir = resolved  # use the resolved dir for the rest of process()

        self.create_output_dir()

        subs = self._compute_substitutions()
        log.info(
            f"UFS config substitutions: "
            f"{subs['YYYY']}-{subs['MM']}-{subs['DD']} {subs['HH']}z, "
            f"NHOURS={subs['NHOURS']}, DT_ATMOS={subs['DT_ATMOS']}, "
            f"NX={subs['NX_GLOBAL']} NY={subs['NY_GLOBAL']}"
        )

        warnings: List[str] = []
        output_files: List[Path] = []

        # 1. model_configure
        mc_path = self._render_template(
            "model_configure.template", "model_configure",
            tokens=("YYYY", "MM", "DD", "HH", "NHOURS", "DT_ATMOS"),
            subs=subs,
        )
        output_files.append(mc_path)

        # 2. datm_in
        di_path = self._render_template(
            "datm_in.template", "datm_in",
            tokens=("DATM_INPUT_DIR", "DATM_MESH_FILE",
                    "NX_GLOBAL", "NY_GLOBAL"),
            subs=subs,
        )
        output_files.append(di_path)

        # 3. datm.streams
        ds_path = self._render_template(
            "datm.streams.template", "datm.streams",
            tokens=("YYYY", "YYYY_FIRST", "YYYY_LAST", "DATM_INPUT_DIR",
                    "DATM_MESH_FILE", "DATM_FORCING_FILE"),
            subs=subs,
        )
        output_files.append(ds_path)

        # 4. ufs.configure (copy + patch PET bounds)
        uc_src = self.fix_dir / "ufs.configure"
        uc_dst = self.output_path / "ufs.configure"
        content = uc_src.read_text()
        wav_tasks = int(getattr(self.config, "ufs_wav_tasks", 0))
        schism_tasks_cfg = int(getattr(self.config, "ufs_schism_tasks", 0))
        model_dt = getattr(self.config, "model_dt", 120.0)
        coupling_interval_cfg = int(getattr(self.config, "ufs_coupling_interval", 0))
        patched, pet_applied = self._patch_pet_bounds(
            content,
            datm_tasks=int(self.config.ufs_datm_tasks),
            total_tasks=int(self.config.ufs_total_tasks),
            wav_tasks=wav_tasks,
            schism_tasks=schism_tasks_cfg,
        )
        # Coupling interval must equal the SCHISM dt (or an integer multiple
        # of it, for a 4-component wave system that couples on a coarser
        # window) or ModelAdvance stalls at it=1 (template ships SECOFS
        # @120; STOFS dt=150 needs @150; DATM+SCHISM+WW3 needs @360 while
        # SCHISM steps @120).
        patched, written_interval = self._patch_runseq_interval(
            patched, model_dt, coupling_interval=coupling_interval_cfg,
        )
        uc_dst.write_text(patched)
        output_files.append(uc_dst)

        # 5/6. Optional copies
        for name in _OPTIONAL_COPIES:
            src = self.fix_dir / name
            if src.exists():
                dst = self.output_path / name
                shutil.copy2(src, dst)
                output_files.append(dst)
            else:
                warnings.append(
                    f"Optional file not found in fix_dir: {name}"
                )

        return ForcingResult(
            success=True,
            source=self.SOURCE_NAME,
            output_files=output_files,
            warnings=warnings,
            metadata={
                "fix_dir": str(self.fix_dir),
                "nhours": int(subs["NHOURS"]),
                "dt_atmos": int(subs["DT_ATMOS"]),
                "nx_global": int(subs["NX_GLOBAL"]),
                "ny_global": int(subs["NY_GLOBAL"]),
                # None (not the config value) whenever _patch_pet_bounds
                # bailed verbatim, so inputs.{stage}.json reports what
                # actually landed in ufs.configure, not what would have
                # applied if the layout had been valid.
                "datm_tasks": int(self.config.ufs_datm_tasks) if pet_applied else None,
                "schism_tasks": schism_tasks_cfg if pet_applied else None,
                "total_tasks": int(self.config.ufs_total_tasks) if pet_applied else None,
                "wav_tasks": wav_tasks if pet_applied else None,
                # Likewise: the interval _patch_runseq_interval actually
                # wrote, or None when it left the file untouched (no
                # @<interval> line, or a non-positive model_dt).
                "coupling_interval": written_interval,
            },
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _compute_substitutions(self) -> Dict[str, str]:
        """Build the @[TOKEN] -> value substitution map.

        Start time is anchored phase-aware:

        * ``phase="nowcast"``: ``start_* = cycle - nowcast_hours``,
          ``nhours_fcst = nowcast_hours``. Covers only the nowcast leg.
        * ``phase="forecast"``: ``start_* = cycle``,
          ``nhours_fcst = forecast_hours``. Covers only the forecast leg.
        * ``phase=None`` (default, backward compat): ``start_* = cycle -
          nowcast_hours``, ``nhours_fcst = nowcast_hours +
          forecast_hours``. Matches the operational COMF Phase-1 Route B
          anchor used by ``param_nml.py`` / ``tidal.py``: SCHISM begins
          at model_t0, runs through the nowcast window, and ends
          ``forecast_hours`` past the cycle.

        When the caller passes ``time_hotstart`` it wins outright over
        the derived start (the orchestrator may have pinned the anchor
        from a hotstart marker), but ``NHOURS`` still follows the phase
        rule above. Operationally the forecast call writes second and
        wins on-disk; the nos-workflow stage-time patcher rewrites the
        nowcast leg's ``model_configure`` correctly at PBS-job time.
        """
        cycle_dt = datetime.strptime(self.config.pdy, "%Y%m%d") + \
                   timedelta(hours=int(self.config.cyc))
        nowcast_hours = int(self.config.nowcast_hours)
        forecast_hours = int(self.config.forecast_hours)

        if self.phase == "forecast":
            phase_t0 = cycle_dt
            phase_nhours = forecast_hours
        elif self.phase == "nowcast":
            phase_t0 = cycle_dt - timedelta(hours=nowcast_hours)
            phase_nhours = nowcast_hours
        else:
            phase_t0 = cycle_dt - timedelta(hours=nowcast_hours)
            phase_nhours = nowcast_hours + forecast_hours

        if self.time_hotstart is not None:
            model_t0 = self.time_hotstart
        else:
            model_t0 = phase_t0

        yyyy = f"{model_t0.year:04d}"
        mm = f"{model_t0.month:02d}"
        dd = f"{model_t0.day:02d}"
        hh = f"{model_t0.hour:02d}"

        # NHOURS covers the phase window. For backward-compat (phase=None)
        # this is the full coupled run from model_t0 to the end of the
        # forecast window. Use ``ufs_nhours_fcst`` only when it represents
        # the full coverage for the combined phase (factory default sums
        # both phases); fall back to the explicit sum so we never under-shoot.
        nhours_attr = getattr(self.config, "ufs_nhours_fcst", None)
        if self.phase is None:
            if nhours_attr is None or int(nhours_attr) < phase_nhours:
                nhours = phase_nhours
            else:
                nhours = int(nhours_attr)
        else:
            # Explicit phase: always use the phase-specific length;
            # ``ufs_nhours_fcst`` reflects the combined-window default
            # and is not appropriate for a single leg.
            nhours = phase_nhours

        dt_atmos = int(getattr(self.config, "ufs_dt_atmos", 720))

        nx, ny = self._resolve_nx_ny()

        # datm.streams is never re-patched at stage time (unlike
        # model_configure/ufs.configure/param.nml/datm_in), so whichever
        # phase renders it must cover the union of both legs: the nowcast
        # leg starts at model_t0, the forecast leg ends at cycle +
        # forecast_hours. A single-year window (the old @[YYYY] token)
        # breaks the Jan-1 cycle, where the nowcast tail and the whole
        # forecast fall in the new year.
        window_end = cycle_dt + timedelta(hours=forecast_hours)
        yyyy_first = model_t0.year
        yyyy_last = max(yyyy_first, window_end.year)

        return {
            "YYYY": yyyy,
            "YYYY_FIRST": f"{yyyy_first:04d}",
            "YYYY_LAST": f"{yyyy_last:04d}",
            "MM": mm,
            "DD": dd,
            "HH": hh,
            "NHOURS": str(nhours),
            "DT_ATMOS": str(dt_atmos),
            "DATM_INPUT_DIR": self.datm_input_dir,
            "DATM_MESH_FILE": self.datm_mesh_file,
            "DATM_FORCING_FILE": self.datm_forcing_file,
            "NX_GLOBAL": str(int(nx)),
            "NY_GLOBAL": str(int(ny)),
        }

    def _resolve_nx_ny(self) -> Tuple[int, int]:
        """Determine NX_GLOBAL / NY_GLOBAL.

        Preference order:
          1. ``datm_forcing_path`` dims (handles 1D x/y or 2D y/x layouts).
          2. ``config.datm_*`` bounds + ``datm_dx``.
          3. Hardcoded fallback (1721, 1721) — matches the shell default.
        """
        if self.datm_forcing_path is not None and self.datm_forcing_path.exists():
            try:
                from netCDF4 import Dataset  # noqa: WPS433
                with Dataset(str(self.datm_forcing_path), "r") as ds:
                    # Prefer explicit dims if present.
                    if "x" in ds.dimensions and "y" in ds.dimensions:
                        return (
                            int(len(ds.dimensions["x"])),
                            int(len(ds.dimensions["y"])),
                        )
                    if "lon" in ds.dimensions and "lat" in ds.dimensions:
                        return (
                            int(len(ds.dimensions["lon"])),
                            int(len(ds.dimensions["lat"])),
                        )
                    # Fall through: read 2D coord shape.
                    for cname in ("longitude", "lon", "LON"):
                        if cname in ds.variables:
                            v = ds.variables[cname]
                            if v.ndim == 2:
                                ny2, nx2 = v.shape
                                return int(nx2), int(ny2)
                            if v.ndim == 1:
                                nx2 = int(v.shape[0])
                                lat_var = None
                                for lname in ("latitude", "lat", "LAT"):
                                    if lname in ds.variables:
                                        lat_var = ds.variables[lname]
                                        break
                                if lat_var is not None and lat_var.ndim == 1:
                                    return nx2, int(lat_var.shape[0])
                            break
            except Exception as exc:
                log.warning(
                    f"Failed to read NX/NY from {self.datm_forcing_path}: {exc}; "
                    "falling back to config-derived dims"
                )

        # Fall back to config.datm_* + datm_dx.
        lon_min = self.config.datm_lon_min
        lon_max = self.config.datm_lon_max
        lat_min = self.config.datm_lat_min
        lat_max = self.config.datm_lat_max
        dx = float(getattr(self.config, "datm_dx", 0.025) or 0.025)
        if (
            lon_min is not None and lon_max is not None
            and lat_min is not None and lat_max is not None
        ):
            nx = int(round((float(lon_max) - float(lon_min)) / dx)) + 1
            ny = int(round((float(lat_max) - float(lat_min)) / dx)) + 1
            return nx, ny

        # Final fallback — matches the shell script's hardcoded default.
        return 1721, 1721

    def _render_template(
        self,
        template_name: str,
        output_name: str,
        tokens: Tuple[str, ...],
        subs: Dict[str, str],
    ) -> Path:
        """Read a template, substitute @[TOKEN] markers, write to output."""
        src = self.fix_dir / template_name
        dst = self.output_path / output_name
        content = src.read_text()
        for token in tokens:
            value = subs[token]
            placeholder = f"@[{token}]"
            content = content.replace(placeholder, value)
        dst.write_text(content)
        log.info(f"  Generated {output_name} from {template_name}")
        return dst

    @staticmethod
    def _patch_pet_bounds(
        content: str, datm_tasks: int, total_tasks: int, wav_tasks: int = 0,
        schism_tasks: int = 0,
    ) -> Tuple[str, bool]:
        """Patch the PET bounds lines in ``ufs.configure``.

        The shell pipeline copies ufs.configure verbatim with the hardcoded
        bounds ``MED 0 119 / ATM 0 119 / OCN 120 1199``. With the v3.9 SECOFS
        mesh (compute=2794) OCN needs PETs ``120 2913``. Patch based on the
        YAML-driven resource layout so the generated file always matches the
        actual job submission.

        Returns ``(content, applied)``: ``applied`` is False whenever a
        guard bailed and ``content`` was handed back verbatim, so callers
        (and ``process()`` metadata) have a single source of truth for
        whether the PET bounds actually reflect the config rather than
        re-deriving the same yes/no from the input values.

        Two layouts, selected by ``wav_tasks``:

        * ``wav_tasks <= 0`` -- today's 3-component DATM+SCHISM layout:
            - MED uses PETs ``0 .. datm_tasks-1`` (co-located with ATM)
            - ATM uses PETs ``0 .. datm_tasks-1``
            - OCN uses PETs ``datm_tasks .. total_tasks-1``
          ``schism_tasks`` is unused here -- kept only so the call site
          can always pass it uniformly.

        * ``wav_tasks > 0`` -- 4-component DATM+SCHISM+WW3 layout, mirroring
          the validated Alaska DATM+SCHISM+WW3 reference config:
            - ATM uses PETs ``0 .. datm_tasks-1``
            - OCN uses PETs ``datm_tasks .. datm_tasks+schism_tasks-1``
            - WAV uses PETs ``datm_tasks+schism_tasks .. total_tasks-1``
            - MED spans the FULL PET range ``0 .. total_tasks-1``. This is
              a deliberate deviation from the 3-component layout above,
              where MED is confined to the DATM ranks: the working Alaska
              config spans MED across the union of ATM+OCN+WAV, not just
              ATM, and that is the pattern this layout follows.

          ``schism_tasks`` here must be the *configured*
          ``config.ufs_schism_tasks`` -- it is cross-checked against
          ``datm_tasks + schism_tasks + wav_tasks == total_tasks`` rather
          than re-derived by subtraction, so a stale ``total_tasks`` (an
          operator bumps ``wav_tasks`` without bumping ``total_tasks``, or
          vice versa) is caught instead of silently mis-sizing OCN while
          partition.prop and the PBS ``select`` stay sized for the old
          total. On mismatch the whole patch is rejected and ``content``
          is returned verbatim -- including the WAV substitution, which is
          only ever committed once it is independently confirmed present
          (see below).
        """
        if datm_tasks <= 0 or total_tasks <= datm_tasks:
            log.warning(
                f"PET layout looks off (datm_tasks={datm_tasks}, "
                f"total_tasks={total_tasks}); leaving ufs.configure verbatim"
            )
            return content, False

        if wav_tasks <= 0:
            med_atm_hi = datm_tasks - 1
            ocn_lo = datm_tasks
            ocn_hi = total_tasks - 1

            replacements = (
                (
                    r"^(\s*MED_petlist_bounds:\s*).*$",
                    lambda m: f"{m.group(1)}0 {med_atm_hi}",
                ),
                (
                    r"^(\s*ATM_petlist_bounds:\s*).*$",
                    lambda m: f"{m.group(1)}0 {med_atm_hi}",
                ),
                (
                    r"^(\s*OCN_petlist_bounds:\s*).*$",
                    lambda m: f"{m.group(1)}{ocn_lo} {ocn_hi}",
                ),
            )
            for pattern, repl in replacements:
                content = re.sub(pattern, repl, content, flags=re.MULTILINE)
            return content, True

        # 4-component layout below. Validate the configured split before
        # touching anything: this is the same check the operator's job
        # submission implicitly makes (datm + schism + wav ranks == the
        # PBS select count), so a mismatch here means the file we are
        # about to write would not match the ranks actually reserved.
        task_sum = datm_tasks + schism_tasks + wav_tasks
        if task_sum != total_tasks:
            log.error(
                f"PET layout mismatch: datm_tasks={datm_tasks} + "
                f"schism_tasks={schism_tasks} + wav_tasks={wav_tasks} = "
                f"{task_sum} != total_tasks={total_tasks}; leaving "
                "ufs.configure verbatim"
            )
            return content, False

        if schism_tasks <= 0:
            log.warning(
                f"Wave PET layout looks off (datm_tasks={datm_tasks}, "
                f"wav_tasks={wav_tasks}, total_tasks={total_tasks} -> "
                f"schism_tasks={schism_tasks}); leaving ufs.configure "
                "verbatim"
            )
            return content, False

        atm_hi = datm_tasks - 1
        ocn_lo = datm_tasks
        ocn_hi = datm_tasks + schism_tasks - 1
        wav_lo = ocn_hi + 1
        wav_hi = total_tasks - 1
        med_hi = total_tasks - 1

        # Commit the WAV edit first, and only if the fix file actually has
        # a WAV_petlist_bounds line. A fix file deployed before the wave
        # system's ufs.configure lands (or rolled back to a pre-wave
        # version) has no WAV component at all -- re.sub would silently
        # no-op on that line while MED/ATM/OCN still got widened/truncated
        # for a WAV component that doesn't exist, orphaning PETs. Using
        # re.subn here and checking the count means the MED/ATM/OCN edits
        # below are only ever applied once WAV is confirmed patched.
        content_with_wav, wav_count = re.subn(
            r"^(\s*WAV_petlist_bounds:\s*).*$",
            lambda m: f"{m.group(1)}{wav_lo} {wav_hi}",
            content,
            flags=re.MULTILINE,
        )
        if not wav_count:
            log.error(
                "ufs.configure has no WAV_petlist_bounds line; a wave "
                "system must supply a wave-enabled ufs.configure fix file "
                "before wav_tasks > 0 can be patched. Leaving "
                "ufs.configure verbatim"
            )
            return content, False

        content = content_with_wav
        replacements = (
            (
                r"^(\s*MED_petlist_bounds:\s*).*$",
                lambda m: f"{m.group(1)}0 {med_hi}",
            ),
            (
                r"^(\s*ATM_petlist_bounds:\s*).*$",
                lambda m: f"{m.group(1)}0 {atm_hi}",
            ),
            (
                r"^(\s*OCN_petlist_bounds:\s*).*$",
                lambda m: f"{m.group(1)}{ocn_lo} {ocn_hi}",
            ),
        )
        for pattern, repl in replacements:
            content = re.sub(pattern, repl, content, flags=re.MULTILINE)

        return content, True

    @staticmethod
    def _patch_runseq_interval(
        content: str, model_dt: float, coupling_interval: int = 0,
    ) -> Tuple[str, Optional[int]]:
        """Patch the NUOPC ``runSeq::`` coupling interval.

        The driver hands SCHISM a coupling window equal to the runSeq
        interval; SCHISM must be able to land exactly on it, so the interval
        has to equal the SCHISM timestep (``dt``), or an integer multiple of
        it. The template ships a SECOFS value (``@120``), which is wrong for
        STOFS-3D-ATL (dt=150): the driver gives SCHISM a 120s window it
        can't step (150 > 120) and ``ModelAdvance`` is stuck at ``it=1`` --
        the nowcast inits clean but never timesteps (no hotstart, empty
        output, false ``rc=0`` PASS).

        ``coupling_interval`` covers the DATM+SCHISM+WW3 case, where WW3
        needs a coarser coupling window than SCHISM's own dt (e.g. @360
        while SCHISM steps @120 -- 360 = 3 SCHISM steps); the old code could
        only ever emit ``@<model_dt>``. When set (> 0) it is used as the
        runSeq interval, but only if it is an integer multiple of
        ``model_dt`` -- SCHISM must still land exactly on the coupling
        window. A non-multiple (e.g. @300 with dt=120) is rejected with a
        clear error naming both values, and the interval falls back to
        ``@<model_dt>`` rather than silently emitting an unsteppable window.
        When ``coupling_interval <= 0`` (default), behavior is unchanged
        from before: the interval is always ``@<model_dt>``.

        Only the numeric interval line inside the runSeq block is replaced
        (``^@\\d+$``). The trailing bare ``@`` line (no digits) that closes
        the block is left untouched. The anchor also assumes the
        conventional NUOPC layout where a nested fast-rate loop is indented
        -- a flush-left nested ``@N`` line would be collapsed along with the
        outer interval.

        Returns ``(content, written_interval)``: ``written_interval`` is
        the integer actually substituted into the runSeq line, or ``None``
        when nothing was written (no ``@<interval>`` line found, or the
        ``model_dt`` guard below rejected the input) -- callers must report
        this, not a value re-derived from config, as the coupling interval
        that ended up in the file.
        """
        n = int(model_dt)
        if n <= 0:
            log.error(
                f"ufs.configure model_dt={model_dt} truncates to a "
                f"non-positive interval ({n}); leaving runSeq coupling "
                "interval verbatim"
            )
            return content, None

        if model_dt != n:
            log.warning(
                f"ufs.configure model_dt={model_dt} is fractional; the "
                f"runSeq divisibility check below uses the truncated "
                f"value ({n})"
            )

        if coupling_interval and int(coupling_interval) > 0:
            ci = int(coupling_interval)
            if ci % n == 0:
                n = ci
            else:
                log.error(
                    f"ufs.configure coupling_interval={ci} is not an "
                    f"integer multiple of model_dt={n}; SCHISM cannot land "
                    f"exactly on that window. Falling back to @{n}."
                )

        new_content, count = re.subn(
            r"(?m)^@\d+\s*$", f"@{n}", content
        )
        if count:
            log.info(f"  ufs.configure runSeq -> @{n}")
            return new_content, n
        else:
            log.warning(
                "ufs.configure has no @<interval> runSeq line; "
                "leaving coupling interval verbatim"
            )
            return content, None
