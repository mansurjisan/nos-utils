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

ufs.configure additionally has its three petlist_bounds lines replaced based
on ``config.ufs_datm_tasks`` and ``config.ufs_total_tasks`` so the v3.9
SECOFS mesh (compute=2794) gets correct OCN PETs (120-2913) instead of the
hardcoded template value (120-1199).

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
            tokens=("YYYY", "DATM_INPUT_DIR", "DATM_MESH_FILE",
                    "DATM_FORCING_FILE"),
            subs=subs,
        )
        output_files.append(ds_path)

        # 4. ufs.configure (copy + patch PET bounds)
        uc_src = self.fix_dir / "ufs.configure"
        uc_dst = self.output_path / "ufs.configure"
        content = uc_src.read_text()
        patched = self._patch_pet_bounds(
            content,
            datm_tasks=int(self.config.ufs_datm_tasks),
            total_tasks=int(self.config.ufs_total_tasks),
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
                "datm_tasks": int(self.config.ufs_datm_tasks),
                "schism_tasks": int(self.config.ufs_schism_tasks),
                "total_tasks": int(self.config.ufs_total_tasks),
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

        return {
            "YYYY": yyyy,
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
        content: str, datm_tasks: int, total_tasks: int,
    ) -> str:
        """Patch the three PET bounds lines in ``ufs.configure``.

        The shell pipeline copies ufs.configure verbatim with the hardcoded
        bounds ``MED 0 119 / ATM 0 119 / OCN 120 1199``. With the v3.9 SECOFS
        mesh (compute=2794) OCN needs PETs ``120 2913``. Patch all three
        based on the YAML-driven resource layout so the generated file always
        matches the actual job submission.

        - MED uses PETs ``0 .. datm_tasks-1`` (co-located with ATM)
        - ATM uses PETs ``0 .. datm_tasks-1``
        - OCN uses PETs ``datm_tasks .. total_tasks-1``
        """
        if datm_tasks <= 0 or total_tasks <= datm_tasks:
            log.warning(
                f"PET layout looks off (datm_tasks={datm_tasks}, "
                f"total_tasks={total_tasks}); leaving ufs.configure verbatim"
            )
            return content

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

        return content
