"""
GRIB2 data extraction backends.

Two backends:
- Wgrib2Extractor: Production — uses wgrib2 subprocess (fast, handles all grids)
- CfgribExtractor: Development — uses cfgrib/xarray (no external binary needed)

Usage:
    extractor = get_extractor()  # auto-detect best available
    data, lons, lats = extractor.extract("gfs.grib2", "UGRD", "10 m above ground",
                                          domain=(-88, -63, 17, 40))
"""

import logging
import shutil
import subprocess
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


class GRIBExtractor(ABC):
    """Abstract GRIB2 extraction interface."""

    @abstractmethod
    def extract(
        self,
        grib_file: Path,
        variable: str,
        level: str,
        domain: Tuple[float, float, float, float],
    ) -> Optional[np.ndarray]:
        """
        Extract a single variable from a GRIB2 file, subsetted to domain.

        Args:
            grib_file: Path to GRIB2 file
            variable: GRIB2 variable name (e.g., "UGRD")
            level: Level string (e.g., "10 m above ground")
            domain: (lon_min, lon_max, lat_min, lat_max)

        Returns:
            2D numpy array (ny, nx) or None if extraction failed
        """
        ...

    @abstractmethod
    def get_grid(
        self,
        grib_file: Path,
        domain: Tuple[float, float, float, float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get lon/lat coordinate arrays for the subsetted domain.

        Returns:
            (lons_1d, lats_1d) arrays
        """
        ...

    def regrid_to_latlon(
        self,
        grib_file: Path,
        domain: Tuple[float, float, float, float],
        dx: float,
        output_path: Path,
    ) -> Optional[Path]:
        """
        Regrid a GRIB2 file from native projection to regular lat/lon.

        Critical for HRRR (Lambert Conformal) — SCHISM cannot handle LCC grids.

        Args:
            grib_file: Input GRIB2 file (any projection)
            domain: Target domain bounds
            dx: Target resolution in degrees
            output_path: Path for regridded output

        Returns:
            Path to regridded GRIB2 file, or None if not supported
        """
        return None


class Wgrib2Extractor(GRIBExtractor):
    """
    Production GRIB2 extractor using wgrib2 subprocess.

    wgrib2 handles all grid projections, variable matching, and domain subsetting
    natively. This is the preferred backend for operational use.

    Auto-detects wgrib2 with IPOLATES support (needed for -new_grid regridding).
    Prefers spack-stack wgrib2 over system wgrib2 when available.
    """

    # Known paths where wgrib2 with IPOLATES may exist
    WGRIB2_SEARCH_PATHS = [
        # spack-stack builds (typically have IPOLATES)
        "/opt/spack-stack/spack-stack-1.9.2/envs/ufs-wm-env/install/gcc/13.3.1/wgrib2-3.6.0-yu365ku/bin/wgrib2",
        "/opt/spack-stack/*/envs/*/install/*/wgrib2-*/bin/wgrib2",
    ]

    def __init__(self, wgrib2_path: str = "wgrib2"):
        # Try to find wgrib2 with IPOLATES first
        self.wgrib2 = self._find_best_wgrib2(wgrib2_path)
        if not self.wgrib2:
            raise FileNotFoundError(f"wgrib2 not found: {wgrib2_path}")

    @classmethod
    def _find_best_wgrib2(cls, default: str) -> Optional[str]:
        """Find wgrib2 binary, preferring one with IPOLATES support."""
        import glob as glob_mod

        # Check spack-stack paths first (likely have IPOLATES)
        for pattern in cls.WGRIB2_SEARCH_PATHS:
            if "*" in pattern:
                matches = glob_mod.glob(pattern)
                for m in matches:
                    if Path(m).exists():
                        log.info(f"Found spack wgrib2: {m}")
                        return m
            elif Path(pattern).exists():
                log.info(f"Found spack wgrib2: {pattern}")
                return pattern

        # Fall back to PATH
        found = shutil.which(default)
        if found:
            return found

        return None

    def extract(
        self,
        grib_file: Path,
        variable: str,
        level: str,
        domain: Tuple[float, float, float, float],
        skip_subset: bool = False,
    ) -> Optional[np.ndarray]:
        """
        Extract a variable from GRIB2 file.

        Args:
            skip_subset: If True, skip -small_grib domain subsetting.
                Use when the file is already regridded to the target domain
                (avoids off-by-one trimming from floating-point boundary matching).
        """
        grib_file = Path(grib_file)
        lon_min, lon_max, lat_min, lat_max = domain

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            subset_file = tmpdir / "subset.grb2"
            bin_file = tmpdir / "data.bin"

            # Step 1: Match variable+level, optionally subset to domain
            match_str = f":{variable}:{level}:"
            if skip_subset:
                cmd = [
                    self.wgrib2, str(grib_file),
                    "-match", match_str,
                    "-grib", str(subset_file),
                ]
            else:
                cmd = [
                    self.wgrib2, str(grib_file),
                    "-match", match_str,
                    "-small_grib",
                    f"{lon_min}:{lon_max}",
                    f"{lat_min}:{lat_max}",
                    str(subset_file),
                ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0 or not subset_file.exists():
                log.debug(f"wgrib2 match failed for {variable}:{level}: {result.stderr[:200]}")
                return None

            # Step 2: Dump first record to binary (avoid double-records like PRATE)
            cmd2 = [
                self.wgrib2, str(subset_file),
                "-d", "1",
                "-no_header", "-bin", str(bin_file),
            ]
            result2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=60)
            if result2.returncode != 0 or not bin_file.exists():
                return None

            data = np.fromfile(bin_file, dtype=np.float32)

            # Step 3: Get grid dimensions
            nx, ny = self._get_nxny(subset_file)
            if nx and ny and data.size == nx * ny:
                return data.reshape((ny, nx))

            log.warning(f"Grid dim mismatch: data.size={data.size}, nx={nx}, ny={ny}")
            return None

    def get_grid(
        self,
        grib_file: Path,
        domain: Tuple[float, float, float, float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        grib_file = Path(grib_file)
        lon_min, lon_max, lat_min, lat_max = domain

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            subset_file = tmpdir / "subset.grb2"

            # Subset any record to get grid info
            cmd = [
                self.wgrib2, str(grib_file),
                "-d", "1",
                "-small_grib",
                f"{lon_min}:{lon_max}",
                f"{lat_min}:{lat_max}",
                str(subset_file),
            ]
            subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            if subset_file.exists():
                nx, ny = self._get_nxny(subset_file)
                if nx and ny:
                    # For regular lat/lon grids, reconstruct coords
                    dx = (lon_max - lon_min) / (nx - 1) if nx > 1 else 0.25
                    dy = (lat_max - lat_min) / (ny - 1) if ny > 1 else 0.25
                    lons = np.linspace(lon_min, lon_max, nx)
                    lats = np.linspace(lat_min, lat_max, ny)
                    return lons, lats

        # Fallback: compute from resolution
        log.warning("Could not determine grid from file, using 0.25° default")
        lons = np.arange(lon_min, lon_max + 0.25, 0.25)
        lats = np.arange(lat_min, lat_max + 0.25, 0.25)
        return lons, lats

    def regrid_to_latlon(
        self,
        grib_file: Path,
        domain: Tuple[float, float, float, float],
        dx: float,
        output_path: Path,
        match_pattern: Optional[str] = None,
    ) -> Optional[Path]:
        """
        Regrid GRIB2 from native projection (e.g., Lambert Conformal) to regular lat/lon.

        Uses: wgrib2 -new_grid_winds earth -new_grid latlon lon0:nx:dx lat0:ny:dy
        The -new_grid_winds earth flag correctly rotates grid-relative winds.

        For wind rotation to work, U and V must be processed together in the same
        command (wgrib2 pairs them automatically when both are present).

        Args:
            grib_file: Input GRIB2 file
            domain: (lon_min, lon_max, lat_min, lat_max)
            dx: Target resolution in degrees
            output_path: Path for regridded output
            match_pattern: Optional wgrib2 -match regex to select variables.
                For HRRR, use a pattern that includes BOTH UGRD and VGRD:
                ":(UGRD:10 m|VGRD:10 m|TMP:2 m|SPFH:2 m|MSLMA|PRATE|DSWRF|DLWRF):"
        """
        lon_min, lon_max, lat_min, lat_max = domain
        nx = int(round((lon_max - lon_min) / dx)) + 1
        ny = int(round((lat_max - lat_min) / dx)) + 1

        output_path = Path(output_path)

        cmd = [self.wgrib2, str(grib_file)]
        if match_pattern:
            cmd.extend(["-match", match_pattern])
        cmd.extend([
            "-new_grid_winds", "earth",
            "-new_grid", "latlon",
            f"{lon_min}:{nx}:{dx}",
            f"{lat_min}:{ny}:{dx}",
            str(output_path),
        ])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
            return output_path

        log.error(f"Regrid failed: {result.stderr[:300]}")
        return None

    def _get_nxny(self, grib_file: Path) -> Tuple[Optional[int], Optional[int]]:
        """Parse nx, ny from wgrib2 -nxny output."""
        cmd = [self.wgrib2, str(grib_file), "-nxny"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return None, None

        for line in result.stdout.strip().split("\n"):
            # Format: "1:0:nx=101 ny=81" or similar
            if "x" in line and "(" in line:
                # Handle "(101 x 81)" format
                try:
                    dims = line.split("(")[1].split(")")[0]
                    nx, ny = map(int, dims.split(" x "))
                    return nx, ny
                except (IndexError, ValueError):
                    pass
            # Handle "nx=101 ny=81" format
            parts = line.split(":")
            for part in parts:
                part = part.strip()
                if part.startswith("(") and " x " in part:
                    try:
                        dims = part.strip("()")
                        nx, ny = map(int, dims.split(" x "))
                        return nx, ny
                    except (ValueError, IndexError):
                        pass

        return None, None


class CfgribExtractor(GRIBExtractor):
    """
    Development GRIB2 extractor using cfgrib/xarray.

    No external wgrib2 binary needed. Useful for local development and testing.
    Slower than wgrib2 and does not support all projection types.
    """

    def __init__(self):
        try:
            import cfgrib  # noqa: F401
            import xarray  # noqa: F401
        except ImportError:
            raise ImportError("cfgrib and xarray required. Install with: pip install cfgrib xarray")

    def extract(
        self,
        grib_file: Path,
        variable: str,
        level: str,
        domain: Tuple[float, float, float, float],
    ) -> Optional[np.ndarray]:
        import xarray as xr

        grib_file = Path(grib_file)
        lon_min, lon_max, lat_min, lat_max = domain

        # Map GRIB2 variable names to cfgrib short names
        var_map = {
            "UGRD": "u10", "VGRD": "v10", "TMP": "t2m",
            "SPFH": "q", "PRMSL": "prmsl", "MSLMA": "prmsl",
            "DLWRF": "dlwrf", "DSWRF": "dswrf", "PRATE": "prate",
            "RH": "r2", "DPT": "d2m", "APCP": "tp",
            "PRES": "sp",
        }

        try:
            ds = xr.open_dataset(str(grib_file), engine="cfgrib")
            cfgrib_name = var_map.get(variable, variable.lower())

            if cfgrib_name not in ds:
                log.debug(f"{cfgrib_name} not found in {grib_file.name}")
                return None

            da = ds[cfgrib_name]

            # Domain subset
            if "longitude" in da.dims and "latitude" in da.dims:
                da = da.sel(
                    longitude=slice(lon_min, lon_max),
                    latitude=slice(lat_max, lat_min),  # lat usually descending
                )

            data = da.values.astype(np.float32)
            if data.ndim > 2:
                data = data.squeeze()

            ds.close()
            return data

        except Exception as e:
            log.debug(f"cfgrib extraction failed: {e}")
            return None

    def get_grid(
        self,
        grib_file: Path,
        domain: Tuple[float, float, float, float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        import xarray as xr

        lon_min, lon_max, lat_min, lat_max = domain

        try:
            ds = xr.open_dataset(str(grib_file), engine="cfgrib")
            lons = ds.longitude.values
            lats = ds.latitude.values

            lon_mask = (lons >= lon_min) & (lons <= lon_max)
            lat_mask = (lats >= lat_min) & (lats <= lat_max)

            ds.close()
            return lons[lon_mask], lats[lat_mask]

        except Exception as e:
            log.warning(f"cfgrib grid extraction failed: {e}")
            lons = np.arange(lon_min, lon_max + 0.25, 0.25)
            lats = np.arange(lat_min, lat_max + 0.25, 0.25)
            return lons, lats


def get_extractor(wgrib2_path: str = "wgrib2") -> GRIBExtractor:
    """
    Auto-detect and return the best available GRIB2 extractor.

    Prefers wgrib2 (faster, handles all projections) over cfgrib.

    Raises:
        RuntimeError: If neither wgrib2 nor cfgrib is available
    """
    if shutil.which(wgrib2_path):
        log.info("Using wgrib2 extractor")
        return Wgrib2Extractor(wgrib2_path)

    try:
        extractor = CfgribExtractor()
        log.info("Using cfgrib extractor (wgrib2 not found)")
        return extractor
    except ImportError:
        pass

    raise RuntimeError(
        "No GRIB2 extraction backend available. "
        "Install wgrib2 or cfgrib: pip install cfgrib xarray"
    )
