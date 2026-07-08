"""
Longitude convention utilities.

NOAA NOS-OFS operates in two longitude spaces:
  - Atlantic systems (SECOFS, STOFS-3D-ATL): -180 to +180
  - Pacific systems (STOFS-3D-PAC):            0 to +360 (domain spans antimeridian)

This module provides a single, tested helper for converting between conventions
so every processor can work in the coordinate space the domain requires, rather
than unconditionally normalising everything to -180/+180.

Usage::

    from nos_utils.coords import normalize_lon, lon_convention

    # Detect convention from ForcingConfig
    conv = lon_convention(config)   # "pm180" or "0360"

    # Convert a numpy array of RTOFS lons to match the mesh convention
    rtofs_lon = normalize_lon(rtofs_lon_raw, conv)

    # Convert mesh boundary lons to -180/+180 for algorithms that require it
    bnd_lons_pm180 = normalize_lon(bnd_lons_0360, "pm180")
"""

import numpy as np

__all__ = ["normalize_lon", "lon_convention"]

# Sentinel strings so callers can use string literals or these constants.
PM180 = "pm180"   # -180 to +180
LON360 = "0360"   # 0 to +360


def normalize_lon(
    lon: "np.ndarray | float",
    convention: str,
) -> "np.ndarray | float":
    """Convert longitude array *lon* to the requested *convention*.

    Args:
        lon: Scalar or ndarray of longitudes (any starting range).
        convention: Target convention — ``"pm180"`` for [-180, +180]
            or ``"0360"`` for [0, +360].

    Returns:
        Array (or scalar) with values remapped into the target range.
        The dtype and shape are preserved; a copy is always returned.

    Examples::

        >>> normalize_lon(np.array([270., -10., 0.]), "pm180")
        array([-90., -10.,   0.])
        >>> normalize_lon(np.array([-90., -10., 0.]), "0360")
        array([270., 350.,   0.])
    """
    lon = np.asarray(lon, dtype=np.float64).copy()
    if convention == PM180:
        # Bring everything into (-360, +360] then fold into [-180, +180]
        lon = np.where(lon > 180.0, lon - 360.0, lon)
        lon = np.where(lon < -180.0, lon + 360.0, lon)
    elif convention == LON360:
        # Bring everything into [-360, 360) then fold into [0, 360)
        lon = np.where(lon < 0.0, lon + 360.0, lon)
        lon = np.where(lon >= 360.0, lon - 360.0, lon)
    else:
        raise ValueError(
            f"Unknown lon convention {convention!r}. "
            f"Use {PM180!r} or {LON360!r}."
        )
    return lon


def lon_convention(config: "object") -> str:
    """Return the longitude convention used by a ``ForcingConfig``.

    Detection rule: if ``config.lon_min >= 0`` and ``config.lon_max > 180``
    the domain is unambiguously 0-360 (e.g. Pacific, lon_min=93, lon_max=290).
    All other configurations use -180/+180 (Atlantic, SECOFS).

    Args:
        config: Any object with ``lon_min`` and ``lon_max`` attributes
            (typically a ``ForcingConfig`` instance).

    Returns:
        ``"0360"`` for Pacific-style 0–360 domains,
        ``"pm180"`` for all other (Atlantic-style) domains.
    """
    try:
        lon_min = float(config.lon_min)
        lon_max = float(config.lon_max)
    except AttributeError as exc:
        raise TypeError(
            "config must have lon_min and lon_max attributes"
        ) from exc
    if lon_min >= 0.0 and lon_max > 180.0:
        return LON360
    return PM180
