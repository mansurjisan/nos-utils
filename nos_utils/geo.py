"""Longitude-frame helpers for dateline-spanning domains.

nos-utils forcing interpolation historically assumed western-hemisphere
(-180..180) longitudes: every ocean/atmos module normalized its source grid to
-180..180 (``np.where(lon > 180, lon - 360, lon)``) and then masked against the
target/domain bounds. STOFS-3D-Pacific spans the dateline in a 0-360 frame
(lon 92.5 -> 290), so the eastern half of every source grid (orig 180..290 ->
converted to -180..-70) failed the ``>= lon_min`` test and was silently dropped.

``unwrap_to_domain`` maps longitudes into the single 360-degree window centered
on the domain midpoint, so ANY domain -- dateline-spanning or not -- becomes
contiguous and a plain min/max bounding box and planar interpolation are correct.
For a western-hemisphere domain (e.g. STOFS-3D-ATL, center ~ -75) it is
equivalent to a no-op, so existing systems are unaffected.

Usage contract: unwrap BOTH the data source grid AND the target (model node)
longitudes with the SAME center before any bbox mask, meshgrid, convex hull, or
nearest-neighbor distance. The center is the domain midpoint,
``domain_center_lon(lon_min, lon_max)``.
"""
from __future__ import annotations

import numpy as np


def unwrap_to_domain(lon, center_lon):
    """Map longitude(s) into ``[center_lon - 180, center_lon + 180)``.

    Both a data source grid and the target model nodes must be unwrapped with
    the SAME ``center_lon`` before comparing or interpolating, so points on
    either side of the 180-degree meridian land in one contiguous frame.

    Parameters
    ----------
    lon : array_like or float
        Longitude(s) in degrees, any convention (0-360, -180..180, or mixed).
    center_lon : float
        Frame center in degrees, typically the domain midpoint
        ``0.5 * (lon_min + lon_max)`` in the authored convention.

    Returns
    -------
    numpy.ndarray
        Longitudes shifted by whole multiples of 360 into the window centered
        on ``center_lon``. Same shape as the input.
    """
    lon = np.asarray(lon, dtype=float)
    return center_lon + (np.mod(lon - center_lon + 180.0, 360.0) - 180.0)


def domain_center_lon(lon_min, lon_max):
    """Return the frame center (domain midpoint) for authored lon bounds."""
    return 0.5 * (float(lon_min) + float(lon_max))


def crosses_dateline(lon_min, lon_max):
    """True if the authored domain box reaches past 180E (a 0-360 domain).

    A pure -180..180 domain has both bounds <= 180; STOFS-3D-Pacific
    (lon_max=290) does not. This is a logging/branch hint only --
    ``unwrap_to_domain`` handles both cases uniformly.
    """
    return float(lon_max) > 180.0 or float(lon_min) > 180.0
