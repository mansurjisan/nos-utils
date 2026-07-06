"""Longitude-frame helper tests (dateline handling)."""
from __future__ import annotations

import numpy as np
import pytest

from nos_utils.geo import crosses_dateline, domain_center_lon, unwrap_to_domain


# Domain midpoints used across the suite.
ATL_CENTER = domain_center_lon(-98.5035, -52.4867)   # ~ -75.5
PAC_CENTER = domain_center_lon(92.5, 290.0)          # 191.25


def test_domain_center_lon():
    assert domain_center_lon(92.5, 290.0) == pytest.approx(191.25)
    assert domain_center_lon(-98.5035, -52.4867) == pytest.approx(-75.4951)


def test_crosses_dateline_flags():
    assert crosses_dateline(92.5, 290.0) is True        # Pacific (0-360)
    assert crosses_dateline(-98.5035, -52.4867) is False  # Atlantic
    assert crosses_dateline(-88.0, -63.0) is False        # Gulf


def test_atlantic_is_noop_equivalent():
    # Western-hemisphere lons already lie in the window centered on -75.5,
    # so unwrapping must leave them unchanged.
    lons = np.array([-98.5035, -75.0, -52.4867])
    out = unwrap_to_domain(lons, ATL_CENTER)
    np.testing.assert_allclose(out, lons)


def test_pacific_eastern_hemisphere_becomes_contiguous():
    # The dateline-crossing failure mode: a source point in the domain's
    # eastern half is stored as -70 in -180..180. Unwrapped to the Pacific
    # center it must land at 290 (inside the 92.5..290 domain), NOT be dropped.
    assert unwrap_to_domain(-70.0, PAC_CENTER).item() == pytest.approx(290.0)
    # A dateline-straddling point near 180 is unchanged.
    assert unwrap_to_domain(200.0, PAC_CENTER).item() == pytest.approx(200.0)
    assert unwrap_to_domain(180.0, PAC_CENTER).item() == pytest.approx(180.0)


def test_source_and_target_share_one_frame():
    # The core contract: a source point and a model node at the same physical
    # location must map to the SAME value regardless of input convention.
    # Node authored 0-360 (290) vs RTOFS source -180..180 (-70) -> both 290.
    node = unwrap_to_domain(290.0, PAC_CENTER).item()
    src = unwrap_to_domain(-70.0, PAC_CENTER).item()
    assert node == pytest.approx(src)


def test_pacific_domain_is_monotonic_after_unwrap():
    # A dense sweep across the whole Pacific domain (0-360) stays monotonic and
    # bounded in [92.5, 290] after unwrapping -> a plain min/max bbox is valid.
    lons = np.linspace(92.5, 290.0, 1000)
    out = unwrap_to_domain(lons, PAC_CENTER)
    np.testing.assert_allclose(out, lons)
    assert out.min() >= 92.5 - 1e-9 and out.max() <= 290.0 + 1e-9


def test_only_shifts_by_multiples_of_360():
    rng = np.array([-180.0, -70.0, 0.0, 92.5, 180.0, 200.0, 290.0, 359.9])
    out = unwrap_to_domain(rng, PAC_CENTER)
    diff = (out - rng) / 360.0
    np.testing.assert_allclose(diff, np.round(diff), atol=1e-9)


def test_idempotent():
    lons = np.array([-70.0, 92.5, 200.0, 290.0])
    once = unwrap_to_domain(lons, PAC_CENTER)
    twice = unwrap_to_domain(once, PAC_CENTER)
    np.testing.assert_allclose(once, twice)


def test_scalar_and_array_consistent():
    lons = [-70.0, 200.0, 290.0]
    arr = unwrap_to_domain(np.array(lons), PAC_CENTER)
    for i, v in enumerate(lons):
        assert unwrap_to_domain(v, PAC_CENTER).item() == pytest.approx(arr[i])
