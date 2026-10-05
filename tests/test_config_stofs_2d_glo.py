"""STOFS-2D-Global ForcingConfig factory and yaml rule."""

import dataclasses

import pytest
yaml = pytest.importorskip("yaml")

from nos_utils.config import ForcingConfig


def test_factory_defaults():
    c = ForcingConfig.for_stofs_2d_glo("20261005", 12)
    assert c.domain == (-180.0, 180.0, -90.0, 90.0)
    assert (c.nws, c.nowcast_hours, c.forecast_hours, c.gfs_resolution) == (14, 6, 180, "0p25")
    assert len(c.tidal_constituents) == 15 and c.adcirc_nodal_reference == "midrun"
    assert not (c.st_lawrence_enabled or c.adt_enabled or c.nudging_enabled
                or c.dynamic_adjust_enabled)


def test_factory_override():
    c = ForcingConfig.for_stofs_2d_glo("20261005", 12, forecast_hours=120,
                                       adcirc_nodal_reference="start")
    assert c.forecast_hours == 120 and c.adcirc_nodal_reference == "start"
    with pytest.raises(ValueError, match="adcirc_nodal_reference"):
        ForcingConfig.for_stofs_2d_glo("20261005", 12, adcirc_nodal_reference="mid")


def _write(tmp_path, body):
    p = tmp_path / "stofs_2d_glo.yaml"
    p.write_text(yaml.safe_dump(body))
    return p


def test_from_yaml_name_rule(tmp_path):
    p = _write(tmp_path, {
        "system": {"name": "stofs_2d_glo"},
        "execution": {"mode": "standalone"},
        "forcing": {"tidal": {"constituents": ["m2", "k1"], "adcirc_nodal_reference": "start"}},
    })
    c = ForcingConfig.from_yaml(p, pdy="20261005", cyc=12)
    assert (c.nws, c.nowcast_hours, c.forecast_hours, c.gfs_resolution) == (14, 6, 180, "0p25")
    assert c.tidal_constituents == ["M2", "K1"] and c.adcirc_nodal_reference == "start"


def test_from_yaml_explicit_values_win(tmp_path):
    p = _write(tmp_path, {
        "system": {"name": "stofs_2d_glo"},
        "model": {"run": {"nowcast_hours": 24, "forecast_hours": 60}},
        "forcing": {"atmospheric": {"gfs": {"resolution": "0.50"}}},
    })
    c = ForcingConfig.from_yaml(p, pdy="20261005", cyc=0)
    assert (c.nowcast_hours, c.forecast_hours, c.gfs_resolution) == (24, 60, "0p50")


def test_other_systems_unchanged(tmp_path):
    p = _write(tmp_path, {"system": {"name": "secofs"}})
    c = ForcingConfig.from_yaml(p, pdy="20261005", cyc=12)
    ref = ForcingConfig.from_yaml(_write(tmp_path, {}), pdy="20261005", cyc=12)
    assert dataclasses.asdict(c) == dataclasses.asdict(ref)
    assert c.nws == 2 and c.adcirc_nodal_reference == "midrun"
