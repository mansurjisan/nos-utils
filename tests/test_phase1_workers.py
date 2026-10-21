import logging

import netCDF4
import pytest

from nos_utils.orchestrator import _phase1_workers


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("NOS_PREP_WORKERS", raising=False)


def test_env_override_wins(monkeypatch):
    monkeypatch.setattr(netCDF4, "__version__", "1.7.2")
    monkeypatch.setenv("NOS_PREP_WORKERS", "3")
    assert _phase1_workers(4) == 3


def test_old_netcdf_parallel(monkeypatch):
    monkeypatch.setattr(netCDF4, "__version__", "1.5.8")
    assert _phase1_workers(4) == 4


def test_new_netcdf_sequential(monkeypatch, caplog):
    monkeypatch.setattr(netCDF4, "__version__", "1.7.2")
    with caplog.at_level(logging.INFO):
        assert _phase1_workers(4) == 1
    assert "Phase 1 sequential" in caplog.text


@pytest.mark.parametrize("bad", ["abc", "0", "-2"])
def test_invalid_env_falls_back(monkeypatch, caplog, bad):
    monkeypatch.setattr(netCDF4, "__version__", "1.7.2")
    monkeypatch.setenv("NOS_PREP_WORKERS", bad)
    with caplog.at_level(logging.WARNING):
        assert _phase1_workers(4) == 1
    assert "Invalid NOS_PREP_WORKERS" in caplog.text
