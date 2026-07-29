"""Cycle-hour contract for :func:`nos_utils.nco_bridge.config_from_env`.

The cycle used to default to 12 when `cyc` was unset, so a run driven with
only `CYC` set silently prepped the wrong hour and still reported success.
"""
import pytest

from nos_utils.nco_bridge import config_from_env

_NCO_VARS = (
    "PDY", "cyc", "CYC", "RUN", "OFS_CONFIG", "DATA", "DATAROOT",
    "COMOUT", "COMIN", "FIXofs", "COMINgfs", "COMINhrrr", "COMINnwm",
    "COMINrtofs", "COMINrtofs_2d", "COMINrtofs_3d", "COMINlaw",
    "COMINadt", "COMINrerun", "RESTART_DIR", "USE_DATM",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Start from a bare environment so a developer's shell cannot mask a bug."""
    for var in _NCO_VARS:
        monkeypatch.delenv(var, raising=False)


def _base_env(monkeypatch, tmp_path):
    monkeypatch.setenv("PDY", "20260724")
    monkeypatch.setenv("RUN", "secofs")
    monkeypatch.setenv("DATA", str(tmp_path / "work"))
    monkeypatch.setenv("COMOUT", str(tmp_path / "com"))


def test_lowercase_cyc_accepted(monkeypatch, tmp_path):
    _base_env(monkeypatch, tmp_path)
    monkeypatch.setenv("cyc", "06")
    config, _ = config_from_env()
    assert config.cyc == 6


def test_uppercase_CYC_accepted(monkeypatch, tmp_path):
    """PBS cards export lowercase `cyc`; launcher/Slurm paths pass `CYC`.

    Reading only the lowercase form silently ran the 12z cycle -- the same bug
    that previously bit the ParMETIS retry and the parity drill.
    """
    _base_env(monkeypatch, tmp_path)
    monkeypatch.setenv("CYC", "18")
    config, _ = config_from_env()
    assert config.cyc == 18


def test_lowercase_wins_when_both_set(monkeypatch, tmp_path):
    _base_env(monkeypatch, tmp_path)
    monkeypatch.setenv("cyc", "00")
    monkeypatch.setenv("CYC", "12")
    config, _ = config_from_env()
    assert config.cyc == 0


def test_missing_cycle_raises_rather_than_defaulting(monkeypatch, tmp_path):
    _base_env(monkeypatch, tmp_path)
    with pytest.raises(EnvironmentError, match="refusing to guess the cycle"):
        config_from_env()


@pytest.mark.parametrize("bad", ["ab", "24", "-1"])
def test_invalid_cycle_rejected(monkeypatch, tmp_path, bad):
    _base_env(monkeypatch, tmp_path)
    monkeypatch.setenv("cyc", bad)
    with pytest.raises(EnvironmentError):
        config_from_env()
