"""Tests for NudgingProcessor."""

from pathlib import Path
import pytest
from nos_utils.config import ForcingConfig
from nos_utils.forcing.nudging import NudgingProcessor


class TestNudgingProcessor:
    def test_disabled_returns_success(self, mock_config, tmp_path):
        mock_config.nudging_enabled = False
        proc = NudgingProcessor(mock_config, tmp_path, tmp_path / "out")
        result = proc.process()
        assert result.success
        assert "disabled" in result.warnings[0].lower()

    def test_no_input_returns_failure(self, mock_config, tmp_path):
        mock_config.nudging_enabled = True
        proc = NudgingProcessor(mock_config, tmp_path / "empty", tmp_path / "out")
        result = proc.process()
        assert not result.success

    def test_source_name(self):
        assert NudgingProcessor.SOURCE_NAME == "NUDGING"
