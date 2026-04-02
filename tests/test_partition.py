"""Tests for PartitionProcessor."""

from pathlib import Path
import numpy as np
import pytest
from nos_utils.forcing.partition import (
    PartitionProcessor, _round_robin_partition, _contiguous_partition,
)


@pytest.fixture
def mock_hgrid(tmp_path):
    """Create a mock hgrid.gr3 with 100 elements."""
    grid = tmp_path / "hgrid.gr3"
    grid.write_text("SECOFS test grid\n100 50\n")  # n_elements=100, n_nodes=50
    return grid


class TestPartitionFunctions:
    def test_round_robin(self):
        ranks = _round_robin_partition(10, 3)
        assert list(ranks) == [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]

    def test_contiguous(self):
        ranks = _contiguous_partition(10, 3)
        # 10/3 = 3 per rank, remainder 1 → ranks 0,1 get 4,3,3
        assert ranks[0] == 0
        assert ranks[3] == 0  # 4th element still rank 0
        assert ranks[4] == 1
        assert ranks[-1] == 2

    def test_single_proc(self):
        ranks = _round_robin_partition(100, 1)
        assert all(r == 0 for r in ranks)


class TestPartitionProcessor:
    def test_basic(self, mock_config, mock_hgrid, tmp_path):
        proc = PartitionProcessor(
            mock_config, mock_hgrid.parent, tmp_path / "out",
            nprocs=4, grid_file=mock_hgrid,
        )
        result = proc.process()

        assert result.success
        assert result.metadata["n_elements"] == 100
        assert result.metadata["nprocs"] == 4

        partition = (tmp_path / "out" / "partition.prop").read_text().strip().split("\n")
        assert len(partition) == 100
        # All ranks should be 0-3
        ranks = [int(r) for r in partition]
        assert set(ranks) == {0, 1, 2, 3}

    def test_no_grid_fails(self, mock_config, tmp_path):
        proc = PartitionProcessor(
            mock_config, tmp_path / "empty", tmp_path / "out", nprocs=4,
        )
        result = proc.process()
        assert not result.success
