"""Tests for the elevation-forced OBC segment subset fix.

Confirms that the RTOFS boundary node set used for elev2D / TEM_3D /
SAL_3D / uv3D ``*.th.nc`` is restricted to the elevation-forced
(SCHISM iettype 4/5) open-boundary segments when ``obc_elev_segments``
is configured, and that the flow-only segment (iettype 0, e.g.
STOFS-3D-ATL's St-Lawrence River boundary) is excluded.

Background: SCHISM aborts at ``misc_subs.F90:641`` when the number of
open nodes in elev2D.th.nc does not equal the sum of hgrid nodes over
the elevation-forced segments only. STOFS-3D-ATL ships no obc.ctl, so
the writer previously stamped all 781 open-boundary nodes; the fix
restricts it to segments [0, 1] = 778 nodes.
"""

from pathlib import Path

import numpy as np
import pytest

from nos_utils.config import ForcingConfig
from nos_utils.forcing.rtofs import RTOFSProcessor
from nos_utils.io.schism_grid import SchismGrid


def _write_hgrid(path: Path, seg_sizes):
    """Write a minimal valid hgrid.ll with len(seg_sizes) open-boundary segments.

    Node coordinates are arbitrary but distinct so per-segment ordering can
    be asserted. The boundary node IDs are allocated contiguously per
    segment from a flat pool, so segment ``s`` owns a known ID block.

    Returns the per-segment list of 1-based node IDs.
    """
    n_bnd_total = sum(seg_sizes)
    # A few extra interior nodes so node IDs > boundary IDs also exist.
    n_nodes = n_bnd_total + 4
    n_elements = 1  # single dummy triangle; element body is skipped by reader

    lines = ["test grid"]
    lines.append(f"{n_elements} {n_nodes}")
    for i in range(n_nodes):
        nid = i + 1
        # lon/lat encode the node id so subset ordering is verifiable.
        lon = -80.0 + nid * 0.01
        lat = 30.0 + nid * 0.01
        depth = 10.0 + nid
        lines.append(f"{nid} {lon:.4f} {lat:.4f} {depth:.4f}")

    # Dummy element connectivity (3-node triangle). Reader skips n_elements lines.
    lines.append("1 3 1 2 3")

    # Open-boundary section. Reader parses the leading integer of each
    # header line (everything before '=' / first whitespace token).
    lines.append(f"{len(seg_sizes)} = Number of open boundaries")
    lines.append(f"{n_bnd_total} = Total number of open boundary nodes")

    seg_node_ids = []
    next_id = 1
    for size in seg_sizes:
        lines.append(f"{size} = Number of nodes for open boundary")
        ids = list(range(next_id, next_id + size))
        seg_node_ids.append(ids)
        for nid in ids:
            lines.append(str(nid))
        next_id += size

    path.write_text("\n".join(lines) + "\n")
    return seg_node_ids


class TestOpenBoundaryNodesSubset:
    """Unit tests for SchismGrid.open_boundary_nodes_subset."""

    def test_subset_excludes_flow_segment(self, tmp_path):
        # 3 segments mirroring STOFS-3D-ATL structure: [B0, B1, B2(flow)].
        grid_path = tmp_path / "test.hgrid.ll"
        seg_ids = _write_hgrid(grid_path, seg_sizes=[5, 3, 2])
        grid = SchismGrid.read(grid_path)

        assert len(grid.open_boundaries) == 3
        # Full set is all 10 nodes.
        all_lons, _, _, all_ids = grid.open_boundary_nodes()
        assert len(all_lons) == 10
        assert len(all_ids) == 10

        # Elevation segments [0, 1] -> 8 nodes (not 10).
        lons, lats, depths, ids = grid.open_boundary_nodes_subset([0, 1])
        assert len(lons) == 8
        assert len(lats) == 8
        assert len(depths) == 8
        assert len(ids) == 8

        # Order: segment 0 node IDs, then segment 1 node IDs (B2 excluded).
        expected_ids = seg_ids[0] + seg_ids[1]
        assert ids == expected_ids
        # Coordinates follow the same order as the node IDs.
        expected_lons = np.array([-80.0 + nid * 0.01 for nid in expected_ids])
        np.testing.assert_allclose(lons, expected_lons)

    def test_subset_single_segment(self, tmp_path):
        grid_path = tmp_path / "test.hgrid.ll"
        seg_ids = _write_hgrid(grid_path, seg_sizes=[5, 3, 2])
        grid = SchismGrid.read(grid_path)

        lons, _, _, ids = grid.open_boundary_nodes_subset([0])
        assert len(ids) == 5
        assert ids == seg_ids[0]

    def test_subset_order_independent_of_argument_order(self, tmp_path):
        """[0, 1] and [1, 0] both select 8 nodes; ordering follows the arg."""
        grid_path = tmp_path / "test.hgrid.ll"
        seg_ids = _write_hgrid(grid_path, seg_sizes=[5, 3, 2])
        grid = SchismGrid.read(grid_path)

        _, _, _, ids_01 = grid.open_boundary_nodes_subset([0, 1])
        _, _, _, ids_10 = grid.open_boundary_nodes_subset([1, 0])
        assert len(ids_01) == len(ids_10) == 8
        assert set(ids_01) == set(ids_10)
        assert ids_01 == seg_ids[0] + seg_ids[1]
        assert ids_10 == seg_ids[1] + seg_ids[0]

    def test_subset_out_of_range_raises(self, tmp_path):
        grid_path = tmp_path / "test.hgrid.ll"
        _write_hgrid(grid_path, seg_sizes=[5, 3, 2])
        grid = SchismGrid.read(grid_path)
        with pytest.raises(IndexError):
            grid.open_boundary_nodes_subset([0, 3])

    def test_open_boundary_nodes_unchanged(self, tmp_path):
        """The original concatenation method is untouched (all 10 nodes)."""
        grid_path = tmp_path / "test.hgrid.ll"
        _write_hgrid(grid_path, seg_sizes=[5, 3, 2])
        grid = SchismGrid.read(grid_path)
        lons, lats, depths, ids = grid.open_boundary_nodes()
        assert len(lons) == len(lats) == len(depths) == len(ids) == 10


class TestLoadGridSubset:
    """Driving RTOFSProcessor._load_grid with obc_elev_segments."""

    def test_load_grid_subsets_to_elevation_segments(self, tmp_path):
        grid_path = tmp_path / "test.hgrid.ll"
        _write_hgrid(grid_path, seg_sizes=[5, 3, 2])  # 10 total, elev sum = 8

        cfg = ForcingConfig.for_stofs_3d_atl(pdy="20260401", cyc=12)
        cfg.obc_elev_segments = [0, 1]
        proc = RTOFSProcessor(
            cfg, tmp_path, tmp_path / "out", grid_file=grid_path,
        )
        assert proc._load_grid() is True

        # Boundary-node count equals the elevation-segment sum, not 10.
        assert len(proc._bnd_lons) == 8
        # All four *.th.nc writers derive n_bnd from len(self._bnd_lons),
        # so TEM_3D / SAL_3D / uv3D share the same subset count.
        n_bnd = len(proc._bnd_lons)
        assert len(proc._bnd_lats) == n_bnd
        assert len(proc._bnd_depths) == n_bnd
        assert len(proc._bnd_ids) == n_bnd
        assert n_bnd == 8

    def test_load_grid_without_segments_uses_all_nodes(self, tmp_path):
        """No obc_elev_segments and no obc.ctl -> full open-boundary set."""
        grid_path = tmp_path / "test.hgrid.ll"
        _write_hgrid(grid_path, seg_sizes=[5, 3, 2])

        cfg = ForcingConfig.for_secofs(pdy="20260401", cyc=12)
        assert cfg.obc_elev_segments is None
        proc = RTOFSProcessor(
            cfg, tmp_path, tmp_path / "out", grid_file=grid_path,
        )
        assert proc._load_grid() is True
        assert len(proc._bnd_lons) == 10  # all nodes, no subset

    def test_load_grid_obc_ctl_bypasses_subset(self, tmp_path):
        """SECOFS guard: an obc.ctl present takes the `if` branch and the
        subset logic is never reached, even if obc_elev_segments is set."""
        grid_path = tmp_path / "test.hgrid.ll"
        _write_hgrid(grid_path, seg_sizes=[5, 3, 2])

        # obc.ctl listing only 4 nodes (1..4) in SECTION 2.
        ctl_path = tmp_path / "secofs.obc.ctl"
        ctl_lines = [
            "SECTION 1",
            "header",
            "SECTION 2",
            "COL NODE_ID extra",
        ]
        for nid in [1, 2, 3, 4]:
            ctl_lines.append(f"0 {nid} 0.0")
        ctl_path.write_text("\n".join(ctl_lines) + "\n")

        cfg = ForcingConfig.for_secofs(pdy="20260401", cyc=12)
        # Even if someone set obc_elev_segments, the obc.ctl branch wins.
        cfg.obc_elev_segments = [0, 1]
        proc = RTOFSProcessor(
            cfg, tmp_path, tmp_path / "out",
            grid_file=grid_path, obc_ctl_file=ctl_path,
        )
        assert proc._load_grid() is True
        # Count == obc.ctl node count (4), NOT the subset (8) or all (10).
        assert len(proc._bnd_lons) == 4
        assert proc._bnd_ids == [1, 2, 3, 4]


class TestConfigContract:
    """obc_elev_segments factory + from_yaml contract."""

    def test_stofs_3d_atl_has_elev_segments(self):
        cfg = ForcingConfig.for_stofs_3d_atl(pdy="20260401", cyc=12)
        assert cfg.obc_elev_segments == [0, 1]

    def test_stofs_3d_atl_ufs_has_elev_segments(self):
        cfg = ForcingConfig.for_stofs_3d_atl_ufs(pdy="20260401", cyc=12)
        assert cfg.obc_elev_segments == [0, 1]

    def test_secofs_elev_segments_none(self):
        cfg = ForcingConfig.for_secofs(pdy="20260401", cyc=12)
        assert cfg.obc_elev_segments is None

    def test_secofs_ufs_elev_segments_none(self):
        cfg = ForcingConfig.for_secofs_ufs(pdy="20260401", cyc=12)
        assert cfg.obc_elev_segments is None

    def test_default_config_elev_segments_none(self):
        cfg = ForcingConfig(
            lon_min=-80.0, lon_max=-70.0,
            lat_min=25.0, lat_max=35.0,
            pdy="20260401", cyc=12,
        )
        assert cfg.obc_elev_segments is None

    def test_from_yaml_parses_elev_segments(self, tmp_path):
        pytest.importorskip("yaml")
        yaml_path = tmp_path / "stofs.yaml"
        yaml_path.write_text(
            "grid:\n"
            "  domain: {lon_min: -98.5, lon_max: -52.5, lat_min: 7.3, lat_max: 52.6}\n"
            "forcing:\n"
            "  ocean:\n"
            "    obc:\n"
            "      roi_2ds: {x1: 1, x2: 2, y1: 1, y2: 2}\n"
            "      elev_segments: [0, 1]\n"
        )
        cfg = ForcingConfig.from_yaml(yaml_path, pdy="20260401", cyc=12)
        assert cfg.obc_elev_segments == [0, 1]

    def test_from_yaml_omitted_elev_segments_none(self, tmp_path):
        pytest.importorskip("yaml")
        yaml_path = tmp_path / "secofs.yaml"
        yaml_path.write_text(
            "grid:\n"
            "  domain: {lon_min: -88.0, lon_max: -63.0, lat_min: 17.0, lat_max: 40.0}\n"
            "forcing:\n"
            "  ocean:\n"
            "    obc:\n"
            "      ssh_offset: 1.25\n"
        )
        cfg = ForcingConfig.from_yaml(yaml_path, pdy="20260401", cyc=12)
        assert cfg.obc_elev_segments is None
