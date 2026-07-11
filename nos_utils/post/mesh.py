"""Mesh helpers shared by the post-product writers."""
from __future__ import annotations

import numpy as np


def split_quads(elnode: np.ndarray) -> np.ndarray:
    """Split quad elements into triangles, appending the extras.

    ``elnode`` is the (ne, 3|4) 0-based element table with -1 (or masked)
    padding in the 4th column for triangles. Returns an (ne', 3) int
    array where each quad (a,b,c,d) contributes (a,b,c) and (a,c,d) --
    the same convention as the operational ``utils.split_quads``.
    """
    elnode = np.ma.masked_values(np.asarray(elnode), -1)
    if elnode.shape[1] == 4:
        quad = np.nonzero(
            ~(np.ma.getmaskarray(elnode[:, -1]) | (elnode[:, -1] < 0))
        )[0]
        elnode = np.r_[
            elnode[:, :3],
            np.c_[elnode[quad, 0][:, None], elnode[quad, 2:]],
        ]
    return np.ma.filled(elnode, -1).astype(int)


__all__ = ["split_quads"]
