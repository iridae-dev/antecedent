"""Calibration lines of a generated supported cell.

The generators of the supported-cell blocks write the cells. A cell binds no record
list: its coverage records are the ones whose test executed it (`parity/unit_index.json`,
ADR 0031). A cell that publishes an interval and that no record measures yet states
`estimator_grid_not_measured`; one that publishes no interval states
`no_interval_reported` itself.

SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

from functools import lru_cache

import gate_support
import unit_index


@lru_cache(maxsize=1)
def _linked() -> dict[str, list[str]]:
    return unit_index.matrix_records()


def calibration_lines(cell: dict, linked: dict[str, list[str]] | None = None) -> str:
    """The trailing TOML lines (newline terminated) that state a cell's calibration."""
    linked = _linked() if linked is None else linked
    if linked.get(gate_support.cell_ident(cell)):
        return ""
    return 'calibration_reason = "estimator_grid_not_measured"\n'
