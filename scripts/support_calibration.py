"""Calibration lines of a generated supported cell.

The generators of the supported-cell blocks write the cells; the collector
(`scripts/collect_coverage_records.py`) binds the measured coverage records onto
them. A generator must therefore reproduce the same binding: a cell that owns
measured records states `calibration = [...]` (and `boundary_record` when every
record at the reported level is a boundary), and only a cell that owns none
states `estimator_grid_not_measured`. Ownership is the collector's own
`supported_record_matches`, so the two cannot disagree.

SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

from functools import lru_cache

import collect_coverage_records as collector


@lru_cache(maxsize=1)
def _registry() -> dict[str, dict]:
    return collector.existing_records()


def owned_records(cell: dict, records: dict[str, dict]) -> list[str]:
    """Sorted ids of the records the cell owns by exact coordinates and estimator."""
    return sorted(rid for rid, rec in records.items() if collector.supported_record_matches(cell, rec))


def calibration_lines(cell: dict, records: dict[str, dict] | None = None) -> str:
    """The trailing TOML lines (newline terminated) that state a cell's calibration."""
    records = _registry() if records is None else records
    ids = owned_records(cell, records)
    if not ids:
        return 'calibration_reason = "estimator_grid_not_measured"\n'
    listed = ", ".join(f'"{rid}"' for rid in ids)
    lines = f"calibration = [{listed}]\n"
    if collector.all_boundary_at_reported_level(ids, records):
        lines += 'calibration_reason = "boundary_record"\n'
    return lines
