"""Mechanical Rust↔Python parity for the identifier and estimator wire-id mirrors.

``antecedent.ids.Identifier`` / ``Estimator`` are hand-written Python mirrors of the
Rust ``IdentifierId`` / ``EstimatorId`` wire ids. Nothing stopped them drifting: 2.1
added 26 Rust estimator ids and only 4 reached ``ids.py``, so
``Estimator(result.estimate.estimator_id)`` raised on ids the result itself reported.
This test derives the authoritative sets from the Rust source and fails if either
mirror is missing or has an extra id, so the two stay in lockstep by construction.
"""

from __future__ import annotations

import re
from pathlib import Path

from antecedent.ids import Estimator, Identifier

_IDS_RS = (
    Path(__file__).resolve().parents[2]
    / "crates"
    / "antecedent"
    / "src"
    / "strategy_table"
    / "ids.rs"
)


def _rust_wire_ids() -> tuple[set[str], set[str]]:
    """Wire ids from the two ``*_data`` tables: identifiers, then estimators."""
    text = _IDS_RS.read_text(encoding="utf-8")
    split = text.index("fn estimator_data")
    name = re.compile(r'name:\s*"([a-z0-9_.]+)"')
    identifiers = set(name.findall(text[:split]))
    estimators = set(name.findall(text[split:]))
    return identifiers, estimators


def test_identifier_mirror_matches_rust() -> None:
    rust_identifiers, _ = _rust_wire_ids()
    python = {member.value for member in Identifier}
    assert python == rust_identifiers, {
        "missing_from_python": sorted(rust_identifiers - python),
        "extra_in_python": sorted(python - rust_identifiers),
    }


def test_estimator_mirror_matches_rust() -> None:
    _, rust_estimators = _rust_wire_ids()
    python = {member.value for member in Estimator}
    assert python == rust_estimators, {
        "missing_from_python": sorted(rust_estimators - python),
        "extra_in_python": sorted(python - rust_estimators),
    }
