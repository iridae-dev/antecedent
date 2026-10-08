"""The no-model fallback ordering: verified sufficiency, then cost units, budget and id."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from .._native import rank_structural_designs as _rank_structural
from .evsi import _raise


@dataclass(frozen=True, slots=True)
class StructuralCandidate:
    """A candidate with a verified structural verdict, for the no-model ordering."""

    id: str
    verified_sufficient: bool
    cost_units: int
    sample_budget: int = 0


@dataclass(frozen=True, slots=True)
class StructuralEntry:
    """One structurally ranked candidate."""

    id: str
    rank: int
    verified_sufficient: bool
    cost_units: int
    sample_budget: int


@dataclass(frozen=True, slots=True)
class StructuralRanking:
    """The preserved 2.2 ordering: verified sufficiency, then cost units, budget, id."""

    entries: tuple[StructuralEntry, ...]
    identity: str
    basis: Literal["structural_sufficiency_cost"] = "structural_sufficiency_cost"


def rank_structural(candidates: Sequence[StructuralCandidate]) -> StructuralRanking:
    """Order candidates when no probabilistic model is licensed.

    Verified structural sufficiency first, then fewer cost units, then a smaller sample
    budget, then the semantic id; invariant to input order. This is neither an
    identification probability nor a value of information: it never reports either. Use
    :func:`antecedent.design.rank_designs` once a structure prior or a decision exists.
    """
    wire = [
        {
            "semantic_id": c.id,
            "verified_sufficient": bool(c.verified_sufficient),
            "cost_units": int(c.cost_units),
            "sample_budget": int(c.sample_budget),
        }
        for c in candidates
    ]
    body, refusal = _rank_structural(json.dumps(wire))
    _raise(refusal)
    assert body is not None
    parsed = json.loads(body)
    return StructuralRanking(
        entries=tuple(
            StructuralEntry(
                id=e["semantic_id"],
                rank=int(e["rank"]),
                verified_sufficient=bool(e["verified_sufficient"]),
                cost_units=int(e["cost_units"]),
                sample_budget=int(e["sample_budget"]),
            )
            for e in parsed["entries"]
        ),
        identity=parsed["identity"],
    )


__all__ = ["StructuralCandidate", "StructuralEntry", "StructuralRanking", "rank_structural"]
