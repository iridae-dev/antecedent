"""The no-model fallback ordering: verified sufficiency, then cost units, budget and id."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from .._native import rank_structural_designs as _rank_structural
from ._declarations import _raise


@dataclass(frozen=True, slots=True)
class StructuralCandidate:
    """Caller-declared structural sufficiency and costs, without a probabilistic model."""

    id: str
    verified_sufficient: bool
    cost_units: int
    sample_budget: int = 0

    def __post_init__(self) -> None:
        from ..errors import CausalTypeError, CausalValueError
        from .plans import _count

        if not isinstance(self.id, str):
            raise CausalTypeError("id must be a string")
        if not self.id:
            raise CausalValueError("id must be non-empty")
        if not isinstance(self.verified_sufficient, bool):
            raise CausalTypeError("verified_sufficient must be a bool")
        _count(self.cost_units, "cost_units")
        _count(self.sample_budget, "sample_budget")


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


def _rank_structural_result(candidates: Sequence[StructuralCandidate]) -> StructuralRanking:
    """Order candidates when no probabilistic model is licensed.

    Verified structural sufficiency first, then fewer cost units, then a smaller sample
    budget, then the semantic id; invariant to input order. This is neither an
    identification probability nor a value of information: it never reports either. Use
    :func:`antecedent.design.rank_designs` once a structure prior or a decision exists.
    """
    wire = [
        {
            "semantic_id": c.id,
            "verified_sufficient": c.verified_sufficient,
            "cost_units": c.cost_units,
            "sample_budget": c.sample_budget,
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


def rank_structural(candidates: Sequence[StructuralCandidate]) -> StructuralRanking:
    """Compatibility view of ``rank_designs(candidates)``; no numeric score is inferred.

    The sufficiency flag is declared by the caller. This ordering does not perform
    identification or prove the declared verdict.
    """
    from .ranking import rank_designs

    result = rank_designs(candidates)
    assert result._structural is not None
    return result._structural


__all__ = ["StructuralCandidate", "StructuralEntry", "StructuralRanking", "rank_structural"]
