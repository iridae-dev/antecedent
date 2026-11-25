"""Copy the winning candidate id and sample size a ranking already stored.

not_scientific: the rank-0 row is copied. This does not rank, does not call
``design.consume``, and creates no support or coverage unit.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

from ..errors import CausalTypeError, CausalValueError
from ._declarations import CandidateValue
from .plans import PLAN_TYPES
from .ranking import DesignRankingResult
from .structural import StructuralEntry, StructuralRanking


@dataclass(frozen=True, slots=True)
class WinningDesign:
    """Candidate id and sample size copied from the ranking's rank-0 row."""

    candidate_id: str
    n: int


def project_winning_design(
    ranking: DesignRankingResult | StructuralRanking | Sequence[object],
) -> WinningDesign:
    """Return the candidate id and sample size the ranking stored for its winner.

    A :class:`DesignRankingResult` or :class:`StructuralRanking` is read by its stored
    ``rank``. A choice-study ranking is the sequence
    :func:`antecedent.choice_inference.rank_studies` stored, winner first; that stored
    order is the rank.
    """
    from .._native import project_winning_design as project_rows

    body = project_rows(json.dumps({"rows": _stored_rows(ranking)}))
    parsed = json.loads(body)
    return WinningDesign(candidate_id=parsed["candidate_id"], n=parsed["n"])


def _stored_rows(ranking: object) -> list[dict[str, object]]:
    if isinstance(ranking, DesignRankingResult):
        return [_design_row(item) for item in ranking.candidates]
    if isinstance(ranking, StructuralRanking):
        return [_design_row(item) for item in ranking.entries]
    if isinstance(ranking, Sequence) and not isinstance(ranking, (str, bytes)):
        return [_choice_row(item, index) for index, item in enumerate(ranking)]
    raise CausalTypeError(
        "project_winning_design reads a DesignRankingResult, a StructuralRanking, "
        "or the choice-study ranking in the order it was stored"
    )


def _design_row(item: object) -> dict[str, object]:
    candidate_id = getattr(item, "id", None)
    rank = getattr(item, "rank", None)
    if not isinstance(candidate_id, str) or isinstance(rank, bool) or not isinstance(rank, int):
        raise CausalTypeError("the ranking stored no candidate id and rank")
    if rank < 0:
        raise CausalValueError("the ranking stored a negative rank")
    return {"candidate_id": candidate_id, "rank": rank, "n": _stored_n(item)}


def _choice_row(item: object, index: int) -> dict[str, object]:
    label = getattr(item, "label", None)
    observations = getattr(item, "observations", None)
    if (
        not isinstance(label, str)
        or isinstance(observations, bool)
        or not isinstance(observations, int)
        or not hasattr(item, "net_value")
    ):
        raise CausalTypeError("a choice-study ranking stores label and observations on each study")
    if observations < 0:
        raise CausalValueError("the ranking stored a negative sample size")
    return {"candidate_id": label, "rank": index, "n": observations}


def _stored_n(item: object) -> int:
    if isinstance(item, CandidateValue):
        return _nonnegative(item.sample_size)
    if isinstance(item, StructuralEntry):
        return _nonnegative(item.sample_budget)
    plan = getattr(item, "plan", None)
    if isinstance(plan, PLAN_TYPES):
        return _plan_n(plan)
    raise CausalValueError("the ranking stored no sample size")


def _plan_n(plan: object) -> int:
    kind = getattr(plan, "kind", None)
    if kind == "increase_sampling_rate":
        return _nonnegative(getattr(plan, "additional_samples", None))
    if kind == "observe_environment":
        return _nonnegative(getattr(plan, "additional_rows", None))
    return _nonnegative(getattr(plan, "sample_budget", None))


def _nonnegative(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CausalValueError("the ranking stored no sample size")
    return value
