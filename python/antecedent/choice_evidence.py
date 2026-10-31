"""Evidence-to-decision lifecycle for repeated categorical choices.

Qualified prior evidence, supported observations, licensed inference, decision evaluation and
next-study ranking, each carrying its meaning, dependence and support conditions:

* :class:`Frame` is the parameterization of a choice law: ordered options, the population choosing
  and what a choice means (``stated_preference`` or ``observed_behavior``).
* :class:`Source` is retained evidence reused as prior; it is admitted only when its option
  order, population and meaning match the target, or a transfer rationale and a discount are
  declared. The same observation identity is never counted twice, and conflict between sources is
  reported (or refused).
* :func:`evaluate` conditions on :class:`Observations`, reports the Dirichlet posterior (exact
  probability each option has the largest share, expected loss), evaluates a sequential
  :class:`Rule` (a computation, not an assignment) and ranks candidate studies by the expected
  value of sample information net of cost.

Stated preferences do not identify behavioural or causal outcomes, and no calibrated coverage or
frequentist error rate is claimed. Every number, identity and refusal is computed by the Rust
core; a refusal raises :class:`ChoiceEvidenceRefusal`.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from . import _native
from ._refusal import StructuredRefusal, raise_refusal
from .errors import CausalValueError

__all__ = [
    "Candidate",
    "ChoiceEvidenceRefusal",
    "Evaluation",
    "Frame",
    "Observations",
    "Rule",
    "Source",
    "evaluate",
]


class ChoiceEvidenceRefusal(StructuredRefusal):
    """A typed refusal of the categorical-choice lifecycle.

    ``detail`` is a ``choice_evidence.*`` reason and ``remedy`` the next attempt.
    """


@dataclass(frozen=True)
class Frame:
    """Ordered options, the population choosing and the meaning of a choice."""

    options: Sequence[str]
    population: str
    meaning: str = "stated_preference"

    def wire(self) -> dict[str, Any]:
        return {
            "options": list(self.options),
            "population": self.population,
            "meaning": self.meaning,
        }


@dataclass(frozen=True)
class Source:
    """Retained choice counts reused as prior evidence at a power-prior ``discount``."""

    label: str
    frame: Frame
    counts: Sequence[float]
    observation_id: str
    discount: float
    transfer_rationale: str | None = None

    def wire(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "frame": self.frame.wire(),
            "counts": [float(c) for c in self.counts],
            "observation_id": self.observation_id,
            "discount": float(self.discount),
            "transfer_rationale": self.transfer_rationale,
        }


@dataclass(frozen=True)
class Observations:
    """New choice counts and the identity of the observation set."""

    counts: Sequence[float]
    observation_id: str


@dataclass(frozen=True)
class Rule:
    """A bounded Bayesian sequential rule."""

    min_probability_best: float
    max_expected_loss: float
    min_observations: float
    max_observations: float
    allocation_floor: float = 0.0


@dataclass(frozen=True)
class Candidate:
    """A candidate additional study: ``observations`` choices at ``cost``."""

    label: str
    observations: int
    cost: float = 0.0


@dataclass(frozen=True)
class Evaluation:
    """Every stage that was requested, as computed by the Rust core."""

    raw: dict[str, Any] = field(repr=False)

    @property
    def prior(self) -> dict[str, Any]:
        return dict(self.raw["prior"])

    @property
    def inference(self) -> dict[str, Any] | None:
        value = self.raw.get("inference")
        return None if value is None else dict(value)

    @property
    def look(self) -> dict[str, Any] | None:
        value = self.raw.get("look")
        return None if value is None else dict(value)

    @property
    def ranking(self) -> list[dict[str, Any]] | None:
        value = self.raw.get("ranking")
        return None if value is None else [dict(row) for row in value]


def evaluate(
    frame: Frame,
    initial: Sequence[float],
    *,
    sources: Sequence[Source] = (),
    conflict_policy: str = "report",
    observations: Observations | None = None,
    rule: Rule | None = None,
    values: Sequence[float] | None = None,
    candidates: Sequence[Candidate] = (),
    seed: int = 0,
) -> Evaluation:
    """Run the lifecycle through the stages that were requested.

    ``rule`` and ``candidates`` need ``observations``. ``values`` gives the value of each option
    when chosen, per unit share; it is required to rank ``candidates``.
    """
    payload: dict[str, Any] = {
        "frame": frame.wire(),
        "initial": [float(a) for a in initial],
        "sources": [s.wire() for s in sources],
        "conflict_policy": conflict_policy,
    }
    if observations is not None:
        payload["observations"] = {
            "counts": [float(c) for c in observations.counts],
            "observation_id": observations.observation_id,
        }
    if rule is not None:
        payload["rule"] = {
            "min_probability_best": rule.min_probability_best,
            "max_expected_loss": rule.max_expected_loss,
            "min_observations": rule.min_observations,
            "max_observations": rule.max_observations,
            "allocation_floor": rule.allocation_floor,
        }
    if candidates:
        if values is None:
            raise CausalValueError("ranking candidates needs `values`, one per option")
        payload["ranking"] = {
            "values": [float(v) for v in values],
            "candidates": [
                {"label": c.label, "observations": c.observations, "cost": float(c.cost)}
                for c in candidates
            ],
            "seed": seed,
        }
    value, refusal = _native.choice_evidence_evaluate(json.dumps(payload))
    raise_refusal(refusal, ChoiceEvidenceRefusal)
    if value is None:  # pragma: no cover - the native layer returns one or the other
        raise CausalValueError("the native layer returned neither a result nor a refusal")
    return Evaluation(json.loads(value))
