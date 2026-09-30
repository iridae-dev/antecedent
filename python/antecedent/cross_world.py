"""Path-specific edge interventions on a fixed Markovian DAG.

One cell: which edges of a supplied explicit DAG see the intervened treatment
value and which see the baseline, evaluated as ``E[Y_1] - E[Y_0]`` with one
abduced exogenous term per unit and variable shared by both worlds. The natural
direct effect is the edge set ``{treatment -> outcome}`` and the natural indirect
effect is ``{treatment -> mediator, mediator -> outcome}``.

The claim is a point. No sampling interval or posterior is published, and
accepted, uncertain, latent-confounded (``Admg``), equivalence-class and
temporal structures are refused. This is not general counterfactual
identification: an edge set with a recanting witness is refused with
``cross_world_not_identified``, and the artifact stores the derivation
(assumptions, rerouted edges, the counterfactual nodes the estimand needs) that
a consumer replays from the stored graph, query and table (same evaluator, plus a
closed-form cross-check for the linear-Gaussian family).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ._data import as_columns
from ._native import (
    consume_cross_world_edge_contrast_artifact as _consume,
)
from ._native import (
    evaluate_cross_world_edge_contrast as _evaluate,
)
from .accepted_graph import AcceptedGraph
from .errors import CausalTypeError, CausalUnsupportedError, CausalValueError
from .graph import Dag

__all__ = [
    "CrossWorldEffect",
    "EdgeIntervention",
    "consume_cross_world_artifact",
    "path_specific_effect",
]

_MECHANISMS = ("linear_gaussian", "non_separable_basis")


def _name(value: object, what: str) -> str:
    if not isinstance(value, str) or not value:
        raise CausalTypeError(f"{what} must be a non-empty string")
    return value


@dataclass(frozen=True, slots=True)
class EdgeIntervention:
    """Which edges see the intervened treatment value.

    ``edges`` lists the graph edges whose child reads its parent from the
    intervened world (treatment ``active``); every other edge reads the baseline
    world (treatment ``control``). The set is canonicalized, so its order does not
    matter.
    """

    treatment: str
    outcome: str
    control: float
    active: float
    edges: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        _name(self.treatment, "treatment")
        _name(self.outcome, "outcome")
        if self.treatment == self.outcome:
            raise CausalValueError(
                "cross_world.invalid_query: treatment and outcome must be distinct",
                reason_code="invalid_argument",
            )
        control, active = float(self.control), float(self.active)
        if not (math.isfinite(control) and math.isfinite(active)) or control == active:
            raise CausalValueError(
                "cross_world.invalid_query: treatment levels must be finite and distinct",
                reason_code="invalid_argument",
            )
        edges = tuple(sorted({(_name(a, "edge"), _name(b, "edge")) for a, b in self.edges}))
        object.__setattr__(self, "control", control)
        object.__setattr__(self, "active", active)
        object.__setattr__(self, "edges", edges)

    @classmethod
    def natural_direct(
        cls,
        treatment: str,
        mediator: str,
        outcome: str,
        *,
        control: float = 0.0,
        active: float = 1.0,
    ) -> EdgeIntervention:
        """``Y(active, M(control)) - Y(control, M(control))``: the edge ``treatment -> outcome``."""
        _name(mediator, "mediator")
        return cls(treatment, outcome, control, active, ((treatment, outcome),))

    @classmethod
    def natural_indirect(
        cls,
        treatment: str,
        mediator: str,
        outcome: str,
        *,
        control: float = 0.0,
        active: float = 1.0,
    ) -> EdgeIntervention:
        """``Y(control, M(active)) - Y(control, M(control))``: the path through the mediator."""
        _name(mediator, "mediator")
        return cls(
            treatment, outcome, control, active, ((treatment, mediator), (mediator, outcome))
        )


@dataclass(frozen=True, slots=True)
class CrossWorldEffect:
    """The point, its per-unit contrasts, the derivation and the artifact.

    ``unit_effects`` holds one contrast per unit (row order), each computed from
    that unit's own abduced exogenous terms in both worlds; the point is their
    mean. ``data_digest`` identifies the factual table the point was computed on.
    ``independently_verified`` is true on a consumed artifact of the
    ``linear_gaussian`` family whose point a separate closed-form least-squares
    recomputation reproduced. Consistency (a unit's observed values are its values
    under the treatment received) is a named assumption that holds by construction
    of abduction; it is not checked from the data.
    """

    point: float
    unit_effects: tuple[float, ...]
    witness: Mapping[str, Any]
    query_text: str
    mechanism: str
    artifact: bytes
    data_digest: str = ""
    independently_verified: bool = False


def _effect(payload: tuple[str, bytes]) -> CrossWorldEffect:
    body = json.loads(payload[0])
    return CrossWorldEffect(
        point=float(body["point"]),
        unit_effects=tuple(float(x) for x in body["unit_effects"]),
        witness=body["witness"],
        query_text=body["query_text"],
        mechanism=body["mechanism"],
        artifact=bytes(payload[1]),
        data_digest=str(body["data_digest"]),
        independently_verified=bool(body["independently_verified"]),
    )


def _refuse_structure(graph: object) -> None:
    if isinstance(graph, Dag):
        return
    if isinstance(graph, AcceptedGraph) or hasattr(graph, "nodes"):
        raise CausalUnsupportedError(
            "cross_world.graph_outside_contract: only a supplied explicit Dag is licensed; "
            "accepted, uncertain, latent-confounded (Admg), equivalence-class and temporal "
            "structures are refused",
            reason_code="cell_not_licensed",
        )
    raise CausalTypeError("graph must be a Dag")


def path_specific_effect(
    graph: Dag,
    data: Any,
    intervention: EdgeIntervention,
    *,
    mechanism: str = "linear_gaussian",
    uncertainty: str | None = None,
    seed: int = 1,
) -> CrossWorldEffect:
    """Evaluate a path-specific edge contrast on ``graph`` and ``data``.

    ``mechanism`` is ``"linear_gaussian"`` (separable) or ``"non_separable_basis"``
    (a cross-parent basis for the outcome, which makes each unit's abduced
    disturbance observable in the answer). ``uncertainty`` must be ``None``: any
    interval or Bayesian request is refused, the claim is point-only.
    """
    _refuse_structure(graph)
    if not isinstance(intervention, EdgeIntervention):
        raise CausalTypeError("intervention must be an EdgeIntervention")
    if mechanism not in _MECHANISMS:
        raise CausalValueError(f"mechanism must be one of {_MECHANISMS}")
    names, columns = as_columns(data)
    return _effect(
        _evaluate(
            names,
            columns,
            list(graph.edges()),
            intervention.treatment,
            intervention.outcome,
            intervention.control,
            intervention.active,
            list(intervention.edges),
            mechanism=mechanism,
            interval_requested=uncertainty is not None,
            seed=seed,
        )
    )


def consume_cross_world_artifact(artifact: bytes, *, seed: int = 1) -> CrossWorldEffect:
    """Replay an exported artifact from its stored premises and accept only an identical one.

    The replay uses the same evaluator (plus, for ``linear_gaussian``, a separate
    closed-form least-squares cross-check). It detects corruption and re-sealed
    edits that change the derivation or the point; it does not detect a producer
    that seals a wrong table or query on purpose.
    """
    if not isinstance(artifact, bytes | bytearray | memoryview):
        raise CausalTypeError("artifact must be bytes")
    return _effect(_consume(bytes(artifact), seed=seed))
