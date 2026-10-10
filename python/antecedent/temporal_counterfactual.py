"""Fixed-population temporal counterfactual and the closed transported route (2.3 X8).

The graph is one fixed, fully observed, Markovian two-slice DAG over the nodes
``covariate_0 -> action_0 -> covariate_1 -> action_1 -> outcome`` (a covariate and an action
per slice, then the final outcome) with no latent confounding. Each non-action node has an
additive-noise linear mechanism ``V = intercept + sum(coef * parent) + U_V``. Each unit's
exogenous history ``(U_covariate_0, U_covariate_1, U_outcome)`` is recovered exactly from its
factual trajectory (abduction), BOTH named two-step action histories are replayed against that
same history (action) and the unit's final outcome is computed from it (prediction). The claim
is a point: per-unit final outcomes in both worlds and their sample mean contrast. Fresh noise
per world would answer a different (interventional) question and is never drawn.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NoReturn

from ._native import (
    consume_temporal_counterfactual_artifact as _consume_temporal,
)
from ._native import (
    evaluate_temporal_counterfactual as _evaluate_temporal,
)
from ._native import (
    transported_path_specific_refusal as _transported_refusal,
)
from .errors import CausalTypeError, CausalUnsupportedError, CausalValueError
from .temporal import TemporalRefusal, _raise_refusal

__all__ = [
    "AbductionReceipt",
    "ActionHistory",
    "NodeMechanism",
    "TemporalCounterfactualEffect",
    "TemporalCounterfactualIdentity",
    "TemporalMechanisms",
    "TransportedPrerequisites",
    "UnitCounterfactual",
    "UnitHistory",
    "consume_temporal_counterfactual_artifact",
    "temporal_fixed_population",
    "transported_path_specific",
]

TEMPORAL_NODES = ("covariate_0", "action_0", "covariate_1", "action_1", "outcome")
_U32_MAX = 2**32 - 1


def _name(value: object, what: str) -> str:
    if not isinstance(value, str) or not value:
        raise CausalTypeError(f"{what} must be a non-empty string")
    return value


def _times(value: object, what: str) -> tuple[int, int]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, str)
        or len(value) != 2
        or not all(isinstance(t, int) and not isinstance(t, bool) for t in value)
    ):
        raise CausalTypeError(f"{what} must be a pair of integer observation times")
    if not all(0 <= t <= _U32_MAX for t in value):
        raise CausalValueError(f"{what} must lie in [0, 2**32)")
    return (int(value[0]), int(value[1]))


def _number(value: object, what: str, *, detail: str, offending: str | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        try:
            number = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError) as error:
            raise CausalTypeError(f"{what} must be a real number") from error
    else:
        number = float(value)
    if not math.isfinite(number):
        # JSON cannot carry a non-finite number: report it with the Rust refusal's code/detail.
        raise TemporalRefusal(
            {
                "code": "invalid_argument",
                "stage": "evaluate",
                "detail": detail,
                "offending": offending or what,
                "message": f"{what} is not finite",
            }
        )
    return number


@dataclass(frozen=True, slots=True)
class UnitHistory:
    """One unit's factual two-slice trajectory ``[L0, A0, L1, A1, Y]``.

    ``history`` is the unit's factual-history id (default ``"<unit>:history"``); ``times`` are
    the observation times of the two slices, which every action history must match.
    """

    unit: str
    covariate_0: float
    action_0: float
    covariate_1: float
    action_1: float
    outcome: float
    history: str | None = None
    times: tuple[int, int] = (0, 1)

    def __post_init__(self) -> None:
        _name(self.unit, "unit")
        if self.history is not None:
            _name(self.history, "history")
        object.__setattr__(self, "times", _times(self.times, "times"))
        for node in TEMPORAL_NODES:
            object.__setattr__(
                self,
                node,
                _number(
                    getattr(self, node),
                    node,
                    detail="temporal_counterfactual.non_finite_history",
                    offending=self.unit,
                ),
            )

    def _wire(self) -> dict[str, Any]:
        return {
            "unit": self.unit,
            "history": self.history if self.history is not None else f"{self.unit}:history",
            "times": list(self.times),
            "values": [getattr(self, node) for node in TEMPORAL_NODES],
        }


@dataclass(frozen=True, slots=True)
class ActionHistory:
    """A named two-step action history ``(A0, A1)`` and the units it is requested for.

    ``times`` must equal the factual observation times. ``units=None`` requests every unit
    of the factual histories; naming a subset (or a unit with no factual history) is refused
    as unpaired (``temporal_counterfactual.unpaired_histories``) with the offending unit as
    the witness.
    """

    name: str
    actions: tuple[float, float]
    times: tuple[int, int] = (0, 1)
    units: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        _name(self.name, "name")
        if not isinstance(self.actions, Sequence) or len(self.actions) != 2:
            raise CausalTypeError("actions must be a pair (action_0, action_1)")
        object.__setattr__(
            self,
            "actions",
            tuple(
                _number(
                    a,
                    "action",
                    detail="temporal_counterfactual.non_finite_history",
                    offending=self.name,
                )
                for a in self.actions
            ),
        )
        object.__setattr__(self, "times", _times(self.times, "times"))
        if self.units is not None:
            object.__setattr__(self, "units", tuple(_name(u, "unit") for u in self.units))

    def _wire(self, units: Sequence[str]) -> dict[str, Any]:
        return {
            "name": self.name,
            "times": list(self.times),
            "actions": list(self.actions),
            "units": list(self.units if self.units is not None else units),
        }


@dataclass(frozen=True, slots=True)
class NodeMechanism:
    """The additive-noise linear mechanism of one non-action node.

    ``node`` is ``"covariate_0"``, ``"covariate_1"`` or ``"outcome"``; ``coefficients`` maps
    each parent node name to its coefficient. ``noise_halfwidth`` optionally declares a
    bounded noise support: a factual history whose abduced residual falls outside it is a
    refuting witness (``temporal_counterfactual.refuting_history``).
    """

    node: str
    intercept: float
    coefficients: Mapping[str, float]
    noise_halfwidth: float | None = None

    def __post_init__(self) -> None:
        _name(self.node, "node")
        object.__setattr__(
            self,
            "intercept",
            _number(self.intercept, "intercept", detail="temporal_counterfactual.fit_mismatch"),
        )
        object.__setattr__(
            self,
            "coefficients",
            {
                _name(parent, "parent"): _number(
                    coef,
                    "coefficient",
                    detail="temporal_counterfactual.fit_mismatch",
                    offending=self.node,
                )
                for parent, coef in dict(self.coefficients).items()
            },
        )
        if self.noise_halfwidth is not None:
            object.__setattr__(
                self,
                "noise_halfwidth",
                _number(
                    self.noise_halfwidth,
                    "noise_halfwidth",
                    detail="temporal_counterfactual.fit_mismatch",
                    offending=self.node,
                ),
            )

    def _wire(self) -> dict[str, Any]:
        return {
            "node": self.node,
            "intercept": self.intercept,
            "parent_coefficients": [[p, c] for p, c in self.coefficients.items()],
            "noise_halfwidth": self.noise_halfwidth,
        }


@dataclass(frozen=True, slots=True)
class TemporalMechanisms:
    """A fitted (or supplied) mechanism per non-action node, under one stable ``fit_id``.

    The temporal graph is derived from the mechanisms' parents, so graph and fit agree by
    construction; ``fit_id`` is part of the artifact identity, so a refit under the same id is
    still a different fit (its coefficients are bound too).
    """

    fit_id: str
    mechanisms: tuple[NodeMechanism, ...]

    def __post_init__(self) -> None:
        _name(self.fit_id, "fit_id")
        object.__setattr__(self, "mechanisms", tuple(self.mechanisms))
        if not all(isinstance(m, NodeMechanism) for m in self.mechanisms):
            raise CausalTypeError("mechanisms must be NodeMechanism objects")

    def _edges(self) -> list[list[str]]:
        return sorted([parent, m.node] for m in self.mechanisms for parent in m.coefficients)

    def _wire(self) -> dict[str, Any]:
        return {"fit_id": self.fit_id, "mechanisms": [m._wire() for m in self.mechanisms]}


@dataclass(frozen=True, slots=True)
class UnitCounterfactual:
    """One unit's final outcome, factual and under both named action histories."""

    unit: str
    history: str
    factual_outcome: float
    plus_outcome: float
    minus_outcome: float

    @property
    def contrast(self) -> float:
        """``plus_outcome - minus_outcome`` for this unit's own abduced history."""
        return self.plus_outcome - self.minus_outcome


@dataclass(frozen=True, slots=True)
class AbductionReceipt:
    """Receipt that both worlds were replayed against one shared abduced history per unit.

    The digests bind the snapshot, graph, mechanism fit, factual histories and each world's
    action history; ``unit_draws`` holds one ``(unit, history, digest)`` of the abduced
    exogenous draw per unit. They are integrity digests (not cryptographic).
    """

    snapshot: str
    graph_digest: str
    fit_digest: str
    factual_digest: str
    plus_digest: str
    minus_digest: str
    unit_draws: tuple[tuple[str, str, str], ...]
    exogenous_digest: str
    n_units: int
    n_worlds: int
    horizon: int
    shared_by_both_worlds: bool
    digest: str


@dataclass(frozen=True, slots=True)
class TemporalCounterfactualIdentity:
    """Identity digests a consumer retains independently of the artifact bytes.

    Pass it to :func:`consume_temporal_counterfactual_artifact` as ``expected_identity=`` to refuse a
    *resealed* change of action time, unit history, snapshot, mechanism fit or graph.
    """

    snapshot: str
    graph_digest: str
    fit_digest: str
    factual_digest: str
    plus_digest: str
    minus_digest: str
    exogenous_digest: str
    receipt_digest: str
    spec_id: str

    def _wire(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in self.__slots__}


@dataclass(frozen=True, slots=True)
class TemporalCounterfactualEffect:
    """Per-unit counterfactual outcomes, their mean contrast, the receipt and the artifact.

    ``contrast`` is the sample mean of ``plus - minus`` over units; each unit's contrast
    uses that unit's own abduced history in both worlds. The claim is a point (no interval,
    no posterior, no transport): ``inference_claim == "point_only"``.
    """

    units: tuple[UnitCounterfactual, ...]
    mean_plus: float
    mean_minus: float
    contrast: float
    plus_name: str
    minus_name: str
    mechanism_class: str
    inference_claim: str
    snapshot: str
    receipt: AbductionReceipt
    identity: TemporalCounterfactualIdentity
    artifact: bytes

    def export(self) -> bytes:
        """The checksummed ``temporal_counterfactual_v1`` artifact."""
        return self.artifact


def _temporal_effect(report_json: str, artifact: bytes) -> TemporalCounterfactualEffect:
    report = json.loads(report_json)
    receipt = report["receipt"]
    return TemporalCounterfactualEffect(
        units=tuple(
            UnitCounterfactual(
                unit=u["unit"],
                history=u["history"],
                factual_outcome=float(u["factual_outcome"]),
                plus_outcome=float(u["plus_outcome"]),
                minus_outcome=float(u["minus_outcome"]),
            )
            for u in report["units"]
        ),
        mean_plus=float(report["mean_plus"]),
        mean_minus=float(report["mean_minus"]),
        contrast=float(report["mean_contrast"]),
        plus_name=report["plus_name"],
        minus_name=report["minus_name"],
        mechanism_class=report["mechanism_class"],
        inference_claim=report["inference_claim"],
        snapshot=report["snapshot"],
        receipt=AbductionReceipt(
            snapshot=receipt["snapshot"],
            graph_digest=receipt["graph_digest"],
            fit_digest=receipt["fit_digest"],
            factual_digest=receipt["factual_digest"],
            plus_digest=receipt["plus_digest"],
            minus_digest=receipt["minus_digest"],
            unit_draws=tuple((d["unit"], d["history"], d["digest"]) for d in receipt["unit_draws"]),
            exogenous_digest=receipt["exogenous_digest"],
            n_units=int(receipt["n_units"]),
            n_worlds=int(receipt["n_worlds"]),
            horizon=int(receipt["horizon"]),
            shared_by_both_worlds=bool(receipt["shared_by_both_worlds"]),
            digest=receipt["digest"],
        ),
        identity=TemporalCounterfactualIdentity(**report["identity"]),
        artifact=artifact,
    )


def temporal_fixed_population(
    histories: Sequence[UnitHistory],
    *,
    plus: ActionHistory,
    minus: ActionHistory,
    mechanisms: TemporalMechanisms,
    snapshot: str,
    horizon: int = 2,
    latent_confounding: bool = False,
) -> TemporalCounterfactualEffect:
    """What would each unit's final outcome have been under two named action histories?

    ``histories`` are the units' factual two-slice trajectories; ``plus`` and ``minus`` are the
    two named action histories (the contrast is ``plus - minus``) and ``mechanisms`` the fitted
    linear-Gaussian additive-noise mechanisms of ``covariate_0``, ``covariate_1`` and
    ``outcome``. ``snapshot`` identifies the data the histories were read from. Abduction,
    action and prediction happen in one operation: each unit's exogenous history is abduced
    once and both worlds replay against it.

    Raises :class:`~antecedent.temporal.TemporalRefusal` (a
    :class:`~antecedent.errors.CausalUnsupportedError`) with ``route_not_supported`` and
    ``temporal_counterfactual.unpaired_histories`` (``witness`` names the unit and world),
    ``.time_misaligned``, ``.shared_history_missing``, ``.latent_confounding`` or
    ``.refuting_history`` (``witness`` retains unit, history, node, residual and bound), and
    with ``invalid_argument`` for ``.horizon_exceeded``, ``.too_many_units``,
    ``.invalid_graph``, ``.fit_mismatch``, ``.history_name_invalid`` or
    ``.non_finite_history``.
    """
    if isinstance(histories, str | bytes) or not isinstance(histories, Sequence):
        raise CausalTypeError("histories must be a sequence of UnitHistory")
    if not all(isinstance(h, UnitHistory) for h in histories):
        raise CausalTypeError("histories must be UnitHistory objects")
    if not isinstance(plus, ActionHistory) or not isinstance(minus, ActionHistory):
        raise CausalTypeError("plus and minus must be ActionHistory objects")
    if not isinstance(mechanisms, TemporalMechanisms):
        raise CausalTypeError("mechanisms must be TemporalMechanisms")
    _name(snapshot, "snapshot")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or not 0 <= horizon <= _U32_MAX:
        raise CausalValueError("horizon must be a non-negative integer")
    units = [h.unit for h in histories]
    request = {
        "graph": {
            "horizon": horizon,
            "edges": mechanisms._edges(),
            "latent_confounding": bool(latent_confounding),
        },
        "fit": mechanisms._wire(),
        "snapshot": snapshot,
        "factual": [h._wire() for h in histories],
        "plus": plus._wire(units),
        "minus": minus._wire(units),
    }
    report, artifact, refusal = _evaluate_temporal(
        json.dumps(request, allow_nan=False), "temporal_counterfactual"
    )
    _raise_refusal(refusal)
    if report is None or artifact is None:  # pragma: no cover - the native contract
        raise CausalValueError(
            "the native temporal counterfactual returned neither a result nor a refusal"
        )
    return _temporal_effect(report, bytes(artifact))


def consume_temporal_counterfactual_artifact(
    artifact: bytes,
    *,
    expected_identity: TemporalCounterfactualIdentity | Mapping[str, str] | None = None,
) -> TemporalCounterfactualEffect:
    """Replay both worlds of an exported artifact and accept only an identical one.

    Every unit's exogenous history is abduced again from the stored factual history and both
    action histories are replayed with the same evaluator; every stored outcome, both means
    and the whole shared-abduction receipt must reproduce bit for bit. With ``expected_identity`` (the
    :attr:`TemporalCounterfactualEffect.identity` retained out-of-band) a changed action time,
    unit history, snapshot, mechanism fit or graph is refused even when the artifact was
    resealed consistently (:class:`~antecedent.temporal.TemporalRefusal`,
    ``route_not_supported``, ``temporal_counterfactual.artifact_changed``). Corruption and
    unknown major versions raise :class:`~antecedent.errors.CausalSerializationError`.
    """
    if not isinstance(artifact, bytes | bytearray | memoryview):
        raise CausalTypeError("artifact must be bytes")
    if expected_identity is None:
        expected_json = None
    elif isinstance(expected_identity, TemporalCounterfactualIdentity):
        expected_json = json.dumps(expected_identity._wire())
    elif isinstance(expected_identity, Mapping):
        expected_json = json.dumps(dict(expected_identity))
    else:
        raise CausalTypeError(
            "expected_identity must be a TemporalCounterfactualIdentity or a mapping"
        )
    data = bytes(artifact)
    report, refusal = _consume_temporal(data, expected_json)
    _raise_refusal(refusal)
    if report is None:  # pragma: no cover - the native contract
        raise CausalValueError("the native consumer returned neither a result nor a refusal")
    return _temporal_effect(report, data)


@dataclass(frozen=True, slots=True)
class TransportedPrerequisites:
    """Which prerequisites of a transported counterfactual have passed.

    All default to ``False``: no composed license exists. Declaring one ``True`` records that
    the gate passed; it never opens the route, because the joint theorem has not passed.
    """

    transport_license: bool = False
    fixed_population_license: bool = False
    cross_population_assumptions: bool = False


def _factor(item: str | tuple[str, str]) -> tuple[str, str]:
    if isinstance(item, str):
        role, _, regime = item.partition(":")
    elif isinstance(item, tuple) and len(item) == 2:
        role, regime = item
    else:
        raise CausalTypeError("a regime factor is 'source:<regime>' / 'target:<regime>' or a pair")
    if role not in ("source", "target") or not isinstance(regime, str) or not regime:
        raise CausalValueError(
            "transported_counterfactual.invalid_factor: a factor names its population "
            "('source' or 'target') and a regime",
            reason_code="invalid_argument",
        )
    return (role, regime)


def transported_path_specific(
    *,
    required_factors: Iterable[str | tuple[str, str]] = (),
    supplied_factors: Mapping[str | tuple[str, str], str] | None = None,
    prerequisites: TransportedPrerequisites | None = None,
) -> NoReturn:
    """The closed transported path-specific counterfactual route: always a typed refusal.

    Transporting a counterfactual needs both an identified target transport functional and a
    fixed-population cross-world theorem for the same quantity; neither a transported mean
    nor a source counterfactual alone licenses their composition, and the joint theorem has
    not passed. Nothing is evaluated whatever is supplied.

    The refusal is :class:`~antecedent.temporal.TemporalRefusal` with
    ``reason_code="cell_not_licensed"`` and detail ``transported_counterfactual.route_frozen``
    (``offending`` is the first missing gate) while a prerequisite gate is missing, and also
    when every gate passes and every regime factor is present. When every gate passes but a
    required factor (``"source:<regime>"`` / ``"target:<regime>"``) is absent from
    ``supplied_factors`` (factor -> evidence id) it is ``reason_code="transport_missing_evidence"``
    with ``transported_counterfactual.factor_missing``. ``missing_gates`` and
    ``missing_factors`` are retained either way.
    """
    declared = prerequisites if prerequisites is not None else TransportedPrerequisites()
    if not isinstance(declared, TransportedPrerequisites):
        raise CausalTypeError("prerequisites must be a TransportedPrerequisites")
    required = [_factor(f) for f in required_factors]
    supplied = [(*_factor(k), str(v)) for k, v in (supplied_factors or {}).items()]
    _raise_refusal(
        _transported_refusal(
            declared.transport_license,
            declared.fixed_population_license,
            declared.cross_population_assumptions,
            required,
            supplied,
        )
    )
    raise CausalUnsupportedError(  # pragma: no cover - the native route always refuses
        "transported_counterfactual.route_frozen: the transported route is closed",
        reason_code="cell_not_licensed",
    )
