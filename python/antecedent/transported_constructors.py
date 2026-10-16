"""Typed constructors of a transported response from the library transport routes.

A transported response (:class:`~antecedent.transported_response.TransportedResponse`) is what a
decision (:func:`~antecedent.transported_response.evaluate_transported`), an inverse query and
the assumption-range report read. Each constructor here takes the typed source of one route,
runs the route in Rust and builds the response from the route's own result, so the populations,
transport identities and assumptions are copied from the producing result and never typed::

    from antecedent import sequential_evaluation as se
    from antecedent import transported_constructors as tc

    construction = tc.from_policy(
        model, policy,
        target_population="plant_b",
        target_initial=(0.2, 0.8),
        diagram=tc.SelectionDiagram(
            "plant_a", "plant_b",
            {"initial_state": "shifted", "utility": "invariant",
             "transition[0]": "invariant", "transition[1]": "invariant",
             "transition[2]": "invariant", "observation[0]": "invariant"},
        ),
        coordinate=value_in_plant_b,
        provider_id="plant-transport",
    )
    construction.response        # a TransportedResponse: evaluate_transported(...) reads it
    construction.evaluation      # the route's own values, identity and premises

* :func:`from_policy`: a policy's value in another population (selection-diagram digest), with
  the target's own utility (:class:`TargetUtility`) when the diagram marks ``utility`` shifted.
* :func:`from_period_shift`: a policy's value in another period (time-invariance digest).
* :func:`from_adaptive_trial`: an adaptive trial's arm mean or contrast (trial-diagram digest).
* :func:`from_synthesis`: a transported synthesis over an evidence ledger (ledger transport
  digests).
* :func:`from_surrogate_bridge`: a surrogate bridge's terminal value (bridge-diagram digest).

What a constructed response is, and is not:

* **A point, labelled ``extrapolative``.** Never native target data; no interval, law or
  calibration claim. The route's own standard error (a synthesis's model variance) is an
  uncalibrated diagnostic that no decision inherits.
* **Stated in the target.** ``coordinate`` must be a quantity of the transport's target
  population; one stated for the source population is refused. A period shift's populations
  are period-qualified (:func:`period_population_label`).
* **Three routes carry the route ``declared``** in the receipt (the response route set has no
  dedicated name for them); the route is named in the assumptions
  (``transport_route:<name>``) and the identities are the typed digests.
* **Refresh.** New data changes the model, contract or ledger, which is a new evaluation and a
  new response identity; the transport declaration is kept only by passing it again. A response
  is never refreshed in place. :func:`from_synthesis` evaluates against a copy of the session's
  ledger: the session is unchanged.

Rust owns the route, the checks, the identities and every refusal; this module declares the
inputs and presents the result. A refusal raises
:class:`~antecedent.transported_response.TransportedResponseRefusal`, a
:class:`~antecedent.errors.StructuredRefusal` with a registered ``reason_code``, a namespaced
``detail`` and a ``remedy``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

from . import _native
from ._refusal import raise_refusal
from ._validate import real, text
from .errors import CausalTypeError, CausalValueError
from .joint_distribution import ScientificQuantity
from .transported_response import TransportedResponse, TransportedResponseRefusal

__all__ = [
    "ActionUtility",
    "AdaptiveTrialDiagram",
    "Borrow",
    "BridgeDiagram",
    "Contribution",
    "InvarianceDeclaration",
    "Pool",
    "SelectionDiagram",
    "SurrogateBridge",
    "TargetPeriod",
    "TargetUtility",
    "TransportedConstruction",
    "TransportedPrior",
    "from_adaptive_trial",
    "from_period_shift",
    "from_policy",
    "from_surrogate_bridge",
    "from_synthesis",
    "period_population_label",
    "verify_adaptive_trial",
    "verify_period_shift",
    "verify_policy",
]

_MECHANISM = re.compile(r"^(initial_state|utility|(transition|observation)\[(\d+)\])$")
_STATUSES = ("invariant", "shifted")
_TRIAL_MECHANISMS = ("stratum_law", "outcome_given_stratum_and_arm")
_ADAPTIVE_VALUES = ("treatment_mean", "control_mean", "contrast")
AdaptiveValue = Literal["treatment_mean", "control_mean", "contrast"]


def period_population_label(population: str, period: str) -> str:
    """The population label of a period of a population: ``population@period``.

    A quantity in the target period of a period shift is stated in this population; the
    model's own period is the source population.
    """
    return (
        f"{text(population, 'population', non_empty=True)}@{text(period, 'period', non_empty=True)}"
    )


@dataclass(frozen=True, slots=True)
class TransportedConstruction:
    """A transported response with the producing route's own result.

    ``evaluation`` holds the route's values, identity and premises (a read-only mapping);
    ``response`` is the object a decision or an inverse query reads.
    """

    response: TransportedResponse
    evaluation: Mapping[str, Any] = field(default_factory=dict)

    def inspect(self) -> Any:
        """The response's inspection report."""
        return self.response.inspect()


def _unwrap(triple: tuple[Any, str | None, str | None]) -> TransportedConstruction:
    handle, evaluation, refusal = triple
    raise_refusal(refusal, TransportedResponseRefusal)
    assert evaluation is not None
    return TransportedConstruction(
        response=TransportedResponse._of(handle),
        evaluation=MappingProxyType(json.loads(evaluation)),
    )


def _dumps(value: object) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"))


def _quantity(value: object) -> str:
    if not isinstance(value, ScientificQuantity):
        raise CausalTypeError("coordinate must be a ScientificQuantity")
    return _dumps(value._wire())


def _provider(value: object) -> str:
    return text(value, "provider_id", non_empty=True)


def _mechanisms(mechanisms: Mapping[str, str]) -> list[dict[str, Any]]:
    if not isinstance(mechanisms, Mapping):
        raise CausalTypeError("mechanisms must map a mechanism name to its status")
    rows: list[dict[str, Any]] = []
    for name, status in mechanisms.items():
        match = _MECHANISM.match(str(name))
        if match is None:
            raise CausalValueError(
                f"mechanism {name!r} must be `initial_state`, `utility`, `transition[<action>]` "
                "or `observation[<action>]`"
            )
        if status not in _STATUSES:
            raise CausalValueError(f"mechanism status must be one of {', '.join(_STATUSES)}")
        kind = match.group(2) or match.group(1)
        action = int(match.group(3)) if match.group(3) is not None else None
        rows.append({"mechanism": kind, "action": action, "status": status})
    return rows


def _initial(values: Sequence[float] | None, what: str) -> list[float] | None:
    if values is None:
        return None
    if isinstance(values, str) or not isinstance(values, Sequence):
        raise CausalTypeError(f"{what} must be a sequence of probabilities")
    return [real(v, what, finite=True) for v in values]


def _model_and_policy(model: Any, policy: Any) -> tuple[str, str]:
    from .sequential_evaluation import FiniteModel, Policy

    if not isinstance(model, FiniteModel):
        raise CausalTypeError("model must be a sequential_evaluation.FiniteModel")
    if not isinstance(policy, Policy):
        raise CausalTypeError("policy must be a sequential_evaluation.Policy")
    return _dumps(model.to_wire()), _dumps(policy.to_wire())


# --------------------------------------------------------------------------- policy


@dataclass(frozen=True, slots=True)
class SelectionDiagram:
    """The declared status of every mechanism of a model between two populations.

    ``mechanisms`` maps ``initial_state``, ``utility``, ``transition[<action>]`` and
    ``observation[<action>]`` (one per action with an observation channel) to ``invariant`` or
    ``shifted``. Only the initial-state law and, with a :class:`TargetUtility`, the utility may
    be shifted; an undeclared mechanism is refused, never assumed invariant.
    """

    source_population: str
    target_population: str
    mechanisms: Mapping[str, str]

    def _wire(self) -> dict[str, Any]:
        return {
            "source_population": self.source_population,
            "target_population": self.target_population,
            "mechanisms": _mechanisms(self.mechanisms),
        }


@dataclass(frozen=True, slots=True)
class ActionUtility:
    """The target's utility of one action: its cost and its value per latent state.

    ``delayed`` is the value per state of the action's delayed outcome and is given exactly when
    the model's action has one (its timing stays the model's).
    """

    cost: float
    immediate: Sequence[float]
    delayed: Sequence[float] | None = None

    def _wire(self) -> dict[str, Any]:
        return {
            "cost": real(self.cost, "cost", finite=True),
            "immediate": _initial(self.immediate, "immediate"),
            "delayed": _initial(self.delayed, "delayed"),
        }


@dataclass(frozen=True, slots=True)
class TargetUtility:
    """The target population's own utility mechanism, with where it comes from.

    A utility is a valuation, not a causal mechanism, so the source model cannot supply the
    target's. ``terminal`` is the value per state when the horizon is reached without a commit
    and ``actions`` one :class:`ActionUtility` per model action, in the model's order, over the
    model's own states. ``source`` labels the target-side source of the valuation and ``digest``
    is its 64-character lowercase hexadecimal content digest; both are bound into the
    evaluation's identity, so a changed value or provenance is a different, non-replaying
    evaluation. Pass it with a diagram that marks ``utility`` shifted (and only then).
    """

    source: str
    digest: str
    terminal: Sequence[float]
    actions: Sequence[ActionUtility]

    def _wire(self) -> dict[str, Any]:
        if isinstance(self.actions, str) or not isinstance(self.actions, Sequence):
            raise CausalTypeError("actions must be a sequence of ActionUtility")
        if not all(isinstance(action, ActionUtility) for action in self.actions):
            raise CausalTypeError("actions must be a sequence of ActionUtility")
        return {
            "source": text(self.source, "source"),
            "digest": text(self.digest, "digest"),
            "terminal": _initial(self.terminal, "terminal"),
            "actions": [action._wire() for action in self.actions],
        }


def _utility_json(utility: TargetUtility | None) -> str | None:
    if utility is None:
        return None
    if not isinstance(utility, TargetUtility):
        raise CausalTypeError("target_utility must be a TargetUtility")
    return _dumps(utility._wire())


def from_policy(
    model: Any,
    policy: Any,
    *,
    target_population: str,
    target_initial: Sequence[float] | None,
    diagram: SelectionDiagram | None,
    coordinate: ScientificQuantity,
    provider_id: str,
    max_nodes: int | None = None,
    target_utility: TargetUtility | None = None,
    cancel: Any = None,
) -> TransportedConstruction:
    """A fixed policy's value in another population, as a transported response.

    The policy is evaluated in the model's own population and in ``target_population`` (the
    model with the target's initial-state law and, when ``target_utility`` is given, the
    target's own utility) by exact enumeration; the response states the target value at
    ``coordinate``. The source population is the model's; the identity is the selection-diagram
    digest, and a target utility's provenance and content are bound into the evaluation's
    identity and named in the response's assumptions.

    Raises:
        TransportedResponseRefusal: a missing diagram or target initial law, a shifted
            transition or observation mechanism, a utility marked shifted with no
            ``target_utility`` (or one supplied while it is marked invariant), a target utility
            without provenance or over other actions or states than the model's, a target outside
            the source's support, a target equal to the source, a coordinate not in the target
            population, and every sequential-evaluation refusal of the policy.
    """
    model_json, policy_json = _model_and_policy(model, policy)
    if diagram is not None and not isinstance(diagram, SelectionDiagram):
        raise CausalTypeError("diagram must be a SelectionDiagram")
    return _unwrap(
        _native.transported_policy_construct(
            model_json,
            policy_json,
            text(target_population, "target_population"),
            _initial(target_initial, "target_initial"),
            None if diagram is None else _dumps(diagram._wire()),
            _quantity(coordinate),
            _provider(provider_id),
            max_nodes,
            _utility_json(target_utility),
            cancel=cancel,
        )
    )


# --------------------------------------------------------------------------- period shift


@dataclass(frozen=True, slots=True)
class TargetPeriod:
    """What is known of the target period: its label, horizon and history alphabet.

    ``alphabet`` is one ``(action label, observation symbol labels)`` pair per action, in action
    order. :meth:`of_model` takes the model's own alphabet, which is the premise that the target
    period has the same actions and observation symbols with the same meanings.
    """

    period: str
    horizon: int
    alphabet: Sequence[tuple[str, Sequence[str]]]

    @classmethod
    def of_model(cls, model: Any, period: str) -> TargetPeriod:
        """The target period with the model's own horizon and alphabet."""
        from .sequential_evaluation import FiniteModel

        if not isinstance(model, FiniteModel):
            raise CausalTypeError("model must be a sequential_evaluation.FiniteModel")
        alphabet, refusal = _native.transported_history_alphabet(_dumps(model.to_wire()))
        raise_refusal(refusal, TransportedResponseRefusal)
        assert alphabet is not None
        rows = json.loads(alphabet)
        return cls(
            period=period,
            horizon=model.horizon,
            alphabet=tuple((row["label"], tuple(row["symbols"])) for row in rows),
        )

    def _wire(self) -> dict[str, Any]:
        return {
            "period": self.period,
            "horizon": self.horizon,
            "alphabet": [
                {"label": label, "symbols": list(symbols)} for label, symbols in self.alphabet
            ],
        }


@dataclass(frozen=True, slots=True)
class InvarianceDeclaration:
    """The declared time invariance of every mechanism over a fixed horizon.

    ``mechanisms`` is as in :class:`SelectionDiagram`; only the initial-state law may vary
    between the periods.
    """

    source_period: str
    target_period: str
    horizon: int
    mechanisms: Mapping[str, str]

    def _wire(self) -> dict[str, Any]:
        return {
            "source_period": self.source_period,
            "target_period": self.target_period,
            "horizon": self.horizon,
            "mechanisms": _mechanisms(self.mechanisms),
        }


def from_period_shift(
    model: Any,
    policy: Any,
    *,
    target: TargetPeriod,
    target_initial: Sequence[float] | None,
    invariance: InvarianceDeclaration | None,
    coordinate: ScientificQuantity,
    provider_id: str,
    max_nodes: int | None = None,
    cancel: Any = None,
) -> TransportedConstruction:
    """A fixed policy's value in another period, as a transported response.

    ``coordinate`` is stated in the period-qualified population of the target
    (``period_population_label(model.population, target.period)``); the model's own period is
    the source. The identity is the digest of ``invariance``; the fixed horizon and every
    mechanism's declared status are assumptions of the receipt.

    Raises:
        TransportedResponseRefusal: a missing invariance declaration or target law, a declaration
            for other periods, a horizon or alphabet that differs, a time-varying transition,
            observation or utility, a target period equal to the source's, and a coordinate not
            in the target period.
    """
    model_json, policy_json = _model_and_policy(model, policy)
    if not isinstance(target, TargetPeriod):
        raise CausalTypeError("target must be a TargetPeriod")
    if invariance is not None and not isinstance(invariance, InvarianceDeclaration):
        raise CausalTypeError("invariance must be an InvarianceDeclaration")
    return _unwrap(
        _native.temporal_policy_construct(
            model_json,
            policy_json,
            _dumps(target._wire()),
            _initial(target_initial, "target_initial"),
            None if invariance is None else _dumps(invariance._wire()),
            _quantity(coordinate),
            _provider(provider_id),
            max_nodes,
            cancel=cancel,
        )
    )


# --------------------------------------------------------------------------- adaptive trial


@dataclass(frozen=True, slots=True)
class AdaptiveTrialDiagram:
    """The declared status of the trial's stratum law and outcome mechanism.

    ``mechanisms`` maps ``stratum_law`` and ``outcome_given_stratum_and_arm`` to ``invariant``
    or ``shifted``. The outcome mechanism must be invariant: the trial cannot supply it for the
    target.
    """

    source_population: str
    target_population: str
    mechanisms: Mapping[str, str]

    def _wire(self) -> dict[str, Any]:
        rows = []
        for name, status in self.mechanisms.items():
            if name not in _TRIAL_MECHANISMS:
                raise CausalValueError(
                    f"trial mechanism must be one of {', '.join(_TRIAL_MECHANISMS)}"
                )
            if status not in _STATUSES:
                raise CausalValueError(f"mechanism status must be one of {', '.join(_STATUSES)}")
            rows.append({"mechanism": name, "status": status})
        return {
            "source_population": self.source_population,
            "target_population": self.target_population,
            "mechanisms": rows,
        }


def from_adaptive_trial(
    design: Any,
    *,
    trial_population: str,
    unit_strata: Sequence[int],
    stratum_labels: Sequence[str],
    target_stratum_law: Sequence[float] | None,
    diagram: AdaptiveTrialDiagram | None,
    coordinate: ScientificQuantity,
    provider_id: str,
    value: AdaptiveValue = "contrast",
    treatment_arm: int = 1,
    control_arm: int = 0,
) -> TransportedConstruction:
    """An adaptive trial's transported arm mean or contrast, as a transported response.

    ``design`` is the trial's retained analysis contract
    (:class:`~antecedent.adaptive_inference.AdaptiveDesign`); the arm means are inverse-
    probability weighted by the realized assignment probabilities within each stratum and
    standardized to ``target_stratum_law``. ``value`` picks the treatment mean, the control mean
    or their contrast. The source population is ``trial_population``; the identity is the
    diagram digest.

    Raises:
        TransportedResponseRefusal: a missing contract, diagram or target law, an incomplete
            contract (``adaptive_contract_incomplete``), a fixed design, a shifted outcome
            mechanism, an assignment below the probability floor, a target stratum unobserved
            under a compared arm, a diagram for other populations and a coordinate not in the
            target population.
    """
    from .adaptive_inference import AdaptiveDesign

    if design is not None and not isinstance(design, AdaptiveDesign):
        raise CausalTypeError("design must be an AdaptiveDesign or None")
    if diagram is not None and not isinstance(diagram, AdaptiveTrialDiagram):
        raise CausalTypeError("diagram must be an AdaptiveTrialDiagram")
    if value not in _ADAPTIVE_VALUES:
        raise CausalValueError(f"value must be one of {', '.join(_ADAPTIVE_VALUES)}")
    strata = [int(s) for s in unit_strata]
    return _unwrap(
        _native.adaptive_trial_construct(
            None if design is None else _dumps(design.to_wire()),
            text(trial_population, "trial_population"),
            strata,
            [text(label, "stratum label") for label in stratum_labels],
            _initial(target_stratum_law, "target_stratum_law"),
            None if diagram is None else _dumps(diagram._wire()),
            int(treatment_arm),
            int(control_arm),
            value,
            _quantity(coordinate),
            _provider(provider_id),
        )
    )


# --------------------------------------------------------------------------- verification


def _stored(evaluation: TransportedConstruction | Mapping[str, Any], route: str) -> str:
    """The stored evaluation of ``route`` as JSON, refusing one of another route."""
    stored = (
        evaluation.evaluation if isinstance(evaluation, TransportedConstruction) else evaluation
    )
    if not isinstance(stored, Mapping):
        raise CausalTypeError("evaluation must be a TransportedConstruction or its mapping")
    if stored.get("route") != route:
        raise CausalValueError(
            f"the stored evaluation is of route {stored.get('route')!r}, not {route!r}"
        )
    return json.dumps(dict(stored), allow_nan=False)


def verify_policy(
    model: Any,
    policy: Any,
    evaluation: TransportedConstruction | Mapping[str, Any],
    *,
    target_population: str,
    target_initial: Sequence[float] | None,
    diagram: SelectionDiagram | None,
    max_nodes: int | None = None,
    target_utility: TargetUtility | None = None,
    cancel: Any = None,
) -> None:
    """Re-derive a transported policy evaluation and compare it bit for bit.

    The model, policy, diagram, target law and target utility are evaluated again; the stored
    identity and both values must equal the re-derived ones, so an edited utility value or
    provenance does not replay. Returns ``None`` when they replay.

    Raises:
        TransportedResponseRefusal: every refusal of :func:`from_policy`, and
            ``transported_policy.value_mismatch`` when a stored value or identity does not replay.
    """
    model_json, policy_json = _model_and_policy(model, policy)
    if diagram is not None and not isinstance(diagram, SelectionDiagram):
        raise CausalTypeError("diagram must be a SelectionDiagram")
    refusal = _native.transported_policy_verify(
        model_json,
        policy_json,
        text(target_population, "target_population"),
        _initial(target_initial, "target_initial"),
        None if diagram is None else _dumps(diagram._wire()),
        _stored(evaluation, "policy"),
        max_nodes,
        _utility_json(target_utility),
        cancel=cancel,
    )
    raise_refusal(refusal, TransportedResponseRefusal)


def verify_period_shift(
    model: Any,
    policy: Any,
    evaluation: TransportedConstruction | Mapping[str, Any],
    *,
    target: TargetPeriod,
    target_initial: Sequence[float] | None,
    invariance: InvarianceDeclaration | None,
    max_nodes: int | None = None,
    cancel: Any = None,
) -> None:
    """Re-derive a period-shift evaluation and compare it bit for bit.

    Raises:
        TransportedResponseRefusal: every refusal of :func:`from_period_shift`, and
            ``temporal_policy_transport.value_mismatch`` when a stored value or identity does
            not replay.
    """
    model_json, policy_json = _model_and_policy(model, policy)
    if not isinstance(target, TargetPeriod):
        raise CausalTypeError("target must be a TargetPeriod")
    if invariance is not None and not isinstance(invariance, InvarianceDeclaration):
        raise CausalTypeError("invariance must be an InvarianceDeclaration")
    refusal = _native.temporal_policy_verify(
        model_json,
        policy_json,
        _dumps(target._wire()),
        _initial(target_initial, "target_initial"),
        None if invariance is None else _dumps(invariance._wire()),
        _stored(evaluation, "temporal_policy"),
        max_nodes,
        cancel=cancel,
    )
    raise_refusal(refusal, TransportedResponseRefusal)


def verify_adaptive_trial(
    design: Any,
    evaluation: TransportedConstruction | Mapping[str, Any],
    *,
    trial_population: str,
    unit_strata: Sequence[int],
    stratum_labels: Sequence[str],
    target_stratum_law: Sequence[float] | None,
    diagram: AdaptiveTrialDiagram | None,
    treatment_arm: int = 1,
    control_arm: int = 0,
) -> None:
    """Re-derive an adaptive-trial transported estimate and compare it bit for bit.

    The stored identity and the treatment mean, control mean and contrast must equal the
    re-derived ones.

    Raises:
        TransportedResponseRefusal: every refusal of :func:`from_adaptive_trial`, and
            ``adaptive_transport.value_mismatch`` when a stored value or identity does not replay.
    """
    from .adaptive_inference import AdaptiveDesign

    if design is not None and not isinstance(design, AdaptiveDesign):
        raise CausalTypeError("design must be an AdaptiveDesign or None")
    if diagram is not None and not isinstance(diagram, AdaptiveTrialDiagram):
        raise CausalTypeError("diagram must be an AdaptiveTrialDiagram")
    refusal = _native.adaptive_trial_verify(
        None if design is None else _dumps(design.to_wire()),
        text(trial_population, "trial_population"),
        [int(s) for s in unit_strata],
        [text(label, "stratum label") for label in stratum_labels],
        _initial(target_stratum_law, "target_stratum_law"),
        None if diagram is None else _dumps(diagram._wire()),
        int(treatment_arm),
        int(control_arm),
        _stored(evaluation, "adaptive_trial"),
    )
    raise_refusal(refusal, TransportedResponseRefusal)


# --------------------------------------------------------------------------- synthesis


@dataclass(frozen=True, slots=True)
class Contribution:
    """A claim from the evidence ledger and the number it contributes.

    ``source`` is the ledger source identity. ``transport_variance`` is the variance of the
    transport error: required exactly when the source carries a transport identity, refused on a
    native source.
    """

    source: str
    estimate: float
    sampling_variance: float
    transport_variance: float | None = None

    def _wire(self) -> dict[str, Any]:
        return {
            "source": text(self.source, "source", non_empty=True),
            "estimate": real(self.estimate, "estimate", finite=True),
            "sampling_variance": real(self.sampling_variance, "sampling_variance", finite=True),
            "transport_variance": (
                None
                if self.transport_variance is None
                else real(self.transport_variance, "transport_variance", finite=True)
            ),
        }


@dataclass(frozen=True, slots=True)
class Pool:
    """Generalized-least-squares pooling; ``sampling_covariance`` is row-major ``k x k`` over the
    contributions sorted by source identity, independent when absent."""

    sampling_covariance: Sequence[float] | None = None

    def _wire(self) -> dict[str, Any]:
        covariance = self.sampling_covariance
        return {
            "kind": "pool",
            "sampling_covariance": (
                None
                if covariance is None
                else [real(v, "covariance", finite=True) for v in covariance]
            ),
        }


@dataclass(frozen=True, slots=True)
class Borrow:
    """Hierarchical normal borrowing across exchangeable studies."""

    mean_prior: Any
    tau_prior: Any

    def _wire(self) -> dict[str, Any]:
        return {
            "kind": "borrow",
            "mean_prior": self.mean_prior.to_wire(),
            "tau_prior": self.tau_prior.to_wire(),
        }


@dataclass(frozen=True, slots=True)
class TransportedPrior:
    """One transported claim (a ledger source of kind prior) is the prior of the native
    contributions."""

    prior: str

    def _wire(self) -> dict[str, Any]:
        return {"kind": "transported_prior", "prior": text(self.prior, "prior", non_empty=True)}


def from_synthesis(
    ledger: Any,
    label: str,
    contributions: Sequence[Contribution],
    method: Pool | Borrow | TransportedPrior,
    *,
    coordinate: ScientificQuantity,
    provider_id: str,
    native_population: str | None = None,
) -> TransportedConstruction:
    """A transported synthesis over an evidence ledger, as a transported response.

    The combination is registered in a copy of the session's ledger; ``ledger`` itself is not
    changed. The source populations and the target are the ones the ledger's transport identities
    recorded, never caller text. With ``native_population`` the native contributions' population
    is checked against the transports' target. The synthesis's model variance is retained as an
    uncalibrated standard error and is not inherited by a decision.

    Raises:
        TransportedResponseRefusal: double counting, unknown overlap, a transported claim without
            its transport variance (or a native one with it), contributions carried to different
            populations, a native population that is not the target, a synthesis of native
            sources only (``not_transported``) and a coordinate not in the target population.
    """
    from .evidence_ledger import EvidenceLedger

    if not isinstance(ledger, EvidenceLedger):
        raise CausalTypeError("ledger must be an evidence_ledger.EvidenceLedger")
    items = list(contributions)
    if not all(isinstance(c, Contribution) for c in items):
        raise CausalTypeError("contributions must be Contribution declarations")
    if not isinstance(method, (Pool, Borrow, TransportedPrior)):
        raise CausalTypeError("method must be Pool, Borrow or TransportedPrior")
    return _unwrap(
        _native.transported_synthesis_construct(
            ledger._handle,
            text(label, "label", non_empty=True),
            _dumps([c._wire() for c in items]),
            _dumps(method._wire()),
            native_population,
            _quantity(coordinate),
            _provider(provider_id),
        )
    )


# --------------------------------------------------------------------------- surrogate bridge


@dataclass(frozen=True, slots=True)
class BridgeDiagram:
    """The declared selection diagram between the population the surrogate was validated in and
    the population the claim is about, by the variables a selection node points into."""

    source_population: str
    target_population: str
    selection_targets: Sequence[str] = ()

    def _wire(self) -> dict[str, Any]:
        return {
            "source_population": self.source_population,
            "target_population": self.target_population,
            "selection_targets": list(self.selection_targets),
        }


@dataclass(frozen=True, slots=True)
class SurrogateBridge:
    """A surrogate bridge for a claim about a population the validation trials were not run in.

    ``surrogate`` and ``terminal`` are the surrogate-effect and terminal-effect quantities, in
    the claim's population. ``validation_trials`` are the ``(surrogate effect, terminal effect)``
    pairs of the validation trials; ``validated_population`` and ``validated_regime`` are where
    they were run. The transportability premise is stated by the bridge and checked against
    ``diagram``: a selection node into ``terminal_variable`` or a relation modifier contradicts
    it.
    """

    surrogate: ScientificQuantity
    terminal: ScientificQuantity
    validation_trials: Sequence[tuple[float, float]]
    validated_population: str
    validated_regime: str
    validation_evidence: str
    diagram: BridgeDiagram
    terminal_variable: str
    relation_modifiers: Sequence[str] = ()
    min_trials: int = 4
    min_r_squared: float = 0.9

    def _wire(self) -> dict[str, Any]:
        if not isinstance(self.diagram, BridgeDiagram):
            raise CausalTypeError("diagram must be a BridgeDiagram")
        return {
            "surrogate": self.surrogate._wire(),
            "terminal": self.terminal._wire(),
            "validation": {
                "trials": [
                    [
                        real(s, "surrogate effect", finite=True),
                        real(t, "terminal effect", finite=True),
                    ]
                    for s, t in self.validation_trials
                ],
                "min_trials": int(self.min_trials),
                "min_r_squared": real(self.min_r_squared, "min_r_squared", finite=True),
            },
            "validated_in": {
                "population_id": self.validated_population,
                "regime_id": self.validated_regime,
            },
            "validation_evidence": self.validation_evidence,
            "diagram": self.diagram._wire(),
            "terminal_variable": self.terminal_variable,
            "relation_modifiers": list(self.relation_modifiers),
        }


def from_surrogate_bridge(
    bridge: SurrogateBridge, claim: Any, *, provider_id: str
) -> TransportedConstruction:
    """The terminal-outcome value a surrogate bridge derives for the claim's population.

    ``claim`` is a :class:`~antecedent.scientific_bridge.SourceClaim` about the surrogate
    quantity. The response states the bridged value at ``bridge.terminal``; the source
    population is where the relation was validated, the identity the diagram digest, and the
    premise, its modifiers and the validation evidence are assumptions.

    Raises:
        TransportedResponseRefusal: a selection node into the terminal outcome or a modifier
            (``bridge_assumption_missing``), a diagram between other populations
            (``bridge_binding_mismatch``), a surrogate effect outside the validated range, an
            unlicensed trial-level fit, and a claim about the validated scope
            (``not_transported``: a native bridged claim).
    """
    from .scientific_bridge import SourceClaim

    if not isinstance(bridge, SurrogateBridge):
        raise CausalTypeError("bridge must be a SurrogateBridge")
    if not isinstance(claim, SourceClaim):
        raise CausalTypeError("claim must be a scientific_bridge.SourceClaim")
    return _unwrap(
        _native.surrogate_bridge_construct(
            _dumps(bridge._wire()), _dumps(claim.to_wire()), _provider(provider_id)
        )
    )
