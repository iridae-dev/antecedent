"""The transported static path-specific counterfactual, narrow class (2.3.0 A5).

Two populations, a source and a target, share one structural model

* covariates ``Z``: pre-treatment variables whose finite-support joint law may differ between
  the populations;
* a treatment ``A`` that is only ever *set*;
* mediator and outcome mechanisms ``V = alpha(Z) + sum_p beta_p(Z) * V_p + U``, where
  ``alpha`` and every ``beta_p`` are affine in ``Z`` and the noises ``U`` are additive and
  independent of ``Z`` and of each other.

A selection diagram marks every variable whose mechanism (or covariate law) may differ. An
*edge assignment* feeds the treated value along some children of ``A`` and the control value
along the others; the path-specific contrast of two assignments (the natural direct and
indirect effects among them) is ``E[Y_plus - Y_minus]`` with one exogenous draw per unit fed
to both worlds. When every selection node points at a covariate or at the treatment, that
contrast does not depend on the noise and the target answer is
``sum_z P_T(z) G(z)``, with ``G(z)`` computed from the source equations alone.

The result reports the target contrast, the source contrast (the answer if the source law were
mistaken for the target's), the per-unit contrast at every target support point and the
derivation: which premises were **checked** (acyclic well-formed model, no selection on a
mediator or outcome mechanism, covariates are pre-treatment roots, target support inside
source support, regime evidence present) and which were only **declared**. The declarations
(``Premises``: additive noise, shared noise laws, cross-world independence) default to
*undeclared*: omitting one refuses with the Rust detail
(``transported_counterfactual.nonadditive_mechanism`` and so on). Declaring one records a claim
the supplied source fit must support; it is never checked here.

A selection node on a mediator or outcome mechanism is outside the class: two models can agree
on the whole source population and differ on the target contrast. When the contrast is
sensitive to the selected mechanism the refusal retains the explicit two-model witness
(:attr:`TransportedCounterfactualRefusal.witness`) and is
``transport_proven_non_transportable``; otherwise it is ``cell_not_licensed`` with no witness
(the class excludes it, no impossibility is claimed).

The GENERAL class (nonparametric mechanisms, recanting witnesses, nonlinear additive noise)
stays closed: :func:`antecedent.temporal_counterfactual.transported_path_specific` never
evaluates. The claim here is a point under a fully specified structural fit; calibration is
unmeasured.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from ._native import (
    consume_transported_counterfactual_artifact as _consume,
)
from ._native import (
    evaluate_transported_counterfactual as _evaluate,
)
from .errors import CausalTypeError, CausalUnsupportedError, CausalValueError

__all__ = [
    "AdditiveNoiseScm",
    "Affine",
    "CovariateLaw",
    "Derivation",
    "EdgeAssignment",
    "Mechanism",
    "NonRecoverableWitness",
    "Premises",
    "SelectionDiagram",
    "TransportedCounterfactualIdentity",
    "TransportedCounterfactualRefusal",
    "TransportedPathSpecificEffect",
    "UnitContrast",
    "consume_transported_counterfactual_artifact",
    "transported_path_specific_effect",
]


def _name(value: object, what: str) -> str:
    if not isinstance(value, str) or not value:
        raise CausalTypeError(f"{what} must be a non-empty string")
    return value


def _real(value: object, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        try:
            number = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError) as error:
            raise CausalTypeError(f"{what} must be a real number") from error
    else:
        number = float(value)
    if not math.isfinite(number):
        raise CausalValueError(
            f"transported_counterfactual.invalid_model: {what} is not finite",
            reason_code="invalid_argument",
        )
    return number


@dataclass(frozen=True, slots=True)
class Affine:
    """``constant + sum slope_k * z_k``: a coefficient that may depend on the covariates."""

    constant: float = 0.0
    slopes: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "constant", _real(self.constant, "constant"))
        object.__setattr__(
            self,
            "slopes",
            {_name(k, "covariate"): _real(v, "slope") for k, v in dict(self.slopes).items()},
        )

    def _wire(self) -> dict[str, Any]:
        return {
            "constant": self.constant,
            "covariate_slopes": [[name, slope] for name, slope in self.slopes.items()],
        }


def _affine(value: Affine | float, what: str) -> Affine:
    if isinstance(value, Affine):
        return value
    return Affine(_real(value, what))


@dataclass(frozen=True, slots=True)
class Mechanism:
    """``node = intercept(z) + sum parents[p](z) * p + U_node`` with additive noise.

    A parent may be a covariate, the treatment or another mechanism node. A coefficient is an
    :class:`Affine` (covariate-dependent) or a plain number (constant).
    """

    node: str
    parents: Mapping[str, Affine | float] = field(default_factory=dict)
    intercept: Affine | float = 0.0

    def __post_init__(self) -> None:
        _name(self.node, "node")
        object.__setattr__(
            self,
            "parents",
            {_name(p, "parent"): _affine(c, "coefficient") for p, c in dict(self.parents).items()},
        )
        object.__setattr__(self, "intercept", _affine(self.intercept, "intercept"))

    def _wire(self) -> dict[str, Any]:
        return {
            "node": self.node,
            "intercept": _affine(self.intercept, "intercept")._wire(),
            "parents": [[p, _affine(c, "coefficient")._wire()] for p, c in self.parents.items()],
        }


@dataclass(frozen=True, slots=True)
class AdditiveNoiseScm:
    """The source structural fit: a treatment, covariates and mediator/outcome mechanisms."""

    treatment: str
    covariates: tuple[str, ...]
    mechanisms: tuple[Mechanism, ...]

    def __post_init__(self) -> None:
        _name(self.treatment, "treatment")
        if isinstance(self.covariates, str):
            raise CausalTypeError("covariates must be a sequence of names")
        object.__setattr__(
            self, "covariates", tuple(_name(c, "covariate") for c in self.covariates)
        )
        object.__setattr__(self, "mechanisms", tuple(self.mechanisms))
        if not all(isinstance(m, Mechanism) for m in self.mechanisms):
            raise CausalTypeError("mechanisms must be Mechanism objects")

    def _wire(self) -> dict[str, Any]:
        return {
            "treatment": self.treatment,
            "covariates": list(self.covariates),
            "mechanisms": [m._wire() for m in self.mechanisms],
        }


@dataclass(frozen=True, slots=True)
class CovariateLaw:
    """A finite-support covariate law: ``(point, weight)`` pairs, weights summing to one.

    Each point maps every covariate name to a value. Use :meth:`univariate` for one covariate.
    """

    points: tuple[tuple[Mapping[str, float], float], ...]

    def __post_init__(self) -> None:
        points = []
        for point, weight in self.points:
            values = {
                _name(k, "covariate"): _real(v, "covariate value") for k, v in dict(point).items()
            }
            points.append((values, _real(weight, "weight")))
        object.__setattr__(self, "points", tuple(points))

    @classmethod
    def univariate(cls, covariate: str, weights: Mapping[float, float]) -> CovariateLaw:
        """The law of one covariate: ``{value: probability}``."""
        return cls(tuple(({covariate: float(v)}, float(w)) for v, w in weights.items()))

    def _wire(self) -> dict[str, Any]:
        return {
            "points": [
                {"values": [[k, v] for k, v in point.items()], "weight": weight}
                for point, weight in self.points
            ]
        }


@dataclass(frozen=True, slots=True)
class SelectionDiagram:
    """Selection nodes: ``label -> the variable whose mechanism or law may differ``."""

    nodes: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        nodes = {
            _name(k, "selection label"): _name(v, "selection target")
            for k, v in dict(self.nodes).items()
        }
        object.__setattr__(self, "nodes", nodes)

    @classmethod
    def on(cls, *targets: str) -> SelectionDiagram:
        """One selection node ``S_<target>`` per listed variable."""
        return cls({f"S_{t}": t for t in targets})

    def _wire(self) -> list[dict[str, str]]:
        return [{"label": label, "target": target} for label, target in self.nodes.items()]


@dataclass(frozen=True, slots=True)
class EdgeAssignment:
    """Which children of the treatment are fed the treated value in each world.

    The added world feeds ``treated_value`` along the edges to ``plus`` and ``control_value``
    along the other treatment edges; the subtracted world does the same with ``minus``. The
    contrast is ``E[Y_plus - Y_minus]``. Use :meth:`natural_direct`, :meth:`natural_indirect`
    or :meth:`total` for the usual ones.
    """

    outcome: str
    treated_value: float
    control_value: float
    plus: tuple[str, ...]
    minus: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _name(self.outcome, "outcome")
        object.__setattr__(self, "treated_value", _real(self.treated_value, "treated_value"))
        object.__setattr__(self, "control_value", _real(self.control_value, "control_value"))
        for what in ("plus", "minus"):
            children = getattr(self, what)
            if isinstance(children, str):
                raise CausalTypeError(f"{what} must be a sequence of node names")
            object.__setattr__(self, what, tuple(_name(c, what) for c in children))

    @classmethod
    def natural_direct(
        cls, outcome: str, *, treated_value: float, control_value: float
    ) -> EdgeAssignment:
        """NDE: only the edge into the outcome carries the treated value (``minus`` is empty)."""
        return cls(outcome, treated_value, control_value, (outcome,), ())

    @classmethod
    def natural_indirect(
        cls,
        outcome: str,
        mediators: Iterable[str],
        *,
        treated_value: float,
        control_value: float,
    ) -> EdgeAssignment:
        """NIE: every treatment edge treated versus only the edge into the outcome treated."""
        return cls(outcome, treated_value, control_value, (*mediators, outcome), (outcome,))

    @classmethod
    def total(
        cls,
        outcome: str,
        mediators: Iterable[str],
        *,
        treated_value: float,
        control_value: float,
    ) -> EdgeAssignment:
        """Total effect: every treatment edge treated versus every edge control."""
        return cls(outcome, treated_value, control_value, (*mediators, outcome), ())

    def _wire(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "treated_value": self.treated_value,
            "control_value": self.control_value,
            "plus": list(self.plus),
            "minus": list(self.minus),
        }


@dataclass(frozen=True, slots=True)
class Premises:
    """The premises the caller declares; each defaults to UNDECLARED.

    * ``additive_noise``: the mediator and outcome mechanisms have the stated additive-noise
      form;
    * ``noise_laws_shared``: the noise laws are the same in both populations and independent
      of the covariates and of each other;
    * ``cross_world_independence``: one exogenous draw per unit is fed to both worlds of the
      contrast (the shared-exogenous semantics of the fixed-population route).

    A declaration is a claim the evidence must support; it is recorded in the derivation and
    bound into the artifact identity, never checked.
    """

    additive_noise: bool = False
    noise_laws_shared: bool = False
    cross_world_independence: bool = False

    def _wire(self) -> dict[str, bool]:
        return {
            "additive_noise": bool(self.additive_noise),
            "noise_laws_shared": bool(self.noise_laws_shared),
            "cross_world_independence": bool(self.cross_world_independence),
        }


@dataclass(frozen=True, slots=True)
class UnitContrast:
    """The target-law contribution of one support point: ``G(z)`` and its target weight."""

    point: Mapping[str, float]
    target_weight: float
    contrast: float


@dataclass(frozen=True, slots=True)
class Derivation:
    """What was checked from the supplied structure and what was only declared."""

    theorem: str
    checked: tuple[str, ...]
    declared: tuple[str, ...]
    claim: str


@dataclass(frozen=True, slots=True)
class TransportedCounterfactualIdentity:
    """Identity digests a consumer retains independently of the artifact bytes.

    Pass it to :func:`consume_transported_counterfactual_artifact` as ``expected=`` to refuse a
    *resealed* change of any premise, law, coefficient, selection, assignment or evidence.
    """

    model_digest: str
    source_law_digest: str
    target_law_digest: str
    selection_digest: str
    assignment_digest: str
    premises_digest: str
    evidence_digest: str
    spec_id: str
    result_digest: str

    def _wire(self) -> dict[str, str]:
        return asdict(self)


def _scm_of(raw: Mapping[str, Any]) -> AdditiveNoiseScm:
    def affine(item: Mapping[str, Any]) -> Affine:
        return Affine(item["constant"], {n: s for n, s in item["covariate_slopes"]})

    return AdditiveNoiseScm(
        treatment=raw["treatment"],
        covariates=tuple(raw["covariates"]),
        mechanisms=tuple(
            Mechanism(
                m["node"],
                {p: affine(c) for p, c in m["parents"]},
                affine(m["intercept"]),
            )
            for m in raw["mechanisms"]
        ),
    )


@dataclass(frozen=True, slots=True)
class NonRecoverableWitness:
    """Two target models that agree with the source and differ on the target contrast.

    ``source_model`` is shared by both candidate worlds; ``target_model_b`` differs from
    ``target_model_a`` (equal to the source) in one coefficient of ``selected_node``'s
    equation: the constant part of ``perturbed_parent``'s coefficient when
    ``perturbed_slope_covariate`` is ``None``, else that covariate's slope, raised by
    ``perturbation``. The source contrast is identical under both; the target contrasts
    differ, verified by arithmetic, so the target contrast is not a function of the source
    population when ``selected_node`` is selected.
    """

    selected_node: str
    perturbed_parent: str
    perturbed_slope_covariate: str | None
    perturbation: float
    source_model: AdditiveNoiseScm
    target_model_a: AdditiveNoiseScm
    target_model_b: AdditiveNoiseScm
    source_contrast: float
    target_contrast_a: float
    target_contrast_b: float


def _witness_of(raw: Mapping[str, Any]) -> NonRecoverableWitness:
    return NonRecoverableWitness(
        selected_node=raw["selected_node"],
        perturbed_parent=raw["perturbed_parent"],
        perturbed_slope_covariate=raw["perturbed_slope_covariate"],
        perturbation=float(raw["perturbation"]),
        source_model=_scm_of(raw["source_model"]),
        target_model_a=_scm_of(raw["target_model_a"]),
        target_model_b=_scm_of(raw["target_model_b"]),
        source_contrast=float(raw["source_contrast"]),
        target_contrast_a=float(raw["target_contrast_a"]),
        target_contrast_b=float(raw["target_contrast_b"]),
    )


class TransportedCounterfactualRefusal(CausalUnsupportedError):
    """A typed refusal of the transported path-specific counterfactual.

    ``reason_code`` and ``remedy`` are the inherited registered fields. ``detail`` is the
    namespaced ``transported_counterfactual.*`` slot: ``nonadditive_mechanism``,
    ``noise_law_not_shared`` and ``cross_world_independence_missing`` (``cell_not_licensed``;
    ``offending`` names the undeclared premise), ``selection_on_mechanism``
    (``transport_proven_non_transportable`` with ``witness`` set, or ``cell_not_licensed``
    without one), ``overlap_failure`` (``transport_support_failure``; ``offending`` is the
    target point outside the source support), ``factor_missing``
    (``transport_missing_evidence``; ``missing_factors``), ``artifact_changed``
    (``route_not_supported``; ``offending`` is the changed identity field) and the
    ``invalid_argument`` family ``invalid_model``, ``invalid_query``, ``invalid_law``,
    ``invalid_diagram`` and ``invalid_factor``.
    """

    def __init__(self, refusal: Mapping[str, Any]) -> None:
        detail = str(refusal["detail"])
        message = detail
        if refusal.get("offending"):
            message += f" at {refusal['offending']}"
        super().__init__(message, reason_code=refusal["code"], remedy=refusal.get("remedy"))
        #: Refusing stage.
        self.stage: str = refusal.get("stage", "")
        #: Namespaced ``family.slot`` detail.
        self.detail: str = detail
        #: Offending node, premise, point or factor, when there is one.
        self.offending: str | None = refusal.get("offending")
        #: The two-model impossibility witness, for a selection on a sensitive mechanism.
        self.witness: NonRecoverableWitness | None = (
            _witness_of(refusal["witness"]) if refusal.get("witness") else None
        )
        #: Regime factors (``source:<regime>`` / ``target:<regime>``) absent from the evidence.
        self.missing_factors: tuple[str, ...] = tuple(refusal.get("missing_factors", ()))


def _raise_refusal(payload: str | None) -> None:
    if payload is not None:
        raise TransportedCounterfactualRefusal(json.loads(payload))


@dataclass(frozen=True, slots=True)
class TransportedPathSpecificEffect:
    """The transported path-specific mean contrast, its derivation and its artifact.

    ``target_contrast`` is ``sum_z P_T(z) G(z)``; ``source_contrast`` is the same contrast
    under the source covariate law (what a source-only analysis would report);
    ``unit_contrasts`` holds ``G`` at every target support point in canonical point order.
    ``inference_claim`` is ``"point_only"``: no interval, and calibration is unmeasured.
    """

    target_contrast: float
    source_contrast: float
    unit_contrasts: tuple[UnitContrast, ...]
    derivation: Derivation
    inference_claim: str
    identity: TransportedCounterfactualIdentity
    artifact: bytes

    def export(self) -> bytes:
        """The checksummed ``transported_counterfactual_v1`` artifact."""
        return self.artifact

    @staticmethod
    def consume(
        artifact: bytes,
        *,
        expected: TransportedCounterfactualIdentity | Mapping[str, str] | None = None,
    ) -> TransportedPathSpecificEffect:
        """Replay an exported artifact; see :func:`consume_transported_counterfactual_artifact`."""
        return consume_transported_counterfactual_artifact(artifact, expected=expected)


def _effect(report_json: str, artifact: bytes) -> TransportedPathSpecificEffect:
    report = json.loads(report_json)
    result = report["result"]
    derivation = result["derivation"]
    return TransportedPathSpecificEffect(
        target_contrast=float(result["target_contrast"]),
        source_contrast=float(result["source_contrast"]),
        unit_contrasts=tuple(
            UnitContrast(
                point={name: float(value) for name, value in u["point"]},
                target_weight=float(u["target_weight"]),
                contrast=float(u["contrast"]),
            )
            for u in result["unit_contrasts"]
        ),
        derivation=Derivation(
            theorem=derivation["theorem"],
            checked=tuple(derivation["checked"]),
            declared=tuple(derivation["declared"]),
            claim=derivation["claim"],
        ),
        inference_claim=report["inference_claim"],
        identity=TransportedCounterfactualIdentity(**report["identity"]),
        artifact=artifact,
    )


def _evidence(
    evidence: Iterable[tuple[str, str, str]] | None,
) -> list[dict[str, str]]:
    if evidence is None:
        # Supplying the source and the target covariate law is the regime evidence the
        # theorem requires: both populations observed, covariates measured.
        evidence = (
            ("source", "observational", "source_covariate_law"),
            ("target", "observational", "target_covariate_law"),
        )
    out = []
    for item in evidence:
        if not isinstance(item, tuple) or len(item) != 3:
            raise CausalTypeError("evidence items are (role, regime, label) triples")
        role, regime, label = item
        out.append({"role": str(role), "regime": _name(regime, "regime"), "label": str(label)})
    return out


def transported_path_specific_effect(
    scm: AdditiveNoiseScm,
    source_law: CovariateLaw,
    target_law: CovariateLaw,
    selection: SelectionDiagram,
    assignment: EdgeAssignment,
    *,
    premises: Premises | None = None,
    evidence: Iterable[tuple[str, str, str]] | None = None,
    artifact_id: str = "transported_counterfactual",
) -> TransportedPathSpecificEffect:
    """The target-population path-specific contrast, from the source equations alone.

    ``scm`` is the source structural fit, ``source_law`` and ``target_law`` the finite covariate
    laws, ``selection`` the selection diagram and ``assignment`` the contrast (see
    :class:`EdgeAssignment`). ``premises`` declares the additive-noise, shared-noise-law and
    cross-world-independence premises explicitly; the default declares none, and omitting one
    refuses. ``evidence`` is a sequence of ``(role, regime, label)`` triples naming the
    supplied source and target regime evidence; by default the two supplied covariate laws are
    the ``observational`` evidence of each population, and ``evidence=()`` withholds it
    (``transported_counterfactual.factor_missing``).

    Raises :class:`TransportedCounterfactualRefusal` (a
    :class:`~antecedent.errors.CausalUnsupportedError`); see its documentation for each
    ``detail`` and ``reason_code``. A selection on a mediator or outcome mechanism retains the
    two-model witness on the exception when the contrast is sensitive to it.
    """
    if not isinstance(scm, AdditiveNoiseScm):
        raise CausalTypeError("scm must be an AdditiveNoiseScm")
    if not isinstance(source_law, CovariateLaw) or not isinstance(target_law, CovariateLaw):
        raise CausalTypeError("source_law and target_law must be CovariateLaw objects")
    if not isinstance(selection, SelectionDiagram):
        raise CausalTypeError("selection must be a SelectionDiagram")
    if not isinstance(assignment, EdgeAssignment):
        raise CausalTypeError("assignment must be an EdgeAssignment")
    declared = premises if premises is not None else Premises()
    if not isinstance(declared, Premises):
        raise CausalTypeError("premises must be a Premises")
    _name(artifact_id, "artifact_id")
    request = {
        "model": scm._wire(),
        "source_law": source_law._wire(),
        "target_law": target_law._wire(),
        "selections": selection._wire(),
        "assignment": assignment._wire(),
        "premises": declared._wire(),
        "evidence": _evidence(evidence),
    }
    report, artifact, refusal = _evaluate(json.dumps(request, allow_nan=False), artifact_id)
    _raise_refusal(refusal)
    if report is None or artifact is None:  # pragma: no cover - the native contract
        raise CausalValueError(
            "the native transported counterfactual returned neither a result nor a refusal"
        )
    return _effect(report, bytes(artifact))


def consume_transported_counterfactual_artifact(
    artifact: bytes,
    *,
    expected: TransportedCounterfactualIdentity | Mapping[str, str] | None = None,
) -> TransportedPathSpecificEffect:
    """Replay an exported artifact and accept only an identical one.

    The request is rebuilt and the transported contrast evaluated again with the same core
    evaluator; the target contrast, the source contrast, every per-unit contrast and the
    derivation must reproduce bit for bit. The replay re-checks the declared premises, so an
    artifact whose premises were relabelled false (or whose selection now points at a mediator
    or outcome mechanism) is refused with the core refusal
    (:class:`TransportedCounterfactualRefusal`). With ``expected`` (the
    :attr:`TransportedPathSpecificEffect.identity` retained out-of-band) a changed premise, law,
    coefficient, selection or assignment is refused even when the artifact was resealed
    consistently (``route_not_supported``, ``transported_counterfactual.artifact_changed``;
    ``offending`` names the changed field). A stored result that does not replay, corruption
    and unknown major versions raise :class:`~antecedent.errors.CausalSerializationError`.
    """
    if not isinstance(artifact, bytes | bytearray | memoryview):
        raise CausalTypeError("artifact must be bytes")
    if expected is None:
        expected_json = None
    elif isinstance(expected, TransportedCounterfactualIdentity):
        expected_json = json.dumps(expected._wire())
    elif isinstance(expected, Mapping):
        expected_json = json.dumps(dict(expected))
    else:
        raise CausalTypeError("expected must be a TransportedCounterfactualIdentity or a mapping")
    data = bytes(artifact)
    report, refusal = _consume(data, expected_json)
    _raise_refusal(refusal)
    if report is None:  # pragma: no cover - the native contract
        raise CausalValueError("the native consumer returned neither a result nor a refusal")
    return _effect(report, data)
