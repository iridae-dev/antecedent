"""Transport of fitted nonlinear and non-Gaussian mechanisms into a target population.

:mod:`antecedent.mechanism_composition` carries fitted additive-noise linear-Gaussian
mechanisms. This module is the same selection-diagram route for typed families: a conditional
mean that is a polynomial or interaction basis of the parents, or a compiled mechanism
expression, under Gaussian, heteroscedastic Gaussian, log-normal, Bernoulli, Poisson or gamma
noise::

    from antecedent import typed_mechanism_transport as tmt

    income = tmt.MechanismSpecification("income", family="gaussian")
    demand = tmt.MechanismSpecification(
        "demand", parents=("price", "income"), family="poisson", mean=tmt.Polynomial(degree=2)
    )
    source_demand = tmt.fit_typed_mechanism(demand, source_data, population="source")
    target_demand = source_demand.refit_in_target(target_data, population="target")
    transport = tmt.TypedTransport(
        source_population="source",
        target_population="target",
        selection_nodes=(tmt.SelectionNode("S", ("demand",)),),
        dependence=tmt.TypedDependence.independent_by_design("disjoint samples"),
    )
    transported = tmt.transport_typed_mechanisms(transport, [source_income, source_demand], [target_demand])
    transported.mean_contrast(tmt.MeanContrast.mean_at("demand", {"price": 2.0}))

What this is, and is not:

* **Per-mechanism invariance from the selection diagram.** A mechanism no selection node points
  into is declared invariant and carried from the source fit. A mechanism a selection node
  points into is shifted and used only from a target fit. A shifted mechanism with no target fit
  is refused (``typed_mechanism_transport.shifted_without_target_evidence``, reason code
  ``transport_missing_evidence``): missing evidence never makes a mechanism invariant.
* **A declared invariance is tested where target data allow.** A target fit supplied for an
  invariant-declared mechanism is compared with the source fit by a Wald test of the full
  parameter vectors; a rejection refuses the transport
  (``typed_mechanism_transport.invariance_contradicted``). A non-rejection is not evidence of
  invariance, and an invariance with no target data is an untested premise that the receipt
  records (``consistency`` is ``None``).
* **Target-specific refits.** :meth:`FittedTypedMechanism.refit_in_target` refits a source
  mechanism's specification on target data on the source's design, so parameters are comparable.
* **Uncertainty has distinct meanings.** The parameter variance of a contrast is first order,
  ``g' Sigma g`` with ``Sigma`` the source fit's covariance for a carried mechanism and the
  target fit's for a refit one, joined by the declared :class:`TypedDependence`; unknown
  dependence between more than one involved mechanism is refused, never defaulted to
  independence. The transport assumption adds no variance (``uncertainty_scope``). The Monte
  Carlo error of a response (``monte_carlo_se``) is numerical-approximation uncertainty and is
  reported apart. No interval is produced: :meth:`TypedTransportedComposition.interval` raises
  ``cell_not_licensed``. Support is ``extrapolative`` and inference ``point_only``.

Every fit, admission, test, response, variance and identity is computed in Rust; this module
builds the declaration and presents the result.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import _native
from ._freeze import thaw
from ._refusal import StructuredRefusal, raise_refusal
from .errors import CausalTypeError, CausalValueError
from .execution import Execution, resolve_execution
from .mechanism_composition import MeanContrast, SelectionNode, TransportInvariance

__all__ = [
    "FAMILIES",
    "ContrastEstimate",
    "Expression",
    "FittedTypedMechanism",
    "InvarianceCheck",
    "MeanContrast",
    "MechanismSpecification",
    "Polynomial",
    "SelectionNode",
    "TransportInvariance",
    "TypedDependence",
    "TypedDisposition",
    "TypedMechanismRefusal",
    "TypedTransport",
    "TypedTransportReceipt",
    "TypedTransportedComposition",
    "TypedUncertainty",
    "check_invariance",
    "fit_typed_mechanism",
    "transport_typed_mechanisms",
    "verify_typed_transport",
]

#: The response families a typed mechanism can have.
FAMILIES = (
    "gaussian",
    "heteroscedastic_gaussian",
    "log_normal",
    "bernoulli",
    "poisson",
    "gamma",
)


class TypedMechanismRefusal(StructuredRefusal):
    """A typed refusal of the typed-mechanism transport route.

    Carries the first defect's registered ``reason_code``, namespaced
    ``typed_mechanism_transport.*`` (or ``mechanism_transport.*``) ``detail``, ``offending``
    subject and ``remedy``. ``refusals`` holds **every** defect found, in checker order.
    """

    def __init__(self, refusal: Mapping[str, Any], *, text: str | None = None) -> None:
        super().__init__(refusal, text=text)
        listed = refusal.get("refusals")
        self.refusals: tuple[Mapping[str, Any], ...] = tuple(listed) if listed else (refusal,)

    @property
    def details(self) -> tuple[str, ...]:
        """The namespaced detail of every refusal."""
        return tuple(str(item["detail"]) for item in self.refusals)


# --------------------------------------------------------------------------- helpers


def _name(value: object, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CausalTypeError(f"{what} must be a non-empty string")
    return value


def _number(value: object, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise CausalTypeError(f"{what} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise CausalValueError(f"{what} must be finite")
    return number


def _dumps(value: object) -> str:
    return json.dumps(thaw(value), allow_nan=False, separators=(",", ":"))


def _unwrap(pair: tuple[Any, str | None]) -> Any:
    value, refusal = pair
    raise_refusal(refusal, TypedMechanismRefusal)
    return value


def _columns(data: Mapping[str, Sequence[float]] | Any) -> dict[str, list[float]]:
    if hasattr(data, "to_dict") and not isinstance(data, Mapping):
        data = data.to_dict("list")
    if not isinstance(data, Mapping) or not data:
        raise CausalTypeError("data must be a non-empty mapping of column name to values")
    return {
        _name(name, "column name"): [
            _number(v, f"column {name!r} value") for v in np.asarray(values, dtype=float).ravel()
        ]
        for name, values in data.items()
    }


# --------------------------------------------------------------------------- declarations


@dataclass(frozen=True, slots=True)
class Polynomial:
    """``intercept + sum_j sum_{d<=degree} b_jd z_j^d`` (plus pairwise products when ``interactions``).

    ``z`` are the parents standardized on the source fit's design.
    """

    degree: int = 1
    interactions: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.degree, bool) or not isinstance(self.degree, int):
            raise CausalTypeError("degree must be an integer")
        if not isinstance(self.interactions, bool):
            raise CausalTypeError("interactions must be a bool")

    def to_wire(self) -> dict[str, Any]:
        return {"kind": "polynomial", "degree": self.degree, "interactions": self.interactions}


@dataclass(frozen=True, slots=True)
class Expression:
    """A compiled mechanism expression over the raw parents (variable ``j`` is parent ``j``).

    ``expression`` is the tagged-operation mapping of Rust's ``MechanismExpression``
    (``constant``, ``variable``, ``parameter``, ``add``, ``multiply``, ``tanh``, ``sin``,
    ``softplus``); ``start`` fixes the parameter count and the fit's starting values.
    """

    expression: Mapping[str, Any]
    start: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.expression, Mapping):
            raise CausalTypeError("expression must be a mapping")
        object.__setattr__(self, "start", tuple(_number(v, "start") for v in self.start))

    def to_wire(self) -> dict[str, Any]:
        return {
            "kind": "expression",
            "expression": thaw(self.expression),
            "start": list(self.start),
        }


@dataclass(frozen=True, slots=True)
class MechanismSpecification:
    """What a typed mechanism is: its variable, parents, conditional-mean form and response family."""

    variable: str
    parents: tuple[str, ...] = ()
    family: str = "gaussian"
    mean: Polynomial | Expression = Polynomial()

    def __post_init__(self) -> None:
        _name(self.variable, "variable")
        if isinstance(self.parents, str):
            raise CausalTypeError("parents must be an iterable of variable names")
        object.__setattr__(self, "parents", tuple(_name(p, "parent") for p in self.parents))
        if self.family not in FAMILIES:
            raise CausalValueError(
                f"family must be one of {', '.join(FAMILIES)}; got {self.family!r}"
            )
        if not isinstance(self.mean, (Polynomial, Expression)):
            raise CausalTypeError("mean must be a Polynomial or an Expression")

    def to_wire(self) -> dict[str, Any]:
        return {
            "variable": self.variable,
            "parents": list(self.parents),
            "family": self.family,
            "mean": self.mean.to_wire(),
        }


@dataclass(frozen=True, slots=True)
class TypedDependence:
    """What is declared about the dependence between the fits' parameter estimates."""

    kind: str
    justification: str | None = None
    variables: tuple[str, ...] = ()
    covariance: tuple[float, ...] = ()
    blocks: tuple[tuple[str, str, tuple[float, ...]], ...] = ()

    @classmethod
    def unknown(cls) -> TypedDependence:
        """Nothing declared: an uncertainty claim over more than one mechanism is refused."""
        return cls("unknown")

    @classmethod
    def independent_by_design(cls, justification: str) -> TypedDependence:
        """The fits used disjoint, independently generated evidence."""
        return cls("independent_by_design", justification=_name(justification, "justification"))

    @classmethod
    def joint(cls, variables: Sequence[str], covariance: Any) -> TypedDependence:
        """One aligned joint covariance of the listed mechanisms' full parameter vectors."""
        flat = np.asarray(covariance, dtype=float)
        return cls(
            "joint",
            variables=tuple(_name(v, "variable") for v in variables),
            covariance=tuple(float(x) for x in flat.ravel()),
        )

    @classmethod
    def cross_covariance(cls, blocks: Sequence[tuple[str, str, Any]]) -> TypedDependence:
        """Cross-covariance blocks ``(first, second, dim(first) x dim(second) matrix)`` per pair."""
        return cls(
            "cross_covariance",
            blocks=tuple(
                (
                    _name(a, "first"),
                    _name(b, "second"),
                    tuple(float(x) for x in np.asarray(m, dtype=float).ravel()),
                )
                for a, b, m in blocks
            ),
        )

    def to_wire(self) -> dict[str, Any]:
        if self.kind == "independent_by_design":
            return {"kind": self.kind, "justification": self.justification}
        if self.kind == "joint":
            return {
                "kind": self.kind,
                "variables": list(self.variables),
                "covariance": list(self.covariance),
            }
        if self.kind == "cross_covariance":
            return {
                "kind": self.kind,
                "blocks": [
                    {"first": a, "second": b, "values": list(values)}
                    for a, b, values in self.blocks
                ],
            }
        return {"kind": "unknown"}


@dataclass(frozen=True, slots=True)
class TypedTransport:
    """What a transport of fitted typed mechanisms to a target population declares.

    ``selection_nodes`` are the selection diagram from ``source_population`` to
    ``target_population``: the variables each node points into are the *shifted* mechanisms.
    ``invariance_level`` is the level of the Wald test of a mechanism declared invariant against
    supplied target data.
    """

    source_population: str
    target_population: str
    selection_nodes: tuple[SelectionNode, ...]
    target_regime: str = "observational"
    invariances: tuple[TransportInvariance, ...] = ()
    dependence: TypedDependence = TypedDependence("unknown")
    invariance_level: float = 0.01

    def __post_init__(self) -> None:
        _name(self.source_population, "source_population")
        _name(self.target_population, "target_population")
        _name(self.target_regime, "target_regime")
        object.__setattr__(self, "selection_nodes", tuple(self.selection_nodes))
        object.__setattr__(self, "invariances", tuple(self.invariances))
        for node in self.selection_nodes:
            if not isinstance(node, SelectionNode):
                raise CausalTypeError("selection_nodes must hold SelectionNode values")
        for item in self.invariances:
            if not isinstance(item, TransportInvariance):
                raise CausalTypeError("invariances must hold TransportInvariance values")
        if not isinstance(self.dependence, TypedDependence):
            raise CausalTypeError("dependence must be a TypedDependence")
        object.__setattr__(
            self, "invariance_level", _number(self.invariance_level, "invariance_level")
        )

    def to_wire(self) -> dict[str, Any]:
        return {
            "target": {"population": self.target_population, "regime": self.target_regime},
            "diagram": {
                "source_population": self.source_population,
                "target_population": self.target_population,
                "selection_nodes": [n.to_wire() for n in self.selection_nodes],
            },
            "invariances": [i.to_wire() for i in self.invariances],
            "dependence": self.dependence.to_wire(),
            "invariance_level": self.invariance_level,
        }


# --------------------------------------------------------------------------- fits


@dataclass(frozen=True, slots=True, repr=False, eq=False)
class FittedTypedMechanism:
    """A typed mechanism fitted from one population's data, with its parameter covariance."""

    variable: str
    family: str
    parents: tuple[str, ...]
    parameters: np.ndarray
    parameter_labels: tuple[str, ...]
    covariance: np.ndarray
    n_obs: int
    population: str
    regime: str
    artifact_id: str
    evidence_id: str
    _handle: Any

    def __repr__(self) -> str:
        return (
            f"FittedTypedMechanism({self.variable!r}, family={self.family!r}, "
            f"population={self.population!r}, n_obs={self.n_obs})"
        )

    def conditional_mean(self, parents: Sequence[float]) -> float:
        """``E[variable | parents]`` at raw parent values in specification order."""
        return float(self._handle.conditional_mean([_number(p, "parent value") for p in parents]))

    def refit_in_target(
        self,
        data: Mapping[str, Sequence[float]] | Any,
        *,
        population: str,
        regime: str = "observational",
        artifact_id: str | None = None,
        evidence_id: str | None = None,
        execution: Execution | None = None,
    ) -> FittedTypedMechanism:
        """Refit this mechanism's specification on target data, on this fit's design."""
        execution = resolve_execution(execution)
        context = _context(self.variable, population, regime, artifact_id, evidence_id)
        handle = _unwrap(
            _native.typed_mechanism_refit(
                self._handle, _columns(data), _dumps(context), cancel=execution.cancel
            )
        )
        return _fitted(handle)


def _context(
    variable: str,
    population: str,
    regime: str,
    artifact_id: str | None,
    evidence_id: str | None,
) -> dict[str, str]:
    return {
        "population": _name(population, "population"),
        "regime": _name(regime, "regime"),
        "artifact_id": artifact_id or f"fit:{population}:{variable}",
        "evidence_id": evidence_id or f"data:{population}",
    }


def _fitted(handle: Any) -> FittedTypedMechanism:
    summary = json.loads(handle.summary_json())
    spec = summary["specification"]
    return FittedTypedMechanism(
        variable=str(spec["variable"]),
        family=str(spec["family"]),
        parents=tuple(spec["parents"]),
        parameters=np.asarray(summary["parameters"], dtype=float),
        parameter_labels=tuple(summary["parameter_labels"]),
        covariance=np.asarray(summary["covariance"], dtype=float).reshape(
            len(summary["parameters"]), len(summary["parameters"])
        ),
        n_obs=int(summary["n_obs"]),
        population=str(summary["population"]),
        regime=str(summary["regime"]),
        artifact_id=str(summary["artifact_id"]),
        evidence_id=str(summary["evidence_id"]),
        _handle=handle,
    )


def fit_typed_mechanism(
    specification: MechanismSpecification,
    data: Mapping[str, Sequence[float]] | Any,
    *,
    population: str,
    regime: str = "observational",
    artifact_id: str | None = None,
    evidence_id: str | None = None,
    design_from: FittedTypedMechanism | None = None,
    execution: Execution | None = None,
) -> FittedTypedMechanism:
    """Fit one typed mechanism from one population's data.

    ``design_from`` fixes the parent standardization to that fit's, so the two fits'
    parameters are comparable (use :meth:`FittedTypedMechanism.refit_in_target`).

    Raises:
        TypedMechanismRefusal: a malformed specification or data, a response outside the
            family's support, a singular design, or a fit that did not converge (including
            complete separation of a Bernoulli response).
    """
    if not isinstance(specification, MechanismSpecification):
        raise CausalTypeError("specification must be a MechanismSpecification")
    if design_from is not None and not isinstance(design_from, FittedTypedMechanism):
        raise CausalTypeError("design_from must be a FittedTypedMechanism")
    execution = resolve_execution(execution)
    context = _context(specification.variable, population, regime, artifact_id, evidence_id)
    handle = _unwrap(
        _native.typed_mechanism_fit(
            _dumps(specification.to_wire()),
            _columns(data),
            _dumps(context),
            None if design_from is None else design_from._handle,
            cancel=execution.cancel,
        )
    )
    return _fitted(handle)


@dataclass(frozen=True, slots=True)
class InvarianceCheck:
    """Wald comparison of a source and a target fit of one mechanism."""

    statistic: float
    degrees_of_freedom: int
    p_value: float
    level: float
    rejected: bool


def _check_of(wire: Mapping[str, Any] | None) -> InvarianceCheck | None:
    if wire is None:
        return None
    return InvarianceCheck(
        statistic=float(wire["statistic"]),
        degrees_of_freedom=int(wire["degrees_of_freedom"]),
        p_value=float(wire["p_value"]),
        level=float(wire["level"]),
        rejected=bool(wire["rejected"]),
    )


def check_invariance(
    source: FittedTypedMechanism, target: FittedTypedMechanism, *, level: float = 0.01
) -> InvarianceCheck:
    """Wald test that a mechanism's full parameter vector is the same in two populations.

    The two fits must share a specification and design (refit with
    :meth:`FittedTypedMechanism.refit_in_target`); their covariances are summed (independent
    samples). A non-rejection is not evidence of invariance.
    """
    for what, fit in (("source", source), ("target", target)):
        if not isinstance(fit, FittedTypedMechanism):
            raise CausalTypeError(f"{what} must be a FittedTypedMechanism")
    wire = _unwrap(
        _native.typed_mechanism_invariance_check(
            source._handle, target._handle, _number(level, "level")
        )
    )
    check = _check_of(json.loads(wire))
    assert check is not None
    return check


# --------------------------------------------------------------------------- results


@dataclass(frozen=True, slots=True)
class TypedDisposition:
    """One component of a transported composition: carried or refitted, and from where."""

    variable: str
    role: str
    family: str
    fitted_population: str
    artifact_id: str
    evidence_id: str
    covariance_source: str
    consistency: InvarianceCheck | None


@dataclass(frozen=True, slots=True)
class TypedTransportReceipt:
    """The typed receipt of one transported typed-mechanism composition.

    ``support`` is always ``extrapolative`` and ``inference`` always ``point_only``.
    ``uncertainty_scope`` says what the variance of a contrast contains: the fits' parameter
    covariance only; the transport assumption is not a variance term.
    """

    route: str
    identity: str
    source_population: str
    target_population: str
    target_regime: str
    selection_nodes: tuple[SelectionNode, ...]
    selection_targets: tuple[str, ...]
    dispositions: tuple[TypedDisposition, ...]
    invariance_level: float
    uncertainty_scope: str
    support: str
    inference: str
    assumptions: tuple[str, ...]

    @property
    def transported_variables(self) -> tuple[str, ...]:
        """Variables whose mechanism was carried from the source fit."""
        return tuple(d.variable for d in self.dispositions if d.role == "transported")

    @property
    def native_variables(self) -> tuple[str, ...]:
        """Variables whose mechanism was refitted in the target population."""
        return tuple(d.variable for d in self.dispositions if d.role == "native_in_target")

    @property
    def untested_invariances(self) -> tuple[str, ...]:
        """Carried variables whose invariance no target data tested."""
        return tuple(
            d.variable
            for d in self.dispositions
            if d.role == "transported" and d.consistency is None
        )

    def as_transport_receipt(self) -> Any:
        """The shared :class:`antecedent.transported_response.TransportReceipt` of this route."""
        from .transported_response import TransportReceipt

        return TransportReceipt(
            route="mechanism",
            source_identity=self.identity,
            source_populations=(self.source_population,),
            target_population=self.target_population,
            transport_identities=(self.identity,),
            assumptions=self.assumptions,
            support=self.support,
            representation="point_only",
        )


def _receipt_of(wire: Mapping[str, Any]) -> TypedTransportReceipt:
    return TypedTransportReceipt(
        route=str(wire["route"]),
        identity=str(wire["identity"]),
        source_population=str(wire["source_population"]),
        target_population=str(wire["target_population"]),
        target_regime=str(wire["target_regime"]),
        selection_nodes=tuple(
            SelectionNode(n["name"], tuple(n["points_into"])) for n in wire["selection_nodes"]
        ),
        selection_targets=tuple(wire["selection_targets"]),
        dispositions=tuple(
            TypedDisposition(
                variable=d["variable"],
                role=d["role"],
                family=d["family"],
                fitted_population=d["fitted_population"],
                artifact_id=d["artifact_id"],
                evidence_id=d["evidence_id"],
                covariance_source=d["covariance_source"],
                consistency=_check_of(d["consistency"]),
            )
            for d in wire["dispositions"]
        ),
        invariance_level=float(wire["invariance_level"]),
        uncertainty_scope=str(wire["uncertainty_scope"]),
        support=str(wire["support"]),
        inference=str(wire["inference"]),
        assumptions=tuple(wire["assumptions"]),
    )


@dataclass(frozen=True, slots=True)
class ContrastEstimate:
    """A Monte Carlo contrast estimate with its numerical-approximation standard error."""

    value: float
    monte_carlo_se: float
    draws: int


@dataclass(frozen=True, slots=True)
class TypedUncertainty:
    """A first-order parameter variance of a contrast, scoped, with the Monte Carlo error apart.

    ``variance`` is the parameter variance ``g' Sigma g`` only (statistical); the estimate's
    ``monte_carlo_se`` is numerical approximation; neither contains the transport assumption.
    Uncalibrated; no interval follows.
    """

    estimate: ContrastEstimate
    variance: float
    standard_error: float
    dependence: str
    scope: str
    gradient: tuple[tuple[str, str, float], ...]


def _estimate_of(wire: Mapping[str, Any]) -> ContrastEstimate:
    return ContrastEstimate(
        value=float(wire["value"]),
        monte_carlo_se=float(wire["monte_carlo_se"]),
        draws=int(wire["draws"]),
    )


@dataclass(frozen=True, slots=True, repr=False, eq=False)
class TypedTransportedComposition:
    """An executable target-population composition of typed mechanisms with its receipt."""

    identity: str
    receipt: TypedTransportReceipt
    _handle: Any

    def __repr__(self) -> str:
        return (
            f"TypedTransportedComposition(target={self.receipt.target_population!r}, "
            f"carried={list(self.receipt.transported_variables)}, "
            f"refit={list(self.receipt.native_variables)})"
        )

    @property
    def transported_variables(self) -> tuple[str, ...]:
        return self.receipt.transported_variables

    def mean_contrast(
        self,
        contrast: MeanContrast,
        *,
        draws: int = 100_000,
        execution: Execution | None = None,
    ) -> ContrastEstimate:
        """``sum_k weight_k * E[outcome | do(arm_k)]`` by Monte Carlo over ``draws`` paired units."""
        execution = resolve_execution(execution)
        wire = _unwrap(
            self._handle.mean_contrast(
                _dumps(_contrast(contrast).to_wire()),
                _draws(draws),
                int(execution.seed),
                cancel=execution.cancel,
            )
        )
        return _estimate_of(json.loads(wire))

    def mean_contrast_uncertainty(
        self,
        contrast: MeanContrast,
        *,
        draws: int = 100_000,
        execution: Execution | None = None,
    ) -> TypedUncertainty:
        """First-order parameter variance of a contrast, scoped by the receipt.

        Raises:
            TypedMechanismRefusal: unknown or incomplete dependence between the involved
                mechanisms (``typed_mechanism_transport.parameter_dependence_unknown``).
        """
        execution = resolve_execution(execution)
        wire = json.loads(
            _unwrap(
                self._handle.mean_contrast_uncertainty(
                    _dumps(_contrast(contrast).to_wire()),
                    _draws(draws),
                    int(execution.seed),
                    cancel=execution.cancel,
                )
            )
        )
        return TypedUncertainty(
            estimate=_estimate_of(wire["estimate"]),
            variance=float(wire["variance"]),
            standard_error=float(wire["standard_error"]),
            dependence=str(wire["dependence"]),
            scope=str(wire["scope"]),
            gradient=tuple(
                (str(g["mechanism"]), str(g["parameter"]), float(g["gradient"]))
                for g in wire["gradient"]
            ),
        )

    def interval(self) -> None:
        """Always raises ``cell_not_licensed``: no coverage includes the transport assumption."""
        raise_refusal(self._handle.interval(), TypedMechanismRefusal)


def _contrast(contrast: object) -> MeanContrast:
    if not isinstance(contrast, MeanContrast):
        raise CausalTypeError("contrast must be a MeanContrast")
    return contrast


def _draws(draws: object) -> int:
    if isinstance(draws, bool) or not isinstance(draws, int) or draws < 1:
        raise CausalValueError("draws must be a positive integer")
    return draws


def _handles(fits: Sequence[FittedTypedMechanism], what: str) -> list[Any]:
    out = []
    for fit in fits:
        if not isinstance(fit, FittedTypedMechanism):
            raise CausalTypeError(f"{what} must hold FittedTypedMechanism values")
        out.append(fit._handle)
    return out


def transport_typed_mechanisms(
    transport: TypedTransport,
    source_fits: Sequence[FittedTypedMechanism],
    target_fits: Sequence[FittedTypedMechanism] = (),
    *,
    execution: Execution | None = None,
) -> TypedTransportedComposition:
    """Carry invariant typed mechanisms and use target refits for shifted ones.

    ``source_fits`` are the mechanisms fitted in the source population; ``target_fits`` are
    those fitted in the target: the refit of every shifted mechanism, and optionally consistency
    evidence for a mechanism declared invariant.

    Raises:
        TypedMechanismRefusal: every unmet premise, in checker order, among them a shifted
            mechanism with no target fit, a declared invariance the target fit contradicts,
            fits on another design, populations the diagram does not describe, and a parent
            without a mechanism.
    """
    if not isinstance(transport, TypedTransport):
        raise CausalTypeError("transport must be a TypedTransport")
    execution = resolve_execution(execution)
    handle, receipt = _unwrap(
        _native.typed_mechanism_transport(
            _dumps(transport.to_wire()),
            _handles(source_fits, "source_fits"),
            _handles(target_fits, "target_fits"),
            cancel=execution.cancel,
        )
    )
    parsed = _receipt_of(json.loads(receipt))
    return TypedTransportedComposition(identity=parsed.identity, receipt=parsed, _handle=handle)


def verify_typed_transport(
    transport: TypedTransport,
    source_fits: Sequence[FittedTypedMechanism],
    target_fits: Sequence[FittedTypedMechanism],
    *,
    expected_identity: str,
    execution: Execution | None = None,
) -> TypedTransportReceipt:
    """Re-derive the transport and require the identity.

    Raises ``typed_mechanism_transport.transport_identity_mismatch`` when the recomputed
    identity is not ``expected_identity``.
    """
    if not isinstance(transport, TypedTransport):
        raise CausalTypeError("transport must be a TypedTransport")
    execution = resolve_execution(execution)
    receipt = _unwrap(
        _native.typed_mechanism_verify_transport(
            _dumps(transport.to_wire()),
            _handles(source_fits, "source_fits"),
            _handles(target_fits, "target_fits"),
            _name(expected_identity, "expected_identity"),
            cancel=execution.cancel,
        )
    )
    return _receipt_of(json.loads(receipt))
