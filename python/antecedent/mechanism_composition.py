"""Composing independently fitted mechanisms into one executable joint model.

Individually valid conditional mechanisms do not imply a valid joint model. A composition is a
*declared* set of mechanisms in semantic coordinates (the quantity each produces, the quantities it
consumes, units, population, regime, fit and evidence identities), with the numbers of each and what
is declared about the dependence between their estimates::

    from antecedent import mechanism_composition as mc

    def quantity(variable, units="u"):
        return mc.Quantity(variable, units=units, population_id="pop")

    def declared(variable, intercept, sigma, parents=()):
        names = [p for p, _ in parents]
        declaration = mc.MechanismDeclaration(
            output=quantity(variable),
            parents=tuple(quantity(p) for p in names),
            artifact_id=f"fit:{variable}",
            evidence_id=f"data:{variable}",
        )
        return mc.Mechanism.declared_linear(
            declaration, intercept=intercept, sigma=sigma, coefficients=dict(parents)
        )

    composition = mc.compose_mechanisms(
        [
            declared("X", 1.0, 2.0),
            declared("M", 0.5, 1.0, [("X", 2.0)]),
            declared("Y", -1.0, 0.5, [("X", 0.5), ("M", 3.0)]),
        ],
        target_population_id="pop",
        admit_declared_mechanisms=True,
    )
    law = composition.intervention_law({"X": 2.0})
    law.mean_of("Y")             # 13.5, exact given the declared mechanisms
    composition.mean_contrast(mc.MeanContrast.ate("Y", "X", 1.0, 0.0))        # 6.5

What this is, and is not:

* **Declared, not native.** A mechanism built from numbers the analyst wrote down is labelled
  declared, is refused unless the composition acknowledges it (``admit_declared_mechanisms``), and
  is listed by :attr:`PreparedMechanismComposition.declared_mechanisms`. Only
  :meth:`Mechanism.fitted_linear`, which reads the numbers of a fitted graphical causal model's
  plain linear-Gaussian mechanism, is labelled fitted. A declaration cannot claim a fit it does not
  have.
* **Composition is exact given the declared mechanisms**; nothing is inferred about units,
  populations, shared causes or dependence. A mechanism from another population or regime is
  admitted only through a declared selection diagram with no arrow into it, or an explicit
  invariance assumption (:class:`Admission`).
* **Uncertainty needs a declared dependence.** Separately fitted mechanisms are not independent by
  default: a variance over more than one mechanism needs an aligned joint law, an explicit
  cross-mechanism covariance block for every pair (a zero block is a declared independence for that
  pair), or independence by design with its justification. Otherwise the refusal is
  ``joint_law_required``. The variance is a first-order parameter-covariance variance only; it
  omits second-order terms, holds residual scales fixed and carries no coverage claim. The interval
  route is closed: :meth:`PreparedMechanismComposition.interval` raises ``cell_not_licensed``.
* **Refresh** (:meth:`PreparedMechanismComposition.refresh_mechanisms`) takes mechanisms refitted on
  new outcome-only data, keeps the plan, recomputes the numbers and the provenance, and refuses a
  refit that changes a mechanism's parents or family. A dependence declared for the earlier fit is
  unknown again until redeclared. A refused refresh leaves the composition as it was.

Every refusal is a :class:`MechanismCompositionRefusal` listing every defect found (``refusals``),
each with a registered ``reason_code``, a ``mechanism_composition.*`` ``detail``, the offending
subject, expected and supplied semantics, and a remedy. Every law, sample, variance, identity and
the artifact is computed in Rust; this module builds the declaration and presents the result.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import _native
from ._artifact_protocol import (
    ExportedArtifact,
    artifact_consumer,
    artifact_export,
    artifact_from_receipt,
    container_bytes,
)
from ._freeze import thaw
from ._mean_binding import bind_mean_reading
from ._refusal import StructuredRefusal, raise_refusal
from ._scientific import ScientificResult
from ._validate import count as _count
from .counterfactual import native_gcm
from .errors import CausalTypeError, CausalValueError
from .execution import Execution, resolve_execution
from .outcomes import Answer, outcome_operation

__all__ = [
    "Admission",
    "MechanismDisposition",
    "MechanismTransport",
    "MechanismTransportReceipt",
    "SelectionNode",
    "TransportedFunctionalUncertainty",
    "TransportedMechanismComposition",
    "TransportInvariance",
    "CompositionExport",
    "ConsumedComposition",
    "CrossCovarianceBlock",
    "Dependence",
    "FiniteDistributionResult",
    "FiniteLaw",
    "FunctionalUncertainty",
    "GaussianFunctionalResult",
    "GaussianLawResult",
    "JointGaussianLaw",
    "LatentCause",
    "Mechanism",
    "MechanismCompositionRefusal",
    "MechanismDeclaration",
    "MechanismRefit",
    "MechanismUncertainty",
    "MeanContrast",
    "ParameterId",
    "ParameterTerm",
    "PreparedMechanismComposition",
    "Quantity",
    "compose_mechanisms",
    "consume_mechanism_composition_artifact",
]

_ROLES = ("treatment", "outcome", "covariate", "mediator", "selection", "utility")
_MEANINGS = ("posterior", "sampling_distribution")


class MechanismCompositionRefusal(StructuredRefusal):
    """A typed refusal of the mechanism-composition route.

    A :class:`~antecedent.errors.CausalUnsupportedError` carrying the first defect's registered
    ``reason_code``, namespaced ``detail``, ``offending`` subject and ``remedy``. ``refusals`` holds
    **every** defect found, in checker order (each a mapping with ``code``, ``detail``,
    ``offending``, ``expected``, ``supplied`` and ``remedy``); :attr:`details` lists their details.
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
    raise_refusal(refusal, MechanismCompositionRefusal)
    return value


def _pairs(mapping: Mapping[str, Any] | Sequence[tuple[str, Any]], what: str) -> list[list[Any]]:
    items = mapping.items() if isinstance(mapping, Mapping) else mapping
    return [[_name(name, what), value] for name, value in items]


# --------------------------------------------------------------------------- declarations


@dataclass(frozen=True, slots=True)
class Quantity:
    """The semantic coordinate of one scalar quantity a mechanism produces or consumes.

    A parent must be a *realization* of the same quantity its producer yields (same variable,
    role, units, population, regime, horizon and transform); a mean or other summary cannot be a
    parent value.
    """

    variable_id: str
    units: str
    population_id: str
    regime_id: str = "observational"
    role: str = "covariate"
    variable_name: str | None = None
    horizon: int = 0
    functional_id: str = "realization"
    conditioning: tuple[tuple[str, str], ...] = ()
    transform_id: str = "identity"

    def __post_init__(self) -> None:
        if self.role not in _ROLES:
            raise CausalValueError(f"role must be one of {', '.join(_ROLES)}; got {self.role!r}")

    def to_wire(self) -> dict[str, Any]:
        """The declaration the native layer reads."""
        return {
            "version": 1,
            "variable_id": self.variable_id,
            "variable_name": self.variable_name or self.variable_id,
            "role": self.role,
            "units": self.units,
            "population_id": self.population_id,
            "regime_id": self.regime_id,
            "horizon": _count(
                self.horizon,
                "horizon",
                minimum=0,
                maximum=2**64 - 1,
                reason_code="invalid_argument",
            ),
            "functional_id": self.functional_id,
            "conditioning": [
                {"variable_id": variable, "value_id": value}
                for variable, value in self.conditioning
            ],
            "transform_id": self.transform_id,
        }


@dataclass(frozen=True, slots=True)
class LatentCause:
    """A declared dependence of a mechanism on an unobserved shared cause."""

    latent_id: str
    shared_with: tuple[str, ...]

    def to_wire(self) -> dict[str, Any]:
        """The declaration the native layer reads."""
        return {
            "latent_id": self.latent_id,
            "shared_with_variable_ids": [_name(v, "shared_with") for v in self.shared_with],
        }


@dataclass(frozen=True, slots=True)
class MechanismDeclaration:
    """One mechanism in semantic coordinates: what it produces, consumes and was fitted on.

    ``conditioning_variable_ids`` defaults to the parents' variables; a fit that conditioned on
    anything else (or not on a declared parent) is refused.
    """

    output: Quantity
    parents: tuple[Quantity, ...]
    artifact_id: str
    evidence_id: str
    latent_causes: tuple[LatentCause, ...] = ()
    conditioning_variable_ids: tuple[str, ...] | None = None

    def to_wire(self) -> dict[str, Any]:
        """The declaration the native layer reads."""
        conditioning = (
            self.conditioning_variable_ids
            if self.conditioning_variable_ids is not None
            else tuple(p.variable_id for p in self.parents)
        )
        return {
            "output": self.output.to_wire(),
            "parents": [p.to_wire() for p in self.parents],
            "conditioning_variable_ids": list(conditioning),
            "artifact_id": self.artifact_id,
            "evidence_id": self.evidence_id,
            "latent_causes": [c.to_wire() for c in self.latent_causes],
        }


@dataclass(frozen=True, slots=True)
class ParameterTerm:
    """A parameter of one linear-Gaussian mechanism: its intercept or a parent's coefficient."""

    kind: str
    parent: str | None = None

    @classmethod
    def intercept(cls) -> ParameterTerm:
        """The intercept."""
        return cls(kind="intercept")

    @classmethod
    def coefficient(cls, parent: str) -> ParameterTerm:
        """The coefficient on the named parent."""
        return cls(kind="coefficient", parent=_name(parent, "parent"))

    def to_wire(self) -> dict[str, Any]:
        """The declaration the native layer reads."""
        if self.kind == "intercept":
            return {"kind": "intercept"}
        return {"kind": "coefficient", "parent": self.parent}


@dataclass(frozen=True, slots=True)
class ParameterId:
    """A parameter of the whole composition: the owning variable and its term."""

    mechanism: str
    term: ParameterTerm

    def to_wire(self) -> dict[str, Any]:
        """The declaration the native layer reads."""
        return {"mechanism": self.mechanism, "term": self.term.to_wire()}


@dataclass(frozen=True, slots=True)
class MechanismUncertainty:
    """Marginal parameter covariance one fit reported for its own mechanism.

    ``covariance`` is row-major ``k x k`` over ``parameters``, in that order. ``meaning`` is
    ``"posterior"`` or ``"sampling_distribution"``; laws of different meanings are never combined.
    """

    parameters: tuple[ParameterTerm, ...]
    covariance: tuple[float, ...]
    meaning: str = "sampling_distribution"

    def to_wire(self) -> dict[str, Any]:
        """The declaration the native layer reads."""
        if self.meaning not in _MEANINGS:
            raise CausalValueError(f"meaning must be one of {', '.join(_MEANINGS)}")
        return {
            "meaning": self.meaning,
            "parameters": [p.to_wire() for p in self.parameters],
            "covariance": [_number(v, "covariance entry") for v in self.covariance],
        }


def _domain(domain: Any) -> dict[str, Any]:
    """A domain wire: ``None`` is the whole line, ``(lo, hi)`` an interval, a list its levels."""
    if domain is None:
        return {"kind": "real"}
    if (
        isinstance(domain, tuple)
        and len(domain) == 2
        and all(d is None or isinstance(d, (int, float)) for d in domain)
    ):
        lo, hi = domain
        return {
            "kind": "interval",
            "lo": None if lo is None or lo == -math.inf else _number(lo, "lower bound"),
            "hi": None if hi is None or hi == math.inf else _number(hi, "upper bound"),
        }
    if isinstance(domain, (list, tuple)) and all(isinstance(level, str) for level in domain):
        return {"kind": "levels", "levels": list(domain)}
    raise CausalTypeError(
        "a domain is None (the whole line), a (lo, hi) pair, or a list of level names"
    )


@dataclass(frozen=True, slots=True)
class Mechanism:
    """One mechanism of a composition: its coordinates, its numbers and what the fit reported.

    Build one with :meth:`declared_linear`, :meth:`declared_table` or :meth:`fitted_linear`.
    ``output_support`` and ``parent_domains`` take ``None`` (the whole line), a ``(lo, hi)`` pair or
    a list of level names; an unlisted parent defaults to the whole line.
    """

    declaration: MechanismDeclaration
    source: Mapping[str, Any]
    output_support: Any = None
    parent_domains: Mapping[str, Any] = field(default_factory=dict)
    uncertainty: MechanismUncertainty | None = None
    latent_loadings: Mapping[str, float] = field(default_factory=dict)

    @classmethod
    def declared_linear(
        cls,
        declaration: MechanismDeclaration,
        *,
        intercept: float,
        sigma: float,
        coefficients: Mapping[str, float],
        output_support: Any = None,
        parent_domains: Mapping[str, Any] | None = None,
        uncertainty: MechanismUncertainty | None = None,
        latent_loadings: Mapping[str, float] | None = None,
    ) -> Mechanism:
        """A linear-Gaussian mechanism whose coefficients the analyst wrote down."""
        return cls(
            declaration=declaration,
            source={
                "kind": "declared_linear",
                "intercept": _number(intercept, "intercept"),
                "sigma": _number(sigma, "sigma"),
                "coefficients": [
                    [name, _number(c, "coefficient")] for name, c in coefficients.items()
                ],
            },
            output_support=output_support,
            parent_domains=dict(parent_domains or {}),
            uncertainty=uncertainty,
            latent_loadings=dict(latent_loadings or {}),
        )

    @classmethod
    def declared_table(
        cls,
        declaration: MechanismDeclaration,
        *,
        levels: Sequence[str],
        rows: Sequence[tuple[Mapping[str, str], Mapping[str, float]]],
        parent_domains: Mapping[str, Any] | None = None,
    ) -> Mechanism:
        """A conditional probability table the analyst wrote down.

        ``rows`` pairs one parent configuration (parent to level) with the probability of every
        level of the produced variable. The output support is ``levels``; an unlisted parent
        defaults to this table's own levels (declare ``parent_domains`` when a parent has other
        levels).
        """
        domains = {p.variable_id: list(levels) for p in declaration.parents}
        domains.update(parent_domains or {})
        return cls(
            declaration=declaration,
            source={
                "kind": "declared_table",
                "levels": list(levels),
                "rows": [
                    {
                        "parent_levels": [[p, lvl] for p, lvl in config.items()],
                        "probabilities": [
                            [lvl, _number(p, "probability")] for lvl, p in probabilities.items()
                        ],
                    }
                    for config, probabilities in rows
                ],
            },
            output_support=list(levels),
            parent_domains=domains,
        )

    @classmethod
    def fitted_linear(
        cls,
        declaration: MechanismDeclaration,
        model: Any,
        *,
        variable: str | None = None,
        output_support: Any = None,
        parent_domains: Mapping[str, Any] | None = None,
        uncertainty: MechanismUncertainty | None = None,
        latent_loadings: Mapping[str, float] | None = None,
    ) -> Mechanism:
        """The numbers of one variable's plain linear-Gaussian mechanism of a fitted model.

        ``model`` is a fitted graphical causal model (``antecedent.fit_gcm``). A basis expansion,
        discrete, constant, vacant or pending mechanism is refused. A fit carries no parameter
        covariance and no fit identity: ``declaration`` supplies the identities of the fit and its
        evidence, and ``uncertainty`` a reported covariance, if any.
        """
        target = variable if variable is not None else declaration.output.variable_id
        source = json.loads(
            _unwrap(
                _native.mechanism_composition_fitted_source(
                    native_gcm(model), _name(target, "variable")
                )
            )
        )
        return cls(
            declaration=declaration,
            source=source,
            output_support=output_support,
            parent_domains=dict(parent_domains or {}),
            uncertainty=uncertainty,
            latent_loadings=dict(latent_loadings or {}),
        )

    @property
    def variable(self) -> str:
        """The variable this mechanism produces."""
        return self.declaration.output.variable_id

    @property
    def declared(self) -> bool:
        """Whether the numbers were written down by the analyst rather than fitted."""
        return str(self.source.get("kind", "")).startswith("declared")

    def to_wire(self) -> dict[str, Any]:
        """The declaration the native layer reads."""
        parents = [p.variable_id for p in self.declaration.parents]
        return {
            "declaration": self.declaration.to_wire(),
            "source": dict(self.source),
            "output_support": _domain(self.output_support),
            "parent_domains": [
                [parent, _domain(self.parent_domains.get(parent))] for parent in parents
            ],
            "uncertainty": None if self.uncertainty is None else self.uncertainty.to_wire(),
            "latent_loadings": [
                [latent, _number(load, "loading")] for latent, load in self.latent_loadings.items()
            ],
        }


@dataclass(frozen=True, slots=True)
class MechanismRefit:
    """The mechanism a refit replaces: new numbers of the same family and parents."""

    variable_id: str
    source: Mapping[str, Any]
    artifact_id: str
    evidence_id: str
    uncertainty: MechanismUncertainty | None = None

    @classmethod
    def declared_linear(
        cls,
        variable_id: str,
        *,
        intercept: float,
        sigma: float,
        coefficients: Mapping[str, float],
        artifact_id: str,
        evidence_id: str,
        uncertainty: MechanismUncertainty | None = None,
    ) -> MechanismRefit:
        """A refit whose numbers the analyst wrote down."""
        return cls(
            variable_id=variable_id,
            source={
                "kind": "declared_linear",
                "intercept": _number(intercept, "intercept"),
                "sigma": _number(sigma, "sigma"),
                "coefficients": [[n, _number(c, "coefficient")] for n, c in coefficients.items()],
            },
            artifact_id=artifact_id,
            evidence_id=evidence_id,
            uncertainty=uncertainty,
        )

    @classmethod
    def fitted_linear(
        cls,
        variable_id: str,
        model: Any,
        *,
        artifact_id: str,
        evidence_id: str,
        uncertainty: MechanismUncertainty | None = None,
    ) -> MechanismRefit:
        """A refit read from a model fitted on new data."""
        source = json.loads(
            _unwrap(
                _native.mechanism_composition_fitted_source(
                    native_gcm(model), _name(variable_id, "variable")
                )
            )
        )
        return cls(
            variable_id=variable_id,
            source=source,
            artifact_id=artifact_id,
            evidence_id=evidence_id,
            uncertainty=uncertainty,
        )

    def to_wire(self) -> dict[str, Any]:
        """The declaration the native layer reads."""
        return {
            "variable_id": self.variable_id,
            "source": dict(self.source),
            "artifact_id": self.artifact_id,
            "evidence_id": self.evidence_id,
            "uncertainty": None if self.uncertainty is None else self.uncertainty.to_wire(),
        }


@dataclass(frozen=True, slots=True)
class CrossCovarianceBlock:
    """Covariance between the parameters of two mechanisms (row-major, first by second).

    A zero block is a declared independence for that pair of mechanisms.
    """

    first: str
    second: str
    first_parameters: tuple[ParameterTerm, ...]
    second_parameters: tuple[ParameterTerm, ...]
    values: tuple[float, ...]

    def to_wire(self) -> dict[str, Any]:
        """The declaration the native layer reads."""
        return {
            "first": self.first,
            "second": self.second,
            "first_parameters": [p.to_wire() for p in self.first_parameters],
            "second_parameters": [p.to_wire() for p in self.second_parameters],
            "values": [_number(v, "covariance entry") for v in self.values],
        }


@dataclass(frozen=True, slots=True)
class Dependence:
    """What is declared about the dependence between the mechanisms' estimates.

    Build one with :meth:`unknown` (the default: no uncertainty claim over several mechanisms),
    :meth:`independent_by_design`, :meth:`aligned_joint_law` or :meth:`cross_covariance`.
    """

    wire: Mapping[str, Any]

    @classmethod
    def unknown(cls) -> Dependence:
        """Nothing declared; point execution is unaffected, uncertainty claims are refused."""
        return cls({"kind": "unknown"})

    @classmethod
    def independent_by_design(cls, justification: str) -> Dependence:
        """Estimated from disjoint, independently generated evidence (never a default)."""
        return cls({"kind": "independent_by_design", "justification": justification})

    @classmethod
    def aligned_joint_law(
        cls,
        law_id: str,
        parameters: Sequence[ParameterId],
        covariance: Sequence[float],
        *,
        meaning: str = "sampling_distribution",
    ) -> Dependence:
        """One aligned joint parameter law: row-major covariance over ``parameters``."""
        if meaning not in _MEANINGS:
            raise CausalValueError(f"meaning must be one of {', '.join(_MEANINGS)}")
        return cls(
            {
                "kind": "aligned_joint_law",
                "law_id": _name(law_id, "law_id"),
                "meaning": meaning,
                "parameters": [p.to_wire() for p in parameters],
                "covariance": [_number(v, "covariance entry") for v in covariance],
            }
        )

    @classmethod
    def cross_covariance(cls, blocks: Sequence[CrossCovarianceBlock]) -> Dependence:
        """Explicit covariance between every pair of mechanisms."""
        return cls({"kind": "cross_covariance", "blocks": [b.to_wire() for b in blocks]})

    def to_wire(self) -> dict[str, Any]:
        """The declaration the native layer reads."""
        return dict(self.wire)


@dataclass(frozen=True, slots=True)
class Admission:
    """A declared admission of one mechanism fitted in another population or regime.

    ``coordinate`` is ``"population"`` or ``"regime"``. State either ``selection_targets`` (the
    variables with a selection arrow; the mechanism of a variable without one is invariant) or an
    ``assumption_id`` (an explicit invariance assumption the analyst owns).
    """

    variable_id: str
    coordinate: str
    source_id: str
    target_id: str
    selection_targets: tuple[str, ...] | None = None
    assumption_id: str | None = None

    def to_wire(self) -> dict[str, Any]:
        """The declaration the native layer reads."""
        if (self.selection_targets is None) == (self.assumption_id is None):
            raise CausalValueError(
                "an admission states exactly one of selection_targets and assumption_id"
            )
        if self.selection_targets is not None:
            basis: dict[str, Any] = {
                "kind": "selection_diagram",
                "targets": list(self.selection_targets),
            }
        else:
            basis = {"kind": "declared_invariance", "assumption_id": self.assumption_id}
        return {
            "variable_id": self.variable_id,
            "coordinate": self.coordinate,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "basis": basis,
        }


@dataclass(frozen=True, slots=True)
class MeanContrast:
    """``sum_k weight_k * E[outcome | do(arm_k)]`` over hard interventions of linear-Gaussian laws."""

    outcome: str
    arms: tuple[tuple[float, tuple[tuple[str, float], ...]], ...]

    @classmethod
    def ate(cls, outcome: str, treatment: str, high: float, low: float) -> MeanContrast:
        """``E[outcome | do(treatment = high)] - E[outcome | do(treatment = low)]``."""
        return cls(
            outcome=outcome,
            arms=((1.0, ((treatment, high),)), (-1.0, ((treatment, low),))),
        )

    @classmethod
    def mean_at(cls, outcome: str, interventions: Mapping[str, float]) -> MeanContrast:
        """The interventional mean ``E[outcome | do(interventions)]`` as a one-arm contrast."""
        return cls(outcome=outcome, arms=((1.0, tuple(interventions.items())),))

    def to_wire(self) -> dict[str, Any]:
        """The declaration the native layer reads."""
        return {
            "outcome": _name(self.outcome, "outcome"),
            "arms": [
                {
                    "weight": _number(weight, "weight"),
                    "interventions": [
                        [_name(v, "variable"), _number(x, "value")] for v, x in interventions
                    ],
                }
                for weight, interventions in self.arms
            ],
        }


# --------------------------------------------------------------------------- results


@dataclass(repr=False, frozen=True, slots=True, eq=False)
class JointGaussianLaw(ScientificResult):
    """Joint Gaussian law of all variables under one regime."""

    variables: tuple[str, ...]
    mean: np.ndarray
    covariance: np.ndarray

    @property
    def answer(self) -> Answer:
        return Answer("distribution", detail="joint_gaussian_law")

    def inspect(self) -> Any:
        from .results._report import SlotModel

        return ScientificResult.inspect(self).model_copy(
            update={
                "uncertainty": SlotModel(
                    available=True,
                    reason="conditional_on_declared_composition",
                    summary="joint Gaussian mean and covariance",
                    payload={"variables": len(self.variables)},
                )
            }
        )

    def _summary_fields(self) -> dict[str, Any]:
        # A bounded display: the first few declared means, never the whole matrix.
        shown: dict[str, Any] = {"n_variables": len(self.variables)}
        for index, name in enumerate(self.variables[:6]):
            shown[f"mean[{name}]"] = float(self.mean[index])
        return shown

    def _index(self, variable: str) -> int:
        try:
            return self.variables.index(variable)
        except ValueError as error:
            raise KeyError(variable) from error

    def mean_of(self, variable: str) -> float:
        """Mean of one variable."""
        return float(self.mean[self._index(variable)])

    def variance_of(self, variable: str) -> float:
        """Variance of one variable."""
        i = self._index(variable)
        return float(self.covariance[i, i])

    def covariance_of(self, first: str, second: str) -> float:
        """Covariance of two variables."""
        return float(self.covariance[self._index(first), self._index(second)])


@dataclass(repr=False, frozen=True, slots=True)
class FunctionalUncertainty(ScientificResult):
    """Parameter-uncertainty propagation for a contrast of interventional means.

    The variance of the functional induced by the declared parameter covariance alone, by the
    first-order delta method. It is not a sampling interval and carries no coverage claim.
    """

    point: float
    variance: float
    standard_error: float
    gradient: tuple[tuple[ParameterId, float], ...]
    meaning: str | None
    dependence: str
    method: str

    @property
    def answer(self) -> Answer:
        return Answer("point", value=self.point, detail="first_order_parameter_uncertainty")

    def inspect(self) -> Any:
        from .results._report import SlotModel

        payload = {
            "variance": self.variance,
            "standard_error": self.standard_error,
            "meaning": self.meaning,
            "dependence": self.dependence,
            "method": self.method,
        }
        scope = getattr(self, "scope", None)
        if scope is not None:
            payload["scope"] = scope
        return ScientificResult.inspect(self).model_copy(
            update={
                "uncertainty": SlotModel(
                    available=True,
                    reason="uncalibrated_no_interval_claim",
                    summary=self.method,
                    payload=payload,
                )
            }
        )


@dataclass(repr=False, frozen=True, slots=True, eq=False)
class FiniteLaw(ScientificResult):
    """The exact joint law of a finite composition under one regime.

    ``probabilities`` lists every joint state with the last variable varying fastest.
    """

    variables: tuple[str, ...]
    levels: tuple[tuple[str, ...], ...]
    probabilities: np.ndarray

    @property
    def answer(self) -> Answer:
        return Answer("distribution", detail="finite_joint_law")

    def probability(self, event: Mapping[str, str]) -> float:
        """Probability that every named variable takes its named level."""
        positions: list[tuple[int, int]] = []
        for variable, level in event.items():
            try:
                i = self.variables.index(variable)
                positions.append((i, self.levels[i].index(level)))
            except ValueError as error:
                raise KeyError(f"unknown variable or level in {variable}={level}") from error
        shape = tuple(len(levels) for levels in self.levels)
        cube = self.probabilities.reshape(shape)
        index: list[Any] = [slice(None)] * len(shape)
        for i, level_index in positions:
            index[i] = level_index
        return float(cube[tuple(index)].sum())

    def marginal(self, variable: str) -> dict[str, float]:
        """Marginal ``level -> probability`` of one variable."""
        i = self.variables.index(variable)
        return {level: self.probability({variable: level}) for level in self.levels[i]}


@dataclass(repr=False, frozen=True, slots=True)
class GaussianLawResult(ScientificResult):
    """Mean and variance of one outcome of a linear-Gaussian composition under intervention."""

    mean: float
    variance: float


@dataclass(repr=False, frozen=True, slots=True)
class GaussianFunctionalResult(ScientificResult):
    """Point, parameter-covariance variance, dependence label and method of a contrast."""

    point: float
    variance: float
    dependence: str
    method: str


@dataclass(repr=False, frozen=True, slots=True)
class FiniteDistributionResult(ScientificResult):
    """``(level, probability)`` of one outcome of a finite composition under intervention."""

    probabilities: tuple[tuple[str, float], ...]


QueryResult = GaussianLawResult | GaussianFunctionalResult | FiniteDistributionResult


def _result(wire: Mapping[str, Any]) -> QueryResult:
    kind = wire["kind"]
    if kind == "gaussian_law":
        return GaussianLawResult(mean=float(wire["mean"]), variance=float(wire["variance"]))
    if kind == "gaussian_functional":
        return GaussianFunctionalResult(
            point=float(wire["point"]),
            variance=float(wire["variance"]),
            dependence=str(wire["dependence"]),
            method=str(wire["method"]),
        )
    if kind == "finite_distribution":
        return FiniteDistributionResult(
            probabilities=tuple((str(level), float(p)) for level, p in wire["probabilities"])
        )
    raise CausalValueError(f"unknown composition result {kind!r}")


@dataclass(frozen=True, slots=True)
class CompositionExport:
    """An exported composition: the artifact, the premises digest and the stored result.

    A consumer that retains ``premises_digest`` independently refuses a consistently resealed
    edit.
    """

    artifact: bytes
    premises_digest: str
    result: QueryResult


@dataclass(repr=False, frozen=True, slots=True)
class ConsumedComposition(ScientificResult):
    """A consumed artifact: the verified identity, the family and the replayed result."""

    _mean_binding: Any = field(default=None, init=False, repr=False, compare=False)

    premises_digest: str
    family: str
    result: QueryResult
    _scope: Mapping[str, Any] | None = field(default=None, repr=False, compare=False)


# --------------------------------------------------------------------------- the prepared object


@dataclass(repr=False, frozen=True, slots=True, eq=False)
class PreparedMechanismComposition(ScientificResult):
    """A composition that compiled: one executable joint model with the plan it compiled under.

    Every method that changes it returns a new object and leaves this one untouched, whether it
    succeeds or refuses.
    """

    _mean_binding: Any = field(default=None, init=False, repr=False, compare=False)

    plan_identity: str
    family: str
    variables: tuple[str, ...]
    declared_mechanisms: tuple[str, ...]
    _handle: Any = field(repr=False, compare=False)
    _scope: Mapping[str, Any] | None = field(default=None, repr=False, compare=False)

    def interval(self) -> None:
        """Always raises ``cell_not_licensed``: the variance carries no measured coverage."""
        raise_refusal(self._handle.interval(), MechanismCompositionRefusal)

    @outcome_operation
    def intervention_law(
        self, interventions: Mapping[str, float] | None = None
    ) -> JointGaussianLaw:
        """The joint Gaussian law under ``do(interventions)``; no interventions is the
        observational law of the composition."""
        wire = {"interventions": _pairs(interventions or {}, "variable")}
        for pair in wire["interventions"]:
            pair[1] = _number(pair[1], "intervention value")
        law = json.loads(_unwrap(self._handle.intervention_law(_dumps(wire))))
        variables = tuple(law["variables"])
        n = len(variables)
        return JointGaussianLaw(
            variables=variables,
            mean=np.asarray(law["mean"], dtype=np.float64),
            covariance=np.asarray(law["covariance"], dtype=np.float64).reshape(n, n),
        )

    @outcome_operation
    def mean_contrast(self, contrast: MeanContrast) -> float:
        """``sum_k weight_k * E[outcome | do(arm_k)]``, exact given the declared mechanisms."""
        return float(_unwrap(self._handle.mean_contrast(_dumps(contrast.to_wire()))))

    @outcome_operation
    def mean_contrast_uncertainty(self, contrast: MeanContrast) -> FunctionalUncertainty:
        """The first-order delta-method variance of a contrast, from the declared parameter law.

        Raises ``joint_law_required`` when the dependence between more than one mechanism is
        unknown or a cross-covariance pair is missing.
        """
        wire = json.loads(
            _unwrap(self._handle.mean_contrast_uncertainty(_dumps(contrast.to_wire())))
        )
        return FunctionalUncertainty(
            point=float(wire["point"]),
            variance=float(wire["variance"]),
            standard_error=float(wire["standard_error"]),
            gradient=tuple(
                (
                    ParameterId(
                        mechanism=g["mechanism"],
                        term=ParameterTerm(kind=g["term"]["kind"], parent=g["term"].get("parent")),
                    ),
                    float(g["gradient"]),
                )
                for g in wire["gradient"]
            ),
            meaning=wire["meaning"],
            dependence=str(wire["dependence"]),
            method=str(wire["method"]),
        )

    @outcome_operation
    def sample(
        self,
        interventions: Mapping[str, float] | None = None,
        *,
        n_rows: int,
        execution: Execution | None = None,
    ) -> dict[str, np.ndarray]:
        """Draw ``n_rows`` units from the (possibly intervened) joint model.

        The noise of each variable is keyed by its name, so a variable's draws do not depend on
        the other variables or their order. The seed of ``execution``
        fixes the table: the same seed gives the same table.
        """
        execution = resolve_execution(execution)
        wire = {"interventions": _pairs(interventions or {}, "variable")}
        for pair in wire["interventions"]:
            pair[1] = _number(pair[1], "intervention value")
        variables, columns = _unwrap(
            self._handle.sample(
                _dumps(wire),
                _count(
                    n_rows, "n_rows", minimum=0, maximum=2**64 - 1, reason_code="invalid_argument"
                ),
                execution.seed,
                cancel=execution.cancel,
            )
        )
        return {
            name: np.asarray(column, dtype=np.float64)
            for name, column in zip(variables, columns, strict=True)
        }

    @outcome_operation
    def finite_law(
        self, interventions: Mapping[str, str] | None = None, *, execution: Execution | None = None
    ) -> FiniteLaw:
        """The exact finite joint law under ``do(interventions)`` by truncated factorization."""
        execution = resolve_execution(execution)
        wire = {"interventions": _pairs(interventions or {}, "variable")}
        law = json.loads(_unwrap(self._handle.finite_law(_dumps(wire), cancel=execution.cancel)))
        return FiniteLaw(
            variables=tuple(law["variables"]),
            levels=tuple(tuple(levels) for levels in law["levels"]),
            probabilities=np.asarray(law["probabilities"], dtype=np.float64),
        )

    def refresh_mechanisms(
        self,
        refits: Sequence[MechanismRefit],
        dependence: Dependence | None = None,
        *,
        execution: Execution | None = None,
    ) -> PreparedMechanismComposition:
        """Replace the named mechanisms with their refits and recompile.

        Keeps the plan; recomputes the numbers, the provenance and the joint model; refuses a
        refit of another family or parents. ``dependence`` redeclares the dependence law (a joint
        law or cross-covariance describes the estimates it was declared for); omitted, it keeps an
        earlier declaration only when still valid.
        """
        execution = resolve_execution(execution)
        handle = _unwrap(
            self._handle.refresh_mechanisms(
                _dumps([r.to_wire() for r in refits]),
                None if dependence is None else _dumps(dependence.to_wire()),
                cancel=execution.cancel,
            )
        )
        return _prepared(handle, self._scope)

    def transport_to(
        self, transport: MechanismTransport, *, execution: Execution | None = None
    ) -> TransportedMechanismComposition:
        """Carry this composition's fitted mechanisms into a target population's model.

        ``transport`` declares the selection diagram (which variables each selection node points
        into), the mechanisms natively refitted in the target, any explicit invariances and the
        dependence between the estimates. A mechanism a selection node points into (or whose latent
        cause it points into) is never carried, whatever invariance is declared for it: a selection
        arrow dominates an invariance; refit it in the target. A refit always replaces the carried
        mechanism and is never averaged with it. Returns a new object and leaves this one untouched,
        whether it succeeds or refuses.

        Raises:
            MechanismCompositionRefusal: every unmet premise, each with a registered code and a
                ``mechanism_transport.*`` (or ``mechanism_composition.*``) detail.
        """
        execution = resolve_execution(execution)
        if not isinstance(transport, MechanismTransport):
            raise CausalTypeError("transport must be a MechanismTransport")
        handle, receipt = _unwrap(
            self._handle.transport_to(_dumps(transport.to_wire()), cancel=execution.cancel)
        )
        return _transported(handle, receipt, self._scope)

    @outcome_operation
    def verify_transport(
        self, transport: MechanismTransport, *, expected_identity: str
    ) -> MechanismTransportReceipt:
        """Re-derive the transport from this composition and ``transport`` and require the identity.

        Raises ``mechanism_transport.transport_identity_mismatch`` when the recomputed identity is
        not ``expected_identity``.
        """
        if not isinstance(transport, MechanismTransport):
            raise CausalTypeError("transport must be a MechanismTransport")
        text = _unwrap(
            self._handle.verify_transport(
                _dumps(transport.to_wire()), _name(expected_identity, "expected_identity")
            )
        )
        return _receipt_of(json.loads(text))

    def derive_inverse_source(
        self,
        actions: Sequence[Any],
        *,
        transport: Any = None,
        accepts_transported: bool = False,
        execution: Execution | None = None,
    ) -> Any:
        """The composition as a forward source of an inverse query, one replayed query per action.

        See :func:`antecedent.inverse_sources.derive_composed_source`.
        """
        from .inverse_sources import derive_composed_source

        return derive_composed_source(
            self,
            actions,
            transport=transport,
            accepts_transported=accepts_transported,
            execution=execution,
        )

    def discretize(
        self,
        levels: Mapping[str, int] | int,
        *,
        method: str = "equal_probability_quantile",
        execution: Execution | None = None,
    ) -> Any:
        """Discretize a linear-Gaussian composition into an executable finite model.

        See :func:`antecedent.discretized_composition.discretize`.
        """
        execution = resolve_execution(execution)
        from .discretized_composition import discretize

        return discretize(self, levels, method=method, execution=execution)

    @artifact_export
    def export(
        self,
        query: MeanContrast | Mapping[str, object],
        *,
        outcome: str | None = None,
        execution: Execution | None = None,
    ) -> ExportedArtifact[ConsumedComposition]:
        """Seal the premises and one query into a ``mechanism_composition_v1`` artifact.

        ``query`` is a :class:`MeanContrast` (a contrast of interventional means with parameter
        uncertainty), or a mapping of hard interventions with ``outcome=`` (the law of one outcome
        of the linear-Gaussian composition, or, with level names, the distribution of one outcome
        of a finite composition). An uncertainty claim without a declared joint law is refused.
        """
        execution = resolve_execution(execution)
        wire = _query_wire(self.family, query, outcome)
        artifact, digest, result = _unwrap(
            self._handle.export(_dumps(wire), cancel=execution.cancel)
        )
        return artifact_from_receipt(
            self,
            CompositionExport(
                artifact=bytes(artifact),
                premises_digest=digest,
                result=_result(json.loads(result)),
            ),
        )


def _query_wire(
    family: str, query: MeanContrast | Mapping[str, Any], outcome: str | None
) -> dict[str, Any]:
    """The composition query the native layer reads: a contrast, a Gaussian intervention law or
    a finite intervention distribution."""
    if isinstance(query, MeanContrast):
        return {
            "gaussian_mean_contrast": {
                "outcome": query.outcome,
                "arms": [
                    [w, [[v, x] for v, x in interventions]] for w, interventions in query.arms
                ],
            }
        }
    if isinstance(query, Mapping):
        target = _name(outcome, "outcome")
        if family == "tabular":
            return {
                "finite_intervention_distribution": {
                    "interventions": [[v, str(level)] for v, level in query.items()],
                    "outcome": target,
                }
            }
        return {
            "gaussian_intervention_law": {
                "interventions": [[v, _number(x, "value")] for v, x in query.items()],
                "outcome": target,
            }
        }
    raise CausalTypeError("query must be a MeanContrast or a mapping of interventions")


@bind_mean_reading
def _prepared(handle: Any, scope: Mapping[str, Any] | None = None) -> PreparedMechanismComposition:
    return PreparedMechanismComposition(
        plan_identity=handle.plan_identity(),
        family=handle.family(),
        variables=tuple(handle.variables()),
        declared_mechanisms=tuple(handle.declared_mechanisms()),
        _handle=handle,
        _scope=scope,
    )


def compose_mechanisms(
    mechanisms: Sequence[Mechanism],
    *,
    target_population_id: str,
    target_regime_id: str = "observational",
    edges: Sequence[tuple[str, str]] | None = None,
    admissions: Sequence[Admission] = (),
    dependence: Dependence | None = None,
    admit_declared_mechanisms: bool = False,
    execution: Execution | None = None,
) -> PreparedMechanismComposition:
    """Prepare a declared composition of independently fitted mechanisms.

    ``edges`` is the declared graph as ``(parent, child)`` pairs; omitted, it is read from each
    mechanism's declared parents. Before any number is formed the semantic coordinates of every
    mechanism are checked (a parent must be a realization of the same quantity its producer yields;
    a mechanism from another population or regime needs a declared admission), and the result is
    compiled under the model-level coherence checker (acyclicity, parents equal conditioning set,
    support overlap, shared latents, provenance, aligned uncertainty).

    Raises:
        MechanismCompositionRefusal: every defect found, each with a registered code, a namespaced
            detail and the declaration that would license the composition.
    """
    execution = resolve_execution(execution)
    listed = list(mechanisms)
    if not all(isinstance(m, Mechanism) for m in listed):
        raise CausalTypeError("mechanisms must be Mechanism objects")
    declared_edges = (
        [(p.variable_id, m.variable) for m in listed for p in m.declaration.parents]
        if edges is None
        else [(_name(p, "edge parent"), _name(c, "edge child")) for p, c in edges]
    )
    wire = {
        "target_population_id": _name(target_population_id, "target_population_id"),
        "target_regime_id": _name(target_regime_id, "target_regime_id"),
        "mechanisms": [m.to_wire() for m in listed],
        "edges": [list(edge) for edge in declared_edges],
        "admissions": [a.to_wire() for a in admissions],
        "dependence": (dependence or Dependence.unknown()).to_wire(),
        "admit_declared_mechanisms": bool(admit_declared_mechanisms),
    }
    handle = _unwrap(_native.mechanism_composition_prepare(_dumps(wire), cancel=execution.cancel))
    return _prepared(
        handle,
        {
            "population": target_population_id,
            "units": {m.variable: m.declaration.output.units for m in listed},
        },
    )


@outcome_operation
@bind_mean_reading
@artifact_consumer(
    proof="mechanism_composition.consume_mechanism_composition_artifact: independent semantic consumer checks declarations, identities and stored result; recomputes the scientific result",
    replayable=True,
)
def consume_mechanism_composition_artifact(
    artifact: bytes,
    expected_identity: str | None = None,
    *,
    execution: Execution | None = None,
) -> ConsumedComposition:
    """Independently consume a composition artifact.

    Recomputes the premises digest, recompiles the premises under the coherence checker and replays
    the query to identical result bits. ``expected_identity`` is the premises digest the consumer
    holds independently of the bytes; with it a consistently resealed edit is refused.

    Raises:
        MechanismCompositionRefusal: ``mechanism_composition.premises_mismatch`` for a digest that
            is not the stored or the retained one, ``mechanism_composition.result_mismatch`` for a
            stored result that does not replay, the structured defects of premises that no longer
            compose, and ``mechanism_composition.artifact_invalid`` for corrupt or unsupported
            bytes.
    """
    execution = resolve_execution(execution)
    artifact = container_bytes(artifact)
    if not isinstance(artifact, (bytes, bytearray, memoryview)):
        raise CausalTypeError("artifact must be bytes")
    wire = json.loads(
        _unwrap(
            _native.mechanism_composition_consume(
                bytes(artifact), expected_identity, cancel=execution.cancel
            )
        )
    )
    from ._artifact_protocol import inspect_artifact

    body = inspect_artifact(bytes(artifact)).metadata["body"]
    premises = body["premises"]
    return ConsumedComposition(
        _scope={
            "population": premises["target_population"],
            "units": {m["variable"]: m["output_units"] for m in premises["mechanisms"]},
        },
        premises_digest=wire["premises_digest"],
        family=wire["family"],
        result=_result(wire["result"]),
    )


# --------------------------------------------------------------------------- transport


@dataclass(frozen=True, slots=True)
class SelectionNode:
    """A selection node of a selection diagram and the variables it points into."""

    name: str
    points_into: tuple[str, ...]

    def __post_init__(self) -> None:
        _name(self.name, "selection node name")
        if isinstance(self.points_into, str):
            raise CausalTypeError("points_into must be an iterable of variable names")
        object.__setattr__(
            self, "points_into", tuple(_name(v, "variable") for v in self.points_into)
        )

    def to_wire(self) -> dict[str, Any]:
        return {"name": self.name, "points_into": list(self.points_into)}


@dataclass(frozen=True, slots=True)
class TransportInvariance:
    """An explicit assumption that one mechanism is invariant across a context coordinate.

    Required for a carried mechanism that loads on a shared latent cause, and to carry across
    regimes. A selection arrow into the variable still dominates it.
    """

    variable: str
    coordinate: str
    source: str
    target: str
    assumption_id: str

    def __post_init__(self) -> None:
        if self.coordinate not in ("population", "regime"):
            raise CausalValueError("coordinate must be 'population' or 'regime'")
        for what in ("variable", "source", "target", "assumption_id"):
            _name(getattr(self, what), what)

    def to_wire(self) -> dict[str, Any]:
        return {
            "variable": self.variable,
            "coordinate": self.coordinate,
            "source": self.source,
            "target": self.target,
            "assumption_id": self.assumption_id,
        }


@dataclass(frozen=True, slots=True)
class MechanismTransport:
    """What a transport of a prepared composition to a target population declares.

    ``selection_nodes`` is the selection diagram from the composition's population to
    ``target_population_id``. ``target_refits`` are mechanisms natively refitted in the target
    (:class:`MechanismRefit`); each replaces the carried mechanism of its variable.
    """

    target_population_id: str
    selection_nodes: tuple[SelectionNode, ...]
    target_regime_id: str = "observational"
    target_refits: tuple[MechanismRefit, ...] = ()
    invariances: tuple[TransportInvariance, ...] = ()
    dependence: Dependence | None = None

    def __post_init__(self) -> None:
        _name(self.target_population_id, "target_population_id")
        _name(self.target_regime_id, "target_regime_id")
        object.__setattr__(self, "selection_nodes", tuple(self.selection_nodes))
        object.__setattr__(self, "target_refits", tuple(self.target_refits))
        object.__setattr__(self, "invariances", tuple(self.invariances))

    def to_wire(self) -> dict[str, Any]:
        return {
            "target_population_id": self.target_population_id,
            "target_regime_id": self.target_regime_id,
            "selection_nodes": [n.to_wire() for n in self.selection_nodes],
            "target_refits": [r.to_wire() for r in self.target_refits],
            "invariances": [i.to_wire() for i in self.invariances],
            "dependence": (self.dependence or Dependence.unknown()).to_wire(),
        }


@dataclass(frozen=True, slots=True)
class MechanismDisposition:
    """One component of a transported composition: carried or refitted, and from where."""

    variable: str
    role: str
    fitted_population: str
    artifact_id: str
    evidence_id: str


@dataclass(frozen=True, slots=True)
class MechanismTransportReceipt:
    """The typed receipt of one transported mechanism composition.

    ``support`` is always ``extrapolative`` and ``inference`` always ``point_only``.
    ``uncertainty_scope`` says what the variance of a contrast contains: the source fits'
    parameter covariance only; the transport assumption is not a variance term.
    """

    route: str
    identity: str
    source_plan_identity: str
    source_population: str
    target_population: str
    target_regime: str
    selection_nodes: tuple[SelectionNode, ...]
    selection_targets: tuple[str, ...]
    dispositions: tuple[MechanismDisposition, ...]
    invariances: tuple[tuple[str, str], ...]
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

    def as_transport_receipt(self) -> Any:
        """The shared :class:`antecedent.transported_response.TransportReceipt` of this route."""
        from .transported_response import TransportReceipt

        return TransportReceipt(
            route="mechanism",
            source_identity=self.source_plan_identity,
            source_populations=(self.source_population,),
            target_population=self.target_population,
            transport_identities=(self.identity,),
            assumptions=self.assumptions,
            support=self.support,
            representation="point_only",
        )


def _receipt_of(wire: Mapping[str, Any]) -> MechanismTransportReceipt:
    return MechanismTransportReceipt(
        route=str(wire["route"]),
        identity=str(wire["identity"]),
        source_plan_identity=str(wire["source_plan_identity"]),
        source_population=str(wire["source_population"]),
        target_population=str(wire["target_population"]),
        target_regime=str(wire["target_regime"]),
        selection_nodes=tuple(
            SelectionNode(n["name"], tuple(n["points_into"])) for n in wire["selection_nodes"]
        ),
        selection_targets=tuple(wire["selection_targets"]),
        dispositions=tuple(
            MechanismDisposition(
                variable=d["variable"],
                role=d["role"],
                fitted_population=d["fitted_population"],
                artifact_id=d["artifact_id"],
                evidence_id=d["evidence_id"],
            )
            for d in wire["dispositions"]
        ),
        invariances=tuple((i["variable"], i["assumption_id"]) for i in wire["invariances"]),
        uncertainty_scope=str(wire["uncertainty_scope"]),
        support=str(wire["support"]),
        inference=str(wire["inference"]),
        assumptions=tuple(wire["assumptions"]),
    )


@dataclass(repr=False, frozen=True, slots=True)
class TransportedFunctionalUncertainty(FunctionalUncertainty):
    """A first-order contrast variance of a transported composition, with its scope label.

    ``scope`` states that the covariance is the source fits' for a carried mechanism and that the
    transport assumption adds no variance term. Uncalibrated; no interval follows.
    """

    scope: str


@dataclass(repr=False, frozen=True, slots=True, eq=False)
class TransportedMechanismComposition(ScientificResult):
    """An executable target-population composition with its transport receipt.

    Every component says whether it was carried from the source fit or refitted in the target.
    The numbers are exact given the mechanisms; the variance is first-order, uncalibrated and
    scoped; the interval route is closed.
    """

    identity: str
    receipt: MechanismTransportReceipt
    native_target_data: bool = False
    _handle: Any = field(default=None, repr=False, compare=False)
    _scope: Mapping[str, Any] | None = field(default=None, repr=False, compare=False)

    @property
    def transported_variables(self) -> tuple[str, ...]:
        return self.receipt.transported_variables

    @outcome_operation
    def mean_contrast(self, contrast: MeanContrast) -> float:
        """``sum_k weight_k * E[outcome | do(arm_k)]`` in the target composition."""
        return float(_unwrap(self._handle.mean_contrast(_dumps(contrast.to_wire()))))

    @outcome_operation
    def mean_contrast_uncertainty(self, contrast: MeanContrast) -> TransportedFunctionalUncertainty:
        """The first-order parameter variance of a contrast, scoped by the transport receipt.

        Raises ``joint_law_required`` when the dependence between more than one mechanism is
        unknown or a cross-covariance pair is missing.
        """
        wire = json.loads(
            _unwrap(self._handle.mean_contrast_uncertainty(_dumps(contrast.to_wire())))
        )
        return TransportedFunctionalUncertainty(
            point=float(wire["point"]),
            variance=float(wire["variance"]),
            standard_error=float(wire["standard_error"]),
            gradient=tuple(
                (
                    ParameterId(
                        mechanism=g["mechanism"],
                        term=ParameterTerm(kind=g["term"]["kind"], parent=g["term"].get("parent")),
                    ),
                    float(g["gradient"]),
                )
                for g in wire["gradient"]
            ),
            meaning=wire["meaning"],
            dependence=str(wire["dependence"]),
            method=str(wire["method"]),
            scope=str(wire["scope"]),
        )

    def interval(self) -> None:
        """Always raises ``cell_not_licensed``: no coverage includes the transport assumption."""
        raise_refusal(self._handle.interval(), MechanismCompositionRefusal)

    @artifact_export
    def export(
        self,
        query: MeanContrast | Mapping[str, object],
        *,
        outcome: str | None = None,
        execution: Execution | None = None,
    ) -> ExportedArtifact[ConsumedComposition]:
        """Seal the transported premises (with the diagram and invariances) and one query.

        The artifact is the existing ``mechanism_composition_v1`` one;
        :func:`consume_mechanism_composition_artifact` recompiles it under the coherence checker.
        """
        execution = resolve_execution(execution)
        wire = _query_wire("linear_gaussian", query, outcome)
        artifact, digest, result = _unwrap(
            self._handle.export(_dumps(wire), cancel=execution.cancel)
        )
        return artifact_from_receipt(
            self,
            CompositionExport(
                artifact=bytes(artifact),
                premises_digest=digest,
                result=_result(json.loads(result)),
            ),
        )


def _transported(
    handle: Any, receipt_json: str, scope: Mapping[str, Any] | None
) -> TransportedMechanismComposition:
    receipt = _receipt_of(json.loads(receipt_json))
    return TransportedMechanismComposition(
        identity=receipt.identity, receipt=receipt, _handle=handle, _scope=scope
    )
