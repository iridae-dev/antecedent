"""Discretize a linear-Gaussian composition into an executable finite model.

A composition of linear-Gaussian mechanisms
(:mod:`antecedent.mechanism_composition`) has a closed-form law. Some routes need a finite model
instead; this module builds one under a *declared plan* (a method and a level count for every
variable) and states how far it is from the closed form::

    from antecedent import discretized_composition as dc

    discrete = composition.discretize(8)                 # or dc.discretize(composition, ...)
    contrast = dc.DiscreteContrast.ate("Y", "X", high=7, low=0)
    value = discrete.mean_contrast(contrast)
    value.value, value.closed_form, value.error_bound   # |value - closed_form| <= error_bound

What this is, and is not:

* **A point with a computed mean-error bound.** The bound is a closed computation over the
  declared cells (``sum_k |w_k| bound_k``), not a statistical claim. The result carries no
  interval, no posterior and no calibration (``inference`` is ``point_only``;
  :meth:`PreparedDiscretizedComposition.uncertainty` always refuses ``cell_not_licensed``,
  ``discretized_scm.point_only``).
* **Parameter uncertainty is not propagated.** A parameter covariance declared on the source
  composition is not carried into the discretized value
  (``parameter_uncertainty_not_propagated``).
* **Refused.** A finite-table composition (it has no Gaussian closed form), a level count
  missing, out of range or naming an unknown variable, a joint space above the enumeration bound,
  a declared support other than the real line, a mechanism loading on a shared latent cause, and
  a degenerate marginal.

**Lifecycle.**

* :meth:`~PreparedDiscretizedComposition.refresh` discretizes a refit composition again under
  the declared plan: the plan is **kept**; the cell edges, the values, the tables and the
  identity are **recomputed**; a refit that adds or removes a variable is **refused**
  (``discretized_scm.levels_undeclared``). A refused refresh leaves the handle as it was.
* :meth:`~PreparedDiscretizedComposition.export` writes the source premises, the plan and the
  claimed numbers; :func:`consume_discretized_composition` rebuilds the discretization from the
  carried premises and requires every claimed number to equal the re-derived one bit for bit; a
  consumer that retained the identity independently refuses a self-consistent artifact of
  another composition.

Rust owns the cells, the conditional tables, the bound, the artifact and every refusal; this
module declares the plan and the contrast and presents the result. Refusals raise
:class:`DiscretizedCompositionRefusal` with a registered ``reason_code``, a ``discretized_scm.*``
(or ``mechanism_composition.*``) ``detail`` and a ``remedy``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from . import _native
from ._refusal import raise_refusal
from ._scientific import ScientificResult
from ._validate import choice, count, real, text
from .errors import CausalTypeError, StructuredRefusal
from .mechanism_composition import PreparedMechanismComposition

__all__ = [
    "ArmDiagnostic",
    "DiscreteContrast",
    "DiscretizedContrast",
    "DiscretizedCompositionRefusal",
    "DiscretizedVariable",
    "PreparedDiscretizedComposition",
    "consume_discretized_composition",
    "discretize",
]

_METHODS = ("equal_probability_quantile",)
_PARADIGMS = ("frequentist", "bayesian")


class DiscretizedCompositionRefusal(StructuredRefusal):
    """A typed refusal of the discretization route.

    ``reason_code`` is registered; ``detail`` is the namespaced ``discretized_scm.*`` literal
    (``levels_undeclared``, ``point_only``, ``identity_mismatch``, ...) or the
    ``mechanism_composition.*`` detail of a composition that cannot be discretized.
    """

    def __init__(self, refusal: Mapping[str, Any]) -> None:
        offending = refusal.get("offending")
        message = str(refusal["detail"]) + (f": {offending}" if offending else "")
        super().__init__(refusal, text=message)


def _unwrap(pair: tuple[Any, str | None]) -> Any:
    value, refusal = pair
    raise_refusal(refusal, DiscretizedCompositionRefusal)
    return value


@dataclass(frozen=True, slots=True)
class DiscreteContrast:
    """``sum_k weight_k * E[outcome | do(arm_k)]`` over interventions at *level indices*.

    Each arm is ``(weight, ((variable, level_index), ...))``; a level index addresses one cell
    of the variable's declared grid, whose value is the cell's conditional mean.
    """

    outcome: str
    arms: tuple[tuple[float, tuple[tuple[str, int], ...]], ...]

    @classmethod
    def ate(cls, outcome: str, treatment: str, high: int, low: int) -> DiscreteContrast:
        """``E[outcome | do(treatment = level high)] - E[outcome | do(treatment = level low)]``."""
        return cls(outcome, ((1.0, ((treatment, high),)), (-1.0, ((treatment, low),))))

    @classmethod
    def mean_at(cls, outcome: str, interventions: Mapping[str, int]) -> DiscreteContrast:
        """The interventional mean ``E[outcome | do(interventions)]`` as a one-arm contrast."""
        return cls(outcome, ((1.0, tuple(interventions.items())),))

    def _wire(self) -> dict[str, Any]:
        return {
            "outcome": text(self.outcome, "outcome", non_empty=True),
            "arms": [
                {
                    "weight": real(weight, "weight", finite=True),
                    "interventions": [
                        [text(v, "variable", non_empty=True), count(level, "level", minimum=0)]
                        for v, level in interventions
                    ],
                }
                for weight, interventions in self.arms
            ],
        }


@dataclass(frozen=True, slots=True)
class ArmDiagnostic:
    """One arm's discretized mean beside its closed-form value and computed bound."""

    weight: float
    discretized_mean: float
    closed_form_mean: float
    mean_error_bound: float


@dataclass(repr=False, frozen=True, slots=True)
class DiscretizedContrast(ScientificResult):
    """A discretized contrast with its closed-form comparison and computed bound.

    ``absolute_error`` is ``|value - closed_form|`` and never exceeds ``error_bound`` beyond
    floating-point rounding. ``inference`` is ``point_only`` and ``calibration`` states that no
    interval or calibration claim is made; ``error_scope`` says what the bound covers.
    """

    value: float
    closed_form: float
    absolute_error: float
    error_bound: float
    arms: tuple[ArmDiagnostic, ...]
    method: str
    inference: str
    calibration: str
    error_scope: str
    parameter_uncertainty_not_propagated: bool


@dataclass(frozen=True, slots=True)
class DiscretizedVariable:
    """One variable's declared cells.

    ``level_values`` are the values the cells carry (conditional means of the marginal),
    ``cell_edges`` the interior edges (ascending) and ``mean_error`` the largest gap between
    the discretized mechanism's table mean and its Gaussian mean at the parents' values.
    """

    variable: str
    levels: int
    level_names: tuple[str, ...]
    level_values: tuple[float, ...]
    cell_edges: tuple[float, ...]
    mean_error: float


@dataclass(repr=False, frozen=True, slots=True, eq=False)
class PreparedDiscretizedComposition(ScientificResult):
    """A composition discretized into an executable finite model.

    ``identity`` covers the source mechanisms, the method and every level count.
    ``source_plan_identity`` is the plan identity of the composition it was built from (absent
    for a consumed artifact, which carries the premises instead). ``inference`` is always
    ``point_only``.
    """

    identity: str
    method: str
    variables: tuple[DiscretizedVariable, ...]
    inference: str
    calibration: str
    error_scope: str
    source_plan_identity: str | None
    _handle: Any = field(repr=False, compare=False, default=None)

    @classmethod
    def _of(cls, handle: Any) -> PreparedDiscretizedComposition:
        info = json.loads(handle.inspect_json())
        return cls(
            identity=str(info["identity"]),
            method=str(info["method"]),
            variables=tuple(
                DiscretizedVariable(
                    variable=v["variable"],
                    levels=int(v["levels"]),
                    level_names=tuple(v["level_names"] or ()),
                    level_values=tuple(float(x) for x in (v["level_values"] or ())),
                    cell_edges=tuple(float(x) for x in (v["cell_edges"] or ())),
                    mean_error=float(v["mean_error"]),
                )
                for v in info["variables"]
            ),
            inference=str(info["inference"]),
            calibration=str(info["calibration"]),
            error_scope=str(info["error_scope"]),
            source_plan_identity=info["source_plan_identity"],
            _handle=handle,
        )

    def variable(self, name: str) -> DiscretizedVariable:
        """One variable's cells."""
        for item in self.variables:
            if item.variable == name:
                return item
        raise CausalTypeError(f"no variable {name!r} in this discretization")

    def mean_contrast(
        self, contrast: DiscreteContrast, *, cancel: Any = None
    ) -> DiscretizedContrast:
        """The contrast on the discretized model beside the closed-form Gaussian value at the
        same intervention values and the computed bound.

        Raises:
            DiscretizedCompositionRefusal: an unknown outcome, variable or level, a repeated
                intervention or cancellation.
        """
        if not isinstance(contrast, DiscreteContrast):
            raise CausalTypeError("contrast must be a DiscreteContrast")
        info = json.loads(
            _unwrap(self._handle.mean_contrast(json.dumps(contrast._wire()), cancel=cancel))
        )
        return DiscretizedContrast(
            value=float(info["value"]),
            closed_form=float(info["closed_form"]),
            absolute_error=float(info["absolute_error"]),
            error_bound=float(info["error_bound"]),
            arms=tuple(
                ArmDiagnostic(
                    float(a["weight"]),
                    float(a["discretized_mean"]),
                    float(a["closed_form_mean"]),
                    float(a["mean_error_bound"]),
                )
                for a in info["arms"]
            ),
            method=str(info["method"]),
            inference=str(info["inference"]),
            calibration=str(info["calibration"]),
            error_scope=str(info["error_scope"]),
            parameter_uncertainty_not_propagated=bool(info["parameter_uncertainty_not_propagated"]),
        )

    def uncertainty(self, paradigm: Literal["frequentist", "bayesian"] = "frequentist") -> None:
        """An interval, band, posterior or law-valued reading: always refuses
        ``cell_not_licensed`` (``discretized_scm.point_only``)."""
        choice(paradigm, "paradigm", _PARADIGMS)
        raise_refusal(self._handle.uncertainty(paradigm), DiscretizedCompositionRefusal)

    def interval(self) -> None:
        """Always refuses ``cell_not_licensed``: a discretized value is a point."""
        self.uncertainty("frequentist")

    def posterior(self) -> None:
        """Always refuses ``cell_not_licensed``: a discretized value is a point."""
        self.uncertainty("bayesian")

    def refresh(
        self, refit: PreparedMechanismComposition, *, cancel: Any = None
    ) -> PreparedDiscretizedComposition:
        """Discretize the refit composition again under the declared plan.

        A refused refresh leaves this object as it was.
        """
        if not isinstance(refit, PreparedMechanismComposition):
            raise CausalTypeError("refit must be a PreparedMechanismComposition")
        handle = _unwrap(self._handle.refresh(refit._handle, cancel=cancel))
        return PreparedDiscretizedComposition._of(handle)

    def export(self) -> bytes:
        """The source premises, the plan and the claimed numbers, for
        :func:`consume_discretized_composition`."""
        return bytes(_unwrap(self._handle.export()))


def discretize(
    composition: PreparedMechanismComposition,
    levels: Mapping[str, int] | int,
    *,
    method: str = "equal_probability_quantile",
    cancel: Any = None,
) -> PreparedDiscretizedComposition:
    """Discretize a linear-Gaussian composition under a declared plan.

    ``levels`` is the level count of every variable (a mapping naming each variable of the
    composition) or one count applied to all of them. ``method`` is
    ``equal_probability_quantile``: ``n`` cells of equal marginal probability with open outer
    cells, each carried by its conditional mean.

    Raises:
        DiscretizedCompositionRefusal: a finite-table composition, a level count missing, out of
            range or naming an unknown variable, a joint space above the enumeration bound, a
            declared support other than the real line, a mechanism loading on a shared latent
            cause, a degenerate marginal, or cancellation.
    """
    if not isinstance(composition, PreparedMechanismComposition):
        raise CausalTypeError("composition must be a PreparedMechanismComposition")
    choice(method, "method", _METHODS)
    if isinstance(levels, Mapping):
        plan = [[name, count(n, "levels", minimum=0)] for name, n in levels.items()]
    else:
        n = count(levels, "levels", minimum=0)
        plan = [[name, n] for name in composition.variables]
    handle = _unwrap(
        _native.discretized_composition_prepare(
            composition._handle,
            json.dumps({"method": method, "levels": plan}),
            cancel=cancel,
        )
    )
    return PreparedDiscretizedComposition._of(handle)


def consume_discretized_composition(
    artifact: bytes, expected_identity: str | None = None, *, cancel: Any = None
) -> PreparedDiscretizedComposition:
    """Consume an exported discretization.

    The model is rebuilt from the carried premises with the current code and every claimed
    number must equal the re-derived one. ``expected_identity`` is the identity (64 hex) the
    consumer holds for the composition it expects; with it a self-consistent artifact of another
    composition is refused.

    Raises:
        DiscretizedCompositionRefusal: malformed bytes, another version, premises that no longer
            discretize, an identity that differs from the claimed or expected one
            (``discretized_scm_artifact.identity_mismatch``) or a claimed number that differs from the
            re-derived one.
    """
    if not isinstance(artifact, (bytes, bytearray, memoryview)):
        raise CausalTypeError("artifact must be bytes")
    handle = _unwrap(
        _native.discretized_composition_consume(bytes(artifact), expected_identity, cancel=cancel)
    )
    return PreparedDiscretizedComposition._of(handle)
