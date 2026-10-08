"""Marginal sensitivity model (MSM) bounds, tipping point and sensitivity artifact (2.3 B3).

The Tan (2006) / Zhao-Small-Bhattacharya (2019) marginal sensitivity model asks how far an
inverse-propensity-weighted average treatment effect (ATE) could move if unmeasured
confounding shifted the odds of treatment, given the covariates and a potential outcome, by a
factor in ``[1/Lambda, Lambda]``. This module specializes it to a binary treatment, a discrete
adjustment set (finite strata) and *exact* population inputs::

    strata = [
        MsmStratum(0.5, 0.50, OutcomeLaw.binary(0.8), OutcomeLaw.binary(0.4)),
        MsmStratum(0.5, 0.25, OutcomeLaw.binary(0.5), OutcomeLaw.binary(0.3)),
    ]
    result = msm_ate_sensitivity(strata, lambda_max=3.0, decision_threshold=0.0)
    result.identified       # 0.3: the Lambda = 1 stratified value
    result.grid[2]          # sharp [lower, upper] ATE bounds at the third Lambda
    result.tipping          # smallest Lambda at which the lower bound reaches the threshold
    artifact = result.to_sensitivity_artifact(effect=..., actions=[...], causal_contract_id="c")

Three things stay distinct and are never merged:

* the **assumption range** (``lower``, ``upper`` at every ``Lambda``) is what the ATE could be
  if the assumption holds anywhere in the declared set; it is not a confidence interval;
* the **identified value** is the ``Lambda = 1`` stratified ATE (no unmeasured confounding);
* the **sampling interval** is withheld (``sampling interval: not reported``): the strata are
  supplied as exact quantities, so nothing here is estimated, and composing a sampling
  interval with the range refuses (``cell_not_licensed`` /
  ``msm_sensitivity.composition_not_licensed``). Calibration is unmeasured.

Rust owns the bounds, the bisection tipping point, the artifact and every refusal; this module
builds declarations and raises each refusal as :class:`MsmSensitivityRefusal`, a
:class:`~antecedent.errors.CausalUnsupportedError` with its registered ``reason_code`` and the
namespaced ``detail``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ._native import msm_sensitivity_artifact_run as _artifact_run
from ._native import msm_sensitivity_run as _run
from .errors import CausalTypeError, CausalValueError, StructuredRefusal
from .joint_distribution import ScientificQuantity
from .sensitivity_decision import SensitivityAction, SensitivityArtifact

__all__ = [
    "MsmPoint",
    "MsmResult",
    "MsmSensitivityRefusal",
    "MsmStratum",
    "MsmTipping",
    "MsmUncertainty",
    "OutcomeLaw",
    "msm_ate_sensitivity",
]

_DEFAULT_GRID_POINTS = 17
_DEFAULT_TOLERANCE = 1e-10
_ARTIFACT_ID = "msm-sensitivity"


class MsmSensitivityRefusal(StructuredRefusal):
    """A marginal sensitivity model refusal carrying the structured Rust fields.

    A :class:`~antecedent.errors.StructuredRefusal`: ``code`` (the registered reason code),
    ``detail``, ``offending`` and ``remedy`` are machine-readable. ``detail`` is the namespaced
    ``msm_sensitivity.<slot>`` (``composition_not_licensed``, ``lambda_below_one``,
    ``lambda_range_empty``, ``positivity``, ``stratum_mass``, ``outcome_law``,
    ``invalid_threshold``, ``invalid_tolerance``, ``bounds_exceeded``) or, for the artifact, the
    engine's own ``sensitivity_decision_composition.<slot>``; ``message`` is the human-readable
    context. A refusal with code ``invalid_argument`` is also a
    :class:`~antecedent.errors.CausalValueError`.
    """


def _raise(refusal: str | None) -> None:
    if refusal is not None:
        raise MsmSensitivityRefusal(json.loads(refusal))


def _number(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CausalTypeError(f"{name} must be a number")
    return float(value)


def _numbers(name: str, values: object) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise CausalTypeError(f"{name} must be a sequence of numbers")
    return tuple(_number(name, value) for value in values)


@dataclass(frozen=True, slots=True)
class OutcomeLaw:
    """A finite-support outcome law: values and probabilities in any order.

    Rust checks that the probabilities are nonnegative and sum to one and that the values are
    finite (``msm_sensitivity.outcome_law``).
    """

    values: tuple[float, ...]
    probabilities: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", _numbers("values", self.values))
        object.__setattr__(self, "probabilities", _numbers("probabilities", self.probabilities))

    @classmethod
    def binary(cls, rate: float) -> OutcomeLaw:
        """The law of a 0/1 outcome with ``P(Y = 1) = rate``.

        Raises:
            CausalTypeError: ``rate`` is not a number.
            CausalValueError: ``rate`` is not a probability in ``[0, 1]``.
        """
        rate = _number("rate", rate)
        if not (math.isfinite(rate) and 0.0 <= rate <= 1.0):
            raise CausalValueError(f"rate must be a probability in [0, 1], got {rate!r}")
        return cls((0.0, 1.0), (1.0 - rate, rate))


def _law(name: str, value: object) -> OutcomeLaw:
    if isinstance(value, OutcomeLaw):
        return value
    if isinstance(value, Mapping):
        return OutcomeLaw(value["values"], value["probabilities"])
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 2:
        return OutcomeLaw(value[0], value[1])
    raise CausalTypeError(f"{name} must be an OutcomeLaw, a mapping or a (values, probs) pair")


@dataclass(frozen=True, slots=True)
class MsmStratum:
    """One stratum of the adjustment set.

    ``mass`` is the population mass ``P(x)`` (the masses sum to one), ``propensity`` is
    ``e(x) = P(T = 1 | x)`` strictly inside ``(0, 1)`` (a boundary propensity makes the weights
    undefined and refuses as ``msm_sensitivity.positivity``), and ``treated`` / ``control`` are
    the outcome laws of the treated and the untreated in the stratum.
    """

    mass: float
    propensity: float
    treated: OutcomeLaw
    control: OutcomeLaw

    def __post_init__(self) -> None:
        mass = _number("mass", self.mass)
        propensity = _number("propensity", self.propensity)
        if not (math.isfinite(propensity) and 0.0 <= propensity <= 1.0):
            raise CausalValueError(
                f"propensity must be a probability in [0, 1] (strictly inside (0, 1) to be "
                f"estimable), got {propensity!r}"
            )
        if not (math.isfinite(mass) and 0.0 <= mass <= 1.0):
            raise CausalValueError(f"mass must be a population mass in [0, 1], got {mass!r}")
        object.__setattr__(self, "mass", mass)
        object.__setattr__(self, "propensity", propensity)
        object.__setattr__(self, "treated", _law("treated", self.treated))
        object.__setattr__(self, "control", _law("control", self.control))

    @classmethod
    def table(
        cls,
        table: Any,
        *,
        mass: str = "mass",
        propensity: str = "propensity",
        treated: str = "treated",
        control: str = "control",
    ) -> tuple[MsmStratum, ...]:
        """The strata of a plain per-stratum summary table, one stratum per row.

        ``table`` is a mapping of equal-length columns (``{"mass": [...], ...}``), a
        sequence of row mappings, or anything with ``to_dict("records")`` (a pandas
        ``DataFrame``). ``mass``, ``propensity``, ``treated`` and ``control`` name the
        columns. A ``treated`` / ``control`` cell is an :class:`OutcomeLaw`, a mapping with
        ``values`` and ``probabilities``, a ``(values, probabilities)`` pair, or a bare
        number, which is read as ``P(Y = 1)`` of a binary outcome (:meth:`OutcomeLaw.binary`).
        Nothing is estimated or normalized: the table must already hold exact stratum
        quantities, and Rust still checks that the masses sum to one and every law is valid.

        Raises:
            CausalTypeError: ``table`` is not a mapping of columns, a sequence of rows or a
                frame, or a cell has the wrong type.
            CausalValueError: a named column is missing, columns differ in length, or the
                table is empty.
        """
        if hasattr(table, "to_dict") and not isinstance(table, Mapping):
            rows: list[Mapping[str, Any]] = list(table.to_dict("records"))
        elif isinstance(table, Mapping):
            columns = {name: list(values) for name, values in table.items()}
            lengths = {len(values) for values in columns.values()}
            if len(lengths) > 1:
                raise CausalValueError(f"table columns differ in length: {sorted(lengths)}")
            count = lengths.pop() if lengths else 0
            rows = [{name: values[i] for name, values in columns.items()} for i in range(count)]
        elif isinstance(table, Sequence) and not isinstance(table, (str, bytes)):
            rows = list(table)
            if any(not isinstance(row, Mapping) for row in rows):
                raise CausalTypeError("table rows must be mappings from column name to value")
        else:
            raise CausalTypeError(
                "table must be a mapping of columns, a sequence of row mappings or a DataFrame"
            )
        if not rows:
            raise CausalValueError("table has no rows")

        def cell(row: Mapping[str, Any], name: str) -> Any:
            if name not in row:
                raise CausalValueError(f"table has no column {name!r}; columns: {sorted(row)}")
            return row[name]

        def law(value: Any) -> Any:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return OutcomeLaw.binary(float(value))
            return value

        return tuple(
            cls(
                cell(row, mass),
                cell(row, propensity),
                law(cell(row, treated)),
                law(cell(row, control)),
            )
            for row in rows
        )

    def _wire(self) -> tuple[float, float, list[float], list[float], list[float], list[float]]:
        return (
            self.mass,
            self.propensity,
            list(self.treated.values),
            list(self.treated.probabilities),
            list(self.control.values),
            list(self.control.probabilities),
        )


@dataclass(frozen=True, slots=True)
class MsmPoint:
    """Sharp ATE assumption range ``[lower, upper]`` at one perturbation ``Lambda``."""

    lambda_value: float
    lower: float
    upper: float

    @property
    def bounds(self) -> tuple[float, float]:
        """``(lower, upper)``."""
        return (self.lower, self.upper)


@dataclass(frozen=True, slots=True)
class MsmTipping:
    """The tipping point of the ATE bound against a declared threshold.

    ``status`` is ``bracketed`` (the first crossing lies inside ``bracket``, the unresolved
    region: not reached at ``bracket[0]``, reached at ``bracket[1]``), ``reached_at_origin``
    (the identified value is already at the threshold; ``bracket == (1.0, 1.0)``) or
    ``not_reached_in_box`` (``lambda_max`` does not reach it; no ``bracket``).
    ``direction`` is ``lower_bound_falls`` (the identified ATE is at or above the threshold) or
    ``upper_bound_rises`` (it is below).
    """

    threshold: float
    direction: str
    status: str
    bracketed: bool
    bracket: tuple[float, float] | None
    iterations: int | None
    tolerance: float

    @property
    def lambda_value(self) -> float | None:
        """The smallest ``Lambda`` certified to reach the threshold (``bracket[1]``), if any."""
        return None if self.bracket is None else self.bracket[1]

    def __str__(self) -> str:
        if self.status == "not_reached_in_box":
            return f"threshold {self.threshold:g} is not reached inside the declared Lambda range"
        if self.status == "reached_at_origin":
            return f"threshold {self.threshold:g} is already reached at Lambda = 1"
        if self.bracket is None:  # pragma: no cover - a bracketed status always has one
            return f"threshold {self.threshold:g}: no bracket"
        low, high = self.bracket
        bound = "lower" if self.direction == "lower_bound_falls" else "upper"
        return (
            f"the {bound} bound reaches {self.threshold:g} at Lambda in "
            f"[{low:.12g}, {high:.12g}] (tolerance {self.tolerance:g})"
        )


@dataclass(frozen=True, slots=True)
class MsmUncertainty:
    """The withheld sampling status, distinct from the assumption range."""

    sampling_interval: str
    reason_code: str
    detail: str


@dataclass(frozen=True, slots=True)
class _Request:
    strata: tuple[MsmStratum, ...]
    lambda_max: float
    grid_points: int
    decision_threshold: float | None
    tolerance: float


@dataclass(frozen=True, slots=True)
class MsmResult:
    """Auditable MSM sensitivity of the stratified ATE.

    ``grid`` holds the sharp ATE bounds on ``Lambda`` from ``1`` to ``lambda_max``; the ranges
    nest, so the bounds widen with ``Lambda``. ``identified`` is the ``Lambda = 1`` value.
    ``tipping`` is present when a ``decision_threshold`` was declared. ``uncertainty``
    withholds the sampling interval. ``inference_claim`` is always ``assumption_range``.
    """

    family: str
    perturbation_scale: str
    normalization: str
    target: str
    identified: float
    lambda_max: float
    grid: tuple[MsmPoint, ...]
    decision_threshold: float | None
    tipping: MsmTipping | None
    tolerance: float
    strata: int
    method: str
    interpretation: str
    inference_claim: str
    uncertainty: MsmUncertainty
    _request: _Request = field(repr=False, compare=False)

    @property
    def lambdas(self) -> tuple[float, ...]:
        """The perturbation grid."""
        return tuple(point.lambda_value for point in self.grid)

    def assumption_range(self, lambda_value: float | None = None) -> tuple[float, float]:
        """The ATE assumption range at a grid ``Lambda`` (default: ``lambda_max``)."""
        target = self.lambda_max if lambda_value is None else lambda_value
        for point in self.grid:
            if math.isclose(point.lambda_value, target, rel_tol=1e-12, abs_tol=1e-12):
                return point.bounds
        raise CausalValueError(f"Lambda {target!r} is not a grid point of this result")

    def explain(self) -> str:
        """The range, the identified value, the tipping point and what they are not."""
        low, high = self.grid[-1].bounds
        text = (
            f"Identified ATE {self.identified:g} (Lambda = 1). If the odds of treatment given "
            f"the covariates and a potential outcome differ from the odds given the covariates "
            f"by at most a factor of {self.lambda_max:g}, the ATE lies in [{low:g}, {high:g}]: "
            f"an assumption range, not a confidence interval."
        )
        if self.tipping is not None:
            text += f" Tipping point: {self.tipping}."
        return text + f" {self.uncertainty.sampling_interval}."

    def to_sensitivity_artifact(
        self,
        *,
        effect: ScientificQuantity,
        actions: Sequence[SensitivityAction],
        causal_contract_id: str,
        point_quantities: Sequence[tuple[ScientificQuantity, Sequence[float]]] = (),
    ) -> SensitivityArtifact:
        """The F17 :class:`~antecedent.sensitivity_decision.SensitivityArtifact` of this result.

        The assumption coordinate is ``Lambda`` (scale ``odds_ratio_bound``, range
        ``[1, lambda_max]``); the one ranged surface quantity is ``effect`` with the sharp ATE
        bounds at each grid ``Lambda``. ``point_quantities`` are assumption-dependent point
        quantities (for example a cost that depends on the assumed ``Lambda``), each with one
        value per grid point. ``actions`` are the compared actions' utilities (read through
        :func:`antecedent.sensitivity_decision.quantity` by ``effect.variable_id``). The sampling
        interval stays withheld and the artifact refuses any composition with the range.
        Refuses with :class:`MsmSensitivityRefusal` on a malformed surface (``invalid_surface``)
        or fewer than two actions or unlike units (``wrong_contract``).

        Raises:
            CausalTypeError: ``effect``, ``actions``, ``point_quantities`` or
                ``causal_contract_id`` has the wrong type.
            MsmSensitivityRefusal: the surface is malformed or the actions are not comparable.
        """
        if not isinstance(effect, ScientificQuantity):
            raise CausalTypeError("effect must be a ScientificQuantity")
        if isinstance(causal_contract_id, bool) or not isinstance(causal_contract_id, str):
            raise CausalTypeError("causal_contract_id must be a string")
        points: list[dict[str, Any]] = []
        for item in point_quantities:
            if not isinstance(item, tuple) or len(item) != 2:
                raise CausalTypeError("point_quantities are (ScientificQuantity, values) pairs")
            quantity, values = item
            if not isinstance(quantity, ScientificQuantity):
                raise CausalTypeError("a point quantity must be a ScientificQuantity")
            wire_values = list(_numbers("values", values))
            points.append({"quantity": quantity._wire(), "values": wire_values})
        declared = list(actions)
        if any(not isinstance(action, SensitivityAction) for action in declared):
            raise CausalTypeError("actions must be sensitivity_decision.SensitivityAction values")
        request = self._request
        data, refusal = _artifact_run(
            [stratum._wire() for stratum in request.strata],
            request.lambda_max,
            request.grid_points,
            request.decision_threshold,
            request.tolerance,
            json.dumps(effect._wire()),
            json.dumps(points),
            json.dumps([action._wire() for action in declared]),
            causal_contract_id,
            _ARTIFACT_ID,
        )
        _raise(refusal)
        if data is None:  # pragma: no cover - the native contract
            raise CausalValueError("the native artifact returned neither a result nor a refusal")
        return SensitivityArtifact.consume(data)


def _result(wire: Mapping[str, Any], request: _Request) -> MsmResult:
    tipping_wire = wire["tipping"]
    tipping: MsmTipping | None = None
    if tipping_wire is not None:
        bracket = tipping_wire["bracket"]
        tipping = MsmTipping(
            threshold=float(tipping_wire["threshold"]),
            direction=str(tipping_wire["direction"]),
            status=str(tipping_wire["status"]),
            bracketed=bool(tipping_wire["bracketed"]),
            bracket=None if bracket is None else (float(bracket["lower"]), float(bracket["upper"])),
            iterations=None if bracket is None else int(bracket["iterations"]),
            tolerance=float(tipping_wire["tolerance"]),
        )
    threshold = wire["decision_threshold"]
    uncertainty = wire["uncertainty"]
    return MsmResult(
        family=str(wire["family"]),
        perturbation_scale=str(wire["perturbation_scale"]),
        normalization=str(wire["normalization"]),
        target=str(wire["target"]),
        identified=float(wire["identified"]),
        lambda_max=float(wire["lambda_max"]),
        grid=tuple(
            MsmPoint(float(p["lambda"]), float(p["lower"]), float(p["upper"])) for p in wire["grid"]
        ),
        decision_threshold=None if threshold is None else float(threshold),
        tipping=tipping,
        tolerance=float(wire["tolerance"]),
        strata=int(wire["strata"]),
        method=str(wire["method"]),
        interpretation=str(wire["interpretation"]),
        inference_claim=str(wire["inference_claim"]),
        uncertainty=MsmUncertainty(
            sampling_interval=str(uncertainty["sampling_interval"]),
            reason_code=str(uncertainty["reason_code"]),
            detail=str(uncertainty["detail"]),
        ),
        _request=request,
    )


def msm_ate_sensitivity(
    strata: Sequence[MsmStratum],
    lambda_max: float,
    *,
    grid_points: int = _DEFAULT_GRID_POINTS,
    decision_threshold: float | None = None,
    tolerance: float = _DEFAULT_TOLERANCE,
    sampling_composition: str | None = None,
) -> MsmResult:
    """Sharp marginal-sensitivity-model bounds of the stratified ATE over a ``Lambda`` grid.

    ``strata`` are the exact adjustment-set strata (mass, propensity, treated and control
    outcome laws). ``lambda_max`` (above one, at most 1000) is the largest odds-ratio bound;
    the grid has ``grid_points`` equally spaced values from ``1`` to ``lambda_max`` (2 to
    1024). With a ``decision_threshold`` the result carries the tipping point: the smallest
    ``Lambda`` at which the identified-side bound reaches it, found by bisection to
    ``tolerance`` (``1e-12`` to ``1e-3``).

    ``sampling_composition`` requests composing a sampling interval with the assumption range
    by a named method; no method is licensed, so any value refuses with
    :class:`MsmSensitivityRefusal` (``cell_not_licensed`` /
    ``msm_sensitivity.composition_not_licensed``). Other refusals: ``Lambda`` below one
    (``lambda_below_one``) or not above one (``lambda_range_empty``), a propensity outside
    ``(0, 1)`` (``positivity``), masses that do not sum to one (``stratum_mass``), a malformed
    outcome law (``outcome_law``), or exceeded bounds (``bounds_exceeded``).

    Raises:
        CausalTypeError: ``strata`` is not a sequence of :class:`MsmStratum`, or a numeric
            argument has the wrong type.
        CausalValueError: ``grid_points`` is negative. A refusal with reason code
            ``invalid_argument`` (a bad ``stratum_mass``, ``outcome_law``, tolerance or
            threshold) is also a :class:`CausalValueError`.
        MsmSensitivityRefusal: the model is refused (see above), with the machine-readable
            ``code``, ``detail`` and ``remedy`` of any structured refusal.
    """
    if isinstance(strata, (str, bytes)) or not isinstance(strata, Sequence):
        raise CausalTypeError("strata must be a sequence of MsmStratum")
    declared = tuple(strata)
    if any(not isinstance(stratum, MsmStratum) for stratum in declared):
        raise CausalTypeError("every stratum must be an MsmStratum")
    lambda_value = _number("lambda_max", lambda_max)
    if isinstance(grid_points, bool) or not isinstance(grid_points, int):
        raise CausalTypeError("grid_points must be an integer")
    if grid_points < 0:
        raise CausalValueError("grid_points must be non-negative")
    threshold = (
        None if decision_threshold is None else _number("decision_threshold", decision_threshold)
    )
    tol = _number("tolerance", tolerance)
    if sampling_composition is not None and not isinstance(sampling_composition, str):
        raise CausalTypeError("sampling_composition must be a string or None")
    request = _Request(declared, lambda_value, grid_points, threshold, tol)
    text, refusal = _run(
        [stratum._wire() for stratum in declared],
        lambda_value,
        grid_points,
        threshold,
        tol,
        sampling_composition,
    )
    _raise(refusal)
    if text is None:  # pragma: no cover - the native contract
        raise CausalValueError("the native MSM bounds returned neither a result nor a refusal")
    return _result(json.loads(text), request)
