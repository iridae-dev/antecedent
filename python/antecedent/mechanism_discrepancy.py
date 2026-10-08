"""Source-target mechanism discrepancy diagnostic (2.3 B3).

A Wald test of the null that the conditional mechanism of one node ``V`` given its parents is
the same in a source and a target population, from comparable measurements::

    result = mechanism_discrepancy(
        source=Sample("source", outcome=y_s, parents={"x": x_s}),
        target=Sample("target", outcome=y_t, parents={"x": x_t}),
        measurement=Measurement("V", "mg", parents=[("x", "cm")], protocol_id="protocol-1"),
        compare_intercept=True,
    )
    result.statistic, result.degrees_of_freedom, result.p_value
    result.coefficients          # per-coefficient breakdown with Holm adjustment
    result.conclusion            # "not_rejected" | "rejected"
    result.non_rejection_certifies_invariance   # always False

The null is one linear regression coefficient vector ``E[V | pa]`` shared by the two populations.
The statistic is ``W = d' (V_s + V_t)^-1 d`` over the compared coefficients, chi-square with
``df`` equal to their number; ``V_s + V_t`` is the sum of the two OLS covariance matrices, valid
only for **independent** samples (shared units or an unknown dependence refuse).

What a result does **not** say:

* **Non-rejection never certifies invariance.** ``non_rejection_certifies_invariance`` is
  always ``False``; only differences larger than the ``minimal_detectable_difference`` of each
  coefficient could have been detected.
* The diagnostic **informs only a selection node on the node itself** (``informs_selection_on``):
  a rejection says the mechanism of the node is not invariant, so such a selection node cannot be
  excluded; non-rejection leaves it open and says nothing about selection nodes on other
  variables.
* It is a linear-Gaussian-mean diagnostic: a shift in a nonlinear or higher-moment feature of the
  mechanism may go undetected, and its Type I error and power are **unmeasured**
  (``calibration == "unmeasured"``).

Rust owns the fits, the statistic, the artifact and every refusal; this module builds
declarations and raises each refusal as :class:`MechanismDiscrepancyRefusal`, a
:class:`~antecedent.errors.CausalUnsupportedError` with its registered ``reason_code`` and the
namespaced ``detail`` (``incomparable_measurements``, ``dependence_unknown``,
``rank_deficient_design``, ``degenerate_covariance``, ``sample_too_small``, ...).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ._native import mechanism_discrepancy_consume as _consume
from ._native import mechanism_discrepancy_run as _run
from ._native import mechanism_discrepancy_summarize as _summarize
from .errors import CausalTypeError, CausalUnsupportedError, CausalValueError

__all__ = [
    "CoefficientDiscrepancy",
    "MechanismDiscrepancyRefusal",
    "MechanismDiscrepancyResult",
    "Measurement",
    "PopulationFit",
    "Sample",
    "SufficientStatistics",
    "mechanism_discrepancy",
]

_ARTIFACT_ID = "mechanism-discrepancy"
_DEPENDENCES = ("independent", "shared_units", "unknown")


class MechanismDiscrepancyRefusal(CausalUnsupportedError):
    """A mechanism discrepancy refusal carrying the structured Rust fields.

    ``reason_code`` is registered. ``detail`` is the namespaced
    ``mechanism_discrepancy.<slot>``; ``message`` is the human-readable context. Typical details:
    ``incomparable_measurements`` (a different node, variable set, unit or protocol id, or a blank
    declaration; ``route_not_supported``), ``dependence_unknown`` (shared units or an unknown
    dependence; ``route_not_supported``), ``wrong_contract`` (a consumed artifact whose identity
    differs from the retained one), ``rank_deficient_design``, ``degenerate_covariance``,
    ``sample_too_small``, ``non_finite_value``, ``row_count_mismatch``, ``inconsistent_summary``,
    ``invalid_alpha``, ``invalid_power`` and ``no_compared_coefficients`` (``invalid_argument``).
    """

    def __init__(self, refusal: Mapping[str, Any]) -> None:
        message = str(refusal.get("message") or "")
        detail = str(refusal["detail"])
        super().__init__(
            detail + (f": {message}" if message else ""), reason_code=str(refusal["code"])
        )
        self.detail: str = detail
        self.message: str = message


def _raise(refusal: str | None) -> None:
    if refusal is not None:
        raise MechanismDiscrepancyRefusal(json.loads(refusal))


def _number(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CausalTypeError(f"{name} must be a number")
    return float(value)


def _floats(name: str, values: Any) -> list[float]:
    """A one-dimensional numeric column (a sequence or an array with ``tolist``)."""
    if hasattr(values, "tolist") and not isinstance(values, (str, bytes)):
        values = values.tolist()
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise CausalTypeError(f"{name} must be a one-dimensional numeric sequence or array")
    return [_number(name, value) for value in values]


def _text(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise CausalTypeError(f"{name} must be a string")
    return value


# ---------------------------------------------------------------------- declarations


@dataclass(frozen=True, slots=True)
class Measurement:
    """What was measured and how: the comparability contract of one population.

    ``node`` is the variable ``V`` whose mechanism is compared and ``node_unit`` its unit or
    coordinate; ``parents`` are ``(name, unit)`` pairs; ``protocol_id`` names the measurement
    protocol. Both populations must declare the same node, parent names, units and protocol id;
    a blank unit or id refuses (comparability is never assumed from silence). Parent order is
    irrelevant to the test; it fixes the column order of a population's summary statistics.
    """

    node: str
    node_unit: str
    parents: tuple[tuple[str, str], ...]
    protocol_id: str

    def __post_init__(self) -> None:
        _text("node", self.node)
        _text("node_unit", self.node_unit)
        _text("protocol_id", self.protocol_id)
        if isinstance(self.parents, (str, bytes)) or not isinstance(self.parents, Sequence):
            raise CausalTypeError("parents must be a sequence of (name, unit) pairs")
        pairs: list[tuple[str, str]] = []
        for item in self.parents:
            if isinstance(item, (str, bytes)) or not isinstance(item, Sequence) or len(item) != 2:
                raise CausalTypeError("every parent is a (name, unit) pair")
            pairs.append((_text("parent name", item[0]), _text("parent unit", item[1])))
        object.__setattr__(self, "parents", tuple(pairs))

    @property
    def parent_names(self) -> tuple[str, ...]:
        """The parent names in declaration order."""
        return tuple(name for name, _ in self.parents)

    def _wire(self) -> dict[str, Any]:
        return {
            "node": self.node,
            "node_unit": self.node_unit,
            "parents": [{"name": name, "unit": unit} for name, unit in self.parents],
            "protocol_id": self.protocol_id,
        }


@dataclass(frozen=True, slots=True)
class SufficientStatistics:
    """One population's sufficient statistics: all the Wald test needs from the rows.

    ``n`` is the row count, ``xtx`` the symmetric ``p x p`` matrix ``X'X`` (nested rows or flat
    row-major), ``xty`` the vector ``X'y`` and ``yty`` the scalar ``y'y``, in design order
    (intercept, then the parents in :attr:`Measurement.parents` order), ``p = 1 + parents``.
    Rust checks symmetry, ``X'X[0, 0] = n`` and at least two residual degrees of freedom
    (``inconsistent_summary``, ``sample_too_small``).
    """

    n: int
    xtx: tuple[float, ...]
    xty: tuple[float, ...]
    yty: float

    def __post_init__(self) -> None:
        if isinstance(self.n, bool) or not isinstance(self.n, int):
            raise CausalTypeError("n must be an integer")
        if self.n < 0:
            raise CausalValueError("n must be non-negative")
        rows: Any = self.xtx
        if hasattr(rows, "tolist"):
            rows = rows.tolist()
        if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
            raise CausalTypeError("xtx must be a matrix or a flat row-major sequence")
        if rows and isinstance(rows[0], Sequence) and not isinstance(rows[0], (str, bytes)):
            flat = [value for row in rows for value in _floats("xtx", row)]
        else:
            flat = _floats("xtx", rows)
        object.__setattr__(self, "xtx", tuple(flat))
        object.__setattr__(self, "xty", tuple(_floats("xty", self.xty)))
        object.__setattr__(self, "yty", _number("yty", self.yty))


@dataclass(frozen=True, slots=True)
class Sample:
    """One population: raw rows (``outcome`` and ``parents``) or summary ``statistics``.

    ``parents`` maps each parent name to its column; the columns must have one value per
    ``outcome`` row and the names must equal the measurement's parent names. ``unit_ids`` are
    optional row identities used only to detect shared units: a common id between the source and
    the target refuses (``dependence_unknown``). ``measurement`` optionally overrides the
    measurement contract passed to :func:`mechanism_discrepancy` for this population (for
    example to declare the population's own units).
    """

    label: str
    outcome: Any = None
    parents: Mapping[str, Any] | None = None
    unit_ids: Sequence[str] | None = None
    statistics: SufficientStatistics | None = None
    measurement: Measurement | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise CausalValueError("label must name the population")
        raw = self.outcome is not None or self.parents is not None
        if raw and self.statistics is not None:
            raise CausalValueError("supply raw rows or summary statistics, not both")
        if self.statistics is not None and not isinstance(self.statistics, SufficientStatistics):
            raise CausalTypeError("statistics must be SufficientStatistics")
        if not raw and self.statistics is None:
            raise CausalValueError("supply outcome and parents, or summary statistics")
        if raw and (self.outcome is None or self.parents is None):
            raise CausalValueError("raw rows need both outcome and parents")
        if self.parents is not None and not isinstance(self.parents, Mapping):
            raise CausalTypeError("parents must map names to columns")
        if self.measurement is not None and not isinstance(self.measurement, Measurement):
            raise CausalTypeError("measurement must be a Measurement")
        if self.unit_ids is not None:
            if isinstance(self.unit_ids, (str, bytes)) or not isinstance(self.unit_ids, Sequence):
                raise CausalTypeError("unit_ids must be a sequence of strings")
            if any(not isinstance(item, str) for item in self.unit_ids):
                raise CausalTypeError("unit_ids must be strings")

    @classmethod
    def from_summary(
        cls,
        label: str,
        *,
        n: int,
        xtx: Any,
        xty: Any,
        yty: float,
        unit_ids: Sequence[str] | None = None,
        measurement: Measurement | None = None,
    ) -> Sample:
        """A population given by its sufficient statistics (no rows)."""
        return cls(
            label,
            statistics=SufficientStatistics(n, xtx, xty, yty),
            unit_ids=unit_ids,
            measurement=measurement,
        )

    def _population(self, declared: Measurement) -> dict[str, Any]:
        """The wire declaration of this population under ``declared``."""
        if self.statistics is not None:
            stats = self.statistics
            n, xtx, xty, yty = stats.n, list(stats.xtx), list(stats.xty), stats.yty
        else:
            assert self.parents is not None
            names = declared.parent_names
            if set(self.parents) != set(names):
                raise CausalValueError(
                    f"{self.label!r} supplies parents {sorted(self.parents)}, "
                    f"but its measurement declares {sorted(names)}"
                )
            text, refusal = _summarize(
                self.label,
                declared.node,
                declared.node_unit,
                list(declared.parents),
                declared.protocol_id,
                _floats("outcome", self.outcome),
                [_floats(f"parent {name!r}", self.parents[name]) for name in names],
            )
            _raise(refusal)
            assert text is not None
            summary = json.loads(text)
            n, xtx, xty, yty = (
                int(summary["n"]),
                summary["xtx"],
                summary["xty"],
                summary["yty"],
            )
        return {
            "label": self.label,
            "measurement": declared._wire(),
            "n": n,
            "xtx": xtx,
            "xty": xty,
            "yty": yty,
        }


# --------------------------------------------------------------------------- results


@dataclass(frozen=True, slots=True)
class PopulationFit:
    """One population's OLS fit; coefficients run intercept first, then parents by name."""

    label: str
    n: int
    residual_df: int
    residual_variance: float
    coefficients: tuple[float, ...]
    standard_errors: tuple[float, ...]

    @classmethod
    def _from_wire(cls, wire: Mapping[str, Any]) -> PopulationFit:
        return cls(
            label=str(wire["label"]),
            n=int(wire["n"]),
            residual_df=int(wire["residual_df"]),
            residual_variance=float(wire["residual_variance"]),
            coefficients=tuple(float(v) for v in wire["coefficients"]),
            standard_errors=tuple(float(v) for v in wire["standard_errors"]),
        )


@dataclass(frozen=True, slots=True)
class CoefficientDiscrepancy:
    """One compared coefficient.

    ``difference`` is ``target - source`` and ``standard_error`` is ``sqrt(V_s,jj + V_t,jj)``;
    ``p_value`` is the two-sided normal p-value and ``p_holm`` its Holm adjustment over the
    compared coefficients. ``minimal_detectable_difference`` is
    ``(z_{1-alpha/2} + z_power) * standard_error`` of a two-sided single-coefficient z test at the
    declared ``alpha`` and ``power`` (no multiplicity adjustment): a smaller difference was not
    detectable.
    """

    name: str
    source: float
    target: float
    difference: float
    standard_error: float
    z: float
    p_value: float
    p_holm: float
    rejected: bool
    minimal_detectable_difference: float

    @classmethod
    def _from_wire(cls, wire: Mapping[str, Any]) -> CoefficientDiscrepancy:
        return cls(
            name=str(wire["name"]),
            source=float(wire["source"]),
            target=float(wire["target"]),
            difference=float(wire["difference"]),
            standard_error=float(wire["standard_error"]),
            z=float(wire["z"]),
            p_value=float(wire["p_value"]),
            p_holm=float(wire["p_holm"]),
            rejected=bool(wire["rejected"]),
            minimal_detectable_difference=float(wire["minimal_detectable_difference"]),
        )


@dataclass(frozen=True, slots=True)
class MechanismDiscrepancyResult:
    """The mechanism discrepancy diagnostic with its identity and standing caveats.

    ``conclusion`` is ``not_rejected`` or ``rejected`` at ``alpha``; a non-rejection never
    certifies invariance (:attr:`non_rejection_certifies_invariance` is always ``False``).
    :meth:`export` serializes the result through a bounded, checksummed container and
    :meth:`consume` recomputes it from the bytes, refusing a resealed mutation.
    """

    null: str
    measurement: Measurement
    statistic: float
    degrees_of_freedom: int
    p_value: float
    conclusion: str
    non_rejection_certifies_invariance: bool
    alpha: float
    power: float
    compare_intercept: bool
    detectability_factor: float
    power_statement: str
    coefficient_names: tuple[str, ...]
    source: PopulationFit
    target: PopulationFit
    coefficients: tuple[CoefficientDiscrepancy, ...]
    informs_selection_on: tuple[str, ...]
    alignment: str
    dependence: str
    dependence_assumption: str
    inference_claim: str
    calibration: str
    caveats: tuple[str, ...]
    identity: Mapping[str, str]
    _artifact: bytes

    @classmethod
    def _from_report(cls, report: Mapping[str, Any], artifact: bytes) -> MechanismDiscrepancyResult:
        body = report["result"]
        measurement = report["measurement"]
        return cls(
            null=str(report["null"]),
            measurement=Measurement(
                node=measurement["node"],
                node_unit=measurement["node_unit"],
                parents=tuple((p["name"], p["unit"]) for p in measurement["parents"]),
                protocol_id=measurement["protocol_id"],
            ),
            statistic=float(body["statistic"]),
            degrees_of_freedom=int(body["degrees_of_freedom"]),
            p_value=float(body["p_value"]),
            conclusion=str(body["conclusion"]),
            non_rejection_certifies_invariance=bool(body["non_rejection_certifies_invariance"]),
            alpha=float(report["alpha"]),
            power=float(report["power"]),
            compare_intercept=bool(report["compare_intercept"]),
            detectability_factor=float(body["detectability_factor"]),
            power_statement=str(body["power_statement"]),
            coefficient_names=tuple(body["coefficient_names"]),
            source=PopulationFit._from_wire(body["source"]),
            target=PopulationFit._from_wire(body["target"]),
            coefficients=tuple(CoefficientDiscrepancy._from_wire(c) for c in body["coefficients"]),
            informs_selection_on=tuple(body["informs_selection_on"]),
            alignment=str(report["alignment"]),
            dependence=str(report["dependence"]),
            dependence_assumption=str(report["dependence_assumption"]),
            inference_claim=str(report["inference_claim"]),
            calibration=str(report["calibration"]),
            caveats=tuple(report["caveats"]),
            identity=dict(report["identity"]),
            _artifact=artifact,
        )

    @property
    def rejected(self) -> bool:
        """Whether the null was rejected at ``alpha`` (``False`` is not evidence of invariance)."""
        return self.conclusion == "rejected"

    @property
    def minimal_detectable_differences(self) -> dict[str, float]:
        """Per compared coefficient, the smallest difference detectable at the declared level."""
        return {c.name: c.minimal_detectable_difference for c in self.coefficients}

    def explain(self) -> str:
        """The decision, what it ranges over and what it is not."""
        node = self.measurement.node
        head = (
            f"Wald statistic {self.statistic:.6g} on {self.degrees_of_freedom} degree(s) of "
            f"freedom, p = {self.p_value:.6g}."
        )
        if self.rejected:
            verdict = (
                f"The null was rejected at alpha {self.alpha:g}: the mechanism of {node!r} given "
                f"its parents differs between {self.source.label!r} and {self.target.label!r}, so a "
                f"selection node on {node!r} cannot be excluded."
            )
        else:
            verdict = (
                f"The null was not rejected at alpha {self.alpha:g}. This does not certify that "
                f"the mechanism of {node!r} is unchanged: {self.power_statement}"
            )
        return f"{head} {verdict} {self.alignment}; {self.caveats[-1]}."

    # -- artifact --------------------------------------------------------------

    def export(self) -> bytes:
        """The result as a bounded, checksummed artifact (recompute with :meth:`consume`)."""
        return self._artifact

    @classmethod
    def consume(
        cls, data: bytes, *, expected_identity: Mapping[str, str] | None = None
    ) -> MechanismDiscrepancyResult:
        """Consume by recomputation; refuses a resealed mutation.

        The request is rebuilt from the stored declarations and summary statistics, both OLS
        fits, the statistic, the p-value, the Holm breakdown and the detectability are
        recomputed and must equal the stored ones bit for bit, and the identity digests are
        recomputed. When the consumer retained ``expected_identity`` (the dict of
        :attr:`identity`) independently of the bytes, every field must match it, so a changed
        measurement contract, summary statistic, level or null is refused even when the artifact
        was resealed consistently (``mechanism_discrepancy.wrong_contract``).
        """
        if not isinstance(data, bytes):
            raise CausalTypeError("artifact must be bytes")
        text, refusal = _consume(
            data, None if expected_identity is None else json.dumps(dict(expected_identity))
        )
        _raise(refusal)
        assert text is not None
        return cls._from_report(json.loads(text), data)

    def __repr__(self) -> str:
        return (
            f"<MechanismDiscrepancyResult {self.measurement.node} W={self.statistic:.6g} "
            f"df={self.degrees_of_freedom} p={self.p_value:.6g} {self.conclusion}>"
        )


# ---------------------------------------------------------------------------- entry


def _declared(sample: Sample, shared: Measurement | None, role: str) -> Measurement:
    declared = sample.measurement if sample.measurement is not None else shared
    if declared is None:
        raise CausalValueError(
            f"the {role} population needs a measurement contract: pass measurement= or "
            "Sample(measurement=...)"
        )
    return declared


def mechanism_discrepancy(
    *,
    source: Sample,
    target: Sample,
    measurement: Measurement | None = None,
    compare_intercept: bool = True,
    alpha: float = 0.05,
    power: float = 0.8,
    dependence: str = "independent",
) -> MechanismDiscrepancyResult:
    """Test whether the mechanism of one node differs between a source and a target population.

    ``source`` and ``target`` are :class:`Sample` values (raw rows or summary statistics) and
    ``measurement`` the comparability contract both share (a sample's own ``measurement``
    overrides it). ``compare_intercept`` declares whether the intercept enters the compared
    coefficient vector (it is always fitted). ``alpha`` (in ``[1e-12, 1)``) is the level of the
    decisions and ``power`` (in ``[0.5, 1 - 1e-12]``) the power of the minimal detectable
    differences. ``dependence`` declares the relationship of the two samples: ``"independent"``
    (the only value the test runs under, disjoint units); ``"shared_units"`` and ``"unknown"``
    refuse with ``dependence_unknown``, as does any ``unit_ids`` overlap.

    Different nodes, parent sets, units or protocol ids, or a blank unit or protocol id, refuse
    with ``route_not_supported`` / ``mechanism_discrepancy.incomparable_measurements``. A
    rank-deficient design, a sample with fewer than two residual degrees of freedom, a population
    that fits its mechanism exactly and non-finite values refuse with ``invalid_argument`` and
    their own detail. The result never certifies invariance on non-rejection.
    """
    if not isinstance(source, Sample) or not isinstance(target, Sample):
        raise CausalTypeError("source and target must be Sample values")
    if measurement is not None and not isinstance(measurement, Measurement):
        raise CausalTypeError("measurement must be a Measurement or None")
    if not isinstance(compare_intercept, bool):
        raise CausalTypeError("compare_intercept must be a bool")
    if not isinstance(dependence, str) or dependence not in _DEPENDENCES:
        raise CausalValueError(f"dependence must be one of {_DEPENDENCES}")
    level = _number("alpha", alpha)
    target_power = _number("power", power)
    if not (math.isfinite(level) and math.isfinite(target_power)):
        raise CausalValueError("alpha and power must be finite")
    request = {
        "source": source._population(_declared(source, measurement, "source")),
        "target": target._population(_declared(target, measurement, "target")),
        "compare_intercept": compare_intercept,
        "alpha": level,
        "power": target_power,
        "dependence": dependence,
    }
    report, data, refusal = _run(
        json.dumps(request),
        list(source.unit_ids or ()),
        list(target.unit_ids or ()),
        _ARTIFACT_ID,
    )
    _raise(refusal)
    assert report is not None
    assert data is not None
    return MechanismDiscrepancyResult._from_report(json.loads(report), data)
