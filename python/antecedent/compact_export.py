"""Compact runtime export of a fitted linear-in-coefficients effect model (B4).

The export stores a coefficient vector, its full covariance, a finite basis (intercept, linear,
polynomial and one-hot terms of named quantities), the declared support of every input and a
refusal mask of query regions the model must refuse, all under one identity::

    from antecedent import compact_export as ce

    export = ce.CompactExport.build(
        response=ce.Quantity("y", units="mmHg", role="outcome"),
        inputs=[ce.Numeric("x", 0.0, 10.0, units="mg"), ce.Categorical("g", ["a", "b", "c"])],
        terms=[ce.Intercept(), ce.Linear("x"), ce.OneHot("g", "b"), ce.OneHot("g", "c")],
        coefficients=[1.0, 2.0, 0.5, -1.0],
        covariance=covariance,                      # full, row-major, aligned with terms
        mask=[ce.MaskRegion("high_dose_c", {"x": (8.0, 10.0), "g": {"c"}})],
    )
    export.evaluate({"x": 1.0, "g": "b"})           # PointEstimate(point=3.5, ...)
    kept = export.identity                           # retain out-of-band
    verified = ce.CompactExport.consume(export.export(), expected_identity=kept)

Scope. The export evaluates a point prediction ``f(x) = b' phi(x)`` inside the declared support
and outside the refusal mask, and nothing else. The only uncertainty it can report is the
model-based standard error ``sqrt(phi' V phi)`` from the stored covariance. It is labelled
``model_based_sqrt_phi_v_phi`` with calibration **unmeasured**: it is never an interval or a
coverage claim, and it carries no extrapolation or misspecification uncertainty. A query outside
the declared support, in a masked region, with a missing, unknown or mistyped quantity, or with a
non-finite value is refused (:class:`CompactExportRefusal`, ``cell_not_licensed``); nothing is
extrapolated.

Verification is independent: :meth:`CompactExport.consume` and every :meth:`~CompactExport.evaluate`
re-decode the artifact from its bytes, recompute the digests and identity from the stored fields
only and require the identity the caller retained, so a changed coefficient, covariance entry,
term, support bound, mask region or scope statement is refused even when the container was
resealed with recomputed checksums (``compact_export.identity_unexpected``).
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._b4_refusal import StructuredRefusal
from ._native import build_compact_export as _build
from ._native import consume_compact_export as _consume
from ._native import evaluate_compact_export as _evaluate
from .errors import CausalTypeError, CausalValueError

__all__ = [
    "Categorical",
    "CompactExport",
    "CompactExportRefusal",
    "Intercept",
    "Linear",
    "MaskRegion",
    "Numeric",
    "OneHot",
    "PointEstimate",
    "Power",
    "Quantity",
]

_ROLES = ("treatment", "outcome", "covariate", "mediator", "selection", "utility")
_DEFAULT_ARTIFACT_ID = "compact_export"


class CompactExportRefusal(StructuredRefusal):
    """A typed refusal of the compact export.

    A :class:`~antecedent.errors.CausalUnsupportedError` with a registered ``reason_code``
    (``cell_not_licensed`` for an unlicensed query, ``invalid_argument`` for a malformed or
    tampered export) and a ``compact_export.*`` ``detail`` such as
    ``compact_export.out_of_support`` or ``compact_export.masked_region``; ``offending`` is the
    quantity, region or identity the refusal is about.
    """


def _name(value: object, what: str) -> str:
    if not isinstance(value, str) or not value:
        raise CausalTypeError(f"{what} must be a non-empty string")
    return value


def _real(value: object, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float | np.floating | np.integer):
        raise CausalTypeError(f"{what} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise CompactExportRefusal(
            {
                "code": "invalid_argument",
                "stage": "build",
                "detail": "compact_export.non_finite_value",
                "offending": what,
                "message": f"{what} is not finite",
            }
        )
    return number


@dataclass(frozen=True, slots=True)
class Quantity:
    """The scientific coordinate of the response: a named quantity with units and role."""

    name: str
    units: str = "1"
    role: str = "outcome"
    population: str = "target"
    regime: str = "observed"
    functional: str = "mean"
    transform: str = "identity"

    def __post_init__(self) -> None:
        for what in ("name", "units", "population", "regime", "functional", "transform"):
            _name(getattr(self, what), what)
        if self.role not in _ROLES:
            raise CausalValueError(f"role must be one of {_ROLES}")

    def _wire(self) -> dict[str, Any]:
        return {
            "version": 1,
            "variable_id": self.name,
            "variable_name": self.name,
            "role": self.role,
            "units": self.units,
            "population_id": self.population,
            "regime_id": self.regime,
            "horizon": 0,
            "functional_id": self.functional,
            "conditioning": [],
            "transform_id": self.transform,
        }


@dataclass(frozen=True, slots=True)
class Numeric:
    """A numeric input with a closed declared support ``[lo, hi]``."""

    name: str
    lo: float
    hi: float
    units: str = "1"
    role: str = "covariate"

    def __post_init__(self) -> None:
        object.__setattr__(self, "lo", _real(self.lo, "lo"))
        object.__setattr__(self, "hi", _real(self.hi, "hi"))
        self._quantity()

    def _quantity(self) -> Quantity:
        return Quantity(self.name, units=self.units, role=self.role)

    def _wire(self) -> dict[str, Any]:
        return self._quantity()._wire()

    def _support(self) -> dict[str, Any]:
        return {"range": {"lo": self.lo, "hi": self.hi}}


@dataclass(frozen=True, slots=True)
class Categorical:
    """A categorical input with a finite declared level set."""

    name: str
    levels: Iterable[str]
    units: str = "category"
    role: str = "covariate"

    def __post_init__(self) -> None:
        if isinstance(self.levels, str):
            raise CausalTypeError("levels must be an iterable of level names")
        object.__setattr__(self, "levels", tuple(_name(x, "level") for x in self.levels))
        self._quantity()

    def _quantity(self) -> Quantity:
        return Quantity(self.name, units=self.units, role=self.role)

    def _wire(self) -> dict[str, Any]:
        return self._quantity()._wire()

    def _support(self) -> dict[str, Any]:
        return {"levels": {"levels": sorted(self.levels)}}


@dataclass(frozen=True, slots=True)
class Intercept:
    """The constant basis term ``1``."""

    def _wire(self) -> str:
        return "intercept"


@dataclass(frozen=True, slots=True)
class Linear:
    """The numeric quantity itself."""

    quantity: str

    def _wire(self) -> dict[str, Any]:
        return {"linear": {"quantity": _name(self.quantity, "quantity")}}


@dataclass(frozen=True, slots=True)
class Power:
    """The numeric quantity raised to ``degree`` (2 to 8)."""

    quantity: str
    degree: int

    def _wire(self) -> dict[str, Any]:
        if isinstance(self.degree, bool) or not isinstance(self.degree, int):
            raise CausalTypeError("degree must be an integer")
        return {"power": {"quantity": _name(self.quantity, "quantity"), "degree": self.degree}}


@dataclass(frozen=True, slots=True)
class OneHot:
    """The indicator that a categorical quantity equals ``level``."""

    quantity: str
    level: str

    def _wire(self) -> dict[str, Any]:
        return {
            "one_hot": {
                "quantity": _name(self.quantity, "quantity"),
                "level": _name(self.level, "level"),
            }
        }


Term = Intercept | Linear | Power | OneHot


@dataclass(frozen=True, slots=True)
class MaskRegion:
    """A query region the model must refuse: every condition holds.

    ``conditions`` maps a quantity to a closed numeric range ``(lo, hi)`` or to a set of
    categorical levels.
    """

    id: str
    conditions: Mapping[str, tuple[float, float] | Iterable[str]]

    def __post_init__(self) -> None:
        _name(self.id, "mask region id")
        if not isinstance(self.conditions, Mapping):
            raise CausalTypeError("conditions must be a mapping of quantity -> range or levels")

    def _wire(self) -> dict[str, Any]:
        wired = []
        for quantity in sorted(self.conditions):
            within = self.conditions[quantity]
            if isinstance(within, tuple) and len(within) == 2 and not isinstance(within[0], str):
                support: dict[str, Any] = {
                    "range": {"lo": _real(within[0], "lo"), "hi": _real(within[1], "hi")}
                }
            elif isinstance(within, str) or not isinstance(within, Iterable):
                raise CausalTypeError("a condition is a (lo, hi) range or a set of levels")
            else:
                support = {"levels": {"levels": sorted({_name(x, "level") for x in within})}}
            wired.append({"quantity": _name(quantity, "quantity"), "within": support})
        return {"id": self.id, "conditions": wired}


@dataclass(frozen=True, slots=True)
class PointEstimate:
    """A point prediction with its model-based standard error.

    ``standard_error`` is ``sqrt(phi' V phi)`` from the stored covariance. It is **not** an
    interval and not a coverage claim (``calibration == "calibration unmeasured"``), and it
    carries no extrapolation or misspecification uncertainty. ``export_identity`` names the
    export that produced the numbers.
    """

    point: float
    standard_error: float
    se_basis: str
    calibration: str
    export_identity: str

    @property
    def se(self) -> float:
        """Alias of :attr:`standard_error`."""
        return self.standard_error


def _point(payload: Mapping[str, Any]) -> PointEstimate:
    return PointEstimate(
        point=float(payload["point"]),
        standard_error=float(payload["model_based_se"]),
        se_basis=payload["se_basis"],
        calibration=payload["calibration"],
        export_identity=payload["export_identity"],
    )


def _query(query: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(query, Mapping):
        raise CausalTypeError("a query is a mapping of quantity -> number or level")
    wired: dict[str, Any] = {}
    for name, value in query.items():
        _name(name, "quantity")
        if isinstance(value, str):
            wired[name] = {"level": value}
        elif isinstance(value, bool) or not isinstance(
            value, int | float | np.floating | np.integer
        ):
            raise CausalTypeError(f"query value for {name!r} must be a number or a level string")
        else:
            number = float(value)
            # JSON cannot carry a non-finite number; ``null`` is the sentinel the export refuses.
            wired[name] = {"number": number if math.isfinite(number) else None}
    return wired


@dataclass(frozen=True, slots=True, eq=False)
class CompactExport:
    """A validated, sealed compact export.

    ``identity`` is the BLAKE3 identity a consumer retains out-of-band; ``body`` is the stored,
    canonicalised body (inputs, terms, coefficients, covariance, mask, scope) as plain data.
    """

    identity: str
    body: Mapping[str, Any]
    artifact: bytes = field(repr=False)

    @classmethod
    def build(
        cls,
        *,
        response: Quantity,
        inputs: Sequence[Numeric | Categorical],
        terms: Sequence[Term],
        coefficients: Any,
        covariance: Any,
        mask: Sequence[MaskRegion] = (),
        artifact_id: str = _DEFAULT_ARTIFACT_ID,
    ) -> CompactExport:
        """Validate, canonicalise and seal an export of a fitted coefficient vector.

        ``coefficients`` and the row-major ``len(terms) x len(terms)`` ``covariance`` are aligned
        with ``terms`` as given; the stored order is canonical. Raises
        :class:`CompactExportRefusal` (``invalid_argument``) for a malformed term, support, mask,
        quantity or covariance (non-finite, asymmetric, a negative variance, or an off-diagonal
        beyond the Cauchy-Schwarz bound).
        """
        if not isinstance(response, Quantity):
            raise CausalTypeError("response must be a Quantity")
        for item in inputs:
            if not isinstance(item, Numeric | Categorical):
                raise CausalTypeError("inputs are Numeric or Categorical quantities")
        try:
            beta = np.asarray(coefficients, dtype=np.float64).ravel()
            matrix = np.asarray(covariance, dtype=np.float64).ravel()
        except (TypeError, ValueError) as error:
            raise CausalTypeError("coefficients and covariance must be numeric arrays") from error
        if not (np.all(np.isfinite(beta)) and np.all(np.isfinite(matrix))):
            raise CompactExportRefusal(
                {
                    "code": "invalid_argument",
                    "stage": "build",
                    "detail": "compact_export.non_finite_value",
                    "offending": "coefficients_or_covariance",
                    "message": "coefficients and covariance must be finite",
                }
            )
        spec = {
            "response": response._wire(),
            "inputs": [{"quantity": i._wire(), "support": i._support()} for i in inputs],
            "terms": [t._wire() for t in terms],
            "coefficients": beta.tolist(),
            "covariance": matrix.tolist(),
            "mask": [m._wire() for m in mask],
        }
        summary, artifact, refusal = _build(
            json.dumps(spec, allow_nan=False), _name(artifact_id, "artifact_id")
        )
        if refusal is not None:
            raise CompactExportRefusal(json.loads(refusal))
        if summary is None or artifact is None:  # pragma: no cover - the native contract
            raise CausalValueError("the native export returned neither a result nor a refusal")
        parsed = json.loads(summary)
        return cls(identity=parsed["identity"], body=parsed["body"], artifact=bytes(artifact))

    @classmethod
    def consume(cls, artifact: bytes, *, expected_identity: str) -> CompactExport:
        """Independently verify ``artifact`` against the identity the caller retained.

        Decodes with bounded sizes, recomputes the digests and identity from the stored fields
        only and requires ``expected_identity``; raises :class:`CompactExportRefusal` for a
        tampered coefficient, covariance, term, support, mask or scope (even when the container
        was resealed), a different identity, an oversized claim or an unknown version.
        """
        if not isinstance(artifact, bytes | bytearray | memoryview):
            raise CausalTypeError("artifact must be bytes")
        _name(expected_identity, "expected_identity")
        data = bytes(artifact)
        summary, refusal = _consume(data, expected_identity)
        if refusal is not None:
            raise CompactExportRefusal(json.loads(refusal))
        if summary is None:  # pragma: no cover - the native contract
            raise CausalValueError("the native verifier returned neither a result nor a refusal")
        parsed = json.loads(summary)
        return cls(identity=parsed["identity"], body=parsed["body"], artifact=data)

    def export(self) -> bytes:
        """The checksummed ``compact_runtime_export_v1`` artifact."""
        return self.artifact

    def evaluate(self, query: Mapping[str, Any]) -> PointEstimate:
        """Point prediction and model-based standard error at ``query``.

        ``query`` maps every declared input to a number (numeric inputs) or a level string
        (categorical inputs). The artifact is re-verified against :attr:`identity` first. Raises
        :class:`CompactExportRefusal` (``cell_not_licensed``) for a query that is out of
        support, in a masked region, missing or has an unknown or mistyped quantity, or is not
        finite; nothing is extrapolated.
        """
        result = self.evaluate_many([query])[0]
        if isinstance(result, CompactExportRefusal):
            raise result
        return result

    def evaluate_many(
        self, queries: Sequence[Mapping[str, Any]]
    ) -> list[PointEstimate | CompactExportRefusal]:
        """Evaluate several queries after one verification.

        Each answer is a :class:`PointEstimate` or the :class:`CompactExportRefusal` that
        query earned (returned, not raised); a verification failure raises.
        """
        wired = [_query(q) for q in queries]
        results, refusal = _evaluate(self.artifact, self.identity, json.dumps(wired))
        if refusal is not None:
            raise CompactExportRefusal(json.loads(refusal))
        if results is None:  # pragma: no cover - the native contract
            raise CausalValueError("the native evaluator returned neither a result nor a refusal")
        answers: list[PointEstimate | CompactExportRefusal] = []
        for item in json.loads(results):
            if "ok" in item:
                answers.append(_point(item["ok"]))
            else:
                answers.append(CompactExportRefusal(item["refusal"]))
        return answers

    def to_dict(self) -> dict[str, Any]:
        """A plain, JSON-serializable summary (no artifact bytes)."""
        return {"identity": self.identity, "body": dict(self.body)}
