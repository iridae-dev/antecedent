"""Feature-build candidate lifecycle for full joint transport posteriors.

Default release wheels keep both producers and consumers frozen. These declarations
prepare a real success lifecycle without measuring or activating calibration.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from .. import _native
from .._data import to_f64
from .._measured_inference import MeasuredInference, _level, _production_limits
from ..errors import CausalTypeError, CausalUnsupportedError, CausalValueError
from ._candidate_data import detached, freeze
from ._impl import TransportIdentification


@dataclass(frozen=True, slots=True)
class JointTransportIdentity:
    snapshot_digest: str
    datum_ids: tuple[str, ...]

    def _wire(self) -> dict[str, Any]:
        if not isinstance(self.snapshot_digest, str) or not self.snapshot_digest.strip():
            raise CausalValueError("snapshot_digest must be nonempty")
        if (
            not isinstance(self.datum_ids, Sequence)
            or isinstance(self.datum_ids, (str, bytes))
            or any(not isinstance(v, str) or not v.strip() for v in self.datum_ids)
        ):
            raise CausalValueError("datum_ids must be nonempty strings")
        if not self.datum_ids or len(set(self.datum_ids)) != len(self.datum_ids):
            raise CausalValueError("datum_ids must be nonempty and unique")
        return {"snapshot_digest": self.snapshot_digest, "datum_ids": list(self.datum_ids)}


@dataclass(frozen=True, slots=True)
class GaussianTransportPrior:
    """A full Gaussian coefficient block; covariance is a square matrix.

    ``bank_id`` and ``consumed`` declare every observation used to construct a
    prior bank. Missing prior/data disjointness cannot be supplied by the posterior.
    """

    mean: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]
    bank_id: str | None = None
    consumed: tuple[JointTransportIdentity, ...] = ()

    def _wire(self) -> dict[str, Any]:
        for name, value in (("mean", self.mean), ("covariance", self.covariance)):
            if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
                raise CausalTypeError(f"prior {name} must be a sequence")
        size = len(self.mean)
        if not 1 <= size <= 256 or len(self.covariance) != size:
            raise CausalValueError("prior covariance must be square and match 1..256 means")
        for row in self.covariance:
            if isinstance(row, (str, bytes)) or not isinstance(row, Sequence) or len(row) != size:
                raise CausalValueError("prior covariance must be square and match 1..256 means")
        try:
            mean = tuple(float(v) for v in self.mean)
            rows = tuple(tuple(float(v) for v in row) for row in self.covariance)
        except (TypeError, ValueError, OverflowError) as error:
            raise CausalValueError(
                "prior means and covariance must contain finite numbers"
            ) from error
        if (
            not mean
            or len(mean) > 256
            or len(rows) != len(mean)
            or any(len(row) != len(mean) for row in rows)
        ):
            raise CausalValueError("prior covariance must be square and match 1..256 means")
        if self.bank_id is None and self.consumed:
            raise CausalValueError("consumed observations need a named prior bank")
        provenance: Any = (
            "declared"
            if self.bank_id is None
            else {"bank": {"bank_id": self.bank_id, "consumed": [v._wire() for v in self.consumed]}}
        )
        return {
            "mean": list(mean),
            "covariance": [v for row in rows for v in row],
            "provenance": provenance,
        }


@dataclass(frozen=True, slots=True)
class JointTransportPriors:
    invariant: GaussianTransportPrior
    varying: GaussianTransportPrior


@dataclass(frozen=True, slots=True)
class JointTransportSource:
    id: str
    population: str
    identity: JointTransportIdentity
    data: Mapping[str, Any]
    noise_variance: float


@dataclass(frozen=True, slots=True)
class JointTransportTarget:
    population: str
    identity: JointTransportIdentity
    data: Mapping[str, Any]


@dataclass(frozen=True, slots=True, init=False)
class JointTransportPosterior:
    """Numerically replayable candidate posterior; calibration is unmeasured.

    All rows are aligned across the original complete joint parameter/effect law.
    A successful internal feature build is not a released interval license.
    """

    kind: Literal["gaussian", "learned_gaussian"]
    _json: str = field(repr=False)
    _bytes: bytes = field(repr=False)
    _body: Mapping[str, Any] = field(init=False, repr=False)

    def __init__(self) -> None:
        raise CausalTypeError("use joint transport producers or consume_joint_transport_posterior")

    @classmethod
    def _from_native(
        cls, kind: Literal["gaussian", "learned_gaussian"], payload: str, artifact: bytes
    ) -> JointTransportPosterior:
        result = object.__new__(cls)

        object.__setattr__(result, "kind", kind)
        object.__setattr__(result, "_body", freeze(json.loads(payload)))
        object.__setattr__(result, "_bytes", bytes(artifact))
        object.__setattr__(result, "_json", "")
        return result

    @property
    def calibration(self) -> Literal["unmeasured"]:
        return "unmeasured"

    @property
    def release_status(self) -> Literal["candidate_only"]:
        return "candidate_only"

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(self._body["result"]["parameter_names"])

    @property
    def posterior_mean(self) -> tuple[float, ...]:
        return tuple(self._body["result"]["posterior_mean"])

    @property
    def posterior_covariance(self) -> tuple[tuple[float, ...], ...]:
        values = self._body["result"]["posterior_covariance"]
        n = len(self.posterior_mean)
        return tuple(tuple(values[i * n : (i + 1) * n]) for i in range(n))

    @property
    def target_effect_mean(self) -> float:
        return float(self._body["result"]["target_effect_mean"])

    @property
    def target_effect_variance(self) -> float:
        return float(self._body["result"]["target_effect_variance"])

    @property
    def draws(self) -> np.ndarray:
        value = self._body["result"]["draws"]
        rows = np.asarray(value["values"], dtype=np.float64).reshape(
            value["n_draws"], len(value["names"])
        )
        rows.setflags(write=False)
        return rows

    @property
    def draw_names(self) -> tuple[str, ...]:
        return tuple(self._body["result"]["draws"]["names"])

    @property
    def diagnostics(self) -> dict[str, Any]:
        return dict(detached(self._body["result"]["diagnostics"]))

    def to_dict(self) -> dict[str, Any]:
        """Copy of original full model, covariance, priors, identities and proof record."""

        return dict(detached(self._body))

    def export(self) -> bytes:
        return self._bytes

    def expectation(self) -> dict[str, str]:
        body = self._body
        return {"premises_digest": body["premises_digest"], "data_digest": body["data_digest"]}


def consume_joint_transport_posterior(
    data: bytes,
    *,
    kind: Literal["gaussian", "learned_gaussian"],
    expected_identity: Mapping[str, str],
) -> JointTransportPosterior:
    """Fresh native re-fit under retained independent model/data identity expectations."""
    consumer = getattr(_native, "consume_joint_transport_candidate", None)
    if consumer is None:
        raise CausalUnsupportedError(
            "joint_transport.route_frozen: candidate lifecycle requires internal feature build",
            reason_code="cell_not_licensed",
        )
    if kind not in ("gaussian", "learned_gaussian"):
        raise CausalValueError("kind must be gaussian or learned_gaussian")
    text = consumer(
        data, json.dumps(dict(expected_identity), allow_nan=False), kind == "learned_gaussian"
    )
    return JointTransportPosterior._from_native(kind, text, bytes(data))


def _columns(data: Mapping[str, Any], names: Sequence[str]) -> list[list[float]]:
    if not isinstance(data, Mapping):
        raise CausalTypeError("joint transport data must be a named mapping")
    missing = set(names) - set(data)
    if missing:
        raise CausalValueError(f"missing joint transport columns: {sorted(missing)}")
    try:
        lengths = [len(data[name]) for name in names]
    except TypeError as error:
        raise CausalValueError("joint transport data must be columns") from error
    if lengths and (max(lengths) > 200_000 or any(n != lengths[0] for n in lengths)):
        raise CausalValueError("joint transport columns must align within the row bound")
    columns = [to_f64(data[name], name=name) for name in names]
    if columns and any(len(column) != len(columns[0]) for column in columns):
        raise CausalValueError("joint transport columns must align")
    return [column.tolist() for column in columns]


def _request_json(
    *,
    sources: Sequence[JointTransportSource | Mapping[str, Any]],
    target: JointTransportTarget | Mapping[str, Any] | None,
    features: Sequence[str],
    identification: TransportIdentification,
    priors: JointTransportPriors,
    treatment: str,
    outcome: str,
    varying: str,
    sharing: str,
    dependence: str,
    basis_degree: int,
    draws: int,
    seed: int,
    learned: bool,
    max_unsupported_mass: float,
    conflict_z_threshold: float,
) -> str:
    if not isinstance(identification, TransportIdentification) or not isinstance(
        identification._native, _native.TransportIdentificationResult
    ):
        raise CausalTypeError("identification must retain its original native transport proof")
    if not isinstance(priors, JointTransportPriors):
        raise CausalTypeError("priors must be JointTransportPriors with complete Gaussian blocks")
    if not all(isinstance(source, JointTransportSource) for source in sources) or not isinstance(
        target, JointTransportTarget
    ):
        raise CausalTypeError(
            "candidate sources/target must be JointTransportSource/JointTransportTarget"
        )
    if not isinstance(treatment, str) or not isinstance(outcome, str) or treatment == outcome:
        raise CausalValueError("treatment and outcome must be distinct named columns")
    if not isinstance(priors.invariant, GaussianTransportPrior) or not isinstance(
        priors.varying, GaussianTransportPrior
    ):
        raise CausalTypeError("priors must contain GaussianTransportPrior blocks")
    if len(sources) > 64 or len(features) > 32:
        raise CausalValueError("joint transport exceeds source/feature bounds")
    total_rows = 0
    for source in sources:
        assert isinstance(source, JointTransportSource)
        if not isinstance(source.identity, JointTransportIdentity) or not isinstance(
            source.data, Mapping
        ):
            raise CausalTypeError("source needs typed identity and named data")
        if outcome not in source.data:
            raise CausalValueError(f"missing joint transport outcome {outcome!r}")
        try:
            total_rows += len(source.data[outcome])
        except TypeError as error:
            raise CausalValueError("source outcome must be a column") from error
        if total_rows > 200_000:
            raise CausalValueError("joint transport exceeds source row bound")
    if not isinstance(target.identity, JointTransportIdentity) or not isinstance(
        target.data, Mapping
    ):
        raise CausalTypeError("target needs typed identity and named data")
    if len(target.identity.datum_ids) > 200_000:
        raise CausalValueError("joint transport exceeds target row bound")
    source_wires = []
    for source in sources:
        assert isinstance(source, JointTransportSource)
        treatment_values, outcome_values, *covariates = _columns(
            source.data, [treatment, outcome, *features]
        )
        if any(value not in (0.0, 1.0) for value in treatment_values):
            raise CausalValueError("joint transport treatment must use original binary 0/1 levels")
        source_wires.append(
            {
                "id": source.id,
                "population": source.population,
                "identity": source.identity._wire(),
                "treatment": [v == 1.0 for v in treatment_values],
                "outcome": outcome_values,
                "covariates": covariates,
                "noise_variance": source.noise_variance,
            }
        )
    covariates = _columns(target.data, features)
    rows = len(covariates[0]) if covariates else len(target.identity.datum_ids)
    request = {
        "features": list(features),
        "treatment": treatment,
        "outcome": outcome,
        "varying": varying,
        "sharing": sharing,
        "dependence": dependence,
        "invariant": priors.invariant._wire(),
        "varying_prior": priors.varying._wire(),
        "sources": source_wires,
        "target": {
            "population": target.population,
            "identity": target.identity._wire(),
            "rows": rows,
            "covariates": covariates,
        },
        "draws": draws,
        "seed": seed,
        "basis_degree": basis_degree,
        "max_unsupported_mass": max_unsupported_mass,
        "conflict_z_threshold": conflict_z_threshold,
    }
    try:
        text = json.dumps(request, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise CausalValueError("joint transport declarations must be finite JSON values") from error
    return text


def _candidate(**kwargs: Any) -> JointTransportPosterior:
    """Explicit internal candidate; retains its original unmeasured wire standing."""
    text = _request_json(**kwargs)
    body, artifact = _native.joint_transport_candidate(
        kwargs["identification"]._native, text, kwargs["learned"]
    )
    return JointTransportPosterior._from_native(
        "learned_gaussian" if kwargs["learned"] else "gaussian", body, bytes(artifact)
    )


def _measured(
    *,
    level: float = 0.95,
    memory_limit_bytes: int | None = None,
    cancel: _native.CancellationToken | None = None,
    **kwargs: Any,
) -> MeasuredInference:
    _production_limits(memory_limit_bytes, cancel)
    _level(level)
    text = _request_json(**kwargs)
    producer = getattr(_native, "joint_transport_measured", None)
    if producer is None:
        raise CausalUnsupportedError(
            "joint_transport.route_frozen: measured native authority unavailable",
            reason_code="cell_not_licensed",
        )
    native = producer(
        kwargs["identification"]._native,
        text,
        kwargs["learned"],
        level=level,
        memory_limit_bytes=memory_limit_bytes,
        cancel=cancel,
    )
    return MeasuredInference._from_native(native)


__all__ = [
    "GaussianTransportPrior",
    "JointTransportIdentity",
    "JointTransportPriors",
    "JointTransportSource",
    "JointTransportTarget",
    "JointTransportPosterior",
    "consume_joint_transport_posterior",
]
