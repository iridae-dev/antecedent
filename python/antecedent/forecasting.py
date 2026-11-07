"""Execute a registered external forecast at one information origin.

Each call supplies only one origin's history. The provider may still retain information
from prior calls or training, so that lineage is unknown unless declared explicitly.
Provider execution and exact-request trust are owned by the existing registry.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from . import predictive_assessment as pa
from ._digest import canonical_digest
from .errors import CausalValueError
from .extensibility import (
    ProviderCapability,
    ProviderRegistry,
    ProviderResult,
    _digest,
    _request_digest,
    _spec_evidence,
)


@dataclass(frozen=True, slots=True)
class ForecastOrigin:
    """Information available at an integer origin, including declared future covariates.

    Covariate availability is caller-attested. History times must strictly increase
    and may not extend beyond the origin.
    """

    origin: int
    history_times: Sequence[int]
    history_values: Sequence[Sequence[float]]
    history_keys: Sequence[str]
    future_covariates: Sequence[Sequence[float]] = ()

    def _wire(self, horizon: int) -> dict[str, Any]:
        if type(self.origin) is not int or self.origin < 0:
            raise CausalValueError("forecast origin must be a nonnegative integer")
        times = pa._counts(self.history_times, "history times")
        if (
            not times
            or len(times) != len(self.history_values)
            or len(times) != len(self.history_keys)
        ):
            raise CausalValueError("forecast history needs aligned nonempty times, values and keys")
        if any(a >= b for a, b in zip(times, times[1:], strict=False)) or times[-1] > self.origin:
            raise CausalValueError("forecast history times must increase and not exceed the origin")
        history = [pa._floats(row, "history values") for row in self.history_values]
        if not history[0] or any(len(row) != len(history[0]) for row in history):
            raise CausalValueError("forecast history must be a rectangular nonempty matrix")
        future = [pa._floats(row, "future covariates") for row in self.future_covariates]
        if future and (len(future) != horizon or any(len(row) != len(future[0]) for row in future)):
            raise CausalValueError("future covariates need one equally sized row per horizon step")
        history_keys = [pa._key(key) for key in self.history_keys]
        if len(set(history_keys)) != len(history_keys):
            raise CausalValueError("forecast history observation keys must be distinct")
        return {
            "origin": self.origin,
            "history_times": times,
            "history_values": history,
            "history_keys": history_keys,
            "future_covariates": future,
        }


@dataclass(frozen=True, slots=True)
class ForecastResult:
    """One origin's validated provider result, source and optional predictive law."""

    execution: ProviderResult
    source: pa.LawSource
    target: pa.Target
    origin: int
    law: pa.Law | None
    fit: pa.FitDeclaration

    def predictor(self, name: str) -> pa.Predictor:
        """Build a single-case predictor; point-only output has no predictive law."""
        if self.law is None:
            raise CausalValueError("a point forecast alone is not a predictive law")
        return pa.Predictor(name, [self.law], self.target, self.source, self.fit)

    def validation_design(self) -> pa.ValidationDesign:
        """The rolling-origin declaration for this forecast."""
        assert self.target.horizon_steps is not None
        return pa.ValidationDesign.rolling_origin([self.origin], self.target.horizon_steps)


@dataclass(frozen=True, slots=True)
class RollingForecast:
    """Origin-scoped forecasts and their individually retained provider receipts."""

    results: tuple[ForecastResult, ...]
    source: pa.LawSource

    def predictor(self, name: str) -> pa.Predictor:
        """Build a case-aligned predictor with one fit declaration per origin."""
        laws: list[pa.Law] = []
        for result in self.results:
            if result.law is None:
                raise CausalValueError("point forecasts alone are not predictive laws")
            laws.append(result.law)
        return pa.Predictor(
            name,
            laws,
            self.results[0].target,
            self.source,
            pa.FitDeclaration.per_case([result.fit for result in self.results]),
        )

    def validation_design(self) -> pa.ValidationDesign:
        """Return the declared rolling origins and shared horizon."""
        horizon = self.results[0].target.horizon_steps
        assert horizon is not None
        return pa.ValidationDesign.rolling_origin(
            [result.origin for result in self.results], horizon
        )


def rolling(
    registry: ProviderRegistry,
    provider: str,
    origins: Sequence[ForecastOrigin],
    *,
    target: pa.Target,
    kind: Literal["point", "gaussian", "samples", "joint_samples"],
    prior_training_keys: Sequence[Sequence[str] | None] | None = None,
    prior_training_times: Sequence[Sequence[int] | None] | None = None,
    configuration: Mapping[str, Any] | None = None,
) -> RollingForecast:
    """Run each increasing origin independently and bind its receipt to a batch identity.

    The aggregate source is conservatively externally attested; exact-request trust
    remains on each individual result. The batch identity hashes those identities and
    their host-sealed receipts. Omitted prior lineage remains unknown for every origin.
    """
    from . import _native

    horizon = target.horizon_steps
    if type(horizon) is not int or horizon < 1:
        raise CausalValueError("forecast target needs a positive horizon")
    if not origins:
        raise CausalValueError("rolling forecast origins must be nonempty and strictly increasing")
    cases = [origin._wire(horizon) for origin in origins]
    if any(a.origin >= b.origin for a, b in zip(origins, origins[1:], strict=False)):
        raise CausalValueError("rolling forecast origins must be nonempty and strictly increasing")
    if prior_training_keys is not None and len(prior_training_keys) != len(origins):
        raise CausalValueError("prior training lineage must align with forecast origins")
    if prior_training_times is not None and len(prior_training_times) != len(origins):
        raise CausalValueError("prior training times must align with forecast origins")
    fits = [None] * len(origins) if prior_training_keys is None else prior_training_keys
    times = [None] * len(origins) if prior_training_times is None else prior_training_times
    known_times: dict[str, int] = {}
    for origin, case, fit, fit_times in zip(origins, cases, fits, times, strict=True):
        if fit is None:
            if fit_times is not None:
                raise CausalValueError("prior training times require prior training keys")
        else:
            prior_keys = [pa._key(key) for key in fit]
            prior_times = pa._counts(() if fit_times is None else fit_times, "prior training times")
            if len(prior_keys) != len(prior_times) or any(
                time > origin.origin for time in prior_times
            ):
                raise CausalValueError(
                    "prior training times must align with keys and not exceed the origin"
                )
            for key, time in zip(prior_keys, prior_times, strict=True):
                if key in known_times and known_times[key] != time:
                    raise CausalValueError(
                        "an observation key cannot have different times across origins"
                    )
                known_times[key] = time
        for key, time in zip(case["history_keys"], case["history_times"], strict=True):
            if key in known_times and known_times[key] != time:
                raise CausalValueError(
                    "an observation key cannot have different times across origins"
                )
            known_times[key] = time
    collected: list[ForecastResult] = []
    previous_history: dict[str, int] = {}
    for origin, case, fit, fit_times in zip(origins, cases, fits, times, strict=True):
        declared = None if fit is None else [*fit, *previous_history]
        declared_times = (
            None
            if fit is None
            else [*([] if fit_times is None else fit_times), *previous_history.values()]
        )
        collected.append(
            forecast(
                registry,
                provider,
                origin,
                target=target,
                kind=kind,
                prior_training_keys=declared,
                prior_training_times=declared_times,
                configuration=configuration,
            )
        )
        previous_history.update(zip(case["history_keys"], case["history_times"], strict=True))
    results = tuple(collected)
    identities = [result.source.wire["identity"] for result in results]
    versions = {identity["version_id"] for identity in identities}
    if len(versions) != 1:
        raise CausalValueError("provider specification changed across forecast origins")

    def receipt_digest(result: ForecastResult) -> str:
        artifact = result.execution.host_artifact
        if artifact is None:
            raise CausalValueError("forecast execution did not retain a host receipt")
        return str(_native.blake3_digest_hex(artifact))

    manifest = [
        {
            "identity": identity,
            "trust": result.execution.trust.value,
            "receipt": receipt_digest(result),
        }
        for identity, result in zip(identities, results, strict=True)
    ]
    digest = canonical_digest("antecedent.forecast.origin_manifest.v1", manifest)
    source = pa.LawSource.provider(
        provider_id=provider,
        object_id="forecast_origin_manifest",
        version_id=versions.pop(),
        snapshot_id=digest,
        request_id=digest,
        trust="externally_attested",
    )
    return RollingForecast(results, source)


def forecast(
    registry: ProviderRegistry,
    provider: str,
    origin: ForecastOrigin,
    *,
    target: pa.Target,
    kind: Literal["point", "gaussian", "samples", "joint_samples"],
    prior_training_keys: Sequence[str] | None = None,
    prior_training_times: Sequence[int] | None = None,
    configuration: Mapping[str, Any] | None = None,
) -> ForecastResult:
    """Run one forecast and retain the exact request's trust and fit declaration.

    Outputs are point ``(1,)``, Gaussian ``(2,)`` (mean, standard deviation), samples
    ``(draws,)``, or joint trajectories ``(draws, dimension)``. ``prior_training_keys``
    declares the complete prior training and tuning lineage; omitted means unknown,
    including for a frozen pretrained model. Nonempty prior lineage requires aligned
    observation times no later than the origin. History keys are included conservatively.
    """
    horizon = target.horizon_steps
    if type(horizon) is not int or horizon < 1:
        raise CausalValueError("forecast target needs a positive horizon")
    if kind not in ("point", "gaussian", "samples", "joint_samples"):
        raise CausalValueError("unknown forecast output kind")
    spec = registry._entry(provider)[0].spec
    if spec.query_family != "forecast":
        raise CausalValueError("provider must declare query_family='forecast'")
    required = (
        {ProviderCapability.SAMPLE}
        if kind in ("samples", "joint_samples")
        else {ProviderCapability.MEAN}
    )
    if kind == "gaussian":
        required.add(ProviderCapability.COVARIANCE)
    if not required.issubset(spec.capabilities):
        raise CausalValueError("provider does not declare the forecast capabilities required")
    if kind != "point" and spec.uncertainty_semantics == "point_only":
        raise CausalValueError("probabilistic forecasts require declared uncertainty semantics")
    case = origin._wire(horizon)
    if prior_training_keys is None:
        if prior_training_times is not None:
            raise CausalValueError("prior training times require prior training keys")
        lineage = None
    else:
        keys = [pa._key(key) for key in prior_training_keys]
        times = pa._counts(
            () if prior_training_times is None else prior_training_times, "prior training times"
        )
        if len(keys) != len(times) or any(time > origin.origin for time in times):
            raise CausalValueError(
                "prior training times must align with keys and not exceed the origin"
            )
        lineage = list(zip(keys, times, strict=True))
        lineage.extend(zip(case["history_keys"], case["history_times"], strict=True))
        seen: dict[str, int] = {}
        for key, time in lineage:
            if key in seen and seen[key] != time:
                raise CausalValueError("an observation key cannot have different training times")
            seen[key] = time
    fit = (
        pa.FitDeclaration.undeclared("prior training lineage not declared")
        if lineage is None
        else pa.FitDeclaration.fitted_on([entry[0] for entry in lineage])
    )
    request = {
        "operation": "forecast",
        "version": 1,
        "target": target.to_wire(),
        "kind": kind,
        "case": case,
        "training_lineage": lineage,
        "configuration": copy.deepcopy(dict(configuration or {})),
    }
    spec_digest = _digest("spec", _spec_evidence(spec))
    request_digest = _request_digest(request)
    execution = registry.execute(provider, request)
    if (
        _request_digest(request) != request_digest
        or _digest("spec", _spec_evidence(registry._entry(provider)[0].spec)) != spec_digest
    ):
        raise CausalValueError(
            "forecast request or provider specification changed during execution"
        )
    value = execution.estimate
    law: pa.Law | None
    if kind == "point" and value.shape == (1,):
        law = None
    elif kind == "gaussian" and value.shape == (2,) and value[1] > 0:
        law = pa.Gaussian(float(value[0]), float(value[1]))
    elif kind == "samples" and value.ndim == 1 and value.size:
        law = pa.EmpiricalSample(value.tolist())
    elif (
        kind == "joint_samples"
        and value.ndim == 2
        and all(value.shape)
        and value.shape[1] % horizon == 0
    ):
        law = pa.JointSample(value.tolist())
    else:
        raise CausalValueError("forecast output shape or Gaussian standard deviation is invalid")
    if execution.uncertainty is not None:
        raise CausalValueError(
            "forecast outputs must declare all distribution parameters in estimate"
        )
    source = pa.LawSource.provider(
        provider_id=provider,
        object_id=spec.query_family,
        version_id=spec_digest,
        snapshot_id=request_digest,
        request_id=request_digest,
        trust=execution.trust.value,
    )
    return ForecastResult(execution, source, target, origin.origin, law, fit)
