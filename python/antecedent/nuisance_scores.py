"""Nuisance scores of a prepared study, retained as artifacts that are refit on load.

A prepared average-effect study knows its treatment and its certified adjustment set.
:func:`treatment_score_posterior` explicitly fits and returns the Laplace
posterior of the logistic treatment model on exactly those columns under a stated
``N(0, prior_sd**2)`` prior on every coefficient, the intercept included. It is a posterior of a
**nuisance**, never of an effect: the scores, their spread and the probability of falling outside
a positivity band are about the treatment model alone.

A posterior of the *weighting effect* cannot be built from these draws. Under a likelihood that
factors into a treatment model and an outcome model with independent priors, the propensity
coefficients are ancillary for the average effect, so propensity draws pushed through the
weighting identity carry propensity uncertainty alone. Asking for a Bayesian
``propensity.weighting`` estimator raises ``CausalUnsupportedError`` with
``reason_code="route_not_supported"`` naming the missing outcome law and the posteriors that do
answer that effect (the Bayesian-bootstrap posterior of the cross-fitted ridge-orthogonal score,
or the Bayesian g-computation posterior of an outcome model).

The product is unlicensed and its calibration is unmeasured. Its artifact embeds the design it was
fitted on; :func:`load_artifact` refits from it and refuses any edited, swapped or resealed one.
"""

from __future__ import annotations

import json
from collections.abc import ItemsView, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite
from typing import TYPE_CHECKING, Any

from ._artifact_protocol import (
    ArtifactAcceptance,
    ArtifactExpectation,
    ExportedArtifact,
    artifact_consumer,
    artifact_export,
    container_bytes,
)
from ._freeze import freeze, thaw
from ._native import _ArrayFit as _NativeArrayFit
from ._native import consume_array_fit_artifact as _consume
from .errors import CausalTypeError, CausalUnsupportedError, CausalValueError
from .execution import Execution
from .outcomes import Answer, Outcome, capture_outcome
from .results._report import InspectionReport, SlotModel

if TYPE_CHECKING:
    from .prepared import PreparedAnalysis

__all__ = [
    "NuisanceScoreProduct",
    "load_artifact",
    "temporal_treatment_score",
    "treatment_score_posterior",
]


def _positive(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CausalTypeError(f"{name} must be a real number")
    if not isfinite(value) or value <= 0:
        raise CausalValueError(f"{name} must be finite and positive")


def _draw_count(draws: int) -> None:
    if isinstance(draws, bool) or not isinstance(draws, int):
        raise CausalTypeError("draws must be an integer")
    if draws <= 0:
        raise CausalValueError("draws must be positive")


def _context(study: PreparedAnalysis[Any], execution: Execution | None) -> tuple[Any, Execution]:
    from .prepared import PreparedAnalysis

    if not isinstance(study, PreparedAnalysis):
        raise CausalTypeError("nuisance scoring requires one immutable prepared study")
    policy = study.execution if execution is None else execution
    if not isinstance(policy, Execution):
        raise CausalTypeError("execution must be an Execution declaration")
    if policy.on_stage is not None:
        raise CausalUnsupportedError(
            "nuisance scores do not expose causal estimation stages",
            reason_code="stage_stream_unavailable",
        )
    if study._refused is not None:
        assert study._refused.refusal is not None
        study._refused.refusal.raise_error()
    return study._engine._native, policy


def treatment_score_posterior(
    study: PreparedAnalysis[Any],
    *,
    prior_sd: float = 2.0,
    draws: int = 4000,
    clip: float = 0.05,
    summary_rows: Sequence[int] = (),
    execution: Execution | None = None,
) -> Outcome[NuisanceScoreProduct]:
    """Fit a nuisance posterior on the retained static certified adjustment design.

    The prior is ``N(0, prior_sd**2)`` on every logit coefficient, including the
    intercept. The likelihood uses the study's analysis mask and declared
    missing-data policy on treatment, outcome and certified adjustment variables;
    the outcome is not a treatment-model predictor. Unsupported sampling weights
    refuse instead of silently fitting an unweighted population.

    ``summary_rows`` refers to original physical row positions, including valid
    held-out rows excluded from fitting. A summary needs observed adjustment
    predictors but does not require that row's treatment or outcome. These are
    treatment probabilities, never a causal-effect posterior. The study remains unchanged.
    Execution supplies the seed and operational controls; its default seed is 1.
    Migration: the former per-method seed default was 0. Pass ``Execution(seed=0)``
    to reproduce that stream; scientific prior, draw and positivity defaults are unchanged.
    """
    _positive(prior_sd, "prior_sd")
    _positive(clip, "clip")
    if clip >= 0.5:
        raise CausalValueError("clip must be strictly below 0.5")
    _draw_count(draws)
    rows = list(summary_rows)
    for row in rows:
        if isinstance(row, bool) or not isinstance(row, int):
            raise CausalTypeError("summary rows must be integer positions")
        if row < 0:
            raise CausalValueError("summary rows must be nonnegative")
    context = capture_outcome(_context, study, execution)
    if context.refusal is not None:
        return Outcome(refusal=context.refusal)
    assert context.value is not None
    native, policy = context.value

    def fit() -> NuisanceScoreProduct:
        native_product = policy._call(
            native.treatment_score_posterior,
            prior_sd=prior_sd,
            draws=draws,
            seed=policy.seed,
            threads=policy.threads,
            clip=clip,
            summary_rows=rows,
        )
        return _product(native_product, policy)

    return capture_outcome(fit)


def temporal_treatment_score(
    study: PreparedAnalysis[Any],
    histories: Sequence[Sequence[float]],
    *,
    penalty: float | None = None,
    intercept_sd: float = 1.0,
    slope_sd: float = 1.0,
    draws: int = 4000,
    execution: Execution | None = None,
) -> Outcome[NuisanceScoreProduct]:
    """Fit scores on one independent-subject row per retained panel trajectory.

    Each subject contributes its earliest full lag window on identified adjustment
    lags. A penalty selects the ridge mode with unpenalized intercept and no
    posterior uncertainty. Otherwise independent Gaussian coefficient priors define
    a Laplace nuisance posterior. A single dependent series is scientifically refused.
    Neither product represents uncertainty about a causal effect.

    The seed comes from the retained or explicit Execution (default 1). To
    reproduce the former per-method default stream, use ``Execution(seed=0)``;
    scientific penalty, prior and draw defaults are unchanged.
    """
    if penalty is not None:
        _positive(penalty, "penalty")
    _positive(intercept_sd, "intercept_sd")
    _positive(slope_sd, "slope_sd")
    _draw_count(draws)
    rows = [list(history) for history in histories]
    for row in rows:
        for value in row:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise CausalTypeError("history values must be real numbers")
            if not isfinite(value):
                raise CausalValueError("history values must be finite")
    context = capture_outcome(_context, study, execution)
    if context.refusal is not None:
        return Outcome(refusal=context.refusal)
    assert context.value is not None
    native, policy = context.value

    def fit() -> NuisanceScoreProduct:
        native_product = policy._call(
            native.temporal_treatment_score,
            rows,
            penalty=penalty,
            intercept_sd=intercept_sd,
            slope_sd=slope_sd,
            draws=draws,
            seed=policy.seed,
            threads=policy.threads,
        )
        return _product(native_product, policy)

    return capture_outcome(fit)


@dataclass(frozen=True, slots=True, eq=False)
class _NuisanceValues(Mapping[str, float]):
    """Read-only scalar views retain their immutable native owner."""

    _native: _NativeArrayFit = field(repr=False)
    _count: int

    def __len__(self) -> int:
        return self._count

    def __getitem__(self, name: str) -> float:
        if not isinstance(name, str):
            raise CausalTypeError("a nuisance value name must be a string")
        value = self._native.value(name)
        if value is None:
            raise KeyError(name)
        return float(value)

    def __iter__(self) -> Iterator[str]:
        for offset in range(0, self._count, 256):
            for name, _value in self._native.values_page(offset=offset, limit=256):
                yield name

    def items(self) -> ItemsView[str, float]:
        return _NuisanceItems(self)


class _NuisanceItems(ItemsView[str, float]):
    def __init__(self, values: _NuisanceValues) -> None:
        super().__init__(values)
        self._values = values

    def __iter__(self) -> Iterator[tuple[str, float]]:
        values = self._values
        for offset in range(0, len(values), 256):
            yield from values._native.values_page(offset=offset, limit=256)


@dataclass(frozen=True, slots=True, repr=False)
class NuisanceScoreProduct:
    """Immutable native nuisance summaries; export alone encodes their replay design."""

    _native: _NativeArrayFit = field(repr=False)
    _metadata: Mapping[str, Any] = field(init=False, repr=False)
    execution: Execution = field(default_factory=Execution, repr=False, compare=False)
    acceptance: ArtifactAcceptance | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _artifact_bytes: bytes | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self._native, _NativeArrayFit):
            raise CausalTypeError("a nuisance product requires an immutable native fit owner")
        if not isinstance(self.execution, Execution):
            raise CausalTypeError("execution must be an Execution declaration")
        metadata = json.loads(self._native.metadata_json())
        object.__setattr__(self, "_metadata", freeze(metadata))
        consumed = metadata["consumed"]
        object.__setattr__(
            self,
            "acceptance",
            ArtifactAcceptance(
                "verified" if consumed else "sealed",
                metadata["identity_bound"],
                True,
                False,
                proof="independent_native_array_fit_refit" if consumed else "native_array_fit_seal",
            ),
        )

    @property
    def estimator_id(self) -> str:
        return str(self._metadata["estimator_id"])

    @property
    def target(self) -> str:
        return str(self._metadata["target"])

    @property
    def uncertainty(self) -> str:
        return str(self._metadata["uncertainty"])

    @property
    def licensing(self) -> str:
        return str(self._metadata["licensing"])

    @property
    def calibration(self) -> str:
        return str(self._metadata["calibration"])

    @property
    def request_digest(self) -> str:
        return str(self._metadata["request_digest"])

    @property
    def values(self) -> Mapping[str, float]:
        """Read-only views share native storage; dict(product.values) materializes."""
        return _NuisanceValues(self._native, int(self._metadata["value_count"]))

    def _strings(self, field: str) -> tuple[str, ...]:
        read = getattr(self._native, field + "_page")
        return tuple(
            value
            for offset in range(0, self._metadata[field[:-1] + "_count"], 256)
            for value in read(offset=offset, limit=256)
        )

    @property
    def assumptions(self) -> tuple[str, ...]:
        return self._strings("assumptions")

    @property
    def diagnostics(self) -> tuple[str, ...]:
        return self._strings("diagnostics")

    @property
    def answer(self) -> Answer:
        """Named nuisance summaries, without publishing an effect or target draws."""
        return Answer(
            "structured",
            detail=self.target,
            structured={"target": self.target, "value_count": len(self.values)},
        )

    @property
    def identities(self) -> Mapping[str, str]:
        """The native request digest; no causal claim or dataset identity is implied."""
        return freeze({"request_digest": self.request_digest})

    @property
    def provenance(self) -> Mapping[str, str]:
        return freeze({"estimator_id": self.estimator_id, "request_digest": self.request_digest})

    def inspect(self) -> InspectionReport:
        """Read retained native standing without fitting or decoding the artifact."""
        posterior = self.uncertainty == "nuisance_posterior_laplace_not_an_effect_posterior"
        return InspectionReport(
            identification=SlotModel(
                available=False,
                reason="nuisance_not_causal_effect",
                summary="treatment-model nuisance; no causal-effect identification claim",
            ),
            support=SlotModel(
                available=False,
                reason=self.licensing,
                summary=self.licensing,
                payload={"licensing": self.licensing, "calibration": self.calibration},
            ),
            uncertainty=SlotModel(
                available=posterior,
                reason=None if posterior else self.uncertainty,
                summary=self.uncertainty,
                payload={"target": self.target, "kind": self.uncertainty},
            ),
            assumptions=SlotModel(
                available=True,
                summary="declared nuisance-model assumptions",
                payload={"declarations": self.assumptions},
            ),
            answer=self.answer,
            diagnostics=self.diagnostics,
            contract={"provenance": self.provenance},
            target={"kind": self.target},
            calibration=self.calibration,
            family="nuisance_scores",
            stage="consumed" if self._metadata["consumed"] else "executed",
            details={
                "value_count": len(self.values),
                "request_digest": self.request_digest,
                "acceptance": self.acceptance,
            },
            operations=("inspect", "export", "to_dict"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Materialize named summaries; obtain the encoded design through export()."""
        return thaw(
            {
                "estimator_id": self.estimator_id,
                "target": self.target,
                "uncertainty": self.uncertainty,
                "licensing": self.licensing,
                "calibration": self.calibration,
                "values": dict(self.values.items()),
                "assumptions": self.assumptions,
                "diagnostics": self.diagnostics,
                "request_digest": self.request_digest,
            }
        )

    def __repr__(self) -> str:
        return (
            f"NuisanceScoreProduct(estimator_id={self.estimator_id!r}, "
            f"target={self.target!r}, uncertainty={self.uncertainty!r}, "
            f"value_count={len(self.values)})"
        )

    def _repr_html_(self) -> str:
        return self.inspect()._repr_html_()

    @property
    def licensed(self) -> bool:
        """Always ``False``: no support license covers this product."""
        return False

    @property
    def calibration_measured(self) -> bool:
        """Always ``False``: repeated-sampling calibration is unmeasured."""
        return False

    def __getitem__(self, name: str) -> float:
        return self.values[name]

    @artifact_export
    def export(self, *, execution: Execution | None = None) -> ExportedArtifact:
        """Explicitly encode the native replay design, or forward its verified original bytes."""
        policy = self.execution if execution is None else execution
        if not isinstance(policy, Execution):
            raise CausalTypeError("execution must be an Execution declaration")
        data = policy._call(self._native.export_bytes)
        return ExportedArtifact(
            data,
            "array_fit_replay_v1",
            ArtifactExpectation("array_fit_replay_v1", self.request_digest),
            {"estimator_id": self.estimator_id, "target": self.target},
        )

    def explain(self) -> str:
        """What the product is, and what it is not."""
        return (
            f"{self.estimator_id}: {self.target}, {self.uncertainty}; {self.licensing}, "
            f"calibration {self.calibration}. A posterior of a nuisance, not of an effect."
        )


def _product(native_product: _NativeArrayFit, execution: Execution) -> NuisanceScoreProduct:
    return NuisanceScoreProduct(native_product, execution)


@artifact_consumer(
    proof="independent_native_array_fit_refit",
    input="artifact",
    identity=("expected_identity",),
    replayable=True,
)
def load_artifact(
    artifact: bytes | ExportedArtifact,
    *,
    expected_identity: str | None = None,
    execution: Execution | None = None,
) -> NuisanceScoreProduct:
    """Independently refit the original nuisance recipe; retain no study or dataset.

    ``expected_identity`` is the producer request digest, retained separately from
    the encoding. Omitting it still replays every stored value but leaves the
    producer identity unbound. Scientific model/seed settings come from the
    artifact; Execution controls resources, threads, cancellation and observers.
    """
    policy = Execution() if execution is None else execution
    if not isinstance(policy, Execution):
        raise CausalTypeError("execution must be an Execution declaration")
    if policy.on_stage is not None:
        raise CausalUnsupportedError(
            "nuisance replay does not expose causal stages", reason_code="stage_stream_unavailable"
        )
    if expected_identity is not None and not isinstance(expected_identity, str):
        raise CausalTypeError("expected_identity must be a native request digest string")
    data = container_bytes(artifact)
    owner = policy._call(
        _consume, data, expected_identity=expected_identity, threads=policy.threads
    )
    return _product(owner, policy)
