"""Staleness and drift diagnostics: is a stored plan or claim still valid for new data?

:func:`staleness` compares new data with an exported staleness baseline (or with a prepared
study, through ``PreparedAnalysis.staleness``) under a declared :class:`StalenessPolicy`, and
returns a :class:`StalenessReport`:

* every diagnostic as a :class:`StalenessFinding` with its statistic, the policy threshold it
  was judged by, its (multiplicity-adjusted) p-value when it is a test, and the dependency keys
  it changes when it exceeds;
* one recommendation: ``"reuse"`` (nothing exceeds), ``"refresh"`` (the data changed but the
  identification premises hold: keep the plan, refit), ``"reidentify"`` (overlap, support, a
  recorded identification check or period drift failed: identify again) or ``"refuse"``
  (schema or target incompatible, or too few rows to assess uncertainty), with the attempt to
  make next;
* the changed dependency keys and the stages selective recalculation recomputes for them.

Every statistic, threshold, multiplicity rule and identity is computed natively; this module
only declares the policy and reads the report. No scheduler or alerting service is shipped:
a host calls the diagnostic. See ``docs/guides/staleness.md``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from typing import TYPE_CHECKING, Any, Literal

from . import _native
from ._artifact_protocol import (
    ArtifactAcceptance,
    ArtifactExpectation,
    ExportedArtifact,
    container_bytes,
)
from ._data import ingest_columns
from ._freeze import freeze, thaw
from ._scientific import ScientificResult
from .errors import CausalTypeError, CausalUnsupportedError, StructuredRefusal
from .execution import Execution
from .outcomes import Answer, Outcome, outcome_operation, require_success

__all__ = [
    "Recommendation",
    "StalenessBaseline",
    "StalenessFinding",
    "StalenessPolicy",
    "StalenessReport",
    "baseline",
    "consume_baseline",
    "staleness",
]

if TYPE_CHECKING:
    from .prepared import PreparedAnalysis

Recommendation = Literal["reuse", "refresh", "reidentify", "refuse"]


@dataclass(frozen=True, slots=True)
class StalenessPolicy:
    """Declared thresholds, tests and sample-size floors; ``None`` keeps the native default.

    The policy is part of the report identity: two reports under different policies never
    compare equal. :meth:`resolved` returns every field with its value and the identity.
    ``seed`` keys the subsampling and permutation streams of the diagnostics; the seed of the
    ``Execution`` does not reach them, so declare the stream here.
    """

    alpha: float | None = None
    multiplicity: Literal["holm", "benjamini_hochberg"] | None = None
    smd_threshold: float | None = None
    psi_threshold: float | None = None
    mahalanobis_threshold: float | None = None
    bins: int | None = None
    subsample: int | None = None
    permutations: int | None = None
    seed: int | None = None
    positivity_epsilon: float | None = None
    max_extreme_fraction: float | None = None
    min_ess_ratio_change: float | None = None
    balance_threshold: float | None = None
    max_condition: float | None = None
    min_complete_fraction: float | None = None
    support_tolerance: float | None = None
    min_level_rows: int | None = None
    missingness_threshold: float | None = None
    min_rows: int | None = None
    min_rows_per_arm: int | None = None
    min_rows_per_period: int | None = None
    period_column: str | None = None
    prior_variance: float | None = None
    prior_shape: float | None = None
    prior_rate: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """The declared (non-``None``) fields."""
        return {key: value for key, value in asdict(self).items() if value is not None}

    def to_json(self) -> str:
        """The declared fields as a JSON object."""
        return json.dumps(self.to_dict(), sort_keys=True)

    def resolved(self) -> dict[str, Any]:
        """Every field with its effective value, and ``identity``, after native validation."""
        resolved: dict[str, Any] = json.loads(_native.staleness_policy(self.to_json()))
        return resolved

    @property
    def identity(self) -> str:
        """Canonical identity of the effective policy."""
        return str(self.resolved()["identity"])


def _policy_json(policy: StalenessPolicy | Mapping[str, Any] | None) -> str | None:
    if policy is None:
        return None
    if isinstance(policy, StalenessPolicy):
        return policy.to_json()
    if isinstance(policy, Mapping):
        known = {f.name for f in fields(StalenessPolicy)}
        unknown = sorted(set(policy) - known)
        if unknown:
            raise CausalTypeError(f"StalenessPolicy has no field(s) {unknown}")
        return StalenessPolicy(**dict(policy)).to_json()
    raise CausalTypeError("policy must be a StalenessPolicy, a mapping of its fields, or None")


@dataclass(frozen=True, slots=True)
class StalenessFinding(ScientificResult):
    """One evaluated diagnostic; ``exceeded`` says whether it counts as a finding."""

    kind: str
    subject: str
    statistic_name: str
    statistic: float | None
    threshold_name: str
    threshold: float | None
    p_value: float | None
    adjusted_p_value: float | None
    exceeded: bool
    forces: Recommendation
    keys: tuple[str, ...]
    auxiliary: Mapping[str, float | None]
    detail: str

    @classmethod
    def _from_dict(cls, raw: Mapping[str, Any]) -> StalenessFinding:
        return cls(
            kind=str(raw["kind"]),
            subject=str(raw["subject"]),
            statistic_name=str(raw["statistic_name"]),
            statistic=raw["statistic"],
            threshold_name=str(raw["threshold_name"]),
            threshold=raw["threshold"],
            p_value=raw["p_value"],
            adjusted_p_value=raw["adjusted_p_value"],
            exceeded=bool(raw["exceeded"]),
            forces=raw["forces"],
            keys=tuple(raw["keys"]),
            auxiliary=dict(raw["auxiliary"]),
            detail=str(raw["detail"]),
        )

    def to_dict(self) -> dict[str, Any]:
        """Plain JSON-compatible view."""
        return {item.name: thaw(getattr(self, item.name)) for item in fields(self)}


@dataclass(frozen=True, slots=True)
class StalenessReport(ScientificResult):
    """The comparison of new data with a stored plan or baseline."""

    recommendation: Recommendation
    next_attempt: str
    evaluations: tuple[StalenessFinding, ...]
    not_evaluated: tuple[tuple[str, str], ...]
    changed_keys: tuple[str, ...]
    invalidated: tuple[str, ...]
    changed_classes: tuple[str, ...]
    invalidated_products: tuple[str, ...]
    plan_identity: str
    old_snapshot: str
    new_snapshot: str
    baseline_identity: str
    policy_identity: str
    new_rows: int
    identity: str

    @property
    def answer(self) -> Answer:
        return Answer("structured", structured={"recommendation": self.recommendation})

    @property
    def identities(self) -> Mapping[str, str]:
        return freeze(
            {
                "identity": self.identity,
                "plan_identity": self.plan_identity,
                "old_snapshot": self.old_snapshot,
                "new_snapshot": self.new_snapshot,
                "baseline_identity": self.baseline_identity,
                "policy_identity": self.policy_identity,
            }
        )

    def _summary_fields(self) -> dict[str, Any]:
        return {
            "recommendation": self.recommendation,
            "finding_count": len(self.findings),
            "new_rows": self.new_rows,
            "identity": self.identity,
        }

    @property
    def findings(self) -> tuple[StalenessFinding, ...]:
        """The diagnostics that exceeded their thresholds."""
        return tuple(f for f in self.evaluations if f.exceeded)

    @classmethod
    def _from_json(cls, text: str) -> StalenessReport:
        raw = json.loads(text)
        return cls(
            recommendation=raw["recommendation"],
            next_attempt=str(raw["next_attempt"]),
            evaluations=tuple(StalenessFinding._from_dict(f) for f in raw["evaluations"]),
            not_evaluated=tuple(
                (str(entry["diagnostic"]), str(entry["reason"])) for entry in raw["not_evaluated"]
            ),
            changed_keys=tuple(raw["changed_keys"]),
            invalidated=tuple(raw["invalidated"]),
            changed_classes=tuple(raw["changed_classes"]),
            invalidated_products=tuple(raw["invalidated_products"]),
            plan_identity=str(raw["plan_identity"]),
            old_snapshot=str(raw["old_snapshot"]),
            new_snapshot=str(raw["new_snapshot"]),
            baseline_identity=str(raw["baseline_identity"]),
            policy_identity=str(raw["policy_identity"]),
            new_rows=int(raw["new_rows"]),
            identity=str(raw["identity"]),
        )

    def to_dict(self) -> dict[str, Any]:
        """Plain JSON-compatible view (every evaluation, keys and stages by label)."""
        return {
            "recommendation": self.recommendation,
            "next_attempt": self.next_attempt,
            "evaluations": [f.to_dict() for f in self.evaluations],
            "not_evaluated": [{"diagnostic": d, "reason": r} for d, r in self.not_evaluated],
            "changed_keys": list(self.changed_keys),
            "invalidated": list(self.invalidated),
            "changed_classes": list(self.changed_classes),
            "invalidated_products": list(self.invalidated_products),
            "plan_identity": self.plan_identity,
            "old_snapshot": self.old_snapshot,
            "new_snapshot": self.new_snapshot,
            "baseline_identity": self.baseline_identity,
            "policy_identity": self.policy_identity,
            "new_rows": self.new_rows,
            "identity": self.identity,
        }


@dataclass(frozen=True, slots=True, init=False, repr=False)
class StalenessBaseline(ScientificResult):
    """Independently checked diagnostic summary, with no live study or full dataset.

    The historical baseline may retain a bounded diagnostic covariate subsample.
    Its seal authenticates its recorded summary; it does not establish causal
    identification or replay the unavailable original observations.
    """

    _artifact_bytes: bytes = field(repr=False)
    _metadata: Mapping[str, Any] = field(repr=False)
    acceptance: ArtifactAcceptance

    def __init__(self, artifact: Any, *, expected_identity: str | None = None) -> None:
        encoded = container_bytes(artifact)
        metadata = json.loads(
            _native.consume_staleness_baseline(encoded, expected_identity=expected_identity)
        )
        object.__setattr__(self, "_artifact_bytes", encoded)
        object.__setattr__(self, "_metadata", freeze(metadata))
        object.__setattr__(
            self,
            "acceptance",
            ArtifactAcceptance(
                status="sealed",
                identity_bound=expected_identity is not None,
                replayable=False,
                resumable=False,
                proof="native baseline consumer verifies its version, required features and summary seal",
                unresolved=(
                    "original observations are unavailable for independent profile replay",
                ),
            ),
        )
        object.__setattr__(self, "_inspection_stage", "consumed")

    @property
    def identity(self) -> str:
        return str(self._metadata["identity"])

    @property
    def plan_identity(self) -> str:
        return str(self._metadata["plan_identity"])

    @property
    def snapshot(self) -> str:
        return str(self._metadata["snapshot"])

    @property
    def policy_identity(self) -> str:
        return str(self._metadata["policy_identity"])

    @property
    def rows(self) -> int:
        return int(self._metadata["rows"])

    @property
    def complete_rows(self) -> int:
        return int(self._metadata["complete_rows"])

    def _summary_fields(self) -> dict[str, Any]:
        return {"identity": self.identity, "rows": self.rows, "complete_rows": self.complete_rows}

    def to_dict(self) -> dict[str, Any]:
        """Bounded native summary metadata; retained profile arrays stay in the artifact."""
        return {**thaw(self._metadata), "acceptance": self.acceptance.to_dict()}

    def export(self) -> ExportedArtifact[StalenessBaseline]:
        return ExportedArtifact(
            self._artifact_bytes,
            "staleness_baseline_v1",
            ArtifactExpectation("staleness_baseline_v1", self.identity),
        )

    def assess(
        self,
        new_data: Any,
        *,
        policy: StalenessPolicy | Mapping[str, Any] | None = None,
        execution: Execution | None = None,
    ) -> Outcome[StalenessReport]:
        return staleness(self, new_data, policy=policy, execution=execution)


@outcome_operation
def consume_baseline(artifact: Any, *, expected_identity: str | None = None) -> StalenessBaseline:
    """Invoke the native independent baseline consumer; never recreate a live study."""
    return StalenessBaseline(artifact, expected_identity=expected_identity)


def _execution(source: Any, execution: Execution | None) -> Execution:
    from .prepared import PreparedAnalysis

    if execution is not None:
        if not isinstance(execution, Execution):
            raise CausalTypeError("execution must be an Execution declaration or None")
        policy = execution
    else:
        policy = source.execution if isinstance(source, PreparedAnalysis) else Execution()
    if policy.on_stage is not None:
        raise CausalUnsupportedError(
            "drift diagnostics do not expose causal estimation stages",
            reason_code="stage_stream_unavailable",
        )
    return policy


def _study_engine(study: Any) -> Any:
    from .estimation import _PreparedEngine as Engine
    from .prepared import PreparedAnalysis

    if not isinstance(study, PreparedAnalysis):
        raise CausalTypeError("baseline requires a canonical PreparedAnalysis")
    if study._refused is not None:
        assert study._refused.refusal is not None
        study._refused.refusal.raise_error()
    engine = study._engine
    if not isinstance(engine, Engine) or engine._native is None or engine._transport is not None:
        raise StructuredRefusal(
            {
                "code": "route_not_supported",
                "detail": "staleness.native_table_required",
                "message": "staleness requires an ordinary prepared tabular or series snapshot",
                "stage": "staleness",
            }
        )
    return engine


@outcome_operation
def baseline(
    study: PreparedAnalysis[Any],
    *,
    policy: StalenessPolicy | Mapping[str, Any] | None = None,
    execution: Execution | None = None,
) -> StalenessBaseline:
    """Explicitly compute a native diagnostic baseline under the declared policy.

    Profiling includes the existing diagnostic propensity fit. Calling this
    operation neither estimates the causal effect nor changes the prepared study.
    Preparation and inspection never invoke it implicitly.
    """
    engine = _study_engine(study)
    controls = _execution(study, execution)

    def execute() -> StalenessBaseline:
        encoded = engine._native.export_staleness_baseline(
            policy_json=_policy_json(policy), seed=controls.seed, threads=controls.threads
        )
        return StalenessBaseline(encoded)

    return controls._call(execute)


@outcome_operation
def staleness(
    source: PreparedAnalysis[Any]
    | StalenessBaseline
    | Outcome[StalenessBaseline]
    | ExportedArtifact[Any]
    | bytes,
    new_data: Any,
    *,
    policy: StalenessPolicy | Mapping[str, Any] | None = None,
    execution: Execution | None = None,
) -> StalenessReport:
    """Explicitly assess new observations against a canonical study or sealed baseline.

    Every native diagnostic and recommendation is retained. A ``refuse``
    recommendation is a successful assessment of the plan's applicability;
    operational failures remain exceptions. The source is never mutated.
    """
    from .prepared import PreparedAnalysis

    source = require_success(source)
    controls = _execution(source, execution)
    policy_json = _policy_json(policy)

    def assess() -> StalenessReport:
        if isinstance(source, PreparedAnalysis):
            engine = _study_engine(source)
            names, columns = engine._staleness_payload(new_data)
            text = engine._native.staleness(
                names,
                columns,
                policy_json=policy_json,
                seed=controls.seed,
                threads=controls.threads,
            )
        else:
            encoded = (
                source._artifact_bytes
                if isinstance(source, StalenessBaseline)
                else container_bytes(source)
            )
            names, columns = ingest_columns(new_data)
            text = _native.staleness_from_artifact(
                encoded,
                names,
                columns,
                policy_json=policy_json,
                seed=controls.seed,
                threads=controls.threads,
            )
        return StalenessReport._from_json(text)

    return controls._call(assess)
