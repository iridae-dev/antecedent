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
from dataclasses import asdict, dataclass, fields
from typing import Any, Literal

from . import _native
from ._artifact_protocol import ExportedArtifact, container_bytes
from ._data import ingest_columns
from .errors import CausalTypeError

__all__ = [
    "Recommendation",
    "StalenessBaseline",
    "StalenessFinding",
    "StalenessPolicy",
    "StalenessReport",
    "staleness",
]

Recommendation = Literal["reuse", "refresh", "reidentify", "refuse"]


@dataclass(frozen=True, slots=True)
class StalenessPolicy:
    """Declared thresholds, tests and sample-size floors; ``None`` keeps the native default.

    The policy is part of the report identity: two reports under different policies never
    compare equal. :meth:`resolved` returns every field with its value and the identity.
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
class StalenessFinding:
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
        out = asdict(self)
        out["keys"] = list(self.keys)
        out["auxiliary"] = dict(self.auxiliary)
        return out


@dataclass(frozen=True, slots=True)
class StalenessReport:
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


@dataclass(frozen=True, slots=True)
class StalenessBaseline:
    """The sealed summary of a plan's data, consumable without the producer."""

    artifact: bytes

    def to_bytes(self) -> bytes:
        """The sealed artifact bytes."""
        return self.artifact

    @classmethod
    def from_bytes(cls, artifact: bytes) -> StalenessBaseline:
        """Wrap artifact bytes; they are verified when assessed."""
        artifact = (
            container_bytes(artifact)
            if isinstance(artifact, (bytes, bytearray, memoryview, ExportedArtifact))
            else artifact
        )
        return cls(bytes(artifact))


def staleness(
    artifact: StalenessBaseline | bytes | Any,
    new_data: Mapping[str, Any] | Any,
    *,
    policy: StalenessPolicy | Mapping[str, Any] | None = None,
    threads: int | None = None,
) -> StalenessReport:
    """Compare ``new_data`` with a staleness baseline under ``policy``.

    ``artifact`` is a :class:`StalenessBaseline`, its bytes (as written by
    ``PreparedAnalysis.staleness_baseline().to_bytes()``), or a prepared study (whose retained
    data is the baseline). Only the artifact and the new data are read: the producer and its
    rows are not needed.
    """
    artifact = (
        container_bytes(artifact)
        if isinstance(artifact, (bytes, bytearray, memoryview, ExportedArtifact))
        else artifact
    )
    if hasattr(artifact, "staleness") and not isinstance(artifact, bytes | bytearray):
        report: StalenessReport = artifact.staleness(new_data, policy=policy, threads=threads)
        return report
    raw = artifact.to_bytes() if isinstance(artifact, StalenessBaseline) else artifact
    if not isinstance(raw, bytes | bytearray | memoryview):
        raise CausalTypeError("artifact must be a StalenessBaseline, bytes, or a prepared study")
    names, columns = ingest_columns(new_data)
    text = _native.staleness_from_artifact(
        bytes(raw), names, columns, policy_json=_policy_json(policy), threads=threads
    )
    return StalenessReport._from_json(text)
