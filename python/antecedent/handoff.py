"""Handoffs to external estimators that consume an adjustment set.

Antecedent identifies. The adapter emits the set and status for estimands that
actually are adjustment estimands. ``attach`` records a caller-fitted payload
as attested, not re-verifiable, evidence. It does not wrap a learner or absorb
ML CATE. EconML is one consumer of ``columns(layout="econml")``; the EconML
names on this module are aliases of the adjustment handoff.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

from ._artifact_protocol import (
    ExportedArtifact,
    artifact_export,
    artifact_from_receipt,
    container_bytes,
)
from ._digest import canonical_digest
from .errors import CausalError, CausalTypeError
from .estimation import IdentifyResult
from .extensibility import ProviderRefusal, ProviderValueRefusal
from .extensibility import _refuse as _provider_refuse
from .identify import Identification
from .ids import Identifier
from .results import AnalysisResult, CausalResponseView, IdentificationView

if TYPE_CHECKING:
    from .external import BoundExternalClaim

_ADJUSTMENT_IDENTIFIERS = frozenset(
    {
        Identifier.BACKDOOR_ADJUSTMENT,
        Identifier.BACKDOOR_EFFICIENT,
        Identifier.GENERALIZED_ADJUSTMENT,
        Identifier.RESPONSE_BACKDOOR,
        Identifier.TEMPORAL_BACKDOOR_UNFOLDED,
    }
)

_GRAPH_POSTERIOR_MARKERS = ("posterior", "mcmc")

# Historical provenance identity. Omitting ``tool`` keeps this string so
# ``CausalProviderSpec`` digests match the EconML-named handoff.
_DEFAULT_TOOL = "antecedent.handoff.EconMLSpec"
_DEFAULT_ADAPTER_ID = "antecedent.handoff.EconMLProviderAdapter"


#: Reason codes of a refusal to hand off an estimand: it was a ``CausalUnsupportedError``
#: and stays one; every other handoff refusal was a ``ValueError`` and still is.
_ESTIMAND_REFUSALS = frozenset({"route_not_supported", "effect_not_identified"})


def _refuse(
    detail: str,
    message: str,
    *,
    remedy: str,
    code: str = "invalid_argument",
    offending: str | None = None,
) -> ProviderRefusal:
    """A typed handoff refusal: registered ``reason_code``, ``detail`` and ``remedy``.

    Each site keeps the class it raised before it gained a code: refusing an estimand
    is a :class:`~antecedent.errors.CausalUnsupportedError`; every other refusal is
    also a :class:`ValueError` (:class:`~antecedent.errors.CausalValueError`).
    """
    return _provider_refuse(
        detail,
        message,
        code=code,
        remedy=remedy,
        offending=offending,
        stage="handoff",
        error=ProviderRefusal if code in _ESTIMAND_REFUSALS else ProviderValueRefusal,
    )


_NAMED_COLUMNS = {
    "Y": "outcome",
    "T": "treatment",
    "W": "confounders",
    "X": "modifiers",
    "sample_weight": "weights",
    "origins": "origins",
}


@dataclass(frozen=True, slots=True)
class AdjustmentHandoff:
    """Adjustment-set handoff for a caller-owned CATE estimator.

    ``confounders`` is ``W`` when ``columns(layout="econml")``. Heterogeneity
    features are the caller's: identification does not name them.

    ``tool`` is the provenance adapter identity. It defaults to the historical
    EconML spec string so an omitted tool leaves the spec digest unchanged.
    """

    treatment: str | tuple[str, ...]
    outcome: str
    confounders: tuple[str, ...]
    identifier: str
    status: str
    treatment_offsets: tuple[int, ...] = ()
    outcome_offset: int = 0
    confounder_offsets: tuple[int, ...] = ()
    modifiers: tuple[str, ...] = ()
    temporal: bool = False
    target_weights: Any | None = None
    outcome_functional: Any | None = None
    tool: str = _DEFAULT_TOOL
    _parent: Any | None = field(default=None, repr=False, compare=False)

    @property
    def treatments(self) -> tuple[str, ...]:
        """All jointly intervened variables, in certificate order."""
        return (self.treatment,) if isinstance(self.treatment, str) else self.treatment

    @property
    def spec(self):
        """Reusable declaration of this handoff's boundaries, not a learner runner."""
        from .extensibility import CausalProviderSpec, ProviderCapability

        if not isinstance(self.tool, str) or not self.tool.strip():
            raise _refuse(
                "handoff.tool_invalid",
                "adjustment handoff tool must be a non-empty string",
                offending="tool",
                remedy="name the adapter that consumes this handoff, or omit tool=",
            )
        return CausalProviderSpec(
            query_family="point_identified_adjustment_estimand",
            identification_requirements=(self.identifier, "point_identified"),
            observed_distributions=("aligned Y and T", "certified W", "optional certified X"),
            nuisance_functions=(),
            support_conditions=("explicit_or_accepted_graph", "common_certified_adjustment_set"),
            data_dependence=(
                "uses_caller_supplied_aligned_rows",
                "handoff_does_not_own_or_record_training_rows",
                "data_snapshot_identity_is_attached_when_available",
            ),
            inference_claims=(
                "caller_attested_point_or_interval_only",
                "interval_calibration_is_not_reverified",
            ),
            influence_function=None,
            fold_policy="external_learner_owned_not_recorded_by_handoff",
            output_shape="provider_defined",
            uncertainty_semantics="externally_attested_not_reverifiable",
            artifact_codec="antecedent_external_estimate_artifact",
            deterministic=False,
            provenance={"adapter": self.tool},
            capabilities=(ProviderCapability.MEAN,),
            meaning="interventional_predictive",
        )

    def as_provider(self, executor: Any) -> AdjustmentProviderAdapter:
        """Wrap a caller-owned fit callback in the Python provider runtime.

        ``executor(columns, request)`` must return ``learner`` and ``effect``;
        optional keys are ``learner_config``, ``interval``, ``label``,
        ``value_types``, and ``contrast``. Data extraction, identity binding,
        and artifact encoding remain owned by this handoff. The provenance
        adapter id is this handoff's ``tool``; the historical adapter id is
        kept when ``tool`` is omitted.
        """
        if not callable(executor):
            raise TypeError("adjustment provider executor must be callable")
        return AdjustmentProviderAdapter(self, executor)

    def columns(
        self,
        data: Mapping[str, Any],
        *,
        layout: Literal["econml", "named"] = "econml",
    ) -> dict[str, Any]:
        """Return aligned outcome, treatment, confounders, and modifiers.

        ``layout="econml"`` (the default) uses EconML's ``fit(Y, T, X, W)``
        names: ``Y``, ``T``, ``W``, optional ``X``, optional ``sample_weight``,
        and ``origins`` on a temporal spec. ``layout="named"`` uses
        ``outcome``, ``treatment``, ``confounders``, ``modifiers``, ``weights``,
        and ``origins`` (missing optional arrays are ``None``).

        For temporal specs the input is one regularly sampled series in row order.
        Offset k selects row origin+k. Boundary rows are trimmed, never wrapped.
        ``origins`` identifies retained rows relative to the input series; separate
        subjects must be aligned separately to avoid crossing panel boundaries.
        """
        if layout not in ("econml", "named"):
            raise _refuse(
                "handoff.layout_unknown",
                "adjustment handoff layout must be 'econml' or 'named'",
                offending="layout",
                remedy="pass layout='econml' (Y, T, W, X) or layout='named'",
            )
        names = (self.outcome, *self.treatments, *self.confounders, *self.modifiers)
        missing = [name for name in names if name not in data]
        if missing:
            raise _refuse(
                "handoff.columns_missing",
                f"adjustment handoff missing columns: {missing}",
                code="unknown_variable",
                offending=", ".join(missing),
                remedy="pass data holding the outcome, treatment, confounder and modifier "
                "columns this handoff names",
            )
        arrays = {name: np.asarray(data[name]) for name in names}
        if any(col.ndim != 1 for col in arrays.values()):
            raise _refuse(
                "handoff.columns_not_one_dimensional",
                "adjustment handoff requires one-dimensional columns",
                code="schema_mismatch",
                remedy="pass each variable as a one-dimensional column",
            )
        if len({len(col) for col in arrays.values()}) != 1:
            raise _refuse(
                "handoff.columns_unaligned",
                "adjustment handoff columns must have the same number of rows",
                code="schema_mismatch",
                remedy="align the columns to the same rows before the handoff",
            )
        t_offsets = self.treatment_offsets or (0,) * len(self.treatments)
        w_offsets = self.confounder_offsets or (0,) * len(self.confounders)
        offsets = (0, self.outcome_offset, *t_offsets, *w_offsets)
        n = len(arrays[self.outcome])
        origins = np.arange(max(0, -min(offsets)), min(n, n - max(offsets)))
        if self.temporal and not len(origins):
            raise _refuse(
                "handoff.no_rows_after_alignment",
                "adjustment handoff has no rows after temporal alignment",
                remedy="supply a longer series than the largest certified lag offset",
            )
        targets = [
            arrays[name][origins + offset]
            for name, offset in zip(self.treatments, t_offsets, strict=True)
        ]
        cols = {
            "Y": arrays[self.outcome][origins + self.outcome_offset],
            "T": targets[0] if len(targets) == 1 else np.column_stack(targets),
            "W": np.column_stack(
                [
                    arrays[name][origins + offset]
                    for name, offset in zip(self.confounders, w_offsets, strict=True)
                ]
            )
            if self.confounders
            else None,
        }
        if self.modifiers:
            cols["X"] = np.column_stack([arrays[name][origins] for name in self.modifiers])
        if self.outcome_functional is not None:
            from .query import coerce_outcome_functional

            functional = coerce_outcome_functional(self.outcome_functional)
            if functional is not None:
                y = np.asarray(cols["Y"])
                if functional["kind"] == "exceedance":
                    cols["Y"] = (y > float(cast(float, functional["threshold"]))).astype(float)
                elif functional["kind"] == "exceedance_grid":
                    cols["Y"] = np.column_stack(
                        [(y > c).astype(float) for c in cast(list[float], functional["thresholds"])]
                    )
        if self.target_weights is not None:
            weights = np.asarray(self.target_weights, dtype=float)
            if (
                weights.ndim != 1
                or len(weights) != n
                or not np.isfinite(weights).all()
                or (weights < 0).any()
            ):
                raise _refuse(
                    "handoff.target_weights_invalid",
                    "target weights must be finite non-negative weights aligned to input rows",
                    offending="target_weights",
                    remedy="give one finite non-negative weight per input row",
                )
            aligned = weights[origins]
            if aligned.sum() <= 0:
                raise _refuse(
                    "handoff.target_weights_no_mass",
                    "target weights have no mass on aligned rows",
                    offending="target_weights",
                    remedy="give positive weight to at least one retained row",
                )
            cols["sample_weight"] = aligned
        if self.temporal:
            cols["origins"] = origins
        if layout == "econml":
            return cols
        return {renamed: cols.get(key) for key, renamed in _NAMED_COLUMNS.items()}

    def attach(
        self,
        *,
        learner: str,
        learner_config: Mapping[str, Any] | None = None,
        effect: Any,
        interval: Any | None = None,
        data: Mapping[str, Any] | None = None,
        label: Literal["ate", "cate"] | None = None,
        value_types: Mapping[str, Literal["continuous", "binary", "count"]] | None = None,
        contrast: tuple[float, float] | None = None,
        learner_version: str | None = None,
        outcome_units: str | None = None,
        support: str | Sequence[str] | None = None,
    ) -> ExternalEstimate:
        """Attach a caller-fitted external estimate as attested, not re-verifiable, evidence.

        Antecedent does not import or wrap the learner. The label, config, and
        payload are hashed onto ``claim.attested``; calibration stays
        ``unavailable`` / ``attested_not_reverifiable``.

        A labelled effect is a claim, so it is bound through
        :meth:`antecedent.external.ExternalSpec.bind` like any foreign response: the
        identified contract, the learner as a :class:`~antecedent.external.ProviderObject`
        (``learner_version`` is its version, never inferred), support labels and the
        lineage come along as :attr:`ExternalEstimate.claim`, and a mismatch refuses
        with the same structured refusal. ``outcome_units`` names the units of the
        outcome; they are never inferred.

        - ``label="ate"``: one scalar effect, bound as a one-coordinate response
          (``support`` is one label).
        - ``label="cate"``: one conditional effect per row (a one-dimensional array
          aligned to the handoff's rows), bound as a
          :class:`~antecedent.external.BoundExternalColumn`: one scientific quantity
          and the column's digest, size and support counts, never one coordinate per
          row. ``support`` is one label for every row or one per row.
        - unlabelled: an opaque attested payload with no bound claim.

        ``value_types`` (name -> ``continuous`` / ``binary`` / ``count``) and ``contrast``
        (``(control, active)`` treatment levels) are recorded in the receipt's target as
        declared; anything not declared is recorded as unspecified / a named 0 -> 1
        placeholder, never silently assumed.
        """
        _require_attachable(self)
        parent = self._parent
        if parent is not None:
            _require_adjustment_parent(parent)
        identification, snapshot, snapshot_payload = _bind_parent_identities(self, parent, data)
        effect_arr = np.asarray(effect, dtype=np.float64)
        scalar: float | None = None
        column: np.ndarray | None = None
        if label == "ate":
            if effect_arr.shape not in ((), (1,)):
                raise _refuse(
                    "handoff.label_ate_not_scalar",
                    "label='ate' requires a scalar effect",
                    offending="effect",
                    remedy="pass one number, or label a row-level array label='cate'",
                )
            scalar = float(np.reshape(effect_arr, ()))
        elif label == "cate":
            column = _row_effects(self, effect_arr, data)
        elif label is not None:
            raise _refuse(
                "handoff.label_unknown",
                f"label must be 'ate', 'cate' or None; got {label!r}",
                offending="label",
                remedy="label a scalar effect 'ate' and a one-value-per-row array 'cate'",
            )
        names = _receipt_names(self)
        declared_types = None
        if value_types is not None:
            unknown = sorted(set(value_types) - set(names))
            missing = [name for name in names if name not in value_types]
            if unknown or missing:
                raise _refuse(
                    "handoff.value_types_mismatch",
                    "value_types must name exactly the receipt variables "
                    f"(unknown: {unknown}, missing: {missing})",
                    offending="value_types",
                    remedy="give one value type for each of: " + ", ".join(names),
                )
            declared_types = [str(value_types[name]) for name in names]
        from . import artifacts
        from ._native import encode_external_estimate_claim

        config_bytes = _canonical_config(learner_config)
        payload = _payload_bytes(effect_arr, interval)
        if scalar is None and column is None:
            if support is not None or learner_version is not None or outcome_units is not None:
                raise _refuse(
                    "handoff.claim_arguments_without_label",
                    "learner_version, outcome_units and support describe a bound claim; "
                    "they apply only to an effect labelled label='ate' or label='cate'",
                    offending="label",
                    remedy="label the effect, or drop the claim-describing arguments",
                )
            claim = None
        else:
            claim = _bind_claim(
                self,
                learner=str(learner),
                config=config_bytes,
                payload=payload,
                value=column if column is not None else cast(float, scalar),
                interval=interval is not None,
                identification=identification,
                snapshot=snapshot,
                snapshot_payload=snapshot_payload,
                contrast=contrast,
                learner_version=learner_version,
                outcome_units=outcome_units,
                support=support,
            )

        encoded = encode_external_estimate_claim(
            learner=str(learner),
            config=config_bytes,
            payload=payload,
            names=names,
            treatment=self.treatments[0],
            outcome=self.outcome,
            identifier=self.identifier,
            status=self.status,
            confounders=list(self.confounders),
            identification=identification,
            data_snapshot=snapshot,
            snapshot_payload=snapshot_payload,
            scalar_value=scalar,
            value_types=declared_types,
            contrast=None if contrast is None else (float(contrast[0]), float(contrast[1])),
            modifiers=list(self.modifiers),
        )
        decoded = artifacts.loads(encoded)
        if not isinstance(decoded.contract, Mapping):
            raise _refuse(
                "handoff.receipt_contract_missing",
                "external estimate receipt is missing a contract",
                code="external_binding_mismatch",
                remedy="re-attach the estimate; the receipt encoder returned no contract",
            )
        return ExternalEstimate(encoded, decoded.contract, claim)


@dataclass(frozen=True, slots=True)
class AdjustmentProviderAdapter:
    """Runtime bridge retaining the adjustment receipt and trust boundary."""

    handoff: AdjustmentHandoff
    executor: Any = field(repr=False, compare=False)

    @property
    def spec(self):
        return self.handoff.spec

    def execute(self, request: Mapping[str, Any]):
        from .extensibility import ProviderExecution

        if "data" not in request:
            raise _refuse(
                "handoff.request_data_missing",
                "adjustment provider request requires caller data",
                offending="data",
                remedy="pass the caller's columns as request['data']",
            )
        columns = self.handoff.columns(request["data"])
        result = self.executor(columns, request)
        if not isinstance(result, Mapping) or "effect" not in result or "learner" not in result:
            raise _refuse(
                "handoff.executor_result_invalid",
                "adjustment executor must return learner and effect",
                offending="executor",
                remedy="return a mapping with at least 'learner' and 'effect'",
            )
        allowed = {
            "learner",
            "learner_config",
            "effect",
            "interval",
            "label",
            "value_types",
            "contrast",
        }
        unknown = set(result) - allowed
        if unknown:
            raise _refuse(
                "handoff.executor_fields_unsupported",
                f"adjustment executor returned unsupported fields: {sorted(unknown)}",
                offending=", ".join(sorted(unknown)),
                remedy="return only: " + ", ".join(sorted(allowed)),
            )
        receipt = self.handoff.attach(
            learner=result["learner"],
            learner_config=result.get("learner_config"),
            effect=result["effect"],
            interval=result.get("interval"),
            data=request["data"],
            label=result.get("label"),
            value_types=result.get("value_types"),
            contrast=result.get("contrast"),
        )
        tool = self.handoff.tool
        adapter_id = _DEFAULT_ADAPTER_ID if tool == _DEFAULT_TOOL else tool
        return ProviderExecution(
            estimate=result["effect"],
            uncertainty=None,
            assumptions=self.spec.identification_requirements,
            support_status=self.handoff.status,
            provenance={
                "adapter": adapter_id,
                "tool": tool,
                "claim_id": receipt.claim_id,
            },
            artifact=bytes(receipt.export()),
        )


@dataclass(frozen=True, slots=True)
class ExternalEstimate:
    """Caller-attested external-estimate receipt. Not a native calibrated claim.

    :attr:`trust` is ``externally_attested``. :meth:`decision_input` is the bridge
    into a decision; the receipt has no exact-request verification.
    """

    _bytes: bytes = field(repr=False)
    _contract: Mapping[str, Any] = field(repr=False)
    _claim: Any | None = field(default=None, repr=False, compare=False)

    @property
    def claim(self) -> BoundExternalClaim | None:
        """The bound claim of a labelled effect, else ``None``.

        The same object ``external.bind`` returns: a
        :class:`~antecedent.external.BoundExternalClaim` for a scalar ``label="ate"``
        and a :class:`~antecedent.external.BoundExternalColumn` for a row-level
        ``label="cate"``. It carries the learner as a provider (with its declared
        version), support labels and the lineage back to the identified contract.
        An unlabelled payload asserts nothing, so it has no claim.
        """
        return cast("BoundExternalClaim | None", self._claim)

    @property
    def support(self) -> tuple[str, ...]:
        """Support labels of :attr:`claim` (``missing_evidence`` if undeclared).

        One per coordinate for a scalar; a row-level column reports its worst row label,
        and ``claim.support_counts`` has the rows per label.
        """
        return () if self._claim is None else self._claim.support

    @property
    def lineage(self):
        """Derivation chain of :attr:`claim`, parents before children."""
        return () if self._claim is None else self._claim.lineage

    @property
    def provider_label(self) -> str | None:
        """The external execution (learner, version, snapshot) that supplied the value."""
        return None if self._claim is None else self._claim.provenance_label

    def _export_claim(self, *, artifact_id: str = "external-claim") -> ExportedArtifact:
        """The bound claim as an ``external_response_claim_v1`` artifact.

        :meth:`export` keeps the attested-estimate receipt. Reload this one with
        :meth:`antecedent.external.ExternalSpec.load` and ``claim.identity``.
        """
        if self._claim is None:
            raise _refuse(
                "handoff.no_bound_claim",
                "this estimate has no bound claim: only an effect labelled label='ate' "
                "(scalar) or label='cate' (one value per row) is bound",
                code="attested_not_reverifiable",
                offending="label",
                remedy="attach the effect with label='ate' or label='cate'",
            )
        return artifact_from_receipt(self._claim, self._claim.export(artifact_id=artifact_id))

    @property
    def trust(self):
        """Caller attestation. Never ``verified_extension`` or ``native_licensed``."""
        from .extensibility import ProviderTrust

        return ProviderTrust.EXTERNALLY_ATTESTED

    @property
    def value(self) -> float | None:
        """Scalar effect when the caller labelled one; otherwise ``None``."""
        import struct
        from math import isfinite

        claim = self._contract.get("claim")
        if not isinstance(claim, Mapping):
            return None
        bits = claim.get("value_bits")
        if not isinstance(bits, int):
            return None
        value = struct.unpack("<d", struct.pack("<Q", bits & 0xFFFFFFFFFFFFFFFF))[0]
        if not isfinite(value):
            return None
        return float(value)

    @property
    def attestor(self) -> str:
        """Learner name recorded on the attested evidence row."""
        claim = self._contract.get("claim")
        attested = claim.get("attested") if isinstance(claim, Mapping) else None
        if isinstance(attested, list) and attested and isinstance(attested[0], Mapping):
            name = attested[0].get("name")
            if isinstance(name, str) and name.strip():
                return name
        return "external-estimate"

    @property
    def snapshot_id(self) -> str:
        """Data-snapshot identity, or the claim id when the contract stores none as text."""
        identities = self._contract.get("identities")
        raw = identities.get("data_snapshot") if isinstance(identities, Mapping) else None
        if isinstance(raw, str) and raw.strip():
            return raw
        if isinstance(raw, list) and raw:
            return bytes(int(item) & 0xFF for item in raw).hex()
        return self.claim_id or "external-estimate"

    def decision_input(self, input_id: str, quantity: Any, **kwargs: Any) -> Any:
        """A decision input at ``quantity``, attested and not exact-request verified."""
        from .composition import DecisionInput

        return DecisionInput.from_estimate(input_id, self, quantity, **kwargs)

    @property
    def claim_id(self) -> str:
        from .results._slots import _identity_hex

        claim = self._contract.get("claim")
        raw = claim.get("claim_id") if isinstance(claim, Mapping) else None
        return _identity_hex(raw) or ""

    @property
    def calibration(self):
        from .results._execution import CalibrationInfo

        return CalibrationInfo.from_contract(self._contract)

    def inspect(self):
        from .results._report import as_inspection
        from .results._slots import ReasoningSlots

        return as_inspection(
            ReasoningSlots.from_result_section(self._contract, calibration=self.calibration)
        )

    @artifact_export
    def export(
        self, *, quantity: Literal["estimate", "claim"] = "estimate", artifact_id: str | None = None
    ) -> ExportedArtifact:
        """Export the attested receipt or its separately bound response claim."""
        if quantity == "claim":
            return self._export_claim(
                artifact_id="external-claim" if artifact_id is None else artifact_id
            )
        if quantity != "estimate":
            raise CausalTypeError("quantity must be 'estimate' or 'claim'")
        if artifact_id not in (None, "external-estimate"):
            raise _refuse(
                "handoff.artifact_id_immutable",
                "An attested external estimate is immutable; its artifact ID cannot be rewritten.",
                offending="artifact_id",
                remedy="export the receipt unchanged, or attach the estimate again",
            )
        return artifact_from_receipt(self, self._bytes)

    def __repr__(self) -> str:
        return f"<ExternalEstimate claim_id={self.claim_id[:12]} calibration={self.calibration.status}>"


def adjustment(
    result: AnalysisResult | CausalResponseView | Identification | IdentifyResult,
    *,
    treatment: str | tuple[str, ...] | None = None,
    outcome: str | None = None,
    structure_source: str | None = None,
    modifiers: str | tuple[str, ...] | Sequence[str] | None = None,
    target_weights: Any | None = None,
    target_depends_on: Sequence[str] = (),
    outcome_functional: Any | None = None,
    tool: str = _DEFAULT_TOOL,
) -> AdjustmentHandoff:
    """Emit an adjustment-set handoff, or refuse.

    Licensed only for point-identified backdoor / generalized-adjustment
    estimands on an explicit or accepted graph. Front-door, IV, general ID,
    general temporal ID functionals,
    partial identification, and graph-posterior mixtures refuse rather than
    pretending they are a single adjustment set.

    ``tool`` is recorded as the provider spec's adapter identity. The default
    is the historical EconML spec string.
    """
    status, method, adjustment_set, identifier, treatment, outcome = _require_adjustment_parent(
        result, treatment=treatment, outcome=outcome, structure_source=structure_source
    )
    resolved = identifier or method
    query = getattr(result, "query", getattr(result, "estimand", None))
    certificate = getattr(result, "certificate", None)
    temporal = resolved == Identifier.TEMPORAL_BACKDOOR_UNFOLDED or bool(
        certificate and str(certificate["graph_class"]).lower().startswith("temporal")
    )
    kwargs: dict[str, Any] = {}
    if certificate:
        sets = [
            case["adjustment_coordinates"][0]
            for case in certificate["cases"]
            if case["adjustment_coordinates"]
        ]
        if not sets or not all(item == sets[0] for item in sets):
            raise _refuse(
                "handoff.no_common_adjustment_set",
                "adjustment handoff requires a common certified adjustment set",
                code="route_not_supported",
                remedy="identify one adjustment set valid for every case, or hand off per case",
            )
        adjustment_set = [item["name"] for item in sets[0]]
        kwargs["confounder_offsets"] = tuple(item["offset"] for item in sets[0])
        kwargs["treatment_offsets"] = tuple(item["offset"] for item in certificate["treatments"])
        kwargs["outcome_offset"] = certificate["outcome"]["offset"]
    elif temporal:
        raise _refuse(
            "handoff.temporal_certificate_missing",
            "adjustment handoff needs an identification certificate preserving temporal offsets; "
            "use identify(graph=..., query=...)",
            code="route_not_supported",
            remedy="call identify(graph=..., query=...) and hand off its result",
        )
    declared = _modifier_set(query, modifiers)
    forbidden = {treatment} if isinstance(treatment, str) else set(treatment)
    forbidden.add(outcome)
    if any(name in forbidden for name in declared):
        raise _refuse(
            "handoff.modifier_not_pretreatment",
            "adjustment handoff modifiers must be pre-treatment and not the outcome",
            offending="modifiers",
            remedy="name modifiers measured before treatment, other than the treatment or outcome",
        )
    certified_modifiers = set(adjustment_set) | set(_modifier_set(query, None))
    if any(name not in certified_modifiers for name in declared):
        raise _refuse(
            "handoff.modifier_uncertified",
            "adjustment handoff modifiers need a pre-treatment certificate: "
            "identify the modifier context first or use certified adjustment variables",
            code="route_not_supported",
            offending="modifiers",
            remedy="identify the modifier context first, or use certified adjustment variables",
        )
    if any(name not in set(adjustment_set) or name in forbidden for name in target_depends_on):
        raise _refuse(
            "handoff.target_weights_dependence",
            "target weights must depend only on certified adjustment variables",
            offending="target_depends_on",
            remedy="declare only certified adjustment variables in target_depends_on",
        )
    if target_weights is not None:
        weights = np.asarray(target_weights, dtype=float)
        if (
            weights.ndim != 1
            or not weights.size
            or not np.isfinite(weights).all()
            or (weights < 0).any()
            or weights.sum() <= 0
        ):
            raise _refuse(
                "handoff.target_weights_invalid",
                "target weights must be a finite non-negative vector with positive mass",
                offending="target_weights",
                remedy="pass a one-dimensional finite non-negative vector with positive total",
            )
        if not target_depends_on and not np.all(weights == weights[0]):
            raise _refuse(
                "handoff.target_weights_undeclared",
                "nonconstant target weights require declared target_depends_on",
                offending="target_depends_on",
                remedy="name the adjustment variables the weights depend on",
            )
    if outcome_functional is None:
        outcome_functional = getattr(query, "outcome_functional", None)
    return AdjustmentHandoff(
        treatment=treatment,
        outcome=outcome,
        confounders=tuple(adjustment_set),
        identifier=resolved,
        status=status,
        modifiers=declared,
        temporal=temporal,
        target_weights=None if target_weights is None else np.asarray(target_weights),
        outcome_functional=outcome_functional,
        tool=tool,
        _parent=result,
        **kwargs,
    )


def _modifier_set(
    query: Any, modifiers: str | tuple[str, ...] | Sequence[str] | None
) -> tuple[str, ...]:
    if modifiers is None:
        single = getattr(query, "modifier", None)
        many = getattr(query, "effect_modifiers", None)
        if isinstance(single, str):
            return (single,)
        if many:
            return tuple(str(name) for name in many)
        return ()
    if isinstance(modifiers, str):
        return (modifiers,)
    return tuple(str(name) for name in modifiers)


def _require_attachable(spec: AdjustmentHandoff) -> None:
    if spec._parent is not None:
        _require_adjustment_parent(spec._parent)
        return
    if spec.identifier not in _ADJUSTMENT_IDENTIFIERS:
        raise _refuse(
            "handoff.identifier_not_adjustment",
            "adjustment handoff requires a backdoor or generalized-adjustment "
            f"identifier; got {spec.identifier!r}",
            code="route_not_supported",
            offending=str(spec.identifier),
            remedy="hand off a backdoor or generalized-adjustment estimand",
        )
    view = IdentificationView(
        status=spec.status,
        method=spec.identifier,
        adjustment_set=list(spec.confounders),
        assumption_count=0,
        derivation_step_count=0,
    )
    if not view:
        raise _refuse(
            "handoff.not_point_identified",
            f"adjustment handoff requires point identification; got status {spec.status!r}",
            code="effect_not_identified",
            offending=str(spec.status),
            remedy="strengthen the identification to point identification before the handoff",
        )


def _require_adjustment_parent(
    result: object,
    *,
    treatment: str | tuple[str, ...] | None = None,
    outcome: str | None = None,
    structure_source: str | None = None,
) -> tuple[str, str, list[str], str | None, str | tuple[str, ...], str]:
    if structure_source == "graph_posterior" or _graph_posterior_result(result):
        raise _refuse(
            "handoff.graph_posterior_mixture",
            "adjustment handoff refuses graph-posterior mixtures; there is no single "
            "adjustment set to pass to another estimator",
            code="route_not_supported",
            remedy="accept one graph and identify on it, then hand that result off",
        )
    status, method, adjustment_set, identifier, treatment, outcome = _unpack(
        result, treatment=treatment, outcome=outcome
    )
    resolved = identifier or method
    if resolved not in _ADJUSTMENT_IDENTIFIERS:
        raise _refuse(
            "handoff.identifier_not_adjustment",
            "adjustment handoff requires a backdoor or generalized-adjustment "
            f"identifier; got {resolved!r}",
            code="route_not_supported",
            offending=str(resolved),
            remedy="hand off a backdoor or generalized-adjustment estimand",
        )
    view = IdentificationView(
        status=status,
        method=method,
        adjustment_set=list(adjustment_set),
        assumption_count=0,
        derivation_step_count=0,
    )
    if not view:
        raise _refuse(
            "handoff.not_point_identified",
            f"adjustment handoff requires point identification; got status {status!r}",
            code="effect_not_identified",
            offending=str(status),
            remedy="strengthen the identification to point identification before the handoff",
        )
    return status, method, adjustment_set, identifier, treatment, outcome


def _row_effects(
    spec: AdjustmentHandoff, effect: np.ndarray, data: Mapping[str, Any] | None
) -> np.ndarray:
    """A row-level effect column: one-dimensional, non-empty and aligned to the handoff's rows."""
    if effect.ndim != 1 or not effect.size:
        raise _refuse(
            "handoff.label_cate_not_column",
            "label='cate' requires a one-dimensional effect with one value per row",
            offending="effect",
            remedy="pass the learner's per-row conditional effects as a 1-D array, or label "
            "a single number label='ate'",
        )
    if data is not None:
        rows = len(spec.columns(data)["Y"])
        if effect.size != rows:
            raise _refuse(
                "handoff.cate_rows_misaligned",
                f"label='cate' effect has {effect.size} values for {rows} aligned rows",
                code="schema_mismatch",
                offending="effect",
                remedy="produce one effect per row of spec.columns(data), in that order",
            )
    return effect


def _bind_claim(
    spec: AdjustmentHandoff,
    *,
    learner: str,
    config: bytes,
    payload: bytes,
    value: float | np.ndarray,
    interval: bool,
    identification: str | None,
    snapshot: str | None,
    snapshot_payload: bytes | None,
    contrast: tuple[float, float] | None,
    learner_version: str | None,
    outcome_units: str | None,
    support: str | Sequence[str] | None,
) -> BoundExternalClaim:
    """Bind a labelled effect through ``external.bind``; every refusal is Rust's.

    A scalar is one coordinate (functional ``ate``). A row-level array is one
    quantity (functional ``cate``) bound as a column, so a CATE of any length is a
    single claim and never one coordinate per row.
    """
    config = container_bytes(config)
    payload = container_bytes(payload)
    snapshot_payload = container_bytes(snapshot_payload) if snapshot_payload is not None else None
    from dataclasses import replace

    from . import _native
    from .external import (
        UNSPECIFIED_UNITS,
        ColumnResponse,
        ExternalSpec,
        ProviderObject,
        Response,
        _status_name,
    )
    from .joint_distribution import ScientificQuantity

    is_column = isinstance(value, np.ndarray)
    if not np.isfinite(value).all():
        raise _refuse(
            "handoff.effect_not_finite",
            "a labelled effect must be finite; refusing to bind a non-finite effect",
            offending="effect",
            remedy="pass finite effect values",
        )
    treatments = ",".join(spec.treatments)
    if contrast is not None and len(spec.treatments) == 1:
        (control, active), name = contrast, spec.treatments[0]
        regime = f"do({name}={active:g}) - do({name}={control:g})"
    else:
        # The caller named no contrast: the placeholder is recorded as such, never assumed.
        regime = f"do({treatments}) contrast unspecified"
    transform = "identity"
    if spec.outcome_functional is not None:
        from .query import coerce_outcome_functional

        functional = coerce_outcome_functional(spec.outcome_functional)
        if functional is not None:
            transform = str(functional["kind"])
    coordinate = ScientificQuantity(
        variable_id=spec.outcome,
        variable_name=spec.outcome,
        role="outcome",
        units=(outcome_units or "").strip() or UNSPECIFIED_UNITS,
        population_id="target",
        regime_id=regime,
        horizon=0,
        functional_id="cate" if is_column else "ate",
        conditioning=(),
        transform_id=transform,
    )
    snapshot_id = snapshot or _native.blake3_digest_hex(snapshot_payload or b"")
    contract_name = {
        "identifier": spec.identifier,
        "treatments": list(spec.treatments),
        "outcome": spec.outcome,
        "adjustment_set": list(spec.confounders),
        "modifiers": list(spec.modifiers),
        "offsets": [
            list(spec.treatment_offsets),
            spec.outcome_offset,
            list(spec.confounder_offsets),
        ],
    }
    graph_id = "handoff:" + canonical_digest(
        "antecedent.handoff.adjustment_contract.v1", contract_name
    )
    external_spec = ExternalSpec(
        contract_id="",
        graph_id=graph_id,
        identification=_status_name(spec.status),
        quantities=(coordinate,),
        statement=f"Effect of {treatments} on {spec.outcome} under {spec.identifier}",
    )
    contract_id = (
        f"contract:{identification}"
        if identification
        else external_spec._content_id(with_coordinates=True)
    )
    external_spec = replace(external_spec, contract_id=contract_id)
    request = canonical_digest(
        "antecedent.handoff.request.v1",
        {
            "learner": learner,
            "config": _native.blake3_digest_hex(config),
            "payload": _native.blake3_digest_hex(payload),
        },
    )
    provider = ProviderObject(
        provider_id=learner,
        object_id=f"{'cate' if is_column else 'effect'}:{spec.outcome}~{treatments}",
        version=(learner_version or "").strip() or UNSPECIFIED_UNITS,
        snapshot=snapshot_id,
        request=request,
        meaning="interventional_predictive",
        capabilities=("mean",),
    )
    method = "caller_attested_interval" if interval else None
    assumptions = (spec.identifier, "point_identified")
    if is_column:
        return external_spec.bind(
            ColumnResponse(
                provider=provider,
                values=np.asarray(value, dtype=np.float64),
                assumptions=assumptions,
                attested_by=learner,
                uncertainty_method=method,
                support=None if support is None else _support(support),
            )
        )
    if isinstance(support, str):
        raise _refuse(
            "handoff.support_not_a_sequence",
            "a scalar effect's support is a sequence with one label",
            offending="support",
            remedy="pass support=['supported'] (one label per coordinate)",
        )
    return external_spec.bind(
        Response(
            provider=provider,
            values=(float(value),),
            assumptions=assumptions,
            attested_by=learner,
            uncertainty_method=method,
            support=None if support is None else tuple(support),
        )
    )


def _support(support: str | Sequence[str]) -> str | tuple[str, ...]:
    return support if isinstance(support, str) else tuple(support)


def _receipt_names(spec: AdjustmentHandoff) -> list[str]:
    names: list[str] = []
    for name in (spec.outcome, *spec.treatments, *spec.confounders, *spec.modifiers):
        if name not in names:
            names.append(name)
    return names


def _canonical_config(config: Mapping[str, Any] | None) -> bytes:
    return json.dumps(
        {} if config is None else dict(config),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def _payload_bytes(effect: np.ndarray, interval: Any | None) -> bytes:
    parts = [
        np.asarray(effect.shape, dtype=np.int64).tobytes(),
        np.asarray(effect, dtype=np.float64, order="C").tobytes(),
    ]
    if interval is not None:
        arr = np.asarray(interval, dtype=np.float64)
        parts.extend(
            [
                b"interval",
                np.asarray(arr.shape, dtype=np.int64).tobytes(),
                np.asarray(arr, dtype=np.float64, order="C").tobytes(),
            ]
        )
    return b"".join(parts)


def _snapshot_payload(spec: AdjustmentHandoff, data: Mapping[str, Any]) -> bytes:
    cols = spec.columns(data)
    chunks: list[bytes] = []
    for key in ("Y", "T", "W", "X", "sample_weight"):
        value = cols.get(key)
        if value is None:
            continue
        arr = np.asarray(value, dtype=np.float64)
        chunks.append(
            key.encode()
            + np.asarray(arr.shape, dtype=np.int64).tobytes()
            + np.asarray(arr, dtype=np.float64, order="C").tobytes()
        )
    return b"".join(chunks)


def _bind_parent_identities(
    spec: AdjustmentHandoff, parent: object, data: Mapping[str, Any] | None
) -> tuple[str | None, str | None, bytes | None]:
    from . import artifacts
    from .results._slots import _identity_hex

    identification = None
    snapshot = None
    export = getattr(parent, "export", None)
    if callable(export):
        try:
            decoded = artifacts.loads(export())
        except (TypeError, ValueError, CausalError):
            decoded = None
        if decoded is not None and isinstance(decoded.contract, Mapping):
            identities = decoded.contract.get("identities")
            identities = identities if isinstance(identities, Mapping) else {}
            identification = _identity_hex(identities.get("identification"))
            snapshot = _identity_hex(identities.get("data_snapshot"))
    snapshot_payload = (
        None if snapshot is not None or data is None else _snapshot_payload(spec, data)
    )
    if snapshot is None and snapshot_payload is None:
        raise _refuse(
            "handoff.snapshot_missing",
            "external estimate attach needs a parent data snapshot or data= for spec.columns",
            offending="data",
            remedy="pass data= (the columns the learner was fitted on), or attach through a "
            "handoff derived from an analysis result",
        )
    return identification, snapshot, snapshot_payload


def _unpack(
    result: object,
    *,
    treatment: str | tuple[str, ...] | None,
    outcome: str | None,
) -> tuple[str, str, list[str], str | None, str | tuple[str, ...], str]:
    if not isinstance(result, (AnalysisResult, CausalResponseView, Identification, IdentifyResult)):
        raise _refuse(
            "handoff.result_type",
            f"adjustment handoff expects an identification or analysis result; got {type(result)!r}",
            offending=type(result).__name__,
            remedy="pass the result of identify(...) or analyze(...)",
        )
    query = getattr(result, "query", getattr(result, "estimand", None))
    certificate = getattr(result, "certificate", None)
    certified_t = getattr(query, "treatment", None)
    certified_y = getattr(query, "outcome", None)
    if certificate:
        names = tuple(item["name"] for item in certificate["treatments"])
        certified_t = names[0] if len(names) == 1 else names
        certified_y = certificate["outcome"]["name"]
    elif query is not None and hasattr(query, "interventions"):
        names = tuple(item.variable for item in query.interventions)
        certified_t = names[0] if len(names) == 1 else names
    if not certified_t or not certified_y:
        raise _refuse(
            "handoff.query_identity_missing",
            "adjustment handoff needs a result carrying its certified treatment and outcome; "
            "caller-supplied names cannot establish query identity",
            offending="result",
            remedy="hand off the result of identify(...) or analyze(...), not bare names",
        )
    requested = (treatment,) if isinstance(treatment, str) else treatment
    expected = (certified_t,) if isinstance(certified_t, str) else certified_t
    if (requested is not None and requested != expected) or (
        outcome is not None and outcome != certified_y
    ):
        raise _refuse(
            "handoff.query_identity_mismatch",
            "adjustment handoff treatment/outcome must match the certified query, "
            "including every joint treatment",
            code="external_binding_mismatch",
            offending="treatment" if requested != expected else "outcome",
            remedy="omit treatment= and outcome=, or pass exactly the certified names",
        )
    if isinstance(result, (AnalysisResult, CausalResponseView)):
        ident = result.identification
        plan = getattr(result, "plan", None)
        plan_id = plan.identifier if plan is not None else None
        if plan_id is None and certificate:
            plan_id = certificate["method"]
        return (
            ident.status,
            ident.method,
            list(ident.adjustment_set),
            plan_id,
            certified_t,
            certified_y,
        )
    return (
        result.status,
        result.method,
        list(result.adjustment_set),
        getattr(result, "identifier", None),
        certified_t,
        certified_y,
    )


def _graph_posterior_result(result: object) -> bool:
    plan = getattr(result, "plan", None)
    source = getattr(plan, "structure_source", None) if plan is not None else None
    if source == "graph_posterior":
        return True
    algo = getattr(plan, "discovery_algorithm", None) if plan is not None else None
    return isinstance(algo, str) and any(
        marker in algo.lower() for marker in _GRAPH_POSTERIOR_MARKERS
    )


# EconML names remain callable aliases. They are the same objects.
EconMLSpec = AdjustmentHandoff
EconMLProviderAdapter = AdjustmentProviderAdapter
econml = adjustment

__all__ = [
    "AdjustmentHandoff",
    "AdjustmentProviderAdapter",
    "EconMLProviderAdapter",
    "EconMLSpec",
    "ExternalEstimate",
    "adjustment",
    "econml",
]
