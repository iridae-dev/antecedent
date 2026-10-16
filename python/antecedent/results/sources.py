"""Canonical mean-only sources and checked adapters for scientific producers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .._columns import coerce_fields
from .._scientific import ScientificResult
from .._validate import real
from ..errors import CausalTypeError, CausalUnsupportedError, CausalValueError
from ..joint_distribution import ScientificQuantity


@dataclass(repr=False, frozen=True, slots=True)
class MeanSource(ScientificResult):
    """One mean per coordinate, with the provenance the decision records.

    It answers only the expectation of an affine utility: any other criterion refuses
    with ``decision_contract_unsatisfied``, because a mean is never an outcome law.
    Build it from a native response with
    :meth:`antecedent.program_claims.NativeClaim.as_decision_source`.
    """

    coordinates: tuple[ScientificQuantity, ...]
    means: tuple[float, ...]
    provider_id: str
    snapshot_id: str
    causal_contract_id: str
    rng_id: str = "none:mean_grid"
    _original_native: Any = field(default=None, repr=False, compare=False)
    _original_effect: Any = field(default=None, repr=False, compare=False)
    _original_cohort: Any = field(default=None, repr=False, compare=False)
    _original_external: Any = field(default=None, repr=False, compare=False)
    _original_producer: Any = field(default=None, repr=False, compare=False)
    evidence: tuple[str, ...] = ()
    lineage: tuple[Any, ...] = ()
    trust: str = "externally_attested"

    def __post_init__(self) -> None:
        coerce_fields(self, coordinates=tuple(self.coordinates))
        coerce_fields(self, means=tuple(real(m, "mean", finite=True) for m in self.means))
        if len(self.coordinates) != len(self.means) or not self.coordinates:
            raise CausalValueError("a mean source needs one mean per scientific coordinate")
        ScientificResult.__post_init__(self)
        if self.trust in ("native_licensed", "verified_extension"):
            from .. import _native

            native_authority = isinstance(
                getattr(self._original_native, "_native", None), _native.NativeResponseClaim
            )
            if self._original_effect is not None:
                native_authority |= isinstance(
                    getattr(self._original_effect[0], "_raw", None), _native.AteAnalysisResult
                )
            if self._original_cohort is not None:
                from ..cohort_decision import consume_cohort_decision_source

                checked = consume_cohort_decision_source(
                    self._original_cohort.export(), expected_identity=self._original_cohort.identity
                )
                native_authority |= checked.is_native
            verified_authority = False
            external_native = getattr(self._original_external, "_native", None)
            if isinstance(external_native, _native.ExternalClaimArtifact):
                verified_authority = (
                    json.loads(external_native.metadata_json)["identity"]["trust"]
                    == "verified_extension"
                )
            if not (native_authority if self.trust == "native_licensed" else verified_authority):
                raise CausalValueError(
                    "a mean declaration cannot issue native_licensed or verified_extension trust; use the checked native or verified-source adapter that retains its authority"
                )

    def _wire(self) -> dict[str, Any]:
        return {
            "coordinates": [q._wire() for q in self.coordinates],
            "means": list(self.means),
            "provider_id": self.provider_id,
            "snapshot_id": self.snapshot_id,
            "causal_contract_id": self.causal_contract_id,
            "rng_id": self.rng_id,
        }

    @classmethod
    def from_external(cls, claim: Any) -> MeanSource:
        identity = claim.identity_fields
        return cls(
            tuple(claim.quantities),
            tuple(float(v) for v in claim.values),
            str(identity["provider_id"]),
            str(identity["snapshot_id"]),
            str(identity["causal_contract_id"]),
            _original_external=claim,
            evidence=tuple(claim._meta.get("evidence_ids", ())),
            lineage=tuple(claim.lineage),
            trust=str(claim.trust),
        )

    def _validated_original(self):
        if self._original_producer is not None:
            producer, quantity, selected_field, contrast = self._original_producer
            actual = adapt_mean_source(
                producer, quantity=quantity, field=selected_field, contrast=contrast
            )
            if (
                self._wire() != actual._wire()
                or self.trust != actual.trust
                or self.evidence != actual.evidence
                or self.lineage != actual.lineage
            ):
                raise CausalValueError("mean source differs from its retained producer binding")
            return None
        if self._original_cohort is not None:
            actual = self._original_cohort._verified_mean_source()
            supplied = {
                "coordinates": [q._wire() for q in self.coordinates],
                "means": list(self.means),
                "provider_id": self.provider_id,
                "snapshot_id": self.snapshot_id,
                "causal_contract_id": self.causal_contract_id,
                "rng_id": self.rng_id,
            }
            if json.loads(json.dumps(supplied)) != json.loads(json.dumps(actual)):
                raise CausalValueError(
                    "a cohort mean source must retain the fields its source replays to"
                )
            return None
        if self._original_effect is not None:
            original, value, units, population = self._original_effect
            raw = original._raw
            wire, refusal = raw.effect_source_json(units, population, value)
            if refusal is not None:
                from ..external import ExternalRefusal

                raise ExternalRefusal(json.loads(refusal))
            actual = json.loads(wire)
            actual.pop("query", None)
            supplied = {
                "coordinates": [q._wire() for q in self.coordinates],
                "means": list(self.means),
                "provider_id": self.provider_id,
                "snapshot_id": self.snapshot_id,
                "causal_contract_id": self.causal_contract_id,
                "rng_id": self.rng_id,
            }
            if json.loads(json.dumps(supplied)) != actual:
                raise CausalValueError(
                    "a native effect mean source must retain its original fields"
                )
            return raw
        if self._original_native is None:
            return None
        from .. import _native

        return _native.source_backed_native_mean(
            self._original_native._native,
            json.dumps(
                {
                    "coordinates": [q._wire() for q in self.coordinates],
                    "means": list(self.means),
                    "provider_id": self.provider_id,
                    "snapshot_id": self.snapshot_id,
                    "causal_contract_id": self.causal_contract_id,
                    "rng_id": self.rng_id,
                }
            ),
        )


_MEAN_PRODUCERS = {
    "MeasurementEstimate",
    "ConsumedMeasurement",
    "SynthesisOutcome",
    "BridgedClaim",
    "ObservationClaim",
    "PropagationClaim",
    "TransportedPathSpecificEffect",
    "GaussianFunctionalResult",
    "PreparedMechanismComposition",
    "ConsumedBridge",
    "ConsumedObservationClaim",
    "ConsumedJointLawPropagation",
    "ConsumedComposition",
}


def is_mean_producer(source: Any) -> bool:
    return isinstance(source, ScientificResult) and type(source).__name__ in _MEAN_PRODUCERS


def adapt_mean_source(
    source: Any,
    *,
    quantity: ScientificQuantity | None = None,
    field: str | None = None,
    contrast: Any = None,
) -> MeanSource:
    """Bind a producer's actual mean to an explicit compatible scientific coordinate.

    Missing units, population, regime or functional semantics must be declared by the caller.
    The adapter does not turn a declaration into native authority or carry an interval.
    """
    if not is_mean_producer(source):
        raise CausalTypeError("this object has no supported mean-source adapter")
    from .._mean_binding import check_mean_reading

    check_mean_reading(source)
    original = source
    scope = getattr(source, "_scope", None)
    if type(source).__name__ in ("ConsumedBridge", "ConsumedObservationClaim"):
        source = source.claim
    if type(source).__name__ == "ConsumedComposition":
        source = source.result
    source_name = type(source).__name__
    if source_name in ("PropagationClaim", "ConsumedJointLawPropagation") and source.is_envelope:
        raise CausalUnsupportedError(
            "a dependence envelope is not a mean source", reason_code="cell_not_licensed"
        )
    if source_name == "SynthesisOutcome" and not (
        hasattr(source.value, "estimate") or hasattr(source.value, "mean")
    ):
        raise CausalValueError(
            "this synthesis carries a range, not a scalar mean; a quantity declaration cannot change that reading"
        )
    if type(original).__name__ == "ConsumedComposition" and not (
        hasattr(source, "point") or hasattr(source, "mean")
    ):
        raise CausalValueError(
            "this composed result has no scalar mean; a quantity declaration cannot turn category probabilities into a mean"
        )
    retained = getattr(source, "quantity", None)
    if quantity is None:
        quantity = retained
    if not isinstance(quantity, ScientificQuantity):
        raise CausalValueError(
            "this producer needs an explicit scientific quantity; call "
            "producer.as_mean_source(quantity=ScientificQuantity.of(...))",
            reason_code="invalid_argument",
        )
    if retained is not None and retained._wire() != quantity._wire():
        raise CausalValueError("declared quantity differs from the producer's retained quantity")
    if scope is not None:
        if quantity.population_id != scope["population"]:
            raise CausalValueError("mean-source population differs from the composed model")
        if quantity.units != scope["units"].get(quantity.variable_id):
            raise CausalValueError("mean-source units or outcome differ from the composed model")
    units = getattr(source, "units", None)
    if units is not None and units != quantity.units:
        raise CausalValueError("mean-source units differ from the producer; no units are converted")
    name = type(source).__name__
    model = getattr(source, "model", None)
    if model is not None and name in ("MeasurementEstimate", "ConsumedMeasurement"):
        for label, declared, retained_value in (
            ("population", quantity.population_id, model.population_id),
            ("regime", quantity.regime_id, model.regime_id),
            ("units", quantity.units, model.construct.units),
        ):
            if declared != retained_value:
                raise CausalValueError(f"mean-source {label} differs from the measurement model")
    if name == "PreparedMechanismComposition":
        if contrast is None:
            raise CausalValueError(
                "a composed model needs contrast=: state the interventional mean contrast"
            )
        if quantity.variable_id != contrast.outcome:
            raise CausalValueError("mean-source outcome differs from the composed contrast")
        value = source.mean_contrast(contrast)
    elif name in ("PropagationClaim", "ConsumedJointLawPropagation"):
        if source.is_envelope:
            raise CausalUnsupportedError(
                "a dependence envelope is not a mean source", reason_code="cell_not_licensed"
            )
        key = field or "mean"
        if key not in source.values or key != "mean":
            raise CausalValueError("the propagation claim must actually carry a mean")
        value = source.values[key]
    elif name == "ObservationClaim":
        if field is None or field not in ("crossing_probability", "evidence_probability"):
            raise CausalValueError(
                "an observation claim needs field='crossing_probability' or 'evidence_probability'"
            )
        value = getattr(source, field)
        if value is None:
            raise CausalValueError(f"this observation claim carries no {field}")
        if quantity.functional_id not in ("probability", field):
            raise CausalValueError(
                "an observation probability needs a probability functional coordinate"
            )
    elif name == "SynthesisOutcome":
        value = getattr(source.value, "estimate", getattr(source.value, "mean", None))
        if value is None:
            raise CausalValueError("this synthesis is not a scalar mean")
    elif name == "TransportedPathSpecificEffect":
        value = source.target_contrast
    elif name == "BridgedClaim":
        if source.value_kind != "mean":
            raise CausalValueError("the bridge does not carry a mean")
        value = source.value
    elif name == "GaussianLawResult":
        value = source.mean
    else:
        value = source.point
    identity = next(
        (
            getattr(original, key)
            for key in (
                "claim_id",
                "identity",
                "model_identity",
                "source_identity",
                "premises_digest",
                "plan_identity",
            )
            if getattr(original, key, None) is not None
        ),
        None,
    )
    if identity is None:
        raise CausalValueError(
            "this detached value has no retained producer identity; adapt its source-bound result"
        )
    if not isinstance(identity, str):
        descriptor = identity
        identity = getattr(descriptor, "digest", None)
        if identity is None:
            identity = json.dumps(
                descriptor._wire()
                if hasattr(descriptor, "_wire")
                else original.to_dict().get("identity"),
                sort_keys=True,
            )
    return MeanSource(
        coordinates=(quantity,),
        means=(value,),
        _original_producer=(original, quantity, field, contrast),
        provider_id=identity,
        snapshot_id=getattr(source, "data_identity", None) or identity,
        causal_contract_id=getattr(source, "contract_identity", identity),
        evidence=tuple(getattr(source, "evidence", ())),
        lineage=tuple(getattr(source, "provenance", ()))
        or tuple(getattr(source, "lineage", ()))
        or (
            {
                "id": identity,
                "stage": "claim",
                "parents": (),
                "producer": type(original).__name__,
                "identity": identity,
                "operation": getattr(source, "operation", None),
                "evidence": tuple(getattr(source, "evidence", ())),
            },
        ),
        trust=getattr(source, "provider_trust", None) or "externally_attested",
    )
