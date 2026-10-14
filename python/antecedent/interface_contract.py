"""Scientific interface contracts.

Two components can agree on array shape, dtype and a variable name and still disagree on units,
population, intervention regime, support, the measurement model a value was corrected under,
whether the producer offers a mean where the consumer needs a law, or what its interval means. An
interface contract declares those things so a connection is accepted or refused from the two
contracts alone, before any component runs::

    from antecedent import interface_contract as ic
    from antecedent.joint_distribution import ScientificQuantity

    effect = ScientificQuantity.of(
        "outcome", "ldl_change", units="mg/dL", population="adults", regime="do(statin=1)"
    )
    estimator = ic.InterfaceContract(
        "estimator",
        produces=[ic.ProducedPort("effect", effect, "estimand:ate", ic.SupportDomain.real(),
                                  operations=("mean",))],
    )
    decision = ic.InterfaceContract(
        "decision",
        consumes=[ic.ConsumedPort("effect", effect, "estimand:ate", ic.SupportDomain.real(),
                                  operations=("mean",))],
    )
    connection = ic.check_connection(estimator, "effect", decision, "effect")

A connection never raises a calibration label: a point-only producer stays point-only, and a
consumer that relies on measured calibration is refused rather than served an unmeasured interval.
A set of components is checked with :func:`check_composition`, which refuses feedback cycles with
their path named. :func:`export_interface_contract` writes ``interface_contract_v1`` (specified in
``spec/``); :func:`consume_interface_contract_artifact` rebuilds and re-validates it against an
identity the consumer retained; :func:`inspect_interface_contract_artifact` reads the stored
metadata without replaying it.

Every decision, identity and refusal is computed in Rust; this module builds the declaration and
presents the result. Refusals are :class:`InterfaceContractRefusal` (a
:class:`~antecedent.errors.CausalUnsupportedError`) with a registered ``reason_code``, an
``interface_contract.*`` ``detail`` and a ``remedy``; ``refusals`` carries every defect found.
"""

from __future__ import annotations

import json
import math
import struct
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from . import _native
from ._refusal import StructuredRefusal
from .errors import CausalTypeError, CausalValueError
from .joint_distribution import QuantityCondition, ScientificQuantity
from .scientific_bridge import _quantity_wire

__all__ = [
    "CALIBRATION_STATUSES",
    "DEPENDENCE_REQUIREMENTS",
    "OPERATIONS",
    "UNCERTAINTY_MEANINGS",
    "Calibration",
    "ComposedInterfaces",
    "Connection",
    "ConsumedInterfaceContract",
    "ConsumedPort",
    "InterfaceContract",
    "InterfaceContractExport",
    "InterfaceContractRefusal",
    "ProducedPort",
    "SupportDomain",
    "check_composition",
    "check_connection",
    "consume_interface_contract_artifact",
    "contract_identity",
    "export_interface_contract",
    "inspect_interface_contract_artifact",
]

#: The operations a port offers or requires. None implies another.
OPERATIONS: tuple[str, ...] = (
    "point",
    "mean",
    "law",
    "draws",
    "joint_draws",
    "covariance",
    "interval",
    "posterior",
)

#: What an uncertainty statement can mean, with the paradigm of each.
UNCERTAINTY_MEANINGS: dict[str, str] = {
    "parameter_posterior": "posterior",
    "causal_functional_posterior": "posterior",
    "posterior_predictive": "posterior",
    "interventional_predictive": "posterior",
    "estimator_sampling": "sampling",
    "bootstrap": "resampling",
    "empirical_outcome": "empirical",
}

#: The calibration labels a produced quantity can hold.
CALIBRATION_STATUSES: tuple[str, ...] = (
    "none",
    "point_only",
    "unmeasured",
    "measured",
    "inherited",
)

#: What a consumed port can require of the dependence of its input.
DEPENDENCE_REQUIREMENTS: tuple[str, ...] = (
    "unconstrained",
    "declared_evidence",
    "disjoint_evidence",
)

_CALIBRATION_REQUIREMENTS = ("not_required", "measured")
_STAGES = (
    "causal_contract",
    "evidence",
    "data",
    "external_provider",
    "distribution_artifact",
    "transformation",
    "decision_contract",
    "sensitivity_input",
    "study_ranking_provider",
    "claim",
)


class InterfaceContractRefusal(StructuredRefusal):
    """A typed refusal of the interface-contract route.

    ``refusals`` lists every defect found (each itself an :class:`InterfaceContractRefusal`); the
    refusal's own ``reason_code``, ``detail`` and ``remedy`` are those of the first.
    """

    refusals: tuple[InterfaceContractRefusal, ...]


def _raise(payload: str | None) -> None:
    if payload is None:
        return
    wire = json.loads(payload)
    items = wire if isinstance(wire, list) else [wire]
    built = tuple(InterfaceContractRefusal(item) for item in items)
    first = built[0]
    first.refusals = built
    raise first


def _name(value: object, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CausalTypeError(f"{what} must be a non-empty string")
    return value


def _choice(value: object, what: str, allowed: Iterable[str]) -> str:
    text = _name(value, what)
    options = tuple(allowed)
    if text not in options:
        raise CausalValueError(f"{what} must be one of {options}, got {text!r}")
    return text


def _bits(value: float) -> int:
    return int(struct.unpack("<Q", struct.pack("<d", float(value)))[0])


def _float(bits: int) -> float:
    return float(struct.unpack("<d", struct.pack("<Q", bits))[0])


def _set(values: Iterable[str], what: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise CausalTypeError(f"{what} must be a collection of strings")
    return tuple(sorted({_name(v, what) for v in values}))


def _operations(values: Iterable[str], what: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise CausalTypeError(f"{what} must be a collection of operation names")
    chosen = {_choice(v, what, OPERATIONS) for v in values}
    return tuple(op for op in OPERATIONS if op in chosen)


# --------------------------------------------------------------------------- declarations


@dataclass(frozen=True, slots=True)
class SupportDomain:
    """The values a quantity takes (produced) or a component accepts (consumed)."""

    kind: str
    lower: float | None = None
    upper: float | None = None
    levels: tuple[str, ...] = ()

    @classmethod
    def real(cls) -> SupportDomain:
        """Any finite real."""
        return cls("real")

    @classmethod
    def non_negative(cls) -> SupportDomain:
        """A real that is zero or positive."""
        return cls("non_negative")

    @classmethod
    def binary(cls) -> SupportDomain:
        """Zero or one."""
        return cls("binary")

    @classmethod
    def interval(cls, lower: float, upper: float) -> SupportDomain:
        """A closed interval ``[lower, upper]`` (bounds are carried as bit patterns)."""
        for name, value in (("lower", lower), ("upper", upper)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise CausalTypeError(f"{name} must be a real number")
            if not math.isfinite(value):
                raise CausalValueError(f"{name} must be finite")
        return cls("interval", float(lower), float(upper))

    @classmethod
    def categorical(cls, levels: Iterable[str]) -> SupportDomain:
        """A finite set of named levels."""
        return cls("categorical", levels=tuple(sorted(set(levels))))

    def _wire(self) -> dict[str, Any]:
        if self.kind not in ("real", "non_negative", "binary", "interval", "categorical"):
            raise CausalValueError(f"unknown support kind {self.kind!r}")
        return {
            "kind": self.kind,
            "lower_bits": _bits(self.lower) if self.kind == "interval" and self.lower is not None else None,
            "upper_bits": _bits(self.upper) if self.kind == "interval" and self.upper is not None else None,
            "levels": list(self.levels) if self.kind == "categorical" else None,
        }

    @classmethod
    def _from_wire(cls, wire: Mapping[str, Any]) -> SupportDomain:
        kind = wire["kind"]
        if kind == "interval":
            return cls("interval", _float(wire["lower_bits"]), _float(wire["upper_bits"]))
        if kind == "categorical":
            return cls("categorical", levels=tuple(wire["levels"]))
        return cls(kind)


@dataclass(frozen=True, slots=True)
class Calibration:
    """A calibration label held by a produced quantity.

    The label is a declaration; ``measured`` names the record that holds coverage, ``inherited``
    names the consumed port the calibration is inherited from.
    """

    status: str = "none"
    record_id: str | None = None
    input: str | None = None

    @classmethod
    def none(cls) -> Calibration:
        return cls("none")

    @classmethod
    def point_only(cls) -> Calibration:
        return cls("point_only")

    @classmethod
    def unmeasured(cls) -> Calibration:
        return cls("unmeasured")

    @classmethod
    def measured(cls, record_id: str) -> Calibration:
        return cls("measured", record_id=record_id)

    @classmethod
    def inherited(cls, input: str) -> Calibration:  # noqa: A002
        return cls("inherited", input=input)

    def _wire(self) -> dict[str, Any]:
        _choice(self.status, "calibration status", CALIBRATION_STATUSES)
        return {"status": self.status, "record_id": self.record_id, "input": self.input}


def _uncertainty(meaning: str | None, what: str) -> dict[str, str] | None:
    if meaning is None:
        return None
    _choice(meaning, what, UNCERTAINTY_MEANINGS)
    return {"meaning": meaning, "paradigm": UNCERTAINTY_MEANINGS[meaning]}


@dataclass(frozen=True, slots=True)
class ProducedPort:
    """A quantity a component produces.

    ``evidence`` is the tuple of evidence-ledger source identities (64 hex digits) the value
    rests on, or ``None`` when undeclared. ``measurement_models`` are measurement-model
    identities. ``provenance`` names composition stages supplied.
    """

    name: str
    quantity: ScientificQuantity
    semantic_identity: str
    support: SupportDomain
    operations: tuple[str, ...]
    uncertainty: str | None = None
    calibration: Calibration = field(default_factory=Calibration)
    measurement_models: tuple[str, ...] = ()
    evidence: tuple[str, ...] | None = None
    provenance: tuple[str, ...] = ()

    def _wire(self) -> dict[str, Any]:
        _name(self.name, "port name")
        for stage in self.provenance:
            _choice(stage, "provenance stage", _STAGES)
        return {
            "name": self.name,
            "quantity": _quantity_wire(self.quantity, "quantity"),
            "semantic_identity": _name(self.semantic_identity, "semantic_identity"),
            "support": self.support._wire(),
            "operations": list(_operations(self.operations, "operations")),
            "uncertainty": _uncertainty(self.uncertainty, "uncertainty"),
            "calibration": self.calibration._wire(),
            "measurement_models": list(_set(self.measurement_models, "measurement model")),
            "evidence": {
                "declared": self.evidence is not None,
                "sources": list(_set(self.evidence or (), "evidence source")),
            },
            "provenance": list(_set(self.provenance, "provenance stage")),
        }


@dataclass(frozen=True, slots=True)
class ConsumedPort:
    """A quantity a component consumes, with what it requires of the producer.

    ``calibration`` is ``"not_required"`` or ``"measured"``; ``dependence`` is one of
    :data:`DEPENDENCE_REQUIREMENTS`.
    """

    name: str
    quantity: ScientificQuantity
    semantic_identity: str
    support: SupportDomain
    operations: tuple[str, ...]
    uncertainty: str | None = None
    calibration: str = "not_required"
    measurement_models: tuple[str, ...] = ()
    dependence: str = "unconstrained"
    provenance: tuple[str, ...] = ()

    def _wire(self) -> dict[str, Any]:
        _name(self.name, "port name")
        for stage in self.provenance:
            _choice(stage, "provenance stage", _STAGES)
        return {
            "name": self.name,
            "quantity": _quantity_wire(self.quantity, "quantity"),
            "semantic_identity": _name(self.semantic_identity, "semantic_identity"),
            "support": self.support._wire(),
            "operations": list(_operations(self.operations, "operations")),
            "uncertainty": _uncertainty(self.uncertainty, "uncertainty"),
            "calibration": _choice(self.calibration, "calibration", _CALIBRATION_REQUIREMENTS),
            "measurement_models": list(_set(self.measurement_models, "measurement model")),
            "dependence": _choice(self.dependence, "dependence", DEPENDENCE_REQUIREMENTS),
            "provenance": list(_set(self.provenance, "provenance stage")),
        }


@dataclass(frozen=True, slots=True)
class InterfaceContract:
    """What one component produces and consumes."""

    component: str
    produces: Sequence[ProducedPort] = ()
    consumes: Sequence[ConsumedPort] = ()

    def _wire(self) -> dict[str, Any]:
        produced = sorted((p._wire() for p in self.produces), key=lambda w: w["name"])
        consumed = sorted((c._wire() for c in self.consumes), key=lambda w: w["name"])
        return {
            "component": _name(self.component, "component"),
            "produces": produced,
            "consumes": consumed,
        }

    @classmethod
    def _from_wire(cls, wire: Mapping[str, Any]) -> InterfaceContract:
        def quantity(q: Mapping[str, Any]) -> ScientificQuantity:
            return ScientificQuantity(
                variable_id=q["variable_id"],
                variable_name=q["variable_name"],
                role=q["role"],
                units=q["units"],
                population_id=q["population_id"],
                regime_id=q["regime_id"],
                horizon=q["horizon"],
                functional_id=q["functional_id"],
                conditioning=tuple(
                    QuantityCondition(c["variable_id"], c["value_id"]) for c in q["conditioning"]
                ),
                transform_id=q["transform_id"],
            )

        def meaning(u: Mapping[str, Any] | None) -> str | None:
            return None if u is None else str(u["meaning"])

        produces = tuple(
            ProducedPort(
                p["name"],
                quantity(p["quantity"]),
                p["semantic_identity"],
                SupportDomain._from_wire(p["support"]),
                tuple(p["operations"]),
                meaning(p["uncertainty"]),
                Calibration(
                    p["calibration"]["status"],
                    p["calibration"]["record_id"],
                    p["calibration"]["input"],
                ),
                tuple(p["measurement_models"]),
                tuple(p["evidence"]["sources"]) if p["evidence"]["declared"] else None,
                tuple(p["provenance"]),
            )
            for p in wire["produces"]
        )
        consumes = tuple(
            ConsumedPort(
                c["name"],
                quantity(c["quantity"]),
                c["semantic_identity"],
                SupportDomain._from_wire(c["support"]),
                tuple(c["operations"]),
                meaning(c["uncertainty"]),
                c["calibration"],
                tuple(c["measurement_models"]),
                c["dependence"],
                tuple(c["provenance"]),
            )
            for c in wire["consumes"]
        )
        return cls(wire["component"], produces, consumes)


def _dumps(value: object) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"))


def contract_identity(contract: InterfaceContract) -> str:
    """The canonical identity of a declaration; refuses a malformed declaration."""
    identity, refusal = _native.interface_contract_identity(_dumps(contract._wire()))
    _raise(refusal)
    assert identity is not None
    return identity


# --------------------------------------------------------------------------- decisions


@dataclass(frozen=True, slots=True)
class Connection:
    """An accepted connection; it carries the producer's label and evidence unchanged."""

    producer: str
    produced_port: str
    consumer: str
    consumed_port: str
    operations: tuple[str, ...]
    uncertainty: str | None
    uncertainty_paradigm: str | None
    calibration: Calibration
    evidence: tuple[str, ...] | None
    identity: str


def _connection(wire: Mapping[str, Any]) -> Connection:
    uncertainty = wire["uncertainty"]
    return Connection(
        producer=wire["producer"],
        produced_port=wire["produced_port"],
        consumer=wire["consumer"],
        consumed_port=wire["consumed_port"],
        operations=tuple(wire["operations"]),
        uncertainty=None if uncertainty is None else uncertainty["meaning"],
        uncertainty_paradigm=None if uncertainty is None else uncertainty["paradigm"],
        calibration=Calibration(
            wire["calibration"]["status"],
            wire["calibration"]["record_id"],
            wire["calibration"]["input"],
        ),
        evidence=tuple(wire["evidence"]["sources"]) if wire["evidence"]["declared"] else None,
        identity=wire["identity"],
    )


def check_connection(
    producer: InterfaceContract,
    produced_port: str,
    consumer: InterfaceContract,
    consumed_port: str,
) -> Connection:
    """Decide, from the declarations alone, whether ``produced_port`` may feed ``consumed_port``.

    Raises:
        InterfaceContractRefusal: any defect; ``refusals`` lists every one.
    """
    wire, refusal = _native.interface_contract_check_connection(
        _dumps(producer._wire()), produced_port, _dumps(consumer._wire()), consumed_port
    )
    _raise(refusal)
    assert wire is not None
    return _connection(json.loads(wire))


@dataclass(frozen=True, slots=True)
class ComposedInterfaces:
    """A composition every connection of which is licensed and which has no cycle."""

    order: tuple[str, ...]
    connections: tuple[Connection, ...]
    identity: str


def check_composition(
    components: Sequence[InterfaceContract],
    connections: Iterable[tuple[str, str, str, str]],
) -> ComposedInterfaces:
    """Check components and the connections ``(producer, port, consumer, port)`` between them.

    Raises:
        InterfaceContractRefusal: a feedback cycle (``reason_code`` ``interface_contract_cyclic_composition``,
            the path in ``offending``), a connection defect, inherited calibration that is not
            held, or inputs required disjoint that share evidence.
    """
    wire, refusal = _native.interface_contract_check_composition(
        _dumps([c._wire() for c in components]),
        _dumps([list(c) for c in connections]),
    )
    _raise(refusal)
    assert wire is not None
    data = json.loads(wire)
    return ComposedInterfaces(
        order=tuple(data["order"]),
        connections=tuple(_connection(c) for c in data["connections"]),
        identity=data["identity"],
    )


# --------------------------------------------------------------------------- artifact


@dataclass(frozen=True, slots=True)
class InterfaceContractExport:
    """An exported contract: the bytes and the identities a consumer retains."""

    artifact: bytes
    identity: str
    contract_identity: str


def export_interface_contract(
    contract: InterfaceContract, artifact_id: str
) -> InterfaceContractExport:
    """Seal a contract as ``interface_contract_v1``."""
    done, refusal = _native.interface_contract_export(
        _dumps(contract._wire()), _name(artifact_id, "artifact_id")
    )
    _raise(refusal)
    assert done is not None
    artifact, identity, contract_id = done
    return InterfaceContractExport(bytes(artifact), identity, contract_id)


@dataclass(frozen=True, slots=True)
class ConsumedInterfaceContract:
    """A contract consumed from an artifact, re-validated, ports in name order."""

    contract: InterfaceContract
    contract_identity: str
    identity: str


def consume_interface_contract_artifact(
    artifact: bytes, expected_identity: str | None = None
) -> ConsumedInterfaceContract:
    """Rebuild and re-validate a contract and recompute both identities.

    ``expected_identity`` is an artifact identity retained independently of the bytes.
    """
    wire, refusal = _native.interface_contract_consume(bytes(artifact), expected_identity)
    _raise(refusal)
    assert wire is not None
    data = json.loads(wire)
    return ConsumedInterfaceContract(
        InterfaceContract._from_wire(data["contract"]),
        data["contract_identity"],
        data["identity"],
    )


def inspect_interface_contract_artifact(artifact: bytes) -> dict[str, Any]:
    """The stored metadata of an artifact, without replaying or checking identities."""
    wire, refusal = _native.interface_contract_inspect(bytes(artifact))
    _raise(refusal)
    assert wire is not None
    meta: dict[str, Any] = json.loads(wire)
    return meta
