"""Fixed-population temporal counterfactual and the closed transported route.

The graph is one fixed, fully observed, Markovian two-slice DAG over the nodes
``covariate_0 -> action_0 -> covariate_1 -> action_1 -> outcome`` (a covariate and an action
per slice, then the final outcome) with no latent confounding. Each non-action node has an
additive-noise linear mechanism ``V = intercept + sum(coef * parent) + U_V``. Each unit's
exogenous history ``(U_covariate_0, U_covariate_1, U_outcome)`` is recovered exactly from its
factual trajectory (abduction), BOTH named two-step action histories are replayed against that
same history (action) and the unit's final outcome is computed from it (prediction). The claim
is a point: per-unit final outcomes in both worlds and their sample mean contrast. Fresh noise
per world would answer a different (interventional) question and is never drawn.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NoReturn

from ._artifact_protocol import (
    ExportedArtifact,
    artifact_consumer,
    artifact_export,
    artifact_from_receipt,
    container_bytes,
)
from ._columns import coerce_fields
from ._native import (
    consume_general_temporal_counterfactual_artifact as _consume_general,
)
from ._native import (
    consume_temporal_counterfactual_artifact as _consume_temporal,
)
from ._native import (
    evaluate_general_temporal_counterfactual as _evaluate_general,
)
from ._native import (
    evaluate_temporal_counterfactual as _evaluate_temporal,
)
from ._native import (
    transported_path_specific_refusal as _transported_refusal,
)
from ._refusal import raise_refusal
from ._scientific import ScientificResult
from ._validate import non_finite_refusal, real
from .errors import CausalTypeError, CausalUnsupportedError, CausalValueError
from .temporal import TemporalRefusal

if TYPE_CHECKING:
    from ._artifact_protocol import ExportedArtifact

__all__ = [
    "AbductionReceipt",
    "ActionHistory",
    "NodeMechanism",
    "TemporalCounterfactualEffect",
    "TemporalCounterfactualIdentity",
    "TemporalMechanisms",
    "TemporalWorldsEffect",
    "TemporalWorldsIdentity",
    "TemporalWorldsReceipt",
    "TransportedPrerequisites",
    "UnitCounterfactual",
    "UnitHistory",
    "UnitWorldOutcomes",
    "WorldActions",
    "WorldHistory",
    "consume_temporal_counterfactual_artifact",
    "consume_temporal_worlds_artifact",
    "temporal_fixed_population",
    "temporal_worlds",
    "transported_path_specific",
]

TEMPORAL_NODES = ("covariate_0", "action_0", "covariate_1", "action_1", "outcome")
_U32_MAX = 2**32 - 1


def _name(value: object, what: str) -> str:
    if not isinstance(value, str) or not value:
        raise CausalTypeError(f"{what} must be a non-empty string")
    return value


def _times(value: object, what: str) -> tuple[int, int]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, str)
        or len(value) != 2
        or not all(isinstance(t, int) and not isinstance(t, bool) for t in value)
    ):
        raise CausalTypeError(f"{what} must be a pair of integer observation times")
    if not all(0 <= t <= _U32_MAX for t in value):
        raise CausalValueError(f"{what} must lie in [0, 2**32)")
    return (int(value[0]), int(value[1]))


def _number(value: object, what: str, *, detail: str, offending: str | None = None) -> float:
    # JSON cannot carry a non-finite number: report it with the Rust refusal's code/detail.
    return real(
        value,
        what,
        finite=True,
        non_finite=non_finite_refusal(
            TemporalRefusal, stage="evaluate", detail=detail, offending=offending
        ),
    )


@dataclass(frozen=True, slots=True)
class UnitHistory:
    """One unit's factual two-slice trajectory ``[L0, A0, L1, A1, Y]``.

    ``history`` is the unit's factual-history id (default ``"<unit>:history"``); ``times`` are
    the observation times of the two slices, which every action history must match.
    """

    unit: str
    covariate_0: float
    action_0: float
    covariate_1: float
    action_1: float
    outcome: float
    history: str | None = None
    times: tuple[int, int] = (0, 1)

    def __post_init__(self) -> None:
        _name(self.unit, "unit")
        if self.history is not None:
            _name(self.history, "history")
        coerce_fields(self, times=_times(self.times, "times"))
        for node in TEMPORAL_NODES:
            coerce_fields(
                self,
                **{
                    node: _number(
                        getattr(self, node),
                        node,
                        detail="temporal_counterfactual.non_finite_history",
                        offending=self.unit,
                    )
                },
            )

    def _wire(self) -> dict[str, Any]:
        return {
            "unit": self.unit,
            "history": self.history if self.history is not None else f"{self.unit}:history",
            "times": list(self.times),
            "values": [getattr(self, node) for node in TEMPORAL_NODES],
        }


@dataclass(frozen=True, slots=True)
class ActionHistory:
    """A named two-step action history ``(A0, A1)`` and the units it is requested for.

    ``times`` must equal the factual observation times. ``units=None`` requests every unit
    of the factual histories; naming a subset (or a unit with no factual history) is refused
    as unpaired (``temporal_counterfactual.unpaired_histories``) with the offending unit as
    the witness.
    """

    name: str
    actions: tuple[float, float]
    times: tuple[int, int] = (0, 1)
    units: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        _name(self.name, "name")
        if not isinstance(self.actions, Sequence) or len(self.actions) != 2:
            raise CausalTypeError("actions must be a pair (action_0, action_1)")
        coerce_fields(
            self,
            actions=tuple(
                _number(
                    a,
                    "action",
                    detail="temporal_counterfactual.non_finite_history",
                    offending=self.name,
                )
                for a in self.actions
            ),
        )
        coerce_fields(self, times=_times(self.times, "times"))
        if self.units is not None:
            coerce_fields(self, units=tuple(_name(u, "unit") for u in self.units))

    def _wire(self, units: Sequence[str]) -> dict[str, Any]:
        return {
            "name": self.name,
            "times": list(self.times),
            "actions": list(self.actions),
            "units": list(self.units if self.units is not None else units),
        }


@dataclass(frozen=True, slots=True)
class NodeMechanism:
    """The additive-noise linear mechanism of one non-action node.

    ``node`` is ``"covariate_0"``, ``"covariate_1"`` or ``"outcome"``; ``coefficients`` maps
    each parent node name to its coefficient. ``noise_halfwidth`` optionally declares a
    bounded noise support: a factual history whose abduced residual falls outside it is a
    refuting witness (``temporal_counterfactual.refuting_history``).
    """

    node: str
    intercept: float
    coefficients: Mapping[str, float]
    noise_halfwidth: float | None = None

    def __post_init__(self) -> None:
        _name(self.node, "node")
        coerce_fields(
            self,
            intercept=_number(
                self.intercept, "intercept", detail="temporal_counterfactual.fit_mismatch"
            ),
        )
        coerce_fields(
            self,
            coefficients={
                _name(parent, "parent"): _number(
                    coef,
                    "coefficient",
                    detail="temporal_counterfactual.fit_mismatch",
                    offending=self.node,
                )
                for parent, coef in dict(self.coefficients).items()
            },
        )
        if self.noise_halfwidth is not None:
            coerce_fields(
                self,
                noise_halfwidth=_number(
                    self.noise_halfwidth,
                    "noise_halfwidth",
                    detail="temporal_counterfactual.fit_mismatch",
                    offending=self.node,
                ),
            )

    def _wire(self) -> dict[str, Any]:
        return {
            "node": self.node,
            "intercept": self.intercept,
            "parent_coefficients": [[p, c] for p, c in self.coefficients.items()],
            "noise_halfwidth": self.noise_halfwidth,
        }


@dataclass(frozen=True, slots=True)
class TemporalMechanisms:
    """A fitted (or supplied) mechanism per non-action node, under one stable ``fit_id``.

    The temporal graph is derived from the mechanisms' parents, so graph and fit agree by
    construction; ``fit_id`` is part of the artifact identity, so a refit under the same id is
    still a different fit (its coefficients are bound too).
    """

    fit_id: str
    mechanisms: tuple[NodeMechanism, ...]

    def __post_init__(self) -> None:
        _name(self.fit_id, "fit_id")
        coerce_fields(self, mechanisms=tuple(self.mechanisms))
        if not all(isinstance(m, NodeMechanism) for m in self.mechanisms):
            raise CausalTypeError("mechanisms must be NodeMechanism objects")

    def _edges(self) -> list[list[str]]:
        return sorted([parent, m.node] for m in self.mechanisms for parent in m.coefficients)

    def _wire(self) -> dict[str, Any]:
        return {"fit_id": self.fit_id, "mechanisms": [m._wire() for m in self.mechanisms]}


@dataclass(frozen=True, slots=True)
class UnitCounterfactual:
    """One unit's final outcome, factual and under both named action histories."""

    unit: str
    history: str
    factual_outcome: float
    plus_outcome: float
    minus_outcome: float

    @property
    def contrast(self) -> float:
        """``plus_outcome - minus_outcome`` for this unit's own abduced history."""
        return self.plus_outcome - self.minus_outcome


@dataclass(frozen=True, slots=True)
class AbductionReceipt:
    """Receipt that both worlds were replayed against one shared abduced history per unit.

    The digests bind the snapshot, graph, mechanism fit, factual histories and each world's
    action history; ``unit_draws`` holds one ``(unit, history, digest)`` of the abduced
    exogenous draw per unit. Each is a 64-hex BLAKE3 digest under its own domain tag.
    """

    snapshot: str
    graph_digest: str
    fit_digest: str
    factual_digest: str
    plus_digest: str
    minus_digest: str
    unit_draws: tuple[tuple[str, str, str], ...]
    exogenous_digest: str
    n_units: int
    n_worlds: int
    horizon: int
    shared_by_both_worlds: bool
    digest: str


@dataclass(frozen=True, slots=True)
class TemporalCounterfactualIdentity:
    """Identity digests a consumer retains independently of the artifact bytes.

    Pass it to :func:`consume_temporal_counterfactual_artifact` as ``expected_identity=`` to refuse a
    *resealed* change of action time, unit history, snapshot, mechanism fit or graph.
    """

    snapshot: str
    graph_digest: str
    fit_digest: str
    factual_digest: str
    plus_digest: str
    minus_digest: str
    exogenous_digest: str
    receipt_digest: str
    spec_id: str

    def _wire(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in self.__slots__}


@dataclass(frozen=True, slots=True)
class TemporalCounterfactualEffect(ScientificResult):
    """Per-unit counterfactual outcomes, their mean contrast, the receipt and the artifact.

    ``contrast`` is the sample mean of ``plus - minus`` over units; each unit's contrast
    uses that unit's own abduced history in both worlds. The claim is a point (no interval,
    no posterior, no transport): ``inference_claim == "point_only"``.
    """

    units: tuple[UnitCounterfactual, ...]
    mean_plus: float
    mean_minus: float
    contrast: float
    plus_name: str
    minus_name: str
    mechanism_class: str
    inference_claim: str
    snapshot: str
    receipt: AbductionReceipt
    identity: TemporalCounterfactualIdentity
    _artifact_bytes: bytes

    @artifact_export
    def export(self) -> ExportedArtifact[TemporalCounterfactualEffect]:
        """The checksummed ``temporal_counterfactual_v1`` artifact."""
        return artifact_from_receipt(self, self._artifact_bytes)


def _temporal_effect(report_json: str, artifact: bytes) -> TemporalCounterfactualEffect:
    artifact = container_bytes(artifact)
    report = json.loads(report_json)
    receipt = report["receipt"]
    return TemporalCounterfactualEffect(
        units=tuple(
            UnitCounterfactual(
                unit=u["unit"],
                history=u["history"],
                factual_outcome=float(u["factual_outcome"]),
                plus_outcome=float(u["plus_outcome"]),
                minus_outcome=float(u["minus_outcome"]),
            )
            for u in report["units"]
        ),
        mean_plus=float(report["mean_plus"]),
        mean_minus=float(report["mean_minus"]),
        contrast=float(report["mean_contrast"]),
        plus_name=report["plus_name"],
        minus_name=report["minus_name"],
        mechanism_class=report["mechanism_class"],
        inference_claim=report["inference_claim"],
        snapshot=report["snapshot"],
        receipt=AbductionReceipt(
            snapshot=receipt["snapshot"],
            graph_digest=receipt["graph_digest"],
            fit_digest=receipt["fit_digest"],
            factual_digest=receipt["factual_digest"],
            plus_digest=receipt["plus_digest"],
            minus_digest=receipt["minus_digest"],
            unit_draws=tuple((d["unit"], d["history"], d["digest"]) for d in receipt["unit_draws"]),
            exogenous_digest=receipt["exogenous_digest"],
            n_units=int(receipt["n_units"]),
            n_worlds=int(receipt["n_worlds"]),
            horizon=int(receipt["horizon"]),
            shared_by_both_worlds=bool(receipt["shared_by_both_worlds"]),
            digest=receipt["digest"],
        ),
        identity=TemporalCounterfactualIdentity(**report["identity"]),
        _artifact_bytes=artifact,
    )


def temporal_fixed_population(
    histories: Sequence[UnitHistory],
    *,
    plus: ActionHistory,
    minus: ActionHistory,
    mechanisms: TemporalMechanisms,
    snapshot: str,
    horizon: int = 2,
    latent_confounding: bool = False,
) -> TemporalCounterfactualEffect:
    """What would each unit's final outcome have been under two named action histories?

    ``histories`` are the units' factual two-slice trajectories; ``plus`` and ``minus`` are the
    two named action histories (the contrast is ``plus - minus``) and ``mechanisms`` the fitted
    linear-Gaussian additive-noise mechanisms of ``covariate_0``, ``covariate_1`` and
    ``outcome``. ``snapshot`` identifies the data the histories were read from. Abduction,
    action and prediction happen in one operation: each unit's exogenous history is abduced
    once and both worlds replay against it.

    Raises :class:`~antecedent.temporal.TemporalRefusal` (a
    :class:`~antecedent.errors.CausalUnsupportedError`) with ``route_not_supported`` and
    ``temporal_counterfactual.unpaired_histories`` (``witness`` names the unit and world),
    ``.time_misaligned``, ``.shared_history_missing``, ``.latent_confounding`` or
    ``.refuting_history`` (``witness`` retains unit, history, node, residual and bound), and
    with ``invalid_argument`` for ``.horizon_exceeded``, ``.too_many_units``,
    ``.invalid_graph``, ``.fit_mismatch``, ``.history_name_invalid`` or
    ``.non_finite_history``.
    """
    if isinstance(histories, str | bytes) or not isinstance(histories, Sequence):
        raise CausalTypeError("histories must be a sequence of UnitHistory")
    if not all(isinstance(h, UnitHistory) for h in histories):
        raise CausalTypeError("histories must be UnitHistory objects")
    if not isinstance(plus, ActionHistory) or not isinstance(minus, ActionHistory):
        raise CausalTypeError("plus and minus must be ActionHistory objects")
    if not isinstance(mechanisms, TemporalMechanisms):
        raise CausalTypeError("mechanisms must be TemporalMechanisms")
    _name(snapshot, "snapshot")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or not 0 <= horizon <= _U32_MAX:
        raise CausalValueError("horizon must be a non-negative integer")
    units = [h.unit for h in histories]
    request = {
        "graph": {
            "horizon": horizon,
            "edges": mechanisms._edges(),
            "latent_confounding": bool(latent_confounding),
        },
        "fit": mechanisms._wire(),
        "snapshot": snapshot,
        "factual": [h._wire() for h in histories],
        "plus": plus._wire(units),
        "minus": minus._wire(units),
    }
    report, artifact, refusal = _evaluate_temporal(
        json.dumps(request, allow_nan=False), "temporal_counterfactual"
    )
    raise_refusal(refusal, TemporalRefusal)
    if report is None or artifact is None:  # pragma: no cover - the native contract
        raise CausalValueError(
            "the native temporal counterfactual returned neither a result nor a refusal"
        )
    return _temporal_effect(report, bytes(artifact))


@artifact_consumer(
    proof="temporal_counterfactual.consume_temporal_counterfactual_artifact: independent semantic consumer checks declarations, identities and stored result; recomputes the scientific result",
    replayable=True,
)
def consume_temporal_counterfactual_artifact(
    artifact: bytes,
    *,
    expected_identity: TemporalCounterfactualIdentity | Mapping[str, str] | None = None,
) -> TemporalCounterfactualEffect:
    """Replay both worlds of an exported artifact and accept only an identical one.

    Every unit's exogenous history is abduced again from the stored factual history and both
    action histories are replayed with the same evaluator; every stored outcome, both means
    and the whole shared-abduction receipt must reproduce bit for bit. With ``expected_identity`` (the
    :attr:`TemporalCounterfactualEffect.identity` retained out-of-band) a changed action time,
    unit history, snapshot, mechanism fit or graph is refused even when the artifact was
    resealed consistently (:class:`~antecedent.temporal.TemporalRefusal`,
    ``route_not_supported``, ``temporal_counterfactual.artifact_changed``). Corruption and
    unknown major versions raise :class:`~antecedent.errors.CausalSerializationError`.
    """
    artifact = container_bytes(artifact)
    if not isinstance(artifact, bytes | bytearray | memoryview):
        raise CausalTypeError("artifact must be bytes")
    if expected_identity is None:
        expected_json = None
    elif isinstance(expected_identity, TemporalCounterfactualIdentity):
        expected_json = json.dumps(expected_identity._wire())
    elif isinstance(expected_identity, Mapping):
        expected_json = json.dumps(dict(expected_identity))
    else:
        raise CausalTypeError(
            "expected_identity must be a TemporalCounterfactualIdentity or a mapping"
        )
    data = bytes(artifact)
    report, refusal = _consume_temporal(data, expected_json)
    raise_refusal(refusal, TemporalRefusal)
    if report is None:  # pragma: no cover - the native contract
        raise CausalValueError("the native consumer returned neither a result nor a refusal")
    return _temporal_effect(report, data)


# --- Any horizon, any number of named action histories ---------------------------------------


def _real_sequence(value: object, what: str, *, detail: str, offending: str) -> tuple[float, ...]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise CausalTypeError(f"{what} must be a sequence of numbers")
    return tuple(_number(v, what, detail=detail, offending=offending) for v in value)


def _time_sequence(value: object, what: str) -> tuple[int, ...]:
    if (
        isinstance(value, str | bytes)
        or not isinstance(value, Sequence)
        or not all(isinstance(t, int) and not isinstance(t, bool) for t in value)
    ):
        raise CausalTypeError(f"{what} must be a sequence of integer observation times")
    if not all(0 <= t <= _U32_MAX for t in value):
        raise CausalValueError(f"{what} must lie in [0, 2**32)")
    return tuple(int(t) for t in value)


def _node_index(name: str, horizon: int) -> int:
    """Index of ``covariate_<t>`` (``2t``), ``action_<t>`` (``2t + 1``) or ``outcome`` (``2h``)."""
    if name == "outcome":
        return 2 * horizon
    kind, _, tail = name.rpartition("_")
    if kind in ("covariate", "action") and tail.isdecimal() and int(tail) < horizon:
        return 2 * int(tail) + (1 if kind == "action" else 0)
    raise CausalValueError(
        f"node {name!r} is not covariate_<t>, action_<t> (t < {horizon}) or outcome",
        reason_code="invalid_argument",
    )


@dataclass(frozen=True, slots=True)
class WorldHistory:
    """One unit's factual trajectory over ``horizon`` slices.

    ``values`` lists the ``2 * horizon + 1`` node values in node order
    ``covariate_0, action_0, covariate_1, action_1, ..., outcome``. ``times`` are the
    observation times of the slices (default ``0 .. horizon - 1``) and ``history`` the factual
    history id (default ``"<unit>:history"``).
    """

    unit: str
    values: tuple[float, ...]
    history: str | None = None
    times: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        _name(self.unit, "unit")
        if self.history is not None:
            _name(self.history, "history")
        coerce_fields(
            self,
            values=_real_sequence(
                self.values,
                "values",
                detail="temporal_counterfactual.non_finite_history",
                offending=self.unit,
            ),
        )
        if self.times is not None:
            coerce_fields(self, times=_time_sequence(self.times, "times"))

    def _wire(self, horizon: int) -> dict[str, Any]:
        return {
            "unit": self.unit,
            "history": self.history if self.history is not None else f"{self.unit}:history",
            "times": list(self.times) if self.times is not None else list(range(horizon)),
            "values": list(self.values),
        }


@dataclass(frozen=True, slots=True)
class WorldActions:
    """A named action history ``(A_0, ..., A_{h-1})`` and the units it is requested for.

    ``times`` must equal the factual observation times (default ``0 .. horizon - 1``);
    ``units=None`` requests every unit of the factual histories.
    """

    name: str
    actions: tuple[float, ...]
    times: tuple[int, ...] | None = None
    units: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        _name(self.name, "name")
        coerce_fields(
            self,
            actions=_real_sequence(
                self.actions,
                "actions",
                detail="temporal_counterfactual.non_finite_history",
                offending=self.name,
            ),
        )
        if self.times is not None:
            coerce_fields(self, times=_time_sequence(self.times, "times"))
        if self.units is not None:
            coerce_fields(self, units=tuple(_name(u, "unit") for u in self.units))

    def _wire(self, horizon: int, units: Sequence[str]) -> dict[str, Any]:
        return {
            "name": self.name,
            "times": list(self.times) if self.times is not None else list(range(horizon)),
            "actions": list(self.actions),
            "units": list(self.units if self.units is not None else units),
        }


@dataclass(frozen=True, slots=True)
class UnitWorldOutcomes:
    """One unit's final outcome, factual and under every named action history."""

    unit: str
    history: str
    factual_outcome: float
    outcomes: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class TemporalWorldsReceipt:
    """Receipt that every world was replayed against one shared abduced history per unit."""

    snapshot: str
    fit_digest: str
    factual_digest: str
    world_digests: tuple[str, ...]
    unit_draws: tuple[tuple[str, str, str], ...]
    exogenous_digest: str
    n_units: int
    n_worlds: int
    horizon: int
    shared_by_all_worlds: bool
    digest: str


@dataclass(frozen=True, slots=True)
class TemporalWorldsIdentity:
    """Identity digests a consumer retains independently of the artifact bytes.

    Pass it to :func:`consume_temporal_worlds_artifact` as ``expected_identity=`` to refuse a
    *resealed* change of action time, unit history, snapshot, mechanism fit or any world.
    """

    snapshot: str
    fit_digest: str
    factual_digest: str
    world_digests: tuple[str, ...]
    exogenous_digest: str
    receipt_digest: str
    spec_id: str

    def _wire(self) -> dict[str, Any]:
        return {
            "snapshot": self.snapshot,
            "fit_digest": self.fit_digest,
            "factual_digest": self.factual_digest,
            "world_digests": list(self.world_digests),
            "exogenous_digest": self.exogenous_digest,
            "receipt_digest": self.receipt_digest,
            "spec_id": self.spec_id,
        }


@dataclass(frozen=True, slots=True)
class TemporalWorldsEffect(ScientificResult):
    """Per-unit counterfactual outcomes under every named action history, with the receipt.

    ``world_means[i]`` is the sample mean of the final outcome under ``world_names[i]``; each
    unit's outcomes in all worlds use that unit's own abduced history. The claim is a point
    (no interval, no posterior, no transport): ``inference_claim == "point_only"``.
    """

    units: tuple[UnitWorldOutcomes, ...]
    world_names: tuple[str, ...]
    world_means: tuple[float, ...]
    horizon: int
    mechanism_class: str
    inference_claim: str
    snapshot: str
    receipt: TemporalWorldsReceipt
    identity: TemporalWorldsIdentity
    _artifact_bytes: bytes

    def mean(self, world: str) -> float:
        """Sample mean final outcome under the named action history."""
        return self.world_means[self._index(world)]

    def contrast(self, plus: str, minus: str) -> float:
        """Sample mean of ``plus - minus`` (each unit's own abduced history in both worlds)."""
        return self.mean(plus) - self.mean(minus)

    def _index(self, world: str) -> int:
        try:
            return self.world_names.index(world)
        except ValueError:
            raise CausalValueError(
                f"unknown world {world!r}; the worlds are {list(self.world_names)}"
            ) from None

    @artifact_export
    def export(self) -> ExportedArtifact[TemporalWorldsEffect]:
        """The checksummed ``temporal_counterfactual_general_v1`` artifact."""
        return artifact_from_receipt(self, self._artifact_bytes)


def _worlds_effect(report_json: str, artifact: bytes) -> TemporalWorldsEffect:
    artifact = container_bytes(artifact)
    report = json.loads(report_json)
    receipt = report["receipt"]
    return TemporalWorldsEffect(
        units=tuple(
            UnitWorldOutcomes(
                unit=u["unit"],
                history=u["history"],
                factual_outcome=float(u["factual_outcome"]),
                outcomes=tuple(float(o) for o in u["outcomes"]),
            )
            for u in report["units"]
        ),
        world_names=tuple(report["world_names"]),
        world_means=tuple(float(m) for m in report["world_means"]),
        horizon=int(report["horizon"]),
        mechanism_class=report["mechanism_class"],
        inference_claim=report["inference_claim"],
        snapshot=report["snapshot"],
        receipt=TemporalWorldsReceipt(
            snapshot=receipt["snapshot"],
            fit_digest=receipt["fit_digest"],
            factual_digest=receipt["factual_digest"],
            world_digests=tuple(receipt["world_digests"]),
            unit_draws=tuple((d["unit"], d["history"], d["digest"]) for d in receipt["unit_draws"]),
            exogenous_digest=receipt["exogenous_digest"],
            n_units=int(receipt["n_units"]),
            n_worlds=int(receipt["n_worlds"]),
            horizon=int(receipt["horizon"]),
            shared_by_all_worlds=bool(receipt["shared_by_all_worlds"]),
            digest=receipt["digest"],
        ),
        identity=TemporalWorldsIdentity(
            **{
                **report["identity"],
                "world_digests": tuple(report["identity"]["world_digests"]),
            }
        ),
        _artifact_bytes=artifact,
    )


def temporal_worlds(
    histories: Sequence[WorldHistory],
    *,
    worlds: Sequence[WorldActions],
    mechanisms: TemporalMechanisms,
    snapshot: str,
    horizon: int,
    latent_confounding: bool = False,
) -> TemporalWorldsEffect:
    """What would each unit's final outcome have been under every named action history?

    The general counterpart of :func:`temporal_fixed_population`: any ``horizon`` of slices
    (a covariate and an action per slice, then the outcome) and any number of ``worlds``. The
    mechanisms are named ``covariate_<t>`` (``t < horizon``) and ``outcome``, with parents named
    ``covariate_<t>``, ``action_<t>`` of strictly earlier nodes. Each unit's exogenous history
    is abduced once from its factual trajectory and every world replays against it. Cost is
    linear in ``units x worlds x nodes`` and bounded by a work budget, not by a count of slices
    or worlds.

    Raises :class:`~antecedent.temporal.TemporalRefusal` with ``route_not_supported`` and
    ``temporal_counterfactual.unpaired_histories`` (``witness`` names the unit and world),
    ``.time_misaligned``, ``.shared_history_missing``, ``.latent_confounding`` or
    ``.refuting_history`` (``witness`` retains unit, history, residual and bound), and with
    ``invalid_argument`` for ``.too_many_units``, ``.invalid_graph`` (zero horizon),
    ``.fit_mismatch``, ``.history_name_invalid`` or ``.non_finite_history`` (including a
    trajectory or action history of the wrong length).
    """
    if isinstance(histories, str | bytes) or not isinstance(histories, Sequence):
        raise CausalTypeError("histories must be a sequence of WorldHistory")
    if not all(isinstance(h, WorldHistory) for h in histories):
        raise CausalTypeError("histories must be WorldHistory objects")
    if isinstance(worlds, str | bytes) or not isinstance(worlds, Sequence):
        raise CausalTypeError("worlds must be a sequence of WorldActions")
    if not all(isinstance(w, WorldActions) for w in worlds):
        raise CausalTypeError("worlds must be WorldActions objects")
    if not isinstance(mechanisms, TemporalMechanisms):
        raise CausalTypeError("mechanisms must be TemporalMechanisms")
    _name(snapshot, "snapshot")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or not 0 <= horizon <= _U32_MAX:
        raise CausalValueError("horizon must be a non-negative integer")
    units = [h.unit for h in histories]
    fit = {
        "fit_id": mechanisms.fit_id,
        "mechanisms": [
            {
                "node": _node_index(m.node, horizon),
                "intercept": m.intercept,
                "parent_coefficients": [
                    [_node_index(parent, horizon), coef] for parent, coef in m.coefficients.items()
                ],
                "noise_halfwidth": m.noise_halfwidth,
            }
            for m in mechanisms.mechanisms
        ],
    }
    request = {
        "horizon": horizon,
        "latent_confounding": bool(latent_confounding),
        "fit": fit,
        "snapshot": snapshot,
        "factual": [h._wire(horizon) for h in histories],
        "worlds": [w._wire(horizon, units) for w in worlds],
    }
    report, artifact, refusal = _evaluate_general(
        json.dumps(request, allow_nan=False), "temporal_counterfactual_general"
    )
    raise_refusal(refusal, TemporalRefusal)
    if report is None or artifact is None:  # pragma: no cover - the native contract
        raise CausalValueError(
            "the native general temporal counterfactual returned neither a result nor a refusal"
        )
    return _worlds_effect(report, bytes(artifact))


@artifact_consumer(
    proof="temporal_counterfactual.consume_temporal_worlds_artifact: independent semantic consumer checks declarations, identities and stored result; recomputes the scientific result",
    replayable=True,
)
def consume_temporal_worlds_artifact(
    artifact: bytes,
    *,
    expected_identity: TemporalWorldsIdentity | Mapping[str, Any] | None = None,
) -> TemporalWorldsEffect:
    """Replay every world of an exported artifact and accept only an identical one.

    Every unit's exogenous history is abduced again from the stored factual history and every
    action history is replayed; every stored outcome, every world mean and the whole receipt
    must reproduce bit for bit. With ``expected_identity`` (the
    :attr:`TemporalWorldsEffect.identity` retained out-of-band) a changed action time, unit
    history, snapshot, mechanism fit or world is refused even when the artifact was resealed
    consistently (:class:`~antecedent.temporal.TemporalRefusal`, ``route_not_supported``,
    ``temporal_counterfactual.artifact_changed``). Corruption and unknown major versions raise
    :class:`~antecedent.errors.CausalSerializationError`.
    """
    artifact = container_bytes(artifact)
    if not isinstance(artifact, bytes | bytearray | memoryview):
        raise CausalTypeError("artifact must be bytes")
    if expected_identity is None:
        expected_json = None
    elif isinstance(expected_identity, TemporalWorldsIdentity):
        expected_json = json.dumps(expected_identity._wire())
    elif isinstance(expected_identity, Mapping):
        expected_json = json.dumps(dict(expected_identity))
    else:
        raise CausalTypeError("expected_identity must be a TemporalWorldsIdentity or a mapping")
    data = bytes(artifact)
    report, refusal = _consume_general(data, expected_json)
    raise_refusal(refusal, TemporalRefusal)
    if report is None:  # pragma: no cover - the native contract
        raise CausalValueError("the native consumer returned neither a result nor a refusal")
    return _worlds_effect(report, data)


@dataclass(frozen=True, slots=True)
class TransportedPrerequisites:
    """Which prerequisites of a transported counterfactual have passed.

    All default to ``False``: no composed license exists. Declaring one ``True`` records that
    the gate passed; it never opens the route, because the joint theorem has not passed.
    """

    transport_license: bool = False
    fixed_population_license: bool = False
    cross_population_assumptions: bool = False


def _factor(item: str | tuple[str, str]) -> tuple[str, str]:
    if isinstance(item, str):
        role, _, regime = item.partition(":")
    elif isinstance(item, tuple) and len(item) == 2:
        role, regime = item
    else:
        raise CausalTypeError("a regime factor is 'source:<regime>' / 'target:<regime>' or a pair")
    if role not in ("source", "target") or not isinstance(regime, str) or not regime:
        raise CausalValueError(
            "transported_counterfactual.invalid_factor: a factor names its population "
            "('source' or 'target') and a regime",
            reason_code="invalid_argument",
        )
    return (role, regime)


def transported_path_specific(
    *,
    required_factors: Iterable[str | tuple[str, str]] = (),
    supplied_factors: Mapping[str | tuple[str, str], str] | None = None,
    prerequisites: TransportedPrerequisites | None = None,
) -> NoReturn:
    """The closed transported path-specific counterfactual route: always a typed refusal.

    Transporting a counterfactual needs both an identified target transport functional and a
    fixed-population cross-world theorem for the same quantity; neither a transported mean
    nor a source counterfactual alone licenses their composition, and the joint theorem has
    not passed. Nothing is evaluated whatever is supplied.

    The refusal is :class:`~antecedent.temporal.TemporalRefusal` with
    ``reason_code="cell_not_licensed"`` and detail ``transported_counterfactual.route_frozen``
    (``offending`` is the first missing gate) while a prerequisite gate is missing, and also
    when every gate passes and every regime factor is present. When every gate passes but a
    required factor (``"source:<regime>"`` / ``"target:<regime>"``) is absent from
    ``supplied_factors`` (factor -> evidence id) it is ``reason_code="transport_missing_evidence"``
    with ``transported_counterfactual.factor_missing``. ``missing_gates`` and
    ``missing_factors`` are retained either way.

    The refusal's ``remedy`` names the successor route for the scope in which an answer
    exists: a finite discrete, static, acyclic semi-Markovian counterfactual transport
    question is answered by ``antecedent.transport.advanced.identify_counterfactual``. This
    function never delegates to it and never answers; the temporal, continuous and
    non-finite scope stays refused here.
    """
    declared = prerequisites if prerequisites is not None else TransportedPrerequisites()
    if not isinstance(declared, TransportedPrerequisites):
        raise CausalTypeError("prerequisites must be a TransportedPrerequisites")
    required = [_factor(f) for f in required_factors]
    supplied = [(*_factor(k), str(v)) for k, v in (supplied_factors or {}).items()]
    raise_refusal(
        _transported_refusal(
            declared.transport_license,
            declared.fixed_population_license,
            declared.cross_population_assumptions,
            required,
            supplied,
        ),
        TemporalRefusal,
    )
    raise CausalUnsupportedError(  # pragma: no cover - the native route always refuses
        "transported_counterfactual.route_frozen: the transported route is closed",
        reason_code="cell_not_licensed",
    )
