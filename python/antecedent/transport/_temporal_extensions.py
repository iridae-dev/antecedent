"""Temporal extensions of the finite two-step sequence: uncertain initial state, new-period refresh.

The sequence ``do(A_1 = a_1, A_2 = a_2)`` is read from a panel of repeated units, each
with complete two-step histories ``(time_id, s0, a1, l2, a2, y)``. Three questions are
answered here, and each has its own typed result so none can be relabeled as another:

* :func:`temporal_initial_state` marginalizes the sequence's response over a finite
  **target** initial-state law, ``sum_s0 P_target(s0) R(s0)``. A source-population law
  or a single state cannot answer it (``initial_state.target_law_missing``); holding the
  state fixed is a different estimand and is reported under its own label
  (``fixed_initial_state``) on :attr:`TemporalInitialStateResult.fixed`. Point only.
* :func:`temporal_new_period_refresh` re-evaluates (never copies) a held result on a
  replacement panel of a new observation period when the graph, horizon, lag alignment,
  intervention history, selection targets, regimes and proof are unchanged; otherwise it
  refuses with a typed ``temporal_refresh.*`` invalidation. A stale interval never
  survives a refresh: the refreshed result has no interval and the receipt records
  whether one was invalidated. Point only.
* :func:`temporal_dependent_interval` is a calibrated claim whose calibration is measured
  only at the release cut, so its public route is closed: after validating its arguments
  it refuses with ``cell_not_licensed`` (``temporal_interval.route_frozen``).

Rust owns every identity, check, replay and refusal rule; this module builds declarations
and raises each refusal as :class:`TemporalExtensionRefusal`, a
:class:`~antecedent.errors.CausalUnsupportedError` with its registered ``reason_code``
and the namespaced ``detail``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NoReturn, cast

from .._native import (
    consume_temporal_initial_state_artifact as _consume_initial_state,
)
from .._native import (
    consume_temporal_refresh_artifact as _consume_refresh,
)
from .._native import (
    temporal_dependent_interval_closed as _dependent_interval_closed,
)
from .._native import (
    temporal_initial_state_prepare as _initial_state_prepare,
)
from ..errors import CausalTypeError, CausalUnsupportedError, CausalValueError

MARGINALIZED_LABEL = "marginalized_initial_state"
FIXED_LABEL = "fixed_initial_state"
OBSERVED_LABEL = "observed_initial_state"
INFERENCE_CLAIM = "point_only"
HORIZON = 2

_U32 = 2**32 - 1
_U64 = 2**64 - 1


class TemporalExtensionRefusal(CausalUnsupportedError):
    """A temporal-extension refusal carrying the structured Rust fields.

    ``reason_code`` and ``remedy`` are inherited and the code is registered. ``detail`` is
    the namespaced ``family.slot`` (for example ``initial_state.target_law_missing``,
    ``temporal_refresh.horizon_changed`` or ``temporal_interval.route_frozen``); it is
    empty for a plain argument refusal. ``stage`` names the refusing stage.
    """

    def __init__(self, refusal: Mapping[str, Any]) -> None:
        detail = str(refusal.get("detail") or "")
        message = str(refusal.get("message") or "")
        text = f"{detail}: {message}" if detail else message
        super().__init__(text, reason_code=refusal["code"], remedy=refusal.get("remedy"))
        self.stage: str = str(refusal.get("stage", ""))
        self.detail: str = detail


def _raise(refusal: str | None) -> None:
    if refusal is not None:
        raise TemporalExtensionRefusal(json.loads(refusal))


def _consumed(payload: str | None, refusal: str | None) -> dict[str, Any]:
    """Decode a consumer's JSON payload, raising its typed refusal first."""
    _raise(refusal)
    if payload is None:
        raise CausalValueError(
            "the native consumer returned no payload", reason_code="invalid_argument"
        )
    decoded: dict[str, Any] = json.loads(payload)
    return decoded


def _index(name: str, value: object, *, limit: int = _U32) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > limit:
        raise CausalValueError(
            f"{name} must be a non-negative integer no larger than {limit}",
            reason_code="invalid_argument",
        )
    return value


def _sequence(sequence: Sequence[int]) -> tuple[int, int]:
    steps = tuple(sequence)
    if len(steps) != HORIZON:
        raise CausalValueError(
            "a two-step sequence names exactly two actions", reason_code="invalid_argument"
        )
    return (_index("sequence[0]", steps[0]), _index("sequence[1]", steps[1]))


@dataclass(frozen=True, slots=True)
class SequenceHistory:
    """One complete two-step history of a unit: ``(time_id, s0, a1, l2, a2, y)``."""

    time_id: int
    s0: int
    a1: int
    l2: int
    a2: int
    y: float


@dataclass(frozen=True, slots=True)
class UnitHistories:
    """Every complete history of one repeated unit, strictly increasing in ``time_id``."""

    unit_id: int
    histories: Sequence[SequenceHistory]

    def __post_init__(self) -> None:
        object.__setattr__(self, "histories", tuple(self.histories))


@dataclass(frozen=True, slots=True)
class TemporalUnitPanel:
    """A snapshot of repeated units with their complete histories.

    ``units=None`` means no unit map was supplied; row-level independence is never
    assumed, so every route refuses it (``temporal_interval.unknown_units``).
    """

    snapshot_id: str
    units: Sequence[UnitHistories] | None

    def __post_init__(self) -> None:
        if self.units is not None:
            object.__setattr__(self, "units", tuple(self.units))

    @classmethod
    def from_rows(
        cls,
        snapshot_id: str,
        rows: Iterable[tuple[int, int, int, int, int, int, float]],
    ) -> TemporalUnitPanel:
        """Group ``(unit_id, time_id, s0, a1, l2, a2, y)`` rows into units, first seen first."""
        grouped: dict[int, list[SequenceHistory]] = {}
        for unit_id, time_id, s0, a1, l2, a2, y in rows:
            grouped.setdefault(unit_id, []).append(SequenceHistory(time_id, s0, a1, l2, a2, y))
        return cls(
            snapshot_id,
            [UnitHistories(u, sorted(h, key=lambda x: x.time_id)) for u, h in grouped.items()],
        )

    def _wire(self) -> list[tuple[int, list[tuple[int, int, int, int, int, float]]]] | None:
        if self.units is None:
            return None
        return [
            (
                _index("unit_id", unit.unit_id, limit=_U64),
                [
                    (
                        _index("time_id", h.time_id, limit=_U64),
                        _index("s0", h.s0),
                        _index("a1", h.a1),
                        _index("l2", h.l2),
                        _index("a2", h.a2),
                        float(h.y),
                    )
                    for h in unit.histories
                ],
            )
            for unit in self.units
        ]


@dataclass(frozen=True, slots=True)
class InitialStateLaw:
    """A finite law of the pre-action state, with its own snapshot id.

    ``population`` is ``"target"`` or ``"source"``; only a target law answers a
    target-marginal query. ``states`` maps each state level to its mass; the masses are
    non-negative and sum to one.
    """

    snapshot_id: str
    states: Mapping[int, float] | Sequence[tuple[int, float]]
    population: str = "target"

    def __post_init__(self) -> None:
        pairs = self.states.items() if isinstance(self.states, Mapping) else self.states
        object.__setattr__(
            self,
            "states",
            tuple((_index("state", level), float(mass)) for level, mass in pairs),
        )

    def _wire(self) -> tuple[str, str, list[tuple[int, float]]]:
        return (
            self.population,
            self.snapshot_id,
            list(cast("Sequence[tuple[int, float]]", self.states)),
        )


@dataclass(frozen=True, slots=True)
class TemporalPremises:
    """What the initial state is and where it sits in time.

    ``time_order`` lists the sequence's coordinates in time order, the initial-state
    variable first. ``graph_id`` and ``proof_id`` identify the identification the
    sequence was derived under; a refresh onto changed ones is invalidated.
    """

    initial_state_variable: str
    time_order: Sequence[str]
    source_regime: str
    target_regime: str
    graph_id: str
    proof_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "time_order", tuple(self.time_order))

    def _wire(self) -> tuple[str, list[str], str, str, str, str]:
        return (
            self.initial_state_variable,
            list(self.time_order),
            self.source_regime,
            self.target_regime,
            self.graph_id,
            self.proof_id,
        )


@dataclass(frozen=True, slots=True)
class TemporalWindow:
    """The observation window a prepared result is valid for.

    ``period`` is the half-open ``(start, end)`` range of time ids the panel observes.
    ``lag_alignment`` names each coordinate with the slice it is aligned to (an ordered
    ``Mapping`` or sequence of pairs; the order is part of the identity),
    ``intervention_history`` the ordered action label of each step and
    ``selection_targets`` the time-indexed mechanism differences. ``horizon`` is two; any
    other value (an appended slice) invalidates a refresh. ``graph_id`` and ``proof_id``
    override the premises' ids for a replacement window (a changed graph or proof).
    """

    period: tuple[int, int]
    lag_alignment: Mapping[str, int] | Sequence[tuple[str, int]]
    intervention_history: Sequence[str]
    selection_targets: Sequence[str] = ()
    horizon: int = HORIZON
    graph_id: str | None = None
    proof_id: str | None = None

    def __post_init__(self) -> None:
        lag = self.lag_alignment
        pairs = lag.items() if isinstance(lag, Mapping) else lag
        object.__setattr__(self, "lag_alignment", tuple((str(n), int(s)) for n, s in pairs))
        object.__setattr__(self, "intervention_history", tuple(self.intervention_history))
        object.__setattr__(self, "selection_targets", tuple(self.selection_targets))
        if len(self.period) != 2:
            raise CausalValueError("period is a (start, end) pair", reason_code="invalid_argument")

    def _wire(
        self,
    ) -> tuple[int, list[tuple[str, int]], list[str], list[str], int, int, str | None, str | None]:
        return (
            _index("horizon", self.horizon, limit=_U64),
            [
                (n, _index("slice", s, limit=255))
                for n, s in cast("Sequence[tuple[str, int]]", self.lag_alignment)
            ],
            list(self.intervention_history),
            sorted(set(self.selection_targets)),
            int(self.period[0]),
            int(self.period[1]),
            self.graph_id,
            self.proof_id,
        )


@dataclass(frozen=True, slots=True)
class StateContribution:
    """One state's contribution to the marginalized response."""

    state: int
    mass: float
    response: float


@dataclass(frozen=True, slots=True)
class FixedStateResult:
    """The sequence response with the initial state held fixed (a different estimand)."""

    label: str
    state: int
    value: float
    panel_snapshot_id: str


@dataclass(frozen=True, slots=True)
class RefreshReceipt:
    """Receipt binding the old and new period, snapshot and proof of a refresh."""

    old_identity_digest: str
    new_identity_digest: str
    old_period: tuple[int, int]
    new_period: tuple[int, int]
    old_snapshot_id: str
    new_snapshot_id: str
    proof_id: str
    interval_invalidated: bool
    inference_claim: str
    digest: str

    @classmethod
    def _from(cls, data: Mapping[str, Any]) -> RefreshReceipt:
        return cls(
            old_identity_digest=data["old_identity_digest"],
            new_identity_digest=data["new_identity_digest"],
            old_period=(data["old_period"][0], data["old_period"][1]),
            new_period=(data["new_period"][0], data["new_period"][1]),
            old_snapshot_id=data["old_snapshot_id"],
            new_snapshot_id=data["new_snapshot_id"],
            proof_id=data["proof_id"],
            interval_invalidated=data["interval_invalidated"],
            inference_claim=data["inference_claim"],
            digest=data["digest"],
        )


def _contributions(rows: Iterable[Mapping[str, Any]]) -> tuple[StateContribution, ...]:
    return tuple(StateContribution(r["state"], r["mass"], r["response"]) for r in rows)


@dataclass(frozen=True, slots=True, eq=False)
class TemporalInitialStateResult:
    """The sequence response marginalized over a target initial-state law (point only).

    ``label`` is always ``marginalized_initial_state``; ``fixed`` is the separate
    fixed-state estimand (label ``fixed_initial_state``) when one was requested, and
    neither converts into the other. ``interval`` is always ``None``: this route is point
    only. A result produced by a refresh carries its ``receipt`` and ``previous_value``.
    """

    label: str
    value: float
    sequence: tuple[int, int]
    contributions: tuple[StateContribution, ...]
    state_snapshot_id: str
    state_law_digest: str
    panel_snapshot_id: str
    inference_claim: str
    fixed: FixedStateResult | None
    window: Mapping[str, Any] | None
    receipt: RefreshReceipt | None
    previous_value: float | None
    interval_invalidated: bool
    _native: Any

    interval: None = None

    @classmethod
    def _from(cls, native: Any) -> TemporalInitialStateResult:
        data = json.loads(native.payload())
        fixed = data["fixed"]
        refresh = data["refresh"]
        return cls(
            label=data["label"],
            value=data["value"],
            sequence=(data["sequence"][0], data["sequence"][1]),
            contributions=_contributions(data["contributions"]),
            state_snapshot_id=data["state_snapshot_id"],
            state_law_digest=data["state_law_digest"],
            panel_snapshot_id=data["panel_snapshot_id"],
            inference_claim=data["inference_claim"],
            fixed=None if fixed is None else FixedStateResult(**fixed),
            window=data["window"],
            receipt=None if refresh is None else RefreshReceipt._from(refresh["receipt"]),
            previous_value=None if refresh is None else refresh["previous_value"],
            interval_invalidated=False if refresh is None else refresh["interval_invalidated"],
            _native=native,
        )

    def export(self) -> bytes:
        """The independently consumable initial-state artifact (framed bytes)."""
        return bytes(self._native.export())

    def export_refresh(self) -> bytes:
        """The artifact of the refresh that produced this result (framed bytes)."""
        data, refusal = self._native.export_refresh()
        _raise(refusal)
        return bytes(data)


def temporal_initial_state(
    panel: TemporalUnitPanel,
    *,
    sequence: Sequence[int],
    target_law: InitialStateLaw | int,
    premises: TemporalPremises,
    fixed_state: int | None = None,
    window: TemporalWindow | None = None,
) -> TemporalInitialStateResult:
    """Marginalize the two-step response over a finite target initial-state law.

    The value is ``sum_s0 P_target(s0) R(s0)`` where ``R(s0)`` is the whole sequence's
    response given ``s0`` read from the panel's histories. ``target_law`` must be an
    :class:`InitialStateLaw` of the ``"target"`` population; a source law or a bare state
    refuses (``transport_missing_evidence`` / ``initial_state.target_law_missing``). A
    state with positive target mass and no history support refuses
    (``transport_support_failure`` / ``initial_state.support_gap``). ``fixed_state``
    additionally reports the fixed-state response under its own label. ``window``
    declares the observation window so the result can later be refreshed.
    """
    if not isinstance(panel, TemporalUnitPanel):
        raise CausalTypeError("panel must be a TemporalUnitPanel")
    if not isinstance(premises, TemporalPremises):
        raise CausalTypeError("premises must be TemporalPremises")
    law, point = _law_args(target_law)
    native, refusal = _initial_state_prepare(
        panel.snapshot_id,
        panel._wire(),
        _sequence(sequence),
        law,
        point,
        None if fixed_state is None else _index("fixed_state", fixed_state),
        premises._wire(),
        None if window is None else window._wire(),
    )
    _raise(refusal)
    return TemporalInitialStateResult._from(native)


def _law_args(
    target_law: InitialStateLaw | int,
) -> tuple[tuple[str, str, list[tuple[int, float]]] | None, int | None]:
    if isinstance(target_law, InitialStateLaw):
        return target_law._wire(), None
    if isinstance(target_law, int) and not isinstance(target_law, bool):
        return None, _index("target_law", target_law)
    raise CausalTypeError("target_law must be an InitialStateLaw (or a state level)")


@dataclass(frozen=True, slots=True, eq=False)
class TemporalRefreshResult:
    """A held result re-evaluated on a new observation period (point only).

    ``value`` is a fresh evaluation on the replacement panel, never the old value
    copied; ``previous_value`` is the held one. ``interval`` is always ``None``: a stale
    interval never survives a refresh, and ``interval_invalidated`` records whether one
    existed for the old window. ``result`` is the refreshed prepared result and can be
    refreshed again.
    """

    label: str
    value: float
    previous_value: float
    decision: str
    receipt: RefreshReceipt
    interval_invalidated: bool
    result: TemporalInitialStateResult
    interval: None = None

    def export(self) -> bytes:
        """The independently consumable refresh artifact (framed bytes)."""
        return self.result.export_refresh()


def temporal_new_period_refresh(
    prior: TemporalInitialStateResult | TemporalRefreshResult,
    panel: TemporalUnitPanel,
    *,
    window: TemporalWindow,
    interval_existed: bool = False,
) -> TemporalRefreshResult:
    """Refresh a held result onto a replacement panel of a new observation period.

    ``prior`` must have declared a window. The proof is reusable only when the horizon is
    two and the lag alignment, intervention history, selection targets, regimes, graph and
    proof are unchanged and both the period and the snapshot are new; the value is then
    re-evaluated on ``panel``. Otherwise a typed refusal names why:
    ``temporal_refresh.horizon_changed``, ``lag_alignment_changed``,
    ``intervention_history_changed``, ``premises_changed`` or ``stale_snapshot``
    (``route_not_supported``). ``interval_existed`` says an interval or fit was reported
    for the old window; it is invalidated and the receipt records it.
    """
    held = prior.result if isinstance(prior, TemporalRefreshResult) else prior
    if not isinstance(held, TemporalInitialStateResult):
        raise CausalTypeError("prior must be a TemporalInitialStateResult or TemporalRefreshResult")
    if not isinstance(panel, TemporalUnitPanel):
        raise CausalTypeError("panel must be a TemporalUnitPanel")
    if not isinstance(window, TemporalWindow):
        raise CausalTypeError("window must be a TemporalWindow")
    native, refusal = held._native.refresh(
        panel.snapshot_id, panel._wire(), window._wire(), bool(interval_existed)
    )
    _raise(refusal)
    refreshed = TemporalInitialStateResult._from(native)
    if refreshed.receipt is None or refreshed.previous_value is None:
        raise CausalValueError(
            "a refresh always carries its receipt", reason_code="invalid_argument"
        )
    return TemporalRefreshResult(
        label=refreshed.label,
        value=refreshed.value,
        previous_value=refreshed.previous_value,
        decision="reusable",
        receipt=refreshed.receipt,
        interval_invalidated=refreshed.interval_invalidated,
        result=refreshed,
    )


def temporal_dependent_interval(
    panel: TemporalUnitPanel,
    *,
    sequence: Sequence[int],
    estimand: str = OBSERVED_LABEL,
    fixed_state: int | None = None,
    target_law: InitialStateLaw | int | None = None,
    replicates: int = 500,
    seed: int = 0,
    level: float = 0.95,
    method: str = "percentile",
    min_units: int = 20,
    max_failed_fraction: float = 0.05,
) -> NoReturn:
    """A dependence-preserving sampling interval: **closed** until its calibration is measured.

    The arguments are validated through the Rust core first, so a panel without a unit
    map (``temporal_interval.unknown_units``), too few units
    (``temporal_interval.too_few_units``), too many replicates
    (``temporal_interval.too_many_replicates``) or an unsupported history
    (``temporal_interval.unsupported_history``) raise their own refusals. A well-formed
    request then raises ``cell_not_licensed`` with ``temporal_interval.route_frozen``;
    no interval is ever returned, and a point estimate is never relabeled as one.
    """
    if not isinstance(panel, TemporalUnitPanel):
        raise CausalTypeError("panel must be a TemporalUnitPanel")
    law, point = (None, None) if target_law is None else _law_args(target_law)
    refusal = _dependent_interval_closed(
        panel.snapshot_id,
        panel._wire(),
        _sequence(sequence),
        estimand,
        None if fixed_state is None else _index("fixed_state", fixed_state),
        law,
        point,
        _index("replicates", replicates, limit=_U64),
        _index("seed", seed, limit=_U64),
        float(level),
        method,
        _index("min_units", min_units, limit=_U64),
        float(max_failed_fraction),
    )
    raise TemporalExtensionRefusal(json.loads(refusal))


@dataclass(frozen=True, slots=True)
class ConsumedInitialState:
    """What an independent consumer replayed from an initial-state artifact."""

    label: str
    value: float
    contributions: tuple[StateContribution, ...]
    fixed: FixedStateResult | None
    replayed_value: float
    replayed_fixed_value: float | None
    law_population: str
    law_snapshot_id: str
    panel_snapshot_id: str
    sequence: tuple[int, int]
    premises: Mapping[str, Any]
    inference_claim: str


def consume_temporal_initial_state_artifact(
    artifact: bytes,
    *,
    max_summary_rows: int = 1_000_000,
    max_units: int = 100_000,
) -> ConsumedInitialState:
    """Independently replay an exported initial-state artifact.

    The target law is rebuilt (a source-labelled law refuses as
    ``initial_state.target_law_missing``) and the marginalized and fixed values are
    re-evaluated from the embedded law and panel summary; the stored labels, identities
    and values must match bit for bit, so a changed population, snapshot or law is refused
    even when the artifact is resealed.
    """
    if not isinstance(artifact, bytes):
        raise CausalTypeError("artifact must be bytes")
    payload, refusal = _consume_initial_state(
        artifact,
        max_summary_rows=_index("max_summary_rows", max_summary_rows, limit=_U64),
        max_units=_index("max_units", max_units, limit=_U64),
    )
    data = _consumed(payload, refusal)
    marginalized, fixed, replayed = data["marginalized"], data["fixed"], data["replayed_fixed"]
    return ConsumedInitialState(
        label=marginalized["label"],
        value=marginalized["value"],
        contributions=_contributions(marginalized["contributions"]),
        fixed=None if fixed is None else FixedStateResult(**fixed),
        replayed_value=data["replayed_value"],
        replayed_fixed_value=None if replayed is None else replayed["value"],
        law_population=data["law"]["population"],
        law_snapshot_id=data["law"]["snapshot_id"],
        panel_snapshot_id=data["panel_snapshot_id"],
        sequence=(data["sequence"][0], data["sequence"][1]),
        premises=data["premises"],
        inference_claim=data["inference_claim"],
    )


@dataclass(frozen=True, slots=True)
class ConsumedRefresh:
    """What an independent consumer re-decided and replayed from a refresh artifact."""

    decision: str
    invalidation: str | None
    old_value: float
    new_value: float | None
    interval_invalidated: bool
    interval_present: bool | None
    label: str | None
    receipt: RefreshReceipt | None
    old_window: Mapping[str, Any]
    new_window: Mapping[str, Any]
    sequence: tuple[int, int]
    inference_claim: str


def consume_temporal_refresh_artifact(
    artifact: bytes,
    *,
    max_summary_rows: int = 1_000_000,
    max_units: int = 100_000,
) -> ConsumedRefresh:
    """Independently re-decide and replay an exported refresh artifact.

    Both windows are checked against their own panel summaries, the refresh is
    re-decided from the identities and both points are re-evaluated; a changed period,
    snapshot or unit set is refused even when the artifact is resealed, and an artifact
    whose refreshed result carries an interval refuses as
    ``temporal_refresh.stale_interval``.
    """
    if not isinstance(artifact, bytes):
        raise CausalTypeError("artifact must be bytes")
    payload, refusal = _consume_refresh(
        artifact,
        max_summary_rows=_index("max_summary_rows", max_summary_rows, limit=_U64),
        max_units=_index("max_units", max_units, limit=_U64),
    )
    data = _consumed(payload, refusal)
    receipt = data["receipt"]
    return ConsumedRefresh(
        decision=data["decision"],
        invalidation=data["invalidation"],
        old_value=data["old_value"],
        new_value=data["new_value"],
        interval_invalidated=data["interval_invalidated"],
        interval_present=data["interval_present"],
        label=data["label"],
        receipt=None if receipt is None else RefreshReceipt._from(receipt),
        old_window=data["old_window"],
        new_window=data["new_window"],
        sequence=(data["sequence"][0], data["sequence"][1]),
        inference_claim=data["inference_claim"],
    )


__all__ = [
    "FIXED_LABEL",
    "MARGINALIZED_LABEL",
    "ConsumedInitialState",
    "ConsumedRefresh",
    "FixedStateResult",
    "InitialStateLaw",
    "RefreshReceipt",
    "SequenceHistory",
    "StateContribution",
    "TemporalExtensionRefusal",
    "TemporalInitialStateResult",
    "TemporalPremises",
    "TemporalRefreshResult",
    "TemporalUnitPanel",
    "TemporalWindow",
    "UnitHistories",
    "consume_temporal_initial_state_artifact",
    "consume_temporal_refresh_artifact",
    "temporal_dependent_interval",
    "temporal_initial_state",
    "temporal_new_period_refresh",
]
