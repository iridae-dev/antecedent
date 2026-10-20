"""Refresh of an autoregressive mechanism across observation periods, with a retained ledger.

The mechanism ``v_t = c + phi_1 v_{t-1} + ... + phi_L v_{t-L} + eps_t`` is fitted by least
squares and, on the same rows, by the conjugate normal-inverse-gamma posterior under a stated
prior. A later observation period is applied under the window-survival rules:

* a mechanism declared invariant across periods **extends** in place: the new rows are added
  to the least-squares summary and the earlier posterior is the prior of the new period. The
  result equals one fit of every retained row;
* otherwise the fit **restarts** on the new period alone under the originally stated prior;
* a period that overlaps the summarized one or reuses its snapshot is refused with the next
  attempt (a later period of a new snapshot); a different prior is refused with the rebuild
  from the retained blocks.

The ledger (:meth:`PeriodRefresh.export`) records the stated prior and its digest, every
period applied, and the blocks of rows the live fit used. :func:`consume_period_refresh_artifact`
recomputes the posterior and the least-squares fit from those rows alone and refuses a tampered
ledger or a block used twice. The route is unlicensed and uncalibrated: the posterior is that of
the stated linear Gaussian model under the stated prior, and a prior alone identifies no causal
target. Rust owns every rule; refusals raise
:class:`~antecedent.transport.advanced.TemporalExtensionRefusal`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from .._columns import coerce_fields
from .._native import NativePeriodRefresh as _NativePeriodRefresh
from .._native import consume_period_refresh_artifact as _consume_ledger
from ..errors import CausalTypeError, CausalValueError
from ._temporal_extensions import _consumed, _index

_U64 = 2**64 - 1
_HORIZON = 2


@dataclass(frozen=True, slots=True)
class PeriodPrior:
    """The stated prior over ``[intercept, phi_1, ..., phi_L]`` and the innovation variance.

    ``beta | sigma2 ~ N(mean, sigma2 * precision^-1)`` and ``sigma2 ~ InvGamma(shape, rate)``;
    ``precision`` is row-major.
    """

    mean: Sequence[float]
    precision: Sequence[float]
    shape: float
    rate: float

    def __post_init__(self) -> None:
        coerce_fields(
            self,
            mean=tuple(float(v) for v in self.mean),
            precision=tuple(float(v) for v in self.precision),
            shape=float(self.shape),
            rate=float(self.rate),
        )

    @classmethod
    def isotropic(cls, width: int, variance: float, shape: float, rate: float) -> PeriodPrior:
        """Independent coefficients with prior variance ``variance`` around zero."""
        if variance <= 0.0 or width < 1:
            raise CausalValueError(
                "an isotropic prior needs a positive variance and width",
                reason_code="invalid_argument",
            )
        precision = [0.0] * (width * width)
        for i in range(width):
            precision[i * width + i] = 1.0 / variance
        return cls([0.0] * width, precision, shape, rate)

    def _wire(self) -> tuple[list[float], list[float], float, float]:
        return (
            list(cast("Sequence[float]", self.mean)),
            list(cast("Sequence[float]", self.precision)),
            float(self.shape),
            float(self.rate),
        )


@dataclass(frozen=True, slots=True)
class PeriodWindow:
    """The observation window of one period: its identity and the declarations it was read under.

    ``period`` is the half-open ``(start, end)`` range of time indices, one value per index.
    ``cadence`` is the tick length of the observation clock and ``initial_law`` the initial-law
    id (empty when none).
    """

    period: tuple[int, int]
    snapshot_id: str
    unit_ids: Sequence[str]
    lag_alignment: Mapping[str, int] | Sequence[tuple[str, int]]
    intervention_history: Sequence[str]
    selection_targets: Sequence[str] = ()
    regimes: Sequence[str] = ("observational",)
    horizon: int = _HORIZON
    graph_id: str = ""
    proof_id: str = ""
    cadence: int = 1
    initial_law: str = ""

    def __post_init__(self) -> None:
        lag = self.lag_alignment
        pairs = lag.items() if isinstance(lag, Mapping) else lag
        coerce_fields(
            self,
            lag_alignment=tuple((str(n), int(s)) for n, s in pairs),
            intervention_history=tuple(self.intervention_history),
            selection_targets=tuple(self.selection_targets),
            regimes=tuple(self.regimes),
            unit_ids=tuple(str(u) for u in self.unit_ids),
        )
        if len(self.period) != _HORIZON:
            raise CausalValueError("period is a (start, end) pair", reason_code="invalid_argument")

    def _wire(self) -> tuple[Any, ...]:
        return (
            self.graph_id,
            _index("horizon", self.horizon, limit=_U64),
            [
                (n, _index("slice", s, limit=255))
                for n, s in cast("Sequence[tuple[str, int]]", self.lag_alignment)
            ],
            list(self.intervention_history),
            sorted(set(self.selection_targets)),
            sorted(set(self.regimes)),
            int(self.period[0]),
            int(self.period[1]),
            sorted(set(self.unit_ids)),
            self.snapshot_id,
            self.proof_id,
        )

    def _declared(self) -> tuple[int, str]:
        return _index("cadence", self.cadence, limit=_U64), self.initial_law


@dataclass(frozen=True, slots=True)
class PeriodRefreshOutcome:
    """How a period reached the cached results: ``extended`` or ``restarted``."""

    outcome: str
    contiguous: bool
    invalidated: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PeriodPosterior:
    """The normal-inverse-gamma posterior of the mechanism."""

    n: int
    mean: tuple[float, ...]
    precision: tuple[float, ...]
    shape: float
    rate: float

    @classmethod
    def _from(cls, data: Mapping[str, Any]) -> PeriodPosterior:
        return cls(
            n=int(data["n"]),
            mean=tuple(data["mean"]),
            precision=tuple(data["precision"]),
            shape=float(data["shape"]),
            rate=float(data["rate"]),
        )


@dataclass(frozen=True, slots=True)
class PeriodLeastSquares:
    """The least-squares fit of the mechanism."""

    n: int
    beta: tuple[float, ...]
    residual_variance: float | None

    @classmethod
    def _from(cls, data: Mapping[str, Any]) -> PeriodLeastSquares:
        variance = data["residual_variance"]
        return cls(
            n=int(data["n"]),
            beta=tuple(data["beta"]),
            residual_variance=None if variance is None else float(variance),
        )


@dataclass(frozen=True, slots=True)
class PeriodEvent:
    """One period applied to the mechanism, in order."""

    snapshot_id: str
    period: tuple[int, int]
    outcome: str
    contiguous: bool
    invalidated: tuple[str, ...]

    @classmethod
    def _from(cls, data: Mapping[str, Any]) -> PeriodEvent:
        return cls(
            snapshot_id=data["snapshot_id"],
            period=(int(data["period"][0]), int(data["period"][1])),
            outcome=data["outcome"],
            contiguous=bool(data["contiguous"]),
            invalidated=tuple(data["invalidated"]),
        )


def _outcome(payload: str | None, refusal: str | None) -> dict[str, Any]:
    return _consumed(payload, refusal)


class PeriodRefresh:
    """An autoregressive mechanism kept across observation periods."""

    __slots__ = ("_native",)

    def __init__(self, native: Any) -> None:
        self._native = native

    @classmethod
    def start(
        cls,
        values: Sequence[float],
        *,
        series_id: str,
        lags: int,
        prior: PeriodPrior,
        window: PeriodWindow,
        mechanism_invariant: bool = False,
    ) -> PeriodRefresh:
        """Fit the first period: one value per time index of ``window.period``."""
        cadence, law = window._declared()
        native, refusal = _NativePeriodRefresh.start(
            str(series_id),
            _index("lags", lags, limit=_U64),
            bool(mechanism_invariant),
            prior._wire(),
            window._wire(),
            cadence,
            law,
            [float(v) for v in values],
        )
        if refusal is not None:
            _consumed(None, refusal)
        if native is None:
            raise CausalValueError("the native fit returned nothing", reason_code="invalid_argument")
        return cls(native)

    def refresh(self, values: Sequence[float], window: PeriodWindow) -> PeriodRefreshOutcome:
        """Apply a later period. A refused period changes nothing."""
        cadence, law = window._declared()
        payload, refusal = self._native.refresh(
            window._wire(), cadence, law, [float(v) for v in values]
        )
        data = _outcome(payload, refusal)
        return PeriodRefreshOutcome(
            outcome=data["outcome"],
            contiguous=bool(data["contiguous"]),
            invalidated=tuple(data["invalidated"]),
        )

    def _payload(self) -> dict[str, Any]:
        payload, refusal = self._native.payload()
        return _outcome(payload, refusal)

    @property
    def posterior(self) -> PeriodPosterior:
        """The posterior over every period the fit currently covers."""
        return PeriodPosterior._from(self._payload()["posterior"])

    @property
    def least_squares(self) -> PeriodLeastSquares:
        """The least-squares fit of the same rows."""
        return PeriodLeastSquares._from(self._payload()["least_squares"])

    @property
    def events(self) -> tuple[PeriodEvent, ...]:
        """Every period applied so far, in order."""
        return tuple(PeriodEvent._from(e) for e in self._payload()["events"])

    @property
    def prior_digest(self) -> str:
        """Digest of the stated prior."""
        return str(self._payload()["prior_digest"])

    def evict_row(self, row_id: int) -> str:
        """Remove one retained row; both summaries are rebuilt from the rows that remain."""
        payload, refusal = self._native.evict_row(_index("row_id", row_id, limit=_U64))
        if refusal is not None:
            _consumed(None, refusal)
        if payload is None:
            raise CausalValueError("the native eviction returned nothing")
        return str(payload)

    def summary_under(self, prior: PeriodPrior) -> int:
        """The number of rows behind the posterior under ``prior``, which must be the stated one.

        Any other prior raises ``period_refresh.prior_changed``; the next attempt is
        :meth:`reprior`, the rebuild from every retained block.
        """
        payload, refusal = self._native.summary_under(prior._wire())
        return int(_outcome(payload, refusal)["n"])

    def reprior(self, prior: PeriodPrior) -> None:
        """Rebuild the posterior from every retained block under ``prior``."""
        _, refusal = self._native.reprior(prior._wire())
        if refusal is not None:
            _consumed(None, refusal)

    def check_against_fresh(self) -> bool:
        """Compare the cached results with a fresh rebuild from the retained rows."""
        ok, refusal = self._native.check_against_fresh()
        if refusal is not None:
            _consumed(None, refusal)
        return bool(ok)

    def export(self) -> bytes:
        """The retained delta ledger, framed."""
        data, refusal = self._native.export_ledger()
        if refusal is not None:
            _consumed(None, refusal)
        if data is None:
            raise CausalValueError("the native export returned nothing")
        return bytes(data)


@dataclass(frozen=True, slots=True)
class ConsumedPeriodRefresh:
    """What an independent consumer recomputed from a retained ledger."""

    series_id: str
    lags: int
    standing: str
    prior_digest: str
    rows_used: int
    events: tuple[PeriodEvent, ...]
    live_snapshots: tuple[str, ...]
    posterior: PeriodPosterior
    least_squares: PeriodLeastSquares


def consume_period_refresh_artifact(artifact: bytes) -> ConsumedPeriodRefresh:
    """Recompute a retained ledger from its blocks and refuse any change to it.

    The posterior and the least-squares fit are recomputed from the retained rows alone; a
    changed seal, prior, block or stored result, a period that overlaps an earlier one or
    reuses a snapshot, and a row or block used twice are refused.
    """
    if not isinstance(artifact, bytes):
        raise CausalTypeError("artifact must be bytes")
    payload, refusal = _consume_ledger(artifact)
    data = _consumed(payload, refusal)
    recomputed = data["recomputed"]
    return ConsumedPeriodRefresh(
        series_id=data["series_id"],
        lags=int(data["lags"]),
        standing=data["standing"],
        prior_digest=data["prior_digest"],
        rows_used=int(data["rows_used"]),
        events=tuple(PeriodEvent._from(e) for e in data["events"]),
        live_snapshots=tuple(recomputed["live_snapshots"]),
        posterior=PeriodPosterior._from(recomputed["posterior"]),
        least_squares=PeriodLeastSquares._from(recomputed["least_squares"]),
    )


__all__ = [
    "ConsumedPeriodRefresh",
    "PeriodEvent",
    "PeriodLeastSquares",
    "PeriodPosterior",
    "PeriodPrior",
    "PeriodRefresh",
    "PeriodRefreshOutcome",
    "PeriodWindow",
    "consume_period_refresh_artifact",
]
