"""Incremental causal state helpers."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, TypeVar

from . import _native
from ._native import (
    CancellationToken,
    GraphScoreCache,
    RollingMechanismDiagnostics,
)
from .errors import CausalStateError, CausalTypeError, CausalValueError
from .execution import CachePolicy, Execution

__all__ = [
    "CancellationToken",
    "CausalState",
    "GraphScoreCache",
    "RefreshMutation",
    "RefreshObservation",
    "RefreshSummary",
    "RetainedBootstrap",
    "RetainedRefresh",
    "RetentionPolicy",
    "RollingMechanismDiagnostics",
    "SpillRegistry",
    "SpilledBootstrap",
    "SpilledRefresh",
    "antecedent_state_append",
    "publish_recomputed_results",
    "result_lineage_fingerprint",
]

T = TypeVar("T")

_LAYERS = (
    "target",
    "identification",
    "identification_product",
    "program",
    "inference_binding",
    "observation",
    "data_snapshot",
)
_OPTIONAL_LAYERS = ("identification_product", "program")


_UNBOUNDED_CACHE_BYTES = 2**64 - 1


def _execution(execution: Execution | None) -> Execution:
    execution = Execution() if execution is None else execution
    if not isinstance(execution, Execution):
        raise CausalTypeError("execution must be an antecedent.Execution")
    return execution


def _cache_bytes(execution: Execution | None) -> int:
    """The result-cache byte budget an execution's cache policy grants."""
    policy: CachePolicy = _execution(execution).cache
    if policy.mode == "disabled":
        return 0
    if policy.mode == "unbounded":
        return _UNBOUNDED_CACHE_BYTES
    assert policy.max_bytes is not None
    return policy.max_bytes


class CausalState:
    """What a data or premise event made stale, with a result cache bounded by an execution.

    ``execution.cache`` sets the cache byte budget (a disabled cache admits nothing, an
    unbounded one is limited only by the machine). The state records staleness and never
    recomputes; every operation of the native state is available unchanged.
    """

    __slots__ = ("_execution", "_native")

    def __init__(self, execution: Execution | None = None) -> None:
        self._execution = None if execution is None else _execution(execution)
        self._native = _native.CausalState(_cache_bytes(execution))

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        member = getattr(self._native, name)
        run = self._execution
        if run is None or not callable(member):
            return member

        def call(*args: Any, **kwargs: Any) -> Any:
            return run._call(member, *args, **kwargs)  # noqa: SLF001

        return call

    def particle_filter_init(
        self,
        key: str,
        n_particles: int,
        *,
        a: float = 0.9,
        process_std: float = 0.3,
        obs_std: float = 0.5,
        execution: Execution | None = None,
    ) -> None:
        """Start a bootstrap particle filter whose stream is seeded by ``execution``."""
        run = _execution(self._execution if execution is None else execution)
        run._call(  # noqa: SLF001
            self._native.particle_filter_init,
            key,
            n_particles,
            a=a,
            process_std=process_std,
            obs_std=obs_std,
            seed=run.seed,
        )

    def __repr__(self) -> str:
        return f"<CausalState version={self._native.version}>"


def antecedent_state_append(
    n_appends: int = 2, *, execution: Execution | None = None
) -> tuple[int, int]:
    """Append synthetic batches to a fresh state and report ``(version, stale_count)``."""
    run = _execution(execution)
    return run._call(  # noqa: SLF001
        _native.antecedent_state_append, n_appends, cache_bytes=_cache_bytes(run)
    )


def _identity_layers(identities: Any) -> dict[str, str | None]:
    """The seven identity layers of a contract, from a mapping keyed by layer name (``target``,
    ``identification``, ...) or by ``<layer>_id`` (an outcome's ``identities``), or any object
    carrying the ``<layer>_id`` attributes. A prepared study's inspection report carries no data
    snapshot until an estimate seals one, so the identities of a recomputed result are its
    outcome's."""
    layers: dict[str, str | None] = {}
    for layer in _LAYERS:
        if isinstance(identities, Mapping):
            value = identities.get(layer, identities.get(f"{layer}_id"))
        else:
            value = getattr(identities, f"{layer}_id", None)
        if value is None and layer not in _OPTIONAL_LAYERS:
            raise CausalValueError(f"identities lack the required layer {layer!r}")
        if value is not None and not isinstance(value, str):
            raise CausalTypeError(f"identity layer {layer!r} must be a hex string")
        layers[layer] = value
    return layers


def result_lineage_fingerprint(identities: Any) -> str:
    """Lineage digest (hex) of the contract a result was computed under.

    It covers every identity layer, so two estimands, two priors or two bootstrap budgets never
    share a lineage key. ``identities`` is a mapping of layer name to hex digest, such as the
    ``identities`` of the outcome of an estimate, or an object carrying the ``<layer>_id`` attributes.
    """
    return str(_native.state_result_lineage_fingerprint(_identity_layers(identities)))


def publish_recomputed_results(
    state: CausalState,
    expected_version: int,
    updates: Sequence[tuple[int, Any, int]],
) -> None:
    """Publish recomputed results for ``expected_version``, binding what was recomputed.

    ``updates`` holds ``(query_id, identities, bytes)`` per recomputed query, with the
    identities of the prepared study the result was recomputed under. Raises
    :class:`~antecedent.errors.CausalStateError` for a version that advanced during the
    recompute, a data snapshot recorded before the latest data event, an identity layer that
    predates a changed stage input, an unknown query or a cache budget overrun; nothing is
    published then. A query published here is lineage-bound and the unchecked
    ``CausalState.refresh_results`` refuses it.
    """
    if not isinstance(state, CausalState):
        raise CausalTypeError("state must be a CausalState")
    bound = [(int(query), _identity_layers(ids), int(nbytes)) for query, ids, nbytes in updates]
    state._native.publish_recomputed(int(expected_version), bound)  # noqa: SLF001


# --------------------------------------------------------------------------- retained refresh

_OPERATIONS = (
    "row_reconstruction",
    "refit_from_retained_rows",
    "exact_row_eviction",
    "in_place_column_insert",
    "in_place_weight_or_mask_update",
    "reuse_full_state",
    "reuse_summary",
    "read_disk_artifact",
)


@dataclass(frozen=True, slots=True)
class RefreshObservation:
    """One observation offered to a retained refresh.

    A least-squares family reads ``x`` as the design row and ``y`` as the response; the
    covariance family reads ``x`` as the observation vector and ignores ``y``. ``weight`` is a
    non-negative integer count for ``ols`` and ``conjugate_blocks``, a non-negative real for
    ``weighted_ols`` and one for ``covariance``. Masked rows are retained and left out of the
    summary.
    """

    id: int
    x: tuple[float, ...]
    y: float = 0.0
    weight: float = 1.0
    masked: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", tuple(float(v) for v in self.x))

    def _wire(self) -> dict[str, Any]:
        return {
            "id": int(self.id),
            "x": list(self.x),
            "y": float(self.y),
            "weight": float(self.weight),
            "masked": bool(self.masked),
        }


@dataclass(frozen=True, slots=True)
class RefreshMutation:
    """A change to the data a retained refresh summarizes (build with the constructors)."""

    _body: Mapping[str, Any] = field(repr=False)
    kind: str = ""

    @classmethod
    def _make(cls, kind: str, **body: Any) -> RefreshMutation:
        return cls(MappingProxyType({"kind": kind, **body}), kind)

    @classmethod
    def append(cls, source: str, rows: Sequence[RefreshObservation]) -> RefreshMutation:
        """Append a window of rows of ``source``; updated in place."""
        return cls._make("append", source=source, rows=[r._wire() for r in rows])

    @classmethod
    def window(cls, max_rows: int) -> RefreshMutation:
        """Keep the most recently appended ``max_rows`` rows; recomputed."""
        return cls._make("window", max_rows=int(max_rows))

    @classmethod
    def evict(cls, row_id: int) -> RefreshMutation:
        """Remove one retained row; recomputed."""
        return cls._make("evict", id=int(row_id))

    @classmethod
    def set_weight(cls, row_id: int, weight: float) -> RefreshMutation:
        """Change the weight of one retained row; recomputed."""
        return cls._make("set_weight", id=int(row_id), weight=float(weight))

    @classmethod
    def set_mask(cls, row_id: int, masked: bool) -> RefreshMutation:
        """Change the mask of one retained row; recomputed."""
        return cls._make("set_mask", id=int(row_id), masked=bool(masked))

    @classmethod
    def replace(cls, source: str, rows: Sequence[RefreshObservation]) -> RefreshMutation:
        """Replace the source and its rows; nothing of the old source survives."""
        return cls._make("replace", source=source, rows=[r._wire() for r in rows])

    @classmethod
    def in_block(cls, label: str, mutation: RefreshMutation) -> RefreshMutation:
        """Apply ``mutation`` inside the labelled block of a conjugate posterior."""
        return cls._make("block", label=label, mutation=dict(mutation._body))

    def _wire(self) -> str:
        return json.dumps(dict(self._body))


@dataclass(frozen=True, slots=True)
class RefreshSummary:
    """The complete result a retained refresh stands for.

    ``family`` is ``ols``, ``weighted_ols``, ``covariance`` or ``conjugate_blocks``; ``values``
    holds that family's fields (coefficients, counts, scatter or posterior).
    """

    family: str
    values: Mapping[str, Any]

    @classmethod
    def _of(cls, text: str) -> RefreshSummary:
        data = json.loads(text)
        return cls(str(data["family"]), MappingProxyType(data))

    def __getitem__(self, key: str) -> Any:
        return self.values[key]


@dataclass(frozen=True, slots=True)
class RetentionPolicy:
    """What a result keeps of the rows it was computed from.

    ``raw_history`` keeps every row; ``bounded_window(n)`` keeps the trailing ``n`` rows (zero
    keeps none); ``sufficient_statistics`` keeps only the summary. A policy that drops rows
    names everything it can no longer serve.
    """

    _native: Any = field(repr=False, compare=False)

    @classmethod
    def raw_history(cls) -> RetentionPolicy:
        return cls(_native.RetentionPolicy.raw_history())

    @classmethod
    def bounded_window(cls, max_rows: int) -> RetentionPolicy:
        if isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows < 0:
            raise CausalValueError("max_rows must be a non-negative integer")
        return cls(_native.RetentionPolicy.bounded_window(max_rows))

    @classmethod
    def sufficient_statistics(cls) -> RetentionPolicy:
        return cls(_native.RetentionPolicy.sufficient_statistics())

    @property
    def kind(self) -> str:
        return str(self._native.kind())

    @property
    def max_retained_rows(self) -> int | None:
        return self._native.max_retained_rows()

    @property
    def result_mode(self) -> str:
        """``full-reusable`` or ``summary-only``."""
        return str(self._native.result_mode().as_str())

    @property
    def no_longer_serves(self) -> tuple[str, ...]:
        return tuple(op.as_str() for op in self._native.no_longer_serves())

    def require(self, operation: str) -> None:
        """Accept ``operation`` under this policy or raise :class:`CausalStateError` naming the
        mode, the operation and everything the mode can no longer serve."""
        self._native.require(_operation(operation))


def _operation(name: str) -> Any:
    if name not in _OPERATIONS:
        raise CausalValueError(f"unknown retained operation {name!r}; use one of {_OPERATIONS}")
    return getattr(_native.RetainedOperation, name)()


def _operation_name(name: str) -> str:
    """The retained operation ``name`` for the native summaries, which take it as text."""
    if name not in _OPERATIONS:
        raise CausalValueError(f"unknown retained operation {name!r}; use one of {_OPERATIONS}")
    return name


def _run(execution: Any, call: Callable[..., T], *args: Any) -> T:
    """Run a native call under the admission pool of ``execution``."""
    return call(*args) if execution is None else execution._call(call, *args)  # noqa: SLF001


class SpillRegistry:
    """A directory of spilled results whose byte budget never exceeds the admission pool of
    the :class:`~antecedent.execution.Execution` it is opened under."""

    def __init__(self, directory: str | Path, budget_bytes: int, *, execution: Any = None) -> None:
        if isinstance(budget_bytes, bool) or not isinstance(budget_bytes, int) or budget_bytes < 0:
            raise CausalValueError("budget_bytes must be a non-negative integer")
        self._native = _run(execution, _native.spill_registry_open, str(directory), budget_bytes)

    @property
    def directory(self) -> Path:
        return Path(self._native.directory())

    @property
    def stats(self) -> Mapping[str, int]:
        """Entries, used and budget bytes, and the spill, eviction and reload counters."""
        return MappingProxyType(json.loads(self._native.stats_json()))

    def contains(self, key: str) -> bool:
        return bool(self._native.contains(key))

    def keys_by_recency(self) -> tuple[str, ...]:
        return tuple(self._native.keys_by_recency())

    def __repr__(self) -> str:
        return f"SpillRegistry({str(self.directory)!r}, {dict(self.stats)!r})"


def _engine_native(study: Any) -> Any:
    native = getattr(getattr(study, "_engine", None), "_native", None)
    if native is None:
        raise CausalStateError("this preparation keeps no native summary to refresh")
    return native


class RetainedRefresh:
    """The least-squares, scatter or conjugate summary a prepared study's estimator keeps, over
    rows of its source.

    Each change is applied to the summary and checked against a fresh reconstruction from the
    retained rows; an append updates in place, everything else recomputes, and a refused change
    leaves the summary as it was. Nothing here re-identifies or re-estimates the study.
    """

    def __init__(self, native: Any, study: Any) -> None:
        self._native = native
        self._study = study

    @classmethod
    def start(
        cls,
        study: Any,
        *,
        source: str,
        width: int,
        rows: Sequence[RefreshObservation],
        prior: Mapping[str, Any] | None = None,
    ) -> RetainedRefresh:
        """Start the summary ``study``'s estimator keeps over ``rows`` of ``source``.

        A conjugate study also names its ``prior`` (``mean``, row-major ``precision``,
        ``shape``, ``rate`` and the ``label`` of the first block). Raises for an estimator
        whose result is not a function of such a summary.
        """
        native = _engine_native(study)
        wire = json.dumps([r._wire() for r in rows])
        prior_wire = None if prior is None else json.dumps(dict(prior))
        handle = _native.retained_refresh_start(native, source, int(width), wire, prior_wire)
        return cls(handle, study)

    @staticmethod
    def family_of(study: Any) -> str:
        """The summary family ``study``'s estimator keeps."""
        return str(_native.retained_refresh_family(_engine_native(study)))

    @property
    def family(self) -> str:
        return str(self._native.family())

    def apply(self, mutation: RefreshMutation) -> Literal["updated", "recomputed"]:
        """Apply one change and verify it against a fresh reconstruction."""
        if not isinstance(mutation, RefreshMutation):
            raise CausalTypeError("mutation must be a RefreshMutation")
        return self._native.apply(mutation._wire())

    def result(self) -> RefreshSummary:
        """The complete cached result."""
        return RefreshSummary._of(self._native.result_json())

    def fresh_result(self) -> RefreshSummary:
        """The complete result rebuilt from the retained rows alone."""
        return RefreshSummary._of(self._native.fresh_result_json())

    def verify(self) -> None:
        """Compare the cached result with a fresh reconstruction; raises on a disagreement."""
        self._native.check_against_fresh()

    def summary_only(self) -> RetainedRefresh:
        """A copy that no longer retains rows; every change needing them is refused."""
        return RetainedRefresh(self._native.summary_only(), self._study)

    def with_retention(self, policy: RetentionPolicy) -> RetainedRefresh:
        """A copy under ``policy``: a window keeps the trailing rows, zero rows keeps none."""
        if not isinstance(policy, RetentionPolicy):
            raise CausalTypeError("policy must be a RetentionPolicy")
        return RetainedRefresh(self._native.with_retention(policy._native), self._study)

    def require(self, operation: str) -> None:
        """Accept ``operation`` under this summary's retention or raise the refusal."""
        self._native.require(_operation_name(operation))

    @property
    def frame_bytes(self) -> int:
        """Bytes a spill or reload asks the admission pool for."""
        return int(self._native.frame_bytes())

    def spill(self, registry: SpillRegistry, key: str, *, execution: Any = None) -> SpilledRefresh:
        """Write the summary to disk under ``registry`` after admitting its frame against the
        admission pool of ``execution`` (the study's own by default)."""
        if not isinstance(registry, SpillRegistry):
            raise CausalTypeError("registry must be a SpillRegistry")
        run = getattr(self._study, "execution", None) if execution is None else execution
        return SpilledRefresh(_run(run, self._native.spill, registry._native, key), self._study)

    def __repr__(self) -> str:
        return f"RetainedRefresh(family={self.family!r})"


class SpilledRefresh:
    """A refresh summary written to disk: it serves a read of the artifact and refuses every
    resident operation with the reload that serves it."""

    def __init__(self, native: Any, study: Any) -> None:
        self._native = native
        self._study = study

    @property
    def family(self) -> str:
        return str(self._native.family())

    @property
    def mode(self) -> str:
        return str(self._native.mode())

    @property
    def key(self) -> str:
        return str(self._native.key())

    @property
    def on_disk(self) -> bool:
        return bool(self._native.on_disk())

    @property
    def evicted(self) -> tuple[str, ...]:
        """Keys evicted from the registry to make room for this spill."""
        return tuple(self._native.evicted())

    def require(self, operation: str) -> None:
        self._native.require(_operation_name(operation))

    def reload(self, *, execution: Any = None) -> RetainedRefresh:
        """Read the summary back as a resident refresh after admitting its frame."""
        run = getattr(self._study, "execution", None) if execution is None else execution
        return RetainedRefresh(_run(run, self._native.reload), self._study)

    def __repr__(self) -> str:
        return f"SpilledRefresh(family={self.family!r}, on_disk={self.on_disk})"


class RetainedBootstrap:
    """The replicate estimates of one bootstrap and the running summary they stand for.

    Under summary-only retention the replicates are dropped: the standard error and the replicate
    accounting stay, an interval from the replicates is refused with its next attempt.
    """

    def __init__(self, estimates: Sequence[float] = (), failed: int = 0, *, handle: Any = None):
        self._native = (
            handle
            if handle is not None
            else _native.RetainedBootstrapHandle([float(v) for v in estimates], failed)
        )

    @property
    def retention(self) -> str:
        return str(self._native.retention())

    @property
    def estimates(self) -> tuple[float, ...]:
        return tuple(self._native.estimates())

    @property
    def accounting(self) -> Mapping[str, Any]:
        """The standard error and the successful and failed replicate counts."""
        return MappingProxyType(json.loads(self._native.accounting_json()))

    @property
    def mean(self) -> float | None:
        return self._native.mean()

    def append(self, estimates: Sequence[float], failed: int = 0) -> None:
        self._native.append(list(map(float, estimates)), failed)

    def percentile_interval(self, level: float) -> tuple[float, float]:
        """Equal-tailed percentile interval; refused when the replicates are not retained."""
        return self._native.percentile_interval(float(level))

    def verify(self) -> None:
        self._native.check_against_fresh()

    def summary_only(self) -> RetainedBootstrap:
        return RetainedBootstrap(handle=self._native.summary_only())

    @property
    def frame_bytes(self) -> int:
        return int(self._native.frame_bytes())

    def spill(
        self, registry: SpillRegistry, key: str, *, execution: Any = None
    ) -> SpilledBootstrap:
        if not isinstance(registry, SpillRegistry):
            raise CausalTypeError("registry must be a SpillRegistry")
        return SpilledBootstrap(_run(execution, self._native.spill, registry._native, key))


class SpilledBootstrap:
    """A bootstrap result written to disk."""

    def __init__(self, native: Any) -> None:
        self._native = native

    @property
    def mode(self) -> str:
        return str(self._native.mode())

    @property
    def on_disk(self) -> bool:
        return bool(self._native.on_disk())

    @property
    def evicted(self) -> tuple[str, ...]:
        return tuple(self._native.evicted())

    def require(self, operation: str) -> None:
        self._native.require(_operation_name(operation))

    def reload(self, *, execution: Any = None) -> RetainedBootstrap:
        return RetainedBootstrap(handle=_run(execution, self._native.reload))
