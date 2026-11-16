"""Execution policies: seed, threads, resource limits, cancellation and observers.

An :class:`Execution` says how a computation is run, never what is computed. Scientific
choices (query, estimator, inference, bootstrap or draw counts) are declared elsewhere; an
``Execution`` is passed as ``execution=`` to :func:`antecedent.prepare`,
:func:`antecedent.analyze` and most other entry points, and can be reused across them, in
which case its memory limit is shared.

Example::

    token = antecedent.CancellationToken()
    execution = antecedent.Execution(
        seed=7,
        threads=4,
        limits=antecedent.ResourceLimits(memory_bytes=2 * 1024**3),
        cancel=token,
    )
    outcome = antecedent.analyze(data, graph=graph, query=query, execution=execution)
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, ParamSpec, TypeVar

from . import _native
from .errors import CausalTypeError, CausalValueError

P = ParamSpec("P")
T = TypeVar("T")


def _integer(value: int | None, name: str, *, positive: bool = False) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int):
        raise CausalTypeError(f"{name} must be an integer")
    if value < (1 if positive else 0) or value > 2**64 - 1:
        raise CausalValueError(f"{name} is outside its supported range")


@dataclass(frozen=True, slots=True)
class ResourceLimits:
    """The memory an :class:`Execution` may admit.

    * ``memory_bytes`` -- upper bound, in bytes, on the native memory admitted under the
      execution: owned data copies, retained products and working storage. Default ``None`` is
      unlimited. A request that would exceed the bound is refused before the allocation is made.

    Every operation that reuses the same ``Execution`` draws on one shared pool, including
    concurrent operations and the results they retain. The bound governs what the engine admits;
    it is not an estimate or measurement of the process's resident memory.

    Raises:
        CausalTypeError: when ``memory_bytes`` is not an integer.
        CausalValueError: when it is negative or exceeds the unsigned 64-bit range.
    """

    memory_bytes: int | None = None

    def __post_init__(self) -> None:
        _integer(self.memory_bytes, "memory_bytes")


@dataclass(frozen=True, slots=True)
class TemporalMaterializationLimits:
    """Finite ceilings for temporal offsets, graph expansion, work and bytes.

    Offsets and work accumulate across one temporal operation; metadata exports
    have their own allowance under the execution policy retained with the result.
    ``max_bytes`` bounds each admitted materialization; graph limits apply to
    each unfolding.
    Use ``ResourceLimits.memory_bytes`` to bound the shared resident-memory pool.
    These limits also apply when that pool is unlimited. Zero refuses any
    materialization requiring that resource.
    """

    max_offsets: int = 100_000
    max_work: int = 100_000_000
    max_bytes: int = 256 * 1024 * 1024
    max_graph_nodes: int = 100_000
    max_graph_edges: int = 1_000_000

    def __post_init__(self) -> None:
        for name in ("max_offsets", "max_work", "max_bytes", "max_graph_nodes", "max_graph_edges"):
            value = getattr(self, name)
            if value is None:
                raise CausalTypeError(f"{name} must be an integer")
            _integer(value, name)

    def _native_tuple(self) -> tuple[int, int, int, int, int]:
        return (
            self.max_offsets,
            self.max_work,
            self.max_bytes,
            self.max_graph_nodes,
            self.max_graph_edges,
        )


@dataclass(frozen=True, slots=True)
class RandomizationTestLimits:
    """Maximum outcome-row visits in one complete-design Fisher test.

    The default is one billion. The full observed and allocation scans are
    charged; exceeding the limit refuses the test rather than reducing draws.
    """

    max_row_visits: int = 1_000_000_000

    def __post_init__(self) -> None:
        if self.max_row_visits is None:
            raise CausalTypeError("max_row_visits must be an integer")
        _integer(self.max_row_visits, "max_row_visits")


@dataclass(frozen=True, slots=True)
class CachePolicy:
    """Whether results of identical sub-computations may be reused, and how much may be kept.

    * ``mode`` -- ``"bounded"`` (default), ``"unbounded"`` or ``"disabled"``.
    * ``max_bytes`` -- size bound of a bounded cache in bytes (default ``4_000_000``). It must be
      set for ``"bounded"`` and ``None`` for the other modes.

    The policy grants or withholds permission to reuse; it does not change any computed number.
    Use the constructors :meth:`disabled`, :meth:`bounded` and :meth:`unbounded`.

    Raises:
        CausalValueError: when ``mode`` is not one of the three names, or ``max_bytes`` does not
            match the mode.
    """

    mode: Literal["disabled", "bounded", "unbounded"] = "bounded"
    max_bytes: int | None = 4_000_000

    def __post_init__(self) -> None:
        if self.mode not in ("disabled", "bounded", "unbounded"):
            raise CausalValueError("cache mode must be disabled, bounded or unbounded")
        _integer(self.max_bytes, "cache max_bytes")
        if (self.mode == "bounded") != (self.max_bytes is not None):
            raise CausalValueError("only a bounded cache declares max_bytes")

    @classmethod
    def disabled(cls) -> CachePolicy:
        """Return a policy that permits no reuse."""
        return cls("disabled", None)

    @classmethod
    def bounded(cls, max_bytes: int) -> CachePolicy:
        """Return a policy that permits reuse within ``max_bytes`` bytes."""
        return cls("bounded", max_bytes)

    @classmethod
    def unbounded(cls) -> CachePolicy:
        """Return a policy that permits reuse without a size bound."""
        return cls("unbounded", None)


@dataclass(frozen=True, slots=True)
class Execution:
    """How a computation runs: seed, threads, limits, cancellation and observers.

    Pass it as ``execution=``. The default ``Execution()`` uses seed 1, the engine's default
    thread count, no memory limit and a bounded cache. An execution is immutable; to change one
    on a prepared study use ``antecedent.reconfigure(study, execution=...)``.

    Scientific declarations and statistical budgets belong to the query, estimator and inference
    objects, not here.

    Fields:

    * ``seed`` -- seed of every random stream (cross-fitting folds, bootstrap, posterior draws).
      Nonnegative integer, default ``1``. The same seed, data and declarations reproduce the same
      streams.
    * ``threads`` -- worker threads, a positive integer; default ``None`` uses the engine's
      default.
    * ``limits`` -- a :class:`ResourceLimits` (default: unlimited memory).
    * ``cache`` -- a :class:`CachePolicy` (default: bounded at 4 MB).
    * ``kernels`` -- numerical kernel selection: ``"auto"`` (default, the production policy),
      ``"portable"`` (optimized kernels without architecture-specific SIMD) or ``"scalar"``
      (scalar kernels only).
    * ``cancel`` -- a :class:`~antecedent.CancellationToken`; calling its ``cancel()`` from
      another thread stops the running operation, which raises
      :class:`~antecedent.errors.CausalCancelledError`. Default ``None``.
    * ``on_progress`` -- callable ``(fraction, stage)`` invoked as work proceeds, with
      ``fraction`` in ``[0, 1]`` and ``stage`` a stage name. Default ``None``.
    * ``on_stage`` -- callable receiving the stage events of an estimate on routes that stream
      them. Routes that do not are refused with ``stage_stream_unavailable`` rather than run
      silently without it. Default ``None``.
    * ``callback_failure_capacity`` -- how many distinct callback failures one run retains so
      that each can be attributed to its original exception (default ``256``, nonnegative). A run
      that exceeds it still fails with its own error; only the attribution of the surplus is
      dropped.
    * ``pulse_extrapolation_ranges`` -- how far a :class:`~antecedent.PulseEffect` may
      extrapolate: the active level may lie at most this many observed treatment ranges
      (``max - min``) beyond the observed treatment interval. Default ``None`` is the engine's 10
      ranges. A level inside the bound but outside the observed interval is computed and named in
      the result's support diagnostics with its distance in ranges and standard deviations; a
      level beyond the bound is refused with ``positivity_violated``. The control level is never
      refused, since a control of 0 outside the data is how a pulse is written.
    * ``exposure_probability_work_limit`` -- bound on the total logical operations of the
      exposure-probability procedure. ``None`` selects the engine default (4 billion); a request
      over the bound is refused without reducing its draw count.
    * ``hmc_work_limit`` -- bound on total HMC sampling and diagnostic work. ``None`` selects the
      engine default (4 billion); a higher value authorizes more work without changing the
      requested sampling schedule.
    * ``temporal_materialization`` -- a :class:`TemporalMaterializationLimits`; ``None`` selects
      the engine defaults.
    * ``randomization_test`` -- a :class:`RandomizationTestLimits`; ``None`` selects the engine
      default of one billion row visits.

    ``cancel``, ``on_progress`` and ``on_stage`` do not take part in equality or ``repr``.
    Observers are bound to each run; keeping the policy does not keep a completed run's
    exceptions alive.

    Raises:
        CausalTypeError: when a field has the wrong type (for example ``cancel`` is not a
            ``CancellationToken`` or an observer is not callable).
        CausalValueError: when a number is outside its range or ``kernels`` is not one of the
            three names.
    """

    seed: int = 1
    threads: int | None = None
    limits: ResourceLimits = field(default_factory=ResourceLimits)
    cache: CachePolicy = field(default_factory=CachePolicy)
    kernels: Literal["auto", "portable", "scalar"] = "auto"
    cancel: Any | None = field(default=None, repr=False, compare=False)
    on_progress: Callable[[float, str], None] | None = field(
        default=None, repr=False, compare=False
    )
    on_stage: Callable[..., None] | None = field(default=None, repr=False, compare=False)
    callback_failure_capacity: int = 256
    pulse_extrapolation_ranges: float | None = None
    exposure_probability_work_limit: int | None = None
    hmc_work_limit: int | None = None
    temporal_materialization: TemporalMaterializationLimits | None = None
    randomization_test: RandomizationTestLimits | None = None
    _handle: Any = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _integer(self.seed, "seed")
        _integer(self.threads, "threads", positive=True)
        _integer(self.exposure_probability_work_limit, "exposure_probability_work_limit")
        _integer(self.hmc_work_limit, "hmc_work_limit")
        if self.callback_failure_capacity is None:
            raise CausalTypeError("callback_failure_capacity must be an integer")
        _integer(self.callback_failure_capacity, "callback_failure_capacity")
        if self.pulse_extrapolation_ranges is not None:
            ranges = self.pulse_extrapolation_ranges
            if isinstance(ranges, bool) or not isinstance(ranges, (int, float)):
                raise CausalTypeError("pulse_extrapolation_ranges must be a number")
            if not (math.isfinite(ranges) and ranges >= 0):
                raise CausalValueError("pulse_extrapolation_ranges must be finite and nonnegative")
        if self.threads is not None and self.threads > 2**32 - 1:
            raise CausalValueError("threads exceeds the native thread range")
        if not isinstance(self.limits, ResourceLimits) or not isinstance(self.cache, CachePolicy):
            raise CausalTypeError("execution limits and cache must use their typed declarations")
        if self.temporal_materialization is not None and not isinstance(
            self.temporal_materialization, TemporalMaterializationLimits
        ):
            raise CausalTypeError(
                "temporal_materialization must be TemporalMaterializationLimits or None"
            )
        if self.randomization_test is not None and not isinstance(
            self.randomization_test, RandomizationTestLimits
        ):
            raise CausalTypeError("randomization_test must be RandomizationTestLimits or None")
        if self.kernels not in ("auto", "portable", "scalar"):
            raise CausalValueError("kernels must be auto, portable or scalar")
        for name in ("on_progress", "on_stage"):
            callback = getattr(self, name)
            if callback is not None and not callable(callback):
                raise CausalTypeError(f"{name} must be callable")
        if self.cancel is not None and not isinstance(self.cancel, _native.CancellationToken):
            raise CausalTypeError("cancel must be a CancellationToken")
        object.__setattr__(
            self,
            "_handle",
            _native._Execution(
                seed=self.seed,
                threads=self.threads,
                memory_bytes=self.limits.memory_bytes,
                cache=self.cache.mode,
                cache_bytes=self.cache.max_bytes,
                kernel=self.kernels,
                cancel=self.cancel,
                on_progress=self.on_progress,
                callback_failure_capacity=self.callback_failure_capacity,
                exposure_probability_work_limit=self.exposure_probability_work_limit,
                hmc_work_limit=self.hmc_work_limit,
                randomization_test_max_row_visits=(
                    None
                    if self.randomization_test is None
                    else self.randomization_test.max_row_visits
                ),
                temporal_materialization_limits=(
                    None
                    if self.temporal_materialization is None
                    else self.temporal_materialization._native_tuple()
                ),
                pulse_extrapolation_ranges=(
                    None
                    if self.pulse_extrapolation_ranges is None
                    else float(self.pulse_extrapolation_ranges)
                ),
            ),
        )

    def _call(self, operation: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs) -> T:
        return self._handle.call(operation, *args, **kwargs)

    def _report_progress(self, fraction: float, stage: str) -> None:
        if self.on_progress is not None:
            try:
                self.on_progress(fraction, stage)
            except BaseException as error:
                # A callback can raise the same class as a scientific refusal.
                # Preserve its operational origin through composed outcome calls.
                object.__setattr__(error, "_antecedent_operational_callback", True)
                raise


def resolve_execution(execution: Execution | None) -> Execution:
    """The execution a public entry runs under: the supplied one, else the default."""
    execution = Execution() if execution is None else execution
    if not isinstance(execution, Execution):
        raise CausalTypeError("execution must be an antecedent.Execution")
    return execution


def native_controls(
    execution: Execution | None,
    *,
    seed: str | None = None,
    threads: str | None = None,
    cancel: str | None = None,
    memory: str | None = None,
    on_progress: str | None = None,
    callback_failure_capacity: str | None = None,
) -> dict[str, Any]:
    """The loose control keywords one native entry takes, read from an ``Execution``.

    Each argument names the native keyword that receives the matching control; a control
    left as ``None`` is not passed. ``execution=None`` means the default ``Execution()``.
    Public entries take ``execution=`` only and translate it here, so a physical control
    has exactly one owner and one spelling.
    """
    execution = Execution() if execution is None else execution
    if not isinstance(execution, Execution):
        raise CausalTypeError("execution must be an antecedent.Execution")
    controls: dict[str, Any] = {}
    if seed is not None:
        controls[seed] = execution.seed
    if threads is not None:
        controls[threads] = execution.threads
    if cancel is not None:
        controls[cancel] = execution.cancel
    if memory is not None:
        controls[memory] = execution.limits.memory_bytes
    if on_progress is not None:
        controls[on_progress] = execution.on_progress
    if callback_failure_capacity is not None:
        controls[callback_failure_capacity] = execution.callback_failure_capacity
    return controls
