"""Reusable execution policies with shared native admission accounting."""

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
    """Maximum admitted native memory; None permits an unlimited pool.

    Reusing an Execution shares its native pool across concurrent operations and
    retained products. The limit is not a process RSS estimate or a measurement.
    """

    memory_bytes: int | None = None

    def __post_init__(self) -> None:
        _integer(self.memory_bytes, "memory_bytes")


@dataclass(frozen=True, slots=True)
class CachePolicy:
    """Permission for semantic reuse, independently of numerical equivalence."""

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
        return cls("disabled", None)

    @classmethod
    def bounded(cls, max_bytes: int) -> CachePolicy:
        return cls("bounded", max_bytes)

    @classmethod
    def unbounded(cls) -> CachePolicy:
        return cls("unbounded", None)


@dataclass(frozen=True, slots=True)
class Execution:
    """Execution settings shared by preparation, execution and exploration.

    Scientific declarations and statistical budgets belong to the query,
    estimator and inference objects. Observers are bound to each executing run;
    retaining this policy never retains a completed run's exception scope.

    ``callback_failure_capacity`` bounds how many distinct callback failures one
    executing run retains so each can be attributed to its original exception
    (default 256, nonnegative). A run that exceeds it still fails with its own
    error; only the original-exception attribution of the surplus is dropped.

    ``pulse_extrapolation_ranges`` bounds how far a ``PulseEffect`` may extrapolate: the
    active level may lie at most this many observed treatment ranges (``max - min``)
    beyond the observed treatment interval (default ``None``, the engine's 10 ranges).
    A level inside the bound but outside the observed interval is computed and named in
    the result's support diagnostics with its distance in ranges and standard
    deviations; a level beyond the bound is refused with ``positivity_violated``. The
    control level is never refused, since a control of 0 outside the data is how a pulse
    is written.

    ``exposure_probability_work_limit`` bounds total logical operations for the
    exposure-probability procedure. None selects the engine default (4 billion);
    requests over the bound are refused without reducing their draw count.
    ``hmc_work_limit`` bounds total HMC sampling and diagnostic work. None selects
    the engine default (4 billion); a higher value authorizes more work without
    changing the requested sampling schedule.
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
