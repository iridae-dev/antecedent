"""Reusable execution policies with shared native admission accounting."""

from __future__ import annotations

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
    _handle: Any = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _integer(self.seed, "seed")
        _integer(self.threads, "threads", positive=True)
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
