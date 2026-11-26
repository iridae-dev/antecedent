"""Asyncio entry points for synchronous Antecedent execution."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable
from contextvars import copy_context
from dataclasses import dataclass, replace
from functools import partial
from threading import Lock
from typing import Any, Generic, TypeVar, overload

from . import _native, prepared
from ._batch import BatchResult, PreparedBatch
from .errors import CausalCancelledError, CausalTypeError
from .execution import Execution, resolve_execution
from .exposure_measurement import MeasuredAnalysis, PreparedMeasuredAnalysis
from .outcomes import Outcome
from .prepared import PreparedAnalysis

__all__ = ["Progress", "Result", "Stage", "analyze", "estimate", "prepare", "stream"]

T = TypeVar("T")
_Study = PreparedAnalysis[Any] | PreparedBatch | PreparedMeasuredAnalysis
_StudyT = TypeVar("_StudyT", bound=_Study)


@dataclass(frozen=True, slots=True)
class Progress:
    """The fraction and stage reported by the execution progress observer."""

    fraction: float
    stage: str


@dataclass(frozen=True, slots=True)
class Stage:
    """A scientific stage name and the payload from the stage observer."""

    name: str
    payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Result(Generic[T]):
    """The final outcome, including any scientific refusal."""

    outcome: Outcome[T]


class _Events:
    """Coalesce adjacent progress updates before waking the event loop."""

    def __init__(self) -> None:
        self.ready = asyncio.Event()
        self._loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        self._pending: list[Progress | Stage] = []
        self._lock = Lock()

    def push(self, event: Progress | Stage) -> None:
        with self._lock:
            if self._loop is None:
                return
            if not self._pending:
                self._loop.call_soon_threadsafe(self.ready.set)
            elif isinstance(event, Progress) and isinstance(self._pending[-1], Progress):
                self._pending[-1] = event
                return
            self._pending.append(event)

    def take(self) -> list[Progress | Stage]:
        self.ready.clear()
        with self._lock:
            pending = self._pending
            self._pending = []
            return pending

    def close(self) -> None:
        with self._lock:
            self._loop = None
            self._pending.clear()
            # asyncio.Event may keep a reference to the loop after waiting on it.
            self.ready = asyncio.Event()


def _controls(
    execution: Execution,
    events: _Events | None = None,
    *,
    stages: bool = False,
) -> tuple[Execution, _native.CancellationToken]:
    token = _native.CancellationToken() if execution.cancel is None else execution.cancel._child()
    progress, stage = execution.on_progress, execution.on_stage
    if events is None:
        return execution._with_async_controls(token, progress, stage), token

    def report_progress(fraction: float, name: str) -> None:
        events.push(Progress(fraction, name))
        if progress is not None:
            progress(fraction, name)

    def report_stage(name: str, payload: dict[str, Any]) -> None:
        events.push(Stage(name, dict(payload)))
        if stage is not None:
            stage(name, payload)

    controls = execution._with_async_controls(
        token, report_progress, report_stage if stages or stage is not None else None
    )
    return controls, token


def _submit(
    operation: Callable[[Execution], T], execution: Execution, token: _native.CancellationToken
) -> asyncio.Future[T]:
    def work() -> T:
        if token.is_cancelled():
            raise CausalCancelledError("execution cancelled before starting")
        return operation(execution)

    # Match to_thread's context propagation while keeping the worker exception on a
    # Future, including BaseExceptions raised by synchronous user callbacks.
    return asyncio.get_running_loop().run_in_executor(None, copy_context().run, work)


async def _stop(worker: asyncio.Future[Any], token: _native.CancellationToken) -> None:
    interrupted = None
    if not worker.done():
        token.cancel()
        done = asyncio.Event()
        worker.add_done_callback(lambda _: done.set())
        while not worker.done():
            try:
                await done.wait()
            except asyncio.CancelledError as error:
                interrupted = error
    if not worker.cancelled():
        worker.exception()
    if interrupted is not None:
        raise interrupted


async def _run(operation: Callable[[Execution], T], execution: Execution) -> T:
    controls, token = _controls(execution)
    worker = _submit(operation, controls, token)
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        await _stop(worker, token)
        raise


async def analyze(
    data: Any, *, query: Any, execution: Execution | None = None, **declarations: Any
) -> Outcome[Any]:
    """Await :func:`antecedent.analyze` with the same data and declarations."""
    policy = resolve_execution(execution)

    def analyze_on_worker(controls: Execution) -> Outcome[Any]:
        outcome = prepared.analyze(data, query=query, execution=controls, **declarations)
        return _restore_execution(outcome, policy)

    return await _run(analyze_on_worker, policy)


async def prepare(
    data: Any, *, query: Any, execution: Execution | None = None, **declarations: Any
) -> PreparedAnalysis[Any] | PreparedMeasuredAnalysis:
    """Await :func:`antecedent.prepare` with the same data and declarations."""
    policy = resolve_execution(execution)

    def prepare_on_worker(controls: Execution) -> PreparedAnalysis[Any] | PreparedMeasuredAnalysis:
        study = prepared.prepare(data, query=query, execution=controls, **declarations)
        return _with_execution(study, policy)

    return await _run(prepare_on_worker, policy)


def _study_execution(study: _Study) -> Execution:
    if not isinstance(study, (PreparedAnalysis, PreparedBatch, PreparedMeasuredAnalysis)):
        raise CausalTypeError("async estimation requires a prepared analysis or batch")
    return resolve_execution(study.execution)


def _with_execution(study: _StudyT, execution: Execution) -> _StudyT:
    if isinstance(study, PreparedMeasuredAnalysis):
        return replace(
            study,
            execution=execution,
            studies=tuple(replace(draw, execution=execution) for draw in study.studies),
        )
    return replace(study, execution=execution)


def _estimate(study: _Study, execution: Execution, *, retained: Execution) -> Outcome[Any]:
    outcome = prepared.estimate(_with_execution(study, execution))
    return _restore_execution(outcome, retained)


@overload
async def estimate(study: PreparedAnalysis[T]) -> Outcome[T]: ...


@overload
async def estimate(study: PreparedBatch) -> Outcome[BatchResult]: ...


@overload
async def estimate(study: PreparedMeasuredAnalysis) -> Outcome[MeasuredAnalysis]: ...


async def estimate(study: _Study) -> Outcome[Any]:
    """Await an estimate of an immutable prepared analysis, measured study or batch."""
    execution = _study_execution(study)
    return await _run(partial(_estimate, study, retained=execution), execution)


async def stream(
    study: _Study, *, stages: bool = False
) -> AsyncGenerator[Progress | Stage | Result[Any], None]:
    """Yield progress and optional scientific stages, followed by one result.

    Existing execution observers still run on the worker. Set `stages=True` to request
    stage events even when the study has no stage observer. Unsupported stage requests
    follow the synchronous route's refusal behavior. Adjacent pending progress updates
    can be coalesced. Close the generator (for example with `contextlib.aclosing`) when
    stopping iteration early to cancel and drain its worker.
    """
    execution = _study_execution(study)
    events = _Events()
    controls, token = _controls(execution, events, stages=stages)
    worker = _submit(partial(_estimate, study, retained=execution), controls, token)
    worker.add_done_callback(lambda _: events.ready.set())
    try:
        while not worker.done():
            await events.ready.wait()
            for event in events.take():
                yield event
        for event in events.take():
            yield event
        outcome = worker.result()
    finally:
        events.close()
        await _stop(worker, token)
    yield Result(outcome)


def _restore_execution(outcome: Outcome[Any], execution: Execution) -> Outcome[Any]:
    """Restore the declared policy on public retained studies."""
    study = outcome.study
    value = outcome.value
    if isinstance(study, PreparedAnalysis):
        return replace(outcome, _study=replace(study, execution=execution))
    if isinstance(study, PreparedBatch):
        retained = replace(study, execution=execution)
        if isinstance(value, BatchResult):
            value = replace(
                value,
                study=retained,
                members=tuple(_restore_execution(member, execution) for member in value.members),
            )
        return replace(outcome, _study=retained, value=value)
    return outcome
