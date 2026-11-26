"""Async execution uses the native run controls and drains its workers."""

from __future__ import annotations

import asyncio
from contextlib import aclosing
from contextvars import ContextVar
from threading import Event, get_ident

import antecedent as ant
import numpy as np
import pytest
from antecedent import aio, prepared
from antecedent.errors import CausalCancelledError


@pytest.fixture
def request_args():
    rng = np.random.default_rng(11)
    z = rng.normal(size=240)
    t = (z + rng.normal(size=len(z)) > 0).astype(float)
    y = 2 * t + z + rng.normal(size=len(z))
    return {
        "data": {"t": t, "y": y, "z": z},
        "graph": [("z", "t"), ("z", "y"), ("t", "y")],
        "query": ant.AverageEffect("t", "y"),
        "inference": ant.Frequentist(bootstrap=8),
        "refute": "none",
    }


def test_async_lifecycle_preserves_results_and_shared_admission(request_args):
    execution = ant.Execution(seed=11, threads=1, limits=ant.ResourceLimits(memory_bytes=10**7))

    async def run():
        study = await aio.prepare(**request_args, execution=execution)
        assert execution._handle.available_memory_bytes < 10**7
        assert study.execution is execution
        estimated, analyzed = await asyncio.gather(
            aio.estimate(study), aio.analyze(**request_args, execution=execution)
        )
        expected = ant.analyze(**request_args, execution=execution)
        assert estimated.answer.value == analyzed.answer.value == expected.answer.value
        assert estimated.study.execution is analyzed.study.execution is execution

    asyncio.run(run())


def test_stream_delivers_native_stages_before_result(request_args):
    stages = []
    callback_threads = []
    loop_thread = get_ident()

    def on_stage(name, _payload):
        stages.append(name)
        callback_threads.append(get_ident())

    execution = ant.Execution(on_stage=on_stage)

    async def run():
        study = await aio.prepare(**request_args, execution=execution)
        events = [event async for event in aio.stream(study, stages=True)]
        assert any(isinstance(event, aio.Progress) for event in events)
        stage_events = [event for event in events if isinstance(event, aio.Stage)]
        assert [event.name for event in stage_events] == [
            "identify",
            "estimate_point",
            "uncertainty",
            "validate",
        ]
        assert stages == [event.name for event in stage_events]
        assert all(thread != loop_thread for thread in callback_threads)
        result = events[-1]
        assert isinstance(result, aio.Result)
        assert result.outcome.accepted
        assert result.outcome.study.execution is study.execution

    asyncio.run(run())


def test_stream_preserves_native_callback_errors(request_args):
    original = RuntimeError("observer failed")

    def fail(*_args):
        raise original

    async def run():
        study = await aio.prepare(**request_args, execution=ant.Execution(on_stage=fail))
        with pytest.raises(RuntimeError, match="observer failed") as raised:
            async for _ in aio.stream(study):
                pass
        assert raised.value is original

    asyncio.run(run())


def test_timeout_cancels_only_its_worker_and_waits_for_exit(monkeypatch):
    parent = ant.CancellationToken()
    execution = ant.Execution(cancel=parent)
    context = ContextVar("investigation")
    tokens = {}
    finished = Event()
    release_sibling = Event()

    async def run():
        loop = asyncio.get_running_loop()
        started = {name: asyncio.Event() for name in ("slow", "sibling")}

        def analyze(name, *, execution, **_kwargs):
            assert context.get() == name
            tokens[name] = execution.cancel
            loop.call_soon_threadsafe(started[name].set)
            if name == "sibling":
                assert release_sibling.wait(5)
                assert not execution.cancel.is_cancelled()
                return ant.Outcome(value=name)
            try:
                for _ in range(500):
                    if execution.cancel.is_cancelled():
                        raise CausalCancelledError("worker observed cancellation")
                    finished.wait(0.01)
                raise AssertionError("worker never received cancellation")
            finally:
                finished.set()

        monkeypatch.setattr(prepared, "analyze", analyze)
        async with asyncio.TaskGroup() as group:
            context.set("sibling")
            sibling = group.create_task(aio.analyze("sibling", query=None, execution=execution))
            try:
                await asyncio.wait_for(started["sibling"].wait(), 5)
                context.set("slow")
                with pytest.raises(TimeoutError):
                    async with asyncio.timeout(None) as timeout:
                        # Start the timeout only after the worker has entered the operation.
                        async def expire():
                            await started["slow"].wait()
                            timeout.reschedule(loop.time())

                        group.create_task(expire())
                        await aio.analyze("slow", query=None, execution=execution)
                assert finished.is_set()
                assert tokens["slow"].is_cancelled()
                assert not tokens["sibling"].is_cancelled()
                assert not parent.is_cancelled()
            finally:
                release_sibling.set()
        assert sibling.result().value == "sibling"

    asyncio.run(run())


def test_cancelled_parent_is_honored_before_start(request_args):
    token = ant.CancellationToken()
    token.cancel()
    with pytest.raises(CausalCancelledError):
        asyncio.run(aio.analyze(**request_args, execution=ant.Execution(cancel=token)))


def test_closing_stream_cancels_and_drains_worker(monkeypatch, request_args):
    finished = Event()

    def estimate(study):
        study.execution.on_progress(0.25, "running")
        try:
            for _ in range(500):
                if study.execution.cancel.is_cancelled():
                    raise CausalCancelledError("worker observed cancellation")
                finished.wait(0.01)
            raise AssertionError("worker never received cancellation")
        finally:
            finished.set()

    async def run():
        study = await aio.prepare(**request_args)
        monkeypatch.setattr(prepared, "estimate", estimate)
        async with aclosing(aio.stream(study)) as events:
            assert await asyncio.wait_for(anext(events), 5) == aio.Progress(0.25, "running")
        assert finished.is_set()
        assert study.execution.cancel is None

    asyncio.run(run())


def test_stream_preserves_scientific_refusals(request_args):
    request_args["identifier"] = "frontdoor"
    request_args["estimator"] = str(ant.Estimator.FRONTDOOR_FUNCTIONAL)

    async def run():
        study = await aio.prepare(**request_args)
        assert study.refusal.code == "effect_not_identified"
        events = [event async for event in aio.stream(study)]
        assert len(events) == 1
        assert isinstance(events[0], aio.Result)
        assert events[0].outcome.refusal is study.refusal

    asyncio.run(run())


def test_stream_batch_can_request_stages_without_configured_observer(request_args):
    from antecedent.estimation import prepare_many

    data = request_args["data"]
    graph = request_args["graph"]
    study = prepare_many(
        {**data, "y2": 0.5 * data["y"] - data["z"]},
        graph=[*graph, ("t", "y2"), ("z", "y2")],
        queries=[ant.AverageEffect("t", "y"), ant.AverageEffect("t", "y2")],
        estimator="aipw",
        inference=ant.Frequentist(bootstrap=0),
        refute="none",
    )

    async def run():
        events = [event async for event in aio.stream(study, stages=True)]
        assert {
            event.payload["query_index"] for event in events if isinstance(event, aio.Stage)
        } == {
            0,
            1,
        }
        result = events[-1]
        assert isinstance(result, aio.Result)
        assert result.outcome.accepted
        assert result.outcome.study is result.outcome.value.study
        assert result.outcome.study.execution.on_stage is None
        assert all(
            member.study.execution.on_stage is None for member in result.outcome.value.members
        )
        assert (await aio.estimate(study)).accepted

    asyncio.run(run())


def test_async_estimate_reuses_measured_draws(request_args):
    from antecedent.exposure_measurement import PreparedMeasuredAnalysis, SoftExposure

    data = request_args["data"]
    request_args.update(
        data={"p": np.where(data["t"], 0.9, 0.1), "y": data["y"], "z": data["z"]},
        measurement=SoftExposure("t", "p", covariates=("z",)),
        measurement_draws=3,
        inference=ant.Frequentist(bootstrap=0),
    )

    async def run():
        study = await aio.prepare(**request_args)
        assert isinstance(study, PreparedMeasuredAnalysis)
        expected = ant.estimate(study)
        result = await aio.estimate(study)
        assert result.accepted and expected.accepted
        assert result.value.effect == expected.value.effect
        assert result.value.exposure is study.exposure

    asyncio.run(run())
