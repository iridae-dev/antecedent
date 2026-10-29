"""Runtime behaviour of the canonical lifecycle: Ctrl-C, concurrency, callbacks, cancellation, leases.

Each test runs real native work. Interrupts are real SIGINT signals delivered to a child
interpreter; concurrency uses real threads over shared prepared branches; callback failures
must surface as the original exception object with its original traceback; cancellation must
return every admission lease so the caller's available native memory equals its starting value.
"""

from __future__ import annotations

import gc
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import antecedent as ant
import numpy as np
import pytest
from antecedent import nuisance_scores as ns
from antecedent.errors import (
    CausalCancelledError,
    CausalError,
    CausalResourceError,
)
from antecedent.execution import Execution, ResourceLimits
from antecedent.outcomes import require_success

GRAPH = [("t", "y"), ("z", "y"), ("z", "t")]


def _data(n: int = 2000, seed: int = 0, shift: float = 0.0):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    t = (z + rng.normal(size=n) > 0).astype(np.float64)
    y = 2.0 * t + z + shift + rng.normal(size=n)
    return {"t": t, "y": y, "z": z}


def _prepare(n: int = 2000, bootstrap: int = 0, seed: int = 0, execution=None, **kwargs):
    return ant.prepare(
        _data(n, seed),
        graph=GRAPH,
        query=ant.AverageEffect("t", "y"),
        inference=ant.Frequentist(bootstrap=bootstrap),
        refute="none",
        execution=execution,
        **kwargs,
    )


def _value(study) -> float:
    return float(ant.estimate(study).answer.value)


def _settle(execution: Execution) -> int:
    gc.collect()
    return execution._handle.available_memory_bytes


# --------------------------------------------------------------------------------------
# Real SIGINT
# --------------------------------------------------------------------------------------

_SIGINT_CHILD = textwrap.dedent(
    """
    import json, signal, sys, time
    import numpy as np
    import antecedent as ant

    signal.signal(signal.SIGINT, signal.default_int_handler)
    rng = np.random.default_rng(0)
    n = 200_000
    z = rng.normal(size=n)
    t = (z + rng.normal(size=n) > 0).astype(float)
    y = 2.0 * t + z + rng.normal(size=n)
    study = ant.prepare(
        {"t": t, "y": y, "z": z},
        graph=[("t", "y"), ("z", "y"), ("z", "t")],
        query=ant.AverageEffect("t", "y"),
        inference=ant.Frequentist(bootstrap=200_000),
        refute="none",
    )
    print("ready", flush=True)
    interrupted = False
    try:
        ant.estimate(study)
    except KeyboardInterrupt:
        interrupted = True
    del study
    small = ant.prepare(
        {"t": t[:500], "y": y[:500], "z": z[:500]},
        graph=[("t", "y"), ("z", "y"), ("z", "t")],
        query=ant.AverageEffect("t", "y"),
        inference=ant.Frequentist(bootstrap=0),
        refute="none",
    )
    value = float(ant.estimate(small).answer.value)
    print(json.dumps({"interrupted": interrupted, "fresh": value}), flush=True)
    """
)


@pytest.mark.skipif(sys.platform == "win32", reason="SIGINT delivery via os.kill is POSIX")
def test_sigint_interrupts_a_running_estimate_and_the_runtime_stays_usable(tmp_path):
    script = tmp_path / "child.py"
    script.write_text(_SIGINT_CHILD)
    child = subprocess.Popen(
        [sys.executable, "-I", str(script)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "ready"
        time.sleep(1.5)  # the estimate is running inside native code with the GIL released
        assert child.poll() is None, "the estimate finished before the interrupt was sent"
        sent = time.monotonic()
        child.send_signal(signal.SIGINT)
        stdout, stderr = child.communicate(timeout=60)
        promptness = time.monotonic() - sent
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
    assert child.returncode == 0, stderr
    report = __import__("json").loads(stdout.strip().splitlines()[-1])
    assert report["interrupted"] is True
    # The uninterrupted run needs tens of seconds; an interrupt ends it in a few.
    assert promptness < 20.0
    assert report["fresh"] == pytest.approx(2.0, abs=0.35)
    # A fresh execution in this process is unaffected by the child's interrupt.
    assert _value(_prepare()) == pytest.approx(2.0, abs=0.2)


# --------------------------------------------------------------------------------------
# Concurrent execution
# --------------------------------------------------------------------------------------


def test_threads_sharing_one_prepared_study_match_serial_runs_and_never_mutate_it():
    study = _prepare(n=3000, bootstrap=40)
    before = study.inspect().to_dict()
    serial = _value(study)
    shifts = [0.0, 1.0, 2.0, 3.0]
    serial_rebound = [_value(ant.rebind(study, data=_data(3000, 0, shift))) for shift in shifts]
    barrier = Barrier(8)

    def estimate_source(_index):
        barrier.wait(timeout=30)
        return _value(study)

    def rebind_then_estimate(shift):
        barrier.wait(timeout=30)
        branch = ant.rebind(study, data=_data(3000, 0, shift))
        return _value(branch)

    with ThreadPoolExecutor(max_workers=8) as pool:
        sources = [pool.submit(estimate_source, i) for i in range(4)]
        rebound = [pool.submit(rebind_then_estimate, shift) for shift in shifts]
        assert [future.result(timeout=120) for future in sources] == [serial] * 4
        assert [future.result(timeout=120) for future in rebound] == serial_rebound
    assert _value(study) == serial
    assert study.inspect().to_dict() == before


def test_branches_with_distinct_execution_seeds_match_serial_runs_concurrently():
    study = _prepare(n=3000, bootstrap=60)
    seeds = [3, 4, 5, 6]
    serial = [_value(ant.reconfigure(study, execution=Execution(seed=seed))) for seed in seeds]
    barrier = Barrier(len(seeds))

    def run(seed):
        branch = ant.reconfigure(study, execution=Execution(seed=seed))
        barrier.wait(timeout=30)
        outcome = ant.estimate(branch)
        interval = outcome.value.estimate
        return float(outcome.answer.value), repr(interval)

    with ThreadPoolExecutor(max_workers=len(seeds)) as pool:
        concurrent = [future.result(timeout=120) for future in [pool.submit(run, s) for s in seeds]]
    serial_full = []
    for seed in seeds:
        outcome = ant.estimate(ant.reconfigure(study, execution=Execution(seed=seed)))
        serial_full.append((float(outcome.answer.value), repr(outcome.value.estimate)))
    assert concurrent == serial_full
    assert [value for value, _ in concurrent] == serial
    # The source branch kept its own seed: it still equals a freshly prepared identical study.
    assert _value(study) == _value(_prepare(n=3000, bootstrap=60))


def test_threads_sharing_one_execution_and_memory_limit_never_exceed_it_and_release_leases():
    probe = Execution(limits=ResourceLimits(memory_bytes=50_000_000))
    held = _prepare(execution=probe)
    footprint = 50_000_000 - probe._handle.available_memory_bytes
    assert footprint > 0
    del held
    assert _settle(probe) == 50_000_000

    limit = int(footprint * 2.5)
    shared = Execution(limits=ResourceLimits(memory_bytes=limit))
    start = shared._handle.available_memory_bytes
    assert start <= limit
    stop = threading.Event()
    observed = []

    def watch():
        while not stop.is_set():
            observed.append(shared._handle.available_memory_bytes)

    barrier = Barrier(6)

    def run(_index):
        barrier.wait(timeout=30)
        try:
            study = _prepare(execution=shared)
            return _value(study)
        except CausalResourceError:
            return "resource_refused"

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    try:
        with ThreadPoolExecutor(max_workers=6) as pool:
            results = [
                future.result(timeout=120) for future in [pool.submit(run, i) for i in range(6)]
            ]
    finally:
        stop.set()
        watcher.join()
    accepted = [value for value in results if value != "resource_refused"]
    assert accepted, "a limit that holds two studies must admit at least one"
    assert all(value == pytest.approx(accepted[0]) for value in accepted)
    # Admission never over-grants: available memory stays within [0, limit] throughout.
    assert observed and max(observed) <= start
    assert _settle(shared) == start
    # And the same Execution still admits fresh work afterwards.
    assert _value(_prepare(execution=shared)) == pytest.approx(accepted[0])
    assert _settle(shared) == start


# --------------------------------------------------------------------------------------
# Original callback identity and traceback
# --------------------------------------------------------------------------------------


class _CallbackFailure(Exception):
    """A user exception type the library must hand back unchanged."""

    def __init__(self, label: str):
        super().__init__(label)
        self.label = label


def _frame_names(error: BaseException) -> list[str]:
    return [frame.name for frame in traceback.extract_tb(error.__traceback__)]


def _run_in_worker(operation):
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(operation).exception(timeout=120)


def test_progress_observer_failure_is_the_original_exception_inline_and_in_a_worker_thread():
    study = _prepare(n=3000, bootstrap=40)

    def observer_that_fails(_fraction, _stage):
        raise original

    original = _CallbackFailure("progress")
    branch = ant.reconfigure(study, execution=Execution(on_progress=observer_that_fails))
    with pytest.raises(_CallbackFailure) as raised:
        ant.estimate(branch)
    assert raised.value is original
    assert "observer_that_fails" in _frame_names(original)

    threaded = _CallbackFailure("progress in a worker")

    def threaded_observer(_fraction, _stage):
        raise threaded

    threaded_branch = ant.reconfigure(study, execution=Execution(on_progress=threaded_observer))
    caught = _run_in_worker(lambda: ant.estimate(threaded_branch))
    assert caught is threaded
    assert "threaded_observer" in _frame_names(threaded)
    assert "<lambda>" in _frame_names(threaded)  # the worker's own frame is retained too
    assert _value(study) == _value(_prepare(n=3000, bootstrap=40))


def test_validator_failure_through_estimate_is_the_original_exception_in_any_thread():
    original = _CallbackFailure("validator")

    def validator_that_fails(**_kwargs):
        raise original

    def estimate():
        study = ant.prepare(
            _data(),
            graph=GRAPH,
            query=ant.AverageEffect("t", "y"),
            inference=ant.Frequentist(bootstrap=0),
            refute="none",
            validators=[validator_that_fails],
        )
        return ant.estimate(study)

    with pytest.raises(_CallbackFailure) as raised:
        estimate()
    assert raised.value is original
    assert "validator_that_fails" in _frame_names(original)
    assert _run_in_worker(estimate) is original
    # The same module still answers once the failing validator is gone.
    assert _value(_prepare()) == pytest.approx(2.0, abs=0.2)


def test_discovery_ci_callback_failure_keeps_the_original_exception_as_its_cause():
    rng = np.random.default_rng(1)
    data = {name: rng.normal(size=150) for name in ("a", "b", "c", "d")}

    def make():
        original = _CallbackFailure("ci test")

        def ci_that_fails(_columns, _queries):
            raise original

        return original, ci_that_fails

    original, ci = make()
    with pytest.raises(CausalError) as raised:
        ant.discovery.discover(data, method=ant.discovery.PC(ci=ci, fdr=False))
    assert raised.value.__cause__ is original
    assert "ci_that_fails" in _frame_names(original)

    other, other_ci = make()
    caught = _run_in_worker(
        lambda: ant.discovery.discover(data, method=ant.discovery.PC(ci=other_ci, fdr=False))
    )
    assert isinstance(caught, CausalError) and caught.__cause__ is other


def test_a_provider_callback_failure_is_the_original_exception_through_estimate_in_a_worker(
    monkeypatch,
):
    from antecedent.extensibility import (
        CausalProviderSpec,
        ProviderQuery,
        providers,
    )

    for name in ("_providers", "_verification", "_entry_points"):
        monkeypatch.setattr(providers, name, {})
    original = _CallbackFailure("provider")

    class Provider:
        spec = CausalProviderSpec(
            query_family="runtime_behaviour_mean",
            identification_requirements=("caller identified",),
            observed_distributions=("x",),
            nuisance_functions=(),
            support_conditions=("caller asserted",),
            data_dependence=("supplied rows",),
            inference_claims=("point_only",),
            influence_function=None,
            fold_policy="provider_owned",
            output_shape=(),
            uncertainty_semantics="point_only",
            artifact_codec="opaque",
            deterministic=True,
            provenance={"package": "runtime-behaviour-test"},
        )

        def execute(self, request):
            raise original

    providers.register("runtime-behaviour", Provider())
    study = ant.prepare(
        {"x": [1.0, 3.0]}, query=ProviderQuery("runtime-behaviour", {"offsets": [0]})
    )
    with pytest.raises(_CallbackFailure) as raised:
        ant.estimate(study)
    assert raised.value is original
    assert "execute" in _frame_names(original)
    assert _run_in_worker(lambda: ant.estimate(study)) is original


# --------------------------------------------------------------------------------------
# Cancellation and lease cleanup
# --------------------------------------------------------------------------------------

_LIMIT = 200_000_000


def _cancellable(limit: int = _LIMIT):
    token = ant.CancellationToken()
    return token, Execution(cancel=token, limits=ResourceLimits(memory_bytes=limit))


def _estimate_cancelled_midway(study, requested: int):
    """Run ``study`` whose token trips mid-run: typed cancellation or a flagged partial bootstrap.

    The interactive contract lets a bootstrap stopped by its token return the replicates it
    finished, marked ``performance.cancelled``; every other stage raises the typed error.
    Either way the run must not report the full replicate count it never evaluated.
    """
    try:
        outcome = ant.estimate(study)
    except CausalCancelledError:
        return None
    result = require_success(outcome)
    assert result.performance.cancelled
    assert (result.performance.bootstrap_replicates_ok or 0) < requested
    return result


def test_estimate_and_rebind_under_a_precancelled_token_are_cancelled_and_return_leases():
    token, execution = _cancellable()
    start = execution._handle.available_memory_bytes
    study = _prepare(execution=execution)
    assert execution._handle.available_memory_bytes < start
    token.cancel()
    with pytest.raises(CausalCancelledError):
        ant.estimate(study)
    with pytest.raises(CausalCancelledError):
        ant.rebind(study, data=_data(seed=3))
    del study
    assert _settle(execution) == start


def test_a_token_cancelled_during_a_long_estimate_stops_it_and_returns_leases():
    token, execution = _cancellable(2_000_000_000)
    start = execution._handle.available_memory_bytes
    study = _prepare(n=100_000, bootstrap=100_000, execution=execution)
    timer = threading.Timer(1.0, token.cancel)
    began = time.monotonic()
    timer.start()
    try:
        _estimate_cancelled_midway(study, 100_000)
    finally:
        timer.cancel()
    assert time.monotonic() - began < 30.0  # an uncancelled run takes several times longer
    del study
    assert _settle(execution) == start
    # Cancellation is scoped to that token; the process still computes.
    assert _value(_prepare()) == pytest.approx(2.0, abs=0.2)


def test_a_token_cancelled_during_a_long_rebound_estimate_returns_the_branch_leases():
    token, execution = _cancellable(2_000_000_000)
    start = execution._handle.available_memory_bytes
    study = _prepare(n=100_000, bootstrap=100_000, execution=execution)
    branch = ant.rebind(study, data=_data(100_000, 5, 1.0))
    timer = threading.Timer(1.0, token.cancel)
    timer.start()
    try:
        _estimate_cancelled_midway(branch, 100_000)
    finally:
        timer.cancel()
    del study, branch
    assert _settle(execution) == start


def test_loading_under_a_cancelled_token_is_cancelled_and_leaves_the_leases_alone():
    artifact = ant.estimate(_prepare()).export()
    token, execution = _cancellable()
    start = execution._handle.available_memory_bytes
    token.cancel()
    with pytest.raises(CausalCancelledError):
        ant.load(artifact, execution=execution)
    with pytest.raises(CausalCancelledError):
        ant.load(bytes(artifact), execution=execution)
    assert _settle(execution) == start
    _, fresh = _cancellable()
    assert ant.load(artifact, execution=fresh).accepted
    assert _settle(fresh) == fresh._handle.available_memory_bytes


def _scm_for_batch(n: int = 400, seed: int = 9):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    t1 = (z + rng.normal(size=n) > 0).astype(np.float64)
    t2 = (z + rng.normal(size=n) > 0.2).astype(np.float64)
    y = 2.0 * t1 + 1.5 * t2 + z + 0.4 * rng.normal(size=n)
    data = {"t1": t1, "t2": t2, "y": y, "z": z}
    edges = [("z", "t1"), ("z", "t2"), ("z", "y"), ("t1", "y"), ("t2", "y")]
    return data, edges


def test_a_cancelled_batch_returns_its_leases_and_the_same_inputs_then_succeed():
    data, edges = _scm_for_batch()
    queries = [ant.AverageEffect("t1", "y"), ant.AverageEffect("t2", "y")]
    token, execution = _cancellable()
    start = execution._handle.available_memory_bytes
    token.cancel()
    with pytest.raises(CausalCancelledError):
        ant.estimation.analyze_many(
            data, graph=edges, queries=queries, refute=False, execution=execution
        )
    assert _settle(execution) == start

    token, execution = _cancellable()
    start = execution._handle.available_memory_bytes
    batch = ant.estimation.analyze_many(
        data, graph=edges, queries=queries, refute=False, execution=execution
    )
    assert batch.accepted
    token.cancel()
    with pytest.raises(CausalCancelledError):
        ant.estimate(batch.study) if batch.study is not None else ant.rebind(batch, data=data)
    del batch
    assert _settle(execution) == start

    _, fresh = _cancellable()
    ok = ant.estimation.analyze_many(
        data, graph=edges, queries=queries, refute=False, execution=fresh
    )
    assert ok.accepted
    del ok
    assert _settle(fresh) == fresh._handle.available_memory_bytes


def _nuisance_study(execution):
    return ant.prepare(
        {
            "z": np.r_[np.zeros(100), np.ones(100)],
            "t": np.r_[np.ones(30), np.zeros(70), np.ones(60), np.zeros(40)],
            "y": np.arange(200, dtype=float) % 7,
        },
        graph=[("z", "t"), ("z", "y"), ("t", "y")],
        query=ant.AverageEffect("t", "y"),
        inference=ant.Frequentist(bootstrap=0),
        refute=False,
        execution=execution,
    )


def test_nuisance_fit_export_and_load_under_cancellation_return_their_leases():
    study = _nuisance_study(Execution(seed=13, threads=1))
    product = require_success(ns.treatment_score_posterior(study, draws=16))
    artifact = product.export()
    token = ant.CancellationToken()
    execution = Execution(cancel=token, limits=ResourceLimits(memory_bytes=_LIMIT))
    start = execution._handle.available_memory_bytes
    token.cancel()
    with pytest.raises(CausalCancelledError):
        ns.treatment_score_posterior(study, draws=16, execution=execution)
    assert _settle(execution) == start
    with pytest.raises(CausalCancelledError):
        product.export(execution=execution)
    assert _settle(execution) == start
    with pytest.raises(CausalCancelledError):
        ant.load(artifact, execution=execution)
    assert _settle(execution) == start
    # Cancelled-token state never leaks into an unrelated caller.
    fresh = Execution(limits=ResourceLimits(memory_bytes=_LIMIT))
    before = fresh._handle.available_memory_bytes
    refit = require_success(ns.treatment_score_posterior(study, draws=16, execution=fresh))
    assert refit.target == product.target
    del refit
    assert _settle(fresh) == before


def test_a_token_cancelled_mid_nuisance_fit_returns_its_leases():
    study = _nuisance_study(Execution(seed=13, threads=1))
    token = ant.CancellationToken()

    def observer(fraction, _stage):
        if fraction > 0.0:
            token.cancel()

    cancelling = Execution(
        cancel=token, limits=ResourceLimits(memory_bytes=_LIMIT), on_progress=observer
    )
    start = cancelling._handle.available_memory_bytes
    with pytest.raises(CausalCancelledError):
        ns.treatment_score_posterior(study, draws=4000, execution=cancelling)
    assert _settle(cancelling) == start
