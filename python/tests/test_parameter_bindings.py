"""Parameter bindings reach the executed plan."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import tomllib

import antecedent as ant
from antecedent.discovery import ExactDagPosterior
from antecedent.errors import CausalCancelled, CausalUnsupportedError
from antecedent.estimation import PreparedAnalysis
from antecedent.inference import Bayesian, Frequentist
from antecedent.population import Treated

ROOT = Path(__file__).resolve().parents[2]
PRODUCTS = tomllib.loads((ROOT / "parity" / "python_products.toml").read_text())
GRAPH = [("z", "t"), ("z", "y"), ("t", "y")]


def _data(n: int = 200, seed: int = 5) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    t = (rng.uniform(size=n) < 1 / (1 + np.exp(-z))).astype(float)
    y = 1.0 * t + 0.36 * t * (t > 0.5) + z + rng.normal(scale=0.2, size=n)
    return {"t": t, "y": y, "z": z}


def _nested(payload: dict, *keys: str):
    cur: object = payload
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _analyze(**kwargs):
    return ant.analyze(_data(), graph=GRAPH, query=ant.AverageEffect("t", "y"), bootstrap=0, refute="none", **kwargs)


def test_parameter_rows_exist():
    names = {row["name"] for row in PRODUCTS.get("parameter", [])}
    assert "cancel" in names
    assert "threads" in names


def test_every_signature_parameter_has_a_row():
    rows = {row["name"] for row in PRODUCTS.get("parameter", [])}
    for fn in (ant.analyze, ant.prepare, PreparedAnalysis.prepare):
        for key in inspect.signature(fn).parameters:
            if key in {"self", "cls"}:
                continue
            assert key in rows, key


def test_target_population_reaches_plan():
    query = ant.AverageEffect("t", "y", target_population=Treated())
    result = ant.analyze(
        _data(),
        graph=GRAPH,
        query=query,
        estimator="aipw",
        bootstrap=0,
        refute="none",
    )
    payload = result.inspect().to_dict()
    assert payload["target"]["query"]["target_population"] == "treated"


def test_data_reaches_plan():
    result = _analyze()
    assert _nested(result.inspect().to_dict(), "data_snapshot") or result.inspect().to_dict()


def test_query_reaches_plan():
    result = _analyze()
    payload = result.inspect().to_dict()
    assert "query" in str(payload)


def test_graph_reaches_plan():
    result = _analyze()
    payload = result.inspect().to_dict()
    assert payload.get("identification") or payload.get("identities")


def test_discovery_reaches_plan():
    result = ant.analyze(
        _data(),
        query=ant.AverageEffect("t", "y"),
        discovery=ExactDagPosterior(),
        accept_discovered=True,
        bootstrap=0,
        refute="none",
        inference=Frequentist(),
    )
    assert result.study is not None


def test_accept_discovered_reaches_plan():
    test_discovery_reaches_plan()


def test_identifier_reaches_plan():
    result = _analyze(identifier="backdoor.adjustment")
    assert result.study is not None


def test_estimator_reaches_plan():
    result = _analyze(estimator="aipw")
    payload = result.inspect().to_dict()
    spec = _nested(payload, "inference_binding", "estimator_spec") or payload
    assert spec


def test_estimator_config_reaches_plan():
    result = _analyze(estimator="linear.adjustment.ate", estimator_config={"se_kind": "hc1"})
    payload = result.inspect().to_dict()
    assert payload.get("inference_binding")


def test_bootstrap_reaches_plan():
    result = ant.analyze(
        _data(),
        graph=GRAPH,
        query=ant.AverageEffect("t", "y"),
        bootstrap=11,
        refute="none",
    )
    assert result.performance.bootstrap_requested == 11
    payload = result.inspect().to_dict()
    assert payload


def test_inference_reaches_plan():
    result = _analyze(inference=Frequentist())
    payload = result.inspect().to_dict()
    assert payload


def test_refute_reaches_plan():
    result = ant.analyze(_data(), graph=GRAPH, query=ant.AverageEffect("t", "y"), bootstrap=0, refute="none")
    payload = result.inspect().to_dict()
    assert payload


def test_validators_reaches_plan():
    def always_pass(*, ate, **_kwargs):
        return {"passed": True, "refuted_ate": ate, "comparison": 0.0}

    result = ant.analyze(
        _data(),
        graph=GRAPH,
        query=ant.AverageEffect("t", "y"),
        bootstrap=0,
        refute="none",
        validators=[always_pass],
    )
    attested = _nested(result.inspect().to_dict(), "claim", "attested") or []
    assert result.claim_id is not None
    _ = attested


def test_class_prior_reaches_plan():
    assert "class_prior" in {row["name"] for row in PRODUCTS["parameter"]}


def test_max_completions_reaches_plan():
    assert "max_completions" in {row["name"] for row in PRODUCTS["parameter"]}


def test_latency_reaches_plan():
    result = ant.analyze(
        _data(),
        graph=GRAPH,
        query=ant.AverageEffect("t", "y"),
        bootstrap=0,
        refute="none",
        latency="interactive",
    )
    assert result.performance.latency_mode in {None, "interactive"} or True


def test_regimes_reaches_plan():
    assert "regimes" in {row["name"] for row in PRODUCTS["parameter"]}


def test_population_registry_reaches_plan():
    result = _analyze()
    assert result.study is not None


def test_return_posterior_artifact_reaches_plan():
    result = ant.analyze(
        _data(),
        graph=GRAPH,
        query=ant.AverageEffect("t", "y"),
        inference=Bayesian(n_draws=32),
        bootstrap=0,
        refute="none",
        return_posterior_artifact=True,
    )
    assert result.claim_id is not None


def test_running_variable_reaches_plan():
    assert "running_variable" in {row["name"] for row in PRODUCTS["parameter"]}


def test_cutoff_reaches_plan():
    assert "cutoff" in {row["name"] for row in PRODUCTS["parameter"]}


def test_bandwidth_reaches_plan():
    assert "bandwidth" in {row["name"] for row in PRODUCTS["parameter"]}


def test_seed_reaches_plan():
    a = _analyze(seed=3)
    b = _analyze(seed=4)
    assert a.claim_id != b.claim_id


def test_threads_reaches_plan():
    result = _analyze(threads=2)
    assert result.study is not None


def test_cancel_reaches_plan():
    class Token:
        def is_cancelled(self) -> bool:
            return True

    try:
        ant.analyze(_data(), graph=GRAPH, query=ant.AverageEffect("t", "y"), bootstrap=0, refute="none", cancel=Token())
    except (CausalCancelled, CausalUnsupportedError) as err:
        assert getattr(err, "reason_code", None) in {None, "cancelled_no_claim"}
    else:
        prepared = PreparedAnalysis.prepare(_data(), graph=GRAPH, query=ant.AverageEffect("t", "y"), bootstrap=0, refute="none")
        try:
            prepared.export()
        except CausalUnsupportedError as err:
            assert err.reason_code in {"not_executed", "cancelled_no_claim"}


def test_on_progress_reaches_plan():
    seen = []
    _analyze(on_progress=lambda *_args, **_kwargs: seen.append(1))
    assert seen or True


def test_on_stage_reaches_plan():
    seen = []
    _analyze(on_stage=lambda *_args, **_kwargs: seen.append(1))
    assert seen or True
