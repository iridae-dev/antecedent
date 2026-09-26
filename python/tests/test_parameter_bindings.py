"""Every entry-point parameter reaches the executed plan, or the call raises.

Each ``binding = "contract"`` row of ``parity/python_products.toml`` names a
``contract_key``: a dotted path into the decoded contract of ``result.export()``.
Each row's test passes
a non-default value and reads it back through :func:`_bound`, which resolves the
row's own key, so a key that stops resolving or a value that stops binding fails
the test named for that parameter. ``execution_control`` rows observe the control
itself (callbacks called, cancellation honoured, draws retained).
"""

from __future__ import annotations

import inspect
import struct
import sys
import tomllib
import warnings
from pathlib import Path
from typing import Any

import antecedent as ant
import numpy as np
import pytest
from antecedent.discovery import PC, RPCMCI
from antecedent.errors import CausalCancelled, ReviewRequired
from antecedent.estimation import PreparedAnalysis
from antecedent.inference import Bayesian, Frequentist
from antecedent.population import CustomDistribution, PopulationRegistry, Treated

from _sealed_loads import assert_answer_kept

ROOT = Path(__file__).resolve().parents[2]
PRODUCTS = tomllib.loads((ROOT / "parity" / "python_products.toml").read_text(encoding="utf-8"))
ROWS = {row["name"]: row for row in PRODUCTS.get("parameter", [])}
GRAPH = [("z", "t"), ("z", "y"), ("t", "y")]
_ABSENT = object()


def _data(n: int = 200, seed: int = 5) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    t = (rng.uniform(size=n) < 1 / (1 + np.exp(-z))).astype(float)
    y = 1.0 * t + 0.36 * t * (t > 0.5) + z + rng.normal(scale=0.2, size=n)
    return {"t": t, "y": y, "z": z}


def _analyze(**kwargs: Any):
    kwargs.setdefault("graph", GRAPH)
    kwargs.setdefault("query", ant.AverageEffect("t", "y"))
    kwargs.setdefault("bootstrap", 0)
    kwargs.setdefault("refute", "none")
    data = kwargs.pop("data", None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ant.analyze(_data() if data is None else data, **kwargs)


def _contract(result: Any) -> dict[str, Any]:
    loaded = ant.load(result.export())
    assert_answer_kept(loaded)
    return loaded.artifact.contract


def _resolve(contract: dict[str, Any], key: str) -> Any:
    current: Any = contract
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return _ABSENT
        current = current[part]
    return current


def _bound(result: Any, name: str) -> Any:
    """The value at parameter ``name``'s ``contract_key`` in ``result``'s exported contract."""
    row = ROWS[name]
    assert row["binding"] == "contract", row
    value = _resolve(_contract(result), row["contract_key"])
    if value is _ABSENT:
        pytest.fail(f"{name}: contract_key {row['contract_key']!r} does not resolve")
    return value


def _bits(value: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", value))[0]


# --- the table itself ------------------------------------------------------------------


def test_parameter_rows_exist():
    assert {"cancel", "threads", "data", "query"} <= set(ROWS)


def test_every_signature_parameter_has_a_row():
    for fn in (ant.analyze, ant.prepare, PreparedAnalysis.prepare):
        for key in inspect.signature(fn).parameters:
            if key in {"self", "cls"}:
                continue
            assert key in ROWS, key


def test_every_row_names_a_test_here_that_reads_its_binding():
    module = sys.modules[__name__]
    for name, row in ROWS.items():
        path, _, test_name = row["test"].partition("::")
        assert path == "python/tests/test_parameter_bindings.py", row
        fn = getattr(module, test_name, None)
        assert callable(fn), f"{name}: {test_name} is not defined"
        if row["binding"] == "contract":
            source = inspect.getsource(fn)
            assert "_bound(" in source and f'"{name}"' in source, (
                f"{name}: {test_name} must read its value through _bound(result, {name!r})"
            )


# --- contract bindings --------------------------------------------------------------------


def test_data_reaches_plan():
    small = _analyze(data=_data(n=150))
    large = _analyze(data=_data(n=220, seed=6))
    assert _bound(small, "data")["row_count"] == 150
    assert _bound(large, "data")["row_count"] == 220
    assert small.inspect().data_snapshot_id != large.inspect().data_snapshot_id


def test_query_reaches_plan():
    forward = _bound(_analyze(), "query")
    assert forward["average_effect"]["treatment"] == 0
    assert forward["average_effect"]["outcome"] == 1
    reverse = _bound(
        _analyze(graph=[("z", "t"), ("z", "y"), ("y", "t")], query=ant.AverageEffect("y", "t")),
        "query",
    )
    assert reverse["average_effect"]["treatment"] == 1
    assert reverse["average_effect"]["outcome"] == 0
    # The query owns its target population; there is no second spelling on the entry points.
    treated = _bound(
        _analyze(query=ant.AverageEffect("t", "y", target_population=Treated()), estimator="aipw"),
        "query",
    )
    assert treated["average_effect"]["target_population"] == "treated"
    assert forward["average_effect"]["target_population"] == "all_observed"
    for fn in (ant.analyze, ant.prepare, PreparedAnalysis.prepare):
        assert "target_population" not in inspect.signature(fn).parameters


def test_graph_reaches_plan():
    graph = _bound(_analyze(), "graph")
    edges = {tuple(edge) for edge in graph["dag"]["edges"]}
    # schema order t=0, y=1, z=2
    assert edges == {(2, 0), (2, 1), (0, 1)}


def test_discovery_reaches_plan():
    data = _data(n=300)
    result = _analyze(data=data, graph=None, discovery=PC(), accept_discovered=True)
    assert _bound(result, "discovery") == PC().run(data).algorithm_id


def test_accept_discovered_reaches_plan():
    """`False` declines auto-acceptance: a review with anything pending refuses.

    PC's undirected marks are the CPDAG class, not pending review, so the flag
    can only be observed where the review artifact really has something to
    accept — FCI's circle marks do.
    """
    from antecedent.discovery import FCI

    data = _data(n=300)
    accepted = _analyze(data=data, graph=None, discovery=PC(), accept_discovered=True)
    assert _bound(accepted, "accept_discovered") == "accepted"
    with pytest.raises(ReviewRequired):
        _analyze(
            data=data,
            graph=None,
            discovery=FCI(alpha=0.2, fdr=False),
            accept_discovered=False,
        )


def test_identifier_reaches_plan():
    assert _bound(_analyze(identifier="backdoor.adjustment"), "identifier") == "backdoor.adjustment"


def test_estimator_reaches_plan():
    aipw = _bound(_analyze(estimator="aipw"), "estimator")
    linear = _bound(_analyze(estimator="linear.adjustment.ate"), "estimator")
    assert "aipw" in str(aipw)
    assert aipw != linear


def test_estimator_config_reaches_plan():
    hc1 = _analyze(estimator="linear.adjustment.ate", estimator_config={"se_kind": "hc1"})
    hc3 = _analyze(estimator="linear.adjustment.ate", estimator_config={"se_kind": "hc3"})
    assert _bound(hc1, "estimator_config") != _bound(hc3, "estimator_config")


def test_bootstrap_reaches_plan():
    assert _bound(_analyze(bootstrap=11), "bootstrap") == 11
    with_config = _analyze(
        bootstrap=11, estimator="linear.adjustment.ate", estimator_config={"se_kind": "hc1"}
    )
    assert _bound(with_config, "bootstrap") == 11
    assert with_config.performance.bootstrap_requested == 11


def test_inference_reaches_plan():
    assert _bound(_analyze(inference=Frequentist()), "inference") == "frequentist"
    assert _bound(_analyze(inference=Bayesian(n_draws=32)), "inference") == "bayesian"


def test_refute_reaches_plan():
    assert _bound(_analyze(refute="none"), "refute") is None
    assert _bound(_analyze(refute="placebo", bootstrap=11), "refute")


def test_validators_reaches_plan():
    def always_pass(*, ate, **_kwargs):
        return {"passed": True, "refuted_ate": ate, "comparison": 0.0}

    def always_fail(*, ate, **_kwargs):
        return {"passed": False, "refuted_ate": ate, "comparison": 1.0}

    passing = _analyze(validators=[always_pass])
    failing = _analyze(validators=[always_fail])
    attested = _bound(passing, "validators")
    assert attested, "a custom validator result must be attested in the claim"
    assert _bound(failing, "validators") != attested
    assert passing.claim_id != failing.claim_id != _analyze().claim_id


def _temporal_class(**kwargs: Any):
    from test_temporal_class_bayesian_envelope import _PIN, _cpdag, _pulse, _series

    kwargs.setdefault("inference", Bayesian(n_draws=32, backend="conjugate"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ant.analyze(
            _series(_PIN),
            graph=_cpdag(),
            query=_pulse(_PIN),
            refute=False,
            bootstrap=0,
            seed=7,
            **kwargs,
        )


def test_class_prior_reaches_plan():
    enumeration = _temporal_class()
    prior = _temporal_class(class_prior=ant.ClassPrior.from_ordered([0.9, 0.1]))
    assert prior.structural_weight_basis == "caller_supplied_class_prior"
    assert _bound(prior, "class_prior") != _bound(enumeration, "class_prior")


def test_max_completions_reaches_plan():
    prior = ant.ClassPrior.from_ordered([1.0])
    capped = _temporal_class(class_prior=prior, max_completions=1)
    full = _temporal_class(class_prior=ant.ClassPrior.from_ordered([0.5, 0.5]), max_completions=2)
    assert _bound(capped, "max_completions") != _bound(full, "max_completions")


def test_latency_reaches_plan():
    """A tier sets the omitted budgets; the bound inference differs between tiers."""
    interactive = _analyze(bootstrap=None, refute=None, latency="interactive")
    report = _analyze(bootstrap=None, refute=None, latency="report")
    assert interactive.performance.latency_mode == "interactive"
    assert (
        _bound(interactive, "latency")["bootstrap_replicates"]
        != _bound(report, "latency")["bootstrap_replicates"]
    )


def test_regimes_reaches_plan():
    from test_analyze_discovery import _lag1_series

    data = _lag1_series(n=160)
    regimes = [0] * len(data["x"])
    query = ant.PulseEffect("x", "y", treatment_lag=1, horizon_steps=1, active_level=1.0)
    config = RPCMCI(max_lag=1, alpha=0.2, fdr=False)
    result = _analyze(data=data, graph=None, discovery=config, regimes=regimes, query=query)
    assert config.run(data, regimes=regimes).n_regimes == 1
    assert _bound(result, "regimes") == config.algorithm_id


def test_population_registry_reaches_plan():
    data = _data(n=240)
    registry = PopulationRegistry()
    weights = [0.5] * 120 + [1.0] * 120
    registry.insert_distribution(7, weights)
    result = _analyze(
        data=data,
        query=ant.AverageEffect("t", "y", target_population=CustomDistribution(7)),
        population_registry=registry,
        estimator="propensity.weighting",
    )
    target = _bound(result, "population_registry")
    assert "7" in str(target["average_effect"]["target_population"])


def test_return_posterior_artifact_reaches_plan():
    kept = _analyze(inference=Bayesian(n_draws=32), return_posterior_artifact=True)
    summary = _analyze(inference=Bayesian(n_draws=32))
    assert kept.posterior is not None and kept.posterior.artifact
    assert summary.posterior is not None and summary.posterior.artifact is None


def test_running_variable_reaches_plan():
    data = _rd_data()
    assert _bound(_rd(data), "running_variable") == 0
    assert _bound(_rd(data, running_variable="r2"), "running_variable") == 3


def test_cutoff_reaches_plan():
    data = _rd_data()
    assert _bound(_rd(data, cutoff=0.25), "cutoff") == _bits(0.25)


def test_bandwidth_reaches_plan():
    data = _rd_data()
    assert _bound(_rd(data, bandwidth=0.8), "bandwidth") == _bits(0.8)
    assert _bound(_rd(data, bandwidth=1.2), "bandwidth") == _bits(1.2)


def _rd_data(n: int = 400) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(9)
    r = rng.normal(size=n)
    t = (r >= 0).astype(float)
    y = 0.5 * r + 2.0 * t + rng.normal(scale=0.3, size=n)
    return {"r": r, "t": t, "y": y, "r2": r + 0.01 * rng.normal(size=n)}


def _rd(data: dict[str, np.ndarray], **kwargs: Any):
    kwargs.setdefault("running_variable", "r")
    kwargs.setdefault("cutoff", 0.0)
    kwargs.setdefault("bandwidth", 1.2)
    rv = kwargs["running_variable"]
    # A sharp design's treatment column is that design's threshold rule.
    data = {**data, "t": (data[rv] >= kwargs["cutoff"]).astype(float)}
    return _analyze(
        data=data,
        graph=[(rv, "t"), (rv, "y"), ("t", "y")],
        estimator="rd.sharp",
        identifier="rd.sharp",
        **kwargs,
    )


def test_seed_reaches_plan():
    assert _bound(_analyze(seed=3), "seed") == 3
    assert _analyze(seed=3).claim_id != _analyze(seed=4).claim_id


def test_threads_reaches_plan():
    assert _bound(_analyze(threads=2), "threads") == 2


# --- execution controls -----------------------------------------------------------------------


def test_cancel_reaches_plan():
    from antecedent.state import CancellationToken

    token = CancellationToken()
    token.cancel()
    with pytest.raises(CausalCancelled):
        _analyze(bootstrap=199, cancel=token)


def test_on_progress_reaches_plan():
    seen: list[tuple[float, str]] = []
    _analyze(bootstrap=40, on_progress=lambda fraction, stage: seen.append((fraction, stage)))
    assert seen, "on_progress was never called"


def test_on_stage_reaches_plan():
    stages: list[str] = []
    _analyze(bootstrap=40, on_stage=lambda stage, _payload: stages.append(stage))
    assert stages[:2] == ["identify", "estimate_point"]


def _transport_query():
    from antecedent import transport

    graph = ant.Admg.from_edges(["x", "y"], [("x", "y")])
    evidence = transport.Evidence(
        source=transport.Source(
            "source", kind="experimental", interventions=["x"], sampling="independent"
        ),
        target_sampling="representative_sample",
    )
    data = transport.StatisticalTransportData(
        samples=(
            transport.RegimeSample(
                "source",
                "source",
                "v1",
                {"y": [0.0] * 50 + [1.0] * 50},
                interventions=(("x", 0.0),),
            ),
            transport.RegimeSample(
                "source",
                "source",
                "v1",
                {"y": [0.0] * 20 + [1.0] * 80},
                interventions=(("x", 1.0),),
            ),
        )
    )
    query = transport.Transport(ant.AverageEffect("x", "y"), target="target", evidence=evidence)
    return graph, data, query


def test_provider_reaches_plan():
    """Applies only to transport.Transport: prepare_transport rejects it elsewhere
    (see the reason on the ``provider`` row of parity/python_products.toml)."""
    from antecedent import transport

    graph, data, query = _transport_query()
    default = ant.analyze(data, query=query, graph=graph)
    learned = ant.analyze(data, query=query, graph=graph, provider=transport.LearnedCategorical())
    assert default.transport.provider == "empirical_table"
    assert learned.transport.provider == "learned_categorical"
    assert default.estimate.estimator_id != learned.estimate.estimator_id


def test_controls_reaches_plan():
    """Applies only to transport.Transport (see the ``controls`` row's reason)."""
    from antecedent import transport
    from antecedent.state import CancellationToken

    graph, data, query = _transport_query()
    token = CancellationToken()
    token.cancel()
    with pytest.raises(ant.errors.CausalCancelledError, match="cancelled"):
        ant.analyze(
            data, query=query, graph=graph, controls=transport.TransportControls(cancel=token)
        )
