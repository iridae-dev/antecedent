"""Every population-scoped query declares its own ``target_population``.

The Rust engine scopes seven query kinds to a target population. Each Python
class of those kinds carries the field, and the Rust study builder decides
what it licenses: an ``AverageEffect`` estimates the treated, untreated,
predicate and custom-distribution targets its estimators support; every other
kind refuses a declared population with ``population_not_estimable`` rather
than estimating the all-observed population under the declared name.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any

import antecedent as ant
import numpy as np
import pytest
from antecedent import population as P
from antecedent.errors import CausalUnsupportedError


def _static(n: int = 1200, seed: int = 3) -> dict[str, np.ndarray]:
    """Binary treatment confounded by ``z`` whose effect grows with ``z``."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    t = (rng.uniform(size=n) < 1 / (1 + np.exp(-1.5 * z))).astype(float)
    m = 0.8 * t + 0.5 * z + rng.normal(size=n)
    w = rng.integers(0, 3, size=n).astype(float)
    y = t * (1.0 + 2.0 * z) + 0.7 * m + z + 0.5 * t * w + rng.normal(size=n)
    return {"z": z, "t": t, "m": m, "w": w, "y": y}


def _continuous(n: int = 1200, seed: int = 4) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    t = z + rng.normal(size=n)
    y = t * (1.0 + 0.5 * z) + z + rng.normal(size=n)
    return {"z": z, "t": t, "y": y}


def _series(n: int = 500, seed: int = 5) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = np.zeros(n)
    for i in range(1, n):
        y[i] = 0.9 * x[i - 1] + 0.3 * rng.normal()
    return {"x": x, "y": y}


STATIC_DAG = [("z", "t"), ("z", "y"), ("t", "y"), ("z", "m"), ("t", "m"), ("m", "y"), ("w", "y")]
CONTINUOUS_DAG = [("z", "t"), ("z", "y"), ("t", "y")]
TEMPORAL_DAG = [("x", 1, "y", 0)]


def _case(name: str, make, graph, query, **options: Any):
    return pytest.param(make, graph, query, options, id=name)


#: Every Python class whose Rust query kind is population-scoped, except
#: AverageEffect (licensed below), as a callable of the declared population.
SCOPED = [
    _case(
        "PulseEffect",
        _series,
        TEMPORAL_DAG,
        lambda p: ant.PulseEffect("x", "y", treatment_lag=1, target_population=p),
    ),
    _case(
        "SustainedEffect",
        _series,
        TEMPORAL_DAG,
        lambda p: ant.SustainedEffect("x", "y", treatment_lag=1, target_population=p),
    ),
    _case(
        "TemporalMediationEffect",
        _series,
        TEMPORAL_DAG,
        lambda p: ant.TemporalMediationEffect("x", "y", "y", target_population=p),
    ),
    _case(
        "MediationEffect",
        _static,
        STATIC_DAG,
        lambda p: ant.MediationEffect("t", "y", mediators=["m"], target_population=p),
    ),
    _case(
        "ConditionalEffect",
        _static,
        STATIC_DAG,
        lambda p: ant.ConditionalEffect("t", "y", "w", target_population=p),
    ),
    _case(
        "PathSpecificEffect",
        _static,
        STATIC_DAG,
        lambda p: ant.PathSpecificEffect("t", "y", path_nodes=["m"], target_population=p),
    ),
    _case(
        "InterventionalDistribution",
        _static,
        STATIC_DAG,
        lambda p: ant.InterventionalDistribution(
            "y", interventions={"t": 1.0}, target_population=p
        ),
    ),
    _case(
        "InterventionResponse",
        _static,
        STATIC_DAG,
        lambda p: ant.InterventionResponse(
            "y", intervention=ant.intervention.Set("t", 1.0), target_population=p
        ),
    ),
    _case(
        "ResponseCurve",
        _continuous,
        CONTINUOUS_DAG,
        lambda p: ant.ResponseCurve("t", "y", grid=[-1.0, 0.0, 1.0], target_population=p),
    ),
    _case(
        "AverageDerivative",
        _continuous,
        CONTINUOUS_DAG,
        lambda p: ant.AverageDerivative("t", "y", target_population=p),
    ),
    _case(
        "PointDerivative",
        _continuous,
        CONTINUOUS_DAG,
        lambda p: ant.PointDerivative("t", "y", at=0.5, target_population=p),
        estimator_config={"bandwidth": 0.8},
    ),
    _case(
        "Elasticity",
        _continuous,
        CONTINUOUS_DAG,
        lambda p: ant.Elasticity("t", "y", at=1.0, target_population=p),
        estimator_config={"bandwidth": 0.8},
    ),
    _case(
        "SemiElasticity",
        _continuous,
        CONTINUOUS_DAG,
        lambda p: ant.SemiElasticity("t", "y", at=0.5, log_scale="outcome", target_population=p),
        estimator_config={"bandwidth": 0.8},
    ),
    _case(
        "DirectionalDerivative",
        _continuous,
        CONTINUOUS_DAG,
        lambda p: ant.DirectionalDerivative(
            ["t"], ["y"], at=[0.5], direction=[1.0], target_population=p
        ),
    ),
    _case(
        "ResponseJacobian",
        _continuous,
        CONTINUOUS_DAG,
        lambda p: ant.ResponseJacobian(["t"], ["y"], at=[0.5], target_population=p),
    ),
]


def test_every_population_scoped_class_has_the_field() -> None:
    scoped = {
        ant.AverageEffect,
        ant.PulseEffect,
        ant.SustainedEffect,
        ant.TemporalMediationEffect,
        ant.MediationEffect,
        ant.ConditionalEffect,
        ant.PathSpecificEffect,
        ant.InterventionalDistribution,
        ant.InterventionResponse,
        ant.ResponseCurve,
        ant.AverageDerivative,
        ant.PointDerivative,
        ant.Elasticity,
        ant.SemiElasticity,
        ant.DirectionalDerivative,
        ant.ResponseJacobian,
    }
    unscoped = {
        ant.Counterfactual,
        ant.NestedCounterfactual,
        ant.AnomalyAttribution,
        ant.ChangeAttribution,
    }
    for cls in scoped:
        field = {f.name: f for f in fields(cls)}["target_population"]
        assert field.default is None and field.kw_only, cls.__name__
    for cls in unscoped:
        assert "target_population" not in {f.name for f in fields(cls)}, cls.__name__
    assert len(scoped) + len(unscoped) == len(
        [name for name in ant.query.__all__ if name in ant.__all__]
    )


@pytest.mark.parametrize(
    "population",
    [
        P.Treated(),
        P.Untreated(),
        P.Named("upper"),
        P.Rows((0, 2, 4, 6, 8)),
        P.CustomDistribution(7),
    ],
    ids=["treated", "untreated", "named", "rows", "custom"],
)
@pytest.mark.parametrize(("make", "graph", "query", "options"), SCOPED)
def test_an_unlicensed_population_refuses_by_code(make, graph, query, options, population) -> None:
    data = make()
    n = len(next(iter(data.values())))
    registry = P.PopulationRegistry()
    registry.insert_predicate("upper", list(range(n // 2, n)))
    registry.insert_distribution(7, [1.0] * n)
    with pytest.raises(CausalUnsupportedError) as caught:
        ant.analyze(
            data,
            graph=graph,
            query=query(population),
            population_registry=registry,
            refute=False,
            **options,
        )
    assert caught.value.reason_code == "population_not_estimable", str(caught.value)


@pytest.mark.parametrize(
    ("make", "graph", "query", "options"),
    [case for case in SCOPED if case.id in {"PulseEffect", "MediationEffect", "ResponseCurve"}],
)
def test_the_all_observed_population_is_the_default_and_survives_export(
    make, graph, query, options
) -> None:
    data = make()
    default = ant.analyze(data, graph=graph, query=query(None), refute=False, **options)
    declared = ant.analyze(data, graph=graph, query=query(P.AllRows()), refute=False, **options)
    assert declared.program_id == default.program_id
    assert declared.inspect().target_id == default.inspect().target_id
    loaded = ant.load(declared.export())
    assert loaded.acceptance.verified
    assert loaded.inspect().target_id == declared.inspect().target_id
    assert loaded.answer == declared.answer


def test_average_effect_population_changes_the_answer_and_the_target() -> None:
    data = _static()

    def run(population: object) -> Any:
        return ant.analyze(
            data,
            graph=STATIC_DAG,
            query=ant.AverageEffect("t", "y", target_population=population),
            estimator="aipw",
            refute=False,
        )

    ate, att = run(None), run(P.Treated())
    # The effect grows with z and the treated have higher z.
    assert att.ate > ate.ate + 0.3
    assert att.inspect().target_id != ate.inspect().target_id
    assert att.program_id != ate.program_id
    loaded = ant.load(att.export())
    assert loaded.acceptance.verified
    assert loaded.inspect().target_id == att.inspect().target_id
    assert loaded.inspect().to_dict()["target"]["query"]["target_population"] == "treated"


def test_linear_adjustment_refuses_a_population_on_a_class_graph() -> None:
    cpdag = ant.Cpdag.from_directed_undirected(
        ["z", "t", "y"], [("z", "y"), ("t", "y")], [("z", "t")]
    )
    data = {k: v for k, v in _static().items() if k in ("z", "t", "y")}
    ant.analyze(data, graph=cpdag, query=ant.AverageEffect("t", "y"), refute=False)
    with pytest.raises(CausalUnsupportedError) as caught:
        ant.analyze(
            data,
            graph=cpdag,
            query=ant.AverageEffect("t", "y", target_population=P.Treated()),
            refute=False,
        )
    assert caught.value.reason_code == "population_not_estimable"


def test_analyze_many_estimates_each_query_population() -> None:
    data = _static()
    queries = [
        ant.AverageEffect("t", "y"),
        ant.AverageEffect("t", "y", target_population=P.Treated()),
    ]
    ate, att = ant.estimation.analyze_many(
        data, graph=STATIC_DAG, queries=queries, estimator="aipw", refute=False
    )
    assert att.ate > ate.ate + 0.3
    single = ant.analyze(data, graph=STATIC_DAG, query=queries[1], estimator="aipw", refute=False)
    assert att.ate == pytest.approx(single.ate, rel=1e-6)
