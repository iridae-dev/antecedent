"""The five lines that are the Antecedent Python API, pinned on every major route.

```python
result = ant.analyze(data, graph=graph, query=ant.AverageEffect("treatment", "outcome"))
study = result.study
updated = study.refresh(new_data)
report = result.inspect().to_dict()
loaded = ant.load(result.export())
```

Each case runs exactly those five statements with every warning turned into an
error, then checks that the report is JSON, the loaded execution is verified,
and the live and loaded answers agree (kind, value or bounds, and identities).
A route that needs one more argument (``inference=``, ``discovery=``) passes it
through ``options``; the five statements themselves never change.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import antecedent as ant
import numpy as np
import pytest


def _static(seed: int, n: int = 1200) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    t = (rng.uniform(size=n) < 1 / (1 + np.exp(-z))).astype(float)
    y = 1.5 * t + z + rng.normal(size=n)
    return {"z": z, "treatment": t, "outcome": y}


def _continuous(seed: int, n: int = 1500) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    t = a + b + rng.normal(size=n)
    y = 1.5 * t + rng.normal(size=n)
    return {"a": a, "b": b, "treatment": t, "outcome": y}


def _temporal(seed: int, n: int = 400) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    pressure = rng.normal(size=n)
    defect = np.zeros(n)
    for i in range(1, n):
        defect[i] = 0.9 * pressure[i - 1] + 0.3 * rng.normal()
    return {"pressure": pressure, "defect": defect}


def _mediated(seed: int, n: int = 800) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    treatment = rng.normal(size=n)
    mediator = 0.8 * treatment + rng.normal(size=n)
    outcome = 0.5 * treatment + 1.2 * mediator + rng.normal(size=n)
    return {"treatment": treatment, "mediator": mediator, "outcome": outcome}


def _moderated(seed: int, n: int = 800) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    w = rng.integers(0, 3, size=n).astype(float)
    t = rng.binomial(1, 0.5, size=n).astype(float)
    y = 1.0 + 2.0 * t + 0.5 * t * w + rng.normal(scale=0.5, size=n)
    return {"treatment": t, "outcome": y, "w": w}


def _transport(seed: int, n: int = 600) -> dict[str, np.ndarray]:
    """Trial membership ``S | x ~ Bern(σ(x/2))`` with known probabilities, ``a`` randomized."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    s = 1 / (1 + np.exp(-0.5 * x))
    trial = rng.uniform(size=n) < s
    a = np.where(trial, (rng.uniform(size=n) < 0.5).astype(float), 0.0)
    y = np.where(trial, 1.0 + a * (1.0 + x) + x + rng.normal(size=n), 0.0)
    return {"a": a, "y": y, "trial": trial.astype(float), "s": s, "e": np.full(n, 0.5), "x": x}


UNITS = 60
#: A fixed ring network and one realized Bernoulli(1/2) assignment: the design.
RING = [((i + 1) % UNITS, i) for i in range(UNITS)] + [((i - 1) % UNITS, i) for i in range(UNITS)]
ASSIGNMENT = [bool(v) for v in np.random.default_rng(0).uniform(size=UNITS) < 0.5]


def _network_outcomes(seed: int) -> dict[str, np.ndarray]:
    """Outcomes of the fixed design; the data a refresh replaces."""
    rng = np.random.default_rng(seed)
    treated = np.array(ASSIGNMENT, dtype=float)
    neighbors = np.array([treated[(i + 1) % UNITS] + treated[i - 1] for i in range(UNITS)])
    return {"y": 1.0 + 2.0 * treated + 0.5 * neighbors + rng.normal(size=UNITS)}


STATIC_DAG = [("z", "treatment"), ("z", "outcome"), ("treatment", "outcome")]


@dataclass(frozen=True)
class Case:
    name: str
    make: Callable[[int], dict[str, np.ndarray]]
    graph: Any
    query: Any
    kind: str
    options: dict[str, Any] = field(default_factory=dict)


CASES = [
    Case(
        "dag-average-frequentist",
        _static,
        STATIC_DAG,
        ant.AverageEffect("treatment", "outcome"),
        "point",
    ),
    Case(
        "dag-average-bayesian",
        _static,
        STATIC_DAG,
        ant.AverageEffect("treatment", "outcome"),
        "point",
        {"inference": ant.Bayesian(n_draws=200)},
    ),
    Case(
        "cpdag-average-bounds",
        _static,
        ant.Cpdag.from_directed_undirected(
            ["z", "treatment", "outcome"],
            [("z", "outcome"), ("treatment", "outcome")],
            [("z", "treatment")],
        ),
        ant.AverageEffect("treatment", "outcome"),
        "bounds",
    ),
    Case(
        "discovery-pc-average",
        _continuous,
        None,
        ant.AverageEffect("treatment", "outcome"),
        "point",
        {"discovery": ant.discovery.PC(alpha=0.05)},
    ),
    Case(
        "temporal-pulse",
        _temporal,
        [("pressure", 1, "defect", 0)],
        ant.PulseEffect("pressure", "defect", treatment_lag=1, horizon_steps=1),
        "point",
    ),
    Case(
        "counterfactual", _static, STATIC_DAG, ant.Counterfactual("treatment", "outcome"), "point"
    ),
    Case(
        "mediation",
        _mediated,
        [("treatment", "mediator"), ("treatment", "outcome"), ("mediator", "outcome")],
        ant.MediationEffect("treatment", "outcome", mediators=["mediator"]),
        "point",
    ),
    Case(
        "conditional",
        _moderated,
        [("treatment", "outcome"), ("w", "outcome")],
        ant.ConditionalEffect("treatment", "outcome", "w"),
        "point",
    ),
    Case(
        "transport-trial-ipw",
        _transport,
        ant.Admg.from_edges(["a", "y", "trial", "s", "e", "x"], [("a", "y"), ("x", "y")]),
        ant.TransportQuery(
            ant.ResponseCurve("a", "y", grid=[0.0, 1.0]),
            ant.transport.SelectionDiagram("trial", "target", ["x"]),
            source_experiments=["a"],
            trial="trial",
            selection_probability="s",
            treatment_probability="e",
        ),
        "point",
    ),
    Case(
        "interference-neighbor-count",
        _network_outcomes,
        [],
        ant.InterferenceQuery(
            ant.interference.BernoulliAssignment(0.5),
            ant.interference.NeighborCount(),
            ant.interference.ExposureContrast(
                "y", ant.interference.ExposureLevel(0.0), ant.interference.ExposureLevel(1.0)
            ),
            network=RING,
            realized_assignment=ASSIGNMENT,
        ),
        "point",
    ),
    Case(
        "response-curve",
        _continuous,
        [("a", "treatment"), ("b", "treatment"), ("treatment", "outcome")],
        ant.ResponseCurve("treatment", "outcome", grid=[-1.0, 0.0, 1.0]),
        "response",
    ),
]


@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_golden_path(case: Case) -> None:
    data, new_data = case.make(1), case.make(2)
    graph, query, options = case.graph, case.query, case.options

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = ant.analyze(data, graph=graph, query=query, **options)
        study = result.study
        updated = study.refresh(new_data)
        report = result.inspect().to_dict()
        loaded = ant.load(result.export())

        # Refresh re-executed the same program on new data, of the same result type,
        # and the earlier result still exports its own execution (checked below
        # through the loaded snapshot identity).
        assert type(updated) is type(result)
        assert updated.program_id == result.program_id
        assert updated.data_snapshot_id != result.data_snapshot_id

        # The report is JSON and carries the same top-level fields as the loaded report.
        json.dumps(report, allow_nan=False)
        loaded_report = loaded.inspect().to_dict()
        json.dumps(loaded_report, allow_nan=False)
        assert set(report) == set(loaded_report)

        # Live and loaded answers agree.
        assert loaded.acceptance.verified
        assert result.answer.kind == case.kind
        assert loaded.answer == result.answer
        assert report["answer"] == loaded_report["answer"]
        assert loaded.program_id == result.program_id
        assert loaded.claim_id == result.claim_id
        assert loaded_report["data_snapshot_id"] == result.data_snapshot_id


# --- refusals on the same verbs ------------------------------------------------------


@pytest.mark.parametrize(
    "graph",
    [
        STATIC_DAG,
        ant.Cpdag.from_directed_undirected(
            ["z", "treatment", "outcome"],
            [("z", "outcome"), ("treatment", "outcome")],
            [("z", "treatment")],
        ),
    ],
    ids=["dag", "cpdag"],
)
def test_response_curve_on_a_binary_treatment_is_a_coded_refusal(graph: Any) -> None:
    """A local-quadratic dose response has three coefficients and a binary
    treatment has two support points, so the design is singular everywhere.
    That is refused by name, with the queries that do answer a binary treatment,
    rather than surfacing the backend's singular-matrix message."""
    with pytest.raises(ant.errors.CausalUnsupportedError) as caught:
        ant.analyze(
            _static(1),
            graph=graph,
            query=ant.ResponseCurve("treatment", "outcome", grid=[0.0, 1.0]),
        )
    assert caught.value.reason_code == "treatment_support_too_discrete"
    message = str(caught.value)
    assert "takes 2 distinct values" in message
    assert "AverageEffect" in message and "InterventionResponse" in message
    assert "backend error" not in message


def _autoregressive(seed: int, n: int = 1000) -> dict[str, np.ndarray]:
    """An autoregressive treatment confounded by ``z``; the lag-one pulse is 0.8.

    ``t[i] = 0.6 t[i-1] + 0.8 z[i-1] + noise`` and
    ``y[i] = 0.8 t[i-1] + 0.9 z[i-2] + noise``.
    """
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    t = np.zeros(n)
    y = np.zeros(n)
    for i in range(2, n):
        t[i] = 0.6 * t[i - 1] + 0.8 * z[i - 1] + 0.5 * rng.normal()
        y[i] = 0.8 * t[i - 1] + 0.9 * z[i - 2] + 0.5 * rng.normal()
    return {"t": t, "y": y, "z": z}


AUTOREGRESSIVE_DAG = [("t", 1, "t", 0), ("z", 1, "t", 0), ("z", 2, "y", 0), ("t", 1, "y", 0)]
PARENT_ADJUSTMENT = "temporal.parent_adjustment"


@pytest.mark.parametrize(
    "options",
    [
        {"graph": AUTOREGRESSIVE_DAG},
        {"discovery": ant.discovery.PCMCI(max_lag=2, alpha=0.01)},
    ],
    ids=["explicit", "pcmci"],
)
def test_autoregressive_treatment_pulse_is_identified_by_parent_adjustment(
    options: dict[str, Any],
) -> None:
    """An autoregressive treatment edge, which PCMCI finds on most real series,
    makes the treatment's ancestry unbounded, so unfolding cannot certify. The
    single-step pulse is identified by adjusting for the treatment's own parents
    ``{t[t-2], z[t-2]}``: the five lines recover the known effect, and the report
    and the loaded contract name the ``temporal.parent_adjustment`` derivation."""
    query = ant.PulseEffect("t", "y", treatment_lag=1, horizon_steps=1)
    if "discovery" in options:
        accepted = options["discovery"].accept(_autoregressive(1))
        assert set(accepted.graph.edges()) == set(AUTOREGRESSIVE_DAG)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = ant.analyze(_autoregressive(1), query=query, **options)
        study = result.study
        updated = study.refresh(_autoregressive(2))
        report = result.inspect().to_dict()
        loaded = ant.load(result.export())

    for answer in (result.answer, updated.answer):
        assert answer.kind == "point"
        assert answer.value == pytest.approx(0.8, abs=0.08)
    assert loaded.acceptance.verified
    assert loaded.answer == result.answer

    identification = report["identification"]["payload"]
    assert identification["status"] == "NonparametricallyIdentified"
    assert result.identification.adjustment_set == ["t", "z"]
    (case,) = report["assumptions"]["payload"]["certificate"]["cases"]
    assert [step["rule"] for step in case["identification"]["derivation"]][0] == PARENT_ADJUSTMENT
    assert [(c["name"], c["offset"]) for c in case["adjustment_coordinates"][0]] == [
        ("t", -2),
        ("z", -2),
    ]
    product = loaded.inspect().to_dict()["contract"]["identification_product"]
    assert product["derivation_rules"][0] == PARENT_ADJUSTMENT
    assert len(product["estimands"][0]["adjustment_set"]) == 2
    loaded_report = loaded.inspect().to_dict()
    assert loaded_report["identification_product_id"] == report["identification_product_id"]


def test_sustained_effect_on_an_autoregressive_treatment_still_refuses() -> None:
    """Parent adjustment identifies a single-step pulse only: a multi-step
    sustained window meets time-varying confounding through the treatment's own
    past, so the refusal still names the lagged cycle."""
    with pytest.raises(ant.errors.CausalIdentifyError) as caught:
        ant.analyze(
            _autoregressive(1),
            graph=AUTOREGRESSIVE_DAG,
            query=ant.SustainedEffect("t", "y", window=(-2, -1)),
        )
    message = str(caught.value)
    assert "lagged cycle" in message
    assert "max_history_lag" not in message
