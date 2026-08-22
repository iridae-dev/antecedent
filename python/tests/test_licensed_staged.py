"""Every licensed cell must run identify → prepare → estimate, not analyze-only."""

from __future__ import annotations

import json
import math
import pathlib

import numpy as np
import pytest

pytest.importorskip("antecedent")
import antecedent


def _confounded(n: int = 240, seed: int = 19):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    t = (rng.random(n) < 1.0 / (1.0 + np.exp(-(-0.4 + 0.9 * z)))).astype(np.float64)
    y = 2.0 * t + z + rng.normal(scale=0.4, size=n)
    return {"t": t, "y": y, "z": z}


_ATE_DATA = _confounded()
_EDGES = [("z", "t"), ("z", "y"), ("t", "y")]
_DAG = antecedent.Dag.from_edges(["z", "t", "y"], _EDGES)
_ACCEPTED = antecedent.AcceptedGraph.from_graph(_DAG, algorithm_id="hand")
_ATE = antecedent.AverageEffect(treatment="t", outcome="y")
_PAG = antecedent.Pag.from_marked_edges(
    ["t", "y", "z"],
    [
        ("z", "t", "tail", "arrow"),
        ("z", "y", "tail", "arrow"),
        ("t", "y", "tail", "arrow"),
    ],
)
_PAG_ACCEPTED = antecedent.AcceptedGraph.from_graph(_PAG, algorithm_id="hand")


def _frontdoor_table(n: int = 300):
    u = np.array([1.0 if (i % 5) < 2 else 0.0 for i in range(n)])
    t = np.array([1.0 if (i % 3) == 0 else 0.0 for i in range(n)])
    m = np.array([float(int(ti + ui) % 2) for ti, ui in zip(t, u, strict=True)])
    y = np.array([float(int(mi + ui) % 2) for mi, ui in zip(m, u, strict=True)])
    return {"t": t, "m": m, "y": y}


_ADMG_DATA = _frontdoor_table()
_ADMG = antecedent.Admg.from_edges(
    ["t", "m", "y"], [("t", "m"), ("m", "y")], bidirected=[("t", "y")]
)
_ADMG_ACCEPTED = antecedent.AcceptedGraph.from_graph(_ADMG, algorithm_id="hand")
_ADMG_ATE = antecedent.AverageEffect(treatment="t", outcome="y")


def _curve_table(n: int = 400, seed: int = 19):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    t = 0.7 * z + rng.normal(size=n)
    y = 2.0 * t + z + rng.normal(scale=0.2, size=n)
    return {"t": t, "y": y, "z": z}


_CURVE_DATA = _curve_table()
_CURVE = antecedent.ResponseCurve("t", "y", grid=[-0.5, 0.0, 0.5])


_BAYES = antecedent.Bayesian(backend="conjugate", n_draws=64)


# -- ConditionalEffect / PathSpecificEffect / InterventionalDistribution fixtures --------
#
# Small deterministic tables mirroring the Rust prepared_analysis.rs fixtures
# (conditional_effect_fixture / path_specific_fixture / distribution_fixture in
# crates/antecedent/tests/prepared_analysis.rs): a `t -> y <- w` interaction table, a
# discrete `t -> m -> y` chain (no direct edge), and a confounded `z -> t -> y <- z` table.


def _conditional_table(n: int = 120):
    t = np.array([0.0 if i % 2 == 0 else 1.0 for i in range(n)])
    w = np.array([float(i % 5) for i in range(n)])
    y = 1.0 + 2.0 * t + 0.5 * t * w
    return {"t": t, "y": y, "w": w}


_CONDITIONAL_DATA = _conditional_table()
_CONDITIONAL_EDGES = [("t", "y"), ("w", "y")]
_CONDITIONAL_DAG = antecedent.Dag.from_edges(["t", "y", "w"], _CONDITIONAL_EDGES)
_CONDITIONAL_ACCEPTED = antecedent.AcceptedGraph.from_graph(_CONDITIONAL_DAG, algorithm_id="hand")
_CONDITIONAL = antecedent.ConditionalEffect(treatment="t", outcome="y", modifier="w")


def _path_specific_table():
    t_vals: list[float] = []
    m_vals: list[float] = []
    y_vals: list[float] = []
    for t in (0.0, 1.0):
        for _ in range(50):
            t_vals.append(t)
            m_vals.append(t)
            y_vals.append(t)
    return {"t": np.array(t_vals), "m": np.array(m_vals), "y": np.array(y_vals)}


_PATH_DATA = _path_specific_table()
_PATH_EDGES = [("t", "m"), ("m", "y")]
_PATH_DAG = antecedent.Dag.from_edges(["t", "m", "y"], _PATH_EDGES)
_PATH_SPECIFIC = antecedent.PathSpecificEffect(treatment="t", outcome="y", path_nodes=["m"])


def _distribution_table():
    combos = [
        (0.0, 0.0, 0.0, 21),
        (0.0, 0.0, 1.0, 9),
        (0.0, 1.0, 0.0, 4),
        (0.0, 1.0, 1.0, 16),
        (1.0, 0.0, 0.0, 12),
        (1.0, 0.0, 1.0, 3),
        (1.0, 1.0, 0.0, 14),
        (1.0, 1.0, 1.0, 21),
    ]
    t_vals: list[float] = []
    y_vals: list[float] = []
    z_vals: list[float] = []
    for z, t, y, count in combos:
        for _ in range(count):
            z_vals.append(z)
            t_vals.append(t)
            y_vals.append(y)
    return {"t": np.array(t_vals), "y": np.array(y_vals), "z": np.array(z_vals)}


_DISTRIBUTION_DATA = _distribution_table()
_DISTRIBUTION_EDGES = [("z", "t"), ("z", "y"), ("t", "y")]
_DISTRIBUTION_DAG = antecedent.Dag.from_edges(["t", "y", "z"], _DISTRIBUTION_EDGES)
_DISTRIBUTION = antecedent.InterventionalDistribution(outcome="y", interventions={"t": 1.0})


# -- InterventionResponse fixture ------------------------------------------------------
#
# Same deterministic generator as the Rust known-truth pin
# (conformance/response/intervention_response/expected.json /
# crates/antecedent/tests/response_facade.rs::intervention_response_conforms_to_known_truth_fixture):
# zero-noise linear structural mean, hard do(t := 0.25).


def _intervention_response_table(n: int = 240):
    z = np.array([math.sin(i / 17.0) for i in range(n)])
    t = z + np.array([math.cos(i / 11.0) for i in range(n)])
    y = 1.0 + 2.0 * t + 0.8 * z
    return {"t": t, "y": y, "z": z}


_IR_DATA = _intervention_response_table()
_IR_EDGES = [("z", "t"), ("z", "y"), ("t", "y")]
_IR_DAG = antecedent.Dag.from_edges(["t", "y", "z"], _IR_EDGES)
_IR_ACCEPTED = antecedent.AcceptedGraph.from_graph(_IR_DAG, algorithm_id="hand")
_IR = antecedent.InterventionResponse(
    outcome="y", intervention=antecedent.intervention.Set("t", 0.25)
)

# `_intervention_response_table` is the exact same deterministic generator as
# `conformance/response/intervention_response/expected.json` (see comment above), so
# unlike most fixtures in this file this coordinate has a real known-truth value to
# compare against, not just self-consistency between `analyze()` and `PreparedAnalysis`.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_IR_FIXTURE = json.loads(
    (
        _REPO_ROOT / "conformance" / "response" / "intervention_response" / "expected.json"
    ).read_text()
)
_IR_TRUE_RESPONSE = _IR_FIXTURE["contract"]["true_response"]
_IR_TOLERANCE = _IR_FIXTURE["tolerance"]["truth_absolute"]


def _pulse_table(n: int = 80):
    t = np.array([math.sin(i / 17.0) for i in range(n)])
    y = np.zeros(n)
    for i in range(1, n):
        y[i] = 0.5 * t[i - 1]
    return {"t": t, "y": y}


_PULSE_DATA = _pulse_table()
_PULSE_EDGES = [("t", 1, "y", 0)]
_PULSE_TDAG = antecedent.TemporalDag.from_lagged_edges(["t", "y"], _PULSE_EDGES)
_PULSE_ACCEPTED = antecedent.AcceptedGraph.from_graph(_PULSE_TDAG, algorithm_id="hand")
_PULSE = antecedent.PulseEffect(treatment="t", outcome="y", treatment_lag=1, horizon_steps=1)
_SUSTAINED = antecedent.SustainedEffect(
    treatment="t", outcome="y", treatment_lag=1, horizon_steps=1
)
# Same TemporalDag as `_PULSE_ACCEPTED`; Sustained shares the graph, only the query differs.
_SUSTAINED_ACCEPTED = _PULSE_ACCEPTED

# `y_i = 0.5 * t_(i-1)`, zero noise: the structural pulse/single-step-sustained
# contrast at treatment_lag=1 is exactly 0.5. Known truth, not a fixture file.
_PULSE_TRUE_ATE = 0.5


def _mediation_table(n: int = 80):
    t = np.zeros(n)
    m = np.zeros(n)
    y = np.zeros(n)
    for i in range(1, n):
        t[i] = 0.3 * t[i - 1] + 0.1 * math.sin(i)
        m[i] = 0.8 * t[i - 1] + 0.05 * math.cos(i)
        y[i] = 0.5 * m[i] + 0.02 * math.sin(i)
    return {"t": t, "m": m, "y": y}


_MED_DATA = _mediation_table()
_MED_EDGES = [("t", 1, "t", 0), ("t", 1, "m", 0), ("m", 0, "y", 0)]
_MED_TDAG = antecedent.TemporalDag.from_lagged_edges(["t", "m", "y"], _MED_EDGES)
_MED_ACCEPTED = antecedent.AcceptedGraph.from_graph(_MED_TDAG, algorithm_id="hand")
_MED = antecedent.TemporalMediationEffect("t", "m", "y", contrast="mediated")


@pytest.mark.parametrize(
    "data, graph, query, refute, inference",
    [
        (_ATE_DATA, _DAG, _ATE, False, None),
        (_ATE_DATA, _DAG, _ATE, "cheap", None),
        (_ATE_DATA, _DAG, _ATE, "full", None),
        (_ATE_DATA, _ACCEPTED, _ATE, False, None),
        (_ATE_DATA, _ACCEPTED, _ATE, "cheap", None),
        (_ATE_DATA, _ACCEPTED, _ATE, "full", None),
        (_ATE_DATA, _DAG, _ATE, False, _BAYES),
        (_ATE_DATA, _DAG, _ATE, "cheap", _BAYES),
        (_ATE_DATA, _DAG, _ATE, "full", _BAYES),
        (_ATE_DATA, _ACCEPTED, _ATE, False, _BAYES),
        (_ATE_DATA, _ACCEPTED, _ATE, "cheap", _BAYES),
        (_ATE_DATA, _ACCEPTED, _ATE, "full", _BAYES),
        (_ATE_DATA, _PAG, _ATE, False, None),
        (_ATE_DATA, _PAG, _ATE, "cheap", None),
        (_ATE_DATA, _PAG, _ATE, "full", None),
        (_ATE_DATA, _PAG_ACCEPTED, _ATE, False, None),
        (_ATE_DATA, _PAG_ACCEPTED, _ATE, "cheap", None),
        (_ATE_DATA, _PAG_ACCEPTED, _ATE, "full", None),
        (_ATE_DATA, _PAG, _ATE, False, _BAYES),
        (_ATE_DATA, _PAG, _ATE, "cheap", _BAYES),
        (_ATE_DATA, _PAG, _ATE, "full", _BAYES),
        (_ATE_DATA, _PAG_ACCEPTED, _ATE, False, _BAYES),
        (_ATE_DATA, _PAG_ACCEPTED, _ATE, "cheap", _BAYES),
        (_ATE_DATA, _PAG_ACCEPTED, _ATE, "full", _BAYES),
        (_ADMG_DATA, _ADMG, _ADMG_ATE, False, None),
        (_ADMG_DATA, _ADMG, _ADMG_ATE, "cheap", None),
        (_ADMG_DATA, _ADMG, _ADMG_ATE, "full", None),
        (_ADMG_DATA, _ADMG_ACCEPTED, _ADMG_ATE, False, None),
        (_ADMG_DATA, _ADMG_ACCEPTED, _ADMG_ATE, "cheap", None),
        (_ADMG_DATA, _ADMG_ACCEPTED, _ADMG_ATE, "full", None),
        (_CURVE_DATA, _EDGES, _CURVE, False, None),
        (_CURVE_DATA, _ACCEPTED, _CURVE, False, None),
        (_CONDITIONAL_DATA, _CONDITIONAL_DAG, _CONDITIONAL, False, None),
        (_CONDITIONAL_DATA, _CONDITIONAL_DAG, _CONDITIONAL, "cheap", None),
        (_CONDITIONAL_DATA, _CONDITIONAL_DAG, _CONDITIONAL, "full", None),
        (_CONDITIONAL_DATA, _CONDITIONAL_ACCEPTED, _CONDITIONAL, False, None),
        (_CONDITIONAL_DATA, _CONDITIONAL_ACCEPTED, _CONDITIONAL, "cheap", None),
        (_CONDITIONAL_DATA, _CONDITIONAL_ACCEPTED, _CONDITIONAL, "full", None),
        (_PATH_DATA, _PATH_DAG, _PATH_SPECIFIC, False, None),
        (_DISTRIBUTION_DATA, _DISTRIBUTION_DAG, _DISTRIBUTION, False, None),
        (_IR_DATA, _IR_DAG, _IR, False, None),
        (_IR_DATA, _IR_ACCEPTED, _IR, False, None),
        (_PULSE_DATA, _PULSE_TDAG, _PULSE, False, None),
        (_PULSE_DATA, _PULSE_ACCEPTED, _PULSE, False, None),
        (_PULSE_DATA, _PULSE_TDAG, _PULSE, "cheap", None),
        (_PULSE_DATA, _PULSE_ACCEPTED, _PULSE, "cheap", None),
        (_PULSE_DATA, _PULSE_TDAG, _PULSE, False, _BAYES),
        (_PULSE_DATA, _PULSE_TDAG, _SUSTAINED, False, None),
        (_PULSE_DATA, _SUSTAINED_ACCEPTED, _SUSTAINED, False, None),
        (_PULSE_DATA, _PULSE_TDAG, _SUSTAINED, "cheap", None),
        (_PULSE_DATA, _SUSTAINED_ACCEPTED, _SUSTAINED, "cheap", None),
        (_PULSE_DATA, _PULSE_TDAG, _SUSTAINED, False, _BAYES),
        (_MED_DATA, _MED_TDAG, _MED, False, None),
        (_MED_DATA, _MED_ACCEPTED, _MED, False, None),
    ],
    ids=[
        "ate_explicit_none",
        "ate_explicit_cheap",
        "ate_explicit_full",
        "ate_accepted_none",
        "ate_accepted_cheap",
        "ate_accepted_full",
        "ate_explicit_bayesian_none",
        "ate_explicit_bayesian_cheap",
        "ate_explicit_bayesian_full",
        "ate_accepted_bayesian_none",
        "ate_accepted_bayesian_cheap",
        "ate_accepted_bayesian_full",
        "ate_pag_explicit_none",
        "ate_pag_explicit_cheap",
        "ate_pag_explicit_full",
        "ate_pag_accepted_none",
        "ate_pag_accepted_cheap",
        "ate_pag_accepted_full",
        "ate_pag_explicit_bayesian_none",
        "ate_pag_explicit_bayesian_cheap",
        "ate_pag_explicit_bayesian_full",
        "ate_pag_accepted_bayesian_none",
        "ate_pag_accepted_bayesian_cheap",
        "ate_pag_accepted_bayesian_full",
        "ate_admg_explicit_none",
        "ate_admg_explicit_cheap",
        "ate_admg_explicit_full",
        "ate_admg_accepted_none",
        "ate_admg_accepted_cheap",
        "ate_admg_accepted_full",
        "curve_explicit_none",
        "curve_accepted_none",
        "conditional_explicit_none",
        "conditional_explicit_cheap",
        "conditional_explicit_full",
        "conditional_accepted_none",
        "conditional_accepted_cheap",
        "conditional_accepted_full",
        "path_specific_explicit_none",
        "distribution_explicit_none",
        "intervention_response_explicit_none",
        "intervention_response_accepted_none",
        "pulse_explicit_none",
        "pulse_accepted_none",
        "pulse_explicit_cheap",
        "pulse_accepted_cheap",
        "pulse_explicit_bayesian_none",
        "sustained_explicit_none",
        "sustained_accepted_none",
        "sustained_explicit_cheap",
        "sustained_accepted_cheap",
        "sustained_explicit_bayesian_none",
        "temporal_mediation_explicit_none",
        "temporal_mediation_accepted_none",
    ],
)
def test_licensed_cell_prepare_matches_analyze(data, graph, query, refute, inference):
    fresh = antecedent.analyze(
        data,
        graph=graph,
        query=query,
        refute=refute,
        bootstrap=0,
        seed=1,
        inference=inference,
    )
    prepared = antecedent.estimation.PreparedAnalysis.prepare(
        data,
        graph=graph,
        query=query,
        refute=refute,
        seed=1,
        latency="interactive",
        inference=inference,
    )
    if isinstance(graph, antecedent.AcceptedGraph) and hasattr(prepared, "structure_source"):
        assert prepared.structure_source == "accepted"
    if hasattr(prepared, "evidence_status"):
        assert prepared.evidence_status == "licensed"
    click = prepared.estimate(data, seed=1)
    assert getattr(click, "evidence_status", None) == "licensed"
    if isinstance(query, antecedent.ResponseCurve):
        assert click.response is not None and fresh.response is not None
        assert click.response.values == fresh.response.values
        return
    if isinstance(query, antecedent.InterventionResponse):
        assert isinstance(click.estimate, float) and isinstance(fresh.estimate, float)
        assert math.isfinite(click.estimate) and math.isfinite(fresh.estimate)
        assert click.estimate == fresh.estimate
        # Known-truth pin, not just self-consistency: `_intervention_response_table`
        # is the same deterministic generator as the Rust fixture this cell is
        # licensed against (`conformance/response/intervention_response`).
        assert click.estimate == pytest.approx(_IR_TRUE_RESPONSE, abs=_IR_TOLERANCE)
        return
    if isinstance(query, antecedent.TemporalMediationEffect):
        assert click.mediation is not None and fresh.mediation is not None
        assert click.mediation.mediated == fresh.mediation.mediated
        assert click.ate == fresh.ate
        return
    assert math.isfinite(click.ate) and math.isfinite(fresh.ate)
    assert abs(click.ate - fresh.ate) < 1e-12
    if isinstance(query, (antecedent.PulseEffect, antecedent.SustainedEffect)):
        # Known-truth pin: `y_i = 0.5 * t_(i-1)` with zero noise, so the structural
        # Pulse/single-step-Sustained contrast at treatment_lag=1 is exactly 0.5.
        # Frequentist OLS recovers it to machine precision; the Bayesian posterior
        # mean carries a small prior-shrinkage bias at n=80, hence the looser bound.
        tol = 1e-9 if inference is None else 1e-3
        assert click.ate == pytest.approx(_PULSE_TRUE_ATE, abs=tol)
        if refute == "cheap":
            # `RefuteSuite::Cheap` on a temporal-unfolded design runs exactly one
            # refuter: `OverlapRefuter` is `NotApplicable` for temporal designs
            # (propensity uses schema adjustment columns, which a temporal
            # unfolded design does not have) and is dropped from the report
            # entirely, not surfaced as a "skipped" entry — see
            # `ValidatorId::Overlap` in `crates/antecedent-validate/src/suite.rs`
            # and `crates/antecedent/tests/manufacturing_temporal.rs` for the
            # Rust-side pin of the same fact. So "cheap" here is E-value only,
            # not "overlap + E-value".
            for view, label in ((fresh.validation, "fresh"), (click.validation, "click")):
                names = [r.refuter for r in view.reports]
                assert names == ["sensitivity.evalue"], f"{label}: {names}"
                assert view.reports[0].informative, f"{label}: e-value must be informative"


# `parity/support_licensed.toml` licenses `PulseEffect`/`SustainedEffect` ×
# `TemporalDag` × `Frequentist` × `full` (explicit and accepted), and
# `RefuteSuite::Full` now actually completes at this coordinate through both the
# one-shot `analyze()` path and `PreparedAnalysis.estimate()`.
#
# Root cause (fixed; see `crates/antecedent/tests/manufacturing_temporal.rs` for the
# full trace): `DataSubsetRefuter` (`crates/antecedent-validate/src/data_subset.rs`)
# used to keep a random subset of rows by calling `TabularData::with_analysis_mask`
# rather than physically dropping rows. The temporal refit path rebuilt a
# `TimeSeriesData` from that masked table and lag-gathered it, which
# unconditionally rejected any non-fully-valid analysis mask (`ensure_unmasked` in
# `crates/antecedent-data/src/sample.rs`) because lag-gather indexes raw row
# positions — a masked-out row would corrupt the lag alignment. Physically dropping
# the masked-out rows instead of hiding them would dodge that error but not the
# underlying problem: an *interior* hole makes rows on either side no longer
# adjacent in time, so lag-1/lag-2 would silently mean something else across the
# seam. The fix keeps one random *contiguous* window of rows for temporal designs
# (`with_contiguous_row_window`, `crates/antecedent-validate/src/common.rs`)
# instead: every retained row's immediate predecessor in the window is still its
# immediate predecessor in the original series, so lag semantics survive, and the
# refit rebuilds the series time index at the window's (shorter) length.
@pytest.mark.parametrize(
    "graph, query",
    [
        (_PULSE_TDAG, _PULSE),
        (_PULSE_ACCEPTED, _PULSE),
        (_PULSE_TDAG, _SUSTAINED),
        (_SUSTAINED_ACCEPTED, _SUSTAINED),
    ],
    ids=[
        "pulse_explicit_full",
        "pulse_accepted_full",
        "sustained_explicit_full",
        "sustained_accepted_full",
    ],
)
def test_temporal_dag_full_refute_completes_with_data_subset_refuter(graph, query):
    def _assert_data_subset_refuter(result):
        # `y_i = 0.5 * t_(i-1)`, zero noise (see `_pulse_table`): OLS on *any*
        # complete contiguous window with enough variation exactly recovers the
        # structural coefficient 0.5, so the contiguous-window subset refit
        # reproduces the original ATE to floating point — a genuine, non-degenerate
        # check that keeps lag semantics valid, not a trivially-passing no-op.
        subset = result.validation["data.subset"]
        assert subset.original_ate == pytest.approx(_PULSE_TRUE_ATE, abs=1e-6)
        assert subset.refuted_ate == pytest.approx(_PULSE_TRUE_ATE, abs=1e-6)
        assert subset.comparison == pytest.approx(1.0, abs=1e-6)
        assert subset.passed
        assert subset.replicates == 20

    result = antecedent.analyze(
        _PULSE_DATA, graph=graph, query=query, refute="full", bootstrap=0, seed=1
    )
    assert result.evidence_status == "licensed"
    assert result.ate == pytest.approx(_PULSE_TRUE_ATE, abs=1e-6)
    _assert_data_subset_refuter(result)

    prepared = antecedent.estimation.PreparedAnalysis.prepare(
        _PULSE_DATA, graph=graph, query=query, refute="full", seed=1, latency="interactive"
    )
    assert prepared.evidence_status == "licensed"
    estimated = prepared.estimate(_PULSE_DATA, seed=1)
    assert estimated.ate == pytest.approx(_PULSE_TRUE_ATE, abs=1e-6)
    _assert_data_subset_refuter(estimated)


def test_pag_analyze_refuses_column_order_mismatch():
    pag = antecedent.Pag.from_marked_edges(
        ["z", "t", "y"],
        [
            ("z", "t", "tail", "arrow"),
            ("z", "y", "tail", "arrow"),
            ("t", "y", "tail", "arrow"),
        ],
    )
    with pytest.raises(ValueError, match="must match data column names and order"):
        antecedent.analyze(_ATE_DATA, graph=pag, query=_ATE, refute=False, bootstrap=0, seed=1)
    with pytest.raises(ValueError, match="must match data column names and order"):
        antecedent.estimation.PreparedAnalysis.prepare(
            _ATE_DATA, graph=pag, query=_ATE, refute=False, seed=1, latency="interactive"
        )


def test_admg_analyze_refuses_column_order_mismatch():
    admg = antecedent.Admg.from_edges(
        ["m", "t", "y"], [("t", "m"), ("m", "y")], bidirected=[("t", "y")]
    )
    with pytest.raises(ValueError, match="must match data column names and order"):
        antecedent.analyze(
            _ADMG_DATA, graph=admg, query=_ADMG_ATE, refute=False, bootstrap=0, seed=1
        )


def test_licensed_graph_posterior_prepare_matches_analyze():
    fresh = antecedent.analyze(
        _ATE_DATA,
        discovery=antecedent.discovery.ExactDagPosterior(),
        query=_ATE,
        inference=_BAYES,
        refute=False,
        bootstrap=0,
        seed=7,
    )
    # CodeQL resolves this facade call to the PyO3 class with the same name even
    # though the Python wrapper's signature explicitly accepts discovery=.
    # lgtm[py/call/wrong-named-argument]
    prepared = antecedent.estimation.PreparedAnalysis.prepare(
        _ATE_DATA,
        discovery=antecedent.discovery.ExactDagPosterior(),
        query=_ATE,
        inference=_BAYES,
        refute=False,
        seed=7,
        latency="interactive",
    )
    click = prepared.estimate(_ATE_DATA, seed=7)
    assert prepared.evidence_status == "licensed"
    assert click.evidence_status == "licensed"
    assert abs(click.ate - fresh.ate) < 1e-12


def test_licensed_dbn_posterior_prepare_matches_analyze():
    n = 400
    rng = np.random.default_rng(42)
    pressure = rng.normal(size=n).astype(np.float64)
    defect = np.zeros(n, dtype=np.float64)
    for t in range(1, n):
        defect[t] = 0.9 * pressure[t - 1]
    data = {"pressure": pressure, "defect": defect}
    query = antecedent.PulseEffect(
        treatment="pressure",
        outcome="defect",
        treatment_lag=1,
        horizon_steps=1,
        active_level=1.0,
    )
    discovery = antecedent.discovery.DbnPosterior(max_lag=1)
    inference = antecedent.Bayesian(n_draws=64, prior_scale=100.0, backend="conjugate")
    fresh = antecedent.analyze(
        data,
        discovery=discovery,
        query=query,
        inference=inference,
        refute=False,
        bootstrap=0,
        seed=11,
    )
    # CodeQL resolves this facade call to the PyO3 class with the same name even
    # though the Python wrapper's signature explicitly accepts discovery=.
    # lgtm[py/call/wrong-named-argument]
    prepared = antecedent.estimation.PreparedAnalysis.prepare(
        data,
        discovery=discovery,
        query=query,
        inference=inference,
        refute=False,
        seed=11,
        latency="interactive",
    )
    click = prepared.estimate(data, seed=11)
    assert prepared.evidence_status == "licensed"
    assert click.evidence_status == "licensed"
    assert abs(click.ate - fresh.ate) < 1e-12
