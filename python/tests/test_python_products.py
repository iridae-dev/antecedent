"""Python product matrix: every route retains a study that re-executes it.

Each `[[route]]` row in `parity/python_products.toml` runs one `ant.analyze(...)`
and then asserts the five things retention has to mean:

1. `outcome.study` exists;
2. `outcome.export()` decodes and accepts as a verified program, loaded against the
   expectation retained at the producer;
3. `ant.estimate(outcome.study)` reproduces the same point (or surface), the same
   `program_id` and `claim_id`, and the same data snapshot — so it is the same
   program on the same data, not a re-run of something adjacent;
4. `ant.rebind(outcome.study, data=<the same observations>)` is a new branch that
   reproduces the same reading while the source study still reproduces its own;
5. panel / multi-environment routes use every unit and environment: the
   fixtures are deliberately different per partition, and a route that quietly
   dropped all but the first would produce the single-partition answer, which
   the dedicated tests below pin as *different*.

Each `[[refusal]]` row pins a combination the study builder does not license,
by its reason code.
"""

from __future__ import annotations

import inspect
import math
import tomllib
from pathlib import Path

import antecedent as ant
import numpy as np
import pytest
from antecedent import (
    AverageDerivative,
    AverageEffect,
    ConditionalEffect,
    Counterfactual,
    DirectionalDerivative,
    Elasticity,
    InterventionalDistribution,
    InterventionResponse,
    MediationEffect,
    PathSpecificEffect,
    PointDerivative,
    PulseEffect,
    ResponseCurve,
    ResponseJacobian,
    SemiElasticity,
    SustainedEffect,
    TemporalMediationEffect,
)
from antecedent.data import event, multi_env, panel
from antecedent.errors import CausalError
from antecedent.inference import Frequentist
from antecedent.intervention import Set
from antecedent.outcomes import require_success

from _repo_text import REPO_ROOT, read_text
from _sealed_loads import assert_answer_kept

ROOT = REPO_ROOT
PRODUCTS = tomllib.loads(read_text(ROOT / "parity" / "python_products.toml"))
GRAPH = [("z", "t"), ("z", "y"), ("t", "y")]
LAGGED = [("t", 1, "y", 0)]


def _tabular(
    kind: str, n: int = 160, seed: int = 3, continuous: bool = False
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    if continuous or kind in {
        "response_curve",
        "average_derivative",
        "point_derivative",
        "elasticity",
        "semi_elasticity",
        "directional_derivative",
        "response_jacobian",
    }:
        t = z + rng.normal(scale=0.35, size=n)
    else:
        t = (rng.uniform(size=n) < 1 / (1 + np.exp(-z))).astype(float)
    m = 0.4 * t + rng.normal(scale=0.2, size=n)
    y = t + z + 0.8 * m + rng.normal(scale=0.2, size=n)
    if kind in {"distribution", "path_specific"}:
        t = (t > np.median(t)).astype(float)
        y = (y > np.median(y)).astype(float)
        z = (z > 0).astype(float)
        m = (m > 0).astype(float)
    return {"t": t, "y": y, "z": z, "m": m}


def _series(n: int = 64, seed: int = 4, beta: float = 0.4) -> dict[str, np.ndarray]:
    """One series; `beta` makes each panel unit / environment distinguishable."""
    rng = np.random.default_rng(seed)
    t = rng.normal(size=n)
    m = np.zeros(n)
    y = np.zeros(n)
    for i in range(1, n):
        m[i] = 0.4 * t[i - 1] + 0.1 * rng.normal()
        y[i] = beta * t[i - 1] + 0.5 * m[i] + 0.2 * rng.normal()
    return {"t": t, "y": y, "m": m}


#: Two clearly different partitions: a route that used only the first would
#: report 0.4 where the pooled answer is between 0.4 and 2.0.
UNIT_A = _series(seed=4, beta=0.4)
UNIT_B = _series(seed=9, beta=2.0)
UNIT_C = _series(seed=13, beta=-1.0)


def _query(kind: str, *, temporal_surface: bool):
    if kind == "average":
        return AverageEffect("t", "y")
    if kind == "conditional":
        return ConditionalEffect("t", "y", "z")
    if kind == "response_curve":
        return ResponseCurve(
            "t", "y", grid=[0.0, 0.5, 1.0], horizons=[1] if temporal_surface else None
        )
    if kind == "intervention_response":
        return InterventionResponse(
            "y",
            intervention=Set("t", 1.0),
            horizons=[1] if temporal_surface else None,
        )
    if kind == "pulse":
        return PulseEffect("t", "y")
    if kind == "sustained":
        return SustainedEffect("t", "y")
    if kind == "mediation":
        return MediationEffect("t", "y", mediators=["m"])
    if kind == "counterfactual":
        return Counterfactual("t", "y")
    if kind == "path_specific":
        return PathSpecificEffect("t", "y", path_nodes=["m"])
    if kind == "distribution":
        return InterventionalDistribution("y", interventions={"t": 1.0})
    if kind == "temporal_mediation":
        return TemporalMediationEffect("t", "m", "y")
    if kind == "average_derivative":
        return AverageDerivative("t", "y")
    if kind == "point_derivative":
        return PointDerivative("t", "y", at=0.5)
    if kind == "elasticity":
        return Elasticity("t", "y", at=0.5)
    if kind == "semi_elasticity":
        return SemiElasticity("t", "y", at=0.5)
    if kind == "directional_derivative":
        return DirectionalDerivative(["t"], ["y"], at={"t": 0.5}, direction={"t": 1.0})
    if kind == "response_jacobian":
        return ResponseJacobian(["t"], ["y"], at={"t": 0.5})
    raise AssertionError(kind)


def _non_gaussian(n: int = 300, seed: int = 3) -> dict[str, np.ndarray]:
    """LiNGAM / NOTEARS solve a linear SEM with non-Gaussian noise."""
    rng = np.random.default_rng(seed)
    z = rng.uniform(-1.0, 1.0, size=n)
    t = z + 0.5 * rng.uniform(-1.0, 1.0, size=n)
    y = t + z + 0.3 * rng.uniform(-1.0, 1.0, size=n)
    return {"t": t, "y": y, "z": z, "m": 0.4 * t + 0.2 * rng.uniform(-1.0, 1.0, size=n)}


def _discovery_series(n: int = 200, seed: int = 12) -> dict[str, np.ndarray]:
    """One clean lag-1 link, so a discovered temporal graph stays inside the history cap."""
    rng = np.random.default_rng(seed)
    t = rng.normal(size=n)
    y = np.zeros(n)
    for i in range(1, n):
        y[i] = 0.8 * t[i - 1] + 0.2 * rng.normal()
    return {"t": t, "y": y}


def _discovery(structure: str):
    from antecedent import discovery as d

    name = structure.split(".", 1)[1]
    return {
        "pc": d.PC(alpha=0.2, fdr=False),
        "ges": d.GES(alpha=0.2, fdr=False),
        "lingam": d.LiNGAM(),
        "notears": d.NOTEARS(),
        "fci": d.FCI(alpha=0.2, fdr=False),
        "pcmci": d.PCMCI(max_lag=1),
        "pcmci_plus": d.PCMCIPlus(max_lag=1),
        "rpcmci": d.RPCMCI(max_lag=1),
        "jpcmci_plus": d.JPCMCIPlus(max_lag=1),
    }[name]


def _frame(kind: str, data: str, temporal: bool):
    if data == "tabular":
        return _tabular(kind)
    units = [UNIT_A, UNIT_B]
    if not temporal:
        units = [{key: value[:64] for key, value in _tabular(kind).items()}] * 2
    if data == "series":
        return units[0]
    if data == "event":
        return event(
            units[0],
            event_times_ns=np.arange(len(units[0]["t"])) * 1_000_000,
            align_interval_ns=1_000_000,
        )
    if data == "panel":
        return panel(units)
    if data == "multi_env":
        return multi_env(units)
    raise AssertionError(data)


def _graph(kind: str, data: str, temporal: bool):
    if not temporal:
        return (
            [("z", "t"), ("z", "y"), ("t", "y"), ("t", "m"), ("m", "y")]
            if kind in {"mediation", "path_specific"}
            else GRAPH
        )
    return (
        [("t", 1, "m", 0), ("m", 0, "y", 0), ("t", 1, "y", 0)]
        if kind == "temporal_mediation"
        else LAGGED
    )


def _point(result):
    """A comparable reading of whatever this route publishes."""
    estimate = getattr(result, "estimate", None)
    if estimate is not None and hasattr(estimate, "ate"):
        return ("ate", estimate.ate)
    if estimate is not None and not hasattr(estimate, "ate"):
        return ("scalar", estimate)
    response = getattr(result, "response", None)
    if response is not None:
        return ("values", tuple(np.ravel(np.asarray(response.values, dtype=float))))
    raise AssertionError(f"no comparable reading for {result!r}")


def _same(left, right) -> bool:
    kind_l, value_l = left
    kind_r, value_r = right
    if kind_l != kind_r:
        return False
    if isinstance(value_l, tuple):
        return len(value_l) == len(value_r) and all(
            a == b or (math.isnan(a) and math.isnan(b))
            for a, b in zip(value_l, value_r, strict=True)
        )
    if isinstance(value_l, float) and math.isnan(value_l):
        return isinstance(value_r, float) and math.isnan(value_r)
    return value_l == value_r


def _kwargs(kind: str, data: str, structure: str, options: str) -> dict[str, object]:
    temporal = kind in {"pulse", "sustained", "temporal_mediation"} or (
        kind in {"response_curve", "intervention_response"} and data != "tabular"
    )
    kwargs: dict[str, object] = {
        "query": _query(kind, temporal_surface=temporal),
        "graph": _graph(kind, data, temporal),
        "refute": "none",
        "inference": Frequentist(bootstrap=0),
        "execution": ant.Execution(seed=1),
    }
    if kind in {
        "point_derivative",
        "elasticity",
        "semi_elasticity",
        "directional_derivative",
        "response_jacobian",
        "average_derivative",
    }:
        kwargs["estimator_config"] = {"bandwidth": 0.45}
    if structure == "accepted":
        from antecedent.accepted_graph import AcceptedGraph

        kwargs["graph"] = AcceptedGraph(kwargs["graph"])
    if structure == "graph_posterior":
        kwargs.pop("graph")
        from antecedent.discovery import DbnPosterior, ExactDagPosterior
        from antecedent.inference import Bayesian

        kwargs["discovery"] = DbnPosterior() if temporal else ExactDagPosterior()
        freq_ok = kind in {
            "average",
            "conditional",
            "response_curve",
            "intervention_response",
            "pulse",
            "sustained",
            "temporal_mediation",
        }
        kwargs["inference"] = Frequentist(bootstrap=0) if freq_ok else Bayesian(n_draws=32)
    if structure.startswith("discovered."):
        kwargs.pop("graph")
        kwargs["discovery"] = _discovery(structure)
    if options == "validators":

        def one_sided(*, ate, **_kwargs):
            return {"passed": not math.isnan(ate), "refuted_ate": ate, "comparison": 0.0}

        kwargs["validators"] = {"matrix.finite_effect": one_sided}
    if options == "target_population":
        from antecedent.population import Treated

        kwargs["query"] = AverageEffect("t", "y", target_population=Treated())
        kwargs["estimator"] = "aipw"
    if options == "population_registry":
        from antecedent.population import CustomDistribution, PopulationRegistry

        registry = PopulationRegistry()
        registry.insert_distribution(3, [1.0] * 80 + [0.5] * 80)
        kwargs["population_registry"] = registry
        kwargs["query"] = AverageEffect("t", "y", target_population=CustomDistribution(3))
        kwargs["estimator"] = "propensity.weighting"
    if options == "rd":
        from antecedent.quasi import SharpRegressionDiscontinuity

        # Sharp RD is a graphless query that synthesizes its own design.
        kwargs.pop("graph", None)
        kwargs["query"] = SharpRegressionDiscontinuity("y", "t", "r", 0.0, 1.5)
    return kwargs


def _rd_data(n: int = 1200, seed: int = 25) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    r = rng.uniform(-2.0, 2.0, size=n)
    t = (r >= 0.0).astype(float)
    y = 1.0 + 2.0 * t + 0.3 * r + rng.normal(scale=0.2, size=n)
    return {"t": t, "y": y, "r": r}


def _execute_discovery(
    kwargs: dict[str, object], kind: str, data: str, structure: str, temporal: bool, frame
) -> None:
    """Run the declared discovery explicitly; preparation only accepts its executed result."""
    if structure == "graph_posterior":
        # A posterior is fitted on the flat observations; the partitioned frame is
        # what the prepared study is then bound to (and refused on, where unlicensed).
        flat = frame if data in {"tabular", "series"} else _frame(kind, "series", temporal)
        kwargs["discovery"] = kwargs["discovery"].run(flat, execution=ant.Execution(seed=1))
    elif structure.startswith("discovered."):
        from antecedent import discovery as d

        kwargs["graph"] = d.discover(
            frame,
            method=kwargs.pop("discovery"),
            execution=ant.Execution(seed=1),
            accept_discovered=True,
            regimes=[0] * 200 if structure.endswith("rpcmci") else None,
        )


def _run_route(kind: str, data: str, structure: str, options: str = "") -> None:
    temporal = kind in {"pulse", "sustained", "temporal_mediation"} or (
        kind in {"response_curve", "intervention_response"} and data != "tabular"
    )
    kwargs = _kwargs(kind, data, structure, options)
    if options == "rd":
        frame = _rd_data()
    elif structure.startswith("discovered.") and temporal:
        frame = _discovery_series()
    elif structure in {"discovered.lingam", "discovered.notears"}:
        frame = _non_gaussian()
    else:
        frame = _frame(kind, data, temporal)
    seen: list[tuple[float, str]] = []
    if options == "callbacks":
        kwargs["execution"] = ant.Execution(
            seed=1,
            cancel=ant.state.CancellationToken(),
            on_progress=lambda fraction, stage: seen.append((fraction, stage)),
        )
    _execute_discovery(kwargs, kind, data, structure, temporal, frame)
    outcome = ant.analyze(frame, **kwargs)
    result = require_success(outcome)

    study = outcome.study
    assert study is not None
    artifact = outcome.export()
    loaded = require_success(ant.load(bytes(artifact), expected=artifact.expectation))
    assert loaded.decoded_artifact.payload_kind == "analysis_result"
    assert_answer_kept(loaded)

    label = f"{kind}/{data}/{structure}"
    ids = ("program_id", "claim_id", "data_snapshot_id")

    def assert_reproduces(branch_outcome, how: str) -> None:
        again = require_success(branch_outcome)
        assert _same(_point(result), _point(again)), f"{label}: {how} did not reproduce its point"
        for name in ids:
            assert branch_outcome.identities[name] == outcome.identities[name], (label, how, name)

    assert_reproduces(ant.estimate(study), "the retained study")
    # A rebind onto the same observations is a new branch; it does not disturb its source.
    branch = ant.rebind(study, data=frame)
    assert branch is not study
    assert_reproduces(ant.estimate(branch), "the rebound branch")
    assert_reproduces(ant.estimate(study), "the source study after its branch ran")
    if options == "callbacks":
        assert seen, "on_progress never fired"
    if options == "validators":
        attested = loaded.decoded_artifact.contract["claim"]["attested"]
        assert [item["name"] for item in attested] == ["matrix.finite_effect"]


def test_rebinding_new_observations_leaves_the_source_study_unchanged():
    """A branch on different observations answers about those; the source keeps its own."""
    source = ant.prepare(_tabular("average"), **_kwargs("average", "tabular", "explicit", ""))
    before = ant.estimate(source)
    branch = ant.rebind(source, data=_tabular("average", seed=11))
    moved = ant.estimate(branch)
    assert moved.identities["data_snapshot_id"] != before.identities["data_snapshot_id"]
    assert not _same(_point(require_success(moved)), _point(require_success(before)))
    again = ant.estimate(source)
    assert again.identities["data_snapshot_id"] == before.identities["data_snapshot_id"]
    assert _same(_point(require_success(again)), _point(require_success(before)))


@pytest.mark.parametrize("seed", [11, 12])
def test_average_effect_route_recovers_the_data_generating_total_effect(seed):
    # y = t + z + 0.8 m with m = 0.4 t + noise: the total effect of t is 1 + 0.8 * 0.4 = 1.32.
    data = _tabular("average", n=1000, seed=seed)
    result = require_success(
        ant.analyze(data, **_kwargs("average", "tabular", "explicit", ""))
    )
    assert _point(result)[1] == pytest.approx(1.32, abs=0.05)


def test_every_route_retains():
    for row in PRODUCTS.get("route", []):
        assert row.get("retains") is True, row


#: Graph classes the Rust licensing table covers that `analyze` cannot be handed: a
#: `TemporalAdmg` is a native-only graph object (the contemporaneous-projection twin of the
#: static latent-graph analyses), not an edge list, so no Python route exists to carry a row.
RUST_ONLY_GRAPH_CLASSES = frozenset({"TemporalAdmg"})


#: Static query kinds whose temporal cores the Rust table licenses through the temporal
#: effect wrapper, which Python has no surface for: its static derivative, counterfactual,
#: distribution and path-specific classes take no lag or horizon, and handing them panel or
#: event data is refused (pinned by the refusal rows). The Python-reachable temporal
#: conditional queries are `TemporalConditionalPulse` / `TemporalConditionalSustained`.
RUST_ONLY_TEMPORAL_KINDS = frozenset(
    {
        "average_derivative",
        "counterfactual",
        "directional_derivative",
        "distribution",
        "elasticity",
        "path_specific",
        "point_derivative",
        "response_jacobian",
        "semi_elasticity",
    }
)


def test_every_licensed_combination_has_a_route_row():
    licensed = tomllib.loads(read_text(ROOT / "parity" / "support_licensed.toml"))
    expected: set[tuple[str, str, str]] = set()
    for cell in licensed.get("cell", []):
        kind = QUERY_KIND.get(cell["query"])
        if kind is None:
            continue
        structure = cell["structure"]
        graph = cell["graph_class"]
        if graph in RUST_ONLY_GRAPH_CLASSES:
            continue
        if kind in RUST_ONLY_TEMPORAL_KINDS and graph.startswith("Temporal"):
            continue
        data = "series" if graph.startswith("Temporal") else "tabular"
        expected.add((kind, data, structure))
        if graph.startswith("Temporal"):
            expected.add((kind, "event", structure))
            # Panel and multi-environment carry the modalities the Rust study
            # builder licenses for this query, and nothing else; the refused
            # combinations are pinned as refusals below.
            if structure != "graph_posterior" and kind != "temporal_mediation":
                expected.add((kind, "panel", structure))
            if structure != "graph_posterior" and kind in {"pulse", "sustained"}:
                expected.add((kind, "multi_env", structure))
    present = {
        (row["kind"], row["data"], row["structure"])
        for row in PRODUCTS.get("route", [])
        if not row.get("options")
    }
    missing = sorted(expected - present)
    assert not missing, f"licensed combinations missing product rows: {missing}"


def test_every_signature_parameter_has_a_row():
    rows = {row["name"] for row in PRODUCTS.get("parameter", [])}
    for name, fn in {"analyze": ant.analyze, "prepare": ant.prepare}.items():
        params = [
            key
            for key, parameter in inspect.signature(fn).parameters.items()
            if parameter.kind
            not in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
        ]
        missing = [key for key in params if key not in rows]
        assert not missing, f"{name} missing product rows: {missing}"
        # `analyze` forwards its declarations to `prepare`, so it takes exactly its parameters.
        assert name == "analyze" or "execution" in params


def test_route_tests_are_named():
    for row in PRODUCTS.get("route", []) + PRODUCTS.get("refusal", []):
        assert row.get("test"), row


def test_every_refusal_row_names_a_reason_code():
    codes = {
        row["id"]
        for row in tomllib.loads(read_text(ROOT / "parity" / "reason_codes.toml"))["code"]
        if "python_product" in row.get("applies_to", [])
    }
    for row in PRODUCTS.get("refusal", []):
        assert row["reason"] in codes, row


QUERY_KIND = {
    "AverageEffect": "average",
    "ConditionalEffect": "conditional",
    "ResponseCurve": "response_curve",
    "InterventionResponse": "intervention_response",
    "PulseEffect": "pulse",
    "SustainedEffect": "sustained",
    "MediationEffect": "mediation",
    "Counterfactual": "counterfactual",
    "PathSpecificEffect": "path_specific",
    "InterventionalDistribution": "distribution",
    "TemporalMediationEffect": "temporal_mediation",
    "AverageDerivative": "average_derivative",
    "PointDerivative": "point_derivative",
    "Elasticity": "elasticity",
    "SemiElasticity": "semi_elasticity",
    "DirectionalDerivative": "directional_derivative",
    "ResponseJacobian": "response_jacobian",
    "TransportQuery": "transport",
    "InterferenceQuery": "interference",
}


# --- panel / multi-environment really use every partition -------------------------


def _pulse(frame):
    return require_success(
        ant.analyze(
            frame,
            graph=LAGGED,
            query=PulseEffect("t", "y"),
            inference=ant.Frequentist(bootstrap=0),
            refute="none",
            execution=ant.Execution(seed=1),
        )
    ).estimate.ate


def test_panel_pulse_uses_every_unit():
    first, second = _pulse(UNIT_A), _pulse(UNIT_B)
    pooled = _pulse(panel([UNIT_A, UNIT_B]))
    assert not math.isclose(first, second, rel_tol=1e-6)
    assert not math.isclose(pooled, first, rel_tol=1e-6)
    assert not math.isclose(pooled, second, rel_tol=1e-6)
    # Known answers: y[i] = beta t[i-1] + 0.5 m[i] and m[i] = 0.4 t[i-1] + noise, so the
    # lag-1 pulse effect is beta + 0.2 = 0.6 and 2.2; unit variances of t are equal, so the
    # pooled slope is their mean 1.4.
    assert first == pytest.approx(0.6, abs=0.12)
    assert second == pytest.approx(2.2, abs=0.06)
    assert pooled == pytest.approx(1.4, abs=0.05)
    # Unit order is not the answer: the pooled estimate is the same either way.
    assert math.isclose(pooled, _pulse(panel([UNIT_B, UNIT_A])), rel_tol=1e-9)


def test_panel_response_uses_every_unit():
    def surface(frame):
        view = require_success(
            ant.analyze(
                frame,
                graph=LAGGED,
                query=ResponseCurve("t", "y", grid=[0.0, 1.0], horizons=[1]),
                inference=ant.Frequentist(bootstrap=0),
                refute="none",
                execution=ant.Execution(seed=1),
            )
        )
        return tuple(np.ravel(np.asarray(view.response.values, dtype=float)))

    pooled = surface(panel([UNIT_A, UNIT_B]))
    assert pooled != surface(UNIT_A)
    assert pooled != surface(UNIT_B)


def test_multi_env_prepare_binds_every_environment():
    """Every environment is bound into the prepared study's data snapshot.

    The multi-environment executor pools the environments as the clusters of one
    regression, so a cluster-robust variance needs at least two of them; the Python
    layer's guarantee is that no environment is dropped before the study is compiled.
    """
    three = ant.prepare(
        multi_env([UNIT_A, UNIT_B, UNIT_C]),
        graph=LAGGED,
        query=PulseEffect("t", "y"),
        inference=ant.Frequentist(bootstrap=0),
        refute="none",
    )
    both = ant.prepare(
        multi_env([UNIT_A, UNIT_B]),
        graph=LAGGED,
        query=PulseEffect("t", "y"),
        inference=ant.Frequentist(bootstrap=0),
        refute="none",
    )
    reordered = ant.prepare(
        multi_env([UNIT_B, UNIT_A]),
        graph=LAGGED,
        query=PulseEffect("t", "y"),
        inference=ant.Frequentist(bootstrap=0),
        refute="none",
    )
    # Preparation does not materialize snapshot identity; the executed standing reads it.
    assert both.inspect().data_snapshot_id is None

    def snapshot(study):
        outcome = ant.estimate(study)
        require_success(outcome)
        return outcome.study.inspect().data_snapshot_id

    # A cluster-robust variance needs at least two environments, so the comparison is
    # between sets of two and three environments: an environment dropped before the study
    # is compiled would collapse them.
    assert snapshot(three) != snapshot(both)
    # Environment order is part of the snapshot, so nothing collapses them.
    assert snapshot(both) != snapshot(reordered)


# --- refusals ---------------------------------------------------------------------


def _refusal_code(kind: str, data: str, structure: str) -> str | None:
    """The reason code a combination is refused with, as a typed refusal or a raised error."""
    temporal = kind in {"pulse", "sustained", "temporal_mediation"} or (
        kind in {"response_curve", "intervention_response"} and data != "tabular"
    )
    kwargs = _kwargs(kind, data, structure, "")
    frame = _frame(kind, data, temporal)
    try:
        _execute_discovery(kwargs, kind, data, structure, temporal, frame)
        outcome = ant.analyze(frame, **kwargs)
    except CausalError as error:
        return getattr(error, "reason_code", None)
    assert outcome.refusal is not None, f"{kind}/{data}/{structure} was not refused"
    return outcome.refusal.code


@pytest.mark.parametrize(
    ("kind", "data", "structure", "reason"),
    [
        (row["kind"], row["data"], row["structure"], row["reason"])
        for row in PRODUCTS.get("refusal", [])
    ],
    ids=[f"{row['kind']}_{row['data']}_{row['structure']}" for row in PRODUCTS.get("refusal", [])],
)
def test_refusal_rows_are_reason_coded(kind, data, structure, reason):
    assert _refusal_code(kind, data, structure) == reason


def _install_route_tests() -> None:
    for row in PRODUCTS.get("route", []):
        path, name = row["test"].rsplit("::", 1)
        if Path(path).name != Path(__file__).name:
            # The row's own test file runs it (the design cells' lifecycle file).
            continue
        kind, data, structure = row["kind"], row["data"], row["structure"]
        options = row.get("options", "")

        def _test(kind=kind, data=data, structure=structure, options=options) -> None:
            _run_route(kind, data, structure, options)

        _test.__name__ = name
        globals()[name] = _test

    for row in PRODUCTS.get("refusal", []):
        name = row["test"].rsplit("::", 1)[-1]
        kind, data, structure, reason = (
            row["kind"],
            row["data"],
            row["structure"],
            row["reason"],
        )

        def _test(kind=kind, data=data, structure=structure, reason=reason) -> None:
            test_refusal_rows_are_reason_coded(kind, data, structure, reason)

        _test.__name__ = name
        globals()[name] = _test


_install_route_tests()
