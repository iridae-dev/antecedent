"""Python product matrix: every route retains a study that re-executes it.

Each `[[route]]` row in `parity/python_products.toml` runs one `analyze(...)`
and then asserts the four things retention has to mean:

1. `result.study` exists;
2. `result.export()` decodes and accepts as a verified program;
3. `result.study.estimate()` reproduces the same point (or surface), the same
   `program_id` and `claim_id`, and the same data snapshot — so it is the same
   program on the same data, not a re-run of something adjacent;
4. panel / multi-environment routes use every unit and environment: the
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
    analyze,
)
from antecedent.data import event, multi_env, panel
from antecedent.errors import CausalError
from antecedent.inference import Frequentist
from antecedent.intervention import Set

from _repo_text import read_text
from _sealed_loads import assert_answer_kept

ROOT = Path(__file__).resolve().parents[2]
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
        "bootstrap": 0,
        "refute": "none",
        "seed": 1,
        "inference": Frequentist(),
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
        kwargs["inference"] = Frequentist() if freq_ok else Bayesian(n_draws=32)
    if structure.startswith("discovered."):
        kwargs.pop("graph")
        kwargs["discovery"] = _discovery(structure)
        if structure.endswith("rpcmci"):
            kwargs["regimes"] = [0] * 200
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
        from antecedent.estimators import SharpRd

        # The sharp design as a graph: the running variable is the treatment's only cause.
        kwargs["graph"] = [("r", "t"), ("t", "y"), ("r", "y")]
        kwargs["estimator"] = SharpRd(running_variable="r", cutoff=0.0, bandwidth=1.5)
    return kwargs


def _rd_data(n: int = 1200, seed: int = 25) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    r = rng.uniform(-2.0, 2.0, size=n)
    t = (r >= 0.0).astype(float)
    y = 1.0 + 2.0 * t + 0.3 * r + rng.normal(scale=0.2, size=n)
    return {"t": t, "y": y, "r": r}


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
        kwargs["on_progress"] = lambda fraction, stage: seen.append((fraction, stage))
        kwargs["cancel"] = ant.state.CancellationToken()
    result = analyze(frame, **kwargs)

    study = result.study
    assert study is not None
    encoded = result.export()
    loaded = ant.load(encoded)
    assert loaded.artifact.payload_kind == "analysis_result"
    assert_answer_kept(loaded)

    again = study.estimate()
    assert _same(_point(result), _point(again)), (
        f"{kind}/{data}/{structure}: the retained study did not reproduce its own point"
    )
    assert again.program_id == result.program_id
    assert again.claim_id == result.claim_id
    assert again.data_snapshot_id == result.data_snapshot_id
    if options == "callbacks":
        assert seen, "on_progress never fired"
    if options == "validators":
        attested = ant.load(result.export()).artifact.contract["claim"]["attested"]
        assert [item["name"] for item in attested] == ["matrix.finite_effect"]


def test_every_route_retains():
    for row in PRODUCTS.get("route", []):
        assert row.get("retains") is True, row


def test_every_licensed_combination_has_a_route_row():
    licensed = tomllib.loads(read_text(ROOT / "parity" / "support_licensed.toml"))
    expected: set[tuple[str, str, str]] = set()
    for cell in licensed.get("cell", []):
        kind = QUERY_KIND.get(cell["query"])
        if kind is None:
            continue
        structure = cell["structure"]
        graph = cell["graph_class"]
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
    for name, fn in {
        "analyze": analyze,
        "prepare": ant.prepare,
        "PreparedAnalysis.prepare": __import__(
            "antecedent.estimation", fromlist=["PreparedAnalysis"]
        ).PreparedAnalysis.prepare,
    }.items():
        params = [
            key
            for key in inspect.signature(fn).parameters
            if key not in {"self", "cls", "args", "kwargs"}
        ]
        missing = [key for key in params if key not in rows]
        assert not missing, f"{name} missing product rows: {missing}"


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
    return analyze(
        frame,
        graph=LAGGED,
        query=PulseEffect("t", "y"),
        bootstrap=0,
        refute="none",
        seed=1,
    ).estimate.ate


def test_panel_pulse_uses_every_unit():
    first, second = _pulse(UNIT_A), _pulse(UNIT_B)
    pooled = _pulse(panel([UNIT_A, UNIT_B]))
    assert not math.isclose(first, second, rel_tol=1e-6)
    assert not math.isclose(pooled, first, rel_tol=1e-6)
    assert not math.isclose(pooled, second, rel_tol=1e-6)
    # Unit order is not the answer: the pooled estimate is the same either way.
    assert math.isclose(pooled, _pulse(panel([UNIT_B, UNIT_A])), rel_tol=1e-9)


def test_panel_response_uses_every_unit():
    def surface(frame):
        view = analyze(
            frame,
            graph=LAGGED,
            query=ResponseCurve("t", "y", grid=[0.0, 1.0], horizons=[1]),
            bootstrap=0,
            refute="none",
            seed=1,
        )
        return tuple(np.ravel(np.asarray(view.response.values, dtype=float)))

    pooled = surface(panel([UNIT_A, UNIT_B]))
    assert pooled != surface(UNIT_A)
    assert pooled != surface(UNIT_B)


def test_multi_env_prepare_binds_every_environment():
    """Every environment is bound into the prepared study's data snapshot.

    What the multi-environment executor then does with them (it currently
    estimates on the first environment) is the Rust multi-environment route's
    contract, not this layer's: the Python layer's guarantee is that no
    environment is dropped before the study is compiled.
    """
    one = ant.prepare(
        multi_env([UNIT_A]), graph=LAGGED, query=PulseEffect("t", "y"), bootstrap=0, refute="none"
    )
    both = ant.prepare(
        multi_env([UNIT_A, UNIT_B]),
        graph=LAGGED,
        query=PulseEffect("t", "y"),
        bootstrap=0,
        refute="none",
    )
    reordered = ant.prepare(
        multi_env([UNIT_B, UNIT_A]),
        graph=LAGGED,
        query=PulseEffect("t", "y"),
        bootstrap=0,
        refute="none",
    )
    assert one.inspect().data_snapshot_id != both.inspect().data_snapshot_id
    # Environment order is part of the snapshot, so nothing collapses them.
    assert both.inspect().data_snapshot_id != reordered.inspect().data_snapshot_id


# --- refusals ---------------------------------------------------------------------


def _refusal_call(kind: str, data: str, structure: str):
    temporal = kind in {"pulse", "sustained", "temporal_mediation"} or (
        kind in {"response_curve", "intervention_response"} and data != "tabular"
    )
    kwargs = _kwargs(kind, data, structure, "")
    if kind == "path_specific" and structure == "graph_posterior":
        from antecedent.discovery import ExactDagPosterior
        from antecedent.inference import Bayesian

        kwargs.pop("graph", None)
        kwargs["discovery"] = ExactDagPosterior()
        kwargs["inference"] = Bayesian(n_draws=32)
    return lambda: analyze(_frame(kind, data, temporal), **kwargs)


@pytest.mark.parametrize(
    ("kind", "data", "structure", "reason"),
    [
        (row["kind"], row["data"], row["structure"], row["reason"])
        for row in PRODUCTS.get("refusal", [])
    ],
    ids=[f"{row['kind']}_{row['data']}_{row['structure']}" for row in PRODUCTS.get("refusal", [])],
)
def test_refusal_rows_are_reason_coded(kind, data, structure, reason):
    with pytest.raises(CausalError) as raised:
        _refusal_call(kind, data, structure)()
    assert getattr(raised.value, "reason_code", None) == reason, str(raised.value)


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
