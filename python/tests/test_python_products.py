"""Python product matrix: every route row executes and retains."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import tomllib

import antecedent as ant
from antecedent.data import event, multi_env, panel
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
from antecedent.inference import Frequentist
from antecedent.intervention import Set

ROOT = Path(__file__).resolve().parents[2]
PRODUCTS = tomllib.loads((ROOT / "parity" / "python_products.toml").read_text())
GRAPH = [("z", "t"), ("z", "y"), ("t", "y")]


def _tabular(kind: str, n: int = 160, seed: int = 3) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    if kind in {
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


def _series(n: int = 64, seed: int = 4) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = rng.normal(size=n)
    m = np.zeros(n)
    y = np.zeros(n)
    for i in range(1, n):
        m[i] = 0.4 * t[i - 1] + 0.1 * rng.normal()
        y[i] = 0.4 * t[i - 1] + 0.5 * m[i] + 0.2 * rng.normal()
    return {"t": t, "y": y, "m": m}


def _query(kind: str, *, temporal_surface: bool):
    if kind == "average":
        return AverageEffect("t", "y")
    if kind == "conditional":
        return ConditionalEffect("t", "y", "z")
    if kind == "response_curve":
        return ResponseCurve("t", "y", grid=[0.0, 0.5, 1.0], horizons=[1] if temporal_surface else None)
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


def _run_route(kind: str, data: str, structure: str) -> None:
    temporal = kind in {"pulse", "sustained", "temporal_mediation"} or (
        kind in {"response_curve", "intervention_response"} and data != "tabular"
    )
    query = _query(kind, temporal_surface=temporal)
    if data == "tabular":
        frame = _tabular(kind)
        graph: object = (
            [("z", "t"), ("z", "y"), ("t", "y"), ("t", "m"), ("m", "y")]
            if kind in {"mediation", "path_specific"}
            else GRAPH
        )
    elif data in {"series", "event", "panel", "multi_env"}:
        series = _series()
        if temporal:
            graph = (
                [("t", 1, "m", 0), ("m", 0, "y", 0), ("t", 1, "y", 0)]
                if kind == "temporal_mediation"
                else [("t", 1, "y", 0)]
            )
        else:
            graph = GRAPH
            series = {key: value[:64] for key, value in _tabular(kind).items()}
        if data == "series":
            frame = series
        elif data == "event":
            frame = event(
                series,
                event_times_ns=np.arange(len(series["t"])) * 1_000_000,
                align_interval_ns=1_000_000,
            )
        elif data == "panel":
            frame = panel([series, series])
        else:
            frame = multi_env([series, series])
    else:
        raise AssertionError(data)
    kwargs: dict[str, object] = {
        "query": query,
        "graph": graph,
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

        kwargs["graph"] = AcceptedGraph(graph)
    if structure == "graph_posterior":
        kwargs.pop("graph")
        from antecedent.discovery import DbnPosterior, ExactDagPosterior
        from antecedent.inference import Bayesian

        kwargs["discovery"] = DbnPosterior() if temporal else ExactDagPosterior()
        kwargs["accept_discovered"] = True
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
    result = analyze(frame, **kwargs)
    study = result.study
    assert study is not None
    encoded = result.export()
    loaded = ant.artifacts.loads(encoded)
    assert loaded.payload_kind == "analysis_result"


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
}


def test_every_route_retains():
    for row in PRODUCTS.get("route", []):
        assert row.get("retains") is True, row


def test_every_licensed_combination_has_a_route_row():
    licensed = tomllib.loads((ROOT / "parity" / "support_licensed.toml").read_text())
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
            expected.add((kind, "panel", structure))
            expected.add((kind, "event", structure))
        if kind in {"average", "pulse", "sustained"}:
            expected.add((kind, "multi_env", structure))
    present = {
        (row["kind"], row["data"], row["structure"]) for row in PRODUCTS.get("route", [])
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
    for row in PRODUCTS.get("route", []):
        assert row.get("test"), row


def _install_route_tests() -> None:
    for row in PRODUCTS.get("route", []):
        name = row["test"].rsplit("::", 1)[-1]
        kind, data, structure = row["kind"], row["data"], row["structure"]

        def _test(kind=kind, data=data, structure=structure) -> None:
            _run_route(kind, data, structure)

        _test.__name__ = name
        globals()[name] = _test


_install_route_tests()
