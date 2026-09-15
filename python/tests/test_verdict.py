"""One identification renderer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import antecedent as ant
from antecedent._native import identification_status_names
from antecedent._verdict import VERDICTS, verdict_for

SCHEMA = json.loads(
    (Path(__file__).resolve().parent / "fixtures" / "certificate_schema.json").read_text()
)


def _assert_schema(instance: dict) -> None:
    required = SCHEMA["required"]
    props = set(SCHEMA["properties"])
    extra = set(instance) - props
    assert not extra, extra
    for key in required:
        assert key in instance, key


def test_verdict_table_is_exhaustive():
    assert set(VERDICTS) == set(identification_status_names())


def test_verdicts_cover_native_status_names():
    test_verdict_table_is_exhaustive()


def test_statement_lines_nonempty_when_assumptions_exist():
    identification = ant.identify(
        graph=[("z", "t"), ("z", "y"), ("t", "y")],
        query=ant.AverageEffect("t", "y"),
        names=["t", "y", "z"],
    )
    assert identification.statement
    assert identification.verdict == verdict_for(identification.status)
    if identification.assumption_count:
        assert identification.assumption_statements


def test_certificate_matches_schema():
    rng = np.random.default_rng(3)
    z = rng.normal(size=96)
    t = (rng.uniform(size=96) < 0.5).astype(float)
    y = t + z + rng.normal(scale=0.2, size=96)
    data = {"t": t, "y": y, "z": z}

    dag = ant.analyze(
        data,
        graph=[("z", "t"), ("z", "y"), ("t", "y")],
        query=ant.AverageEffect("t", "y"),
        bootstrap=0,
        refute="none",
    )
    admg = ant.analyze(
        data,
        graph=ant.Admg.from_edges(["t", "y", "z"], directed=[("t", "y"), ("z", "t"), ("z", "y")]),
        query=ant.AverageEffect("t", "y"),
        bootstrap=0,
        refute="none",
    )
    cpdag = ant.analyze(
        data,
        graph=ant.Cpdag.from_directed_undirected(
            ["t", "y", "z"], directed=[("z", "t"), ("z", "y"), ("t", "y")], undirected=[]
        ),
        query=ant.AverageEffect("t", "y"),
        bootstrap=0,
        refute="none",
    )
    pin = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "conformance"
            / "estimate"
            / "temporal_class_envelope"
            / "identified_pag.json"
        ).read_text()
    )
    n = int(pin["n"])
    i = np.arange(n, dtype=float)
    z = np.sin(0.37 * i) + 0.5 * np.cos(1.3 * i)
    series_t = 0.6 * z + 0.8 * np.sin(0.23 * i + 0.4)
    v = 0.5 * series_t + np.cos(0.41 * i)
    m = 0.7 * z + 0.6 * np.cos(0.29 * i + 0.2)
    series_y = np.zeros(n)
    series_y[1:] = 1.0 + 2.0 * series_t[:-1] + 1.5 * m[:-1] + 0.3 * np.sin(0.53 * i[1:])
    pag = ant.analyze(
        {"t": series_t, "y": series_y, "z": z, "m": m, "v": v},
        graph=ant.graph.TemporalPag.from_marked_lagged_edges(
            pin["columns"], [tuple(edge) for edge in pin["marked_edges"]]
        ),
        query=ant.PulseEffect("t", "y", treatment_lag=1, horizon_steps=1),
        bootstrap=0,
        refute="none",
    )
    for result in (dag, admg, cpdag, pag):
        certificate = result.certificate
        assert isinstance(certificate, dict), type(result)
        _assert_schema(certificate)
