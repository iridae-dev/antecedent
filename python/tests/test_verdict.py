"""One identification renderer."""

from __future__ import annotations

from pathlib import Path

import antecedent as ant
import numpy as np
from antecedent._native import identification_status_names
from antecedent._verdict import VERDICTS, verdict_for
from antecedent.outcomes import require_success

from _repo_text import REPO_ROOT, load_json

SCHEMA = load_json(Path(__file__).resolve().parent / "fixtures" / "certificate_schema.json")


def _assert_schema(instance: dict) -> None:
    required = SCHEMA["required"]
    props = set(SCHEMA["properties"])
    extra = set(instance) - props
    assert not extra, extra
    for key in required:
        assert key in instance, key


def test_verdict_table_is_exhaustive():
    # DesignBased is a facade status; NotClaimed is the host's domain status for a noncausal
    # association. Neither is a native IdentificationStatus, so both are listed by name.
    assert set(VERDICTS) == set(identification_status_names()) | {"DesignBased", "NotClaimed"}


def test_statement_lines_nonempty_when_assumptions_exist():
    identification = ant.identify(
        graph=[("z", "t"), ("z", "y"), ("t", "y")],
        query=ant.AverageEffect("t", "y"),
        names=["t", "y", "z"],
    )
    # z is the common parent of t and y, so the backdoor adjustment on z identifies the effect
    # nonparametrically; the renderer must say so rather than hedge.
    assert identification.status == "NonparametricallyIdentified"
    assert identification.verdict == "identified"
    assert identification.verdict == verdict_for(identification.status)
    assert identification.statement == (
        "AverageEffect of y from t is identified by backdoor.adjustment, adjusting for z."
    )
    # A partial or unidentified status never renders as an identified claim.
    assert verdict_for("NotIdentified") == "not identified"
    assert verdict_for("PartiallyIdentified") == "partially identified"
    assert verdict_for("GraphDependent") == "graph-dependent"
    assert verdict_for("NotClaimed") == "not claimed"


def test_certificate_matches_schema():
    rng = np.random.default_rng(3)
    z = rng.normal(size=96)
    t = (rng.uniform(size=96) < 0.5).astype(float)
    y = t + z + rng.normal(scale=0.2, size=96)
    data = {"t": t, "y": y, "z": z}

    dag = require_success(
        ant.analyze(
            data,
            graph=[("z", "t"), ("z", "y"), ("t", "y")],
            query=ant.AverageEffect("t", "y"),
            inference=ant.Frequentist(bootstrap=0),
            refute="none",
        )
    )
    admg = require_success(
        ant.analyze(
            data,
            graph=ant.Admg.from_edges(
                ["t", "y", "z"], directed=[("t", "y"), ("z", "t"), ("z", "y")]
            ),
            query=ant.AverageEffect("t", "y"),
            inference=ant.Frequentist(bootstrap=0),
            refute="none",
        )
    )
    cpdag = require_success(
        ant.analyze(
            data,
            graph=ant.Cpdag.from_directed_undirected(
                ["t", "y", "z"], directed=[("z", "t"), ("z", "y"), ("t", "y")], undirected=[]
            ),
            query=ant.AverageEffect("t", "y"),
            inference=ant.Frequentist(bootstrap=0),
            refute="none",
        )
    )
    pin = load_json(
        REPO_ROOT / "conformance" / "estimate" / "temporal_class_envelope" / "identified_pag.json"
    )
    n = int(pin["n"])
    i = np.arange(n, dtype=float)
    z = np.sin(0.37 * i) + 0.5 * np.cos(1.3 * i)
    series_t = 0.6 * z + 0.8 * np.sin(0.23 * i + 0.4)
    v = 0.5 * series_t + np.cos(0.41 * i)
    m = 0.7 * z + 0.6 * np.cos(0.29 * i + 0.2)
    series_y = np.zeros(n)
    series_y[1:] = 1.0 + 2.0 * series_t[:-1] + 1.5 * m[:-1] + 0.3 * np.sin(0.53 * i[1:])
    pag = require_success(
        ant.analyze(
            {"t": series_t, "y": series_y, "z": z, "m": m, "v": v},
            graph=ant.graph.TemporalPag.from_marked_lagged_edges(
                pin["columns"], [tuple(edge) for edge in pin["marked_edges"]]
            ),
            query=ant.PulseEffect("t", "y", treatment_lag=1, horizon_steps=1),
            inference=ant.Frequentist(bootstrap=0),
            refute="none",
        )
    )
    expected = [
        (dag, "point", "NonparametricallyIdentified", "backdoor.adjustment", "Dag"),
        (admg, "point", "NonparametricallyIdentified", "backdoor.adjustment", "Admg"),
        (cpdag, "cpdag_envelope", "NonparametricallyIdentified", "generalized.adjustment", "Cpdag"),
        (pag, "temporal_envelope", pin["identification"]["status"], pin["identification"]["identifier"], "TemporalPag"),
    ]
    for result, kind, status, method, graph_class in expected:
        certificate = result.to_dict()["certificate"]
        assert isinstance(certificate, dict), type(result)
        _assert_schema(certificate)
        assert (certificate["kind"], certificate["status"], certificate["method"]) == (
            kind,
            status,
            method,
        )
        assert certificate["graph_class"] == graph_class
        assert certificate["outcome"]["name"] == "y"
        assert [t["name"] for t in certificate["treatments"]] == ["t"]
    # The pinned multi-completion PAG: seven completions, six identified and one not.
    certificate = pag.to_dict()["certificate"]
    assert len(certificate["cases"]) == pin["identification"]["completion_count"]
    assert certificate["identified_weight"] == pin["identification"]["identified_mass"]
    assert certificate["unidentified_weight"] == pin["identification"]["unidentified_mass"]
