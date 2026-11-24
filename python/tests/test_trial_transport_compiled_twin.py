"""The IPW trial-to-target estimate is one computation, reached directly or through the study.

``transport.estimate_trial_effect`` warns that the compiled study replaces it and returns the same
point estimate the study reports. The AIPW call with ``mu0``/``mu1`` warns too: the study with
``outcome_regressions=`` states the same augmented contrast
(python/tests/test_compiled_transport_supplied.py).
"""

from __future__ import annotations

import warnings

import antecedent as ant
import numpy as np
import pytest
from antecedent._deprecation import AntecedentDeprecationWarning
from antecedent.outcomes import require_success
from antecedent.transport import advanced as transport

NAMES = ["a", "y", "trial", "s", "e", "x"]


def _data(n: int = 300, seed: int = 5) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    s = 1.0 / (1.0 + np.exp(-0.5 * x))
    trial = rng.uniform(size=n) < s
    a = np.where(trial, (rng.uniform(size=n) < 0.5).astype(float), 0.0)
    y = np.where(trial, 1.0 + a * (1.0 + x) + x + rng.normal(size=n), 0.0)
    return {"a": a, "y": y, "trial": trial.astype(float), "s": s, "e": np.full(n, 0.5), "x": x}


def _graph() -> ant.Admg:
    return ant.Admg.from_edges(NAMES, [("a", "y"), ("x", "y")])


def _query(**overrides: object) -> transport.TransportQuery:
    fields: dict[str, object] = {
        "trial": "trial",
        "selection_probability": "s",
        "treatment_probability": "e",
    }
    fields.update(overrides)
    return transport.TransportQuery(
        ant.ResponseCurve("a", "y", grid=[0.0, 1.0]),
        transport.SelectionDiagram("trial", "target", ["x"]),
        source_experiments=["a"],
        **fields,  # type: ignore[arg-type]
    )


def _identification() -> transport.TransportIdentification:
    return transport.identify(
        graph=_graph(),
        query=_query(trial=None, selection_probability=None, treatment_probability=None),
    )


def _direct(data: dict[str, np.ndarray], **extra: object) -> transport.TrialTransportEstimate:
    return transport.estimate_trial_effect(
        _identification(),
        data["a"] != 0.0,
        data["y"],
        data["trial"] != 0.0,
        data["s"],
        data["e"],
        **extra,  # type: ignore[arg-type]
    )


def test_the_ipw_call_warns_and_names_the_compiled_study() -> None:
    with pytest.warns(AntecedentDeprecationWarning, match="estimate_trial_effect") as caught:
        _direct(_data())
    assert "analyze" in str(caught[0].message)


def test_the_deprecated_call_and_the_compiled_study_return_the_same_numbers() -> None:
    data = _data()
    with pytest.warns(AntecedentDeprecationWarning):
        direct = _direct(data)
    study = require_success(ant.analyze(data, graph=_graph(), query=_query()))
    assert direct.ipw == study.estimate.ate
    assert direct.overlap == study.transport_overlap
    assert direct.aipw is None


def test_the_deprecated_call_is_deterministic_across_calls() -> None:
    data = _data(seed=9)
    with pytest.warns(AntecedentDeprecationWarning):
        first = _direct(data)
    with pytest.warns(AntecedentDeprecationWarning):
        second = _direct(data)
    assert first.ipw == second.ipw


def test_the_aipw_call_with_supplied_regressions_warns_and_names_the_regressions() -> None:
    data = _data()
    mu = np.zeros(len(data["y"]))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", AntecedentDeprecationWarning)
        result = _direct(data, mu0=mu, mu1=mu)
    assert result.aipw is not None
    messages = [str(w.message) for w in caught if w.category is AntecedentDeprecationWarning]
    assert messages and "outcome_regressions" in messages[0]
