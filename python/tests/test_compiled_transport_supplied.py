"""Caller-supplied transport nuisances through the compiled routes, from Python.

* ``TransportQuery(..., outcome_regressions=(mu0, mu1))`` on ``antecedent.analyze`` publishes
  the IPW point of the trial-to-target cell and states the augmented contrast that the
  deprecated ``transport.estimate_trial_effect(..., mu0=, mu1=)`` returns, exactly.
* ``transport.advanced.compile_augmented_response_grid`` compiles the caller's augmented
  response equation into the ``TransportResponseGrid`` route; its grid equals the deprecated
  ``estimate_augmented_response_grid`` and the means worked by hand from the formula.
"""

from __future__ import annotations

import re

import antecedent as ant
import numpy as np
import pytest
from antecedent._deprecation import AntecedentDeprecationWarning
from antecedent.errors import CausalValueError
from antecedent.outcomes import require_success
from antecedent.transport import advanced as transport

NAMES = ["a", "y", "trial", "s", "e", "x", "mu0", "mu1"]
AUGMENTATION = "estimate.transport.trial_aipw_supplied"


def _data(n: int = 400, seed: int = 5) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    s = 1.0 / (1.0 + np.exp(-0.5 * x))
    trial = rng.uniform(size=n) < s
    a = np.where(trial, (rng.uniform(size=n) < 0.5).astype(float), 0.0)
    y = np.where(trial, 1.0 + a * (1.0 + x) + x + rng.normal(size=n), 0.0)
    return {
        "a": a,
        "y": y,
        "trial": trial.astype(float),
        "s": s,
        "e": np.full(n, 0.5),
        "x": x,
        "mu0": 1.0 + x,
        "mu1": 2.0 + 2.0 * x,
    }


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


def _stated_aipw(diagnostics: object) -> float:
    for line in diagnostics:  # type: ignore[attr-defined]
        if str(line).startswith(AUGMENTATION):
            match = re.search(r"aipw=([^ ;]+)", str(line))
            assert match is not None, line
            return float(match.group(1))
    raise AssertionError(f"no {AUGMENTATION} diagnostic")


def test_supplied_regressions_on_analyze_state_the_direct_aipw_exactly() -> None:
    data = _data()
    with pytest.warns(AntecedentDeprecationWarning, match="outcome_regressions"):
        direct = transport.estimate_trial_effect(
            _identification(),
            data["a"] != 0.0,
            data["y"],
            data["trial"] != 0.0,
            data["s"],
            data["e"],
            mu0=data["mu0"],
            mu1=data["mu1"],
        )
    study = require_success(
        ant.analyze(data, graph=_graph(), query=_query(outcome_regressions=("mu0", "mu1")))
    )
    assert direct.aipw is not None
    assert study.estimate.ate == direct.ipw
    assert _stated_aipw(study.diagnostics).hex() == direct.aipw.hex()
    plain = require_success(ant.analyze(data, graph=_graph(), query=_query()))
    assert plain.estimate.ate == study.estimate.ate


def test_outcome_regressions_need_two_distinct_columns_and_the_trial_columns() -> None:
    with pytest.raises(CausalValueError):
        _query(outcome_regressions=("mu0", "mu0"))
    with pytest.raises(CausalValueError):
        _query(
            trial=None,
            selection_probability=None,
            treatment_probability=None,
            outcome_regressions=("mu0", "mu1"),
        )


GRID_NAMES = ["a", "y", "trial", "s", "e", "m", "m0", "m1", "w0", "w1"]
GRID_DATA = {
    "a": [0.0, 1.0, 0.0, 1.0],
    "y": [2.0, 4.0, 0.0, 0.0],
    "trial": [1.0, 1.0, 0.0, 0.0],
    "s": [0.5, 0.25, 0.5, 0.5],
    "e": [0.5, 0.5, 0.5, 0.5],
    "m": [1.5, 3.0, 0.0, 0.0],
    "m0": [0.0, 0.0, 1.0, 2.0],
    "m1": [0.0, 0.0, 3.0, 5.0],
    "w0": [1.0, 0.5, 0.0, 0.0],
    "w1": [0.25, 2.0, 0.0, 0.0],
}


def _grid_query(**overrides: object) -> transport.TransportQuery:
    fields: dict[str, object] = {
        "trial": "trial",
        "selection_probability": "s",
        "treatment_probability": "e",
    }
    fields.update(overrides)
    return transport.TransportQuery(
        ant.ResponseCurve("a", "y", grid=[0.0, 1.0]),
        transport.SelectionDiagram("trial", "target", []),
        source_experiments=["a"],
        **fields,  # type: ignore[arg-type]
    )


def _compiled_grid() -> transport.CompiledAugmentedResponseGrid:
    return transport.compile_augmented_response_grid(
        GRID_DATA,
        graph=ant.Admg.from_edges(GRID_NAMES, [("a", "y")]),
        query=_grid_query(),
        observed_regression="m",
        grid_regressions=["m0", "m1"],
        response_weights=["w0", "w1"],
    )


def test_the_compiled_grid_equals_the_hand_worked_means() -> None:
    # Selection odds 1 and 3 on the trial rows, residuals 0.5 and 1:
    # (1 + 2 + 1 * 1 * 0.5 + 3 * 0.5 * 1) / 2 = 2.5 and (3 + 5 + 0.25 * 0.5 + 3 * 2 * 1) / 2.
    compiled = _compiled_grid()
    assert compiled.route == "TransportResponseGrid"
    result = compiled.execute()
    assert result.grid == (0.0, 1.0)
    assert result.mean == (2.5, 7.0625)
    assert result.claim == "point_only"


def test_the_deprecated_grid_warns_and_equals_the_compiled_grid() -> None:
    identification = transport.identify(
        graph=ant.Admg.from_edges(GRID_NAMES, [("a", "y")]),
        query=_grid_query(trial=None, selection_probability=None, treatment_probability=None),
    )
    with pytest.warns(AntecedentDeprecationWarning, match="compile_augmented_response_grid"):
        direct = transport.estimate_augmented_response_grid(
            identification,
            outcome=GRID_DATA["y"],
            trial=[True, True, False, False],
            selection_probability=GRID_DATA["s"],
            grid=[0.0, 1.0],
            observed_regression=GRID_DATA["m"],
            target_regression=[GRID_DATA["m0"], GRID_DATA["m1"]],
            response_weight=[GRID_DATA["w0"], GRID_DATA["w1"]],
        )
    assert _compiled_grid().execute() == direct
