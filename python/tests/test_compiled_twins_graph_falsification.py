"""The direct graph falsification warns and agrees with the compiled falsification of a study."""

from __future__ import annotations

import antecedent as ant
import numpy as np
import pytest
from antecedent._deprecation import AntecedentDeprecationWarning
from antecedent.errors import CausalTypeError
from antecedent.graph_falsification import evaluate_graph, falsify_study_graph

EDGES = [("a", "b"), ("b", "c"), ("c", "d")]
SETTINGS = {"n_permutations": 40, "min_permutations": 10, "ci_replicates": 49}


def _data(n: int = 400, seed: int = 3) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = 0.9 * a + rng.normal(scale=0.6, size=n)
    c = -0.7 * b + rng.normal(scale=0.6, size=n)
    d = 1.2 * c + rng.normal(scale=0.6, size=n)
    return {"a": a, "b": b, "c": c, "d": d}


def test_evaluate_graph_warns_and_equals_the_compiled_falsification_of_the_study_graph() -> None:
    data = _data()
    execution = ant.Execution(seed=5)
    with pytest.warns(
        AntecedentDeprecationWarning, match="antecedent.graph_falsification.evaluate_graph"
    ):
        direct = evaluate_graph(data, EDGES, execution=execution, **SETTINGS)
    study = ant.prepare(
        data,
        graph=EDGES,
        query=ant.AverageEffect(treatment="a", outcome="d"),
        execution=execution,
    )
    compiled = falsify_study_graph(study, **SETTINGS)
    assert compiled == direct
    assert compiled.n_implications > 0


def test_falsify_study_graph_refuses_what_is_not_a_prepared_study() -> None:
    with pytest.raises(CausalTypeError, match="PreparedAnalysis"):
        falsify_study_graph(object())
