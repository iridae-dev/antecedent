"""Deprecated learner wrappers and the compiled study agree on recorded fits.

The compatibility wrappers retain their warning and result fields. The common compiled
route supplies the configured estimator's effect, rows and fold receipts.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from antecedent._deprecation import AntecedentDeprecationWarning
from antecedent.extensibility import ProviderRegistry
from antecedent.external_learners import (
    ExternalLearner,
    Learners,
    aipw_ate,
    analyze_ate,
    causal_forest_ate,
    cross_fit_aipw,
    dml_ate,
    dr_learner_ate,
)

from test_external_learners import MeanProvider

GRAPH = [("z", "t"), ("z", "y"), ("t", "y")]


def _learners():
    registry = ProviderRegistry()
    registry.register("outcome", MeanProvider())
    registry.register("treatment", MeanProvider())
    outcome = ExternalLearner(registry, "outcome", "regression", supports_weights=True)
    treatment = ExternalLearner(registry, "treatment", "binary_probability", supports_weights=True)
    return outcome, treatment


def _data(n: int = 240):
    arms = [float(i % 2) for i in range(n)]
    z = [float((i * 7) % 11) for i in range(n)]
    y = [
        1.0 + 2.0 * a + 0.3 * zi + 0.1 * ((i * 5) % 3)
        for i, (a, zi) in enumerate(zip(arms, z, strict=True))
    ]
    return {"z": z, "t": arms, "y": y}


def _assert_same(direct, compiled):
    assert compiled.estimator == direct.estimator
    assert compiled.score == direct.score
    assert compiled.fold_policy is direct.fold_policy
    assert direct.propensity is None
    assert compiled.assumptions == direct.assumptions
    assert compiled.estimate == direct.estimate
    assert compiled.standard_error == direct.standard_error
    assert compiled.row_ids == direct.row_ids
    assert compiled.fold_ids == direct.fold_ids
    assert compiled.adjustment_set == direct.adjustment_set
    assert compiled.provenance == direct.provenance
    assert compiled.cate == direct.cate
    assert compiled.cate_standard_errors == direct.cate_standard_errors
    assert len(compiled.receipts) == len(direct.receipts)
    assert any("dml.caller_factory.unverified" in str(item) for item in compiled.assumptions)


@pytest.mark.parametrize(
    ("estimator", "score", "folds"),
    [("dml", "aipw", 2), ("dml", "partially_linear", 2), ("dr_learner", "aipw", 2)],
)
def test_dml_ate_warns_and_matches_the_declared_learner_study(estimator, score, folds):
    outcome, treatment = _learners()
    data = _data()
    with pytest.warns(AntecedentDeprecationWarning, match="dml_ate"):
        direct = dml_ate(
            data,
            GRAPH,
            "t",
            "y",
            outcome=outcome,
            treatment=treatment,
            folds=folds,
            estimator=estimator,
            score=score,
        )
    with warnings.catch_warnings():
        warnings.simplefilter("error", AntecedentDeprecationWarning)
        compiled = analyze_ate(
            data,
            GRAPH,
            "t",
            "y",
            learners=Learners(outcome, treatment),
            folds=folds,
            estimator=estimator,
            score=score,
        )
    _assert_same(direct, compiled)


def test_aipw_ate_warns_and_matches_the_declared_learner_study():
    outcome, treatment = _learners()
    data = _data()
    with pytest.warns(AntecedentDeprecationWarning, match="aipw_ate"):
        direct = aipw_ate(data, GRAPH, "t", "y", outcome=outcome, treatment=treatment)
    compiled = analyze_ate(
        data, GRAPH, "t", "y", learners=Learners(outcome, treatment), estimator="aipw"
    )
    _assert_same(direct, compiled)
    assert compiled.estimate == pytest.approx(2.0, abs=0.5)


def test_dr_learner_and_causal_forest_entries_warn_and_match_the_declared_learner_study():
    outcome, treatment = _learners()
    data = _data()
    with pytest.warns(AntecedentDeprecationWarning, match="dr_learner_ate"):
        dr = dr_learner_ate(data, GRAPH, "t", "y", outcome=outcome, treatment=treatment, folds=2)
    _assert_same(
        dr,
        analyze_ate(
            data,
            GRAPH,
            "t",
            "y",
            learners=Learners(outcome, treatment),
            folds=2,
            estimator="dr_learner",
        ),
    )
    with pytest.warns(AntecedentDeprecationWarning, match="causal_forest_ate"):
        forest = causal_forest_ate(data, GRAPH, "t", "y", outcome=outcome, treatment=treatment)
    _assert_same(
        forest,
        analyze_ate(
            data, GRAPH, "t", "y", learners=Learners(outcome, treatment), estimator="causal_forest"
        ),
    )


def test_direct_entry_still_returns_its_numbers_after_the_warning():
    outcome, treatment = _learners()
    data = _data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AntecedentDeprecationWarning)
        first = dml_ate(data, GRAPH, "t", "y", outcome=outcome, treatment=treatment, folds=2)
        second = dml_ate(data, GRAPH, "t", "y", outcome=outcome, treatment=treatment, folds=2)
    assert first.estimate == second.estimate
    assert first.standard_error == second.standard_error
    assert first.propensity is None


def test_cross_fit_aipw_warns_and_matches_the_compiled_propensities():
    outcome, treatment = _learners()
    data = _data()
    compiled = analyze_ate(data, GRAPH, "t", "y", learners=Learners(outcome, treatment), folds=2)
    assert compiled.propensity is not None
    rows = list(compiled.row_ids)
    z = np.asarray(data["z"])[rows]
    design = np.column_stack([np.ones(len(rows)), z])
    with pytest.warns(AntecedentDeprecationWarning, match="cross_fit_aipw"):
        direct = cross_fit_aipw(
            outcome,
            treatment,
            design,
            np.asarray(data["y"])[rows],
            np.asarray(data["t"])[rows],
            np.asarray(compiled.fold_ids, dtype=np.int64),
        )
    assert direct.propensity == compiled.propensity
    assert direct.fold_ids == compiled.fold_ids


def test_learners_require_external_learners():
    with pytest.raises(ValueError, match="ExternalLearner"):
        Learners(object(), object())  # type: ignore[arg-type]
    outcome, treatment = _learners()
    with pytest.raises(ValueError, match="Learners"):
        analyze_ate(_data(), GRAPH, "t", "y", learners=(outcome, treatment))  # type: ignore[arg-type]
    data = _data()
    for wrapper in (dml_ate, aipw_ate, dr_learner_ate, causal_forest_ate):
        with (
            pytest.warns(AntecedentDeprecationWarning, match=wrapper.__name__),
            pytest.raises(ValueError, match="ExternalLearner"),
        ):
            wrapper(data, GRAPH, "t", "y", outcome=object(), treatment=treatment)
