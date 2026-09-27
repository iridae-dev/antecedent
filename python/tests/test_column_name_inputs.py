"""Design queries accept column names for their row-aligned inputs.

A design-carrying query may name a row-aligned input by the data column that holds
it instead of passing the array inline. The column is read from the raw data (any
dtype, so string unit ids work) and dropped before the numeric ingest, so the
result is identical to the inline-array spelling. See antecedent._columns.
"""

from __future__ import annotations

import antecedent as ant
import numpy as np
import pytest
from antecedent.errors import CausalValueError
from antecedent.interference import BernoulliAssignment


def _bernoulli(seed: int = 0, n: int = 400):
    rng = np.random.default_rng(seed)
    assign = rng.random(n) < 0.5
    outcome = assign.astype(float) + rng.normal(size=n)
    ids = np.array([f"u{i}" for i in range(n)])
    return assign, outcome, ids


def test_randomized_effect_column_names_match_inline_arrays():
    assign, outcome, ids = _bernoulli()
    inline = ant.analyze(
        {"y": outcome},
        query=ant.experiment.RandomizedEffect(
            "y",
            ant.experiment.ExperimentDesign(
                BernoulliAssignment(0.5), [bool(v) for v in assign], list(ids), list(ids)
            ),
        ),
    )
    named = ant.analyze(
        {"y": outcome, "arm": assign, "id": ids},
        query=ant.experiment.RandomizedEffect(
            "y", ant.experiment.ExperimentDesign(BernoulliAssignment(0.5), "arm", "id", "id")
        ),
    )
    assert named.answer.value == pytest.approx(inline.answer.value)
    # The string unit column is dropped before the numeric ingest, not rejected.
    assert named.answer.kind == "point"


def test_column_name_that_is_absent_is_a_typed_error():
    _, outcome, ids = _bernoulli()
    query = ant.experiment.RandomizedEffect(
        "y", ant.experiment.ExperimentDesign(BernoulliAssignment(0.5), "missing", "id", "id")
    )
    with pytest.raises(CausalValueError, match="not in the data"):
        ant.analyze({"y": outcome, "id": ids}, query=query)


def test_refresh_reuses_the_frozen_design_with_outcome_only_data():
    assign, outcome, ids = _bernoulli()
    result = ant.analyze(
        {"y": outcome, "arm": assign, "id": ids},
        query=ant.experiment.RandomizedEffect(
            "y", ant.experiment.ExperimentDesign(BernoulliAssignment(0.5), "arm", "id", "id")
        ),
    )
    rng = np.random.default_rng(1)
    updated = result.study.refresh({"y": assign.astype(float) + rng.normal(size=len(assign))})
    assert updated.answer.kind == "point"
    assert updated.program_id == result.program_id
