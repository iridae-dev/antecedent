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
from antecedent.outcomes import require_success


def _bernoulli(seed: int = 0, n: int = 400):
    rng = np.random.default_rng(seed)
    assign = rng.random(n) < 0.5
    outcome = assign.astype(float) + rng.normal(size=n)
    ids = np.array([f"u{i}" for i in range(n)])
    return assign, outcome, ids


def test_randomized_effect_column_names_match_inline_arrays():
    assign, outcome, ids = _bernoulli()
    inline = require_success(
        ant.analyze(
            {"y": outcome},
            query=ant.experiment.RandomizedEffect(
                "y",
                ant.experiment.ExperimentDesign(
                    BernoulliAssignment(0.5), [bool(v) for v in assign], list(ids), list(ids)
                ),
            ),
        )
    )
    named = require_success(
        ant.analyze(
            {"y": outcome, "arm": assign, "id": ids},
            query=ant.experiment.RandomizedEffect(
                "y", ant.experiment.ExperimentDesign(BernoulliAssignment(0.5), "arm", "id", "id")
            ),
        )
    )
    assert named.answer.value == pytest.approx(inline.answer.value)
    # Independent value: Horvitz-Thompson contrast under Bernoulli(0.5) assignment.
    ht = float(np.mean(outcome * assign / 0.5) - np.mean(outcome * ~assign / 0.5))
    assert named.answer.value == pytest.approx(ht, abs=1e-9)
    # The string unit column is dropped before the numeric ingest, not rejected.
    assert named.answer.kind == "point"


def test_column_name_that_is_absent_is_a_typed_error():
    _, outcome, ids = _bernoulli()
    query = ant.experiment.RandomizedEffect(
        "y", ant.experiment.ExperimentDesign(BernoulliAssignment(0.5), "missing", "id", "id")
    )
    with pytest.raises(CausalValueError, match="not in the data"):
        require_success(ant.analyze({"y": outcome, "id": ids}, query=query))


def _randomized_named_query():
    return ant.experiment.RandomizedEffect(
        "y", ant.experiment.ExperimentDesign(BernoulliAssignment(0.5), "arm", "id", "id")
    )


@pytest.mark.parametrize(
    "encode",
    [
        pytest.param(lambda a: np.where(np.arange(len(a)) < 40, np.nan, a.astype(float)), id="nan"),
        pytest.param(lambda a: np.where(a, 1 + np.arange(len(a)) % 2, 0), id="arm-codes-0-1-2"),
        pytest.param(lambda a: np.where(a, "True", "False"), id="bool-strings"),
        pytest.param(lambda a: np.where(a, "treated", "control"), id="labels"),
    ],
)
def test_bool_column_accepts_only_bool_or_exact_zero_one(encode):
    assign, outcome, ids = _bernoulli()
    with pytest.raises(CausalValueError, match="'arm'.*bool or encoded as 0/1"):
        require_success(
            ant.analyze(
                {"y": outcome, "arm": encode(assign), "id": ids}, query=_randomized_named_query()
            )
        )


def test_bool_column_encoded_as_zero_one_matches_bool_column():
    assign, outcome, ids = _bernoulli()
    as_bool = require_success(
        ant.analyze({"y": outcome, "arm": assign, "id": ids}, query=_randomized_named_query())
    )
    as_int = require_success(
        ant.analyze(
            {"y": outcome, "arm": assign.astype(int), "id": ids}, query=_randomized_named_query()
        )
    )
    as_float = require_success(
        ant.analyze(
            {"y": outcome, "arm": assign.astype(float), "id": ids}, query=_randomized_named_query()
        )
    )
    assert as_int.answer.value == pytest.approx(as_bool.answer.value)
    assert as_float.answer.value == pytest.approx(as_bool.answer.value)


@pytest.mark.parametrize(
    ("folds", "match"),
    [
        pytest.param([0.0, 1.0, 2.7, 1.0], "finite integers", id="fractional"),
        pytest.param([0.0, 1.0, np.nan, 1.0], "finite integers", id="nan"),
        pytest.param(["0", "1", "0", "1"], "must be integers", id="strings"),
    ],
)
def test_int_column_requires_finite_integral_values(folds, match):
    from antecedent._columns import resolve_query

    query = _policy.PolicyValue(
        "y",
        "asg",
        0.5,
        _policy.BinaryPolicy([True, False] * 2),
        [f"e{i}" for i in range(4)],
        fold_ids="fold",
    )
    data = {"y": [1.0, 2.0, 3.0, 4.0], "asg": [True, False] * 2, "fold": folds}
    with pytest.raises(CausalValueError, match=f"fold_ids='fold'.*{match}"):
        resolve_query(query, data)


def test_refresh_reuses_the_frozen_design_with_outcome_only_data():
    assign, outcome, ids = _bernoulli()
    result = require_success(
        ant.analyze(
            {"y": outcome, "arm": assign, "id": ids},
            query=ant.experiment.RandomizedEffect(
                "y", ant.experiment.ExperimentDesign(BernoulliAssignment(0.5), "arm", "id", "id")
            ),
        )
    )
    rng = np.random.default_rng(1)
    updated = require_success(
        result.study.rebind(
            data={"y": assign.astype(float) + rng.normal(size=len(assign))}
        ).estimate()
    )
    assert updated.answer.kind == "point"
    assert updated.program_id == result.program_id


def test_refresh_accepts_the_table_used_at_analyze():
    import pandas as pd

    assign, outcome, ids = _bernoulli()
    table = {"y": outcome, "arm": assign, "id": ids}
    for data in (table, pd.DataFrame(table)):
        result = require_success(ant.analyze(data, query=_randomized_named_query()))
        refreshed = require_success(result.study.rebind(data=data).estimate())
        assert refreshed.answer.value == pytest.approx(result.answer.value)
        outcome_only = require_success(result.study.rebind(data={"y": outcome}).estimate())
        assert outcome_only.answer.value == pytest.approx(result.answer.value)


def test_refresh_checks_named_design_columns_without_rebinding_them():
    assign, outcome, ids = _bernoulli()
    result = require_success(
        ant.analyze({"y": outcome, "arm": assign, "id": ids}, query=_randomized_named_query())
    )
    shifted = outcome + 5.0
    original_inspection = result.study.inspect()
    original_export = bytes(result.export())
    frozen = require_success(result.study.rebind(data={"y": shifted}).estimate())
    with pytest.raises(ant.errors.CausalValueError, match="frozen design column.*arm"):
        require_success(
            result.study.rebind(data={"y": shifted, "arm": ~assign, "id": ids}).estimate()
        )
    assert result.study.inspect() == original_inspection
    assert bytes(result.export()) == original_export
    assert require_success(result.study.estimate()).answer == result.answer
    rebound = require_success(
        ant.prepare(
            {"y": shifted, "arm": ~assign, "id": ids}, query=_randomized_named_query()
        ).estimate()
    )
    assert rebound.answer.value == pytest.approx(-frozen.answer.value)


def test_named_design_columns_are_dropped_from_an_arrow_table():
    pa = pytest.importorskip("pyarrow")
    from antecedent._columns import drop_columns

    assign, outcome, ids = _bernoulli()
    table = pa.table({"y": outcome, "arm": assign, "id": ids})
    assert drop_columns(table, {"arm", "id"}).column_names == ["y"]
    named = require_success(ant.analyze(table, query=_randomized_named_query()))
    inline = require_success(
        ant.analyze({"y": outcome, "arm": assign, "id": ids}, query=_randomized_named_query())
    )
    assert named.answer.value == pytest.approx(inline.answer.value)


# --- Adopted classes: array vs column-name equality (see antecedent._columns) ---

from antecedent import interference as _interference  # noqa: E402
from antecedent import policy as _policy  # noqa: E402
from antecedent.regimes import LongitudinalRegime  # noqa: E402


def test_policy_value_column_names_match_inline_arrays():
    n = 8
    assignment = [True, False] * 4
    outcome = 5.0 + 2.0 * np.asarray(assignment, dtype=float)
    actions = [True] * 4 + [False] * 4
    ids = [f"e{i}" for i in range(n)]
    inline = require_success(
        ant.analyze(
            {"y": outcome},
            query=_policy.PolicyValue("y", assignment, 0.5, _policy.BinaryPolicy(actions), ids),
            refute="none",
        )
    )
    # assignment, propensity (per-row), evaluation ids, and the nested BinaryPolicy
    # actions all named by data column; string ids ride in the data and are dropped.
    named = require_success(
        ant.analyze(
            {"y": outcome, "asg": assignment, "prop": [0.5] * n, "act": actions, "sid": ids},
            query=_policy.PolicyValue("y", "asg", "prop", _policy.BinaryPolicy("act"), "sid"),
            refute="none",
        )
    )
    assert named.policy_value.policy_value == pytest.approx(inline.policy_value.policy_value)
    assert named.policy_value.incremental_value == pytest.approx(
        inline.policy_value.incremental_value
    )


def test_multi_action_policy_value_column_names_match_inline_arrays():
    labels = ("control", "A", "B")
    assigned = ["control", "A", "B"] * 3
    outcomes = [1.0, 2.0, 4.0] * 3
    recommendations = ["control"] * 3 + ["A"] * 3 + ["B"] * 3
    ids = [f"s{i}" for i in range(9)]
    props = [[1 / 3] * 3 for _ in assigned]
    inline = require_success(
        ant.analyze(
            {"y": outcomes},
            query=_policy.MultiActionPolicyValue(
                outcome="y",
                assignment=assigned,
                propensities=props,
                policy=_policy.MultiActionPolicy(labels, recommendations, capacities=[9, 3, 3]),
                evaluation_subject_ids=ids,
            ),
            refute="none",
        )
    )
    # assignment and evaluation ids, plus the nested MultiActionPolicy recommendations.
    named = require_success(
        ant.analyze(
            {"y": outcomes, "asg": assigned, "rec": recommendations, "sid": ids},
            query=_policy.MultiActionPolicyValue(
                outcome="y",
                assignment="asg",
                propensities=props,
                policy=_policy.MultiActionPolicy(labels, "rec", capacities=[9, 3, 3]),
                evaluation_subject_ids="sid",
            ),
            refute="none",
        )
    )
    assert named.policy_value.policy_value == pytest.approx(inline.policy_value.policy_value)


def test_complier_effect_column_names_match_inline_arrays():
    assignment = [False, True] * 8
    receipt = [True] * 4 + [False] * 8 + [False, True, False, True]
    outcome = 4.0 * np.asarray(receipt, dtype=float)
    units = [f"u{i}" for i in range(16)]
    rows = [f"r{i}" for i in range(16)]
    inline = require_success(
        ant.analyze(
            {"outcome": outcome},
            query=ant.experiment.ComplierEffect(
                "outcome",
                ant.experiment.ExperimentDesign(BernoulliAssignment(0.5), assignment, units, rows),
                receipt,
            ),
            refute="none",
        )
    )
    named = require_success(
        ant.analyze(
            {"outcome": outcome, "asg": assignment, "au": units, "ou": rows, "rcv": receipt},
            query=ant.experiment.ComplierEffect(
                "outcome",
                ant.experiment.ExperimentDesign(BernoulliAssignment(0.5), "asg", "au", "ou"),
                "rcv",
            ),
            refute="none",
        )
    )
    assert named.randomized_effect.effect == pytest.approx(inline.randomized_effect.effect)
    assert named.randomized_effect.intention_to_treat_effect == pytest.approx(
        inline.randomized_effect.intention_to_treat_effect
    )


def test_switchback_column_names_match_inline_arrays():
    sequences = [f"s{s}" for s in range(4) for _ in range(4)]
    periods = [f"p{p}" for _ in range(4) for p in range(4)]
    assignment = [True, False, False, True] * 4
    outcomes = [
        10.0 * s + (2.0 if t else 0.0) for s in range(4) for t in (True, False, False, True)
    ]
    inline = require_success(
        ant.analyze(
            {"y": outcomes},
            query=ant.experiment.SwitchbackEffect(
                "y",
                ant.experiment.SwitchbackDesign(
                    assignment, sequences, periods, [0.5] * 16, ("off", "on")
                ),
            ),
            refute="none",
        )
    )
    named = require_success(
        ant.analyze(
            {"y": outcomes, "a": assignment, "s": sequences, "p": periods, "pr": [0.5] * 16},
            query=ant.experiment.SwitchbackEffect(
                "y",
                ant.experiment.SwitchbackDesign("a", "s", "p", "pr", ("off", "on")),
            ),
            refute="none",
        )
    )
    assert named.answer.value == pytest.approx(inline.answer.value)


def test_multi_arm_design_column_names_match_inline_arrays():
    labels = ("control", "low", "high")
    assignment = ["control", "low", "high"] * 2
    units = [f"u{i}" for i in range(6)]
    rows = [f"r{i}" for i in range(6)]
    data = {"outcome": [0.0, 2.0, 5.0, 0.0, 2.0, 5.0]}
    inline = require_success(
        ant.analyze(
            data,
            query=ant.experiment.RandomizedEffect(
                "outcome",
                ant.experiment.MultiArmExperimentDesign(
                    assignment, labels, [[1 / 3] * 3] * 6, units, rows
                ),
            ),
            refute="none",
        )
    )
    named = require_success(
        ant.analyze(
            {**data, "asg": assignment, "au": units, "ou": rows},
            query=ant.experiment.RandomizedEffect(
                "outcome",
                ant.experiment.MultiArmExperimentDesign(
                    "asg", labels, [[1 / 3] * 3] * 6, "au", "ou"
                ),
            ),
            refute="none",
        )
    )
    assert named.answer.value == pytest.approx(inline.answer.value)


def test_factorial_second_factor_column_name_matches_inline_array():
    primary = [True, False] * 4
    second = [True, True, False, False] * 2
    counts = [0, 0, 0, 0]
    for prime, sec in zip(primary, second, strict=True):
        counts[int(prime) + 2 * int(sec)] += 1
    units = [f"u{i}" for i in range(8)]
    rows = [f"r{i}" for i in range(8)]
    outcome = 2.0 * np.asarray(primary, dtype=float) + np.asarray(second, dtype=float)
    inline = require_success(
        ant.analyze(
            {"y": outcome},
            query=ant.experiment.RandomizedEffect(
                "y",
                ant.experiment.ExperimentDesign(
                    ant.experiment.FactorialRandomization(second, tuple(counts)),
                    primary,
                    units,
                    rows,
                ),
            ),
            refute="none",
        )
    )
    # The FactorialRandomization second-factor arm names a data column; it resolves
    # through ExperimentDesign's nested-assignment recursion.
    named = require_success(
        ant.analyze(
            {"y": outcome, "sf": second},
            query=ant.experiment.RandomizedEffect(
                "y",
                ant.experiment.ExperimentDesign(
                    ant.experiment.FactorialRandomization("sf", tuple(counts)),
                    primary,
                    units,
                    rows,
                ),
            ),
            refute="none",
        )
    )
    assert named.answer.value == pytest.approx(inline.answer.value)


def test_interference_query_column_name_matches_inline_array():
    clusters = 80
    edges = [
        (first + s, first + t)
        for first in range(0, 3 * clusters, 3)
        for s in range(3)
        for t in range(3)
        if s != t
    ]
    rng = np.random.default_rng(0)
    assignment = [bool(v) for v in (rng.random(3 * clusters) < 0.5)]
    outcome = np.array([1.0 + 2.0 * a for a in assignment])
    contrast = _interference.ExposureContrast(
        "y", _interference.ExposureLevel(0.0, 0.0), _interference.ExposureLevel(1.0, 0.0)
    )
    inline = require_success(
        ant.analyze(
            {"y": outcome},
            graph=[],
            query=_interference.InterferenceQuery(
                _interference.BernoulliAssignment(0.5),
                _interference.NeighborCount(),
                contrast,
                network=edges,
                realized_assignment=assignment,
            ),
        )
    )
    named = require_success(
        ant.analyze(
            {"y": outcome, "asg": assignment},
            graph=[],
            query=_interference.InterferenceQuery(
                _interference.BernoulliAssignment(0.5),
                _interference.NeighborCount(),
                contrast,
                network=edges,
                realized_assignment="asg",
            ),
        )
    )
    assert named.interference.contrast.horvitz_thompson == pytest.approx(
        inline.interference.contrast.horvitz_thompson
    )


def test_longitudinal_regime_column_names_match_inline_arrays():
    history = [[True, True], [True, False], [False, True], [False, False]]
    outcome = [4.0, 0.0, 0.0, 0.0]
    ids = ["a", "b", "c", "d"]
    inline = require_success(
        ant.analyze(
            {"y": outcome},
            query=LongitudinalRegime(
                outcome="y",
                treatment_history=history,
                actions=[True, True],
                treatment_probabilities=[[0.5, 0.5]] * 4,
                subject_ids=ids,
            ),
        )
    )
    named = require_success(
        ant.analyze(
            {"y": outcome, "id": ids},
            query=LongitudinalRegime(
                outcome="y",
                treatment_history=history,
                actions=[True, True],
                treatment_probabilities=[[0.5, 0.5]] * 4,
                subject_ids="id",
            ),
        )
    )
    assert named.longitudinal_regime.value == pytest.approx(inline.longitudinal_regime.value)
