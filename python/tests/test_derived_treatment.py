"""2.2 E5: derived treatments and factorized joint cells through the Python surface.

Truth is a hand-enumerated joint law: ``P(cell | z)`` is written down as a table, the data
are drawn from it, and ``Y = 0.4 cell + 1{cell = 3} + 0.5 z + 0.3 N(0, 1)`` gives the true
cell means ``E[Y^do(cell)] = 0.4 cell + 1{cell = 3} + 0.25`` and an interaction of exactly 1.
"""

from __future__ import annotations

import numpy as np
import pytest
from antecedent.derived import (
    DeclaredExclusion,
    DerivedTreatment,
    SourceColumn,
    check_derived_treatment,
    factorized_joint_cells,
)
from antecedent.errors import CausalCancelledError, CausalUnsupportedError, CausalValueError
from antecedent.state import CancellationToken

from _refusal import assert_registered_refusal

LAW = [[0.4, 0.2, 0.2, 0.2], [0.1, 0.3, 0.2, 0.4]]
LAW_EMPTY_CELL = [[0.4, 0.3, 0.3, 0.0], [0.2, 0.4, 0.4, 0.0]]


def draw(n, seed, law):
    rng = np.random.default_rng(seed)
    z = (rng.random(n) < 0.5).astype(float)
    cell = np.empty(n, dtype=int)
    for i in range(n):
        cell[i] = rng.choice(4, p=law[int(z[i])])
    y = 0.4 * cell + (cell == 3) + 0.5 * z + 0.3 * rng.standard_normal(n)
    return {"t0": (cell & 1).astype(float), "t1": (cell >> 1).astype(float), "z": z, "y": y}


def components(*names):
    return [
        SourceColumn(name, "treatment_construction", "at_treatment") for name in names
    ]


def declaration(**edits):
    base = {
        "name": "t0_and_t1",
        "sources": [
            *components("t0", "t1"),
            SourceColumn("z", "admissible_pre_treatment_covariate", "pre_treatment"),
            SourceColumn("post", "forbidden_descendant", "post_treatment"),
        ],
        "legal_values": [0.0, 1.0, 2.0, 3.0],
    }
    return DerivedTreatment(**{**base, **edits})


def refusal(call):
    with pytest.raises(CausalUnsupportedError) as caught:
        call()
    assert_registered_refusal(caught.value)
    return caught.value


def truth(cell):
    return 0.4 * cell + float(cell == 3) + 0.25


def test_a_declared_joint_cell_recovers_the_known_law_and_the_interaction():
    data = draw(6000, 1, LAW)
    result = factorized_joint_cells(
        data, declaration(), outcome="y", adjustment=["z"], seed=3, contrasts=["interaction"]
    )
    assert result.claim == "point_only"
    assert result.treatments == ("t0", "t1")
    assert len(result.supported) == 4 and not result.unsupported
    for cell in result.cells:
        assert cell.estimate == pytest.approx(truth(cell.cell), abs=0.08)
        assert cell.ess > 100 and 0 < cell.propensity_min < cell.propensity_max < 1
    assert result.cell(1, 1).estimate == pytest.approx(truth(3), abs=0.08)
    (interaction,) = result.contrasts
    assert interaction.value == pytest.approx(1.0, abs=0.15)
    by_hand = (
        result.cells[0].estimate
        - result.cells[1].estimate
        - result.cells[2].estimate
        + result.cells[3].estimate
    )
    assert interaction.value == pytest.approx(by_hand, abs=1e-9)
    assert all(n.max_abs_error < 1e-9 for n in result.normalization)
    assert {n.ordering for n in result.normalization} == {("t0", "t1"), ("t1", "t0")}
    assert result.sensitivity.orderings == (("t0", "t1"), ("t1", "t0"))
    assert not result.sensitivity.disagreement
    assert "propensity=ridge_logistic" in result.provenance
    assert result.plan.observed_levels == (0.0, 1.0, 2.0, 3.0)
    assert result.plan.retained_covariates == ("z",)


def test_a_family_with_one_unsupported_cell_keeps_the_rest():
    data = draw(5000, 2, LAW_EMPTY_CELL)
    result = factorized_joint_cells(
        data,
        declaration(),
        outcome="y",
        adjustment=["z"],
        ordering=["t1", "t0"],
        contrasts=["interaction", "cell_minus_control:1", "cell_minus_control:3"],
    )
    assert [c.status for c in result.cells] == ["supported"] * 3 + ["unsupported"]
    refused = result.cells[3]
    assert refused.refusal.code == "arm_not_populated"
    assert refused.refusal.detail == "joint_cells.cell_empty"
    assert refused.estimate is None and refused.rows == 0
    assert result.degenerate_conditionals > 0
    assert result.sensitivity.orderings[0] == ("t1", "t0")
    by_name = {c.name: c for c in result.contrasts}
    assert by_name["cell_minus_control:1"].value == pytest.approx(0.4, abs=0.08)
    for name in ("interaction", "cell_minus_control:3"):
        assert by_name[name].value is None
        assert by_name[name].refusal_code == "joint_cell_unsupported"


def test_constituent_and_descendant_leakage_needs_a_declared_exclusion():
    data = {**draw(400, 3, LAW), "post": np.zeros(400)}
    error = refusal(
        lambda: check_derived_treatment(
            data, declaration(), outcome="y", adjustment=["z", "t1"]
        )
    )
    assert error.reason_code == "derived_treatment_invalid"
    assert "joint_cells.derived_constituent_in_adjustment" in str(error)
    assert error.refusal_fields["implicated_columns"] == ["t1"]

    excluded = declaration(
        exclusions=[DeclaredExclusion("t1", "constituent_of_treatment", "t1 is a component")]
    )
    plan = check_derived_treatment(data, excluded, outcome="y", adjustment=["z", "t1"])
    assert plan.adjustment == ("z",)
    assert plan.exclusions == tuple(excluded.exclusions)

    error = refusal(
        lambda: check_derived_treatment(
            data, declaration(), outcome="y", adjustment=["z", "post"]
        )
    )
    assert "joint_cells.derived_descendant_in_adjustment" in str(error)
    wrong_rule = declaration(
        exclusions=[DeclaredExclusion("z", "constituent_of_treatment", "an admissible covariate")]
    )
    error = refusal(
        lambda: check_derived_treatment(data, wrong_rule, outcome="y", adjustment=["z"])
    )
    assert "joint_cells.derived_exclusion_rule_mismatch" in str(error)


def test_exact_duplicate_treatments_and_rank_failure_are_refused_with_columns_named():
    data = draw(400, 4, LAW)
    duplicated = {**data, "t1": data["t0"].copy()}
    error = refusal(
        lambda: check_derived_treatment(
            duplicated, declaration(), outcome="y", adjustment=["z"]
        )
    )
    assert "joint_cells.derived_duplicate_constituents" in str(error)
    assert error.refusal_fields["implicated_columns"] == ["t0", "t1"]

    rank = {**data, "z_dup": data["z"].copy(), "z2": data["y"] * 0 + np.arange(400) % 7}
    error = refusal(
        lambda: check_derived_treatment(
            rank, declaration(), outcome="y", adjustment=["z", "z_dup", "z2"]
        )
    )
    assert error.reason_code == "design_rank_deficient"
    assert error.refusal_fields["implicated_columns"] == ["z_dup"]
    assert error.refusal_fields["numerical_rank"] == 3
    assert error.refusal_fields["design_columns"] == 4
    plan = check_derived_treatment(rank, declaration(), outcome="y", adjustment=["z", "z2"])
    assert plan.adjustment == ("z", "z2")


def test_illegal_values_and_contradictory_declarations_are_refused():
    data = draw(300, 5, LAW)
    narrow = declaration(legal_values=[0.0, 1.0, 2.0])
    error = refusal(
        lambda: check_derived_treatment(data, narrow, outcome="y", adjustment=["z"])
    )
    assert "joint_cells.derived_illegal_value" in str(error)
    for edit in (
        {"intervention": "joint_components", "transformation": "product"},
        {"legal_values": [0.0]},
        {"legal_values": [0.0, 7.0]},
        {
            "sources": [
                *components("t0", "t1"),
                SourceColumn("z", "admissible_pre_treatment_covariate", "post_treatment"),
            ]
        },
    ):
        error = refusal(
            lambda edit=edit: check_derived_treatment(
                data, declaration(**edit), outcome="y", adjustment=["z"]
            )
        )
        assert error.reason_code == "derived_treatment_invalid"
    with pytest.raises(CausalValueError):
        check_derived_treatment(
            data,
            declaration(sources=[SourceColumn("t0", "not_a_role", "at_treatment")]),
            outcome="y",
            adjustment=["z"],
        )


def test_intervals_machine_learning_nuisance_and_lasso_are_closed():
    data = draw(300, 6, LAW)

    def call(**options):
        return factorized_joint_cells(
            data, declaration(), outcome="y", adjustment=["z"], **options
        )

    error = refusal(lambda: call(level=0.95))
    assert error.reason_code == "penalized_interval_not_licensed"
    assert "joint_cells.interval_withheld" in str(error)
    for name in ("ml", "neural_network", "gradient_boosting"):
        error = refusal(lambda name=name: call(nuisance=name))
        assert error.reason_code == "ml_nuisance_not_licensed"
        assert "joint_cells.ml_learner_not_declared" in str(error)
    # An interval over a learner-supplied family is closed whether or not the provider is
    # built: it is refused before any data is read.
    error = refusal(lambda: call(nuisance="random_forest", level=0.95))
    assert error.reason_code == "ml_nuisance_not_licensed"
    assert "joint_cells.ml_interval_withheld" in str(error)
    with pytest.raises(CausalValueError):
        call(nuisance="random_forest", penalties=[0.1])
    with pytest.raises(CausalValueError):
        call(nuisance=3)
    assert refusal(lambda: call(nuisance="lasso")).reason_code == "selection_inference_not_licensed"
    assert refusal(lambda: call(nuisance="typo")).reason_code == "invalid_argument"
    assert refusal(lambda: call(ordering=["t0", "t0"])).reason_code == "invalid_argument"


def test_a_declared_learner_nuisance_is_cross_fitted_and_point_only():
    data = draw(4000, 8, LAW)
    try:
        result = factorized_joint_cells(
            data,
            declaration(),
            outcome="y",
            adjustment=["z"],
            nuisance="random_forest",
            seed=3,
            contrasts=["interaction"],
        )
    except CausalUnsupportedError as error:
        # A build without the forest provider refuses typed instead of substituting ridge.
        assert error.reason_code == "route_not_supported"
        assert "joint_cells.learner_unavailable" in str(error)
        pytest.skip("the random_forest provider is not built into this wheel")
    assert result.claim == "point_only"
    assert len(result.supported) == 4
    for cell in result.cells:
        # Loose band: a forest's leaf means carry bootstrap noise and no rate is claimed.
        assert cell.estimate == pytest.approx(truth(cell.cell), abs=0.2)
    assert "nuisance=ml_joint_cell" in result.provenance
    assert "propensity=ridge_logistic" not in result.provenance
    assert "fold_seed=3" in result.provenance
    (interaction,) = result.contrasts
    assert interaction.value == pytest.approx(1.0, abs=0.3)


def test_a_cancelled_fit_is_never_a_result():
    data = draw(300, 7, LAW)
    token = CancellationToken()
    token.cancel()
    with pytest.raises((CausalCancelledError, CausalUnsupportedError)):
        factorized_joint_cells(
            data, declaration(), outcome="y", adjustment=["z"], cancel=token
        )
