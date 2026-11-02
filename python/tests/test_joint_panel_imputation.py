"""Missing panel outcomes and confounders are imputed and pooled by Rust."""

from __future__ import annotations

import pytest
from antecedent import Execution, ResourceLimits, joint_panel, state
from antecedent.errors import (
    CausalCancelledError,
    CausalError,
    CausalResourceError,
    CausalTypeError,
)
from antecedent.missing import ImputationSpec

from test_joint_panel import declarations

SPEC = ImputationSpec(imputations=4, iterations=2, seed=11, domains=(("confounder", "continuous"),))


def options(draws=8):
    return {
        "optimizer": {"max_iterations": 2000, "gradient_tolerance": 1e-4},
        "inference": {"kind": "frequentist", "draws": draws},
        "initial": None,
    }


def panel(edit=None, *, draws=8, spec=SPEC, execution=None):
    groups, worlds = declarations()
    if edit is not None:
        edit(groups)
    return joint_panel.prepare_imputed(
        groups,
        imputation=spec,
        channels=["a", "b", "c"],
        outcome="y",
        confounder="z",
        season="season",
        seasons=4,
        interval_ns=604_800_000_000_000,
        worlds=worlds,
        interactions={},
        options=options(draws),
        execution=execution,
    )


def with_holes(groups):
    for row in (10, 15, 20):
        groups[2]["y"][row] = None
    groups[1]["z"][12] = None


def test_pooling_follows_rubins_rules_over_the_imputed_fits():
    result = panel(with_holes).estimate(execution=Execution(seed=20261101))
    assert result.method == "rubin"
    assert result.imputations == 4
    assert result.programs == ("baseline", "joint")
    assert result.horizons == (24, 24)
    assert len(result.point) == len(result.total_covariance) == len(result.coordinates) == 48
    assert len(result.fits) == 4
    assert result.imputed_cells == {"outcome": 3, "confounder": 1}
    m = 4.0
    for coordinate, pooled in enumerate(result.coordinates):
        values = [fit[coordinate] for fit in result.fits]
        mean = sum(values) / m
        between = sum((v - mean) ** 2 for v in values) / (m - 1.0)
        assert result.point[coordinate] == pytest.approx(mean, abs=1e-12)
        assert pooled["point"] == pytest.approx(mean, abs=1e-12)
        assert pooled["between"] == pytest.approx(between, rel=1e-9, abs=1e-12)
        assert pooled["within"] > 0.0  # the fit's own uncertainty enters
        total = pooled["within"] + (1.0 + 1.0 / m) * between
        assert pooled["total"] == pytest.approx(total, rel=1e-9)
        assert result.total_covariance[coordinate][coordinate] == pytest.approx(total, rel=1e-9)
    # The imputed cells matter: the imputations differ from one another.
    assert any(pooled["between"] > 0.0 for pooled in result.coordinates)
    responses = result.responses()
    assert tuple(responses) == result.programs
    assert all(len(values) == 24 for values in responses.values())
    assert responses["baseline"] + responses["joint"] == result.point


def test_the_assumptions_record_and_the_streams_do_not_depend_on_the_call():
    prepared = panel(with_holes)
    first = prepared.estimate(execution=Execution(seed=1))
    second = prepared.estimate(execution=Execution(seed=2))
    assumptions = first.assumptions
    assert assumptions["model"] == "chained_equations"
    assert (assumptions["imputations"], assumptions["iterations"], assumptions["seed"]) == (
        4,
        2,
        11,
    )
    assert assumptions["assumption"] == "missing_at_random"
    assert assumptions["valid_under_declared_assumption"] is True
    assert assumptions["domains"] == [["confounder", "continuous"]]
    assert len(assumptions["digest"]) == 64
    # Imputation table j opens its own stream from the declared seed.
    assert first.point == second.point
    assert second.assumptions["digest"] == assumptions["digest"]
    other = panel(with_holes, spec=ImputationSpec(imputations=4, iterations=2, seed=12))
    assert other.estimate().assumptions["digest"] != assumptions["digest"]


def set_cell(name, row):
    def edit(groups):
        groups[0][name][row] = None

    return edit


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (set_cell("a", 12), "intervention itself"),
        (set_cell("season", 12), "design label"),
    ],
)
def test_a_missing_assignment_or_season_level_is_refused_not_imputed(edit, message):
    with pytest.raises(CausalError, match=message):
        panel(edit).estimate()


def test_pooling_needs_each_fits_own_uncertainty_draws():
    with pytest.raises(CausalError, match="uncertainty draws"):
        panel(with_holes, draws=0)


def test_a_declaration_imputation_cannot_serve_is_refused():
    with pytest.raises(CausalError, match="not at random|not_at_random"):
        panel(with_holes, spec=ImputationSpec(assumption="missing_not_at_random"))
    with pytest.raises(CausalError, match="absent"):
        panel(with_holes, spec=ImputationSpec(domains=(("absent", "binary"),)))
    with pytest.raises(CausalTypeError, match="ImputationSpec"):
        panel(with_holes, spec={"imputations": 4})


def test_preparation_and_pooling_run_under_the_callers_execution():
    tight = Execution(limits=ResourceLimits(memory_bytes=1))
    with pytest.raises(CausalResourceError):
        panel(with_holes, execution=tight)
    # An unfunded completed table and a cancellation are operational stops, never request errors.
    prepared = panel(with_holes)
    with pytest.raises(CausalResourceError, match="admission pool"):
        prepared.estimate(execution=tight)
    token = state.CancellationToken()
    token.cancel()
    with pytest.raises(CausalCancelledError):
        prepared.estimate(execution=Execution(cancel=token))
