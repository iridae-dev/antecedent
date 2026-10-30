"""Family tables: estimates of a method family joined into one aligned joint-draw table.

Truth is the declared moments: the draws carry the estimates as their means and the declared
covariance (squared standard errors on the diagonal, a declared shared-evidence term off it).
"""

from __future__ import annotations

import numpy as np
import pytest
from antecedent import propagation_chain as pc
from antecedent.errors import CausalValueError
from antecedent.outcomes import require_success

ROWS = 1597


def table_array(table: pc.FamilyTable) -> np.ndarray:
    return np.column_stack([np.asarray(source.values) for source in table.sources])


def test_independent_estimates_with_disjoint_evidence_form_a_diagonal_table():
    a = pc.FamilyEstimate("a", 0.4, 0.1, factors=(("model", "model-a"),))
    b = pc.FamilyEstimate("b", 0.9, 0.2, factors=(("model", "model-b"),))
    table = pc.family_table([a, b], alignment="family-1")
    assert [source.id for source in table.sources] == ["a", "b"]
    assert {source.alignment for source in table.sources} == {"family-1"}
    assert all(source.meaning == "sampling" for source in table.sources)
    assert table.sources[0].factors == (("model", "model-a"),)
    np.testing.assert_allclose(table.covariance, np.diag([0.01, 0.04]), rtol=0, atol=1e-15)
    draws = table_array(table)
    assert draws.shape == (ROWS, 2)
    np.testing.assert_allclose(draws.mean(axis=0), [0.4, 0.9], atol=1e-9)
    stds = draws.std(axis=0, ddof=1)
    np.testing.assert_allclose(stds, [0.1, 0.2], rtol=2e-3)
    assert abs(np.corrcoef(draws.T)[0, 1]) < 1e-6
    assert table.authority == ()


def test_a_declared_covariance_with_a_shared_factor_is_carried_by_the_draws():
    shared = (("row", "trial-rows"),)
    a = pc.FamilyEstimate("a", 0.4, 0.1, factors=shared)
    b = pc.FamilyEstimate("b", 0.9, 0.2, factors=shared)
    covariance = np.array([[0.01, 0.01], [0.01, 0.04]])  # correlation 0.5
    table = pc.family_table(
        [a, b], alignment="family-2", dependence="covariance", covariance=covariance
    )
    np.testing.assert_allclose(table.covariance, covariance, rtol=0, atol=1e-15)
    draws = table_array(table)
    assert np.corrcoef(draws.T)[0, 1] == pytest.approx(0.5, abs=1e-6)
    assert {f for source in table.sources for f in source.factors} == {("row", "trial-rows")}


def test_the_table_flows_into_a_chain_as_ordinary_sources():
    shared = (("row", "trial-rows"),)
    table = pc.family_table(
        [
            pc.FamilyEstimate("a", 0.4, 0.1, factors=shared),
            pc.FamilyEstimate("b", 0.9, 0.2, factors=shared),
        ],
        alignment="family-3",
        dependence="covariance",
        covariance=[[0.01, 0.01], [0.01, 0.04]],
    )
    chain = require_success(
        pc.propagate(
            table.sources,
            [pc.binary("gap", "derived_quantity", ["b", "a"], "difference")],
            output="gap",
            reading="spread",
        )
    )
    # Var(b - a) = 0.04 + 0.01 - 2 * 0.01 = 0.03, with the shared evidence counted once.
    gap = np.asarray(table.sources[1].values) - np.asarray(table.sources[0].values)
    assert gap.var(ddof=1) == pytest.approx(0.03, rel=2e-3)
    # The chain itself, not only the table, carries the shared evidence once: mean 0.9 - 0.4
    # and variance 0.03 (independent would be 0.05).
    assert chain.answer.values["mean"] == pytest.approx(0.5, abs=1e-9)
    assert chain.answer.values["variance"] == pytest.approx(0.03, abs=1e-9)
    assert chain.answer.values["n_rows"] == ROWS


def test_independence_claimed_over_shared_evidence_is_refused():
    shared = (("row", "trial-rows"),)
    with pytest.raises(pc.PropagationChainRefusal) as info:
        pc.family_table(
            [
                pc.FamilyEstimate("a", 0.4, 0.1, factors=shared),
                pc.FamilyEstimate("b", 0.9, 0.2, factors=shared),
            ],
            alignment="family-4",
        )
    assert info.value.code == "evidence_double_counted"


def test_a_covariance_that_counts_shared_evidence_as_independent_is_refused():
    shared = (("row", "trial-rows"),)
    with pytest.raises(pc.PropagationChainRefusal) as info:
        pc.family_table(
            [
                pc.FamilyEstimate("a", 0.4, 0.1, factors=shared),
                pc.FamilyEstimate("b", 0.9, 0.2, factors=shared),
            ],
            alignment="family-5",
            dependence="covariance",
            covariance=[[0.01, 0.0], [0.0, 0.04]],
        )
    assert info.value.code == "evidence_double_counted"


def test_a_covariance_term_without_a_declared_shared_factor_is_refused():
    with pytest.raises(pc.PropagationChainRefusal) as info:
        pc.family_table(
            [
                pc.FamilyEstimate("a", 0.4, 0.1, factors=(("model", "model-a"),)),
                pc.FamilyEstimate("b", 0.9, 0.2, factors=(("model", "model-b"),)),
            ],
            alignment="family-6",
            dependence="covariance",
            covariance=[[0.01, 0.01], [0.01, 0.04]],
        )
    assert info.value.code == "evidence_overlap_unknown"


def test_a_covariance_diagonal_that_is_not_the_published_variance_is_refused():
    with pytest.raises(pc.PropagationChainRefusal, match="family_covariance") as info:
        pc.family_table(
            [pc.FamilyEstimate("a", 0.4, 0.1), pc.FamilyEstimate("b", 0.9, 0.2)],
            alignment="family-7",
            dependence="covariance",
            covariance=[[0.02, 0.0], [0.0, 0.04]],
        )
    assert info.value.code == "invalid_argument"


def test_a_ledger_overlap_needs_the_ledger_that_holds_the_authorizing_sources():
    with pytest.raises(pc.PropagationChainRefusal, match="family_authority"):
        pc.family_table(
            [
                pc.FamilyEstimate("a", 0.4, 0.1, authority="0" * 64),
                pc.FamilyEstimate("b", 0.9, 0.2, authority="1" * 64),
            ],
            alignment="family-8",
            dependence="ledger_overlap",
        )


def test_malformed_declarations_are_refused_before_the_native_call():
    estimate = pc.FamilyEstimate("a", 0.4, 0.1)
    with pytest.raises(CausalValueError, match="dependence must be independent"):
        pc.family_table([estimate], alignment="family-9", dependence="correlated")
    with pytest.raises(CausalValueError, match="needs a covariance matrix"):
        pc.family_table([estimate], alignment="family-9", dependence="covariance")
    with pytest.raises(CausalValueError, match="applies only to dependence"):
        pc.family_table([estimate], alignment="family-9", covariance=[[0.01]])
    with pytest.raises(CausalValueError, match="meaning must be"):
        pc.family_table([pc.FamilyEstimate("a", 0.4, 0.1, meaning="belief")], alignment="x")


def test_family_table_observes_execution_cancellation():
    import antecedent
    token = antecedent.CancellationToken()
    token.cancel()
    estimate = pc.FamilyEstimate("a", 0.4, 0.1)
    with pytest.raises(antecedent.errors.CausalCancelledError):
        pc.family_table([estimate], alignment="family-cancel", execution=antecedent.Execution(cancel=token))
