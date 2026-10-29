"""The effect of a treatment on an outcome after discovery on a row split, through Python.

Oracles: the generating law is ``z1 -> x <- w``, ``x -> y <- z3``, ``w -> y`` with the true
coefficient of ``x`` on ``y`` equal to ``XY``; every edge is oriented by a collider, so the
treatment's parents are exactly ``z1`` and ``w`` and the interval must cover ``XY``. The
analysis half of a 6000-row table holds 3000 rows, which no measured grid point reaches, so an
attestation there is a typed refusal whatever has been measured.
"""

from __future__ import annotations

import numpy as np
import pytest
from antecedent import Execution
from antecedent import selective_discovery as sd
from antecedent.errors import CausalValueError

from _refusal import assert_registered_refusal

XY = 1.2
SPLIT_SEED = 17
SEED = 4242


def collider_table(seed: int, rows: int = 6000) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    z1, w, z3 = rng.normal(size=rows), rng.normal(size=rows), rng.normal(size=rows)
    x = 0.8 * z1 - 0.6 * w + rng.normal(size=rows)
    y = XY * x + 1.4 * w + 0.8 * z3 + rng.normal(size=rows)
    return {"z1": z1, "w": w, "x": x, "y": y, "z3": z3}


def chain_table(seed: int, rows: int = 6000) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    columns = [rng.normal(size=rows)]
    for _ in range(4):
        columns.append(0.8 * columns[-1] + 0.6 * rng.normal(size=rows))
    return dict(zip("abcde", columns, strict=True))


def test_the_effect_adjusts_for_the_treatments_parents_and_covers_the_truth() -> None:
    estimate = sd.sample_split_effect(
        collider_table(1),
        treatment="x",
        outcome="y",
        split_seed=SPLIT_SEED,
        execution=Execution(seed=SEED),
    )
    assert set(estimate.adjustment_set) == {"z1", "w"}
    assert estimate.discovery_rows == 3000
    assert estimate.analysis_rows == 3000
    assert estimate.level == pytest.approx(0.95)
    assert estimate.lower < estimate.effect < estimate.upper
    assert estimate.lower <= XY <= estimate.upper
    half_width = (estimate.upper - estimate.lower) / 2
    # A Student quantile on about 2995 degrees of freedom times the standard error.
    assert half_width == pytest.approx(1.9608 * estimate.standard_error, rel=2e-3)


def test_the_basis_is_keyed_under_the_split_structure_not_a_known_graph() -> None:
    estimate = sd.sample_split_effect(
        collider_table(2),
        treatment="x",
        outcome="y",
        split_seed=SPLIT_SEED,
        execution=Execution(seed=SEED),
    )
    basis = estimate.basis
    assert basis["structure"] == "discovered_sample_split"
    assert basis["dependence"] == "disjoint_rows"
    assert basis["query"] == "AverageEffect"
    assert basis["estimator"] != "linear.adjustment.ate"
    assert basis["row_count"] == 3000
    assert basis["structure"] not in {"fixed", "graph_posterior"}


def test_the_same_inputs_give_the_same_estimate() -> None:
    data = collider_table(3)
    first = sd.sample_split_effect(
        data, treatment="x", outcome="y", split_seed=SPLIT_SEED, execution=Execution(seed=SEED)
    )
    again = sd.sample_split_effect(
        data, treatment="x", outcome="y", split_seed=SPLIT_SEED, execution=Execution(seed=SEED)
    )
    assert first == again
    other = sd.sample_split_effect(
        data, treatment="x", outcome="y", split_seed=SPLIT_SEED + 1, execution=Execution(seed=SEED)
    )
    assert other.receipt_identity != first.receipt_identity


def test_an_unresolved_structure_reports_no_interval() -> None:
    with pytest.raises(sd.SelectiveDiscoveryRefusal) as info:
        sd.sample_split_effect(
            chain_table(4),
            treatment="b",
            outcome="d",
            split_seed=SPLIT_SEED,
            execution=Execution(seed=SEED),
        )
    assert_registered_refusal(info.value)
    assert info.value.reason_code == "cell_not_licensed"
    assert info.value.detail == "split_effect.structure_unresolved"


def test_a_table_outside_five_to_seven_variables_is_refused() -> None:
    rng = np.random.default_rng(5)
    data = {name: rng.normal(size=60) for name in ("a", "b", "c")}
    with pytest.raises(sd.SelectiveDiscoveryRefusal) as info:
        sd.sample_split_effect(
            data, treatment="a", outcome="b", split_seed=1, execution=Execution(seed=SEED)
        )
    assert info.value.reason_code == "cell_not_licensed"
    assert info.value.detail == "split_effect.variable_count_not_measured"


def test_unknown_roles_are_rejected_before_any_native_work() -> None:
    with pytest.raises(CausalValueError, match=r"treatment 'missing' is not a column of data"):
        sd.sample_split_effect(
            collider_table(6),
            treatment="missing",
            outcome="y",
            split_seed=1,
            execution=Execution(seed=SEED),
        )


def test_an_interval_outside_the_measured_row_range_is_never_attested() -> None:
    estimate = sd.sample_split_effect(
        collider_table(7),
        treatment="x",
        outcome="y",
        split_seed=SPLIT_SEED,
        execution=Execution(seed=SEED),
    )
    with pytest.raises(sd.SelectiveDiscoveryRefusal) as info:
        estimate.measure()
    assert_registered_refusal(info.value)
    assert info.value.reason_code == "cell_not_licensed"
    assert info.value.detail == "measured_inference.record_not_attesting"


def test_reusing_the_rows_that_selected_is_still_refused() -> None:
    contract = sd.StaticContract(
        variables=("z1", "w", "x", "y", "z3"), alpha=0.01, fdr=None, seed=SEED
    )
    split = sd.discover_on_split(
        contract, collider_table(8), discovery_fraction=0.5, split_seed=SPLIT_SEED
    )
    with pytest.raises(sd.SelectiveDiscoveryRefusal) as info:
        split.request_unsplit_interval()
    assert info.value.reason_code == "cell_not_licensed"
    assert info.value.detail == "selective_discovery.downstream_interval_not_licensed"
