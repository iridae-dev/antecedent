"""The effect after discovery on a row split, accepted on the treatment's parent set, through Python.

Oracles: the generating law is a hub ``h`` that confounds the treatment ``x`` and the outcome
``y`` (``h -> x <- p``, ``x -> m -> y``, ``h -> y``) with nuisance chains hanging off ``h``
(twice), ``p`` and ``m``. Every column is standardized (a node's noise variance is what its
parents leave of a unit variance) and the coefficients keep the population partial correlation
PC's decisions rest on at least 0.15 in magnitude. The treatment's parents are exactly ``h`` and
``p`` and the true total effect is the path product ``X_M * M_Y``; the chains off ``h`` and
``p`` are undirected in the equivalence class and must be listed, not completed, while the chain
off ``m`` is oriented away from the treatment. A log-linear study exponentiates every column, so
the intervention mean ratio is ``exp(shift * truth)``.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from antecedent import selective_discovery as sd
from antecedent.errors import CausalTypeError, CausalValueError

from _refusal import assert_registered_refusal

SPLIT_SEED = 17
SEED = 4242
ROWS = 4000

#: Coefficients of ``x`` on ``(h, p)``, ``m`` on ``x`` and ``y`` on ``(m, h)``; the first of the
#: ten-variable weights is the path ``x -> m -> y``.
CORE = {"x": (0.62, -0.56), "m": (0.64,), "y": (0.57, 0.46)}
XM, MY = 0.64, 0.57
TRUTH = XM * MY
#: The ten- and twelve-variable fixtures are checked against the family's margin rule: the
#: smallest population partial correlation PC's decisions rest on is 0.234 (ten variables) and
#: 0.211 (twelve), both above the published 0.15 (computed from the population covariance of
#: exactly these coefficients over every conditioning set the true neighbourhoods allow).
#: Parent and coefficient of each nuisance node, in order; the parent of ``n{j}`` is the anchor of
#: slot ``j % 4`` (``h``, ``p``, ``m``, ``h``) or the previous node of that slot's chain.
CHAIN = (0.62, 0.61, -0.52, 0.72, 0.81, 0.6, -0.6)


def hub_table(seed: int, rows: int = ROWS, chains: int = 5) -> dict[str, np.ndarray]:
    """Ten variables for ``chains = 5``; more chain nodes extend the same four chains."""
    rng = np.random.default_rng(seed)
    names = ["h", "p", "x", "m", "y"]
    weights: dict[str, list[tuple[str, float]]] = {"h": [], "p": []}
    weights["x"] = [("h", CORE["x"][0]), ("p", CORE["x"][1])]
    weights["m"] = [("x", CORE["m"][0])]
    weights["y"] = [("m", CORE["y"][0]), ("h", CORE["y"][1])]
    tails = ["h", "p", "m", "h"]
    for j in range(chains):
        slot = j % 4
        name = f"n{j}"
        weights[name] = [(tails[slot], CHAIN[j])]
        tails[slot] = name
        names.append(name)
    columns: dict[str, np.ndarray] = {}
    cov: dict[tuple[str, str], float] = {}
    order: list[str] = []
    for name in names:
        parents = weights[name]
        explained = sum(
            wa * wb * (1.0 if a == b else cov[(a, b)]) for a, wa in parents for b, wb in parents
        )
        columns[name] = sum((w * columns[a] for a, w in parents), np.zeros(rows)) + np.sqrt(
            1.0 - explained
        ) * rng.normal(size=rows)
        for earlier in order:
            c = sum(w * (1.0 if a == earlier else cov[(a, earlier)]) for a, w in parents)
            cov[(name, earlier)] = cov[(earlier, name)] = c
        order.append(name)
    return {name: columns[name] for name in names}


def fixture_structure(chains: int) -> list[tuple[str, list[tuple[str, float]]]]:
    """The fixture's nodes in order with their ``(parent, coefficient)`` lists."""
    nodes: list[tuple[str, list[tuple[str, float]]]] = [
        ("h", []),
        ("p", []),
        ("x", [("h", CORE["x"][0]), ("p", CORE["x"][1])]),
        ("m", [("x", CORE["m"][0])]),
        ("y", [("m", CORE["y"][0]), ("h", CORE["y"][1])]),
    ]
    tails = ["h", "p", "m", "h"]
    for j in range(chains):
        slot = j % 4
        nodes.append((f"n{j}", [(tails[slot], CHAIN[j])]))
        tails[slot] = f"n{j}"
    return nodes


def population_margin(nodes: list[tuple[str, list[tuple[str, float]]]]) -> float:
    """The family's strong-faithfulness margin: the smallest absolute population partial
    correlation PC's decisions rest on (adjacent pairs and pairs with a common neighbour, each
    given every set of at most four true neighbours of one endpoint that does not d-separate
    the pair and is no larger than the smallest set that does)."""
    index = {name: i for i, (name, _) in enumerate(nodes)}
    size = len(nodes)
    cov = np.eye(size)
    for child, (_, parents) in enumerate(nodes):
        for earlier in range(child):
            cov[child, earlier] = cov[earlier, child] = sum(
                w * cov[index[a], earlier] for a, w in parents
            )
    adjacent: list[set[int]] = [set() for _ in nodes]
    for child, (_, parents) in enumerate(nodes):
        for a, _w in parents:
            adjacent[child].add(index[a])
            adjacent[index[a]].add(child)

    def partial(u: int, v: int, given: tuple[int, ...]) -> float:
        ids = [u, v, *given]
        precision = np.linalg.inv(cov[np.ix_(ids, ids)])
        return abs(float(-precision[0, 1] / np.sqrt(precision[0, 0] * precision[1, 1])))

    margin = np.inf
    for u in range(size):
        for v in range(u + 1, size):
            if v not in adjacent[u] and not adjacent[u] & adjacent[v]:
                continue
            sets = []
            for end, other in ((u, v), (v, u)):
                pool = sorted(adjacent[end] - {other})
                for r in range(min(4, len(pool)) + 1):
                    sets.extend((r, partial(u, v, c)) for c in itertools.combinations(pool, r))
            separating = [r for r, value in sets if value < 1e-9]
            smallest = min(separating) if separating else 5
            margin = min([margin] + [v_ for r, v_ in sets if v_ >= 1e-9 and r <= smallest])
    return float(margin)


def test_the_fixtures_satisfy_the_published_margin() -> None:
    for chains in (5, 7):
        assert population_margin(fixture_structure(chains)) >= 0.15


def positive(table: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: np.exp(column) for name, column in table.items()}


def test_a_linear_study_reports_the_total_effect_and_the_unresolved_edges() -> None:
    estimate = sd.sample_split_parent_set_effect(
        hub_table(1),
        treatment="x",
        outcome="y",
        transformation="linear",
        split_seed=SPLIT_SEED,
        seed=SEED,
    )
    assert estimate.scalar == "total_effect"
    assert estimate.transformation == "linear"
    assert set(estimate.treatment_parents) == {"h", "p"}
    assert set(estimate.treatment_children) == {"m"}
    assert estimate.discovery_rows == estimate.analysis_rows == ROWS // 2
    assert estimate.level == pytest.approx(0.95)
    assert estimate.lower < estimate.effect < estimate.upper
    assert abs(estimate.effect - TRUTH) <= 4 * estimate.standard_error
    assert estimate.coefficient == pytest.approx(estimate.effect)
    # The chains off the roots stay undirected: listed, never completed. The chain off the
    # mediator is oriented away from the treatment.
    unresolved = {frozenset(edge[:2]) for edge in estimate.unresolved_edges}
    assert unresolved == {
        frozenset({"h", "n0"}),
        frozenset({"n0", "n4"}),
        frozenset({"h", "n3"}),
        frozenset({"p", "n1"}),
    }
    assert all(edge[2] == "undirected_in_equivalence_class" for edge in estimate.unresolved_edges)
    assert estimate.acceptance == "treatment_parent_set_unambiguous"


def test_a_log_linear_study_reports_an_intervention_mean_ratio() -> None:
    estimate = sd.sample_split_parent_set_effect(
        positive(hub_table(2)),
        treatment="x",
        outcome="y",
        transformation="log_linear",
        split_seed=SPLIT_SEED,
        seed=SEED,
    )
    assert estimate.scalar == "intervention_mean_ratio"
    assert set(estimate.treatment_parents) == {"h", "p"}
    assert estimate.lower > 0
    assert estimate.lower < estimate.effect < estimate.upper
    # The default intervention multiplies the treatment by e: the ratio is exp(beta).
    assert estimate.effect == pytest.approx(np.exp(estimate.coefficient))
    assert abs(estimate.effect - np.exp(TRUTH)) <= 4 * estimate.standard_error
    assert estimate.basis["estimator"] == "parent_set_split.pc_parcorr.log_linear_mean_ratio"
    assert (
        estimate.basis["functional"]
        == "intervention_mean_ratio.variables_10.max_degree_4.topology_hub_confounded_multi_path"
    )


def test_a_prespecified_log_shift_rescales_the_ratio() -> None:
    data = positive(hub_table(3))
    unit = sd.sample_split_parent_set_effect(
        data, treatment="x", outcome="y", transformation="log_linear", split_seed=SPLIT_SEED
    )
    half = sd.sample_split_parent_set_effect(
        data,
        treatment="x",
        outcome="y",
        transformation="log_linear",
        log_shift=0.5,
        split_seed=SPLIT_SEED,
    )
    assert half.coefficient == pytest.approx(unit.coefficient)
    assert half.effect == pytest.approx(np.exp(0.5 * unit.coefficient))
    assert half.lower == pytest.approx(np.exp(0.5 * np.log(unit.lower)))


def test_the_basis_is_keyed_under_its_own_structure_and_never_the_directed_route() -> None:
    estimate = sd.sample_split_parent_set_effect(
        hub_table(4),
        treatment="x",
        outcome="y",
        transformation="linear",
        split_seed=SPLIT_SEED,
        seed=SEED,
    )
    basis = estimate.basis
    assert basis["structure"] == "discovered_sample_split.treatment_parent_set"
    assert basis["estimator"] == "parent_set_split.pc_parcorr.total_effect_adjustment"
    assert basis["dependence"] == "disjoint_rows"
    assert (
        basis["functional"]
        == "total_effect.variables_10.max_degree_4.topology_hub_confounded_multi_path"
    )
    assert basis["row_count"] == ROWS // 2
    assert basis["structure"] != "discovered_sample_split"


def test_an_undirected_edge_at_the_treatment_is_refused_by_name() -> None:
    # The hub's edge to the first nuisance node is undirected, so its parent set is ambiguous.
    with pytest.raises(sd.SelectiveDiscoveryRefusal) as info:
        sd.sample_split_parent_set_effect(
            hub_table(5),
            treatment="h",
            outcome="y",
            transformation="linear",
            split_seed=SPLIT_SEED,
            seed=SEED,
        )
    assert_registered_refusal(info.value)
    assert info.value.reason_code == "cell_not_licensed"
    assert info.value.detail == "parent_set_effect.treatment_parents_ambiguous"
    assert "h - n0" in info.value.message
    assert "no completion is chosen" in info.value.message


def test_log_linear_inputs_must_be_positive() -> None:
    data = positive(hub_table(6, rows=600))
    data["m"] = data["m"].copy()
    data["m"][:5] *= -1.0
    with pytest.raises(sd.SelectiveDiscoveryRefusal) as info:
        sd.sample_split_parent_set_effect(
            data,
            treatment="x",
            outcome="y",
            transformation="log_linear",
            split_seed=SPLIT_SEED,
            seed=SEED,
        )
    assert info.value.reason_code == "invalid_argument"
    assert info.value.detail == "parent_set_effect.log_linear_input_not_positive"
    assert "column `m` has 5 non-positive value(s)" in info.value.message
    assert isinstance(info.value, CausalValueError)


@pytest.mark.parametrize("declared", ["undeclared", "", "log"])
def test_the_transformation_must_be_declared_before_the_data_are_touched(declared: str) -> None:
    # The declaration is parsed before any column is analysed.
    with pytest.raises(sd.SelectiveDiscoveryRefusal) as info:
        sd.sample_split_parent_set_effect(
            hub_table(7, rows=600),
            treatment="x",
            outcome="y",
            transformation=declared,
            split_seed=SPLIT_SEED,
            seed=SEED,
        )
    assert info.value.reason_code == "invalid_argument"
    assert info.value.detail == "parent_set_effect.transformation_undeclared"
    assert "before any data are read" in info.value.message


def test_a_missing_transformation_is_a_type_error() -> None:
    with pytest.raises(CausalTypeError):
        sd.sample_split_parent_set_effect(
            hub_table(8, rows=600),
            treatment="x",
            outcome="y",
            transformation=None,  # type: ignore[arg-type]
            split_seed=SPLIT_SEED,
        )


def test_a_log_shift_is_declared_only_by_a_log_linear_study() -> None:
    with pytest.raises(sd.SelectiveDiscoveryRefusal) as info:
        sd.sample_split_parent_set_effect(
            hub_table(9, rows=600),
            treatment="x",
            outcome="y",
            transformation="linear",
            log_shift=1.0,
            split_seed=SPLIT_SEED,
        )
    assert info.value.detail == "parent_set_effect.source_invalid"


def test_a_maximum_degree_above_four_is_refused_with_both_degrees_named() -> None:
    with pytest.raises(sd.SelectiveDiscoveryRefusal) as info:
        sd.sample_split_parent_set_effect(
            hub_table(10, rows=600),
            treatment="x",
            outcome="y",
            transformation="linear",
            max_degree=5,
            split_seed=SPLIT_SEED,
        )
    assert_registered_refusal(info.value)
    assert info.value.reason_code == "cell_not_licensed"
    assert info.value.detail == "parent_set_effect.degree_above_supported"
    assert "declared maximum graph degree is 5" in info.value.message
    assert "supported maximum is 4" in info.value.message


def test_a_table_outside_ten_to_twenty_variables_is_refused() -> None:
    rng = np.random.default_rng(11)
    data = {name: rng.normal(size=60) for name in ("a", "b", "c")}
    with pytest.raises(sd.SelectiveDiscoveryRefusal) as info:
        sd.sample_split_parent_set_effect(
            data, treatment="a", outcome="b", transformation="linear", split_seed=1, seed=SEED
        )
    assert info.value.reason_code == "cell_not_licensed"
    assert info.value.detail == "parent_set_effect.variable_count_not_measured"


def test_unknown_roles_are_rejected_before_any_native_work() -> None:
    with pytest.raises(CausalValueError):
        sd.sample_split_parent_set_effect(
            hub_table(12, rows=600),
            treatment="missing",
            outcome="y",
            transformation="linear",
            split_seed=1,
        )


def test_the_same_seeds_give_the_same_estimate_and_another_split_seed_another_split() -> None:
    data = hub_table(13, rows=2000)
    first = sd.sample_split_parent_set_effect(
        data, treatment="x", outcome="y", transformation="linear", split_seed=SPLIT_SEED, seed=SEED
    )
    again = sd.sample_split_parent_set_effect(
        data, treatment="x", outcome="y", transformation="linear", split_seed=SPLIT_SEED, seed=SEED
    )
    assert first == again
    other = sd.sample_split_parent_set_effect(
        data,
        treatment="x",
        outcome="y",
        transformation="linear",
        split_seed=SPLIT_SEED + 1,
        seed=SEED,
    )
    assert other.receipt_identity != first.receipt_identity


def test_an_unmeasured_variable_count_is_computed_and_never_attested() -> None:
    # Twelve variables lie between measured counts: the estimate is computed, and attesting it
    # is refused rather than borrowing another count's record.
    estimate = sd.sample_split_parent_set_effect(
        hub_table(14, rows=2000, chains=7),
        treatment="x",
        outcome="y",
        transformation="linear",
        split_seed=SPLIT_SEED,
        seed=SEED,
    )
    assert (
        estimate.basis["functional"]
        == "total_effect.variables_12.max_degree_4.topology_hub_confounded_multi_path"
    )
    with pytest.raises(sd.SelectiveDiscoveryRefusal) as info:
        estimate.measure()
    assert_registered_refusal(info.value)
    assert info.value.reason_code == "cell_not_licensed"
