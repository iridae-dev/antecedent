"""Per-population discovery through the Python surface.

Known graphs (large linear-Gaussian samples, strict threshold so the skeleton is the true one):

* ``north``: the chain ``x -> y -> z``; no v-structure, so both edges stay undirected;
* ``south``: the collider ``x -> y <- z``; the unshielded collider compels both edges.

By hand: both edges are present in both populations and both are marked differently.
"""

from __future__ import annotations

import numpy as np
import pytest
from antecedent import selective_discovery as sd
from antecedent.errors import CausalValueError

from _refusal import assert_registered_refusal

N = 4000


def chain(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=N)
    y = 0.8 * x + 0.6 * rng.normal(size=N)
    z = 0.8 * y + 0.6 * rng.normal(size=N)
    return {"x": x, "y": y, "z": z}


def collider(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=N)
    z = rng.normal(size=N)
    y = 0.8 * x + 0.8 * z + 0.5 * rng.normal(size=N)
    return {"x": x, "y": y, "z": z}


CONTRACT = sd.StaticContract(variables=("x", "y", "z"), alpha=0.001, fdr=None, seed=5)


def test_each_population_is_discovered_on_its_own_rows_and_the_edges_are_lined_up() -> None:
    driver = sd.PerPopulationDiscovery()
    report = driver.run(CONTRACT, {"south": collider(2), "north": chain(1)})

    assert report.label == "descriptive_comparison"
    assert [p.population for p in report.populations] == ["north", "south"]
    assert report.population("north").unresolved_edges == 2
    assert report.population("south").unresolved_edges == 0
    assert all(not p.search_incomplete and p.retained_statistics > 0 for p in report.populations)
    assert report.population("north").structure_identity != report.population("south").structure_identity
    assert report.population("north").data_identity != report.population("south").data_identity

    assert [(str(e.a), str(e.b)) for e in report.edges] == [("x", "y"), ("y", "z")]
    assert len(report.differing_edges) == 2
    assert report.edges[0].marks["south"] == ("tail", "arrow")
    assert report.edges[1].marks["south"] == ("arrow", "tail")
    assert all(e.marks["north"] != e.marks["south"] for e in report.edges)

    north = driver.outcome("north")
    assert north is not None and len(north.unresolved_edges) == 2
    assert driver.retained_statistics("south") > 0
    assert driver.outcome("east") is None
    assert driver.report is not None and driver.report.identity == report.identity


def test_a_changed_population_recomputes_itself_and_the_other_is_served_from_its_store() -> None:
    driver = sd.PerPopulationDiscovery()
    first = driver.run(CONTRACT, {"north": chain(1), "south": collider(2)})
    second = driver.run(CONTRACT, {"north": chain(9), "south": collider(2)})

    assert second.population("north").data_identity != first.population("north").data_identity
    assert second.population("north").result_identity != first.population("north").result_identity
    south_first, south_second = first.population("south"), second.population("south")
    assert south_second.data_identity == south_first.data_identity
    assert south_second.result_identity == south_first.result_identity
    assert south_second.report.pure_reuse
    assert south_second.report.work.is_pure_reuse
    assert second.identity != first.identity


@pytest.mark.parametrize(
    ("populations", "contract", "code", "detail"),
    [
        (
            {"north": chain(1), "south": chain(1)},
            CONTRACT,
            "invalid_argument",
            "selective_discovery.pooled_population_data",
        ),
        (
            {"north": chain(1), "south": collider(2)},
            sd.replace(CONTRACT, additional_populations=("south",)),
            "route_not_supported",
            "selective_discovery.pooled_population_search",
        ),
        (
            {"north": chain(1)},
            CONTRACT,
            "invalid_argument",
            "selective_discovery.populations_missing",
        ),
    ],
)
def test_pooled_and_malformed_requests_are_refused_without_retaining_anything(
    populations: dict[str, dict[str, np.ndarray]],
    contract: sd.StaticContract,
    code: str,
    detail: str,
) -> None:
    driver = sd.PerPopulationDiscovery()
    with pytest.raises(sd.SelectiveDiscoveryRefusal) as info:
        driver.run(contract, populations)
    assert_registered_refusal(info.value)
    assert info.value.reason_code == code
    assert info.value.detail == detail
    assert driver.report is None
    assert driver.outcome("north") is None


def test_differences_between_populations_are_not_selection_targets() -> None:
    driver = sd.PerPopulationDiscovery()
    driver.run(CONTRACT, {"north": chain(1), "south": collider(2)})
    with pytest.raises(sd.SelectiveDiscoveryRefusal) as info:
        driver.request_selection_targets(["y"])
    assert_registered_refusal(info.value)
    assert info.value.reason_code == "discovered_structure_unresolved"
    assert info.value.detail == "selective_discovery.differences_are_not_selection_targets"
    assert "2 edge(s) differ" in str(info.value.remedy)
    with pytest.raises(CausalValueError):
        driver.request_selection_targets(["nope"])
