"""Background knowledge through the ordinary discovery classes, and masked/weighted splits.

Oracles are worked out by hand, not by the library:

* a linear-Gaussian chain ``a -> b -> c`` has no v-structure, so PC returns the undirected chain
  and every orientation in these tests comes from the knowledge (and Meek rule 1 from it);
* FCI reads a later tier as "no tail at the later end": an arrowhead there, nothing at the
  other end, and rule R1 then orients the next edge of the chain;
* the Kish effective sample size of weights ``w`` is ``(sum w)^2 / sum w^2``.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from antecedent import _native
from antecedent import selective_discovery as sd
from antecedent.accepted_graph import AcceptedGraph
from antecedent.discovery import (
    FCI,
    GES,
    PC,
    PCMCI,
    RFCI,
    BackgroundKnowledge,
    Link,
    PCMCIPlus,
)
from antecedent.errors import CausalUnsupportedError

N = 3000


def chain(seed: int = 7) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=N)
    b = 0.9 * a + 0.5 * rng.normal(size=N)
    c = 0.9 * b + 0.5 * rng.normal(size=N)
    return {"a": a, "b": b, "c": c}


def marks(result, x: str, y: str) -> tuple[str, str] | None:
    """The marks (at x, at y) of the edge between x and y, or None when absent."""
    for edge in result.graph_edges:
        if (edge.source, edge.target) == (x, y):
            return edge.at_source, edge.at_target
        if (edge.source, edge.target) == (y, x):
            return edge.at_target, edge.at_source
    return None


def refused(error: pytest.ExceptionInfo[CausalUnsupportedError], code: str) -> None:
    assert error.value.reason_code == code, error.value


# -- the statements orient what the data leave undirected ------------------------------------


def test_pc_without_knowledge_leaves_the_chain_undirected() -> None:
    result = PC(alpha=0.001, fdr=False).run(chain())
    assert marks(result, "a", "b") == ("tail", "tail")
    assert marks(result, "b", "c") == ("tail", "tail")
    assert marks(result, "a", "c") is None
    assert result.background_knowledge is None


def test_pc_required_link_orients_the_chain_and_meek_propagates() -> None:
    knowledge = BackgroundKnowledge(required=(Link("a", "b"),))
    result = PC(alpha=0.001, fdr=False, knowledge=knowledge).run(chain())
    # a -> b is stated; a and c are not adjacent, so Meek rule 1 gives b -> c.
    assert marks(result, "a", "b") == ("tail", "arrow")
    assert marks(result, "b", "c") == ("tail", "arrow")


def test_pc_tiers_and_forbidden_links_orient_across_the_pair() -> None:
    tiers = BackgroundKnowledge(tiers=(("a",), ("b",), ("c",)))
    result = PC(alpha=0.001, fdr=False, knowledge=tiers).run(chain())
    assert marks(result, "a", "b") == ("tail", "arrow")
    assert marks(result, "b", "c") == ("tail", "arrow")

    # Forbidding c -> b leaves b -> c for that pair. Nothing else fixes a - b: no Meek rule
    # orients it from b -> c alone, so it stays undirected.
    forbidden = BackgroundKnowledge(forbidden=(Link("c", "b"),))
    result = PC(alpha=0.001, fdr=False, knowledge=forbidden).run(chain())
    assert marks(result, "b", "c") == ("tail", "arrow")
    assert marks(result, "a", "b") == ("tail", "tail")


def test_forbidding_both_directions_removes_the_adjacency() -> None:
    knowledge = BackgroundKnowledge(forbidden=(Link("a", "b"), Link("b", "a")))
    result = PC(alpha=0.001, fdr=False, knowledge=knowledge).run(chain())
    assert marks(result, "a", "b") is None


def test_fci_places_arrowheads_only_for_tiers_and_tail_and_arrow_for_required() -> None:
    tiers = BackgroundKnowledge(tiers=(("a",), ("b",), ("c",)))
    result = FCI(alpha=0.001, fdr=False, knowledge=tiers).run(chain())
    # No tail at the later end of a-b; the earlier end stays a circle. R1 then gives b -> c.
    assert marks(result, "a", "b") == ("circle", "arrow")
    assert marks(result, "b", "c") == ("tail", "arrow")

    required = BackgroundKnowledge(required=(Link("b", "a"),))
    result = FCI(alpha=0.001, fdr=False, knowledge=required).run(chain())
    assert marks(result, "a", "b") == ("arrow", "tail")


def test_ges_orients_the_chain_from_tiers() -> None:
    knowledge = BackgroundKnowledge(tiers=(("c",), ("b",), ("a",)))
    result = GES(alpha=0.001, fdr=False, knowledge=knowledge).run(chain())
    # Every adjacent pair is oriented from the earlier tier (c) to the later (a).
    assert marks(result, "b", "c") == ("arrow", "tail")
    assert marks(result, "a", "b") == ("arrow", "tail")


# -- parent limits, contradictions, unknown names ---------------------------------------------


def test_a_parent_limit_is_enforced_by_every_algorithm_that_takes_one() -> None:
    capped = BackgroundKnowledge(max_parents=0)
    for algorithm in (PC, FCI, GES):
        result = algorithm(alpha=0.001, fdr=False, knowledge=capped).run(chain())
        # With no adjacency allowed to be a parent, no edge survives.
        assert result.graph_edges == [], algorithm
    # A limit that binds is reported for the constraint-based searches; the chain has
    # two possible parents of b, so a limit of one removes the weaker adjacency.
    one = BackgroundKnowledge(max_parents=1)
    result = PC(alpha=0.001, fdr=False, knowledge=one).run(chain())
    assert marks(result, "a", "b") is None or marks(result, "b", "c") is None


def test_contradictory_knowledge_is_refused() -> None:
    cases = {
        "required and forbidden": BackgroundKnowledge(
            required=(Link("a", "b"),), forbidden=(Link("a", "b"),)
        ),
        "required against a tier": BackgroundKnowledge(
            required=(Link("c", "a"),), tiers=(("a",), ("c",))
        ),
        "required cycle": BackgroundKnowledge(
            required=(Link("a", "b"), Link("b", "c"), Link("c", "a"))
        ),
    }
    for name, knowledge in cases.items():
        with pytest.raises(CausalUnsupportedError) as error:
            PC(alpha=0.001, fdr=False, knowledge=knowledge).run(chain())
        assert error.value.reason_code == "invalid_argument", name


def test_unknown_variables_are_refused_by_name_and_by_position() -> None:
    knowledge = BackgroundKnowledge(forbidden=(Link("a", "nope"),))
    with pytest.raises(CausalUnsupportedError) as error:
        PC(alpha=0.001, fdr=False, knowledge=knowledge).run(chain())
    refused(error, "unknown_variable")

    # A position outside the columns reaches the native layer and is refused there too.
    data = chain()
    wire = json.dumps({"forbidden": [[0, 0, 9, 0]]})
    with pytest.raises(CausalUnsupportedError) as error:
        _native.discover_pc(
            list(data), [np.ascontiguousarray(v) for v in data.values()], background_knowledge=wire
        )
    refused(error, "unknown_variable")


def test_a_lagged_link_in_a_search_over_a_table_is_refused() -> None:
    knowledge = BackgroundKnowledge(forbidden=(Link("a", "b", source_lag=1),))
    with pytest.raises(CausalUnsupportedError) as error:
        PC(alpha=0.001, fdr=False, knowledge=knowledge).run(chain())
    refused(error, "option_not_applicable")


# -- lagged search ----------------------------------------------------------------------------


def lag1_series(n: int = 600) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(3)
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    y = np.zeros(n)
    y[1:] = 0.8 * x[:-1] + 0.3 * rng.normal(size=n - 1)
    return {"x": x, "y": y, "z": z}


def links(result) -> set[tuple[str, int, str, int]]:
    return {(k.source, k.source_lag, k.target, k.target_lag) for k in result.links}


def test_pcmci_forbidden_lagged_link_is_not_a_candidate() -> None:
    data = lag1_series()
    names = list(data)
    columns = [data[n] for n in names]
    plain = PCMCI(max_lag=1, alpha=0.01, fdr=False).run((names, columns), seed=1)
    assert ("x", 1, "y", 0) in links(plain)

    knowledge = BackgroundKnowledge(forbidden=(Link("x", "y", source_lag=1),))
    constrained = PCMCI(max_lag=1, alpha=0.01, fdr=False, knowledge=knowledge).run(
        (names, columns), seed=1
    )
    assert ("x", 1, "y", 0) not in links(constrained)
    assert constrained.background_knowledge is not None


def test_pcmci_accepts_a_parent_limit_and_refuses_a_required_link_outside_its_window() -> None:
    data = lag1_series()
    names = list(data)
    columns = [data[n] for n in names]
    limited = BackgroundKnowledge(max_parents=1)
    PCMCI(max_lag=1, fdr=False, knowledge=limited).run((names, columns), seed=1)

    contemporaneous = BackgroundKnowledge(required=(Link("x", "y"),))
    with pytest.raises(CausalUnsupportedError) as error:
        PCMCI(max_lag=1, fdr=False, knowledge=contemporaneous).run((names, columns), seed=1)
    refused(error, "option_not_applicable")


# -- identity ---------------------------------------------------------------------------------


def test_the_digest_changes_with_every_statement_and_ignores_link_order() -> None:
    data = chain()
    base = BackgroundKnowledge(
        required=(Link("a", "b"), Link("b", "c")),
        forbidden=(Link("c", "a"),),
        tiers=(("a",), ("b", "c")),
    )

    def digest(knowledge: BackgroundKnowledge | None) -> str | None:
        return PC(alpha=0.001, fdr=False, knowledge=knowledge).run(data).background_knowledge

    reference = digest(base)
    assert reference is not None and digest(None) is None
    assert digest(BackgroundKnowledge()) is None  # nothing asserted is no knowledge
    reordered = BackgroundKnowledge(
        required=(Link("b", "c"), Link("a", "b")), forbidden=base.forbidden, tiers=base.tiers
    )
    assert digest(reordered) == reference
    edits = [
        BackgroundKnowledge(required=(Link("a", "b"),), forbidden=base.forbidden, tiers=base.tiers),
        BackgroundKnowledge(required=base.required, tiers=base.tiers),
        BackgroundKnowledge(required=base.required, forbidden=base.forbidden),
    ]
    digests = {digest(edit) for edit in edits}
    assert reference not in digests and len(digests) == len(edits)


def test_the_algorithm_wire_and_the_accepted_graph_carry_the_knowledge() -> None:
    knowledge = BackgroundKnowledge(required=(Link("a", "b"),), max_parents=None)
    config = PC(alpha=0.001, fdr=False, knowledge=knowledge)
    assert config._wire()["background_knowledge"]["required"] == [["a", 0, "b", 0]]
    assert "background_knowledge" not in PC()._wire()

    data = chain()
    accepted = config.accept(data)
    assert accepted.background_knowledge == config.run(data).background_knowledge
    assert accepted.background_knowledge is not None
    restored = AcceptedGraph.from_json(accepted.to_json())
    assert restored.background_knowledge == accepted.background_knowledge

    other = PC(alpha=0.001, fdr=False, knowledge=BackgroundKnowledge(required=(Link("b", "a"),)))
    rediscovered = accepted.rediscover(data, other)
    assert rediscovered.background_knowledge != accepted.background_knowledge
    plain = AcceptedGraph.from_discovery(PC(alpha=0.001, fdr=False).run(data), algorithm_id="pc")
    assert plain.background_knowledge is None


def test_selective_contracts_share_the_statement_and_the_name_translation() -> None:
    knowledge = BackgroundKnowledge(required=(Link("a", "b"),), tiers=(("a",), ("b", "c")))
    wire = knowledge.to_wire(("a", "b", "c"))
    contract = sd.StaticContract(
        variables=("a", "b", "c"), required=(Link("a", "b"),), tiers=(("a",), ("b", "c"))
    )
    contract_wire = contract.to_wire()
    for key in ("required", "forbidden", "tiers"):
        assert contract_wire[key] == wire[key]
    with pytest.raises(CausalUnsupportedError) as error:
        sd.StaticContract(variables=("a", "b"), required=(Link("a", "zzz"),)).to_wire()
    refused(error, "unknown_variable")


# -- masked and weighted splits ---------------------------------------------------------------


def split_contract(ci_test: str = "parcorr") -> sd.StaticContract:
    return sd.StaticContract(
        variables=("a", "b", "c"), ci_test=ci_test, alpha=0.001, fdr=None, seed=1
    )


def test_a_masked_split_records_the_exclusions_and_keeps_original_row_ids() -> None:
    data = chain()
    keep = np.ones(N, dtype=bool)
    keep[[0, 17]] = False
    split = sd.discover_on_split(split_contract(), data, discovery_fraction=0.4, seed=5, mask=keep)
    receipt = split.receipt
    assert receipt.masked_out_rows == (0, 17)
    assert receipt.incomplete_rows == ()
    assert receipt.n_discovery + receipt.n_analysis == N - 2
    ids = np.concatenate([split.discovery_row_ids, split.analysis_row_ids])
    assert sorted(ids.tolist()) == [r for r in range(N) if r not in (0, 17)]
    # The analysis values are the source values at the analysis row ids.
    assert np.array_equal(split.analysis["b"], data["b"][split.analysis_row_ids])
    split.verify()
    split.verify_source(data, mask=keep)
    with pytest.raises(sd.SelectiveDiscoveryRefusal) as error:
        split.verify_source(data)
    assert error.value.detail == "selective_discovery.split_mask_mismatch"


def test_incomplete_rows_are_excluded_before_the_split() -> None:
    data = chain()
    data["b"] = data["b"].copy()
    data["b"][[4, 9]] = np.nan
    split = sd.discover_on_split(split_contract(), data, discovery_fraction=0.5, seed=2)
    assert split.receipt.incomplete_rows == (4, 9)
    assert not {4, 9} & set(split.analysis_row_ids.tolist() + split.discovery_row_ids.tolist())
    assert split.receipt.n_discovery + split.receipt.n_analysis == N - 2


def test_a_weighted_split_gathers_weights_and_reports_the_effective_size() -> None:
    data = chain()
    weights = np.where(np.arange(N) % 4 == 0, 0.0, 1.0 + (np.arange(N) % 3))
    split = sd.discover_on_split(
        split_contract("weighted_parcorr"),
        data,
        discovery_fraction=0.5,
        seed=8,
        weights=weights,
    )
    gathered = weights[split.analysis_row_ids]
    assert np.array_equal(split.analysis_weights, gathered)
    kish = gathered.sum() ** 2 / (gathered**2).sum()
    assert split.analysis_effective_sample_size == pytest.approx(kish)
    assert gathered.sum() > 1.0  # unnormalized
    split.verify()
    split.verify_source(data, weights=weights)

    # Other weights are another source, and re-weighted analysis rows are other rows.
    with pytest.raises(sd.SelectiveDiscoveryRefusal) as error:
        split.verify_source(data, weights=np.ones(N))
    assert error.value.detail == "selective_discovery.split_weights_mismatch"
    split.check_analysis_table(split.analysis, weights=gathered)
    with pytest.raises(sd.SelectiveDiscoveryRefusal) as error:
        split.check_analysis_table(split.analysis, weights=np.ones(len(gathered)))
    assert error.value.detail == "selective_discovery.analysis_rows_not_the_split_part"


def test_weights_must_match_the_declared_test() -> None:
    data = chain()
    weights = np.ones(N)
    with pytest.raises(sd.SelectiveDiscoveryRefusal) as error:
        sd.discover_on_split(
            split_contract(), data, discovery_fraction=0.5, seed=1, weights=weights
        )
    assert error.value.detail == "selective_discovery.weights_not_read_by_test"
    with pytest.raises(sd.SelectiveDiscoveryRefusal) as error:
        sd.discover_on_split(
            split_contract("weighted_parcorr"), data, discovery_fraction=0.5, seed=1
        )
    assert error.value.detail == "selective_discovery.weighted_test_without_weights"


def test_rfci_reads_the_same_statements_as_fci() -> None:
    required = BackgroundKnowledge(required=(Link("b", "a"),))
    result = RFCI(alpha=0.001, fdr=False, knowledge=required).run(chain())
    assert marks(result, "a", "b") == ("arrow", "tail")
    assert result.background_knowledge is not None
    contradictory = BackgroundKnowledge(required=(Link("a", "b"),), forbidden=(Link("a", "b"),))
    with pytest.raises(CausalUnsupportedError) as error:
        RFCI(alpha=0.001, fdr=False, knowledge=contradictory).run(chain())
    refused(error, "invalid_argument")


def contemporaneous_pair(n: int = 1500) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(5)
    x = rng.normal(size=n)
    return {"x": x, "y": 0.9 * x + 0.4 * rng.normal(size=n), "z": rng.normal(size=n)}


def test_pcmci_plus_orients_a_contemporaneous_edge_from_knowledge() -> None:
    data = contemporaneous_pair()
    names = list(data)
    table = (names, [data[n] for n in names])
    plain = PCMCIPlus(max_lag=1, alpha=0.001, fdr=False).run(table, seed=1)
    assert marks(plain, "x", "y") == ("tail", "tail")

    tiered = BackgroundKnowledge(tiers=(("x",), ("y",)))
    result = PCMCIPlus(max_lag=1, alpha=0.001, fdr=False, knowledge=tiered).run(table, seed=1)
    assert marks(result, "x", "y") == ("tail", "arrow")
    assert result.background_knowledge is not None

    reverse = BackgroundKnowledge(forbidden=(Link("x", "y"),))
    result = PCMCIPlus(max_lag=1, alpha=0.001, fdr=False, knowledge=reverse).run(table, seed=1)
    assert marks(result, "x", "y") == ("arrow", "tail")

    clash = BackgroundKnowledge(required=(Link("x", "y"),), tiers=(("y",), ("x",)))
    with pytest.raises(CausalUnsupportedError) as error:
        PCMCIPlus(max_lag=1, fdr=False, knowledge=clash).run(table, seed=1)
    refused(error, "invalid_argument")
