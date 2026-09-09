"""Wire coercion for target-population specs (no native calls)."""

from __future__ import annotations

import pytest
from antecedent.population import (
    AllRows,
    CustomDistribution,
    Named,
    Population,
    PopulationRegistry,
    Rows,
    Treated,
    Untreated,
    coerce_target_population,
    registry_wire,
    target_all,
    target_custom_distribution,
    target_named,
    target_rows,
    target_treated,
    target_untreated,
)


def test_helpers_and_registry_round_trip():
    assert target_all()._wire() == {"kind": "all"}
    assert target_treated()._wire() == {"kind": "treated"}
    assert target_untreated()._wire() == {"kind": "untreated"}
    assert target_named("high_risk")._wire() == {"kind": "named", "name": "high_risk"}
    assert target_rows([3, 1])._wire() == {"kind": "rows", "rows": [3, 1]}
    assert target_custom_distribution(7)._wire() == {"kind": "custom_distribution", "id": 7}

    registry = PopulationRegistry()
    registry.insert_predicate("high_risk", (1, 2))
    registry.insert_distribution(7, (0.5, 1.5))
    predicates, distributions = registry_wire(registry)
    assert predicates == {"high_risk": [1, 2]}
    assert distributions == {7: [0.5, 1.5]}
    assert registry_wire(None) == ({}, {})


def test_coerce_accepts_instances_strings_and_mappings():
    assert coerce_target_population(None) is None
    assert coerce_target_population(AllRows()) == {"kind": "all"}
    assert coerce_target_population(Treated()) == {"kind": "treated"}
    assert coerce_target_population(Untreated()) == {"kind": "untreated"}
    assert coerce_target_population(Named("g")) == {"kind": "named", "name": "g"}
    assert coerce_target_population(Rows((0,))) == {"kind": "rows", "rows": [0]}
    assert coerce_target_population(CustomDistribution(2)) == {
        "kind": "custom_distribution",
        "id": 2,
    }

    assert coerce_target_population("all") == {"kind": "all"}
    assert coerce_target_population("all_observed") == {"kind": "all"}
    assert coerce_target_population("Observed") == {"kind": "all"}
    assert coerce_target_population("treated") == {"kind": "treated"}
    assert coerce_target_population("untreated") == {"kind": "untreated"}
    assert coerce_target_population("control") == {"kind": "untreated"}

    assert coerce_target_population({"kind": "all"}) == {"kind": "all"}
    assert coerce_target_population({"kind": "all_observed"}) == {"kind": "all"}
    assert coerce_target_population({"kind": "treated"}) == {"kind": "treated"}
    assert coerce_target_population({"kind": "untreated"}) == {"kind": "untreated"}
    assert coerce_target_population({"kind": "named", "name": "g"}) == {
        "kind": "named",
        "name": "g",
    }
    assert coerce_target_population({"kind": "rows", "rows": [4]}) == {"kind": "rows", "rows": [4]}
    assert coerce_target_population({"kind": "custom", "id": 9}) == {
        "kind": "custom_distribution",
        "id": 9,
    }
    assert coerce_target_population({"kind": "custom_distribution", "id": 9}) == {
        "kind": "custom_distribution",
        "id": 9,
    }


def test_coerce_rejects_unknown_and_base_class_is_abstract():
    with pytest.raises(ValueError, match="unknown target_population string"):
        coerce_target_population("att")
    with pytest.raises(ValueError, match="unknown target_population mapping"):
        coerce_target_population({"kind": "att"})
    with pytest.raises(TypeError, match="unsupported target_population type"):
        coerce_target_population(3)

    class Bare(Population):
        pass

    with pytest.raises(NotImplementedError):
        Bare()._wire()
