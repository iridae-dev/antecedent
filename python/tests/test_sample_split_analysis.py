"""Discovery followed by analysis on a row split through the Python surface.

Oracles: the seeded permutation assigns ``round(fraction * n)`` rows to discovery (arithmetic);
the sorted union of the two parts of a column equals the sorted column (the parts are a
partition of the rows); an explicit rule names the discovery rows, so the parts are slices; and
discovery on the split equals, bit for bit, a direct run on the discovery part.
"""

from __future__ import annotations

import numpy as np
import pytest
from antecedent import selective_discovery as sd
from antecedent.errors import CausalValueError

from _refusal import assert_registered_refusal

N = 500


def table(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=N)
    y = 0.8 * x + 0.6 * rng.normal(size=N)
    z = 0.8 * y + 0.6 * rng.normal(size=N)
    return {"x": x, "y": y, "z": z}


CONTRACT = sd.StaticContract(variables=("x", "y", "z"), alpha=0.001, fdr=None, seed=9)
WRONG = "selective_discovery.analysis_rows_not_the_split_part"


def test_the_parts_partition_the_rows_and_discovery_sees_only_its_part() -> None:
    data = table(1)
    split = sd.discover_on_split(CONTRACT, data, discovery_fraction=0.4, seed=5)

    receipt = split.receipt
    assert (receipt.n_rows, receipt.n_discovery, receipt.n_analysis) == (N, 200, 300)
    assert receipt.whole_method_calibration == "unmeasured"
    assert receipt.conditional_on_selected_graph
    split.verify()

    for name, column in data.items():
        merged = np.concatenate([split.discovery[name], split.analysis[name]])
        assert len(split.discovery[name]) == 200
        np.testing.assert_array_equal(np.sort(merged), np.sort(column))

    direct = sd.SelectiveDiscovery("static")
    direct.run(CONTRACT, split.discovery)
    assert split.outcome.result_identity == direct.outcome.result_identity  # type: ignore[union-attr]
    assert split.outcome.structure_identity == direct.outcome.structure_identity  # type: ignore[union-attr]
    assert len(split.outcome.unresolved_edges) == 2
    assert split.report.work.ci_tests_attempted > 0


def test_an_explicit_rule_names_the_discovery_rows() -> None:
    data = table(2)
    split = sd.discover_on_split(CONTRACT, data, discovery_rows=range(250))
    assert split.receipt.n_discovery == 250
    for name, column in data.items():
        np.testing.assert_array_equal(split.discovery[name], column[:250])
        np.testing.assert_array_equal(split.analysis[name], column[250:])


def test_only_the_analysis_part_passes_the_check_and_the_binding_says_what_it_licenses() -> None:
    data = table(3)
    split = sd.discover_on_split(CONTRACT, data, discovery_fraction=0.4, seed=6)
    binding = split.check_analysis_table(split.analysis)
    assert binding.conditional_on_selected_graph
    assert binding.whole_method_calibration == "unmeasured"
    assert binding.receipt_identity == split.receipt.identity
    assert binding.structure_identity == split.outcome.structure_identity

    for rejected in (split.discovery, data, table(4)):
        with pytest.raises(sd.SelectiveDiscoveryRefusal) as info:
            split.check_analysis_table(rejected)
        assert_registered_refusal(info.value)
        assert info.value.detail == WRONG
    other = sd.discover_on_split(CONTRACT, table(4), discovery_fraction=0.4, seed=6)
    with pytest.raises(sd.SelectiveDiscoveryRefusal) as info:
        split.check_analysis_table(other.analysis)
    assert info.value.detail == WRONG


def test_a_changed_rule_is_a_new_receipt_and_a_new_binding() -> None:
    data = table(5)
    base = sd.discover_on_split(CONTRACT, data, discovery_fraction=0.4, seed=7)
    again = sd.discover_on_split(CONTRACT, data, discovery_fraction=0.4, seed=7)
    assert base.receipt.identity == again.receipt.identity
    assert (
        base.check_analysis_table(base.analysis).identity
        == again.check_analysis_table(again.analysis).identity
    )
    reseeded = sd.discover_on_split(CONTRACT, data, discovery_fraction=0.4, seed=8)
    assert reseeded.receipt.identity != base.receipt.identity
    assert (
        reseeded.check_analysis_table(reseeded.analysis).identity
        != base.check_analysis_table(base.analysis).identity
    )
    wider = sd.discover_on_split(CONTRACT, data, discovery_fraction=0.6, seed=7)
    assert wider.receipt.n_discovery == 300


def test_reusing_the_rows_that_selected_is_refused() -> None:
    split = sd.discover_on_split(CONTRACT, table(6), discovery_fraction=0.4, seed=9)
    with pytest.raises(sd.SelectiveDiscoveryRefusal) as info:
        split.request_unsplit_interval()
    assert_registered_refusal(info.value)
    assert info.value.reason_code == "cell_not_licensed"
    assert info.value.detail == "selective_discovery.downstream_interval_not_licensed"


@pytest.mark.parametrize(
    ("kwargs", "detail"),
    [
        ({"discovery_rows": [3, 3]}, "selective_discovery.split_rows_repeated"),
        ({"discovery_rows": [N]}, "selective_discovery.split_row_out_of_range"),
        ({"discovery_rows": list(range(N))}, "selective_discovery.split_part_empty"),
        (
            {"discovery_fraction": 1.0, "seed": 1},
            "selective_discovery.split_fraction_invalid",
        ),
    ],
)
def test_malformed_rules_are_refused_with_registered_details(
    kwargs: dict[str, object], detail: str
) -> None:
    with pytest.raises(sd.SelectiveDiscoveryRefusal) as info:
        sd.discover_on_split(CONTRACT, table(7), **kwargs)  # type: ignore[arg-type]
    assert_registered_refusal(info.value)
    assert info.value.detail == detail


def test_an_undeclared_rule_is_a_value_error() -> None:
    with pytest.raises(CausalValueError):
        sd.discover_on_split(CONTRACT, table(8))
    with pytest.raises(CausalValueError):
        sd.discover_on_split(CONTRACT, table(8), discovery_rows=[1], discovery_fraction=0.5, seed=1)


def test_the_split_parts_are_independent_ledger_sources_that_overlap_the_table_by_their_rows() -> (
    None
):
    from antecedent import evidence_ledger as el

    split = sd.discover_on_split(CONTRACT, table(4), discovery_fraction=0.4, seed=7)
    snapshot = b"split-table"
    discovery, analysis = split.evidence_sources(snapshot)
    assert discovery.kind == analysis.kind == "observation_set"
    assert len(discovery.observation_ids or ()) == 200
    assert len(analysis.observation_ids or ()) == 300

    ledger = el.EvidenceLedger()
    d, a = ledger.add_source(discovery), ledger.add_source(analysis)
    whole = ledger.add_source(sd.discovery_data_source("unsplit table", snapshot, range(N)))
    assert ledger.account_observations([d, a]).pairs[0].overlap == "independent"
    assert ledger.account_observations([d, whole]).pairs[0].shared == 200
    assert ledger.account_observations([a, whole]).pairs[0].shared == 300

    # A different snapshot id shares no observation with the table.
    other = ledger.add_source(sd.discovery_data_source("other table", b"other", range(N)))
    assert ledger.account_observations([d, other]).pairs[0].overlap == "independent"
