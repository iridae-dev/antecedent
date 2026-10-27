"""Masked, survey-weighted sample splitting of 3,000 participants and ten variables.

The table is a sparse linear-Gaussian study with survey expansion weights of coefficient of
variation 0.8 (log-normal, a function of the design alone) and 300 rows outside the analysis
mask. What the split must keep is worked out here from the inputs: the exclusions and their
reasons, the original row of every part row, the weights of those rows bit for bit and their
Kish effective sizes, and a replay that reproduces the receipt.
"""

from __future__ import annotations

import numpy as np
import pytest
from antecedent import selective_discovery as sd

from _repo_text import REPO_ROOT, read_text

N = 3000
MASKED = 300
NAMES = [f"v{i}" for i in range(10)]
GUIDE = REPO_ROOT / "docs/guides/discovery-background-knowledge.md"


def survey_weighted_sparse_scm(seed: int, *, cv: float = 0.8):
    """Ten variables (two chains, a collider at v4, one at v8), the weights and a mask.

    Returns ``(data, weights, keep)``: ``keep`` is False for the ``MASKED`` rows outside the
    analysis mask. The weights are ``exp(s z - s^2 / 2)`` with ``s^2 = ln(1 + cv^2)``, so they
    have mean one and coefficient of variation ``cv``.
    """
    rng = np.random.default_rng(seed)
    noise = lambda: rng.normal(size=N)  # noqa: E731 — one draw per variable, in order
    v0 = noise()
    v1 = 0.8 * v0 + 0.6 * noise()
    v2 = 0.8 * v1 + 0.6 * noise()
    v3 = 0.8 * v2 + 0.6 * noise()
    v5 = noise()
    v4 = 0.7 * v3 + 0.7 * v5 + 0.5 * noise()
    v6 = 0.8 * v5 + 0.6 * noise()
    v7 = 0.8 * v6 + 0.6 * noise()
    v9 = noise()
    v8 = 0.6 * v7 + 0.6 * v9 + 0.5 * noise()
    data = dict(zip(NAMES, [v0, v1, v2, v3, v4, v5, v6, v7, v8, v9], strict=True))
    sigma2 = np.log1p(cv**2)
    weights = np.exp(np.sqrt(sigma2) * rng.normal(size=N) - sigma2 / 2.0)
    keep = np.ones(N, dtype=bool)
    keep[rng.choice(N, size=MASKED, replace=False)] = False
    return data, weights, keep


def contract(ci_test: str = "weighted_parcorr") -> sd.StaticContract:
    return sd.StaticContract(variables=tuple(NAMES), ci_test=ci_test, alpha=0.01, fdr=None, seed=3)


def kish(weights: np.ndarray) -> float:
    return float(weights.sum() ** 2 / (weights**2).sum())


def test_a_masked_weighted_split_keeps_masks_weights_and_original_row_identity():
    data, weights, keep = survey_weighted_sparse_scm(9290586)
    assert weights.std() / weights.mean() == pytest.approx(0.8, abs=0.1)
    split = sd.discover_on_split(
        contract(), data, discovery_fraction=0.5, split_seed=7, weights=weights, mask=keep
    )
    receipt = split.receipt
    masked = tuple(int(row) for row in np.flatnonzero(~keep))
    assert len(masked) == MASKED and receipt.masked_out_rows == masked
    assert receipt.incomplete_rows == () and receipt.n_rows == N
    assert receipt.n_discovery == receipt.n_analysis == (N - MASKED) // 2
    # Row identity: both parts hold original rows, ascending, disjoint, covering the mask.
    discovery_ids, analysis_ids = split.discovery_row_ids, split.analysis_row_ids
    assert np.all(np.diff(discovery_ids) > 0) and np.all(np.diff(analysis_ids) > 0)
    assert not set(discovery_ids.tolist()) & set(analysis_ids.tolist())
    assert (
        sorted([*discovery_ids.tolist(), *analysis_ids.tolist()]) == np.flatnonzero(keep).tolist()
    )
    assert not set(masked) & {*discovery_ids.tolist(), *analysis_ids.tolist()}
    for name in NAMES:
        assert np.array_equal(split.discovery[name], data[name][discovery_ids])
        assert np.array_equal(split.analysis[name], data[name][analysis_ids])
    # Weights: carried bit for bit and unnormalized, with each part's Kish effective size.
    assert np.array_equal(split.analysis_weights, weights[analysis_ids])
    assert split.analysis_weights.sum() != pytest.approx(len(analysis_ids))
    recorded = receipt.weights
    assert recorded["analysis_effective_sample_size"] == pytest.approx(kish(weights[analysis_ids]))
    assert recorded["discovery_effective_sample_size"] == pytest.approx(
        kish(weights[discovery_ids])
    )
    assert split.analysis_effective_sample_size == pytest.approx(kish(weights[analysis_ids]))
    assert len({recorded["source"], recorded["discovery"], recorded["analysis"]}) == 3
    # Masks: the receipt names the mask the rows were filtered by, and a source with another
    # mask, other weights, or no mask or weights at all is not the table that was split.
    assert receipt.mask_identity is not None
    split.verify()
    split.verify_source(data, weights=weights, mask=keep)
    other_mask = keep.copy()
    other_mask[int(np.flatnonzero(keep)[0])] = False
    for kwargs, detail in (
        ({"weights": weights, "mask": other_mask}, "selective_discovery.split_mask_mismatch"),
        ({"weights": weights}, "selective_discovery.split_mask_mismatch"),
        ({"weights": np.ones(N), "mask": keep}, "selective_discovery.split_weights_mismatch"),
    ):
        with pytest.raises(sd.SelectiveDiscoveryRefusal) as refused:
            split.verify_source(data, **kwargs)
        assert refused.value.detail == detail
    # Both parts are exactly the analysis table: re-weighted rows are other rows.
    split.check_analysis_table(split.analysis, weights=weights[analysis_ids])
    with pytest.raises(sd.SelectiveDiscoveryRefusal) as refused:
        split.check_analysis_table(split.analysis, weights=np.ones(len(analysis_ids)))
    assert refused.value.detail == "selective_discovery.analysis_rows_not_the_split_part"
    unmasked = sd.discover_on_split(
        contract(), data, discovery_fraction=0.5, split_seed=7, weights=weights
    )
    assert unmasked.receipt.mask_identity is None and unmasked.receipt.masked_out_rows == ()
    assert unmasked.receipt.identity != receipt.identity


def test_filtering_happens_before_the_split_and_the_order_is_declared_and_reproducible():
    data, weights, keep = survey_weighted_sparse_scm(8571308)
    data["v2"] = data["v2"].copy()
    masked_rows = np.flatnonzero(~keep)
    kept_rows = np.flatnonzero(keep)
    # 40 incomplete rows among the kept ones, and 10 incomplete rows that the mask also hides.
    incomplete = kept_rows[::65][:40]
    data["v2"][incomplete] = np.nan
    data["v2"][masked_rows[:10]] = np.nan
    eligible = np.setdiff1d(kept_rows, incomplete)
    assert len(eligible) == N - MASKED - 40

    def split(seed: int):
        return sd.discover_on_split(
            contract(), data, discovery_fraction=0.4, split_seed=seed, weights=weights, mask=keep
        )

    first = split(11)
    receipt = first.receipt
    # The mask comes first: a row both masked out and incomplete is a masked-out row.
    assert receipt.masked_out_rows == tuple(int(r) for r in masked_rows)
    assert receipt.incomplete_rows == tuple(int(r) for r in incomplete)
    # The rule splits the eligible rows: its part sizes are the fraction of those, not of N.
    assert receipt.n_discovery == round(0.4 * len(eligible))
    assert receipt.n_discovery + receipt.n_analysis == len(eligible)
    assert sorted([*first.discovery_row_ids.tolist(), *first.analysis_row_ids.tolist()]) == (
        eligible.tolist()
    )
    # The excluded rows never consume a draw: splitting the table with those rows removed
    # assigns the same participants to discovery.
    compact = {name: values[eligible] for name, values in data.items()}
    reduced = sd.discover_on_split(
        contract(), compact, discovery_fraction=0.4, split_seed=11, weights=weights[eligible]
    )
    assert eligible[reduced.discovery_row_ids].tolist() == first.discovery_row_ids.tolist()
    # Reproducible: the same declaration is the same split, another seed is another one.
    again = split(11)
    assert again.receipt == receipt
    assert again.receipt.identity == receipt.identity
    assert np.array_equal(again.discovery_row_ids, first.discovery_row_ids)
    assert not np.array_equal(split(12).discovery_row_ids, first.discovery_row_ids)
    # The order is part of the documented method.
    guide = read_text(GUIDE)
    assert "`selective_discovery.discover_on_split` filters first and splits second" in guide
    assert guide.index("rows outside the analysis mask") < guide.index("rows with a missing value")
    # A rule that names an excluded row is refused rather than reassigned.
    with pytest.raises(sd.SelectiveDiscoveryRefusal) as refused:
        sd.discover_on_split(
            contract(),
            data,
            discovery_rows=[int(incomplete[0]), *eligible[:100].tolist()],
            weights=weights,
            mask=keep,
        )
    assert refused.value.detail == "selective_discovery.split_row_excluded"


def test_the_split_receipt_lists_every_exclusion_by_its_reason():
    data, weights, keep = survey_weighted_sparse_scm(2266102)
    data["v5"] = data["v5"].copy()
    kept_rows = np.flatnonzero(keep)
    incomplete = kept_rows[5::97]
    data["v5"][incomplete] = np.nan
    data["v5"][np.flatnonzero(~keep)[::3]] = np.nan  # hidden by the mask first
    split = sd.discover_on_split(
        contract(), data, discovery_fraction=0.5, split_seed=4, weights=weights, mask=keep
    )
    receipt = split.receipt
    by_reason = {
        "analysis_mask": receipt.masked_out_rows,
        "incomplete_searched_variable": receipt.incomplete_rows,
    }
    assert by_reason["analysis_mask"] == tuple(int(r) for r in np.flatnonzero(~keep))
    assert by_reason["incomplete_searched_variable"] == tuple(int(r) for r in incomplete)
    assert not set(by_reason["analysis_mask"]) & set(by_reason["incomplete_searched_variable"])
    excluded = {*by_reason["analysis_mask"], *by_reason["incomplete_searched_variable"]}
    assert receipt.n_rows - len(excluded) == receipt.n_discovery + receipt.n_analysis
    assert excluded.isdisjoint(
        {*split.discovery_row_ids.tolist(), *split.analysis_row_ids.tolist()}
    )
    # The receipt's exclusions are covered by its identity: removing one changes the receipt.
    split.verify()
    clean = sd.discover_on_split(
        contract(),
        {**data, "v5": np.nan_to_num(data["v5"])},
        discovery_fraction=0.5,
        split_seed=4,
        weights=weights,
        mask=keep,
    )
    assert clean.receipt.incomplete_rows == () and clean.receipt.identity != receipt.identity


def test_replaying_the_split_reproduces_the_receipt():
    data, weights, keep = survey_weighted_sparse_scm(1346138)
    split = sd.discover_on_split(
        contract(), data, discovery_fraction=0.5, split_seed=21, weights=weights, mask=keep
    )
    receipt = split.receipt
    replay = sd.discover_on_split(
        contract(), data, discovery_fraction=0.5, split_seed=21, weights=weights, mask=keep
    )
    assert replay.receipt == receipt and replay.receipt.identity == receipt.identity
    assert np.array_equal(replay.analysis_weights, split.analysis_weights)
    assert np.array_equal(replay.discovery_row_ids, split.discovery_row_ids)
    assert replay.outcome.directed_edges == split.outcome.directed_edges
    # The recorded rule re-derives the partition from the source alone.
    split.verify_source(data, weights=weights, mask=keep)
    explicit = sd.discover_on_split(
        contract(),
        data,
        discovery_rows=split.discovery_row_ids.tolist(),
        weights=weights,
        mask=keep,
    )
    assert np.array_equal(explicit.analysis_row_ids, split.analysis_row_ids)
    assert explicit.receipt.analysis_rows_identity == receipt.analysis_rows_identity
    assert explicit.receipt.weights == receipt.weights
    # A source that changed in one cell, or in its weights, is not the replayed table.
    changed = {**data, "v3": data["v3"].copy()}
    changed["v3"][int(split.analysis_row_ids[0])] += 1e-9
    with pytest.raises(sd.SelectiveDiscoveryRefusal) as refused:
        split.verify_source(changed, weights=weights, mask=keep)
    assert refused.value.detail == "selective_discovery.split_source_mismatch"
    nudged = weights.copy()
    nudged[int(split.analysis_row_ids[0])] *= 1.0 + 1e-9
    with pytest.raises(sd.SelectiveDiscoveryRefusal) as refused:
        split.verify_source(data, weights=nudged, mask=keep)
    assert refused.value.detail == "selective_discovery.split_weights_mismatch"
