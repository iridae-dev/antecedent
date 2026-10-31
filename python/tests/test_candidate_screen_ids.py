"""Candidate-screen identifiers are exact integers; a fractional id is refused, not truncated."""

import numpy as np
import pytest
from antecedent._batch import CandidateScreen
from antecedent.errors import CausalValueError


def test_fractional_entity_id_refused():
    entities = [1, 1.9, 2, 2, 3, 3, 4, 4]
    with pytest.raises(CausalValueError, match=r"entity_ids\[1\]"):
        CandidateScreen.from_units("s", "max_t", entity_ids=entities, seed=5)


def test_fractional_dyad_endpoint_refused():
    with pytest.raises(CausalValueError, match=r"first\[0\]"):
        CandidateScreen.from_units("s", "max_t", first=[0.5, 1], second=[2, 3], seed=5)


def test_integral_ids_still_split():
    screen = CandidateScreen.from_units("s", "max_t", entity_ids=[1, 1, 2, 2, 3, 3, 4, 4], seed=5)
    assert len(screen.screen_rows) + len(screen.estimate_rows) == 8


def test_batch_edges_and_tiers_are_not_both_accepted():
    from antecedent import _native

    columns = [np.array([float(i % 2) for i in range(12)]), np.arange(12.0)]
    with pytest.raises(ValueError, match="edges or tiers"):
        _native.prepare_ate_batch(
            ["t", "y"],
            columns,
            [("t", "y")],
            [("t", "y", 0.0, 1.0, None, None)],
            tiers=[["t"], ["y"]],
            within_tier="unconstrained",
        )
