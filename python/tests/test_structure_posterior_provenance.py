"""Accepting the structure posterior a compiled search returned, with its provenance."""

from __future__ import annotations

import numpy as np
import pytest
from antecedent import _native
from antecedent.errors import CausalValueError


def _rows(seed: int = 4, n: int = 200) -> tuple[list[str], list[np.ndarray]]:
    rng = np.random.default_rng(seed)
    z = rng.uniform(-1.0, 1.0, size=n)
    t = 0.8 * z + rng.uniform(-1.0, 1.0, size=n)
    y = 1.5 * t + rng.uniform(-1.0, 1.0, size=n)
    return ["z", "t", "y"], [z, t, y]


def test_accepting_a_searched_posterior_records_where_it_came_from():
    names, columns = _rows()
    posterior = _native.discover_exact_dag_posterior(names, columns, seed=1)
    accepted = posterior.accept_posterior()
    assert accepted["inference"] == "bayesian"
    assert accepted["review"] == "posterior"
    assert accepted["resolved_marks"] == "0"
    assert accepted["claim"] == posterior.claim["claim_id"]
    assert accepted["program"] == posterior.route.program_identity
    assert accepted["atoms"] == str(posterior.n_graphs)
    assert len(accepted["digest"]) == 64
    assert posterior.accept_posterior() == accepted  # accepting again records the same thing


def test_another_search_is_another_provenance():
    names, columns = _rows()
    first = _native.discover_exact_dag_posterior(names, columns, seed=1).accept_posterior()
    other_names, other_columns = _rows(seed=9)
    second = _native.discover_exact_dag_posterior(other_names, other_columns, seed=1)
    assert second.accept_posterior()["data_snapshot"] != first["data_snapshot"]


def test_a_posterior_supplied_as_atoms_has_no_search_to_accept():
    supplied = _native.GraphPosterior.from_atoms(["a", "b"], [1.0], [0])
    with pytest.raises(CausalValueError):
        supplied.accept_posterior()
