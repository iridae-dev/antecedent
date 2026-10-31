"""Graph constructors refuse duplicate or empty node names."""

from __future__ import annotations

import antecedent as ant
import pytest
from antecedent.graph import TemporalAdmg, TemporalCpdag

_NAMES = ["x", "y", "x"]


@pytest.mark.parametrize(
    "build",
    [
        lambda names: ant.Dag.from_edges(names, [("x", "y")]),
        lambda names: ant.Cpdag.from_edges(names, [("x", "y", "directed")]),
        lambda names: ant.Admg.from_edges(names, [("x", "y")]),
        lambda names: ant.Cpdag.from_directed_undirected(names, [("x", "y")]),
        lambda names: ant.Pag.from_marked_edges(names, [("x", "y", "tail", "arrow")]),
        lambda names: ant.TemporalDag.from_lagged_edges(names, [("x", 1, "y", 0)]),
        lambda names: TemporalAdmg.from_lagged_edges(names, [("x", 1, "y", 0)]),
        lambda names: TemporalCpdag.from_lagged_edges(names, [("x", 1, "y", 0)]),
    ],
)
@pytest.mark.parametrize("names", [_NAMES, ["x", ""]])
def test_duplicate_names_refused(build, names):
    with pytest.raises(ValueError):
        build(names)
