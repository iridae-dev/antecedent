"""``identify()`` over an ADMG.

A ``Dag`` cannot express "this variable is unobservable". Flattening a latent
common cause into one makes it look like an ordinary adjustable node, so the
effect is reported identified by adjusting on a variable no study can measure.
An ADMG carries that structure as a bidirected edge, so identification can
honestly fail.
"""

from __future__ import annotations

import antecedent
import pytest
from antecedent.errors import CausalError

QUERY = antecedent.AverageEffect(treatment="T", outcome="Y")


def _confounded_dag() -> antecedent.Dag:
    """``T <- U -> Y`` plus ``T -> Y``, with ``U`` the unmeasured confounder."""
    return antecedent.Dag.from_edges(["U", "T", "Y"], [("U", "T"), ("U", "Y"), ("T", "Y")])


def _frontdoor_dag() -> antecedent.Dag:
    """``U -> T``, ``U -> Y``, ``T -> M -> Y``: front-door identifiable via ``M``."""
    return antecedent.Dag.from_edges(
        ["U", "T", "M", "Y"], [("U", "T"), ("U", "Y"), ("T", "M"), ("M", "Y")]
    )


def _observed_dag() -> antecedent.Dag:
    """``Z -> T``, ``Z -> Y``, ``T -> Y``: plain backdoor, nothing latent."""
    return antecedent.Dag.from_edges(["Z", "T", "Y"], [("Z", "T"), ("Z", "Y"), ("T", "Y")])


def test_latent_confounding_is_not_identifiable_as_an_admg() -> None:
    admg = _confounded_dag().latent_project(["T", "Y"])
    with pytest.raises(CausalError):
        antecedent.identify(graph=admg, query=QUERY)


def test_the_same_graph_as_a_dag_adjusts_on_the_unmeasurable_node() -> None:
    """Documents why the ADMG path exists, and why callers should prefer it."""
    result = antecedent.identify(graph=_confounded_dag(), query=QUERY)
    assert bool(result)
    assert result.adjustment_set == ["U"]


def test_frontdoor_is_identified_through_an_admg() -> None:
    admg = _frontdoor_dag().latent_project(["T", "M", "Y"])
    result = antecedent.identify(graph=admg, query=QUERY)
    assert bool(result)
    assert result.method == "general.id"


def test_an_admg_without_bidirected_edges_is_treated_as_a_dag() -> None:
    """No latent structure means nothing for general ID to reason about.

    Coercing keeps the caller's identifier choice meaningful and keeps the
    adjustment set identical to the equivalent DAG.
    """
    dag_result = antecedent.identify(graph=_observed_dag(), query=QUERY)
    admg = _observed_dag().latent_project(["Z", "T", "Y"])
    admg_result = antecedent.identify(graph=admg, query=QUERY)
    assert admg_result.adjustment_set == dag_result.adjustment_set == ["Z"]
    assert admg_result.method == dag_result.method


def test_an_explicit_identifier_survives_the_dag_coercion() -> None:
    admg = _observed_dag().latent_project(["Z", "T", "Y"])
    result = antecedent.identify(graph=admg, query=QUERY, identifier="backdoor.adjustment")
    assert result.method == "backdoor.adjustment"
    assert result.adjustment_set == ["Z"]


def test_staged_identification_carries_the_admg_forward() -> None:
    admg = _frontdoor_dag().latent_project(["T", "M", "Y"])
    result = antecedent.identify(graph=admg, query=QUERY)
    assert result.graph is admg
