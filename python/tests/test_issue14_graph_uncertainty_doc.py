"""Earning test for docs/graph_uncertainty.md (issue #14).

Asserts the practitioner interpretation page exists with required sections,
is linked from docs/index.md, and that the marketing vignette's concrete
unidentified-mass claim matches a live ExactDagPosterior analyze.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from _repo_text import REPO_ROOT, read_text

DOC = REPO_ROOT / "docs" / "graph_uncertainty.md"
INDEX = REPO_ROOT / "docs" / "index.md"

# Keywords / section anchors the issue acceptance requires.
_REQUIRED_SUBSTRINGS = (
    "Exact DAG posterior",
    "PAG",
    "unidentified_mass",
    "CausalReviewError",
    "AcceptedGraph",
    "marketing_channel_structural_uncertainty.ipynb",
    "MAP CPDAG",
    "renormalize",
    "conformance/pag__envelope_unidentified_mass.md",
    "conformance/bayesian__graph_effect_envelope.md",
    "paper",
    "dashboard",
)


def test_graph_uncertainty_doc_exists_with_required_sections():
    assert DOC.is_file(), f"missing {DOC}"
    text = read_text(DOC)
    missing = [s for s in _REQUIRED_SUBSTRINGS if s not in text]
    assert not missing, f"docs/graph_uncertainty.md missing required content: {missing}"
    # Walk at least one concrete vignette number (50% / 0.5 unidentified mass).
    assert re.search(r"unidentified_mass\s*==\s*0\.5|0\.5\s*\(50%\)|50%", text)


def test_index_links_graph_uncertainty_doc():
    index = read_text(INDEX)
    assert "graph_uncertainty.md" in index
    assert re.search(r"\[([^\]]*graph uncertainty[^\]]*)\]\(graph_uncertainty\.md\)", index, re.I)


def test_vignette_unidentified_mass_matches_live_analyze():
    """Earn the doc's 50% claim against a compact marketing-style SCM."""
    pytest.importorskip("antecedent")
    import antecedent

    seed = 7
    n = 400
    rng = np.random.default_rng(seed)
    market_demand_index = rng.normal(size=n)
    paid_search_spend_kgbp = 20.0 + 4.0 * market_demand_index + 5.0 * rng.normal(size=n)
    qualified_pipeline_kgbp = (
        80.0 + 1.5 * paid_search_spend_kgbp + 12.0 * market_demand_index + 8.0 * rng.normal(size=n)
    )
    data = {
        "market_demand_index": market_demand_index,
        "paid_search_spend_kgbp": paid_search_spend_kgbp,
        "qualified_pipeline_kgbp": qualified_pipeline_kgbp,
    }

    result = antecedent.analyze(
        data=data,
        discovery=antecedent.discovery.ExactDagPosterior(),
        query=antecedent.AverageEffect(
            treatment="paid_search_spend_kgbp",
            outcome="qualified_pipeline_kgbp",
        ),
        inference=antecedent.Bayesian(n_draws=80, backend="conjugate"),
        refute=False,
        bootstrap=0,
        seed=seed,
    )
    assert result.posterior is not None
    mass = float(result.posterior.unidentified_mass)
    # Notebook pins 0.5 on the full vignette; compact replay should stay near it.
    assert mass == pytest.approx(0.5, abs=0.15), (
        f"expected ~0.5 unidentified mass for marketing-style MEC, got {mass}"
    )
    doc = read_text(DOC)
    assert "0.5" in doc and "50%" in doc
