"""The deliberate root namespace: ``antecedent.__all__`` is an explicit contract.

``__init__.py`` keeps the root namespace deliberately small.  The 0.5 release
explicitly reopened it for the causal-response queries while leaving their
configuration and result helpers on stage modules.  This test spells out the
resulting contract so future changes remain conscious.  Previously nothing
enforced that claim — ``test_notebook_api_surface.py`` only checked names the
example notebooks happened to use. That gap is exactly how
``antecedent.estimators`` went missing from the deliberate-but-unlisted
import block while every sibling stage module resolved fine (see the report
for this change): nothing asserted the unlisted-but-reachable set either.

This test hardcodes both sets in full so a future change to either one must
consciously edit this file rather than silently drift.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("antecedent")
import antecedent

# --- 1. The root `__all__` contract, spelled out in full. -------------------------

_EXPECTED_ALL = {
    # Verbs
    "analyze",
    "identify",
    "estimate",
    # Structure and results
    "AcceptedGraph",
    "Identification",
    "AnalysisResult",
    # Queries
    "AverageDerivative",
    "AverageEffect",
    "ConditionalEffect",
    "Counterfactual",
    "DirectionalDerivative",
    "Elasticity",
    "InterventionalDistribution",
    "InterventionResponse",
    "MediationEffect",
    "PathSpecificEffect",
    "PulseEffect",
    "PointDerivative",
    "ResponseCurve",
    "ResponseJacobian",
    "SemiElasticity",
    "SustainedEffect",
    "TemporalMediationEffect",
    # Graphs (five graph classes)
    "Dag",
    "Cpdag",
    "Pag",
    "Admg",
    "TemporalDag",
    # Selectors
    "Frequentist",
    "Bayesian",
    "Identifier",
    "Estimator",
    "Latency",
    "Refute",
    # Errors
    "CausalError",
    "ReviewRequired",
    # Stage modules (twelve)
    "attribution",
    "data",
    "design",
    "discovery",
    "errors",
    "estimation",
    "extensibility",
    "gcm",
    "graph",
    "priors",
    "state",
    "validation",
    # Version
    "__version__",
}

# --- 2. Reachable as `antecedent.<name>` but deliberately outside `__all__`. ------
#
# Their public content is re-exported on the root surface above (queries,
# inference selectors) or belongs to a narrower stage surface. `estimators`
# (the typed `estimator_config=` front-end) is included here as of this fix —
# see the module docstring above.

_EXPECTED_UNLISTED_BUT_REACHABLE = {
    "counterfactual",
    "estimators",
    "inference",
    "interference",
    "model",
    "observation",
    "population",
    "query",
    "transport",
}


def test_all_matches_the_documented_root_set():
    assert set(antecedent.__all__) == _EXPECTED_ALL


def test_all_has_no_duplicates():
    assert len(antecedent.__all__) == len(set(antecedent.__all__))


@pytest.mark.parametrize("name", sorted(_EXPECTED_ALL))
def test_every_root_name_resolves(name):
    assert hasattr(antecedent, name), f"antecedent.{name} is in __all__ but does not resolve"


@pytest.mark.parametrize("name", sorted(_EXPECTED_UNLISTED_BUT_REACHABLE))
def test_every_deliberately_unlisted_name_still_resolves(name):
    """These stay off `__all__` on purpose, but must still be reachable.

    ``antecedent.estimators`` unreachable (defect this test file guards
    against) would fail silently here before this fix: ``import antecedent;
    antecedent.estimators`` raised ``AttributeError`` even though
    ``from antecedent.estimators import LinearAdjustment`` worked fine.
    """
    assert hasattr(antecedent, name), f"antecedent.{name} should be reachable but does not resolve"
    assert name not in antecedent.__all__, f"antecedent.{name} should not be in __all__"


def test_estimators_module_is_reachable_and_not_root_exported():
    """Defect 1, directly: `antecedent.estimators` resolves without a direct import."""
    assert hasattr(antecedent, "estimators")
    assert antecedent.estimators.LinearAdjustment is not None
    assert "estimators" not in antecedent.__all__


# --- 3. Retired-name migration signpost (`__getattr__` in `__init__.py`). ---------


def test_retired_module_rename_has_a_signpost_message():
    with pytest.raises(AttributeError, match="renamed to antecedent.priors"):
        _ = antecedent.prior_bank


def test_retired_discover_functions_have_a_signpost_message():
    with pytest.raises(AttributeError, match="discovery config dataclasses"):
        _ = antecedent.discover_pc


def test_unknown_name_still_raises_plain_attribute_error():
    with pytest.raises(AttributeError):
        _ = antecedent.this_name_was_never_a_thing


# --- 4. Defect 6 smoke test: GCM discovered-attribution helpers reuse the fit. ----
#
# `attribute_paths_discovered` / `anomaly_attribution_discovered` /
# `attribute_distribution_change_discovered` in `antecedent/gcm.py` used to
# call `fit_gcm_discovered` and discard the fitted model, then re-derive edges
# and call a native `attribute_*` free function that re-fits internally —
# silent double work, and none of the three had a caller or test. This is a
# smoke test for one of them (per the fix, they now call the fitted model's
# own method instead of the free function).


def test_attribute_paths_discovered_runs_end_to_end():
    n = 400
    rng = np.random.default_rng(7)
    # Non-Gaussian noise so LiNGAM can orient (matches test_gcm_discovered.py's pattern).
    z = rng.uniform(-1.0, 1.0, size=n)
    t = 0.8 * z + rng.uniform(-1.0, 1.0, size=n)
    y = 1.5 * t + 0.6 * z + rng.uniform(-1.0, 1.0, size=n)
    data = {"z": z, "t": t, "y": y}
    # A single source: `attribute_paths` rejects sources where one is a
    # directed ancestor of another (z -> t here), so ["z", "t"] would raise
    # CausalAttributionError — a pre-existing native constraint, not part of
    # what this smoke test is checking.
    result, edges = antecedent.gcm.attribute_paths_discovered(
        data,
        discovery=antecedent.discovery.LiNGAM(),
        sources=["z"],
        outcome="y",
        seed=1,
    )
    assert edges
    assert result is not None
    assert hasattr(result, "total_change")
