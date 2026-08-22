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

import ast
from pathlib import Path

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
# see the module docstring above. `artifacts` and `intervention` were both
# missing from this set until this fix: `intervention` is load-bearing for
# `InterventionResponse` (`_analyze.py`) and documented in
# `docs/causal-responses.md`; `artifacts` backs the documented
# `antecedent.artifacts.dumps`/`.loads` surface (`docs/artifacts.md`). Neither
# omission was caught because nothing checked this set for completeness --
# see ``test_unlisted_but_reachable_set_matches_init_py`` below, which now
# derives the deliberate-import block from ``__init__.py`` itself so a future
# stage module cannot be silently added there without this set changing too.

_EXPECTED_UNLISTED_BUT_REACHABLE = {
    "accepted_graph",
    "artifacts",
    "counterfactual",
    "estimators",
    "ids",
    "inference",
    "interference",
    "intervention",
    "model",
    "observation",
    "population",
    "query",
    "results",
    "transport",
}


# The twelve root-exported stage modules are public surfaces too.  Freezing only
# the package root would still let a refactor silently add or remove names from
# ``antecedent.discovery`` (or any sibling) while the advertised 0.9 API freeze
# continued to pass.  Keep these lists literal: changing one is an API decision.
_EXPECTED_STAGE_ALL = {
    "attribution": {
        "AnomalyScores",
        "ChangeAttributionResult",
        "Contribution",
        "FeatureRelevance",
        "MechanismChangeDetection",
        "anomaly_attribution",
        "attribute_distribution_change",
        "attribute_distribution_change_robust",
        "attribute_feature_relevance",
        "attribute_path_specific",
        "attribute_paths",
        "attribute_structure_change",
        "attribute_unit_change",
        "mechanism_change_detection",
        "rank_root_causes",
    },
    "data": {
        "ArrowLoadInfo",
        "EventFrame",
        "MultiEnvFrame",
        "PanelFrame",
        "event",
        "load_float64_arrow_c_columns",
        "load_float64_columns",
        "multi_env",
        "panel",
        "to_f64",
    },
    "design": {"DecisionEvaluation", "DesignRanking", "evaluate_decision", "rank_designs"},
    "discovery": {
        "CiScreenedPosterior",
        "DbnPosterior",
        "DiscoveredLink",
        "DiscoveryResult",
        "ExactDagPosterior",
        "FCI",
        "GES",
        "GraphEdge",
        "GraphPosterior",
        "JPCMCIPlus",
        "LPCMCI",
        "LiNGAM",
        "NOTEARS",
        "OrderMcmc",
        "PC",
        "PCMCI",
        "PCMCIPlus",
        "PcmciDiscoveryResult",
        "RFCI",
        "RPCMCI",
        "RpcmciDiscoverySummary",
        "StructureMcmc",
        "cpdag_oriented_edges",
        "discovery_algorithm",
        "discovery_to_dag",
        "graph_posterior_map_dag",
        "graph_posterior_map_edges",
        "run_static_discovery",
        "run_temporal_discovery",
        "two_regime_half_split",
    },
    "errors": {
        "CausalAttributionError",
        "CausalCancelledError",
        "CausalCompileError",
        "CausalCounterfactualError",
        "CausalDataError",
        "CausalDesignError",
        "CausalDiscoveryError",
        "CausalEstimateError",
        "CausalError",
        "CausalGraphError",
        "CausalIdentifyError",
        "CausalModelError",
        "CausalResourceError",
        "CausalReviewError",
        "CausalSerializationError",
        "CausalStateError",
        "CausalTypeError",
        "CausalUnsupportedError",
        "CausalValidateError",
        "CausalValueError",
        "PendingEdge",
        "ReviewRequired",
        "build_review_error",
        "pending_edges",
    },
    "estimation": {
        "AnalysisResult",
        "ConflictSummaryView",
        "EffectEnvelope",
        "EstimateView",
        "IdentificationView",
        "IdentifyResult",
        "MediationEffectsSummary",
        "MediationView",
        "PerformanceView",
        "PhysicalPlanView",
        "PlanView",
        "PosteriorView",
        "PredictiveCheckReport",
        "PreparedAnalysis",
        "PriorSensitivityReport",
        "RefutationReport",
        "ValidationView",
        "analyze_many",
        "identify",
        "mediation_effects_summary",
    },
    "extensibility": {"CiBatchTest", "EffectValidator", "MechanismWrapper", "UtilityFn"},
    "gcm": {
        "anomaly_attribution_discovered",
        "attribute_distribution_change_discovered",
        "attribute_paths_discovered",
        "fit_gcm_discovered",
    },
    "graph": {
        "Admg",
        "Cpdag",
        "Dag",
        "Pag",
        "TemporalCpdag",
        "TemporalDag",
        "TemporalPag",
        "cpdag_oriented_edges",
        "discovery_to_dag",
    },
    "priors": {
        "BetaHyperparameters",
        "CompatibilityReport",
        "ComposedPrior",
        "ConflictPolicy",
        "DesignVariable",
        "EstimandFingerprint",
        "ExternalPriorSourceSpec",
        "ExternalPriorWeight",
        "GammaHyperparameters",
        "POPULATION_TAG_KEY",
        "PriorCatalog",
        "PriorMapping",
        "PriorSource",
        "PriorSourceMeta",
        "TransportPolicy",
        "beta_from_mean_and_ess",
        "beta_from_moments",
        "compose_external_priors",
        "gamma_from_mean_and_ess",
        "gamma_from_moments",
        "populations_from_prior_sources",
    },
    "state": {"CancellationToken", "CausalState", "antecedent_state_append"},
    "validation": {
        "validate_environment_holdout",
        "validate_pcmci_alpha_sensitivity",
        "validate_pcmci_block_bootstrap",
        "validate_pcmci_ci_sensitivity",
        "validate_pcmci_false_positive",
        "validate_pcmci_lag_sensitivity",
        "validate_pcmci_plus_orientation",
        "validate_regime_stability",
        "validate_synthetic_null_calibration",
    },
}


def test_all_matches_the_documented_root_set():
    assert set(antecedent.__all__) == _EXPECTED_ALL


def test_all_has_no_duplicates():
    assert len(antecedent.__all__) == len(set(antecedent.__all__))


def test_api_naming_counts_match_the_frozen_surfaces():
    text = (Path(__file__).resolve().parents[2] / "docs" / "api_naming.md").read_text()
    assert f"frozen at {len(antecedent.__all__)} names" in text
    assert f"**{len(_EXPECTED_UNLISTED_BUT_REACHABLE)}** further modules" in text


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


def _deliberate_unlisted_reachable_imports_from_init_py() -> set[str]:
    """Parse ``__init__.py`` for its ``from . import X as X`` block.

    That self-aliased spelling (``as X`` repeating the imported name) is what
    marks a stage-module import as the deliberate "reachable but outside
    ``__all__``" family in this file -- as opposed to plain ``from . import
    (a, b, c)`` (the root-exported stage modules, no alias) or ``from
    ._native import name as name`` (re-exported constants/classes, not this
    package's own submodules). Deriving the set this way means a future
    ``from . import newmod as newmod`` line added to that block, without a
    matching update here, fails this test instead of silently drifting.
    """

    source = Path(antecedent.__file__).read_text()
    tree = ast.parse(source, filename=antecedent.__file__)
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module is not None or node.level != 1:
            continue  # not a plain `from . import ...`
        for alias in node.names:
            if alias.asname == alias.name:
                names.add(alias.name)
    return names


def test_unlisted_but_reachable_set_matches_init_py():
    """Defects 3 & 4, guarded structurally: this set must track `__init__.py`.

    Before this fix, `_EXPECTED_UNLISTED_BUT_REACHABLE` was a hand-maintained
    list that had already drifted from `__init__.py` twice (missing both
    `intervention` and `artifacts`). This test makes that drift impossible to
    reintroduce silently: it derives the actual deliberate-import block from
    the source rather than trusting a second hand-copied list.
    """
    assert _deliberate_unlisted_reachable_imports_from_init_py() == _EXPECTED_UNLISTED_BUT_REACHABLE


@pytest.mark.parametrize("module_name", sorted(_EXPECTED_STAGE_ALL))
def test_stage_module_all_is_frozen(module_name):
    module = getattr(antecedent, module_name)
    actual = tuple(module.__all__)
    assert len(actual) == len(set(actual)), f"antecedent.{module_name}.__all__ has duplicates"
    assert set(actual) == _EXPECTED_STAGE_ALL[module_name]


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
