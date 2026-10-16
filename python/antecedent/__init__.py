"""antecedent — Python bindings for the Antecedent causal engine.

Day-1 surface::

    result = antecedent.analyze(
        data,
        graph=edges,
        query=antecedent.AverageEffect(treatment="t", outcome="y"),
    )

The root namespace is deliberately small: it holds the analysis verbs (:func:`analyze`, :func:`prepare`,
:func:`identify`, :func:`estimate`) and :func:`load`, the accepted-structure and result types,
the first-class typed queries, the five graph classes, the inference / identifier /
estimator selectors, and the two error names most callers catch. The specialized
families — randomized and factorial experiments, held-out policy value,
difference-in-differences and other quasi-experimental designs, survival, and
longitudinal regimes — live on their stage modules (``antecedent.experiment.RandomizedEffect``,
``antecedent.quasi.SyntheticControl``, ``antecedent.policy.PolicyValue``,
``antecedent.survival.SurvivalOutcome``, ``antecedent.regimes.LongitudinalRegime``),
not at the root.

Root-exported stage modules (listed in ``__all__``):

``antecedent.attribution``, ``antecedent.data``, ``antecedent.design``,
``antecedent.discovery``, ``antecedent.errors``, ``antecedent.experiment``,
``antecedent.estimation``, ``antecedent.extensibility``, ``antecedent.factorial``,
``antecedent.gcm``, ``antecedent.graph``, ``antecedent.policy``,
``antecedent.priors``, ``antecedent.quasi``, ``antecedent.regimes``,
``antecedent.state``, ``antecedent.survival``, and ``antecedent.validation``.

Narrower modules imported eagerly, reachable but deliberately outside ``__all__``:
``accepted_graph``, ``artifacts``, ``counterfactual``, ``decision``, ``derived``, ``estimators``,
``external``, ``handoff``, ``ids``, ``inference``, ``interference``, ``intervention``, ``learners``,
``matched``, ``model``, ``observation``, ``population``, ``prediction``, ``query``,
``results``, and ``transport``.

The stage modules listed in ``_LAZY_MODULES`` are loaded on first attribute access (``antecedent.recalc_cell`` imports
the module then, so ``import antecedent`` pays nothing for them) and are listed by
``dir(antecedent)``; none is in ``__all__``. The full tuple is ``_LAZY_MODULES``. They are
grouped here by workflow stage.

Lifecycle -- analyze, bind external evidence, inspect the claim, decide, rank a study, bundle:

* ``program_claims`` binds a claim to the identified program and the request it answers, and
  ``external`` (eager) binds an externally supplied response to it;
* ``decision`` (eager) states the :class:`~antecedent.decision.Contract` and evaluates it on a
  result, a bound claim or a distribution; ``decision_robust`` adds admissibility rules and
  robustness under structural uncertainty, ``scenario_decision`` decides over a scenario or CPDAG
  completion set, ``sensitivity_decision`` over an assumption-sensitivity result, and ``inverse`` /
  ``inverse_query`` answer which enumerated interventions reach a target;
* ``design`` (root-exported) ranks the study to run next, ``repair`` states evidence obligations
  and identification repair, and ``proposals`` links the two with a per-proposal receipt;
* ``joint_distribution`` holds the aligned distribution artifacts decisions consume, and
  ``composition`` / ``composition_bundle`` define what may be combined and export the whole chain
  as one portable bundle an independent consumer checks against retained identities.

Transport and counterfactual:

* ``transport`` (eager) answers the ordinary transport question; ``scenario_invariance`` reports the
  selection differences and invariances behind each scenario answer, and ``mechanism_discrepancy``
  diagnoses whether a source mechanism agrees with the target;
* ``transported_counterfactual`` transports a static path-specific counterfactual,
  ``temporal_counterfactual`` is its fixed-population temporal counterpart, ``cross_world`` is the
  path-specific edge intervention on a Markovian DAG and ``counterfactual_id`` the effect of
  treatment on the treated on a bounded ADMG;
* ``transported_response`` carries a response value to a target population with its transport
  receipt and decides on it, ``transported_constructors`` builds those responses from checked
  inputs, ``transport_handoff`` seals the connection of a transported claim to a callback input,
  ``discretized_composition`` turns a linear-Gaussian composition into an executable finite model
  with its distance from the closed form, and ``inverse_sources`` adapts each forward source of an
  inverse query with its own refusals. The revalidation, sensitivity and discovered transport
  routes are submodules of ``transport``.

Sensitivity and robustness:

* ``msm_sensitivity`` gives marginal-sensitivity-model bounds, the tipping point and a durable
  sensitivity artifact; ``descriptive`` compares raw with adjusted contrasts on a declared
  reporting scale; ``dose_grid`` reads a named functional off a randomized-dose response curve;
  ``temporal`` tests effect constancy across a time or region partition.

Recalculation (selective, with a visible receipt):

* ``recalc`` provides receipts and the shared surface. Its adapters cover static, adjusted,
  cell-AIPW (with portable score resume), DML / DR-Learner, Bayesian, temporal, external,
  design and composite workflows. ``recalc_capabilities`` inventories what each can recompute.

Specialized effect families: ``categorical_treatment``, ``vector_treatment``,
``nonlinear_mediation``, ``latent_class``, ``recovery_chain`` and ``compact_export`` (a compact
runtime export of a fitted effect model); ``preflight`` holds the pre-fit diagnostics, rank-drop
plans and cost counts.

Measurement, composition and evidence accounting: ``measurement_models`` corrects a functional of
a latent construct measured with error under a declared model (point and uncalibrated standard
error, never an interval); ``mechanism_composition`` composes independently fitted mechanisms into
one executable joint model with declared dependence; ``evidence_ledger`` accounts the overlap of
evidence sources before any pooling, borrowing or reuse and reclassifies stale results.
``interface_contract`` declares what a component produces and consumes and decides, before any
number is computed, whether two components may be connected (cycles are refused with their path).
``scientific_bridge`` states the assumptions under which a claim about one quantity answers a
question about another (surrogate to terminal outcome, measured to latent, predictive to causal)
as a new, separately identified, point-only claim. ``composition_verification`` classifies every
downstream claim of a composed analysis as still valid, recompute or withdrawn when an upstream
object changes, and reports the lineage and calibration standing behind each claim.
``provider_callbacks`` declares an arbitrary law, posterior or nested callback you supply, negotiates
mean, draws, distribution function, quantile and posterior update separately, reuses a stored answer
only under its declared replay contract and accepts an answer only as a well-formed law; a callback
supplies no interval, native trust, joint law or identification licence.

``observation_query`` asks what an observation says: the outcome probability given a recorded
signal kept distinct from the probability under an intervention on the same model, a measured
signal inverted through its measurement model, posteriors after new evidence and the evidence
required to reach a level (exact given a declared model, never an interval).

``predictive_assessment`` scores predictive laws against realized observations under a declared
validation design: proper scores, paired differences with declared dependence, labelled
diagnostics and a held-out check from row identities (point-only; a good score is not causal
validity and no interval is reported).

Graph interchange is on the classes: ``Dag.from_dot`` / ``Dag.to_dot`` and the
JSON / GML / NetworkX peers, likewise on ``Cpdag`` / ``Pag`` / ``Admg``.

Public analysis results share :data:`Analysis` (``AnalysisResult`` or
``CausalResponseView``). Consume ``result.answer`` / ``result.claim()``.
The native DTOs live on ``antecedent._native`` only.
"""

from __future__ import annotations

from importlib import import_module as _import_module
from types import ModuleType as _ModuleType
from typing import TYPE_CHECKING as _TYPE_CHECKING
from typing import NoReturn as _NoReturn

# A debug-profile extension returns bit-identical estimates while running
# ~50x slower, so nothing downstream would ever notice on its own. The flag
# is absent unless compiled in; only an explicit `False` proves the
# module was compiled without optimizations.
from . import _native as _native_module

# These explicit self-alias imports belong to the "reachable but deliberately
# outside `__all__`" family described below. Isort's alphabetical ordering
# places the first pair ahead of the `__all__`-exported module block rather
# than next to their siblings.
from . import accepted_graph as accepted_graph
from . import artifacts as artifacts
from . import (
    attribution,
    data,
    design,
    discovery,
    errors,
    estimation,
    experiment,
    extensibility,
    factorial,
    gcm,
    graph,
    policy,
    priors,
    quasi,
    regimes,
    state,
    survival,
    validation,
)

# Reachable as ``antecedent.<name>`` but deliberately outside the root
# ``__all__``: their public content is re-exported above (queries, inference
# selectors) or belongs to a narrower stage surface. ``estimators`` (the
# typed ``estimator_config=`` front-end) belongs here too: it is a real,
# documented stage module, but its content (per-estimator dataclasses) has no
# root-level re-export the way queries/selectors do, so it stays off
# ``__all__`` alongside its siblings rather than being added to it.
# (``artifacts`` is also part of this family -- see the comment above.)
from . import counterfactual as counterfactual
from . import decision as decision
from . import derived as derived
from . import estimators as estimators
from . import external as external
from . import handoff as handoff
from . import ids as ids
from . import inference as inference
from . import interference as interference
from . import intervention as intervention
from . import learners as learners
from . import matched as matched
from . import model as model
from . import observation as observation
from . import population as population
from . import prediction as prediction
from . import query as query
from . import results as results
from . import transport as transport
from ._analyze import analyze
from ._native import (
    Admg,
    Cpdag,
    Dag,
    Pag,
    TemporalDag,
)

if getattr(_native_module, "__build_optimized__", True) is False:
    import warnings as _warnings

    _warnings.warn(
        "antecedent._native was compiled in Cargo's debug profile; estimation "
        "runs ~50x slower than a release build (results are unaffected). "
        "Reinstall the package (e.g. `uv sync --reinstall-package antecedent` "
        "or `maturin develop --release`) to rebuild it optimized.",
        RuntimeWarning,
        stacklevel=2,
    )
from ._workflow import load, prepare
from .accepted_graph import AcceptedGraph
from .errors import CausalError, ReviewRequired
from .identify import Identification, estimate, identify
from .ids import Estimator, Identifier, Latency, Refute
from .inference import Bayesian, ClassPrior, Frequentist
from .interference import InterferenceQuery
from .query import (
    AnomalyAttribution,
    AnomalyReference,
    AverageDerivative,
    AverageEffect,
    ChangeAttribution,
    ConditionalEffect,
    Counterfactual,
    DirectionalDerivative,
    Elasticity,
    InterventionalDistribution,
    InterventionResponse,
    MediationEffect,
    NestedCounterfactual,
    PathSpecificEffect,
    PointDerivative,
    PulseEffect,
    ResponseCurve,
    ResponseJacobian,
    SemiElasticity,
    SustainedEffect,
    TemporalConditionalPulse,
    TemporalConditionalSustained,
    TemporalMediationEffect,
)
from .results import Analysis, AnalysisResult

__all__ = [
    # Verbs
    "analyze",
    "prepare",
    "load",
    "identify",
    "estimate",
    # Structure and results
    "AcceptedGraph",
    "Identification",
    "Analysis",
    "AnalysisResult",
    # Queries
    "AnomalyAttribution",
    "AnomalyReference",
    "AverageDerivative",
    "AverageEffect",
    "ChangeAttribution",
    "ConditionalEffect",
    "Counterfactual",
    "DirectionalDerivative",
    "Elasticity",
    "InterferenceQuery",
    "InterventionalDistribution",
    "InterventionResponse",
    "MediationEffect",
    "NestedCounterfactual",
    "PathSpecificEffect",
    "PulseEffect",
    "PointDerivative",
    "ResponseCurve",
    "ResponseJacobian",
    "SemiElasticity",
    "SustainedEffect",
    "TemporalConditionalPulse",
    "TemporalConditionalSustained",
    "TemporalMediationEffect",
    # Graphs
    "Dag",
    "Cpdag",
    "Pag",
    "Admg",
    "TemporalDag",
    # Selectors
    "Frequentist",
    "Bayesian",
    "ClassPrior",
    "Identifier",
    "Estimator",
    "Latency",
    "Refute",
    # Errors
    "CausalError",
    "ReviewRequired",
    # Stage modules
    "attribution",
    "data",
    "design",
    "discovery",
    "errors",
    "experiment",
    "estimation",
    "extensibility",
    "factorial",
    "gcm",
    "graph",
    "policy",
    "quasi",
    "priors",
    "regimes",
    "state",
    "survival",
    "validation",
    "__version__",
]

try:
    from ._native import __version__ as __version__
except ImportError:  # pragma: no cover - extension not built
    # Derive from installed package metadata rather than a hand-maintained literal;
    # a clearly-unknown sentinel when even that is absent (a source checkout with
    # neither the extension built nor the package installed).
    from importlib import metadata as _metadata

    try:
        __version__ = _metadata.version("antecedent")
    except _metadata.PackageNotFoundError:
        __version__ = "unknown"


# --- Migration signpost for retired 0.4.0 names ------------------------------------
#
# ``CHANGELOG.md``'s "[0.4.0]" entry documents the migration policy as a
# "silent hard break: there are no deprecated aliases and no shims for the
# old spellings." This hook does not violate that — every branch below ends
# in a raise, never a returned value, so a retired name is exactly as broken
# as it would be without this function. It only replaces Python's generic
# "module 'antecedent' has no attribute ..." with a message naming the 0.4.0
# replacement, for the name families the changelog documents as large,
# mechanical, and easy to hit from muscle memory / an old example:
#
# - the 16 free ``discover_*`` functions -> methods on ``antecedent.discovery``
#   config dataclasses (``.run()`` / ``.accept()``);
# - the 10 free ``dag_from_*`` / ``dag_to_*`` helpers -> ``Dag``/``Cpdag``/
#   ``Pag``/``Admg`` class methods (``.from_dot`` / ``.to_dot()`` / ...);
# - the ``target_*`` population builders -> ``antecedent.population``;
# - the ``prior_bank`` module -> ``antecedent.priors``.
#
# ``antecedent.CausalAnalysis`` is deliberately NOT listed here: per the same
# changelog entry, ``CausalAnalysis`` was a Rust-only facade struct (renamed
# to ``Study`` in Rust) that was never itself a Python attribute — "Python is
# unaffected: the PyO3-exposed class name (``PreparedAnalysis``) was
# deliberately left unchanged by this rename." Adding a shim for a name that
# was never actually reachable from Python would misrepresent history rather
# than document it, so it is omitted (see this module's docstring companion
# report for the verification).
#
# --- Lazy stage modules (PEP 562) ----------------------------------------------------
#
# The lazy stage modules are public (their classes and functions are documented in each
# module's docstring) but none is in ``__all__`` and importing them all would put the whole
# family on the cost of ``import antecedent``. ``__getattr__`` imports one on first access and
# caches it on the package, ``__dir__`` lists them so tab completion finds them, and neither
# adds a name to ``__all__``. ``identify`` is absent on purpose: the root name is the function.
_LAZY_MODULES: tuple[str, ...] = (
    "adaptive_inference",
    "categorical_treatment",
    "cohort_decision",
    "compact_export",
    "composition",
    "composition_bundle",
    "composition_verification",
    "counterfactual_id",
    "cross_world",
    "cross_world_execution",
    "decision_reliability",
    "decision_robust",
    "descriptive",
    "discretized_composition",
    "dose_grid",
    "effect_constancy_review",
    "evidence_ledger",
    "execution_attempt",
    "functional_source",
    "interface_contract",
    "inverse",
    "inverse_query",
    "inverse_sources",
    "joint_distribution",
    "joint_law_propagation",
    "latent_class",
    "measurement_models",
    "mechanism_composition",
    "mechanism_discrepancy",
    "missing",
    "msm_sensitivity",
    "multi_level_treatment",
    "nonlinear_mediation",
    "observation_query",
    "observational_policy",
    "observational_survival",
    "predictive_assessment",
    "preflight",
    "program_claims",
    "proposal_arrival",
    "proposals",
    "provider_callbacks",
    "recalc",
    "recalc_adjusted",
    "recalc_bayesian",
    "recalc_capabilities",
    "recalc_cell",
    "recalc_composite",
    "recalc_design",
    "recalc_dr",
    "recalc_external",
    "recalc_static",
    "recalc_temporal",
    "recovery_chain",
    "repair",
    "scenario_decision",
    "scenario_invariance",
    "scientific_bridge",
    "selective_discovery",
    "sensitivity_decision",
    "sequential_evaluation",
    "source_evidence",
    "source_projection",
    "temporal",
    "temporal_counterfactual",
    "tier_diagnostics",
    "transport_handoff",
    "transported_constructors",
    "transported_counterfactual",
    "transported_response",
    "vector_treatment",
)

if _TYPE_CHECKING:
    from . import adaptive_inference as adaptive_inference
    from . import categorical_treatment as categorical_treatment
    from . import cohort_decision as cohort_decision
    from . import compact_export as compact_export
    from . import composition as composition
    from . import composition_bundle as composition_bundle
    from . import composition_verification as composition_verification
    from . import counterfactual_id as counterfactual_id
    from . import cross_world as cross_world
    from . import cross_world_execution as cross_world_execution
    from . import decision_reliability as decision_reliability
    from . import decision_robust as decision_robust
    from . import descriptive as descriptive
    from . import discretized_composition as discretized_composition
    from . import dose_grid as dose_grid
    from . import effect_constancy_review as effect_constancy_review
    from . import evidence_ledger as evidence_ledger
    from . import execution_attempt as execution_attempt
    from . import functional_source as functional_source
    from . import interface_contract as interface_contract
    from . import inverse as inverse
    from . import inverse_query as inverse_query
    from . import inverse_sources as inverse_sources
    from . import joint_distribution as joint_distribution
    from . import joint_law_propagation as joint_law_propagation
    from . import latent_class as latent_class
    from . import measurement_models as measurement_models
    from . import mechanism_composition as mechanism_composition
    from . import mechanism_discrepancy as mechanism_discrepancy
    from . import missing as missing
    from . import msm_sensitivity as msm_sensitivity
    from . import multi_level_treatment as multi_level_treatment
    from . import nonlinear_mediation as nonlinear_mediation
    from . import observation_query as observation_query
    from . import observational_policy as observational_policy
    from . import observational_survival as observational_survival
    from . import predictive_assessment as predictive_assessment
    from . import preflight as preflight
    from . import program_claims as program_claims
    from . import proposal_arrival as proposal_arrival
    from . import proposals as proposals
    from . import provider_callbacks as provider_callbacks
    from . import recalc as recalc
    from . import recalc_adjusted as recalc_adjusted
    from . import recalc_bayesian as recalc_bayesian
    from . import recalc_capabilities as recalc_capabilities
    from . import recalc_cell as recalc_cell
    from . import recalc_composite as recalc_composite
    from . import recalc_design as recalc_design
    from . import recalc_dr as recalc_dr
    from . import recalc_external as recalc_external
    from . import recalc_static as recalc_static
    from . import recalc_temporal as recalc_temporal
    from . import recovery_chain as recovery_chain
    from . import repair as repair
    from . import scenario_decision as scenario_decision
    from . import scenario_invariance as scenario_invariance
    from . import scientific_bridge as scientific_bridge
    from . import selective_discovery as selective_discovery
    from . import sensitivity_decision as sensitivity_decision
    from . import sequential_evaluation as sequential_evaluation
    from . import source_evidence as source_evidence
    from . import source_projection as source_projection
    from . import temporal as temporal
    from . import temporal_counterfactual as temporal_counterfactual
    from . import tier_diagnostics as tier_diagnostics
    from . import transport_handoff as transport_handoff
    from . import transported_constructors as transported_constructors
    from . import transported_counterfactual as transported_counterfactual
    from . import transported_response as transported_response
    from . import vector_treatment as vector_treatment

_RETIRED_MODULES = {"prior_bank": "priors"}
_RETIRED_DISCOVER_PREFIX = "discover_"
_RETIRED_DAG_PREFIXES = ("dag_from_", "dag_to_")
_RETIRED_TARGET_PREFIX = "target_"


def __dir__() -> list[str]:
    # ``annotations`` is the ``from __future__`` binding every module carries, not API.
    return sorted((set(globals()) | set(_LAZY_MODULES)) - {"annotations"})


def __getattr__(name: str) -> _ModuleType:
    if name in _LAZY_MODULES:
        module = _import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    return _raise_missing_attribute(name)


def _raise_missing_attribute(name: str) -> _NoReturn:
    if name in _RETIRED_MODULES:
        raise AttributeError(
            f"antecedent.{name} was renamed to antecedent.{_RETIRED_MODULES[name]} in 0.4.0"
        )
    if name.startswith(_RETIRED_DISCOVER_PREFIX):
        raise AttributeError(
            f"antecedent.{name}(...) was removed in 0.4.0; the 16 free discover_* "
            "functions were replaced by methods on the discovery config dataclasses "
            "in antecedent.discovery (e.g. antecedent.discovery.PC(...).run(data) "
            "/ .accept(data))"
        )
    if name.startswith(_RETIRED_DAG_PREFIXES):
        raise AttributeError(
            f"antecedent.{name}(...) was removed in 0.4.0; the 10 free dag_from_*/"
            "dag_to_* helpers were replaced by class methods -- Dag.from_dot / "
            ".to_dot() and the JSON/GML/NetworkX peers, likewise on Cpdag/Pag/Admg"
        )
    if name.startswith(_RETIRED_TARGET_PREFIX):
        raise AttributeError(
            f"antecedent.{name}(...) was removed in 0.4.0; the target_* population "
            "builders moved to antecedent.population "
            "(e.g. antecedent.population.target_all())"
        )
    if name == "TransportQuery":
        raise AttributeError(
            "antecedent.TransportQuery was removed from the root in 2.0; the trial-IPW "
            "query is antecedent.transport.advanced.TransportQuery, and the ordinary "
            "question wrapper is antecedent.transport.Transport "
            "(see docs/migrations/2.0-transport-day1.md)"
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
