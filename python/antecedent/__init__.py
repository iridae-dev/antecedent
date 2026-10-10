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
2.1 families — randomized and factorial experiments, held-out policy value,
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

The 2.3 stage modules are loaded on first attribute access (``antecedent.recalc_cell`` imports
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
  treatment on the treated on a bounded ADMG.

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

Graph interchange is on the classes: ``Dag.from_dot`` / ``Dag.to_dot`` and the
JSON / GML / NetworkX peers, likewise on ``Cpdag`` / ``Pag`` / ``Admg``.

Public analysis results share :data:`Analysis` (``AnalysisResult`` or
``CausalResponseView``). Consume ``result.answer`` / ``result.claim()``.
The native DTOs live on ``antecedent._native`` only.
"""

from __future__ import annotations

from importlib import import_module as _import_module
from types import ModuleType as _ModuleType
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
# The 2.3 stage modules are public (their classes and functions are documented in each
# module's docstring) but none is in ``__all__`` and importing them all would put the whole
# family on the cost of ``import antecedent``. ``__getattr__`` imports one on first access and
# caches it on the package, ``__dir__`` lists them so tab completion finds them, and neither
# adds a name to ``__all__``. ``identify`` is absent on purpose: the root name is the function.
_LAZY_MODULES: tuple[str, ...] = (
    "categorical_treatment",
    "compact_export",
    "composition",
    "composition_bundle",
    "counterfactual_id",
    "cross_world",
    "decision_robust",
    "descriptive",
    "dose_grid",
    "effect_constancy_review",
    "execution_attempt",
    "functional_source",
    "inverse",
    "inverse_query",
    "joint_distribution",
    "latent_class",
    "mechanism_discrepancy",
    "msm_sensitivity",
    "nonlinear_mediation",
    "preflight",
    "program_claims",
    "proposal_arrival",
    "proposals",
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
    "sensitivity_decision",
    "source_evidence",
    "source_projection",
    "temporal",
    "temporal_counterfactual",
    "transported_counterfactual",
    "vector_treatment",
)

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
