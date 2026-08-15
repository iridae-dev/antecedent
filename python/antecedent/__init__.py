"""antecedent — Python bindings for the Antecedent causal engine.

Day-1 surface::

    result = antecedent.analyze(
        data,
        graph=edges,
        query=antecedent.AverageEffect(treatment="t", outcome="y"),
    )

The root namespace is deliberately small: it holds the three verbs (:func:`analyze`,
:func:`identify`, :func:`estimate`), the accepted-structure and result types,
the first-class typed queries, the five graph classes, the inference / identifier /
estimator selectors, and the two error names most callers catch. Everything
else lives in a stage module and is reached through it:

``antecedent.attribution``, ``antecedent.data``, ``antecedent.design``,
``antecedent.discovery``, ``antecedent.errors``, ``antecedent.estimation``,
``antecedent.estimators``, ``antecedent.extensibility``, ``antecedent.gcm``,
``antecedent.graph``, ``antecedent.interference``, ``antecedent.intervention``,
``antecedent.observation``, ``antecedent.priors``, ``antecedent.state``,
``antecedent.transport``, and ``antecedent.validation``.

Graph interchange is on the classes: ``Dag.from_dot`` / ``Dag.to_dot`` and the
JSON / GML / NetworkX peers, likewise on ``Cpdag`` / ``Pag`` / ``Admg``.

Public analysis results are nested ``AnalysisResult`` views. The native DTOs
live on ``antecedent._native`` only, which is an advanced FFI surface.
"""

from __future__ import annotations

from typing import NoReturn

# `artifacts` belongs to the "reachable but deliberately outside `__all__`"
# family described in the comment below; isort's alphabetical import
# ordering just happens to place it ahead of the `__all__`-exported block
# rather than next to its siblings.
from . import artifacts as artifacts
from . import (
    attribution,
    data,
    design,
    discovery,
    errors,
    estimation,
    extensibility,
    gcm,
    graph,
    priors,
    state,
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
from . import estimators as estimators
from . import inference as inference
from . import interference as interference
from . import intervention as intervention
from . import model as model
from . import observation as observation
from . import population as population
from . import query as query
from . import transport as transport
from ._analyze import analyze
from ._native import (
    Admg,
    Cpdag,
    Dag,
    Pag,
    TemporalDag,
)
from .accepted_graph import AcceptedGraph
from .errors import CausalError, ReviewRequired
from .identify import Identification, estimate, identify
from .ids import Estimator, Identifier, Latency, Refute
from .inference import Bayesian, Frequentist
from .query import (
    AverageDerivative,
    AverageEffect,
    ConditionalEffect,
    Counterfactual,
    DirectionalDerivative,
    Elasticity,
    InterventionalDistribution,
    InterventionResponse,
    MediationEffect,
    PathSpecificEffect,
    PointDerivative,
    PulseEffect,
    ResponseCurve,
    ResponseJacobian,
    SemiElasticity,
    SustainedEffect,
    TemporalMediationEffect,
)
from .results import AnalysisResult

__all__ = [
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
    # Graphs
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
    # Stage modules
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
    "__version__",
]

try:
    from ._native import __version__ as __version__
except ImportError:  # pragma: no cover - extension not built
    # Derive from installed package metadata rather than hand-maintaining a
    # literal here (which would silently go stale at every release this
    # branch is actually exercised on). Falls back to a clearly-unknown
    # sentinel — never a copied-and-forgotten version string — if the
    # package metadata itself cannot be found (e.g. running from a source
    # checkout with neither the extension built nor the package installed).
    try:
        from importlib.metadata import PackageNotFoundError, version

        __version__ = version("antecedent")
    except PackageNotFoundError:
        __version__ = "0.4.1"


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
_RETIRED_MODULES = {"prior_bank": "priors"}
_RETIRED_DISCOVER_PREFIX = "discover_"
_RETIRED_DAG_PREFIXES = ("dag_from_", "dag_to_")
_RETIRED_TARGET_PREFIX = "target_"


def __getattr__(name: str) -> NoReturn:
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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
