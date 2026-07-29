"""antecedent — Python bindings for the Antecedent causal engine.

Day-1 surface::

    result = antecedent.analyze(
        data,
        graph=edges,
        query=antecedent.AverageEffect(treatment="t", outcome="y"),
    )

The root namespace is **frozen**: it holds the three verbs (:func:`analyze`,
:func:`identify`, :func:`estimate`), the accepted-structure and result types,
the nine typed queries, the five graph classes, the inference / identifier /
estimator selectors, and the two error names most callers catch. Everything
else lives in a stage module and is reached through it:

``antecedent.attribution``, ``antecedent.data``, ``antecedent.design``,
``antecedent.discovery``, ``antecedent.errors``, ``antecedent.estimation``,
``antecedent.extensibility``, ``antecedent.gcm``, ``antecedent.graph``,
``antecedent.priors``, ``antecedent.state``, ``antecedent.validation``.

Graph interchange is on the classes: ``Dag.from_dot`` / ``Dag.to_dot`` and the
JSON / GML / NetworkX peers, likewise on ``Cpdag`` / ``Pag`` / ``Admg``.

Public analysis results are nested ``AnalysisResult`` views. The native DTOs
live on ``antecedent._native`` only, which is an advanced FFI surface.
"""

from __future__ import annotations

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

# Reachable as ``antecedent.<name>`` but deliberately outside the frozen
# ``__all__``: their public content is re-exported above (queries, inference
# selectors) or belongs to a narrower stage surface.
from . import counterfactual as counterfactual
from . import inference as inference
from . import model as model
from . import population as population
from . import query as query
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
    AverageEffect,
    ConditionalEffect,
    Counterfactual,
    InterventionalDistribution,
    MediationEffect,
    PathSpecificEffect,
    PulseEffect,
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
    "AverageEffect",
    "ConditionalEffect",
    "Counterfactual",
    "InterventionalDistribution",
    "MediationEffect",
    "PathSpecificEffect",
    "PulseEffect",
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
    __version__ = "0.4.0"
