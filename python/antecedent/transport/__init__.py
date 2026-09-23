"""Transport a response into a target population.

Day-1 surface::

    query = antecedent.transport.Transport(
        antecedent.ResponseCurve("price", "sales", grid=[8, 9, 10, 11, 12]),
        target="new_market",
        evidence=evidence,
    )
    result = antecedent.analyze(data, graph=graph, query=query)

``evidence`` carries source identity, intervention regime, and sampling claims.
Selection differences stay on ``Transport.selections`` or per-``Source``.
Conservative estimation uses :class:`EmpiricalTable`; :class:`LearnedCategorical`
and :class:`TrialAipw` are opt-in assumption changes.

The licensed trial-IPW cell is :class:`antecedent.transport.advanced.TransportQuery`.
Theorem-stage request types live in :mod:`antecedent.transport.advanced`.
"""

from __future__ import annotations

from typing import NoReturn

from . import advanced as advanced
from ._day1 import Evidence, Source, Transport
from ._impl import (
    EmpiricalTable,
    ExactDiscreteLaw,
    ExactTransportData,
    LearnedCategorical,
    RegimeSample,
    StatisticalTransportData,
    TransportControls,
    TransportInference,
    TrialAipw,
    TrialAipwData,
)

# Names 1.11 exported from ``antecedent.transport`` that 2.0 moved to
# ``antecedent.transport.advanced``. They are not re-exported here: one spelling
# per object. The error names the new home instead of a bare AttributeError.
_MOVED_TO_ADVANCED = frozenset(
    {
        "DirectFormula",
        "NonTransportableCertificate",
        "PopulationFactor",
        "RecursiveFactorizationFormula",
        "SelectionDiagram",
        "StandardizationFormula",
        "TransportCertificate",
        "TransportIdentification",
        "TransportQuery",
        "TrialTransportEstimate",
        "estimate_trial_effect",
        "identify",
    }
)
# Removed from the public namespace; reachable only as the ``.overlap`` field of
# ``TrialTransportEstimate``.
_REMOVED = frozenset({"OverlapDiagnostic", "TransportOverlapReport"})


def __getattr__(name: str) -> NoReturn:
    if name in _MOVED_TO_ADVANCED:
        raise AttributeError(
            f"antecedent.transport.{name} moved to antecedent.transport.advanced.{name} "
            "in 2.0 (see docs/migrations/2.0-transport-day1.md)"
        )
    if name in _REMOVED:
        raise AttributeError(
            f"antecedent.transport.{name} is no longer exported in 2.0; it is the type of "
            "TrialTransportEstimate.overlap (see docs/migrations/2.0-transport-day1.md)"
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Transport",
    "Evidence",
    "Source",
    "EmpiricalTable",
    "LearnedCategorical",
    "TrialAipw",
    "TransportInference",
    "TransportControls",
    "ExactTransportData",
    "StatisticalTransportData",
    "TrialAipwData",
    "ExactDiscreteLaw",
    "RegimeSample",
]
