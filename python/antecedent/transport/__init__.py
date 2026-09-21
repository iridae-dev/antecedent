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
