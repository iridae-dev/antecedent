"""Actual finite transport proposal arrivals; structural checks stay in proposals.

Observed count tables are evaluated through the original checked transport program.
The exported artifact retains the original repair/ranking and complete raw providers;
its consumer reproduces every output without granting interval calibration.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from . import _native
from .errors import CausalTypeError, CausalValueError
from .graph import Admg
from .transport._impl import EvidenceCatalog, ExactDiscreteLaw


@dataclass(frozen=True, slots=True)
class ArrivedStudy:
    """One independent, joint finite study, with actual count-bearing providers.

    The catalog includes the original unchanged catalog, one Available delivered
    regime and its actual snapshot binding. Distinct treatment worlds are disjoint
    study arms: their observed counts must sum to the planned study sample size.
    """

    graph: Admg
    catalog: EvidenceCatalog
    arrived_laws: Sequence[ExactDiscreteLaw]
    assignment: Mapping[str, float]
    base_laws: Sequence[ExactDiscreteLaw] = ()
    operations: int = 100_000
    depth: int = 128
    support_rows: int = 65_536


@dataclass(frozen=True, slots=True)
class ArrivalResult:
    """Original checked atom masses, mean, source lineage and point-only status."""

    identity: str
    _report: str
    _artifact: bytes

    @property
    def mean(self) -> float | None:
        value = json.loads(self._report)["result"]["mean"]
        return None if value is None else float(value)

    @property
    def proposal_identity(self) -> str:
        return str(json.loads(self._report)["result"]["proposal_identity"])

    def to_dict(self) -> dict[str, Any]:
        return dict(json.loads(self._report)["result"])

    def export(self) -> bytes:
        return self._artifact


def _estimate(
    repair_artifact: bytes,
    ranking_artifact: bytes,
    candidate_id: str,
    study: ArrivedStudy,
    expected_proposal: str,
    seed: int,
) -> ArrivalResult:
    if not isinstance(study, ArrivedStudy):
        raise CausalTypeError("study must be an ArrivedStudy")
    if not isinstance(study.graph, Admg) or not isinstance(study.catalog, EvidenceCatalog):
        raise CausalTypeError("ArrivedStudy requires Admg and EvidenceCatalog")
    if len(study.arrived_laws) > 64 or len(study.base_laws) > 64:
        raise CausalValueError(
            "proposal_arrival.bounds_exceeded", reason_code="route_not_supported"
        )
    names = study.graph.nodes()
    if len(names) > 12:
        raise CausalValueError(
            "proposal_arrival.bounds_exceeded", reason_code="route_not_supported"
        )
    if any(name not in names for name in study.assignment):
        raise CausalValueError(
            "proposal_arrival.assignment_mismatch", reason_code="invalid_argument"
        )
    request = {
        "repair_artifact": [],
        "ranking_artifact": [],
        "candidate_id": candidate_id,
        "catalog": {"environments": [], "regimes": [], "bindings": [], "target_sampling": None},
        "base_laws": [],
        "arrived_laws": [],
        "assignment": [
            (names.index(name), {"float64": float(value)})
            for name, value in study.assignment.items()
        ],
        "operations": study.operations,
        "depth": study.depth,
        "support_rows": study.support_rows,
    }
    report, artifact = _native.estimate_proposal_arrival(
        study.graph,
        study.catalog,
        list(study.base_laws),
        list(study.arrived_laws),
        repair_artifact,
        ranking_artifact,
        json.dumps(request, allow_nan=False),
        seed,
    )
    value = ArrivalResult(str(json.loads(report)["identity"]), report, bytes(artifact))
    if value.proposal_identity != expected_proposal:
        raise CausalValueError(
            "proposal_arrival.expected_proposal_mismatch", reason_code="invalid_argument"
        )
    return value


def consume_arrival(data: bytes, *, expected_proposal: str, seed: int = 3) -> ArrivalResult:
    """Replay original consumers and actual raw count providers in a fresh process."""
    if not isinstance(data, bytes):
        raise CausalTypeError("data must be bytes from ArrivalResult.export()")
    if len(data) > 32 * 1024 * 1024:
        raise CausalValueError(
            "proposal_arrival.bounds_exceeded", reason_code="route_not_supported"
        )
    report = _native.consume_proposal_arrival(data, expected_proposal, seed)
    return ArrivalResult(str(json.loads(report)["identity"]), report, data)


__all__ = ["ArrivedStudy", "ArrivalResult", "consume_arrival"]
