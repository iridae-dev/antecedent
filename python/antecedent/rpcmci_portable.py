"""Retained-source learned RPCMCI, with independent replay and explicit regime review."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

from . import _native
from ._artifact_protocol import (
    ArtifactExpectation,
    ExportedArtifact,
    artifact_consumer,
    artifact_export,
    container_bytes,
)
from ._coerce import coerce_data
from ._refusal import raise_refusal
from ._scientific import ScientificResult
from ._validate import count
from .accepted_graph import AcceptedGraph
from .errors import CausalValueError, StructuredRefusal
from .execution import Execution, native_controls, resolve_execution
from .outcomes import outcome_operation

if TYPE_CHECKING:
    from .discovery import RPCMCI

T = TypeVar("T")


class LearnedRpcmciRefusal(StructuredRefusal):
    """A refused declaration, replay, resource admission, or regime review."""


def _checked(pair: tuple[T | None, str | None]) -> T:
    result, refusal = pair
    raise_refusal(refusal, LearnedRpcmciRefusal)
    if result is None:
        raise CausalValueError("learned RPCMCI returned neither a result nor a refusal")
    return result


@dataclass(frozen=True, slots=True)
class LearnedRpcmciRegimeScope:
    """Source rows, contiguous original lag windows and stability for one graph."""

    artifact_identity: str
    data_identity: str
    regime: int
    graph_identity: str
    assigned_rows: tuple[int, ...]
    window_end_rows: tuple[int, ...]
    lookback: int
    graph_agreement: float | None
    assignment_agreement: float | None


class LearnedRpcmciArtifact(ScientificResult):
    """The original data-driven labels, graphs and stability report as replayable evidence."""

    def __init__(self, handle: _native.LearnedRpcmciHandle) -> None:
        self._handle = handle

    @property
    def identity(self) -> str:
        """Identity to retain independently of the portable bytes."""
        return self._handle.identity

    @property
    def report(self) -> dict[str, Any]:
        """The retained program, seed, assignments, graph identities and diagnostics."""
        return dict(json.loads(self._handle.report_json()))

    def plan_view(self) -> dict[str, Any]:
        """Read the regular causal compiler plan reconstructed from the source."""
        return dict(json.loads(self._handle.plan_view_json()))

    @artifact_export
    def export(self) -> ExportedArtifact[LearnedRpcmciArtifact]:
        """Serialize the native configuration, raw series and verified report."""
        kind = "learned_rpcmci_v1"
        return ExportedArtifact(
            self._handle.export(), kind, ArtifactExpectation(kind, self.identity)
        )

    @outcome_operation
    def regime_scope(self, regime: int) -> LearnedRpcmciRegimeScope:
        """Read the original lag-window endpoints for a verified regime graph."""
        payload = json.loads(
            _checked(self._handle.regime_scope_json(count(regime, "regime", minimum=0)))
        )
        payload["assigned_rows"] = tuple(payload["assigned_rows"])
        payload["window_end_rows"] = tuple(payload["window_end_rows"])
        return LearnedRpcmciRegimeScope(**payload)

    @outcome_operation
    def accept_regime(self, regime: int) -> AcceptedGraph:
        """Choose one learned graph. Read regime_scope for its original data windows."""
        graph = _checked(self._handle.accept_regime(count(regime, "regime", minimum=0)))
        return AcceptedGraph.from_graph(graph, algorithm_id="rpcmci")


def _run(config: RPCMCI, data: Any, execution: Execution | None) -> LearnedRpcmciArtifact:
    learning = config._learning()
    if learning is None or config.ci != "parcorr":
        raise CausalValueError("portable learned RPCMCI needs a count policy and named parcorr")
    count_policy, select_count, starts, iterations, penalty, min_length = learning
    names, cols = coerce_data(data, temporal=True)
    policy = resolve_execution(execution)
    request = json.dumps(
        {
            "names": names,
            "columns": [column.tolist() for column in cols],
            "interval_ns": 1,
            "count": count_policy,
            "select_count": select_count,
            "starts": starts,
            "max_iterations": iterations,
            "switch_penalty": penalty,
            "min_regime_len": min_length,
            "max_lag": config.max_lag,
            "alpha": config.alpha,
            "fdr": config.fdr,
            "max_cond_size": config.max_cond_size,
        },
        allow_nan=False,
    )
    return LearnedRpcmciArtifact(_checked(policy._call(
        _native.learned_rpcmci_run,
        request,
        **native_controls(policy, seed="seed", threads="threads", cancel="cancel", memory="memory_bytes"),
    )))


@outcome_operation
@artifact_consumer(
    proof="Recompile the raw time series and rerun the seeded regime search to recover the assignments, graphs and stability report.",
    replayable=True,
    input="data",
)
def consume_learned_rpcmci(
    data: Any,
    *,
    expected_identity: str,
    execution: Execution | None = None,
) -> LearnedRpcmciArtifact:
    """Rebuild from the embedded series and seed under the caller's resource controls."""
    policy = resolve_execution(execution)
    return LearnedRpcmciArtifact(_checked(policy._call(
        _native.learned_rpcmci_consume,
        container_bytes(data),
        expected_identity,
        **native_controls(policy, threads="threads", cancel="cancel", memory="memory_bytes"),
    )))
