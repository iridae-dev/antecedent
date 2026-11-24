"""Prepared fitted-model operations: the lifecycle backend of the ``model_operations`` queries.

Preparation checks the declarations and retains an owned copy of the rows; estimation fits the
standard mechanism program on the supplied graph and runs the declared operation through its
compiled program (an operation of a prepared study, the ``Attribute`` route for the
do-contrast, or the ``Simulate`` route for the means and the nested counterfactual).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from ._study_lifecycle import retain
from ._table_snapshot import Table, table
from .errors import CausalUnsupportedError
from .execution import CachePolicy, Execution
from .model_operations import ModelOperationQuery, _edges, analyze_model_operation
from .outcomes import Outcome, capture_outcome
from .results._report import InspectionReport, SlotModel
from .results.model_operation import ModelOperationResult


def _inapplicable(name: str, query: Any) -> None:
    raise CausalUnsupportedError(
        f"{name} does not apply to {type(query).__name__}: a fitted-model operation fits the "
        "standard mechanism program on the supplied graph and reads only data, graph and the "
        "execution's seed and threads",
        reason_code="option_not_applicable",
    )


def _check_execution(policy: Execution, query: Any) -> None:
    for name, changed in (
        ("execution.cache", policy.cache != CachePolicy()),
        ("execution.kernels", policy.kernels != "auto"),
        ("execution.on_stage", policy.on_stage is not None),
    ):
        if changed:
            _inapplicable(name, query)


@dataclass(frozen=True, slots=True)
class PreparedModelOperation:
    """Private backend of the common prepared-analysis lifecycle for a fitted-model operation."""

    _data: Table
    _query: ModelOperationQuery
    _edges: tuple[tuple[str, str], ...]
    execution: Execution
    _completed: bool = False

    @property
    def validator_names(self) -> tuple[str, ...]:
        return ()

    @classmethod
    def prepare(
        cls,
        data: Any,
        *,
        query: ModelOperationQuery,
        graph: Any,
        execution: Execution,
        **declarations: Any,
    ) -> PreparedModelOperation:
        _check_execution(execution, query)
        defaults = {
            "accept_discovered": False,
            "retain_joint_horizons": False,
            "on_refusal": "refuse",
            "quantity_population": "target",
            "quantity_transform": "identity",
        }
        for name, value in declarations.items():
            changed = value != defaults[name] if name in defaults else value is not None
            if changed:
                _inapplicable(name, query)
        return cls(execution._call(table, data), retain(query), tuple(_edges(graph)), execution)

    def _branch(self, execution: Execution) -> PreparedModelOperation:
        _check_execution(execution, self._query)
        return replace(self, execution=execution, _completed=False)

    def _bind(self, data: Any, execution: Execution) -> PreparedModelOperation:
        branch = self._branch(execution)
        return replace(branch, _data=execution._call(table, data))

    def estimate(self) -> Outcome[ModelOperationResult]:
        return capture_outcome(self._estimate)

    def _estimate(self) -> ModelOperationResult:
        execution = self.execution
        execution._report_progress(0.0, "model_operation")
        output = analyze_model_operation(
            self._data,
            query=self._query,
            graph=list(self._edges),
            execution=execution,
            inapplicable={},
        )
        execution._report_progress(1.0, "model_operation")
        object.__setattr__(output, "_prepared", replace(self, _completed=True))
        return output

    def inspect(self) -> InspectionReport:
        return InspectionReport(
            identification=SlotModel(
                available=False,
                reason="fitted_model_operation",
                summary="A statement about the mechanisms fitted on the supplied graph",
            ),
            support=SlotModel(
                available=False,
                reason="route_contract" if self._completed else "not_executed",
                summary="The compiled route's contract and claim are on the result's route",
            ),
            uncertainty=SlotModel(
                available=False,
                reason="not_measured",
                summary="Only the Monte Carlo error a sampler reports; no interval is claimed",
            ),
            assumptions=SlotModel(
                available=True,
                summary="The supplied DAG and the standard mechanism program",
                payload={"edges": [list(edge) for edge in self._edges]},
            ),
            stage="executed" if self._completed else "prepared",
            family="model_operation",
            details={"query": type(self._query).__name__},
            operations=("inspect", "estimate", "rebind", "reconfigure"),
        )

    def report(self, format: str = "markdown") -> str:
        from .errors import CausalValueError

        if format not in ("markdown", "html"):
            raise CausalValueError("report format must be markdown or html")
        report = self.inspect()
        if format == "html":
            return report._repr_html_()
        return (
            f"{type(self._query).__name__} ({report.stage})\n\n"
            "A fitted-model operation run through its compiled route; the answer describes the "
            "fitted mechanisms, not an identified effect."
        )


__all__ = ["PreparedModelOperation"]
