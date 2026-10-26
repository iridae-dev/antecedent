"""Immutable policy preparation over a checked native topology and owned observations.

Preparation parses and lowers programs, validates their dependency order and owns
observations. Mechanism fitting and Monte Carlo sampling occur only on execution.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from . import _native
from ._data import as_columns, project_columns, table_column_names
from .errors import CausalTypeError, CausalUnsupportedError, CausalValueError
from .execution import Execution
from .graph import Dag
from .intervention_programs import (
    PolicyContract,
    PolicyPreparation,
    PolicyProgramResult,
    PolicyProgramValue,
    PreparedPolicy,
    _text,
)
from .outcomes import Outcome, capture_outcome
from .results._report import InspectionReport, SlotModel

_STRUCTURAL_ASSUMPTIONS = (
    "the declared graph is the causal structure: its edges are the only direct causes and no "
    "unobserved variable causes two of its variables",
    "each variable's exogenous noise is independent of every other variable's",
    "the fitted mechanisms are the structural mechanisms, with their fitted parameters taken as "
    "exact (the model's own uncertainty is unquantified)",
    "the program changes only the value it states for its node; every other mechanism is unchanged",
)


def _graph_edges(graph: Any) -> tuple[list[tuple[str, str]], tuple[str, ...] | None]:
    if isinstance(graph, Dag):
        return [(str(a), str(b)) for a, b in graph.edges()], tuple(graph.nodes())
    if (
        isinstance(graph, Sequence)
        and not isinstance(graph, str)
        and all(
            isinstance(edge, Sequence) and not isinstance(edge, str) and len(edge) == 2
            for edge in graph
        )
    ):
        return [(str(a), str(b)) for a, b in graph], None
    raise CausalTypeError(
        "a policy-program question needs graph=Dag(...) or a list of (cause, effect) edges"
    )


@dataclass(frozen=True, slots=True)
class PreparedPolicyAnalysis:
    """Private backend of the common immutable prepared-analysis lifecycle."""

    _native: Any
    _query: PolicyProgramValue
    _column_names: tuple[str, ...]
    execution: Execution
    _completed: bool = False

    @classmethod
    def build(
        cls,
        data: Any,
        *,
        query: PolicyProgramValue,
        graph: Any,
        execution: Execution | None = None,
        seed: int = 1,
        threads: int | None = None,
        options: Mapping[str, Any] | None = None,
        accept_discovered: bool = True,
    ) -> PreparedPolicyAnalysis:
        options = {} if options is None else options
        for name, value in options.items():
            if name not in ("cancel", "on_progress", "on_stage") and value is not None:
                raise CausalUnsupportedError(
                    f"{name}= does not apply to a policy-program question",
                    reason_code="option_not_applicable",
                )
        if not accept_discovered:
            raise CausalUnsupportedError(
                "a policy-program question has no discovered structure to review",
                reason_code="option_not_applicable",
            )
        policy = execution or Execution(
            seed=seed,
            threads=threads,
            cancel=options.get("cancel"),
            on_progress=options.get("on_progress"),
            on_stage=options.get("on_stage"),
        )
        edges, nodes = _graph_edges(graph)
        names = tuple(nodes) if nodes is not None else tuple(table_column_names(data) or ())
        if not names:
            raise CausalValueError("a policy-program question needs a table of named columns")
        table_names, columns = as_columns(project_columns(data, names, what="prepared"))
        arguments = (table_names, columns, edges, _text(query.programs), query.outcome)
        # The ordinary lowering adapter already runs inside the canonical Execution.
        # Preserve that active context when it has not passed its declaration explicitly.
        native = (
            _native._PreparedPolicyAnalysis(*arguments, **query.fit_options())
            if execution is None
            else execution._call(_native._PreparedPolicyAnalysis, *arguments, **query.fit_options())
        )
        native.quote(query.n_draws)
        return cls(native, query, names, policy)

    def _retain_preparation(
        self, request: Mapping[str, Any], columns: Mapping[str, Any], query: Any, resolved: Any
    ) -> PreparedPolicyAnalysis:
        # Policy expressions name structural variables, not row-bound design columns.
        # The common lifecycle already detached this checked declaration recursively.
        return replace(self, _query=query)

    def _branch(self, execution: Execution) -> PreparedPolicyAnalysis:
        return replace(self, _native=self._native.fork(), execution=execution, _completed=False)

    def _bind(self, data: Any, execution: Execution) -> PreparedPolicyAnalysis:
        names, columns = as_columns(project_columns(data, self._column_names, what="prepared"))
        native = execution._call(self._native.bind, names, columns)
        return replace(self, _native=native, execution=execution, _completed=False)

    def estimate(self) -> Outcome[PolicyProgramResult]:
        return capture_outcome(self.execution._call, self._execute)

    def _execute(self) -> PolicyProgramResult:
        if self.execution.on_stage is not None:
            raise CausalUnsupportedError(
                "policy simulation has no scalar identification stage stream; use on_progress",
                reason_code="stage_stream_unavailable",
            )
        fitted, native_policy = self._native.materialize(
            seed=self.execution.seed, threads=self.execution.threads
        )
        policy = object.__new__(PreparedPolicy)
        policy._native = native_policy
        query = self._query
        value = policy.value(
            query.outcome,
            n=query.n_draws,
            seed=self.execution.seed,
            threads=self.execution.threads,
        )
        conditional = tuple(
            f"{route} licence: {decision.assumption}"
            for route in ("simulation", "counterfactual", "identification")
            if (decision := getattr(value.support, route)).assumption
        )
        contract = PolicyContract(
            route="simulation",
            assumptions=_STRUCTURAL_ASSUMPTIONS + conditional,
            mechanisms={
                str(record["variable"]): str(record["selected"])
                for record in fitted.mechanism_selection()
            },
            program_identity=policy.identity,
        )
        return PolicyProgramResult(
            value=value,
            contract=contract,
            preparation=PolicyPreparation(program_compilations=1, model_bindings=1, runs=1),
            _native=fitted,
            _prepared=replace(self, _completed=True),
        )

    @property
    def preparation(self) -> PolicyPreparation:
        return PolicyPreparation(
            program_compilations=1,
            model_bindings=int(self._completed),
            runs=int(self._completed),
        )

    def inspect(self) -> InspectionReport:
        return InspectionReport(
            identification=SlotModel(
                available=True, summary="model-based policy value; topology and program checked"
            ),
            support=SlotModel(
                available=self._completed,
                summary="route licences resolved against fitted mechanisms"
                if self._completed
                else "mechanism-dependent route licences resolve on execution",
            ),
            uncertainty=SlotModel(
                available=True,
                summary="Monte Carlo error only; fitted model uncertainty is unquantified",
            ),
            assumptions=SlotModel(
                available=True,
                summary=f"{len(_STRUCTURAL_ASSUMPTIONS)} structural assumptions",
                payload={"assumptions": _STRUCTURAL_ASSUMPTIONS},
            ),
            program_id=self._native.identity,
            stage="executed" if self._completed else "prepared",
            family="policy_program",
            details={
                "phase": "executed" if self._completed else "compiled",
                "fit_materialized": self._native.fit_materialized,
                "draws": self._query.n_draws,
                "work_dimensions": self._native.quote(self._query.n_draws),
            },
            operations=("inspect", "estimate", "rebind", "reconfigure"),
        )


__all__ = ["PreparedPolicyAnalysis"]
