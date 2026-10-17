"""The retained handle of a policy-program question.

``antecedent.prepare(data, query=PolicyProgramValue(...), graph=...)`` fits the structural model
once, compiles the program and resolves the licence of each route against it once (the library's
``PreparedPolicy``), and returns a :class:`~antecedent.estimation.PreparedAnalysis`. Every
``estimate`` / ``refresh`` runs the library's ``policy_value`` executor on the retained program;
nothing here evaluates an expression or draws a number. The estimand operations that do not
denote for a model-based value (refutation, retargeting, portable claims, adjustment diagnostics)
are refused with ``option_not_applicable``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Any, NoReturn

from . import _native
from ._api import describe_refusal
from ._data import as_columns, project_columns, table_column_names
from .errors import CausalTypeError, CausalUnsupportedError, CausalValueError
from .estimation import _UNSET, PreparedAnalysis
from .graph import Dag
from .intervention_programs import (
    PolicyContract,
    PolicyPreparation,
    PolicyProgramResult,
    PolicyProgramValue,
    PreparedPolicy,
)
from .results._report import InspectionReport, SlotModel
from .results._report import InspectionStage as _Stage

#: What a model-based policy value assumes, whatever the program.
_STRUCTURAL_ASSUMPTIONS = (
    "the declared graph is the causal structure: its edges are the only direct causes and no "
    "unobserved variable causes two of its variables",
    "each variable's exogenous noise is independent of every other variable's",
    "the fitted mechanisms are the structural mechanisms, with their fitted parameters taken as "
    "exact (the model's own uncertainty is unquantified)",
    "the program changes only the value it states for its node; every other mechanism is unchanged",
)

#: Options of ``antecedent.prepare`` a policy-program question does not take.
_NOT_TAKEN = (
    "discovery",
    "inference",
    "identifier",
    "estimator",
    "estimator_config",
    "refute",
    "bootstrap",
    "latency",
    "class_prior",
    "max_completions",
    "participant",
    "missing",
    "population_registry",
    "cancel",
    "on_progress",
    "on_stage",
    "validators",
    "regimes",
    "provider",
    "controls",
)


def _not_applicable(what: str) -> CausalUnsupportedError:
    return CausalUnsupportedError(
        f"{what} does not apply to a policy-program question: its value is a property of the "
        "fitted model, not an identified estimand",
        reason_code="option_not_applicable",
    )


def _refusal(operation: str) -> Any:
    def refuse(self: Any, *args: Any, **kwargs: Any) -> NoReturn:
        raise _not_applicable(operation)

    refuse.__name__ = operation
    return refuse


def _graph_edges(graph: Any) -> tuple[list[tuple[str, str]], tuple[str, ...] | None]:
    """The directed edges, and the nodes when the graph names them (``None``: every column)."""
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


def _slot(summary: str, **payload: Any) -> SlotModel:
    return SlotModel(available=True, summary=summary, payload=payload)


def _route_summary(support: Any) -> str:
    def one(name: str, decision: Any) -> str:
        if not decision.licensed:
            return f"{name} refused ({decision.reason_code})"
        return f"{name} licensed" + (" under an assumption" if decision.assumption else "")

    return "; ".join(
        one(name, getattr(support, name))
        for name in ("simulation", "counterfactual", "identification")
    )


class PreparedPolicyAnalysis(PreparedAnalysis[PolicyProgramResult]):
    """Compile-once / rerun-many handle of a :class:`PolicyProgramValue` question.

    **Frozen at prepare:** the schema, the graph, the model-fit options, the program and its
    compiled overlay and route licences, the outcome and the draw budget. **Estimate click:**
    ``seed`` / ``threads``, and optionally same-schema data, on which the model is refitted and the
    program rebound (not compiled again); :meth:`refresh` also replaces the retained data.
    :attr:`preparation` is the receipt: ``program_compilations`` stays 1 across every rerun.
    """

    def __init__(
        self,
        policy: PreparedPolicy,
        fitted: Any,
        *,
        query: PolicyProgramValue,
        edges: list[tuple[str, str]],
        seed: int,
        threads: int | None,
    ) -> None:
        super().__init__(
            policy,
            kind="policy",
            query=query,
            seed=seed,
            threads=threads,
            column_names=tuple(fitted.names),
        )
        self._policy = policy
        self._fitted = fitted
        self._edges = edges
        self._runs = 0

    @classmethod
    def build(
        cls,
        data: Any,
        *,
        query: PolicyProgramValue,
        graph: Any,
        seed: int,
        threads: int | None,
        options: Mapping[str, Any],
        accept_discovered: bool,
    ) -> PreparedPolicyAnalysis:
        """Fit the model, compile the program against it and resolve the licences, once."""
        for name in _NOT_TAKEN:
            if options.get(name) is not None:
                raise CausalUnsupportedError(
                    f"{name}= does not apply to a policy-program question: the model is fitted "
                    "along graph= and the value is a simulation of it",
                    reason_code="option_not_applicable",
                )
        if not accept_discovered:
            raise CausalUnsupportedError(
                "accept_discovered=False does not apply: a policy-program question has no "
                "discovery to review",
                reason_code="option_not_applicable",
            )
        if graph is None:
            raise CausalTypeError(
                "a policy-program question needs graph=Dag(...) or a list of (cause, effect) edges"
            )
        edges, nodes = _graph_edges(graph)
        names = list(nodes) if nodes is not None else table_column_names(data)
        if not names:
            raise CausalValueError("a policy-program question needs a table of named columns")
        fitted = cls._fit(data, tuple(names), edges, query, threads)
        if query.outcome not in fitted.names:
            raise CausalValueError(f"outcome {query.outcome!r} is not a variable of the model")
        policy = PreparedPolicy(fitted, query.programs)
        return cls(policy, fitted, query=query, edges=edges, seed=seed, threads=threads)

    @staticmethod
    def _fit(
        data: Any,
        names: tuple[str, ...],
        edges: list[tuple[str, str]],
        query: PolicyProgramValue,
        threads: int | None,
    ) -> Any:
        table_names, columns = as_columns(project_columns(data, names, what="prepared"))
        return _native.fit_gcm(table_names, columns, edges, threads=threads, **query.fit_options())

    def _contract(self, policy: PreparedPolicy) -> PolicyContract:
        support = policy.support
        conditional = tuple(
            f"{route} licence: {decision.assumption}"
            for route in ("simulation", "counterfactual", "identification")
            if (decision := getattr(support, route)).assumption
        )
        return PolicyContract(
            route="simulation",
            assumptions=_STRUCTURAL_ASSUMPTIONS + conditional,
            mechanisms={
                str(record["variable"]): str(record["selected"])
                for record in self._fitted.mechanism_selection()
            },
            program_identity=policy.identity,
        )

    def _run(self, policy: PreparedPolicy, seed: int | None, threads: int | None) -> Any:
        query = self._query
        assert isinstance(query, PolicyProgramValue)
        value = policy.value(
            query.outcome,
            n=query.n_draws,
            seed=self._seed if seed is None else seed,
            threads=self._threads if threads is None else threads,
        )
        return value, self._contract(policy)

    def _click_policy(
        self,
        data: Any,
        *,
        refresh: bool,
        seed: int | None,
        threads: int | None,
        controls: tuple[Any, Any, Any],
    ) -> PolicyProgramResult:
        if any(control is not _UNSET and control is not None for control in controls):
            raise _not_applicable("cancel, on_progress and on_stage")
        query = self._query
        assert isinstance(query, PolicyProgramValue)
        fitted, policy = self._fitted, self._policy
        if data is not None:
            fitted = self._fit(
                data, tuple(self._column_names or ()), self._edges, query, self._threads
            )
            policy = policy.rebind(fitted)
        value, contract = self._run(policy, seed, threads)
        # Only a completed run commits the new data, so a refused click leaves the handle as it was.
        if refresh and data is not None:
            self._fitted, self._policy, self._native = fitted, policy, policy
        self._runs += 1
        result = PolicyProgramResult(value=value, contract=contract, preparation=self.preparation)
        self._last_result = result
        return result

    @property
    def preparation(self) -> PolicyPreparation:
        """The receipt of the handle: program compilations, model bindings and runs so far."""
        return PolicyPreparation(
            program_compilations=self._policy.program_compilations,
            model_bindings=self._policy.model_bindings,
            runs=self._runs,
        )

    @describe_refusal
    def estimate(
        self,
        data: Mapping[str, Any] | Any = None,
        *,
        seed: int | None = None,
        threads: int | None = None,
        cancel: Any = _UNSET,
        on_progress: Any = _UNSET,
        on_stage: Any = _UNSET,
    ) -> PolicyProgramResult:
        """Rerun the policy value without compiling the program again.

        With no ``data`` the retained model is used. New same-schema ``data`` refits the model
        and rebinds the program to it for this run only.
        """
        return self._click_policy(
            data,
            refresh=False,
            seed=seed,
            threads=threads,
            controls=(cancel, on_progress, on_stage),
        )

    @describe_refusal
    def refresh(
        self,
        data: Mapping[str, Any] | Any,
        *,
        seed: int | None = None,
        threads: int | None = None,
        cancel: Any = _UNSET,
        on_progress: Any = _UNSET,
        on_stage: Any = _UNSET,
    ) -> PolicyProgramResult:
        """Refit the model on ``data``, keep it, and rerun the value (the program is not
        compiled again)."""
        return self._click_policy(
            data,
            refresh=True,
            seed=seed,
            threads=threads,
            controls=(cancel, on_progress, on_stage),
        )

    @property
    def validator_names(self) -> tuple[str, ...]:
        """A policy-program question has no custom validators."""
        return ()

    @property
    def structure_source(self) -> str:
        """The graph is declared (``explicit``)."""
        return "explicit"

    @property
    def evidence_status(self) -> str | None:
        """A policy-program question is not a cell of the estimand support matrix."""
        return None

    @property
    def allowlist_reason(self) -> str | None:
        return None

    @property
    def allowlist_parent(self) -> str | None:
        return None

    def rebind_validators(self, validators: Mapping[str, Any]) -> None:
        if validators:
            raise _not_applicable("custom validators")

    def _report(self, stage: _Stage) -> InspectionReport:
        support = self._policy.support
        contract = self._contract(self._policy)
        return InspectionReport(
            identification=_slot(
                f"model-based value on the simulation route; {_route_summary(support)}",
                **{
                    route: {
                        "licensed": getattr(support, route).licensed,
                        "assumption": getattr(support, route).assumption,
                        "reason_code": getattr(support, route).reason_code,
                    }
                    for route in ("simulation", "counterfactual", "identification")
                },
            ),
            support=_slot(
                "simulation error only", program_compilations=self._policy.program_compilations
            ),
            uncertainty=_slot(
                "Monte Carlo error of the draws; the fitted model's uncertainty is unquantified"
            ),
            assumptions=_slot(
                f"{len(contract.assumptions)} assumptions", assumptions=contract.assumptions
            ),
            program_id=contract.program_identity,
            stage=stage,
            family="policy_program",
            details={
                "provenance_id": support.provenance_id,
                "preparation": asdict(self.preparation),
            },
        )

    def inspect(self) -> InspectionReport:
        """The licences, assumptions and provenance of the prepared question."""
        return self._report("executed" if self._last_result is not None else "prepared")

    def inspect_plan(self) -> InspectionReport:
        return self._report("prepared")

    def preflight(self) -> InspectionReport:
        return self._report("prepared")

    def __repr__(self) -> str:
        return (
            f"PreparedAnalysis(query=PolicyProgramValue, "
            f"stage={'executed' if self._last_result is not None else 'prepared'})"
        )


def _no_info(self: Any) -> None:
    """The retained-plan summaries of the estimand routes: none for a policy question."""
    return None


for _name in (
    "export_artifact",
    "export",
    "retarget",
    "reexecute_retarget",
    "refute",
    "diagnose",
    "diagnose_fit",
    "plan_rank_drop",
    "estimate_cost",
    "preview_transform",
    "replace_snapshot",
    "mechanism_sensitivity",
    "identification",
    "plan",
):
    setattr(
        PreparedPolicyAnalysis,
        _name,
        property(_refusal(_name)) if _name in ("identification", "plan") else _refusal(_name),
    )
del _name
for _name in (
    "checked_static_dag_response_info",
    "checked_conditional_effect_info",
    "checked_static_mediation_info",
    "checked_bayesian_dag_ate_info",
    "checked_graph_posterior_effect_info",
    "checked_temporal_effect_info",
    "checked_temporal_response_info",
):
    setattr(PreparedPolicyAnalysis, _name, _no_info)
del _name

__all__ = ["PreparedPolicyAnalysis"]
