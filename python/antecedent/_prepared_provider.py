"""Owned, immutable preparations for explicitly registered Python providers.

The registry remains the output validation and external trust boundary. Preparing
a request does not run provider code or create native scientific authority.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from ._study_lifecycle import retain
from ._table_snapshot import Table, table
from .errors import CausalUnsupportedError, CausalValueError
from .execution import CachePolicy, Execution
from .extensibility import ProviderQuery, _digest, _spec_evidence, providers
from .outcomes import Outcome, capture_outcome, require_success
from .results._report import InspectionReport, SlotModel
from .results.provider import ProviderAnalysisResult


def _inapplicable(name: str) -> None:
    raise CausalUnsupportedError(
        f"{name} does not apply to ProviderQuery; the provider execute(request) "
        "protocol owns its execution controls",
        reason_code="option_not_applicable",
    )


def _check_execution(policy: Execution) -> None:
    # This protocol has no seed, worker, budget, observer or cancellation hooks.
    # Accept only the baseline policy, without promising to seed provider code.
    for name, changed in (
        ("execution.seed", policy.seed != 1),
        ("execution.threads", policy.threads is not None),
        ("execution.limits", policy.limits.memory_bytes is not None),
        ("execution.cache", policy.cache != CachePolicy()),
        ("execution.kernels", policy.kernels != "auto"),
        ("execution.cancel", policy.cancel is not None),
        ("execution.on_progress", policy.on_progress is not None),
        ("execution.on_stage", policy.on_stage is not None),
    ):
        if changed:
            _inapplicable(name)


def _callback_identity(provider: Any) -> tuple[Any, Any]:
    callback = provider.execute
    return getattr(callback, "__self__", None), getattr(callback, "__func__", callback)


@dataclass(frozen=True, slots=True)
class PreparedProvider:
    """Private backend of the common prepared-analysis lifecycle."""

    _data: Table
    _query: ProviderQuery
    execution: Execution
    _provider: Any
    _spec: Any
    _spec_digest: str
    _callback: tuple[Any, Any]
    _registry_entry: tuple[Any, Any]
    _verification: Any
    _entry_point: str | None
    _completed: bool = False
    _report: InspectionReport | None = None

    @property
    def validator_names(self) -> tuple[str, ...]:
        # Provider requests reject sampled-data validators at preparation.
        return ()

    @classmethod
    def prepare(
        cls, data: Any, *, query: ProviderQuery, execution: Execution, **declarations: Any
    ) -> PreparedProvider:
        _check_execution(execution)
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
                _inapplicable(name)
        if "data" in query.request:
            raise CausalValueError(
                "ProviderQuery.request must not contain data; pass it to prepare(data, ...)"
            )
        provider = providers.get(query.provider)
        entry = providers._entry(query.provider)
        return cls(
            execution._call(table, data),
            retain(query),
            execution,
            provider,
            retain(provider.spec),
            _digest("spec", _spec_evidence(provider.spec)),
            _callback_identity(provider),
            entry,
            providers._verification.get(query.provider),
            providers._entry_points.get(query.provider),
        )

    def _check_provider(self) -> None:
        name = self._query.provider
        entry = providers._entry(name)
        callback = _callback_identity(self._provider)
        if (
            entry[0] is not self._provider
            or entry[1] is not self._registry_entry[1]
            or any(
                current is not pinned
                for current, pinned in zip(callback, self._callback, strict=True)
            )
            or _digest("spec", _spec_evidence(self._provider.spec)) != self._spec_digest
            or providers._verification.get(name) is not self._verification
            or providers._entry_points.get(name) != self._entry_point
        ):
            raise CausalValueError(
                "prepared provider identity, specification or verification changed; "
                "prepare the provider request again",
                reason_code="external_binding_mismatch",
            )

    def _branch(self, execution: Execution) -> PreparedProvider:
        _check_execution(execution)
        self._check_provider()
        return replace(self, execution=execution, _completed=False, _report=None)

    def _bind(self, data: Any, execution: Execution) -> PreparedProvider:
        branch = self._branch(execution)
        return replace(branch, _data=execution._call(table, data))

    def estimate(self) -> Outcome[ProviderAnalysisResult]:
        return capture_outcome(self._estimate)

    def _estimate(self) -> ProviderAnalysisResult:
        self._check_provider()
        result = require_success(
            providers.execute(self._query.provider, {"data": self._data, **self._query.request})
        )
        self._check_provider()
        output = ProviderAnalysisResult(
            provider_name=self._query.provider,
            query_family=self._spec.query_family,
            provider_result=result,
            query=self._query,
        )
        completed = replace(self, _completed=True, _report=output.inspect())
        object.__setattr__(output, "_prepared", completed)
        return output

    def inspect(self) -> InspectionReport:
        if self._report is not None:
            return self._report.model_copy(
                update={"operations": ("inspect", "estimate", "rebind", "reconfigure")}
            )
        spec = self._spec
        return InspectionReport(
            identification=SlotModel(
                available=False,
                reason="provider_identification_not_independently_verified",
                summary="Provider-declared requirements only",
            ),
            support=SlotModel(
                available=False,
                reason="provider_output_not_retained" if self._completed else "not_executed",
                summary="Provider output remains externally classified",
            ),
            uncertainty=SlotModel(
                available=False, reason="provider_owned", summary=spec.uncertainty_semantics
            ),
            assumptions=SlotModel(
                available=True,
                summary="Provider-declared requirements",
                payload={"requirements": spec.identification_requirements},
            ),
            stage="executed" if self._completed else "prepared",
            family="provider",
            details={
                "provider": self._query.provider,
                "query_family": spec.query_family,
                "spec_digest": self._spec_digest,
                "execution_controls": "provider_owned; baseline Execution only",
            },
            operations=("inspect", "estimate", "rebind", "reconfigure"),
        )

    def report(self, format: str = "markdown") -> str:
        if format not in ("markdown", "html"):
            raise CausalValueError("report format must be markdown or html")
        report = self.inspect()
        if format == "html":
            return report._repr_html_()
        return (
            f"Provider {self._query.provider!r} ({report.stage})\n\n"
            "Provider-declared requirements only. Output remains externally classified; "
            "no native licensed causal claim is created."
        )


__all__ = ["PreparedProvider"]
