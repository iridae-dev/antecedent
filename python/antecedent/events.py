"""Compile event forecasts and compose them with stochastic worlds and decisions."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from . import _native, _reason_details
from .execution import Execution, native_controls

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .extensibility import ProviderVerificationReceipt


class EventRefusal(ValueError):
    """A scientific, validation or resource refusal specific to event processes."""

    def __init__(self, detail: str, message: str) -> None:
        self.detail = detail
        super().__init__(message)


def _call(operation: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        return operation(*args, **kwargs)
    except ValueError as error:
        message = str(error)
        detail = message.split(":", 1)[0]
        if detail in {
            _reason_details.events.INVALID_REQUEST,
            _reason_details.events.UNSTABLE,
            _reason_details.events.FIT_FAILED,
            _reason_details.events.BUDGET,
            _reason_details.events.NUMERICAL,
            _reason_details.events.UNVERIFIED_SOURCE,
            _reason_details.events.INCOMPLETE_LAW,
            _reason_details.events.INTERVAL_UNMEASURED,
        }:
            raise EventRefusal(detail, message) from error
        raise


def _wire(value: Mapping[str, Any]) -> str:
    return json.dumps(dict(value), allow_nan=False, separators=(",", ":"))


def _controls(execution: Execution | None) -> dict[str, Any]:
    return native_controls(execution, memory="memory_bytes", cancel="cancel")


def provider_request_id(request: Mapping[str, Any]) -> str:
    """Return the request identity an external execution receipt must cover."""
    return _call(_native.events_provider_request_id, _wire(request))


class PreparedEventProgram:
    """An immutable compiler plan. Inspecting it performs no fitting or simulation."""

    def __init__(self, native: _native.NativePreparedEventProgram) -> None:
        self._native = native

    def inspect(self) -> dict[str, Any]:
        return json.loads(self._native.inspect())

    def unit_support(self) -> dict[str, Any] | None:
        """The support decision of the unit this plan's route runs under: ``unit``, ``status``,
        ``supported`` and ``missing_evidence``. ``None`` when no unit is bound to the route."""
        wire = self._native.unit_support()
        return None if wire is None else json.loads(wire)

    def execute(self, *, execution: Execution | None = None) -> EventResult:
        """Run the plan under ``execution``'s memory limit and cancellation."""
        return EventResult(_call(self._native.execute, **_controls(execution)))

    def count_interval(self, kind: int, *, execution: Execution | None = None) -> dict[str, Any]:
        """Publish a count interval of a compiled forecast under ``execution``."""
        return json.loads(_call(self._native.count_interval, kind, **_controls(execution)))


class EventResult:
    """Retained law or operation result with its compiler and source identity."""

    def __init__(self, native: _native.NativeEventResult) -> None:
        self._native = native

    def report(self) -> dict[str, Any]:
        return json.loads(self._native.report())

    def paths(self) -> list[list[dict[str, Any]]]:
        """Return event paths in the order used for joint world pairing."""
        return json.loads(self._native.paths())

    def distribution(self) -> bytes:
        """Export the existing joint distribution artifact for a world result."""
        return self._native.distribution()


def prepare(
    request: Mapping[str, Any],
    *,
    source: EventResult | None = None,
    receipt: ProviderVerificationReceipt | None = None,
    execution: Execution | None = None,
) -> PreparedEventProgram:
    """Compile a forecast, likelihood, worlds, assessment, update or future study.

    For world composition, pass a retained event result as `source`. Decision
    contracts accept an existing decision.Contract or its native JSON declaration.
    ``execution`` supplies the memory limit and cancellation of the compile; the
    random streams are the seeds the request declares.
    """
    declaration = dict(request)
    if declaration.get("contract") is not None:
        from .decision import Contract

        if isinstance(declaration["contract"], Contract):
            declaration["contract"] = _wire(declaration["contract"]._wire())
    native = _call(
        _native.events_prepare,
        _wire(declaration),
        None if source is None else source._native,
        None if receipt is None else receipt._native_value,
        **_controls(execution),
    )
    return PreparedEventProgram(native)


def replay(
    request: Mapping[str, Any],
    expected_program: str,
    *,
    source: EventResult | None = None,
    receipt: ProviderVerificationReceipt | None = None,
    execution: Execution | None = None,
) -> EventResult:
    """Recompile and execute a forecast or world request under an expected identity."""
    plan = prepare(request, source=source, receipt=receipt, execution=execution)
    if plan._native.replay_identity() != expected_program:
        detail = _reason_details.events.INVALID_REQUEST
        raise EventRefusal(detail, f"{detail}: replay program identity changed")
    return plan.execute(execution=execution)
