"""Internal transport lifecycle adapter; scientific authority remains native."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .errors import CausalUnsupportedError, CausalValueError


@dataclass(frozen=True, slots=True)
class TransportLifecycle:
    wrap: Callable[[Any, Any], Any]
    payload: Callable[[Any], Any]
    compact_payload: bool = False

    def execute(
        self,
        native: Any,
        data: Any,
        *,
        seed: int | None,
        threads: int | None,
        controls: dict[str, Any],
    ) -> Any:
        if (
            seed is not None
            or threads is not None
            or any(controls.get(name) is not None for name in ("on_progress", "on_stage"))
        ):
            raise CausalUnsupportedError(
                "Transport retains inference settings; only cancellation can change per execution",
                reason_code="option_not_applicable",
            )
        if data is None:
            execute = native.estimate_summary if self.compact_payload else native.estimate
            raw = execute(cancel=controls.get("cancel"))
        else:
            raise CausalValueError("Use rebind for an explicit transport snapshot change")
        return self.wrap(native, raw)


def refuse_inapplicable_controls(execution: Any, *, threads: bool) -> None:
    """Refuse the physical controls a transport route cannot honour, before any work.

    Cancellation and the memory limit reach the native transport engines; ``on_progress``
    reports the start and end of each estimate. Transport engines stream no stages, and only
    the typed provider families run a worker pool, so ``on_stage`` and (elsewhere) ``threads``
    are typed refusals instead of being dropped.
    """
    if execution.on_stage is not None:
        raise CausalUnsupportedError(
            "Execution.on_stage does not apply to transport queries: transport engines stream "
            "no stages (use on_progress, which reports the start and end of each estimate)",
            reason_code="option_not_applicable",
        )
    if execution.threads is not None and not threads:
        raise CausalUnsupportedError(
            "Execution.threads does not apply to this transport query: only the typed exact, "
            "statistical and response-grid transport queries run a worker pool",
            reason_code="option_not_applicable",
        )


def transport_lifecycle(kind: str) -> TransportLifecycle | None:
    from .transport._impl import (
        _exact_distribution,
        _learned_trial,
        _response_grid,
        _statistical_distribution,
    )

    adapters = {
        "learned_trial": TransportLifecycle(
            _learned_trial, lambda data: data, compact_payload=True
        ),
        "exact_transport": TransportLifecycle(_exact_distribution, lambda data: data.laws),
        "statistical_transport": TransportLifecycle(_statistical_distribution, lambda data: data),
        "transport_grid": TransportLifecycle(_response_grid, lambda data: data),
        "z_transport": TransportLifecycle(lambda _native, payload: payload, lambda data: data),
    }
    return adapters.get(kind)
