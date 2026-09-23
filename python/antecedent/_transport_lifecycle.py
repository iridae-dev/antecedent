"""Internal transport lifecycle adapter; scientific authority remains native."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class TransportLifecycle:
    wrap: Callable[[Any, Any], Any]
    payload: Callable[[Any], Any]

    def execute(
        self,
        native: Any,
        data: Any,
        *,
        refresh: bool,
        seed: int | None,
        threads: int | None,
        controls: dict[str, Any],
    ) -> Any:
        if (
            seed is not None
            or threads is not None
            or any(controls.get(name) is not None for name in ("on_progress", "on_stage"))
        ):
            raise ValueError(
                "Transport retains inference settings; only cancellation can change per execution"
            )
        if refresh:
            raw = native.refresh(self.payload(data), cancel=controls.get("cancel"))
        elif data is None:
            raw = native.estimate(cancel=controls.get("cancel"))
        else:
            raise ValueError(
                "Use replace_snapshot or refresh for an explicit transport snapshot change"
            )
        return self.wrap(native, raw)


def transport_lifecycle(kind: str) -> TransportLifecycle | None:
    from .transport._impl import (
        _exact_distribution,
        _learned_trial,
        _response_grid,
        _statistical_distribution,
    )

    adapters = {
        "learned_trial": TransportLifecycle(_learned_trial, lambda data: data),
        "exact_transport": TransportLifecycle(_exact_distribution, lambda data: data.laws),
        "statistical_transport": TransportLifecycle(_statistical_distribution, lambda data: data),
        "transport_grid": TransportLifecycle(_response_grid, lambda data: data),
    }
    return adapters.get(kind)
