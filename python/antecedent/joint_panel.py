"""Retained multichannel panel execution through the Rust causal compiler.

Features use :mod:`antecedent.intervention_programs` expressions. Their variable
reads explicitly mean lag-weighted, Michaelis-Menten saturated channel values.
Python declares inputs; Rust checks the source/history/target, fits the model,
retains shared draws, checks support, and evaluates every world.

A fitted curve does not identify an intervention. Missing design premises remain
visible in ``inspect()`` and every result. Historical artifacts replay numerical
results without granting current scientific authority.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from . import _native
from .errors import CausalValueError
from .intervention_programs import Expr

__all__ = ["PanelResult", "PreparedPanel", "consume", "prepare"]


def _json(value: Any) -> str:
    try:
        return json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise CausalValueError(f"invalid joint panel declaration: {error}") from error


@dataclass(frozen=True, slots=True)
class PanelResult:
    """One native shared-draw execution, retaining its compiled scientific scope."""

    _handle: Any

    @property
    def execution_facts(self) -> Any:
        """Engine-owned resource/operation measurements for this execution."""
        return self._handle.execution_facts

    def report(self) -> dict[str, Any]:
        """Actual means, failed draw slots, diagnostics, support and fit count."""
        return json.loads(self._handle.report())

    def export(self, **execution: Any) -> bytes:
        """Encode the actual frozen compiler product; performs no second fit."""
        return bytes(self._handle.export(**execution))


@dataclass(frozen=True, slots=True)
class PreparedPanel:
    """Native checked preparation with synchronized reuse of compatible fits."""

    _handle: Any

    def inspect(self) -> dict[str, Any]:
        """Inspect source/design obligations without fitting or simulation."""
        return json.loads(self._handle.inspect())

    def estimate(self, **execution: Any) -> PanelResult:
        """Run the retained Rust program with caller seed/thread/memory/cancel policy."""
        return PanelResult(self._handle.estimate(**execution))

    def retarget(self, worlds: Sequence[Mapping[str, Any]], **execution: Any) -> PreparedPanel:
        """Check replacement worlds against the same source and reuse compatible fits."""
        return PreparedPanel(self._handle.retarget(_json(list(worlds)), **execution))


def prepare(
    groups: Mapping[int, Mapping[str, Sequence[float | None]]],
    *,
    channels: Sequence[str],
    outcome: str,
    confounder: str,
    season: str,
    seasons: int,
    interval_ns: int,
    worlds: Sequence[Mapping[str, Any]],
    interactions: Mapping[str, Expr],
    options: Mapping[str, Any],
    lag_order: int = 2,
    temporal: Mapping[str, Any] | None = None,
    functional: Mapping[str, Any] | None = None,
    assignment: Mapping[str, Any] | None = None,
    exogenous_context: bool = False,
    **execution: Any,
) -> PreparedPanel:
    """Compile an explicitly declared panel through ``Study.prepare_joint_panel``.

    ``groups`` maps integer group IDs to complete named source columns. With lag
    order two, its first two rows are actual exposure prehistory; outcome and
    confounder may be ``None`` in those rows. Remaining rows must be complete.

    Every world supplies ``id``, ``reference``, keyed ``channels`` schedules and
    ``scenarios`` with ``group``, keyed two-value ``history``, ``confounder`` and
    ``season`` future arrays. No future context is extrapolated implicitly.

    ``options`` is the Rust ``PanelFitOptions`` declaration (optimizer, inference,
    initial); ``temporal`` is the same Rust change/smoothing specification.
    ``functional`` optionally supplies finite ``terms`` (world-index, coefficient)
    and a ``reference`` index. ``assignment`` uses the existing randomized-effect
    wire contract, checked against whole source-group programs in Rust.

    ``seed``, ``threads``, ``max_bytes`` and ``cancel`` are caller execution policy.
    Neither ``exogenous_context`` nor a finite fit establishes a causal license.
    """
    names = list(dict.fromkeys([*channels, outcome, confounder, season]))
    source = []
    for group, columns in groups.items():
        if set(columns) != set(names):
            raise CausalValueError("each group must supply exactly the declared source columns")
        source.append({"id": group, "columns": [list(columns[name]) for name in names]})
    features = []
    for name, expression in interactions.items():
        if not isinstance(expression, Expr):
            raise CausalValueError("panel features require intervention_programs.Expr")
        features.append({"id": name, "expression": expression.node})
    declaration = {
        "names": names,
        "channels": list(channels),
        "outcome": outcome,
        "confounder": confounder,
        "season": season,
        "seasons": seasons,
        "lag_order": lag_order,
        "interval_ns": interval_ns,
        "groups": source,
        "worlds": list(worlds),
        "interactions": features,
        "options": dict(options),
        "temporal": temporal,
        "functional": functional,
        "assignment": assignment,
        "exogenous_context": exogenous_context,
    }
    return PreparedPanel(_native.prepare_joint_panel(_json(declaration), **execution))


def consume(artifact: bytes, **execution: Any) -> dict[str, Any]:
    """Replay the frozen Rust program in an independent historical consumer."""
    return json.loads(_native.consume_joint_panel(artifact, **execution))
