"""What every decision result shares: the choice, the verdict word, its standing and statement.

A decision is answered by one of three engines: an ordinary contract with one utility
(:class:`antecedent.decision.Decision`), a contract over several causal structures
(:class:`antecedent.decision_robust.RobustDecision`) or a contract whose trade-offs are
declared as ranges (:class:`antecedent.decision.TradeoffDecision`). Each keeps its own fields;
:class:`DecisionResult` is what they all answer the same way. The standing of a verdict is
computed natively and read from the reply, never derived here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from ._freeze import freeze
from ._scientific import ScientificResult

Standing = Literal[
    "best",
    "tied",
    "policy_choice",
    "depends",
    "undetermined",
    "no_admissible_action",
    "report_only",
    "insufficient_evidence",
    "sampled_estimate",
]


@dataclass(frozen=True, slots=True)
class CompiledReceipt:
    """What a decision compiled with its analysis ran: the bound and contract identities, the
    causal contracts it was bound to, the identity of every compiled plan and of the result each
    executed (``None`` where a plan reports none), and the measure and options the decision
    read."""

    bound_identity: str
    contract_identity: str
    causal: Any
    programs: tuple[str, ...]
    executed: tuple[str | None, ...]
    measure: Any
    options: Any
    mixture: Mapping[str, Any] | None = None

    @classmethod
    def _from_body(cls, body: Mapping[str, Any] | None) -> CompiledReceipt | None:
        wire = None if body is None else body.get("compiled")
        if not isinstance(wire, Mapping):
            return None
        mixture = wire.get("mixture")
        return cls(
            bound_identity=str(wire["bound_identity"]),
            contract_identity=str(wire["contract_identity"]),
            causal=freeze(wire.get("causal")),
            programs=tuple(str(p) for p in wire.get("programs") or ()),
            executed=tuple(None if e is None else str(e) for e in wire.get("executed") or ()),
            measure=freeze(wire.get("measure")),
            options=freeze(wire.get("options")),
            mixture=freeze(mixture),
        )


class DecisionResult(ScientificResult):
    """The reading every decision result shares.

    * :attr:`choice` is the action the verdict chooses, or ``None`` when it chooses none (a tie,
      a choice that depends on the trade-offs or structures, a report);
    * :attr:`kind` is the engine's own verdict word (``uniquely_optimal``,
      ``necessarily_best``, ``structurally_robust``, ...);
    * :attr:`standing` places that word in one vocabulary: ``best``, ``tied``,
      ``policy_choice`` (a declared policy chose), ``depends``, ``undetermined``,
      ``no_admissible_action``, ``report_only``, ``insufficient_evidence`` or
      ``sampled_estimate``;
    * :attr:`statement` says the verdict in plain words;
    * :attr:`receipt` is what a compiled decision ran, or ``None`` for a decision evaluated on
      a source already in hand.
    """

    @property
    def choice(self) -> str | None:
        """The chosen action, or ``None`` when the verdict chooses none."""
        raise NotImplementedError

    @property
    def kind(self) -> str:
        """The verdict word of the engine that answered."""
        raise NotImplementedError

    @property
    def standing(self) -> Standing:
        """The verdict in the shared vocabulary."""
        raise NotImplementedError

    @property
    def statement(self) -> str:
        """The verdict in plain words."""
        raise NotImplementedError

    @property
    def contract_identity(self) -> str:
        """Canonical identity of the contract the result answers."""
        raise NotImplementedError

    @property
    def contract(self) -> Any:
        """The contract as declared."""
        raise NotImplementedError

    @property
    def effective_tie_tolerance(self) -> float | None:
        """The tolerance within which criterion values counted as equal (``0.0`` for exact
        equality)."""
        raise NotImplementedError

    @property
    def receipt(self) -> CompiledReceipt | None:
        """What a compiled decision ran, or ``None``."""
        return CompiledReceipt._from_body(getattr(self, "_body", None))

    def explain(self) -> str:
        """Why this action, or why none."""
        raise NotImplementedError


def _standing(body: Mapping[str, Any]) -> Standing:
    """The natively computed standing a decision reply carries."""
    return body["standing"]


__all__ = ["CompiledReceipt", "DecisionResult", "Standing"]
