"""Compile and replay a marginal or conditional interventional query on an oracle PAG."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from . import _native
from ._artifact_protocol import (
    ArtifactExpectation,
    ExportedArtifact,
    artifact_consumer,
    artifact_export,
    container_bytes,
)
from ._refusal import raise_refusal
from ._scientific import ScientificResult
from ._unit_support import read_unit_support
from ._validate import count
from .errors import CausalTypeError, CausalValueError, StructuredRefusal
from .execution import Execution, native_controls, resolve_execution
from .outcomes import outcome_operation

_DEFAULT_STEPS = 1_048_576

__all__ = [
    "CompiledPagIdentification",
    "DirectPagResult",
    "PagExactDistribution",
    "PagIdentificationRefusal",
    "compile_pag_identification",
    "consume_pag_identification",
]


class PagIdentificationRefusal(StructuredRefusal):
    """A typed input, execution, replay or support stop in direct PAG identification."""


def _checked(pair: tuple[Any, str | None]) -> Any:
    result, refusal = pair
    raise_refusal(refusal, PagIdentificationRefusal)
    if result is None:  # pragma: no cover - native contract
        raise CausalValueError("direct PAG route returned neither result nor refusal")
    return result


def _controls(policy: Execution) -> dict[str, Any]:
    return native_controls(
        policy, seed="seed", threads="threads", cancel="cancel", memory="memory_bytes"
    )


@dataclass(frozen=True, slots=True)
class PagExactDistribution:
    """An exact law on the requested outcomes for one specified intervention and observation."""

    outcomes: tuple[str, ...]
    atoms: tuple[tuple[int, ...], ...]
    probabilities: tuple[float, ...]


class DirectPagResult(ScientificResult):
    """An observational functional or a graphical nonidentification witness."""

    def __init__(self, handle: _native.DirectPagResultHandle, program_identity: str) -> None:
        self._handle = handle
        self._program_identity = program_identity

    @property
    def identity(self) -> str:
        """Identity of the graph, original query and reconstructed result."""
        return self._handle.identity

    @property
    def program_identity(self) -> str:
        """Identity of the causal program independently reconstructed during replay."""
        return self._program_identity

    @property
    def work(self) -> int:
        """Metered graph and path operations performed by direct identification."""
        return self._handle.work

    @property
    def proof(self) -> dict[str, Any]:
        """The interned functional expression or obstruction, indexed by graph coordinates."""
        return dict(json.loads(self._handle.proof_json()))

    @property
    def status(
        self,
    ) -> Literal["identified", "marginal_obstruction", "conditional_obstruction", "resource_stop"]:
        """The scientific conclusion or an explicitly retained deterministic resource stop."""
        status: Literal[
            "identified", "marginal_obstruction", "conditional_obstruction", "resource_stop"
        ] = self.proof["status"]
        return status

    @artifact_export
    def export(self) -> ExportedArtifact[DirectPagResult]:
        """Export the graph, query and result for independent symbolic replay."""
        kind = "pag_identification_v1"
        return ExportedArtifact(
            self._handle.export(),
            kind,
            ArtifactExpectation(
                kind, {"expected_program": self.program_identity, "expected_result": self.identity}
            ),
        )

    @outcome_operation
    def evaluate(
        self,
        levels: Sequence[Sequence[int]],
        probabilities: Sequence[float],
        at: Mapping[str, int],
        *,
        operations: int = 10_000_000,
        execution: Execution | None = None,
    ) -> PagExactDistribution:
        """Evaluate the functional on an exact positive joint law compatible with the PAG.

        Supply integer level labels in retained PAG node order and probabilities
        in row-major order, with the last axis varying fastest. The assignment
        names every originally intervened and conditioned variable.
        """
        policy = resolve_execution(execution)
        data = _checked(
            policy._call(
                self._handle.evaluate,
                [list(axis) for axis in levels],
                list(probabilities),
                list(at.items()),
                operations=count(operations, "operations", minimum=1),
                **_controls(policy),
            )
        )
        wire = json.loads(data)
        return PagExactDistribution(
            tuple(wire["outcomes"]),
            tuple(tuple(atom) for atom in wire["atoms"]),
            tuple(wire["probabilities"]),
        )


class CompiledPagIdentification:
    """The causal compiler's direct IDP or CIDP plan for a declared PAG."""

    def __init__(self, handle: _native.DirectPagPlanHandle) -> None:
        self._handle = handle

    @property
    def program_identity(self) -> str:
        """Identity binding the graph, disjoint query sets and observational population."""
        return self._handle.program_identity

    def plan_view(self) -> dict[str, Any]:
        """Inspect the compiled layers before running identification."""
        return dict(json.loads(self._handle.plan_view_json()))

    def unit_support(self) -> dict[str, Any] | None:
        """Read the compiled route's support verdict."""
        return read_unit_support(self._handle)

    @outcome_operation
    def execute(
        self,
        *,
        steps: int = _DEFAULT_STEPS,
        retain_resource_stop: bool = False,
        execution: Execution | None = None,
    ) -> DirectPagResult:
        """Run bounded identification, optionally retaining a reproducible limit stop."""
        policy = resolve_execution(execution)
        result = _checked(
            policy._call(
                self._handle.execute,
                steps=count(steps, "steps", minimum=0),
                retain_resource_stop=retain_resource_stop,
                **_controls(policy),
            )
        )
        return DirectPagResult(result, self.program_identity)


@outcome_operation
def compile_pag_identification(
    pag: _native.Pag,
    *,
    outcomes: Sequence[str],
    actions: Sequence[str] = (),
    given: Sequence[str] = (),
    population: str = "observed",
) -> CompiledPagIdentification:
    """Compile a static query from a declared, valid oracle PAG without selection.

    Actions are the hard intervention coordinates, and given names the
    conditioning coordinates. A nonempty conditioning set selects conditional CIDP.
    Identification assumes the Markov-equivalence graph and positive observational law.
    """
    if not isinstance(pag, _native.Pag):
        raise CausalTypeError("pag must be a Pag")
    return CompiledPagIdentification(
        _checked(
            _native.direct_pag_compile(
                pag,
                list(actions),
                list(outcomes),
                list(given),
                population,
            )
        )
    )


@outcome_operation
@artifact_consumer(
    proof="Repeat direct identification on the original graph, query and bounds and verify its scientific conclusion.",
    replayable=True,
    identity=("expected_program", "expected_result"),
)
def consume_pag_identification(
    artifact: bytes,
    *,
    expected_result: str,
    expected_program: str,
    steps: int = _DEFAULT_STEPS,
    execution: Execution | None = None,
) -> DirectPagResult:
    """Replay identification under independently retained result and program identities."""
    policy = resolve_execution(execution)
    result = _checked(
        policy._call(
            _native.direct_pag_consume,
            container_bytes(artifact),
            expected_result,
            expected_program,
            steps=count(steps, "steps", minimum=0),
            **_controls(policy),
        )
    )
    return DirectPagResult(result, expected_program)
