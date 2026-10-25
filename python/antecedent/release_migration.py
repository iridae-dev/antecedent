"""Consume artifacts written by a previous release.

A previous-release artifact keeps its row ids, the score-row digest it retained, a carried
scientific-target digest and a recalculation receipt. Reading it never widens a stale digest::

    from antecedent import release_migration as rm

    blob = rm.previous_release_artifact(
        row_ids=[7, 11, 13],
        score_row_digest=rm.score_row_digest([7, 11, 13]),
        target_digest=rm.target_digest(0.0),
        receipt=receipt.export(),
    )
    adopted = rm.consume_previous_release_artifact(blob, intervention_level=0.0)

A four-byte score-row digest, a target digest that is not the declared target's identity, and a
fresh-process receipt that claims reuse of a derived stage are each refused
(:class:`~antecedent.errors.CausalUnsupportedError`) with the remedy in the message. A route
artifact is carried record by record onto the current methods, and every record gets its own
outcome or its own break, so one removed route does not hide the others.

The declared target is the mean of variable 1 when variable 0 is set to ``intervention_level``
for the observed population at the decision time.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from ._native import consume_previous_release as _consume_release
from ._native import consume_previous_routes as _consume_routes
from ._native import current_score_row_digest_bytes as _current_digest
from ._native import declared_target_digest as _target_digest
from ._native import narrow_score_row_digest_bytes as _narrow_digest

__all__ = [
    "AdoptedRelease",
    "CurrentMethod",
    "MethodAssignment",
    "RouteRecordMigration",
    "consume_previous_release_artifact",
    "consume_previous_route_records",
    "narrow_score_row_digest",
    "previous_release_artifact",
    "score_row_digest",
    "target_digest",
]


@dataclass(frozen=True)
class AdoptedRelease:
    """What a previous-release artifact adopted under the current contracts."""

    score_row_digest: bytes
    target_digest: bytes
    boundary: str
    """``in_process`` or ``fresh_process``, as the receipt declares it."""
    stage_status: tuple[tuple[str, str], ...]
    """Stage label and canonical status, in receipt order."""

    def status(self, stage: str) -> str:
        """The status text of one stage."""
        for name, text in self.stage_status:
            if name == stage:
                return text
        raise KeyError(stage)


@dataclass(frozen=True)
class CurrentMethod:
    """One current method record a previous route may be carried onto."""

    id: str
    route: str
    estimator: str
    design: str
    inference: str
    status: str

    def _row(self) -> tuple[str, str, str, str, str, str]:
        return (self.id, self.route, self.estimator, self.design, self.inference, self.status)


@dataclass(frozen=True)
class MethodAssignment:
    """A current method that carries a previous route."""

    method: str
    estimator: str
    status: str
    axes: tuple[str, ...]
    narrowed: bool
    """The previous route was licensed and this method is not."""


@dataclass(frozen=True)
class RouteRecordMigration:
    """What carrying one previous route record onto the current methods did.

    ``outcome`` is ``one_to_one``, ``split``, ``retired``, ``refusal_kept`` or ``break``. A break
    names its ``kind`` and the ``message`` carries the remedy.
    """

    id: str
    outcome: str
    methods: tuple[MethodAssignment, ...] = ()
    unserved_axes: tuple[str, ...] = ()
    kind: str | None = None
    unlicensed_methods: tuple[str, ...] = ()
    message: str | None = None

    @property
    def is_break(self) -> bool:
        return self.outcome == "break"


def score_row_digest(row_ids: Iterable[int]) -> bytes:
    """The current score-row digest of eight-byte row ids."""
    return bytes(_current_digest([int(r) for r in row_ids]))


def narrow_score_row_digest(row_ids: Iterable[int]) -> bytes:
    """The four-byte-id digest a previous release retained; it is not the current digest."""
    return bytes(_narrow_digest([int(r) for r in row_ids]))


def target_digest(intervention_level: float) -> bytes:
    """The identity of the declared target, which a carried target digest must equal."""
    return bytes(_target_digest(float(intervention_level)))


def previous_release_artifact(
    *,
    row_ids: Sequence[int],
    score_row_digest: bytes,
    target_digest: bytes,
    receipt: bytes,
) -> bytes:
    """Write the previous-release object from its four parts."""
    return json.dumps(
        {
            "row_ids": [int(r) for r in row_ids],
            "retained_score_digest": list(bytes(score_row_digest)),
            "carried_target_digest": list(bytes(target_digest)),
            "receipt": list(bytes(receipt)),
        },
        separators=(",", ":"),
    ).encode()


def consume_previous_release_artifact(
    artifact: bytes, *, intervention_level: float
) -> AdoptedRelease:
    """Adopt a previous-release artifact, or refuse it with the remedy."""
    score, target, boundary, rows = _consume_release(bytes(artifact), float(intervention_level))
    return AdoptedRelease(
        score_row_digest=bytes(score),
        target_digest=bytes(target),
        boundary=boundary,
        stage_status=tuple((str(stage), str(text)) for stage, text in rows),
    )


def _assignment(row: Mapping[str, object]) -> MethodAssignment:
    return MethodAssignment(
        method=str(row["method"]),
        estimator=str(row["estimator"]),
        status=str(row["status"]),
        axes=tuple(row["axes"]),  # type: ignore[arg-type]
        narrowed=bool(row["narrowed"]),
    )


def consume_previous_route_records(
    artifact: bytes, methods: Iterable[CurrentMethod]
) -> tuple[RouteRecordMigration, ...]:
    """Carry every record of a previous-release route artifact onto the current methods."""
    rows = json.loads(_consume_routes(bytes(artifact), [m._row() for m in methods]))
    return tuple(
        RouteRecordMigration(
            id=row["id"],
            outcome=row["outcome"],
            methods=tuple(_assignment(m) for m in row.get("methods", ()))
            if row["outcome"] != "break"
            else (),
            unserved_axes=tuple(row.get("unserved_axes", ())),
            kind=row.get("kind"),
            unlicensed_methods=tuple(row.get("unlicensed_methods", ())),
            message=row.get("message"),
        )
        for row in rows
    )
