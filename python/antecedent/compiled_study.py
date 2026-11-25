"""Save and open a compiled study in the CAUSAL container.

The container is format 0.6, kind ``compiled_study_v1``. ``program.invariant`` is
the writer output, byte for byte. ``program.catalog`` records each role's digest,
length, and placement. Inline bytes are ``program.body.<role>``. An external
placement keeps the caller's URI and takes its bytes from the resolver.

Retain the expected identity the writer returns separately from the container.
For a portable prepared study, opening with its required bodies re-quotes layer
5 and runs the saved estimator. A missing required body leaves the sealed claim
available with ``resumable`` false. A different invariant, a corrupt body, or a
claim from another program or data snapshot is refused.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ._native import open_compiled_study, write_compiled_study


@dataclass(frozen=True)
class StudyBody:
    """One catalog role stored beside the invariant."""

    role: str
    data: bytes
    uri: str | None = None
    required: bool = True
    summary_only: bool = False
    names: tuple[str, ...] = ()


@dataclass(frozen=True)
class WrittenStudy:
    """Container bytes and the program identity retained apart from them."""

    data: bytes
    expected_identity: str


@dataclass(frozen=True)
class UnresolvedBody:
    """A required role the resolver did not supply."""

    role: str
    uri: str | None


@dataclass(frozen=True)
class OpenedStudy:
    """A compiled study after acceptance."""

    resumable: bool
    unresolved: tuple[UnresolvedBody, ...]
    layer5_bytes: int | None
    estimate: float | None
    claim_intact: bool | None
    claim_estimate: float | None
    refit_refused: str | None
    callback_requires_caller: bool
    callback_live: bool
    invariant: bytes
    bodies: Mapping[str, bytes]


def write(
    invariant: bytes, bodies: Sequence[StudyBody], claim: bytes | None = None
) -> WrittenStudy:
    """Store ``invariant`` and ``bodies`` as a compiled study."""
    data, identity = write_compiled_study(
        bytes(invariant),
        [body.role for body in bodies],
        [bytes(body.data) for body in bodies],
        ["" if body.uri is None else body.uri for body in bodies],
        [body.required for body in bodies],
        [body.summary_only for body in bodies],
        [list(body.names) for body in bodies],
        None if claim is None else bytes(claim),
    )
    return WrittenStudy(data, identity)


def open_study(
    data: bytes,
    expected_identity: str,
    *,
    resolved: Mapping[str, bytes] | None = None,
    caller_supplied: bool = False,
) -> OpenedStudy:
    """Open ``data`` under ``expected_identity``.

    ``resolved`` maps an external URI to the bytes the caller supplies for it.
    Pass ``saved.expected_identity`` from the original write. A portable study
    with all required bodies re-executes its estimator. Read ``resumable``
    and ``unresolved`` before using ``estimate``.
    ``caller_supplied`` declares whether the caller has provided a live callback.
    """
    uris = []
    payloads = []
    for uri, payload in (resolved or {}).items():
        uris.append(uri)
        payloads.append(bytes(payload))
    status, invariant, roles, bodies = open_compiled_study(
        bytes(data),
        expected_identity,
        uris,
        payloads,
        caller_supplied,
    )
    loaded = json.loads(status)
    return OpenedStudy(
        resumable=loaded["resumable"],
        unresolved=tuple(
            UnresolvedBody(item["role"], item["uri"]) for item in loaded["unresolved"]
        ),
        layer5_bytes=loaded["layer5_bytes"],
        estimate=loaded["estimate"],
        claim_intact=loaded["claim_intact"],
        claim_estimate=loaded["claim_estimate"],
        refit_refused=loaded["refit_refused"],
        callback_requires_caller=loaded["callback_requires_caller"],
        callback_live=loaded["callback_live"],
        invariant=bytes(invariant),
        bodies={role: bytes(body) for role, body in zip(roles, bodies, strict=True)},
    )
