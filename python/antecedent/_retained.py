"""Admit an artifact only as the artifact its consumer retained.

A consumer retains ``artifact_identity(data)`` when the producer hands it the bytes, through its
own store or registry, and passes it back as ``retained_identity=`` when it loads them. Every
artifact loader calls :func:`admit` before it reads the bytes. The comparison is the native
``bind_retained_artifact`` (the same Rust ``bind`` the Rust ``*_expecting`` loaders go through),
never a Python string comparison.
"""

from __future__ import annotations

from typing import Any

from . import _deprecation
from . import _native as _native_module
from .errors import CausalTypeError


class _Admitted(str):
    """A caller inside the package has already admitted the bytes (or produced them itself)."""

    __slots__ = ()


ADMITTED: str = _Admitted("admitted-by-caller")
"""Passed as ``retained_identity`` by an internal caller that already ran :func:`admit`: the
inner loader neither binds again nor warns."""


def artifact_identity(data: Any) -> str:
    """The identity to retain for artifact ``data``: 64 lowercase hex characters.

    Compute it when the artifact is produced or first stored, never from bytes about to be
    loaded, and pass it to a loader as ``retained_identity=``.
    """
    return str(_native_module.retained_artifact_identity(_container_bytes(data)))


def admit(
    data: Any,
    retained_identity: str | None,
    name: str,
    replacement: str,
    *,
    bound: bool = False,
) -> None:
    """Bind ``data`` to ``retained_identity`` natively, or warn that it is admitted unbound.

    ``bound`` is ``True`` when the call already binds an identity the consumer retained (a
    required or supplied semantic expectation); then a missing ``retained_identity`` is not
    warned about.
    """
    if retained_identity is ADMITTED:
        return
    if retained_identity is not None:
        if not isinstance(retained_identity, str):
            raise CausalTypeError("retained_identity must be a 64-character hex string")
        _native_module.bind_retained_artifact(_container_bytes(data), retained_identity)
        return
    if not bound:
        _deprecation.unbound_artifact_deprecated(name, replacement)


def _container_bytes(data: Any) -> bytes:
    if isinstance(data, (bytes, bytearray, memoryview)):
        return bytes(data)
    if isinstance(data, str):
        # A text artifact (a JSON posterior) is identified by its UTF-8 bytes.
        return data.encode("utf-8")
    inner = getattr(data, "data", None)
    if isinstance(inner, (bytes, bytearray, memoryview)):
        return bytes(inner)
    if hasattr(data, "__bytes__"):
        return bytes(data)
    raise CausalTypeError("an artifact to admit must be bytes or an exported artifact")
