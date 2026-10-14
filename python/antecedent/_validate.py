"""Argument validators shared by the public Python entry points (private leaf module).

Every helper takes ``(value, what)`` and returns the checked value. The defaults are the
library-wide policy: ``bool`` (and ``np.bool_``) and ``str`` are never a number, numpy integer
and floating scalars are numbers, and a number too large for a float is a
:class:`CausalValueError` rather than a bare ``OverflowError``. A module that already published a
message or an exception class keeps it through the keyword parameters, so the text callers match
on does not change.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from decimal import Decimal
from numbers import Complex, Real
from typing import Any

import numpy as np

from .errors import CausalTypeError, CausalValueError, StructuredRefusal

__all__ = ["choice", "count", "names", "non_finite_refusal", "real", "text"]

_REAL_TYPES: tuple[type, ...] = (int, float, np.floating, np.integer)
_INT_TYPES: tuple[type, ...] = (int, float, np.integer, np.floating, Real, Decimal)


def _message(template: str, what: str, value: object) -> str:
    # Callers choose among these fields per message; unused fields are intentional.
    # format_map preserves normal str.format field/conversion behavior without
    # expanding a dynamic template into a fixed set of keyword arguments.
    return template.format_map({"what": what, "got": type(value).__name__, "value": value})


def real(
    value: object,
    what: str,
    *,
    finite: bool = False,
    types: tuple[type, ...] = _REAL_TYPES,
    type_error: type[CausalTypeError] | type[CausalValueError] = CausalTypeError,
    type_message: str = "{what} must be a real number",
    overflow_message: str = "{what} is too large for a real number",
    non_finite: Callable[[str], Exception] | None = None,
    reason_code: str | None = None,
) -> float:
    """``value`` as a float, refusing a bool, a string or any other non-real.

    ``types=(object,)`` accepts anything ``float()`` converts except a bool, a string or a complex
    number.
    ``type_message`` and ``overflow_message`` are ``str.format`` templates over ``what``, ``got``
    (the type name) and ``value``. With ``finite`` a NaN or infinity raises the exception built by
    ``non_finite(what)`` (default: :class:`CausalValueError` ``"<what> must be a finite number"``).
    ``reason_code`` is attached to the default type, overflow and finiteness errors.
    """
    if (
        isinstance(value, bool | np.bool_ | str | bytes)
        or (isinstance(value, Complex) and not isinstance(value, Real))
        or not isinstance(value, types)
    ):
        raise type_error(_message(type_message, what, value), reason_code=reason_code)
    try:
        number = float(value)  # type: ignore[arg-type]
    except OverflowError as error:
        raise CausalValueError(
            _message(overflow_message, what, value), reason_code=reason_code
        ) from error
    except (TypeError, ValueError) as error:
        # Only reachable with a permissive ``types`` (for example ``(object,)``).
        raise type_error(_message(type_message, what, value), reason_code=reason_code) from error
    if finite and not math.isfinite(number):
        if non_finite is not None:
            raise non_finite(what)
        raise CausalValueError(f"{what} must be a finite number", reason_code=reason_code)
    return number


def non_finite_refusal(
    refusal: type[StructuredRefusal],
    *,
    stage: str,
    detail: str,
    offending: str | None = None,
) -> Callable[[str], StructuredRefusal]:
    """The ``non_finite`` callback of :func:`real` for a module that refuses with ``refusal``.

    The refusal it builds is the native ``invalid_argument`` shape, at ``stage`` with ``detail``,
    naming ``offending`` (default: the checked argument) as the culprit. A module that refuses a
    non-finite number the same way everywhere binds it once with
    ``partial(real, finite=True, non_finite=non_finite_refusal(...))``.
    """

    def build(what: str) -> StructuredRefusal:
        return refusal(
            {
                "code": "invalid_argument",
                "stage": stage,
                "detail": detail,
                "offending": offending or what,
                "message": f"{what} is not finite",
            }
        )

    return build


def count(
    value: object,
    what: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
    types: tuple[type, ...] = _INT_TYPES,
    type_error: type[CausalTypeError] | type[CausalValueError] = CausalTypeError,
    type_message: str = "{what} must be an integer",
    range_message: str | None = None,
    reason_code: str | None = None,
) -> int:
    """``value`` as an int, refusing a bool or a non-integer, within ``[minimum, maximum]``.

    ``range_message`` is a ``str.format`` template over ``what``, ``value``, ``minimum`` and
    ``maximum``; the default names whichever bounds are set.
    """
    if isinstance(value, bool | np.bool_) or not isinstance(value, types):
        raise type_error(_message(type_message, what, value), reason_code=reason_code)
    try:
        number = int(value)  # type: ignore[call-overload]
        integral = value == number
    except (ValueError, TypeError, OverflowError):
        integral = False
    if not integral:
        raise type_error(_message(type_message, what, value), reason_code=reason_code)
    if (minimum is not None and number < minimum) or (maximum is not None and number > maximum):
        if range_message is None:
            if maximum is None:
                range_message = "{what} must be at least {minimum}, got {value}"
            elif minimum is None:
                range_message = "{what} must be at most {maximum}, got {value}"
            else:
                range_message = "{what} must lie in {minimum}..={maximum}"
        raise CausalValueError(
            range_message.format(what=what, value=number, minimum=minimum, maximum=maximum),
            reason_code=reason_code,
        )
    return number


def text(
    value: object,
    what: str,
    *,
    non_empty: bool = False,
    type_message: str = "{what} must be a string",
    empty_message: str = "{what} must be non-empty",
) -> str:
    """``value`` as a string; with ``non_empty``, one that is not blank."""
    if not isinstance(value, str):
        raise CausalTypeError(_message(type_message, what, value))
    if non_empty and not value.strip():
        raise CausalValueError(_message(empty_message, what, value))
    return value


def names(
    values: Sequence[str],
    what: str,
    *,
    allow_empty: bool = True,
    sequence_message: str | None = None,
    entry_error: type[CausalTypeError] | type[CausalValueError] = CausalTypeError,
    entry_message: str = "{what} must be non-empty strings",
    duplicate_message: str = "{what} must be distinct",
    empty_message: str = "{what} must not be empty",
    as_list: bool = False,
) -> Any:
    """``values`` as a tuple of distinct, non-blank strings.

    A single string is never a sequence of names. ``sequence_message`` (when given) also refuses
    any non-:class:`~collections.abc.Sequence` input with that message. The result is a tuple, or a
    list with ``as_list``.
    """
    if isinstance(values, str) or (
        sequence_message is not None and not isinstance(values, Sequence)
    ):
        raise CausalTypeError(
            (sequence_message or "{what} must be a sequence of names, not a single string").format(
                what=what
            )
        )
    out = tuple(values)
    if any(not isinstance(v, str) or not v.strip() for v in out):
        raise entry_error(entry_message.format(what=what))
    if len(set(out)) != len(out):
        raise CausalValueError(duplicate_message.format(what=what))
    if not allow_empty and not out:
        raise CausalValueError(empty_message.format(what=what))
    return list(out) if as_list else out


def choice(value: str, what: str, allowed: Sequence[str]) -> str:
    """``value`` when it is one of ``allowed``."""
    if value not in allowed:
        raise CausalValueError(f"{what} is one of {', '.join(allowed)}; got {value!r}")
    return value
