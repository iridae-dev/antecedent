"""Little-endian float encodings used by declaration wires."""

from __future__ import annotations

import math
import struct

from .errors import CausalValueError


def float_bits(value: float) -> int:
    """Encode the IEEE-754 representation of a float, including non-finite values."""
    return int(struct.unpack("<Q", struct.pack("<d", float(value)))[0])


def finite_float_bits(value: float) -> int:
    """Encode a finite declared number, preserving the public refusal."""
    number = float(value)
    if not math.isfinite(number):
        raise CausalValueError("a declared number must be finite")
    return float_bits(number)
