"""Compiler-retained declared-shift MNAR candidate with a coverage-licensed interval."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from ._refusal import raise_refusal
from .errors import CausalValueError, StructuredRefusal
from .mnar_sensitivity import _outcome, _real, _treatment

__all__ = ["MnarFixedShiftRefusal", "PreparedMnarFixedShift", "prepare_fixed_shift"]


class MnarFixedShiftRefusal(StructuredRefusal):
    """The distinct compiled fixed-shift candidate's native refusal."""


@dataclass(frozen=True, slots=True)
class PreparedMnarFixedShift:
    """Retained native compiler operation under explicitly declared assumptions.

    Its measured interval is released only when an attesting coverage record covers
    this construction, the declared shift and missingness, and the row count; without
    one it refuses with ``cell_not_licensed``.
    """

    _native: Any

    def measured_interval(self, *, execution: Any = None) -> Any:
        """Request the measured treated-minus-control interval; uncovered scope refuses."""
        from ._measured_inference import MeasuredInference
        from .execution import native_controls

        native, refusal = self._native.measured_interval(
            **native_controls(execution, memory="memory_bytes", cancel="cancel")
        )
        raise_refusal(refusal, MnarFixedShiftRefusal)
        if native is None:
            raise CausalValueError("native MNAR measurement returned no result or refusal")
        return MeasuredInference._from_native(native)


def prepare_fixed_shift(
    subject_ids: list[int],
    treatment: object,
    outcome: object,
    *,
    population: str,
    design: str,
    treated_shift: float,
    execution: Any = None,
) -> PreparedMnarFixedShift:
    """Compile the declared-shift candidate with explicit, untestable premises.

    ``design`` must be ``independent_randomized_arms``. ``treated_shift`` is the
    declared mean difference, in outcome units, between the treated arm's missing and
    observed outcomes; the control arm's is zero. It must be finite and at most two
    pooled within-arm standard deviations of the observed outcomes in magnitude.
    Neither the design nor the shift follows from observed rows.
    Subject IDs must be distinct across both arms and aligned with the rows; this
    cannot establish independence or randomization. The source needs 50 to 10000
    rows, at least 15 assigned and two observed outcomes per arm, no more than half
    of either arm missing, and varying observed outcomes. Arms may be unbalanced.
    No outcome or observation probability need be known.
    """
    from . import _native
    from .execution import native_controls

    if design != "independent_randomized_arms":
        raise CausalValueError("declare independent_randomized_arms")
    if not isinstance(population, str) or not population.strip():
        raise CausalValueError("population must be a nonempty string")
    shift = _real(treated_shift, "treated_shift")
    if isinstance(treated_shift, bool) or not math.isfinite(shift):
        raise CausalValueError("treated_shift must be a finite number")
    native, refusal = _native.mnar_fixed_shift_prepare(
        subject_ids,
        _treatment(treatment),
        _outcome(outcome),
        population,
        shift,
        **native_controls(execution, memory="memory_bytes", cancel="cancel"),
    )
    raise_refusal(refusal, MnarFixedShiftRefusal)
    if native is None:
        raise CausalValueError("native MNAR preparation returned no result or refusal")
    return PreparedMnarFixedShift(native)
