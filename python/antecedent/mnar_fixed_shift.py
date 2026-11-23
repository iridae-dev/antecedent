"""Compiler-retained declared-shift MNAR candidate with a coverage-supported interval."""

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

    Its interval carries a calibration status that is ``calibrated`` only when a current coverage
    record covers this construction, the declared shift and missingness, and the row count.
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
    observed outcomes; the control arm's is zero. It must be finite; it is not capped,
    and a coverage record calibrates only the shift bands it measured.
    Neither the design nor the shift follows from observed rows.
    Subject IDs must be distinct across both arms and aligned with the rows; this
    cannot establish independence or randomization. The source needs at least
    two observed outcomes per arm and varying observed outcomes; its size is limited
    only by the execution's resource limits. Arms may be unbalanced and any share of
    either arm may be missing.
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
