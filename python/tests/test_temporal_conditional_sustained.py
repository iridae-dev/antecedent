"""A conditional sustained effect on a temporal series has an exact counted known truth.

Reference. The series of ``test_temporal_admg_window`` has outcome ``y_t = x_{t-1}`` exactly when
``w_{t-1} = 1`` (else 0) up to four flipped periods. The conditional sustained contrast, on a
single time step, is the contrast of ``y_{t+1}`` between ``x_t = 1`` and ``x_t = 0`` within each
level of ``w_t``, averaged over the law of ``w`` on the window. It is counted here from the
series; the engine's value must equal that count to 1e-12 and differ from the unadjusted
difference of means. The sustained form shares the pulse functional on one step, so it equals the
conditional pulse.
"""

from __future__ import annotations

import antecedent

from test_temporal_admg_window import _counted_conditional, _counted_unadjusted, _run, _series


def test_a_conditional_sustained_effect_equals_the_counted_stratified_contrast():
    series = _series()
    query = antecedent.TemporalConditionalSustained("x", "y", "w", treatment_lag=1, horizon_steps=1)
    result = _run(query)
    assert abs(result.ate - _counted_conditional(series)) < 1e-12
    assert abs(result.ate - _counted_unadjusted(series)) > 1e-3
    assert result.evidence_status == "supported"
    pulse = _run(antecedent.TemporalConditionalPulse("x", "y", "w", treatment_lag=1, horizon_steps=1))
    assert abs(result.ate - pulse.ate) < 1e-12
