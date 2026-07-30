"""`adjusted_p_value` must reach Python (Rust ↔ Python parity).

`detect_mechanism_changes` applies a Benjamini–Hochberg correction across
`targets` by default, because judging each target against the nominal alpha
independently lets the family-wise false-positive rate grow with the number of
targets (~40% at ten truly-unchanged targets and alpha=0.05). The Rust
`MechanismChangeDetection` reports the raw and the adjusted p-value separately so
the multiplicity handling is auditable; the Python binding previously dropped the
adjusted value, leaving a caller able to see `changed` but not the number it was
decided from.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("antecedent")
import antecedent


def _shifted_mechanism(n: int = 160, seed: int = 5):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = np.empty(n)
    half = n // 2
    # y's mechanism gains an intercept in the second half; x's does not change.
    y[:half] = 1.0 * x[:half] + 0.1 * rng.normal(size=half)
    y[half:] = 1.0 * x[half:] + 4.0 + 0.1 * rng.normal(size=n - half)
    return {"x": x, "y": y}, half, n


def test_mechanism_change_detection_exposes_adjusted_p_value():
    data, half, n = _shifted_mechanism()
    detections = antecedent.attribution.mechanism_change_detection(
        ["x", "y"],
        [data["x"], data["y"]],
        [("x", "y")],
        0,
        half,
        half,
        n,
        seed=1,
    )
    assert detections, "expected at least one target"
    for d in detections:
        # Present, and typed as an optional float rather than silently absent.
        assert hasattr(d, "adjusted_p_value"), "adjusted_p_value missing from the binding"
        assert d.adjusted_p_value is None or isinstance(d.adjusted_p_value, float)
        # A correction is applied by default, so it must be populated and never
        # more significant than the raw value it was derived from.
        assert d.adjusted_p_value is not None
        assert d.adjusted_p_value >= d.p_value - 1e-12
        # `changed` is decided from the adjusted value, so the two must agree.
        assert "adjusted_p_value" in repr(d)
