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

import antecedent
import numpy as np
import pytest


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
        execution=antecedent.Execution(seed=1),
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
        # The default mean-difference test has a closed-form null: no permutation
        # resolution to report, but the fields must exist.
        assert d.n_permutations is None
        assert d.p_value_floor is None
    # Independent reference. x's mechanism is its marginal: the statistic is the raw difference
    # of the second-half and first-half means. y's gains an intercept of 4: the statistic is
    # the shift of the residual y - x between halves (the fitted slope is within 1e-3 of 1).
    by_node = {d.node: d for d in detections}
    x, y = data["x"], data["y"]
    assert by_node["x"].statistic == pytest.approx(x[half:].mean() - x[:half].mean(), abs=1e-9)
    residual = y - x
    assert by_node["y"].statistic == pytest.approx(
        residual[half:].mean() - residual[:half].mean(), abs=0.01
    )
    # Benjamini-Hochberg over two targets: the smaller raw p is multiplied by 2/1, the larger by 2/2.
    assert by_node["y"].p_value < by_node["x"].p_value
    assert by_node["y"].adjusted_p_value == pytest.approx(2.0 * by_node["y"].p_value, rel=1e-9)
    assert by_node["x"].adjusted_p_value == pytest.approx(by_node["x"].p_value, rel=1e-9)
    assert (by_node["x"].changed, by_node["y"].changed) == (False, True)
