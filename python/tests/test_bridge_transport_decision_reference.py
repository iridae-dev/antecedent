"""Bridge transport decisions against oracles written from the definitions, not from the route.

``test_bridge_transport_decision.py`` pins the route's lifecycle and compares several numbers with
columns the route itself returned. Here every expected number is computed from the declared
inputs alone:

* an affine bridge ``0.5 + 2 x`` maps each source draw to ``0.5 + 2 x`` (a known mapping, no
  spread), so a linear utility scores ``0.5 + 2 mean(x)`` and a squared utility
  ``mean((0.5 + 2 x)^2)``, both rebuilt from the generated source draws with the standard library
  normal quantile;
* the premise assessment's two lines are the least-squares lines of the trials (numpy), the
  prediction variance at a grid point is ``rss / (n - 2) (1 / n + (x - xbar)^2 / Sxx)`` per
  trial set, the bands are the Bonferroni Wald bands ``d +- z se`` with
  ``z = Phi^{-1}(1 - 0.05 / (2 k))``, and the planted shift of the target trials is the known
  difference;
* a trial-level bridge maps the source draws through its fitted line, so the bridged mean is the
  line at the source draws to within the Monte Carlo error of the coefficient draws, and the
  robust bounds of a linear utility are that mean shifted by the widest oracle band.
"""

from __future__ import annotations

from dataclasses import replace
from statistics import NormalDist

import numpy as np
import pytest
from antecedent import bridge_transport_decision as btd
from antecedent import decision
from antecedent.execution import Execution
from test_bridge_transport_decision import (
    PREMISE,
    ROWS,
    TRIALS,
    assess,
    choice,
    draws,
    premise_contract,
    trial_contract,
    trials,
)

_NORMAL = NormalDist()
GRID = (-0.8, 0.0, 0.8)


def source_column(mean: float, sd: float) -> np.ndarray:
    return np.array([mean + sd * _NORMAL.inv_cdf((i + 0.5) / ROWS) for i in range(ROWS)])


def partner_column() -> np.ndarray:
    return np.array(
        [0.2 + 0.3 * _NORMAL.inv_cdf(((i * 487) % ROWS + 0.5) / ROWS) for i in range(ROWS)]
    )


def line_of(mapping) -> tuple[float, float, float, float, float]:
    """Intercept, slope, mean x, Sxx and the residual variance of a trial set's OLS line."""
    xs = np.array([x for x, _ in mapping.trials])
    ys = np.array([y for _, y in mapping.trials])
    slope, intercept = np.polyfit(xs, ys, 1)
    rss = float(np.sum((ys - intercept - slope * xs) ** 2))
    sxx = float(np.sum((xs - xs.mean()) ** 2))
    return float(intercept), float(slope), float(xs.mean()), sxx, rss / (len(xs) - 2)


def prediction_variance(mapping, x: float) -> float:
    _, _, centre, sxx, sigma2 = line_of(mapping)
    n = len(mapping.trials)
    return sigma2 * (1.0 / n + (x - centre) ** 2 / sxx)


def bands(source, target, grid=GRID):
    """The independent Bonferroni Wald bands of target minus source at every grid point."""
    z = _NORMAL.inv_cdf(1.0 - 0.05 / (2.0 * len(grid)))
    out = []
    for x in grid:
        (b0, b1, *_), (c0, c1, *_) = line_of(source), line_of(target)
        difference = (c0 + c1 * x) - (b0 + b1 * x)
        se = float(np.sqrt(prediction_variance(source, x) + prediction_variance(target, x)))
        out.append((x, difference, se, difference - z * se, difference + z * se))
    return out


def test_an_affine_bridge_decision_scores_the_known_mapping_of_the_source_draws() -> None:
    execution = Execution(seed=31)
    result = btd.compile_bridge_transport_decision(
        premise_contract(), draws(1.0, 0.3), choice(), stream=9, execution=execution
    ).execute()
    source = source_column(1.0, 0.3)
    mapped = 0.5 + 2.0 * source
    column = np.asarray(result.distribution())
    # Row for row: the bridged column is the known map of the source column, and the other
    # coordinate of the aligned table is untouched.
    assert column[:, result.bridge.column] == pytest.approx(mapped, abs=1e-12)
    other = 1 - result.bridge.column
    assert column[:, other] == pytest.approx(partner_column(), abs=1e-12)
    assert result.bridge.mean == pytest.approx(0.5 + 2.0 * source.mean(), rel=1e-12)
    chosen = result.decision
    # A linear utility scores the mean of the mapped draws, a squared one their second moment
    # (never the square of the mean), and the larger second moment wins.
    assert chosen.action("use-bridge").expected_utility == pytest.approx(mapped.mean(), rel=1e-12)
    assert chosen.action("risk-sensitive").expected_utility == pytest.approx(
        float(np.mean(mapped**2)), rel=1e-12
    )
    assert float(np.mean(mapped**2)) - mapped.mean() ** 2 == pytest.approx(
        4.0 * source.var(), rel=1e-12
    )
    assert chosen.verdict.actions[0] == "risk-sensitive"


def test_the_premise_assessment_bands_equal_the_independent_wald_construction() -> None:
    result = assess(GRID).execute(execution=Execution(seed=73))
    for fit, mapping in ((result.source_fit, trials(0.0)), (result.target_fit, trials(0.05))):
        intercept, slope, _, _, _ = line_of(mapping)
        assert fit.intercept == pytest.approx(intercept, rel=1e-9)
        assert fit.slope == pytest.approx(slope, rel=1e-9)
    expected = bands(trials(0.0), trials(0.05))
    assert len(result.grid) == len(expected)
    for point, (x, difference, se, lower, upper) in zip(result.grid, expected, strict=True):
        assert point.surrogate_effect == x
        # The planted shift of the target trials is the known discrepancy at every grid point.
        assert point.difference == pytest.approx(0.05, abs=1e-9)
        assert point.difference == pytest.approx(difference, abs=1e-12)
        assert point.standard_error == pytest.approx(se, rel=1e-9)
        assert point.lower == pytest.approx(lower, rel=1e-9, abs=1e-12)
        assert point.upper == pytest.approx(upper, rel=1e-9, abs=1e-12)
    assert result.equivalence_screen_passed == all(
        lower > -0.2 and upper < 0.2 for *_, lower, upper in expected
    )


def test_a_planted_target_shift_beyond_the_margin_fails_the_equivalence_screen() -> None:
    shift = 0.5
    compiled = btd.compile_premise_assessment(
        PREMISE,
        source=trials(0.0),
        target=trials(shift),
        source_trial_ids=[f"source-trial-{i}" for i in range(TRIALS)],
        target_trial_ids=[f"target-trial-{i}" for i in range(TRIALS)],
        surrogate_grid=list(GRID),
        equivalence_margin=0.2,
    )
    result = compiled.execute(execution=Execution(seed=73))
    assert not result.equivalence_screen_passed
    for point in result.grid:
        assert point.difference == pytest.approx(shift, abs=1e-9)
        assert point.lower > 0.2  # the whole band lies beyond the margin


def test_a_trial_level_bridge_follows_the_fitted_line_and_robust_bounds_shift_it() -> None:
    execution = Execution(seed=43)
    intercept, slope, *_ = line_of(trials(0.0))
    source = source_column(0.0, 0.1)
    line_mean = float(np.mean(intercept + slope * source))
    maximin = replace(choice(), criterion=decision.Criterion.maximin_over_structures())
    result = btd.compile_bridge_transport_decision(
        trial_contract(),
        draws(0.0, 0.1),
        choice(),
        stream=11,
        assessment=assess(),
        robust=maximin,
        execution=execution,
    ).execute()
    mapped_mean = float(np.asarray(result.distribution())[:, result.bridge.column].mean())
    # Coefficient draws spread around the fitted line by about 0.003 per row; 64 rows average it.
    assert mapped_mean == pytest.approx(line_mean, abs=0.01)
    expected = bands(trials(0.0), trials(0.05))
    lowest = min(lower for *_, lower, _ in expected)
    highest = max(upper for *_, upper in expected)
    linear = result.robust.action("use-bridge")
    assert linear.utility_lower == pytest.approx(mapped_mean + lowest, abs=1e-9)
    assert linear.utility_upper == pytest.approx(mapped_mean + highest, abs=1e-9)


def _consistent(support: dict | None, unit: str) -> None:
    assert support is not None, "the route is bound to its capability record"
    assert support["unit"] == unit
    assert support["supported"] == (support["status"] == "supported")
    assert bool(support["missing_evidence"]) != support["supported"]


def test_the_compiled_routes_report_the_support_unit_of_their_records() -> None:
    execution = Execution(seed=31)
    conditional = btd.compile_bridge_transport_decision(
        premise_contract(), draws(1.0, 0.3), choice(), stream=9, execution=execution
    )
    _consistent(conditional.unit_support(), "transport.bridge_decision")
    assessed = btd.compile_bridge_transport_decision(
        trial_contract(),
        draws(0.4, 0.2),
        choice(),
        stream=11,
        assessment=assess(),
        execution=execution,
    )
    _consistent(assessed.unit_support(), "transport.bridge_decision")
    _consistent(assess().unit_support(), "transport.bridge_premise_assessment")
    # Reading the decision changes no identity.
    assert conditional.plan_view()["identity"] == conditional.program_identity
