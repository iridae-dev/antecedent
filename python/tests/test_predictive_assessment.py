"""Predictive assessment through the Python facade.

Closed forms are recomputed here in pure Python (``math.erf``), never by native code:

* Gaussian CRPS ``sigma (z (2 Phi(z) - 1) + 2 phi(z) - 1/sqrt(pi))``, ``z = (y - mu)/sigma``;
  the log score ``ln sigma + ln sqrt(2 pi) + z^2/2``;
* Bernoulli ``p = 0.7``: Brier ``0.09`` (hit) and ``0.49`` (miss), log score ``-ln 0.7`` and
  ``-ln 0.3``, CRPS equal to the Brier score;
* a sample ``{0, 1, 2}`` at ``1.5``: the energy form ``7/18`` and the fair form ``1/6``;
* a finite law ``{0: 1/2, 1: 1/2}`` at ``0.5``: CRPS ``1/4``;
* the variance of a mean difference: iid ``s^2/n``, Bartlett-kernel serial and cluster-robust,
  recomputed from the per-case scores the report returns.

The assessment is point-only: no test asserts an interval, and the ones that ask for one assert the
typed refusal.
"""

from __future__ import annotations

import json
import math
import random
import subprocess
import sys

import pytest
from antecedent import evidence_ledger as el
from antecedent import predictive_assessment as pa
from antecedent.errors import CausalTypeError, CausalValueError
from antecedent.predictive_assessment import PredictiveAssessmentRefusal

from _refusal import assert_registered_refusal

TOL = 1e-12
INV_SQRT_PI = 1.0 / math.sqrt(math.pi)


# --------------------------------------------------------------------------- oracles


def phi(z: float) -> float:
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def big_phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def crps_gaussian(mean: float, sd: float, y: float) -> float:
    z = (y - mean) / sd
    return sd * (z * (2.0 * big_phi(z) - 1.0) + 2.0 * phi(z) - INV_SQRT_PI)


def log_gaussian(mean: float, sd: float, y: float) -> float:
    z = (y - mean) / sd
    return math.log(sd) + 0.5 * math.log(2.0 * math.pi) + 0.5 * z * z


def sample_crps(draws: list[float], y: float, *, fair: bool = False) -> float:
    m = len(draws)
    first = sum(abs(x - y) for x in draws) / m
    pairs = sum(abs(a - b) for a in draws for b in draws)
    return first - pairs / (2.0 * m * (m - 1) if fair else 2.0 * m * m)


def mean_of(values: list[float]) -> float:
    return sum(values) / len(values)


def iid_variance(d: list[float]) -> float:
    m = mean_of(d)
    return sum((x - m) ** 2 for x in d) / (len(d) - 1) / len(d)


def serial_variance(d: list[float], lag: int) -> float:
    n = len(d)
    m = mean_of(d)
    c = [x - m for x in d]

    def gamma(k: int) -> float:
        return sum(c[i] * c[i - k] for i in range(k, n)) / n

    long_run = gamma(0) + 2.0 * sum((1.0 - k / (lag + 1.0)) * gamma(k) for k in range(1, lag + 1))
    return max(long_run, 0.0) / n


def cluster_variance(d: list[float], labels: list[int]) -> float:
    n = len(d)
    m = mean_of(d)
    sums: dict[int, float] = {}
    for x, label in zip(d, labels, strict=True):
        sums[label] = sums.get(label, 0.0) + (x - m)
    g = len(sums)
    return g / (g - 1.0) * sum(v * v for v in sums.values()) / (n * n)


# --------------------------------------------------------------------------- fixtures


TARGET = pa.Target("outcome", "cohort", "mmHg")


def observed(values: list[float], *, target: pa.Target = TARGET, start: int = 0) -> pa.Observations:
    ids = pa.keys("cohort", range(start, start + len(values)))
    return pa.Observations(target, ids, "real", values)


def predictor(name: str, laws, *, target: pa.Target = TARGET, **kw) -> pa.Predictor:
    return pa.Predictor(name, laws, target, fit=pa.FitDeclaration.not_fitted(), **kw)


def normals(n: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    return [rng.gauss(0.0, 1.0) for _ in range(n)]


def simulation() -> pa.ValidationDesign:
    return pa.ValidationDesign.simulation_stream("python random.Random seed 20240611")


def refused(call) -> PredictiveAssessmentRefusal:
    with pytest.raises(PredictiveAssessmentRefusal) as caught:
        call()
    assert_registered_refusal(caught.value)
    return caught.value


def expect(error: PredictiveAssessmentRefusal, code: str, detail: str) -> None:
    assert error.reason_code == code, (error.reason_code, str(error))
    assert error.detail == detail, (error.detail, str(error))


# --------------------------------------------------------------------------- closed forms


@pytest.mark.parametrize(
    ("mean", "sd", "y"), [(0.0, 1.0, 0.0), (0.0, 1.0, 1.0), (1.0, 2.0, -1.0), (-0.5, 0.3, 0.4)]
)
def test_gaussian_scores_equal_the_closed_forms(mean: float, sd: float, y: float) -> None:
    law = pa.Gaussian(mean, sd)
    assert pa.score_case("crps", law, y) == pytest.approx(crps_gaussian(mean, sd, y), abs=TOL)
    assert pa.score_case("log_score", law, y) == pytest.approx(log_gaussian(mean, sd, y), abs=TOL)


def test_the_gaussian_crps_at_the_centre_is_the_textbook_constant() -> None:
    assert pa.score_case("crps", pa.Gaussian(0.0, 1.0), 0.0) == pytest.approx(
        (math.sqrt(2.0) - 1.0) / math.sqrt(math.pi), abs=TOL
    )


def test_bernoulli_brier_log_and_crps_equal_the_hand_values() -> None:
    law = pa.Bernoulli(0.7)
    assert pa.score_case("brier", law, 1) == pytest.approx(0.09, abs=TOL)
    assert pa.score_case("brier", law, 0) == pytest.approx(0.49, abs=TOL)
    assert pa.score_case("log_score", law, 1) == pytest.approx(-math.log(0.7), abs=TOL)
    assert pa.score_case("log_score", law, 0) == pytest.approx(-math.log(0.3), abs=TOL)
    # The CRPS of a binary law is its Brier score.
    assert pa.score_case("crps", law, 1) == pytest.approx(0.09, abs=TOL)
    assert pa.score_case("crps", law, 0) == pytest.approx(0.49, abs=TOL)


def test_categorical_brier_and_log_equal_the_hand_values() -> None:
    law = pa.Categorical([0.5, 0.3, 0.2])
    assert pa.score_case("brier", law, 1) == pytest.approx(0.25 + 0.49 + 0.04, abs=TOL)
    assert pa.score_case("log_score", law, 2) == pytest.approx(-math.log(0.2), abs=TOL)


def test_finite_and_sample_crps_equal_the_energy_form() -> None:
    finite = pa.FiniteReal([0.0, 1.0], [0.5, 0.5])
    assert pa.score_case("crps", finite, 0.5) == pytest.approx(0.25, abs=TOL)
    draws = [0.0, 1.0, 2.0]
    sample = pa.EmpiricalSample(draws)
    assert pa.score_case("crps", sample, 1.5) == pytest.approx(7.0 / 18.0, abs=TOL)
    assert pa.score_case("crps", sample, 1.5) == pytest.approx(sample_crps(draws, 1.5), abs=TOL)
    assert pa.score_case("crps", sample, 1.5, fair=True) == pytest.approx(1.0 / 6.0, abs=TOL)
    assert pa.score_case("crps", sample, 1.5, fair=True) == pytest.approx(
        sample_crps(draws, 1.5, fair=True), abs=TOL
    )
    rng = random.Random(7)
    wide = [rng.gauss(0.0, 1.0) for _ in range(40)]
    assert pa.score_case("crps", pa.EmpiricalSample(wide), 0.3) == pytest.approx(
        sample_crps(wide, 0.3), abs=1e-10
    )


def test_a_sample_converges_to_the_gaussian_crps() -> None:
    rng = random.Random(11)
    wide = [rng.gauss(0.0, 1.0) for _ in range(2000)]
    value = pa.score_case("crps", pa.EmpiricalSample(wide), 0.7)
    assert value == pytest.approx(crps_gaussian(0.0, 1.0, 0.7), abs=0.05)


def test_energy_score_equals_the_hand_value_and_the_crps_in_one_dimension() -> None:
    joint = pa.JointFinite([[0.0, 0.0], [1.0, 0.0]], [0.5, 0.5])
    assert pa.score_case("energy_score", joint, [0.0, 0.0]) == pytest.approx(0.25, abs=TOL)
    line = pa.JointFinite([[0.0], [1.0]], [0.5, 0.5])
    assert pa.score_case("energy_score", line, [0.5]) == pytest.approx(0.25, abs=TOL)
    sample = pa.JointSample([[0.0, 0.0], [1.0, 0.0]])
    assert pa.score_case("energy_score", sample, [0.0, 0.0]) == pytest.approx(0.25, abs=TOL)


def test_unsupported_scores_and_infinite_log_scores_are_refused() -> None:
    error = refused(lambda: pa.score_case("log_score", pa.EmpiricalSample([0.0, 1.0]), 0.5))
    expect(error, "route_not_supported", "predictive_assessment.score_not_defined")
    error = refused(lambda: pa.score_case("log_score", pa.Bernoulli(0.0), 1))
    expect(error, "invalid_argument", "predictive_assessment.log_score_infinite")
    error = refused(lambda: pa.score_case("energy_score", pa.Gaussian(0.0, 1.0), 0.0))
    expect(error, "route_not_supported", "predictive_assessment.score_not_defined")


def test_the_true_finite_law_has_the_smallest_expected_score() -> None:
    support = [-1.0, 0.0, 2.0]
    truth = {"kind": "real", "support": support, "probabilities": [0.2, 0.5, 0.3]}
    honest = pa.FiniteReal(support, [0.2, 0.5, 0.3])
    entropy = -sum(p * math.log(p) for p in (0.2, 0.5, 0.3))
    assert pa.exact_expected_score("log_score", honest, truth) == pytest.approx(entropy, abs=TOL)
    for score in ("log_score", "crps"):
        best = pa.exact_expected_score(score, honest, truth)
        for q in ([0.3, 0.4, 0.3], [0.2, 0.4, 0.4], [1 / 3, 1 / 3, 1 / 3], [0.1, 0.7, 0.2]):
            assert pa.exact_expected_score(score, pa.FiniteReal(support, q), truth) > best + 1e-6
    # E Brier(q) = p (1 - p) + (q - p)^2 under a Bernoulli(p) truth.
    p = 0.3
    category = {"kind": "category", "probabilities": [1 - p, p]}
    for q in (0.0, 0.1, 0.3, 0.5, 0.9):
        value = pa.exact_expected_score("brier", pa.Bernoulli(q), category)
        assert value == pytest.approx(p * (1 - p) + (q - p) ** 2, abs=TOL)


# --------------------------------------------------------------------------- generators


def generator_assessment(n: int = 1000, seed: int = 20240611) -> pa.PredictiveAssessment:
    values = normals(n, seed)
    return pa.assess(
        observed(values),
        [
            predictor("calibrated", [pa.Gaussian(0.0, 1.0)] * n),
            predictor("overconfident", [pa.Gaussian(0.0, 0.5)] * n),
            predictor("biased", [pa.Gaussian(1.0, 1.0)] * n),
        ],
        simulation(),
        scores=("log_score", "crps"),
        dependence=pa.Dependence.iid(),
    )


def test_a_calibrated_generator_has_a_uniform_pit_and_nominal_coverage() -> None:
    n = 1000
    report = generator_assessment(n).report
    d = report.predictor("calibrated").diagnostics
    assert (d.label, d.calibration) == ("diagnostic", "unmeasured")
    assert d.pit is not None and d.coverage is not None
    # Recomputed from the returned mid-PIT values: Kolmogorov-Smirnov against U(0, 1).
    u = sorted(d.pit.mid_pit)
    ks = max(max((i + 1) / n - x, x - i / n) for i, x in enumerate(u))
    assert ks == pytest.approx(d.pit.ks_distance, abs=TOL)
    assert ks < 0.07  # a 99.9% critical value of n = 1000 is about 0.062
    assert sum(d.pit.counts) == pytest.approx(n, abs=1e-9)
    assert abs(d.coverage[0].empirical_coverage - 0.5) < 0.06
    assert abs(d.coverage[1].empirical_coverage - 0.9) < 0.04
    # The expected CRPS of the true law is 1 / sqrt(pi).
    assert report.predictor("calibrated").score("crps").mean == pytest.approx(INV_SQRT_PI, abs=0.05)


def test_miscalibrated_generators_are_flagged_with_the_expected_outcomes() -> None:
    report = generator_assessment().report
    calibrated = report.predictor("calibrated").diagnostics
    overconfident = report.predictor("overconfident").diagnostics
    biased = report.predictor("biased").diagnostics
    assert calibrated.pit is not None and overconfident.pit is not None and biased.pit is not None
    # Population KS distances: 0.161 for sd 0.5 against unit-sd outcomes and 0.383 for mean 1.
    assert overconfident.pit.ks_distance > 0.10
    assert biased.pit.ks_distance > 0.25
    assert overconfident.pit.chi_square > 10 * calibrated.pit.chi_square
    assert (
        overconfident.coverage is not None and overconfident.coverage[1].empirical_coverage < 0.72
    )
    assert overconfident.mean_predictive_sd == pytest.approx(0.5, abs=TOL)
    for score in ("log_score", "crps"):
        good = report.predictor("calibrated").score(score).mean
        assert good < report.predictor("overconfident").score(score).mean
        assert good < report.predictor("biased").score(score).mean


def test_score_differences_are_paired_with_the_stated_dependence() -> None:
    report = generator_assessment(600).report
    assert len(report.differences) == 4
    for diff in report.differences:
        assert diff.reference == "calibrated"
        assert diff.pairing == "paired_on_shared_observations"
        assert diff.dependence == "iid"
        assert (diff.label, diff.calibration, diff.interval) == (
            "diagnostic",
            "unmeasured",
            "not_reported",
        )
        assert diff.mean_difference > 0
        assert diff.standard_error is not None and diff.standard_error > 0
        assert diff.dm_statistic is not None and diff.dm_statistic > 3
        cand = report.predictor(diff.candidate).score(diff.score).per_case
        ref = report.predictor("calibrated").score(diff.score).per_case
        d = [a - b for a, b in zip(cand, ref, strict=True)]
        assert diff.mean_difference == pytest.approx(mean_of(d), abs=TOL)
        assert diff.standard_error == pytest.approx(math.sqrt(iid_variance(d)), abs=1e-10)


def test_without_a_declared_dependence_no_standard_error_is_reported() -> None:
    n = 100
    result = pa.assess(
        observed(normals(n, 3)),
        [
            predictor("a", [pa.Gaussian(0.0, 1.0)] * n),
            predictor("b", [pa.Gaussian(0.5, 1.0)] * n),
        ],
        simulation(),
        scores=("crps",),
    )
    assert result.report.dependence == "undeclared"
    diff = result.report.differences[0]
    assert diff.standard_error is None and diff.dm_statistic is None
    assert diff.dependence == "undeclared"
    assert result.report.predictor("a").score("crps").mean > 0


# --------------------------------------------------------------------------- dependence


def rolling_result(dependence: pa.Dependence, horizon: int = 2) -> pa.PredictiveAssessment:
    n = 40
    target = pa.Target("outcome", "cohort", "mmHg", horizon)
    rng = random.Random(5)
    # An AR(1) series of outcomes so that serial dependence is real.
    y, prev = [], 0.0
    for _ in range(n):
        prev = 0.6 * prev + rng.gauss(0.0, 1.0)
        y.append(prev)
    return pa.assess(
        observed(y, target=target),
        [
            predictor("a", [pa.Gaussian(0.0, 1.3)] * n, target=target),
            predictor("b", [pa.Gaussian(0.4, 1.0)] * n, target=target),
        ],
        pa.ValidationDesign.rolling_origin([i // 4 for i in range(n)], horizon),
        scores=("crps",),
        dependence=dependence,
    )


def test_a_serial_standard_error_is_the_bartlett_long_run_variance() -> None:
    n = 40
    result = rolling_result(pa.Dependence.serial(list(range(n)), 3))
    diff = result.report.differences[0]
    cand = result.report.predictor("b").score("crps").per_case
    ref = result.report.predictor("a").score("crps").per_case
    d = [x - y for x, y in zip(cand, ref, strict=True)]
    assert diff.dependence == "serial"
    assert diff.standard_error == pytest.approx(math.sqrt(serial_variance(d, 3)), abs=1e-10)
    folds = result.report.predictor("a").score("crps").folds
    assert [f.fold for f in folds] == list(range(10))
    assert all(f.cases == 4 for f in folds)
    # Case order is a time order, so the PIT drift between the halves is reported.
    drift = result.report.predictor("a").diagnostics.drift
    assert drift is not None and (drift.reference_cases, drift.comparison_cases) == (20, 20)


def test_a_cluster_standard_error_is_the_cluster_robust_variance() -> None:
    n = 40
    labels = [i // 5 for i in range(n)]
    rng = random.Random(9)
    values = [rng.gauss(0.0, 1.0) + 0.5 * (i // 5) for i in range(n)]
    result = pa.assess(
        observed(values),
        [
            predictor("a", [pa.Gaussian(0.0, 1.0)] * n),
            predictor("b", [pa.Gaussian(1.0, 1.0)] * n),
        ],
        pa.ValidationDesign.blocked(labels),
        scores=("crps",),
        dependence=pa.Dependence.cluster(labels),
    )
    diff = result.report.differences[0]
    cand = result.report.predictor("b").score("crps").per_case
    ref = result.report.predictor("a").score("crps").per_case
    d = [x - y for x, y in zip(cand, ref, strict=True)]
    # Canonical order sorts by observation key; recompute with the labels in that order.
    by_key = dict(zip(pa.keys("cohort", range(n)), labels, strict=True))
    canonical = [by_key[k] for k in result.report.case_keys]
    assert diff.standard_error == pytest.approx(
        math.sqrt(cluster_variance(d, canonical)), abs=1e-10
    )
    assert diff.dependence == "cluster"


def test_iid_is_refused_for_temporal_and_clustered_designs_and_short_lags_too() -> None:
    error = refused(lambda: rolling_result(pa.Dependence.iid()))
    expect(
        error,
        "sampling_dependence_unknown",
        "predictive_assessment.iid_not_licensed_for_dependent_design",
    )
    n = 40
    error = refused(lambda: rolling_result(pa.Dependence.serial(list(range(n)), 0), horizon=4))
    expect(error, "sampling_dependence_unknown", "predictive_assessment.serial_lag_below_horizon")
    error = refused(
        lambda: pa.assess(
            observed([0.1, 0.2, 0.3, 0.4]),
            [predictor("a", [pa.Gaussian(0.0, 1.0)] * 4)],
            pa.ValidationDesign.blocked([0, 0, 1, 1]),
            scores=("crps",),
            dependence=pa.Dependence.iid(),
        )
    )
    expect(
        error,
        "sampling_dependence_unknown",
        "predictive_assessment.iid_not_licensed_for_dependent_design",
    )


# --------------------------------------------------------------------------- refusals


def test_a_mismatched_target_or_horizon_is_refused_not_converted() -> None:
    obs = observed([0.1, 0.2, 0.3])
    laws = [pa.Gaussian(0.0, 1.0)] * 3
    other_units = pa.Target("outcome", "cohort", "kPa")
    error = refused(
        lambda: pa.assess(
            obs, [predictor("a", laws, target=other_units)], simulation(), scores=("crps",)
        )
    )
    expect(error, "quantity_semantics_mismatch", "predictive_assessment.target_mismatch")
    error = refused(
        lambda: pa.assess(
            obs,
            [predictor("a", laws, target=pa.Target("outcome", "cohort", "mmHg", 3))],
            simulation(),
            scores=("crps",),
        )
    )
    expect(error, "quantity_semantics_mismatch", "predictive_assessment.horizon_mismatch")


def test_callback_laws_and_incomplete_provider_identities_are_refused_by_name() -> None:
    obs = observed([0.1, 0.2])
    laws = [pa.Gaussian(0.0, 1.0)] * 2
    callback = predictor("cb", laws, source=pa.LawSource.callback("python closure"))
    error = refused(lambda: pa.assess(obs, [callback], simulation(), scores=("crps",)))
    expect(error, "route_not_supported", "predictive_assessment.law_callback_not_built")
    incomplete = predictor(
        "svc",
        laws,
        source=pa.LawSource.provider(
            provider_id="svc", object_id="law", version_id="1", snapshot_id="", request_id="r"
        ),
    )
    error = refused(lambda: pa.assess(obs, [incomplete], simulation(), scores=("crps",)))
    expect(error, "invalid_argument", "predictive_assessment.provider_identity_incomplete")


def test_the_closed_routes_raise_typed_refusals() -> None:
    n = 20
    result = pa.assess(
        observed(normals(n, 1)),
        [predictor("a", [pa.Gaussian(0.0, 1.0)] * n)],
        simulation(),
        scores=("crps",),
        dependence=pa.Dependence.iid(),
    )
    assert result.report.inference_claim == "point_only"
    assert result.report.causal_validity == "not_established"
    assert result.report.interval == "not reported"
    expect(
        refused(result.score_interval),
        "cell_not_licensed",
        "predictive_assessment.interval_not_licensed",
    )
    expect(
        refused(result.causal_validity),
        "route_not_supported",
        "predictive_assessment.not_a_causal_claim",
    )
    expect(
        refused(result.score_as_probability),
        "route_not_supported",
        "predictive_assessment.score_not_probability",
    )
    error = refused(
        lambda: pa.assess(
            observed(normals(n, 1)),
            [predictor("a", [pa.Gaussian(0.0, 1.0)] * n)],
            simulation(),
            scores=("crps",),
            score_interval=(0.95, "stationary bootstrap"),
        )
    )
    expect(error, "cell_not_licensed", "predictive_assessment.interval_not_licensed")


def test_malformed_declarations_are_typed_input_errors() -> None:
    obs = observed([0.1, 0.2])
    laws = [pa.Gaussian(0.0, 1.0)] * 2
    with pytest.raises(CausalValueError):
        pa.assess(obs, [predictor("a", laws)], simulation(), scores=("nope",))
    with pytest.raises(CausalTypeError):
        pa.assess("not observations", [], simulation(), scores=("crps",))  # type: ignore[arg-type]
    with pytest.raises(CausalValueError):
        pa.Gaussian(0.0, float("nan")).to_wire()
    error = refused(
        lambda: pa.assess(
            obs, [predictor("a", [pa.Gaussian(0.0, 0.0)] * 2)], simulation(), scores=("crps",)
        )
    )
    expect(error, "invalid_argument", "predictive_assessment.invalid_law")
    # Repeated observation identities are refused, not scored twice.
    twice = pa.Observations(TARGET, [pa.keys("cohort", [1])[0]] * 2, "real", [0.0, 1.0])
    error = refused(
        lambda: pa.assess(twice, [predictor("a", laws)], simulation(), scores=("crps",))
    )
    expect(error, "invalid_argument", "predictive_assessment.invalid_observations")


# --------------------------------------------------------------------------- overlap


def overlap_result(fit: pa.FitDeclaration, *, strict: bool = False) -> pa.PredictiveAssessment:
    n = 6
    return pa.assess(
        observed(normals(n, 4)),
        [pa.Predictor("model", [pa.Gaussian(0.0, 1.0)] * n, TARGET, fit=fit)],
        pa.ValidationDesign.held_out_split(),
        scores=("crps",),
        dependence=pa.Dependence.iid(),
        require_held_out=strict,
    )


def test_overlap_is_decided_from_row_identities() -> None:
    disjoint = pa.FitDeclaration.fitted_on(pa.keys("cohort", range(100, 110)))
    ok = overlap_result(disjoint, strict=True)
    assert ok.report.predictor("model").overlap.label == "held_out_verified"
    shared = pa.FitDeclaration.fitted_on(pa.keys("cohort", range(4, 10)))
    labelled = overlap_result(shared)
    overlap = labelled.report.predictor("model").overlap
    assert (overlap.label, overlap.shared) == ("in_sample", 2)
    error = refused(lambda: overlap_result(shared, strict=True))
    expect(error, "evidence_double_counted", "predictive_assessment.in_sample")


def test_an_undeclared_fit_is_unverified_and_never_held_out() -> None:
    undeclared = pa.FitDeclaration.undeclared("provider declared no fit data")
    assert overlap_result(undeclared).report.predictor("model").overlap.label == "unverified"
    error = refused(lambda: overlap_result(undeclared, strict=True))
    expect(error, "evidence_overlap_unknown", "predictive_assessment.overlap_unknown")


def test_the_fit_set_is_resolved_from_the_evidence_ledger() -> None:
    ledger = el.EvidenceLedger()
    train = ledger.add_source(
        el.SourceDeclaration(
            "train", "observation_set", observations=el.rows("cohort", range(0, 50))
        )
    )
    test = ledger.add_source(
        el.SourceDeclaration(
            "test", "observation_set", observations=el.rows("cohort", range(100, 104))
        )
    )
    model = ledger.add_source(
        el.SourceDeclaration(
            "model", "fitted_model", lineage=el.Lineage.derived_from_used_data([train])
        )
    )
    fit = pa.FitDeclaration.from_ledger(ledger, [model])
    assert len(fit.wire["keys"]) == 50
    assessed = pa.ledger_observations(ledger, test)
    assert len(assessed) == 4
    outcomes = [0.0, 1.0, 0.0, 1.0]
    held = pa.assess(
        pa.Observations(TARGET, assessed, "real", outcomes),
        [pa.Predictor("m", [pa.Gaussian(0.0, 1.0)] * 4, TARGET, fit=fit)],
        pa.ValidationDesign.held_out_split(),
        scores=("crps",),
        dependence=pa.Dependence.iid(),
        require_held_out=True,
    )
    assert held.report.predictor("m").overlap.label == "held_out_verified"
    # A model derived from the test rows too is in sample, by lineage and not by label.
    leaky = ledger.add_source(
        el.SourceDeclaration(
            "leaky", "fitted_model", lineage=el.Lineage.derived_from_used_data([train, test])
        )
    )
    leaky_fit = pa.FitDeclaration.from_ledger(ledger, [leaky])
    error = refused(
        lambda: pa.assess(
            pa.Observations(TARGET, assessed, "real", outcomes),
            [pa.Predictor("m", [pa.Gaussian(0.0, 1.0)] * 4, TARGET, fit=leaky_fit)],
            pa.ValidationDesign.held_out_split(),
            scores=("crps",),
            dependence=pa.Dependence.iid(),
            require_held_out=True,
        )
    )
    expect(error, "evidence_double_counted", "predictive_assessment.in_sample")
    # An undeclared source is unknown, never held out.
    vendor = ledger.add_source(el.SourceDeclaration("vendor", "external_claim"))
    assert pa.FitDeclaration.from_ledger(ledger, [vendor]).wire["kind"] == "undeclared"
    unknown = refused(lambda: pa.ledger_observations(ledger, vendor))
    expect(unknown, "evidence_overlap_unknown", "predictive_assessment.overlap_unknown")


# --------------------------------------------------------------------------- drift and adapters


def test_drift_compares_the_pit_of_two_windows() -> None:
    same = pa.distribution_drift([0.1, 0.4, 0.6, 0.9], [0.1, 0.4, 0.6, 0.9], 4)
    assert same.ks_distance == pytest.approx(0.0, abs=TOL)
    assert same.population_stability_index == pytest.approx(0.0, abs=TOL)
    assert (same.label, same.calibration, same.route) == (
        "diagnostic",
        "unmeasured",
        "distribution_drift",
    )
    moved = pa.distribution_drift([0.1, 0.2, 0.3], [0.7, 0.8, 0.9], 4)
    assert moved.ks_distance == pytest.approx(1.0, abs=TOL)
    # Hand value with a half-count pseudocount: sum (pb - pa) ln(pb / pa).
    pa_, pb = [0.5, 0.3, 0.1, 0.1], [0.1, 0.1, 0.3, 0.5]
    expected = sum((b - a) * math.log(b / a) for a, b in zip(pa_, pb, strict=True))
    assert moved.population_stability_index == pytest.approx(expected, abs=1e-9)
    error = refused(lambda: pa.distribution_drift([], [0.5], 4))
    expect(error, "invalid_argument", "predictive_assessment.diagnostic_not_defined")


def test_posterior_predictive_replicates_are_scored_as_sample_laws() -> None:
    replicates = [[0.0, 5.0], [1.0, 5.0], [2.0, 5.0]]
    laws = pa.replicate_laws(replicates)
    assert [law.draws for law in laws] == [[0.0, 1.0, 2.0], [5.0, 5.0, 5.0]]
    result = pa.assess(
        observed([1.5, 5.0]),
        [
            pa.Predictor(
                "ppc",
                laws,
                TARGET,
                fit=pa.FitDeclaration.undeclared("fitted on the scored rows"),
            )
        ],
        pa.ValidationDesign.held_out_split(),
        scores=("crps",),
        dependence=pa.Dependence.iid(),
    )
    crps = result.report.predictor("ppc").score("crps")
    assert crps.mean == pytest.approx(0.5 * (7.0 / 18.0 + 0.0), abs=TOL)
    assert result.report.predictor("ppc").overlap.label == "unverified"


def test_a_discrete_law_pit_and_reliability_equal_the_hand_values() -> None:
    n = 5
    result = pa.assess(
        pa.Observations(TARGET, pa.keys("cohort", range(n)), "category", [1, 0, 0, 0, 0]),
        [predictor("event", [pa.Bernoulli(0.2)] * n)],
        pa.ValidationDesign.held_out_split(),
        scores=("brier", "log_score"),
        dependence=pa.Dependence.iid(),
        pit_bins=2,
    )
    d = result.report.predictor("event").diagnostics
    assert d.reliability is not None
    assert d.reliability[2].count == 5
    assert d.reliability[2].observed_frequency == pytest.approx(0.2, abs=TOL)
    assert d.pit is not None
    # Each non-event has PIT interval (0, 0.8): mass 0.5/0.8 in the lower of two bins.
    assert d.pit.counts[0] == pytest.approx(4 * 0.5 / 0.8, abs=TOL)
    assert result.report.predictor("event").score("brier").mean == pytest.approx(
        (0.64 + 4 * 0.04) / 5, abs=TOL
    )


# --------------------------------------------------------------------------- artifact


def artifact_assessment(seed: int = 21, shift: float = 0.0) -> pa.PredictiveAssessment:
    n = 30
    rng = random.Random(seed)
    values = [rng.gauss(0.0, 1.0) for _ in range(n)]
    return pa.assess(
        observed(values),
        [
            predictor("a", [pa.Gaussian(0.0, 1.0)] * n),
            predictor("b", [pa.Gaussian(0.3 + shift, 1.2)] * n),
            predictor("sample", [pa.EmpiricalSample([-1.0, 0.0, 0.5, 1.0])] * n),
        ],
        simulation(),
        scores=("crps",),
        dependence=pa.Dependence.iid(),
    )


def test_a_consumer_rescores_every_case_and_reproduces_the_report() -> None:
    result = artifact_assessment()
    artifact = result.export("assessment-test")
    consumed = pa.consume_predictive_assessment_artifact(artifact, result.identity)
    assert consumed.report == result.report
    assert consumed.identity == result.identity
    for name in ("a", "b", "sample"):
        assert (
            consumed.report.predictor(name).score("crps").mean
            == result.report.predictor(name).score("crps").mean
        )
    # Same inputs, same bytes: the artifact is deterministic.
    assert artifact_assessment().export("assessment-test") == artifact


def test_an_assessment_of_other_laws_fails_the_retained_identity() -> None:
    original = artifact_assessment()
    other = artifact_assessment(shift=0.25)
    assert other.identity != original.identity
    error = refused(
        lambda: pa.consume_predictive_assessment_artifact(other.export(), original.identity)
    )
    expect(error, "route_not_supported", "predictive_assessment.wrong_contract")
    # An identity edited in transit is refused the same way.
    wire = dict(original.identity.wire)
    wire["configuration"] = "0" * 64
    error = refused(lambda: pa.consume_predictive_assessment_artifact(original.export(), wire))
    expect(error, "route_not_supported", "predictive_assessment.wrong_contract")


def test_a_damaged_artifact_is_refused_and_never_consumed() -> None:
    original = artifact_assessment()
    artifact = original.export()
    for position in (len(artifact) // 2, len(artifact) // 3, len(artifact) - 5):
        flipped = bytearray(artifact)
        flipped[position] ^= 0xFF
        try:
            consumed = pa.consume_predictive_assessment_artifact(bytes(flipped))
        except PredictiveAssessmentRefusal as error:
            assert_registered_refusal(error)
            assert error.detail.startswith("predictive_assessment.")
        else:  # a byte the container does not cover must not change what is consumed
            assert consumed.identity == original.identity
    error = refused(
        lambda: pa.consume_predictive_assessment_artifact(artifact[: len(artifact) // 2])
    )
    assert error.reason_code == "invalid_argument"
    bounded = refused(lambda: pa.consume_predictive_assessment_artifact(artifact, max_cases=5))
    assert bounded.detail.startswith("predictive_assessment.")
    with pytest.raises(CausalTypeError):
        pa.consume_predictive_assessment_artifact("not bytes")  # type: ignore[arg-type]


def test_a_refused_assessment_exports_nothing() -> None:
    error = refused(
        lambda: pa.assess(
            observed([0.1, 0.2]),
            [predictor("cb", [pa.Gaussian(0.0, 1.0)] * 2, source=pa.LawSource.callback("closure"))],
            simulation(),
            scores=("crps",),
        )
    )
    assert error.detail == "predictive_assessment.law_callback_not_built"


def test_listing_the_cases_in_another_order_gives_the_same_assessment() -> None:
    n = 12
    values = normals(n, 8)
    keys = pa.keys("cohort", range(n))
    laws = [pa.Gaussian(0.05 * i, 1.0 + 0.02 * i) for i in range(n)]

    def run(order: list[int]) -> pa.PredictiveAssessment:
        return pa.assess(
            pa.Observations(TARGET, [keys[i] for i in order], "real", [values[i] for i in order]),
            [predictor("a", [laws[i] for i in order])],
            pa.ValidationDesign.held_out_split(),
            scores=("crps", "log_score"),
            dependence=pa.Dependence.iid(),
        )

    forward = run(list(range(n)))
    backward = run(list(reversed(range(n))))
    assert forward.identity == backward.identity
    assert forward.report == backward.report


def test_a_fresh_process_rescores_the_artifact_against_the_retained_identity(tmp_path) -> None:
    result = artifact_assessment()
    path = tmp_path / "assessment.bin"
    path.write_bytes(result.export("fresh"))
    script = (
        "import json, sys\n"
        "from antecedent import predictive_assessment as pa\n"
        "bytes_ = open(sys.argv[1], 'rb').read()\n"
        "identity = json.loads(sys.argv[2])\n"
        "c = pa.consume_predictive_assessment_artifact(bytes_, identity)\n"
        "print(json.dumps({'digest': c.identity.digest, "
        "'means': {p.name: p.score('crps').mean for p in c.report.predictors}}))\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", script, str(path), result.identity.to_json()],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        check=True,
    )
    wire = json.loads(done.stdout)
    assert wire["digest"] == result.identity.digest
    for name, mean in wire["means"].items():
        assert mean == pytest.approx(result.report.predictor(name).score("crps").mean, abs=TOL)
    other = artifact_assessment(shift=0.5).identity
    refusal = subprocess.run(
        [sys.executable, "-c", script, str(path), other.to_json()],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert refusal.returncode != 0
    assert "wrong_contract" in refusal.stderr
