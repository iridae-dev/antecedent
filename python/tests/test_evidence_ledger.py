"""The evidence ledger through the Python facade.

Hand oracles (also recomputed here with numpy, never by native code):

* ``y = (1, 3)``, variances ``(1, 4)``: estimate ``7/5``, variance ``4/5``, ``Q = 4/5``,
  weights ``(4/5, 1/5)``.
* ``y = (1, 3)``, covariance ``[[1, 1], [1, 4]]``: weights ``(1, 0)``, estimate 1, variance 1.
* ``y = (0, 2, 6)``, ``v = (1, 2, 4)``: ``Q = 52/7``, ``tau^2 = 38/7``, DerSimonian-Laird estimate
  ``3330/1457``, variance ``25740/10199``, ``I^2 = 19/26``.
* ``tau = 1`` with a flat mean prior on the same studies: mean ``56/31``, variance ``30/31``; with a
  ``N(0, 1)`` prior on the mean ``56/61`` and ``30/61``.
* ``y = (1, 3)``, ``se = (1, 2)``: the pooled estimate over a common correlation ``rho`` is
  ``(7 - 8 rho)/(5 - 4 rho)`` and its variance ``(4 - 4 rho^2)/(5 - 4 rho)``, which peaks at
  ``rho = 1/2`` with value 1.

The ledger is point-only: no test asserts an interval, and the one that asks for one asserts the
typed refusal.
"""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest
from antecedent import _native
from antecedent import evidence_ledger as el
from antecedent.errors import CausalTypeError, CausalUnsupportedError, CausalValueError
from antecedent.evidence_ledger import EvidenceLedgerRefusal

from _refusal import assert_registered_refusal

TOL = 1e-12


# --------------------------------------------------------------------------- fixtures


def data(label: str, start: int, stop: int) -> el.SourceDeclaration:
    return el.SourceDeclaration(
        label, "observation_set", observations=el.rows("cohort", range(start, stop))
    )


def vendor(label: str = "vendor") -> el.SourceDeclaration:
    """An external claim with undeclared observations: its overlap is unknown."""
    return el.SourceDeclaration(label, "external_claim")


def summary(source: str, estimate: float, variance: float) -> el.StudySummary:
    return el.StudySummary(source, estimate, variance)


def expect(error: EvidenceLedgerRefusal, code: str, detail: str) -> None:
    assert_registered_refusal(error)
    assert error.reason_code == code, (error.reason_code, str(error))
    assert error.detail == detail, (error.detail, str(error))


def refused(call) -> EvidenceLedgerRefusal:
    with pytest.raises(EvidenceLedgerRefusal) as caught:
        call()
    assert isinstance(caught.value, CausalUnsupportedError)
    return caught.value


def inverse_variance(y, v):
    y, v = np.asarray(y, float), np.asarray(v, float)
    w = 1.0 / v
    estimate = float((w * y).sum() / w.sum())
    return estimate, float(1.0 / w.sum()), float((w * (y - estimate) ** 2).sum()), w / w.sum()


def dersimonian_laird(y, v):
    y, v = np.asarray(y, float), np.asarray(v, float)
    w = 1.0 / v
    est = (w * y).sum() / w.sum()
    q = float((w * (y - est) ** 2).sum())
    c = w.sum() - (w**2).sum() / w.sum()
    tau2 = max(0.0, (q - (len(y) - 1)) / c)
    wr = 1.0 / (v + tau2)
    mean = float((wr * y).sum() / wr.sum())
    i2 = max(0.0, (q - (len(y) - 1)) / q) if q > 0 else 0.0
    return mean, float(1.0 / wr.sum()), tau2, q, i2


def reml_tau2(y, v) -> float:
    """Maximize the restricted likelihood over tau^2 by golden-section search."""
    y, v = np.asarray(y, float), np.asarray(v, float)

    def loglik(t2: float) -> float:
        w = 1.0 / (v + t2)
        mu = (w * y).sum() / w.sum()
        return float(
            -0.5 * np.log(v + t2).sum() - 0.5 * np.log(w.sum()) - 0.5 * (w * (y - mu) ** 2).sum()
        )

    lo, hi = 0.0, 1_000.0
    phi = (np.sqrt(5.0) - 1.0) / 2.0
    for _ in range(200):
        a, b = hi - phi * (hi - lo), lo + phi * (hi - lo)
        if loglik(a) < loglik(b):
            lo = a
        else:
            hi = b
    t2 = 0.5 * (lo + hi)
    return t2 if loglik(t2) >= loglik(0.0) else 0.0


class Scenario:
    """Three disjoint studies, pools over pairs and all three, a citation of one pool, and a
    fitted model built from study 1 pooled with study 3."""

    def __init__(self) -> None:
        ledger = el.EvidenceLedger()
        self.ledger = ledger
        self.s1 = ledger.add_source(data("s1", 0, 10))
        self.s2 = ledger.add_source(data("s2", 10, 20))
        self.s3 = ledger.add_source(data("s3", 20, 30))
        self.r_s1_s2 = ledger.pool_summaries(
            "s1_s2", [summary(self.s1, 1.0, 1.0), summary(self.s2, 3.0, 4.0)]
        ).result
        self.r_s2_s3 = ledger.pool_summaries(
            "s2_s3", [summary(self.s2, 3.0, 4.0), summary(self.s3, 2.0, 1.0)]
        ).result
        self.r_all = ledger.pool_summaries(
            "all",
            [summary(self.s1, 0.0, 1.0), summary(self.s2, 2.0, 2.0), summary(self.s3, 6.0, 4.0)],
        ).result
        self.r_cite = ledger.reuse_result("cite_s1_s2", self.r_s1_s2)
        self.model = ledger.add_source(
            el.SourceDeclaration(
                "model_on_s1",
                "fitted_model",
                lineage=el.Lineage.derived_from_used_data([self.s1]),
            )
        )
        self.r_model = ledger.pool_summaries(
            "model_s3", [summary(self.model, 0.5, 1.0), summary(self.s3, 2.0, 1.0)]
        ).result


# --------------------------------------------------------------------------- overlap and pooling


def test_a_pooled_value_matches_the_hand_calculation_and_mass_counts_each_observation_once():
    ledger = el.EvidenceLedger()
    s1 = ledger.add_source(data("study_1", 0, 40))
    s2 = ledger.add_source(data("study_2", 40, 90))
    outcome = ledger.pool_summaries("pool", [summary(s1, 1.0, 1.0), summary(s2, 3.0, 4.0)])
    estimate, variance, q, weights = inverse_variance([1.0, 3.0], [1.0, 4.0])
    assert estimate == pytest.approx(1.4, abs=TOL)
    assert isinstance(outcome.value, el.FixedEffect)
    assert outcome.value.estimate == pytest.approx(estimate, abs=TOL)
    assert outcome.value.variance == pytest.approx(variance, abs=TOL)
    assert outcome.value.heterogeneity_q == pytest.approx(q, abs=TOL)
    assert outcome.value.weights == pytest.approx(tuple(weights), abs=TOL)
    assert outcome.value.degrees_of_freedom == 1
    assert (outcome.claim_class, outcome.claim_label) == ("point_only", "point_only")
    assert outcome.mass == el.MassAccount(
        sources=2, declared_slots=90, distinct_observations=90, repeated_slots=0, unknown_sources=()
    )
    assert ledger.synthesis(outcome.result) == outcome.value
    # No field of a pooled value is an interval, a posterior or a coverage statement.
    assert not any(
        hasattr(outcome.value, name)
        for name in ("interval", "lower", "upper", "posterior", "coverage")
    )

    # The account of two disjoint studies, and of an overlapping pair.
    account = ledger.account_observations([s1, s2])
    assert (account.pairs[0].overlap, account.pairs[0].relation) == ("independent", "independent")
    s4 = ledger.add_source(data("study_4", 80, 130))
    account = ledger.account_observations([s2, s4])
    assert account.pairs[0].overlap == "shared_observations"
    assert account.pairs[0].shared == 10
    assert account.pairs[0].relation == "shared_data"
    # 50 + 50 slots over 90 distinct observations: 10 would be double counted.
    assert account.mass.declared_slots == 100
    assert account.mass.distinct_observations == 90
    assert account.mass.repeated_slots == 10


def test_overlap_refusals_precede_the_numbers_and_leave_the_session_unchanged():
    ledger = el.EvidenceLedger()
    s3 = ledger.add_source(data("s3", 90, 120))
    s4 = ledger.add_source(data("s4", 110, 150))
    claim = ledger.add_source(vendor())
    head, state = ledger.head, ledger.state_digest

    # Shared rows declared independent: the numbers are not even looked at (these variances
    # would be refused).
    error = refused(
        lambda: ledger.pool_summaries("p", [summary(s3, 1.0, -1.0), summary(s4, 3.0, -1.0)])
    )
    expect(error, "evidence_double_counted", "evidence_ledger.shared_observations")
    # Undeclared observations are unknown, never independent.
    error = refused(
        lambda: ledger.pool_summaries("p", [summary(s3, 1.0, 1.0), summary(claim, 3.0, 4.0)])
    )
    expect(error, "evidence_overlap_unknown", "evidence_ledger.overlap_unknown")
    # A diagonal covariance over shared rows is the same double count.
    error = refused(
        lambda: ledger.pool_summaries(
            "p", [summary(s3, 1.0, 1.0), summary(s4, 3.0, 4.0)], covariance=[1.0, 0.0, 0.0, 4.0]
        )
    )
    expect(error, "evidence_double_counted", "evidence_ledger.diagonal_covariance_shared_rows")
    # One source listed twice.
    error = refused(
        lambda: ledger.pool_summaries("p", [summary(s3, 1.0, 1.0), summary(s3, 2.0, 1.0)])
    )
    expect(error, "evidence_double_counted", "evidence_ledger.repeated_source")
    # None of the refusals changed the session.
    assert (ledger.head, ledger.state_digest) == (head, state)
    assert ledger.results == ()

    # A duplicate re-entering through lineage is refused as a duplicate.
    copy = ledger.add_source(
        el.SourceDeclaration("s3_again", "observation_set", lineage=el.Lineage.duplicate(s3))
    )
    error = refused(
        lambda: ledger.pool_summaries("p", [summary(s3, 1.0, 1.0), summary(copy, 1.0, 1.0)])
    )
    expect(error, "evidence_double_counted", "evidence_ledger.reentry_duplicate")

    # The numbers refuse after the overlap check passes, and the staged registration is
    # discarded with them.
    s5 = ledger.add_source(data("s5", 500, 520))
    before, before_head = ledger.state_digest, ledger.head
    error = refused(
        lambda: ledger.pool_summaries("p", [summary(s3, 1.0, 1.0), summary(s5, 3.0, -4.0)])
    )
    expect(error, "invalid_argument", "evidence_ledger.invalid_declaration")
    assert isinstance(error, ValueError)
    assert (ledger.state_digest, ledger.head) == (before, before_head)
    assert ledger.results == ()
    assert error.sources == () or all(len(s) == 64 for s in error.sources)


def test_a_declared_covariance_with_cross_terms_licenses_the_shared_pair():
    ledger = el.EvidenceLedger()
    s3 = ledger.add_source(data("s3", 90, 120))
    s4 = ledger.add_source(data("s4", 110, 150))
    covariance = np.array([[1.0, 1.0], [1.0, 4.0]])
    outcome = ledger.pool_summaries(
        "shared",
        [summary(s3, 1.0, 1.0), summary(s4, 3.0, 4.0)],
        covariance=covariance.reshape(-1).tolist(),
    )
    # Oracle: generalized least squares 1' S^-1 y / 1' S^-1 1, variance 1 / 1' S^-1 1.
    ones = np.ones(2)
    s_inv_one = np.linalg.solve(covariance, ones)
    weights = s_inv_one / (ones @ s_inv_one)
    estimate = float(weights @ np.array([1.0, 3.0]))
    variance = float(1.0 / (ones @ s_inv_one))
    assert isinstance(outcome.value, el.FixedEffect)
    assert outcome.value.estimate == pytest.approx(estimate, abs=TOL)
    assert outcome.value.estimate == pytest.approx(1.0, abs=TOL)
    assert outcome.value.variance == pytest.approx(variance, abs=TOL)
    assert outcome.value.weights == pytest.approx((1.0, 0.0), abs=TOL)
    # The shared rows are counted once: 30 + 40 slots, 60 distinct.
    assert (outcome.mass.declared_slots, outcome.mass.distinct_observations) == (70, 60)
    assert outcome.mass.repeated_slots == 10
    record = ledger.result(outcome.result)
    assert record is not None and record.dependence_declared

    # The covariance must be a covariance: eigenvalues 3 and -1 refuse, and the wrong shape is
    # refused before the ledger sees it.
    s5 = ledger.add_source(data("s5", 500, 520))
    s6 = ledger.add_source(data("s6", 600, 620))
    error = refused(
        lambda: ledger.pool_summaries(
            "bad", [summary(s5, 1.0, 1.0), summary(s6, 2.0, 1.0)], covariance=[1.0, 2.0, 2.0, 1.0]
        )
    )
    expect(error, "invalid_argument", "evidence_ledger.invalid_declaration")
    error = refused(
        lambda: ledger.pool_summaries(
            "short", [summary(s5, 1.0, 1.0), summary(s6, 2.0, 1.0)], covariance=[1.0, 0.0, 4.0]
        )
    )
    expect(error, "invalid_argument", "evidence_ledger.invalid_declaration")
    assert len(ledger.results) == 1


def test_pooling_is_invariant_to_the_order_of_the_studies():
    y, v = [0.0, 2.0, 6.0], [1.0, 2.0, 4.0]

    def build(order):
        ledger = el.EvidenceLedger()
        ids = [
            ledger.add_source(data("a", 0, 10)),
            ledger.add_source(data("b", 10, 20)),
            ledger.add_source(data("c", 20, 30)),
        ]
        return ledger.pool_summaries("p", [summary(ids[i], y[i], v[i]) for i in order])

    a, b = build([0, 1, 2]), build([2, 0, 1])
    # The result identity is a function of the source set, not its order.
    assert a.result == b.result
    assert a.value.estimate == pytest.approx(b.value.estimate, abs=TOL)
    assert a.value.variance == pytest.approx(b.value.variance, abs=TOL)
    estimate, variance, _q, _w = inverse_variance(y, v)
    assert a.value.estimate == pytest.approx(estimate, abs=TOL)
    assert a.value.variance == pytest.approx(variance, abs=TOL)
    assert estimate == pytest.approx(10.0 / 7.0, abs=TOL)


# --------------------------------------------------------------------------- random effects, borrowing, range


def three_studies(ledger: el.EvidenceLedger, y=(0.0, 2.0, 6.0), v=(1.0, 2.0, 4.0)):
    ids = [ledger.add_source(data(f"study_{i}", i * 10, (i + 1) * 10)) for i in range(3)]
    return [summary(i, a, b) for i, a, b in zip(ids, y, v, strict=True)]


def test_random_effects_match_the_dersimonian_laird_derivation_and_the_reml_oracle():
    ledger = el.EvidenceLedger()
    studies = three_studies(ledger)
    y, v = [0.0, 2.0, 6.0], [1.0, 2.0, 4.0]
    outcome = ledger.random_effects("re", studies, method="dersimonian_laird")
    fit = outcome.value
    assert isinstance(fit, el.RandomEffects)
    mean, variance, tau2, q, i2 = dersimonian_laird(y, v)
    assert (q, tau2) == (pytest.approx(52.0 / 7.0), pytest.approx(38.0 / 7.0))
    assert fit.q == pytest.approx(52.0 / 7.0, abs=TOL)
    assert fit.degrees_of_freedom == 2
    assert fit.tau_squared == pytest.approx(38.0 / 7.0, abs=TOL)
    assert fit.estimate == pytest.approx(3330.0 / 1457.0, abs=TOL)
    assert fit.variance == pytest.approx(25740.0 / 10199.0, abs=TOL)
    assert fit.i_squared == pytest.approx(19.0 / 26.0, abs=TOL)
    assert fit.estimate == pytest.approx(mean, abs=TOL)
    assert fit.variance == pytest.approx(variance, abs=TOL)
    assert fit.i_squared == pytest.approx(i2, abs=TOL)
    assert fit.method == "dersimonian_laird"
    assert sum(fit.weights) == pytest.approx(1.0, abs=TOL)

    # REML is a different, registered estimator of the same quantity; its oracle is the
    # maximizer of the restricted likelihood.
    reml = ledger.random_effects("re_reml", studies, method="reml").value
    t2 = reml_tau2(y, v)
    assert reml.tau_squared == pytest.approx(t2, abs=1e-6)
    weights = 1.0 / (np.array(v) + t2)
    assert reml.estimate == pytest.approx(float((weights * y).sum() / weights.sum()), abs=1e-6)
    assert reml.variance == pytest.approx(float(1.0 / weights.sum()), abs=1e-6)
    assert reml.q == pytest.approx(52.0 / 7.0, abs=TOL)  # Q is method independent
    assert reml.method == "reml"

    # Fewer than two studies cannot estimate between-study variance.
    lonely = ledger.add_source(data("lonely", 100, 110))
    error = refused(
        lambda: ledger.random_effects(
            "one", [summary(lonely, 1.0, 1.0)], method="dersimonian_laird"
        )
    )
    assert error.reason_code == "invalid_argument"
    with pytest.raises(CausalValueError):
        ledger.random_effects("x", studies, method="moments")


def hierarchical_oracle(y, v, tau_support, tau_prob, mean_prior=None):
    """Exact finite-mixture posterior of the common mean (normal-normal)."""
    y, v = np.asarray(y, float), np.asarray(v, float)
    mixture = []
    for tau, prior_p in zip(tau_support, tau_prob, strict=True):
        cov = np.diag(v + tau**2)
        ones = np.ones(len(y))
        if mean_prior is None:
            w = np.linalg.inv(cov)
            precision = float(ones @ w @ ones)
            mean = float(ones @ w @ y) / precision
            var = 1.0 / precision
            resid = y - mean
            # Flat prior: the marginal likelihood up to a tau-free constant is
            # |V|^-1/2 |1' V^-1 1|^-1/2 exp(-resid' V^-1 resid / 2).
            log_m = (
                -0.5 * np.log(np.linalg.det(cov))
                - 0.5 * np.log(precision)
                - 0.5 * resid @ w @ resid
            )
        else:
            m0, v0 = mean_prior
            marginal_cov = cov + v0 * np.outer(ones, ones)
            diff = y - m0
            log_m = -0.5 * np.log(np.linalg.det(marginal_cov)) - 0.5 * diff @ np.linalg.solve(
                marginal_cov, diff
            )
            w = np.linalg.inv(cov)
            precision = 1.0 / v0 + float(ones @ w @ ones)
            mean = (m0 / v0 + float(ones @ w @ y)) / precision
            var = 1.0 / precision
        mixture.append((np.log(prior_p) + log_m, mean, var))
    logs = np.array([m[0] for m in mixture])
    post = np.exp(logs - logs.max())
    post /= post.sum()
    means = np.array([m[1] for m in mixture])
    vars_ = np.array([m[2] for m in mixture])
    mean = float(post @ means)
    variance = float(post @ (vars_ + means**2) - mean**2)
    return mean, variance, post


def test_hierarchical_borrowing_matches_the_closed_form_and_the_finite_mixture():
    ledger = el.EvidenceLedger()
    studies = three_studies(ledger)
    y, v = [0.0, 2.0, 6.0], [1.0, 2.0, 4.0]
    single = el.TauPrior((1.0,), (1.0,))
    flat = ledger.hierarchical_borrowing(
        "flat", studies, mean_prior=el.MeanPrior.flat(), tau_prior=single
    )
    assert isinstance(flat.value, el.HierarchicalPosterior)
    assert flat.value.mean == pytest.approx(56.0 / 31.0, abs=TOL)
    assert flat.value.variance == pytest.approx(30.0 / 31.0, abs=TOL)
    mean, variance, _ = hierarchical_oracle(y, v, [1.0], [1.0])
    assert flat.value.mean == pytest.approx(mean, abs=1e-10)
    assert flat.value.variance == pytest.approx(variance, abs=1e-10)
    assert flat.claim_label == "point_only"

    normal = ledger.hierarchical_borrowing(
        "normal", studies, mean_prior=el.MeanPrior.normal(0.0, 1.0), tau_prior=single
    ).value
    assert normal.mean == pytest.approx(56.0 / 61.0, abs=TOL)
    assert normal.variance == pytest.approx(30.0 / 61.0, abs=TOL)

    # A two-point prior on tau: the posterior is the exact finite mixture of the closed forms.
    mixture = ledger.hierarchical_borrowing(
        "mixture",
        studies,
        mean_prior=el.MeanPrior.normal(0.0, 1.0),
        tau_prior=el.TauPrior((0.5, 1.0), (0.5, 0.5)),
    ).value
    mean, variance, post = hierarchical_oracle(y, v, [0.5, 1.0], [0.5, 0.5], (0.0, 1.0))
    assert mixture.mean == pytest.approx(mean, abs=1e-10)
    assert mixture.variance == pytest.approx(variance, abs=1e-10)
    assert mixture.tau_posterior == pytest.approx(tuple(post), abs=1e-10)
    assert sum(mixture.tau_posterior) == pytest.approx(1.0, abs=TOL)


def test_a_prior_built_from_the_likelihoods_data_is_double_counting():
    ledger = el.EvidenceLedger()
    studies = three_studies(ledger)
    # A power prior made from rows study 1 already contributed.
    power = ledger.add_source(
        el.SourceDeclaration("power_prior", "prior", observations=el.rows("cohort", range(0, 5)))
    )

    def borrow(priors):
        return ledger.hierarchical_borrowing(
            "borrow",
            studies,
            mean_prior=el.MeanPrior.normal(0.0, 1.0),
            tau_prior=el.TauPrior((0.5, 1.0), (0.5, 0.5)),
            priors=priors,
        )

    error = refused(lambda: borrow([power]))
    assert error.reason_code == "evidence_double_counted"
    assert ledger.results == ()

    # An elicited prior has an empty observation set and overlaps nothing.
    elicited = ledger.add_source(el.SourceDeclaration("elicited", "prior", observations=()))
    outcome = borrow([elicited])
    assert sum(outcome.value.tau_posterior) == pytest.approx(1.0, abs=TOL)
    # The numbers are the session's; the artifact format carries one summary per source, so this
    # result's numbers are listed as not carried rather than dropped silently.
    export = ledger.artifact("borrowing")
    assert export.metadata["carried"] == ()
    assert list(export.metadata["not_carried"]) == [outcome.result]
    assert "prior" in export.metadata["not_carried"][outcome.result]


def envelope_oracle(estimates, standard_errors, lo, hi, points=40_001):
    y, se = np.asarray(estimates, float), np.asarray(standard_errors, float)
    ones = np.ones(len(y))
    rhos = np.linspace(lo, hi, points)
    est = np.empty(points)
    var = np.empty(points)
    for i, rho in enumerate(rhos):
        corr = np.full((len(y), len(y)), rho)
        np.fill_diagonal(corr, 1.0)
        cov = np.outer(se, se) * corr
        w = np.linalg.solve(cov, ones)
        est[i] = (w @ y) / (w @ ones)
        var[i] = 1.0 / (w @ ones)
    return rhos, est, var


def test_the_dependence_range_is_an_exact_labelled_envelope():
    ledger = el.EvidenceLedger()
    s3 = ledger.add_source(data("s3", 0, 30))
    claim = ledger.add_source(vendor())
    studies = [el.StudyEstimate(s3, 1.0, 1.0), el.StudyEstimate(claim, 3.0, 2.0)]
    outcome = ledger.dependence_range("range", studies, rho_lo=0.0, rho_hi=0.9)
    assert outcome.claim_class == "assumption_range"
    assert outcome.claim_label == "assumption_range"
    env = outcome.value
    assert isinstance(env, el.DependenceEnvelope)

    # Closed forms of this fixture.
    rho = np.linspace(0.0, 0.9, 1001)
    assert env.estimate_max == pytest.approx((7 - 8 * 0.0) / (5 - 4 * 0.0), abs=1e-9)
    assert env.estimate_min == pytest.approx((7 - 8 * 0.9) / (5 - 4 * 0.9), abs=1e-9)
    assert env.estimate_min == pytest.approx(-1.0 / 7.0, abs=1e-9)
    assert env.variance_max == pytest.approx(1.0, abs=1e-9)
    assert env.variance_min == pytest.approx(0.76 / 1.4, abs=1e-9)
    assert env.rho_at_estimate_max == pytest.approx(0.0, abs=1e-9)
    assert env.rho_at_estimate_min == pytest.approx(0.9, abs=1e-9)
    assert env.rho_at_variance_max == pytest.approx(0.5, abs=1e-9)  # s_min / s_max
    closed_form = (4 - 4 * rho**2) / (5 - 4 * rho)
    assert closed_form.max() == pytest.approx(1.0, abs=1e-6)

    # A brute-force scan over the declared range agrees with the exact extremes.
    rhos, est, var = envelope_oracle([1.0, 3.0], [1.0, 2.0], 0.0, 0.9)
    assert env.estimate_max == pytest.approx(est.max(), abs=1e-7)
    assert env.estimate_min == pytest.approx(est.min(), abs=1e-7)
    assert env.variance_max == pytest.approx(var.max(), abs=1e-7)
    assert env.variance_min == pytest.approx(var.min(), abs=1e-7)
    assert abs(env.rho_at_variance_max - rhos[var.argmax()]) < 1e-3

    # The unknown source stays unknown in the accounting.
    record = ledger.result(outcome.result)
    assert record is not None and claim in record.mass.unknown_sources
    # No interval, posterior or coverage field exists on the envelope.
    assert not any(hasattr(env, name) for name in ("interval", "lower", "upper", "posterior"))

    # A range that is not strictly inside the definite interval refuses.
    for lo, hi in [(0.3, 0.3), (0.0, 1.0), (-1.5, 0.2)]:
        error = refused(
            lambda lo=lo, hi=hi: ledger.dependence_range("bad", studies, rho_lo=lo, rho_hi=hi)
        )
        assert error.reason_code == "invalid_argument", (lo, hi)


# --------------------------------------------------------------------------- other operations


def transport_identity(target: str = "target") -> el.TransportIdentity:
    return el.TransportIdentity(
        source_populations=("trial_a", "trial_b"),
        target_population=target,
        selection_diagram=bytes([7] * 32).hex(),
        assumptions=("s-admissible adjustment set",),
    )


def test_a_transported_claim_is_never_native_target_data():
    ledger = el.EvidenceLedger()
    native = ledger.add_source(data("target_cohort", 0, 30))
    transported = ledger.add_source(
        el.SourceDeclaration(
            "effect_from_trials",
            "external_claim",
            observations=el.rows("trials", range(100, 160)),
            transport=transport_identity(),
        )
    )
    result = ledger.transport_claim("transport", transported=[transported], native=[native])
    record = ledger.result(result)
    assert record is not None and len(record.transport_identities) == 1
    # Accounting-only results carry no numbers.
    assert ledger.synthesis(result) is None

    # The same source native and transported is a category error.
    error = refused(
        lambda: ledger.transport_claim("bad", transported=[transported], native=[transported])
    )
    expect(error, "evidence_double_counted", "evidence_ledger.native_and_transported")
    # A claim without its transport identity cannot be transported.
    bare = ledger.add_source(data("bare", 300, 310))
    error = refused(lambda: ledger.transport_claim("bad", transported=[bare], native=[native]))
    expect(error, "invalid_argument", "evidence_ledger.transport_identity_missing")
    # Another transported claim is not native evidence.
    other = ledger.add_source(
        el.SourceDeclaration(
            "other_transported",
            "external_claim",
            observations=el.rows("trials", range(400, 410)),
            transport=transport_identity(),
        )
    )
    error = refused(
        lambda: ledger.transport_claim("bad", transported=[transported], native=[other])
    )
    expect(error, "invalid_argument", "evidence_ledger.transported_as_native")


def test_citing_a_result_counts_its_observations_once_and_refuses_a_stale_one():
    ledger = el.EvidenceLedger()
    a = ledger.add_source(data("a", 0, 30))
    b = ledger.add_source(data("b", 30, 50))
    c = ledger.add_source(data("c", 50, 60))
    pooled = ledger.pool_summaries(
        "pooled", [summary(a, 1.0, 1.0), summary(b, 2.0, 1.0), summary(c, 3.0, 1.0)]
    )
    assert pooled.mass.distinct_observations == 60
    first = ledger.reuse_result("second_analysis", pooled.result)
    second = ledger.reuse_result("third_analysis", pooled.result)
    assert first != second
    for cited in (first, second):
        record = ledger.result(cited)
        # Each citation reports the once-counted mass, not a multiple of it.
        assert record is not None and record.mass == pooled.mass
    # A stale result is recomputed before it is cited.
    ledger.revise_source(el.SourceRevision.update(c, data("c_more", 50, 70)))
    assert ledger.standing(pooled.result) == "recompute"
    error = refused(lambda: ledger.reuse_result("cite", pooled.result))
    expect(error, "evidence_source_withdrawn", "evidence_ledger.result_not_valid")


def test_repeated_observation_use_is_flagged_and_stays_flagged_when_cited():
    ledger = el.EvidenceLedger()
    a = ledger.add_source(data("a", 0, 30))
    repeated = ledger.repeat_observations("interim_then_final", [a], "final")
    record = ledger.result(repeated)
    assert record is not None and record.data_reuse
    assert record.mass.distinct_observations == 30
    cited = ledger.reuse_result("cite", repeated)
    cited_record = ledger.result(cited)
    assert cited_record is not None and cited_record.data_reuse
    # A stage with no name is not a declaration.
    error = refused(lambda: ledger.repeat_observations("nameless", [a], ""))
    expect(error, "invalid_argument", "evidence_ledger.operation_arity")
    # Undeclared observations cannot be accounted: their reuse is unknown.
    claim = ledger.add_source(vendor())
    error = refused(lambda: ledger.repeat_observations("vendor_again", [claim], "final"))
    assert error.reason_code == "evidence_overlap_unknown"


def test_a_pooled_interval_inside_the_measured_scope_is_bound_to_its_record():
    ledger = el.EvidenceLedger()
    sources = [ledger.add_source(data(f"study_{i}", i * 100, (i + 1) * 100)) for i in range(5)]

    def summaries(shift: float, count: int = 5):
        return [summary(sources[i], 0.5 + shift * (i - 2), 0.01 + 0.002 * i) for i in range(count)]

    fixed = ledger.pool_summaries("fixed", summaries(0.01)).result
    random = ledger.random_effects("random", summaries(0.05)).result
    for result, method in ((fixed, "wald_z"), (random, "hksj_t")):
        interval = ledger.pooled_interval(result)
        assert interval["method"] == method
        assert interval["lower"] < interval["estimate"] < interval["upper"]
        assert interval["level"] == 0.95
        assert interval["calibration"]["status"] == "calibrated"
        assert interval["calibration"]["record_id"]
    three = ledger.pool_summaries("three", summaries(0.01, 3)).result
    error = refused(lambda: ledger.pooled_interval(three))
    expect(error, "cell_not_licensed", "evidence_ledger.interval_not_measured")


def test_a_pooled_interval_refuses_without_a_measured_record():
    scenario = Scenario()
    error = refused(lambda: scenario.ledger.pooled_interval(scenario.r_all))
    assert error.reason_code == "cell_not_licensed"
    assert error.detail in {
        "evidence_ledger.interval_not_measured",
        "evidence_ledger.interval_not_licensed",
    }
    error = refused(lambda: scenario.ledger.pooled_interval("ab" * 32))
    assert error.reason_code == "invalid_argument"
    with pytest.raises(CausalTypeError):
        scenario.ledger.pooled_interval("")


# --------------------------------------------------------------------------- revision, stale use, refresh


def test_withdrawing_a_source_classifies_dependents_and_refuses_their_use():
    s = Scenario()
    report = s.ledger.revise_source(el.SourceRevision.withdraw(s.s1, "consent withdrawn"))
    assert report.revalidation[s.r_s1_s2] == "withdrawn"
    assert report.revalidation[s.r_cite] == "withdrawn"
    assert report.revalidation[s.r_s2_s3] == "still_valid"
    assert report.revalidation[s.r_all] == "recompute"
    # The model derived from s1 is lost through lineage.
    assert report.revalidation[s.r_model] == "withdrawn"
    assert report.retired_source == s.s1

    # Stale numbers are not served; valid ones are.
    for stale in (s.r_s1_s2, s.r_cite, s.r_all, s.r_model):
        expect(
            refused(lambda stale=stale: s.ledger.synthesis(stale)),
            "evidence_source_withdrawn",
            "evidence_ledger.result_not_valid",
        )
    assert s.ledger.synthesis(s.r_s2_s3) is not None
    # New work cannot rest on the withdrawn source, directly or by lineage.
    expect(
        refused(
            lambda: s.ledger.pool_summaries(
                "again", [summary(s.s1, 1.0, 1.0), summary(s.s3, 2.0, 1.0)]
            )
        ),
        "evidence_source_withdrawn",
        "evidence_ledger.source_withdrawn",
    )
    expect(
        refused(
            lambda: s.ledger.pool_summaries(
                "via_model", [summary(s.model, 0.5, 1.0), summary(s.s2, 3.0, 4.0)]
            )
        ),
        "evidence_source_withdrawn",
        "evidence_ledger.source_withdrawn",
    )
    expect(
        refused(lambda: s.ledger.reuse_result("cite_again", s.r_s1_s2)),
        "evidence_source_withdrawn",
        "evidence_ledger.result_not_valid",
    )

    # The recompute on the remaining studies names the result it replaces and the original is
    # preserved but unusable.
    recomputed = s.ledger.pool_summaries(
        "all_minus_s1", [summary(s.s2, 2.0, 2.0), summary(s.s3, 6.0, 4.0)], replaces=s.r_all
    )
    assert s.ledger.standing(recomputed.result) == "still_valid"
    assert s.ledger.standing(s.r_all) == "withdrawn"
    assert s.ledger.result(s.r_all) is not None
    refused(lambda: s.ledger.synthesis(s.r_all))
    # y = (2, 6), v = (2, 4): estimate (1 + 1.5) / (3/4) = 10/3, variance 4/3.
    estimate, variance, _q, _w = inverse_variance([2.0, 6.0], [2.0, 4.0])
    assert recomputed.value.estimate == pytest.approx(10.0 / 3.0, abs=TOL)
    assert recomputed.value.estimate == pytest.approx(estimate, abs=TOL)
    assert recomputed.value.variance == pytest.approx(variance, abs=TOL)
    assert recomputed.value.variance == pytest.approx(4.0 / 3.0, abs=TOL)
    assert s.ledger.result(recomputed.result).replaces == s.r_all


def test_superseding_a_model_marks_dependents_for_recompute():
    s = Scenario()
    successor = el.SourceDeclaration(
        "model_v2_on_s1", "fitted_model", lineage=el.Lineage.derived_from_used_data([s.s1])
    )
    report = s.ledger.revise_source(el.SourceRevision.supersede(s.model, successor))
    successor_id = report.created_source
    assert successor_id is not None
    assert report.revalidation[s.r_model] == "recompute"
    assert report.revalidation[s.r_all] == "still_valid"
    # A result that needs recomputation is refused, not served.
    expect(
        refused(lambda: s.ledger.synthesis(s.r_model)),
        "evidence_source_withdrawn",
        "evidence_ledger.result_not_valid",
    )
    # The superseded model cannot enter new results, and the refusal names its successor.
    error = refused(
        lambda: s.ledger.pool_summaries(
            "stale", [summary(s.model, 0.5, 1.0), summary(s.s2, 3.0, 4.0)]
        )
    )
    expect(error, "evidence_source_withdrawn", "evidence_ledger.source_superseded")
    assert successor_id in error.message
    # Recompute on the successor.
    recomputed = s.ledger.pool_summaries(
        "model_s3",
        [summary(successor_id, 0.7, 1.0), summary(s.s3, 2.0, 1.0)],
        replaces=s.r_model,
    )
    assert s.ledger.standing(recomputed.result) == "still_valid"
    assert s.ledger.synthesis(recomputed.result) is not None


def test_a_result_derived_from_a_valid_result_is_reached_by_a_later_withdrawal():
    s = Scenario()
    declaration = s.ledger.declare_from_result(s.r_all, "claim_from_all", "external_claim")
    assert declaration.kind == "external_claim"
    assert declaration.lineage.wire["kind"] == "derived_from_used_data"
    assert set(declaration.lineage.wire["inputs"]) == {s.s1, s.s2, s.s3}
    claim = s.ledger.add_source(declaration)
    report = s.ledger.revise_source(el.SourceRevision.withdraw(s.s3, "retracted"))
    assert report.revalidation  # classified
    # The derived claim is lost through lineage: it cannot enter new results.
    other = s.ledger.add_source(data("fresh", 700, 710))
    error = refused(
        lambda: s.ledger.pool_summaries(
            "via_claim", [summary(claim, 1.0, 1.0), summary(other, 2.0, 1.0)]
        )
    )
    assert error.detail == "evidence_ledger.source_withdrawn"
    # A kind that is not a model or claim is not a derivation of a result.
    error = refused(lambda: s.ledger.declare_from_result(s.r_s1_s2, "x", "observation_set"))
    assert error.reason_code == "invalid_argument"


def test_refresh_is_all_or_nothing_and_keeps_untouched_results():
    s = Scenario()
    head, state = s.ledger.head, s.ledger.state_digest
    result_count = len(s.ledger.results)

    refreshed = s.ledger.refresh(
        [
            el.SourceChange.add(data("s4", 200, 230)),
            el.SourceChange.revise(el.SourceRevision.withdraw(s.s1, "consent withdrawn")),
        ]
    )
    assert len(refreshed.reports) == 2
    # Kept: results whose sources are untouched, with their numbers.
    assert refreshed.session.synthesis(s.r_s2_s3) is not None
    # Not served: results whose input changed, until a recompute replaces them.
    refused(lambda: refreshed.session.synthesis(s.r_all))
    assert refreshed.classification[s.r_all] == "recompute"
    assert refreshed.classification[s.r_s1_s2] == "withdrawn"
    # The operation history is chained and the new head differs.
    assert refreshed.session.head != head
    assert len(refreshed.session.results) == result_count
    # The session refreshed from is untouched.
    assert (s.ledger.head, s.ledger.state_digest) == (head, state)
    assert s.ledger.synthesis(s.r_all) is not None

    # One refused change refuses the batch: the valid addition is not kept.
    unknown = "cd" * 32
    error = refused(
        lambda: s.ledger.refresh(
            [
                el.SourceChange.add(data("s5", 300, 320)),
                el.SourceChange.revise(el.SourceRevision.withdraw(unknown, "x")),
            ]
        )
    )
    expect(error, "invalid_argument", "evidence_ledger.unknown_source")
    assert s.ledger.state_digest == state
    assert len(s.ledger.sources) == 4


# --------------------------------------------------------------------------- artifact


def test_the_artifact_replays_and_a_consumer_refuses_another_history():
    s = Scenario()
    export = s.ledger.artifact("ledger-1")
    assert len(export.metadata["carried"]) == 4
    assert export.metadata["not_carried"] == {}
    consumed = el.consume_evidence_ledger_artifact(export.data, export.expectation.identity)
    assert consumed.identity == export.expectation.identity
    assert consumed.head == s.ledger.head
    assert consumed.state_digest == s.ledger.state_digest
    assert [r.id for r in consumed.results] == [r.id for r in s.ledger.results]
    # The consumer recomputed the numbers: the all-studies pool is a fixed-effect pool of
    # y = (0, 2, 6), v = (1, 2, 4): 10/7 and 4/7.
    all_studies = consumed.synthesis_of(s.r_all)
    assert all_studies is not None
    assert all_studies.method == "fixed_effect_inverse_variance"
    estimate, variance, q, _w = inverse_variance([0.0, 2.0, 6.0], [1.0, 2.0, 4.0])
    assert all_studies.values[0] == pytest.approx(estimate, abs=TOL)
    assert all_studies.values[1] == pytest.approx(variance, abs=TOL)
    assert all_studies.values[2] == pytest.approx(q, abs=TOL)
    assert all_studies.values[:3] == pytest.approx((10 / 7, 4 / 7, 52 / 7), abs=TOL)

    # A consumer that retained another identity refuses these bytes.
    other = s.ledger.refresh([el.SourceChange.add(data("extra", 900, 910))])
    other_export = other.session.artifact("ledger-2")
    assert other_export.expectation.identity != export.expectation.identity
    error = refused(
        lambda: el.consume_evidence_ledger_artifact(export.data, other_export.expectation.identity)
    )
    assert error.detail == "evidence_ledger.identity_mismatch"
    # One that holds none still refuses a damaged container, and truncated bytes.
    damaged = bytearray(export.data)
    damaged[len(damaged) // 2] ^= 0x55
    refused(lambda: el.consume_evidence_ledger_artifact(bytes(damaged)))
    refused(lambda: el.consume_evidence_ledger_artifact(export.data[:30]))
    with pytest.raises(CausalTypeError):
        el.consume_evidence_ledger_artifact("not bytes")  # type: ignore[arg-type]


def test_a_stored_value_that_differs_is_a_different_history():
    """A resealed edit of a stored number is a different ledger identity: a consumer that
    retained the original refuses it, and it replays only against its own."""
    original = el.EvidenceLedger()
    studies = three_studies(original, y=(0.0, 2.0, 6.0))
    original.pool_summaries("p", studies)
    export = original.artifact("ledger")

    edited = el.EvidenceLedger()
    studies = three_studies(edited, y=(0.0, 2.0, 7.0))  # same sources, another stored number
    edited.pool_summaries("p", studies)
    forged = edited.artifact("ledger")
    assert forged.expectation.identity != export.expectation.identity
    expect(
        refused(
            lambda: el.consume_evidence_ledger_artifact(forged.data, export.expectation.identity)
        ),
        "route_not_supported",
        "evidence_ledger.identity_mismatch",
    )
    own = el.consume_evidence_ledger_artifact(forged.data, forged.expectation.identity)
    values = own.syntheses[0].values
    assert values[0] == pytest.approx(
        inverse_variance([0.0, 2.0, 7.0], [1.0, 2.0, 4.0])[0], abs=TOL
    )
    # The same history and numbers give the same identity, bit for bit.
    again = el.EvidenceLedger()
    again.pool_summaries("p", three_studies(again, y=(0.0, 2.0, 6.0)))
    assert again.artifact("ledger").expectation.identity == export.expectation.identity
    assert again.head == original.head


def range_ledger(rho_hi: float = 0.9) -> tuple[el.EvidenceLedger, str, str]:
    ledger = el.EvidenceLedger()
    s3 = ledger.add_source(data("s3", 0, 30))
    claim = ledger.add_source(vendor())
    studies = [el.StudyEstimate(s3, 1.0, 1.0), el.StudyEstimate(claim, 3.0, 2.0)]
    outcome = ledger.dependence_range("range", studies, rho_lo=0.0, rho_hi=rho_hi)
    return ledger, outcome.result, claim


def test_a_range_outside_the_definite_interval_or_non_finite_is_refused_and_registers_nothing():
    ledger, _result, claim = range_ledger()
    s3 = next(r.id for r in ledger.sources if r.label == "s3")
    head, state, count = ledger.head, ledger.state_digest, len(ledger.results)
    # Inverted, perfectly correlated, on the boundary -1/(k-1) of the definite interval, a
    # non-finite bound and a non-positive standard error.
    cases = [
        (0.5, 0.2, 2.0),
        (0.0, 1.0, 2.0),
        (-1.0, 0.2, 2.0),
        (0.0, 0.5, 0.0),
    ]
    for lo, hi, se in cases:
        studies = [el.StudyEstimate(s3, 1.0, 1.0), el.StudyEstimate(claim, 3.0, se)]
        error = refused(
            lambda studies=studies, lo=lo, hi=hi: ledger.dependence_range(
                "bad", studies, rho_lo=lo, rho_hi=hi
            )
        )
        assert error.reason_code == "invalid_argument", (lo, hi, se)
    # Non-finite bounds are refused by the Python boundary before any native call.
    pair = [el.StudyEstimate(s3, 1.0, 1.0), el.StudyEstimate(claim, 3.0, 2.0)]
    for lo, hi in [(float("nan"), 0.2), (0.0, float("inf"))]:
        with pytest.raises(CausalValueError):
            ledger.dependence_range("bad", pair, rho_lo=lo, rho_hi=hi)
    assert (ledger.head, ledger.state_digest, len(ledger.results)) == (head, state, count)


def test_an_exported_dependence_range_is_replayed_as_an_assumption_range():
    ledger, result, claim = range_ledger()
    export = ledger.artifact("range-ledger")
    assert export.metadata["carried"] == (result,)
    assert export.metadata["not_carried"] == {}
    consumed = el.consume_evidence_ledger_artifact(export.data, export.expectation.identity)
    # The consumer recomputed the envelope in the method's fixed order: estimate min, rho at
    # min, estimate max, rho at max, variance min, rho at min, variance max, rho at max.
    # Hand values: -1/7 at 0.9, 7/5 at 0, 19/35 at 0.9, and the interior maximum 1 at 1/2.
    replay = consumed.synthesis_of(result)
    assert replay is not None
    assert replay.method == "dependence_envelope"
    assert replay.values == pytest.approx(
        (-1 / 7, 0.9, 7 / 5, 0.0, 19 / 35, 0.9, 1.0, 0.5), abs=1e-9
    )
    # It agrees with the brute-force grid the producer is checked against.
    _rhos, est, var = envelope_oracle([1.0, 3.0], [1.0, 2.0], 0.0, 0.9)
    assert replay.values[0] == pytest.approx(est.min(), abs=1e-7)
    assert replay.values[2] == pytest.approx(est.max(), abs=1e-7)
    assert replay.values[4] == pytest.approx(var.min(), abs=1e-7)
    assert replay.values[6] == pytest.approx(var.max(), abs=1e-7)
    # The label survives the replay, and the unknown source stays unknown.
    record = next(r for r in consumed.results if r.id == result)
    assert record.claim_class == "assumption_range"
    assert claim in record.mass.unknown_sources
    # The same history gives the same identity; another range is another artifact, so a
    # consumer that retained the original refuses it.
    assert (
        range_ledger()[0].artifact("range-ledger").expectation.identity
        == export.expectation.identity
    )
    widened, _, _ = range_ledger(rho_hi=0.95)
    forged = widened.artifact("range-ledger")
    assert forged.expectation.identity != export.expectation.identity
    expect(
        refused(
            lambda: el.consume_evidence_ledger_artifact(forged.data, export.expectation.identity)
        ),
        "route_not_supported",
        "evidence_ledger.identity_mismatch",
    )
    own = el.consume_evidence_ledger_artifact(forged.data, forged.expectation.identity)
    widened_replay = own.syntheses[0]
    assert widened_replay.method == "dependence_envelope"
    assert widened_replay.values[0] == pytest.approx((7 - 8 * 0.95) / (5 - 4 * 0.95), abs=TOL)


def test_a_fresh_process_consumes_the_artifact_against_the_retained_identity(tmp_path):
    s = Scenario()
    export = s.ledger.artifact("fresh")
    path = tmp_path / "artifact.bin"
    path.write_bytes(export.data)
    script = (
        "import json, sys\n"
        "from antecedent import evidence_ledger as el\n"
        "c = el.consume_evidence_ledger_artifact(open(sys.argv[1], 'rb').read(), sys.argv[2])\n"
        "print(json.dumps({'identity': c.identity, 'head': c.head, "
        "'syntheses': [[s.result, s.method, list(s.values)] for s in c.syntheses]}))\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", script, str(path), export.expectation.identity],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        check=True,
    )
    wire = json.loads(done.stdout)
    assert wire["identity"] == export.expectation.identity
    assert wire["head"] == s.ledger.head
    by_result = {r: (m, v) for r, m, v in wire["syntheses"]}
    assert by_result[s.r_all][0] == "fixed_effect_inverse_variance"
    assert by_result[s.r_all][1][0] == pytest.approx(10 / 7, abs=TOL)
    refusal = subprocess.run(
        [sys.executable, "-c", script, str(path), "0" * 64],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert refusal.returncode != 0
    assert "identity_mismatch" in refusal.stderr


# --------------------------------------------------------------------------- cancellation, determinism


def test_cancellation_stops_before_a_synthesis_and_leaves_the_session_unchanged():
    ledger = el.EvidenceLedger()
    s1 = ledger.add_source(data("s1", 0, 10))
    s2 = ledger.add_source(data("s2", 10, 20))
    state = ledger.state_digest
    token = _native.CancellationToken()
    token.cancel()
    error = refused(
        lambda: ledger.pool_summaries(
            "p", [summary(s1, 1.0, 1.0), summary(s2, 3.0, 4.0)], cancel=token
        )
    )
    assert error.reason_code == "cancelled_no_claim"
    assert ledger.state_digest == state
    assert ledger.results == ()
    # The same request under a live context succeeds.
    assert ledger.pool_summaries("p", [summary(s1, 1.0, 1.0), summary(s2, 3.0, 4.0)]).value


def test_identities_are_deterministic():
    one, two = Scenario(), Scenario()
    assert one.ledger.head == two.ledger.head
    assert one.ledger.state_digest == two.ledger.state_digest
    assert (one.s1, one.r_all) == (two.s1, two.r_all)


def test_observation_keys_are_domain_separated():
    assert el.observation_id("cohort", 7) == el.observation_id("cohort", 7)
    assert el.observation_id("cohort", 7) != el.observation_id("cohort", 8)
    assert el.observation_id("cohort", 7) != el.observation_id("other", 7)
    assert len(el.observation_id("cohort", "row-7")) == 64
    # An integer key is its 8-byte little-endian encoding, a string its UTF-8 bytes.
    assert el.observation_id("cohort", 7) == el.observation_id("cohort", (7).to_bytes(8, "little"))
    assert el.observation_id("cohort", "row-7") == el.observation_id("cohort", b"row-7")


def test_rows_declare_row_level_observations_by_domain_and_key():
    declared = el.rows("cohort", range(3))
    assert declared == (("cohort", 0), ("cohort", 1), ("cohort", 2))
    ledger = el.EvidenceLedger()
    first = ledger.add_source(data("first", 0, 10))
    second = ledger.add_source(data("second", 5, 15))
    other_domain = ledger.add_source(
        el.SourceDeclaration(
            "elsewhere", "observation_set", observations=el.rows("registry", range(0, 10))
        )
    )
    # Equal (domain, key) rows are one observation: 5 shared; another domain shares none.
    assert ledger.account_observations([first, second]).pairs[0].shared == 5
    assert ledger.account_observations([first, other_domain]).pairs[0].overlap == "independent"
    negative = el.SourceDeclaration("x", "observation_set", observations=el.rows("cohort", [-1]))
    with pytest.raises(CausalValueError):
        negative.to_wire()
    with pytest.raises(CausalValueError):
        el.SourceDeclaration("x", "not_a_kind")
