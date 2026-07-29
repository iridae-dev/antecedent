"""Nested-section anti-drift tests (P5c).

Both native DTOs — ``AteAnalysisResult`` (static) and ``AnalysisResult``
(temporal) — now carry five nested sections (``identification`` /
``estimate`` / ``posterior`` / ``validation`` / ``performance``) built on the
Rust side alongside their existing flat fields (see ``python/src/lib.rs``,
``ate_api.rs``, ``temporal_api.rs``). These tests are the anti-drift gate for
that change:

- Every nested section field is checked against its flat-field sibling on the
  same raw object (``raw.identification.status == raw.identification_status``
  and so on) — proving the section is a *view* onto the same data, not a
  reinterpretation.
- ``validation.passed`` / ``validation.ran`` are checked against the shared
  aggregate rule (``ran and all(r.passed for r in reports)``, never ``True``
  when nothing ran) on *both* DTO shapes, including the "nothing ran" case.
- Fields the temporal DTO genuinely cannot supply (no overlap report, no
  bootstrap-ok count, no draw effort, no per-stage timings) are checked to
  read as ``None`` / empty rather than a fabricated zero or string.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

pytest.importorskip("antecedent")
import antecedent


def _confounded_scm(n: int = 300, seed: int = 7):
    rng = random.Random(seed)
    z = np.empty(n, dtype=np.float64)
    t = np.empty(n, dtype=np.float64)
    y = np.empty(n, dtype=np.float64)
    for i in range(n):
        zi = rng.gauss(0.0, 1.0)
        p = 1.0 / (1.0 + math.exp(-(-0.4 + 0.9 * zi)))
        ti = 1.0 if rng.random() < p else 0.0
        yi = 2.0 * ti + zi + rng.gauss(0.0, 0.4)
        z[i] = zi
        t[i] = ti
        y[i] = yi
    return {"t": t, "y": y, "z": z}, [("z", "t"), ("z", "y"), ("t", "y")]


def _pulse_series(n: int = 300, seed: int = 11):
    rng = random.Random(seed)
    pressure = np.array([math.sin(0.04 * i) for i in range(n)], dtype=np.float64)
    defect = np.zeros(n, dtype=np.float64)
    for t in range(1, n):
        defect[t] = 0.9 * pressure[t - 1] + rng.gauss(0.0, 0.05)
    return {"pressure": pressure, "defect": defect}


_PULSE_QUERY_KWARGS = dict(
    treatment="pressure", outcome="defect", treatment_lag=1, horizon_steps=1, active_level=1.0
)


# --- Static (AteAnalysisResult) --------------------------------------------------


def test_static_nested_sections_mirror_flat_fields():
    data, edges = _confounded_scm()
    result = antecedent.analyze(
        data,
        graph=edges,
        query=antecedent.AverageEffect(treatment="t", outcome="y"),
        refute="placebo",
        bootstrap=20,
        seed=1,
    )
    raw = result._raw

    assert raw.identification.status == raw.identification_status
    assert raw.identification.method == raw.method
    assert list(raw.identification.adjustment_set) == list(raw.adjustment_set)
    assert raw.identification.assumption_count == raw.assumption_count
    assert raw.identification.derivation_step_count == raw.derivation_step_count

    assert raw.estimate.ate == raw.ate
    assert raw.estimate.se_analytic == raw.se_analytic
    assert raw.estimate.se_bootstrap == raw.se_bootstrap
    assert raw.estimate.estimator_id == raw.estimator_id
    assert raw.estimate.method == raw.method
    assert raw.estimate.overlap_ess == raw.overlap_ess
    assert raw.estimate.overlap_propensity_min == raw.overlap_propensity_min

    assert raw.validation.passed == raw.refutation_passed
    assert raw.validation.ran == raw.refutation_ran
    assert raw.validation.count == raw.refutation_count
    assert len(raw.validation.reports) == len(raw.refutations)
    for sec_report, flat_report in zip(raw.validation.reports, raw.refutations, strict=True):
        assert sec_report.refuter == flat_report.refuter
        assert sec_report.passed == flat_report.passed
        assert sec_report.comparison == flat_report.comparison

    assert raw.performance.plan_id == raw.plan_id
    assert raw.performance.modality == raw.modality
    assert raw.performance.peak_memory_bytes == raw.peak_memory_bytes
    assert raw.performance.latency_mode == raw.latency_mode
    assert raw.performance.wall_time_ns == raw.wall_time_ns
    assert raw.performance.bootstrap_replicates_requested == raw.bootstrap_replicates_requested
    assert raw.performance.bootstrap_replicates_ok == raw.bootstrap_replicates_ok
    assert raw.performance.n_draws == raw.n_draws_effort
    assert raw.performance.cancelled == raw.cancelled
    assert raw.performance.early_stopped == raw.early_stopped
    assert list(raw.performance.stage_timings) == list(raw.stage_timings)

    # Frequentist path: no posterior computed, on either the section or the flat
    # fields it mirrors.
    assert raw.posterior.effect_mean is None
    assert raw.posterior_effect_mean is None
    assert raw.posterior.artifact == raw.posterior_artifact


def test_static_posterior_section_mirrors_flat_fields_when_bayesian():
    data, edges = _confounded_scm()
    result = antecedent.analyze(
        data,
        graph=edges,
        query=antecedent.AverageEffect(treatment="t", outcome="y"),
        inference=antecedent.Bayesian(n_draws=256),
        refute=False,
        bootstrap=0,
        seed=2,
    )
    raw = result._raw

    assert raw.posterior.effect_mean == raw.posterior_effect_mean
    assert raw.posterior.effect_sd == raw.posterior_effect_sd
    assert raw.posterior.q025 == raw.posterior_q025
    assert raw.posterior.q975 == raw.posterior_q975
    assert raw.posterior.n_draws == raw.posterior_n_draws
    assert raw.posterior.p_below_zero == raw.posterior_p_below_zero
    assert raw.posterior.backend == raw.posterior_backend
    assert raw.posterior.unidentified_mass == raw.posterior_unidentified_mass
    assert raw.posterior.n_draws is not None
    assert raw.posterior.n_draws > 0


def test_static_validation_nothing_ran_is_false_not_true():
    data, edges = _confounded_scm()
    result = antecedent.analyze(
        data,
        graph=edges,
        query=antecedent.AverageEffect(treatment="t", outcome="y"),
        refute=False,
        bootstrap=0,
        seed=3,
    )
    raw = result._raw
    assert raw.validation.ran is False
    assert raw.validation.passed is False
    assert raw.validation.count == 0
    # And the section agrees with the flat scalars it mirrors.
    assert raw.refutation_ran is False
    assert raw.refutation_passed is False
    assert raw.validation.ran == raw.refutation_ran
    assert raw.validation.passed == raw.refutation_passed


# --- Temporal (AnalysisResult) --------------------------------------------------


def test_temporal_nested_sections_mirror_flat_fields():
    data = _pulse_series()
    result = antecedent.analyze(
        data,
        graph=[("pressure", 1, "defect", 0)],
        query=antecedent.PulseEffect(**_PULSE_QUERY_KWARGS),
        refute="placebo",
        bootstrap=10,
        seed=4,
    )
    raw = result._raw

    assert raw.identification.status == raw.identification_status
    assert raw.identification.method == raw.method
    assert list(raw.identification.adjustment_set) == list(raw.adjustment_set)
    assert raw.identification.assumption_count == raw.assumption_count
    assert raw.identification.derivation_step_count == raw.derivation_step_count

    assert raw.estimate.ate == raw.ate
    assert raw.estimate.se_analytic == raw.se_analytic
    assert raw.estimate.se_bootstrap == raw.se_bootstrap
    assert raw.estimate.estimator_id == raw.estimator_id
    assert raw.estimate.method == raw.method
    # The temporal facade always fixes OverlapPolicy::ExplicitOverride, under
    # which the shared adjustment estimator never computes an overlap report —
    # so there is genuinely nothing to report here.
    assert raw.estimate.overlap_ess is None
    assert raw.estimate.overlap_propensity_min is None

    assert raw.validation.count == raw.refutation_count
    assert len(raw.validation.reports) == len(raw.refutations)
    for sec_report, flat_report in zip(raw.validation.reports, raw.refutations, strict=True):
        assert sec_report.refuter == flat_report.refuter
        assert sec_report.passed == flat_report.passed
    expected_ran = len(raw.refutations) > 0
    expected_passed = expected_ran and all(r.passed for r in raw.refutations)
    assert raw.validation.ran == expected_ran
    assert raw.validation.passed == expected_passed

    assert raw.performance.plan_id == raw.plan_id
    assert raw.performance.modality == raw.modality
    assert raw.performance.peak_memory_bytes == raw.peak_memory_bytes
    # Genuinely populated on the temporal path too: every temporal execution
    # path records real wall-clock time and the requested bootstrap count on
    # `StudyResult.performance` — it was simply never wired into the flat
    # temporal fields before this change.
    assert raw.performance.wall_time_ns is not None
    assert raw.performance.bootstrap_replicates_requested is not None
    # Genuinely never populated on any temporal execution path (confirmed
    # against every `AssembleArgs` literal in `temporal_path.rs`/`panel_path.rs`)
    # — `None` / empty here is accurate, not a placeholder.
    assert raw.performance.bootstrap_replicates_ok is None
    assert raw.performance.n_draws is None
    assert raw.performance.stage_timings == []


def test_temporal_validation_nothing_ran_is_false_not_true():
    data = _pulse_series()
    result = antecedent.analyze(
        data,
        graph=[("pressure", 1, "defect", 0)],
        query=antecedent.PulseEffect(**_PULSE_QUERY_KWARGS),
        refute=False,
        bootstrap=0,
        seed=5,
    )
    raw = result._raw
    assert raw.validation.ran is False
    assert raw.validation.passed is False
    assert raw.validation.count == 0


def test_temporal_posterior_section_mirrors_flat_fields_when_bayesian():
    data = _pulse_series()
    result = antecedent.analyze(
        data,
        graph=[("pressure", 1, "defect", 0)],
        query=antecedent.PulseEffect(**_PULSE_QUERY_KWARGS),
        inference=antecedent.Bayesian(n_draws=256),
        refute=False,
        bootstrap=0,
        seed=6,
    )
    raw = result._raw

    assert raw.posterior.effect_mean == raw.posterior_effect_mean
    assert raw.posterior.effect_sd == raw.posterior_effect_sd
    assert raw.posterior.q025 == raw.posterior_q025
    assert raw.posterior.q975 == raw.posterior_q975
    assert raw.posterior.n_draws == raw.posterior_n_draws
    assert raw.posterior.p_below_zero == raw.posterior_p_below_zero
    assert raw.posterior.backend == raw.posterior_backend
    assert raw.posterior.unidentified_mass == raw.posterior_unidentified_mass
    assert raw.posterior.n_draws is not None
    assert raw.posterior.n_draws > 0
    # Temporal `posterior_artifact` is always `None` (never requested/populated
    # on the temporal path) — the section must mirror that, not fabricate bytes.
    assert raw.posterior.artifact is None
    assert raw.posterior_artifact is None
