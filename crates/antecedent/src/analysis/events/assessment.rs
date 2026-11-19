//! Time-ordered forecast assessment and separately measured count coverage.

use std::sync::Arc;

use antecedent_core::execution::{AdmissionLease, RngFactory, StreamDomain};
use antecedent_core::{CalibrationBasis, ExecutionContext};
use antecedent_io_core::calibration::{CalibrationBasisWire, calibration_slot};
use antecedent_io_core::contract_section::CalibrationSlotWire;
use serde::{Deserialize, Serialize};

use super::{
    CAPABILITY, COUNT_INTERVAL, EventError, EventHistory, EventSource, EventWindow, HawkesFitSpec,
    HawkesParameters, PreparedEventForecast, PreparedKernel, admit, digest, hawkes, history_bytes,
    release_supported, validate_budget,
};

/// One nonoverlapping forecast evaluated using only evidence before its cutoff.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ForecastOrigin {
    /// Training cutoff and held-out window start.
    pub cutoff: f64,
    /// Exclusive end of the held-out window.
    pub end: f64,
    /// True for validation, false for final test. Validation must precede test.
    pub validation: bool,
}

/// Proper count and threshold scores at one cutoff. Smaller scores are better.
#[derive(Clone, Debug, Serialize)]
pub struct ForecastScore {
    /// Evaluation window and partition.
    pub origin: ForecastOrigin,
    /// Actual count in the held-out window.
    pub observed_count: usize,
    /// Counts at the declared forecast type, ranked probability score.
    pub count_score: f64,
    /// Poisson comparison under the same training cutoff.
    pub poisson_count_score: f64,
    /// Probability that the declared count threshold was met.
    pub threshold_probability: f64,
    /// Squared probability error for that event.
    pub threshold_brier: f64,
    /// Poisson baseline Brier score for the same event.
    pub poisson_threshold_brier: f64,
    /// All-type held-out conditional log score for the realized timestamps.
    pub log_likelihood: f64,
    /// Held-out integrated-transformed event gaps, in unit-rate time.
    pub rescaled_gaps: Vec<f64>,
    /// Number of arrivals before the cutoff, within the fitted window.
    pub training_events: usize,
    /// Conditional point fit used at this cutoff.
    pub weak_identification: bool,
}

/// Reliability observations grouped by forecast-probability bin.
#[derive(Clone, Debug, Serialize)]
pub struct ReliabilityBin {
    /// Lower bound of the probability bin.
    pub lower: f64,
    /// Number of temporally disjoint forecasts contributing to the bin.
    pub windows: usize,
    /// Mean declared forecast probability in the bin.
    pub forecast: Option<f64>,
    /// Observed frequency in the bin.
    pub observed: Option<f64>,
    /// Whether enough observations and both outcomes were present for description.
    pub sufficient: bool,
}

/// Rolling-origin diagnostic report; deployment performance is separate from support.
#[derive(Clone, Debug, Serialize)]
pub struct ForecastAssessment {
    /// Compiler identity binding the frozen scope and assessment data.
    pub program: String,
    /// Validation and test scores in temporal order.
    pub scores: Vec<ForecastScore>,
    /// Test-window reliability observations, stratified by reported probability.
    pub test_reliability: Vec<ReliabilityBin>,
    /// Number of test threshold events, before any binning.
    pub test_threshold_events: usize,
    /// Chronologically adjacent windows can still be dependent.
    pub uncertainty: &'static str,
}

/// Unpublished count interval and exact basis for an independent coverage experiment.
#[derive(Clone, Debug)]
pub struct CountCoverageCandidate {
    /// Compiler identity of the forecast-and-count operation.
    pub program: String,
    /// Nearest-rank inclusive 90% interval.
    pub bounds: [usize; 2],
    /// Exact construction and design scope.
    pub basis: CalibrationBasis,
}

/// An inclusive count interval and the specific record that covers it.
#[derive(Clone, Debug, Serialize)]
pub struct PublishedCountInterval {
    /// Retained compiler identity.
    pub program: String,
    /// Inclusive nearest-rank count bounds.
    pub bounds: [usize; 2],
    /// Matched construction, measurement and current evidence.
    pub calibration: CalibrationSlotWire,
}

impl PreparedEventForecast {
    /// Compute an interval candidate for a coverage experiment. This carries no coverage claim.
    ///
    /// # Errors
    /// Unidentified bootstrap scope, resource refusal or forecast failure.
    #[allow(
        clippy::cast_possible_truncation,
        reason = "the path count is validated to at most 4096"
    )]
    pub fn count_coverage_candidate(
        &self,
        kind: usize,
        ctx: &ExecutionContext,
    ) -> Result<CountCoverageCandidate, EventError> {
        let request = &self.request;
        let EventSource::Fitted { fit, refits } = &request.source else {
            return Err(EventError::invalid("coverage candidates use recursive refit forecasts"));
        };
        if *refits == 0 || request.paths < 128 || kind >= request.history.catalog.len() {
            return Err(EventError::invalid(
                "count coverage requires a catalog coordinate and at least 128 refit-predictive draws",
            ));
        }
        let history = &request.history;
        let design = digest(
            "antecedent.events.count.coverage.scope.v1",
            &(
                &history.catalog,
                &history.time_unit,
                kind,
                fit,
                refits,
                request.paths,
                request.max_events,
                history.left.len(),
                history.window.end - history.window.start,
                history.window.start - history.empty_origin,
                request.forecast.end - request.forecast.start,
            ),
        )?
        .to_hex();
        let functional = format!("event_count:scope={design}");
        let basis = CalibrationBasis::new(
            [
                "OutcomePrediction",
                "Event",
                "fixed",
                "event",
                "Frequentist",
                COUNT_INTERVAL,
                "nearest_rank_90",
                "",
                "complete_event_history_and_future",
                "recursive_hawkes_refit_v1",
                &functional,
            ]
            .map(Arc::from),
            0.90,
            Arc::from("model_conditional"),
            history.events.len() as u64,
            None,
            Some(request.paths as u32),
            0.0,
        )
        .with_validation("none");
        let source = digest(
            "antecedent.events.count.interval.v1",
            &(self.program.identity().to_hex(), kind, &design),
        )?;
        let interval_plan = PreparedKernel::new(source, "count_interval", COUNT_INTERVAL)?;
        #[cfg(feature = "calibration-internal")]
        let law = {
            super::super::lowering::refuse_reentry(ctx)?;
            let scoped = ctx.enter_lowered_dispatch(self.program.identity());
            super::poll(&scoped)?;
            self.sample_retained(&scoped)?
        };
        #[cfg(not(feature = "calibration-internal"))]
        let law = self.execute(ctx)?;
        let mut counts: Vec<_> = law.paths().iter().map(|events| count(events, kind)).collect();
        Ok(CountCoverageCandidate {
            program: interval_plan.identity().to_hex(),
            bounds: nearest_rank_count_bounds(&mut counts),
            basis,
        })
    }

    /// Publish the interval only with an exact matching current calibration record.
    ///
    /// # Errors
    /// Unmeasured or outside measured scope, or construction failure.
    pub fn count_interval(
        &self,
        kind: usize,
        ctx: &ExecutionContext,
    ) -> Result<PublishedCountInterval, EventError> {
        if !release_supported(COUNT_INTERVAL) {
            return Err(EventError::IntervalUnmeasured);
        }
        let candidate = self.count_coverage_candidate(kind, ctx)?;
        let slot = calibration_slot(&CalibrationBasisWire::from(&candidate.basis));
        if slot.status != antecedent_io_core::contract_section::CalibrationStatus::Calibrated {
            return Err(EventError::IntervalUnmeasured);
        }
        Ok(PublishedCountInterval {
            program: candidate.program,
            bounds: candidate.bounds,
            calibration: slot,
        })
    }
}

fn count(events: &[super::Arrival], kind: usize) -> usize {
    events.iter().filter(|event| event.kind == kind).count()
}

/// Complete setup for rolling-origin forecast assessment.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ForecastAssessmentRequest {
    /// Observation timeline through the last test window.
    pub history: EventHistory,
    /// Fixed fitting scope, declared before the assessment.
    pub fit: HawkesFitSpec,
    /// Disjoint ordered validation and test windows.
    pub origins: Vec<ForecastOrigin>,
    /// Event type used for count scores.
    pub kind: usize,
    /// Minimum count for the threshold event.
    pub minimum_count: usize,
    /// Forecast draws per origin.
    pub paths: usize,
    /// Max events in any simulated path.
    pub max_events: usize,
    /// Stable seed for assessment simulations.
    pub seed: u64,
}

/// Native compiler plan for a time-ordered evaluation.
pub struct PreparedForecastAssessment {
    request: ForecastAssessmentRequest,
    plan: PreparedKernel,
    _lease: AdmissionLease,
}

impl crate::Study {
    /// Compile assessment with all cutoffs validated before fitting any model.
    ///
    /// # Errors
    /// Gaps, ambiguous chronology, invalid target or resource refusal.
    pub fn prepare_event_assessment(
        request: ForecastAssessmentRequest,
        ctx: &ExecutionContext,
    ) -> Result<PreparedForecastAssessment, EventError> {
        super::super::lowering::refuse_reentry(ctx)?;
        request.history.validate(true)?;
        request.fit.validate(request.history.catalog.len())?;
        validate_budget(request.paths, request.max_events)?;
        if !(2..=64).contains(&request.origins.len())
            || !(32..=4096).contains(&request.paths)
            || request.kind >= request.history.catalog.len()
            || !(1..=request.max_events).contains(&request.minimum_count)
            || !request.origins[0].validation
            || request.origins.last().is_none_or(|origin| origin.validation)
            || request
                .origins
                .windows(2)
                .any(|w| w[0].end > w[1].cutoff || (!w[0].validation && w[1].validation))
        {
            return Err(EventError::invalid(
                "assessment requires a frozen target and ordered disjoint validation then test windows",
            ));
        }
        for origin in &request.origins {
            EventWindow { start: origin.cutoff, end: origin.end }.validate()?;
            if origin.cutoff <= request.history.window.start
                || origin.end > request.history.window.end
            {
                return Err(EventError::invalid(
                    "assessment origin is outside the eligible observation range",
                ));
            }
        }
        let lease = admit(ctx, 0, history_bytes(&request.history) + 32_000)?;
        let source = digest("antecedent.events.assessment.v1", &request)?;
        let plan = PreparedKernel::new(source, "assess_forecasts", CAPABILITY)?;
        Ok(PreparedForecastAssessment { request, plan, _lease: lease })
    }
}

impl PreparedForecastAssessment {
    /// Inspect without evaluating a held-out window.
    pub fn plan_view(&self) -> crate::planner::PlanView {
        self.plan.plan_view()
    }

    /// Fit strictly before each cutoff and score against the subsequent window.
    ///
    /// # Errors
    /// Fit or operational failure, including a nonfinite held-out point-process score.
    #[allow(clippy::too_many_lines, reason = "one pass from origin fitting through scoring")]
    #[allow(
        clippy::cast_possible_truncation,
        reason = "threshold probabilities lie in [0, 1], so the reliability bin index is at most 5"
    )]
    pub fn execute(&self, ctx: &ExecutionContext) -> Result<ForecastAssessment, EventError> {
        let scoped = super::scoped_execution(ctx, self.plan.identity(), CAPABILITY)?;
        let ctx = &scoped;
        let r = &self.request;
        let bytes = history_bytes(&r.history);
        let _lease = admit(
            ctx,
            bytes * 10 + r.max_events as u64 * 64 + r.paths as u64 * 32,
            bytes + 64_000,
        )?;
        let mut work = hawkes::Work::new(ctx);
        let rng = RngFactory::from_seed(r.seed);
        let baseline_rng = StreamDomain::external(0x4556_4e54_504f_4953);
        let mut scores = Vec::with_capacity(r.origins.len());
        for (origin_index, &origin) in r.origins.iter().enumerate() {
            let mut training = r.history.clone();
            let split = training.events.partition_point(|event| event.time < origin.cutoff);
            let held_end = training.events.partition_point(|event| event.time < origin.end);
            let actual = &r.history.events[split..held_end];
            training.events.truncate(split);
            training.window.end = origin.cutoff;
            let model = hawkes::fit(&training, &r.fit, &mut work)?;
            let n = training.catalog.len();
            let mut rate = vec![0.0; n];
            let length = training.window.end - training.window.start;
            for event in &training.events {
                rate[event.kind] += 1.0;
            }
            for value in &mut rate {
                *value /= length;
            }
            let poisson = HawkesParameters {
                baseline: rate,
                branching: vec![0.0; n * n],
                decay: r.fit.decay.clone(),
            };
            let prefix = training.prefix();
            let window = EventWindow { start: origin.cutoff, end: origin.end };
            let mut hawkes_count = Vec::with_capacity(r.paths);
            let mut poisson_count = Vec::with_capacity(r.paths);
            for i in 0..r.paths {
                let index = (origin_index * r.paths + i) as u64;
                let draw = hawkes::sample(
                    &training,
                    &prefix,
                    &model.parameters,
                    window,
                    (&rng, hawkes::FORECAST, index),
                    r.max_events,
                    &mut work,
                )?;
                let baseline = hawkes::sample(
                    &training,
                    &[],
                    &poisson,
                    window,
                    (&rng, baseline_rng, index),
                    r.max_events,
                    &mut work,
                )?;
                hawkes_count.push(count(&draw, r.kind));
                poisson_count.push(count(&baseline, r.kind));
            }
            let observed = count(actual, r.kind);
            let outcome = f64::from(u8::from(observed >= r.minimum_count));
            let probability = |samples: &[usize]| {
                samples.iter().filter(|&&count| count >= r.minimum_count).count() as f64
                    / r.paths as f64
            };
            let threshold_probability = probability(&hawkes_count);
            let training_events = training.events.len();
            let observed_segment =
                EventHistory { window, left: prefix, events: actual.to_vec(), ..training };
            let log_likelihood =
                hawkes::log_likelihood(&observed_segment, &model.parameters, &mut work)?;
            if !log_likelihood.is_finite() {
                return Err(EventError::Numerical("zero held-out event intensity"));
            }
            let gaps = hawkes::rescaled_gaps(&observed_segment, &model.parameters, &mut work)?;
            scores.push(ForecastScore {
                origin,
                observed_count: observed,
                count_score: count_score(&mut hawkes_count, observed),
                poisson_count_score: count_score(&mut poisson_count, observed),
                threshold_probability,
                threshold_brier: (threshold_probability - outcome).powi(2),
                poisson_threshold_brier: (probability(&poisson_count) - outcome).powi(2),
                log_likelihood,
                rescaled_gaps: gaps,
                training_events,
                weak_identification: model.weak_identification,
            });
        }
        let test_threshold_events = scores
            .iter()
            .filter(|s| !s.origin.validation && s.observed_count >= r.minimum_count)
            .count();
        let test_reliability = (0..5)
            .map(|bin| {
                let lower = f64::from(bin) / 5.0;
                let outcomes: Vec<_> = scores
                    .iter()
                    .filter(|score| {
                        !score.origin.validation
                            && ((score.threshold_probability * 5.0) as i32).min(4) == bin
                    })
                    .collect();
                let windows = outcomes.len();
                let true_count =
                    outcomes.iter().filter(|score| score.observed_count >= r.minimum_count).count();
                ReliabilityBin {
                    lower,
                    windows,
                    forecast: (windows > 0).then(|| {
                        outcomes.iter().map(|score| score.threshold_probability).sum::<f64>()
                            / windows as f64
                    }),
                    observed: (windows > 0).then(|| true_count as f64 / windows as f64),
                    sufficient: windows >= 20 && true_count >= 5 && windows - true_count >= 5,
                }
            })
            .collect();
        Ok(ForecastAssessment {
            program: self.plan.identity().to_hex(),
            scores,
            test_reliability,
            test_threshold_events,
            uncertainty: "descriptive rolling windows; dependence and sparse events can limit reliability",
        })
    }
}

fn count_score(draws: &mut [usize], observed: usize) -> f64 {
    draws.sort_unstable();
    let n = draws.len() as f64;
    let absolute = draws.iter().map(|&x| x.abs_diff(observed) as f64).sum::<f64>() / n;
    let spread =
        draws.iter().enumerate().map(|(i, &x)| (2.0 * i as f64 + 1.0 - n) * x as f64).sum::<f64>()
            / (n * n);
    (absolute - spread).max(0.0)
}

fn nearest_rank_count_bounds(counts: &mut [usize]) -> [usize; 2] {
    counts.sort_unstable();
    let rank = |numerator: usize| (counts.len() * numerator).div_ceil(100) - 1;
    [counts[rank(5)], counts[rank(95)]]
}

#[cfg(test)]
mod tests {
    #[test]
    fn count_interval_matches_hand_ranked_discrete_tails() {
        let mut counts = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 1, 2, 2, 3, 4, 4, 5, 20];
        // On twenty draws the endpoints are ranks one and nineteen. Ties remain
        // as integer counts and the last draw is outside the inclusive interval.
        assert_eq!(super::nearest_rank_count_bounds(&mut counts), [0, 9]);
    }
}
