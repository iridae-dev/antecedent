//! Whole-procedure simulation under a hypothesized truth, for the estimators
//! whose value depends on the order of the history (adaptively weighted
//! scores), where the exact law of [`super::procedure_law`] does not apply.
//!
//! One replication runs the declared procedure end to end: the policy is
//! evaluated from the observed counts at every look, units are enrolled,
//! exposure failures and outcome delay and loss are drawn, the stopping rule
//! is evaluated at each look, and the estimator is computed from the history
//! retained at the terminal look (units whose outcome was revealed and
//! observed, each with the policy's realized probabilities at its
//! enrolment). The collection of replicates is the sampling law of the
//! estimator *under the stopping time and policy selection*, with Monte Carlo
//! error that the caller reports; it is not a coverage claim.
//!
//! Streams are keyed by the seed and the replicate index, so a replicate does
//! not depend on how many others are run.
//!
//! This is the route for procedures too large to enumerate. Its cost is
//! replications times units, admitted against a work and memory budget before
//! any replicate runs ([`super::capacity::ResourceBudget`]), and the result
//! carries a [`ResourceReceipt`] of the work it did.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::VecDeque;

use antecedent_core::CausalRng;

use super::AdaptiveInferenceError as Error;
use super::capacity::{CapacityError, ResourceBudget, ResourceReceipt, SimulationShape};
use super::procedure_contract::{Attribution, ProcedureSpec};
use super::procedure_law::{Rules, deliveries};
use super::weighting::{
    Estimand, OutcomeModelPrior, WeightScheme, adaptive_scores, weighted_estimate,
};

/// Most replications one simulation accepts.
pub const MAX_SIMULATION_REPLICATIONS: u32 = 1_000_000;

/// Entries each policy and posterior cache of one simulation may hold.
const SIMULATION_CACHE_ENTRIES: usize = 50_000;
/// Bytes one cache entry is budgeted at (key, value and tree overhead).
const CACHE_ENTRY_BYTES: u64 = 256;
/// Caches the rules keep.
const CACHES: u64 = 4;

const STREAM_MIX: u64 = 0x9E37_79B9_7F4A_7C15;

/// What to simulate and how to estimate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimulationRequest {
    /// Replications to run.
    pub replications: u32,
    /// Seed of the replicate streams.
    pub seed: u64,
    /// The estimand of the weighted estimator.
    pub estimand: Estimand,
    /// How scores are weighted.
    pub scheme: WeightScheme,
    /// Prior of the predictable outcome model.
    pub outcome_prior: OutcomeModelPrior,
    /// Smallest assignment probability an estimand arm may have had.
    pub probability_floor: f64,
}

/// One simulated run of the whole procedure.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimulatedReplicate {
    /// Weighted estimate at the terminal look; `None` with fewer than two
    /// analysed units.
    pub estimate: Option<f64>,
    /// Plug-in variance of the estimate (uncalibrated).
    pub variance: Option<f64>,
    /// Terminal look.
    pub look: usize,
    /// Whether the stopping rule ended the run.
    pub rule_stopped: bool,
    /// The arm decided or selected.
    pub decision: Option<usize>,
    /// Units enrolled at the end.
    pub enrolled: u32,
    /// Units that entered the estimator.
    pub analysed_units: usize,
}

/// The collection of replicates.
#[derive(Clone, Debug, PartialEq)]
pub struct SimulatedLaw {
    /// Replicates, in index order.
    pub replicates: Vec<SimulatedReplicate>,
    /// The work the simulation was admitted for and did.
    pub receipt: ResourceReceipt,
}

/// Summary of the estimates of a simulated law against a known truth.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimulationSummary {
    /// Replicates with a defined estimate.
    pub defined: usize,
    /// Mean estimate.
    pub mean: f64,
    /// Sample standard deviation of the estimates.
    pub standard_deviation: f64,
    /// Mean estimate minus the truth.
    pub bias: f64,
    /// Monte Carlo standard error of the mean estimate.
    pub monte_carlo_standard_error: f64,
    /// Mean plug-in variance.
    pub mean_reported_variance: f64,
    /// Fraction of defined replicates whose plug-in Wald interval at the
    /// stated multiplier contains the truth (a measured property of this
    /// simulation, not a coverage statement).
    pub wald_contains_truth: f64,
}

impl SimulatedLaw {
    /// Summarize the estimates against `truth`, with the Wald half-width
    /// multiplier `z`.
    ///
    /// # Errors
    /// Fewer than two replicates with a defined estimate, or a non-finite
    /// truth or multiplier.
    pub fn summarize(&self, truth: f64, z: f64) -> Result<SimulationSummary, Error> {
        if !(truth.is_finite() && z.is_finite() && z > 0.0) {
            return Err(Error::InvalidParameter("truth and multiplier must be finite, z positive"));
        }
        let defined: Vec<(f64, f64)> =
            self.replicates.iter().filter_map(|r| r.estimate.zip(r.variance)).collect();
        if defined.len() < 2 {
            return Err(Error::Shape("a summary needs at least two defined replicates"));
        }
        let n = defined.len() as f64;
        let mean = defined.iter().map(|(e, _)| e).sum::<f64>() / n;
        let variance = defined.iter().map(|(e, _)| (e - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let reported = defined.iter().map(|(_, v)| v).sum::<f64>() / n;
        let contained = defined.iter().filter(|(e, v)| (e - truth).abs() <= z * v.sqrt()).count();
        Ok(SimulationSummary {
            defined: defined.len(),
            mean,
            standard_deviation: variance.sqrt(),
            bias: mean - truth,
            monte_carlo_standard_error: (variance / n).sqrt(),
            mean_reported_variance: reported,
            wald_contains_truth: contained as f64 / n,
        })
    }

    /// Mean enrolled units over the replicates.
    #[must_use]
    pub fn mean_enrolled(&self) -> f64 {
        let n = self.replicates.len().max(1) as f64;
        self.replicates.iter().map(|r| f64::from(r.enrolled)).sum::<f64>() / n
    }

    /// Fraction of replicates ended by the stopping rule.
    #[must_use]
    pub fn rule_stop_fraction(&self) -> f64 {
        let n = self.replicates.len().max(1) as f64;
        self.replicates.iter().filter(|r| r.rule_stopped).count() as f64 / n
    }
}

pub(crate) fn draw(rng: &mut CausalRng, probabilities: &[f64]) -> usize {
    let u = rng.next_f64();
    let mut cumulative = 0.0;
    for (index, &p) in probabilities.iter().enumerate() {
        cumulative += p;
        if u < cumulative {
            return index;
        }
    }
    probabilities.len() - 1
}

struct Unit {
    context: usize,
    attributed: usize,
    /// Index of the unit's assignment row in the replication's row store.
    row: usize,
    success: bool,
    observed: bool,
}

/// The assignment probabilities that enter the estimator for each context
/// at one look: the policy's row for the assigned arm, or the delivered-arm
/// distribution it induces.
fn effective_row(spec: &ProcedureSpec, row: &[f64]) -> Vec<f64> {
    match spec.exposure().attribution {
        Attribution::Assigned => row.to_vec(),
        Attribution::Delivered => {
            let mut effective = vec![0.0; spec.arms()];
            for (a, &pi) in row.iter().enumerate() {
                for (arm, p) in deliveries(spec, a) {
                    effective[arm] += pi * p;
                }
            }
            effective
        }
    }
}

/// Run the declared procedure `request.replications` times under the success
/// probabilities `truth` (one per arm-by-context cell), within the library
/// work and memory bounds.
///
/// # Errors
/// A truth of the wrong shape, a request outside bounds, a positivity
/// failure of the estimand arms, or cancellation.
pub fn simulate_procedure(
    spec: &ProcedureSpec,
    truth: &[f64],
    request: &SimulationRequest,
    cancelled: &dyn Fn() -> bool,
) -> Result<SimulatedLaw, Error> {
    simulate_procedure_within(spec, truth, request, &ResourceBudget::DEFAULT, cancelled)
}

/// [`simulate_procedure`] within an explicit work and memory budget.
///
/// The request is admitted before any replicate runs: replications times
/// units must be within `budget.max_work` and the estimated peak bytes of the
/// replicate records and of one replication's history within
/// `budget.max_memory_bytes`.
///
/// # Errors
/// As [`simulate_procedure`], and a capacity refusal for a request over the
/// replication bound, the work budget or the memory budget.
#[allow(
    clippy::too_many_lines,
    reason = "one replicate is one linear walk through looks; splitting it would scatter its state"
)]
pub fn simulate_procedure_within(
    spec: &ProcedureSpec,
    truth: &[f64],
    request: &SimulationRequest,
    budget: &ResourceBudget,
    cancelled: &dyn Fn() -> bool,
) -> Result<SimulatedLaw, Error> {
    if truth.len() != spec.cells() {
        return Err(Error::Shape("one success probability per arm-by-context cell"));
    }
    if truth.iter().any(|p| !(p.is_finite() && (0.0..=1.0).contains(p))) {
        return Err(Error::InvalidParameter("a success probability must lie in [0, 1]"));
    }
    if request.replications == 0 {
        return Err(Error::InvalidParameter("a simulation needs at least one replication"));
    }
    if request.replications > MAX_SIMULATION_REPLICATIONS {
        return Err(Error::Capacity(CapacityError::SimulationReplications {
            limit: u64::from(MAX_SIMULATION_REPLICATIONS),
            requested: u64::from(request.replications),
        }));
    }
    let arms = spec.arms();
    let contexts = spec.contexts().to_vec();
    let cumulative = spec.cumulative_enrolment();
    let lag = spec.observation().lag_looks;
    let looks = spec.enrolment().len();
    let total_units = cumulative.last().copied().unwrap_or(0);
    let admitted = budget.admit(&SimulationShape {
        replications: request.replications,
        units: total_units,
        measurements: 1,
        record_bytes: std::mem::size_of::<SimulatedReplicate>() as u64,
        unit_bytes: (std::mem::size_of::<Unit>() + 4 * std::mem::size_of::<usize>()) as u64
            + 4 * arms as u64 * std::mem::size_of::<f64>() as u64,
        fixed_bytes: CACHES * SIMULATION_CACHE_ENTRIES as u64 * CACHE_ENTRY_BYTES,
    })?;
    let states = 2 * spec.cells() as u64;
    let mut rules = Rules::new(spec, cancelled, SIMULATION_CACHE_ENTRIES);
    let mut replicates = Vec::with_capacity(request.replications as usize);
    let mut receipt = ResourceReceipt {
        work_planned: admitted.work_planned,
        estimated_peak_bytes: admitted.estimated_peak_bytes,
        ..ResourceReceipt::default()
    };
    for replicate in 0..request.replications {
        if cancelled() {
            return Err(Error::Cancelled);
        }
        let mut rng = CausalRng::from_seed(
            request.seed ^ (u64::from(replicate) + 1).wrapping_mul(STREAM_MIX),
        );
        let mut observed = vec![0_u16; 2 * spec.cells()];
        let mut units: Vec<Unit> = Vec::new();
        let mut row_store: Vec<Vec<f64>> = Vec::new();
        let mut pending: VecDeque<Vec<usize>> = VecDeque::new();
        let mut revealed: Vec<usize> = Vec::new();
        for (look, &cumulative_look) in cumulative.iter().enumerate().take(looks) {
            if cancelled() {
                return Err(Error::Cancelled);
            }
            let mut rows = Vec::with_capacity(contexts.len());
            let mut row_index = Vec::with_capacity(contexts.len());
            for c in 0..contexts.len() {
                let row = rules.policy_row(&observed[2 * arms * c..2 * arms * (c + 1)])?;
                row_index.push(row_store.len());
                row_store.push(effective_row(spec, &row));
                rows.push(row);
            }
            let mut batch = Vec::new();
            for _ in 0..spec.enrolment()[look] {
                let context = draw(&mut rng, &contexts);
                let assigned = draw(&mut rng, &rows[context]);
                let options = deliveries(spec, assigned);
                let weights: Vec<f64> = options.iter().map(|o| o.1).collect();
                let delivered = options[draw(&mut rng, &weights)].0;
                let success = rng.next_f64() < truth[context * arms + delivered];
                let retained = if success {
                    spec.observation().retain_success
                } else {
                    spec.observation().retain_failure
                };
                let observed_flag = rng.next_f64() < retained;
                let attributed = match spec.exposure().attribution {
                    Attribution::Assigned => assigned,
                    Attribution::Delivered => delivered,
                };
                batch.push(units.len());
                units.push(Unit {
                    context,
                    attributed,
                    row: row_index[context],
                    success,
                    observed: observed_flag,
                });
            }
            pending.push_back(batch);
            if pending.len() > lag {
                if let Some(aged) = pending.pop_front() {
                    for index in aged {
                        let unit = &units[index];
                        if unit.observed {
                            let cell = unit.context * arms + unit.attributed;
                            observed[2 * cell + usize::from(!unit.success)] += 1;
                            revealed.push(index);
                        }
                    }
                }
            }
            let assessment = rules.assess(look, cumulative_look, &observed)?;
            if assessment.stop || look + 1 == looks {
                let decision = if assessment.stop {
                    assessment.decision.or_else(|| rules.select(&observed))
                } else {
                    rules.select(&observed)
                };
                revealed.sort_unstable();
                let mut estimate = None;
                let mut variance = None;
                if revealed.len() >= 2 {
                    let assigned: Vec<usize> =
                        revealed.iter().map(|&i| units[i].attributed).collect();
                    let probabilities: Vec<f64> = revealed
                        .iter()
                        .flat_map(|&i| row_store[units[i].row].iter().copied())
                        .collect();
                    let outcomes: Vec<f64> =
                        revealed.iter().map(|&i| f64::from(u8::from(units[i].success))).collect();
                    let scores = adaptive_scores(
                        arms,
                        &assigned,
                        &probabilities,
                        &outcomes,
                        request.outcome_prior,
                        cancelled,
                    )?;
                    let fit = weighted_estimate(
                        arms,
                        &probabilities,
                        &scores,
                        request.estimand,
                        request.scheme,
                        None,
                        request.probability_floor,
                    )?;
                    estimate = Some(fit.estimate);
                    variance = Some(fit.variance);
                }
                replicates.push(SimulatedReplicate {
                    estimate,
                    variance,
                    look,
                    rule_stopped: assessment.stop,
                    decision,
                    enrolled: cumulative[look],
                    analysed_units: revealed.len(),
                });
                receipt
                    .record_replicate(look + 1, cumulative[look], 1, states)
                    .map_err(Error::Capacity)?;
                break;
            }
        }
    }
    Ok(SimulatedLaw { replicates, receipt })
}
