//! The exact law of a complete adaptive procedure.
//!
//! For a [`ProcedureSpec`] and a hypothesized outcome kernel, the law of the
//! procedure's terminal state is computed by dynamic programming over the
//! *observed* state of the trial. Everything that shapes the data is inside
//! the recursion, not inherited from a fixed-sample analysis:
//!
//! * the assignment rule is re-evaluated at every look from the observed
//!   counts, so the realized assignment probabilities are exactly the
//!   policy's;
//! * outcomes are revealed after the declared delay and may be lost,
//!   possibly depending on their value, so the observed state at a look is a
//!   function of what has aged, not of what has happened;
//! * an assigned arm may fail to be delivered (exploration failure) and the
//!   outcome is then attributed as declared;
//! * the stopping rule is evaluated at every look on the observed counts, so
//!   the terminal state is the state *at the stopping time*;
//! * the final selection of an arm is part of the procedure.
//!
//! The state is the vector of observed success and failure counts per
//! arm-by-context cell, the full outcome counts per delivered cell when the
//! kernel is a prior predictive (a lost outcome still updates the truth it
//! came from), and the assigned counts of the batches that have not yet
//! aged. Units of a batch are expanded one at a time and merged by state, so
//! the work is bounded by the number of distinct states, not by the number of
//! outcome sequences.
//!
//! Two outcome kernels are supported. [`OutcomeKernel::Fixed`] hypothesizes a
//! success probability per cell and yields the sampling law of the
//! procedure under that truth. [`OutcomeKernel::PriorPredictive`] integrates
//! independent Beta priors out (a Polya urn per cell) and yields the
//! prior-predictive law, the one under which a posterior is checked.
//!
//! Exact finite-law arithmetic supplies no interval or calibration licence:
//! the law is of the declared procedure under a hypothesized truth, nothing
//! more.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use super::AdaptiveInferenceError as Error;
use super::beta_bernoulli::{BetaPosterior, best_arm_probabilities, expected_losses};
use super::group_sequential::Sidedness;
use super::procedure_contract::{
    AssignmentRule, Attribution, BetaPrior, FinalSelection, ProcedureSpec, StoppingRule,
};

/// Most distinct live states one look may hold.
pub const MAX_LAW_STATES: usize = 1_000_000;
/// Most unit expansions one law may perform.
pub const MAX_LAW_WORK: u64 = 60_000_000;

const CANCEL_STRIDE: u64 = 1024;

/// The outcome mechanism of a hypothesized truth.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum OutcomeKernel {
    /// A success probability per arm-by-context cell (context-major).
    Fixed {
        /// One probability per cell.
        success_probabilities: Vec<f64>,
    },
    /// Independent Beta priors per cell, integrated out.
    PriorPredictive {
        /// One prior per cell.
        priors: Vec<BetaPrior>,
    },
}

impl OutcomeKernel {
    fn check(&self, cells: usize) -> Result<(), Error> {
        match self {
            Self::Fixed { success_probabilities } => {
                if success_probabilities.len() != cells {
                    return Err(Error::Shape("one success probability per arm-by-context cell"));
                }
                if success_probabilities.iter().any(|p| !(p.is_finite() && (0.0..=1.0).contains(p)))
                {
                    return Err(Error::InvalidParameter(
                        "a success probability must lie in [0, 1]",
                    ));
                }
            }
            Self::PriorPredictive { priors } => {
                if priors.len() != cells {
                    return Err(Error::Shape("one prior per arm-by-context cell"));
                }
                if priors.iter().any(|p| !p.is_valid()) {
                    return Err(Error::InvalidParameter(
                        "a Beta prior shape must be finite and at least one",
                    ));
                }
            }
        }
        Ok(())
    }

    const fn is_predictive(&self) -> bool {
        matches!(self, Self::PriorPredictive { .. })
    }
}

/// One terminal state of the procedure and its probability.
#[derive(Clone, Debug, PartialEq)]
pub struct TerminalAtom {
    /// Look (zero-based) at which the procedure ended.
    pub look: usize,
    /// Whether the stopping rule, rather than the planned horizon, ended it.
    pub rule_stopped: bool,
    /// Observed `(successes, failures)` per cell, `2 * cells` entries.
    pub observed: Vec<u16>,
    /// The arm the procedure decided for or selected, if any.
    pub decision: Option<usize>,
    /// Units enrolled when the procedure ended.
    pub enrolled: u32,
    /// Probability of this terminal state.
    pub probability: f64,
}

impl TerminalAtom {
    /// Observed `(successes, failures)` of a cell.
    #[must_use]
    pub fn cell_counts(&self, cell: usize) -> (u16, u16) {
        (self.observed[2 * cell], self.observed[2 * cell + 1])
    }

    /// Observed units of a cell.
    #[must_use]
    pub fn cell_units(&self, cell: usize) -> u32 {
        let (s, f) = self.cell_counts(cell);
        u32::from(s) + u32::from(f)
    }

    /// Observed mean of a cell, `None` without an observation.
    #[must_use]
    pub fn cell_mean(&self, cell: usize) -> Option<f64> {
        let n = self.cell_units(cell);
        (n > 0).then(|| f64::from(self.cell_counts(cell).0) / f64::from(n))
    }
}

/// The law of the terminal state of a procedure.
#[derive(Clone, Debug, PartialEq)]
pub struct ProcedureLaw {
    /// Terminal atoms, in a deterministic order.
    pub atoms: Vec<TerminalAtom>,
    /// Cumulative units enrolled through each look.
    pub cumulative_enrolment: Vec<u32>,
    /// Arms.
    pub arms: usize,
    /// Arm-by-context cells.
    pub cells: usize,
    /// Unit expansions performed (observed work).
    pub work_units: u64,
    /// Largest number of live states at any look (observed work).
    pub peak_states: usize,
}

/// The assignment, stopping and selection logic of a procedure as functions
/// of the observed counts, shared by the exact law and the simulation.
pub(crate) struct Rules<'a> {
    spec: &'a ProcedureSpec,
    cancelled: &'a dyn Fn() -> bool,
    policy_cache: BTreeMap<Vec<u16>, Vec<f64>>,
    best_cache: BTreeMap<Vec<u16>, Vec<f64>>,
    loss_cache: BTreeMap<Vec<u16>, Vec<f64>>,
}

pub(crate) struct Assessment {
    pub stop: bool,
    pub decision: Option<usize>,
}

fn posteriors(prior: BetaPrior, counts: &[u16]) -> Result<Vec<BetaPosterior>, Error> {
    counts
        .chunks_exact(2)
        .map(|c| {
            BetaPosterior::from_counts(prior.alpha, prior.beta, u64::from(c[0]), u64::from(c[1]))
        })
        .collect()
}

fn first_maximum(values: &[f64]) -> usize {
    let mut best = 0;
    for (index, &value) in values.iter().enumerate().skip(1) {
        if value > values[best] {
            best = index;
        }
    }
    best
}

impl<'a> Rules<'a> {
    pub(crate) fn new(spec: &'a ProcedureSpec, cancelled: &'a dyn Fn() -> bool) -> Self {
        Self {
            spec,
            cancelled,
            policy_cache: BTreeMap::new(),
            best_cache: BTreeMap::new(),
            loss_cache: BTreeMap::new(),
        }
    }

    /// The policy's distribution over arms given one context's observed
    /// `(successes, failures)` per arm.
    pub(crate) fn policy_row(&mut self, counts: &[u16]) -> Result<Vec<f64>, Error> {
        let arms = self.spec.arms();
        match self.spec.assignment() {
            AssignmentRule::Fixed { probabilities } => Ok(probabilities.clone()),
            AssignmentRule::EpsilonGreedy { epsilon, prior } => {
                let numerator = |a: usize| prior.alpha + f64::from(counts[2 * a]);
                let denominator = |a: usize| {
                    prior.alpha
                        + prior.beta
                        + f64::from(counts[2 * a])
                        + f64::from(counts[2 * a + 1])
                };
                let mut greedy = vec![0_usize];
                for a in 1..arms {
                    let b = greedy[0];
                    let lhs = numerator(a) * denominator(b);
                    let rhs = numerator(b) * denominator(a);
                    if lhs > rhs {
                        greedy = vec![a];
                    } else if (lhs - rhs).abs() == 0.0 {
                        greedy.push(a);
                    }
                }
                let mut row = vec![epsilon / arms as f64; arms];
                let share = (1.0 - epsilon) / greedy.len() as f64;
                for a in greedy {
                    row[a] += share;
                }
                Ok(row)
            }
            AssignmentRule::ThompsonBeta { prior } => {
                if let Some(row) = self.policy_cache.get(counts) {
                    return Ok(row.clone());
                }
                let posts = posteriors(*prior, counts)?;
                let mut row = best_arm_probabilities(&posts, self.cancelled)?;
                for p in &mut row {
                    *p = p.max(0.0);
                }
                let total: f64 = row.iter().sum();
                if !(total.is_finite() && total > 0.0) {
                    return Err(Error::Numerical("Thompson probabilities did not normalize"));
                }
                for p in &mut row {
                    *p /= total;
                }
                self.policy_cache.insert(counts.to_vec(), row.clone());
                Ok(row)
            }
            AssignmentRule::UpperConfidenceBound { exploration } => {
                let units = |a: usize| u32::from(counts[2 * a]) + u32::from(counts[2 * a + 1]);
                let mut row = vec![0.0; arms];
                if let Some(untried) = (0..arms).find(|&a| units(a) == 0) {
                    row[untried] = 1.0;
                    return Ok(row);
                }
                let total: f64 = (0..arms).map(|a| f64::from(units(a))).sum();
                let scores: Vec<f64> = (0..arms)
                    .map(|a| {
                        let n = f64::from(units(a));
                        f64::from(counts[2 * a]) / n + (exploration * total.ln() / n).sqrt()
                    })
                    .collect();
                row[first_maximum(&scores)] = 1.0;
                Ok(row)
            }
        }
    }

    fn best_probabilities(&mut self, prior: BetaPrior, counts: &[u16]) -> Result<Vec<f64>, Error> {
        if let Some(found) = self.best_cache.get(counts) {
            return Ok(found.clone());
        }
        let value = best_arm_probabilities(&posteriors(prior, counts)?, self.cancelled)?;
        self.best_cache.insert(counts.to_vec(), value.clone());
        Ok(value)
    }

    fn losses(&mut self, prior: BetaPrior, counts: &[u16]) -> Result<Vec<f64>, Error> {
        if let Some(found) = self.loss_cache.get(counts) {
            return Ok(found.clone());
        }
        let value = expected_losses(&posteriors(prior, counts)?, self.cancelled)?;
        self.loss_cache.insert(counts.to_vec(), value.clone());
        Ok(value)
    }

    /// Evaluate the stopping rule at `look` on the observed counts.
    pub(crate) fn assess(
        &mut self,
        look: usize,
        enrolled: u32,
        observed: &[u16],
    ) -> Result<Assessment, Error> {
        let none = Assessment { stop: false, decision: None };
        match self.spec.stopping().clone() {
            StoppingRule::NoStopping => Ok(none),
            StoppingRule::ResourceCap { max_units } => {
                Ok(Assessment { stop: enrolled >= max_units, decision: None })
            }
            StoppingRule::PosteriorProbabilityBest { threshold, prior } => {
                let best = self.best_probabilities(prior, observed)?;
                let arm = first_maximum(&best);
                if best[arm] >= threshold {
                    Ok(Assessment { stop: true, decision: Some(arm) })
                } else {
                    Ok(none)
                }
            }
            StoppingRule::PosteriorExpectedLoss { loss_threshold, prior } => {
                let losses = self.losses(prior, observed)?;
                let negated: Vec<f64> = losses.iter().map(|l| -l).collect();
                let arm = first_maximum(&negated);
                if losses[arm] < loss_threshold {
                    Ok(Assessment { stop: true, decision: Some(arm) })
                } else {
                    Ok(none)
                }
            }
            StoppingRule::WaldBoundaries { critical, sidedness } => {
                let (s0, f0) = (f64::from(observed[0]), f64::from(observed[1]));
                let (s1, f1) = (f64::from(observed[2]), f64::from(observed[3]));
                let (n0, n1) = (s0 + f0, s1 + f1);
                if n0 == 0.0 || n1 == 0.0 {
                    return Ok(none);
                }
                let (p0, p1) = (s0 / n0, s1 / n1);
                let variance = p0 * (1.0 - p0) / n0 + p1 * (1.0 - p1) / n1;
                if variance.is_nan() || variance <= 0.0 {
                    return Ok(none);
                }
                let z = (p1 - p0) / variance.sqrt();
                let c = critical[look];
                Ok(match sidedness {
                    Sidedness::OneSidedUpper if z >= c => {
                        Assessment { stop: true, decision: Some(1) }
                    }
                    Sidedness::TwoSidedSymmetric if z.abs() >= c => {
                        Assessment { stop: true, decision: Some(usize::from(z > 0.0)) }
                    }
                    _ => none,
                })
            }
        }
    }

    /// The arm the procedure selects from the observed counts when no rule
    /// decided.
    pub(crate) fn select(&self, observed: &[u16]) -> Option<usize> {
        let arms = self.spec.arms();
        let (prior, require_observation) = match self.spec.selection() {
            FinalSelection::NoSelection => return None,
            FinalSelection::HighestObservedMean => (BetaPrior { alpha: 0.0, beta: 0.0 }, true),
            FinalSelection::HighestPosteriorMean(prior) => (*prior, false),
        };
        let numerator = |a: usize| prior.alpha + f64::from(observed[2 * a]);
        let denominator = |a: usize| {
            prior.alpha + prior.beta + f64::from(observed[2 * a]) + f64::from(observed[2 * a + 1])
        };
        let mut best: Option<usize> = None;
        for a in 0..arms {
            if require_observation && denominator(a) == 0.0 {
                continue;
            }
            best = match best {
                None => Some(a),
                Some(b) if numerator(a) * denominator(b) > numerator(b) * denominator(a) => Some(a),
                keep => keep,
            };
        }
        best
    }
}

/// The delivered arm distribution of an assigned arm: `(arm, probability)`.
pub(crate) fn deliveries(spec: &ProcedureSpec, assigned: usize) -> Vec<(usize, f64)> {
    let failure = spec.exposure().failure_probability[assigned];
    let fallback = spec.exposure().fallback_arm;
    if failure <= 0.0 || fallback == assigned {
        vec![(assigned, 1.0)]
    } else {
        vec![(assigned, 1.0 - failure), (fallback, failure)]
    }
}

struct Layout {
    cells: usize,
    arms: usize,
    lag: usize,
    observed_len: usize,
    full_offset: usize,
    pending_offset: usize,
    total: usize,
}

impl Layout {
    fn new(spec: &ProcedureSpec, predictive: bool) -> Self {
        let cells = spec.cells();
        let lag = spec.observation().lag_looks;
        let observed_len = 2 * cells;
        let full_offset = observed_len;
        let full_len = if predictive { 2 * cells } else { 0 };
        let pending_offset = full_offset + full_len;
        Self {
            cells,
            arms: spec.arms(),
            lag,
            observed_len,
            full_offset,
            pending_offset,
            total: pending_offset + (lag + 1) * cells,
        }
    }

    const fn current_offset(&self) -> usize {
        self.pending_offset + self.lag * self.cells
    }
}

type StateMap = BTreeMap<Vec<u16>, f64>;

struct Engine<'a> {
    spec: &'a ProcedureSpec,
    kernel: &'a OutcomeKernel,
    layout: Layout,
    rules: Rules<'a>,
    work: u64,
    cancelled: &'a dyn Fn() -> bool,
}

impl Engine<'_> {
    fn tick(&mut self, amount: u64) -> Result<(), Error> {
        let before = self.work / CANCEL_STRIDE;
        self.work += amount;
        if self.work > MAX_LAW_WORK {
            return Err(Error::Limit("the procedure law exceeded its work bound"));
        }
        if self.work / CANCEL_STRIDE != before && (self.cancelled)() {
            return Err(Error::Cancelled);
        }
        Ok(())
    }

    /// Assigned counts of one batch as a law over per-cell compositions.
    fn composition_law(
        &mut self,
        cell_probabilities: &[f64],
        batch: u32,
    ) -> Result<BTreeMap<Vec<u16>, f64>, Error> {
        let mut law: BTreeMap<Vec<u16>, f64> = BTreeMap::new();
        law.insert(vec![0_u16; cell_probabilities.len()], 1.0);
        for _ in 0..batch {
            let mut next: BTreeMap<Vec<u16>, f64> = BTreeMap::new();
            for (composition, probability) in &law {
                for (cell, &q) in cell_probabilities.iter().enumerate() {
                    if q > 0.0 {
                        let mut grown = composition.clone();
                        grown[cell] += 1;
                        *next.entry(grown).or_insert(0.0) += probability * q;
                    }
                }
            }
            self.tick(next.len() as u64)?;
            law = next;
        }
        Ok(law)
    }

    fn enrol(&mut self, live: StateMap, batch: u32) -> Result<StateMap, Error> {
        let arms = self.layout.arms;
        let contexts = self.spec.contexts().to_vec();
        let current = self.layout.current_offset();
        let mut next: StateMap = BTreeMap::new();
        for (key, probability) in live {
            let mut cell_probabilities = vec![0.0; self.layout.cells];
            for (c, &context_probability) in contexts.iter().enumerate() {
                let counts = &key[2 * arms * c..2 * arms * (c + 1)];
                let row = self.rules.policy_row(counts)?;
                for (a, &p) in row.iter().enumerate() {
                    cell_probabilities[c * arms + a] = context_probability * p;
                }
            }
            let compositions = self.composition_law(&cell_probabilities, batch)?;
            for (composition, weight) in compositions {
                let mut grown = key.clone();
                for (cell, &count) in composition.iter().enumerate() {
                    grown[current + cell] += count;
                }
                *next.entry(grown).or_insert(0.0) += probability * weight;
            }
        }
        Ok(next)
    }

    /// Reveal one assigned unit of `cell`.
    fn reveal_unit(&mut self, live: StateMap, cell: usize) -> Result<StateMap, Error> {
        let arms = self.layout.arms;
        let (context, assigned) = (cell / arms, cell % arms);
        let delivered = deliveries(self.spec, assigned);
        let attribution = self.spec.exposure().attribution;
        let retain_success = self.spec.observation().retain_success;
        let retain_failure = self.spec.observation().retain_failure;
        let mut next: StateMap = BTreeMap::new();
        for (key, probability) in live {
            for &(arm, delivery) in &delivered {
                let delivered_cell = context * arms + arm;
                let attributed = match attribution {
                    Attribution::Assigned => cell,
                    Attribution::Delivered => delivered_cell,
                };
                let success = match self.kernel {
                    OutcomeKernel::Fixed { success_probabilities } => {
                        success_probabilities[delivered_cell]
                    }
                    OutcomeKernel::PriorPredictive { priors } => {
                        let full = self.layout.full_offset + 2 * delivered_cell;
                        let (s, f) = (f64::from(key[full]), f64::from(key[full + 1]));
                        (priors[delivered_cell].alpha + s)
                            / (priors[delivered_cell].alpha + priors[delivered_cell].beta + s + f)
                    }
                };
                for (outcome, outcome_probability, retained) in
                    [(0_usize, 1.0 - success, retain_failure), (1, success, retain_success)]
                {
                    let mass = probability * delivery * outcome_probability;
                    if mass <= 0.0 {
                        continue;
                    }
                    let mut updated = key.clone();
                    if self.kernel.is_predictive() {
                        updated[self.layout.full_offset + 2 * delivered_cell + (1 - outcome)] += 1;
                    }
                    // Index 0 of a pair counts successes, index 1 failures.
                    let slot = 2 * attributed + (1 - outcome);
                    if retained < 1.0 {
                        let lost = updated.clone();
                        *next.entry(lost).or_insert(0.0) += mass * (1.0 - retained);
                    }
                    updated[slot] += 1;
                    *next.entry(updated).or_insert(0.0) += mass * retained;
                }
            }
        }
        self.tick(next.len() as u64 + 1)?;
        Ok(next)
    }

    fn reveal(&mut self, live: StateMap) -> Result<StateMap, Error> {
        let cells = self.layout.cells;
        let pending = self.layout.pending_offset;
        let lag = self.layout.lag;
        let mut next: StateMap = BTreeMap::new();
        for (key, probability) in live {
            let batch: Vec<u16> = key[pending..pending + cells].to_vec();
            let mut base = key.clone();
            for slot in 0..lag {
                for cell in 0..cells {
                    base[pending + slot * cells + cell] = key[pending + (slot + 1) * cells + cell];
                }
            }
            for cell in 0..cells {
                base[pending + lag * cells + cell] = 0;
            }
            let mut local: StateMap = BTreeMap::new();
            local.insert(base, probability);
            for (cell, &count) in batch.iter().enumerate() {
                for _ in 0..count {
                    local = self.reveal_unit(local, cell)?;
                }
            }
            for (state, mass) in local {
                *next.entry(state).or_insert(0.0) += mass;
            }
        }
        Ok(next)
    }
}

/// The exact law of the terminal state of `spec` under `kernel`.
///
/// # Errors
/// A kernel whose shape does not match the procedure, a state or work bound
/// exceeded, a numerical failure of a posterior, or cancellation.
pub fn enumerate_law(
    spec: &ProcedureSpec,
    kernel: &OutcomeKernel,
    cancelled: &dyn Fn() -> bool,
) -> Result<ProcedureLaw, Error> {
    kernel.check(spec.cells())?;
    let layout = Layout::new(spec, kernel.is_predictive());
    let cumulative = spec.cumulative_enrolment();
    let total_length = layout.total;
    let observed_len = layout.observed_len;
    let mut engine =
        Engine { spec, kernel, layout, rules: Rules::new(spec, cancelled), work: 0, cancelled };
    let mut live: StateMap = BTreeMap::new();
    live.insert(vec![0_u16; total_length], 1.0);
    let mut terminal: BTreeMap<(usize, bool, Vec<u16>, Option<usize>), f64> = BTreeMap::new();
    let looks = spec.enrolment().len();
    let mut peak = 1;
    for (look, &batch) in spec.enrolment().iter().enumerate() {
        if cancelled() {
            return Err(Error::Cancelled);
        }
        live = engine.enrol(live, batch)?;
        live = engine.reveal(live)?;
        peak = peak.max(live.len());
        if live.len() > MAX_LAW_STATES {
            return Err(Error::Limit("the procedure law holds too many live states"));
        }
        let last = look + 1 == looks;
        let mut continuing: StateMap = BTreeMap::new();
        for (key, probability) in live {
            let observed = &key[..observed_len];
            let assessment = engine.rules.assess(look, cumulative[look], observed)?;
            if assessment.stop || last {
                let decision = if assessment.stop {
                    assessment.decision.or_else(|| engine.rules.select(observed))
                } else {
                    engine.rules.select(observed)
                };
                *terminal
                    .entry((look, assessment.stop, observed.to_vec(), decision))
                    .or_insert(0.0) += probability;
            } else {
                continuing.insert(key, probability);
            }
        }
        live = continuing;
    }
    let atoms: Vec<TerminalAtom> = terminal
        .into_iter()
        .map(|((look, rule_stopped, observed, decision), probability)| TerminalAtom {
            look,
            rule_stopped,
            observed,
            decision,
            enrolled: cumulative[look],
            probability,
        })
        .collect();
    let mass: f64 = atoms.iter().map(|a| a.probability).sum();
    if mass.is_nan() || (mass - 1.0).abs() >= 1e-9 {
        return Err(Error::Numerical("the procedure law does not sum to one"));
    }
    Ok(ProcedureLaw {
        atoms,
        cumulative_enrolment: cumulative,
        arms: spec.arms(),
        cells: spec.cells(),
        work_units: engine.work,
        peak_states: peak,
    })
}
