//! Same-stream observation, finite belief update and preposterior decisions.

use std::sync::Arc;

use antecedent_core::ExecutionContext;
use antecedent_core::execution::{AdmissionLease, CausalRng, RngFactory};
use antecedent_design::decision_eval::evaluate_contract;
use antecedent_design::evsi::{CostToUtilityMap, StudyCostSpec};
use antecedent_design::{
    DecisionContract, DecisionCriterion, DecisionProblem, DecisionSignal, DesignError,
    PreposteriorAnalysis, Utility,
};
use serde::{Deserialize, Serialize};

use super::dynamics::CompiledWorlds;
use super::hawkes::{self, HawkesParameters};
use super::{
    Arrival, CAPABILITY, EventError, EventHistory, EventLaw, EventLawMeaning, EventWindow,
    EventWorldRequest, MAX_EVENTS, PreparedKernel, admit, digest, history_bytes, path_bytes,
    release_supported, validate_arrivals, validate_budget,
};

/// One weighted hypothesis conditional on the declared initial history.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EventHypothesis {
    /// Prior probability weight.
    pub weight: f64,
    /// Stable fixed-parameter Hawkes model.
    pub parameters: HawkesParameters,
}

/// A finite conditional belief and the history already used in it.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EventBelief {
    /// Data available when the finite belief was declared.
    pub history: EventHistory,
    /// Explicit prior weights and model hypotheses.
    pub hypotheses: Vec<EventHypothesis>,
}

impl EventBelief {
    fn validate(&self, paths: usize) -> Result<(), EventError> {
        self.history.validate(true)?;
        let n = self.hypotheses.len();
        if !(1..=32).contains(&n) || !paths.is_multiple_of(n) || paths / n < 8 {
            return Err(EventError::invalid(
                "use at most 32 hypotheses and at least eight equally allocated paths per hypothesis",
            ));
        }
        let mut sum = 0.0;
        for h in &self.hypotheses {
            h.parameters.validate(self.history.catalog.len())?;
            if !h.weight.is_finite() || h.weight <= 0.0 {
                return Err(EventError::invalid(
                    "parameter hypotheses require positive finite weights",
                ));
            }
            sum += h.weight;
        }
        if !sum.is_finite() || (sum - 1.0).abs() > 1e-10 {
            return Err(EventError::invalid("hypothesis weights must sum to one"));
        }
        Ok(())
    }
}

/// A newly observed, complete segment of the same stream.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EventObservation {
    /// Observation window beginning exactly at the prior cutoff.
    pub window: EventWindow,
    /// Exact stable source of the observed segment. Must match the prior stream.
    pub source: String,
    /// Arrivals in the new segment.
    pub events: Vec<Arrival>,
}

/// Requested observation policy and continuation sample for a future study.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EventStudyRequest {
    /// Same-stream observation ends at this decision time.
    pub decision_time: f64,
    /// End of the decision outcome horizon.
    pub end: f64,
    /// Count this event type.
    pub kind: usize,
    /// The signal is true if this many or more arrivals are observed.
    pub minimum_count: usize,
    /// Total independent paths, equally stratified over the hypotheses.
    pub paths: usize,
    /// Max arrivals in either segment, per path.
    pub max_events: usize,
    /// Event-stream seed.
    pub seed: u64,
}

/// Native value of the declared future observation under the modeled continuation.
#[derive(Clone, Debug, Serialize)]
pub struct EventStudyValue {
    /// Future-study compiler program identity.
    pub program: String,
    /// Predictive masses for false and true.
    pub signal_probabilities: [f64; 2],
    /// Gross utility of observing the signal, at the same decision time as the comparator.
    pub gross_value: f64,
    /// Approximate delete-one-block Monte Carlo standard error, when both branches remain resolved.
    pub monte_carlo_standard_error: Option<f64>,
    /// Gross value less the declared converted study cost, when supplied.
    pub net_value: Option<f64>,
    /// Best allowed action without the signal.
    pub baseline_action: String,
    /// Best allowed action given each signal branch with positive simulated mass.
    pub branch_actions: [Option<String>; 2],
    /// Canonical identity of the already-used history and belief.
    pub prior: String,
    /// Common decision time for observing and forgoing the observation.
    pub decision_time: f64,
    /// Whether this route has earned release support.
    pub release_supported: bool,
    /// The signal prices the release only when every hypothesis is the kernel the bridge kept.
    pub release_information: bool,
}

/// Compiled actual-observation update and continuation forecast.
pub struct PreparedEventUpdate {
    prior: EventBelief,
    observation: EventObservation,
    forecast: EventWindow,
    paths: usize,
    max_events: usize,
    seed: u64,
    plan: PreparedKernel,
    _lease: AdmissionLease,
}

/// Compiled future-study experiment with its continuation and terminal actions.
pub struct PreparedEventStudy {
    prior: EventBelief,
    request: EventStudyRequest,
    worlds: CompiledWorlds,
    contract: DecisionContract,
    cost: Option<f64>,
    prior_identity: String,
    plan: PreparedKernel,
    release_information: bool,
    _lease: AdmissionLease,
}

impl crate::Study {
    /// Compile a same-stream update that uses the new observation once.
    ///
    /// # Errors
    /// Incomplete or overlapping windows, invalid prior, or resource refusal.
    #[allow(
        clippy::float_cmp,
        reason = "window boundaries are copied values and must meet exactly"
    )]
    pub fn prepare_event_update(
        prior: EventBelief,
        observation: EventObservation,
        forecast: EventWindow,
        paths: usize,
        max_events: usize,
        seed: u64,
        ctx: &ExecutionContext,
    ) -> Result<PreparedEventUpdate, EventError> {
        super::super::lowering::refuse_reentry(ctx)?;
        validate_budget(paths, max_events)?;
        prior.validate(paths)?;
        observation.window.validate()?;
        forecast.validate()?;
        if observation.source != prior.history.source
            || observation.window.start != prior.history.window.end
            || forecast.start != observation.window.end
            || observation.events.len() > max_events
            || prior.history.left.len() + prior.history.events.len() + observation.events.len()
                > MAX_EVENTS
        {
            return Err(EventError::invalid(
                "update must be the next complete window from the exact prior stream",
            ));
        }
        validate_arrivals(
            &observation.events,
            observation.window.start,
            observation.window.end,
            prior.history.catalog.len(),
            true,
        )?;
        let source = digest(
            "antecedent.events.update.v1",
            &(&prior, &observation, forecast, paths, max_events, seed),
        )?;
        let plan = PreparedKernel::new(source, "update_event_belief", CAPABILITY)?;
        let lease = admit(
            ctx,
            0,
            history_bytes(&prior.history) + observation.events.len() as u64 * 16 + 128_000,
        )?;
        Ok(PreparedEventUpdate {
            prior,
            observation,
            forecast,
            paths,
            max_events,
            seed,
            plan,
            _lease: lease,
        })
    }

    /// Compile observation and no-observation alternatives at the same decision time.
    ///
    /// # Errors
    /// Invalid belief, world declaration, constraints, score criterion or cost.
    #[allow(
        clippy::needless_pass_by_value,
        reason = "the optional cost is part of the public signature and is passed as an owned value"
    )]
    pub fn prepare_event_study(
        prior: EventBelief,
        request: EventStudyRequest,
        worlds: EventWorldRequest,
        contract: DecisionContract,
        study_cost: Option<(StudyCostSpec, CostToUtilityMap)>,
        ctx: &ExecutionContext,
    ) -> Result<PreparedEventStudy, EventError> {
        super::super::lowering::refuse_reentry(ctx)?;
        validate_budget(request.paths, request.max_events)?;
        prior.validate(request.paths)?;
        let observation =
            EventWindow { start: prior.history.window.end, end: request.decision_time };
        let continuation = EventWindow { start: request.decision_time, end: request.end };
        observation.validate()?;
        continuation.validate()?;
        if request.kind >= prior.history.catalog.len()
            || request.minimum_count == 0
            || request.minimum_count > request.max_events
            || contract.criterion != DecisionCriterion::PosteriorExpectedUtility
            || !contract.constraints.is_empty()
        {
            return Err(EventError::invalid(
                "event study requires a count threshold and unconstrained expected utility",
            ));
        }
        let cost = study_cost
            .as_ref()
            .map(|(cost, map)| {
                if cost.unit.trim().is_empty()
                    || map.utility_unit != contract.utility_units
                    || map.cost_unit != cost.unit
                    || !cost.amount.is_finite()
                    || cost.amount < 0.0
                    || !map.utility_per_cost.is_finite()
                    || map.utility_per_cost <= 0.0
                    || !(cost.amount * map.utility_per_cost).is_finite()
                {
                    return Err(EventError::invalid(
                        "study cost and conversion require matching finite utility units",
                    ));
                }
                Ok(cost.amount * map.utility_per_cost)
            })
            .transpose()?;
        let prior_digest = digest("antecedent.events.prior.v1", &prior)?;
        let prior_identity = prior_digest.to_hex();
        let cost_declaration = study_cost
            .as_ref()
            .map(|(c, m)| (&c.unit, c.amount, &m.utility_unit, m.utility_per_cost));
        let source = digest(
            "antecedent.events.study.v1",
            &(
                &prior_identity,
                &request,
                &worlds,
                contract.identity().map_err(EventError::invalid)?,
                cost_declaration,
            ),
        )?;
        let lease = admit(ctx, 0, history_bytes(&prior.history) + 2 * 1024 * 1024)?;
        let worlds = CompiledWorlds::new(worlds, &prior.history, continuation, request.paths, ctx)?;
        worlds.validate_decision(&contract)?;
        let plan =
            PreparedKernel::composed(source, prior_digest, "future_event_study", Some(&contract))?;
        Ok(PreparedEventStudy {
            prior,
            request,
            worlds,
            contract,
            cost,
            prior_identity,
            plan,
            release_information: false,
            _lease: lease,
        })
    }

    /// Compile the same study only when the belief is the kernel a bridge kept.
    ///
    /// # Errors
    /// A hypothesis is not that kernel, the kernel is marked, or the study declaration is invalid.
    pub fn prepare_release_study(
        accepted: &super::question::AcceptedExcitation,
        prior: EventBelief,
        request: EventStudyRequest,
        worlds: EventWorldRequest,
        contract: DecisionContract,
        study_cost: Option<(StudyCostSpec, CostToUtilityMap)>,
        ctx: &ExecutionContext,
    ) -> Result<PreparedEventStudy, EventError> {
        let Some(parameters) = accepted.catalog_parameters() else {
            return Err(EventError::Fit(
                "the observation study prices a catalog kernel the bridge kept",
            ));
        };
        if prior.hypotheses.iter().any(|hypothesis| {
            hypothesis.parameters.baseline != parameters.baseline
                || hypothesis.parameters.branching != parameters.branching
                || hypothesis.parameters.decay != parameters.decay
        }) {
            return Err(EventError::Fit("value of information uses the kernel the bridge kept"));
        }
        let mut prepared =
            Self::prepare_event_study(prior, request, worlds, contract, study_cost, ctx)?;
        prepared.release_information = true;
        Ok(prepared)
    }
}

impl PreparedEventUpdate {
    /// Inspect the compiler plan before updating.
    pub fn plan_view(&self) -> crate::planner::PlanView {
        self.plan.plan_view()
    }

    /// Execute the exact finite-hypothesis likelihood update and sample its continuation.
    ///
    /// # Errors
    /// Zero predictive mass of the observation under all hypotheses, or execution refusal.
    pub fn execute(&self, ctx: &ExecutionContext) -> Result<EventLaw, EventError> {
        let (law, _) = self.execute_with_belief(ctx)?;
        Ok(law)
    }

    /// Execute and retain the updated belief for further disjoint observations.
    ///
    /// # Errors
    /// Zero predictive mass, or execution refusal.
    pub fn execute_with_belief(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<(EventLaw, EventBelief), EventError> {
        let scoped = super::scoped_execution(ctx, self.plan.identity(), CAPABILITY)?;
        let ctx = &scoped;
        let mut work = hawkes::Work::new(ctx);
        let original = &self.prior.history;
        let scratch = history_bytes(original) * 4
            + self.max_events as u64 * (64 + 8 * original.catalog.len() as u64);
        let _scratch = admit(ctx, scratch, 0)?;
        let mut history = original.clone();
        history.left = original.prefix();
        history.events.clone_from(&self.observation.events);
        history.window = self.observation.window;
        let likelihood = self
            .prior
            .hypotheses
            .iter()
            .map(
                |h| Ok(h.weight.ln() + hawkes::log_likelihood(&history, &h.parameters, &mut work)?),
            )
            .collect::<Result<Vec<_>, EventError>>()?;
        let top = likelihood.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if !top.is_finite() {
            return Err(EventError::Numerical(
                "observation has zero likelihood under the finite belief",
            ));
        }
        let weights: Vec<_> = likelihood.into_iter().map(|v| (v - top).exp()).collect();
        let total = weights.iter().sum::<f64>();
        let hypotheses: Vec<_> = self
            .prior
            .hypotheses
            .iter()
            .zip(weights)
            .filter(|&(_, weight)| weight > 0.0)
            .map(|(h, weight)| EventHypothesis {
                weight: weight / total,
                parameters: h.parameters.clone(),
            })
            .collect();
        let lease = Arc::new(admit(
            ctx,
            0,
            path_bytes(self.paths, self.max_events) + history_bytes(&history) + 128_000,
        )?);
        let rng = RngFactory::from_seed(self.seed);
        let prefix = history.prefix();
        let mut paths = Vec::with_capacity(self.paths);
        let mut row_weights = Vec::with_capacity(self.paths);
        let mut cursor = 0;
        // Preserve the declared sampling allocation even when a hypothesis is ruled out.
        let group = self.paths / self.prior.hypotheses.len();
        for hypothesis in &hypotheses {
            for _ in 0..group {
                paths.push(hawkes::sample(
                    &history,
                    &prefix,
                    &hypothesis.parameters,
                    self.forecast,
                    (&rng, hawkes::UPDATE, cursor),
                    self.max_events,
                    &mut work,
                )?);
                row_weights.push(hypothesis.weight / group as f64);
                cursor += 1;
            }
        }
        // A smaller law makes the allocation inspectable after exact zero-weight removal.
        let law = EventLaw {
            history: Arc::new(history.clone()),
            window: self.forecast,
            paths: Arc::new(paths),
            weights: Some(Arc::new(row_weights)),
            path_left: None,
            meaning: EventLawMeaning::WeightedBelief,
            identity: self.plan.identity(),
            provider: None,
            provider_receipt: None,
            model_fit: None,
            refit_parameters: None,
            release_supported: release_supported(CAPABILITY),
            _lease: lease,
        };
        Ok((law, EventBelief { history, hypotheses }))
    }
}

impl PreparedEventStudy {
    /// Inspect without generating arrivals or solving worlds.
    pub fn plan_view(&self) -> crate::planner::PlanView {
        self.plan.plan_view()
    }

    /// Evaluate a binary count signal on paired continuations.
    ///
    /// # Errors
    /// Numerical, resource, decision-contract or cancellation refusal.
    #[allow(
        clippy::too_many_lines,
        reason = "one linear sequence of steps that reads top to bottom"
    )]
    pub fn execute(&self, ctx: &ExecutionContext) -> Result<EventStudyValue, EventError> {
        let scoped = super::scoped_execution(ctx, self.plan.identity(), CAPABILITY)?;
        let ctx = &scoped;
        let r = &self.request;
        let history = &self.prior.history;
        let observation = EventWindow { start: history.window.end, end: r.decision_time };
        let continuation = EventWindow { start: r.decision_time, end: r.end };
        let prefix = history.prefix();
        let lease = Arc::new(admit(
            ctx,
            history_bytes(history) + path_bytes(1, r.max_events),
            path_bytes(r.paths, r.max_events) * 2 + history_bytes(history),
        )?);
        let mut work = hawkes::Work::new(ctx);
        let rng = RngFactory::from_seed(r.seed);
        let mut paths = Vec::with_capacity(r.paths);
        let mut prior_paths = Vec::with_capacity(r.paths);
        let mut observed = Vec::with_capacity(r.paths);
        let mut weights = Vec::with_capacity(r.paths);
        let group = r.paths / self.prior.hypotheses.len();
        for hypothesis in &self.prior.hypotheses {
            for _ in 0..group {
                let index = paths.len() as u64;
                let before = hawkes::sample(
                    history,
                    &prefix,
                    &hypothesis.parameters,
                    observation,
                    (&rng, hawkes::STUDY, 2 * index),
                    r.max_events,
                    &mut work,
                )?;
                let mut combined = prefix.clone();
                combined.extend_from_slice(&before);
                let after = hawkes::sample(
                    history,
                    &combined,
                    &hypothesis.parameters,
                    continuation,
                    (&rng, hawkes::STUDY, 2 * index + 1),
                    r.max_events,
                    &mut work,
                )?;
                observed.push(
                    before.iter().filter(|event| event.kind == r.kind).count() >= r.minimum_count,
                );
                weights.push(hypothesis.weight / group as f64);
                prior_paths.push(before);
                paths.push(after);
            }
        }
        let law = EventLaw {
            history: Arc::new(history.clone()),
            window: continuation,
            paths: Arc::new(paths),
            weights: Some(Arc::new(weights.clone())),
            path_left: Some(Arc::new(prior_paths)),
            meaning: EventLawMeaning::WeightedBelief,
            identity: self.plan.identity(),
            provider: None,
            provider_receipt: None,
            model_fit: None,
            refit_parameters: None,
            release_supported: release_supported(CAPABILITY),
            _lease: lease,
        };
        let result = self.worlds.execute(&law, self.plan.identity(), None, ctx)?;
        let summarize = |deleted: Option<usize>| {
            let mut rows = [Vec::new(), Vec::new()];
            let mut masses = [0.0; 2];
            let mut row_weights = [Vec::new(), Vec::new()];
            let width = result.distribution.shape()[1];
            for (i, row) in result.distribution.draws().chunks_exact(width).enumerate() {
                if deleted == Some((i % group) % 8) {
                    continue;
                }
                let branch = usize::from(observed[i]);
                rows[branch].extend_from_slice(row);
                masses[branch] += weights[i];
                row_weights[branch].push(weights[i]);
            }
            let mut utilities = [Vec::new(), Vec::new()];
            for (branch, values) in rows.into_iter().enumerate() {
                if masses[branch] > 0.0 {
                    let subset = digest(
                        "antecedent.events.signal.branch.v1",
                        &(self.plan.identity().to_hex(), branch, deleted),
                    )?;
                    let law = self.worlds.artifact(
                        &law,
                        subset,
                        values,
                        Some(std::mem::take(&mut row_weights[branch])),
                    )?;
                    utilities[branch] = evaluate_contract(&self.contract, &law)
                        .map_err(EventError::invalid)?
                        .actions
                        .into_iter()
                        .map(|action| action.expected_utility)
                        .collect();
                }
            }
            let total = masses[0] + masses[1];
            for mass in &mut masses {
                *mass /= total;
            }
            preposterior(&utilities, masses, self.contract.actions.len())
                .map(|values| (masses, values))
        };
        let (masses, (gross, base, branches)) = summarize(None)?;
        let se = if masses.iter().all(|&mass| mass > 0.0) && group.is_multiple_of(8) {
            let mut deletes = Vec::with_capacity(8);
            for block in 0..8 {
                super::poll(ctx)?;
                let (probabilities, (value, _, _)) = summarize(Some(block))?;
                if probabilities.contains(&0.0) {
                    break;
                }
                deletes.push(value);
            }
            if deletes.len() == 8 {
                let mean = deletes.iter().sum::<f64>() / 8.0;
                Some(
                    (7.0 / 8.0 * deletes.iter().map(|value| (value - mean).powi(2)).sum::<f64>())
                        .sqrt(),
                )
            } else {
                None
            }
        } else {
            None
        };
        let actions = &self.contract.actions;
        Ok(EventStudyValue {
            program: self.plan.identity().to_hex(),
            signal_probabilities: masses,
            gross_value: gross,
            monte_carlo_standard_error: se,
            net_value: self.cost.map(|cost| gross - cost),
            baseline_action: actions[base].id.clone(),
            branch_actions: branches.map(|index| index.map(|i| actions[i].id.clone())),
            prior: self.prior_identity.clone(),
            decision_time: r.decision_time,
            release_supported: release_supported(CAPABILITY),
            release_information: self.release_information,
        })
    }
}

struct BranchUtility([Vec<f64>; 2]);

impl Utility<usize, usize> for BranchUtility {
    fn evaluate_batch(
        &self,
        actions: &[usize],
        states: &[usize],
        output: &mut [f64],
    ) -> Result<(), DesignError> {
        for (a, &action) in actions.iter().enumerate() {
            for (s, &state) in states.iter().enumerate() {
                output[a * states.len() + s] = self.0[state][action];
            }
        }
        Ok(())
    }
}

struct ObservedBranch;

impl DecisionSignal<usize> for ObservedBranch {
    fn name(&self) -> &'static str {
        "same_stream_count_threshold"
    }
    fn finite_support(&self, _n: u64) -> Option<Vec<f64>> {
        Some(vec![0.0, 1.0])
    }
    #[allow(
        clippy::float_cmp,
        reason = "the statistic and the state are both integer counts held in f64, so equality is exact"
    )]
    fn log_likelihood(
        &self,
        statistic: f64,
        _n: u64,
        states: &[usize],
        output: &mut [f64],
    ) -> Result<(), DesignError> {
        for (&state, output) in states.iter().zip(output) {
            *output = if statistic == state as f64 { 0.0 } else { f64::NEG_INFINITY };
        }
        Ok(())
    }
    fn sample_statistic(
        &self,
        state: &usize,
        _n: u64,
        _rng: &mut CausalRng,
    ) -> Result<f64, DesignError> {
        Ok(*state as f64)
    }
}

fn preposterior(
    utilities: &[Vec<f64>; 2],
    weights: [f64; 2],
    actions: usize,
) -> Result<(f64, usize, [Option<usize>; 2]), EventError> {
    let states: Vec<_> = (0..2).filter(|&i| weights[i] > 0.0).collect();
    let weights_retained: Vec<_> = states.iter().map(|&i| weights[i]).collect();
    let problem = DecisionProblem::new(
        (0..actions).collect(),
        Arc::new(BranchUtility(utilities.clone())),
        vec![],
    );
    let analysis =
        PreposteriorAnalysis::new_weighted(&problem, &states, &weights_retained, &ObservedBranch)
            .map_err(EventError::invalid)?;
    let value = analysis
        .exact_evsi(1)
        .map_err(EventError::invalid)?
        .ok_or(EventError::Numerical("binary signal has no finite support"))?;
    let best = |values: &[f64]| {
        values
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
            .map(|(i, _)| i)
    };
    let baseline: Vec<_> =
        (0..actions).map(|a| states.iter().map(|&s| weights[s] * utilities[s][a]).sum()).collect();
    Ok((
        value,
        best(&baseline).ok_or(EventError::invalid("empty action set"))?,
        [best(&utilities[0]), best(&utilities[1])],
    ))
}

#[cfg(test)]
mod tests {
    #[test]
    #[allow(clippy::float_cmp, reason = "the expected values are exact sums of dyadic rationals")]
    fn finite_same_stream_observation_changes_choice() {
        let scores = [vec![10.0, 0.0], vec![0.0, 18.0]];
        let (value, base, actions) = super::preposterior(&scores, [0.75, 0.25], 2).unwrap();
        assert_eq!((value, base, actions), (4.5, 0, [Some(0), Some(1)]));
        let constant = [vec![10.0, 0.0], vec![15.0, 8.0]];
        assert_eq!(super::preposterior(&constant, [0.75, 0.25], 2).unwrap().0, 0.0);
    }
}
