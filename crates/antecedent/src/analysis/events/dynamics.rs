//! Composition of retained joint event paths with the cyclic SDE executor.

use std::collections::BTreeSet;

use antecedent_contract::{QuantityRole, ScientificQuantity};
use antecedent_core::execution::AdmissionLease;
use antecedent_core::{ExecutionContext, SemanticDigest};
use antecedent_design::DecisionContract;
use antecedent_design::decision_eval::{DecisionResult, evaluate_contract};
use antecedent_io_core::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DistributionIdentity, DistributionIntakeLimits,
    DistributionMetadata, DistributionProvenance, DistributionTrust, DrawAlignment,
};
use antecedent_io_core::quantity_wire::DistributionMeaningWire;
use antecedent_model::mechanism_expr::MechanismExpression;
use antecedent_model::nonlinear_sde::{
    BASELINE_WORLD, CompiledSde, MAX_SDE_KNOTS, SdeInput, SdeInterpolation, SdeSimulationPolicy,
    SdeSystem, SdeWorld, solve_event_input_path,
};
use serde::{Deserialize, Serialize};

use super::{
    Arrival, CAPABILITY, EventError, EventLaw, EventLawMeaning, PreparedKernel, admit, digest, poll,
};

/// Bind one event type to one shared drift input through an exponential pulse.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EventBinding {
    /// Event type index in the retained catalog.
    pub kind: usize,
    /// Existing external input name in the SDE system.
    pub input: String,
    /// Pulse magnitude in external-input units.
    pub magnitude: f64,
    /// Exponential decay in inverse time units.
    pub decay: f64,
}

/// Full-grid functional for a named state coordinate.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum PathFunctional {
    /// Maximum of the recorded grid states.
    Maximum,
    /// Trapezoidal integral of the recorded grid states.
    Integral,
    /// One when any grid state strictly exceeds the threshold.
    Exceeds {
        /// Threshold in state units.
        threshold: f64,
    },
}

/// Named causal quantity in one intervention world.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PathOutput {
    /// Name matched to a declared SDE state.
    pub state: String,
    /// Exact world ID, including `baseline`.
    pub world: String,
    /// Schema-qualified identity of the state.
    pub variable_id: String,
    /// Units of the output functional.
    pub units: String,
    /// Full-grid reduction.
    pub functional: PathFunctional,
}

impl PathOutput {
    /// Quantity coordinate accepted by the existing decision contract.
    #[must_use]
    pub fn quantity(&self, population: &str, horizon: u32) -> ScientificQuantity {
        let functional_id = match self.functional {
            PathFunctional::Maximum => "grid_maximum".into(),
            PathFunctional::Integral => "grid_trapezoidal_integral".into(),
            PathFunctional::Exceeds { threshold } => {
                format!("grid_exceeds_{:016x}", threshold.to_bits())
            }
        };
        ScientificQuantity {
            variable_id: self.variable_id.clone(),
            variable_name: self.state.clone(),
            role: QuantityRole::Outcome,
            units: self.units.clone(),
            population_id: population.to_owned(),
            regime_id: self.world.clone(),
            horizon,
            functional_id,
            conditioning: Vec::new(),
            transform_id: "identity".to_owned(),
        }
    }
}

/// Causal world and output declaration shared by all event sources.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EventWorldRequest {
    /// Certified dynamic structural equations.
    pub system: SdeSystem,
    /// Population identity used in decision contracts.
    pub population: String,
    /// Stable decision horizon coordinate used in the decision contract.
    pub horizon: u32,
    /// Units of the SDE clock.
    pub time_unit: String,
    /// Origin of the SDE clock.
    pub origin: String,
    /// Common fixed grid, initial time and Brownian seed.
    pub policy: SdeSimulationPolicy,
    /// Common initial state in system order, conditional at the start of the grid.
    pub initial: Vec<f64>,
    /// Interventions in addition to the implicit baseline.
    pub worlds: Vec<SdeWorld>,
    /// Event-to-input pulse declarations.
    pub bindings: Vec<EventBinding>,
    /// Joint pathwise outcomes.
    pub outputs: Vec<PathOutput>,
}

/// A compiled joint response program over a retained event law.
pub struct PreparedEventWorlds {
    compiled: CompiledWorlds,
    law: EventLaw,
    contract: Option<DecisionContract>,
    plan: PreparedKernel,
    _lease: AdmissionLease,
}

/// Grid comparison per output, in declared order.
#[derive(Clone, Debug, Serialize)]
pub struct GridDifference {
    /// Weighted mean of the paired coarse-minus-fine differences.
    pub mean: f64,
    /// Weighted root-mean-square paired difference.
    pub rms: f64,
    /// Paired IID or independent-refit-block standard error, where available.
    pub standard_error: Option<f64>,
    /// Whether the root-mean-square discrepancy clears the declared grid tolerance.
    pub within_tolerance: bool,
}

/// Joint predictive world law and optional evaluation by the existing decision engine.
pub struct EventDecision {
    /// Retained draw-major law with one paired row per path.
    pub distribution: DistributionArtifact,
    /// Existing decision-evaluation result, if requested.
    pub decision: Option<DecisionResult>,
    /// Paired grid differences in output order.
    pub grid: Vec<GridDifference>,
    /// Monte Carlo standard errors of output means in their declared order.
    /// Each recursive refit contributes one independent cluster.
    pub outcome_monte_carlo_standard_errors: Vec<Option<f64>>,
    /// Forecast interpretation retained alongside the predictive outcome artifact.
    pub event_meaning: EventLawMeaning,
    /// Exact external event-provider verification digest, when present.
    pub provider_receipt: Option<String>,
    /// Release support of the composed route and source.
    pub release_supported: bool,
    _lease: AdmissionLease,
}

pub(super) struct CompiledWorlds {
    pub(super) request: EventWorldRequest,
    compiled: CompiledSde,
    worlds: Vec<SdeWorld>,
    slots: Vec<(usize, usize)>,
    quantities: Vec<ScientificQuantity>,
}

impl crate::Study {
    /// Compile a retained event law into paired intervention worlds.
    ///
    /// # Errors
    /// Clock, path or world mismatch, unsupported bindings or resource refusal.
    pub fn prepare_event_worlds(
        law: &EventLaw,
        request: EventWorldRequest,
        contract: Option<DecisionContract>,
        ctx: &ExecutionContext,
    ) -> Result<PreparedEventWorlds, EventError> {
        super::super::lowering::refuse_reentry(ctx)?;
        let contract_id = contract
            .as_ref()
            .map(DecisionContract::identity)
            .transpose()
            .map_err(EventError::invalid)?;
        let source = digest(
            "antecedent.events.worlds.v1",
            &(law.identity().to_hex(), &request, contract_id),
        )?;
        let plan = PreparedKernel::composed(
            source,
            law.identity(),
            "paired_event_worlds",
            contract.as_ref(),
        )?;
        let lease = admit(ctx, 0, 2 * 1024 * 1024)?;
        let compiled =
            CompiledWorlds::new(request, &law.history, law.window, law.paths.len(), ctx)?;
        if let Some(contract) = &contract {
            compiled.validate_decision(contract)?;
        }
        Ok(PreparedEventWorlds { compiled, law: law.clone(), contract, plan, _lease: lease })
    }
}

impl PreparedEventWorlds {
    /// Stable identity of the prepared world program.
    pub fn program_identity(&self) -> SemanticDigest {
        self.plan.identity()
    }

    /// Inspect the retained event-to-world compiler route.
    pub fn plan_view(&self) -> crate::planner::PlanView {
        self.plan.plan_view()
    }

    /// Execute paired worlds, retaining declared-grid outputs and their refinement checks.
    ///
    /// # Errors
    /// Incomplete path law, failed decision validation, cancellation or resource refusal.
    pub fn execute(&self, ctx: &ExecutionContext) -> Result<EventDecision, EventError> {
        let (scoped, publication) = super::scoped_execution(ctx, &self.plan)?;
        publication.finish(self.compiled.execute(
            &self.law,
            self.plan.identity(),
            self.contract.as_ref(),
            &scoped,
        ))
    }
}

impl CompiledWorlds {
    #[allow(clippy::too_many_lines, reason = "the steps read in order in one place")]
    #[allow(
        clippy::float_cmp,
        reason = "the policy grid must reproduce the retained window endpoints exactly"
    )]
    pub(super) fn new(
        request: EventWorldRequest,
        history: &super::EventHistory,
        window: super::EventWindow,
        paths: usize,
        ctx: &ExecutionContext,
    ) -> Result<Self, EventError> {
        poll(ctx)?;
        let p = &request.policy;
        if p.start != window.start
            || p.start + p.steps as f64 * p.step != window.end
            || p.paths != paths
            || request.time_unit != history.time_unit
            || request.origin != history.origin
            || !p.check_step
            || p.output_every != 1
            || p.steps.saturating_mul(2).saturating_add(1) > MAX_SDE_KNOTS
            // The SDE compiler retains at most 64 worlds, including the baseline
            // this composer adds. The cell-product check below is the execution budget.
            || request.worlds.len() + 1 > 64
            || !(1..=128).contains(&request.outputs.len())
            || !(1..=64).contains(&request.bindings.len())
        {
            return Err(EventError::invalid(
                "world clock, path count, full-grid policy or shape differs from the retained event law",
            ));
        }
        p.validate().map_err(EventError::invalid)?;
        let compiled = request.system.compile().map_err(EventError::invalid)?;
        let inputs: BTreeSet<_> =
            request.system.inputs.iter().map(|input| input.name.as_str()).collect();
        for rule in &request.bindings {
            if rule.kind >= history.catalog.len()
                || !inputs.contains(rule.input.as_str())
                || !rule.magnitude.is_finite()
                || !rule.decay.is_finite()
                || rule.decay <= 0.0
            {
                return Err(EventError::invalid(
                    "event pulse references an unknown type/input or an invalid magnitude/decay",
                ));
            }
            let input = request
                .system
                .inputs
                .iter()
                .find(|input| input.name == rule.input)
                .expect("known input");
            let slot = request.system.variables.len()
                + request
                    .system
                    .inputs
                    .iter()
                    .position(|input| input.name == rule.input)
                    .expect("known input");
            if request.system.diffusion.iter().any(|diffusion| reads(&diffusion.expression, slot))
                || !request.system.drift.iter().any(|drift| reads(drift, slot))
            {
                return Err(EventError::invalid("event-bound inputs must enter drift only"));
            }
            if input.envelope.is_none() {
                return Err(EventError::invalid(
                    "event input needs an explicit certified envelope",
                ));
            }
        }
        let named_inputs: BTreeSet<_> = request.bindings.iter().map(|b| b.input.as_str()).collect();
        let mut ids = BTreeSet::from([BASELINE_WORLD]);
        for world in &request.worlds {
            if !ids.insert(&world.id)
                || world.inputs.iter().any(|i| named_inputs.contains(i.name.as_str()))
            {
                return Err(EventError::invalid(
                    "duplicate world or replacement of an event-owned input",
                ));
            }
        }
        let worlds: Vec<_> =
            std::iter::once(SdeWorld { id: BASELINE_WORLD.into(), ..SdeWorld::default() })
                .chain(request.worlds.iter().cloned())
                .collect();
        for world in &worlds {
            request.system.plan_world(world, p.start, p.step).map_err(EventError::invalid)?;
        }
        let mut slots = Vec::with_capacity(request.outputs.len());
        let mut quantities = Vec::with_capacity(request.outputs.len());
        for output in &request.outputs {
            let w = worlds
                .iter()
                .position(|world| world.id == output.world)
                .ok_or_else(|| EventError::invalid("unknown path-output world"))?;
            let v = request
                .system
                .variables
                .iter()
                .position(|name| name == &output.state)
                .ok_or_else(|| EventError::invalid("unknown path-output state"))?;
            if matches!(output.functional, PathFunctional::Exceeds { threshold } if !threshold.is_finite() || output.units != "dimensionless")
            {
                return Err(EventError::invalid(
                    "threshold must be finite and its indicator dimensionless",
                ));
            }
            let quantity = output.quantity(&request.population, request.horizon);
            quantity.validate().map_err(|_| {
                EventError::invalid("path-output quantity has an empty or ambiguous coordinate")
            })?;
            slots.push((w, v));
            quantities.push(quantity);
        }
        let work = p
            .steps
            .saturating_mul(p.paths)
            .saturating_mul(worlds.len())
            .saturating_mul(request.system.variables.len() + request.bindings.len());
        if work > 75_000_000 {
            return Err(EventError::Budget);
        }
        Ok(Self { request, compiled, worlds, slots, quantities })
    }

    pub(super) fn validate_decision(&self, contract: &DecisionContract) -> Result<(), EventError> {
        contract.validate().map_err(EventError::invalid)?;
        if contract.target_population != self.request.population
            || contract.horizon != self.request.horizon
        {
            return Err(EventError::invalid(
                "decision population or horizon differs from compiled worlds",
            ));
        }
        for input in contract.actions.iter().flat_map(|action| &action.inputs) {
            if !self
                .quantities
                .iter()
                .any(|quantity| input.require_same_coordinate(quantity).is_ok())
            {
                return Err(EventError::invalid(
                    "decision input is absent from the compiled joint coordinates",
                ));
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_lines, reason = "the steps read in order in one place")]
    pub(super) fn execute(
        &self,
        law: &EventLaw,
        identity: SemanticDigest,
        contract: Option<&DecisionContract>,
        ctx: &ExecutionContext,
    ) -> Result<EventDecision, EventError> {
        let p = &self.request.policy;
        let width = self.quantities.len();
        let retained = p.paths as u64 * width as u64 * 8 + 128_000;
        let working = retained * 2
            + (p.steps as u64 * 2 + 1)
                * (self.request.system.inputs.len() as u64 * 64
                    + self.compiled.dimension() as u64 * 32
                    + 64);
        let lease = admit(ctx, working, retained)?;
        let mut values = vec![0.0; p.paths * width];
        let mut differences = vec![0.0; values.len()];
        let times: Vec<_> =
            (0..=2 * p.steps).map(|step| p.start + step as f64 * (p.step / 2.0)).collect();
        let prehistory = law.history.prefix();
        let initial_pulses = self.initial_pulses(&prehistory, &times, ctx)?;
        for (path, arrivals) in law.paths.iter().enumerate() {
            poll(ctx)?;
            let schedules = if let Some(left) = &law.path_left {
                let carried = self.initial_pulses(&left[path], &times, ctx)?;
                let initial: Vec<_> =
                    initial_pulses.iter().zip(carried).map(|(a, b)| a + b).collect();
                self.bind(arrivals, &times, &initial)
            } else {
                self.bind(arrivals, &times, &initial_pulses)
            };
            for (w, world) in self.worlds.iter().enumerate() {
                let failed = |error: &dyn std::fmt::Display| EventError::IncompleteLaw {
                    path,
                    world: world.id.clone(),
                    detail: error.to_string(),
                };
                let bound = SdeWorld {
                    id: world.id.clone(),
                    actions: world.actions.clone(),
                    inputs: world.inputs.iter().chain(&schedules).cloned().collect(),
                };
                let plan = self
                    .request
                    .system
                    .plan_world(&bound, p.start, p.step)
                    .map_err(|e| failed(&e))?;
                let (coarse, fine) = solve_event_input_path(
                    &self.compiled,
                    &plan,
                    &self.request.initial,
                    p,
                    path as u64,
                    ctx,
                )
                .map_err(|e| {
                    if ctx.cancellation.is_cancelled() {
                        crate::error::CausalError::Cancelled { stage: "paired event path" }.into()
                    } else {
                        failed(&e)
                    }
                })?;
                for (coordinate, (&(world_index, state), output)) in
                    self.slots.iter().zip(&self.request.outputs).enumerate()
                {
                    if world_index == w {
                        let d = self.compiled.dimension();
                        let a = summarize(&coarse, d, state, p.step, output.functional)?;
                        let b = summarize(&fine, d, state, p.step / 2.0, output.functional)?;
                        values[path * width + coordinate] = a;
                        differences[path * width + coordinate] = a - b;
                    }
                }
            }
        }
        let grid = (0..width)
            .map(|col| {
                let deltas: Vec<_> = differences[col..].iter().step_by(width).copied().collect();
                let scale = values[col..]
                    .iter()
                    .step_by(width)
                    .enumerate()
                    .map(|(i, x)| weight(law, i) * x.abs())
                    .sum::<f64>()
                    .max(1.0);
                summarize_grid(law, &deltas, p.weak_tolerance * scale)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let outcome_monte_carlo_standard_errors = (0..width)
            .map(|col| {
                let column: Vec<_> = values[col..].iter().step_by(width).copied().collect();
                let mean = column.iter().enumerate().map(|(i, value)| weight(law, i) * value).sum();
                paired_standard_error(law, &column, mean)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let distribution = self.artifact(law, identity, values, None)?;
        let decision = contract
            .map(|c| evaluate_contract(c, &distribution))
            .transpose()
            .map_err(EventError::invalid)?;
        Ok(EventDecision {
            distribution,
            decision,
            grid,
            outcome_monte_carlo_standard_errors,
            event_meaning: law.meaning(),
            provider_receipt: law.provider_receipt.clone(),
            release_supported: law.release_supported() && super::release_supported(CAPABILITY),
            _lease: lease,
        })
    }

    fn initial_pulses(
        &self,
        history: &[Arrival],
        times: &[f64],
        ctx: &ExecutionContext,
    ) -> Result<Vec<f64>, EventError> {
        self.request
            .bindings
            .iter()
            .map(|binding| {
                poll(ctx)?;
                Ok(history
                    .iter()
                    .filter(|event| event.kind == binding.kind)
                    .map(|event| {
                        binding.magnitude * (-binding.decay * (times[0] - event.time)).exp()
                    })
                    .sum())
            })
            .collect()
    }

    fn bind(&self, events: &[Arrival], times: &[f64], initial_pulses: &[f64]) -> Vec<SdeInput> {
        let mut inputs = Vec::new();
        for original in &self.request.system.inputs {
            let rules: Vec<_> = self
                .request
                .bindings
                .iter()
                .zip(initial_pulses)
                .filter(|(rule, _)| rule.input == original.name)
                .collect();
            if rules.is_empty() {
                continue;
            }
            let mut pulse: Vec<_> = rules.iter().map(|(_, initial)| **initial).collect();
            let mut cursor = 0;
            let mut previous = times[0];
            let mut values = Vec::with_capacity(times.len());
            for &time in times {
                for ((rule, _), value) in rules.iter().zip(&mut pulse) {
                    *value *= (-rule.decay * (time - previous)).exp();
                }
                while let Some(event) = events.get(cursor).filter(|event| event.time <= time) {
                    for ((rule, _), value) in rules.iter().zip(&mut pulse) {
                        if event.kind == rule.kind {
                            *value += rule.magnitude * (-rule.decay * (time - event.time)).exp();
                        }
                    }
                    cursor += 1;
                }
                values.push(original.value_at(time) + pulse.iter().sum::<f64>());
                previous = time;
            }
            inputs.push(SdeInput {
                name: original.name.clone(),
                interpolation: SdeInterpolation::PiecewiseConstant,
                times: times.to_vec(),
                values,
                envelope: None,
            });
        }
        inputs
    }

    pub(super) fn artifact(
        &self,
        law: &EventLaw,
        identity: SemanticDigest,
        values: Vec<f64>,
        weights: Option<Vec<f64>>,
    ) -> Result<DistributionArtifact, EventError> {
        let source = law.identity().to_hex();
        let causal = identity.to_hex();
        let rng = format!("events-v1/sde-counter-v1/{}", self.request.policy.seed);
        let provenance = law.provider.as_ref().map_or_else(
            || DistributionProvenance {
                source_id: source.clone(),
                provider_id: "antecedent.events".into(),
                rng_id: rng.clone(),
                snapshot_id: source.clone(),
                causal_contract_id: causal.clone(),
            },
            |provider| DistributionProvenance::from_provider(provider, &source, &rng, &causal),
        );
        let id = DistributionIdentity::new(
            DistributionMeaningWire::InterventionalPredictive,
            &self.quantities,
            DrawAlignment::Joint,
            provenance,
        )
        .map_err(EventError::invalid)?;
        let metadata = DistributionMetadata {
            version: 1,
            identity: id,
            axes: ["draw".into(), "quantity".into()],
            shape: [values.len() / self.quantities.len(), self.quantities.len()],
            weights: weights.or_else(|| law.weights().map(<[f64]>::to_vec)),
            supported: None,
            calibration: DistributionCalibration::NotMeasured,
            trust: if law.release_supported() {
                DistributionTrust::NativeSupported
            } else {
                DistributionTrust::Unverified
            },
            legacy_posterior: None,
            legacy_bindings: None,
        };
        DistributionArtifact::new_within(metadata, values, &DistributionIntakeLimits::default())
            .map_err(EventError::invalid)
    }
}

fn weight(law: &EventLaw, i: usize) -> f64 {
    law.weights().map_or_else(|| 1.0 / law.paths.len() as f64, |weights| weights[i])
}

fn summarize_grid(
    law: &EventLaw,
    deltas: &[f64],
    tolerance: f64,
) -> Result<GridDifference, EventError> {
    let mean: f64 = deltas.iter().enumerate().map(|(i, v)| weight(law, i) * v).sum();
    let rms = deltas.iter().enumerate().map(|(i, v)| weight(law, i) * v * v).sum::<f64>().sqrt();
    let standard_error = paired_standard_error(law, deltas, mean)?;
    if !rms.is_finite() {
        return Err(EventError::Numerical("nonfinite grid diagnostic"));
    }
    Ok(GridDifference { mean, rms, standard_error, within_tolerance: rms <= tolerance })
}

fn paired_standard_error(
    law: &EventLaw,
    deltas: &[f64],
    mean: f64,
) -> Result<Option<f64>, EventError> {
    let group_count = match law.meaning() {
        EventLawMeaning::Fixed | EventLawMeaning::PlugIn | EventLawMeaning::Intervened => {
            deltas.len()
        }
        EventLawMeaning::RecursiveRefit => {
            law.refit_parameters().map_or(0, <[super::HawkesParameters]>::len)
        }
        _ => 0,
    };
    let standard_error = (group_count > 1).then(|| {
        let group_size = deltas.len() / group_count;
        let squared: f64 = deltas
            .chunks_exact(group_size)
            .map(|group| {
                let average = group.iter().sum::<f64>() / group_size as f64;
                (average - mean).powi(2)
            })
            .sum();
        (squared / (group_count * (group_count - 1)) as f64).sqrt()
    });
    if !mean.is_finite() || standard_error.is_some_and(|se| !se.is_finite()) {
        return Err(EventError::Numerical("nonfinite Monte Carlo diagnostic"));
    }
    Ok(standard_error)
}

fn summarize(
    row: &[f64],
    width: usize,
    state: usize,
    step: f64,
    functional: PathFunctional,
) -> Result<f64, EventError> {
    let values = row[state..].iter().step_by(width).copied();
    let value = match functional {
        PathFunctional::Maximum => values.fold(f64::NEG_INFINITY, f64::max),
        PathFunctional::Integral => {
            let mut prev = row[state];
            values
                .skip(1)
                .map(|value| {
                    let area = (prev * 0.5 + value * 0.5) * step;
                    prev = value;
                    area
                })
                .sum()
        }
        PathFunctional::Exceeds { threshold } => {
            f64::from(u8::from(values.into_iter().any(|value| value > threshold)))
        }
    };
    if !value.is_finite() {
        return Err(EventError::Numerical("nonfinite path functional"));
    }
    Ok(value)
}

fn reads(expression: &MechanismExpression, slot: usize) -> bool {
    match expression {
        MechanismExpression::Variable { variable } => *variable == slot,
        MechanismExpression::Add { terms } | MechanismExpression::Multiply { factors: terms } => {
            terms.iter().any(|term| reads(term, slot))
        }
        MechanismExpression::Tanh { argument }
        | MechanismExpression::Sin { argument }
        | MechanismExpression::Softplus { argument } => reads(argument, slot),
        MechanismExpression::Constant { .. } | MechanismExpression::Parameter { .. } => false,
    }
}
