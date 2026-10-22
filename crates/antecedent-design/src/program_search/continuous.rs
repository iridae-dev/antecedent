//! Bounded derivative-free search over actual compiled programs.
// SPDX-License-Identifier: MIT OR Apache-2.0
use super::{
    FiniteProgramDomain, PreparedFiniteProgramSearch, ProgramCandidate, ProgramFeasibility,
    ProgramSearchError, ProgramSearchRow, ProgramSearchScope, cancelled, invalid, program_row,
    validate_domain,
};
use crate::decision_contract::{ProgramObjective, ProgramSearchContract};
use crate::scientific_source::{ScientificConsumptionScope, ScientificDecisionSource};
use antecedent_core::ExecutionContext;
use antecedent_core::SemanticDigest;
use std::sync::Arc;

/// One program compiled by a retained response owner, before numerical decision evaluation.
#[derive(Clone, Debug)]
pub struct CompiledContinuousCandidate {
    /// Actual controls and exact response quantities produced by compilation.
    pub candidate: ProgramCandidate,
    /// Target-bound source with its original scientific standing.
    pub source: ScientificDecisionSource,
}
/// Adapter implemented by native retained compiler owners; it does not estimate a new model.
pub trait CompiledProgramEvaluator: Sync {
    /// Compile/evaluate these exact controls using the same frozen fit and aligned draws.
    ///
    /// # Errors
    /// Target, numerical, resource and cancellation failures remain explicit.
    fn evaluate_program(
        &self,
        controls: &[f64],
        ctx: &ExecutionContext,
    ) -> Result<CompiledContinuousCandidate, ProgramSearchError>;
}
/// Explicit finite work and stopping policy; no derivatives or global certificate are implied.
#[derive(Clone, Debug, PartialEq)]
pub struct ContinuousSearchOptions {
    /// Maximum evaluated or operationally refused proposals, including the initial point.
    pub max_evaluations: usize,
    /// Maximum coordinate-poll iterations.
    pub max_iterations: usize,
    /// Initial fraction of each coordinate's declared bound width.
    pub initial_step: f64,
    /// Minimum absolute step before a locally stationary poll stops.
    pub step_tolerance: f64,
    /// Improvement required to move the incumbent under the supplied numerical source.
    pub objective_tolerance: f64,
}
/// Actual reason this bounded procedure stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContinuousSearchStop {
    /// A complete unsuccessful poll reached the step tolerance.
    StepTolerance,
    /// Caller evaluation budget was exhausted.
    EvaluationLimit,
    /// Caller iteration budget was exhausted.
    IterationLimit,
}
/// One actual attempt; failed numerical proposals remain visible instead of becoming zero utility.
#[derive(Clone, Debug)]
pub struct ContinuousSearchAttempt {
    /// Exact proposed controls.
    pub controls: Vec<f64>,
    /// Actual feasibility/objective when evaluation succeeded.
    pub row: Option<ProgramSearchRow>,
    /// A compiler/numerical failure, distinct from operational or risk infeasibility.
    pub failure: Option<String>,
    /// Exact target and proof, when a checked source was obtained.
    pub target_identity: Option<SemanticDigest>,
    /// Exact retained fit dependency, when evaluated.
    pub dependency_identity: Option<SemanticDigest>,
}
/// Best-found result scoped to the actual evaluated source and bounded poll procedure.
#[derive(Clone, Debug)]
pub struct ContinuousProgramSearchResult {
    /// Best feasible candidate under the supplied numerical functional; None if none succeeded.
    pub best: Option<ProgramSearchRow>,
    /// Every actual proposal and failure in execution order.
    pub attempts: Vec<ContinuousSearchAttempt>,
    /// Explicit stopping reason; step tolerance is not a global convergence theorem.
    pub stop: ContinuousSearchStop,
    /// Scientific permission requested for every candidate consumption.
    pub scope: ScientificConsumptionScope,
    /// A valid global optimality gap is unavailable to bounded pattern search.
    pub gap: Option<f64>,
    /// Always false: local numerical polling does not certify a global optimum.
    pub global_certificate: bool,
    /// Selected-decision calibration is unmeasured.
    pub reliability: &'static str,
    retention: Arc<antecedent_core::execution::AdmissionLease>,
}
impl ContinuousProgramSearchResult {
    /// Retained admitted bytes, not process RSS.
    #[must_use]
    pub fn admitted_bytes(&self) -> u64 {
        self.retention.observe_current();
        self.retention.remainder().bytes()
    }
}
/// Frozen operational/objective policy; actual response programs are supplied by their compiler owner.
#[derive(Clone, Debug)]
pub struct PreparedContinuousProgramSearch {
    contract: ProgramSearchContract,
    domain: FiniteProgramDomain,
    scope: ProgramSearchScope,
    scientific_scope: ScientificConsumptionScope,
    options: ContinuousSearchOptions,
    initial: Vec<f64>,
}
impl PreparedContinuousProgramSearch {
    /// Validate finite bounds and a bounded numerical search policy without fitting a response.
    ///
    /// # Errors
    /// Invalid controls/policy or a comparative objective that needs an explicit competitor set.
    pub fn compile(
        contract: ProgramSearchContract,
        domain: FiniteProgramDomain,
        scope: ProgramSearchScope,
        scientific_scope: ScientificConsumptionScope,
        options: ContinuousSearchOptions,
        initial: Vec<f64>,
    ) -> Result<Self, ProgramSearchError> {
        contract.validate().map_err(invalid)?;
        validate_domain(&domain, &scope)?;
        if matches!(
            contract.objective,
            ProgramObjective::Optimality(_)
                | ProgramObjective::Regret(_)
                | ProgramObjective::Robust
        ) {
            return Err(invalid(
                "continuous point polling requires a separable functional; comparative optimality/regret or robust structure requires the explicit finite-set/identified-set search",
            ));
        }
        if options.max_evaluations == 0
            || options.max_evaluations > 100_000
            || options.max_iterations == 0
            || options.max_iterations > 100_000
            || !options.initial_step.is_finite()
            || options.initial_step <= 0.0
            || options.initial_step > 1.0
            || !options.step_tolerance.is_finite()
            || options.step_tolerance <= 0.0
            || !options.objective_tolerance.is_finite()
            || options.objective_tolerance < 0.0
            || initial.len() != domain.controls.len()
            || initial.iter().any(|value| !value.is_finite())
        {
            return Err(invalid("invalid bounded continuous search policy or initial controls"));
        }
        let initial_candidate =
            ProgramCandidate { id: "initial".into(), controls: initial.clone(), inputs: vec![] };
        if program_row(&initial_candidate, &domain)?.feasibility != ProgramFeasibility::Feasible {
            return Err(invalid(
                "continuous search requires an operationally feasible initial point; no feasibility restoration is implied",
            ));
        }
        Ok(Self { contract, domain, scope, scientific_scope, options, initial })
    }
    /// Reuse the identical response/operational search space with another separable objective.
    ///
    /// # Errors
    /// Invalid or changed response input space and unsupported comparative objectives.
    pub fn with_objective(
        &self,
        contract: ProgramSearchContract,
    ) -> Result<Self, ProgramSearchError> {
        if contract.space != self.contract.space {
            return Err(invalid("objective update changed response input space"));
        }
        Self::compile(
            contract,
            self.domain.clone(),
            self.scope.clone(),
            self.scientific_scope,
            self.options.clone(),
            self.initial.clone(),
        )
    }
    /// Poll feasible numeric actions using only the retained compiler's response evaluator.
    ///
    /// # Errors
    /// Cancellation, admission, changed fit dependencies or a missing scoped source abort execution.
    pub fn execute(
        &self,
        evaluator: &impl CompiledProgramEvaluator,
        ctx: &ExecutionContext,
    ) -> Result<ContinuousProgramSearchResult, ProgramSearchError> {
        let mut rows = ctx
            .try_map_indexed(1, |_, worker| self.execute_inner(evaluator, worker))
            .map_err(|error| match error {
                antecedent_core::execution::MapExecutionError::Work(error) => error,
                antecedent_core::execution::MapExecutionError::Admission(
                    antecedent_core::execution::AdmissionError::Cancelled,
                ) => ProgramSearchError::Cancelled,
                antecedent_core::execution::MapExecutionError::Admission(error) => {
                    ProgramSearchError::Resource(error.to_string())
                }
            })?;
        rows.pop().ok_or_else(|| invalid("continuous search worker returned no result"))
    }
    fn execute_inner(
        &self,
        evaluator: &impl CompiledProgramEvaluator,
        ctx: &ExecutionContext,
    ) -> Result<ContinuousProgramSearchResult, ProgramSearchError> {
        cancelled(ctx)?;
        let bytes = self
            .options
            .max_evaluations
            .checked_mul(
                self.domain
                    .controls
                    .len()
                    .checked_mul(16)
                    .and_then(|n| n.checked_add(1024))
                    .ok_or_else(|| {
                        ProgramSearchError::Resource("continuous row width overflow".into())
                    })?,
            )
            .and_then(|n| u64::try_from(n).ok())
            .ok_or_else(|| {
                ProgramSearchError::Resource("continuous history quote overflow".into())
            })?;
        if ctx.memory_limit_bytes().is_some_and(|limit| bytes > limit) {
            return Err(ProgramSearchError::Resource(
                "continuous attempt history exceeds hard limit".into(),
            ));
        }
        let retention = Arc::new(
            ctx.acquire_admission(0, bytes)
                .map_err(|e| ProgramSearchError::Resource(e.to_string()))?,
        );
        let mut attempts = Vec::new();
        let mut dependency = None;
        let first = self.evaluate(&self.initial, evaluator, ctx, &mut dependency)?;
        let mut best = first.row.as_ref().filter(|row| row.value.is_some()).cloned();
        attempts.push(first);
        let mut incumbent = self.initial.clone();
        let mut steps = self
            .domain
            .controls
            .iter()
            .map(|c| (c.upper - c.lower) * self.options.initial_step)
            .collect::<Vec<_>>();
        let mut stop = ContinuousSearchStop::IterationLimit;
        'search: for _ in 0..self.options.max_iterations {
            cancelled(ctx)?;
            let mut improved = false;
            for coordinate in 0..incumbent.len() {
                for direction in [-1.0, 1.0] {
                    if attempts.len() >= self.options.max_evaluations {
                        stop = ContinuousSearchStop::EvaluationLimit;
                        break 'search;
                    }
                    let mut proposal = incumbent.clone();
                    let control = &self.domain.controls[coordinate];
                    proposal[coordinate] = (proposal[coordinate] + direction * steps[coordinate])
                        .clamp(control.lower, control.upper);
                    if proposal.iter().zip(&incumbent).all(|(a, b)| a.to_bits() == b.to_bits()) {
                        continue;
                    }
                    let attempt = self.evaluate(&proposal, evaluator, ctx, &mut dependency)?;
                    if let Some(row) = attempt.row.as_ref().filter(|row| row.value.is_some()) {
                        let better = best.as_ref().and_then(|row| row.value).is_none_or(|value| {
                            if matches!(self.contract.objective, ProgramObjective::Variance(_)) {
                                row.value.unwrap_or(f64::INFINITY)
                                    < value - self.options.objective_tolerance
                            } else {
                                row.value.unwrap_or(f64::NEG_INFINITY)
                                    > value + self.options.objective_tolerance
                            }
                        });
                        if better {
                            best = Some(row.clone());
                            incumbent = proposal;
                            improved = true;
                        }
                    }
                    attempts.push(attempt);
                }
            }
            if !improved {
                for step in &mut steps {
                    *step *= 0.5;
                }
                if steps.iter().all(|step| *step <= self.options.step_tolerance) {
                    stop = ContinuousSearchStop::StepTolerance;
                    break;
                }
            }
        }
        Ok(ContinuousProgramSearchResult {
            best,
            attempts,
            stop,
            scope: self.scientific_scope,
            gap: None,
            global_certificate: false,
            reliability: "unmeasured",
            retention,
        })
    }
    fn evaluate(
        &self,
        controls: &[f64],
        evaluator: &impl CompiledProgramEvaluator,
        ctx: &ExecutionContext,
        dependency: &mut Option<SemanticDigest>,
    ) -> Result<ContinuousSearchAttempt, ProgramSearchError> {
        cancelled(ctx)?;
        let candidate =
            ProgramCandidate { id: "proposal".into(), controls: controls.to_vec(), inputs: vec![] };
        let operational = program_row(&candidate, &self.domain)?;
        if operational.feasibility != ProgramFeasibility::Feasible {
            return Ok(ContinuousSearchAttempt {
                controls: controls.to_vec(),
                row: Some(operational),
                failure: None,
                target_identity: None,
                dependency_identity: None,
            });
        }
        let compiled = match evaluator.evaluate_program(controls, ctx) {
            Ok(value) => value,
            Err(ProgramSearchError::Invalid(error)) => {
                return Ok(ContinuousSearchAttempt {
                    controls: controls.to_vec(),
                    row: None,
                    failure: Some(bounded_failure(error)),
                    target_identity: None,
                    dependency_identity: None,
                });
            }
            Err(error) => return Err(error),
        };
        if compiled.candidate.controls != controls {
            return Err(invalid("compiler returned a different continuous control vector"));
        }
        let current = compiled.source.dependency_identity();
        if dependency.is_some_and(|previous| previous != current) {
            return Err(invalid("continuous candidate changed the frozen fit dependency"));
        }
        *dependency = Some(current);
        let plan = PreparedFiniteProgramSearch::compile(
            self.contract.clone(),
            self.domain.clone(),
            vec![compiled.candidate],
            self.scope.clone(),
            ctx,
        )?;
        let result = compiled.source.search(&plan, self.scientific_scope, ctx).map_err(
            |error| match error {
                crate::scientific_source::ScientificSourceError::Search(error) => error,
                error => invalid(error),
            },
        )?;
        let row = result
            .result
            .rows
            .into_iter()
            .next()
            .ok_or_else(|| invalid("decision engine returned no proposal row"))?;
        Ok(ContinuousSearchAttempt {
            controls: controls.to_vec(),
            row: Some(row),
            failure: None,
            target_identity: Some(result.target_identity),
            dependency_identity: Some(current),
        })
    }
}

fn bounded_failure(mut message: String) -> String {
    const PAYLOAD_BYTES: usize = 1000;
    if message.len() > PAYLOAD_BYTES {
        let mut boundary = PAYLOAD_BYTES;
        while !message.is_char_boundary(boundary) {
            boundary -= 1;
        }
        message.truncate(boundary);
        message.push_str(" [truncated]");
    }
    message
}
