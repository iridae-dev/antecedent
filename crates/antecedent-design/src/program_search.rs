//! Exhaustive bounded program search over checked response quantities.
//!
//! The causal compiler supplies the response law. This owner checks operational feasibility
//! and delegates every statistical functional to the existing decision engine. Exhaustion
//! certifies the supplied finite set under its retained law, never a continuous or statistical
//! optimum. Failed or unsupported source coordinates refuse a complete certificate.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::decision_contract::{
    ActionKind, DecisionAction, DecisionContract, DecisionCriterion, DecisionFunctional,
    ProgramObjective, ProgramSearchContract, SourceRepresentation, StructuralPolicy,
};
use crate::decision_eval::{
    ActionLaw, DecisionResult, MeanSource, evaluate_contract_for_program_search,
    evaluate_contract_on_means_for_program_search,
};
use antecedent_contract::ScientificQuantity;
use antecedent_core::execution_attempt::{Operation, OperationGuard};
use antecedent_core::{CanonicalWriter, ExecutionContext, VariableId};
use antecedent_io_core::distribution_artifact::{DistributionArtifact, DistributionCalibration};
use std::collections::HashSet;
use std::sync::Arc;

/// One controllable coordinate, including its schedule offset and physical cost.
#[derive(Clone, Debug, PartialEq)]
pub struct ProgramControl {
    /// Causal variable the compiled program changes.
    pub variable: VariableId,
    /// Offset relative to the program origin; distinct offsets remain distinct controls.
    pub offset: i32,
    /// Inclusive lower bound.
    pub lower: f64,
    /// Inclusive upper bound.
    pub upper: f64,
    /// Nonnegative linear cost per unit of this nonnegative control.
    pub unit_cost: f64,
}
/// Deterministic operational restriction on control values: dot(coefficients, controls) <= bound.
#[derive(Clone, Debug, PartialEq)]
pub struct ProgramRestriction {
    /// Stable restriction identity.
    pub id: String,
    /// One coefficient per control, including each scheduled offset.
    pub coefficients: Vec<f64>,
    /// Inclusive upper bound.
    pub bound: f64,
}
/// Explicit finite search space, not an optimizer over an implied continuous box.
#[derive(Clone, Debug, PartialEq)]
pub struct FiniteProgramDomain {
    /// Controllable variables and schedules.
    pub controls: Vec<ProgramControl>,
    /// Optional total cost bound across all scheduled coordinates.
    pub budget: Option<f64>,
    /// Additional hard operational restrictions.
    pub restrictions: Vec<ProgramRestriction>,
}
/// One candidate and its already compiled response coordinates.
#[derive(Clone, Debug, PartialEq)]
pub struct ProgramCandidate {
    /// Stable program identity, also used by the response compiler.
    pub id: String,
    /// One value per scheduled control coordinate.
    pub controls: Vec<f64>,
    /// Response quantities in the search objective's input order.
    pub inputs: Vec<ScientificQuantity>,
}
/// Decision population, horizon, utility units and caller enumeration limit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProgramSearchScope {
    /// Exact response population identity.
    pub population: String,
    /// Exact response horizon, zero for a static target.
    pub horizon: u32,
    /// Common units of compared utility values.
    pub utility_units: String,
    /// Maximum candidates this consumer accepts; at most the shared decision limit.
    pub max_candidates: usize,
}
/// Input the search consumes, selected against the existing source-demand contract.
#[derive(Clone, Copy)]
pub enum ProgramSearchSource<'a> {
    /// Means permit affine expected utility without simulating a law.
    Means(&'a MeanSource),
    /// One aligned retained law permits its declared functionals and risk constraints.
    Law(&'a DistributionArtifact),
}
/// Typed search refusal; the numerical law cannot turn an invalid search into a result.
#[derive(Debug, thiserror::Error)]
pub enum ProgramSearchError {
    /// Invalid space, candidate, contract, source or statistical functional.
    #[error("program_search: {0}")]
    Invalid(String),
    /// Caller resources refused before output allocation.
    #[error("program_search resource refusal: {0}")]
    Resource(String),
    /// Caller cancellation; no partial result is certified.
    #[error("program search cancelled")]
    Cancelled,
}
fn invalid(error: impl std::fmt::Debug) -> ProgramSearchError {
    ProgramSearchError::Invalid(format!("{error:?}"))
}
fn cancelled(ctx: &ExecutionContext) -> Result<(), ProgramSearchError> {
    if ctx.cancellation.is_cancelled() { Err(ProgramSearchError::Cancelled) } else { Ok(()) }
}
/// Operational or statistical feasibility of one finite program.
#[derive(Clone, Debug, PartialEq)]
pub enum ProgramFeasibility {
    /// Every deterministic and law-based constraint passed.
    Feasible,
    /// This operational bound failed before a statistical ranking was attempted.
    Operational(String),
    /// One or more hard risk constraints excluded this candidate.
    Risk(Vec<String>),
}
/// One evaluated candidate; an infeasible candidate has no objective value.
#[derive(Clone, Debug, PartialEq)]
pub struct ProgramSearchRow {
    /// Candidate identity.
    pub id: String,
    /// Controls actually checked.
    pub controls: Vec<f64>,
    /// Actual linear scheduled cost.
    pub cost: f64,
    /// Feasibility with a concrete exclusion reason.
    pub feasibility: ProgramFeasibility,
    /// Objective functional under the retained source law; not an optimality probability.
    pub value: Option<f64>,
}
/// Complete finite-set result. A sampled-law ranking carries no selected-decision license.
#[derive(Clone, Debug)]
pub struct FiniteProgramSearchResult {
    /// Every declared candidate, including excluded ones.
    pub rows: Vec<ProgramSearchRow>,
    /// All equally best feasible programs; empty when none is feasible.
    pub best: Vec<String>,
    /// Identity of the actual finite set and scheduled controls.
    pub candidate_set_identity: [u8; 32],
    /// Exact objective and risk contract used for this search.
    pub search_identity: [u8; 32],
    /// Exact supplied response source identity, including its law or means.
    pub source_identity: String,
    /// Finite-set exhaustion under the supplied source; never a continuous optimum.
    pub exhaustive: bool,
    /// Whether source integration was exact under its declared finite law.
    pub exact_finite_law: bool,
    /// Whole selected-decision reliability has not been measured by this computation.
    pub reliability: &'static str,
    /// Bounds on the continuous-search gap are unavailable from finite exhaustion.
    pub continuous_gap: Option<f64>,
    retention: Arc<antecedent_core::execution::AdmissionLease>,
}
impl FiniteProgramSearchResult {
    /// Retained admission, not process RSS or an allocator measurement.
    #[must_use]
    pub fn admitted_bytes(&self) -> u64 {
        self.retention.observe_current();
        self.retention.remainder().bytes()
    }
}
/// Frozen search over actual compiled candidates; changing only the objective refits no response.
#[derive(Clone, Debug)]
pub struct PreparedFiniteProgramSearch {
    contract: ProgramSearchContract,
    domain: FiniteProgramDomain,
    candidates: Arc<[ProgramCandidate]>,
    population: String,
    horizon: u32,
    units: String,
    identity: [u8; 32],
}
impl PreparedFiniteProgramSearch {
    /// Compile a bounded candidate space and operational contract without evaluating a model.
    ///
    /// # Errors
    /// Invalid contracts, bounds, schedules, costs, duplicate candidates or consumer size limits.
    pub fn compile(
        contract: ProgramSearchContract,
        domain: FiniteProgramDomain,
        candidates: Vec<ProgramCandidate>,
        scope: ProgramSearchScope,
        ctx: &ExecutionContext,
    ) -> Result<Self, ProgramSearchError> {
        cancelled(ctx)?;
        validate_programs(&contract, &domain, &candidates, &scope)?;
        let identity = candidate_set_identity(&contract, &domain, &candidates);
        Ok(Self {
            contract,
            domain,
            candidates: candidates.into(),
            population: scope.population,
            horizon: scope.horizon,
            units: scope.utility_units,
            identity,
        })
    }
    /// Preserve response candidates and their identity while changing downstream utility/risk only.
    ///
    /// # Errors
    /// A changed response-input space or invalid replacement contract.
    pub fn with_objective(
        &self,
        contract: ProgramSearchContract,
    ) -> Result<Self, ProgramSearchError> {
        contract.validate().map_err(invalid)?;
        if contract.space != self.contract.space {
            return Err(invalid("objective update changed the response input space"));
        }
        let mut next = self.clone();
        next.contract = contract;
        Ok(next)
    }
    /// Exhaustively evaluate all feasible declared candidates through the original decision engine.
    ///
    /// # Errors
    /// Missing/unsupported quantities, insufficient law, resource refusal, invalid functional
    /// or cancellation. Partial enumeration never produces a complete certificate.
    pub fn execute(
        &self,
        source: ProgramSearchSource<'_>,
        ctx: &ExecutionContext,
    ) -> Result<FiniteProgramSearchResult, ProgramSearchError> {
        cancelled(ctx)?;
        let mut results = ctx
            .try_map_indexed(1, |_, worker| self.execute_on_worker(source, worker))
            .map_err(|error| match error {
                antecedent_core::execution::MapExecutionError::Work(error) => error,
                antecedent_core::execution::MapExecutionError::Admission(error) => {
                    if ctx.cancellation.is_cancelled() {
                        ProgramSearchError::Cancelled
                    } else {
                        ProgramSearchError::Resource(error.to_string())
                    }
                }
            })?;
        results.pop().ok_or_else(|| invalid("admitted search produced no result"))
    }
    fn execute_on_worker(
        &self,
        source: ProgramSearchSource<'_>,
        ctx: &ExecutionContext,
    ) -> Result<FiniteProgramSearchResult, ProgramSearchError> {
        cancelled(ctx)?;
        let (retained_bytes, scratch_bytes) = self.memory_quote(&source)?;
        let bytes = retained_bytes
            .checked_add(scratch_bytes)
            .ok_or_else(|| ProgramSearchError::Resource("quote overflow".into()))?;
        if ctx.memory_limit_bytes().is_some_and(|limit| bytes > limit) {
            return Err(ProgramSearchError::Resource(
                "search exceeds configured memory limit".into(),
            ));
        }
        let retention = Arc::new(
            ctx.acquire_admission(0, retained_bytes)
                .map_err(|error| ProgramSearchError::Resource(error.to_string()))?,
        );
        let _scratch = ctx
            .acquire_admission(0, scratch_bytes)
            .map_err(|error| ProgramSearchError::Resource(error.to_string()))?;
        let search_identity = self.contract.identity().map_err(invalid)?;
        let source_identity = source_identity(&source);
        let operation = OperationGuard::begin(Operation::Decision);
        let mut rows = Vec::with_capacity(self.candidates.len());
        let mut feasible = Vec::new();
        for candidate in self.candidates.iter() {
            cancelled(ctx)?;
            rows.push(program_row(candidate, &self.domain)?);
            if matches!(rows.last().map(|r| &r.feasibility), Some(ProgramFeasibility::Feasible)) {
                feasible.push(candidate.clone());
            }
        }
        let exact = matches!(&source, ProgramSearchSource::Law(law) if law.metadata().calibration==DistributionCalibration::Exact);
        self.evaluate_rows(&source, &feasible, &mut rows, ctx)?;
        cancelled(ctx)?;
        let minimize = matches!(
            self.contract.objective,
            ProgramObjective::Variance(_) | ProgramObjective::Regret(_)
        );
        let best_value = rows
            .iter()
            .filter_map(|row| row.value)
            .reduce(|a, b| if (minimize && b < a) || (!minimize && b > a) { b } else { a });
        let best = rows
            .iter()
            .filter(|row| row.value.is_some() && row.value == best_value)
            .map(|row| row.id.clone())
            .collect();
        operation.complete();
        Ok(FiniteProgramSearchResult {
            rows,
            best,
            candidate_set_identity: self.identity,
            search_identity,
            source_identity,
            exhaustive: true,
            exact_finite_law: exact,
            reliability: "unmeasured",
            continuous_gap: None,
            retention,
        })
    }
    fn memory_quote(
        &self,
        source: &ProgramSearchSource<'_>,
    ) -> Result<(u64, u64), ProgramSearchError> {
        let draws = match source {
            ProgramSearchSource::Means(_) => 0,
            ProgramSearchSource::Law(law) => law.n_draws(),
        };
        let overflow = || ProgramSearchError::Resource("quote overflow".into());
        let mut retained = 512_usize;
        let mut scratch = draws
            .checked_mul(self.candidates.len())
            .and_then(|value| value.checked_mul(48))
            .ok_or_else(overflow)?;
        let constraint_bytes = self
            .contract
            .constraints
            .iter()
            .try_fold(0_usize, |sum, constraint| {
                sum.checked_add(constraint.id.len() + std::mem::size_of::<String>())
            })
            .ok_or_else(overflow)?;
        for candidate in self.candidates.iter() {
            let row = std::mem::size_of::<ProgramSearchRow>()
                .checked_add(candidate.controls.len() * 8)
                .and_then(|value| value.checked_add(candidate.id.len().checked_mul(3)?))
                .and_then(|value| value.checked_add(constraint_bytes))
                .ok_or_else(overflow)?;
            retained = retained.checked_add(row).ok_or_else(overflow)?;
            scratch = scratch.checked_add(row).ok_or_else(overflow)?;
            for quantity in &candidate.inputs {
                scratch = scratch.checked_add(quantity_bytes(quantity)?).ok_or_else(overflow)?;
            }
        }
        let coordinates = match source {
            ProgramSearchSource::Means(source) => source.coordinates.len(),
            ProgramSearchSource::Law(source) => source.quantities().len(),
        };
        scratch = scratch
            .checked_add(
                coordinates
                    .checked_mul(std::mem::size_of::<ScientificQuantity>())
                    .ok_or_else(overflow)?,
            )
            .ok_or_else(overflow)?;
        Ok((
            u64::try_from(retained).map_err(|_| overflow())?,
            u64::try_from(scratch).map_err(|_| overflow())?,
        ))
    }
    fn evaluate_rows(
        &self,
        source: &ProgramSearchSource<'_>,
        feasible: &[ProgramCandidate],
        rows: &mut [ProgramSearchRow],
        ctx: &ExecutionContext,
    ) -> Result<(), ProgramSearchError> {
        if feasible.is_empty() {
            return Ok(());
        }
        let available = match source {
            ProgramSearchSource::Means(_) => SourceRepresentation::Mean,
            ProgramSearchSource::Law(law) => match law.metadata().identity.alignment {
                antecedent_io_core::distribution_artifact::DrawAlignment::Joint => {
                    SourceRepresentation::JointDraws
                }
                antecedent_io_core::distribution_artifact::DrawAlignment::IndependentMarginals => {
                    SourceRepresentation::MarginalDraws
                }
            },
        };
        self.contract
            .source_requirement()
            .map_err(invalid)?
            .check(&[available])
            .map_err(invalid)?;
        let decision =
            decision(&self.contract, feasible, &self.population, self.horizon, &self.units)?;
        let outcome: DecisionResult = match source {
            ProgramSearchSource::Means(means) => {
                evaluate_contract_on_means_for_program_search(&decision, means)
            }
            ProgramSearchSource::Law(law) => evaluate_contract_for_program_search(&decision, law),
        }
        .map_err(invalid)?;
        for row in rows {
            cancelled(ctx)?;
            if let Some(action) = outcome.actions.iter().find(|action| action.id == row.id) {
                if !action.admissible {
                    row.feasibility = ProgramFeasibility::Risk(
                        action.exclusions.iter().map(|x| x.constraint_id.clone()).collect(),
                    );
                    continue;
                }
                row.value = Some(objective_value(
                    &self.contract.objective,
                    source,
                    &decision,
                    &row.id,
                    action,
                )?);
            }
        }
        Ok(())
    }
}
fn quantity_bytes(quantity: &ScientificQuantity) -> Result<usize, ProgramSearchError> {
    let fields = [
        &quantity.variable_id,
        &quantity.variable_name,
        &quantity.units,
        &quantity.population_id,
        &quantity.regime_id,
        &quantity.functional_id,
        &quantity.transform_id,
    ];
    let mut bytes = std::mem::size_of::<ScientificQuantity>();
    for field in fields {
        bytes = bytes
            .checked_add(field.len())
            .ok_or_else(|| ProgramSearchError::Resource("quantity width overflow".into()))?;
    }
    for condition in &quantity.conditioning {
        bytes = bytes
            .checked_add(std::mem::size_of_val(condition))
            .and_then(|value| value.checked_add(condition.variable_id.len()))
            .and_then(|value| value.checked_add(condition.value_id.len()))
            .ok_or_else(|| ProgramSearchError::Resource("condition width overflow".into()))?;
    }
    Ok(bytes)
}

fn validate_programs(
    contract: &ProgramSearchContract,
    domain: &FiniteProgramDomain,
    candidates: &[ProgramCandidate],
    scope: &ProgramSearchScope,
) -> Result<(), ProgramSearchError> {
    contract.validate().map_err(invalid)?;
    validate_domain(domain, scope)?;
    if candidates.is_empty() || candidates.len() > scope.max_candidates {
        return Err(invalid("invalid finite candidate count"));
    }
    let mut ids = HashSet::new();
    for candidate in candidates {
        if candidate.id.trim().is_empty()
            || !ids.insert(&candidate.id)
            || candidate.controls.len() != domain.controls.len()
            || candidate.controls.iter().any(|x| !x.is_finite())
            || candidate.inputs.len() != contract.space.variables.len()
        {
            return Err(invalid("invalid program or response binding"));
        }
        // Reuse the decision contract's exact quantity/population/horizon validation.
    }
    decision(contract, candidates, &scope.population, scope.horizon, &scope.utility_units)?
        .validate_for_program_search()
        .map_err(invalid)?;
    Ok(())
}
fn validate_domain(
    domain: &FiniteProgramDomain,
    scope: &ProgramSearchScope,
) -> Result<(), ProgramSearchError> {
    if scope.max_candidates == 0
        || scope.max_candidates > 1024
        || domain.controls.is_empty()
        || domain.controls.len() > 4096
        || domain.budget.is_some_and(|budget| !budget.is_finite() || budget < 0.)
    {
        return Err(invalid("invalid finite domain or consumer limit"));
    }
    let mut coords = HashSet::new();
    for control in &domain.controls {
        if !coords.insert((control.variable, control.offset))
            || !control.lower.is_finite()
            || !control.upper.is_finite()
            || control.lower < 0.
            || control.upper < control.lower
            || !control.unit_cost.is_finite()
            || control.unit_cost < 0.
        {
            return Err(invalid("invalid or duplicate scheduled control"));
        }
    }
    let mut restrictions = HashSet::new();
    for restriction in &domain.restrictions {
        if restriction.id.trim().is_empty()
            || !restrictions.insert(&restriction.id)
            || restriction.coefficients.len() != domain.controls.len()
            || !restriction.bound.is_finite()
            || restriction.coefficients.iter().any(|x| !x.is_finite())
        {
            return Err(invalid("invalid operational restriction"));
        }
    }
    Ok(())
}
fn candidate_set_identity(
    contract: &ProgramSearchContract,
    domain: &FiniteProgramDomain,
    candidates: &[ProgramCandidate],
) -> [u8; 32] {
    let mut canonical: Vec<_> = candidates.iter().collect();
    canonical.sort_by(|a, b| a.id.cmp(&b.id));
    let mut writer = CanonicalWriter::new("antecedent.finite_program_set.v1");
    writer.text(&contract.space.id).word(domain.controls.len() as u64);
    for control in &domain.controls {
        writer
            .word(u64::from(control.variable.raw()))
            .bytes(&control.offset.to_le_bytes())
            .real(control.lower)
            .real(control.upper)
            .real(control.unit_cost);
    }
    writer.word(u64::from(domain.budget.is_some()));
    if let Some(budget) = domain.budget {
        writer.real(budget);
    }
    writer.word(domain.restrictions.len() as u64);
    for restriction in &domain.restrictions {
        writer
            .text(&restriction.id)
            .real(restriction.bound)
            .word(restriction.coefficients.len() as u64);
        for coefficient in &restriction.coefficients {
            writer.real(*coefficient);
        }
    }
    writer.word(canonical.len() as u64);
    for candidate in canonical {
        writer.text(&candidate.id);
        for value in &candidate.controls {
            writer.real(*value);
        }
        for input in &candidate.inputs {
            input.write_canonical(&mut writer);
        }
    }
    writer.finish()
}
fn source_identity(source: &ProgramSearchSource<'_>) -> String {
    match source {
        ProgramSearchSource::Law(law) => crate::decision_artifact::source_digest(law),
        ProgramSearchSource::Means(means) => {
            let mut writer = CanonicalWriter::new("antecedent.program_search_means.v1");
            writer
                .text(&means.provider_id)
                .text(&means.snapshot_id)
                .text(&means.causal_contract_id)
                .text(&means.rng_id)
                .word(means.coordinates.len() as u64);
            for coordinate in &means.coordinates {
                coordinate.write_canonical(&mut writer);
            }
            writer.reals(&means.means);
            antecedent_core::to_hex(&writer.finish())
        }
    }
}
fn program_row(
    candidate: &ProgramCandidate,
    domain: &FiniteProgramDomain,
) -> Result<ProgramSearchRow, ProgramSearchError> {
    let cost = candidate
        .controls
        .iter()
        .zip(&domain.controls)
        .map(|(value, control)| value * control.unit_cost)
        .sum::<f64>();
    if !cost.is_finite() {
        return Err(invalid("non-finite total program cost"));
    }
    let mut reason = if candidate
        .controls
        .iter()
        .zip(&domain.controls)
        .any(|(value, control)| *value < control.lower || *value > control.upper)
    {
        Some("control bounds".to_owned())
    } else if domain.budget.is_some_and(|budget| cost > budget) {
        Some("budget".to_owned())
    } else {
        None
    };
    for restriction in &domain.restrictions {
        let total = candidate
            .controls
            .iter()
            .zip(&restriction.coefficients)
            .map(|(value, coefficient)| value * coefficient)
            .sum::<f64>();
        if !total.is_finite() {
            return Err(invalid("non-finite operational restriction"));
        }
        if reason.is_none() && total > restriction.bound {
            reason = Some(restriction.id.clone());
        }
    }
    Ok(ProgramSearchRow {
        id: candidate.id.clone(),
        controls: candidate.controls.clone(),
        cost,
        feasibility: reason.map_or(ProgramFeasibility::Feasible, ProgramFeasibility::Operational),
        value: None,
    })
}
fn objective_value(
    objective: &ProgramObjective,
    source: &ProgramSearchSource<'_>,
    decision: &DecisionContract,
    action_id: &str,
    action: &crate::decision_eval::ActionOutcome,
) -> Result<f64, ProgramSearchError> {
    Ok(match (objective, source) {
        (ProgramObjective::Regret(_), _) => {
            action.max_regret.ok_or_else(|| invalid("regret unavailable"))?
        }
        (ProgramObjective::Variance(_), ProgramSearchSource::Law(law)) => {
            ActionLaw::of_program_search_action(decision, action_id, law)
                .and_then(|law| law.functional(decision, DecisionFunctional::Variance))
                .map_err(invalid)?
                .value
        }
        (ProgramObjective::Threshold { threshold, tail, .. }, ProgramSearchSource::Law(law)) => {
            ActionLaw::of_program_search_action(decision, action_id, law)
                .and_then(|law| {
                    law.functional(
                        decision,
                        DecisionFunctional::Probability { threshold: *threshold, tail: *tail },
                    )
                })
                .map_err(invalid)?
                .value
        }
        (ProgramObjective::Quantile { p, .. }, ProgramSearchSource::Law(law)) => {
            ActionLaw::of_program_search_action(decision, action_id, law)
                .and_then(|law| law.functional(decision, DecisionFunctional::Quantile { p: *p }))
                .map_err(invalid)?
                .value
        }
        (
            ProgramObjective::Variance(_)
            | ProgramObjective::Threshold { .. }
            | ProgramObjective::Quantile { .. },
            _,
        ) => return Err(invalid("objective requires an uncertainty source")),
        _ => action.value,
    })
}

fn decision(
    search: &ProgramSearchContract,
    candidates: &[ProgramCandidate],
    population: &str,
    horizon: u32,
    units: &str,
) -> Result<DecisionContract, ProgramSearchError> {
    let (expr, criterion) = match &search.objective {
        ProgramObjective::ExpectedUtility(expr)
        | ProgramObjective::Variance(expr)
        | ProgramObjective::Threshold { expr, .. }
        | ProgramObjective::Quantile { expr, .. }
        | ProgramObjective::Optimality(expr) => (expr, DecisionCriterion::PosteriorExpectedUtility),
        ProgramObjective::Regret(expr) => (expr, DecisionCriterion::Regret),
        ProgramObjective::Robust => {
            return Err(invalid("robust search requires explicit identified-set/structure source"));
        }
    };
    Ok(DecisionContract {
        actions: candidates
            .iter()
            .map(|candidate| DecisionAction {
                id: candidate.id.clone(),
                kind: ActionKind::Regime,
                inputs: candidate.inputs.clone(),
                utility: expr.clone(),
            })
            .collect(),
        utility_units: units.into(),
        criterion,
        constraints: search.constraints.clone(),
        target_population: population.into(),
        horizon,
        structural_policy: StructuralPolicy::ReportOnly,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decision_contract::{HardConstraint, ProgramSearchSpace, UtilityExpr};
    use antecedent_contract::QuantityRole;
    fn quantity(id: &str) -> ScientificQuantity {
        ScientificQuantity {
            variable_id: "y".into(),
            variable_name: "y".into(),
            role: QuantityRole::Outcome,
            units: "USD".into(),
            population_id: "observed".into(),
            regime_id: id.into(),
            horizon: 2,
            functional_id: "mean".into(),
            conditioning: vec![],
            transform_id: "identity".into(),
        }
    }
    fn fixture() -> (PreparedFiniteProgramSearch, MeanSource) {
        let contract = ProgramSearchContract {
            space: ProgramSearchSpace {
                id: "two-channel schedules".into(),
                variables: vec!["response".into()],
            },
            objective: ProgramObjective::ExpectedUtility(UtilityExpr::Input(0)),
            constraints: vec![],
        };
        let domain = FiniteProgramDomain {
            controls: vec![
                ProgramControl {
                    variable: VariableId::from_raw(0),
                    offset: 0,
                    lower: 0.,
                    upper: 4.,
                    unit_cost: 1.,
                },
                ProgramControl {
                    variable: VariableId::from_raw(1),
                    offset: 1,
                    lower: 0.,
                    upper: 4.,
                    unit_cost: 1.,
                },
            ],
            budget: Some(5.),
            restrictions: vec![],
        };
        let candidates = vec![
            ProgramCandidate {
                id: "reference".into(),
                controls: vec![0., 0.],
                inputs: vec![quantity("reference")],
            },
            ProgramCandidate {
                id: "a".into(),
                controls: vec![4., 0.],
                inputs: vec![quantity("a")],
            },
            ProgramCandidate {
                id: "joint".into(),
                controls: vec![4., 4.],
                inputs: vec![quantity("joint")],
            },
        ];
        let source = MeanSource {
            coordinates: candidates.iter().map(|c| c.inputs[0].clone()).collect(),
            means: vec![0., 3., 8.],
            estimate: crate::decision_eval::MeanEstimate::Frequentist,
            provider_id: "compiled-model".into(),
            snapshot_id: "source".into(),
            causal_contract_id: "target".into(),
            rng_id: "no-sampling".into(),
        };
        (
            PreparedFiniteProgramSearch::compile(
                contract,
                domain,
                candidates,
                ProgramSearchScope {
                    population: "observed".into(),
                    horizon: 2,
                    utility_units: "USD".into(),
                    max_candidates: 32,
                },
                &ExecutionContext::for_tests(19),
            )
            .unwrap(),
            source,
        )
    }
    #[test]
    fn finite_program_search_checks_budget_and_reuses_outputs_for_utility_changes() {
        let (search, source) = fixture();
        let ctx = ExecutionContext::for_tests(19);
        let first = search.execute(ProgramSearchSource::Means(&source), &ctx).unwrap();
        assert_eq!(first.best, vec!["a"]);
        assert!(first.exhaustive);
        assert!(!first.exact_finite_law);
        assert_eq!(first.continuous_gap, None);
        assert_eq!(first.reliability, "unmeasured");
        assert!(
            matches!(&first.rows[2].feasibility,ProgramFeasibility::Operational(reason) if reason=="budget")
        );
        let mut replacement = search.contract.clone();
        replacement.objective =
            ProgramObjective::ExpectedUtility(UtilityExpr::Neg(Box::new(UtilityExpr::Input(0))));
        let second = search
            .with_objective(replacement)
            .unwrap()
            .execute(ProgramSearchSource::Means(&source), &ctx)
            .unwrap();
        assert_eq!(second.best, vec!["reference"]);
        assert_eq!(second.candidate_set_identity, first.candidate_set_identity);
        assert_ne!(second.search_identity, first.search_identity);
        assert_eq!(second.source_identity, first.source_identity);
    }
    #[test]
    fn finite_program_search_refuses_means_for_risk_or_optimality_law() {
        let (search, source) = fixture();
        let mut contract = search.contract.clone();
        contract.objective = ProgramObjective::Optimality(UtilityExpr::Input(0));
        assert!(
            search
                .with_objective(contract)
                .unwrap()
                .execute(ProgramSearchSource::Means(&source), &ExecutionContext::for_tests(19))
                .is_err()
        );
        let mut contract = search.contract.clone();
        contract.constraints.push(HardConstraint {
            id: "risk".into(),
            expr: UtilityExpr::Input(0),
            bound: 4.,
            min_probability: 0.95,
            units: "USD".into(),
            applies_to: vec![],
        });
        assert!(
            search
                .with_objective(contract)
                .unwrap()
                .execute(ProgramSearchSource::Means(&source), &ExecutionContext::for_tests(19))
                .is_err()
        );
    }
    #[test]
    fn finite_program_search_retains_admission_and_refuses_cancelled_work() {
        let (search, source) = fixture();
        let mut ctx = ExecutionContext::for_tests(19);
        ctx.admission = antecedent_core::execution::AdmissionPool::fixed(1, 1 << 20);
        let before = ctx.admission.available().bytes();
        let result = search.execute(ProgramSearchSource::Means(&source), &ctx).unwrap();
        assert!(ctx.admission.available().bytes() < before);
        drop(result);
        assert_eq!(ctx.admission.available().bytes(), before);
        ctx.cancellation.cancel();
        assert!(matches!(
            search.execute(ProgramSearchSource::Means(&source), &ctx),
            Err(ProgramSearchError::Cancelled)
        ));
    }
    fn law(source: &MeanSource) -> DistributionArtifact {
        use antecedent_io_core::distribution_artifact::{
            DistributionIdentity, DistributionMetadata, DistributionProvenance, DistributionTrust,
            DrawAlignment,
        };
        let identity = DistributionIdentity::new(
            antecedent_io_core::quantity_wire::DistributionMeaningWire::CausalFunctionalPosterior,
            &source.coordinates,
            DrawAlignment::Joint,
            DistributionProvenance {
                source_id: "finite-joint-law".into(),
                provider_id: source.provider_id.clone(),
                rng_id: source.rng_id.clone(),
                snapshot_id: source.snapshot_id.clone(),
                causal_contract_id: source.causal_contract_id.clone(),
            },
        )
        .unwrap();
        DistributionArtifact::new(
            DistributionMetadata {
                version: 1,
                identity,
                axes: ["draw".into(), "quantity".into()],
                shape: [2, 3],
                weights: None,
                supported: None,
                calibration: DistributionCalibration::Exact,
                trust: DistributionTrust::Unverified,
                legacy_posterior: None,
                legacy_bindings: None,
            },
            vec![0., 3., 8., 0., 1., 8.],
        )
        .unwrap()
    }
    #[test]
    fn finite_program_search_uses_joint_law_for_risk_and_variance_and_preserves_exclusions() {
        let (search, source) = fixture();
        let law = law(&source);
        let ctx = ExecutionContext::for_tests(19);
        let mut contract = search.contract.clone();
        contract.constraints.push(HardConstraint {
            id: "response risk".into(),
            expr: UtilityExpr::Input(0),
            bound: 2.,
            min_probability: 0.75,
            units: "USD".into(),
            applies_to: vec![],
        });
        let result = search
            .with_objective(contract)
            .unwrap()
            .execute(ProgramSearchSource::Law(&law), &ctx)
            .unwrap();
        assert_eq!(result.best, vec!["reference"]);
        assert!(result.exact_finite_law);
        assert!(
            matches!(&result.rows[1].feasibility,ProgramFeasibility::Risk(ids) if ids==&vec!["response risk".to_owned()])
        );
        let mut contract = search.contract.clone();
        contract.objective = ProgramObjective::Variance(UtilityExpr::Input(0));
        let result = search
            .with_objective(contract)
            .unwrap()
            .execute(ProgramSearchSource::Law(&law), &ctx)
            .unwrap();
        assert_eq!(result.rows[1].value, Some(1.));
        assert_eq!(result.best, vec!["reference"]);
    }
    #[test]
    fn finite_program_search_refuses_unsupported_worlds_and_resource_failure_without_certificate() {
        let (search, source) = fixture();
        let mut metadata = law(&source).metadata().clone();
        metadata.supported = Some(vec![true, false, true]);
        let unsupported =
            DistributionArtifact::new(metadata, vec![0., 3., 8., 0., 1., 8.]).unwrap();
        assert!(
            search
                .execute(ProgramSearchSource::Law(&unsupported), &ExecutionContext::for_tests(19))
                .is_err()
        );
        let mut ctx = ExecutionContext::for_tests(19);
        ctx.admission = antecedent_core::execution::AdmissionPool::fixed(1, 1);
        assert!(matches!(
            search.execute(ProgramSearchSource::Means(&source), &ctx),
            Err(ProgramSearchError::Resource(_))
        ));
        assert_eq!(ctx.admission.available().bytes(), 1);
    }
    #[test]
    fn finite_program_search_handles_single_candidate_without_changing_decision_contract() {
        let (search, source) = fixture();
        let only = vec![search.candidates[0].clone()];
        let ctx = ExecutionContext::for_tests(19);
        let ordinary =
            decision(&search.contract, &only, &search.population, search.horizon, &search.units)
                .unwrap();
        assert!(ordinary.validate().is_err());
        let singleton = PreparedFiniteProgramSearch::compile(
            search.contract.clone(),
            search.domain.clone(),
            only,
            ProgramSearchScope {
                population: search.population.clone(),
                horizon: search.horizon,
                utility_units: search.units.clone(),
                max_candidates: 8,
            },
            &ctx,
        )
        .unwrap();
        let result = singleton.execute(ProgramSearchSource::Means(&source), &ctx).unwrap();
        assert_eq!(result.best, vec!["reference"]);
        assert!(result.exhaustive);
    }
}

/// Deterministic interval certificates for bounded compiled scalar objectives.
pub mod certified;
/// Bounded best-found continuous search using compiled program evaluators.
pub mod continuous;
/// Joint resampling and observational holdout checks for selected program values.
pub mod selection;
