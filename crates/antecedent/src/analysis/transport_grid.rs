//! Retained finite transport grids. A local coverage failure is a point, not a deleted row.
use super::{
    ExactPreparedState, ExactStudyResult, PreparedStudy, StatisticalPreparedState,
    StatisticalStudyResult, StudyBuilder,
};
pub use antecedent_core::TransportGridFailure;
use antecedent_core::{ExecutionContext, IdentityDomain, NodeRef, VariableDomain, VariableId};
use antecedent_estimate::{EmpiricalTableOptions, StatisticalTransportInput};
use antecedent_expr::{
    Assignment, EvalError, ExactDistribution, ExactEvaluationLimits, ExactTransportData,
};
use antecedent_graph::SelectionDiagram;
use antecedent_identify::{BoundTransportFunctional, SidLimits};
use antecedent_io::transport_grid_wire::TransportGridFailureWire;
use antecedent_io::transport_grid_wire::{
    GridPointWire, GridWire, SampleSummary, StatisticalOptionsWire,
};
use antecedent_io::{
    IoError, exact_law_wire::ExactLawWire, query_wire::ValueWire,
    transport_catalog_wire::EvidenceCatalogWire, transport_proof::TransportProofWire,
};
use serde::Deserialize;
use std::sync::Arc;

use super::transport_common::{GraphFields, digest, err, rebind_snapshots, rebuild_checked_proof};
fn assignments(request: &Assignment) -> Vec<(u32, ValueWire)> {
    let mut values: Vec<_> =
        request.entries().iter().map(|(v, x)| (v.raw(), ValueWire::from_value(x))).collect();
    values.sort_by_key(|x| x.0);
    values
}
fn request(values: &[(u32, ValueWire)]) -> Assignment {
    Assignment::from_pairs(values.iter().map(|(v, x)| (VariableId::from_raw(*v), x.to_value())))
}
/// Provider modality; empirical tables remain distinct from supplied laws.
#[derive(Clone, Debug)]
pub enum TransportGridData {
    /// Fully specified population laws, without sampling uncertainty.
    Exact(ExactTransportData),
    /// Supplied laws and sampled datasets, with an explicit inference contract.
    Statistical(StatisticalTransportInput, EmpiricalTableOptions),
}
/// Immutable structural and numerical request for a finite response grid.
#[derive(Clone, Debug)]
pub struct TransportGridQuery {
    /// Shared graph and primary selection diagram; other sources live in the checked proof.
    pub diagram: SelectionDiagram,
    /// Checked population-tagged functional with catalog bindings.
    pub functional: BoundTransportFunctional,
    /// Requested order is retained, including unsupported coordinates.
    pub at: Vec<Assignment>,
    /// Global numerical limits, partitioned conservatively across points.
    pub limits: ExactEvaluationLimits,
}

fn local_failure(
    error: EvalError,
    functional: &BoundTransportFunctional,
) -> Result<TransportGridFailure, IoError> {
    let kind = match &error {
        EvalError::EmptySupport(_) => "missing_evidence",
        EvalError::ExactLaw(e)
            if matches!(e.kind, "missing_exact_provider" | "missing_law_axis") =>
        {
            "missing_evidence"
        }
        EvalError::ExactLaw(e) if matches!(e.kind, "sampling_zero" | "zero_conditioning_mass") => {
            "support_failure"
        }
        EvalError::ExactRatioSupport { .. } => "support_failure",
        _ => return Err(err(error)),
    };
    let mut failure = TransportGridFailure {
        kind: kind.into(),
        detail: error.to_string(),
        code: kind.into(),
        variables: vec![],
        expression: None,
        assignment: vec![],
        bindings: vec![],
        interventions: vec![],
    };
    match error {
        EvalError::EmptySupport(v) => {
            failure.code = "missing_discrete_provider_domain".into();
            failure.variables.push(v.raw());
            failure.bindings = functional
                .arena()
                .leaf_bindings(functional.root())
                .into_iter()
                .map(|b| (b.population.to_string(), b.regime.map(antecedent_core::RegimeId::raw)))
                .collect();
        }
        EvalError::ExactLaw(e) => {
            failure.code = e.kind.into();
            failure.variables = e.variables.iter().map(|v| v.raw()).collect();
            failure.assignment = e.conditioning.iter().map(|(v, x)| (v.raw(), x.clone())).collect();
            failure
                .bindings
                .push((e.population.to_string(), e.regime.map(antecedent_core::RegimeId::raw)));
            failure.interventions =
                e.interventions.iter().map(|a| (a.variable.raw(), a.value.clone())).collect();
        }
        EvalError::ExactRatioSupport { expression, assignment, bindings } => {
            failure.code = "zero_ratio_denominator".into();
            failure.expression = Some(expression.raw());
            failure.assignment = assignment.iter().map(|(v, x)| (v.raw(), x.clone())).collect();
            failure.bindings = bindings
                .iter()
                .map(|b| (b.population.to_string(), b.regime.map(antecedent_core::RegimeId::raw)))
                .collect();
        }
        _ => unreachable!(),
    }
    Ok(failure)
}
/// One retained coordinate and its checked result.
#[derive(Clone, Debug)]
pub enum TransportGridPoint {
    /// Supplied-law target distribution.
    Exact(PreparedStudy<ExactPreparedState>, Box<ExactStudyResult>),
    /// Empirical target distribution with joint pointwise uncertainty.
    Statistical(PreparedStudy<StatisticalPreparedState>, Box<StatisticalStudyResult>),
    /// This coordinate remains in the response but cannot be executed.
    Unavailable(TransportGridFailure),
}
impl TransportGridPoint {
    /// Complete target law when this coordinate is executable.
    #[must_use]
    pub fn distribution(&self) -> Option<&ExactDistribution> {
        match self {
            Self::Exact(_, r) => Some(r.distribution()),
            Self::Statistical(_, r) => Some(r.distribution()),
            Self::Unavailable(_) => None,
        }
    }
}
fn point_evidence(point: &TransportGridPoint) -> Result<String, IoError> {
    match point {
        TransportGridPoint::Exact(_, r) => {
            antecedent_io::transport_grid_wire::point_evidence_identity(
                &r.identities().execution,
                r.distribution(),
                None,
            )
        }
        TransportGridPoint::Statistical(_, r) => {
            antecedent_io::transport_grid_wire::point_evidence_identity(
                &r.identities().execution,
                r.distribution(),
                Some(r.estimate()),
            )
        }
        TransportGridPoint::Unavailable(failure) => {
            digest(IdentityDomain::Execution, &TransportGridFailureWire::from_failure(failure))
        }
    }
}
/// Retained common prepared-study state for a finite grid.
#[derive(Clone, Debug)]
pub struct TransportGridState {
    query: TransportGridQuery,
    input: Option<TransportGridData>,
    data: ExactTransportData,
    seed: u64,
    plans: Vec<Result<antecedent_expr::ExactEvaluationPlan, TransportGridFailure>>,
    template: GridWire,
}
/// A checked curve whose requested coordinates and local failures are durable.
#[derive(Clone, Debug)]
pub struct TransportGridResult {
    points: Vec<TransportGridPoint>,
    wire: GridWire,
    identity: String,
}
impl TransportGridResult {
    /// Points in the original requested order.
    #[must_use]
    pub fn points(&self) -> &[TransportGridPoint] {
        &self.points
    }
    /// All four durable reasoning slots; per-point inference is retained in each point.
    #[must_use]
    pub fn reasoning(&self) -> antecedent_core::ReasoningView {
        let unavailable =
            self.points.iter().filter(|p| matches!(p, TransportGridPoint::Unavailable(_))).count();
        grid_reasoning_view(self.points.len(), unavailable, self.wire.statistical)
    }
    /// Complete family, evidence, provider and inference identity.
    #[must_use]
    pub fn identity(&self) -> &str {
        &self.identity
    }
    /// Explicit paired mean contrast; both coordinates must be executable.
    /// # Errors
    /// Unavailable point, invalid index/outcome, or incompatible native authority.
    pub fn contrast(
        &self,
        left: usize,
        right: usize,
        outcome: VariableId,
    ) -> Result<super::StatisticalContrast, IoError> {
        match (self.points.get(left), self.points.get(right)) {
            (
                Some(TransportGridPoint::Statistical(_, a)),
                Some(TransportGridPoint::Statistical(_, b)),
            ) => a.contrast(b, outcome),
            (Some(TransportGridPoint::Exact(_, a)), Some(TransportGridPoint::Exact(_, b))) => {
                Ok(super::StatisticalContrast {
                    estimate: a
                        .distribution()
                        .mean_difference(b.distribution(), outcome)
                        .map_err(err)?,
                    interval: None,
                    coverage_target: None,
                    replicates_ok: 0,
                    replicates_failed: 0,
                    reason: Some("exact_supplied_law_no_sampling_uncertainty"),
                })
            }
            _ => Err(err("contrast requires two executable coordinates of this grid")),
        }
    }
    /// Project one mean with an explicit loss receipt; the scalar is not a complete claim.
    /// # Errors
    /// Unavailable point or nonnumeric outcome.
    pub fn scalar_projection(
        &self,
        point: usize,
        outcome: VariableId,
    ) -> Result<(f64, antecedent_core::HandoffReceipt), IoError> {
        let distribution = self
            .points
            .get(point)
            .and_then(TransportGridPoint::distribution)
            .ok_or_else(|| err("unavailable grid point"))?;
        let identity = antecedent_core::SemanticDigest::from_bytes(
            antecedent_io::external_estimate::parse_digest_hex(&self.identity)?,
        );
        Ok((
            distribution.mean(outcome).map_err(err)?,
            antecedent_core::HandoffReceipt::lossy_scalar(identity, "transport_grid_scalar"),
        ))
    }
    /// Export a sectioned artifact with explicit required transport features.
    /// # Errors
    /// Serialization or container validation failure.
    pub fn export(&self) -> Result<Vec<u8>, IoError> {
        let mut wire = self.wire.clone();
        if wire.version == 2 {
            wire.points = self
                .points
                .iter()
                .map(|point| match point {
                    TransportGridPoint::Exact(study, result) => {
                        study.export(result).map(GridPointWire::Exact)
                    }
                    TransportGridPoint::Statistical(study, result) => {
                        study.export(result).map(GridPointWire::Statistical)
                    }
                    TransportGridPoint::Unavailable(failure) => Ok(GridPointWire::Unavailable(
                        TransportGridFailureWire::from_failure(failure),
                    )),
                })
                .collect::<Result<_, _>>()?;
        }
        wire.export(&self.identity)
    }
}

/// The one author of a grid's reasoning slots: the in-memory view and the portable section
/// both derive from it, so the two can never disagree.
fn grid_reasoning_view(
    total: usize,
    unavailable: usize,
    statistical: bool,
) -> antecedent_core::ReasoningView {
    use antecedent_core::{
        AssumptionSlot, AssumptionSource, AssumptionStatus, IdentificationSlot,
        IdentificationStatus, ObligationKind, ObligationRecord, ObligationScope, ReasoningView,
        SlotAvailability, SupportSlot,
    };
    ReasoningView::new(
        SlotAvailability::Available(IdentificationSlot::identified_singleton(
            IdentificationStatus::NonparametricallyIdentified,
        )),
        SlotAvailability::Available(SupportSlot::new(
            "stage_contract",
            Some(Arc::from("transport.response_grid")),
            SlotAvailability::Available(Arc::from(format!(
                "{} executable; {unavailable} unavailable; population positivity assumed",
                total - unavailable
            ))),
        )),
        SlotAvailability::unavailable(if statistical {
            "pointwise_components_in_retained_points; calibration_not_bound"
        } else {
            "exact_supplied_law_no_sampling_uncertainty"
        }),
        SlotAvailability::Available(AssumptionSlot::new(vec![ObligationRecord::new(
            "transport.population_selection_graph",
            ObligationScope::Program,
            AssumptionSource::UserDeclared,
            ObligationKind::UserAssertion,
            AssumptionStatus::Declared,
            "The accepted graph and source-specific mechanism selections describe the declared populations.",
        )])),
    )
}
fn grid_reasoning(
    points: &[GridPointWire],
    statistical: bool,
) -> antecedent_io::contract_section::ReasoningSectionWire {
    let unavailable = points.iter().filter(|p| matches!(p, GridPointWire::Unavailable(_))).count();
    super::contract::reasoning_section(&grid_reasoning_view(points.len(), unavailable, statistical))
}
fn validate_grid(
    functional: &BoundTransportFunctional,
    at: &[Assignment],
    limits: ExactEvaluationLimits,
    ctx: &ExecutionContext,
) -> Result<(), IoError> {
    if at.is_empty() || at.len() > limits.operations || ctx.cancellation.is_cancelled() {
        return Err(err("invalid grid or computation budget/cancellation"));
    }
    let arena = functional.arena();
    for id in arena.distribution_leaves(functional.root()) {
        if let antecedent_expr::ExprNode::Distribution {
            variables,
            conditioned_on,
            intervention,
            ..
        } = arena.node(id)
        {
            let variables = arena
                .var_set(*variables)
                .iter()
                .chain(arena.var_set(*conditioned_on))
                .copied()
                .chain(arena.intervention_assignments(*intervention).iter().map(|a| a.variable));
            for variable in variables {
                let domains: Vec<_> = functional
                    .catalog()
                    .environments
                    .iter()
                    .flat_map(|e| e.variables.iter())
                    .filter(|c| c.variable == variable)
                    .collect();
                if domains.is_empty()
                    || domains.iter().any(|c| {
                        !matches!(
                            c.domain,
                            VariableDomain::Binary | VariableDomain::Categorical { .. }
                        )
                    })
                {
                    return Err(err(
                        "transport grid providers require declared finite discrete domains",
                    ));
                }
            }
        }
    }
    let treatments = &functional.derivation().query().treatments;
    for at in at {
        if at.entries().len() != treatments.len() || treatments.iter().any(|v| at.get(*v).is_none())
        {
            return Err(err("grid must bind precisely the certified treatments"));
        }
        for (v, x) in at.entries() {
            if !x.as_f64().is_some_and(f64::is_finite) {
                return Err(err("grid values must be finite"));
            }
            let domains: Vec<_> = functional
                .catalog()
                .environments
                .iter()
                .flat_map(|e| e.variables.iter())
                .filter(|c| c.variable == *v)
                .collect();
            if domains.is_empty()
                || domains.iter().any(|c| {
                    !matches!(c.domain, VariableDomain::Binary | VariableDomain::Categorical { .. })
                })
            {
                return Err(err(
                    "transport grids require declared finite discrete treatment domains",
                ));
            }
            for c in domains {
                let cardinality = match c.domain {
                    VariableDomain::Binary => 2,
                    VariableDomain::Categorical { cardinality } => cardinality,
                    _ => unreachable!(),
                };
                let x = x.as_f64().unwrap();
                if x.fract() != 0.0 || x < 0.0 || x >= f64::from(cardinality) {
                    return Err(err("grid value outside declared finite domain"));
                }
            }
        }
    }
    Ok(())
}
fn compile_grid(
    query: &TransportGridQuery,
    data: &ExactTransportData,
    ctx: &ExecutionContext,
) -> Result<Vec<Result<antecedent_expr::ExactEvaluationPlan, TransportGridFailure>>, IoError> {
    let limits = ExactEvaluationLimits {
        operations: query.limits.operations / query.at.len() / 8,
        depth: query.limits.depth,
    };
    query
        .at
        .iter()
        .map(|at| {
            match antecedent_estimate::prepare_exact_transport(
                &query.functional,
                data.clone(),
                at.clone(),
                limits,
                ctx,
            ) {
                Ok(plan) => Ok(Ok(plan)),
                Err(error) => local_failure(error, &query.functional).map(Err),
            }
        })
        .collect()
}

impl StudyBuilder {
    /// Prepare a finite mean-response grid without discarding unsupported assignments.
    /// # Errors
    /// Invalid schema, incompatible evidence, or resource/cancellation failure.
    pub fn transport_grid(
        query: TransportGridQuery,
        input: TransportGridData,
        ctx: &ExecutionContext,
    ) -> Result<PreparedStudy<TransportGridState>, IoError> {
        antecedent_identify::verify_classical_transport(
            &query.diagram,
            query.functional.derivation().query(),
            query.functional.derivation(),
            SidLimits { steps: query.limits.operations, depth: query.limits.depth },
            ctx,
        )
        .map_err(err)?;
        validate_grid(&query.functional, &query.at, query.limits, ctx)?;
        let mut input = input;
        if let TransportGridData::Statistical(input, _) = &mut input {
            super::statistical::canonicalize_input(input)?;
        }
        let (data, samples, options) = match &input {
            TransportGridData::Exact(data) => {
                if data
                    .laws()
                    .iter()
                    .any(|l| l.origin() != antecedent_expr::LawOrigin::SuppliedExact)
                {
                    return Err(err("empirical laws require statistical transport"));
                }
                (data.clone(), vec![], None)
            }
            TransportGridData::Statistical(input, options) => {
                antecedent_estimate::statistical_transport::check_statistical_resources(
                    input,
                    &query.functional,
                    options,
                    ctx,
                )
                .map_err(err)?;
                let summaries = input
                    .samples
                    .iter()
                    .map(SampleSummary::from_sample)
                    .collect::<Result<Vec<_>, _>>()?;
                (
                    antecedent_estimate::empirical_table::assemble_grid_point_laws(
                        input,
                        &query.functional,
                        options,
                        ctx,
                    )
                    .map_err(err)?,
                    summaries,
                    Some(StatisticalOptionsWire::from_options(options)),
                )
            }
        };
        let data = data.with_shared_factor_cache(1024);
        let footprint = data
            .laws()
            .iter()
            .try_fold(0usize, |bytes, law| {
                bytes.checked_add(law.probabilities().len().checked_mul(128)?)
            })
            .and_then(|bytes| bytes.checked_mul(query.at.len().checked_add(2)?))
            .ok_or_else(|| err("transport grid memory overflow"))?;
        if ctx
            .memory
            .hard_limit_bytes
            .is_some_and(|limit| u64::try_from(footprint).map_or(true, |n| n > limit))
        {
            return Err(err("transport grid memory budget"));
        }
        let plans = compile_grid(&query, &data, ctx)?;
        let graph = query.diagram.causal_graph();
        let edges = antecedent_io::admg_to_wire(graph)?;
        let template = GridWire {
            version: 2,
            required_features: vec![
                "checked_transport_proof_v1".into(),
                "retained_transport_grid_v2".into(),
            ],
            nodes: graph
                .nodes()
                .iter()
                .map(|n| match n {
                    NodeRef::Static(v) => Ok(v.raw()),
                    _ => Err(err("static transport only")),
                })
                .collect::<Result<_, _>>()?,
            directed: edges.directed,
            bidirected: edges.bidirected,
            selections: query.diagram.selection_targets().iter().map(|v| v.raw()).collect(),
            proof: TransportProofWire::from_checked(query.functional.derivation())?,
            catalog: EvidenceCatalogWire::from_catalog(query.functional.catalog()),
            laws: data.laws().iter().map(ExactLawWire::from_law).collect(),
            at: query.at.iter().map(assignments).collect(),
            operations: query.limits.operations,
            depth: query.limits.depth,
            seed: ctx.rng.master_seed(),
            statistical: matches!(input, TransportGridData::Statistical(..)),
            samples,
            options,
            points: vec![],
            point_evidence: vec![],
            reasoning: grid_reasoning(&[], matches!(input, TransportGridData::Statistical(..))),
        };
        Ok(PreparedStudy {
            state: TransportGridState {
                query,
                input: Some(input),
                data,
                seed: ctx.rng.master_seed(),
                template,
                plans,
            },
        })
    }
}
impl PreparedStudy<TransportGridState> {
    /// Metadata-only inspection of the immutable request, evidence and supported operations.
    #[must_use]
    pub fn query(&self) -> &TransportGridQuery {
        &self.state.query
    }
    /// Whether the retained native providers permit estimator replay without refresh.
    #[must_use]
    pub fn can_reestimate(&self) -> bool {
        self.state.input.is_some()
    }
    /// Frozen empirical inference settings, if this grid uses sampled providers.
    #[must_use]
    pub fn statistical_options(&self) -> Option<EmpiricalTableOptions> {
        self.state.template.options.as_ref().and_then(|o| o.to_options().ok())
    }
    /// Execute all supported points atomically; retain local support/missing-evidence outcomes.
    /// # Errors
    /// Global computational/numerical failure, cancellation, or missing raw samples after load.
    pub fn estimate(&self, ctx: &ExecutionContext) -> Result<TransportGridResult, IoError> {
        let input = self
            .state
            .input
            .as_ref()
            .ok_or_else(|| err("transport.samples_not_embedded; refresh required"))?;
        let mut ctx = ctx.clone();
        ctx.rng = antecedent_core::RngFactory::from_seed(self.state.seed);
        let query = &self.state.query;
        let limits = ExactEvaluationLimits {
            operations: query.limits.operations / query.at.len() / 8,
            depth: query.limits.depth,
        };
        ctx.memory.hard_limit_bytes =
            ctx.memory.hard_limit_bytes.map(|n| n / query.at.len() as u64);
        let mut failures = Vec::new();
        let mut eligible = Vec::new();
        // A point's eligibility pass already computes its distribution on the retained point
        // laws; keep it rather than evaluating the same plan again (an exact point publishes
        // it, a statistical point uses it as its point estimate under the joint bootstrap).
        let mut evaluated_points = Vec::new();
        for (at, plan) in query.at.iter().zip(&self.state.plans) {
            let evaluated = match plan {
                Ok(plan) => plan.evaluate(&ctx).map_err(|e| local_failure(e, &query.functional)),
                Err(failure) => Err(Ok(failure.clone())),
            };
            match evaluated {
                Ok(distribution) => {
                    failures.push(None);
                    eligible.push(at.clone());
                    evaluated_points.push(distribution);
                }
                Err(e) => failures.push(Some(e?)),
            }
        }
        let mut evaluated_points = Some(evaluated_points);
        let mut statistical = if let TransportGridData::Statistical(input, options) = input {
            if let Some(first) = eligible.first() {
                let study = PreparedStudy::<StatisticalPreparedState>::build_fitted(
                    query.diagram.clone(),
                    query.functional.clone(),
                    input.clone(),
                    self.state.data.clone(),
                    self.state.template.samples.clone(),
                    first.clone(),
                    limits,
                    *options,
                    &ctx,
                    false,
                )?;
                study
                    .estimate_grid_evaluated(
                        &eligible,
                        evaluated_points.take().unwrap_or_default(),
                        &ctx,
                    )?
                    .into_iter()
            } else {
                vec![].into_iter()
            }
        } else {
            vec![].into_iter()
        };
        let mut evaluated_exact = evaluated_points.unwrap_or_default().into_iter();
        let mut points = Vec::new();
        let mut wire = self.state.template.clone();
        wire.version = 2;
        wire.required_features =
            vec!["checked_transport_proof_v1".into(), "retained_transport_grid_v2".into()];
        wire.points.clear();
        wire.point_evidence.clear();
        for ((at, failure), plan) in query.at.iter().zip(failures).zip(&self.state.plans) {
            if let Some(failure) = failure {
                wire.points.push(GridPointWire::Unavailable(
                    TransportGridFailureWire::from_failure(&failure),
                ));
                points.push(TransportGridPoint::Unavailable(failure));
                continue;
            }
            match input {
                TransportGridData::Exact(_) => {
                    let study = PreparedStudy::<ExactPreparedState>::from_checked_plan(
                        query.diagram.clone(),
                        query.functional.clone(),
                        self.state.data.clone(),
                        at.clone(),
                        limits,
                        plan.as_ref().map_err(|_| err("missing compiled grid point"))?.clone(),
                        false,
                    )?;
                    let result = study.result_from_evaluation(
                        evaluated_exact.next().ok_or_else(|| err("missing exact grid result"))?,
                    );
                    wire.points.push(GridPointWire::Exact(vec![]));
                    points.push(TransportGridPoint::Exact(study, Box::new(result)));
                }
                TransportGridData::Statistical(..) => {
                    let (study, result) =
                        statistical.next().ok_or_else(|| err("missing joint grid result"))?;
                    wire.points.push(GridPointWire::Statistical(vec![]));
                    points.push(TransportGridPoint::Statistical(study, Box::new(result)));
                }
            }
        }
        if ctx.cancellation.is_cancelled() {
            return Err(err("transport.cancelled"));
        }
        wire.reasoning = grid_reasoning(&wire.points, wire.statistical);
        wire.point_evidence = points.iter().map(point_evidence).collect::<Result<_, _>>()?;
        Ok(TransportGridResult {
            identity: antecedent_io::transport_grid_wire::grid_identity(&wire)?,
            wire,
            points,
        })
    }
    /// Explicit atomic refresh, retaining the previous state if any global check fails.
    /// # Errors
    /// Changed structural contract, invalid providers, or failed execution.
    pub fn refresh(
        &mut self,
        input: TransportGridData,
        ctx: &ExecutionContext,
    ) -> Result<TransportGridResult, IoError> {
        let mut frozen = ctx.clone();
        frozen.rng = antecedent_core::RngFactory::from_seed(self.state.seed);
        let candidate = self.replacement(input, &frozen)?;
        let result = candidate.estimate(&frozen)?;
        *self = candidate;
        Ok(result)
    }
    /// Replace compatible provider snapshots and invalidate prior executions without estimating.
    /// # Errors
    /// Changed provider shape or incompatible sampling/evidence contract.
    pub fn replace_snapshot(
        &mut self,
        input: TransportGridData,
        ctx: &ExecutionContext,
    ) -> Result<(), IoError> {
        let candidate = self.replacement(input, ctx)?;
        *self = candidate;
        Ok(())
    }
    fn replacement(
        &self,
        input: TransportGridData,
        ctx: &ExecutionContext,
    ) -> Result<Self, IoError> {
        let mut query = self.state.query.clone();
        let catalog = rebind_snapshots(query.functional.catalog(), |regime| match &input {
            TransportGridData::Exact(data) => data
                .laws()
                .iter()
                .filter(|l| l.regime() == regime)
                .map(antecedent_expr::ExactDiscreteLaw::snapshot_identity)
                .collect(),
            TransportGridData::Statistical(data, _) => data
                .samples
                .iter()
                .filter(|s| s.regime == regime)
                .map(|s| s.snapshot_identity.as_ref())
                .chain(
                    data.supplied
                        .iter()
                        .filter(|l| l.regime() == regime)
                        .map(antecedent_expr::ExactDiscreteLaw::snapshot_identity),
                )
                .collect(),
        })?;
        query.functional = query.functional.derivation().bind_catalog(&catalog).map_err(err)?;
        let mut frozen = ctx.clone();
        frozen.rng = antecedent_core::RngFactory::from_seed(self.state.seed);
        let candidate = StudyBuilder::transport_grid(query, input, &frozen)?;

        let shape = |laws: &[ExactLawWire]| -> Result<Vec<Vec<u8>>, IoError> {
            let mut rows = laws
                .iter()
                .map(|law| {
                    let mut axes = law.axes.clone();
                    axes.sort_by_key(|(v, _)| *v);
                    let mut interventions = law.interventions.clone();
                    interventions.sort_by_key(|(v, _)| *v);
                    antecedent_io::to_cbor(&(
                        law.population.as_str(),
                        law.regime,
                        interventions,
                        axes,
                        law.origin.as_str(),
                    ))
                })
                .collect::<Result<Vec<_>, _>>()?;
            rows.sort();
            Ok(rows)
        };
        if shape(&candidate.state.template.laws)? != shape(&self.state.template.laws)? {
            return Err(err("transport.reprepare_required"));
        }
        if candidate.state.template.options != self.state.template.options {
            return Err(err("transport.reprepare_required: inference settings changed"));
        }
        Ok(candidate)
    }
}

// Extract only the enclosing bindings; the scalar consumers check every claim.
#[derive(Deserialize)]
struct PointBinding {
    proof: TransportProofWire,
    catalog: EvidenceCatalogWire,
    laws: Vec<ExactLawWire>,
    request: Vec<(u32, ValueWire)>,
    #[serde(default)]
    grid: Vec<Vec<(u32, ValueWire)>>,
    #[serde(default)]
    samples: Vec<SampleSummary>,
    #[serde(default)]
    options: Option<StatisticalOptionsWire>,
    #[serde(default)]
    seed: u64,
}
fn law_set(laws: &[ExactLawWire]) -> Result<Vec<Vec<u8>>, IoError> {
    let mut result = laws.iter().map(antecedent_io::to_cbor).collect::<Result<Vec<_>, _>>()?;
    result.sort();
    Ok(result)
}
impl PreparedStudy<TransportGridState> {
    /// Preview lifecycle invalidations using the common transformation contract.
    /// # Errors
    /// Identity serialization failure.
    pub fn preview_transform(
        &self,
        intent: antecedent_core::TransformIntent,
    ) -> Result<antecedent_core::TransformationReport, IoError> {
        use antecedent_core::{IdentityRef, SemanticDigest, TransformIntent, TransformationReport};
        let id = SemanticDigest::from_bytes(antecedent_io::external_estimate::parse_digest_hex(
            &digest(IdentityDomain::Execution, &self.state.template)?,
        )?);
        let report = TransformationReport::new(
            intent,
            vec![IdentityRef::new(IdentityDomain::Execution, id)],
            antecedent_core::intent_effects(intent).iter().cloned(),
            vec![],
        );
        Ok(
            if matches!(
                intent,
                TransformIntent::DisplayPrecision
                    | TransformIntent::FilterDisplay
                    | TransformIntent::CompatibleDataReplace
            ) {
                report
            } else {
                report.refused_on_handle(
                    "transport.reprepare_required: structural or evidence contract changed",
                )
            },
        )
    }
    /// Independently verify a durable curve using embedded proofs and tables only.
    /// Raw samples are never reconstructed; statistical re-estimation requires refresh.
    /// # Errors
    /// Unknown required features, substituted claims, malformed references, or resource failure.
    pub fn consume(
        bytes: &[u8],
        limits: ExactEvaluationLimits,
        ctx: &ExecutionContext,
    ) -> Result<(Self, TransportGridResult), IoError> {
        if ctx.cancellation.is_cancelled()
            || ctx.memory.hard_limit_bytes.is_some_and(|n| bytes.len() as u64 > n)
        {
            return Err(err("transport grid budget/cancellation"));
        }
        let artifact =
            antecedent_io::transport_certificate::read_bounded_transport_artifact(bytes, ctx)?;
        if artifact.manifest.artifact_kind
            != antecedent_io::wire::ArtifactKind::Other("transport_grid".into())
            || artifact.sections.len() != 1
            || artifact.sections[0].id != "transport_grid_v1"
        {
            return Err(err("unsupported transport grid artifact"));
        }
        let wire: GridWire = antecedent_io::from_cbor(&artifact.sections[0].data)?;
        if !matches!(wire.version, 1 | 2)
            || wire.required_features
                != [
                    "checked_transport_proof_v1",
                    if wire.version == 1 {
                        "retained_transport_grid_v1"
                    } else {
                        "retained_transport_grid_v2"
                    },
                ]
            || (wire.version == 1 && !wire.point_evidence.is_empty())
            || (wire.version == 2 && wire.point_evidence.len() != wire.points.len())
            || wire.operations > limits.operations
            || wire.depth > limits.depth
            || wire.at.is_empty()
            || wire.at.len() != wire.points.len()
            || wire.at.len() > limits.operations
        {
            return Err(err("unsupported transport grid features/version/limits"));
        }
        if wire.reasoning != grid_reasoning(&wire.points, wire.statistical) {
            return Err(err("grid reasoning mismatch"));
        }
        if antecedent_io::transport_grid_wire::grid_identity(&wire)?
            != artifact.manifest.artifact_id
        {
            return Err(err("transport grid identity mismatch"));
        }
        let (diagram, proof) = rebuild_checked_proof(
            &GraphFields {
                nodes: &wire.nodes,
                directed: &wire.directed,
                bidirected: &wire.bidirected,
                selections: &wire.selections,
            },
            &wire.proof,
            limits,
            ctx,
        )?;
        let catalog = wire.catalog.to_catalog()?;
        let functional = proof
            .bind_catalog_with_context(
                &catalog,
                SidLimits { steps: limits.operations, depth: limits.depth },
                ctx,
            )
            .map_err(err)?;
        validate_grid(
            &functional,
            &wire.at.iter().map(|a| request(a)).collect::<Vec<_>>(),
            ExactEvaluationLimits { operations: wire.operations, depth: wire.depth },
            ctx,
        )?;
        if wire.at.iter().any(|a| a.len() != request(a).entries().len()) {
            return Err(err("duplicate grid assignment coordinate"));
        }
        let data = ExactTransportData::try_new(
            wire.laws.iter().map(ExactLawWire::to_law).collect::<Result<Vec<_>, _>>()?,
            limits.operations,
        )
        .map_err(err)?;
        super::statistical::validate_sample_summaries(&wire.samples, &data, &catalog)?;
        if !wire.statistical
            && (!wire.samples.is_empty()
                || data
                    .laws()
                    .iter()
                    .any(|law| law.origin() != antecedent_expr::LawOrigin::SuppliedExact))
        {
            return Err(err("empirical grid mislabeled as supplied exact"));
        }
        if wire.statistical != wire.options.is_some() {
            return Err(err("grid provider modality mismatch"));
        }
        if let Some(options) = &wire.options {
            antecedent_estimate::empirical_table::validate_options(&options.to_options()?)
                .map_err(err)?;
        }
        let at: Vec<_> = wire.at.iter().map(|values| request(values)).collect();
        if at.iter().zip(&wire.at).any(|(a, b)| a.entries().len() != b.len()) {
            return Err(err("duplicate grid assignment"));
        }
        let executable: Vec<_> = wire
            .at
            .iter()
            .zip(&wire.points)
            .filter(|(_, p)| !matches!(p, GridPointWire::Unavailable(_)))
            .map(|(a, _)| a.clone())
            .collect();
        let point_limits = ExactEvaluationLimits {
            operations: wire.operations / wire.at.len(),
            depth: wire.depth,
        };
        let mut points = Vec::new();
        for ((at, values), point) in at.iter().zip(&wire.at).zip(&wire.points) {
            match point {
                GridPointWire::Unavailable(expected) => {
                    let evaluated = antecedent_estimate::prepare_exact_transport(
                        &functional,
                        data.clone(),
                        at.clone(),
                        point_limits,
                        ctx,
                    )
                    .and_then(|plan| plan.evaluate(ctx));
                    let actual = local_failure(
                        evaluated.err().ok_or_else(|| err("claimed grid failure is executable"))?,
                        &functional,
                    )?;
                    if actual != expected.to_failure() {
                        return Err(err("grid support claim mismatch"));
                    }
                    points.push(TransportGridPoint::Unavailable(actual));
                }
                GridPointWire::Exact(bytes) | GridPointWire::Statistical(bytes) => {
                    let binding: PointBinding = antecedent_io::from_cbor(bytes)?;
                    if antecedent_io::to_cbor(&binding.proof)?
                        != antecedent_io::to_cbor(&wire.proof)?
                        || binding.catalog != wire.catalog
                        || law_set(&binding.laws)? != law_set(&wire.laws)?
                        || &binding.request != values
                    {
                        return Err(err("substituted grid point inputs"));
                    }
                    match point {
                        GridPointWire::Exact(_) if !wire.statistical => {
                            let (study, result) = PreparedStudy::<ExactPreparedState>::consume(
                                bytes,
                                point_limits,
                                ctx,
                            )?;
                            points.push(TransportGridPoint::Exact(study, Box::new(result)));
                        }
                        GridPointWire::Statistical(_) if wire.statistical => {
                            if binding.grid != executable
                                || binding.samples != wire.samples
                                || binding.options != wire.options
                                || binding.seed != wire.seed
                            {
                                return Err(err("substituted grid inference family"));
                            }
                            let (study, result) =
                                PreparedStudy::<StatisticalPreparedState>::consume(
                                    bytes,
                                    point_limits,
                                    ctx,
                                )?;
                            points.push(TransportGridPoint::Statistical(study, Box::new(result)));
                        }
                        _ => return Err(err("substituted grid modality")),
                    }
                }
            }
        }
        if wire.version == 2
            && points.iter().map(point_evidence).collect::<Result<Vec<_>, _>>()?
                != wire.point_evidence
        {
            return Err(err("grid point evidence mismatch"));
        }
        let mut wire = wire;
        if wire.version == 2 {
            for point in &mut wire.points {
                match point {
                    GridPointWire::Exact(bytes) | GridPointWire::Statistical(bytes) => {
                        bytes.clear()
                    }
                    GridPointWire::Unavailable(_) => {}
                }
            }
        }
        let input =
            if wire.statistical { None } else { Some(TransportGridData::Exact(data.clone())) };
        let compiled_query = TransportGridQuery {
            diagram: diagram.clone(),
            functional: functional.clone(),
            at: at.clone(),
            limits: ExactEvaluationLimits { operations: wire.operations, depth: wire.depth },
        };
        let plans = compile_grid(&compiled_query, &data, ctx)?;
        let state = TransportGridState {
            plans,
            query: TransportGridQuery {
                diagram,
                functional,
                at,
                limits: ExactEvaluationLimits { operations: wire.operations, depth: wire.depth },
            },
            input,
            data,
            seed: wire.seed,
            template: GridWire { points: vec![], ..wire.clone() },
        };
        Ok((
            PreparedStudy { state },
            TransportGridResult { points, identity: artifact.manifest.artifact_id, wire },
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::{
        DistributionAvailability, Environment, EvidenceCatalog, EvidenceKind, EvidenceRegime,
        RegimeId, RegimeKind, Value, VariableCoordinate,
    };
    use antecedent_expr::{DiscreteAxis, ExactDiscreteLaw, InterventionAssignment, LawTolerance};
    use antecedent_graph::{Admg, DenseNodeId};
    use antecedent_identify::ClassicalTransportQuery;
    fn v(n: u32) -> VariableId {
        VariableId::from_raw(n)
    }
    fn fixture() -> PreparedStudy<TransportGridState> {
        let mut graph = Admg::with_variables(2);
        graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        graph.insert_bidirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let diagram = SelectionDiagram::try_new(graph, []).unwrap();
        let query = ClassicalTransportQuery {
            outcomes: Arc::from([v(1)]),
            treatments: Arc::from([v(0)]),
            source: Arc::from("source"),
            target: Arc::from("target"),
        };
        let ctx = ExecutionContext::for_tests(0);
        let antecedent_identify::ClassicalTransportResult::Identified(proof) =
            antecedent_identify::identify_classical_transport(
                &diagram,
                &query,
                SidLimits::default(),
                &ctx,
            )
            .unwrap()
        else {
            panic!()
        };
        let env = |name| {
            Environment::try_new(
                name,
                [0, 1].map(|n| VariableCoordinate {
                    variable: v(n),
                    domain: VariableDomain::Binary,
                    unit: None,
                }),
                [],
            )
            .unwrap()
        };
        let regime = EvidenceRegime::try_new(
            RegimeId::from_raw(0),
            RegimeKind::Experimental,
            EvidenceKind::Available,
            [v(0)],
            [],
            [v(1)],
            "source",
            DistributionAvailability::Joint,
        )
        .unwrap();
        let catalog =
            EvidenceCatalog::try_new([env("source"), env("target")], [regime], [], None).unwrap();
        let law = ExactDiscreteLaw::try_new(
            "source",
            RegimeId::from_raw(0),
            [InterventionAssignment { variable: v(0), value: Value::Int64(1) }],
            [DiscreteAxis {
                variable: v(1),
                values: Arc::from([Value::Int64(0), Value::Int64(1)]),
            }],
            [0.3, 0.7],
            "snapshot",
            LawTolerance::default(),
        )
        .unwrap();
        let data = ExactTransportData::try_new([law], 1000).unwrap();
        StudyBuilder::transport_grid(
            TransportGridQuery {
                diagram,
                functional: proof.bind_catalog(&catalog).unwrap(),
                at: vec![
                    Assignment::from_pairs([(v(0), Value::Int64(0))]),
                    Assignment::from_pairs([(v(0), Value::Int64(1))]),
                ],
                limits: ExactEvaluationLimits::default(),
            },
            TransportGridData::Exact(data),
            &ctx,
        )
        .unwrap()
    }
    #[test]
    fn portable_grid_reasoning_is_the_in_memory_view() {
        let ctx = ExecutionContext::for_tests(0);
        let result = fixture().estimate(&ctx).unwrap();
        assert_eq!(
            result.wire.reasoning,
            crate::analysis::contract::reasoning_section(&result.reasoning()),
            "one author for the in-memory and the exported slots"
        );
        let support = result.wire.reasoning.support.value.as_ref().expect("support slot");
        assert_eq!(support.matrix_status, "stage_contract");
        assert_eq!(support.matrix_coordinate.as_deref(), Some("transport.response_grid"));
        assert_eq!(support.empirical, "1 executable; 1 unavailable; population positivity assumed");
    }

    #[test]
    fn native_partial_grid_checks_semantics_beyond_container_checksums() {
        let ctx = ExecutionContext::for_tests(0);
        let study = fixture();
        let result = study.estimate(&ctx).unwrap();
        let (_, loaded) = PreparedStudy::<TransportGridState>::consume(
            &result.export().unwrap(),
            ExactEvaluationLimits::default(),
            &ctx,
        )
        .unwrap();
        assert_eq!(result.identity(), loaded.identity());
        assert!(matches!(loaded.points()[0], TransportGridPoint::Unavailable(_)));
        assert!(result.wire.points.iter().all(|point| match point {
            GridPointWire::Exact(bytes) | GridPointWire::Statistical(bytes) => bytes.is_empty(),
            GridPointWire::Unavailable(_) => true,
        }));
        let artifact = antecedent_io::transport_certificate::read_bounded_transport_artifact(
            &result.export().unwrap(),
            &ctx,
        )
        .unwrap();
        let encoded: GridWire = antecedent_io::from_cbor(&artifact.sections[0].data).unwrap();
        let mut legacy = encoded.clone();
        legacy.version = 1;
        legacy.point_evidence.clear();
        legacy.required_features =
            vec!["checked_transport_proof_v1".into(), "retained_transport_grid_v1".into()];
        let legacy_bytes =
            legacy.export(&digest(IdentityDomain::Execution, &legacy).unwrap()).unwrap();
        let (_, legacy_result) = PreparedStudy::<TransportGridState>::consume(
            &legacy_bytes,
            ExactEvaluationLimits::default(),
            &ctx,
        )
        .unwrap();
        assert_eq!(legacy_result.export().unwrap(), legacy_bytes);
        let reject = |mutated: GridWire| {
            let id = antecedent_io::transport_grid_wire::grid_identity(&mutated).unwrap();
            assert!(
                PreparedStudy::<TransportGridState>::consume(
                    &mutated.export(&id).unwrap(),
                    ExactEvaluationLimits::default(),
                    &ctx
                )
                .is_err()
            );
        };
        let mut wire = encoded.clone();
        wire.required_features.push("future_required_feature".into());
        reject(wire);
        let mut wire = encoded.clone();
        wire.at[0][0].1 = ValueWire::from_value(&Value::Int64(1));
        reject(wire);
        let mut wire = encoded.clone();
        wire.points.swap(0, 1);
        reject(wire);
        let mut wire = encoded.clone();
        wire.point_evidence[1] = "changed".into();
        reject(wire);
        let mut wire = encoded.clone();
        wire.laws[0].axes[0].0 = 0;
        reject(wire);
        let mut wire = encoded.clone();
        wire.proof.proof.target = "substituted".into();
        reject(wire);
        let mut wire = encoded.clone();
        wire.catalog.regimes[0].population = "wrong_population".into();
        reject(wire);
        let mut wire = encoded;
        if let GridPointWire::Unavailable(failure) = &mut wire.points[0] {
            failure.kind = "support_failure".into();
        }
        reject(wire);
        let mut tiny = ctx.clone();
        tiny.memory.hard_limit_bytes = Some(1);
        assert!(study.estimate(&tiny).is_err());
    }
}
