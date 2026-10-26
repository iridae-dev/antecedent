//! Retained finite transport grids. A local coverage failure is a point, not a deleted row.
use super::{
    ExactPreparedState, ExactStudyResult, PreparedStudy, StatisticalPreparedState,
    StatisticalStudyResult, StudyBuilder,
};
pub use antecedent_contract::TransportGridFailure;
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

use super::transport_common::{
    GraphFields, assignments, digest, err, rebind_snapshots, rebuild_checked_proof,
};
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
// Public declarations move into this private owner once. Scalar point studies and
// target branches share its immutable rows rather than constructing another DTO.
#[derive(Clone, Debug)]
enum RetainedGridData {
    Exact(ExactTransportData),
    Statistical(Arc<StatisticalTransportInput>, EmpiricalTableOptions),
}
fn retain_grid_input(input: TransportGridData) -> Result<Arc<RetainedGridData>, IoError> {
    Ok(Arc::new(match input {
        TransportGridData::Exact(data) => RetainedGridData::Exact(data),
        TransportGridData::Statistical(mut raw, options) => {
            super::statistical::canonicalize_input(&mut raw)?;
            RetainedGridData::Statistical(Arc::new(raw), options)
        }
    }))
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

/// Checked grid declarations whose statistical providers are fitted only on execution.
#[derive(Clone, Debug)]
pub struct DeferredTransportGrid {
    query: TransportGridQuery,
    input: Arc<RetainedGridData>,
    seed: u64,
    retention: Arc<antecedent_core::execution::AdmissionLease>,
    cache: Arc<super::statistical::MaterializationCache<GridProvider>>,
}
#[derive(Clone, Debug)]
struct GridProvider {
    data: ExactTransportData,
    samples: Vec<SampleSummary>,
    options: Option<StatisticalOptionsWire>,
    retention: Arc<antecedent_core::execution::AdmissionLease>,
}
impl DeferredTransportGrid {
    /// Retain checked proof, finite requests and raw providers without fitting.
    /// # Errors
    /// Invalid declarations, proof, cancellation or resource admission failure.
    pub fn compile(
        query: TransportGridQuery,
        input: TransportGridData,
        ctx: &ExecutionContext,
    ) -> Result<Self, IoError> {
        Self::compile_retained(query, retain_grid_input(input)?, None, ctx)
    }
    fn compile_retained(
        query: TransportGridQuery,
        input: Arc<RetainedGridData>,
        retained: Option<Arc<antecedent_core::execution::AdmissionLease>>,
        ctx: &ExecutionContext,
    ) -> Result<Self, IoError> {
        super::statistical::check_deferred_cancel(ctx)?;
        antecedent_identify::verify_classical_transport(
            &query.diagram,
            query.functional.derivation().query(),
            query.functional.derivation(),
            SidLimits { steps: query.limits.operations, depth: query.limits.depth },
            ctx,
        )
        .map_err(err)?;
        validate_grid(&query.functional, &query.at, query.limits, ctx)?;
        let bytes = match input.as_ref() {
            RetainedGridData::Exact(data) => {
                if data
                    .laws()
                    .iter()
                    .any(|law| law.origin() != antecedent_expr::LawOrigin::SuppliedExact)
                {
                    return Err(err("empirical laws require statistical transport"));
                }
                super::statistical::retained_law_bytes(data)?
            }
            RetainedGridData::Statistical(raw, options) => {
                antecedent_estimate::statistical_transport::check_statistical_resources(
                    raw,
                    &query.functional,
                    options,
                    ctx,
                )
                .map_err(err)?;
                // Validate recorded sample declarations without fitting their numerical laws.
                for sample in &raw.samples {
                    SampleSummary::from_sample(sample)?;
                }
                super::statistical::retained_input_bytes(raw)?
            }
        };
        let retention = match retained {
            Some(retention) => retention,
            None => Arc::new(
                ctx.acquire_admission(0, bytes)
                    .map_err(super::statistical::deferred_admission_error)?,
            ),
        };
        super::statistical::check_deferred_cancel(ctx)?;
        Ok(Self { query, input, seed: ctx.rng.master_seed(), retention, cache: Arc::default() })
    }
    /// Frozen structural request; it does not assert an executed grid identity.
    #[must_use]
    pub fn query(&self) -> &TransportGridQuery {
        &self.query
    }
    /// Structural reasoning before any provider is fitted or grid point executed.
    #[must_use]
    pub fn inspect(&self) -> antecedent_contract::ReasoningView {
        grid_reasoning_view(None, false)
    }
    /// Declared inference settings, if the providers are statistical.
    #[must_use]
    pub fn statistical_options(&self) -> Option<EmpiricalTableOptions> {
        match self.input.as_ref() {
            RetainedGridData::Statistical(_, options) => Some(*options),
            RetainedGridData::Exact(_) => None,
        }
    }
    /// Branch to a new ordered grid, sharing immutable raw evidence and compatible fitted laws.
    /// # Errors
    /// Invalid grid declarations, cancellation or computation budget failure.
    pub fn rebind(&self, at: Vec<Assignment>, ctx: &ExecutionContext) -> Result<Self, IoError> {
        super::statistical::check_deferred_cancel(ctx)?;
        validate_grid(&self.query.functional, &at, self.query.limits, ctx)?;
        let mut branch = self.clone();
        branch.query.at = at;
        Ok(branch)
    }
    /// Bind compatible provider snapshots without fitting or executing a target.
    /// # Errors
    /// Changed provider shape/options, invalid proof bindings or admission failure.
    pub fn bind_snapshot(
        &self,
        input: TransportGridData,
        ctx: &ExecutionContext,
    ) -> Result<Self, IoError> {
        let input = retain_grid_input(input)?;
        let shape = |input: &RetainedGridData| -> Result<String, IoError> {
            match input {
                RetainedGridData::Statistical(raw, options) => {
                    super::statistical::declared_statistical_shape(
                        raw,
                        &self.query.functional,
                        *options,
                    )
                }
                RetainedGridData::Exact(data) => digest(
                    IdentityDomain::Program,
                    &data.laws().iter().map(ExactLawWire::metadata).collect::<Vec<_>>(),
                ),
            }
        };
        if shape(&input)? != shape(&self.input)? {
            return Err(err("transport.reprepare_required"));
        }
        match (self.input.as_ref(), input.as_ref()) {
            (RetainedGridData::Exact(_), RetainedGridData::Exact(_)) => {}
            (RetainedGridData::Statistical(_, old), RetainedGridData::Statistical(_, new))
                if old == new => {}
            _ => return Err(err("transport.reprepare_required: inference settings changed")),
        }
        let mut query = self.query.clone();
        let catalog =
            rebind_snapshots(query.functional.catalog(), |regime| match input.as_ref() {
                RetainedGridData::Exact(data) => data
                    .laws()
                    .iter()
                    .filter(|law| law.regime() == regime)
                    .map(antecedent_expr::ExactDiscreteLaw::snapshot_identity)
                    .collect(),
                RetainedGridData::Statistical(raw, _) => raw
                    .samples
                    .iter()
                    .filter(|sample| sample.regime == regime)
                    .map(|sample| sample.snapshot_identity.as_ref())
                    .chain(
                        raw.supplied
                            .iter()
                            .filter(|law| law.regime() == regime)
                            .map(antecedent_expr::ExactDiscreteLaw::snapshot_identity),
                    )
                    .collect(),
            })?;
        query.functional = query
            .functional
            .derivation()
            .bind_catalog_with_context(
                &catalog,
                SidLimits { steps: query.limits.operations, depth: query.limits.depth },
                ctx,
            )
            .map_err(err)?;
        let mut frozen = ctx.clone();
        frozen.rng = antecedent_core::RngFactory::from_seed(self.seed);
        Self::compile_retained(query, input, None, &frozen)
    }
    /// Reconfigure an execution stream and physical policy without fitting or copying rows.
    /// # Errors
    /// Invalid declarations, cancelled execution or refused resource bounds.
    pub fn with_seed(&self, seed: u64, ctx: &ExecutionContext) -> Result<Self, IoError> {
        let mut frozen = ctx.clone();
        frozen.rng = antecedent_core::RngFactory::from_seed(seed);
        let mut branch = Self::compile_retained(
            self.query.clone(),
            Arc::clone(&self.input),
            Some(Arc::clone(&self.retention)),
            &frozen,
        )?;
        if seed == self.seed {
            branch.cache = Arc::clone(&self.cache);
        }
        Ok(branch)
    }
    /// Materialize finite provider laws under the actual caller's policy and admission pool.
    /// # Errors
    /// Provider fit, cancellation, admission or finite numerical plan failure.
    pub fn materialize(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<PreparedStudy<TransportGridState>, IoError> {
        super::statistical::check_deferred_cancel(ctx)?;
        self.retention.observe_current();
        let mut frozen = ctx.clone();
        frozen.rng = antecedent_core::RngFactory::from_seed(self.seed);
        let provider = self.cache.materialize(ctx, || {
            let bytes = match self.input.as_ref() {
                RetainedGridData::Exact(data) => super::statistical::retained_law_bytes(data)?,
                RetainedGridData::Statistical(raw, _) => {
                    super::statistical::declared_provider_bytes(raw, &self.query.functional)?
                }
            };
            let retention = Arc::new(
                ctx.acquire_admission(0, bytes)
                    .map_err(super::statistical::deferred_admission_error)?,
            );
            let (data, samples, options) = grid_provider_data(&self.input, &self.query, &frozen)?;
            if super::statistical::retained_law_bytes(&data)? > bytes {
                return Err(err("transport retained provider memory budget quote exceeded"));
            }
            super::statistical::check_deferred_cancel(ctx)?;
            Ok(super::statistical::MaterializedProvider {
                value: GridProvider { data, samples, options, retention: Arc::clone(&retention) },
                bytes,
                retention,
                kernel: ctx.kernel_policy,
            })
        })?;
        let mut prepared = build_grid(
            self.query.clone(),
            Arc::clone(&self.input),
            provider.data,
            provider.samples,
            provider.options,
            &frozen,
        )?;
        prepared.state.retention = vec![Arc::clone(&self.retention), provider.retention];
        super::statistical::check_deferred_cancel(ctx)?;
        Ok(prepared)
    }
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
    /// This coordinate's support in the shared response vocabulary.
    ///
    /// An executable coordinate is [`SupportStatus::Supported`](antecedent_core::SupportStatus).
    /// An unavailable one keeps its failure kind apart: absent evidence (no supplied law,
    /// provider, or declared domain) is `MissingEvidence`, while a support failure in the
    /// supplied evidence (an empty stratum, a zero denominator) is `OutsideEmpiricalSupport`.
    #[must_use]
    pub fn support_status(&self) -> antecedent_core::SupportStatus {
        match self {
            Self::Exact(..) | Self::Statistical(..) => antecedent_core::SupportStatus::Supported,
            Self::Unavailable(failure) if failure.kind == "missing_evidence" => {
                antecedent_core::SupportStatus::MissingEvidence
            }
            Self::Unavailable(_) => antecedent_core::SupportStatus::OutsideEmpiricalSupport,
        }
    }
}
/// Executed grid tally behind the support slot: points, missing-evidence points,
/// support-failure points, and whether the provider is statistical.
#[derive(Clone, Copy)]
struct GridTally {
    points: usize,
    missing: usize,
    failed: usize,
    statistical: bool,
}
impl GridTally {
    fn of<'a>(kinds: impl Iterator<Item = Option<&'a str>>, statistical: bool) -> Self {
        let mut tally = Self { points: 0, missing: 0, failed: 0, statistical };
        for kind in kinds {
            tally.points += 1;
            match kind {
                Some("missing_evidence") => tally.missing += 1,
                Some(_) => tally.failed += 1,
                None => {}
            }
        }
        tally
    }
    /// The support summary. A fully executable grid keeps the summary it always had; a
    /// grid with unavailable points separates missing evidence from support failures.
    /// `legacy` is the undifferentiated summary written before that split.
    fn empirical(self, legacy: bool) -> String {
        let unavailable = self.missing + self.failed;
        let executable = self.points - unavailable;
        if legacy || unavailable == 0 {
            format!(
                "{executable} executable; {unavailable} unavailable; population positivity assumed"
            )
        } else {
            format!(
                "{executable} executable; {unavailable} unavailable ({} missing evidence, {} \
                 support failure); population positivity assumed",
                self.missing, self.failed
            )
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
/// The checked diagram and bound functional every grid point shares. Held behind `Arc` so a
/// point's prepared study borrows them by reference count instead of copying the proof
/// arena and evidence catalog once per coordinate.
#[derive(Clone, Debug)]
struct SharedProof {
    diagram: Arc<SelectionDiagram>,
    functional: Arc<BoundTransportFunctional>,
}
impl SharedProof {
    fn of(query: &TransportGridQuery) -> Self {
        Self {
            diagram: Arc::new(query.diagram.clone()),
            functional: Arc::new(query.functional.clone()),
        }
    }
}
/// One compiled grid point: the shared evaluation plan or the local failure that replaced it.
type CompiledPoint = Result<Arc<antecedent_expr::ExactEvaluationPlan>, TransportGridFailure>;
/// Retained common prepared-study state for a finite grid.
#[derive(Clone, Debug)]
pub struct TransportGridState {
    query: TransportGridQuery,
    proof: SharedProof,
    input: Option<Arc<RetainedGridData>>,
    data: ExactTransportData,
    seed: u64,
    plans: Vec<CompiledPoint>,
    template: GridWire,
    retention: Vec<Arc<antecedent_core::execution::AdmissionLease>>,
}
/// A checked curve whose requested coordinates and local failures are durable.
#[derive(Clone, Debug)]
pub struct TransportGridResult {
    points: Vec<TransportGridPoint>,
    wire: GridWire,
    identity: String,
    retention: Vec<Arc<antecedent_core::execution::AdmissionLease>>,
}
impl TransportGridResult {
    /// Points in the original requested order.
    #[must_use]
    pub fn points(&self) -> &[TransportGridPoint] {
        &self.points
    }
    /// All four durable reasoning slots; per-point inference is retained in each point.
    #[must_use]
    pub fn reasoning(&self) -> antecedent_contract::ReasoningView {
        let kinds = self.points.iter().map(|point| match point {
            TransportGridPoint::Unavailable(failure) => Some(failure.kind.as_str()),
            _ => None,
        });
        grid_reasoning_view(Some(GridTally::of(kinds, self.wire.statistical)), false)
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
    ) -> Result<(f64, antecedent_contract::HandoffReceipt), IoError> {
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
            antecedent_contract::HandoffReceipt::lossy_scalar(identity, "transport_grid_scalar"),
        ))
    }
    /// Export a sectioned artifact with explicit required transport features.
    /// # Errors
    /// Serialization or container validation failure.
    pub fn export(&self) -> Result<Vec<u8>, IoError> {
        for retention in &self.retention {
            retention.observe_current();
        }
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

/// The one author of a grid's reasoning slots: the in-memory view, the portable section and
/// the pre-execution inspection all derive from it, so they can never disagree.
/// `execution` is the executed tally once the grid has run; before that, support and
/// uncertainty are execution-specific. `legacy` writes the support summary as it was
/// before missing evidence and support failures were counted apart.
fn grid_reasoning_view(
    execution: Option<GridTally>,
    legacy: bool,
) -> antecedent_contract::ReasoningView {
    use antecedent_contract::{
        AssumptionSlot, IdentificationSlot, ObligationKind, ObligationRecord, ObligationScope,
        ReasoningView, SlotAvailability, SupportSlot,
    };
    use antecedent_core::{AssumptionSource, AssumptionStatus, IdentificationStatus};
    ReasoningView::new(
        SlotAvailability::Available(IdentificationSlot::identified_singleton(
            IdentificationStatus::NonparametricallyIdentified,
        )),
        execution.map_or_else(
            || SlotAvailability::unavailable("execution_specific"),
            |tally| {
                SlotAvailability::Available(SupportSlot::new(
                    "stage_contract",
                    Some(Arc::from("transport.response_grid")),
                    SlotAvailability::Available(Arc::from(tally.empirical(legacy))),
                ))
            },
        ),
        SlotAvailability::unavailable(match execution {
            None => "execution_specific",
            Some(GridTally { statistical: true, .. }) => {
                "pointwise_components_in_retained_points; calibration_not_bound"
            }
            Some(GridTally { statistical: false, .. }) => {
                "exact_supplied_law_no_sampling_uncertainty"
            }
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
    legacy: bool,
) -> antecedent_io::contract_section::ReasoningSectionWire {
    let kinds = points.iter().map(|point| match point {
        GridPointWire::Unavailable(failure) => Some(failure.kind.as_str()),
        _ => None,
    });
    super::contract::reasoning_section(&grid_reasoning_view(
        Some(GridTally::of(kinds, statistical)),
        legacy,
    ))
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
) -> Result<Vec<CompiledPoint>, IoError> {
    let limits = ExactEvaluationLimits {
        operations: query.limits.operations / query.at.len() / 8,
        depth: query.limits.depth,
    };
    query
        .at
        .iter()
        .map(|at| {
            // `prepare_exact_transport` takes its providers by value; `ExactTransportData`
            // is a bundle of `Arc`s, so each point shares the laws rather than copying them.
            match antecedent_estimate::prepare_exact_transport(
                &query.functional,
                data.clone(),
                at.clone(),
                limits,
                ctx,
            ) {
                Ok(plan) => Ok(Ok(Arc::new(plan))),
                Err(error) => local_failure(error, &query.functional).map(Err),
            }
        })
        .collect()
}

/// Resolves the grid's provider laws, sample summaries, and frozen inference options from the supplied evidence.
fn grid_provider_data(
    input: &RetainedGridData,
    query: &TransportGridQuery,
    ctx: &ExecutionContext,
) -> Result<(ExactTransportData, Vec<SampleSummary>, Option<StatisticalOptionsWire>), IoError> {
    Ok(match input {
        RetainedGridData::Exact(data) => {
            if data.laws().iter().any(|l| l.origin() != antecedent_expr::LawOrigin::SuppliedExact) {
                return Err(err("empirical laws require statistical transport"));
            }
            (data.clone(), vec![], None)
        }
        RetainedGridData::Statistical(input, options) => {
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
    })
}

/// Per-point local failures, the eligible assignments, and their retained distributions.
struct EligibleGridPoints {
    failures: Vec<Option<TransportGridFailure>>,
    eligible: Vec<Assignment>,
    evaluated: Vec<ExactDistribution>,
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
        let input = retain_grid_input(input)?;
        let (data, samples, options) = grid_provider_data(&input, &query, ctx)?;
        build_grid(query, input, data, samples, options, ctx)
    }
}

fn build_grid(
    query: TransportGridQuery,
    input: Arc<RetainedGridData>,
    data: ExactTransportData,
    samples: Vec<SampleSummary>,
    options: Option<StatisticalOptionsWire>,
    ctx: &ExecutionContext,
) -> Result<PreparedStudy<TransportGridState>, IoError> {
    let data = if ctx.cache_policy.enabled
        && ctx.cache_policy.max_bytes.is_none()
        && ctx.memory_limit_bytes().is_none()
    {
        data.with_shared_factor_cache(1024)
    } else {
        data
    };
    let footprint = data
        .laws()
        .iter()
        .try_fold(0usize, |bytes, law| {
            bytes.checked_add(law.probabilities().len().checked_mul(128)?)
        })
        .and_then(|bytes| bytes.checked_mul(query.at.len().checked_add(2)?))
        .ok_or_else(|| err("transport grid memory overflow"))?;
    if ctx
        .memory_limit_bytes()
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
        statistical: matches!(input.as_ref(), RetainedGridData::Statistical(..)),
        samples,
        options,
        points: vec![],
        point_evidence: vec![],
        reasoning: grid_reasoning(
            &[],
            matches!(input.as_ref(), RetainedGridData::Statistical(..)),
            false,
        ),
    };
    let proof = SharedProof::of(&query);
    Ok(PreparedStudy {
        state: TransportGridState {
            query,
            proof,
            input: Some(input),
            data,
            seed: ctx.rng.master_seed(),
            template,
            plans,
            retention: Vec::new(),
        },
    })
}
impl PreparedStudy<TransportGridState> {
    /// Reconfigure an executable retained grid using its original immutable provider declarations.
    /// # Errors
    /// A loaded statistical grid has no raw declarations, or resource/cancellation refusal.
    pub fn defer_execution(
        &self,
        seed: u64,
        ctx: &ExecutionContext,
    ) -> Result<DeferredTransportGrid, IoError> {
        let input =
            self.state.input.as_ref().ok_or_else(|| {
                err("transport.samples_not_embedded; bind_snapshot before execution")
            })?;
        let mut frozen = ctx.clone();
        frozen.rng = antecedent_core::RngFactory::from_seed(seed);
        DeferredTransportGrid::compile_retained(
            self.state.query.clone(),
            Arc::clone(input),
            self.state.retention.first().cloned(),
            &frozen,
        )
    }
    /// Bind a compatible raw snapshot without provider fitting or target execution.
    /// # Errors
    /// Changed provider schema/options or failed proof/admission checks.
    pub fn deferred_snapshot(
        &self,
        input: TransportGridData,
        ctx: &ExecutionContext,
    ) -> Result<DeferredTransportGrid, IoError> {
        let mut query = self.state.query.clone();
        let catalog = rebind_snapshots(query.functional.catalog(), |regime| match &input {
            TransportGridData::Exact(data) => data
                .laws()
                .iter()
                .filter(|law| law.regime() == regime)
                .map(antecedent_expr::ExactDiscreteLaw::snapshot_identity)
                .collect(),
            TransportGridData::Statistical(raw, _) => raw
                .samples
                .iter()
                .filter(|sample| sample.regime == regime)
                .map(|sample| sample.snapshot_identity.as_ref())
                .chain(
                    raw.supplied
                        .iter()
                        .filter(|law| law.regime() == regime)
                        .map(antecedent_expr::ExactDiscreteLaw::snapshot_identity),
                )
                .collect(),
        })?;
        query.functional = query
            .functional
            .derivation()
            .bind_catalog_with_context(
                &catalog,
                SidLimits { steps: query.limits.operations, depth: query.limits.depth },
                ctx,
            )
            .map_err(err)?;
        let shape = match &input {
            TransportGridData::Statistical(raw, options) => {
                if Some(StatisticalOptionsWire::from_options(options))
                    != self.state.template.options
                {
                    return Err(err("transport.reprepare_required: inference settings changed"));
                }
                super::statistical::declared_statistical_shape(raw, &query.functional, *options)?
            }
            TransportGridData::Exact(data) if !self.state.template.statistical => {
                super::statistical::statistical_shape_wires(
                    data.laws().iter().map(ExactLawWire::metadata).collect(),
                )?
            }
            _ => return Err(err("transport.reprepare_required: provider modality changed")),
        };
        if shape != super::statistical::statistical_shape_wires(self.state.template.laws.clone())? {
            return Err(err("transport.reprepare_required"));
        }
        let mut frozen = ctx.clone();
        frozen.rng = antecedent_core::RngFactory::from_seed(self.state.seed);
        DeferredTransportGrid::compile(query, input, &frozen)
    }
    /// Branch to a new finite request using retained numerical providers without fitting.
    /// # Errors
    /// Missing raw statistical samples after loading or invalid request/plan.
    pub fn rebind_assignments(
        &self,
        at: Vec<Assignment>,
        ctx: &ExecutionContext,
    ) -> Result<Self, IoError> {
        let input = self
            .state
            .input
            .as_ref()
            .ok_or_else(|| err("transport.samples_not_embedded; bind_snapshot required"))?;
        let mut query = self.state.query.clone();
        query.at = at;
        validate_grid(&query.functional, &query.at, query.limits, ctx)?;
        let mut frozen = ctx.clone();
        frozen.rng = antecedent_core::RngFactory::from_seed(self.state.seed);
        let mut branch = build_grid(
            query,
            Arc::clone(input),
            self.state.data.clone(),
            self.state.template.samples.clone(),
            self.state.template.options.clone(),
            &frozen,
        )?;
        branch.state.retention.extend(self.state.retention.clone());
        Ok(branch)
    }
    /// Metadata-only inspection of the immutable request, evidence and supported operations.
    #[must_use]
    pub fn query(&self) -> &TransportGridQuery {
        &self.state.query
    }
    /// Reasoning before any point has executed: identification is a structural
    /// property of the accepted proof and the assumptions are declared; support
    /// and uncertainty stay execution-specific until the grid runs.
    #[must_use]
    pub fn inspect(&self) -> antecedent_contract::ReasoningView {
        grid_reasoning_view(None, false)
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
            .as_deref()
            .ok_or_else(|| err("transport.samples_not_embedded; refresh required"))?;
        let mut ctx = ctx.clone();
        ctx.rng = antecedent_core::RngFactory::from_seed(self.state.seed);
        let query = &self.state.query;
        let limits = ExactEvaluationLimits {
            operations: query.limits.operations / query.at.len() / 8,
            depth: query.limits.depth,
        };
        // Keep the caller's shared pool: point plans already carry their own operation limits.
        for retention in &self.state.retention {
            retention.observe_current();
        }
        let EligibleGridPoints { failures, eligible, evaluated } =
            self.evaluate_eligible_points(&ctx)?;
        let mut evaluated_points = Some(evaluated);
        let mut statistical = if let RetainedGridData::Statistical(input, options) = input {
            self.joint_statistical_estimates(
                input,
                options,
                &eligible,
                limits,
                &mut evaluated_points,
                &ctx,
            )?
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
                RetainedGridData::Exact(_) => {
                    let mut study = PreparedStudy::<ExactPreparedState>::from_checked_plan(
                        Arc::clone(&self.state.proof.diagram),
                        Arc::clone(&self.state.proof.functional),
                        self.state.data.clone(),
                        at.clone(),
                        limits,
                        Arc::clone(plan.as_ref().map_err(|_| err("missing compiled grid point"))?),
                        false,
                    )?;
                    study.state.provider_retention = self.state.retention.clone();
                    let result = study.result_from_evaluation(
                        evaluated_exact.next().ok_or_else(|| err("missing exact grid result"))?,
                    );
                    wire.points.push(GridPointWire::Exact(vec![]));
                    points.push(TransportGridPoint::Exact(study, Box::new(result)));
                }
                RetainedGridData::Statistical(..) => {
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
        wire.reasoning = grid_reasoning(&wire.points, wire.statistical, false);
        wire.point_evidence = points.iter().map(point_evidence).collect::<Result<_, _>>()?;
        Ok(TransportGridResult {
            identity: antecedent_io::transport_grid_wire::grid_identity(&wire)?,
            wire,
            points,
            retention: self.state.retention.clone(),
        })
    }
    /// Evaluates every compiled point once, splitting local failures from the eligible points and their distributions.
    fn evaluate_eligible_points(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<EligibleGridPoints, IoError> {
        let query = &self.state.query;
        let mut failures = Vec::new();
        let mut eligible = Vec::new();
        // A point's eligibility pass already computes its distribution on the retained point
        // laws; keep it rather than evaluating the same plan again (an exact point publishes
        // it, a statistical point uses it as its point estimate under the joint bootstrap).
        let mut evaluated_points = Vec::new();
        for (at, plan) in query.at.iter().zip(&self.state.plans) {
            let evaluated = match plan {
                Ok(plan) => plan.evaluate(ctx).map_err(|e| local_failure(e, &query.functional)),
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
        Ok(EligibleGridPoints { failures, eligible, evaluated: evaluated_points })
    }

    /// Fits the joint bootstrap over the eligible statistical points, consuming the retained point distributions.
    fn joint_statistical_estimates(
        &self,
        input: &Arc<StatisticalTransportInput>,
        options: &EmpiricalTableOptions,
        eligible: &[Assignment],
        limits: ExactEvaluationLimits,
        evaluated_points: &mut Option<Vec<ExactDistribution>>,
        ctx: &ExecutionContext,
    ) -> Result<
        std::vec::IntoIter<(PreparedStudy<StatisticalPreparedState>, StatisticalStudyResult)>,
        IoError,
    > {
        let query = &self.state.query;
        Ok(if let Some(first) = eligible.first() {
            let mut study = PreparedStudy::<StatisticalPreparedState>::build_fitted(
                super::statistical::StatisticalPremises {
                    diagram: query.diagram.clone(),
                    functional: query.functional.clone(),
                    input: Arc::clone(input),
                    data: self.state.data.clone(),
                    samples: self.state.template.samples.clone(),
                    request: first.clone(),
                    limits,
                    options: *options,
                },
                ctx,
                false,
            )?;
            study.state.provider_retention = self.state.retention.clone();
            study
                .estimate_grid_evaluated(
                    eligible,
                    evaluated_points.take().unwrap_or_default(),
                    ctx,
                )?
                .into_iter()
        } else {
            vec![].into_iter()
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
        intent: antecedent_contract::TransformIntent,
    ) -> Result<antecedent_contract::TransformationReport, IoError> {
        use antecedent_contract::{TransformIntent, TransformationReport};
        use antecedent_core::{IdentityRef, SemanticDigest};
        let id = SemanticDigest::from_bytes(antecedent_io::external_estimate::parse_digest_hex(
            &digest(IdentityDomain::Execution, &self.state.template)?,
        )?);
        let report = TransformationReport::new(
            intent,
            vec![IdentityRef::new(IdentityDomain::Execution, id)],
            antecedent_contract::intent_effects(intent).iter().cloned(),
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
        let (wire, artifact_id) = read_grid_wire(bytes, limits, ctx)?;
        let (diagram, functional, data) = rebuild_grid_premises(&wire, limits, ctx)?;
        let at: Vec<_> = wire.at.iter().map(|values| request(values)).collect();
        if at.iter().zip(&wire.at).any(|(a, b)| a.entries().len() != b.len()) {
            return Err(err("duplicate grid assignment"));
        }
        let points = verify_grid_points(&wire, &at, &functional, &data, ctx)?;
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
        // The template is the wire without its points; take them out so they are not copied
        // only to be dropped.
        let retained_points = std::mem::take(&mut wire.points);
        let template = wire.clone();
        wire.points = retained_points;
        let input = if wire.statistical {
            None
        } else {
            Some(Arc::new(RetainedGridData::Exact(data.clone())))
        };
        let query = TransportGridQuery {
            diagram,
            functional,
            at,
            limits: ExactEvaluationLimits { operations: wire.operations, depth: wire.depth },
        };
        let plans = compile_grid(&query, &data, ctx)?;
        let state = TransportGridState {
            plans,
            retention: Vec::new(),
            proof: SharedProof::of(&query),
            query,
            input,
            data,
            seed: wire.seed,
            template,
        };
        Ok((
            PreparedStudy { state },
            TransportGridResult { points, identity: artifact_id, wire, retention: Vec::new() },
        ))
    }
}

/// Reads the bounded artifact, checks its container, features, reasoning, and identity, and returns the wire with its artifact id.
fn read_grid_wire(
    bytes: &[u8],
    limits: ExactEvaluationLimits,
    ctx: &ExecutionContext,
) -> Result<(GridWire, String), IoError> {
    if ctx.cancellation.is_cancelled()
        || ctx.memory_limit_bytes().is_some_and(|n| bytes.len() as u64 > n)
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
    // A grid written before the support summary separated missing evidence from
    // support failures carries the undifferentiated summary; both are this grid's.
    if wire.reasoning != grid_reasoning(&wire.points, wire.statistical, false)
        && wire.reasoning != grid_reasoning(&wire.points, wire.statistical, true)
    {
        return Err(err("grid reasoning mismatch"));
    }
    if antecedent_io::transport_grid_wire::grid_identity(&wire)? != artifact.manifest.artifact_id {
        return Err(err("transport grid identity mismatch"));
    }
    Ok((wire, artifact.manifest.artifact_id))
}

/// Rebuilds the checked proof, bound functional, and exact laws a grid artifact claims, validating its declared inputs.
fn rebuild_grid_premises(
    wire: &GridWire,
    limits: ExactEvaluationLimits,
    ctx: &ExecutionContext,
) -> Result<(SelectionDiagram, BoundTransportFunctional, ExactTransportData), IoError> {
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
    Ok((diagram, functional, data))
}

/// Re-derives each retained point from its embedded claim, rejecting substituted inputs, modalities, and failures.
fn verify_grid_points(
    wire: &GridWire,
    at: &[Assignment],
    functional: &BoundTransportFunctional,
    data: &ExactTransportData,
    ctx: &ExecutionContext,
) -> Result<Vec<TransportGridPoint>, IoError> {
    let executable: Vec<_> = wire
        .at
        .iter()
        .zip(&wire.points)
        .filter(|(_, p)| !matches!(p, GridPointWire::Unavailable(_)))
        .map(|(a, _)| a.clone())
        .collect();
    let point_limits =
        ExactEvaluationLimits { operations: wire.operations / wire.at.len(), depth: wire.depth };
    let mut points = Vec::new();
    for ((at, values), point) in at.iter().zip(&wire.at).zip(&wire.points) {
        match point {
            GridPointWire::Unavailable(expected) => {
                let evaluated = antecedent_estimate::prepare_exact_transport(
                    functional,
                    data.clone(),
                    at.clone(),
                    point_limits,
                    ctx,
                )
                .and_then(|plan| plan.evaluate(ctx));
                let actual = local_failure(
                    evaluated.err().ok_or_else(|| err("claimed grid failure is executable"))?,
                    functional,
                )?;
                if actual != expected.to_failure() {
                    return Err(err("grid support claim mismatch"));
                }
                points.push(TransportGridPoint::Unavailable(actual));
            }
            GridPointWire::Exact(bytes) | GridPointWire::Statistical(bytes) => {
                let binding: PointBinding = antecedent_io::from_cbor(bytes)?;
                if antecedent_io::to_cbor(&binding.proof)? != antecedent_io::to_cbor(&wire.proof)?
                    || binding.catalog != wire.catalog
                    || law_set(&binding.laws)? != law_set(&wire.laws)?
                    || &binding.request != values
                {
                    return Err(err("substituted grid point inputs"));
                }
                match point {
                    GridPointWire::Exact(_) if !wire.statistical => {
                        let (study, result) =
                            PreparedStudy::<ExactPreparedState>::consume(bytes, point_limits, ctx)?;
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
                        let (study, result) = PreparedStudy::<StatisticalPreparedState>::consume(
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
    Ok(points)
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
    fn sampled_grid_input() -> TransportGridData {
        let raw = StatisticalTransportInput {
            supplied: vec![],
            samples: vec![antecedent_estimate::RegimeSample {
                population: Arc::from("source"),
                regime: RegimeId::from_raw(0),
                snapshot_identity: Arc::from("snapshot"),
                interventions: Arc::from([InterventionAssignment {
                    variable: v(0),
                    value: Value::Int64(1),
                }]),
                columns: std::collections::BTreeMap::from([(
                    v(1),
                    vec![
                        Some(0.0),
                        Some(0.0),
                        Some(0.0),
                        Some(1.0),
                        Some(1.0),
                        Some(1.0),
                        Some(1.0),
                        Some(1.0),
                        Some(1.0),
                        Some(1.0),
                    ],
                )]),
            }],
        };
        let options = EmpiricalTableOptions { bootstrap_replicates: 0, ..Default::default() };
        TransportGridData::Statistical(raw, options)
    }

    #[test]
    fn deferred_grid_preserves_ordered_failures_identity_and_artifact_bytes() {
        let mut ctx = ExecutionContext::for_tests(0);
        ctx.cache_policy = antecedent_core::CachePolicy::enabled(None);
        let eager = fixture();
        let deferred = DeferredTransportGrid::compile(
            eager.state.query.clone(),
            TransportGridData::Exact(eager.state.data.clone()),
            &ctx,
        )
        .unwrap();
        assert!(!deferred.inspect().support.is_available());
        let expected = eager.estimate(&ctx).unwrap();
        let materialized = deferred.materialize(&ctx).unwrap();
        let actual = materialized.estimate(&ctx).unwrap();
        assert_eq!(actual.identity(), expected.identity());
        assert_eq!(actual.export().unwrap(), expected.export().unwrap());
        assert!(matches!(actual.points()[0], TransportGridPoint::Unavailable(_)));
        let branch = deferred.rebind(vec![eager.state.query.at[1].clone()], &ctx).unwrap();
        assert_eq!(deferred.query().at.len(), 2);
        assert_eq!(branch.query().at.len(), 1);
        let next = branch.materialize(&ctx).unwrap();
        assert!(Arc::ptr_eq(&materialized.state.retention[1], &next.state.retention[1]));
        assert_eq!(next.estimate(&ctx).unwrap().points().len(), 1);
    }
    #[test]
    fn deferred_statistical_grid_fits_only_on_execution_and_honors_caller_cache_pool() {
        let mut ctx = ExecutionContext::for_tests(0);
        ctx.cache_policy = antecedent_core::CachePolicy::enabled(None);
        let eager = fixture();
        let input = sampled_grid_input();
        let deferred =
            DeferredTransportGrid::compile(eager.state.query.clone(), input.clone(), &ctx).unwrap();
        let first = deferred.materialize(&ctx).unwrap();
        let second = deferred.materialize(&ctx).unwrap();
        assert!(Arc::ptr_eq(&first.state.retention[1], &second.state.retention[1]));
        let old = StudyBuilder::transport_grid(eager.state.query.clone(), input, &ctx)
            .unwrap()
            .estimate(&ctx)
            .unwrap();
        let actual = first.estimate(&ctx).unwrap();
        assert!(matches!(actual.points()[0], TransportGridPoint::Unavailable(_)));
        assert_eq!(actual.identity(), old.identity());
        assert_eq!(actual.export().unwrap(), old.export().unwrap());
        let mut disabled = ctx.clone();
        disabled.cache_policy.enabled = false;
        let uncached = deferred.materialize(&disabled).unwrap();
        assert!(!Arc::ptr_eq(&first.state.retention[1], &uncached.state.retention[1]));
        let mut bounded = ctx.clone();
        bounded.cache_policy.max_bytes = Some(0);
        let bounded = deferred.materialize(&bounded).unwrap();
        assert!(!Arc::ptr_eq(&first.state.retention[1], &bounded.state.retention[1]));
        let mut tiny = ctx.clone();
        tiny.set_memory_limit(Some(1));
        assert!(deferred.materialize(&tiny).is_err());
        assert_eq!(tiny.admission.available().bytes(), 1);
        let cancelled = ctx.clone();
        cancelled.cancellation.cancel();
        assert!(deferred.materialize(&cancelled).is_err());
    }
    #[test]
    fn deferred_grid_shares_raw_and_fitted_storage_across_targets_with_one_admission() {
        let baseline = fixture();
        let input = sampled_grid_input();
        let TransportGridData::Statistical(raw, options) = &input else {
            panic!("statistical fixture");
        };
        let raw_bytes = super::super::statistical::retained_input_bytes(raw).unwrap();
        let law_bytes = super::super::statistical::declared_provider_bytes(
            raw,
            &baseline.state.query.functional,
        )
        .unwrap();
        // Empirical fitting has a separate transient-workspace preflight. Allow
        // that actual quote for the initial fit, then occupy all spare capacity
        // before branching: retained rows and laws must never be admitted twice.
        let (_, workspace_bytes) =
            antecedent_estimate::statistical_transport::check_statistical_resources(
                raw,
                &baseline.state.query.functional,
                options,
                &ExecutionContext::for_tests(0),
            )
            .unwrap();
        let retained_bytes = raw_bytes + law_bytes;
        let capacity = retained_bytes.max(u64::try_from(workspace_bytes).unwrap());
        let pool = antecedent_core::execution::AdmissionPool::fixed(1, capacity);
        let mut ctx = ExecutionContext::for_tests(0).with_admission(pool.clone());
        ctx.cache_policy = antecedent_core::CachePolicy::enabled(None);
        let deferred =
            DeferredTransportGrid::compile(baseline.state.query.clone(), input, &ctx).unwrap();
        let first = deferred.materialize(&ctx).unwrap();
        assert_eq!(pool.available().bytes(), capacity - retained_bytes);
        let headroom = ctx.acquire_admission(0, capacity - retained_bytes).unwrap();
        assert_eq!(pool.available().bytes(), 0);
        let target = vec![baseline.state.query.at[1].clone()];
        let branch = deferred.rebind(target.clone(), &ctx).unwrap();
        let second = branch.materialize(&ctx).unwrap();
        let third = first.rebind_assignments(target, &ctx).unwrap();
        let rows = |state: &TransportGridState| match state.input.as_deref().unwrap() {
            RetainedGridData::Statistical(raw, _) => Arc::clone(raw),
            RetainedGridData::Exact(_) => panic!("statistical fixture"),
        };
        assert!(Arc::ptr_eq(&rows(&first.state), &rows(&second.state)));
        assert!(Arc::ptr_eq(&rows(&first.state), &rows(&third.state)));
        assert!(Arc::ptr_eq(&first.state.retention[1], &second.state.retention[1]));
        assert!(std::ptr::eq(
            first.state.data.laws()[0].probabilities().as_ptr(),
            second.state.data.laws()[0].probabilities().as_ptr()
        ));
        assert_eq!(pool.available().bytes(), 0);
        drop((deferred, branch, first, second, third, headroom));
        assert_eq!(pool.available().bytes(), capacity);
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
        assert_eq!(
            support.empirical,
            "1 executable; 1 unavailable (1 missing evidence, 0 support failure); \
             population positivity assumed"
        );
    }

    #[test]
    fn grid_support_distinguishes_missing_evidence_from_a_support_failure() {
        use antecedent_core::SupportStatus;
        let ctx = ExecutionContext::for_tests(0);
        let result = fixture().estimate(&ctx).unwrap();
        // No law is supplied under do(X = 0): that coordinate lacks evidence; it is not a
        // support verdict about the evidence that was supplied.
        let TransportGridPoint::Unavailable(missing) = &result.points()[0] else {
            panic!("the unbound coordinate is retained as unavailable");
        };
        assert_eq!(missing.kind, "missing_evidence");
        assert_eq!(result.points()[0].support_status(), SupportStatus::MissingEvidence);
        assert_eq!(result.points()[1].support_status(), SupportStatus::Supported);
        let mut failed = missing.clone();
        failed.kind = "support_failure".into();
        assert_eq!(
            TransportGridPoint::Unavailable(failed).support_status(),
            SupportStatus::OutsideEmpiricalSupport
        );
        let support = result.wire.reasoning.support.value.as_ref().expect("support slot");
        assert_eq!(
            support.empirical,
            "1 executable; 1 unavailable (1 missing evidence, 0 support failure); \
             population positivity assumed"
        );
    }

    #[test]
    fn grid_artifact_with_the_undifferentiated_support_summary_still_loads() {
        // Artifacts written before the support summary split its unavailable count keep
        // loading, and re-export byte for byte.
        let ctx = ExecutionContext::for_tests(0);
        let result = fixture().estimate(&ctx).unwrap();
        let artifact = antecedent_io::transport_certificate::read_bounded_transport_artifact(
            &result.export().unwrap(),
            &ctx,
        )
        .unwrap();
        let mut wire: GridWire = antecedent_io::from_cbor(&artifact.sections[0].data).unwrap();
        let support = wire.reasoning.support.value.as_mut().expect("support slot");
        support.empirical = "1 executable; 1 unavailable; population positivity assumed".into();
        let id = antecedent_io::transport_grid_wire::grid_identity(&wire).unwrap();
        let bytes = wire.export(&id).unwrap();
        let (_, loaded) = PreparedStudy::<TransportGridState>::consume(
            &bytes,
            ExactEvaluationLimits::default(),
            &ctx,
        )
        .unwrap();
        assert_eq!(loaded.identity(), id);
        assert_eq!(loaded.export().unwrap(), bytes);
        assert_eq!(
            loaded.points()[0].support_status(),
            antecedent_core::SupportStatus::MissingEvidence
        );
        // Any other summary is still a mismatch.
        let support = wire.reasoning.support.value.as_mut().expect("support slot");
        support.empirical = "2 executable; 0 unavailable; population positivity assumed".into();
        let id = antecedent_io::transport_grid_wire::grid_identity(&wire).unwrap();
        assert!(
            PreparedStudy::<TransportGridState>::consume(
                &wire.export(&id).unwrap(),
                ExactEvaluationLimits::default(),
                &ctx
            )
            .is_err()
        );
    }

    #[test]
    fn prepared_grid_inspection_shares_the_executed_identification_and_assumptions() {
        let ctx = ExecutionContext::for_tests(0);
        let study = fixture();
        let before = study.inspect();
        let after = study.estimate(&ctx).unwrap().reasoning();
        assert_eq!(before.identification, after.identification);
        assert_eq!(before.assumptions, after.assumptions);
        assert!(!before.support.is_available() && after.support.is_available());
        assert!(!before.uncertainty.is_available());
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
        tiny.set_memory_limit(Some(1));
        assert!(study.estimate(&tiny).is_err());
    }
}
