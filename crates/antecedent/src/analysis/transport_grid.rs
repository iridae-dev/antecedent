//! Retained finite transport grids. A local coverage failure is a point, not a deleted row.
use super::statistical::{SampleSummary, StatisticalOptionsWire};
use super::{
    ExactPreparedState, ExactStudyResult, PreparedStudy, StatisticalPreparedState,
    StatisticalStudyResult, StudyBuilder,
};
use antecedent_core::{ExecutionContext, IdentityDomain, NodeRef, VariableDomain, VariableId};
use antecedent_estimate::{EmpiricalTableOptions, StatisticalTransportInput};
use antecedent_expr::{
    Assignment, EvalError, ExactDistribution, ExactEvaluationLimits, ExactTransportData,
};
use antecedent_graph::{Admg, DenseNodeId, SelectionDiagram};
use antecedent_identify::{BoundTransportFunctional, ClassicalTransportQuery, SidLimits};
use antecedent_io::{
    IoError, exact_law_wire::ExactLawWire, query_wire::ValueWire,
    transport_catalog_wire::EvidenceCatalogWire, transport_proof::TransportProofWire,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

fn err(e: impl std::fmt::Display) -> IoError {
    IoError::Convert(e.to_string())
}
fn digest(value: &impl Serialize) -> Result<String, IoError> {
    Ok(antecedent_io::identity::digest_wire(IdentityDomain::Execution, value)?.to_hex())
}
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
/// Located point-local missing evidence or support outcome.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TransportGridFailure {
    /// Stable kind: missing_evidence or support_failure.
    pub kind: String,
    /// Located provider/denominator explanation.
    pub detail: String,
    /// Stable provider/support failure code.
    pub code: String,
    /// Original variables whose provider support is required.
    pub variables: Vec<u32>,
    /// Original expression coordinate for a failed ratio.
    pub expression: Option<u32>,
    /// Located original variable assignments.
    pub assignment: Vec<(u32, ValueWire)>,
    /// Population/regime dependencies of the failing factor.
    pub bindings: Vec<(String, Option<u32>)>,
    /// Concrete intervention world required by this factor.
    pub interventions: Vec<(u32, ValueWire)>,
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
            failure.assignment =
                e.conditioning.iter().map(|(v, x)| (v.raw(), ValueWire::from_value(x))).collect();
            failure
                .bindings
                .push((e.population.to_string(), e.regime.map(antecedent_core::RegimeId::raw)));
            failure.interventions = e
                .interventions
                .iter()
                .map(|a| (a.variable.raw(), ValueWire::from_value(&a.value)))
                .collect();
        }
        EvalError::ExactRatioSupport { expression, assignment, bindings } => {
            failure.code = "zero_ratio_denominator".into();
            failure.expression = Some(expression.raw());
            failure.assignment =
                assignment.iter().map(|(v, x)| (v.raw(), ValueWire::from_value(x))).collect();
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
/// Retained common prepared-study state for a finite grid.
#[derive(Clone, Debug)]
pub struct TransportGridState {
    query: TransportGridQuery,
    input: Option<TransportGridData>,
    data: ExactTransportData,
    seed: u64,
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
    pub fn reasoning(&self) -> &antecedent_io::contract_section::ReasoningSectionWire {
        &self.wire.reasoning
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
        use antecedent_io::container::{
            ArtifactManifest, CompressPolicy, EncodedArtifact, pack_section,
        };
        use antecedent_io::wire::{ArtifactKind, FormatVersion, ProvenanceWire, SemanticVersion};
        let (descriptor, section) = pack_section(
            "transport_grid_v1",
            "application/cbor",
            antecedent_io::to_cbor(&self.wire)?,
            CompressPolicy::Never,
        );
        let artifact=EncodedArtifact { manifest:ArtifactManifest {format_version:FormatVersion {major:1,minor:0},minimum_reader_version:FormatVersion {major:1,minor:0},artifact_kind:ArtifactKind::Other("transport_grid".into()),library_version:SemanticVersion::from_crate_version(env!("CARGO_PKG_VERSION")).map_err(err)?,artifact_id:self.identity.clone(),sections:vec![descriptor],provenance:ProvenanceWire {note:"checked finite transport grid; pointwise uncertainty; no new calibration claim".into()}},sections:vec![section]};
        let mut bytes = Vec::new();
        artifact.write_to(&mut bytes)?;
        Ok(bytes)
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct GridWire {
    version: u32,
    required_features: Vec<String>,
    nodes: Vec<u32>,
    directed: Vec<(u32, u32)>,
    bidirected: Vec<(u32, u32)>,
    selections: Vec<u32>,
    proof: TransportProofWire,
    catalog: EvidenceCatalogWire,
    laws: Vec<ExactLawWire>,
    at: Vec<Vec<(u32, ValueWire)>>,
    operations: usize,
    depth: usize,
    seed: u64,
    statistical: bool,
    samples: Vec<SampleSummary>,
    options: Option<StatisticalOptionsWire>,
    points: Vec<GridPointWire>,
    reasoning: antecedent_io::contract_section::ReasoningSectionWire,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
enum GridPointWire {
    Exact(Vec<u8>),
    Statistical(Vec<u8>),
    Unavailable(TransportGridFailure),
}
fn grid_reasoning(
    proof: &TransportProofWire,
    points: &[GridPointWire],
    statistical: bool,
) -> antecedent_io::contract_section::ReasoningSectionWire {
    use antecedent_io::contract_section::{SlotSectionWire, SupportSlotWire};
    let mut slots = antecedent_io::transport_certificate::structural_reasoning(
        &antecedent_io::transport_certificate::CertificateOutcome::Identified(proof.clone()),
    );
    let unavailable = points.iter().filter(|p| matches!(p, GridPointWire::Unavailable(_))).count();
    slots.support = SlotSectionWire {
        value: Some(SupportSlotWire {
            matrix_status: "exact_validation_only".into(),
            matrix_coordinate: None,
            empirical: format!(
                "{} executable; {} unavailable; population positivity assumed",
                points.len() - unavailable,
                unavailable
            ),
        }),
        unavailable: None,
    };
    slots.uncertainty.unavailable = Some(
        if statistical {
            "pointwise_components_in_retained_points; calibration_not_bound"
        } else {
            "exact_supplied_law_no_sampling_uncertainty"
        }
        .into(),
    );
    slots
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
                    .map(super::statistical::SampleSummary::from_sample)
                    .collect::<Result<Vec<_>, _>>()?;
                (
                    antecedent_estimate::empirical_table::assemble_grid_point_laws(
                        input,
                        &query.functional,
                        options,
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
        let graph = query.diagram.causal_graph();
        let edges = antecedent_io::admg_to_wire(graph)?;
        let template = GridWire {
            version: 1,
            required_features: vec![
                "checked_transport_proof_v1".into(),
                "retained_transport_grid_v1".into(),
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
            reasoning: grid_reasoning(
                &TransportProofWire::from_checked(query.functional.derivation())?,
                &[],
                matches!(input, TransportGridData::Statistical(..)),
            ),
        };
        Ok(PreparedStudy {
            state: TransportGridState {
                query,
                input: Some(input),
                data,
                seed: ctx.rng.master_seed(),
                template,
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
        for at in &query.at {
            let evaluated = antecedent_estimate::prepare_exact_transport(
                &query.functional,
                self.state.data.clone(),
                at.clone(),
                limits,
                &ctx,
            )
            .and_then(|plan| plan.evaluate(&ctx));
            match evaluated {
                Ok(_) => {
                    failures.push(None);
                    eligible.push(at.clone());
                }
                Err(e) => failures.push(Some(local_failure(e, &query.functional)?)),
            }
        }
        let mut statistical = if let TransportGridData::Statistical(input, options) = input {
            if let Some(first) = eligible.first() {
                let study = StudyBuilder::statistical_transport(
                    query.diagram.clone(),
                    query.functional.clone(),
                    input.clone(),
                    first.clone(),
                    limits,
                    *options,
                    &ctx,
                )?;
                study.estimate_grid(&eligible, &ctx)?.into_iter()
            } else {
                vec![].into_iter()
            }
        } else {
            vec![].into_iter()
        };
        let mut points = Vec::new();
        let mut wire = self.state.template.clone();
        for (at, failure) in query.at.iter().zip(failures) {
            if let Some(failure) = failure {
                wire.points.push(GridPointWire::Unavailable(failure.clone()));
                points.push(TransportGridPoint::Unavailable(failure));
                continue;
            }
            match input {
                TransportGridData::Exact(_) => {
                    let study = StudyBuilder::exact_transport(
                        query.diagram.clone(),
                        query.functional.clone(),
                        self.state.data.clone(),
                        at.clone(),
                        limits,
                        &ctx,
                    )?;
                    let result = study.estimate(&ctx)?;
                    wire.points.push(GridPointWire::Exact(study.export(&result)?));
                    points.push(TransportGridPoint::Exact(study, Box::new(result)));
                }
                TransportGridData::Statistical(..) => {
                    let (study, result) =
                        statistical.next().ok_or_else(|| err("missing joint grid result"))?;
                    wire.points.push(GridPointWire::Statistical(study.export(&result)?));
                    points.push(TransportGridPoint::Statistical(study, Box::new(result)));
                }
            }
        }
        if ctx.cancellation.is_cancelled() {
            return Err(err("transport.cancelled"));
        }
        wire.reasoning = grid_reasoning(&wire.proof, &wire.points, wire.statistical);
        Ok(TransportGridResult { identity: digest(&wire)?, wire, points })
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
        let mut catalog = query.functional.catalog().clone();
        let mut bindings = catalog.bindings.to_vec();
        for binding in &mut bindings {
            let mut snapshots: Vec<&str> = match &input {
                TransportGridData::Exact(data) => data
                    .laws()
                    .iter()
                    .filter(|l| l.regime() == binding.regime)
                    .map(antecedent_expr::ExactDiscreteLaw::snapshot_identity)
                    .collect(),
                TransportGridData::Statistical(data, _) => data
                    .samples
                    .iter()
                    .filter(|s| s.regime == binding.regime)
                    .map(|s| s.snapshot_identity.as_ref())
                    .chain(
                        data.supplied
                            .iter()
                            .filter(|l| l.regime() == binding.regime)
                            .map(antecedent_expr::ExactDiscreteLaw::snapshot_identity),
                    )
                    .collect(),
            };
            snapshots.sort_unstable();
            snapshots.dedup();
            if snapshots.is_empty() {
                continue;
            }
            if snapshots.len() != 1 {
                return Err(err(
                    "snapshot replacement requires one consistent snapshot per regime",
                ));
            }
            binding.snapshot_identity = Arc::from(snapshots[0]);
        }
        catalog.bindings = bindings.into();
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
            &digest(&self.state.template)?,
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
        if wire.version != 1
            || wire.required_features
                != ["checked_transport_proof_v1", "retained_transport_grid_v1"]
            || wire.operations > limits.operations
            || wire.depth > limits.depth
            || wire.at.is_empty()
            || wire.at.len() != wire.points.len()
            || wire.at.len() > limits.operations
        {
            return Err(err("unsupported transport grid features/version/limits"));
        }
        if wire.reasoning != grid_reasoning(&wire.proof, &wire.points, wire.statistical) {
            return Err(err("grid reasoning mismatch"));
        }
        if digest(&wire)? != artifact.manifest.artifact_id {
            return Err(err("transport grid identity mismatch"));
        }
        let mut graph = Admg::empty();
        for node in &wire.nodes {
            graph.add_node(NodeRef::Static(VariableId::from_raw(*node))).map_err(err)?;
        }
        for (a, b) in &wire.directed {
            graph
                .insert_directed(DenseNodeId::from_raw(*a), DenseNodeId::from_raw(*b))
                .map_err(err)?;
        }
        for (a, b) in &wire.bidirected {
            graph
                .insert_bidirected(DenseNodeId::from_raw(*a), DenseNodeId::from_raw(*b))
                .map_err(err)?;
        }
        graph.validate().map_err(err)?;
        let diagram = SelectionDiagram::try_new(
            graph,
            wire.selections.iter().copied().map(VariableId::from_raw).collect::<Vec<_>>(),
        )
        .map_err(err)?;
        let query = ClassicalTransportQuery {
            outcomes: wire.proof.proof.outcomes.iter().copied().map(VariableId::from_raw).collect(),
            treatments: wire
                .proof
                .proof
                .treatments
                .iter()
                .copied()
                .map(VariableId::from_raw)
                .collect(),
            source: Arc::from(wire.proof.proof.source.as_str()),
            target: Arc::from(wire.proof.proof.target.as_str()),
        };
        let proof = wire.proof.check(
            &diagram,
            &query,
            SidLimits { steps: limits.operations, depth: limits.depth },
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
                    if &actual != expected {
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
        let input =
            if wire.statistical { None } else { Some(TransportGridData::Exact(data.clone())) };
        let state = TransportGridState {
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
        let reject = |mutated: GridWire| {
            let forged = TransportGridResult {
                identity: digest(&mutated).unwrap(),
                wire: mutated,
                points: result.points.clone(),
            };
            assert!(
                PreparedStudy::<TransportGridState>::consume(
                    &forged.export().unwrap(),
                    ExactEvaluationLimits::default(),
                    &ctx
                )
                .is_err()
            );
        };
        let mut wire = result.wire.clone();
        wire.required_features.push("future_required_feature".into());
        reject(wire);
        let mut wire = result.wire.clone();
        wire.at[0][0].1 = ValueWire::from_value(&Value::Int64(1));
        reject(wire);
        let mut wire = result.wire.clone();
        wire.points.swap(0, 1);
        reject(wire);
        let mut wire = result.wire.clone();
        wire.laws[0].axes[0].0 = 0;
        reject(wire);
        let mut wire = result.wire.clone();
        wire.proof.proof.target = "substituted".into();
        reject(wire);
        let mut wire = result.wire.clone();
        wire.catalog.regimes[0].population = "wrong_population".into();
        reject(wire);
        let mut wire = result.wire.clone();
        if let GridPointWire::Unavailable(failure) = &mut wire.points[0] {
            failure.kind = "support_failure".into();
        }
        reject(wire);
        let mut tiny = ctx.clone();
        tiny.memory.hard_limit_bytes = Some(1);
        assert!(study.estimate(&tiny).is_err());
    }
}
