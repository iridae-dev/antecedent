//! Shared-plan point execution of checked binary two-step whole-history functionals.
//! SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(feature = "calibration-internal")]
#[path = "recalc_temporal_studentized.rs"]
mod studentized;

use super::{
    recalc_receipt::{
        Counter, DecisionValue, LawValue, RecalcOutcome, RecalcRunError, ReceiptRecorder,
        UtilitySpec, decide,
    },
    temporal_extensions::{IntervalEstimand, temporal_dependent_interval},
};
use antecedent_core::recalc::{
    Boundary, RecalcCapabilities, RecalcPlan, RequestSupport, ResumeContext, RetargetSupport,
    Stage, StageIdentities, StageIdentity, StageStatus,
};
use antecedent_core::{
    DependenceGroup, DistributionAvailability, EvidenceCatalog, EvidenceKind, EvidenceRegime,
    ExecutionContext, RegimeBinding, RegimeId, RegimeKind, SamplingDesign, Value, VariableDomain,
    VariableId, reason_code,
};
use antecedent_estimate::{
    EstimationError,
    temporal_dependent_interval::{
        DependentIntervalConfig, SequenceHistory, TemporalUnitPanel, UnitHistories,
    },
    temporal_history_fit::{FittedSequenceHistory, fit_binary_history_joint},
    temporal_initial_state::{InitialStateLaw, InitialStatePopulation, InitialStateSpec},
};
use antecedent_expr::execution_counts::{StaticWorkCounts, count_static_work};
use antecedent_expr::{
    Assignment, ExactDiscreteLaw, ExactDistribution, ExactEvaluationLimits, ExactEvaluationPlan,
    ExactTransportData, LawTolerance,
};
use antecedent_graph::{Admg, DenseNodeId, SelectionDiagram};
use antecedent_identify::sid::{
    SidLimits,
    scenarios::ScenarioCoordinate,
    temporal_initial_shift::{
        CheckedInitialShiftSource, INITIAL_SHIFT_SOURCE_RULE, identify_initial_shift_source,
    },
    temporal_sequence::{TemporalSequenceSpec, TemporalSlots},
};
pub use antecedent_io::temporal_recalc_artifact::{
    TEMPORAL_RECALC_INFERENCE, TemporalFunctionalWire as TemporalFunctional, TemporalHistoryWire,
    TemporalRecalcArtifactWire, TemporalRecalcRequestWire as TemporalRequest, TemporalUnitWire,
};
use antecedent_io::{
    IoError,
    mz_transport_artifact::MzTransportPointWire,
    temporal_recalc_artifact::{TemporalInitialShiftReportWire, TemporalSourceIdProofWire},
};
use std::{collections::BTreeMap, sync::Arc};

const SEARCH: SidLimits = SidLimits { steps: 100_000, depth: 256 };
const NAMES: [&str; 5] = ["s0", "a1", "l2", "a2", "y"];
fn id(label: &str, text: &str) -> StageIdentity {
    StageIdentity::of(label, &[text.as_bytes()])
}
fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}
/// Typed planner, checked proof, support or source-artifact refusal.
#[derive(Debug)]
pub enum TemporalRunError {
    /// Shared stage planner, cancellation or malformed request.
    Recalc(RecalcRunError),
    /// Existing checked temporal identification/artifact failure.
    Io(IoError),
    /// Existing whole-history mechanism/support failure.
    Estimation(EstimationError),
}
impl From<RecalcRunError> for TemporalRunError {
    fn from(e: RecalcRunError) -> Self {
        Self::Recalc(e)
    }
}
impl From<IoError> for TemporalRunError {
    fn from(e: IoError) -> Self {
        Self::Io(e)
    }
}
impl From<EstimationError> for TemporalRunError {
    fn from(e: EstimationError) -> Self {
        Self::Estimation(e)
    }
}
impl std::fmt::Display for TemporalRunError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Recalc(e) => write!(f, "{e}"),
            Self::Io(e) => write!(f, "{e}"),
            Self::Estimation(e) => write!(f, "{e}"),
        }
    }
}
impl std::error::Error for TemporalRunError {}
fn refusal(detail: &str) -> TemporalRunError {
    IoError::Refused { code: reason_code!("route_not_supported"), message: detail.into() }.into()
}
fn validate(r: &TemporalRequest, ctx: &ExecutionContext) -> Result<(), TemporalRunError> {
    UtilitySpec { benefit_per_unit: r.benefit_per_unit, cost: r.cost }.validate()?;
    if ctx.cancellation.is_cancelled() {
        return Err(RecalcRunError::Request("recalc.cancelled").into());
    }
    if r.horizon != 2 {
        return Err(refusal("temporal_transport.horizon: only two-step histories are supported"));
    }
    if r.lag_alignment != [0, 1, 2, 2, 2] {
        return Err(refusal("temporal_refresh.lag_alignment_changed"));
    }
    if r.selection_targets != [0] || !r.bidirected.is_empty() {
        return Err(refusal(
            "temporal_initial_shift.source_scope: requires a DAG with selection only at root S0",
        ));
    }
    if r.period.0 >= r.period.1
        || r.edges.len() > 25
        || r.bidirected.len() > 25
        || r.snapshot_id.len() > 4096
        || r.initial_state_id.len() > 4096
        || r.units.is_empty()
        || r.units.len() > 8192
        || r.units.iter().map(|u| u.histories.len()).sum::<usize>() > 131_072
        || r.units.iter().flat_map(|u| &u.histories).any(|h| {
            h.time_id < r.period.0 || h.time_id.checked_add(2).is_none_or(|end| end >= r.period.1)
        })
        || r.functional.sequences().iter().flatten().any(|a| *a > 1)
    {
        return Err(RecalcRunError::Request("recalc.invalid_temporal_window").into());
    }
    Ok(())
}
fn initial_law(r: &TemporalRequest) -> Result<InitialStateLaw, TemporalRunError> {
    Ok(InitialStateLaw::new(
        InitialStatePopulation::Target,
        r.initial_state_id.as_str(),
        vec![(0, r.initial_state[0]), (1, r.initial_state[1])],
    )?)
}
fn panel(r: &TemporalRequest) -> Result<TemporalUnitPanel, TemporalRunError> {
    Ok(TemporalUnitPanel::new(
        &r.snapshot_id,
        Some(
            r.units
                .iter()
                .map(|u| UnitHistories {
                    unit_id: u.unit_id,
                    histories: u
                        .histories
                        .iter()
                        .map(|h| SequenceHistory {
                            time_id: h.time_id,
                            s0: h.s0,
                            a1: h.a1,
                            l2: h.l2,
                            a2: h.a2,
                            y: h.y,
                        })
                        .collect(),
                })
                .collect(),
        ),
    )?)
}
fn spec(r: &TemporalRequest) -> Result<TemporalSequenceSpec, TemporalRunError> {
    let mut graph = Admg::with_variables(5);
    for &(a, b) in &r.edges {
        graph
            .insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b))
            .map_err(|_| RecalcRunError::Request("recalc.invalid_graph"))?;
    }
    for &(a, b) in &r.bidirected {
        graph
            .insert_bidirected(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b))
            .map_err(|_| RecalcRunError::Request("recalc.invalid_graph"))?;
    }
    let diagram = SelectionDiagram::try_new(
        graph,
        r.selection_targets.iter().copied().map(v).collect::<Vec<_>>(),
    )
    .map_err(|_| RecalcRunError::Request("recalc.invalid_graph"))?;
    let coordinates = NAMES
        .iter()
        .zip(0..5)
        .map(|(name, raw)| ScenarioCoordinate {
            variable: v(raw),
            name: Arc::from(*name),
            domain: VariableDomain::Binary,
            unit: None,
        })
        .collect::<Vec<_>>();
    TemporalSequenceSpec::try_new(
        r.horizon,
        TemporalSlots {
            baseline: vec![v(0)],
            covariates: [vec![], vec![v(2)]],
            actions: [v(1), v(3)],
            outcome: v(4),
        },
        diagram,
        coordinates,
    )
    .map_err(|e| IoError::Refused { code: e.code, message: e.to_string() }.into())
}
fn catalog(r: &TemporalRequest) -> Result<EvidenceCatalog, TemporalRunError> {
    let source = EvidenceRegime::try_new(
        RegimeId::from_raw(0),
        RegimeKind::Observational,
        EvidenceKind::Available,
        [],
        [],
        (0..5).map(v).collect::<Vec<_>>(),
        "source",
        DistributionAvailability::Joint,
    )
    .map_err(|_| RecalcRunError::Request("recalc.invalid_evidence"))?;
    let target = EvidenceRegime::try_new(
        RegimeId::from_raw(1),
        RegimeKind::Observational,
        EvidenceKind::Available,
        [],
        [],
        [v(0)],
        "target",
        DistributionAvailability::Joint,
    )
    .map_err(|_| RecalcRunError::Request("recalc.invalid_evidence"))?;
    let bindings = [
        (0, &r.snapshot_id, SamplingDesign::Clustered),
        (1, &r.initial_state_id, SamplingDesign::Unknown),
    ]
    .into_iter()
    .map(|(raw, snapshot, sampling)| RegimeBinding {
        dataset_identity: None,
        regime: RegimeId::from_raw(raw),
        snapshot_identity: Arc::from(snapshot.as_str()),
        schema_names: Arc::from([]),
        sampling,
        weights: None,
        dependence: if raw == 0 {
            DependenceGroup::LinkedUnits
        } else {
            DependenceGroup::UnknownDependence
        },
    })
    .collect::<Vec<_>>();
    EvidenceCatalog::try_new([], vec![source, target], bindings, None)
        .map_err(|_| RecalcRunError::Request("recalc.invalid_evidence").into())
}
fn source_identity(r: &TemporalRequest) -> StageIdentity {
    id("temporal.source", &format!("{}:{:?}:{:?}", r.snapshot_id, r.period, r.units))
}
/// Stage identities bind full history/unit ownership, window, lag alignment and initial law.
#[must_use]
pub fn temporal_request_identities(r: &TemporalRequest, ctx: &ExecutionContext) -> StageIdentities {
    let mut ids = StageIdentities::new();
    let mut edges = r.edges.clone();
    edges.sort_unstable();
    edges.dedup();
    let mut bidirected = r.bidirected.clone();
    bidirected.sort_unstable();
    bidirected.dedup();
    ids.set(
        Stage::Graph,
        id(
            "temporal.graph",
            &format!(
                "{edges:?}:{bidirected:?}:{}:{:?}:{:?}",
                r.horizon, r.lag_alignment, r.selection_targets
            ),
        ),
    );
    ids.set(Stage::Query, id("temporal.source_query", "joint(S0,Y)|do(A1,A2)"));
    ids.set(Stage::Regime, id("temporal.regime", "whole.sequence.source0.target1"));
    ids.set(
        Stage::Evidence,
        id("temporal.evidence", "source.joint0_4.target.initial0.clustered.linked"),
    );
    ids.set(Stage::SourcePopulation, id("temporal.population", "source"));
    ids.set(
        Stage::TargetPopulation,
        id(
            "temporal.target",
            &format!("{}:{:?}", r.initial_state_id, r.initial_state.map(f64::to_bits)),
        ),
    );
    ids.set(Stage::DataSnapshot, source_identity(r));
    ids.set(
        Stage::RowDesign,
        id(
            "temporal.rows",
            &format!(
                "{:?}:{:?}",
                r.period,
                r.units
                    .iter()
                    .map(|u| (u.unit_id, u.histories.iter().map(|h| h.time_id).collect::<Vec<_>>()))
                    .collect::<Vec<_>>()
            ),
        ),
    );
    ids.set(Stage::TreatmentGrid, id("temporal.actions", &format!("{:?}", r.functional)));
    ids.set(
        Stage::LearnerFoldsRng,
        id("temporal.fit", &format!("empirical.joint.whole.unit:{}", ctx.rng.master_seed())),
    );
    ids.set(
        Stage::Utility,
        id("temporal.utility", &format!("{:?}", [r.benefit_per_unit.to_bits(), r.cost.to_bits()])),
    );
    for (stage, label) in [
        (Stage::Identification, "checked.whole.sequence"),
        (Stage::ScoreArtifact, "retained.history.mechanisms"),
        (Stage::Law, "empirical.point.only"),
        (Stage::Decision, "named.functional.utility"),
    ] {
        ids.set(stage, id("temporal.stage", label));
    }
    ids
}
fn recomputed(plan: &RecalcPlan, stage: Stage) -> bool {
    matches!(plan.status(stage), Some(StageStatus::Recomputed { .. }))
}
fn record(recorder: &mut ReceiptRecorder, work: StaticWorkCounts) {
    for (counter, n) in [
        (Counter::FactorBuild, work.factor_builds),
        (Counter::ProgramCompilation, work.program_compilations),
        (Counter::ProviderBinding, work.provider_bindings),
        (Counter::FactorEvaluation, work.factor_evaluations),
        (Counter::Integration, work.integrations),
        (Counter::ProviderCall, work.provider_calls),
        (Counter::LawSummary, work.law_summaries),
    ] {
        recorder.record(counter, n);
    }
}
/// Named point answer with the executing shared plan and measured work receipt.
pub struct TemporalOutcome {
    /// Common stages/work/utility. Its scalar law is the declared named functional below.
    pub recalc: RecalcOutcome,
    /// Mean response or effect declaration, never inferred from a scalar placeholder.
    pub functional: TemporalFunctional,
    /// Whole-sequence means in request order (one response, or active/control).
    pub means: Vec<f64>,
}
struct Live {
    source: StageIdentity,
    panel: Arc<TemporalUnitPanel>,
    joint: Arc<ExactDiscreteLaw>,
    fits: BTreeMap<[u32; 2], Arc<FittedSequenceHistory>>,
    prepared: BTreeMap<[u32; 2], Arc<PreparedInitialShift>>,
    request: TemporalRequest,
    reports: Vec<TemporalInitialShiftReportWire>,
    means: Vec<f64>,
    law: LawValue,
    decision: DecisionValue,
    seed: u64,
}
/// Actual immutable checked mechanisms retained between calls.
pub struct TemporalSession {
    previous: StageIdentities,
    boundary: Boundary,
    live: Option<Live>,
}
impl Default for TemporalSession {
    fn default() -> Self {
        Self::new()
    }
}
impl TemporalSession {
    /// Empty executing in-process session.
    #[must_use]
    pub fn new() -> Self {
        Self { previous: StageIdentities::new(), boundary: Boundary::InProcess, live: None }
    }
    /// Portable flags do not supply mechanisms; only supplied raw histories can refit.
    #[must_use]
    pub fn resume(previous: StageIdentities, resume: ResumeContext) -> Self {
        Self {
            previous,
            boundary: Boundary::FreshProcess(ResumeContext {
                supplied_data: resume.supplied_data,
                ..ResumeContext::default()
            }),
            live: None,
        }
    }
    /// Last successful inputs.
    #[must_use]
    pub const fn identities(&self) -> &StageIdentities {
        &self.previous
    }
    /// Actual fit availability.
    #[must_use]
    pub const fn is_live(&self) -> bool {
        self.live.is_some()
    }
    /// Actual current execution boundary for receipt sealing.
    #[must_use]
    pub const fn boundary(&self) -> Boundary {
        self.boundary
    }
    /// Plan from actual retained state; target-law updates rebind checked support.
    #[must_use]
    pub fn plan(&self, r: &TemporalRequest, ctx: &ExecutionContext) -> RecalcPlan {
        RecalcPlan::plan(
            &self.previous,
            &temporal_request_identities(r, ctx),
            &RecalcCapabilities {
                retarget: RetargetSupport::NotDeclared,
                request: RequestSupport::OnGrid,
                boundary: self.boundary,
            },
        )
    }
    /// Borrow the actual retained empirical source factor.
    #[must_use]
    pub fn source_factor(&self) -> Option<&ExactDiscreteLaw> {
        self.live.as_ref().map(|l| l.joint.as_ref())
    }
    /// Borrow the sequence-specific mechanism fit.
    #[must_use]
    pub fn sequence_fit(&self, seq: [u32; 2]) -> Option<&FittedSequenceHistory> {
        self.live.as_ref()?.fits.get(&seq).map(Arc::as_ref)
    }
    /// Export full unit/history lineage and actual checked reports under empirical point semantics.
    /// # Errors
    /// No actual fitted state or artifact encoding failure.
    pub fn export_result(&self) -> Result<Vec<u8>, TemporalRunError> {
        let l = self.live.as_ref().ok_or(RecalcRunError::NoLiveState(Stage::ScoreArtifact))?;
        Ok(TemporalRecalcArtifactWire::seal(
            l.request.clone(),
            l.seed,
            l.reports.clone(),
            l.law.ate,
        )?
        .export()?)
    }
    /// Validate the selected whole-unit interval request and preserve the public frozen gate.
    /// # Errors
    /// Missing state, invalid unit/config/support, or always temporal_interval.route_frozen.
    pub fn dependent_interval(
        &self,
        config: &DependentIntervalConfig,
    ) -> Result<std::convert::Infallible, TemporalRunError> {
        let l = self.live.as_ref().ok_or(RecalcRunError::NoLiveState(Stage::ScoreArtifact))?;
        let sequence = l.request.functional.sequences()[0];
        temporal_dependent_interval(
            l.panel.snapshot_id(),
            Some(l.panel.units().to_vec()),
            sequence,
            IntervalEstimand::Marginalized(InitialStateSpec::Law(initial_law(&l.request)?)),
            config,
        )
        .map_err(TemporalRunError::from)
    }

    /// Closed whole-unit interval candidate for internal acceptance/calibration.
    /// Both arms of an effect share each whole-unit draw. This produces an
    /// unmeasured candidate and does not open the public dependent interval gate.
    /// # Errors
    /// Missing actual state, changed producing context, invalid configuration,
    /// cancellation or unsupported resampled histories.
    #[cfg(feature = "calibration-internal")]
    #[doc(hidden)]
    pub fn candidate_interval_internal(
        &self,
        config: &DependentIntervalConfig,
        ctx: &ExecutionContext,
    ) -> Result<antecedent_estimate::temporal_dependent_interval::DependentInterval, TemporalRunError>
    {
        let live = self.live.as_ref().ok_or(RecalcRunError::NoLiveState(Stage::ScoreArtifact))?;
        if live.seed != ctx.rng.master_seed() {
            return Err(RecalcRunError::Request("recalc.temporal_artifact_seed_mismatch").into());
        }
        if config.method
            == antecedent_estimate::temporal_dependent_interval::IntervalMethod::Studentized
        {
            return Ok(antecedent_estimate::temporal_dependent_interval::dependent_unit_interval(
                &live.panel,
                &studentized::CheckedStudentizedUnitDraw(CheckedUnitDraw { live, ctx }),
                config,
                ctx,
            )?);
        }
        Ok(antecedent_estimate::temporal_dependent_interval::dependent_unit_interval(
            &live.panel,
            &CheckedUnitDraw { live, ctx },
            config,
            ctx,
        )?)
    }
}
#[cfg(feature = "calibration-internal")]
struct CheckedUnitDraw<'a> {
    live: &'a Live,
    ctx: &'a ExecutionContext,
}
#[cfg(feature = "calibration-internal")]
impl antecedent_estimate::temporal_dependent_interval::TemporalEstimator for CheckedUnitDraw<'_> {
    fn label(&self) -> &'static str {
        match self.live.request.functional {
            TemporalFunctional::Response { .. } => "checked_initial_shift_sequence_response",
            TemporalFunctional::Effect { .. } => "checked_initial_shift_paired_sequence_effect",
        }
    }
    fn estimate(&self, units: &[&UnitHistories]) -> Result<f64, EstimationError> {
        checked_unit_draw(self.live, units, self.ctx).map_err(|error| match error {
            TemporalRunError::Estimation(error) => error,
            error => EstimationError::refused(
                reason_code!("transport_support_failure"),
                error.to_string(),
            ),
        })
    }
}
#[cfg(feature = "calibration-internal")]
fn checked_unit_draw(
    live: &Live,
    units: &[&UnitHistories],
    ctx: &ExecutionContext,
) -> Result<f64, TemporalRunError> {
    let joint = Arc::new(antecedent_estimate::temporal_history_fit::fit_binary_history_units(
        live.panel.snapshot_id(),
        units,
    )?);
    let mut sequences = live.request.functional.sequences();
    sequences.sort_unstable();
    sequences.dedup();
    let fits = FittedSequenceHistory::fit_many(units, &sequences)
        .into_iter()
        .map(|fit| (fit.sequence(), Arc::new(fit)))
        .collect();
    let proof = &live
        .prepared
        .values()
        .next()
        .ok_or(RecalcRunError::NoLiveState(Stage::Identification))?
        .proof;
    let laws = data(&joint)?;
    let prepared = sequences
        .into_iter()
        .map(|seq| Ok((seq, Arc::new(prepare_shift(Arc::clone(proof), laws.clone(), seq, ctx)?))))
        .collect::<Result<BTreeMap<_, _>, TemporalRunError>>()?;
    let rebuilt = Rebuilt { panel: Arc::clone(&live.panel), joint, fits, prepared };
    Ok(evaluate(&live.request, &rebuilt, ctx)?.2.ate)
}
fn data(joint: &ExactDiscreteLaw) -> Result<ExactTransportData, TemporalRunError> {
    ExactTransportData::try_new(vec![joint.clone()], 4096)
        .map_err(|_| RecalcRunError::Request("recalc.invalid_temporal_provider").into())
}
struct PreparedInitialShift {
    proof: Arc<CheckedInitialShiftSource>,
    program: ExactEvaluationPlan,
}
fn prepare_shift(
    proof: Arc<CheckedInitialShiftSource>,
    laws: ExactTransportData,
    sequence: [u32; 2],
    ctx: &ExecutionContext,
) -> Result<PreparedInitialShift, TemporalRunError> {
    let request = Assignment::from_pairs([
        (v(1), Value::f64(f64::from(sequence[0]))),
        (v(3), Value::f64(f64::from(sequence[1]))),
    ]);
    let program = ExactEvaluationPlan::compile(
        proof.arena(),
        proof.root(),
        laws,
        Arc::from([v(0), v(4)]),
        request,
        ExactEvaluationLimits::default(),
        LawTolerance::default(),
        ctx,
    )
    .map_err(|error| antecedent_estimate::transport::refuse_eval(&error))?;
    Ok(PreparedInitialShift { proof, program })
}
fn fit_guard(r: &TemporalRequest, ctx: &ExecutionContext) -> Result<(), TemporalRunError> {
    let n = r
        .units
        .iter()
        .map(|u| u64::try_from(u.histories.len()).unwrap_or(u64::MAX))
        .fold(0_u64, u64::saturating_add);
    let bytes = n.saturating_mul(256).saturating_add(4096 * 256);
    if ctx
        .memory
        .hard_limit_bytes
        .into_iter()
        .chain(ctx.memory.soft_limit_bytes)
        .any(|lim| bytes > lim)
    {
        return Err(RecalcRunError::Request("recalc.memory_budget_exceeded").into());
    }
    Ok(())
}
fn source_proof(
    r: &TemporalRequest,
    ctx: &ExecutionContext,
    recorder: &mut ReceiptRecorder,
) -> Result<Arc<CheckedInitialShiftSource>, TemporalRunError> {
    let proof =
        identify_initial_shift_source(&spec(r)?, &catalog(r)?, SEARCH, ctx).map_err(|error| {
            match error {
                antecedent_identify::IdentificationError::UnsupportedQuery { .. } => {
                    IoError::Refused {
                        code: reason_code!("route_not_supported"),
                        message: error.to_string(),
                    }
                }
                antecedent_identify::IdentificationError::MissingEvidence { .. } => {
                    IoError::Refused {
                        code: reason_code!("transport_missing_evidence"),
                        message: error.to_string(),
                    }
                }
                other => IoError::from(other),
            }
        })?;
    recorder.record(Counter::Identification, 1);
    Ok(Arc::new(proof))
}
struct Rebuilt {
    panel: Arc<TemporalUnitPanel>,
    joint: Arc<ExactDiscreteLaw>,
    fits: BTreeMap<[u32; 2], Arc<FittedSequenceHistory>>,
    prepared: BTreeMap<[u32; 2], Arc<PreparedInitialShift>>,
}
fn rebuild(
    held: Option<&Live>,
    r: &TemporalRequest,
    ctx: &ExecutionContext,
    plan: &RecalcPlan,
    recorder: &mut ReceiptRecorder,
) -> Result<Rebuilt, TemporalRunError> {
    fit_guard(r, ctx)?;
    let proof = if recomputed(plan, Stage::Identification) {
        source_proof(r, ctx, recorder)?
    } else {
        Arc::clone(
            &held
                .ok_or(RecalcRunError::NoLiveState(Stage::Identification))?
                .prepared
                .values()
                .next()
                .ok_or(RecalcRunError::NoLiveState(Stage::Identification))?
                .proof,
        )
    };
    let same = held.filter(|l| l.source == source_identity(r));
    let (built, work) = count_static_work(|| -> Result<Rebuilt, TemporalRunError> {
        let panel = match same {
            Some(l) => Arc::clone(&l.panel),
            None => Arc::new(panel(r)?),
        };
        let joint = match same {
            Some(l) => Arc::clone(&l.joint),
            None => Arc::new(fit_binary_history_joint(&panel)?),
        };
        let mut fits = same.map_or_else(BTreeMap::new, |l| l.fits.clone());
        let units = panel.units().iter().collect::<Vec<_>>();
        let mut missing = r.functional.sequences();
        missing.sort_unstable();
        missing.dedup();
        missing.retain(|seq| !fits.contains_key(seq));
        for fitted in FittedSequenceHistory::fit_many(&units, &missing) {
            fits.insert(fitted.sequence(), Arc::new(fitted));
        }
        let laws = data(&joint)?;
        let mut prepared = BTreeMap::new();
        for seq in r.functional.sequences() {
            if prepared.contains_key(&seq) {
                continue;
            }
            let p = if let Some(existing) = same
                .and_then(|l| l.prepared.get(&seq))
                .filter(|_| !recomputed(plan, Stage::Identification))
            {
                Arc::clone(existing)
            } else {
                Arc::new(prepare_shift(Arc::clone(&proof), laws.clone(), seq, ctx)?)
            };
            prepared.insert(seq, p);
        }
        Ok(Rebuilt { panel, joint, fits, prepared })
    });
    record(recorder, work);
    built
}
fn source_conditional_means(joint: &ExactDistribution) -> Result<[f64; 2], TemporalRunError> {
    let mut masses = [0.; 2];
    let mut totals = [0.; 2];
    for (atom, &probability) in joint.atoms.iter().zip(joint.probabilities.iter()) {
        let [Value::Float64(state), Value::Float64(outcome)] = atom.as_ref() else {
            return Err(refusal("temporal_initial_shift.invalid_source_joint"));
        };
        let index = match state.to_bits() {
            0 => 0,
            bits if bits == 1_f64.to_bits() => 1,
            _ => return Err(refusal("temporal_initial_shift.invalid_source_joint")),
        };
        masses[index] += probability;
        totals[index] += probability * outcome;
    }
    if masses.iter().any(|p| *p <= 0.) {
        return Err(EstimationError::refused(
            reason_code!("transport_support_failure"),
            "initial_state.support_gap: every source S0 state must have positive intervention mass",
        )
        .into());
    }
    Ok([totals[0] / masses[0], totals[1] / masses[1]])
}
fn evaluate(
    r: &TemporalRequest,
    rebuilt: &Rebuilt,
    ctx: &ExecutionContext,
) -> Result<(Vec<f64>, Vec<TemporalInitialShiftReportWire>, LawValue), TemporalRunError> {
    let law = initial_law(r)?;
    let mut means = Vec::new();
    let mut reports = Vec::new();
    for seq in r.functional.sequences() {
        let fitted =
            rebuilt.fits.get(&seq).ok_or(RecalcRunError::NoLiveState(Stage::ScoreArtifact))?;
        let selected_mean =
            fitted.contributions(&law)?.iter().map(|c| c.mass * c.response).sum::<f64>();
        let prepared =
            rebuilt.prepared.get(&seq).ok_or(RecalcRunError::NoLiveState(Stage::Identification))?;
        let joint = prepared
            .program
            .evaluate(ctx)
            .map_err(|error| antecedent_estimate::transport::refuse_eval(&error))?;
        let conditional_means = source_conditional_means(&joint)?;
        let mean =
            r.initial_state[0] * conditional_means[0] + r.initial_state[1] * conditional_means[1];
        if (selected_mean - mean).abs() > 1e-10 {
            return Err(refusal("temporal_recalc.selected_history_binding_mismatch"));
        }
        means.push(mean);
        reports.push(TemporalInitialShiftReportWire {
            rule: INITIAL_SHIFT_SOURCE_RULE.into(),
            sequence: seq,
            source_proof: TemporalSourceIdProofWire {
                source_graph: antecedent_io::admg_to_wire(prepared.proof.graph())?,
                method: "general_id".into(),
                population: "source".into(),
                source_expression: antecedent_io::expr_wire::expr_arena_to_wire(
                    &prepared.proof.result().arena,
                )?,
                source_root: prepared.proof.result().estimands[0].functional.raw(),
                executable_expression: antecedent_io::expr_wire::expr_arena_to_wire(
                    prepared.proof.arena(),
                )?,
                executable_root: prepared.proof.root().raw(),
                derivation: prepared
                    .proof
                    .result()
                    .derivation
                    .steps
                    .iter()
                    .map(|s| (s.rule.to_string(), s.detail.to_string()))
                    .collect(),
            },
            source_joint: MzTransportPointWire::from_distribution(&joint),
            conditional_means,
            mean,
        });
    }
    let value = match r.functional {
        TemporalFunctional::Response { .. } => means[0],
        TemporalFunctional::Effect { .. } => means[0] - means[1],
    };
    Ok((means, reports, LawValue { ate: value, std_error: f64::NAN }))
}
/// Execute actual checked whole-sequence proofs and retained history mechanisms transactionally.
/// # Errors
/// Invalid data/window, unsupported history/horizon, failed proof, cancellation/memory,
/// unavailable raw state at a fresh boundary, or inconsistent executing receipt.
pub fn execute_temporal_with_receipt(
    session: &mut TemporalSession,
    r: &TemporalRequest,
    ctx: &ExecutionContext,
) -> Result<TemporalOutcome, TemporalRunError> {
    validate(r, ctx)?;
    initial_law(r)?;
    let requested = temporal_request_identities(r, ctx);
    let plan = session.plan(r, ctx);
    if !plan.is_executable() {
        return Err(RecalcRunError::Refused(Box::new(plan)).into());
    }
    let mut recorder = ReceiptRecorder::new();
    let held = session.live.as_ref();
    let rebuilt = if recomputed(&plan, Stage::ScoreArtifact) {
        rebuild(held, r, ctx, &plan, &mut recorder)?
    } else {
        let l = held.ok_or(RecalcRunError::NoLiveState(Stage::ScoreArtifact))?;
        Rebuilt {
            panel: Arc::clone(&l.panel),
            joint: Arc::clone(&l.joint),
            fits: l.fits.clone(),
            prepared: l.prepared.clone(),
        }
    };
    let (means, reports, law) = if recomputed(&plan, Stage::Law) {
        let (answer, work) = count_static_work(|| evaluate(r, &rebuilt, ctx));
        record(&mut recorder, work);
        answer?
    } else {
        let l = held.ok_or(RecalcRunError::NoLiveState(Stage::Law))?;
        (l.means.clone(), l.reports.clone(), l.law)
    };
    let decision = if recomputed(&plan, Stage::Decision) {
        decide(
            law,
            UtilitySpec { benefit_per_unit: r.benefit_per_unit, cost: r.cost },
            &mut recorder,
        )?
    } else {
        held.ok_or(RecalcRunError::NoLiveState(Stage::Decision))?.decision
    };
    let receipt = recorder.finish(&plan).map_err(RecalcRunError::from)?;
    session.live = Some(Live {
        source: source_identity(r),
        panel: rebuilt.panel,
        joint: rebuilt.joint,
        fits: rebuilt.fits,
        prepared: rebuilt.prepared,
        request: r.clone(),
        reports,
        means: means.clone(),
        law,
        decision,
        seed: ctx.rng.master_seed(),
    });
    session.previous = requested;
    session.boundary = Boundary::InProcess;
    Ok(TemporalOutcome {
        recalc: RecalcOutcome { plan, receipt, law, decision },
        functional: r.functional.clone(),
        means,
    })
}
/// Independently decode raw unit histories and rebuild checked proofs/mechanisms/points.
/// The returned receipt counts this fresh execution and grants no interval claim.
/// # Errors
/// Malformed artifact, producing-seed mismatch, checked/support failure or changed report.
pub fn consume_temporal_recalc_artifact(
    bytes: &[u8],
    ctx: &ExecutionContext,
) -> Result<(TemporalSession, TemporalOutcome), TemporalRunError> {
    let wire = TemporalRecalcArtifactWire::decode(bytes)?;
    if wire.seed != ctx.rng.master_seed() {
        return Err(RecalcRunError::Request("recalc.temporal_artifact_seed_mismatch").into());
    }
    let mut session = TemporalSession::new();
    let outcome = execute_temporal_with_receipt(&mut session, &wire.request, ctx)?;
    let reports = &session.live.as_ref().ok_or(RecalcRunError::NoLiveState(Stage::Law))?.reports;
    if !wire.matches_reports(reports, outcome.recalc.law.ate)? {
        return Err(RecalcRunError::Request("recalc.temporal_artifact_mismatch").into());
    }
    Ok((session, outcome))
}
