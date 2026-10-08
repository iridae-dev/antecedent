//! Selective recalculation on the cell-AIPW route, with portable score resume (2.3 C2).
//!
//! [`super::recalc_receipt`] executes the cross-fitted AIPW average effect under a
//! [`RecalcPlan`] and counts the work that ran. This module adds, without touching that
//! executor:
//!
//! - **The cell-AIPW route.** [`CellSession`] and [`execute_cell_with_receipt`] mirror
//!   `RecalcSession` and `execute_with_receipt` over discrete joint binary treatments
//!   (`CellSaturatedAipw`): a first run fits the cell models; a compatible target-weight
//!   retarget reweights the frozen cell scores with zero refits; a utility-only change
//!   recomputes only the decision; a changed outcome, fold seed, graph or adjustment set
//!   refits. The fit instrument is the estimator's own
//!   [`antecedent_estimate::cell_aipw::count_cell_model_fits`]: it counts every multinomial
//!   propensity and per-cell outcome regression that actually completed, so zero refits is
//!   proven by the estimator, not asserted by the wrapper.
//! - **Portable score resume.** [`ScoreResumeSession`] is built only from a
//!   [`FrozenScoreTable`] (cell or cross-fit scores exported from a live session or read from
//!   artifact bytes) and holds no data and no fitted model. It supports exactly one licensed
//!   operation: retarget the frozen scores by row weights over the same row ids and recompute
//!   the law and decision, with zero fits. Anything that would need the data or a fit (a
//!   changed outcome, folds, graph, adjustment, data snapshot or new rows) is refused with the
//!   plan's `Unavailable { missing: Data }` before any work.
//!
//! How resume is planned. A fresh process holding portable scores is planned under
//! `ResumeContext { portable_scores: true, scores_snapshot_bound: true, ..Default::default() }`.
//! The artifact binds the unchanged snapshot identity; raw data remain unavailable. A request
//! is executable only when the planner reuses the score artifact. Every other request is
//! refused with the plan made without the bound-snapshot permission, whose first
//! refusal is the honest `recalc.unavailable_data`.
//!
//! Identification is recomputed in a fresh process (the planner never reuses a derived stage
//! there), and it runs for real: the declared-adjustment descendant screen over the requested
//! graph, counted once.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;
use std::sync::Arc;

use antecedent_core::recalc::{
    Boundary, RecalcCapabilities, RecalcPlan, RequestSupport, ResumeContext, RetargetSupport,
    Stage, StageIdentities, StageIdentity, StageStatus,
};
use antecedent_core::{ExecutionContext, OutcomeFunctional, VariableId};
use antecedent_data::TabularData;
use antecedent_estimate::cell_aipw::{
    cell_minus_control_coefficients, count_cell_model_fits, interaction_coefficients,
};
use antecedent_estimate::{
    CellSaturatedAipw, DirectedAncestry, EstimationError, RetargetResult, ScoreTable,
};
use antecedent_io::frozen_scores_artifact::{
    FrozenScoreParts, FrozenScoreTable, FrozenScoresArtifactError,
};

use crate::error::CausalError;
use crate::graph::{Dag, DenseNodeId};

use super::recalc_receipt::{
    Counter, DecisionValue, LawValue, RecalcOutcome, RecalcRunError, RecalcSession,
    ReceiptRecorder, TargetWeights, UtilitySpec, decide,
};

type Columns = Vec<(String, Vec<f64>)>;

fn to_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

fn f64_bytes(values: &[f64]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect()
}

fn digest(label: &str, parts: &[Vec<u8>]) -> StageIdentity {
    let refs: Vec<&[u8]> = parts.iter().map(Vec::as_slice).collect();
    StageIdentity::of(label, &refs)
}

fn literal(label: &str, text: &str) -> StageIdentity {
    StageIdentity::of(label, &[text.as_bytes()])
}

fn estimation(error: EstimationError) -> RecalcRunError {
    RecalcRunError::Execution(CausalError::from(error))
}

/// The quantity a score table is reduced to.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ScoreQuantity {
    /// Arm 1 minus arm 0 (a binary cross-fit table, or a one-treatment cell table).
    AverageEffect,
    /// Requested cell minus the all-zero control cell.
    CellMinusControl {
        /// The requested cell mask.
        arm: u32,
    },
    /// The 2x2 interaction contrast `mu00 - mu10 - mu01 + mu11`.
    Interaction,
}

/// The cell-AIPW question and the declared inputs around it.
#[derive(Clone, Debug)]
pub struct CellSpec {
    /// Directed edges `(from, to)` over column positions.
    pub edges: Vec<(u32, u32)>,
    /// Jointly intervened binary treatment column positions (one to three).
    pub treatments: Vec<u32>,
    /// Outcome column position.
    pub outcome: u32,
    /// Declared adjustment column positions (screened against the graph).
    pub adjustment: Vec<u32>,
    /// Estimator configuration; the run's master seed is applied as its fold seed.
    pub estimator: CellSaturatedAipw,
    /// The reported contrast.
    pub quantity: ScoreQuantity,
    /// Target population weights; `None` is the observed population.
    pub target: Option<TargetWeights>,
    /// Net-benefit rule.
    pub utility: UtilitySpec,
}

/// One cell-AIPW request over raw columns.
#[derive(Clone, Debug)]
pub struct CellRequest {
    /// Named data columns; the variable id of a column is its position.
    pub columns: Columns,
    /// The question and declared inputs.
    pub spec: CellSpec,
}

fn graph_identity(n_variables: usize, edges: &[(u32, u32)]) -> StageIdentity {
    let mut edges = edges.to_vec();
    edges.sort_unstable();
    edges.dedup();
    let edge_bytes: Vec<u8> =
        edges.iter().flat_map(|(a, b)| [a.to_le_bytes(), b.to_le_bytes()]).flatten().collect();
    digest("graph", &[to_u64(n_variables).to_le_bytes().to_vec(), edge_bytes])
}

fn target_identity(target: Option<&TargetWeights>) -> StageIdentity {
    target.map_or_else(
        || literal("target_population", "all_observed"),
        |t| {
            let parents: Vec<u8> =
                t.depends_on.iter().flat_map(|v| v.raw().to_le_bytes()).collect();
            digest("target_population", &[f64_bytes(&t.weights), parents])
        },
    )
}

fn utility_identity(utility: UtilitySpec) -> StageIdentity {
    digest("utility", &[f64_bytes(&[utility.benefit_per_unit, utility.cost])])
}

fn snapshot_identity(columns: &Columns) -> StageIdentity {
    let mut parts = Vec::new();
    for (name, values) in columns {
        parts.push(name.as_bytes().to_vec());
        parts.push(f64_bytes(values));
    }
    digest("data_snapshot", &parts)
}

impl CellRequest {
    fn rows(&self) -> usize {
        self.columns.first().map_or(0, |(_, v)| v.len())
    }

    /// Declared own-input identity of every stage of this request under `ctx`.
    #[must_use]
    pub fn identities(&self, ctx: &ExecutionContext) -> StageIdentities {
        let spec = &self.spec;
        let mut ids = StageIdentities::new();
        ids.set(Stage::Graph, graph_identity(self.columns.len(), &spec.edges));
        ids.set(
            Stage::Query,
            literal(
                "query",
                &format!(
                    "cell:{:?}:{}:{:?}:{:?}",
                    spec.treatments, spec.outcome, spec.adjustment, spec.quantity
                ),
            ),
        );
        ids.set(Stage::Regime, literal("regime", "static.binary_joint_cells"));
        ids.set(Stage::Evidence, literal("evidence", "observational.no_external"));
        ids.set(Stage::SourcePopulation, literal("source_population", "all_observed"));
        ids.set(Stage::TargetPopulation, target_identity(spec.target.as_ref()));
        ids.set(Stage::DataSnapshot, snapshot_identity(&self.columns));
        ids.set(
            Stage::RowDesign,
            literal("row_design", &format!("complete_case.rows={}", self.rows())),
        );
        ids.set(
            Stage::TreatmentGrid,
            literal("treatment_grid", &format!("binary.cells.k={}", spec.treatments.len())),
        );
        ids.set(
            Stage::LearnerFoldsRng,
            digest(
                "learner_folds_rng",
                &[
                    format!("{:?}", spec.estimator).into_bytes(),
                    to_u64(spec.estimator.folds).to_le_bytes().to_vec(),
                    ctx.rng.master_seed().to_le_bytes().to_vec(),
                ],
            ),
        );
        ids.set(Stage::Utility, utility_identity(spec.utility));
        ids.set(
            Stage::Identification,
            literal("identification", "backdoor.declared_adjustment.descendant_screen"),
        );
        ids.set(Stage::ScoreArtifact, literal("score_artifact", "cell_aipw.scores"));
        ids.set(Stage::Law, literal("law", "weighted_mean_of_scores"));
        ids.set(Stage::Decision, literal("decision", "net_benefit"));
        ids
    }

    fn data(&self) -> Result<TabularData, RecalcRunError> {
        let borrowed: Vec<(&str, &[f64])> =
            self.columns.iter().map(|(n, v)| (n.as_str(), v.as_slice())).collect();
        TabularData::from_f64_columns(borrowed)
            .map_err(|_| RecalcRunError::Request("recalc.invalid_data"))
    }
}

// -- graph screen and law -------------------------------------------------------------------

fn dag_of(n_variables: u32, edges: &[(u32, u32)]) -> Result<Dag, RecalcRunError> {
    let mut edges = edges.to_vec();
    edges.sort_unstable();
    edges.dedup();
    let mut graph = Dag::with_variables(n_variables);
    for (from, to) in edges {
        graph
            .insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to))
            .map_err(|_| RecalcRunError::Request("recalc.invalid_graph"))?;
    }
    Ok(graph)
}

fn descendants(graph: &Dag, sources: &[u32]) -> BTreeSet<u32> {
    let mut seen: BTreeSet<u32> = BTreeSet::new();
    let mut stack: Vec<u32> = sources.to_vec();
    while let Some(node) = stack.pop() {
        for child in graph.children(DenseNodeId::from_raw(node)) {
            if seen.insert(child.raw()) {
                stack.push(child.raw());
            }
        }
    }
    seen
}

/// The declared-adjustment screen: every id is a graph node, the adjustment set holds no
/// treatment, no outcome and no descendant of a treatment.
fn screen_adjustment(
    graph: &Dag,
    n_variables: u32,
    treatments: &[u32],
    outcome: Option<u32>,
    adjustment: &[u32],
) -> Result<(), RecalcRunError> {
    let invalid = RecalcRunError::Request("recalc.invalid_adjustment_set");
    let in_range = |v: &u32| *v < n_variables;
    if treatments.is_empty()
        || !treatments.iter().all(in_range)
        || !adjustment.iter().all(in_range)
        || outcome.is_some_and(|y| y >= n_variables || treatments.contains(&y))
    {
        return Err(invalid);
    }
    let downstream = descendants(graph, treatments);
    let bad = adjustment
        .iter()
        .any(|z| treatments.contains(z) || Some(*z) == outcome || downstream.contains(z));
    if bad { Err(invalid) } else { Ok(()) }
}

fn quantity_value(
    table: &ScoreTable,
    result: &RetargetResult,
    quantity: ScoreQuantity,
) -> Result<(f64, f64), RecalcRunError> {
    let coefficients = match quantity {
        ScoreQuantity::AverageEffect => {
            return result
                .contrast
                .map(|c| (c.value, c.se))
                .ok_or(RecalcRunError::Request("recalc.contrast_unavailable"));
        }
        ScoreQuantity::CellMinusControl { arm } => {
            cell_minus_control_coefficients(table, arm).map_err(estimation)?
        }
        ScoreQuantity::Interaction => interaction_coefficients(table).map_err(estimation)?,
    };
    let contrast = table.linear_contrast(&result.summary, &coefficients).map_err(estimation)?;
    Ok((contrast.value, contrast.se))
}

/// Reweight the frozen scores into the law of `quantity`; counts one reweight.
fn score_law(
    table: &ScoreTable,
    target: Option<&TargetWeights>,
    graph: &Dag,
    quantity: ScoreQuantity,
    recorder: &mut ReceiptRecorder,
) -> Result<LawValue, RecalcRunError> {
    let ones;
    let (weights, depends_on): (&[f64], &[VariableId]) = if let Some(t) = target {
        (t.weights.as_slice(), t.depends_on.as_slice())
    } else {
        ones = vec![1.0; table.n_rows];
        (ones.as_slice(), &[])
    };
    let ancestry: &dyn DirectedAncestry = graph;
    let (result, overlap_failed) =
        antecedent_estimate::retarget(table, weights, depends_on, Some(ancestry), None, None)
            .map_err(estimation)?;
    if overlap_failed {
        return Err(RecalcRunError::Request("recalc.weighted_overlap_failed"));
    }
    recorder.record(Counter::Reweight, 1);
    let (ate, std_error) = quantity_value(table, &result, quantity)?;
    Ok(LawValue { ate, std_error })
}

fn is_recomputed(plan: &RecalcPlan, stage: Stage) -> bool {
    matches!(plan.status(stage), Some(StageStatus::Recomputed { .. }))
}

fn is_reused(plan: &RecalcPlan, stage: Stage) -> bool {
    matches!(plan.status(stage), Some(StageStatus::Reused { .. }))
}

// -- freezing scores ------------------------------------------------------------------------

fn freeze(
    table: &ScoreTable,
    declared: &StageIdentities,
    input_rows: u64,
    retarget: RetargetSupport,
) -> Result<FrozenScoreTable, RecalcRunError> {
    if retarget != RetargetSupport::Licensed {
        return Err(RecalcRunError::Request("recalc.retarget_not_licensed"));
    }
    let fit_identity = declared
        .effective(RetargetSupport::Licensed)
        .get(&Stage::ScoreArtifact)
        .copied()
        .ok_or(RecalcRunError::Request("recalc.score_stage_undeclared"))?;
    FrozenScoreTable::seal(FrozenScoreParts {
        estimator: table.nuisance_provenance.to_string(),
        fit_identity,
        declared: declared.clone(),
        input_rows,
        table: table.to_wire(),
    })
    .map_err(|_| RecalcRunError::Request("recalc.frozen_scores_rejected"))
}

/// Export the frozen cross-fit AIPW scores of a live [`RecalcSession`] as a portable
/// artifact. `input_rows` is the row count of the data the session last ran on, and
/// `retarget` the license the session was run under (only a licensed route is exportable).
///
/// # Errors
///
/// [`RecalcRunError::NoLiveState`] without a live score table; `Request` for an unlicensed
/// route or scores the artifact format rejects.
pub fn freeze_crossfit_scores(
    session: &RecalcSession,
    input_rows: u64,
    retarget: RetargetSupport,
) -> Result<FrozenScoreTable, RecalcRunError> {
    let table = session.score_table().ok_or(RecalcRunError::NoLiveState(Stage::ScoreArtifact))?;
    freeze(table, session.identities(), input_rows, retarget)
}

// -- in-process cell session ----------------------------------------------------------------

struct CellLive {
    table: Arc<ScoreTable>,
    law: LawValue,
    decision: DecisionValue,
}

/// What a caller holds between cell-route calls: the frozen cell scores, the last law and
/// decision, and the last identities.
pub struct CellSession {
    live: Option<CellLive>,
    previous: StageIdentities,
    input_rows: u64,
    retarget: RetargetSupport,
}

impl Default for CellSession {
    fn default() -> Self {
        Self::new()
    }
}

impl CellSession {
    /// An empty session with the retarget route declared licensed.
    #[must_use]
    pub fn new() -> Self {
        Self {
            live: None,
            previous: StageIdentities::new(),
            input_rows: 0,
            retarget: RetargetSupport::Licensed,
        }
    }

    /// Declare the estimator's retarget license for later plans.
    pub fn set_retarget_support(&mut self, retarget: RetargetSupport) {
        self.retarget = retarget;
    }

    /// Identities of the last successful run.
    #[must_use]
    pub const fn identities(&self) -> &StageIdentities {
        &self.previous
    }

    /// Whether the session holds live cell scores.
    #[must_use]
    pub const fn is_live(&self) -> bool {
        self.live.is_some()
    }

    /// The frozen cell score table, for independent checks.
    #[must_use]
    pub fn score_table(&self) -> Option<&ScoreTable> {
        self.live.as_ref().map(|live| live.table.as_ref())
    }

    /// The plan `request` would run under, without running it.
    #[must_use]
    pub fn plan(&self, request: &CellRequest, ctx: &ExecutionContext) -> RecalcPlan {
        let caps = RecalcCapabilities {
            retarget: self.retarget,
            request: RequestSupport::OnGrid,
            boundary: Boundary::InProcess,
        };
        RecalcPlan::plan(&self.previous, &request.identities(ctx), &caps)
    }

    /// Export the live cell scores as a portable artifact.
    ///
    /// # Errors
    ///
    /// [`RecalcRunError::NoLiveState`] without live scores; `Request` for an unlicensed route
    /// or scores the artifact format rejects.
    pub fn export_scores(&self) -> Result<FrozenScoreTable, RecalcRunError> {
        let live = self.live.as_ref().ok_or(RecalcRunError::NoLiveState(Stage::ScoreArtifact))?;
        freeze(&live.table, &self.previous, self.input_rows, self.retarget)
    }
}

fn fit_cells(
    request: &CellRequest,
    recorder: &mut ReceiptRecorder,
    ctx: &ExecutionContext,
) -> Result<ScoreTable, RecalcRunError> {
    let data = request.data()?;
    let spec = &request.spec;
    let estimator = spec.estimator.clone().with_fold_seed(ctx.rng.master_seed());
    let treatments: Vec<VariableId> =
        spec.treatments.iter().map(|&v| VariableId::from_raw(v)).collect();
    let adjustment: Vec<VariableId> =
        spec.adjustment.iter().map(|&v| VariableId::from_raw(v)).collect();
    recorder.record(Counter::ScoreComputation, 1);
    let (fitted, fits) = count_cell_model_fits(|| {
        estimator.fit_scores(
            &data,
            &treatments,
            VariableId::from_raw(spec.outcome),
            &adjustment,
            &OutcomeFunctional::Mean,
            None,
        )
    });
    recorder.record(Counter::FoldFit, fits);
    fitted.map_err(estimation)
}

fn cell_n_variables(request: &CellRequest) -> Result<u32, RecalcRunError> {
    u32::try_from(request.columns.len())
        .map_err(|_| RecalcRunError::Request("recalc.invalid_graph"))
}

fn identify_cells(
    request: &CellRequest,
    recorder: &mut ReceiptRecorder,
) -> Result<(), RecalcRunError> {
    let n = cell_n_variables(request)?;
    let spec = &request.spec;
    if spec.treatments.len() > antecedent_estimate::MAX_JOINT_BINARY {
        return Err(RecalcRunError::Request("recalc.invalid_adjustment_set"));
    }
    let graph = dag_of(n, &spec.edges)?;
    screen_adjustment(&graph, n, &spec.treatments, Some(spec.outcome), &spec.adjustment)?;
    recorder.record(Counter::Identification, 1);
    Ok(())
}

fn run_cell_stages(
    live: Option<CellLive>,
    plan: &RecalcPlan,
    request: &CellRequest,
    recorder: &mut ReceiptRecorder,
    ctx: &ExecutionContext,
) -> Result<CellLive, RecalcRunError> {
    if is_recomputed(plan, Stage::Identification) {
        identify_cells(request, recorder)?;
    }
    let (table, held) = if is_recomputed(plan, Stage::ScoreArtifact) {
        (Arc::new(fit_cells(request, recorder, ctx)?), None)
    } else {
        let live = live.ok_or(RecalcRunError::NoLiveState(Stage::ScoreArtifact))?;
        (live.table, Some((live.law, live.decision)))
    };
    let law = if is_recomputed(plan, Stage::Law) {
        let graph = dag_of(cell_n_variables(request)?, &request.spec.edges)?;
        score_law(&table, request.spec.target.as_ref(), &graph, request.spec.quantity, recorder)?
    } else {
        held.map(|(law, _)| law).ok_or(RecalcRunError::NoLiveState(Stage::Law))?
    };
    let decision = if is_recomputed(plan, Stage::Decision) {
        decide(law, request.spec.utility, recorder)?
    } else {
        held.map(|(_, decision)| decision).ok_or(RecalcRunError::NoLiveState(Stage::Decision))?
    };
    Ok(CellLive { table, law, decision })
}

/// The registered reason code of a malformed cell or resume request (`invalid_argument`): the
/// `recalc.*` request details this module emits all carry it.
#[must_use]
pub fn cell_request_reason_code() -> &'static str {
    antecedent_core::reason_code!("invalid_argument")
}

/// Plan `request` against the session, run only what the plan says must run, count the work,
/// and return the verified receipt with the law and decision.
///
/// The counts are: identifications (the adjustment screen), cell-model fits (read from the
/// estimator's own instrument), score computations, reweights and decisions. A refused plan
/// returns [`RecalcRunError::Refused`] before any work and leaves the session unchanged; a
/// failure while running clears the session's identities.
///
/// # Errors
///
/// [`RecalcRunError`] for a refused plan, a malformed request, a failed estimator call, or a
/// receipt that contradicts its plan.
pub fn execute_cell_with_receipt(
    session: &mut CellSession,
    request: &CellRequest,
    ctx: &ExecutionContext,
) -> Result<RecalcOutcome, RecalcRunError> {
    request.spec.utility.validate()?;
    let plan = session.plan(request, ctx);
    if !plan.is_executable() {
        return Err(RecalcRunError::Refused(Box::new(plan)));
    }
    if session.live.is_none() {
        let unbacked = plan
            .entries()
            .iter()
            .find(|e| !e.stage.is_input() && matches!(e.status, StageStatus::Reused { .. }));
        if let Some(entry) = unbacked {
            return Err(RecalcRunError::NoLiveState(entry.stage));
        }
    }
    let mut recorder = ReceiptRecorder::new();
    let live = session.live.take();
    let live = match run_cell_stages(live, &plan, request, &mut recorder, ctx) {
        Ok(live) => live,
        Err(error) => {
            session.previous = StageIdentities::new();
            return Err(error);
        }
    };
    let receipt = match recorder.finish(&plan) {
        Ok(receipt) => receipt,
        Err(error) => {
            session.previous = StageIdentities::new();
            return Err(error.into());
        }
    };
    let (law, decision) = (live.law, live.decision);
    session.previous = request.identities(ctx);
    session.input_rows = to_u64(request.rows());
    session.live = Some(live);
    Ok(RecalcOutcome { plan, receipt, law, decision })
}

// -- portable score resume ------------------------------------------------------------------

/// A retarget request against portable frozen scores. It carries no data.
#[derive(Clone, Debug)]
pub struct ScoreResumeRequest {
    /// Number of variables (columns) of the original data; part of the graph identity.
    pub n_variables: u32,
    /// Directed edges `(from, to)` over column positions.
    pub edges: Vec<(u32, u32)>,
    /// Target weights, one per row of the frozen scores, in row order; `None` is the observed
    /// population.
    pub target: Option<TargetWeights>,
    /// The original row ids the weights are indexed by. When given they must equal the frozen
    /// scores' row ids exactly; when `None` only the row count is checked.
    pub target_row_ids: Option<Vec<u32>>,
    /// Net-benefit rule.
    pub utility: UtilitySpec,
    /// The reported contrast.
    pub quantity: ScoreQuantity,
    /// Declared inputs the caller says differ from the artifact's workflow (a changed outcome
    /// query, fold seed, data snapshot or row design). Only input stages are accepted. Any
    /// such change needs the data or a fit and is refused with `Unavailable`.
    pub changed_inputs: Vec<(Stage, StageIdentity)>,
}

/// A fresh-process session built only from a [`FrozenScoreTable`].
///
/// It holds the frozen scores and the producing workflow's identities. It holds no data, no
/// fitted model and no prepared study, and it can run exactly one operation: retarget the
/// scores and recompute the law and decision.
pub struct ScoreResumeSession {
    table: Arc<ScoreTable>,
    artifact_identity: String,
    anchor: StageIdentities,
    previous: StageIdentities,
    live: Option<(LawValue, DecisionValue)>,
}

impl ScoreResumeSession {
    /// A resumable session from a validated artifact (cell or cross-fit scores).
    ///
    /// # Errors
    ///
    /// [`FrozenScoresArtifactError`] when the artifact's table or declarations cannot be read.
    pub fn resume_from_scores(
        artifact: &FrozenScoreTable,
    ) -> Result<Self, FrozenScoresArtifactError> {
        let anchor = artifact.declared()?;
        Ok(Self {
            table: Arc::new(artifact.score_table()?),
            artifact_identity: artifact.identity().to_owned(),
            previous: anchor.clone(),
            anchor,
            live: None,
        })
    }

    /// A resumable session from artifact bytes, optionally checked against an identity the
    /// caller retained independently.
    ///
    /// # Errors
    ///
    /// [`FrozenScoresArtifactError`] for corrupt, oversized, malformed or resealed bytes.
    pub fn resume_from_score_bytes(
        bytes: &[u8],
        expected_identity: Option<&str>,
    ) -> Result<Self, FrozenScoresArtifactError> {
        Self::resume_from_scores(&FrozenScoreTable::from_bytes(bytes, expected_identity)?)
    }

    /// The frozen scores the session retargets.
    #[must_use]
    pub fn score_table(&self) -> &ScoreTable {
        &self.table
    }

    /// Identity (hex) of the artifact the session was built from.
    #[must_use]
    pub fn artifact_identity(&self) -> &str {
        &self.artifact_identity
    }

    /// Identities of the last successful run (the artifact's workflow before any run).
    #[must_use]
    pub const fn identities(&self) -> &StageIdentities {
        &self.previous
    }

    /// Whether a resumed run has completed (the session then holds its law and decision).
    #[must_use]
    pub const fn has_run(&self) -> bool {
        self.live.is_some()
    }

    /// The requested workflow: the artifact's, with the graph, target and utility replaced by
    /// the request's and the declared input changes applied.
    fn requested(&self, request: &ScoreResumeRequest) -> Result<StageIdentities, RecalcRunError> {
        let mut ids = self.anchor.clone();
        ids.set(Stage::Graph, graph_identity(request.n_variables as usize, &request.edges));
        ids.set(Stage::TargetPopulation, target_identity(request.target.as_ref()));
        ids.set(Stage::Utility, utility_identity(request.utility));
        let mut changed = BTreeSet::new();
        for (stage, own) in &request.changed_inputs {
            if !stage.is_input()
                || matches!(stage, Stage::Graph | Stage::TargetPopulation | Stage::Utility)
                || !changed.insert(*stage)
            {
                return Err(RecalcRunError::Request("recalc.invalid_changed_input"));
            }
            ids.set(*stage, *own);
        }
        // Scores support multiple contrasts. The selected reduction is a law input,
        // independent of the original fitted query, and must not reuse another contrast.
        ids.set(
            Stage::Law,
            StageIdentity::of(
                "resumed_score_law",
                &[
                    self.anchor.own(Stage::Law).as_bytes(),
                    format!("{:?}", request.quantity).as_bytes(),
                ],
            ),
        );
        Ok(ids)
    }

    /// The plan the request runs under: a fresh process with portable scores whose snapshot
    /// identity is bound to the artifact. Before the first run the boundary is a fresh
    /// process; afterwards the session holds its law and decision and plans in process.
    #[must_use]
    fn licensed_plan(&self, requested: &StageIdentities) -> RecalcPlan {
        let boundary = if self.live.is_some() {
            Boundary::InProcess
        } else {
            Boundary::FreshProcess(ResumeContext {
                portable_scores: true,
                scores_snapshot_bound: true,
                ..ResumeContext::default()
            })
        };
        let caps = RecalcCapabilities {
            retarget: RetargetSupport::Licensed,
            request: RequestSupport::OnGrid,
            boundary,
        };
        RecalcPlan::plan(&self.previous, requested, &caps)
    }

    /// The plan with the data honestly not supplied: its first refusal is
    /// `Unavailable { missing: Data }`.
    fn refusal_plan(&self, requested: &StageIdentities) -> RecalcPlan {
        let caps = RecalcCapabilities {
            retarget: RetargetSupport::Licensed,
            request: RequestSupport::OnGrid,
            boundary: Boundary::FreshProcess(ResumeContext {
                portable_scores: true,
                ..ResumeContext::default()
            }),
        };
        RecalcPlan::plan(&self.previous, requested, &caps)
    }

    /// The plan `request` would run under, or the refusal plan when it needs data or a fit.
    ///
    /// # Errors
    ///
    /// [`RecalcRunError::Refused`] with the honest plan, or `Request` for a malformed change.
    pub fn plan(&self, request: &ScoreResumeRequest) -> Result<RecalcPlan, RecalcRunError> {
        let requested = self.requested(request)?;
        let plan = self.licensed_plan(&requested);
        if plan.is_executable() && is_reused(&plan, Stage::ScoreArtifact) {
            Ok(plan)
        } else {
            Err(RecalcRunError::Refused(Box::new(self.refusal_plan(&requested))))
        }
    }

    fn check_rows(&self, request: &ScoreResumeRequest) -> Result<(), RecalcRunError> {
        if let Some(ids) = &request.target_row_ids {
            if ids.as_slice() != self.table.row_index.as_ref() {
                return Err(RecalcRunError::Request("recalc.row_ids_mismatch"));
            }
        }
        if let Some(target) = &request.target {
            if target.weights.len() != self.table.n_rows {
                return Err(RecalcRunError::Request("recalc.row_count_mismatch"));
            }
        }
        Ok(())
    }

    fn run(
        &self,
        plan: &RecalcPlan,
        request: &ScoreResumeRequest,
        recorder: &mut ReceiptRecorder,
    ) -> Result<(LawValue, DecisionValue), RecalcRunError> {
        let graph = dag_of(request.n_variables, &request.edges)?;
        if is_recomputed(plan, Stage::Identification) {
            let mut treatments: Vec<u32> = vec![self.table.treatment.raw()];
            treatments.extend(self.table.intervened.iter().map(|v| v.raw()));
            treatments.sort_unstable();
            treatments.dedup();
            let adjustment: Vec<u32> = self.table.adjustment_set.iter().map(|v| v.raw()).collect();
            screen_adjustment(&graph, request.n_variables, &treatments, None, &adjustment)?;
            recorder.record(Counter::Identification, 1);
        }
        let held = self.live;
        let law = if is_recomputed(plan, Stage::Law) {
            score_law(&self.table, request.target.as_ref(), &graph, request.quantity, recorder)?
        } else {
            held.map(|(law, _)| law).ok_or(RecalcRunError::NoLiveState(Stage::Law))?
        };
        let decision = if is_recomputed(plan, Stage::Decision) {
            decide(law, request.utility, recorder)?
        } else {
            held.map(|(_, d)| d).ok_or(RecalcRunError::NoLiveState(Stage::Decision))?
        };
        Ok((law, decision))
    }
}

/// Retarget the frozen scores of a resumed session by row weights and recompute the law and
/// decision, with zero fits.
///
/// The plan is made first. The only executable request is one whose planner reuses the score
/// artifact; the receipt then records the score-artifact stage as reused with no work, no fit
/// and no score computation. A request that would need the data or a fit (changed outcome,
/// folds, graph, adjustment, snapshot or row design) is refused before any work with the
/// plan's `Unavailable { missing: Data }` and leaves the session unchanged. Weights over other
/// row ids, or of another length, are refused before any work with a `recalc.*` detail.
///
/// # Errors
///
/// [`RecalcRunError::Refused`], `Request`, `Execution` (illegal `depends_on`, weight shape)
/// or `Receipt`. After a failure that occurred while running, the session returns to the
/// artifact's state.
pub fn execute_resumed_retarget(
    session: &mut ScoreResumeSession,
    request: &ScoreResumeRequest,
) -> Result<RecalcOutcome, RecalcRunError> {
    request.utility.validate()?;
    let requested = session.requested(request)?;
    let plan = session.licensed_plan(&requested);
    if !plan.is_executable() || !is_reused(&plan, Stage::ScoreArtifact) {
        return Err(RecalcRunError::Refused(Box::new(session.refusal_plan(&requested))));
    }
    session.check_rows(request)?;
    let mut recorder = ReceiptRecorder::new();
    let outcome = session.run(&plan, request, &mut recorder).and_then(|(law, decision)| {
        let receipt = recorder.finish(&plan)?;
        Ok((law, decision, receipt))
    });
    match outcome {
        Ok((law, decision, receipt)) => {
            session.previous = requested;
            session.live = Some((law, decision));
            Ok(RecalcOutcome { plan, receipt, law, decision })
        }
        Err(error) => {
            session.previous = session.anchor.clone();
            session.live = None;
            Err(error)
        }
    }
}
