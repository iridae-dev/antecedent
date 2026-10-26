//! Selective recalculation over retained, checked adjusted-regression models.
//! The model-support query is distinct from the requested contrast: changing causal roles
//! invalidates identification and fit; selecting another finite contrast reuses coefficients.
//! SPDX-License-Identifier: MIT OR Apache-2.0
use super::recalc_common::{
    Fields, canonical_bits, dag_from_edges, float_bytes, identity, literal,
};
use super::recalc_receipt::{
    Columns, Counter, DecisionValue, LawValue, NonFinite, RecalcOutcome, RecalcRequestCode,
    RecalcRunError, ReceiptRecorder, TargetWeights, UtilitySpec, decide,
};
use crate::graph::Dag;
use antecedent_core::ExecutionContext;
use antecedent_core::VariableId;
use antecedent_core::recalc::{
    Boundary, RecalcCapabilities, RecalcPlan, RequestSupport, ResumeContext, RetargetSupport,
    Stage, StageIdentities, StageIdentity, StageStatus,
};
use antecedent_estimate::adjustment_resume::{
    AdjustedFit, PredictionRefusal, count_adjusted_model_fits,
};
use antecedent_estimate::categorical_treatment::{
    CategoricalTreatmentInput, CategoricalTreatmentSpec, LevelScale, MonotonicityDirection,
};
use antecedent_estimate::vector_treatment::{
    NamedColumn, TreatmentColumn, VectorCovariance, VectorTreatmentInput, VectorTreatmentOptions,
};
use antecedent_identify::backdoor::{CheckedJointAdjustment, check_joint_adjustment};
use antecedent_stats::{GlmFamily, GlmOptions};
use std::sync::Arc;

/// Declared adjusted model and inference configuration.
#[derive(Clone, Debug)]
pub enum AdjustedModel {
    /// Joint ordinary least squares and declared coefficient covariance.
    Linear {
        /// Covariance of the jointly fitted coefficients.
        covariance: VectorCovariance,
    },
    /// Unpenalized GLM with Fisher covariance on its response scale.
    Glm {
        /// Outcome mean/link family.
        family: GlmFamily,
        /// IRLS convergence options.
        options: GlmOptions,
    },
    /// Dummy-coded declared categorical regime.
    Categorical {
        /// Level, reference, and covariance declaration.
        spec: CategoricalTreatmentSpec,
        /// Aligned observed level labels.
        levels: Vec<String>,
    },
}
/// Mean contrast evaluated against one checked model-support query.
#[derive(Clone, Debug)]
pub enum AdjustedContrast {
    /// Joint numeric intervention vectors in treatment role order.
    Numeric {
        /// Active vector.
        active: Vec<f64>,
        /// Control vector.
        control: Vec<f64>,
    },
    /// A pair of declared categorical levels.
    Categorical {
        /// Control level.
        from: String,
        /// Active level.
        to: String,
    },
}
/// Raw inputs, checked causal roles, model and requested mean contrast.
#[derive(Clone, Debug)]
pub struct AdjustedRequest {
    /// Variables in stable id order.
    pub columns: Columns,
    /// Directed graph edges.
    pub edges: Vec<(u32, u32)>,
    /// Joint treatment roles.
    pub treatments: Vec<u32>,
    /// Outcome role.
    pub outcome: u32,
    /// Declared common adjustment roles.
    pub adjustment: Vec<u32>,
    /// Model configuration.
    pub model: AdjustedModel,
    /// Mean contrast within model support.
    pub contrast: AdjustedContrast,
    /// Population weights over unchanged complete rows.
    pub target: Option<TargetWeights>,
    /// Net benefit utility.
    pub utility: UtilitySpec,
}
fn level_scale_tag(scale: LevelScale) -> &'static str {
    match scale {
        LevelScale::Unordered => "unordered",
        LevelScale::Ordered => "ordered",
    }
}
fn model_fields(model: &AdjustedModel) -> Fields {
    match model {
        AdjustedModel::Linear { covariance } => {
            Fields::new().text("linear").text(covariance.as_str())
        }
        AdjustedModel::Glm { family, options } => Fields::new()
            .text("glm")
            .text(crate::analysis::contract_identity::glm_family(*family))
            .wire(&crate::analysis::contract_identity::glm_options(options)),
        AdjustedModel::Categorical { spec, levels } => {
            let mut fields = Fields::new()
                .text("categorical")
                .texts(levels)
                .texts(&spec.declared_levels)
                .text(level_scale_tag(spec.scale))
                .text(&spec.reference)
                .len(spec.min_level_rows)
                .len(spec.pairwise.len());
            for (from, to) in &spec.pairwise {
                fields = fields.text(from).text(to);
            }
            fields
                .text(match spec.monotonicity {
                    None => "none",
                    Some(MonotonicityDirection::NonDecreasing) => "non_decreasing",
                    Some(MonotonicityDirection::NonIncreasing) => "non_increasing",
                })
                .text(spec.covariance.as_str())
        }
    }
}
fn contrast_identity(contrast: &AdjustedContrast) -> StageIdentity {
    match contrast {
        AdjustedContrast::Numeric { active, control } => Fields::new()
            .text("numeric")
            .reals(active)
            .reals(control)
            .identity("adjusted.mean_contrast"),
        AdjustedContrast::Categorical { from, to } => {
            Fields::new().text("categorical").text(from).text(to).identity("adjusted.mean_contrast")
        }
    }
}
impl AdjustedRequest {
    /// Own identities including model-support roles and contrast-specific law inputs.
    #[must_use]
    pub fn identities(&self, ctx: &ExecutionContext) -> StageIdentities {
        let mut ids = StageIdentities::new();
        let mut edges = self.edges.clone();
        edges.sort_unstable();
        edges.dedup();
        ids.set(
            Stage::Graph,
            Fields::new().len(self.columns.len()).pairs(&edges).identity("adjusted.graph"),
        );
        ids.set(
            Stage::Query,
            Fields::new()
                .ids(&self.treatments)
                .word(u64::from(self.outcome))
                .ids(&self.adjustment)
                .text("mean")
                .identity("adjusted.model_support_query"),
        );
        ids.set(
            Stage::Regime,
            literal(
                "adjusted.regime",
                if matches!(self.model, AdjustedModel::Categorical { .. }) {
                    "static.categorical"
                } else {
                    "static.numeric"
                },
            ),
        );
        ids.set(
            Stage::Evidence,
            literal("adjusted.evidence", "complete.observational.no_external"),
        );
        ids.set(Stage::SourcePopulation, literal("adjusted.source", "all_observed"));
        let target = self.target.as_ref().map_or_else(
            || literal("adjusted.target", "all_observed"),
            |t| {
                identity(
                    "adjusted.target",
                    &[
                        float_bytes(&t.weights),
                        Fields::new()
                            .ids(&t.depends_on.iter().map(|id| id.raw()).collect::<Vec<_>>())
                            .into_bytes(),
                    ],
                )
            },
        );
        ids.set(Stage::TargetPopulation, target);
        let mut data = Vec::new();
        for (name, values) in &self.columns {
            data.push(name.as_bytes().to_vec());
            data.push(float_bytes(values));
        }
        ids.set(Stage::DataSnapshot, identity("adjusted.snapshot", &data));
        ids.set(
            Stage::RowDesign,
            literal("adjusted.rows", &format!("complete.rows={}", self.columns.n_rows())),
        );
        // Numeric models declare their finite continuous parametric support. Categorical
        // level/reference changes are model-design changes, never just a selected contrast.
        let grid = match &self.model {
            AdjustedModel::Categorical { spec, .. } => Fields::new()
                .texts(&spec.declared_levels)
                .text(level_scale_tag(spec.scale))
                .text(&spec.reference)
                .identity("adjusted.support"),
            _ => literal("adjusted.support", "parametric.finite_numeric"),
        };
        ids.set(Stage::TreatmentGrid, grid);
        ids.set(
            Stage::LearnerFoldsRng,
            identity(
                "adjusted.model",
                &[
                    model_fields(&self.model).into_bytes(),
                    ctx.rng.master_seed().to_le_bytes().to_vec(),
                ],
            ),
        );
        ids.set(
            Stage::Utility,
            identity(
                "adjusted.utility",
                &[float_bytes(&[self.utility.benefit_per_unit, self.utility.cost])],
            ),
        );
        ids.set(
            Stage::Identification,
            literal("adjusted.identification", "joint_backdoor.declared_set.dseparation.v1"),
        );
        ids.set(
            Stage::ScoreArtifact,
            literal("adjusted.fit", "retained.coefficients.full_covariance.v1"),
        );
        ids.set(Stage::Law, contrast_identity(&self.contrast));
        ids.set(Stage::Decision, literal("adjusted.decision", "net_benefit"));
        ids
    }
    fn support(&self) -> RequestSupport {
        match self.validate() {
            Err(RecalcRunError::Request(
                RecalcRequestCode::ContrastOutOfSupport
                | RecalcRequestCode::InvalidCategoricalContrast,
            )) => RequestSupport::OffGrid { licensed_route: None },
            Err(RecalcRunError::Request(RecalcRequestCode::IncompatibleContrast)) => {
                RequestSupport::Unsupported { licensed_route: None }
            }
            _ => RequestSupport::OnGrid,
        }
    }
    fn graph(&self) -> Result<Dag, RecalcRunError> {
        dag_from_edges(self.columns.len(), &self.edges, RecalcRequestCode::InvalidGraph)
    }
    #[allow(
        clippy::float_cmp,
        reason = "binary and categorical support checks require exact coded levels"
    )]
    fn validate(&self) -> Result<(), RecalcRunError> {
        self.utility.validate()?;
        let n = self.columns.n_rows();
        if !self.columns.is_well_formed(NonFinite::Reject) {
            return Err(RecalcRunError::Request(RecalcRequestCode::InvalidData));
        }
        // Treatments, adjustment and outcome must name distinct existing columns.
        let roles: Vec<u32> = self
            .treatments
            .iter()
            .chain(&self.adjustment)
            .copied()
            .chain(std::iter::once(self.outcome))
            .collect();
        if self.treatments.is_empty() || !self.columns.are_distinct_roles(&roles) {
            return Err(RecalcRunError::Request(RecalcRequestCode::InvalidAdjustmentSet));
        }
        match (&self.model, &self.contrast) {
            (
                AdjustedModel::Categorical { spec, levels },
                AdjustedContrast::Categorical { from, to },
            ) => {
                if self.treatments.len() != 1
                    || levels.len() != n
                    || !spec.declared_levels.contains(from)
                    || !spec.declared_levels.contains(to)
                {
                    return Err(RecalcRunError::Request(
                        RecalcRequestCode::InvalidCategoricalContrast,
                    ));
                }
                let values = self
                    .columns
                    .values(self.treatments[0])
                    .ok_or(RecalcRunError::Request(RecalcRequestCode::InvalidAdjustmentSet))?;
                let mut code_to_label = std::collections::BTreeMap::new();
                let mut label_to_code = std::collections::BTreeMap::new();
                for (value, label) in values.iter().zip(levels) {
                    let code = canonical_bits(*value);
                    if code_to_label.insert(code, label).is_some_and(|old| old != label)
                        || label_to_code.insert(label, code).is_some_and(|old| old != code)
                    {
                        return Err(RecalcRunError::Request(
                            RecalcRequestCode::CategoricalRoleMismatch,
                        ));
                    }
                }
            }
            (
                AdjustedModel::Linear { .. } | AdjustedModel::Glm { .. },
                AdjustedContrast::Numeric { active, control },
            ) => {
                if active.len() != self.treatments.len()
                    || control.len() != self.treatments.len()
                    || active.iter().chain(control).any(|x| !x.is_finite())
                {
                    return Err(RecalcRunError::Request(RecalcRequestCode::InvalidContrast));
                }
                for ((a, b), t) in active.iter().zip(control).zip(&self.treatments) {
                    let values = self
                        .columns
                        .values(*t)
                        .ok_or(RecalcRunError::Request(RecalcRequestCode::InvalidAdjustmentSet))?;
                    let lo = values.iter().copied().fold(f64::INFINITY, f64::min);
                    let hi = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let binary = values.iter().all(|x| *x == 0. || *x == 1.);
                    if [*a, *b].iter().any(|x| *x < lo || *x > hi || binary && *x != 0. && *x != 1.)
                    {
                        return Err(RecalcRunError::Request(
                            RecalcRequestCode::ContrastOutOfSupport,
                        ));
                    }
                }
            }
            _ => return Err(RecalcRunError::Request(RecalcRequestCode::IncompatibleContrast)),
        }
        if let Some(target) = &self.target {
            if !target.are_valid(Some(n))
                || target.depends_on.iter().any(|id| !self.adjustment.contains(&id.raw()))
            {
                return Err(RecalcRunError::Request(RecalcRequestCode::InvalidTargetWeights));
            }
        }
        Ok(())
    }
    /// The column of role `id`; validation has already refused roles outside the table.
    fn role(&self, id: u32) -> Result<&(String, Vec<f64>), RecalcRunError> {
        self.columns
            .column(id)
            .ok_or(RecalcRunError::Request(RecalcRequestCode::InvalidAdjustmentSet))
    }
    fn input(&self, snapshot: &str) -> Result<VectorTreatmentInput, RecalcRunError> {
        let adjustment = self
            .adjustment
            .iter()
            .map(|&id| {
                let (name, values) = self.role(id)?;
                Ok::<_, RecalcRunError>(NamedColumn { name: name.clone(), values: values.clone() })
            })
            .collect::<Result<Vec<_>, RecalcRunError>>()?;
        let names = adjustment.iter().map(|z| z.name.clone()).collect::<Vec<_>>();
        let treatments = self
            .treatments
            .iter()
            .map(|&id| {
                let (name, values) = self.role(id)?;
                Ok::<_, RecalcRunError>(TreatmentColumn {
                    name: name.clone(),
                    values: values.clone(),
                    adjustment_set: names.clone(),
                    row_snapshot: snapshot.to_owned(),
                })
            })
            .collect::<Result<Vec<_>, RecalcRunError>>()?;
        Ok(VectorTreatmentInput {
            outcome: self.role(self.outcome)?.1.clone(),
            row_snapshot: snapshot.to_owned(),
            adjustment,
            treatments,
        })
    }
}
fn identify(request: &AdjustedRequest) -> Result<CheckedJointAdjustment, RecalcRunError> {
    let graph = request.graph()?;
    let treatments: Vec<_> =
        request.treatments.iter().map(|&id| VariableId::from_raw(id)).collect();
    let adjustment: Vec<_> =
        request.adjustment.iter().map(|&id| VariableId::from_raw(id)).collect();
    check_joint_adjustment(&graph, &treatments, VariableId::from_raw(request.outcome), &adjustment)
        .map_err(|_| RecalcRunError::Request(RecalcRequestCode::InvalidAdjustmentSet))?
        .ok_or(RecalcRunError::Request(RecalcRequestCode::AdjustmentNotIdentified))
}
struct Live {
    checked: Arc<CheckedJointAdjustment>,
    fit: Arc<AdjustedFit>,
    law: LawValue,
    decision: DecisionValue,
    prediction_columns: Arc<[String]>,
}
/// Actual live fitted state, rather than caller-provided resumability flags.
pub struct AdjustedSession {
    previous: StageIdentities,
    live: Option<Live>,
    boundary: Boundary,
}
impl Default for AdjustedSession {
    fn default() -> Self {
        Self::new()
    }
}
impl AdjustedSession {
    /// Empty in-process session.
    #[must_use]
    pub fn new() -> Self {
        Self { previous: StageIdentities::new(), live: None, boundary: Boundary::InProcess }
    }
    /// Resume identities for explanation. Data/model artifacts are not supplied by flags.
    #[must_use]
    pub fn resume(previous: StageIdentities, resume: ResumeContext) -> Self {
        let resume =
            ResumeContext { supplied_data: resume.supplied_data, ..ResumeContext::default() };
        Self { previous, live: None, boundary: Boundary::FreshProcess(resume) }
    }
    /// Previous successful declared identities.
    #[must_use]
    pub const fn identities(&self) -> &StageIdentities {
        &self.previous
    }
    /// Whether this session holds the actual fit.
    #[must_use]
    pub const fn is_live(&self) -> bool {
        self.live.is_some()
    }
    /// Actual process boundary.
    #[must_use]
    pub const fn boundary(&self) -> Boundary {
        self.boundary
    }
    /// Plan supported raw-data requests against actual retained state.
    #[must_use]
    pub fn plan(&self, request: &AdjustedRequest, ctx: &ExecutionContext) -> RecalcPlan {
        RecalcPlan::plan(
            &self.previous,
            &request.identities(ctx),
            &RecalcCapabilities {
                retarget: RetargetSupport::Licensed,
                request: request.support(),
                boundary: self.boundary,
            },
        )
    }
    /// Full covariance in intercept, adjustment, treatment design order.
    #[must_use]
    pub fn covariance(&self) -> Option<&[f64]> {
        self.live.as_ref().map(|live| live.fit.covariance())
    }
    /// Feature names bound by the actual fitted design, in prediction order.
    #[must_use]
    pub fn prediction_columns(&self) -> Option<&[String]> {
        self.live.as_ref().map(|live| live.prediction_columns.as_ref())
    }
    /// Predict means with retained fit: adjustment features then numeric/dummy treatments.
    ///
    /// # Errors
    /// Refuses absent fitted state, mismatched/nonfinite features, and unsupported treatments.
    /// Numerical failure is returned without changing the held fit.
    pub fn predict(&self, rows: &[Vec<f64>]) -> Result<Vec<f64>, RecalcRunError> {
        let fit = &self.live.as_ref().ok_or(RecalcRunError::NoLiveState(Stage::ScoreArtifact))?.fit;
        if let Some(refusal) = fit.prediction_refusal(rows) {
            return Err(RecalcRunError::Request(match refusal {
                PredictionRefusal::SchemaMismatch => {
                    RecalcRequestCode::AdjustedPredictionSchemaMismatch
                }
                PredictionRefusal::OutOfSupport => {
                    RecalcRequestCode::AdjustedPredictionOutOfSupport
                }
            }));
        }
        fit.predict(rows).map_err(|e| RecalcRunError::Execution(e.into()))
    }
}
fn recomputed(plan: &RecalcPlan, stage: Stage) -> bool {
    matches!(plan.status(stage), Some(StageStatus::Recomputed { .. }))
}
/// Refuses a fit whose design and Gram matrices would exceed the memory budget.
fn guard_adjusted_memory(
    request: &AdjustedRequest,
    ctx: &ExecutionContext,
) -> Result<(), RecalcRunError> {
    let n = request.columns.n_rows() as u64;
    let k = match &request.model {
        AdjustedModel::Categorical { spec, .. } => spec.declared_levels.len().saturating_sub(1),
        _ => request.treatments.len(),
    };
    let p = (1 + request.adjustment.len() + k) as u64;
    let bytes = n
        .saturating_mul(p)
        .saturating_mul(64)
        .saturating_add(p.saturating_mul(p).saturating_mul(64));
    if ctx.memory_limit_bytes().into_iter().any(|limit| bytes > limit) {
        return Err(RecalcRunError::Request(RecalcRequestCode::MemoryBudgetExceeded));
    }
    Ok(())
}

/// Refits the adjusted model from the request, or reuses the live fit.
fn adjusted_fit(
    request: &AdjustedRequest,
    requested: &StageIdentities,
    plan: &RecalcPlan,
    held: Option<&Live>,
    recorder: &mut ReceiptRecorder,
) -> Result<Arc<AdjustedFit>, RecalcRunError> {
    Ok(if recomputed(plan, Stage::ScoreArtifact) {
        let input = request.input(&requested.own(Stage::DataSnapshot).to_hex())?;
        let (fit, n) = count_adjusted_model_fits(|| match &request.model {
            AdjustedModel::Linear { covariance } => AdjustedFit::linear(
                &input,
                &VectorTreatmentOptions { covariance: *covariance, contrasts: vec![] },
            ),
            AdjustedModel::Glm { family, options } => AdjustedFit::glm(&input, *family, options),
            AdjustedModel::Categorical { spec, levels } => AdjustedFit::categorical(
                &CategoricalTreatmentInput {
                    outcome: input.outcome,
                    row_snapshot: input.row_snapshot,
                    adjustment: input.adjustment,
                    levels: levels.clone(),
                },
                spec,
            ),
        });
        recorder.record(Counter::ModelFit, n);
        Arc::new(fit.map_err(|e| RecalcRunError::Execution(e.into()))?)
    } else {
        held.as_ref().ok_or(RecalcRunError::NoLiveState(Stage::ScoreArtifact))?.fit.clone()
    })
}

/// Names the design columns the fitted model predicts from: adjustment, then treatment or level indicators.
fn prediction_columns(request: &AdjustedRequest) -> Result<Vec<String>, RecalcRunError> {
    let name_of = |id: &u32| request.role(*id).map(|(name, _)| name.clone());
    let mut names = request.adjustment.iter().map(name_of).collect::<Result<Vec<_>, _>>()?;
    match &request.model {
        AdjustedModel::Categorical { spec, .. } => {
            let mut levels = spec.declared_levels.clone();
            if spec.scale == antecedent_estimate::categorical_treatment::LevelScale::Unordered {
                levels.sort();
            }
            names.extend(
                levels
                    .into_iter()
                    .filter(|level| level != &spec.reference)
                    .map(|level| format!("level:{level}")),
            );
        }
        _ => {
            for id in &request.treatments {
                names.push(name_of(id)?);
            }
        }
    }
    Ok(names)
}

/// Execute the shared plan, measuring actual completed model solves inside numeric kernels.
///
/// # Errors
/// Refuses unsupported contrasts, unavailable state, invalid schemas/weights/adjustment,
/// resource exhaustion and numerical failures. The last successful state is retained on failure.
pub fn execute_adjusted_with_receipt(
    session: &mut AdjustedSession,
    request: &AdjustedRequest,
    ctx: &ExecutionContext,
) -> Result<RecalcOutcome, RecalcRunError> {
    if let Err(error) = request.validate() {
        if request.support() != RequestSupport::OnGrid {
            return Err(RecalcRunError::Refused(Box::new(session.plan(request, ctx))));
        }
        return Err(error);
    }
    if ctx.cancellation.is_cancelled() {
        return Err(RecalcRunError::Request(RecalcRequestCode::Cancelled));
    }
    let requested = request.identities(ctx);
    let plan = RecalcPlan::plan(
        &session.previous,
        &requested,
        &RecalcCapabilities {
            retarget: RetargetSupport::Licensed,
            request: request.support(),
            boundary: session.boundary,
        },
    );
    if !plan.is_executable() {
        return Err(RecalcRunError::Refused(Box::new(plan)));
    }
    if recomputed(&plan, Stage::ScoreArtifact) {
        guard_adjusted_memory(request, ctx)?;
    }
    let mut recorder = ReceiptRecorder::charged_to(ctx);
    let held = session.live.as_ref();
    let mut execute = || -> Result<Live, RecalcRunError> {
        let checked = if recomputed(&plan, Stage::Identification) {
            let checked = identify(request)?;
            recorder.record(Counter::Identification, 1);
            Arc::new(checked)
        } else {
            held.as_ref().ok_or(RecalcRunError::NoLiveState(Stage::Identification))?.checked.clone()
        };
        let fit = adjusted_fit(request, &requested, &plan, held, &mut recorder)?;
        let law = if recomputed(&plan, Stage::Law) {
            let (active, control) = match &request.contrast {
                AdjustedContrast::Numeric { active, control } => (active.clone(), control.clone()),
                AdjustedContrast::Categorical { from, to } => fit
                    .categorical_arms(from, to)
                    .map_err(|e| RecalcRunError::Execution(e.into()))?,
            };
            let (ate, std_error) = fit
                .contrast(&active, &control, request.target.as_ref().map(|t| t.weights.as_slice()))
                .map_err(|e| RecalcRunError::Execution(e.into()))?;
            recorder.record(Counter::Reweight, 1);
            LawValue { ate, std_error }
        } else {
            held.as_ref().ok_or(RecalcRunError::NoLiveState(Stage::Law))?.law
        };
        let decision = if recomputed(&plan, Stage::Decision) {
            decide(law, request.utility, &mut recorder)?
        } else {
            held.as_ref().ok_or(RecalcRunError::NoLiveState(Stage::Decision))?.decision
        };
        // Keep the checked graph in live state; it is the proof context bound by identities.
        let _ = checked.graph();
        let prediction_columns = prediction_columns(request)?;
        Ok(Live { checked, fit, law, decision, prediction_columns: Arc::from(prediction_columns) })
    };
    let live = execute()?;
    let receipt = recorder.finish(&plan)?;
    let law = live.law;
    let decision = live.decision;
    session.live = Some(live);
    session.previous = requested;
    session.boundary = Boundary::InProcess;
    Ok(RecalcOutcome { plan, receipt, law, decision })
}
