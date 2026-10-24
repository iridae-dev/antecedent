//! Certified search derived from the same retained response evaluated by ordinary search.
// SPDX-License-Identifier: MIT OR Apache-2.0
use super::*;
use antecedent_core::{ValueExpr, VariableId};
use antecedent_design::decision_contract::{ProgramObjective, UtilityExpr};
use antecedent_design::program_search::certified::{
    CertificateOptions, CertifiedControl, CertifiedRestriction, GlobalProgramCertificate,
    ObjectiveInterval, PreparedCertifiedSearch,
};
use engine::expression::{
    CompiledPanelExpression, PanelExpressionControl, compile_panel_expression,
};

/// Certified scalar optimization bound to an actual retained panel compiler.
#[derive(Clone, Debug)]
pub struct JointPanelCertifiedSearch {
    plan: PreparedCertifiedSearch,
    evaluator: FrozenPanelEvaluator,
    expression: Arc<CompiledPanelExpression>,
    initial: Vec<f64>,
    offset: f64,
    scale: f64,
    bindings: Vec<PanelExpressionControl>,
    time_weights: Vec<f64>,
}
/// Numerical certificate and independently usable scientific source for its actual incumbent.
#[derive(Clone, Debug)]
pub struct JointPanelCertifiedSearchResult {
    /// Bounds for the exact compiler-derived scalar objective and declared domain.
    pub certificate: GlobalProgramCertificate,
    /// Checked response source for the returned controls, retaining its scientific scope.
    pub best_source: ScientificDecisionSource,
    /// Exact fitted source/parameter law from which the objective was derived.
    pub fitted_dependency: SemanticDigest,
    /// Original compiled scientific question.
    pub parent_target: SemanticDigest,
}
/// Strongest executable numerical search route admitted by the actual response.
#[derive(Clone, Debug)]
pub enum JointPanelProgramSearchResult {
    /// A compiler-derived objective with a valid retained global bound.
    Certified(JointPanelCertifiedSearchResult),
    /// Executed bounded search when this expression/objective has no admitted
    /// certificate evaluator. No global gap is manufactured.
    BestFound {
        /// Actual completed search and its original scientific source.
        result: super::JointPanelContinuousSearchResult,
        /// Concrete certificate capability that was unavailable.
        certificate_refusal: String,
    },
}
fn supports_interval(expression: &ValueExpr) -> bool {
    match expression {
        ValueExpr::Const(_) | ValueExpr::Var(_) => true,
        ValueExpr::Neg(v) => supports_interval(v),
        ValueExpr::Arith { op, lhs, rhs } => {
            matches!(
                op,
                antecedent_core::ArithOp::Add
                    | antecedent_core::ArithOp::Sub
                    | antecedent_core::ArithOp::Mul
                    | antecedent_core::ArithOp::Div
                    | antecedent_core::ArithOp::Min
                    | antecedent_core::ArithOp::Max
            ) && supports_interval(lhs)
                && supports_interval(rhs)
        }
        _ => false,
    }
}

fn affine(expr: &UtilityExpr) -> Result<(f64, f64), JointPanelError> {
    let result = match expr {
        UtilityExpr::Const(c) => (*c, 0.0),
        UtilityExpr::Input(0) => (0.0, 1.0),
        UtilityExpr::Neg(value) => {
            let (a, b) = affine(value)?;
            (-a, -b)
        }
        UtilityExpr::Add(lhs, rhs) | UtilityExpr::Sub(lhs, rhs) => {
            let (a, b) = affine(lhs)?;
            let (c, d) = affine(rhs)?;
            let sign = if matches!(expr, UtilityExpr::Sub(..)) { -1.0 } else { 1.0 };
            (a + sign * c, b + sign * d)
        }
        UtilityExpr::Mul(lhs, rhs) => {
            let (a, b) = affine(lhs)?;
            let (c, d) = affine(rhs)?;
            if b != 0.0 && d != 0.0 {
                return Err(JointPanelError::Unsupported(
                "mean-only certificate requires affine utility; use the aligned-law search for nonlinear utility".into()));
            }
            (a * c, a * d + b * c)
        }
        _ => {
            return Err(JointPanelError::Unsupported(
                "certificate objective requires an affine utility of the actual scalar response"
                    .into(),
            ));
        }
    };
    if !result.0.is_finite() || !result.1.is_finite() {
        return Err(JointPanelError::compile("affine objective coefficients overflow"));
    }
    Ok(result)
}

impl JointPanelResult {
    /// Execute a certificate where the compiler admits one, otherwise execute
    /// the existing best-found search with its actual refusal and stopping facts.
    /// Cancellation, invalid scientific scope and resources remain errors.
    /// # Errors
    /// Invalid original search, source authorization, cancellation or resources.
    pub fn search_program(
        &self,
        request: JointPanelContinuousSearchRequest,
        options: CertificateOptions,
        ctx: &ExecutionContext,
    ) -> Result<JointPanelProgramSearchResult, JointPanelError> {
        if !options.absolute_gap.is_finite() || options.absolute_gap < 0.0 {
            return Err(JointPanelError::compile("invalid certificate gap tolerance"));
        }
        let fallback = self.compile_continuous_search(request.clone(), ctx)?;
        match self.compile_certified_search(request, ctx) {
            Ok(plan) => Ok(JointPanelProgramSearchResult::Certified(plan.execute(options, ctx)?)),
            Err(JointPanelError::Unsupported(reason)) => {
                Ok(JointPanelProgramSearchResult::BestFound {
                    result: fallback.execute(ctx)?,
                    certificate_refusal: reason,
                })
            }
            Err(error) => Err(error),
        }
    }
    /// Derive a global-bounded objective from this actual fitted response. There
    /// is no caller-supplied expression or unchecked monotonicity declaration.
    /// Scheduled controls, remaining schedules, history, population, temporal path,
    /// lag/saturation/interaction parameters and source authority stay bound.
    ///
    /// # Errors
    /// Invalid original search, unsupported mean/law demand, numerical expression,
    /// resources, cancellation, or unsupported causal extrapolation.
    pub fn compile_certified_search(
        &self,
        request: JointPanelContinuousSearchRequest,
        ctx: &ExecutionContext,
    ) -> Result<JointPanelCertifiedSearch, JointPanelError> {
        source::check_cancelled(ctx)?;
        if request.uncertainty != JointPanelSearchUncertainty::PointMeans
            || !request.contract.constraints.is_empty()
        {
            return Err(JointPanelError::Unsupported(
                "scalar mean certificate cannot replace a risk or uncertainty demand; use aligned-law program search".into()));
        }
        let ProgramObjective::ExpectedUtility(utility) = &request.contract.objective else {
            return Err(JointPanelError::Unsupported(
                "this certificate requires expected scalar utility".into(),
            ));
        };
        let (offset, scale) = affine(utility)?;
        let admitted = self.compile_continuous_search(request.clone(), ctx)?;
        if self
            .prepared()
            .model_data()
            .interactions
            .iter()
            .any(|feature| !supports_interval(&feature.expression.to_domain()))
        {
            return Err(JointPanelError::Unsupported("the compiled response contains an expression without a certified interval evaluator; bounded best-found search remains available".into()));
        }
        let world_index =
            self.request().worlds.iter().position(|w| w.world == request.base_world).ok_or_else(
                || JointPanelError::compile("missing compiled certificate base world"),
            )?;
        let weights = self.projection(&request.functional)?.1.remove(world_index);
        let bindings = request
            .domain
            .controls
            .iter()
            .enumerate()
            .map(|(index, c)| {
                Ok(PanelExpressionControl {
                    channel: c.variable.raw().to_string(),
                    period: usize::try_from(c.offset).map_err(JointPanelError::compile)?,
                    variable: VariableId::from_raw(
                        u32::try_from(index).map_err(JointPanelError::compile)?,
                    ),
                })
            })
            .collect::<Result<Vec<_>, JointPanelError>>()?;
        let path = match &self.fit {
            super::super::JointPanelFit::Invariant(fit) => vec![fit.point.clone()],
            super::super::JointPanelFit::Varying(fit) => fit
                .point
                .parameters
                .iter()
                .map(|parameters| engine::PanelDraw {
                    parameters: parameters.clone(),
                    group_intercepts: fit.point.baseline.group_intercepts.clone(),
                })
                .collect(),
        };
        let expression = Arc::new(compile_panel_expression(
            self.prepared().model_data(),
            &path,
            &admitted.evaluator.base.program,
            &bindings,
            &weights,
            ctx,
        )?);
        let objective = ValueExpr::constant(offset)
            .add(ValueExpr::constant(scale).mul(expression.expression.clone()));
        let controls = request
            .domain
            .controls
            .iter()
            .zip(&bindings)
            .map(|(c, b)| CertifiedControl {
                variable: b.variable,
                bounds: ObjectiveInterval { lower: c.lower, upper: c.upper },
            })
            .collect();
        let mut restrictions = request
            .domain
            .restrictions
            .iter()
            .map(|r| CertifiedRestriction { coefficients: r.coefficients.clone(), bound: r.bound })
            .collect::<Vec<_>>();
        if let Some(budget) = request.domain.budget {
            restrictions.push(CertifiedRestriction {
                coefficients: request.domain.controls.iter().map(|c| c.unit_cost).collect(),
                bound: budget,
            });
        }
        let plan = PreparedCertifiedSearch::compile(objective, controls, restrictions).map_err(
            |error| match error {
                antecedent_design::program_search::ProgramSearchError::Invalid(reason) => {
                    JointPanelError::Unsupported(format!(
                        "certificate enclosure unavailable: {reason}"
                    ))
                }
                error => map_search_error(error),
            },
        )?;
        Ok(JointPanelCertifiedSearch {
            plan,
            evaluator: admitted.evaluator,
            expression,
            initial: request.initial,
            offset,
            scale,
            bindings,
            time_weights: weights,
        })
    }
}
impl JointPanelCertifiedSearch {
    /// Export both the frozen panel evaluation and the independently replayable
    /// numerical certificate, including the exact control and time bindings.
    /// # Errors
    /// Original execution, encoding, resource or cancellation failure.
    pub fn export(
        &self,
        options: CertificateOptions,
        ctx: &ExecutionContext,
    ) -> Result<Vec<u8>, JointPanelError> {
        // Verify the incumbent against the original response evaluator before
        // issuing an artifact, even when the caller has not executed this plan.
        let _verified = self.execute(options, ctx)?;
        let document = PanelCertificateDocument {
            version: 1,
            panel: self.evaluator.result.export(ctx).map_err(JointPanelError::compile)?,
            numerical: self
                .plan
                .export_replay(self.initial.clone(), options, ctx)
                .map_err(map_search_error)?,
            world: self.evaluator.base.world.raw(),
            bindings: self
                .bindings
                .iter()
                .map(|c| (c.channel.clone(), c.period, c.variable.raw()))
                .collect(),
            time_weights: self.time_weights.clone(),
            offset: self.offset,
            scale: self.scale,
        };
        antecedent_io::convert::to_cbor(&document).map_err(JointPanelError::compile)
    }
    /// Replay a certificate against the caller's current fitted evidence and
    /// compiled domain. Replaying an older fit cannot grant current permission.
    /// # Errors
    /// Changed fit, objective, controls, bounds or malformed proof.
    pub fn consume_current(
        &self,
        bytes: &[u8],
        ctx: &ExecutionContext,
    ) -> Result<GlobalProgramCertificate, JointPanelError> {
        let (program_identity, certificate) = replay_panel_certificate(bytes, ctx)?;
        if program_identity
            != SemanticDigest::from_bytes(
                self.evaluator
                    .result
                    .artifact_program()
                    .and_then(|p| p.identity())
                    .map_err(JointPanelError::compile)?,
            )
            || certificate.identity != self.plan.identity()
        {
            return Err(JointPanelError::compile("current panel certificate dependencies changed"));
        }
        Ok(certificate)
    }
    /// Certify the bounded fitted objective, then re-evaluate the incumbent through
    /// the original compiler to bind its returned scientific source. A certificate
    /// is scoped to this numerical fitted objective, not selected-value reliability.
    ///
    /// # Errors
    /// Numerical mismatch, changed authority, resources, cancellation or bad options.
    pub fn execute(
        &self,
        options: CertificateOptions,
        ctx: &ExecutionContext,
    ) -> Result<JointPanelCertifiedSearchResult, JointPanelError> {
        use antecedent_design::program_search::continuous::CompiledProgramEvaluator;
        let _admitted_expression = self.expression.admitted_bytes();
        let certificate =
            self.plan.execute(self.initial.clone(), options, ctx).map_err(map_search_error)?;
        let compiled = self
            .evaluator
            .evaluate_program(&certificate.controls, ctx)
            .map_err(map_search_error)?;
        compiled
            .source
            .authorize(JointPanelScientificScope::ModelConditional)
            .map_err(map_scientific_error)?;
        let ScientificSourcePayload::Means(means) = compiled.source.as_numerical_source() else {
            return Err(JointPanelError::compile(
                "certificate incumbent changed its mean representation",
            ));
        };
        let value = self.offset + self.scale * means.means[0];
        let point = certificate
            .controls
            .iter()
            .map(|x| ObjectiveInterval { lower: *x, upper: *x })
            .collect::<Vec<_>>();
        let enclosure = self.plan.bounds(&point).map_err(map_search_error)?;
        if value < enclosure.lower || value > enclosure.upper {
            return Err(JointPanelError::Engine(engine::PanelError::Numerical(
                "compiler-derived certificate disagrees with the retained response evaluator"
                    .into(),
            )));
        }
        Ok(JointPanelCertifiedSearchResult {
            certificate,
            best_source: compiled.source,
            fitted_dependency: self.evaluator.result.dependency_identity(),
            parent_target: self.evaluator.result.request().target.identity(),
        })
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct PanelCertificateDocument {
    version: u32,
    panel: Vec<u8>,
    numerical: Vec<u8>,
    world: u32,
    bindings: Vec<(String, usize, u32)>,
    time_weights: Vec<f64>,
    offset: f64,
    scale: f64,
}
/// Replay both the authentic frozen response evaluator and the numerical bound
/// proof in an independent process. Returns the historical panel identity and
/// certificate; neither value grants current causal or selected-value authority.
/// # Errors
/// Invalid artifact, changed compiler expression, forged bounds or resource refusal.
pub fn replay_panel_certificate(
    bytes: &[u8],
    ctx: &ExecutionContext,
) -> Result<(SemanticDigest, GlobalProgramCertificate), JointPanelError> {
    source::check_cancelled(ctx)?;
    if bytes.len() > 64 * 1024 * 1024 {
        return Err(JointPanelError::compile("panel certificate artifact size limit"));
    }
    let _input = ctx
        .acquire_admission(0, (bytes.len() as u64).saturating_mul(16))
        .map_err(|e| engine::PanelError::Resource(e.to_string()))?;
    let d: PanelCertificateDocument =
        antecedent_io::convert::from_cbor(bytes).map_err(JointPanelError::compile)?;
    if d.version != 1 || !d.offset.is_finite() || !d.scale.is_finite() {
        return Err(JointPanelError::compile("invalid panel certificate version/objective"));
    }
    let panel = antecedent_io::joint_panel_artifact::consume_panel_execution(&d.panel, ctx)
        .map_err(JointPanelError::compile)?;
    let bindings = d
        .bindings
        .into_iter()
        .map(|(channel, period, variable)| PanelExpressionControl {
            channel,
            period,
            variable: VariableId::from_raw(variable),
        })
        .collect::<Vec<_>>();
    let derived = panel
        .program()
        .controlled_response_expression(d.world, &bindings, &d.time_weights, ctx)
        .map_err(JointPanelError::compile)?;
    let (plan, certificate) =
        PreparedCertifiedSearch::replay(&d.numerical, None, ctx).map_err(map_search_error)?;
    plan.verify_expression(
        ValueExpr::constant(d.offset)
            .add(ValueExpr::constant(d.scale).mul(derived.expression.clone())),
    )
    .map_err(map_search_error)?;
    Ok((
        SemanticDigest::from_bytes(panel.program().identity().map_err(JointPanelError::compile)?),
        certificate,
    ))
}
