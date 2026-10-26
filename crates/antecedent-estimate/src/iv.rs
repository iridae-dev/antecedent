//! Instrumental-variable estimators: Wald ratio and two-stage least squares.
//!
//! Both estimators require an `"iv"` estimand with a non-empty
//! [`IdentifiedEstimand::instruments`] slice (see `antecedent_identify::iv`). Positivity is not
//! meaningful for IV — it is not a propensity-based method — so
//! [`OverlapPolicy::ExplicitOverride`] is the only supported policy, matching
//! [`crate::adjustment::LinearAdjustmentAte`].
//!
//! [`WaldIv`] implements the simple ratio-of-differences estimator for a single binary
//! instrument. [`TwoStageLeastSquares`] handles one or more instruments (continuous or binary)
//! and optional exogenous covariates via `antecedent_stats::fit_2sls`.
//!
//! # What is estimated
//!
//! Neither estimator targets the population average treatment effect without a further
//! restriction, which the identification side records as
//! `iv.constant_linear_effect_or_monotonicity` (`antecedent_identify::iv`):
//!
//! - under a constant linear structural effect of the treatment on the outcome, the Wald
//!   ratio and the 2SLS coefficient equal that effect, which is then also the average effect;
//! - with effects that differ across units, a binary instrument and a binary treatment, the
//!   Wald ratio is the local average treatment effect for compliers, and only under
//!   monotonicity (no defiers). With several or continuous instruments, 2SLS has no
//!   generic complier-average interpretation: an effect-weight representation needs
//!   additional restrictions, and multiple-instrument weights need not be positive.
//!
//! Neither case is the population average effect under heterogeneity. The `ate` field of
//! the returned [`EffectEstimate`] holds the estimate scaled to the queried contrast; the
//! field name is shared by every estimator and does not assert which of the two readings
//! applies.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg_attr(
    test,
    allow(
        clippy::cast_possible_truncation,
        clippy::float_cmp,
        reason = "test fixtures compare exact constants and index with small literals"
    )
)]

use std::sync::Arc;

use antecedent_core::{
    AssumptionSet, AverageEffectQuery, ExecutionContext, TargetPopulation, Value, VariableId,
};
use antecedent_data::{TableView, TabularData};
use antecedent_expr::{
    DomainRef, ExprNode, FactorRequirement, FunctionalProgram, IdentifiedEstimand, OutcomeExprId,
    ProgramLimits, ProgramSchema, ProgramVariable,
};
use antecedent_identify::{IdentificationResult, IdentificationStatus};
use antecedent_stats::{
    FaerBackend, FirstStageDiagnostics, IvBlocks, LeastSquaresWorkspace,
    anderson_rubin_confidence_set, fit_2sls,
};

use crate::adjustment::{EffectEstimate, intervention_f64};
use crate::error::EstimationError;
use crate::overlap::OverlapPolicy;
use crate::se::AnalyticSeKind;
use crate::util::{arm_moments};

/// Prepared IV problem: column-major instrument and exogenous-covariate designs, shared by
/// [`WaldIv`] and [`TwoStageLeastSquares`].
#[derive(Clone, Debug)]
pub struct PreparedIvProblem {
    /// Column-major `[1 | Z…]` instrument design.
    pub instruments_matrix: Arc<[f64]>,
    /// Instrument design column count (`1 + instruments.len()`).
    pub z_ncols: usize,
    /// Column-major `[1 | X…]` exogenous (non-instrumented) design.
    pub exogenous_matrix: Arc<[f64]>,
    /// Exogenous design column count (`1 + adjustment_set.len()`).
    pub x_ncols: usize,
    /// Complete-case row count.
    pub nrows: usize,
    /// Endogenous treatment, length `nrows`.
    pub treatment: Arc<[f64]>,
    /// Outcome, length `nrows`.
    pub outcome: Arc<[f64]>,
    /// Estimand method tag (always `"iv"`).
    pub method: Arc<str>,
    /// Instrument variables.
    pub instruments: Arc<[VariableId]>,
    /// Optional exogenous covariates (empty unless the estimand carries an adjustment set).
    pub adjustment_set: Arc<[VariableId]>,
    /// Overlap policy applied.
    pub overlap: OverlapPolicy,
    /// Active − control treatment contrast that scales the per-unit IV effect.
    pub treatment_delta: f64,
    /// Preparation receipt for the finite table this problem was read from.
    pub preparation: antecedent_data::materialize::PreparationReceipt,
}

/// Estimator procedure retained by a checked IV lowering.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CheckedIvProcedure {
    /// Single binary instrument Wald ratio.
    Wald,
    /// Two-stage least squares.
    TwoStageLeastSquares,
}

/// Typed semantic roles and procedure checked before IV design preparation.
#[derive(Clone, Debug)]
pub struct CheckedIvLowering {
    /// Selected causal per-unit Wald ratio root.
    pub functional: antecedent_expr::ExprId,
    /// Observational `E[Y|Z=z]` child program.
    pub outcome_program: FunctionalProgram,
    /// Observational `E[T|Z=z]` child program.
    pub treatment_program: FunctionalProgram,
    /// Endogenous treatment role.
    pub treatment: VariableId,
    /// Outcome role.
    pub outcome: VariableId,
    /// Instrument roles in selected-estimand order.
    pub instruments: Arc<[VariableId]>,
    /// Exogenous adjustment roles.
    pub adjustment: Arc<[VariableId]>,
    /// Queried active treatment level.
    pub active: f64,
    /// Queried control treatment level.
    pub control: f64,
    /// Instrument value used as the first conditional arm.
    pub instrument_active: f64,
    /// Instrument value used as the second conditional arm.
    pub instrument_control: f64,
    /// Procedure selected by the estimator.
    pub procedure: CheckedIvProcedure,
    /// Weak-instrument uncertainty method selected by the estimator.
    pub se_kind: AnalyticSeKind,
}

/// Checked IV preparation carrying its justification, expression program, and physical data.
#[derive(Clone, Debug)]
pub struct CheckedIvPreparation {
    /// Selected identified estimand.
    target: IdentifiedEstimand,
    /// Checked provider factor requirements.
    factor_requirements: Arc<[FactorRequirement]>,
    /// Checked typed IV lowering.
    lowering: CheckedIvLowering,
    /// Query whose contrast and population semantics were checked.
    query: AverageEffectQuery,
    /// Selected identification assumptions.
    required_assumptions: AssumptionSet,
    /// Prepared numerical design.
    problem: PreparedIvProblem,
}

impl CheckedIvPreparation {
    /// Rebind the physical IV design to compatible data while retaining the checked operation.
    #[allow(
        clippy::float_cmp,
        reason = "IV arm and binary-instrument values are exact category labels; approximate equality changes their semantics"
    )]
    pub fn rebind(&self, data: &TabularData) -> Result<Self, EstimationError> {
        let schema = ProgramSchema::new(
            data.schema()
                .variables()
                .iter()
                .map(|v| (v.id, ProgramVariable { name: Arc::clone(&v.name) })),
        );
        let active = intervention_f64(&self.query.active)?;
        let control = intervention_f64(&self.query.control)?;
        let mut arena = self.outcome_program().arena().clone();
        let expected_functional = arena
            .iv_wald(
                self.query.treatment,
                self.query.outcome,
                &self.target.instruments,
                &Value::f64(active),
                &Value::f64(control),
            )
            .map_err(|e| EstimationError::data_msg(format!("invalid retained IV roles: {e}")))?;
        if self.outcome_program().schema() != &schema
            || self.treatment_program().schema() != &schema
            || self.lowering.functional != self.target.functional
            || expected_functional != self.target.functional
            || self.lowering.treatment != self.query.treatment
            || self.lowering.outcome != self.query.outcome
            || self.lowering.instruments != self.target.instruments
            || self.lowering.adjustment != self.target.adjustment_set
            || self.lowering.active != active
            || self.lowering.control != control
            || self.lowering.instrument_active != 1.0
            || self.lowering.instrument_control != 0.0
            || self.problem.method.as_ref() != "iv"
            || self.problem.instruments != self.target.instruments
            || self.problem.adjustment_set != self.target.adjustment_set
            || self.problem.treatment_delta != active - control
        {
            return Err(EstimationError::data_msg(
                "checked IV target, query, lowering, or semantic schema changed",
            ));
        }
        let mut rebound = self.clone();
        rebound.problem =
            prepare_iv_problem(data, &self.target, &self.query, self.problem.overlap)?;
        let n = rebound.problem.nrows;
        if !rebound.problem.instruments_matrix[n..2 * n]
            .iter()
            .all(|&value| value == 0.0 || value == 1.0)
        {
            return Err(EstimationError::unsupported(
                "checked IV rebound requires a binary 0/1 instrument",
            ));
        }
        Ok(rebound)
    }

    /// Selected IV target.
    #[must_use]
    pub fn target(&self) -> &IdentifiedEstimand {
        &self.target
    }
    /// Checked observational outcome-mean child program.
    #[must_use]
    pub fn outcome_program(&self) -> &FunctionalProgram {
        &self.lowering.outcome_program
    }
    /// Checked observational treatment-mean child program.
    #[must_use]
    pub fn treatment_program(&self) -> &FunctionalProgram {
        &self.lowering.treatment_program
    }
    /// Provider factor requirements.
    #[must_use]
    pub fn factor_requirements(&self) -> &[FactorRequirement] {
        &self.factor_requirements
    }
    /// Typed IV lowering.
    #[must_use]
    pub fn lowering(&self) -> &CheckedIvLowering {
        &self.lowering
    }
    /// Selected identification assumptions.
    #[must_use]
    pub fn required_assumptions(&self) -> &AssumptionSet {
        &self.required_assumptions
    }
    /// Prepared numerical design.
    #[must_use]
    pub fn problem(&self) -> &PreparedIvProblem {
        &self.problem
    }
}

#[allow(
    clippy::too_many_lines,
    clippy::float_cmp,
    reason = "the sealed IV operation validates the selected claim, typed children, binary instrument design, and retained procedure together; arm and instrument values are exact category labels"
)]
fn prepare_iv_checked(
    data: &TabularData,
    identification: &IdentificationResult,
    estimand_index: usize,
    overlap: OverlapPolicy,
    procedure: CheckedIvProcedure,
    se_kind: AnalyticSeKind,
) -> Result<CheckedIvPreparation, EstimationError> {
    let crate::prepare::CheckedSelection { estimand: target, claim } =
        crate::prepare::CheckedSelection::resolve(identification, estimand_index)?;
    if claim.status != IdentificationStatus::IdentifiedUnderParametricRestrictions {
        return Err(EstimationError::data_msg(
            "checked IV requires the selected parametric-restriction identification claim",
        ));
    }
    let query = identification.average_effect().ok_or_else(|| {
        EstimationError::data_msg("checked IV requires an average-effect identification query")
    })?;
    if target.method_kind().ok() != Some(antecedent_expr::EstimandMethod::Iv)
        || target.instruments.len() != 1
        || !target.mediators.is_empty()
        || !target.adjustment_set.is_empty()
        || target.instruments[0] == query.treatment
        || target.instruments[0] == query.outcome
    {
        return Err(EstimationError::IncompatibleEstimand {
            message: "checked IV lowering requires one instrument and no mediator roles",
        });
    }
    let active = intervention_f64(&query.active)?;
    let control = intervention_f64(&query.control)?;
    #[allow(
        clippy::float_cmp,
        reason = "distinct treatment arm labels are a semantic requirement, not a numerical tolerance check"
    )]
    if active == control {
        return Err(EstimationError::unsupported(
            "active and control treatment levels must differ",
        ));
    }
    let mut arena = identification.arena.clone();
    let expected = arena
        .iv_wald(
            query.treatment,
            query.outcome,
            &target.instruments,
            &Value::f64(active),
            &Value::f64(control),
        )
        .map_err(|e| EstimationError::data_msg(format!("invalid checked IV target: {e}")))?;
    if expected != target.functional {
        return Err(EstimationError::data_msg(
            "selected IV functional does not match query treatment/outcome and instrument roles",
        ));
    }
    let schema = ProgramSchema::new(
        data.schema()
            .variables()
            .iter()
            .map(|v| (v.id, ProgramVariable { name: Arc::clone(&v.name) })),
    );
    let problem = prepare_iv_problem(data, target, query, overlap)?;
    let instrument = target.instruments[0];
    let column = problem.nrows;
    if !problem.instruments_matrix[column..2 * column]
        .iter()
        .all(|&value| value == 0.0 || value == 1.0)
    {
        return Err(EstimationError::unsupported(
            "checked IV currently requires a binary 0/1 instrument matching the checked Wald functional",
        ));
    }
    if instrument == query.treatment || instrument == query.outcome {
        return Err(EstimationError::data_msg("IV instrument role overlaps treatment or outcome"));
    }
    let zset = arena.intern_var_set([instrument]);
    let yset = arena.intern_var_set([query.outcome]);
    let tset = arena.intern_var_set([query.treatment]);
    let no_bindings = arena.empty_intervention_set();
    let y_factor = arena.intern_distribution(yset, zset, no_bindings, DomainRef::Observational);
    let t_factor = arena.intern_distribution(tset, zset, no_bindings, DomainRef::Observational);
    let outcome_root = arena.intern(ExprNode::Expectation {
        function: OutcomeExprId::identity(query.outcome),
        distribution: y_factor,
    });
    let treatment_root = arena.intern(ExprNode::Expectation {
        function: OutcomeExprId::identity(query.treatment),
        distribution: t_factor,
    });
    let outcome_program = FunctionalProgram::new(
        arena.clone(),
        schema.clone(),
        outcome_root,
        outcome_root,
        ProgramLimits::default(),
    )
    .map_err(|e| EstimationError::data_msg(format!("checked IV outcome child program: {e}")))?;
    let treatment_program = FunctionalProgram::new(
        arena,
        schema,
        treatment_root,
        treatment_root,
        ProgramLimits::default(),
    )
    .map_err(|e| EstimationError::data_msg(format!("checked IV treatment child program: {e}")))?;
    let factor_requirements: Arc<[FactorRequirement]> = Arc::from(
        outcome_program
            .factor_requirements()
            .iter()
            .chain(treatment_program.factor_requirements())
            .cloned()
            .collect::<Vec<_>>(),
    );
    Ok(CheckedIvPreparation {
        target: target.clone(),
        factor_requirements,
        lowering: CheckedIvLowering {
            functional: target.functional,
            outcome_program,
            treatment_program,
            treatment: query.treatment,
            outcome: query.outcome,
            instruments: Arc::clone(&target.instruments),
            adjustment: Arc::clone(&target.adjustment_set),
            active,
            control,
            instrument_active: 1.0,
            instrument_control: 0.0,
            procedure,
            se_kind,
        },
        query: query.clone(),
        required_assumptions: claim.required_assumptions.clone(),
        problem,
    })
}

fn prepare_iv_problem(
    data: &TabularData,
    estimand: &IdentifiedEstimand,
    query: &AverageEffectQuery,
    overlap: OverlapPolicy,
) -> Result<PreparedIvProblem, EstimationError> {
    crate::util::require_explicit_override(
        overlap,
        "IV estimators require ExplicitOverride overlap policy (not propensity-based)",
    )?;
    if estimand.method_kind().ok() != Some(antecedent_expr::EstimandMethod::Iv) {
        return Err(EstimationError::IncompatibleEstimand {
            message: "IV estimators expect an \"iv\" estimand",
        });
    }
    if estimand.instruments.is_empty() {
        return Err(EstimationError::IncompatibleEstimand {
            message: "IV estimators require a non-empty instrument set",
        });
    }
    query.validate()?;
    if !query.effect_modifiers.is_empty() {
        return Err(EstimationError::unsupported("IV estimators do not support effect modifiers"));
    }
    if query.target_population != TargetPopulation::AllObserved {
        return Err(EstimationError::refused(
            antecedent_core::reason_code!("population_not_estimable"),
            "IV estimators only support TargetPopulation::AllObserved",
        ));
    }
    let treatment = query.treatment;
    let outcome = query.outcome;
    let active = intervention_f64(&query.active)?;
    let control = intervention_f64(&query.control)?;
    let treatment_delta = active - control;
    if treatment_delta == 0.0 {
        return Err(EstimationError::unsupported(
            "active and control treatment levels must differ",
        ));
    }

    let mut ids =
        Vec::with_capacity(2 + estimand.instruments.len() + estimand.adjustment_set.len());
    ids.push(treatment);
    ids.push(outcome);
    ids.extend_from_slice(&estimand.instruments);
    ids.extend_from_slice(&estimand.adjustment_set);
    let crate::prepare::AdmittedColumns { data, row_mask, receipt: preparation } =
        crate::prepare::admit_estimator_columns(data, &ids, crate::prepare::FINITE_TABLE_ONLY)?;
    // Each selected column is gathered once, straight into the allocation the problem keeps.
    let t = data.float64_selected_arc(treatment, &row_mask).map_err(EstimationError::from)?;
    let y = data.float64_selected_arc(outcome, &row_mask).map_err(EstimationError::from)?;
    let nrows = t.len();

    let z_ncols = 1 + estimand.instruments.len();
    let mut instruments_matrix = antecedent_data::design_layout::zeroed_shared(nrows * z_ncols);
    let instruments_dest = shared_slice(&mut instruments_matrix)?;
    instruments_dest[..nrows].fill(1.0);
    for (i, &z_id) in estimand.instruments.iter().enumerate() {
        let base = (1 + i) * nrows;
        data.float64_selected_into(z_id, &row_mask, &mut instruments_dest[base..base + nrows])
            .map_err(EstimationError::from)?;
    }

    let x_ncols = 1 + estimand.adjustment_set.len();
    let mut exogenous_matrix = antecedent_data::design_layout::zeroed_shared(nrows * x_ncols);
    let exogenous_dest = shared_slice(&mut exogenous_matrix)?;
    exogenous_dest[..nrows].fill(1.0);
    for (i, &x_id) in estimand.adjustment_set.iter().enumerate() {
        let base = (1 + i) * nrows;
        data.float64_selected_into(x_id, &row_mask, &mut exogenous_dest[base..base + nrows])
            .map_err(EstimationError::from)?;
    }

    Ok(PreparedIvProblem {
        instruments_matrix,
        z_ncols,
        exogenous_matrix,
        x_ncols,
        nrows,
        treatment: t,
        outcome: y,
        method: Arc::clone(&estimand.method),
        instruments: Arc::clone(&estimand.instruments),
        adjustment_set: Arc::clone(&estimand.adjustment_set),
        overlap,
        treatment_delta,
        preparation,
    })
}

/// The unique contents of a freshly built shared design slice.
fn shared_slice(shared: &mut Arc<[f64]>) -> Result<&mut [f64], EstimationError> {
    Arc::get_mut(shared)
        .ok_or_else(|| EstimationError::data_msg("a freshly built shared design is not unique"))
}

// ---------------------------------------------------------------------------------------------
// Wald ratio estimator (single binary instrument)
// ---------------------------------------------------------------------------------------------

/// Wald (ratio-of-differences) IV estimator for a single binary instrument:
///
/// `(E[Y|Z=1] − E[Y|Z=0]) / (E[T|Z=1] − E[T|Z=0])`
///
/// The ratio is the constant treatment effect under a constant linear structural effect, and
/// otherwise, for a binary treatment under monotonicity, the local average treatment effect
/// for compliers. It is not the population average effect when effects are heterogeneous
/// (see the module docs).
///
/// Several or multi-valued instruments, or exogenous covariates, are routed to
/// [`TwoStageLeastSquares`], whose estimand is the instrument-weighted combination of the
/// pairwise Wald ratios.
#[derive(Clone, Debug)]
pub struct WaldIv {
    /// Bootstrap replicates (0 = skip bootstrap).
    pub bootstrap_replicates: u32,
    /// Overlap policy (must be [`OverlapPolicy::ExplicitOverride`]).
    pub overlap: OverlapPolicy,
    /// Analytic SE kind (default: delta-method on Y arms).
    pub se_kind: AnalyticSeKind,
    /// Optional cluster ids for [`AnalyticSeKind::Cluster`] (length = prepared `nrows`).
    pub cluster_ids: Option<Vec<u32>>,
    /// Multiway cluster ids.
    pub multiway_ids: Option<Vec<Vec<u32>>>,
    /// Optional panel time labels for panel HAC.
    pub panel_times: Option<Vec<i64>>,
}

impl Default for WaldIv {
    fn default() -> Self {
        Self::new()
    }
}

impl WaldIv {
    /// Default: 200 bootstrap replicates, explicit overlap override.
    #[must_use]
    pub fn new() -> Self {
        Self {
            bootstrap_replicates: 200,
            overlap: OverlapPolicy::ExplicitOverride,
            se_kind: AnalyticSeKind::Homoskedastic,
            cluster_ids: None,
            multiway_ids: None,
            panel_times: None,
        }
    }

    crate::inference::with_bootstrap_replicates!();

    /// Set the overlap policy. Must remain [`OverlapPolicy::ExplicitOverride`] — IV is not a
    /// propensity-based method.
    #[must_use]
    pub const fn with_overlap(mut self, overlap: OverlapPolicy) -> Self {
        self.overlap = overlap;
        self
    }

    crate::inference::with_se_kind!();

    crate::inference::with_cluster_ids!();

    crate::inference::with_multiway_ids!();

    crate::inference::with_panel_times!();

    /// Prepare the instrument/outcome/treatment design.
    ///
    /// # Errors
    ///
    /// Incompatible estimand (method or variable roles do not match an IV
    /// problem) or unsupported query options (e.g. effect modifiers) during
    /// problem preparation.
    pub fn prepare(
        &self,
        data: &TabularData,
        estimand: &IdentifiedEstimand,
        query: &AverageEffectQuery,
    ) -> Result<PreparedIvProblem, EstimationError> {
        prepare_iv_problem(data, estimand, query, self.overlap)
    }

    /// Prepare a Wald problem from one checked IV identification claim.
    pub fn prepare_checked(
        &self,
        data: &TabularData,
        identification: &IdentificationResult,
        estimand_index: usize,
    ) -> Result<CheckedIvPreparation, EstimationError> {
        prepare_iv_checked(
            data,
            identification,
            estimand_index,
            self.overlap,
            CheckedIvProcedure::Wald,
            self.se_kind,
        )
    }

    /// Rebind a checked Wald preparation to data with the same semantic schema and IV roles.
    ///
    /// The checked Wald operation, contrast, binary instrument contract, uncertainty method,
    /// and retained identification assumptions are preserved. Only physical row and column
    /// bindings are rebuilt.
    pub fn rebind_checked(
        &self,
        checked: &CheckedIvPreparation,
        data: &TabularData,
    ) -> Result<CheckedIvPreparation, EstimationError> {
        if checked.lowering.procedure != CheckedIvProcedure::Wald
            || checked.lowering.se_kind != self.se_kind
            || checked.problem.overlap != self.overlap
            || checked.target.instruments.len() != 1
            || !checked.target.adjustment_set.is_empty()
        {
            return Err(EstimationError::IncompatibleEstimand {
                message: "checked IV receipt does not match the Wald procedure and contract",
            });
        }
        checked.rebind(data)
    }

    /// Fit a sealed Wald preparation with its retained identification assumptions.
    pub fn fit_checked(
        &self,
        checked: &CheckedIvPreparation,
        ctx: &ExecutionContext,
    ) -> Result<EffectEstimate, EstimationError> {
        if checked.lowering.procedure != CheckedIvProcedure::Wald
            || checked.lowering.se_kind != self.se_kind
            || checked.problem.overlap != self.overlap
        {
            return Err(EstimationError::IncompatibleEstimand {
                message: "checked IV receipt was prepared for two-stage least squares",
            });
        }
        self.fit(&checked.problem, ctx, checked.required_assumptions.clone())
    }

    /// Compute the Wald ratio (constant effect, or complier LATE; see [`WaldIv`]).
    ///
    /// Licensed uncertainty is the homoskedastic Anderson–Rubin set attached to
    /// [`crate::EstimateDiagnostics::first_stage`]. Wald analytic and bootstrap
    /// standard errors are never published.
    ///
    /// # Errors
    ///
    /// A degenerate (zero) first stage, or the two-stage fit's own failures when the design
    /// has several or non-binary instruments or covariates.
    #[allow(
        clippy::float_cmp,
        reason = "a binary instrument is coded exactly 0.0 or 1.0, so exact comparison is the intended test"
    )]
    pub fn fit(
        &self,
        problem: &PreparedIvProblem,
        ctx: &ExecutionContext,
        assumptions: AssumptionSet,
    ) -> Result<EffectEstimate, EstimationError> {
        let n = problem.nrows;
        let binary_single = problem.instruments.len() == 1
            && problem.adjustment_set.is_empty()
            && (0..n).all(|r| {
                let v = problem.instruments_matrix[n + r];
                v == 0.0 || v == 1.0
            });
        if !binary_single {
            // The ratio of two arm differences is the one-binary-instrument case. Several or
            // multi-valued instruments and exogenous covariates are its instrument-weighted
            // generalization, which is exactly what two-stage least squares estimates.
            let two_stage = TwoStageLeastSquares {
                backend: FaerBackend,
                bootstrap_replicates: self.bootstrap_replicates,
                overlap: self.overlap,
                se_kind: self.se_kind,
                cluster_ids: self.cluster_ids.clone(),
                multiway_ids: self.multiway_ids.clone(),
                panel_times: self.panel_times.clone(),
            };
            return two_stage.fit(
                problem,
                &mut TwoStageLeastSquaresWorkspace::default(),
                ctx,
                assumptions,
            );
        }
        let z: Vec<f64> = (0..n).map(|r| problem.instruments_matrix[n + r]).collect();

        let wald = wald_ratio(&z, &problem.treatment, &problem.outcome)?;
        let ate = wald.ratio * problem.treatment_delta;
        let mut first_stage_diagnostics = wald_first_stage_diagnostics(&z, &problem.treatment);
        attach_anderson_rubin(
            &mut first_stage_diagnostics,
            self.se_kind,
            &ArData {
                y: &problem.outcome,
                t: &problem.treatment,
                instruments_colmajor: &z,
                z_ncols: 1,
                exogenous_colmajor: &problem.exogenous_matrix,
                x_ncols: problem.x_ncols,
                treatment_delta: problem.treatment_delta,
            },
        )?;
        // Licensed IV uncertainty is the Anderson–Rubin set on first-stage
        // diagnostics — never a Wald SE (pretest or otherwise) or bootstrap SE.
        let se_analytic = f64::NAN;

        Ok(EffectEstimate::new(ate, se_analytic, assumptions, problem.overlap)
            .with_n_obs(u64::try_from(n).unwrap_or(u64::MAX))
            .with_preparation(Some(crate::prepare::preparation_summary(&problem.preparation)))
            .with_first_stage_diagnostics(first_stage_diagnostics)
            .with_se_kind(self.se_kind))
    }
}

struct WaldResult {
    ratio: f64,
}

/// Ratio-of-differences point estimate.
fn wald_ratio(z: &[f64], t: &[f64], y: &[f64]) -> Result<WaldResult, EstimationError> {
    let outcome = arm_moments(z, y);
    let treatment = arm_moments(z, t);
    if outcome.n[0] == 0 || outcome.n[1] == 0 {
        return Err(EstimationError::data_msg(
            "Wald IV requires both instrument arms (Z=0 and Z=1) to be present",
        ));
    }
    let denom = treatment.mean[1] - treatment.mean[0];
    if denom.abs() < 1e-10 {
        return Err(EstimationError::stats_msg(
            "degenerate first stage: instrument is uncorrelated with treatment",
        ));
    }
    let ratio = (outcome.mean[1] - outcome.mean[0]) / denom;
    Ok(WaldResult { ratio })
}

/// Nominal level for the licensed Anderson–Rubin IV confidence set (matches the
/// Wald interval level previously claimed by `se_analytic`).
const AR_LEVEL: f64 = 0.95;

/// Column-major inputs of the Anderson–Rubin set: outcome, treatment, the excluded
/// instruments, the exogenous controls, and the treatment contrast that scales the set.
struct ArData<'a> {
    y: &'a [f64],
    t: &'a [f64],
    instruments_colmajor: &'a [f64],
    z_ncols: usize,
    exogenous_colmajor: &'a [f64],
    x_ncols: usize,
    treatment_delta: f64,
}

/// Attach a homoskedastic Anderson–Rubin set for the ATE-scale structural effect.
///
/// Non-homoskedastic `se_kind` values withhold AR with an explicit reason: a robust
/// AR score test is not derived here, and the F≥10 Wald SE is never used as a fallback.
fn attach_anderson_rubin(
    diagnostics: &mut Option<FirstStageDiagnostics>,
    se_kind: AnalyticSeKind,
    data: &ArData<'_>,
) -> Result<(), EstimationError> {
    let ArData {
        y,
        t,
        instruments_colmajor,
        z_ncols,
        exogenous_colmajor,
        x_ncols,
        treatment_delta,
    } = *data;
    let Some(diag) = diagnostics.as_mut() else {
        return Ok(());
    };
    if !matches!(se_kind, AnalyticSeKind::Homoskedastic) {
        diag.anderson_rubin = None;
        diag.uncertainty_withheld = Some("anderson_rubin_requires_homoskedastic");
        return Ok(());
    }
    let mut ws = LeastSquaresWorkspace::default();
    let blocks =
        IvBlocks { nrows: y.len(), instruments_colmajor, z_ncols, exogenous_colmajor, x_ncols };
    let (ar, reason) =
        anderson_rubin_confidence_set(&blocks, y, t, AR_LEVEL, &FaerBackend, &mut ws)
            .map_err(crate::error::EstimationError::from)?;
    // Scale structural-β endpoints to the published ATE contrast.
    diag.anderson_rubin = ar.map(|(lo, hi, level)| {
        let (mut lo_a, mut hi_a) = (lo * treatment_delta, hi * treatment_delta);
        if treatment_delta < 0.0 {
            std::mem::swap(&mut lo_a, &mut hi_a);
        }
        (lo_a, hi_a, level)
    });
    diag.uncertainty_withheld = reason;
    Ok(())
}

/// Weak-instrument diagnostic for the binary single-instrument Wald design.
///
/// Equivalent to an OLS F-test of `T ~ 1 + Z` for the null that the instrument's
/// coefficient is zero — i.e. the squared two-sample t-statistic comparing `T`'s `Z=1`
/// and `Z=0` arms with pooled variance, with `df1 = 1` and `df2 = n1 + n0 - 2`. Returns
/// `None` when either arm has fewer than 2 observations (pooled variance undefined),
/// mirroring [`fit_2sls`]'s [`FirstStageDiagnostics`] shape so both IV estimators expose
/// the same diagnostic type.
fn wald_first_stage_diagnostics(z: &[f64], t: &[f64]) -> Option<FirstStageDiagnostics> {
    let arms = arm_moments(z, t);
    let (n0, n1) = (arms.n[0], arms.n[1]);
    if n1 < 2 || n0 < 2 {
        return None;
    }
    let n1f = n1 as f64;
    let n0f = n0 as f64;
    let (mean_t0, mean_t1) = (arms.mean[0], arms.mean[1]);
    let (ss0, ss1) = (arms.sum_sq[0], arms.sum_sq[1]);
    let df2 = n1 + n0 - 2;
    let sse = ss1 + ss0;
    let ssr = (n1f * n0f / (n1f + n0f)) * (mean_t1 - mean_t0).powi(2);
    let sst = ssr + sse;
    let f_statistic = if sse > 0.0 { ssr / (sse / df2 as f64) } else { f64::INFINITY };
    let partial_r2 = if sst > 0.0 { ssr / sst } else { 0.0 };
    Some(FirstStageDiagnostics {
        f_statistic,
        df1: 1,
        df2,
        partial_r2,
        anderson_rubin: None,
        uncertainty_withheld: None,
    })
}

// ---------------------------------------------------------------------------------------------
// Two-stage least squares
// ---------------------------------------------------------------------------------------------

/// Estimation workspace for [`TwoStageLeastSquares`] (reusable across bootstrap replicates).
#[derive(Clone, Debug, Default)]
pub struct TwoStageLeastSquaresWorkspace {
    /// Least-squares scratch reused by both the first- and second-stage fits.
    pub ols: LeastSquaresWorkspace,
}

/// Two-stage least squares IV estimator.
///
/// The treatment coefficient is the constant structural effect under linearity; with
/// heterogeneous effects it is an instrument-weighted average of complier effects, not the
/// population average effect (see the module docs).
///
/// Stage 1 regresses the endogenous treatment on the FULL instrument set
/// `[instruments… | 1 | adjustment_set…]` (included exogenous regressors instrument
/// themselves); stage 2 regresses the outcome on `[fitted_T | 1 | adjustment_set…]`.
/// Supports one or more instruments (continuous or binary) and optional exogenous
/// covariates.
#[derive(Clone, Debug)]
pub struct TwoStageLeastSquares {
    /// Dense linear-algebra backend used by both least-squares stages.
    pub backend: FaerBackend,
    /// Bootstrap replicates (0 = skip bootstrap).
    pub bootstrap_replicates: u32,
    /// Overlap policy (must be [`OverlapPolicy::ExplicitOverride`]).
    pub overlap: OverlapPolicy,
    /// Analytic SE kind.
    pub se_kind: AnalyticSeKind,
    /// Optional cluster ids for cluster / panel SE.
    pub cluster_ids: Option<Vec<u32>>,
    /// Optional multiway cluster ids for [`AnalyticSeKind::Multiway`].
    pub multiway_ids: Option<Vec<Vec<u32>>>,
    /// Optional panel time labels for panel HAC.
    pub panel_times: Option<Vec<i64>>,
}

impl Default for TwoStageLeastSquares {
    fn default() -> Self {
        Self::new()
    }
}

impl TwoStageLeastSquares {
    /// Default: 200 bootstrap replicates, explicit overlap override.
    #[must_use]
    pub fn new() -> Self {
        Self {
            backend: FaerBackend,
            bootstrap_replicates: 200,
            overlap: OverlapPolicy::ExplicitOverride,
            se_kind: AnalyticSeKind::Homoskedastic,
            cluster_ids: None,
            multiway_ids: None,
            panel_times: None,
        }
    }

    crate::inference::with_bootstrap_replicates!();

    /// Set the overlap policy. Must remain [`OverlapPolicy::ExplicitOverride`] — IV is not a
    /// propensity-based method.
    #[must_use]
    pub const fn with_overlap(mut self, overlap: OverlapPolicy) -> Self {
        self.overlap = overlap;
        self
    }

    crate::inference::with_se_kind!();

    crate::inference::with_cluster_ids!();

    crate::inference::with_multiway_ids!();

    crate::inference::with_panel_times!();

    /// Prepare the instrument/exogenous/outcome/treatment design.
    ///
    /// # Errors
    ///
    /// Incompatible estimand (method or variable roles do not match an IV
    /// problem) or unsupported query options (e.g. effect modifiers) during
    /// problem preparation.
    pub fn prepare(
        &self,
        data: &TabularData,
        estimand: &IdentifiedEstimand,
        query: &AverageEffectQuery,
    ) -> Result<PreparedIvProblem, EstimationError> {
        prepare_iv_problem(data, estimand, query, self.overlap)
    }

    /// Prepare a two-stage least-squares problem from one checked IV claim.
    pub fn prepare_checked(
        &self,
        data: &TabularData,
        identification: &IdentificationResult,
        estimand_index: usize,
    ) -> Result<CheckedIvPreparation, EstimationError> {
        prepare_iv_checked(
            data,
            identification,
            estimand_index,
            self.overlap,
            CheckedIvProcedure::TwoStageLeastSquares,
            self.se_kind,
        )
    }

    /// Rebind a checked binary-instrument 2SLS preparation to compatible data.
    ///
    /// The target roles, 2SLS procedure, binary instrument premise, uncertainty method, and
    /// weak-instrument diagnostic contract remain fixed. Only physical row and column bindings
    /// are rebuilt.
    pub fn rebind_checked(
        &self,
        checked: &CheckedIvPreparation,
        data: &TabularData,
    ) -> Result<CheckedIvPreparation, EstimationError> {
        if checked.lowering.procedure != CheckedIvProcedure::TwoStageLeastSquares
            || checked.lowering.se_kind != self.se_kind
            || checked.problem.overlap != self.overlap
        {
            return Err(EstimationError::IncompatibleEstimand {
                message: "checked IV receipt does not match the 2SLS procedure and contract",
            });
        }
        checked.rebind(data)
    }

    /// Fit a sealed two-stage least-squares preparation with retained assumptions.
    pub fn fit_checked(
        &self,
        checked: &CheckedIvPreparation,
        workspace: &mut TwoStageLeastSquaresWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<EffectEstimate, EstimationError> {
        if checked.lowering.procedure != CheckedIvProcedure::TwoStageLeastSquares
            || checked.lowering.se_kind != self.se_kind
            || checked.problem.overlap != self.overlap
        {
            return Err(EstimationError::IncompatibleEstimand {
                message: "checked IV receipt was prepared for the Wald estimator",
            });
        }
        self.fit(&checked.problem, workspace, ctx, checked.required_assumptions.clone())
    }

    /// Fit 2SLS and report the instrumented treatment coefficient (constant effect, or an
    /// instrument-weighted complier effect; see the module docs).
    ///
    /// Licensed uncertainty is the homoskedastic Anderson–Rubin set attached to
    /// [`crate::EstimateDiagnostics::first_stage`]. Wald analytic and bootstrap
    /// standard errors are never published.
    ///
    /// # Errors
    ///
    /// Backend/rank failure in either stage.
    pub fn fit(
        &self,
        problem: &PreparedIvProblem,
        workspace: &mut TwoStageLeastSquaresWorkspace,
        ctx: &ExecutionContext,
        assumptions: AssumptionSet,
    ) -> Result<EffectEstimate, EstimationError> {
        let _ = ctx;
        // The exogenous block carries the intercept, so pass the excluded instruments
        // without their leading intercept column (fit_2sls appends the exogenous block
        // to form the full first-stage instrument set).
        let blocks = IvBlocks {
            nrows: problem.nrows,
            instruments_colmajor: &problem.instruments_matrix[problem.nrows..],
            z_ncols: problem.z_ncols - 1,
            exogenous_colmajor: &problem.exogenous_matrix,
            x_ncols: problem.x_ncols,
        };
        let fit = fit_2sls(
            &blocks,
            &problem.treatment,
            &problem.outcome,
            &self.backend,
            &mut workspace.ols,
        )
        .map_err(crate::error::EstimationError::from)?;
        let coef = fit.second_stage.coefficients[0];
        let ate = coef * problem.treatment_delta;
        let mut diagnostics = Some(fit.first_stage_diagnostics);
        attach_anderson_rubin(
            &mut diagnostics,
            self.se_kind,
            &ArData {
                y: &problem.outcome,
                t: &problem.treatment,
                instruments_colmajor: &problem.instruments_matrix[problem.nrows..],
                z_ncols: problem.z_ncols - 1,
                exogenous_colmajor: &problem.exogenous_matrix,
                x_ncols: problem.x_ncols,
                treatment_delta: problem.treatment_delta,
            },
        )?;
        let se_analytic = f64::NAN;

        Ok(EffectEstimate::new(ate, se_analytic, assumptions, problem.overlap)
            .with_n_obs(u64::try_from(problem.nrows).unwrap_or(u64::MAX))
            .with_preparation(Some(crate::prepare::preparation_summary(&problem.preparation)))
            .with_first_stage_diagnostics(diagnostics)
            .with_se_kind(self.se_kind))
    }
}

#[cfg(test)]
#[allow(
    clippy::many_single_char_names,
    clippy::float_cmp,
    reason = "this unit-test module compares floats that are copied, clamped or hand-set without rounding, so exact equality is intended"
)]
mod tests {
    use antecedent_core::StreamDomain;

    use std::sync::Arc;

    use antecedent_core::{
        AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, MeasurementSpec, RoleHint,
        SmallRoleSet, ValueType, VariableId,
    };
    use antecedent_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
    };
    use antecedent_expr::ExprId;
    use antecedent_expr::IdentifiedEstimand;
    use antecedent_identify::{
        DerivationTrace, IdentificationPerformanceRecord, IdentificationResult,
    };

    use super::*;
    use crate::overlap::OverlapPolicy;
    use antecedent_kernels::standard_normal;

    /// `Z → T → Y` with `U` confounding `T-Y`: `T = Z + U + noise`, `Y = 2T + U + noise`.
    /// `Z` is a continuous instrument uncorrelated with `U`. True structural effect = 2.0.
    fn continuous_iv_scm(n: usize, seed: u64) -> (TabularData, IdentifiedEstimand) {
        let mut rng =
            ExecutionContext::for_tests(seed).rng.stream_for(StreamDomain::Estimate, 0x1E70_u64);
        let mut z = vec![0.0; n];
        let mut t = vec![0.0; n];
        let mut y = vec![0.0; n];
        for i in 0..n {
            let zi = (i as f64) / (n as f64) - 0.5;
            let u = standard_normal(&mut rng);
            let ti = zi + u + 0.1 * standard_normal(&mut rng);
            let yi = 2.0 * ti + u + 0.1 * standard_normal(&mut rng);
            z[i] = zi;
            t[i] = ti;
            y[i] = yi;
        }
        (build_iv_data(n, t, y, z), instrumental_estimand())
    }

    #[test]
    fn iv_preparation_gathers_each_selected_column_once_into_what_it_keeps() {
        use antecedent_core::execution_attempt::{PhysicalEvent, observe_execution};

        let n = 200;
        let (data, estimand) = continuous_iv_scm(n, 41);
        let query = query();
        let attempt =
            observe_execution(|| TwoStageLeastSquares::new().prepare(&data, &estimand, &query));
        let problem = attempt.result.expect("prepared");
        assert_eq!(problem.nrows, n);
        // Treatment, outcome and the one instrument are the retained buffers: one copy each,
        // straight into the allocation the problem keeps. No temporary column exists.
        assert_eq!(attempt.report.physical(PhysicalEvent::Copy), Some(3));
        assert_eq!(attempt.report.physical(PhysicalEvent::CopiedBytes), Some((3 * n * 8) as u64));
        let z = data.float64_masked(VariableId::from_raw(2), &vec![true; n]).expect("z");
        assert_eq!(&problem.instruments_matrix[n..2 * n], z.as_slice());
        assert!(problem.instruments_matrix[..n].iter().all(|v| *v == 1.0));
    }

    /// `Z ∈ {0,1} → T → Y` with `U` confounding `T-Y`. True structural effect = 2.0.
    fn binary_iv_scm(n: usize, seed: u64) -> (TabularData, IdentifiedEstimand) {
        let mut rng =
            ExecutionContext::for_tests(seed).rng.stream_for(StreamDomain::Estimate, 0x1E71_u64);
        let mut z = vec![0.0; n];
        let mut t = vec![0.0; n];
        let mut y = vec![0.0; n];
        for i in 0..n {
            let zi = (i % 2) as f64;
            let u = standard_normal(&mut rng);
            let ti = 0.5 * zi + u + 0.1 * standard_normal(&mut rng);
            let yi = 2.0 * ti + u + 0.1 * standard_normal(&mut rng);
            z[i] = zi;
            t[i] = ti;
            y[i] = yi;
        }
        (build_iv_data(n, t, y, z), instrumental_estimand())
    }

    /// Same DGP as [`binary_iv_scm`] but with the instrument's effect on `T` shrunk to
    /// `0.01` (vs `0.5`) — a deliberately weak first stage for the F-statistic test.
    fn weak_binary_iv_scm(n: usize, seed: u64) -> (TabularData, IdentifiedEstimand) {
        let mut rng =
            ExecutionContext::for_tests(seed).rng.stream_for(StreamDomain::Estimate, 0x1E73_u64);
        let mut z = vec![0.0; n];
        let mut t = vec![0.0; n];
        let mut y = vec![0.0; n];
        for i in 0..n {
            let zi = (i % 2) as f64;
            let u = standard_normal(&mut rng);
            let ti = 0.01 * zi + u + 0.1 * standard_normal(&mut rng);
            let yi = 2.0 * ti + u + 0.1 * standard_normal(&mut rng);
            z[i] = zi;
            t[i] = ti;
            y[i] = yi;
        }
        (build_iv_data(n, t, y, z), instrumental_estimand())
    }

    /// Same DGP as [`continuous_iv_scm`] but with the instrument's effect on `T` shrunk to
    /// `0.01` (vs `1.0`) — a deliberately weak first stage for the F-statistic test.
    fn weak_continuous_iv_scm(n: usize, seed: u64) -> (TabularData, IdentifiedEstimand) {
        let mut rng =
            ExecutionContext::for_tests(seed).rng.stream_for(StreamDomain::Estimate, 0x1E74_u64);
        let mut z = vec![0.0; n];
        let mut t = vec![0.0; n];
        let mut y = vec![0.0; n];
        for i in 0..n {
            let zi = (i as f64) / (n as f64) - 0.5;
            let u = standard_normal(&mut rng);
            let ti = 0.01 * zi + u + 0.1 * standard_normal(&mut rng);
            let yi = 2.0 * ti + u + 0.1 * standard_normal(&mut rng);
            z[i] = zi;
            t[i] = ti;
            y[i] = yi;
        }
        (build_iv_data(n, t, y, z), instrumental_estimand())
    }

    fn instrumental_estimand() -> IdentifiedEstimand {
        IdentifiedEstimand::instrumental(
            "iv",
            Arc::from([VariableId::from_raw(2)]),
            ExprId::from_raw(0),
        )
    }

    fn build_iv_data(n: usize, t: Vec<f64>, y: Vec<f64>, z: Vec<f64>) -> TabularData {
        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "t",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "y",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "z",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        let schema = b.build().unwrap();
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(0),
                    Arc::from(t),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(1),
                    Arc::from(y),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(2),
                    Arc::from(z),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
        ];
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        TabularData::new(storage)
    }

    fn query() -> AverageEffectQuery {
        AverageEffectQuery::with_levels(VariableId::from_raw(0), VariableId::from_raw(1), 0.0, 1.0)
    }

    fn checked_result(query: AverageEffectQuery) -> IdentificationResult {
        let mut arena = antecedent_expr::CausalExprArena::new();
        let functional = arena
            .iv_wald(
                query.treatment,
                query.outcome,
                &[VariableId::from_raw(2)],
                &Value::f64(intervention_f64(&query.active).unwrap()),
                &Value::f64(intervention_f64(&query.control).unwrap()),
            )
            .unwrap();
        IdentificationResult::identified_under_parametric_restrictions(
            antecedent_core::CausalQuery::average_effect(query),
            vec![IdentifiedEstimand::instrumental(
                "iv",
                Arc::from([VariableId::from_raw(2)]),
                functional,
            )],
            arena,
            DerivationTrace::default(),
            AssumptionSet::new(),
            IdentificationPerformanceRecord::default(),
        )
    }

    fn ctx() -> ExecutionContext {
        ExecutionContext::for_tests(21)
    }

    #[test]
    fn two_sls_recovers_effect_two() {
        let (data, estimand) = continuous_iv_scm(2000, 1);
        let est = TwoStageLeastSquares { bootstrap_replicates: 30, ..TwoStageLeastSquares::new() };
        let prep = est.prepare(&data, &estimand, &query()).unwrap();
        let mut ws = TwoStageLeastSquaresWorkspace::default();
        let effect = est.fit(&prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap();
        assert!((effect.ate - 2.0).abs() < 0.3, "ate={}", effect.ate);
        assert!(effect.se_bootstrap().is_none(), "IV must not publish bootstrap SE");
        assert!(!effect.se_analytic().is_finite(), "IV must not publish Wald SE");
        crate::prepare::assert_records_preparation(&effect, 2000);
        let diag = effect.diagnostics.first_stage.as_ref().unwrap();
        let (lo, hi, level) = diag.anderson_rubin.expect("homoskedastic AR set");
        assert_eq!(level, 0.95);
        assert!(lo <= 2.0 && 2.0 <= hi, "AR [{lo}, {hi}]");
    }

    #[test]
    fn checked_two_sls_retains_iv_program_and_recovers_independent_truth() {
        let (data, _) = binary_iv_scm(20_000, 31);
        let identification = checked_result(query());
        let estimator = TwoStageLeastSquares::new().with_bootstrap_replicates(0);
        let checked = estimator.prepare_checked(&data, &identification, 0).unwrap();
        assert_eq!(checked.lowering.procedure, CheckedIvProcedure::TwoStageLeastSquares);
        assert_eq!(
            checked.lowering.outcome_program.mapping().source,
            checked.lowering.outcome_program.mapping().executable
        );
        assert_eq!(
            checked.lowering.treatment_program.mapping().source,
            checked.lowering.treatment_program.mapping().executable
        );
        assert_eq!(checked.lowering.instrument_active, 1.0);
        assert_eq!(checked.lowering.instrument_control, 0.0);
        assert!(checked.factor_requirements.iter().all(|factor| {
            factor.domain == antecedent_expr::DomainRef::Observational
                && factor.intervention.is_empty()
        }));
        let result = estimator
            .fit_checked(&checked, &mut TwoStageLeastSquaresWorkspace::default(), &ctx())
            .unwrap();
        let (refreshed_data, _) = binary_iv_scm(20_000, 32);
        let rebound = estimator.rebind_checked(&checked, &refreshed_data).unwrap();
        let rebound_result = estimator
            .fit_checked(&rebound, &mut TwoStageLeastSquaresWorkspace::default(), &ctx())
            .unwrap();
        assert_eq!(checked.target.functional, rebound.target.functional);
        assert_eq!(checked.lowering.procedure, rebound.lowering.procedure);
        assert_eq!(checked.lowering.se_kind, rebound.lowering.se_kind);
        assert_eq!(rebound.problem.nrows, 20_000);
        assert!((result.ate - 2.0).abs() < 0.08, "estimate={}", result.ate);
        assert!((rebound_result.ate - 2.0).abs() < 0.08, "rebound={}", rebound_result.ate);

        let wald = WaldIv::new().with_bootstrap_replicates(0);
        let wald_checked = wald.prepare_checked(&data, &identification, 0).unwrap();
        let wald_result = wald.fit_checked(&wald_checked, &ctx()).unwrap();
        assert!((wald_result.ate - 2.0).abs() < 0.2, "Wald={}", wald_result.ate);
        let (wald_refresh, _) = binary_iv_scm(20_000, 33);
        let wald_rebound = wald.rebind_checked(&wald_checked, &wald_refresh).unwrap();
        let wald_rebound_result = wald.fit_checked(&wald_rebound, &ctx()).unwrap();
        assert_eq!(wald_checked.target.functional, wald_rebound.target.functional);
        assert_eq!(wald_checked.lowering.procedure, wald_rebound.lowering.procedure);
        assert!((wald_rebound_result.ate - 2.0).abs() < 0.2);

        let mut changed_roles = checked.clone();
        changed_roles.lowering.outcome = VariableId::from_raw(2);
        assert!(estimator.rebind_checked(&changed_roles, &refreshed_data).is_err());
        assert!(wald.rebind_checked(&checked, &refreshed_data).is_err());
        assert!(estimator.rebind_checked(&wald_checked, &refreshed_data).is_err());

        let robust = TwoStageLeastSquares::new().with_se_kind(AnalyticSeKind::Hc1);
        assert!(robust.rebind_checked(&checked, &refreshed_data).is_err());

        let changed_schema = TabularData::from_f64_columns([
            ("renamed_t", &[0.0, 1.0][..]),
            ("y", &[0.0, 2.0][..]),
            ("z", &[0.0, 1.0][..]),
        ])
        .unwrap();
        assert!(estimator.rebind_checked(&checked, &changed_schema).is_err());

        let mut tampered = checked_result(query());
        tampered.estimands[0].instruments = Arc::from([VariableId::from_raw(1)]);
        assert!(estimator.prepare_checked(&data, &tampered, 0).is_err());
    }

    #[test]
    fn two_sls_withholds_wald_and_bootstrap_se() {
        let (data, estimand) = continuous_iv_scm(2000, 7);
        let est = TwoStageLeastSquares { bootstrap_replicates: 60, ..TwoStageLeastSquares::new() };
        let prep = est.prepare(&data, &estimand, &query()).unwrap();
        let mut ws = TwoStageLeastSquaresWorkspace::default();
        let effect = est.fit(&prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap();
        assert!(effect.se_bootstrap().is_none());
        assert!(!effect.se_analytic().is_finite());
        let diag = effect.diagnostics.first_stage.as_ref().unwrap();
        assert!(diag.anderson_rubin.is_some());
        assert!(diag.uncertainty_withheld.is_none());
    }

    #[test]
    fn two_sls_first_stage_diagnostics_strong_vs_weak() {
        let est = TwoStageLeastSquares::new();

        let (strong_data, strong_estimand) = continuous_iv_scm(2000, 1);
        let strong_prep = est.prepare(&strong_data, &strong_estimand, &query()).unwrap();
        let mut strong_ws = TwoStageLeastSquaresWorkspace::default();
        let strong_effect =
            est.fit(&strong_prep, &mut strong_ws, &ctx(), AssumptionSet::new()).unwrap();
        let strong_diag =
            strong_effect.diagnostics.first_stage.expect("2SLS always reports diagnostics");
        assert_eq!(strong_diag.df1, 1);
        assert!(
            strong_diag.f_statistic > 50.0,
            "expected a strong instrument to clear F=50, got {}",
            strong_diag.f_statistic
        );

        let (weak_data, weak_estimand) = weak_continuous_iv_scm(2000, 1);
        let weak_prep = est.prepare(&weak_data, &weak_estimand, &query()).unwrap();
        let mut weak_ws = TwoStageLeastSquaresWorkspace::default();
        let weak_effect = est.fit(&weak_prep, &mut weak_ws, &ctx(), AssumptionSet::new()).unwrap();
        let weak_diag =
            weak_effect.diagnostics.first_stage.expect("2SLS always reports diagnostics");
        assert!(
            weak_diag.f_statistic < 10.0,
            "expected a weak instrument to stay under the F=10 rule of thumb, got {}",
            weak_diag.f_statistic
        );
        assert!(!weak_effect.uncertainty.analytic.se.is_finite());
        assert!(weak_effect.uncertainty.bootstrap.se.is_none());
        assert!(!strong_effect.uncertainty.analytic.se.is_finite());
        match &weak_diag.anderson_rubin {
            Some((lo, hi, level)) => {
                assert_eq!(*level, 0.95);
                // Finite AR sets under weak IV may miss on a single draw; unbounded
                // sets must contain the truth on the infinite side.
                if !lo.is_finite() || !hi.is_finite() {
                    let covers = (!lo.is_finite() || 2.0 >= *lo) && (!hi.is_finite() || 2.0 <= *hi);
                    assert!(covers, "unbounded AR [{lo}, {hi}] must contain 2");
                }
            }
            None => {
                assert_eq!(
                    weak_diag.uncertainty_withheld,
                    Some("anderson_rubin_set_is_union"),
                    "honest withhold when AR set is a union"
                );
            }
        }
        assert!(
            strong_diag.f_statistic > 10.0 * weak_diag.f_statistic,
            "expected strong F ({}) to dwarf weak F ({})",
            strong_diag.f_statistic,
            weak_diag.f_statistic
        );
    }

    #[test]
    fn checked_two_sls_rebind_preserves_weak_instrument_diagnostics() {
        let (data, _) = weak_binary_iv_scm(2_000, 41);
        let identification = checked_result(query());
        let estimator = TwoStageLeastSquares::new().with_bootstrap_replicates(0);
        let checked = estimator.prepare_checked(&data, &identification, 0).unwrap();

        let (refreshed_data, _) = weak_binary_iv_scm(2_000, 42);
        let rebound = estimator.rebind_checked(&checked, &refreshed_data).unwrap();
        assert_eq!(rebound.lowering.procedure, CheckedIvProcedure::TwoStageLeastSquares);
        assert_eq!(rebound.lowering.se_kind, AnalyticSeKind::Homoskedastic);

        let result = estimator
            .fit_checked(&rebound, &mut TwoStageLeastSquaresWorkspace::default(), &ctx())
            .unwrap();
        let diagnostics = result.diagnostics.first_stage.expect("2SLS reports first stage");
        assert!(diagnostics.f_statistic < 10.0, "F={}", diagnostics.f_statistic);
        assert!(diagnostics.anderson_rubin.is_some() || diagnostics.uncertainty_withheld.is_some());
        assert!(result.uncertainty.bootstrap.se.is_none());
        assert!(!result.uncertainty.analytic.se.is_finite());
    }

    #[test]
    fn wald_iv_first_stage_diagnostics_strong_vs_weak() {
        let est = WaldIv { bootstrap_replicates: 0, ..WaldIv::new() };

        let (strong_data, strong_estimand) = binary_iv_scm(4000, 5);
        let strong_prep = est.prepare(&strong_data, &strong_estimand, &query()).unwrap();
        let strong_effect = est.fit(&strong_prep, &ctx(), AssumptionSet::new()).unwrap();
        let strong_diag =
            strong_effect.diagnostics.first_stage.expect("WaldIv always reports diagnostics");
        assert_eq!(strong_diag.df1, 1);
        assert!(
            strong_diag.f_statistic > 50.0,
            "expected a strong instrument to clear F=50, got {}",
            strong_diag.f_statistic
        );

        let (weak_data, weak_estimand) = weak_binary_iv_scm(200, 5);
        let weak_prep = est.prepare(&weak_data, &weak_estimand, &query()).unwrap();
        let weak_effect = est.fit(&weak_prep, &ctx(), AssumptionSet::new()).unwrap();
        let weak_diag =
            weak_effect.diagnostics.first_stage.expect("WaldIv always reports diagnostics");
        assert!(
            weak_diag.f_statistic < 10.0,
            "expected a weak instrument to stay under the F=10 rule of thumb, got {}",
            weak_diag.f_statistic
        );
        assert!(!weak_effect.uncertainty.analytic.se.is_finite());
        assert!(weak_effect.uncertainty.bootstrap.se.is_none());
        assert!(!strong_effect.uncertainty.analytic.se.is_finite());
        match &weak_diag.anderson_rubin {
            Some((lo, hi, level)) => {
                assert_eq!(*level, 0.95);
                if !lo.is_finite() || !hi.is_finite() {
                    let covers = (!lo.is_finite() || 2.0 >= *lo) && (!hi.is_finite() || 2.0 <= *hi);
                    assert!(covers, "unbounded AR [{lo}, {hi}] must contain 2");
                }
            }
            None => {
                assert_eq!(weak_diag.uncertainty_withheld, Some("anderson_rubin_set_is_union"));
            }
        }
        assert!(
            strong_diag.f_statistic > 10.0 * weak_diag.f_statistic,
            "expected strong F ({}) to dwarf weak F ({})",
            strong_diag.f_statistic,
            weak_diag.f_statistic
        );
    }

    #[test]
    fn two_sls_rejects_explicit_override_violation() {
        let (data, estimand) = continuous_iv_scm(100, 2);
        let est = TwoStageLeastSquares {
            overlap: OverlapPolicy::require_diagnostics(),
            ..TwoStageLeastSquares::new()
        };
        let err = est.prepare(&data, &estimand, &query()).unwrap_err();
        assert!(matches!(err, EstimationError::Overlap { .. }));
    }

    #[test]
    fn two_sls_rejects_non_iv_estimand() {
        let (data, mut estimand) = continuous_iv_scm(100, 3);
        estimand.method = Arc::from("backdoor.adjustment");
        let est = TwoStageLeastSquares::new();
        let err = est.prepare(&data, &estimand, &query()).unwrap_err();
        assert!(matches!(err, EstimationError::IncompatibleEstimand { .. }));
    }

    #[test]
    fn two_sls_rejects_empty_instruments() {
        let (data, mut estimand) = continuous_iv_scm(100, 4);
        estimand.instruments = Arc::from([]);
        let est = TwoStageLeastSquares::new();
        let err = est.prepare(&data, &estimand, &query()).unwrap_err();
        assert!(matches!(err, EstimationError::IncompatibleEstimand { .. }));
    }

    #[test]
    fn wald_iv_recovers_effect_two() {
        let (data, estimand) = binary_iv_scm(4000, 5);
        let est = WaldIv { bootstrap_replicates: 30, ..WaldIv::new() };
        let prep = est.prepare(&data, &estimand, &query()).unwrap();
        let effect = est.fit(&prep, &ctx(), AssumptionSet::new()).unwrap();
        assert!((effect.ate - 2.0).abs() < 0.6, "ate={}", effect.ate);
        assert!(effect.se_bootstrap().is_none());
        assert!(!effect.se_analytic().is_finite());
        let (lo, hi, _) =
            effect.diagnostics.first_stage.as_ref().unwrap().anderson_rubin.expect("AR set");
        assert!(lo <= 2.0 && 2.0 <= hi);
    }

    #[test]
    fn wald_iv_hc1_withholds_anderson_rubin() {
        let (data, estimand) = binary_iv_scm(800, 5);
        let est =
            WaldIv { bootstrap_replicates: 20, se_kind: AnalyticSeKind::Hc1, ..WaldIv::new() };
        let prep = est.prepare(&data, &estimand, &query()).unwrap();
        let effect = est.fit(&prep, &ctx(), AssumptionSet::new()).unwrap();
        assert!(!effect.se_analytic().is_finite());
        assert!(effect.se_bootstrap().is_none());
        let diag = effect.diagnostics.first_stage.as_ref().unwrap();
        assert!(diag.anderson_rubin.is_none());
        assert_eq!(diag.uncertainty_withheld, Some("anderson_rubin_requires_homoskedastic"));
    }

    #[test]
    fn anderson_rubin_coverage_on_weakish_dgp() {
        // First stage ~0.45·Z at n=300 leaves a substantial F<10 share; AR coverage
        // of the acceptance set (published interval, unbounded ray, or withheld union
        // that still contains the truth) must sit near 0.95.
        let n = 300usize;
        let n_sim = 400u32;
        let mut covered = 0u32;
        let mut weak_f = 0u32;
        let mut scored = 0u32;
        let est = WaldIv { bootstrap_replicates: 0, ..WaldIv::new() };
        for s in 0..n_sim {
            let (data, estimand) = moderate_binary_iv_scm(n, 90_000 + u64::from(s));
            let prep = est.prepare(&data, &estimand, &query()).unwrap();
            let effect = est.fit(&prep, &ctx(), AssumptionSet::new()).unwrap();
            assert!(!effect.se_analytic().is_finite());
            assert!(effect.se_bootstrap().is_none());
            let diag = effect.diagnostics.first_stage.as_ref().unwrap();
            if diag.f_statistic.is_finite() && diag.f_statistic < 10.0 {
                weak_f += 1;
            }
            scored += 1;
            let z: Vec<f64> =
                (0..prep.nrows).map(|r| prep.instruments_matrix[prep.nrows + r]).collect();
            let mut ws = LeastSquaresWorkspace::default();
            let ar_true = antecedent_stats::anderson_rubin_statistic(
                &IvBlocks {
                    nrows: prep.nrows,
                    instruments_colmajor: &z,
                    z_ncols: 1,
                    exogenous_colmajor: &prep.exogenous_matrix,
                    x_ncols: prep.x_ncols,
                },
                &prep.outcome,
                &prep.treatment,
                2.0,
                &FaerBackend,
                &mut ws,
            )
            .unwrap();
            let df = prep.nrows.saturating_sub(1 + prep.x_ncols);
            let crit = antecedent_stats::anderson_rubin_kf_critical(0.95, 1, df);
            let accepts_truth = ar_true.is_finite() && ar_true <= crit;
            let covers = match diag.anderson_rubin {
                Some((lo, hi, level)) => {
                    assert_eq!(level, 0.95);
                    let left_ok = !lo.is_finite() || 2.0 >= lo;
                    let right_ok = !hi.is_finite() || 2.0 <= hi;
                    left_ok && right_ok
                }
                None => {
                    // Union of disjoint rays is not published as one interval, but the
                    // AR non-rejection set still covers when the test accepts at truth.
                    diag.uncertainty_withheld == Some("anderson_rubin_set_is_union")
                        && accepts_truth
                }
            };
            if covers {
                covered += 1;
            }
        }
        let rate = f64::from(covered) / f64::from(scored);
        let weak_share = f64::from(weak_f) / f64::from(scored);
        assert!(weak_share > 0.05, "DGP must produce a non-trivial F<10 share, got {weak_share}");
        assert!(
            (0.85..=0.99).contains(&rate),
            "AR coverage {rate} outside wide band around 0.95 (covered={covered}/{scored}, weak_share={weak_share})"
        );
    }

    /// Binary IV with a moderate first stage (`0.45·Z`) so many draws have F < 10.
    fn moderate_binary_iv_scm(n: usize, seed: u64) -> (TabularData, IdentifiedEstimand) {
        let mut rng =
            ExecutionContext::for_tests(seed).rng.stream_for(StreamDomain::Estimate, 0x1E75_u64);
        let mut z = vec![0.0; n];
        let mut t = vec![0.0; n];
        let mut y = vec![0.0; n];
        for i in 0..n {
            let zi = (i % 2) as f64;
            let u = standard_normal(&mut rng);
            let ti = 0.45 * zi + u + 0.1 * standard_normal(&mut rng);
            let yi = 2.0 * ti + u + 0.1 * standard_normal(&mut rng);
            z[i] = zi;
            t[i] = ti;
            y[i] = yi;
        }
        (build_iv_data(n, t, y, z), instrumental_estimand())
    }

    #[test]
    fn wald_iv_rejects_continuous_instrument() {
        let (data, estimand) = continuous_iv_scm(200, 6);
        let est = WaldIv::new();
        let prep = est.prepare(&data, &estimand, &query()).unwrap();
        let err = est.fit(&prep, &ctx(), AssumptionSet::new()).unwrap_err();
        assert!(matches!(err, EstimationError::Unsupported { .. }));
    }

    #[test]
    fn wald_iv_rejects_nonempty_adjustment_set_while_two_sls_recovers() {
        // T = Z + U, Y = 2T + 3U; Z binary. Unconditional Wald is 3; 2SLS is 2.
        let z = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let u = [0.0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0];
        let t: Vec<f64> = z.iter().zip(u).map(|(zi, ui)| zi + ui).collect();
        let y: Vec<f64> = t.iter().zip(u).map(|(ti, ui)| 2.0 * ti + 3.0 * ui).collect();
        let mut b = CausalSchemaBuilder::new();
        for (name, hint) in [
            ("t", RoleHint::TreatmentCandidate),
            ("y", RoleHint::OutcomeCandidate),
            ("z", RoleHint::Context),
            ("u", RoleHint::Context),
        ] {
            b.add_variable(
                name,
                ValueType::Continuous,
                SmallRoleSet::from_hint(hint),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
        }
        let cols = [t, y, z.to_vec(), u.to_vec()]
            .into_iter()
            .enumerate()
            .map(|(i, values)| {
                OwnedColumn::Float64(
                    Float64Column::new(
                        VariableId::from_raw(u32::try_from(i).unwrap()),
                        Arc::from(values),
                        ValidityBitmap::all_valid(8),
                    )
                    .unwrap(),
                )
            })
            .collect();
        let data = TabularData::new(
            OwnedColumnarStorage::try_new(b.build().unwrap(), cols, None, None).unwrap(),
        );
        let mut estimand = instrumental_estimand();
        estimand.adjustment_set = Arc::from([VariableId::from_raw(3)]);
        let q = query();
        let wald = WaldIv { bootstrap_replicates: 8, ..WaldIv::new() };
        let wald_prep = wald.prepare(&data, &estimand, &q).unwrap();
        let err = wald.fit(&wald_prep, &ctx(), AssumptionSet::new()).unwrap_err();
        assert!(matches!(err, EstimationError::Unsupported { .. }), "{err:?}");

        let tsls = TwoStageLeastSquares::new();
        let tsls_prep = tsls.prepare(&data, &estimand, &q).unwrap();
        let mut ws = TwoStageLeastSquaresWorkspace::default();
        let effect = tsls.fit(&tsls_prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap();
        assert!((effect.ate - 2.0).abs() < 1e-12, "ate={}", effect.ate);
    }
}
