//! Refutation, sensitivity and diagnostic checks of an estimate, and their reports.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

pub use antecedent_validate::{
    PosteriorPredictiveCheck, PredictiveCheckKind, PredictiveCheckReport, PredictiveNoise,
    PriorPredictiveCheck, SerialDiscrepancy,
};

// --- Further items the language bindings use, re-exported from their defining crates ---

/// A sensitivity analysis reported as an assumption range tied to its target.
pub use antecedent_validate::AssumptionRange;
/// Object-safe custom effect validators.
pub use antecedent_validate::CustomEffectValidator;
/// Environment holdout validation via J-PCMCI+.
pub use antecedent_validate::EnvironmentHoldout;
/// Omitted-variable-bias sensitivity for linear adjustment.
pub use antecedent_validate::LinearAdjustmentSummary;
/// Orientation stability via PCMCI+ block bootstrap.
pub use antecedent_validate::OrientationStability;
/// Confounding sensitivity of a whole target vector, and of a posterior.
pub use antecedent_validate::OvbPosterior;
/// Regime stability via RPCMCI block bootstrap.
pub use antecedent_validate::RegimeStability;
/// Synthetic-null discovery calibration.
pub use antecedent_validate::SyntheticNullCalibration;
/// Validation errors.
pub use antecedent_validate::ValidationError;
/// Rosenbaum sensitivity bounds for matched designs.
pub use antecedent_validate::amplify;
/// PCMCI link-frequency stability and parameter grids.
pub use antecedent_validate::{
    AlphaThresholdSensitivity, BlockBootstrapStability, CiTestSensitivity,
    DiscoveryStabilityReport, LagWindowSensitivity,
};
/// Confounding sensitivity of the components of a mediation analysis and of the parameters of.
pub use antecedent_validate::{
    ComponentModel, MediationComponent, MediationComponents, MediationPath, MediationPosterior,
    MsmWeightPeriod,
};
/// Conflict policy: shrink external prior α from prior-PPC / KL signals.
pub use antecedent_validate::{ConflictPolicy, ConflictSignals, apply_conflict_and_compose};
/// E-values on the risk-ratio.
pub use antecedent_validate::{EffectInterval, EffectScale};
/// False-positive checks via permute / phase-randomize surrogates.
pub use antecedent_validate::{FalsePositiveCheck, NullTransform};
/// Exact sensitivity ranges for finite discrete outcome mechanisms.
pub use antecedent_validate::{
    FixedGraphMechanismSensitivitySpec, FixedGraphSensitivityError, SourceOutcomeKernelRow,
    SourceParentLawRow, TargetParentLawRow, ZTransportSensitivityError,
};
/// Joint mechanism deviations for the registered surrogate z-transport formula.
pub use antecedent_validate::{
    JointDeviationSpec, JointFactor, JointFactorBound, JointMechanismSensitivityResult,
    JointSensitivityLimits, TippingBracket,
};
/// Shared refuter types and data transforms.
pub use antecedent_validate::{RefutationProblem, RefutationReport, RefutationScope};

/// The signed-rank sensitivity of a vector of matched designs as one `AssumptionRange`.
///
/// `differences[k]` are the matched-pair differences behind coordinate `k`. At every probed
/// `Γ` the tested target (and test) is the largest upper-bound p-value across the coordinates,
/// the weakest evidence; the break-even value is the smallest `Γ*` across the coordinates, and
/// is absent when some coordinate does not reject even at `Γ = 1`.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_validate::rosenbaum_vector_range`].
///
/// # Errors
///
/// A coordinate count other than the target's dimension, or any error of
/// `signed_rank_sensitivity`.
#[deprecated(
    since = "3.1.0",
    note = "use antecedent::analysis::supplied_sensitivity_programs::compile_matched_design_sensitivity (MatchedDesign::Pairs, PairVector or Sets) and execute; the compiled path passes the program's execution context to the kernel where the direct call may pass none"
)]
pub fn rosenbaum_vector_range(
    target: antecedent_validate::RefutationTarget,
    differences: &[Vec<f64>],
    gammas: &[f64],
    alpha: f64,
    ctx: Option<&antecedent_core::ExecutionContext>,
) -> Result<
    (Vec<antecedent_validate::SignedRankSensitivity>, antecedent_validate::AssumptionRange),
    antecedent_validate::ValidationError,
> {
    antecedent_validate::rosenbaum_vector_range(target, differences, gammas, alpha, ctx)
}

/// Signed-rank sensitivity of matched-pair differences as an `AssumptionRange` on `target`.
///
/// The tested range is `gammas`; at each `Γ` the target is the upper-bound p-value. The
/// break-even (significance) value is `Γ*`.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_validate::rosenbaum_range`].
///
/// # Errors
///
/// As `signed_rank_sensitivity`.
#[deprecated(
    since = "3.1.0",
    note = "use antecedent::analysis::supplied_sensitivity_programs::compile_matched_design_sensitivity (MatchedDesign::Pairs, PairVector or Sets) and execute; the compiled path passes the program's execution context to the kernel where the direct call may pass none"
)]
pub fn rosenbaum_range(
    target: antecedent_validate::RefutationTarget,
    differences: &[f64],
    gammas: &[f64],
    alpha: f64,
    ctx: Option<&antecedent_core::ExecutionContext>,
) -> Result<
    (antecedent_validate::SignedRankSensitivity, antecedent_validate::AssumptionRange),
    antecedent_validate::ValidationError,
> {
    antecedent_validate::rosenbaum_range(target, differences, gammas, alpha, ctx)
}

/// The partial-R² analysis of a vector of linear-adjustment coefficients.
///
/// At each probed treatment partial R² (against an outcome partial R² of `r2yz_extreme`) the
/// tested target is the smallest share of the reported estimate left across the coordinates
/// (`adjusted / estimate`, negative once a coordinate is explained past zero), and the test
/// value is the smallest share left of the adjusted `1 - alpha` limit nearest zero
/// (`direction * limit / |estimate|`, positive while that limit excludes zero). The break-even
/// values are the smallest robustness values `RV_q` and `RV_{q,alpha}` across the coordinates.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_validate::ovb_vector_range`].
///
/// # Errors
///
/// A coordinate count other than the target's dimension, `q` not positive, `alpha` outside
/// `(0, 1)` or `r2yz_extreme` outside `(0, 1]`.
#[deprecated(
    since = "3.1.0",
    note = "use sensitivity_vector_range with SensitivityVectorRequest::OmittedVariableBias over one average-effect study per outcome: it calls ovb_vector_range on each study's linear_adjustment_fit summary"
)]
pub fn ovb_vector_range(
    target: antecedent_validate::RefutationTarget,
    summaries: &[antecedent_validate::LinearAdjustmentSummary],
    q: f64,
    alpha: f64,
    r2yz_extreme: f64,
) -> Result<antecedent_validate::AssumptionRange, antecedent_validate::ValidationError> {
    antecedent_validate::ovb_vector_range(target, summaries, q, alpha, r2yz_extreme)
}

/// The partial-R² analysis of a posterior over a vector of linear-adjustment coefficients.
///
/// The worst-case bias of a confounder with partial R²s `(r2dz, r2yz)` moves each draw
/// toward the null by `bias_scale * sqrt(r2yz * r2dz / (1 - r2dz))`. At every probed treatment
/// partial R² (against `r2yz_extreme`) the range reports the smallest share of the posterior
/// median left across the coordinates and the joint posterior probability that every
/// coordinate keeps its reported side. The break-even values are the smallest equal partial R²
/// (`r2dz = r2yz`) at which a coordinate's adjusted posterior median, and its adjusted
/// credible limit nearest the null, reach zero.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_validate::ovb_posterior_range`].
///
/// # Errors
///
/// Misaligned or non-finite draws, a non-positive bias scale, a level outside `(0, 1)`, or
/// `r2yz_extreme` outside `(0, 1]`.
#[deprecated(
    since = "3.1.0",
    note = "use PreparedStudy::sensitivity_range with SensitivityRangeRequest::OmittedVariableBiasPosterior on a Bayesian average effect under a Gaussian outcome model: it calls ovb_posterior_range on the retained effect draws with bias scale |active - control| sigma_d / sd(T | X)"
)]
pub fn ovb_posterior_range(
    target: antecedent_validate::RefutationTarget,
    posterior: &antecedent_validate::OvbPosterior<'_>,
    level: f64,
    r2yz_extreme: f64,
) -> Result<antecedent_validate::AssumptionRange, antecedent_validate::ValidationError> {
    antecedent_validate::ovb_posterior_range(target, posterior, level, r2yz_extreme)
}

/// The omitted-variable-bias analysis of `summary` as an `AssumptionRange` on `target`.
///
/// The tested range is the extreme-scenario curve: a confounder explaining the share
/// `r2yz_extreme` of the residual outcome variance (one is the most extreme) and each probed
/// share of the residual treatment variance; at each the adjusted estimate and the adjusted
/// `1 − alpha` limit nearest zero are reported. The break-even values are `RV_q` (equal
/// partial R²s that reduce the estimate by `100·q`%) and `RV_{q,α}`.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_validate::ovb_range`].
///
/// # Errors
///
/// `q` not positive, `alpha` outside `(0, 1)` or `r2yz_extreme` outside `(0, 1]`.
#[deprecated(
    since = "3.1.0",
    note = "use PreparedStudy::sensitivity_range with SensitivityRangeRequest::OmittedVariableBias on the study's own result: it fits linear_adjustment_fit on the study's treatment, outcome and adjustment set and calls ovb_range on that fit, bound to the result's target"
)]
pub fn ovb_range(
    target: antecedent_validate::RefutationTarget,
    summary: &antecedent_validate::LinearAdjustmentSummary,
    q: f64,
    alpha: f64,
    r2yz_extreme: f64,
) -> Result<antecedent_validate::AssumptionRange, antecedent_validate::ValidationError> {
    antecedent_validate::ovb_range(target, summary, q, alpha, r2yz_extreme)
}

/// Fit the total-effect and mediator regressions and summarize them.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_validate::mediation_sensitivity_fit`].
///
/// # Errors
///
/// Mismatched lengths, too few rows, non-finite values, a rank-deficient design or a
/// residual with no variance.
#[deprecated(
    since = "3.1.0",
    note = "use PreparedStudy::sensitivity_range with SensitivityRangeRequest::Mediation; its outcome carries the same MediationSensitivitySummary fitted on the study's own columns"
)]
pub fn mediation_sensitivity_fit(
    treatment: &[f64],
    mediator: &[f64],
    outcome: &[f64],
    covariates: &[Vec<f64>],
) -> Result<antecedent_validate::MediationSensitivitySummary, antecedent_validate::ValidationError>
{
    antecedent_validate::mediation_sensitivity_fit(treatment, mediator, outcome, covariates)
}

/// The mediation sensitivity curve as an `AssumptionRange` on `target`.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_validate::mediation_range`].
///
/// # Errors
///
/// None beyond a degenerate summary (non-finite values).
#[deprecated(
    since = "3.1.0",
    note = "use PreparedStudy::sensitivity_range with SensitivityRangeRequest::Mediation on a single-mediator MediationQuery study: it fits mediation_sensitivity_fit on the study's treatment, mediator, outcome and adjustment set and returns the whole mediation_range curve bound to the result's mediation target (the working model the compiled target.sequential_ignorability refuter reads)"
)]
pub fn mediation_range(
    target: antecedent_validate::RefutationTarget,
    summary: &antecedent_validate::MediationSensitivitySummary,
) -> Result<antecedent_validate::AssumptionRange, antecedent_validate::ValidationError> {
    antecedent_validate::mediation_range(target, summary)
}

/// Evaluate the checked z formula under a single coherent outcome-kernel contamination.
///
/// The compatible formula is `sum_w P_source(Y | w, X, do(Z)) P_source(w | do(Z))`.
/// The same conditional kernel is used to form both treatment arms. The returned
/// range is an assumption range over replacement outcome distributions.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_validate::z_transport_mechanism_sensitivity`].
///
/// # Errors
///
/// Returns the typed refusal of the proof, formula, provider, domain or kernel check.
#[deprecated(
    since = "3.1.0",
    note = "use antecedent::analysis::z_transport_sensitivity_program::compile_z_transport_mechanism_sensitivity and execute"
)]
pub fn z_transport_mechanism_sensitivity(
    diagram: &antecedent_graph::selection::SelectionDiagram,
    functional: &antecedent_identify::BoundZTransportFunctional,
    data: &antecedent_expr::ExactTransportData,
    max_fraction: f64,
    decision_threshold: Option<f64>,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<
    antecedent_validate::ZTransportMechanismSensitivityResult,
    antecedent_validate::ZTransportSensitivityError,
> {
    antecedent_validate::z_transport_mechanism_sensitivity(
        diagram,
        functional,
        data,
        max_fraction,
        decision_threshold,
        ctx,
    )
}

/// Bound the matched-set statistic under hidden bias at most `gamma`.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_validate::matched_sets_upper_p`].
///
/// # Errors
///
/// A set with fewer than two units, a treated index out of range, non-finite outcomes,
/// `gamma < 1`, or zero worst-case variance.
#[deprecated(
    since = "3.1.0",
    note = "use antecedent::analysis::supplied_sensitivity_programs::compile_matched_design_sensitivity (MatchedDesign::Pairs, PairVector or Sets) and execute; the compiled path passes the program's execution context to the kernel where the direct call may pass none"
)]
pub fn matched_sets_upper_p(
    sets: &[antecedent_validate::MatchedSet],
    score: antecedent_validate::MatchedSetScore,
    gamma: f64,
) -> Result<antecedent_validate::MatchedSetBound, antecedent_validate::ValidationError> {
    antecedent_validate::matched_sets_upper_p(sets, score, gamma)
}

/// Break-even `Γ` of the matched-set statistic at level `alpha`.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_validate::matched_sets_break_even`].
///
/// # Errors
///
/// As `matched_sets_upper_p`.
#[deprecated(
    since = "3.1.0",
    note = "use antecedent::analysis::supplied_sensitivity_programs::compile_matched_design_sensitivity (MatchedDesign::Pairs, PairVector or Sets) and execute; the compiled path passes the program's execution context to the kernel where the direct call may pass none"
)]
pub fn matched_sets_break_even(
    sets: &[antecedent_validate::MatchedSet],
    score: antecedent_validate::MatchedSetScore,
    alpha: f64,
) -> Result<Option<f64>, antecedent_validate::ValidationError> {
    antecedent_validate::matched_sets_break_even(sets, score, alpha)
}

/// Fit `Y ~ 1 + D + X` and the partial R²s of each covariate.
///
/// Partial R²s come from the residual sums of squares of the full and the leave-one-out
/// regressions, `R² = 1 − RSS_full / RSS_reduced`; the treatment standard error from
/// `t² = dof · (RSS_without_D − RSS_full) / RSS_full`.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_validate::linear_adjustment_fit`].
///
/// # Errors
///
/// Mismatched lengths, too few rows, non-finite values, a rank-deficient design or a
/// treatment that explains nothing.
#[deprecated(
    since = "3.1.0",
    note = "use PreparedStudy::sensitivity_range with SensitivityRangeRequest::OmittedVariableBias: its outcome carries linear_adjustment_fit of the study's outcome on its treatment and identified adjustment set (also sensitivity_vector_range with OmittedVariableBias, one study per outcome)"
)]
pub fn linear_adjustment_fit(
    outcome: &[f64],
    treatment: &[f64],
    covariates: &[Vec<f64>],
) -> Result<antecedent_validate::LinearAdjustmentFit, antecedent_validate::ValidationError> {
    antecedent_validate::linear_adjustment_fit(outcome, treatment, covariates)
}

/// E-values of every coordinate of a vector target as one `AssumptionRange` on `target`.
///
/// At each probed confounder strength `s` (risk-ratio scale, equal with treatment and outcome)
/// the tested target is the weakest coordinate's worst-case adjusted risk ratio, oriented away
/// from the null, and the test value is the weakest coordinate's adjusted limit nearest the
/// null (when every coordinate reports an interval). The break-even values are the smallest
/// point E-value and the smallest limit E-value across the coordinates.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_validate::evalue_vector_range`].
///
/// # Errors
///
/// A coordinate count other than the target's dimension, or any error of `evalue_bounds`.
#[deprecated(
    since = "3.1.0",
    note = "use sensitivity_vector_range with SensitivityVectorRequest::EValue over one average-effect study per outcome: it calls evalue_vector_range on each study's standardized effect and standard error. Ratio scales of estimates computed outside the library have no compiled replacement"
)]
pub fn evalue_vector_range(
    target: antecedent_validate::RefutationTarget,
    coordinates: &[antecedent_validate::EffectInterval],
    scale: antecedent_validate::EffectScale,
) -> Result<
    (Vec<antecedent_validate::EValueBounds>, antecedent_validate::AssumptionRange),
    antecedent_validate::ValidationError,
> {
    antecedent_validate::evalue_vector_range(target, coordinates, scale)
}

/// E-values of `interval` on `scale` as an `AssumptionRange` on `target`.
///
/// The tested range is the grid of confounder strengths `s` (equal association with
/// treatment and outcome) from `1` to `10`; at each the worst-case adjusted risk ratio
/// `RR / B(s)` (or `RR·B(s)` for a protective effect, capped at the null) and the adjusted
/// limit nearest the null are reported. The break-even values are the point and limit
/// E-values: the strength at which `B(s)` reaches the risk ratio.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_validate::evalue_range`].
///
/// # Errors
///
/// As `evalue_bounds`.
#[deprecated(
    since = "3.1.0",
    note = "use PreparedStudy::sensitivity_range with SensitivityRangeRequest::EValue on the study's own result: it calls evalue_range on the result's effect and standard error as a standardized mean difference (the effect over the residual standard deviation of the outcome regressed on the treatment and the adjustment set), bound to the result's target. Ratio scales of estimates computed outside the library have no compiled replacement"
)]
pub fn evalue_range(
    target: antecedent_validate::RefutationTarget,
    interval: antecedent_validate::EffectInterval,
    scale: antecedent_validate::EffectScale,
) -> Result<
    (antecedent_validate::EValueBounds, antecedent_validate::AssumptionRange),
    antecedent_validate::ValidationError,
> {
    antecedent_validate::evalue_range(target, interval, scale)
}

/// The E-value analysis of a posterior over a vector target.
///
/// `draws[k][d]` is draw `d` of coordinate `k` on `scale` (a ratio for the ratio scales, the
/// standardized difference for `EffectScale::StandardizedMeanDifference`); `level` is the
/// credible level. Each coordinate's reported side of the null is the side of its posterior
/// median. At every probed strength `s` the range reports the weakest coordinate's posterior
/// median adjusted by the bounding factor `B(s)`, and the joint posterior probability that
/// every coordinate's draw is still on its reported side after dividing (multiplying, for a
/// protective effect) by `B(s)`; the probability at `s = 1` is the posterior probability of
/// the reported sign. The break-even values are the smallest E-value of a posterior median and
/// of a credible limit nearest the null.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_validate::evalue_posterior_range`].
///
/// # Errors
///
/// Misaligned or non-finite draws, fewer than two draws, a level outside `(0, 1)`, or a
/// non-positive ratio draw.
#[deprecated(
    since = "3.1.0",
    note = "use PreparedStudy::sensitivity_range with SensitivityRangeRequest::EValuePosterior on a Bayesian average effect under a Gaussian outcome model: it calls evalue_posterior_range on each retained effect draw over the residual standard deviation of the same draw"
)]
pub fn evalue_posterior_range(
    target: antecedent_validate::RefutationTarget,
    draws: &[Vec<f64>],
    scale: antecedent_validate::EffectScale,
    level: f64,
) -> Result<
    (antecedent_validate::PosteriorEValue, antecedent_validate::AssumptionRange),
    antecedent_validate::ValidationError,
> {
    antecedent_validate::evalue_posterior_range(target, draws, scale, level)
}

/// The MSM parameters whose standard error carries the cost of estimating the weights.
///
/// The weights are the stabilized inverse-probability weights of `periods`. The fixed-weight
/// robust standard error of `msm_parameter_fit` holds them as known; here a unit bootstrap
/// refits every numerator and denominator model and the weighted regression in each replicate
/// (a replicate resamples subjects with replacement, as multiplicities on both fits), and the
/// reported standard error of each parameter is the standard deviation of its replicates, with
/// the effective degrees of freedom of the full-data weights. Ignoring the estimation of a
/// denominator model is conservative, so the refit error is the one that matches the
/// repeated-sampling spread; `fixed_weights` is reported beside it.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_validate::msm_parameter_fit_refitting_weights`].
///
/// # Errors
///
/// Misaligned inputs or a treatment not coded 0/1, fewer than `MIN_MSM_REPLICATES` replicates,
/// a weight model that does not converge on the full data, more than five percent failed
/// replicates, or any error of `msm_parameter_fit`.
#[deprecated(
    since = "3.1.0",
    note = "use PreparedStudy::sensitivity_range with SensitivityRangeRequest::MsmParametersRefittingWeights (MsmRefitRequest naming each period's numerator and denominator columns of the table) on a longitudinal regime study fitted with the marginal structural model method"
)]
pub fn msm_parameter_fit_refitting_weights(
    outcome: &[f64],
    exposures: &[Vec<f64>],
    periods: &[antecedent_validate::MsmWeightPeriod],
    replicates: usize,
    seed: u64,
) -> Result<antecedent_validate::MsmRefitFit, antecedent_validate::ValidationError> {
    antecedent_validate::msm_parameter_fit_refitting_weights(
        outcome, exposures, periods, replicates, seed,
    )
}

/// The weighted regression of a marginal structural model and its exposure parameters.
///
/// `exposures[j]` are the columns of the MSM's exposure terms (for example the cumulative dose,
/// or the indicator of a regime) and `weights` the subject weights (for example stabilized
/// inverse-probability weights). The model is `Y = β₀ + Σ_j β_j E_j` fitted by weighted least
/// squares; each `β_j` is reported with the heteroskedasticity-robust (HC1) standard error
/// `sqrt(n/(n-p) · [A⁻¹ (Σ w² e² x xᵀ) A⁻¹]_jj)`, `A = Σ w x xᵀ`, and the residual degrees of
/// freedom `n_eff - p` with the effective sample size `n_eff = (Σw)² / Σw²` of the weights. The
/// weights are held fixed.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_validate::msm_parameter_fit`].
///
/// # Errors
///
/// Misaligned or non-finite inputs, negative weights or none positive, a rank-deficient design,
/// or fewer than two effective residual degrees of freedom.
#[deprecated(
    since = "3.1.0",
    note = "use PreparedStudy::sensitivity_range with SensitivityRangeRequest::MsmParameters on a longitudinal regime study fitted with the marginal structural model method: it calls msm_parameter_fit on the route's own design and stabilized weights"
)]
pub fn msm_parameter_fit(
    outcome: &[f64],
    exposures: &[Vec<f64>],
    weights: &[f64],
) -> Result<Vec<antecedent_validate::LinearAdjustmentSummary>, antecedent_validate::ValidationError>
{
    antecedent_validate::msm_parameter_fit(outcome, exposures, weights)
}

/// The mediator-outcome sensitivity of a vector of mediation coordinates, read at the weakest.
///
/// At each probed `ρ` the tested target is the smallest share of the reported component left
/// across the coordinates (`direction · effect(ρ) / |effect(0)|`, negative once a coordinate is
/// explained past zero). The break-even value is the correlation of smallest magnitude at which
/// some coordinate's component reaches zero, so the claim "every coordinate keeps its sign"
/// first fails there; it is absent when no coordinate reaches zero.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_validate::mediation_vector_range`].
///
/// # Errors
///
/// A coordinate count other than the target's dimension, or a coordinate whose component is
/// zero at `ρ = 0` (nothing to explain away).
#[deprecated(
    since = "3.1.0",
    note = "use sensitivity_vector_range with SensitivityVectorRequest::MediationComponent over one mediation study per outcome, or PreparedStudy::sensitivity_range with SensitivityRangeRequest::MediationComponent for one"
)]
pub fn mediation_vector_range(
    target: antecedent_validate::RefutationTarget,
    coordinates: &[antecedent_validate::MediationComponents],
    component: antecedent_validate::MediationComponent,
) -> Result<antecedent_validate::AssumptionRange, antecedent_validate::ValidationError> {
    antecedent_validate::mediation_vector_range(target, coordinates, component)
}

/// The mediator-outcome sensitivity of a posterior over a vector of mediation coordinates.
///
/// Every draw of the summaries moves through the Imai-Keele-Yamamoto formula at each probed
/// `ρ`. Each coordinate's reported side is the side of its posterior median at `ρ = 0`. The
/// tested target is the smallest share of the posterior median left across the coordinates and
/// the test is the joint posterior probability that every coordinate's draw is still on its
/// reported side. The break-even value is the correlation of smallest magnitude at which some
/// coordinate's posterior median reaches zero, and the significance break-even the same for the
/// credible limit nearest zero at `level`.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_validate::mediation_posterior_range`].
///
/// # Errors
///
/// Misaligned or non-finite draws, fewer than two draws, a non-positive residual standard
/// deviation, a residual correlation outside `(-1, 1)`, a level outside `(0, 1)`, or a
/// coordinate whose posterior median is zero at `ρ = 0`.
#[deprecated(
    since = "3.1.0",
    note = "use PreparedStudy::sensitivity_range with SensitivityRangeRequest::MediationPosterior on a single-mediator mediation study: it calls mediation_posterior_range on the reference posterior of the linear working model drawn from the retained table under the context seed"
)]
pub fn mediation_posterior_range(
    target: antecedent_validate::RefutationTarget,
    posterior: &antecedent_validate::MediationPosterior<'_>,
    component: antecedent_validate::MediationComponent,
    level: f64,
) -> Result<antecedent_validate::AssumptionRange, antecedent_validate::ValidationError> {
    antecedent_validate::mediation_posterior_range(target, posterior, component, level)
}

/// Fit the three path summaries of one outcome by least squares.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_validate::mediation_paths_fit`].
///
/// # Errors
///
/// As `linear_adjustment_fit` for each of the three regressions.
#[deprecated(
    since = "3.1.0",
    note = "use PreparedStudy::sensitivity_range with SensitivityRangeRequest::MediationPath; its outcome carries the MediationPaths fitted on the study's own columns and the path's partial-R2 range"
)]
pub fn mediation_paths_fit(
    treatment: &[f64],
    mediator: &[f64],
    outcome: &[f64],
    covariates: &[Vec<f64>],
) -> Result<antecedent_validate::MediationPaths, antecedent_validate::ValidationError> {
    antecedent_validate::mediation_paths_fit(treatment, mediator, outcome, covariates)
}

/// The partial-R² range of one path of a vector of mediation coordinates (outcomes or
/// horizons), read at the weakest coordinate and bound to the assumption of that path.
///
/// A single path is the analysis of `ovb_vector_range` on that coefficient. The indirect
/// effect is the product of the treatment-mediator and mediator-outcome coefficients, each
/// moving toward zero by its own worst-case bias `se · sqrt(dof) · sqrt(R²yz R²dz / (1 - R²dz))`
/// when both are confounded at the tested strength: its share left is the product of the two
/// paths' shares (each floored at zero), the test value the product of the shares of their
/// adjusted `1 - alpha` limits, and the break-even values are the smallest equal partial R²
/// (`R²dz = R²yz`) at which the product of the shares falls to `1 - q`, for the estimate and for
/// the limits. At `q = 1` the product vanishes when either path does, so the break-even is the
/// smaller of the two paths' robustness values.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_validate::mediation_path_ovb_range`].
///
/// # Errors
///
/// An empty coordinate list, a count other than the target's dimension, `q` not positive,
/// `alpha` outside `(0, 1)` or `r2yz_extreme` outside `(0, 1]`.
#[deprecated(
    since = "3.1.0",
    note = "use sensitivity_vector_range with SensitivityVectorRequest::MediationPath over one mediation study per outcome, or PreparedStudy::sensitivity_range with SensitivityRangeRequest::MediationPath for one"
)]
pub fn mediation_path_ovb_range(
    target: antecedent_validate::RefutationTarget,
    coordinates: &[antecedent_validate::MediationPaths],
    path: antecedent_validate::MediationPath,
    q: f64,
    alpha: f64,
    r2yz_extreme: f64,
) -> Result<antecedent_validate::AssumptionRange, antecedent_validate::ValidationError> {
    antecedent_validate::mediation_path_ovb_range(target, coordinates, path, q, alpha, r2yz_extreme)
}

/// Fit the mediation summaries and the total effect of one outcome.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_validate::mediation_components_fit`].
///
/// # Errors
///
/// As `mediation_sensitivity_fit`.
#[deprecated(
    since = "3.1.0",
    note = "use PreparedStudy::sensitivity_range with SensitivityRangeRequest::MediationComponent; its outcome carries the MediationComponents fitted on the study's own columns and the component's range"
)]
pub fn mediation_components_fit(
    treatment: &[f64],
    mediator: &[f64],
    outcome: &[f64],
    covariates: &[Vec<f64>],
) -> Result<antecedent_validate::MediationComponents, antecedent_validate::ValidationError> {
    antecedent_validate::mediation_components_fit(treatment, mediator, outcome, covariates)
}

/// The partial-R² analysis of linear-adjustment summaries of components or parameters, bound to
/// `model`. For a mediation component the summaries are those of the mediator-outcome path
/// coefficient (the indirect effect is that coefficient times the unconfounded
/// treatment-mediator coefficient, so its share left and robustness value are the path's); for
/// an MSM they come from `msm_parameter_fit`.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_validate::component_ovb_range`].
///
/// # Errors
///
/// As `ovb_vector_range`.
#[deprecated(
    since = "3.1.0",
    note = "use sensitivity_vector_range with SensitivityVectorRequest::MediationComponentOvb over one mediation study per outcome (the mediator-outcome path summaries), or PreparedStudy::sensitivity_range with SensitivityRangeRequest::MsmParameters or MsmParametersRefittingWeights on a longitudinal regime study (route LongitudinalRegime)"
)]
pub fn component_ovb_range(
    target: antecedent_validate::RefutationTarget,
    summaries: &[antecedent_validate::LinearAdjustmentSummary],
    q: f64,
    alpha: f64,
    r2yz_extreme: f64,
    model: antecedent_validate::ComponentModel,
) -> Result<antecedent_validate::AssumptionRange, antecedent_validate::ValidationError> {
    antecedent_validate::component_ovb_range(target, summaries, q, alpha, r2yz_extreme, model)
}

/// The partial-R² analysis of a posterior over components or parameters, bound to `model`.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_validate::component_ovb_posterior_range`].
///
/// # Errors
///
/// As `ovb_posterior_range`.
#[deprecated(
    since = "3.1.0",
    note = "use PreparedStudy::sensitivity_range with SensitivityRangeRequest::MediationComponentOvbPosterior on a mediation study. The marginal-structural-model setting has no compiled posterior (the longitudinal regime route's MSM is point-only)"
)]
pub fn component_ovb_posterior_range(
    target: antecedent_validate::RefutationTarget,
    posterior: &antecedent_validate::OvbPosterior<'_>,
    level: f64,
    r2yz_extreme: f64,
    model: antecedent_validate::ComponentModel,
) -> Result<antecedent_validate::AssumptionRange, antecedent_validate::ValidationError> {
    antecedent_validate::component_ovb_posterior_range(
        target,
        posterior,
        level,
        r2yz_extreme,
        model,
    )
}

/// E-values of every component or parameter, read at the weakest, bound to `model`'s assumption.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_validate::component_evalue_range`].
///
/// # Errors
///
/// As `evalue_vector_range`.
#[deprecated(
    since = "3.1.0",
    note = "use sensitivity_vector_range with SensitivityVectorRequest::MediationComponentEValue over one mediation study per outcome (each requested contrast standardized by the outcome regression's residual standard deviation), or PreparedStudy::sensitivity_range with SensitivityRangeRequest::MsmParameterEValue on a longitudinal regime study fitted with the marginal structural model method (route LongitudinalRegime)"
)]
pub fn component_evalue_range(
    target: antecedent_validate::RefutationTarget,
    coordinates: &[antecedent_validate::EffectInterval],
    scale: antecedent_validate::EffectScale,
    model: antecedent_validate::ComponentModel,
) -> Result<
    (Vec<antecedent_validate::EValueBounds>, antecedent_validate::AssumptionRange),
    antecedent_validate::ValidationError,
> {
    antecedent_validate::component_evalue_range(target, coordinates, scale, model)
}

/// The E-value analysis of a posterior over components or parameters, bound to `model`.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_validate::component_evalue_posterior_range`].
///
/// # Errors
///
/// As `evalue_posterior_range`.
#[deprecated(
    since = "3.1.0",
    note = "use PreparedStudy::sensitivity_range with SensitivityRangeRequest::MediationComponentEValuePosterior on a mediation study. The marginal-structural-model setting has no compiled posterior (the longitudinal regime route's MSM is point-only)"
)]
pub fn component_evalue_posterior_range(
    target: antecedent_validate::RefutationTarget,
    draws: &[Vec<f64>],
    scale: antecedent_validate::EffectScale,
    level: f64,
    model: antecedent_validate::ComponentModel,
) -> Result<
    (antecedent_validate::PosteriorEValue, antecedent_validate::AssumptionRange),
    antecedent_validate::ValidationError,
> {
    antecedent_validate::component_evalue_posterior_range(target, draws, scale, level, model)
}

/// Confounder bounds from an observed covariate's partial R²s.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_validate::benchmark_bound`].
///
/// # Errors
///
/// Partial R²s outside `[0, 1)`, a non-positive multiplier, or a multiplier so large that
/// the implied treatment partial R² reaches one.
#[deprecated(
    since = "3.1.0",
    note = "use PreparedStudy::sensitivity_range with SensitivityRangeRequest::OmittedVariableBias: its outcome carries benchmark_bound against every adjustment covariate of the study with the declared kd and ky"
)]
pub fn benchmark_bound(
    r2dxj: f64,
    r2yxj: f64,
    kd: f64,
    ky: f64,
) -> Result<antecedent_validate::BenchmarkBound, antecedent_validate::ValidationError> {
    antecedent_validate::benchmark_bound(r2dxj, r2yxj, kd, ky)
}

/// Bounded, fit-free descriptions of suspicious data characteristics.
pub mod data_diagnostics {
    pub use antecedent_validate::data_diagnostics::DATA_DIAGNOSTIC_CHECKS;
}
/// Assessment of predictive laws.
pub mod predictive_assessment {
    pub use antecedent_validate::predictive_assessment::{PredictiveAssessmentError, score_case};
}
/// Rosenbaum sensitivity bounds for matched designs.
pub mod rosenbaum {
    pub use antecedent_validate::rosenbaum::{DEFAULT_GAMMA_GRID, MatchedSet, MatchedSetScore};
}
/// Target-native refutation: refuters that test the quantity a claim actually reports.
pub mod target_native {
    pub use antecedent_validate::target_native::{
        DerivativeCheckPoint, PosteriorPredictiveTarget, PriorSensitivityTarget,
    };

    /// Prior sensitivity of the whole target vector.
    ///
    /// Publishes the largest measured shift (`statistic.value`, in baseline posterior standard
    /// deviations) and whether it is within the declared tolerance (`passed`). This is a comparison
    /// with a bound the caller chose, not a hypothesis test or a refutation verdict with an error
    /// level.
    ///
    /// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
    /// names. This forwards to [`antecedent_validate::target_native::prior_sensitivity_target_check`].
    ///
    /// # Errors
    ///
    /// No alternative, too few baseline draws for a covariance, mismatched dimensions, a
    /// non-positive tolerance, or non-finite draws.
    #[deprecated(
        since = "3.1.0",
        note = "use PreparedStudy::refute_retained_target on a Bayesian study: its target.prior_sensitivity refuter refits the study under each alternative prior scale (TargetRefitOptions::prior_scale_factors) and calls prior_sensitivity_target_check on the refits' draws of the whole target. The direct check reads draws the caller supplies"
    )]
    pub fn prior_sensitivity_target_check(
        input: &antecedent_validate::target_native::PriorSensitivityTarget<'_>,
    ) -> Result<antecedent_validate::TargetRefutationReport, antecedent_validate::ValidationError>
    {
        antecedent_validate::target_native::prior_sensitivity_target_check(input)
    }

    /// Posterior-predictive check of the target quantity.
    ///
    /// The pass/fail threshold is a diagnostic convention; neither the verdict nor its
    /// finite-draw tail estimate is calibrated here. A zero fraction means no exceedances
    /// were observed in these draws.
    ///
    /// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
    /// names. This forwards to [`antecedent_validate::target_native::posterior_predictive_target_check`].
    ///
    /// # Errors
    ///
    /// Mismatched dimensions, too few draws, a degenerate replicate covariance, or `alpha`
    /// outside `(0, 1)`.
    #[deprecated(
        since = "3.1.0",
        note = "use the compiled StaticResponseDag route, which calls this kernel function with the same inputs"
    )]
    pub fn posterior_predictive_target_check(
        input: &antecedent_validate::target_native::PosteriorPredictiveTarget<'_>,
    ) -> Result<antecedent_validate::TargetRefutationReport, antecedent_validate::ValidationError>
    {
        antecedent_validate::target_native::posterior_predictive_target_check(input)
    }

    /// Second-difference curvature of `response` relative to its slope over `step`.
    ///
    /// The slope floor `slope_floor` keeps a flat response (zero slope) from dividing by zero:
    /// the relative error is `κ·h / (2·max(|f'|, slope_floor))`.
    ///
    /// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
    /// names. This forwards to [`antecedent_validate::target_native::local_linearity`].
    ///
    /// # Errors
    ///
    /// A non-positive step, tolerance or slope floor, no points, or a non-finite response value.
    #[deprecated(
        since = "3.1.0",
        note = "use the compiled StaticResponseDag route, which calls this kernel function with the same inputs"
    )]
    pub fn local_linearity(
        response: impl FnMut(f64) -> Result<f64, antecedent_validate::ValidationError>,
        points: &[f64],
        step: f64,
        tolerance: f64,
        slope_floor: f64,
    ) -> Result<
        antecedent_validate::target_native::LocalLinearity,
        antecedent_validate::ValidationError,
    > {
        antecedent_validate::target_native::local_linearity(
            response,
            points,
            step,
            tolerance,
            slope_floor,
        )
    }

    /// Compare reported derivatives with Richardson-extrapolated central differences of
    /// `response`.
    ///
    /// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
    /// names. This forwards to [`antecedent_validate::target_native::finite_difference_consistency`].
    ///
    /// # Errors
    ///
    /// A non-positive step or tolerance, no points, or a non-finite response value.
    #[deprecated(
        since = "3.1.0",
        note = "use the compiled StaticResponseDag route, which calls this kernel function with the same inputs"
    )]
    pub fn finite_difference_consistency(
        response: impl FnMut(f64) -> Result<f64, antecedent_validate::ValidationError>,
        points: &[antecedent_validate::target_native::DerivativeCheckPoint],
        step: f64,
        tolerance: f64,
    ) -> Result<
        antecedent_validate::target_native::DerivativeConsistency,
        antecedent_validate::ValidationError,
    > {
        antecedent_validate::target_native::finite_difference_consistency(
            response, points, step, tolerance,
        )
    }
}
