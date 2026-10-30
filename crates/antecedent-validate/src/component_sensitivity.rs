//! Confounding sensitivity of the components of a mediation analysis and of the parameters of a
//! marginal structural model, as vectors and as posteriors.
//!
//! A mediation result has several components (the indirect and direct effects, per outcome or
//! horizon) and a marginal structural model (MSM) has several parameters, so a claim about them
//! is explained away as soon as *any* coordinate is. Each analysis here reads its range at the
//! weakest coordinate, as [`crate::vector_sensitivity`] does, and is bound to the assumption it
//! relaxes:
//!
//! * [`mediation_vector_range`] and [`mediation_posterior_range`] carry the sensitivity
//!   parameter `ρ = corr(ε₂, ε₃)` of Imai, Keele and Yamamoto (2010) through the indirect and
//!   the direct effect. The total effect is identified without the mediator assumption, so the
//!   direct effect is the total effect minus the indirect effect at `ρ`, and its break-even
//!   correlation has a closed form (see [`MediationComponents::break_even`]). The posterior
//!   version applies the formula draw by draw to the posterior of the regression summaries and
//!   reports the joint posterior probability that every coordinate keeps its sign.
//! * [`component_evalue_range`], [`component_evalue_posterior_range`], [`component_ovb_range`]
//!   and [`component_ovb_posterior_range`] are the E-value and the partial-R² analyses of
//!   [`crate::vector_sensitivity`] read on a mediation component or an MSM parameter and bound
//!   to the assumption of that setting ([`ComponentModel`]): no unmeasured mediator-outcome
//!   confounding, or sequential exchangeability (no unmeasured time-varying confounding).
//! * [`msm_parameter_fit`] fits the weighted regression of an MSM and reports each exposure
//!   parameter as a [`LinearAdjustmentSummary`] with a heteroskedasticity-robust standard error
//!   and the effective degrees of freedom of the weights, the inputs of the partial-R² range.
//!   The weights are held fixed, so the standard error does not carry the cost of estimating
//!   them; that is stated by the assumption the range is bound to.
//!
//! Every range binds to its [`RefutationTarget`]. The confounding structure is an assumption of
//! the analysis, not something the data certify.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_kernels::quantile_type7_sorted as quantile_sorted;
use antecedent_stats::{DenseLinearAlgebra, FaerBackend, LeastSquaresWorkspace, student_t_ppf};

use crate::ValidationError;
use crate::assumption_range::{AssumptionRange, RangePoint, SensitivityAssumption};
use crate::common::{not_applicable, require_dimension};
use crate::evalue_scales::{EValueBounds, EffectInterval, EffectScale};
use crate::mediation_sensitivity::{MediationSensitivitySummary, mediation_sensitivity_fit};
use crate::omitted_variable_bias::{LinearAdjustmentSummary, linear_adjustment_fit};
use crate::target_native::RefutationTarget;
use crate::vector_sensitivity::{
    OvbPosterior, PosteriorEValue, evalue_posterior_range, evalue_vector_range,
    ovb_posterior_range, ovb_vector_range,
};

/// Probed mediator-outcome error correlations (the grid of the scalar analysis).
const RHO_GRID: [f64; 13] =
    [-0.9, -0.7, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.5, 0.7];

/// Outermost correlation the posterior break-even search reaches.
const RHO_LIMIT: f64 = 0.9975;
/// Steps of the posterior break-even search from zero to [`RHO_LIMIT`] on each side.
const RHO_STEPS: usize = 400;

/// Which component of a mediation analysis a range is about.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MediationComponent {
    /// The mediated (indirect) effect.
    Indirect,
    /// The direct effect: the total effect minus the indirect effect.
    Direct,
}

/// The regression summaries of one mediation coordinate (an outcome or a horizon): the
/// summary the indirect-effect curve is read from, and the total effect it is subtracted from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MediationComponents {
    /// The mediator-outcome sensitivity summary.
    pub mediation: MediationSensitivitySummary,
    /// The treatment coefficient of the total-effect regression, `β₁`.
    pub total_effect: f64,
}

impl MediationComponents {
    /// Validated summaries.
    ///
    /// # Errors
    ///
    /// A non-finite value, a non-positive residual standard deviation, or a residual
    /// correlation outside `(-1, 1)`.
    pub fn new(
        mediation: MediationSensitivitySummary,
        total_effect: f64,
    ) -> Result<Self, ValidationError> {
        let finite = [
            mediation.beta_mediator,
            mediation.sigma_total,
            mediation.sigma_mediator,
            mediation.rho_tilde,
            total_effect,
        ]
        .iter()
        .all(|value| value.is_finite());
        if !(finite
            && mediation.sigma_total > 0.0
            && mediation.sigma_mediator > 0.0
            && mediation.rho_tilde > -1.0
            && mediation.rho_tilde < 1.0)
        {
            return Err(not_applicable(
                "mediation components need finite summaries, positive residual standard \
                 deviations and a residual correlation inside (-1, 1)",
            ));
        }
        Ok(Self { mediation, total_effect })
    }

    /// The component's effect when the mediator-outcome error correlation is `rho`.
    ///
    /// # Errors
    ///
    /// `rho` outside `(-1, 1)`.
    pub fn effect_at(
        &self,
        component: MediationComponent,
        rho: f64,
    ) -> Result<f64, ValidationError> {
        let indirect = self.mediation.acme(rho)?;
        Ok(match component {
            MediationComponent::Indirect => indirect,
            MediationComponent::Direct => self.total_effect - indirect,
        })
    }

    /// The error correlation at which the component is exactly zero, when there is one.
    ///
    /// The indirect effect vanishes at `ρ̃`. The direct effect `β₁ - A (ρ̃ - ρ k / sqrt(1 - ρ²))`
    /// with `A = β₂ σ₁ / σ₂` and `k = sqrt(1 - ρ̃²)` vanishes at `ρ = w / sqrt(1 + w²)` with
    /// `w = (ρ̃ - β₁ / A) / k`, because `ρ / sqrt(1 - ρ²)` is a bijection of `(-1, 1)` onto the
    /// line. There is none when `A = 0`, where the mediator carries no effect at any `ρ`.
    #[must_use]
    pub fn break_even(&self, component: MediationComponent) -> Option<f64> {
        let summary = &self.mediation;
        let scale = summary.beta_mediator * summary.sigma_total / summary.sigma_mediator;
        if scale == 0.0 {
            return None;
        }
        match component {
            MediationComponent::Indirect => Some(summary.rho_tilde),
            MediationComponent::Direct => {
                let k = (1.0 - summary.rho_tilde * summary.rho_tilde).sqrt();
                let w = (summary.rho_tilde - self.total_effect / scale) / k;
                Some(w / (1.0 + w * w).sqrt())
            }
        }
    }
}

/// Fit the mediation summaries and the total effect of one outcome.
///
/// # Errors
///
/// As [`mediation_sensitivity_fit`].
pub fn mediation_components_fit(
    treatment: &[f64],
    mediator: &[f64],
    outcome: &[f64],
    covariates: &[Vec<f64>],
) -> Result<MediationComponents, ValidationError> {
    let summary = mediation_sensitivity_fit(treatment, mediator, outcome, covariates)?;
    let n = outcome.len();
    let p = 2 + covariates.len();
    let mut design = vec![1.0; n];
    design.extend_from_slice(treatment);
    for column in covariates {
        design.extend_from_slice(column);
    }
    let mut ws = LeastSquaresWorkspace::default();
    let total = FaerBackend.least_squares(&design, n, p, outcome, &mut ws)?;
    MediationComponents::new(summary, total.coefficients[1])
}

/// The mediator-outcome sensitivity of a vector of mediation coordinates, read at the weakest.
///
/// At each probed `ρ` the tested target is the smallest share of the reported component left
/// across the coordinates (`direction · effect(ρ) / |effect(0)|`, negative once a coordinate is
/// explained past zero). The break-even value is the correlation of smallest magnitude at which
/// some coordinate's component reaches zero, so the claim "every coordinate keeps its sign"
/// first fails there; it is absent when no coordinate reaches zero.
///
/// # Errors
///
/// A coordinate count other than the target's dimension, or a coordinate whose component is
/// zero at `ρ = 0` (nothing to explain away).
pub fn mediation_vector_range(
    target: RefutationTarget,
    coordinates: &[MediationComponents],
    component: MediationComponent,
) -> Result<AssumptionRange, ValidationError> {
    require_dimension(&target, coordinates.len())?;
    let mut baselines = Vec::with_capacity(coordinates.len());
    for coordinate in coordinates {
        let base = coordinate.effect_at(component, 0.0)?;
        if !(base.is_finite() && base != 0.0) {
            return Err(not_applicable(
                "a coordinate's component is zero at rho = 0: nothing to explain away",
            ));
        }
        baselines.push(base);
    }
    let mut tested = Vec::with_capacity(RHO_GRID.len());
    for &rho in &RHO_GRID {
        let mut remaining = f64::INFINITY;
        for (coordinate, base) in coordinates.iter().zip(&baselines) {
            let share = base.signum() * coordinate.effect_at(component, rho)? / base.abs();
            remaining = remaining.min(share);
        }
        tested.push(RangePoint { parameter: rho, target: remaining, test: None });
    }
    let break_even = coordinates
        .iter()
        .filter_map(|coordinate| coordinate.break_even(component))
        .min_by(|a, b| a.abs().total_cmp(&b.abs()));
    Ok(AssumptionRange::new(
        target,
        SensitivityAssumption::MediatorOutcomeConfounding,
        "rho",
        0.0,
        tested,
        break_even,
        None,
    ))
}

/// Coordinate-major posterior draws of the mediation summaries behind a vector of components.
#[derive(Clone, Debug)]
pub struct MediationPosterior<'a> {
    /// `beta_mediator[k][d]`: the treatment coefficient of the mediator regression.
    pub beta_mediator: &'a [Vec<f64>],
    /// `sigma_total[k][d]`: the residual standard deviation of the total-effect regression.
    pub sigma_total: &'a [Vec<f64>],
    /// `sigma_mediator[k][d]`: the residual standard deviation of the mediator regression.
    pub sigma_mediator: &'a [Vec<f64>],
    /// `rho_tilde[k][d]`: the correlation of the two residuals.
    pub rho_tilde: &'a [Vec<f64>],
    /// `total_effect[k][d]`: the treatment coefficient of the total-effect regression.
    pub total_effect: &'a [Vec<f64>],
}

fn quantile_of(values: &[f64], p: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    quantile_sorted(&sorted, p)
}

/// The correlation of smallest magnitude, on either side of zero, at which `f` first falls to
/// zero or below; `Some(0.0)` when it already has at zero.
fn nearest_crossing(f: impl Fn(f64) -> f64) -> Option<f64> {
    if f(0.0) <= 0.0 {
        return Some(0.0);
    }
    let mut best: Option<f64> = None;
    for side in [1.0_f64, -1.0] {
        let mut previous = 0.0_f64;
        for step in 1..=RHO_STEPS {
            let rho = side * RHO_LIMIT * step as f64 / RHO_STEPS as f64;
            if f(rho) <= 0.0 {
                let (mut inside, mut outside) = (previous, rho);
                for _ in 0..80 {
                    let middle = 0.5 * (inside + outside);
                    if f(middle) > 0.0 {
                        inside = middle;
                    } else {
                        outside = middle;
                    }
                }
                let root = 0.5 * (inside + outside);
                if best.is_none_or(|current| root.abs() < current.abs()) {
                    best = Some(root);
                }
                break;
            }
            previous = rho;
        }
    }
    best
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
/// # Errors
///
/// Misaligned or non-finite draws, fewer than two draws, a non-positive residual standard
/// deviation, a residual correlation outside `(-1, 1)`, a level outside `(0, 1)`, or a
/// coordinate whose posterior median is zero at `ρ = 0`.
pub fn mediation_posterior_range(
    target: RefutationTarget,
    posterior: &MediationPosterior<'_>,
    component: MediationComponent,
    level: f64,
) -> Result<AssumptionRange, ValidationError> {
    let k = target.dimension();
    let blocks = [
        posterior.beta_mediator,
        posterior.sigma_total,
        posterior.sigma_mediator,
        posterior.rho_tilde,
        posterior.total_effect,
    ];
    if blocks.iter().any(|block| block.len() != k) {
        return Err(not_applicable(
            "a posterior mediation range needs one column of draws per coordinate of the target",
        ));
    }
    let n = posterior.beta_mediator[0].len();
    if n < 2 || blocks.iter().flat_map(|block| block.iter()).any(|column| column.len() != n) {
        return Err(not_applicable(
            "posterior draws need at least two draws, the same number for every summary",
        ));
    }
    if blocks.iter().flat_map(|block| block.iter()).flatten().any(|value| !value.is_finite()) {
        return Err(not_applicable("a posterior draw is not finite"));
    }
    if posterior.sigma_total.iter().chain(posterior.sigma_mediator).flatten().any(|v| *v <= 0.0)
        || posterior.rho_tilde.iter().flatten().any(|v| !(*v > -1.0 && *v < 1.0))
    {
        return Err(not_applicable(
            "posterior residual standard deviations must be positive and residual correlations \
             must lie inside (-1, 1)",
        ));
    }
    if !(level > 0.0 && level < 1.0) {
        return Err(not_applicable("the credible level must lie in (0, 1)"));
    }
    // Draw `d` of coordinate `j` at correlation `rho` (inside (-1, 1) by construction).
    let effect = |j: usize, d: usize, rho: f64| -> f64 {
        let summary = MediationSensitivitySummary {
            beta_mediator: posterior.beta_mediator[j][d],
            sigma_total: posterior.sigma_total[j][d],
            sigma_mediator: posterior.sigma_mediator[j][d],
            rho_tilde: posterior.rho_tilde[j][d],
        };
        let scale = summary.beta_mediator * summary.sigma_total / summary.sigma_mediator;
        let shrink = ((1.0 - summary.rho_tilde * summary.rho_tilde) / (1.0 - rho * rho)).sqrt();
        let indirect = scale * (summary.rho_tilde - rho * shrink);
        match component {
            MediationComponent::Indirect => indirect,
            MediationComponent::Direct => posterior.total_effect[j][d] - indirect,
        }
    };
    let draws_at = |j: usize, rho: f64| -> Vec<f64> { (0..n).map(|d| effect(j, d, rho)).collect() };
    let mut direction = Vec::with_capacity(k);
    let mut base_median = Vec::with_capacity(k);
    for j in 0..k {
        let median = quantile_of(&draws_at(j, 0.0), 0.5);
        if median == 0.0 {
            return Err(not_applicable(
                "a coordinate's posterior median is zero at rho = 0: nothing to explain away",
            ));
        }
        direction.push(median.signum());
        base_median.push(median.abs());
    }
    let oriented = |j: usize, rho: f64| -> Vec<f64> {
        draws_at(j, rho).into_iter().map(|value| direction[j] * value).collect()
    };
    let mut tested = Vec::with_capacity(RHO_GRID.len());
    for &rho in &RHO_GRID {
        let mut remaining = f64::INFINITY;
        let mut kept = vec![true; n];
        for (j, &median) in base_median.iter().enumerate() {
            let values = oriented(j, rho);
            remaining = remaining.min(quantile_of(&values, 0.5) / median);
            for (flag, value) in kept.iter_mut().zip(&values) {
                *flag &= *value > 0.0;
            }
        }
        let probability = kept.iter().filter(|flag| **flag).count() as f64 / n as f64;
        tested.push(RangePoint { parameter: rho, target: remaining, test: Some(probability) });
    }
    let tail = 0.5 * (1.0 - level);
    let weakest = |p: f64, rho: f64| -> f64 {
        (0..k).map(|j| quantile_of(&oriented(j, rho), p)).fold(f64::INFINITY, f64::min)
    };
    Ok(AssumptionRange::new(
        target,
        SensitivityAssumption::MediatorOutcomeConfounding,
        "rho_posterior",
        0.0,
        tested,
        nearest_crossing(|rho| weakest(0.5, rho)),
        nearest_crossing(|rho| weakest(tail, rho)),
    ))
}

/// The setting a vector of coordinates is read in, which fixes the assumption a range relaxes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComponentModel {
    /// Components of a mediation analysis, confounded through the mediator and the outcome.
    Mediation,
    /// Parameters of a marginal structural model, confounded through the time-varying history.
    MarginalStructural,
}

impl ComponentModel {
    const fn risk_ratio(self) -> SensitivityAssumption {
        match self {
            Self::Mediation => SensitivityAssumption::MediatorOutcomeConfoundingRiskRatio,
            Self::MarginalStructural => SensitivityAssumption::SequentialExchangeabilityRiskRatio,
        }
    }

    const fn partial_r2(self) -> SensitivityAssumption {
        match self {
            Self::Mediation => SensitivityAssumption::MediatorOutcomeConfoundingPartialR2,
            Self::MarginalStructural => SensitivityAssumption::SequentialExchangeabilityPartialR2,
        }
    }
}

/// The same numbers under another assumption (the identity records the assumption).
fn rebind(range: &AssumptionRange, assumption: SensitivityAssumption) -> AssumptionRange {
    AssumptionRange::new(
        range.target.clone(),
        assumption,
        &range.parameter,
        range.null_parameter,
        range.tested.to_vec(),
        range.break_even,
        range.break_even_significance,
    )
}

/// E-values of every component or parameter, read at the weakest, bound to `model`'s assumption.
///
/// # Errors
///
/// As [`evalue_vector_range`].
pub fn component_evalue_range(
    target: RefutationTarget,
    coordinates: &[EffectInterval],
    scale: EffectScale,
    model: ComponentModel,
) -> Result<(Vec<EValueBounds>, AssumptionRange), ValidationError> {
    let (bounds, range) = evalue_vector_range(target, coordinates, scale)?;
    Ok((bounds, rebind(&range, model.risk_ratio())))
}

/// The E-value analysis of a posterior over components or parameters, bound to `model`.
///
/// # Errors
///
/// As [`evalue_posterior_range`].
pub fn component_evalue_posterior_range(
    target: RefutationTarget,
    draws: &[Vec<f64>],
    scale: EffectScale,
    level: f64,
    model: ComponentModel,
) -> Result<(PosteriorEValue, AssumptionRange), ValidationError> {
    let (summary, range) = evalue_posterior_range(target, draws, scale, level)?;
    Ok((summary, rebind(&range, model.risk_ratio())))
}

/// The partial-R² analysis of linear-adjustment summaries of components or parameters, bound to
/// `model`. For a mediation component the summaries are those of the mediator-outcome path
/// coefficient (the indirect effect is that coefficient times the unconfounded
/// treatment-mediator coefficient, so its share left and robustness value are the path's); for
/// an MSM they come from [`msm_parameter_fit`].
///
/// # Errors
///
/// As [`ovb_vector_range`].
pub fn component_ovb_range(
    target: RefutationTarget,
    summaries: &[LinearAdjustmentSummary],
    q: f64,
    alpha: f64,
    r2yz_extreme: f64,
    model: ComponentModel,
) -> Result<AssumptionRange, ValidationError> {
    let range = ovb_vector_range(target, summaries, q, alpha, r2yz_extreme)?;
    Ok(rebind(&range, model.partial_r2()))
}

/// The partial-R² analysis of a posterior over components or parameters, bound to `model`.
///
/// # Errors
///
/// As [`ovb_posterior_range`].
pub fn component_ovb_posterior_range(
    target: RefutationTarget,
    posterior: &OvbPosterior<'_>,
    level: f64,
    r2yz_extreme: f64,
    model: ComponentModel,
) -> Result<AssumptionRange, ValidationError> {
    let range = ovb_posterior_range(target, posterior, level, r2yz_extreme)?;
    Ok(rebind(&range, model.partial_r2()))
}

/// The three regression coefficients a mediation analysis rests on, each as a
/// linear-adjustment summary: `T -> M` in `M ~ T + X`, `M -> Y` in `Y ~ M + T + X` and
/// `T -> Y` (the direct effect) in `Y ~ T + M + X`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MediationPaths {
    /// Treatment coefficient of the mediator regression.
    pub treatment_mediator: LinearAdjustmentSummary,
    /// Mediator coefficient of the outcome regression.
    pub mediator_outcome: LinearAdjustmentSummary,
    /// Treatment coefficient of the outcome regression (the direct path).
    pub treatment_outcome: LinearAdjustmentSummary,
}

/// A path of a mediation analysis a partial-R² range can be about.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MediationPath {
    /// The treatment-mediator coefficient, confounded by a common cause of treatment and mediator.
    TreatmentMediator,
    /// The mediator-outcome coefficient, confounded by a common cause of mediator and outcome.
    MediatorOutcome,
    /// The treatment-outcome (direct) coefficient, confounded by a common cause of treatment and outcome.
    TreatmentOutcome,
    /// The indirect effect, the product of the first two coefficients, with both paths confounded.
    Indirect,
}

/// Fit the three path summaries of one outcome by least squares.
///
/// # Errors
///
/// As [`linear_adjustment_fit`] for each of the three regressions.
pub fn mediation_paths_fit(
    treatment: &[f64],
    mediator: &[f64],
    outcome: &[f64],
    covariates: &[Vec<f64>],
) -> Result<MediationPaths, ValidationError> {
    let with = |extra: &[f64]| -> Vec<Vec<f64>> {
        let mut columns = vec![extra.to_vec()];
        columns.extend(covariates.iter().cloned());
        columns
    };
    Ok(MediationPaths {
        treatment_mediator: linear_adjustment_fit(mediator, treatment, covariates)?.summary,
        mediator_outcome: linear_adjustment_fit(outcome, mediator, &with(treatment))?.summary,
        treatment_outcome: linear_adjustment_fit(outcome, treatment, &with(mediator))?.summary,
    })
}

/// Share of a coefficient left after a worst-case omitted-variable bias, floored at zero:
/// `max(0, direction · adjusted / |estimate|)`. With `limit` the share of the adjusted
/// `1 - alpha` limit nearest zero.
fn share_left(
    summary: &LinearAdjustmentSummary,
    r2dz: f64,
    r2yz: f64,
    limit_alpha: Option<f64>,
) -> f64 {
    let direction = if summary.estimate >= 0.0 { 1.0 } else { -1.0 };
    let adjusted = summary.adjusted_estimate(r2dz, r2yz);
    let value = match limit_alpha {
        None => direction * adjusted,
        Some(alpha) => {
            let z = student_t_ppf(1.0 - alpha / 2.0, summary.dof - 1.0);
            direction * adjusted - z * summary.adjusted_standard_error(r2dz, r2yz)
        }
    };
    let share = value / summary.estimate.abs().max(f64::MIN_POSITIVE);
    if share.is_finite() { share.max(0.0) } else { f64::NAN }
}

/// Smallest equal partial R² in `(0, 1)` at which `share(r)` falls to `floor` or below, by
/// bisection (`share` is non-increasing in `r`); `None` when it never does.
fn smallest_equal_r2(
    share: impl Fn(f64) -> f64,
    floor: f64,
) -> Result<Option<f64>, ValidationError> {
    let checked = |r| {
        let value = share(r);
        if value.is_finite() {
            Ok(value)
        } else {
            Err(not_applicable("mediation omitted-variable-bias share is not finite"))
        }
    };
    if checked(0.0)? <= floor {
        return Ok(Some(0.0));
    }
    let upper = 1.0 - 1e-12;
    if checked(upper)? > floor {
        return Ok(None);
    }
    let (mut inside, mut outside) = (0.0_f64, upper);
    for _ in 0..100 {
        let middle = 0.5 * (inside + outside);
        if checked(middle)? > floor {
            inside = middle;
        } else {
            outside = middle;
        }
    }
    Ok(Some(outside))
}

/// The partial-R² range of one path of a vector of mediation coordinates (outcomes or
/// horizons), read at the weakest coordinate and bound to the assumption of that path.
///
/// A single path is the analysis of [`ovb_vector_range`] on that coefficient. The indirect
/// effect is the product of the treatment-mediator and mediator-outcome coefficients, each
/// moving toward zero by its own worst-case bias `se · sqrt(dof) · sqrt(R²yz R²dz / (1 - R²dz))`
/// when both are confounded at the tested strength: its share left is the product of the two
/// paths' shares (each floored at zero), the test value the product of the shares of their
/// adjusted `1 - alpha` limits, and the break-even values are the smallest equal partial R²
/// (`R²dz = R²yz`) at which the product of the shares falls to `1 - q`, for the estimate and for
/// the limits. At `q = 1` the product vanishes when either path does, so the break-even is the
/// smaller of the two paths' robustness values.
///
/// # Errors
///
/// An empty coordinate list, a count other than the target's dimension, `q` not positive,
/// `alpha` outside `(0, 1)` or `r2yz_extreme` outside `(0, 1]`.
pub fn mediation_path_ovb_range(
    target: RefutationTarget,
    coordinates: &[MediationPaths],
    path: MediationPath,
    q: f64,
    alpha: f64,
    r2yz_extreme: f64,
) -> Result<AssumptionRange, ValidationError> {
    require_dimension(&target, coordinates.len())?;
    let select = |paths: &MediationPaths| match path {
        MediationPath::TreatmentMediator | MediationPath::Indirect => paths.treatment_mediator,
        MediationPath::MediatorOutcome => paths.mediator_outcome,
        MediationPath::TreatmentOutcome => paths.treatment_outcome,
    };
    if path != MediationPath::Indirect {
        let summaries: Vec<LinearAdjustmentSummary> = coordinates.iter().map(select).collect();
        let range = ovb_vector_range(target, &summaries, q, alpha, r2yz_extreme)?;
        let assumption = match path {
            MediationPath::TreatmentMediator => {
                SensitivityAssumption::TreatmentMediatorConfoundingPartialR2
            }
            MediationPath::TreatmentOutcome => {
                SensitivityAssumption::TreatmentOutcomeConfoundingPartialR2
            }
            _ => SensitivityAssumption::MediatorOutcomeConfoundingPartialR2,
        };
        return Ok(rebind(&range, assumption));
    }
    if !(q > 0.0
        && q.is_finite()
        && alpha > 0.0
        && alpha < 1.0
        && r2yz_extreme > 0.0
        && r2yz_extreme <= 1.0)
    {
        return Err(not_applicable(
            "q must be positive, alpha must lie in (0, 1) and the extreme outcome partial R2 in \
             (0, 1]",
        ));
    }
    for paths in coordinates {
        for summary in [&paths.treatment_mediator, &paths.mediator_outcome] {
            summary.validate()?;
            if !student_t_ppf(1.0 - alpha / 2.0, summary.dof - 1.0).is_finite() {
                return Err(not_applicable(
                    "mediation omitted-variable-bias critical value is not finite",
                ));
            }
        }
    }
    let product = |paths: &MediationPaths, r2dz: f64, r2yz: f64, limit: Option<f64>| {
        share_left(&paths.treatment_mediator, r2dz, r2yz, limit)
            * share_left(&paths.mediator_outcome, r2dz, r2yz, limit)
    };
    let tested: Vec<RangePoint> = crate::omitted_variable_bias::R2DZ_GRID
        .iter()
        .map(|&r2dz| {
            let mut point = f64::INFINITY;
            let mut limit = f64::INFINITY;
            for paths in coordinates {
                let next_point = product(paths, r2dz, r2yz_extreme, None);
                let next_limit = product(paths, r2dz, r2yz_extreme, Some(alpha));
                if !next_point.is_finite() || !next_limit.is_finite() {
                    return Err(not_applicable(
                        "mediation omitted-variable-bias range is not finite",
                    ));
                }
                point = point.min(next_point);
                limit = limit.min(next_limit);
            }
            Ok(RangePoint { parameter: r2dz, target: point, test: Some(limit) })
        })
        .collect::<Result<Vec<_>, ValidationError>>()?;
    let weakest = |limit: Option<f64>| -> Result<Option<f64>, ValidationError> {
        let mut result = f64::INFINITY;
        for paths in coordinates {
            let Some(value) = smallest_equal_r2(|r| product(paths, r, r, limit), 1.0 - q.min(1.0))?
            else {
                return Ok(None);
            };
            result = result.min(value);
        }
        Ok(Some(result))
    };
    Ok(AssumptionRange::new(
        target,
        SensitivityAssumption::IndirectEffectConfoundingPartialR2,
        "partial_r2",
        0.0,
        tested,
        weakest(None)?,
        weakest(Some(alpha))?,
    ))
}

/// Inverse of a small symmetric matrix by Gauss-Jordan elimination with partial pivoting.
fn invert(matrix: &[f64], p: usize) -> Option<Vec<f64>> {
    let mut a = matrix.to_vec();
    let mut inverse = vec![0.0; p * p];
    for i in 0..p {
        inverse[i * p + i] = 1.0;
    }
    for column in 0..p {
        let pivot_row = (column..p)
            .max_by(|&r, &s| a[r * p + column].abs().total_cmp(&a[s * p + column].abs()))?;
        if a[pivot_row * p + column].abs() < 1e-12 {
            return None;
        }
        if pivot_row != column {
            for k in 0..p {
                a.swap(column * p + k, pivot_row * p + k);
                inverse.swap(column * p + k, pivot_row * p + k);
            }
        }
        let pivot = a[column * p + column];
        for k in 0..p {
            a[column * p + k] /= pivot;
            inverse[column * p + k] /= pivot;
        }
        for row in 0..p {
            if row == column {
                continue;
            }
            let factor = a[row * p + column];
            if factor == 0.0 {
                continue;
            }
            for k in 0..p {
                a[row * p + k] -= factor * a[column * p + k];
                inverse[row * p + k] -= factor * inverse[column * p + k];
            }
        }
    }
    Some(inverse)
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
/// # Errors
///
/// Misaligned or non-finite inputs, negative weights or none positive, a rank-deficient design,
/// or fewer than two effective residual degrees of freedom.
pub fn msm_parameter_fit(
    outcome: &[f64],
    exposures: &[Vec<f64>],
    weights: &[f64],
) -> Result<Vec<LinearAdjustmentSummary>, ValidationError> {
    let n = outcome.len();
    let p = 1 + exposures.len();
    if exposures.is_empty() || weights.len() != n || exposures.iter().any(|c| c.len() != n) {
        return Err(not_applicable(
            "an MSM fit needs the outcome, at least one exposure column and one weight per row",
        ));
    }
    if outcome.iter().chain(weights).chain(exposures.iter().flatten()).any(|v| !v.is_finite())
        || weights.iter().any(|w| *w < 0.0)
    {
        return Err(not_applicable("an MSM fit needs finite values and non-negative weights"));
    }
    let positive = weights.iter().filter(|w| **w > 0.0).count();
    let total: f64 = weights.iter().sum();
    let total_squares: f64 = weights.iter().map(|w| w * w).sum();
    if positive <= p + 1 || total <= 0.0 {
        return Err(not_applicable(
            "an MSM fit needs more positively weighted rows than parameters",
        ));
    }
    let effective = total * total / total_squares;
    let dof = effective - p as f64;
    if dof < 2.0 {
        return Err(not_applicable(
            "the effective sample size of the weights leaves fewer than two residual degrees of \
             freedom",
        ));
    }
    let root: Vec<f64> = weights.iter().map(|w| w.sqrt()).collect();
    let mut design: Vec<f64> = root.clone();
    for column in exposures {
        design.extend(column.iter().zip(&root).map(|(x, r)| x * r));
    }
    let scaled_outcome: Vec<f64> = outcome.iter().zip(&root).map(|(y, r)| y * r).collect();
    let mut ws = LeastSquaresWorkspace::default();
    let fit = FaerBackend.least_squares(&design, n, p, &scaled_outcome, &mut ws)?;
    if fit.rank < p {
        return Err(not_applicable("an MSM fit needs a full-rank weighted design"));
    }
    let beta = fit.coefficients;
    // Row `i` of the unscaled design: intercept, then the exposure columns.
    let row = |i: usize| -> Vec<f64> {
        std::iter::once(1.0).chain(exposures.iter().map(|column| column[i])).collect()
    };
    let mut bread = vec![0.0; p * p];
    let mut meat = vec![0.0; p * p];
    for i in 0..n {
        let w = weights[i];
        if w <= 0.0 {
            continue;
        }
        let x = row(i);
        let residual = outcome[i] - x.iter().zip(&beta).map(|(a, b)| a * b).sum::<f64>();
        for a in 0..p {
            for b in 0..p {
                bread[a * p + b] += w * x[a] * x[b];
                meat[a * p + b] += w * w * residual * residual * x[a] * x[b];
            }
        }
    }
    let inverse =
        invert(&bread, p).ok_or_else(|| not_applicable("the weighted design is singular"))?;
    let correction = positive as f64 / (positive - p) as f64;
    let mut summaries = Vec::with_capacity(exposures.len());
    for j in 1..p {
        let mut variance = 0.0;
        for a in 0..p {
            for b in 0..p {
                variance += inverse[j * p + a] * meat[a * p + b] * inverse[b * p + j];
            }
        }
        let standard_error = (correction * variance).sqrt();
        summaries.push(LinearAdjustmentSummary::new(beta[j], standard_error, dof)?);
    }
    Ok(summaries)
}

/// The treatment and stabilized-weight models of one time point of an MSM.
///
/// The stabilized weight of a subject is the product over time points of
/// `P(A_k | numerator covariates) / P(A_k | denominator covariates)`, each a logistic model with
/// an intercept; `numerator` may be empty (marginal treatment probability).
#[derive(Clone, Debug, PartialEq)]
pub struct MsmWeightPeriod {
    /// Treatment taken, coded 0 or 1, one per subject.
    pub treatment: Vec<f64>,
    /// Covariate columns of the numerator model.
    pub numerator: Vec<Vec<f64>>,
    /// Covariate columns of the denominator model (the measured history).
    pub denominator: Vec<Vec<f64>>,
}

/// The MSM parameters with the cost of estimating the weights inside their standard errors.
#[derive(Clone, Debug, PartialEq)]
pub struct MsmRefitFit {
    /// The parameters with the robust standard error that holds the fitted weights fixed.
    pub fixed_weights: Vec<LinearAdjustmentSummary>,
    /// The parameters with the standard deviation of a unit bootstrap that refits every
    /// weight model in each replicate, so the weight-model estimation is inside it.
    pub refit: Vec<LinearAdjustmentSummary>,
    /// Bootstrap replicates that succeeded, and those that failed.
    pub replicates: (usize, usize),
}

/// Fewest bootstrap replicates of the refit standard error.
pub const MIN_MSM_REPLICATES: usize = 99;

/// `SplitMix64` stream of the weight-refit bootstrap.
struct SplitMix(u64);

impl SplitMix {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    #[allow(clippy::cast_possible_truncation, reason = "the index is below `n`")]
    fn index(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// Logistic coefficients (intercept first) of `y` on `columns` under case weights; `None` when
/// the fit separates or does not converge.
fn weighted_logistic(columns: &[Vec<f64>], y: &[f64], weights: &[f64]) -> Option<Vec<f64>> {
    let n = y.len();
    let p = 1 + columns.len();
    let row = |i: usize| -> Vec<f64> {
        std::iter::once(1.0).chain(columns.iter().map(|column| column[i])).collect()
    };
    let mut beta = vec![0.0; p];
    for _ in 0..60 {
        let mut information = vec![0.0; p * p];
        let mut gradient = vec![0.0; p];
        for i in 0..n {
            let w = weights[i];
            if w <= 0.0 {
                continue;
            }
            let x = row(i);
            let eta: f64 = x.iter().zip(&beta).map(|(a, b)| a * b).sum();
            let mu = 1.0 / (1.0 + (-eta).exp());
            for a in 0..p {
                gradient[a] += w * (y[i] - mu) * x[a];
                for b in 0..p {
                    information[a * p + b] += w * mu * (1.0 - mu) * x[a] * x[b];
                }
            }
        }
        for a in 0..p {
            information[a * p + a] += 1e-8;
        }
        let inverse = invert(&information, p)?;
        let step: Vec<f64> =
            (0..p).map(|a| (0..p).map(|b| inverse[a * p + b] * gradient[b]).sum()).collect();
        let largest = step.iter().fold(0.0_f64, |acc, value| acc.max(value.abs()));
        for (coefficient, delta) in beta.iter_mut().zip(&step) {
            *coefficient += delta;
        }
        if !largest.is_finite() || beta.iter().any(|value| value.abs() > 30.0) {
            return None;
        }
        if largest < 1e-10 {
            return Some(beta);
        }
    }
    None
}

/// Probability of the taken treatment under a fitted logistic model, per subject.
fn taken_probability(columns: &[Vec<f64>], y: &[f64], beta: &[f64]) -> Vec<f64> {
    (0..y.len())
        .map(|i| {
            let eta: f64 = beta[0]
                + columns.iter().zip(&beta[1..]).map(|(column, b)| column[i] * b).sum::<f64>();
            let mu = 1.0 / (1.0 + (-eta).exp());
            if y[i] > 0.5 { mu } else { 1.0 - mu }
        })
        .collect()
}

/// Stabilized weights of every subject with both models of every period refitted under
/// `unit_weights`; `None` when a model does not converge.
fn stabilized_weights(periods: &[MsmWeightPeriod], unit_weights: &[f64]) -> Option<Vec<f64>> {
    let n = unit_weights.len();
    let mut weights = vec![1.0; n];
    for period in periods {
        let numerator = weighted_logistic(&period.numerator, &period.treatment, unit_weights)?;
        let denominator = weighted_logistic(&period.denominator, &period.treatment, unit_weights)?;
        let top = taken_probability(&period.numerator, &period.treatment, &numerator);
        let bottom = taken_probability(&period.denominator, &period.treatment, &denominator);
        for ((weight, t), b) in weights.iter_mut().zip(top).zip(bottom) {
            *weight *= t / b;
        }
    }
    weights.iter().all(|w| w.is_finite()).then_some(weights)
}

/// The MSM parameters whose standard error carries the cost of estimating the weights.
///
/// The weights are the stabilized inverse-probability weights of `periods`. The fixed-weight
/// robust standard error of [`msm_parameter_fit`] holds them as known; here a unit bootstrap
/// refits every numerator and denominator model and the weighted regression in each replicate
/// (a replicate resamples subjects with replacement, as multiplicities on both fits), and the
/// reported standard error of each parameter is the standard deviation of its replicates, with
/// the effective degrees of freedom of the full-data weights. Ignoring the estimation of a
/// denominator model is conservative, so the refit error is the one that matches the
/// repeated-sampling spread; `fixed_weights` is reported beside it.
///
/// # Errors
///
/// Misaligned inputs or a treatment not coded 0/1, fewer than [`MIN_MSM_REPLICATES`] replicates,
/// a weight model that does not converge on the full data, more than five percent failed
/// replicates, or any error of [`msm_parameter_fit`].
#[allow(
    clippy::float_cmp,
    reason = "treatment labels are validated against the exact binary category codes zero and one"
)]
pub fn msm_parameter_fit_refitting_weights(
    outcome: &[f64],
    exposures: &[Vec<f64>],
    periods: &[MsmWeightPeriod],
    replicates: usize,
    seed: u64,
) -> Result<MsmRefitFit, ValidationError> {
    let n = outcome.len();
    if periods.is_empty()
        || periods.iter().any(|period| {
            period.treatment.len() != n
                || period.treatment.iter().any(|a| *a != 0.0 && *a != 1.0)
                || period.numerator.iter().chain(&period.denominator).any(|c| c.len() != n)
                || period
                    .numerator
                    .iter()
                    .chain(&period.denominator)
                    .flatten()
                    .any(|v| !v.is_finite())
        })
    {
        return Err(not_applicable(
            "every weight period needs a 0/1 treatment and finite covariates, one value per row",
        ));
    }
    if replicates < MIN_MSM_REPLICATES {
        return Err(not_applicable(
            "the weight-refit standard error needs at least 99 bootstrap replicates",
        ));
    }
    let ones = vec![1.0; n];
    let point = stabilized_weights(periods, &ones)
        .ok_or_else(|| not_applicable("a weight model did not converge on the full data"))?;
    let fixed = msm_parameter_fit(outcome, exposures, &point)?;
    let mut rng = SplitMix(seed);
    let mut columns: Vec<Vec<f64>> = vec![Vec::with_capacity(replicates); fixed.len()];
    let mut failed = 0_usize;
    let mut counts = vec![0.0; n];
    for _ in 0..replicates {
        counts.iter_mut().for_each(|c| *c = 0.0);
        for _ in 0..n {
            counts[rng.index(n)] += 1.0;
        }
        let replicate = stabilized_weights(periods, &counts).and_then(|weights| {
            let total: Vec<f64> = weights.iter().zip(&counts).map(|(w, c)| w * c).collect();
            msm_parameter_fit(outcome, exposures, &total).ok()
        });
        match replicate {
            Some(summaries) => {
                for (column, summary) in columns.iter_mut().zip(&summaries) {
                    column.push(summary.estimate);
                }
            }
            None => failed += 1,
        }
    }
    if failed as f64 > 0.05 * replicates as f64 {
        return Err(not_applicable(
            "more than five percent of the weight-refit bootstrap replicates failed (separated \
             weight models or a degenerate resample)",
        ));
    }
    let mut refit = Vec::with_capacity(fixed.len());
    for (summary, column) in fixed.iter().zip(&columns) {
        let m = column.len() as f64;
        let mean = column.iter().sum::<f64>() / m;
        let sd = (column.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (m - 1.0)).sqrt();
        refit.push(LinearAdjustmentSummary::new(summary.estimate, sd, summary.dof)?);
    }
    Ok(MsmRefitFit { fixed_weights: fixed, refit, replicates: (replicates - failed, failed) })
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    reason = "tests locate exact values copied verbatim from the declared sensitivity grids"
)]
mod tests {
    use super::*;
    use crate::evalue_scales::evalue_range;
    use crate::target_native::{TargetFamily, TargetQuadrant};

    fn target(family: TargetFamily, quadrant: TargetQuadrant, labels: &[&str]) -> RefutationTarget {
        RefutationTarget::new(family, quadrant, labels.iter().copied(), [5_u8; 32]).unwrap()
    }

    /// Treatment, mediator, two outcomes and a covariate over deterministic periodic noise.
    #[allow(clippy::type_complexity, reason = "one fixture of five aligned columns")]
    fn data() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let (mut t, mut m, mut y1, mut y2, mut x) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for i in 0..120 {
            let a = f64::from(i);
            let xi = (a * 0.3).sin();
            let ti = f64::from(i % 2);
            let mi = 0.8 * ti + 0.5 * xi + 0.4 * (a * 1.7).sin();
            t.push(ti);
            m.push(mi);
            y1.push(0.3 * ti + 1.2 * mi - 0.7 * xi + 0.5 * (a * 0.9).cos());
            y2.push(-0.4 * ti + 0.5 * mi + 0.2 * xi + 0.3 * (a * 2.3).sin());
            x.push(xi);
        }
        (t, m, y1, y2, x)
    }

    /// Coefficients of `Y ~ 1 + T + M + X` by least squares, the structural outcome model.
    fn structural(t: &[f64], m: &[f64], x: &[f64], y: &[f64]) -> Vec<f64> {
        let n = y.len();
        let mut design = vec![1.0; n];
        design.extend_from_slice(t);
        design.extend_from_slice(m);
        design.extend_from_slice(x);
        FaerBackend
            .least_squares(&design, n, 4, y, &mut LeastSquaresWorkspace::default())
            .unwrap()
            .coefficients
    }

    #[test]
    fn the_components_at_zero_correlation_are_the_structural_product_and_direct_coefficient() {
        let (t, m, y1, _, x) = data();
        let components = mediation_components_fit(&t, &m, &y1, &[x.clone()]).unwrap();
        let beta = structural(&t, &m, &x, &y1);
        // The mediator coefficient of the mediator regression times the outcome-model slope.
        let indirect = components.effect_at(MediationComponent::Indirect, 0.0).unwrap();
        assert!((indirect - components.mediation.beta_mediator * beta[2]).abs() < 1e-9);
        // Total = direct + indirect for least squares with a common covariate set.
        let direct = components.effect_at(MediationComponent::Direct, 0.0).unwrap();
        assert!((direct - beta[1]).abs() < 1e-9, "{direct} vs {}", beta[1]);
        assert!((direct + indirect - components.total_effect).abs() < 1e-12);
    }

    #[test]
    fn each_component_vanishes_at_its_closed_form_break_even() {
        let (t, m, y1, y2, x) = data();
        for outcome in [&y1, &y2] {
            let components = mediation_components_fit(&t, &m, outcome, &[x.clone()]).unwrap();
            for component in [MediationComponent::Indirect, MediationComponent::Direct] {
                let rho = components.break_even(component).expect("a mediator effect exists");
                assert!(rho.abs() < 1.0);
                let at = components.effect_at(component, rho).unwrap();
                assert!(at.abs() < 1e-9, "{component:?}: effect {at} at rho {rho}");
            }
        }
    }

    #[test]
    fn the_vector_range_is_read_at_the_weakest_coordinate() {
        let (t, m, y1, y2, x) = data();
        let a = mediation_components_fit(&t, &m, &y1, &[x.clone()]).unwrap();
        let b = mediation_components_fit(&t, &m, &y2, &[x]).unwrap();
        let outcomes = target(
            TargetFamily::Mediation,
            TargetQuadrant::STATIC_FREQUENTIST,
            &["indirect@y1", "indirect@y2"],
        );
        for component in [MediationComponent::Indirect, MediationComponent::Direct] {
            let range = mediation_vector_range(outcomes.clone(), &[a, b], component).unwrap();
            let expected = [a, b]
                .iter()
                .filter_map(|c| c.break_even(component))
                .min_by(|p, q| p.abs().total_cmp(&q.abs()));
            assert_eq!(range.break_even, expected);
            for point in range.tested.iter() {
                let share = |c: &MediationComponents| {
                    let base = c.effect_at(component, 0.0).unwrap();
                    base.signum() * c.effect_at(component, point.parameter).unwrap() / base.abs()
                };
                let want = share(&a).min(share(&b));
                assert!((point.target - want).abs() < 1e-12, "rho {}", point.parameter);
            }
            let zero = range.tested.iter().find(|p| p.parameter == 0.0).unwrap();
            assert!((zero.target - 1.0).abs() < 1e-12);
            assert!(range.bound_to(&outcomes).is_ok());
        }
        // A single coordinate reproduces the scalar curve of the indirect effect.
        let one = target(TargetFamily::Mediation, TargetQuadrant::STATIC_FREQUENTIST, &["only"]);
        let vector =
            mediation_vector_range(one.clone(), &[a], MediationComponent::Indirect).unwrap();
        let scalar = crate::mediation_range(one, &a.mediation).unwrap();
        assert_eq!(vector.break_even, scalar.break_even);
        for (v, s) in vector.tested.iter().zip(scalar.tested.iter()) {
            assert!((v.target - s.target / a.mediation.acme(0.0).unwrap()).abs() < 1e-12);
        }
        // The wrong number of coordinates is refused.
        assert!(mediation_vector_range(outcomes, &[a], MediationComponent::Indirect).is_err());
    }

    fn spread(center: f64, half: f64, n: usize) -> Vec<f64> {
        (0..n).map(|i| center + half * (2.0 * i as f64 / (n - 1) as f64 - 1.0)).collect()
    }

    #[test]
    fn the_posterior_range_moves_every_draw_through_the_formula() {
        let n = 201;
        let horizons = target(
            TargetFamily::Mediation,
            TargetQuadrant::STATIC_BAYESIAN,
            &["indirect@h1", "indirect@h2"],
        );
        // The indirect effect of every draw vanishes at that draw's residual correlation, so a
        // common correlation gives the posterior median its exact zero there.
        let beta = vec![spread(0.8, 0.1, n), spread(0.5, 0.1, n)];
        let sigma_total = vec![vec![1.1; n], vec![0.9; n]];
        let sigma_mediator = vec![vec![0.7; n], vec![0.6; n]];
        let rho_tilde = vec![vec![0.2; n], vec![0.35; n]];
        let total = vec![spread(1.4, 0.05, n), spread(0.4, 0.05, n)];
        let posterior = MediationPosterior {
            beta_mediator: &beta,
            sigma_total: &sigma_total,
            sigma_mediator: &sigma_mediator,
            rho_tilde: &rho_tilde,
            total_effect: &total,
        };
        let range = mediation_posterior_range(
            horizons.clone(),
            &posterior,
            MediationComponent::Indirect,
            0.9,
        )
        .unwrap();
        // The weakest coordinate fails first, at the smaller residual correlation.
        assert!((range.break_even.unwrap() - 0.2).abs() < 1e-6, "{:?}", range.break_even);
        // The credible limit reaches zero no later than the median.
        assert!(
            range.break_even_significance.unwrap().abs() <= range.break_even.unwrap().abs() + 1e-9
        );
        let zero = range.tested.iter().find(|p| p.parameter == 0.0).unwrap();
        assert!((zero.target - 1.0).abs() < 1e-9);
        assert!((zero.test.unwrap() - 1.0).abs() < 1e-12);
        // Retention against an independent count of draws still on their side.
        for point in range.tested.iter() {
            let rho = point.parameter;
            let kept = (0..n)
                .filter(|&d| {
                    (0..2).all(|j| {
                        let scale = beta[j][d] * sigma_total[j][d] / sigma_mediator[j][d];
                        let shrink = ((1.0 - rho_tilde[j][d].powi(2)) / (1.0 - rho * rho)).sqrt();
                        scale * (rho_tilde[j][d] - rho * shrink) > 0.0
                    })
                })
                .count();
            assert!((point.test.unwrap() - kept as f64 / n as f64).abs() < 1e-12, "rho {rho}");
        }
        // Past the break-even correlation no draw keeps the sign; below it, all do.
        let beyond = range.tested.iter().find(|p| p.parameter == 0.3).unwrap();
        assert!(beyond.test.unwrap() < 1e-12 && beyond.target < 0.0);
        let before = range.tested.iter().find(|p| p.parameter == 0.1).unwrap();
        assert!(before.test.unwrap() > 0.999);
        // The direct effect uses the total effect of each draw; misaligned draws are refused.
        let direct = mediation_posterior_range(
            horizons.clone(),
            &posterior,
            MediationComponent::Direct,
            0.9,
        )
        .unwrap();
        assert!(direct.break_even.is_some());
        let short = vec![vec![0.2; n], vec![0.35; n - 1]];
        let bad = MediationPosterior { rho_tilde: &short, ..posterior.clone() };
        assert!(
            mediation_posterior_range(horizons, &bad, MediationComponent::Indirect, 0.9).is_err()
        );
    }

    #[test]
    fn evalue_and_partial_r2_ranges_bind_to_the_assumption_of_their_setting() {
        let estimate = EffectInterval::Ratio { estimate: 3.9, lower: Some(1.8), upper: Some(8.7) };
        let one = target(TargetFamily::Mediation, TargetQuadrant::STATIC_FREQUENTIST, &["e"]);
        let (_, plain) = evalue_range(one.clone(), estimate, EffectScale::RiskRatio).unwrap();
        let (_, mediation) = component_evalue_range(
            one.clone(),
            &[estimate],
            EffectScale::RiskRatio,
            ComponentModel::Mediation,
        )
        .unwrap();
        let (_, msm) = component_evalue_range(
            one.clone(),
            &[estimate],
            EffectScale::RiskRatio,
            ComponentModel::MarginalStructural,
        )
        .unwrap();
        assert_eq!(mediation.tested, plain.tested);
        assert_eq!(mediation.break_even, plain.break_even);
        assert_eq!(
            mediation.assumption,
            SensitivityAssumption::MediatorOutcomeConfoundingRiskRatio
        );
        assert_eq!(msm.assumption, SensitivityAssumption::SequentialExchangeabilityRiskRatio);
        // The assumption is part of the identity of a range.
        assert_ne!(mediation.identity, plain.identity);
        assert_ne!(mediation.identity, msm.identity);
        let summary = LinearAdjustmentSummary::new(0.0973, 0.0233, 783.0).unwrap();
        let ovb = component_ovb_range(
            one.clone(),
            &[summary],
            1.0,
            0.05,
            1.0,
            ComponentModel::MarginalStructural,
        )
        .unwrap();
        let direct = ovb_vector_range(one, &[summary], 1.0, 0.05, 1.0).unwrap();
        assert_eq!(ovb.tested, direct.tested);
        assert_eq!(ovb.assumption, SensitivityAssumption::SequentialExchangeabilityPartialR2);
        assert_eq!(ovb.break_even, direct.break_even);
    }

    #[test]
    fn each_mediation_path_has_its_own_robustness_value_and_assumption() {
        let paths = MediationPaths {
            treatment_mediator: LinearAdjustmentSummary::new(0.0973, 0.0233, 783.0).unwrap(),
            mediator_outcome: LinearAdjustmentSummary::new(-0.31, 0.06, 500.0).unwrap(),
            treatment_outcome: LinearAdjustmentSummary::new(0.2, 0.05, 500.0).unwrap(),
        };
        let one = target(TargetFamily::Mediation, TargetQuadrant::STATIC_FREQUENTIST, &["y"]);
        for (path, assumption, summary) in [
            (
                MediationPath::TreatmentMediator,
                SensitivityAssumption::TreatmentMediatorConfoundingPartialR2,
                paths.treatment_mediator,
            ),
            (
                MediationPath::MediatorOutcome,
                SensitivityAssumption::MediatorOutcomeConfoundingPartialR2,
                paths.mediator_outcome,
            ),
            (
                MediationPath::TreatmentOutcome,
                SensitivityAssumption::TreatmentOutcomeConfoundingPartialR2,
                paths.treatment_outcome,
            ),
        ] {
            let range =
                mediation_path_ovb_range(one.clone(), &[paths], path, 1.0, 0.05, 1.0).unwrap();
            assert_eq!(range.assumption, assumption);
            // Closed form: RV_1 = (sqrt(f^4 + 4 f^2) - f^2) / 2 with f = |t| / sqrt(dof).
            let f = (summary.estimate / summary.standard_error).abs() / summary.dof.sqrt();
            let rv = 0.5 * ((f.powi(4) + 4.0 * f * f).sqrt() - f * f);
            assert!((range.break_even.unwrap() - rv).abs() < 1e-12, "{path:?}");
        }
        // The treatment-mediator robustness value is the published Darfur value.
        let darfur = mediation_path_ovb_range(
            one,
            &[paths],
            MediationPath::TreatmentMediator,
            1.0,
            0.05,
            1.0,
        )
        .unwrap();
        assert!((darfur.break_even.unwrap() - 0.1388).abs() < 1e-3);
    }

    #[test]
    fn indirect_ovb_refuses_invalid_public_summary() {
        let valid = LinearAdjustmentSummary::new(1.0, 0.2, 10.0).unwrap();
        let bad = LinearAdjustmentSummary { standard_error: f64::NAN, ..valid };
        let paths = MediationPaths {
            treatment_mediator: valid,
            mediator_outcome: bad,
            treatment_outcome: valid,
        };
        let one = target(TargetFamily::Mediation, TargetQuadrant::STATIC_FREQUENTIST, &["y"]);
        assert!(
            mediation_path_ovb_range(one, &[paths], MediationPath::Indirect, 1.0, 0.05, 1.0)
                .is_err()
        );
    }

    #[test]
    fn the_indirect_effect_is_the_product_of_its_two_paths_shares() {
        let paths = MediationPaths {
            treatment_mediator: LinearAdjustmentSummary::new(0.0973, 0.0233, 783.0).unwrap(),
            mediator_outcome: LinearAdjustmentSummary::new(-0.31, 0.06, 500.0).unwrap(),
            treatment_outcome: LinearAdjustmentSummary::new(0.2, 0.05, 500.0).unwrap(),
        };
        let one = target(TargetFamily::Mediation, TargetQuadrant::STATIC_FREQUENTIST, &["y"]);
        // Independent closed form: with R2yz = 1 the bias of path j is
        // se_j sqrt(dof_j) sqrt(R2dz / (1 - R2dz)), a share |bias / estimate| of the path.
        let share = |s: &LinearAdjustmentSummary, r2dz: f64| {
            (1.0 - s.standard_error * s.dof.sqrt() * (r2dz / (1.0 - r2dz)).sqrt()
                / s.estimate.abs())
            .max(0.0)
        };
        let range = mediation_path_ovb_range(
            one.clone(),
            &[paths],
            MediationPath::Indirect,
            1.0,
            0.05,
            1.0,
        )
        .unwrap();
        assert_eq!(range.assumption, SensitivityAssumption::IndirectEffectConfoundingPartialR2);
        for point in range.tested.iter() {
            let want = share(&paths.treatment_mediator, point.parameter)
                * share(&paths.mediator_outcome, point.parameter);
            assert!((point.target - want).abs() < 1e-12, "r2dz {}", point.parameter);
        }
        // At q = 1 the product vanishes when either path does.
        let rv = |s: &LinearAdjustmentSummary| s.robustness_value(1.0, None);
        let expected = rv(&paths.treatment_mediator).min(rv(&paths.mediator_outcome));
        assert!((range.break_even.unwrap() - expected).abs() < 1e-9);
        // For q < 1 the product of shares at the break-even is exactly 1 - q.
        let q = 0.5;
        let half =
            mediation_path_ovb_range(one, &[paths], MediationPath::Indirect, q, 0.05, 1.0).unwrap();
        let r = half.break_even.unwrap();
        let at = |s: &LinearAdjustmentSummary| {
            (1.0 - s.standard_error * s.dof.sqrt() * (r * r / (1.0 - r)).sqrt() / s.estimate.abs())
                .max(0.0)
        };
        assert!(
            (at(&paths.treatment_mediator) * at(&paths.mediator_outcome) - (1.0 - q)).abs() < 1e-9
        );
        assert!(half.break_even_significance.unwrap() <= r + 1e-12);
    }

    #[test]
    fn the_paths_are_fitted_from_the_three_regressions() {
        let (t, m, y1, _, x) = data();
        let paths = mediation_paths_fit(&t, &m, &y1, &[x.clone()]).unwrap();
        let beta = structural(&t, &m, &x, &y1);
        assert!((paths.mediator_outcome.estimate - beta[2]).abs() < 1e-9);
        assert!((paths.treatment_outcome.estimate - beta[1]).abs() < 1e-9);
        let components = mediation_components_fit(&t, &m, &y1, &[x]).unwrap();
        assert!(
            (paths.treatment_mediator.estimate - components.mediation.beta_mediator).abs() < 1e-9
        );
    }

    #[test]
    fn the_msm_fit_matches_replicated_rows_and_the_hand_formula() {
        // Integer weights are replicated rows: the coefficients are identical.
        let x: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let y: Vec<f64> = vec![0.3, 1.1, 1.8, 3.4, 3.9, 5.2, 5.8, 7.5];
        let weights = [1.0, 2.0, 1.0, 3.0, 1.0, 2.0, 1.0, 1.0];
        let weighted = msm_parameter_fit(&y, &[x.clone()], &weights).unwrap();
        let (mut rx, mut ry) = (Vec::new(), Vec::new());
        for ((xi, yi), w) in x.iter().zip(&y).zip(weights) {
            for _ in 0..(w as usize) {
                rx.push(*xi);
                ry.push(*yi);
            }
        }
        let replicated = msm_parameter_fit(&ry, &[rx], &vec![1.0; ry.len()]).unwrap();
        assert!((weighted[0].estimate - replicated[0].estimate).abs() < 1e-10);
        // Unit weights: the HC1 standard error of a simple regression slope.
        let n = x.len() as f64;
        let ones = vec![1.0; x.len()];
        let fit = msm_parameter_fit(&y, &[x.clone()], &ones).unwrap();
        let (mx, my) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
        let sxx: f64 = x.iter().map(|v| (v - mx).powi(2)).sum();
        let slope: f64 = x.iter().zip(&y).map(|(a, b)| (a - mx) * (b - my)).sum::<f64>() / sxx;
        let intercept = my - slope * mx;
        let meat: f64 =
            x.iter().zip(&y).map(|(a, b)| ((a - mx) * (b - intercept - slope * a)).powi(2)).sum();
        let expected = (meat / (sxx * sxx) * n / (n - 2.0)).sqrt();
        assert!((fit[0].estimate - slope).abs() < 1e-10);
        assert!(
            (fit[0].standard_error - expected).abs() < 1e-10,
            "{} vs {expected}",
            fit[0].standard_error
        );
        assert!((fit[0].dof - (n - 2.0)).abs() < 1e-10);
        // Unequal weights shrink the effective sample size and its degrees of freedom.
        assert!(weighted[0].dof < n - 2.0);
        // Misaligned inputs, a negative weight and a constant exposure are refused.
        assert!(msm_parameter_fit(&y, &[x.clone()], &weights[..7]).is_err());
        let mut negative = weights;
        negative[0] = -1.0;
        assert!(msm_parameter_fit(&y, &[x.clone()], &negative).is_err());
        assert!(msm_parameter_fit(&y, &[vec![1.0; 8]], &weights).is_err());
    }

    /// Uniform stream of the weight-estimation simulation.
    fn uniform(state: &mut u64) -> f64 {
        let mut rng = SplitMix(*state);
        let value = (rng.next_u64() >> 11) as f64 / (1_u64 << 53) as f64;
        *state = rng.0;
        value
    }

    /// `L ~ N(0, 1)` confounds a binary treatment `A ~ Bernoulli(expit(L))`; the outcome is
    /// `Y = 1 + 2A + L + e`, so the marginal structural slope of `A` is exactly 2.
    fn confounded(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut state = seed;
        let (mut y, mut a, mut l) = (Vec::new(), Vec::new(), Vec::new());
        let normal = |state: &mut u64| {
            let (u, v) = (uniform(state).max(1e-12), uniform(state));
            (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
        };
        for _ in 0..n {
            let li = normal(&mut state);
            let ai = f64::from(u8::from(uniform(&mut state) < 1.0 / (1.0 + (-li).exp())));
            y.push(1.0 + 2.0 * ai + li + normal(&mut state));
            a.push(ai);
            l.push(li);
        }
        (y, a, l)
    }

    #[test]
    fn refitting_the_weight_models_matches_the_repeated_sampling_spread() {
        let n = 400;
        let period = |a: &[f64], l: &[f64]| MsmWeightPeriod {
            treatment: a.to_vec(),
            numerator: Vec::new(),
            denominator: vec![l.to_vec()],
        };
        // Monte Carlo spread of the slope with the weights estimated in every dataset.
        let mut slopes = Vec::new();
        for seed in 0..300_u64 {
            let (y, a, l) = confounded(n, 1_000 + seed);
            let weights = stabilized_weights(&[period(&a, &l)], &vec![1.0; n]).unwrap();
            slopes.push(msm_parameter_fit(&y, &[a], &weights).unwrap()[0].estimate);
        }
        let mean = slopes.iter().sum::<f64>() / slopes.len() as f64;
        let monte_carlo = (slopes.iter().map(|s| (s - mean).powi(2)).sum::<f64>()
            / (slopes.len() as f64 - 1.0))
            .sqrt();
        assert!((mean - 2.0).abs() < 0.1, "the MSM slope is 2: {mean}");
        // The corrected standard error, averaged over datasets, matches that spread.
        let (mut corrected, mut fixed) = (0.0, 0.0);
        let datasets = 6_u64;
        for seed in 0..datasets {
            let (y, a, l) = confounded(n, 50_000 + seed);
            let fit = msm_parameter_fit_refitting_weights(
                &y,
                &[a.clone()],
                &[period(&a, &l)],
                120,
                seed + 1,
            )
            .unwrap();
            assert_eq!(fit.replicates.0 + fit.replicates.1, 120);
            corrected += fit.refit[0].standard_error / datasets as f64;
            fixed += fit.fixed_weights[0].standard_error / datasets as f64;
            assert!(
                (fit.refit[0].standard_error - fit.fixed_weights[0].standard_error).abs() > 1e-6
            );
        }
        assert!(
            (corrected / monte_carlo - 1.0).abs() < 0.2,
            "refit {corrected} vs Monte Carlo {monte_carlo} (fixed-weight {fixed})"
        );
        // Ignoring the estimation of the denominator model never understates the spread.
        assert!(fixed > 0.95 * monte_carlo, "fixed {fixed} vs Monte Carlo {monte_carlo}");
        // Too few replicates and a non-binary treatment are refused.
        let (y, a, l) = confounded(100, 7);
        assert!(
            msm_parameter_fit_refitting_weights(&y, &[a.clone()], &[period(&a, &l)], 10, 1)
                .is_err()
        );
        let halves = vec![0.5; 100];
        assert!(
            msm_parameter_fit_refitting_weights(&y, &[a], &[period(&halves, &l)], 120, 1).is_err()
        );
    }

    #[test]
    fn an_msm_parameter_vector_has_a_partial_r2_range_at_its_weakest_parameter() {
        // Two exposure terms of one MSM; the second is weak.
        let n = 400;
        let first: Vec<f64> = (0..n).map(|i| f64::from(i % 5)).collect();
        let second: Vec<f64> = (0..n).map(|i| (f64::from(i) * 0.37).sin()).collect();
        let y: Vec<f64> = (0..n)
            .map(|i| {
                let a = f64::from(i);
                1.0 + 0.5 * first[i as usize] + 0.05 * second[i as usize] + 0.8 * (a * 1.3).sin()
            })
            .collect();
        let weights: Vec<f64> = (0..n).map(|i| 0.5 + f64::from(i % 3) * 0.5).collect();
        let summaries = msm_parameter_fit(&y, &[first, second], &weights).unwrap();
        let parameters = target(
            TargetFamily::JointContrast,
            TargetQuadrant::STATIC_FREQUENTIST,
            &["dose", "trend"],
        );
        let range = component_ovb_range(
            parameters,
            &summaries,
            1.0,
            0.05,
            1.0,
            ComponentModel::MarginalStructural,
        )
        .unwrap();
        let expected =
            summaries.iter().map(|s| s.robustness_value(1.0, None)).fold(f64::INFINITY, f64::min);
        assert!((range.break_even.unwrap() - expected).abs() < 1e-12);
        assert!(summaries[0].t_statistic().abs() > summaries[1].t_statistic().abs());
    }
}
