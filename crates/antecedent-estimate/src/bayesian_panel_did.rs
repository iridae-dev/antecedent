//! Bayesian difference-in-differences under a Gaussian outcome model with conjugate priors.
//!
//! The identifying premise is the frequentist one and is untouched by the prior: parallel
//! untreated trends (conditional on the supplied covariates for the augmented form), no
//! anticipation, and a stable population across periods. A posterior puts a distribution on the
//! same estimand under a stated outcome model; it neither tests nor licenses parallel trends, which
//! is untestable after treatment. Every fit is a Normal-Inverse-Gamma linear model
//! `y | beta, sigma2 ~ N(X beta, sigma2 I)`, `beta | sigma2 ~ N(0, sigma2 s^2 I)`,
//! `sigma2 ~ InvGamma(a, b)`, whose marginal posterior of a coefficient is a Student-t.
//!
//! Units are modelled as independent. A cluster that holds more than one unit is refused rather
//! than silently treated as independent.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;

use antecedent_prob::{
    BayesDesignRef, BayesFitOptions, BayesFitResult, GaussianCoefficientPrior, InvGammaPrior,
    LaplaceWorkspace, PriorSet, PriorSpec, fit_conjugate_gaussian,
};

use crate::error::EstimationError;
use crate::util::DrawSpec;

/// Seed offset that separates the independent arm-specific fits of one call.
const ARM_SEED_STRIDE: u64 = 0x9E37_79B9_7F4A_7C15;

/// Normal-Inverse-Gamma prior of a Gaussian regression, shared by every fit in this module.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DidPrior {
    /// Prior scale `s` of each coefficient in units of the residual sd:
    /// `beta | sigma2 ~ N(0, sigma2 s^2)`.
    pub coefficient_scale: f64,
    /// Inverse-Gamma shape of the residual variance.
    pub variance_shape: f64,
    /// Inverse-Gamma scale of the residual variance.
    pub variance_scale: f64,
}

impl DidPrior {
    /// A diffuse default: coefficient scale 100 residual sds and a vague variance prior.
    #[must_use]
    pub const fn diffuse() -> Self {
        Self { coefficient_scale: 100.0, variance_shape: 1e-3, variance_scale: 1e-3 }
    }

    fn validate(self) -> Result<(), EstimationError> {
        let ok = |v: f64| v.is_finite() && v > 0.0;
        if ok(self.coefficient_scale) && ok(self.variance_shape) && ok(self.variance_scale) {
            Ok(())
        } else {
            Err(crate::util::refuse(
                antecedent_core::reason_code!("invalid_argument"),
                "bayesian_did_prior",
                "bayesian DiD prior needs finite positive coefficient scale and variance shape and scale. Next attempt: declare a finite positive coefficient scale, variance shape and variance scale",
            ))
        }
    }
}

/// Posterior of a difference-in-differences effect with its model scope.
#[derive(Clone, Debug, PartialEq)]
pub struct BayesianDidEstimate {
    /// Posterior effect draws in draw order.
    pub effect_draws: Vec<f64>,
    /// Posterior mean.
    pub mean: f64,
    /// Posterior standard deviation.
    pub sd: f64,
    /// Equal-tailed 95% lower posterior quantile.
    pub lower: f64,
    /// Equal-tailed 95% upper posterior quantile.
    pub upper: f64,
    /// Treated rows (cross section) or subjects (panel).
    pub n_treated: usize,
    /// Comparison rows (cross section) or subjects (panel).
    pub n_comparison: usize,
    /// Prior used.
    pub prior: DidPrior,
    /// Explicit model scope.
    pub model: &'static str,
}

impl BayesianDidEstimate {
    fn new(
        effect_draws: Vec<f64>,
        counts: (usize, usize),
        prior: DidPrior,
        model: &'static str,
    ) -> Self {
        let (mean, sd, lower, upper) = summarize(&effect_draws);
        Self {
            effect_draws,
            mean,
            sd,
            lower,
            upper,
            n_treated: counts.0,
            n_comparison: counts.1,
            prior,
            model,
        }
    }
}

/// Mean, sd and equal-tailed 95% interval of posterior draws.
pub(crate) fn summarize(draws: &[f64]) -> (f64, f64, f64, f64) {
    let n = draws.len() as f64;
    let mean = draws.iter().sum::<f64>() / n;
    let sd = (draws.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0)).sqrt();
    let mut sorted = draws.to_vec();
    sorted.sort_by(f64::total_cmp);
    let q = |p: f64| {
        antecedent_stats::quantile_sorted(
            &sorted,
            p,
            antecedent_stats::QuantileRule::ExchangeableRank,
        )
    };
    (mean, sd, q(0.025), q(0.975))
}

/// Refuse a design in which one cluster holds more than one independent unit.
pub(crate) fn require_one_unit_per_cluster(
    clusters: &[impl AsRef<str>],
) -> Result<(), EstimationError> {
    let mut seen = BTreeSet::new();
    for (i, cluster) in clusters.iter().enumerate() {
        if !seen.insert(cluster.as_ref()) {
            return Err(crate::util::refuse(
                antecedent_core::reason_code!("sampling_dependence_unknown"),
                "bayesian_did_dependent_units",
                &format!(
                    "unit {i} shares cluster `{}` with an earlier unit. This posterior models \
                     units as independent. Next attempt: a hierarchical cluster random-effect \
                     model, or the cluster-robust interval of the frequentist estimate",
                    cluster.as_ref()
                ),
            ));
        }
    }
    Ok(())
}

/// Conjugate Normal-Inverse-Gamma fit of `y` on the given columns with seeded joint draws.
///
/// Coefficient `j` is column `j` of the returned draws; the residual variance is column `p`.
pub(crate) fn fit_nig(
    columns: &[Vec<f64>],
    y: &[f64],
    prior: DidPrior,
    sampling: DrawSpec,
) -> Result<BayesFitResult, EstimationError> {
    prior.validate()?;
    let n = y.len();
    let p = columns.len();
    if p == 0 || n <= p + 1 || sampling.draws < 2 {
        return Err(crate::util::refuse(
            antecedent_core::reason_code!("invalid_argument"),
            "bayesian_did_sample_size",
            "bayesian DiD needs more rows than coefficients plus one, and draws >= 2. Next attempt: supply more rows than coefficients plus one and at least two draws",
        ));
    }
    if columns.iter().any(|c| c.len() != n)
        || y.iter().chain(columns.iter().flatten()).any(|v| !v.is_finite())
    {
        return Err(crate::util::refuse(
            antecedent_core::reason_code!("invalid_argument"),
            "bayesian_did_columns",
            "bayesian DiD needs equal-length finite columns. Next attempt: supply equal-length finite columns",
        ));
    }
    let x: Vec<f64> = columns.iter().flatten().copied().collect();
    let prior_set = PriorSet {
        specs: vec![
            PriorSpec::GaussianCoefficients(GaussianCoefficientPrior::isotropic(
                p,
                prior.coefficient_scale,
            )),
            PriorSpec::ResidualInvGamma(InvGammaPrior {
                shape: prior.variance_shape,
                scale: prior.variance_scale,
            }),
        ],
        contrast: None,
        categorical: Vec::new(),
        restrictions: Vec::new(),
    };
    let design =
        BayesDesignRef { x_colmajor: &x, nrows: n, ncols: p, y, weights: None, offsets: None };
    let options = BayesFitOptions {
        n_draws: sampling.draws,
        seed: sampling.seed,
        ..BayesFitOptions::default()
    };
    Ok(fit_conjugate_gaussian(design, &prior_set, &options, &mut LaplaceWorkspace::default())?)
}

fn indicator(flags: &[bool]) -> Vec<f64> {
    flags.iter().map(|&flag| f64::from(flag)).collect()
}

/// Repeated cross-section 2x2 `DiD`: the posterior of `delta` in
/// `y = a + b G + c Post + delta G Post + e`.
///
/// This is the difference of the treated and comparison group-mean changes between the two
/// periods under a common residual variance. Rows must be independent: pass one cluster id per
/// row, distinct across rows.
pub fn fit_bayesian_cross_section_did(
    outcome: &[f64],
    treated_group: &[bool],
    post_period: &[bool],
    clusters: &[impl AsRef<str>],
    prior: DidPrior,
    sampling: DrawSpec,
) -> Result<BayesianDidEstimate, EstimationError> {
    let n = outcome.len();
    if n == 0 || treated_group.len() != n || post_period.len() != n || clusters.len() != n {
        return Err(crate::util::refuse(
            antecedent_core::reason_code!("invalid_argument"),
            "bayesian_did_shape",
            "bayesian DiD needs equal non-empty outcome, group, period and cluster vectors. Next attempt: supply equal-length non-empty outcome, group, period and cluster columns",
        ));
    }
    require_one_unit_per_cluster(clusters)?;
    let mut cells = [[0_usize; 2]; 2];
    for (&group, &post) in treated_group.iter().zip(post_period) {
        cells[usize::from(group)][usize::from(post)] += 1;
    }
    if cells.iter().flatten().any(|&count| count < 2) {
        return Err(crate::util::refuse(
            antecedent_core::reason_code!("arm_not_populated"),
            "bayesian_did_cell_support",
            &format!(
                "each group-by-period cell needs at least two rows, found {cells:?}. Next \
                 attempt: collect more rows in the thin cell, or the frequentist four-cell \
                 contrast, which reports a point estimate without an interval"
            ),
        ));
    }
    let group = indicator(treated_group);
    let post = indicator(post_period);
    let interaction: Vec<f64> = group.iter().zip(&post).map(|(g, p)| g * p).collect();
    let columns = [vec![1.0; n], group, post, interaction];
    let fit = fit_nig(&columns, outcome, prior, sampling)?;
    Ok(BayesianDidEstimate::new(
        fit.draws.column(3)?.to_vec(),
        (cells[1][0] + cells[1][1], cells[0][0] + cells[0][1]),
        prior,
        "repeated cross-section 2x2 DiD; Gaussian common-variance cell model; independent rows; parallel trends declared",
    ))
}

/// Two-period panel `DiD`: the posterior of `delta` in `post_i - pre_i = a + delta G_i + e_i`.
///
/// Differencing within subject removes the subject effect and the within-subject serial
/// dependence of the two periods; subjects are independent and share one change-score variance.
/// Pass one cluster id per subject, distinct across subjects.
pub fn fit_bayesian_panel_did(
    pre: &[f64],
    post: &[f64],
    treated: &[bool],
    clusters: &[impl AsRef<str>],
    prior: DidPrior,
    sampling: DrawSpec,
) -> Result<BayesianDidEstimate, EstimationError> {
    let n = pre.len();
    if n == 0 || post.len() != n || treated.len() != n || clusters.len() != n {
        return Err(crate::util::refuse(
            antecedent_core::reason_code!("invalid_argument"),
            "bayesian_panel_did_shape",
            "bayesian panel DiD needs equal non-empty pre, post, treatment and cluster vectors. Next attempt: supply equal-length non-empty pre, post, treatment and cluster columns",
        ));
    }
    require_one_unit_per_cluster(clusters)?;
    let n_treated = treated.iter().filter(|&&flag| flag).count();
    if n_treated < 2 || n - n_treated < 2 {
        return Err(crate::util::refuse(
            antecedent_core::reason_code!("arm_not_populated"),
            "bayesian_did_arm_support",
            "each arm needs at least two subjects. Next attempt: more subjects in the thin arm, \
             or the frequentist change-score contrast, which reports a point estimate without an \
             interval",
        ));
    }
    let change: Vec<f64> = pre.iter().zip(post).map(|(a, b)| b - a).collect();
    let columns = [vec![1.0; n], indicator(treated)];
    let fit = fit_nig(&columns, &change, prior, sampling)?;
    Ok(BayesianDidEstimate::new(
        fit.draws.column(1)?.to_vec(),
        (n_treated, n - n_treated),
        prior,
        "two-period panel DiD; Gaussian change-score model; independent subjects; parallel trends declared",
    ))
}

/// Covariate-adjusted two-period panel ATT on the treated.
///
/// Each arm's change score follows its own Normal-Inverse-Gamma linear model in the covariates.
/// The effect is the treated-arm mean change minus the comparison-arm regression evaluated at the
/// treated covariate mean, that is the ATT conditional on the observed treated covariates under
/// parallel untreated trends given those covariates. The posterior uses the outcome model alone:
/// it does not use a propensity score, so unlike the doubly robust point estimate it is only as
/// good as the linear-Gaussian change-score model in each arm.
pub fn fit_bayesian_augmented_panel_did(
    pre: &[f64],
    post: &[f64],
    treated: &[bool],
    covariates: &[Vec<f64>],
    clusters: &[impl AsRef<str>],
    prior: DidPrior,
    sampling: DrawSpec,
) -> Result<BayesianDidEstimate, EstimationError> {
    let n = pre.len();
    if n == 0
        || post.len() != n
        || treated.len() != n
        || clusters.len() != n
        || covariates.iter().any(|c| c.len() != n || c.iter().any(|v| !v.is_finite()))
    {
        return Err(crate::util::refuse(
            antecedent_core::reason_code!("invalid_argument"),
            "bayesian_augmented_did_shape",
            "bayesian augmented DiD needs equal non-empty pre, post, treatment, cluster and covariate columns. Next attempt: supply equal-length non-empty pre, post, treatment, cluster and finite covariate columns",
        ));
    }
    require_one_unit_per_cluster(clusters)?;
    let p = covariates.len() + 1;
    let mut arm_draws: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
    let mut counts = [0_usize; 2];
    let mut treated_mean = vec![0.0; covariates.len()];
    for arm in [false, true] {
        let rows: Vec<usize> = (0..n).filter(|&i| treated[i] == arm).collect();
        if rows.len() < p + 2 {
            return Err(crate::util::refuse(
                antecedent_core::reason_code!("arm_not_populated"),
                "bayesian_did_arm_support",
                &format!(
                    "the {} arm has {} subjects but needs at least {} for {} coefficients. Next \
                     attempt: fewer covariates, or more subjects in the thin arm",
                    if arm { "treated" } else { "comparison" },
                    rows.len(),
                    p + 2,
                    p
                ),
            ));
        }
        if arm {
            for (mean, column) in treated_mean.iter_mut().zip(covariates) {
                *mean = rows.iter().map(|&i| column[i]).sum::<f64>() / rows.len() as f64;
            }
        }
        counts[usize::from(arm)] = rows.len();
    }
    for arm in [false, true] {
        let rows: Vec<usize> = (0..n).filter(|&i| treated[i] == arm).collect();
        let change: Vec<f64> = rows.iter().map(|&i| post[i] - pre[i]).collect();
        let mut columns = vec![vec![1.0; rows.len()]];
        for column in covariates {
            columns.push(rows.iter().map(|&i| column[i]).collect());
        }
        let seed = sampling.seed.wrapping_add(ARM_SEED_STRIDE.wrapping_mul(u64::from(arm)));
        let fit = fit_nig(&columns, &change, prior, DrawSpec::new(sampling.draws, seed))?;
        let mut prediction = fit.draws.column(0)?.to_vec();
        for (j, mean) in treated_mean.iter().enumerate() {
            for (value, coefficient) in prediction.iter_mut().zip(fit.draws.column(j + 1)?) {
                *value += mean * coefficient;
            }
        }
        arm_draws[usize::from(arm)] = prediction;
    }
    let effect: Vec<f64> = arm_draws[1]
        .iter()
        .zip(&arm_draws[0])
        .map(|(treated, control)| treated - control)
        .collect();
    Ok(BayesianDidEstimate::new(
        effect,
        (counts[1], counts[0]),
        prior,
        "two-period panel ATT on the treated; arm-specific Gaussian change-score regressions; outcome model only; conditional parallel trends declared",
    ))
}

#[cfg(test)]
mod tests;
