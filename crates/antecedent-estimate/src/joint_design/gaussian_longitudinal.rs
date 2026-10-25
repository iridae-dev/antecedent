//! Full-history continuous longitudinal g-computation with exact conditional-mean integration.
// SPDX-License-Identifier: MIT OR Apache-2.0
use super::{JointDesignEstimate, JointDesignPosterior, JointUncertainty};
use antecedent_core::{CausalRng, ExecutionContext};
use antecedent_stats::{DenseLinearAlgebra, FaerBackend, LeastSquaresWorkspace};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// One original independent continuous-valued longitudinal episode.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GaussianLongitudinalEpisode {
    /// Original independent-unit identity.
    pub id: String,
    /// Pretreatment covariate vector at every time, retaining treatment-affected states.
    pub covariates: Vec<Vec<f64>>,
    /// Actual simultaneous treatment vector at each time.
    pub treatments: Vec<Vec<f64>>,
    /// Terminal observed outcome.
    pub outcome: f64,
}
/// Exact continuous joint intervention regime.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GaussianLongitudinalWorld {
    /// Stable complete-regime key.
    pub key: String,
    /// Actual hard-set treatment vectors in time order.
    pub treatments: Vec<Vec<f64>>,
}
/// Conditional assignment density's admitted domain and fitted transformation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "support", rename_all = "snake_case", deny_unknown_fields)]
pub enum GaussianAssignmentSupport {
    /// A jointly Gaussian treatment vector on the real coordinate.
    Real,
    /// Gaussian log-treatment conditional model, supporting strictly positive doses.
    Positive,
    /// Gaussian logit-treatment conditional model on an explicit open interval.
    Interval {
        /// Lower domain endpoint.
        lower: f64,
        /// Upper domain endpoint.
        upper: f64,
    },
}
impl GaussianAssignmentSupport {
    fn transform(&self, value: f64) -> Result<f64, String> {
        if !value.is_finite() {
            return Err("assignment values must be finite".into());
        }
        match self {Self::Real=>Ok(value),Self::Positive if value>0.0=>Ok(value.ln()),Self::Interval{lower,upper}if lower.is_finite()&&upper.is_finite()&&lower<upper&&value>*lower&&value<*upper=>Ok(((value-lower)/(upper-value)).ln()),_=>Err("joint intervention lies outside its admitted continuous assignment-density support".into())}
    }
}
/// A conditional mean feature linear in covariate states under a fixed regime.
/// Treatment polynomials and covariate-by-treatment terms retain exact Gaussian
/// state-mean integration; nonlinear functions of unknown covariates are not hidden.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "feature", rename_all = "snake_case", deny_unknown_fields)]
pub enum GaussianHistoryFeature {
    /// Constant regression feature.
    Intercept,
    /// A measured covariate at its actual time.
    Covariate {
        /// Time measured before that time's joint treatment.
        time: usize,
        /// Coordinate in the original covariate axis.
        coordinate: usize,
    },
    /// One actual joint-treatment coordinate.
    Treatment {
        /// Treatment time.
        time: usize,
        /// Physical channel coordinate.
        channel: usize,
    },
    /// Prespecified interaction or square of actual treatment coordinates.
    TreatmentProduct {
        /// First treatment time.
        first_time: usize,
        /// First channel.
        first_channel: usize,
        /// Second treatment time.
        second_time: usize,
        /// Second channel.
        second_channel: usize,
    },
    /// Treatment-dependent effect of a covariate, preserving linear state recursion.
    CovariateTreatment {
        /// Covariate time.
        covariate_time: usize,
        /// Covariate coordinate.
        coordinate: usize,
        /// Treatment time.
        treatment_time: usize,
        /// Treatment channel.
        channel: usize,
    },
}
impl GaussianHistoryFeature {
    fn value(&self, l: &[Vec<f64>], a: &[Vec<f64>]) -> f64 {
        match *self {
            Self::Intercept => 1.0,
            Self::Covariate { time, coordinate } => l[time][coordinate],
            Self::Treatment { time, channel } => a[time][channel],
            Self::TreatmentProduct { first_time, first_channel, second_time, second_channel } => {
                a[first_time][first_channel] * a[second_time][second_channel]
            }
            Self::CovariateTreatment { covariate_time, coordinate, treatment_time, channel } => {
                l[covariate_time][coordinate] * a[treatment_time][channel]
            }
        }
    }
    fn valid(
        &self,
        horizon: usize,
        dimensions: usize,
        channels: usize,
        cov_before: (usize, usize),
        treatment_before: usize,
    ) -> bool {
        let cov = |t: usize, d: usize| {
            t < horizon
                && d < dimensions
                && (t < cov_before.0 || (t == cov_before.0 && d < cov_before.1))
        };
        let treatment = |t: usize, c: usize| t < treatment_before && t < horizon && c < channels;
        match *self {
            Self::Intercept => true,
            Self::Covariate { time, coordinate } => cov(time, coordinate),
            Self::Treatment { time, channel } => treatment(time, channel),
            Self::TreatmentProduct { first_time, first_channel, second_time, second_channel } => {
                treatment(first_time, first_channel) && treatment(second_time, second_channel)
            }
            Self::CovariateTreatment { covariate_time, coordinate, treatment_time, channel } => {
                cov(covariate_time, coordinate) && treatment(treatment_time, channel)
            }
        }
    }
}
/// Actual conditional Gaussian mechanism equations, retaining full measured histories.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GaussianLongitudinalModel {
    /// Features for each covariate mechanism [time][coordinate][feature]. Same-time
    /// earlier covariates give a triangular factorization of their joint distribution.
    pub covariates: Vec<Vec<Vec<GaussianHistoryFeature>>>,
    /// Terminal outcome mechanism features over complete histories.
    pub outcome: Vec<GaussianHistoryFeature>,
    /// Conditional transformed joint-assignment mean features at each time.
    pub assignment: Vec<Vec<GaussianHistoryFeature>>,
}
impl GaussianLongitudinalModel {
    /// Dense full-history main effects and within-time joint-treatment interactions.
    /// Every treatment-affected covariate remains in downstream equations. Callers
    /// may explicitly add supported treatment/state interactions before compilation.
    #[must_use]
    pub fn full_history(horizon: usize, dimensions: usize, channels: usize) -> Self {
        let features = |cov_before: (usize, usize), treatment_before: usize| {
            let mut f = vec![GaussianHistoryFeature::Intercept];
            for t in 0..horizon {
                for d in 0..dimensions {
                    if t < cov_before.0 || (t == cov_before.0 && d < cov_before.1) {
                        f.push(GaussianHistoryFeature::Covariate { time: t, coordinate: d });
                    }
                }
            }
            for t in 0..treatment_before {
                for c in 0..channels {
                    f.push(GaussianHistoryFeature::Treatment { time: t, channel: c });
                    for b in 0..c {
                        f.push(GaussianHistoryFeature::TreatmentProduct {
                            first_time: t,
                            first_channel: b,
                            second_time: t,
                            second_channel: c,
                        });
                    }
                }
            }
            f
        };
        Self {
            covariates: (0..horizon)
                .map(|t| (0..dimensions).map(|d| features((t, d), t)).collect())
                .collect(),
            outcome: features((horizon, 0), horizon),
            assignment: (0..horizon).map(|t| features((t + 1, 0), t)).collect(),
        }
    }
}
/// Scientific longitudinal premises plus the exact conditional model being fitted.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GaussianLongitudinalDesign {
    /// Explicit original physical intervention variable IDs in treatment-column order.
    pub channel_variables: Vec<u32>,
    /// Original measured covariate IDs in covariate-column order.
    pub covariate_variables: Vec<u32>,
    /// Every requested complete continuous regime.
    pub worlds: Vec<GaussianLongitudinalWorld>,
    /// One supported transformed assignment-density family per physical channel.
    pub assignment_support: Vec<GaussianAssignmentSupport>,
    /// Explicit sequential exchangeability conditional on the full recorded history.
    pub sequential_exchangeability: String,
    /// Consistency and fixed treatment versions.
    pub consistency: String,
    /// Original independent-episode population and sampling frame.
    pub population: String,
    /// Declared joint Gaussian conditional mechanism and transformed assignment restrictions.
    pub conditional_model: String,
    /// Actual equation features, checked for temporal order and verified rank.
    pub model: GaussianLongitudinalModel,
}
/// Proper conjugate prior shared by the declared Gaussian conditional mechanisms.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GaussianLongitudinalPrior {
    /// Coefficients have independent N(0, `coefficient_scale` * sigma²) priors.
    pub coefficient_scale: f64,
    /// Proper inverse-gamma variance shape.
    pub variance_shape: f64,
    /// Proper inverse-gamma variance scale.
    pub variance_scale: f64,
    /// Aligned full-mechanism posterior draws.
    pub draws: usize,
    /// Deterministic stream.
    pub seed: u64,
}
/// Retained complete independent-episode bootstrap, including every failed slot.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LongitudinalGaussianBootstrap {
    /// Original point functional and joint covariance only when all realizations are valid.
    /// An empty covariance explicitly withholds uncertainty after any failed realization.
    pub estimate: JointDesignEstimate,
    /// Complete joint regime means or failed slots in original replicate order.
    pub realizations: Vec<Option<Vec<f64>>>,
    /// Failure reason for each failed slot; no failed fit is dropped or replaced.
    pub failures: Vec<Option<String>>,
}
#[derive(Clone, Debug)]
struct Regression {
    features: Vec<GaussianHistoryFeature>,
    coefficients: Vec<f64>,
    x: Vec<f64>,
    y: Vec<f64>,
    n: usize,
}
#[derive(Clone, Debug)]
struct Fit {
    covariates: Vec<Vec<Regression>>,
    outcome: Regression,
}
fn cancelled(ctx: &ExecutionContext) -> Result<(), String> {
    if ctx.cancellation.is_cancelled() {
        Err("continuous longitudinal execution cancelled".into())
    } else {
        Ok(())
    }
}
fn validate(
    design: &GaussianLongitudinalDesign,
    episodes: &[GaussianLongitudinalEpisode],
) -> Result<(usize, usize, usize), String> {
    let h = design.worlds.first().map_or(0, |w| w.treatments.len());
    let d = design.covariate_variables.len();
    let m = design.channel_variables.len();
    if h == 0
        || h > 16
        || d > 8
        || m == 0
        || m > 8
        || episodes.len() < 4
        || episodes.len() > 8192
        || design.worlds.len() > 256
        || design.assignment_support.len() != m
        || design
            .channel_variables
            .iter()
            .chain(&design.covariate_variables)
            .collect::<BTreeSet<_>>()
            .len()
            != m + d
        || [
            &design.sequential_exchangeability,
            &design.consistency,
            &design.population,
            &design.conditional_model,
        ]
        .iter()
        .any(|s| s.trim().is_empty())
        || design.worlds.iter().map(|w| &w.key).collect::<BTreeSet<_>>().len()
            != design.worlds.len()
    {
        return Err("continuous longitudinal design requires explicit source axes, full sequential premises and bounded distinct regimes".into());
    }
    for world in &design.worlds {
        if world.key.is_empty()
            || world.treatments.len() != h
            || world.treatments.iter().any(|a| a.len() != m)
        {
            return Err("continuous longitudinal regime axes differ from source".into());
        }
        for a in &world.treatments {
            for (value, support) in a.iter().zip(&design.assignment_support) {
                support.transform(*value)?;
            }
        }
    }
    if design.model.covariates.len() != h
        || design.model.covariates.iter().any(|row| row.len() != d)
        || design.model.assignment.len() != h
    {
        return Err("conditional model omits initial/state/assignment mechanisms".into());
    }
    let feature_count = design
        .model
        .covariates
        .iter()
        .flatten()
        .map(Vec::len)
        .chain(design.model.assignment.iter().map(Vec::len))
        .chain(std::iter::once(design.model.outcome.len()))
        .sum::<usize>();
    if feature_count > 16384 {
        return Err("full longitudinal mechanism feature workspace exceeds bounded implementation resources".into());
    }
    let check = |features: &[GaussianHistoryFeature], cov_before, treatment_before| {
        features.is_empty()
            || features.len() > 512
            || features.iter().any(|f| !f.valid(h, d, m, cov_before, treatment_before))
            || features.iter().enumerate().any(|(i, f)| features[..i].contains(f))
            || !features.contains(&GaussianHistoryFeature::Intercept)
    };
    for t in 0..h {
        for j in 0..d {
            if check(&design.model.covariates[t][j], (t, j), t) {
                return Err(
                    "covariate mechanism has missing/duplicate or future-history features".into()
                );
            }
        }
        if check(&design.model.assignment[t], (t + 1, 0), t) {
            return Err("assignment model may use only measured current/past covariates and past treatments".into());
        }
    }
    if check(&design.model.outcome, (h, 0), h) {
        return Err("terminal outcome mechanism has invalid history features".into());
    }
    validate_episodes(design, episodes, h, d, m)?;
    Ok((h, d, m))
}
/// Validate independent source episodes against the already checked history axes
/// and declared treatment support, before any regression or integration.
fn validate_episodes(
    design: &GaussianLongitudinalDesign,
    episodes: &[GaussianLongitudinalEpisode],
    h: usize,
    d: usize,
    m: usize,
) -> Result<(), String> {
    let mut ids = BTreeSet::new();
    for e in episodes {
        if e.id.is_empty()
            || !ids.insert(&e.id)
            || e.covariates.len() != h
            || e.treatments.len() != h
            || e.covariates.iter().any(|l| l.len() != d)
            || e.treatments.iter().any(|a| a.len() != m)
            || !e.outcome.is_finite()
            || e.covariates
                .iter()
                .flatten()
                .chain(e.treatments.iter().flatten())
                .any(|v| !v.is_finite())
        {
            return Err("invalid original continuous longitudinal episode".into());
        }
        for a in &e.treatments {
            for (value, support) in a.iter().zip(&design.assignment_support) {
                support.transform(*value)?;
            }
        }
    }
    Ok(())
}
fn regression(
    features: &[GaussianHistoryFeature],
    episodes: &[&GaussianLongitudinalEpisode],
    response: impl Fn(&GaussianLongitudinalEpisode) -> Result<f64, String>,
) -> Result<(Regression, Vec<f64>), String> {
    let n = episodes.len();
    let p = features.len();
    if n <= p {
        return Err("conditional Gaussian mechanism has insufficient independent episodes".into());
    }
    let mut x = Vec::with_capacity(n * p);
    for feature in features {
        for e in episodes {
            x.push(feature.value(&e.covariates, &e.treatments));
        }
    }
    let y = episodes.iter().map(|e| response(e)).collect::<Result<Vec<_>, _>>()?;
    if x.iter().chain(&y).any(|v| !v.is_finite()) {
        return Err("conditional mechanism feature overflow".into());
    }
    let fit = FaerBackend
        .least_squares(&x, n, p, &y, &mut LeastSquaresWorkspace::default())
        .map_err(|e| e.to_string())?;
    if fit.rank != p {
        return Err("conditional mechanism design loses full history rank".into());
    }
    Ok((
        Regression { features: features.to_vec(), coefficients: fit.coefficients, x, y, n },
        fit.residuals,
    ))
}
fn fit(
    design: &GaussianLongitudinalDesign,
    episodes: &[&GaussianLongitudinalEpisode],
    ctx: &ExecutionContext,
) -> Result<Fit, String> {
    let h = design.worlds[0].treatments.len();
    let d = design.covariate_variables.len();
    let m = design.channel_variables.len();
    let mut covariates = Vec::with_capacity(h);
    for t in 0..h {
        cancelled(ctx)?;
        let mut row = Vec::with_capacity(d);
        for j in 0..d {
            row.push(
                regression(&design.model.covariates[t][j], episodes, |e| Ok(e.covariates[t][j]))?.0,
            );
        }
        covariates.push(row);
        let mut residuals = Vec::with_capacity(m);
        for c in 0..m {
            residuals.push(
                regression(&design.model.assignment[t], episodes, |e| {
                    design.assignment_support[c].transform(e.treatments[t][c])
                })?
                .1,
            );
        }
        let mut covariance = vec![0.0; m * m];
        for j in 0..m {
            for l in 0..m {
                covariance[j * m + l] =
                    residuals[j].iter().zip(&residuals[l]).map(|(a, b)| a * b).sum::<f64>()
                        / episodes.len() as f64;
            }
        }
        if antecedent_stats::cholesky_spd(&covariance, m).is_none() {
            return Err("conditional joint assignment has singular residual covariance: joint positivity was not earned by marginal support".into());
        }
    }
    Ok(Fit {
        covariates,
        outcome: regression(&design.model.outcome, episodes, |e| Ok(e.outcome))?.0,
    })
}
fn evaluate(design: &GaussianLongitudinalDesign, fit: &Fit) -> Result<Vec<f64>, String> {
    let h = design.worlds[0].treatments.len();
    let d = design.covariate_variables.len();
    design
        .worlds
        .iter()
        .map(|w| {
            let mut means = vec![vec![0.0; d]; h];
            for t in 0..h {
                for j in 0..d {
                    let model = &fit.covariates[t][j];
                    means[t][j] = model
                        .features
                        .iter()
                        .zip(&model.coefficients)
                        .map(|(f, b)| f.value(&means, &w.treatments) * b)
                        .sum::<f64>();
                }
            }
            let value = fit
                .outcome
                .features
                .iter()
                .zip(&fit.outcome.coefficients)
                .map(|(f, b)| f.value(&means, &w.treatments) * b)
                .sum::<f64>();
            if value.is_finite() {
                Ok(value)
            } else {
                Err("continuous g-formula mean overflow".into())
            }
        })
        .collect()
}
/// Fit and integrate the complete Gaussian history law. This analytic reduction
/// integrates treatment-affected covariates instead of freezing observed future paths.
/// # Errors
/// Invalid model/premises, history rank, joint positivity or resource/cancellation failure.
pub fn gaussian_longitudinal_point(
    design: &GaussianLongitudinalDesign,
    episodes: &[GaussianLongitudinalEpisode],
    ctx: &ExecutionContext,
) -> Result<Vec<f64>, String> {
    validate(design, episodes)?;
    let fit = fit(design, &episodes.iter().collect::<Vec<_>>(), ctx)?;
    evaluate(design, &fit)
}
/// Entire independent episodes are redrawn; initial population mechanisms, every
/// treatment-affected state equation, policy-support fit and terminal model are refit.
/// All regimes consume each one complete refit, retaining failures in their original slots.
/// # Errors
/// Invalid point procedure, draw count, source/policy support or cancellation.
pub fn gaussian_longitudinal_bootstrap(
    design: &GaussianLongitudinalDesign,
    episodes: &[GaussianLongitudinalEpisode],
    replicates: usize,
    seed: u64,
    ctx: &ExecutionContext,
) -> Result<LongitudinalGaussianBootstrap, String> {
    let point = gaussian_longitudinal_point(design, episodes, ctx)?;
    if !(2..=100_000).contains(&replicates) {
        return Err("full longitudinal bootstrap needs bounded multiple realizations".into());
    }
    let mut rng = CausalRng::from_seed(seed);
    let mut realizations = Vec::with_capacity(replicates);
    let mut failures = Vec::with_capacity(replicates);
    for _ in 0..replicates {
        cancelled(ctx)?;
        let sample = (0..episodes.len())
            .map(|_| &episodes[crate::rollout_resampling::uniform_index(&mut rng, episodes.len())])
            .collect::<Vec<_>>();
        match fit(design, &sample, ctx).and_then(|f| evaluate(design, &f)) {
            Ok(values) => {
                realizations.push(Some(values));
                failures.push(None);
            }
            Err(reason) => {
                if ctx.cancellation.is_cancelled() {
                    return Err(reason);
                }
                realizations.push(None);
                failures.push(Some(reason));
            }
        }
    }
    let k = point.len();
    let mut covariance = Vec::new();
    if realizations.iter().all(Option::is_some) {
        let center = (0..k)
            .map(|j| realizations.iter().flatten().map(|v| v[j]).sum::<f64>() / replicates as f64)
            .collect::<Vec<_>>();
        covariance = vec![0.0; k * k];
        for v in realizations.iter().flatten() {
            for j in 0..k {
                for l in 0..k {
                    covariance[j * k + l] +=
                        (v[j] - center[j]) * (v[l] - center[l]) / (replicates - 1) as f64;
                }
            }
        }
    }
    Ok(LongitudinalGaussianBootstrap {
        estimate: JointDesignEstimate {
            worlds: design.worlds.iter().map(|w| w.key.clone()).collect(),
            means: point,
            covariance,
            sampling_unit: "independent complete continuous longitudinal episode".into(),
            uncertainty: JointUncertainty::FullProcedureBootstrap,
        },
        realizations,
        failures,
    })
}
struct PosteriorRegression {
    mean: Vec<f64>,
    chol: Vec<f64>,
    shape: f64,
    scale: f64,
}
fn posterior_regression(
    model: &Regression,
    prior: &GaussianLongitudinalPrior,
) -> Result<PosteriorRegression, String> {
    let p = model.coefficients.len();
    let n = model.n;
    let mut precision = vec![0.0; p * p];
    let mut rhs = vec![0.0; p];
    for j in 0..p {
        precision[j * p + j] = 1.0 / prior.coefficient_scale;
        for i in 0..n {
            rhs[j] += model.x[j * n + i] * model.y[i];
            for l in 0..p {
                precision[j * p + l] += model.x[j * n + i] * model.x[l * n + i];
            }
        }
    }
    let chol = antecedent_stats::cholesky_spd(&precision, p)
        .ok_or("proper Gaussian history posterior factorization failed")?;
    let mean = antecedent_stats::chol_solve(&chol, p, &rhs)
        .ok_or("Gaussian history posterior mean solve failed")?;
    let scale = prior.variance_scale
        + 0.5
            * (model.y.iter().map(|y| y * y).sum::<f64>()
                - mean.iter().zip(&rhs).map(|(a, b)| a * b).sum::<f64>());
    if !scale.is_finite() || scale <= 0.0 {
        return Err("conjugate history posterior variance scale is nonpositive".into());
    }
    Ok(PosteriorRegression { mean, chol, shape: prior.variance_shape + 0.5 * n as f64, scale })
}
impl PosteriorRegression {
    fn draw(&self, rng: &mut CausalRng) -> Result<Vec<f64>, String> {
        let p = self.mean.len();
        let variance = antecedent_kernels::sample_inv_gamma(self.shape, self.scale, rng);
        if !variance.is_finite() || variance <= 0.0 {
            return Err("history variance posterior draw overflow".into());
        }
        let mut delta = (0..p)
            .map(|_| antecedent_kernels::standard_normal(rng) * variance.sqrt())
            .collect::<Vec<_>>();
        for j in (0..p).rev() {
            let sum = (j + 1..p).map(|l| self.chol[l * p + j] * delta[l]).sum::<f64>();
            delta[j] = (delta[j] - sum) / self.chol[j * p + j];
        }
        Ok(self.mean.iter().zip(delta).map(|(m, d)| m + d).collect())
    }
}
/// Proper full-mechanism Bayesian g-computation: independent conditional NIG
/// priors update from each actual full-history regression likelihood, then every
/// intervention world shares each sampled initial/state/outcome mechanism.
/// # Errors
/// Invalid priors, original joint assignment support, mechanism rank or posterior draw failure.
pub fn gaussian_longitudinal_posterior(
    design: &GaussianLongitudinalDesign,
    episodes: &[GaussianLongitudinalEpisode],
    prior: &GaussianLongitudinalPrior,
    ctx: &ExecutionContext,
) -> Result<JointDesignPosterior, String> {
    validate(design, episodes)?;
    if [prior.coefficient_scale, prior.variance_shape, prior.variance_scale]
        .iter()
        .any(|v| !v.is_finite() || *v <= 0.0)
        || prior.draws < 2
        || prior.draws > 100_000
    {
        return Err("continuous Gaussian g-computation requires proper NIG priors and bounded multiple draws".into());
    }
    let mut mechanism = fit(design, &episodes.iter().collect::<Vec<_>>(), ctx)?;
    let covariate_posteriors = mechanism
        .covariates
        .iter()
        .map(|row| {
            row.iter()
                .map(|model| posterior_regression(model, prior))
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    let outcome_posterior = posterior_regression(&mechanism.outcome, prior)?;
    let mut rng = CausalRng::from_seed(prior.seed);
    let mut draws = Vec::with_capacity(prior.draws);
    for _ in 0..prior.draws {
        cancelled(ctx)?;
        for (row, posteriors) in mechanism.covariates.iter_mut().zip(&covariate_posteriors) {
            for (model, posterior) in row.iter_mut().zip(posteriors) {
                model.coefficients = posterior.draw(&mut rng)?;
            }
        }
        mechanism.outcome.coefficients = outcome_posterior.draw(&mut rng)?;
        draws.push(evaluate(design, &mechanism)?);
    }
    JointDesignPosterior::from_draws(design.worlds.iter().map(|w|w.key.clone()).collect(),draws,"independent complete continuous longitudinal episode".into(),"full-history Gaussian initial/state/outcome conditional likelihood with proper Gaussian coefficients scaled by inverse-gamma mechanism variances; triangular same-time covariate factorization, explicit treatment polynomials and covariate-treatment interactions, exact regime mean integration; original transformed joint Gaussian assignment support fitted and retained".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> (GaussianLongitudinalDesign, Vec<GaussianLongitudinalEpisode>) {
        let mut episodes = Vec::new();
        for repeat in 0..3 {
            for l in [-1.0, 1.0] {
                for v0 in [-1.0, 1.0] {
                    for u in [-1.0, 1.0] {
                        for v1 in [-1.0, 1.0] {
                            for e in [-1.0, 1.0] {
                                let a0 = 0.6 * l + v0;
                                let l1 = 1.0 + 2.0 * a0 + 0.5 * l + u;
                                let a1 = 0.4 * l1 + 0.2 * a0 + v1;
                                episodes.push(GaussianLongitudinalEpisode {
                                    id: format!("{repeat}/{l}/{v0}/{u}/{v1}/{e}"),
                                    covariates: vec![vec![l], vec![l1]],
                                    treatments: vec![vec![a0], vec![a1]],
                                    outcome: 3.0
                                        + a0
                                        + 2.0 * a1
                                        + 4.0 * l1
                                        + 0.3 * l
                                        + 5.0 * a0 * a1
                                        + 0.2 * l1 * a1
                                        + e,
                                });
                            }
                        }
                    }
                }
            }
        }
        let mut model = GaussianLongitudinalModel::full_history(2, 1, 1);
        model.outcome.push(GaussianHistoryFeature::TreatmentProduct {
            first_time: 0,
            first_channel: 0,
            second_time: 1,
            second_channel: 0,
        });
        model.outcome.push(GaussianHistoryFeature::CovariateTreatment {
            covariate_time: 1,
            coordinate: 0,
            treatment_time: 1,
            channel: 0,
        });
        let design=GaussianLongitudinalDesign{channel_variables:vec![1],covariate_variables:vec![2],worlds:[(0.0,0.0),(1.0,0.0),(0.0,1.0),(1.0,1.0)].into_iter().enumerate().map(|(i,(a,b))|GaussianLongitudinalWorld{key:i.to_string(),treatments:vec![vec![a],vec![b]]}).collect(),assignment_support:vec![GaussianAssignmentSupport::Real],sequential_exchangeability:"measured fullhistory suffices before each joint assignment".into(),consistency:"stable continuous treatment versions".into(),population:"original independent fullhistory source episodes".into(),conditional_model:"Gaussian full-history state/outcome mechanisms and Gaussian conditional treatment density".into(),model};
        (design, episodes)
    }
    #[test]
    fn continuous_treatment_affected_histories_have_correct_analytic_joint_regime_truth() {
        let (design, episodes) = fixture();
        let ctx = ExecutionContext::for_tests(552);
        let means = gaussian_longitudinal_point(&design, &episodes, &ctx).unwrap();
        // Independently integrate E[L0]=0 and E[L1(a0)]=1+2a0. The A1*L1
        // term changes both main and interaction effects; fixing L1 loses these terms.
        for (mean, truth) in means.iter().zip([7.0, 16.0, 9.2, 23.6]) {
            assert!((mean - truth).abs() < 1e-9);
        }
        let bootstrap = gaussian_longitudinal_bootstrap(&design, &episodes, 64, 553, &ctx).unwrap();
        assert_eq!(bootstrap.realizations.len(), 64);
        assert!(bootstrap.failures.iter().all(Option::is_none));
        assert_eq!(bootstrap.estimate.covariance.len(), 16);
        for values in bootstrap.realizations.iter().flatten() {
            assert!(values.iter().all(|v| v.is_finite()));
        }
        let posterior = gaussian_longitudinal_posterior(
            &design,
            &episodes,
            &GaussianLongitudinalPrior {
                coefficient_scale: 1e6,
                variance_shape: 2.0,
                variance_scale: 1.0,
                draws: 10000,
                seed: 554,
            },
            &ctx,
        )
        .unwrap();
        for (j, truth) in [7.0, 16.0, 9.2, 23.6].iter().enumerate() {
            let mcse =
                (posterior.estimate.covariance[j * 4 + j] / posterior.draws.len() as f64).sqrt();
            assert!((posterior.estimate.means[j] - truth).abs() < 6.0 * mcse + 1e-4);
        }
    }
    #[test]
    fn continuous_joint_positivity_and_temporal_premises_cannot_be_invented() {
        let (mut design, mut episodes) = fixture();
        let ctx = ExecutionContext::for_tests(555);
        design.sequential_exchangeability.clear();
        assert!(gaussian_longitudinal_point(&design, &episodes, &ctx).is_err());
        design.sequential_exchangeability = "full-history exchangeability".into();
        design.model.covariates[0][0]
            .push(GaussianHistoryFeature::Treatment { time: 0, channel: 0 });
        assert!(gaussian_longitudinal_point(&design, &episodes, &ctx).is_err());
        design.model.covariates[0][0].pop();
        design.assignment_support[0] = GaussianAssignmentSupport::Positive;
        assert!(gaussian_longitudinal_point(&design, &episodes, &ctx).is_err());
        design.assignment_support[0] = GaussianAssignmentSupport::Real;
        // Both marginal channels vary, but their joint assignment density is singular.
        design.channel_variables.push(3);
        design.assignment_support.push(GaussianAssignmentSupport::Real);
        for world in &mut design.worlds {
            for a in &mut world.treatments {
                a.push(a[0]);
            }
        }
        for episode in &mut episodes {
            for a in &mut episode.treatments {
                a.push(a[0]);
            }
        }
        assert!(gaussian_longitudinal_point(&design, &episodes, &ctx).is_err());
    }
}
