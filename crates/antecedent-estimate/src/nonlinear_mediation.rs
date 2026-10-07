//! Nonlinear continuous-mediator mediation (B4).
//!
//! Estimand (primary): the natural direct and natural indirect effects of a binary treatment
//! `A` on `Y` through a continuous mediator `M`, averaged over the observed covariate rows:
//!
//! * `NDE = E[Y(1, M(0))] - E[Y(0, M(0))]`
//! * `NIE = E[Y(1, M(1))] - E[Y(1, M(0))]`
//! * `TE  = E[Y(1, M(1))] - E[Y(0, M(0))] = NDE + NIE` (an algebraic telescoping identity).
//!
//! Identification needs cross-world independence `Y(a, m) _||_ M(a') | X` together with
//! sequential ignorability: no unmeasured treatment-outcome, treatment-mediator or
//! mediator-outcome confounding given `X`, and no mediator-outcome confounder affected by the
//! treatment. The mediation formula is
//! `theta(a, a') = mean_x ∫ E[Y | a, m, x] dN(m; mu_M(a', x), sigma^2)`.
//!
//! The interventional (randomized-draw) direct and indirect effects, which drop the
//! cross-world assumption and tolerate treatment-induced confounders, are a closed route: asking
//! for them is refused with `nonlinear_mediation.interventional_effects_closed`.
//!
//! Models: the mediator is linear-Gaussian in `(A, X)` with a homoscedastic normal residual;
//! the outcome is a polynomial of declared degree in the standardized mediator with treatment
//! by mediator interaction, linear in `A` and `X`. The integral over `M | a', x` is computed by
//! deterministic Gauss-Hermite quadrature with a declared node count `n`; the estimate is
//! evaluated at `n` and `2n` nodes, the difference is the reported integration error, and an
//! error above the declared tolerance is refused. Standard errors come from a nonparametric
//! bootstrap with a fixed seed and replicate ids. Calibration is `unmeasured`: no coverage
//! claim is made and no public interval is produced.
// SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(
    clippy::needless_range_loop,
    reason = "dense numeric kernels index several parallel columns by row"
)]
use crate::EstimationError;
use crate::splitmix::{mix64, seed_mix, splitmix64};
use std::f64::consts::{PI, SQRT_2};

/// Calibration status of the estimator's inference. Measured only at the release cut.
pub const NONLINEAR_MEDIATION_CALIBRATION: &str = "unmeasured";
/// Interval status: standard errors are reported, public intervals are closed.
pub const NONLINEAR_MEDIATION_INTERVAL_STATUS: &str = "closed_calibration_unmeasured";
/// Replicate-id scheme of the bootstrap.
pub const NONLINEAR_MEDIATION_REPLICATE_SCHEME: &str = "splitmix64 stream seeded by seed_mix(seed ^ mix64(replicate_id + 1)); n row draws with replacement";
/// Largest outcome polynomial degree.
pub const NONLINEAR_MEDIATION_MAX_DEGREE: usize = 4;
/// Largest declared Gauss-Hermite node count (the check evaluates `2n` nodes).
pub const NONLINEAR_MEDIATION_MAX_NODES: usize = 64;

/// Which mediation estimand is requested.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NonlinearMediationEstimand {
    /// Natural direct and indirect effects (cross-world independence + sequential
    /// ignorability). The one open route.
    NaturalEffects,
    /// Interventional (randomized-draw) effects. Closed: refused.
    InterventionalEffects,
}

/// Declared graph premises. Every field is a declaration by the caller; the estimator checks
/// them and refuses when one is incompatible with natural-effect identification.
// Independent declaration flags, not a state machine.
#[allow(clippy::struct_excessive_bools)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NonlinearMediationPremises {
    /// An unmeasured common cause of treatment and outcome is declared.
    pub unmeasured_treatment_outcome_confounding: bool,
    /// An unmeasured common cause of treatment and mediator is declared.
    pub unmeasured_treatment_mediator_confounding: bool,
    /// An unmeasured common cause of mediator and outcome is declared.
    pub unmeasured_mediator_outcome_confounding: bool,
    /// Declared mediator-outcome confounders that are descendants of the treatment.
    pub treatment_induced_mediator_outcome_confounders: Vec<String>,
    /// The cross-world independence `Y(a, m) _||_ M(a') | X` is declared.
    pub cross_world_independence: bool,
}

impl NonlinearMediationPremises {
    /// Premises under which the natural effects are identified: no confounding of any of the
    /// three relations beyond the supplied covariates, no treatment-induced confounder, and the
    /// cross-world independence declared.
    #[must_use]
    pub const fn sequentially_ignorable() -> Self {
        Self {
            unmeasured_treatment_outcome_confounding: false,
            unmeasured_treatment_mediator_confounding: false,
            unmeasured_mediator_outcome_confounding: false,
            treatment_induced_mediator_outcome_confounders: Vec::new(),
            cross_world_independence: true,
        }
    }
}

/// Estimator configuration.
#[derive(Clone, Debug, PartialEq)]
pub struct NonlinearMediationConfig {
    /// Requested estimand.
    pub estimand: NonlinearMediationEstimand,
    /// Degree of the outcome polynomial in the standardized mediator (1 ..= 4).
    pub outcome_degree: usize,
    /// Declared Gauss-Hermite node count `n` (1 ..= 64); the check also runs `2n`.
    pub quadrature_nodes: usize,
    /// Largest accepted integration error, as a fraction of the outcome standard deviation.
    pub integration_tolerance: f64,
    /// Fewest rows required in each treatment arm.
    pub min_arm_count: usize,
    /// Largest accepted share of the cross-world mediator law that falls outside the observed
    /// mediator range of the treated arm.
    pub max_support_violation: f64,
    /// Bootstrap replicates for the standard errors (0 reports none).
    pub bootstrap_replicates: u32,
    /// Bootstrap seed.
    pub seed: u64,
}

impl Default for NonlinearMediationConfig {
    fn default() -> Self {
        Self {
            estimand: NonlinearMediationEstimand::NaturalEffects,
            outcome_degree: 2,
            quadrature_nodes: 16,
            integration_tolerance: 1e-6,
            min_arm_count: 10,
            max_support_violation: 0.25,
            bootstrap_replicates: 200,
            seed: 0x4234_4D45_4449_4154,
        }
    }
}

/// Borrowed analysis columns. Treatment must be exactly 0 or 1.
#[derive(Clone, Copy, Debug)]
pub struct NonlinearMediationInput<'a> {
    /// Binary treatment, coded 0.0 / 1.0.
    pub treatment: &'a [f64],
    /// Continuous mediator.
    pub mediator: &'a [f64],
    /// Outcome.
    pub outcome: &'a [f64],
    /// Measured pre-treatment covariates, one slice per covariate.
    pub covariates: &'a [&'a [f64]],
}

/// Overlap diagnostics.
#[derive(Clone, Debug, PartialEq)]
pub struct NonlinearMediationOverlap {
    /// Rows with treatment 1.
    pub treated_count: usize,
    /// Rows with treatment 0.
    pub control_count: usize,
    /// Observed mediator range of the treated arm, `(min, max)`.
    pub treated_mediator_range: (f64, f64),
    /// Average probability mass of `M | A = 0, X` outside the treated arm's observed mediator
    /// range: the share of the cross-world integral that the outcome model extrapolates.
    pub mediator_support_violation: f64,
}

/// Mediator-model diagnostics.
#[derive(Clone, Debug, PartialEq)]
pub struct NonlinearMediationMediatorDiagnostics {
    /// Mediator coefficients: intercept, treatment, then covariates.
    pub coefficients: Vec<f64>,
    /// Residual variance (degrees-of-freedom corrected).
    pub residual_variance: f64,
    /// Skewness of the standardized residuals (normal law: 0). Reported, not gated.
    pub residual_skewness: f64,
    /// Excess kurtosis of the standardized residuals (normal law: 0). Reported, not gated.
    pub residual_excess_kurtosis: f64,
}

/// Bootstrap record.
#[derive(Clone, Debug, PartialEq)]
pub struct NonlinearMediationBootstrap {
    /// Seed used.
    pub seed: u64,
    /// Replicates requested; replicate ids are `0 .. requested`.
    pub replicates_requested: u32,
    /// Replicates that fit and integrated.
    pub replicates_succeeded: u32,
    /// Ids of replicates that failed (rank deficiency or an unpopulated arm).
    pub failed_replicate_ids: Vec<u64>,
    /// Id scheme.
    pub id_scheme: &'static str,
}

/// Nonlinear mediation estimate.
#[derive(Clone, Debug, PartialEq)]
pub struct NonlinearMediationEstimate {
    /// Natural direct effect `E[Y(1, M(0))] - E[Y(0, M(0))]`.
    pub natural_direct: f64,
    /// Natural indirect effect `E[Y(1, M(1))] - E[Y(1, M(0))]`.
    pub natural_indirect: f64,
    /// Total effect `E[Y(1, M(1))] - E[Y(0, M(0))]`.
    pub total: f64,
    /// Bootstrap standard error of the natural direct effect.
    pub natural_direct_se: Option<f64>,
    /// Bootstrap standard error of the natural indirect effect.
    pub natural_indirect_se: Option<f64>,
    /// Bootstrap standard error of the total effect.
    pub total_se: Option<f64>,
    /// `|total_rowwise - (NDE + NIE)|`, the identity check.
    pub identity_residual: f64,
    /// Largest `n` versus `2n` node disagreement over the three effects, as a fraction of the
    /// outcome standard deviation.
    pub integration_error: f64,
    /// Declared node count `n`.
    pub quadrature_nodes: usize,
    /// Outcome polynomial degree.
    pub outcome_degree: usize,
    /// Rows used.
    pub n_rows: usize,
    /// Overlap diagnostics.
    pub overlap: NonlinearMediationOverlap,
    /// Mediator-model diagnostics.
    pub mediator: NonlinearMediationMediatorDiagnostics,
    /// Bootstrap record.
    pub bootstrap: NonlinearMediationBootstrap,
    /// Always [`NONLINEAR_MEDIATION_CALIBRATION`].
    pub calibration: &'static str,
    /// Always [`NONLINEAR_MEDIATION_INTERVAL_STATUS`]: no public interval is produced.
    pub interval_status: &'static str,
    /// Assumption identifiers the estimate relies on.
    pub assumptions: Vec<&'static str>,
}

fn refuse(code: &'static str, detail: &str, message: &str) -> EstimationError {
    EstimationError::refused(code, format!("{detail}: {message}"))
}

fn count(n: usize) -> f64 {
    f64::from(u32::try_from(n).unwrap_or(u32::MAX))
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / count(v.len())
}

fn sample_sd(v: &[f64]) -> Option<f64> {
    if v.len() < 2 {
        return None;
    }
    let mu = mean(v);
    let ss: f64 = v.iter().map(|x| (x - mu) * (x - mu)).sum();
    Some((ss / (count(v.len()) - 1.0)).sqrt())
}

/// Owned analysis columns.
struct Columns {
    a: Vec<f64>,
    m: Vec<f64>,
    y: Vec<f64>,
    x: Vec<Vec<f64>>,
}

impl Columns {
    fn len(&self) -> usize {
        self.a.len()
    }

    fn resample(&self, idx: &[usize]) -> Self {
        let pick = |c: &[f64]| idx.iter().map(|&i| c[i]).collect::<Vec<f64>>();
        Self {
            a: pick(&self.a),
            m: pick(&self.m),
            y: pick(&self.y),
            x: self.x.iter().map(|c| pick(c)).collect(),
        }
    }
}

fn check_premises(
    premises: &NonlinearMediationPremises,
    config: &NonlinearMediationConfig,
) -> Result<(), EstimationError> {
    let not_identified = antecedent_core::reason_code!("effect_not_identified");
    let cross_world = antecedent_core::reason_code!("cross_world_not_identified");
    if config.estimand == NonlinearMediationEstimand::InterventionalEffects {
        return Err(refuse(
            antecedent_core::reason_code!("route_not_supported"),
            "nonlinear_mediation.interventional_effects_closed",
            "only natural direct and indirect effects are open for the nonlinear continuous mediator",
        ));
    }
    if premises.unmeasured_treatment_outcome_confounding
        || premises.unmeasured_treatment_mediator_confounding
        || premises.unmeasured_mediator_outcome_confounding
    {
        return Err(refuse(
            not_identified,
            "nonlinear_mediation.confounding",
            "an unmeasured treatment-outcome, treatment-mediator or mediator-outcome confounder is declared; sequential ignorability fails",
        ));
    }
    if !premises.treatment_induced_mediator_outcome_confounders.is_empty() {
        return Err(refuse(
            cross_world,
            "nonlinear_mediation.treatment_induced_confounding",
            &format!(
                "mediator-outcome confounders affected by treatment ({}) break natural-effect identification",
                premises.treatment_induced_mediator_outcome_confounders.join(", ")
            ),
        ));
    }
    if !premises.cross_world_independence {
        return Err(refuse(
            cross_world,
            "nonlinear_mediation.cross_world_independence_not_declared",
            "natural effects need the cross-world independence declared",
        ));
    }
    Ok(())
}

fn validate(
    input: &NonlinearMediationInput<'_>,
    config: &NonlinearMediationConfig,
) -> Result<Columns, EstimationError> {
    let invalid = antecedent_core::reason_code!("invalid_argument");
    let n = input.treatment.len();
    if input.mediator.len() != n
        || input.outcome.len() != n
        || input.covariates.iter().any(|c| c.len() != n)
    {
        return Err(refuse(
            invalid,
            "nonlinear_mediation.length_mismatch",
            "columns differ in length",
        ));
    }
    if config.outcome_degree == 0 || config.outcome_degree > NONLINEAR_MEDIATION_MAX_DEGREE {
        return Err(refuse(
            invalid,
            "nonlinear_mediation.invalid_degree",
            "outcome degree must lie in 1..=4",
        ));
    }
    if config.quadrature_nodes == 0 || config.quadrature_nodes > NONLINEAR_MEDIATION_MAX_NODES {
        return Err(refuse(
            invalid,
            "nonlinear_mediation.invalid_nodes",
            "node count must lie in 1..=64",
        ));
    }
    if !(config.integration_tolerance.is_finite() && config.integration_tolerance > 0.0) {
        return Err(refuse(
            invalid,
            "nonlinear_mediation.invalid_tolerance",
            "tolerance must be positive and finite",
        ));
    }
    if !(config.max_support_violation > 0.0 && config.max_support_violation <= 1.0) {
        return Err(refuse(
            invalid,
            "nonlinear_mediation.invalid_support_bound",
            "support bound must lie in (0, 1]",
        ));
    }
    let finite = |c: &[f64]| c.iter().all(|v| v.is_finite());
    if !(finite(input.treatment)
        && finite(input.mediator)
        && finite(input.outcome)
        && input.covariates.iter().all(|c| finite(c)))
    {
        return Err(refuse(
            invalid,
            "nonlinear_mediation.non_finite_input",
            "columns must be finite",
        ));
    }
    if !input.treatment.iter().all(|&v| v.abs() < 1e-12 || (v - 1.0).abs() < 1e-12) {
        return Err(refuse(
            invalid,
            "nonlinear_mediation.treatment_not_binary",
            "treatment must be coded 0 or 1",
        ));
    }
    Ok(Columns {
        a: input.treatment.iter().map(|&v| if v > 0.5 { 1.0 } else { 0.0 }).collect(),
        m: input.mediator.to_vec(),
        y: input.outcome.to_vec(),
        x: input.covariates.iter().map(|c| c.to_vec()).collect(),
    })
}

fn check_arms(
    cols: &Columns,
    config: &NonlinearMediationConfig,
) -> Result<(usize, usize), EstimationError> {
    let treated = cols.a.iter().filter(|&&v| v > 0.5).count();
    let control = cols.len() - treated;
    if treated < config.min_arm_count.max(1) || control < config.min_arm_count.max(1) {
        return Err(refuse(
            antecedent_core::reason_code!("arm_not_populated"),
            "nonlinear_mediation.overlap",
            &format!(
                "treatment arms hold {treated} treated and {control} control rows, below the required {}",
                config.min_arm_count.max(1)
            ),
        ));
    }
    Ok((treated, control))
}

/// Least squares by Jacobi-scaled normal equations with partial-pivot elimination.
struct Ols {
    beta: Vec<f64>,
    residuals: Vec<f64>,
    rss: f64,
}

fn ols(cols: &[Vec<f64>], y: &[f64]) -> Result<Ols, EstimationError> {
    let rank_code = antecedent_core::reason_code!("design_rank_deficient");
    let n = y.len();
    let p = cols.len();
    if n <= p {
        return Err(refuse(
            rank_code,
            "nonlinear_mediation.too_few_rows",
            "fewer rows than regression columns",
        ));
    }
    let nf = count(n);
    let mut scales = Vec::with_capacity(p);
    for c in cols {
        let s = (c.iter().map(|v| v * v).sum::<f64>() / nf).sqrt();
        if !(s.is_finite() && s > 1e-12) {
            return Err(refuse(
                rank_code,
                "nonlinear_mediation.rank_deficient",
                "a regression column is identically zero",
            ));
        }
        scales.push(s);
    }
    let mut aug = vec![0.0; p * (p + 1)];
    for j in 0..p {
        for k in j..p {
            let dot: f64 = cols[j].iter().zip(&cols[k]).map(|(u, v)| u * v).sum();
            let v = dot / (scales[j] * scales[k]);
            aug[j * (p + 1) + k] = v;
            aug[k * (p + 1) + j] = v;
        }
        let r: f64 = cols[j].iter().zip(y).map(|(u, v)| u * v).sum();
        aug[j * (p + 1) + p] = r / scales[j];
    }
    let sol = solve(&mut aug, p, nf).ok_or_else(|| {
        refuse(
            rank_code,
            "nonlinear_mediation.rank_deficient",
            "regression design is rank deficient",
        )
    })?;
    let beta: Vec<f64> = sol.iter().zip(&scales).map(|(b, s)| b / s).collect();
    let mut residuals = Vec::with_capacity(n);
    let mut rss = 0.0;
    for i in 0..n {
        let fitted: f64 = cols.iter().zip(&beta).map(|(c, b)| c[i] * b).sum();
        let r = y[i] - fitted;
        rss += r * r;
        residuals.push(r);
    }
    Ok(Ols { beta, residuals, rss })
}

/// Gauss-Jordan on an augmented `p x (p + 1)` row-major system. `None` when a pivot falls
/// below `1e-10 * diag_scale` (the Jacobi-scaled Gram diagonal equals the row count).
fn solve(aug: &mut [f64], p: usize, diag_scale: f64) -> Option<Vec<f64>> {
    let w = p + 1;
    for col in 0..p {
        let mut best = col;
        for r in col + 1..p {
            if aug[r * w + col].abs() > aug[best * w + col].abs() {
                best = r;
            }
        }
        if aug[best * w + col].abs().partial_cmp(&(1e-10 * diag_scale)) != Some(std::cmp::Ordering::Greater) {
            return None;
        }
        if best != col {
            for k in 0..w {
                aug.swap(col * w + k, best * w + k);
            }
        }
        let pivot = aug[col * w + col];
        for r in 0..p {
            if r == col {
                continue;
            }
            let factor = aug[r * w + col] / pivot;
            if factor.abs() > 0.0 {
                for k in col..w {
                    let sub = factor * aug[col * w + k];
                    aug[r * w + k] -= sub;
                }
            }
        }
    }
    let sol: Vec<f64> = (0..p).map(|j| aug[j * w + p] / aug[j * w + j]).collect();
    sol.iter().all(|v| v.is_finite()).then_some(sol)
}

/// Fitted mediator and outcome models.
struct Fit {
    med: Vec<f64>,
    sigma: f64,
    out: Vec<f64>,
    center: f64,
    scale: f64,
    degree: usize,
    n_x: usize,
    residuals: Vec<f64>,
}

fn fit_model(cols: &Columns, degree: usize) -> Result<Fit, EstimationError> {
    let invalid = antecedent_core::reason_code!("invalid_argument");
    let n = cols.len();
    let center = mean(&cols.m);
    let scale = sample_sd(&cols.m).unwrap_or(0.0);
    if !(scale.is_finite() && scale > 1e-12) {
        return Err(refuse(
            invalid,
            "nonlinear_mediation.degenerate_mediator",
            "the mediator has no spread",
        ));
    }
    let mut med_design = vec![vec![1.0; n], cols.a.clone()];
    med_design.extend(cols.x.iter().cloned());
    let med = ols(&med_design, &cols.m)?;
    let sigma2 = med.rss / (count(n) - count(med_design.len()));
    if !(sigma2.is_finite() && sigma2 > 1e-12 * scale * scale) {
        return Err(refuse(
            invalid,
            "nonlinear_mediation.degenerate_mediator_residual",
            "the mediator residual has no spread",
        ));
    }
    let z: Vec<f64> = cols.m.iter().map(|&m| (m - center) / scale).collect();
    let mut out_design = vec![vec![1.0; n], cols.a.clone()];
    out_design.extend(cols.x.iter().cloned());
    let mut pow = vec![1.0; n];
    let mut z_cols: Vec<Vec<f64>> = Vec::with_capacity(degree);
    for _ in 0..degree {
        for (p, zi) in pow.iter_mut().zip(&z) {
            *p *= zi;
        }
        z_cols.push(pow.clone());
    }
    let interactions: Vec<Vec<f64>> =
        z_cols.iter().map(|zc| zc.iter().zip(&cols.a).map(|(zv, av)| zv * av).collect()).collect();
    out_design.extend(z_cols);
    out_design.extend(interactions);
    let out = ols(&out_design, &cols.y)?;
    Ok(Fit {
        med: med.beta,
        sigma: sigma2.sqrt(),
        out: out.beta,
        center,
        scale,
        degree,
        n_x: cols.x.len(),
        residuals: med.residuals,
    })
}

/// Gauss-Hermite rule for the standard normal: nodes `t` and weights summing to one, so that
/// `E[f(mu + sigma * sqrt(2) * t)] ~ sum_j w_j f(...)`.
struct Quadrature {
    nodes: Vec<f64>,
    weights: Vec<f64>,
}

/// Value and derivative scale of the orthonormal Hermite recursion at `z`: returns
/// `(h_n(z), sqrt(2n) h_{n-1}(z))`.
fn hermite_eval(n: usize, z: f64) -> (f64, f64) {
    let mut p1 = PI.powf(-0.25);
    let mut p2 = 0.0;
    for j in 0..n {
        let p3 = p2;
        p2 = p1;
        let jf = count(j);
        p1 = z * (2.0 / (jf + 1.0)).sqrt() * p2 - (jf / (jf + 1.0)).sqrt() * p3;
    }
    (p1, (2.0 * count(n)).sqrt() * p2)
}

fn gauss_hermite(n: usize) -> Result<Quadrature, EstimationError> {
    let nf = count(n);
    let mut x = vec![0.0; n];
    let mut w = vec![0.0; n];
    let mut z = 0.0_f64;
    for i in 0..n.div_ceil(2) {
        z = match i {
            0 => (2.0 * nf + 1.0).sqrt() - 1.855_75 * (2.0 * nf + 1.0).powf(-1.0 / 6.0),
            1 => z - 1.14 * nf.powf(0.426) / z,
            2 => 1.86 * z - 0.86 * x[0],
            3 => 1.91 * z - 0.91 * x[1],
            _ => 2.0 * z - x[i - 2],
        };
        let mut converged = false;
        for _ in 0..40 {
            let (p1, pp) = hermite_eval(n, z);
            let next = z - p1 / pp;
            let done = (next - z).abs() <= 1e-14 * (1.0 + z.abs());
            z = next;
            if done {
                converged = true;
                break;
            }
        }
        if !converged || !z.is_finite() {
            return Err(refuse(
                antecedent_core::reason_code!("mechanism_fit_not_converged"),
                "nonlinear_mediation.quadrature_nodes_not_converged",
                "Gauss-Hermite node iteration did not converge",
            ));
        }
        let mirror = n - 1 - i;
        if mirror == i {
            z = 0.0;
        }
        let (_, pp) = hermite_eval(n, z);
        let weight = 2.0 / (pp * pp);
        x[i] = z;
        w[i] = weight;
        x[mirror] = -z;
        w[mirror] = weight;
    }
    let norm = PI.sqrt();
    let weights: Vec<f64> = w.iter().map(|v| v / norm).collect();
    let total: f64 = weights.iter().sum();
    if (total - 1.0).abs().partial_cmp(&1e-9) != Some(std::cmp::Ordering::Less) {
        return Err(refuse(
            antecedent_core::reason_code!("mechanism_fit_not_converged"),
            "nonlinear_mediation.quadrature_weights_invalid",
            "Gauss-Hermite weights do not sum to one",
        ));
    }
    Ok(Quadrature { nodes: x, weights })
}

fn row_dot(x: &[Vec<f64>], i: usize, coefs: &[f64]) -> f64 {
    x.iter().zip(coefs).map(|(c, b)| c[i] * b).sum()
}

/// Outcome regression at treatment `a`, mediator `m`, covariate part `xb`.
fn outcome_at(fit: &Fit, xb: f64, a: f64, m: f64) -> f64 {
    let z = (m - fit.center) / fit.scale;
    let k = fit.degree;
    let base = 2 + fit.n_x;
    let mut acc = fit.out[0] + fit.out[1] * a + xb;
    let mut pow = 1.0;
    for d in 0..k {
        pow *= z;
        acc += (fit.out[base + d] + fit.out[base + k + d] * a) * pow;
    }
    acc
}

/// `∫ E[Y | a, m, x] dN(m; mu_M(a', x), sigma^2)` for one row.
fn row_theta(fit: &Fit, xb: f64, xm: f64, a: f64, a_med: f64, q: &Quadrature) -> f64 {
    let mu = fit.med[0] + fit.med[1] * a_med + xm;
    q.nodes
        .iter()
        .zip(&q.weights)
        .map(|(t, w)| w * outcome_at(fit, xb, a, mu + fit.sigma * SQRT_2 * t))
        .sum()
}

struct Thetas {
    t11: f64,
    t10: f64,
    t00: f64,
    total_rowwise: f64,
}

struct Effects {
    direct: f64,
    indirect: f64,
    total: f64,
}

fn thetas(cols: &Columns, fit: &Fit, q: &Quadrature) -> Thetas {
    let n = cols.len();
    let (mut s11, mut s10, mut s00, mut st) = (0.0, 0.0, 0.0, 0.0);
    for i in 0..n {
        let xb = row_dot(&cols.x, i, &fit.out[2..]);
        let xm = row_dot(&cols.x, i, &fit.med[2..]);
        let r11 = row_theta(fit, xb, xm, 1.0, 1.0, q);
        let r10 = row_theta(fit, xb, xm, 1.0, 0.0, q);
        let r00 = row_theta(fit, xb, xm, 0.0, 0.0, q);
        s11 += r11;
        s10 += r10;
        s00 += r00;
        st += r11 - r00;
    }
    let nf = count(n);
    Thetas { t11: s11 / nf, t10: s10 / nf, t00: s00 / nf, total_rowwise: st / nf }
}

fn effects(t: &Thetas) -> Effects {
    Effects { direct: t.t10 - t.t00, indirect: t.t11 - t.t10, total: t.t11 - t.t00 }
}

fn treated_range(cols: &Columns) -> (f64, f64) {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for (m, a) in cols.m.iter().zip(&cols.a) {
        if *a > 0.5 {
            lo = lo.min(*m);
            hi = hi.max(*m);
        }
    }
    (lo, hi)
}

/// Average mass of `M | A = 0, X` outside the treated arm's observed mediator range.
fn support_violation(cols: &Columns, fit: &Fit, q: &Quadrature, range: (f64, f64)) -> f64 {
    let n = cols.len();
    let mut mass = 0.0;
    for i in 0..n {
        let mu = fit.med[0] + row_dot(&cols.x, i, &fit.med[2..]);
        for (t, w) in q.nodes.iter().zip(&q.weights) {
            let m = mu + fit.sigma * SQRT_2 * t;
            if m < range.0 || m > range.1 {
                mass += w;
            }
        }
    }
    mass / count(n)
}

fn residual_shape(fit: &Fit) -> (f64, f64) {
    let e: Vec<f64> = fit.residuals.iter().map(|r| r / fit.sigma).collect();
    let skew = mean(&e.iter().map(|v| v * v * v).collect::<Vec<f64>>());
    let kurt = mean(&e.iter().map(|v| v * v * v * v).collect::<Vec<f64>>()) - 3.0;
    (skew, kurt)
}

struct BootstrapOutcome {
    record: NonlinearMediationBootstrap,
    se: Option<(f64, f64, f64)>,
}

fn run_bootstrap(
    cols: &Columns,
    config: &NonlinearMediationConfig,
    q: &Quadrature,
) -> BootstrapOutcome {
    let n = cols.len();
    let n64 = u64::try_from(n).unwrap_or(u64::MAX).max(1);
    let (mut d, mut i_, mut t) = (Vec::new(), Vec::new(), Vec::new());
    let mut failed = Vec::new();
    for r in 0..config.bootstrap_replicates {
        let id = u64::from(r);
        let mut state = seed_mix(config.seed ^ mix64(id.wrapping_add(1)));
        let idx: Vec<usize> =
            (0..n).map(|_| usize::try_from(splitmix64(&mut state) % n64).unwrap_or(0)).collect();
        let sample = cols.resample(&idx);
        let fitted = fit_model(&sample, config.outcome_degree);
        match fitted {
            Ok(fit) => {
                let e = effects(&thetas(&sample, &fit, q));
                if e.direct.is_finite() && e.indirect.is_finite() && e.total.is_finite() {
                    d.push(e.direct);
                    i_.push(e.indirect);
                    t.push(e.total);
                } else {
                    failed.push(id);
                }
            }
            Err(_) => failed.push(id),
        }
    }
    let ok = u32::try_from(d.len()).unwrap_or(u32::MAX);
    let enough = config.bootstrap_replicates > 0
        && ok >= 2
        && u64::from(ok) * 2 >= u64::from(config.bootstrap_replicates);
    let se = if enough {
        match (sample_sd(&d), sample_sd(&i_), sample_sd(&t)) {
            (Some(a), Some(b), Some(c)) => Some((a, b, c)),
            _ => None,
        }
    } else {
        None
    };
    BootstrapOutcome {
        record: NonlinearMediationBootstrap {
            seed: config.seed,
            replicates_requested: config.bootstrap_replicates,
            replicates_succeeded: ok,
            failed_replicate_ids: failed,
            id_scheme: NONLINEAR_MEDIATION_REPLICATE_SCHEME,
        },
        se,
    }
}

/// Estimate the natural direct, indirect and total effects of a binary treatment through a
/// continuous mediator with a nonlinear outcome model.
///
/// The returned point estimates use `2n` Gauss-Hermite nodes; `integration_error` is the
/// disagreement with the `n`-node evaluation. Calibration is `unmeasured` and no public
/// interval is produced.
///
/// # Errors
/// Refuses with `nonlinear_mediation.confounding`,
/// `nonlinear_mediation.treatment_induced_confounding`,
/// `nonlinear_mediation.cross_world_independence_not_declared`,
/// `nonlinear_mediation.interventional_effects_closed`, `nonlinear_mediation.overlap`,
/// `nonlinear_mediation.integration_error`, `nonlinear_mediation.identity_violation`,
/// `nonlinear_mediation.rank_deficient` or an input-validation detail.
pub fn estimate_nonlinear_mediation(
    input: &NonlinearMediationInput<'_>,
    premises: &NonlinearMediationPremises,
    config: &NonlinearMediationConfig,
) -> Result<NonlinearMediationEstimate, EstimationError> {
    check_premises(premises, config)?;
    let cols = validate(input, config)?;
    let (treated_count, control_count) = check_arms(&cols, config)?;
    let fit = fit_model(&cols, config.outcome_degree)?;
    let coarse_q = gauss_hermite(config.quadrature_nodes)?;
    let fine_q = gauss_hermite(2 * config.quadrature_nodes)?;
    let range = treated_range(&cols);
    let violation = support_violation(&cols, &fit, &fine_q, range);
    if violation > config.max_support_violation {
        return Err(refuse(
            antecedent_core::reason_code!("effect_not_identified"),
            "nonlinear_mediation.overlap",
            &format!(
                "{violation:.4} of the cross-world mediator law lies outside the treated arm's observed mediator range, above {}",
                config.max_support_violation
            ),
        ));
    }
    let coarse = effects(&thetas(&cols, &fit, &coarse_q));
    let fine_thetas = thetas(&cols, &fit, &fine_q);
    let fine = effects(&fine_thetas);
    let sd_y = sample_sd(&cols.y).unwrap_or(0.0);
    if !(sd_y.is_finite() && sd_y > 1e-12) {
        return Err(refuse(
            antecedent_core::reason_code!("invalid_argument"),
            "nonlinear_mediation.degenerate_outcome",
            "the outcome has no spread",
        ));
    }
    let integration_error = (coarse.direct - fine.direct)
        .abs()
        .max((coarse.indirect - fine.indirect).abs())
        .max((coarse.total - fine.total).abs())
        / sd_y;
    if !matches!(
        integration_error.partial_cmp(&config.integration_tolerance),
        Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
    ) {
        return Err(refuse(
            antecedent_core::reason_code!("mechanism_fit_not_converged"),
            "nonlinear_mediation.integration_error",
            &format!(
                "{} versus {} node effects differ by {integration_error:.3e} of the outcome sd, above {}",
                config.quadrature_nodes,
                2 * config.quadrature_nodes,
                config.integration_tolerance
            ),
        ));
    }
    let identity_residual = (fine_thetas.total_rowwise - (fine.direct + fine.indirect)).abs();
    if !matches!(
        identity_residual.partial_cmp(&(1e-9 * (1.0 + fine.total.abs()))),
        Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
    ) {
        return Err(refuse(
            antecedent_core::reason_code!("mechanism_fit_not_converged"),
            "nonlinear_mediation.identity_violation",
            "total effect differs from direct plus indirect",
        ));
    }
    let boot = run_bootstrap(&cols, config, &fine_q);
    let (skew, kurt) = residual_shape(&fit);
    Ok(NonlinearMediationEstimate {
        natural_direct: fine.direct,
        natural_indirect: fine.indirect,
        total: fine.total,
        natural_direct_se: boot.se.map(|s| s.0),
        natural_indirect_se: boot.se.map(|s| s.1),
        total_se: boot.se.map(|s| s.2),
        identity_residual,
        integration_error,
        quadrature_nodes: config.quadrature_nodes,
        outcome_degree: config.outcome_degree,
        n_rows: cols.len(),
        overlap: NonlinearMediationOverlap {
            treated_count,
            control_count,
            treated_mediator_range: range,
            mediator_support_violation: violation,
        },
        mediator: NonlinearMediationMediatorDiagnostics {
            coefficients: fit.med.clone(),
            residual_variance: fit.sigma * fit.sigma,
            residual_skewness: skew,
            residual_excess_kurtosis: kurt,
        },
        bootstrap: boot.record,
        calibration: NONLINEAR_MEDIATION_CALIBRATION,
        interval_status: NONLINEAR_MEDIATION_INTERVAL_STATUS,
        assumptions: vec![
            "nonlinear_mediation.sequential_ignorability",
            "nonlinear_mediation.cross_world_independence",
            "nonlinear_mediation.no_treatment_induced_mediator_outcome_confounding",
            "nonlinear_mediation.linear_gaussian_mediator",
            "nonlinear_mediation.polynomial_outcome_in_mediator",
        ],
    })
}
