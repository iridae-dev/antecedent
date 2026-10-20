//! Untempered Gaussian-likelihood posterior of a single-step temporal effect.
//!
//! The Bayesian Pulse and Sustained cells publish a generalized (power) posterior: their Gaussian
//! likelihood on time-ordered rows is tempered by the long-run-variance ratio, because the
//! lag-aligned regression residual may be serially dependent. That meaning is the honest default
//! and nothing here changes it.
//!
//! A full-likelihood posterior of the same effect, with the likelihood untempered, is a
//! different object and is licensed only under one more premise: the outcome innovation is
//! serially independent given the declared lag window and adjustment set. Then the lag-aligned
//! regression residual is that innovation, the rows factor as a product of independent Gaussian
//! terms, and the conjugate or Laplace posterior is the posterior of a correctly specified model.
//!
//! The premise is declared by the caller and is never established by this module. A
//! Ljung–Box test of the least-squares residuals refutes it when it can: a rejection is a
//! refusal that names the premise and points to the tempered posterior. A pass does not prove
//! whiteness. The test has finite power, and residuals of a regression that contains a lagged
//! dependent variable have attenuated autocorrelation, so a pass is not evidence for the premise.
//!
//! Only a single step ahead is accepted. An unfolded regression at horizon `h ≥ 2` omits the
//! intermediate treatments and innovations, so its residual is moving-average of order `h − 1`
//! by construction, whatever the innovation process. A multi-step window needs per-mechanism
//! posteriors composed through the structural equations instead.
//!
//! Rows must be the contiguous lag windows of one series, in time order. A gapped series is not
//! accepted here.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg_attr(
    test,
    allow(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::float_cmp,
        reason = "test fixtures compare exact constants and index with small literals"
    )
)]

use std::sync::Arc;

use antecedent_core::{
    Assumption, AssumptionRecord, AssumptionScope, AssumptionSource, AssumptionStatus,
    ExecutionContext, IdentificationStatus, ParametricAssumption,
};
use antecedent_prob::BayesLikelihood;
use antecedent_stats::{
    CompiledDesign, DenseLinearAlgebra, FaerBackend, LeastSquaresWorkspace, chi_square_sf,
};

use crate::error::EstimationError;
use crate::serial_dependence::SerialDependence;
use crate::util::refuse;
use crate::{
    BayesianGCompWorkspace, BayesianGComputationAte, CausalPosterior, PreparedEstimationProblem,
};

/// Assumption id recorded on a posterior fitted under the white-innovation premise.
pub const WHITE_INNOVATION_ASSUMPTION_ID: &str = "bayes.temporal.white_innovation_likelihood";

/// Stable prefix of the diagnostics note that records the residual test.
pub const WHITE_INNOVATION_NOTE_PREFIX: &str = "white_innovation.ljung_box";

/// Residual autocorrelation test that can refute the white-innovation premise.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InnovationWhitenessGate {
    /// Number of residual autocorrelations the Ljung–Box statistic sums. At least 1.
    pub max_lag: usize,
    /// Level of the test, strictly between 0 and 1. A p-value below it refuses the fit.
    pub level: f64,
}

/// What the Ljung–Box test of the residuals found.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InnovationWhitenessReport {
    /// Ljung–Box statistic.
    pub q_statistic: f64,
    /// Autocorrelations summed, which is also the degrees of freedom.
    pub lags: usize,
    /// Upper-tail chi-square probability of the statistic.
    pub p_value: f64,
    /// Level the gate used.
    pub level: f64,
    /// Rows tested.
    pub rows: usize,
}

/// The premise the caller declares and the test that may refute it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WhiteInnovationDeclaration {
    /// Steps from treatment to outcome. Only 1 is accepted.
    pub horizon_steps: u32,
    /// Refutation test of the premise.
    pub gate: InnovationWhitenessGate,
}

/// Untempered posterior with the residual test that did not refute its premise.
#[derive(Clone, Debug)]
pub struct WhiteInnovationPosterior {
    /// Posterior of the effect under the iid Gaussian likelihood.
    pub posterior: CausalPosterior,
    /// Result of the residual test. A pass is not a proof of the premise.
    pub whiteness: InnovationWhitenessReport,
}

/// Ljung–Box statistic of the least-squares residuals of `design`.
///
/// # Errors
///
/// An invalid gate, too few rows for the lags, or a failed least-squares solve.
pub fn innovation_whiteness_of(
    design: &CompiledDesign,
    gate: &InnovationWhitenessGate,
) -> Result<InnovationWhitenessReport, EstimationError> {
    if gate.max_lag == 0 || !(gate.level > 0.0 && gate.level < 1.0) {
        return Err(invalid("the whiteness gate needs max_lag >= 1 and a level in (0, 1)"));
    }
    let n = design.nrows;
    let p = design.ncols;
    if n < p + 2 || n <= 2 * gate.max_lag {
        return Err(invalid("too few rows for the requested residual autocorrelation lags"));
    }
    let x = design.matrix.get(..n * p).ok_or_else(|| invalid("design matrix is shorter"))?;
    let y = design.outcome.get(..n).ok_or_else(|| invalid("outcome is shorter"))?;
    let residuals =
        FaerBackend.least_squares(x, n, p, y, &mut LeastSquaresWorkspace::default())?.residuals;
    ljung_box(&residuals, gate)
}

fn ljung_box(
    residuals: &[f64],
    gate: &InnovationWhitenessGate,
) -> Result<InnovationWhitenessReport, EstimationError> {
    let n = residuals.len();
    let total: f64 = residuals.iter().map(|e| e * e).sum();
    if !total.is_finite() {
        return Err(invalid("residuals are not finite"));
    }
    let n_f = n as f64;
    let mut q = 0.0_f64;
    if total > 0.0 {
        for lag in 1..=gate.max_lag {
            let cross: f64 = residuals.iter().skip(lag).zip(residuals).map(|(a, b)| a * b).sum();
            let r = cross / total;
            q += r * r / (n_f - lag as f64);
        }
        q *= n_f * (n_f + 2.0);
    }
    Ok(InnovationWhitenessReport {
        q_statistic: q,
        lags: gate.max_lag,
        p_value: chi_square_sf(q, gate.max_lag),
        level: gate.level,
        rows: n,
    })
}

/// Fit the iid Gaussian-likelihood posterior of a single-step temporal effect.
///
/// `prep` is the lag-aligned design of [`crate::TemporalLinearAdjustment`]. The estimator must
/// use the Gaussian identity likelihood. The returned posterior carries an assumption record
/// and a diagnostics note, so it cannot be read as the tempered posterior.
///
/// # Errors
///
/// A horizon other than one step, a non-Gaussian likelihood, an invalid gate, residuals whose
/// Ljung–Box test rejects whiteness (the refusal points to the tempered posterior), or a failed
/// fit.
pub fn fit_white_innovation_posterior(
    estimator: &BayesianGComputationAte,
    prep: &PreparedEstimationProblem,
    declaration: &WhiteInnovationDeclaration,
    identification: IdentificationStatus,
    ctx: &ExecutionContext,
) -> Result<WhiteInnovationPosterior, EstimationError> {
    if declaration.horizon_steps != 1 {
        return Err(refuse(
            antecedent_core::reason_code!("route_not_supported"),
            "white_innovation.multi_step",
            "an unfolded regression at horizon h >= 2 has moving-average residuals of order h - 1 \
             whatever the innovation process, so an untempered likelihood is not the model; \
             next attempt: the tempered posterior, or per-mechanism posteriors composed through \
             the structural equations",
        ));
    }
    if estimator.effective_likelihood() != BayesLikelihood::GaussianIdentity {
        return Err(refuse(
            antecedent_core::reason_code!("route_not_supported"),
            "white_innovation.likelihood",
            "the temporal linear effect is the Gaussian identity-link model",
        ));
    }
    let whiteness = innovation_whiteness_of(&prep.design, &declaration.gate)?;
    if whiteness.p_value < whiteness.level {
        return Err(refuse(
            antecedent_core::reason_code!("route_not_supported"),
            "white_innovation.residual_autocorrelation",
            &format!(
                "Ljung-Box Q = {:.3} on {} lags (p = {:.4}) rejects serially independent \
                 innovations at level {}; the untempered likelihood is not licensed. Next \
                 attempt: widen the lag window or adjustment set until the residual is white, \
                 or use the tempered posterior, whose meaning is a generalized posterior",
                whiteness.q_statistic, whiteness.lags, whiteness.p_value, whiteness.level
            ),
        ));
    }
    let problem = BayesianGComputationAte::from_prepared_estimation(prep)
        .with_serial_dependence(SerialDependence::Iid);
    let mut posterior =
        estimator.fit(&problem, identification, &mut BayesianGCompWorkspace::default(), ctx)?;
    posterior.diagnostics.notes.push(Arc::from(format!(
        "{WHITE_INNOVATION_NOTE_PREFIX}: q={:.6} lags={} p={:.6} level={} rows={}",
        whiteness.q_statistic, whiteness.lags, whiteness.p_value, whiteness.level, whiteness.rows
    )));
    posterior.assumptions.push(white_innovation_assumption());
    Ok(WhiteInnovationPosterior { posterior, whiteness })
}

fn white_innovation_assumption() -> AssumptionRecord {
    AssumptionRecord {
        assumption: Assumption::ParametricRestriction(ParametricAssumption {
            id: Arc::from(WHITE_INNOVATION_ASSUMPTION_ID),
            description: Arc::from(
                "full-likelihood posterior: the outcome innovation is Gaussian, serially \
                 independent and homoskedastic given the declared lag window and adjustment set, \
                 so the lag-aligned rows are independent terms and the likelihood is untempered. \
                 Declared by the caller; a Ljung-Box test of the residuals can refute it and a \
                 pass does not prove it. Single step ahead only. Not the generalized posterior of \
                 the tempered Pulse and Sustained cells",
            ),
        }),
        source: AssumptionSource::AlgorithmDefault {
            algorithm: Arc::from("estimate.temporal.white_innovation_posterior"),
        },
        scope: AssumptionScope::Estimation,
        status: AssumptionStatus::Declared,
    }
}

fn invalid(message: &str) -> EstimationError {
    refuse(
        antecedent_core::reason_code!("invalid_argument"),
        "white_innovation.invalid_argument",
        message,
    )
}

#[cfg(test)]
mod tests {
    use antecedent_core::{TargetPopulation, VariableId};

    use super::*;
    use crate::overlap::OverlapPolicy;

    struct Lcg(u64);

    impl Lcg {
        fn uniform(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((self.0 >> 11) as f64 + 0.5) / (1_u64 << 53) as f64
        }

        fn normal(&mut self) -> f64 {
            let u1 = self.uniform();
            let u2 = self.uniform();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        }
    }

    /// Series `y_t = 2 t_t + 0.5 z_t + e_t` with treatment and covariate persistent when
    /// `rho_x > 0` and the residual `e` AR(1) with coefficient `rho_e`.
    fn series(n: usize, rho_x: f64, rho_e: f64, seed: u64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut rng = Lcg(seed);
        let (mut t, mut z, mut e) = (0.0_f64, 0.0_f64, 0.0_f64);
        let (mut ts, mut zs, mut ys) = (Vec::new(), Vec::new(), Vec::new());
        for _ in 0..n {
            t = rho_x * t + rng.normal();
            z = rho_x * z + rng.normal();
            e = rho_e * e + rng.normal();
            ts.push(t);
            zs.push(z);
            ys.push(2.0 * t + 0.5 * z + e);
        }
        (ts, zs, ys)
    }

    fn prepared(t: &[f64], z: &[f64], y: &[f64]) -> PreparedEstimationProblem {
        let design = CompiledDesign::linear_adjustment(t, &[(VariableId::from_raw(1), z)], y, &[])
            .expect("design");
        PreparedEstimationProblem {
            design,
            method: Arc::from("temporal.linear.adjustment"),
            adjustment_set: Arc::from(vec![VariableId::from_raw(1)]),
            overlap: OverlapPolicy::ExplicitOverride,
            treatment_delta: 1.0,
            target_population: TargetPopulation::AllObserved,
            treatment: Arc::from(t.to_vec()),
            active: 1.0,
            control: 0.0,
            sampling_weights: None,
        }
    }

    fn declaration(horizon_steps: u32) -> WhiteInnovationDeclaration {
        WhiteInnovationDeclaration {
            horizon_steps,
            gate: InnovationWhitenessGate { max_lag: 6, level: 0.01 },
        }
    }

    fn estimator() -> BayesianGComputationAte {
        BayesianGComputationAte {
            prior_scale: 1.0e3,
            n_draws: 4000,
            seed: 11,
            ..BayesianGComputationAte::conjugate()
        }
    }

    /// Coefficient on `t` and its OLS standard error by Frisch–Waugh: partial the intercept and
    /// `z` out of `t`, then the residual variance of the full model.
    fn ols_treatment(t: &[f64], z: &[f64], y: &[f64]) -> (f64, f64) {
        let n = t.len() as f64;
        let mean = |v: &[f64]| v.iter().sum::<f64>() / n;
        let (tm, zm) = (mean(t), mean(z));
        let szz: f64 = z.iter().map(|v| (v - zm).powi(2)).sum();
        let stz: f64 = t.iter().zip(z).map(|(a, b)| (a - tm) * (b - zm)).sum();
        let r: Vec<f64> = t.iter().zip(z).map(|(a, b)| (a - tm) - stz / szz * (b - zm)).collect();
        let srr: f64 = r.iter().map(|v| v * v).sum();
        let sry: f64 = r.iter().zip(y).map(|(a, b)| a * b).sum();
        let beta = sry / srr;
        let adjusted: Vec<f64> = y.iter().zip(t).map(|(yy, tt)| yy - beta * tt).collect();
        let am = mean(&adjusted);
        let sza: f64 = z.iter().zip(&adjusted).map(|(a, b)| (a - zm) * (b - am)).sum();
        let slope = sza / szz;
        let rss: f64 =
            z.iter().zip(&adjusted).map(|(a, b)| (b - am - slope * (a - zm)).powi(2)).sum();
        (beta, (rss / (n - 3.0) / srr).sqrt())
    }

    #[test]
    fn white_innovations_give_the_ols_posterior() {
        let (t, z, y) = series(400, 0.0, 0.0, 7);
        let prep = prepared(&t, &z, &y);
        let fit = fit_white_innovation_posterior(
            &estimator(),
            &prep,
            &declaration(1),
            IdentificationStatus::NonparametricallyIdentified,
            &ExecutionContext::for_tests(1),
        )
        .expect("white residuals pass the gate");
        let eq = fit.posterior.effect_column().expect("effect column");
        let (beta, se) = ols_treatment(&t, &z, &y);
        assert!(
            (fit.posterior.summaries.mean[eq] - beta).abs() < 3.0 * se / 4000_f64.sqrt() + 2e-3
        );
        let ratio = fit.posterior.summaries.sd[eq] / se;
        assert!((ratio - 1.0).abs() < 0.06, "posterior sd over OLS se {ratio}");
        assert!(fit.whiteness.p_value >= fit.whiteness.level);
        assert!(
            fit.posterior
                .diagnostics
                .notes
                .iter()
                .any(|n| n.starts_with(WHITE_INNOVATION_NOTE_PREFIX))
        );
        assert!(fit.posterior.assumptions.entries.iter().any(|record| matches!(
            &record.assumption,
            Assumption::ParametricRestriction(p) if p.id.as_ref() == WHITE_INNOVATION_ASSUMPTION_ID
        )));
    }

    #[test]
    fn autocorrelated_innovations_are_refused_and_the_tempered_posterior_is_wider() {
        let (t, z, y) = series(400, 0.8, 0.8, 3);
        let prep = prepared(&t, &z, &y);
        let err = fit_white_innovation_posterior(
            &estimator(),
            &prep,
            &declaration(1),
            IdentificationStatus::NonparametricallyIdentified,
            &ExecutionContext::for_tests(1),
        )
        .unwrap_err();
        let text = err.to_string();
        assert!(text.contains("white_innovation.residual_autocorrelation"), "{text}");
        assert!(text.contains("tempered posterior"), "{text}");
        // The iid posterior this refusal withholds is narrower than the tempered one: the
        // likelihood the refusal protects against would understate the spread.
        let ctx = ExecutionContext::for_tests(1);
        let iid = estimator()
            .fit(
                &BayesianGComputationAte::from_prepared_estimation(&prep),
                IdentificationStatus::NonparametricallyIdentified,
                &mut BayesianGCompWorkspace::default(),
                &ctx,
            )
            .expect("iid fit");
        let tempered = estimator()
            .fit(
                &BayesianGComputationAte::from_prepared_estimation(&prep).with_serial_dependence(
                    SerialDependence::LongRunTempering(
                        crate::serial_dependence::DependenceScope::Treatment,
                    ),
                ),
                IdentificationStatus::NonparametricallyIdentified,
                &mut BayesianGCompWorkspace::default(),
                &ctx,
            )
            .expect("tempered fit");
        let eq = iid.effect_column().expect("effect column");
        assert!(tempered.summaries.sd[eq] > 1.2 * iid.summaries.sd[eq]);
    }

    #[test]
    fn a_multi_step_horizon_and_a_bad_gate_are_refused() {
        let (t, z, y) = series(200, 0.0, 0.0, 5);
        let prep = prepared(&t, &z, &y);
        let ctx = ExecutionContext::for_tests(1);
        let status = IdentificationStatus::NonparametricallyIdentified;
        let multi =
            fit_white_innovation_posterior(&estimator(), &prep, &declaration(2), status, &ctx)
                .unwrap_err();
        assert!(multi.to_string().contains("white_innovation.multi_step"), "{multi}");
        let mut bad = declaration(1);
        bad.gate.max_lag = 0;
        let gate =
            fit_white_innovation_posterior(&estimator(), &prep, &bad, status, &ctx).unwrap_err();
        assert!(gate.to_string().contains("white_innovation.invalid_argument"), "{gate}");
        let laplace =
            BayesianGComputationAte::new().with_likelihood(BayesLikelihood::BernoulliLogit);
        let link = fit_white_innovation_posterior(&laplace, &prep, &declaration(1), status, &ctx)
            .unwrap_err();
        assert!(link.to_string().contains("white_innovation.likelihood"), "{link}");
    }

    #[test]
    fn ljung_box_matches_a_hand_computed_statistic() {
        // Residuals 1, -1, 1, -1, 1, -1: sum of squares 6, lag-1 cross-product -5, so
        // r1 = -5/6, and Q = n (n + 2) r1^2 / (n - 1) = 6 * 8 * (25/36) / 5 = 20/3.
        let residuals = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let gate = InnovationWhitenessGate { max_lag: 1, level: 0.05 };
        let report = ljung_box(&residuals, &gate).expect("statistic");
        assert!((report.q_statistic - 20.0 / 3.0).abs() < 1e-12, "{}", report.q_statistic);
        // chi-square(1) survival at 20/3 is erfc(sqrt(10/3)) = 0.009823...
        assert!((report.p_value - 0.009_823).abs() < 1e-5, "{}", report.p_value);
    }
}
