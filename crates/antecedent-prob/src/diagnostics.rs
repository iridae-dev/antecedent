//! Inference diagnostics for Laplace / conjugate / MCMC backends
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::diagnostic::{RouteCheck, RouteKind, RouteMass, RouteNativeDiagnostic};

use crate::error::ProbError;

/// Largest Cholesky condition lower bound at which a Laplace covariance is
/// still published. Beyond ~1/ε the inverse Hessian carries no correct digits.
pub(crate) const MAX_HESSIAN_CONDITION: f64 = 1e14;

/// Factorization used for the Laplace covariance / MCMC marker.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum HessianFactorization {
    /// Cholesky of the negative Hessian.
    Cholesky,
    /// Analytic conjugate (exact posterior; no Hessian).
    Analytic,
    /// Multi-chain MCMC (HMC); curvature from sampling, not Hessian.
    Mcmc,
}

/// Convergence / curvature / chain diagnostics required before reporting a posterior.
#[derive(Clone, Debug, PartialEq)]
pub struct InferenceDiagnostics {
    /// Whether the optimizer / sampler reported convergence.
    pub converged: bool,
    /// Iterations used (Newton steps or post-warmup length).
    pub iterations: u32,
    /// Final gradient infinity-norm (MAP); unused for MCMC (`0.0`).
    pub grad_inf_norm: f64,
    /// Estimated condition number of −Hessian (or NaN if unavailable).
    pub hessian_condition: f64,
    /// Factorization path used.
    pub factorization: HessianFactorization,
    /// Separation / complete-separation warning for Bernoulli models.
    pub separation_warning: bool,
    /// Human-readable notes.
    pub notes: Vec<Arc<str>>,
    /// Backend identifier (e.g. "laplace", "`conjugate_gaussian`", "hmc").
    pub backend_id: Arc<str>,
    /// MCMC: number of chains (None for Laplace / conjugate).
    pub n_chains: Option<u32>,
    /// MCMC: warmup iterations per chain.
    pub n_warmup: Option<u32>,
    /// MCMC: minimum bulk ESS across parameters.
    pub ess_bulk_min: Option<f64>,
    /// MCMC: minimum tail ESS across parameters.
    pub ess_tail_min: Option<f64>,
    /// MCMC: maximum rank∪folded split-Ř across parameters.
    pub rhat_max: Option<f64>,
    /// MCMC: leapfrog / trajectory divergence count (post-warmup; legacy alias).
    pub n_divergences: Option<u32>,
    /// MCMC: mean Metropolis acceptance probability over all transitions.
    pub mean_accept_prob: Option<f64>,
    /// MCMC: divergences during warmup only.
    pub n_warmup_divergences: Option<u32>,
    /// MCMC: divergences after warmup (publication uses this).
    pub n_postwarmup_divergences: Option<u32>,
    /// MCMC: maximum absolute Hamiltonian energy error observed.
    pub max_abs_delta_h: Option<f64>,
    /// MCMC: every chain moved on at least one unconstrained parameter.
    pub all_chains_moved: Option<bool>,
}

impl InferenceDiagnostics {
    /// Laplace (Cholesky) diagnostics. MCMC fields stay unset.
    #[must_use]
    pub fn laplace(
        backend_id: impl Into<Arc<str>>,
        converged: bool,
        iterations: u32,
        grad_inf_norm: f64,
        hessian_condition: f64,
        separation_warning: bool,
        notes: Vec<Arc<str>>,
    ) -> Self {
        Self {
            converged,
            iterations,
            grad_inf_norm,
            hessian_condition,
            factorization: HessianFactorization::Cholesky,
            separation_warning,
            notes,
            backend_id: backend_id.into(),
            n_chains: None,
            n_warmup: None,
            ess_bulk_min: None,
            ess_tail_min: None,
            rhat_max: None,
            n_divergences: None,
            mean_accept_prob: None,
            n_warmup_divergences: None,
            n_postwarmup_divergences: None,
            max_abs_delta_h: None,
            all_chains_moved: None,
        }
    }

    /// Write the post-warmup divergence count and its legacy alias together.
    ///
    /// `n_divergences` is the historical name of `n_postwarmup_divergences`. Publication
    /// reads the post-warmup field; both are set so older readers see the same count.
    pub fn record_postwarmup_divergences(&mut self, n: u32) {
        self.n_postwarmup_divergences = Some(n);
        self.n_divergences = Some(n);
    }

    /// Analytic conjugate path (always "converged").
    #[must_use]
    pub fn analytic(backend_id: impl Into<Arc<str>>) -> Self {
        Self {
            converged: true,
            iterations: 0,
            grad_inf_norm: 0.0,
            hessian_condition: 1.0,
            factorization: HessianFactorization::Analytic,
            separation_warning: false,
            notes: Vec::new(),
            backend_id: backend_id.into(),
            n_chains: None,
            n_warmup: None,
            ess_bulk_min: None,
            ess_tail_min: None,
            rhat_max: None,
            n_divergences: None,
            mean_accept_prob: None,
            n_warmup_divergences: None,
            n_postwarmup_divergences: None,
            max_abs_delta_h: None,
            all_chains_moved: None,
        }
    }

    /// Whether this diagnostic set is sufficient to publish a posterior.
    ///
    /// Narrow Laplace posteriors without convergence + curvature are refused.
    /// Bernoulli Laplace with fitted probabilities at the 0/1 boundary
    /// (`separation_warning`) is refused — the Gaussian Hessian approximation
    /// is not a publication-grade posterior under (quasi-)complete separation.
    /// MCMC requires finite Ř ≤ 1.01, bulk and tail ESS ≥ 100, zero post-warmup
    /// divergences, and movement on every chain. An analytic (conjugate)
    /// posterior requires a finite posterior-precision condition estimate no
    /// larger than [`MAX_HESSIAN_CONDITION`].
    #[must_use]
    pub fn allows_posterior(&self) -> bool {
        match self.factorization {
            // The analytic posterior is exact, but its precision solve is only
            // trustworthy below the same condition ceiling as a Laplace factor.
            HessianFactorization::Analytic => {
                self.hessian_condition > 0.0
                    && self.hessian_condition.is_finite()
                    && self.hessian_condition <= MAX_HESSIAN_CONDITION
            }
            HessianFactorization::Mcmc => self.converged && self.mcmc_publication_ok(),
            HessianFactorization::Cholesky => self.curvature_refusal().is_none(),
        }
    }

    /// Why a Laplace (Cholesky) posterior is refused, or `None` when its
    /// convergence and curvature diagnostics allow publication.
    ///
    /// Names the actual cause so a separation refusal is not reported as
    /// missing diagnostics. The condition ceiling applies to the Cholesky
    /// lower bound on κ(−Hessian), so a factor at working-precision singularity
    /// is refused rather than published with a finite-looking covariance.
    #[must_use]
    pub fn curvature_refusal(&self) -> Option<String> {
        if !self.converged {
            return Some("Laplace posterior refused: the mode search did not converge".into());
        }
        if self.separation_warning {
            return Some(
                "Laplace posterior refused: at least one observation is fitted within 1e-8 of \
                 certainty at the mode (possible separation), so the Gaussian approximation is \
                 not a publication-grade posterior"
                    .into(),
            );
        }
        if !self.grad_inf_norm.is_finite() {
            return Some("Laplace posterior refused: non-finite gradient at the mode".into());
        }
        if !(self.hessian_condition > 0.0 && self.hessian_condition.is_finite()) {
            return Some(
                "Laplace posterior refused: the negative Hessian at the mode has no finite \
                 condition estimate"
                    .into(),
            );
        }
        if self.hessian_condition > MAX_HESSIAN_CONDITION {
            return Some(format!(
                "Laplace posterior refused: negative Hessian condition lower bound {:.3e} exceeds \
                 {MAX_HESSIAN_CONDITION:.0e} (numerically singular curvature)",
                self.hessian_condition
            ));
        }
        None
    }

    /// Full MATH-002 MCMC publication predicate (independent of `converged`).
    #[must_use]
    pub fn mcmc_publication_ok(&self) -> bool {
        self.factorization == HessianFactorization::Mcmc
            && self.mcmc_publication_failures().is_empty()
    }

    /// The clauses of the MCMC publication predicate that fail, in a fixed order; empty exactly
    /// when [`Self::mcmc_publication_ok`] holds for an MCMC factorization. Lets a caller say
    /// *which* criterion blocked publication instead of printing the summary numbers.
    #[must_use]
    pub fn mcmc_publication_failures(&self) -> Vec<&'static str> {
        // Stan/Vehtari (2021): rank-normalized / folded R̂ ≤ 1.01.
        let rhat_ok = self.rhat_max.is_some_and(|r| r.is_finite() && r <= 1.01);
        // Stan/ArviZ guidance is ~100 bulk- and tail-ESS **per chain**, not in total. A flat
        // 100 is reachable with 25 effective draws per chain at the 4-chain default used
        // throughout `antecedent-estimate::bayesian`, which is far too thin for the 2.5% /
        // 97.5% quantiles a credible interval reports. Scaling by `n_chains` also lands on
        // the 400 floor this codebase documents elsewhere for reliable MCMC inference.
        let ess_floor = 100.0 * f64::from(self.n_chains.unwrap_or(1));
        let ess_bulk_ok = self.ess_bulk_min.is_some_and(|e| e.is_finite() && e >= ess_floor);
        let ess_tail_ok = self.ess_tail_min.is_some_and(|e| e.is_finite() && e >= ess_floor);
        let div_ok = self.n_postwarmup_divergences.is_some_and(|n| n == 0);
        let moved_ok = self.all_chains_moved == Some(true);
        let accept_ok = self.mean_accept_prob.is_some_and(f64::is_finite);
        let delta_ok = self.max_abs_delta_h.is_some_and(f64::is_finite);
        let chains_ok = self.n_chains.is_some_and(|c| c >= 2);
        let warmup_ok = self.n_warmup.is_some();
        [
            (rhat_ok, "split-R̂ is missing, non-finite, or above 1.01"),
            (ess_bulk_ok, "bulk ESS is missing, non-finite, or below 100 per chain"),
            (ess_tail_ok, "tail ESS is missing, non-finite, or below 100 per chain"),
            (div_ok, "post-warmup divergences are missing or nonzero"),
            (moved_ok, "not every chain moved (or chain movement was not recorded)"),
            (accept_ok, "mean acceptance probability is missing or non-finite"),
            (delta_ok, "maximum Hamiltonian energy error is missing or non-finite"),
            (chains_ok, "fewer than two chains (R̂ is undefined)"),
            (warmup_ok, "warmup length was not recorded"),
        ]
        .into_iter()
        .filter_map(|(ok, reason)| (!ok).then_some(reason))
        .collect()
    }

    /// Route records for a posterior this diagnostic set will not publish.
    ///
    /// Empty exactly when [`Self::allows_posterior`] is true. An MCMC
    /// `converged` flag is not itself a convergence failure: R-hat, posterior
    /// effective sample size, and chain movement stay separate checks, and a
    /// missing or non-finite sampler measurement is a computational failure.
    #[must_use = "a failed publication check is dropped if the records are discarded"]
    pub fn route_records(&self, unevaluable_mass: RouteMass) -> Vec<RouteNativeDiagnostic> {
        match self.factorization {
            HessianFactorization::Cholesky => self.laplace_route_records(unevaluable_mass),
            HessianFactorization::Mcmc => self.mcmc_route_records(unevaluable_mass),
            HessianFactorization::Analytic => {
                if self.precision_condition_ok() {
                    Vec::new()
                } else {
                    vec![RouteNativeDiagnostic::computational_failure(unevaluable_mass)]
                }
            }
        }
    }

    /// Publication refusal carrying [`Self::route_records`], or `None` when the
    /// posterior may be published.
    ///
    /// `summary` is prose. The records are the outcome.
    #[must_use = "a publication refusal is dropped if the error is discarded"]
    pub fn publication_error(&self, summary: impl Into<String>) -> Option<ProbError> {
        let diagnostics = self.route_records(RouteMass::unreported());
        if diagnostics.is_empty() {
            None
        } else {
            Some(ProbError::Route { summary: summary.into(), diagnostics })
        }
    }

    fn precision_condition_ok(&self) -> bool {
        self.hessian_condition > 0.0
            && self.hessian_condition.is_finite()
            && self.hessian_condition <= MAX_HESSIAN_CONDITION
    }

    fn laplace_route_records(&self, mass: RouteMass) -> Vec<RouteNativeDiagnostic> {
        let mut records = Vec::new();
        let failed = RouteCheck::new(true, mass);
        if !self.converged {
            records.push(RouteNativeDiagnostic::check(RouteKind::Convergence, failed));
        }
        if self.separation_warning {
            records.push(RouteNativeDiagnostic::model_restriction(mass));
        }
        if !self.grad_inf_norm.is_finite() || !self.precision_condition_ok() {
            records.push(RouteNativeDiagnostic::computational_failure(mass));
        }
        records
    }

    fn mcmc_route_records(&self, mass: RouteMass) -> Vec<RouteNativeDiagnostic> {
        let mut records = Vec::new();
        let failed = RouteCheck::new(true, mass);
        let rhat_ok = self.rhat_max.is_some_and(|r| r.is_finite() && r <= 1.01);
        if !rhat_ok {
            records.push(RouteNativeDiagnostic::check(RouteKind::RHat, failed));
        }
        let ess_floor = 100.0 * f64::from(self.n_chains.unwrap_or(1));
        let ess_bulk_ok = self.ess_bulk_min.is_some_and(|ess| ess.is_finite() && ess >= ess_floor);
        let ess_tail_ok = self.ess_tail_min.is_some_and(|ess| ess.is_finite() && ess >= ess_floor);
        if !ess_bulk_ok || !ess_tail_ok {
            records.push(RouteNativeDiagnostic::check(RouteKind::PosteriorEss, failed));
        }
        if self.all_chains_moved == Some(false) {
            records.push(RouteNativeDiagnostic::check(RouteKind::Convergence, failed));
        }
        let measured = self.n_postwarmup_divergences.is_some_and(|n| n == 0)
            && self.all_chains_moved.is_some()
            && self.mean_accept_prob.is_some_and(f64::is_finite)
            && self.max_abs_delta_h.is_some_and(f64::is_finite)
            && self.n_chains.is_some_and(|chains| chains >= 2)
            && self.n_warmup.is_some();
        if !measured {
            records.push(RouteNativeDiagnostic::computational_failure(mass));
        }
        if records.is_empty() && !self.converged {
            records.push(RouteNativeDiagnostic::computational_failure(mass));
        }
        records
    }
}

/// Which prior family a prior-sensitivity grid perturbed.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum PriorSensitivityFamily {
    /// Isotropic `N(0, scale²)` coefficient priors over `prior_scales`. Used only
    /// when no prior was supplied, so the isotropic prior is the prior in force.
    #[default]
    IsotropicScale,
    /// External power-prior α multipliers over `alphas` (prior bank compose).
    ExternalAlpha,
    /// The resolved prior in force (explicit / transferred) with its coefficient
    /// variances multiplied by `variance_multipliers`; means kept.
    ResolvedPriorVariance,
}

impl PriorSensitivityFamily {
    /// Stable `snake_case` tag.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::IsotropicScale => "isotropic_scale",
            Self::ExternalAlpha => "external_alpha",
            Self::ResolvedPriorVariance => "resolved_prior_variance",
        }
    }
}

/// Optional prior-sensitivity summary attached to a causal posterior.
///
/// Mode-select by [`Self::family`]: isotropic scale grid (`prior_scales`), external
/// power-prior α-multiplier grid (`alphas`), or a variance-multiplier grid around the
/// resolved prior in force (`variance_multipliers`). Exactly the family's grid is
/// non-empty; `effect_means` / `effect_sds` always align with it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PriorSensitivitySummary {
    /// Which prior family was perturbed.
    pub family: PriorSensitivityFamily,
    /// Prior scale grid evaluated (isotropic mode).
    pub prior_scales: Arc<[f64]>,
    /// External α multipliers applied to post-conflict `alphas_applied` (bank mode).
    pub alphas: Arc<[f64]>,
    /// Multipliers on the resolved prior's coefficient variances (resolved-prior mode).
    pub variance_multipliers: Arc<[f64]>,
    /// Posterior mean of the primary effect at each grid point.
    pub effect_means: Arc<[f64]>,
    /// Posterior SD of the primary effect at each grid point.
    pub effect_sds: Arc<[f64]>,
}

/// External-prior conflict shrink summary (attached beside prior sensitivity).
///
/// Records requested vs applied power-prior alphas after conflict policy shrink
/// (orchestration lives in `antecedent-validate`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ConflictSummary {
    /// Source artifact / catalog ids.
    pub source_ids: Arc<[Arc<str>]>,
    /// Caller-requested alphas.
    pub alphas_requested: Arc<[f64]>,
    /// Alphas after conflict shrink (`≤` requested).
    pub alphas_applied: Arc<[f64]>,
    /// Prior-PPC p-values used per source (`None` if unused).
    pub p_values: Arc<[Option<f64>]>,
    /// Gaussian KL (nats) used per source (`None` if unused).
    pub kl_values: Arc<[Option<f64>]>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mcmc_ok_base() -> InferenceDiagnostics {
        InferenceDiagnostics {
            converged: true,
            iterations: 400,
            grad_inf_norm: 0.0,
            hessian_condition: f64::NAN,
            factorization: HessianFactorization::Mcmc,
            separation_warning: false,
            notes: Vec::new(),
            backend_id: Arc::from("hmc"),
            n_chains: Some(4),
            n_warmup: Some(100),
            // Healthy baseline for the 4 chains below: the publication gate requires ~100
            // bulk/tail ESS *per chain*, so a genuinely well-mixed 4-chain run clears 400.
            ess_bulk_min: Some(500.0),
            ess_tail_min: Some(450.0),
            rhat_max: Some(1.005),
            n_divergences: Some(0),
            mean_accept_prob: Some(0.8),
            n_warmup_divergences: Some(0),
            n_postwarmup_divergences: Some(0),
            max_abs_delta_h: Some(0.1),
            all_chains_moved: Some(true),
        }
    }

    #[test]
    fn laplace_requires_convergence() {
        let mut d =
            InferenceDiagnostics::laplace("laplace", false, 10, 1.0, 10.0, false, Vec::new());
        assert!(!d.allows_posterior());
        d.converged = true;
        assert!(d.allows_posterior());
        d.separation_warning = true;
        assert!(!d.allows_posterior());
    }

    #[test]
    fn laplace_refusal_names_its_cause() {
        let mut d = mcmc_ok_base();
        d.factorization = HessianFactorization::Cholesky;
        d.hessian_condition = 10.0;
        assert!(d.curvature_refusal().is_none());
        d.separation_warning = true;
        assert!(d.curvature_refusal().unwrap().contains("separation"));
        d.separation_warning = false;
        d.hessian_condition = f64::NAN;
        assert!(d.curvature_refusal().unwrap().contains("condition"));
        assert!(!d.allows_posterior());
        d.hessian_condition = 1e15;
        assert!(d.curvature_refusal().unwrap().contains("exceeds"));
        assert!(!d.allows_posterior());
        d.hessian_condition = 1e13;
        assert!(d.allows_posterior());
        d.converged = false;
        assert!(d.curvature_refusal().unwrap().contains("converge"));
    }

    #[test]
    fn publication_failures_name_exactly_the_failing_clauses() {
        let mut d = mcmc_ok_base();
        assert!(d.mcmc_publication_failures().is_empty());
        assert!(d.mcmc_publication_ok());
        // Tail ESS alone and chain movement alone: the reason list names those, and only those.
        d.ess_tail_min = Some(50.0);
        d.all_chains_moved = Some(false);
        let failures = d.mcmc_publication_failures();
        assert_eq!(failures.len(), 2, "{failures:?}");
        assert!(failures[0].contains("tail ESS"), "{failures:?}");
        assert!(failures[1].contains("chain moved"), "{failures:?}");
        assert!(!d.mcmc_publication_ok());
        // Every clause missing: all nine are listed.
        d.rhat_max = None;
        d.ess_bulk_min = None;
        d.n_postwarmup_divergences = None;
        d.mean_accept_prob = None;
        d.max_abs_delta_h = None;
        d.n_chains = Some(1);
        d.n_warmup = None;
        assert_eq!(d.mcmc_publication_failures().len(), 9);
    }

    #[test]
    fn mcmc_requires_full_gate() {
        let mut d = mcmc_ok_base();
        assert!(d.allows_posterior());
        d.ess_bulk_min = Some(50.0);
        assert!(!d.allows_posterior());
        d.ess_bulk_min = Some(500.0);
        d.n_postwarmup_divergences = Some(1);
        assert!(!d.allows_posterior());
        d.n_postwarmup_divergences = Some(0);
        d.all_chains_moved = Some(false);
        assert!(!d.allows_posterior());
        d.all_chains_moved = Some(true);
        d.rhat_max = Some(1.02);
        assert!(!d.allows_posterior());
        d.rhat_max = Some(1.01);
        assert!(d.allows_posterior());
    }

    /// The ESS floor must scale with chain count, not sit at a flat total.
    ///
    /// Stan/ArviZ state the ~100 bulk/tail-ESS guidance **per chain**. A flat 100 total is
    /// satisfied by 25 effective draws per chain at the 4-chain default, which is far too
    /// thin for the 2.5%/97.5% quantiles a credible interval reports. The same total ESS
    /// therefore has to pass with few chains and fail with many.
    #[test]
    fn ess_gate_scales_with_chain_count() {
        let with_chains = |n_chains: u32, ess: f64| {
            let mut d = mcmc_ok_base();
            d.n_chains = Some(n_chains);
            d.ess_bulk_min = Some(ess);
            d.ess_tail_min = Some(ess);
            d.allows_posterior()
        };

        // 250 total ESS: fine across 2 chains (125 each), too thin across 4 (62.5 each).
        assert!(with_chains(2, 250.0), "250 ESS over 2 chains is 125/chain and should pass");
        assert!(!with_chains(4, 250.0), "250 ESS over 4 chains is 62.5/chain and must not pass");

        // The 4-chain case passes once it genuinely clears 100/chain.
        assert!(!with_chains(4, 399.0));
        assert!(with_chains(4, 401.0));

        // A flat-100 gate would have accepted every case above; pin that it does not.
        assert!(!with_chains(4, 100.0), "a flat 100-total floor must not be what gates this");
    }

    #[test]
    fn mcmc_divergence_presence_is_not_enough() {
        let mut d = mcmc_ok_base();
        d.n_postwarmup_divergences = Some(3);
        d.n_divergences = Some(3);
        assert!(!d.allows_posterior());
    }

    fn is_failed(record: &RouteNativeDiagnostic, name: &str) -> bool {
        record.failed() && record.route_name() == name
    }

    #[test]
    fn mcmc_publication_failures_stay_distinct_checks() {
        let mass = RouteMass::zero();
        let mut rhat = mcmc_ok_base();
        rhat.rhat_max = Some(1.2);
        let records = rhat.route_records(mass);
        assert!(records.iter().any(|record| is_failed(record, "r_hat")));
        assert!(records.iter().all(|record| record.route_name() != "convergence"));
        assert!(records.iter().all(|record| !record.is_model_restriction()));
        assert!(records.iter().all(|record| !record.is_unsupported_inference()));

        let mut ess = mcmc_ok_base();
        ess.ess_bulk_min = Some(10.0);
        let records = ess.route_records(mass);
        assert!(records.iter().any(|record| is_failed(record, "posterior_ess")));
        assert!(records.iter().all(|record| record.route_name() != "r_hat"));
        assert!(records.iter().all(|record| record.route_name() != "convergence"));

        let mut movement = mcmc_ok_base();
        movement.all_chains_moved = Some(false);
        let records = movement.route_records(mass);
        assert!(records.iter().any(|record| is_failed(record, "convergence")));
        assert!(records.iter().all(|record| record.route_name() != "r_hat"));
        assert!(records.iter().all(|record| record.route_name() != "posterior_ess"));

        let mut diverged = mcmc_ok_base();
        diverged.n_postwarmup_divergences = Some(2);
        let records = diverged.route_records(mass);
        assert!(records.iter().any(|record| record.is_computational_failure()));
        assert!(records.iter().all(|record| !record.is_unsupported_inference()));
        assert!(records.iter().all(|record| record.route_name() != "convergence"));
    }

    #[test]
    fn laplace_separation_is_a_model_restriction_and_a_singular_hessian_is_not() {
        let mass = RouteMass::zero();
        let separated =
            InferenceDiagnostics::laplace("laplace", true, 4, 1e-6, 10.0, true, Vec::new());
        let records = separated.route_records(mass);
        assert!(records.iter().any(|record| record.is_model_restriction()));
        assert!(records.iter().all(|record| !record.is_computational_failure()));
        assert!(records.iter().all(|record| !record.is_unsupported_inference()));
        assert!(records.iter().all(|record| record.route_name() != "convergence"));

        let singular =
            InferenceDiagnostics::laplace("laplace", true, 4, 1e-6, 1e15, false, Vec::new());
        let records = singular.route_records(mass);
        assert!(records.iter().any(|record| record.is_computational_failure()));
        assert!(records.iter().all(|record| !record.is_model_restriction()));
    }
}
