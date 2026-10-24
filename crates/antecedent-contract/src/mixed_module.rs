//! Mixed-module policy for a release workflow.
//!
//! Shared probabilistic modules are one coherent joint model, or a modular
//! alternative that records how shared parameters and priors were reconciled
//! and states a [`CutFeedback`] policy. [`CutFeedback::Cut`] and
//! [`CutFeedback::Feedback`] are both named; the modular constructor takes one
//! of them as an argument.
//!
//! The policy is also executable on a linear-Gaussian pair of modules that
//! share one parameter, where every posterior is exact:
//!
//! * the first module observes the shared parameter `theta`, `y1 = theta + e1`;
//! * the second observes it through a module-specific parameter `phi`,
//!   `y2 = theta + phi + e2`, with `phi ~ N(m_phi, v_phi)`;
//! * each module carries its own prior for `theta`. [`reconcile_priors`] pools
//!   them by a weighted product of densities (the log-linear pool of Markov
//!   melding), so the shared prior is counted once, not once per module;
//! * [`SharedParameterModel::posterior`] fits the semi-modular family
//!   `p(theta | y1, y2^eta) p(phi | theta, y2)`. `eta = 1` is the coherent
//!   joint posterior, in which the second module feeds back into `theta`;
//!   `eta = 0` is the cut posterior, in which `theta` is learned from the first
//!   module alone and the second module only conditions `phi` on that `theta`.
//!   The cut keeps the dependence between `theta` and `phi` that flows forward;
//!   it removes only what flows back;
//! * [`MixedModulePolicy::posterior`] picks `eta` from the declared policy;
//! * [`SharedParameterModel::pooling_envelope`] is a robust envelope over a
//!   range of pooling weights, distinct from both the melded and the cut fit.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::{error::Error, fmt, sync::Arc};

use antecedent_core::named_variants;
use antecedent_core::text::is_blank;

/// How a release workflow treats shared probabilistic modules.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
#[must_use = "a mixed-module policy records the release workflow's modeling choice"]
pub enum MixedModulePolicy {
    /// One coherent joint model.
    ///
    /// Shared parameters and priors belong to that model. No cut or feedback
    /// policy is stored.
    CoherentJoint,
    /// A modular alternative with a reconciliation record and a named cut or
    /// feedback policy.
    DeclaredModular {
        /// Non-blank shared-parameter and prior reconciliation.
        reconciliation: ReconciliationRecord,
        /// The cut or feedback policy the caller named.
        cut_feedback: CutFeedback,
    },
}

impl MixedModulePolicy {
    /// Coherent joint modeling, with no cut stored.
    pub const fn coherent_joint() -> Self {
        Self::CoherentJoint
    }

    /// A modular policy that keeps `reconciliation` and `cut_feedback`.
    ///
    /// # Errors
    ///
    /// [`MixedModuleRefusal::BlankReconciliation`] when `reconciliation` is
    /// empty or only whitespace. That result contains the refusal alone.
    pub fn declared_modular(
        reconciliation: &str,
        cut_feedback: CutFeedback,
    ) -> Result<Self, MixedModuleRefusal> {
        Ok(Self::DeclaredModular {
            reconciliation: ReconciliationRecord::try_new(reconciliation)?,
            cut_feedback,
        })
    }

    /// Cut or feedback stored on a modular policy.
    ///
    /// Coherent joint modeling returns [`None`].
    #[must_use]
    pub const fn cut_feedback(&self) -> Option<CutFeedback> {
        match self {
            Self::CoherentJoint => None,
            Self::DeclaredModular { cut_feedback, .. } => Some(*cut_feedback),
        }
    }

    /// Shared-parameter and prior reconciliation stored on a modular policy.
    ///
    /// Coherent joint modeling returns [`None`].
    #[must_use]
    pub const fn reconciliation(&self) -> Option<&ReconciliationRecord> {
        match self {
            Self::CoherentJoint => None,
            Self::DeclaredModular { reconciliation, .. } => Some(reconciliation),
        }
    }
}

/// Cut or feedback for a declared modular alternative.
///
/// Both alternatives are explicit arguments. [`MixedModulePolicy::coherent_joint`]
/// stores neither.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum CutFeedback {
    /// The modules are cut: the caller named the absence of feedback.
    Cut,
    /// The modules feed back; the reconciliation record states how.
    Feedback,
}

named_variants! {
    CutFeedback {
        Cut => "cut",
        Feedback => "feedback",
    }
}

impl fmt::Display for CutFeedback {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Caller's account of how shared parameters and their priors were reconciled.
///
/// The text is the record. It is produced only from non-blank text.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct ReconciliationRecord {
    text: Arc<str>,
}

impl ReconciliationRecord {
    /// Keep `text` as the reconciliation record.
    ///
    /// # Errors
    ///
    /// [`MixedModuleRefusal::BlankReconciliation`] when `text` is empty or
    /// only whitespace.
    pub fn try_new(text: &str) -> Result<Self, MixedModuleRefusal> {
        if is_blank(text) || is_invisible(text) {
            Err(MixedModuleRefusal::BlankReconciliation)
        } else {
            Ok(Self { text: Arc::from(text) })
        }
    }

    /// The reconciliation text the caller supplied.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.text
    }
}

/// Text made only of whitespace, control, or zero-width format characters.
///
/// `str::trim` treats Unicode white space as blank but not a zero-width space or
/// joiner, which render as nothing. A record of those characters says nothing.
fn is_invisible(text: &str) -> bool {
    text.chars().all(|c| {
        c.is_whitespace()
            || c.is_control()
            || matches!(c, '\u{200b}'..='\u{200f}' | '\u{2060}' | '\u{feff}')
    })
}

impl fmt::Display for ReconciliationRecord {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Refusal to build a modular policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
#[must_use = "a refusal is the failed modular construction"]
pub enum MixedModuleRefusal {
    /// The modular declaration had a blank reconciliation record.
    BlankReconciliation,
    /// A shared-parameter model or an envelope request was not well formed;
    /// the string names what was wrong.
    InvalidModel(&'static str),
}

impl MixedModuleRefusal {
    /// Registered reason code.
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self {
            Self::BlankReconciliation | Self::InvalidModel(_) => {
                antecedent_core::reason_code!("invalid_argument")
            }
        }
    }

    /// Stable `mixed_module.*` detail.
    #[must_use]
    pub const fn detail(self) -> &'static str {
        match self {
            Self::BlankReconciliation => "mixed_module.blank_reconciliation",
            Self::InvalidModel(_) => "mixed_module.invalid_model",
        }
    }
}

impl fmt::Display for MixedModuleRefusal {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::BlankReconciliation => {
                "a declared modular policy needs a non-blank shared-parameter and prior reconciliation record"
            }
            Self::InvalidModel(reason) => *reason,
        };
        write!(
            formatter,
            "{}{}: {}: {message}",
            antecedent_core::reason_code::PREFIX,
            self.code(),
            self.detail(),
        )
    }
}

impl Error for MixedModuleRefusal {}

impl MixedModulePolicy {
    /// The posterior this policy prescribes for `model`.
    ///
    /// Coherent joint modeling is the full joint posterior (`eta = 1`). A
    /// declared cut removes the second module's influence on the shared
    /// parameter (`eta = 0`). A declared feedback lets it through with the
    /// model's feedback weight.
    ///
    /// # Errors
    /// [`MixedModuleRefusal::InvalidModel`] when `model` is not well formed or
    /// the policy names a cut or feedback this module does not implement.
    pub fn posterior(
        &self,
        model: &SharedParameterModel,
    ) -> Result<ModularPosterior, MixedModuleRefusal> {
        let eta = match self {
            Self::CoherentJoint => 1.0,
            Self::DeclaredModular { cut_feedback, .. } => match cut_feedback {
                CutFeedback::Cut => 0.0,
                CutFeedback::Feedback => model.feedback_weight,
            },
        };
        model.posterior(eta)
    }
}

/// A Gaussian density, stated by its mean and variance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GaussianLaw {
    /// Mean.
    pub mean: f64,
    /// Variance, finite and positive.
    pub variance: f64,
}

impl GaussianLaw {
    fn check(&self, what: &'static str) -> Result<(), MixedModuleRefusal> {
        if self.mean.is_finite() && self.variance.is_finite() && self.variance > 0.0 {
            Ok(())
        } else {
            Err(MixedModuleRefusal::InvalidModel(what))
        }
    }
}

/// Log-linear (weighted product) pool of two Gaussian priors for one shared
/// parameter: density proportional to `p1^w p2^(1-w)`.
///
/// # Errors
/// [`MixedModuleRefusal::InvalidModel`] for a prior that is not a finite mean
/// with a positive finite variance, or a weight outside `[0, 1]`.
pub fn reconcile_priors(
    first: GaussianLaw,
    second: GaussianLaw,
    first_weight: f64,
) -> Result<GaussianLaw, MixedModuleRefusal> {
    first.check("the first module's prior is a finite mean with a positive variance")?;
    second.check("the second module's prior is a finite mean with a positive variance")?;
    if !(0.0..=1.0).contains(&first_weight) {
        return Err(MixedModuleRefusal::InvalidModel("the pooling weight lies in [0, 1]"));
    }
    let precision = first_weight / first.variance + (1.0 - first_weight) / second.variance;
    let mean = (first_weight * first.mean / first.variance
        + (1.0 - first_weight) * second.mean / second.variance)
        / precision;
    Ok(GaussianLaw { mean, variance: 1.0 / precision })
}

/// Two modules that share the parameter `theta`.
///
/// The first observes `y1 = theta + e1`, `e1 ~ N(0, margin_observation.variance)`.
/// The second observes `y2 = theta + phi + e2`, with module-specific
/// `phi ~ N(specific_prior)` and `e2 ~ N(0, cost_observation.variance)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SharedParameterModel {
    /// The first module's prior for `theta`.
    pub margin_prior: GaussianLaw,
    /// The second module's own prior for `theta`.
    pub cost_prior: GaussianLaw,
    /// Weight of the first prior in the log-linear pool, in `[0, 1]`.
    pub pooling_weight: f64,
    /// Observation `y1` and its error variance.
    pub margin_observation: GaussianLaw,
    /// Observation `y2` and its error variance.
    pub cost_observation: GaussianLaw,
    /// Prior of the second module's specific parameter `phi`.
    pub specific_prior: GaussianLaw,
    /// `eta` a declared feedback policy uses, in `(0, 1]`.
    pub feedback_weight: f64,
}

/// Exact Gaussian posterior of `(theta, phi)` and of the contrast `theta + phi`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ModularPosterior {
    /// Strength with which the second module informs `theta`.
    pub eta: f64,
    /// The reconciled prior of `theta`.
    pub pooled_prior: GaussianLaw,
    /// Posterior mean of `theta`.
    pub theta_mean: f64,
    /// Posterior variance of `theta`.
    pub theta_variance: f64,
    /// Posterior mean of `phi`.
    pub phi_mean: f64,
    /// Posterior variance of `phi`.
    pub phi_variance: f64,
    /// Posterior covariance of `theta` and `phi`.
    pub theta_phi_covariance: f64,
    /// Conflict of the two module priors: `(m1 - m2)^2 / (v1 + v2)`.
    pub prior_conflict: f64,
}

impl ModularPosterior {
    /// Posterior mean of the second module's contrast `theta + phi`.
    #[must_use]
    pub fn contrast_mean(&self) -> f64 {
        self.theta_mean + self.phi_mean
    }

    /// Posterior variance of the contrast `theta + phi`.
    #[must_use]
    pub fn contrast_variance(&self) -> f64 {
        self.theta_variance + self.phi_variance + 2.0 * self.theta_phi_covariance
    }

    /// Posterior covariance of `theta` and the contrast `theta + phi`.
    #[must_use]
    pub fn theta_contrast_covariance(&self) -> f64 {
        self.theta_variance + self.theta_phi_covariance
    }

    /// Posterior correlation of `theta` and the contrast.
    #[must_use]
    pub fn theta_contrast_correlation(&self) -> f64 {
        self.theta_contrast_covariance()
            / (self.theta_variance.sqrt() * self.contrast_variance().sqrt())
    }
}

/// Range of one posterior summary over a family of pooling weights.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SummaryRange {
    /// Smallest value over the family.
    pub low: f64,
    /// Largest value over the family.
    pub high: f64,
}

/// Robust envelope of a modular posterior over a range of pooling weights.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PoolingEnvelope {
    /// Range of the posterior mean of `theta`.
    pub theta_mean: SummaryRange,
    /// Range of the posterior variance of `theta`.
    pub theta_variance: SummaryRange,
    /// Range of the posterior mean of the contrast.
    pub contrast_mean: SummaryRange,
    /// Number of pooling weights evaluated.
    pub weights: usize,
}

impl SharedParameterModel {
    fn check(&self) -> Result<(), MixedModuleRefusal> {
        self.margin_prior
            .check("the first module's prior is a finite mean with a positive variance")?;
        self.cost_prior
            .check("the second module's prior is a finite mean with a positive variance")?;
        self.margin_observation
            .check("the first observation is a finite value with a positive error variance")?;
        self.cost_observation
            .check("the second observation is a finite value with a positive error variance")?;
        self.specific_prior
            .check("the specific parameter's prior is a finite mean with a positive variance")?;
        if !(0.0..=1.0).contains(&self.pooling_weight) {
            return Err(MixedModuleRefusal::InvalidModel("the pooling weight lies in [0, 1]"));
        }
        if !(self.feedback_weight > 0.0 && self.feedback_weight <= 1.0) {
            return Err(MixedModuleRefusal::InvalidModel("the feedback weight lies in (0, 1]"));
        }
        Ok(())
    }

    /// Posterior `p(theta | y1, y2^eta) p(phi | theta, y2)` for `eta` in `[0, 1]`.
    ///
    /// The second observation reaches `theta` through its marginal over `phi`,
    /// `y2 | theta ~ N(theta + m_phi, v_e2 + v_phi)`, raised to the power
    /// `eta`. It conditions `phi` given `theta` at full strength whatever `eta`
    /// is, so `eta = 0` is the cut and `eta = 1` the coherent joint posterior.
    ///
    /// # Errors
    /// [`MixedModuleRefusal::InvalidModel`] for a model that is not well
    /// formed or an `eta` outside `[0, 1]`.
    pub fn posterior(&self, eta: f64) -> Result<ModularPosterior, MixedModuleRefusal> {
        self.check()?;
        if !(0.0..=1.0).contains(&eta) {
            return Err(MixedModuleRefusal::InvalidModel("the feedback strength lies in [0, 1]"));
        }
        let pooled_prior =
            reconcile_priors(self.margin_prior, self.cost_prior, self.pooling_weight)?;
        let s1 = self.margin_observation.variance;
        let s2 = self.cost_observation.variance;
        let phi = self.specific_prior;
        let marginal_variance = s2 + phi.variance;
        let precision = 1.0 / pooled_prior.variance + 1.0 / s1 + eta / marginal_variance;
        let theta_mean = (pooled_prior.mean / pooled_prior.variance
            + self.margin_observation.mean / s1
            + eta * (self.cost_observation.mean - phi.mean) / marginal_variance)
            / precision;
        let theta_variance = 1.0 / precision;
        // phi | theta, y2 is N(a - k theta, 1 / q).
        let q = 1.0 / phi.variance + 1.0 / s2;
        let a = (phi.mean / phi.variance + self.cost_observation.mean / s2) / q;
        let k = (1.0 / s2) / q;
        Ok(ModularPosterior {
            eta,
            pooled_prior,
            theta_mean,
            theta_variance,
            phi_mean: a - k * theta_mean,
            phi_variance: k * k * theta_variance + 1.0 / q,
            theta_phi_covariance: -k * theta_variance,
            prior_conflict: (self.margin_prior.mean - self.cost_prior.mean).powi(2)
                / (self.margin_prior.variance + self.cost_prior.variance),
        })
    }

    /// Envelope of the posterior at strength `eta` over `steps + 1` equally
    /// spaced pooling weights from `low` to `high`.
    ///
    /// # Errors
    /// [`MixedModuleRefusal::InvalidModel`] for a range outside `[0, 1]`, a
    /// reversed range, no steps, or an invalid model.
    pub fn pooling_envelope(
        &self,
        eta: f64,
        low: f64,
        high: f64,
        steps: usize,
    ) -> Result<PoolingEnvelope, MixedModuleRefusal> {
        if !(0.0 <= low && low <= high && high <= 1.0) || steps == 0 {
            return Err(MixedModuleRefusal::InvalidModel(
                "the pooling envelope is an ordered range inside [0, 1] with at least one step",
            ));
        }
        let mut envelope: Option<PoolingEnvelope> = None;
        for index in 0..=steps {
            let weight = low + (high - low) * index as f64 / steps as f64;
            let posterior = Self { pooling_weight: weight, ..*self }.posterior(eta)?;
            let point = (posterior.theta_mean, posterior.theta_variance, posterior.contrast_mean());
            envelope = Some(match envelope {
                None => PoolingEnvelope {
                    theta_mean: SummaryRange { low: point.0, high: point.0 },
                    theta_variance: SummaryRange { low: point.1, high: point.1 },
                    contrast_mean: SummaryRange { low: point.2, high: point.2 },
                    weights: 1,
                },
                Some(mut current) => {
                    current.theta_mean = widen(current.theta_mean, point.0);
                    current.theta_variance = widen(current.theta_variance, point.1);
                    current.contrast_mean = widen(current.contrast_mean, point.2);
                    current.weights += 1;
                    current
                }
            });
        }
        envelope.ok_or(MixedModuleRefusal::InvalidModel("the pooling envelope has no weights"))
    }
}

fn widen(range: SummaryRange, value: f64) -> SummaryRange {
    SummaryRange { low: range.low.min(value), high: range.high.max(value) }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::{CutFeedback, MixedModulePolicy, MixedModuleRefusal, ReconciliationRecord};

    #[test]
    fn coherent_joint_stores_no_cut_or_reconciliation() {
        let policy = MixedModulePolicy::coherent_joint();
        assert_eq!(policy.cut_feedback(), None);
        assert!(policy.reconciliation().is_none());
    }

    #[test]
    fn text_that_renders_as_nothing_is_not_a_reconciliation() {
        let blanks = [
            "",
            " ",
            "\t\n",
            "\u{a0}",
            "\u{2003}\u{3000}",
            "\u{200b}",
            "\u{200d}\u{feff}",
            " \u{2060} ",
            "\u{0}",
        ];
        for blank in blanks {
            for policy in CutFeedback::ALL {
                assert_eq!(
                    MixedModulePolicy::declared_modular(blank, policy),
                    Err(MixedModuleRefusal::BlankReconciliation),
                    "{blank:?}"
                );
            }
            assert_eq!(
                ReconciliationRecord::try_new(blank),
                Err(MixedModuleRefusal::BlankReconciliation)
            );
        }
        // One visible character is enough, wherever it sits among invisible ones.
        for record in ["x", "\u{200b}x", " x ", "\u{a0}\u{200d}é\u{feff}"] {
            assert!(ReconciliationRecord::try_new(record).is_ok(), "{record:?}");
        }
    }

    #[test]
    fn the_record_keeps_the_callers_text_without_trimming() {
        let record = "  shared theta;\n prior product renormalized on the overlap \t";
        let policy = MixedModulePolicy::declared_modular(record, CutFeedback::Cut).unwrap();
        let kept = policy.reconciliation().unwrap();
        assert_eq!(kept.as_str(), record);
        assert_eq!(kept.to_string(), record);
        // Reconciliations that differ only in padding are different records.
        let trimmed = MixedModulePolicy::declared_modular(record.trim(), CutFeedback::Cut).unwrap();
        assert_ne!(policy, trimmed);
    }

    #[test]
    fn policies_differ_by_cut_feedback_and_by_record_and_hash_accordingly() {
        let record = "shared theta; prior product renormalized on the overlap";
        let cut = MixedModulePolicy::declared_modular(record, CutFeedback::Cut).unwrap();
        let feedback = MixedModulePolicy::declared_modular(record, CutFeedback::Feedback).unwrap();
        let other_record =
            MixedModulePolicy::declared_modular("pooled theta", CutFeedback::Cut).unwrap();
        let same_as_cut = MixedModulePolicy::declared_modular(record, CutFeedback::Cut).unwrap();

        assert_eq!(cut, same_as_cut);
        let distinct = [&cut, &feedback, &other_record, &MixedModulePolicy::CoherentJoint];
        let set: HashSet<&MixedModulePolicy> = distinct.into_iter().collect();
        assert_eq!(set.len(), 4);
        assert!(set.contains(&same_as_cut));
        assert_eq!(cut.cut_feedback(), Some(CutFeedback::Cut));
        assert_eq!(feedback.cut_feedback(), Some(CutFeedback::Feedback));
    }

    #[test]
    fn cut_feedback_names_parse_exactly() {
        for policy in CutFeedback::ALL {
            assert_eq!(CutFeedback::from_name(policy.as_str()), Some(policy));
            assert_eq!(policy.to_string(), policy.as_str());
            // Neither case variants nor padding are the same spelling.
            assert_eq!(CutFeedback::from_name(&policy.as_str().to_uppercase()), None);
            assert_eq!(CutFeedback::from_name(&format!(" {} ", policy.as_str())), None);
        }
        assert_ne!(CutFeedback::Cut.as_str(), CutFeedback::Feedback.as_str());
        assert_eq!(CutFeedback::from_name(""), None);
        assert_eq!(CutFeedback::from_name("cut_feedback"), None);
    }

    #[test]
    fn a_refusal_message_carries_the_registered_code_and_detail_it_reports() {
        let refusal = MixedModulePolicy::declared_modular(" ", CutFeedback::Feedback).unwrap_err();
        let message = refusal.to_string();
        assert!(message.starts_with(antecedent_core::reason_code::PREFIX), "{message}");
        assert!(message.contains(refusal.code()), "{message}");
        assert!(message.contains(refusal.detail()), "{message}");
        assert!(message.contains("reconciliation"), "{message}");
        assert_eq!(refusal.code(), antecedent_core::reason_code!("invalid_argument"));
        assert!(refusal.detail().starts_with("mixed_module."));
        // The refusal is an error value usable through the standard trait object.
        let boxed: Box<dyn std::error::Error> = Box::new(refusal);
        assert_eq!(boxed.to_string(), message);
    }

    use super::{GaussianLaw, SharedParameterModel, reconcile_priors};

    fn law(mean: f64, variance: f64) -> GaussianLaw {
        GaussianLaw { mean, variance }
    }

    fn model() -> SharedParameterModel {
        SharedParameterModel {
            margin_prior: law(0.5, 2.0),
            cost_prior: law(-0.4, 1.0),
            pooling_weight: 0.4,
            margin_observation: law(1.3, 0.25),
            cost_observation: law(2.1, 0.5),
            specific_prior: law(0.3, 1.5),
            feedback_weight: 1.0,
        }
    }

    fn normal_pdf(x: f64, mean: f64, variance: f64) -> f64 {
        (-(x - mean).powi(2) / (2.0 * variance)).exp() / (2.0 * std::f64::consts::PI * variance).sqrt()
    }

    /// Moments of `(theta, phi)` by quadrature over the defining densities:
    /// `p(theta) = pool * L1 * L2^eta` with `L2` the marginal likelihood of
    /// `y2` given `theta`, and `p(phi | theta)` normalized numerically from
    /// prior times likelihood. Nothing here uses the closed forms.
    fn quadrature(model: &SharedParameterModel, eta: f64) -> [f64; 5] {
        let pooled = reconcile_priors(model.margin_prior, model.cost_prior, model.pooling_weight)
            .unwrap();
        let (theta_lo, theta_hi, theta_n) = (-9.0, 9.0, 721);
        let (phi_lo, phi_hi, phi_n) = (-14.0, 14.0, 1121);
        let theta_step = (theta_hi - theta_lo) / (theta_n - 1) as f64;
        let phi_step = (phi_hi - phi_lo) / (phi_n - 1) as f64;
        let (mut mass, mut e_t, mut e_p, mut e_tt, mut e_pp, mut e_tp) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        for i in 0..theta_n {
            let theta = theta_lo + theta_step * i as f64;
            // marginal likelihood of y2 given theta, integrating phi
            let mut marginal = 0.0;
            for j in 0..phi_n {
                let phi = phi_lo + phi_step * j as f64;
                marginal += normal_pdf(phi, model.specific_prior.mean, model.specific_prior.variance)
                    * normal_pdf(model.cost_observation.mean, theta + phi, model.cost_observation.variance)
                    * phi_step;
            }
            let weight_theta = normal_pdf(theta, pooled.mean, pooled.variance)
                * normal_pdf(model.margin_observation.mean, theta, model.margin_observation.variance)
                * marginal.powf(eta);
            for j in 0..phi_n {
                let phi = phi_lo + phi_step * j as f64;
                let conditional = normal_pdf(phi, model.specific_prior.mean, model.specific_prior.variance)
                    * normal_pdf(model.cost_observation.mean, theta + phi, model.cost_observation.variance)
                    / marginal;
                let density = weight_theta * conditional * theta_step * phi_step;
                mass += density;
                e_t += density * theta;
                e_p += density * phi;
                e_tt += density * theta * theta;
                e_pp += density * phi * phi;
                e_tp += density * theta * phi;
            }
        }
        let (mt, mp) = (e_t / mass, e_p / mass);
        [mt, mp, e_tt / mass - mt * mt, e_pp / mass - mp * mp, e_tp / mass - mt * mp]
    }

    #[test]
    fn the_modular_posterior_matches_quadrature_of_its_defining_densities() {
        let model = model();
        for eta in [0.0, 0.35, 1.0] {
            let exact = model.posterior(eta).unwrap();
            let [mt, mp, vt, vp, ctp] = quadrature(&model, eta);
            assert!((exact.theta_mean - mt).abs() < 1e-6, "eta {eta}: {} vs {mt}", exact.theta_mean);
            assert!((exact.phi_mean - mp).abs() < 1e-6, "eta {eta}");
            assert!((exact.theta_variance - vt).abs() < 1e-6, "eta {eta}");
            assert!((exact.phi_variance - vp).abs() < 1e-5, "eta {eta}");
            assert!((exact.theta_phi_covariance - ctp).abs() < 1e-5, "eta {eta}");
        }
    }

    #[test]
    fn full_feedback_is_the_coherent_joint_posterior_from_the_joint_precision() {
        let model = model();
        let pooled = reconcile_priors(model.margin_prior, model.cost_prior, model.pooling_weight)
            .unwrap();
        // Joint Gaussian over (theta, phi): y1 = theta + e1, y2 = theta + phi + e2.
        let (s1, s2) = (model.margin_observation.variance, model.cost_observation.variance);
        let (vp, mp) = (model.specific_prior.variance, model.specific_prior.mean);
        let p11 = 1.0 / pooled.variance + 1.0 / s1 + 1.0 / s2;
        let p12 = 1.0 / s2;
        let p22 = 1.0 / vp + 1.0 / s2;
        let b1 = pooled.mean / pooled.variance
            + model.margin_observation.mean / s1
            + model.cost_observation.mean / s2;
        let b2 = mp / vp + model.cost_observation.mean / s2;
        let det = p11 * p22 - p12 * p12;
        let (c11, c12, c22) = (p22 / det, -p12 / det, p11 / det);
        let policy = MixedModulePolicy::coherent_joint();
        let joint = policy.posterior(&model).unwrap();
        assert!((joint.theta_mean - (c11 * b1 + c12 * b2)).abs() < 1e-12);
        assert!((joint.phi_mean - (c12 * b1 + c22 * b2)).abs() < 1e-12);
        assert!((joint.theta_variance - c11).abs() < 1e-12);
        assert!((joint.phi_variance - c22).abs() < 1e-12);
        assert!((joint.theta_phi_covariance - c12).abs() < 1e-12);
        let feedback = MixedModulePolicy::declared_modular("pooled theta", CutFeedback::Feedback)
            .unwrap()
            .posterior(&model)
            .unwrap();
        assert_eq!(feedback, joint);
    }

    #[test]
    fn a_cut_blocks_what_flows_back_and_keeps_what_flows_forward() {
        let base = model();
        let cut = MixedModulePolicy::declared_modular("pooled theta", CutFeedback::Cut).unwrap();
        let joint = MixedModulePolicy::coherent_joint();
        let moved = SharedParameterModel {
            cost_observation: law(base.cost_observation.mean + 3.0, base.cost_observation.variance),
            ..base
        };
        // Theta under the cut ignores the second module's data entirely.
        let before = cut.posterior(&base).unwrap();
        let after = cut.posterior(&moved).unwrap();
        assert_eq!(before.theta_mean.to_bits(), after.theta_mean.to_bits());
        assert_eq!(before.theta_variance.to_bits(), after.theta_variance.to_bits());
        // Under feedback it moves, and the variance is smaller than the cut's.
        let fed_before = joint.posterior(&base).unwrap();
        let fed_after = joint.posterior(&moved).unwrap();
        assert!((fed_after.theta_mean - fed_before.theta_mean).abs() > 0.1);
        assert!(fed_before.theta_variance < before.theta_variance);
        // phi still conditions on theta under the cut: the dependence survives.
        assert!(before.theta_phi_covariance < 0.0);
        assert!(before.theta_contrast_correlation().abs() > 0.1);
        assert!(
            (before.theta_contrast_correlation() - fed_before.theta_contrast_correlation()).abs()
                > 1e-3
        );
        // The cut posterior of the contrast is a different law from the joint.
        assert!((before.contrast_variance() - fed_before.contrast_variance()).abs() > 1e-4);
    }

    #[test]
    fn partial_feedback_interpolates_and_the_envelope_brackets_every_pooling_weight() {
        let model = model();
        let cut = model.posterior(0.0).unwrap();
        let half = model.posterior(0.5).unwrap();
        let full = model.posterior(1.0).unwrap();
        assert!(full.theta_variance < half.theta_variance && half.theta_variance < cut.theta_variance);
        let envelope = model.pooling_envelope(1.0, 0.0, 1.0, 40).unwrap();
        assert_eq!(envelope.weights, 41);
        for index in 0..=200 {
            let weight = f64::from(index) / 200.0;
            let at = SharedParameterModel { pooling_weight: weight, ..model }.posterior(1.0).unwrap();
            assert!(at.theta_mean >= envelope.theta_mean.low - 1e-3);
            assert!(at.theta_mean <= envelope.theta_mean.high + 1e-3);
        }
        assert!(envelope.theta_mean.high - envelope.theta_mean.low > 0.05);
        // Identical priors have no pooling sensitivity.
        let agreed = SharedParameterModel { cost_prior: model.margin_prior, ..model };
        let flat = agreed.pooling_envelope(1.0, 0.0, 1.0, 10).unwrap();
        assert!(flat.theta_mean.high - flat.theta_mean.low < 1e-12);
        assert_eq!(agreed.posterior(1.0).unwrap().prior_conflict, 0.0);
    }

    #[test]
    fn malformed_models_and_pools_are_refused() {
        let model = model();
        assert!(model.posterior(1.5).is_err());
        assert!(SharedParameterModel { pooling_weight: 1.2, ..model }.posterior(1.0).is_err());
        assert!(SharedParameterModel { feedback_weight: 0.0, ..model }.posterior(1.0).is_err());
        assert!(
            SharedParameterModel { specific_prior: law(0.0, 0.0), ..model }.posterior(1.0).is_err()
        );
        assert!(reconcile_priors(law(0.0, 1.0), law(1.0, f64::NAN), 0.5).is_err());
        assert!(model.pooling_envelope(1.0, 0.7, 0.2, 5).is_err());
        let error = model.posterior(-0.1).unwrap_err();
        assert_eq!(error.detail(), "mixed_module.invalid_model");
        assert!(error.to_string().contains("feedback strength"));
        let pooled = reconcile_priors(law(0.0, 1.0), law(2.0, 1.0), 0.5).unwrap();
        assert!((pooled.mean - 1.0).abs() < 1e-12 && (pooled.variance - 1.0).abs() < 1e-12);
    }

}
