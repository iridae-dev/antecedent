//! Learned joint source-target transport (2.3A cell X4 remainder and 2.3B B1 model-provider
//! row `learned_joint_transport`): the outcome mechanism is **fitted through
//! `antecedent-learn`**.
//!
//! # The named row
//!
//! * **Graph.** The fixed DAG and selection assumptions of one already identified 2.2
//!   transport row (a direct or baseline-standardization [`TransportIdentification`]). An
//!   ADMG or a graph posterior is refused (`learned_joint_transport.unsupported_graph`).
//! * **Provider.** `antecedent-learn`'s [`KnownVarianceBasisRegression`] over a declared
//!   [`PolynomialBasis`] in the certified covariates, with a dense Gaussian coefficient
//!   prior and a declared known noise variance per source. This is the learn crate's own
//!   fitted model object (basis declaration, prior declaration, backend provenance), not a
//!   posterior wrapper on a frequentist point: this module assembles the transport design
//!   and reads the estimand; every posterior moment and draw comes from the learn fit.
//! * **Query.** The target average effect `psi = mean_target[E(Y|A=1,x) - E(Y|A=0,x)]`.
//!
//! # Model
//!
//! With basis `phi(x)` (degree `d` per covariate, `m = p * d` terms) the outcome mechanism
//! of source `s` is `Y | A, X ~ Normal(w' theta + v' gamma_s, sigma_s^2)` where
//! `w = [A, A phi(x)]` (effect block) plus `phi(x)` when the basis main effects are
//! invariant ([`VaryingBlock::Intercept`]), and the population-varying block `v` is
//! `[1]` or `[1, phi(x)]` and never interacts with treatment. `theta` is shared by every
//! source and the target, so a varying treatment effect is structurally impossible to
//! declare. The varying block is one per source or one shared block (explicit draw
//! sharing, [`SourceSharing`]). With `d = 1` the model is exactly that of
//! [`crate::joint_bayesian_transport`], which the tests use as a labelled regression
//! cross-check between two Antecedent paths.
//!
//! # What `antecedent-learn` exposes
//!
//! The fit is conjugate Gaussian with a known noise variance, so the learn object exposes
//! the exact posterior mean and covariance and iid draws from that Gaussian. There is no
//! sampler: the effective sample size equals the draw count and no R-hat exists. A
//! non-conjugate learned model (unknown variance, non-Gaussian likelihood, MCMC) is not
//! provided by this row.
//!
//! # Guards (same discipline as the joint engine)
//!
//! * Identification is read from the supplied certificate and copied to the result; prior
//!   mass never changes it.
//! * A datum or snapshot may enter exactly one of the likelihood and a prior bank.
//! * Sources are independent with disjoint units; conflicting sources are never pooled
//!   silently: pairwise posterior disagreement of per-source target effects (each from a
//!   learn fit of that source alone) is reported.
//! * Target covariate rows must lie in the axis-aligned box of a source sample (declared
//!   unsupported-mass tolerance).
//! * The weighted information matrix of the basis design must have full rank: a
//!   proper prior would otherwise answer a design the data cannot identify.
//! * Calibration is **unmeasured**; no coverage is claimed and the public interval route
//!   stays closed ([`route_frozen_refusal`]).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::joint_bayesian_transport::{
    FailedFitDiagnostics, GaussianPrior, IdentificationRecord, JOINT_TRANSPORT_MAX_DRAWS,
    JOINT_TRANSPORT_MAX_PARAMETERS, JointDraws, JointPriors, JointTransportCalibration,
    JointTransportOptions, JointTransportRefusal, OverlapDiagnostic, PairDisagreement, SourceData,
    SourceDependence, SourceDisagreement, SourceEffectSummary, SourceSharing, TargetData,
    TransportGraphClass, VaryingBlock, check_prior_data_overlap,
};
use antecedent_core::ExecutionContext;
use antecedent_identify::{TransportFormula, TransportIdentification};
use antecedent_learn::{
    BasisDesign, BasisRegressionPosterior, KnownVarianceBasisRegression, KnownVarianceBasisSpec,
    LearnError, PolynomialBasis, information_rank,
};
use antecedent_prob::ProbError;
use std::collections::BTreeSet;

/// Route name of the closed public interval route.
pub const LEARNED_JOINT_ROUTE: &str = "antecedent.transport.learned_joint";
/// Sampler recorded on fits: iid draws from the learn crate's exact Gaussian posterior.
pub const LEARNED_JOINT_SAMPLER: &str = "learn_conjugate_gaussian_iid";
/// The provider crate of the outcome mechanism.
pub const LEARNED_JOINT_PROVIDER: &str = "antecedent-learn";
/// The query the row answers.
pub const LEARNED_JOINT_QUERY: &str = "target_average_effect";

/// Detail of the route-level refusal of the public interval route.
pub const DETAIL_ROUTE_FROZEN: &str = "learned_joint_transport.route_frozen";
/// Detail of an unsupported graph class.
pub const DETAIL_UNSUPPORTED_GRAPH: &str = "learned_joint_transport.unsupported_graph";
/// Detail of observations used by both a prior bank and the likelihood.
pub const DETAIL_PRIOR_DATA_OVERLAP: &str = "learned_joint_transport.prior_data_overlap";
/// Detail of target covariate mass outside the source support.
pub const DETAIL_WEAK_OVERLAP: &str = "learned_joint_transport.weak_overlap";
/// Detail of a rank-deficient basis design.
pub const DETAIL_RANK_DEFICIENT: &str = "learned_joint_transport.rank_deficient";
/// Detail of a failed learn fit.
pub const DETAIL_FIT_FAILED: &str = "learned_joint_transport.fit_failed";
/// Detail of a draw count outside `1..=JOINT_TRANSPORT_MAX_DRAWS`.
pub const DETAIL_TOO_MANY_DRAWS: &str = "learned_joint_transport.too_many_draws";
/// Detail of a model with more parameters than the bound.
pub const DETAIL_TOO_MANY_PARAMETERS: &str = "learned_joint_transport.too_many_parameters";
/// Detail of a missing target covariate law.
pub const DETAIL_MISSING_LAW: &str = "learned_joint_transport.missing_law";
/// Detail of absent source evidence.
pub const DETAIL_MISSING_SOURCE: &str = "learned_joint_transport.missing_source";
/// Detail of undeclared, overlapping or unknown source dependence.
pub const DETAIL_SOURCE_DEPENDENCE: &str = "learned_joint_transport.source_dependence";
/// Detail of a certificate that does not license the model.
pub const DETAIL_IDENTIFICATION: &str = "learned_joint_transport.identification_not_certified";
/// Detail of covariates that differ from the certified standardizers.
pub const DETAIL_FEATURE_MISMATCH: &str = "learned_joint_transport.feature_mismatch";
/// Detail of a prior of the wrong dimension.
pub const DETAIL_PRIOR_DIMENSION: &str = "learned_joint_transport.prior_dimension";
/// Detail of a prior that is not a symmetric positive definite Gaussian.
pub const DETAIL_INVALID_PRIOR: &str = "learned_joint_transport.invalid_prior";
/// Detail of an unsupported basis declaration.
pub const DETAIL_INVALID_BASIS: &str = "learned_joint_transport.invalid_basis";
/// Detail of malformed data or options.
pub const DETAIL_INVALID_INPUT: &str = "learned_joint_transport.invalid_input";
/// Detail of a cancelled fit.
pub const DETAIL_CANCELLED: &str = "learned_joint_transport.cancelled";

/// Declaration of the learned joint source-target model.
#[derive(Clone, Debug, PartialEq)]
pub struct LearnedJointModel {
    /// Graph class; only [`TransportGraphClass::FixedDag`] is supported.
    pub graph: TransportGraphClass,
    /// Raw variable ids of the covariates; must equal the certified standardizers.
    pub features: Vec<u32>,
    /// Polynomial degree of the basis in every covariate, `1..=6`.
    pub basis_degree: usize,
    /// Varying coefficient block.
    pub varying: VaryingBlock,
    /// Sharing of the varying block across sources.
    pub sharing: SourceSharing,
    /// Declared dependence between the sources.
    pub dependence: SourceDependence,
    /// Priors of the invariant block (`1 + 2m` or `1 + m` coefficients) and of one
    /// varying block (`1` or `1 + m` coefficients).
    pub priors: JointPriors,
    /// Largest accepted unsupported target mass, in `[0, 1)`.
    pub max_unsupported_mass: f64,
    /// Standardized disagreement above which two sources are flagged, positive.
    pub conflict_z_threshold: f64,
}

/// The named graph / provider / query row a fit answered.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LearnedProviderRecord {
    /// [`LEARNED_JOINT_PROVIDER`].
    pub provider: &'static str,
    /// `fixed_dag`.
    pub graph: &'static str,
    /// [`LEARNED_JOINT_QUERY`].
    pub query: &'static str,
    /// Model identity reported by the learn fit.
    pub learn_model_id: String,
    /// Backend implementation reported by the learn fit.
    pub learn_implementation: String,
    /// Basis declaration identity, for example `polynomial_degree_2`.
    pub basis_id: String,
    /// Basis term names in evaluation order.
    pub basis_terms: Vec<String>,
}

/// Diagnostics of a successful fit.
#[derive(Clone, Debug, PartialEq)]
pub struct LearnedJointDiagnostics {
    /// [`LEARNED_JOINT_SAMPLER`].
    pub sampler: &'static str,
    /// Number of draws.
    pub draw_count: usize,
    /// Effective sample size (equal to `draw_count` for exact iid draws).
    pub effective_sample_size: f64,
    /// Split R-hat; absent for exact iid draws.
    pub r_hat: Option<f64>,
    /// Smallest relative pivot of the weighted data information matrix.
    pub min_pivot_ratio: f64,
    /// Rank of the weighted data information matrix (equal to the parameter count, as a
    /// deficient design is refused).
    pub basis_rank: usize,
    /// Condition lower bound of the learn fit's equilibrated posterior precision.
    pub precision_condition: f64,
    /// Target overlap.
    pub overlap: OverlapDiagnostic,
    /// Source disagreement.
    pub disagreement: SourceDisagreement,
    /// Always [`JointTransportCalibration::Unmeasured`].
    pub calibration: JointTransportCalibration,
}

/// Learn-fitted joint posterior of the transport row.
#[derive(Clone, Debug, PartialEq)]
pub struct LearnedJointFit {
    /// Parameter names in order: invariant, then every varying block.
    pub parameter_names: Vec<String>,
    /// Exact posterior mean of the parameters.
    pub posterior_mean: Vec<f64>,
    /// Exact posterior covariance, row-major `D * D`.
    pub posterior_covariance: Vec<f64>,
    /// Names of the effects: every source (`effect.source.<id>`) then `effect.target`.
    pub effect_names: Vec<String>,
    /// Exact posterior mean of each effect.
    pub effect_means: Vec<f64>,
    /// Exact posterior covariance of the effects, row-major `E * E`.
    pub effect_covariance: Vec<f64>,
    /// Posterior mean of the target effect.
    pub target_effect_mean: f64,
    /// Posterior variance of the target effect.
    pub target_effect_variance: f64,
    /// Aligned joint draws of parameters and effects.
    pub draws: JointDraws,
    /// Diagnostics.
    pub diagnostics: LearnedJointDiagnostics,
    /// Identification of the checked derivation.
    pub identification: IdentificationRecord,
    /// The graph / provider / query row.
    pub provider: LearnedProviderRecord,
    /// Canonical identity of the model, basis, priors and bounds.
    pub model_identity: String,
}

impl LearnedJointFit {
    /// Effect columns (source effects then target effect) of the joint draws, draw-major.
    #[must_use]
    pub fn effect_draws(&self) -> (Vec<String>, Vec<f64>) {
        let width = self.draws.width();
        let first = width - self.effect_names.len();
        let mut values = Vec::with_capacity(self.draws.n_draws * self.effect_names.len());
        for draw in 0..self.draws.n_draws {
            values.extend_from_slice(&self.draws.values[draw * width + first..(draw + 1) * width]);
        }
        (self.effect_names.clone(), values)
    }
}

/// Private fallible helpers box the refusal so their `Result` stays small.
type Refused = Box<JointTransportRefusal>;

fn refuse(
    code: &'static str,
    detail: &'static str,
    stage: &'static str,
    message: impl Into<String>,
) -> JointTransportRefusal {
    JointTransportRefusal {
        code,
        detail,
        message: message.into(),
        failure: FailedFitDiagnostics { stage, ..FailedFitDiagnostics::default() },
    }
}

fn invalid(detail: &'static str, message: impl Into<String>) -> JointTransportRefusal {
    refuse(antecedent_core::reason_code!("invalid_argument"), detail, "validate", message)
}

/// The route-level refusal of the public interval route: the learned joint posterior
/// route lacks calibration evidence, so no interval is published.
#[must_use]
pub fn route_frozen_refusal() -> JointTransportRefusal {
    refuse(
        antecedent_core::reason_code!("cell_not_licensed"),
        DETAIL_ROUTE_FROZEN,
        "route",
        format!(
            "{LEARNED_JOINT_ROUTE} lacks likelihood, value and calibration evidence; \
             the learn fit's posterior is not a published interval"
        ),
    )
}

/// The refusal of a graph class other than the fixed DAG.
#[must_use]
pub fn unsupported_graph_refusal() -> JointTransportRefusal {
    refuse(
        antecedent_core::reason_code!("route_not_supported"),
        DETAIL_UNSUPPORTED_GRAPH,
        "validate",
        "an ADMG or graph-posterior transport query is not supported by the fixed-DAG model",
    )
}

/// Map a learn error to a typed refusal.
fn learn_refusal(error: &LearnError) -> Refused {
    let refusal = match error {
        LearnError::Probability(ProbError::InvalidPrior { .. }) => {
            invalid(DETAIL_INVALID_PRIOR, format!("the learn fit rejected the prior: {error}"))
        }
        LearnError::Shape { .. } => {
            invalid(DETAIL_INVALID_INPUT, format!("the learn fit rejected the design: {error}"))
        }
        _ => refuse(
            antecedent_core::reason_code!("transport_numerical_failure"),
            DETAIL_FIT_FAILED,
            "fit",
            format!("the antecedent-learn Bayesian basis fit failed: {error}"),
        ),
    };
    Box::new(refusal)
}

/// Basis dimension, block sizes and parameter index map.
#[derive(Clone, Copy)]
struct Layout {
    m: usize,
    q: usize,
    r: usize,
    blocks: usize,
    sharing: SourceSharing,
    varying: VaryingBlock,
}

impl Layout {
    fn new(m: usize, model: &LearnedJointModel, sources: usize) -> Self {
        let (q, r) = match model.varying {
            VaryingBlock::Intercept => (1 + 2 * m, 1),
            VaryingBlock::InterceptAndCovariates => (1 + m, 1 + m),
        };
        let blocks = match model.sharing {
            SourceSharing::IndependentVaryingBlocks => sources,
            SourceSharing::SharedVaryingBlock => 1,
        };
        Self { m, q, r, blocks, sharing: model.sharing, varying: model.varying }
    }

    const fn dim(&self) -> usize {
        self.q + self.blocks * self.r
    }

    const fn block(&self, position: usize) -> usize {
        match self.sharing {
            SourceSharing::IndependentVaryingBlocks => position,
            SourceSharing::SharedVaryingBlock => 0,
        }
    }

    const fn index(&self, k: usize, block: usize) -> usize {
        if k < self.q { k } else { self.q + block * self.r + (k - self.q) }
    }

    /// Row features `[w, v]` of one unit from its basis row `phi`.
    fn row(&self, treated: bool, phi: &[f64], out: &mut Vec<f64>) {
        out.clear();
        let a = if treated { 1.0 } else { 0.0 };
        out.push(a);
        out.extend(phi.iter().map(|value| a * value));
        if self.varying == VaryingBlock::Intercept {
            out.extend_from_slice(phi);
        }
        out.push(1.0);
        if self.varying == VaryingBlock::InterceptAndCovariates {
            out.extend_from_slice(phi);
        }
    }

    /// Effect vector `[1, phibar, 0..]` over the full parameter space.
    fn effect_vector(&self, mean_phi: &[f64]) -> Vec<f64> {
        let mut c = vec![0.0; self.dim()];
        c[0] = 1.0;
        c[1..=self.m].copy_from_slice(mean_phi);
        c
    }
}

fn parameter_names(layout: &Layout, terms: &[String], ids: &[&str]) -> Vec<String> {
    let mut names = vec!["theta.treatment".to_owned()];
    for term in terms {
        names.push(format!("theta.treatment:{term}"));
    }
    if layout.varying == VaryingBlock::Intercept {
        for term in terms {
            names.push(format!("theta.{term}"));
        }
    }
    let owners: Vec<&str> = match layout.sharing {
        SourceSharing::IndependentVaryingBlocks => ids[..layout.blocks].to_vec(),
        SourceSharing::SharedVaryingBlock => vec!["shared"; layout.blocks],
    };
    for owner in owners {
        names.push(format!("gamma.{owner}.intercept"));
        if layout.varying == VaryingBlock::InterceptAndCovariates {
            for term in terms {
                names.push(format!("gamma.{owner}.{term}"));
            }
        }
    }
    names
}

fn quadratic(a: &[f64], n: usize, left: &[f64], right: &[f64]) -> f64 {
    let mut total = 0.0;
    for i in 0..n {
        let image: f64 = (0..n).map(|j| a[i * n + j] * right[j]).sum();
        total += left[i] * image;
    }
    total
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Whether a symmetric matrix is positive definite, by an unpivoted Cholesky.
fn positive_definite(a: &[f64], n: usize) -> bool {
    let mut l = vec![0.0; n * n];
    for j in 0..n {
        let mut d = a[j * n + j];
        for k in 0..j {
            d -= l[j * n + k] * l[j * n + k];
        }
        if d.is_nan() || d <= 1e-10 * a[j * n + j].abs().max(f64::MIN_POSITIVE) {
            return false;
        }
        let root = d.sqrt();
        l[j * n + j] = root;
        for i in (j + 1)..n {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            l[i * n + j] = s / root;
        }
    }
    true
}

fn check_prior(prior: &GaussianPrior, dim: usize) -> Result<(), Refused> {
    if prior.mean.len() != dim || prior.covariance.len() != dim * dim {
        return Err(Box::new(refuse(
            antecedent_core::reason_code!("prior_dimension_mismatch"),
            DETAIL_PRIOR_DIMENSION,
            "validate",
            format!("prior block needs {dim} coefficients and a {dim}x{dim} covariance"),
        )));
    }
    let scale = prior.covariance.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let mut symmetric = true;
    for i in 0..dim {
        for j in 0..i {
            let gap = (prior.covariance[i * dim + j] - prior.covariance[j * dim + i]).abs();
            symmetric &= gap <= 1e-12 * scale;
        }
    }
    let finite = prior.mean.iter().chain(&prior.covariance).all(|value| value.is_finite());
    if !symmetric || !finite || !positive_definite(&prior.covariance, dim) {
        return Err(invalid(
            DETAIL_INVALID_PRIOR,
            "prior covariance must be finite, symmetric and positive definite",
        )
        .into());
    }
    Ok(())
}

fn ids_overlap(left: &[String], right: &[String]) -> Vec<String> {
    let set: BTreeSet<&String> = left.iter().collect();
    right.iter().filter(|id| set.contains(id)).cloned().collect()
}

fn certified_record(
    id: &TransportIdentification,
) -> Result<(IdentificationRecord, Vec<u32>), Refused> {
    match id {
        TransportIdentification::Transportable {
            formula: TransportFormula::Direct(_),
            certificate,
        } => Ok((
            IdentificationRecord {
                rule: certificate.rule.to_string(),
                status: "identified",
                formula: "direct",
            },
            Vec::new(),
        )),
        TransportIdentification::Transportable {
            formula: TransportFormula::Standardize { over, .. },
            certificate,
        } => Ok((
            IdentificationRecord {
                rule: certificate.rule.to_string(),
                status: "identified",
                formula: "standardize",
            },
            over.iter().map(|v| v.raw()).collect(),
        )),
        TransportIdentification::MissingEvidence(_) => Err(Box::new(refuse(
            antecedent_core::reason_code!("transport_missing_evidence"),
            DETAIL_IDENTIFICATION,
            "identify",
            "the transport derivation lacks a required source regime",
        ))),
        _ => Err(Box::new(refuse(
            antecedent_core::reason_code!("transport_not_certified"),
            DETAIL_IDENTIFICATION,
            "identify",
            "a direct or standardization transport certificate is required; a prior \
             cannot supply identification",
        ))),
    }
}

fn validate_declaration(
    id: &TransportIdentification,
    model: &LearnedJointModel,
    options: &JointTransportOptions,
) -> Result<(IdentificationRecord, PolynomialBasis), Refused> {
    if model.graph != TransportGraphClass::FixedDag {
        return Err(unsupported_graph_refusal().into());
    }
    let (record, mut expected) = certified_record(id)?;
    expected.sort_unstable();
    let mut actual = model.features.clone();
    actual.sort_unstable();
    if expected != actual {
        return Err(Box::new(refuse(
            antecedent_core::reason_code!("invalid_argument"),
            DETAIL_FEATURE_MISMATCH,
            "validate",
            "model covariates must equal the certified standardizers",
        )));
    }
    let basis = PolynomialBasis::new(model.basis_degree).map_err(|_| {
        invalid(DETAIL_INVALID_BASIS, "the polynomial basis degree must be in 1..=6")
    })?;
    if options.draws == 0 || options.draws > JOINT_TRANSPORT_MAX_DRAWS {
        return Err(invalid(
            DETAIL_TOO_MANY_DRAWS,
            format!("draw count must be in 1..={JOINT_TRANSPORT_MAX_DRAWS}"),
        )
        .into());
    }
    if model.dependence != SourceDependence::IndependentSamples {
        return Err(Box::new(refuse(
            antecedent_core::reason_code!("sampling_dependence_unknown"),
            DETAIL_SOURCE_DEPENDENCE,
            "validate",
            "only independent source samples with disjoint units are supported",
        )));
    }
    let tolerance_ok = model.max_unsupported_mass.is_finite()
        && (0.0..1.0).contains(&model.max_unsupported_mass)
        && model.conflict_z_threshold.is_finite()
        && model.conflict_z_threshold > 0.0;
    if !tolerance_ok {
        return Err(invalid(
            DETAIL_INVALID_INPUT,
            "overlap tolerance must be in [0, 1) and the conflict threshold positive",
        )
        .into());
    }
    Ok((record, basis))
}

fn validate_sources(model: &LearnedJointModel, sources: &[SourceData]) -> Result<(), Refused> {
    if sources.is_empty() {
        return Err(Box::new(refuse(
            antecedent_core::reason_code!("transport_missing_evidence"),
            DETAIL_MISSING_SOURCE,
            "validate",
            "at least one source trial is required",
        )));
    }
    let p = model.features.len();
    let mut seen = BTreeSet::new();
    for source in sources {
        let n = source.outcome.len();
        let finite = source.outcome.iter().all(|v| v.is_finite())
            && source.covariates.iter().all(|c| c.len() == n && c.iter().all(|v| v.is_finite()));
        if n == 0
            || source.treatment.len() != n
            || source.covariates.len() != p
            || !finite
            || !source.noise_variance.is_finite()
            || source.noise_variance <= 0.0
            || source.id.trim().is_empty()
            || source.identity.snapshot_digest.trim().is_empty()
            || !seen.insert(source.id.as_str())
        {
            return Err(invalid(
                DETAIL_INVALID_INPUT,
                format!(
                    "source `{}` is malformed (shape, finiteness, noise or identity)",
                    source.id
                ),
            )
            .into());
        }
    }
    Ok(())
}

fn validate_target<'a>(
    model: &LearnedJointModel,
    target: Option<&'a TargetData>,
) -> Result<&'a TargetData, Refused> {
    let missing = || {
        refuse(
            antecedent_core::reason_code!("joint_law_required"),
            DETAIL_MISSING_LAW,
            "validate",
            "the target covariate law (a nonempty target sample) is required",
        )
    };
    let target = target.ok_or_else(|| Box::new(missing()))?;
    if target.rows == 0 {
        return Err(Box::new(missing()));
    }
    let finite =
        target.covariates.iter().all(|c| c.len() == target.rows && c.iter().all(|v| v.is_finite()));
    if target.covariates.len() != model.features.len()
        || !finite
        || target.identity.snapshot_digest.trim().is_empty()
    {
        return Err(invalid(DETAIL_INVALID_INPUT, "target covariates are malformed").into());
    }
    Ok(target)
}

/// Unit ownership: sources are disjoint from each other and from the target.
fn check_unit_ownership(sources: &[SourceData], target: &TargetData) -> Result<(), Refused> {
    let mut owners: Vec<&[String]> =
        sources.iter().map(|source| source.identity.datum_ids.as_slice()).collect();
    owners.push(&target.identity.datum_ids);
    for i in 0..owners.len() {
        for j in (i + 1)..owners.len() {
            let shared = ids_overlap(owners[i], owners[j]);
            if !shared.is_empty() {
                let mut refusal = refuse(
                    antecedent_core::reason_code!("sampling_dependence_unknown"),
                    DETAIL_SOURCE_DEPENDENCE,
                    "validate",
                    "sources and target must own disjoint units; shared units would be \
                     counted twice",
                );
                refusal.failure.overlapping_ids = shared;
                return Err(Box::new(refusal));
            }
        }
    }
    Ok(())
}

/// The prior-bank/likelihood double-use refusal, restated under this row's detail.
fn check_double_use(priors: &JointPriors, sources: &[SourceData]) -> Result<(), Refused> {
    check_prior_data_overlap(priors, sources).map_err(|inner| {
        let mut refusal = refuse(inner.code, DETAIL_PRIOR_DATA_OVERLAP, "validate", inner.message);
        refusal.failure.overlapping_ids = inner.failure.overlapping_ids;
        Box::new(refusal)
    })
}

/// Fraction of target rows outside every source covariate box.
fn unsupported_mass(sources: &[SourceData], target: &TargetData) -> f64 {
    let boxes: Vec<Vec<(f64, f64)>> = sources
        .iter()
        .map(|source| {
            source
                .covariates
                .iter()
                .map(|column| {
                    column.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| {
                        (lo.min(*v), hi.max(*v))
                    })
                })
                .collect()
        })
        .collect();
    let outside = (0..target.rows)
        .filter(|&i| {
            !boxes.iter().any(|bounds| {
                bounds
                    .iter()
                    .zip(&target.covariates)
                    .all(|((lo, hi), column)| column[i] >= *lo && column[i] <= *hi)
            })
        })
        .count();
    outside as f64 / target.rows as f64
}

/// Basis rows of `rows` units, row-major `rows * m`.
fn expand(basis: PolynomialBasis, covariates: &[Vec<f64>], rows: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(rows * basis.width(covariates.len()));
    let mut x = vec![0.0; covariates.len()];
    let mut phi = Vec::new();
    for i in 0..rows {
        for (j, column) in covariates.iter().enumerate() {
            x[j] = column[i];
        }
        basis.evaluate(&x, &mut phi);
        out.extend_from_slice(&phi);
    }
    out
}

fn column_means(values: &[f64], rows: usize, m: usize) -> Vec<f64> {
    let mut means = vec![0.0; m];
    for i in 0..rows {
        for (k, mean) in means.iter_mut().enumerate() {
            *mean += values[i * m + k];
        }
    }
    means.iter().map(|total| total / rows as f64).collect()
}

/// One source with its basis rows.
#[derive(Clone, Copy)]
struct Unit<'a> {
    source: &'a SourceData,
    phi: &'a [f64],
}

/// Column-major stacked transport design over the units.
struct Stacked {
    x: Vec<f64>,
    y: Vec<f64>,
    w: Vec<f64>,
    nrows: usize,
}

fn stack(layout: &Layout, units: &[Unit<'_>]) -> Stacked {
    let dim = layout.dim();
    let nrows: usize = units.iter().map(|unit| unit.source.outcome.len()).sum();
    let mut x = vec![0.0; nrows * dim];
    let mut y = Vec::with_capacity(nrows);
    let mut w = Vec::with_capacity(nrows);
    let mut row = Vec::new();
    let mut at = 0;
    for (position, unit) in units.iter().enumerate() {
        let block = layout.block(position);
        let precision = 1.0 / unit.source.noise_variance;
        for i in 0..unit.source.outcome.len() {
            let phi = &unit.phi[i * layout.m..(i + 1) * layout.m];
            layout.row(unit.source.treatment[i], phi, &mut row);
            for (k, value) in row.iter().enumerate() {
                x[layout.index(k, block) * nrows + at] = *value;
            }
            y.push(unit.source.outcome[i]);
            w.push(precision);
            at += 1;
        }
    }
    Stacked { x, y, w, nrows }
}

/// Block-diagonal prior mean and covariance over the full parameter space.
fn joint_prior(layout: &Layout, priors: &JointPriors) -> (Vec<f64>, Vec<f64>) {
    let dim = layout.dim();
    let mut mean = vec![0.0; dim];
    let mut covariance = vec![0.0; dim * dim];
    mean[..layout.q].copy_from_slice(&priors.invariant.mean);
    for i in 0..layout.q {
        for j in 0..layout.q {
            covariance[i * dim + j] = priors.invariant.covariance[i * layout.q + j];
        }
    }
    for block in 0..layout.blocks {
        let offset = layout.q + block * layout.r;
        mean[offset..offset + layout.r].copy_from_slice(&priors.varying.mean);
        for i in 0..layout.r {
            for j in 0..layout.r {
                covariance[(offset + i) * dim + offset + j] =
                    priors.varying.covariance[i * layout.r + j];
            }
        }
    }
    (mean, covariance)
}

/// Everything a learn fit needs besides the units.
struct Engine<'a> {
    priors: &'a JointPriors,
    terms: &'a [String],
    exec: &'a ExecutionContext,
}

struct LearnFit {
    posterior: BasisRegressionPosterior,
    min_pivot_ratio: f64,
}

fn fit_blocks(
    eng: &Engine<'_>,
    layout: &Layout,
    units: &[Unit<'_>],
    draws: usize,
    seed: u64,
    check_rank: bool,
) -> Result<LearnFit, Refused> {
    if eng.exec.cancellation.is_cancelled() {
        return Err(Box::new(refuse(
            antecedent_core::reason_code!("transport_budget_cancel"),
            DETAIL_CANCELLED,
            "fit",
            "learned joint transport was cancelled",
        )));
    }
    let dim = layout.dim();
    let stacked = stack(layout, units);
    let design = BasisDesign {
        x_colmajor: &stacked.x,
        nrows: stacked.nrows,
        ncols: dim,
        y: &stacked.y,
        precision_weights: &stacked.w,
    };
    let rank = information_rank(&design).map_err(|error| learn_refusal(&error))?;
    if let (true, Some(column)) = (check_rank, rank.deficient_column) {
        let ids: Vec<&str> = units.iter().map(|unit| unit.source.id.as_str()).collect();
        let names = parameter_names(layout, eng.terms, &ids);
        let mut refusal = refuse(
            antecedent_core::reason_code!("design_rank_deficient"),
            DETAIL_RANK_DEFICIENT,
            "fit",
            format!("the basis design is rank deficient at parameter {}", names[column]),
        );
        refusal.failure.min_pivot_ratio = Some(rank.min_pivot_ratio);
        refusal.failure.failing_column = Some(names[column].clone());
        return Err(Box::new(refusal));
    }
    let (prior_mean, prior_covariance) = joint_prior(layout, eng.priors);
    let spec = KnownVarianceBasisSpec {
        prior_mean: &prior_mean,
        prior_covariance: &prior_covariance,
        n_draws: draws,
        seed,
    };
    let posterior = KnownVarianceBasisRegression
        .fit(design, &spec, eng.exec)
        .map_err(|error| learn_refusal(&error))?;
    Ok(LearnFit { posterior, min_pivot_ratio: rank.min_pivot_ratio })
}

fn disagreement(
    eng: &Engine<'_>,
    layout: &Layout,
    model: &LearnedJointModel,
    units: &[Unit<'_>],
    target_vector: &[f64],
    seed: u64,
) -> Result<SourceDisagreement, Refused> {
    if units.len() < 2 {
        return Ok(SourceDisagreement {
            per_source: Vec::new(),
            pairs: Vec::new(),
            max_abs_z: 0.0,
            flagged: false,
        });
    }
    let alone_layout = Layout { blocks: 1, ..*layout };
    let mut per_source = Vec::with_capacity(units.len());
    for unit in units {
        let alone = fit_blocks(eng, &alone_layout, std::slice::from_ref(unit), 1, seed, false)?;
        let c = &target_vector[..alone_layout.dim()];
        per_source.push(SourceEffectSummary {
            source_id: unit.source.id.clone(),
            mean: dot(c, &alone.posterior.mean),
            variance: quadratic(&alone.posterior.covariance, alone_layout.dim(), c, c),
        });
    }
    let mut pairs = Vec::new();
    let mut max_abs_z = 0.0_f64;
    for i in 0..per_source.len() {
        for j in (i + 1)..per_source.len() {
            let (a, b) = (&per_source[i], &per_source[j]);
            let z = (a.mean - b.mean) / (a.variance + b.variance).sqrt();
            max_abs_z = max_abs_z.max(z.abs());
            pairs.push(PairDisagreement {
                left: a.source_id.clone(),
                right: b.source_id.clone(),
                z,
            });
        }
    }
    Ok(SourceDisagreement {
        per_source,
        pairs,
        max_abs_z,
        flagged: max_abs_z > model.conflict_z_threshold,
    })
}

fn model_identity(model: &LearnedJointModel, layout: &Layout, sources: &[SourceData]) -> String {
    let prior = |p: &GaussianPrior| {
        let bits: Vec<String> =
            p.mean.iter().chain(&p.covariance).map(|v| format!("{:016x}", v.to_bits())).collect();
        bits.join(",")
    };
    let ids: Vec<&str> = sources.iter().map(|s| s.id.as_str()).collect();
    format!(
        "learned_joint_transport_v1|provider={LEARNED_JOINT_PROVIDER}|graph=fixed_dag|\
         query={LEARNED_JOINT_QUERY}|features={:?}|basis=polynomial_degree_{}|varying={:?}|\
         sharing={:?}|sources={}|noise={:?}|theta_prior={}|gamma_prior={}|dim={}|max_draws={}|\
         max_parameters={}",
        model.features,
        model.basis_degree,
        model.varying,
        model.sharing,
        ids.join(","),
        sources.iter().map(|s| s.noise_variance.to_bits()).collect::<Vec<_>>(),
        prior(&model.priors.invariant),
        prior(&model.priors.varying),
        layout.dim(),
        JOINT_TRANSPORT_MAX_DRAWS,
        JOINT_TRANSPORT_MAX_PARAMETERS,
    )
}

struct EffectSet {
    names: Vec<String>,
    vectors: Vec<Vec<f64>>,
}

fn effect_set(layout: &Layout, units: &[Unit<'_>], target_phi: &[f64], rows: usize) -> EffectSet {
    let mut names = Vec::with_capacity(units.len() + 1);
    let mut vectors = Vec::with_capacity(units.len() + 1);
    for unit in units {
        names.push(format!("effect.source.{}", unit.source.id));
        let means = column_means(unit.phi, unit.source.outcome.len(), layout.m);
        vectors.push(layout.effect_vector(&means));
    }
    names.push("effect.target".to_owned());
    vectors.push(layout.effect_vector(&column_means(target_phi, rows, layout.m)));
    EffectSet { names, vectors }
}

fn draw_joint(
    posterior: &BasisRegressionPosterior,
    effects: &EffectSet,
    parameter_names: &[String],
    options: &JointTransportOptions,
    ctx: &ExecutionContext,
) -> Result<JointDraws, Refused> {
    let dim = posterior.n_coefficients();
    let width = dim + effects.vectors.len();
    let mut values = Vec::with_capacity(options.draws * width);
    let mut parameters = vec![0.0; dim];
    for draw in 0..options.draws {
        if draw % 1024 == 0 && ctx.cancellation.is_cancelled() {
            return Err(Box::new(refuse(
                antecedent_core::reason_code!("transport_budget_cancel"),
                DETAIL_CANCELLED,
                "draw",
                "learned joint transport draws were cancelled",
            )));
        }
        for (coefficient, slot) in parameters.iter_mut().enumerate() {
            *slot = posterior.draws[coefficient * posterior.n_draws + draw];
        }
        values.extend_from_slice(&parameters);
        for c in &effects.vectors {
            values.push(dot(c, &parameters));
        }
    }
    let mut names = parameter_names.to_vec();
    names.extend(effects.names.iter().cloned());
    Ok(JointDraws {
        names,
        values,
        n_draws: options.draws,
        rng_id: format!(
            "antecedent_learn_conjugate_gaussian_v1:seed={}:stream=direct_draw",
            options.seed
        ),
    })
}

/// Fit the learned joint source-target model through `antecedent-learn` and draw aligned
/// joint posterior samples.
///
/// Identification is read from `id` and copied to the result; no prior can change it.
///
/// # Errors
/// A [`JointTransportRefusal`] with a `learned_joint_transport.*` detail (convertible to
/// [`crate::EstimationError`]): unsupported graph, uncertified derivation, mismatched
/// covariates, invalid basis, missing law or source, undeclared source dependence,
/// prior/likelihood double use, prior or draw bounds, weak target overlap, a
/// rank-deficient basis design, a failed learn fit, or cancellation.
// The public refusal type is deliberately rich and part of the stable signature; the private
// pipeline boxes it and this boundary unboxes it.
#[allow(clippy::result_large_err)]
pub fn fit_learned_joint_transport(
    id: &TransportIdentification,
    model: &LearnedJointModel,
    sources: &[SourceData],
    target: Option<&TargetData>,
    options: &JointTransportOptions,
    ctx: &ExecutionContext,
) -> Result<LearnedJointFit, JointTransportRefusal> {
    fit_inner(id, model, sources, target, options, ctx).map_err(|refusal| *refusal)
}

fn fit_inner(
    id: &TransportIdentification,
    model: &LearnedJointModel,
    sources: &[SourceData],
    target: Option<&TargetData>,
    options: &JointTransportOptions,
    ctx: &ExecutionContext,
) -> Result<LearnedJointFit, Refused> {
    let (identification, basis) = validate_declaration(id, model, options)?;
    validate_sources(model, sources)?;
    let target = validate_target(model, target)?;
    let m = basis.width(model.features.len());
    let layout = Layout::new(m, model, sources.len());
    if layout.dim() > JOINT_TRANSPORT_MAX_PARAMETERS {
        return Err(invalid(
            DETAIL_TOO_MANY_PARAMETERS,
            format!(
                "the model has {} parameters; the bound is {JOINT_TRANSPORT_MAX_PARAMETERS}",
                layout.dim()
            ),
        )
        .into());
    }
    check_prior(&model.priors.invariant, layout.q)?;
    check_prior(&model.priors.varying, layout.r)?;
    check_unit_ownership(sources, target)?;
    check_double_use(&model.priors, sources)?;
    let mass = unsupported_mass(sources, target);
    if mass > model.max_unsupported_mass {
        let mut refusal = refuse(
            antecedent_core::reason_code!("transport_support_failure"),
            DETAIL_WEAK_OVERLAP,
            "overlap",
            "target covariate mass outside the source support exceeds the declared tolerance",
        );
        refusal.failure.unsupported_mass = Some(mass);
        refusal.failure.tolerance = Some(model.max_unsupported_mass);
        return Err(Box::new(refusal));
    }
    let feature_names: Vec<String> = model.features.iter().map(|f| format!("x{f}")).collect();
    let terms = basis.term_names(&feature_names);
    let phis: Vec<Vec<f64>> =
        sources.iter().map(|s| expand(basis, &s.covariates, s.outcome.len())).collect();
    let units: Vec<Unit<'_>> =
        sources.iter().zip(&phis).map(|(source, phi)| Unit { source, phi }).collect();
    let target_phi = expand(basis, &target.covariates, target.rows);
    let eng = Engine { priors: &model.priors, terms: &terms, exec: ctx };
    let learned = fit_blocks(&eng, &layout, &units, options.draws, options.seed, true)?;
    let effects = effect_set(&layout, &units, &target_phi, target.rows);
    let target_vector = effects.vectors[effects.vectors.len() - 1].clone();
    let disagreement = disagreement(&eng, &layout, model, &units, &target_vector, options.seed)?;
    let ids: Vec<&str> = sources.iter().map(|s| s.id.as_str()).collect();
    let names = parameter_names(&layout, &terms, &ids);
    let draws = draw_joint(&learned.posterior, &effects, &names, options, ctx)?;
    Ok(assemble(
        model,
        &layout,
        sources,
        &learned,
        effects,
        (draws, names),
        (identification, mass, disagreement),
        (&basis, terms),
        options,
    ))
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn assemble(
    model: &LearnedJointModel,
    layout: &Layout,
    sources: &[SourceData],
    learned: &LearnFit,
    effects: EffectSet,
    drawn: (JointDraws, Vec<String>),
    checked: (IdentificationRecord, f64, SourceDisagreement),
    basis: (&PolynomialBasis, Vec<String>),
    options: &JointTransportOptions,
) -> LearnedJointFit {
    let (draws, parameter_names) = drawn;
    let (identification, mass, disagreement) = checked;
    let posterior = &learned.posterior;
    let dim = layout.dim();
    let count = effects.vectors.len();
    let effect_means: Vec<f64> = effects.vectors.iter().map(|c| dot(c, &posterior.mean)).collect();
    let mut effect_covariance = vec![0.0; count * count];
    for i in 0..count {
        for j in 0..count {
            effect_covariance[i * count + j] =
                quadratic(&posterior.covariance, dim, &effects.vectors[i], &effects.vectors[j]);
        }
    }
    LearnedJointFit {
        target_effect_mean: effect_means[count - 1],
        target_effect_variance: effect_covariance[count * count - 1],
        model_identity: model_identity(model, layout, sources),
        diagnostics: LearnedJointDiagnostics {
            sampler: LEARNED_JOINT_SAMPLER,
            draw_count: options.draws,
            effective_sample_size: options.draws as f64,
            r_hat: None,
            min_pivot_ratio: learned.min_pivot_ratio,
            basis_rank: dim,
            precision_condition: posterior.precision_condition,
            overlap: OverlapDiagnostic {
                unsupported_mass: mass,
                tolerance: model.max_unsupported_mass,
            },
            disagreement,
            calibration: JointTransportCalibration::Unmeasured,
        },
        provider: LearnedProviderRecord {
            provider: LEARNED_JOINT_PROVIDER,
            graph: "fixed_dag",
            query: LEARNED_JOINT_QUERY,
            learn_model_id: posterior.model_id.to_string(),
            learn_implementation: posterior.provenance.implementation.clone(),
            basis_id: basis.0.id(),
            basis_terms: basis.1,
        },
        parameter_names,
        posterior_mean: posterior.mean.clone(),
        posterior_covariance: posterior.covariance.clone(),
        effect_names: effects.names,
        effect_means,
        effect_covariance,
        draws,
        identification,
    }
}
