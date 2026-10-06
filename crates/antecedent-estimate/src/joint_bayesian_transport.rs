//! Joint Bayesian source-target transport (2.3A cell X4, engine only).
//!
//! # The one model
//!
//! This module is a **model provider**, not a posterior wrapper on a frequentist point.
//! It reads one already identified 2.2 transport row (the learned continuous trial
//! transport of [`crate::learned_continuous`]): a certified direct or
//! baseline-standardization derivation ([`TransportIdentification`]) over a fixed DAG,
//! a randomized binary source treatment `A`, certified covariates `X` (the
//! standardizers) and a continuous outcome `Y`, plus a target covariate sample.
//!
//! * **Graph class.** The fixed DAG and selection assumptions of the supplied 2.2
//!   certificate. An ADMG or a graph posterior is refused
//!   (`route_not_supported`, `bayesian_transport.unsupported_graph`).
//! * **Likelihood (one per source `s`, independent units).**
//!   `Y_i | A_i, X_i ~ Normal(w_i' theta + v_i' gamma_s, sigma_s^2)` with the noise
//!   variance `sigma_s^2` **declared and known**. The unknown-variance case is not
//!   modeled: the interval of this cell would not account for estimating `sigma^2`.
//! * **Invariant mechanism.** `theta` is shared by every source and the target. With
//!   `w = [A, A*x_1..A*x_p]` (the effect block) plus `[x_1..x_p]` when the covariate
//!   main effects are invariant ([`VaryingBlock::Intercept`]). Mean exchangeability of
//!   the 2.2 row is the statement that the whole outcome mechanism except the declared
//!   varying block transports.
//! * **Varying mechanism.** `gamma_s` is a population-varying block that never
//!   interacts with treatment: `v = [1]` ([`VaryingBlock::Intercept`]) or
//!   `v = [1, x_1..x_p]` ([`VaryingBlock::InterceptAndCovariates`]). Because the block
//!   does not interact with `A`, the contrast below never involves it, so a varying
//!   treatment effect is structurally impossible to declare. The block is either one
//!   independent block per source ([`SourceSharing::IndependentVaryingBlocks`]) or one
//!   block shared by all sources ([`SourceSharing::SharedVaryingBlock`], explicit draw
//!   sharing).
//! * **Priors.** `theta ~ N(m_theta, S_theta)` and each varying block
//!   `gamma ~ N(m_gamma, S_gamma)`, independent across blocks, with full dense
//!   covariances. A prior carries a [`PriorProvenance`]: a prior built from a prior bank
//!   names every datum or snapshot it consumed.
//! * **Overlap condition.** The target covariate rows must lie inside the axis-aligned
//!   box of at least one source covariate sample. The unsupported target mass must not
//!   exceed the declared tolerance, else the fit refuses
//!   (`bayesian_transport.weak_overlap`). A box is a conservative support check, not a
//!   density-ratio bound.
//! * **Estimand (posterior predictive).** The average intervention effect in the target
//!   sample, `psi = mean_target[ E(Y|A=1,x) - E(Y|A=0,x) ] = c_t' theta` with
//!   `c_t = [1, xbar_t, 0]`, as a function of the posterior of `theta`. The source sample
//!   effect `c_s' theta` of every source is reported on the same draws.
//!
//! # Exact posterior and draws
//!
//! The model is conjugate (Gaussian prior, Gaussian likelihood, known variance), so the
//! joint posterior of `(theta, gamma_1..gamma_B)` is Gaussian with precision
//! `P0 + sum_s Z_s' Z_s / sigma_s^2` and mean `Sigma (P0 m0 + sum_s Z_s' y_s / sigma_s^2)`.
//! The engine reports these exact moments and draws **aligned joint** samples
//! (`DrawAlignment::Joint`): one draw is one realization of every parameter and of every
//! effect, so source-target coupling through the shared `theta` is preserved. The draws
//! are iid, so the effective sample size equals the draw count; the field exists for a
//! future numerical (MCMC) path, which this module does not provide.
//!
//! # What this is not
//!
//! * Prior mass never changes the structural identification status: identification is
//!   the supplied [`TransportIdentification`], read once and copied to the result.
//! * A datum or snapshot may enter exactly one of the likelihood and the construction of
//!   a prior bank; both refuse as `bayesian_transport.prior_data_overlap`.
//! * Sources must be independent samples with disjoint units; overlapping or unknown
//!   dependence refuses. Conflicting sources are never silently pooled: pairwise
//!   posterior disagreement of the per-source target effects is reported.
//! * Calibration is **unmeasured**. No coverage is claimed and the public interval route
//!   stays closed ([`route_frozen_refusal`]).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::splitmix::{seed_mix, splitmix64};
use crate::{EstimationError, RefusalFields};
use antecedent_core::ExecutionContext;
use antecedent_identify::{TransportFormula, TransportIdentification};
use std::collections::BTreeSet;
use std::fmt;

/// Most posterior draws one fit may request.
pub const JOINT_TRANSPORT_MAX_DRAWS: usize = 100_000;
/// Most model parameters (invariant plus every varying block) one fit may declare.
pub const JOINT_TRANSPORT_MAX_PARAMETERS: usize = 256;
/// Route name of the closed public interval route.
pub const JOINT_TRANSPORT_ROUTE: &str = "antecedent.transport.joint_bayesian";
/// Sampler recorded on exact conjugate fits.
pub const JOINT_TRANSPORT_SAMPLER: &str = "exact_conjugate_iid";
/// Relative pivot below which a data information matrix counts as singular.
pub const JOINT_TRANSPORT_PIVOT_TOLERANCE: f64 = 1e-10;

/// Detail of the route-level refusal of the public interval route.
pub const DETAIL_ROUTE_FROZEN: &str = "bayesian_transport.route_frozen";
/// Detail of an unsupported graph class.
pub const DETAIL_UNSUPPORTED_GRAPH: &str = "bayesian_transport.unsupported_graph";
/// Detail of observations used by both a prior bank and the likelihood.
pub const DETAIL_PRIOR_DATA_OVERLAP: &str = "bayesian_transport.prior_data_overlap";
/// Detail of target covariate mass outside the source support.
pub const DETAIL_WEAK_OVERLAP: &str = "bayesian_transport.weak_overlap";
/// Detail of a singular or numerically failed fit.
pub const DETAIL_SINGULAR_FIT: &str = "bayesian_transport.singular_fit";
/// Detail of a draw count outside `1..=JOINT_TRANSPORT_MAX_DRAWS`.
pub const DETAIL_TOO_MANY_DRAWS: &str = "bayesian_transport.too_many_draws";
/// Detail of a model with more than `JOINT_TRANSPORT_MAX_PARAMETERS` parameters.
pub const DETAIL_TOO_MANY_PARAMETERS: &str = "bayesian_transport.too_many_parameters";
/// Detail of a missing target covariate law.
pub const DETAIL_MISSING_LAW: &str = "bayesian_transport.missing_law";
/// Detail of absent source evidence.
pub const DETAIL_MISSING_SOURCE: &str = "bayesian_transport.missing_source";
/// Detail of undeclared, overlapping or unknown source dependence.
pub const DETAIL_SOURCE_DEPENDENCE: &str = "bayesian_transport.source_dependence";
/// Detail of a certificate that does not license the model.
pub const DETAIL_IDENTIFICATION: &str = "bayesian_transport.identification_not_certified";
/// Detail of covariates that differ from the certified standardizers.
pub const DETAIL_FEATURE_MISMATCH: &str = "bayesian_transport.feature_mismatch";
/// Detail of a prior of the wrong dimension.
pub const DETAIL_PRIOR_DIMENSION: &str = "bayesian_transport.prior_dimension";
/// Detail of a prior that is not a symmetric positive definite Gaussian.
pub const DETAIL_INVALID_PRIOR: &str = "bayesian_transport.invalid_prior";
/// Detail of malformed data or options.
pub const DETAIL_INVALID_INPUT: &str = "bayesian_transport.invalid_input";
/// Detail of a cancelled fit.
pub const DETAIL_CANCELLED: &str = "bayesian_transport.cancelled";

/// Graph class a model declaration names.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TransportGraphClass {
    /// The fixed DAG and selection assumptions of the supplied certificate.
    FixedDag,
    /// An ADMG transport query (refused).
    Admg,
    /// A graph posterior (refused).
    GraphPosterior,
}

/// Population-varying coefficient block of the outcome mechanism.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VaryingBlock {
    /// Source-specific intercept; covariate main effects are invariant.
    Intercept,
    /// Source-specific intercept and covariate main effects (none interact with `A`).
    InterceptAndCovariates,
}

/// Declared sharing of the varying block across sources.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SourceSharing {
    /// One independent varying block per source.
    IndependentVaryingBlocks,
    /// One varying block shared by every source (explicit draw sharing).
    SharedVaryingBlock,
}

/// Declared dependence between the source samples.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SourceDependence {
    /// Independent samples with disjoint units: the only supported design.
    IndependentSamples,
    /// Sources share units (refused: the likelihood would double count them).
    OverlappingUnits,
    /// Dependence not declared (refused).
    Unknown,
}

/// Identity of the observations a likelihood or a prior consumed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DataIdentity {
    /// Snapshot digest of the dataset.
    pub snapshot_digest: String,
    /// Stable ids of the individual data or units.
    pub datum_ids: Vec<String>,
}

/// Where a prior came from.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PriorProvenance {
    /// Declared by the analyst without reference to data.
    Declared,
    /// Built from a prior bank; every consumed observation is named.
    Bank {
        /// Bank identity.
        bank_id: String,
        /// Observations the bank construction consumed.
        consumed: Vec<DataIdentity>,
    },
}

/// A dense Gaussian prior block.
#[derive(Clone, Debug, PartialEq)]
pub struct GaussianPrior {
    /// Prior mean, one entry per coefficient.
    pub mean: Vec<f64>,
    /// Row-major symmetric positive definite covariance, `dim * dim` entries.
    pub covariance: Vec<f64>,
    /// Provenance of the prior.
    pub provenance: PriorProvenance,
}

impl GaussianPrior {
    /// Independent prior with one mean and one variance for every coefficient.
    #[must_use]
    pub fn isotropic(
        dim: usize,
        mean: f64,
        variance: f64,
        provenance: PriorProvenance,
    ) -> Self {
        let mut covariance = vec![0.0; dim * dim];
        for i in 0..dim {
            covariance[i * dim + i] = variance;
        }
        Self { mean: vec![mean; dim], covariance, provenance }
    }
}

/// Priors of the invariant block and of every varying block.
#[derive(Clone, Debug, PartialEq)]
pub struct JointPriors {
    /// Prior of the invariant coefficients `theta`.
    pub invariant: GaussianPrior,
    /// Prior of one varying block `gamma` (applied to every block).
    pub varying: GaussianPrior,
}

/// Declaration of the joint source-target model.
#[derive(Clone, Debug, PartialEq)]
pub struct JointTransportModel {
    /// Graph class; only [`TransportGraphClass::FixedDag`] is supported.
    pub graph: TransportGraphClass,
    /// Raw variable ids of the covariates; must equal the certified standardizers.
    pub features: Vec<u32>,
    /// Varying coefficient block.
    pub varying: VaryingBlock,
    /// Sharing of the varying block across sources.
    pub sharing: SourceSharing,
    /// Declared dependence between the sources.
    pub dependence: SourceDependence,
    /// Priors.
    pub priors: JointPriors,
    /// Largest accepted unsupported target mass, in `[0, 1)`.
    pub max_unsupported_mass: f64,
    /// Standardized disagreement above which two sources are flagged, positive.
    pub conflict_z_threshold: f64,
}

/// One source trial (a randomized binary treatment and a continuous outcome).
#[derive(Clone, Debug, PartialEq)]
pub struct SourceData {
    /// Stable, unique source id.
    pub id: String,
    /// Identity of the observations this likelihood consumes.
    pub identity: DataIdentity,
    /// Treatment indicator per unit.
    pub treatment: Vec<bool>,
    /// Outcome per unit.
    pub outcome: Vec<f64>,
    /// Covariate columns (one vector per feature, one entry per unit).
    pub covariates: Vec<Vec<f64>>,
    /// Declared known outcome noise variance, positive.
    pub noise_variance: f64,
}

/// The target covariate sample (no outcomes: nothing enters the likelihood).
#[derive(Clone, Debug, PartialEq)]
pub struct TargetData {
    /// Identity of the target observations.
    pub identity: DataIdentity,
    /// Number of target units.
    pub rows: usize,
    /// Covariate columns (one vector per feature, `rows` entries each).
    pub covariates: Vec<Vec<f64>>,
}

/// Draw request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct JointTransportOptions {
    /// Number of aligned joint draws, `1..=`[`JOINT_TRANSPORT_MAX_DRAWS`].
    pub draws: usize,
    /// Master seed of the draw stream.
    pub seed: u64,
}

/// Diagnostics retained with a refused fit; absent entries were not measured.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FailedFitDiagnostics {
    /// Stage that refused (`validate`, `overlap`, `fit`, ...).
    pub stage: &'static str,
    /// Smallest relative pivot of the failed factorization.
    pub min_pivot_ratio: Option<f64>,
    /// Parameter the failed pivot belongs to.
    pub failing_column: Option<String>,
    /// Target mass outside the source support.
    pub unsupported_mass: Option<f64>,
    /// Declared tolerance of the unsupported mass.
    pub tolerance: Option<f64>,
    /// Snapshot digests or datum ids implicated by an ownership refusal.
    pub overlapping_ids: Vec<String>,
}

/// A typed refusal with a registered reason code and a `bayesian_transport.*` detail.
#[derive(Clone, Debug, PartialEq)]
pub struct JointTransportRefusal {
    /// Registered reason code.
    pub code: &'static str,
    /// Namespaced detail, a `DETAIL_*` constant.
    pub detail: &'static str,
    /// What was refused and what the caller can change.
    pub message: String,
    /// Retained failure diagnostics.
    pub failure: FailedFitDiagnostics,
}

impl JointTransportRefusal {
    fn new(code: &'static str, detail: &'static str, message: impl Into<String>) -> Self {
        Self { code, detail, message: message.into(), failure: FailedFitDiagnostics::default() }
    }

    fn at(mut self, stage: &'static str) -> Self {
        self.failure.stage = stage;
        self
    }
}

impl fmt::Display for JointTransportRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}{}: {}: {}",
            antecedent_core::reason_code::PREFIX,
            self.code,
            self.detail,
            self.message
        )
    }
}

impl std::error::Error for JointTransportRefusal {}

impl From<JointTransportRefusal> for EstimationError {
    fn from(refusal: JointTransportRefusal) -> Self {
        let fields = RefusalFields {
            stage: Some(refusal.failure.stage.to_owned()),
            subject: Some(refusal.detail.to_owned()),
            reason: Some(refusal.message.clone()),
            implicated_columns: refusal.failure.failing_column.into_iter().collect(),
            ..RefusalFields::default()
        };
        Self::RefusedWithFields {
            code: refusal.code,
            message: format!("{}: {}", refusal.detail, refusal.message),
            fields: Box::new(fields),
        }
    }
}

/// Private fallible helpers box the refusal so their `Result` stays small; the public entry
/// point unboxes it.
type Refused = Box<JointTransportRefusal>;

fn invalid(detail: &'static str, message: impl Into<String>) -> JointTransportRefusal {
    JointTransportRefusal::new(antecedent_core::reason_code!("invalid_argument"), detail, message)
        .at("validate")
}

/// The route-level refusal of the public interval route: the native joint posterior route
/// lacks calibration evidence, so no interval is published.
#[must_use]
pub fn route_frozen_refusal() -> JointTransportRefusal {
    JointTransportRefusal::new(
        antecedent_core::reason_code!("cell_not_licensed"),
        DETAIL_ROUTE_FROZEN,
        format!(
            "{JOINT_TRANSPORT_ROUTE} lacks likelihood, value and calibration evidence; \
             the engine's posterior is not a published interval"
        ),
    )
    .at("route")
}

/// The refusal of a graph class other than the fixed DAG.
#[must_use]
pub fn unsupported_graph_refusal() -> JointTransportRefusal {
    JointTransportRefusal::new(
        antecedent_core::reason_code!("route_not_supported"),
        DETAIL_UNSUPPORTED_GRAPH,
        "an ADMG or graph-posterior transport query is not supported by the fixed-DAG model",
    )
    .at("validate")
}

/// Identification read from the checked transport derivation; never changed by a prior.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IdentificationRecord {
    /// Rule id of the certificate.
    pub rule: String,
    /// `identified`: the only status a fit can carry.
    pub status: &'static str,
    /// `direct` or `standardize`.
    pub formula: &'static str,
}

/// Inferential status of a fit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum JointTransportCalibration {
    /// Calibration of credible intervals is unmeasured; no coverage is claimed.
    Unmeasured,
}

/// Target-effect posterior of one source alone, for disagreement diagnostics.
#[derive(Clone, Debug, PartialEq)]
pub struct SourceEffectSummary {
    /// Source id.
    pub source_id: String,
    /// Posterior mean of the target effect from this source alone.
    pub mean: f64,
    /// Posterior variance of the target effect from this source alone.
    pub variance: f64,
}

/// Standardized posterior disagreement of two sources.
#[derive(Clone, Debug, PartialEq)]
pub struct PairDisagreement {
    /// First source id.
    pub left: String,
    /// Second source id.
    pub right: String,
    /// `(mean_left - mean_right) / sqrt(var_left + var_right)`.
    pub z: f64,
}

/// Disagreement between sources; reported, never silently pooled away.
#[derive(Clone, Debug, PartialEq)]
pub struct SourceDisagreement {
    /// Per-source target-effect posteriors (empty for a single source).
    pub per_source: Vec<SourceEffectSummary>,
    /// Pairwise standardized differences.
    pub pairs: Vec<PairDisagreement>,
    /// Largest absolute standardized difference (zero for a single source).
    pub max_abs_z: f64,
    /// Whether `max_abs_z` exceeds the declared threshold.
    pub flagged: bool,
}

/// Overlap measured against the source support boxes.
#[derive(Clone, Debug, PartialEq)]
pub struct OverlapDiagnostic {
    /// Target mass outside every source box.
    pub unsupported_mass: f64,
    /// Declared tolerance.
    pub tolerance: f64,
}

/// Diagnostics of a successful fit.
#[derive(Clone, Debug, PartialEq)]
pub struct JointTransportDiagnostics {
    /// [`JOINT_TRANSPORT_SAMPLER`].
    pub sampler: &'static str,
    /// Number of draws.
    pub draw_count: usize,
    /// Effective sample size (equal to `draw_count` for exact iid draws).
    pub effective_sample_size: f64,
    /// Split R-hat of a future chain-based path; absent for exact iid draws.
    pub r_hat: Option<f64>,
    /// Smallest relative pivot of the data information matrix.
    pub min_pivot_ratio: f64,
    /// Target overlap.
    pub overlap: OverlapDiagnostic,
    /// Source disagreement.
    pub disagreement: SourceDisagreement,
    /// Always [`JointTransportCalibration::Unmeasured`].
    pub calibration: JointTransportCalibration,
}

/// Aligned joint draws: row `i` is one realization of every coordinate.
#[derive(Clone, Debug, PartialEq)]
pub struct JointDraws {
    /// Coordinate names in column order: parameters, source effects, target effect.
    pub names: Vec<String>,
    /// Draw-major values, `n_draws * names.len()`.
    pub values: Vec<f64>,
    /// Number of draws.
    pub n_draws: usize,
    /// RNG algorithm, seed and stream identity.
    pub rng_id: String,
}

impl JointDraws {
    /// Number of coordinates per draw.
    #[must_use]
    pub fn width(&self) -> usize {
        self.names.len()
    }

    /// Column of the named coordinate.
    #[must_use]
    pub fn column(&self, name: &str) -> Option<usize> {
        self.names.iter().position(|candidate| candidate == name)
    }

    /// One coordinate across draws.
    #[must_use]
    pub fn coordinate(&self, column: usize) -> Vec<f64> {
        let width = self.width();
        (0..self.n_draws).map(|draw| self.values[draw * width + column]).collect()
    }
}

/// Exact conjugate fit of the joint model.
#[derive(Clone, Debug, PartialEq)]
pub struct JointTransportFit {
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
    pub diagnostics: JointTransportDiagnostics,
    /// Identification of the checked derivation.
    pub identification: IdentificationRecord,
    /// Canonical identity of the model, priors and bounds.
    pub model_identity: String,
}

impl JointTransportFit {
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

/// Number of covariates and sizes of the parameter blocks.
#[derive(Clone, Copy)]
struct Layout {
    p: usize,
    q: usize,
    r: usize,
    blocks: usize,
    sharing: SourceSharing,
    varying: VaryingBlock,
}

impl Layout {
    fn new(p: usize, model: &JointTransportModel, sources: usize) -> Self {
        let (q, r) = match model.varying {
            VaryingBlock::Intercept => (1 + 2 * p, 1),
            VaryingBlock::InterceptAndCovariates => (1 + p, 1 + p),
        };
        let blocks = match model.sharing {
            SourceSharing::IndependentVaryingBlocks => sources,
            SourceSharing::SharedVaryingBlock => 1,
        };
        Self { p, q, r, blocks, sharing: model.sharing, varying: model.varying }
    }

    const fn dim(&self) -> usize {
        self.q + self.blocks * self.r
    }

    /// Block of the source at `position` among the sources of the fit.
    const fn block(&self, position: usize) -> usize {
        match self.sharing {
            SourceSharing::IndependentVaryingBlocks => position,
            SourceSharing::SharedVaryingBlock => 0,
        }
    }

    /// Global parameter index of local row feature `k` (`0..q+r`) in `block`.
    const fn index(&self, k: usize, block: usize) -> usize {
        if k < self.q { k } else { self.q + block * self.r + (k - self.q) }
    }

    /// Row features `[w, v]` of one unit.
    fn row(&self, treated: bool, x: &[f64], out: &mut Vec<f64>) {
        out.clear();
        let a = if treated { 1.0 } else { 0.0 };
        out.push(a);
        out.extend(x.iter().map(|value| a * value));
        if self.varying == VaryingBlock::Intercept {
            out.extend_from_slice(x);
        }
        out.push(1.0);
        if self.varying == VaryingBlock::InterceptAndCovariates {
            out.extend_from_slice(x);
        }
    }

    /// Effect vector `[1, xbar, 0..]` over the full parameter space.
    fn effect_vector(&self, mean_x: &[f64]) -> Vec<f64> {
        let mut c = vec![0.0; self.dim()];
        c[0] = 1.0;
        c[1..=self.p].copy_from_slice(mean_x);
        c
    }
}

fn feature_name(model: &JointTransportModel, j: usize) -> String {
    format!("x{}", model.features[j])
}

fn parameter_names(layout: &Layout, model: &JointTransportModel, ids: &[&str]) -> Vec<String> {
    let mut names = vec!["theta.treatment".to_owned()];
    for j in 0..layout.p {
        names.push(format!("theta.treatment:{}", feature_name(model, j)));
    }
    if layout.varying == VaryingBlock::Intercept {
        for j in 0..layout.p {
            names.push(format!("theta.{}", feature_name(model, j)));
        }
    }
    let owners: Vec<&str> = match layout.sharing {
        SourceSharing::IndependentVaryingBlocks => ids[..layout.blocks].to_vec(),
        SourceSharing::SharedVaryingBlock => vec!["shared"; layout.blocks],
    };
    for owner in owners {
        names.push(format!("gamma.{owner}.intercept"));
        if layout.varying == VaryingBlock::InterceptAndCovariates {
            for j in 0..layout.p {
                names.push(format!("gamma.{owner}.{}", feature_name(model, j)));
            }
        }
    }
    names
}

/// Lower Cholesky factor, or the failing index and its relative pivot.
fn cholesky(a: &[f64], n: usize) -> Result<Vec<f64>, (usize, f64)> {
    let mut l = vec![0.0; n * n];
    for j in 0..n {
        let mut d = a[j * n + j];
        for k in 0..j {
            d -= l[j * n + k] * l[j * n + k];
        }
        let scale = a[j * n + j].abs().max(f64::MIN_POSITIVE);
        let ratio = d / scale;
        if ratio.is_nan() || ratio <= JOINT_TRANSPORT_PIVOT_TOLERANCE {
            return Err((j, ratio));
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
    Ok(l)
}

/// Solve `L y = b`.
fn forward_solve(l: &[f64], n: usize, b: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i * n + k] * y[k];
        }
        y[i] = s / l[i * n + i];
    }
    y
}

/// Solve `L' x = y`.
fn back_solve(l: &[f64], n: usize, y: &[f64]) -> Vec<f64> {
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = y[i];
        for k in (i + 1)..n {
            s -= l[k * n + i] * x[k];
        }
        x[i] = s / l[i * n + i];
    }
    x
}

fn chol_solve(l: &[f64], n: usize, b: &[f64]) -> Vec<f64> {
    back_solve(l, n, &forward_solve(l, n, b))
}

fn chol_inverse(l: &[f64], n: usize) -> Vec<f64> {
    let mut inverse = vec![0.0; n * n];
    let mut unit = vec![0.0; n];
    for j in 0..n {
        unit.fill(0.0);
        unit[j] = 1.0;
        let column = chol_solve(l, n, &unit);
        for i in 0..n {
            inverse[i * n + j] = column[i];
        }
    }
    inverse
}

fn mat_vec(a: &[f64], n: usize, x: &[f64]) -> Vec<f64> {
    (0..n).map(|i| (0..n).map(|j| a[i * n + j] * x[j]).sum()).collect()
}

fn quadratic(a: &[f64], n: usize, left: &[f64], right: &[f64]) -> f64 {
    let image = mat_vec(a, n, right);
    left.iter().zip(&image).map(|(l, r)| l * r).sum()
}

fn column_means(columns: &[Vec<f64>], rows: usize) -> Vec<f64> {
    columns.iter().map(|column| column.iter().sum::<f64>() / rows as f64).collect()
}

/// Validated prior precision of one block.
struct PriorBlock {
    mean: Vec<f64>,
    precision: Vec<f64>,
}

fn prior_block(prior: &GaussianPrior, dim: usize) -> Result<PriorBlock, Refused> {
    if prior.mean.len() != dim || prior.covariance.len() != dim * dim {
        return Err(JointTransportRefusal::new(
            antecedent_core::reason_code!("prior_dimension_mismatch"),
            DETAIL_PRIOR_DIMENSION,
            format!("prior block needs {dim} coefficients and a {dim}x{dim} covariance"),
        )
        .at("validate")
        .into());
    }
    let scale = prior.covariance.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let mut symmetric = true;
    for i in 0..dim {
        for j in 0..i {
            let gap = (prior.covariance[i * dim + j] - prior.covariance[j * dim + i]).abs();
            symmetric &= gap <= 1e-12 * scale.max(1.0);
        }
    }
    if !symmetric
        || prior.mean.iter().chain(&prior.covariance).any(|value| !value.is_finite())
    {
        return Err(invalid(DETAIL_INVALID_PRIOR, "prior covariance must be finite and symmetric")
            .into());
    }
    let factor = cholesky(&prior.covariance, dim).map_err(|(index, ratio)| {
        let mut refusal =
            invalid(DETAIL_INVALID_PRIOR, "prior covariance must be positive definite");
        refusal.failure.min_pivot_ratio = Some(ratio);
        refusal.failure.failing_column = Some(format!("prior[{index}]"));
        refusal
    })?;
    Ok(PriorBlock { mean: prior.mean.clone(), precision: chol_inverse(&factor, dim) })
}

/// Exact Gaussian posterior with its precision factor.
struct Posterior {
    mean: Vec<f64>,
    covariance: Vec<f64>,
    precision_factor: Vec<f64>,
    min_pivot_ratio: f64,
}

fn singular(
    names: &[String],
    index: usize,
    ratio: f64,
    what: &str,
) -> JointTransportRefusal {
    let mut refusal = JointTransportRefusal::new(
        antecedent_core::reason_code!("design_rank_deficient"),
        DETAIL_SINGULAR_FIT,
        format!("{what} is singular at parameter {}", names[index]),
    )
    .at("fit");
    refusal.failure.min_pivot_ratio = Some(ratio);
    refusal.failure.failing_column = Some(names[index].clone());
    refusal
}

/// Smallest relative pivot of a factorization that succeeded (a diagnostic only).
fn smallest_pivot_ratio(a: &[f64], factor: &[f64], n: usize) -> f64 {
    (0..n)
        .map(|j| factor[j * n + j] * factor[j * n + j] / a[j * n + j].abs().max(f64::MIN_POSITIVE))
        .fold(f64::INFINITY, f64::min)
}

struct Assembly<'a> {
    layout: &'a Layout,
    theta: &'a PriorBlock,
    gamma: &'a PriorBlock,
    names: &'a [String],
}

fn assemble_posterior(
    asm: &Assembly<'_>,
    sources: &[&SourceData],
    check_data_information: bool,
) -> Result<Posterior, Refused> {
    let layout = asm.layout;
    let dim = layout.dim();
    let mut info = vec![0.0; dim * dim];
    let mut moment = vec![0.0; dim];
    let mut row = Vec::new();
    let mut x = vec![0.0; layout.p];
    for (position, source) in sources.iter().enumerate() {
        let block = layout.block(position);
        let weight = 1.0 / source.noise_variance;
        for i in 0..source.outcome.len() {
            for (j, column) in source.covariates.iter().enumerate() {
                x[j] = column[i];
            }
            layout.row(source.treatment[i], &x, &mut row);
            for (k, zk) in row.iter().enumerate() {
                let gk = layout.index(k, block);
                moment[gk] += weight * zk * source.outcome[i];
                for (m, zm) in row.iter().enumerate() {
                    info[gk * dim + layout.index(m, block)] += weight * zk * zm;
                }
            }
        }
    }
    let data_pivot = match cholesky(&info, dim) {
        Ok(factor) => smallest_pivot_ratio(&info, &factor, dim),
        Err((index, ratio)) if check_data_information => {
            return Err(singular(asm.names, index, ratio, "the data information matrix").into());
        }
        Err((_, ratio)) => ratio,
    };
    let mut precision = info;
    let mut prior_mean = vec![0.0; dim];
    let q = layout.q;
    prior_mean[..q].copy_from_slice(&asm.theta.mean);
    for i in 0..q {
        for j in 0..q {
            precision[i * dim + j] += asm.theta.precision[i * q + j];
        }
    }
    for block in 0..layout.blocks {
        let offset = q + block * layout.r;
        prior_mean[offset..offset + layout.r].copy_from_slice(&asm.gamma.mean);
        for i in 0..layout.r {
            for j in 0..layout.r {
                precision[(offset + i) * dim + offset + j] += asm.gamma.precision[i * layout.r + j];
            }
        }
    }
    let prior_part = prior_precision_times_mean(asm, &prior_mean);
    for (m, p) in moment.iter_mut().zip(prior_part) {
        *m += p;
    }
    let factor = cholesky(&precision, dim)
        .map_err(|(index, ratio)| singular(asm.names, index, ratio, "the posterior precision"))?;
    let mean = chol_solve(&factor, dim, &moment);
    let covariance = chol_inverse(&factor, dim);
    Ok(Posterior { mean, covariance, precision_factor: factor, min_pivot_ratio: data_pivot })
}

/// `P0 m0` block by block, from the prior precisions.
fn prior_precision_times_mean(asm: &Assembly<'_>, prior_mean: &[f64]) -> Vec<f64> {
    let layout = asm.layout;
    let mut out = vec![0.0; layout.dim()];
    let head = mat_vec(&asm.theta.precision, layout.q, &prior_mean[..layout.q]);
    out[..layout.q].copy_from_slice(&head);
    for block in 0..layout.blocks {
        let offset = layout.q + block * layout.r;
        let part = mat_vec(&asm.gamma.precision, layout.r, &prior_mean[offset..offset + layout.r]);
        out[offset..offset + layout.r].copy_from_slice(&part);
    }
    out
}

fn ids_overlap(left: &[String], right: &[String]) -> Vec<String> {
    let set: BTreeSet<&String> = left.iter().collect();
    right.iter().filter(|id| set.contains(id)).cloned().collect()
}

fn validate_declaration(
    id: &TransportIdentification,
    model: &JointTransportModel,
    options: &JointTransportOptions,
) -> Result<IdentificationRecord, Refused> {
    if model.graph != TransportGraphClass::FixedDag {
        return Err(unsupported_graph_refusal().into());
    }
    let (record, over): (IdentificationRecord, Vec<u32>) = match id {
        TransportIdentification::Transportable {
            formula: TransportFormula::Direct(_),
            certificate,
        } => (
            IdentificationRecord {
                rule: certificate.rule.to_string(),
                status: "identified",
                formula: "direct",
            },
            Vec::new(),
        ),
        TransportIdentification::Transportable {
            formula: TransportFormula::Standardize { over, .. },
            certificate,
        } => (
            IdentificationRecord {
                rule: certificate.rule.to_string(),
                status: "identified",
                formula: "standardize",
            },
            over.iter().map(|v| v.raw()).collect(),
        ),
        TransportIdentification::MissingEvidence(_) => {
            return Err(JointTransportRefusal::new(
                antecedent_core::reason_code!("transport_missing_evidence"),
                DETAIL_IDENTIFICATION,
                "the transport derivation lacks a required source regime",
            )
            .at("identify")
            .into());
        }
        _ => {
            return Err(JointTransportRefusal::new(
                antecedent_core::reason_code!("transport_not_certified"),
                DETAIL_IDENTIFICATION,
                "a direct or standardization transport certificate is required; a prior \
                 cannot supply identification",
            )
            .at("identify")
            .into());
        }
    };
    let mut expected = over;
    expected.sort_unstable();
    let mut actual = model.features.clone();
    actual.sort_unstable();
    if expected != actual {
        return Err(JointTransportRefusal::new(
            antecedent_core::reason_code!("invalid_argument"),
            DETAIL_FEATURE_MISMATCH,
            "model covariates must equal the certified standardizers",
        )
        .at("validate")
        .into());
    }
    if options.draws == 0 || options.draws > JOINT_TRANSPORT_MAX_DRAWS {
        return Err(invalid(
            DETAIL_TOO_MANY_DRAWS,
            format!("draw count must be in 1..={JOINT_TRANSPORT_MAX_DRAWS}"),
        )
        .into());
    }
    if model.dependence != SourceDependence::IndependentSamples {
        return Err(JointTransportRefusal::new(
            antecedent_core::reason_code!("sampling_dependence_unknown"),
            DETAIL_SOURCE_DEPENDENCE,
            "only independent source samples with disjoint units are supported",
        )
        .at("validate")
        .into());
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
    Ok(record)
}

fn validate_sources(
    model: &JointTransportModel,
    sources: &[SourceData],
) -> Result<(), Refused> {
    if sources.is_empty() {
        return Err(JointTransportRefusal::new(
            antecedent_core::reason_code!("transport_missing_evidence"),
            DETAIL_MISSING_SOURCE,
            "at least one source trial is required",
        )
        .at("validate")
        .into());
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
                format!("source `{}` is malformed (shape, finiteness, noise or identity)", source.id),
            )
            .into());
        }
    }
    Ok(())
}

fn validate_target<'a>(
    model: &JointTransportModel,
    target: Option<&'a TargetData>,
) -> Result<&'a TargetData, Refused> {
    let missing = || {
        JointTransportRefusal::new(
            antecedent_core::reason_code!("joint_law_required"),
            DETAIL_MISSING_LAW,
            "the target covariate law (a nonempty target sample) is required",
        )
        .at("validate")
    };
    let target = target.ok_or_else(missing)?;
    if target.rows == 0 {
        return Err(missing().into());
    }
    let finite = target
        .covariates
        .iter()
        .all(|c| c.len() == target.rows && c.iter().all(|v| v.is_finite()));
    if target.covariates.len() != model.features.len()
        || !finite
        || target.identity.snapshot_digest.trim().is_empty()
    {
        return Err(invalid(DETAIL_INVALID_INPUT, "target covariates are malformed").into());
    }
    Ok(target)
}

/// Unit ownership: sources are disjoint from each other and from the target.
fn check_unit_ownership(
    sources: &[SourceData],
    target: &TargetData,
) -> Result<(), Refused> {
    let mut owners: Vec<&[String]> =
        sources.iter().map(|source| source.identity.datum_ids.as_slice()).collect();
    owners.push(&target.identity.datum_ids);
    for i in 0..owners.len() {
        for j in (i + 1)..owners.len() {
            let shared = ids_overlap(owners[i], owners[j]);
            if !shared.is_empty() {
                let mut refusal = JointTransportRefusal::new(
                    antecedent_core::reason_code!("sampling_dependence_unknown"),
                    DETAIL_SOURCE_DEPENDENCE,
                    "sources and target must own disjoint units; shared units would be \
                     counted twice",
                )
                .at("validate");
                refusal.failure.overlapping_ids = shared;
                return Err(refusal.into());
            }
        }
    }
    Ok(())
}

/// A datum or snapshot may enter exactly one of the likelihood and a prior bank.
///
/// # Errors
/// `invalid_argument` / [`DETAIL_PRIOR_DATA_OVERLAP`] naming the shared snapshot or ids.
// The public refusal type is deliberately rich; no caller or test depends on boxing it.
#[allow(clippy::result_large_err)]
pub fn check_prior_data_overlap(
    priors: &JointPriors,
    sources: &[SourceData],
) -> Result<(), JointTransportRefusal> {
    for prior in [&priors.invariant, &priors.varying] {
        let PriorProvenance::Bank { consumed, .. } = &prior.provenance else {
            continue;
        };
        let mut shared = Vec::new();
        for used in consumed {
            for source in sources {
                if used.snapshot_digest == source.identity.snapshot_digest {
                    shared.push(used.snapshot_digest.clone());
                }
                shared.extend(ids_overlap(&used.datum_ids, &source.identity.datum_ids));
            }
        }
        if !shared.is_empty() {
            shared.sort();
            shared.dedup();
            let mut refusal = JointTransportRefusal::new(
                antecedent_core::reason_code!("invalid_argument"),
                DETAIL_PRIOR_DATA_OVERLAP,
                "observations used to build the prior bank also enter the likelihood; each \
                 datum may enter exactly one",
            )
            .at("validate");
            refusal.failure.overlapping_ids = shared;
            return Err(refusal);
        }
    }
    Ok(())
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

/// Box-Muller normals over a `SplitMix64` stream.
struct NormalStream {
    state: u64,
    spare: Option<f64>,
}

impl NormalStream {
    const fn new(seed: u64) -> Self {
        Self { state: seed_mix(seed), spare: None }
    }

    fn unit(&mut self) -> f64 {
        (splitmix64(&mut self.state) >> 11) as f64 / (1_u64 << 53) as f64
    }

    fn next(&mut self) -> f64 {
        if let Some(value) = self.spare.take() {
            return value;
        }
        let u1 = 1.0 - self.unit();
        let u2 = self.unit();
        let radius = (-2.0 * u1.ln()).sqrt();
        let angle = std::f64::consts::TAU * u2;
        self.spare = Some(radius * angle.sin());
        radius * angle.cos()
    }
}

fn disagreement(
    asm: &Assembly<'_>,
    model: &JointTransportModel,
    sources: &[SourceData],
    target_effect: &[f64],
) -> Result<SourceDisagreement, Refused> {
    if sources.len() < 2 {
        return Ok(SourceDisagreement {
            per_source: Vec::new(),
            pairs: Vec::new(),
            max_abs_z: 0.0,
            flagged: false,
        });
    }
    let alone_layout = Layout { blocks: 1, ..*asm.layout };
    let mut per_source = Vec::with_capacity(sources.len());
    for source in sources {
        let names = parameter_names(&alone_layout, model, &[source.id.as_str()]);
        let alone = Assembly { layout: &alone_layout, names: &names, ..*asm };
        let posterior = assemble_posterior(&alone, &[source], false)?;
        let c = &target_effect[..alone_layout.dim()];
        per_source.push(SourceEffectSummary {
            source_id: source.id.clone(),
            mean: c.iter().zip(&posterior.mean).map(|(a, b)| a * b).sum(),
            variance: quadratic(&posterior.covariance, alone_layout.dim(), c, c),
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

fn model_identity(model: &JointTransportModel, layout: &Layout, sources: &[SourceData]) -> String {
    let prior = |p: &GaussianPrior| {
        let bits: Vec<String> =
            p.mean.iter().chain(&p.covariance).map(|v| format!("{:016x}", v.to_bits())).collect();
        bits.join(",")
    };
    let ids: Vec<&str> = sources.iter().map(|s| s.id.as_str()).collect();
    format!(
        "joint_bayesian_transport_v1|graph=fixed_dag|features={:?}|varying={:?}|sharing={:?}|\
         sources={}|noise={:?}|theta_prior={}|gamma_prior={}|dim={}|max_draws={}|max_parameters={}",
        model.features,
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

fn effect_set(layout: &Layout, sources: &[SourceData], target: &TargetData) -> EffectSet {
    let mut names = Vec::with_capacity(sources.len() + 1);
    let mut vectors = Vec::with_capacity(sources.len() + 1);
    for source in sources {
        names.push(format!("effect.source.{}", source.id));
        vectors.push(layout.effect_vector(&column_means(&source.covariates, source.outcome.len())));
    }
    names.push("effect.target".to_owned());
    vectors.push(layout.effect_vector(&column_means(&target.covariates, target.rows)));
    EffectSet { names, vectors }
}

fn draw_joint(
    posterior: &Posterior,
    effects: &EffectSet,
    parameter_names: &[String],
    options: &JointTransportOptions,
    ctx: &ExecutionContext,
) -> Result<JointDraws, Refused> {
    let dim = posterior.mean.len();
    let width = dim + effects.vectors.len();
    let mut stream = NormalStream::new(options.seed);
    let mut values = Vec::with_capacity(options.draws * width);
    let mut z = vec![0.0; dim];
    for draw in 0..options.draws {
        if draw % 1024 == 0 && ctx.cancellation.is_cancelled() {
            return Err(JointTransportRefusal::new(
                antecedent_core::reason_code!("transport_budget_cancel"),
                DETAIL_CANCELLED,
                "joint transport draws were cancelled",
            )
            .at("draw")
            .into());
        }
        for value in &mut z {
            *value = stream.next();
        }
        let shift = back_solve(&posterior.precision_factor, dim, &z);
        let parameters: Vec<f64> =
            posterior.mean.iter().zip(&shift).map(|(m, s)| m + s).collect();
        values.extend_from_slice(&parameters);
        for c in &effects.vectors {
            values.push(c.iter().zip(&parameters).map(|(a, b)| a * b).sum());
        }
    }
    let mut names = parameter_names.to_vec();
    names.extend(effects.names.iter().cloned());
    Ok(JointDraws {
        names,
        values,
        n_draws: options.draws,
        rng_id: format!("splitmix64_box_muller_v1:seed={}:stream=0", options.seed),
    })
}

/// Fit the joint source-target model exactly and draw aligned joint posterior samples.
///
/// Identification is read from `id` and copied to the result; no prior can change it.
///
/// # Errors
/// A [`JointTransportRefusal`] (convertible to [`EstimationError`]): unsupported graph,
/// uncertified derivation, mismatched covariates, missing law or source, undeclared source
/// dependence, prior/likelihood double use, prior or draw bounds, weak target overlap,
/// singular data information, or cancellation.
// The public refusal type is deliberately rich and part of the stable signature; the private
// pipeline boxes it and this boundary unboxes it.
#[allow(clippy::result_large_err)]
pub fn fit_joint_bayesian_transport(
    id: &TransportIdentification,
    model: &JointTransportModel,
    sources: &[SourceData],
    target: Option<&TargetData>,
    options: &JointTransportOptions,
    ctx: &ExecutionContext,
) -> Result<JointTransportFit, JointTransportRefusal> {
    fit_inner(id, model, sources, target, options, ctx).map_err(|refusal| *refusal)
}

fn fit_inner(
    id: &TransportIdentification,
    model: &JointTransportModel,
    sources: &[SourceData],
    target: Option<&TargetData>,
    options: &JointTransportOptions,
    ctx: &ExecutionContext,
) -> Result<JointTransportFit, Refused> {
    let identification = validate_declaration(id, model, options)?;
    validate_sources(model, sources)?;
    let target = validate_target(model, target)?;
    let layout = Layout::new(model.features.len(), model, sources.len());
    if layout.dim() > JOINT_TRANSPORT_MAX_PARAMETERS {
        return Err(invalid(
            DETAIL_TOO_MANY_PARAMETERS,
            format!("the model has {} parameters; the bound is {JOINT_TRANSPORT_MAX_PARAMETERS}", layout.dim()),
        )
        .into());
    }
    let theta = prior_block(&model.priors.invariant, layout.q)?;
    let gamma = prior_block(&model.priors.varying, layout.r)?;
    check_unit_ownership(sources, target)?;
    check_prior_data_overlap(&model.priors, sources)?;
    let mass = unsupported_mass(sources, target);
    if mass > model.max_unsupported_mass {
        let mut refusal = JointTransportRefusal::new(
            antecedent_core::reason_code!("transport_support_failure"),
            DETAIL_WEAK_OVERLAP,
            "target covariate mass outside the source support exceeds the declared tolerance",
        )
        .at("overlap");
        refusal.failure.unsupported_mass = Some(mass);
        refusal.failure.tolerance = Some(model.max_unsupported_mass);
        return Err(refusal.into());
    }
    let ids: Vec<&str> = sources.iter().map(|s| s.id.as_str()).collect();
    let names = parameter_names(&layout, model, &ids);
    let asm = Assembly { layout: &layout, theta: &theta, gamma: &gamma, names: &names };
    let refs: Vec<&SourceData> = sources.iter().collect();
    let posterior = assemble_posterior(&asm, &refs, true)?;
    let effects = effect_set(&layout, sources, target);
    let target_vector = effects.vectors[effects.vectors.len() - 1].clone();
    let disagreement = disagreement(&asm, model, sources, &target_vector)?;
    let draws = draw_joint(&posterior, &effects, &names, options, ctx)?;
    let dim = layout.dim();
    let count = effects.vectors.len();
    let effect_means: Vec<f64> = effects
        .vectors
        .iter()
        .map(|c| c.iter().zip(&posterior.mean).map(|(a, b)| a * b).sum())
        .collect();
    let mut effect_covariance = vec![0.0; count * count];
    for i in 0..count {
        for j in 0..count {
            effect_covariance[i * count + j] =
                quadratic(&posterior.covariance, dim, &effects.vectors[i], &effects.vectors[j]);
        }
    }
    Ok(JointTransportFit {
        target_effect_mean: effect_means[count - 1],
        target_effect_variance: effect_covariance[count * count - 1],
        model_identity: model_identity(model, &layout, sources),
        diagnostics: JointTransportDiagnostics {
            sampler: JOINT_TRANSPORT_SAMPLER,
            draw_count: options.draws,
            effective_sample_size: options.draws as f64,
            r_hat: None,
            min_pivot_ratio: posterior.min_pivot_ratio,
            overlap: OverlapDiagnostic {
                unsupported_mass: mass,
                tolerance: model.max_unsupported_mass,
            },
            disagreement,
            calibration: JointTransportCalibration::Unmeasured,
        },
        parameter_names: names,
        posterior_mean: posterior.mean,
        posterior_covariance: posterior.covariance,
        effect_names: effects.names,
        effect_means,
        effect_covariance,
        draws,
        identification,
    })
}
