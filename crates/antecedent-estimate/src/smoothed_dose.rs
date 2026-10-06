//! Smoothed dose-response transport grid (2.2B cell X4).
//!
//! # The one cell
//!
//! A randomized continuous source dose `A` drawn from a **known** conditional density
//! `pi(a | x)` on a declared bounded support, a continuous outcome `Y`, complete baseline
//! covariates `X` equal to the certified standardizers, and an overlap-supported target
//! population. For each declared grid dose `a`, one declared bandwidth `h` and the
//! Epanechnikov kernel `K`, the estimand is
//!
//! ```text
//! psi_h(a) = E_target[ nu_h(X; a) ],   nu_h(x; a) = integral K_h(a - t) mu(t, x) dt,
//! mu(t, x) = E(Y | X = x, A = t, S = 1),   K_h(u) = K(u / h) / h,
//! ```
//!
//! read through a direct or baseline-standardization certificate (the `prepare_trial`
//! graph contract) with the dose as the single source experiment. The bandwidth and the
//! kernel are part of the estimand: `psi_h` is a different target for each `h`. Every
//! grid window `[a - h, a + h]` must lie inside the declared dose support (no boundary
//! kernels); extrapolative points are refused. The two IID designs of the learned-trial
//! cells ([`TrialSampling::NestedCohort`], [`TrialSampling::IndependentSamples`]) are
//! the only sampling semantics.
//!
//! # Estimator
//!
//! With `n_0` target rows, out-of-fold membership `p(X) = P(S=1 | X)`, odds
//! `w(X) = (1 - p) / p` and out-of-fold `mu`:
//!
//! ```text
//! psi_hat(a) = (1/n_0) [ sum_{S=0} nu_hat(X_i; a)
//!                      + sum_{S=1} w(X_i) K_h(a - A_i) / pi(A_i | X_i) (Y_i - mu_hat(A_i, X_i)) ]
//! ```
//!
//! `nu_hat` is the Gauss-Legendre quadrature of the fitted curve over the window. The
//! derivation of the score, its model double robustness and the numerical checks that
//! verify it are in `docs/architecture/smoothed-dose-response-transport.md`.
//!
//! The only learned nuisances are `mu(t, x)` (a Regression learner over a deterministic
//! row-wise dose-by-covariate basis) and source membership (a `BinaryProbability` learner
//! over `[1, x]`), both from `antecedent-learn` `LearnerSpec`s and cross-fitted on one
//! shared stratified fold assignment. The basis is a row-wise transform with no fitted
//! statistic, so preprocessing inside folds holds trivially. Every fold model is exported
//! as a provider-independent [`PortablePredictor`] and every prediction - the point's and
//! a consumer's replay - is made through it, which makes the point replayable bit for bit
//! from the stored models. A learner without a portable map is refused.
//!
//! # Numerical and smoothing error, kept apart
//!
//! Every integration window is split at the declared basis knots inside it, and each
//! piece gets its own Gauss-Legendre rule. A linear-family fitted curve on the dose basis
//! is then a polynomial of degree at most 3 on every piece, so both rules integrate it
//! exactly up to rounding (`QuadratureRecord::exact`) and no tolerance is gated. For any
//! other fitted curve (a tree learner) the `Q`-node and `2Q`-node rules are compared on
//! every target row and the grid dose is refused when the largest row difference exceeds
//! the declared tolerance; that difference is an error *estimate*, not a bound (a knot
//! sweep in this module's tests shows the unsplit estimate missing the finer rule's true
//! error). The reported point uses the `2Q` rule. The smoothing-bias diagnostic is the
//! plug-in difference `psi_hat_h - psi_hat_{h/2}` with its local-quadratic extrapolation
//! `4/3` of it; it sees only the fitted curve, so it is exactly zero for a fit linear in
//! the dose whatever the true curvature. Neither record is ever added to the estimate or
//! to an interval. The analytic influence-function standard error is a diagnostic only.
//!
//! # Claimed theorem
//!
//! Model double robustness of the point (outcome regression or participation model
//! correct, `pi` known). No efficiency, rate, normality or bandwidth-selection claim.
//!
//! # Inference
//!
//! One interval method: the pointwise joint outer refit percentile bootstrap of the whole
//! cross-fitted composed estimator (`smoothed_dose_interval_internal`, compiled only under
//! the `calibration-internal` feature). The public route is closed (`cell_not_licensed`)
//! until its coverage records are measured.
//!
//! # Bounded computation
//!
//! No search: the quadrature is a bounded loop over at most 16 grid doses, two node
//! counts and chunks of target rows. Cancellation is polled on every chunk. A mandatory
//! workspace cap ([`SMOOTHED_DOSE_MAX_WORKSPACE_BYTES`], lowered by a context hard memory
//! limit) refuses before any fit when the cumulative estimate of the rows, the designs
//! actually allocated, one training-design copy and normal matrix per concurrently fitted
//! fold and the concurrent quadrature chunks exceeds it. Learner internals beyond that
//! copy (tree storage, boosting state) are not modelled.
use crate::EstimationError;
use crate::estimator_menu::{EstimatorMenu, EstimatorMenuEntry, MenuRefusal};
use crate::learned_continuous::FoldProvenance;
use crate::learned_trial::TrialSampling;
use antecedent_core::{ExecutionContext, SmoothedDoseTransportQuery, SmoothingKernel, VariableId};
use antecedent_identify::{TransportFormula, TransportIdentification};
use antecedent_learn::{
    DesignView, FittedPredictor, LearnerProvenance, LearnerSpec, PortablePredictor, PredictionMap,
    PredictionTask, RowSelection, TargetView, resolve_for,
};
use serde::{Deserialize, Serialize};

/// Most grid doses one request may declare.
pub const SMOOTHED_DOSE_MAX_GRID: usize = 16;
/// The accepted Gauss-Legendre node counts; each is checked against twice as many.
pub const SMOOTHED_DOSE_QUADRATURE_NODES: [usize; 2] = [16, 32];
/// Most cross-fit folds one request may ask for.
pub const SMOOTHED_DOSE_MAX_FOLDS: usize = 20;
/// Fewest bootstrap replicates of the calibrated interval; below it the interval is withheld.
pub const SMOOTHED_DOSE_MIN_BOOTSTRAP: u32 = 199;
/// Most bootstrap replicates one request may ask for.
pub const SMOOTHED_DOSE_MAX_BOOTSTRAP: u32 = 2000;
/// Most rows (trial plus target) one request may carry.
pub const SMOOTHED_DOSE_MAX_ROWS: usize = 200_000;
/// Most baseline covariates one request may carry.
pub const SMOOTHED_DOSE_MAX_FEATURES: usize = 256;
/// Highest polynomial degree of the dose basis.
pub const SMOOTHED_DOSE_MAX_BASIS_DEGREE: u8 = 3;
/// Most hinge knots of the dose basis.
pub const SMOOTHED_DOSE_MAX_KNOTS: usize = 8;
/// The fold scheme recorded on every estimate.
pub const SMOOTHED_DOSE_FOLD_SCHEME: &str = "stratified_round_robin_source_and_target";
/// The one supported target of a request.
pub const SMOOTHED_DOSE_TARGET: &str = "smoothed_dose_response";
/// Most design rows per quadrature chunk; cancellation is polled once per chunk.
const CHUNK_DESIGN_ROWS: usize = 1 << 14;
/// Most design bytes per quadrature chunk; wide designs get fewer rows per chunk.
const CHUNK_DESIGN_BYTES: usize = 8 << 20;
/// Mandatory cap on the estimated workspace of one request or one replay (512 MiB, like
/// the default search budget); a context hard memory limit lowers it, never raises it.
pub const SMOOTHED_DOSE_MAX_WORKSPACE_BYTES: u64 = 536_870_912;

/// The frozen bounds an artifact's premises bind.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SmoothedDoseBounds {
    /// [`SMOOTHED_DOSE_MAX_GRID`].
    pub max_grid: usize,
    /// [`SMOOTHED_DOSE_QUADRATURE_NODES`].
    pub quadrature_nodes: [usize; 2],
    /// [`SMOOTHED_DOSE_MAX_FOLDS`].
    pub max_folds: usize,
    /// [`SMOOTHED_DOSE_MIN_BOOTSTRAP`].
    pub min_bootstrap: u32,
    /// [`SMOOTHED_DOSE_MAX_BOOTSTRAP`].
    pub max_bootstrap: u32,
    /// [`SMOOTHED_DOSE_MAX_ROWS`].
    pub max_rows: usize,
    /// [`SMOOTHED_DOSE_MAX_FEATURES`].
    pub max_features: usize,
    /// [`SMOOTHED_DOSE_MAX_BASIS_DEGREE`].
    pub max_basis_degree: u8,
    /// [`SMOOTHED_DOSE_MAX_KNOTS`].
    pub max_knots: usize,
    /// [`SMOOTHED_DOSE_MAX_WORKSPACE_BYTES`].
    pub max_workspace_bytes: u64,
}

/// The frozen bounds of this cell.
pub const SMOOTHED_DOSE_BOUNDS: SmoothedDoseBounds = SmoothedDoseBounds {
    max_grid: SMOOTHED_DOSE_MAX_GRID,
    quadrature_nodes: SMOOTHED_DOSE_QUADRATURE_NODES,
    max_folds: SMOOTHED_DOSE_MAX_FOLDS,
    min_bootstrap: SMOOTHED_DOSE_MIN_BOOTSTRAP,
    max_bootstrap: SMOOTHED_DOSE_MAX_BOOTSTRAP,
    max_rows: SMOOTHED_DOSE_MAX_ROWS,
    max_features: SMOOTHED_DOSE_MAX_FEATURES,
    max_basis_degree: SMOOTHED_DOSE_MAX_BASIS_DEGREE,
    max_knots: SMOOTHED_DOSE_MAX_KNOTS,
    max_workspace_bytes: SMOOTHED_DOSE_MAX_WORKSPACE_BYTES,
};

/// Point-only status: no interval was requested.
pub const DOSE_POINT_ONLY: &str = "point_only";
/// Withheld status: an interval was requested and is withheld with a reason.
pub const DOSE_WITHHELD: &str = "withheld";

/// The deterministic dose-by-covariate basis the outcome learner is fed:
/// `[1, t, .., t^degree, (t - k)_+ for each knot, x_1..x_d, b(t) x_j if interactions]`.
///
/// It is a row-wise transform with no statistic fitted on the rows.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DoseBasis {
    /// Polynomial degree in the dose, `1..=`[`SMOOTHED_DOSE_MAX_BASIS_DEGREE`].
    pub degree: u8,
    /// Hinge knots inside the dose support, at most [`SMOOTHED_DOSE_MAX_KNOTS`].
    pub knots: Vec<f64>,
    /// Whether every dose term is also multiplied by every covariate.
    pub interactions: bool,
}

impl Default for DoseBasis {
    fn default() -> Self {
        Self { degree: 2, knots: Vec::new(), interactions: true }
    }
}

impl DoseBasis {
    /// Dose terms (excluding the intercept).
    #[must_use]
    pub fn dose_terms(&self) -> usize {
        usize::from(self.degree) + self.knots.len()
    }

    /// Width of the outcome design for `features` covariates, intercept included.
    #[must_use]
    pub fn width(&self, features: usize) -> usize {
        let dose = self.dose_terms();
        1 + dose + features + if self.interactions { dose * features } else { 0 }
    }

    fn dose_value(&self, term: usize, t: f64) -> f64 {
        let degree = usize::from(self.degree);
        if term < degree {
            // `term + 1` is at most the degree, itself at most 3.
            t.powi(i32::try_from(term + 1).unwrap_or(i32::MAX))
        } else {
            (t - self.knots[term - degree]).max(0.0)
        }
    }
}

/// Learners, basis, folds, quadrature, support thresholds and bootstrap request.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SmoothedDoseOptions {
    /// Source outcome nuisance `mu(t, x)` over the dose basis.
    pub outcome: LearnerSpec,
    /// Source-membership nuisance over `[1, x]`.
    pub membership: LearnerSpec,
    /// Dose-by-covariate basis of the outcome design.
    pub basis: DoseBasis,
    /// Shared cross-fit fold count, `2..=`[`SMOOTHED_DOSE_MAX_FOLDS`].
    pub folds: usize,
    /// Gauss-Legendre nodes per window piece, one of [`SMOOTHED_DOSE_QUADRATURE_NODES`];
    /// the check uses twice as many.
    pub quadrature_nodes: usize,
    /// Largest accepted `|nu_hat_Q - nu_hat_2Q|` on any target row at any grid dose, gated
    /// only for a fitted curve that is not piecewise polynomial in the dose.
    pub quadrature_tolerance: f64,
    /// Smallest out-of-fold membership probability accepted on any row, in `(0, 0.5)`.
    pub min_membership_probability: f64,
    /// Smallest known dose density accepted on any source row.
    pub min_dose_density: f64,
    /// Smallest Kish effective sample size of the source kernel weights at a grid dose.
    pub min_local_ess: f64,
    /// Fewest distinct source doses carrying kernel weight at a grid dose.
    pub min_distinct_doses: usize,
    /// Requested bootstrap replicates; the public route runs none.
    pub bootstrap: u32,
    /// Nominal pointwise coverage of the internal interval.
    pub coverage_level: f64,
}

impl Default for SmoothedDoseOptions {
    fn default() -> Self {
        Self {
            outcome: LearnerSpec::Linear(crate::LinearSpec::default()),
            membership: LearnerSpec::Logistic(crate::LogisticSpec::default()),
            basis: DoseBasis::default(),
            folds: 5,
            quadrature_nodes: 16,
            quadrature_tolerance: 1e-6,
            min_membership_probability: 0.05,
            min_dose_density: 1e-3,
            min_local_ess: 30.0,
            min_distinct_doses: 5,
            bootstrap: 0,
            coverage_level: 0.95,
        }
    }
}

/// Rows of one smoothed dose-response transport.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SmoothedDoseInput {
    /// Covariate coordinates, exactly the certified standardizers.
    pub features: Vec<u32>,
    /// Raw baseline covariates in column-major order, without an intercept.
    pub covariates: Vec<Vec<f64>>,
    /// Source outcomes; target entries are ignored.
    pub outcome: Vec<f64>,
    /// Randomized source doses; target entries are ignored.
    pub dose: Vec<f64>,
    /// Known conditional dose density `pi(A_i | X_i)` at each source row; target entries ignored.
    pub dose_density: Vec<f64>,
    /// Source membership.
    pub source: Vec<bool>,
    /// Declared sampling design.
    pub sampling: TrialSampling,
}

/// Interval status of a public estimate. No interval is attached while the route is closed.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SmoothedDoseUncertainty {
    /// [`DOSE_POINT_ONLY`] or [`DOSE_WITHHELD`].
    pub status: String,
    /// `no_interval_requested`, `estimator_inference_mismatch` or `cell_not_licensed`.
    pub reason: String,
    /// The namespaced detail when the request was below the replicate floor.
    pub detail: Option<String>,
    /// Bootstrap replicates the request asked for; none ran.
    pub replicates_requested: u32,
}

impl SmoothedDoseUncertainty {
    /// The status of a request for `bootstrap` replicates on the closed route.
    #[must_use]
    pub fn for_request(bootstrap: u32) -> Self {
        let (status, reason, detail) = if bootstrap == 0 {
            (DOSE_POINT_ONLY, "no_interval_requested", None)
        } else if bootstrap < SMOOTHED_DOSE_MIN_BOOTSTRAP {
            (
                DOSE_WITHHELD,
                antecedent_core::reason_code!("estimator_inference_mismatch"),
                Some("dose_response.bootstrap_below_floor"),
            )
        } else {
            (DOSE_WITHHELD, antecedent_core::reason_code!("cell_not_licensed"), None)
        };
        Self {
            status: status.into(),
            reason: reason.into(),
            detail: detail.map(Into::into),
            replicates_requested: bootstrap,
        }
    }

    /// Whether an interval is available. Never true while the route is closed.
    #[must_use]
    pub fn available(&self) -> bool {
        false
    }
}

/// Every fold's fitted nuisance, as portable prediction maps in fold order.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SmoothedDoseModels {
    /// Outcome model of each fold, fitted on the source rows of the other folds.
    pub outcome: Vec<PortablePredictor>,
    /// Membership model of each fold, fitted on all rows of the other folds.
    pub membership: Vec<PortablePredictor>,
}

/// The quadrature (numerical) error record of one grid dose, apart from sampling error.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QuadratureRecord {
    /// Gauss-Legendre nodes of the coarse rule on each window piece.
    pub nodes: usize,
    /// Nodes of the check rule on each piece (twice as many); the point uses it.
    pub check_nodes: usize,
    /// Pieces of the window `[a - h, a + h]` after splitting at the basis knots inside it.
    pub pieces: usize,
    /// Every fold's fitted curve is a linear map of the dose basis, so it is a polynomial
    /// of degree at most 3 on every piece and both rules are exact up to rounding; the
    /// tolerance is then not gated.
    pub exact: bool,
    /// `|psi_hat_Q - psi_hat_2Q|`: an estimate of the coarse rule's error, not a bound.
    pub estimate_error: f64,
    /// Largest `|nu_hat_Q - nu_hat_2Q|` over target rows: the gated quantity when the
    /// curve is not exact. An estimate, not a bound.
    pub max_row_error: f64,
}

/// The smoothing-bias diagnostic of one grid dose; never added to the estimate.
///
/// It is computed from the fitted outcome curve alone, so it inherits the curve's
/// misspecification: a fit linear in the dose gives exactly zero whatever the true
/// curvature.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SmoothingBiasDiagnostic {
    /// Plug-in `psi_hat_h - psi_hat_{h/2}` (target term only).
    pub half_bandwidth_difference: f64,
    /// `4/3` of the difference: the estimate of `psi_h - psi_0` when the fitted curve is
    /// locally quadratic in the dose.
    pub local_quadratic_bias: f64,
}

/// Local dose support of one grid dose, from the rows and known densities alone.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LocalDoseSupport {
    /// Kish effective sample size of `K_h(a - A_i) / pi_i` over source rows.
    pub effective_sample_size: f64,
    /// Distinct source doses with positive kernel weight.
    pub distinct_doses: usize,
    /// Largest `K_h(a - A_i) / pi_i`.
    pub max_kernel_weight: f64,
}

/// One grid dose of the executed point.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SmoothedDoseGridPoint {
    /// Grid dose `a`.
    pub dose: f64,
    /// `psi_hat_h(a)` (check-rule quadrature).
    pub estimate: f64,
    /// Target term alone: the outcome plug-in `mean_T nu_hat(X; a)`.
    pub plug_in: f64,
    /// Inverse-odds, inverse-density residual term divided by `n_0`.
    pub augmentation: f64,
    /// Quadrature error, separate from sampling error.
    pub quadrature: QuadratureRecord,
    /// Smoothing-bias diagnostic, separate from both.
    pub smoothing_bias: SmoothingBiasDiagnostic,
    /// Local dose support.
    pub support: LocalDoseSupport,
    /// Analytic influence-function standard error: a diagnostic, not a licensed claim.
    pub influence_se_diagnostic: f64,
}

/// Held-out nuisance losses.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SmoothedDoseDiagnostics {
    /// Out-of-fold membership log loss over every row.
    pub membership_logloss: f64,
    /// Out-of-fold outcome RMSE over source rows at their observed doses.
    pub outcome_rmse: f64,
}

/// Out-of-fold source-membership overlap.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MembershipOverlap {
    /// Smallest out-of-fold membership probability over every row (the gated quantity).
    pub probability_min: f64,
    /// Largest out-of-fold membership probability over every row.
    pub probability_max: f64,
    /// Largest inverse-odds weight `(1 - p) / p` over source rows.
    pub max_odds_weight: f64,
}

/// One executed point estimate with the evidence a consumer replays.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SmoothedDoseEstimate {
    /// One entry per grid dose, in the query's grid order.
    pub grid: Vec<SmoothedDoseGridPoint>,
    /// Interval status; the interval route is closed.
    pub uncertainty: SmoothedDoseUncertainty,
    /// Every fold's portable outcome and membership model.
    pub models: SmoothedDoseModels,
    /// Provider provenance of every fitted nuisance: membership folds, then outcome folds.
    pub provenance: Vec<LearnerProvenance>,
    /// Fold count, scheme and assignment.
    pub folds: FoldProvenance,
    /// Held-out losses.
    pub diagnostics: SmoothedDoseDiagnostics,
    /// Membership overlap.
    pub overlap: MembershipOverlap,
}

/// What the stored models and rows imply: the grid, diagnostics and overlap.
#[derive(Clone, Debug, PartialEq)]
#[doc(hidden)]
pub struct SmoothedDoseEvaluation {
    /// Per grid dose, in the query's grid order.
    pub grid: Vec<SmoothedDoseGridPoint>,
    /// Out-of-fold membership probabilities of every row.
    pub membership: Vec<f64>,
    /// Out-of-fold outcome predictions at observed doses (zero on target rows).
    pub mu_observed: Vec<f64>,
    /// Held-out losses.
    pub diagnostics: SmoothedDoseDiagnostics,
    /// Membership overlap.
    pub overlap: MembershipOverlap,
}

fn refuse(code: &'static str, detail: &str, message: &str) -> EstimationError {
    EstimationError::refused(code, format!("{detail}: {message}"))
}

fn support_refusal(detail: &str, message: &str) -> EstimationError {
    refuse(antecedent_core::reason_code!("transport_support_failure"), detail, message)
}

fn unsupported(detail: &str, message: &str) -> EstimationError {
    refuse(antecedent_core::reason_code!("route_not_supported"), detail, message)
}

fn cancelled(ctx: &ExecutionContext) -> Result<(), EstimationError> {
    if ctx.cancellation.is_cancelled() {
        Err(EstimationError::Refused {
            code: antecedent_core::reason_code!("transport_budget_cancel"),
            message: "smoothed dose transport cancelled".into(),
        })
    } else {
        Ok(())
    }
}

/// The refusal of the closed interval route: the point is retained, the interval is
/// withheld until this cell's coverage records are measured.
#[must_use]
#[doc(hidden)]
pub fn refuse_smoothed_dose_interval() -> EstimationError {
    refuse(
        antecedent_core::reason_code!("cell_not_licensed"),
        "dose_response.interval_withheld",
        "the joint outer bootstrap interval route is closed until its coverage records are \
         measured; the point estimate is retained",
    )
}

/// Refuse a sampling design other than the two declared IID designs.
///
/// # Errors
/// `sampling_dependence_unknown` (`dose_response.non_iid_design`).
#[doc(hidden)]
pub fn parse_smoothed_dose_sampling(name: &str) -> Result<TrialSampling, EstimationError> {
    match name {
        "nested_cohort" => Ok(TrialSampling::NestedCohort),
        "independent_samples" => Ok(TrialSampling::IndependentSamples),
        other => Err(refuse(
            antecedent_core::reason_code!("sampling_dependence_unknown"),
            "dose_response.non_iid_design",
            &format!(
                "sampling '{other}' is not a declared IID design; only nested_cohort and \
                 independent_samples are licensed"
            ),
        )),
    }
}

/// Refuse every target but the fixed-bandwidth smoothed response.
///
/// # Errors
/// `route_not_supported`: `dose_response.cate_or_simultaneous` for conditional,
/// heterogeneous or simultaneous-band targets; `dose_response.target_not_smoothed` for
/// point-curve, stochastic, coarsened, incremental, derivative or any other target.
#[doc(hidden)]
pub fn parse_smoothed_dose_target(name: &str) -> Result<(), EstimationError> {
    match name {
        SMOOTHED_DOSE_TARGET => Ok(()),
        "cate" | "conditional" | "heterogeneous" | "simultaneous_band" | "uniform_band" => {
            Err(unsupported(
                "dose_response.cate_or_simultaneous",
                &format!(
                    "target '{name}' is conditional or simultaneous; only the pointwise \
                     population smoothed response is licensed"
                ),
            ))
        }
        other => Err(unsupported(
            "dose_response.target_not_smoothed",
            &format!(
                "target '{other}' is not the fixed-bandwidth smoothed response psi_h; point \
                 curves, stochastic, coarsened, incremental and derivative targets are not \
                 licensed"
            ),
        )),
    }
}

/// Refuse a dose density that is not known by design.
///
/// # Errors
/// `route_not_supported` (`dose_response.estimated_dose_density`).
#[doc(hidden)]
pub fn parse_dose_density_provenance(name: &str) -> Result<(), EstimationError> {
    if name == "known" {
        return Ok(());
    }
    Err(unsupported(
        "dose_response.estimated_dose_density",
        &format!(
            "dose density provenance '{name}' is not known by design; an estimated \
             (generalized-propensity) density is not licensed"
        ),
    ))
}

/// Parse a kernel name; only Epanechnikov is supported.
///
/// # Errors
/// `route_not_supported` (`dose_response.kernel_not_supported`).
#[doc(hidden)]
pub fn parse_smoothing_kernel(name: &str) -> Result<SmoothingKernel, EstimationError> {
    if name == SmoothingKernel::Epanechnikov.name() {
        return Ok(SmoothingKernel::Epanechnikov);
    }
    Err(unsupported(
        "dose_response.kernel_not_supported",
        &format!("kernel '{name}' is not supported; the estimand fixes the Epanechnikov kernel"),
    ))
}

fn bounds_refusal(message: &str) -> EstimationError {
    unsupported("dose_response.bounds_exceeded", message)
}

/// The grid-count bound of a query (`dose_response.bounds_exceeded`).
fn check_grid_bound(query: &SmoothedDoseTransportQuery) -> Result<(), EstimationError> {
    if query.grid.len() > SMOOTHED_DOSE_MAX_GRID {
        return Err(bounds_refusal(&format!(
            "{} grid doses; at most {SMOOTHED_DOSE_MAX_GRID}",
            query.grid.len()
        )));
    }
    Ok(())
}

/// Every grid window `[a - h, a + h]` lies inside the declared dose support; there are
/// no boundary kernels (`dose_response.grid_outside_dose_support`).
fn check_grid_windows(query: &SmoothedDoseTransportQuery) -> Result<(), EstimationError> {
    let (lo, hi) = query.dose_support;
    let h = query.bandwidth;
    for &a in query.grid.iter() {
        if a - h < lo || a + h > hi {
            return Err(support_refusal(
                "dose_response.grid_outside_dose_support",
                &format!(
                    "window [{}, {}] of grid dose {a} leaves the declared dose support \
                     [{lo}, {hi}]; boundary kernels are not supported",
                    a - h,
                    a + h
                ),
            ));
        }
    }
    Ok(())
}

/// The option bounds: node count, folds, bootstrap replicates, basis degree and knots
/// (`dose_response.bounds_exceeded`).
fn check_option_bounds(options: &SmoothedDoseOptions) -> Result<(), EstimationError> {
    if !SMOOTHED_DOSE_QUADRATURE_NODES.contains(&options.quadrature_nodes) {
        return Err(bounds_refusal(&format!(
            "{} quadrature nodes; one of {SMOOTHED_DOSE_QUADRATURE_NODES:?}",
            options.quadrature_nodes
        )));
    }
    if options.folds > SMOOTHED_DOSE_MAX_FOLDS || options.bootstrap > SMOOTHED_DOSE_MAX_BOOTSTRAP {
        return Err(bounds_refusal(&format!(
            "at most {SMOOTHED_DOSE_MAX_FOLDS} folds and {SMOOTHED_DOSE_MAX_BOOTSTRAP} \
             bootstrap replicates"
        )));
    }
    if options.basis.degree == 0
        || options.basis.degree > SMOOTHED_DOSE_MAX_BASIS_DEGREE
        || options.basis.knots.len() > SMOOTHED_DOSE_MAX_KNOTS
    {
        return Err(bounds_refusal(&format!(
            "basis degree 1..={SMOOTHED_DOSE_MAX_BASIS_DEGREE} and at most \
             {SMOOTHED_DOSE_MAX_KNOTS} knots"
        )));
    }
    Ok(())
}

/// Every frozen bound of a request: the grid, the options and the rows.
fn check_bounds(
    query: &SmoothedDoseTransportQuery,
    input: &SmoothedDoseInput,
    options: &SmoothedDoseOptions,
) -> Result<(), EstimationError> {
    check_grid_bound(query)?;
    check_option_bounds(options)?;
    if input.source.len() > SMOOTHED_DOSE_MAX_ROWS
        || input.features.len() > SMOOTHED_DOSE_MAX_FEATURES
    {
        return Err(bounds_refusal(&format!(
            "at most {SMOOTHED_DOSE_MAX_ROWS} rows and {SMOOTHED_DOSE_MAX_FEATURES} covariates"
        )));
    }
    Ok(())
}

fn check_learners(options: &SmoothedDoseOptions) -> Result<(), EstimationError> {
    for spec in [options.outcome, options.membership] {
        spec.validate().map_err(crate::learn_nuisance::learn_err)?;
        if matches!(spec, LearnerSpec::NeuralNet(_)) {
            return Err(unsupported(
                "dose_response.learner_not_portable",
                "the neural learner exports no portable predictor, so its predictions cannot \
                 be replayed bit for bit",
            ));
        }
    }
    Ok(())
}

/// Source rows and target rows, in row order.
fn strata(input: &SmoothedDoseInput) -> [Vec<usize>; 2] {
    let mut rows = [vec![], vec![]];
    for (i, source) in input.source.iter().enumerate() {
        rows[usize::from(!*source)].push(i);
    }
    rows
}

/// Stratified round-robin fold of every row: the source rows and the target rows are
/// each dealt across the folds in row order, so every fold holds both roles. The
/// assignment is deterministic in the source flags alone; no seed enters it.
#[must_use]
#[doc(hidden)]
#[allow(
    clippy::cast_possible_truncation,
    reason = "i % folds is below the fold count, which validation bounds by 20"
)]
pub fn smoothed_dose_fold_assignment(input: &SmoothedDoseInput, folds: usize) -> Vec<u16> {
    let mut assignment = vec![0u16; input.source.len()];
    for rows in strata(input) {
        for (i, row) in rows.into_iter().enumerate() {
            assignment[row] = (i % folds.max(1)) as u16;
        }
    }
    assignment
}

fn kernel_weight(
    query: &SmoothedDoseTransportQuery,
    a: f64,
    h: f64,
    dose: f64,
    density: f64,
) -> f64 {
    query.kernel.density((dose - a) / h) / h / density
}

/// The local dose support at grid dose `a` with bandwidth `h`, from rows and densities.
fn local_support(
    query: &SmoothedDoseTransportQuery,
    input: &SmoothedDoseInput,
    a: f64,
) -> LocalDoseSupport {
    let h = query.bandwidth;
    let (mut sum, mut squares, mut max) = (0.0, 0.0, 0.0_f64);
    let mut doses = Vec::new();
    for i in 0..input.source.len() {
        if !input.source[i] {
            continue;
        }
        let w = kernel_weight(query, a, h, input.dose[i], input.dose_density[i]);
        if w > 0.0 {
            sum += w;
            squares += w * w;
            max = max.max(w);
            doses.push(input.dose[i]);
        }
    }
    doses.sort_by(f64::total_cmp);
    doses.dedup_by(|x, y| x.total_cmp(y).is_eq());
    LocalDoseSupport {
        effective_sample_size: if squares > 0.0 { sum * sum / squares } else { 0.0 },
        distinct_doses: doses.len(),
        max_kernel_weight: max,
    }
}

/// Validate a request before any nuisance is fitted.
///
/// # Errors
/// An uncertified or non-standardization certificate, a malformed query or rows,
/// declared bounds above the frozen caps (`dose_response.bounds_exceeded`), an estimated
/// density, a non-portable learner, an invalid or floored known density, a grid window
/// outside the dose support, or thin or too-discrete local dose support.
#[doc(hidden)]
#[allow(clippy::too_many_lines, reason = "one guard per declared refusal, in refusal order")]
pub fn validate_smoothed_dose(
    id: &TransportIdentification,
    query: &SmoothedDoseTransportQuery,
    input: &SmoothedDoseInput,
    options: &SmoothedDoseOptions,
) -> Result<(), EstimationError> {
    query.validate().map_err(|e| EstimationError::data_msg(e.to_string()))?;
    parse_dose_density_provenance(&query.density_provenance)?;
    check_bounds(query, input, options)?;
    check_learners(options)?;
    let over: &[VariableId] = match id {
        TransportIdentification::Transportable { formula: TransportFormula::Direct(_), .. } => &[],
        TransportIdentification::Transportable {
            formula: TransportFormula::Standardize { over, .. },
            ..
        } => over,
        _ => {
            return Err(EstimationError::data_msg(
                "the smoothed dose transport requires a direct or standardization certificate",
            ));
        }
    };
    let mut expected: Vec<_> = over.iter().map(|v| v.raw()).collect();
    expected.sort_unstable();
    let mut actual = input.features.clone();
    actual.sort_unstable();
    let n = input.source.len();
    let (lo, hi) = query.dose_support;
    if expected != actual
        || input.covariates.len() != actual.len()
        || n == 0
        || input.outcome.len() != n
        || input.dose.len() != n
        || input.dose_density.len() != n
        || input.covariates.iter().any(|col| col.len() != n || col.iter().any(|v| !v.is_finite()))
        || options.folds < 2
        || !options.coverage_level.is_finite()
        || options.coverage_level <= 0.0
        || options.coverage_level >= 1.0
        || !(options.min_membership_probability > 0.0 && options.min_membership_probability < 0.5)
        || !(options.min_dose_density.is_finite() && options.min_dose_density > 0.0)
        || !(options.min_local_ess.is_finite() && options.min_local_ess >= 1.0)
        || options.min_distinct_doses < 2
        || !(options.quadrature_tolerance.is_finite() && options.quadrature_tolerance > 0.0)
        || options.basis.knots.iter().any(|k| !k.is_finite() || *k <= lo || *k >= hi)
    {
        return Err(EstimationError::data_msg(
            "invalid certified smoothed dose transport input or options",
        ));
    }
    for i in 0..n {
        if !input.source[i] {
            continue;
        }
        let (dose, density) = (input.dose[i], input.dose_density[i]);
        if !input.outcome[i].is_finite() {
            return Err(EstimationError::data_msg("source outcomes must be finite"));
        }
        if !dose.is_finite() || !density.is_finite() || density <= 0.0 || dose < lo || dose > hi {
            return Err(support_refusal(
                "dose_response.dose_density_invalid",
                &format!(
                    "source row {i}: dose {dose} with known density {density} is not a \
                     positive density inside the declared support [{lo}, {hi}]"
                ),
            ));
        }
        if density < options.min_dose_density {
            return Err(support_refusal(
                "dose_response.dose_density_floor",
                &format!(
                    "source row {i}: known dose density {density} is below the declared {}",
                    options.min_dose_density
                ),
            ));
        }
    }
    if strata(input).iter().any(|rows| rows.len() < options.folds) {
        return Err(EstimationError::data_msg(
            "the source and target samples each need at least one row per fold",
        ));
    }
    check_grid_windows(query)?;
    for &a in query.grid.iter() {
        let support = local_support(query, input, a);
        if support.effective_sample_size < options.min_local_ess {
            return Err(support_refusal(
                "dose_response.local_dose_ess",
                &format!(
                    "local dose effective sample size {} at grid dose {a} is below the \
                     declared {}",
                    support.effective_sample_size, options.min_local_ess
                ),
            ));
        }
        if support.distinct_doses < options.min_distinct_doses {
            return Err(refuse(
                antecedent_core::reason_code!("treatment_support_too_discrete"),
                "dose_response.dose_too_discrete",
                &format!(
                    "{} distinct source doses carry kernel weight at grid dose {a}; at least \
                     {} are declared",
                    support.distinct_doses, options.min_distinct_doses
                ),
            ));
        }
    }
    Ok(())
}

/// Membership design `[1, x]` over every row, column-major.
fn membership_design(input: &SmoothedDoseInput) -> Vec<f64> {
    let n = input.source.len();
    let mut design = vec![1.0; n];
    for col in &input.covariates {
        design.extend_from_slice(col);
    }
    design
}

/// Outcome design at `(dose(r), x_{rows[r]})` for every `r`, column-major.
fn outcome_design(
    basis: &DoseBasis,
    covariates: &[Vec<f64>],
    rows: &[usize],
    dose: impl Fn(usize) -> f64,
) -> Vec<f64> {
    let m = rows.len();
    let dose_terms = basis.dose_terms();
    let width = basis.width(covariates.len());
    let mut design = vec![0.0; m * width];
    design[..m].fill(1.0);
    let terms: Vec<Vec<f64>> = (0..dose_terms)
        .map(|term| (0..m).map(|r| basis.dose_value(term, dose(r))).collect())
        .collect();
    let mut column = 1;
    for values in &terms {
        design[column * m..(column + 1) * m].copy_from_slice(values);
        column += 1;
    }
    for col in covariates {
        for (r, row) in rows.iter().enumerate() {
            design[column * m + r] = col[*row];
        }
        column += 1;
    }
    if basis.interactions {
        for values in &terms {
            for col in covariates {
                for (r, row) in rows.iter().enumerate() {
                    design[column * m + r] = values[r] * col[*row];
                }
                column += 1;
            }
        }
    }
    design
}

fn predict(
    model: &PortablePredictor,
    design: &[f64],
    rows: usize,
    ctx: &ExecutionContext,
) -> Result<Vec<f64>, EstimationError> {
    let mut out = vec![0.0; rows];
    if rows == 0 {
        return Ok(out);
    }
    let x = DesignView::from_column_major(design, rows, model.columns)
        .map_err(crate::learn_nuisance::learn_err)?;
    model.predict(x, &mut out, ctx).map_err(crate::learn_nuisance::learn_err)?;
    Ok(out)
}

fn not_portable(error: &antecedent_learn::LearnError) -> EstimationError {
    unsupported(
        "dose_response.learner_not_portable",
        &format!("a fitted nuisance exports no portable predictor: {error}"),
    )
}

/// Fit every fold's outcome and membership model on the other folds and export each as
/// a portable predictor.
fn fit_models(
    input: &SmoothedDoseInput,
    options: &SmoothedDoseOptions,
    folds: &[u16],
    ctx: &ExecutionContext,
) -> Result<SmoothedDoseModels, EstimationError> {
    cancelled(ctx)?;
    let n = input.source.len();
    let fold_count = options.folds;
    let membership_factory = resolve_for(options.membership, PredictionTask::BinaryProbability)
        .map_err(crate::learn_nuisance::learn_err)?;
    let outcome_factory = resolve_for(options.outcome, PredictionTask::Regression)
        .map_err(crate::learn_nuisance::learn_err)?;
    let membership_x = membership_design(input);
    let membership_view = DesignView::from_column_major(&membership_x, n, input.features.len() + 1)
        .map_err(crate::learn_nuisance::learn_err)?;
    let labels: Vec<f64> = input.source.iter().map(|s| f64::from(*s)).collect();
    let source_rows: Vec<usize> = (0..n).filter(|i| input.source[*i]).collect();
    let outcome_x = outcome_design(&options.basis, &input.covariates, &source_rows, |r| {
        input.dose[source_rows[r]]
    });
    let outcome_view = DesignView::from_column_major(
        &outcome_x,
        source_rows.len(),
        options.basis.width(input.features.len()),
    )
    .map_err(crate::learn_nuisance::learn_err)?;
    let source_outcome: Vec<f64> = source_rows.iter().map(|i| input.outcome[*i]).collect();
    let fits = ctx.map_indexed(fold_count, |fold, inner| -> Result<_, EstimationError> {
        cancelled(inner)?;
        let train: Vec<u32> = (0..n)
            .filter(|i| usize::from(folds[*i]) != fold)
            .map(|i| u32::try_from(i).unwrap_or(u32::MAX))
            .collect();
        let membership = membership_factory
            .fit(
                membership_view
                    .with_rows(RowSelection::new(&train))
                    .map_err(crate::learn_nuisance::learn_err)?,
                TargetView::new(&labels),
                None,
                inner,
            )
            .map_err(crate::learn_nuisance::learn_err)?
            .portable()
            .map_err(|e| not_portable(&e))?;
        let outcome_train: Vec<u32> = (0..source_rows.len())
            .filter(|r| usize::from(folds[source_rows[*r]]) != fold)
            .map(|r| u32::try_from(r).unwrap_or(u32::MAX))
            .collect();
        let outcome = outcome_factory
            .fit(
                outcome_view
                    .with_rows(RowSelection::new(&outcome_train))
                    .map_err(crate::learn_nuisance::learn_err)?,
                TargetView::new(&source_outcome),
                None,
                inner,
            )
            .map_err(crate::learn_nuisance::learn_err)?
            .portable()
            .map_err(|e| not_portable(&e))?;
        Ok((membership, outcome))
    })?;
    let (membership, outcome) = fits.into_iter().unzip();
    Ok(SmoothedDoseModels { outcome, membership })
}

/// Design rows per quadrature chunk for a design of `width` columns: at most
/// [`CHUNK_DESIGN_ROWS`] and at most [`CHUNK_DESIGN_BYTES`] of design.
fn chunk_design_rows(width: usize) -> usize {
    CHUNK_DESIGN_ROWS.min((CHUNK_DESIGN_BYTES / (8 * width.max(1))).max(1))
}

/// The composite Gauss-Legendre rule on `[-1, 1]` for the window `[a - h, a + h]`, split at
/// every basis knot strictly inside it: the base rule mapped onto each piece. Without a
/// knot inside the window it is the base rule itself, bit for bit.
fn split_rule(
    knots: &[f64],
    (a, h): (f64, f64),
    (nodes, weights): (&[f64], &[f64]),
) -> (Vec<f64>, Vec<f64>) {
    let mut edges = vec![-1.0];
    let mut cuts: Vec<f64> =
        knots.iter().map(|k| (k - a) / h).filter(|u| *u > -1.0 && *u < 1.0).collect();
    cuts.sort_by(f64::total_cmp);
    cuts.dedup_by(|x, y| x.total_cmp(y).is_eq());
    edges.extend(cuts);
    edges.push(1.0);
    let pieces = edges.len() - 1;
    let (mut u, mut w) = (Vec::with_capacity(pieces * nodes.len()), Vec::new());
    w.reserve(pieces * nodes.len());
    for pair in edges.windows(2) {
        let (mid, half) = ((pair[0] + pair[1]) / 2.0, (pair[1] - pair[0]) / 2.0);
        for (x, v) in nodes.iter().zip(weights) {
            u.push(mid + half * x);
            w.push(v * half);
        }
    }
    (u, w)
}

/// Whether every fold's fitted outcome curve is a linear map of the dose basis, hence a
/// polynomial of degree at most 3 in the dose between knots.
fn piecewise_polynomial(models: &SmoothedDoseModels) -> bool {
    models.outcome.iter().all(|m| matches!(m.model, PredictionMap::Linear { logistic: false, .. }))
}

/// Predicts `rows` design rows (column-major `design`) with fold `fold`'s outcome model.
type FoldPredictor<'a> = dyn Fn(usize, &[f64], usize, &ExecutionContext) -> Result<Vec<f64>, EstimationError>
    + Sync
    + 'a;

/// Target-term quadrature of one grid dose under one rule: `nu_hat` per target row, in
/// target-row order, each from its own fold's model. Cancellation is polled before every
/// chunk of at most [`CHUNK_DESIGN_ROWS`] design rows.
#[allow(clippy::too_many_arguments, reason = "one quadrature rule of one grid dose")]
fn smoothed_target_rows(
    basis: &DoseBasis,
    input: &SmoothedDoseInput,
    kernel: SmoothingKernel,
    folds: &[u16],
    (fold_count, predictor): (usize, &FoldPredictor<'_>),
    (a, h): (f64, f64),
    (nodes, weights): (&[f64], &[f64]),
    target_rows: &[usize],
    ctx: &ExecutionContext,
) -> Result<Vec<f64>, EstimationError> {
    let q = nodes.len();
    let kernel_weights: Vec<f64> =
        nodes.iter().zip(weights).map(|(u, w)| w * kernel.density(*u)).collect();
    let mut nu = vec![0.0; target_rows.len()];
    let chunk = (chunk_design_rows(basis.width(input.features.len())) / q.max(1)).max(1);
    for fold in 0..fold_count {
        let positions: Vec<usize> = (0..target_rows.len())
            .filter(|p| usize::from(folds[target_rows[*p]]) == fold)
            .collect();
        for block in positions.chunks(chunk) {
            cancelled(ctx)?;
            // Row r*q + k of the chunk design is target row block[r] at node k.
            let expanded: Vec<usize> =
                block.iter().flat_map(|p| std::iter::repeat_n(target_rows[*p], q)).collect();
            let design =
                outcome_design(basis, &input.covariates, &expanded, |r| a + h * nodes[r % q]);
            let predictions = predictor(fold, &design, expanded.len(), ctx)?;
            for (r, p) in block.iter().enumerate() {
                let mut value = 0.0;
                for k in 0..q {
                    value += kernel_weights[k] * predictions[r * q + k];
                }
                nu[*p] = value;
            }
        }
    }
    Ok(nu)
}

fn mean_in_order(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

/// Evaluate the point from stored fold models, rows and premises: out-of-fold
/// membership and outcome predictions, the quadratures, the score, the diagnostics and
/// every post-fit refusal (membership overlap on every row, source and target; quadrature
/// tolerance on the largest target-row difference of a non-polynomial curve). The producer and
/// the artifact consumer both call it, so the consumer's replay is bit for bit; its
/// correctness is established separately by the known-truth and exact-integration
/// fixtures.
///
/// # Errors
/// Malformed models, cancellation, `dose_response.membership_overlap` or
/// `dose_response.quadrature_tolerance`.
#[doc(hidden)]
#[allow(clippy::too_many_lines, reason = "the score, its quadratures and its diagnostics")]
pub fn evaluate_smoothed_dose_models(
    query: &SmoothedDoseTransportQuery,
    input: &SmoothedDoseInput,
    options: &SmoothedDoseOptions,
    folds: &[u16],
    models: &SmoothedDoseModels,
    ctx: &ExecutionContext,
) -> Result<SmoothedDoseEvaluation, EstimationError> {
    cancelled(ctx)?;
    let n = input.source.len();
    let fold_count = options.folds;
    let width = options.basis.width(input.features.len());
    if folds.len() != n
        || models.outcome.len() != fold_count
        || models.membership.len() != fold_count
        || models.outcome.iter().any(|m| m.columns != width || m.validate().is_err())
        || models
            .membership
            .iter()
            .any(|m| m.columns != input.features.len() + 1 || m.validate().is_err())
    {
        return Err(EstimationError::data_msg("fold models do not match the request's design"));
    }
    // Out-of-fold membership over every row and outcome at the observed source doses.
    let membership_x = membership_design(input);
    let mut membership = vec![0.0; n];
    let mut mu_observed = vec![0.0; n];
    for fold in 0..fold_count {
        cancelled(ctx)?;
        let rows: Vec<usize> = (0..n).filter(|i| usize::from(folds[*i]) == fold).collect();
        let mut design = Vec::with_capacity(rows.len() * (input.features.len() + 1));
        for c in 0..=input.features.len() {
            design.extend(rows.iter().map(|i| membership_x[c * n + i]));
        }
        let p = predict(&models.membership[fold], &design, rows.len(), ctx)?;
        for (i, value) in rows.iter().zip(p) {
            membership[*i] = value;
        }
        let source: Vec<usize> = rows.into_iter().filter(|i| input.source[*i]).collect();
        let design =
            outcome_design(&options.basis, &input.covariates, &source, |r| input.dose[source[r]]);
        let mu = predict(&models.outcome[fold], &design, source.len(), ctx)?;
        for (i, value) in source.iter().zip(mu) {
            mu_observed[*i] = value;
        }
    }
    let lowest = membership.iter().copied().fold(f64::INFINITY, f64::min);
    let highest = membership.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if lowest < options.min_membership_probability || membership.iter().any(|p| !p.is_finite()) {
        return Err(support_refusal(
            "dose_response.membership_overlap",
            &format!(
                "out-of-fold membership probability {lowest} is below the declared {}",
                options.min_membership_probability
            ),
        ));
    }
    let source_rows: Vec<usize> = (0..n).filter(|i| input.source[*i]).collect();
    let target_rows: Vec<usize> = (0..n).filter(|i| !input.source[*i]).collect();
    let odds: Vec<f64> =
        source_rows.iter().map(|i| (1.0 - membership[*i]) / membership[*i]).collect();
    let labels: Vec<f64> = input.source.iter().map(|s| f64::from(*s)).collect();
    let membership_logloss =
        antecedent_learn::diagnose(PredictionTask::BinaryProbability, &labels, &membership)
            .logloss
            .ok_or_else(|| EstimationError::data_msg("empty membership role"))?;
    let observed: Vec<f64> = source_rows.iter().map(|i| input.outcome[*i]).collect();
    let fitted: Vec<f64> = source_rows.iter().map(|i| mu_observed[*i]).collect();
    let outcome_rmse = antecedent_learn::diagnose(PredictionTask::Regression, &observed, &fitted)
        .rmse
        .ok_or_else(|| EstimationError::data_msg("empty source outcome role"))?;
    let n0 = target_rows.len() as f64;
    let coarse = antecedent_stats::special::gauss_legendre(options.quadrature_nodes);
    let fine = antecedent_stats::special::gauss_legendre(2 * options.quadrature_nodes);
    let exact = piecewise_polynomial(models);
    let knots = &options.basis.knots;
    let h = query.bandwidth;
    let predictor = |fold: usize, design: &[f64], rows: usize, c: &ExecutionContext| {
        predict(&models.outcome[fold], design, rows, c)
    };
    let grid = ctx.map_indexed(query.grid.len(), |g, inner| -> Result<_, EstimationError> {
        let a = query.grid[g];
        let rule = |(nodes, weights): (&[f64], &[f64]), bandwidth: f64| {
            let (nodes, weights) = split_rule(knots, (a, bandwidth), (nodes, weights));
            smoothed_target_rows(
                &options.basis,
                input,
                query.kernel,
                folds,
                (fold_count, &predictor),
                (a, bandwidth),
                (&nodes, &weights),
                &target_rows,
                inner,
            )
        };
        let pieces = split_rule(knots, (a, h), (&coarse.0, &coarse.1)).0.len() / coarse.0.len();
        let nu_coarse = rule((&coarse.0, &coarse.1), h)?;
        let nu_fine = rule((&fine.0, &fine.1), h)?;
        let nu_half = rule((&fine.0, &fine.1), h / 2.0)?;
        cancelled(inner)?;
        let mut residual_sum = 0.0;
        let mut residual_terms = Vec::with_capacity(source_rows.len());
        for (s, i) in source_rows.iter().enumerate() {
            let w = kernel_weight(query, a, h, input.dose[*i], input.dose_density[*i]);
            let term = odds[s] * w * (input.outcome[*i] - mu_observed[*i]);
            residual_sum += term;
            residual_terms.push(term);
        }
        let plug_in = mean_in_order(&nu_fine);
        let coarse_plug_in = mean_in_order(&nu_coarse);
        let augmentation = residual_sum / n0;
        let estimate = plug_in + augmentation;
        let estimate_error = ((coarse_plug_in + augmentation) - estimate).abs();
        let max_row_error =
            nu_coarse.iter().zip(&nu_fine).map(|(c, f)| (c - f).abs()).fold(0.0, f64::max);
        if !estimate.is_finite()
            || estimate_error.is_nan()
            || max_row_error.is_nan()
            || (!exact && max_row_error > options.quadrature_tolerance)
        {
            return Err(refuse(
                antecedent_core::reason_code!("transport_numerical_failure"),
                "dose_response.quadrature_tolerance",
                &format!(
                    "at grid dose {a} the {}-node and {}-node quadratures of a target row's \
                     fitted curve differ by up to {max_row_error}, above the declared tolerance \
                     {} (the curve is not piecewise polynomial in the dose)",
                    coarse.0.len(),
                    fine.0.len(),
                    options.quadrature_tolerance
                ),
            ));
        }
        let half_bandwidth_difference = plug_in - mean_in_order(&nu_half);
        let squares: f64 = nu_fine.iter().map(|v| (v - estimate) * (v - estimate)).sum::<f64>()
            + residual_terms.iter().map(|t| t * t).sum::<f64>();
        Ok(SmoothedDoseGridPoint {
            dose: a,
            estimate,
            plug_in,
            augmentation,
            quadrature: QuadratureRecord {
                nodes: coarse.0.len(),
                check_nodes: fine.0.len(),
                pieces,
                exact,
                estimate_error,
                max_row_error,
            },
            smoothing_bias: SmoothingBiasDiagnostic {
                half_bandwidth_difference,
                local_quadratic_bias: half_bandwidth_difference * 4.0 / 3.0,
            },
            support: local_support(query, input, a),
            influence_se_diagnostic: squares.sqrt() / n0,
        })
    })?;
    let max_odds_weight = odds.iter().copied().fold(0.0, f64::max);
    Ok(SmoothedDoseEvaluation {
        grid,
        membership,
        mu_observed,
        diagnostics: SmoothedDoseDiagnostics { membership_logloss, outcome_rmse },
        overlap: MembershipOverlap {
            probability_min: lowest,
            probability_max: highest,
            max_odds_weight,
        },
    })
}

/// The sizes a workspace estimate depends on.
#[derive(Clone, Copy, Debug)]
struct WorkspaceShape {
    rows: usize,
    source: usize,
    features: usize,
    width: usize,
    folds: usize,
    grid: usize,
    /// Most design rows one target row expands to in one quadrature call.
    nodes_per_row: usize,
    threads: usize,
}

/// Estimated live bytes: the rows, every design actually allocated and, when `fit`, one
/// training-design copy, normal matrix and index set per concurrently fitted fold; the
/// evaluation's per-fold designs and prediction vectors; and one quadrature chunk plus
/// its per-row vectors per concurrently integrated grid dose. `None` on overflow.
fn workspace(shape: WorkspaceShape, fit: bool) -> Option<u64> {
    let WorkspaceShape {
        rows: n,
        source: n1,
        features: d,
        width: w,
        folds,
        grid,
        nodes_per_row,
        threads,
    } = shape;
    let n0 = n.checked_sub(n1)?;
    let folds = folds.max(1);
    let membership = n.checked_mul(d + 1)?;
    let outcome = n1.checked_mul(w)?;
    // Stored rows: covariates, outcome, dose, density and flags; fold labels.
    let stored = n.checked_mul(d + 5)?;
    let fitting = if fit {
        let per_fold = membership
            .checked_add(outcome)?
            .checked_add((d + 1).checked_mul(d + 1)?)?
            .checked_add(w.checked_mul(w)?)?
            .checked_add(n.checked_mul(2)?)?;
        membership
            .checked_add(outcome)?
            .checked_add(n.checked_mul(2)?)?
            .checked_add(threads.clamp(1, folds).checked_mul(per_fold)?)?
    } else {
        0
    };
    let fold_designs =
        n.div_ceil(folds).checked_mul(d + 1)?.checked_add(n1.div_ceil(folds).checked_mul(w)?)?;
    let chunk = chunk_design_rows(w).max(nodes_per_row);
    let per_grid = chunk
        .checked_mul(w + 3)?
        .checked_add(n0.checked_mul(4)?)?
        .checked_add(n1.checked_mul(2)?)?;
    let evaluation = membership
        .checked_add(n.checked_mul(8)?)?
        .checked_add(fold_designs)?
        .checked_add(threads.clamp(1, grid.max(1)).checked_mul(per_grid)?)?;
    let total = stored.checked_add(fitting)?.checked_add(evaluation)?.checked_mul(8)?;
    u64::try_from(total).ok()
}

/// The estimated workspace of one request of `grid` grid doses under `ctx`'s thread
/// budget: with `fit` for
/// the producer (fits and evaluation), without for a consumer's replay (evaluation only).
/// `None` on overflow.
#[must_use]
#[doc(hidden)]
pub fn smoothed_dose_workspace_bytes(
    grid: usize,
    input: &SmoothedDoseInput,
    options: &SmoothedDoseOptions,
    ctx: &ExecutionContext,
    fit: bool,
) -> Option<u64> {
    workspace(
        WorkspaceShape {
            rows: input.source.len(),
            source: input.source.iter().filter(|s| **s).count(),
            features: input.features.len(),
            width: options.basis.width(input.features.len()),
            folds: options.folds,
            grid,
            nodes_per_row: 2 * options.quadrature_nodes * (options.basis.knots.len() + 1),
            threads: ctx.parallelism.max_threads.get() as usize,
        },
        fit,
    )
}

/// The memory cap in force: [`SMOOTHED_DOSE_MAX_WORKSPACE_BYTES`], lowered by a context
/// hard memory limit.
#[must_use]
#[doc(hidden)]
pub fn smoothed_dose_memory_cap(ctx: &ExecutionContext) -> u64 {
    ctx.memory
        .hard_limit_bytes
        .map_or(SMOOTHED_DOSE_MAX_WORKSPACE_BYTES, |h| h.min(SMOOTHED_DOSE_MAX_WORKSPACE_BYTES))
}

/// Refuse a workspace estimate above the cap in force, before anything is allocated.
fn check_workspace(bytes: Option<u64>, ctx: &ExecutionContext) -> Result<(), EstimationError> {
    let cap = smoothed_dose_memory_cap(ctx);
    match bytes {
        Some(bytes) if bytes <= cap => Ok(()),
        _ => Err(EstimationError::Refused {
            code: antecedent_core::reason_code!("transport_budget_cancel"),
            message: format!(
                "the smoothed dose workspace estimate of {} bytes exceeds the memory cap of \
                 {cap} bytes",
                bytes.map_or_else(|| "more than u64::MAX".into(), |b| b.to_string())
            ),
        }),
    }
}

/// Provenance of every fold model: membership folds, then outcome folds.
fn provenance(models: &SmoothedDoseModels) -> Vec<LearnerProvenance> {
    models.membership.iter().chain(&models.outcome).map(|m| m.provenance.clone()).collect()
}

fn fit_and_evaluate(
    query: &SmoothedDoseTransportQuery,
    input: &SmoothedDoseInput,
    options: &SmoothedDoseOptions,
    folds: &[u16],
    ctx: &ExecutionContext,
) -> Result<(SmoothedDoseModels, SmoothedDoseEvaluation), EstimationError> {
    let models = fit_models(input, options, folds, ctx)?;
    let evaluation = evaluate_smoothed_dose_models(query, input, options, folds, &models, ctx)?;
    Ok((models, evaluation))
}

/// Execute the point estimate: validate, cross-fit every nuisance, integrate, apply the
/// support and quadrature refusals and report the interval status of the closed route.
///
/// # Errors
/// Any [`validate_smoothed_dose`] refusal, a workspace estimate above the mandatory cap
/// or the context's lower hard memory limit, or cancellation (`transport_budget_cancel`),
/// a fit failure, membership overlap or quadrature tolerance.
pub fn estimate_smoothed_dose(
    id: &TransportIdentification,
    query: &SmoothedDoseTransportQuery,
    input: &SmoothedDoseInput,
    options: &SmoothedDoseOptions,
    ctx: &ExecutionContext,
) -> Result<SmoothedDoseEstimate, EstimationError> {
    validate_smoothed_dose(id, query, input, options)?;
    cancelled(ctx)?;
    check_workspace(
        smoothed_dose_workspace_bytes(query.grid.len(), input, options, ctx, true),
        ctx,
    )?;
    let folds = smoothed_dose_fold_assignment(input, options.folds);
    let (models, evaluation) = fit_and_evaluate(query, input, options, &folds, ctx)?;
    Ok(SmoothedDoseEstimate {
        grid: evaluation.grid,
        uncertainty: SmoothedDoseUncertainty::for_request(options.bootstrap),
        provenance: provenance(&models),
        models,
        folds: FoldProvenance {
            count: options.folds,
            scheme: SMOOTHED_DOSE_FOLD_SCHEME.into(),
            assignment: folds,
        },
        diagnostics: evaluation.diagnostics,
        overlap: evaluation.overlap,
    })
}

/// The resampling units of the joint outer bootstrap: one cohort for a nested cohort,
/// and the trial and target samples separately (each keeping its size) otherwise.
#[must_use]
#[doc(hidden)]
pub fn smoothed_dose_bootstrap_groups(input: &SmoothedDoseInput) -> Vec<Vec<usize>> {
    let n = input.source.len();
    match input.sampling {
        TrialSampling::NestedCohort => vec![(0..n).collect()],
        TrialSampling::IndependentSamples => vec![
            (0..n).filter(|i| input.source[*i]).collect(),
            (0..n).filter(|i| !input.source[*i]).collect(),
        ],
    }
}

fn menu_refusal(
    code: &'static str,
    detail: Option<&str>,
    reason: impl Into<String>,
) -> MenuRefusal {
    MenuRefusal { code: code.into(), detail: detail.map(Into::into), reason: reason.into() }
}

fn owned(values: &[&str]) -> Vec<String> {
    values.iter().map(|v| (*v).to_owned()).collect()
}

fn coordinates(vars: &[VariableId]) -> String {
    if vars.is_empty() {
        return "none".into();
    }
    vars.iter().map(|v| format!("v{}", v.raw())).collect::<Vec<_>>().join(", ")
}

/// The refusal that the certificate and the query put on the smoothed estimator.
fn dose_structural_refusal(
    id: &TransportIdentification,
    query: &SmoothedDoseTransportQuery,
) -> Option<MenuRefusal> {
    if let Err(error) = query.validate() {
        return Some(menu_refusal(
            antecedent_core::reason_code!("invalid_argument"),
            None,
            error.to_string(),
        ));
    }
    let refused = |error: EstimationError| match error {
        EstimationError::Refused { code, message } => {
            let detail = message.split(':').next().map(str::to_owned);
            Some(MenuRefusal { code: code.into(), detail, reason: message })
        }
        other => Some(menu_refusal(
            antecedent_core::reason_code!("invalid_argument"),
            None,
            other.to_string(),
        )),
    };
    if let Err(error) = parse_dose_density_provenance(&query.density_provenance)
        .and_then(|()| check_grid_bound(query))
        .and_then(|()| check_grid_windows(query))
    {
        return refused(error);
    }
    match id {
        TransportIdentification::Transportable {
            formula: TransportFormula::Direct(_) | TransportFormula::Standardize { .. },
            ..
        } => None,
        TransportIdentification::Transportable {
            formula: TransportFormula::RecursiveFactorization { .. },
            certificate,
        } => Some(menu_refusal(
            antecedent_core::reason_code!("transport_not_certified"),
            None,
            format!(
                "certificate rule '{}' yields a recursive factorization, which is identify-only",
                certificate.rule
            ),
        )),
        TransportIdentification::NotCertified(certificate) => Some(menu_refusal(
            antecedent_core::reason_code!("transport_not_certified"),
            None,
            format!("{}: {}", certificate.reason, certificate.message),
        )),
        TransportIdentification::MissingEvidence(certificate) => Some(menu_refusal(
            antecedent_core::reason_code!("transport_missing_evidence"),
            None,
            format!("{}: {}", certificate.reason, certificate.message),
        )),
    }
}

/// The refusal the learners and options put on the smoothed estimator.
fn dose_provider_refusal(options: &SmoothedDoseOptions) -> Option<MenuRefusal> {
    let error = check_learners(options).and_then(|()| check_option_bounds(options)).err()?;
    Some(match error {
        EstimationError::Refused { code, message } => MenuRefusal {
            code: code.into(),
            detail: message.split(':').next().map(str::to_owned),
            reason: message,
        },
        other => {
            menu_refusal(antecedent_core::reason_code!("invalid_argument"), None, other.to_string())
        }
    })
}

/// The estimator menu for a smoothed dose-response transport query: inspection only,
/// nothing is fitted and nothing is recommended.
///
/// For the smoothed estimator every requirement is computed from the certificate, the
/// query (grid, bandwidth, kernel, support) and the options in force (release defaults
/// when `options` is `None`, and nuisance tasks then say so); the refused alternatives
/// list their fixed descriptions in `static_fields`.
#[must_use]
#[allow(clippy::too_many_lines, reason = "one literal entry per named estimator")]
pub fn smoothed_dose_estimator_menu(
    id: &TransportIdentification,
    query: &SmoothedDoseTransportQuery,
    options: Option<&SmoothedDoseOptions>,
    sampling: Option<TrialSampling>,
) -> EstimatorMenu {
    let defaults = SmoothedDoseOptions::default();
    let defaulted = options.is_none();
    let options = options.unwrap_or(&defaults);
    let structural = dose_structural_refusal(id, query);
    let refusal = structural.clone().or_else(|| dose_provider_refusal(options));
    let (lo, hi) = query.dose_support;
    let h = query.bandwidth;
    let (laws, conditions) = match id {
        TransportIdentification::Transportable { formula, certificate } => {
            let over = match formula {
                TransportFormula::Standardize { over, .. } => coordinates(over),
                _ => "none".into(),
            };
            let mut conditions = vec![
                format!("certificate rule '{}'", certificate.rule),
                format!("selection acts on [{}] only", coordinates(&certificate.selection_targets)),
                format!("the dose v{} is the single source experiment", query.dose.raw()),
            ];
            conditions.extend(certificate.premises.iter().map(|p| format!("premise: {p}")));
            conditions
                .push(format!("the standardizers [{over}] equal the supplied baseline covariates"));
            (
                vec![
                    format!(
                        "source outcome regression mu(t, x) = E(v{} | X=[{over}], A=t, S=1) on \
                         every window [a - {h}, a + {h}]",
                        query.outcome.raw()
                    ),
                    format!("source-membership probability P(S=1 | X=[{over}])"),
                    format!(
                        "known conditional dose density pi(t | x) of v{}, positive on [{lo}, {hi}]",
                        query.dose.raw()
                    ),
                ],
                conditions,
            )
        }
        _ => (
            owned(&[
                "not derivable: no certificate (outcome regression over the dose, membership \
                     probability and a known dose density would be required)",
            ]),
            owned(&["not derivable: the graph and query carry no certificate"]),
        ),
    };
    let tag = if defaulted { " (default: none requested)" } else { "" };
    let basis = &options.basis;
    let knots = if basis.knots.is_empty() {
        String::new()
    } else {
        format!(", hinge knots {:?}", basis.knots)
    };
    let interactions = if basis.interactions { ", dose-by-covariate interactions" } else { "" };
    let uncertainty = SmoothedDoseUncertainty::for_request(options.bootstrap);
    let detail = uncertainty.detail.as_deref().map_or_else(String::new, |d| format!(" ({d})"));
    let designs = sampling.map_or_else(
        || owned(&["nested_cohort", "independent_samples"]),
        |s| {
            vec![match s {
                TrialSampling::NestedCohort => "nested_cohort".to_owned(),
                TrialSampling::IndependentSamples => "independent_samples".to_owned(),
            }]
        },
    );
    let all = [
        "required_laws",
        "required_graph_conditions",
        "nuisance_tasks",
        "support_requirements",
        "sampling_design",
        "uncertainty_status",
    ]
    .map(String::from)
    .to_vec();
    let refused_entry = |estimator: &str,
                         law: &str,
                         task: &str,
                         support: &str,
                         status: &str,
                         refusal: MenuRefusal| EstimatorMenuEntry {
        estimator: estimator.into(),
        eligible: false,
        required_laws: owned(&[law]),
        required_graph_conditions: owned(&["a certified transport derivation over the dose"]),
        nuisance_tasks: owned(&[task]),
        support_requirements: owned(&[support]),
        sampling_design: owned(&["nested_cohort", "independent_samples"]),
        uncertainty_status: status.into(),
        static_fields: all.clone(),
        refusal: Some(refusal),
    };
    let entries = vec![
        EstimatorMenuEntry {
            estimator: "smoothed_dose_transport_aipw".into(),
            eligible: refusal.is_none(),
            required_laws: laws,
            required_graph_conditions: conditions,
            nuisance_tasks: vec![
                format!(
                    "outcome regression mu(t, x), learner {}, dose basis degree {}{knots}{interactions}, \
                     {}-fold cross-fit on one shared fold assignment{tag}",
                    options.outcome.name(),
                    basis.degree,
                    options.folds
                ),
                format!(
                    "source-membership classification, learner {}, {}-fold cross-fit on one \
                     shared fold assignment{tag}",
                    options.membership.name(),
                    options.folds
                ),
            ],
            support_requirements: vec![
                format!(
                    "every grid window [a - {h}, a + {h}] inside the dose support [{lo}, {hi}] \
                     ({} grid doses, {} kernel)",
                    query.grid.len(),
                    query.kernel.name()
                ),
                format!(
                    "out-of-fold membership probability at least {} on every row",
                    options.min_membership_probability
                ),
                format!(
                    "known dose density at least {} on every source row",
                    options.min_dose_density
                ),
                format!(
                    "local dose effective sample size at least {} and at least {} distinct doses \
                     in every window",
                    options.min_local_ess, options.min_distinct_doses
                ),
                format!(
                    "{}-node and {}-node quadratures agree within {} at every grid dose",
                    options.quadrature_nodes,
                    2 * options.quadrature_nodes,
                    options.quadrature_tolerance
                ),
            ],
            sampling_design: designs,
            uncertainty_status: format!(
                "{}: {}{detail}; {} bootstrap replicates requested; the pointwise joint outer \
                 bootstrap interval for psi_h is uncalibrated and its route is closed \
                 (cell_not_licensed)",
                uncertainty.status, uncertainty.reason, options.bootstrap
            ),
            static_fields: vec![],
            refusal,
        },
        refused_entry(
            "kennedy_local_linear_point_curve",
            "the unsmoothed point curve E_target[mu(a, X)] (h -> 0)",
            "cross-fitted pseudo-outcome local-linear regression with an estimated dose density",
            "local positivity at every evaluation dose",
            "not pathwise differentiable: bandwidth error enters the interval; not licensed",
            menu_refusal(
                antecedent_core::reason_code!("route_not_supported"),
                Some("dose_response.target_not_smoothed"),
                "the point curve is not the fixed-bandwidth smoothed response psi_h",
            ),
        ),
        refused_entry(
            "generalized_propensity_estimated_density",
            "an estimated conditional dose density (generalized propensity score)",
            "conditional density regression of the dose on the covariates",
            "estimated density bounded away from zero on every window",
            "not licensed: a second nuisance with its own calibration surface",
            menu_refusal(
                antecedent_core::reason_code!("route_not_supported"),
                Some("dose_response.estimated_dose_density"),
                "the observational estimated-density path is not part of this cell",
            ),
        ),
        refused_entry(
            "dose_response_simultaneous_band",
            "the smoothed response jointly over the grid, or conditionally on covariates",
            "joint multiplier bootstrap over the grid",
            "uniform overlap over the grid",
            "not licensed: no simultaneous or conditional claim",
            menu_refusal(
                antecedent_core::reason_code!("route_not_supported"),
                Some("dose_response.cate_or_simultaneous"),
                "simultaneous bands and conditional targets are not licensed in this cell",
            ),
        ),
        refused_entry(
            "conditional_group_dose_response",
            "group-level exchangeability of the dose within one population",
            "none: the dose density is supplied and group means are kernel-weighted",
            "local support in every group-dose cell",
            "the 2.1 conditional continuous-dose response; not calibrated by this cell",
            menu_refusal(
                antecedent_core::reason_code!("route_not_supported"),
                None,
                "the conditional continuous-dose response does not transport to a target population",
            ),
        ),
    ];
    EstimatorMenu { selection: "manual".into(), entries }
}

/// One bootstrap replicate's rows and fold labels: row `j` of the draw is input row
/// `rows[j]` and keeps that row's fold label from the point run, so a row drawn twice
/// sits in one fold twice (never on both sides of a cross-fit split). The labels are not
/// recomputed from the draw.
#[cfg(feature = "calibration-internal")]
#[must_use]
#[doc(hidden)]
pub fn smoothed_dose_replicate(
    input: &SmoothedDoseInput,
    point_folds: &[u16],
    rows: &[usize],
) -> (SmoothedDoseInput, Vec<u16>) {
    let pick = |values: &[f64]| rows.iter().map(|i| values[*i]).collect::<Vec<_>>();
    let draw = SmoothedDoseInput {
        features: input.features.clone(),
        covariates: input.covariates.iter().map(|col| pick(col)).collect(),
        outcome: pick(&input.outcome),
        dose: pick(&input.dose),
        dose_density: pick(&input.dose_density),
        source: rows.iter().map(|i| input.source[*i]).collect(),
        sampling: input.sampling,
    };
    (draw, rows.iter().map(|i| point_folds[*i]).collect())
}

/// The internal interval run the calibration harness measures.
#[cfg(feature = "calibration-internal")]
#[derive(Clone, Debug)]
#[doc(hidden)]
pub struct SmoothedDoseIntervalRun {
    /// The point run.
    pub point: SmoothedDoseEstimate,
    /// Pointwise percentile interval per grid dose, withheld when any replicate fails.
    pub intervals: Vec<Option<(f64, f64)>>,
    /// Why no interval was produced.
    pub uncertainty_reason: Option<String>,
    /// Successful replicate estimates (one per grid dose), in replicate order.
    pub replicates: Vec<(u32, Vec<f64>)>,
    /// Failed replicate count; failures withhold every interval.
    pub failures: u32,
}

/// The internal estimator the calibration harness measures: the pointwise joint outer
/// refit percentile bootstrap of the whole cross-fitted composed estimator, grouped per
/// design. A replicate reuses the point run's fold label of every resampled row and
/// runs the whole fit, integration and every refusal; any failed replicate withholds
/// every interval (strict replicate policy).
///
/// The public route never calls this; it stays closed until the coverage records exist.
/// It is compiled only with the `calibration-internal` feature, which only
/// dev-dependencies enable.
///
/// # Errors
/// As [`estimate_smoothed_dose`]; and `estimator_inference_mismatch`
/// (`dose_response.bootstrap_below_floor`) below the replicate floor.
#[cfg(feature = "calibration-internal")]
#[doc(hidden)]
pub fn smoothed_dose_interval_internal(
    id: &TransportIdentification,
    query: &SmoothedDoseTransportQuery,
    input: &SmoothedDoseInput,
    options: &SmoothedDoseOptions,
    ctx: &ExecutionContext,
) -> Result<SmoothedDoseIntervalRun, EstimationError> {
    validate_smoothed_dose(id, query, input, options)?;
    if options.bootstrap < SMOOTHED_DOSE_MIN_BOOTSTRAP {
        return Err(refuse(
            antecedent_core::reason_code!("estimator_inference_mismatch"),
            "dose_response.bootstrap_below_floor",
            &format!(
                "{} replicates is below the floor of {SMOOTHED_DOSE_MIN_BOOTSTRAP}",
                options.bootstrap
            ),
        ));
    }
    let point = estimate_smoothed_dose(id, query, input, options, ctx)?;
    // Replicates run concurrently, each on a serial inner context: the cap covers one
    // replicate workspace per concurrent replicate.
    let concurrent =
        (ctx.parallelism.max_threads.get() as usize).clamp(1, options.bootstrap as usize);
    let replicate_bytes =
        smoothed_dose_workspace_bytes(query.grid.len(), input, options, &ctx.serial_inner(), true)
            .and_then(|b| b.checked_mul(concurrent as u64));
    check_workspace(replicate_bytes, ctx)?;
    let groups = smoothed_dose_bootstrap_groups(input);
    let folds = &point.folds.assignment;
    let outcomes = ctx.map_indexed(
        options.bootstrap as usize,
        |index, inner| -> Result<_, EstimationError> {
            cancelled(inner)?;
            let replicate = u32::try_from(index).unwrap_or(u32::MAX);
            let rows = crate::learned_trial::bootstrap_rows(&groups, replicate, inner)?;
            let (draw, draw_folds) = smoothed_dose_replicate(input, folds, &rows);
            let run = validate_smoothed_dose(id, query, &draw, options)
                .and_then(|()| fit_and_evaluate(query, &draw, options, &draw_folds, inner));
            match run {
                Ok((_, evaluation)) => Ok((
                    replicate,
                    Some(evaluation.grid.iter().map(|p| p.estimate).collect::<Vec<_>>()),
                )),
                Err(err) => {
                    if std::env::var_os("ANTECEDENT_CALIBRATION_DIAGNOSTIC").is_some() {
                        eprintln!("smoothed dose bootstrap replicate {replicate}: {err}");
                    }
                    cancelled(inner)?;
                    Ok((replicate, None))
                }
            }
        },
    )?;
    let mut replicates = Vec::new();
    let mut failures = 0u32;
    for (replicate, values) in outcomes {
        match values {
            Some(values) => replicates.push((replicate, values)),
            None => failures += 1,
        }
    }
    let mut intervals = Vec::with_capacity(query.grid.len());
    let mut reason = None;
    for g in 0..query.grid.len() {
        let values: Vec<f64> = replicates.iter().map(|(_, v)| v[g]).collect();
        let (interval, why) = crate::learned_trial::learned_trial_uncertainty(
            &values,
            failures,
            options.bootstrap,
            options.coverage_level,
        );
        intervals.push(interval);
        reason = reason.or(why.map(str::to_owned));
    }
    Ok(SmoothedDoseIntervalRun {
        point,
        intervals,
        uncertainty_reason: reason,
        replicates,
        failures,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(n: usize) -> SmoothedDoseInput {
        let covariate: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.37).sin()).collect();
        SmoothedDoseInput {
            features: vec![0],
            covariates: vec![covariate],
            outcome: (0..n).map(|i| (i as f64 * 0.11).cos()).collect(),
            dose: (0..n).map(|i| 4.0 * ((i as f64 * 0.618_033_988_7).fract())).collect(),
            dose_density: vec![0.25; n],
            source: (0..n).map(|i| i % 3 != 0).collect(),
            sampling: TrialSampling::IndependentSamples,
        }
    }

    use antecedent_stats::special::gauss_legendre;

    #[test]
    fn requests_report_their_interval_status() {
        let none = SmoothedDoseUncertainty::for_request(0);
        assert_eq!(
            (none.status.as_str(), none.reason.as_str()),
            ("point_only", "no_interval_requested")
        );
        let low = SmoothedDoseUncertainty::for_request(50);
        assert_eq!(low.detail.as_deref(), Some("dose_response.bootstrap_below_floor"));
        let closed = SmoothedDoseUncertainty::for_request(199);
        assert_eq!(
            (closed.status.as_str(), closed.reason.as_str()),
            ("withheld", "cell_not_licensed")
        );
        assert!(!closed.available());
    }

    /// Cancellation raised by the predictor on its third quadrature chunk stops the
    /// bounded loop at the next poll: no later chunk is predicted.
    #[test]
    fn cancellation_mid_quadrature_stops_at_the_next_chunk() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let data = input(3 * 4096);
        let basis = DoseBasis::default();
        let folds = smoothed_dose_fold_assignment(&data, 2);
        let target: Vec<usize> = (0..data.source.len()).filter(|i| !data.source[*i]).collect();
        let (nodes, weights) = antecedent_stats::special::gauss_legendre(16);
        // 4096 target rows over two folds at 16 nodes: 1024 rows per chunk, four chunks.
        let chunks = 2 * target.len().div_ceil(2).div_ceil(CHUNK_DESIGN_ROWS / 16);
        assert_eq!(chunks, 4);
        let run = |cancel_at: usize| {
            let ctx = ExecutionContext::for_tests(1);
            let calls = AtomicUsize::new(0);
            let token = ctx.cancellation.clone();
            let predictor = |_: usize, _: &[f64], rows: usize, _: &ExecutionContext| {
                if calls.fetch_add(1, Ordering::SeqCst) + 1 == cancel_at {
                    token.cancel();
                }
                Ok(vec![1.0; rows])
            };
            let result = smoothed_target_rows(
                &basis,
                &data,
                SmoothingKernel::Epanechnikov,
                &folds,
                (2, &predictor),
                (2.0, 0.5),
                (&nodes, &weights),
                &target,
                &ctx,
            );
            (result, calls.load(Ordering::SeqCst))
        };
        // Uncancelled, every chunk runs and the constant curve integrates to one.
        let (full, calls) = run(usize::MAX);
        assert_eq!(calls, chunks);
        assert!(full.unwrap().iter().all(|v| (v - 1.0).abs() < 1e-12));
        // Cancelled on the third chunk: the fourth is never predicted.
        let (stopped, calls) = run(3);
        assert_eq!(calls, 3);
        assert!(matches!(
            stopped.unwrap_err(),
            EstimationError::Refused { code: "transport_budget_cancel", .. }
        ));
    }

    /// `integral_{-1}^{1} K(u) (u - c)_+ du` for the Epanechnikov kernel, `c` in `[-1, 1]`.
    fn hinge_moment(c: f64) -> f64 {
        0.75 * (0.25 - 2.0 * c / 3.0 + c * c / 2.0 - c.powi(4) / 12.0)
    }

    fn hinge_rule(nodes: &[f64], weights: &[f64], c: f64) -> f64 {
        nodes
            .iter()
            .zip(weights)
            .map(|(u, w)| w * SmoothingKernel::Epanechnikov.density(*u) * (u - c).max(0.0))
            .sum()
    }

    /// What this shows: across a sweep of hinge-knot positions, the rule split at the knot
    /// integrates `K(u) (u - c)_+` exactly for 16 and 32 nodes, while the unsplit rule's
    /// doubling difference `|I_Q - I_2Q|` falls below the finer rule's own true error at a
    /// visible fraction of positions: an estimate, not a bound, which is why a
    /// piecewise-polynomial fit is split rather than gated. The counts asserted here are
    /// the ones `docs/architecture/smoothed-dose-response-transport.md` and the promotion record cite:
    /// 1440 (Q = 16) and 1368 (Q = 32) of 20,000 positions, and 56 and 76 positions where
    /// the estimate passes a `1e-6` tolerance that the true error exceeds.
    #[test]
    #[allow(clippy::cast_precision_loss, reason = "sweep indices are small")]
    fn splitting_at_the_knot_makes_the_hinge_exact_where_the_doubling_estimate_misses() {
        let positions = 20_000usize;
        for (q, cited_misses, cited_dangerous) in [(16usize, 1440usize, 56usize), (32, 1368, 76)] {
            let (base, fine) = (gauss_legendre(q), gauss_legendre(2 * q));
            let (mut worst_split, mut misses, mut dangerous) = (0.0_f64, 0usize, 0usize);
            for i in 0..positions {
                let c = -1.0 + 2.0 * (i as f64 + 0.5) / positions as f64;
                let exact = hinge_moment(c);
                for rule in [&base, &fine] {
                    let (u, w) = split_rule(&[c], (0.0, 1.0), (&rule.0, &rule.1));
                    assert_eq!(u.len(), 2 * rule.0.len());
                    worst_split = worst_split.max((hinge_rule(&u, &w, c) - exact).abs());
                }
                let coarse = hinge_rule(&base.0, &base.1, c);
                let check = hinge_rule(&fine.0, &fine.1, c);
                let (estimate, truth) = ((coarse - check).abs(), (check - exact).abs());
                if estimate < truth {
                    misses += 1;
                }
                if estimate <= 1e-6 && truth > 1e-6 {
                    dangerous += 1;
                }
            }
            eprintln!(
                "Q = {q}: split worst error {worst_split:e}; unsplit |I_Q - I_2Q| below the \
                 2Q rule's true error at {misses}/{positions} knot positions, passing a 1e-6 \
                 tolerance while the true error exceeds it at {dangerous}"
            );
            assert!(worst_split < 1e-14, "Q = {q}: {worst_split:e}");
            assert!(misses * 100 > positions, "Q = {q}: {misses}");
            // The sweep is deterministic: the cited counts are the measured ones.
            assert_eq!((misses, dangerous), (cited_misses, cited_dangerous), "Q = {q}");
        }
        // Without a knot inside the window the split rule is the base rule, bit for bit.
        let base = gauss_legendre(16);
        let (u, w) = split_rule(&[2.5, -3.0], (0.0, 1.0), (&base.0, &base.1));
        assert!(u.iter().zip(&base.0).all(|(a, b)| a.to_bits() == b.to_bits()));
        assert!(w.iter().zip(&base.1).all(|(a, b)| a.to_bits() == b.to_bits()));
        // Knots map through the window: (k - a) / h.
        let (u, _) = split_rule(&[2.1, 2.1], (2.0, 0.5), (&base.0, &base.1));
        assert_eq!(u.len(), 32);
        assert!(u[..16].iter().all(|x| *x < 0.2) && u[16..].iter().all(|x| *x > 0.2));
    }

    /// What this shows: the workspace cap is mandatory, a hard limit only lowers it, the
    /// estimate covers the designs actually allocated, and a request at every declared cap
    /// is far above the default cap.
    #[test]
    fn the_workspace_cap_is_mandatory_and_a_hard_limit_only_lowers_it() {
        let basis =
            DoseBasis { degree: 3, knots: (1..=8).map(f64::from).collect(), interactions: true };
        let width = basis.width(256);
        assert_eq!(width, 3084);
        let at_caps = WorkspaceShape {
            rows: 200_000,
            source: 120_000,
            features: 256,
            width,
            folds: 20,
            grid: 16,
            nodes_per_row: 64 * 9,
            threads: 1,
        };
        let producer = workspace(at_caps, true).unwrap();
        let replay = workspace(at_caps, false).unwrap();
        // The producer's outcome design of the source rows alone is 2.96 GB; a replay
        // allocates one fold's outcome design at a time.
        assert!(producer >= 8 * 120_000 * 3084 && replay >= 8 * 6000 * 3084);
        assert!(producer > replay && replay > SMOOTHED_DOSE_MAX_WORKSPACE_BYTES);
        assert!(workspace(WorkspaceShape { threads: 4, ..at_caps }, true).unwrap() > producer);
        // A wide design gets fewer rows per chunk: at most 8 MiB of design.
        assert!(chunk_design_rows(width) * width * 8 <= CHUNK_DESIGN_BYTES);
        assert_eq!(chunk_design_rows(6), CHUNK_DESIGN_ROWS);
        let cap = SMOOTHED_DOSE_MAX_WORKSPACE_BYTES;
        let refused = |r: Result<(), EstimationError>| {
            matches!(r, Err(EstimationError::Refused { code: "transport_budget_cancel", message })
                if message.contains("memory cap") && !message.contains("cancel"))
        };
        let unlimited = ExecutionContext::for_tests(0);
        assert!(refused(check_workspace(Some(producer), &unlimited)));
        assert!(check_workspace(Some(cap), &unlimited).is_ok());
        assert!(refused(check_workspace(Some(cap + 1), &unlimited)));
        assert!(refused(check_workspace(None, &unlimited)));
        let mut raised = ExecutionContext::for_tests(0);
        raised.memory.hard_limit_bytes = Some(u64::MAX);
        assert!(refused(check_workspace(Some(cap + 1), &raised)));
        let mut lowered = ExecutionContext::for_tests(0);
        lowered.memory.hard_limit_bytes = Some(4096);
        assert!(check_workspace(Some(4096), &lowered).is_ok());
        assert!(refused(check_workspace(Some(4097), &lowered)));
    }

    #[test]
    fn the_fold_assignment_deals_each_role_round_robin_and_ignores_values() {
        let data = input(30);
        let folds = smoothed_dose_fold_assignment(&data, 3);
        for fold in 0..3u16 {
            assert!((0..30).any(|i| data.source[i] && folds[i] == fold));
            assert!((0..30).any(|i| !data.source[i] && folds[i] == fold));
        }
        let mut edited = data.clone();
        edited.outcome.iter_mut().for_each(|y| *y += 5.0);
        edited.dose.reverse();
        edited.covariates[0].reverse();
        assert_eq!(smoothed_dose_fold_assignment(&edited, 3), folds);
    }
}
