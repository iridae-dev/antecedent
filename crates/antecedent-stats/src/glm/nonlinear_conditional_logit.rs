//! Conditional logit whose option value is nonlinear in some of its parameters.
//!
//! Each choice set holds two or more options and one chosen option. The probability of option
//! `i` of a set is proportional to `exp(lambda * V_i)` with
//!
//! `V_i = F_i(theta_S) + sum_k theta_k T_ik(theta_S)`,
//!
//! where `theta_S` are the shape parameters (each ranging over a declared bounded interval),
//! `theta_k` the linear parameters and `lambda` the scale of the value (choice consistency). For
//! fixed `theta_S` the model is a conditional logit linear in `beta = (lambda, lambda theta)` on
//! the features `(F_i, T_i1, ..., T_iK)`, so the profile log-likelihood `l_p(theta_S)` is the
//! maximum of an ordinary conditional logit fit ([`super::fit_conditional_logit`]).
//!
//! Search. The profile is evaluated at the centre of the declared box and at the first points of a
//! Sobol sequence over it (a deterministic low-discrepancy sequence), then refined by a compass
//! search inside the box from the best point, then polished by Fisher scoring in the direct
//! parameters `psi = (lambda, theta_L, theta_S)` with the shape parameters kept inside their
//! bounds. Columns of an inner design that are linearly dependent at some shape point do not
//! change the maximum of the likelihood there and are dropped for that point.
//!
//! Information. With `g_i = d(lambda V_i)/d psi` (first derivatives of the value only), the
//! information is `sum_s sum_i p_i (g_i - gbar_s)(g_i - gbar_s)'`, the expected information of a
//! multinomial choice with a nonlinear linear predictor; its inverse is the model covariance, and
//! the sandwich with per-cluster score sums, times `C / (C - 1)` for `C` clusters (the
//! finite-cluster correction of Cameron and Miller, 2015), is the cluster covariance. A null
//! direction of the information (pivoted Cholesky with a relative tolerance) is refused, naming
//! the parameters in it.
//!
//! Convergence. The polish is Fisher scoring; where the full scoring step is not accepted (the
//! expected information omits the curvature of the utility's second derivatives, so it can
//! vanish in a shape direction where the utility's shape derivative is zero), Newton's step with
//! the observed information, from central differences of the analytic score, is tried as well
//! and the step reaching the higher log-likelihood is taken. The polish must end at a point whose bound-projected score passes the inner
//! fits' convergence test (or where a further accepted step no longer moves any coordinate);
//! otherwise the fit is refused as not converged. The profiles are then maximised from the
//! polished point: when a profile value exceeds the polished log-likelihood by more than
//! `1e-6 (1 + |l|)`, the search and the polish stopped at a lower maximum, so the polish is
//! started once more from the best profile point, and the fit is refused as not converged if a
//! profile still lies above it. The estimate, its covariance and its intervals are therefore
//! read at the maximum the profiles show.
//!
//! Identification of the shape parameters. For each shape parameter the profile over its declared
//! range (maximising over the other shape parameters) is evaluated on an even grid. The fit is
//! refused when the profile varies by less than `chi2_1(level) / 2` over the whole range (the
//! linear estimates would rest on an arbitrary shape value), when two profile maxima within that
//! distance of the overall maximum are separated by a dip deeper than it (the likelihood-ratio set
//! is not an interval), and when the estimate lies on a declared bound (the normal approximation
//! does not hold there; the likelihood-ratio interval truncated at the bound is reported in the
//! refusal). Otherwise each shape parameter carries the profile likelihood-ratio interval
//! `{v : 2 (l_max - l_p(v)) <= chi2_1(level)}`, its ends located by bisection between grid
//! points.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use antecedent_core::ExecutionContext;
use antecedent_kernels::norm_inv;
use thiserror::Error;

use super::conditional_logit::{
    ConditionalLogitDesign, ConditionalLogitError, ConditionalLogitOptions, fit_conditional_logit,
    fit_conditional_logit_with_context,
};
use crate::gram::{invert_square, solve_spd};

/// The utility of the options of every choice set as a function of the shape parameters.
pub trait ChoiceUtility {
    /// Number of linear parameters `theta_L`.
    fn n_linear(&self) -> usize;
    /// Number of shape parameters `theta_S`.
    fn n_shape(&self) -> usize;
    /// Set boundaries: set `s` owns option rows `set_starts[s]..set_starts[s + 1]`.
    fn set_starts(&self) -> &[usize];
    /// Chosen option of each set, as an index within the set.
    fn chosen(&self) -> &[usize];
    /// Cluster label of each set, when the sets are clustered.
    fn clusters(&self) -> Option<&[u32]>;
    /// Row-major `rows x (1 + n_linear)`: column 0 is `F_i(theta_S)`, column `1 + k` is
    /// `T_ik(theta_S)`.
    ///
    /// # Errors
    /// The utility is not defined at `shape`.
    fn features(&self, shape: &[f64]) -> Result<Vec<f64>, UtilityNotEvaluable>;
    /// Row-major `rows x (1 + n_linear) x n_shape`: the derivative of each feature column with
    /// respect to each shape parameter.
    ///
    /// # Errors
    /// The utility is not defined at `shape`.
    fn shape_jacobian(&self, shape: &[f64]) -> Result<Vec<f64>, UtilityNotEvaluable>;
}

/// The utility could not be evaluated at a shape point.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct UtilityNotEvaluable;

/// Controls of the search over the shape parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShapeSearchOptions {
    /// Points at which the profile is evaluated before refinement: the box centre and then the
    /// Sobol sequence; at least one.
    pub grid_points: usize,
    /// Largest number of shape parameters searched; at most [`SOBOL_DIMENSIONS`].
    pub max_dimensions: usize,
    /// Points of the even grid on which each shape parameter's profile is evaluated; at least
    /// three.
    pub profile_points: usize,
    /// Largest number of profile evaluations of one compass search; at least one.
    pub max_refinements: usize,
}

impl Default for ShapeSearchOptions {
    fn default() -> Self {
        Self { grid_points: 64, max_dimensions: 3, profile_points: 33, max_refinements: 200 }
    }
}

/// Largest number of shape parameters the built-in Sobol directions cover.
pub const SOBOL_DIMENSIONS: usize = 8;

/// Controls of [`fit_nonlinear_conditional_logit`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NonlinearConditionalLogitOptions {
    /// Controls of every inner conditional logit fit and of the Fisher-scoring polish (its
    /// iteration limit and gradient tolerance).
    pub fit: ConditionalLogitOptions,
    /// Controls of the shape search.
    pub search: ShapeSearchOptions,
    /// Level of the profile likelihood-ratio intervals and of the identification thresholds;
    /// strictly between 0 and 1.
    pub level: f64,
}

impl Default for NonlinearConditionalLogitOptions {
    fn default() -> Self {
        Self {
            fit: ConditionalLogitOptions::default(),
            search: ShapeSearchOptions::default(),
            level: 0.95,
        }
    }
}

/// The profile of one shape parameter over its declared range.
#[derive(Clone, Debug, PartialEq)]
pub struct ShapeProfile {
    /// Values at which the profile was evaluated, ascending: the even grid and the estimate.
    pub values: Vec<f64>,
    /// Profile log-likelihood at each value (maximised over every other parameter).
    pub log_likelihood: Vec<f64>,
    /// Profile likelihood-ratio interval at the options' level.
    pub interval: (f64, f64),
    /// The interval reaches the declared lower bound, where it is cut.
    pub lower_truncated: bool,
    /// The interval reaches the declared upper bound, where it is cut.
    pub upper_truncated: bool,
}

/// Maximum-likelihood fit of a conditional logit with shape parameters.
#[derive(Clone, Debug, PartialEq)]
pub struct NonlinearConditionalLogitFit {
    /// Choice consistency `lambda`.
    pub consistency: f64,
    /// Linear parameters `theta_L`.
    pub linear: Vec<f64>,
    /// Shape parameters `theta_S`.
    pub shape: Vec<f64>,
    /// Log-likelihood at the estimate.
    pub log_likelihood: f64,
    /// Log-likelihood with every option of a set equally likely.
    pub null_log_likelihood: f64,
    /// Expected information in `psi = (lambda, theta_L, theta_S)`, row-major.
    pub information: Vec<f64>,
    /// Inverse of `information`, row-major.
    pub covariance: Vec<f64>,
    /// Cluster sandwich covariance of `psi`, present when clusters were given.
    pub cluster_covariance: Option<Vec<f64>>,
    /// One profile per shape parameter.
    pub profiles: Vec<ShapeProfile>,
    /// Profile evaluations (inner fits) made by the search and the profiles.
    pub profile_evaluations: usize,
}

/// Why a nonlinear conditional logit fit was refused.
#[derive(Clone, Debug, PartialEq, Error)]
pub enum NonlinearChoiceError {
    /// A search control or the level is out of range.
    #[error("search controls or level out of range")]
    InvalidOptions,
    /// The bounds do not hold one finite, increasing interval per shape parameter.
    #[error("bounds of shape parameter {dimension} are invalid")]
    BoundsInvalid {
        /// Shape parameter index; the number of shape parameters when the count differs.
        dimension: usize,
    },
    /// More shape parameters than the search covers.
    #[error("{dimensions} shape parameters exceed the search limit of {limit}")]
    ShapeSearchBudget {
        /// Shape parameters to search.
        dimensions: usize,
        /// The limit.
        limit: usize,
    },
    /// The design or an inner fit was refused for a reason of its own.
    #[error("inner conditional logit refused: {0}")]
    Design(ConditionalLogitError),
    /// The utility could not be evaluated at a shape point of the search.
    #[error("the utility is not defined at shape point {shape:?}")]
    UtilityNotEvaluable {
        /// The shape point.
        shape: Vec<f64>,
    },
    /// The choices are separated at a shape point: the likelihood has no finite maximum.
    #[error("choices are separated at shape point {shape:?}")]
    Separated {
        /// The shape point.
        shape: Vec<f64>,
    },
    /// An inner fit did not converge.
    #[error("no convergence after {iterations} iterations; gradient norm {gradient_norm}")]
    NotConverged {
        /// Accepted Newton steps.
        iterations: usize,
        /// Largest absolute gradient entry at the last iterate.
        gradient_norm: f64,
    },
    /// The estimated consistency is not positive.
    #[error("estimated consistency {consistency} is not positive")]
    ConsistencyNotPositive {
        /// The estimate.
        consistency: f64,
    },
    /// The profile of a shape parameter varies by less than `chi2_1(level) / 2` over its range.
    #[error("shape parameter {dimension} is not identified (profile range {profile_range})")]
    ShapeNotIdentified {
        /// Shape parameter index.
        dimension: usize,
        /// Largest minus smallest profile log-likelihood over the range.
        profile_range: f64,
    },
    /// Two profile maxima close to the overall maximum are separated by a deeper dip.
    #[error("the profile of shape parameter {dimension} has two maxima {modes:?}")]
    ProfileMultimodal {
        /// Shape parameter index.
        dimension: usize,
        /// The two maximising values.
        modes: (f64, f64),
    },
    /// The estimate of a shape parameter lies on a declared bound.
    #[error("shape parameter {dimension} is estimated on a bound ({estimate})")]
    ShapeEstimateAtBound {
        /// Shape parameter index.
        dimension: usize,
        /// The estimate.
        estimate: f64,
        /// The profile likelihood-ratio interval, truncated at the bound.
        interval: (f64, f64),
    },
    /// The information at the maximum has a null direction.
    #[error("information at the maximum is singular; dependent parameters {columns:?}")]
    RankDeficient {
        /// Parameter indices in `psi = (lambda, theta_L, theta_S)` taking part, ascending.
        columns: Vec<usize>,
    },
    /// The context's cancellation token was set.
    #[error("cancelled")]
    Cancelled,
}

impl NonlinearChoiceError {
    /// Stable reason id in the `nonlinear_conditional_logit` family.
    #[must_use]
    pub fn detail(&self) -> &'static str {
        match self {
            Self::InvalidOptions => "nonlinear_conditional_logit.invalid_options",
            Self::BoundsInvalid { .. } => "nonlinear_conditional_logit.bounds_invalid",
            Self::ShapeSearchBudget { .. } => "nonlinear_conditional_logit.shape_search_budget",
            Self::Design(inner) => inner.detail(),
            Self::UtilityNotEvaluable { .. } => "nonlinear_conditional_logit.utility_not_evaluable",
            Self::Separated { .. } => "nonlinear_conditional_logit.separated",
            Self::NotConverged { .. } => "nonlinear_conditional_logit.not_converged",
            Self::ConsistencyNotPositive { .. } => {
                "nonlinear_conditional_logit.consistency_not_positive"
            }
            Self::ShapeNotIdentified { .. } => "nonlinear_conditional_logit.shape_not_identified",
            Self::ProfileMultimodal { .. } => "nonlinear_conditional_logit.profile_multimodal",
            Self::ShapeEstimateAtBound { .. } => {
                "nonlinear_conditional_logit.shape_estimate_at_bound"
            }
            Self::RankDeficient { .. } => "nonlinear_conditional_logit.rank_deficient",
            Self::Cancelled => "nonlinear_conditional_logit.cancelled",
        }
    }
}

/// A shape estimate closer to a bound than this fraction of the declared width is on the bound.
const BOUND_FRACTION: f64 = 1e-6;
/// A compass step smaller than this fraction of the declared width ends the search.
const STEP_FRACTION: f64 = 1e-9;
/// Bisection of an interval end stops below this fraction of the declared width.
const BISECTION_FRACTION: f64 = 1e-9;
/// A profile value above the polished log-likelihood by more than this fraction of
/// `1 + |log-likelihood|` shows the polish stopped below the maximum.
const PROFILE_EXCESS_RELATIVE: f64 = 1e-6;
/// A profile log-likelihood with the shape point attaining it.
type ProfileTop = (f64, Vec<f64>);
/// Relative step of the central differences of the score that give the observed information.
const OBSERVED_STEP: f64 = 1e-6;
/// Largest number of bisection steps for one interval end.
const MAX_BISECTIONS: usize = 60;
/// Pivots of the information below this fraction of its largest diagonal are null directions.
const PIVOT_RELATIVE: f64 = 1e-10;
/// Step halvings of one Fisher-scoring step.
const MAX_HALVINGS: usize = 40;

/// Sobol direction parameters `(degree, coefficients, initial m)` for dimensions 2 to 8.
const SOBOL_PARAMETERS: [(usize, u32, [u32; 5]); 7] = [
    (1, 0, [1, 0, 0, 0, 0]),
    (2, 1, [1, 3, 0, 0, 0]),
    (3, 1, [1, 3, 1, 0, 0]),
    (3, 2, [1, 1, 1, 0, 0]),
    (4, 1, [1, 1, 3, 3, 0]),
    (4, 4, [1, 3, 5, 13, 0]),
    (5, 2, [1, 1, 5, 5, 17]),
];

/// Thirty-two direction numbers of Sobol dimension `dimension` (0-based).
fn sobol_directions(dimension: usize) -> [u32; 32] {
    let mut v = [0_u32; 32];
    if dimension == 0 {
        for (i, slot) in v.iter_mut().enumerate() {
            *slot = 1_u32 << (31 - i);
        }
        return v;
    }
    let (s, a, m) = SOBOL_PARAMETERS[(dimension - 1).min(SOBOL_PARAMETERS.len() - 1)];
    for i in 0..32 {
        if i < s {
            v[i] = m[i] << (31 - i);
        } else {
            let mut value = v[i - s] ^ (v[i - s] >> s);
            for k in 1..s {
                if (a >> (s - 1 - k)) & 1 == 1 {
                    value ^= v[i - k];
                }
            }
            v[i] = value;
        }
    }
    v
}

/// Point `index` of the Sobol sequence in `[0, 1)^d`.
fn sobol_point(directions: &[[u32; 32]], index: u32) -> Vec<f64> {
    directions
        .iter()
        .map(|v| {
            let mut x = 0_u32;
            for (bit, direction) in v.iter().enumerate() {
                if (index >> bit) & 1 == 1 {
                    x ^= direction;
                }
            }
            f64::from(x) / 4_294_967_296.0
        })
        .collect()
}

/// The profile at one shape point.
struct ProfilePoint {
    log_likelihood: f64,
    /// `beta = (lambda, lambda theta_L)`; zero where a dependent column was dropped.
    beta: Vec<f64>,
}

/// Log-likelihood and derivatives in `psi`.
struct PsiEvaluation {
    log_likelihood: f64,
    score: Vec<f64>,
    information: Vec<f64>,
    set_scores: Vec<f64>,
    /// `sum_s max_i |g_i - g_chosen|_inf`, the scale of the convergence test.
    scale: f64,
}

struct Fitter<'a> {
    utility: &'a dyn ChoiceUtility,
    bounds: &'a [(f64, f64)],
    options: NonlinearConditionalLogitOptions,
    ctx: Option<&'a ExecutionContext>,
    /// Rows of the design.
    rows: usize,
    evaluations: std::cell::Cell<usize>,
}

/// Fit a conditional logit whose option value is nonlinear in the shape parameters; see the
/// module documentation for the search, the information and the identification checks.
///
/// `bounds` holds the declared `(lower, upper)` of each shape parameter, `lower < upper`. With no
/// shape parameter the fit is the ordinary conditional logit, reported in the direct parameters.
///
/// # Errors
/// Every [`NonlinearChoiceError`] variant, as documented on each.
pub fn fit_nonlinear_conditional_logit(
    utility: &dyn ChoiceUtility,
    bounds: &[(f64, f64)],
    options: NonlinearConditionalLogitOptions,
    ctx: Option<&ExecutionContext>,
) -> Result<NonlinearConditionalLogitFit, NonlinearChoiceError> {
    let search = options.search;
    if !(options.level.is_finite() && options.level > 0.0 && options.level < 1.0)
        || search.grid_points == 0
        || search.profile_points < 3
        || search.max_refinements == 0
        || search.max_dimensions > SOBOL_DIMENSIONS
    {
        return Err(NonlinearChoiceError::InvalidOptions);
    }
    let n_shape = utility.n_shape();
    if bounds.len() != n_shape {
        return Err(NonlinearChoiceError::BoundsInvalid { dimension: bounds.len() });
    }
    if let Some(dimension) =
        bounds.iter().position(|(lo, hi)| !(lo.is_finite() && hi.is_finite() && lo < hi))
    {
        return Err(NonlinearChoiceError::BoundsInvalid { dimension });
    }
    if n_shape > search.max_dimensions {
        return Err(NonlinearChoiceError::ShapeSearchBudget {
            dimensions: n_shape,
            limit: search.max_dimensions,
        });
    }
    let rows = utility.set_starts().last().copied().unwrap_or(0);
    let fitter =
        Fitter { utility, bounds, options, ctx, rows, evaluations: std::cell::Cell::new(0) };
    fitter.run()
}

/// Columns of `features` (row-major, `p` columns) kept in order.
fn keep_columns(features: &[f64], p: usize, keep: &[usize]) -> Vec<f64> {
    features.chunks_exact(p).flat_map(|row| keep.iter().map(move |&k| row[k])).collect()
}

/// `-sum_s ln(n_s)`.
fn null_log_likelihood(set_starts: &[usize]) -> f64 {
    -set_starts.windows(2).map(|w| ((w[1] - w[0]) as f64).ln()).sum::<f64>()
}

impl Fitter<'_> {
    fn observe(&self) -> Result<(), NonlinearChoiceError> {
        if self.ctx.is_some_and(|ctx| ctx.cancellation.is_cancelled()) {
            return Err(NonlinearChoiceError::Cancelled);
        }
        Ok(())
    }

    fn p_beta(&self) -> usize {
        1 + self.utility.n_linear()
    }

    fn inner_fit(
        &self,
        features: &[f64],
        p: usize,
    ) -> Result<super::conditional_logit::ConditionalLogitFit, ConditionalLogitError> {
        let design = ConditionalLogitDesign {
            features,
            n_features: p,
            set_starts: self.utility.set_starts(),
            chosen: self.utility.chosen(),
            clusters: None,
        };
        match self.ctx {
            Some(ctx) => fit_conditional_logit_with_context(&design, self.options.fit, ctx),
            None => fit_conditional_logit(&design, self.options.fit),
        }
    }

    fn map_inner(error: ConditionalLogitError, shape: &[f64]) -> NonlinearChoiceError {
        match error {
            ConditionalLogitError::Separated { .. } => {
                NonlinearChoiceError::Separated { shape: shape.to_vec() }
            }
            ConditionalLogitError::NotConverged { iterations, gradient_norm } => {
                NonlinearChoiceError::NotConverged { iterations, gradient_norm }
            }
            ConditionalLogitError::Cancelled => NonlinearChoiceError::Cancelled,
            other => NonlinearChoiceError::Design(other),
        }
    }

    /// The profile log-likelihood at `shape`, dependent columns dropped.
    fn profile(&self, shape: &[f64]) -> Result<ProfilePoint, NonlinearChoiceError> {
        self.observe()?;
        self.evaluations.set(self.evaluations.get() + 1);
        let p = self.p_beta();
        let features = self
            .utility
            .features(shape)
            .map_err(|_| NonlinearChoiceError::UtilityNotEvaluable { shape: shape.to_vec() })?;
        match self.inner_fit(&features, p) {
            Ok(fit) => {
                Ok(ProfilePoint { log_likelihood: fit.log_likelihood, beta: fit.coefficients })
            }
            Err(ConditionalLogitError::RankDeficient { columns }) => {
                let keep: Vec<usize> = (0..p).filter(|k| !columns.contains(k)).collect();
                let mut beta = vec![0.0; p];
                if keep.is_empty() {
                    return Ok(ProfilePoint {
                        log_likelihood: null_log_likelihood(self.utility.set_starts()),
                        beta,
                    });
                }
                let reduced = keep_columns(&features, p, &keep);
                let fit = self
                    .inner_fit(&reduced, keep.len())
                    .map_err(|error| Self::map_inner(error, shape))?;
                for (slot, value) in keep.iter().zip(&fit.coefficients) {
                    beta[*slot] = *value;
                }
                Ok(ProfilePoint { log_likelihood: fit.log_likelihood, beta })
            }
            Err(error) => Err(Self::map_inner(error, shape)),
        }
    }

    /// The profile at `shape`, or `None` where the utility is not defined.
    fn profile_or_none(&self, shape: &[f64]) -> Result<Option<ProfilePoint>, NonlinearChoiceError> {
        match self.profile(shape) {
            Ok(point) => Ok(Some(point)),
            Err(NonlinearChoiceError::UtilityNotEvaluable { .. }) => Ok(None),
            Err(error) => Err(error),
        }
    }

    fn clamp(&self, shape: &mut [f64]) {
        for (value, (lo, hi)) in shape.iter_mut().zip(self.bounds) {
            *value = value.clamp(*lo, *hi);
        }
    }

    /// Compass search maximising the profile over `free` dimensions from `start`.
    fn compass(
        &self,
        start: Vec<f64>,
        start_value: f64,
        free: &[usize],
    ) -> Result<(Vec<f64>, f64), NonlinearChoiceError> {
        let mut best = start;
        let mut best_value = start_value;
        let mut steps: Vec<f64> =
            free.iter().map(|&m| 0.25 * (self.bounds[m].1 - self.bounds[m].0)).collect();
        let mut spent = 0;
        while spent < self.options.search.max_refinements {
            let active = free
                .iter()
                .zip(&steps)
                .any(|(&m, step)| *step > STEP_FRACTION * (self.bounds[m].1 - self.bounds[m].0));
            if !active {
                break;
            }
            let mut improved = false;
            'directions: for (position, &m) in free.iter().enumerate() {
                for sign in [1.0, -1.0] {
                    let mut candidate = best.clone();
                    candidate[m] += sign * steps[position];
                    self.clamp(&mut candidate);
                    if candidate[m].total_cmp(&best[m]).is_eq() {
                        continue;
                    }
                    spent += 1;
                    if let Some(point) = self.profile_or_none(&candidate)?
                        && point.log_likelihood > best_value
                    {
                        best = candidate;
                        best_value = point.log_likelihood;
                        improved = true;
                        break 'directions;
                    }
                    if spent >= self.options.search.max_refinements {
                        break 'directions;
                    }
                }
            }
            if !improved {
                for step in &mut steps {
                    *step *= 0.5;
                }
            }
        }
        Ok((best, best_value))
    }

    /// The best shape point of the grid and the compass search.
    fn search(&self) -> Result<Vec<f64>, NonlinearChoiceError> {
        let d = self.bounds.len();
        let centre: Vec<f64> = self.bounds.iter().map(|(lo, hi)| 0.5 * lo + 0.5 * hi).collect();
        let directions: Vec<[u32; 32]> = (0..d).map(sobol_directions).collect();
        let mut best: Option<(Vec<f64>, f64)> = None;
        for index in 0..self.options.search.grid_points {
            let point = if index == 0 {
                centre.clone()
            } else {
                let unit = sobol_point(&directions, u32::try_from(index).unwrap_or(u32::MAX));
                unit.iter().zip(self.bounds).map(|(u, (lo, hi))| lo + u * (hi - lo)).collect()
            };
            if let Some(profile) = self.profile_or_none(&point)?
                && best.as_ref().is_none_or(|(_, value)| profile.log_likelihood > *value)
            {
                best = Some((point, profile.log_likelihood));
            }
        }
        let (start, value) =
            best.ok_or(NonlinearChoiceError::UtilityNotEvaluable { shape: centre })?;
        let free: Vec<usize> = (0..d).collect();
        Ok(self.compass(start, value, &free)?.0)
    }

    /// Log-likelihood, score and expected information in `psi`.
    fn evaluate_psi(&self, psi: &[f64]) -> Result<PsiEvaluation, UtilityNotEvaluable> {
        let n_linear = self.utility.n_linear();
        let n_shape = self.bounds.len();
        let q = 1 + n_linear + n_shape;
        let columns = 1 + n_linear;
        let lambda = psi[0];
        let theta = &psi[1..columns];
        let shape = &psi[columns..];
        let features = self.utility.features(shape)?;
        let jacobian = if n_shape > 0 { self.utility.shape_jacobian(shape)? } else { Vec::new() };
        let mut g = vec![0.0; self.rows * q];
        let mut eta = vec![0.0; self.rows];
        for r in 0..self.rows {
            let row = &features[r * columns..(r + 1) * columns];
            let value = row[0] + theta.iter().zip(&row[1..]).map(|(t, x)| t * x).sum::<f64>();
            eta[r] = lambda * value;
            let gr = &mut g[r * q..(r + 1) * q];
            gr[0] = value;
            for k in 0..n_linear {
                gr[1 + k] = lambda * row[1 + k];
            }
            for m in 0..n_shape {
                let at = |c: usize| jacobian[(r * columns + c) * n_shape + m];
                let derivative =
                    at(0) + theta.iter().enumerate().map(|(k, t)| t * at(1 + k)).sum::<f64>();
                gr[columns + m] = lambda * derivative;
            }
        }
        let starts = self.utility.set_starts();
        let chosen = self.utility.chosen();
        let n_sets = chosen.len();
        let mut log_likelihood = 0.0;
        let mut score = vec![0.0; q];
        let mut information = vec![0.0; q * q];
        let mut set_scores = vec![0.0; n_sets * q];
        let mut scale = 0.0;
        let mut gbar = vec![0.0; q];
        for s in 0..n_sets {
            let (start, end) = (starts[s], starts[s + 1]);
            let c = start + chosen[s];
            let top = eta[start..end].iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let weights: Vec<f64> = eta[start..end].iter().map(|e| (e - top).exp()).collect();
            let total: f64 = weights.iter().sum();
            log_likelihood += eta[c] - top - total.ln();
            gbar.fill(0.0);
            for (offset, w) in weights.iter().enumerate() {
                let r = start + offset;
                for j in 0..q {
                    gbar[j] += w / total * g[r * q + j];
                }
            }
            let mut largest = 0.0_f64;
            for r in start..end {
                for j in 0..q {
                    largest = largest.max((g[r * q + j] - g[c * q + j]).abs());
                }
            }
            scale += largest;
            for j in 0..q {
                let value = g[c * q + j] - gbar[j];
                score[j] += value;
                set_scores[s * q + j] = value;
            }
            for (offset, w) in weights.iter().enumerate() {
                let r = start + offset;
                let p = w / total;
                for i in 0..q {
                    let di = g[r * q + i] - gbar[i];
                    for j in i..q {
                        information[i * q + j] += p * di * (g[r * q + j] - gbar[j]);
                    }
                }
            }
        }
        for i in 0..q {
            for j in 0..i {
                information[i * q + j] = information[j * q + i];
            }
        }
        if !log_likelihood.is_finite() {
            return Err(UtilityNotEvaluable);
        }
        Ok(PsiEvaluation { log_likelihood, score, information, set_scores, scale })
    }

    /// The score with every shape component that pushes against the bound it sits on set to
    /// zero, and its largest absolute entry.
    fn projected_score(&self, psi: &[f64], evaluation: &PsiEvaluation) -> (Vec<f64>, f64) {
        let columns = self.p_beta();
        let mut score = evaluation.score.clone();
        for (m, (lo, hi)) in self.bounds.iter().enumerate() {
            let value = psi[columns + m];
            let j = columns + m;
            if (value <= *lo && score[j] < 0.0) || (value >= *hi && score[j] > 0.0) {
                score[j] = 0.0;
            }
        }
        let largest = score.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        (score, largest)
    }

    /// Whether the projected score passes the convergence test of the inner fits: its largest
    /// entry at most the gradient tolerance times the gradient scale.
    fn stationary(&self, largest: f64, evaluation: &PsiEvaluation) -> bool {
        largest <= self.options.fit.gradient_tolerance * evaluation.scale.max(f64::MIN_POSITIVE)
    }

    /// Fisher scoring in `psi` from `start`, shape parameters kept inside their bounds; where the
    /// full scoring step is not accepted, Newton's step with the observed information (central
    /// differences of the score) is tried as well and the better of the two is taken.
    ///
    /// The polish ends when the projected score passes the convergence test, or when an
    /// accepted step leaves every coordinate unchanged (the maximum is located to the precision
    /// of the coordinates). Ending otherwise (neither step can be solved or accepted, or the
    /// iteration limit is reached) at a point whose projected score fails the test refuses with [`NonlinearChoiceError::NotConverged`]: the covariance and the profile
    /// intervals would be read at a point that is not a maximum.
    fn polish(&self, start: Vec<f64>) -> Result<(Vec<f64>, PsiEvaluation), NonlinearChoiceError> {
        let columns = self.p_beta();
        let not_evaluable = |psi: &[f64]| NonlinearChoiceError::UtilityNotEvaluable {
            shape: psi[columns..].to_vec(),
        };
        let mut psi = start;
        let mut current = self.evaluate_psi(&psi).map_err(|_| not_evaluable(&psi))?;
        let mut iterations = 0;
        let mut settled = false;
        for _ in 0..self.options.fit.max_iterations {
            self.observe()?;
            let (score, largest) = self.projected_score(&psi, &current);
            if self.stationary(largest, &current) {
                settled = true;
                break;
            }
            // Fisher scoring first. Where its full step is not accepted (singular, or halved
            // because the expected information misses curvature), Newton's step with the
            // observed information is tried too, and the step reaching the higher
            // log-likelihood is taken.
            let scoring = self.scoring_step(&psi, &current, &current.information, &score);
            let accepted = match scoring {
                Some((next, evaluation, true)) => Some((next, evaluation)),
                scoring => {
                    let newton = self
                        .observed_information(&psi)?
                        .and_then(|observed| self.scoring_step(&psi, &current, &observed, &score));
                    match (scoring, newton) {
                        (Some((a, ea, _)), Some((b, eb, _))) => {
                            if eb.log_likelihood > ea.log_likelihood {
                                Some((b, eb))
                            } else {
                                Some((a, ea))
                            }
                        }
                        (Some((a, ea, _)), None) | (None, Some((a, ea, _))) => Some((a, ea)),
                        (None, None) => None,
                    }
                }
            };
            let Some((next, evaluation)) = accepted else {
                break;
            };
            let moved = next.iter().zip(&psi).any(|(a, b)| a.total_cmp(b).is_ne());
            psi = next;
            current = evaluation;
            iterations += 1;
            if !moved {
                settled = true;
                break;
            }
        }
        if !settled {
            let (_, largest) = self.projected_score(&psi, &current);
            if !self.stationary(largest, &current) {
                return Err(NonlinearChoiceError::NotConverged {
                    iterations,
                    gradient_norm: largest,
                });
            }
        }
        Ok((psi, current))
    }

    /// The step `matrix^{-1} score` from `psi`, shape parameters clamped to their bounds, halved
    /// until the log-likelihood does not fall by more than rounding, and whether the full step
    /// was accepted; `None` when `matrix` is not positive definite or no halving is accepted.
    fn scoring_step(
        &self,
        psi: &[f64],
        current: &PsiEvaluation,
        matrix: &[f64],
        score: &[f64],
    ) -> Option<(Vec<f64>, PsiEvaluation, bool)> {
        let columns = self.p_beta();
        let q = columns + self.bounds.len();
        let step = solve_spd(matrix, q, score)?;
        let allowance = 64.0 * f64::EPSILON * (current.log_likelihood.abs() + 1.0);
        let mut factor = 1.0;
        for halvings in 0..=MAX_HALVINGS {
            let mut candidate: Vec<f64> =
                psi.iter().zip(&step).map(|(value, d)| value + factor * d).collect();
            for (m, (lo, hi)) in self.bounds.iter().enumerate() {
                candidate[columns + m] = candidate[columns + m].clamp(*lo, *hi);
            }
            if let Ok(evaluation) = self.evaluate_psi(&candidate)
                && evaluation.log_likelihood >= current.log_likelihood - allowance
            {
                return Some((candidate, evaluation, halvings == 0));
            }
            factor *= 0.5;
        }
        None
    }

    /// The observed information at `psi`: minus the central difference of the analytic score,
    /// symmetrised, each step `OBSERVED_STEP (1 + |psi_k|)` and kept inside the shape bounds.
    /// The expected information drops the curvature that the second derivatives of the utility
    /// carry, so it can vanish in a shape direction at a maximum (where the utility's shape
    /// derivative is zero) while the observed information does not. `None` where the score
    /// cannot be evaluated on either side.
    fn observed_information(&self, psi: &[f64]) -> Result<Option<Vec<f64>>, NonlinearChoiceError> {
        self.observe()?;
        let columns = self.p_beta();
        let q = columns + self.bounds.len();
        let mut matrix = vec![0.0; q * q];
        for k in 0..q {
            let h = OBSERVED_STEP * (1.0 + psi[k].abs());
            let mut up = psi.to_vec();
            let mut down = psi.to_vec();
            up[k] += h;
            down[k] -= h;
            if k >= columns {
                let (lo, hi) = self.bounds[k - columns];
                up[k] = up[k].min(hi);
                down[k] = down[k].max(lo);
            }
            let width = up[k] - down[k];
            if width.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
                return Ok(None);
            }
            let (Ok(above), Ok(below)) = (self.evaluate_psi(&up), self.evaluate_psi(&down)) else {
                return Ok(None);
            };
            for j in 0..q {
                matrix[j * q + k] = -(above.score[j] - below.score[j]) / width;
            }
        }
        for j in 0..q {
            for k in 0..j {
                let mean = 0.5 * (matrix[j * q + k] + matrix[k * q + j]);
                matrix[j * q + k] = mean;
                matrix[k * q + j] = mean;
            }
        }
        Ok(Some(matrix))
    }

    /// The polished direct parameters started from the profile at `shape`.
    fn polish_from(
        &self,
        shape: &[f64],
    ) -> Result<(Vec<f64>, PsiEvaluation), NonlinearChoiceError> {
        let beta = self.profile(shape)?.beta;
        let consistency = beta[0];
        if consistency.is_nan() || consistency <= 0.0 {
            return Err(NonlinearChoiceError::ConsistencyNotPositive { consistency });
        }
        let mut start = Vec::with_capacity(beta.len() + shape.len());
        start.push(consistency);
        start.extend(beta[1..].iter().map(|b| b / consistency));
        start.extend(shape.iter().copied());
        self.polish(start)
    }

    #[allow(clippy::too_many_lines, reason = "the fit's stages in the order the module documents")]
    fn run(&self) -> Result<NonlinearConditionalLogitFit, NonlinearChoiceError> {
        self.observe()?;
        let columns = self.p_beta();
        let n_shape = self.bounds.len();
        let q = columns + n_shape;
        let null = null_log_likelihood(self.utility.set_starts());
        let (mut psi, mut evaluation) = if n_shape == 0 {
            let features = self
                .utility
                .features(&[])
                .map_err(|_| NonlinearChoiceError::UtilityNotEvaluable { shape: Vec::new() })?;
            let fit = self.inner_fit(&features, columns).map_err(|error| match error {
                ConditionalLogitError::RankDeficient { columns } => {
                    NonlinearChoiceError::RankDeficient { columns }
                }
                other => Self::map_inner(other, &[]),
            })?;
            self.evaluations.set(1);
            let beta = fit.coefficients;
            let consistency = beta[0];
            if consistency.is_nan() || consistency <= 0.0 {
                return Err(NonlinearChoiceError::ConsistencyNotPositive { consistency });
            }
            let mut start = Vec::with_capacity(q);
            start.push(consistency);
            start.extend(beta[1..].iter().map(|b| b / consistency));
            let evaluation = self
                .evaluate_psi(&start)
                .map_err(|_| NonlinearChoiceError::UtilityNotEvaluable { shape: Vec::new() })?;
            (start, evaluation)
        } else {
            let shape = self.search()?;
            self.polish_from(&shape)?
        };
        if psi[0].is_nan() || psi[0] <= 0.0 {
            return Err(NonlinearChoiceError::ConsistencyNotPositive { consistency: psi[0] });
        }
        // The profiles are maximised from the polished point. A profile value above the polished
        // log-likelihood means the search and the polish stopped at a lower maximum: polish once
        // more from the best profile point, and refuse if a profile still lies above.
        let mut profiles = Vec::with_capacity(n_shape);
        let mut repolished = false;
        loop {
            profiles.clear();
            let mut best: Option<ProfileTop> = None;
            for m in 0..n_shape {
                let (profile, top) =
                    self.shape_profile(&psi[columns..], m, evaluation.log_likelihood)?;
                if let Some((value, point)) = top
                    && best.as_ref().is_none_or(|(current, _)| value > *current)
                {
                    best = Some((value, point));
                }
                profiles.push(profile);
            }
            let excess = PROFILE_EXCESS_RELATIVE * (1.0 + evaluation.log_likelihood.abs());
            let Some((_, point)) =
                best.filter(|(value, _)| *value > evaluation.log_likelihood + excess)
            else {
                break;
            };
            if repolished {
                let (_, largest) = self.projected_score(&psi, &evaluation);
                return Err(NonlinearChoiceError::NotConverged {
                    iterations: self.options.fit.max_iterations,
                    gradient_norm: largest,
                });
            }
            repolished = true;
            (psi, evaluation) = self.polish_from(&point)?;
            if psi[0].is_nan() || psi[0] <= 0.0 {
                return Err(NonlinearChoiceError::ConsistencyNotPositive { consistency: psi[0] });
            }
        }
        let log_likelihood = evaluation.log_likelihood;
        let threshold = self.threshold();
        for (m, profile) in profiles.iter().enumerate() {
            let finite = profile.log_likelihood.iter().copied().filter(|v| v.is_finite());
            let (low, high) =
                finite.fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), v| (l.min(v), h.max(v)));
            let profile_range = high - low;
            if profile_range.is_nan() || profile_range < threshold {
                return Err(NonlinearChoiceError::ShapeNotIdentified {
                    dimension: m,
                    profile_range,
                });
            }
        }
        for (m, profile) in profiles.iter().enumerate() {
            if let Some(modes) = two_modes(profile, log_likelihood, threshold) {
                return Err(NonlinearChoiceError::ProfileMultimodal { dimension: m, modes });
            }
        }
        for (m, (profile, (lo, hi))) in profiles.iter().zip(self.bounds).enumerate() {
            let estimate = psi[columns + m];
            let width = hi - lo;
            if estimate <= lo + BOUND_FRACTION * width || estimate >= hi - BOUND_FRACTION * width {
                return Err(NonlinearChoiceError::ShapeEstimateAtBound {
                    dimension: m,
                    estimate,
                    interval: profile.interval,
                });
            }
        }
        let dependent = null_directions(&evaluation.information, q);
        if !dependent.is_empty() {
            return Err(NonlinearChoiceError::RankDeficient { columns: dependent });
        }
        let mut covariance = invert_square(&evaluation.information, q)
            .ok_or(NonlinearChoiceError::RankDeficient { columns: (0..q).collect() })?;
        symmetrise(&mut covariance, q);
        let cluster_covariance = self
            .utility
            .clusters()
            .map(|clusters| cluster_sandwich(&covariance, &evaluation.set_scores, clusters, q));
        Ok(NonlinearConditionalLogitFit {
            consistency: psi[0],
            linear: psi[1..columns].to_vec(),
            shape: psi[columns..].to_vec(),
            log_likelihood: evaluation.log_likelihood,
            null_log_likelihood: null,
            information: evaluation.information,
            covariance,
            cluster_covariance,
            profiles,
            profile_evaluations: self.evaluations.get(),
        })
    }

    /// `chi2_1(level) / 2`.
    fn threshold(&self) -> f64 {
        let z = norm_inv(0.5 + 0.5 * self.options.level);
        0.5 * z * z
    }

    /// The profile of shape parameter `m` at `value`, maximised over the other shape
    /// parameters from `estimate`, with the shape point attaining it; `None` where the utility is
    /// not defined.
    fn profile_at(
        &self,
        estimate: &[f64],
        m: usize,
        value: f64,
    ) -> Result<Option<ProfileTop>, NonlinearChoiceError> {
        let mut start = estimate.to_vec();
        start[m] = value;
        let Some(point) = self.profile_or_none(&start)? else {
            return Ok(None);
        };
        if estimate.len() == 1 {
            return Ok(Some((point.log_likelihood, start)));
        }
        let free: Vec<usize> = (0..estimate.len()).filter(|k| *k != m).collect();
        let (best, value) = self.compass(start, point.log_likelihood, &free)?;
        Ok(Some((value, best)))
    }

    /// The profile of shape parameter `m` and its largest value with the shape point attaining
    /// it.
    fn shape_profile(
        &self,
        estimate: &[f64],
        m: usize,
        maximum: f64,
    ) -> Result<(ShapeProfile, Option<ProfileTop>), NonlinearChoiceError> {
        let (lo, hi) = self.bounds[m];
        let n = self.options.search.profile_points;
        let mut values: Vec<f64> =
            (0..n).map(|i| lo + (hi - lo) * (i as f64) / ((n - 1) as f64)).collect();
        values.push(estimate[m]);
        values.sort_by(f64::total_cmp);
        values.dedup_by(|a, b| a.total_cmp(b).is_eq());
        let mut log_likelihood = Vec::with_capacity(values.len());
        let mut best: Option<ProfileTop> = None;
        for &value in &values {
            match self.profile_at(estimate, m, value)? {
                Some((profile, point)) => {
                    if best.as_ref().is_none_or(|(current, _)| profile > *current) {
                        best = Some((profile, point));
                    }
                    log_likelihood.push(profile);
                }
                None => log_likelihood.push(f64::NEG_INFINITY),
            }
        }
        let top = log_likelihood.iter().copied().fold(maximum, f64::max);
        let cutoff = top - self.threshold();
        let centre = values.iter().position(|v| v.total_cmp(&estimate[m]).is_eq()).unwrap_or(0);
        let (lower, lower_truncated) =
            self.interval_end(estimate, m, &values, &log_likelihood, centre, cutoff, false)?;
        let (upper, upper_truncated) =
            self.interval_end(estimate, m, &values, &log_likelihood, centre, cutoff, true)?;
        Ok((
            ShapeProfile {
                values,
                log_likelihood,
                interval: (lower, upper),
                lower_truncated,
                upper_truncated,
            },
            best,
        ))
    }

    /// One end of the likelihood-ratio interval: the first grid value past the centre whose
    /// profile falls below `cutoff`, refined by bisection; the bound when none does.
    #[allow(clippy::too_many_arguments, reason = "the profile grid and the end to locate")]
    fn interval_end(
        &self,
        estimate: &[f64],
        m: usize,
        values: &[f64],
        log_likelihood: &[f64],
        centre: usize,
        cutoff: f64,
        upward: bool,
    ) -> Result<(f64, bool), NonlinearChoiceError> {
        let order: Vec<usize> =
            if upward { (centre + 1..values.len()).collect() } else { (0..centre).rev().collect() };
        let mut inside = values[centre];
        for index in order {
            if log_likelihood[index] < cutoff {
                let mut outside = values[index];
                let width = self.bounds[m].1 - self.bounds[m].0;
                for _ in 0..MAX_BISECTIONS {
                    if (outside - inside).abs() <= BISECTION_FRACTION * width {
                        break;
                    }
                    let middle = 0.5 * inside + 0.5 * outside;
                    match self.profile_at(estimate, m, middle)? {
                        Some((value, _)) if value >= cutoff => inside = middle,
                        _ => outside = middle,
                    }
                }
                return Ok((0.5 * inside + 0.5 * outside, false));
            }
            inside = values[index];
        }
        Ok((if upward { self.bounds[m].1 } else { self.bounds[m].0 }, true))
    }
}

/// Two local maxima of the profile within `threshold` of the overall maximum separated by a
/// value more than `threshold` below the lower of the two.
fn two_modes(profile: &ShapeProfile, maximum: f64, threshold: f64) -> Option<(f64, f64)> {
    let values = &profile.log_likelihood;
    let n = values.len();
    let peaks: Vec<usize> = (0..n)
        .filter(|&i| {
            let left = i == 0 || values[i] >= values[i - 1];
            let right = i + 1 == n || values[i] > values[i + 1];
            left && right && values[i] >= maximum - threshold
        })
        .collect();
    for (a, &i) in peaks.iter().enumerate() {
        for &j in &peaks[a + 1..] {
            let lower = values[i].min(values[j]);
            let dip = values[i + 1..j].iter().copied().fold(f64::INFINITY, f64::min);
            if (values[i] - values[j]).abs() <= threshold && dip < lower - threshold {
                return Some((profile.values[i], profile.values[j]));
            }
        }
    }
    None
}

/// Indices of the parameters in a null direction of `information`: pivoted Cholesky with
/// diagonal pivoting; a pivot at most [`PIVOT_RELATIVE`] times the largest diagonal entry is
/// null, and every parameter with a non-negligible coefficient in the corresponding null
/// vector of the remaining block takes part.
fn null_directions(information: &[f64], q: usize) -> Vec<usize> {
    let largest = (0..q).map(|i| information[i * q + i]).fold(0.0_f64, f64::max);
    if !(largest.is_finite() && largest > 0.0) {
        return (0..q).collect();
    }
    let mut a = information.to_vec();
    let mut order: Vec<usize> = (0..q).collect();
    let mut rank = 0;
    for k in 0..q {
        let (pivot, value) = (k..q)
            .map(|i| (i, a[i * q + i]))
            .fold((k, f64::NEG_INFINITY), |best, item| if item.1 > best.1 { item } else { best });
        if value.is_nan() || value <= PIVOT_RELATIVE * largest {
            break;
        }
        if pivot != k {
            for c in 0..q {
                a.swap(k * q + c, pivot * q + c);
            }
            for r in 0..q {
                a.swap(r * q + k, r * q + pivot);
            }
            order.swap(k, pivot);
        }
        let root = value.sqrt();
        for i in k + 1..q {
            a[i * q + k] /= root;
        }
        for i in k + 1..q {
            for j in k + 1..=i {
                a[i * q + j] -= a[i * q + k] * a[j * q + k];
                a[j * q + i] = a[i * q + j];
            }
        }
        a[k * q + k] = root;
        rank += 1;
    }
    if rank == q {
        return Vec::new();
    }
    let mut involved: Vec<usize> = order[rank..].to_vec();
    // Each dependent parameter is a combination of the accepted ones: solve L11' x = l21 for the
    // coefficients and name every accepted parameter with a non-negligible share.
    for d in rank..q {
        let mut x = vec![0.0; rank];
        for i in (0..rank).rev() {
            let mut value = a[d * q + i];
            for j in i + 1..rank {
                value -= a[j * q + i] * x[j];
            }
            x[i] = value / a[i * q + i];
        }
        let norm = x.iter().map(|v| v * v).sum::<f64>().sqrt().max(f64::MIN_POSITIVE);
        for (i, value) in x.iter().enumerate() {
            if value.abs() > 1e-6 * norm {
                involved.push(order[i]);
            }
        }
    }
    involved.sort_unstable();
    involved.dedup();
    involved
}

fn symmetrise(m: &mut [f64], p: usize) {
    for i in 0..p {
        for j in 0..i {
            let mean = 0.5 * (m[i * p + j] + m[j * p + i]);
            m[i * p + j] = mean;
            m[j * p + i] = mean;
        }
    }
}

/// `K/(K - 1) C (sum_c s_c s_c') C` with `s_c` the per-cluster sum of set scores, `K` clusters,
/// clusters in label order.
fn cluster_sandwich(
    covariance: &[f64],
    set_scores: &[f64],
    clusters: &[u32],
    q: usize,
) -> Vec<f64> {
    let mut sums: BTreeMap<u32, Vec<f64>> = BTreeMap::new();
    for (s, &label) in clusters.iter().enumerate() {
        let entry = sums.entry(label).or_insert_with(|| vec![0.0; q]);
        for (e, v) in entry.iter_mut().zip(&set_scores[s * q..(s + 1) * q]) {
            *e += v;
        }
    }
    let mut meat = vec![0.0; q * q];
    for score in sums.values() {
        for i in 0..q {
            for j in 0..q {
                meat[i * q + j] += score[i] * score[j];
            }
        }
    }
    let product = |a: &[f64], b: &[f64]| {
        let mut out = vec![0.0; q * q];
        for i in 0..q {
            for k in 0..q {
                let aik = a[i * q + k];
                for j in 0..q {
                    out[i * q + j] += aik * b[k * q + j];
                }
            }
        }
        out
    };
    let mut sandwich = product(&product(covariance, &meat), covariance);
    symmetrise(&mut sandwich, q);
    let correction = finite_cluster_correction(sums.len());
    for value in &mut sandwich {
        *value *= correction;
    }
    sandwich
}

/// `C / (C - 1)` for `C` clusters, one for a single cluster: the finite-cluster correction of the
/// sandwich for a maximum-likelihood estimator (Cameron and Miller, 2015, *A Practitioner's
/// Guide to Cluster-Robust Inference*).
#[allow(clippy::cast_precision_loss, reason = "cluster counts are far below 2^53")]
fn finite_cluster_correction(clusters: usize) -> f64 {
    if clusters < 2 { 1.0 } else { clusters as f64 / (clusters as f64 - 1.0) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_first_sobol_dimensions_are_the_known_sequences() {
        let directions: Vec<[u32; 32]> = (0..2).map(sobol_directions).collect();
        let points: Vec<Vec<f64>> = (1..5).map(|i| sobol_point(&directions, i)).collect();
        assert_eq!(points[0], vec![0.5, 0.5]);
        assert_eq!(points[1], vec![0.25, 0.75]);
        assert_eq!(points[2], vec![0.75, 0.25]);
        assert_eq!(points[3], vec![0.125, 0.625]);
    }

    #[test]
    fn null_directions_name_every_dependent_parameter() {
        // Column 2 equals column 0 plus column 1.
        let x = [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 2.0], [2.0, -1.0, 1.0]];
        let mut information = vec![0.0; 9];
        for row in &x {
            for i in 0..3 {
                for j in 0..3 {
                    information[i * 3 + j] += row[i] * row[j];
                }
            }
        }
        assert_eq!(null_directions(&information, 3), vec![0, 1, 2]);
        let independent = [2.0, 0.5, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 3.0];
        assert!(null_directions(&independent, 3).is_empty());
    }

    #[test]
    fn two_separated_maxima_are_found() {
        let profile = ShapeProfile {
            values: vec![0.0, 1.0, 2.0, 3.0, 4.0],
            log_likelihood: vec![-1.0, -10.0, -1.2, -11.0, -5.0],
            interval: (0.0, 4.0),
            lower_truncated: true,
            upper_truncated: true,
        };
        assert_eq!(two_modes(&profile, -1.0, 1.92), Some((0.0, 2.0)));
        let single = ShapeProfile { log_likelihood: vec![-5.0, -2.0, -1.0, -2.0, -5.0], ..profile };
        assert_eq!(two_modes(&single, -1.0, 1.92), None);
    }
}
