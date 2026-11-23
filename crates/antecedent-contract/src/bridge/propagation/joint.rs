//! Joint nonlinear propagation with bounded delta, sigma-point and simulation methods.

use super::super::super::refusal::{
    joint_law_dimension_limit, propagation_admission_refused, propagation_cancelled,
    propagation_work_limit, simulation_draw_limit,
};
use super::{
    BridgeRefusal, MappingUncertainty, SourceMappingDependence, dependence_not_a_covariance,
    malformed_mapping, mapping_uncertainty_invalid, propagation_overflow, source_moment_not_finite,
    source_variance_invalid,
};
use antecedent_core::ExecutionContext;
use antecedent_core::execution::{AdmissionError, AdmissionLease};

// ---------------------------------------------------------------------------
// Joint propagation of a nonlinear function of the source and the mapping.
// ---------------------------------------------------------------------------

/// Fewest simulated draws a propagation accepts: below this the Monte Carlo standard error of
/// the variance, which needs a fourth moment, is not meaningful.
const MIN_SIMULATION_DRAWS: usize = 64;
/// Most simulated draws a propagation accepts. The simulation streams its moments, so the
/// draw count costs time and never memory; the cap and [`MAX_PROPAGATION_WORK`] bound the time.
pub const MAX_SIMULATION_DRAWS: usize = 100_000_000;
/// Most coordinates a [`JointLaw`] admits. The covariance is `n` by `n`, the factorization is
/// cubic in `n` and the second-order delta method evaluates the function `2n² + 1` times.
pub const MAX_JOINT_LAW_DIMENSION: usize = 64;
/// Most logical operations one propagation may plan: each evaluation of the function counts
/// as the coordinates it is built from plus one, each simulated draw as the triangular
/// factor product that makes it, and each factorization or matrix product as its cube.
/// A propagation whose plan passes the cap is refused before it starts.
pub const MAX_PROPAGATION_WORK: u64 = 1 << 34;
/// Draws between two polls of the cancellation token during a simulation.
const SIMULATION_POLL_STRIDE: usize = 1024;
/// Relative step of the finite differences behind the second-order terms, in
/// units of each coordinate's standard deviation.
const DIFFERENCE_STEP: f64 = 1e-2;
/// Variances below this fraction of the largest variance are a degenerate
/// direction in the factorization; negative pivots and off-diagonal residuals
/// beyond [`PSD_SLACK`] of it are refused.
const DEGENERATE_FRACTION: f64 = 1e-13;
const PSD_SLACK: f64 = 1e-9;

/// The joint moment law of every uncertain input of a mapped quantity: the
/// source estimate and each mapping coefficient, with their full covariance.
///
/// The covariance carries the dependence between the source and the mapping
/// that [`SourceMappingDependence`] declares, together with the mapping's own
/// intercept-slope covariance; a nonlinear function of those inputs is then
/// propagated against the joint law, not against a marginal of each.
#[derive(Clone, Debug, PartialEq)]
pub struct JointLaw {
    names: Vec<String>,
    mean: Vec<f64>,
    covariance: Vec<f64>,
}

impl JointLaw {
    /// A joint law from coordinate names, means and a row-major covariance.
    ///
    /// # Errors
    /// The dimension is above [`MAX_JOINT_LAW_DIMENSION`] (family `resource`) or zero; the
    /// lengths disagree; a mean or covariance entry is not finite; the covariance is not
    /// symmetric or not positive semi-definite.
    pub fn new(
        names: Vec<String>,
        mean: Vec<f64>,
        covariance: Vec<f64>,
    ) -> Result<Self, BridgeRefusal> {
        let n = mean.len();
        // Checked before anything is sized from the dimension: the factorization below is
        // cubic in it, and the cap is a compile-time constant, so this constructor needs no
        // context.
        if n > MAX_JOINT_LAW_DIMENSION {
            return Err(joint_law_dimension_limit(n));
        }
        if n == 0 || names.len() != n || n.checked_mul(n) != Some(covariance.len()) {
            return Err(malformed_mapping(
                "joint_law",
                "state one name and one mean per coordinate and an n by n covariance",
            ));
        }
        if names.iter().any(|name| name.trim().is_empty()) {
            return Err(malformed_mapping("joint_law.names", "name every coordinate"));
        }
        if mean.iter().any(|value| !value.is_finite()) {
            return Err(source_moment_not_finite("joint_law.mean", "state finite means"));
        }
        if covariance.iter().any(|value| !value.is_finite()) {
            return Err(mapping_uncertainty_invalid(
                "joint_law.covariance",
                "state a finite joint covariance",
            ));
        }
        let scale = (0..n).map(|index| covariance[index * n + index]).fold(0.0_f64, f64::max);
        for row in 0..n {
            if covariance[row * n + row] < 0.0 {
                return Err(mapping_uncertainty_invalid(
                    "joint_law.covariance",
                    "state non-negative variances on the diagonal",
                ));
            }
            for column in 0..row {
                if (covariance[row * n + column] - covariance[column * n + row]).abs()
                    > PSD_SLACK * scale
                {
                    return Err(mapping_uncertainty_invalid(
                        "joint_law.covariance",
                        "state a symmetric joint covariance",
                    ));
                }
            }
        }
        if cholesky(&covariance, n).is_none() {
            return Err(mapping_uncertainty_invalid(
                "joint_law.covariance",
                "state a positive semi-definite joint covariance: the declared covariances are not jointly feasible",
            ));
        }
        Ok(Self { names, mean, covariance })
    }

    /// The joint law of an affine mapping `a + b M` declared by `uncertainty`
    /// and `dependence`: coordinates `intercept`, `slope`, `source`.
    ///
    /// The source estimator is independent of the intercept estimator; its
    /// covariance with the slope estimator is the declared dependence.
    ///
    /// # Errors
    /// The declarations are malformed, the source moments are not finite and
    /// non-negative, the slope-source covariance exceeds what the two variances
    /// allow, or the three covariances are not jointly feasible.
    pub fn from_affine_declaration(
        intercept: f64,
        slope: f64,
        uncertainty: &MappingUncertainty,
        dependence: &SourceMappingDependence,
        source_mean: f64,
        source_variance: f64,
    ) -> Result<Self, BridgeRefusal> {
        uncertainty.validate()?;
        dependence.validate()?;
        if !source_mean.is_finite() {
            return Err(source_moment_not_finite("source_mean", "state a finite source mean"));
        }
        if !(source_variance.is_finite() && source_variance >= 0.0) {
            return Err(source_variance_invalid(source_variance.to_string()));
        }
        let covariance = dependence.covariance();
        let product = uncertainty.slope_variance * source_variance;
        if !product.is_finite() || covariance * covariance > product {
            return Err(dependence_not_a_covariance(covariance.to_string()));
        }
        Self::new(
            vec!["intercept".into(), "slope".into(), "source".into()],
            vec![intercept, slope, source_mean],
            vec![
                uncertainty.intercept_variance,
                uncertainty.intercept_slope_covariance,
                0.0,
                uncertainty.intercept_slope_covariance,
                uncertainty.slope_variance,
                covariance,
                0.0,
                covariance,
                source_variance,
            ],
        )
    }

    /// Number of coordinates.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.mean.len()
    }

    /// Coordinate names in order.
    #[must_use]
    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// Coordinate means.
    #[must_use]
    pub fn mean(&self) -> &[f64] {
        &self.mean
    }

    /// Row-major joint covariance.
    #[must_use]
    pub fn covariance(&self) -> &[f64] {
        &self.covariance
    }

    fn factor(&self) -> Vec<f64> {
        // `new` proved the factorization exists.
        cholesky(&self.covariance, self.mean.len()).unwrap_or_default()
    }
}

/// Lower-triangular `L` with `L Lᵀ = covariance`, tolerating zero-variance
/// directions; `None` when the matrix is not positive semi-definite.
fn cholesky(covariance: &[f64], n: usize) -> Option<Vec<f64>> {
    let scale = (0..n).map(|index| covariance[index * n + index]).fold(0.0_f64, f64::max);
    let degenerate = DEGENERATE_FRACTION * scale;
    let slack = PSD_SLACK * scale;
    let mut factor = vec![0.0; n * n];
    for column in 0..n {
        let mut pivot = covariance[column * n + column];
        for k in 0..column {
            pivot -= factor[column * n + k] * factor[column * n + k];
        }
        if pivot < -slack {
            return None;
        }
        if pivot <= degenerate {
            for row in column + 1..n {
                let mut residual = covariance[row * n + column];
                for k in 0..column {
                    residual -= factor[row * n + k] * factor[column * n + k];
                }
                if residual.abs() > slack {
                    return None;
                }
            }
        } else {
            let diagonal = pivot.sqrt();
            factor[column * n + column] = diagonal;
            for row in column + 1..n {
                let mut residual = covariance[row * n + column];
                for k in 0..column {
                    residual -= factor[row * n + k] * factor[column * n + k];
                }
                factor[row * n + column] = residual / diagonal;
            }
        }
    }
    Some(factor)
}

/// How a nonlinear function of a joint law is carried to a mean and a variance.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum PropagationMethod {
    /// Second-order delta method under the Gaussian moment closure of the
    /// joint law: `g(m) + ½ tr(HΣ)` and `∇ᵀΣ∇ + ½ tr((HΣ)²)`, with the
    /// gradient and Hessian from central differences.
    SecondOrderDelta,
    /// Sigma-point (unscented) transform: `2n + 1` deterministic points chosen
    /// to match the joint mean and covariance, with the fourth-moment
    /// correction `κ = max(3 − n, 0)`.
    Unscented,
    /// Seeded Monte Carlo draws of the joint law. The result carries the
    /// Monte Carlo standard errors of the mean and of the variance.
    Simulation {
        /// Number of draws.
        draws: usize,
        /// Seed of the draw stream.
        seed: u64,
    },
}

impl PropagationMethod {
    /// Stable name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SecondOrderDelta => "second_order_delta",
            Self::Unscented => "unscented",
            Self::Simulation { .. } => "simulation",
        }
    }
}

/// Mean and variance of `g` over a joint law.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JointPropagation {
    /// The method that produced the moments.
    pub method: PropagationMethod,
    /// Coordinates of the joint law.
    pub dimension: usize,
    /// `g` at the joint mean: the plug-in value.
    pub plug_in: f64,
    /// `E[g]`.
    pub mean: f64,
    /// `Var(g)`.
    pub variance: f64,
    /// First-order variance `∇ᵀΣ∇`; set by the delta method only.
    pub linearized_variance: Option<f64>,
    /// Monte Carlo standard error of `mean`; set by simulation only.
    pub mean_standard_error: Option<f64>,
    /// Monte Carlo standard error of `variance`; set by simulation only.
    pub variance_standard_error: Option<f64>,
    /// Evaluations of `g` the method used.
    pub evaluations: usize,
}

impl JointPropagation {
    /// `E[g] − g(m)`: what the plug-in value omits.
    #[must_use]
    pub fn bias(&self) -> f64 {
        self.mean - self.plug_in
    }

    /// Standard deviation of `g`.
    #[must_use]
    pub fn standard_deviation(&self) -> f64 {
        self.variance.sqrt()
    }
}

fn checked(value: f64) -> Result<f64, BridgeRefusal> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(propagation_overflow(
            "nonlinear_map",
            "the function produced a value that is not finite inside the joint law; restrict it to its domain or use a smaller spread",
        ))
    }
}

/// What one propagation holds while it runs: the admitted working set and the context whose
/// cancellation it polls. Dropping it returns the lease.
struct Admitted<'a> {
    ctx: &'a ExecutionContext,
    _lease: AdmissionLease,
}

impl Admitted<'_> {
    /// Stop with a `cancelled_no_claim` refusal once the context is cancelled.
    fn poll(&self) -> Result<(), BridgeRefusal> {
        if self.ctx.cancellation.is_cancelled() { Err(propagation_cancelled()) } else { Ok(()) }
    }
}

/// Bytes the working set of a propagation over `n` coordinates holds: the covariance factor,
/// the Hessian, its product with the covariance, and the points and gradients.
fn working_bytes(n: u64) -> u64 {
    8 * (4 * n * n + 8 * n + 16)
}

/// Operations of one evaluation of the function: the point is built from `n` coordinates.
fn evaluation_work(n: u64) -> u64 {
    n + 1
}

/// Planned operations of the second-order delta method: `2n² + 1` evaluations, then the
/// covariance products behind the trace terms.
pub(super) fn delta_work(n: u64) -> u64 {
    (2 * n * n + 1).saturating_mul(evaluation_work(n)).saturating_add(2 * n * n * n + 3 * n * n)
}

/// Planned operations of the sigma-point transform: `2n + 1` evaluations and one factorization.
pub(super) fn unscented_work(n: u64) -> u64 {
    (2 * n + 1).saturating_mul(evaluation_work(n)).saturating_add(n * n * n)
}

/// Planned operations of a simulation: one factorization, then per draw the `n` normals, the
/// triangular product and the evaluation.
pub(super) fn simulation_work(n: u64, draws: u64) -> u64 {
    let per_draw = n * (n + 1) / 2 + 2 * n + 1;
    draws.saturating_mul(per_draw).saturating_add(n * n * n)
}

/// Admit one propagation: refuse a cancelled context, refuse a plan over the work cap, and
/// reserve the working set against the context's memory pool, all before anything is allocated.
fn admit(
    ctx: &ExecutionContext,
    dimension: usize,
    planned_work: u64,
) -> Result<Admitted<'_>, BridgeRefusal> {
    if ctx.cancellation.is_cancelled() {
        return Err(propagation_cancelled());
    }
    if planned_work > MAX_PROPAGATION_WORK {
        return Err(propagation_work_limit(planned_work));
    }
    let bytes = working_bytes(dimension as u64);
    let lease = ctx.acquire_admission(0, bytes).map_err(|error| match error {
        AdmissionError::Cancelled => propagation_cancelled(),
        other => propagation_admission_refused(bytes, &other),
    })?;
    Ok(Admitted { ctx, _lease: lease })
}

/// Mean and variance of `g(Z)` for `Z` with the joint moments of `law`.
///
/// The propagation is admitted under `ctx` before it allocates: it refuses with family
/// `resource` when its planned work passes [`MAX_PROPAGATION_WORK`], when a simulation asks for
/// more than [`MAX_SIMULATION_DRAWS`] draws, or when `ctx`'s memory pool cannot hold the working
/// set. Cancellation is polled once per row of the second-order Hessian, once per sigma point
/// and once per 1024 simulated draws, and returns `cancelled_no_claim` with no moments.
///
/// # Errors
/// The function is not finite at a point the method evaluates; a simulation asks for fewer
/// than 64 draws; a resource refusal or a cancellation as above.
pub fn propagate_joint(
    law: &JointLaw,
    g: &dyn Fn(&[f64]) -> f64,
    method: PropagationMethod,
    ctx: &ExecutionContext,
) -> Result<JointPropagation, BridgeRefusal> {
    match method {
        PropagationMethod::SecondOrderDelta => second_order_delta(law, g, ctx),
        PropagationMethod::Unscented => unscented(law, g, ctx),
        PropagationMethod::Simulation { draws, seed } => simulate(law, g, draws, seed, ctx),
    }
}

fn second_order_delta(
    law: &JointLaw,
    g: &dyn Fn(&[f64]) -> f64,
    ctx: &ExecutionContext,
) -> Result<JointPropagation, BridgeRefusal> {
    let n = law.dimension();
    let admitted = admit(ctx, n, delta_work(n as u64))?;
    let sigma = law.covariance();
    let center = law.mean();
    let step: Vec<f64> = (0..n).map(|i| DIFFERENCE_STEP * sigma[i * n + i].sqrt()).collect();
    let active: Vec<usize> = (0..n).filter(|&i| sigma[i * n + i] > 0.0).collect();
    let mut evaluations = 0_usize;
    let mut eval = |shifts: &[(usize, f64)]| -> Result<f64, BridgeRefusal> {
        let mut point = center.to_vec();
        for &(index, delta) in shifts {
            point[index] += delta;
        }
        evaluations += 1;
        checked(g(&point))
    };
    let plug_in = eval(&[])?;
    let mut gradient = vec![0.0; n];
    let mut hessian = vec![0.0; n * n];
    for &i in &active {
        admitted.poll()?;
        let up = eval(&[(i, step[i])])?;
        let down = eval(&[(i, -step[i])])?;
        gradient[i] = (up - down) / (2.0 * step[i]);
        hessian[i * n + i] = (up - 2.0 * plug_in + down) / (step[i] * step[i]);
    }
    for (position, &i) in active.iter().enumerate() {
        admitted.poll()?;
        for &j in &active[position + 1..] {
            let upper = eval(&[(i, step[i]), (j, step[j])])?;
            let left = eval(&[(i, step[i]), (j, -step[j])])?;
            let right = eval(&[(i, -step[i]), (j, step[j])])?;
            let lower = eval(&[(i, -step[i]), (j, -step[j])])?;
            let mixed = (upper - left - right + lower) / (4.0 * step[i] * step[j]);
            hessian[i * n + j] = mixed;
            hessian[j * n + i] = mixed;
        }
    }
    let mut correction = 0.0;
    let mut linearized = 0.0;
    for i in 0..n {
        for j in 0..n {
            correction += hessian[i * n + j] * sigma[i * n + j];
            linearized += gradient[i] * sigma[i * n + j] * gradient[j];
        }
    }
    // A = H Σ; tr(A²) = Σ_ij A_ij A_ji.
    let mut product = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                sum += hessian[i * n + k] * sigma[k * n + j];
            }
            product[i * n + j] = sum;
        }
    }
    let mut trace_square = 0.0;
    for i in 0..n {
        for j in 0..n {
            trace_square += product[i * n + j] * product[j * n + i];
        }
    }
    let mean = plug_in + 0.5 * correction;
    let variance = (linearized + 0.5 * trace_square).max(0.0);
    if !mean.is_finite() || !variance.is_finite() {
        return Err(propagation_overflow(
            "second_order_delta",
            "the second-order delta method overflowed the finite range",
        ));
    }
    Ok(JointPropagation {
        method: PropagationMethod::SecondOrderDelta,
        dimension: n,
        plug_in,
        mean,
        variance,
        linearized_variance: Some(linearized),
        mean_standard_error: None,
        variance_standard_error: None,
        evaluations,
    })
}

fn unscented(
    law: &JointLaw,
    g: &dyn Fn(&[f64]) -> f64,
    ctx: &ExecutionContext,
) -> Result<JointPropagation, BridgeRefusal> {
    let n = law.dimension();
    let admitted = admit(ctx, n, unscented_work(n as u64))?;
    let factor = law.factor();
    let center = law.mean();
    let kappa = if n < 3 { (3 - n) as f64 } else { 0.0 };
    let spread = (n as f64 + kappa).sqrt();
    let central_weight = kappa / (n as f64 + kappa);
    let side_weight = 1.0 / (2.0 * (n as f64 + kappa));
    let plug_in = checked(g(center))?;
    let mut values = Vec::with_capacity(2 * n);
    for column in 0..n {
        admitted.poll()?;
        for sign in [1.0, -1.0] {
            let point: Vec<f64> =
                (0..n).map(|row| center[row] + sign * spread * factor[row * n + column]).collect();
            values.push(checked(g(&point))?);
        }
    }
    let mean = central_weight * plug_in + side_weight * values.iter().sum::<f64>();
    let mut variance = central_weight * (plug_in - mean).powi(2);
    for value in &values {
        variance += side_weight * (value - mean).powi(2);
    }
    if !mean.is_finite() || !variance.is_finite() {
        return Err(propagation_overflow(
            "unscented",
            "the sigma-point transform overflowed the finite range",
        ));
    }
    Ok(JointPropagation {
        method: PropagationMethod::Unscented,
        dimension: n,
        plug_in,
        mean,
        variance,
        linearized_variance: None,
        mean_standard_error: None,
        variance_standard_error: None,
        evaluations: 1 + values.len(),
    })
}

/// `SplitMix64` stream: a fixed seed reproduces the draws on every platform.
struct DrawStream(u64);

impl DrawStream {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform on the open interval (0, 1).
    fn uniform(&mut self) -> f64 {
        (((self.next_u64() >> 11) as f64) + 0.5) / 9_007_199_254_740_992.0
    }

    fn normal_pair(&mut self) -> (f64, f64) {
        let radius = (-2.0 * self.uniform().ln()).sqrt();
        let angle = std::f64::consts::TAU * self.uniform();
        (radius * angle.cos(), radius * angle.sin())
    }
}

fn simulate(
    law: &JointLaw,
    g: &dyn Fn(&[f64]) -> f64,
    draws: usize,
    seed: u64,
    ctx: &ExecutionContext,
) -> Result<JointPropagation, BridgeRefusal> {
    if draws < MIN_SIMULATION_DRAWS {
        return Err(malformed_mapping(
            "simulation.draws",
            "state at least 64 simulated draws, and at most the simulation draw limit",
        ));
    }
    if draws > MAX_SIMULATION_DRAWS {
        return Err(simulation_draw_limit(draws));
    }
    let admitted =
        admit(ctx, law.dimension(), simulation_work(law.dimension() as u64, draws as u64))?;
    let n = law.dimension();
    let factor = law.factor();
    let center = law.mean();
    let plug_in = checked(g(center))?;
    let mut stream = DrawStream(seed);
    let mut normals = vec![0.0; n];
    let mut point = vec![0.0; n];
    // Streaming central moments (Pébay's update through the fourth), so the draw count does not
    // bound memory and a long run keeps the accuracy of a Welford recurrence.
    let (mut seen, mut mean, mut m2, mut m3, mut m4) =
        (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    for draw in 0..draws {
        if draw % SIMULATION_POLL_STRIDE == 0 {
            admitted.poll()?;
        }
        let mut index = 0;
        while index < n {
            let (first, second) = stream.normal_pair();
            normals[index] = first;
            if index + 1 < n {
                normals[index + 1] = second;
            }
            index += 2;
        }
        for row in 0..n {
            let mut value = center[row];
            for column in 0..=row {
                value += factor[row * n + column] * normals[column];
            }
            point[row] = value;
        }
        let value = checked(g(&point))?;
        let previous = seen;
        seen += 1.0;
        let delta = value - mean;
        let delta_n = delta / seen;
        let delta_n_squared = delta_n * delta_n;
        let term = delta * delta_n * previous;
        mean += delta_n;
        m4 += term * delta_n_squared * (seen * seen - 3.0 * seen + 3.0)
            + 6.0 * delta_n_squared * m2
            - 4.0 * delta_n * m3;
        m3 += term * delta_n * (seen - 2.0) - 3.0 * delta_n * m2;
        m2 += term;
    }
    let count = draws as f64;
    let (second, fourth) = (m2, m4);
    let population_variance = second / count;
    let variance = second / (count - 1.0);
    let fourth_moment = fourth / count;
    let mean_standard_error = (variance / count).sqrt();
    let variance_standard_error =
        ((fourth_moment - population_variance * population_variance).max(0.0) / count).sqrt();
    if !mean.is_finite() || !variance.is_finite() {
        return Err(propagation_overflow(
            "simulation",
            "the simulated moments overflowed the finite range",
        ));
    }
    Ok(JointPropagation {
        method: PropagationMethod::Simulation { draws, seed },
        dimension: n,
        plug_in,
        mean,
        variance,
        linearized_variance: None,
        mean_standard_error: Some(mean_standard_error),
        variance_standard_error: Some(variance_standard_error),
        evaluations: draws + 1,
    })
}
