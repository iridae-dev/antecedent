//! Small dense linear programs with checkable answers.
//!
//! [`solve`] minimises `objective . x` over `{x : constraints, lower <= x <= upper}` with a
//! working-set primal simplex. Every constraint and finite bound is written as a row
//! `g . x <= h` (an equality gives two rows, each row is scaled to unit maximum coefficient). A
//! vertex is the solution of `n` linearly independent rows held in the working set; a variable
//! with no finite bound starts as a placeholder row `x_j = value` that carries a free multiplier
//! and leaves the working set once the objective pushes on it. At each vertex the multipliers
//! `mu` solve `W^T mu = -objective`; a negative multiplier names a row whose release improves the
//! objective, and the ratio test picks the first row the move reaches. Placeholders are
//! released first and then Bland's rule (smallest row index to release, smallest row index among
//! the blocking rows) prevents cycling on degenerate vertices. The working matrix is factorised
//! afresh at every vertex, so the point never drifts away from the rows that define it.
//!
//! Phase one adds one auxiliary variable `t in [0, 1]`: every row violated by the start point
//! (each variable at a finite bound, or zero when it has none) is relaxed by its violation times
//! `t`, so the start point with `t = 1` is a vertex of the relaxed program, and the relaxed
//! program is minimised over `t`. A positive minimum means the rows conflict; the phase-one
//! multipliers are then a Farkas certificate. A zero minimum gives a vertex of the original rows.
//!
//! [`verify`] rechecks any outcome without the pivoting code: primal feasibility, the reported
//! objective and a non-negative least-squares fit of the objective by the tight rows
//! (Lawson-Hanson active set on Householder least squares) for an optimum; signs and the
//! contradiction for a certificate; feasibility of the point and of the ray and strict
//! improvement for an unbounded program. [`solve`] verifies its own outcome before returning it.
//!
//! [`vertices`] enumerates every choice of `n` hyperplanes among the constraints and finite
//! bounds, solves each square system and keeps the feasible distinct points.
//! [`irreducible_infeasible_groups`] runs a deletion filter over groups of constraints and
//! [`smallest_infeasible_groups`] searches group subsets by increasing size.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::cmp::Ordering;

use thiserror::Error;

/// Rows whose slope along a move is at most this (unit-scaled row, unit-scaled move) never block
/// the move.
const PIVOT_TOLERANCE: f64 = 1e-12;
/// A pivot at most this fraction of the largest entry marks a working matrix as singular.
const SINGULAR_RATIO: f64 = 1e-12;
/// A Householder column whose remaining norm falls to this fraction of its original norm is
/// dependent on the columns before it.
const DEPENDENT_COLUMN_RATIO: f64 = 1e-10;

/// Direction of a linear constraint.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConstraintSense {
    /// `coefficients . x <= rhs`.
    AtMost,
    /// `coefficients . x >= rhs`.
    AtLeast,
    /// `coefficients . x == rhs`.
    Equal,
}

/// One linear constraint `coefficients . x (<= | >= | ==) rhs`.
#[derive(Clone, Debug, PartialEq)]
pub struct LinearConstraint {
    /// One coefficient per variable.
    pub coefficients: Vec<f64>,
    /// Direction of the constraint.
    pub sense: ConstraintSense,
    /// Right-hand side.
    pub rhs: f64,
}

/// `minimize objective . x` subject to constraints and `lower[i] <= x[i] <= upper[i]`
/// (bounds may be infinite).
#[derive(Clone, Debug, PartialEq)]
pub struct LinearProgram {
    /// Objective coefficients; their number is the number of variables.
    pub objective: Vec<f64>,
    /// Linear constraints.
    pub constraints: Vec<LinearConstraint>,
    /// Lower bounds; `f64::NEG_INFINITY` for none.
    pub lower: Vec<f64>,
    /// Upper bounds; `f64::INFINITY` for none.
    pub upper: Vec<f64>,
}

/// An optimal point.
#[derive(Clone, Debug, PartialEq)]
pub struct LpSolution {
    /// Optimal point.
    pub x: Vec<f64>,
    /// `objective . x`.
    pub objective: f64,
    /// Indices of the constraints tight at `x` (within the feasibility tolerance), ascending.
    pub active: Vec<usize>,
}

/// Multipliers proving that constraints and bounds have no common point (a Farkas certificate).
///
/// Signs: `constraint_multipliers[i] >= 0` for an `AtMost` constraint, `<= 0` for an `AtLeast`
/// constraint, any sign for an `Equal` constraint; bound multipliers are `>= 0` and zero on an
/// infinite bound. With `r = sum_i y_i a_i - lower_multipliers + upper_multipliers` and
/// `q = sum_i y_i b_i - lower_multipliers . lower + upper_multipliers . upper`, every feasible
/// `x` would satisfy `r . x <= q`; the certificate has `r = 0` and `q < 0`.
#[derive(Clone, Debug, PartialEq)]
#[allow(
    clippy::struct_field_names,
    reason = "each field is the multiplier vector of one kind of row"
)]
pub struct InfeasibilityCertificate {
    /// One multiplier per constraint.
    pub constraint_multipliers: Vec<f64>,
    /// One multiplier per lower bound.
    pub lower_multipliers: Vec<f64>,
    /// One multiplier per upper bound.
    pub upper_multipliers: Vec<f64>,
}

impl InfeasibilityCertificate {
    fn zeros(constraints: usize, variables: usize) -> Self {
        Self {
            constraint_multipliers: vec![0.0; constraints],
            lower_multipliers: vec![0.0; variables],
            upper_multipliers: vec![0.0; variables],
        }
    }
}

/// Result of a linear program.
#[derive(Clone, Debug, PartialEq)]
pub enum LpOutcome {
    /// A finite optimum.
    Optimal(LpSolution),
    /// Non-negative multipliers proving infeasibility (a Farkas certificate) over constraints and
    /// bounds.
    Infeasible(InfeasibilityCertificate),
    /// The objective decreases without limit from `point` along `ray`.
    Unbounded {
        /// A feasible point.
        point: Vec<f64>,
        /// A direction that keeps every constraint and bound and strictly decreases the
        /// objective; maximum absolute entry one.
        ray: Vec<f64>,
    },
}

/// Tolerances and the work limit of [`solve`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LpOptions {
    /// Allowed violation of a constraint or bound, relative to the size of its terms.
    pub feasibility_tolerance: f64,
    /// Allowed wrong-signed multiplier, relative to the largest objective coefficient.
    pub optimality_tolerance: f64,
    /// Largest number of working-set exchanges over both phases.
    pub max_pivots: usize,
}

impl Default for LpOptions {
    fn default() -> Self {
        Self { feasibility_tolerance: 1e-9, optimality_tolerance: 1e-9, max_pivots: 100_000 }
    }
}

/// Refusals of the linear-program routines.
#[derive(Clone, Debug, PartialEq, Error)]
#[non_exhaustive]
pub enum LpError {
    /// `lower` or `upper` does not have one entry per variable.
    #[error("{variables} variables but {lower} lower and {upper} upper bounds")]
    BoundsLength {
        /// Number of variables.
        variables: usize,
        /// Number of lower bounds.
        lower: usize,
        /// Number of upper bounds.
        upper: usize,
    },
    /// A constraint does not have one coefficient per variable.
    #[error("constraint {constraint} has {found} coefficients for {variables} variables")]
    ConstraintLength {
        /// Constraint index.
        constraint: usize,
        /// Number of variables.
        variables: usize,
        /// Number of coefficients given.
        found: usize,
    },
    /// The group labels do not have one entry per constraint.
    #[error("{groups} group labels for {constraints} constraints")]
    GroupLength {
        /// Number of constraints.
        constraints: usize,
        /// Number of group labels.
        groups: usize,
    },
    /// An outcome passed to [`verify`] has a vector of the wrong length.
    #[error("outcome part {part} has length {found}, expected {expected}")]
    OutcomeLength {
        /// Which vector of the outcome.
        part: &'static str,
        /// Expected length.
        expected: usize,
        /// Length given.
        found: usize,
    },
    /// An objective coefficient is not finite.
    #[error("objective coefficient {variable} is not finite")]
    NonFiniteObjective {
        /// Variable index.
        variable: usize,
    },
    /// A constraint coefficient is not finite.
    #[error("coefficient {variable} of constraint {constraint} is not finite")]
    NonFiniteCoefficient {
        /// Constraint index.
        constraint: usize,
        /// Variable index.
        variable: usize,
    },
    /// A constraint right-hand side is not finite.
    #[error("right-hand side of constraint {constraint} is not finite")]
    NonFiniteRhs {
        /// Constraint index.
        constraint: usize,
    },
    /// A bound is NaN.
    #[error("a bound of variable {variable} is NaN")]
    NanBound {
        /// Variable index.
        variable: usize,
    },
    /// A lower bound of `+inf` or an upper bound of `-inf`.
    #[error("a bound of variable {variable} is infinite on the wrong side")]
    InfiniteBoundWrongSide {
        /// Variable index.
        variable: usize,
    },
    /// `lower > upper`.
    #[error("variable {variable} has lower bound {lower} above upper bound {upper}")]
    InvertedBounds {
        /// Variable index.
        variable: usize,
        /// Lower bound.
        lower: f64,
        /// Upper bound.
        upper: f64,
    },
    /// A tolerance is not a finite positive number.
    #[error("option {option} must be finite and positive")]
    InvalidOptions {
        /// Name of the option.
        option: &'static str,
    },
    /// More working-set exchanges than `max_pivots` were needed.
    #[error("more than {max_pivots} pivots")]
    PivotLimit {
        /// The limit that was reached.
        max_pivots: usize,
    },
    /// The pivoting lost the linear independence of its working rows.
    #[error("numerical breakdown during {stage}")]
    NumericalBreakdown {
        /// Where the breakdown happened.
        stage: &'static str,
    },
    /// An outcome failed its independent check.
    #[error("verification failed: {check}")]
    VerificationFailed {
        /// The check that failed.
        check: &'static str,
    },
    /// The region is unbounded along a coordinate, so it has no finite vertex list.
    #[error("the region is unbounded along variable {variable}")]
    UnboundedRegion {
        /// Variable index.
        variable: usize,
    },
    /// Vertex enumeration would solve more candidate bases than allowed.
    #[error("{candidate_bases} candidate bases exceed the cap of {cap}")]
    VertexCapExceeded {
        /// Number of candidate bases needed (saturated at `usize::MAX`).
        candidate_bases: usize,
        /// The cap given.
        cap: usize,
    },
}

impl LpError {
    /// Stable identifier of the refusal, in the family `linear_program`.
    #[must_use]
    pub fn detail(&self) -> &'static str {
        match self {
            Self::BoundsLength { .. } => "linear_program.bounds_length_mismatch",
            Self::ConstraintLength { .. } => "linear_program.constraint_length_mismatch",
            Self::GroupLength { .. } => "linear_program.group_length_mismatch",
            Self::OutcomeLength { .. } => "linear_program.outcome_length_mismatch",
            Self::NonFiniteObjective { .. } => "linear_program.non_finite_objective",
            Self::NonFiniteCoefficient { .. } => "linear_program.non_finite_coefficient",
            Self::NonFiniteRhs { .. } => "linear_program.non_finite_rhs",
            Self::NanBound { .. } => "linear_program.nan_bound",
            Self::InfiniteBoundWrongSide { .. } => "linear_program.infinite_bound_wrong_side",
            Self::InvertedBounds { .. } => "linear_program.inverted_bounds",
            Self::InvalidOptions { .. } => "linear_program.invalid_options",
            Self::PivotLimit { .. } => "linear_program.pivot_limit",
            Self::NumericalBreakdown { .. } => "linear_program.numerical_breakdown",
            Self::VerificationFailed { .. } => "linear_program.verification_failed",
            Self::UnboundedRegion { .. } => "linear_program.unbounded_region",
            Self::VertexCapExceeded { .. } => "linear_program.vertex_cap_exceeded",
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Validation and small vector helpers
// ---------------------------------------------------------------------------------------------

fn validate_options(options: LpOptions) -> Result<(), LpError> {
    let positive = |v: f64| v.is_finite() && v > 0.0;
    if !positive(options.feasibility_tolerance) {
        return Err(LpError::InvalidOptions { option: "feasibility_tolerance" });
    }
    if !positive(options.optimality_tolerance) {
        return Err(LpError::InvalidOptions { option: "optimality_tolerance" });
    }
    Ok(())
}

fn validate_region(
    variables: usize,
    constraints: &[LinearConstraint],
    lower: &[f64],
    upper: &[f64],
) -> Result<(), LpError> {
    if lower.len() != variables || upper.len() != variables {
        return Err(LpError::BoundsLength { variables, lower: lower.len(), upper: upper.len() });
    }
    for (constraint, row) in constraints.iter().enumerate() {
        if row.coefficients.len() != variables {
            return Err(LpError::ConstraintLength {
                constraint,
                variables,
                found: row.coefficients.len(),
            });
        }
        if let Some(variable) = row.coefficients.iter().position(|v| !v.is_finite()) {
            return Err(LpError::NonFiniteCoefficient { constraint, variable });
        }
        if !row.rhs.is_finite() {
            return Err(LpError::NonFiniteRhs { constraint });
        }
    }
    for variable in 0..variables {
        let (l, u) = (lower[variable], upper[variable]);
        if l.is_nan() || u.is_nan() {
            return Err(LpError::NanBound { variable });
        }
        if (l.is_infinite() && l > 0.0) || (u.is_infinite() && u < 0.0) {
            return Err(LpError::InfiniteBoundWrongSide { variable });
        }
        if l > u {
            return Err(LpError::InvertedBounds { variable, lower: l, upper: u });
        }
    }
    Ok(())
}

fn validate_program(program: &LinearProgram, options: LpOptions) -> Result<(), LpError> {
    validate_options(options)?;
    if let Some(variable) = program.objective.iter().position(|v| !v.is_finite()) {
        return Err(LpError::NonFiniteObjective { variable });
    }
    validate_region(program.objective.len(), &program.constraints, &program.lower, &program.upper)
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn max_abs(values: &[f64]) -> f64 {
    values.iter().fold(0.0_f64, |m, v| m.max(v.abs()))
}

/// Size of the terms of one constraint at `x`, the yardstick of its allowed violation.
fn row_size(coefficients: &[f64], rhs: f64, x: &[f64]) -> f64 {
    1.0 + rhs.abs() + max_abs(coefficients) * (1.0 + max_abs(x))
}

/// Signed amount by which `x` violates the constraint (positive means violated).
fn constraint_violation(constraint: &LinearConstraint, x: &[f64]) -> f64 {
    let residual = dot(&constraint.coefficients, x) - constraint.rhs;
    match constraint.sense {
        ConstraintSense::AtMost => residual,
        ConstraintSense::AtLeast => -residual,
        ConstraintSense::Equal => residual.abs(),
    }
}

/// Constraint indices tight at `x` within the feasibility tolerance, ascending.
fn tight_constraints(program: &LinearProgram, x: &[f64], tolerance: f64) -> Vec<usize> {
    program
        .constraints
        .iter()
        .enumerate()
        .filter(|(_, c)| {
            (dot(&c.coefficients, x) - c.rhs).abs()
                <= tolerance * row_size(&c.coefficients, c.rhs, x)
        })
        .map(|(i, _)| i)
        .collect()
}

// ---------------------------------------------------------------------------------------------
// Dense LU with partial pivoting (working matrices and vertex bases)
// ---------------------------------------------------------------------------------------------

/// `P A = L U` of a square row-major matrix; row `k` of `P A` is row `permutation[k]` of `A`.
struct Lu {
    factors: Vec<f64>,
    permutation: Vec<usize>,
    dim: usize,
}

impl Lu {
    /// `None` when a pivot falls to `SINGULAR_RATIO` times the largest entry or an entry is not
    /// finite.
    fn factor(mut a: Vec<f64>, dim: usize) -> Option<Self> {
        let scale = max_abs(&a);
        if !scale.is_finite() || (dim > 0 && scale <= 0.0) {
            return None;
        }
        let mut permutation: Vec<usize> = (0..dim).collect();
        for k in 0..dim {
            let mut pivot_row = k;
            let mut best = a[k * dim + k].abs();
            for i in k + 1..dim {
                let v = a[i * dim + k].abs();
                if v > best {
                    best = v;
                    pivot_row = i;
                }
            }
            if best <= SINGULAR_RATIO * scale {
                return None;
            }
            if pivot_row != k {
                for j in 0..dim {
                    a.swap(k * dim + j, pivot_row * dim + j);
                }
                permutation.swap(k, pivot_row);
            }
            let pivot = a[k * dim + k];
            for i in k + 1..dim {
                let factor = a[i * dim + k] / pivot;
                a[i * dim + k] = factor;
                for j in k + 1..dim {
                    a[i * dim + j] -= factor * a[k * dim + j];
                }
            }
        }
        Some(Self { factors: a, permutation, dim })
    }

    /// Solves `A x = b`.
    fn solve(&self, b: &[f64]) -> Vec<f64> {
        let (n, a) = (self.dim, &self.factors);
        let mut y: Vec<f64> = self.permutation.iter().map(|&p| b[p]).collect();
        for i in 0..n {
            let mut s = y[i];
            for j in 0..i {
                s -= a[i * n + j] * y[j];
            }
            y[i] = s;
        }
        for i in (0..n).rev() {
            let mut s = y[i];
            for j in i + 1..n {
                s -= a[i * n + j] * y[j];
            }
            y[i] = s / a[i * n + i];
        }
        y
    }

    /// Solves `A^T z = c` (`A^T = U^T L^T P`).
    fn solve_transposed(&self, c: &[f64]) -> Vec<f64> {
        let (n, a) = (self.dim, &self.factors);
        let mut w = c.to_vec();
        for i in 0..n {
            let mut s = w[i];
            for j in 0..i {
                s -= a[j * n + i] * w[j];
            }
            w[i] = s / a[i * n + i];
        }
        for i in (0..n).rev() {
            let mut s = w[i];
            for j in i + 1..n {
                s -= a[j * n + i] * w[j];
            }
            w[i] = s;
        }
        let mut z = vec![0.0; n];
        for (k, &p) in self.permutation.iter().enumerate() {
            z[p] = w[k];
        }
        z
    }
}

// ---------------------------------------------------------------------------------------------
// Working-set primal simplex
// ---------------------------------------------------------------------------------------------

/// Where an internal row `g . x <= h` comes from.
#[derive(Clone, Copy, Debug)]
enum RowOrigin {
    /// Constraint `index` (negated for the `>=` side), divided by `scale > 0`.
    Constraint { index: usize, negated: bool, scale: f64 },
    /// `-x_j <= -lower_j`.
    Lower(usize),
    /// `x_j <= upper_j`.
    Upper(usize),
}

/// Every constraint and finite bound as unit-scaled rows `g . x <= h`. Constraint rows come
/// first in constraint order, then bound rows in variable order; that order is the index order
/// of Bland's rule.
struct Rows {
    coefficients: Vec<Vec<f64>>,
    rhs: Vec<f64>,
    origin: Vec<RowOrigin>,
    lower_row: Vec<Option<usize>>,
    upper_row: Vec<Option<usize>>,
}

/// A member of the working set: a row held tight, or a placeholder `x_variable = value` for a
/// variable with no bound in the working set (its multiplier must vanish at an optimum).
#[derive(Clone, Copy, Debug, PartialEq)]
enum Entry {
    Row(usize),
    Free { variable: usize, value: f64 },
}

enum SimplexEnd {
    Optimal { x: Vec<f64>, multipliers: Vec<f64> },
    Unbounded { x: Vec<f64>, ray: Vec<f64> },
}

/// Rows of a program, or the index and multiplier of a constraint with no coefficients that no
/// point satisfies.
fn build_rows(program: &LinearProgram, tolerance: f64) -> Result<Rows, (usize, f64)> {
    let n = program.objective.len();
    let mut rows = Rows {
        coefficients: Vec::new(),
        rhs: Vec::new(),
        origin: Vec::new(),
        lower_row: vec![None; n],
        upper_row: vec![None; n],
    };
    for (index, constraint) in program.constraints.iter().enumerate() {
        let scale = max_abs(&constraint.coefficients);
        let sides: &[bool] = match constraint.sense {
            ConstraintSense::AtMost => &[false],
            ConstraintSense::AtLeast => &[true],
            ConstraintSense::Equal => &[false, true],
        };
        for &negated in sides {
            let sign = if negated { -1.0 } else { 1.0 };
            let h = sign * constraint.rhs;
            if scale <= 0.0 {
                // `0 <= h`: either always true or a contradiction on its own.
                if h < -tolerance * (1.0 + constraint.rhs.abs()) {
                    return Err((index, sign));
                }
            } else {
                rows.coefficients
                    .push(constraint.coefficients.iter().map(|v| sign * v / scale).collect());
                rows.rhs.push(h / scale);
                rows.origin.push(RowOrigin::Constraint { index, negated, scale });
            }
        }
    }
    for j in 0..n {
        if program.lower[j].is_finite() {
            let mut g = vec![0.0; n];
            g[j] = -1.0;
            rows.lower_row[j] = Some(rows.rhs.len());
            rows.coefficients.push(g);
            rows.rhs.push(-program.lower[j]);
            rows.origin.push(RowOrigin::Lower(j));
        }
        if program.upper[j].is_finite() {
            let mut g = vec![0.0; n];
            g[j] = 1.0;
            rows.upper_row[j] = Some(rows.rhs.len());
            rows.coefficients.push(g);
            rows.rhs.push(program.upper[j]);
            rows.origin.push(RowOrigin::Upper(j));
        }
    }
    Ok(rows)
}

/// The working matrix (row-major, one row per entry) and its right-hand side.
fn working_system(
    coefficients: &[Vec<f64>],
    rhs: &[f64],
    working: &[Entry],
    dim: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut matrix = vec![0.0; dim * dim];
    let mut values = vec![0.0; dim];
    for (r, entry) in working.iter().enumerate() {
        match *entry {
            Entry::Row(i) => {
                matrix[r * dim..(r + 1) * dim].copy_from_slice(&coefficients[i]);
                values[r] = rhs[i];
            }
            Entry::Free { variable, value } => {
                matrix[r * dim + variable] = 1.0;
                values[r] = value;
            }
        }
    }
    (matrix, values)
}

/// Working-set position to release, or `None` at an optimum. Placeholders with a non-zero
/// multiplier go first (smallest variable); then rows with a negative multiplier (smallest row
/// index: Bland's rule).
fn choose_release(working: &[Entry], multipliers: &[f64], tolerance: f64) -> Option<usize> {
    let mut best: Option<(usize, usize)> = None;
    for (r, entry) in working.iter().enumerate() {
        if let Entry::Free { variable, .. } = *entry
            && multipliers[r].abs() > tolerance
            && best.is_none_or(|(key, _)| variable < key)
        {
            best = Some((variable, r));
        }
    }
    if best.is_none() {
        for (r, entry) in working.iter().enumerate() {
            if let Entry::Row(i) = *entry
                && multipliers[r] < -tolerance
                && best.is_none_or(|(key, _)| i < key)
            {
                best = Some((i, r));
            }
        }
    }
    best.map(|(_, r)| r)
}

/// Row that blocks the move from `x` along `direction`, or `None` when nothing blocks.
///
/// Harris bound: the largest step that leaves every row violated by at most a quarter of the
/// feasibility tolerance; among the rows reached within that step the smallest index enters
/// (Bland's rule on exact ties). Rows already violated count as reached at step zero.
fn ratio_test(
    coefficients: &[Vec<f64>],
    rhs: &[f64],
    in_working: &[bool],
    x: &[f64],
    direction: &[f64],
    feasibility_tolerance: f64,
) -> Option<usize> {
    let shift = 0.25 * feasibility_tolerance;
    let mut bound = f64::INFINITY;
    let mut first = None;
    let mut slopes = vec![0.0; rhs.len()];
    let mut slacks = vec![0.0; rhs.len()];
    for i in 0..rhs.len() {
        if in_working[i] {
            continue;
        }
        let slope = dot(&coefficients[i], direction);
        if slope > PIVOT_TOLERANCE {
            let slack = (rhs[i] - dot(&coefficients[i], x)).max(0.0);
            slopes[i] = slope;
            slacks[i] = slack;
            let step = (slack + shift) / slope;
            if step < bound {
                bound = step;
                first = Some(i);
            }
        }
    }
    let first = first?;
    let chosen = (0..rhs.len())
        .find(|&i| !in_working[i] && slopes[i] > PIVOT_TOLERANCE && slacks[i] / slopes[i] <= bound);
    Some(chosen.unwrap_or(first))
}

/// Minimises `objective . x` over rows `g . x <= h` from a working set whose vertex is feasible.
/// Invariant: `working` holds `dim` entries whose matrix is non-singular, and its vertex keeps
/// every row within the Harris shift.
fn run_simplex(
    coefficients: &[Vec<f64>],
    rhs: &[f64],
    objective: &[f64],
    working: &mut [Entry],
    pivots: &mut usize,
    options: LpOptions,
) -> Result<SimplexEnd, LpError> {
    let dim = objective.len();
    let mut in_working = vec![false; rhs.len()];
    for entry in &*working {
        if let Entry::Row(i) = *entry {
            in_working[i] = true;
        }
    }
    let negated_objective: Vec<f64> = objective.iter().map(|v| -v).collect();
    loop {
        let (matrix, values) = working_system(coefficients, rhs, working, dim);
        let lu = Lu::factor(matrix, dim)
            .ok_or(LpError::NumericalBreakdown { stage: "working_matrix" })?;
        let x = lu.solve(&values);
        // objective = -W^T mu: a row with mu < 0 can be released to decrease the objective.
        let multipliers = lu.solve_transposed(&negated_objective);
        let Some(release) = choose_release(working, &multipliers, options.optimality_tolerance)
        else {
            return Ok(SimplexEnd::Optimal { x, multipliers });
        };
        // W d = -s e_r moves off row r; objective . d = s mu_r < 0 for s = -sign(mu_r).
        let s = if multipliers[release] > 0.0 { -1.0 } else { 1.0 };
        let mut unit = vec![0.0; dim];
        unit[release] = -s;
        let mut direction = lu.solve(&unit);
        let norm = max_abs(&direction);
        if !(norm.is_finite() && norm > 0.0) {
            return Err(LpError::NumericalBreakdown { stage: "direction" });
        }
        for v in &mut direction {
            *v /= norm;
        }
        let Some(entering) = ratio_test(
            coefficients,
            rhs,
            &in_working,
            &x,
            &direction,
            options.feasibility_tolerance,
        ) else {
            return Ok(SimplexEnd::Unbounded { x, ray: direction });
        };
        if let Entry::Row(i) = working[release] {
            in_working[i] = false;
        }
        working[release] = Entry::Row(entering);
        in_working[entering] = true;
        *pivots += 1;
        if *pivots > options.max_pivots {
            return Err(LpError::PivotLimit { max_pivots: options.max_pivots });
        }
    }
}

enum PhaseOne {
    Feasible(Vec<Entry>),
    Infeasible(InfeasibilityCertificate),
}

/// Adds the multiplier `mu >= 0` of internal row `origin` to the certificate in the caller's
/// units.
fn add_row_multiplier(certificate: &mut InfeasibilityCertificate, origin: RowOrigin, mu: f64) {
    match origin {
        RowOrigin::Constraint { index, negated, scale } => {
            let y = mu / scale;
            certificate.constraint_multipliers[index] += if negated { -y } else { y };
        }
        RowOrigin::Lower(j) => certificate.lower_multipliers[j] += mu,
        RowOrigin::Upper(j) => certificate.upper_multipliers[j] += mu,
    }
}

/// Minimises the auxiliary `t` of the relaxed rows `g . x - violation * t <= h`, `0 <= t <= 1`,
/// from the start working set plus `t <= 1`.
fn phase_one(
    rows: &Rows,
    violation: &[f64],
    largest: f64,
    mut working: Vec<Entry>,
    constraints: usize,
    pivots: &mut usize,
    options: LpOptions,
) -> Result<PhaseOne, LpError> {
    let n = working.len();
    let row_count = rows.rhs.len();
    let mut coefficients: Vec<Vec<f64>> = Vec::with_capacity(row_count + 2);
    for (row, v) in rows.coefficients.iter().zip(violation) {
        let mut relaxed = row.clone();
        relaxed.push(-v);
        coefficients.push(relaxed);
    }
    let mut rhs = rows.rhs.clone();
    let (t_lower_row, t_upper_row) = (row_count, row_count + 1);
    let mut t_lower = vec![0.0; n + 1];
    t_lower[n] = -1.0;
    coefficients.push(t_lower);
    rhs.push(0.0);
    let mut t_upper = vec![0.0; n + 1];
    t_upper[n] = 1.0;
    coefficients.push(t_upper);
    rhs.push(1.0);
    working.push(Entry::Row(t_upper_row));
    let mut objective = vec![0.0; n + 1];
    objective[n] = 1.0;

    let SimplexEnd::Optimal { x, multipliers } =
        run_simplex(&coefficients, &rhs, &objective, &mut working, pivots, options)?
    else {
        return Err(LpError::NumericalBreakdown { stage: "phase_one_ray" });
    };
    if x[n] * largest > 0.5 * options.feasibility_tolerance {
        // At the minimum t* > 0: objective + W^T mu = 0 gives sum mu_i g_i = 0 over the
        // original rows and sum mu_i h_i = -t* < 0.
        let mut certificate = InfeasibilityCertificate::zeros(constraints, n);
        for (r, entry) in working.iter().enumerate() {
            if let Entry::Row(i) = *entry
                && i < row_count
            {
                add_row_multiplier(&mut certificate, rows.origin[i], multipliers[r].max(0.0));
            }
        }
        return Ok(PhaseOne::Infeasible(certificate));
    }
    if !working.contains(&Entry::Row(t_lower_row)) {
        // Exchange `t >= 0` for the entry r with the largest |(W^-1)_{t r}|, which keeps the
        // working matrix non-singular; then t = 0 and the other rows define a vertex in x.
        let (matrix, _) = working_system(&coefficients, &rhs, &working, n + 1);
        let lu = Lu::factor(matrix, n + 1)
            .ok_or(LpError::NumericalBreakdown { stage: "phase_one_exit" })?;
        let mut unit = vec![0.0; n + 1];
        unit[n] = 1.0;
        let inverse_row = lu.solve_transposed(&unit);
        let mut position = 0;
        for (r, v) in inverse_row.iter().enumerate() {
            if v.abs() > inverse_row[position].abs() {
                position = r;
            }
        }
        working[position] = Entry::Row(t_lower_row);
    }
    working.retain(|entry| *entry != Entry::Row(t_lower_row));
    if working.len() != n || working.contains(&Entry::Row(t_upper_row)) {
        return Err(LpError::NumericalBreakdown { stage: "phase_one_exit" });
    }
    Ok(PhaseOne::Feasible(working))
}

/// Both phases, without validation or verification.
fn solve_unchecked(program: &LinearProgram, options: LpOptions) -> Result<LpOutcome, LpError> {
    let n = program.objective.len();
    let m = program.constraints.len();
    let tolerance = options.feasibility_tolerance;
    let rows = match build_rows(program, tolerance) {
        Ok(rows) => rows,
        Err((index, sign)) => {
            let mut certificate = InfeasibilityCertificate::zeros(m, n);
            certificate.constraint_multipliers[index] = sign;
            return Ok(LpOutcome::Infeasible(certificate));
        }
    };
    // Start: each variable at a finite bound (lower first), or a placeholder at zero.
    let mut start = vec![0.0; n];
    let mut working = Vec::with_capacity(n + 1);
    for j in 0..n {
        if let Some(row) = rows.lower_row[j] {
            start[j] = program.lower[j];
            working.push(Entry::Row(row));
        } else if let Some(row) = rows.upper_row[j] {
            start[j] = program.upper[j];
            working.push(Entry::Row(row));
        } else {
            working.push(Entry::Free { variable: j, value: 0.0 });
        }
    }
    let violation: Vec<f64> = rows
        .coefficients
        .iter()
        .zip(&rows.rhs)
        .map(|(g, h)| (dot(g, &start) - h).max(0.0))
        .collect();
    let largest = max_abs(&violation);
    let mut pivots = 0;
    if largest > 0.25 * tolerance {
        match phase_one(&rows, &violation, largest, working, m, &mut pivots, options)? {
            PhaseOne::Infeasible(certificate) => return Ok(LpOutcome::Infeasible(certificate)),
            PhaseOne::Feasible(feasible) => working = feasible,
        }
    }
    let scale = max_abs(&program.objective);
    let objective: Vec<f64> = if scale > 0.0 {
        program.objective.iter().map(|v| v / scale).collect()
    } else {
        vec![0.0; n]
    };
    match run_simplex(
        &rows.coefficients,
        &rows.rhs,
        &objective,
        &mut working,
        &mut pivots,
        options,
    )? {
        SimplexEnd::Optimal { x, .. } => {
            let objective = dot(&program.objective, &x);
            let active = tight_constraints(program, &x, tolerance);
            Ok(LpOutcome::Optimal(LpSolution { x, objective, active }))
        }
        SimplexEnd::Unbounded { x, ray } => Ok(LpOutcome::Unbounded { point: x, ray }),
    }
}

/// Minimises `program.objective . x` over the constraints and bounds.
///
/// The outcome has passed [`verify`] before it is returned.
///
/// # Errors
///
/// Shape, finiteness, bound and option refusals; [`LpError::PivotLimit`] when more than
/// `max_pivots` exchanges are needed; [`LpError::NumericalBreakdown`] when the working rows lose
/// independence; [`LpError::VerificationFailed`] when the outcome fails its own check.
pub fn solve(program: &LinearProgram, options: LpOptions) -> Result<LpOutcome, LpError> {
    validate_program(program, options)?;
    let outcome = solve_unchecked(program, options)?;
    verify(program, &outcome, options)?;
    Ok(outcome)
}

/// Maximises `program.objective . x`. An optimum reports the maximum value; an unbounded ray
/// increases the objective. The outcome is verified as the minimisation of the negated
/// objective, so pass that negated program to [`verify`] to recheck it.
///
/// # Errors
///
/// As [`solve`].
pub fn maximize(program: &LinearProgram, options: LpOptions) -> Result<LpOutcome, LpError> {
    let negated = LinearProgram {
        objective: program.objective.iter().map(|v| -v).collect(),
        ..program.clone()
    };
    match solve(&negated, options)? {
        LpOutcome::Optimal(mut solution) => {
            solution.objective = -solution.objective;
            Ok(LpOutcome::Optimal(solution))
        }
        other => Ok(other),
    }
}

// ---------------------------------------------------------------------------------------------
// Independent verification
// ---------------------------------------------------------------------------------------------

fn check(condition: bool, name: &'static str) -> Result<(), LpError> {
    if condition { Ok(()) } else { Err(LpError::VerificationFailed { check: name }) }
}

fn check_length(part: &'static str, expected: usize, found: usize) -> Result<(), LpError> {
    if expected == found { Ok(()) } else { Err(LpError::OutcomeLength { part, expected, found }) }
}

/// `x` satisfies every bound and constraint within the feasibility tolerance.
fn verify_point(program: &LinearProgram, x: &[f64], tolerance: f64) -> Result<(), LpError> {
    check_length("x", program.objective.len(), x.len())?;
    check(x.iter().all(|v| v.is_finite()), "non_finite_point")?;
    for (j, &v) in x.iter().enumerate() {
        let (l, u) = (program.lower[j], program.upper[j]);
        check(!l.is_finite() || v >= l - tolerance * (1.0 + l.abs()), "lower_bound_violated")?;
        check(!u.is_finite() || v <= u + tolerance * (1.0 + u.abs()), "upper_bound_violated")?;
    }
    for constraint in &program.constraints {
        let allowed = tolerance * row_size(&constraint.coefficients, constraint.rhs, x);
        check(constraint_violation(constraint, x) <= allowed, "constraint_violated")?;
    }
    Ok(())
}

/// Least squares `min ||A s - b||` over the columns `set` of `columns` by Householder QR;
/// `None` when the chosen columns are linearly dependent.
fn least_squares(columns: &[Vec<f64>], set: &[usize], target: &[f64]) -> Option<Vec<f64>> {
    let n = target.len();
    let p = set.len();
    if p > n {
        return None;
    }
    let mut a: Vec<Vec<f64>> = set.iter().map(|&i| columns[i].clone()).collect();
    let mut b = target.to_vec();
    for k in 0..p {
        let original = a[k].iter().map(|v| v * v).sum::<f64>().sqrt();
        let norm = a[k][k..].iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm <= DEPENDENT_COLUMN_RATIO * original || norm <= 0.0 {
            return None;
        }
        let alpha = if a[k][k] > 0.0 { -norm } else { norm };
        let mut v = a[k][k..].to_vec();
        v[0] -= alpha;
        let v_norm2 = dot(&v, &v);
        if v_norm2 > 0.0 {
            for column in a.iter_mut().skip(k) {
                let f = 2.0 * dot(&v, &column[k..]) / v_norm2;
                for (entry, vi) in column[k..].iter_mut().zip(&v) {
                    *entry -= f * vi;
                }
            }
            let f = 2.0 * dot(&v, &b[k..]) / v_norm2;
            for (entry, vi) in b[k..].iter_mut().zip(&v) {
                *entry -= f * vi;
            }
        }
    }
    let mut s = vec![0.0; p];
    for k in (0..p).rev() {
        let mut t = b[k];
        for j in k + 1..p {
            t -= a[j][k] * s[j];
        }
        s[k] = t / a[k][k];
    }
    Some(s)
}

fn combination_residual(columns: &[Vec<f64>], z: &[f64], target: &[f64]) -> Vec<f64> {
    let mut residual = target.to_vec();
    for (column, &weight) in columns.iter().zip(z) {
        for (r, c) in residual.iter_mut().zip(column) {
            *r -= weight * c;
        }
    }
    residual
}

/// Lawson-Hanson non-negative least squares: `min ||sum_k z_k columns[k] - target||`, `z >= 0`.
fn nonnegative_least_squares(columns: &[Vec<f64>], target: &[f64]) -> Vec<f64> {
    let k = columns.len();
    let mut z = vec![0.0; k];
    let mut passive = vec![false; k];
    // Columns whose entry failed (dependent on the passive set, or dropped at once); cleared
    // whenever the passive solution changes.
    let mut excluded = vec![false; k];
    let floor = 1e-13 * (1.0 + max_abs(target));
    for _ in 0..3 * (k + target.len()) + 10 {
        let residual = combination_residual(columns, &z, target);
        let mut entering = None;
        let mut best = floor;
        for j in 0..k {
            if !passive[j] && !excluded[j] {
                let gradient = dot(&columns[j], &residual);
                if gradient > best {
                    best = gradient;
                    entering = Some(j);
                }
            }
        }
        let Some(j) = entering else { break };
        passive[j] = true;
        for inner in 0..=k {
            let set: Vec<usize> = (0..k).filter(|&i| passive[i]).collect();
            let Some(s) = least_squares(columns, &set, target) else {
                passive[j] = false;
                excluded[j] = true;
                break;
            };
            if s.iter().all(|&v| v > 0.0) {
                for (&i, &v) in set.iter().zip(&s) {
                    z[i] = v;
                }
                excluded.fill(false);
                break;
            }
            // Move from z toward s until the first passive coefficient reaches zero.
            // At least one s_i <= 0 here, so a blocking coefficient exists and alpha <= 1.
            let mut alpha = f64::INFINITY;
            let mut blocking = None;
            for (&i, &v) in set.iter().zip(&s) {
                if v <= 0.0 {
                    let step = if z[i] > 0.0 { z[i] / (z[i] - v) } else { 0.0 };
                    if step < alpha {
                        alpha = step;
                        blocking = Some(i);
                    }
                }
            }
            let Some(blocking) = blocking else { break };
            for (&i, &v) in set.iter().zip(&s) {
                z[i] += alpha * (v - z[i]);
            }
            z[blocking] = 0.0;
            for &i in &set {
                if z[i] <= 0.0 {
                    z[i] = 0.0;
                    passive[i] = false;
                }
            }
            if inner == 0 && !passive[j] {
                excluded[j] = true;
                break;
            }
        }
    }
    z
}

fn verify_optimal(
    program: &LinearProgram,
    solution: &LpSolution,
    options: LpOptions,
) -> Result<(), LpError> {
    let tolerance = options.feasibility_tolerance;
    let x = &solution.x;
    verify_point(program, x, tolerance)?;
    let computed = dot(&program.objective, x);
    let magnitude: f64 = program.objective.iter().zip(x).map(|(c, v)| (c * v).abs()).sum();
    check(
        solution.objective.is_finite()
            && (solution.objective - computed).abs()
                <= options.optimality_tolerance * (1.0 + magnitude),
        "objective_mismatch",
    )?;
    let tight = tight_constraints(program, x, tolerance);
    check(solution.active.windows(2).all(|w| w[0] < w[1]), "active_not_ascending")?;
    check(solution.active.iter().all(|i| tight.binary_search(i).is_ok()), "active_not_tight")?;

    // KKT: -objective = sum mu_k g_k over the tight rows g_k . x <= h_k with mu >= 0.
    let n = x.len();
    let mut columns: Vec<Vec<f64>> = Vec::new();
    for &i in &tight {
        let constraint = &program.constraints[i];
        let scale = max_abs(&constraint.coefficients);
        if scale > 0.0 {
            let g: Vec<f64> = constraint.coefficients.iter().map(|v| v / scale).collect();
            if constraint.sense != ConstraintSense::AtLeast {
                columns.push(g.clone());
            }
            if constraint.sense != ConstraintSense::AtMost {
                columns.push(g.iter().map(|v| -v).collect());
            }
        }
    }
    for j in 0..n {
        let (l, u) = (program.lower[j], program.upper[j]);
        if l.is_finite() && (x[j] - l).abs() <= tolerance * (1.0 + l.abs()) {
            let mut g = vec![0.0; n];
            g[j] = -1.0;
            columns.push(g);
        }
        if u.is_finite() && (x[j] - u).abs() <= tolerance * (1.0 + u.abs()) {
            let mut g = vec![0.0; n];
            g[j] = 1.0;
            columns.push(g);
        }
    }
    let target: Vec<f64> = program.objective.iter().map(|v| -v).collect();
    let z = nonnegative_least_squares(&columns, &target);
    let residual = combination_residual(&columns, &z, &target);
    let allowed = options.optimality_tolerance * (n as f64 + 1.0) * max_abs(&program.objective);
    check(max_abs(&residual) <= allowed, "optimality_not_proven")
}

fn verify_infeasible(
    program: &LinearProgram,
    certificate: &InfeasibilityCertificate,
    options: LpOptions,
) -> Result<(), LpError> {
    let n = program.objective.len();
    let y = &certificate.constraint_multipliers;
    let (p, q) = (&certificate.lower_multipliers, &certificate.upper_multipliers);
    check_length("constraint_multipliers", program.constraints.len(), y.len())?;
    check_length("lower_multipliers", n, p.len())?;
    check_length("upper_multipliers", n, q.len())?;
    check(y.iter().chain(p).chain(q).all(|v| v.is_finite()), "non_finite_multiplier")?;
    for (constraint, &yi) in program.constraints.iter().zip(y) {
        let sign_ok = match constraint.sense {
            ConstraintSense::AtMost => yi >= 0.0,
            ConstraintSense::AtLeast => yi <= 0.0,
            ConstraintSense::Equal => true,
        };
        check(sign_ok, "multiplier_sign")?;
    }
    check(p.iter().chain(q).all(|&v| v >= 0.0), "multiplier_sign")?;
    // r = sum y_i a_i - p + q, rhs = sum y_i b_i - p . lower + q . upper, size = scale of terms.
    let mut r = vec![0.0; n];
    let mut rhs = 0.0;
    let mut size = 0.0;
    for (constraint, &yi) in program.constraints.iter().zip(y) {
        for (rj, a) in r.iter_mut().zip(&constraint.coefficients) {
            *rj += yi * a;
        }
        rhs += yi * constraint.rhs;
        size += yi.abs() * (max_abs(&constraint.coefficients) + constraint.rhs.abs());
    }
    for j in 0..n {
        let (l, u) = (program.lower[j], program.upper[j]);
        check(l.is_finite() || p[j] <= 0.0, "multiplier_on_infinite_bound")?;
        check(u.is_finite() || q[j] <= 0.0, "multiplier_on_infinite_bound")?;
        if l.is_finite() {
            r[j] -= p[j];
            rhs -= p[j] * l;
            size += p[j] * (1.0 + l.abs());
        }
        if u.is_finite() {
            r[j] += q[j];
            rhs += q[j] * u;
            size += q[j] * (1.0 + u.abs());
        }
    }
    let tolerance = options.feasibility_tolerance;
    check(size > 0.0, "empty_certificate")?;
    check(max_abs(&r) <= tolerance * size, "combination_not_zero")?;
    check(rhs < -tolerance * size, "no_contradiction")
}

fn verify_unbounded(
    program: &LinearProgram,
    point: &[f64],
    ray: &[f64],
    options: LpOptions,
) -> Result<(), LpError> {
    let tolerance = options.feasibility_tolerance;
    verify_point(program, point, tolerance)?;
    check_length("ray", program.objective.len(), ray.len())?;
    check(ray.iter().all(|v| v.is_finite()), "non_finite_ray")?;
    let length = max_abs(ray);
    check(length > 0.0, "zero_ray")?;
    for constraint in &program.constraints {
        let slope = dot(&constraint.coefficients, ray);
        let allowed = tolerance * max_abs(&constraint.coefficients) * length;
        let ok = match constraint.sense {
            ConstraintSense::AtMost => slope <= allowed,
            ConstraintSense::AtLeast => slope >= -allowed,
            ConstraintSense::Equal => slope.abs() <= allowed,
        };
        check(ok, "ray_leaves_constraint")?;
    }
    for (j, &d) in ray.iter().enumerate() {
        check(!program.lower[j].is_finite() || d >= -tolerance * length, "ray_leaves_bound")?;
        check(!program.upper[j].is_finite() || d <= tolerance * length, "ray_leaves_bound")?;
    }
    let improvement = dot(&program.objective, ray);
    check(
        improvement < -options.optimality_tolerance * max_abs(&program.objective) * length,
        "ray_not_improving",
    )
}

/// Rechecks an outcome of `minimize program.objective . x` from first principles.
///
/// `Optimal`: the point satisfies every constraint and bound within the feasibility tolerance,
/// the reported objective equals `objective . x`, every listed active constraint is tight and
/// listed once in ascending order, and the negated objective is a non-negative combination of
/// the tight rows (computed here by non-negative least squares) within the optimality tolerance.
/// `Infeasible`: the multipliers have the signs of [`InfeasibilityCertificate`] and combine the
/// constraints and bounds into `0 . x <= q` with `q < 0`. `Unbounded`: the point is feasible,
/// the ray keeps every constraint and bound and strictly decreases the objective.
///
/// # Errors
///
/// The refusals of [`solve`] for the program and options; [`LpError::OutcomeLength`] for an
/// outcome of the wrong shape; [`LpError::VerificationFailed`] naming the failed check.
pub fn verify(
    program: &LinearProgram,
    outcome: &LpOutcome,
    options: LpOptions,
) -> Result<(), LpError> {
    validate_program(program, options)?;
    match outcome {
        LpOutcome::Optimal(solution) => verify_optimal(program, solution, options),
        LpOutcome::Infeasible(certificate) => verify_infeasible(program, certificate, options),
        LpOutcome::Unbounded { point, ray } => verify_unbounded(program, point, ray, options),
    }
}

// ---------------------------------------------------------------------------------------------
// Vertex enumeration
// ---------------------------------------------------------------------------------------------

/// `C(total, choose)`, saturated at `usize::MAX`.
fn binomial(total: usize, choose: usize) -> usize {
    if choose > total {
        return 0;
    }
    // C(total, i) is non-decreasing for i up to total / 2, so saturation is final.
    let r = choose.min(total - choose);
    let mut value: u128 = 1;
    for i in 0..r {
        let Some(product) = value.checked_mul((total - i) as u128) else {
            return usize::MAX;
        };
        value = product / (i + 1) as u128;
        if value > usize::MAX as u128 {
            return usize::MAX;
        }
    }
    usize::try_from(value).unwrap_or(usize::MAX)
}

/// Advances `combination` (strictly increasing indices below `total`) to the next one in
/// lexicographic order; `false` after the last.
fn next_combination(combination: &mut [usize], total: usize) -> bool {
    let k = combination.len();
    for i in (0..k).rev() {
        if combination[i] < total - k + i {
            combination[i] += 1;
            for j in i + 1..k {
                combination[j] = combination[j - 1] + 1;
            }
            return true;
        }
    }
    false
}

fn lexicographic(a: &[f64], b: &[f64]) -> Ordering {
    for (x, y) in a.iter().zip(b) {
        let order = x.total_cmp(y);
        if order != Ordering::Equal {
            return order;
        }
    }
    a.len().cmp(&b.len())
}

fn same_point(a: &[f64], b: &[f64], tolerance: f64) -> bool {
    a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tolerance * (1.0 + x.abs().max(y.abs())))
}

/// All vertices of a bounded polytope `{x : constraints, lower <= x <= upper}`, deduplicated, in
/// lexicographic order. The number of variables is `lower.len()`.
///
/// Every choice of `n` hyperplanes among the constraints (`a . x = b`) and the finite bounds
/// (`x_j = lower_j`, `x_j = upper_j`) is solved; feasible solutions are kept once. An empty
/// region gives an empty list.
///
/// # Errors
///
/// The shape, finiteness, bound and option refusals of [`solve`];
/// [`LpError::UnboundedRegion`] when some coordinate is unbounded over the region (checked with
/// [`solve`] along each axis); [`LpError::VertexCapExceeded`] when more than `cap` candidate
/// bases would be solved.
pub fn vertices(
    constraints: &[LinearConstraint],
    lower: &[f64],
    upper: &[f64],
    cap: usize,
    options: LpOptions,
) -> Result<Vec<Vec<f64>>, LpError> {
    validate_options(options)?;
    let n = lower.len();
    validate_region(n, constraints, lower, upper)?;
    let mut program = LinearProgram {
        objective: vec![0.0; n],
        constraints: constraints.to_vec(),
        lower: lower.to_vec(),
        upper: upper.to_vec(),
    };
    if let LpOutcome::Infeasible(_) = solve(&program, options)? {
        return Ok(Vec::new());
    }
    for j in 0..n {
        for (sign, bounded) in [(1.0, lower[j].is_finite()), (-1.0, upper[j].is_finite())] {
            if !bounded {
                program.objective.fill(0.0);
                program.objective[j] = sign;
                if let LpOutcome::Unbounded { .. } = solve(&program, options)? {
                    return Err(LpError::UnboundedRegion { variable: j });
                }
            }
        }
    }
    program.objective.fill(0.0);

    // Hyperplanes scaled to unit maximum coefficient; a zero constraint never joins a basis.
    let mut planes: Vec<(Vec<f64>, f64)> = Vec::new();
    for constraint in constraints {
        let scale = max_abs(&constraint.coefficients);
        if scale > 0.0 {
            planes.push((
                constraint.coefficients.iter().map(|v| v / scale).collect(),
                constraint.rhs / scale,
            ));
        }
    }
    // Planes from here on are variable bounds: `pinned[k - first_bound]` is the variable and
    // the value it takes on that plane.
    let first_bound = planes.len();
    let mut pinned: Vec<(usize, f64)> = Vec::new();
    for j in 0..n {
        for bound in [lower[j], upper[j]] {
            if bound.is_finite() {
                let mut unit = vec![0.0; n];
                unit[j] = 1.0;
                planes.push((unit, bound));
                pinned.push((j, bound));
            }
        }
    }
    let candidate_bases = binomial(planes.len(), n);
    if candidate_bases > cap {
        return Err(LpError::VertexCapExceeded { candidate_bases, cap });
    }
    let mut found: Vec<Vec<f64>> = Vec::new();
    if n > planes.len() {
        return Ok(found);
    }
    let tolerance = options.feasibility_tolerance;
    let mut basis: Vec<usize> = (0..n).collect();
    loop {
        let mut matrix = Vec::with_capacity(n * n);
        let mut rhs = Vec::with_capacity(n);
        for &k in &basis {
            matrix.extend_from_slice(&planes[k].0);
            rhs.push(planes[k].1);
        }
        if let Some(lu) = Lu::factor(matrix, n) {
            let mut x = lu.solve(&rhs);
            // A variable on one of its bounds takes the bound exactly, so roundoff in the
            // solve cannot reorder vertices that share that coordinate.
            for &k in &basis {
                if k >= first_bound {
                    let (j, bound) = pinned[k - first_bound];
                    x[j] = bound;
                }
            }
            if verify_point(&program, &x, tolerance).is_ok()
                && !found.iter().any(|v| same_point(v, &x, tolerance))
            {
                found.push(x);
            }
        }
        if !next_combination(&mut basis, planes.len()) {
            break;
        }
    }
    found.sort_by(|a, b| lexicographic(a, b));
    Ok(found)
}

// ---------------------------------------------------------------------------------------------
// Conflicting groups of constraints
// ---------------------------------------------------------------------------------------------

/// Feasibility of subsets of constraint groups, with the bounds always kept.
struct GroupFeasibility<'a> {
    constraints: &'a [LinearConstraint],
    group: &'a [usize],
    lower: &'a [f64],
    upper: &'a [f64],
    options: LpOptions,
    /// Distinct group labels, ascending.
    groups: Vec<usize>,
}

impl<'a> GroupFeasibility<'a> {
    fn new(
        constraints: &'a [LinearConstraint],
        group: &'a [usize],
        lower: &'a [f64],
        upper: &'a [f64],
        options: LpOptions,
    ) -> Result<Self, LpError> {
        validate_options(options)?;
        validate_region(lower.len(), constraints, lower, upper)?;
        if group.len() != constraints.len() {
            return Err(LpError::GroupLength {
                constraints: constraints.len(),
                groups: group.len(),
            });
        }
        let mut groups = group.to_vec();
        groups.sort_unstable();
        groups.dedup();
        Ok(Self { constraints, group, lower, upper, options, groups })
    }

    /// Whether the constraints of the groups in `selected` (ascending) have a common point.
    fn feasible(&self, selected: &[usize]) -> Result<bool, LpError> {
        let program = LinearProgram {
            objective: vec![0.0; self.lower.len()],
            constraints: self
                .constraints
                .iter()
                .zip(self.group)
                .filter(|&(_, &g)| selected.binary_search(&g).is_ok())
                .map(|(c, _)| c.clone())
                .collect(),
            lower: self.lower.to_vec(),
            upper: self.upper.to_vec(),
        };
        Ok(!matches!(solve(&program, self.options)?, LpOutcome::Infeasible(_)))
    }
}

/// An irreducible set of constraint groups that is infeasible together with the bounds, or
/// `None` when every constraint holds at some point.
///
/// `group[i]` labels constraint `i`; bounds are always kept. Deletion filter in ascending group
/// order: a group is dropped when the remaining groups stay infeasible without it. The result
/// is ascending, and dropping any one of its groups makes it feasible.
///
/// # Errors
///
/// The refusals of [`solve`] for the region and options; [`LpError::GroupLength`] when `group`
/// does not label every constraint.
pub fn irreducible_infeasible_groups(
    constraints: &[LinearConstraint],
    group: &[usize],
    lower: &[f64],
    upper: &[f64],
    options: LpOptions,
) -> Result<Option<Vec<usize>>, LpError> {
    let search = GroupFeasibility::new(constraints, group, lower, upper, options)?;
    let mut kept = search.groups.clone();
    if search.feasible(&kept)? {
        return Ok(None);
    }
    for &candidate in &search.groups {
        let trial: Vec<usize> = kept.iter().copied().filter(|&g| g != candidate).collect();
        if !search.feasible(&trial)? {
            kept = trial;
        }
    }
    Ok(Some(kept))
}

/// A smallest infeasible set of constraint groups (with the bounds always kept), searching sizes
/// `1..=max_size`; among the smallest, the lexicographically first. `None` when the whole set is
/// feasible or no infeasible set of at most `max_size` groups exists.
///
/// # Errors
///
/// As [`irreducible_infeasible_groups`].
pub fn smallest_infeasible_groups(
    constraints: &[LinearConstraint],
    group: &[usize],
    lower: &[f64],
    upper: &[f64],
    max_size: usize,
    options: LpOptions,
) -> Result<Option<Vec<usize>>, LpError> {
    let search = GroupFeasibility::new(constraints, group, lower, upper, options)?;
    if search.feasible(&search.groups)? {
        return Ok(None);
    }
    let total = search.groups.len();
    for size in 1..=max_size.min(total) {
        let mut chosen: Vec<usize> = (0..size).collect();
        loop {
            let subset: Vec<usize> = chosen.iter().map(|&k| search.groups[k]).collect();
            if !search.feasible(&subset)? {
                return Ok(Some(subset));
            }
            if !next_combination(&mut chosen, total) {
                break;
            }
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lu_solves_both_orientations_with_pivoting() {
        // A needs a row exchange at the first step.
        let a = vec![0.0, 2.0, 1.0, 1.0, 1.0, 0.0, 3.0, 0.0, 1.0];
        let lu = Lu::factor(a.clone(), 3).unwrap();
        let x = lu.solve(&[5.0, 2.0, 7.0]);
        for i in 0..3 {
            let row: f64 = (0..3).map(|j| a[i * 3 + j] * x[j]).sum();
            assert!((row - [5.0, 2.0, 7.0][i]).abs() < 1e-12);
        }
        let z = lu.solve_transposed(&[1.0, -1.0, 2.0]);
        for j in 0..3 {
            let column: f64 = (0..3).map(|i| a[i * 3 + j] * z[i]).sum();
            assert!((column - [1.0, -1.0, 2.0][j]).abs() < 1e-12);
        }
        assert!(Lu::factor(vec![1.0, 2.0, 2.0, 4.0], 2).is_none());
        assert!(Lu::factor(Vec::new(), 0).is_some());
    }

    #[test]
    fn nonnegative_least_squares_matches_hand_solutions() {
        // target inside the cone of e1, e2: exact non-negative fit.
        let columns = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let z = nonnegative_least_squares(&columns, &[2.0, 3.0]);
        let r = combination_residual(&columns, &z, &[2.0, 3.0]);
        assert!(max_abs(&r) < 1e-12 && z.iter().all(|&v| v >= 0.0));
        // target outside the cone: best fit puts zero weight on e1.
        let z = nonnegative_least_squares(&columns[..2], &[-1.0, 3.0]);
        assert!(z[0].abs() < 1e-15 && (z[1] - 3.0).abs() < 1e-12);
        // no columns
        assert!(nonnegative_least_squares(&[], &[1.0]).is_empty());
    }

    #[test]
    fn least_squares_refuses_dependent_columns() {
        let columns = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
        assert!(least_squares(&columns, &[0, 1], &[1.0, 1.0]).is_none());
        let s = least_squares(&columns, &[0], &[1.0, 2.0]).unwrap();
        assert!((s[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn binomial_and_combinations_count_alike() {
        assert_eq!(binomial(5, 2), 10);
        assert_eq!(binomial(3, 0), 1);
        assert_eq!(binomial(2, 3), 0);
        assert_eq!(binomial(10_000, 40), usize::MAX);
        let mut combination = vec![0, 1, 2];
        let mut count = 1;
        while next_combination(&mut combination, 6) {
            count += 1;
        }
        assert_eq!(count, binomial(6, 3));
        let mut empty: Vec<usize> = Vec::new();
        assert!(!next_combination(&mut empty, 4));
    }

    #[test]
    fn ratio_test_takes_the_smallest_index_among_ties() {
        let coefficients = vec![vec![1.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];
        let rhs = vec![1.0, 1.0, 5.0];
        let entering =
            ratio_test(&coefficients, &rhs, &[false, false, false], &[0.0, 0.0], &[1.0, 0.0], 1e-9);
        assert_eq!(entering, Some(0));
        let none = ratio_test(
            &coefficients,
            &rhs,
            &[false, false, false],
            &[0.0, 0.0],
            &[-1.0, 0.0],
            1e-9,
        );
        assert_eq!(none, None);
    }
}
