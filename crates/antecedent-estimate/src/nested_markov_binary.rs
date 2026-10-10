//! Bounded binary nested-Markov likelihood pilot (2.3A A3, record
//! `2.3A.X4.binary_nested_markov_pilot`).
//!
//! An ordinary DAG factorization is **not** an ADMG likelihood: factorizing the
//! selected graph below as `P(x1) P(x2|x1) P(x3|x2) P(x4|x3)` would ignore the
//! bidirected edge and is a different (wrong) model. This module fits the nested
//! Markov model of one selected graph through its district (c-factor) Mobius
//! parameterization and compares the resulting target contrast with the
//! empirical plug-in of the same identifying formula.
//!
//! # Selected graph
//!
//! Four binary observed variables `X1..X4` (limit: at most 6 observed variables,
//! exactly this graph, exactly these four binary variables) with directed edges
//! `X1 -> X2 -> X3 -> X4` and the bidirected edge `X2 <-> X4` (the Verma graph).
//! Districts: `{X1}`, `{X3}`, `{X2, X4}`. Every other ADMG, regime or non-binary
//! domain is refused (`route_not_supported`, `nested_markov.outside_binary_pilot`).
//!
//! # Parameterization
//!
//! Following the head/tail Mobius parameterization of the discrete nested Markov
//! model (Evans and Richardson 2019, Bernoulli 25(2), arXiv:1511.06813; nested
//! Markov property and the Verma constraint, Richardson, Evans, Robins and Shpitser
//! 2023, Annals of Statistics 51(1), arXiv:1701.06686; the section and theorem
//! numbers are recorded at provenance time, this module derives every identity
//! below directly), with `a` the probability of level `0` of a variable:
//!
//! * `a    = P(X1 = 0)`                              (district `{X1}`, 1 parameter)
//! * `c[x2] = P(X3 = 0 | X2 = x2)`                    (district `{X3}`, 2 parameters)
//! * the c-factor of `{X2, X4}`: `Q(x2, x4 | x1, x3) = P(x2 | x1) P(x4 | x1, x2, x3)`
//!   with Mobius parameters `q2[x1] = Q(X2 = 0 | x1)` (head `{X2}`, tail `{X1}`),
//!   `q4[x3] = Q(X4 = 0 | x3)` (head `{X4}`, tail `{X3}`) and
//!   `g[x1][x3] = Q(X2 = 0, X4 = 0 | x1, x3)` (head `{X2, X4}`, tail `{X1, X3}`),
//!   cells `Q(0,0) = g`, `Q(0,1) = q2 - g`, `Q(1,0) = q4 - g`,
//!   `Q(1,1) = 1 - q2 - q4 + g`. Eleven free parameters in all.
//!
//! # Likelihood
//!
//! `P(x1,x2,x3,x4) = Pa(x1) * C(x3 | x2) * Q(x2, x4 | x1, x3)`. For `n` cell
//! counts the log-likelihood is `sum n_i ln P_i` over the 16 cells. Because
//! `sum_{x4} Q(x2, x4 | x1, x3) = q2[x1]` does not depend on `x3`, the sixteen
//! cells sum to one; that is checked numerically (within `1e-12`), never assumed.
//!
//! # Equality constraints
//!
//! The parameterization has 11 free parameters against 15 for the saturated
//! model, so four equalities hold on every law of the model (positive `P`):
//!
//! * (E1) `X3` independent of `X1` given `X2`: `P(x3=0|x1=0,x2) = P(x3=0|x1=1,x2)`;
//! * (E2) the Verma constraint: `sum_{x2} P(x2|x1) P(x4=0|x1,x2,x3)` does not
//!   depend on `x1`, for each `x3`.
//!
//! [`constraint_residuals`] reports all four residuals. Data whose empirical law
//! violates them are never fitted silently: the diagnostics carry the residuals
//! and a [`ConstraintStatus::ViolatedByData`] status, and
//! [`FitOptions::refuse_constraint_residual_above`] turns that into a refusal.
//!
//! # Positivity
//!
//! Every one of the 16 regime cell counts must be finite and strictly positive
//! (conditionals of the functional and of the constraints are defined), and the
//! fitted cells must stay at least `1e-9` from 0 and 1 (boundary refusal).
//!
//! # Estimand
//!
//! `E[X4 | do(X2 = 1)] - E[X4 | do(X2 = 0)]` from the observational regime
//! alone. The identifying formula is
//! `P(x4 | do(x2)) = sum_{x3} P(x3 | x2) sum_{x1,x2'} P(x1,x2') P(x4 | x1,x2',x3)`.
//! The model value is `sum_{x3} C(x3 | x2) (1 - q4[x3] if x4 = 1)`; the plug-in is
//! the same formula with empirical frequencies. They agree exactly on a law that
//! satisfies the constraints and differ when it does not.
//!
//! # Fit
//!
//! A law satisfying the constraints is its own maximum-likelihood fit. Otherwise
//! `a` and `c` are closed form and the c-factor block is maximized by profiling
//! `g` (a monotone score, bisection) inside block coordinate ascent over
//! `q2, q4` (a concave profile on a box), bounded by
//! [`FitOptions::max_iterations`] sweeps. Non-convergence is a typed refusal
//! keeping the diagnostics.
//!
//! # Status
//!
//! Nonparametric identification, likelihood fit and inferential status are
//! separate fields ([`PilotStatus`]). Calibration is unmeasured: no posterior
//! draws are produced and the interval route stays closed
//! ([`refuse_public_interval_route`]).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::error::{EstimationError, ExactF64, RefusalFields};
use antecedent_core::ExecutionContext;

/// Observed-variable limit of the pilot.
pub const MAX_OBSERVED: usize = 6;
/// Observed variables of the selected graph.
pub const PILOT_OBSERVED: usize = 4;
/// Cells of the binary observational regime.
pub const CELL_COUNT: usize = 16;
/// Distance from 0 and 1 below which a fitted probability is a boundary fit.
pub const BOUNDARY_TOLERANCE: f64 = 1e-9;
/// Default sweep bound.
pub const DEFAULT_MAX_ITERATIONS: usize = 50_000;
/// Largest sweep bound a caller may request.
pub const MAX_ITERATIONS_CAP: usize = 1_000_000;
/// Default convergence tolerance on the largest parameter change per sweep.
pub const DEFAULT_TOLERANCE: f64 = 1e-11;
/// Cell probabilities must sum to one within this.
pub const NORMALIZATION_TOLERANCE: f64 = 1e-12;
/// Detail of the refusal for anything outside the one selected pilot.
pub const OUTSIDE_PILOT: &str = "nested_markov.outside_binary_pilot";
/// Detail of the refusal of the public interval route.
pub const ROUTE_FROZEN: &str = "nested_markov.route_frozen";

const FEASIBLE_TOLERANCE: f64 = 1e-12;
const CONSTRAINT_TOLERANCE: f64 = 1e-9;
const BISECTION_STEPS: usize = 100;

fn outside(message: &str) -> EstimationError {
    EstimationError::refused(
        antecedent_core::reason_code!("route_not_supported"),
        format!("{OUTSIDE_PILOT}: {message}"),
    )
}

fn invalid(detail: &str, message: &str) -> EstimationError {
    EstimationError::refused(
        antecedent_core::reason_code!("invalid_argument"),
        format!("nested_markov.{detail}: {message}"),
    )
}

fn positivity(message: &str) -> EstimationError {
    EstimationError::refused(
        antecedent_core::reason_code!("transport_support_failure"),
        format!("nested_markov.positivity: {message}"),
    )
}

fn numerical(detail: &str, message: &str, fields: RefusalFields) -> EstimationError {
    EstimationError::refused_with_fields(
        antecedent_core::reason_code!("transport_numerical_failure"),
        format!("nested_markov.{detail}: {message}"),
        fields,
    )
}

/// The refusal of a numerical or posterior request on the public pilot route:
/// the exact functional, likelihood and calibration gates have not all passed.
#[must_use]
pub fn refuse_public_interval_route() -> EstimationError {
    EstimationError::refused(
        antecedent_core::reason_code!("cell_not_licensed"),
        format!(
            "{ROUTE_FROZEN}: the binary nested-Markov pilot is a closed research route; \
             calibration is unmeasured and no interval or posterior is published"
        ),
    )
}

/// Index of a cell; `x1` is the most significant digit.
#[must_use]
pub const fn cell_index(x1: usize, x2: usize, x3: usize, x4: usize) -> usize {
    x1 * 8 + x2 * 4 + x3 * 2 + x4
}

fn bern(p0: f64, level: usize) -> f64 {
    if level == 0 { p0 } else { 1.0 - p0 }
}

/// Variables, sorted directed edges and sorted unordered bidirected edges.
type CanonicalAdmg = (Vec<String>, Vec<(usize, usize)>, Vec<(usize, usize)>);

/// A declared ADMG over named observed variables (indices into `variables`).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdmgDeclaration {
    /// Variable names in coordinate order.
    pub variables: Vec<String>,
    /// Directed edges `(from, to)`.
    pub directed: Vec<(usize, usize)>,
    /// Bidirected edges (unordered pairs).
    pub bidirected: Vec<(usize, usize)>,
}

impl AdmgDeclaration {
    /// The one selected pilot graph: `X1 -> X2 -> X3 -> X4`, `X2 <-> X4`.
    #[must_use]
    pub fn selected() -> Self {
        Self {
            variables: ["X1", "X2", "X3", "X4"].iter().map(|s| (*s).to_owned()).collect(),
            directed: vec![(0, 1), (1, 2), (2, 3)],
            bidirected: vec![(1, 3)],
        }
    }

    fn canonical(&self) -> CanonicalAdmg {
        let mut directed = self.directed.clone();
        directed.sort_unstable();
        directed.dedup();
        let mut bidirected: Vec<(usize, usize)> =
            self.bidirected.iter().map(|&(a, b)| (a.min(b), a.max(b))).collect();
        bidirected.sort_unstable();
        bidirected.dedup();
        (self.variables.clone(), directed, bidirected)
    }

    /// Whether this is exactly the selected pilot graph.
    #[must_use]
    pub fn is_selected(&self) -> bool {
        self.canonical() == Self::selected().canonical()
    }
}

/// An evidence regime of a count table.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Regime {
    /// Passive observation (the only regime of the pilot).
    Observational,
    /// An experiment fixing the listed variables.
    Interventional(Vec<usize>),
}

/// Cell counts of one regime with their declared domain sizes.
#[derive(Clone, Debug, PartialEq)]
pub struct RegimeCounts {
    /// The regime the counts were collected under.
    pub regime: Regime,
    /// Declared number of levels per variable.
    pub levels: Vec<usize>,
    /// Cell counts, `x1` most significant.
    pub cells: Vec<f64>,
}

/// A request to the pilot: the declared graph and the regime-specific counts.
#[derive(Clone, Debug, PartialEq)]
pub struct NestedMarkovInput {
    /// Declared ADMG.
    pub graph: AdmgDeclaration,
    /// Regime-specific counts.
    pub regimes: Vec<RegimeCounts>,
}

/// Validated non-negative binary cell counts of the observational regime.
#[derive(Clone, Debug, PartialEq)]
pub struct BinaryCells {
    counts: [f64; CELL_COUNT],
    total: f64,
}

impl BinaryCells {
    /// Validate 16 finite non-negative counts with a positive total. Counts may be
    /// fractional (exact, infinite-sample tables).
    ///
    /// # Errors
    /// `route_not_supported` for a non-binary table size; `invalid_argument`
    /// (`nested_markov.invalid_counts`) for negative, non-finite or all-zero counts.
    pub fn new(cells: &[f64]) -> Result<Self, EstimationError> {
        let Ok(counts) = <[f64; CELL_COUNT]>::try_from(cells) else {
            return Err(outside("the table is not the 2x2x2x2 binary cell table"));
        };
        if counts.iter().any(|n| !n.is_finite() || *n < 0.0) {
            return Err(invalid("invalid_counts", "counts must be finite and non-negative"));
        }
        let total: f64 = counts.iter().sum();
        if !total.is_finite() || total <= 0.0 {
            return Err(invalid("invalid_counts", "the table has no mass"));
        }
        Ok(Self { counts, total })
    }

    /// The counts.
    #[must_use]
    pub const fn counts(&self) -> &[f64; CELL_COUNT] {
        &self.counts
    }

    /// Total count.
    #[must_use]
    pub const fn total(&self) -> f64 {
        self.total
    }

    /// The empirical law `counts / total`.
    #[must_use]
    pub fn empirical_law(&self) -> [f64; CELL_COUNT] {
        let mut law = self.counts;
        for p in &mut law {
            *p /= self.total;
        }
        law
    }
}

/// Check that a request is exactly the selected pilot and return its counts.
///
/// # Errors
/// `route_not_supported` / `nested_markov.outside_binary_pilot` for another
/// graph, more than six observed variables, a regime other than the observational
/// one, a non-binary domain or a table that is not 2x2x2x2; a count error as in
/// [`BinaryCells::new`]. Never a nonidentification claim.
pub fn binary_cells(input: &NestedMarkovInput) -> Result<BinaryCells, EstimationError> {
    if input.graph.variables.len() > MAX_OBSERVED {
        return Err(outside("more than 6 observed variables"));
    }
    if !input.graph.is_selected() {
        return Err(outside("the declared ADMG is not the selected pilot graph"));
    }
    let [only] = input.regimes.as_slice() else {
        return Err(outside("the pilot needs exactly the observational regime"));
    };
    if only.regime != Regime::Observational {
        return Err(outside("the regime is not the observational regime the functional cites"));
    }
    if only.levels.len() != PILOT_OBSERVED || only.levels.iter().any(|l| *l != 2) {
        return Err(outside("a variable domain is not binary"));
    }
    BinaryCells::new(&only.cells)
}

/// The nested-Markov (c-factor Mobius) parameters of the selected graph; each
/// entry is the probability of level `0`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NestedParameters {
    /// `P(X1 = 0)`.
    pub a: f64,
    /// `P(X3 = 0 | X2 = x2)`.
    pub c: [f64; 2],
    /// `Q(X2 = 0 | x1)`.
    pub q2: [f64; 2],
    /// `Q(X4 = 0 | x3)`.
    pub q4: [f64; 2],
    /// `Q(X2 = 0, X4 = 0 | x1, x3)`, indexed `[x1][x3]`.
    pub g: [[f64; 2]; 2],
}

impl NestedParameters {
    fn q24(&self, x2: usize, x4: usize, x1: usize, x3: usize) -> f64 {
        let g = self.g[x1][x3];
        let q2 = self.q2[x1];
        let q4 = self.q4[x3];
        match (x2, x4) {
            (0, 0) => g,
            (0, _) => q2 - g,
            (_, 0) => q4 - g,
            _ => 1.0 - q2 - q4 + g,
        }
    }

    /// The 16 cell probabilities `Pa(x1) C(x3|x2) Q(x2,x4|x1,x3)`.
    #[must_use]
    pub fn cell_probabilities(&self) -> [f64; CELL_COUNT] {
        let mut cells = [0.0; CELL_COUNT];
        for x1 in 0..2 {
            for x2 in 0..2 {
                for x3 in 0..2 {
                    for x4 in 0..2 {
                        cells[cell_index(x1, x2, x3, x4)] =
                            bern(self.a, x1) * bern(self.c[x2], x3) * self.q24(x2, x4, x1, x3);
                    }
                }
            }
        }
        cells
    }

    /// The probabilities themselves (all parameters), for boundary diagnostics.
    fn probabilities(&self) -> Vec<f64> {
        let mut all = vec![self.a];
        all.extend(self.c);
        all.extend(self.q2);
        all.extend(self.q4);
        all
    }
}

/// Normalized law with the marginals the functional and constraints use.
struct LawMarginals {
    law: [f64; CELL_COUNT],
    p12: [[f64; 2]; 2],
    p123: [[[f64; 2]; 2]; 2],
}

impl LawMarginals {
    fn new(law: &[f64; CELL_COUNT]) -> Result<Self, EstimationError> {
        let total: f64 = law.iter().sum();
        if !total.is_finite() || total <= 0.0 {
            return Err(positivity("the law has no mass"));
        }
        let mut normalized = *law;
        for p in &mut normalized {
            *p /= total;
        }
        if normalized.iter().any(|p| p.is_nan() || *p <= 0.0) {
            return Err(positivity("a cell of the law has zero mass"));
        }
        let mut p12 = [[0.0; 2]; 2];
        let mut p123 = [[[0.0; 2]; 2]; 2];
        for x1 in 0..2 {
            for x2 in 0..2 {
                for x3 in 0..2 {
                    let mass = normalized[cell_index(x1, x2, x3, 0)]
                        + normalized[cell_index(x1, x2, x3, 1)];
                    p123[x1][x2][x3] = mass;
                    p12[x1][x2] += mass;
                }
            }
        }
        Ok(Self { law: normalized, p12, p123 })
    }

    /// `P(x4 = level | x1, x2, x3)`.
    fn p4(&self, level: usize, x1: usize, x2: usize, x3: usize) -> f64 {
        self.law[cell_index(x1, x2, x3, level)] / self.p123[x1][x2][x3]
    }
}

/// The four equality-constraint residuals of a law.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConstraintResiduals {
    /// (E1) `P(x3=0|x1=0,x2) - P(x3=0|x1=1,x2)` for `x2 = 0, 1`.
    pub x3_independent_of_x1: [f64; 2],
    /// (E2) Verma: `R(0, x3) - R(1, x3)` for `x3 = 0, 1`.
    pub verma: [f64; 2],
    /// Largest absolute residual.
    pub max_abs: f64,
}

/// Residuals of the nested-Markov equality constraints on a law (normalized
/// internally; every cell must be positive).
///
/// # Errors
/// `transport_support_failure` / `nested_markov.positivity` for a zero cell.
pub fn constraint_residuals(
    law: &[f64; CELL_COUNT],
) -> Result<ConstraintResiduals, EstimationError> {
    let m = LawMarginals::new(law)?;
    let mut e1 = [0.0; 2];
    for (x2, slot) in e1.iter_mut().enumerate() {
        *slot = m.p123[0][x2][0] / m.p12[0][x2] - m.p123[1][x2][0] / m.p12[1][x2];
    }
    let r = |x1: usize, x3: usize| -> f64 {
        let norm = m.p12[x1][0] + m.p12[x1][1];
        (0..2).map(|x2| m.p12[x1][x2] / norm * m.p4(0, x1, x2, x3)).sum()
    };
    let verma = [r(0, 0) - r(1, 0), r(0, 1) - r(1, 1)];
    let max_abs = e1.iter().chain(verma.iter()).fold(0.0_f64, |acc, v| acc.max(v.abs()));
    Ok(ConstraintResiduals { x3_independent_of_x1: e1, verma, max_abs })
}

/// Parameters read off a positive law: exact when the law satisfies the
/// constraints (see [`constraint_residuals`]), a projection onto the formula's
/// blocks otherwise.
///
/// # Errors
/// `transport_support_failure` for a zero cell.
pub fn parameters_from_law(law: &[f64; CELL_COUNT]) -> Result<NestedParameters, EstimationError> {
    let m = LawMarginals::new(law)?;
    let a = (m.p12[0][0] + m.p12[0][1]) / (m.p12[0][0] + m.p12[0][1] + m.p12[1][0] + m.p12[1][1]);
    let mut c = [0.0; 2];
    for (x2, slot) in c.iter_mut().enumerate() {
        *slot = (m.p123[0][x2][0] + m.p123[1][x2][0]) / (m.p12[0][x2] + m.p12[1][x2]);
    }
    let mut q2 = [0.0; 2];
    for (x1, slot) in q2.iter_mut().enumerate() {
        *slot = m.p12[x1][0] / (m.p12[x1][0] + m.p12[x1][1]);
    }
    let mut g = [[0.0; 2]; 2];
    let mut q4 = [0.0; 2];
    for (x3, q4_slot) in q4.iter_mut().enumerate() {
        for (x1, &q2_x1) in q2.iter().enumerate() {
            g[x1][x3] = q2_x1 * m.p4(0, x1, 0, x3);
            *q4_slot += m.p12[x1][0] * m.p4(0, x1, 0, x3) + m.p12[x1][1] * m.p4(0, x1, 1, x3);
        }
    }
    Ok(NestedParameters { a, c, q2, q4, g })
}

/// Result of the normalization and positivity check of fitted cells.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LikelihoodCheck {
    /// Sum of the 16 cell probabilities.
    pub total: f64,
    /// `|total - 1|`.
    pub normalization_error: f64,
    /// Smallest cell probability.
    pub min_cell: f64,
    /// Whether the cells sum to one within [`NORMALIZATION_TOLERANCE`].
    pub normalized: bool,
    /// Whether every cell is strictly positive.
    pub positive: bool,
}

/// Normalization and positivity of a set of cell probabilities.
#[must_use]
pub fn check_likelihood(cells: &[f64; CELL_COUNT]) -> LikelihoodCheck {
    let total: f64 = cells.iter().sum();
    let error = (total - 1.0).abs();
    let min_cell = cells.iter().copied().fold(f64::INFINITY, f64::min);
    LikelihoodCheck {
        total,
        normalization_error: error,
        min_cell,
        normalized: error <= NORMALIZATION_TOLERANCE,
        positive: min_cell > 0.0,
    }
}

/// Exact log-likelihood `sum n_i ln p_i` (zero-count cells contribute zero; a
/// positive count on a non-positive probability gives negative infinity).
#[must_use]
pub fn log_likelihood(counts: &[f64; CELL_COUNT], cells: &[f64; CELL_COUNT]) -> f64 {
    counts
        .iter()
        .zip(cells.iter())
        .map(|(n, p)| {
            if *n <= 0.0 {
                0.0
            } else if *p > 0.0 {
                n * p.ln()
            } else {
                f64::NEG_INFINITY
            }
        })
        .sum()
}

/// How the fit was obtained.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FitMethod {
    /// The empirical law satisfies the constraints and is its own fit.
    SaturatedFeasible,
    /// Profile-likelihood block coordinate ascent on the c-factor block.
    CoordinateAscent,
}

/// Whether the empirical law satisfies the equality constraints.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConstraintStatus {
    /// Max residual within `1e-9`.
    Satisfied,
    /// The data violate a constraint; the fit is the constrained projection.
    ViolatedByData,
}

/// Fit options with explicit bounds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FitOptions {
    /// Sweep bound, `1..=MAX_ITERATIONS_CAP`.
    pub max_iterations: usize,
    /// Convergence tolerance on the largest parameter change in a sweep.
    pub tolerance: f64,
    /// When set, refuse (rather than fit) data whose empirical constraint
    /// residual exceeds this value.
    pub refuse_constraint_residual_above: Option<f64>,
}

impl Default for FitOptions {
    fn default() -> Self {
        Self {
            max_iterations: DEFAULT_MAX_ITERATIONS,
            tolerance: DEFAULT_TOLERANCE,
            refuse_constraint_residual_above: None,
        }
    }
}

/// Fit diagnostics, retained on success and on numerical refusal.
#[derive(Clone, Debug, PartialEq)]
pub struct FitDiagnostics {
    /// How the fit was obtained.
    pub method: FitMethod,
    /// Sweeps run (zero for the saturated-feasible fit).
    pub iterations: usize,
    /// Whether the fit converged.
    pub converged: bool,
    /// Largest parameter change of the last sweep.
    pub final_change: Option<f64>,
    /// Log-likelihood of the saturated (empirical) model.
    pub saturated_log_likelihood: f64,
    /// Log-likelihood of the nested model, when a fit was produced.
    pub model_log_likelihood: Option<f64>,
    /// `2 (saturated - model)`, when a fit was produced.
    pub deviance: Option<f64>,
    /// Constraint residuals of the empirical law.
    pub empirical_residuals: ConstraintResiduals,
    /// Whether the data satisfy the constraints.
    pub constraint_status: ConstraintStatus,
    /// Normalization and positivity of the fitted cells.
    pub normalization: Option<LikelihoodCheck>,
    /// Smallest `min(p, 1 - p)` over fitted cells and parameters.
    pub boundary_margin: Option<f64>,
    /// Fitted probabilities within [`BOUNDARY_TOLERANCE`] of 0 or 1.
    pub boundary_count: Option<usize>,
}

/// A refusal with the diagnostics the failing step produced.
#[derive(Clone, Debug, PartialEq)]
pub struct NestedMarkovRefusal {
    /// The typed refusal.
    pub error: EstimationError,
    /// Diagnostics retained from the failing fit, when any were measured.
    pub diagnostics: Option<FitDiagnostics>,
}

type Refusal = Box<NestedMarkovRefusal>;

fn plain(error: EstimationError) -> Refusal {
    Box::new(NestedMarkovRefusal { error, diagnostics: None })
}

fn with(error: EstimationError, diagnostics: &FitDiagnostics) -> Refusal {
    Box::new(NestedMarkovRefusal { error, diagnostics: Some(diagnostics.clone()) })
}

/// A fitted nested-Markov model.
#[derive(Clone, Debug, PartialEq)]
pub struct NestedMarkovFit {
    /// Fitted parameters.
    pub parameters: NestedParameters,
    /// Fitted cell probabilities.
    pub cells: [f64; CELL_COUNT],
    /// Exact log-likelihood at the fit.
    pub log_likelihood: f64,
    /// Diagnostics.
    pub diagnostics: FitDiagnostics,
}

fn tables(counts: &[f64; CELL_COUNT]) -> [[[f64; 4]; 2]; 2] {
    let mut out = [[[0.0; 4]; 2]; 2];
    for x1 in 0..2 {
        for x3 in 0..2 {
            out[x1][x3] = [
                counts[cell_index(x1, 0, x3, 0)],
                counts[cell_index(x1, 0, x3, 1)],
                counts[cell_index(x1, 1, x3, 0)],
                counts[cell_index(x1, 1, x3, 1)],
            ];
        }
    }
    out
}

fn guard(x: f64) -> f64 {
    x.max(f64::MIN_POSITIVE)
}

/// Maximizer of one 2x2 table's log-likelihood in `g` at fixed margins: the
/// score is strictly decreasing on `(max(0, q2+q4-1), min(q2, q4))`.
fn solve_g(n: &[f64; 4], q2: f64, q4: f64) -> f64 {
    let mut lo = (q2 + q4 - 1.0).max(0.0);
    let mut hi = q2.min(q4);
    if hi <= lo {
        return 0.5 * (lo + hi);
    }
    for _ in 0..BISECTION_STEPS {
        let g = 0.5 * (lo + hi);
        let score = n[0] / guard(g) - n[1] / guard(q2 - g) - n[2] / guard(q4 - g)
            + n[3] / guard(1.0 - q2 - q4 + g);
        if score > 0.0 {
            lo = g;
        } else {
            hi = g;
        }
        if hi - lo <= 1e-17 {
            break;
        }
    }
    0.5 * (lo + hi)
}

fn bisect_unit(score: impl Fn(f64) -> f64) -> f64 {
    let mut lo = 0.0_f64;
    let mut hi = 1.0_f64;
    for _ in 0..BISECTION_STEPS {
        let mid = 0.5 * (lo + hi);
        if score(mid) > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
        if hi - lo <= 1e-17 {
            break;
        }
    }
    0.5 * (lo + hi)
}

fn q2_score(tabs: &[[[f64; 4]; 2]; 2], x1: usize, q2: f64, q4: [f64; 2]) -> f64 {
    (0..2)
        .map(|x3| {
            let n = &tabs[x1][x3];
            let g = solve_g(n, q2, q4[x3]);
            n[1] / guard(q2 - g) - n[3] / guard(1.0 - q2 - q4[x3] + g)
        })
        .sum()
}

fn q4_score(tabs: &[[[f64; 4]; 2]; 2], x3: usize, q4: f64, q2: [f64; 2]) -> f64 {
    (0..2)
        .map(|x1| {
            let n = &tabs[x1][x3];
            let g = solve_g(n, q2[x1], q4);
            n[2] / guard(q4 - g) - n[3] / guard(1.0 - q2[x1] - q4 + g)
        })
        .sum()
}

/// Pooled empirical starting margins of the c-factor block.
fn start_margins(tabs: &[[[f64; 4]; 2]; 2]) -> ([f64; 2], [f64; 2]) {
    let mut q2 = [0.5; 2];
    let mut q4 = [0.5; 2];
    for (x1, slot) in q2.iter_mut().enumerate() {
        let zero: f64 = (0..2).map(|x3| tabs[x1][x3][0] + tabs[x1][x3][1]).sum();
        let all: f64 = (0..2).map(|x3| tabs[x1][x3].iter().sum::<f64>()).sum();
        *slot = zero / all;
    }
    for (x3, slot) in q4.iter_mut().enumerate() {
        let zero: f64 = (0..2).map(|x1| tabs[x1][x3][0] + tabs[x1][x3][2]).sum();
        let all: f64 = (0..2).map(|x1| tabs[x1][x3].iter().sum::<f64>()).sum();
        *slot = zero / all;
    }
    (q2, q4)
}

fn closed_form_blocks(counts: &[f64; CELL_COUNT], total: f64) -> (f64, [f64; 2]) {
    let mut x1_zero = 0.0;
    let mut x2_mass = [0.0; 2];
    let mut x2_x3_zero = [0.0; 2];
    for x1 in 0..2 {
        for x2 in 0..2 {
            for x3 in 0..2 {
                let n = counts[cell_index(x1, x2, x3, 0)] + counts[cell_index(x1, x2, x3, 1)];
                if x1 == 0 {
                    x1_zero += n;
                }
                x2_mass[x2] += n;
                if x3 == 0 {
                    x2_x3_zero[x2] += n;
                }
            }
        }
    }
    (x1_zero / total, [x2_x3_zero[0] / x2_mass[0], x2_x3_zero[1] / x2_mass[1]])
}

fn boundary_stats(params: &NestedParameters, cells: &[f64; CELL_COUNT]) -> (f64, usize) {
    let mut margin = f64::INFINITY;
    let mut count = 0;
    for p in cells.iter().copied().chain(params.probabilities()) {
        let m = p.min(1.0 - p);
        margin = margin.min(m);
        if m.is_nan() || m <= BOUNDARY_TOLERANCE {
            count += 1;
        }
    }
    (margin, count)
}

fn cancelled(diagnostics: &FitDiagnostics) -> Refusal {
    with(
        EstimationError::refused(
            antecedent_core::reason_code!("transport_budget_cancel"),
            "nested_markov.cancelled: the fit was cancelled; no estimate is reported",
        ),
        diagnostics,
    )
}

fn validate_options(options: &FitOptions) -> Result<(), EstimationError> {
    if options.max_iterations == 0 || options.max_iterations > MAX_ITERATIONS_CAP {
        return Err(invalid("iteration_bound", "max_iterations must be in 1..=1000000"));
    }
    if !options.tolerance.is_finite() || options.tolerance <= 0.0 {
        return Err(invalid("invalid_options", "tolerance must be finite and positive"));
    }
    if options.refuse_constraint_residual_above.is_some_and(|t| !t.is_finite() || t < 0.0) {
        return Err(invalid("invalid_options", "the residual threshold must be finite, >= 0"));
    }
    Ok(())
}

/// Fit the nested-Markov model to the observational counts.
///
/// # Errors
/// `invalid_argument` for bad options; `transport_support_failure`
/// (`nested_markov.positivity`) for a non-positive count;
/// `transport_numerical_failure` with `nested_markov.constraint_violated` (only
/// when [`FitOptions::refuse_constraint_residual_above`] is set),
/// `nested_markov.fit_not_converged` (sweep bound reached),
/// `nested_markov.not_normalized` or `nested_markov.fitted_boundary`, each
/// keeping the diagnostics; `transport_budget_cancel` when cancelled.
pub fn fit_nested_markov(
    counts: &BinaryCells,
    options: &FitOptions,
    ctx: &ExecutionContext,
) -> Result<NestedMarkovFit, Refusal> {
    validate_options(options).map_err(plain)?;
    if ctx.cancellation.is_cancelled() {
        return Err(plain(EstimationError::refused(
            antecedent_core::reason_code!("transport_budget_cancel"),
            "nested_markov.cancelled: the fit was cancelled; no estimate is reported",
        )));
    }
    let n = counts.counts();
    if n.iter().any(|v| *v <= 0.0) {
        return Err(plain(positivity("every regime cell count must be strictly positive")));
    }
    let law = counts.empirical_law();
    let residuals = constraint_residuals(&law).map_err(plain)?;
    let saturated = log_likelihood(n, &law);
    let status = if residuals.max_abs <= CONSTRAINT_TOLERANCE {
        ConstraintStatus::Satisfied
    } else {
        ConstraintStatus::ViolatedByData
    };
    let mut diag = FitDiagnostics {
        method: FitMethod::CoordinateAscent,
        iterations: 0,
        converged: false,
        final_change: None,
        saturated_log_likelihood: saturated,
        model_log_likelihood: None,
        deviance: None,
        empirical_residuals: residuals,
        constraint_status: status,
        normalization: None,
        boundary_margin: None,
        boundary_count: None,
    };
    if let Some(limit) = options.refuse_constraint_residual_above {
        if residuals.max_abs > limit {
            return Err(with(
                numerical(
                    "constraint_violated",
                    "the empirical law violates the nested-Markov equality constraints",
                    RefusalFields {
                        stage: Some("nested_markov.fit".to_owned()),
                        reason: Some("equality constraint violated by the data".to_owned()),
                        ..RefusalFields::default()
                    },
                ),
                &diag,
            ));
        }
    }
    let direct = parameters_from_law(&law).map_err(plain)?;
    let rebuilt = direct.cell_probabilities();
    let feasible = rebuilt.iter().zip(law.iter()).all(|(r, p)| (r - p).abs() <= FEASIBLE_TOLERANCE);
    let params = if feasible {
        diag.method = FitMethod::SaturatedFeasible;
        diag.converged = true;
        direct
    } else {
        ascend(counts, options, ctx, &mut diag)?
    };
    finish(counts, params, diag)
}

fn ascend(
    counts: &BinaryCells,
    options: &FitOptions,
    ctx: &ExecutionContext,
    diag: &mut FitDiagnostics,
) -> Result<NestedParameters, Refusal> {
    let tabs = tables(counts.counts());
    let (mut q2, mut q4) = start_margins(&tabs);
    for sweep in 1..=options.max_iterations {
        if ctx.cancellation.is_cancelled() {
            return Err(cancelled(diag));
        }
        let mut change = 0.0_f64;
        for (x1, slot) in q2.iter_mut().enumerate() {
            let fixed = q4;
            let next = bisect_unit(|q| q2_score(&tabs, x1, q, fixed));
            change = change.max((next - *slot).abs());
            *slot = next;
        }
        for (x3, slot) in q4.iter_mut().enumerate() {
            let fixed = q2;
            let next = bisect_unit(|q| q4_score(&tabs, x3, q, fixed));
            change = change.max((next - *slot).abs());
            *slot = next;
        }
        diag.iterations = sweep;
        diag.final_change = Some(change);
        if change <= options.tolerance {
            diag.converged = true;
            break;
        }
    }
    if !diag.converged {
        return Err(with(
            numerical(
                "fit_not_converged",
                "the constrained fit did not converge within the iteration bound",
                RefusalFields {
                    stage: Some("nested_markov.fit".to_owned()),
                    reason: Some("iteration bound reached".to_owned()),
                    ..RefusalFields::default()
                },
            ),
            diag,
        ));
    }
    let (a, c) = closed_form_blocks(counts.counts(), counts.total());
    let mut g = [[0.0; 2]; 2];
    for (x1, (g_row, &q2_x1)) in g.iter_mut().zip(&q2).enumerate() {
        for (x3, (slot, &q4_x3)) in g_row.iter_mut().zip(&q4).enumerate() {
            *slot = solve_g(&tabs[x1][x3], q2_x1, q4_x3);
        }
    }
    Ok(NestedParameters { a, c, q2, q4, g })
}

fn finish(
    counts: &BinaryCells,
    parameters: NestedParameters,
    mut diag: FitDiagnostics,
) -> Result<NestedMarkovFit, Refusal> {
    let cells = parameters.cell_probabilities();
    let check = check_likelihood(&cells);
    diag.normalization = Some(check);
    let ll = log_likelihood(counts.counts(), &cells);
    diag.model_log_likelihood = Some(ll);
    diag.deviance = Some(2.0 * (diag.saturated_log_likelihood - ll));
    if !check.normalized || !check.positive || !ll.is_finite() {
        return Err(with(
            numerical(
                "not_normalized",
                "the fitted likelihood is not a normalized positive law",
                RefusalFields {
                    stage: Some("nested_markov.fit".to_owned()),
                    reason: Some("normalization or positivity check failed".to_owned()),
                    ..RefusalFields::default()
                },
            ),
            &diag,
        ));
    }
    let (margin, count) = boundary_stats(&parameters, &cells);
    diag.boundary_margin = Some(margin);
    diag.boundary_count = Some(count);
    if count > 0 {
        return Err(with(
            numerical(
                "fitted_boundary",
                "a fitted probability is within 1e-9 of 0 or 1",
                RefusalFields {
                    stage: Some("nested_markov.fit".to_owned()),
                    reason: Some("boundary fit".to_owned()),
                    boundary_margin: Some(ExactF64(margin)),
                    boundary_count: u64::try_from(count).ok(),
                    ..RefusalFields::default()
                },
            ),
            &diag,
        ));
    }
    Ok(NestedMarkovFit { parameters, cells, log_likelihood: ll, diagnostics: diag })
}

/// Model-implied means `E[X4 | do(X2 = x2)]` for `x2 = 0, 1`.
#[must_use]
pub fn model_means(params: &NestedParameters) -> [f64; 2] {
    let mean = |x2: usize| -> f64 {
        (0..2).map(|x3| bern(params.c[x2], x3) * (1.0 - params.q4[x3])).sum()
    };
    [mean(0), mean(1)]
}

/// Empirical plug-in means of the same identifying formula,
/// `sum_{x3} P^(x3|x2) sum_{x1,x2'} P^(x1,x2') P^(x4=1|x1,x2',x3)`.
///
/// # Errors
/// `transport_support_failure` / `nested_markov.positivity` for a zero cell.
pub fn plugin_means(counts: &BinaryCells) -> Result<[f64; 2], EstimationError> {
    let m = LawMarginals::new(&counts.empirical_law())?;
    let p2 = |x2: usize| m.p12[0][x2] + m.p12[1][x2];
    let p23 = |x2: usize, x3: usize| m.p123[0][x2][x3] + m.p123[1][x2][x3];
    let q = |x3: usize| -> f64 {
        let mut s = 0.0;
        for x1 in 0..2 {
            for x2 in 0..2 {
                s += m.p12[x1][x2] * m.p4(1, x1, x2, x3);
            }
        }
        s
    };
    let mean = |x2: usize| -> f64 { (0..2).map(|x3| p23(x2, x3) / p2(x2) * q(x3)).sum() };
    Ok([mean(0), mean(1)])
}

/// The target contrast through the fitted model and the empirical plug-in.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContrastComparison {
    /// Model means of `X4` under `do(X2 = 0)` and `do(X2 = 1)`.
    pub model_means: [f64; 2],
    /// Plug-in means.
    pub plugin_means: [f64; 2],
    /// Model contrast `mean(1) - mean(0)`.
    pub model_contrast: f64,
    /// Plug-in contrast.
    pub plugin_contrast: f64,
    /// `model_contrast - plugin_contrast`.
    pub difference: f64,
}

/// Compare the fitted-model contrast with the empirical plug-in on the same data.
///
/// # Errors
/// As [`plugin_means`].
pub fn compare_target_contrast(
    counts: &BinaryCells,
    fit: &NestedMarkovFit,
) -> Result<ContrastComparison, EstimationError> {
    let model = model_means(&fit.parameters);
    let plugin = plugin_means(counts)?;
    let model_contrast = model[1] - model[0];
    let plugin_contrast = plugin[1] - plugin[0];
    Ok(ContrastComparison {
        model_means: model,
        plugin_means: plugin,
        model_contrast,
        plugin_contrast,
        difference: model_contrast - plugin_contrast,
    })
}

/// Nonparametric identification standing (a property of the graph and formula).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IdentificationStanding {
    /// The functional is identified from the observational law by the formula
    /// above, with no parametric assumption.
    NonparametricallyIdentified,
}

/// Likelihood-fit standing (a property of this data and fit).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LikelihoodFitStanding {
    /// Converged and the data satisfy the equality constraints.
    ConvergedConstraintsSatisfied,
    /// Converged, but the data violate the constraints (fit is a projection).
    ConvergedConstraintsViolatedByData,
}

/// Inferential standing.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InferenceStanding {
    /// Calibration unmeasured: no interval or posterior is published.
    IntervalWithheldCalibrationUnmeasured,
}

/// The three statuses, kept as separate fields.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PilotStatus {
    /// Identification standing.
    pub identification: IdentificationStanding,
    /// Fit standing.
    pub likelihood_fit: LikelihoodFitStanding,
    /// Inferential standing.
    pub inference: InferenceStanding,
}

/// A pilot result: fit, both contrasts and the separate statuses.
#[derive(Clone, Debug, PartialEq)]
pub struct PilotReport {
    /// The fitted model.
    pub fit: NestedMarkovFit,
    /// Model versus plug-in contrast.
    pub comparison: ContrastComparison,
    /// Separate statuses.
    pub status: PilotStatus,
}

/// Bind the pilot estimand to the existing general-ID checker rather than
/// inferring identification standing from the selected likelihood model.
fn checked_pilot_identification(input: &NestedMarkovInput) -> Result<(), EstimationError> {
    use antecedent_core::{AverageEffectQuery, IdentificationStatus, VariableId};
    use antecedent_graph::{Admg, DenseNodeId};
    use antecedent_identify::{IdIdentifier, IdentificationWorkspace};
    let unavailable = || {
        EstimationError::refused(
            antecedent_core::reason_code!("effect_not_identified"),
            "nested_markov.identification_unavailable: the checked general-ID engine did not identify the selected pilot contrast",
        )
    };
    let (_, directed, bidirected) = input.graph.canonical();
    let mut graph = Admg::with_variables(4);
    let node =
        |raw: usize| u32::try_from(raw).map(DenseNodeId::from_raw).map_err(|_| unavailable());
    for &(from, to) in &directed {
        graph.insert_directed(node(from)?, node(to)?).map_err(|_| unavailable())?;
    }
    for &(left, right) in &bidirected {
        graph.insert_bidirected(node(left)?, node(right)?).map_err(|_| unavailable())?;
    }
    let identifier = IdIdentifier::new();
    let prepared = identifier.prepare(&graph).map_err(|_| unavailable())?;
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(1), VariableId::from_raw(3));
    let identified = identifier
        .identify_ate(&prepared, &query, &mut IdentificationWorkspace::default())
        .map_err(|_| unavailable())?;
    if identified.status != IdentificationStatus::NonparametricallyIdentified
        || identified.estimands.is_empty()
    {
        return Err(unavailable());
    }
    Ok(())
}

/// Scope check, fit and comparison in one call.
///
/// # Errors
/// Any refusal of [`binary_cells`], [`fit_nested_markov`] or [`plugin_means`].
pub fn evaluate_nested_markov_pilot(
    input: &NestedMarkovInput,
    options: &FitOptions,
    ctx: &ExecutionContext,
) -> Result<PilotReport, Refusal> {
    let counts = binary_cells(input).map_err(plain)?;
    checked_pilot_identification(input).map_err(plain)?;
    let fit = fit_nested_markov(&counts, options, ctx)?;
    let comparison = compare_target_contrast(&counts, &fit).map_err(plain)?;
    let likelihood_fit = match fit.diagnostics.constraint_status {
        ConstraintStatus::Satisfied => LikelihoodFitStanding::ConvergedConstraintsSatisfied,
        ConstraintStatus::ViolatedByData => {
            LikelihoodFitStanding::ConvergedConstraintsViolatedByData
        }
    };
    Ok(PilotReport {
        fit,
        comparison,
        status: PilotStatus {
            identification: IdentificationStanding::NonparametricallyIdentified,
            likelihood_fit,
            inference: InferenceStanding::IntervalWithheldCalibrationUnmeasured,
        },
    })
}

#[cfg(test)]
mod checked_identification_tests {
    #[test]
    fn checked_binding_refuses_an_unidentified_adjacent_bow() {
        let input = super::NestedMarkovInput {
            graph: super::AdmgDeclaration {
                variables: super::AdmgDeclaration::selected().variables,
                directed: vec![(1, 3)],
                bidirected: vec![(1, 3)],
            },
            regimes: Vec::new(),
        };
        let (error, checks) =
            antecedent_identify::execution_counts::count_checked_identifications(|| {
                super::checked_pilot_identification(&input).unwrap_err()
            });
        assert_eq!(checks, 1);
        match error {
            crate::EstimationError::Refused { code, message, .. } => {
                assert_eq!(code, "effect_not_identified");
                assert!(message.starts_with("nested_markov.identification_unavailable:"));
            }
            other => panic!("unexpected checked binding refusal: {other}"),
        }
    }
}
