// SPDX-License-Identifier: MIT OR Apache-2.0
//! Sparse Jacobian storage, sparse LU and sparse diagnostics for large feedback systems.
//!
//! A feedback block of `k` coupled variables with `m` interactions is stored as a
//! compressed-sparse-row matrix, not a dense `k x k` array. Newton steps are solved by a
//! sparse direct LU with partial pivoting and a minimum-degree column order,
//! rather than by an iterative method. A direct factorization is chosen because it is
//! deterministic and replayable, it detects a numerically singular Jacobian by a pivot floor
//! (an iterative solver can only stall), and its two factors (of `A` and `A^T`) give the
//! Hager-Higham inverse-norm estimate that reports conditioning. Fill-in is counted and
//! bounded, so a block whose factor would exceed its budget is a typed refusal, never an
//! unbounded allocation.
//!
//! Existence and uniqueness diagnostics are bounds on the derivative-bound matrix `B`
//! (`B_ij >= sup |df_i/dx_j|`): its infinity norm (a Gershgorin row-sum bound) and a rigorous
//! Collatz-Wielandt upper bound on its spectral radius, found by power iteration on `B + I`.
//! Either bound below one proves a contraction in a (weighted) max norm. A Jacobian at a
//! numerical root that is strictly diagonally dominant is nonsingular (Levy-Desplanques).

use crate::nonlinear_equilibrium::EquilibriumError;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// Distinct typed reasons a sparse solve is refused for resources, not for mathematics.
///
/// Each code is carried as the stable prefix of an [`EquilibriumError::Resource`] message,
/// so existing callers keep their refusal channel; [`EquilibriumError::budget_code`] recovers it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SolverBudgetCode {
    /// The model order exceeds the dense disturbance-covariance admission.
    ModelOrder,
    /// The free Jacobian has more structural nonzeros than the policy admits.
    JacobianNnz,
    /// A strongly connected block is larger than the policy admits.
    BlockSize,
    /// The estimated solver memory exceeds the policy admission.
    Memory,
    /// Sparse LU fill-in exceeded the policy bound.
    FactorFill,
    /// The total iteration count across blocks exceeded the policy bound.
    IterationBudget,
    /// The wall-clock deadline of the policy elapsed.
    WallClock,
}
impl SolverBudgetCode {
    /// Stable identifier used as the message prefix and in serialized evidence.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ModelOrder => antecedent_core::reason_code!("solver_model_order_budget"),
            Self::JacobianNnz => antecedent_core::reason_code!("solver_jacobian_nnz_budget"),
            Self::BlockSize => antecedent_core::reason_code!("solver_block_size_budget"),
            Self::Memory => antecedent_core::reason_code!("solver_memory_budget"),
            Self::FactorFill => antecedent_core::reason_code!("solver_factor_fill_budget"),
            Self::IterationBudget => antecedent_core::reason_code!("solver_iteration_budget"),
            Self::WallClock => antecedent_core::reason_code!("solver_wall_clock_budget"),
        }
    }
    const ALL: [Self; 7] = [
        Self::ModelOrder,
        Self::JacobianNnz,
        Self::BlockSize,
        Self::Memory,
        Self::FactorFill,
        Self::IterationBudget,
        Self::WallClock,
    ];
    /// A resource refusal that names this code, the amount required and the bound.
    #[must_use]
    pub fn refusal(self, required: u64, limit: u64) -> EquilibriumError {
        EquilibriumError::Resource(format!(
            "{}: required {required} exceeds admitted {limit}",
            self.as_str()
        ))
    }
}
impl EquilibriumError {
    /// The solver budget code of a resource refusal, if it is one of them.
    #[must_use]
    pub fn budget_code(&self) -> Option<SolverBudgetCode> {
        let Self::Resource(message) = self else {
            return None;
        };
        SolverBudgetCode::ALL.into_iter().find(|code| {
            message.strip_prefix(code.as_str()).is_some_and(|rest| rest.starts_with(':'))
        })
    }
}

/// Compressed-sparse-row square matrix with strictly increasing columns in each row.
#[derive(Clone, Debug, PartialEq)]
pub struct CsrMatrix {
    n: usize,
    row_ptr: Vec<usize>,
    col_idx: Vec<usize>,
    values: Vec<f64>,
}
impl CsrMatrix {
    /// Build from per-row `(column, value)` entries. Duplicates are summed; exact zeros kept
    /// only if no entry survives to carry structure, so callers add the diagonal explicitly.
    pub fn from_rows(n: usize, mut rows: Vec<Vec<(usize, f64)>>) -> Result<Self, EquilibriumError> {
        if rows.len() != n {
            return Err(EquilibriumError::Invalid("sparse matrix row count".into()));
        }
        let mut row_ptr = Vec::with_capacity(n + 1);
        let mut col_idx = Vec::new();
        let mut values = Vec::new();
        row_ptr.push(0);
        for row in &mut rows {
            row.sort_by_key(|entry| entry.0);
            let mut last: Option<usize> = None;
            for &(col, value) in row.iter() {
                if col >= n {
                    return Err(EquilibriumError::Invalid("sparse matrix column".into()));
                }
                if last == Some(col) {
                    if let Some(slot) = values.last_mut() {
                        *slot += value;
                    }
                } else {
                    col_idx.push(col);
                    values.push(value);
                    last = Some(col);
                }
            }
            row_ptr.push(col_idx.len());
        }
        Ok(Self { n, row_ptr, col_idx, values })
    }
    /// Matrix order.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.n
    }
    /// Stored entries.
    #[must_use]
    pub fn nnz(&self) -> usize {
        self.values.len()
    }
    /// Bytes held by the three CSR arrays.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        ((self.row_ptr.len() + self.col_idx.len() + self.values.len()) * 8) as u64
    }
    /// Columns and values of row `i`.
    #[must_use]
    pub fn row(&self, i: usize) -> (&[usize], &[f64]) {
        let (lo, hi) = (self.row_ptr[i], self.row_ptr[i + 1]);
        (&self.col_idx[lo..hi], &self.values[lo..hi])
    }
    fn row_pairs(&self, i: usize) -> Vec<(usize, f64)> {
        let (cols, values) = self.row(i);
        cols.iter().copied().zip(values.iter().copied()).collect()
    }
    /// `A x`.
    #[must_use]
    pub fn matvec(&self, x: &[f64]) -> Vec<f64> {
        (0..self.n)
            .map(|i| {
                let (cols, values) = self.row(i);
                cols.iter().zip(values).map(|(&c, &v)| v * x[c]).sum()
            })
            .collect()
    }
    /// The transpose, also in CSR form.
    #[must_use]
    pub fn transpose(&self) -> Self {
        let mut rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); self.n];
        for i in 0..self.n {
            let (cols, values) = self.row(i);
            for (&c, &v) in cols.iter().zip(values) {
                rows[c].push((i, v));
            }
        }
        // Rows were produced in increasing `i`, so each is already sorted and duplicate free.
        let mut row_ptr = Vec::with_capacity(self.n + 1);
        let mut col_idx = Vec::with_capacity(self.nnz());
        let mut values = Vec::with_capacity(self.nnz());
        row_ptr.push(0);
        for row in rows {
            for (c, v) in row {
                col_idx.push(c);
                values.push(v);
            }
            row_ptr.push(col_idx.len());
        }
        Self { n: self.n, row_ptr, col_idx, values }
    }
    /// Largest absolute row sum.
    #[must_use]
    pub fn infinity_norm(&self) -> f64 {
        (0..self.n).map(|i| self.row(i).1.iter().map(|v| v.abs()).sum::<f64>()).fold(0.0, f64::max)
    }
    /// True when every stored value is finite.
    #[must_use]
    pub fn is_finite(&self) -> bool {
        self.values.iter().all(|v| v.is_finite())
    }
    /// `min_i (|a_ii| - sum_{j != i} |a_ij|)`: positive means strictly diagonally dominant, so
    /// the matrix is nonsingular (Levy-Desplanques). `None` for an empty matrix.
    #[must_use]
    pub fn diagonal_dominance_margin(&self) -> Option<f64> {
        (0..self.n)
            .map(|i| {
                let (cols, values) = self.row(i);
                let mut diagonal = 0.0;
                let mut off = 0.0;
                for (&c, &v) in cols.iter().zip(values) {
                    if c == i {
                        diagonal = v.abs();
                    } else {
                        off += v.abs();
                    }
                }
                diagonal - off
            })
            .reduce(f64::min)
    }
}

/// Why a factorization stopped.
#[derive(Debug)]
pub enum FactorError {
    /// A pivot column had no entry above the numerical floor.
    Singular,
    /// A stored or computed value was not finite.
    NonFinite,
    /// Fill-in exceeded the bound; carries the entries that would have been held.
    FillBudget(usize),
    /// Cancellation or a deadline surfaced while factoring.
    Interrupted(EquilibriumError),
}

struct Step {
    pivot_row: usize,
    pivot_col: usize,
    row: Vec<(usize, f64)>,
    multipliers: Vec<(usize, f64)>,
}

/// Sparse LU of a square matrix as a recorded sequence of elimination steps.
pub struct SparseLu {
    n: usize,
    steps: Vec<Step>,
    fill: usize,
}

/// Greedy minimum-degree column order on the symmetrized pattern. The clique-update work is
/// bounded; once it is exhausted the remaining columns keep their natural order.
fn minimum_degree_order(
    rows: &[Vec<(usize, f64)>],
    n: usize,
    work_limit: usize,
    tick: &mut dyn FnMut() -> Result<(), EquilibriumError>,
) -> Result<Vec<usize>, EquilibriumError> {
    let mut adjacency: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); n];
    for (r, row) in rows.iter().enumerate() {
        if r % 32 == 0 {
            tick()?;
        }
        for &(c, _) in row {
            if c != r {
                adjacency[r].insert(c);
                adjacency[c].insert(r);
            }
        }
    }
    let mut done = vec![false; n];
    let mut order = Vec::with_capacity(n);
    let mut work = 0_usize;
    while order.len() < n {
        tick()?;
        if work > work_limit {
            order.extend((0..n).filter(|&v| !done[v]));
            break;
        }
        let Some(v) = (0..n).filter(|&v| !done[v]).min_by_key(|&v| adjacency[v].len()) else {
            break;
        };
        done[v] = true;
        order.push(v);
        let neighbours: Vec<usize> = adjacency[v].iter().copied().collect();
        for &u in &neighbours {
            adjacency[u].remove(&v);
        }
        work += neighbours.len() * neighbours.len();
        if work <= work_limit {
            for (a, &u) in neighbours.iter().enumerate() {
                if a % 32 == 0 {
                    tick()?;
                }
                for &w in &neighbours[a + 1..] {
                    adjacency[u].insert(w);
                    adjacency[w].insert(u);
                }
            }
        }
        adjacency[v].clear();
    }
    Ok(order)
}

/// `target - m * pivot` over sorted sparse rows, dropping column `skip`. Columns that the
/// pivot introduces are appended to `introduced`.
fn eliminate_row(
    target: &[(usize, f64)],
    pivot: &[(usize, f64)],
    m: f64,
    skip: usize,
    introduced: &mut Vec<usize>,
) -> Vec<(usize, f64)> {
    let mut out = Vec::with_capacity(target.len() + pivot.len());
    let (mut a, mut b) = (0_usize, 0_usize);
    loop {
        let ca = target.get(a).map_or(usize::MAX, |e| e.0);
        let cb = pivot.get(b).map_or(usize::MAX, |e| e.0);
        if ca == usize::MAX && cb == usize::MAX {
            break;
        }
        if ca == cb {
            if ca != skip {
                let value = target[a].1 - m * pivot[b].1;
                if value != 0.0 {
                    out.push((ca, value));
                }
            }
            a += 1;
            b += 1;
        } else if ca < cb {
            if ca != skip {
                out.push(target[a]);
            }
            a += 1;
        } else {
            if cb != skip {
                out.push((cb, -m * pivot[b].1));
                introduced.push(cb);
            }
            b += 1;
        }
    }
    out
}

impl SparseLu {
    /// Factor `matrix`. `max_fill` bounds the entries held by the factors plus the active
    /// rows. `tick` is called regularly and may return a cancellation or deadline refusal.
    pub fn factor(
        matrix: &CsrMatrix,
        max_fill: usize,
        tick: &mut dyn FnMut() -> Result<(), EquilibriumError>,
    ) -> Result<Self, FactorError> {
        let n = matrix.dim();
        let norm = matrix.infinity_norm();
        if !norm.is_finite() || !matrix.is_finite() {
            return Err(FactorError::NonFinite);
        }
        let mut rows: Vec<Vec<(usize, f64)>> = (0..n).map(|i| matrix.row_pairs(i)).collect();
        let mut live: usize = rows.iter().map(Vec::len).sum();
        if live > max_fill {
            return Err(FactorError::FillBudget(live));
        }
        let mut column_rows: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (r, row) in rows.iter().enumerate() {
            for &(c, _) in row {
                column_rows[c].push(r);
            }
        }
        let order = minimum_degree_order(&rows, n, 64 * matrix.nnz() + 4_000_000, tick)
            .map_err(FactorError::Interrupted)?;
        let mut active = vec![true; n];
        let mut stored = 0_usize;
        let floor = (norm * 1e-13).max(f64::MIN_POSITIVE);
        let mut steps = Vec::with_capacity(n);
        for (k, &c) in order.iter().enumerate() {
            if k % 16 == 0 {
                tick().map_err(FactorError::Interrupted)?;
            }
            let mut candidates_rows = std::mem::take(&mut column_rows[c]);
            candidates_rows.sort_unstable();
            candidates_rows.dedup();
            let mut candidates: Vec<(usize, f64)> = Vec::new();
            for r in candidates_rows {
                if active[r] {
                    if let Ok(p) = rows[r].binary_search_by_key(&c, |e| e.0) {
                        candidates.push((r, rows[r][p].1));
                    }
                }
            }
            let largest = candidates.iter().map(|e| e.1.abs()).fold(0.0_f64, f64::max);
            if largest.is_nan() || largest <= floor {
                return Err(FactorError::Singular);
            }
            let Some(&(pivot_index, pivot_value)) = candidates
                .iter()
                .filter(|e| e.1.abs() >= (1.0 - 1e-9) * largest)
                .min_by_key(|e| rows[e.0].len())
            else {
                return Err(FactorError::Singular);
            };
            let pivot_row = std::mem::take(&mut rows[pivot_index]);
            active[pivot_index] = false;
            live -= pivot_row.len();
            let mut multipliers = Vec::new();
            for &(r, value) in &candidates {
                if r == pivot_index {
                    continue;
                }
                let m = value / pivot_value;
                let mut introduced = Vec::new();
                let before = rows[r].len();
                let updated = eliminate_row(&rows[r], &pivot_row, m, c, &mut introduced);
                live = live + updated.len() - before;
                rows[r] = updated;
                for column in introduced {
                    column_rows[column].push(r);
                }
                multipliers.push((r, m));
            }
            stored += pivot_row.len() + multipliers.len();
            if stored + live > max_fill {
                return Err(FactorError::FillBudget(stored + live));
            }
            if !pivot_value.is_finite() || pivot_row.iter().any(|e| !e.1.is_finite()) {
                return Err(FactorError::NonFinite);
            }
            steps.push(Step { pivot_row: pivot_index, pivot_col: c, row: pivot_row, multipliers });
        }
        Ok(Self { n, steps, fill: stored })
    }
    /// Entries held by the factors (U rows plus L multipliers).
    #[must_use]
    pub fn fill(&self) -> usize {
        self.fill
    }
    /// Solve `A x = b`.
    #[must_use]
    pub fn solve(&self, b: &[f64]) -> Vec<f64> {
        let mut y = b.to_vec();
        for step in &self.steps {
            let top = y[step.pivot_row];
            if top != 0.0 {
                for &(r, m) in &step.multipliers {
                    y[r] -= m * top;
                }
            }
        }
        let mut x = vec![0.0; self.n];
        for step in self.steps.iter().rev() {
            let mut sum = y[step.pivot_row];
            let mut diagonal = 1.0;
            for &(c, v) in &step.row {
                if c == step.pivot_col {
                    diagonal = v;
                } else {
                    sum -= v * x[c];
                }
            }
            x[step.pivot_col] = sum / diagonal;
        }
        x
    }
    /// `ln |det A|`, the sum of log absolute pivots.
    #[must_use]
    pub fn log_abs_determinant(&self) -> f64 {
        self.steps
            .iter()
            .map(|step| {
                step.row
                    .iter()
                    .find(|e| e.0 == step.pivot_col)
                    .map_or(f64::NEG_INFINITY, |e| e.1.abs().ln())
            })
            .sum()
    }
}

/// Largest block order whose inverse infinity norm is formed exactly, column by column.
const EXACT_INVERSE_LIMIT: usize = 32;

/// Infinity-norm condition number `||A||_inf ||A^-1||_inf` of a factored matrix. `lu` factors
/// `A` and `lu_transpose` factors `A^T`. Exact for small orders; otherwise the Hager-Higham
/// 1-norm estimator is applied to `A^-T`, a lower bound on the true condition number that is
/// almost always within a small factor of it. `None` when the estimate is not finite.
#[must_use]
pub fn infinity_condition_estimate(
    matrix: &CsrMatrix,
    lu: &SparseLu,
    lu_transpose: &SparseLu,
) -> Option<f64> {
    let n = matrix.dim();
    if n == 0 {
        return None;
    }
    let inverse_norm = if n <= EXACT_INVERSE_LIMIT {
        let mut row_sums = vec![0.0; n];
        for j in 0..n {
            let mut unit = vec![0.0; n];
            unit[j] = 1.0;
            for (sum, value) in row_sums.iter_mut().zip(lu.solve(&unit)) {
                *sum += value.abs();
            }
        }
        row_sums.into_iter().fold(0.0, f64::max)
    } else {
        // ||A^-1||_inf = ||A^-T||_1; Hager's iteration uses B = A^-T and B^T = A^-1.
        let mut x = vec![1.0 / n as f64; n];
        let mut estimate = 0.0_f64;
        let mut last = usize::MAX;
        for iteration in 0..6 {
            let y = lu_transpose.solve(&x);
            let candidate: f64 = y.iter().map(|v| v.abs()).sum();
            if iteration > 0 && candidate <= estimate {
                break;
            }
            estimate = estimate.max(candidate);
            let signs: Vec<f64> = y.iter().map(|v| if *v >= 0.0 { 1.0 } else { -1.0 }).collect();
            let z = lu.solve(&signs);
            let (j, zmax) = z
                .iter()
                .enumerate()
                .map(|(j, v)| (j, v.abs()))
                .fold((0, 0.0), |a, b| if b.1 > a.1 { b } else { a });
            let zx: f64 = z.iter().zip(&x).map(|(a, b)| a * b).sum();
            if zmax <= zx || j == last {
                break;
            }
            last = j;
            x = vec![0.0; n];
            x[j] = 1.0;
        }
        let alternating: Vec<f64> = (0..n)
            .map(|i| {
                let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
                sign * (1.0 + i as f64 / (n - 1) as f64)
            })
            .collect();
        let alt: f64 = lu_transpose.solve(&alternating).iter().map(|v| v.abs()).sum();
        estimate.max(2.0 * alt / (3.0 * n as f64))
    };
    let kappa = matrix.infinity_norm() * inverse_norm;
    kappa.is_finite().then_some(kappa)
}

/// A rigorous upper bound on the spectral radius of a nonnegative matrix.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpectralBound {
    /// Collatz-Wielandt bound `max_i (B w)_i / w_i` for a strictly positive `w`, inflated for
    /// floating-point rounding.
    pub upper: f64,
    /// Power iterations performed.
    pub iterations: usize,
}

/// Power iteration on `B + I` (which is primitive on a strongly connected block) supplies a
/// positive vector `w`; for any positive `w`, `max_i (B w)_i / w_i >= rho(B)`, so the returned
/// bound is rigorous regardless of how far the iteration converged.
#[must_use]
pub fn nonnegative_spectral_upper_bound(b: &CsrMatrix, max_iterations: usize) -> SpectralBound {
    let n = b.dim();
    if n == 0 {
        return SpectralBound { upper: 0.0, iterations: 0 };
    }
    let mut w = vec![1.0; n];
    let mut iterations = 0;
    let mut previous = f64::INFINITY;
    for _ in 0..max_iterations {
        let mut v = b.matvec(&w);
        for (vi, wi) in v.iter_mut().zip(&w) {
            *vi += wi;
        }
        let top = v.iter().copied().fold(0.0_f64, f64::max);
        if !top.is_finite() || top <= 0.0 {
            return SpectralBound { upper: f64::INFINITY, iterations };
        }
        for vi in &mut v {
            *vi = (*vi / top).max(1e-12);
        }
        w = v;
        iterations += 1;
        if (top - previous).abs() <= 1e-14 * top {
            break;
        }
        previous = top;
    }
    let bw = b.matvec(&w);
    let ratio = bw.iter().zip(&w).map(|(a, c)| a / c).fold(0.0_f64, f64::max);
    let widest = (0..n).map(|i| b.row(i).0.len()).max().unwrap_or(0);
    let upper = ratio * (1.0 + (widest + 2) as f64 * f64::EPSILON);
    SpectralBound { upper: if upper.is_finite() { upper } else { f64::INFINITY }, iterations }
}

/// What the sparse solver established and spent. Every field is deterministic for a given
/// model, regime and policy, so a replay reproduces it; wall-clock time is not retained.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SolveEvidence {
    /// Free (unclamped) variables solved.
    pub free_variables: usize,
    /// Strongly connected blocks, singletons included.
    pub block_count: usize,
    /// Blocks that are genuinely coupled (more than one variable, or self-feedback).
    pub cyclic_block_count: usize,
    /// Largest block order.
    pub largest_block: usize,
    /// Structural nonzeros of the free Jacobian `I - df/dx`.
    pub jacobian_nnz: usize,
    /// Contraction iterations, across blocks.
    pub contraction_iterations: usize,
    /// Newton steps, across blocks.
    pub newton_steps: usize,
    /// Sparse factorizations performed (Newton steps plus the final conditioning factors).
    pub factorizations: usize,
    /// Largest factor fill (entries) of any factorization.
    pub peak_factor_fill: usize,
    /// Wall-clock and cancellation checks performed.
    pub budget_checks: usize,
    /// Bytes charged against the execution admission pool.
    pub admitted_bytes: u64,
    /// Estimated peak solver memory the policy was checked against.
    pub estimated_bytes: u64,
    /// Largest per-block Gershgorin row-sum bound on the derivative-bound matrix.
    pub row_sum_bound: f64,
    /// Largest per-block rigorous spectral-radius upper bound of that matrix (the row-sum bound
    /// where it was already below one and no power iteration was needed).
    pub spectral_bound: f64,
    /// Power iterations spent on spectral bounds.
    pub spectral_iterations: usize,
    /// Largest per-block contraction bound used (the smaller of the two bounds, per block).
    pub contraction_bound: f64,
    /// `min_i (|J_ii| - sum |J_ij|)` of the free Jacobian at the returned state; positive proves
    /// it nonsingular. `None` when there is nothing to solve.
    pub jacobian_diagonal_dominance_margin: Option<f64>,
    /// How nonsingularity of the free Jacobian is established: `contraction`,
    /// `strict_diagonal_dominance`, `factorization` (numerical only), `singular`,
    /// `factor_fill_budget` (not attempted) or `not_applicable`.
    pub jacobian_nonsingular_basis: String,
    /// How a unique root is established: `contraction`, `affine_bijection` or `none`.
    pub unique_root_basis: String,
    /// True when the condition estimate exceeds the policy warning threshold.
    pub ill_conditioned: bool,
}

impl SolveEvidence {
    /// True for the default value of a solver that records no sparse evidence.
    #[must_use]
    pub fn is_unrecorded(&self) -> bool {
        *self == Self::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn never() -> impl FnMut() -> Result<(), EquilibriumError> {
        || Ok(())
    }
    fn dense_rows(a: &[Vec<f64>]) -> Vec<Vec<(usize, f64)>> {
        a.iter()
            .map(|row| {
                row.iter().enumerate().filter(|(_, v)| **v != 0.0).map(|(j, v)| (j, *v)).collect()
            })
            .collect()
    }
    fn dense_solve(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
        let n = b.len();
        let mut m: Vec<Vec<f64>> = a
            .iter()
            .zip(b)
            .map(|(row, rhs)| {
                let mut r = row.clone();
                r.push(*rhs);
                r
            })
            .collect();
        for j in 0..n {
            let p = (j..n).max_by(|&x, &y| m[x][j].abs().total_cmp(&m[y][j].abs())).unwrap();
            m.swap(j, p);
            for i in 0..n {
                if i != j {
                    let f = m[i][j] / m[j][j];
                    for k in j..=n {
                        let t = m[j][k];
                        m[i][k] -= f * t;
                    }
                }
            }
        }
        (0..n).map(|i| m[i][n] / m[i][i]).collect()
    }
    /// Deterministic banded matrix with a pivoting-hostile zero diagonal stretch.
    fn banded(n: usize) -> Vec<Vec<f64>> {
        let mut a = vec![vec![0.0; n]; n];
        for i in 0..n {
            a[i][i] = if i % 7 == 3 { 0.0 } else { 2.0 + (i % 5) as f64 * 0.1 };
            if i + 1 < n {
                a[i][i + 1] = -0.6 - (i % 3) as f64 * 0.1;
            }
            if i > 0 {
                a[i][i - 1] = 0.9;
            }
            if i + 2 < n {
                a[i][i + 2] = 0.3;
            }
        }
        a
    }

    #[test]
    fn sparse_lu_matches_a_dense_solve_on_a_banded_matrix_with_zero_diagonals() {
        let n = 60;
        let a = banded(n);
        let csr = CsrMatrix::from_rows(n, dense_rows(&a)).unwrap();
        let lu = SparseLu::factor(&csr, 1_000_000, &mut never()).unwrap();
        let b: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37).sin()).collect();
        let sparse = lu.solve(&b);
        let dense = dense_solve(&a, &b);
        for (s, d) in sparse.iter().zip(&dense) {
            assert!((s - d).abs() < 1e-9 * (1.0 + d.abs()), "{s} vs {d}");
        }
        let residual = csr.matvec(&sparse);
        assert!(residual.iter().zip(&b).all(|(r, c)| (r - c).abs() < 1e-9));
        assert!(lu.fill() < n * n / 4);
    }

    #[test]
    fn determinant_and_condition_agree_with_the_dense_values() {
        // Exact order: 3x3 with known inverse.
        let a = vec![vec![4.0, 1.0, 0.0], vec![1.0, 3.0, 1.0], vec![0.0, 1.0, 2.0]];
        let csr = CsrMatrix::from_rows(3, dense_rows(&a)).unwrap();
        let t = csr.transpose();
        let lu = SparseLu::factor(&csr, 100, &mut never()).unwrap();
        let lut = SparseLu::factor(&t, 100, &mut never()).unwrap();
        let det = 4.0 * (3.0 * 2.0 - 1.0) - 1.0 * (1.0 * 2.0 - 0.0);
        assert!((lu.log_abs_determinant() - f64::ln(det)).abs() < 1e-12);
        let mut inverse_rows = vec![0.0; 3];
        for j in 0..3 {
            let mut e = vec![0.0; 3];
            e[j] = 1.0;
            for (s, v) in inverse_rows.iter_mut().zip(lu.solve(&e)) {
                *s += v.abs();
            }
        }
        let exact = 5.0 * inverse_rows.iter().copied().fold(0.0, f64::max);
        let got = infinity_condition_estimate(&csr, &lu, &lut).unwrap();
        assert!((got - exact).abs() < 1e-12 * exact);
        // Larger order uses the Hager-Higham estimator: a lower bound within a small factor.
        let n = 80;
        let a = banded(n);
        let csr = CsrMatrix::from_rows(n, dense_rows(&a)).unwrap();
        let lu = SparseLu::factor(&csr, 1_000_000, &mut never()).unwrap();
        let lut = SparseLu::factor(&csr.transpose(), 1_000_000, &mut never()).unwrap();
        let mut sums = vec![0.0; n];
        for j in 0..n {
            let mut e = vec![0.0; n];
            e[j] = 1.0;
            for (s, v) in sums.iter_mut().zip(lu.solve(&e)) {
                *s += v.abs();
            }
        }
        let exact = csr.infinity_norm() * sums.into_iter().fold(0.0, f64::max);
        let estimate = infinity_condition_estimate(&csr, &lu, &lut).unwrap();
        assert!(estimate <= exact * (1.0 + 1e-9) && estimate >= exact / 4.0, "{estimate} {exact}");
    }

    #[test]
    fn a_singular_matrix_and_a_fill_overflow_are_typed_failures() {
        let singular =
            CsrMatrix::from_rows(2, vec![vec![(0, 1.0), (1, 1.0)], vec![(0, 1.0), (1, 1.0)]])
                .unwrap();
        assert!(matches!(
            SparseLu::factor(&singular, 100, &mut never()),
            Err(FactorError::Singular)
        ));
        let dense_rows_full: Vec<Vec<(usize, f64)>> = (0..20)
            .map(|i| (0..20).map(|j| (j, if i == j { 25.0 } else { 1.0 })).collect())
            .collect();
        let full = CsrMatrix::from_rows(20, dense_rows_full).unwrap();
        assert!(matches!(
            SparseLu::factor(&full, 50, &mut never()),
            Err(FactorError::FillBudget(_))
        ));
        let mut stop = || Err(EquilibriumError::Resource("cancelled".into()));
        assert!(matches!(
            SparseLu::factor(&full, 1_000_000, &mut stop),
            Err(FactorError::Interrupted(_))
        ));
    }

    #[test]
    fn spectral_bound_is_rigorous_and_tighter_than_the_row_sum() {
        // Cycle with weights 1.2 and 0.5: rho = sqrt(0.6) < 1 although the row sum is 1.2.
        let b = CsrMatrix::from_rows(2, vec![vec![(1, 1.2)], vec![(0, 0.5)]]).unwrap();
        let bound = nonnegative_spectral_upper_bound(&b, 200);
        let rho = 0.6_f64.sqrt();
        assert!(bound.upper >= rho * (1.0 - 1e-12));
        assert!(bound.upper < 0.7746 + 1e-6, "{}", bound.upper);
        assert!(b.infinity_norm() > 1.0);
    }

    #[test]
    fn budget_codes_round_trip_through_resource_refusals() {
        let e = SolverBudgetCode::FactorFill.refusal(10, 5);
        assert_eq!(e.budget_code(), Some(SolverBudgetCode::FactorFill));
        assert_eq!(EquilibriumError::Resource("cancelled".into()).budget_code(), None);
        assert_eq!(EquilibriumError::Singular.budget_code(), None);
    }
}
