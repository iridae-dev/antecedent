//! The deliberately independent closed-form recomputation of a cross-world
//! point.
//!
//! This file shares no code with the fitting or evaluation path (`gcm`,
//! `antecedent-counterfactual`): the spline basis, the least-squares fits and
//! the node-by-node evaluation are re-derived here on purpose, so that agreement
//! with the replayed point is evidence rather than tautology. Do not merge it
//! with the estimate code.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{CrossWorldQuery, VariableId};
use antecedent_stats::{DenseLinearAlgebra, FaerBackend, LeastSquaresWorkspace};

use crate::gcm::NestedOutcomeMechanism;

/// Relative tolerance of the closed-form cross-check against the replayed point.
const CLOSED_FORM_TOLERANCE: f64 = 1e-6;

/// Ordinary least squares of `columns[v]` on `columns[parents]` with an intercept,
/// by column-pivoted QR of the centered, column-equilibrated design
/// ([`FaerBackend`]): `(intercept, slopes)`, or `None` when the design is
/// numerically rank deficient.
pub(super) fn ols_on_parents(
    columns: &[Vec<f64>],
    v: usize,
    parents: &[usize],
) -> Option<(f64, Vec<f64>)> {
    let rows = columns[v].len();
    let p = parents.len();
    let mean = |c: &[f64]| c.iter().sum::<f64>() / rows as f64;
    let ybar = mean(&columns[v]);
    if p == 0 {
        return Some((ybar, Vec::new()));
    }
    let xbar: Vec<f64> = parents.iter().map(|&j| mean(&columns[j])).collect();
    // Column-major centered design and centered response.
    let mut design = Vec::with_capacity(rows * p);
    for (a, &j) in parents.iter().enumerate() {
        design.extend(columns[j].iter().map(|x| x - xbar[a]));
    }
    let centered: Vec<f64> = columns[v].iter().map(|y| y - ybar).collect();
    let fit = FaerBackend
        .least_squares(&design, rows, p, &centered, &mut LeastSquaresWorkspace::default())
        .ok()?;
    let beta = fit.coefficients;
    let intercept = ybar - beta.iter().zip(&xbar).map(|(b, x)| b * x).sum::<f64>();
    Some((intercept, beta))
}

// The non-separable family's estimator, restated from its specification and not
// from the fitter's code. The outcome is regressed on the expansion of its
// standardized parents: each parent's main effect, and for a parent with at
// least `SPLINE_MIN_DISTINCT` distinct standardized values its square, cube and
// one truncated cubic per interior knot (the 0.25 / 0.5 / 0.75 sample quantiles
// of the standardized column, nearest rank, duplicates removed); then, for every
// ordered pair of distinct parents (j, l), the main effect of j times every
// column of l, keeping the main-effect product z_j z_l once per pair. The
// least-squares problem is on columns scaled to unit root-mean-square, with a
// relative ridge on every column but the intercept: minimize
// |y - Xb|^2 + SPLINE_CONDITIONING_RIDGE * n * |b_columns|^2. The ridge is part of
// the estimator (the truncated power basis is nearly collinear), so the check
// solves the same penalized problem and differs only in numerics.
const SPLINE_KNOT_QUANTILES: [f64; 3] = [0.25, 0.50, 0.75];
const SPLINE_MIN_DISTINCT: usize = 5;
const SPLINE_CONDITIONING_RIDGE: f64 = 1e-6;
/// Rows required per fitted column; the fitter refuses below it, so no
/// comparison exists for such a table.
const SPLINE_MIN_ROWS_PER_COLUMN: usize = 10;

/// A fitted non-separable outcome mean: `intercept + coeffs . phi(z(parents))`.
struct SplineMean {
    intercept: f64,
    coeffs: Vec<f64>,
    centers: Vec<f64>,
    scales: Vec<f64>,
    knots: Vec<Vec<f64>>,
}

/// The expansion columns (no intercept) of one standardized parent row.
fn spline_columns(z: &[f64], knots: &[Vec<f64>], out: &mut Vec<f64>) {
    out.clear();
    let mut own: Vec<Vec<f64>> = Vec::with_capacity(z.len());
    for (p, &zp) in z.iter().enumerate() {
        let mut cols = vec![zp];
        if !knots[p].is_empty() {
            cols.push(zp * zp);
            cols.push(zp * zp * zp);
            cols.extend(knots[p].iter().map(|&t| if zp > t { (zp - t).powi(3) } else { 0.0 }));
        }
        own.push(cols);
    }
    for cols in &own {
        out.extend_from_slice(cols);
    }
    for (j, &zj) in z.iter().enumerate() {
        for (l, cols) in own.iter().enumerate() {
            if l == j {
                continue;
            }
            for (k, &c) in cols.iter().enumerate() {
                if k == 0 && l < j {
                    continue;
                }
                out.push(zj * c);
            }
        }
    }
}

impl SplineMean {
    fn mean(&self, raw: &[f64], z: &mut Vec<f64>, cols: &mut Vec<f64>) -> f64 {
        z.clear();
        z.extend(raw.iter().enumerate().map(|(p, x)| (x - self.centers[p]) / self.scales[p]));
        spline_columns(z, &self.knots, cols);
        self.intercept + self.coeffs.iter().zip(cols.iter()).map(|(b, c)| b * c).sum::<f64>()
    }
}

/// Interior knots per parent on the standardized scale: the sample quantiles in
/// `SPLINE_KNOT_QUANTILES` (nearest rank, duplicates removed), none for a parent
/// with fewer than `SPLINE_MIN_DISTINCT` distinct standardized values.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the rank is a rounded quantile of a non-empty vector, so it is non-negative and below its length"
)]
fn spline_knots(
    columns: &[Vec<f64>],
    parents: &[usize],
    centers: &[f64],
    scales: &[f64],
) -> Vec<Vec<f64>> {
    let mut knots: Vec<Vec<f64>> = Vec::with_capacity(parents.len());
    for (a, &j) in parents.iter().enumerate() {
        let mut sorted: Vec<f64> =
            columns[j].iter().map(|x| (x - centers[a]) / scales[a]).collect();
        sorted.sort_by(f64::total_cmp);
        let mut distinct = sorted.clone();
        distinct.dedup_by(|x, y| (*x - *y).abs() < 1e-9);
        if distinct.len() < SPLINE_MIN_DISTINCT {
            knots.push(Vec::new());
            continue;
        }
        let mut chosen: Vec<f64> = SPLINE_KNOT_QUANTILES
            .iter()
            .map(|q| sorted[((sorted.len() - 1) as f64 * q).round() as usize])
            .collect();
        chosen.dedup_by(|x, y| (*x - *y).abs() < 1e-9);
        knots.push(chosen);
    }
    knots
}

/// Fit the non-separable outcome mean of `columns[v]` on `columns[parents]` by a
/// streaming Givens QR of the ridge-augmented, unit-scaled design (no normal
/// equations, so the condition number is not squared): the augmented design's
/// smallest singular value is at least `sqrt(ridge)` against a largest of at most
/// `sqrt(n * ncols)`, a condition number below about `sqrt(ncols / 1e-6)`, so the
/// coefficients carry a relative error of order `1e-16 * cond`, far below the
/// `CLOSED_FORM_TOLERANCE`. `None` for a table the fitter refuses (fewer than two
/// parents, no parent with enough distinct values, too few rows per column) or a
/// numerically singular design.
fn fit_spline_mean(columns: &[Vec<f64>], v: usize, parents: &[usize]) -> Option<SplineMean> {
    let (n, p) = (columns[v].len(), parents.len());
    if p < 2 || n == 0 {
        return None;
    }
    let nf = n as f64;
    let mut centers = Vec::with_capacity(p);
    let mut scales = Vec::with_capacity(p);
    for &j in parents {
        let mean = columns[j].iter().sum::<f64>() / nf;
        let sd = (columns[j].iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / nf).sqrt();
        centers.push(mean);
        scales.push(if sd.is_finite() && sd > 1e-12 { sd } else { 1.0 });
    }
    let knots = spline_knots(columns, parents, &centers, &scales);
    if knots.iter().all(Vec::is_empty) {
        return None;
    }
    let row_of = |i: usize, z: &mut Vec<f64>, out: &mut Vec<f64>| {
        z.clear();
        z.extend(
            parents.iter().enumerate().map(|(a, &j)| (columns[j][i] - centers[a]) / scales[a]),
        );
        spline_columns(z, &knots, out);
    };
    let (mut z, mut cols) = (Vec::new(), Vec::new());
    row_of(0, &mut z, &mut cols);
    let m = 1 + cols.len();
    if n < m.saturating_mul(SPLINE_MIN_ROWS_PER_COLUMN) {
        return None;
    }
    // Pass one: root-mean-square of every non-intercept column.
    let mut rms = vec![0.0_f64; m];
    for i in 0..n {
        row_of(i, &mut z, &mut cols);
        for (c, x) in cols.iter().enumerate() {
            rms[1 + c] += x * x;
        }
    }
    let scale: Vec<f64> = rms
        .iter()
        .enumerate()
        .map(|(c, s)| {
            let r = (s / nf).sqrt();
            if c > 0 && r.is_finite() && r > 1e-12 { r } else { 1.0 }
        })
        .collect();
    // Pass two: rotate every data row, then every ridge row, into R and Q'y.
    let mut r = vec![vec![0.0_f64; m]; m];
    let mut rhs = vec![0.0_f64; m];
    let mut rotate = |row: &mut [f64], target: f64| {
        let mut t = target;
        for k in 0..m {
            let (a, b) = (r[k][k], row[k]);
            if b == 0.0 {
                continue;
            }
            let h = a.hypot(b);
            let (c, s) = (a / h, b / h);
            for (rj, xj) in r[k][k..].iter_mut().zip(row[k..].iter_mut()) {
                let (u, w) = (*rj, *xj);
                *rj = c * u + s * w;
                *xj = -s * u + c * w;
            }
            let (u, w) = (rhs[k], t);
            rhs[k] = c * u + s * w;
            t = -s * u + c * w;
        }
    };
    let mut row = vec![0.0_f64; m];
    for i in 0..n {
        row_of(i, &mut z, &mut cols);
        row[0] = 1.0;
        for (c, x) in cols.iter().enumerate() {
            row[1 + c] = x / scale[1 + c];
        }
        rotate(&mut row, columns[v][i]);
    }
    let sqrt_ridge = (SPLINE_CONDITIONING_RIDGE * nf).sqrt();
    for j in 1..m {
        row.fill(0.0);
        row[j] = sqrt_ridge;
        rotate(&mut row, 0.0);
    }
    let largest = (0..m).map(|k| r[k][k].abs()).fold(0.0_f64, f64::max);
    if !largest.is_finite() || (0..m).any(|k| r[k][k].abs() <= 1e-13 * largest) {
        return None;
    }
    let mut beta = vec![0.0_f64; m];
    for k in (0..m).rev() {
        let tail: f64 = ((k + 1)..m).map(|j| r[k][j] * beta[j]).sum();
        beta[k] = (rhs[k] - tail) / r[k][k];
    }
    let coeffs: Vec<f64> = (1..m).map(|c| beta[c] / scale[c]).collect();
    if beta.iter().any(|b| !b.is_finite()) {
        return None;
    }
    Some(SplineMean { intercept: beta[0], coeffs, centers, scales, knots })
}

/// One variable's fitted mean function in the independent check.
enum NodeMean {
    Linear { intercept: f64, slopes: Vec<f64> },
    Spline(SplineMean),
}

/// Whether the closed-form point agrees with the replayed one.
pub(super) fn closed_form_agrees(closed: f64, replayed: f64) -> bool {
    (closed - replayed).abs() <= CLOSED_FORM_TOLERANCE * (1.0 + replayed.abs())
}

/// The point of a cross-world contrast from first principles, sharing no code
/// with the fitting or evaluation path: per variable a least-squares mean on its
/// parents (ordinary least squares by column-pivoted QR of the centered
/// design; for `outcome` under the
/// non-separable family the ridge-stabilized basis regression of
/// [`fit_spline_mean`] by Givens QR), each unit's exogenous term is its residual
/// about that mean, and each world is evaluated node by node in topological
/// order with every edge reading the world its route names.
///
/// Tolerance argument. The linear solve is a column-equilibrated pivoted QR of
/// the centered design, so its coefficient error is of order `1e-16 * cond(X)`;
/// the check declines (`None`) rather than compare when the backend reports the
/// design rank deficient. The basis solve is bounded above (condition
/// number below about `5e3` by the ridge). Both errors enter the point through
/// one linear map of the residuals, so a relative `1e-6` on `1 + |point|` leaves
/// several orders of magnitude for legitimate rounding and is still far below any
/// mistake in a basis column, a knot, a scale, the ridge, a route or a noise
/// term (each of which moves the point by at least a fraction of the fit error).
///
/// `None` when a system is numerically singular or the fitter itself would have
/// refused the table, so no closed form exists to compare with.
pub(super) fn closed_form_point(
    n: usize,
    edges: &[(u32, u32)],
    columns: &[Vec<f64>],
    query: &CrossWorldQuery,
    mechanism: NestedOutcomeMechanism,
    outcome: u32,
) -> Option<f64> {
    let rows = columns.first()?.len();
    let rows_f = rows as f64;
    if antecedent_graph::error::refuse_directed_cycle(
        antecedent_graph::error::DagOperation::Identification,
        antecedent_graph::error::CyclicNextAttempt::Unlicensed {
            consumer: "cross_world closed form",
            attempt: "hold the edges as a CyclicDigraph and read feedback from condensation; \
                      this recomputation does not evaluate a cyclic contrast",
        },
        n,
        edges,
    )
    .is_err()
    {
        return None;
    }
    let parents: Vec<Vec<usize>> = (0..n)
        .map(|v| edges.iter().filter(|e| e.1 as usize == v).map(|e| e.0 as usize).collect())
        .collect();
    // Topological order (Kahn).
    let mut indegree: Vec<usize> = parents.iter().map(Vec::len).collect();
    let mut order = Vec::with_capacity(n);
    let mut ready: Vec<usize> = (0..n).filter(|&v| indegree[v] == 0).collect();
    while let Some(v) = ready.pop() {
        order.push(v);
        for &(a, b) in edges {
            if a as usize == v {
                indegree[b as usize] -= 1;
                if indegree[b as usize] == 0 {
                    ready.push(b as usize);
                }
            }
        }
    }
    if order.len() != n {
        return None;
    }
    // Per variable: the mean function and each unit's residual (exogenous term).
    let mut fits: Vec<NodeMean> = Vec::with_capacity(n);
    let mut residual: Vec<Vec<f64>> = vec![Vec::new(); n];
    let (mut z, mut cols) = (Vec::new(), Vec::new());
    for v in 0..n {
        let fit = if mechanism == NestedOutcomeMechanism::NonSeparableBasis && v == outcome as usize
        {
            NodeMean::Spline(fit_spline_mean(columns, v, &parents[v])?)
        } else {
            let (intercept, slopes) = ols_on_parents(columns, v, &parents[v])?;
            NodeMean::Linear { intercept, slopes }
        };
        let mut pv = vec![0.0; parents[v].len()];
        residual[v] = (0..rows)
            .map(|i| {
                for (a, &j) in parents[v].iter().enumerate() {
                    pv[a] = columns[j][i];
                }
                columns[v][i] - fit.mean(&pv, &mut z, &mut cols)
            })
            .collect();
        fits.push(fit);
    }
    let worlds = query.worlds();
    let mut values = vec![vec![Vec::<f64>::new(); n]; worlds.len()];
    let mut pv = Vec::new();
    for &v in &order {
        let variable = VariableId::from_raw(u32::try_from(v).ok()?);
        for (w, world) in worlds.iter().enumerate() {
            let column: Vec<f64> = if let Some(level) = world.intervention_of(variable) {
                vec![level; rows]
            } else {
                let sources: Vec<usize> = parents[v]
                    .iter()
                    .map(|&j| {
                        let parent = VariableId::from_raw(u32::try_from(j).ok()?);
                        Some(
                            world
                                .routes()
                                .iter()
                                .find(|r| r.parent == parent && r.child == variable)
                                .map_or(w, |r| r.source.index()),
                        )
                    })
                    .collect::<Option<Vec<usize>>>()?;
                (0..rows)
                    .map(|i| {
                        pv.clear();
                        pv.extend(parents[v].iter().zip(&sources).map(|(&j, &s)| values[s][j][i]));
                        fits[v].mean(&pv, &mut z, &mut cols) + residual[v][i]
                    })
                    .collect()
            };
            values[w][v] = column;
        }
    }
    let read =
        |o: antecedent_core::WorldObservation| &values[o.world.index()][o.variable.raw() as usize];
    let (plus, minus) = (read(query.plus()), read(query.minus()));
    Some(plus.iter().zip(minus).map(|(a, b)| a - b).sum::<f64>() / rows_f)
}

impl NodeMean {
    fn mean(&self, parent_values: &[f64], z: &mut Vec<f64>, cols: &mut Vec<f64>) -> f64 {
        match self {
            Self::Linear { intercept, slopes } => {
                let mut y = *intercept;
                for (b, x) in slopes.iter().zip(parent_values) {
                    y += b * x;
                }
                y
            }
            Self::Spline(spline) => spline.mean(parent_values, z, cols),
        }
    }
}

#[cfg(test)]
mod tests {
    use antecedent_core::{CrossWorldQuery, VariableId};

    use super::closed_form_point;
    use crate::gcm::NestedOutcomeMechanism;

    fn query() -> CrossWorldQuery {
        CrossWorldQuery::natural_direct(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
            VariableId::from_raw(2),
            0.0,
            1.0,
        )
        .expect("three distinct variables")
    }

    #[test]
    fn a_cycle_is_refused_before_the_order_and_a_chain_still_evaluates() {
        let columns =
            vec![vec![0.0, 1.0, 2.0, 3.0], vec![1.0, 2.0, 3.0, 4.0], vec![1.0, 3.0, 5.0, 7.0]];
        let asked = query();
        assert!(
            closed_form_point(
                3,
                &[(0, 1), (1, 0), (1, 2)],
                &columns,
                &asked,
                NestedOutcomeMechanism::LinearGaussian,
                2,
            )
            .is_none()
        );
        assert!(
            closed_form_point(
                3,
                &[(0, 1), (1, 2)],
                &columns,
                &asked,
                NestedOutcomeMechanism::LinearGaussian,
                2,
            )
            .is_some()
        );
    }
}
