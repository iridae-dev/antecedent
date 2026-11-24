//! The admissible region of a preference set and its linear programs, and the analysis of
//! scores over it: value ranges, pairwise margins with the points attaining them, and how each
//! action stands against its rivals.

// SPDX-License-Identifier: MIT OR Apache-2.0

#[allow(clippy::wildcard_imports, reason = "a part of its parent module")]
use super::*;

// ---------------------------------------------------------------------------------------------
// The admissible region and its linear programs
// ---------------------------------------------------------------------------------------------

/// `{theta : rows, lower <= theta <= upper}`, every row `coefficients . theta <= rhs`.
#[derive(Clone, Debug)]
pub(super) struct Region {
    pub(super) rows: Vec<LinearConstraint>,
    pub(super) lower: Vec<f64>,
    pub(super) upper: Vec<f64>,
    /// Read before every linear program over the region.
    pub(super) cancellation: Option<CancellationToken>,
}

/// An optimum of a function over a region and a point attaining it.
#[derive(Clone, Debug)]
pub(super) struct Extreme {
    pub(super) value: f64,
    pub(super) point: Vec<f64>,
}

/// `f[0] + sum_j f[j + 1] theta_j`.
pub(super) fn affine(f: &[f64], theta: &[f64]) -> f64 {
    f[0] + f[1..].iter().zip(theta).map(|(c, t)| c * t).sum::<f64>()
}

pub(super) fn difference(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| x - y).collect()
}

pub(super) fn negated(f: &[f64]) -> Vec<f64> {
    f.iter().map(|v| -v).collect()
}

/// The worst of several affine pieces at a point: `min_s f_s(theta)`.
pub(super) fn worst_at(pieces: &[Vec<f64>], theta: &[f64]) -> f64 {
    pieces.iter().map(|f| affine(f, theta)).fold(f64::INFINITY, f64::min)
}

impl Region {
    /// The declared ranges as variable bounds and every relation row of the set.
    pub(super) fn of(set: &PreferenceSet) -> Self {
        let rows = set.inequality_rows();
        let rows = rows
            .coefficients
            .into_iter()
            .zip(rows.rhs)
            .map(|(coefficients, rhs)| LinearConstraint {
                coefficients,
                sense: ConstraintSense::AtMost,
                rhs,
            })
            .collect();
        let (lower, upper) = set.bounds().into_iter().unzip();
        Self { rows, lower, upper, cancellation: None }
    }

    pub(super) fn dimension(&self) -> usize {
        self.lower.len()
    }

    pub(super) fn outcome(
        &self,
        program: &LinearProgram,
        lp: LpOptions,
    ) -> Result<Option<LpSolution>, E> {
        if self.cancellation.as_ref().is_some_and(CancellationToken::is_cancelled) {
            return Err(E::Cancelled);
        }
        match linear_program::solve(program, lp).map_err(E::Lp)? {
            LpOutcome::Optimal(solution) => Ok(Some(solution)),
            LpOutcome::Infeasible(_) => Ok(None),
            LpOutcome::Unbounded { .. } => Err(E::UnboundedProgram),
        }
    }

    /// A point of the region with `f(theta) <= 0` for each affine `f`, or `None`.
    pub(super) fn feasible(
        &self,
        extra: &[Vec<f64>],
        lp: LpOptions,
    ) -> Result<Option<Vec<f64>>, E> {
        let constant_holds = |f: &Vec<f64>| f[0] <= lp.feasibility_tolerance * (1.0 + f[0].abs());
        let mut constraints = self.rows.clone();
        for f in extra {
            if f[1..].iter().all(|c| c.abs().total_cmp(&0.0).is_eq()) {
                if !constant_holds(f) {
                    return Ok(None);
                }
                continue;
            }
            constraints.push(LinearConstraint {
                coefficients: f[1..].to_vec(),
                sense: ConstraintSense::AtMost,
                rhs: -f[0],
            });
        }
        if self.dimension() == 0 {
            return Ok(Some(Vec::new()));
        }
        let program = LinearProgram {
            objective: vec![0.0; self.dimension()],
            constraints,
            lower: self.lower.clone(),
            upper: self.upper.clone(),
        };
        Ok(self.outcome(&program, lp)?.map(|solution| solution.x))
    }

    /// `min over the region of max_k f_k(theta)` and a minimising point.
    pub(super) fn min_of_max(&self, pieces: &[Vec<f64>], lp: LpOptions) -> Result<Extreme, E> {
        let n = self.dimension();
        let point = if n == 0 {
            Vec::new()
        } else if let [f] = pieces {
            let program = LinearProgram {
                objective: f[1..].to_vec(),
                constraints: self.rows.clone(),
                lower: self.lower.clone(),
                upper: self.upper.clone(),
            };
            self.outcome(&program, lp)?.ok_or(E::UnexpectedInfeasible)?.x
        } else {
            // Epigraph form: minimise t subject to f_k(theta) <= t.
            let mut constraints: Vec<LinearConstraint> = self
                .rows
                .iter()
                .map(|row| {
                    let mut coefficients = row.coefficients.clone();
                    coefficients.push(0.0);
                    LinearConstraint { coefficients, sense: ConstraintSense::AtMost, rhs: row.rhs }
                })
                .collect();
            for f in pieces {
                let mut coefficients = f[1..].to_vec();
                coefficients.push(-1.0);
                constraints.push(LinearConstraint {
                    coefficients,
                    sense: ConstraintSense::AtMost,
                    rhs: -f[0],
                });
            }
            let mut objective = vec![0.0; n];
            objective.push(1.0);
            let mut lower = self.lower.clone();
            lower.push(f64::NEG_INFINITY);
            let mut upper = self.upper.clone();
            upper.push(f64::INFINITY);
            let program = LinearProgram { objective, constraints, lower, upper };
            let mut x = self.outcome(&program, lp)?.ok_or(E::UnexpectedInfeasible)?.x;
            x.truncate(n);
            x
        };
        let value = pieces.iter().map(|f| affine(f, &point)).fold(f64::NEG_INFINITY, f64::max);
        Ok(Extreme { value, point })
    }

    /// Lowest and highest value of each coordinate over the region: two linear programs per
    /// coordinate.
    pub(super) fn parameter_ranges(&self, lp: LpOptions) -> Result<Vec<(f64, f64)>, E> {
        let n = self.dimension();
        (0..n)
            .map(|j| {
                let mut coordinate = vec![0.0; n + 1];
                coordinate[j + 1] = 1.0;
                let pieces = std::slice::from_ref(&coordinate);
                Ok((self.min_of_max(pieces, lp)?.value, self.max_of_min(pieces, lp)?.value))
            })
            .collect()
    }

    /// `max over the region of min_k f_k(theta)` and a maximising point.
    pub(super) fn max_of_min(&self, pieces: &[Vec<f64>], lp: LpOptions) -> Result<Extreme, E> {
        let flipped: Vec<Vec<f64>> = pieces.iter().map(|f| negated(f)).collect();
        let extreme = self.min_of_max(&flipped, lp)?;
        Ok(Extreme { value: -extreme.value, point: extreme.point })
    }

    /// `min over the region of (g_a - g_b)` with `g = min over pieces`:
    /// `min_s min_theta max_s' (v_a(s) - v_b(s'))`.
    pub(super) fn min_margin(
        &self,
        a: &[Vec<f64>],
        b: &[Vec<f64>],
        lp: LpOptions,
    ) -> Result<Extreme, E> {
        let mut best: Option<Extreme> = None;
        for va in a {
            let pieces: Vec<Vec<f64>> = b.iter().map(|vb| difference(va, vb)).collect();
            let extreme = self.min_of_max(&pieces, lp)?;
            if best.as_ref().is_none_or(|current| extreme.value < current.value) {
                best = Some(extreme);
            }
        }
        best.ok_or(E::UnexpectedInfeasible)
    }

    /// `max over the region of (g_a - g_b)`: `max_s' max_theta min_s (v_a(s) - v_b(s'))`.
    pub(super) fn max_margin(
        &self,
        a: &[Vec<f64>],
        b: &[Vec<f64>],
        lp: LpOptions,
    ) -> Result<Extreme, E> {
        let mut best: Option<Extreme> = None;
        for vb in b {
            let pieces: Vec<Vec<f64>> = a.iter().map(|va| difference(va, vb)).collect();
            let extreme = self.max_of_min(&pieces, lp)?;
            if best.as_ref().is_none_or(|current| extreme.value > current.value) {
                best = Some(extreme);
            }
        }
        best.ok_or(E::UnexpectedInfeasible)
    }
}

// ---------------------------------------------------------------------------------------------
// Analysis of scores over the region
// ---------------------------------------------------------------------------------------------

/// Range of `g_a - g_b` over the region with the points attaining it.
#[derive(Clone, Debug)]
pub(super) struct Margin {
    pub(super) min: f64,
    pub(super) min_at: Vec<f64>,
    pub(super) max: f64,
    pub(super) max_at: Vec<f64>,
}

impl Margin {
    pub(super) fn flipped(&self) -> Self {
        Self {
            min: -self.max,
            min_at: self.max_at.clone(),
            max: -self.min,
            max_at: self.min_at.clone(),
        }
    }
}

/// Pairwise margins, stored for `a < b`.
#[derive(Clone, Debug)]
pub(super) struct MarginTable {
    pub(super) n: usize,
    pub(super) entries: Vec<Option<Margin>>,
}

impl MarginTable {
    pub(super) fn new(n: usize) -> Self {
        Self { n, entries: vec![None; n * n] }
    }

    pub(super) fn get(&self, a: usize, b: usize) -> Option<Margin> {
        if a < b {
            self.entries[a * self.n + b].clone()
        } else {
            self.entries[b * self.n + a].as_ref().map(Margin::flipped)
        }
    }
}

/// Everything computed about a set of scores over the region, in larger-is-better
/// orientation.
#[derive(Clone, Debug)]
pub(super) struct Analysis {
    pub(super) value: Vec<Option<(f64, f64)>>,
    pub(super) margins: MarginTable,
    pub(super) standing: Standing,
    pub(super) possibly: Vec<Option<bool>>,
    pub(super) regret: Vec<Option<f64>>,
    pub(super) share: Vec<Option<f64>>,
    pub(super) measure_value: Vec<Option<f64>>,
    pub(super) unresolved_share: Option<f64>,
    /// Margins within this distance of zero count as ties (see `Prepared::tie_tolerance`).
    pub(super) tolerance: f64,
}

/// How each admissible action stands against its rivals over the whole region, read from the
/// margin ranges.
#[derive(Clone, Debug)]
pub(super) struct Standing {
    /// Margin over every rival at least `-tolerance` everywhere.
    pub(super) never_beaten: Vec<bool>,
    /// Margin over every rival above `tolerance` everywhere.
    pub(super) uniquely: Vec<bool>,
    /// Rivals whose margin over the action exceeds `tolerance` everywhere.
    pub(super) dominated_by: Vec<Vec<usize>>,
    /// Rivals whose margin range against the action touches zero within `tolerance`.
    pub(super) touching: Vec<Vec<usize>>,
}

impl Standing {
    /// Read from the margin table over the active actions. With no rival an action is
    /// vacuously never beaten and uniquely best.
    pub(super) fn of(margins: &MarginTable, active: &[usize], n: usize, tolerance: f64) -> Self {
        let mut standing = Self {
            never_beaten: vec![false; n],
            uniquely: vec![false; n],
            dominated_by: vec![Vec::new(); n],
            touching: vec![Vec::new(); n],
        };
        for &a in active {
            let mut never_beaten = true;
            let mut uniquely = true;
            for &b in active.iter().filter(|&&b| b != a) {
                let Some(margin) = margins.get(a, b) else { continue };
                never_beaten &= margin.min >= -tolerance;
                uniquely &= margin.min > tolerance;
                if margin.max < -tolerance {
                    standing.dominated_by[a].push(b);
                }
                if margin.min <= tolerance && margin.max >= -tolerance {
                    standing.touching[a].push(b);
                }
            }
            standing.never_beaten[a] = never_beaten;
            standing.uniquely[a] = uniquely;
        }
        standing
    }

    pub(super) fn empty(n: usize) -> Self {
        Self {
            never_beaten: vec![false; n],
            uniquely: vec![false; n],
            dominated_by: vec![Vec::new(); n],
            touching: vec![Vec::new(); n],
        }
    }
}

/// Bound on the floating error of a weighted sum of `n` terms whose weighted magnitudes sum to
/// `magnitude`: `2 (n + 1) eps magnitude`.
#[allow(clippy::cast_precision_loss, reason = "draw counts are far below 2^53")]
pub(super) fn summation_rounding(n: usize, magnitude: f64) -> f64 {
    2.0 * (n as f64 + 1.0) * f64::EPSILON * magnitude
}

/// The unique best index by `score`, or every index within `tolerance` of the best.
pub(super) fn pick(
    active: &[usize],
    score: impl Fn(usize) -> f64,
    tolerance: f64,
) -> Result<usize, Vec<usize>> {
    let best = active.iter().map(|&a| score(a)).fold(f64::NEG_INFINITY, f64::max);
    let tied: Vec<usize> =
        active.iter().copied().filter(|&a| score(a) >= best - tolerance).collect();
    match tied.as_slice() {
        [only] => Ok(*only),
        _ => Err(tied),
    }
}
