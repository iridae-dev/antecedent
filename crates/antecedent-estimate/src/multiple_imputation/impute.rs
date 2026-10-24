//! Proper imputation models: chained equations and joint normal data augmentation.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;
use std::sync::Arc;

use antecedent_core::execution::AdmissionLease;
use antecedent_core::rng_key::{
    ChainId, FoldId, ReductionPolicy, ReplicateId, RngKey, SemanticOperation,
};
use antecedent_core::{CausalRng, ExecutionContext};
use antecedent_data::{ImputationModel, ImputationSpec};
use antecedent_kernels::{
    bayesian_bootstrap_weights, chol_inverse, chol_solve_into, cholesky_spd_into, sample_gamma,
    standard_normal,
};

use super::{
    DECLARE_COMPLETE_CASE, IMPUTATION_OPERATION, ImputationUnits, MAX_IMPUTATION_VARIABLES,
    MultipleImputationError, USE_CHAINED_EQUATIONS, check_spec,
};

/// Logistic IRLS iterations of a binary conditional model.
const LOGISTIC_ITERATIONS: usize = 50;
/// Ridge on the non-intercept coefficients of a logistic conditional model; keeps a
/// separated bootstrap sample finite without moving a well-identified fit.
const LOGISTIC_RIDGE: f64 = 1e-4;
/// Relative jitter added to a diagonal that failed to factor.
const JITTER: f64 = 1e-9;

/// Writes row `i`'s design (intercept, then the other variables) from the current columns.
type DesignRow<'a> = dyn Fn(usize, &mut [f64], &[Vec<f64>]) + 'a;

/// One variable of the imputation model: its name and its values, `NaN` where missing.
#[derive(Clone, Debug, PartialEq)]
pub struct ImputationColumn {
    /// Variable name.
    pub name: Arc<str>,
    /// Values; `NaN` marks a missing cell.
    pub values: Arc<[f64]>,
}

impl ImputationColumn {
    /// A column from its name and values.
    #[must_use]
    pub fn new(name: impl Into<Arc<str>>, values: impl Into<Arc<[f64]>>) -> Self {
        Self { name: name.into(), values: values.into() }
    }
}

/// How a variable's conditional model is drawn.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum VariableKind {
    /// Normal linear regression.
    Continuous,
    /// Logistic regression; every observed value is 0 or 1.
    Binary,
}

impl VariableKind {
    /// Stable name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Continuous => "continuous",
            Self::Binary => "binary",
        }
    }
}

/// What the imputer learned about one variable.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImputedVariable {
    /// Variable name.
    pub name: Arc<str>,
    /// Conditional model kind.
    pub kind: VariableKind,
    /// Cells imputed.
    pub missing: usize,
}

/// One completed table: every variable of the imputation model with its missing cells
/// drawn. Holds the admission lease that funded it; dropping the table releases it.
#[derive(Debug)]
pub struct CompletedTable {
    index: u32,
    names: Arc<[Arc<str>]>,
    columns: Vec<Arc<[f64]>>,
    _lease: AdmissionLease,
}

impl CompletedTable {
    /// Zero-based imputation index.
    #[must_use]
    pub const fn index(&self) -> u32 {
        self.index
    }

    /// Variable names, in the imputer's order.
    #[must_use]
    pub fn names(&self) -> &[Arc<str>] {
        &self.names
    }

    /// Completed columns, in the imputer's order.
    #[must_use]
    pub fn columns(&self) -> &[Arc<[f64]>] {
        &self.columns
    }

    /// The completed column named `name`.
    #[must_use]
    pub fn column(&self, name: &str) -> Option<&Arc<[f64]>> {
        self.names.iter().position(|n| n.as_ref() == name).map(|j| &self.columns[j])
    }
}

/// A checked imputation model over fixed data.
#[derive(Clone, Debug)]
pub struct Imputer {
    spec: ImputationSpec,
    units: ImputationUnits,
    unit_count: usize,
    rows: usize,
    names: Arc<[Arc<str>]>,
    values: Vec<Arc<[f64]>>,
    kinds: Vec<VariableKind>,
    missing_rows: Vec<Vec<usize>>,
    observed_rows: Vec<Vec<usize>>,
    visit: Vec<usize>,
    /// Neighbouring time steps on each side whose values enter a conditional model; zero for
    /// exchangeable rows.
    lags: usize,
    /// Observed rows with a full neighbourhood, per variable; read only when `lags > 0`.
    fit_rows: Vec<Vec<usize>>,
}

impl Imputer {
    /// Check `spec` and the data and prepare the imputation model.
    ///
    /// # Errors
    ///
    /// The specification's own refusals ([`check_spec`]); no variables or more than
    /// [`MAX_IMPUTATION_VARIABLES`]; columns of different lengths; an infinite value; units
    /// whose length differs from the rows; a variable with too few observed cells to
    /// identify its conditional model; an incomplete binary variable under the joint normal
    /// model; too few rows for the joint normal posterior.
    #[allow(clippy::float_cmp, reason = "a binary variable's observed values are exactly 0 or 1")]
    #[allow(clippy::too_many_lines, reason = "one pass of checks, each its own typed refusal")]
    pub fn new(
        columns: Vec<ImputationColumn>,
        spec: ImputationSpec,
        units: ImputationUnits,
    ) -> Result<Self, MultipleImputationError> {
        check_spec(&spec)?;
        let p = columns.len();
        if p == 0 || p > MAX_IMPUTATION_VARIABLES {
            return Err(MultipleImputationError::invalid(
                "multiple_imputation.variable_count",
                format!(
                    "an imputation model reads 1 to {MAX_IMPUTATION_VARIABLES} variables, got {p}"
                ),
            ));
        }
        let rows = columns[0].values.len();
        if columns.iter().any(|c| c.values.len() != rows) {
            return Err(MultipleImputationError::invalid(
                "multiple_imputation.column_length",
                "the imputation model's columns differ in length",
            ));
        }
        if columns.iter().any(|c| c.values.iter().any(|v| v.is_infinite())) {
            return Err(MultipleImputationError::invalid(
                "multiple_imputation.non_finite",
                "an imputation column holds an infinite value",
            ));
        }
        if let ImputationUnits::Groups(groups) = &units {
            if groups.len() != rows {
                return Err(MultipleImputationError::invalid(
                    "multiple_imputation.units_shape",
                    format!("{} group labels for {rows} rows", groups.len()),
                ));
            }
        }
        let unit_count = units.count(rows);
        let mut kinds = Vec::with_capacity(p);
        let mut missing_rows = Vec::with_capacity(p);
        let mut observed_rows = Vec::with_capacity(p);
        for column in &columns {
            let missing: Vec<usize> = (0..rows).filter(|&i| column.values[i].is_nan()).collect();
            let observed: Vec<usize> = (0..rows).filter(|&i| !column.values[i].is_nan()).collect();
            let binary = observed.iter().all(|&i| {
                let v = column.values[i];
                v == 0.0 || v == 1.0
            });
            let has_both = observed.iter().any(|&i| column.values[i] == 0.0)
                && observed.iter().any(|&i| column.values[i] == 1.0);
            let kind =
                if binary && has_both { VariableKind::Binary } else { VariableKind::Continuous };
            // A conditional model on the other p - 1 variables and an intercept needs more
            // observed cells than coefficients, with at least one degree of freedom left.
            if !missing.is_empty() && observed.len() < p + 2 {
                return Err(MultipleImputationError::not_applicable(
                    "multiple_imputation.not_identified",
                    format!(
                        "variable `{}` has {} observed cells; its conditional model on {} other \
                         variables needs at least {}",
                        column.name,
                        observed.len(),
                        p - 1,
                        p + 2
                    ),
                    DECLARE_COMPLETE_CASE,
                ));
            }
            if observed.len() < 2 {
                return Err(MultipleImputationError::not_applicable(
                    "multiple_imputation.not_identified",
                    format!("variable `{}` has fewer than two observed cells", column.name),
                    DECLARE_COMPLETE_CASE,
                ));
            }
            if spec.model == ImputationModel::JointNormal
                && kind == VariableKind::Binary
                && !missing.is_empty()
            {
                return Err(MultipleImputationError::not_applicable(
                    "multiple_imputation.binary_under_joint_normal",
                    format!(
                        "variable `{}` is binary and incomplete; the joint normal model would \
                         impute values other than 0 and 1",
                        column.name
                    ),
                    USE_CHAINED_EQUATIONS,
                ));
            }
            kinds.push(kind);
            missing_rows.push(missing);
            observed_rows.push(observed);
        }
        if spec.model == ImputationModel::JointNormal
            && matches!(units, ImputationUnits::Rows)
            && rows <= p + 1
        {
            return Err(MultipleImputationError::not_applicable(
                "multiple_imputation.not_identified",
                format!(
                    "the joint normal posterior over {p} variables needs more than {} rows, got {rows}",
                    p + 1
                ),
                DECLARE_COMPLETE_CASE,
            ));
        }
        if unit_count < 2 {
            return Err(MultipleImputationError::not_applicable(
                "multiple_imputation.not_identified",
                "the parameter draw needs at least two sampling units",
                DECLARE_COMPLETE_CASE,
            ));
        }
        let mut visit: Vec<usize> = (0..p).filter(|&j| !missing_rows[j].is_empty()).collect();
        visit.sort_by_key(|&j| (missing_rows[j].len(), j));
        Ok(Self {
            spec,
            units,
            unit_count,
            rows,
            names: columns.iter().map(|c| Arc::clone(&c.name)).collect(),
            values: columns.into_iter().map(|c| c.values).collect(),
            kinds,
            missing_rows,
            observed_rows,
            visit,
            lags: 0,
            fit_rows: Vec::new(),
        })
    }

    /// Condition every variable's imputation model on the values `lags` time steps before and
    /// after the row, for a series whose rows are ordered in time.
    ///
    /// The model of variable `j` regresses it on the other variables at the same step and on
    /// every variable at steps `t - lags ..= t - 1` and `t + 1 ..= t + lags`, read from the
    /// current completed columns, so the lag columns of an imputed cell are the imputed value
    /// itself. A series is not a sample of exchangeable rows; the conditional model is the full
    /// conditional of a stationary process of order `lags`, and the parameter draw resamples
    /// contiguous blocks (declare them as the imputation's [`ImputationUnits::Groups`]). Fit
    /// rows are the observed rows whose whole neighbourhood lies inside the series; a missing
    /// cell closer to an end than `lags` is drawn with the nearest end row standing in for the
    /// absent neighbour.
    ///
    /// # Errors
    ///
    /// The joint normal model (it has no lagged form); no lags; or a variable with too few
    /// interior observed rows to identify its conditional model.
    pub fn with_dynamics(mut self, lags: usize) -> Result<Self, MultipleImputationError> {
        if lags == 0 {
            return Err(MultipleImputationError::invalid(
                "multiple_imputation.dynamics_lags",
                "a dynamic imputation model conditions on at least one neighbouring time step",
            ));
        }
        if self.spec.model == ImputationModel::JointNormal {
            return Err(MultipleImputationError::not_applicable(
                "multiple_imputation.dynamics_joint_normal",
                "the joint normal model draws each row from one mean and covariance and has no \
                 lagged form",
                USE_CHAINED_EQUATIONS,
            ));
        }
        self.lags = lags;
        let width = self.width();
        let interior = |i: usize| i >= lags && i + lags < self.rows;
        let mut fit_rows = Vec::with_capacity(self.names.len());
        for (j, observed) in self.observed_rows.iter().enumerate() {
            let rows: Vec<usize> = observed.iter().copied().filter(|&i| interior(i)).collect();
            if !self.missing_rows[j].is_empty() && rows.len() < width + 2 {
                return Err(MultipleImputationError::not_applicable(
                    "multiple_imputation.not_identified",
                    format!(
                        "variable `{}` has {} observed cells with a full neighbourhood of {lags} \
                         steps; its dynamic conditional model has {width} coefficients and needs \
                         at least {}",
                        self.names[j],
                        rows.len(),
                        width + 2
                    ),
                    DECLARE_COMPLETE_CASE,
                ));
            }
            fit_rows.push(rows);
        }
        self.fit_rows = fit_rows;
        Ok(self)
    }

    /// Neighbouring steps on each side that enter a conditional model (zero for rows).
    #[must_use]
    pub const fn lags(&self) -> usize {
        self.lags
    }

    /// Coefficients of one conditional model: the intercept, the other variables at the same
    /// step, and every variable at each neighbouring step.
    fn width(&self) -> usize {
        let p = self.names.len();
        p + 2 * self.lags * p
    }

    /// Rows variable `j`'s conditional model is fitted on: its observed rows, or those with a
    /// full neighbourhood under a dynamic model.
    fn fit_rows_of(&self, j: usize) -> &[usize] {
        if self.lags == 0 { &self.observed_rows[j] } else { &self.fit_rows[j] }
    }

    /// Writes row `i`'s design for variable `j` from the current completed columns: the
    /// intercept, the other variables at row `i`, then each variable at the neighbouring
    /// rows `i - lags ..= i - 1` and `i + 1 ..= i + lags` (the nearest end row stands in for a
    /// neighbour beyond the series).
    fn design_row(&self, j: usize) -> impl Fn(usize, &mut [f64], &[Vec<f64>]) + '_ {
        let lags = self.lags;
        let last = self.rows.saturating_sub(1);
        move |i: usize, out: &mut [f64], current: &[Vec<f64>]| {
            out[0] = 1.0;
            let mut c = 1;
            for (other, column) in current.iter().enumerate() {
                if other != j {
                    out[c] = column[i];
                    c += 1;
                }
            }
            for step in 1..=lags {
                let before = i.saturating_sub(step);
                let after = (i + step).min(last);
                for column in current {
                    out[c] = column[before];
                    out[c + 1] = column[after];
                    c += 2;
                }
            }
        }
    }

    /// The declared specification.
    #[must_use]
    pub const fn spec(&self) -> &ImputationSpec {
        &self.spec
    }

    /// The resampling units.
    #[must_use]
    pub const fn units(&self) -> &ImputationUnits {
        &self.units
    }

    /// Number of resampling units.
    #[must_use]
    pub const fn unit_count(&self) -> usize {
        self.unit_count
    }

    /// Rows of the data.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Each variable's name, kind and imputed cells.
    #[must_use]
    pub fn variables(&self) -> Vec<ImputedVariable> {
        (0..self.names.len())
            .map(|j| ImputedVariable {
                name: Arc::clone(&self.names[j]),
                kind: self.kinds[j],
                missing: self.missing_rows[j].len(),
            })
            .collect()
    }

    /// `(name, cells imputed)` per variable.
    #[must_use]
    pub fn imputed_cells(&self) -> Vec<(Arc<str>, usize)> {
        (0..self.names.len())
            .map(|j| (Arc::clone(&self.names[j]), self.missing_rows[j].len()))
            .collect()
    }

    /// Bytes one completed table and its working buffers occupy; leased before each draw.
    #[must_use]
    pub fn imputation_bytes(&self) -> u64 {
        let p = self.names.len();
        let k = self.width();
        // The completed columns, the per-row design of one conditional model, the row
        // weights and the scratch of the solves.
        let cells = self.rows.saturating_mul(2 * p + 4).saturating_add(8 * k * k + 64);
        u64::try_from(cells.saturating_mul(size_of::<f64>())).unwrap_or(u64::MAX)
    }

    /// Draw completed table `index`.
    ///
    /// # Errors
    ///
    /// Cancellation, an admission pool that cannot fund the table (nothing is allocated),
    /// an index beyond the declared imputations, or a conditional model that cannot be
    /// fitted on a bootstrap draw.
    pub fn impute(
        &self,
        index: u32,
        ctx: &ExecutionContext,
    ) -> Result<CompletedTable, MultipleImputationError> {
        if index >= self.spec.imputations {
            return Err(MultipleImputationError::invalid(
                "multiple_imputation.index",
                format!("imputation {index} is beyond the declared {}", self.spec.imputations),
            ));
        }
        check_cancelled(ctx)?;
        let lease = ctx.acquire_admission(0, self.imputation_bytes()).map_err(|error| {
            MultipleImputationError::invalid(
                "multiple_imputation.admission_refused",
                format!(
                    "the admission pool cannot fund one completed table of {} bytes: {error}",
                    self.imputation_bytes()
                ),
            )
        })?;
        let mut rng = RngKey::new(
            SemanticOperation::new(IMPUTATION_OPERATION),
            FoldId::new(0),
            ChainId::new(u64::from(index)),
            ReplicateId::new(0),
            ReductionPolicy::Strict,
        )
        .open(self.spec.seed);
        let mut current: Vec<Vec<f64>> = self.values.iter().map(|v| v.to_vec()).collect();
        for &j in &self.visit {
            let observed = &self.observed_rows[j];
            for &i in &self.missing_rows[j] {
                let pick = observed[uniform_index(&mut rng, observed.len())];
                current[j][i] = self.values[j][pick];
            }
        }
        if !self.visit.is_empty() {
            for _ in 0..self.spec.iterations {
                check_cancelled(ctx)?;
                match self.spec.model {
                    ImputationModel::JointNormal => self.augment(&mut current, &mut rng)?,
                    _ => {
                        for &j in &self.visit {
                            self.chained_step(j, &mut current, &mut rng)?;
                        }
                    }
                }
            }
        }
        Ok(CompletedTable {
            index,
            names: Arc::clone(&self.names),
            columns: current.into_iter().map(Arc::from).collect(),
            _lease: lease,
        })
    }

    /// Bayesian-bootstrap weight of every row: one Dirichlet(1, …, 1) draw per unit,
    /// normalized to mean one over units, shared by the unit's rows.
    fn row_weights(&self, rng: &mut CausalRng) -> Vec<f64> {
        let unit_weights = bayesian_bootstrap_weights(rng, self.unit_count, true);
        match &self.units {
            ImputationUnits::Rows => unit_weights,
            ImputationUnits::Groups(groups) => {
                groups.iter().map(|&g| unit_weights[usize::try_from(g).unwrap_or(0)]).collect()
            }
        }
    }

    /// One chained-equation step: refit variable `j`'s conditional model on a Bayesian
    /// bootstrap of its observed rows and redraw its missing cells.
    fn chained_step(
        &self,
        j: usize,
        current: &mut [Vec<f64>],
        rng: &mut CausalRng,
    ) -> Result<(), MultipleImputationError> {
        let k = self.width(); // intercept, the p - 1 other variables, then every neighbour
        let weights = self.row_weights(rng);
        let design = self.design_row(j);
        let observed = self.fit_rows_of(j);
        let mut x = vec![0.0; k];
        let kind = self.kinds[j];
        match kind {
            VariableKind::Continuous => {
                let mut gram = vec![0.0; k * k];
                let mut cross = vec![0.0; k];
                for &i in observed {
                    design(i, &mut x, &*current);
                    let w = weights[i];
                    let y = self.values[j][i];
                    for a in 0..k {
                        cross[a] += w * x[a] * y;
                        for b in 0..=a {
                            gram[a * k + b] += w * x[a] * x[b];
                        }
                    }
                }
                symmetrize(&mut gram, k);
                let beta = solve_spd(&gram, &cross, k).ok_or_else(|| not_fitted(&self.names[j]))?;
                let mut rss = 0.0;
                let mut total_weight = 0.0;
                for &i in observed {
                    design(i, &mut x, &*current);
                    let r = self.values[j][i] - dot(&x, &beta);
                    rss += weights[i] * r * r;
                    total_weight += weights[i];
                }
                let dof = total_weight - count(k);
                let sigma2 = if dof > 0.0 { rss / dof } else { rss / total_weight.max(1.0) };
                let sigma = sigma2.max(0.0).sqrt();
                for &i in &self.missing_rows[j] {
                    design(i, &mut x, &*current);
                    let draw = dot(&x, &beta) + sigma * standard_normal(rng);
                    current[j][i] = draw;
                }
            }
            VariableKind::Binary => {
                let beta = self.logistic_fit(j, &weights, &*current, &design)?;
                for &i in &self.missing_rows[j] {
                    design(i, &mut x, &*current);
                    let prob = logistic(dot(&x, &beta));
                    current[j][i] = if rng.next_f64() < prob { 1.0 } else { 0.0 };
                }
            }
        }
        Ok(())
    }

    /// Weighted ridge-stabilized logistic IRLS of variable `j` on the other variables.
    fn logistic_fit(
        &self,
        j: usize,
        weights: &[f64],
        current: &[Vec<f64>],
        design: &DesignRow<'_>,
    ) -> Result<Vec<f64>, MultipleImputationError> {
        let k = self.width();
        let observed = self.fit_rows_of(j);
        let mut beta = vec![0.0; k];
        let mut x = vec![0.0; k];
        for _ in 0..LOGISTIC_ITERATIONS {
            let mut hessian = vec![0.0; k * k];
            let mut score = vec![0.0; k];
            for &i in observed {
                design(i, &mut x, current);
                let prob = logistic(dot(&x, &beta));
                let w = weights[i];
                let v = (prob * (1.0 - prob)).max(1e-12);
                let resid = self.values[j][i] - prob;
                for a in 0..k {
                    score[a] += w * x[a] * resid;
                    for b in 0..=a {
                        hessian[a * k + b] += w * v * x[a] * x[b];
                    }
                }
            }
            for a in 1..k {
                hessian[a * k + a] += LOGISTIC_RIDGE;
                score[a] -= LOGISTIC_RIDGE * beta[a];
            }
            symmetrize(&mut hessian, k);
            let step = solve_spd(&hessian, &score, k).ok_or_else(|| not_fitted(&self.names[j]))?;
            let mut largest: f64 = 0.0;
            for a in 0..k {
                beta[a] += step[a];
                largest = largest.max(step[a].abs());
            }
            if beta.iter().any(|b| !b.is_finite()) {
                return Err(not_fitted(&self.names[j]));
            }
            if largest < 1e-10 {
                break;
            }
        }
        Ok(beta)
    }

    /// One data-augmentation sweep of the joint normal model: draw the mean and covariance
    /// given the completed data (P-step), then every missing cell given its row's observed
    /// cells (I-step).
    fn augment(
        &self,
        current: &mut [Vec<f64>],
        rng: &mut CausalRng,
    ) -> Result<(), MultipleImputationError> {
        let p = self.names.len();
        let (mu, sigma) = match &self.units {
            ImputationUnits::Rows => self.jeffreys_draw(current, rng)?,
            ImputationUnits::Groups(_) => {
                let weights = self.row_weights(rng);
                weighted_moments(current, &weights, self.rows, p)
            }
        };
        // Missingness pattern of every row with a missing cell, in row order.
        let mut patterns: BTreeMap<Vec<bool>, PatternLaw> = BTreeMap::new();
        let mut rows: Vec<(usize, Vec<bool>)> = Vec::new();
        for i in 0..self.rows {
            let missing: Vec<bool> = (0..p).map(|j| self.values[j][i].is_nan()).collect();
            if missing.iter().any(|m| *m) {
                rows.push((i, missing));
            }
        }
        for (_, pattern) in &rows {
            if !patterns.contains_key(pattern) {
                let law = PatternLaw::new(pattern, &sigma, p)
                    .ok_or_else(|| not_fitted(&Arc::from("joint normal covariance")))?;
                patterns.insert(pattern.clone(), law);
            }
        }
        let mut z = vec![0.0; p];
        for (i, pattern) in &rows {
            let Some(law) = patterns.get(pattern) else { continue };
            let observed_dev: Vec<f64> =
                law.observed.iter().map(|&o| current[o][*i] - mu[o]).collect();
            let m = law.missing.len();
            for slot in z.iter_mut().take(m) {
                *slot = standard_normal(rng);
            }
            for (a, &target) in law.missing.iter().enumerate() {
                let mut value = mu[target];
                for (b, dev) in observed_dev.iter().enumerate() {
                    value += law.regression[a * law.observed.len() + b] * dev;
                }
                for (c, zc) in z.iter().enumerate().take(a + 1) {
                    value += law.chol[a * m + c] * zc;
                }
                current[target][*i] = value;
            }
        }
        Ok(())
    }

    /// `μ, Σ` from their posterior under the Jeffreys prior given the completed data:
    /// `Σ ~ InvWishart(n − 1, S)`, `μ | Σ ~ N(x̄, Σ / n)`.
    fn jeffreys_draw(
        &self,
        current: &[Vec<f64>],
        rng: &mut CausalRng,
    ) -> Result<(Vec<f64>, Vec<f64>), MultipleImputationError> {
        let p = self.names.len();
        let n = self.rows;
        let ones = vec![1.0; n];
        let (mean, cov) = weighted_moments(current, &ones, n, p);
        let n_f = count(n);
        // Scatter S = n · (biased covariance).
        let scatter: Vec<f64> = cov.iter().map(|v| v * n_f).collect();
        let failed = || not_fitted(&Arc::from("joint normal scatter"));
        let scatter_chol = cholesky(&scatter, p).ok_or_else(failed)?;
        let precision_scale = chol_inverse(&scatter_chol, p); // V = S⁻¹
        let l = cholesky(&precision_scale, p).ok_or_else(failed)?;
        // Bartlett: W = (L A)(L A)ᵀ ~ Wishart(ν, V), A lower triangular.
        let dof = n_f - 1.0;
        let mut a = vec![0.0; p * p];
        for r in 0..p {
            let chi2 = 2.0 * sample_gamma(0.5 * (dof - count(r)), 1.0, rng);
            a[r * p + r] = chi2.max(0.0).sqrt();
            for c in 0..r {
                a[r * p + c] = standard_normal(rng);
            }
        }
        let la = lower_product(&l, &a, p);
        let mut wishart = vec![0.0; p * p];
        for r in 0..p {
            for c in 0..p {
                wishart[r * p + c] = (0..p).map(|t| la[r * p + t] * la[c * p + t]).sum();
            }
        }
        let wishart_chol = cholesky(&wishart, p).ok_or_else(failed)?;
        let sigma = chol_inverse(&wishart_chol, p);
        let scaled: Vec<f64> = sigma.iter().map(|v| v / n_f).collect();
        let mean_chol = cholesky(&scaled, p).ok_or_else(failed)?;
        let z: Vec<f64> = (0..p).map(|_| standard_normal(rng)).collect();
        let mu = (0..p)
            .map(|r| mean[r] + (0..=r).map(|c| mean_chol[r * p + c] * z[c]).sum::<f64>())
            .collect();
        Ok((mu, sigma))
    }
}

/// The conditional law of one missingness pattern: regression of the missing block on the
/// observed block and the Cholesky factor of the conditional covariance.
struct PatternLaw {
    observed: Vec<usize>,
    missing: Vec<usize>,
    /// Row-major `missing x observed` coefficients `Σ_mo Σ_oo⁻¹`.
    regression: Vec<f64>,
    /// Lower Cholesky factor of `Σ_mm − Σ_mo Σ_oo⁻¹ Σ_om`, row-major `m x m`.
    chol: Vec<f64>,
}

impl PatternLaw {
    fn new(pattern: &[bool], sigma: &[f64], p: usize) -> Option<Self> {
        let observed: Vec<usize> = (0..p).filter(|&j| !pattern[j]).collect();
        let missing: Vec<usize> = (0..p).filter(|&j| pattern[j]).collect();
        let (o, m) = (observed.len(), missing.len());
        let mut regression = vec![0.0; m * o];
        let mut conditional = vec![0.0; m * m];
        for (a, &ma) in missing.iter().enumerate() {
            for (b, &mb) in missing.iter().enumerate() {
                conditional[a * m + b] = sigma[ma * p + mb];
            }
        }
        if o > 0 {
            let mut s_oo = vec![0.0; o * o];
            for (a, &oa) in observed.iter().enumerate() {
                for (b, &ob) in observed.iter().enumerate() {
                    s_oo[a * o + b] = sigma[oa * p + ob];
                }
            }
            let chol_oo = cholesky(&s_oo, o)?;
            let mut y = vec![0.0; o];
            let mut coef = vec![0.0; o];
            for (a, &ma) in missing.iter().enumerate() {
                let rhs: Vec<f64> = observed.iter().map(|&ob| sigma[ob * p + ma]).collect();
                chol_solve_into(&chol_oo, o, &rhs, &mut y, &mut coef);
                regression[a * o..(a + 1) * o].copy_from_slice(&coef);
            }
            for a in 0..m {
                for b in 0..m {
                    let mb = missing[b];
                    let reduction: f64 =
                        (0..o).map(|t| regression[a * o + t] * sigma[observed[t] * p + mb]).sum();
                    conditional[a * m + b] -= reduction;
                }
            }
            symmetrize_average(&mut conditional, m);
        }
        let chol = cholesky(&conditional, m)?;
        Some(Self { observed, missing, regression, chol })
    }
}

fn check_cancelled(ctx: &ExecutionContext) -> Result<(), MultipleImputationError> {
    if ctx.cancellation.is_cancelled() {
        return Err(MultipleImputationError::new(
            antecedent_core::reason_code!("cancelled_no_claim"),
            "multiple_imputation.cancelled",
            "cancelled before the completed tables were drawn; this is not a verdict on the data",
            None,
        ));
    }
    Ok(())
}

fn not_fitted(name: &Arc<str>) -> MultipleImputationError {
    MultipleImputationError::not_applicable(
        "multiple_imputation.conditional_model_not_fitted",
        format!(
            "the conditional model of `{name}` is singular on a bootstrap draw of the observed \
             rows; its predictors are collinear or a category is too rare"
        ),
        DECLARE_COMPLETE_CASE,
    )
}

/// Weighted mean and (biased) covariance of the completed columns, row-major `p x p`.
fn weighted_moments(
    current: &[Vec<f64>],
    weights: &[f64],
    rows: usize,
    p: usize,
) -> (Vec<f64>, Vec<f64>) {
    let total: f64 = weights.iter().take(rows).sum();
    let total = if total > 0.0 { total } else { 1.0 };
    let mean: Vec<f64> = (0..p)
        .map(|j| (0..rows).map(|i| weights[i] * current[j][i]).sum::<f64>() / total)
        .collect();
    let mut cov = vec![0.0; p * p];
    for a in 0..p {
        for b in 0..=a {
            let s: f64 = (0..rows)
                .map(|i| weights[i] * (current[a][i] - mean[a]) * (current[b][i] - mean[b]))
                .sum();
            cov[a * p + b] = s / total;
            cov[b * p + a] = s / total;
        }
    }
    (mean, cov)
}

/// Lower Cholesky factor, retrying once with a small relative jitter on the diagonal.
fn cholesky(a: &[f64], k: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0; k * k];
    if cholesky_spd_into(a, k, &mut l).is_ok() {
        return Some(l);
    }
    let scale = (0..k).map(|i| a[i * k + i].abs()).fold(0.0_f64, f64::max).max(f64::MIN_POSITIVE);
    let mut jittered = a.to_vec();
    for i in 0..k {
        jittered[i * k + i] += JITTER * scale;
    }
    cholesky_spd_into(&jittered, k, &mut l).ok().map(|()| l)
}

fn solve_spd(a: &[f64], b: &[f64], k: usize) -> Option<Vec<f64>> {
    let l = cholesky(a, k)?;
    let mut y = vec![0.0; k];
    let mut x = vec![0.0; k];
    chol_solve_into(&l, k, b, &mut y, &mut x);
    x.iter().all(|v| v.is_finite()).then_some(x)
}

/// `L A` for two lower-triangular row-major `p x p` matrices.
fn lower_product(l: &[f64], a: &[f64], p: usize) -> Vec<f64> {
    let mut out = vec![0.0; p * p];
    for r in 0..p {
        for c in 0..=r {
            out[r * p + c] = (c..=r).map(|t| l[r * p + t] * a[t * p + c]).sum();
        }
    }
    out
}

fn symmetrize(lower_filled: &mut [f64], k: usize) {
    for a in 0..k {
        for b in 0..a {
            lower_filled[b * k + a] = lower_filled[a * k + b];
        }
    }
}

fn symmetrize_average(m: &mut [f64], k: usize) {
    for a in 0..k {
        for b in 0..a {
            let avg = 0.5 * (m[a * k + b] + m[b * k + a]);
            m[a * k + b] = avg;
            m[b * k + a] = avg;
        }
    }
}

fn dot(x: &[f64], beta: &[f64]) -> f64 {
    x.iter().zip(beta).map(|(a, b)| a * b).sum()
}

fn logistic(eta: f64) -> f64 {
    if eta >= 0.0 {
        1.0 / (1.0 + (-eta).exp())
    } else {
        let e = eta.exp();
        e / (1.0 + e)
    }
}

fn uniform_index(rng: &mut CausalRng, n: usize) -> usize {
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        reason = "a uniform in [0, 1) scaled by a row count stays below the count"
    )]
    let index = (rng.next_f64() * n as f64) as usize;
    index.min(n.saturating_sub(1))
}

#[allow(clippy::cast_precision_loss, reason = "row and variable counts are far below 2^52")]
fn count(n: usize) -> f64 {
    n as f64
}
