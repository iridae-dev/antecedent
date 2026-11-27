//! Seeded hard-assignment regime learning for the linear, persistent regime PCMCI+ scope.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    Arc, DiscoveryDiagnostic, DiscoveryError, DiscoveryWorkspace, ExecutionContext,
    RegimeAssignment, RegimeId, RegimeModel, Rpcmci, RpcmciDiscoveryResult, TableView,
    TimeSeriesData, VariableId, fit_regime_models, retained_regime_parents,
};
use antecedent_core::execution::StreamDomain;

const MAX_COUNT: u32 = 8;
const MAX_STARTS: u32 = 32;
const MAX_ITERATIONS: u32 = 64;
const INITIALIZATION: StreamDomain = StreamDomain::external(0x5250_434d_4349);

/// Number of regimes to fit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RegimeCountPolicy {
    /// Keep this number of independently occupied regimes.
    Fixed(u32),
    /// Compare eligible counts from one to this maximum using classification BIC.
    Bic {
        /// Largest candidate count (inclusive).
        max_regimes: u32,
    },
}

impl RegimeCountPolicy {
    /// Inclusive candidate count bounds.
    #[must_use]
    pub const fn bounds(self) -> (u32, u32) {
        match self {
            Self::Fixed(k) => (k, k),
            Self::Bic { max_regimes } => (1, max_regimes),
        }
    }
}

/// Linear assignment model and reproducible search controls.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RpcmciLearning {
    /// Fixed count or classification-BIC search over eligible candidates.
    pub count: RegimeCountPolicy,
    /// Initial assignments attempted per candidate.
    pub starts: u32,
    /// Maximum re-estimation updates per start.
    pub max_iterations: u32,
    /// Standardized prediction error cost for a change of label.
    pub switch_penalty: f64,
}

impl RpcmciLearning {
    /// Default search controls for the chosen regime-count policy.
    #[must_use]
    pub const fn new(count: RegimeCountPolicy) -> Self {
        Self { count, starts: 6, max_iterations: 12, switch_penalty: 12.0 }
    }

    /// Validate the declared search bounds.
    ///
    /// # Errors
    ///
    /// Invalid or out-of-range count, start, iteration or penalty.
    pub fn validate(self) -> Result<(), DiscoveryError> {
        let (min, max) = self.count.bounds();
        if !(1..=MAX_COUNT).contains(&min)
            || !(1..=MAX_COUNT).contains(&max)
            || !(1..=MAX_STARTS).contains(&self.starts)
            || !(1..=MAX_ITERATIONS).contains(&self.max_iterations)
            || !self.switch_penalty.is_finite()
            || self.switch_penalty < 0.0
        {
            return Err(DiscoveryError::unsupported(
                "RPCMCI learning needs 1..=8 regimes, 1..=32 starts, 1..=64 iterations and a finite nonnegative switch penalty",
            ));
        }
        Ok(())
    }
}

/// Retained diagnostics from learning regimes on the same series as their graphs.
#[derive(Clone, Debug, PartialEq)]
pub struct RpcmciLearningDiagnostics {
    /// Search controls used by the estimator.
    pub options: RpcmciLearning,
    /// Master seed from the execution context.
    pub seed: u64,
    /// The selected count.
    pub selected_count: u32,
    /// Classification BIC for the best penalized-prediction fit at each feasible count.
    pub candidate_scores: Vec<(u32, f64)>,
    /// Counts that lacked admissible fits, including counts too large for the series.
    pub infeasible_counts: Vec<u32>,
    /// Stops by unchanged reassignments or no admissible improving update, as opposed
    /// to reaching the iteration bound.
    pub converged: bool,
    /// Accepted alternating updates for the selected fit.
    pub iterations: u32,
    /// Number of successful fits for the selected count.
    pub successful_starts: usize,
    /// Mean fraction of labels agreeing with the selected fit after optimal permutation,
    /// across successful alternative starts; absent when no alternative start succeeded.
    pub assignment_agreement: Option<f64>,
    /// Exact CPDAG agreement per selected regime across the same aligned alternative starts.
    pub graph_agreement: Vec<Option<f64>>,
}

struct Candidate {
    result: RpcmciDiscoveryResult,
    residual_cost: f64,
    bic: f64,
    iterations: u32,
    converged: bool,
}

struct Series<'a> {
    data: &'a TimeSeriesData,
    variables: &'a [VariableId],
    columns: Vec<Vec<f64>>,
    scale: Vec<f64>,
    lag: usize,
    dwell: usize,
}

impl Rpcmci {
    /// Discover regimes without labels, alternating PCMCI+ graph estimation and
    /// minimum-duration, switch-penalized hard assignment. Assumes fully observed finite
    /// series and regime-specific linear lagged prediction equations. Contemporaneous
    /// structure is reported by PCMCI+ and does not enter assignment prediction.
    ///
    /// For `Bic`, compare the profiled per-variable Gaussian residual likelihood, counting
    /// an intercept and retained lagged coefficients per variable and regime, plus one
    /// common residual variance per variable, regime proportions and switching locations. This is a classification criterion
    /// on fitted hard labels, not a calibrated test for the presence of a regime.
    ///
    /// # Errors
    ///
    /// Invalid options, insufficient observations, non-finite input, no admissible fit,
    /// cancellation or discovery failure.
    pub fn learn(
        &self,
        data: &TimeSeriesData,
        variables: &[VariableId],
        options: RpcmciLearning,
        workspace: &mut DiscoveryWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<RpcmciDiscoveryResult, DiscoveryError> {
        options.validate()?;
        self.pcmci_plus.engine().constraints.validate_temporal_ci_options()?;
        let lag = self.pcmci_plus.engine().constraints.temporal.max_lag.raw() as usize;
        let dwell = self.min_regime_len.max(lag.saturating_mul(2).saturating_add(8));
        let (_, max_count) = options.count.bounds();
        let rows = data.row_count() as u64;
        let width = variables.len() as u64;
        let regimes = u64::from(max_count);
        let starts = u64::from(options.starts).saturating_mul(2).saturating_add(2);
        let nodes = width.saturating_mul(lag.saturating_add(1) as u64);
        let duration = rows.saturating_add(1).saturating_mul(regimes).saturating_mul(32);
        let columns = rows.saturating_mul(width).saturating_mul(8);
        let candidates = starts.saturating_mul(rows.saturating_mul(8).saturating_add(
            regimes.saturating_mul(nodes).saturating_mul(nodes).saturating_mul(256),
        ));
        let _learning_memory = ctx
            .acquire_admission(0, duration.saturating_add(columns).saturating_add(candidates))
            .map_err(|error| {
                if ctx.cancellation.is_cancelled() {
                    DiscoveryError::Cancelled
                } else {
                    DiscoveryError::Resource(error.to_string())
                }
            })?;
        let cols = variables
            .iter()
            .map(|&v| data.float64_values(v).map_err(DiscoveryError::from))
            .collect::<Result<Vec<_>, _>>()?;
        if cols.is_empty() || cols.iter().flatten().any(|v| !v.is_finite()) {
            return Err(DiscoveryError::data_msg(
                "RPCMCI learning requires a nonempty set of complete finite series",
            ));
        }
        let scale = cols.iter().map(|c| variance(&c[lag.min(c.len())..])).collect();
        let series = Series { data, variables, columns: cols, scale, lag, dwell };
        let (min_count, max_count) = options.count.bounds();
        let mut winner: Option<(u32, Vec<Candidate>)> = None;
        let mut scores = Vec::new();
        let mut infeasible = Vec::new();
        for k in min_count..=max_count {
            if ctx.cancellation.is_cancelled() {
                return Err(DiscoveryError::Cancelled);
            }
            if data.row_count() / (k as usize) < dwell {
                infeasible.push(k);
                continue;
            }
            let attempts = if k == 1 { 1 } else { options.starts };
            let mut fits = Vec::new();
            for start in 0..attempts {
                let assignment = initialize(data.row_count(), dwell, k, start, ctx)?;
                if let Some(fit) =
                    self.fit_start(&series, &assignment, k, options, workspace, ctx)?
                {
                    fits.push(fit);
                }
            }
            fits.sort_by(|a, b| a.residual_cost.total_cmp(&b.residual_cost));
            let Some(best) = fits.first() else {
                infeasible.push(k);
                continue;
            };
            scores.push((k, best.bic));
            if winner.as_ref().is_none_or(|(_, current)| best.bic < current[0].bic) {
                winner = Some((k, fits));
            }
        }
        let (k, mut fits) = winner.ok_or_else(|| {
            DiscoveryError::unsupported("RPCMCI found no admissible fit for the requested regime count and minimum duration")
        })?;
        let mut best = fits.remove(0);
        let agreement = stability(&best.result, &fits, k);
        best.result.learning = Some(RpcmciLearningDiagnostics {
            options,
            seed: ctx.rng.master_seed(),
            selected_count: k,
            candidate_scores: scores,
            infeasible_counts: infeasible,
            converged: best.converged,
            iterations: best.iterations,
            successful_starts: fits.len() + 1,
            assignment_agreement: agreement.0,
            graph_agreement: agreement.1,
        });
        best.result.diagnostics.push(DiscoveryDiagnostic {
            code: Arc::from("rpcmci.post_selection"),
            message: Arc::from(
                "assignments and graphs use the same series; per-regime p-values are conditional on data-driven labels and graph agreement only measures sensitivity to initialization",
            ),
        });
        if !best.converged {
            best.result.diagnostics.push(DiscoveryDiagnostic {
                code: Arc::from("rpcmci.iteration_limit"),
                message: Arc::from("regime assignment reached the iteration bound"),
            });
        }
        Ok(best.result)
    }

    fn fit_start(
        &self,
        series: &Series<'_>,
        assignment: &RegimeAssignment,
        k: u32,
        options: RpcmciLearning,
        workspace: &mut DiscoveryWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<Option<Candidate>, DiscoveryError> {
        let Some((mut candidate, mut models)) =
            self.learned_candidate(series, assignment, k, options, workspace, ctx)?
        else {
            return Ok(None);
        };
        if k == 1 {
            candidate.converged = true;
            return Ok(Some(candidate));
        }
        for _ in 0..options.max_iterations {
            if ctx.cancellation.is_cancelled() {
                return Err(DiscoveryError::Cancelled);
            }
            let next = duration_path(series, &models, options.switch_penalty, ctx)?;
            if next == candidate.result.assignments || next.unique_regimes().len() != k as usize {
                candidate.converged = true;
                break;
            }
            let Some((updated, updated_models)) =
                self.learned_candidate(series, &next, k, options, workspace, ctx)?
            else {
                candidate.converged = true;
                break;
            };
            let tolerance = 1e-10 * candidate.residual_cost.abs().max(1.0);
            if candidate.residual_cost - updated.residual_cost <= tolerance {
                candidate.converged = true;
                break;
            }
            candidate = Candidate { iterations: candidate.iterations + 1, ..updated };
            models = updated_models;
        }
        Ok(Some(candidate))
    }

    fn learned_candidate(
        &self,
        series: &Series<'_>,
        assignment: &RegimeAssignment,
        k: u32,
        options: RpcmciLearning,
        workspace: &mut DiscoveryWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<Option<(Candidate, Vec<RegimeModel>)>, DiscoveryError> {
        let result =
            self.discover_regimes(series.data, series.variables, assignment, workspace, ctx)?;
        if result.graphs.graphs.len() != k as usize {
            return Ok(None);
        }
        let parents = retained_regime_parents(&result.graphs, series.variables);
        let models = fit_regime_models(
            &parents,
            &series.columns,
            series.lag,
            series.data.row_count(),
            assignment,
            series.lag,
        );
        if models.len() != k as usize {
            return Ok(None);
        }
        let (residual_cost, bic) = criteria(series, &models, assignment, options.switch_penalty);
        Ok((residual_cost.is_finite() && bic.is_finite()).then_some((
            Candidate { result, residual_cost, bic, iterations: 0, converged: false },
            models,
        )))
    }
}

fn variance(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 1.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let var = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
    if var.is_finite() && var > 0.0 { var } else { 1.0 }
}

/// Balanced initial segments with reproducible jitter, plus recurrent-label starts when
/// enough rows exist. The first start is the balanced contiguous partition.
fn initialize(
    rows: usize,
    dwell: usize,
    k: u32,
    start: u32,
    ctx: &ExecutionContext,
) -> Result<RegimeAssignment, DiscoveryError> {
    let mut rng = ctx.rng.stream_for(INITIALIZATION, (u64::from(k) << 32) | u64::from(start));
    let k = k as usize;
    let runs = if start >= 3 && rows / dwell >= k * 2 { k * 2 } else { k };
    let slack = rows - dwell * runs;
    let mut labels = Vec::with_capacity(rows);
    let mut used = 0;
    for run in 0..runs {
        let balanced = slack * (run + 1) / runs;
        let jitter = if start == 0 || run + 1 == runs {
            0
        } else {
            let width = slack / runs + 1;
            usize::try_from(rng.next_u64() % width as u64)
                .expect("the remainder is below a usize width")
        };
        let next = (balanced.saturating_sub(slack / (2 * runs)) + jitter).clamp(used, slack);
        let next = if start == 0 || run + 1 == runs { balanced } else { next };
        let rotation = usize::from(run >= k && start.is_multiple_of(2));
        let label = RegimeId::from_raw(crate::indexing::dense_u32((run + rotation) % k));
        labels.extend(std::iter::repeat_n(label, dwell + next - used));
        used = next;
    }
    canonical_assignment(labels)
}

fn canonical_assignment(mut labels: Vec<RegimeId>) -> Result<RegimeAssignment, DiscoveryError> {
    let mut encounter = Vec::new();
    for label in &mut labels {
        let index = if let Some(index) = encounter.iter().position(|id| id == label) {
            index
        } else {
            encounter.push(*label);
            encounter.len() - 1
        };
        *label = RegimeId::from_raw(crate::indexing::dense_u32(index));
    }
    RegimeAssignment::try_new(labels)
}

fn residuals<'a>(
    series: &'a Series<'_>,
    model: &'a RegimeModel,
    t: usize,
) -> impl Iterator<Item = f64> + 'a {
    model.equations.iter().enumerate().map(move |(target, equation)| {
        let mut fitted = equation.coefficients[0];
        for (&(source, lag), beta) in equation.parents.iter().zip(&equation.coefficients[1..]) {
            fitted += beta * series.columns[source][t - lag];
        }
        (series.columns[target][t] - fitted).powi(2) / series.scale[target]
    })
}

fn criteria(
    series: &Series<'_>,
    models: &[RegimeModel],
    assignment: &RegimeAssignment,
    switch_penalty: f64,
) -> (f64, f64) {
    let mut sums = vec![0.0; series.columns.len()];
    for t in series.lag..series.data.row_count() {
        let model = &models[assignment.regimes[t].raw() as usize];
        for (sum, residual) in sums.iter_mut().zip(residuals(series, model, t)) {
            *sum += residual;
        }
    }
    let switches = assignment.regimes.windows(2).filter(|p| p[0] != p[1]).count();
    let parameters = switches + models.len() - 1
        + models.iter().flat_map(|m| &m.equations).map(|e| e.coefficients.len()).sum::<usize>()
        + series.columns.len();
    let n = (series.data.row_count() - series.lag) as f64;
    let bic = n * sums.iter().map(|ss| (ss / n).max(1e-12).ln()).sum::<f64>()
        + parameters as f64 * n.ln();
    (sums.iter().sum::<f64>() + switch_penalty * switches as f64, bic)
}

/// Viterbi over mature runs. A new label must spend `dwell` consecutive rows before
/// another switch is possible. The prefix predictions before the maximum lag have zero cost.
fn duration_path(
    series: &Series<'_>,
    models: &[RegimeModel],
    penalty: f64,
    ctx: &ExecutionContext,
) -> Result<RegimeAssignment, DiscoveryError> {
    let labels = duration_indices(
        series.data.row_count(),
        models.len(),
        series.dwell,
        penalty,
        ctx,
        |t, r| {
            if t < series.lag { 0.0 } else { residuals(series, &models[r], t).sum() }
        },
    )?;
    canonical_assignment(labels.into_iter().map(|label| models[label].regime).collect())
}

fn duration_indices(
    rows: usize,
    k: usize,
    dwell: usize,
    penalty: f64,
    ctx: &ExecutionContext,
    cost: impl Fn(usize, usize) -> f64,
) -> Result<Vec<usize>, DiscoveryError> {
    let mut prefix = vec![0.0; (rows + 1) * k];
    let mut value = vec![f64::INFINITY; (rows + 1) * k];
    let mut back = vec![(0, 0); (rows + 1) * k];
    for end in 1..=rows {
        if end % 256 == 0 && ctx.cancellation.is_cancelled() {
            return Err(DiscoveryError::Cancelled);
        }
        for r in 0..k {
            let at = end * k + r;
            let t = end - 1;
            let err = cost(t, r);
            prefix[at] = prefix[at - k] + err;
            if end < dwell {
                continue;
            }
            if end == dwell {
                value[at] = prefix[at];
                continue;
            }
            value[at] = value[at - k] + err;
            back[at] = (end - 1, r);
            let split = end - dwell;
            let (prev, prev_cost) = (0..k)
                .filter(|&s| s != r)
                .map(|s| (s, value[split * k + s]))
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .expect("duration search only runs for multiple regimes");
            let switched = prev_cost + penalty + prefix[at] - prefix[split * k + r];
            if switched < value[at] {
                value[at] = switched;
                back[at] = (split, prev);
            }
        }
    }
    let mut end = rows;
    let mut r = (0..k)
        .min_by(|&a, &b| value[rows * k + a].total_cmp(&value[rows * k + b]))
        .expect("multiple regimes");
    let mut labels = vec![0; rows];
    while end > 0 {
        let (begin, prev) = back[end * k + r];
        labels[begin..end].fill(r);
        end = begin;
        r = prev;
    }
    Ok(labels)
}

fn stability(
    best: &RpcmciDiscoveryResult,
    others: &[Candidate],
    count: u32,
) -> (Option<f64>, Vec<Option<f64>>) {
    let k = count as usize;
    if others.is_empty() {
        return (None, vec![None; k]);
    }
    let mut assignments = 0;
    let mut graphs = vec![0; k];
    for candidate in others {
        let other = &candidate.result;
        let mut overlaps = vec![vec![0; k]; k];
        for (a, b) in best.assignments.regimes.iter().zip(other.assignments.regimes.iter()) {
            overlaps[a.raw() as usize][b.raw() as usize] += 1;
        }
        let mut values = vec![0; 1 << k];
        let mut choices = vec![0; 1 << k];
        for used in (0..(1_usize << k) - 1).rev() {
            let row = used.count_ones() as usize;
            for r in (0..k).filter(|r| used & (1 << r) == 0) {
                let agree = overlaps[row][r] + values[used | (1 << r)];
                if agree >= values[used] {
                    values[used] = agree;
                    choices[used] = r;
                }
            }
        }
        assignments += values[0];
        let mut used = 0;
        for (index, agree) in graphs.iter_mut().enumerate() {
            let matched = choices[used];
            used |= 1 << matched;
            if best.graphs.graphs[index].1 == other.graphs.graphs[matched].1 {
                *agree += 1;
            }
        }
    }
    let n = others.len() as f64;
    (
        Some(f64::from(assignments) / (n * best.assignments.len() as f64)),
        graphs.into_iter().map(|m| Some(f64::from(m) / n)).collect(),
    )
}

#[cfg(test)]
mod duration_oracle {
    use super::duration_indices;
    use antecedent_core::ExecutionContext;

    fn admissible(path: &[usize], dwell: usize) -> bool {
        let mut start = 0;
        for end in 1..=path.len() {
            if end == path.len() || path[end] != path[start] {
                if end - start < dwell {
                    return false;
                }
                start = end;
            }
        }
        true
    }

    #[test]
    #[allow(clippy::float_cmp, reason = "the oracle evaluates the same objective on the same path")]
    fn duration_updates_match_independent_exhaustive_path_cost() {
        let ctx = ExecutionContext::for_tests(14);
        for rows in 4..=7 {
            for regimes in 2_usize..=3 {
                for dwell in 1..=3 {
                    for salt in 0..13 {
                        let cost = |t: usize, r: usize| {
                            (((t * 13 + r * 7 + 11 * salt) ^ (r * t + salt)) % 19) as f64
                        };
                        for penalty in [0.0, 1.5, 9.0] {
                            let objective = |labels: &[usize]| {
                                labels.iter().enumerate().map(|(t, &r)| cost(t, r)).sum::<f64>()
                                    + penalty
                                        * labels.windows(2).filter(|p| p[0] != p[1]).count() as f64
                            };
                            let live = duration_indices(rows, regimes, dwell, penalty, &ctx, cost)
                                .unwrap();
                            assert!(admissible(&live, dwell));
                            let mut oracle = f64::INFINITY;
                            for mut code in 0..regimes.pow(rows as u32) {
                                let mut path = vec![0; rows];
                                for label in &mut path {
                                    *label = code % regimes;
                                    code /= regimes;
                                }
                                if admissible(&path, dwell) {
                                    oracle = oracle.min(objective(&path));
                                }
                            }
                            assert_eq!(
                                objective(&live),
                                oracle,
                                "{rows} {regimes} {dwell} {salt} {penalty}"
                            );
                        }
                    }
                }
            }
        }
    }
}
