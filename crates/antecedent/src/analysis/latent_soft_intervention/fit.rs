//! Baseline-anchored Gaussian shift likelihood and outcome-independent comparison.

use antecedent_core::search::{SearchBudget, SearchLimits, SearchStop};
use antecedent_core::{CausalRng, ExecutionContext, SemanticDigest};
use antecedent_learn::crossfit_plan::row_fold_ids_u16;
use antecedent_stats::gram::{accumulate_xtx_xty_row, chol_solve, cholesky_spd, solve_spd};

use super::{LatentSoftInterventionRequest, LatentTargetHypothesis, cancelled, invalid};
use crate::error::CausalError;

const RIDGE: f64 = 0.01;
const ITERATIONS: usize = 80;
const MIN_WEIGHT: f64 = 0.05;

#[derive(Clone)]
pub(super) struct Equation {
    pub(super) parents: Vec<usize>,
    pub(super) beta: Vec<f64>,
    pub(super) sigma: f64,
}

impl Equation {
    fn residual(&self, row: &[f64], node: usize) -> f64 {
        row[node]
            - self.beta[0]
            - self
                .parents
                .iter()
                .zip(&self.beta[1..])
                .map(|(&parent, beta)| row[parent] * beta)
                .sum::<f64>()
    }
}

pub(super) struct Context {
    center: Vec<f64>,
    scale: Vec<f64>,
    lo: Vec<f64>,
    hi: Vec<f64>,
    action_region: Option<SupportRegion>,
    control_region: Option<SupportRegion>,
}

struct SupportRegion {
    center: Vec<f64>,
    cholesky: Vec<f64>,
    radius_squared: f64,
}

impl SupportRegion {
    fn distance(&self, coordinates: &[f64]) -> Option<f64> {
        let delta: Vec<_> =
            coordinates.iter().zip(&self.center).map(|(x, mean)| x - mean).collect();
        let score = dot(&delta, &chol_solve(&self.cholesky, delta.len(), &delta)?);
        score.is_finite().then_some(score)
    }

    fn fit(context: &Context, names: &[usize], rows: &[&[f64]]) -> Option<Self> {
        let d = names.len();
        let designs: Vec<_> =
            rows.iter().map(|row| context.design(names.iter().map(|&i| row[i]))).collect();
        let n = rows.len() as f64;
        let center: Vec<_> =
            (0..d).map(|i| designs.iter().map(|x| x[i + 1]).sum::<f64>() / n).collect();
        let mut covariance = vec![0.0; d * d];
        for design in &designs {
            for i in 0..d {
                for j in 0..d {
                    covariance[i * d + j] +=
                        (design[i + 1] - center[i]) * (design[j + 1] - center[j]) / n;
                }
            }
        }
        let cholesky = cholesky_spd(&covariance, d)?;
        let mut region = Self { center, cholesky, radius_squared: 0.0 };
        for design in &designs {
            region.radius_squared = region.radius_squared.max(region.distance(&design[1..])?);
        }
        Some(region)
    }
}

impl Context {
    fn fit(request: &LatentSoftInterventionRequest, rows: &[&[f64]]) -> Result<Self, CausalError> {
        let mut context = Self {
            center: Vec::new(),
            scale: Vec::new(),
            lo: Vec::new(),
            hi: Vec::new(),
            action_region: None,
            control_region: None,
        };
        for &node in &request.context {
            let mean = rows.iter().map(|r| r[node]).sum::<f64>() / rows.len() as f64;
            let scale = (rows.iter().map(|r| (r[node] - mean).powi(2)).sum::<f64>()
                / rows.len() as f64)
                .sqrt();
            if !mean.is_finite() || !scale.is_finite() || scale <= 0.0 {
                return Err(invalid("a context coordinate has no estimable scale"));
            }
            context.center.push(mean);
            context.scale.push(scale);
            context.lo.push(rows.iter().map(|r| r[node]).fold(f64::INFINITY, f64::min));
            context.hi.push(rows.iter().map(|r| r[node]).fold(f64::NEG_INFINITY, f64::max));
        }
        Ok(context)
    }

    fn design(&self, values: impl IntoIterator<Item = f64>) -> Vec<f64> {
        let mut design = vec![1.0];
        design.extend(
            values
                .into_iter()
                .zip(self.center.iter().zip(&self.scale))
                .map(|(value, (center, scale))| (value - center) / scale),
        );
        design
    }

    pub(super) fn forecast_design(&self, values: &[f64]) -> Result<Vec<f64>, CausalError> {
        if values.len() != self.center.len()
            || values
                .iter()
                .zip(self.lo.iter().zip(&self.hi))
                .any(|(&value, (&lo, &hi))| !value.is_finite() || value < lo || value > hi)
        {
            return Err(invalid("forecast context is outside marginal training support"));
        }
        let design = self.design(values.iter().copied());
        for region in [&self.action_region, &self.control_region].into_iter().flatten() {
            if region.distance(&design[1..]).is_none_or(|distance| distance > region.radius_squared)
            {
                return Err(invalid("forecast context is outside joint training support"));
            }
        }
        Ok(design)
    }
}

#[derive(Clone)]
pub(super) struct Mixture {
    pub(super) shifts: Vec<Vec<f64>>,
    gate: Vec<f64>,
}

impl Mixture {
    pub(super) fn weights(&self, design: &[f64]) -> Vec<f64> {
        if self.shifts.len() == 1 {
            return vec![1.0];
        }
        let p = logistic(dot(design, &self.gate));
        vec![1.0 - p, p]
    }

    fn log_terms(&self, residual: &[f64], design: &[f64]) -> [f64; 2] {
        let mut terms = [f64::NEG_INFINITY; 2];
        let probability =
            if self.shifts.len() == 1 { 0.0 } else { logistic(dot(design, &self.gate)) };
        for (k, (shift, weight)) in
            self.shifts.iter().zip([1.0 - probability, probability]).enumerate()
        {
            terms[k] = weight.ln()
                - 0.5
                    * residual
                        .iter()
                        .zip(shift)
                        .map(|(observed, mean)| (observed - mean).powi(2))
                        .sum::<f64>();
        }
        terms
    }
}

pub(super) struct Fit {
    pub(super) equations: Vec<Equation>,
    pub(super) context: Context,
    pub(super) mixture: Mixture,
    pub(super) log_evidence: f64,
    pub(super) targets_resolved: bool,
    pub(super) hypotheses: Vec<LatentTargetHypothesis>,
    pub(super) forecast_admissible: bool,
    pub(super) training_rows: usize,
    pub(super) evaluation_rows: usize,
}

struct Rows {
    residuals: Vec<Vec<f64>>,
    design: Vec<Vec<f64>>,
}

fn search_refusal(stop: SearchStop) -> CausalError {
    if stop == SearchStop::Cancelled {
        return CausalError::Cancelled { stage: "latent soft intervention mixture search" };
    }
    CausalError::Resource { message: format!("latent soft intervention: {}", stop.code()) }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn logistic(value: f64) -> f64 {
    (1.0 / (1.0 + (-value).exp())).clamp(1e-12, 1.0 - 1e-12)
}

fn log_sum_exp([a, b]: [f64; 2]) -> f64 {
    let top = a.max(b);
    top + ((a - top).exp() + (b - top).exp()).ln()
}

fn regress(design: &[Vec<f64>], outcome: impl IntoIterator<Item = f64>) -> Option<Vec<f64>> {
    let d = design.first()?.len();
    let mut gram = vec![0.0; d * d];
    let mut rhs = vec![0.0; d];
    for (x, y) in design.iter().zip(outcome) {
        accumulate_xtx_xty_row(x, y, &mut gram, &mut rhs);
    }
    let answer = solve_spd(&gram, d, &rhs)?;
    answer.iter().all(|value| value.is_finite()).then_some(answer)
}

fn fit_equations(
    request: &LatentSoftInterventionRequest,
    rows: &[&[f64]],
    ctx: &ExecutionContext,
) -> Result<Vec<Equation>, CausalError> {
    (0..request.variables.len())
        .map(|node| {
            cancelled(ctx, "latent soft intervention baseline fit")?;
            let parents: Vec<_> = request
                .edges
                .iter()
                .filter_map(|&(parent, child)| (child == node).then_some(parent))
                .collect();
            let design: Vec<Vec<f64>> = rows
                .iter()
                .map(|row| std::iter::once(1.0).chain(parents.iter().map(|&p| row[p])).collect())
                .collect();
            let beta = regress(&design, rows.iter().map(|row| row[node]))
                .ok_or_else(|| invalid("baseline equation has a singular design"))?;
            let sigma = (rows
                .iter()
                .zip(&design)
                .map(|(row, design)| (row[node] - dot(&beta, design)).powi(2))
                .sum::<f64>()
                / rows.len() as f64)
                .sqrt();
            if !sigma.is_finite() || sigma <= 0.0 {
                return Err(invalid("baseline equation has no positive finite residual variance"));
            }
            Ok(Equation { parents, beta, sigma })
        })
        .collect()
}

fn arm_split(rows: &[Vec<f64>], seed: u64) -> (Vec<&[f64]>, Vec<&[f64]>) {
    let ids = row_fold_ids_u16(&vec![0; rows.len()], 3, seed);
    let (eval, train): (Vec<_>, Vec<_>) = rows.iter().zip(ids).partition(|(_, id)| *id == 0);
    (
        train.into_iter().map(|(row, _)| row.as_slice()).collect(),
        eval.into_iter().map(|(row, _)| row.as_slice()).collect(),
    )
}

fn action_rows(
    equations: &[Equation],
    context: &Context,
    request: &LatentSoftInterventionRequest,
    rows: &[&[f64]],
) -> Rows {
    let residuals = rows
        .iter()
        .map(|row| {
            equations
                .iter()
                .enumerate()
                .filter(|(i, _)| !request.context.contains(i))
                .map(|(i, eq)| eq.residual(row, i) / eq.sigma)
                .collect()
        })
        .collect();
    let design =
        rows.iter().map(|row| context.design(request.context.iter().map(|&i| row[i]))).collect();
    Rows { residuals, design }
}

fn cross_entropy(design: &[Vec<f64>], labels: &[f64], beta: &[f64]) -> f64 {
    design
        .iter()
        .zip(labels)
        .map(|(x, y)| {
            let p = logistic(dot(x, beta));
            y * p.ln() + (1.0 - y) * (-p).ln_1p()
        })
        .sum::<f64>()
        - 0.5 * RIDGE * dot(beta, beta)
}

fn update_gate(design: &[Vec<f64>], labels: &[f64], gate: &mut Vec<f64>) -> bool {
    let d = gate.len();
    for _ in 0..12 {
        let mut curvature = vec![0.0; d * d];
        let mut scores: Vec<_> = gate.iter().map(|b| -RIDGE * b).collect();
        for (x, y) in design.iter().zip(labels) {
            let p = logistic(dot(x, gate));
            for i in 0..d {
                scores[i] += x[i] * (y - p);
                for j in 0..d {
                    curvature[i * d + j] += x[i] * x[j] * p * (1.0 - p);
                }
            }
        }
        for i in 0..d {
            curvature[i * d + i] += RIDGE;
        }
        let Some(step) = solve_spd(&curvature, d, &scores) else { return false };
        if step.iter().all(|v| v.abs() < 1e-7) {
            return true;
        }
        let before = cross_entropy(design, labels, gate);
        let mut accepted = false;
        for shrink in 0..12 {
            let scale = 0.5_f64.powi(shrink);
            let proposal: Vec<_> =
                gate.iter().zip(&step).map(|(b, step)| b + scale * step).collect();
            if cross_entropy(design, labels, &proposal) >= before {
                *gate = proposal;
                accepted = true;
                break;
            }
        }
        if !accepted {
            break;
        }
    }
    true
}

fn weighted_shift(rows: &Rows, weights: &[f64], mask: &[usize], mass: f64) -> Vec<f64> {
    let mut shifts = vec![0.0; rows.residuals[0].len()];
    for &node in mask {
        shifts[node] =
            rows.residuals.iter().zip(weights).map(|(row, weight)| row[node] * weight).sum::<f64>()
                / mass;
    }
    shifts
}

fn fit_pair(
    rows: &Rows,
    masks: [&[usize]; 2],
    seed: usize,
    budget: &mut SearchBudget<'_>,
) -> Result<Option<(f64, Mixture)>, CausalError> {
    let n = rows.residuals.len();
    let mut a = &rows.residuals[seed % n];
    let mut b = &rows.residuals[(seed + n / 2) % n];
    // Choose distant starting rows in the candidate subspace.
    let coordinates: Vec<_> = masks.iter().flat_map(|mask| mask.iter().copied()).collect();
    let distance = |left: &[f64], right: &[f64]| {
        coordinates.iter().map(|&i| (left[i] - right[i]).powi(2)).sum::<f64>()
    };
    for row in &rows.residuals {
        if distance(row, a) > distance(b, a) {
            b = row;
        }
    }
    if (seed % 2) == 1 {
        std::mem::swap(&mut a, &mut b);
    }
    let project = |row: &[f64], mask: &[usize]| {
        row.iter()
            .enumerate()
            .map(|(i, &v)| if mask.contains(&i) { v } else { 0.0 })
            .collect::<Vec<_>>()
    };
    let mut fit = Mixture {
        shifts: vec![project(a, masks[0]), project(b, masks[1])],
        gate: vec![0.0; rows.design[0].len()],
    };
    let mut previous = f64::NEG_INFINITY;
    let mut labels = vec![0.0; n];
    let mut complement = vec![0.0; n];
    for _ in 0..ITERATIONS {
        budget.charge(2, 0).map_err(search_refusal)?;
        let mut likelihood = 0.0;
        for (i, (row, design)) in rows.residuals.iter().zip(&rows.design).enumerate() {
            let terms = fit.log_terms(row, design);
            let normalizer = log_sum_exp(terms);
            likelihood += normalizer;
            labels[i] = (terms[1] - normalizer).exp();
            complement[i] = 1.0 - labels[i];
        }
        let mass: f64 = labels.iter().sum();
        if !likelihood.is_finite() || mass.min(n as f64 - mass) < MIN_WEIGHT * n as f64 {
            return Ok(None);
        }
        likelihood -= 0.5 * RIDGE * dot(&fit.gate, &fit.gate);
        if likelihood < previous - 1e-8 * (1.0 + previous.abs()) {
            return Ok(None);
        }
        if (likelihood - previous).abs() <= 1e-7 * (1.0 + likelihood.abs()) {
            return Ok(Some((likelihood, fit)));
        }
        previous = likelihood;
        fit.shifts[0] = weighted_shift(rows, &complement, masks[0], n as f64 - mass);
        fit.shifts[1] = weighted_shift(rows, &labels, masks[1], mass);
        if !update_gate(&rows.design, &labels, &mut fit.gate) {
            return Ok(None);
        }
    }
    Ok(None)
}

fn score(fit: &Mixture, rows: &Rows) -> f64 {
    rows.residuals.iter().zip(&rows.design).map(|(row, x)| log_sum_exp(fit.log_terms(row, x))).sum()
}

// Evaluate the fitted joint density by successive conditional probability integral
// transforms. Later coordinates update the membership weights with earlier residuals;
// a check of marginal mixtures alone would ignore the joint version structure.
fn predictive_adequacy(
    fit: &Mixture,
    rows: &Rows,
    ctx: &ExecutionContext,
) -> Result<bool, CausalError> {
    use antecedent_kernels::special::norm_cdf;

    let count = rows.residuals.len();
    let width = rows.residuals.first().map_or(0, Vec::len);
    if count == 0 || width == 0 {
        return Ok(false);
    }
    let mut pits: Vec<Vec<f64>> = (0..width).map(|_| Vec::with_capacity(count)).collect();
    for (row, design) in rows.residuals.iter().zip(&rows.design) {
        cancelled(ctx, "latent soft intervention predictive adequacy")?;
        let mut log_weights = [f64::NEG_INFINITY; 2];
        for (slot, weight) in log_weights.iter_mut().zip(fit.weights(design)) {
            *slot = weight.ln();
        }
        for ((coordinate, pit), residual) in pits.iter_mut().enumerate().zip(row) {
            let norm = log_sum_exp(log_weights);
            if !norm.is_finite() {
                return Ok(false);
            }
            let mut value = 0.0;
            for (weight, shift) in log_weights.iter_mut().zip(&fit.shifts) {
                let z = residual - shift[coordinate];
                value += (*weight - norm).exp() * norm_cdf(z);
                *weight -= 0.5 * z * z;
            }
            if !value.is_finite() {
                return Ok(false);
            }
            pit.push(value.clamp(0.0, 1.0));
        }
    }
    // A conservative screen for the two prefitted candidates across coordinates.
    // This uses an independent-uniform reference, not a calibration claim about
    // the estimated or selected model.
    let bound = ((400.0 * width as f64).ln() / (2.0 * count as f64)).sqrt();
    for pit in &mut pits {
        cancelled(ctx, "latent soft intervention predictive adequacy")?;
        pit.sort_unstable_by(f64::total_cmp);
        for (index, &value) in pit.iter().enumerate() {
            let left = index as f64 / count as f64;
            let right = (index + 1) as f64 / count as f64;
            if (value - left).max(right - value) > bound {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

struct GaussianComparator {
    coefficients: Vec<Vec<f64>>,
    cholesky: Vec<f64>,
    log_determinant: f64,
}

impl GaussianComparator {
    fn fit(rows: &Rows) -> Option<Self> {
        let n = rows.residuals.len();
        let d = rows.residuals.first()?.len();
        if n <= d + rows.design.first()?.len() {
            return None;
        }
        let mut coefficients = Vec::with_capacity(d);
        let mut errors = vec![vec![0.0; d]; n];
        for j in 0..d {
            let beta = regress(&rows.design, rows.residuals.iter().map(|row| row[j]))?;
            for (i, row) in errors.iter_mut().enumerate() {
                row[j] = rows.residuals[i][j] - dot(&rows.design[i], &beta);
            }
            coefficients.push(beta);
        }
        let mut covariance = vec![0.0; d * d];
        for row in errors {
            for i in 0..d {
                for j in 0..d {
                    covariance[i * d + j] += row[i] * row[j] / n as f64;
                }
            }
        }
        let cholesky = cholesky_spd(&covariance, d)?;
        let log_determinant = (0..d).map(|i| 2.0 * cholesky[i * d + i].ln()).sum::<f64>();
        log_determinant.is_finite().then_some(Self { coefficients, cholesky, log_determinant })
    }

    // Both scores omit the same constant -n * d * ln(2*pi) / 2.
    fn maximized_log_likelihood(&self, count: usize) -> f64 {
        -0.5 * count as f64 * (self.coefficients.len() as f64 + self.log_determinant)
    }

    fn log_likelihood(&self, rows: &Rows) -> Option<f64> {
        let d = self.coefficients.len();
        let mut log_likelihood = 0.0;
        for (row, x) in rows.residuals.iter().zip(&rows.design) {
            let residual: Vec<_> = row
                .iter()
                .zip(&self.coefficients)
                .map(|(value, beta)| value - dot(x, beta))
                .collect();
            let precision_residual = chol_solve(&self.cholesky, d, &residual)?;
            log_likelihood -= 0.5 * (self.log_determinant + dot(&residual, &precision_residual));
        }
        log_likelihood.is_finite().then_some(log_likelihood)
    }
}

fn masks(candidates: &[usize]) -> Vec<Vec<usize>> {
    let mut masks = vec![Vec::new()];
    for (i, &a) in candidates.iter().enumerate() {
        masks.push(vec![a]);
        masks.extend(candidates[i + 1..].iter().map(|&b| vec![a, b]));
    }
    masks
}

pub(super) fn run(
    request: &LatentSoftInterventionRequest,
    program: SemanticDigest,
    refit_seed: Option<u64>,
    ctx: &ExecutionContext,
) -> Result<Fit, CausalError> {
    if ctx.lowered_dispatch() != Some(program) {
        return Err(invalid("discovery requires its retained compiler program"));
    }
    // The fixed hypothesis cap is six targets, two shifts per version, and two starts.
    let mut budget = SearchBudget::new(
        SearchLimits { operations: request.search_operations as usize, depth: 2 },
        ctx,
    )
    .map_err(|receipt| search_refusal(receipt.stop))?;
    let (mut control_train, _) = arm_split(&request.baseline, request.seed ^ 0x434f4e54524f4c);
    let (mut action_train, mut action_eval) =
        arm_split(&request.action, request.seed ^ 0x414354494f4e);
    if let Some(seed) = refit_seed {
        let mut rng = CausalRng::from_seed(seed);
        // A source row cannot cross the honest split in any refit.
        for partition in [&mut control_train, &mut action_train, &mut action_eval] {
            let source = partition.clone();
            for row in partition {
                *row = source[(rng.next_u64() % source.len() as u64) as usize];
            }
        }
    }
    let equations = fit_equations(request, &control_train, ctx)?;
    let mut context = Context::fit(request, &action_train)?;
    let control_context = Context::fit(request, &control_train)?;
    for i in 0..context.lo.len() {
        context.lo[i] = context.lo[i].max(control_context.lo[i]);
        context.hi[i] = context.hi[i].min(control_context.hi[i]);
        if context.lo[i] >= context.hi[i] {
            return Err(invalid("the two arms lack overlapping pre-intervention support"));
        }
    }
    if action_train.is_empty() || action_eval.is_empty() {
        return Err(invalid("the split did not produce two nonempty partitions"));
    }
    if !request.context.is_empty() {
        let action_region = SupportRegion::fit(&context, &request.context, &action_train)
            .ok_or_else(|| invalid("the action-arm context covariance is singular"))?;
        let control_region = SupportRegion::fit(&context, &request.context, &control_train)
            .ok_or_else(|| invalid("the control-arm context covariance is singular"))?;
        context.action_region = Some(action_region);
        context.control_region = Some(control_region);
    }
    let train = action_rows(&equations, &context, request, &action_train);
    let eval = action_rows(&equations, &context, request, &action_eval);
    if [&train, &eval]
        .into_iter()
        .any(|rows| rows.residuals.iter().chain(&rows.design).flatten().any(|v| !v.is_finite()))
    {
        return Err(invalid("the anchored residual or context calculation is not finite"));
    }
    let columns: Vec<_> = (0..equations.len()).filter(|i| !request.context.contains(i)).collect();
    let candidates: Vec<_> = columns
        .iter()
        .enumerate()
        .filter_map(|(i, node)| request.candidates.contains(node).then_some(i))
        .collect();
    let masks = masks(&candidates);
    let n = train.residuals.len();
    let one = (0..n).map(|_| 1.0).collect::<Vec<_>>();
    let single = masks
        .iter()
        .map(|mask| {
            let model = Mixture {
                shifts: vec![weighted_shift(&train, &one, mask, n as f64)],
                gate: Vec::new(),
            };
            let criterion = score(&model, &train) - 0.5 * (n as f64).ln() * mask.len() as f64;
            (criterion, model)
        })
        .max_by(|a, b| a.0.total_cmp(&b.0))
        .ok_or_else(|| invalid("empty single-version search"))?
        .1;
    let null = GaussianComparator::fit(&eval)
        .ok_or_else(|| invalid("evaluation Gaussian comparator is not estimable"))?
        .maximized_log_likelihood(eval.residuals.len());
    let mut best: Option<(f64, Mixture, [usize; 2])> = None;
    let mut runner_up: Option<(f64, [usize; 2])> = None;
    for (i, a) in masks.iter().enumerate() {
        for (j, b) in masks[i..].iter().enumerate() {
            let mut candidate: Option<(f64, Mixture)> = None;
            for restart in 0..2 {
                let seed = (request.seed as usize).wrapping_add(i * 31 + j * 17 + restart);
                if let Some((likelihood, model)) = fit_pair(&train, [a, b], seed, &mut budget)? {
                    let penalty = 0.5 * (n as f64).ln() * (a.len() + b.len()) as f64;
                    let criterion = likelihood - penalty;
                    if candidate.as_ref().is_none_or(|(old, _)| criterion > *old) {
                        candidate = Some((criterion, model));
                    }
                }
            }
            if let Some((criterion, fit)) = candidate {
                let coordinates = [i, i + j];
                let displaced = if best.as_ref().is_none_or(|(old, _, _)| criterion > *old) {
                    best.replace((criterion, fit, coordinates))
                        .map(|(score, _, pair)| (score, pair))
                } else {
                    Some((criterion, coordinates))
                };
                if let Some((score, pair)) = displaced
                    && runner_up.is_none_or(|(old, _)| score > old)
                {
                    runner_up = Some((score, pair));
                }
            }
        }
    }
    let mut log_evidence = f64::NEG_INFINITY;
    let mut targets_resolved = false;
    let mut hypotheses = Vec::new();
    let mut mixture = single;
    if let Some((train_score, alternative, pair)) = best {
        log_evidence = score(&alternative, &eval) - null;
        let separation = alternative.shifts[0]
            .iter()
            .zip(&alternative.shifts[1])
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        if log_evidence.is_finite()
            && separation.is_finite()
            && log_evidence >= 20.0_f64.ln()
            && separation >= 0.5
        {
            // This margin is a stability diagnostic, not a confidence set for the targets.
            let hypothesis = |score, pair: [usize; 2]| LatentTargetHypothesis {
                target_sets: pair.map(|mask| masks[mask].iter().map(|&i| columns[i]).collect()),
                training_gap: train_score - score,
            };
            hypotheses.push(hypothesis(train_score, pair));
            if let Some((score, pair)) = runner_up {
                targets_resolved = train_score - score >= 2.0;
                hypotheses.push(hypothesis(score, pair));
            }
            mixture = alternative;
        }
    }
    // A failed split does not license the restricted single-version law. Check that
    // law against a context-dependent alternative fitted without evaluation outcomes.
    let single_version_adequate = if mixture.shifts.len() == 1 {
        let n_eval = eval.residuals.len();
        let eval_weights = vec![1.0; n_eval];
        let best_restricted = masks
            .iter()
            .map(|mask| {
                let model = Mixture {
                    shifts: vec![weighted_shift(&eval, &eval_weights, mask, n_eval as f64)],
                    gate: Vec::new(),
                };
                score(&model, &eval)
            })
            .fold(f64::NEG_INFINITY, f64::max);
        GaussianComparator::fit(&train)
            .and_then(|model| model.log_likelihood(&eval))
            .is_some_and(|score| score - best_restricted < 20.0_f64.ln())
    } else {
        true
    };
    let forecast_admissible = single_version_adequate && predictive_adequacy(&mixture, &eval, ctx)?;
    for shift in &mut mixture.shifts {
        let mut full = vec![0.0; equations.len()];
        for (&node, &effect) in columns.iter().zip(shift.iter()) {
            full[node] = effect * equations[node].sigma;
        }
        *shift = full;
    }
    if mixture.shifts.len() == 2
        && mixture.shifts[0]
            .iter()
            .zip(&mixture.shifts[1])
            .map(|(a, b)| a.total_cmp(b))
            .find(|order| !order.is_eq())
            .is_some_and(|order| order.is_gt())
    {
        mixture.shifts.swap(0, 1);
        for coefficient in &mut mixture.gate {
            *coefficient = -*coefficient;
        }
    }
    Ok(Fit {
        equations,
        context,
        mixture,
        log_evidence,
        targets_resolved,
        hypotheses,
        forecast_admissible,
        training_rows: action_train.len(),
        evaluation_rows: action_eval.len(),
    })
}

impl Fit {
    pub(super) fn write_forecast_identity(&self, writer: &mut antecedent_core::CanonicalWriter) {
        writer.reals(&self.mixture.gate);
        writer
            .reals(&self.context.center)
            .reals(&self.context.scale)
            .reals(&self.context.lo)
            .reals(&self.context.hi);
        for region in [&self.context.action_region, &self.context.control_region] {
            writer.word(u64::from(region.is_some()));
            if let Some(region) = region {
                writer.reals(&region.center).reals(&region.cholesky).real(region.radius_squared);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_anchored_mixture_must_also_fit_held_out_residuals() {
        use antecedent_stats::special::normal_ppf;
        let ctx = ExecutionContext::for_tests(0);
        let mixture = Mixture { shifts: vec![vec![-3.0], vec![3.0]], gate: vec![0.0] };
        let expected = Rows {
            residuals: [-3.0, 3.0]
                .into_iter()
                .flat_map(|shift| {
                    (0..100).map(move |i| vec![shift + normal_ppf((f64::from(i) + 0.5) / 100.0)])
                })
                .collect(),
            design: vec![vec![1.0]; 200],
        };
        assert!(predictive_adequacy(&mixture, &expected, &ctx).unwrap());

        // Both arms can remain distinguishable while their within-version spread
        // violates the control-anchored forecast.
        let wider = Rows {
            residuals: [-3.0, 3.0]
                .into_iter()
                .flat_map(|shift| {
                    (0..100)
                        .map(move |i| vec![shift + 4.0 * normal_ppf((f64::from(i) + 0.5) / 100.0)])
                })
                .collect(),
            design: vec![vec![1.0]; 200],
        };
        assert!(!predictive_adequacy(&mixture, &wider, &ctx).unwrap());
    }

    #[test]
    fn full_covariance_comparator_matches_closed_form_on_orthogonal_residuals() {
        let root = 2.0_f64.sqrt();
        let rows = Rows {
            residuals: vec![
                vec![3.0 - root, -4.0],
                vec![3.0 + root, -4.0],
                vec![3.0, -4.0 - 2.0 * root],
                vec![3.0, -4.0 + 2.0 * root],
            ],
            design: vec![vec![1.0]; 4],
        };
        // Mean [3, -4], full MLE covariance diag(1, 4), log|cov| = log 4.
        // Sum of the four Mahalanobis squares is 8.
        let expected = -0.5 * (4.0 * 4.0_f64.ln() + 8.0);
        let reference = GaussianComparator::fit(&rows).unwrap();
        assert!((reference.maximized_log_likelihood(4) - expected).abs() < 1e-12);
        assert!((reference.log_likelihood(&rows).unwrap() - expected).abs() < 1e-12);
    }
}
