//! Mechanism-change *detection* — separate from attribution.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    CausalRng, ExecutionContext, MechanismChangeQuery, StreamDomain, VariableId,
};
use antecedent_data::{TableView, TabularData};
use antecedent_model::{
    CompiledCausalModel, MechanismRegistry, ParentBatch, SelectionPolicy, infer_noise_column_rng,
};
use antecedent_stats::{
    DEFAULT_MECHANISM_PERMUTATIONS, FdrAdjustment, adjust_pvalues, change_point_two_sample,
    classifier_two_sample, kernel_two_sample_with_permutations, likelihood_ratio_permutations,
    mean_diff_two_sample, residual_likelihood_ratio,
};

use crate::change_common::{KERNEL_STREAM, attribution_stream};
use crate::error::AttributionError;
use crate::population::subset_table;
use crate::prep::resolve_population_pair;
use crate::result::MechanismChangeDetection;

mod block;
mod posterior;
#[cfg(test)]
use block::{block_statistic, partition_permutation_blocks};
pub use block::{detect_mechanism_changes_block, detect_mechanism_changes_block_with_runs};
pub use posterior::{
    MechanismChangePosterior, mechanism_change_posterior, mechanism_change_posterior_with_runs,
};
#[cfg(test)]
use posterior::{nig_log_marginal, serial_inflation, serial_inflation_with_runs};

/// Minimum rows required in each of the baseline and comparison populations.
///
/// Below this, neither a residual variance estimate nor an out-of-sample cross-fit is
/// reliable: a population this small can be exactly constant by chance (guaranteeing a
/// degenerate test) and cannot support the asymptotics the p-values rely on.
const MIN_POPULATION_ROWS: usize = 10;

/// Detection method selector.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum MechanismChangeMethod {
    /// Compare residual distributions via Gaussian KL / LR proxy.
    LikelihoodRatio,
    /// Mean difference on structural residuals (mechanism noise).
    MeanDiff,
    /// Classifier / two-sample proxy on residuals.
    ClassifierTwoSample,
    /// Kernel two-sample (MMD² + RBF) on residuals.
    KernelTwoSample,
    /// Known-split change-point test on concatenated baseline→comparison residuals.
    ChangePoint,
}

/// Detect which mechanisms differ between baseline and comparison populations.
///
/// This does **not** attribute outcome change — a changed mechanism need not
/// drive the target.
///
/// Applies a Benjamini–Hochberg false-discovery-rate correction across
/// `query.targets` by default (see
/// [`detect_mechanism_changes_with_correction`] to override or disable this).
/// `query.targets` is bounded by `query.max_targets`, i.e. the API is designed
/// to scan multiple candidate mechanisms in a single call, so judging each
/// target against the nominal `significance_level` independently would let the
/// family-wise false-positive rate grow with the number of targets (≈40% at
/// ten truly-unchanged targets and α=0.05). BH is chosen as the default
/// because it is the same default `antecedent_discovery::PcmciPlus` uses for
/// an analogous multi-hypothesis family (`PcmciPlus::new` sets
/// `fdr: Some(FdrAdjustment::bh())`), and it controls FDR rather than the
/// stricter (and more conservative) family-wise error rate, which is the
/// usual choice when the family is a moderate, human-reviewed set of
/// candidate mechanisms rather than a single make-or-break decision.
///
/// # Errors
///
/// Query / fit / stats failures.
pub fn detect_mechanism_changes(
    graph_model: &CompiledCausalModel,
    data: &TabularData,
    query: &MechanismChangeQuery,
    method: MechanismChangeMethod,
    ctx: &ExecutionContext,
) -> Result<Vec<MechanismChangeDetection>, AttributionError> {
    detect_mechanism_changes_with_correction(
        graph_model,
        data,
        query,
        method,
        ctx,
        Some(FdrAdjustment::bh()),
    )
}

/// Detect which mechanisms differ between baseline and comparison populations,
/// with explicit control over the multiple-testing correction applied across
/// `query.targets`.
///
/// `correction = None` reproduces the historical, uncorrected behavior where
/// each target's raw p-value is compared against `query.significance_level`
/// independently. `Some(adjustment)` adjusts the family of per-target p-values
/// with `adjustment.method` (see `antecedent_stats::fdr`) before thresholding.
///
/// [`MechanismChangeDetection::p_value`] always carries the *raw* per-target
/// p-value, and [`MechanismChangeDetection::adjusted_p_value`] carries the
/// corrected one (`None` when `correction` is `None`). `changed` is decided
/// from the adjusted value when present, so the flag and the number behind it
/// always agree while the unadjusted per-test result stays visible.
///
/// This does **not** attribute outcome change — a changed mechanism need not
/// drive the target.
///
/// # Errors
///
/// Query / fit / stats failures.
#[allow(
    clippy::too_many_lines,
    reason = "fit both sides, test every mechanism, then correct the whole family together; the stages share the per-mechanism results"
)]
pub(crate) fn detect_mechanism_changes_with_correction(
    graph_model: &CompiledCausalModel,
    data: &TabularData,
    query: &MechanismChangeQuery,
    method: MechanismChangeMethod,
    ctx: &ExecutionContext,
    correction: Option<FdrAdjustment>,
) -> Result<Vec<MechanismChangeDetection>, AttributionError> {
    query.validate()?;
    if matches!(method, MechanismChangeMethod::ChangePoint) {
        crate::population::reject_mask_for_ordered(data)?;
    }
    if query.targets.len() > query.max_targets {
        return Err(AttributionError::SizeLimit {
            kind: "targets",
            requested: query.targets.len(),
            max: query.max_targets,
        });
    }
    let alpha = query.significance_level.to_f64();
    let (baseline, comparison) = resolve_population_pair(data, &query.baseline, &query.comparison)?;
    if baseline.row_count() < MIN_POPULATION_ROWS || comparison.row_count() < MIN_POPULATION_ROWS {
        return Err(AttributionError::invalid_input(
            "baseline and comparison populations need at least 10 rows each: fewer cannot \
             support a calibrated residual distribution and risk a degenerate (e.g. constant) \
             segment",
        ));
    }
    // A residual two-sample test asks "do these two noise samples look alike", which only
    // answers "did the mechanism change" when both samples are noise from the *same*
    // covariate region. If a target's parents differ materially in distribution between
    // the populations, the baseline-fitted mechanism's own parameter-estimation error
    // becomes a bias shared by every comparison residual alike (the same fitted
    // coefficients applied to every row), and that shared bias does not shrink with the
    // comparison sample size the way genuine per-row noise does — no reconstruction of the
    // residuals themselves can make a same-shape-distribution test calibrated against it;
    // the correct comparison is between the *fitted mechanisms* (e.g. a Chow/Wald test on
    // pooled regression with a population interaction term), not their residuals. Refusing
    // here is the honest fallback until that comparison is implemented: a wrong "changed"
    // claim from covariate drift is worse than no claim.
    for &target in query.targets.iter() {
        if parent_covariate_shift_detected(graph_model, &baseline, &comparison, target)? {
            return Err(AttributionError::invalid_input(
                "baseline and comparison populations differ materially in a tested \
                 mechanism's parent covariates; residual two-sample tests are not \
                 calibrated under a covariate shift with no accompanying mechanism change, \
                 so the comparison is refused",
            ));
        }
    }

    // Fitting the mechanism on the full baseline and handing that one model every
    // baseline row too would be in-sample for the baseline side (shrunk variance, no
    // estimation error) while comparison residuals stay out-of-sample — an asymmetry
    // that is anti-conservative on its own, independent of the covariate-shift defect
    // guarded against above. Cross-fitting the baseline fixes it: each baseline row's
    // residual comes from the one fold model that excluded it, a genuine held-out
    // prediction, matching the kind of estimation error already present on the
    // comparison side.
    let CrossFittedBaseline { base_model, fold_models, fold_tables, baseline_folds } =
        fit_cross_fitted_baseline(graph_model, &baseline)?;

    let raw = ctx.map_indexed(query.targets.len(), |ti, worker| {
        let target = query.targets[ti];
        let mut baseline_rng =
            worker.rng.stream_for(StreamDomain::Attribution, residual_stream_id(target, 0));
        let mut comparison_rng =
            worker.rng.stream_for(StreamDomain::Attribution, residual_stream_id(target, 1));
        let rb = cross_fitted_residuals(
            &fold_models,
            &fold_tables,
            &baseline_folds,
            baseline.row_count(),
            target,
            &mut baseline_rng,
        )?;
        let rc = residuals(&base_model, &comparison, target, &mut comparison_rng)?;
        let (stat, p_value, method_name, n_permutations) = match method {
            MechanismChangeMethod::LikelihoodRatio => {
                let (s, p) = residual_likelihood_ratio(&rb, &rc)?;
                (s, p, "likelihood_ratio", likelihood_ratio_permutations(rb.len(), rc.len()))
            }
            MechanismChangeMethod::MeanDiff => {
                let (s, p) = mean_diff_two_sample(&rb, &rc)?;
                (s, p, "mean_diff", None)
            }
            MechanismChangeMethod::ClassifierTwoSample => {
                let (s, p) = classifier_two_sample(&rb, &rc)?;
                (s, p, "classifier_two_sample", None)
            }
            MechanismChangeMethod::KernelTwoSample => {
                let seed =
                    attribution_stream(worker, KERNEL_STREAM, target.as_usize() as u64).next_u64();
                let r = kernel_two_sample_with_permutations(
                    &rb,
                    &rc,
                    seed,
                    DEFAULT_MECHANISM_PERMUTATIONS,
                )?;
                (r.statistic, r.p_value, "kernel_two_sample", Some(r.n_permutations))
            }
            MechanismChangeMethod::ChangePoint => {
                let (s, p) = change_point_two_sample(&rb, &rc)?;
                (s, p, "change_point", likelihood_ratio_permutations(rb.len(), rc.len()))
            }
        };
        Ok::<_, AttributionError>((target, stat, p_value, method_name, n_permutations))
    })?;

    // Report the raw p-value and the adjusted one separately rather than overwriting the
    // former. `changed` is decided from whichever is authoritative, but a caller reading
    // `p_value` must get the unadjusted per-target quantity its documentation promises —
    // silently returning a corrected number under that name is exactly the kind of
    // mislabelled statistic this correction exists to guard against.
    let adjusted: Option<Vec<f64>> = correction.map(|adjustment| {
        let p_values: Vec<f64> = raw.iter().map(|&(_, _, p, _, _)| p).collect();
        adjust_pvalues(&p_values, adjustment.method)
    });

    let mut out = Vec::with_capacity(raw.len());
    for (i, (target, stat, raw_p, method_name, n_permutations)) in raw.into_iter().enumerate() {
        let adjusted_p_value = adjusted.as_ref().map(|a| a[i]);
        let decisive = adjusted_p_value.unwrap_or(raw_p);
        out.push(MechanismChangeDetection {
            variable: target,
            changed: decisive < alpha,
            statistic: stat,
            p_value: raw_p,
            adjusted_p_value,
            method: Arc::from(method_name),
            n_permutations,
            p_value_floor: n_permutations.map(|b| 1.0 / (b as f64 + 1.0)),
            posterior_probability: None,
        });
    }
    Ok(out)
}

struct CrossFittedBaseline {
    base_model: CompiledCausalModel,
    fold_models: Vec<CompiledCausalModel>,
    fold_tables: Vec<TabularData>,
    baseline_folds: Vec<Vec<usize>>,
}

fn fit_cross_fitted_baseline(
    graph_model: &CompiledCausalModel,
    baseline: &TabularData,
) -> Result<CrossFittedBaseline, AttributionError> {
    let (base_mechs, _) = MechanismRegistry::standard().assign_and_fit(
        graph_model,
        baseline,
        SelectionPolicy::BestScore,
    )?;
    let base_model = graph_model.clone().with_mechanisms(base_mechs);
    let n_base = baseline.row_count();
    let fold_count = baseline_cross_fit_folds(n_base);
    let baseline_folds = kfold_row_indices(n_base, fold_count);
    let mut fold_models = Vec::with_capacity(baseline_folds.len());
    // Folds are round-robin (`row % k`), so the complement is the same test without
    // scanning each held-out vector per row.
    for fold in 0..baseline_folds.len() {
        let train_rows: Vec<usize> = (0..n_base).filter(|r| r % fold_count != fold).collect();
        let train = subset_table(baseline, &train_rows)?;
        let (mechs, _) = MechanismRegistry::standard().assign_and_fit(
            graph_model,
            &train,
            SelectionPolicy::BestScore,
        )?;
        fold_models.push(graph_model.clone().with_mechanisms(mechs));
    }
    let fold_tables = baseline_folds
        .iter()
        .map(|assigned| subset_table(baseline, assigned))
        .collect::<Result<Vec<_>, _>>()?;

    Ok(CrossFittedBaseline { base_model, fold_models, fold_tables, baseline_folds })
}

/// A target with its baseline and comparison residuals.
type TargetResiduals = (VariableId, Vec<f64>, Vec<f64>);

/// The baseline (cross-fitted, out of sample) and comparison residuals of every target of
/// `query`, with the guards of [`detect_mechanism_changes_with_correction`]: populations of at
/// least ten rows and a refusal under a parent-covariate shift.
fn residual_pairs(
    graph_model: &CompiledCausalModel,
    data: &TabularData,
    query: &MechanismChangeQuery,
    ctx: &ExecutionContext,
) -> Result<Vec<TargetResiduals>, AttributionError> {
    query.validate()?;
    if query.targets.len() > query.max_targets {
        return Err(AttributionError::SizeLimit {
            kind: "targets",
            requested: query.targets.len(),
            max: query.max_targets,
        });
    }
    let (baseline, comparison) = resolve_population_pair(data, &query.baseline, &query.comparison)?;
    if baseline.row_count() < MIN_POPULATION_ROWS || comparison.row_count() < MIN_POPULATION_ROWS {
        return Err(AttributionError::invalid_input(
            "baseline and comparison populations need at least 10 rows each: fewer cannot \
             support a calibrated residual distribution and risk a degenerate (e.g. constant) \
             segment",
        ));
    }
    for &target in query.targets.iter() {
        if parent_covariate_shift_detected(graph_model, &baseline, &comparison, target)? {
            return Err(AttributionError::invalid_input(
                "baseline and comparison populations differ materially in a tested \
                 mechanism's parent covariates; residual two-sample tests are not \
                 calibrated under a covariate shift with no accompanying mechanism change, \
                 so the comparison is refused",
            ));
        }
    }
    let CrossFittedBaseline { base_model, fold_models, fold_tables, baseline_folds } =
        fit_cross_fitted_baseline(graph_model, &baseline)?;

    ctx.map_indexed(query.targets.len(), |ti, worker| {
        let target = query.targets[ti];
        let mut baseline_rng =
            worker.rng.stream_for(StreamDomain::Attribution, residual_stream_id(target, 0));
        let mut comparison_rng =
            worker.rng.stream_for(StreamDomain::Attribution, residual_stream_id(target, 1));
        let rb = cross_fitted_residuals(
            &fold_models,
            &fold_tables,
            &baseline_folds,
            baseline.row_count(),
            target,
            &mut baseline_rng,
        )?;
        let rc = residuals(&base_model, &comparison, target, &mut comparison_rng)?;
        Ok::<_, AttributionError>((target, rb, rc))
    })
    .map_err(Into::into)
}

/// Fewest block permutations a dependent-series test runs (the smallest attainable p-value is
/// `1 / (permutations + 1)`).
pub const MIN_BLOCK_PERMUTATIONS: usize = 99;

/// Gaussian two-sample likelihood-ratio statistic `2 log LR` of a common mean and variance
/// against separate ones (two extra parameters).
fn gaussian_lr_statistic(a: &[f64], b: &[f64]) -> f64 {
    let pooled: Vec<f64> = a.iter().chain(b).copied().collect();
    let (_, var_pooled) = antecedent_stats::mean_var(&pooled);
    let (_, var_a) = antecedent_stats::mean_var(a);
    let (_, var_b) = antecedent_stats::mean_var(b);
    let floor = f64::MIN_POSITIVE;
    (pooled.len() as f64) * var_pooled.max(floor).ln()
        - (a.len() as f64) * var_a.max(floor).ln()
        - (b.len() as f64) * var_b.max(floor).ln()
}

type PopulationRuns<'a> = (&'a [std::ops::Range<usize>], &'a [std::ops::Range<usize>]);

fn poll_run_cancellation(ctx: &ExecutionContext, index: usize) -> Result<(), AttributionError> {
    if index % 256 == 0 && ctx.cancellation.is_cancelled() {
        Err(AttributionError::Cancelled)
    } else {
        Ok(())
    }
}

// Account for caller-provided partition lists before inspecting their elements or doing fits.
fn charge_run_validation(
    runs: PopulationRuns<'_>,
    budget: &mut antecedent_core::TemporalBudget,
    ctx: &ExecutionContext,
) -> Result<(), AttributionError> {
    if ctx.cancellation.is_cancelled() {
        return Err(AttributionError::Cancelled);
    }
    let count = runs
        .0
        .len()
        .checked_add(runs.1.len())
        .and_then(|n| u64::try_from(n).ok())
        .ok_or(AttributionError::Budget { message: "run validation work overflow".into() })?;
    budget.charge_work(count).map_err(|e| AttributionError::Budget { message: e.to_string() })
}

fn validate_runs(
    runs: &[std::ops::Range<usize>],
    len: usize,
    ctx: &ExecutionContext,
) -> Result<(), AttributionError> {
    if runs.len() > len {
        return Err(AttributionError::invalid_input("runs must partition the compact population"));
    }
    let mut end = 0;
    for (i, run) in runs.iter().enumerate() {
        poll_run_cancellation(ctx, i)?;
        if run.start != end || run.end <= run.start || run.end > len {
            return Err(AttributionError::invalid_input(
                "runs must partition the compact population",
            ));
        }
        end = run.end;
    }
    if end != len {
        return Err(AttributionError::invalid_input("runs must partition the compact population"));
    }
    Ok(())
}

/// Distinct RNG stream per (target, population) so seeding is deterministic
/// under [`ExecutionContext::for_tests`] / any fixed `master_seed` but doesn't
/// alias across targets or across the baseline/comparison populations.
fn residual_stream_id(target: VariableId, population: u64) -> u64 {
    antecedent_core::derive_stream(
        antecedent_core::derive_stream(0x_5EED_C0DE, target.as_usize() as u64),
        population,
    )
}

fn residuals(
    model: &CompiledCausalModel,
    data: &TabularData,
    target: VariableId,
    rng: &mut CausalRng,
) -> Result<Vec<f64>, AttributionError> {
    let dense =
        model.dense_of(target).ok_or_else(|| AttributionError::missing_var("target", target))?;
    let gather =
        model.gather_for(dense).ok_or(AttributionError::MissingArtifact("missing gather"))?;
    let n = data.row_count();
    let y = data.float64_values(target)?;
    let mut parent_mat = vec![0.0; n * gather.n_parents().max(1)];
    for (pi, &p) in gather.parents.iter().enumerate() {
        let pv = model.output_layout.variables[p.as_usize()];
        let col = data.float64_values(pv)?;
        parent_mat[pi * n..(pi + 1) * n].copy_from_slice(&col[..n]);
    }
    let parents = ParentBatch {
        n_rows: n,
        n_parents: gather.n_parents(),
        values: &parent_mat[..gather.n_parents().saturating_mul(n)],
    };
    let mut noise = vec![0.0; n];
    infer_noise_column_rng(model.mechanisms.get(dense), &y, parents, &mut noise, rng)?;
    Ok(noise)
}

/// Row indices for each held-out fold of a `k`-fold split of `0..n` (round-robin
/// assignment, so folds differ in size by at most one row).
fn kfold_row_indices(n: usize, k: usize) -> Vec<Vec<usize>> {
    let k = k.max(1);
    let mut folds = vec![Vec::new(); k];
    for row in 0..n {
        folds[row % k].push(row);
    }
    folds
}

/// Fold count for cross-fitting the baseline.
///
/// [`MIN_POPULATION_ROWS`] already guarantees `n >= 10`. More folds hold out less data per
/// fit (each fold model stays closer to a full-baseline fit) at the cost of more refits;
/// fewer folds keep every training set large relative to `n` when `n` is only just above
/// the minimum.
fn baseline_cross_fit_folds(n: usize) -> usize {
    if n >= 30 { 5 } else { 3 }
}

/// Out-of-sample baseline residuals for `target` via `k`-fold cross-fitting.
///
/// `fold_models[i]` must be fit on the rows *not* in `folds[i]`; each baseline row's
/// residual then comes from a mechanism that never saw it during fitting, rather than the
/// full-baseline fit's own (shrunk-variance) in-sample residual.
fn cross_fitted_residuals(
    fold_models: &[CompiledCausalModel],
    fold_tables: &[TabularData],
    folds: &[Vec<usize>],
    n_rows: usize,
    target: VariableId,
    rng: &mut CausalRng,
) -> Result<Vec<f64>, AttributionError> {
    let mut out = vec![0.0; n_rows];
    for ((model, table), assigned) in fold_models.iter().zip(fold_tables.iter()).zip(folds.iter()) {
        let fold_residuals = residuals(model, table, target, rng)?;
        for (&row, &value) in assigned.iter().zip(fold_residuals.iter()) {
            out[row] = value;
        }
    }
    Ok(out)
}

/// Significance level for the parent-covariate-shift guard in
/// [`parent_covariate_shift_detected`].
///
/// This is a nuisance check, not the mechanism-change hypothesis itself, so it does not
/// borrow `query.significance_level`. It is also checked once per parent of every target in
/// `query.targets`, uncorrected, so it needs its own margin against the same family-wise
/// inflation `detect_mechanism_changes`'s BH default exists to control for the primary
/// tests: a much stricter fixed threshold keeps the false-refusal rate low across a
/// realistic number of targets under a true null of "same parent distribution", while
/// still reliably catching the magnitude of shift that makes the residual tests
/// anti-conservative (a shift of even one parent standard deviation separates two
/// `n >= 10` samples by a z-score several orders of magnitude past this threshold).
const PARENT_SHIFT_ALPHA: f64 = 1e-4;

/// Whether `target`'s parent covariates differ materially between `baseline` and
/// `comparison`, via a two-sample mean-difference test on each parent column.
///
/// See the comment in [`detect_mechanism_changes_with_correction`] for why this matters: a
/// residual two-sample test cannot be calibrated against a covariate shift no matter how
/// the residuals are computed, so detecting the shift and refusing is the honest fallback.
///
/// # Errors
///
/// Missing variable / gather / column data, or a parent whose mean-difference test cannot
/// be evaluated ([`AttributionError::InvalidInput`]).
fn parent_covariate_shift_detected(
    model: &CompiledCausalModel,
    baseline: &TabularData,
    comparison: &TabularData,
    target: VariableId,
) -> Result<bool, AttributionError> {
    let dense =
        model.dense_of(target).ok_or_else(|| AttributionError::missing_var("target", target))?;
    let gather =
        model.gather_for(dense).ok_or(AttributionError::MissingArtifact("missing gather"))?;
    for &parent in gather.parents.iter() {
        let pv = model.output_layout.variables[parent.as_usize()];
        let b = baseline.float64_values(pv)?;
        let c = comparison.float64_values(pv)?;
        // A check that cannot run (e.g. non-finite parent values) is not evidence of "no
        // shift": skipping it would let an uncalibrated residual test through.
        let (_, p) = mean_diff_two_sample(&b, &c).map_err(|_| {
            AttributionError::invalid_input(
                "the parent-covariate shift check could not be evaluated on a tested \
                 mechanism's parents (too few or non-finite values), so the comparison is \
                 refused",
            )
        })?;
        if p < PARENT_SHIFT_ALPHA {
            return Ok(true);
        }
    }
    Ok(false)
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod block_null_tests {
    use super::{
        MechanismChangeMethod, block_statistic, gaussian_lr_statistic, nig_log_marginal,
        serial_inflation,
    };

    fn ar1(n: usize, rho: f64, level: f64, state: &mut u64) -> Vec<f64> {
        let mut next = || {
            *state ^= *state >> 12;
            *state ^= *state << 25;
            *state ^= *state >> 27;
            let u = (state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64;
            u - 0.5
        };
        let mut x = 0.0;
        (0..n)
            .map(|_| {
                x = rho * x + next();
                x + level
            })
            .collect()
    }

    #[test]
    fn the_likelihood_ratio_statistic_separates_a_shift_from_none() {
        let mut state = 0x1234_5678_9ABC_DEF1_u64;
        let a = ar1(200, 0.0, 0.0, &mut state);
        let same = ar1(200, 0.0, 0.0, &mut state);
        let shifted = ar1(200, 0.0, 1.0, &mut state);
        assert!(gaussian_lr_statistic(&a, &same) < 10.0);
        assert!(gaussian_lr_statistic(&a, &shifted) > 100.0);
    }

    #[test]
    fn the_change_point_statistic_has_a_block_null_that_a_shift_exceeds() {
        let mut state = 0x0BAD_CAFE_1234_5678_u64;
        let a = ar1(200, 0.5, 0.0, &mut state);
        let shifted = ar1(200, 0.5, 1.5, &mut state);
        let observed = block_statistic(MechanismChangeMethod::ChangePoint, &a, &shifted).unwrap();
        let pooled: Vec<f64> = a.iter().chain(shifted.iter()).copied().collect();
        // Permute whole blocks of 20 consecutive times: the split no longer separates the
        // regimes, so the statistic falls below the observed one in most permutations.
        let blocks: Vec<&[f64]> = pooled.chunks(20).collect();
        let mut below = 0;
        for shift in 1..blocks.len() {
            let mut rotated = Vec::with_capacity(pooled.len());
            for k in 0..blocks.len() {
                rotated.extend_from_slice(blocks[(k * 7 + shift) % blocks.len()]);
            }
            let (x, y) = rotated.split_at(200);
            if block_statistic(MechanismChangeMethod::ChangePoint, x, y).unwrap() < observed {
                below += 1;
            }
        }
        assert!(below >= blocks.len() - 2, "{below} of {}", blocks.len() - 1);
    }

    #[test]
    fn serial_dependence_inflates_the_statistic_null_and_white_noise_does_not() {
        let mut state = 0x0FED_CBA9_8765_4321_u64;
        let white_a = ar1(400, 0.0, 0.0, &mut state);
        let white_b = ar1(400, 0.0, 0.0, &mut state);
        let slow_a = ar1(400, 0.9, 0.0, &mut state);
        let slow_b = ar1(400, 0.9, 0.0, &mut state);
        let white = serial_inflation(&white_a, &white_b, 12);
        let slow = serial_inflation(&slow_a, &slow_b, 12);
        assert!((1.0..1.6).contains(&white), "white noise inflation {white}");
        assert!(slow > 2.0 * white, "persistent residuals inflate more: {slow} against {white}");
    }

    fn log_bayes_factor(a: &[f64], b: &[f64]) -> f64 {
        let part = |xs: &[f64]| {
            let n = xs.len() as f64;
            let mean = xs.iter().sum::<f64>() / n;
            let ss = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>();
            (n, mean, ss)
        };
        let pooled: Vec<f64> = a.iter().chain(b).copied().collect();
        let (n, mean, ss) = part(&pooled);
        let beta0 = ss / n;
        let (na, ma, sa) = part(a);
        let (nb, mb, sb) = part(b);
        nig_log_marginal(na, ma, sa, beta0) + nig_log_marginal(nb, mb, sb, beta0)
            - nig_log_marginal(n, mean, ss, beta0)
    }

    #[test]
    fn the_conjugate_bayes_factor_favours_a_shift_and_penalises_a_needless_split() {
        let mut state = 0xABCD_EF01_2345_6789_u64;
        let a = ar1(200, 0.0, 0.0, &mut state);
        let same = ar1(200, 0.0, 0.0, &mut state);
        let shifted = ar1(200, 0.0, 1.0, &mut state);
        assert!(log_bayes_factor(&a, &shifted) > 10.0, "{}", log_bayes_factor(&a, &shifted));
        assert!(log_bayes_factor(&a, &same) < 0.0, "{}", log_bayes_factor(&a, &same));
    }
}
