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
    let (base_mechs, _) = MechanismRegistry::standard().assign_and_fit(
        graph_model,
        &baseline,
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
        let train = subset_table(&baseline, &train_rows)?;
        let (mechs, _) = MechanismRegistry::standard().assign_and_fit(
            graph_model,
            &train,
            SelectionPolicy::BestScore,
        )?;
        fold_models.push(graph_model.clone().with_mechanisms(mechs));
    }
    let fold_tables = baseline_folds
        .iter()
        .map(|assigned| subset_table(&baseline, assigned))
        .collect::<Result<Vec<_>, _>>()?;

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
    let (base_mechs, _) = MechanismRegistry::standard().assign_and_fit(
        graph_model,
        &baseline,
        SelectionPolicy::BestScore,
    )?;
    let base_model = graph_model.clone().with_mechanisms(base_mechs);
    let n_base = baseline.row_count();
    let fold_count = baseline_cross_fit_folds(n_base);
    let baseline_folds = kfold_row_indices(n_base, fold_count);
    let mut fold_models = Vec::with_capacity(baseline_folds.len());
    for fold in 0..baseline_folds.len() {
        let train_rows: Vec<usize> = (0..n_base).filter(|r| r % fold_count != fold).collect();
        let train = subset_table(&baseline, &train_rows)?;
        let (mechs, _) = MechanismRegistry::standard().assign_and_fit(
            graph_model,
            &train,
            SelectionPolicy::BestScore,
        )?;
        fold_models.push(graph_model.clone().with_mechanisms(mechs));
    }
    let fold_tables = baseline_folds
        .iter()
        .map(|assigned| subset_table(&baseline, assigned))
        .collect::<Result<Vec<_>, _>>()?;
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

/// Seed of the kernel statistic's bandwidth stream in a dependent-series test: fixed, so the
/// observed and permuted statistics use the same bandwidth rule.
const BLOCK_KERNEL_SEED: u64 = 0xB10C_4E57;

/// The statistic a dependent-series test permutes, by method: the Gaussian likelihood ratio, the
/// absolute mean difference, the kernel MMD² (RBF, median-heuristic bandwidth) or the
/// classifier two-sample statistic, or the known-split change-point likelihood ratio (the
/// statistic only; its chi-square p-value is replaced by the block null).
fn block_statistic(
    method: MechanismChangeMethod,
    a: &[f64],
    b: &[f64],
) -> Result<f64, AttributionError> {
    match method {
        MechanismChangeMethod::LikelihoodRatio => Ok(gaussian_lr_statistic(a, b)),
        MechanismChangeMethod::MeanDiff => {
            let mean = |xs: &[f64]| xs.iter().sum::<f64>() / xs.len().max(1) as f64;
            Ok((mean(a) - mean(b)).abs())
        }
        MechanismChangeMethod::KernelTwoSample => {
            Ok(kernel_two_sample_with_permutations(a, b, BLOCK_KERNEL_SEED, 1)?.statistic)
        }
        MechanismChangeMethod::ClassifierTwoSample => Ok(classifier_two_sample(a, b)?.0),
        MechanismChangeMethod::ChangePoint => Ok(change_point_two_sample(a, b)?.0),
    }
}

/// Detect mechanism changes between two populations that are segments of one dependent series.
///
/// The residual two-sample tests of [`detect_mechanism_changes`] draw their null law as if the
/// residuals were independent, which the consecutive times of one series are not. Here the null
/// law is a **block permutation** one: the baseline and comparison residuals are concatenated in
/// time, cut into blocks of `block` consecutive times, the blocks are permuted at random and the
/// first `n_baseline` residuals are called the baseline, so the serial dependence inside a block
/// is kept under the null. The p-value is `(1 + #{statistic* ≥ statistic}) / (B + 1)`; the
/// Benjamini–Hochberg correction across targets is applied as in the independent case.
///
/// Every statistic has a block null, the known-split change-point statistic included: the
/// null law of its mean-and-variance likelihood ratio is that of the same statistic computed at
/// the same split of block-permuted residuals.
///
/// # Errors
///
/// As [`detect_mechanism_changes`], an unsupported method, or fewer than
/// [`MIN_BLOCK_PERMUTATIONS`] permutations.
pub fn detect_mechanism_changes_block(
    graph_model: &CompiledCausalModel,
    data: &TabularData,
    query: &MechanismChangeQuery,
    method: MechanismChangeMethod,
    block: usize,
    permutations: usize,
    ctx: &ExecutionContext,
) -> Result<Vec<MechanismChangeDetection>, AttributionError> {
    if permutations < MIN_BLOCK_PERMUTATIONS {
        return Err(AttributionError::invalid_input(
            "a block permutation test needs at least 99 permutations",
        ));
    }
    let alpha = query.significance_level.to_f64();
    let pairs = residual_pairs(graph_model, data, query, ctx)?;
    let raw = ctx.map_indexed(pairs.len(), |i, worker| {
        let (target, rb, rc) = &pairs[i];
        let observed = block_statistic(method, rb, rc)?;
        let mut rng = CausalRng::from_seed(
            attribution_stream(worker, KERNEL_STREAM, target.as_usize() as u64).next_u64(),
        );
        let pooled: Vec<f64> = rb.iter().chain(rc.iter()).copied().collect();
        let n = pooled.len();
        let block = block.clamp(1, n);
        let starts: Vec<usize> = (0..n).step_by(block).collect();
        let mut order: Vec<usize> = (0..starts.len()).collect();
        let mut shuffled = Vec::with_capacity(n);
        let mut exceed = 0_usize;
        for _ in 0..permutations {
            for k in (1..order.len()).rev() {
                #[allow(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "next_f64() is in [0, 1), so the scaled draw is an index at most k"
                )]
                let j = ((rng.next_f64() * (k + 1) as f64) as usize).min(k);
                order.swap(k, j);
            }
            shuffled.clear();
            for &o in &order {
                let start = starts[o];
                shuffled.extend_from_slice(&pooled[start..(start + block).min(n)]);
            }
            let (a, b) = shuffled.split_at(rb.len());
            let statistic = block_statistic(method, a, b)?;
            if statistic >= observed - 1e-12 {
                exceed += 1;
            }
        }
        let p = (1 + exceed) as f64 / (permutations + 1) as f64;
        Ok::<_, AttributionError>((*target, observed, p))
    })?;
    let p_values: Vec<f64> = raw.iter().map(|&(_, _, p)| p).collect();
    let adjusted = adjust_pvalues(&p_values, FdrAdjustment::bh().method);
    let name = match method {
        MechanismChangeMethod::LikelihoodRatio => "likelihood_ratio.block_permutation",
        MechanismChangeMethod::KernelTwoSample => "kernel_two_sample.block_permutation",
        MechanismChangeMethod::ClassifierTwoSample => "classifier_two_sample.block_permutation",
        MechanismChangeMethod::ChangePoint => "change_point.block_permutation",
        MechanismChangeMethod::MeanDiff => "mean_diff.block_permutation",
    };
    Ok(raw
        .into_iter()
        .zip(adjusted)
        .map(|((target, statistic, p), adjusted_p)| MechanismChangeDetection {
            variable: target,
            changed: adjusted_p < alpha,
            statistic,
            p_value: p,
            adjusted_p_value: Some(adjusted_p),
            method: Arc::from(name),
            n_permutations: Some(permutations),
            p_value_floor: Some(1.0 / (permutations as f64 + 1.0)),
            posterior_probability: None,
        })
        .collect())
}

/// The posterior probability that one tested mechanism changed.
#[derive(Clone, Debug, PartialEq)]
pub struct MechanismChangePosterior {
    /// Tested node.
    pub variable: VariableId,
    /// Gaussian likelihood-ratio statistic `2 log LR` of separate against common residual mean
    /// and variance.
    pub statistic: f64,
    /// Variance inflation of the residuals' serial dependence (at least 1): the likelihood is
    /// tempered by its reciprocal, the effective-sample deflation.
    pub inflation: f64,
    /// `log` Bayes factor of a change against no change from the conjugate
    /// normal-inverse-gamma marginal likelihoods (see [`mechanism_change_posterior`]).
    pub log_bayes_factor: f64,
    /// The declared prior probability of a change.
    pub prior: f64,
    /// The posterior probability of a change given the residuals.
    pub probability: f64,
}

/// Prior pseudo-count of the residual mean (its prior sd is `sigma / sqrt(KAPPA0)`).
const NIG_KAPPA0: f64 = 1.0;
/// Shape of the inverse-gamma prior of the residual variance (prior mean `beta0`).
const NIG_ALPHA0: f64 = 2.0;

/// `log` marginal likelihood of `n` (effective) Gaussian observations with mean `mean` and
/// centred sum of squares `ss` under the conjugate prior `mu | s2 ~ N(0, s2 / kappa0)`,
/// `s2 ~ InvGamma(alpha0, beta0)`.
fn nig_log_marginal(n: f64, mean: f64, ss: f64, beta0: f64) -> f64 {
    let kappa_n = NIG_KAPPA0 + n;
    let alpha_n = NIG_ALPHA0 + 0.5 * n;
    let beta_n = beta0 + 0.5 * ss + 0.5 * NIG_KAPPA0 * n * mean * mean / kappa_n;
    antecedent_stats::ln_gamma(alpha_n) - antecedent_stats::ln_gamma(NIG_ALPHA0)
        + NIG_ALPHA0 * beta0.ln()
        - alpha_n * beta_n.ln()
        + 0.5 * (NIG_KAPPA0 / kappa_n).ln()
        - 0.5 * n * (2.0 * std::f64::consts::PI).ln()
}

impl MechanismChangePosterior {
    /// The posterior as a detection record: `changed` when the probability exceeds one half, the
    /// adjusted statistic, no p-value (`NaN`), and the probability itself.
    #[must_use]
    pub fn into_detection(self) -> MechanismChangeDetection {
        MechanismChangeDetection {
            variable: self.variable,
            changed: self.probability > 0.5,
            statistic: self.statistic / self.inflation,
            p_value: f64::NAN,
            adjusted_p_value: None,
            method: Arc::from("posterior_probability.bic_block"),
            n_permutations: None,
            p_value_floor: None,
            posterior_probability: Some(self.probability),
        }
    }
}

/// Bartlett long-run variance inflation `1 + 2 Σ_{k<block} (1 − k/block) ρ_k` of the residuals
/// of two populations, each centred on its own mean, floored at 1.
fn serial_inflation(rb: &[f64], rc: &[f64], block: usize) -> f64 {
    let centred = |xs: &[f64]| {
        let mean = xs.iter().sum::<f64>() / xs.len().max(1) as f64;
        xs.iter().map(|x| x - mean).collect::<Vec<f64>>()
    };
    let (a, b) = (centred(rb), centred(rc));
    let n = (a.len() + b.len()) as f64;
    let variance = (a.iter().chain(&b).map(|x| x * x).sum::<f64>() / n).max(f64::MIN_POSITIVE);
    let lag_product = |series: &[f64], k: usize| -> f64 {
        series.iter().zip(series.iter().skip(k)).map(|(x, y)| x * y).sum::<f64>()
    };
    let block = block.max(1);
    let mut inflation = 1.0;
    for k in 1..block.min(a.len().min(b.len())) {
        let rho = (lag_product(&a, k) + lag_product(&b, k)) / (n * variance);
        inflation += 2.0 * (1.0 - k as f64 / block as f64) * rho;
    }
    inflation.max(1.0)
}

/// The posterior probability of a mechanism change for each target, under a declared prior over
/// the change indicator.
///
/// Per target, the residuals of the two populations (as in [`detect_mechanism_changes_block`])
/// are modelled as Gaussian. With no change both populations share one mean and variance; with a
/// change each has its own. Each (mean, variance) carries the conjugate
/// normal-inverse-gamma prior `mu | s2 ~ N(0, s2)`, `s2 ~ InvGamma(2, v)` with `v` the pooled
/// residual variance (an empirical-Bayes scale, the same under both models), so the marginal
/// likelihoods are closed form and the Bayes factor is exact for that model. The likelihood is
/// tempered by the reciprocal of the Bartlett variance inflation of the residuals' serial
/// dependence at `block` lags (effective-sample deflation). With the declared prior `π` of a
/// change, `P(change | data) = π BF / (π BF + 1 − π)`. The prior is the caller's declaration and
/// is never assumed; the posterior is for the Gaussian model, not model-free.
///
/// # Errors
///
/// As [`detect_mechanism_changes`], and a `prior` outside (0, 1).
pub fn mechanism_change_posterior(
    graph_model: &CompiledCausalModel,
    data: &TabularData,
    query: &MechanismChangeQuery,
    prior: f64,
    block: usize,
    ctx: &ExecutionContext,
) -> Result<Vec<MechanismChangePosterior>, AttributionError> {
    if !(prior > 0.0 && prior < 1.0) {
        return Err(AttributionError::invalid_input(
            "the prior probability of a change lies strictly between 0 and 1",
        ));
    }
    let pairs = residual_pairs(graph_model, data, query, ctx)?;
    Ok(pairs
        .into_iter()
        .map(|(variable, rb, rc)| {
            let statistic = gaussian_lr_statistic(&rb, &rc);
            let inflation = serial_inflation(&rb, &rc, block);
            // Tempering by the reciprocal inflation: the sufficient statistics count as
            // n / inflation effective observations.
            let weight = 1.0 / inflation;
            let part = |xs: &[f64]| {
                let n = xs.len() as f64;
                let mean = xs.iter().sum::<f64>() / n.max(1.0);
                let ss = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>();
                (weight * n, mean, weight * ss)
            };
            let pooled: Vec<f64> = rb.iter().chain(rc.iter()).copied().collect();
            let (n_all, mean_all, ss_all) = part(&pooled);
            let (n_a, mean_a, ss_a) = part(&rb);
            let (n_b, mean_b, ss_b) = part(&rc);
            // The variance prior is centred on the pooled residual variance (an
            // empirical-Bayes scale, the same under both models).
            let beta0 = (ss_all / n_all.max(1.0)).max(f64::MIN_POSITIVE) * (NIG_ALPHA0 - 1.0);
            let log_bayes_factor = nig_log_marginal(n_a, mean_a, ss_a, beta0)
                + nig_log_marginal(n_b, mean_b, ss_b, beta0)
                - nig_log_marginal(n_all, mean_all, ss_all, beta0);
            let log_odds = (prior / (1.0 - prior)).ln() + log_bayes_factor;
            let probability = 1.0 / (1.0 + (-log_odds).exp());
            MechanismChangePosterior {
                variable,
                statistic,
                inflation,
                log_bayes_factor,
                prior,
                probability,
            }
        })
        .collect())
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
mod tests {
    use super::*;
    use antecedent_core::{
        CausalSchemaBuilder, MeasurementSpec, PopulationSelector, RoleHint, SmallRoleSet, ValueType,
    };
    use antecedent_data::column::{Float64Column, ValidityBitmap};
    use antecedent_data::{OwnedColumn, OwnedColumnarStorage};
    use antecedent_graph::{Dag, DenseNodeId};

    #[test]
    fn detects_y_shift_not_necessarily_x() {
        let n = 80usize;
        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "x",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "y",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        let schema = b.build().unwrap();
        let mut xv = Vec::new();
        let mut yv = Vec::new();
        for i in 0..n {
            let x = (i % 40) as f64 * 0.1;
            xv.push(x);
            yv.push(if i < 40 { 1.0 + 2.0 * x } else { 6.0 + 2.0 * x });
        }
        let validity = ValidityBitmap::all_valid(n);
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(0), Arc::from(xv), validity.clone())
                    .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(1), Arc::from(yv), validity).unwrap(),
            ),
        ];
        let data =
            TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());
        let mut g = Dag::with_variables(2);
        g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let model = CompiledCausalModel::compile(g).unwrap();
        let q = MechanismChangeQuery::new(
            [VariableId::from_raw(0), VariableId::from_raw(1)],
            PopulationSelector::TimeRange { start: 0, end: 40 },
            PopulationSelector::TimeRange { start: 40, end: 80 },
            0.05,
            10,
        );
        let dets = detect_mechanism_changes(
            &model,
            &data,
            &q,
            MechanismChangeMethod::MeanDiff,
            &ExecutionContext::for_tests(1),
        )
        .unwrap();
        let y = dets.iter().find(|d| d.variable == VariableId::from_raw(1)).unwrap();
        assert!(y.changed, "y should be flagged changed: {y:?}");
    }

    fn two_period_data() -> (CompiledCausalModel, TabularData) {
        let n = 80usize;
        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "x",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "y",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        let schema = b.build().unwrap();
        let mut xv = Vec::new();
        let mut yv = Vec::new();
        for i in 0..n {
            let x = (i % 40) as f64 * 0.1;
            xv.push(x);
            yv.push(if i < 40 { 1.0 + 2.0 * x } else { 6.0 + 2.0 * x });
        }
        let validity = ValidityBitmap::all_valid(n);
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(0), Arc::from(xv), validity.clone())
                    .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(1), Arc::from(yv), validity).unwrap(),
            ),
        ];
        let data =
            TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());
        let mut g = Dag::with_variables(2);
        g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let model = CompiledCausalModel::compile(g).unwrap();
        (model, data)
    }

    #[test]
    fn parent_shift_check_that_cannot_run_refuses() {
        let (model, data) = two_period_data();
        // Non-finite parent value in the comparison population: the parent mean-difference
        // test cannot be evaluated, so the covariate-shift guard must refuse, not pass.
        let mut xv = data.float64_values(VariableId::from_raw(0)).unwrap();
        xv[50] = f64::NAN;
        let yv = data.float64_values(VariableId::from_raw(1)).unwrap();
        let n = xv.len();
        let validity = ValidityBitmap::all_valid(n);
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(0), Arc::from(xv), validity.clone())
                    .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(1), Arc::from(yv), validity).unwrap(),
            ),
        ];
        let schema = data.schema().clone();
        let data =
            TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());
        let q = MechanismChangeQuery::new(
            [VariableId::from_raw(1)],
            PopulationSelector::TimeRange { start: 0, end: 40 },
            PopulationSelector::TimeRange { start: 40, end: 80 },
            0.05,
            10,
        );
        let err = detect_mechanism_changes(
            &model,
            &data,
            &q,
            MechanismChangeMethod::MeanDiff,
            &ExecutionContext::for_tests(1),
        )
        .unwrap_err();
        assert!(
            matches!(&err, AttributionError::InvalidInput { message }
                if message.contains("parent-covariate shift check")),
            "{err:?}"
        );
    }

    #[test]
    fn kernel_two_sample_flags_y_shift() {
        let (model, data) = two_period_data();
        let q = MechanismChangeQuery::new(
            [VariableId::from_raw(0), VariableId::from_raw(1)],
            PopulationSelector::TimeRange { start: 0, end: 40 },
            PopulationSelector::TimeRange { start: 40, end: 80 },
            0.05,
            10,
        );
        let dets = detect_mechanism_changes(
            &model,
            &data,
            &q,
            MechanismChangeMethod::KernelTwoSample,
            &ExecutionContext::for_tests(1),
        )
        .unwrap();
        let y = dets.iter().find(|d| d.variable == VariableId::from_raw(1)).unwrap();
        assert!(y.changed, "y should be flagged changed: {y:?}");
        assert_eq!(&*y.method, "kernel_two_sample");
        // The resolution of a permutation p-value is reported with it: 999 permutations
        // give a floor of exactly 1/1000, and a raw p-value cannot sit below it.
        assert_eq!(y.n_permutations, Some(999));
        assert!((y.p_value_floor.unwrap() - 0.001).abs() < 1e-15);
        assert!(y.p_value >= 0.001 - 1e-15);
    }

    /// The kernel null draws from the context's attribution stream: it follows the master
    /// seed (reproducibly) and the verdict on a real shift is stable across seeds, with
    /// every p-value inside the same resolution band.
    #[test]
    fn kernel_two_sample_follows_the_context_seed_and_agrees_across_seeds() {
        let (model, data) = two_period_data();
        let q = MechanismChangeQuery::new(
            [VariableId::from_raw(1)],
            PopulationSelector::TimeRange { start: 0, end: 40 },
            PopulationSelector::TimeRange { start: 40, end: 80 },
            0.05,
            10,
        );
        let run = |seed: u64| {
            detect_mechanism_changes(
                &model,
                &data,
                &q,
                MechanismChangeMethod::KernelTwoSample,
                &ExecutionContext::for_tests(seed),
            )
            .unwrap()
            .remove(0)
        };
        let again = run(1);
        assert_eq!(again.p_value.to_bits(), run(1).p_value.to_bits());
        for seed in 1..=6u64 {
            let d = run(seed);
            assert!(d.changed, "seed {seed}: {d:?}");
            assert!(d.p_value >= d.p_value_floor.unwrap() - 1e-15);
            assert!(d.p_value <= 0.01, "seed {seed}: p={}", d.p_value);
            assert_eq!(d.statistic.to_bits(), again.statistic.to_bits(), "statistic is seed-free");
        }
    }

    #[test]
    fn change_point_flags_y_shift() {
        let (model, data) = two_period_data();
        let q = MechanismChangeQuery::new(
            [VariableId::from_raw(0), VariableId::from_raw(1)],
            PopulationSelector::TimeRange { start: 0, end: 40 },
            PopulationSelector::TimeRange { start: 40, end: 80 },
            0.05,
            10,
        );
        let dets = detect_mechanism_changes(
            &model,
            &data,
            &q,
            MechanismChangeMethod::ChangePoint,
            &ExecutionContext::for_tests(1),
        )
        .unwrap();
        let y = dets.iter().find(|d| d.variable == VariableId::from_raw(1)).unwrap();
        assert!(y.changed, "y should be flagged changed: {y:?}");
        assert_eq!(&*y.method, "change_point");
    }

    /// A 1-row comparison population has zero within-population variance by
    /// construction. Before the fix, `mechanism_change` resolved populations directly
    /// (bypassing `prep::resolve_population_pair`'s emptiness check and imposing no
    /// minimum row count), and `LikelihoodRatio` reached `gaussian_segment_lr`'s
    /// `v1 == 0.0` branch, which reported `changed: true, p_value: 0.0, statistic: inf`
    /// — the strongest possible "changed" claim, manufactured from a single observation.
    /// The fix refuses any population below a documented minimum before fitting anything.
    #[test]
    fn refuses_single_row_comparison_population() {
        let (model, data) = two_period_data();
        let q = MechanismChangeQuery::new(
            [VariableId::from_raw(1)],
            PopulationSelector::TimeRange { start: 0, end: 40 },
            PopulationSelector::Rows(std::sync::Arc::from([17usize])),
            0.05,
            10,
        );
        for method in [MechanismChangeMethod::LikelihoodRatio, MechanismChangeMethod::ChangePoint] {
            let result = detect_mechanism_changes(
                &model,
                &data,
                &q,
                method,
                &ExecutionContext::for_tests(1),
            );
            assert!(
                result.is_err(),
                "a 1-row comparison population must be refused, not scored as a \
                 definitive mechanism-change claim (method={method:?}): {result:?}"
            );
        }
    }

    /// Same defect, reached via a constant (rather than merely small) comparison
    /// population: several rows, but every value bit-identical, so the segment still has
    /// zero variance. This is caught by the same minimum-row gate here (below
    /// `MIN_POPULATION_ROWS`); a comparison population at or above the minimum but still
    /// exactly constant is caught downstream by `gaussian_segment_lr`'s own refusal
    /// (`antecedent_stats::divergence::residual_lr_refuses_single_row_or_constant_segments`).
    #[test]
    fn refuses_undersized_comparison_population() {
        let (model, data) = two_period_data();
        let q = MechanismChangeQuery::new(
            [VariableId::from_raw(1)],
            PopulationSelector::TimeRange { start: 0, end: 40 },
            PopulationSelector::Rows(std::sync::Arc::from([10usize, 11, 12])),
            0.05,
            10,
        );
        let result = detect_mechanism_changes(
            &model,
            &data,
            &q,
            MechanismChangeMethod::LikelihoodRatio,
            &ExecutionContext::for_tests(1),
        );
        assert!(result.is_err(), "a 3-row comparison population must be refused: {result:?}");
    }

    /// `x`'s range genuinely differs by 5 units between the two populations here — exactly
    /// the material parent-covariate shift `parent_covariate_shift_detected` now refuses
    /// on, so this documents the current, honest behavior (refusal) rather than the
    /// original claim (a same-mechanism verdict). The original pass was itself an artifact
    /// of this test's near-deterministic 0.01-amplitude periodic "noise": with that little
    /// noise, the baseline fit has almost no parameter uncertainty to be wrong about, so
    /// the covariate shift never actually caused a false positive here even before the
    /// refusal existed — see `mean_diff_covariate_shift_is_calibrated_or_refused` for the
    /// case (realistic Gaussian noise) where it does.
    #[test]
    fn mean_diff_refuses_under_material_covariate_shift_even_with_same_mechanism() {
        let n = 80usize;
        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "x",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "y",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        let schema = b.build().unwrap();
        let mut xv = Vec::new();
        let mut yv = Vec::new();
        for i in 0..n {
            // Same y = 1 + 2x + noise; x ranges differ → different marginal y mean, same mechanism.
            let x = if i < 40 { (i % 40) as f64 * 0.1 } else { 5.0 + (i % 40) as f64 * 0.1 };
            xv.push(x);
            yv.push(1.0 + 2.0 * x + 0.01 * ((i % 7) as f64 - 3.0));
        }
        let validity = ValidityBitmap::all_valid(n);
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(0), Arc::from(xv), validity.clone())
                    .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(1), Arc::from(yv), validity).unwrap(),
            ),
        ];
        let data =
            TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());
        let mut g = Dag::with_variables(2);
        g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let model = CompiledCausalModel::compile(g).unwrap();
        let q = MechanismChangeQuery::new(
            [VariableId::from_raw(1)],
            PopulationSelector::TimeRange { start: 0, end: 40 },
            PopulationSelector::TimeRange { start: 40, end: 80 },
            0.05,
            10,
        );
        let result = detect_mechanism_changes(
            &model,
            &data,
            &q,
            MechanismChangeMethod::MeanDiff,
            &ExecutionContext::for_tests(1),
        );
        assert!(
            result.is_err(),
            "a materially shifted parent covariate should be refused, not scored: {result:?}"
        );
    }

    /// Null split of a homogeneous SCM: residual `MeanDiff` Type I smoke.
    /// Not a full calibration gate — only checks flag rate stays near α.
    #[test]
    fn mean_diff_null_split_type_i_smoke() {
        let n = 200usize;
        let mid = n / 2;
        let mut flags = 0usize;
        let trials = 20usize;
        for trial in 0..trials {
            let mut rng = ExecutionContext::for_tests(0x4C01u64.wrapping_add(trial as u64))
                .rng
                .stream_for(StreamDomain::Attribution, 0x7E51);
            let mut b = CausalSchemaBuilder::new();
            b.add_variable(
                "x",
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::Context),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
            b.add_variable(
                "y",
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
            let schema = b.build().unwrap();
            let mut xv = Vec::with_capacity(n);
            let mut yv = Vec::with_capacity(n);
            for _ in 0..n {
                let x = antecedent_kernels::standard_normal(&mut rng);
                let e = antecedent_kernels::standard_normal(&mut rng);
                xv.push(x);
                yv.push(1.0 + 2.0 * x + 0.25 * e);
            }
            let validity = ValidityBitmap::all_valid(n);
            let cols = vec![
                OwnedColumn::Float64(
                    Float64Column::new(VariableId::from_raw(0), Arc::from(xv), validity.clone())
                        .unwrap(),
                ),
                OwnedColumn::Float64(
                    Float64Column::new(VariableId::from_raw(1), Arc::from(yv), validity).unwrap(),
                ),
            ];
            let data =
                TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());
            let mut g = Dag::with_variables(2);
            g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
            let model = CompiledCausalModel::compile(g).unwrap();
            let q = MechanismChangeQuery::new(
                [VariableId::from_raw(1)],
                PopulationSelector::TimeRange { start: 0, end: mid },
                PopulationSelector::TimeRange { start: mid, end: n },
                0.05,
                10,
            );
            let dets = detect_mechanism_changes(
                &model,
                &data,
                &q,
                MechanismChangeMethod::MeanDiff,
                &ExecutionContext::for_tests(1),
            )
            .unwrap();
            let y = dets.iter().find(|d| d.variable == VariableId::from_raw(1)).unwrap();
            if y.changed {
                flags += 1;
            }
        }
        // α=0.05 → expect ~1 flag in 20; allow a few extras for smoke (not a calibration gate).
        assert!(
            flags <= 4,
            "null-split Type I smoke: MeanDiff flagged {flags}/{trials} (want ≤4 at α=0.05)"
        );
    }

    /// Family-wise Type I inflation across several truly-unchanged targets, and control via the
    /// default Benjamini–Hochberg correction (Defect A).
    ///
    /// `mean_diff_null_split_type_i_smoke` above only checks the *per-target* flag rate with a
    /// single target, so it cannot see the family-wise effect: with `m` independent null targets
    /// each judged against α=0.05 independently, `P(>=1 false flag) = 1 - 0.95^m`, which is
    /// ≈40% at m=10. This test builds `m` independent, unrelated mechanisms (no true baseline →
    /// comparison change in any of them) and compares the family-wise flag rate (>=1 of the `m`
    /// targets flagged) with `correction: None` against the corrected default
    /// `detect_mechanism_changes`.
    #[test]
    #[allow(
        clippy::too_many_lines,
        reason = "one fixture exercised end to end; later assertions read locals built above them"
    )]
    fn mean_diff_null_split_family_wise_type_i_rate() {
        let m = 10usize;
        let n = 120usize;
        let mid = n / 2;
        let trials = 150usize;
        let alpha = 0.05;

        let mut uncorrected_family_flags = 0usize;
        let mut corrected_family_flags = 0usize;

        for trial in 0..trials {
            let mut rng = ExecutionContext::for_tests(0x9A11u64.wrapping_add(trial as u64))
                .rng
                .stream_for(StreamDomain::Attribution, 0x51EE);
            let mut b = CausalSchemaBuilder::new();
            for k in 0..m {
                b.add_variable(
                    format!("x{k}"),
                    ValueType::Continuous,
                    SmallRoleSet::from_hint(RoleHint::Context),
                    None,
                    None,
                    MeasurementSpec::default(),
                )
                .unwrap();
                b.add_variable(
                    format!("y{k}"),
                    ValueType::Continuous,
                    SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
                    None,
                    None,
                    MeasurementSpec::default(),
                )
                .unwrap();
            }
            let schema = b.build().unwrap();
            let mut cols = Vec::with_capacity(2 * m);
            for k in 0..m {
                let mut xv = Vec::with_capacity(n);
                let mut yv = Vec::with_capacity(n);
                for _ in 0..n {
                    let x = antecedent_kernels::standard_normal(&mut rng);
                    let e = antecedent_kernels::standard_normal(&mut rng);
                    xv.push(x);
                    yv.push(1.0 + 2.0 * x + 0.25 * e);
                }
                let validity = ValidityBitmap::all_valid(n);
                cols.push(OwnedColumn::Float64(
                    Float64Column::new(
                        VariableId::from_raw((2 * k) as u32),
                        Arc::from(xv),
                        validity.clone(),
                    )
                    .unwrap(),
                ));
                cols.push(OwnedColumn::Float64(
                    Float64Column::new(
                        VariableId::from_raw((2 * k + 1) as u32),
                        Arc::from(yv),
                        validity,
                    )
                    .unwrap(),
                ));
            }
            let data =
                TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());
            let mut g = Dag::with_variables((2 * m) as u32);
            for k in 0..m {
                g.insert_directed(
                    DenseNodeId::from_raw((2 * k) as u32),
                    DenseNodeId::from_raw((2 * k + 1) as u32),
                )
                .unwrap();
            }
            let model = CompiledCausalModel::compile(g).unwrap();
            let targets: Vec<VariableId> =
                (0..m).map(|k| VariableId::from_raw((2 * k + 1) as u32)).collect();
            let q = MechanismChangeQuery::new(
                targets,
                PopulationSelector::TimeRange { start: 0, end: mid },
                PopulationSelector::TimeRange { start: mid, end: n },
                alpha,
                m,
            );

            let uncorrected = detect_mechanism_changes_with_correction(
                &model,
                &data,
                &q,
                MechanismChangeMethod::MeanDiff,
                &ExecutionContext::for_tests(1),
                None,
            )
            .unwrap();
            if uncorrected.iter().any(|d| d.changed) {
                uncorrected_family_flags += 1;
            }

            let corrected = detect_mechanism_changes(
                &model,
                &data,
                &q,
                MechanismChangeMethod::MeanDiff,
                &ExecutionContext::for_tests(1),
            )
            .unwrap();
            if corrected.iter().any(|d| d.changed) {
                corrected_family_flags += 1;
            }
        }

        let uncorrected_rate = uncorrected_family_flags as f64 / trials as f64;
        let corrected_rate = corrected_family_flags as f64 / trials as f64;
        // Expected uncorrected family-wise rate ≈ 1 - 0.95^10 ≈ 0.40; allow slop for MC noise.
        assert!(
            uncorrected_rate > 0.20,
            "expected inflated uncorrected family-wise rate (~0.40 expected), got \
             {uncorrected_rate} ({uncorrected_family_flags}/{trials})"
        );
        assert!(
            corrected_rate < uncorrected_rate - 0.10,
            "BH-corrected family-wise rate ({corrected_rate}, {corrected_family_flags}/{trials}) \
             should be well below the uncorrected rate ({uncorrected_rate})"
        );
        assert!(
            corrected_rate <= 0.15,
            "BH-corrected family-wise rate should stay near α=0.05 under the global null, got \
             {corrected_rate} ({corrected_family_flags}/{trials})"
        );
    }

    /// Build one draw of `y = 1 + 2x + N(0,1)` with the *same* mechanism in both
    /// populations, but `x` shifted between them: `x_baseline ~ N(0,1)`,
    /// `x_comparison ~ N(shift,1)`. Baseline rows come first.
    fn covariate_shift_dataset(
        n_b: usize,
        n_c: usize,
        shift: f64,
        rng: &mut CausalRng,
    ) -> (CompiledCausalModel, TabularData) {
        let n = n_b + n_c;
        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "x",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "y",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        let schema = b.build().unwrap();
        let mut xv = Vec::with_capacity(n);
        let mut yv = Vec::with_capacity(n);
        for i in 0..n {
            let mean_x = if i < n_b { 0.0 } else { shift };
            let x = mean_x + antecedent_kernels::standard_normal(rng);
            let e = antecedent_kernels::standard_normal(rng);
            xv.push(x);
            yv.push(1.0 + 2.0 * x + e);
        }
        let validity = ValidityBitmap::all_valid(n);
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(0), Arc::from(xv), validity.clone())
                    .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(1), Arc::from(yv), validity).unwrap(),
            ),
        ];
        let data =
            TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());
        let mut g = Dag::with_variables(2);
        g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let model = CompiledCausalModel::compile(g).unwrap();
        (model, data)
    }

    /// Empirical Type-I (false-positive) rate of a residual two-sample method over
    /// `trials` independent draws of [`covariate_shift_dataset`], at nominal `alpha`.
    /// Trials on which the call is refused (material covariate shift detected) are
    /// excluded from the denominator and counted separately, since a refusal is not a
    /// false positive — it is the honest alternative to one.
    fn covariate_shift_outcome_rates(
        method: MechanismChangeMethod,
        shift: f64,
        trials: usize,
        alpha: f64,
        seed_base: u64,
    ) -> (f64, f64) {
        let n_b = 60usize;
        let n_c = 60usize;
        let mut flags = 0usize;
        let mut refusals = 0usize;
        let mut scored = 0usize;
        for trial in 0..trials {
            let mut gen_rng =
                ExecutionContext::for_tests(seed_base.wrapping_add(trial as u64)).rng.stream(0xC0);
            let (model, data) = covariate_shift_dataset(n_b, n_c, shift, &mut gen_rng);
            let q = MechanismChangeQuery::new(
                [VariableId::from_raw(1)],
                PopulationSelector::TimeRange { start: 0, end: n_b },
                PopulationSelector::TimeRange { start: n_b, end: n_b + n_c },
                alpha,
                10,
            );
            match detect_mechanism_changes_with_correction(
                &model,
                &data,
                &q,
                method,
                &ExecutionContext::for_tests(1),
                None,
            ) {
                Ok(dets) => {
                    scored += 1;
                    let y = dets.iter().find(|d| d.variable == VariableId::from_raw(1)).unwrap();
                    if y.changed {
                        flags += 1;
                    }
                }
                Err(_) => refusals += 1,
            }
        }
        let flag_rate = if scored == 0 { 0.0 } else { flags as f64 / scored as f64 };
        let refusal_rate = refusals as f64 / trials as f64;
        (flag_rate, refusal_rate)
    }

    /// `attr-design-state-2`: residual two-sample tests compared in-sample baseline
    /// residuals against out-of-sample comparison residuals. Under an *unchanged*
    /// mechanism but a shifted parent distribution, the baseline model's own
    /// extrapolation error inflates only the comparison side, so the null is rejected far
    /// above the nominal rate — the reviewer's Python simulation found ≈12% at a 1σ
    /// shift, ≈40% at 3σ, ≈61% at 5σ, against a nominal α=0.05.
    ///
    /// This mirrors that simulation in-crate: `y = 1 + 2x + N(0,1)` with the identical
    /// mechanism in both populations, only `x`'s mean shifted. At `shift=0` (no covariate
    /// difference), the parent-shift guard essentially never fires and the cross-fitted
    /// baseline keeps the flag rate near nominal. At `shift=3` and `shift=5`, no residual
    /// reconstruction can make a same-distribution two-sample test calibrated against the
    /// resulting shared bias (see the comment in
    /// `detect_mechanism_changes_with_correction`), so the call is refused instead of
    /// reporting an anti-conservative p-value: refusals should dominate, not false flags.
    #[test]
    fn mean_diff_covariate_shift_is_calibrated_or_refused() {
        let alpha = 0.05;
        let trials = 300usize;
        let (flag_rate, refusal_rate) = covariate_shift_outcome_rates(
            MechanismChangeMethod::MeanDiff,
            0.0,
            trials,
            alpha,
            0xC0_1B_A5_E0,
        );
        assert!(
            refusal_rate < 0.05,
            "shift=0 should almost never trigger the covariate-shift refusal, got {refusal_rate}"
        );
        assert!(
            flag_rate < 0.15,
            "MeanDiff Type-I rate at shift=0 should stay close to nominal α={alpha}, got {flag_rate}"
        );
        for shift in [3.0, 5.0] {
            let (flag_rate, refusal_rate) = covariate_shift_outcome_rates(
                MechanismChangeMethod::MeanDiff,
                shift,
                trials,
                alpha,
                0xC0_1B_A5_E0,
            );
            assert!(
                refusal_rate > 0.85,
                "MeanDiff at shift={shift} should be refused, not scored, on almost every \
                 trial (a residual two-sample test cannot be calibrated against this \
                 covariate shift): refusal_rate={refusal_rate}, flag_rate={flag_rate}"
            );
        }
    }

    /// Same reproduction as above, for `LikelihoodRatio` (the other test the finding
    /// names explicitly).
    #[test]
    fn likelihood_ratio_covariate_shift_is_calibrated_or_refused() {
        let alpha = 0.05;
        let trials = 300usize;
        let (flag_rate, refusal_rate) = covariate_shift_outcome_rates(
            MechanismChangeMethod::LikelihoodRatio,
            0.0,
            trials,
            alpha,
            0x11_4E_11_D0,
        );
        assert!(
            refusal_rate < 0.05,
            "shift=0 should almost never trigger the covariate-shift refusal, got {refusal_rate}"
        );
        assert!(
            flag_rate < 0.15,
            "LikelihoodRatio Type-I rate at shift=0 should stay close to nominal α={alpha}, \
             got {flag_rate}"
        );
        for shift in [3.0, 5.0] {
            let (flag_rate, refusal_rate) = covariate_shift_outcome_rates(
                MechanismChangeMethod::LikelihoodRatio,
                shift,
                trials,
                alpha,
                0x11_4E_11_D0,
            );
            assert!(
                refusal_rate > 0.85,
                "LikelihoodRatio at shift={shift} should be refused, not scored, on almost \
                 every trial: refusal_rate={refusal_rate}, flag_rate={flag_rate}"
            );
        }
    }

    /// Defect B: `residuals()` must thread a properly seeded RNG into posterior noise recovery
    /// (`infer_noise_column_rng`) rather than resetting to a fixed `CausalRng::from_seed(0)` on
    /// every call (the old `infer_noise_column` shim).
    ///
    /// A `Discrete`-family target's noise-inference mode is always `Posterior` (see
    /// `infer_noise_column_rng` in `antecedent-model/src/mechanism.rs`), so its residual draws
    /// are genuinely a function of the RNG stream, not just of the data. This test forces
    /// `MechanismRegistry::standard()` to fit `y` as `Discrete` (by giving it few distinct
    /// values, since family eligibility is driven by data cardinality via `is_low_cardinality`,
    /// not by the declared `ValueType`) and then runs `detect_mechanism_changes` twice under two
    /// different `ExecutionContext` seeds. Post-fix, the seed is threaded through to the
    /// posterior draws, so the resulting `MeanDiff` statistic differs between the two seeds.
    /// Pre-fix, both calls silently reset to seed 0 regardless of `ExecutionContext`, so the
    /// statistic is bit-identical across seeds — the RNG parameter is a no-op for any
    /// `Posterior`-mode family.
    ///
    /// Per the task brief: this does **not** attempt to show a flipped `changed` verdict (the
    /// within-population draws stay i.i.d. conditioned on whichever seed is used, and the
    /// two-sample tests treat samples as unordered sets, so a verdict flip could not be
    /// constructed). It demonstrates the narrower, honest claim that the execution context's
    /// seed is actually being honored after the fix and was being silently ignored before it.
    #[test]
    fn discrete_target_residuals_depend_on_execution_context_seed() {
        let n = 60usize;
        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "x",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "y",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        let schema = b.build().unwrap();
        // x has 4 distinct categories, y has 2 — both well under the is_low_cardinality(_, 8)
        // threshold that routes the target to MechanismRegistry::standard().discrete =
        // [Discrete, Constant], and the x >= 2 split gives Discrete a real edge over Constant.
        let mut xv = Vec::with_capacity(n);
        let mut yv = Vec::with_capacity(n);
        for i in 0..n {
            let x = (i % 4) as f64;
            xv.push(x);
            yv.push(if x >= 2.0 { 1.0 } else { 0.0 });
        }
        let validity = ValidityBitmap::all_valid(n);
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(0), Arc::from(xv), validity.clone())
                    .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(1), Arc::from(yv), validity).unwrap(),
            ),
        ];
        let data =
            TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());
        let mut g = Dag::with_variables(2);
        g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let model = CompiledCausalModel::compile(g).unwrap();
        let q = MechanismChangeQuery::new(
            [VariableId::from_raw(1)],
            PopulationSelector::TimeRange { start: 0, end: n },
            PopulationSelector::TimeRange { start: 0, end: n },
            0.05,
            10,
        );

        let d1 = detect_mechanism_changes(
            &model,
            &data,
            &q,
            MechanismChangeMethod::MeanDiff,
            &ExecutionContext::for_tests(1),
        )
        .unwrap();
        let d2 = detect_mechanism_changes(
            &model,
            &data,
            &q,
            MechanismChangeMethod::MeanDiff,
            &ExecutionContext::for_tests(2),
        )
        .unwrap();
        let s1 = d1.iter().find(|d| d.variable == VariableId::from_raw(1)).unwrap().statistic;
        let s2 = d2.iter().find(|d| d.variable == VariableId::from_raw(1)).unwrap().statistic;
        // Separated by a real margin, not just `!=`: posterior-sampled noise under two
        // different seeds should move the statistic materially, and an exact-inequality
        // check would also pass on a one-ULP difference.
        assert!(
            (s1 - s2).abs() > 1e-9,
            "Discrete-family residual statistic should depend on the ExecutionContext seed \
             (s1={s1}, s2={s2}); identical values mean the RNG is being ignored"
        );
    }
}

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
