//! Block-permutation tests for dependent-series mechanism changes.

use super::{
    Arc, AttributionError, CausalRng, CompiledCausalModel, ExecutionContext, FdrAdjustment,
    KERNEL_STREAM, MIN_BLOCK_PERMUTATIONS, MechanismChangeDetection, MechanismChangeMethod,
    MechanismChangeQuery, PopulationRuns, TabularData, adjust_pvalues, attribution_stream,
    change_point_two_sample, charge_run_validation, classifier_two_sample, gaussian_lr_statistic,
    kernel_two_sample_with_permutations, poll_run_cancellation, residual_pairs, validate_runs,
};

/// Seed of the kernel statistic's bandwidth stream in a dependent-series test: fixed, so the
/// observed and permuted statistics use the same bandwidth rule.
const BLOCK_KERNEL_SEED: u64 = 0xB10C_4E57;

/// The statistic a dependent-series test permutes, by method: the Gaussian likelihood ratio, the
/// absolute mean difference, the kernel MMD² (RBF, median-heuristic bandwidth) or the
/// classifier two-sample statistic, or the known-split change-point likelihood ratio (the
/// statistic only; its chi-square p-value is replaced by the block null).
pub(super) fn block_statistic(
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
    crate::population::reject_mask_for_ordered(data)?;
    detect_mechanism_changes_block_impl(
        graph_model,
        data,
        query,
        method,
        block,
        permutations,
        None,
        None,
        ctx,
    )
}

/// Block-permutation detection with contiguous ranges in each compact population.
/// Ranges must be nonempty, ordered, and partition their population exactly. No block
/// crosses a supplied run boundary, including the boundary between populations.
#[allow(clippy::too_many_arguments, reason = "the inputs describe one attribution calculation")]
pub fn detect_mechanism_changes_block_with_runs(
    graph_model: &CompiledCausalModel,
    data: &TabularData,
    query: &MechanismChangeQuery,
    method: MechanismChangeMethod,
    block: usize,
    permutations: usize,
    baseline_runs: &[std::ops::Range<usize>],
    comparison_runs: &[std::ops::Range<usize>],
    budget: &mut antecedent_core::TemporalBudget,
    ctx: &ExecutionContext,
) -> Result<Vec<MechanismChangeDetection>, AttributionError> {
    detect_mechanism_changes_block_impl(
        graph_model,
        data,
        query,
        method,
        block,
        permutations,
        Some((baseline_runs, comparison_runs)),
        Some(budget),
        ctx,
    )
}

// Segment original parent blocks at every compact run boundary. Each tuple records
// (parent block, run ordinal, pooled start, pooled end).
pub(super) fn partition_permutation_blocks(
    starts: &[usize],
    block: usize,
    n: usize,
    baseline_len: usize,
    baseline: &[std::ops::Range<usize>],
    comparison: &[std::ops::Range<usize>],
    ctx: &ExecutionContext,
) -> Result<Vec<(usize, usize, usize, usize)>, AttributionError> {
    let mut boundaries = Vec::with_capacity(baseline.len() + comparison.len());
    for (i, run) in baseline.iter().enumerate() {
        poll_run_cancellation(ctx, i)?;
        boundaries.push(run.end);
    }
    for (i, run) in comparison.iter().enumerate() {
        poll_run_cancellation(ctx, i)?;
        boundaries.push(baseline_len + run.end);
    }
    let mut segments = Vec::new();
    for (parent, &start) in starts.iter().enumerate() {
        poll_run_cancellation(ctx, parent)?;
        let end = start.saturating_add(block).min(n);
        let mut pos = start;
        let first = boundaries.partition_point(|&boundary| boundary <= pos);
        for (run, &boundary) in boundaries.iter().enumerate().skip(first) {
            poll_run_cancellation(ctx, run)?;
            let stop = end.min(boundary);
            segments.push((parent, run, pos, stop));
            pos = stop;
            if pos == end {
                break;
            }
        }
    }
    Ok(segments)
}

#[allow(
    clippy::too_many_lines,
    reason = "the bounded attribution calculation keeps its checks together"
)]
#[allow(clippy::too_many_arguments, reason = "the inputs describe one attribution calculation")]
fn detect_mechanism_changes_block_impl(
    graph_model: &CompiledCausalModel,
    data: &TabularData,
    query: &MechanismChangeQuery,
    method: MechanismChangeMethod,
    block: usize,
    permutations: usize,
    runs: Option<PopulationRuns<'_>>,
    mut budget: Option<&mut antecedent_core::TemporalBudget>,
    ctx: &ExecutionContext,
) -> Result<Vec<MechanismChangeDetection>, AttributionError> {
    if let (Some(runs), Some(budget)) = (runs, budget.as_deref_mut()) {
        charge_run_validation(runs, budget, ctx)?;
    }
    if permutations < MIN_BLOCK_PERMUTATIONS {
        return Err(AttributionError::invalid_input(
            "a block permutation test needs at least 99 permutations",
        ));
    }
    let alpha = query.significance_level.to_f64();
    let pairs = residual_pairs(graph_model, data, query, ctx)?;
    let max_bytes = budget.as_ref().map(|b| b.limits().max_bytes);
    if let Some(budget) = budget {
        let mut work = 0_u64;
        for (_, rb, rc) in &pairs {
            let n = rb
                .len()
                .checked_add(rc.len())
                .ok_or(AttributionError::Budget { message: "temporal work overflow".into() })?;
            let rounds = permutations
                .checked_add(1)
                .ok_or(AttributionError::Budget { message: "temporal work overflow".into() })?;
            let cost = n
                .checked_mul(rounds)
                .and_then(|v| v.checked_mul((usize::BITS - n.max(1).leading_zeros()) as usize + 1))
                .and_then(|v| u64::try_from(v).ok())
                .ok_or(AttributionError::Budget { message: "temporal work overflow".into() })?;
            work = work
                .checked_add(cost)
                .ok_or(AttributionError::Budget { message: "temporal work overflow".into() })?;
        }
        budget
            .charge_work(work)
            .map_err(|e| AttributionError::Budget { message: e.to_string() })?;
    }
    let raw = ctx.map_indexed(pairs.len(), |i, worker| {
        let (target, rb, rc) = &pairs[i];
        let observed = block_statistic(method, rb, rc)?;
        let seed = attribution_stream(worker, KERNEL_STREAM, target.as_usize() as u64).next_u64();
        let mut rng = CausalRng::from_seed(seed);
        let pooled: Vec<f64> = rb.iter().chain(rc.iter()).copied().collect();
        let n = pooled.len();
        let block = block.clamp(1, n);
        let starts: Vec<usize> = (0..n).step_by(block).collect();
        let mut order: Vec<usize> = (0..starts.len()).collect();
        let mut shuffled = Vec::with_capacity(n);
        let segments = if let Some((baseline, comparison)) = runs {
            validate_runs(baseline, rb.len(), ctx)?;
            validate_runs(comparison, rc.len(), ctx)?;
            let count = starts
                .len()
                .checked_add(baseline.len())
                .and_then(|x| x.checked_add(comparison.len()))
                .ok_or(AttributionError::Budget { message: "run segment count overflow".into() })?;
            let bytes = count
                .checked_mul(std::mem::size_of::<(usize, usize, usize, usize)>() * 4)
                .and_then(|x| u64::try_from(x).ok())
                .ok_or(AttributionError::Budget {
                    message: "run segment allocation overflow".into(),
                })?;
            if max_bytes.is_some_and(|limit| bytes > limit) {
                return Err(AttributionError::Budget { message: "temporal bytes limit".into() });
            }
            let lease = ctx.acquire_admission(0, bytes).map_err(|e| {
                if matches!(e, antecedent_core::execution::AdmissionError::Cancelled) {
                    AttributionError::Cancelled
                } else {
                    AttributionError::Budget { message: e.to_string() }
                }
            })?;
            let segments = partition_permutation_blocks(
                &starts,
                block,
                n,
                rb.len(),
                baseline,
                comparison,
                ctx,
            )?;
            Some((segments, lease))
        } else {
            None
        };
        let mut exceed = 0_usize;
        for replicate in 0..permutations {
            if runs.is_some() && ctx.cancellation.is_cancelled() {
                return Err(AttributionError::Cancelled);
            }
            for k in (1..order.len()).rev() {
                if runs.is_some() {
                    poll_run_cancellation(ctx, k)?;
                }
                #[allow(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "next_f64() is in [0, 1), so the scaled draw is an index at most k"
                )]
                let j = ((rng.next_f64() * (k + 1) as f64) as usize).min(k);
                order.swap(k, j);
            }
            shuffled.clear();
            if let Some((segments, _lease)) = &segments {
                // Shuffle whole original blocks with the primary stream, then insert
                // split-off segments uniformly using independent v2 streams.
                let mut parent_keys = Vec::with_capacity(order.len());
                for i in 0..order.len() {
                    poll_run_cancellation(ctx, i)?;
                    let key = antecedent_core::derive_stream(
                        antecedent_core::derive_stream(
                            antecedent_core::derive_stream(seed, 0x6d65_6368_7061_7232),
                            replicate as u64,
                        ),
                        i as u64,
                    );
                    parent_keys.push(CausalRng::from_seed(key).next_f64());
                }
                parent_keys.sort_by(f64::total_cmp);
                let mut rank = vec![0; order.len()];
                for (i, &parent) in order.iter().enumerate() {
                    poll_run_cancellation(ctx, i)?;
                    rank[parent] = i;
                }
                let mut arranged = Vec::with_capacity(segments.len());
                let mut previous = None;
                for (i, &(parent, run, start, end)) in segments.iter().enumerate() {
                    poll_run_cancellation(ctx, i)?;
                    let key = if previous == Some(parent) {
                        let mut key = antecedent_core::derive_stream(seed, 0x6d65_6368_7365_6732);
                        for value in [replicate, parent, run, start] {
                            key = antecedent_core::derive_stream(key, value as u64);
                        }
                        CausalRng::from_seed(key).next_f64()
                    } else {
                        parent_keys[rank[parent]]
                    };
                    previous = Some(parent);
                    arranged.push((key, parent, run, start, end));
                }
                arranged.sort_by(|a, b| {
                    a.0.total_cmp(&b.0).then(rank[a.1].cmp(&rank[b.1])).then(a.3.cmp(&b.3))
                });
                for (i, (_, _, _, start, end)) in arranged.into_iter().enumerate() {
                    poll_run_cancellation(ctx, i)?;
                    for chunk in pooled[start..end].chunks(256) {
                        if ctx.cancellation.is_cancelled() {
                            return Err(AttributionError::Cancelled);
                        }
                        shuffled.extend_from_slice(chunk);
                    }
                }
            } else {
                for &o in &order {
                    let start = starts[o];
                    shuffled.extend_from_slice(&pooled[start..(start + block).min(n)]);
                }
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
