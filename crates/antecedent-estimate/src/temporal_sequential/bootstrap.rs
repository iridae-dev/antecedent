use super::{
    Arc, AssumptionSet, EffectEstimate, EstimationError, ExecutionContext, LeastSquaresWorkspace,
    OverlapPolicy, SEQUENTIAL_BLOCK_STREAM, SequentialSetup,
};

/// The frequentist estimate: circular blocks of consecutive lag-aligned rows (every row keeps
/// its intact unfolded window), every mechanism refit on the same rows, and the replicate SD
/// scaled by the circular-Bartlett fixed-b factor and the kernel-bias factor of the contrast
/// influence, as on the single-window path ([`crate::temporal_block`]). Blocks are at least
/// the unfolded span and lengthen when the contrast's estimating score, or any mechanism's
/// normal-equation score, is persistently dependent. The influence and block length are
/// returned on the estimate so callers can report the resampling geometry without rebuilding
/// the design. The lengthening refits every mechanism for its scores and scans each; without
/// replicates no interval is published and the rule length is returned instead.
#[allow(clippy::cast_possible_truncation, reason = "the maximum lag is a small u32 history length")]
pub(super) fn block_bootstrap_estimate(
    setup: &SequentialSetup,
    assumptions: AssumptionSet,
    bootstrap_replicates: u32,
    ctx: &ExecutionContext,
    ls_ws: &mut LeastSquaresWorkspace,
) -> Result<EffectEstimate, EstimationError> {
    let full = setup.fit_full(ls_ws);
    let coefficients = setup.coefficients_of(&full)?;
    let point = setup.propagate(&coefficients, &setup.factual, None);
    let influence = setup.influence_from(&coefficients);
    let resampled_rows = setup.window_complete.as_ref().map_or(setup.n, Vec::len);
    let block_length = match &setup.window_complete {
        // On a series with gaps the scores of the compacted rows are not consecutive, so the
        // dependence-adaptive length is not defined; the structural rule stands, on the
        // original window axis.
        Some(_) => {
            antecedent_data::circular_block_length(setup.max_lag as usize + 1, resampled_rows)
        }
        None if bootstrap_replicates > 0 => {
            setup.dependence_block_length_with(influence.as_deref(), &full)
        }
        None => antecedent_data::circular_block_length(setup.max_lag as usize + 1, setup.n),
    };
    let influence_targets: Vec<&[f64]> = influence.as_deref().into_iter().collect();
    let boot = match &setup.window_complete {
        Some(complete) => crate::temporal_block::gapped_row_block_bootstrap(
            complete,
            block_length,
            bootstrap_replicates,
            SEQUENTIAL_BLOCK_STREAM,
            ctx,
            |rows| setup.evaluate(Some(rows), ls_ws).ok().map(|value| vec![value]),
        ),
        None => crate::temporal_block::row_block_bootstrap_vec(
            setup.n,
            block_length,
            bootstrap_replicates,
            SEQUENTIAL_BLOCK_STREAM,
            ctx,
            |rows| setup.evaluate(Some(rows), ls_ws).ok().map(|value| vec![value]),
        )
        .with_kernel_bias(&influence_targets),
    };
    boot.ensure_admitted()?;
    let kernel_bias = boot.kernel_bias;
    let se = boot.se_result(0);
    let mut effect = EffectEstimate::from_parts(
        point,
        f64::NAN,
        crate::adjustment::BootstrapAccounting {
            se: se.se,
            replicates_ok: (bootstrap_replicates > 0).then_some(se.replicates_ok),
            replicates_failed: (bootstrap_replicates > 0).then_some(se.replicates_failed),
            cancelled: ctx.cancellation.is_cancelled(),
            early_stopped: false,
        },
        assumptions,
        OverlapPolicy::ExplicitOverride,
        None,
        None,
    );
    effect.scores.influence = influence.map(Arc::from);
    effect.diagnostics.sampling_weights = setup.sampling_weights;
    effect.uncertainty.block.resampling = Some(crate::adjustment::BlockResampling {
        block_length,
        rows: resampled_rows,
        kernel_bias,
    });
    Ok(effect)
}
