//! Work-budget and cancellation preflight for HMC.

use antecedent_core::ExecutionContext;

use crate::backend::{BayesDesignRef, BayesFitOptions};
use crate::error::ProbError;
use crate::prior::PriorSet;

use super::{HmcOptions, check_hmc_cancelled};

pub(super) fn hmc_has_absolute_scale(prior: &PriorSet) -> bool {
    prior.restrictions.iter().any(|restriction| {
        restriction
            .id
            .strip_prefix(crate::prior::ABSOLUTE_COEFFICIENT_SCALE_ID)
            .is_some_and(|suffix| suffix.starts_with(':'))
    })
}

/// Preflight every chain, diagnostic and prior setup before data-sized work.
pub(super) fn admit_hmc_work(
    design: BayesDesignRef<'_>,
    prior: &PriorSet,
    opts: &BayesFitOptions,
    hmc: HmcOptions,
    ctx: Option<&ExecutionContext>,
) -> Result<(), ProbError> {
    if ctx.is_some_and(|ctx| ctx.cancellation.is_cancelled()) {
        return Err(ProbError::Cancelled { stage: "hmc.cancelled" });
    }
    check_hmc_cancelled()?;
    let total_transitions = hmc
        .n_warmup
        .checked_add(opts.n_draws)
        .ok_or(ProbError::Shape { message: "HMC transition count overflow" })?;
    let limit = ctx.map_or_else(
        antecedent_core::numerical_admission::hmc_work_limit,
        ExecutionContext::hmc_work_limit,
    );
    let work = (|| {
        let rows = u64::try_from(design.nrows).ok()?;
        let p = u64::try_from(design.ncols).ok()?;
        let cols = p.checked_add(1)?;
        let square = cols.checked_mul(cols)?;
        let sweep = rows.checked_mul(cols)?.checked_add(square)?.checked_add(cols)?;
        let steps = u64::from(hmc.leapfrog_steps).checked_add(1)?;
        let transitions = u64::try_from(total_transitions).ok()?;
        let chains = u64::try_from(hmc.n_chains).ok()?;
        let sampling = sweep.checked_mul(steps)?.checked_mul(transitions)?.checked_mul(chains)?;
        let draws = u64::try_from(opts.n_draws).ok()?;
        let diagnostics =
            draws.checked_mul(draws)?.checked_mul(chains)?.checked_mul(cols)?.checked_mul(3)?;
        // Dense prior precision and absolute-scale OLS can each require a
        // factorization; the latter also builds X'X over all observations.
        let dense = prior.specs.iter().try_fold(0_u64, |total, spec| {
            if let crate::prior::PriorSpec::CoefficientCorrelation(corr) = spec {
                let d = u64::try_from(corr.dim()).ok()?;
                total.checked_add(d.checked_mul(d)?.checked_mul(d)?.checked_mul(3)?)
            } else {
                Some(total)
            }
        })?;
        let prior_metadata = prior
            .restrictions
            .iter()
            .try_fold(0_u64, |sum, r| sum.checked_add(u64::try_from(r.id.len()).ok()?))?
            .checked_add(u64::try_from(prior.specs.len()).ok()?)?;
        let ols = if hmc_has_absolute_scale(prior) && prior.known_residual_variance().is_none() {
            rows.checked_mul(square)?.checked_add(square.checked_mul(cols)?.checked_mul(2)?)?
        } else {
            0
        };
        sampling
            .checked_add(diagnostics)?
            .checked_add(dense)?
            .checked_add(ols)?
            .checked_add(prior_metadata)
    })();
    if work.is_none_or(|work| work > limit) {
        return Err(ProbError::Refused {
            code: antecedent_core::reason_code!("hmc_work_budget"),
            message: format!(
                "{}{}: requested work {} exceeds limit {limit}",
                antecedent_core::reason_code::PREFIX,
                antecedent_core::reason_code!("hmc_work_budget"),
                work.unwrap_or(u64::MAX)
            ),
        });
    }
    Ok(())
}
