//! Caller-owned leases for the numerical buffers of contextual reliability work.

use antecedent_core::ExecutionContext;
use antecedent_core::execution::{AdmissionError, AdmissionLease};

use super::{BatchPlan, Functional, ReliabilityError, WithinStreamDependence};

pub(super) fn reserve<T>(
    ctx: &ExecutionContext,
    count: usize,
) -> Result<AdmissionLease, ReliabilityError> {
    let count = u64::try_from(count)
        .map_err(|_| ReliabilityError::Admission { error: AdmissionError::Overflow })?;
    let width = u64::try_from(std::mem::size_of::<T>())
        .map_err(|_| ReliabilityError::Admission { error: AdmissionError::Overflow })?;
    ctx.acquire_admission_product(0, count, width).map_err(|error| match error {
        AdmissionError::Cancelled => ReliabilityError::Cancelled,
        error => ReliabilityError::Admission { error },
    })
}

fn add(a: usize, b: usize) -> Result<usize, ReliabilityError> {
    a.checked_add(b).ok_or(ReliabilityError::Admission { error: AdmissionError::Overflow })
}

pub(super) fn product(a: usize, b: usize) -> Result<usize, ReliabilityError> {
    a.checked_mul(b).ok_or(ReliabilityError::Admission { error: AdmissionError::Overflow })
}

/// Peak live f64 buffers in `estimate`: influence series (plus a signed CVaR
/// copy), or sorted quantile plus batch quantiles and one sorted batch.
pub(super) fn estimate_slots(
    n: usize,
    dependence: WithinStreamDependence,
    plan: BatchPlan,
    functional: &Functional,
) -> Result<usize, ReliabilityError> {
    let batches =
        if dependence == WithinStreamDependence::Autocorrelated { plan.batches(n)? } else { 0 };
    let base = match functional {
        Functional::Cvar { .. } => product(n, 2)?,
        _ => n,
    };
    let batch_sort = if batches != 0 && matches!(functional, Functional::Quantile { .. }) {
        n / batches
    } else {
        0
    };
    add(add(base, batches)?, batch_sort)
}

/// Peak live buffers of the paired phase, compared with the two sequential
/// marginal estimates. These are the exact existing influence/quantile vectors.
pub(super) fn difference_slots(
    n: usize,
    dependence: WithinStreamDependence,
    plan: BatchPlan,
    functional: &Functional,
) -> Result<usize, ReliabilityError> {
    let marginal = estimate_slots(n, dependence, plan, functional)?;
    let paired = if matches!(functional, Functional::Quantile { .. }) {
        let batches = plan.batches(n)?;
        add(product(batches, 2)?, n / batches)?.max(product(batches, 3)?)
    } else {
        let batches =
            if dependence == WithinStreamDependence::Autocorrelated { plan.batches(n)? } else { 0 };
        add(product(n, 3)?, batches)?
    };
    Ok(marginal.max(paired))
}
