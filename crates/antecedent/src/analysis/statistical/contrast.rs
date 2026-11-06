//! Paired mean contrasts for statistical transport results.

use super::{
    INTERVAL_NUMERICAL_FAILURE, IntervalWithholdReason, IoError, PERCENTILE_95_MIN_REPLICATES,
    ReplicatePolicy, StatisticalContrast, StatisticalStudyResult, VariableId, err,
};

impl StatisticalStudyResult {
    /// Paired difference of means, preserving shared-sample covariance.
    /// # Errors
    /// Different evidence, snapshots, inference settings or unavailable outcome.
    pub fn contrast(
        &self,
        reference: &Self,
        outcome: VariableId,
    ) -> Result<StatisticalContrast, IoError> {
        if self.identities.identification != reference.identities.identification
            || self.identities.snapshot != reference.identities.snapshot
            || self.identities.inference_binding != reference.identities.inference_binding
        {
            return Err(err(
                "contrast requires the same evidence, snapshots and bootstrap settings",
            ));
        }
        let point =
            self.distribution().mean_difference(reference.distribution(), outcome).map_err(err)?;
        let mut contrast = StatisticalContrast {
            estimate: point,
            interval: None,
            coverage_target: None,
            replicates_ok: 0,
            replicates_failed: 0,
            reason: Some(IntervalWithholdReason::UncertaintyUnavailable.as_str()),
        };
        // A point's own `uncertainty_reason` also covers the facade's support floor
        // (`insufficient_bootstrap_replicates`), which withholds only that point's
        // *pointwise* percentile edges while still retaining its replicate draws. The
        // contrast draws its own paired sample from those retained replicates and
        // applies its own floor below, so only a hard absence of replicate data (no
        // bootstrap ran at all, or the law came from an exact supplied source) should
        // withhold the contrast; that absence is what the destructure below detects.
        let (Some(row), Some(ids), Some(reference_ids), Some(left), Some(right)) = (
            &self.estimate.uncertainty,
            &self.estimate.replicate_ids,
            &reference.estimate.replicate_ids,
            &self.estimate.mean_replicates,
            &reference.estimate.mean_replicates,
        ) else {
            return Ok(contrast);
        };
        let left = left
            .iter()
            .find(|(v, _)| *v == outcome)
            .ok_or_else(|| err("unknown contrast outcome"))?;
        let right = right
            .iter()
            .find(|(v, _)| *v == outcome)
            .ok_or_else(|| err("unknown contrast outcome"))?;
        let pairs: std::collections::BTreeMap<_, _> =
            reference_ids.iter().zip(right.1.iter()).collect();
        let draws: Vec<_> = ids
            .iter()
            .zip(left.1.iter())
            .filter_map(|(id, value)| pairs.get(id).map(|other| value - **other))
            .collect();
        contrast.replicates_ok = u32::try_from(draws.len()).map_err(err)?;
        contrast.replicates_failed = row
            .replicates_requested
            .checked_sub(contrast.replicates_ok)
            .ok_or_else(|| err("invalid replicate accounting"))?;
        contrast.coverage_target = Some(row.coverage_target);
        let paired = ReplicatePolicy {
            min_requested: 0,
            min_ok: PERCENTILE_95_MIN_REPLICATES,
            ..ReplicatePolicy::BOOTSTRAP
        };
        match paired
            .decide(row.replicates_requested, contrast.replicates_ok, contrast.replicates_failed)
            .and_then(|()| {
                antecedent_estimate::percentile_interval(&draws, row.coverage_target)
                    .ok_or(INTERVAL_NUMERICAL_FAILURE)
            }) {
            Ok(interval) => {
                contrast.interval = Some(interval);
                contrast.reason = None;
            }
            Err(reason) => contrast.reason = Some(reason),
        }
        Ok(contrast)
    }
}
