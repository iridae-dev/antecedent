//! Staleness and drift diagnostics: is a stored plan or claim still valid for new data?
//!
//! A [`StalenessBaseline`] summarizes the data a plan was prepared (or a claim was estimated)
//! on: the schema, per-column moments, quantile bins and level counts, missingness, the joint
//! moments and a uniform subsample of the covariates, the propensity overlap of the treatment
//! on the stored adjustment set, a catalogue of recorded assumption checks, and (for a Bayesian
//! plan) the sufficient statistics of the outcome regression. It exports as a sealed artifact,
//! so a consumer that holds only those bytes and new data can run the diagnostic without the
//! producer or the original rows.
//!
//! [`StalenessBaseline::assess`] streams once over the new data (plus one propensity fit and a
//! bounded permutation test), and returns a [`StalenessReport`]: every diagnostic as a typed
//! [`StalenessFinding`] with its statistic, the policy threshold it was judged by and the
//! dependency keys it invalidates, and one [`Recommendation`].
//!
//! # Diagnostics
//!
//! | Kind | Statistic | Judged by | Forces | Keys |
//! |---|---|---|---|---|
//! | covariate shift | standardized mean difference (Welch test); population stability index (chi-square test of the binned density ratio) | `smd_threshold`, `psi_threshold`, `alpha` | refresh | `data_snapshot` |
//! | joint shift | Mahalanobis distance (Hotelling F test); energy statistic (keyed permutation test) | `mahalanobis_threshold`, `alpha` | refresh | `data_snapshot` |
//! | positivity change | fraction of propensities outside `[ε, 1 − ε]` | `max_extreme_fraction` | reidentify | `evidence` |
//! | overlap change | new over recorded effective-sample-size fraction of the inverse-propensity weights | `min_ess_ratio_change` | reidentify | `evidence` |
//! | assumption regression | the recorded check's statistic on new data | the recorded threshold | by check level | by check level |
//! | support loss | fraction of new values beyond the recorded range (exact exchangeability test) | `support_tolerance`, `alpha` | reidentify on the treatment or adjustment set, refresh otherwise | `evidence` or `data_snapshot` |
//! | level change | rows at a new level; a recorded level that disappeared (binomial test) | `min_level_rows`, `alpha` | as support loss | as support loss |
//! | prior-likelihood conflict | Box prior predictive F statistic of the new coefficients under the stored posterior; Gaussian divergence | `alpha` | refresh | `prior.0`, `data_snapshot` |
//! | period drift | CUSUM of per-period means (Brownian-bridge supremum) | `smd_threshold`, `alpha` | reidentify | `source_population`, `evidence` |
//! | schema change | columns added, removed, retyped or reordered | zero | refuse | `data_snapshot`, `evidence` |
//! | missingness change | change of a column's missing rate (two-proportion test); total-variation change of the patterns (chi-square test) | `missingness_threshold`, `alpha` | refresh | `row_design` |
//! | insufficient data, target incompatible | complete rows, rows per arm, missing role columns or arms | `min_rows`, `min_rows_per_arm` | refuse | `data_snapshot`, `row_design` or `query` |
//!
//! Every test-based diagnostic of one assessment is one multiplicity family, adjusted by the
//! policy's [`Multiplicity`]; a test-based diagnostic exceeds only when its adjusted p-value is
//! at most `alpha` and its effect size reaches its threshold. With no shift at all, the
//! probability that any test-based diagnostic exceeds is therefore at most `alpha`.
//!
//! # Recommendation
//!
//! The report takes the most severe recommendation any exceeded diagnostic forces:
//!
//! * [`Recommendation::Reuse`]: nothing exceeds. The stored claim stands; no key changes.
//! * [`Recommendation::Refresh`]: the data changed but every identification-level premise
//!   holds. Keep the plan and its identification; refit. The changed keys are `data_snapshot`
//!   (and `row_design` when the complete rows changed, `prior.0` on a prior conflict).
//! * [`Recommendation::Reidentify`]: an identification-level premise failed (positivity,
//!   overlap, support of the treatment or adjustment set, a recorded identification check,
//!   period drift). The changed keys add `evidence` (and `source_population`), which the
//!   recalculation planner propagates to the identification product.
//! * [`Recommendation::Refuse`]: the schema or the target is incompatible, or too few rows
//!   remain to assess uncertainty.
//!
//! The invalidated stages are not restated here: they are what
//! [`antecedent_core::recalc::RecalcPlan`] recomputes when the changed keys change, so a
//! recommendation always agrees with what selective recalculation invalidates.
//!
//! No scheduler or alerting service is part of this module; a host calls the diagnostic.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod assess;
mod policy;
mod profile;
mod report;
mod stats;

use antecedent_core::ExecutionContext;
use antecedent_data::{DataError, TableView, TabularData};
use antecedent_io_core::staleness_artifact::StalenessBaselineWire;

pub use policy::{Multiplicity, StalenessPolicy};
pub use report::{
    FindingKind, Recommendation, StalenessFinding, StalenessReport, dependency_class,
};

/// Why a staleness diagnostic could not run. A diagnostic that runs and finds the data
/// unusable is a [`Recommendation::Refuse`] report, not an error.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum StalenessError {
    /// The policy cannot judge a finding.
    #[error("staleness policy: {0}")]
    Policy(&'static str),
    /// The plan names a column its data does not carry, or a role twice.
    #[error("staleness plan: {0}")]
    Plan(String),
    /// A column could not be read.
    #[error(transparent)]
    Data(#[from] DataError),
    /// The baseline artifact does not verify or cannot be encoded.
    #[error("staleness baseline: {0}")]
    Artifact(String),
    /// The admission pool cannot fund the diagnostic's buffers. Nothing was allocated.
    #[error("staleness diagnostic needs {bytes} bytes the admission pool cannot fund: {reason}")]
    Admission {
        /// Bytes requested.
        bytes: u64,
        /// The pool's refusal.
        reason: String,
    },
    /// The execution context was cancelled.
    #[error("staleness diagnostic cancelled")]
    Cancelled,
}

/// The roles of a plan that a baseline records.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StalenessPlan {
    /// Identity of the plan (a contract or program digest, or any stable caller identity).
    pub plan_identity: String,
    /// Treatment column.
    pub treatment: Option<String>,
    /// Outcome column.
    pub outcome: Option<String>,
    /// Stored adjustment set.
    pub adjustment: Vec<String>,
    /// Whether the plan is Bayesian (enables the prior-likelihood check).
    pub bayesian: bool,
}

impl StalenessPlan {
    fn validate(&self, data: &TabularData) -> Result<(), StalenessError> {
        let mut seen = std::collections::BTreeSet::new();
        for name in self.treatment.iter().chain(self.outcome.iter()).chain(self.adjustment.iter()) {
            if !seen.insert(name.as_str()) {
                return Err(StalenessError::Plan(format!("column {name} has two roles")));
            }
            if data.schema().id_of(name).is_err() {
                return Err(StalenessError::Plan(format!("the plan's data has no column {name}")));
            }
        }
        Ok(())
    }
}

/// Content identity of a table (lowercase hex).
#[must_use]
pub fn snapshot_identity(data: &TabularData) -> String {
    antecedent_core::SemanticDigest::from_bytes(data.storage().content_digest()).to_hex()
}

/// Lease the peak bytes of one pass over `data` before any buffer is allocated.
fn lease(
    data: &TabularData,
    policy: &StalenessPolicy,
    ctx: &ExecutionContext,
) -> Result<antecedent_core::execution::AdmissionLease, StalenessError> {
    if ctx.cancellation.is_cancelled() {
        return Err(StalenessError::Cancelled);
    }
    let bytes = profile::peak_bytes(data.row_count(), data.schema().len(), policy);
    ctx.acquire_admission(0, bytes).map_err(|error| match error {
        antecedent_core::execution::AdmissionError::Cancelled => StalenessError::Cancelled,
        other => StalenessError::Admission { bytes, reason: other.to_string() },
    })
}

/// The sealed summary of the data a plan was prepared on.
#[derive(Clone, Debug, PartialEq)]
pub struct StalenessBaseline {
    wire: StalenessBaselineWire,
}

impl StalenessBaseline {
    /// Summarize `data` under `plan` and `policy`.
    ///
    /// # Errors
    /// An invalid policy or plan, an unreadable column, a refused admission lease, or
    /// cancellation.
    pub fn build(
        data: &TabularData,
        plan: &StalenessPlan,
        policy: &StalenessPolicy,
        ctx: &ExecutionContext,
    ) -> Result<Self, StalenessError> {
        policy.validate()?;
        plan.validate(data)?;
        let _lease = lease(data, policy, ctx)?;
        let table = profile::ReadTable::new(data)?;
        if ctx.cancellation.is_cancelled() {
            return Err(StalenessError::Cancelled);
        }
        let inputs = profile::ProfileInputs {
            plan,
            policy,
            reference: None,
            subsample_op: stats::BASELINE_SUBSAMPLE_OP,
        };
        let wire = profile::profile(&table, snapshot_identity(data), &inputs)
            .sealed()
            .map_err(|e| StalenessError::Artifact(e.to_string()))?;
        Ok(Self { wire })
    }

    /// A baseline from a decoded wire, after its seal verifies.
    ///
    /// # Errors
    /// The wire does not verify.
    pub fn from_wire(wire: StalenessBaselineWire) -> Result<Self, StalenessError> {
        wire.verify().map_err(|e| StalenessError::Artifact(e.to_string()))?;
        Ok(Self { wire })
    }

    /// Decode and verify an exported baseline.
    ///
    /// # Errors
    /// The bytes are not a baseline, or it does not verify.
    pub fn consume(bytes: &[u8]) -> Result<Self, StalenessError> {
        StalenessBaselineWire::consume(bytes)
            .map(|wire| Self { wire })
            .map_err(|e| StalenessError::Artifact(e.to_string()))
    }

    /// Encode the sealed baseline.
    ///
    /// # Errors
    /// Encoding failure or the byte limit.
    pub fn export(&self) -> Result<Vec<u8>, StalenessError> {
        self.wire.export().map_err(|e| StalenessError::Artifact(e.to_string()))
    }

    /// The sealed wire.
    #[must_use]
    pub const fn wire(&self) -> &StalenessBaselineWire {
        &self.wire
    }

    /// Seal of the baseline (its identity).
    #[must_use]
    pub fn identity(&self) -> &str {
        &self.wire.seal
    }

    /// Content identity of the summarized snapshot.
    #[must_use]
    pub fn snapshot(&self) -> &str {
        &self.wire.snapshot
    }

    /// Identity of the plan the baseline describes.
    #[must_use]
    pub fn plan_identity(&self) -> &str {
        &self.wire.plan_identity
    }

    /// Compare `data` with this baseline under `policy`.
    ///
    /// # Errors
    /// An invalid policy, an unreadable column, a refused admission lease, or cancellation.
    pub fn assess(
        &self,
        data: &TabularData,
        policy: &StalenessPolicy,
        ctx: &ExecutionContext,
    ) -> Result<StalenessReport, StalenessError> {
        policy.validate()?;
        let _lease = lease(data, policy, ctx)?;
        assess::assess(&self.wire, data, policy, ctx)
    }
}

/// Compare `data` with an exported baseline (`bytes`) under `policy`, without the producer.
///
/// # Errors
/// The baseline does not verify, or the errors of [`StalenessBaseline::assess`].
pub fn assess_exported_baseline(
    bytes: &[u8],
    data: &TabularData,
    policy: &StalenessPolicy,
    ctx: &ExecutionContext,
) -> Result<StalenessReport, StalenessError> {
    StalenessBaseline::consume(bytes)?.assess(data, policy, ctx)
}
