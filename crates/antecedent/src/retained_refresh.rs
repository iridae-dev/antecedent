//! Retained refresh of the summary a prepared study's estimator family rests on.
//!
//! A prepared study re-estimates on new rows without re-identifying. For the estimator
//! families whose result is a function of a least-squares or scatter summary, the summary is
//! kept with the rows it came from so it can follow the data instead of being refitted:
//!
//! * [`RefreshFamily::Ols`] (integer-count least squares: linear adjustment, linear
//!   mediation) is [`OlsRefresh`];
//! * [`RefreshFamily::WeightedOls`] (propensity-weighted and doubly robust fits, weighted
//!   instruments) is [`WeightedOlsRefresh`] and carries real weights;
//! * [`RefreshFamily::Covariance`] (moment estimators such as the Wald instrument ratio) is
//!   [`CovarianceRefresh`];
//! * [`RefreshFamily::ConjugateBlocks`] (the conjugate Gaussian Bayesian routes) is
//!   [`ConjugateBlockRefresh`], one block of rows per labelled source.
//!
//! [`RefreshMutation`] is the closed set of changes a prepared handle accepts: append a
//! window of rows, keep a trailing window, evict a row, set a row's weight, mask a row, or
//! replace the source. An append updates the summary in place ([`RefreshOutcome::Updated`]);
//! every other change drops the summary and recomputes it from the retained rows
//! ([`RefreshOutcome::Recomputed`]). [`RetainedRefresh::apply_verified`] applies a mutation
//! and compares the complete result (statistics, diagnostics, source identity, ledger) with a
//! fresh reconstruction from the rows, leaving the handle unchanged when they disagree.
//!
//! Retention follows [`RetentionMode`]: a summary-only handle keeps no rows, extends in place
//! and refuses every change that needs the rows with the next attempt. A disk-backed handle is
//! [`SpilledRefresh`]: [`RetainedRefresh::spill`] writes the rows through a
//! [`SpillRegistry`] after admitting the frame against the operation's admission pool, and
//! [`SpilledRefresh::reload`] admits the same bytes again before it reads the artifact back.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::path::PathBuf;
use std::sync::Arc;

use antecedent_core::ExecutionContext;
use antecedent_core::execution::AdmissionError;
use antecedent_state::retention::{RetainedOperation, RetentionMode};
use antecedent_state::{
    ConjugateBlockRefresh, ConjugateOlsPrior, ConjugateOlsSummary, CovarianceRefresh,
    CovarianceRow, CovarianceSnapshot, OlsRefresh, OlsSnapshot, RefreshOutcome, RefreshRow,
    SpillHandle, SpillRegistry, Spillable, StateError, WeightedOlsRefresh, WeightedRow,
    WeightedSnapshot,
};

use crate::error::CausalError;
use crate::strategy_table::EstimatorId;

/// Which refresh summary an estimator family keeps.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum RefreshFamily {
    /// Integer-count least squares ([`OlsRefresh`]).
    Ols,
    /// Real-weight least squares ([`WeightedOlsRefresh`]).
    WeightedOls,
    /// Streaming scatter ([`CovarianceRefresh`]).
    Covariance,
    /// Conjugate Gaussian posterior over labelled blocks ([`ConjugateBlockRefresh`]).
    ConjugateBlocks,
}

impl RefreshFamily {
    /// Stable name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Ols => "ols",
            Self::WeightedOls => "weighted_ols",
            Self::Covariance => "covariance",
            Self::ConjugateBlocks => "conjugate_blocks",
        }
    }

    /// The summary the estimator `id` keeps, or `None` for an estimator whose result is not a
    /// function of one of these summaries (matching, forests, cross-fitted learners, discrete
    /// plug-ins).
    #[must_use]
    pub const fn of_estimator(id: EstimatorId) -> Option<Self> {
        match id {
            EstimatorId::LinearAdjustmentAte
            | EstimatorId::ConditionalLinearAdjustment
            | EstimatorId::TemporalLinearAdjustment
            | EstimatorId::StaticMediationLinear
            | EstimatorId::TemporalMediation
            | EstimatorId::FrontDoorTwoStage => Some(Self::Ols),
            EstimatorId::PropensityWeighting
            | EstimatorId::Aipw
            | EstimatorId::Dml
            | EstimatorId::Iv2Sls
            | EstimatorId::LongitudinalMarginalStructuralModel => Some(Self::WeightedOls),
            EstimatorId::IvWald => Some(Self::Covariance),
            EstimatorId::BayesianGcomp
            | EstimatorId::BayesianBasisGcomp
            | EstimatorId::BayesianConditional
            | EstimatorId::BayesianTemporalGcomp
            | EstimatorId::BayesianTemporalWhiteInnovation
            | EstimatorId::BayesianTemporalMediation
            | EstimatorId::TemporalResponseBayesian
            | EstimatorId::ResponseBayesian => Some(Self::ConjugateBlocks),
            _ => None,
        }
    }
}

/// One observation offered to a retained refresh.
///
/// A least-squares family reads `x` as the design row and `y` as the response; the covariance
/// family reads `x` as the observation vector and ignores `y`. `weight` is a nonnegative
/// integer count for [`RefreshFamily::Ols`] and the conjugate blocks, a nonnegative real for
/// [`RefreshFamily::WeightedOls`], and must be one for [`RefreshFamily::Covariance`], whose
/// scatter carries no weights.
#[derive(Clone, Debug, PartialEq)]
pub struct RefreshObservation {
    /// Identity of the observation within its source.
    pub id: u64,
    /// Design row, or the observation vector.
    pub x: Vec<f64>,
    /// Response.
    pub y: f64,
    /// Weight.
    pub weight: f64,
    /// Masked rows are retained and left out of the summary.
    pub masked: bool,
}

impl RefreshObservation {
    /// A unit-weight, unmasked observation.
    #[must_use]
    pub const fn new(id: u64, x: Vec<f64>, y: f64) -> Self {
        Self { id, x, y, weight: 1.0, masked: false }
    }

    /// A weighted, unmasked observation.
    #[must_use]
    pub const fn weighted(id: u64, x: Vec<f64>, y: f64, weight: f64) -> Self {
        Self { id, x, y, weight, masked: false }
    }
}

/// A change to the data a retained refresh summarizes.
#[derive(Clone, Debug, PartialEq)]
pub enum RefreshMutation {
    /// Append a window of rows of `source`. In place.
    Append {
        /// Source the rows belong to; another source is refused.
        source: Arc<str>,
        /// Rows to append.
        rows: Vec<RefreshObservation>,
    },
    /// Keep the most recently appended `max_rows` rows. Recomputed.
    Window {
        /// Rows to keep.
        max_rows: usize,
    },
    /// Remove one retained row. Recomputed.
    Evict {
        /// Row to remove.
        id: u64,
    },
    /// Change the weight of one retained row. Recomputed.
    SetWeight {
        /// Row to change.
        id: u64,
        /// New weight.
        weight: f64,
    },
    /// Change the mask of one retained row. Recomputed.
    SetMask {
        /// Row to change.
        id: u64,
        /// New mask.
        masked: bool,
    },
    /// Replace the source and its rows. Nothing of the old source survives. Recomputed.
    Replace {
        /// New source.
        source: Arc<str>,
        /// Rows of the new source.
        rows: Vec<RefreshObservation>,
    },
    /// A change inside one labelled block of a conjugate posterior.
    Block {
        /// Block the change applies to.
        label: Arc<str>,
        /// The change.
        mutation: Box<RefreshMutation>,
    },
}

impl RefreshMutation {
    /// Apply `mutation` inside the block `label` of a conjugate posterior.
    #[must_use]
    pub fn in_block(label: &str, mutation: Self) -> Self {
        Self::Block { label: Arc::from(label), mutation: Box::new(mutation) }
    }
}

/// The complete result a refresh summary stands for.
#[derive(Clone, Debug, PartialEq)]
pub enum RefreshResult {
    /// Integer-count least squares.
    Ols(OlsSnapshot),
    /// Real-weight least squares.
    Weighted(WeightedSnapshot),
    /// Streaming scatter.
    Covariance(CovarianceSnapshot),
    /// Conjugate posterior and the identity of its prior and blocks.
    Conjugate {
        /// Posterior summary.
        summary: ConjugateOlsSummary,
        /// Identity of the prior and of every block's source and rows.
        identity: String,
    },
}

impl RefreshResult {
    /// Identity of the source the result rests on.
    #[must_use]
    pub fn source_digest(&self) -> &str {
        match self {
            Self::Ols(snapshot) => &snapshot.source_digest,
            Self::Weighted(snapshot) => &snapshot.source_digest,
            Self::Covariance(snapshot) => &snapshot.source_digest,
            Self::Conjugate { identity, .. } => identity,
        }
    }
}

#[derive(Clone, Debug)]
enum Summary {
    Ols(OlsRefresh),
    Weighted(WeightedOlsRefresh),
    Covariance(CovarianceRefresh),
    Conjugate(ConjugateBlockRefresh),
}

/// A refresh summary of the family a prepared study's estimator keeps, with its rows.
#[derive(Clone, Debug)]
pub struct RetainedRefresh {
    summary: Summary,
}

fn invalid(message: String) -> CausalError {
    CausalError::State(StateError::InvalidEvent(message))
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the value is a nonnegative integer no larger than u32::MAX"
)]
fn count_of(weight: f64) -> Result<u32, CausalError> {
    let integral = weight.is_finite() && weight >= 0.0 && weight.fract() == 0.0;
    if integral && weight <= f64::from(u32::MAX) {
        return Ok(weight as u32);
    }
    Err(invalid(format!(
        "weight {weight} is not a nonnegative integer count; supply integer counts, or use an \
         estimator whose family keeps real weights"
    )))
}

fn ols_rows(rows: &[RefreshObservation]) -> Result<Vec<RefreshRow>, CausalError> {
    rows.iter()
        .map(|row| {
            Ok(RefreshRow {
                id: row.id,
                x: row.x.clone(),
                y: row.y,
                count: count_of(row.weight)?,
                masked: row.masked,
            })
        })
        .collect()
}

fn weighted_rows(rows: &[RefreshObservation]) -> Vec<WeightedRow> {
    rows.iter()
        .map(|row| WeightedRow {
            id: row.id,
            x: row.x.clone(),
            y: row.y,
            weight: row.weight,
            masked: row.masked,
        })
        .collect()
}

fn covariance_rows(rows: &[RefreshObservation]) -> Result<Vec<CovarianceRow>, CausalError> {
    rows.iter()
        .map(|row| {
            if row.weight.to_bits() != 1.0f64.to_bits() {
                return Err(invalid(format!(
                    "observation {} carries weight {}; a scatter carries no weights, so mask the \
                     row or use an estimator whose family keeps real weights",
                    row.id, row.weight
                )));
            }
            Ok(CovarianceRow { id: row.id, z: row.x.clone(), masked: row.masked })
        })
        .collect()
}

fn block_required() -> CausalError {
    invalid(
        "a conjugate posterior keeps one block of rows per label; name the block with \
         RefreshMutation::in_block"
            .into(),
    )
}

fn block_unexpected() -> CausalError {
    invalid(
        "only a conjugate posterior has labelled blocks; apply the change without in_block".into(),
    )
}

fn ols_frame_bytes(rows: &[RefreshRow], source: &str) -> u64 {
    rows.iter().map(|row| 8 * row.x.len() as u64 + 48).sum::<u64>() + source.len() as u64 + 256
}

impl RetainedRefresh {
    /// Start the summary of `family` over `rows` of `source`, `width` columns wide.
    ///
    /// The conjugate family needs `conjugate`: the prior and the label of the first block,
    /// whose `rows` are the observations of that block.
    ///
    /// # Errors
    /// A malformed row, a repeated id, a width that disagrees with the prior, a missing prior
    /// for the conjugate family, or a weight the family cannot carry.
    pub fn start(
        family: RefreshFamily,
        source: &str,
        width: usize,
        rows: &[RefreshObservation],
        conjugate: Option<(&ConjugateOlsPrior, &str)>,
    ) -> Result<Self, CausalError> {
        let summary = match family {
            RefreshFamily::Ols => {
                Summary::Ols(OlsRefresh::from_rows(source, width, ols_rows(rows)?)?)
            }
            RefreshFamily::WeightedOls => Summary::Weighted(WeightedOlsRefresh::from_rows(
                source,
                width,
                weighted_rows(rows),
            )?),
            RefreshFamily::Covariance => Summary::Covariance(CovarianceRefresh::from_rows(
                source,
                width,
                covariance_rows(rows)?,
            )?),
            RefreshFamily::ConjugateBlocks => {
                let Some((prior, label)) = conjugate else {
                    return Err(invalid(
                        "the conjugate family needs a prior and a first block label".into(),
                    ));
                };
                if prior.ncols() != width {
                    return Err(CausalError::State(StateError::Shape(format!(
                        "prior has {} columns, the rows have {width}",
                        prior.ncols()
                    ))));
                }
                let mut fit = ConjugateBlockRefresh::new(prior.clone());
                fit.add_block(label, source, ols_rows(rows)?)?;
                Summary::Conjugate(fit)
            }
        };
        Ok(Self { summary })
    }

    /// The family this summary belongs to.
    #[must_use]
    pub const fn family(&self) -> RefreshFamily {
        match self.summary {
            Summary::Ols(_) => RefreshFamily::Ols,
            Summary::Weighted(_) => RefreshFamily::WeightedOls,
            Summary::Covariance(_) => RefreshFamily::Covariance,
            Summary::Conjugate(_) => RefreshFamily::ConjugateBlocks,
        }
    }

    /// What the summary retains for later operations.
    #[must_use]
    pub const fn retention(&self) -> RetentionMode {
        match &self.summary {
            Summary::Ols(fit) => fit.retention(),
            Summary::Weighted(fit) => fit.retention(),
            Summary::Covariance(fit) => fit.retention(),
            Summary::Conjugate(fit) => fit.retention(),
        }
    }

    /// How many times the summary was rebuilt from the retained rows.
    #[must_use]
    pub const fn recomputes(&self) -> u64 {
        match &self.summary {
            Summary::Ols(fit) => fit.recomputes(),
            Summary::Weighted(fit) => fit.recomputes(),
            Summary::Covariance(fit) => fit.recomputes(),
            Summary::Conjugate(fit) => fit.recomputes(),
        }
    }

    /// Stop retaining rows. The result stays; every later change that needs the rows is
    /// refused with its next attempt.
    #[must_use]
    pub fn into_summary_only(self) -> Self {
        let summary = match self.summary {
            Summary::Ols(fit) => Summary::Ols(fit.into_summary_only()),
            Summary::Weighted(fit) => Summary::Weighted(fit.into_summary_only()),
            Summary::Covariance(fit) => Summary::Covariance(fit.into_summary_only()),
            Summary::Conjugate(fit) => Summary::Conjugate(fit.into_summary_only()),
        };
        Self { summary }
    }

    /// Accept `operation` under the retention of this summary, or refuse it.
    ///
    /// # Errors
    /// The retention cannot serve `operation`.
    pub fn require(&self, operation: RetainedOperation) -> Result<(), CausalError> {
        self.retention().require(operation).map_err(|refusal| {
            invalid(format!(
                "{refusal}; reload the rows of this source and rebuild from them, or keep full \
                 reusable retention"
            ))
        })
    }

    /// Apply one change. An append updates in place; everything else recomputes from the
    /// retained rows.
    ///
    /// # Errors
    /// A mutation the family cannot carry, a refusal of the retention mode (naming the next
    /// attempt), a malformed row, a repeated or unknown id, or a source change on an append.
    /// The summary is unchanged on an error.
    pub fn apply(&mut self, mutation: &RefreshMutation) -> Result<RefreshOutcome, CausalError> {
        match &mut self.summary {
            Summary::Ols(fit) => apply_ols(fit, mutation),
            Summary::Weighted(fit) => apply_weighted(fit, mutation),
            Summary::Covariance(fit) => apply_covariance(fit, mutation),
            Summary::Conjugate(fit) => apply_conjugate(fit, mutation),
        }
    }

    /// Apply one change, then compare the complete result with a fresh reconstruction from the
    /// retained rows. The summary is left as it was when the change or the comparison fails.
    ///
    /// # Errors
    /// The errors of [`Self::apply`], rows that are not retained (the comparison needs them),
    /// or the first field that disagrees with the rebuild.
    pub fn apply_verified(
        &mut self,
        mutation: &RefreshMutation,
    ) -> Result<RefreshOutcome, CausalError> {
        let before = self.clone();
        let outcome = self.apply(mutation).and_then(|outcome| {
            self.check_against_fresh()?;
            Ok(outcome)
        });
        if outcome.is_err() {
            *self = before;
        }
        outcome
    }

    /// The complete cached result.
    ///
    /// # Errors
    /// A singular conjugate posterior precision.
    pub fn result(&self) -> Result<RefreshResult, CausalError> {
        Ok(match &self.summary {
            Summary::Ols(fit) => RefreshResult::Ols(fit.snapshot()),
            Summary::Weighted(fit) => RefreshResult::Weighted(fit.snapshot()),
            Summary::Covariance(fit) => RefreshResult::Covariance(fit.snapshot()),
            Summary::Conjugate(fit) => {
                RefreshResult::Conjugate { summary: fit.summary()?, identity: fit.identity() }
            }
        })
    }

    /// The complete result rebuilt from the retained rows alone.
    ///
    /// # Errors
    /// Rows are not retained.
    pub fn fresh_result(&self) -> Result<RefreshResult, CausalError> {
        Ok(match &self.summary {
            Summary::Ols(fit) => RefreshResult::Ols(fit.fresh_snapshot()?),
            Summary::Weighted(fit) => RefreshResult::Weighted(fit.fresh_snapshot()?),
            Summary::Covariance(fit) => RefreshResult::Covariance(fit.fresh_snapshot()?),
            Summary::Conjugate(fit) => {
                let rebuilt = rebuilt_conjugate(fit)?;
                RefreshResult::Conjugate {
                    summary: rebuilt.summary()?,
                    identity: fit.fresh_identity()?,
                }
            }
        })
    }

    /// Compare the cached result with a fresh reconstruction from the retained rows.
    ///
    /// # Errors
    /// Rows are not retained, or the first field that disagrees.
    pub fn check_against_fresh(&self) -> Result<(), CausalError> {
        match &self.summary {
            Summary::Ols(fit) => fit.check_against_fresh()?,
            Summary::Weighted(fit) => fit.check_against_fresh()?,
            Summary::Covariance(fit) => fit.check_against_fresh()?,
            Summary::Conjugate(fit) => fit.check_against_fresh()?,
        }
        Ok(())
    }

    /// Bytes the frame of this summary is expected to take: what a spill or a reload asks the
    /// admission pool for.
    #[must_use]
    pub fn frame_bytes(&self) -> u64 {
        match &self.summary {
            Summary::Ols(fit) => ols_frame_bytes(fit.rows(), fit.source()),
            Summary::Weighted(fit) => {
                fit.rows().iter().map(|row| 8 * row.x.len() as u64 + 56).sum::<u64>()
                    + fit.source().len() as u64
                    + 256
            }
            Summary::Covariance(fit) => {
                fit.rows().iter().map(|row| 8 * row.z.len() as u64 + 40).sum::<u64>()
                    + fit.source().len() as u64
                    + 256
            }
            Summary::Conjugate(fit) => fit
                .labels()
                .filter_map(|label| fit.block(label))
                .map(|block| ols_frame_bytes(block.rows(), block.source()))
                .sum::<u64>()
                .saturating_add(256),
        }
    }

    /// Write the rows under `registry` as a disk-backed result, after admitting the frame
    /// against the operation's admission pool. The lease is returned when the write ends.
    ///
    /// # Errors
    /// The admission pool cannot hold the frame, the rows are not retained, the frame is
    /// larger than the whole budget of `registry`, or the file cannot be written.
    pub fn spill(
        &self,
        ctx: &ExecutionContext,
        registry: &Arc<SpillRegistry>,
        key: &str,
    ) -> Result<SpilledRefresh, CausalError> {
        let _lease = admit(ctx, self.frame_bytes(), "spilling a prepared refresh")?;
        let handle = match &self.summary {
            Summary::Ols(fit) => fit.spill(registry, key)?,
            Summary::Weighted(fit) => fit.spill(registry, key)?,
            Summary::Covariance(fit) => fit.spill(registry, key)?,
            Summary::Conjugate(fit) => fit.spill(registry, key)?,
        };
        Ok(SpilledRefresh { handle, family: self.family() })
    }
}

/// The posterior rebuilt from a copy of every block's retained rows.
fn rebuilt_conjugate(fit: &ConjugateBlockRefresh) -> Result<ConjugateBlockRefresh, CausalError> {
    let mut fresh = ConjugateBlockRefresh::new(fit.prior().clone());
    for label in fit.labels() {
        let Some(block) = fit.block(label) else { continue };
        block.retention().require(RetainedOperation::RefitFromRetainedRows).map_err(|refusal| {
            invalid(format!(
                "block {label}: {refusal}; reload the rows of this block and rebuild from them"
            ))
        })?;
        fresh.add_block(label, block.source(), block.rows().to_vec())?;
    }
    Ok(fresh)
}

fn apply_ols(
    fit: &mut OlsRefresh,
    mutation: &RefreshMutation,
) -> Result<RefreshOutcome, CausalError> {
    Ok(match mutation {
        RefreshMutation::Append { source, rows } => fit.append_window(source, &ols_rows(rows)?)?,
        RefreshMutation::Window { max_rows } => fit.retain_last(*max_rows)?,
        RefreshMutation::Evict { id } => fit.evict_row(*id)?,
        RefreshMutation::SetWeight { id, weight } => fit.set_count(*id, count_of(*weight)?)?,
        RefreshMutation::SetMask { id, masked } => fit.set_mask(*id, *masked)?,
        RefreshMutation::Replace { source, rows } => {
            fit.replace_source(Arc::clone(source), ols_rows(rows)?)?
        }
        RefreshMutation::Block { .. } => return Err(block_unexpected()),
    })
}

fn apply_weighted(
    fit: &mut WeightedOlsRefresh,
    mutation: &RefreshMutation,
) -> Result<RefreshOutcome, CausalError> {
    Ok(match mutation {
        RefreshMutation::Append { source, rows } => {
            fit.append_window(source, &weighted_rows(rows))?
        }
        RefreshMutation::Window { max_rows } => fit.retain_last(*max_rows)?,
        RefreshMutation::Evict { id } => fit.evict_row(*id)?,
        RefreshMutation::SetWeight { id, weight } => fit.set_weight(*id, *weight)?,
        RefreshMutation::SetMask { id, masked } => fit.set_mask(*id, *masked)?,
        RefreshMutation::Replace { source, rows } => {
            fit.replace_source(Arc::clone(source), weighted_rows(rows))?
        }
        RefreshMutation::Block { .. } => return Err(block_unexpected()),
    })
}

fn apply_covariance(
    fit: &mut CovarianceRefresh,
    mutation: &RefreshMutation,
) -> Result<RefreshOutcome, CausalError> {
    Ok(match mutation {
        RefreshMutation::Append { source, rows } => {
            fit.append_window(source, &covariance_rows(rows)?)?
        }
        RefreshMutation::Window { max_rows } => fit.retain_last(*max_rows)?,
        RefreshMutation::Evict { id } => fit.evict_row(*id)?,
        RefreshMutation::SetWeight { id, weight } => {
            return Err(invalid(format!(
                "observation {id} cannot take weight {weight}; a scatter carries no weights, so \
                 mask the row or use an estimator whose family keeps real weights"
            )));
        }
        RefreshMutation::SetMask { id, masked } => fit.set_mask(*id, *masked)?,
        RefreshMutation::Replace { source, rows } => {
            fit.replace_source(Arc::clone(source), covariance_rows(rows)?)?
        }
        RefreshMutation::Block { .. } => return Err(block_unexpected()),
    })
}

fn apply_conjugate(
    fit: &mut ConjugateBlockRefresh,
    mutation: &RefreshMutation,
) -> Result<RefreshOutcome, CausalError> {
    let RefreshMutation::Block { label, mutation } = mutation else {
        return Err(block_required());
    };
    Ok(match &**mutation {
        RefreshMutation::Append { source, rows } => {
            let rows = ols_rows(rows)?;
            if fit.block(label).is_some() {
                fit.extend_block(label, source, &rows)?
            } else {
                fit.add_block(label, source, rows)?
            }
        }
        RefreshMutation::Window { max_rows } => fit.retain_last(label, *max_rows)?,
        RefreshMutation::Evict { id } => fit.evict_row(label, *id)?,
        RefreshMutation::SetWeight { id, weight } => {
            fit.set_count(label, *id, count_of(*weight)?)?
        }
        RefreshMutation::SetMask { id, masked } => fit.set_mask(label, *id, *masked)?,
        RefreshMutation::Replace { source, rows } => {
            let rows = ols_rows(rows)?;
            let mut fresh = ConjugateBlockRefresh::new(fit.prior().clone());
            if fit.retention() != RetentionMode::FullReusable {
                fresh = fresh.into_summary_only();
            }
            fresh.add_block(label, source, rows)?;
            *fit = fresh;
            RefreshOutcome::Recomputed
        }
        RefreshMutation::Block { .. } => return Err(block_unexpected()),
    })
}

/// Take `bytes` of the operation's admission pool for the duration of the returned lease.
pub(crate) fn admit(
    ctx: &ExecutionContext,
    bytes: u64,
    what: &str,
) -> Result<antecedent_core::execution::AdmissionLease, CausalError> {
    ctx.acquire_admission(0, bytes).map_err(|error| match error {
        AdmissionError::Cancelled => CausalError::Resource {
            message: format!("{what} stopped because the study was cancelled"),
        },
        other => CausalError::Resource {
            message: format!(
                "{what} needs {bytes} bytes of the operation's admission budget: {other}; raise \
                 the memory limit, or spill a smaller result"
            ),
        },
    })
}

/// A spill registry whose byte budget is the smaller of `requested_bytes` and the byte budget
/// of the operation's admission pool, so spilled results never claim more than the operation
/// may hold.
///
/// # Errors
/// The budget is zero, or the directory cannot be created.
pub fn spill_registry_within_admission(
    ctx: &ExecutionContext,
    directory: impl Into<PathBuf>,
    requested_bytes: u64,
) -> Result<Arc<SpillRegistry>, CausalError> {
    let budget = requested_bytes.min(ctx.admission.byte_budget());
    if budget == 0 {
        return Err(CausalError::Resource {
            message: "no bytes of the operation's admission budget are available for spilled \
                      results; raise the memory limit"
                .into(),
        });
    }
    Ok(SpillRegistry::new(directory, budget)?)
}

/// Write `value` under `key` after admitting `expected_bytes` against the operation's
/// admission pool.
pub(crate) fn spill_admitted<T: Spillable>(
    ctx: &ExecutionContext,
    registry: &Arc<SpillRegistry>,
    key: &str,
    value: &T,
    expected_bytes: u64,
) -> Result<SpillHandle, CausalError> {
    let _lease = admit(ctx, expected_bytes, "spilling a retained result")?;
    Ok(registry.spill(key, value)?)
}

/// Read `handle` back after admitting the frame bytes against the operation's admission pool.
pub(crate) fn reload_admitted<T: Spillable>(
    ctx: &ExecutionContext,
    handle: &SpillHandle,
) -> Result<T, CausalError> {
    let _lease = admit(ctx, handle.bytes(), "reloading a spilled result")?;
    Ok(handle.reload::<T>()?)
}

/// A refresh summary written to disk. Its mode is [`RetentionMode::DiskBacked`]: it serves a
/// read of the artifact and refuses every resident operation with the reload that serves it.
#[derive(Clone, Debug)]
pub struct SpilledRefresh {
    handle: SpillHandle,
    family: RefreshFamily,
}

impl SpilledRefresh {
    /// Family of the spilled summary.
    #[must_use]
    pub const fn family(&self) -> RefreshFamily {
        self.family
    }

    /// The mode of a spilled result.
    #[must_use]
    pub const fn mode(&self) -> RetentionMode {
        RetentionMode::DiskBacked
    }

    /// The handle on the artifact.
    #[must_use]
    pub const fn handle(&self) -> &SpillHandle {
        &self.handle
    }

    /// Keys evicted from the registry to make room for this spill.
    #[must_use]
    pub fn evicted(&self) -> &[String] {
        self.handle.evicted()
    }

    /// Whether the artifact is still on disk (it is not once evicted).
    #[must_use]
    pub fn on_disk(&self) -> bool {
        self.handle.on_disk()
    }

    /// Accept `operation` on the spilled result, or refuse it with the next attempt.
    ///
    /// # Errors
    /// The disk-backed mode cannot serve `operation`.
    pub fn require(&self, operation: RetainedOperation) -> Result<(), CausalError> {
        Ok(self.handle.require(operation)?)
    }

    /// Read the artifact back as a resident summary, after admitting its frame against the
    /// operation's admission pool.
    ///
    /// # Errors
    /// The admission pool cannot hold the frame, the artifact was evicted (the next attempt is
    /// to rebuild from the rows), or the file was changed.
    pub fn reload(&self, ctx: &ExecutionContext) -> Result<RetainedRefresh, CausalError> {
        let summary = match self.family {
            RefreshFamily::Ols => Summary::Ols(reload_admitted(ctx, &self.handle)?),
            RefreshFamily::WeightedOls => Summary::Weighted(reload_admitted(ctx, &self.handle)?),
            RefreshFamily::Covariance => Summary::Covariance(reload_admitted(ctx, &self.handle)?),
            RefreshFamily::ConjugateBlocks => {
                Summary::Conjugate(reload_admitted(ctx, &self.handle)?)
            }
        };
        Ok(RetainedRefresh { summary })
    }
}
