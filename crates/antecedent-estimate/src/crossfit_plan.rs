//! The one cross-fit fold plan shared by every cross-fitted pipeline.
//!
//! A [`CrossfitPlan`] owns the fold id of every row and the train / validation index lists of
//! every fold, built once. Fold ids come from a [`FoldSource`]: a caller-supplied (shared)
//! assignment, a seeded stratified unit shuffle, or whole-cluster folds. Deriving the ids and
//! range-checking them lives here only; the pipelines name the source and the plan hands back
//! the per-fold row lists. The seeded planner and the row bookkeeping live in
//! `antecedent_learn::crossfit_plan`; this module adds the estimator-side refusals, the
//! cluster source and the fold engines.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_learn::LearnError;

use crate::error::EstimationError;

pub(crate) use antecedent_learn::crossfit_plan::FoldRows;

/// The refusal text of each fold-id derivation failure; callers keep their own wording.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FoldIdErrors {
    /// A shared assignment does not cover every row.
    pub length: &'static str,
    /// The cluster unit labels do not cover every row.
    pub units: &'static str,
    /// A fold id is not below the fold count.
    pub range: &'static str,
}

/// The wording of the batch-shared and per-pipeline AIPW cross-fit refusals.
pub(crate) const SHARED_FOLD_ID_ERRORS: FoldIdErrors = FoldIdErrors {
    length: "shared fold assignment length must match complete-case rows",
    units: "cluster fold units length must match complete-case rows",
    range: "shared fold ids must lie in 0..folds",
};

/// Where the fold id of every row comes from.
#[derive(Clone, Copy, Debug)]
pub(crate) enum FoldSource<'a> {
    /// A recorded assignment (one id per row), shared across a batch.
    Shared(&'a [u32]),
    /// The seeded unit-level shuffle dealt round-robin within each stratum.
    Stratified { strata: &'a [u32], units: &'a [antecedent_data::GlobalId], seed: u64 },
    /// Whole clusters dealt to folds (no stratification).
    Clustered { units: &'a [u32], seed: u64 },
}

/// Rows `rows` of a column-major `nrows × ncols` matrix as a fresh column-major buffer.
pub(crate) fn rows_colmajor(
    matrix: &[f64],
    nrows: usize,
    ncols: usize,
    rows: &[usize],
) -> Vec<f64> {
    let mut out = Vec::new();
    crate::aipw::select_rows_colmajor(matrix, nrows, ncols, rows, &mut out);
    out
}

/// Fold ids plus the per-fold training and validation row lists.
#[derive(Clone, Debug)]
pub(crate) struct CrossfitPlan(antecedent_learn::crossfit_plan::CrossfitPlan);

impl CrossfitPlan {
    /// Derive the fold ids of `n` rows from `source` and precompute every fold's row lists.
    ///
    /// # Errors
    ///
    /// A shared or cluster assignment that does not cover `n` rows, a fold id outside
    /// `0..folds`, or any failure of the seeded stratified plan.
    pub(crate) fn derive(
        n: usize,
        folds: usize,
        source: FoldSource<'_>,
        errors: &FoldIdErrors,
    ) -> Result<Self, EstimationError> {
        Ok(Self::from_fold_ids(folds, derive_fold_ids(n, folds, source, errors)?))
    }

    /// A plan over already validated fold ids (every id below `folds`).
    pub(crate) fn from_fold_ids(folds: usize, fold_ids: Vec<u32>) -> Self {
        Self(antecedent_learn::crossfit_plan::CrossfitPlan::from_fold_ids(folds, fold_ids))
    }

    /// The fold count.
    pub(crate) fn folds(&self) -> usize {
        self.0.folds()
    }

    /// One fold id per row.
    pub(crate) fn fold_ids(&self) -> &[u32] {
        self.0.fold_ids()
    }

    /// Row indices outside `fold`, in row order.
    pub(crate) fn train(&self, fold: usize) -> &[usize] {
        self.0.train(fold)
    }

    /// Row indices inside `fold`, in row order.
    pub(crate) fn valid(&self, fold: usize) -> &[usize] {
        self.0.valid(fold)
    }

    /// The rows of one fold.
    pub(crate) fn rows(&self, fold: usize) -> FoldRows<'_> {
        self.0.rows(fold)
    }

    /// The sequential cross-fit engine: for each fold in order, `poll` (cancellation), refuse
    /// an empty train or validation side with `empty` when given, `fit_nuisance` on the
    /// training rows, then `score` the validation rows with that nuisance.
    ///
    /// # Errors
    ///
    /// The first error of `poll`, an empty fold, `fit_nuisance` or `score`.
    pub(crate) fn run<N>(
        &self,
        empty: Option<&'static str>,
        mut poll: impl FnMut(usize) -> Result<(), EstimationError>,
        mut fit_nuisance: impl FnMut(FoldRows<'_>) -> Result<N, EstimationError>,
        mut score: impl FnMut(FoldRows<'_>, N) -> Result<(), EstimationError>,
    ) -> Result<(), EstimationError> {
        for fold in 0..self.folds() {
            poll(fold)?;
            let rows = self.rows(fold);
            if let Some(message) = empty {
                if rows.train.is_empty() || rows.valid.is_empty() {
                    return Err(EstimationError::data_msg(message));
                }
            }
            let nuisance = fit_nuisance(rows)?;
            score(rows, nuisance)?;
        }
        Ok(())
    }

    /// Fit folds on leased workers and consume their results in fold order on the caller,
    /// with at most `window` fold results alive (being produced, waiting, or being consumed).
    ///
    /// A consumer that stalls stops the fold fits after `window` folds instead of letting
    /// every fold's nuisance output pile up. The lowest-fold error wins.
    ///
    /// # Errors
    ///
    /// The error of the lowest failing fold or of `consume`, or a cancellation.
    pub(crate) fn pipeline_folds<T: Send, E: Send>(
        &self,
        ctx: &antecedent_core::ExecutionContext,
        window: usize,
        produce: impl Fn(FoldRows<'_>, &antecedent_core::ExecutionContext) -> Result<T, E> + Sync,
        mut consume: impl FnMut(usize, T) -> Result<(), E>,
    ) -> Result<
        antecedent_core::execution::PipelineReceipt,
        antecedent_core::execution::MapExecutionError<E>,
    > {
        ctx.pipeline_indexed(
            self.folds(),
            window,
            |fold, inner| produce(self.rows(fold), inner),
            |fold, value| {
                consume(fold, value).map(|()| antecedent_core::execution::PipelineStep::Continue)
            },
        )
    }

    /// Release the fold ids.
    pub(crate) fn into_fold_ids(self) -> Vec<u32> {
        self.0.into_fold_ids()
    }
}

/// Fold id of every row of `source`, length- and range-checked.
///
/// # Errors
///
/// See [`CrossfitPlan::derive`].
pub(crate) fn derive_fold_ids(
    n: usize,
    folds: usize,
    source: FoldSource<'_>,
    errors: &FoldIdErrors,
) -> Result<Vec<u32>, EstimationError> {
    let ids = match source {
        FoldSource::Shared(ids) => {
            if ids.len() != n {
                return Err(EstimationError::data_msg(errors.length));
            }
            ids.to_vec()
        }
        FoldSource::Clustered { units, seed } => {
            if units.len() != n {
                return Err(EstimationError::data_msg(errors.units));
            }
            crate::cluster_dml_aipw::cluster_fold_plan(units, folds, seed)?
        }
        FoldSource::Stratified { strata, units, seed } => {
            crossfit_fold_plan(strata, units, folds, seed)?
        }
    };
    if ids.iter().any(|&id| id as usize >= folds) {
        return Err(EstimationError::data_msg(errors.range));
    }
    Ok(ids)
}

/// The estimator-side error of a planner refusal: the fold-count refusal stays unsupported
/// and every data refusal keeps its message.
fn plan_error(error: LearnError) -> EstimationError {
    match error {
        LearnError::Unsupported { message } => EstimationError::unsupported(message),
        LearnError::Shape { message } => EstimationError::data_msg(message),
        other => EstimationError::from(other),
    }
}

/// The seeded stratified fold id of every row, each row its own unit, as the `u16` ids the
/// retained fold assignments record; empty when the assignment is not defined (see
/// [`antecedent_learn::crossfit_plan::row_fold_ids_u16`]).
pub(crate) fn row_fold_ids_u16(strata: &[u32], folds: usize, seed: u64) -> Vec<u16> {
    antecedent_learn::crossfit_plan::row_fold_ids_u16(strata, folds, seed)
}

/// Seeded, stratified, unit-level cross-fit fold plan (one fold id per row); see
/// [`antecedent_learn::crossfit_plan::crossfit_fold_plan`].
///
/// # Errors
///
/// `folds < 2` (unsupported), a length mismatch, or fewer distinct units than folds (data).
pub(crate) fn crossfit_fold_plan<U>(
    strata: &[u32],
    units: &[U],
    folds: usize,
    seed: u64,
) -> Result<Vec<u32>, EstimationError>
where
    U: Copy + Ord + std::hash::Hash,
{
    antecedent_learn::crossfit_plan::crossfit_fold_plan(strata, units, folds, seed)
        .map_err(plan_error)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planner_refusals_keep_their_estimator_variants_and_messages() {
        let few = crossfit_fold_plan(&[0, 0], &[1u64, 2], 1, 5).unwrap_err();
        assert!(matches!(few, EstimationError::Unsupported { .. }), "{few:?}");
        assert!(few.to_string().contains("cross-fitting requires at least two folds"), "{few}");
        let misaligned = crossfit_fold_plan(&[0, 0, 0], &[1u64, 2], 2, 5).unwrap_err();
        assert!(misaligned.to_string().contains("strata and unit ids must align"), "{misaligned}");
        let few_units = crossfit_fold_plan(&[0, 0, 0], &[1u64, 1, 2], 3, 5).unwrap_err();
        assert!(few_units.to_string().contains("distinct units"), "{few_units}");
    }

    #[test]
    fn shared_ids_are_length_and_range_checked_with_the_callers_wording() {
        let errors = FoldIdErrors { length: "len", units: "units", range: "range" };
        let short = CrossfitPlan::derive(3, 2, FoldSource::Shared(&[0, 1]), &errors).unwrap_err();
        assert!(short.to_string().contains("len"), "{short}");
        let high = CrossfitPlan::derive(3, 2, FoldSource::Shared(&[0, 1, 2]), &errors).unwrap_err();
        assert!(high.to_string().contains("range"), "{high}");
        let bad_units =
            CrossfitPlan::derive(3, 2, FoldSource::Clustered { units: &[1, 2], seed: 0 }, &errors)
                .unwrap_err();
        assert!(bad_units.to_string().contains("units"), "{bad_units}");
    }

    #[test]
    fn plan_lists_partition_the_rows_per_fold() {
        let plan = CrossfitPlan::derive(
            5,
            2,
            FoldSource::Shared(&[0, 1, 1, 0, 1]),
            &SHARED_FOLD_ID_ERRORS,
        )
        .unwrap();
        assert_eq!(plan.folds(), 2);
        assert_eq!(plan.valid(0), &[0, 3]);
        assert_eq!(plan.train(0), &[1, 2, 4]);
        assert_eq!(plan.valid(1), &[1, 2, 4]);
        assert_eq!(plan.train(1), &[0, 3]);
        assert_eq!(plan.fold_ids(), &[0, 1, 1, 0, 1]);
    }

    #[test]
    fn stratified_source_equals_the_free_function() {
        let units: Vec<antecedent_data::GlobalId> =
            (0..40).map(antecedent_data::GlobalId::new).collect();
        let strata: Vec<u32> =
            units.iter().map(|u| u32::try_from(u.get() % 2).expect("bit")).collect();
        let direct = crossfit_fold_plan(&strata, &units, 4, 9).unwrap();
        let plan = CrossfitPlan::derive(
            40,
            4,
            FoldSource::Stratified { strata: &strata, units: &units, seed: 9 },
            &SHARED_FOLD_ID_ERRORS,
        )
        .unwrap();
        assert_eq!(plan.fold_ids(), direct.as_slice());
    }

    #[test]
    fn a_stalled_fold_consumer_keeps_live_fold_results_at_the_window() {
        use antecedent_core::ExecutionContext;
        use antecedent_core::execution::AdmissionPool;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let ids: Vec<u32> = (0..240).map(|i| i % 12).collect();
        let plan = CrossfitPlan::from_fold_ids(12, ids);
        let pool = AdmissionPool::fixed(4, 1 << 20);
        let ctx = ExecutionContext::production(1, 4).with_admission(pool.clone());
        let live = AtomicUsize::new(0);
        let peak = AtomicUsize::new(0);
        let mut order = Vec::new();
        let receipt = plan
            .pipeline_folds(
                &ctx,
                3,
                |rows, _| {
                    peak.fetch_max(live.fetch_add(1, Ordering::SeqCst) + 1, Ordering::SeqCst);
                    Ok::<_, ()>(rows.valid.to_vec())
                },
                |fold, valid| {
                    if fold == 0 {
                        std::thread::sleep(std::time::Duration::from_millis(80));
                    }
                    assert!(valid.iter().all(|&i| i % 12 == fold));
                    order.push(fold);
                    live.fetch_sub(1, Ordering::SeqCst);
                    Ok(())
                },
            )
            .unwrap();
        assert_eq!(order, (0..12).collect::<Vec<_>>());
        assert!(peak.load(Ordering::SeqCst) <= 3, "{peak:?}");
        assert!(receipt.peak_live <= 3);
        assert_eq!(pool.available().threads(), 4);
    }
}
