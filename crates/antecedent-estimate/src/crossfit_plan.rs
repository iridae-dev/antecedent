//! The one cross-fit fold plan shared by every cross-fitted pipeline.
//!
//! A [`CrossfitPlan`] owns the fold id of every row and the train / validation index lists of
//! every fold, built once. Fold ids come from a [`FoldSource`]: a caller-supplied (shared)
//! assignment, a seeded stratified unit shuffle, or whole-cluster folds. Deriving the ids and
//! range-checking them lives here only; the pipelines name the source and the plan hands back
//! the per-fold row lists.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::CausalRng;

use crate::error::EstimationError;

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

/// One fold's rows.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FoldRows<'a> {
    /// The fold index.
    pub fold: usize,
    /// Rows outside the fold, in row order.
    pub train: &'a [usize],
    /// Rows inside the fold, in row order.
    pub valid: &'a [usize],
}

/// Fold ids plus the per-fold training and validation row lists.
#[derive(Clone, Debug)]
pub(crate) struct CrossfitPlan {
    folds: usize,
    fold_ids: Vec<u32>,
    train: Vec<Vec<usize>>,
    valid: Vec<Vec<usize>>,
}

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
        let mut train = Vec::with_capacity(folds);
        let mut valid = Vec::with_capacity(folds);
        for fold in 0..folds {
            let (v, t): (Vec<usize>, Vec<usize>) =
                (0..fold_ids.len()).partition(|&i| fold_ids[i] as usize == fold);
            train.push(t);
            valid.push(v);
        }
        Self { folds, fold_ids, train, valid }
    }

    /// The fold count.
    pub(crate) fn folds(&self) -> usize {
        self.folds
    }

    /// One fold id per row.
    pub(crate) fn fold_ids(&self) -> &[u32] {
        &self.fold_ids
    }

    /// Row indices outside `fold`, in row order.
    pub(crate) fn train(&self, fold: usize) -> &[usize] {
        &self.train[fold]
    }

    /// Row indices inside `fold`, in row order.
    pub(crate) fn valid(&self, fold: usize) -> &[usize] {
        &self.valid[fold]
    }

    /// The rows of one fold.
    pub(crate) fn rows(&self, fold: usize) -> FoldRows<'_> {
        FoldRows { fold, train: &self.train[fold], valid: &self.valid[fold] }
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
        for fold in 0..self.folds {
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
            self.folds,
            window,
            |fold, inner| produce(self.rows(fold), inner),
            |fold, value| {
                consume(fold, value).map(|()| antecedent_core::execution::PipelineStep::Continue)
            },
        )
    }

    /// Release the fold ids.
    pub(crate) fn into_fold_ids(self) -> Vec<u32> {
        self.fold_ids
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

/// The seeded stratified fold id of every row, each row its own unit, as the `u16` ids the
/// retained fold assignments record.
///
/// Empty when the assignment is not defined: fewer than two folds, more folds than rows in
/// the smallest stratum's union, or more than `u32::MAX` rows or `u16::MAX + 1` folds. Every
/// caller validates its bounds first, so an empty plan never reaches a fit, and a retained
/// assignment of the right length never equals it.
pub(crate) fn row_fold_ids_u16(strata: &[u32], folds: usize, seed: u64) -> Vec<u16> {
    let Ok(n) = u64::try_from(strata.len()) else { return Vec::new() };
    let units: Vec<antecedent_data::GlobalId> =
        (0..n).map(antecedent_data::GlobalId::new).collect();
    match crossfit_fold_plan(strata, &units, folds, seed) {
        Ok(ids) => ids.into_iter().filter_map(|id| u16::try_from(id).ok()).collect(),
        Err(_) => Vec::new(),
    }
}

/// Seeded, stratified, unit-level cross-fit fold plan (one fold id per row).
///
/// `units[i]` names the physical unit behind row `i` (the original row index): every row of a
/// unit — a bootstrap resample's duplicates — shares one fold, so a unit is never both in a
/// training set and in its own validation fold. Within each stratum (`strata[i]`, e.g. the
/// treatment arm) the distinct units are permuted by a `seed`-keyed Fisher–Yates shuffle and
/// dealt round-robin, so every fold sees every stratum and the plan is a function of the seed
/// and the unit ids, not of any periodic structure in the file order.
///
/// # Errors
///
/// `folds < 2`, length mismatch, more distinct units than the fold count allows, or fewer
/// distinct units than folds.
#[allow(
    clippy::cast_possible_truncation,
    reason = "the product is a uniform draw in [0, 1) times i + 1, so it is at most i; the index is additionally clamped to i; dealt % folds is below the fold count, which was checked to fit u32 on entry"
)]
#[allow(
    clippy::cast_sign_loss,
    reason = "the product is a non-negative uniform draw in [0, 1) scaled by i + 1"
)]
pub(crate) fn crossfit_fold_plan<U>(
    strata: &[u32],
    units: &[U],
    folds: usize,
    seed: u64,
) -> Result<Vec<u32>, EstimationError>
where
    U: Copy + Ord + std::hash::Hash,
{
    if folds < 2 || u32::try_from(folds).is_err() {
        return Err(EstimationError::unsupported("cross-fitting requires at least two folds"));
    }
    if strata.len() != units.len() {
        return Err(EstimationError::data_msg("fold plan strata and unit ids must align"));
    }
    let mut stratum_of = std::collections::BTreeMap::new();
    for (&u, &s) in units.iter().zip(strata) {
        stratum_of.entry(u).or_insert(s);
    }
    if stratum_of.len() < folds {
        return Err(EstimationError::data_msg("cross-fitting folds cannot exceed distinct units"));
    }
    // Units grouped by stratum, each group sorted by unit id (determinism), then shuffled.
    let mut groups: std::collections::BTreeMap<u32, Vec<U>> = std::collections::BTreeMap::new();
    for (&u, &s) in &stratum_of {
        groups.entry(s).or_default().push(u);
    }
    let mut fold_of_unit = std::collections::HashMap::with_capacity(stratum_of.len());
    let mut dealt = 0usize;
    for (&stratum, members) in &mut groups {
        let mut rng = CausalRng::from_seed(
            seed ^ u64::from(stratum).wrapping_add(1).wrapping_mul(crate::splitmix::INDEX_GAMMA),
        );
        for i in (1..members.len()).rev() {
            let j = (rng.next_f64() * (i as f64 + 1.0)) as usize;
            members.swap(i, j.min(i));
        }
        for &u in members.iter() {
            fold_of_unit.insert(u, (dealt % folds) as u32);
            dealt += 1;
        }
    }
    Ok(units.iter().map(|u| fold_of_unit[u]).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fold_plan_is_seeded_stratified_and_balanced() {
        // 100 units, stratum = parity: each fold must hold exactly 10 units of each stratum,
        // and the plan must change with the seed and not follow `i % folds`.
        let units: Vec<antecedent_data::GlobalId> =
            (0..100).map(antecedent_data::GlobalId::new).collect();
        let strata: Vec<u32> =
            units.iter().map(|u| u32::try_from(u.get() % 2).expect("bit")).collect();
        let plan = crossfit_fold_plan(&strata, &units, 5, 11).unwrap();
        for fold in 0..5u32 {
            for stratum in 0..2u32 {
                let count = (0..100).filter(|&i| plan[i] == fold && strata[i] == stratum).count();
                assert_eq!(count, 10, "fold {fold} stratum {stratum}");
            }
        }
        let modulo: Vec<u32> = (0..100).map(|i| i % 5).collect();
        assert_ne!(plan, modulo);
        assert_ne!(plan, crossfit_fold_plan(&strata, &units, 5, 12).unwrap());
        assert_eq!(plan, crossfit_fold_plan(&strata, &units, 5, 11).unwrap());
    }

    #[test]
    fn fold_plan_keeps_duplicated_units_together_and_refuses_too_few_units() {
        // A bootstrap resample repeats unit ids; every copy must share its unit's fold.
        let units = [7u64, 7, 3, 3, 3, 9, 1, 2, 4, 5, 6, 8].map(antecedent_data::GlobalId::new);
        let strata = [0u32; 12];
        let plan = crossfit_fold_plan(&strata, &units, 4, 5).unwrap();
        for (i, &u) in units.iter().enumerate() {
            for (j, &v) in units.iter().enumerate() {
                if u == v {
                    assert_eq!(plan[i], plan[j], "unit {}", u.get());
                }
            }
        }
        assert!(plan.iter().all(|&f| f < 4));
        let err = crossfit_fold_plan(
            &[0, 0, 0],
            &[
                antecedent_data::GlobalId::new(1),
                antecedent_data::GlobalId::new(1),
                antecedent_data::GlobalId::new(2),
            ],
            3,
            5,
        )
        .unwrap_err();
        assert!(err.to_string().contains("distinct units"), "{err}");
        assert!(
            crossfit_fold_plan(
                &[0, 0],
                &[antecedent_data::GlobalId::new(1), antecedent_data::GlobalId::new(2)],
                1,
                5,
            )
            .is_err()
        );
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
    fn row_fold_ids_deal_every_stratum_across_every_fold_and_refuse_by_emptiness() {
        let strata: Vec<u32> = (0..60).map(|i| u32::from(i % 3 == 0)).collect();
        let ids = row_fold_ids_u16(&strata, 4, 21);
        assert_eq!(ids.len(), 60);
        for stratum in 0..2u32 {
            let rows = (0..60).filter(|&i| strata[i] == stratum).count();
            for fold in 0..4u16 {
                let held = (0..60).filter(|&i| strata[i] == stratum && ids[i] == fold).count();
                assert!(held.abs_diff(rows / 4) <= 1, "stratum {stratum} fold {fold}: {held}");
            }
        }
        assert_eq!(ids, row_fold_ids_u16(&strata, 4, 21));
        assert_ne!(ids, row_fold_ids_u16(&strata, 4, 22));
        assert!(row_fold_ids_u16(&strata, 1, 21).is_empty());
        assert!(row_fold_ids_u16(&strata[..3], 4, 21).is_empty());
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
