//! The seeded, stratified, unit-level cross-fit fold planner and its row bookkeeping.
//!
//! [`crossfit_fold_plan`] gives every row a fold id as a pure function of the seed, the strata
//! and the unit ids; [`CrossfitPlan`] holds those ids with the train / validation row list of
//! every fold. Both are free of estimator types, so any layer that must replay a cross-fitted
//! estimate derives the same folds from the same inputs.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{CausalRng, INDEX_GAMMA};

use crate::error::LearnError;

/// One fold's rows.
#[derive(Clone, Copy, Debug)]
pub struct FoldRows<'a> {
    /// The fold index.
    pub fold: usize,
    /// Rows outside the fold, in row order.
    pub train: &'a [usize],
    /// Rows inside the fold, in row order.
    pub valid: &'a [usize],
}

/// Fold ids plus the per-fold training and validation row lists.
#[derive(Clone, Debug)]
pub struct CrossfitPlan {
    folds: usize,
    fold_ids: Vec<u32>,
    train: Vec<Vec<usize>>,
    valid: Vec<Vec<usize>>,
}

impl CrossfitPlan {
    /// A plan over already validated fold ids (every id below `folds`).
    #[must_use]
    pub fn from_fold_ids(folds: usize, fold_ids: Vec<u32>) -> Self {
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
    #[must_use]
    pub fn folds(&self) -> usize {
        self.folds
    }

    /// One fold id per row.
    #[must_use]
    pub fn fold_ids(&self) -> &[u32] {
        &self.fold_ids
    }

    /// Row indices outside `fold`, in row order.
    #[must_use]
    pub fn train(&self, fold: usize) -> &[usize] {
        &self.train[fold]
    }

    /// Row indices inside `fold`, in row order.
    #[must_use]
    pub fn valid(&self, fold: usize) -> &[usize] {
        &self.valid[fold]
    }

    /// The rows of one fold.
    #[must_use]
    pub fn rows(&self, fold: usize) -> FoldRows<'_> {
        FoldRows { fold, train: &self.train[fold], valid: &self.valid[fold] }
    }

    /// The sequential cross-fit engine: for each fold in order, `poll` (cancellation), refuse
    /// an empty train or validation side with `empty` when given, `fit_nuisance` on the
    /// training rows, then `score` the validation rows with that nuisance.
    ///
    /// # Errors
    ///
    /// The first error of `poll`, an empty fold, `fit_nuisance` or `score`.
    #[allow(
        clippy::needless_pass_by_value,
        reason = "the optional callback is owned for the duration of the cross-fit"
    )]
    pub fn run<N, E>(
        &self,
        empty: Option<impl Fn() -> E>,
        mut poll: impl FnMut(usize) -> Result<(), E>,
        mut fit_nuisance: impl FnMut(FoldRows<'_>) -> Result<N, E>,
        mut score: impl FnMut(FoldRows<'_>, N) -> Result<(), E>,
    ) -> Result<(), E> {
        for fold in 0..self.folds() {
            poll(fold)?;
            let rows = self.rows(fold);
            if let Some(refuse) = &empty
                && (rows.train.is_empty() || rows.valid.is_empty())
            {
                return Err(refuse());
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
    pub fn pipeline_folds<T: Send, E: Send>(
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
    #[must_use]
    pub fn into_fold_ids(self) -> Vec<u32> {
        self.fold_ids
    }
}

/// The seeded stratified fold id of every row, each row its own unit, as the `u16` ids the
/// retained fold assignments record.
///
/// Empty when the assignment is not defined: fewer than two folds, more folds than rows in
/// the smallest stratum's union, or more than `u32::MAX` rows or `u16::MAX + 1` folds. Every
/// caller validates its bounds first, so an empty plan never reaches a fit, and a retained
/// assignment of the right length never equals it.
#[must_use]
pub fn row_fold_ids_u16(strata: &[u32], folds: usize, seed: u64) -> Vec<u16> {
    let Ok(n) = u64::try_from(strata.len()) else { return Vec::new() };
    let units: Vec<u64> = (0..n).collect();
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
/// [`LearnError::Unsupported`] for `folds < 2`; [`LearnError::Shape`] for a length mismatch
/// between `strata` and `units`, or fewer distinct units than folds.
#[allow(
    clippy::cast_possible_truncation,
    reason = "the product is a uniform draw in [0, 1) times i + 1, so it is at most i; the index is additionally clamped to i; dealt % folds is below the fold count, which was checked to fit u32 on entry"
)]
#[allow(
    clippy::cast_sign_loss,
    reason = "the product is a non-negative uniform draw in [0, 1) scaled by i + 1"
)]
pub fn crossfit_fold_plan<U>(
    strata: &[u32],
    units: &[U],
    folds: usize,
    seed: u64,
) -> Result<Vec<u32>, LearnError>
where
    U: Copy + Ord + std::hash::Hash,
{
    if folds < 2 || u32::try_from(folds).is_err() {
        return Err(LearnError::Unsupported {
            message: "cross-fitting requires at least two folds",
        });
    }
    if strata.len() != units.len() {
        return Err(LearnError::Shape { message: "fold plan strata and unit ids must align" });
    }
    let mut stratum_of = std::collections::BTreeMap::new();
    for (&u, &s) in units.iter().zip(strata) {
        stratum_of.entry(u).or_insert(s);
    }
    if stratum_of.len() < folds {
        return Err(LearnError::Shape {
            message: "cross-fitting folds cannot exceed distinct units",
        });
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
            seed ^ u64::from(stratum).wrapping_add(1).wrapping_mul(INDEX_GAMMA),
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
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    reason = "the reference restates the documented algorithm with the same small-index casts"
)]
mod tests {
    use super::*;

    /// An independent restatement of the documented algorithm, written without the production
    /// maps: distinct units keep the stratum of their first row, each stratum's units are
    /// listed in ascending unit order, shuffled by a Fisher–Yates pass keyed
    /// `seed ^ (stratum + 1) * GAMMA`, and dealt to folds by one global round-robin counter in
    /// ascending stratum order.
    fn reference(strata: &[u32], units: &[u64], folds: usize, seed: u64) -> Vec<u32> {
        const GAMMA: u64 = 0xD1B5_4A32_D192_ED03;
        let mut firsts: Vec<(u32, u64)> = Vec::new();
        for (&u, &s) in units.iter().zip(strata) {
            if !firsts.iter().any(|&(_, v)| v == u) {
                firsts.push((s, u));
            }
        }
        firsts.sort_unstable();
        let mut assigned: Vec<(u64, u32)> = Vec::new();
        let mut counter = 0usize;
        let mut start = 0;
        while start < firsts.len() {
            let stratum = firsts[start].0;
            let mut end = start;
            while end < firsts.len() && firsts[end].0 == stratum {
                end += 1;
            }
            let mut members: Vec<u64> = firsts[start..end].iter().map(|&(_, u)| u).collect();
            let mut rng = CausalRng::from_seed(seed ^ (u64::from(stratum) + 1).wrapping_mul(GAMMA));
            let mut i = members.len();
            while i > 1 {
                i -= 1;
                let draw = rng.next_f64();
                let j = ((draw * (i as f64 + 1.0)) as usize).min(i);
                members.swap(i, j);
            }
            for u in members {
                assigned.push((u, (counter % folds) as u32));
                counter += 1;
            }
            start = end;
        }
        units
            .iter()
            .map(|u| assigned.iter().find(|&&(v, _)| v == *u).expect("every unit is assigned").1)
            .collect()
    }

    #[test]
    fn the_stream_multiplier_is_the_documented_odd_constant() {
        assert_eq!(INDEX_GAMMA, 0xD1B5_4A32_D192_ED03);
        assert_eq!(INDEX_GAMMA & 1, 1);
    }

    #[test]
    fn single_stratum_twelve_units_four_folds_matches_the_reference() {
        let units: Vec<u64> = (0..12).collect();
        let strata = [0u32; 12];
        let plan = crossfit_fold_plan(&strata, &units, 4, 7).unwrap();
        assert_eq!(plan, reference(&strata, &units, 4, 7));
        for fold in 0..4u32 {
            assert_eq!(plan.iter().filter(|&&f| f == fold).count(), 3);
        }
    }

    #[test]
    fn two_imbalanced_strata_three_folds_match_the_reference() {
        // Seven units in stratum 0 and four in stratum 1: the counter runs across strata, so
        // the second stratum starts where the first stopped.
        let units: Vec<u64> = (0..11).collect();
        let strata: Vec<u32> = (0..11).map(|i| u32::from(i >= 7)).collect();
        let plan = crossfit_fold_plan(&strata, &units, 3, 11).unwrap();
        assert_eq!(plan, reference(&strata, &units, 3, 11));
        let in_stratum =
            |s: u32, f: u32| (0..11).filter(|&i| strata[i] == s && plan[i] == f).count();
        for fold in 0..3 {
            assert!((2..=3).contains(&in_stratum(0, fold)), "fold {fold}");
            assert!((1..=2).contains(&in_stratum(1, fold)), "fold {fold}");
        }
    }

    #[test]
    fn duplicated_units_share_one_fold_and_match_the_reference() {
        let units = [7u64, 7, 3, 3, 3, 9, 1, 2, 4, 5, 6, 8];
        let strata = [0u32; 12];
        let plan = crossfit_fold_plan(&strata, &units, 4, 5).unwrap();
        assert_eq!(plan, reference(&strata, &units, 4, 5));
        assert_eq!(plan[0], plan[1]);
        assert_eq!(plan[2], plan[3]);
        assert_eq!(plan[3], plan[4]);
        assert!(plan.iter().all(|&f| f < 4));
    }

    #[test]
    fn the_plan_equals_the_reference_across_seeds_and_shapes() {
        for n in [4usize, 9, 25, 64] {
            for strata_count in [1u32, 2, 3] {
                for folds in [2usize, 3, 5] {
                    if folds > n / strata_count as usize {
                        continue;
                    }
                    for seed in [0u64, 1, 7, 11, 12, 99, u64::MAX, 0xDEAD_BEEF] {
                        // Unit ids are a scrambled, partly repeated labelling of the rows.
                        let units: Vec<u64> =
                            (0..n).map(|i| ((i * 37 + 5) % (n - n / 4)) as u64).collect();
                        let strata: Vec<u32> =
                            (0..n).map(|i| (i as u32 * 5 + 1) % strata_count).collect();
                        let distinct = {
                            let mut u = units.clone();
                            u.sort_unstable();
                            u.dedup();
                            u.len()
                        };
                        if distinct < folds {
                            continue;
                        }
                        let got = crossfit_fold_plan(&strata, &units, folds, seed).unwrap();
                        assert_eq!(
                            got,
                            reference(&strata, &units, folds, seed),
                            "n {n} strata {strata_count} folds {folds} seed {seed}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn the_unit_type_does_not_change_the_plan() {
        let units: Vec<u64> = (0..30).map(|i| (i * 7) % 30).collect();
        let narrow: Vec<u32> = units.iter().map(|&u| u as u32).collect();
        let strata: Vec<u32> = (0..30).map(|i| i % 2).collect();
        assert_eq!(
            crossfit_fold_plan(&strata, &units, 5, 3).unwrap(),
            crossfit_fold_plan(&strata, &narrow, 5, 3).unwrap()
        );
    }

    #[test]
    fn fold_plan_is_seeded_stratified_and_balanced() {
        // 100 units, stratum = parity: each fold must hold exactly 10 units of each stratum,
        // and the plan must change with the seed and not follow `i % folds`.
        let units: Vec<u64> = (0..100).collect();
        let strata: Vec<u32> = units.iter().map(|u| u32::try_from(u % 2).expect("bit")).collect();
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
    fn the_planner_refuses_few_folds_misaligned_inputs_and_too_few_units() {
        let few = crossfit_fold_plan(&[0, 0], &[1u64, 2], 1, 5).unwrap_err();
        assert_eq!(
            few,
            LearnError::Unsupported { message: "cross-fitting requires at least two folds" }
        );
        let zero = crossfit_fold_plan(&[0, 0], &[1u64, 2], 0, 5).unwrap_err();
        assert!(matches!(zero, LearnError::Unsupported { .. }));
        let misaligned = crossfit_fold_plan(&[0, 0, 0], &[1u64, 2], 2, 5).unwrap_err();
        assert_eq!(
            misaligned,
            LearnError::Shape { message: "fold plan strata and unit ids must align" }
        );
        let too_few = crossfit_fold_plan(&[0, 0, 0], &[1u64, 1, 2], 3, 5).unwrap_err();
        assert_eq!(
            too_few,
            LearnError::Shape { message: "cross-fitting folds cannot exceed distinct units" }
        );
    }

    #[test]
    fn plan_lists_partition_the_rows_per_fold() {
        let plan = CrossfitPlan::from_fold_ids(2, vec![0, 1, 1, 0, 1]);
        assert_eq!(plan.folds(), 2);
        assert_eq!(plan.valid(0), &[0, 3]);
        assert_eq!(plan.train(0), &[1, 2, 4]);
        assert_eq!(plan.valid(1), &[1, 2, 4]);
        assert_eq!(plan.train(1), &[0, 3]);
        assert_eq!(plan.fold_ids(), &[0, 1, 1, 0, 1]);
        let rows = plan.rows(1);
        assert_eq!((rows.fold, rows.train, rows.valid), (1, &[0, 3][..], &[1, 2, 4][..]));
        assert_eq!(plan.into_fold_ids(), vec![0, 1, 1, 0, 1]);
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
    fn row_fold_ids_equal_the_plan_over_row_index_units() {
        let strata: Vec<u32> = (0..45).map(|i| i % 3).collect();
        let units: Vec<u64> = (0..45).collect();
        let wide = crossfit_fold_plan(&strata, &units, 5, 17).unwrap();
        let narrow: Vec<u16> = wide.iter().map(|&f| u16::try_from(f).unwrap()).collect();
        assert_eq!(row_fold_ids_u16(&strata, 5, 17), narrow);
    }
}
