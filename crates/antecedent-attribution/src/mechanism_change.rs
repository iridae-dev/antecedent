//! Mechanism-change *detection* — separate from attribution.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    CausalRng, ExecutionContext, MechanismChangeQuery, StreamDomain, VariableId,
};
use antecedent_data::{TableView, TabularData};
use antecedent_model::{
    CompiledCausalModel, MechanismRegistry, ParentBatch, SelectionPolicy, infer_noise_column_rng,
};
use antecedent_stats::{
    FdrAdjustment, adjust_pvalues, change_point_two_sample, classifier_two_sample,
    kernel_two_sample, mean_diff_two_sample, residual_likelihood_ratio,
};

use crate::error::AttributionError;
use crate::population::subset_table;
use crate::prep::resolve_population_pair;
use crate::result::MechanismChangeDetection;

/// Minimum rows required in each of the baseline and comparison populations.
///
/// Below this, neither a residual variance estimate nor an out-of-sample cross-fit is
/// reliable: a population this small can be exactly constant by chance (guaranteeing a
/// degenerate test) and cannot support the asymptotics the p-values rely on.
const MIN_POPULATION_ROWS: usize = 10;

/// Detection method selector.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum MechanismChangeMethod {
    /// Compare residual distributions via Gaussian KL / LR proxy.
    LikelihoodRatio,
    /// Mean difference on structural residuals (mechanism noise).
    MeanDiff,
    /// Classifier / two-sample proxy on residuals.
    ClassifierTwoSample,
    /// Kernel two-sample (MMD² + RBF) on residuals.
    KernelTwoSample,
    /// Known-split change-point test on concatenated baseline→comparison residuals.
    ChangePoint,
}

/// Detect which mechanisms differ between baseline and comparison populations.
///
/// This does **not** attribute outcome change — a changed mechanism need not
/// drive the target.
///
/// Applies a Benjamini–Hochberg false-discovery-rate correction across
/// `query.targets` by default (see
/// [`detect_mechanism_changes_with_correction`] to override or disable this).
/// `query.targets` is bounded by `query.max_targets`, i.e. the API is designed
/// to scan multiple candidate mechanisms in a single call, so judging each
/// target against the nominal `significance_level` independently would let the
/// family-wise false-positive rate grow with the number of targets (≈40% at
/// ten truly-unchanged targets and α=0.05). BH is chosen as the default
/// because it is the same default `antecedent_discovery::PcmciPlus` uses for
/// an analogous multi-hypothesis family (`PcmciPlus::new` sets
/// `fdr: Some(FdrAdjustment::bh())`), and it controls FDR rather than the
/// stricter (and more conservative) family-wise error rate, which is the
/// usual choice when the family is a moderate, human-reviewed set of
/// candidate mechanisms rather than a single make-or-break decision.
///
/// # Errors
///
/// Query / fit / stats failures.
pub fn detect_mechanism_changes(
    graph_model: &CompiledCausalModel,
    data: &TabularData,
    query: &MechanismChangeQuery,
    method: MechanismChangeMethod,
    ctx: &ExecutionContext,
) -> Result<Vec<MechanismChangeDetection>, AttributionError> {
    detect_mechanism_changes_with_correction(
        graph_model,
        data,
        query,
        method,
        ctx,
        Some(FdrAdjustment::bh()),
    )
}

/// Detect which mechanisms differ between baseline and comparison populations,
/// with explicit control over the multiple-testing correction applied across
/// `query.targets`.
///
/// `correction = None` reproduces the historical, uncorrected behavior where
/// each target's raw p-value is compared against `query.significance_level`
/// independently. `Some(adjustment)` adjusts the family of per-target p-values
/// with `adjustment.method` (see `antecedent_stats::fdr`) before thresholding.
///
/// [`MechanismChangeDetection::p_value`] always carries the *raw* per-target
/// p-value, and [`MechanismChangeDetection::adjusted_p_value`] carries the
/// corrected one (`None` when `correction` is `None`). `changed` is decided
/// from the adjusted value when present, so the flag and the number behind it
/// always agree while the unadjusted per-test result stays visible.
///
/// This does **not** attribute outcome change — a changed mechanism need not
/// drive the target.
///
/// # Errors
///
/// Query / fit / stats failures.
pub fn detect_mechanism_changes_with_correction(
    graph_model: &CompiledCausalModel,
    data: &TabularData,
    query: &MechanismChangeQuery,
    method: MechanismChangeMethod,
    ctx: &ExecutionContext,
    correction: Option<FdrAdjustment>,
) -> Result<Vec<MechanismChangeDetection>, AttributionError> {
    query.validate()?;
    if query.targets.len() > query.max_targets {
        return Err(AttributionError::SizeLimit {
            kind: "targets",
            requested: query.targets.len(),
            max: query.max_targets,
        });
    }
    let alpha = query.significance_level.to_f64();
    let (baseline, comparison) = resolve_population_pair(data, &query.baseline, &query.comparison)?;
    if baseline.row_count() < MIN_POPULATION_ROWS || comparison.row_count() < MIN_POPULATION_ROWS {
        return Err(AttributionError::invalid_input(
            "baseline and comparison populations need at least 10 rows each: fewer cannot \
             support a calibrated residual distribution and risk a degenerate (e.g. constant) \
             segment",
        ));
    }
    // A residual two-sample test asks "do these two noise samples look alike", which only
    // answers "did the mechanism change" when both samples are noise from the *same*
    // covariate region. If a target's parents differ materially in distribution between
    // the populations, the baseline-fitted mechanism's own parameter-estimation error
    // becomes a bias shared by every comparison residual alike (the same fitted
    // coefficients applied to every row), and that shared bias does not shrink with the
    // comparison sample size the way genuine per-row noise does — no reconstruction of the
    // residuals themselves can make a same-shape-distribution test calibrated against it;
    // the correct comparison is between the *fitted mechanisms* (e.g. a Chow/Wald test on
    // pooled regression with a population interaction term), not their residuals. Refusing
    // here is the honest fallback until that comparison is implemented: a wrong "changed"
    // claim from covariate drift is worse than no claim.
    for &target in query.targets.iter() {
        if parent_covariate_shift_detected(graph_model, &baseline, &comparison, target)? {
            return Err(AttributionError::invalid_input(
                "baseline and comparison populations differ materially in a tested \
                 mechanism's parent covariates; residual two-sample tests are not \
                 calibrated under a covariate shift with no accompanying mechanism change, \
                 so the comparison is refused",
            ));
        }
    }

    // Fitting the mechanism on the full baseline and handing that one model every
    // baseline row too would be in-sample for the baseline side (shrunk variance, no
    // estimation error) while comparison residuals stay out-of-sample — an asymmetry
    // that is anti-conservative on its own, independent of the covariate-shift defect
    // guarded against above. Cross-fitting the baseline fixes it: each baseline row's
    // residual comes from the one fold model that excluded it, a genuine held-out
    // prediction, matching the kind of estimation error already present on the
    // comparison side.
    let (base_mechs, _) = MechanismRegistry::standard().assign_and_fit(
        graph_model,
        &baseline,
        SelectionPolicy::BestScore,
    )?;
    let base_model = graph_model.clone().with_mechanisms(base_mechs);
    let fold_count = baseline_cross_fit_folds(baseline.row_count());
    let baseline_folds = kfold_row_indices(baseline.row_count(), fold_count);
    let mut fold_models = Vec::with_capacity(baseline_folds.len());
    for held_out in &baseline_folds {
        let train_rows: Vec<usize> =
            (0..baseline.row_count()).filter(|r| !held_out.contains(r)).collect();
        let train = subset_table(&baseline, &train_rows)?;
        let (mechs, _) = MechanismRegistry::standard().assign_and_fit(
            graph_model,
            &train,
            SelectionPolicy::BestScore,
        )?;
        fold_models.push(graph_model.clone().with_mechanisms(mechs));
    }

    let mut raw = Vec::with_capacity(query.targets.len());
    for &target in query.targets.iter() {
        let mut baseline_rng =
            ctx.rng.stream_for(StreamDomain::Attribution, residual_stream_id(target, 0));
        let mut comparison_rng =
            ctx.rng.stream_for(StreamDomain::Attribution, residual_stream_id(target, 1));
        let rb = cross_fitted_residuals(
            &fold_models,
            &baseline_folds,
            &baseline,
            target,
            &mut baseline_rng,
        )?;
        let rc = residuals(&base_model, &comparison, target, &mut comparison_rng)?;
        let (stat, p_value, method_name) = match method {
            MechanismChangeMethod::LikelihoodRatio => {
                let (s, p) = residual_likelihood_ratio(&rb, &rc)?;
                (s, p, "likelihood_ratio")
            }
            MechanismChangeMethod::MeanDiff => {
                let (s, p) = mean_diff_two_sample(&rb, &rc)?;
                (s, p, "mean_diff")
            }
            MechanismChangeMethod::ClassifierTwoSample => {
                let (s, p) = classifier_two_sample(&rb, &rc)?;
                (s, p, "classifier_two_sample")
            }
            MechanismChangeMethod::KernelTwoSample => {
                let seed = 0x_4E12_A001u64
                    .wrapping_add(target.as_usize() as u64)
                    .wrapping_mul(0x9E37_79B9);
                let (s, p) = kernel_two_sample(&rb, &rc, seed)?;
                (s, p, "kernel_two_sample")
            }
            MechanismChangeMethod::ChangePoint => {
                let (s, p) = change_point_two_sample(&rb, &rc)?;
                (s, p, "change_point")
            }
        };
        raw.push((target, stat, p_value, method_name));
    }

    // Report the raw p-value and the adjusted one separately rather than overwriting the
    // former. `changed` is decided from whichever is authoritative, but a caller reading
    // `p_value` must get the unadjusted per-target quantity its documentation promises —
    // silently returning a corrected number under that name is exactly the kind of
    // mislabelled statistic this correction exists to guard against.
    let adjusted: Option<Vec<f64>> = correction.map(|adjustment| {
        let p_values: Vec<f64> = raw.iter().map(|&(_, _, p, _)| p).collect();
        adjust_pvalues(&p_values, adjustment.method)
    });

    let mut out = Vec::with_capacity(raw.len());
    for (i, (target, stat, raw_p, method_name)) in raw.into_iter().enumerate() {
        let adjusted_p_value = adjusted.as_ref().map(|a| a[i]);
        let decisive = adjusted_p_value.unwrap_or(raw_p);
        out.push(MechanismChangeDetection {
            variable: target,
            changed: decisive < alpha,
            statistic: stat,
            p_value: raw_p,
            adjusted_p_value,
            method: Arc::from(method_name),
        });
    }
    Ok(out)
}

/// Distinct RNG stream per (target, population) so seeding is deterministic
/// under [`ExecutionContext::for_tests`] / any fixed `master_seed` but doesn't
/// alias across targets or across the baseline/comparison populations.
fn residual_stream_id(target: VariableId, population: u64) -> u64 {
    0x_5EED_C0DEu64
        .wrapping_add(target.as_usize() as u64)
        .wrapping_mul(0x9E37_79B9)
        .wrapping_add(population)
}

fn residuals(
    model: &CompiledCausalModel,
    data: &TabularData,
    target: VariableId,
    rng: &mut CausalRng,
) -> Result<Vec<f64>, AttributionError> {
    let dense =
        model.dense_of(target).ok_or_else(|| AttributionError::missing_var("target", target))?;
    let gather =
        model.gather_for(dense).ok_or(AttributionError::MissingArtifact("missing gather"))?;
    let n = data.row_count();
    let y = data.float64_values(target)?;
    let mut parent_mat = vec![0.0; n * gather.n_parents().max(1)];
    for (pi, &p) in gather.parents.iter().enumerate() {
        let pv = model.output_layout.variables[p.as_usize()];
        let col = data.float64_values(pv)?;
        parent_mat[pi * n..(pi + 1) * n].copy_from_slice(&col[..n]);
    }
    let parents = ParentBatch {
        n_rows: n,
        n_parents: gather.n_parents(),
        values: &parent_mat[..gather.n_parents().saturating_mul(n)],
    };
    let mut noise = vec![0.0; n];
    infer_noise_column_rng(model.mechanisms.get(dense), &y, parents, &mut noise, rng)?;
    Ok(noise)
}

/// Row indices for each held-out fold of a `k`-fold split of `0..n` (round-robin
/// assignment, so folds differ in size by at most one row).
fn kfold_row_indices(n: usize, k: usize) -> Vec<Vec<usize>> {
    let k = k.max(1);
    let mut folds = vec![Vec::new(); k];
    for row in 0..n {
        folds[row % k].push(row);
    }
    folds
}

/// Fold count for cross-fitting the baseline.
///
/// [`MIN_POPULATION_ROWS`] already guarantees `n >= 10`. More folds hold out less data per
/// fit (each fold model stays closer to a full-baseline fit) at the cost of more refits;
/// fewer folds keep every training set large relative to `n` when `n` is only just above
/// the minimum.
fn baseline_cross_fit_folds(n: usize) -> usize {
    if n >= 30 { 5 } else { 3 }
}

/// Out-of-sample baseline residuals for `target` via `k`-fold cross-fitting.
///
/// `fold_models[i]` must be fit on the rows *not* in `folds[i]`; each baseline row's
/// residual then comes from a mechanism that never saw it during fitting, rather than the
/// full-baseline fit's own (shrunk-variance) in-sample residual.
fn cross_fitted_residuals(
    fold_models: &[CompiledCausalModel],
    folds: &[Vec<usize>],
    population: &TabularData,
    target: VariableId,
    rng: &mut CausalRng,
) -> Result<Vec<f64>, AttributionError> {
    let mut out = vec![0.0; population.row_count()];
    for (model, assigned) in fold_models.iter().zip(folds.iter()) {
        let assigned_table = subset_table(population, assigned)?;
        let fold_residuals = residuals(model, &assigned_table, target, rng)?;
        for (&row, &value) in assigned.iter().zip(fold_residuals.iter()) {
            out[row] = value;
        }
    }
    Ok(out)
}

/// Significance level for the parent-covariate-shift guard in
/// [`parent_covariate_shift_detected`].
///
/// This is a nuisance check, not the mechanism-change hypothesis itself, so it does not
/// borrow `query.significance_level`. It is also checked once per parent of every target in
/// `query.targets`, uncorrected, so it needs its own margin against the same family-wise
/// inflation `detect_mechanism_changes`'s BH default exists to control for the primary
/// tests: a much stricter fixed threshold keeps the false-refusal rate low across a
/// realistic number of targets under a true null of "same parent distribution", while
/// still reliably catching the magnitude of shift that makes the residual tests
/// anti-conservative (a shift of even one parent standard deviation separates two
/// `n >= 10` samples by a z-score several orders of magnitude past this threshold).
const PARENT_SHIFT_ALPHA: f64 = 1e-4;

/// Whether `target`'s parent covariates differ materially between `baseline` and
/// `comparison`, via a two-sample mean-difference test on each parent column.
///
/// See the comment in [`detect_mechanism_changes_with_correction`] for why this matters: a
/// residual two-sample test cannot be calibrated against a covariate shift no matter how
/// the residuals are computed, so detecting the shift and refusing is the honest fallback.
///
/// # Errors
///
/// Missing variable / gather / column data.
fn parent_covariate_shift_detected(
    model: &CompiledCausalModel,
    baseline: &TabularData,
    comparison: &TabularData,
    target: VariableId,
) -> Result<bool, AttributionError> {
    let dense =
        model.dense_of(target).ok_or_else(|| AttributionError::missing_var("target", target))?;
    let gather =
        model.gather_for(dense).ok_or(AttributionError::MissingArtifact("missing gather"))?;
    for &parent in gather.parents.iter() {
        let pv = model.output_layout.variables[parent.as_usize()];
        let b = baseline.float64_values(pv)?;
        let c = comparison.float64_values(pv)?;
        if let Ok((_, p)) = mean_diff_two_sample(&b, &c) {
            if p < PARENT_SHIFT_ALPHA {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::{
        CausalSchemaBuilder, MeasurementSpec, PopulationSelector, RoleHint, SmallRoleSet, ValueType,
    };
    use antecedent_data::column::{Float64Column, ValidityBitmap};
    use antecedent_data::{OwnedColumn, OwnedColumnarStorage};
    use antecedent_graph::{Dag, DenseNodeId};

    #[test]
    fn detects_y_shift_not_necessarily_x() {
        let n = 80usize;
        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "x",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "y",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        let schema = b.build().unwrap();
        let mut xv = Vec::new();
        let mut yv = Vec::new();
        for i in 0..n {
            let x = (i % 40) as f64 * 0.1;
            xv.push(x);
            yv.push(if i < 40 { 1.0 + 2.0 * x } else { 6.0 + 2.0 * x });
        }
        let validity = ValidityBitmap::all_valid(n);
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(0), Arc::from(xv), validity.clone())
                    .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(1), Arc::from(yv), validity).unwrap(),
            ),
        ];
        let data =
            TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());
        let mut g = Dag::with_variables(2);
        g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let model = CompiledCausalModel::compile(g).unwrap();
        let q = MechanismChangeQuery::new(
            [VariableId::from_raw(0), VariableId::from_raw(1)],
            PopulationSelector::TimeRange { start: 0, end: 40 },
            PopulationSelector::TimeRange { start: 40, end: 80 },
            0.05,
            10,
        );
        let dets = detect_mechanism_changes(
            &model,
            &data,
            &q,
            MechanismChangeMethod::MeanDiff,
            &ExecutionContext::for_tests(1),
        )
        .unwrap();
        let y = dets.iter().find(|d| d.variable == VariableId::from_raw(1)).unwrap();
        assert!(y.changed, "y should be flagged changed: {y:?}");
    }

    fn two_period_data() -> (CompiledCausalModel, TabularData) {
        let n = 80usize;
        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "x",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "y",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        let schema = b.build().unwrap();
        let mut xv = Vec::new();
        let mut yv = Vec::new();
        for i in 0..n {
            let x = (i % 40) as f64 * 0.1;
            xv.push(x);
            yv.push(if i < 40 { 1.0 + 2.0 * x } else { 6.0 + 2.0 * x });
        }
        let validity = ValidityBitmap::all_valid(n);
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(0), Arc::from(xv), validity.clone())
                    .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(1), Arc::from(yv), validity).unwrap(),
            ),
        ];
        let data =
            TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());
        let mut g = Dag::with_variables(2);
        g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let model = CompiledCausalModel::compile(g).unwrap();
        (model, data)
    }

    #[test]
    fn kernel_two_sample_flags_y_shift() {
        let (model, data) = two_period_data();
        let q = MechanismChangeQuery::new(
            [VariableId::from_raw(0), VariableId::from_raw(1)],
            PopulationSelector::TimeRange { start: 0, end: 40 },
            PopulationSelector::TimeRange { start: 40, end: 80 },
            0.05,
            10,
        );
        let dets = detect_mechanism_changes(
            &model,
            &data,
            &q,
            MechanismChangeMethod::KernelTwoSample,
            &ExecutionContext::for_tests(1),
        )
        .unwrap();
        let y = dets.iter().find(|d| d.variable == VariableId::from_raw(1)).unwrap();
        assert!(y.changed, "y should be flagged changed: {y:?}");
        assert_eq!(&*y.method, "kernel_two_sample");
    }

    #[test]
    fn change_point_flags_y_shift() {
        let (model, data) = two_period_data();
        let q = MechanismChangeQuery::new(
            [VariableId::from_raw(0), VariableId::from_raw(1)],
            PopulationSelector::TimeRange { start: 0, end: 40 },
            PopulationSelector::TimeRange { start: 40, end: 80 },
            0.05,
            10,
        );
        let dets = detect_mechanism_changes(
            &model,
            &data,
            &q,
            MechanismChangeMethod::ChangePoint,
            &ExecutionContext::for_tests(1),
        )
        .unwrap();
        let y = dets.iter().find(|d| d.variable == VariableId::from_raw(1)).unwrap();
        assert!(y.changed, "y should be flagged changed: {y:?}");
        assert_eq!(&*y.method, "change_point");
    }

    /// A 1-row comparison population has zero within-population variance by
    /// construction. Before the fix, `mechanism_change` resolved populations directly
    /// (bypassing `prep::resolve_population_pair`'s emptiness check and imposing no
    /// minimum row count), and `LikelihoodRatio` reached `gaussian_segment_lr`'s
    /// `v1 == 0.0` branch, which reported `changed: true, p_value: 0.0, statistic: inf`
    /// — the strongest possible "changed" claim, manufactured from a single observation.
    /// The fix refuses any population below a documented minimum before fitting anything.
    #[test]
    fn refuses_single_row_comparison_population() {
        let (model, data) = two_period_data();
        let q = MechanismChangeQuery::new(
            [VariableId::from_raw(1)],
            PopulationSelector::TimeRange { start: 0, end: 40 },
            PopulationSelector::Rows(std::sync::Arc::from([17usize])),
            0.05,
            10,
        );
        for method in [MechanismChangeMethod::LikelihoodRatio, MechanismChangeMethod::ChangePoint] {
            let result = detect_mechanism_changes(
                &model,
                &data,
                &q,
                method,
                &ExecutionContext::for_tests(1),
            );
            assert!(
                result.is_err(),
                "a 1-row comparison population must be refused, not scored as a \
                 definitive mechanism-change claim (method={method:?}): {result:?}"
            );
        }
    }

    /// Same defect, reached via a constant (rather than merely small) comparison
    /// population: several rows, but every value bit-identical, so the segment still has
    /// zero variance. This is caught by the same minimum-row gate here (below
    /// `MIN_POPULATION_ROWS`); a comparison population at or above the minimum but still
    /// exactly constant is caught downstream by `gaussian_segment_lr`'s own refusal
    /// (`antecedent_stats::divergence::residual_lr_refuses_single_row_or_constant_segments`).
    #[test]
    fn refuses_undersized_comparison_population() {
        let (model, data) = two_period_data();
        let q = MechanismChangeQuery::new(
            [VariableId::from_raw(1)],
            PopulationSelector::TimeRange { start: 0, end: 40 },
            PopulationSelector::Rows(std::sync::Arc::from([10usize, 11, 12])),
            0.05,
            10,
        );
        let result = detect_mechanism_changes(
            &model,
            &data,
            &q,
            MechanismChangeMethod::LikelihoodRatio,
            &ExecutionContext::for_tests(1),
        );
        assert!(result.is_err(), "a 3-row comparison population must be refused: {result:?}");
    }

    /// `x`'s range genuinely differs by 5 units between the two populations here — exactly
    /// the material parent-covariate shift `parent_covariate_shift_detected` now refuses
    /// on, so this documents the current, honest behavior (refusal) rather than the
    /// original claim (a same-mechanism verdict). The original pass was itself an artifact
    /// of this test's near-deterministic 0.01-amplitude periodic "noise": with that little
    /// noise, the baseline fit has almost no parameter uncertainty to be wrong about, so
    /// the covariate shift never actually caused a false positive here even before the
    /// refusal existed — see `mean_diff_covariate_shift_is_calibrated_or_refused` for the
    /// case (realistic Gaussian noise) where it does.
    #[test]
    fn mean_diff_refuses_under_material_covariate_shift_even_with_same_mechanism() {
        let n = 80usize;
        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "x",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "y",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        let schema = b.build().unwrap();
        let mut xv = Vec::new();
        let mut yv = Vec::new();
        for i in 0..n {
            // Same y = 1 + 2x + noise; x ranges differ → different marginal y mean, same mechanism.
            let x = if i < 40 { (i % 40) as f64 * 0.1 } else { 5.0 + (i % 40) as f64 * 0.1 };
            xv.push(x);
            yv.push(1.0 + 2.0 * x + 0.01 * ((i % 7) as f64 - 3.0));
        }
        let validity = ValidityBitmap::all_valid(n);
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(0), Arc::from(xv), validity.clone())
                    .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(1), Arc::from(yv), validity).unwrap(),
            ),
        ];
        let data =
            TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());
        let mut g = Dag::with_variables(2);
        g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let model = CompiledCausalModel::compile(g).unwrap();
        let q = MechanismChangeQuery::new(
            [VariableId::from_raw(1)],
            PopulationSelector::TimeRange { start: 0, end: 40 },
            PopulationSelector::TimeRange { start: 40, end: 80 },
            0.05,
            10,
        );
        let result = detect_mechanism_changes(
            &model,
            &data,
            &q,
            MechanismChangeMethod::MeanDiff,
            &ExecutionContext::for_tests(1),
        );
        assert!(
            result.is_err(),
            "a materially shifted parent covariate should be refused, not scored: {result:?}"
        );
    }

    /// Null split of a homogeneous SCM: residual `MeanDiff` Type I smoke.
    /// Not a full calibration gate — only checks flag rate stays near α.
    #[test]
    fn mean_diff_null_split_type_i_smoke() {
        let n = 200usize;
        let mid = n / 2;
        let mut flags = 0usize;
        let trials = 20usize;
        for trial in 0..trials {
            let mut rng = ExecutionContext::for_tests(0x4C01u64.wrapping_add(trial as u64))
                .rng
                .stream_for(StreamDomain::Attribution, 0x7E51);
            let mut b = CausalSchemaBuilder::new();
            b.add_variable(
                "x",
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::Context),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
            b.add_variable(
                "y",
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
            let schema = b.build().unwrap();
            let mut xv = Vec::with_capacity(n);
            let mut yv = Vec::with_capacity(n);
            for _ in 0..n {
                let x = antecedent_kernels::standard_normal(&mut rng);
                let e = antecedent_kernels::standard_normal(&mut rng);
                xv.push(x);
                yv.push(1.0 + 2.0 * x + 0.25 * e);
            }
            let validity = ValidityBitmap::all_valid(n);
            let cols = vec![
                OwnedColumn::Float64(
                    Float64Column::new(VariableId::from_raw(0), Arc::from(xv), validity.clone())
                        .unwrap(),
                ),
                OwnedColumn::Float64(
                    Float64Column::new(VariableId::from_raw(1), Arc::from(yv), validity).unwrap(),
                ),
            ];
            let data =
                TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());
            let mut g = Dag::with_variables(2);
            g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
            let model = CompiledCausalModel::compile(g).unwrap();
            let q = MechanismChangeQuery::new(
                [VariableId::from_raw(1)],
                PopulationSelector::TimeRange { start: 0, end: mid },
                PopulationSelector::TimeRange { start: mid, end: n },
                0.05,
                10,
            );
            let dets = detect_mechanism_changes(
                &model,
                &data,
                &q,
                MechanismChangeMethod::MeanDiff,
                &ExecutionContext::for_tests(1),
            )
            .unwrap();
            let y = dets.iter().find(|d| d.variable == VariableId::from_raw(1)).unwrap();
            if y.changed {
                flags += 1;
            }
        }
        // α=0.05 → expect ~1 flag in 20; allow a few extras for smoke (not a calibration gate).
        assert!(
            flags <= 4,
            "null-split Type I smoke: MeanDiff flagged {flags}/{trials} (want ≤4 at α=0.05)"
        );
    }

    /// Family-wise Type I inflation across several truly-unchanged targets, and control via the
    /// default Benjamini–Hochberg correction (Defect A).
    ///
    /// `mean_diff_null_split_type_i_smoke` above only checks the *per-target* flag rate with a
    /// single target, so it cannot see the family-wise effect: with `m` independent null targets
    /// each judged against α=0.05 independently, `P(>=1 false flag) = 1 - 0.95^m`, which is
    /// ≈40% at m=10. This test builds `m` independent, unrelated mechanisms (no true baseline →
    /// comparison change in any of them) and compares the family-wise flag rate (>=1 of the `m`
    /// targets flagged) with `correction: None` against the corrected default
    /// `detect_mechanism_changes`.
    #[test]
    fn mean_diff_null_split_family_wise_type_i_rate() {
        let m = 10usize;
        let n = 120usize;
        let mid = n / 2;
        let trials = 150usize;
        let alpha = 0.05;

        let mut uncorrected_family_flags = 0usize;
        let mut corrected_family_flags = 0usize;

        for trial in 0..trials {
            let mut rng = ExecutionContext::for_tests(0x9A11u64.wrapping_add(trial as u64))
                .rng
                .stream_for(StreamDomain::Attribution, 0x51EE);
            let mut b = CausalSchemaBuilder::new();
            for k in 0..m {
                b.add_variable(
                    format!("x{k}"),
                    ValueType::Continuous,
                    SmallRoleSet::from_hint(RoleHint::Context),
                    None,
                    None,
                    MeasurementSpec::default(),
                )
                .unwrap();
                b.add_variable(
                    format!("y{k}"),
                    ValueType::Continuous,
                    SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
                    None,
                    None,
                    MeasurementSpec::default(),
                )
                .unwrap();
            }
            let schema = b.build().unwrap();
            let mut cols = Vec::with_capacity(2 * m);
            for k in 0..m {
                let mut xv = Vec::with_capacity(n);
                let mut yv = Vec::with_capacity(n);
                for _ in 0..n {
                    let x = antecedent_kernels::standard_normal(&mut rng);
                    let e = antecedent_kernels::standard_normal(&mut rng);
                    xv.push(x);
                    yv.push(1.0 + 2.0 * x + 0.25 * e);
                }
                let validity = ValidityBitmap::all_valid(n);
                cols.push(OwnedColumn::Float64(
                    Float64Column::new(
                        VariableId::from_raw((2 * k) as u32),
                        Arc::from(xv),
                        validity.clone(),
                    )
                    .unwrap(),
                ));
                cols.push(OwnedColumn::Float64(
                    Float64Column::new(
                        VariableId::from_raw((2 * k + 1) as u32),
                        Arc::from(yv),
                        validity,
                    )
                    .unwrap(),
                ));
            }
            let data =
                TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());
            let mut g = Dag::with_variables((2 * m) as u32);
            for k in 0..m {
                g.insert_directed(
                    DenseNodeId::from_raw((2 * k) as u32),
                    DenseNodeId::from_raw((2 * k + 1) as u32),
                )
                .unwrap();
            }
            let model = CompiledCausalModel::compile(g).unwrap();
            let targets: Vec<VariableId> =
                (0..m).map(|k| VariableId::from_raw((2 * k + 1) as u32)).collect();
            let q = MechanismChangeQuery::new(
                targets,
                PopulationSelector::TimeRange { start: 0, end: mid },
                PopulationSelector::TimeRange { start: mid, end: n },
                alpha,
                m,
            );

            let uncorrected = detect_mechanism_changes_with_correction(
                &model,
                &data,
                &q,
                MechanismChangeMethod::MeanDiff,
                &ExecutionContext::for_tests(1),
                None,
            )
            .unwrap();
            if uncorrected.iter().any(|d| d.changed) {
                uncorrected_family_flags += 1;
            }

            let corrected = detect_mechanism_changes(
                &model,
                &data,
                &q,
                MechanismChangeMethod::MeanDiff,
                &ExecutionContext::for_tests(1),
            )
            .unwrap();
            if corrected.iter().any(|d| d.changed) {
                corrected_family_flags += 1;
            }
        }

        let uncorrected_rate = uncorrected_family_flags as f64 / trials as f64;
        let corrected_rate = corrected_family_flags as f64 / trials as f64;
        // Expected uncorrected family-wise rate ≈ 1 - 0.95^10 ≈ 0.40; allow slop for MC noise.
        assert!(
            uncorrected_rate > 0.20,
            "expected inflated uncorrected family-wise rate (~0.40 expected), got \
             {uncorrected_rate} ({uncorrected_family_flags}/{trials})"
        );
        assert!(
            corrected_rate < uncorrected_rate - 0.10,
            "BH-corrected family-wise rate ({corrected_rate}, {corrected_family_flags}/{trials}) \
             should be well below the uncorrected rate ({uncorrected_rate})"
        );
        assert!(
            corrected_rate <= 0.15,
            "BH-corrected family-wise rate should stay near α=0.05 under the global null, got \
             {corrected_rate} ({corrected_family_flags}/{trials})"
        );
    }

    /// Build one draw of `y = 1 + 2x + N(0,1)` with the *same* mechanism in both
    /// populations, but `x` shifted between them: `x_baseline ~ N(0,1)`,
    /// `x_comparison ~ N(shift,1)`. Baseline rows come first.
    fn covariate_shift_dataset(
        n_b: usize,
        n_c: usize,
        shift: f64,
        rng: &mut CausalRng,
    ) -> (CompiledCausalModel, TabularData) {
        let n = n_b + n_c;
        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "x",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "y",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        let schema = b.build().unwrap();
        let mut xv = Vec::with_capacity(n);
        let mut yv = Vec::with_capacity(n);
        for i in 0..n {
            let mean_x = if i < n_b { 0.0 } else { shift };
            let x = mean_x + antecedent_kernels::standard_normal(rng);
            let e = antecedent_kernels::standard_normal(rng);
            xv.push(x);
            yv.push(1.0 + 2.0 * x + e);
        }
        let validity = ValidityBitmap::all_valid(n);
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(0), Arc::from(xv), validity.clone())
                    .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(1), Arc::from(yv), validity).unwrap(),
            ),
        ];
        let data =
            TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());
        let mut g = Dag::with_variables(2);
        g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let model = CompiledCausalModel::compile(g).unwrap();
        (model, data)
    }

    /// Empirical Type-I (false-positive) rate of a residual two-sample method over
    /// `trials` independent draws of [`covariate_shift_dataset`], at nominal `alpha`.
    /// Trials on which the call is refused (material covariate shift detected) are
    /// excluded from the denominator and counted separately, since a refusal is not a
    /// false positive — it is the honest alternative to one.
    fn covariate_shift_outcome_rates(
        method: MechanismChangeMethod,
        shift: f64,
        trials: usize,
        alpha: f64,
        seed_base: u64,
    ) -> (f64, f64) {
        let n_b = 60usize;
        let n_c = 60usize;
        let mut flags = 0usize;
        let mut refusals = 0usize;
        let mut scored = 0usize;
        for trial in 0..trials {
            let mut gen_rng =
                ExecutionContext::for_tests(seed_base.wrapping_add(trial as u64)).rng.stream(0xC0);
            let (model, data) = covariate_shift_dataset(n_b, n_c, shift, &mut gen_rng);
            let q = MechanismChangeQuery::new(
                [VariableId::from_raw(1)],
                PopulationSelector::TimeRange { start: 0, end: n_b },
                PopulationSelector::TimeRange { start: n_b, end: n_b + n_c },
                alpha,
                10,
            );
            match detect_mechanism_changes_with_correction(
                &model,
                &data,
                &q,
                method,
                &ExecutionContext::for_tests(1),
                None,
            ) {
                Ok(dets) => {
                    scored += 1;
                    let y = dets.iter().find(|d| d.variable == VariableId::from_raw(1)).unwrap();
                    if y.changed {
                        flags += 1;
                    }
                }
                Err(_) => refusals += 1,
            }
        }
        let flag_rate = if scored == 0 { 0.0 } else { flags as f64 / scored as f64 };
        let refusal_rate = refusals as f64 / trials as f64;
        (flag_rate, refusal_rate)
    }

    /// `attr-design-state-2`: residual two-sample tests compared in-sample baseline
    /// residuals against out-of-sample comparison residuals. Under an *unchanged*
    /// mechanism but a shifted parent distribution, the baseline model's own
    /// extrapolation error inflates only the comparison side, so the null is rejected far
    /// above the nominal rate — the reviewer's Python simulation found ≈12% at a 1σ
    /// shift, ≈40% at 3σ, ≈61% at 5σ, against a nominal α=0.05.
    ///
    /// This mirrors that simulation in-crate: `y = 1 + 2x + N(0,1)` with the identical
    /// mechanism in both populations, only `x`'s mean shifted. At `shift=0` (no covariate
    /// difference), the parent-shift guard essentially never fires and the cross-fitted
    /// baseline keeps the flag rate near nominal. At `shift=3` and `shift=5`, no residual
    /// reconstruction can make a same-distribution two-sample test calibrated against the
    /// resulting shared bias (see the comment in
    /// `detect_mechanism_changes_with_correction`), so the call is refused instead of
    /// reporting an anti-conservative p-value: refusals should dominate, not false flags.
    #[test]
    fn mean_diff_covariate_shift_is_calibrated_or_refused() {
        let alpha = 0.05;
        let trials = 300usize;
        let (flag_rate, refusal_rate) = covariate_shift_outcome_rates(
            MechanismChangeMethod::MeanDiff,
            0.0,
            trials,
            alpha,
            0xC0_1B_A5_E0,
        );
        assert!(
            refusal_rate < 0.05,
            "shift=0 should almost never trigger the covariate-shift refusal, got {refusal_rate}"
        );
        assert!(
            flag_rate < 0.15,
            "MeanDiff Type-I rate at shift=0 should stay close to nominal α={alpha}, got {flag_rate}"
        );
        for shift in [3.0, 5.0] {
            let (flag_rate, refusal_rate) = covariate_shift_outcome_rates(
                MechanismChangeMethod::MeanDiff,
                shift,
                trials,
                alpha,
                0xC0_1B_A5_E0,
            );
            assert!(
                refusal_rate > 0.85,
                "MeanDiff at shift={shift} should be refused, not scored, on almost every \
                 trial (a residual two-sample test cannot be calibrated against this \
                 covariate shift): refusal_rate={refusal_rate}, flag_rate={flag_rate}"
            );
        }
    }

    /// Same reproduction as above, for `LikelihoodRatio` (the other test the finding
    /// names explicitly).
    #[test]
    fn likelihood_ratio_covariate_shift_is_calibrated_or_refused() {
        let alpha = 0.05;
        let trials = 300usize;
        let (flag_rate, refusal_rate) = covariate_shift_outcome_rates(
            MechanismChangeMethod::LikelihoodRatio,
            0.0,
            trials,
            alpha,
            0x11_4E_11_D0,
        );
        assert!(
            refusal_rate < 0.05,
            "shift=0 should almost never trigger the covariate-shift refusal, got {refusal_rate}"
        );
        assert!(
            flag_rate < 0.15,
            "LikelihoodRatio Type-I rate at shift=0 should stay close to nominal α={alpha}, \
             got {flag_rate}"
        );
        for shift in [3.0, 5.0] {
            let (flag_rate, refusal_rate) = covariate_shift_outcome_rates(
                MechanismChangeMethod::LikelihoodRatio,
                shift,
                trials,
                alpha,
                0x11_4E_11_D0,
            );
            assert!(
                refusal_rate > 0.85,
                "LikelihoodRatio at shift={shift} should be refused, not scored, on almost \
                 every trial: refusal_rate={refusal_rate}, flag_rate={flag_rate}"
            );
        }
    }

    /// Defect B: `residuals()` must thread a properly seeded RNG into posterior noise recovery
    /// (`infer_noise_column_rng`) rather than resetting to a fixed `CausalRng::from_seed(0)` on
    /// every call (the old `infer_noise_column` shim).
    ///
    /// A `Discrete`-family target's noise-inference mode is always `Posterior` (see
    /// `infer_noise_column_rng` in `antecedent-model/src/mechanism.rs`), so its residual draws
    /// are genuinely a function of the RNG stream, not just of the data. This test forces
    /// `MechanismRegistry::standard()` to fit `y` as `Discrete` (by giving it few distinct
    /// values, since family eligibility is driven by data cardinality via `is_low_cardinality`,
    /// not by the declared `ValueType`) and then runs `detect_mechanism_changes` twice under two
    /// different `ExecutionContext` seeds. Post-fix, the seed is threaded through to the
    /// posterior draws, so the resulting `MeanDiff` statistic differs between the two seeds.
    /// Pre-fix, both calls silently reset to seed 0 regardless of `ExecutionContext`, so the
    /// statistic is bit-identical across seeds — the RNG parameter is a no-op for any
    /// `Posterior`-mode family.
    ///
    /// Per the task brief: this does **not** attempt to show a flipped `changed` verdict (the
    /// within-population draws stay i.i.d. conditioned on whichever seed is used, and the
    /// two-sample tests treat samples as unordered sets, so a verdict flip could not be
    /// constructed). It demonstrates the narrower, honest claim that the execution context's
    /// seed is actually being honored after the fix and was being silently ignored before it.
    #[test]
    fn discrete_target_residuals_depend_on_execution_context_seed() {
        let n = 60usize;
        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "x",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "y",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        let schema = b.build().unwrap();
        // x has 4 distinct categories, y has 2 — both well under the is_low_cardinality(_, 8)
        // threshold that routes the target to MechanismRegistry::standard().discrete =
        // [Discrete, Constant], and the x >= 2 split gives Discrete a real edge over Constant.
        let mut xv = Vec::with_capacity(n);
        let mut yv = Vec::with_capacity(n);
        for i in 0..n {
            let x = (i % 4) as f64;
            xv.push(x);
            yv.push(if x >= 2.0 { 1.0 } else { 0.0 });
        }
        let validity = ValidityBitmap::all_valid(n);
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(0), Arc::from(xv), validity.clone())
                    .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(1), Arc::from(yv), validity).unwrap(),
            ),
        ];
        let data =
            TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());
        let mut g = Dag::with_variables(2);
        g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let model = CompiledCausalModel::compile(g).unwrap();
        let q = MechanismChangeQuery::new(
            [VariableId::from_raw(1)],
            PopulationSelector::TimeRange { start: 0, end: n },
            PopulationSelector::TimeRange { start: 0, end: n },
            0.05,
            10,
        );

        let d1 = detect_mechanism_changes(
            &model,
            &data,
            &q,
            MechanismChangeMethod::MeanDiff,
            &ExecutionContext::for_tests(1),
        )
        .unwrap();
        let d2 = detect_mechanism_changes(
            &model,
            &data,
            &q,
            MechanismChangeMethod::MeanDiff,
            &ExecutionContext::for_tests(2),
        )
        .unwrap();
        let s1 = d1.iter().find(|d| d.variable == VariableId::from_raw(1)).unwrap().statistic;
        let s2 = d2.iter().find(|d| d.variable == VariableId::from_raw(1)).unwrap().statistic;
        // Separated by a real margin, not just `!=`: posterior-sampled noise under two
        // different seeds should move the statistic materially, and an exact-inequality
        // check would also pass on a one-ULP difference.
        assert!(
            (s1 - s2).abs() > 1e-9,
            "Discrete-family residual statistic should depend on the ExecutionContext seed \
             (s1={s1}, s2={s2}); identical values mean the RNG is being ignored"
        );
    }
}
