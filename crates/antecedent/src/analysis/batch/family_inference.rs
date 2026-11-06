//! Shared-design diagnostics and joint family inference.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    Arc, AverageEffectQuery, BatchQuery, CandidateScreen, CellFamilyContrast,
    CrossfitNuisanceCache, ExecutionContext, NuisanceScopeUse, ResponseQuery,
    ScorePosteriorOptions, SharedBatchDesign, StudyResult, TableView, TabularData, VariableId,
    attach_candidate_selection, resolve_max_t_critical,
};

/// Record the shared design and, per result, whether its cross-fitted propensity /
/// outcome fit was shared with another query of the batch (`uses[i]` is result `i`'s
/// nuisance scope).
pub(super) fn attach_shared_design_diagnostics(
    results: &mut [StudyResult],
    shared: Option<&Arc<SharedBatchDesign>>,
    nuisances: &CrossfitNuisanceCache,
    uses: &[NuisanceScopeUse],
    estimator_fingerprint: &str,
) {
    let Some(shared) = shared else {
        return;
    };
    let shares_covariates = shared.covariate.is_some();
    let flag = |b: bool| Arc::<str>::from(if b { "true" } else { "false" });
    for (i, result) in results.iter_mut().enumerate() {
        let (shares_propensity, shares_outcome) =
            uses.get(i).map_or((false, false), |used| nuisances.shared_with_another_scope(used));
        let mut d = antecedent_core::Diagnostic::new(
            "batch.shared_design",
            antecedent_core::DiagnosticKind::Scientific,
            antecedent_core::DiagnosticSeverity::Info,
            format!(
                "shared fold seed/count (n_folds={}); covariates={}; each query draws its own matching fold plan; cross-fitted propensity {} and outcome regressions {} with another query (reused only when rows, treatment coding, adjustment design, fold plan and learner options are identical)",
                shared.n_folds,
                if shares_covariates { "shared" } else { "per-query" },
                if shares_propensity { "shared" } else { "not shared" },
                if shares_outcome { "shared" } else { "not shared" },
            ),
        );
        d.fields = Arc::from([
            (Arc::from("n_folds"), Arc::from(shared.n_folds.to_string())),
            (Arc::from("shares_covariates"), flag(shares_covariates)),
            (Arc::from("shares_propensity"), flag(shares_propensity)),
            (Arc::from("shares_outcome_residualization"), flag(shares_outcome)),
            (Arc::from("estimator_config"), Arc::from(estimator_fingerprint)),
        ]);
        // A prepared plan can carry the diagnostic from its prepare; this estimate's
        // sharing replaces it.
        result.view.diagnostics.retain(|d| d.code.as_ref() != "batch.shared_design");
        result.view.diagnostics.push(d);
    }
}

pub(super) fn attach_prepared_batch_joint_inference(
    results: &mut [StudyResult],
    data: &TabularData,
    queries: &[BatchQuery],
    screen: Option<&CandidateScreen>,
    family_contrast: Option<CellFamilyContrast>,
    posterior: Option<(ScorePosteriorOptions, &ExecutionContext)>,
) {
    attach_batch_family_joint_inference(results, data, queries, screen, family_contrast, posterior);
}

pub(super) fn attach_batch_joint_inference(
    results: &mut [StudyResult],
    data: &TabularData,
    queries: &[AverageEffectQuery],
    screen: Option<&CandidateScreen>,
) {
    let family: Vec<BatchQuery> = queries.iter().cloned().map(BatchQuery::Average).collect();
    attach_batch_family_joint_inference(results, data, &family, screen, None, None);
}

fn batch_claim_alignment(
    query: &BatchQuery,
) -> (antecedent_core::TargetPopulation, Vec<VariableId>) {
    match query {
        BatchQuery::Average(q) => {
            let mut ids = vec![q.treatment, q.outcome];
            ids.extend(q.effect_modifiers.iter().copied());
            (q.target_population.clone(), ids)
        }
        BatchQuery::Response(q) => {
            let mut ids = q.functional.treatment_ids();
            ids.extend(q.functional.outcome_ids());
            (q.target_population.clone(), ids)
        }
    }
}

/// Drop every family-level p-value and contrast from `results` and record why on each.
fn withhold_family_inference(results: &mut [StudyResult], message: &str) {
    for result in results.iter_mut() {
        let Some(effect) = result.estimate.as_effect_mut() else {
            continue;
        };
        effect.family.adjusted_p_values = None;
        effect.family.contrast = None;
        effect.family.contrast_interval = None;
        effect.family.compiled_contrast = None;
        result.view.diagnostics.push(antecedent_core::Diagnostic::new(
            "batch.joint_if",
            antecedent_core::DiagnosticKind::Scientific,
            antecedent_core::DiagnosticSeverity::Info,
            message,
        ));
    }
}

/// Why a family's joint inference was not published.
enum FamilyWithheld {
    /// Fewer than two claims: there is no family to test jointly, and nothing to report.
    SingleClaim,
    /// Joint inference could not be formed at all (misaligned rows, singular covariance).
    Unavailable(String),
    /// Family-level p-values and contrast were withheld for the stated reason; the level
    /// covariance and simultaneous band stay published.
    Inference(String),
}

/// Result of the family joint-inference pipeline.
enum FamilyOutcome {
    /// Family p-values were attached (and candidate selection recorded with them).
    Published,
    /// Family inference was withheld.
    Withheld(FamilyWithheld),
}

/// Every claim's centred influence embedded on the shared row universe, with the
/// complete-case rows each claim was scored on.
struct AlignedLevels {
    columns: Vec<Vec<f64>>,
    row_sets: Vec<Vec<usize>>,
}

fn attach_batch_family_joint_inference(
    results: &mut [StudyResult],
    data: &TabularData,
    queries: &[BatchQuery],
    screen: Option<&CandidateScreen>,
    family_contrast: Option<CellFamilyContrast>,
    posterior: Option<(ScorePosteriorOptions, &ExecutionContext)>,
) {
    let outcome =
        family_joint_inference(results, data, queries, screen, family_contrast, posterior);
    // One epilogue for every withheld path: record why, then record the selection without
    // any ranking statistics.
    let FamilyOutcome::Withheld(reason) = outcome else {
        return;
    };
    match reason {
        FamilyWithheld::SingleClaim => {}
        FamilyWithheld::Unavailable(message) => {
            for result in results.iter_mut() {
                result.view.diagnostics.push(antecedent_core::Diagnostic::new(
                    "batch.joint_if.unavailable",
                    antecedent_core::DiagnosticKind::Scientific,
                    antecedent_core::DiagnosticSeverity::Warning,
                    message.as_str(),
                ));
            }
        }
        FamilyWithheld::Inference(message) => withhold_family_inference(results, &message),
    }
    attach_candidate_selection(results, screen, &[], &[], None);
}

fn family_joint_inference(
    results: &mut [StudyResult],
    data: &TabularData,
    queries: &[BatchQuery],
    screen: Option<&CandidateScreen>,
    family_contrast: Option<CellFamilyContrast>,
    posterior: Option<(ScorePosteriorOptions, &ExecutionContext)>,
) -> FamilyOutcome {
    if results.len() < 2 {
        return FamilyOutcome::Withheld(FamilyWithheld::SingleClaim);
    }
    let aligned = match align_levels(results, data, queries) {
        Ok(aligned) => aligned,
        Err(reason) => return FamilyOutcome::Withheld(FamilyWithheld::Unavailable(reason.into())),
    };
    let cols: Vec<&[f64]> = aligned.columns.iter().map(Vec::as_slice).collect();
    let Ok(level_cov) = antecedent_estimate::joint_influence_covariance(&cols, None) else {
        return FamilyOutcome::Withheld(FamilyWithheld::Unavailable(
            "batch joint IF covariance could not be formed".into(),
        ));
    };
    let crit = attach_level_intervals(results, &level_cov);

    if queries.iter().any(|q| matches!(q, BatchQuery::Response(_))) {
        return attach_contrast_family(
            results,
            data,
            queries,
            &aligned.row_sets,
            family_contrast,
            posterior,
            screen,
        );
    }
    let values: Vec<f64> = results
        .iter()
        .filter_map(|result| result.estimate.as_effect().map(|effect| effect.ate))
        .collect();
    attach_family_p_values(results, &values, &level_cov, crit, None, screen);
    FamilyOutcome::Published
}

/// Align every claim's influence to the shared row universe.
fn align_levels(
    results: &[StudyResult],
    data: &TabularData,
    queries: &[BatchQuery],
) -> Result<AlignedLevels, &'static str> {
    let mut columns = Vec::with_capacity(results.len());
    let mut row_sets = Vec::with_capacity(results.len());
    for (result, query) in results.iter().zip(queries) {
        let (col, rows) = align_influence_to_rows(
            data,
            result,
            query,
            result.estimate.as_effect().and_then(|effect| effect.scores.influence.as_deref()),
        )?;
        columns.push(col);
        row_sets.push(rows);
    }
    Ok(AlignedLevels { columns, row_sets })
}

/// Record the level covariance on every result and, when the max-t critical value can be
/// formed, the simultaneous interval on cell levels. Returns the critical value.
fn attach_level_intervals(
    results: &mut [StudyResult],
    level_cov: &antecedent_estimate::JointCovariance,
) -> Option<f64> {
    let crit = resolve_max_t_critical(results, level_cov, "simultaneous interval");
    for (i, result) in results.iter_mut().enumerate() {
        let Some(effect) = result.estimate.as_effect_mut() else {
            continue;
        };
        effect.joint.covariance = Some(level_cov.clone());
        if let Some(c) = crit {
            effect.family.simultaneous_interval =
                Some((effect.ate - c * level_cov.se(i), effect.ate + c * level_cov.se(i), 0.95));
        }
    }
    crit
}

/// Family p-values, interval and posterior on the declared contrast of a joint-cell family.
fn attach_contrast_family(
    results: &mut [StudyResult],
    data: &TabularData,
    queries: &[BatchQuery],
    row_sets: &[Vec<usize>],
    family_contrast: Option<CellFamilyContrast>,
    posterior: Option<(ScorePosteriorOptions, &ExecutionContext)>,
    screen: Option<&CandidateScreen>,
) -> FamilyOutcome {
    let Some(kind) = family_contrast else {
        return FamilyOutcome::Withheld(FamilyWithheld::Inference(
            "shared-row joint IF on cell levels; no family contrast declared, p-values omitted"
                .into(),
        ));
    };
    let full_n = data.row_count();
    let mut contrast_aligned = Vec::with_capacity(results.len());
    let mut contrast_values = Vec::with_capacity(results.len());
    let mut contrast_scores = Vec::with_capacity(results.len());
    let mut contrast_bindings = Vec::with_capacity(results.len());
    for ((result, query), rows) in results.iter().zip(queries).zip(row_sets) {
        let CellFamilyClaim { value, scores, binding } = match cell_family_contrast_claim(
            data, result, query, kind,
        ) {
            Ok(claim) => claim,
            Err(reason) => {
                return FamilyOutcome::Withheld(FamilyWithheld::Inference(format!(
                    "shared-row joint IF on cell levels; {kind} contrast withheld: {reason}; p-values omitted"
                )));
            }
        };
        contrast_bindings.push(binding);
        if scores.len() != rows.len() {
            return FamilyOutcome::Withheld(FamilyWithheld::Unavailable(
                "batch family contrast influence length did not match the complete-case row universe"
                    .into(),
            ));
        }
        contrast_aligned.push(embed_centered_influence(&scores, rows, full_n));
        contrast_scores.push(scores);
        contrast_values.push(value);
    }
    let contrast_cols: Vec<&[f64]> = contrast_aligned.iter().map(Vec::as_slice).collect();
    let Ok(contrast_cov) = antecedent_estimate::joint_influence_covariance(&contrast_cols, None)
    else {
        return FamilyOutcome::Withheld(FamilyWithheld::Inference(
            "shared-row joint IF on cell levels; family contrast covariance could not be formed, p-values omitted"
                .into(),
        ));
    };
    let contrast_crit = resolve_max_t_critical(results, &contrast_cov, "family-contrast interval");
    for (result, binding) in results.iter_mut().zip(contrast_bindings) {
        if let Some(effect) = result.estimate.as_effect_mut() {
            effect.family.compiled_contrast = Some(binding);
        }
    }
    if let Some((options, ctx)) = posterior {
        attach_family_contrast_posterior(results, &contrast_scores, options, ctx);
    }
    attach_family_p_values(
        results,
        &contrast_values,
        &contrast_cov,
        contrast_crit,
        Some(kind),
        screen,
    );
    FamilyOutcome::Published
}

fn align_influence_to_rows(
    data: &TabularData,
    result: &StudyResult,
    query: &BatchQuery,
    inf: Option<&[f64]>,
) -> Result<(Vec<f64>, Vec<usize>), &'static str> {
    let (target_population, mut ids) = batch_claim_alignment(query);
    if !matches!(target_population, antecedent_core::TargetPopulation::AllObserved) {
        return Err("batch joint IF requires AllObserved");
    }
    let Some(inf) = inf else {
        return Err("batch joint IF requires per-claim influence");
    };
    ids.extend(result.estimand.adjustment_set.iter().copied());
    ids.extend(result.estimand.instruments.iter().copied());
    ids.extend(result.estimand.mediators.iter().copied());
    if let Some(table) = result.estimate.as_effect().and_then(|effect| effect.scores.table.as_ref())
    {
        ids.push(table.treatment);
        ids.extend(table.intervened.iter().copied());
    }
    let Ok(mask) = data.complete_case_mask(&ids) else {
        return Err("batch joint IF complete-case masks did not align");
    };
    let rows: Vec<_> = mask.iter().enumerate().filter_map(|(i, &keep)| keep.then_some(i)).collect();
    if rows.len() != inf.len() || rows.len() < 2 {
        return Err("batch joint IF influence length did not match the complete-case row universe");
    }
    Ok((embed_centered_influence(inf, &rows, data.row_count()), rows))
}

/// Place a claim's influence values on the full row universe so claims with different
/// complete cases share one covariance. The joint covariance divides by `N(N-1)` where the
/// claim's own standard error divides by `n(n-1)`, so the centred values are scaled by
/// `sqrt(N(N-1)/(n(n-1)))`: the embedded column then has exactly the claim's variance.
pub(super) fn embed_centered_influence(inf: &[f64], rows: &[usize], full_n: usize) -> Vec<f64> {
    let n = inf.len() as f64;
    let big_n = full_n as f64;
    let scale = (big_n * (big_n - 1.0) / (n * (n - 1.0))).sqrt();
    let mean = inf.iter().sum::<f64>() / n;
    let mut col = vec![0.0; full_n];
    for (&r, &v) in rows.iter().zip(inf) {
        col[r] = (v - mean) * scale;
    }
    col
}

fn response_requested_arm(query: &ResponseQuery) -> Option<u32> {
    super::super::helpers::requested_joint_arm(query).ok()
}

struct CellFamilyClaim {
    value: f64,
    scores: Vec<f64>,
    binding: Arc<antecedent_estimate::effect_estimate::CompiledFamilyContrast>,
}

fn cell_family_contrast_claim(
    data: &TabularData,
    result: &StudyResult,
    query: &BatchQuery,
    kind: CellFamilyContrast,
) -> Result<CellFamilyClaim, String> {
    let BatchQuery::Response(query) = query else {
        return Err("a cell-family contrast requires a response query".into());
    };
    let table = result
        .estimate
        .as_effect()
        .and_then(|effect| effect.scores.table.as_ref())
        .ok_or_else(|| "family contrast requires frozen joint-cell scores".to_owned())?;
    let binding = super::super::joint_contrast_binding::bind(data, result, query, table, kind)?;
    let arm =
        response_requested_arm(query).ok_or_else(|| "invalid joint intervention arm".to_owned())?;
    let (contrast, scores) = antecedent_estimate::family_cell_contrast(table, arm, kind.as_str())
        .map_err(|error| error.to_string())?;
    Ok(CellFamilyClaim { value: contrast.value, scores, binding })
}

/// Two-sided p-value for a Wald statistic `value / se` against the standard normal.
///
/// The reference distribution is the normal one: `se` comes from the shared-row influence
/// function covariance, an asymptotic standard error with no estimated degrees of freedom,
/// and the family's max-t band is drawn from the same normal limit, so p-values and
/// simultaneous intervals agree. The tail is `2 * norm_sf(z)`, which stays accurate until the
/// true value underflows (near `z = 38`).
///
/// A non-finite or non-positive SE describes a degenerate or broken estimate
/// (a constant influence column, a negative-variance covariance entry), not
/// infinite precision. Returning `NaN` in that case — rather than folding it
/// into `z = +inf`, `p = 0` — keeps the test out of BH/BY ranking instead of
/// handing it the family's strongest "discovery".
pub(super) fn two_sided_p(value: f64, se: f64) -> f64 {
    if !se.is_finite() || se <= 0.0 {
        return f64::NAN;
    }
    let z = value.abs() / se;
    2.0 * antecedent_kernels::norm_sf(z)
}

/// Publish the Bayesian-bootstrap posterior of each member's contrast score mean.
///
/// All members must be scored on the same complete-case rows so that one Dirichlet weight
/// vector per draw is shared by the whole family; otherwise the posterior is withheld, never
/// formed from independent draws.
fn attach_family_contrast_posterior(
    results: &mut [StudyResult],
    scores: &[Vec<f64>],
    options: ScorePosteriorOptions,
    ctx: &ExecutionContext,
) {
    let columns: Vec<&[f64]> = scores.iter().map(Vec::as_slice).collect();
    let outcome = if columns.windows(2).any(|pair| pair[0].len() != pair[1].len()) {
        Err("family members are scored on different complete-case rows".to_owned())
    } else {
        antecedent_estimate::score_posterior::bayesian_bootstrap_score_posteriors(
            &columns, options, ctx,
        )
        .map_err(|error| error.to_string())
    };
    match outcome {
        Ok(posteriors) => {
            for (result, posterior) in results.iter_mut().zip(posteriors) {
                if let Some(effect) = result.estimate.as_effect_mut() {
                    effect.family.contrast_posterior = Some(posterior);
                }
            }
        }
        Err(reason) => {
            for result in results.iter_mut() {
                result.view.diagnostics.push(antecedent_core::Diagnostic::new(
                    "batch.family_posterior",
                    antecedent_core::DiagnosticKind::Scientific,
                    antecedent_core::DiagnosticSeverity::Info,
                    format!("family contrast posterior withheld: {reason}"),
                ));
            }
        }
    }
}

fn attach_family_p_values(
    results: &mut [StudyResult],
    values: &[f64],
    cov: &antecedent_estimate::JointCovariance,
    crit: Option<f64>,
    contrast: Option<CellFamilyContrast>,
    screen: Option<&CandidateScreen>,
) {
    let mut p_values = Vec::with_capacity(results.len());
    let mut rank_stats = Vec::with_capacity(results.len());
    for (i, value) in values.iter().copied().enumerate() {
        let se = cov.se(i);
        p_values.push(two_sided_p(value, se));
        rank_stats.push((value, se));
    }
    let bh = antecedent_stats::benjamini_hochberg(&p_values);
    let by = antecedent_stats::benjamini_yekutieli(&p_values);
    for (i, result) in results.iter_mut().enumerate() {
        let Some(effect) = result.estimate.as_effect_mut() else {
            continue;
        };
        effect.family.adjusted_p_values = Some((bh[i], by[i]));
        if contrast.is_some() {
            let se = cov.se(i);
            effect.family.contrast = Some((values[i], se));
            effect.family.contrast_interval =
                crit.map(|c| (values[i] - c * se, values[i] + c * se, 0.95));
        }
        let mut d = antecedent_core::Diagnostic::new(
            "batch.joint_if",
            antecedent_core::DiagnosticKind::Scientific,
            antecedent_core::DiagnosticSeverity::Info,
            format!(
                "shared-row joint IF;{} BH={:.4} BY={:.4}{}",
                contrast
                    .map(|kind| format!(
                        " family contrast {}={:.4} (se={:.4});",
                        kind.as_str(),
                        values[i],
                        cov.se(i)
                    ))
                    .unwrap_or_default(),
                bh.get(i).copied().unwrap_or(f64::NAN),
                by.get(i).copied().unwrap_or(f64::NAN),
                crit.map(|c| format!(" max-t_0.95={c:.3}")).unwrap_or_default()
            ),
        );
        let mut fields = vec![
            (
                std::sync::Arc::from("bh_q"),
                std::sync::Arc::from(format!("{}", bh.get(i).copied().unwrap_or(f64::NAN))),
            ),
            (
                std::sync::Arc::from("by_q"),
                std::sync::Arc::from(format!("{}", by.get(i).copied().unwrap_or(f64::NAN))),
            ),
        ];
        if let Some(kind) = contrast {
            fields.push((
                std::sync::Arc::from("family_contrast"),
                std::sync::Arc::from(kind.as_str()),
            ));
            fields.push((
                std::sync::Arc::from("contrast_value"),
                std::sync::Arc::from(format!("{}", values[i])),
            ));
        }
        d.fields = std::sync::Arc::from(fields);
        result.view.diagnostics.push(d);
    }
    attach_candidate_selection(results, screen, &bh, &by, Some(&rank_stats));
}
