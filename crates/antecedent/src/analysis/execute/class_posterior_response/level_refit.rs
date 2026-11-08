//! Response-level refits and scoped class-completion sensitivity.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    Arc, AtomRefutationPolicy, AverageEffectQuery, CachedGraphPosteriorIdentification, CausalError,
    CausalResponse, ClassAtomResponseEval, ContinuousResponseEstimator, Diagnostic, DiagnosticKind,
    DiagnosticSeverity, EffectEstimate, EstimationWorkspace, ExecutionContext,
    IdentificationStatus, IdentifiedEstimand, IdentifiedKeyWeights, InferenceMode, OverlapPolicy,
    RefuteSuite, ResponseFunctional, ResponseQuery, ResponseValue, StructuralAggregationPolicy,
    Study, TableView, TabularData, VariableId, bayesian_gcomp, response_identified_value,
};

pub(super) fn atom_plugin_scalar(response: &CausalResponse) -> Option<f64> {
    match response_identified_value(response)? {
        ResponseValue::Scalar(value) if value.is_finite() => Some(value),
        ResponseValue::Surface { mean, .. } if mean.len() == 1 => {
            mean.first().copied().filter(|value| value.is_finite())
        }
        _ => None,
    }
}

/// Refit the same intervention *level* and response procedure on perturbed rows.
#[derive(Clone, Copy)]
pub(in crate::analysis::execute) struct StaticResponseLevelRefit<'a> {
    pub(in crate::analysis::execute) study: &'a Study,
    pub(in crate::analysis::execute) query: &'a ResponseQuery,
    pub(in crate::analysis::execute) estimand: &'a IdentifiedEstimand,
    pub(in crate::analysis::execute) status: IdentificationStatus,
    pub(in crate::analysis::execute) assumptions: &'a antecedent_core::AssumptionSet,
    pub(in crate::analysis::execute) operational:
        Option<&'a std::cell::RefCell<Option<antecedent_validate::ValidationError>>>,
}

impl std::fmt::Debug for StaticResponseLevelRefit<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StaticResponseLevelRefit").finish_non_exhaustive()
    }
}

/// Convert an analysis mask into column validity for the response estimator,
/// which reads column validity but does not consume the table analysis mask.
/// The admission covers copied float buffers, validity maps and storage metadata.
pub(super) fn level_refit_masked_data(
    data: &TabularData,
    ctx: &ExecutionContext,
) -> Result<
    (Option<TabularData>, Option<antecedent_core::execution::AdmissionLease>),
    antecedent_validate::ValidationError,
> {
    use antecedent_data::{Float64Column, OwnedColumn, OwnedColumnarStorage, ValidityBitmap};
    use antecedent_validate::ValidationError;
    let Some(mask) = data.storage().analysis_mask() else {
        return Ok((None, None));
    };
    if ctx.cancellation.is_cancelled() {
        return Err(ValidationError::Cancelled);
    }
    let rows = u64::try_from(data.row_count()).map_err(|_| ValidationError::Resource {
        message: "level-refit row count overflow".into(),
    })?;
    let cols = u64::try_from(data.storage().columns().len()).map_err(|_| {
        ValidationError::Resource { message: "level-refit column count overflow".into() }
    })?;
    let bytes = rows
        .checked_mul(cols)
        .and_then(|v| v.checked_mul(16))
        .and_then(|v| rows.checked_mul(8).and_then(|r| v.checked_add(r)))
        .and_then(|v| cols.checked_mul(512).and_then(|c| v.checked_add(c)))
        .and_then(|v| v.checked_add(4096))
        .ok_or_else(|| ValidationError::Resource {
            message: "level-refit mask admission overflow".into(),
        })?;
    let lease = ctx.acquire_admission(0, bytes).map_err(|error| match error {
        antecedent_core::execution::AdmissionError::Cancelled => ValidationError::Cancelled,
        other => ValidationError::Resource { message: other.to_string() },
    })?;
    let mut columns = Vec::with_capacity(data.storage().columns().len());
    for column in data.storage().columns() {
        if ctx.cancellation.is_cancelled() {
            return Err(ValidationError::Cancelled);
        }
        let original = column.as_view().validity();
        let mut keep = Vec::with_capacity(data.row_count());
        for row in 0..data.row_count() {
            if row % 256 == 0 && ctx.cancellation.is_cancelled() {
                return Err(ValidationError::Cancelled);
            }
            keep.push(mask.is_valid(row) && original.is_valid(row));
        }
        let validity = ValidityBitmap::from_bools(&keep);
        let selected = match column.clone() {
            OwnedColumn::Float64(column) => {
                OwnedColumn::Float64(Float64Column::new(column.id, column.values, validity)?)
            }
            OwnedColumn::Int64(mut column) => {
                column.validity = validity;
                OwnedColumn::Int64(column)
            }
            OwnedColumn::Boolean(mut column) => {
                column.validity = validity;
                OwnedColumn::Boolean(column)
            }
            // The response estimator cannot coerce these types; keep their
            // payloads and metadata intact rather than rewriting them.
            OwnedColumn::Categorical(column) => OwnedColumn::Categorical(column),
            OwnedColumn::Timestamp(column) => OwnedColumn::Timestamp(column),
            OwnedColumn::FixedVector(column) => OwnedColumn::FixedVector(column),
        };
        columns.push(selected);
    }
    let storage = OwnedColumnarStorage::try_new(
        data.storage().schema().clone(),
        columns,
        None,
        data.storage().weights().map(Arc::from),
    )?;
    Ok((Some(TabularData::new(storage)), Some(lease)))
}

impl StaticResponseLevelRefit<'_> {
    fn refit_inner(
        &self,
        data: &TabularData,
        extras: &[VariableId],
        ctx: &ExecutionContext,
    ) -> Result<EffectEstimate, antecedent_validate::ValidationError> {
        if ctx.cancellation.is_cancelled() {
            return Err(antecedent_validate::ValidationError::Cancelled);
        }
        if !extras.is_empty() {
            return Err(antecedent_validate::ValidationError::NotApplicable {
                message: "additional nuisance variables are not supported for this intervention-level refit",
            });
        }
        let (selected, _selection_lease) = level_refit_masked_data(data, ctx)?;
        let data = selected.as_ref().unwrap_or(data);
        let mut estimator =
            ContinuousResponseEstimator::new(Arc::clone(&self.estimand.adjustment_set));
        estimator.options = self.study.response_options.clone().unwrap_or_default();
        let response = match &self.study.settings.inference {
            InferenceMode::Frequentist => estimator.estimate_identified(
                data,
                self.query,
                self.status,
                self.assumptions.clone(),
            ),
            InferenceMode::Bayesian(config) => {
                let mut bayes = bayesian_gcomp(config, ctx);
                bayes.prior.clone_from(&config.prior);
                estimator.estimate_bayesian(
                    data,
                    self.query,
                    self.status,
                    self.assumptions.clone(),
                    &bayes,
                    ctx,
                )
            }
        }
        .map_err(antecedent_validate::ValidationError::from)?;
        let value = atom_plugin_scalar(&response).ok_or_else(|| {
            antecedent_validate::ValidationError::estimation_msg(
                "intervention-level refit did not produce a finite scalar",
            )
        })?;
        Ok(EffectEstimate::new(
            value,
            f64::NAN,
            self.assumptions.clone(),
            OverlapPolicy::ExplicitOverride,
        ))
    }
}

impl antecedent_validate::common::EffectRefit for StaticResponseLevelRefit<'_> {
    fn static_bootstrap_uses_witness_columns(&self) -> bool {
        true
    }

    fn refit(
        &self,
        data: &TabularData,
        extras: &[VariableId],
        ctx: &ExecutionContext,
    ) -> Result<EffectEstimate, antecedent_validate::ValidationError> {
        let result = self.refit_inner(data, extras, ctx);
        if let (Err(error), Some(latch)) = (&result, self.operational) {
            let classified = CausalError::from(error.clone());
            if matches!(classified, CausalError::Cancelled { .. } | CausalError::Resource { .. }) {
                *latch.borrow_mut() = Some(error.clone());
            }
        }
        result
    }
}

/// Drop each adjustment column while refitting the same level on one fixed
/// complete-case population. The generic `GraphRefuter` cannot do this through
/// `EffectRefit`, whose trait does not receive the reduced estimand.
#[allow(
    clippy::too_many_lines,
    reason = "the adjustment refits share one fixed complete-case population"
)]
pub(in crate::analysis::execute) fn level_adjustment_sensitivity(
    refitter: &StaticResponseLevelRefit<'_>,
    data: &TabularData,
    original: &EffectEstimate,
    treatment: VariableId,
    outcome: VariableId,
    ctx: &ExecutionContext,
) -> Result<antecedent_validate::RefutationReport, antecedent_validate::ValidationError> {
    use antecedent_validate::common::EffectRefit;
    use antecedent_validate::{RefutationReport, ValidationError};
    let adjustments = &refitter.estimand.adjustment_set;
    if !refitter.estimand.is_adjustment_shaped() {
        return Err(ValidationError::NotApplicable {
            message: "adjustment drop requires an adjustment-shaped estimand",
        });
    }
    if adjustments.is_empty() {
        return Ok(RefutationReport::new(
            "adjustment.drop_covariate",
            original.ate,
            original.ate,
            0.0,
            false,
            true,
            None,
            0,
        ));
    }
    if ctx.cancellation.is_cancelled() {
        return Err(ValidationError::Cancelled);
    }
    let n = u64::try_from(data.row_count()).map_err(|_| ValidationError::Resource {
        message: "level adjustment row count overflow".into(),
    })?;
    let columns = u64::try_from(data.storage().columns().len()).map_err(|_| {
        ValidationError::Resource { message: "level adjustment column count overflow".into() }
    })?;
    let adjusted = u64::try_from(adjustments.len()).map_err(|_| ValidationError::Resource {
        message: "level adjustment count overflow".into(),
    })?;
    let bytes = n
        .checked_mul(40)
        .and_then(|v| columns.checked_mul(512).and_then(|c| v.checked_add(c)))
        .and_then(|v| adjusted.checked_mul(64).and_then(|a| v.checked_add(a)))
        .and_then(|v| v.checked_add(4096))
        .ok_or_else(|| ValidationError::Resource {
            message: "level adjustment scratch admission overflow".into(),
        })?;
    let _scratch = ctx.acquire_admission(0, bytes).map_err(|error| match error {
        antecedent_core::execution::AdmissionError::Cancelled => ValidationError::Cancelled,
        other => ValidationError::Resource { message: other.to_string() },
    })?;
    let mut ids = vec![treatment, outcome];
    for &id in adjustments.iter() {
        if !ids.contains(&id) {
            ids.push(id);
        }
    }
    let mut mask = vec![true; data.row_count()];
    for id in ids {
        let column = data.column(id)?;
        for (row, keep) in mask.iter_mut().enumerate() {
            if ctx.cancellation.is_cancelled() {
                return Err(ValidationError::Cancelled);
            }
            let finite = match column {
                antecedent_data::ColumnView::Float64(values) => values.values[row].is_finite(),
                antecedent_data::ColumnView::Int64(_) | antecedent_data::ColumnView::Boolean(_) => {
                    true
                }
                _ => {
                    return Err(ValidationError::NotApplicable {
                        message: "adjustment drop requires numeric fitting columns",
                    });
                }
            };
            *keep &= column.validity().is_valid(row) && finite;
        }
    }
    if let Some(existing) = data.storage().analysis_mask() {
        for (row, keep) in mask.iter_mut().enumerate() {
            if row % 256 == 0 && ctx.cancellation.is_cancelled() {
                return Err(ValidationError::Cancelled);
            }
            *keep &= existing.is_valid(row);
        }
    }
    let numeric = |id| -> Result<Vec<f64>, ValidationError> {
        let column = data.column(id)?;
        let mut values = Vec::with_capacity(data.row_count());
        for (row, &keep) in mask.iter().enumerate() {
            if row % 256 == 0 && ctx.cancellation.is_cancelled() {
                return Err(ValidationError::Cancelled);
            }
            if !keep {
                continue;
            }
            let value = match column {
                antecedent_data::ColumnView::Float64(c) => c.values[row],
                antecedent_data::ColumnView::Int64(c) => c.values[row] as f64,
                antecedent_data::ColumnView::Boolean(c) => f64::from(c.values[row]),
                _ => {
                    return Err(ValidationError::NotApplicable {
                        message: "adjustment drop requires numeric treatment and outcome",
                    });
                }
            };
            values.push(value);
        }
        Ok(values)
    };
    let sd_t = antecedent_stats::sample_std(&numeric(treatment)?);
    let sd_y = antecedent_stats::sample_std(&numeric(outcome)?);
    if !(sd_t.is_finite() && sd_t > 0.0 && sd_y.is_finite() && sd_y > 0.0) {
        return Err(ValidationError::NotApplicable {
            message: "adjustment drop requires non-degenerate treatment and outcome",
        });
    }
    let fixed = data.with_analysis_mask(antecedent_data::ValidityBitmap::from_bools(&mask))?;
    let centre = refitter.refit(&fixed, &[], ctx)?.ate;
    let floor = 1e-3 * (sd_y / sd_t);
    let mut worst = 0.0_f64;
    let mut worst_value = centre;
    let mut worst_dropped = adjustments[0];
    for (index, &dropped) in adjustments.iter().enumerate() {
        if ctx.cancellation.is_cancelled() {
            return Err(ValidationError::Cancelled);
        }
        let remaining: Vec<_> = adjustments
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| (i != index).then_some(v))
            .collect();
        let reduced = IdentifiedEstimand::new(
            Arc::clone(&refitter.estimand.method),
            remaining.into(),
            Arc::clone(&refitter.estimand.instruments),
            Arc::clone(&refitter.estimand.mediators),
            refitter.estimand.functional,
            refitter.estimand.rd_design,
        );
        let altered = StaticResponseLevelRefit { estimand: &reduced, ..*refitter };
        let value = altered.refit(&fixed, &[], ctx)?.ate;
        let delta = (value - centre).abs() / centre.abs().max(floor);
        if delta >= worst {
            worst = delta;
            worst_value = value;
            worst_dropped = dropped;
        }
    }
    let passed = worst < 0.5;
    Ok(RefutationReport::new("adjustment.drop_covariate", original.ate, worst_value, worst,
        false, passed, (!passed).then(|| Arc::from(format!("relative intervention-level change {worst} exceeded 0.5 after dropping {worst_dropped:?}"))),
        u32::try_from(adjustments.len()).unwrap_or(u32::MAX)))
}

#[allow(
    clippy::too_many_lines,
    reason = "each completion is checked and scoped in one bounded traversal"
)]
#[allow(
    clippy::too_many_arguments,
    reason = "the refuter binds the compiled route, data and completion state"
)]
pub(super) fn plugin_level_on_class_response_atoms(
    study: &Study,
    data: &TabularData,
    identified: &CachedGraphPosteriorIdentification,
    evals: &[ClassAtomResponseEval],
    (treatment, outcome): (VariableId, VariableId),
    query: &ResponseQuery,
    source: Option<[u8; 32]>,
    ctx: &ExecutionContext,
    refutation: AtomRefutationPolicy<'_>,
) -> Result<(Vec<antecedent_validate::RefutationReport>, Vec<Diagnostic>), CausalError> {
    let AtomRefutationPolicy { aggregation: policy, estimator } = refutation;
    let suite = study.settings.refute;
    let custom = &study.declared.custom_validators[..];
    let source = source.ok_or_else(|| CausalError::Compile {
        message: "static completion refutation requires its lowered program identity".into(),
    })?;
    let mut ate_query = AverageEffectQuery::binary_ate(treatment, outcome);
    if let ResponseFunctional::InterventionResponse { interventions, .. } = &query.functional
        && let [action] = interventions.as_ref()
    {
        ate_query.active = action.clone();
    }
    ate_query.target_population = query.target_population.clone();
    let mut refute_ws = EstimationWorkspace::default();
    let mut scoped_reports = Vec::new();
    let mut diagnostics = Vec::new();
    let key_weights = IdentifiedKeyWeights::new(&identified.graphs);
    for eval in evals {
        let atom_weight = key_weights.get(eval.key);
        if atom_weight <= 0.0 {
            continue;
        }
        if matches!(policy, StructuralAggregationPolicy::SameEstimandWeightedMean)
            && eval.response.as_ref().and_then(response_identified_value).is_none()
        {
            continue;
        }
        let plugins: Vec<_> =
            eval.plugin_levels.iter().filter(|(weight, _, _, _, _, _)| *weight > 0.0).collect();
        let available: Vec<f64> =
            eval.plugin_levels.iter().map(|(weight, _, _, _, _, _)| *weight).collect();
        // Incomplete coverage cannot certify the whole mixture; the available
        // completion reports remain useful, explicitly scoped evidence.
        if matches!(policy, StructuralAggregationPolicy::SameEstimandWeightedMean)
            && super::super::atom_mix::nested_complete_weights(
                atom_weight,
                &eval.point_weights,
                &available,
            )
            .is_none()
        {
            diagnostics.push(Diagnostic::new(
                "refute.class_response.incomplete_completions", DiagnosticKind::Scientific,
                DiagnosticSeverity::Warning,
                "completion reports cover only available refuter inputs and do not certify the aggregate",
            ));
        }
        let inner_scale = eval.point_weights.iter().copied().fold(0.0_f64, f64::max);
        let inner_total = eval.point_weights.iter().map(|w| w / inner_scale).sum::<f64>();
        for (completion_weight, completion_digest, status, estimand, scalar, assumptions) in plugins
        {
            let estimate = EffectEstimate::new(
                *scalar,
                f64::NAN,
                assumptions.clone(),
                OverlapPolicy::ExplicitOverride,
            );
            let operational = std::cell::RefCell::new(None);
            let refitter = StaticResponseLevelRefit {
                study,
                query,
                estimand,
                status: *status,
                assumptions,
                operational: Some(&operational),
            };
            let problem = antecedent_validate::RefutationProblem::new(
                data,
                estimand,
                &ate_query,
                &estimate,
                Some(estimator),
                None,
            )
            .with_effect_refit(&refitter);
            let mut validation = match suite {
                RefuteSuite::None => antecedent_validate::ValidationSuite::new(),
                RefuteSuite::Cheap | RefuteSuite::PlaceboAndRcc => {
                    antecedent_validate::ValidationSuite::overlap_only()
                }
                RefuteSuite::Full => antecedent_validate::ValidationSuite::new()
                    .with(antecedent_validate::ValidatorId::Overlap)
                    .with(antecedent_validate::ValidatorId::OverlapRule)
                    .with(antecedent_validate::ValidatorId::Bootstrap)
                    .with(antecedent_validate::ValidatorId::DataSubset),
            };
            for validator in custom {
                validation = validation.with_custom(Arc::clone(validator));
            }
            let outcome_result = validation.run(&problem, &mut refute_ws, ctx);
            if let Some(error) = operational.borrow_mut().take() {
                return Err(CausalError::from(error));
            }
            let outcomes = outcome_result.map_err(CausalError::from)?;
            diagnostics.extend(super::super::validator_not_applicable_diagnostics(&outcomes));
            let label = format!(
                "scope:v1:static_class_completion:{}:{}|level",
                eval.key,
                antecedent_core::identity::to_hex(completion_digest)
            );
            let target = antecedent_validate::RefutationTarget::new(
                antecedent_validate::TargetFamily::WorldValued,
                antecedent_validate::TargetQuadrant {
                    temporal: false,
                    bayesian: study.settings.inference.is_bayesian(),
                },
                [label],
                source,
            )
            .map_err(CausalError::from)?;
            let scope = antecedent_validate::RefutationScope::StaticClassCompletion {
                outer_atom: eval.key,
                completion_digest: *completion_digest,
                target,
            };
            let mut reports = antecedent_validate::ValidationSuite::reports_only(&outcomes);
            if matches!(suite, RefuteSuite::Full) {
                match level_adjustment_sensitivity(
                    &refitter, data, &estimate, treatment, outcome, ctx,
                ) {
                    Ok(report) => reports.push(report),
                    Err(antecedent_validate::ValidationError::NotApplicable { message }) => {
                        let mut diagnostic = Diagnostic::new(
                            "refute.validator.not_applicable",
                            DiagnosticKind::Scientific,
                            DiagnosticSeverity::Info,
                            message,
                        );
                        diagnostic.fields =
                            Arc::from([(Arc::from("validator"), Arc::from("graph"))]);
                        diagnostics.push(diagnostic);
                    }
                    Err(error) => {
                        let reason = error.to_string();
                        let causal = CausalError::from(error);
                        if matches!(
                            causal,
                            CausalError::Cancelled { .. } | CausalError::Resource { .. }
                        ) {
                            return Err(causal);
                        }
                        let mut diagnostic = Diagnostic::new(
                            "refute.validator.failed",
                            DiagnosticKind::Scientific,
                            DiagnosticSeverity::Warning,
                            format!("validator graph could not compute a verdict: {reason}"),
                        );
                        diagnostic.fields = Arc::from([
                            (Arc::from("validator"), Arc::from("graph")),
                            (Arc::from("reason"), Arc::from(reason)),
                        ]);
                        diagnostics.push(diagnostic);
                    }
                }
            }
            for report in &mut reports {
                if report.scope.as_ref().is_some_and(|existing| existing != &scope) {
                    return Err(CausalError::Compile {
                        message:
                            "validator report scope disagrees with its actual class completion"
                                .into(),
                    });
                }
                report.scope = Some(scope.clone());
            }
            if !reports.is_empty() {
                let mut provenance = Diagnostic::new(
                    "refute.class_response.completion_weight",
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Info,
                    "point-producer completion weights are descriptive and do not certify an aggregate",
                );
                let conditional_weight = if inner_scale.is_finite()
                    && inner_scale > 0.0
                    && inner_total.is_finite()
                    && inner_total > 0.0
                    && eval.point_weights.iter().all(|weight| weight.is_finite() && *weight >= 0.0)
                {
                    ((*completion_weight / inner_scale) / inner_total).to_string()
                } else {
                    "unavailable".to_owned()
                };
                provenance.fields = Arc::from([
                    (Arc::from("outer_atom"), Arc::from(eval.key.to_string())),
                    (
                        Arc::from("completion_digest"),
                        Arc::from(antecedent_core::identity::to_hex(completion_digest)),
                    ),
                    (Arc::from("outer_graph_weight"), Arc::from(atom_weight.to_string())),
                    (Arc::from("completion_weight"), Arc::from(completion_weight.to_string())),
                    (Arc::from("within_graph_weight"), Arc::from(conditional_weight)),
                ]);
                diagnostics.push(provenance);
            }
            scoped_reports.extend(reports);
        }
    }
    // Each report is a completion-level result. Do not reduce it into an
    // apparently authoritative whole-target pass.
    Ok((scoped_reports, diagnostics))
}
