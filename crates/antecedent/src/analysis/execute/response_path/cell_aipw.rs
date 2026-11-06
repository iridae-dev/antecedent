//! Cell-AIPW response execution.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    Arc, AverageEffectQuery, CausalError, CellAipwReading, CellAipwTarget, Diagnostic,
    DiagnosticKind, DiagnosticSeverity, EffectEstimate, EstimationWorkspace, ExecuteRun,
    ExecutionContext, IdentifiedClaim, IdentifiedEstimand, IdentifiedExecuteExtras,
    IdentifiedExecuteFinish, Intervention, OverlapPolicy, RefutationRun, RefuteSuite,
    ResponseFunctional, ResponseQuery, Strategy, StudyResult, TabularData, VariableId,
    cell_aipw_response, cell_contrast_for_refute, elapsed_ns, identify_cached_diagnostic,
    response_primary_pair,
};

impl super::super::Study {
    /// Reads the binary joint `Set` levels of a cell.aipw intervention response.
    #[allow(clippy::float_cmp)] // Exact membership in binary intervention levels.
    fn parse_cell_aipw_interventions(
        interventions: &[Intervention],
    ) -> Result<(Vec<VariableId>, u32), CausalError> {
        let mut treatments = Vec::new();
        let mut requested_arm = 0u32;
        for (j, iv) in interventions.iter().enumerate() {
            let Intervention::Set { variable, value } = iv else {
                return Err(CausalError::Unsupported {
                    message: "cell.aipw requires every intervention to be Set",
                });
            };
            let level = value.as_f64().ok_or(CausalError::Unsupported {
                message: "cell.aipw requires binary numeric Set levels",
            })?;
            if level != 0.0 && level != 1.0 {
                return Err(CausalError::Unsupported {
                    message: "cell.aipw requires binary 0/1 Set levels",
                });
            }
            if j >= antecedent_estimate::cell_aipw::MAX_JOINT_BINARY
                || treatments.contains(variable)
            {
                return Err(CausalError::Unsupported {
                    message: "cell.aipw requires distinct coordinates, at most as many as a cell code addresses",
                });
            }
            treatments.push(*variable);
            requested_arm |= u32::from(level == 1.0) << j;
        }
        Ok((treatments, requested_arm))
    }

    /// Cell-saturated AIPW score table for the joint arms, unless one was precomputed.
    fn cell_aipw_score_table(
        &self,
        data: &TabularData,
        query: &ResponseQuery,
        target: &CellAipwTarget<'_>,
        estimand: &IdentifiedEstimand,
        precomputed_table: Option<antecedent_estimate::ScoreTable>,
        ctx: &ExecutionContext,
    ) -> Result<antecedent_estimate::ScoreTable, CausalError> {
        let CellAipwTarget { treatments, outcome } = *target;
        let est =
            antecedent_estimate::CellSaturatedAipw::new().with_fold_seed(ctx.rng.master_seed());
        let continuous = None;
        // Folds are deliberately not shared here: `fit_scores_with_assignment` leaves
        // them unset so the cell path draws its own cell-stratified
        // `crossfit_fold_plan` (keyed by `est.fold_seed`), reproducing the solo fold
        // plan for this query's own cells bit-for-bit instead of a private,
        // unstratified batch shuffle. See `SharedBatchDesign` docs.
        let fold_ids: Option<Vec<u32>> = None;
        let design = match self.shared_batch_design.as_ref() {
            Some(shared) => {
                let ids: Vec<_> = treatments
                    .iter()
                    .copied()
                    .chain(std::iter::once(outcome))
                    .chain(estimand.adjustment_set.iter().copied())
                    .collect();
                let row_index = match data.complete_case_mask(&ids) {
                    Ok(mask) => mask
                        .iter()
                        .enumerate()
                        .filter_map(|(i, &keep)| {
                            keep.then_some(antecedent_data::GlobalId::new(
                                u64::try_from(i).expect("row id"),
                            ))
                        })
                        .collect::<Vec<_>>(),
                    Err(_) => Vec::new(),
                };
                if row_index.is_empty() {
                    None
                } else {
                    shared.design_for(&estimand.adjustment_set, &row_index)?
                }
            }
            None => None,
        };
        let table = if let Some(table) = precomputed_table {
            table
        } else {
            let spec = antecedent_estimate::CellScoreSpec {
                treatments,
                outcome,
                adjustment: &estimand.adjustment_set,
                functional: &query.outcome_functional,
                continuous,
            };
            let shared = antecedent_estimate::SharedCellInputs {
                fold_ids: fold_ids.as_deref(),
                design: design.as_deref(),
            };
            // A table that carries observation weights is fitted for the weighted population.
            if data.storage().weights().is_some() {
                est.fit_scores_sampling_weighted(data, &spec, shared)
                    .map(|(table, _)| table)
                    .map_err(CausalError::from)?
            } else {
                est.fit_scores_with_assignment(data, &spec, shared).map_err(CausalError::from)?
            }
        };
        Ok(table)
    }

    /// Refuse the options of a joint-cell study that the weighted or dependence-aware
    /// construction does not read.
    ///
    /// The quantile inversion, the refutation suite and a bootstrap resample the rows as
    /// independent and unweighted, so under sampling weights or a declared dependence each
    /// would publish a number for another population or another sampling design.
    fn refuse_cell_aipw_unread_options(&self, query: &ResponseQuery) -> Result<(), CausalError> {
        if query.outcome_functional.quantile_level().is_some() {
            return Err(crate::unsupported_reason!(
                "option_not_applicable",
                "the joint-cell quantile inverts an unweighted, independent-row exceedance \
                 grid, so sampling weights or a declared cluster or serial dependence would be \
                 silently ignored by it. Next attempt: ask for the mean or an exceedance \
                 probability, which read the weights and the dependence, or drop them"
            ));
        }
        if !matches!(self.settings.refute, RefuteSuite::None) {
            return Err(crate::unsupported_reason!(
                "option_not_applicable",
                "the refutation suite of the joint cell reads unweighted independent rows, so \
                 sampling weights or a declared cluster or serial dependence would be silently \
                 ignored by it. Next attempt: run the study with RefuteSuite::None"
            ));
        }
        Ok(())
    }

    /// The sampling weights and the dependence the joint cell's variance must read, aligned to
    /// the rows of `table`.
    ///
    /// The weights are the table's observation weights of the analysed rows, and a table that
    /// was fitted without them (or with them when the data has none) is refused rather than
    /// summarized under the other population.
    fn cell_aipw_reading(
        &self,
        data: &TabularData,
        table: &antecedent_estimate::ScoreTable,
        target: &CellAipwTarget<'_>,
        estimand: &IdentifiedEstimand,
    ) -> Result<CellAipwReading, CausalError> {
        let fitted_weighted = table.nuisance_provenance.contains("sampling_weights");
        let rows: Vec<usize> = table
            .row_index
            .iter()
            .map(|id| usize::try_from(id.get()))
            .collect::<Result<_, _>>()
            .map_err(|_| CausalError::Compile {
                message: "a cell score row index does not fit the platform's address width".into(),
            })?;
        let (weights, record) = match data.storage().weights() {
            None if fitted_weighted => {
                return Err(CausalError::Compile {
                    message: "the cell scores were fitted with sampling weights but the data \
                              carries none"
                        .into(),
                });
            }
            None => (None, None),
            Some(_) if !fitted_weighted => {
                return Err(CausalError::Compile {
                    message: "the data carries sampling weights but the cell scores were fitted \
                              without them; refit the scores on this data"
                        .into(),
                });
            }
            Some(source) => {
                let used: Vec<f64> = rows
                    .iter()
                    .map(|&row| {
                        source.get(row).copied().ok_or_else(|| CausalError::Compile {
                            message: "a cell score row lies beyond the table's weights".into(),
                        })
                    })
                    .collect::<Result<_, _>>()?;
                let sum: f64 = used.iter().sum();
                let square: f64 = used.iter().map(|w| w * w).sum();
                let mut ids: Vec<_> = target.treatments.to_vec();
                ids.push(target.outcome);
                ids.extend(estimand.adjustment_set.iter().copied());
                let zero_weight_rows = data
                    .complete_case_mask(&ids)
                    .map(|mask| {
                        mask.iter()
                            .zip(source)
                            .filter(|(keep, weight)| **keep && **weight == 0.0)
                            .count()
                    })
                    .map_err(CausalError::from)?;
                let record = antecedent_estimate::SamplingWeightRecord {
                    rows_used: used.len() as u64,
                    zero_weight_rows: zero_weight_rows as u64,
                    effective_sample_size: sum * sum / square,
                    weight_sum: sum,
                };
                (Some(used), Some(record))
            }
        };
        let dependence = self
            .declared
            .score_dependence
            .as_ref()
            .map(|declared| declared.restricted(&rows))
            .transpose()
            .map_err(CausalError::from)?;
        Ok(CellAipwReading { weights, dependence, record })
    }

    /// Level, standard error and influence column of the requested cell.
    #[allow(clippy::float_cmp)] // Exact equality of duplicate threshold values.
    fn cell_aipw_level(
        table: &antecedent_estimate::ScoreTable,
        summary: &antecedent_estimate::ScoreSummary,
        query: &ResponseQuery,
        requested_arm: u32,
        functional_diagnostics: &mut Vec<Diagnostic>,
    ) -> Result<(f64, f64, Arc<[f64]>), CausalError> {
        let n_thresholds = {
            let mut t: Vec<f64> = table.columns.iter().filter_map(|c| c.threshold).collect();
            t.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            t.dedup_by(|a, b| *a == *b);
            t.len()
        };
        let column = table
            .columns
            .iter()
            .position(|c| {
                c.arm == requested_arm
                    && (n_thresholds <= 1
                        || c.threshold == table.columns.first().and_then(|c| c.threshold))
            })
            .or_else(|| table.columns.iter().position(|c| c.arm == requested_arm))
            .ok_or(CausalError::Unsupported { message: "unsupported requested cell" })?;
        let level = antecedent_estimate::LinearContrast {
            value: summary.means[column],
            se: summary.covariance.se(column),
        };
        let quantile = query
            .outcome_functional
            .quantile_level()
            .map(|tau| antecedent_estimate::quantile::quantile_arm(table, None, tau, requested_arm))
            .transpose()?;
        let (scalar, se) = if let Some(q) = &quantile {
            functional_diagnostics.push(super::super::super::helpers::quantile_scope_diagnostic());
            (q.value, antecedent_estimate::joint_influence_covariance(&[&q.influence], None)?.se(0))
        } else if n_thresholds > 1 {
            functional_diagnostics.push(Diagnostic::new(
                "estimate.functional.grid_scalar_cleared",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "exceedance grids do not publish a first-threshold scalar ATE; use exceedance_cdf and the score table",
            ));
            (f64::NAN, f64::NAN)
        } else {
            (level.value, level.se)
        };
        let influence = match quantile {
            Some(q) => Arc::from(q.influence),
            None => Arc::from(table.column(column)?),
        };
        Ok((scalar, se, influence))
    }

    #[allow(clippy::float_cmp)] // Exact comparison of the requested cell with the control contrast.
    #[allow(
        clippy::too_many_lines,
        reason = "the compiled route keeps its bounded response execution together"
    )]
    pub(in crate::analysis::execute) fn execute_cell_aipw_response(
        &self,
        data: &TabularData,
        query: &ResponseQuery,
        run: ExecuteRun<'_>,
        claim: IdentifiedClaim,
        strategy: Strategy,
        precomputed_table: Option<antecedent_estimate::ScoreTable>,
    ) -> Result<StudyResult, CausalError> {
        let ExecuteRun { physical, ctx, started, identify_cached } = run;
        let IdentifiedClaim { identification, estimand } = claim;
        let Strategy { identifier: identifier_id, estimator: estimator_id } = strategy;
        let ResponseFunctional::InterventionResponse { outcome, interventions } = &query.functional
        else {
            return Err(CausalError::Unsupported {
                message: "cell.aipw is supported for discrete joint InterventionResponse",
            });
        };
        let (treatments, requested_arm) = Self::parse_cell_aipw_interventions(interventions)?;
        if self.continuous_cell.is_some() {
            return Err(CausalError::Unsupported {
                message: antecedent_estimate::POINT_CDE_UNSUPPORTED,
            });
        }
        if treatments.len() < 2 {
            return Err(CausalError::Unsupported {
                message: "cell.aipw requires at least two binary Set interventions",
            });
        }
        let target = CellAipwTarget { treatments: &treatments, outcome: *outcome };
        let declared_reading =
            data.storage().weights().is_some() || self.declared.score_dependence.is_some();
        if declared_reading {
            self.refuse_cell_aipw_unread_options(query)?;
        }
        let table =
            self.cell_aipw_score_table(data, query, &target, &estimand, precomputed_table, ctx)?;
        let reading = self.cell_aipw_reading(data, &table, &target, &estimand)?;
        let (summary, monotone_rearranged, mut functional_diagnostics) =
            antecedent_estimate::summarize_functional_with(
                &table,
                reading.weights.as_deref(),
                reading.dependence.as_ref(),
            )?;
        let (scalar, se, influence) = Self::cell_aipw_level(
            &table,
            &summary,
            query,
            requested_arm,
            &mut functional_diagnostics,
        )?;
        let cdf = antecedent_estimate::exceedance_cdf_values(&summary, &table);
        let response =
            cell_aipw_response(query, &identification, scalar, se, treatments.len(), requested_arm);
        let score_inference =
            table.inference_with(reading.weights.as_deref(), reading.dependence.as_ref())?;
        let mut estimate = EffectEstimate::new(
            scalar,
            se,
            identification.required_assumptions.clone(),
            OverlapPolicy::RequireDiagnostics { clip: Some(0.01), trim: None },
        )
        .with_score_table(Some(table))
        // The influence column is the unweighted, independent-row score; with sampling weights
        // or a declared dependence it is not the influence the reported error is built from.
        .with_influence(if scalar.is_finite() && !declared_reading {
            Some(influence)
        } else {
            None
        })
        .with_sampling_weights(reading.record)
        .with_exceedance_cdf(cdf)
        .with_joint_covariance(Some(summary.covariance))
        .with_monotone_rearranged(monotone_rearranged);
        estimate.scores.inference = Some(score_inference);
        let (treatment, outcome) = response_primary_pair(&query.functional)?;
        let mut diagnostics = identification.diagnostics.clone();
        diagnostics.extend(functional_diagnostics);
        if identify_cached {
            diagnostics.push(identify_cached_diagnostic());
        }
        let ate_query = AverageEffectQuery::binary_ate(treatment, outcome);
        let refute_estimate = cell_contrast_for_refute(&estimate, requested_arm);
        if !matches!(self.settings.refute, RefuteSuite::None) && refute_estimate.ate != estimate.ate
        {
            diagnostics.push(Diagnostic::new(
                "refute.cell_aipw.contrast",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "cell.aipw cheap/full apply overlap and E-value to the requested-cell minus control-cell contrast, not the intervention level",
            ));
        }
        let mut refute_ws = EstimationWorkspace::default();
        let (refutations, refute_diags) = RefutationRun {
            data,
            estimand: &estimand,
            query: &ate_query,
            estimate: &refute_estimate,
            suite: self.settings.refute,
            estimator: "cell.aipw",
            custom: &self.declared.custom_validators,
            temporal: None,
        }
        .reports(&mut refute_ws, None, ctx)?;
        diagnostics.extend(refute_diags);
        Ok(self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand,
                estimate,
                Strategy::new(identifier_id, estimator_id),
                treatment,
                outcome,
            )
            .identify_cached(identify_cached)
            .refutations(refutations)
            .wall_time_ns(elapsed_ns(started))
            .extras(IdentifiedExecuteExtras {
                diagnostics: Some(diagnostics),
                response: Some(response),
                bootstrap_replicates_requested: Some(None),
                ..Default::default()
            }),
        ))
    }
}
