//! Score-table retargeting of a prepared handle.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{CachedStaticIdentification, ClusterUnits, PreparedStudy, score_table_treatment_col};
use crate::error::CausalError;
use crate::inference::InferenceMode;
use crate::result::StudyResult;
use antecedent_core::{CausalQuery, ExecutionContext};
use antecedent_estimate::{
    EffectEstimate, OverlapPolicy, RetargetResult, ScoreTable, exceedance_cdf_values,
};
use std::sync::Arc;

impl PreparedStudy {
    /// The prepared score table a retarget reweights.
    ///
    /// # Errors
    ///
    /// `score_table_unavailable` when prepare built none.
    fn retarget_score_table(&self) -> Result<&antecedent_estimate::ScoreTable, CausalError> {
        self.score_table.as_ref().ok_or_else(|| {
            crate::unsupported_reason!(
                "score_table_unavailable",
                "retarget requires a prepared score table on AverageEffect or discrete joint \
                 InterventionResponse"
            )
        })
    }

    /// Whether this handle can perform `intent` at all, before any data or
    /// weights are supplied.
    ///
    /// The one check both [`Self::preview_transform`] and the apply run, so a
    /// preview never authorizes what the apply refuses. Only retarget depends
    /// on the handle (it needs a prepared score table); a compatible-data
    /// refresh is always available, and its schema check needs the new data.
    /// The remaining intents change the question or structure and are
    /// performed by preparing a new study, not applied to this handle.
    ///
    /// # Errors
    ///
    /// The reason-coded refusal the apply raises.
    pub fn transform_capability(
        &self,
        intent: antecedent_contract::TransformIntent,
    ) -> Result<(), CausalError> {
        match intent {
            antecedent_contract::TransformIntent::Retarget => {
                self.retarget_score_table().map(|_| ())
            }
            _ => Ok(()),
        }
    }

    /// Estimate `E_Q[μ_a(X)]` from frozen scores. Does not refit or re-identify.
    ///
    /// `weights` must align with the score-table complete-case rows.
    ///
    /// **Score-table lifetime.** The table is the one frozen at prepare (or rebuilt by
    /// [`Self::refresh`], which replaces the retained data and its scores together).
    /// [`Self::estimate`] takes `&self` and never replaces it, so a retarget after an
    /// `estimate` on new rows still reweights the prepare-time rows; retarget the estimate's
    /// own rows through its result's `score_table` or, for a batch, through
    /// [`crate::analysis::PreparedBatch::estimate_scored`]. Where no table exists the refusal is
    /// `score_table_unavailable`, never a silent fall back to another construction.
    ///
    /// `depends_on` is the declared parent set of `w`; it must be a subset of
    /// the certified adjustment set and must not name the treatment, an
    /// intervened coordinate, or a descendant.
    ///
    /// # Errors
    ///
    /// Missing score table, illegal `depends_on`, weight shape, or weighted
    /// overlap failure (support refusal).
    pub fn retarget(
        &self,
        weights: &[f64],
        depends_on: &[antecedent_core::VariableId],
        _ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let table = self.retarget_score_table()?;
        let out = self.retarget_table(table, weights, depends_on)?;
        let inference = table.inference(Some(weights))?;
        self.retarget_to_result(out, inference, weights, depends_on)
    }

    /// Reweight `table` (this handle's prepare-time table, or the scores of a later
    /// estimate on the same plan) under the declared target, with the handle's graph for the
    /// `depends_on` check. The one gate behind [`Self::retarget`] and the batch retarget.
    ///
    /// # Errors
    ///
    /// Illegal `depends_on`, weight shape, or a weighted-overlap support refusal.
    pub(crate) fn retarget_table(
        &self,
        table: &ScoreTable,
        weights: &[f64],
        depends_on: &[antecedent_core::VariableId],
    ) -> Result<RetargetResult, CausalError> {
        let graph: Option<&dyn antecedent_estimate::DirectedAncestry> = self
            .analysis
            .graph
            .as_dag()
            .map(|g| g as _)
            .or_else(|| self.analysis.graph.as_admg().map(|g| g as _));
        let treatment = score_table_treatment_col(&self.analysis, table);
        let (mut out, overlap_failed) = antecedent_estimate::retarget(
            table,
            weights,
            depends_on,
            graph,
            treatment.as_deref(),
            None,
        )
        .map_err(|error| {
            crate::analysis::preflight::with_plan_subject(&self.analysis, CausalError::from(error))
        })?;
        if overlap_failed {
            let refusal = CausalError::Support {
                id: crate::support::SupportRefusal::Refused,
                message: antecedent_estimate::RetargetRefusal::WeightedOverlap.as_str(),
            }
            .diagnosed(antecedent_estimate::weighted_overlap_fields(
                table,
                weights,
                &out.support,
            ));
            return Err(crate::analysis::preflight::with_plan_subject(&self.analysis, refusal));
        }
        self.apply_cluster_covariance(table, weights, &mut out)?;
        Ok(out)
    }

    /// Declared independence units of a cluster-DML plan (labels aligned with the rows of its
    /// score table), or `None` for any other plan.
    pub(crate) fn cluster_units(&self, table: &ScoreTable) -> Option<ClusterUnits> {
        match self.analysis.estimator_spec.as_deref() {
            Some(crate::estimator_spec::EstimatorSpec::Aipw(config)) => {
                let (first, second) = config.unit_labels(&table.row_index).ok()?;
                Some(ClusterUnits { spec: config.cluster_dml?, first: first?, second })
            }
            Some(crate::estimator_spec::EstimatorSpec::Dml(config)) => {
                declared_units(&config.units, table)
            }
            Some(crate::estimator_spec::EstimatorSpec::DrLearner(config)) => {
                declared_units(&config.units, table)
            }
            _ => None,
        }
    }

    /// Replace the iid plug-in covariance of a retarget over a cluster-DML score table by the
    /// covariance of the cluster-summed weighted influence columns (the receipt's
    /// `G/(G-1)` sandwich, or the two-way `V_a + V_b - V_ab`), and the arm contrast's standard
    /// error with it. A plan without declared units, or a table that is not a cluster-DML mean
    /// table, is left as the iid retarget (whose cluster-marked provenance withholds it).
    fn apply_cluster_covariance(
        &self,
        table: &ScoreTable,
        weights: &[f64],
        out: &mut RetargetResult,
    ) -> Result<(), CausalError> {
        if !table.nuisance_provenance.contains("cluster_dml=unit=")
            || table.distinct_threshold_count() > 0
        {
            return Ok(());
        }
        let Some(units) = self.cluster_units(table) else {
            return Ok(());
        };
        if units.first.len() != table.n_rows {
            return Err(CausalError::Unsupported {
                message: "cluster labels are not aligned with the score table rows",
            });
        }
        let n = table.n_rows as f64;
        let total: f64 = weights.iter().sum();
        let mut columns: Vec<Vec<f64>> = Vec::with_capacity(table.n_columns());
        for j in 0..table.n_columns() {
            let theta = out.summary.means[j];
            columns.push(
                table
                    .column(j)?
                    .iter()
                    .zip(weights)
                    .map(|(phi, w)| n * (w / total) * (phi - theta))
                    .collect(),
            );
        }
        let refs: Vec<&[f64]> = columns.iter().map(Vec::as_slice).collect();
        let covariance =
            units.spec.influence_covariance(&refs, &units.first, units.second.as_deref())?;
        let arm = |arm: u32| table.columns.iter().position(|c| c.arm == arm);
        if let (Some(contrast), Some(control), Some(active)) =
            (out.contrast.as_mut(), arm(0), arm(1))
        {
            let variance = covariance.get(active, active) + covariance.get(control, control)
                - 2.0 * covariance.get(active, control);
            contrast.se = variance.max(0.0).sqrt();
        }
        out.summary.covariance = covariance.clone();
        out.covariance = covariance;
        out.cluster_reference_df =
            Some(units.spec.reference_df(&units.first, units.second.as_deref()));
        Ok(())
    }

    /// Weighted quantile value and row influence of a quantile-functional query, if it is one.
    fn retarget_quantile(
        &self,
        table: &ScoreTable,
        weights: &[f64],
    ) -> Result<Option<(f64, Vec<f64>)>, CausalError> {
        Ok(match &self.analysis.query {
            CausalQuery::AverageEffect(q) => q
                .outcome_functional
                .quantile_level()
                .map(|tau| {
                    antecedent_estimate::quantile::quantile_contrast(table, Some(weights), tau)
                        .map(|q| (q.value, q.influence))
                })
                .transpose()?,
            CausalQuery::Response(q) => q
                .outcome_functional
                .quantile_level()
                .map(|tau| {
                    let arm = crate::analysis::helpers::requested_joint_arm(q)?;
                    let q = antecedent_estimate::quantile::quantile_arm(
                        table,
                        Some(weights),
                        tau,
                        arm,
                    )?;
                    Ok::<_, CausalError>((q.value, q.influence))
                })
                .transpose()?,
            _ => None,
        })
    }

    /// Point value and standard error of the retargeted scalar the query reports.
    fn retarget_point_and_se(
        &self,
        table: &ScoreTable,
        out: &RetargetResult,
        n_thresholds: usize,
    ) -> Result<(f64, f64), CausalError> {
        Ok(match &self.analysis.query {
            CausalQuery::AverageEffect(_) | CausalQuery::Response(_) if n_thresholds > 1 => {
                (f64::NAN, f64::NAN)
            }
            CausalQuery::AverageEffect(_) => {
                let c = out
                    .contrast
                    .as_ref()
                    .ok_or(CausalError::Unsupported { message: "missing arm contrast" })?;
                (c.value, c.se)
            }
            CausalQuery::Response(q) => {
                let arm = crate::analysis::helpers::requested_joint_arm(q)?;
                let col = table
                    .columns
                    .iter()
                    .position(|c| c.arm == arm)
                    .ok_or(CausalError::Unsupported { message: "missing joint cell" })?;
                (out.summary.means[col], out.covariance.se(col))
            }
            _ => return Err(CausalError::Unsupported { message: "unsupported retarget query" }),
        })
    }

    /// The identification certificate a retargeted result carries from the prepare-time cache.
    fn retarget_certificate(
        &self,
        cache: &CachedStaticIdentification,
    ) -> crate::result::AnalysisIdentification {
        crate::result::AnalysisIdentification {
            identification: crate::Identification::Point {
                result: cache.identification.clone(),
                temporal_indexer: None,
                strategy: self
                    .plan
                    .binding()
                    .identifier
                    .named()
                    .unwrap_or(crate::strategy_table::DEFAULT_IDENTIFIER_ID),
                structure_version: self.analysis.graph.version(),
            },
            query: self.analysis.query.clone(),
            graph_class: self.analysis.graph.class(),
        }
    }

    /// The scalar response payload of a retargeted `InterventionResponse` result.
    fn retarget_response_payload(
        q: &antecedent_core::ResponseQuery,
        cache: &CachedStaticIdentification,
        result: &StudyResult,
    ) -> antecedent_core::CausalResponse {
        antecedent_core::CausalResponse {
            estimand: q.functional.clone(),
            identification_status: cache.identification.status,
            estimate: antecedent_core::ResponseIdentification::PointIdentified(
                antecedent_core::ResponseValue::Scalar(
                    result.estimate.as_effect().map_or(f64::NAN, |effect| effect.ate),
                ),
            ),
            uncertainty: antecedent_core::ResponseUncertainty::Scalar {
                standard_error: result
                    .estimate
                    .as_effect()
                    .map_or(f64::NAN, antecedent_estimate::EffectEstimate::se_analytic),
                lower: result.estimate.as_effect().map_or(f64::NAN, |effect| {
                    effect.ate - crate::result::reported_se_interval_z() * effect.se_analytic()
                }),
                upper: result.estimate.as_effect().map_or(f64::NAN, |effect| {
                    effect.ate + crate::result::reported_se_interval_z() * effect.se_analytic()
                }),
                level: 0.95,
                interpretation: antecedent_core::IntervalInterpretation::Confidence,
                draws: None,
            },
            support: antecedent_core::SupportReport {
                status: antecedent_core::SupportStatus::Supported,
                query_region: antecedent_core::SupportRegion {
                    minima: Arc::from([]),
                    maxima: Arc::from([]),
                },
                diagnostics: Vec::new(),
                warnings: Vec::new(),
                // Same single-coordinate label as the solo cell route.
                point_status: Some(Arc::from([antecedent_core::SupportStatus::Supported])),
            },
            assumptions: cache.identification.required_assumptions.clone(),
            provenance_id: Arc::from("estimate.cell.aipw.retarget"),
            horizon_identification: None,
            interaction_structurally_zero: false,
        }
    }

    /// Publish the cluster-summed standard error with its few-cluster t reference.
    fn publish_cluster_reference(&self, estimate: &mut EffectEstimate, se: f64, df: usize) {
        let dyadic = match self.analysis.estimator_spec.as_deref() {
            Some(crate::estimator_spec::EstimatorSpec::Aipw(config)) => config.has_second_units(),
            Some(crate::estimator_spec::EstimatorSpec::Dml(config)) => config.units.has_second(),
            Some(crate::estimator_spec::EstimatorSpec::DrLearner(config)) => {
                config.units.has_second()
            }
            _ => false,
        };
        estimate.uncertainty.analytic.se = se;
        estimate.uncertainty.analytic.reference_df =
            Some(f64::from(u32::try_from(df).unwrap_or(u32::MAX)));
        estimate.uncertainty.analytic.kind = Some(if dyadic {
            antecedent_estimate::AnalyticSeKind::Multiway
        } else {
            antecedent_estimate::AnalyticSeKind::Cluster
        });
    }

    /// Treatment and outcome the retargeted query reports on.
    fn retarget_treatment_outcome(
        &self,
    ) -> Result<(antecedent_core::VariableId, antecedent_core::VariableId), CausalError> {
        match &self.analysis.query {
            CausalQuery::AverageEffect(q) => Ok((q.treatment, q.outcome)),
            CausalQuery::Response(q) => q
                .functional
                .primary_pair()
                .ok_or(CausalError::Unsupported { message: "retarget response missing pair" }),
            _ => Err(CausalError::Unsupported {
                message: "retarget is licensed for AverageEffect and InterventionResponse",
            }),
        }
    }

    #[allow(clippy::float_cmp)] // Exact membership in binary intervention levels.
    fn retarget_to_result(
        &self,
        out: RetargetResult,
        inference: antecedent_estimate::scores::ScoreInference,
        weights: &[f64],
        depends_on: &[antecedent_core::VariableId],
    ) -> Result<StudyResult, CausalError> {
        let cache =
            self.analysis.identification_cache.as_ref().ok_or(CausalError::Unsupported {
                message: "retarget requires prepare-time identification",
            })?;
        let table = self
            .score_table
            .as_ref()
            .ok_or(CausalError::Unsupported { message: "missing frozen scores" })?;
        let n_thresholds = table.distinct_threshold_count();
        let quantile = self.retarget_quantile(table, weights)?;
        let (ate, se) = self.retarget_point_and_se(table, &out, n_thresholds)?;
        let cdf = self.score_table.as_ref().and_then(|t| exceedance_cdf_values(&out.summary, t));
        // Scores of whole-cluster folds or learner-supplied joint cells retarget to a point; their
        // iid covariance is not a licensed sampling distribution, so no standard error, joint
        // covariance or band.
        let cluster_df = out.cluster_reference_df;
        let interval_withheld =
            antecedent_estimate::provenance_withholds_interval(&table.nuisance_provenance)
                && cluster_df.is_none();
        let se = if interval_withheld { f64::NAN } else { se };
        let mut estimate = EffectEstimate::new(
            ate,
            se,
            cache.identification.required_assumptions.clone(),
            OverlapPolicy::RequireDiagnostics { clip: Some(0.01), trim: None },
        )
        .with_score_table(self.score_table.clone())
        .with_joint_covariance((!interval_withheld).then(|| out.covariance.clone()))
        .with_exceedance_cdf(cdf)
        .with_monotone_rearranged(out.monotone_rearranged);
        // The iid max-t score inference is not a cluster object: it stays withheld there.
        estimate.scores.inference =
            (!interval_withheld && cluster_df.is_none()).then_some(inference);
        if let Some(df) = cluster_df {
            self.publish_cluster_reference(&mut estimate, se, df);
        }
        if let Some((value, influence)) = quantile.as_ref() {
            estimate.ate = *value;
            estimate.uncertainty.analytic.se =
                antecedent_estimate::joint_influence_covariance(&[influence], None)?.se(0);
            estimate.scores.influence = Some(influence.clone().into());
        }
        let (treatment, outcome) = self.retarget_treatment_outcome()?;
        let diagnostics = retarget_diagnostics(
            out.diagnostics,
            RetargetDiagnosticFlags {
                interval_withheld,
                quantile: quantile.is_some(),
                grid_scalar_cleared: n_thresholds > 1 && quantile.is_none(),
            },
        );
        let mut result =
            crate::analysis::helpers::assemble_result(crate::analysis::helpers::AssembleArgs {
                logical: &self.plan.logical.record,
                physical: &self.plan.record,
                identification: cache.identification.clone(),
                estimand: cache.estimand.clone(),
                estimate,
                posterior: None,
                payload: crate::result::ResultPayload::Scalar,
                refutations: Vec::new(),
                diagnostics,
                provenance: antecedent_contract::ProvenanceGraph::new(),
                treatment,
                outcome,
                wall_time_ns: 0,
                latency_mode: None,
                stage_timings_ns: Vec::new(),
                bootstrap_replicates_requested: None,
                bootstrap_replicates_ok: None,
                n_draws: None,
                cancelled: false,
                early_stopped: false,
                bayesian: matches!(self.analysis.inference, InferenceMode::Bayesian(_)),
            });
        result.certificate = Some(self.retarget_certificate(cache));
        if antecedent_estimate::changes_target(weights) {
            let binding = self.row_weights_binding(weights, depends_on)?;
            if let Some(certificate) = &mut result.certificate {
                if let Some(target) = certificate.query.target_population_mut() {
                    *target = binding.population();
                }
            }
            result.row_weights = Some(binding);
        }
        if let CausalQuery::Response(q) = &self.analysis.query {
            result.payload = crate::result::ResultPayload::Response(
                Self::retarget_response_payload(q, cache, &result),
            );
        }
        result.rebind_interval(matches!(self.analysis.inference, InferenceMode::Bayesian(_)));
        result.support_status = self.analysis.support_status;
        result.structure_source = self.analysis.structure_source;
        let retargeted = result.retarget_population();
        result.executed_contract = Some(self.executed_contract(
            &self.analysis.data,
            self.analysis.refute,
            retargeted.as_ref(),
        )?);
        Ok(result)
    }
}

/// Which optional notes a retargeted result carries.
#[derive(Clone, Copy)]
struct RetargetDiagnosticFlags {
    interval_withheld: bool,
    quantile: bool,
    grid_scalar_cleared: bool,
}

/// The retarget diagnostics appended after the weighting step's own, in contract order.
fn retarget_diagnostics(
    mut diagnostics: Vec<antecedent_core::Diagnostic>,
    flags: RetargetDiagnosticFlags,
) -> Vec<antecedent_core::Diagnostic> {
    if flags.interval_withheld {
        diagnostics.push(antecedent_core::Diagnostic::new(
            "estimate.aipw.penalized_interval_withheld",
            antecedent_core::DiagnosticKind::Scientific,
            antecedent_core::DiagnosticSeverity::Info,
            "the prepared scores come from whole-cluster folds or learner-supplied joint-cell nuisances; the retargeted point is reported without a standard error, covariance or band because no iid interval is licensed for that construction",
        ));
    }
    if flags.quantile {
        diagnostics.push(antecedent_core::Diagnostic::new(
            "estimate.functional.quantile", antecedent_core::DiagnosticKind::Scientific,
            antecedent_core::DiagnosticSeverity::Info,
            "weighted piecewise-linear CDF inversion conditional on the frozen grid; grid-selection uncertainty and interpolation bias are excluded",
        ));
    }
    if flags.grid_scalar_cleared {
        diagnostics.push(antecedent_core::Diagnostic::new(
            "estimate.functional.grid_scalar_cleared",
            antecedent_core::DiagnosticKind::Scientific,
            antecedent_core::DiagnosticSeverity::Info,
            "exceedance grids do not publish a first-threshold scalar ATE; use exceedance_cdf and the score table",
        ));
    }
    diagnostics.push(antecedent_core::Diagnostic::new(
        "estimate.aipw.crossfit_scores",
        antecedent_core::DiagnosticKind::Scientific,
        antecedent_core::DiagnosticSeverity::Info,
        "retarget averages the prepared cross-fitted φ table; it is not a residualized full-sample AIPW refit, so it can differ from the estimator's own point value under uniform weights",
    ));
    diagnostics.push(antecedent_core::Diagnostic::new(
        "retarget.selection_assumption", antecedent_core::DiagnosticKind::Scientific,
        antecedent_core::DiagnosticSeverity::Info,
        "weights are caller-declared fixed functions of certified covariates; inference assumes iid sampling, positivity, and nuisance convergence; selection or weight-estimation uncertainty is excluded",
    ));
    diagnostics.push(antecedent_core::Diagnostic::new(
        "exec.identify.cached",
        antecedent_core::DiagnosticKind::Execution,
        antecedent_core::DiagnosticSeverity::Info,
        "identification reused from the prepare-time cache",
    ));
    diagnostics
}

/// The declared independence units of a whole-cluster cross-fit, on the rows of `table`.
fn declared_units(
    units: &antecedent_estimate::DeclaredUnits,
    table: &ScoreTable,
) -> Option<ClusterUnits> {
    let (first, second) = units.labels(&table.row_index).ok()?;
    Some(ClusterUnits { spec: units.spec?, first: first?, second })
}
