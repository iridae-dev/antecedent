// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    Admg, Arc, AverageEffectQuery, BayesianConfig, CausalError, CausalQuery, CausalResponse,
    ClassAtomOutcome, ClassEnvelopeRun, Diagnostic, DiagnosticKind, DiagnosticSeverity,
    EffectEstimate, EnvelopeRefutation, EnvelopeRefuteAtom, EstimationWorkspace, EstimatorId,
    ExecuteRun, ExecutionContext, IdentificationEnvelope, IdentificationResult,
    IdentificationStatus, IdentifiedEstimand, IdentifiedExecuteExtras, IdentifiedExecuteFinish,
    IdentifierId, InferenceDiagnostics, InferenceMode, Instant, Intervention, OverlapPolicy, Pag,
    PhysicalExecutionPlan, PriorSet, RefutationRun, RefuteSuite, ResponseFunctional,
    ResponseIdentification, ResponseQuery, ResponseUncertainty, ResponseValue, StaticEstimateInput,
    StaticEstimateWorkspaces, Strategy, StudyResult, TabularData, VariableId,
    credible_scalar_uncertainty, effect_from_posterior, elapsed_ns,
    envelope_se_omission_diagnostic, envelope_to_identification_result, estimate_static_effect,
    identification_from_cache_or, identification_status_ok_for_case, identify_admg,
    identify_admg_query, identify_cached_diagnostic, mix_static_envelope_se,
    mix_weighted_analytic_se, mixed_static_influence, nonidentified_with_prior, overlap_diagnostic,
    overlap_limit_diagnostic, provenance_ids, report_identify_compute, select_estimand,
    static_aligned_influence, support_from_functional_eval,
};
use crate::planner::RouteDefaults;

/// Route and borrowed inputs shared by every level of an ADMG response.
struct AdmgLevelRun<'a> {
    data: &'a TabularData,
    admg: &'a Admg,
    curve: bool,
    bayesian: bool,
    identifier_id: IdentifierId,
    estimator_id: EstimatorId,
    treatment: VariableId,
    outcome: VariableId,
    physical: &'a PhysicalExecutionPlan,
    ctx: &'a ExecutionContext,
}

/// Per-level results of an ADMG response, accumulated in level order.
struct AdmgLevelFits {
    means: Vec<f64>,
    grid: Vec<f64>,
    identification: Option<IdentificationResult>,
    estimand: Option<IdentifiedEstimand>,
    identify_cached: bool,
    extras_posterior: Option<antecedent_estimate::CausalPosterior>,
    n_draws: Option<u32>,
    curve_bounds: Vec<(f64, f64)>,
    freq_curve_bounds: Vec<(f64, f64)>,
    freq_scalar: Option<ResponseUncertainty>,
    support: Option<antecedent_core::SupportReport>,
    extra_diagnostics: Vec<Diagnostic>,
}

/// The published response of an ADMG level set with the values its result assembly reads.
struct AdmgLevelsResponse {
    response: CausalResponse,
    identification: IdentificationResult,
    estimand: IdentifiedEstimand,
    identify_cached: bool,
    extra_diagnostics: Vec<Diagnostic>,
    n_draws: Option<u32>,
    posterior: Option<antecedent_estimate::CausalPosterior>,
}

impl AdmgLevelFits {
    /// Publishes the collected levels as one response, scalar or curve.
    fn into_response(self, query: &ResponseQuery, curve: bool) -> AdmgLevelsResponse {
        let AdmgLevelFits {
            means,
            grid,
            identification,
            estimand,
            identify_cached,
            extras_posterior,
            n_draws,
            curve_bounds,
            freq_curve_bounds,
            freq_scalar,
            support,
            extra_diagnostics,
        } = self;
        let identification = identification.expect("ADMG response identified at least one level");
        let estimand = estimand.expect("ADMG response selected at least one estimand");
        let mut response_support = support.unwrap_or_else(|| {
            support_from_functional_eval(None).expect("empty functional eval publishes support")
        });
        if !grid.is_empty() {
            let (minima, maxima) = if curve {
                let min = grid.iter().copied().fold(f64::INFINITY, f64::min);
                let max = grid.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                (vec![min], vec![max])
            } else {
                (grid.clone(), grid.clone())
            };
            response_support.query_region = antecedent_core::SupportRegion {
                minima: Arc::from(minima),
                maxima: Arc::from(maxima),
            };
        }
        let response_len = means.len();
        let estimate_payload = if curve {
            ResponseIdentification::PointIdentified(ResponseValue::Surface {
                grid: Arc::from(grid),
                dimension: 1,
                mean: Arc::from(means),
            })
        } else {
            ResponseIdentification::PointIdentified(ResponseValue::Scalar(
                means.first().copied().unwrap_or(f64::NAN),
            ))
        };
        let response = CausalResponse {
            estimand: query.functional.clone(),
            identification_status: identification.status,
            estimate: estimate_payload,
            uncertainty: if curve && curve_bounds.len() == response_len && !curve_bounds.is_empty()
            {
                ResponseUncertainty::PointwiseBand {
                    level: crate::result::REPORTED_SE_INTERVAL_LEVEL,
                    lower: Arc::from(
                        curve_bounds.iter().map(|(lower, _)| *lower).collect::<Vec<_>>(),
                    ),
                    upper: Arc::from(
                        curve_bounds.iter().map(|(_, upper)| *upper).collect::<Vec<_>>(),
                    ),
                    interpretation: antecedent_core::IntervalInterpretation::Credible,
                    draws: None,
                }
            } else if curve
                && freq_curve_bounds.len() == response_len
                && !freq_curve_bounds.is_empty()
            {
                ResponseUncertainty::PointwiseBand {
                    level: crate::result::REPORTED_SE_INTERVAL_LEVEL,
                    lower: Arc::from(
                        freq_curve_bounds.iter().map(|(lower, _)| *lower).collect::<Vec<_>>(),
                    ),
                    upper: Arc::from(
                        freq_curve_bounds.iter().map(|(_, upper)| *upper).collect::<Vec<_>>(),
                    ),
                    interpretation: antecedent_core::IntervalInterpretation::Confidence,
                    draws: None,
                }
            } else if curve {
                ResponseUncertainty::None
            } else {
                extras_posterior
                    .as_ref()
                    .and_then(credible_scalar_uncertainty)
                    .or(freq_scalar)
                    .unwrap_or(ResponseUncertainty::None)
            },
            support: response_support,
            assumptions: identification.required_assumptions.clone(),
            provenance_id: Arc::from("estimate.response.general_id"),
            horizon_identification: None,
            interaction_structurally_zero: false,
        };
        AdmgLevelsResponse {
            response,
            identification,
            estimand,
            identify_cached,
            extra_diagnostics,
            n_draws,
            posterior: extras_posterior,
        }
    }
}

/// Names a CPDAG/PAG class path in its refusals and disclosures.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ClassTag {
    Cpdag,
    Pag,
    /// The finite window of a temporal PAG.
    TemporalPag,
}

impl ClassTag {
    /// Lowercase tag used in diagnostic ids and prose.
    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::Cpdag => "cpdag",
            Self::Pag => "pag",
            Self::TemporalPag => "temporal_pag",
        }
    }

    /// Uppercase name used at the head of a refusal.
    const fn upper(self) -> &'static str {
        match self {
            Self::Cpdag => "CPDAG",
            Self::Pag => "PAG",
            Self::TemporalPag => "TEMPORAL PAG",
        }
    }
}

impl std::fmt::Display for ClassTag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Per-completion results of a class ATE, accumulated in envelope order.
struct ClassAteAtoms {
    weighted_ate: f64,
    se_items: Vec<(f64, f64)>,
    atom_ifs: Vec<Vec<f64>>,
    atom_weights: Vec<f64>,
    total_w: f64,
    primary_estimand: Option<IdentifiedEstimand>,
    assumptions: antecedent_core::AssumptionSet,
    refute_atoms: Vec<EnvelopeRefuteAtom>,
    outcomes: Vec<ClassAtomOutcome>,
}

impl ClassAteAtoms {
    /// Mass-weighted point estimate, primary estimand and contributing-completion count.
    fn pooled_estimate(
        &self,
        upper: &str,
    ) -> Result<(EffectEstimate, IdentifiedEstimand, usize), CausalError> {
        if !matches!(self.total_w.partial_cmp(&0.0), Some(std::cmp::Ordering::Greater)) {
            return Err(CausalError::Compile {
                message: format!("{upper} envelope had no estimable identified cases"),
            });
        }
        let estimand = self.primary_estimand.clone().ok_or_else(|| CausalError::Compile {
            message: format!("{upper} envelope missing estimand"),
        })?;
        let n_contributing = self.se_items.len();
        let se = if self.atom_ifs.len() == n_contributing {
            mix_static_envelope_se(&self.atom_ifs, &self.atom_weights)
        } else {
            mix_weighted_analytic_se(self.se_items.iter().copied())
        };
        let mut estimate = EffectEstimate::new(
            self.weighted_ate / self.total_w,
            se,
            self.assumptions.clone(),
            OverlapPolicy::ExplicitOverride,
        );
        if self.atom_ifs.len() == n_contributing {
            if let Some(inf) = mixed_static_influence(&self.atom_ifs, &self.atom_weights) {
                estimate.scores.influence = Some(inf);
            }
        }
        Ok((estimate, estimand, n_contributing))
    }
}

/// The response query and the per-level queries an ADMG response finish evaluates.
#[derive(Clone, Copy)]
struct AdmgResponseLevels<'a> {
    query: &'a ResponseQuery,
    level_queries: &'a [ResponseQuery],
    /// A curve over a grid rather than one intervention level.
    curve: bool,
}

impl super::Study {
    /// ADMG ATE via general ID + functional plug-in (bidirected case).
    pub(super) fn execute_admg(
        &self,
        data: &TabularData,
        admg: &Admg,
        query: &AverageEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let (identifier_id, estimator_id) = physical.binding().resolve(RouteDefaults::ADMG)?;
        if !matches!(estimator_id, EstimatorId::FunctionalEffect) {
            return Err(CausalError::Compile {
                message: format!(
                    "ADMG ATE requires estimator functional.effect; got {estimator_id}"
                ),
            });
        }

        let reused = self.reuse_static_identification_if(physical, |_| {
            physical.binding().resolves_to(RouteDefaults::STATIC_ATE, identifier_id, estimator_id)
        });
        let (identification, estimand, identify_cached) =
            identification_from_cache_or(ctx, reused, || {
                let identification = identify_admg(identifier_id, admg, query)?;
                let estimand = select_estimand(&identification, estimator_id)?;
                Ok((identification, estimand))
            })?;
        let (estimate, posterior) = self.estimate_functional_effect(
            data,
            &estimand,
            &identification,
            &[query.treatment, query.outcome],
            ctx,
        )?;

        let mut refute_ws = EstimationWorkspace::default();
        let (refutations, extra_diagnostics) = RefutationRun {
            data,
            estimand: &estimand,
            query,
            estimate: &estimate,
            suite: self.settings.refute,
            estimator: estimator_id.as_str(),
            custom: &self.declared.custom_validators,
            temporal: None,
        }
        .reports(&mut refute_ws, None, ctx)?;

        Ok(self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand,
                estimate,
                Strategy::new(identifier_id, estimator_id),
                query.treatment,
                query.outcome,
            )
            .identify_cached(identify_cached)
            .extra_diagnostics(extra_diagnostics)
            .refutations(refutations)
            .wall_time_ns(elapsed_ns(started))
            .extras(IdentifiedExecuteExtras {
                n_draws: posterior
                    .as_ref()
                    .map(|p| u32::try_from(p.draws.n_draws).unwrap_or(u32::MAX)),
                posterior,
                ..Default::default()
            }),
        ))
    }

    /// ADMG conditional effect via IDC at the two hard-set levels + functional plug-in
    /// (bidirected case).
    ///
    /// The effect modifiers are free variables of the identified contrast. The evaluator
    /// reports the contrast averaged over their observed law, the quantity the Dag
    /// conditional routes report. Validation is `none`: the ATE refuters do not apply to
    /// an IDC contrast.
    pub(super) fn execute_admg_conditional(
        &self,
        data: &TabularData,
        admg: &Admg,
        query: &antecedent_core::ConditionalEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let (identifier_id, estimator_id) = physical.binding().resolve(RouteDefaults::ADMG)?;
        if identifier_id != IdentifierId::GeneralId
            || !matches!(estimator_id, EstimatorId::FunctionalEffect)
        {
            return Err(CausalError::Compile {
                message: format!(
                    "ADMG ConditionalEffect requires general.id and functional.effect; got \
                     {identifier_id} and {estimator_id}"
                ),
            });
        }
        if self.settings.refute != RefuteSuite::None || !self.declared.custom_validators.is_empty()
        {
            return Err(CausalError::Unsupported {
                message: "ADMG ConditionalEffect is licensed at validation none; the ATE \
                          refuters and custom validators do not apply to an IDC contrast",
            });
        }
        // The retained IDC claim is this route's when it carries the one estimand the
        // executor would select.
        let reused = self.reuse_static_identification_if(physical, |cache| {
            cache.identification.estimands.len() == 1
        });
        let (identification, estimand, identify_cached) =
            identification_from_cache_or(ctx, reused, || {
                let identification =
                    crate::analysis::conditional_admg::admg_conditional_identification(
                        admg, query,
                    )?;
                let estimand = select_estimand(&identification, estimator_id)?;
                Ok((identification, estimand))
            })?;
        let (estimate, posterior) = self.estimate_functional_effect(
            data,
            &estimand,
            &identification,
            &[query.inner.treatment, query.inner.outcome],
            ctx,
        )?;
        Ok(self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand,
                estimate,
                Strategy::new(identifier_id, estimator_id),
                query.inner.treatment,
                query.inner.outcome,
            )
            .identify_cached(identify_cached)
            .wall_time_ns(elapsed_ns(started))
            .extras(IdentifiedExecuteExtras {
                n_draws: posterior
                    .as_ref()
                    .map(|p| u32::try_from(p.draws.n_draws).unwrap_or(u32::MAX)),
                posterior,
                ..Default::default()
            }),
        ))
    }

    /// ADMG response via general ID + functional plug-in (bidirected case).
    ///
    /// Bidirected edges stay bidirected. Each grid / Set level is the identified
    /// `P(Y | do(X=x))` mean, not an adjustment g-formula.
    pub(super) fn execute_admg_response(
        &self,
        data: &TabularData,
        admg: &Admg,
        query: &ResponseQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let is_intervention =
            matches!(query.functional, ResponseFunctional::InterventionResponse { .. });
        if self.settings.refute != RefuteSuite::None && !is_intervention {
            return Err(CausalError::Unsupported {
                message: "Admg ResponseCurve has no native non-ATE refuter suite; \
                          ATE refuters do not apply to a function-valued surface",
            });
        }
        match &query.functional {
            ResponseFunctional::InterventionResponse { .. } => self.finish_admg_response_levels(
                data,
                admg,
                AdmgResponseLevels {
                    query,
                    level_queries: &[admg_response_single_level(query)?],
                    curve: false,
                },
                physical,
                ctx,
                started,
            ),
            ResponseFunctional::MeanCurve { outcome, treatment } => {
                let levels = treatment.grid.values()?;
                let level_queries = levels
                    .iter()
                    .map(|level| {
                        admg_response_at_level(query, treatment.variable, *outcome, *level)
                    })
                    .collect::<Vec<_>>();
                self.finish_admg_response_levels(
                    data,
                    admg,
                    AdmgResponseLevels { query, level_queries: &level_queries, curve: true },
                    physical,
                    ctx,
                    started,
                )
            }
            _ => Err(CausalError::Unsupported {
                message: "Admg response is licensed for InterventionResponse and ResponseCurve",
            }),
        }
    }

    fn finish_admg_response_levels(
        &self,
        data: &TabularData,
        admg: &Admg,
        levels: AdmgResponseLevels<'_>,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
        started: Instant,
    ) -> Result<StudyResult, CausalError> {
        let AdmgResponseLevels { query, level_queries, curve } = levels;
        let (identifier_id, estimator_id) = physical.binding().resolve(RouteDefaults::ADMG)?;
        if !matches!(estimator_id, EstimatorId::FunctionalEffect) {
            return Err(CausalError::Compile {
                message: format!(
                    "ADMG response requires estimator functional.effect; got {estimator_id}"
                ),
            });
        }
        let (treatment, outcome) = super::response_path::response_primary_pair(&query.functional)?;
        let run = AdmgLevelRun {
            data,
            admg,
            curve,
            bayesian: self.settings.inference.is_bayesian(),
            identifier_id,
            estimator_id,
            treatment,
            outcome,
            physical,
            ctx,
        };
        let AdmgLevelsResponse {
            response,
            identification,
            estimand,
            identify_cached,
            extra_diagnostics,
            n_draws,
            posterior: extras_posterior,
        } = self.fit_admg_response_levels(&run, level_queries)?.into_response(query, curve);
        let (scalar, standard_error) = super::response_path::response_scalar_summary(&response);
        let estimate = EffectEstimate::new(
            scalar,
            standard_error,
            response.assumptions.clone(),
            OverlapPolicy::ExplicitOverride,
        );
        let mut diagnostics = identification.diagnostics.clone();
        diagnostics.push(Diagnostic::new(
            "identify.response.general_id",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "intervention mean identified by Shpitser–Pearl ID; levels are the \
             discrete functional.effect plug-in, not an adjustment g-formula",
        ));
        if identify_cached {
            diagnostics.push(identify_cached_diagnostic());
        }
        diagnostics.extend(extra_diagnostics);
        let (refutations, refute_diags) =
            self.refute_admg_response_level(&run, &estimand, &estimate, scalar, &mut diagnostics)?;
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
            .refutations(refutations)
            .wall_time_ns(elapsed_ns(started))
            .extras(IdentifiedExecuteExtras {
                estimate_provenance: Some(provenance_ids(
                    "estimate.response.general_id",
                    "estimate.response.general_id",
                )),
                diagnostics: Some(diagnostics),
                response: Some(response),
                bootstrap_replicates_requested: Some(None),
                n_draws,
                posterior: extras_posterior,
                ..Default::default()
            }),
        ))
    }

    /// Identifies and estimates every intervention level of an ADMG response.
    fn fit_admg_response_levels(
        &self,
        run: &AdmgLevelRun<'_>,
        level_queries: &[ResponseQuery],
    ) -> Result<AdmgLevelFits, CausalError> {
        let mut fits = AdmgLevelFits {
            means: Vec::with_capacity(level_queries.len()),
            grid: Vec::with_capacity(level_queries.len()),
            identification: None,
            estimand: None,
            identify_cached: false,
            extras_posterior: None,
            n_draws: None,
            curve_bounds: Vec::new(),
            freq_curve_bounds: Vec::new(),
            freq_scalar: None,
            support: None,
            extra_diagnostics: Vec::new(),
        };
        for (i, level_query) in level_queries.iter().enumerate() {
            let (level_id, level_est, cached) = if i == 0 {
                let reused = self.reuse_static_identification_if(run.physical, |_| {
                    run.physical.binding().resolves_to(
                        RouteDefaults::STATIC_ATE,
                        run.identifier_id,
                        run.estimator_id,
                    )
                });
                identification_from_cache_or(run.ctx, reused, || {
                    let identification = identify_admg_query(
                        run.identifier_id,
                        run.admg,
                        &CausalQuery::Response(level_query.clone()),
                    )?;
                    let estimand = select_estimand(&identification, run.estimator_id)?;
                    Ok((identification, estimand))
                })?
            } else {
                let identification = identify_admg_query(
                    run.identifier_id,
                    run.admg,
                    &CausalQuery::Response(level_query.clone()),
                )?;
                let estimand = select_estimand(&identification, run.estimator_id)?;
                (identification, estimand, false)
            };
            if i == 0 {
                fits.identify_cached = cached;
            }
            let (mean, posterior, level_support) =
                self.fit_admg_response_level(run, level_query, &level_id, &level_est, &mut fits)?;
            if let ResponseFunctional::InterventionResponse { interventions, .. } =
                &level_query.functional
            {
                if let Some(Intervention::Set { value, .. }) = interventions.first() {
                    if let Some(level) = value.as_f64() {
                        fits.grid.push(level);
                    }
                }
            }
            fits.means.push(mean);
            if fits.extras_posterior.is_none() {
                fits.extras_posterior.clone_from(&posterior);
                fits.n_draws =
                    posterior.as_ref().map(|p| u32::try_from(p.draws.n_draws).unwrap_or(u32::MAX));
            }
            if run.curve {
                if let Some(ResponseUncertainty::Scalar { lower, upper, .. }) =
                    posterior.as_ref().and_then(credible_scalar_uncertainty)
                {
                    fits.curve_bounds.push((lower, upper));
                }
            }
            if fits.support.is_none() {
                fits.support = Some(level_support);
            }
            if fits.identification.is_none() {
                fits.identification = Some(level_id);
                fits.estimand = Some(level_est);
            }
        }
        Ok(fits)
    }

    /// Estimates one ADMG response level, Bayesian or as a frequentist plug-in.
    fn fit_admg_response_level(
        &self,
        run: &AdmgLevelRun<'_>,
        level_query: &ResponseQuery,
        level_id: &IdentificationResult,
        level_est: &IdentifiedEstimand,
        fits: &mut AdmgLevelFits,
    ) -> Result<
        (f64, Option<antecedent_estimate::CausalPosterior>, antecedent_core::SupportReport),
        CausalError,
    > {
        if run.bayesian {
            let (estimate, posterior) = self.estimate_functional_effect(
                run.data,
                level_est,
                level_id,
                &[run.treatment, run.outcome],
                run.ctx,
            )?;
            Ok((
                estimate.ate,
                posterior,
                support_from_functional_eval(None).map_err(|e| {
                    CausalError::from(antecedent_estimate::EstimationError::data_msg(e.to_string()))
                })?,
            ))
        } else {
            let (response, _) = super::response_path::estimate_general_id_response(
                run.data,
                level_query,
                level_id,
                level_est,
                self.functional_bootstrap(self.settings.bootstrap_replicates)?,
                run.ctx,
            )?;
            fits.extra_diagnostics.extend(response.support.warnings.iter().cloned());
            if let ResponseUncertainty::Scalar { lower, upper, .. } = &response.uncertainty {
                if run.curve {
                    fits.freq_curve_bounds.push((*lower, *upper));
                } else if fits.freq_scalar.is_none() {
                    fits.freq_scalar = Some(response.uncertainty.clone());
                }
            }
            let (scalar, _) = super::response_path::response_scalar_summary(&response);
            Ok((scalar, None, response.support))
        }
    }

    /// Plugin-level refutation of a scalar ADMG response level, when one is licensed.
    fn refute_admg_response_level(
        &self,
        run: &AdmgLevelRun<'_>,
        estimand: &IdentifiedEstimand,
        estimate: &EffectEstimate,
        scalar: f64,
        diagnostics: &mut Vec<Diagnostic>,
    ) -> Result<(Vec<antecedent_validate::RefutationReport>, Vec<Diagnostic>), CausalError> {
        if !run.curve
            && scalar.is_finite()
            && (!matches!(self.settings.refute, RefuteSuite::None)
                || !self.declared.custom_validators.is_empty())
        {
            let ate_query = AverageEffectQuery::binary_ate(run.treatment, run.outcome);
            let mut refute_ws = EstimationWorkspace::default();
            if matches!(self.settings.refute, RefuteSuite::Cheap | RefuteSuite::Full) {
                diagnostics.push(Diagnostic::new(
                    "refute.evalue.not_a_contrast",
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Info,
                    "contrast-shaped refuters are not licensed for a plugin intervention level; \
                     cheap runs overlap only and full runs overlap plus sampling-stability of the \
                     identified intervention mean",
                ));
            }
            RefutationRun {
                data: run.data,
                estimand,
                query: &ate_query,
                estimate,
                suite: self.settings.refute,
                estimator: run.estimator_id.as_str(),
                custom: &self.declared.custom_validators,
                temporal: None,
            }
            .plugin_level(&mut refute_ws, run.ctx)
        } else {
            Ok((Vec::new(), Vec::new()))
        }
    }

    /// PAG ATE via generalized-adjustment envelope + mass-weighted estimates.
    pub(in crate::analysis) fn execute_pag(
        &self,
        data: &TabularData,
        pag: &Pag,
        query: &AverageEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let (identifier_id, estimator_id) = physical.binding().resolve(RouteDefaults::PAG)?;
        let (envelope, identification, identify_cached) =
            if let Some((cache, prepared)) = self.reuse_pag_identification(physical) {
                (Arc::clone(&cache.envelope), cache.identification.clone(), prepared)
            } else {
                report_identify_compute(ctx);
                let envelope = crate::strategy_table::identify_pag_in_context(
                    identifier_id,
                    pag,
                    query,
                    Some(ctx),
                )?;
                let identification = envelope_to_identification_result(&envelope, query);
                (Arc::new(envelope), identification, false)
            };
        let certificate = crate::Identification::Envelope {
            envelope: (*envelope).clone(),
            strategy: identifier_id,
            structure_version: self.graph.version(),
        };
        self.finish_class_ate(
            data,
            query,
            ClassEnvelopeRun {
                run: ExecuteRun { physical, ctx, started, identify_cached },
                envelope: &envelope,
                envelope_diagnostic: pag_envelope_diagnostic(&envelope),
            },
            identification,
            Strategy::new(identifier_id, estimator_id),
            ClassTag::Pag,
        )
        .map(|result| self.attach_certificate(result, certificate))
    }

    /// CPDAG ATE via MEC-completion envelope + mass-weighted estimates.
    pub(in crate::analysis) fn execute_cpdag(
        &self,
        data: &TabularData,
        cpdag: &antecedent_graph::Cpdag,
        query: &AverageEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let (identifier_id, estimator_id) = physical.binding().resolve(RouteDefaults::PAG)?;
        let (envelope, identification, identify_cached) =
            if let Some((cache, prepared)) = self.reuse_cpdag_identification(physical) {
                (Arc::clone(&cache.envelope), cache.identification.clone(), prepared)
            } else {
                report_identify_compute(ctx);
                let envelope = crate::strategy_table::identify_cpdag_in_context(
                    identifier_id,
                    cpdag,
                    query,
                    Some(ctx),
                )?;
                let identification = envelope_to_identification_result(&envelope, query);
                (Arc::new(envelope), identification, false)
            };
        let certificate = crate::Identification::CpdagEnvelope {
            envelope: (*envelope).clone(),
            strategy: identifier_id,
            structure_version: self.graph.version(),
        };
        self.finish_class_ate(
            data,
            query,
            ClassEnvelopeRun {
                run: ExecuteRun { physical, ctx, started, identify_cached },
                envelope: &envelope,
                envelope_diagnostic: cpdag_envelope_diagnostic(&envelope),
            },
            identification,
            Strategy::new(identifier_id, estimator_id),
            ClassTag::Cpdag,
        )
        .map(|result| self.attach_certificate(result, certificate))
    }

    /// Shared CPDAG/PAG ATE finish: refuses or hands off an unidentified or Bayesian envelope,
    /// otherwise pools the identified completions.
    fn finish_class_ate<G>(
        &self,
        data: &TabularData,
        query: &AverageEffectQuery,
        run: ClassEnvelopeRun<'_, G>,
        identification: IdentificationResult,
        strategy: Strategy,
        label: ClassTag,
    ) -> Result<StudyResult, CausalError> {
        let (tag, upper) = (label.as_str(), label.upper());
        let estimator_id = strategy.estimator;
        let envelope = run.envelope;
        if matches!(envelope.status, IdentificationStatus::NotIdentified)
            || envelope.identified_weight.0 <= 0.0
        {
            if self.settings.inference.is_bayesian()
                || matches!(estimator_id, EstimatorId::BayesianGcomp)
            {
                return self.execute_pag_nonidentified_prior(query, run, identification, tag);
            }
            return Err(CausalError::not_identified(
                envelope.status,
                envelope.truncated_completions > 0,
                &format!("{upper} effect not identified (no identified mass in envelope)"),
            ));
        }
        if matches!(estimator_id, EstimatorId::BayesianGcomp) {
            return self.execute_pag_bayesian(data, query, run, identification, label);
        }
        self.finish_class_ate_frequentist(data, query, run, identification, strategy, label)
    }

    /// Mass-weighted frequentist ATE over the identified completions of a class envelope.
    fn finish_class_ate_frequentist<G>(
        &self,
        data: &TabularData,
        query: &AverageEffectQuery,
        run: ClassEnvelopeRun<'_, G>,
        identification: IdentificationResult,
        strategy: Strategy,
        label: ClassTag,
    ) -> Result<StudyResult, CausalError> {
        let (tag, upper) = (label.as_str(), label.upper());
        let Strategy { identifier: identifier_id, estimator: estimator_id } = strategy;
        let ClassEnvelopeRun {
            run: ExecuteRun { physical, ctx, started, identify_cached },
            envelope,
            envelope_diagnostic,
        } = run;
        let mut diagnostics = vec![envelope_diagnostic];
        let atoms = self.evaluate_class_ate_atoms(data, query, envelope, estimator_id, ctx)?;
        let (estimate, estimand, n_contributing) = atoms.pooled_estimate(upper)?;
        let mut refute_ws = EstimationWorkspace::default();
        let (refutations, na_diagnostics) = EnvelopeRefutation {
            data,
            query,
            suite: self.settings.refute,
            estimator: estimator_id.as_str(),
            custom: &self.declared.custom_validators,
            temporal: None,
        }
        .run(&atoms.refute_atoms, &mut refute_ws, ctx)?;
        diagnostics.extend(na_diagnostics);

        diagnostics.push(overlap_diagnostic(estimate.provenance.overlap));

        diagnostics.extend(overlap_limit_diagnostic(estimate.diagnostics.overlap_report.as_ref()));
        diagnostics.extend(envelope_se_omission_diagnostic(n_contributing, estimate.se_analytic()));
        // A partially identified envelope carries its identified set and the
        // per-completion values it spans; a point-identified one stays a point.
        let structural_response = static_class_structural_mixture(envelope, &atoms.outcomes);
        if let Some(mixture) = structural_response.as_ref() {
            diagnostics.extend(static_class_identified_set_diagnostic(mixture, tag));
        }
        Ok(self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand,
                estimate,
                Strategy::new(identifier_id, estimator_id),
                query.treatment,
                query.outcome,
            )
            .identify_cached(identify_cached)
            .refutations(refutations)
            .wall_time_ns(elapsed_ns(started))
            .extras(IdentifiedExecuteExtras {
                diagnostics: Some(diagnostics),
                structural_response,
                ..Default::default()
            }),
        ))
    }

    /// Estimates every identified completion of a class envelope with the configured estimator.
    fn evaluate_class_ate_atoms<G>(
        &self,
        data: &TabularData,
        query: &AverageEffectQuery,
        envelope: &IdentificationEnvelope<G>,
        estimator_id: EstimatorId,
        ctx: &ExecutionContext,
    ) -> Result<ClassAteAtoms, CausalError> {
        let mut atoms = ClassAteAtoms {
            weighted_ate: 0.0,
            se_items: Vec::new(),
            atom_ifs: Vec::new(),
            atom_weights: Vec::new(),
            total_w: 0.0,
            primary_estimand: None,
            assumptions: antecedent_core::AssumptionSet::default(),
            refute_atoms: Vec::new(),
            outcomes: vec![ClassAtomOutcome::NotEvaluated; envelope.cases.len()],
        };
        for (i, case) in envelope.cases.iter().enumerate() {
            if !identification_status_ok_for_case(case.result.status)
                || case.result.estimands.is_empty()
            {
                continue;
            }
            let estimand = select_estimand(&case.result, estimator_id)?;
            let mut case_ws = StaticEstimateWorkspaces::default();
            // Honour a caller-configured estimator across every equivalence-class case,
            // falling back to id-only selection when none was supplied.
            let case_spec = self
                .declared
                .estimator_spec
                .as_deref()
                .cloned()
                .unwrap_or(crate::estimator_spec::EstimatorSpec::Default(estimator_id));
            let estimate = estimate_static_effect(
                &case_spec,
                StaticEstimateInput {
                    data,
                    estimand: &estimand,
                    query,
                    bootstrap_replicates: self.settings.bootstrap_replicates,
                    overlap_policy: self.declared.overlap_policy,
                    population_registry: self.declared.population_registry.as_ref(),
                    sampling_design: self.declared.sampling_design,
                },
                case.result.required_assumptions.clone(),
                ctx,
                &mut case_ws,
            )?;
            let w = case.weight.0;
            atoms.outcomes[i] = ClassAtomOutcome::Evaluated(estimate.ate);
            atoms.weighted_ate += w * estimate.ate;
            atoms.se_items.push((w, estimate.se_analytic()));
            if let Some(inf) = estimate
                .scores
                .influence
                .as_deref()
                .and_then(|inf| static_aligned_influence(data, query, &estimand, inf))
            {
                atoms.atom_ifs.push(inf);
                atoms.atom_weights.push(w);
            }
            atoms.total_w += w;
            if atoms.primary_estimand.is_none() {
                atoms.primary_estimand = Some(estimand.clone());
                atoms.assumptions = estimate.provenance.assumptions.clone();
            }
            atoms.refute_atoms.push(EnvelopeRefuteAtom {
                key: i as u64,
                weight: w,
                estimand,
                indexer: None,
                original: estimate,
            });
        }
        Ok(atoms)
    }

    /// Non-identified PAG with Bayesian inference: prior-predictive draws, no invented ID.
    pub(super) fn execute_pag_nonidentified_prior<G>(
        &self,
        query: &AverageEffectQuery,
        run: ClassEnvelopeRun<'_, G>,
        identification: IdentificationResult,
        class_tag: &str,
    ) -> Result<StudyResult, CausalError> {
        let ClassEnvelopeRun {
            run: ExecuteRun { physical, ctx, started, identify_cached },
            envelope,
            envelope_diagnostic,
        } = run;
        let cfg = match &self.settings.inference {
            InferenceMode::Bayesian(c) => c.clone(),
            InferenceMode::Frequentist => BayesianConfig::laplace(),
        };
        let scale = cfg.prior_scale.max(1e-6);
        let mut prior = PriorSet::weakly_informative(1);
        if let Some(g) = prior.specs.iter_mut().find_map(|s| match s {
            antecedent_prob::PriorSpec::GaussianCoefficients(p) => Some(p),
            _ => None,
        }) {
            *g = antecedent_prob::GaussianCoefficientPrior::isotropic(1, scale);
        }
        // The single coefficient of `prior` is the effect: N(0, scale²).
        let posterior = nonidentified_with_prior(
            &prior,
            0.0,
            scale,
            InferenceDiagnostics::analytic(format!("{class_tag}_nonidentified_prior")),
            cfg.n_draws.max(1),
            ctx.rng.master_seed(),
        )
        .map_err(CausalError::from)?;
        let estimate = effect_from_posterior(&posterior)?;
        let estimand = envelope.invariant.clone().unwrap_or_else(|| {
            IdentifiedEstimand::backdoor(
                format!("{class_tag}.nonidentified"),
                Arc::from([]),
                antecedent_expr::ExprId::from_raw(0),
            )
        });
        let mut diagnostics = identification.diagnostics.clone();
        diagnostics.push(envelope_diagnostic);
        diagnostics.push(
            Diagnostic::new(
                format!("estimate.{class_tag}.nonidentified_prior"),
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Warning,
                format!(
                    "{class_tag} not identified; returning prior-predictive draws (unidentified_mass={})",
                    posterior.unidentified_mass
                ),
            )
            .with_fields(super::mass_fields(
                None,
                posterior.unidentified_mass,
            )),
        );
        Ok(self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand,
                estimate,
                Strategy::new(IdentifierId::GeneralizedAdjustment, EstimatorId::BayesianGcomp),
                query.treatment,
                query.outcome,
            )
            .identify_cached(identify_cached)
            .wall_time_ns(elapsed_ns(started))
            .extras(IdentifiedExecuteExtras {
                estimate_provenance: Some(provenance_ids(
                    "estimate.bayesian_gcomp",
                    "estimate.nonidentified_with_prior",
                )),
                posterior: Some(posterior),
                diagnostics: Some(diagnostics),
                ..Default::default()
            }),
        ))
    }
}

/// Structural uncertainty of a static class (CPDAG/PAG) envelope, keyed by
/// case index.
///
/// The mixture itself is [`super::class_structural_mixture`],
/// shared with the temporal class routes. This arm adds only the static
/// class's publishing rule: nothing to publish when the envelope carried no
/// mass or no completion produced a finite value, and `None` for a
/// point-identified envelope — one identified completion, or several that
/// agree, with no unidentified, unevaluable or subsampled-out mass. Such a
/// result is a point, not an identified set, and attaching a mixture would
/// restate it as bounds.
///
/// Static class envelopes publish the frozen-weight mixture's sampling
/// interval (the construction `static_envelope_calibration` scores); no
/// interval for the identified set is constructed on this path.
pub(super) fn static_class_structural_mixture<G>(
    envelope: &IdentificationEnvelope<G>,
    outcomes: &[ClassAtomOutcome],
) -> Option<crate::result::StructuralResponseMixture> {
    let (mixture, facts) = super::class_structural_mixture(
        envelope,
        crate::result::StructuralWeightBasis::CompletionEnumeration,
        |index, _| u64::try_from(index).unwrap_or(u64::MAX),
        None,
        outcomes,
    );
    if facts.total_weight <= 0.0 || mixture.identified_set.is_none() || facts.point_identified {
        return None;
    }
    Some(mixture)
}

/// The identified-set diagnostic a partially identified static class effect
/// reports next to its mixture summary.
pub(super) fn static_class_identified_set_diagnostic(
    mixture: &crate::result::StructuralResponseMixture,
    class_tag: &str,
) -> Option<Diagnostic> {
    let set = mixture.identified_set.as_ref()?;
    Some(
        Diagnostic::new(
            format!("estimate.{class_tag}.identified_set"),
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            format!(
                "identified set over identified completions: [{}, {}]; identified_mass={}; \
             unidentified_mass={}; unevaluable_mass={}; incomplete_search_mass={}; the \
             reported scalar is the frozen-weight mixture over identified completions, \
             and completion weights are enumeration weights, not posterior probabilities",
                set.lower[0],
                set.upper[0],
                mixture.identified_mass,
                mixture.unidentified_mass,
                mixture.unevaluable_mass,
                mixture.subsampled_out_mass,
            ),
        )
        .with_fields(super::mass_fields(Some(mixture.identified_mass), mixture.unidentified_mass)),
    )
}

/// Completion-mass summary every PAG arm (Frequentist, Bayesian, non-identified
/// prior) reports, so the envelope a fixture pins is observable on each.
pub(super) fn pag_envelope_diagnostic<G>(envelope: &IdentificationEnvelope<G>) -> Diagnostic {
    super::class_envelope_diagnostic(
        "identify.pag.envelope",
        "generalized.adjustment envelope",
        envelope,
        &[],
    )
}

/// Completion-mass summary for the CPDAG MEC envelope.
pub(super) fn cpdag_envelope_diagnostic<G>(envelope: &IdentificationEnvelope<G>) -> Diagnostic {
    super::class_envelope_diagnostic("identify.cpdag.envelope", "cpdag.mec envelope", envelope, &[])
}

fn admg_response_at_level(
    query: &ResponseQuery,
    treatment: VariableId,
    outcome: VariableId,
    level: f64,
) -> ResponseQuery {
    let mut level_query = query.clone();
    level_query.functional = ResponseFunctional::InterventionResponse {
        outcome,
        interventions: Arc::from([Intervention::set(
            treatment,
            antecedent_core::Value::f64(level),
        )]),
    };
    level_query
}

fn admg_response_single_level(query: &ResponseQuery) -> Result<ResponseQuery, CausalError> {
    match &query.functional {
        ResponseFunctional::InterventionResponse { .. } => Ok(query.clone()),
        _ => Err(CausalError::Compile {
            message: "ADMG intervention response requires InterventionResponse".into(),
        }),
    }
}
