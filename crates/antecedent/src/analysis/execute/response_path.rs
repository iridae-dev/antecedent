// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

impl super::Study {
    /// Identify and estimate a continuous response, including licensed observation correction.
    pub(super) fn execute_response(
        &self,
        data: &TabularData,
        graph: &Dag,
        query: &ResponseQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let identifier =
            physical.logical.record.identifier.as_deref().unwrap_or(DEFAULT_RESPONSE_IDENTIFIER);
        let estimator =
            physical.logical.record.estimator.as_deref().unwrap_or(DEFAULT_RESPONSE_ESTIMATOR);
        let identifier_id: IdentifierId = identifier.parse()?;
        let estimator_id: EstimatorId = estimator.parse()?;
        let cell_aipw = matches!(estimator_id, EstimatorId::CellAipw);

        let (identification, estimand, identify_cached) =
            identification_from_cache_or(ctx, self.identification_cache.as_deref(), || {
                let identification = identify_static_query(
                    identifier_id,
                    graph,
                    &CausalQuery::Response(query.clone()),
                )?;
                let estimand = identification.estimands.first().cloned().ok_or_else(|| {
                    CausalError::Compile {
                        message: "response identifier returned no estimand".into(),
                    }
                })?;
                Ok((identification, estimand))
            })?;
        if identification
            .estimands
            .iter()
            .any(|candidate| candidate.adjustment_set != estimand.adjustment_set)
        {
            return Err(CausalError::Unsupported {
                message: "multi-pair response estimation requires one common adjustment set",
            });
        }

        if estimand.method_kind().ok() == Some(EstimandMethod::GeneralId)
            || estimand.method_kind().ok() == Some(EstimandMethod::FrontDoor)
        {
            let (response, _) =
                estimate_general_id_response(data, query, &identification, &estimand, ctx)?;
            let (scalar, standard_error) = response_scalar_summary(&response);
            let estimate = EffectEstimate::new(
                scalar,
                standard_error,
                response.assumptions.clone(),
                OverlapPolicy::ExplicitOverride,
            );
            let (treatment, outcome) = response_primary_pair(&query.functional)?;
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
            for warning in &response.support.warnings {
                diagnostics.push(warning.clone());
            }
            return Ok(self.finish_identified_execute(IdentifiedExecuteFinish {
                physical,
                identification,
                estimand,
                estimate,
                identifier_id,
                estimator_id,
                treatment,
                outcome,
                identify_cached: false,
                extra_diagnostics: Vec::new(),
                refutations: Vec::new(),
                distribution: None,
                mediation: None,
                wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
                bootstrap_replicates_ok: None,
                cancelled: false,
                early_stopped: false,
                extras: IdentifiedExecuteExtras {
                    estimate_provenance: Some(provenance_ids(
                        "estimate.response.general_id",
                        "estimate.response.general_id",
                    )),
                    diagnostics: Some(diagnostics),
                    response: Some(response),
                    bootstrap_replicates_requested: Some(None),
                    ..Default::default()
                },
            }));
        }

        if cell_aipw {
            return self.execute_cell_aipw_response(
                data,
                query,
                physical,
                ctx,
                identification,
                estimand,
                identifier_id,
                estimator_id,
                identify_cached,
                started,
            );
        }

        let (_, outcome) = response_primary_pair(&query.functional)?;
        let data_est = super::super::helpers::apply_scalar_outcome_functional(
            data,
            outcome,
            &query.outcome_functional,
        )?;

        let mut response_estimator =
            ContinuousResponseEstimator::new(Arc::clone(&estimand.adjustment_set));
        if let Some(options) = &self.response_options {
            response_estimator.options = options.clone();
        }
        let mut response_scores = None;
        let response = if let InferenceMode::Bayesian(cfg) = &self.inference {
            if cfg.prior_artifact.is_some() || cfg.external_compose.is_some() {
                return Err(CausalError::Unsupported { message: "Bayesian response prior transfer requires a response-specific mapping; only explicit coefficient or isotropic priors are supported" });
            }
            let mut bayes = bayesian_gcomp(cfg, ctx);
            bayes.prior.clone_from(&cfg.prior);
            response_estimator.estimate_bayesian(&data_est, query, identification.status, identification.required_assumptions.clone(), &bayes, ctx)
        } else if query.observation == ObservationSpec::Complete {
            let (response, scores) = response_estimator
                .estimate_identified_scored(
                    &data_est,
                    query,
                    identification.status,
                    identification.required_assumptions.clone(),
                )
                .map_err(CausalError::from)?;
            response_scores = scores;
            Ok(response)
        } else {
            ObservationMechanismEstimator::new(self.observation_options).estimate_mean_curve(
                &response_estimator,
                &data_est,
                query,
                self.observation_delayed_entry,
                identification.status,
                identification.required_assumptions.clone(),
            )
        }
        .map_err(CausalError::from)?;
        let (scalar, standard_error) = response_scalar_summary(&response);
        let mut estimate = EffectEstimate::new(
            scalar,
            standard_error,
            response.assumptions.clone(),
            OverlapPolicy::ExplicitOverride,
        );
        attach_response_influence(&mut estimate, response_scores.as_ref())?;
        let (treatment, outcome) = response_primary_pair(&query.functional)?;
        let mut diagnostics = identification.diagnostics.clone();
        if identify_cached {
            diagnostics.push(identify_cached_diagnostic());
        }
        let scalar_intervention = scalar.is_finite()
            && matches!(query.functional, ResponseFunctional::InterventionResponse { .. });
        let (refutations, refute_diags) = if scalar_intervention
            && !matches!(self.refute, RefuteSuite::None)
        {
            let ate_query = AverageEffectQuery::binary_ate(treatment, outcome);
            let mut refute_ws = EstimationWorkspace::default();
            if matches!(self.refute, RefuteSuite::Cheap | RefuteSuite::Full) {
                diagnostics.push(Diagnostic::new(
                    "refute.evalue.not_a_contrast",
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Info,
                    "contrast-shaped refuters are not licensed for a plugin intervention level; cheap runs overlap only and full runs overlap plus sampling-stability of the g-comp level. cell.aipw cheap uses the cell-versus-control contrast.",
                ));
            }
            run_plugin_level_refuters(
                data,
                &estimand,
                &ate_query,
                &estimate,
                &mut refute_ws,
                ctx,
                self.refute,
                estimator_id.as_str(),
                &self.custom_validators,
            )?
        } else {
            if !matches!(self.refute, RefuteSuite::None) || !scalar_intervention {
                diagnostics.push(Diagnostic::new(
                    "refute.response.skipped",
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Info,
                    "scalar ATE refuters are not applicable to a function-valued response",
                ));
            }
            (Vec::new(), Vec::new())
        };
        diagnostics.extend(refute_diags);
        if !scalar.is_finite() {
            // `result.effect` exists for every plan, but a curve, vector, Jacobian or
            // identified set has no scalar reading. The NaN is a "not applicable", not a
            // failed computation, and the caller must read `result.response` instead.
            diagnostics.push(Diagnostic::new(
                "estimate.response.no_scalar_summary",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "this response is function-valued or not point identified; the scalar effect summary is not applicable and the result is carried by the response payload",
            ));
        }
        for warning in &response.support.warnings {
            diagnostics.push(warning.clone());
        }

        let estimate_provenance = if query.observation == ObservationSpec::Complete {
            provenance_ids(Arc::clone(&response.provenance_id), Arc::clone(&response.provenance_id))
        } else {
            provenance_ids(
                "estimate.response.observation_adjusted",
                "estimate.response.observation_adjusted",
            )
        };
        Ok(self.finish_identified_execute(IdentifiedExecuteFinish {
            physical,
            identification,
            estimand,
            estimate,
            identifier_id,
            estimator_id,
            treatment,
            outcome,
            identify_cached: false,
            extra_diagnostics: Vec::new(),
            refutations,
            distribution: None,
            mediation: None,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            bootstrap_replicates_ok: None,
            cancelled: false,
            early_stopped: false,
            extras: IdentifiedExecuteExtras {
                estimate_provenance: Some(estimate_provenance),
                diagnostics: Some(diagnostics),
                response: Some(response),
                bootstrap_replicates_requested: Some(None),
                ..Default::default()
            },
        }))
    }

    pub(super) fn execute_codetermined_joint_response(
        &self,
        data: &TabularData,
        query: &ResponseQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let background = self.tiered.as_ref().ok_or_else(|| CausalError::Compile {
            message: "CoDetermined joint execute requires a tiered background".into(),
        })?;
        let identifier =
            physical.logical.record.identifier.as_deref().unwrap_or("generalized.adjustment");
        let estimator = physical.logical.record.estimator.as_deref().unwrap_or("cell.aipw");
        let identifier_id: IdentifierId = identifier.parse()?;
        let estimator_id: EstimatorId = estimator.parse()?;
        if estimator_id != EstimatorId::CellAipw {
            return Err(CausalError::Unsupported {
                message: "CoDetermined joint cells require estimator cell.aipw",
            });
        }
        let (identification, estimand, identify_cached) =
            identification_from_cache_or(ctx, self.identification_cache.as_deref(), || {
                let identification = match self.graph.as_admg() {
                    Some(admg) => {
                        antecedent_identify::identify_tiered_joint_on(background, admg, query)?
                    }
                    None => antecedent_identify::identify_tiered_joint(
                        background,
                        data.schema(),
                        query,
                    )?,
                };
                let estimand =
                    identification.estimands.first().cloned().ok_or(CausalError::Unsupported {
                        message: antecedent_identify::TIERED_JOINT_ADJUSTMENT_REFUSE,
                    })?;
                Ok((identification, estimand))
            })?;
        self.execute_cell_aipw_response(
            data,
            query,
            physical,
            ctx,
            identification,
            estimand,
            identifier_id,
            estimator_id,
            identify_cached,
            started,
        )
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::float_cmp)] // Exact membership in binary intervention levels.
    fn execute_cell_aipw_response(
        &self,
        data: &TabularData,
        query: &ResponseQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
        identification: IdentificationResult,
        estimand: IdentifiedEstimand,
        identifier_id: IdentifierId,
        estimator_id: EstimatorId,
        identify_cached: bool,
        started: Instant,
    ) -> Result<StudyResult, CausalError> {
        let ResponseFunctional::InterventionResponse { outcome, interventions } = &query.functional
        else {
            return Err(CausalError::Unsupported {
                message: "cell.aipw is licensed for discrete joint InterventionResponse",
            });
        };
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
                    message: "cell.aipw requires at most three distinct coordinates",
                });
            }
            treatments.push(*variable);
            requested_arm |= u32::from(level == 1.0) << j;
        }
        if self.continuous_cell.is_some() {
            return Err(CausalError::Unsupported {
                message: antecedent_estimate::POINT_CDE_UNLICENSED,
            });
        }
        if treatments.len() < 2 {
            return Err(CausalError::Unsupported {
                message: "cell.aipw requires at least two binary Set interventions",
            });
        }
        let est = antecedent_estimate::CellSaturatedAipw::new();
        let continuous = None;
        let (fold_ids, design) = match self.shared_batch_design.as_ref() {
            Some(shared) => {
                let ids: Vec<_> = treatments
                    .iter()
                    .copied()
                    .chain(std::iter::once(*outcome))
                    .chain(estimand.adjustment_set.iter().copied())
                    .collect();
                let row_index = match data.complete_case_mask(&ids) {
                    Ok(mask) => mask
                        .iter()
                        .enumerate()
                        .filter_map(|(i, &keep)| {
                            keep.then_some(u32::try_from(i).unwrap_or(u32::MAX))
                        })
                        .collect::<Vec<_>>(),
                    Err(_) => Vec::new(),
                };
                let folds =
                    if row_index.is_empty() { None } else { Some(shared.folds_for(&row_index)?) };
                let design = if row_index.is_empty() {
                    None
                } else {
                    shared.design_for(&estimand.adjustment_set, &row_index)?
                };
                (folds, design)
            }
            None => (None, None),
        };
        let table = est
            .fit_scores_with_assignment(
                data,
                &treatments,
                *outcome,
                &estimand.adjustment_set,
                &query.outcome_functional,
                continuous,
                fold_ids.as_deref(),
                design.as_deref(),
            )
            .map_err(CausalError::from)?;
        let (summary, monotone_rearranged, mut functional_diagnostics) =
            antecedent_estimate::summarize_functional(&table, None)?;
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
            .map(|tau| {
                antecedent_estimate::quantile::quantile_arm(&table, None, tau, requested_arm)
            })
            .transpose()?;
        let (scalar, se) = if let Some(q) = &quantile {
            functional_diagnostics.push(super::super::helpers::quantile_scope_diagnostic());
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
        let cdf = antecedent_estimate::exceedance_cdf_values(&summary, &table);
        let influence = match quantile {
            Some(q) => Arc::from(q.influence),
            None => Arc::from(table.column(column)?),
        };
        let response = CausalResponse {
            estimand: query.functional.clone(),
            identification_status: identification.status,
            estimate: ResponseIdentification::PointIdentified(ResponseValue::Scalar(scalar)),
            uncertainty: ResponseUncertainty::Scalar {
                standard_error: se,
                lower: scalar - 1.96 * se,
                upper: scalar + 1.96 * se,
                level: 0.95,
            },
            support: antecedent_core::SupportReport {
                status: antecedent_core::SupportStatus::Supported,
                query_region: antecedent_core::SupportRegion {
                    minima: Arc::from([]),
                    maxima: Arc::from([]),
                },
                diagnostics: Vec::new(),
                warnings: Vec::new(),
                point_status: None,
            },
            assumptions: identification.required_assumptions.clone(),
            provenance_id: Arc::from("estimate.cell.aipw"),
            horizon_identification: None,
            interaction_structurally_zero: false,
        };
        let score_inference = table.inference(None)?;
        let mut estimate = EffectEstimate::new(
            scalar,
            se,
            identification.required_assumptions.clone(),
            OverlapPolicy::RequireDiagnostics { clip: Some(0.01), trim: None },
        )
        .with_score_table(Some(table))
        .with_influence(if scalar.is_finite() { Some(influence) } else { None })
        .with_exceedance_cdf(cdf)
        .with_joint_covariance(Some(summary.covariance))
        .with_monotone_rearranged(monotone_rearranged);
        estimate.score_inference = Some(score_inference);
        let (treatment, outcome) = response_primary_pair(&query.functional)?;
        let mut diagnostics = identification.diagnostics.clone();
        diagnostics.extend(functional_diagnostics);
        if identify_cached {
            diagnostics.push(identify_cached_diagnostic());
        }
        let ate_query = AverageEffectQuery::binary_ate(treatment, outcome);
        let refute_estimate = cell_contrast_for_refute(&estimate, requested_arm);
        if !matches!(self.refute, RefuteSuite::None) && refute_estimate.ate != estimate.ate {
            diagnostics.push(Diagnostic::new(
                "refute.cell_aipw.contrast",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "cell.aipw cheap/full apply overlap and E-value to the requested-cell minus control-cell contrast, not the intervention level",
            ));
        }
        let mut refute_ws = EstimationWorkspace::default();
        let (refutations, refute_diags) = run_refuters(
            data,
            &estimand,
            &ate_query,
            &refute_estimate,
            &mut refute_ws,
            None,
            ctx,
            self.refute,
            "cell.aipw",
            &self.custom_validators,
            None,
        )?;
        diagnostics.extend(refute_diags);
        Ok(self.finish_identified_execute(IdentifiedExecuteFinish {
            physical,
            identification,
            estimand,
            estimate,
            identifier_id,
            estimator_id,
            treatment,
            outcome,
            identify_cached,
            extra_diagnostics: Vec::new(),
            refutations,
            distribution: None,
            mediation: None,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            bootstrap_replicates_ok: None,
            cancelled: false,
            early_stopped: false,
            extras: IdentifiedExecuteExtras {
                diagnostics: Some(diagnostics),
                response: Some(response),
                bootstrap_replicates_requested: Some(None),
                ..Default::default()
            },
        }))
    }
}

pub(super) fn response_scalar_summary(response: &antecedent_core::CausalResponse) -> (f64, f64) {
    let value = match &response.estimate {
        ResponseIdentification::PointIdentified(value)
        | ResponseIdentification::PartiallyIdentified(value) => value,
        ResponseIdentification::GraphDependent(_) | ResponseIdentification::Unidentified { .. } => {
            return (f64::NAN, f64::NAN);
        }
    };
    let scalar = match value {
        ResponseValue::Scalar(value) => *value,
        ResponseValue::Surface { .. }
        | ResponseValue::Vector(_)
        | ResponseValue::Jacobian { .. }
        | ResponseValue::Envelope(_) => f64::NAN,
    };
    let se = match response.uncertainty {
        ResponseUncertainty::Scalar { standard_error, .. } => standard_error,
        _ => f64::NAN,
    };
    (scalar, se)
}

pub(super) fn response_primary_pair(
    functional: &ResponseFunctional,
) -> Result<(VariableId, VariableId), CausalError> {
    functional.primary_pair().ok_or_else(|| CausalError::Compile {
        message: "response query has no treatment/outcome pair".into(),
    })
}

pub(crate) fn class_aware_response_supported(query: &ResponseQuery) -> bool {
    query.temporal.is_none()
        && !query.functional.treatment_ids().is_empty()
        && query.observation == ObservationSpec::Complete
        && matches!(
            query.functional,
            ResponseFunctional::MeanCurve { .. } | ResponseFunctional::InterventionResponse { .. }
        )
}

pub(crate) fn response_witness_ate(
    query: &ResponseQuery,
) -> Result<AverageEffectQuery, CausalError> {
    if !class_aware_response_supported(query) {
        return Err(CausalError::Unsupported {
            message: "class-aware response identification requires at least one intervention on complete-observation static data",
        });
    }
    let (treatment, outcome) = response_primary_pair(&query.functional)?;
    Ok(AverageEffectQuery::binary_ate(treatment, outcome))
}

impl super::Study {
    /// Cpdag/Pag response via the same generalized-adjustment envelope as ATE.
    pub(super) fn execute_class_response(
        &self,
        data: &TabularData,
        query: &ResponseQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        if !class_aware_response_supported(query) {
            return Err(CausalError::Compile {
                message: "Cpdag/Pag response supports complete-observation MeanCurve and \
                          InterventionResponse only"
                    .into(),
            });
        }
        let started = Instant::now();
        let identifier = physical
            .logical
            .record
            .identifier
            .as_deref()
            .unwrap_or(DEFAULT_PAG_IDENTIFIER_ID.as_str());
        let estimator = physical
            .logical
            .record
            .estimator
            .as_deref()
            .unwrap_or(EstimatorId::default_for_response(&query.functional).as_str());
        let identifier_id: IdentifierId = identifier.parse()?;
        let estimator_id: EstimatorId = estimator.parse()?;
        match self.graph.class() {
            GraphClass::Pag => {
                let pag = physical.static_pag().ok_or_else(|| CausalError::Compile {
                    message: "PAG response execute missing resolved static PAG".into(),
                })?;
                let (envelope, identify_cached) = if let Some(cache) =
                    self.pag_identification_cache.as_deref()
                {
                    (cache.envelope.clone(), true)
                } else {
                    report_identify_compute(ctx);
                    (
                        crate::strategy_table::identify_pag_response(identifier_id, pag, query)?,
                        false,
                    )
                };
                let envelope_diag = super::pag_path::pag_envelope_diagnostic(&envelope);
                self.finish_class_response(
                    data,
                    query,
                    physical,
                    ctx,
                    &envelope,
                    identify_cached,
                    envelope_diag,
                    identifier_id,
                    estimator_id,
                    started,
                )
                .map(|result| {
                    self.attach_certificate(
                        result,
                        crate::Identification::Envelope {
                            envelope,
                            strategy: identifier_id,
                            structure_version: self.graph.version(),
                        },
                    )
                })
            }
            GraphClass::Cpdag => {
                let cpdag = self.graph.as_cpdag().ok_or_else(|| CausalError::Compile {
                    message: "CPDAG response execute missing supplied graph".into(),
                })?;
                let (envelope, identify_cached) =
                    if let Some(cache) = self.cpdag_identification_cache.as_deref() {
                        (cache.envelope.clone(), true)
                    } else {
                        report_identify_compute(ctx);
                        (
                            crate::strategy_table::identify_cpdag_response(
                                identifier_id,
                                cpdag,
                                query,
                            )?,
                            false,
                        )
                    };
                let envelope_diag = super::pag_path::cpdag_envelope_diagnostic(&envelope);
                self.finish_class_response(
                    data,
                    query,
                    physical,
                    ctx,
                    &envelope,
                    identify_cached,
                    envelope_diag,
                    identifier_id,
                    estimator_id,
                    started,
                )
                .map(|result| {
                    self.attach_certificate(
                        result,
                        crate::Identification::CpdagEnvelope {
                            envelope,
                            strategy: identifier_id,
                            structure_version: self.graph.version(),
                        },
                    )
                })
            }
            _ => Err(CausalError::Unsupported {
                message: "class-aware response execute requires a Cpdag or Pag",
            }),
        }
    }

    fn finish_class_response<G>(
        &self,
        data: &TabularData,
        query: &ResponseQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
        envelope: &IdentificationEnvelope<G>,
        identify_cached: bool,
        envelope_diag: Diagnostic,
        identifier_id: IdentifierId,
        estimator_id: EstimatorId,
        started: Instant,
    ) -> Result<StudyResult, CausalError> {
        if matches!(envelope.status, IdentificationStatus::NotIdentified)
            || envelope.identified_weight.0 <= 0.0
        {
            return Err(CausalError::Compile {
                message: "class-aware response not identified (no identified mass in envelope)"
                    .into(),
            });
        }
        let (treatment, outcome) = response_primary_pair(&query.functional)?;
        let data_est = super::super::helpers::apply_scalar_outcome_functional(
            data,
            outcome,
            &query.outcome_functional,
        )?;
        let mut weighted = Vec::new();
        let mut atom_scores = Vec::new();
        let mut total_w = 0.0;
        let mut unestimated_id_mass = 0.0;
        let mut unevaluable_general_id: Option<antecedent_core::CausalResponse> = None;
        let mut primary_estimand: Option<IdentifiedEstimand> = None;
        for case in &envelope.cases {
            if !identification_status_ok_for_case(case.result.status)
                || case.result.estimands.is_empty()
            {
                continue;
            }
            let estimand = case.result.estimands[0].clone();
            if estimand.method_kind().ok() == Some(EstimandMethod::GeneralId)
                || estimand.method_kind().ok() == Some(EstimandMethod::FrontDoor)
            {
                match estimate_general_id_response(data, query, &case.result, &estimand, ctx) {
                    Ok((response, scores)) => {
                        if matches!(response.estimate, ResponseIdentification::Unidentified { .. })
                        {
                            // Required CPT cell was not evaluable: do not mix a number.
                            unestimated_id_mass += case.weight.0;
                            if primary_estimand.is_none() {
                                primary_estimand = Some(estimand);
                            }
                            if unevaluable_general_id.is_none() {
                                unevaluable_general_id = Some(response);
                            }
                            continue;
                        }
                        let w = case.weight.0;
                        total_w += w;
                        if primary_estimand.is_none() {
                            primary_estimand = Some(estimand);
                        }
                        if let Some(scores) = scores {
                            atom_scores.push((w, scores));
                        }
                        weighted.push((w, response));
                    }
                    Err(err)
                        if err.to_string().contains("discrete level cap")
                            || err.to_string().contains("continuous") =>
                    {
                        // Identified by ID, unevaluable with the discrete plug-in.
                        // Keep the mass out of the mix and refuse a subset SE.
                        unestimated_id_mass += case.weight.0;
                    }
                    Err(err) => return Err(err),
                }
                continue;
            }
            let mut response_estimator =
                ContinuousResponseEstimator::new(Arc::clone(&estimand.adjustment_set));
            if let Some(options) = &self.response_options {
                response_estimator.options = options.clone();
            }
            let (response, scores) = if let InferenceMode::Bayesian(cfg) = &self.inference {
                if cfg.prior_artifact.is_some() || cfg.external_compose.is_some() {
                    return Err(CausalError::Unsupported {
                        message: "Bayesian class-aware response prior transfer requires a \
                                  response-specific mapping; only explicit coefficient or \
                                  isotropic priors are supported",
                    });
                }
                let mut bayes = bayesian_gcomp(cfg, ctx);
                bayes.prior.clone_from(&cfg.prior);
                (
                    response_estimator
                        .estimate_bayesian(
                            &data_est,
                            query,
                            case.result.status,
                            case.result.required_assumptions.clone(),
                            &bayes,
                            ctx,
                        )
                        .map_err(CausalError::from)?,
                    None,
                )
            } else {
                response_estimator
                    .estimate_identified_scored(
                        &data_est,
                        query,
                        case.result.status,
                        case.result.required_assumptions.clone(),
                    )
                    .map_err(CausalError::from)?
            };
            let w = case.weight.0;
            total_w += w;
            if primary_estimand.is_none() {
                primary_estimand = Some(estimand);
            }
            if let Some(scores) = scores {
                atom_scores.push((w, scores));
            }
            weighted.push((w, response));
        }
        if !matches!(total_w.partial_cmp(&0.0), Some(std::cmp::Ordering::Greater)) {
            if let Some(response) = unevaluable_general_id {
                // Sole general-ID atom could not evaluate a required cell.
                // Publish the support report; do not invent a number.
                weighted.push((1.0, response));
            } else {
                return Err(CausalError::Compile {
                    message: "class-aware response envelope had no estimable identified cases"
                        .into(),
                });
            }
        }
        let estimand = primary_estimand.ok_or_else(|| CausalError::Compile {
            message: "class-aware response envelope missing estimand".into(),
        })?;
        let mut mixed = mix_class_responses(&weighted, envelope.status)?;
        let identification =
            envelope_to_identification_result_for(envelope, CausalQuery::Response(query.clone()));
        // A multi-completion envelope must not keep a single atom's plugin
        // interval. Mix IFs only when every contributing identified atom
        // supplies one and every completion is identified. Unidentified
        // mass is retained on status; it is not a license to publish the
        // identified-atom SE as the envelope.
        if envelope.cases.len() > 1 {
            mixed.uncertainty = ResponseUncertainty::None;
        }
        let mixed_scores = if atom_scores.len() == weighted.len()
            && envelope.unidentified_weight.0 <= 1e-12
            && unestimated_id_mass <= 1e-12
        {
            mix_response_influences(&atom_scores, data.row_count())
        } else {
            None
        };
        if let Some(scores) = mixed_scores.as_ref() {
            if matches!(mixed.uncertainty, ResponseUncertainty::None) {
                let cols: Vec<&[f64]> = scores.columns.iter().map(Vec::as_slice).collect();
                let cov = antecedent_estimate::joint_influence_covariance(&cols, None)?;
                let options = self.response_options.clone().unwrap_or_default();
                let level = options.confidence_level;
                let value = match &mixed.estimate {
                    ResponseIdentification::PointIdentified(value)
                    | ResponseIdentification::PartiallyIdentified(value) => Some(value),
                    _ => None,
                };
                match value {
                    Some(ResponseValue::Scalar(v)) => {
                        let se = cov.se(0);
                        let z = antecedent_stats::normal_ppf(0.5 + level / 2.0);
                        mixed.uncertainty = ResponseUncertainty::Scalar {
                            standard_error: se,
                            level,
                            lower: v - z * se,
                            upper: v + z * se,
                        };
                    }
                    Some(ResponseValue::Surface { mean, .. }) if mean.len() == cov.dim => {
                        let critical = match options.simultaneous_replicates {
                            Some(reps) => antecedent_estimate::max_t_critical(
                                &cov,
                                level,
                                reps,
                                options.multiplier_seed,
                            )?,
                            None => antecedent_stats::normal_ppf(0.5 + level / 2.0),
                        };
                        let lower = mean
                            .iter()
                            .enumerate()
                            .map(|(j, v)| v - critical * cov.se(j))
                            .collect();
                        let upper = mean
                            .iter()
                            .enumerate()
                            .map(|(j, v)| v + critical * cov.se(j))
                            .collect();
                        mixed.uncertainty = match options.simultaneous_replicates {
                            Some(replicates) => ResponseUncertainty::SimultaneousBand {
                                level,
                                lower,
                                upper,
                                replicates,
                            },
                            None => ResponseUncertainty::PointwiseBand { level, lower, upper },
                        };
                    }
                    _ => {}
                }
            }
        }
        let (scalar, mixed_se) = response_scalar_summary(&mixed);
        let mut estimate = EffectEstimate::new(
            scalar,
            mixed_se,
            mixed.assumptions.clone(),
            OverlapPolicy::ExplicitOverride,
        );
        attach_response_influence(&mut estimate, mixed_scores.as_ref())?;
        let mut diagnostics = vec![envelope_diag];
        if envelope.cases.iter().any(|c| {
            c.result
                .estimands
                .iter()
                .any(|e| e.method_kind().ok() == Some(EstimandMethod::GeneralId))
        }) {
            diagnostics.push(Diagnostic::new(
                "identify.response.general_id",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "at least one MAG/CPDAG completion identified the response by \
                 Shpitser–Pearl ID after generalized adjustment failed",
            ));
        }
        if !estimate.se_analytic.is_finite() {
            diagnostics.push(envelope_se_omits_between_atom_variance());
        }
        if matches!(self.inference, InferenceMode::Bayesian(_)) {
            diagnostics.push(Diagnostic::new(
                "estimate.envelope.response_posterior_not_mixed",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "per-completion posterior draws are not mixed; the curve is the \
                 identified-mass mean; uncertainty is omitted when multiple atoms contribute",
            ));
        }
        if matches!(envelope.status, IdentificationStatus::GraphDependent) {
            diagnostics.push(Diagnostic::new(
                "estimate.envelope.response_identified_mass_mix",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "response payload is the identified-mass mix; unidentified \
                 completions are retained on identification status",
            ));
        }
        if identify_cached {
            diagnostics.push(identify_cached_diagnostic());
        }
        diagnostics.push(Diagnostic::new(
            "refute.response.skipped",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "scalar ATE refuters are not applicable to a function-valued response",
        ));
        if !scalar.is_finite() {
            diagnostics.push(Diagnostic::new(
                "estimate.response.no_scalar_summary",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "this response is function-valued or not point identified; the scalar effect summary is not applicable and the result is carried by the response payload",
            ));
        }
        for warning in &mixed.support.warnings {
            diagnostics.push(warning.clone());
        }

        Ok(self.finish_identified_execute(IdentifiedExecuteFinish {
            physical,
            identification,
            estimand,
            estimate,
            identifier_id,
            estimator_id,
            treatment,
            outcome,
            identify_cached: false,
            extra_diagnostics: Vec::new(),
            refutations: Vec::new(),
            distribution: None,
            mediation: None,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            bootstrap_replicates_ok: None,
            cancelled: false,
            early_stopped: false,
            extras: IdentifiedExecuteExtras {
                estimate_provenance: Some(provenance_ids(
                    Arc::clone(&mixed.provenance_id),
                    Arc::clone(&mixed.provenance_id),
                )),
                diagnostics: Some(diagnostics),
                response: Some(mixed),
                bootstrap_replicates_requested: Some(None),
                ..Default::default()
            },
        }))
    }
}

fn mix_class_responses(
    weighted: &[(f64, antecedent_core::CausalResponse)],
    envelope_status: IdentificationStatus,
) -> Result<antecedent_core::CausalResponse, CausalError> {
    if weighted
        .iter()
        .all(|(_, r)| matches!(r.estimate, ResponseIdentification::Unidentified { .. }))
    {
        return Ok(weighted[0].1.clone());
    }
    let first = &weighted[0].1;
    let items: Vec<(f64, &ResponseValue)> = weighted
        .iter()
        .map(|(w, r)| {
            let value = match &r.estimate {
                ResponseIdentification::PointIdentified(v)
                | ResponseIdentification::PartiallyIdentified(v) => v,
                ResponseIdentification::GraphDependent(_)
                | ResponseIdentification::Unidentified { .. } => {
                    return Err(CausalError::Compile {
                        message: "class-aware response cannot mix unidentified case payloads"
                            .into(),
                    });
                }
            };
            Ok((*w, value))
        })
        .collect::<Result<_, _>>()?;
    let mixed_value = mix_response_values(&items)?;
    let mixed_uncertainty = mix_response_uncertainty(
        &weighted.iter().map(|(w, r)| (*w, &r.uncertainty)).collect::<Vec<_>>(),
    );
    let mixed_support =
        mix_support_reports(&weighted.iter().map(|(_, r)| &r.support).collect::<Vec<_>>());
    let mut assumptions = first.assumptions.clone();
    for (_, response) in weighted.iter().skip(1) {
        for record in &response.assumptions.entries {
            if !assumptions.entries.contains(record) {
                assumptions.push(record.clone());
            }
        }
    }
    // Payload is the identified-mass mix (same object ATE publishes as `ate`).
    // `GraphDependent(vec)` would drop that mix and the Python binder refuses it.
    // Status lives on `identification_status`, not this variant.
    let estimate = if matches!(envelope_status, IdentificationStatus::NonparametricallyIdentified) {
        ResponseIdentification::PointIdentified(mixed_value)
    } else {
        ResponseIdentification::PartiallyIdentified(mixed_value)
    };
    Ok(antecedent_core::CausalResponse {
        estimand: first.estimand.clone(),
        identification_status: envelope_status,
        estimate,
        uncertainty: mixed_uncertainty,
        support: mixed_support,
        assumptions,
        provenance_id: Arc::clone(&first.provenance_id),
        horizon_identification: None,
        interaction_structurally_zero: first.interaction_structurally_zero,
    })
}

fn mix_response_values(items: &[(f64, &ResponseValue)]) -> Result<ResponseValue, CausalError> {
    let Some((_, first)) = items.first() else {
        return Err(CausalError::Compile {
            message: "class-aware response mix requires at least one identified case".into(),
        });
    };
    match first {
        ResponseValue::Scalar(_) => {
            let mut acc = 0.0;
            let mut wsum = 0.0;
            for (w, value) in items {
                let ResponseValue::Scalar(x) = value else {
                    return Err(CausalError::Compile {
                        message: "class-aware response cases returned mixed payload shapes".into(),
                    });
                };
                acc += *w * *x;
                wsum += *w;
            }
            Ok(ResponseValue::Scalar(acc / wsum))
        }
        ResponseValue::Surface { grid, dimension, mean } => {
            let n = mean.len();
            let mut acc = vec![0.0; n];
            let mut wsum = 0.0;
            for (w, value) in items {
                let ResponseValue::Surface { grid: g, dimension: d, mean: m } = value else {
                    return Err(CausalError::Compile {
                        message: "class-aware response cases returned mixed payload shapes".into(),
                    });
                };
                if g.as_ref() != grid.as_ref() || *d != *dimension || m.len() != n {
                    return Err(CausalError::Compile {
                        message: "class-aware response cases returned incompatible grids".into(),
                    });
                }
                for (i, point) in m.iter().enumerate() {
                    acc[i] += *w * *point;
                }
                wsum += *w;
            }
            for point in &mut acc {
                *point /= wsum;
            }
            Ok(ResponseValue::Surface {
                grid: Arc::clone(grid),
                dimension: *dimension,
                mean: Arc::from(acc),
            })
        }
        _ => Err(CausalError::Compile {
            message: "class-aware response only mixes scalar and surface payloads".into(),
        }),
    }
}

fn mix_response_uncertainty(items: &[(f64, &ResponseUncertainty)]) -> ResponseUncertainty {
    // A single atom retains its valid uncertainty. Averaging confidence limits
    // does not produce confidence limits for the weighted mean (the fits share
    // observations), nor quantiles of a distribution over graphs.
    if let [(weight, uncertainty)] = items {
        if weight.is_finite() && *weight > 0.0 {
            return (*uncertainty).clone();
        }
    }
    ResponseUncertainty::None
}

fn mix_support_reports(
    reports: &[&antecedent_core::SupportReport],
) -> antecedent_core::SupportReport {
    let first = reports[0];
    let mut status = first.status;
    let mut warnings = first.warnings.clone();
    for report in reports.iter().skip(1) {
        if support_rank(report.status) > support_rank(status) {
            status = report.status;
        }
        warnings.extend(report.warnings.iter().cloned());
    }
    antecedent_core::SupportReport {
        status,
        query_region: first.query_region.clone(),
        diagnostics: first.diagnostics.clone(),
        warnings,
        point_status: first.point_status.clone(),
    }
}

fn support_rank(status: antecedent_core::SupportStatus) -> u8 {
    match status {
        antecedent_core::SupportStatus::Supported => 0,
        antecedent_core::SupportStatus::WeakOverlap => 1,
        antecedent_core::SupportStatus::Extrapolative => 2,
        antecedent_core::SupportStatus::OutsideEmpiricalSupport => 3,
    }
}

/// E-value / overlap on cell.aipw use the requested cell minus the all-zero control cell.
fn cell_contrast_for_refute(estimate: &EffectEstimate, requested_arm: u32) -> EffectEstimate {
    let Some(table) = estimate.score_table.as_ref() else {
        return estimate.clone();
    };
    let Ok((contrast, _)) = antecedent_estimate::cell_minus_control_contrast(table, requested_arm)
    else {
        return estimate.clone();
    };
    let mut out = estimate.clone();
    out.ate = contrast.value;
    out.se_analytic = contrast.se;
    out
}

fn attach_response_influence(
    estimate: &mut EffectEstimate,
    scores: Option<&antecedent_estimate::ResponseInfluence>,
) -> Result<(), CausalError> {
    let Some(scores) = scores else {
        return Ok(());
    };
    if scores.columns.is_empty() || scores.columns.iter().any(|c| c.len() < 2) {
        return Ok(());
    }
    let cols: Vec<&[f64]> = scores.columns.iter().map(Vec::as_slice).collect();
    match antecedent_estimate::joint_influence_covariance(&cols, None) {
        Ok(cov) => {
            if estimate.se_analytic.is_nan() {
                estimate.se_analytic = cov.se(0);
            }
            estimate.joint_covariance = Some(cov);
        }
        Err(err) if cols.len() > 1 => {
            return Err(CausalError::from(err));
        }
        Err(_) => {}
    }
    estimate.influence = scores.columns.first().map(|c| Arc::from(c.as_slice()));
    Ok(())
}

fn mix_response_influences(
    atoms: &[(f64, antecedent_estimate::ResponseInfluence)],
    full_n: usize,
) -> Option<antecedent_estimate::ResponseInfluence> {
    if atoms.is_empty() {
        return None;
    }
    let n_cols = atoms[0].1.columns.len();
    if full_n < 2 || n_cols == 0 {
        return None;
    }
    let mut aligned = Vec::with_capacity(atoms.len());
    for (_, scores) in atoms {
        let n = scores.row_index.len();
        if n < 2 || scores.columns.len() != n_cols || scores.columns.iter().any(|c| c.len() != n) {
            return None;
        }
        let mut seen = vec![false; full_n];
        for &row in scores.row_index.iter() {
            let slot = seen.get_mut(row as usize)?;
            if *slot {
                return None;
            }
            *slot = true;
        }
        let mut columns = Vec::new();
        for col in &scores.columns {
            let center = col.iter().sum::<f64>() / n as f64;
            let mut full = vec![0.0; full_n];
            for (&row, &v) in scores.row_index.iter().zip(col) {
                full[row as usize] = (v - center) * full_n as f64 / n as f64;
            }
            columns.push(full);
        }
        aligned.push(columns);
    }
    let weights: Vec<f64> = atoms.iter().map(|(w, _)| *w).collect();
    let mut columns = Vec::with_capacity(n_cols);
    for j in 0..n_cols {
        let refs: Vec<&[f64]> = aligned.iter().map(|cols| cols[j].as_slice()).collect();
        columns.push(antecedent_estimate::frozen_weight_mixture_scores(&refs, &weights).ok()?);
    }
    Some(antecedent_estimate::ResponseInfluence {
        columns,
        row_index: (0..full_n).map(u32::try_from).collect::<Result<Vec<_>, _>>().ok()?.into(),
    })
}

#[cfg(test)]
mod uncertainty_tests {
    use super::*;

    #[test]
    fn disjoint_atom_intervals_are_not_averaged_into_a_confidence_interval() {
        let a = ResponseUncertainty::Scalar {
            standard_error: 0.1,
            lower: -0.2,
            upper: 0.2,
            level: 0.95,
        };
        let b = ResponseUncertainty::Scalar {
            standard_error: 0.1,
            lower: 9.8,
            upper: 10.2,
            level: 0.95,
        };
        assert!(matches!(
            mix_response_uncertainty(&[(0.5, &a), (0.5, &b)]),
            ResponseUncertainty::None
        ));
        assert!(matches!(
            mix_response_uncertainty(&[(1.0, &a)]),
            ResponseUncertainty::Scalar { .. }
        ));
    }
}

fn estimate_general_id_response(
    data: &TabularData,
    query: &ResponseQuery,
    identification: &antecedent_identify::IdentificationResult,
    estimand: &IdentifiedEstimand,
    ctx: &ExecutionContext,
) -> Result<(CausalResponse, Option<antecedent_estimate::ResponseInfluence>), CausalError> {
    if !matches!(query.functional, ResponseFunctional::InterventionResponse { .. }) {
        return Err(CausalError::Unsupported {
            message: "general-ID MAG/PAG/DAG response atoms are licensed for \
                      InterventionResponse Set means; MeanCurve stays on the \
                      adjustment envelope",
        });
    }
    if !matches!(query.target_population, antecedent_core::TargetPopulation::AllObserved)
        || !matches!(
            query.outcome_functional,
            antecedent_core::OutcomeFunctional::Mean
                | antecedent_core::OutcomeFunctional::Exceedance(_)
        )
    {
        return Err(CausalError::Unsupported {
            message: "general-ID response estimation supports AllObserved scalar mean or exceedance outcomes",
        });
    }
    let (treatment, outcome) = response_primary_pair(&query.functional)?;
    let data_est = super::super::helpers::apply_scalar_outcome_functional(
        data,
        outcome,
        &query.outcome_functional,
    )?;
    let est = FunctionalEffect::new();
    let prepared = est
        .prepare(
            &data_est,
            estimand,
            &identification.arena,
            identification.required_assumptions.clone(),
            &[treatment, outcome],
        )
        .map_err(CausalError::from)?;
    let _ = ctx;
    let eval =
        prepared.compiled.evaluate(&prepared.arena, &prepared.provider, &EvalContext::default());
    let map_eval = |e: EvalError| {
        CausalError::from(antecedent_estimate::EstimationError::data_msg(e.to_string()))
    };
    let (estimate, support) = match eval {
        Ok(ate) => (
            ResponseIdentification::PointIdentified(ResponseValue::Scalar(ate)),
            support_from_functional_eval(None).map_err(map_eval)?,
        ),
        Err(err) if functional_cell_unevaluable(&err) => (
            ResponseIdentification::Unidentified {
                certificate: Arc::from("estimate.response.general_id.unevaluable_cell"),
            },
            support_from_functional_eval(Some(&err)).map_err(map_eval)?,
        ),
        Err(err) => return Err(map_eval(err)),
    };
    Ok((
        CausalResponse {
            estimand: query.functional.clone(),
            identification_status: identification.status,
            estimate,
            uncertainty: ResponseUncertainty::None,
            support,
            assumptions: identification.required_assumptions.clone(),
            provenance_id: Arc::from("estimate.response.general_id"),
            horizon_identification: None,
            interaction_structurally_zero: false,
        },
        None,
    ))
}

#[cfg(test)]
mod influence_review_tests {
    use super::*;

    #[test]
    fn response_mixture_aligns_original_row_ids() {
        let a = antecedent_estimate::ResponseInfluence {
            columns: vec![vec![-1.0, 1.0]],
            row_index: Arc::from([0, 2]),
        };
        let b = antecedent_estimate::ResponseInfluence {
            columns: vec![vec![-2.0, 2.0]],
            row_index: Arc::from([1, 3]),
        };
        let mixed = mix_response_influences(&[(0.5, a), (0.5, b)], 4).unwrap();
        assert_eq!(mixed.row_index.as_ref(), &[0, 1, 2, 3]);
        assert_eq!(mixed.columns[0], vec![-1.0, -2.0, 1.0, 2.0]);
    }
}
