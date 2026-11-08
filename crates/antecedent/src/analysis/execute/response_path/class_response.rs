//! Class-aware response execution.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    Arc, CausalError, CausalQuery, CausalResponse, ClassEnvelopeRun, ClassResponseAtoms,
    ClassResponseInputs, CompletionFitKey, CompletionFitMemo, ContinuousResponseEstimator,
    Diagnostic, DiagnosticKind, DiagnosticSeverity, EffectEstimate, EstimandMethod, EstimatorId,
    ExecuteRun, ExecutionContext, FrontDoorClaim, GraphClass, IdentificationResult,
    IdentificationStatus, IdentifiedEstimand, IdentifiedExecuteExtras, IdentifiedExecuteFinish,
    InferenceMode, Instant, MixedClassResponse, OverlapPolicy, PhysicalExecutionPlan, RefuteSuite,
    ResponseFunctional, ResponseIdentification, ResponseQuery, ResponseUncertainty, ResponseValue,
    RouteDefaults, Strategy, StudyResult, TableView, TabularData, attach_response_influence,
    bayesian_gcomp, class_aware_response_supported, derivative_identified_set_band, elapsed_ns,
    envelope_se_omits_between_atom_variance, envelope_to_identification_result_for,
    identification_status_ok_for_case, identified_set_interval_diagnostics,
    identify_cached_diagnostic, mix_class_responses, mix_response_influences, provenance_ids,
    report_identify_compute, response_functional_is_derivative, response_identified_value,
    response_primary_pair, response_prior_targets, response_scalar_summary,
    shared_completion_uncertainty, structural_class_response,
};

impl super::super::Study {
    /// Cpdag/Pag response via the same generalized-adjustment envelope as ATE.
    pub(in crate::analysis::execute) fn execute_class_response(
        &self,
        data: &TabularData,
        query: &ResponseQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        if !class_aware_response_supported(query) {
            return Err(CausalError::Compile {
                message: "Cpdag/Pag response supports complete-observation MeanCurve, \
                          InterventionResponse and derivative functionals only"
                    .into(),
            });
        }
        let started = Instant::now();
        let (identifier_id, estimator_id) = physical.binding().resolve(RouteDefaults {
            estimator: EstimatorId::default_for_response(&query.functional),
            ..RouteDefaults::PAG
        })?;
        match self.graph.class() {
            GraphClass::Pag => {
                let pag = physical.static_pag().ok_or_else(|| CausalError::Compile {
                    message: "PAG response execute missing resolved static PAG".into(),
                })?;
                let (envelope, identify_cached) =
                    if let Some((cache, prepared)) = self.reuse_pag_identification(physical) {
                        (Arc::clone(&cache.envelope), prepared)
                    } else {
                        report_identify_compute(ctx);
                        (
                            Arc::new(crate::strategy_table::identify_pag_response_in_context(
                                identifier_id,
                                pag,
                                query,
                                Some(ctx),
                            )?),
                            false,
                        )
                    };
                let envelope_diag = super::super::pag_path::pag_envelope_diagnostic(&envelope);
                self.finish_class_response(
                    data,
                    query,
                    ClassEnvelopeRun {
                        run: ExecuteRun { physical, ctx, started, identify_cached },
                        envelope: &envelope,
                        envelope_diagnostic: envelope_diag,
                    },
                    Strategy::new(identifier_id, estimator_id),
                )
                .map(|result| {
                    self.attach_certificate(
                        result,
                        crate::Identification::Envelope {
                            envelope: Arc::unwrap_or_clone(envelope),
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
                    if let Some((cache, prepared)) = self.reuse_cpdag_identification(physical) {
                        (Arc::clone(&cache.envelope), prepared)
                    } else {
                        report_identify_compute(ctx);
                        (
                            Arc::new(crate::strategy_table::identify_cpdag_response_in_context(
                                identifier_id,
                                cpdag,
                                query,
                                Some(ctx),
                            )?),
                            false,
                        )
                    };
                let envelope_diag = super::super::pag_path::cpdag_envelope_diagnostic(&envelope);
                self.finish_class_response(
                    data,
                    query,
                    ClassEnvelopeRun {
                        run: ExecuteRun { physical, ctx, started, identify_cached },
                        envelope: &envelope,
                        envelope_diagnostic: envelope_diag,
                    },
                    Strategy::new(identifier_id, estimator_id),
                )
                .map(|result| {
                    self.attach_certificate(
                        result,
                        crate::Identification::CpdagEnvelope {
                            envelope: Arc::unwrap_or_clone(envelope),
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
        run: ClassEnvelopeRun<'_, G>,
        strategy: Strategy,
    ) -> Result<StudyResult, CausalError> {
        let Strategy { identifier: identifier_id, estimator: estimator_id } = strategy;
        let ClassEnvelopeRun {
            run: ExecuteRun { physical, ctx, started, identify_cached },
            envelope,
            envelope_diagnostic: envelope_diag,
        } = run;
        if matches!(envelope.status, IdentificationStatus::NotIdentified)
            || envelope.identified_weight.0 <= 0.0
        {
            return Err(CausalError::not_identified(
                envelope.status,
                envelope.truncated_completions > 0,
                "class-aware response not identified (no identified mass in envelope)",
            ));
        }
        let (treatment, outcome) = response_primary_pair(&query.functional)?;
        let data_est = super::super::super::helpers::apply_scalar_outcome_functional(
            data,
            outcome,
            &query.outcome_functional,
        )?;
        let inputs = ClassResponseInputs { data, data_est: &data_est, query, ctx };
        let atoms = self.evaluate_class_response_atoms(&inputs, envelope)?;
        let mixed = self.mix_class_response(data, envelope, query, atoms)?;
        let mut diagnostics =
            self.class_response_diagnostics(envelope, envelope_diag, &mixed, identify_cached);
        // A class-aware intervention level runs its per-completion refuter state
        // (unsupported); a curve keeps the skipped note.
        let mut refutations = Vec::new();
        let mut predictive_checks = Vec::new();
        if !matches!(self.settings.refute, RefuteSuite::None)
            && response_functional_is_derivative(&query.functional)
        {
            let (reports, checks, notes) = self.refute_class_derivative_atoms(
                &data_est,
                query,
                envelope,
                &mixed.weighted,
                ctx,
            )?;
            refutations = reports;
            predictive_checks = checks;
            diagnostics.retain(|diagnostic| diagnostic.code.as_ref() != "refute.response.skipped");
            diagnostics.extend(notes);
        } else if !matches!(self.settings.refute, RefuteSuite::None)
            && super::super::curve_refutation::is_mean_curve(&query.functional)
        {
            let (reports, notes) =
                self.refute_class_curve_atoms(&data_est, query, envelope, &mixed.weighted, ctx)?;
            refutations = reports;
            diagnostics.retain(|diagnostic| diagnostic.code.as_ref() != "refute.response.skipped");
            diagnostics.extend(notes);
        } else if !matches!(self.settings.refute, RefuteSuite::None)
            && matches!(query.functional, ResponseFunctional::InterventionResponse { .. })
        {
            let (reports, notes) =
                super::super::class_response_refutation::refute_class_intervention_atoms(
                    self,
                    (data, query),
                    envelope,
                    &mixed.weighted,
                    (treatment, outcome),
                    estimator_id,
                    ctx,
                )?;
            refutations = reports;
            diagnostics.retain(|diagnostic| diagnostic.code.as_ref() != "refute.response.skipped");
            diagnostics.extend(notes);
        }
        let MixedClassResponse {
            identification,
            estimand,
            estimate,
            response: mixed,
            structural_response,
            ..
        } = mixed;

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
                    Arc::clone(&mixed.provenance_id),
                    Arc::clone(&mixed.provenance_id),
                )),
                diagnostics: Some(diagnostics),
                response: Some(mixed),
                structural_response: Some(structural_response),
                predictive_checks,
                bootstrap_replicates_requested: Some(None),
                ..Default::default()
            }),
        ))
    }

    /// Estimates each identified completion of the class envelope, memoising shared fits.
    fn evaluate_class_response_atoms<G>(
        &self,
        inputs: &ClassResponseInputs<'_>,
        envelope: &antecedent_identify::IdentificationEnvelope<G>,
    ) -> Result<ClassResponseAtoms, CausalError> {
        let structural_atoms = envelope
            .cases
            .iter()
            .enumerate()
            .map(|(index, case)| crate::result::StructuralResponseAtom {
                posterior: None,
                response: None,
                graph_key: u64::try_from(index).unwrap_or(u64::MAX),
                weight: case.weight.0,
                status: case.result.status,
                value: None,
            })
            .collect::<Vec<_>>();
        let mut atoms = ClassResponseAtoms {
            weighted: Vec::new(),
            structural_atoms,
            atom_scores: Vec::new(),
            total_w: 0.0,
            unestimated_id_mass: 0.0,
            unevaluable_general_id: None,
            primary_estimand: None,
        };
        let mut completion_fits: CompletionFitMemo<(
            CausalResponse,
            Option<antecedent_estimate::ResponseInfluence>,
        )> = CompletionFitMemo::default();
        for (case_index, case) in envelope.cases.iter().enumerate() {
            if !identification_status_ok_for_case(case.result.status)
                || case.result.estimands.is_empty()
            {
                continue;
            }
            let estimand = case.result.estimands[0].clone();
            if response_functional_is_derivative(&inputs.query.functional)
                && estimand.method_kind().ok() == Some(EstimandMethod::FrontDoor)
            {
                // Identified through a front door: the additive-outcome derivative, with its
                // own whole-procedure interval (no row scores to mix).
                let response = self.estimate_front_door_derivative_response(
                    inputs.data_est,
                    inputs.query,
                    FrontDoorClaim {
                        estimand: &estimand,
                        status: case.result.status,
                        assumptions: case.result.required_assumptions.clone(),
                    },
                    1,
                    inputs.ctx,
                )?;
                let w = case.weight.0;
                atoms.total_w += w;
                if atoms.primary_estimand.is_none() {
                    atoms.primary_estimand = Some(estimand);
                }
                atoms.structural_atoms[case_index].value = response_identified_value(&response);
                atoms.weighted.push((u64::try_from(case_index).unwrap_or(u64::MAX), w, response));
                continue;
            }
            if estimand.method_kind().ok() == Some(EstimandMethod::GeneralId)
                || estimand.method_kind().ok() == Some(EstimandMethod::FrontDoor)
            {
                atoms.add_general_id_case(
                    inputs,
                    case_index,
                    &case.result,
                    case.weight.0,
                    estimand,
                )?;
                continue;
            }
            // Completions that share an adjustment set, status and assumptions
            // run one fit: every other input is fixed for this execution.
            let fit_key = CompletionFitKey::new(
                &estimand.adjustment_set,
                case.result.status,
                &case.result.required_assumptions,
            );
            let (response, scores) = if let Some((response, scores)) = completion_fits.get(&fit_key)
            {
                (response.clone(), scores.clone())
            } else {
                let fitted = self.fit_class_completion(inputs, &case.result, &estimand)?;
                completion_fits.insert(fit_key, (fitted.0.clone(), fitted.1.clone()));
                fitted
            };
            let w = case.weight.0;
            atoms.total_w += w;
            if atoms.primary_estimand.is_none() {
                atoms.primary_estimand = Some(estimand);
            }
            if let Some(scores) = scores {
                atoms.atom_scores.push((w, scores));
            }
            atoms.structural_atoms[case_index].value = response_identified_value(&response);
            atoms.weighted.push((u64::try_from(case_index).unwrap_or(u64::MAX), w, response));
        }
        Ok(atoms)
    }

    /// Fits one completion's response, frequentist or Bayesian, before the shared-fit memo stores it.
    fn fit_class_completion(
        &self,
        inputs: &ClassResponseInputs<'_>,
        result: &IdentificationResult,
        estimand: &IdentifiedEstimand,
    ) -> Result<(CausalResponse, Option<antecedent_estimate::ResponseInfluence>), CausalError> {
        let mut response_estimator =
            ContinuousResponseEstimator::new(Arc::clone(&estimand.adjustment_set));
        if let Some(options) = &self.response_options {
            response_estimator.options = options.clone();
        }
        let (response, scores) = if let InferenceMode::Bayesian(cfg) = &self.settings.inference {
            if cfg.has_transferred_source() && cfg.prior_mapping.is_none() && cfg.prior.is_none() {
                return Err(CausalError::Unsupported {
                    message: "Bayesian class-aware response prior transfer requires a \
                              response-specific mapping; mappings are not a class prior \
                              and must not fall back to isotropic coefficients",
                });
            }
            let mut bayes = bayesian_gcomp(cfg, inputs.ctx);
            if cfg.has_transferred_source() {
                let (outcome, treatments) = response_prior_targets(&inputs.query.functional)?;
                let prep = response_estimator
                    .prepare_linear_response_problem(inputs.data_est, outcome, &treatments)
                    .map_err(CausalError::from)?;
                let (resolved, _) = crate::inference::resolve_bayesian_prior_with_conflict(
                    cfg,
                    &prep,
                    Some(inputs.ctx),
                )?;
                bayes.prior = resolved;
            } else {
                bayes.prior.clone_from(&cfg.prior);
            }
            (
                response_estimator
                    .estimate_bayesian(
                        inputs.data_est,
                        inputs.query,
                        result.status,
                        result.required_assumptions.clone(),
                        &bayes,
                        inputs.ctx,
                    )
                    .map_err(CausalError::from)?,
                None,
            )
        } else {
            let (response, scores) = response_estimator
                .estimate_identified_scored(
                    inputs.data_est,
                    inputs.query,
                    result.status,
                    result.required_assumptions.clone(),
                )
                .map_err(CausalError::from)?;
            // The derivative routes publish their own interval but no row scores; the class
            // mixture needs the scores to place completions on shared rows.
            let scores = if scores.is_none()
                && response_functional_is_derivative(&inputs.query.functional)
            {
                response_estimator
                    .derivative_influence(inputs.data_est, inputs.query)
                    .map_err(CausalError::from)?
            } else {
                scores
            };
            (response, scores)
        };
        Ok((response, scores))
    }

    /// Mixes the completion responses into the envelope response, estimate and mass accounting.
    fn mix_class_response<G>(
        &self,
        data: &TabularData,
        envelope: &antecedent_identify::IdentificationEnvelope<G>,
        query: &ResponseQuery,
        atoms: ClassResponseAtoms,
    ) -> Result<MixedClassResponse, CausalError> {
        let mut atoms = atoms;
        atoms.require_estimable(query)?;
        let ClassResponseAtoms {
            weighted,
            structural_atoms,
            atom_scores,
            total_w,
            unestimated_id_mass,
            unevaluable_general_id: _,
            primary_estimand,
        } = atoms;
        let estimand = primary_estimand.ok_or_else(|| CausalError::Compile {
            message: "class-aware response envelope missing estimand".into(),
        })?;
        let mut mixed = mix_class_responses(&weighted, envelope.status)?;
        let structural_response = structural_class_response(
            envelope,
            structural_atoms,
            &weighted,
            total_w,
            unestimated_id_mass,
        );
        let identification =
            envelope_to_identification_result_for(envelope, CausalQuery::Response(query.clone()));
        // A multi-completion envelope must not keep a single atom's plugin
        // interval. Mix IFs only when every contributing identified atom
        // supplies one and every completion is identified. Unidentified
        // mass is retained on status; it is not support to publish the
        // identified-atom SE as the envelope.
        if envelope.cases.len() > 1 {
            mixed.uncertainty = ResponseUncertainty::None;
        }
        // Bayesian completions carry no influence scores to mix. When every
        // completion is identified and each returns the same response (they
        // share the adjustment set, so their posteriors are one posterior),
        // any mixture of them is that posterior: its band is the class band.
        if self.settings.inference.is_bayesian()
            && envelope.unidentified_weight.0 <= 1e-12
            && unestimated_id_mass <= 1e-12
            && let Some(shared) = shared_completion_uncertainty(&weighted)
        {
            mixed.uncertainty = shared;
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
            self.attach_class_response_band(&mut mixed, scores)?;
        }
        // Completions of a derivative that disagree publish the identified set; the Imbens-Manski
        // interval of the set, from the completions' joint row influence, is its sampling band.
        let mut identified_set_banded = false;
        let mut structural_response = structural_response;
        if response_functional_is_derivative(&query.functional)
            && matches!(
                &mixed.estimate,
                ResponseIdentification::PartiallyIdentified(ResponseValue::Envelope(_))
            )
            && envelope.unidentified_weight.0 <= 1e-12
            && unestimated_id_mass <= 1e-12
        {
            let level = self.response_options.clone().unwrap_or_default().confidence_level;
            if let Some((band, interval)) =
                derivative_identified_set_band(&weighted, &atom_scores, data.row_count(), level)
            {
                mixed.uncertainty = band;
                structural_response.identified_set_interval = interval;
                identified_set_banded = true;
            }
        }
        let (scalar, mixed_se) = response_scalar_summary(&mixed);
        let mut estimate = EffectEstimate::new(
            scalar,
            mixed_se,
            mixed.assumptions.clone(),
            OverlapPolicy::ExplicitOverride,
        );
        // Completions that disagree publish the completion identified set, not a
        // mass-weighted summary, so the frozen-weight joint-IF SE has no reported
        // point to attach to. Attaching it anyway left `se_analytic` describing a
        // mixture the result never states.
        let reports_mixed_point = matches!(
            &mixed.estimate,
            ResponseIdentification::PointIdentified(
                ResponseValue::Scalar(_) | ResponseValue::Surface { .. }
            ) | ResponseIdentification::PartiallyIdentified(
                ResponseValue::Scalar(_) | ResponseValue::Surface { .. }
            )
        );
        let identified_set_unbanded = !reports_mixed_point
            && !identified_set_banded
            && matches!(
                &mixed.estimate,
                ResponseIdentification::PartiallyIdentified(ResponseValue::Envelope(_))
            );
        if reports_mixed_point {
            attach_response_influence(&mut estimate, mixed_scores.as_ref())?;
        }
        Ok(MixedClassResponse {
            identification,
            estimand,
            estimate,
            response: mixed,
            structural_response,
            identified_set_unbanded,
            weighted,
        })
    }

    /// Attaches the joint-influence band to a mixed response that published none.
    fn attach_class_response_band(
        &self,
        mixed: &mut CausalResponse,
        scores: &antecedent_estimate::ResponseInfluence,
    ) -> Result<(), CausalError> {
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
                        interpretation: antecedent_core::IntervalInterpretation::Confidence,
                        draws: None,
                    };
                }
                Some(
                    ResponseValue::Vector(coordinates)
                    | ResponseValue::Jacobian { values: coordinates, .. },
                ) if coordinates.len() == cov.dim => {
                    // A derivative vector or Jacobian: one coordinate per column, pointwise.
                    let z = antecedent_stats::normal_ppf(0.5 + level / 2.0);
                    let lower =
                        coordinates.iter().enumerate().map(|(j, v)| v - z * cov.se(j)).collect();
                    let upper =
                        coordinates.iter().enumerate().map(|(j, v)| v + z * cov.se(j)).collect();
                    mixed.uncertainty = ResponseUncertainty::PointwiseBand {
                        level,
                        lower,
                        upper,
                        interpretation: antecedent_core::IntervalInterpretation::Confidence,
                        draws: None,
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
                    let lower =
                        mean.iter().enumerate().map(|(j, v)| v - critical * cov.se(j)).collect();
                    let upper =
                        mean.iter().enumerate().map(|(j, v)| v + critical * cov.se(j)).collect();
                    mixed.uncertainty = match options.simultaneous_replicates {
                        Some(replicates) => ResponseUncertainty::SimultaneousBand {
                            level,
                            lower,
                            upper,
                            replicates,
                            interpretation: antecedent_core::IntervalInterpretation::Confidence,
                        },
                        None => ResponseUncertainty::PointwiseBand {
                            level,
                            lower,
                            upper,
                            interpretation: antecedent_core::IntervalInterpretation::Confidence,
                            draws: None,
                        },
                    };
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Disclosure diagnostics of a mixed class-aware response.
    fn class_response_diagnostics<G>(
        &self,
        envelope: &antecedent_identify::IdentificationEnvelope<G>,
        envelope_diag: Diagnostic,
        mixed_class: &MixedClassResponse,
        identify_cached: bool,
    ) -> Vec<Diagnostic> {
        let MixedClassResponse { response: mixed, estimate, weighted, .. } = mixed_class;
        let identified_set_unbanded = mixed_class.identified_set_unbanded;
        let (scalar, _) = response_scalar_summary(mixed);
        let mut diagnostics = vec![envelope_diag];
        if identified_set_unbanded {
            diagnostics.push(Diagnostic::new(
                "estimate.envelope.response_identified_set_unbanded",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "completions disagree, so the response is the completion identified set \
                 (pointwise lower/upper over identified completions); no sampling band and no \
                 scalar SE are published for it. Per-completion values and enumeration weights \
                 are in structural_response; a frozen-weight mixture is not reported",
            ));
        }
        if let Some(interval) = &mixed_class.structural_response.identified_set_interval {
            diagnostics.extend(identified_set_interval_diagnostics(interval));
        }
        if response_functional_is_derivative(&mixed.estimand)
            && !matches!(mixed.uncertainty, ResponseUncertainty::None)
            && weighted.iter().all(|(_, _, r)| matches!(r.uncertainty, ResponseUncertainty::None))
        {
            diagnostics.push(Diagnostic::new(
                "estimate.response.derivative_class_band_not_measured",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "the band is formed from the completions' joint row influence (the additive-GAM \
                 coefficient sandwich for a Jacobian or directional derivative, the robust \
                 bias-corrected local fit for a point derivative); no coverage record measures it \
                 and the cell is unsupported",
            ));
        }
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
        if !estimate.se_analytic().is_finite() && !identified_set_unbanded {
            diagnostics.push(envelope_se_omits_between_atom_variance());
        }
        // Disclose only when the envelope actually dropped per-completion
        // posterior uncertainty; a single contributing completion keeps its own
        // band and nothing is omitted.
        let atom_uncertainty_dropped = matches!(mixed.uncertainty, ResponseUncertainty::None)
            && weighted.iter().any(|(_, _, r)| !matches!(r.uncertainty, ResponseUncertainty::None));
        if self.settings.inference.is_bayesian() && atom_uncertainty_dropped {
            diagnostics.push(Diagnostic::new(
                "estimate.envelope.response_posterior_not_mixed",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "per-completion posterior draws are not assigned structural probabilities; \
                 completion-enumeration responses remain an identified set and uncertainty is \
                 omitted when multiple atoms contribute",
            ));
        }
        if matches!(envelope.status, IdentificationStatus::GraphDependent) {
            diagnostics.push(Diagnostic::new(
                "estimate.envelope.response_graph_dependent",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "response payload retains graph-conditional atom values; completion enumeration \
                 is not treated as a posterior distribution and unidentified mass remains explicit",
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
        diagnostics
    }
}
