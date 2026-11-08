//! Temporal derivative and ADMG response execution.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    CausalError, CausalResponse, DagDerivativeFit, DerivativeResponseInputs, Diagnostic,
    DiagnosticKind, DiagnosticSeverity, EffectEstimate, EstimandMethod, EstimatorId, ExecuteRun,
    ExecutionContext, IdentificationResult, IdentifiedClaim, IdentifiedEstimand,
    IdentifiedExecuteFinish, IdentifierId, InferenceMode, Instant, OverlapPolicy,
    PhysicalExecutionPlan, RefuteSuite, ResponseFunctional, ResponseQuery, Strategy, StudyResult,
    TableView, TemporalIndexer, TimeSeriesData, VariableId, aggregate_temporal_horizon_evidence,
    combine_horizon_derivatives, elapsed_ns, horizon_note, report_identify_compute,
    temporal_dag_derivative_response, unit_contrast_query,
};

/// One horizon's immutable identification and whether preparation supplied it.
struct AdmgDerivativeClaim {
    horizon: u32,
    identification: IdentificationResult,
    estimand: IdentifiedEstimand,
    indexer: TemporalIndexer,
    cached: bool,
}

impl AdmgDerivativeClaim {
    /// A derivative's adjustment refuter uses the unit contrast at this same horizon.
    fn pulse_witness(
        &self,
        input: AdmgDerivativeInputs<'_>,
    ) -> super::super::temporal_response_refutation::PulseWitnessSite {
        super::super::temporal_response_refutation::PulseWitnessSite {
            key: u64::from(self.horizon),
            horizon: self.horizon,
            weight: 1.0,
            witness: unit_contrast_query(
                input.temporal,
                input.reported,
                self.horizon,
                input.query.target_population.clone(),
            ),
            estimand: self.estimand.clone(),
            indexer: self.indexer.clone(),
            status: self.identification.status,
            assumptions: self.identification.required_assumptions.clone(),
        }
    }
}

/// Native fits and refuter state in the requested horizon order.
struct AdmgDerivativeFits {
    responses: Vec<CausalResponse>,
    witness_sites: Vec<super::super::temporal_response_refutation::PulseWitnessSite>,
    horizon_reports: Vec<antecedent_validate::RefutationReport>,
    horizon_notes: Vec<Diagnostic>,
}

/// Shared source/target and reporting frame for native ADMG derivative phases.
#[derive(Clone, Copy)]
struct AdmgDerivativeInputs<'a> {
    data: &'a TimeSeriesData,
    template: &'a antecedent_graph::TemporalAdmg,
    query: &'a ResponseQuery,
    temporal: &'a antecedent_core::TemporalResponseSpec,
    reported: (VariableId, VariableId),
    run: ExecuteRun<'a>,
}

/// A front-door derivative of one horizon of a temporal ADMG with the window table, the
/// window-coordinate query and the resampling block it was estimated on.
struct TemporalFrontDoorFit {
    response: CausalResponse,
    table: crate::analysis::temporal_admg_contrast::AdmittedWindowTable,
    window_query: ResponseQuery,
    block: usize,
}

impl super::super::Study {
    /// A temporal derivative response: the local slope of the linear unfolded
    /// fit at the one requested horizon, with its whole-procedure band
    /// ([`temporal_dag_derivative_response`]).
    pub(super) fn execute_temporal_derivative_response(
        &self,
        inputs: DerivativeResponseInputs<'_>,
        claim: IdentifiedClaim,
        run: ExecuteRun<'_>,
    ) -> Result<StudyResult, CausalError> {
        let DerivativeResponseInputs { data, query, temporal, indexer, treatment, outcome } =
            inputs;
        let IdentifiedClaim { identification, estimand } = claim;
        let ExecuteRun { physical, ctx, started, identify_cached } = run;
        if temporal.horizons.len() != 1 {
            return Err(CausalError::Unsupported {
                message: "a temporal derivative is the local slope at one horizon",
            });
        }
        if let InferenceMode::Bayesian(cfg) = &self.settings.inference
            && cfg.has_transfer()
        {
            return Err(CausalError::Unsupported {
                message: "temporal derivative uses the isotropic conjugate prior scale; a mapped coefficient prior is not supported",
            });
        }
        let probe = unit_contrast_query(
            temporal,
            (treatment, outcome),
            temporal.horizons[0],
            query.target_population.clone(),
        );
        let response = temporal_dag_derivative_response(
            data,
            query,
            DagDerivativeFit {
                estimand: &estimand,
                indexer,
                identification: &identification,
                offsets: (probe.try_treatment_offset()?, probe.outcome_offset()?),
            },
            (&self.settings.inference, self.settings.bootstrap_replicates),
            ctx,
        )?;
        let scalar = super::super::response_path::response_scalar_summary(&response).0;
        let estimator_id = EstimatorId::temporal_response_for(&self.settings.inference);
        let estimate = EffectEstimate::new(
            scalar,
            f64::NAN,
            identification.required_assumptions.clone(),
            OverlapPolicy::ExplicitOverride,
        );
        let mut diagnostics = vec![Diagnostic::new(
            "estimate.temporal.linear_derivative",
            DiagnosticKind::Execution,
            DiagnosticSeverity::Info,
            "published value is the slope of the linear unfolded temporal fit; its band refits \
             every outcome regression (frequentist shared circular blocks, or Bayesian joint \
             slope-posterior draws)",
        )];
        diagnostics.extend(response.support.warnings.iter().cloned());
        // A derivative's cheap/full suite is its refuter state on the horizon's Pulse
        // witness (unsupported).
        let (refutations, predictive_checks) = if matches!(self.settings.refute, RefuteSuite::None)
        {
            (Vec::new(), Vec::new())
        } else {
            let site = super::super::temporal_response_refutation::PulseWitnessSite {
                key: u64::from(temporal.horizons[0]),
                horizon: temporal.horizons[0],
                weight: 1.0,
                witness: probe.clone(),
                estimand: estimand.clone(),
                indexer: indexer.clone(),
                status: identification.status,
                assumptions: identification.required_assumptions.clone(),
            };
            let (reports, checks, notes) =
                super::super::temporal_response_refutation::refute_temporal_derivative(
                    self,
                    data,
                    query,
                    &response,
                    &[site],
                    ctx,
                )?;
            diagnostics.extend(notes);
            (reports, checks)
        };
        let mut result = self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand,
                estimate,
                Strategy::new(IdentifierId::TemporalBackdoorUnfolded, estimator_id),
                treatment,
                outcome,
            )
            .identify_cached(identify_cached)
            .refutations(refutations)
            .wall_time_ns(elapsed_ns(started)),
        );
        result.payload = crate::result::ResultPayload::Response(response);
        result.posterior = None;
        result.predictive_checks = predictive_checks;
        result.rebind_interval(self.settings.inference.is_bayesian());
        result.diagnostics.extend(diagnostics);
        Ok(result)
    }

    /// A derivative at each horizon of a temporal ADMG: per horizon, one adjustment set (or a
    /// front-door set) of the unrolled window of the template (a cause that precedes the window
    /// is a latent common cause of its in-window effects), and the slope of the lag-aligned
    /// regression on it, with the whole-procedure band the `TemporalDag` derivative publishes.
    /// Several horizons publish one value per horizon (pointwise bands, horizons not mixed).
    ///
    /// # Errors
    ///
    /// A horizon whose derivative has neither an adjustment set nor a front-door set on its
    /// window (a hedge is refused with the variables and periods that carry it), a custom
    /// validator, or a Jacobian or directional derivative over several horizons.
    pub(in crate::analysis::execute) fn execute_temporal_admg_derivative(
        &self,
        data: &TimeSeriesData,
        query: &ResponseQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        super::super::series_weights::refuse_unread_series_weights(
            data,
            "the temporal ADMG derivative",
        )?;
        let started = Instant::now();
        let Some(temporal) = query.temporal.as_ref() else {
            return Err(CausalError::Compile {
                message: "temporal response route requires TemporalResponseSpec".into(),
            });
        };
        let crate::accepted::GraphRef::TemporalAdmg(template) = self.graph.graph_ref() else {
            return Err(CausalError::Compile {
                message: "a temporal ADMG derivative requires a supplied TemporalAdmg".into(),
            });
        };
        if !self.declared.custom_validators.is_empty() {
            return Err(crate::unsupported_reason!(
                "validators_not_applicable",
                "a custom validator refutes one scalar average effect and none applies to a \
                 derivative identified on the unrolled window of a temporal ADMG. Next attempt: \
                 drop the validator, or analyse the derivative on a TemporalDag when the lag \
                 structure has no latent confounding"
            ));
        }
        let (treatment, outcome) =
            super::super::response_path::response_primary_pair(&query.functional)?;
        let variable_count = u32::try_from(data.schema().len()).map_err(|_| {
            CausalError::Compile { message: "series schema exceeds the variable id range".into() }
        })?;
        if temporal.horizons.len() > 1
            && !matches!(
                query.functional,
                ResponseFunctional::PointDerivative { .. }
                    | ResponseFunctional::AverageDerivative { .. }
            )
        {
            return Err(CausalError::Unsupported {
                message: "a Jacobian or directional derivative on a temporal ADMG is the local \
                          slope at one horizon; name one horizon, or a point or average \
                          derivative over several",
            });
        }
        let claims = self.identify_admg_derivative_horizons(
            template,
            query,
            &temporal.horizons,
            variable_count,
            physical,
            ctx,
        )?;
        let identify_cached = claims.iter().all(|claim| claim.cached);
        let run = ExecuteRun { physical, ctx, started, identify_cached };
        let input = AdmgDerivativeInputs {
            data,
            template,
            query,
            temporal,
            reported: (treatment, outcome),
            run,
        };
        if let [claim] = claims.as_slice() {
            return self.execute_single_admg_derivative(input, claim);
        }
        self.execute_multi_admg_derivative(input, &claims)
    }

    /// Retain one proof per requested horizon before any response fitting.
    fn identify_admg_derivative_horizons(
        &self,
        template: &antecedent_graph::TemporalAdmg,
        query: &ResponseQuery,
        horizons: &[u32],
        variable_count: u32,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<Vec<AdmgDerivativeClaim>, CausalError> {
        // The products compile retained or the prepared handle froze, else fresh identifications.
        let cache = self.reuse_temporal_identification(physical);
        let mut claims = Vec::with_capacity(horizons.len());
        let mut budget =
            antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
        for &horizon in horizons {
            let reused = cache.as_ref().and_then(|(cache, prepared)| {
                cache.get(horizon).map(|entry| {
                    (
                        entry.identification.clone(),
                        entry.estimand.clone(),
                        entry.indexer.clone(),
                        *prepared,
                    )
                })
            });
            let (identification, estimand, indexer, cached) = if let Some(found) = reused {
                found
            } else {
                report_identify_compute(ctx);
                let single =
                    crate::analysis::temporal_admg_contrast::single_horizon_query(query, horizon)?;
                let found =
                    crate::analysis::temporal_admg_contrast::identify_window_derivative_within(
                        template,
                        &single,
                        variable_count,
                        &mut budget,
                        ctx,
                    )?;
                (found.identification, found.estimand, found.indexer, false)
            };
            claims.push(AdmgDerivativeClaim { horizon, identification, estimand, indexer, cached });
        }
        Ok(claims)
    }

    /// One horizon keeps its native front-door or adjustment response and refuter.
    fn execute_single_admg_derivative(
        &self,
        input: AdmgDerivativeInputs<'_>,
        claim: &AdmgDerivativeClaim,
    ) -> Result<StudyResult, CausalError> {
        let AdmgDerivativeInputs {
            data,
            template,
            query,
            temporal,
            reported: (treatment, outcome),
            run,
        } = input;
        let ctx = run.ctx;
        let AdmgDerivativeClaim { identification, estimand, indexer, .. } = claim;
        if estimand.method_kind().ok() == Some(EstimandMethod::FrontDoor) {
            let fit = self.temporal_admg_front_door_response(
                (data, template, query),
                (identification, estimand),
                ctx,
            )?;
            let refuted = if self.settings.refute == RefuteSuite::None {
                (Vec::new(), Vec::new(), Vec::new())
            } else {
                let (reports, notes) =
                    self.refute_temporal_admg_front_door(&fit, (identification, estimand), ctx)?;
                (reports, Vec::new(), notes)
            };
            let mut result = self.finish_temporal_admg_response(
                fit.response,
                IdentifiedClaim {
                    identification: identification.clone(),
                    estimand: estimand.clone(),
                },
                (treatment, outcome),
                run,
                refuted,
            );
            // No evidence row names the front-door derivative route: the cell's support belongs to
            // the adjustment-identified plug-ins, so a front-door run reports its status as refused.
            result.support_status = Some(crate::support::CellStatus::NotSupported);
            return Ok(result);
        }
        self.execute_temporal_derivative_response(
            DerivativeResponseInputs { data, query, temporal, indexer, treatment, outcome },
            IdentifiedClaim { identification: identification.clone(), estimand: estimand.clone() },
            run,
        )
    }

    /// Fit horizons separately, then publish their joint response without mixing horizons.
    fn execute_multi_admg_derivative(
        &self,
        input: AdmgDerivativeInputs<'_>,
        claims: &[AdmgDerivativeClaim],
    ) -> Result<StudyResult, CausalError> {
        let AdmgDerivativeInputs { data, query, reported: (treatment, outcome), run, .. } = input;
        let ctx = run.ctx;
        let AdmgDerivativeFits { responses, witness_sites, horizon_reports, horizon_notes } =
            self.fit_admg_derivative_horizons(input, claims)?;
        let mut combined = combine_horizon_derivatives(&query.functional, &responses)?;
        let first = claims.first().ok_or(CausalError::Compile {
            message: "temporal ADMG derivative has no horizon claims".into(),
        })?;
        let mut identification = first.identification.clone();
        let (status, assumptions) =
            aggregate_temporal_horizon_evidence(claims.iter().map(|claim| &claim.identification))?;
        identification.status = status;
        identification.required_assumptions = assumptions.clone();
        combined.identification_status = status;
        combined.assumptions = assumptions;
        let refuted = self.refute_admg_derivative_horizons(
            data,
            &combined,
            &witness_sites,
            horizon_reports,
            horizon_notes,
            ctx,
        )?;
        let mut result = self.finish_temporal_admg_response(
            combined,
            IdentifiedClaim { identification, estimand: first.estimand.clone() },
            (treatment, outcome),
            run,
            refuted,
        );
        // The evidence rows of the temporal-ADMG derivative cells run one horizon; several
        // horizons execute and report their status as refused.
        result.support_status = Some(crate::support::CellStatus::NotSupported);
        Ok(result)
    }

    /// Fit each horizon on its native window, retaining its own witness and refuter state.
    fn fit_admg_derivative_horizons(
        &self,
        input: AdmgDerivativeInputs<'_>,
        claims: &[AdmgDerivativeClaim],
    ) -> Result<AdmgDerivativeFits, CausalError> {
        let AdmgDerivativeInputs {
            data,
            template,
            query,
            temporal,
            reported: (treatment, outcome),
            run,
        } = input;
        let ctx = run.ctx;
        let mut responses = Vec::with_capacity(claims.len());
        let mut witness_sites = Vec::new();
        let mut horizon_reports = Vec::new();
        let mut horizon_notes = Vec::new();
        for claim in claims {
            let AdmgDerivativeClaim { horizon, identification, estimand, indexer, .. } = claim;
            let single =
                crate::analysis::temporal_admg_contrast::single_horizon_query(query, *horizon)?;
            responses.push(if estimand.method_kind().ok() == Some(EstimandMethod::FrontDoor) {
                let fit = self.temporal_admg_front_door_response(
                    (data, template, &single),
                    (identification, estimand),
                    ctx,
                )?;
                if self.settings.refute != RefuteSuite::None {
                    let (reports, notes) = self.refute_temporal_admg_front_door(
                        &fit,
                        (identification, estimand),
                        ctx,
                    )?;
                    horizon_reports.extend(reports);
                    horizon_notes.extend(notes);
                    horizon_notes.push(horizon_note(*horizon, "front-door derivative state"));
                }
                fit.response
            } else {
                if self.settings.refute != RefuteSuite::None {
                    witness_sites.push(claim.pulse_witness(input));
                }
                let probe = unit_contrast_query(
                    temporal,
                    (treatment, outcome),
                    *horizon,
                    query.target_population.clone(),
                );
                temporal_dag_derivative_response(
                    data,
                    &single,
                    DagDerivativeFit {
                        estimand,
                        indexer,
                        identification,
                        offsets: (probe.try_treatment_offset()?, probe.outcome_offset()?),
                    },
                    (&self.settings.inference, self.settings.bootstrap_replicates),
                    ctx,
                )?
            });
        }
        Ok(AdmgDerivativeFits { responses, witness_sites, horizon_reports, horizon_notes })
    }

    /// Combine the published support check with each native horizon's refuter evidence.
    fn refute_admg_derivative_horizons(
        &self,
        data: &TimeSeriesData,
        combined: &CausalResponse,
        witness_sites: &[super::super::temporal_response_refutation::PulseWitnessSite],
        horizon_reports: Vec<antecedent_validate::RefutationReport>,
        horizon_notes: Vec<Diagnostic>,
        ctx: &ExecutionContext,
    ) -> Result<super::super::temporal_response_refutation::TemporalLevelRefutation, CausalError>
    {
        // A multi-horizon derivative's cheap/full suite is the support of the published
        // values, the front-door state of each front-door horizon and the TemporalDag
        // Pulse-native suite on each adjustment horizon's witness; horizons are not mixed
        // (unsupported).
        let mut refuted = (Vec::new(), Vec::new(), Vec::new());
        if self.settings.refute != RefuteSuite::None {
            let value = super::super::derivative_refutation::derivative_scalar(combined).0;
            refuted.0.push(super::super::derivative_refutation::support_report(combined, value));
            refuted.0.extend(horizon_reports);
            refuted.2.extend(horizon_notes);
            if !witness_sites.is_empty() {
                let (reports, checks, notes, _) =
                    super::super::temporal_response_refutation::refute_pulse_witness_sites(
                        self,
                        data,
                        witness_sites,
                        ctx,
                    )?;
                refuted.0.extend(reports);
                refuted.1.extend(checks);
                refuted.2.extend(notes);
                refuted.2.push(super::super::temporal_response_refutation::state_note(
                    "derivative",
                    witness_sites.len(),
                ));
            }
        }
        Ok(refuted)
    }

    /// The front-door derivative of one horizon of a temporal ADMG, on the lag-aligned window
    /// table (column `period * variables + variable`); the interval resamples circular blocks of
    /// consecutive windows (a block never shorter than the window) or shares one Dirichlet weight
    /// per block.
    fn temporal_admg_front_door_response(
        &self,
        (series, template, query): (
            &TimeSeriesData,
            &antecedent_graph::TemporalAdmg,
            &ResponseQuery,
        ),
        (identification, estimand): (&IdentificationResult, &IdentifiedEstimand),
        ctx: &ExecutionContext,
    ) -> Result<TemporalFrontDoorFit, CausalError> {
        let Some(temporal) = query.temporal.as_ref() else {
            return Err(CausalError::Compile {
                message: "temporal response route requires TemporalResponseSpec".into(),
            });
        };
        let (treatment, outcome) =
            super::super::response_path::response_primary_pair(&query.functional)?;
        let variable_count = u32::try_from(series.schema().len()).map_err(|_| {
            CausalError::Compile { message: "series schema exceeds the variable id range".into() }
        })?;
        let probe = unit_contrast_query(
            temporal,
            (treatment, outcome),
            temporal.horizons[0],
            query.target_population.clone(),
        );
        let mut budget =
            antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
        let plan = antecedent_identify::temporal_admg::plan_temporal_admg_window_within(
            template,
            &probe,
            &mut budget,
            ctx,
        )?;
        let window =
            template.unroll_window_over_within(plan.periods, variable_count, &mut budget, ctx)?;
        let table = crate::analysis::temporal_admg_contrast::window_table(
            series,
            &window,
            &crate::analysis::temporal_admg_contrast::used_variables(template, &probe),
            &mut budget,
            ctx,
        )?;
        let treated = *plan.treated.first().ok_or_else(|| CausalError::Compile {
            message: "a temporal derivative needs a treated period in its policy".into(),
        })?;
        let coordinate = |variable: VariableId, period: u32| {
            window.coordinate(variable, period).ok_or_else(|| CausalError::Compile {
                message: "a derivative variable is outside the unrolled window".into(),
            })
        };
        let treatment_column = coordinate(treatment, treated)?;
        let outcome_column = coordinate(outcome, plan.outcome_period)?;
        let functional = match &query.functional {
            ResponseFunctional::PointDerivative { at, order, scale, .. } => {
                ResponseFunctional::PointDerivative {
                    outcome: outcome_column,
                    treatment: treatment_column,
                    at: *at,
                    order: *order,
                    scale: *scale,
                }
            }
            ResponseFunctional::AverageDerivative { weighting, .. } => {
                ResponseFunctional::AverageDerivative {
                    outcome: outcome_column,
                    treatment: treatment_column,
                    weighting: weighting.clone(),
                }
            }
            _ => {
                return Err(CausalError::Unsupported {
                    message: "a front-door temporal derivative serves a point or average \
                              derivative",
                });
            }
        };
        let window_query = ResponseQuery::new(functional);
        let block = antecedent_estimate::RowResampling::WindowBlocks { span: plan.periods }
            .block_length(table.row_count())
            .unwrap_or(1);
        let mut response = self.estimate_front_door_derivative_response(
            &table,
            &window_query,
            super::super::response_path::FrontDoorClaim {
                estimand,
                status: identification.status,
                assumptions: identification.required_assumptions.clone(),
            },
            block,
            ctx,
        )?;
        // The response answers the caller's query, not the window-coordinate one.
        response.estimand = query.functional.clone();
        Ok(TemporalFrontDoorFit { response, table, window_query, block })
    }

    /// The front-door derivative refuter state of one horizon of a temporal ADMG, on the
    /// window table the published value was estimated on (unsupported).
    fn refute_temporal_admg_front_door(
        &self,
        fit: &TemporalFrontDoorFit,
        (identification, estimand): (&IdentificationResult, &IdentifiedEstimand),
        ctx: &ExecutionContext,
    ) -> Result<(Vec<antecedent_validate::RefutationReport>, Vec<Diagnostic>), CausalError> {
        self.refute_front_door_derivative_response(
            (&fit.table, &fit.window_query),
            &super::super::response_path::FrontDoorClaim {
                estimand,
                status: identification.status,
                assumptions: identification.required_assumptions.clone(),
            },
            &fit.response,
            fit.block,
            ctx,
        )
    }

    /// The result of a temporal ADMG derivative from its response.
    fn finish_temporal_admg_response(
        &self,
        response: CausalResponse,
        claim: IdentifiedClaim,
        (treatment, outcome): (VariableId, VariableId),
        run: ExecuteRun<'_>,
        refuted: super::super::temporal_response_refutation::TemporalLevelRefutation,
    ) -> StudyResult {
        let (refutations, predictive_checks, notes) = refuted;
        let IdentifiedClaim { identification, estimand } = claim;
        let ExecuteRun { physical, ctx: _, started, identify_cached } = run;
        let scalar = super::super::response_path::response_scalar_summary(&response).0;
        let estimate = EffectEstimate::new(
            scalar,
            f64::NAN,
            identification.required_assumptions.clone(),
            OverlapPolicy::ExplicitOverride,
        );
        let mut diagnostics: Vec<Diagnostic> = response.support.warnings.clone();
        diagnostics.extend(notes);
        let mut result = self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand,
                estimate,
                Strategy::new(
                    IdentifierId::GeneralizedAdjustment,
                    EstimatorId::temporal_response_for(&self.settings.inference),
                ),
                treatment,
                outcome,
            )
            .identify_cached(identify_cached)
            .refutations(refutations)
            .wall_time_ns(elapsed_ns(started)),
        );
        result.payload = crate::result::ResultPayload::Response(response);
        result.posterior = None;
        result.predictive_checks = predictive_checks;
        result.rebind_interval(self.settings.inference.is_bayesian());
        result.diagnostics.extend(diagnostics);
        result
    }
}
