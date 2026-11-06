//! Per-unit panel response fitting, aggregation and uncertainty disclosures.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    Arc, CausalError, CausalResponse, CompletionFitMemo, Diagnostic, DiagnosticKind,
    DiagnosticSeverity, EffectEstimate, EstimatorId, ExecutionContext, IdentificationBasis,
    IdentificationResult, IdentificationStatus, IdentifiedEstimand, IdentifiedExecuteExtras,
    IdentifiedExecuteFinish, IdentifierId, InferenceMode, Instant, ObservationSpec, OverlapPolicy,
    PANEL_INTERVAL_LEVEL, PanelClusters, PanelData, PhysicalExecutionPlan, ResponseFitKey,
    ResponseIdentification, ResponseQuery, ResponseUncertainty, ResponseValue, RouteDefaults,
    Strategy, StudyResult, TemporalDag, TemporalResponseEstimator, bayesian_gcomp, elapsed_ns,
    identification_status_ok_for_case, identify_cached_diagnostic, indexer_span,
    panel_estimand_assumption, provenance_ids, push_conflict_diagnostics, refuse_short_panel_units,
    report_identify_compute, require_identified, select_estimand, unit_average_estimand_diagnostic,
};

/// Equal-weight panel average of per-unit surfaces.
///
/// The published response is the panel's own: the mean surface, the between-unit
/// t band ([`panel_between_unit_band`]), the support merged over every unit
/// ([`merge_panel_unit_supports`]), and one panel band record in place of the
/// unit surfaces' own band assumptions.
pub(super) fn average_unit_panel_responses(
    units: &[CausalResponse],
    bayesian: bool,
) -> Result<CausalResponse, CausalError> {
    let first = units.first().ok_or_else(|| CausalError::Compile {
        message: "panel response requires at least one unit".into(),
    })?;
    let (grid, dimension, n_cells) = match &first.estimate {
        ResponseIdentification::PointIdentified(ResponseValue::Surface {
            grid,
            dimension,
            mean,
        }) => (Arc::clone(grid), *dimension, mean.len()),
        _ => {
            return Err(CausalError::Compile {
                message: "panel response requires a point-identified surface per unit".into(),
            });
        }
    };
    let n = units.len() as f64;
    let mut acc = vec![0.0; n_cells];
    let mut unit_means = Vec::with_capacity(units.len());
    for unit in units {
        match &unit.estimate {
            ResponseIdentification::PointIdentified(ResponseValue::Surface {
                grid: unit_grid,
                dimension: unit_dim,
                mean,
            }) if unit_grid.as_ref() == grid.as_ref()
                && *unit_dim == dimension
                && mean.len() == n_cells =>
            {
                unit_means.push(mean.clone());
                for (total, value) in acc.iter_mut().zip(mean.iter()) {
                    *total += *value;
                }
            }
            _ => {
                return Err(CausalError::Compile {
                    message: "panel unit surfaces disagree on grid or identification".into(),
                });
            }
        }
    }
    for value in &mut acc {
        *value /= n;
    }
    let uncertainty = panel_between_unit_band(&acc, &unit_means);
    let mut response = first.clone();
    response.estimate = ResponseIdentification::PointIdentified(ResponseValue::Surface {
        grid,
        dimension,
        mean: acc.into(),
    });
    response.uncertainty = uncertainty;
    response.support = merge_panel_unit_supports(units);
    response.assumptions =
        panel_unit_average_assumptions(&first.assumptions, units.len(), bayesian);
    response.provenance_id = Arc::from("estimate.temporal_response.gcomp.panel");
    Ok(response)
}

/// Assumption records a unit surface carries about its own band, which the panel
/// does not publish.
const UNIT_BAND_ASSUMPTIONS: [&str; 3] = [
    "ols.homoskedastic.pointwise",
    "bayesian.temporal_response.linear_additive",
    antecedent_estimate::DEPENDENCE_ASSUMPTION_ID,
];

/// Support warnings that describe a unit surface's own (withheld) band.
const UNIT_BAND_WARNINGS: [&str; 2] = [
    antecedent_estimate::TEMPORAL_RESPONSE_NO_BAND_METHOD,
    "estimate.temporal_response.bootstrap_degraded",
];

fn panel_assumption(id: &str, description: String) -> antecedent_core::AssumptionRecord {
    antecedent_core::AssumptionRecord {
        assumption: antecedent_core::Assumption::ParametricRestriction(
            antecedent_core::ParametricAssumption {
                id: Arc::from(id),
                description: Arc::from(description),
            },
        ),
        source: antecedent_core::AssumptionSource::AlgorithmDefault {
            algorithm: Arc::from("estimate.temporal_response.gcomp.panel"),
        },
        scope: antecedent_core::AssumptionScope::Estimation,
        status: antecedent_core::AssumptionStatus::Declared,
    }
}

/// The unit surfaces' shared assumptions without their own band records, plus the
/// panel's estimand-and-band record (and, for Bayesian units, what each unit
/// surface is).
fn panel_unit_average_assumptions(
    unit: &antecedent_core::AssumptionSet,
    units: usize,
    bayesian: bool,
) -> antecedent_core::AssumptionSet {
    let mut assumptions = unit.clone();
    assumptions.entries.retain(|record| {
        !matches!(
            &record.assumption,
            antecedent_core::Assumption::ParametricRestriction(p)
                if UNIT_BAND_ASSUMPTIONS.contains(&p.id.as_ref())
        )
    });
    if bayesian {
        assumptions.push(panel_assumption(
            "bayesian.temporal_response.panel_unit_posterior_means",
            "each unit surface is the posterior mean of that unit's Gaussian linear-additive \
             unfolded outcome model at each horizon under its own long-run tempering (support \
             diagnostic response.temporal_bayesian.tempering reports the largest factor over \
             units); no posterior interval or credible band is published for the panel average"
                .to_owned(),
        ));
    }
    assumptions.push(panel_estimand_assumption(PanelClusters::UnitAverage, units));
    assumptions.push(panel_assumption(
        "panel.response.between_unit_band",
        format!(
            "the surface is the equal-weight average of the {units} unit-specific surfaces (every \
             unit weighs 1/{units} whatever its length); the pointwise 95% band is \
             mean ± t_(N-1)(0.975)·sd/sqrt(N) over the unit surfaces, which treats units as \
             independent draws from the unit population; no simultaneous band is published"
        ),
    ));
    assumptions
}

/// One support report for the panel average: the most conservative status and
/// per-cell status over units, every unit's warnings (minus the unit surfaces' own
/// band warnings), the union of the query regions, the per-horizon treatment range
/// shared by every unit (the average needs every unit to support a cell), and the
/// largest Bayesian tempering factor. No unit's simultaneous band survives.
fn merge_panel_unit_supports(units: &[CausalResponse]) -> antecedent_core::SupportReport {
    let refs: Vec<&antecedent_core::SupportReport> =
        units.iter().map(|unit| &unit.support).collect();
    let mut support = super::super::response_path::mix_support_reports(&refs);
    support.warnings.retain(|warning| !UNIT_BAND_WARNINGS.contains(&warning.code.as_ref()));
    let dims = support.query_region.minima.len();
    if refs
        .iter()
        .all(|r| r.query_region.minima.len() == dims && r.query_region.maxima.len() == dims)
    {
        support.query_region.minima = (0..dims)
            .map(|d| refs.iter().map(|r| r.query_region.minima[d]).fold(f64::INFINITY, f64::min))
            .collect();
        support.query_region.maxima = (0..dims)
            .map(|d| {
                refs.iter().map(|r| r.query_region.maxima[d]).fold(f64::NEG_INFINITY, f64::max)
            })
            .collect();
    }
    for diagnostic in &mut support.diagnostics {
        let id = diagnostic.id.clone();
        let per_unit: Option<Vec<&[f64]>> = refs
            .iter()
            .map(|r| {
                r.diagnostics
                    .iter()
                    .find(|d| d.id == id && d.values.len() == diagnostic.values.len())
                    .map(|d| d.values.as_ref())
            })
            .collect();
        let Some(per_unit) = per_unit else { continue };
        match id.as_ref() {
            "response.temporal.horizon_treatment_range"
            | "response.temporal.shifted_treatment_range" => {
                diagnostic.values = (0..diagnostic.values.len())
                    .map(|k| {
                        let values = per_unit.iter().map(|v| v[k]);
                        if k % 2 == 0 {
                            values.fold(f64::NEG_INFINITY, f64::max)
                        } else {
                            values.fold(f64::INFINITY, f64::min)
                        }
                    })
                    .collect();
                diagnostic.detail = Arc::from(format!(
                    "{}; intersection over the {} panel units (the largest minimum and the \
                     smallest maximum)",
                    diagnostic.detail,
                    units.len()
                ));
            }
            antecedent_estimate::TEMPORAL_BAYESIAN_TEMPERING_DIAGNOSTIC => {
                diagnostic.values = (0..diagnostic.values.len())
                    .map(|k| per_unit.iter().map(|v| v[k]).fold(f64::NEG_INFINITY, f64::max))
                    .collect();
                diagnostic.detail = Arc::from(format!(
                    "{}; largest factor over the {} panel units",
                    diagnostic.detail,
                    units.len()
                ));
            }
            _ => {}
        }
    }
    support
}

/// Between-unit pointwise band of an equal-weight average of `N` unit surfaces:
/// `mean ± t_{N−1}(0.975) · sd_{N−1} / √N` per cell.
///
/// The unit surfaces are treated as `N` independent draws from the unit population,
/// so the band is for the equal-weight super-population mean and carries each
/// unit's own estimation error through their dispersion. With `N − 1` degrees of
/// freedom the critical value is Student-t, not normal (at `N = 3` a normal band
/// covered about 0.82 of nominal 0.95). Below two units there is no dispersion and
/// no band.
fn panel_between_unit_band(mean: &[f64], unit_means: &[Arc<[f64]>]) -> ResponseUncertainty {
    let n = unit_means.len();
    if n < 2 {
        return ResponseUncertainty::None;
    }
    let critical =
        antecedent_stats::student_t_ppf(0.5 * (1.0 + PANEL_INTERVAL_LEVEL), (n - 1) as f64);
    let root_n = (n as f64).sqrt();
    let (lower, upper): (Vec<f64>, Vec<f64>) = mean
        .iter()
        .enumerate()
        .map(|(cell, center)| {
            let column: Vec<f64> = unit_means.iter().map(|row| row[cell]).collect();
            let half = critical * antecedent_stats::sample_std(&column) / root_n;
            (center - half, center + half)
        })
        .unzip();
    ResponseUncertainty::PointwiseBand {
        level: PANEL_INTERVAL_LEVEL,
        lower: lower.into(),
        upper: upper.into(),
        interpretation: antecedent_core::IntervalInterpretation::Confidence,
        draws: None,
    }
}

/// Diagnostic naming the between-unit band's construction and degrees of freedom,
/// or saying why no band is published.
pub(super) fn panel_between_unit_band_diagnostic(units: usize) -> Diagnostic {
    if units < 2 {
        return Diagnostic::new(
            "estimate.temporal_response.panel.no_band_method",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Warning,
            "a one-unit panel has no between-unit dispersion, so no pointwise band is \
             published",
        );
    }
    Diagnostic::new(
        "estimate.temporal_response.panel.between_unit_band",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        format!(
            "pointwise 95% band = mean ± t_(N-1)(0.975)·sd/sqrt(N) over N={units} unit \
             surfaces ({} degrees of freedom, critical value {:.4}); units are independent \
             draws from the unit population",
            units - 1,
            antecedent_stats::student_t_ppf(0.5 * (1.0 + PANEL_INTERVAL_LEVEL), (units - 1) as f64),
        ),
    )
}

/// Requested bootstrap replicates are not used by the per-unit panel routes.
pub(super) fn panel_unit_average_bootstrap_note(requested: u32) -> Option<Diagnostic> {
    (requested > 0).then(|| {
        Diagnostic::new(
            "estimate.temporal_response.panel.bootstrap_not_used",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            format!(
                "{requested} requested bootstrap replicates are not used: resampling the fitted \
                 unit surfaces would not refit them and adds nothing to the closed-form \
                 between-unit t band, which already carries each unit's estimation error"
            ),
        )
    })
}

/// Fit every unit's own temporal response surface.
///
/// Lags never cross a unit boundary: each unit's series is fit on its own. Under
/// Bayesian inference a caller prior (explicit, artifact or external composition) is
/// resolved against each unit's design with the series resolver, and its conflict
/// diagnostics are reported once per distinct message.
pub(super) fn fit_unit_panel_responses(
    panel: &PanelData,
    entries: &[&crate::analysis::prepared::CachedTemporalHorizonIdentification],
    query: &ResponseQuery,
    basis: IdentificationBasis<'_>,
    inference: &InferenceMode,
    ctx: &ExecutionContext,
    diagnostics: &mut Vec<Diagnostic>,
) -> Result<Vec<CausalResponse>, CausalError> {
    let IdentificationBasis { status, assumptions } = basis;
    let span = entries.iter().map(|entry| indexer_span(&entry.indexer)).max().unwrap_or(0);
    refuse_short_panel_units(panel, span)?;
    let identifications: Vec<_> =
        entries.iter().map(|entry| (&entry.estimand, &entry.indexer)).collect();
    let mut estimator = TemporalResponseEstimator::new();
    estimator.inner.inference.bootstrap_replicates = 0;
    let mut unit_responses = Vec::with_capacity(panel.unit_count());
    match inference {
        InferenceMode::Bayesian(cfg) => {
            let temporal = query.temporal.as_ref().ok_or_else(|| CausalError::Compile {
                message: "panel response route requires TemporalResponseSpec".into(),
            })?;
            let (treatment, outcome) =
                super::super::response_path::response_primary_pair(&query.functional)?;
            let mut conflicts = Vec::new();
            for unit in panel.units() {
                let mut bayes = bayesian_gcomp(cfg, ctx);
                let (prior, conflict) =
                    super::super::temporal_path::resolve_temporal_response_prior(
                        cfg,
                        &unit.series,
                        temporal,
                        entries,
                        treatment,
                        outcome,
                        ctx,
                    )?;
                if let Some(summary) = conflict.as_ref() {
                    push_conflict_diagnostics(&mut conflicts, summary);
                }
                bayes.prior = prior;
                unit_responses.push(
                    estimator
                        .estimate_bayesian(
                            &unit.series,
                            antecedent_estimate::TemporalIdentification {
                                horizons: &identifications,
                                status,
                                assumptions: assumptions.clone(),
                            },
                            query,
                            &bayes,
                            ctx,
                        )
                        .map_err(CausalError::from)?,
                );
            }
            for diagnostic in conflicts {
                if !diagnostics
                    .iter()
                    .any(|seen| seen.code == diagnostic.code && seen.message == diagnostic.message)
                {
                    diagnostics.push(diagnostic);
                }
            }
        }
        InferenceMode::Frequentist => {
            for unit in panel.units() {
                unit_responses.push(
                    estimator
                        .estimate(
                            &unit.series,
                            &identifications,
                            query,
                            status,
                            assumptions.clone(),
                            ctx,
                        )
                        .map_err(CausalError::from)?,
                );
            }
        }
    }
    Ok(unit_responses)
}

impl super::super::Study {
    /// Refuses panel response queries outside the supported slice and returns their temporal spec.
    fn panel_response_temporal_spec<'a>(
        query: &'a ResponseQuery,
        ctx: &ExecutionContext,
    ) -> Result<&'a antecedent_core::TemporalResponseSpec, CausalError> {
        if query.observation != ObservationSpec::Complete {
            return Err(CausalError::Unsupported {
                message: "panel response is supported for complete observations only",
            });
        }
        let Some(temporal) = query.temporal.as_ref() else {
            return Err(CausalError::Compile {
                message: "panel response route requires TemporalResponseSpec".into(),
            });
        };
        let mut budget =
            antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
        if antecedent_estimate::plan_from_response_query(query, &mut budget, ctx)
            .map_err(CausalError::from)?
            .is_some_and(|plan| plan.mechanism_overlays().is_some())
        {
            return Err(CausalError::Unsupported {
                message: "panel multi-step Sequence overlays are not supported; keep the \
                          series sequential owner",
            });
        }
        Ok(temporal)
    }

    /// Per-horizon temporal identification of a panel response, from the prepared cache when present.
    fn panel_response_identification(
        &self,
        graph: &TemporalDag,
        query: &ResponseQuery,
        temporal: &antecedent_core::TemporalResponseSpec,
        (treatment, outcome): (antecedent_core::VariableId, antecedent_core::VariableId),
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<(Arc<crate::analysis::prepared::CachedTemporalIdentification>, bool), CausalError>
    {
        let (cache, identify_cached) =
            if let Some(reused) = self.reuse_temporal_identification(physical) {
                reused
            } else {
                report_identify_compute(ctx);
                let mut budget =
                    antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
                (
                    Arc::new(crate::analysis::prepared::identify_temporal_response_horizons(
                        graph,
                        EstimatorId::TemporalResponseGcomp,
                        &crate::analysis::prepared::TemporalResponseIdentificationRequest {
                            treatment,
                            outcome,
                            temporal,
                            target_population: &query.target_population,
                            schedule: None,
                            dose: crate::analysis::prepared::single_step_dose(query)?,
                        },
                        &mut budget,
                        ctx,
                    )?),
                    false,
                )
            };
        Ok((cache, identify_cached))
    }

    /// Discloses the unit-average estimand, band contract and skipped refuters of a panel response.
    fn push_panel_response_disclosures(
        &self,
        diagnostics: &mut Vec<Diagnostic>,
        units: usize,
        response: &CausalResponse,
    ) {
        diagnostics.push(Diagnostic::new(
            "estimate.temporal_response.panel.cluster_units",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            format!(
                "panel response averages {units} unit surfaces; pointwise bands use \
                 between-unit cluster variation, not stacked-row circular blocks"
            ),
        ));
        diagnostics.push(unit_average_estimand_diagnostic(
            "estimate.temporal_response.panel.estimand",
            units,
        ));
        diagnostics.push(panel_between_unit_band_diagnostic(units));
        diagnostics.extend(panel_unit_average_bootstrap_note(self.settings.bootstrap_replicates));
        if self.settings.inference.is_bayesian() {
            diagnostics.push(Diagnostic::new(
                "estimate.temporal_response.panel.bayesian_cluster_band",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "Bayesian panel response mixes per-unit posterior means; the published band is \
                 frequentist between-unit cluster variation, not a posterior interval",
            ));
        }
        diagnostics.push(Diagnostic::new(
            "response.no_simultaneous_band_method",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Warning,
            "panel response withholds a simultaneous band: series circular-block simultaneous \
             coverage does not transfer to a unit-cluster contract",
        ));
        diagnostics.push(Diagnostic::new(
            "refute.temporal_response.skipped",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "scalar ATE refuters are not applicable to a function-valued temporal response",
        ));
        diagnostics.extend(response.support.warnings.iter().cloned());
    }

    pub(in crate::analysis::execute) fn execute_panel_response(
        &self,
        panel: &PanelData,
        graph: &TemporalDag,
        query: &ResponseQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        self.refuse_unsupported_panel(panel)?;
        let started = Instant::now();
        let temporal = Self::panel_response_temporal_spec(query, ctx)?;
        let (treatment, outcome) =
            super::super::response_path::response_primary_pair(&query.functional)?;
        let (cache, identify_cached) = self.panel_response_identification(
            graph,
            query,
            temporal,
            (treatment, outcome),
            physical,
            ctx,
        )?;
        let mut aligned = Vec::with_capacity(temporal.horizons.len());
        for &horizon in temporal.horizons.iter() {
            let entry = cache.get(horizon).ok_or_else(|| CausalError::Compile {
                message: format!("temporal identification missing horizon {horizon}"),
            })?;
            require_identified(&entry.identification)?;
            aligned.push(entry);
        }
        let first = aligned[0];
        let (aggregate_status, aggregate_assumptions) =
            super::super::temporal_path::aggregate_temporal_horizon_evidence(
                aligned.iter().map(|entry| &entry.identification),
            )?;
        let mut identification = first.identification.clone();
        identification.status = aggregate_status;
        identification.required_assumptions = aggregate_assumptions.clone();
        let estimand = first.estimand.clone();

        let mut diagnostics = Vec::new();
        let unit_responses = fit_unit_panel_responses(
            panel,
            &aligned,
            query,
            IdentificationBasis { status: aggregate_status, assumptions: &aggregate_assumptions },
            &self.settings.inference,
            ctx,
            &mut diagnostics,
        )?;
        let mut response =
            average_unit_panel_responses(&unit_responses, self.settings.inference.is_bayesian())?;
        if let Some(per_horizon) = response.horizon_identification.as_mut() {
            for (record, entry) in Arc::make_mut(per_horizon).iter_mut().zip(&aligned) {
                record.status = entry.identification.status;
            }
        }

        let (scalar, standard_error) =
            super::super::response_path::response_scalar_summary(&response);
        let estimate = EffectEstimate::new(
            scalar,
            standard_error,
            response.assumptions.clone(),
            OverlapPolicy::ExplicitOverride,
        );
        for entry in &aligned {
            diagnostics.extend(entry.identification.diagnostics.iter().cloned());
        }
        if identify_cached {
            diagnostics.push(identify_cached_diagnostic());
        }
        self.push_panel_response_disclosures(&mut diagnostics, unit_responses.len(), &response);

        Ok(self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand,
                estimate,
                Strategy::new(
                    IdentifierId::TemporalBackdoorUnfolded,
                    EstimatorId::temporal_response_for(&self.settings.inference),
                ),
                treatment,
                outcome,
            )
            .identify_cached(identify_cached)
            .wall_time_ns(elapsed_ns(started))
            .cancelled(ctx.cancellation.is_cancelled())
            .extras(IdentifiedExecuteExtras {
                identify_provenance: Some(provenance_ids(
                    "identify.temporal_backdoor",
                    "identify.temporal.backdoor.unfolded",
                )),
                estimate_provenance: Some(provenance_ids(
                    Arc::clone(&response.provenance_id),
                    Arc::clone(&response.provenance_id),
                )),
                diagnostics: Some(diagnostics),
                response: Some(response),
                bootstrap_replicates_requested: Some(None),
                ..Default::default()
            }),
        ))
    }

    pub(in crate::analysis::execute) fn execute_panel_class_response(
        &self,
        panel: &PanelData,
        query: &ResponseQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        self.refuse_unsupported_panel(panel)?;
        let started = Instant::now();
        let temporal = Self::panel_response_temporal_spec(query, ctx)?;
        let (treatment, outcome) =
            super::super::response_path::response_primary_pair(&query.functional)?;
        let identifier_id = physical.binding().identifier_or(RouteDefaults::PAG)?;
        let identify_cached = self.panel_class_response_cache_covers(temporal)?;
        let run = PanelClassResponseRun {
            panel,
            query,
            temporal,
            treatment,
            outcome,
            identifier_id,
            identify_cached,
            physical,
            ctx,
            budget: antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits()),
        };
        let mut state = PanelClassResponseState::new(
            super::super::temporal_path::ClassResponseAssembly::new(query, temporal)?,
            super::super::temporal_path::TemporalClassWeights::new(
                self.declared.class_prior.as_ref(),
            ),
        );
        for (horizon_index, &horizon) in temporal.horizons.iter().enumerate() {
            self.fit_panel_class_response_horizon(&run, &mut state, horizon_index, horizon)?;
        }
        let PanelClassResponseState {
            mut diagnostics,
            assembly,
            primary_estimand,
            primary_identification,
            primary_envelope,
            assumptions,
            n_fitted,
            ..
        } = state;
        let super::super::temporal_path::ClassResponseParts {
            response,
            response_envelope,
            identified_mass,
            unidentified_mass,
            unevaluable_mass,
        } = assembly.finish(
            query,
            temporal,
            assumptions.clone(),
            "estimate.temporal_response.gcomp.panel.class",
        )?;
        let (scalar, standard_error) =
            super::super::response_path::response_scalar_summary(&response);
        let estimate = EffectEstimate::new(
            scalar,
            standard_error,
            assumptions,
            OverlapPolicy::ExplicitOverride,
        );
        let mut identification = primary_identification.ok_or_else(|| CausalError::Compile {
            message: "panel class-aware response missing identification".into(),
        })?;
        identification.status = response.identification_status;
        let estimand = primary_estimand.ok_or_else(|| CausalError::Compile {
            message: "panel class-aware response missing estimand".into(),
        })?;
        self.push_panel_class_response_disclosures(
            &mut diagnostics,
            n_fitted,
            panel.unit_count(),
            &response,
            identify_cached,
        );
        let structural = self.panel_class_response_structural(
            assembly,
            response_envelope,
            temporal.horizons.len(),
            [identified_mass, unidentified_mass, unevaluable_mass],
        );
        Ok(self.finish_class_response_tail(super::super::temporal_path::ClassResponseTail {
            physical,
            identification,
            estimand,
            estimate,
            identifier_id,
            endpoints: (treatment, outcome),
            envelope: primary_envelope,
            response,
            structural,
            diagnostics,
            identify_cached,
            bootstrap_replicates: None,
            started,
            ctx,
            refutations: Vec::new(),
            identify_provenance: None,
            estimate_provenance: None,
        }))
    }

    /// Whether the prepared class cache covers every requested horizon of the panel response.
    fn panel_class_response_cache_covers(
        &self,
        temporal: &antecedent_core::TemporalResponseSpec,
    ) -> Result<bool, CausalError> {
        let identify_cached = self.temporal_class_cache_covers(&temporal.horizons);
        if self.temporal_class_cache().is_some() && !identify_cached {
            return Err(CausalError::Compile {
                message: "prepared panel class identification cache missing a requested horizon"
                    .into(),
            });
        }
        Ok(identify_cached)
    }

    /// Opens one horizon of a panel class response: identifies its envelope and builds its query.
    fn open_panel_class_response_horizon(
        &self,
        run: &PanelClassResponseRun<'_>,
        state: &mut PanelClassResponseState<'_>,
        horizon: u32,
    ) -> Result<
        (
            Arc<crate::analysis::prepared::CachedTemporalClassIdentification>,
            Vec<f64>,
            ResponseQuery,
        ),
        CausalError,
    > {
        let horizon_query = super::super::temporal_path::unit_contrast_query(
            run.temporal,
            (run.treatment, run.outcome),
            horizon,
            run.query.target_population.clone(),
        );
        let bundle = if let Some(bundle) =
            self.retained_temporal_class_horizon(run.physical, horizon_query.horizon_steps)
        {
            bundle
        } else {
            if !run.identify_cached {
                report_identify_compute(run.ctx);
            }
            self.identify_temporal_class_within(
                run.identifier_id,
                &horizon_query,
                &mut run.budget.clone(),
                run.ctx,
            )?
        };
        let envelope = &bundle.envelope.envelope;
        state.diagnostics.push(super::super::temporal_path::temporal_class_envelope_diagnostic(
            envelope,
            self.graph.class(),
        ));
        if matches!(envelope.status, IdentificationStatus::NotIdentified)
            || envelope.identified_weight.0 <= 0.0
        {
            return Err(CausalError::not_identified(
                envelope.status,
                envelope.truncated_completions > 0,
                "panel class-aware response not identified (no identified mass in envelope)",
            ));
        }
        let weights = state.class_weights.for_envelope(&bundle.envelope)?;
        state.assembly.begin_horizon(envelope);
        let mut qh = run.query.clone();
        let mut temporal_h = run.temporal.clone();
        temporal_h.horizons = Arc::from([horizon]);
        qh.temporal = Some(temporal_h);
        Ok((bundle, weights, qh))
    }

    /// Fits every identified completion of one horizon and records it in the class assembly.
    fn fit_panel_class_response_horizon(
        &self,
        run: &PanelClassResponseRun<'_>,
        state: &mut PanelClassResponseState<'_>,
        horizon_index: usize,
        horizon: u32,
    ) -> Result<(), CausalError> {
        let (bundle, weights, qh) = self.open_panel_class_response_horizon(run, state, horizon)?;
        let envelope = &bundle.envelope.envelope;
        state.n_fitted = 0;
        let mut unit_fits: CompletionFitMemo<Vec<CausalResponse>, ResponseFitKey> =
            CompletionFitMemo::default();
        for (case_index, (case, indexer)) in
            envelope.cases.iter().zip(bundle.envelope.indexers.iter()).enumerate()
        {
            if !identification_status_ok_for_case(case.result.status)
                || case.result.estimands.is_empty()
            {
                state.assembly.push_unevaluated(
                    horizon_index,
                    case_index,
                    weights[case_index],
                    case.result.status,
                );
                continue;
            }
            let estimand = select_estimand(&case.result, EstimatorId::TemporalLinearAdjustment)?;
            let entry = crate::analysis::prepared::CachedTemporalHorizonIdentification {
                horizon,
                identification: case.result.clone(),
                estimand: estimand.clone(),
                indexer: indexer.clone(),
            };
            // Completions given the same estimand, indexer, status and
            // state.assumptions fit every unit once: the run.panel, the horizon's
            // query, the inference mode and the seed are fixed here, and
            // the conflict state.diagnostics of a repeated fit are already
            // present (they are reported once per distinct message).
            let fit_key = ResponseFitKey::new(
                &estimand,
                indexer,
                case.result.status,
                &case.result.required_assumptions,
            );
            let unit_responses = if let Some(hit) = unit_fits.get(&fit_key) {
                hit.clone()
            } else {
                let fitted = fit_unit_panel_responses(
                    run.panel,
                    &[&entry],
                    &qh,
                    IdentificationBasis {
                        status: case.result.status,
                        assumptions: &case.result.required_assumptions,
                    },
                    &self.settings.inference,
                    run.ctx,
                    &mut state.diagnostics,
                )?;
                unit_fits.insert(fit_key, fitted.clone());
                fitted
            };
            let response = average_unit_panel_responses(
                &unit_responses,
                self.settings.inference.is_bayesian(),
            )?;
            let atom_assumptions = response.assumptions.clone();
            state.assembly.push_atom(
                horizon_index,
                case_index,
                weights[case_index],
                case.result.status,
                horizon,
                response,
            )?;
            state.n_fitted += 1;
            if state.primary_estimand.is_none() {
                state.primary_estimand = Some(estimand);
                state.primary_identification = Some(case.result.clone());
                state.primary_envelope = Some(bundle.envelope.clone());
                state.assumptions = atom_assumptions;
            }
        }
        state.assembly.end_horizon(horizon_index, horizon)
    }

    /// Disclosures of a panel class response: completion count, band contract, estimand.
    fn push_panel_class_response_disclosures(
        &self,
        diagnostics: &mut Vec<Diagnostic>,
        n_fitted: usize,
        panel_units: usize,
        response: &CausalResponse,
        identify_cached: bool,
    ) {
        diagnostics.push(Diagnostic::new(
            "estimate.temporal_response.panel.class.cluster_units",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            format!(
                "panel class response identifies each horizon and averages unit surfaces on \
                 {n_fitted} identified completions per horizon; the identified set is the \
                 pointwise range over those completion surfaces, and units are not stacked"
            ),
        ));
        diagnostics.extend(super::super::temporal_path::class_response_identified_set_diagnostics(
            self.settings.inference.is_bayesian(),
            self.declared.class_prior.is_some(),
        ));
        diagnostics.push(Diagnostic::new(
            "response.no_simultaneous_band_method",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Warning,
            "panel response withholds a simultaneous band: series circular-block simultaneous \
             coverage does not transfer to a unit-cluster contract",
        ));
        diagnostics.push(unit_average_estimand_diagnostic(
            "estimate.temporal_response.panel.estimand",
            panel_units,
        ));
        diagnostics.push(panel_between_unit_band_diagnostic(panel_units));
        diagnostics.extend(panel_unit_average_bootstrap_note(self.settings.bootstrap_replicates));
        diagnostics.extend(response.support.warnings.iter().cloned());
        if identify_cached {
            diagnostics.push(identify_cached_diagnostic());
        }
    }

    /// Structural mixture of a panel class response, with the caller-prior conditional mean.
    fn panel_class_response_structural(
        &self,
        assembly: super::super::temporal_path::ClassResponseAssembly,
        response_envelope: antecedent_core::ResponseEnvelope,
        n_horizons: usize,
        masses: [f64; 3],
    ) -> crate::result::StructuralResponseMixture {
        let [identified_mass, unidentified_mass, unevaluable_mass] = masses;
        let conditional = self.declared.class_prior.as_ref().and_then(|_| {
            super::super::temporal_path::temporal_class_response_mean(
                &assembly.structural_atoms,
                &response_envelope,
                n_horizons,
                assembly.full_mass_scope,
            )
        });
        crate::result::StructuralResponseMixture {
            weight_basis: if self.declared.class_prior.is_some() {
                crate::result::StructuralWeightBasis::CallerSuppliedClassPrior
            } else {
                crate::result::StructuralWeightBasis::CompletionEnumeration
            },
            atoms: assembly.structural_atoms,
            identified_mass,
            unidentified_mass,
            unevaluable_mass,
            subsampled_out_mass: 0.0,
            identified_set: Some(response_envelope),
            identified_set_interval: None,
            conditional_on_identified: conditional,
            full_mass_scope: assembly.full_mass_scope,
            truncated_atoms: assembly.truncated_atoms,
            aggregation_policy: None,
        }
    }
}

/// Borrowed inputs of every horizon of a panel class response.
struct PanelClassResponseRun<'a> {
    panel: &'a PanelData,
    query: &'a ResponseQuery,
    temporal: &'a antecedent_core::TemporalResponseSpec,
    treatment: antecedent_core::VariableId,
    outcome: antecedent_core::VariableId,
    identifier_id: IdentifierId,
    identify_cached: bool,
    physical: &'a PhysicalExecutionPlan,
    ctx: &'a ExecutionContext,
    budget: antecedent_core::TemporalBudget,
}

/// Accumulated horizons of a panel class response.
struct PanelClassResponseState<'a> {
    diagnostics: Vec<Diagnostic>,
    assembly: super::super::temporal_path::ClassResponseAssembly,
    class_weights: super::super::temporal_path::TemporalClassWeights<'a>,
    primary_estimand: Option<IdentifiedEstimand>,
    primary_identification: Option<IdentificationResult>,
    primary_envelope: Option<Arc<antecedent_identify::TemporalClassEnvelope>>,
    assumptions: antecedent_core::AssumptionSet,
    n_fitted: usize,
}

impl<'a> PanelClassResponseState<'a> {
    /// Empty accumulation for a class response over `assembly`'s horizons.
    fn new(
        assembly: super::super::temporal_path::ClassResponseAssembly,
        class_weights: super::super::temporal_path::TemporalClassWeights<'a>,
    ) -> Self {
        Self {
            diagnostics: Vec::new(),
            assembly,
            class_weights,
            primary_estimand: None,
            primary_identification: None,
            primary_envelope: None,
            assumptions: antecedent_core::AssumptionSet::default(),
            n_fitted: 0,
        }
    }
}
