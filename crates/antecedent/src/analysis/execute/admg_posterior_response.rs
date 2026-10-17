//! ADMG graph-posterior response surfaces: general.id + functional.effect per atom.
//!
//! Atoms are single ADMGs, not MEC completions. Function-valued surfaces mix by
//! frozen posterior weight when [`StructuralAggregationPolicy`] is
//! `SameEstimandWeightedMean`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::response_path::{
    estimate_general_id_response, graph_posterior_response_supported, mix_response_values,
    mix_support_reports, response_envelope_from_weighted, response_identified_value,
    response_primary_pair, response_scalar_summary,
};
use super::{
    Arc, AverageEffectQuery, CausalError, CausalQuery, CausalResponse, Diagnostic, DiagnosticKind,
    DiagnosticSeverity, EffectEstimate, EstimationWorkspace, EstimatorId, ExecutionContext,
    GraphIdentFlag, GraphPosterior, IdentificationResult, IdentificationStatus, IdentifiedEstimand,
    IdentifiedExecuteExtras, IdentifiedExecuteFinish, IdentifiedKeyWeights, IdentifierId, Instant,
    Intervention, OverlapPolicy, PhysicalExecutionPlan, PosteriorMassSummary, RefutationRun,
    RefuteSuite, ResponseFunctional, ResponseIdentification, ResponseQuery, ResponseUncertainty,
    ResponseValue, Strategy, Study, StudyResult, TabularData, VariableId, elapsed_ns,
    identify_admg_query, identify_cached_diagnostic, mass_fields, provenance_ids,
    push_graph_posterior_structural_aggregation_diagnostic, resolve_posterior_status,
    resolve_structural_aggregation, select_estimand, support_from_functional_eval,
};
use crate::analysis::prepared::CachedGraphPosteriorIdentification;
use crate::result::{
    StructuralAggregationPolicy, StructuralResponseAtom, StructuralResponseMixture,
};

/// Borrowed inputs shared by the phases of an ADMG graph-posterior response.
#[derive(Clone, Copy)]
struct AdmgPosteriorRun<'a> {
    data: &'a TabularData,
    gp: &'a GraphPosterior,
    query: &'a ResponseQuery,
    treatment: VariableId,
    outcome: VariableId,
    ctx: &'a ExecutionContext,
}

/// Atom responses of an ADMG graph posterior with the primary estimand and failed mass.
struct AdmgPosteriorFits {
    atoms: Vec<StructuralResponseAtom>,
    weighted: Vec<(u64, f64, CausalResponse)>,
    evals: Vec<AdmgPosteriorResponseAtom>,
    estimand: IdentifiedEstimand,
    failed_mass: f64,
}

/// The pooled ADMG response with the values its diagnostics and result read.
struct AdmgPosteriorPooled {
    identification: IdentificationResult,
    response: CausalResponse,
    estimate: EffectEstimate,
    scalar: f64,
    aggregation_policy: StructuralAggregationPolicy,
    conditional: ResponseValue,
    structural_set: Option<antecedent_core::ResponseEnvelope>,
    identified_mass: f64,
    unidentified_mass: f64,
    total_mass: f64,
}

struct AdmgPosteriorResponseAtom {
    estimand: IdentifiedEstimand,
}

impl super::Study {
    pub(in crate::analysis) fn execute_checked_admg_graph_posterior_response(
        &self,
        data: &TabularData,
        operation: &super::super::CheckedAdmgGraphPosteriorResponse,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let mut bound = self.clone();
        bound.query = CausalQuery::Response(operation.query().clone());
        bound.graph_posterior = Some(operation.posterior().clone());
        bound.graph_posterior_identification_cache = Some(Arc::clone(operation.identification()));
        bound.inference = operation.inference().clone();
        bound.refute = operation.validation();
        bound.execute_admg_graph_posterior_response(
            data,
            operation.posterior(),
            operation.query(),
            operation.physical(),
            ctx,
        )
    }

    /// Mix ADMG posterior atoms through `identify_admg_query` + `functional.effect`.
    pub(super) fn execute_admg_graph_posterior_response(
        &self,
        data: &TabularData,
        gp: &GraphPosterior,
        query: &ResponseQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        if !matches!(
            gp.atom_kind,
            antecedent_discovery::GraphPosteriorAtomKind::Admg
                | antecedent_discovery::GraphPosteriorAtomKind::TemporalAdmg
        ) {
            return Err(CausalError::Compile {
                message: "execute_admg_graph_posterior_response requires Admg posterior atoms"
                    .into(),
            });
        }
        graph_posterior_response_supported(query)?;
        let (treatment, outcome) = response_primary_pair(&query.functional)?;
        let (identified, identify_cached) =
            self.admg_posterior_response_identification(gp, query, ctx)?;
        if identified.atoms.is_empty() && identified.graphs.identified_mass() <= 0.0 {
            return Err(CausalError::Compile {
                message: "ADMG graph-posterior response: no identified ADMG atoms".into(),
            });
        }

        let run = AdmgPosteriorRun { data, gp, query, treatment, outcome, ctx };
        let (fits, identification) = self.fit_admg_posterior_atoms(&run, &identified)?;
        let pooled = Self::pool_admg_posterior_response(query, &identified, &fits, identification)?;
        let mut diagnostics =
            self.admg_posterior_response_diagnostics(&fits, &pooled, identify_cached);
        let (refutations, refute_diagnostics) =
            self.refute_admg_posterior_response(&run, &fits.estimand, &pooled, &mut diagnostics)?;
        diagnostics.extend(refute_diagnostics);
        let AdmgPosteriorPooled {
            identification,
            response,
            estimate,
            scalar,
            aggregation_policy,
            conditional,
            structural_set,
            identified_mass,
            unidentified_mass,
            total_mass,
        } = pooled;
        let AdmgPosteriorFits { atoms, estimand, failed_mass, .. } = fits;
        if !scalar.is_finite() {
            diagnostics.push(Diagnostic::new(
                "estimate.response.no_scalar_summary",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "this response is function-valued or not point identified; the scalar effect \
                 summary is not applicable and the result is carried by the response payload",
            ));
        }
        for warning in &response.support.warnings {
            diagnostics.push(warning.clone());
        }

        let algo =
            physical.logical.record.discovery_algorithm.as_deref().unwrap_or("graph_posterior");
        Ok(self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand,
                estimate,
                Strategy::new(IdentifierId::GeneralId, EstimatorId::FunctionalEffect),
                treatment,
                outcome,
            )
            .identify_cached(identify_cached)
            .refutations(refutations)
            .wall_time_ns(elapsed_ns(started))
            .extras(IdentifiedExecuteExtras {
                identify_provenance: Some(provenance_ids("discover.graph_posterior", algo)),
                estimate_provenance: Some(provenance_ids(
                    "estimate.response.admg_graph_posterior",
                    aggregation_policy.as_str(),
                )),
                diagnostics: Some(diagnostics),
                response: Some(response),
                structural_response: Some(StructuralResponseMixture {
                    weight_basis: crate::result::StructuralWeightBasis::PosteriorProbability,
                    atoms,
                    identified_mass: identified_mass / total_mass,
                    unidentified_mass: unidentified_mass / total_mass,
                    unevaluable_mass: failed_mass / total_mass,
                    subsampled_out_mass: 0.0,
                    identified_set: structural_set,
                    identified_set_interval: None,
                    conditional_on_identified: Some(conditional),
                    full_mass_scope: true,
                    truncated_atoms: 0,
                }),
                bootstrap_replicates_requested: Some(None),
                ..Default::default()
            }),
        ))
    }

    /// The prepared ADMG posterior identification cache, or a live build of it.
    fn admg_posterior_response_identification(
        &self,
        gp: &GraphPosterior,
        query: &ResponseQuery,
        ctx: &ExecutionContext,
    ) -> Result<(Arc<CachedGraphPosteriorIdentification>, bool), CausalError> {
        if let Some(cache) = self.graph_posterior_identification_cache.as_ref() {
            Ok((Arc::clone(cache), true))
        } else {
            Ok((
                Arc::new(
                    crate::analysis::prepared::build_admg_graph_posterior_response_identification_cache(
                        gp, query, ctx,
                    )?,
                ),
                false,
            ))
        }
    }

    /// Estimates every identified ADMG atom's response and records it on the structural atoms.
    fn fit_admg_posterior_atoms(
        &self,
        run: &AdmgPosteriorRun<'_>,
        identified: &CachedGraphPosteriorIdentification,
    ) -> Result<(AdmgPosteriorFits, IdentificationResult), CausalError> {
        let mut atoms = identified
            .graphs
            .graph_keys
            .iter()
            .zip(identified.graphs.weights.iter())
            .zip(identified.graphs.identified.iter())
            .map(|((&graph_key, &weight), flag)| StructuralResponseAtom {
                posterior: None,
                response: None,
                graph_key,
                weight,
                status: if *flag == GraphIdentFlag::Identified {
                    IdentificationStatus::NonparametricallyIdentified
                } else {
                    IdentificationStatus::NotIdentified
                },
                value: None,
            })
            .collect::<Vec<_>>();
        let mut weighted = Vec::new();
        let mut evals = Vec::new();
        let mut primary = None;
        let mut failed_mass = 0.0;
        let key_weights = IdentifiedKeyWeights::new(&identified.graphs);
        let work: Vec<_> =
            identified.atoms.iter().filter(|atom| key_weights.get(atom.key) > 0.0).collect();
        let fitted = run.ctx.map_indexed(work.len(), |i, inner| {
            let atom = work[i];
            let mask = mask_for_atom_key(run.gp, atom.key)?;
            let response = estimate_admg_posterior_atom_response(
                self,
                run.data,
                run.query,
                AtomAdjacency { mask, n_vars: run.gp.n_vars },
                &atom.identification,
                &atom.estimand,
                inner,
            );
            Ok::<_, CausalError>((atom, response))
        })?;
        for (atom, response) in fitted {
            let weight = key_weights.get(atom.key);
            let Ok(response) = response else {
                failed_mass += weight;
                for slot in atoms.iter_mut().filter(|candidate| candidate.graph_key == atom.key) {
                    slot.status = atom.identification.status;
                }
                continue;
            };
            let value =
                response_identified_value(&response).ok_or_else(|| CausalError::Compile {
                    message: "identified ADMG graph-posterior response atom had no numerical value"
                        .into(),
                })?;
            for slot in atoms.iter_mut().filter(|candidate| candidate.graph_key == atom.key) {
                slot.status = atom.identification.status;
                slot.value = Some(value.clone());
                slot.response = Some(response.clone());
            }
            if primary.is_none() {
                primary = Some((atom.estimand.clone(), atom.identification.clone()));
            }
            weighted.push((atom.key, weight, response.clone()));
            evals.push(AdmgPosteriorResponseAtom { estimand: atom.estimand.clone() });
        }
        let (estimand, identification) = primary.ok_or_else(|| CausalError::Compile {
            message: "ADMG graph-posterior response has no evaluable identified atom".into(),
        })?;
        Ok((AdmgPosteriorFits { atoms, weighted, evals, estimand, failed_mass }, identification))
    }

    /// Pools the atom responses into the published response, estimate and mass accounting.
    fn pool_admg_posterior_response(
        query: &ResponseQuery,
        identified: &CachedGraphPosteriorIdentification,
        fits: &AdmgPosteriorFits,
        mut identification: IdentificationResult,
    ) -> Result<AdmgPosteriorPooled, CausalError> {
        let AdmgPosteriorFits { weighted, evals, failed_mass, .. } = fits;
        let failed_mass = *failed_mass;
        let identified_mass = weighted.iter().map(|(_, weight, _)| weight).sum::<f64>();
        let total_mass = identified.graphs.total_weight();
        let unidentified_mass = identified.graphs.unidentified_mass();
        let contributing: Vec<&IdentifiedEstimand> =
            evals.iter().map(|eval| &eval.estimand).collect();
        let any_partial = weighted.iter().any(|(key, _, response)| {
            identified.atoms.iter().find(|atom| atom.key == *key).is_some_and(|atom| {
                matches!(atom.identification.status, IdentificationStatus::PartiallyIdentified)
            }) || response.identification_status == IdentificationStatus::PartiallyIdentified
        });
        let aggregation_policy = resolve_structural_aggregation(&contributing, any_partial);
        let conditional_values = weighted
            .iter()
            .filter_map(|(_, weight, response)| {
                response_identified_value(response).map(|value| (*weight, value))
            })
            .collect::<Vec<_>>();
        let conditional_refs =
            conditional_values.iter().map(|(weight, value)| (*weight, value)).collect::<Vec<_>>();
        let conditional = mix_response_values(&conditional_refs)?;
        let structural_set = response_envelope_from_weighted(weighted);
        let first = &weighted[0].2;
        identification.status = resolve_posterior_status(
            identification.status,
            aggregation_policy,
            PosteriorMassSummary {
                unidentified: unidentified_mass,
                failed: failed_mass,
                ..PosteriorMassSummary::default()
            },
        );
        let graph_dependent = identification.status == IdentificationStatus::GraphDependent;
        let uncertainty =
            if weighted.len() == 1 { first.uncertainty.clone() } else { ResponseUncertainty::None };
        let mut assumptions = first.assumptions.clone();
        for (_, _, response) in weighted.iter().skip(1) {
            assumptions.extend_unique(&response.assumptions.entries);
        }
        let response = CausalResponse {
            estimand: query.functional.clone(),
            identification_status: identification.status,
            estimate: if graph_dependent {
                ResponseIdentification::GraphDependent(
                    weighted
                        .iter()
                        .filter_map(|(key, _, response)| {
                            response_identified_value(response).map(|value| (*key, value))
                        })
                        .collect(),
                )
            } else {
                ResponseIdentification::PointIdentified(conditional.clone())
            },
            uncertainty,
            support: mix_support_reports(
                &weighted.iter().map(|(_, _, response)| &response.support).collect::<Vec<_>>(),
            ),
            assumptions,
            provenance_id: Arc::from("estimate.response.admg_graph_posterior"),
            horizon_identification: None,
            interaction_structurally_zero: first.interaction_structurally_zero,
        };
        let (scalar, standard_error) = response_scalar_summary(&response);
        let estimate = EffectEstimate::new(
            scalar,
            standard_error,
            response.assumptions.clone(),
            OverlapPolicy::ExplicitOverride,
        );
        Ok(AdmgPosteriorPooled {
            identification,
            response,
            estimate,
            scalar,
            aggregation_policy,
            conditional,
            structural_set,
            identified_mass,
            unidentified_mass,
            total_mass,
        })
    }

    /// Mass accounting, withheld-band and identification disclosures of a pooled ADMG response.
    fn admg_posterior_response_diagnostics(
        &self,
        fits: &AdmgPosteriorFits,
        pooled: &AdmgPosteriorPooled,
        identify_cached: bool,
    ) -> Vec<Diagnostic> {
        let AdmgPosteriorPooled {
            aggregation_policy,
            identified_mass,
            unidentified_mass,
            total_mass,
            ..
        } = *pooled;
        let weighted = &fits.weighted;
        let failed_mass = fits.failed_mass;
        let mut diagnostics = vec![
            Diagnostic::new(
                "estimate.response.admg_graph_posterior",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                format!(
                    "each ADMG posterior atom was identified with general.id and estimated with \
                     functional.effect; outer weights are posterior probabilities; ADMG atoms are \
                     single graphs, not MEC completions; identified_mass={}; \
                     unidentified_mass={}; unevaluable_mass={}",
                    identified_mass / total_mass,
                    unidentified_mass / total_mass,
                    failed_mass / total_mass,
                ),
            )
            .with_fields(mass_fields(
                Some(identified_mass / total_mass),
                unidentified_mass / total_mass,
            )),
            Diagnostic::new(
                "identify.response.general_id",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "intervention mean identified by Shpitser–Pearl ID; levels are the discrete \
                 functional.effect plug-in, not an adjustment g-formula",
            ),
        ];
        if weighted.len() > 1 {
            diagnostics.push(Diagnostic::new(
                "estimate.response.admg_graph_posterior.uncertainty_withheld",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Warning,
                if self.inference.is_bayesian() {
                    "multi-atom Bayesian ADMG graph-posterior response withholds an aggregate \
                     credible band: graph-specific dispersion and a frozen-weight aggregate are \
                     different objects"
                } else {
                    "multi-atom Frequentist ADMG graph-posterior response withholds an aggregate \
                     band: atom surfaces are mixed by frozen graph weight and their marginal \
                     bands are not combined as independent"
                },
            ));
        }
        if identify_cached {
            diagnostics.push(identify_cached_diagnostic());
        }
        push_graph_posterior_structural_aggregation_diagnostic(
            &mut diagnostics,
            aggregation_policy,
            identified_mass / total_mass,
            unidentified_mass / total_mass,
            failed_mass / total_mass,
            0.0,
        );
        diagnostics
    }

    /// Plugin-level refutation of a scalar ADMG posterior response under frozen-weight mixing.
    fn refute_admg_posterior_response(
        &self,
        run: &AdmgPosteriorRun<'_>,
        estimand: &IdentifiedEstimand,
        pooled: &AdmgPosteriorPooled,
        diagnostics: &mut Vec<Diagnostic>,
    ) -> Result<(Vec<antecedent_validate::RefutationReport>, Vec<Diagnostic>), CausalError> {
        let AdmgPosteriorPooled { aggregation_policy, scalar, .. } = *pooled;
        let is_intervention =
            matches!(run.query.functional, ResponseFunctional::InterventionResponse { .. });
        if is_intervention
            && scalar.is_finite()
            && matches!(aggregation_policy, StructuralAggregationPolicy::SameEstimandWeightedMean)
            && (!matches!(self.refute, RefuteSuite::None) || !self.custom_validators.is_empty())
        {
            let ate_query = AverageEffectQuery::binary_ate(run.treatment, run.outcome);
            let mut refute_ws = EstimationWorkspace::default();
            if matches!(self.refute, RefuteSuite::Cheap | RefuteSuite::Full) {
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
                estimate: &pooled.estimate,
                suite: self.refute,
                estimator: EstimatorId::FunctionalEffect.as_str(),
                custom: &self.custom_validators,
                temporal: None,
            }
            .plugin_level(&mut refute_ws, run.ctx)
        } else {
            if !matches!(self.refute, RefuteSuite::None) {
                diagnostics.push(Diagnostic::new(
                    "refute.response.skipped",
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Info,
                    if matches!(
                        aggregation_policy,
                        StructuralAggregationPolicy::GraphDependentAtoms
                    ) {
                        "query-native validation does not compare refuters against a withheld \
                         aggregate under GraphDependentAtoms"
                    } else {
                        "scalar ATE refuters are not applicable to a function-valued ADMG \
                         graph-posterior response"
                    },
                ));
            }
            Ok((Vec::new(), Vec::new()))
        }
    }
}

fn mask_for_atom_key(
    gp: &GraphPosterior,
    key: u64,
) -> Result<antecedent_discovery::AdjacencyMask, CausalError> {
    gp.graph_keys
        .iter()
        .zip(gp.atoms.iter())
        .find(|(candidate, _)| **candidate == key)
        .map(|(_, mask)| *mask)
        .ok_or_else(|| CausalError::Compile {
            message: format!("ADMG graph-posterior atom missing adjacency mask for key {key:x}"),
        })
}

/// An ADMG posterior atom's adjacency bitmask and the variable count it indexes.
#[derive(Clone, Copy)]
struct AtomAdjacency {
    mask: antecedent_discovery::AdjacencyMask,
    n_vars: usize,
}

fn estimate_admg_posterior_atom_response(
    study: &Study,
    data: &TabularData,
    query: &ResponseQuery,
    adjacency: AtomAdjacency,
    identification: &IdentificationResult,
    estimand: &IdentifiedEstimand,
    ctx: &ExecutionContext,
) -> Result<CausalResponse, CausalError> {
    let (treatment, outcome) = response_primary_pair(&query.functional)?;
    let bayesian = study.inference.is_bayesian();
    match &query.functional {
        ResponseFunctional::InterventionResponse { .. } => {
            let replicates = if bayesian { 0 } else { study.bootstrap_replicates };
            let (mut response, _) = estimate_general_id_response(
                data,
                query,
                identification,
                estimand,
                replicates,
                ctx,
            )?;
            if bayesian {
                let (estimate, _) = study.estimate_functional_effect(
                    data,
                    estimand,
                    identification,
                    &[treatment, outcome],
                    ctx,
                )?;
                if let ResponseIdentification::PointIdentified(ResponseValue::Scalar(slot)) =
                    &mut response.estimate
                {
                    *slot = estimate.ate;
                }
            }
            Ok(response)
        }
        ResponseFunctional::MeanCurve { .. } => {
            estimate_admg_mean_curve(study, data, query, adjacency, identification, estimand, ctx)
        }
        _ => Err(CausalError::Unsupported {
            message: "ADMG graph-posterior response is licensed for InterventionResponse and \
                      ResponseCurve",
        }),
    }
}

/// A mean-curve ADMG atom response: one general-ID plug-in per grid level.
fn estimate_admg_mean_curve(
    study: &Study,
    data: &TabularData,
    query: &ResponseQuery,
    adjacency: AtomAdjacency,
    identification: &IdentificationResult,
    estimand: &IdentifiedEstimand,
    ctx: &ExecutionContext,
) -> Result<CausalResponse, CausalError> {
    let AtomAdjacency { mask, n_vars } = adjacency;
    let bayesian = study.inference.is_bayesian();
    let ResponseFunctional::MeanCurve { outcome, treatment: tx } = &query.functional else {
        return Err(CausalError::Unsupported {
            message: "ADMG graph-posterior response is licensed for InterventionResponse and \
                      ResponseCurve",
        });
    };
    let admg = antecedent_discovery::adjacency_mask::admg_from_adjacency_mask(mask, n_vars)?;
    let levels = tx.grid.values()?;
    let mut means = Vec::with_capacity(levels.len());
    let mut grid = Vec::with_capacity(levels.len());
    let mut freq_curve_bounds: Vec<(f64, f64)> = Vec::new();
    let mut support = None;
    let replicates = if bayesian { 0 } else { study.bootstrap_replicates };
    for level in levels {
        let level_query = admg_response_at_level(query, tx.variable, *outcome, level);
        let (level_id, level_est) = if means.is_empty() {
            (identification.clone(), estimand.clone())
        } else {
            let level_id = identify_admg_query(
                IdentifierId::GeneralId,
                &admg,
                &CausalQuery::Response(level_query.clone()),
            )?;
            let level_est = select_estimand(&level_id, EstimatorId::FunctionalEffect)?;
            (level_id, level_est)
        };
        let mean = if bayesian {
            let (estimate, _) = study.estimate_functional_effect(
                data,
                &level_est,
                &level_id,
                &[tx.variable, *outcome],
                ctx,
            )?;
            estimate.ate
        } else {
            let (response, _) = estimate_general_id_response(
                data,
                &level_query,
                &level_id,
                &level_est,
                replicates,
                ctx,
            )?;
            if let ResponseUncertainty::Scalar { lower, upper, .. } = &response.uncertainty {
                freq_curve_bounds.push((*lower, *upper));
            }
            let (scalar, _) = response_scalar_summary(&response);
            if support.is_none() {
                support = Some(response.support);
            }
            scalar
        };
        grid.push(level);
        means.push(mean);
    }
    let mut response_support = support.unwrap_or_else(|| {
        support_from_functional_eval(None).expect("empty functional eval publishes support")
    });
    let (min, max) = (
        grid.iter().copied().fold(f64::INFINITY, f64::min),
        grid.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    );
    response_support.query_region = antecedent_core::SupportRegion {
        minima: Arc::from(vec![min]),
        maxima: Arc::from(vec![max]),
    };
    let means_len = means.len();
    Ok(CausalResponse {
        estimand: query.functional.clone(),
        identification_status: identification.status,
        estimate: ResponseIdentification::PointIdentified(ResponseValue::Surface {
            grid: Arc::from(grid),
            dimension: 1,
            mean: Arc::from(means),
        }),
        uncertainty: if !freq_curve_bounds.is_empty() && freq_curve_bounds.len() == means_len {
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
        } else {
            ResponseUncertainty::None
        },
        support: response_support,
        assumptions: identification.required_assumptions.clone(),
        provenance_id: Arc::from("estimate.response.admg_graph_posterior"),
        horizon_identification: None,
        interaction_structurally_zero: false,
    })
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
