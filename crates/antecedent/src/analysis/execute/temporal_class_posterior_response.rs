//! TemporalCpdag/Pag graph-posterior response: class envelope per atom, then policy.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::response_path::{
    mix_response_values, mix_support_reports, response_envelope_from_weighted,
    response_identified_value, response_primary_pair, response_scalar_summary,
};
use super::temporal_path::bare_class_atom;
use super::temporal_posterior_response::dbn_posterior_response_supported;
use super::{
    AcceptedGraph, Arc, AverageEffectQuery, CausalError, CausalQuery, CausalResponse, Diagnostic,
    DiagnosticKind, DiagnosticSeverity, EffectEstimate, EnvelopeRefutation, EnvelopeRefuteAtom,
    EstimationWorkspace, EstimatorId, ExecutionContext, GraphIdentFlag, GraphPosterior,
    GraphPosteriorScope, IdentificationResult, IdentificationStatus, IdentifiedEstimand,
    IdentifierId, Instant, Intervention, OverlapPolicy, PhysicalExecutionPlan,
    PosteriorMassSummary, RefuteSuite, ResponseFunctional, ResponseIdentification, ResponseQuery,
    ResponseUncertainty, ResponseValue, StructuralAggregationPolicy, Study, StudyResult, TableView,
    TabularData, TemporalAtomDesign, TemporalEffectQuery, TemporalRefutation, TimeSeriesData,
    VariableId, identification_status_ok_for_case, provenance_ids,
    push_graph_posterior_structural_aggregation_diagnostic, resolve_posterior_status,
    resolve_structural_aggregation, scalar_identified_set, select_estimand,
    shared_circular_block_mixture_se, temporal_class_block_span,
};
use crate::result::{StructuralResponseAtom, StructuralResponseMixture, StructuralWeightBasis};
use antecedent_discovery::dbn_posterior::{
    temporal_cpdag_from_dbn_atoms, temporal_pag_from_dbn_atoms,
};

impl super::Study {
    /// Mix TemporalCpdag/Pag posterior atoms through the existing class response envelope.
    pub(super) fn execute_temporal_class_graph_posterior_response(
        &self,
        data: &TimeSeriesData,
        gp: &GraphPosterior,
        query: &ResponseQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        dbn_posterior_response_supported(query)?;
        query.require_licensed_temporal_observation()?;
        let class_tag = temporal_class_posterior_tag(gp.atom_kind())?;
        let (treatment, outcome) = response_primary_pair(&query.functional)?;
        let vars = data.schema().variables().iter().map(|variable| variable.id).collect::<Vec<_>>();
        let inputs = ClassPosteriorInputs { data, gp, query, vars: &vars, ctx };
        let mapped = self.run_temporal_class_posterior_atoms(&inputs, physical)?;
        let ClassPosteriorFold {
            weighted,
            atoms,
            contributing,
            failed_mass,
            unidentified_mass,
            any_partial,
            primary,
        } = fold_temporal_class_posterior_atoms(mapped);
        let (identification, estimand) = primary.ok_or_else(|| CausalError::Compile {
            message: "temporal class graph-posterior response has no evaluable identified atom"
                .into(),
        })?;
        let masses = ClassPosteriorMasses::of(&weighted, gp, (unidentified_mass, failed_mass));
        let mixed = self.mix_temporal_class_posterior_atoms(
            &inputs,
            (&weighted, &atoms, &contributing),
            identification,
            (any_partial, unidentified_mass, failed_mass),
        )?;
        let ClassPosteriorMix {
            response,
            identification,
            aggregation_policy,
            structural_set,
            conditional,
        } = mixed;
        let (scalar, standard_error) = response_scalar_summary(&response);
        let estimate = EffectEstimate::new(
            scalar,
            standard_error,
            response.assumptions.clone(),
            OverlapPolicy::ExplicitOverride,
        );
        let mut diagnostics =
            class_posterior_diagnostics(class_tag, aggregation_policy, masses, weighted.len() > 1);
        let refutations = self.refute_temporal_class_posterior_response(
            &inputs,
            &weighted,
            aggregation_policy,
            &mut diagnostics,
        )?;
        if !scalar.is_finite() {
            diagnostics.push(Diagnostic::new(
                "estimate.response.no_scalar_summary",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "this response is function-valued or not point identified; the scalar effect \
                 summary is not applicable and the result is carried by the response payload",
            ));
        }
        let mut mixture = class_posterior_mixture(
            atoms,
            masses,
            (structural_set, conditional),
            (unidentified_mass + failed_mass) == 0.0,
        );
        mixture.aggregation_policy = Some(aggregation_policy);
        let algo =
            physical.logical.record.discovery_algorithm.as_deref().unwrap_or("graph_posterior");
        // Same identified-execute tail as the class response. The posterior
        // keeps its provenance strings, refutations, and the study bootstrap
        // count (the class band's `Some(count)` write). No envelope certificate.
        Ok(self.finish_class_response_tail(super::temporal_path::ClassResponseTail {
            physical,
            identification,
            estimand,
            estimate,
            identifier_id: IdentifierId::GeneralizedAdjustment,
            endpoints: (treatment, outcome),
            envelope: None,
            response,
            structural: mixture,
            diagnostics,
            identify_cached: self.class_posterior_cache_retained(gp),
            bootstrap_replicates: Some(self.settings.bootstrap_replicates),
            started,
            ctx,
            refutations,
            identify_provenance: Some(provenance_ids("discover.graph_posterior", algo)),
            estimate_provenance: Some(provenance_ids(
                "estimate.temporal_class_graph_posterior_response",
                aggregation_policy.as_str(),
            )),
        }))
    }

    /// Whether the prepared handle retains an identification for this posterior.
    fn class_posterior_cache_retained(&self, gp: &GraphPosterior) -> bool {
        self.temporal_class_posterior_cache()
            .is_some_and(|cache| cache.graphs.n_samples == gp.n_graphs())
    }

    /// Evaluates each posterior atom through the temporal class response on
    /// the context's workers. An atom the retained envelope marked
    /// unidentified, or whose graph cannot be rebuilt, keeps its mass without
    /// a result; a refusal (`Unsupported`) aborts the response.
    fn run_temporal_class_posterior_atoms(
        &self,
        inputs: &ClassPosteriorInputs<'_>,
        physical: &PhysicalExecutionPlan,
    ) -> Result<Vec<ClassPosteriorAtomRun>, CausalError> {
        let ClassPosteriorInputs { data, gp, query, vars, ctx } = *inputs;
        let lag_masks = gp.lag_masks().ok_or_else(|| CausalError::Compile {
            message: "temporal class posterior missing per-atom lag masks".into(),
        })?;
        let max_lag = gp.max_lag().ok_or_else(|| CausalError::Compile {
            message: "temporal class posterior missing max_lag".into(),
        })?;
        // A prepared handle retains one completion envelope per atom for the
        // first requested horizon, keyed by atom position. Atoms the retained
        // envelope marked unidentified keep their mass without re-enumeration,
        // and evaluable atoms reuse their retained proof for that horizon.
        let retained = self
            .temporal_class_posterior_cache()
            .filter(|cache| cache.graphs.n_samples == gp.n_graphs());
        let first_horizon = query.temporal.as_ref().and_then(|spec| spec.horizons.first().copied());
        ctx.map_indexed(gp.n_graphs(), |i, inner| {
            let key = u64::try_from(i).map_err(|_| CausalError::Compile {
                message: "temporal class graph-posterior response: too many atoms".into(),
            })?;
            let weight = gp.weights()[i];
            if retained.is_some_and(|cache| {
                cache.graphs.identified.get(i) == Some(&GraphIdentFlag::Unidentified)
            }) {
                return Ok((key, weight, None));
            }
            let retained_atom = retained
                .and_then(|cache| cache.class_atoms.iter().find(|atom| atom.key == key))
                .zip(first_horizon)
                .map(|(atom, horizon)| {
                    Arc::new(crate::analysis::prepared::CachedTemporalClassIdentification {
                        envelope: atom.envelope.clone(),
                        by_horizon: vec![(horizon, atom.envelope.clone())],
                    })
                });
            let mark = gp.mark_atom(i);
            let reconstructed = reconstruct_temporal_class_atom(
                gp.atom_kind(),
                &gp.atoms()[i],
                &lag_masks[i],
                &mark,
                gp.n_vars(),
                max_lag,
                vars,
            );
            let Ok(graph) = reconstructed else {
                return Ok((key, weight, None));
            };
            let mut atom_study = self.clone();
            atom_study.graph_posterior = None;
            atom_study.graph = Arc::new(graph);
            atom_study.settings.refute = RefuteSuite::None;
            atom_study.declared.custom_validators.clear();
            atom_study.identification =
                retained_atom.map(crate::analysis::IdentificationCache::TemporalClass);
            match atom_study.execute_temporal_class_response(data, query, physical, inner) {
                Ok(result) => Ok((key, weight, Some(Ok(result)))),
                Err(err @ CausalError::Unsupported { .. }) => Err(err),
                Err(_) => Ok((key, weight, Some(Err(())))),
            }
        })
        .map_err(Into::into)
    }

    /// Mixes the evaluated atoms into the posterior response: the aggregation
    /// policy, the conditional mix and identified set, the resolved status and
    /// the pooled uncertainty.
    fn mix_temporal_class_posterior_atoms(
        &self,
        inputs: &ClassPosteriorInputs<'_>,
        (weighted, atoms, contributing): ClassPosteriorAtoms<'_>,
        mut identification: IdentificationResult,
        (any_partial, unidentified_mass, failed_mass): (bool, f64, f64),
    ) -> Result<ClassPosteriorMix, CausalError> {
        let query = inputs.query;
        let refs: Vec<&IdentifiedEstimand> = contributing.iter().collect();
        let aggregation_policy = if refs.is_empty() && any_partial {
            StructuralAggregationPolicy::IdentifiedSetEnvelope
        } else {
            resolve_structural_aggregation(&refs, any_partial)
        };
        let conditional_values = atoms
            .iter()
            .filter_map(|atom| atom.value.clone().map(|value| (atom.weight, value)))
            .collect::<Vec<_>>();
        let conditional_refs =
            conditional_values.iter().map(|(weight, value)| (*weight, value)).collect::<Vec<_>>();
        let mixable =
            matches!(aggregation_policy, StructuralAggregationPolicy::SameEstimandWeightedMean)
                && !conditional_refs.is_empty()
                && conditional_refs.iter().all(|(_, value)| {
                    matches!(value, ResponseValue::Scalar(_) | ResponseValue::Surface { .. })
                });
        let conditional =
            if mixable { Some(mix_response_values(&conditional_refs)?) } else { None };
        let structural_set = union_identified_values(
            &conditional_values.iter().map(|(_, value)| value).collect::<Vec<_>>(),
        )
        .or_else(|| response_envelope_from_weighted(weighted));
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
        let uncertainty = if weighted.len() == 1 {
            first.uncertainty.clone()
        } else if let (true, Some(ResponseValue::Scalar(value))) = (mixable, conditional.as_ref()) {
            self.pooled_class_posterior_uncertainty(inputs, weighted, first, *value)?
        } else {
            ResponseUncertainty::None
        };
        let mut assumptions = first.assumptions.clone();
        for (_, _, response) in weighted.iter().skip(1) {
            assumptions.extend_unique(&response.assumptions.entries);
        }
        let estimate = class_posterior_estimate(
            weighted,
            aggregation_policy,
            (graph_dependent, identification.status),
            (structural_set.as_ref(), conditional.as_ref().filter(|_| mixable)),
        )?;
        let response = antecedent_core::CausalResponse {
            estimand: query.functional.clone(),
            identification_status: identification.status,
            estimate,
            uncertainty,
            support: mix_support_reports(
                &weighted.iter().map(|(_, _, response)| &response.support).collect::<Vec<_>>(),
            ),
            assumptions,
            provenance_id: Arc::from("estimate.response.temporal_class_graph_posterior"),
            horizon_identification: first.horizon_identification.as_ref().map(|horizons| {
                horizons
                    .iter()
                    .map(|horizon| antecedent_core::HorizonIdentification {
                        status: identification.status,
                        ..horizon.clone()
                    })
                    .collect()
            }),
            interaction_structurally_zero: first.interaction_structurally_zero,
        };
        identification.status = response.identification_status;
        Ok(ClassPosteriorMix {
            response,
            identification,
            aggregation_policy,
            structural_set,
            conditional: conditional.filter(|_| mixable),
        })
    }

    /// The pooled scalar intervention-response uncertainty: a joint
    /// circular-block bootstrap over the atoms (the value is the pulse
    /// effect), matching the temporal class effect path; point-only when no
    /// block SE is available.
    fn pooled_class_posterior_uncertainty(
        &self,
        inputs: &ClassPosteriorInputs<'_>,
        weighted: &[(u64, f64, CausalResponse)],
        first: &CausalResponse,
        value: f64,
    ) -> Result<ResponseUncertainty, CausalError> {
        let ClassPosteriorInputs { data, gp, query, vars, ctx } = *inputs;
        Ok(
            match temporal_class_response_pooled_block_se(
                self, data, gp, vars, query, weighted, ctx,
            )? {
                Some(se) => {
                    let level = match &first.uncertainty {
                        ResponseUncertainty::Scalar { level, .. } => *level,
                        _ => 0.95,
                    };
                    let z = antecedent_stats::normal_ppf(0.5 + level / 2.0);
                    ResponseUncertainty::Scalar {
                        standard_error: se,
                        level,
                        lower: value - z * se,
                        upper: value + z * se,
                        interpretation: antecedent_core::IntervalInterpretation::Confidence,
                        draws: None,
                    }
                }
                None => ResponseUncertainty::None,
            },
        )
    }

    /// Runs the Pulse-witness refutation suite for an `InterventionResponse`
    /// when the study refutes, otherwise notes that scalar refuters are skipped.
    fn refute_temporal_class_posterior_response(
        &self,
        inputs: &ClassPosteriorInputs<'_>,
        weighted: &[(u64, f64, CausalResponse)],
        aggregation_policy: StructuralAggregationPolicy,
        diagnostics: &mut Vec<Diagnostic>,
    ) -> Result<Vec<antecedent_validate::RefutationReport>, CausalError> {
        let ClassPosteriorInputs { data, gp, query, vars, ctx } = *inputs;
        let is_intervention =
            matches!(query.functional, ResponseFunctional::InterventionResponse { .. });
        let (refutations, refute_diags) = if is_intervention
            && (!matches!(self.settings.refute, RefuteSuite::None)
                || !self.declared.custom_validators.is_empty())
        {
            refute_temporal_class_graph_posterior_intervention(
                self,
                data,
                query,
                GraphPosteriorScope { posterior: gp, vars },
                weighted,
                aggregation_policy,
                ctx,
            )?
        } else {
            if !matches!(self.settings.refute, RefuteSuite::None) {
                diagnostics.push(Diagnostic::new(
                    "refute.temporal_response.skipped",
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Info,
                    "scalar ATE refuters are not applicable to a function-valued temporal response",
                ));
            }
            (Vec::new(), Vec::new())
        };
        diagnostics.extend(refute_diags);
        Ok(refutations)
    }
}

/// Identified, unidentified and unevaluable shares of the posterior mass.
#[derive(Clone, Copy)]
struct ClassPosteriorMasses {
    identified: f64,
    unidentified: f64,
    failed: f64,
}

impl ClassPosteriorMasses {
    /// The shares of the total posterior weight the evaluated atoms, the
    /// unidentified atoms and the failed estimations hold.
    fn of(
        weighted: &[(u64, f64, CausalResponse)],
        gp: &GraphPosterior,
        (unidentified_mass, failed_mass): (f64, f64),
    ) -> Self {
        let identified_mass = weighted.iter().map(|(_, weight, _)| *weight).sum::<f64>();
        let total_mass = gp.weights().iter().sum::<f64>();
        Self {
            identified: identified_mass / total_mass,
            unidentified: unidentified_mass / total_mass,
            failed: failed_mass / total_mass,
        }
    }
}

/// The posterior-probability mixture of the class atoms.
fn class_posterior_mixture(
    atoms: Vec<StructuralResponseAtom>,
    masses: ClassPosteriorMasses,
    (structural_set, conditional): (
        Option<antecedent_core::ResponseEnvelope>,
        Option<ResponseValue>,
    ),
    full_mass_scope: bool,
) -> StructuralResponseMixture {
    StructuralResponseMixture {
        weight_basis: StructuralWeightBasis::PosteriorProbability,
        atoms,
        identified_mass: masses.identified,
        unidentified_mass: masses.unidentified,
        unevaluable_mass: masses.failed,
        subsampled_out_mass: 0.0,
        identified_set: structural_set,
        identified_set_interval: None,
        conditional_on_identified: conditional,
        full_mass_scope,
        truncated_atoms: 0,
        aggregation_policy: None,
    }
}

/// The tag a class graph-posterior response names its atoms by, or the
/// refusal when the atoms are not CPDAG or PAG completions.
fn temporal_class_posterior_tag(
    kind: antecedent_discovery::GraphPosteriorAtomKind,
) -> Result<&'static str, CausalError> {
    match kind {
        antecedent_discovery::GraphPosteriorAtomKind::Cpdag => Ok("temporal_cpdag"),
        antecedent_discovery::GraphPosteriorAtomKind::Pag => Ok("temporal_pag"),
        _ => Err(CausalError::Compile {
            message: "temporal class graph-posterior response requires Cpdag or Pag \
                      atom_kind"
                .into(),
        }),
    }
}

/// The diagnostics of a class graph-posterior response before refutation:
/// the evaluation note, the structural aggregation and the withheld band.
fn class_posterior_diagnostics(
    class_tag: &str,
    aggregation_policy: StructuralAggregationPolicy,
    masses: ClassPosteriorMasses,
    multi_atom: bool,
) -> Vec<Diagnostic> {
    let mut diagnostics = vec![Diagnostic::new(
        "estimate.response.temporal_class_graph_posterior",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        format!(
            "each {class_tag} posterior atom was evaluated with the existing temporal class \
             response envelope; outer weights are frozen posterior probabilities"
        ),
    )];
    push_graph_posterior_structural_aggregation_diagnostic(
        &mut diagnostics,
        aggregation_policy,
        masses.identified,
        masses.unidentified,
        masses.failed,
        0.0,
    );
    if multi_atom {
        diagnostics.push(Diagnostic::new(
            "estimate.response.graph_posterior.uncertainty_withheld",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Warning,
            "multi-atom temporal class-posterior response withholds an aggregate band: \
             atom surfaces are mixed by frozen graph weight",
        ));
    }
    diagnostics
}

/// The evaluated responses, structural atoms and contributing estimands the
/// class posterior mix reads.
type ClassPosteriorAtoms<'a> =
    (&'a [(u64, f64, CausalResponse)], &'a [StructuralResponseAtom], &'a [IdentifiedEstimand]);

/// The series, posterior, query and context of a class graph-posterior response.
#[derive(Clone, Copy)]
struct ClassPosteriorInputs<'a> {
    data: &'a TimeSeriesData,
    gp: &'a GraphPosterior,
    query: &'a ResponseQuery,
    vars: &'a [VariableId],
    ctx: &'a ExecutionContext,
}

/// One posterior atom's class response run: its key and frozen weight, and
/// the result, absent when it kept its mass without re-estimation, or failed.
type ClassPosteriorAtomRun = (u64, f64, Option<Result<StudyResult, ()>>);

/// The class posterior atoms folded into structural atoms and the evaluated
/// responses, with the unidentified and unevaluable mass.
struct ClassPosteriorFold {
    weighted: Vec<(u64, f64, CausalResponse)>,
    atoms: Vec<StructuralResponseAtom>,
    contributing: Vec<IdentifiedEstimand>,
    failed_mass: f64,
    unidentified_mass: f64,
    any_partial: bool,
    primary: Option<(IdentificationResult, IdentifiedEstimand)>,
}

/// The mixed class posterior response with the products the structural
/// mixture and diagnostics are built from.
struct ClassPosteriorMix {
    response: CausalResponse,
    identification: IdentificationResult,
    aggregation_policy: StructuralAggregationPolicy,
    structural_set: Option<antecedent_core::ResponseEnvelope>,
    conditional: Option<ResponseValue>,
}

/// Folds each atom's class response into a structural atom, classifying it as
/// unidentified, unevaluable or contributing.
fn fold_temporal_class_posterior_atoms(mapped: Vec<ClassPosteriorAtomRun>) -> ClassPosteriorFold {
    let mut fold = ClassPosteriorFold {
        weighted: Vec::new(),
        atoms: Vec::new(),
        contributing: Vec::new(),
        failed_mass: 0.0,
        unidentified_mass: 0.0,
        any_partial: false,
        primary: None,
    };
    for (key, weight, outcome) in mapped {
        match outcome {
            None | Some(Err(())) => {
                fold.unidentified_mass += weight;
                fold.atoms.push(bare_class_atom(key, weight, IdentificationStatus::NotIdentified));
            }
            Some(Ok(result)) => {
                let Some(response) = result.payload.into_response() else {
                    fold.failed_mass += weight;
                    fold.atoms.push(bare_class_atom(key, weight, result.identification.status));
                    continue;
                };
                let Some(value) =
                    response_identified_value(&response).map(collapse_singleton_envelope)
                else {
                    fold.failed_mass += weight;
                    fold.atoms.push(StructuralResponseAtom {
                        response: Some(response),
                        ..bare_class_atom(key, weight, result.identification.status)
                    });
                    continue;
                };
                fold.any_partial |= matches!(value, ResponseValue::Envelope(_))
                    || matches!(
                        result.identification.status,
                        IdentificationStatus::PartiallyIdentified
                    ) && !matches!(
                        value,
                        ResponseValue::Scalar(_) | ResponseValue::Surface { .. }
                    );
                if fold.primary.is_none() {
                    fold.primary = Some((result.identification.clone(), result.estimand.clone()));
                }
                fold.contributing.push(result.estimand);
                fold.atoms.push(StructuralResponseAtom {
                    response: Some(response.clone()),
                    value: Some(value),
                    ..bare_class_atom(key, weight, result.identification.status)
                });
                fold.weighted.push((key, weight, response));
            }
        }
    }
    fold
}

/// The status-ladder estimate of the mixed class response: per-atom values
/// when graph dependent, the identified set when set-identified, the mixed
/// value when point identified.
fn class_posterior_estimate(
    weighted: &[(u64, f64, CausalResponse)],
    aggregation_policy: StructuralAggregationPolicy,
    (graph_dependent, status): (bool, IdentificationStatus),
    (structural_set, mixed_value): (
        Option<&antecedent_core::ResponseEnvelope>,
        Option<&ResponseValue>,
    ),
) -> Result<ResponseIdentification, CausalError> {
    let per_atom_values = || {
        weighted
            .iter()
            .filter_map(|(key, _, response)| {
                determinate_response_value(response).map(|value| (*key, value))
            })
            .collect()
    };
    Ok(if graph_dependent {
        ResponseIdentification::GraphDependent(per_atom_values())
    } else if matches!(aggregation_policy, StructuralAggregationPolicy::IdentifiedSetEnvelope)
        || status == IdentificationStatus::PartiallyIdentified
    {
        // A set-identified atom whose set is one point keeps its status and its
        // set: the status ladder never promotes it to point identification.
        ResponseIdentification::PartiallyIdentified(ResponseValue::Envelope(
            structural_set.cloned().ok_or_else(|| CausalError::Compile {
                message: "temporal class graph-posterior identified set missing envelope".into(),
            })?,
        ))
    } else if let Some(value) = mixed_value {
        ResponseIdentification::PointIdentified(value.clone())
    } else {
        ResponseIdentification::GraphDependent(per_atom_values())
    })
}

fn collapse_singleton_envelope(value: ResponseValue) -> ResponseValue {
    let ResponseValue::Envelope(envelope) = &value else {
        return value;
    };
    if envelope.lower.len() != envelope.upper.len()
        || envelope
            .lower
            .iter()
            .zip(envelope.upper.iter())
            .any(|(lo, hi)| lo.to_bits() != hi.to_bits())
    {
        return value;
    }
    ResponseValue::Surface {
        grid: Arc::clone(&envelope.grid),
        dimension: envelope.dimension,
        mean: Arc::clone(&envelope.lower),
    }
}

fn determinate_response_value(response: &CausalResponse) -> Option<ResponseValue> {
    match collapse_singleton_envelope(response_identified_value(response)?) {
        value @ (ResponseValue::Scalar(_) | ResponseValue::Surface { .. }) => Some(value),
        _ => None,
    }
}

fn union_identified_values(values: &[&ResponseValue]) -> Option<antecedent_core::ResponseEnvelope> {
    let mut envelope: Option<antecedent_core::ResponseEnvelope> = None;
    for value in values {
        let next = match value {
            ResponseValue::Envelope(item) => item.clone(),
            ResponseValue::Surface { grid, dimension, mean } => antecedent_core::ResponseEnvelope {
                grid: Arc::clone(grid),
                dimension: *dimension,
                lower: Arc::clone(mean),
                upper: Arc::clone(mean),
            },
            ResponseValue::Scalar(item) => scalar_identified_set(*item, *item),
            _ => return None,
        };
        envelope = Some(match envelope {
            None => next,
            Some(mut acc) => {
                if acc.grid.as_ref() != next.grid.as_ref()
                    || acc.dimension != next.dimension
                    || acc.lower.len() != next.lower.len()
                {
                    return None;
                }
                let lower = acc
                    .lower
                    .iter()
                    .zip(next.lower.iter())
                    .map(|(a, b)| a.min(*b))
                    .collect::<Vec<_>>();
                let upper = acc
                    .upper
                    .iter()
                    .zip(next.upper.iter())
                    .map(|(a, b)| a.max(*b))
                    .collect::<Vec<_>>();
                acc.lower = Arc::from(lower);
                acc.upper = Arc::from(upper);
                acc
            }
        });
    }
    envelope
}

/// Joint circular-block bootstrap SE for a scalar temporal `InterventionResponse` pooled over the
/// class-posterior atoms. The scalar intervention-response value is the pulse effect at the horizon
/// (`response_pulse_witness`), so each contributing atom is reconstructed into its pulse
/// `TemporalAtomDesign` exactly as the envelope refuter does, and the atoms are resampled jointly on
/// their shared time window — the same block-bootstrap the temporal class effect path uses. Returns
/// `None` (leaving the aggregate point-only) unless a Frequentist bootstrap is requested, the
/// functional is `InterventionResponse`, and at least two designs reconstruct. The estimator SE
/// this reports must be calibrated before it is trusted.
fn temporal_class_response_pooled_block_se(
    study: &Study,
    data: &TimeSeriesData,
    gp: &GraphPosterior,
    vars: &[VariableId],
    query: &ResponseQuery,
    weighted: &[(u64, f64, CausalResponse)],
    ctx: &ExecutionContext,
) -> Result<Option<f64>, CausalError> {
    if !study.settings.inference.is_frequentist() || study.settings.bootstrap_replicates == 0 {
        return Ok(None);
    }
    if !matches!(query.functional, antecedent_core::ResponseFunctional::InterventionResponse { .. })
    {
        return Ok(None);
    }
    let pulse = response_pulse_witness(query)?;
    let lag_masks = gp.lag_masks().ok_or_else(|| CausalError::Compile {
        message: "temporal class posterior missing per-atom lag masks".into(),
    })?;
    let max_lag = gp.max_lag().ok_or_else(|| CausalError::Compile {
        message: "temporal class posterior missing max_lag".into(),
    })?;
    let mut designs = Vec::new();
    let mut weights = Vec::new();
    let mut indexers = Vec::new();
    for (key, weight, _) in weighted {
        let i = usize::try_from(*key).map_err(|_| CausalError::Compile {
            message: "temporal class graph-posterior response SE: atom key overflow".into(),
        })?;
        let Some(bundle) =
            identify_posterior_atom_pulse(study, (gp, vars), (lag_masks, max_lag), i, &pulse, ctx)?
        else {
            continue;
        };
        for (case, indexer) in
            bundle.envelope.envelope.cases.iter().zip(bundle.envelope.indexers.iter())
        {
            if !identification_status_ok_for_case(case.result.status)
                || case.result.estimands.is_empty()
            {
                continue;
            }
            let Ok(estimand) = select_estimand(&case.result, EstimatorId::TemporalLinearAdjustment)
            else {
                continue;
            };
            let Ok(design) = TemporalAtomDesign::linear(
                data,
                &estimand,
                &pulse,
                indexer,
                study.declared.split.as_ref(),
                ctx,
            ) else {
                continue;
            };
            designs.push(design);
            weights.push(*weight * case.weight.0);
            indexers.push(indexer.clone());
        }
    }
    if designs.len() < 2 {
        return Ok(None);
    }
    let design_refs: Vec<&TemporalAtomDesign> = designs.iter().collect();
    let block = shared_circular_block_mixture_se(
        &design_refs,
        &weights,
        temporal_class_block_span(indexers.iter()),
        study.settings.bootstrap_replicates,
        0x9E3D_7A11,
        ctx,
    );
    Ok(block.se.is_finite().then_some(block.se))
}

/// The class envelope of a posterior atom's reconstructed graph for the Pulse
/// witness, or `None` when the atom cannot be rebuilt or identified.
fn identify_posterior_atom_pulse(
    study: &Study,
    (gp, vars): (&GraphPosterior, &[VariableId]),
    (lag_masks, max_lag): (&[antecedent_discovery::LagMask], u32),
    i: usize,
    pulse: &TemporalEffectQuery,
    ctx: &ExecutionContext,
) -> Result<Option<Arc<crate::analysis::prepared::CachedTemporalClassIdentification>>, CausalError>
{
    let mark = gp.mark_atom(i);
    let Ok(graph) = reconstruct_temporal_class_atom(
        gp.atom_kind(),
        &gp.atoms()[i],
        &lag_masks[i],
        &mark,
        gp.n_vars(),
        max_lag,
        vars,
    ) else {
        return Ok(None);
    };
    let mut atom_study = study.clone();
    atom_study.graph_posterior = None;
    atom_study.graph = Arc::new(graph);
    atom_study.query = CausalQuery::TemporalEffect(pulse.clone());
    atom_study.clear_identification();
    match atom_study.identify_temporal_class(IdentifierId::GeneralizedAdjustment, pulse, ctx) {
        Ok(bundle) => Ok(Some(bundle)),
        Err(error @ (CausalError::Cancelled { .. } | CausalError::Resource { .. })) => Err(error),
        Err(_) => Ok(None),
    }
}

fn reconstruct_temporal_class_atom(
    kind: antecedent_discovery::GraphPosteriorAtomKind,
    adjacency: &antecedent_discovery::AdjacencyMask,
    lag_mask: &antecedent_discovery::LagMask,
    mark: &antecedent_discovery::AdjacencyMask,
    n_vars: usize,
    max_lag: u32,
    variables: &[VariableId],
) -> Result<AcceptedGraph, CausalError> {
    match kind {
        antecedent_discovery::GraphPosteriorAtomKind::Cpdag => {
            let cpdag =
                temporal_cpdag_from_dbn_atoms(adjacency, lag_mask, n_vars, max_lag, variables)
                    .map_err(CausalError::rejected)?;
            Ok(AcceptedGraph::from(cpdag))
        }
        antecedent_discovery::GraphPosteriorAtomKind::Pag => {
            let pag =
                temporal_pag_from_dbn_atoms(adjacency, lag_mask, mark, n_vars, max_lag, variables)
                    .map_err(CausalError::rejected)?;
            Ok(AcceptedGraph::from(pag))
        }
        _ => Err(CausalError::Compile {
            message: "temporal class graph-posterior response requires Cpdag or Pag atom_kind"
                .into(),
        }),
    }
}

fn response_pulse_witness(query: &ResponseQuery) -> Result<TemporalEffectQuery, CausalError> {
    let temporal = query.temporal.as_ref().ok_or_else(|| CausalError::Compile {
        message:
            "temporal class graph-posterior InterventionResponse requires TemporalResponseSpec"
                .into(),
    })?;
    let (treatment, outcome) = response_primary_pair(&query.functional)?;
    let horizon = *temporal.horizons.first().ok_or_else(|| CausalError::Compile {
        message:
            "temporal class graph-posterior InterventionResponse requires at least one horizon"
                .into(),
    })?;
    let active = match &query.functional {
        ResponseFunctional::InterventionResponse { interventions, .. } => {
            interventions.first().cloned().ok_or_else(|| CausalError::Compile {
                message: "InterventionResponse has no intervention coordinate".into(),
            })?
        }
        _ => Intervention::set(treatment, antecedent_core::Value::f64(1.0)),
    };
    Ok(TemporalEffectQuery {
        active,
        ..super::temporal_path::unit_contrast_query(
            temporal,
            (treatment, outcome),
            horizon,
            query.target_population.clone(),
        )
    })
}

fn refute_temporal_class_graph_posterior_intervention(
    study: &Study,
    data: &TimeSeriesData,
    query: &ResponseQuery,
    scope: GraphPosteriorScope<'_>,
    weighted: &[(u64, f64, CausalResponse)],
    policy: StructuralAggregationPolicy,
    ctx: &ExecutionContext,
) -> Result<(Vec<antecedent_validate::RefutationReport>, Vec<Diagnostic>), CausalError> {
    let GraphPosteriorScope { posterior: gp, vars } = scope;
    if matches!(policy, StructuralAggregationPolicy::GraphDependentAtoms) {
        return Ok((
            Vec::new(),
            vec![Diagnostic::new(
                "refute.envelope.temporal_class_graph_posterior",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "outer scalar mix skipped because StructuralAggregationPolicy is \
                 graph_dependent_atoms — per-atom class envelopes are retained",
            )],
        ));
    }
    let pulse = response_pulse_witness(query)?;
    let lag_masks = gp.lag_masks().ok_or_else(|| CausalError::Compile {
        message: "temporal class posterior missing per-atom lag masks".into(),
    })?;
    let max_lag = gp.max_lag().ok_or_else(|| CausalError::Compile {
        message: "temporal class posterior missing max_lag".into(),
    })?;
    let mut refute_atoms = Vec::new();
    for (key, weight, _) in weighted {
        let i = usize::try_from(*key).map_err(|_| CausalError::Compile {
            message: "temporal class graph-posterior IR refuter: atom key overflow".into(),
        })?;
        let Some(bundle) =
            identify_posterior_atom_pulse(study, (gp, vars), (lag_masks, max_lag), i, &pulse, ctx)?
        else {
            continue;
        };
        for (case_index, (case, indexer)) in
            bundle.envelope.envelope.cases.iter().zip(bundle.envelope.indexers.iter()).enumerate()
        {
            if !identification_status_ok_for_case(case.result.status)
                || case.result.estimands.is_empty()
            {
                continue;
            }
            let Ok(estimand) = select_estimand(&case.result, EstimatorId::TemporalLinearAdjustment)
            else {
                continue;
            };
            let Ok(design) = TemporalAtomDesign::linear(
                data,
                &estimand,
                &pulse,
                indexer,
                study.declared.split.as_ref(),
                ctx,
            ) else {
                continue;
            };
            refute_atoms.push(EnvelopeRefuteAtom {
                key: (*key << 32) | u64::try_from(case_index).unwrap_or(u64::MAX),
                weight: *weight * case.weight.0,
                estimand,
                indexer: Some(indexer.clone()),
                original: design.effect_estimate(case.result.required_assumptions.clone()),
            });
        }
    }
    if refute_atoms.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }
    let tabular = TabularData::new(data.storage().clone());
    let ate = AverageEffectQuery::binary_ate(pulse.treatment, pulse.outcome);
    let (reports, mut notes) = EnvelopeRefutation {
        data: &tabular,
        query: &ate,
        suite: study.settings.refute,
        estimator: EstimatorId::TemporalLinearAdjustment.as_str(),
        custom: &study.declared.custom_validators,
        temporal: Some(TemporalRefutation {
            query: &pulse,
            split: study.declared.split.as_ref(),
            time_index: data.time_index(),
            gaps: false,
        }),
    }
    .run(&refute_atoms, &mut EstimationWorkspace::default(), ctx)?;
    notes.insert(
        0,
        Diagnostic::new(
            "refute.envelope.temporal_class_graph_posterior",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "InterventionResponse cheap/full run the TemporalDag Pulse-native suite on each \
             contributing class-posterior atom against that atom's own Pulse witness and mix \
             by frozen graph weight; the mixed check passes only if every atom passes",
        ),
    );
    Ok((reports, notes))
}
