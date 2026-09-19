//! ADMG graph-posterior AverageEffect: general.id + functional.effect per atom.
//!
//! Atoms are single graphs, not MEC completions. Outer mix uses posterior
//! weights and [`StructuralAggregationPolicy`]. Unidentified mass is retained.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use crate::analysis::prepared::CachedGraphPosteriorIdentification;
use crate::result::{
    StructuralAggregationPolicy, StructuralResponseAtom, StructuralResponseMixture,
};

struct AdmgAtomEval {
    key: u64,
    status: IdentificationStatus,
    estimand: IdentifiedEstimand,
    estimate: EffectEstimate,
    posterior: Option<CausalPosterior>,
    refute_atoms: Vec<EnvelopeRefuteAtom>,
}

impl super::Study {
    /// Mix ADMG posterior atoms through `identify_admg` + `functional.effect`.
    ///
    /// Never routes through CPDAG completion enumeration or `bayesian.gcomp`.
    pub(super) fn execute_admg_graph_posterior(
        &self,
        data: &TabularData,
        gp: &GraphPosterior,
        query: &AverageEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        if matches!(self.query, CausalQuery::ConditionalEffect(_)) {
            return Err(CausalError::Unsupported {
                message: "ConditionalEffect graph-posterior mixing is licensed only for DAG atoms",
            });
        }
        if !matches!(gp.atom_kind, antecedent_discovery::GraphPosteriorAtomKind::Admg) {
            return Err(CausalError::Compile {
                message: "execute_admg_graph_posterior requires Admg posterior atoms".into(),
            });
        }
        let (identified, identify_cached) =
            if let Some(cache) = self.graph_posterior_identification_cache.as_deref() {
                (cache.clone(), true)
            } else {
                (
                    crate::analysis::prepared::build_graph_posterior_identification_cache(
                        gp, query, ctx,
                    )?,
                    false,
                )
            };
        if identified.atoms.is_empty() && identified.graphs.identified_mass() <= 0.0 {
            return Err(CausalError::Compile {
                message: "ADMG graph-posterior envelope: no identified ADMG atoms".into(),
            });
        }

        let mut evals = Vec::new();
        for atom in identified.atoms.iter() {
            let (estimate, posterior) = self.estimate_functional_effect(
                data,
                &atom.estimand,
                &atom.identification,
                &[query.treatment, query.outcome],
                ctx,
            )?;
            let weight = identified_weight_for_key(&identified.graphs, atom.key);
            let refute_atoms = if estimate.ate.is_finite() {
                vec![EnvelopeRefuteAtom {
                    key: atom.key,
                    weight,
                    estimand: atom.estimand.clone(),
                    indexer: None,
                    original: estimate.clone(),
                }]
            } else {
                Vec::new()
            };
            evals.push(AdmgAtomEval {
                key: atom.key,
                status: atom.identification.status,
                estimand: atom.estimand.clone(),
                estimate,
                posterior,
                refute_atoms,
            });
        }

        let unidentified_mass = identified.graphs.unidentified_mass();
        let mixed = mix_admg_posterior_evals(&identified, &evals, unidentified_mass)?;
        let mut identification = mixed.identification;
        identification.status = if unidentified_mass > 1e-12
            || matches!(mixed.policy, StructuralAggregationPolicy::GraphDependentAtoms)
        {
            IdentificationStatus::GraphDependent
        } else if matches!(mixed.policy, StructuralAggregationPolicy::IdentifiedSetEnvelope) {
            IdentificationStatus::PartiallyIdentified
        } else {
            identification.status
        };

        let mut diagnostics = identification.diagnostics.clone();
        if matches!(self.inference, InferenceMode::Bayesian(_)) && mixed.posterior.is_none() {
            diagnostics.push(Diagnostic::new(
                "estimate.graph_posterior.posterior_withheld",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Warning,
                "aggregate posterior withheld: a graph- or completion-conditional posterior \
                 does not describe the full weighted target; retained atom posteriors are conditional",
            ));
        }
        diagnostics.push(Diagnostic::new(
            "estimate.graph_posterior.admg_functional",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "each ADMG posterior atom was identified with general.id and estimated with \
             functional.effect; outer weights are posterior probabilities; ADMG atoms are \
             single graphs, not MEC completions",
        ));
        diagnostics.push(
            Diagnostic::new(
                "estimate.graph_posterior.structural_aggregation",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                format!(
                    "policy={}; weight_basis=posterior_probability; completion enumeration is \
                     not posterior probability; identified_mass={}; unidentified_mass={}",
                    mixed.policy.as_str(),
                    mixed.mixture.identified_mass,
                    mixed.mixture.unidentified_mass
                ),
            )
            .with_fields(mass_fields(
                Some(mixed.mixture.identified_mass),
                mixed.mixture.unidentified_mass,
            )),
        );
        diagnostics.push(overlap_diagnostic(mixed.estimate.overlap));

        let (refutations, refute_diagnostics) =
            refute_admg_graph_posterior(self, data, query, &identified, &evals, mixed.policy, ctx)?;
        diagnostics.extend(refute_diagnostics);

        let algo =
            physical.logical.record.discovery_algorithm.as_deref().unwrap_or("graph_posterior");
        Ok(self.finish_identified_execute(IdentifiedExecuteFinish {
            physical,
            identification,
            estimand: mixed.estimand,
            estimate: mixed.estimate,
            identifier_id: IdentifierId::GeneralId,
            estimator_id: EstimatorId::FunctionalEffect,
            treatment: query.treatment,
            outcome: query.outcome,
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
                identify_provenance: Some(provenance_ids("discover.graph_posterior", algo)),
                estimate_provenance: Some(provenance_ids(
                    "estimate.admg_graph_posterior",
                    mixed.policy.as_str(),
                )),
                posterior: mixed.posterior,
                diagnostics: Some(diagnostics),
                structural_response: Some(mixed.mixture),
                ..Default::default()
            },
        }))
    }
}

struct MixedAdmgPosterior {
    policy: StructuralAggregationPolicy,
    identification: IdentificationResult,
    estimand: IdentifiedEstimand,
    estimate: EffectEstimate,
    posterior: Option<CausalPosterior>,
    mixture: StructuralResponseMixture,
}

fn mix_admg_posterior_evals(
    identified: &CachedGraphPosteriorIdentification,
    evals: &[AdmgAtomEval],
    unidentified_mass: f64,
) -> Result<MixedAdmgPosterior, CausalError> {
    let mut atoms = Vec::new();
    let mut identified_weight = 0.0;
    let mut unevaluable_weight = 0.0;
    let mut mixable = 0.0;
    let mut weighted = 0.0;
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    let mut contributing = Vec::new();
    let mut any_partial = false;
    let mut primary_identification = None;
    let mut primary_estimand = None;
    let mut assumptions = antecedent_core::AssumptionSet::default();
    let mut atom_posteriors = Vec::new();

    for eval in evals {
        let w = identified_weight_for_key(&identified.graphs, eval.key);
        let value = eval.estimate.ate.is_finite().then_some(eval.estimate.ate);
        if let Some(v) = value {
            identified_weight += w;
            lo = lo.min(v);
            hi = hi.max(v);
            contributing.push(eval.estimand.clone());
            if primary_estimand.is_none() {
                primary_estimand = Some(eval.estimand.clone());
                primary_identification = Some(eval_identification(eval, identified));
                assumptions = eval.estimate.assumptions.clone();
            }
            any_partial |= matches!(eval.status, IdentificationStatus::PartiallyIdentified);
            if let Some(p) = eval.posterior.clone() {
                atom_posteriors.push((w, p));
            }
            mixable += w;
            weighted += w * v;
        } else if w > 0.0 {
            unevaluable_weight += w;
        }
        atoms.push(StructuralResponseAtom {
            graph_key: eval.key,
            weight: w,
            status: eval.status,
            value: value.map(antecedent_core::ResponseValue::Scalar),
            posterior: eval.posterior.clone(),
            response: None,
        });
    }

    for (key, weight, flag) in identified
        .graphs
        .graph_keys
        .iter()
        .zip(identified.graphs.weights.iter())
        .zip(identified.graphs.identified.iter())
        .map(|((k, w), f)| (*k, *w, *f))
    {
        if flag != GraphIdentFlag::Unidentified {
            continue;
        }
        if atoms.iter().any(|atom| atom.graph_key == key) {
            continue;
        }
        atoms.push(StructuralResponseAtom {
            graph_key: key,
            weight,
            status: IdentificationStatus::NotIdentified,
            value: None,
            posterior: None,
            response: None,
        });
    }

    let total = identified.graphs.total_weight();
    let identified_mass = if total > 0.0 { identified_weight / total } else { 0.0 };
    let unevaluable_mass = if total > 0.0 { unevaluable_weight / total } else { 0.0 };
    let refs: Vec<&IdentifiedEstimand> = contributing.iter().collect();
    let policy = if refs.is_empty() && any_partial {
        StructuralAggregationPolicy::IdentifiedSetEnvelope
    } else {
        resolve_structural_aggregation(&refs, any_partial)
    };
    let identified_set = (lo.is_finite() && hi.is_finite()).then(|| scalar_identified_set(lo, hi));
    let mixable_scalar =
        matches!(policy, StructuralAggregationPolicy::SameEstimandWeightedMean) && mixable > 0.0;
    let ate = if mixable_scalar { weighted / mixable } else { f64::NAN };
    let conditional_on_identified =
        mixable_scalar.then(|| antecedent_core::ResponseValue::Scalar(ate));
    let mixture = StructuralResponseMixture {
        weight_basis: crate::result::StructuralWeightBasis::PosteriorProbability,
        atoms,
        identified_mass,
        unidentified_mass,
        unevaluable_mass,
        subsampled_out_mass: 0.0,
        identified_set,
        identified_set_interval: None,
        conditional_on_identified,
        full_mass_scope: true,
        truncated_atoms: 0,
    };
    let identification = primary_identification.ok_or_else(|| CausalError::Compile {
        message: format!(
            "ADMG graph-posterior mix: no evaluable atom values (evals={}, mixable={mixable})",
            evals.len()
        ),
    })?;
    let estimand = primary_estimand
        .or_else(|| identification.estimands.first().cloned())
        .ok_or_else(|| CausalError::Compile {
            message: "ADMG graph-posterior envelope: missing estimand".into(),
        })?;
    let estimate = EffectEstimate::new(ate, f64::NAN, assumptions, OverlapPolicy::ExplicitOverride);
    // A component posterior is not the posterior of a weighted aggregate.
    // Retain it at the top level only when it covers the entire target.
    let posterior = if mixable_scalar
        && contributing.len() == 1
        && atom_posteriors.len() == 1
        && unidentified_mass == 0.0
        && unevaluable_mass == 0.0
        && mixture.full_mass_scope
    {
        atom_posteriors.into_iter().next().map(|(_, p)| p)
    } else {
        None
    };
    Ok(MixedAdmgPosterior { policy, identification, estimand, estimate, posterior, mixture })
}

fn eval_identification(
    eval: &AdmgAtomEval,
    identified: &CachedGraphPosteriorIdentification,
) -> IdentificationResult {
    identified.atoms.iter().find(|atom| atom.key == eval.key).map_or_else(
        || identified.atoms[0].identification.clone(),
        |atom| atom.identification.clone(),
    )
}

/// Per-atom functional.effect refuters; outer mix by frozen posterior weight when
/// [`StructuralAggregationPolicy`] is `SameEstimandWeightedMean`.
fn refute_admg_graph_posterior(
    study: &Study,
    data: &TabularData,
    query: &AverageEffectQuery,
    identified: &CachedGraphPosteriorIdentification,
    evals: &[AdmgAtomEval],
    policy: StructuralAggregationPolicy,
    ctx: &ExecutionContext,
) -> Result<(Vec<antecedent_validate::RefutationReport>, Vec<Diagnostic>), CausalError> {
    if matches!(study.refute, RefuteSuite::None) && study.custom_validators.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }
    let mix_scalar = matches!(policy, StructuralAggregationPolicy::SameEstimandWeightedMean);
    let mut refute_ws = EstimationWorkspace::default();
    let mut per_atom = Vec::new();
    let mut diagnostics = Vec::new();
    for eval in evals {
        if eval.refute_atoms.is_empty() {
            continue;
        }
        let weight = identified_weight_for_key(&identified.graphs, eval.key);
        if weight <= 0.0 {
            continue;
        }
        let (reports, notes) = run_envelope_effect_refuters(
            data,
            query,
            &eval.refute_atoms,
            &mut refute_ws,
            ctx,
            study.refute,
            EstimatorId::FunctionalEffect.as_str(),
            &study.custom_validators,
            None,
            study.split.as_ref(),
            None,
        )?;
        diagnostics.extend(
            notes.into_iter().filter(|d| d.code.as_ref() != "refute.envelope.effect_mixture"),
        );
        per_atom.push((weight, reports));
    }
    let inner_keys: String = evals
        .iter()
        .filter(|eval| !eval.refute_atoms.is_empty())
        .map(|eval| format!("{:x}", eval.key))
        .collect::<Vec<_>>()
        .join(",");
    diagnostics.push(Diagnostic::new(
        "refute.envelope.admg_posterior",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        if mix_scalar {
            format!(
                "inner: effect refuters evaluated each ADMG posterior atom [{inner_keys}] against \
                 that atom's own functional.effect estimate; ADMG atoms are single graphs, not \
                 MEC completions; outer: those atom reports mix by frozen posterior graph weight \
                 and pass only if every contributing atom passes"
            )
        } else {
            format!(
                "inner: effect refuters evaluated each ADMG posterior atom [{inner_keys}] against \
                 that atom's own functional.effect estimate; ADMG atoms are single graphs, not \
                 MEC completions; outer scalar mix skipped because StructuralAggregationPolicy is \
                 {} — per-atom reports are retained and are not compared against a pooled \
                 mixture ATE",
                policy.as_str()
            )
        },
    ));
    let reports = if mix_scalar {
        mix_admg_posterior_refutation_reports(&per_atom)
    } else {
        per_atom.into_iter().flat_map(|(_, reports)| reports).collect()
    };
    Ok((reports, diagnostics))
}

fn mix_admg_posterior_refutation_reports(
    per_atom: &[(f64, Vec<antecedent_validate::RefutationReport>)],
) -> Vec<antecedent_validate::RefutationReport> {
    let mut order = Vec::new();
    let mut by_refuter: std::collections::HashMap<
        Arc<str>,
        Vec<(f64, antecedent_validate::RefutationReport)>,
    > = std::collections::HashMap::new();
    for (weight, reports) in per_atom {
        for report in reports {
            let bucket = by_refuter.entry(Arc::clone(&report.refuter)).or_insert_with(|| {
                order.push(Arc::clone(&report.refuter));
                Vec::new()
            });
            bucket.push((*weight, report.clone()));
        }
    }
    let mut out = Vec::with_capacity(order.len());
    for id in order {
        let Some(items) = by_refuter.get(&id) else {
            continue;
        };
        let borrowed: Vec<(f64, &antecedent_validate::RefutationReport)> =
            items.iter().map(|(w, report)| (*w, report)).collect();
        if let Some(mixed) = antecedent_validate::RefutationReport::mixture_weighted(&borrowed) {
            out.push(mixed);
        }
    }
    out
}
