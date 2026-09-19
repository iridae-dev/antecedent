//! CPDAG/PAG graph-posterior AverageEffect: class envelope per atom, then policy.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use crate::analysis::prepared::{
    CachedClassPosteriorAtomIdentification, CachedGraphPosteriorIdentification,
};
use crate::result::{
    StructuralAggregationPolicy, StructuralResponseAtom, StructuralResponseMixture,
};

struct ClassAtomEval {
    key: u64,
    status: IdentificationStatus,
    estimand: Option<IdentifiedEstimand>,
    estimate: Option<EffectEstimate>,
    posterior: Option<CausalPosterior>,
    identified_set: Option<(f64, f64)>,
    partial: bool,
    refute_atoms: Vec<EnvelopeRefuteAtom>,
}

impl super::Study {
    /// Mix CPDAG/PAG posterior atoms through the existing class ATE evaluator.
    pub(super) fn execute_class_graph_posterior(
        &self,
        data: &TabularData,
        gp: &GraphPosterior,
        query: &AverageEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let conditional = matches!(self.query, CausalQuery::ConditionalEffect(_));
        let class_tag = gp.atom_kind.as_str().to_ascii_lowercase();
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
        if identified.class_atoms.is_empty() && identified.graphs.identified_mass() <= 0.0 {
            return Err(CausalError::Compile {
                message: "class graph-posterior envelope: no identified class atoms".into(),
            });
        }

        let evals = ctx.map_indexed(identified.class_atoms.len(), |i, inner| {
            let atom = &identified.class_atoms[i];
            match &self.inference {
                InferenceMode::Frequentist => {
                    evaluate_class_atom_frequentist(self, data, query, atom, conditional, inner)
                }
                InferenceMode::Bayesian(_) => {
                    evaluate_class_atom_bayesian(self, data, query, atom, conditional, inner)
                }
            }
        })?;

        let unidentified_mass = identified.graphs.unidentified_mass();
        let mixed = mix_class_posterior_evals(&identified, &evals, unidentified_mass)?;
        let mut identification = mixed.identification;
        identification.status = if unidentified_mass > 0.0
            || mixed.mixture.unevaluable_mass > 0.0
            || !mixed.mixture.full_mass_scope
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
            "estimate.graph_posterior.class_envelope",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            format!(
                "each {class_tag} posterior atom was evaluated with the existing class ATE \
                 envelope; outer weights are posterior probabilities"
            ),
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
        let n_contributing = evals
            .iter()
            .filter(|eval| eval.estimate.as_ref().is_some_and(|estimate| estimate.ate.is_finite()))
            .count();
        diagnostics
            .extend(envelope_se_omission_diagnostic(n_contributing, mixed.estimate.se_analytic));
        if n_contributing > 1
            && mixed.estimate.se_analytic.is_finite()
            && mixed
                .estimate
                .influence
                .as_ref()
                .is_some_and(|inf| inf.iter().all(|v| v.is_finite()))
        {
            diagnostics.push(Diagnostic::new(
                "estimate.graph_posterior.joint_if_se",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                format!(
                    "SE of E[τ | identified] from the joint influence-function covariance of \
                     {n_contributing} class-posterior atoms fit on shared rows, with frozen \
                     graph weights; graph-weight uncertainty and unidentified mass are not in the \
                     SE; the interval is for the reported aggregate, not a distribution over \
                     graph-specific effects"
                ),
            ));
        }
        diagnostics.push(
            Diagnostic::new(
                "estimate.graph_posterior.envelope",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                format!(
                    "published effect is E[τ | identified]; identified_mass={}, unidentified_mass={}, atoms={}",
                    mixed.mixture.identified_mass,
                    mixed.mixture.unidentified_mass,
                    evals.len()
                ),
            )
            .with_fields(mass_fields(
                Some(mixed.mixture.identified_mass),
                mixed.mixture.unidentified_mass,
            )),
        );

        let estimator_id = match &self.inference {
            InferenceMode::Frequentist => EstimatorId::LinearAdjustmentAte,
            InferenceMode::Bayesian(_) => EstimatorId::BayesianGcomp,
        };
        let (refutations, refute_diagnostics) = refute_class_graph_posterior(
            self,
            data,
            query,
            &identified,
            &evals,
            mixed.policy,
            estimator_id.as_str(),
            ctx,
        )?;
        diagnostics.extend(refute_diagnostics);
        let algo =
            physical.logical.record.discovery_algorithm.as_deref().unwrap_or("graph_posterior");
        Ok(self.finish_identified_execute(IdentifiedExecuteFinish {
            physical,
            identification,
            estimand: mixed.estimand,
            estimate: mixed.estimate,
            identifier_id: IdentifierId::GeneralizedAdjustment,
            estimator_id,
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
                    "estimate.class_graph_posterior",
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

struct MixedClassPosterior {
    policy: StructuralAggregationPolicy,
    identification: IdentificationResult,
    estimand: IdentifiedEstimand,
    estimate: EffectEstimate,
    posterior: Option<CausalPosterior>,
    mixture: StructuralResponseMixture,
}

fn evaluate_class_atom_frequentist(
    study: &Study,
    data: &TabularData,
    query: &AverageEffectQuery,
    atom: &CachedClassPosteriorAtomIdentification,
    conditional: bool,
    ctx: &ExecutionContext,
) -> Result<ClassAtomEval, CausalError> {
    let mut weighted = 0.0;
    let mut total = 0.0;
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    let mut primary = None;
    let mut se_items = Vec::new();
    let mut atom_ifs = Vec::new();
    let mut atom_weights = Vec::new();
    let mut refute_atoms = Vec::new();
    let estimator_id = if conditional {
        EstimatorId::ConditionalLinearAdjustment
    } else {
        EstimatorId::LinearAdjustmentAte
    };
    for (i, case) in atom.cases.iter().enumerate() {
        let Some(estimand) = case.estimand.as_ref() else {
            continue;
        };
        let estimate = if conditional {
            let q = antecedent_core::ConditionalEffectQuery::try_new(query.clone())
                .map_err(|e| CausalError::Compile { message: e.to_string() })?;
            ConditionalLinearAdjustment::new()
                .estimate(data, estimand, &q, ctx)
                .map_err(CausalError::from)?
        } else {
            let mut case_ws = StaticEstimateWorkspaces::default();
            let case_spec = study
                .estimator_spec
                .clone()
                .unwrap_or(crate::estimator_spec::EstimatorSpec::Default(estimator_id));
            estimate_static_effect(
                &case_spec,
                data,
                estimand,
                query,
                atom.identification.required_assumptions.clone(),
                study.bootstrap_replicates,
                study.overlap_policy,
                study.population_registry.as_ref(),
                ctx,
                &mut case_ws,
            )?
        };
        let w = case.weight;
        if estimate.ate.is_finite() {
            weighted += w * estimate.ate;
            total += w;
            lo = lo.min(estimate.ate);
            hi = hi.max(estimate.ate);
            se_items.push((w, estimate.se_analytic));
            if let Some(inf) = estimate
                .influence
                .as_deref()
                .and_then(|inf| static_aligned_influence(data, query, estimand, inf))
            {
                atom_ifs.push(inf);
                atom_weights.push(w);
            }
            if primary.is_none() {
                primary = Some((estimand.clone(), estimate.assumptions.clone()));
            }
            refute_atoms.push(EnvelopeRefuteAtom {
                key: i as u64,
                weight: w,
                estimand: estimand.clone(),
                indexer: None,
                original: estimate,
            });
        }
    }
    let estimate = if total > 0.0 {
        let (_, assumptions) = primary.clone().ok_or_else(|| CausalError::Compile {
            message: "class posterior atom missing estimand".into(),
        })?;
        let n_contributing = se_items.len();
        let se = if atom_ifs.len() == n_contributing {
            mix_static_envelope_se(&atom_ifs, &atom_weights)
        } else {
            mix_weighted_analytic_se(se_items)
        };
        let mut estimate =
            EffectEstimate::new(weighted / total, se, assumptions, OverlapPolicy::ExplicitOverride);
        if atom_ifs.len() == n_contributing {
            if let Some(inf) = mixed_static_influence(&atom_ifs, &atom_weights) {
                estimate.influence = Some(inf);
            }
        }
        Some(estimate)
    } else {
        None
    };
    Ok(ClassAtomEval {
        key: atom.key,
        status: atom.identification.status,
        estimand: atom.invariant.clone().or_else(|| primary.map(|(e, _)| e)),
        estimate,
        posterior: None,
        identified_set: (lo.is_finite() && hi.is_finite()).then_some((lo, hi)),
        partial: atom.invariant.is_none()
            || matches!(atom.identification.status, IdentificationStatus::PartiallyIdentified),
        refute_atoms,
    })
}

fn evaluate_class_atom_bayesian(
    study: &Study,
    data: &TabularData,
    query: &AverageEffectQuery,
    atom: &CachedClassPosteriorAtomIdentification,
    conditional: bool,
    ctx: &ExecutionContext,
) -> Result<ClassAtomEval, CausalError> {
    let cfg = match &study.inference {
        InferenceMode::Bayesian(c) => c.clone(),
        InferenceMode::Frequentist => {
            return Err(CausalError::Unsupported {
                message: "evaluate_class_atom_bayesian requires inference=Bayesian",
            });
        }
    };
    let est = bayesian_gcomp(&cfg, ctx);
    let mut ws = BayesianGCompWorkspace::default();
    let mut weighted = 0.0;
    let mut total = 0.0;
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    let mut primary = None;
    let mut atom_posterior = None;
    let mut assumptions = antecedent_core::AssumptionSet::default();
    let mut refute_atoms = Vec::new();
    for (i, case) in atom.cases.iter().enumerate() {
        let Some(estimand) = case.estimand.as_ref() else {
            continue;
        };
        let prep = if conditional {
            let q = antecedent_core::ConditionalEffectQuery::try_new(query.clone())
                .map_err(|e| CausalError::Compile { message: e.to_string() })?;
            est.prepare_conditional(data, estimand, &q)
        } else {
            est.prepare(data, estimand, query)
        }
        .map_err(CausalError::from)?;
        let posterior = est.fit(&prep, case.status, &mut ws, ctx).map_err(CausalError::from)?;
        let summary = effect_from_posterior(&posterior)?;
        if summary.ate.is_finite() {
            weighted += case.weight * summary.ate;
            total += case.weight;
            lo = lo.min(summary.ate);
            hi = hi.max(summary.ate);
            if primary.is_none() {
                primary = Some(estimand.clone());
                assumptions = summary.assumptions.clone();
                atom_posterior = Some(posterior);
            }
            refute_atoms.push(EnvelopeRefuteAtom {
                key: i as u64,
                weight: case.weight,
                estimand: estimand.clone(),
                indexer: None,
                original: summary,
            });
        }
    }
    let estimate = (total > 0.0).then(|| {
        EffectEstimate::new(
            weighted / total,
            f64::NAN,
            assumptions,
            OverlapPolicy::ExplicitOverride,
        )
    });
    Ok(ClassAtomEval {
        key: atom.key,
        status: atom.identification.status,
        estimand: atom.invariant.clone().or(primary),
        estimate,
        posterior: if atom.cases.len() == 1 { atom_posterior } else { None },
        identified_set: (lo.is_finite() && hi.is_finite()).then_some((lo, hi)),
        partial: atom.invariant.is_none()
            || matches!(atom.identification.status, IdentificationStatus::PartiallyIdentified),
        refute_atoms,
    })
}

fn mix_class_posterior_evals(
    identified: &CachedGraphPosteriorIdentification,
    evals: &[ClassAtomEval],
    unidentified_mass: f64,
) -> Result<MixedClassPosterior, CausalError> {
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
    let mut outer_se_items = Vec::new();
    let mut outer_atom_ifs: Vec<Vec<f64>> = Vec::new();
    let mut outer_atom_weights = Vec::new();

    for eval in evals {
        let w = identified_weight_for_key(&identified.graphs, eval.key);
        let value = eval.estimate.as_ref().and_then(|e| e.ate.is_finite().then_some(e.ate));
        if value.is_some() {
            identified_weight += w;
            if let Some((a, b)) = eval.identified_set {
                lo = lo.min(a);
                hi = hi.max(b);
            } else if let Some(v) = value {
                lo = lo.min(v);
                hi = hi.max(v);
            }
            if let Some(estimand) = eval.estimand.as_ref() {
                contributing.push(estimand.clone());
                if primary_estimand.is_none() {
                    primary_estimand = Some(estimand.clone());
                    primary_identification = Some(eval_identification(eval, identified));
                    if let Some(estimate) = eval.estimate.as_ref() {
                        assumptions = estimate.assumptions.clone();
                    }
                }
            }
            any_partial |= eval.partial;
            if let Some(p) = eval.posterior.clone() {
                atom_posteriors.push((w, p));
            }
            if let Some(estimate) = eval.estimate.as_ref() {
                outer_se_items.push((w, estimate.se_analytic));
                if let Some(inf) = estimate.influence.as_ref() {
                    outer_atom_ifs.push(inf.to_vec());
                    outer_atom_weights.push(w);
                }
            }
            if let Some(v) = value {
                mixable += w;
                weighted += w * v;
            }
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

    // Unidentified posterior atoms keep their mass and appear in the mixture.
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
    let n_contributing = outer_se_items.len();
    let joint = n_contributing > 1 && outer_atom_ifs.len() == n_contributing;
    let se = if mixable_scalar {
        if joint {
            mix_static_envelope_se(&outer_atom_ifs, &outer_atom_weights)
        } else {
            mix_weighted_analytic_se(outer_se_items)
        }
    } else {
        f64::NAN
    };
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
        full_mass_scope: identified.class_atoms.iter().all(|a| a.truncated_completions == 0),
        truncated_atoms: evals.iter().map(|_| 0).sum::<usize>()
            + identified.class_atoms.iter().map(|a| a.truncated_completions).sum::<usize>(),
    };
    let identification = primary_identification.ok_or_else(|| CausalError::Compile {
        message: format!(
            "class graph-posterior mix: no evaluable atom values (evals={}, mixable={mixable})",
            evals.len()
        ),
    })?;
    let estimand = primary_estimand
        .or_else(|| identification.estimands.first().cloned())
        .ok_or_else(|| CausalError::Compile {
            message: "class graph-posterior envelope: missing estimand".into(),
        })?;
    let mut estimate = EffectEstimate::new(ate, se, assumptions, OverlapPolicy::ExplicitOverride);
    if joint {
        if let Some(inf) = mixed_static_influence(&outer_atom_ifs, &outer_atom_weights) {
            estimate.influence = Some(inf);
        }
    }
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
    Ok(MixedClassPosterior { policy, identification, estimand, estimate, posterior, mixture })
}

fn eval_identification(
    eval: &ClassAtomEval,
    identified: &CachedGraphPosteriorIdentification,
) -> IdentificationResult {
    identified
        .class_atoms
        .iter()
        .find(|atom| atom.key == eval.key)
        .map(|atom| atom.identification.clone())
        .unwrap_or_else(|| identified.class_atoms[0].identification.clone())
}

/// Inner: per-completion envelope refuters inside each posterior atom.
/// Outer: mix those atom reports by frozen posterior weight when estimands agree.
fn refute_class_graph_posterior(
    study: &Study,
    data: &TabularData,
    query: &AverageEffectQuery,
    identified: &CachedGraphPosteriorIdentification,
    evals: &[ClassAtomEval],
    policy: StructuralAggregationPolicy,
    estimator: &str,
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
            estimator,
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
        "refute.envelope.class_posterior",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        if mix_scalar {
            format!(
                "inner: effect refuters evaluated each completion within class-posterior atoms \
                 [{inner_keys}] against that completion's own estimate and mixed by completion \
                 enumeration weight (not posterior probability); outer: those atom reports mix \
                 by frozen posterior graph weight and pass only if every contributing atom passes"
            )
        } else {
            format!(
                "inner: effect refuters evaluated each completion within class-posterior atoms \
                 [{inner_keys}] against that completion's own estimate and mixed by completion \
                 enumeration weight (not posterior probability); outer scalar mix skipped \
                 because StructuralAggregationPolicy is {} — per-atom reports are retained and \
                 are not compared against a pooled mixture ATE",
                policy.as_str()
            )
        },
    ));
    let reports = if mix_scalar {
        mix_class_posterior_refutation_reports(&per_atom)
    } else {
        per_atom.into_iter().flat_map(|(_, reports)| reports).collect()
    };
    Ok((reports, diagnostics))
}

fn mix_class_posterior_refutation_reports(
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
