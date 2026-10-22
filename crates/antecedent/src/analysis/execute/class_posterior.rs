//! CPDAG/PAG graph-posterior `AverageEffect`: class envelope per atom, then policy.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::atom_mix::{
    AtomBoundRule, AtomEvalView, AtomMixInput, AtomRefuterBatch, ScalarRefuterMix,
    drive_posterior_refuters, mix_atom_evals,
};
use super::{
    Arc, AtomRefutationPolicy, AverageEffectQuery, BayesianGCompWorkspace, CausalError,
    CausalPosterior, CausalQuery, ConditionalLinearAdjustment, Diagnostic, DiagnosticKind,
    DiagnosticSeverity, EffectEstimate, EnvelopeRefutation, EnvelopeRefuteAtom,
    EstimationWorkspace, EstimatorId, ExecutionContext, GraphPosterior, IdentificationResult,
    IdentificationStatus, IdentifiedEstimand, IdentifiedExecuteExtras, IdentifiedExecuteFinish,
    IdentifiedKeyWeights, IdentifierId, InferenceMode, Instant, OverlapPolicy,
    PhysicalExecutionPlan, PosteriorMassSummary, RefuteSuite, StaticEstimateInput,
    StaticEstimateWorkspaces, Strategy, Study, StudyResult, TabularData, bayesian_gcomp,
    effect_from_posterior, elapsed_ns, envelope_se_omission_diagnostic, estimate_static_effect,
    mass_fields, mix_static_envelope_se, mix_weighted_analytic_se, mixed_static_influence,
    overlap_diagnostic, provenance_ids, resolve_posterior_status, static_aligned_influence,
};
use crate::analysis::prepared::{
    CachedClassPosteriorAtomIdentification, CachedGraphPosteriorIdentification,
};
use crate::analysis::proof_support::graph_posterior_withheld_diagnostic;
use crate::result::{StructuralAggregationPolicy, StructuralResponseMixture};

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
    pub(in crate::analysis) fn execute_class_graph_posterior(
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
        let (identified, identify_cached) = if let Some(cache) =
            self.graph_posterior_identification_cache.as_ref()
        {
            (Arc::clone(cache), true)
        } else {
            (
                Arc::new(crate::analysis::prepared::build_graph_posterior_identification_cache(
                    gp, query, ctx,
                )?),
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
        let mut mixed = mix_class_posterior_evals(&identified, &evals, unidentified_mass)?;
        mixed.identification.status = resolve_posterior_status(
            mixed.identification.status,
            mixed.policy,
            PosteriorMassSummary {
                unidentified: unidentified_mass,
                unevaluable: mixed.mixture.unevaluable_mass,
                full_scope: mixed.mixture.full_mass_scope,
                ..PosteriorMassSummary::default()
            },
        );

        let mut diagnostics = self.class_graph_posterior_diagnostics(&mixed, &evals, &class_tag);
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
            AtomRefutationPolicy { aggregation: mixed.policy, estimator: estimator_id.as_str() },
            ctx,
        )?;
        diagnostics.extend(refute_diagnostics);
        let algo =
            physical.logical.record.discovery_algorithm.as_deref().unwrap_or("graph_posterior");
        let MixedClassPosterior {
            policy,
            identification,
            estimand,
            estimate,
            posterior,
            mixture,
            ..
        } = mixed;
        Ok(self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand,
                estimate,
                Strategy::new(IdentifierId::GeneralizedAdjustment, estimator_id),
                query.treatment,
                query.outcome,
            )
            .identify_cached(identify_cached)
            .refutations(refutations)
            .wall_time_ns(elapsed_ns(started))
            .extras(IdentifiedExecuteExtras {
                identify_provenance: Some(provenance_ids("discover.graph_posterior", algo)),
                estimate_provenance: Some(provenance_ids(
                    "estimate.class_graph_posterior",
                    policy.as_str(),
                )),
                posterior,
                diagnostics: Some(diagnostics),
                structural_response: Some(mixture),
                ..Default::default()
            }),
        ))
    }

    /// Disclosures of a class graph-posterior effect: withheld posterior, aggregation policy,
    /// overlap, SE route and published mass.
    fn class_graph_posterior_diagnostics(
        &self,
        mixed: &MixedClassPosterior,
        evals: &[ClassAtomEval],
        class_tag: &str,
    ) -> Vec<Diagnostic> {
        let mut diagnostics = mixed.identification.diagnostics.clone();
        if self.inference.is_bayesian() && mixed.posterior.is_none() {
            diagnostics.push(graph_posterior_withheld_diagnostic());
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
        diagnostics.push(overlap_diagnostic(mixed.estimate.provenance.overlap));
        let n_contributing = evals
            .iter()
            .filter(|eval| eval.estimate.as_ref().is_some_and(|estimate| estimate.ate.is_finite()))
            .count();
        diagnostics
            .extend(envelope_se_omission_diagnostic(n_contributing, mixed.estimate.se_analytic()));
        if n_contributing > 1
            && mixed.estimate.se_analytic().is_finite()
            && mixed
                .estimate
                .scores
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
        diagnostics
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

/// One completion's effect: conditional linear adjustment or the configured static estimator.
fn fit_class_case(
    study: &Study,
    data: &TabularData,
    query: &AverageEffectQuery,
    atom: &CachedClassPosteriorAtomIdentification,
    estimand: &IdentifiedEstimand,
    conditional: bool,
    ctx: &ExecutionContext,
) -> Result<EffectEstimate, CausalError> {
    let estimator_id = if conditional {
        EstimatorId::ConditionalLinearAdjustment
    } else {
        EstimatorId::LinearAdjustmentAte
    };
    if conditional {
        let q = antecedent_core::ConditionalEffectQuery::try_new(query.clone())?;
        ConditionalLinearAdjustment::new()
            .estimate(data, estimand, &q, ctx)
            .map_err(CausalError::from)
    } else {
        let mut case_ws = StaticEstimateWorkspaces::default();
        let case_spec = study
            .estimator_spec
            .as_deref()
            .cloned()
            .unwrap_or(crate::estimator_spec::EstimatorSpec::Default(estimator_id));
        estimate_static_effect(
            &case_spec,
            StaticEstimateInput {
                data,
                estimand,
                query,
                bootstrap_replicates: study.bootstrap_replicates,
                overlap_policy: study.overlap_policy,
                population_registry: study.population_registry.as_ref(),
                sampling_design: study.sampling_design,
            },
            atom.identification.required_assumptions.clone(),
            ctx,
            &mut case_ws,
        )
    }
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
    for (i, case) in atom.cases.iter().enumerate() {
        let Some(estimand) = case.estimand.as_ref() else {
            continue;
        };
        let estimate = fit_class_case(study, data, query, atom, estimand, conditional, ctx)?;
        let w = case.weight;
        if estimate.ate.is_finite() {
            weighted += w * estimate.ate;
            total += w;
            lo = lo.min(estimate.ate);
            hi = hi.max(estimate.ate);
            se_items.push((w, estimate.se_analytic()));
            if let Some(inf) = estimate
                .scores
                .influence
                .as_deref()
                .and_then(|inf| static_aligned_influence(data, query, estimand, inf))
            {
                atom_ifs.push(inf);
                atom_weights.push(w);
            }
            if primary.is_none() {
                primary = Some((estimand.clone(), estimate.provenance.assumptions.clone()));
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
                estimate.scores.influence = Some(inf);
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
            return Err(CausalError::unsupported_inference(
                "evaluate_class_atom_bayesian requires inference=Bayesian",
            ));
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
            let q = antecedent_core::ConditionalEffectQuery::try_new(query.clone())?;
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
                assumptions = summary.provenance.assumptions.clone();
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
        // Completions that all return one estimand share one posterior: the draws of the
        // first fit are the atom's. Completions that disagree, or that do not all identify
        // the estimand, have no shared posterior.
        posterior: if atom.cases.len() == 1
            || (atom.invariant.is_some() && atom.cases.iter().all(|case| case.estimand.is_some()))
        {
            atom_posterior
        } else {
            None
        },
        identified_set: (lo.is_finite() && hi.is_finite()).then_some((lo, hi)),
        partial: atom.invariant.is_none()
            || matches!(atom.identification.status, IdentificationStatus::PartiallyIdentified),
        refute_atoms,
    })
}

impl AtomEvalView for ClassAtomEval {
    fn key(&self) -> u64 {
        self.key
    }

    fn status(&self) -> IdentificationStatus {
        self.status
    }

    fn estimand(&self) -> Option<&IdentifiedEstimand> {
        self.estimand.as_ref()
    }

    fn scalar(&self) -> Option<f64> {
        self.estimate.as_ref().and_then(|estimate| estimate.ate.is_finite().then_some(estimate.ate))
    }

    fn posterior(&self) -> Option<&CausalPosterior> {
        self.posterior.as_ref()
    }

    fn assumptions(&self) -> Option<&antecedent_core::AssumptionSet> {
        self.estimate.as_ref().map(|estimate| &estimate.provenance.assumptions)
    }

    fn identified_bounds(&self) -> Option<(f64, f64)> {
        self.identified_set
    }

    fn partial(&self) -> bool {
        self.partial
    }

    fn analytic_se(&self) -> Option<f64> {
        self.estimate.as_ref().map(antecedent_estimate::EffectEstimate::se_analytic)
    }

    fn influence(&self) -> Option<&[f64]> {
        self.estimate.as_ref().and_then(|estimate| estimate.scores.influence.as_deref())
    }
}

fn mix_class_posterior_evals(
    identified: &CachedGraphPosteriorIdentification,
    evals: &[ClassAtomEval],
    unidentified_mass: f64,
) -> Result<MixedClassPosterior, CausalError> {
    let mixed = mix_atom_evals(
        evals,
        AtomMixInput {
            graphs: &identified.graphs,
            proofs: &identified.class_atoms,
            unidentified_mass,
            full_mass_scope: identified
                .class_atoms
                .iter()
                .all(|atom| atom.truncated_completions == 0),
            truncated_atoms: evals.iter().map(|_| 0).sum::<usize>()
                + identified
                    .class_atoms
                    .iter()
                    .map(|atom| atom.truncated_completions)
                    .sum::<usize>(),
            bounds: AtomBoundRule::SetOrPoint,
            estimand_fallback: true,
        },
        |n, mixable| {
            format!(
                "class graph-posterior mix: no evaluable atom values (evals={n}, mixable={mixable})"
            )
        },
        || "class graph-posterior envelope: missing estimand".to_string(),
    )?;
    Ok(MixedClassPosterior {
        policy: mixed.policy,
        identification: mixed.identification,
        estimand: mixed.estimand,
        estimate: mixed.estimate,
        posterior: mixed.posterior,
        mixture: mixed.mixture,
    })
}

/// Inner: per-completion envelope refuters inside each posterior atom.
/// Outer: mix those atom reports by frozen posterior weight when estimands agree.
fn refute_class_graph_posterior(
    study: &Study,
    data: &TabularData,
    query: &AverageEffectQuery,
    identified: &CachedGraphPosteriorIdentification,
    evals: &[ClassAtomEval],
    refutation: AtomRefutationPolicy<'_>,
    ctx: &ExecutionContext,
) -> Result<(Vec<antecedent_validate::RefutationReport>, Vec<Diagnostic>), CausalError> {
    let AtomRefutationPolicy { aggregation: policy, estimator } = refutation;
    if matches!(study.refute, RefuteSuite::None) && study.custom_validators.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }
    let mix_scalar = matches!(policy, StructuralAggregationPolicy::SameEstimandWeightedMean);
    let mut refute_ws = EstimationWorkspace::default();
    let key_weights = IdentifiedKeyWeights::new(&identified.graphs);
    drive_posterior_refuters(
        evals,
        mix_scalar,
        ScalarRefuterMix::Effect,
        |eval| !eval.refute_atoms.is_empty(),
        |eval| eval.key,
        |eval| {
            if eval.refute_atoms.is_empty() {
                return Ok(Vec::new());
            }
            let weight = key_weights.get(eval.key);
            if weight <= 0.0 {
                return Ok(Vec::new());
            }
            let (reports, notes) = EnvelopeRefutation {
                data,
                query,
                suite: study.refute,
                estimator,
                custom: &study.custom_validators,
                temporal: None,
            }
            .run(&eval.refute_atoms, &mut refute_ws, ctx)?;
            Ok(vec![AtomRefuterBatch {
                weight,
                reports,
                notes: notes
                    .into_iter()
                    .filter(|diagnostic| {
                        diagnostic.code.as_ref() != "refute.envelope.effect_mixture"
                    })
                    .collect(),
            }])
        },
        |inner_keys, mix_scalar| {
            Diagnostic::new(
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
            )
        },
    )
}
