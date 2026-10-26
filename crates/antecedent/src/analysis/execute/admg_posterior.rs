//! ADMG graph-posterior `AverageEffect`: general.id + functional.effect per atom.
//!
//! Atoms are single graphs, not MEC completions. Outer mix uses posterior
//! weights and [`StructuralAggregationPolicy`]. Unidentified mass is retained.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::atom_mix::{
    AtomBoundRule, AtomEvalView, AtomMixInput, AtomRefuterBatch, ScalarRefuterMix,
    drive_posterior_refuters, mix_atom_evals,
};
use super::{
    Arc, AverageEffectQuery, CausalError, CausalPosterior, CausalQuery, Diagnostic, DiagnosticKind,
    DiagnosticSeverity, EffectEstimate, EnvelopeRefutation, EnvelopeRefuteAtom,
    EstimationWorkspace, EstimatorId, ExecutionContext, GraphPosterior, IdentificationResult,
    IdentificationStatus, IdentifiedEstimand, IdentifiedExecuteExtras, IdentifiedExecuteFinish,
    IdentifiedKeyWeights, IdentifierId, Instant, PhysicalExecutionPlan, PosteriorMassSummary,
    RefuteSuite, Strategy, Study, StudyResult, TabularData, elapsed_ns, mass_fields,
    overlap_diagnostic, provenance_ids, resolve_posterior_status,
};
use crate::analysis::prepared::CachedGraphPosteriorIdentification;
use crate::analysis::proof_support::graph_posterior_withheld_diagnostic;
use crate::result::{StructuralAggregationPolicy, StructuralResponseMixture};

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
        self.execute_admg_graph_posterior_with(data, gp, query, physical, None, ctx)
    }

    pub(in crate::analysis) fn execute_checked_admg_graph_posterior(
        &self,
        data: &TabularData,
        physical: &PhysicalExecutionPlan,
        operation: &super::super::CheckedGraphPosteriorEffect,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        self.execute_admg_graph_posterior_with(
            data,
            operation.posterior(),
            operation.query(),
            physical,
            Some(operation.identification()),
            ctx,
        )
    }

    fn execute_admg_graph_posterior_with(
        &self,
        data: &TabularData,
        gp: &GraphPosterior,
        query: &AverageEffectQuery,
        physical: &PhysicalExecutionPlan,
        checked: Option<&Arc<CachedGraphPosteriorIdentification>>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        if matches!(self.query, CausalQuery::ConditionalEffect(_)) {
            return Err(CausalError::Unsupported {
                message: "ConditionalEffect graph-posterior mixing is licensed only for DAG atoms",
            });
        }
        if !matches!(
            gp.atom_kind(),
            antecedent_discovery::GraphPosteriorAtomKind::Admg
                | antecedent_discovery::GraphPosteriorAtomKind::TemporalAdmg
        ) {
            return Err(CausalError::Compile {
                message: "execute_admg_graph_posterior requires Admg posterior atoms".into(),
            });
        }
        let (identified, identify_cached) = if let Some(cache) = checked {
            (Arc::clone(cache), true)
        } else if let Some(cache) = self.graph_posterior_cache() {
            (Arc::clone(cache), true)
        } else if let Some(cache) =
            crate::analysis::lowering::plan_retained_graph_posterior(self, physical)
        {
            (cache, false)
        } else {
            (
                Arc::new(crate::analysis::prepared::build_graph_posterior_identification_cache(
                    gp, query, ctx,
                )?),
                false,
            )
        };
        if identified.atoms.is_empty() && identified.graphs.identified_mass() <= 0.0 {
            return Err(CausalError::Compile {
                message: "ADMG graph-posterior envelope: no identified ADMG atoms".into(),
            });
        }

        let evals = self.evaluate_admg_atoms(data, query, &identified, ctx)?;

        let unidentified_mass = identified.graphs.unidentified_mass();
        let mut mixed = mix_admg_posterior_evals(&identified, &evals, unidentified_mass)?;
        mixed.identification.status = resolve_posterior_status(
            mixed.identification.status,
            mixed.policy,
            PosteriorMassSummary {
                unidentified: unidentified_mass,
                ..PosteriorMassSummary::default()
            },
        );

        let mut diagnostics = self.admg_graph_posterior_diagnostics(&mixed);

        let (refutations, refute_diagnostics) =
            refute_admg_graph_posterior(self, data, query, &identified, &evals, mixed.policy, ctx)?;
        diagnostics.extend(refute_diagnostics);

        let algo =
            physical.logical.record.discovery_algorithm.as_deref().unwrap_or("graph_posterior");
        let MixedAdmgPosterior {
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
                Strategy::new(IdentifierId::GeneralId, EstimatorId::FunctionalEffect),
                query.treatment,
                query.outcome,
            )
            .identify_cached(identify_cached)
            .refutations(refutations)
            .wall_time_ns(elapsed_ns(started))
            .extras(IdentifiedExecuteExtras {
                identify_provenance: Some(provenance_ids("discover.graph_posterior", algo)),
                estimate_provenance: Some(provenance_ids(
                    "estimate.admg_graph_posterior",
                    policy.as_str(),
                )),
                posterior,
                diagnostics: Some(diagnostics),
                structural_response: Some(mixture),
                ..Default::default()
            }),
        ))
    }

    /// Estimates every identified ADMG atom with `functional.effect` on the worker pool.
    fn evaluate_admg_atoms(
        &self,
        data: &TabularData,
        query: &AverageEffectQuery,
        identified: &CachedGraphPosteriorIdentification,
        ctx: &ExecutionContext,
    ) -> Result<Vec<AdmgAtomEval>, CausalError> {
        let key_weights = IdentifiedKeyWeights::new(&identified.graphs);
        ctx.map_indexed(identified.atoms.len(), |i, inner| {
            let atom = &identified.atoms[i];
            let (estimate, posterior) = self.estimate_functional_effect(
                data,
                &atom.estimand,
                &atom.identification,
                &[query.treatment, query.outcome],
                inner,
            )?;
            let weight = key_weights.get(atom.key);
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
            Ok::<_, CausalError>(AdmgAtomEval {
                key: atom.key,
                status: atom.identification.status,
                estimand: atom.estimand.clone(),
                estimate,
                posterior,
                refute_atoms,
            })
        })
    }

    /// Withheld-posterior, class-envelope, aggregation-policy and overlap disclosures.
    fn admg_graph_posterior_diagnostics(&self, mixed: &MixedAdmgPosterior) -> Vec<Diagnostic> {
        let mut diagnostics = mixed.identification.diagnostics.clone();
        if self.inference.is_bayesian() && mixed.posterior.is_none() {
            diagnostics.push(graph_posterior_withheld_diagnostic());
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
        diagnostics.push(overlap_diagnostic(mixed.estimate.provenance.overlap));
        diagnostics
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

impl AtomEvalView for AdmgAtomEval {
    fn key(&self) -> u64 {
        self.key
    }

    fn status(&self) -> IdentificationStatus {
        self.status
    }

    fn estimand(&self) -> Option<&IdentifiedEstimand> {
        Some(&self.estimand)
    }

    fn scalar(&self) -> Option<f64> {
        self.estimate.ate.is_finite().then_some(self.estimate.ate)
    }

    fn posterior(&self) -> Option<&CausalPosterior> {
        self.posterior.as_ref()
    }

    fn assumptions(&self) -> Option<&antecedent_core::AssumptionSet> {
        Some(&self.estimate.provenance.assumptions)
    }

    fn identified_bounds(&self) -> Option<(f64, f64)> {
        None
    }

    fn partial(&self) -> bool {
        matches!(self.status, IdentificationStatus::PartiallyIdentified)
    }

    fn analytic_se(&self) -> Option<f64> {
        None
    }

    fn influence(&self) -> Option<&[f64]> {
        None
    }

    fn reported_uncertainty(&self) -> Option<antecedent_estimate::EstimateUncertainty> {
        Some(self.estimate.uncertainty)
    }
}

fn mix_admg_posterior_evals(
    identified: &CachedGraphPosteriorIdentification,
    evals: &[AdmgAtomEval],
    unidentified_mass: f64,
) -> Result<MixedAdmgPosterior, CausalError> {
    let mixed = mix_atom_evals(
        evals,
        AtomMixInput {
            graphs: &identified.graphs,
            proofs: &identified.atoms,
            unidentified_mass,
            full_mass_scope: true,
            truncated_atoms: 0,
            bounds: AtomBoundRule::Point,
            estimand_fallback: true,
        },
        |n, mixable| {
            format!(
                "ADMG graph-posterior mix: no evaluable atom values (evals={n}, mixable={mixable})"
            )
        },
        || "ADMG graph-posterior envelope: missing estimand".to_string(),
    )?;
    Ok(MixedAdmgPosterior {
        policy: mixed.policy,
        identification: mixed.identification,
        estimand: mixed.estimand,
        estimate: mixed.estimate,
        posterior: mixed.posterior,
        mixture: mixed.mixture,
    })
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
                estimator: EstimatorId::FunctionalEffect.as_str(),
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
            )
        },
    )
}
