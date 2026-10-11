//! ADMG graph-posterior AverageEffect: general.id + functional.effect per atom.
//!
//! Atoms are single graphs, not MEC completions. Outer mix uses posterior
//! weights and [`StructuralAggregationPolicy`]. Unidentified mass is retained.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::atom_mix::{
    AtomBoundRule, AtomEvalView, AtomMixInput, AtomRefuterBatch, ScalarRefuterMix,
    drive_posterior_refuters, mix_atom_evals,
};
use super::*;
use crate::analysis::prepared::CachedGraphPosteriorIdentification;
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
        checked: Option<&CachedGraphPosteriorIdentification>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        if matches!(self.query, CausalQuery::ConditionalEffect(_)) {
            return Err(CausalError::Unsupported {
                message: "ConditionalEffect graph-posterior mixing is licensed only for DAG atoms",
            });
        }
        if !matches!(
            gp.atom_kind,
            antecedent_discovery::GraphPosteriorAtomKind::Admg
                | antecedent_discovery::GraphPosteriorAtomKind::TemporalAdmg
        ) {
            return Err(CausalError::Compile {
                message: "execute_admg_graph_posterior requires Admg posterior atoms".into(),
            });
        }
        let (identified, identify_cached) = if let Some(cache) = checked {
            (cache.clone(), true)
        } else if let Some(cache) = self.graph_posterior_identification_cache.as_deref() {
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

        let evals = ctx.map_indexed(identified.atoms.len(), |i, inner| {
            let atom = &identified.atoms[i];
            let (estimate, posterior) = self.estimate_functional_effect(
                data,
                &atom.estimand,
                &atom.identification,
                &[query.treatment, query.outcome],
                inner,
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
            Ok::<_, CausalError>(AdmgAtomEval {
                key: atom.key,
                status: atom.identification.status,
                estimand: atom.estimand.clone(),
                estimate,
                posterior,
                refute_atoms,
            })
        })?;

        let unidentified_mass = identified.graphs.unidentified_mass();
        let mixed = mix_admg_posterior_evals(&identified, &evals, unidentified_mass)?;
        let mut identification = mixed.identification;
        identification.status = resolve_posterior_status(
            identification.status,
            mixed.policy,
            PosteriorMassSummary {
                unidentified: unidentified_mass,
                ..PosteriorMassSummary::default()
            },
        );

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
        Some(&self.estimate.assumptions)
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
            let weight = identified_weight_for_key(&identified.graphs, eval.key);
            if weight <= 0.0 {
                return Ok(Vec::new());
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
