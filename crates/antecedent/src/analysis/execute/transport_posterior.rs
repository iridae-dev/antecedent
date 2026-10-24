//! `TransportQuery` over a supplied graph posterior.
//!
//! Every posterior atom is a causal graph, so each is a selection diagram with the declared
//! selection targets, and sID decides it on its own graph. An atom with a certified formula
//! is estimated by the trial-to-target weighting of the supplied selection and treatment
//! probabilities; an atom sID does not certify is unidentified mass, and an atom whose
//! certified formula the weighting cannot evaluate (a recursive factorization) is unevaluable
//! mass. The atoms mix by their posterior probability, and neither kind of lost mass is
//! redistributed: the published value is the transported contrast conditional on the atoms
//! that identify and evaluate, and the result states the mass it does not cover.
//!
//! The weighting reads a certified formula only through its certificate, so every evaluable
//! atom shares one transported contrast and one sampling or posterior law. What the posterior
//! moves is the identification: which atoms certify a formula, by which rule, and how much
//! posterior mass certifies none.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, HashSet};

use antecedent_data::TableView;
use antecedent_estimate::{TransportEffectEstimate, trial_to_target_effect};
use antecedent_identify::{TransportFormula, TransportIdentification};

use super::atom_mix::{AtomBoundRule, AtomEvalView, AtomMixInput, mix_atom_evals};
use super::transport_interference_path::{
    TrialColumns, bayesian_transport_estimate, frequentist_transport_estimate,
    known_probability_assumption, transport_primary_pair, transport_sid_identification,
};
use super::transport_refutation::{TransportRefutationInput, refute_transport};
use super::{
    Arc, CausalError, CausalPosterior, CausalQuery, Diagnostic, DiagnosticKind, DiagnosticSeverity,
    EnvelopeOptions, EstimatorId, ExecutionContext, GraphEffectDraws, GraphIdentFlag,
    GraphPosterior, IdentificationResult, IdentificationStatus, IdentifiedEstimand,
    IdentifiedExecuteExtras, IdentifiedExecuteFinish, IdentifiedKeyWeights, IdentifierId,
    InferenceDiagnostics, InferenceMode, Instant, PhysicalExecutionPlan, PosteriorMassSummary,
    Strategy, StudyResult, TabularData, WeightedGraphSamples, aggregate_effect_envelope,
    effect_from_posterior, elapsed_ns, overlap_diagnostic, provenance_ids,
    push_graph_posterior_structural_aggregation_diagnostic, resolve_posterior_status,
};
use crate::analysis::identification_cache::TRANSPORT_CLASS_ATOM_CODE;
use crate::analysis::prepared::CachedGraphPosteriorIdentification;

/// Method tag every posterior atom's transport estimand carries. The certified rule differs
/// by atom (direct use, standardization, S-admissible standardization); the target they
/// identify is the one transported contrast, so the estimand identity is the target's.
const ATOM_ESTIMAND_METHOD: &str = "transport.sid";

impl super::Study {
    /// The identification one posterior atom states for `query` from its certified `identified`
    /// formula, with the estimand tagged by the target rather than by the atom's rule so atoms
    /// that identify the one target by different formulas agree. `None` when the formula is not
    /// certified or the query names no treatment/outcome pair.
    pub(crate) fn transport_posterior_atom(
        query: &antecedent_core::TransportQuery,
        identified: &TransportIdentification,
    ) -> Option<(IdentificationResult, IdentifiedEstimand)> {
        if !matches!(identified, TransportIdentification::Transportable { .. }) {
            return None;
        }
        let (treatment, outcome) = transport_primary_pair(query).ok()?;
        let (mut identification, mut estimand) = transport_sid_identification(
            CausalQuery::Transport(query.clone()),
            treatment,
            outcome,
            identified,
        );
        estimand.method = Arc::from(ATOM_ESTIMAND_METHOD);
        for listed in &mut identification.estimands {
            listed.method = Arc::from(ATOM_ESTIMAND_METHOD);
        }
        Some((identification, estimand))
    }

    /// Mix the sID-certified trial-to-target contrast over the atoms of a graph posterior.
    ///
    /// # Errors
    ///
    /// A posterior of class atoms or with DBN lag structure, no atom whose formula sID
    /// certifies, no atom whose certified formula the weighting can evaluate, invalid
    /// probability columns, or a cancelled run.
    pub(super) fn execute_transport_graph_posterior(
        &self,
        data: &TabularData,
        gp: &GraphPosterior,
        query: &antecedent_core::TransportQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        query.validate()?;
        let trial_spec = self.transport_trial.as_ref().ok_or(CausalError::Unsupported {
            message: "TransportQuery execute requires transport_trial columns",
        })?;
        let (treatment, outcome) = transport_primary_pair(query)?;
        let (identified, identify_cached) =
            self.transport_posterior_identification(gp, query, physical, ctx)?;
        if identified.atoms.is_empty() {
            return Err(CausalError::not_identified(
                IdentificationStatus::NotIdentified,
                false,
                "TransportQuery graph posterior: sID certifies a transport formula for no \
                 posterior atom",
            ));
        }

        let outcomes = data.float64_values(outcome).map_err(CausalError::rejected)?;
        let treatment_col = data.float64_values(treatment).map_err(CausalError::rejected)?;
        let trial_col = data.float64_values(trial_spec.trial).map_err(CausalError::rejected)?;
        let selection =
            data.float64_values(trial_spec.selection_probability).map_err(CausalError::rejected)?;
        let propensity =
            data.float64_values(trial_spec.treatment_probability).map_err(CausalError::rejected)?;
        let treatment_bool: Vec<bool> = treatment_col.iter().map(|v| *v != 0.0).collect();
        let trial_bool: Vec<bool> = trial_col.iter().map(|v| *v != 0.0).collect();

        // Every certified atom is run through the weighting's own certificate check; only an
        // atom whose formula it cannot evaluate is unevaluable, never a data failure.
        let atom_estimates: Vec<Option<TransportEffectEstimate>> =
            ctx.map_indexed(identified.atoms.len(), |index, _inner| {
                let Some(transport) = identified.atoms[index].transport.as_deref() else {
                    return Ok::<_, CausalError>(None);
                };
                if !ipw_estimable(transport) {
                    return Ok(None);
                }
                trial_to_target_effect(
                    transport,
                    &outcomes,
                    &treatment_bool,
                    &trial_bool,
                    &selection,
                    &propensity,
                    None,
                )
                .map(Some)
                .map_err(CausalError::from)
            })?;
        let Some(representative) = atom_estimates.iter().position(Option::is_some) else {
            return Err(crate::support_reason!(
                "construction_not_licensed",
                "every certified posterior atom is a RecursiveFactorization, identified but not \
                 estimable on the licensed trial-to-target IPW path"
            ));
        };
        let (Some(representative_transport), Some(mut transported)) =
            (identified.atoms[representative].transport.as_deref(), atom_estimates[representative])
        else {
            return Err(CausalError::Compile {
                message: "transport graph-posterior representative atom lost its proof".into(),
            });
        };

        let mut assumptions =
            identified.atoms[representative].identification.required_assumptions.clone();
        assumptions.push(known_probability_assumption());
        let status = identified.atoms[representative].identification.status;
        let series_block = self.transport_series_block(outcomes.len())?;
        let columns = TrialColumns {
            outcomes: &outcomes,
            treatment: &treatment_bool,
            trial: &trial_bool,
            selection: &selection,
            propensity: &propensity,
        };
        let (point, representative_posterior, estimator_id, estimator_diagnostic) =
            match &self.inference {
                InferenceMode::Frequentist => {
                    let (estimate, diagnostic) = frequentist_transport_estimate(
                        &columns,
                        (transported.ipw, series_block),
                        assumptions,
                    )?;
                    (estimate, None, EstimatorId::TransportTrialIpw, diagnostic)
                }
                InferenceMode::Bayesian(config) => {
                    let (estimate, posterior, diagnostic) = bayesian_transport_estimate(
                        config,
                        (representative_transport, series_block),
                        &columns,
                        assumptions,
                        status,
                        ctx,
                    )?;
                    transported.ipw = estimate.ate;
                    (
                        estimate,
                        Some(posterior),
                        EstimatorId::TransportTrialBayesianBootstrap,
                        diagnostic,
                    )
                }
            };

        let bayesian = representative_posterior.is_some();
        let evals: Vec<ScalarAtomEval> = identified
            .atoms
            .iter()
            .zip(&atom_estimates)
            .map(|(atom, estimated)| ScalarAtomEval {
                key: atom.key,
                status: atom.identification.status,
                estimand: atom.estimand.clone(),
                value: estimated.map(|each| if bayesian { point.ate } else { each.ipw }),
            })
            .collect();
        let unidentified_mass = identified.graphs.unidentified_mass();
        let mut mixed = mix_atom_evals(
            &evals,
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
                    "transport graph-posterior mix: no evaluable atom values (evals={n}, \
                     mixable={mixable})"
                )
            },
            || "transport graph-posterior envelope: missing estimand".to_string(),
        )?;
        mixed.identification.status = resolve_posterior_status(
            mixed.identification.status,
            mixed.policy,
            PosteriorMassSummary {
                unidentified: unidentified_mass,
                unevaluable: mixed.mixture.unevaluable_mass,
                ..PosteriorMassSummary::default()
            },
        );

        // An atom that is a class whose members certify different formulas is partially
        // identified, and so is the mixture that contains it unless lost mass already made it
        // graph dependent.
        if mixed.identification.status == IdentificationStatus::NonparametricallyIdentified
            && evals.iter().any(|eval| {
                eval.value.is_some() && eval.status == IdentificationStatus::PartiallyIdentified
            })
        {
            mixed.identification.status = IdentificationStatus::PartiallyIdentified;
        }

        let (estimate, posterior) = match representative_posterior {
            Some(component) => {
                let posterior = envelope_posterior(&identified, &evals, &component)?;
                (effect_from_posterior(&posterior)?, Some(posterior))
            }
            None if mixed.mixable_scalar => (point, None),
            None => (mixed.estimate.clone(), None),
        };

        let refutations = refute_transport(
            &TransportRefutationInput {
                transport: representative_transport,
                outcomes: &outcomes,
                treatment: &treatment_bool,
                trial: &trial_bool,
                selection: &selection,
                propensity: &propensity,
                overlap: transported.overlap,
                published: estimate.ate,
                suite: self.refute,
                series_block,
                replicates: self.bootstrap_replicates,
            },
            ctx,
        )?;

        let mut diagnostics = mixed.identification.diagnostics.clone();
        diagnostics.push(self.transport_atom_rule_diagnostic(
            &identified,
            &mixed.mixture,
            unidentified_mass,
        ));
        diagnostics.extend(Self::transport_class_member_diagnostic(&identified));
        push_graph_posterior_structural_aggregation_diagnostic(
            &mut diagnostics,
            mixed.policy,
            mixed.mixture.identified_mass,
            mixed.mixture.unidentified_mass,
            mixed.mixture.unevaluable_mass,
            mixed.mixture.subsampled_out_mass,
        );
        diagnostics.push(estimator_diagnostic);
        diagnostics.push(overlap_diagnostic(estimate.provenance.overlap));

        let algo =
            physical.logical.record.discovery_algorithm.as_deref().unwrap_or("graph_posterior");
        let policy = mixed.policy;
        let mut result = self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                mixed.identification,
                mixed.estimand,
                estimate,
                Strategy::new(IdentifierId::TransportSid, estimator_id),
                treatment,
                outcome,
            )
            .identify_cached(identify_cached)
            .refutations(refutations)
            .wall_time_ns(elapsed_ns(started))
            .cancelled(ctx.cancellation.is_cancelled())
            .extras(IdentifiedExecuteExtras {
                identify_provenance: Some(provenance_ids("discover.graph_posterior", algo)),
                estimate_provenance: Some(provenance_ids(
                    "estimate.transport_graph_posterior",
                    policy.as_str(),
                )),
                posterior,
                diagnostics: Some(diagnostics),
                structural_response: Some(mixed.mixture),
                ..Default::default()
            }),
        );
        result.payload = crate::result::ResultPayload::Transport(transported);
        Ok(result)
    }

    /// The frozen per-atom transport identification: the prepared cache, the product the plan's
    /// lowered program retained at compile, or a fresh run of sID on every atom.
    fn transport_posterior_identification(
        &self,
        gp: &GraphPosterior,
        query: &antecedent_core::TransportQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<(Arc<CachedGraphPosteriorIdentification>, bool), CausalError> {
        if let Some(cache) = self.graph_posterior_identification_cache.as_ref() {
            return Ok((Arc::clone(cache), true));
        }
        if let Some(cache) =
            crate::analysis::lowering::plan_retained_graph_posterior(self, physical)
        {
            return Ok((cache, false));
        }
        let targets: Vec<_> = self
            .selection_diagram
            .as_ref()
            .map(|diagram| diagram.selection_targets().to_vec())
            .unwrap_or_default();
        Ok((
            Arc::new(
                crate::analysis::identification_cache::build_transport_graph_posterior_identification_cache(
                    gp, query, &targets, ctx,
                )?,
            ),
            false,
        ))
    }

    /// Which sID rule certified how much posterior mass, with the mass sID did not certify and
    /// the mass it certified but the weighting cannot evaluate.
    fn transport_atom_rule_diagnostic(
        &self,
        identified: &CachedGraphPosteriorIdentification,
        mixture: &crate::result::StructuralResponseMixture,
        unidentified_mass: f64,
    ) -> Diagnostic {
        let key_weights = IdentifiedKeyWeights::new(&identified.graphs);
        let mut by_rule: BTreeMap<String, f64> = BTreeMap::new();
        for atom in identified.atoms.iter() {
            let rule = match atom.transport.as_deref() {
                Some(TransportIdentification::Transportable { certificate, .. }) => {
                    certificate.rule.to_string()
                }
                _ => "uncertified".to_owned(),
            };
            *by_rule.entry(rule).or_insert(0.0) += key_weights.get(atom.key);
        }
        let rules = by_rule
            .iter()
            .map(|(rule, mass)| format!("{rule}={mass}"))
            .collect::<Vec<_>>()
            .join(", ");
        let targets = self
            .selection_diagram
            .as_ref()
            .map(|diagram| {
                diagram
                    .selection_targets()
                    .iter()
                    .map(|target| target.raw().to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            })
            .unwrap_or_default();
        Diagnostic::new(
            "estimate.graph_posterior.transport_sid",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            format!(
                "each posterior atom was run through sID on its own selection diagram \
                 (selection targets [{targets}]); certified rules by posterior mass: [{rules}]; \
                 unidentified_mass={unidentified_mass} (sID certifies no formula); \
                 unevaluable_mass={} (certified, but not estimable by trial-to-target \
                 weighting); the weighting reads a formula only through its certificate, so the \
                 published contrast is the one transported contrast conditional on the atoms \
                 that identify and evaluate, with the supplied selection and treatment \
                 probabilities held fixed",
                mixture.unevaluable_mass
            ),
        )
    }
}

impl super::Study {
    /// What the classes of selection diagrams that Cpdag and Pag atoms stand for left
    /// uncertified: the posterior mass of the atoms that are classes and the share of that
    /// mass whose class members sID did not certify. `None` when no atom is a class.
    fn transport_class_member_diagnostic(
        identified: &CachedGraphPosteriorIdentification,
    ) -> Option<Diagnostic> {
        let key_weights = IdentifiedKeyWeights::new(&identified.graphs);
        let mut atoms = 0usize;
        let mut class_mass = 0.0;
        let mut uncertified_mass = 0.0;
        for atom in identified.atoms.iter() {
            let Some(note) = atom
                .identification
                .diagnostics
                .iter()
                .find(|note| note.code.as_ref() == TRANSPORT_CLASS_ATOM_CODE)
            else {
                continue;
            };
            let field = |name: &str| {
                note.fields
                    .iter()
                    .find(|(key, _)| key.as_ref() == name)
                    .and_then(|(_, value)| value.parse::<f64>().ok())
            };
            let (Some(certified), Some(uncertified)) =
                (field("identified_weight"), field("unidentified_weight"))
            else {
                continue;
            };
            let mass = key_weights.get(atom.key);
            atoms += 1;
            class_mass += mass;
            if certified + uncertified > 0.0 {
                uncertified_mass += mass * uncertified / (certified + uncertified);
            }
        }
        (atoms > 0).then(|| {
            Diagnostic::new(
                "estimate.graph_posterior.transport_class_members",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                format!(
                    "{atoms} identified atom(s) stand for a class of selection diagrams (a Cpdag \
                     or a Pag): sID decided every DAG completion, or every ADMG sharing the \
                     m-separation model of a MAG completion, on its own diagram; the identified \
                     class mass is {class_mass} of which {uncertified_mass} sits on members sID \
                     did not certify (read as identified conditional on the certified members, \
                     as the class effect posterior reads its atoms); members that certify \
                     different formulas are evaluated by the one trial-to-target weighting, \
                     which reads a formula only through its certificate"
                ),
            )
            .with_fields([
                ("class_atoms", atoms.to_string()),
                ("class_mass", class_mass.to_string()),
                ("uncertified_member_mass", uncertified_mass.to_string()),
            ])
        })
    }
}

/// Whether the trial-to-target weighting can evaluate the certified formula.
pub(crate) fn ipw_estimable(transport: &TransportIdentification) -> bool {
    matches!(
        transport,
        TransportIdentification::Transportable { formula, .. }
            if !matches!(formula, TransportFormula::RecursiveFactorization { .. })
    )
}

/// One evaluated posterior atom of an exposure query: the contrast it estimates, `None` when
/// the atom's certified formula is not evaluable.
pub(super) struct ScalarAtomEval {
    pub(super) key: u64,
    pub(super) status: IdentificationStatus,
    pub(super) estimand: IdentifiedEstimand,
    pub(super) value: Option<f64>,
}

impl AtomEvalView for ScalarAtomEval {
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
        self.value.filter(|value| value.is_finite())
    }

    fn posterior(&self) -> Option<&CausalPosterior> {
        None
    }

    fn assumptions(&self) -> Option<&antecedent_core::AssumptionSet> {
        None
    }

    fn identified_bounds(&self) -> Option<(f64, f64)> {
        None
    }

    fn partial(&self) -> bool {
        false
    }

    fn analytic_se(&self) -> Option<f64> {
        None
    }

    fn influence(&self) -> Option<&[f64]> {
        None
    }
}

/// The posterior of the mixture: every evaluable atom contributes the one Dirichlet
/// trial-row bootstrap of the contrast with its posterior mass, atoms that are unevaluable are
/// moved out of the identified mass, and the unidentified and unevaluable masses stay separate
/// axes that are never redistributed into the effect posterior.
fn envelope_posterior(
    identified: &CachedGraphPosteriorIdentification,
    evals: &[ScalarAtomEval],
    component: &CausalPosterior,
) -> Result<CausalPosterior, CausalError> {
    let column = component.effect_column().ok_or_else(|| CausalError::Compile {
        message: "transport posterior missing its effect column".into(),
    })?;
    let draws: Arc<[f64]> = Arc::from(component.draws.column(column)?.to_vec());
    let unevaluable: HashSet<u64> =
        evals.iter().filter(|eval| eval.value.is_none()).map(|eval| eval.key).collect();
    let flags: Vec<GraphIdentFlag> = identified
        .graphs
        .graph_keys
        .iter()
        .zip(identified.graphs.identified.iter())
        .map(
            |(key, flag)| {
                if unevaluable.contains(key) { GraphIdentFlag::Unidentified } else { *flag }
            },
        )
        .collect();
    let effective = WeightedGraphSamples::new(
        Arc::clone(&identified.graphs.weights),
        flags,
        Arc::clone(&identified.graphs.graph_keys),
    )?;
    let per_graph: Vec<GraphEffectDraws> = evals
        .iter()
        .filter(|eval| eval.value.is_some())
        .map(|eval| GraphEffectDraws { graph_key: eval.key, effect_draws: Arc::clone(&draws) })
        .collect();
    let mut posterior = aggregate_effect_envelope(
        &effective,
        &per_graph,
        InferenceDiagnostics::analytic("transport.graph_posterior_envelope"),
        EnvelopeOptions::default(),
    )
    .map_err(CausalError::from)?;
    // The aggregator flags an unevaluable atom unidentified (the envelope has no third state);
    // split that mass back out so a refusal to estimate is never published as
    // non-identification.
    let total = identified.graphs.total_weight();
    let unevaluable_weight: f64 = identified
        .graphs
        .graph_keys
        .iter()
        .zip(identified.graphs.weights.iter())
        .zip(identified.graphs.identified.iter())
        .filter(|((key, _), flag)| {
            unevaluable.contains(*key) && **flag == GraphIdentFlag::Identified
        })
        .map(|((_, weight), _)| *weight)
        .sum();
    let unevaluable_mass = if total > 0.0 { unevaluable_weight / total } else { 0.0 };
    posterior.unevaluable_mass = unevaluable_mass;
    posterior.unidentified_mass = (posterior.unidentified_mass - unevaluable_mass).max(0.0);
    posterior.assumptions.entries.extend(component.assumptions.entries.iter().cloned());
    Ok(posterior)
}
