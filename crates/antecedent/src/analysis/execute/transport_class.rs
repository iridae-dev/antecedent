//! `TransportQuery` over a supplied CPDAG or PAG.
//!
//! A transport request reads a selection diagram, which is an ADMG. A CPDAG assumes causal
//! sufficiency, so each member of its equivalence class is a DAG; a PAG allows latent
//! confounding, so each MAG completion is read through every ADMG that shares its m-separation
//! model. sID decides every member on its own graph with the declared selection targets.
//!
//! * Members that certify the same formula identify the transported quantity by it: the class
//!   determines the quantity though it may not determine the graph.
//! * Members that certify different formulas leave the quantity an identified set, one case
//!   per distinct formula; the status is partially identified.
//! * A member sID does not certify, and a PAG completion any of whose ADMG candidates sID does
//!   not certify, is unidentified mass: no formula is then correct for every model the class
//!   represents.
//!
//! Every certified case is estimated by the trial-to-target weighting of the supplied
//! selection and treatment probabilities, which reads a formula only through its certificate.
//! The cases that identify and evaluate therefore share one transported contrast and one
//! sampling or posterior law, so the identified set over their values is a point; what the
//! class moves is the identification, and the result states it: which formulas the members
//! certify, the unidentified mass and the mass whose certified formula the weighting cannot
//! evaluate. Completion weights are enumeration weights, not posterior probabilities.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_data::TableView;
use antecedent_estimate::{TransportEffectEstimate, trial_to_target_effect};
use antecedent_identify::{IdentificationEnvelope, TransportIdentification};

use super::pag_path::ClassTag;
use super::transport_interference_path::{
    TrialColumns, bayesian_transport_estimate, frequentist_transport_estimate,
    known_probability_assumption, transport_primary_pair,
};
use super::transport_posterior::ipw_estimable;
use super::transport_refutation::{
    MAX_CLASS_MEMBERS, TransportRefutationInput, refute_transport, refute_transport_members,
};
use super::{
    Arc, CausalError, CausalQuery, ClassAtomOutcome, Diagnostic, DiagnosticKind,
    DiagnosticSeverity, EstimatorId, ExecutionContext, GraphRef, IdentificationStatus,
    IdentifiedExecuteExtras, IdentifiedExecuteFinish, IdentifierId, InferenceMode, Instant,
    PhysicalExecutionPlan, Strategy, StudyResult, TabularData, VariableId, elapsed_ns,
    envelope_to_identification_result_for, identification_status_ok_for_case, overlap_diagnostic,
};
use crate::analysis::IdentificationCache;
use crate::analysis::prepared::{CachedCpdagIdentification, CachedPagIdentification};

impl super::Study {
    /// Identify a transport query over every member of the supplied CPDAG or PAG, retaining the
    /// sID outcome of each case beside the envelope. `None` when the structure is neither.
    ///
    /// # Errors
    ///
    /// A selection target outside the graph, or a query sID does not read.
    pub(crate) fn identify_class_transport(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<Option<IdentificationCache>, CausalError> {
        let CausalQuery::Transport(query) = &self.query else { return Ok(None) };
        let targets: Vec<VariableId> = self
            .selection_diagram
            .as_ref()
            .map(|diagram| diagram.selection_targets().to_vec())
            .unwrap_or_default();
        let identifier = crate::strategy_table::class_identifier_in_context(Some(ctx));
        let target = CausalQuery::Transport(query.clone());
        Ok(match self.graph.graph_ref() {
            GraphRef::Cpdag(cpdag) => {
                let class = identifier.identify_cpdag_transport_envelope(cpdag, &targets, query)?;
                let identification = envelope_to_identification_result_for(&class.envelope, target);
                Some(IdentificationCache::Cpdag(Arc::new(CachedCpdagIdentification {
                    envelope: Arc::new(class.envelope),
                    identification,
                    transport: Some(Arc::from(class.proofs)),
                })))
            }
            GraphRef::Pag(pag) => {
                let class = identifier.identify_pag_transport_envelope(pag, &targets, query)?;
                let identification = envelope_to_identification_result_for(&class.envelope, target);
                Some(IdentificationCache::Pag(Arc::new(CachedPagIdentification {
                    envelope: Arc::new(class.envelope),
                    identification,
                    transport: Some(Arc::from(class.proofs)),
                })))
            }
            _ => None,
        })
    }

    /// Execute the trial-to-target contrast over the members of a supplied CPDAG or PAG.
    ///
    /// # Errors
    ///
    /// A structure that is neither, no member whose formula sID certifies, no certified
    /// formula the weighting can evaluate, invalid probability columns, or a cancelled run.
    pub(super) fn execute_transport_class(
        &self,
        data: &TabularData,
        query: &antecedent_core::TransportQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        query.validate()?;
        let (identification_cache, identify_cached) = self.class_transport_cache(physical, ctx)?;
        match identification_cache {
            IdentificationCache::Cpdag(cache) => {
                let proofs = cache.transport.as_deref().ok_or_else(missing_proofs)?;
                let certificate = crate::Identification::CpdagEnvelope {
                    envelope: (*cache.envelope).clone(),
                    strategy: IdentifierId::TransportSid,
                    structure_version: self.graph.version(),
                };
                self.finish_transport_class(ClassTransportRun {
                    data,
                    query,
                    envelope: &*cache.envelope,
                    proofs,
                    class: ClassTag::Cpdag,
                    identify_cached,
                    certificate,
                    started,
                    physical,
                    ctx,
                })
            }
            IdentificationCache::Pag(cache) => {
                let proofs = cache.transport.as_deref().ok_or_else(missing_proofs)?;
                let certificate = crate::Identification::Envelope {
                    envelope: (*cache.envelope).clone(),
                    strategy: IdentifierId::TransportSid,
                    structure_version: self.graph.version(),
                };
                self.finish_transport_class(ClassTransportRun {
                    data,
                    query,
                    envelope: &*cache.envelope,
                    proofs,
                    class: ClassTag::Pag,
                    identify_cached,
                    certificate,
                    started,
                    physical,
                    ctx,
                })
            }
            _ => Err(CausalError::Unsupported {
                message: "class TransportQuery execute requires a supplied Cpdag or Pag",
            }),
        }
    }

    /// The class transport identification: the prepared envelope, the product the plan's lowered
    /// program retained at compile, or a fresh run of sID on every member.
    fn class_transport_cache(
        &self,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<(IdentificationCache, bool), CausalError> {
        match self.graph.graph_ref() {
            GraphRef::Cpdag(_) => {
                if let Some((cache, prepared)) = self.reuse_cpdag_identification(physical) {
                    return Ok((IdentificationCache::Cpdag(cache), prepared));
                }
            }
            GraphRef::Pag(_) => {
                if let Some((cache, prepared)) = self.reuse_pag_identification(physical) {
                    return Ok((IdentificationCache::Pag(cache), prepared));
                }
            }
            _ => {}
        }
        super::report_identify_compute(ctx);
        let fresh = self.identify_class_transport(ctx)?.ok_or(CausalError::Unsupported {
            message: "class TransportQuery execute requires a supplied Cpdag or Pag",
        })?;
        Ok((fresh, false))
    }

    /// Estimate every certified case of a class envelope and publish the class result.
    #[allow(
        clippy::too_many_lines,
        reason = "one publication: per-case estimates, the representative inference, the class mixture and the disclosures share the envelope's cases"
    )]
    fn finish_transport_class<G>(
        &self,
        run: ClassTransportRun<'_, G>,
    ) -> Result<StudyResult, CausalError> {
        let ClassTransportRun {
            data,
            query,
            envelope,
            proofs,
            class,
            identify_cached,
            certificate,
            started,
            physical,
            ctx,
        } = run;
        if proofs.len() != envelope.cases.len() {
            return Err(CausalError::Compile {
                message: "class transport proofs are not aligned with the envelope cases".into(),
            });
        }
        let trial_spec = self.transport_trial.as_ref().ok_or(CausalError::Unsupported {
            message: "TransportQuery execute requires transport_trial columns",
        })?;
        let (treatment, outcome) = transport_primary_pair(query)?;
        if envelope.identified_weight.0 <= 0.0 {
            return Err(CausalError::not_identified(
                envelope.status,
                envelope.truncated_completions > 0,
                "class TransportQuery: sID certifies no formula for any member of the class",
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

        // Each certified case runs the weighting's own certificate check; only a case whose
        // formula it cannot evaluate is unevaluable, never a data failure.
        let mut estimates: Vec<Option<TransportEffectEstimate>> = vec![None; proofs.len()];
        for (index, case) in envelope.cases.iter().enumerate() {
            if !identification_status_ok_for_case(case.result.status)
                || !ipw_estimable(&proofs[index])
            {
                continue;
            }
            estimates[index] = Some(
                trial_to_target_effect(
                    &proofs[index],
                    &outcomes,
                    &treatment_bool,
                    &trial_bool,
                    &selection,
                    &propensity,
                    None,
                )
                .map_err(CausalError::from)?,
            );
        }
        let Some(representative) = estimates.iter().position(Option::is_some) else {
            return Err(crate::support_reason!(
                "construction_not_licensed",
                "every certified member of the class is a RecursiveFactorization, identified but \
                 not estimable on the licensed trial-to-target IPW path"
            ));
        };
        let (Some(mut transported), representative_case) =
            (estimates[representative], &envelope.cases[representative])
        else {
            return Err(CausalError::Compile {
                message: "class transport representative case lost its estimate".into(),
            });
        };
        let representative_proof = &proofs[representative];
        let mut assumptions = representative_case.result.required_assumptions.clone();
        assumptions.push(known_probability_assumption());
        let series_block = self.transport_series_block(outcomes.len())?;
        let columns = TrialColumns {
            outcomes: &outcomes,
            treatment: &treatment_bool,
            trial: &trial_bool,
            selection: &selection,
            propensity: &propensity,
        };
        let (estimate, mut posterior, estimator_id, estimator_diagnostic) = match &self.inference {
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
                    (representative_proof, series_block),
                    &columns,
                    assumptions,
                    representative_case.result.status,
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

        let bayesian = posterior.is_some();
        let case_outcomes: Vec<ClassAtomOutcome> = estimates
            .iter()
            .map(|estimated| {
                estimated.map_or(ClassAtomOutcome::NotEvaluated, |each| {
                    ClassAtomOutcome::Evaluated(if bayesian { estimate.ate } else { each.ipw })
                })
            })
            .collect();
        let mixture = super::pag_path::static_class_structural_mixture(envelope, &case_outcomes);

        let mut identification =
            envelope_to_identification_result_for(envelope, CausalQuery::Transport(query.clone()));
        let estimand =
            identification.estimands.first().cloned().ok_or_else(|| CausalError::Compile {
                message: "class transport identification carries no estimand".into(),
            })?;
        let (unidentified_mass, unevaluable_mass) = mixture
            .as_ref()
            .map_or((0.0, 0.0), |mixture| (mixture.unidentified_mass, mixture.unevaluable_mass));
        if unevaluable_mass > antecedent_core::POSTERIOR_MASS_TOLERANCE
            && matches!(identification.status, IdentificationStatus::NonparametricallyIdentified)
        {
            identification.status = IdentificationStatus::GraphDependent;
        }
        if let Some(posterior) = posterior.as_mut() {
            posterior.unidentified_mass = unidentified_mass;
            posterior.unevaluable_mass = unevaluable_mass;
            posterior.identification = identification.status;
        }

        let refutation_input = TransportRefutationInput {
            transport: representative_proof,
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
        };
        let mut refutations = refute_transport(&refutation_input, ctx)?;
        // Members that certify a formula with a value of its own are bootstrapped through
        // their own certificate on the same resampled tables.
        let mut seen = vec![estimates[representative].map_or(f64::NAN, |each| each.ipw)];
        let mut members = Vec::new();
        for (index, each) in estimates.iter().enumerate() {
            let Some(each) = each else { continue };
            if index == representative || members.len() + 1 >= MAX_CLASS_MEMBERS {
                continue;
            }
            if seen
                .iter()
                .any(|earlier| (earlier - each.ipw).abs() <= 1e-9 * earlier.abs().max(1.0))
            {
                continue;
            }
            seen.push(each.ipw);
            members.push((index, &proofs[index], each.ipw));
        }
        refutations.extend(refute_transport_members(&refutation_input, &members, ctx)?);

        let class_tag = class.as_str();
        let mut diagnostics = vec![
            if matches!(class, ClassTag::Pag) {
                super::pag_path::pag_envelope_diagnostic(envelope)
            } else {
                super::pag_path::cpdag_envelope_diagnostic(envelope)
            },
            class_transport_diagnostic(class_tag, envelope, proofs, &case_outcomes),
        ];
        if let Some(mixture) = mixture.as_ref() {
            diagnostics.extend(super::pag_path::static_class_identified_set_diagnostic(
                mixture, class_tag,
            ));
        }
        diagnostics.push(estimator_diagnostic);
        diagnostics.push(overlap_diagnostic(estimate.provenance.overlap));

        let mut result = self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand,
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
                n_draws: posterior
                    .as_ref()
                    .map(|each| u32::try_from(each.draws.n_draws).unwrap_or(u32::MAX)),
                posterior,
                diagnostics: Some(diagnostics),
                structural_response: mixture,
                ..Default::default()
            }),
        );
        result.payload = crate::result::ResultPayload::Transport(transported);
        Ok(self.attach_certificate(result, certificate))
    }
}

/// Everything one class transport publication reads.
struct ClassTransportRun<'a, G> {
    data: &'a TabularData,
    query: &'a antecedent_core::TransportQuery,
    envelope: &'a IdentificationEnvelope<G>,
    proofs: &'a [TransportIdentification],
    class: ClassTag,
    identify_cached: bool,
    certificate: crate::Identification,
    started: Instant,
    physical: &'a PhysicalExecutionPlan,
    ctx: &'a ExecutionContext,
}

fn missing_proofs() -> CausalError {
    CausalError::Compile {
        message: "the class identification retains no sID proof for a TransportQuery".into(),
    }
}

/// Which certified formulas the members of the class state and how the weighting treats them.
fn class_transport_diagnostic<G>(
    class_tag: &str,
    envelope: &IdentificationEnvelope<G>,
    proofs: &[TransportIdentification],
    outcomes: &[ClassAtomOutcome],
) -> Diagnostic {
    let mut rules: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    for proof in proofs {
        let rule = match proof {
            TransportIdentification::Transportable { certificate, .. } => {
                certificate.rule.to_string()
            }
            TransportIdentification::NotCertified(certificate) => {
                format!("not_certified:{}", certificate.reason)
            }
            TransportIdentification::MissingEvidence(certificate) => {
                format!("missing_evidence:{}", certificate.reason)
            }
        };
        *rules.entry(rule).or_insert(0) += 1;
    }
    let listed =
        rules.iter().map(|(rule, count)| format!("{rule}={count}")).collect::<Vec<_>>().join(", ");
    let evaluated =
        outcomes.iter().filter(|outcome| matches!(outcome, ClassAtomOutcome::Evaluated(_))).count();
    Diagnostic::new(
        format!("estimate.{class_tag}.transport_sid"),
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        format!(
            "sID decided every member of the {class_tag} class on its own selection diagram; \
             case rules: [{listed}]; identified_weight={} unidentified_weight={}; {evaluated} of \
             {} cases were evaluated by the trial-to-target weighting, which reads a certified \
             formula only through its certificate, so the evaluated cases share one transported \
             contrast: a class whose members certify different formulas is partially identified \
             while the identified set over the values collapses to that contrast; completion \
             weights are enumeration weights, not posterior probabilities",
            envelope.identified_weight.0,
            envelope.unidentified_weight.0,
            envelope.cases.len(),
        ),
    )
}
