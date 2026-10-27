//! Path-restricted natural effects and natural direct effects over the completions of a CPDAG
//! or the ADMGs a PAG leaves open.
//!
//! A CPDAG assumes causal sufficiency, so each member of its equivalence class is a DAG on
//! which the Avin-Shpitser-Pearl identifier decides the effect. Every identified completion is
//! estimated by the same finite-discrete functional plug-in (bootstrap interval, or Bayesian
//! bootstrap posterior) as the supplied-DAG cell. Completions that identify the same effect
//! keep that estimate and its uncertainty; completions that disagree leave the effect
//! partially identified, and the result publishes the enumeration-weighted mean with the
//! identified set over the completions. Completion weights are enumeration weights, not
//! posterior probabilities, and a completion with a recanting witness is unidentified mass.
//!
//! A PAG allows latent confounding, and a MAG completion fixes the latent projection only up
//! to its m-separation model, which does not determine a path-specific effect. Each MAG
//! completion is therefore read through every ADMG that shares its m-separation model (see
//! the PAG identifier), and the effect is reported only when each of those yields an
//! identified functional: the same functional is a point, different functionals are the
//! identified set, and a candidate the identifier refuses refuses the whole effect, since no
//! functional is then correct for every model the PAG represents.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::pag_path::ClassTag;
use super::{
    CausalError, ClassAtomOutcome, Diagnostic, EffectEstimate, EstimatorId, ExecutionContext,
    IdentificationEnvelope, IdentificationStatus, IdentifiedExecuteExtras, IdentifiedExecuteFinish,
    IdentifierId, Instant, OverlapPolicy, PathSpecificMixedSource, PhysicalExecutionPlan, Strategy,
    StudyResult, TableView, TabularData, elapsed_ns, envelope_se_omission_diagnostic,
    envelope_to_identification_result_for, identification_status_ok_for_case, overlap_diagnostic,
    report_identify_compute, select_estimand,
};
use crate::planner::RouteDefaults;

/// One class path-specific run: the envelope its identifier produced and what names it.
pub(super) struct ClassPathRun<'a, G> {
    pub(super) envelope: &'a IdentificationEnvelope<G>,
    pub(super) class: ClassTag,
    pub(super) envelope_diagnostic: Diagnostic,
    pub(super) certificate: crate::Identification,
    pub(super) identifier_id: IdentifierId,
    pub(super) estimator_id: EstimatorId,
    pub(super) started: Instant,
    /// How the rows are resampled when the table's rows overlap (a lag-aligned window table);
    /// `None` reads the study's own row resampling.
    pub(super) resampling: Option<(
        antecedent_estimate::RowResampling,
        antecedent_estimate::util::SamplingDeclaration,
    )>,
}

/// Ordered fit products of the identified class completions.
struct ClassPathEstimates {
    outcomes: Vec<ClassAtomOutcome>,
    evaluated: Vec<(f64, EffectEstimate, Option<super::CausalPosterior>)>,
    prepared_completions:
        Vec<(antecedent_estimate::FunctionalEffect, antecedent_estimate::PreparedFunctionalEffect)>,
    first_estimand: Option<super::IdentifiedEstimand>,
    assumptions: antecedent_core::AssumptionSet,
}

impl super::Study {
    /// Identify the effect in every DAG completion of `cpdag`, estimate the identified
    /// completions and publish the class-level result.
    pub(super) fn execute_class_path_specific(
        &self,
        data: &TabularData,
        cpdag: &antecedent_graph::Cpdag,
        source: PathSpecificMixedSource<'_>,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        super::static_path::ensure_admg_path_specific_licensed(self.settings.refute)?;
        let started = Instant::now();
        let (identifier_id, estimator_id) = physical.binding().resolve(RouteDefaults::PATH)?;
        require_functional_estimator(estimator_id)?;
        let (identified_query, _, _, _) = super::static_path::path_specific_source_parts(source);

        report_identify_compute(ctx);
        let envelope = crate::strategy_table::identify_cpdag_path_specific_in_context(
            identifier_id,
            cpdag,
            &identified_query,
            Some(ctx),
        )?;
        if matches!(envelope.status, IdentificationStatus::NotIdentified)
            || envelope.identified_weight.0 <= 0.0
        {
            return Err(CausalError::not_identified(
                envelope.status,
                envelope.truncated_completions > 0,
                "CPDAG path-specific effect not identified (no identified mass in envelope)",
            ));
        }
        let certificate = crate::Identification::CpdagEnvelope {
            envelope: envelope.clone(),
            strategy: identifier_id,
            structure_version: self.graph.version(),
        };
        let envelope_diagnostic = super::pag_path::cpdag_envelope_diagnostic(&envelope);
        self.finish_class_path_specific(
            data,
            source,
            ClassPathRun {
                envelope: &envelope,
                class: ClassTag::Cpdag,
                envelope_diagnostic,
                certificate,
                identifier_id,
                estimator_id,
                started,
                resampling: None,
            },
            physical,
            ctx,
        )
    }

    /// Identify the effect over the ADMGs every MAG completion of `pag` stands for, estimate
    /// each distinct identified functional and publish the class-level result. The effect is
    /// refused as soon as one candidate ADMG leaves it unidentified: no functional is then
    /// correct for every model the PAG represents.
    pub(super) fn execute_pag_path_specific(
        &self,
        data: &TabularData,
        pag: &antecedent_graph::Pag,
        source: PathSpecificMixedSource<'_>,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        super::static_path::ensure_admg_path_specific_licensed(self.settings.refute)?;
        let started = Instant::now();
        let (identifier_id, estimator_id) = physical.binding().resolve(RouteDefaults::PATH)?;
        require_functional_estimator(estimator_id)?;
        let (identified_query, _, _, _) = super::static_path::path_specific_source_parts(source);

        report_identify_compute(ctx);
        let envelope = crate::strategy_table::identify_pag_path_specific_in_context(
            identifier_id,
            pag,
            &identified_query,
            Some(ctx),
        )?;
        if envelope.unidentified_weight.0 > 0.0
            || envelope.identified_weight.0 <= 0.0
            || matches!(envelope.status, IdentificationStatus::NotIdentified)
        {
            return Err(CausalError::not_identified(
                envelope.status,
                envelope.truncated_completions > 0,
                "PAG path-specific effect not identified: an ADMG sharing the m-separation \
                 model of a MAG completion leaves the effect unidentified, so no functional is \
                 correct for every model the PAG represents",
            ));
        }
        let certificate = crate::Identification::Envelope {
            envelope: envelope.clone(),
            strategy: identifier_id,
            structure_version: self.graph.version(),
        };
        let envelope_diagnostic = super::pag_path::pag_envelope_diagnostic(&envelope);
        self.finish_class_path_specific(
            data,
            source,
            ClassPathRun {
                envelope: &envelope,
                class: ClassTag::Pag,
                envelope_diagnostic,
                certificate,
                identifier_id,
                estimator_id,
                started,
                resampling: None,
            },
            physical,
            ctx,
        )
    }

    /// Evaluate completions in envelope order, retaining their matched bootstrap preparations.
    fn estimate_class_path_completions<G>(
        &self,
        data: &TabularData,
        envelope: &IdentificationEnvelope<G>,
        estimator_id: EstimatorId,
        roles: &[antecedent_core::VariableId],
        resampling: Option<(
            antecedent_estimate::RowResampling,
            antecedent_estimate::util::SamplingDeclaration,
        )>,
        ctx: &ExecutionContext,
    ) -> Result<ClassPathEstimates, CausalError> {
        let mut outcomes = vec![ClassAtomOutcome::NotEvaluated; envelope.cases.len()];
        let mut evaluated = Vec::new();
        let mut prepared_completions = Vec::new();
        let mut first_estimand = None;
        let mut assumptions = antecedent_core::AssumptionSet::default();
        for (index, case) in envelope.cases.iter().enumerate() {
            if !identification_status_ok_for_case(case.result.status)
                || case.result.estimands.is_empty()
            {
                continue;
            }
            let estimand = select_estimand(&case.result, estimator_id)?;
            first_estimand.get_or_insert_with(|| estimand.clone());
            let (estimate, posterior) = match resampling {
                Some(resampling) => self.estimate_functional_effect_resampled(
                    data,
                    &estimand,
                    &case.result,
                    roles,
                    resampling,
                    ctx,
                )?,
                None => {
                    self.estimate_functional_effect(data, &estimand, &case.result, roles, ctx)?
                }
            };
            outcomes[index] = ClassAtomOutcome::Evaluated(estimate.ate);
            if evaluated.is_empty() {
                assumptions = estimate.provenance.assumptions.clone();
            }
            if self.settings.bootstrap_replicates > 0 && !self.settings.inference.is_bayesian() {
                let (row_resampling, sampling_design) = match resampling {
                    Some(resampling) => resampling,
                    None => self.functional_row_resampling()?,
                };
                let fitter = antecedent_estimate::FunctionalEffect {
                    bootstrap_replicates: self.settings.bootstrap_replicates,
                    ..antecedent_estimate::FunctionalEffect::new()
                }
                .with_row_resampling(row_resampling, sampling_design);
                let prepared = fitter
                    .prepare(
                        data,
                        &estimand,
                        &case.result.arena,
                        case.result.required_assumptions.clone(),
                        roles,
                    )
                    .map_err(CausalError::from)?;
                prepared_completions.push((fitter, prepared));
            }
            evaluated.push((case.weight.0, estimate, posterior));
        }
        Ok(ClassPathEstimates {
            outcomes,
            evaluated,
            prepared_completions,
            first_estimand,
            assumptions,
        })
    }

    fn class_path_intervals(
        &self,
        rows: usize,
        truncated: bool,
        evaluated: &[(f64, EffectEstimate, Option<super::CausalPosterior>)],
        prepared_completions: &[(
            antecedent_estimate::FunctionalEffect,
            antecedent_estimate::PreparedFunctionalEffect,
        )],
        mixture: &mut Option<crate::result::StructuralResponseMixture>,
        ctx: &ExecutionContext,
    ) -> Result<(), CausalError> {
        // Completions that disagree: an Imbens-Manski interval for the identified set, from
        // every completion refit on the same resampled rows.
        if let Some(mixture) = mixture.as_mut() {
            if self.settings.inference.is_bayesian()
                && evaluated.iter().all(|(_, _, p)| p.is_some())
            {
                mixture.identified_set_interval = super::posterior_identified_set_interval(
                    evaluated.iter().filter_map(|(_, _, p)| p.as_ref()),
                    rows,
                    truncated,
                );
            }
        }
        if let (Some(mixture), Some((fitter, _))) = (mixture.as_mut(), prepared_completions.first())
        {
            let refs: Vec<&antecedent_estimate::PreparedFunctionalEffect> =
                prepared_completions.iter().map(|(_, prepared)| prepared).collect();
            let draws = fitter.shared_replicate_draws(&refs, ctx).map_err(CausalError::from)?;
            let points: Vec<f64> = evaluated.iter().map(|(_, estimate, _)| estimate.ate).collect();
            if points.len() == refs.len() {
                mixture.identified_set_interval =
                    antecedent_estimate::imbens_manski_shared_replicates(
                        &points, &draws, 1.0, rows, 0.95,
                    );
            }
        }
        Ok(())
    }

    /// Estimate every identified case of a class envelope and publish the class result.
    pub(super) fn finish_class_path_specific<G>(
        &self,
        data: &TabularData,
        source: PathSpecificMixedSource<'_>,
        run: ClassPathRun<'_, G>,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let ClassPathRun {
            envelope,
            class,
            envelope_diagnostic,
            certificate,
            identifier_id,
            estimator_id,
            started,
            resampling,
        } = run;
        let (_, treatment, outcome, roles) = super::static_path::path_specific_source_parts(source);
        let identification = envelope_to_identification_result_for(envelope, self.query.clone());

        let ClassPathEstimates {
            outcomes,
            mut evaluated,
            prepared_completions,
            first_estimand,
            assumptions,
        } = self.estimate_class_path_completions(
            data,
            envelope,
            estimator_id,
            &roles,
            resampling,
            ctx,
        )?;
        let total: f64 = evaluated.iter().map(|(weight, _, _)| *weight).sum();
        if evaluated.is_empty()
            || !matches!(total.partial_cmp(&0.0), Some(std::cmp::Ordering::Greater))
        {
            return Err(CausalError::Compile {
                message: format!(
                    "{} path-specific envelope had no estimable identified completions",
                    class.as_str().to_uppercase()
                ),
            });
        }

        let class_tag = class.as_str();
        let mut mixture = super::pag_path::static_class_structural_mixture(envelope, &outcomes);
        self.class_path_intervals(
            data.row_count(),
            envelope.truncated_completions > 0,
            &evaluated,
            &prepared_completions,
            &mut mixture,
            ctx,
        )?;
        let mut diagnostics = vec![envelope_diagnostic];
        if let Some(mixture) = mixture.as_ref() {
            diagnostics.extend(super::pag_path::static_class_identified_set_diagnostic(
                mixture, class_tag,
            ));
            if let Some(interval) = mixture.identified_set_interval.as_ref() {
                diagnostics.extend(super::identified_set_interval_diagnostics(interval));
            }
        }
        let (estimate, posterior) = if evaluated.len() == 1 || mixture.is_none() {
            // One identified completion, or completions that identify the same effect: a
            // point, with the uncertainty of its own fit.
            let (_, estimate, posterior) = evaluated.remove(0);
            (estimate, posterior)
        } else {
            let weighted: f64 =
                evaluated.iter().map(|(weight, estimate, _)| weight * estimate.ate).sum();
            let pooled = EffectEstimate::new(
                weighted / total,
                f64::NAN,
                assumptions,
                OverlapPolicy::ExplicitOverride,
            );
            diagnostics
                .extend(envelope_se_omission_diagnostic(evaluated.len(), pooled.se_analytic()));
            (pooled, None)
        };
        diagnostics.push(overlap_diagnostic(estimate.provenance.overlap));

        // Completions that identify different functionals list one estimand each in the combined
        // identification; the result names the first evaluated completion's functional, the
        // others are disclosed through the structural mixture.
        let estimand = match first_estimand {
            Some(estimand) if identification.estimands.len() > 1 => estimand,
            _ => select_estimand(&identification, estimator_id)?,
        };
        let result = self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand,
                estimate,
                Strategy::new(identifier_id, estimator_id),
                treatment,
                outcome,
            )
            .identify_cached(false)
            .wall_time_ns(elapsed_ns(started))
            .extras(IdentifiedExecuteExtras {
                n_draws: posterior
                    .as_ref()
                    .map(|p| u32::try_from(p.draws.n_draws).unwrap_or(u32::MAX)),
                posterior,
                diagnostics: Some(diagnostics),
                structural_response: mixture,
                ..Default::default()
            }),
        );
        Ok(self.attach_certificate(result, certificate))
    }
}

/// The class path-specific cells evaluate the identified functional by the finite-discrete
/// plug-in only.
fn require_functional_estimator(estimator_id: EstimatorId) -> Result<(), CausalError> {
    if matches!(estimator_id, EstimatorId::FunctionalEffect) {
        return Ok(());
    }
    Err(CausalError::Compile {
        message: format!(
            "PathSpecific execute requires estimator functional.effect; got {estimator_id}"
        ),
    })
}
