//! `InterferenceQuery` over a supplied graph posterior.
//!
//! The exposure contrast is identified by the known assignment mechanism and the exposure
//! mapping on the fixed network, so no edge of any atom of a posterior over structures enters
//! the identification: the contrast, its sampling law or its posterior, are those of the same
//! query on any one structure. The route therefore executes the contrast once, through the
//! executor of the declared assignment design, and carries it through the graph-posterior
//! machinery: every atom that is a member of the posterior's graph class is an identified
//! atom with the one design identification, and an atom that is not a member is unidentified
//! mass the result states. The structural mixture, the posterior mass axes and the status
//! ladder are those of every other graph-posterior route, so the result says what the
//! posterior did and did not change.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_data::TableView;

use super::atom_mix::{AtomBoundRule, AtomMixInput, MixedAtomEvals, mix_atom_evals};
use super::transport_posterior::ScalarAtomEval;
use super::{
    Arc, CausalError, CheckedInterferenceOperation, Diagnostic, DiagnosticKind, DiagnosticSeverity,
    ExecutionContext, GraphPosterior, IdentificationStatus, InferenceMode, PhysicalExecutionPlan,
    PosteriorMassSummary, StudyResult, TabularData, identify_cached_diagnostic,
    push_graph_posterior_structural_aggregation_diagnostic, resolve_posterior_status,
};
use crate::analysis::prepared::CachedGraphPosteriorIdentification;

impl super::Study {
    /// Execute the exposure contrast once and report it over the atoms of a graph posterior.
    ///
    /// # Errors
    ///
    /// A posterior whose atoms do not cover the unit schema or that carries DBN lag structure,
    /// a posterior with no atom that is a member of its graph class, or any refusal of the
    /// declared assignment design's executor.
    pub(super) fn execute_interference_graph_posterior(
        &self,
        data: &TabularData,
        gp: &GraphPosterior,
        query: &antecedent_core::InterferenceQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        query.validate()?;
        if gp.n_vars() != data.schema().len() {
            return Err(CausalError::Unsupported {
                message: "InterferenceQuery over a graph posterior binds the unit schema: the \
                          posterior's atoms must cover exactly the unit table's variables",
            });
        }
        let (identified, identify_cached) =
            self.interference_posterior_identification(gp, query, physical, ctx)?;
        if identified.atoms.is_empty() {
            return Err(CausalError::not_identified(
                IdentificationStatus::NotIdentified,
                false,
                "InterferenceQuery graph posterior: no posterior atom is a member of the \
                 posterior's graph class",
            ));
        }

        // The contrast does not depend on the structure: it is executed once.
        let mut result = if matches!(
            query.assignment,
            antecedent_core::AssignmentDesign::ClusterRandomization { .. }
                | antecedent_core::AssignmentDesign::TwoStageSaturation { .. }
                | antecedent_core::AssignmentDesign::ObservedExposure { .. }
        ) {
            let shared = Arc::new(physical.clone());
            CheckedInterferenceOperation::checked(self, data, &shared)?.execute(data, ctx)?
        } else {
            self.execute_interference(data, query, physical, ctx)?
        };
        let contrast =
            result.estimate.as_effect().map(|estimate| estimate.ate).filter(|ate| ate.is_finite());

        let evals: Vec<ScalarAtomEval> = identified
            .atoms
            .iter()
            .map(|atom| ScalarAtomEval {
                key: atom.key,
                status: atom.identification.status,
                estimand: atom.estimand.clone(),
                value: contrast,
            })
            .collect();
        let unidentified_mass = identified.graphs.unidentified_mass();
        let mixed = mix_atom_evals(
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
                    "interference graph-posterior mix: the contrast is not finite (evals={n}, \
                     mixable={mixable})"
                )
            },
            || "interference graph-posterior envelope: missing estimand".to_string(),
        )?;
        result.identification.status = resolve_posterior_status(
            result.identification.status,
            mixed.policy,
            PosteriorMassSummary {
                unidentified: unidentified_mass,
                unevaluable: mixed.mixture.unevaluable_mass,
                ..PosteriorMassSummary::default()
            },
        );
        if let Some(posterior) = result.view.posterior.as_mut() {
            // The contrast posterior is conditional on the declared design; the structure
            // posterior adds only the mass of atoms that are not members of the graph class.
            let total = identified.graphs.total_weight();
            posterior.unidentified_mass = if total > 0.0 { unidentified_mass / total } else { 0.0 };
            posterior.identification = result.view.identification.status;
        }
        self.annotate_interference_posterior(
            &mut result,
            physical,
            &mixed,
            unidentified_mass,
            identify_cached,
        );
        result.structural_response = Some(mixed.mixture);
        Ok(result)
    }

    fn annotate_interference_posterior(
        &self,
        result: &mut StudyResult,
        physical: &PhysicalExecutionPlan,
        mixed: &MixedAtomEvals,
        unidentified_mass: f64,
        identify_cached: bool,
    ) {
        let algo =
            physical.logical.record.discovery_algorithm.as_deref().unwrap_or("graph_posterior");
        result.diagnostics.push(Diagnostic::new(
            "estimate.graph_posterior.interference_design",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            format!(
                "the exposure contrast is identified by the known assignment mechanism and the \
                 exposure mapping on the fixed network, so no edge of any atom of the {algo} \
                 posterior enters the identification: the contrast and its {} are those of the \
                 same query on one structure, executed once; every posterior atom that is a \
                 member of the graph class carries the one design identification; \
                 unidentified_mass={unidentified_mass} is the mass of atoms that are not members",
                match &self.settings.inference {
                    InferenceMode::Frequentist => "sampling interval",
                    InferenceMode::Bayesian(_) => "posterior",
                }
            ),
        ));
        push_graph_posterior_structural_aggregation_diagnostic(
            &mut result.diagnostics,
            mixed.policy,
            mixed.mixture.identified_mass,
            mixed.mixture.unidentified_mass,
            mixed.mixture.unevaluable_mass,
            mixed.mixture.subsampled_out_mass,
        );
        if identify_cached {
            let cached = "exec.identify.cached";
            if !result.diagnostics.iter().any(|diagnostic| diagnostic.code.as_ref() == cached) {
                result.diagnostics.push(identify_cached_diagnostic());
            }
        }
    }

    /// The frozen per-atom interference identification: the prepared cache, the product the
    /// plan's lowered program retained at compile, or a fresh membership check of every atom.
    fn interference_posterior_identification(
        &self,
        gp: &GraphPosterior,
        query: &antecedent_core::InterferenceQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<(Arc<CachedGraphPosteriorIdentification>, bool), CausalError> {
        if let Some(cache) = self.graph_posterior_cache() {
            return Ok((Arc::clone(cache), true));
        }
        if let Some(cache) =
            crate::analysis::lowering::plan_retained_graph_posterior(self, physical)
        {
            return Ok((cache, false));
        }
        Ok((
            Arc::new(
                crate::analysis::identification_cache::build_interference_graph_posterior_identification_cache(
                    gp,
                    query,
                    &self.settings.inference,
                    ctx,
                )?,
            ),
            false,
        ))
    }
}
