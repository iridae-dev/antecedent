//! Re-execution and target-preserving readback of refits.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    CausalError, DataInput, ExecutionContext, LoweredProgram, Modal, PreparedRefit,
    PreparedTargetData, StudyResult, TargetRefitError, TimeSeriesData, ValidationError,
    check_static_response_contributors, check_temporal_response_contributors,
    exact_admg_contributors, failed, result_target_values, same_admg_posterior_source,
    same_static_class_witnesses, temporal_response_member_values, unsupported,
};

impl PreparedRefit<'_> {
    /// Re-execute this handle's program on `data`.
    pub(super) fn reestimate(
        &self,
        data: &Modal,
        ctx: &ExecutionContext,
    ) -> Result<Vec<f64>, TargetRefitError> {
        let refitted = match data {
            Modal::Tabular(data) => self.prepared.estimate(data, ctx),
            Modal::Series(data) => self.prepared.estimate_series(data, ctx),
            Modal::Panel(data) => self.prepared.estimate_panel(data, ctx),
            Modal::MultiEnv(data) => self.prepared.estimate_multi_env(data, ctx),
        }
        .map_err(|error| failed(&error))?;
        self.read_back(&refitted, true, ctx)
    }

    pub(super) fn reestimate_materialised_series(
        &self,
        data: &TimeSeriesData,
        provenance: Option<&crate::analysis::execute::TemporalRowProvenance>,
        ctx: &ExecutionContext,
    ) -> Result<Vec<f64>, TargetRefitError> {
        let result = self
            .prepared
            .estimate_series_with_provenance(data, provenance, ctx)
            .map_err(|error| failed(&error))?;
        self.read_back(&result, true, ctx)
    }

    pub(super) fn reestimate_gathered_series(
        &self,
        data: &TimeSeriesData,
        source_ticks: &[usize],
        block_length: Option<usize>,
        user_sampling_weights: bool,
        budget: &mut antecedent_core::TemporalBudget,
        ctx: &ExecutionContext,
    ) -> Result<Vec<f64>, TargetRefitError> {
        let provenance = crate::analysis::execute::TemporalRowProvenance::admit_fixed_blocks(
            source_ticks,
            block_length,
            user_sampling_weights,
            budget,
            ctx,
        )
        .map_err(|error| failed(&error))?;
        let result = self
            .prepared
            .estimate_series_with_provenance(data, Some(&provenance), ctx)
            .map_err(|error| failed(&error))?;
        self.read_back(&result, true, ctx)
    }

    /// The refitted target vector, in the original coordinate order.
    #[allow(
        clippy::too_many_lines,
        reason = "source, contributor and coordinate checks share one bounded refit"
    )]
    pub(super) fn read_back(
        &self,
        refitted: &StudyResult,
        same_prepared: bool,
        ctx: &ExecutionContext,
    ) -> Result<Vec<f64>, TargetRefitError> {
        let mut budget = self.temporal_budget.clone();
        if matches!(self.data, PreparedTargetData::Tabular(_)) && self.result.response().is_some() {
            match (&self.result.structural_response, &refitted.structural_response) {
                (Some(before), Some(after)) => {
                    // Graph keys are opaque, not globally collision-resistant. They
                    // identify masks only within this exact retained DAG posterior:
                    // construction rejects collisions between distinct adjacency masks.
                    let class_witnesses_match =
                        same_static_class_witnesses(self.result, refitted, &mut budget, ctx)?;
                    let program =
                        self.prepared.plan().lowered().and_then(LoweredProgram::semantic_identity);
                    let same_program = program.is_some()
                        && program == self.result.lowered_program_identity()
                        && program == refitted.lowered_program_identity();
                    let mut admg_source_matches = false;
                    let admg_match = if same_prepared && same_program {
                        if let Some(posterior) = self.study().graph_posterior.as_ref() {
                            admg_source_matches = same_admg_posterior_source(
                                posterior,
                                self.result.static_posterior_source.as_deref(),
                                refitted.static_posterior_source.as_deref(),
                            );
                            exact_admg_contributors(
                                posterior,
                                before,
                                after,
                                admg_source_matches,
                                &mut budget,
                                ctx,
                            )?
                        } else {
                            false
                        }
                    } else {
                        false
                    };
                    let posterior = self.study().graph_posterior.as_ref().filter(|posterior| {
                        same_prepared && (posterior.atom_kind() == antecedent_discovery::GraphPosteriorAtomKind::Dag
                            || admg_match
                            || (same_program && matches!(posterior.atom_kind(), antecedent_discovery::GraphPosteriorAtomKind::Cpdag | antecedent_discovery::GraphPosteriorAtomKind::Pag) && class_witnesses_match))
                    }).ok_or_else(|| unsupported(
                        "structural contributor identities cannot be verified for this refit",
                        "retain full completion identities for the refitted structure",
                    ))?;
                    let class_proven = same_prepared
                        && same_program
                        && class_witnesses_match
                        && matches!(
                            posterior.atom_kind(),
                            antecedent_discovery::GraphPosteriorAtomKind::Cpdag
                                | antecedent_discovery::GraphPosteriorAtomKind::Pag
                        );
                    let source_bound_ordered = admg_match && admg_source_matches;
                    if !source_bound_ordered {
                        let lookup_work = u64::try_from(before.atoms.len())
                            .ok()
                            .and_then(|n| {
                                u64::try_from(posterior.graph_keys().len())
                                    .ok()
                                    .and_then(|m| n.checked_mul(m))
                            })
                            .ok_or_else(|| {
                                failed(&CausalError::Resource {
                                    message: "static contributor lookup overflow".into(),
                                })
                            })?;
                        budget.charge_work(lookup_work).map_err(|error| {
                            failed(&CausalError::Resource { message: error.to_string() })
                        })?;
                        if !class_proven && before.atoms.len() != posterior.graph_keys().len() {
                            return Err(unsupported(
                                "structural contributor key is absent or ambiguous in the retained posterior",
                                "retain unique semantic graph identities",
                            ));
                        }
                        for atom in &before.atoms {
                            let mut matches = 0usize;
                            for key in posterior.graph_keys().iter() {
                                if ctx.cancellation.is_cancelled() {
                                    return Err(failed(&CausalError::Cancelled {
                                        stage: "static contributor lookup",
                                    }));
                                }
                                if *key == atom.graph_key {
                                    matches += 1;
                                }
                            }
                            if matches == 0 || (!class_proven && matches != 1) {
                                return Err(unsupported(
                                    "structural contributor key is absent or ambiguous in the retained posterior",
                                    "retain unique semantic graph identities",
                                ));
                            }
                        }
                    }
                    check_static_response_contributors(
                        before,
                        after,
                        source_bound_ordered,
                        &mut budget,
                        ctx,
                    )?;
                }
                (None, None) => {}
                _ => {
                    return Err(unsupported(
                        "refit changed structural response availability",
                        "refit the same identified structural contributors",
                    ));
                }
            }
        }
        check_temporal_response_contributors(
            self.result,
            refitted,
            &mut budget,
            self.member.is_some(),
            ctx,
        )?;
        if let Some(selector) = &self.member {
            let values = temporal_response_member_values(refitted, selector, &mut budget, ctx)
                .map_err(|error| match error {
                    CausalError::Cancelled { .. } | CausalError::Resource { .. } => failed(&error),
                    _ => unsupported(error.to_string(), "refit the original completion"),
                })?;
            if values.len() != self.target.labels().len() {
                return Err(TargetRefitError::Failed(ValidationError::estimation_msg(
                    "the refitted completion reports a different target dimension",
                )));
            }
            return Ok(values);
        }
        let (family, labels, values) =
            result_target_values(refitted).map_err(|error| failed(&error))?;
        if family != self.target.family() || labels.as_slice() != self.target.labels() {
            return Err(TargetRefitError::Failed(ValidationError::estimation_msg(
                "the refitted result reports a different target than the original",
            )));
        }
        Ok(values)
    }

    /// Prepare `study` (a one-change copy of this handle's study) on `data` and execute it.
    pub(super) fn compile_and_run(
        mut study: crate::analysis::execute::Study,
        data: &Modal,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        study.settings.refute = crate::analysis::builder::RefuteSuite::None;
        study.data = match data {
            Modal::Tabular(d) => DataInput::Tabular(d.clone()),
            Modal::Series(d) => match &study.data {
                DataInput::Event(_) => DataInput::Event(d.clone()),
                _ => DataInput::Temporal(d.clone()),
            },
            Modal::Panel(d) => DataInput::Panel(d.clone()),
            Modal::MultiEnv(d) => DataInput::MultiEnv(d.clone()),
        };
        let prepared = study.prepare(ctx)?;
        match data {
            Modal::Tabular(d) => prepared.estimate(d, ctx),
            Modal::Series(d) => prepared.estimate_series(d, ctx),
            Modal::Panel(d) => prepared.estimate_panel(d, ctx),
            Modal::MultiEnv(d) => prepared.estimate_multi_env(d, ctx),
        }
    }
}
