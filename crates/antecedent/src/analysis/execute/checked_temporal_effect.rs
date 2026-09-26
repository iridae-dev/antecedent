//! Builder-independent execution of a checked temporal effect.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

#[derive(Clone)]
pub(crate) struct CheckedTemporalEffectExecution {
    operation: crate::analysis::CheckedTemporalEffectOperation,
    result_context: IdentifiedResultContext,
    physical: PhysicalExecutionPlan,
}

impl std::fmt::Debug for CheckedTemporalEffectExecution {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fmt.debug_struct("CheckedTemporalEffectExecution")
            .field("operation", &self.operation)
            .field("physical", &self.physical)
            .finish_non_exhaustive()
    }
}

impl CheckedTemporalEffectExecution {
    pub(crate) fn checked(
        operation: crate::analysis::CheckedTemporalEffectOperation,
        result_context: IdentifiedResultContext,
        physical: PhysicalExecutionPlan,
    ) -> Result<Self, CausalError> {
        if !operation.matches_query(&result_context.query)
            || !operation.matches_query(&physical.logical.query)
        {
            return Err(CausalError::Compile {
                message: "temporal effect result context does not match its checked query".into(),
            });
        }
        let (identifier, estimator, _, _) = operation.procedure();
        if physical
            .logical
            .record
            .identifier
            .as_deref()
            .is_some_and(|name| name != identifier.as_str())
            || physical
                .logical
                .record
                .estimator
                .as_deref()
                .is_some_and(|name| name != estimator.as_str())
        {
            return Err(CausalError::Compile {
                message: "temporal effect physical plan disagrees on its checked procedure".into(),
            });
        }
        Ok(Self { operation, result_context, physical })
    }

    #[must_use]
    pub(crate) fn operation(&self) -> &crate::analysis::CheckedTemporalEffectOperation {
        &self.operation
    }

    pub(crate) fn execute(
        &self,
        data: &TimeSeriesData,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let op = &self.operation;
        let query = op.query();
        let identification = op.identification().clone();
        let estimand = op.estimand().clone();
        let indexer = op.indexer();
        let (identifier_id, estimator_id, suite, bootstrap_replicates) = op.procedure();
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled {
                stage: crate::analysis::stage::STAGE_ESTIMATE_POINT,
            });
        }

        let (estimate, refutations, mut diagnostics, bootstrap_replicates_ok) = if query
            .is_multi_step_sustained()
        {
            let mut assumptions = identification.required_assumptions.clone();
            assumptions.push(antecedent_core::AssumptionRecord {
                    assumption: antecedent_core::Assumption::ParametricRestriction(
                        antecedent_core::ParametricAssumption {
                            id: Arc::from("temporal.sequential.linear_sem"),
                            description: Arc::from("linear additive mechanisms on the identified unfolded DAG; every sustained time is intervened on; frequentist intervals use a shared circular-block row bootstrap"),
                        },
                    ),
                    source: antecedent_core::AssumptionSource::AlgorithmDefault {
                        algorithm: Arc::from("temporal.sequential.gcomp"),
                    },
                    scope: antecedent_core::AssumptionScope::Estimation,
                    status: antecedent_core::AssumptionStatus::Declared,
                });
            let mut mechanisms = Vec::new();
            let (estimate, posterior) =
                    antecedent_estimate::temporal_sequential::estimate_sustained_window_with_validation(
                        data, op.graph(), indexer, &estimand, query, identification.status,
                        assumptions, bootstrap_replicates, None, ctx, Some(&mut mechanisms),
                    )
                    .map_err(CausalError::from)?;
            debug_assert!(posterior.is_none());
            let atom = super::sequential_validation::SequentialValidationAtom {
                weight: 1.0,
                graph: op.graph().clone(),
                indexer: indexer.clone(),
                estimand: estimand.clone(),
                status: identification.status,
                estimate: estimate.clone(),
                mechanisms,
            };
            let (refutations, mut diagnostics, _) =
                super::sequential_validation::validate_sequential(
                    data,
                    query,
                    &[atom],
                    suite,
                    &[],
                    None,
                    None,
                    estimate.ate,
                    ctx,
                    0,
                )?;
            diagnostics.push(Diagnostic::new(
                "estimate.temporal.sustained_window",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "the contrast propagates through every intervened time in topological order",
            ));
            diagnostics.extend(super::temporal_path::sequential_dependence_se_diagnostics(
                &estimate,
                bootstrap_replicates,
            ));
            let estimate =
                estimate.with_block_family(antecedent_estimate::CircularBlockFamily::Sequential);
            let ok = estimate.bootstrap_replicates_ok;
            (estimate, refutations, diagnostics, ok)
        } else {
            let mut estimator = TemporalLinearAdjustment::new();
            estimator.inner.bootstrap_replicates = bootstrap_replicates;
            estimator.inner.overlap = OverlapPolicy::ExplicitOverride;
            let prep = estimator
                .prepare(data, &estimand, query, indexer, None, &ctx.kernel_policy)
                .map_err(CausalError::from)?;
            let (estimate, dependence) = estimator
                .fit_dependence_honest(
                    &prep,
                    indexer,
                    ctx,
                    identification.required_assumptions.clone(),
                )
                .map_err(CausalError::from)?;
            let estimate = estimate
                .with_block_family(antecedent_estimate::CircularBlockFamily::SingleWindow)
                .with_n_obs(u64::try_from(prep.design.nrows).unwrap_or(u64::MAX));
            let tabular = TabularData::new(data.storage().clone());
            let mut average = AverageEffectQuery::binary_ate(query.treatment, query.outcome);
            average.active = query.active.clone();
            average.control = query.control.clone();
            average.target_population = query.target_population.clone();
            let mut workspace = EstimationWorkspace::default();
            let temporal_ctx = TemporalRefitContext {
                indexer,
                temporal_query: query,
                split: None,
                kernel_policy: &ctx.kernel_policy,
                time_index: Some(data.time_index()),
                panel: None,
            };
            let (refutations, mut diagnostics) = run_refuters(
                &tabular,
                &estimand,
                &average,
                &estimate,
                &mut workspace,
                None,
                ctx,
                suite,
                estimator_id.as_str(),
                &[],
                Some(temporal_ctx),
            )?;
            diagnostics.extend(super::temporal_path::temporal_dependence_se_diagnostics(
                antecedent_estimate::CircularBlockFamily::SingleWindow,
                dependence.block_length,
                dependence.rows,
                dependence.kernel_bias,
                dependence.effective_rows,
                dependence.replicates_attempted > 0,
                &format!(
                    "se_bootstrap refits the lag-aligned design on circular blocks of \
                     consecutive rows ({}/{} replicates); se_analytic is NaN: the iid OLS SE \
                     ignores serial dependence and no HAC SE is calibrated for this cell{}",
                    estimate.bootstrap_replicates_ok.unwrap_or(0),
                    dependence.replicates_attempted,
                    if dependence.replicates_attempted == 0 {
                        "; request bootstrap_replicates > 0 for an interval"
                    } else {
                        ""
                    },
                ),
            ));
            let ok = estimate.bootstrap_replicates_ok;
            (estimate, refutations, diagnostics, ok)
        };

        diagnostics.push(identify_cached_diagnostic());
        let cancelled = estimate.bootstrap_cancelled;
        let args = IdentifiedExecuteFinish {
            physical: &self.physical,
            identification: identification.clone(),
            estimand,
            estimate,
            identifier_id,
            estimator_id,
            treatment: query.treatment,
            outcome: query.outcome,
            identify_cached: true,
            extra_diagnostics: Vec::new(),
            refutations,
            distribution: None,
            mediation: None,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            bootstrap_replicates_ok,
            cancelled,
            early_stopped: false,
            extras: IdentifiedExecuteExtras {
                certificate: Some(crate::Identification::Point {
                    result: identification,
                    temporal_indexer: Some(indexer.clone()),
                    strategy: identifier_id,
                    structure_version: self.result_context.graph_version,
                }),
                identify_provenance: Some(provenance_ids(
                    "identify.temporal_backdoor",
                    "identify.temporal.backdoor.unfolded",
                )),
                estimate_provenance: Some(provenance_ids(
                    if query.is_multi_step_sustained() {
                        "estimate.temporal_sequential_gcomp"
                    } else {
                        "estimate.temporal_linear_adjustment"
                    },
                    if query.is_multi_step_sustained() {
                        "estimate.temporal.sequential.gcomp"
                    } else {
                        "estimate.temporal.linear.adjustment"
                    },
                )),
                diagnostics: Some(diagnostics),
                ..Default::default()
            },
        };
        Ok(super::finish_identified_execute_with_context(&self.result_context, None, args))
    }
}
