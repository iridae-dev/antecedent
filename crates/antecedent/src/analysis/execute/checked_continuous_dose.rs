//! Retained graphless conditional continuous-dose response execution.
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use antecedent_core::{
    Assumption, AssumptionRecord, AssumptionScope, AssumptionSource, AssumptionStatus,
};

#[derive(Clone)]
pub(crate) struct CheckedContinuousDoseOperation {
    query: antecedent_core::ContinuousDoseResponseQuery,
    schema: antecedent_core::CausalSchema,
    rows: usize,
    identification: IdentificationResult,
    estimand: IdentifiedEstimand,
    physical: PhysicalExecutionPlan,
    result_context: IdentifiedResultContext,
}

impl std::fmt::Debug for CheckedContinuousDoseOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CheckedContinuousDoseOperation")
            .field("query", &self.query)
            .field("rows", &self.rows)
            .finish_non_exhaustive()
    }
}

impl CheckedContinuousDoseOperation {
    pub(crate) fn checked(
        study: &Study,
        data: &TabularData,
        physical: &PhysicalExecutionPlan,
    ) -> Result<Self, CausalError> {
        let CausalQuery::ContinuousDoseResponse(query) = &study.query else {
            return Err(CausalError::Compile {
                message: "continuous-dose operation requires ContinuousDoseResponseQuery".into(),
            });
        };
        query.validate().map_err(|error| CausalError::Compile { message: error.to_string() })?;
        if study.graph.class() != GraphClass::RandomizedTrial
            || study.structure_source != crate::support::StructureSource::RandomizedTrial
            || !matches!(study.inference, InferenceMode::Frequentist)
            || study.refute != RefuteSuite::None
            || study.bootstrap_replicates != 0
            || !study.custom_validators.is_empty()
            || study.graph_posterior.is_some()
            || study.tiered.is_some()
            || query.baseline_groups.len() != data.row_count()
        {
            return Err(CausalError::Unsupported {
                message: "continuous-dose response requires a graphless fixed baseline-group table and point-only frequentist execution",
            });
        }
        for id in [query.outcome, query.dose, query.dose_density] {
            data.schema()
                .get(id)
                .map_err(|error| CausalError::Compile { message: error.to_string() })?;
        }
        if physical.logical.query != study.query
            || physical.logical.record.identifier.as_deref()
                != Some(IdentifierId::ContinuousDoseExchangeability.as_str())
            || physical.logical.record.estimator.as_deref()
                != Some(EstimatorId::ContinuousDoseKernel.as_str())
        {
            return Err(CausalError::Compile {
                message: "continuous-dose design differs from compiled plan".into(),
            });
        }
        let (identification, estimand) = continuous_dose_identification(query);
        Ok(Self {
            query: query.clone(),
            schema: data.schema().clone(),
            rows: data.row_count(),
            identification,
            estimand,
            physical: physical.clone(),
            result_context: IdentifiedResultContext::from_study(study),
        })
    }

    pub(crate) fn execute(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        if data.schema() != &self.schema || data.row_count() != self.rows {
            return Err(CausalError::Unsupported {
                message: "continuous-dose refresh requires prepared schema and row order",
            });
        }
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled {
                stage: super::super::stage::STAGE_ESTIMATE_POINT,
            });
        }
        let numeric = |id| match data.column(id).map_err(CausalError::from)? {
            antecedent_data::ColumnView::Float64(column) => Ok(column.values.as_slice().to_vec()),
            _ => Err(CausalError::Unsupported {
                message: "continuous-dose outcome, dose, and density columns must be numeric",
            }),
        };
        let outcome = numeric(self.query.outcome)?;
        let dose = numeric(self.query.dose)?;
        let density = numeric(self.query.dose_density)?;
        let groups: Vec<String> =
            self.query.baseline_groups.iter().map(ToString::to_string).collect();
        let points = if self.query.target_doses.is_empty() { Vec::new() } else {
            antecedent_estimate::continuous_dose::conditional_dose_response(
                &outcome, &dose, &groups, &density, &self.query.target_doses,
                self.query.bandwidth, self.query.min_local_support,
            ).map_err(|message| CausalError::Compile { message })?
        };
        let fixed_policy = self.query.fixed_policy.as_ref().map(|rule| {
            let policy = rule.policy_doses.iter().map(|(group, dose)| (group.to_string(), *dose))
                .collect::<Vec<_>>();
            let reference = rule.reference_doses.iter().map(|(group, dose)| (group.to_string(), *dose))
                .collect::<Vec<_>>();
            antecedent_estimate::continuous_dose::fixed_dose_policy_value(
                &outcome, &dose, &groups, &density, &policy, &reference,
                self.query.bandwidth, self.query.min_local_support,
                self.query.density_provenance.as_ref() == "known",
            ).map_err(|message| CausalError::Compile { message })
        }).transpose()?;
        let licensed_fixed_policy = fixed_policy.as_ref().is_some_and(|value| {
            let mut group_rows = std::collections::BTreeMap::<&str, usize>::new();
            for group in &groups {
                *group_rows.entry(group.as_str()).or_default() += 1;
            }
            let intervals = [value.policy_interval_95, value.reference_interval_95,
                value.incremental_interval_95];
            crate::support::license_if_graphless(
                crate::support::GraphlessSupportKey {
                    family: "continuous_dose_policy",
                    design: "fixed_group_kernel",
                    method: "inverse_density_kernel_paired_scores",
                    inference_claim: "policy_reference_incremental_pointwise_95_normal_intervals",
                },
                crate::support::GraphlessAssignmentSupport {
                    assignment_unit: "unit", rows: self.rows,
                    min_group_rows: group_rows.values().copied().min().unwrap_or(0),
                    min_local_rows: value.minimum_local_rows,
                    min_effective_sample_size: value.minimum_effective_sample_size,
                    max_normalized_weight: value.maximum_normalized_weight,
                    min_dose_density: value.minimum_dose_density,
                    interval_95_published: intervals.iter().all(Option::is_some),
                    reported_intervals: intervals.iter().filter(|interval| interval.is_some()).count(),
                    all_reported_intervals: intervals.iter().all(Option::is_some),
                    known_density: self.query.density_provenance.as_ref() == "known",
                    ..Default::default()
                },
            ).is_some()
        });
        let mut result = finish_identified_execute_with_context(
            &self.result_context,
            Some(data),
            IdentifiedExecuteFinish {
                physical: &self.physical,
                identification: self.identification.clone(),
                estimand: self.estimand.clone(),
                estimate: EffectEstimate::new(
                    0.0,
                    f64::NAN,
                    self.identification.required_assumptions.clone(),
                    OverlapPolicy::ExplicitOverride,
                ),
                identifier_id: IdentifierId::ContinuousDoseExchangeability,
                estimator_id: EstimatorId::ContinuousDoseKernel,
                treatment: self.query.dose,
                outcome: self.query.outcome,
                identify_cached: true,
                extra_diagnostics: vec![Diagnostic::new(
                    "estimate.policy.continuous_dose_support",
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Info,
                    if fixed_policy.is_some() {
                        "fixed group-to-dose policy and reference estimate kernel-smoothed intervention values with paired independent-row variance; the result does not identify an exact-dose intervention"
                    } else {
                        "all baseline-group target cells met local support; density provenance is caller-declared; local SD is descriptive; no interval or continuous-dose policy value"
                    },
                )],
                refutations: Vec::new(),
                distribution: None,
                mediation: None,
                wall_time_ns: 0,
                bootstrap_replicates_ok: None,
                cancelled: false,
                early_stopped: false,
                extras: IdentifiedExecuteExtras::default(),
            },
        );
        result.estimate = crate::PrimaryEstimate::NotAnEffect;
        result.continuous_dose_response = Some(crate::ContinuousDoseResponseEstimate {
            points: points.into(),
            bandwidth: self.query.bandwidth,
            density_provenance: self.query.density_provenance.clone(),
            uncertainty: Arc::from(match &fixed_policy {
                Some(value) if value.incremental_interval_95.is_some() =>
                    "fixed_group_kernel_smoothed_paired_pointwise_95_normal_intervals",
                Some(_) => "fixed_group_kernel_smoothed_paired_variance_no_interval",
                None => "point_only_no_interval",
            }),
            fixed_policy,
        });
        if licensed_fixed_policy {
            result.support_status = Some(crate::support::CellStatus::Licensed);
        }
        result.treatment = None;
        Ok(result)
    }
}

impl Study {
    pub(super) fn execute_continuous_dose_response(
        &self,
        data: &TabularData,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        CheckedContinuousDoseOperation::checked(self, data, physical)?.execute(data, ctx)
    }
}

pub(crate) fn continuous_dose_identification(
    query: &antecedent_core::ContinuousDoseResponseQuery,
) -> (IdentificationResult, IdentifiedEstimand) {
    let mut assumptions = antecedent_core::AssumptionSet::default();
    for (id, description) in [
        (
            "conditional_dose_exchangeability",
            "potential outcomes are exchangeable across dose conditional on the supplied pre-treatment baseline groups",
        ),
        ("correct_observed_dose_density", "supplied densities are correct at observed doses"),
        (
            "continuous_dose_positivity",
            "every requested target has positive density and adequate local support in every baseline group",
        ),
        (
            "consistency_no_interference",
            "observed outcomes equal potential outcomes at realized dose and subjects do not interfere",
        ),
    ] {
        assumptions.push(AssumptionRecord {
            assumption: Assumption::Custom {
                id: Arc::from(id),
                description: Arc::from(description),
            },
            source: AssumptionSource::UserDeclared,
            scope: AssumptionScope::Identification,
            status: AssumptionStatus::Declared,
        });
    }
    if query.fixed_policy.is_some() {
        assumptions.push(AssumptionRecord {
            assumption: Assumption::Custom {
                id: Arc::from("fixed_group_dose_policy"),
                description: Arc::from("policy and reference doses were fixed for pre-treatment groups independently of these evaluation outcomes; the estimand averages each group's inverse-density kernel-smoothed intervention response, not its exact-dose potential outcome"),
            },
            source: AssumptionSource::UserDeclared,
            scope: AssumptionScope::Identification,
            status: AssumptionStatus::Declared,
        });
    }
    let mut arena = CausalExprArena::new();
    let outcomes = arena.intern_var_set([query.outcome]);
    let empty = arena.empty_var_set();
    let intervention = arena.intern_intervention_set([]);
    let distribution = arena.intern_distribution(
        outcomes,
        empty,
        intervention,
        antecedent_expr::DomainRef::Observational,
    );
    let functional = arena.intern(antecedent_expr::ExprNode::Expectation {
        function: antecedent_expr::OutcomeExprId::identity(query.outcome),
        distribution,
    });
    let rule = "policy.conditional_dose_response";
    arena.set_derivation(
        functional,
        antecedent_expr::DerivationMeta::rule(
            rule,
            Some(Arc::from(
                "baseline-group exchangeability and caller-supplied density bind each target dose",
            )),
        ),
    );
    let estimand = IdentifiedEstimand::new(
        rule,
        Arc::from([]),
        Arc::from([]),
        Arc::from([]),
        functional,
        None,
    );
    let mut derivation = DerivationTrace::default();
    derivation.push(
        rule,
        "group-conditional local inverse-density response at each prespecified target dose",
    );
    let identification = IdentificationResult::identified(
        CausalQuery::ContinuousDoseResponse(query.clone()),
        vec![estimand.clone()],
        arena,
        derivation,
        assumptions,
        IdentificationPerformanceRecord::default(),
    );
    (identification, estimand)
}
