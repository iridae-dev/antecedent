//! Prepared native execution for randomized two-arm ITT.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::randomized_fit::{
    RandomizedMethod, assignment_probability_floor, exact_randomization_test, fit_randomized,
};
use super::{
    Arc, CausalError, CausalExprArena, CausalQuery, DerivationTrace, Diagnostic, DiagnosticKind,
    DiagnosticSeverity, EffectEstimate, EstimatorId, ExecutionContext, GraphClass,
    IdentificationPerformanceRecord, IdentificationResult, IdentifiedEstimand,
    IdentifiedExecuteExtras, IdentifiedExecuteFinish, IdentifierId, InferenceMode, Instant,
    OverlapPolicy, PhysicalExecutionPlan, RandomizedTrialOperation, RandomizedTrialQuery,
    RefuteSuite, Study, StudyResult, TableView, TabularData,
    finish_identified_execute_with_context,
};
use antecedent_core::{
    Assumption, AssumptionRecord, AssumptionScope, AssumptionSource, AssumptionStatus,
};

/// Frozen randomized-design analysis. Assignment metadata is part of query identity.
pub(crate) type CheckedRandomizedOperation =
    RandomizedTrialOperation<antecedent_core::RandomizedEffectQuery>;

impl RandomizedTrialQuery for antecedent_core::RandomizedEffectQuery {
    const OPERATION: &'static str = "CheckedRandomizedOperation";
    const REFRESH_MISMATCH: &'static str =
        "randomized ITT refresh requires the prepared schema and row order";
}

impl CheckedRandomizedOperation {
    pub(crate) fn checked(
        study: &Study,
        data: &TabularData,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
    ) -> Result<Self, CausalError> {
        let CausalQuery::RandomizedEffect(query) = &study.query else {
            return Err(CausalError::Compile {
                message: "randomized execution requires RandomizedEffectQuery".into(),
            });
        };
        query.validate()?;
        if study.graph.class() != GraphClass::RandomizedTrial
            || study.structure_source != crate::support::StructureSource::RandomizedTrial
            || !matches!(study.inference, InferenceMode::Frequentist)
            || study.refute != RefuteSuite::None
            || study.bootstrap_replicates != 0
            || !study.custom_validators.is_empty()
            || study.graph_posterior.is_some()
            || study.tiered.is_some()
        {
            return Err(CausalError::Unsupported {
                message: "randomized ITT supports Bernoulli, complete, stratified, cluster, and switchback estimation only, without a graph, validation suite, or bootstrap",
            });
        }
        if query.realized_assignment.len() != data.row_count() {
            return Err(CausalError::Compile {
                message: "randomized assignment metadata must align with the table rows".into(),
            });
        }
        data.schema().get(query.outcome).map_err(CausalError::rejected)?;
        if let Some((covariate, _)) = query.fixed_cuped {
            data.schema().get(covariate).map_err(CausalError::rejected)?;
        }
        for &covariate in query.ancova_covariates.iter() {
            data.schema().get(covariate).map_err(CausalError::rejected)?;
        }
        if physical.logical.query != study.query
            || !physical
                .binding()
                .names(IdentifierId::RandomizedDesign, RandomizedMethod::of(query).estimator_id())
        {
            return Err(CausalError::Compile {
                message: "randomized target or design differs from its compiled plan".into(),
            });
        }
        let (identification, estimand) = randomized_identification(query);
        Ok(Self::freeze(study, query, data, physical, identification, estimand, ()))
    }

    pub(crate) fn execute(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        self.begin_execute(data, ctx)?;
        self.query.validate()?;
        let outcomes = match data.column(self.query.outcome).map_err(CausalError::from)? {
            antecedent_data::ColumnView::Float64(column) => column.values.as_slice(),
            _ => {
                return Err(CausalError::Unsupported {
                    message: "randomized ITT outcome must be continuous",
                });
            }
        };
        if outcomes.iter().any(|y| !y.is_finite()) {
            return Err(CausalError::Unsupported {
                message: "randomized ITT requires complete finite outcomes",
            });
        }
        let adjusted_outcomes = self.cuped_adjusted_outcomes(data, outcomes)?;
        let outcomes = adjusted_outcomes.as_deref().unwrap_or(outcomes);
        let method = RandomizedMethod::of(&self.query);
        let fit = fit_randomized(&self.query, data, outcomes)?;
        if !fit.effect.is_finite() || !fit.variance.is_finite() || fit.variance < 0.0 {
            return Err(CausalError::Unsupported {
                message: "randomized effect or design variance is not finite under the declared probabilities",
            });
        }
        let interval_published = fit.interval_95.is_some();
        let uncertainty = Arc::<str>::from(method.uncertainty(interval_published));
        let diagnostic = method.diagnostic(interval_published);
        let exact_test = self
            .query
            .exact_randomization_test
            .then(|| exact_randomization_test(&self.query, outcomes, fit.effect));
        let mut estimate = EffectEstimate::new(
            fit.effect,
            fit.interval_standard_error.unwrap_or(f64::NAN),
            self.identification.required_assumptions.clone(),
            OverlapPolicy::ExplicitOverride,
        );
        if interval_published && method.clustered_standard_error() {
            estimate = estimate.with_se_kind(antecedent_estimate::AnalyticSeKind::Cluster);
        }
        let started = Instant::now();
        let mut result = finish_identified_execute_with_context(
            &self.result_context,
            Some(data),
            IdentifiedExecuteFinish {
                physical: &self.physical,
                identification: self.identification.clone(),
                estimand: self.estimand.clone(),
                estimate,
                identifier_id: IdentifierId::RandomizedDesign,
                estimator_id: method.estimator_id(),
                treatment: self.query.outcome,
                outcome: self.query.outcome,
                identify_cached: true,
                extra_diagnostics: vec![Diagnostic::new(
                    method.diagnostic_id(),
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Info,
                    diagnostic,
                )],
                refutations: Vec::new(),
                distribution: None,
                mediation: None,
                wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
                bootstrap_replicates_ok: None,
                cancelled: false,
                early_stopped: false,
                extras: IdentifiedExecuteExtras::default(),
            },
        );
        result.payload = crate::result::ResultPayload::RandomizedEffect(self.randomized_payload(
            fit,
            method,
            exact_test,
            uncertainty,
        ));
        if self.graphless_license(&result, method).is_some() {
            result.support_status = Some(crate::support::CellStatus::Licensed);
        }
        result.rebind_interval(false);
        result.treatment = None;
        Ok(result)
    }

    /// Outcomes minus the fixed CUPED covariate term, when the query declares one.
    fn cuped_adjusted_outcomes(
        &self,
        data: &TabularData,
        outcomes: &[f64],
    ) -> Result<Option<Vec<f64>>, CausalError> {
        let Some((covariate, coefficient)) = self.query.fixed_cuped else {
            return Ok(None);
        };
        let n = outcomes.len();
        let values = match data.column(covariate).map_err(CausalError::from)? {
            antecedent_data::ColumnView::Float64(column) => column.values.as_slice(),
            _ => {
                return Err(CausalError::Unsupported {
                    message: "fixed CUPED pre-assignment covariate must be continuous",
                });
            }
        };
        if values.len() != n || values.iter().any(|x| !x.is_finite()) {
            return Err(CausalError::Unsupported {
                message: "fixed CUPED requires a complete finite pre-assignment covariate",
            });
        }
        Ok(Some(outcomes.iter().zip(values.iter()).map(|(y, x)| y - coefficient * x).collect()))
    }

    /// Wire payload of a fitted randomized effect.
    fn randomized_payload(
        &self,
        mut fit: super::randomized_fit::TrialFit,
        method: RandomizedMethod,
        exact_test: Option<(f64, u64)>,
        uncertainty: Arc<str>,
    ) -> crate::RandomizedEffectEstimate {
        let (control_units, treatment_units) = (fit.control_units, fit.treatment_units);
        crate::RandomizedEffectEstimate {
            effect: fit.effect,
            estimand: Arc::from(method.estimand_label()),
            intention_to_treat_effect: fit.complier_components.map(|(itt, _)| itt),
            first_stage_effect: fit.complier_components.map(|(_, stage)| stage),
            received_treatment: self.query.received_treatment.clone(),
            randomization_p_value: exact_test.map(|(p, _)| p),
            randomization_allocations: exact_test.map(|(_, count)| count),
            second_factor_effect: fit.factorial.map(|f| f.second_factor_effect),
            factorial_interaction: fit.factorial.map(|f| f.interaction),
            second_factor_variance: fit.factorial.map(|f| f.second_factor_variance),
            factorial_interaction_variance: fit.factorial.map(|f| f.interaction_variance),
            multi_arm_values: std::mem::replace(&mut fit.multi_arm_values, Arc::from([])),
            variance_upper_bound: fit.variance,
            standard_error: fit.interval_standard_error,
            interval_95: fit.interval_95,
            second_factor_interval_95: fit.second_factor_interval_95,
            factorial_interaction_interval_95: fit.factorial_interaction_interval_95,
            multi_arm_intervals_95: std::mem::replace(
                &mut fit.multi_arm_intervals_95,
                Arc::from([]),
            ),
            minimum_assignment_probability: assignment_probability_floor(
                &self.query,
                matches!(method, RandomizedMethod::Switchback),
            ),
            assignment_design: Arc::from(method.design_tag()),
            blocks: fit.blocks,
            periods: fit.periods,
            control_units,
            treatment_units,
            uncertainty,
            assignment_units: Arc::clone(&self.query.assignment_units),
            outcome_units: Arc::clone(&self.query.outcome_units),
            treatment_arms: self.query.treatment_arms.clone(),
        }
    }

    /// Graphless licenses are exact design/method/inference claims. A
    /// published interval alone is insufficient: sparse or other designs
    /// must not inherit a geometric matrix coordinate by analogy.
    fn graphless_license(
        &self,
        result: &StudyResult,
        method: RandomizedMethod,
    ) -> Option<crate::support::CellStatus> {
        let fit = result.randomized_effect().expect("randomized result was just set");
        let (family, design, method_id, assignment_unit, claim) =
            antecedent_core::randomized_graphless_coordinate(
                &self.query.design,
                self.query.estimand,
                self.query.received_treatment.is_some(),
                self.query.fixed_cuped.is_some(),
                !self.query.ancova_covariates.is_empty(),
            );
        let switchback = method == RandomizedMethod::Switchback;
        let mut block_counts = std::collections::BTreeMap::<&str, (usize, usize)>::new();
        let support_blocks = if switchback { &self.query.assignment_units } else { &fit.blocks };
        for (block, assigned) in support_blocks.iter().zip(self.query.realized_assignment.iter()) {
            let counts = block_counts.entry(block).or_default();
            if *assigned {
                counts.1 += 1;
            } else {
                counts.0 += 1;
            }
        }
        let min_block_arm = block_counts
            .values()
            .flat_map(|(control, treated)| [*control, *treated])
            .min()
            .unwrap_or(0);
        let balanced_sequences = switchback
            && block_counts.values().next().is_some_and(|first| {
                let size = first.0 + first.1;
                size > 0 && block_counts.values().all(|counts| counts.0 + counts.1 == size)
            });
        let min_factorial_cell = if let antecedent_core::RandomizationDesign::Factorial2x2 {
            second_factor_assignment,
            ..
        } = &self.query.design
        {
            let mut counts = [0_usize; 4];
            for (primary, secondary) in
                self.query.realized_assignment.iter().zip(second_factor_assignment.iter())
            {
                counts[usize::from(*primary) + 2 * usize::from(*secondary)] += 1;
            }
            *counts.iter().min().unwrap_or(&0)
        } else {
            0
        };
        let multi_arm = method == RandomizedMethod::MultiArm;
        let factorial = method == RandomizedMethod::Factorial2x2;
        let reported_intervals = if multi_arm {
            fit.multi_arm_intervals_95.iter().filter(|interval| interval.is_some()).count()
        } else {
            usize::from(fit.interval_95.is_some())
                + usize::from(fit.second_factor_interval_95.is_some())
                + usize::from(fit.factorial_interaction_interval_95.is_some())
        };
        let all_reported_intervals = if factorial {
            reported_intervals == 3
        } else if multi_arm {
            fit.multi_arm_intervals_95.len() == fit.multi_arm_values.len()
                && reported_intervals + 1 == fit.multi_arm_values.len()
        } else {
            fit.interval_95.is_some()
        };
        crate::support::license_if_graphless(
            crate::support::GraphlessSupportKey {
                family,
                design,
                method: method_id,
                inference_claim: claim,
            },
            crate::support::GraphlessAssignmentSupport {
                assignment_unit,
                treated: fit.treatment_units,
                control: fit.control_units,
                interval_95_published: fit.interval_95.is_some(),
                rows: self.rows,
                blocks: block_counts.len(),
                min_block_arm,
                min_factorial_cell,
                min_action_rows: if fit.multi_arm_values.is_empty() {
                    fit.control_units.min(fit.treatment_units)
                } else {
                    fit.multi_arm_values.iter().map(|(_, _, _, count)| *count).min().unwrap_or(0)
                },
                min_probability: assignment_probability_floor(&self.query, true),
                reported_intervals,
                all_reported_intervals,
                covariates: self.query.ancova_covariates.len(),
                balanced_sequences,
                ..Default::default()
            },
        )
    }
}

impl Study {
    pub(super) fn execute_policy_value(
        &self,
        data: &TabularData,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        CheckedPolicyValueOperation::checked(self, data, physical)?.execute(data, ctx)
    }

    pub(super) fn execute_randomized(
        &self,
        data: &TabularData,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        CheckedRandomizedOperation::checked(self, data, physical)?.execute(data, ctx)
    }
}

/// Frozen, held-out policy scoring operation. Prediction vectors and ownership
/// declarations are part of the retained query identity.
pub(crate) type CheckedPolicyValueOperation =
    RandomizedTrialOperation<antecedent_core::PolicyValueQuery>;

impl RandomizedTrialQuery for antecedent_core::PolicyValueQuery {
    const OPERATION: &'static str = "CheckedPolicyValueOperation";
    const REFRESH_MISMATCH: &'static str =
        "policy value refresh requires the prepared schema and row order";
}

impl CheckedPolicyValueOperation {
    pub(crate) fn checked(
        study: &Study,
        data: &TabularData,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
    ) -> Result<Self, CausalError> {
        let CausalQuery::PolicyValue(query) = &study.query else {
            return Err(CausalError::Compile {
                message: "policy-value execution requires PolicyValueQuery".into(),
            });
        };
        query.validate()?;
        if query
            .multi_action
            .as_ref()
            .map_or(query.assignment.len(), |multi| multi.assignment.len())
            != data.row_count()
            || !matches!(study.inference, InferenceMode::Frequentist)
            || study.refute != RefuteSuite::None
            || study.bootstrap_replicates != 0
            || !study.custom_validators.is_empty()
            || study.graph.class() != GraphClass::RandomizedTrial
        {
            return Err(CausalError::Unsupported {
                message: "policy value requires aligned graphless randomized rows and does not accept refuters, bootstrap, or Bayesian inference",
            });
        }
        data.schema().get(query.outcome).map_err(CausalError::rejected)?;
        let n = data.row_count();
        let (binary_assignment, binary_propensity) = if let Some(multi) = &query.multi_action {
            let k = multi.action_labels.len();
            (
                multi.assignment.iter().map(|&a| a != 0).collect::<Vec<_>>(),
                (0..n).map(|i| 1.0 - multi.propensities[i * k]).collect::<Vec<_>>(),
            )
        } else {
            (
                query.assignment.to_vec(),
                (0..n)
                    .map(|i| query.propensity[if query.propensity.len() == 1 { 0 } else { i }])
                    .collect::<Vec<_>>(),
            )
        };
        let synthetic = antecedent_core::RandomizedEffectQuery::bernoulli_itt(
            query.outcome,
            binary_assignment,
            binary_propensity,
            query.evaluation_subject_ids.clone(),
            query.evaluation_subject_ids.clone(),
            ("control", "treated"),
        );
        let (mut identification, mut estimand) = randomized_identification(&synthetic);
        identification.query = CausalQuery::PolicyValue(query.clone());
        estimand.method = Arc::from(antecedent_expr::EstimandMethod::RandomizedPolicyValue);
        identification.estimands = vec![estimand.clone()];
        identification.arena.set_derivation(
            estimand.functional,
            antecedent_expr::DerivationMeta::rule(
                "randomized.policy_value",
                Some(Arc::from(
                    "known randomized action probabilities identify the value of a fixed policy",
                )),
            ),
        );
        identification.derivation = DerivationTrace::default();
        identification.derivation.push(
            "randomized.policy_value",
            "known randomized action probabilities identify the value of fixed policy and reference recommendations",
        );
        identification.required_assumptions.push(AssumptionRecord {
            assumption: Assumption::Custom {
                id: Arc::from("policy_fixed_without_evaluation_outcomes"),
                description: Arc::from("policy and reference recommendations were fixed without using the corresponding evaluation outcomes"),
            },
            source: AssumptionSource::UserDeclared,
            scope: AssumptionScope::Identification,
            status: AssumptionStatus::Declared,
        });
        if query.multi_action.is_some() {
            identification.required_assumptions.push(AssumptionRecord {
                assumption: Assumption::Custom {
                    id: Arc::from("known_random_action_probabilities"),
                    description: Arc::from("each evaluation row followed the declared randomized multi-action probability vector"),
                },
                source: AssumptionSource::UserDeclared,
                scope: AssumptionScope::Identification,
                status: AssumptionStatus::Declared,
            });
            if query.multi_action.as_ref().is_some_and(|policy| !policy.cate_groups.is_empty()) {
                identification.required_assumptions.push(AssumptionRecord {
                    assumption: Assumption::Custom {
                        id: Arc::from("fixed_pre_outcome_cate_groups"),
                        description: Arc::from("baseline groups for multi-action conditional contrasts were fixed before observing evaluation outcomes"),
                    },
                    source: AssumptionSource::UserDeclared,
                    scope: AssumptionScope::Identification,
                    status: AssumptionStatus::Declared,
                });
            }
        }
        Ok(Self::freeze(study, query, data, physical, identification, estimand, ()))
    }

    pub(crate) fn execute(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        self.begin_execute(data, ctx)?;
        let y = match data.column(self.query.outcome).map_err(CausalError::from)? {
            antecedent_data::ColumnView::Float64(values) => values.values.as_slice(),
            _ => {
                return Err(CausalError::Unsupported {
                    message: "policy value outcome must be continuous",
                });
            }
        };
        let evaluation = self.evaluate_policy_value(y)?;
        let counts = (
            evaluation.policy_matches,
            evaluation.reference_matches,
            evaluation.independent_policy_rows,
        );
        let mut result = self.policy_value_result(data, evaluation);
        self.license_policy_value(&mut result, y.len(), counts);
        Ok(result)
    }

    /// Scores, uplift bins, regret and intervals of the declared policy on the outcomes.
    fn evaluate_policy_value(&self, y: &[f64]) -> Result<PolicyValueEvaluation, CausalError> {
        let multi = self.query.multi_action.as_ref();
        let multi_action_cate = if let Some(policy) = multi {
            antecedent_estimate::policy_value::evaluate_multi_action_cate(y, policy)
                .map_err(|message| CausalError::Unsupported { message })?
        } else {
            Vec::new()
        };
        let regret = self
            .query
            .regret
            .as_ref()
            .map(|design| {
                antecedent_estimate::policy_value::evaluate_fixed_candidate_regret(
                    y,
                    &self.query.assignment,
                    &self.query.propensity,
                    &design
                        .candidates
                        .iter()
                        .map(|candidate| candidate.to_vec())
                        .collect::<Vec<_>>(),
                    design.selected_index,
                    &self.query.costs,
                    !design.training_subject_ids.is_empty(),
                )
                .map_err(|message| CausalError::Unsupported { message })
            })
            .transpose()?;
        let ipw = self.query.mu0.is_empty();
        let uplift_bins = if self.query.uplift_bin_count > 0 {
            antecedent_estimate::policy_value::evaluate_uplift_bins(
                y,
                &self.query.assignment,
                &self.query.propensity,
                &self.query.uplift_bins,
                self.query.uplift_bin_count,
            )
            .map_err(|message| CausalError::Unsupported { message })?
        } else {
            Vec::new()
        };
        let score = self.policy_value_score(y, multi, ipw)?;
        let (policy_matches, reference_matches) = if let Some(policy) = multi {
            (
                policy.assignment.iter().zip(policy.actions.iter()).filter(|(a, b)| a == b).count(),
                policy
                    .assignment
                    .iter()
                    .zip(policy.reference.iter())
                    .filter(|(a, b)| a == b)
                    .count(),
            )
        } else {
            (
                self.query
                    .assignment
                    .iter()
                    .zip(self.query.actions.iter())
                    .filter(|(a, b)| a == b)
                    .count(),
                self.query
                    .assignment
                    .iter()
                    .zip(self.query.reference.iter())
                    .filter(|(a, b)| a == b)
                    .count(),
            )
        };
        // Cross-fitted predictions on caller-declared excluded folds do not use
        // a row's own outcome, so the independent-row influence-function
        // variance is valid; IPW and disjoint held-out AIPW have fixed score
        // rules. Global capacity/budget coupling still withholds the interval.
        let independent_policy_rows = !self.query.global_constraints_present
            && multi.is_none_or(|policy| !policy.global_constraints_couple_rows());
        let minimum_calibrated_rows = if multi.is_some() || !ipw { 300 } else { 120 };
        let intervals = if y.len() >= minimum_calibrated_rows
            && independent_policy_rows
            && (ipw
                || self.query.disjoint_training_subjects
                || self.query.crossfit_fold_ownership_valid)
        {
            antecedent_estimate::policy_value::pointwise_intervals_95(
                &score,
                y.len(),
                policy_matches,
                reference_matches,
            )
        } else {
            None
        };
        Ok(PolicyValueEvaluation {
            multi_action_cate,
            regret,
            uplift_bins,
            score,
            policy_matches,
            reference_matches,
            independent_policy_rows,
            intervals,
        })
    }

    /// Row scores of the declared policy: multi-action IPW, binary IPW, or doubly robust.
    fn policy_value_score(
        &self,
        y: &[f64],
        multi: Option<&antecedent_core::MultiActionPolicyInputs>,
        ipw: bool,
    ) -> Result<antecedent_estimate::policy_value::PolicyValueScores, CausalError> {
        let score = if let Some(multi) = multi {
            antecedent_estimate::policy_value::evaluate_multi_action_policy_value_scores(y, multi)
        } else if ipw {
            antecedent_estimate::policy_value::evaluate_policy_value_ipw_scores(
                y,
                &antecedent_estimate::policy_value::BinaryPolicyDesign {
                    assignment: &self.query.assignment,
                    actions: &self.query.actions,
                    propensity: &self.query.propensity,
                    reference: &self.query.reference,
                    costs: &self.query.costs,
                    reference_costs: &self.query.reference_costs,
                },
            )
        } else {
            antecedent_estimate::policy_value::evaluate_policy_value_scores(
                y,
                &antecedent_estimate::policy_value::BinaryPolicyDesign {
                    assignment: &self.query.assignment,
                    actions: &self.query.actions,
                    propensity: &self.query.propensity,
                    reference: &self.query.reference,
                    costs: &self.query.costs,
                    reference_costs: &self.query.reference_costs,
                },
                &self.query.mu0,
                &self.query.mu1,
            )
        }
        .map_err(|message| CausalError::Unsupported { message })?;
        Ok(score)
    }

    /// The policy-value result: not-an-effect estimate with the policy payload attached.
    fn policy_value_result(
        &self,
        data: &TabularData,
        evaluation: PolicyValueEvaluation,
    ) -> StudyResult {
        let PolicyValueEvaluation {
            multi_action_cate, regret, uplift_bins, score, intervals, ..
        } = evaluation;
        let multi = self.query.multi_action.as_ref();
        let ipw = self.query.mu0.is_empty();
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
                identifier_id: IdentifierId::RandomizedDesign,
                estimator_id: if multi.is_some() {
                    EstimatorId::RandomizedMultiActionIpwPolicy
                } else if ipw {
                    EstimatorId::RandomizedIpwPolicy
                } else {
                    EstimatorId::RandomizedDrPolicy
                },
                treatment: self.query.outcome,
                outcome: self.query.outcome,
                identify_cached: true,
                extra_diagnostics: vec![Diagnostic::new(
                    if multi.is_some() {
                        "estimate.policy_value.multi_action_ipw"
                    } else if ipw {
                        "estimate.policy_value.ipw"
                    } else {
                        "estimate.policy_value.doubly_robust"
                    },
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Info,
                    if multi.is_some() {
                        "fixed randomized multi-action IPW policy value; row-score standard errors assume independent subjects"
                    } else if ipw {
                        "fixed randomized IPW policy value; row-score standard errors assume independent subjects"
                    } else {
                        "held-out randomized AIPW policy value; row-score standard errors assume independent subjects"
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
        result.treatment = None;
        result.payload = crate::result::ResultPayload::PolicyValue(crate::PolicyValueEstimate {
            policy_value: score.policy_value,
            reference_value: score.reference_value,
            incremental_value: score.incremental_value,
            relative_value_gap: score.relative_value_gap,
            treatment_rate: score.treatment_rate,
            total_cost: score.total_cost,
            policy_standard_error: score.policy_standard_error,
            reference_standard_error: score.reference_standard_error,
            incremental_standard_error: score.incremental_standard_error,
            policy_interval_95: intervals.map(|value| value.policy),
            incremental_interval_95: intervals.map(|value| value.incremental),
            prediction_ownership: Arc::from(if ipw {
                "no_outcome_nuisance_predictions"
            } else if self.query.disjoint_training_subjects {
                "declared_disjoint_training_subject_ids"
            } else {
                "caller_declared_cross_fitted_excluded_fold_ids"
            }),
            propensity_min: score.propensity_min,
            propensity_max: score.propensity_max,
            uncertainty: Arc::from(if multi.is_some() {
                "multi_action_ipw_row_score_standard_error_independent_subjects"
            } else if ipw {
                "ipw_row_score_standard_error_independent_subjects"
            } else {
                "row_score_standard_error_independent_subjects"
            }),
            uplift_bins,
            multi_action_cate,
            regret,
        });
        result
    }

    /// Treated and control counts, thinnest action arm and smallest assignment probability.
    fn policy_assignment_support(&self, rows: usize) -> (usize, usize, usize, f64) {
        if let Some(multi) = self.query.multi_action.as_ref() {
            let k = multi.action_labels.len();
            (
                multi.assignment.iter().filter(|&&action| action != 0).count(),
                multi.assignment.iter().filter(|&&action| action == 0).count(),
                (0..k)
                    .map(|action| {
                        multi.assignment.iter().filter(|&&observed| observed == action).count()
                    })
                    .min()
                    .unwrap_or(0),
                multi.propensities.iter().copied().fold(f64::INFINITY, f64::min),
            )
        } else {
            let treated = self.query.assignment.iter().filter(|&&assigned| assigned).count();
            (
                treated,
                rows - treated,
                treated.min(rows - treated),
                self.query
                    .propensity
                    .iter()
                    .copied()
                    .map(|p| p.min(1.0 - p))
                    .fold(f64::INFINITY, f64::min),
            )
        }
    }

    /// Licenses the graphless cell when the observed assignment support clears its gates.
    fn license_policy_value(
        &self,
        result: &mut StudyResult,
        rows: usize,
        counts: (usize, usize, bool),
    ) {
        let (policy_matches, reference_matches, independent_policy_rows) = counts;
        let multi = self.query.multi_action.as_ref();
        let ipw = self.query.mu0.is_empty();
        let policy = result.policy_value().expect("policy value was just attached");
        let (design, method, claim) = antecedent_core::policy_graphless_coordinate(
            multi.is_some(),
            ipw,
            !policy.uplift_bins.is_empty(),
            !policy.multi_action_cate.is_empty(),
            self.query.regret.is_some(),
            !ipw && self.query.crossfit_fold_ownership_valid
                && !self.query.disjoint_training_subjects,
        );
        let (treated, control, min_action_rows, min_probability) =
            self.policy_assignment_support(rows);
        let min_bin_rows =
            policy.uplift_bins.iter().map(|bin| bin.evaluation_rows).min().unwrap_or(0);
        let min_bin_arm_rows = if policy.uplift_bins.is_empty() {
            0
        } else {
            (0..policy.uplift_bins.len())
                .flat_map(|bin| {
                    let treated = self
                        .query
                        .uplift_bins
                        .iter()
                        .enumerate()
                        .filter(|&(row, &rank)| rank == bin && self.query.assignment[row])
                        .count();
                    let total = policy.uplift_bins[bin].evaluation_rows;
                    [treated, total - treated]
                })
                .min()
                .unwrap_or(0)
        };
        let min_group_rows =
            policy.multi_action_cate.iter().map(|point| point.evaluation_rows).min().unwrap_or(0);
        let min_group_arm_rows = policy
            .multi_action_cate
            .iter()
            .flat_map(|point| [point.observed_action_rows, point.observed_control_rows])
            .min()
            .unwrap_or(0);
        let scalar_intervals = usize::from(policy.policy_interval_95.is_some())
            + usize::from(policy.incremental_interval_95.is_some());
        let extras = policy.uplift_bins.iter().filter(|bin| bin.interval_95.is_some()).count()
            + policy.multi_action_cate.iter().filter(|point| point.interval_95.is_some()).count();
        let all_reported_intervals = scalar_intervals == 2
            && policy.uplift_bins.iter().all(|bin| bin.interval_95.is_some())
            && policy.multi_action_cate.iter().all(|point| point.interval_95.is_some());
        let observed = crate::support::GraphlessAssignmentSupport {
            assignment_unit: "unit",
            treated,
            control,
            rows,
            min_action_rows,
            min_probability,
            interval_95_published: scalar_intervals == 2,
            reported_intervals: scalar_intervals + extras,
            all_reported_intervals,
            policy_matches,
            reference_matches,
            min_bin_rows,
            min_bin_arm_rows,
            min_group_rows,
            min_group_arm_rows,
            uncoupled_constraints: independent_policy_rows,
            disjoint_nuisance_training: self.query.disjoint_training_subjects,
            rank_ownership: !self.query.uplift_training_subject_ids.is_empty()
                && self
                    .query
                    .uplift_training_subject_ids
                    .iter()
                    .all(|id| !self.query.evaluation_subject_ids.contains(id)),
            ..Default::default()
        };
        if crate::support::license_if_graphless(
            crate::support::GraphlessSupportKey {
                family: "policy_value",
                design,
                method,
                inference_claim: claim,
            },
            observed,
        )
        .is_some()
        {
            result.support_status = Some(crate::support::CellStatus::Licensed);
        }
    }
}

/// Policy scores and the support counts a policy-value result is assembled from.
struct PolicyValueEvaluation {
    multi_action_cate: Vec<antecedent_estimate::policy_value::MultiActionCatePoint>,
    regret: Option<antecedent_estimate::policy_value::FixedCandidateRegret>,
    uplift_bins: Vec<antecedent_estimate::policy_value::UpliftBinScore>,
    score: antecedent_estimate::policy_value::PolicyValueScores,
    policy_matches: usize,
    reference_matches: usize,
    independent_policy_rows: bool,
    intervals: Option<antecedent_estimate::policy_value::PolicyValueIntervals95>,
}

/// Consistency, positivity, interference and compliance assumptions of a randomized design.
fn push_randomized_core_assumptions(
    query: &antecedent_core::RandomizedEffectQuery,
    assumptions: &mut antecedent_core::AssumptionSet,
) {
    let mut standard = vec![Assumption::Consistency, Assumption::Positivity];
    if !matches!(
        query.design,
        antecedent_core::RandomizationDesign::Cluster { .. }
            | antecedent_core::RandomizationDesign::Switchback { .. }
    ) {
        standard.push(Assumption::NoInterference);
    }
    for assumption in standard {
        assumptions.push(AssumptionRecord {
            assumption,
            source: AssumptionSource::UserDeclared,
            scope: AssumptionScope::Identification,
            status: AssumptionStatus::Declared,
        });
    }
    if query.received_treatment.is_some() {
        for (id, description) in [
            (
                "exclusion_restriction",
                "encouragement affects the outcome only through treatment receipt",
            ),
            (
                "monotonicity_no_defiers",
                "encouragement does not reduce treatment receipt for any unit",
            ),
            (
                "nonzero_receipt_first_stage",
                "randomized encouragement changes the probability of treatment receipt",
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
        if query.estimand == antecedent_core::RandomizedEstimand::TreatmentOnTreated {
            assumptions.push(AssumptionRecord {
                assumption: Assumption::Custom {
                    id: Arc::from("one_sided_noncompliance"),
                    description: Arc::from("no control-assigned unit could receive treatment; the observed control-arm receipt check does not establish this counterfactual claim"),
                },
                source: AssumptionSource::UserDeclared,
                scope: AssumptionScope::Identification,
                status: AssumptionStatus::Declared,
            });
        }
    }
    if query.exact_randomization_test {
        assumptions.push(AssumptionRecord {
            assumption: Assumption::Custom {
                id: Arc::from("fisher_sharp_null_two_sided"),
                description: Arc::from("the reported exact p-value tests the sharp null of no unit-level assignment effect using the absolute difference in means over every fixed-count assignment"),
            },
            source: AssumptionSource::UserDeclared,
            scope: AssumptionScope::Estimation,
            status: AssumptionStatus::Declared,
        });
    }
}

/// Cluster, switchback and known-assignment assumptions of the declared randomization design.
fn push_randomized_design_assumptions(
    query: &antecedent_core::RandomizedEffectQuery,
    assumptions: &mut antecedent_core::AssumptionSet,
) {
    if matches!(query.design, antecedent_core::RandomizationDesign::Cluster { .. }) {
        assumptions.push(AssumptionRecord {
            assumption: Assumption::Custom {
                id: Arc::from("no_between_cluster_interference"),
                description: Arc::from("one cluster's assignment does not affect outcomes in another cluster; effects within a cluster are included in the cluster-assignment ITT"),
            },
            source: AssumptionSource::UserDeclared,
            scope: AssumptionScope::Identification,
            status: AssumptionStatus::Declared,
        });
    }
    if matches!(query.design, antecedent_core::RandomizationDesign::Switchback { .. }) {
        assumptions.push(AssumptionRecord {
            assumption: Assumption::Custom {
                id: Arc::from("switchback_no_carryover"),
                description: Arc::from("a unit-period outcome depends on its current assignment, not assignments in earlier periods"),
            },
            source: AssumptionSource::UserDeclared,
            scope: AssumptionScope::Identification,
            status: AssumptionStatus::Declared,
        });
        assumptions.push(AssumptionRecord {
            assumption: Assumption::Custom {
                id: Arc::from("independent_sequences_no_between_sequence_interference"),
                description: Arc::from("switching sequences are independent and treatment in one sequence does not affect another sequence's outcomes"),
            },
            source: AssumptionSource::UserDeclared,
            scope: AssumptionScope::Identification,
            status: AssumptionStatus::Declared,
        });
    }
    assumptions.push(AssumptionRecord {
        assumption: Assumption::Custom {
            id: Arc::from("known_random_assignment"),
            description: Arc::from(match &query.design {
                antecedent_core::RandomizationDesign::Bernoulli =>
                    "Treatment assignment follows the declared independent Bernoulli probabilities",
                antecedent_core::RandomizationDesign::Complete { .. } =>
                    "Treatment assignment follows complete randomization with the declared fixed treatment count",
                antecedent_core::RandomizationDesign::Stratified { .. } =>
                    "Treatment assignment follows independent complete randomization within each declared block",
                antecedent_core::RandomizationDesign::Factorial2x2 { .. } =>
                    "The two factors are jointly completely randomized to the declared four fixed cell counts",
                antecedent_core::RandomizationDesign::MultiArm { .. } =>
                    "Each unit is independently assigned to one of the declared actions with its known probability vector",
                antecedent_core::RandomizationDesign::Cluster { .. } =>
                    "Treatment assignment follows complete randomization of independent clusters; arbitrary dependence is allowed within clusters",
                antecedent_core::RandomizationDesign::Switchback { .. } =>
                    "Unit-period treatment has the declared known marginal assignment probabilities within independent switching sequences",
            }),
        },
        source: AssumptionSource::UserDeclared,
        scope: AssumptionScope::Identification,
        status: AssumptionStatus::Declared,
    });
}

/// Fixed CUPED and ANCOVA pre-assignment covariate assumptions.
fn push_randomized_covariate_assumptions(
    query: &antecedent_core::RandomizedEffectQuery,
    assumptions: &mut antecedent_core::AssumptionSet,
) {
    if query.fixed_cuped.is_some() {
        assumptions.push(AssumptionRecord {
            assumption: Assumption::Custom {
                id: Arc::from("fixed_pre_assignment_cuped"),
                description: Arc::from("the CUPED covariate was measured before randomization and its coefficient was fixed independently of these outcomes"),
            },
            source: AssumptionSource::UserDeclared,
            scope: AssumptionScope::Identification,
            status: AssumptionStatus::Declared,
        });
    }
    if !query.ancova_covariates.is_empty() {
        assumptions.push(AssumptionRecord {
            assumption: Assumption::Custom {
                id: Arc::from("ancova_pre_assignment_covariates"),
                description: Arc::from("ANCOVA covariates were measured before randomization; its HC0 variance treats independently assigned rows as the sampling units"),
            },
            source: AssumptionSource::UserDeclared,
            scope: AssumptionScope::Estimation,
            status: AssumptionStatus::Declared,
        });
    }
}

pub(crate) fn randomized_identification(
    query: &antecedent_core::RandomizedEffectQuery,
) -> (IdentificationResult, IdentifiedEstimand) {
    let mut assumptions = antecedent_core::AssumptionSet::default();
    push_randomized_core_assumptions(query, &mut assumptions);
    push_randomized_design_assumptions(query, &mut assumptions);
    push_randomized_covariate_assumptions(query, &mut assumptions);
    let mut arena = CausalExprArena::new();
    let outcomes = arena.intern_var_set([query.outcome]);
    let empty = arena.empty_var_set();
    let no_intervention = arena.intern_intervention_set([]);
    let distribution = arena.intern_distribution(
        outcomes,
        empty,
        no_intervention,
        antecedent_expr::DomainRef::Observational,
    );
    let functional = arena.intern(antecedent_expr::ExprNode::Expectation {
        function: antecedent_expr::OutcomeExprId::identity(query.outcome),
        distribution,
    });
    let (identification_rule, identification_note) = if matches!(
        query.design,
        antecedent_core::RandomizationDesign::MultiArm { .. }
    ) {
        (
            "randomized.multi_arm_itt",
            "known randomized action probabilities identify all arm means and their assignment contrasts",
        )
    } else if query.estimand == antecedent_core::RandomizedEstimand::TreatmentOnTreated {
        (
            "randomized.one_sided_treatment_on_treated",
            "one-sided noncompliance identifies the effect among recipients through the Wald ratio under exclusion and random encouragement",
        )
    } else if query.received_treatment.is_some() {
        (
            "randomized.wald_cace_late",
            "randomized encouragement identifies the Wald complier contrast under exclusion and monotonicity",
        )
    } else {
        ("randomized.itt", "randomized assignment identifies the intention-to-treat contrast")
    };
    arena.set_derivation(
        functional,
        antecedent_expr::DerivationMeta::rule(
            identification_rule,
            Some(Arc::from(identification_note)),
        ),
    );
    let estimand = IdentifiedEstimand::new(
        identification_rule,
        Arc::from([]),
        Arc::from([]),
        Arc::from([]),
        functional,
        None,
    );
    let mut trace = DerivationTrace::default();
    trace.push(identification_rule, identification_note);
    let result = IdentificationResult::identified(
        CausalQuery::RandomizedEffect(query.clone()),
        vec![estimand.clone()],
        arena,
        trace,
        assumptions,
        IdentificationPerformanceRecord::default(),
    );
    (result, estimand)
}
