//! Prepared native execution for randomized two-arm ITT.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use antecedent_core::{
    Assumption, AssumptionRecord, AssumptionScope, AssumptionSource, AssumptionStatus,
};

/// Frozen randomized-design analysis. Assignment metadata is part of query identity.
#[derive(Clone)]
pub(crate) struct CheckedRandomizedOperation {
    query: antecedent_core::RandomizedEffectQuery,
    source_schema: antecedent_core::CausalSchema,
    source_rows: usize,
    identification: IdentificationResult,
    estimand: IdentifiedEstimand,
    physical: PhysicalExecutionPlan,
    result_context: IdentifiedResultContext,
}

impl std::fmt::Debug for CheckedRandomizedOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CheckedRandomizedOperation")
            .field("query", &self.query)
            .field("source_rows", &self.source_rows)
            .finish_non_exhaustive()
    }
}

impl CheckedRandomizedOperation {
    pub(crate) fn checked(
        study: &Study,
        data: &TabularData,
        physical: &PhysicalExecutionPlan,
    ) -> Result<Self, CausalError> {
        let CausalQuery::RandomizedEffect(query) = &study.query else {
            return Err(CausalError::Compile {
                message: "randomized execution requires RandomizedEffectQuery".into(),
            });
        };
        query.validate().map_err(|e| CausalError::Compile { message: e.to_string() })?;
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
                message: "randomized ITT supports Bernoulli, complete, stratified, and cluster design-based estimation only, without a graph, validation suite, or bootstrap",
            });
        }
        if query.realized_assignment.len() != data.row_count() {
            return Err(CausalError::Compile {
                message: "randomized assignment metadata must align with the table rows".into(),
            });
        }
        data.schema()
            .get(query.outcome)
            .map_err(|e| CausalError::Compile { message: e.to_string() })?;
        if let Some((covariate, _)) = query.fixed_cuped {
            data.schema().get(covariate)
                .map_err(|e| CausalError::Compile { message: e.to_string() })?;
        }
        if physical.logical.query != study.query
            || physical.logical.record.identifier.as_deref()
                != Some(IdentifierId::RandomizedDesign.as_str())
            || physical.logical.record.estimator.as_deref()
                != Some(randomized_estimator_id(query).as_str())
        {
            return Err(CausalError::Compile {
                message: "randomized target or design differs from its compiled plan".into(),
            });
        }
        let (identification, estimand) = randomized_identification(query);
        Ok(Self {
            query: query.clone(),
            source_schema: data.schema().clone(),
            source_rows: data.row_count(),
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
        if data.schema() != &self.source_schema || data.row_count() != self.source_rows {
            return Err(CausalError::Unsupported {
                message: "randomized ITT refresh requires the prepared schema and row order",
            });
        }
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled {
                stage: super::super::stage::STAGE_ESTIMATE_POINT,
            });
        }
        self.query.validate().map_err(|e| CausalError::Compile { message: e.to_string() })?;
        let outcomes = match data.column(self.query.outcome).map_err(CausalError::from)? {
            antecedent_data::ColumnView::Float64(column) => column.values.as_slice(),
            _ => {
                return Err(CausalError::Unsupported {
                    message: "randomized ITT outcome must be continuous",
                });
            }
        };
        let n = outcomes.len();
        if outcomes.iter().any(|y| !y.is_finite()) {
            return Err(CausalError::Unsupported {
                message: "randomized ITT requires complete finite outcomes",
            });
        }
        let adjusted_outcomes;
        let outcomes = if let Some((covariate, coefficient)) = self.query.fixed_cuped {
            let values = match data.column(covariate).map_err(CausalError::from)? {
                antecedent_data::ColumnView::Float64(column) => column.values.as_slice(),
                _ => return Err(CausalError::Unsupported {
                    message: "fixed CUPED pre-assignment covariate must be continuous",
                }),
            };
            if values.len() != n || values.iter().any(|x| !x.is_finite()) {
                return Err(CausalError::Unsupported {
                    message: "fixed CUPED requires a complete finite pre-assignment covariate",
                });
            }
            adjusted_outcomes = outcomes.iter().zip(values.iter())
                .map(|(y, x)| y - coefficient * x).collect::<Vec<_>>();
            adjusted_outcomes.as_slice()
        } else {
            outcomes
        };
        let sample_variance = |values: &[f64]| -> f64 {
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            values.iter().map(|value| (value - mean).powi(2)).sum::<f64>()
                / (values.len() - 1) as f64
        };
        let (
            effect,
            variance,
            control_units,
            treatment_units,
            blocks,
            assignment_design,
            uncertainty,
            diagnostic,
        ) = match &self.query.design {
            antecedent_core::RandomizationDesign::Bernoulli => {
                let mut effect_sum = 0.0;
                let mut variance_sum = 0.0;
                let mut control_units = 0;
                let mut treatment_units = 0;
                // With a fixed pre-assignment adjustment, E[observed y² / assignment²]
                // equals Y(1)²/p + Y(0)²/(1-p). This exceeds the Bernoulli
                // variance by (Y(1)-Y(0))² for each unit. No interval follows
                // from the random bound estimate without calibration.
                for ((&y, &treated), &p) in outcomes
                    .iter()
                    .zip(self.query.realized_assignment.iter())
                    .zip(self.query.assignment_probabilities.iter())
                {
                    if treated {
                        treatment_units += 1;
                        effect_sum += y / p;
                        variance_sum += if self.query.fixed_cuped.is_some() {
                            y * y / (p * p)
                        } else {
                            y * y * (1.0 - p) / (p * p)
                        };
                    } else {
                        control_units += 1;
                        effect_sum -= y / (1.0 - p);
                        variance_sum += if self.query.fixed_cuped.is_some() {
                            y * y / ((1.0 - p) * (1.0 - p))
                        } else {
                            y * y * p / ((1.0 - p) * (1.0 - p))
                        };
                    }
                }
                (
                    effect_sum / n as f64,
                    variance_sum / (n * n) as f64,
                    control_units,
                    treatment_units,
                    Arc::from([]),
                    Arc::<str>::from("bernoulli"),
                    Arc::<str>::from(if self.query.fixed_cuped.is_some() {
                        "bernoulli_fixed_cuped_ht_conservative_variance_no_interval"
                    } else {
                        "bernoulli_ht_design_variance_no_interval"
                    }),
                    if self.query.fixed_cuped.is_some() {
                        "Horvitz-Thompson ITT on outcomes adjusted by a declared pre-assignment covariate and externally fixed coefficient; observed-arm upper-bound estimator is conservative in randomization expectation, no interval"
                    } else {
                        "Horvitz-Thompson ITT with Bernoulli design variance; no confidence interval is reported"
                    },
                )
            }
            antecedent_core::RandomizationDesign::Complete { treated_units } => {
                let treated: Vec<f64> = outcomes
                    .iter()
                    .zip(self.query.realized_assignment.iter())
                    .filter_map(|(&y, &z)| z.then_some(y))
                    .collect();
                let control: Vec<f64> = outcomes
                    .iter()
                    .zip(self.query.realized_assignment.iter())
                    .filter_map(|(&y, &z)| (!z).then_some(y))
                    .collect();
                let effect = treated.iter().sum::<f64>() / treated.len() as f64
                    - control.iter().sum::<f64>() / control.len() as f64;
                let variance = sample_variance(&treated) / treated.len() as f64
                    + sample_variance(&control) / control.len() as f64;
                (
                    effect,
                    variance,
                    n - treated_units,
                    *treated_units,
                    Arc::from([]),
                    Arc::<str>::from("complete"),
                    Arc::<str>::from("complete_neyman_variance_upper_bound_no_interval"),
                    "Complete-randomization difference in means with Neyman conservative variance estimate; no confidence interval is reported",
                )
            }
            antecedent_core::RandomizationDesign::Cluster { treated_clusters } => {
                let mut cluster_totals: std::collections::BTreeMap<&str, (bool, f64)> =
                    std::collections::BTreeMap::new();
                for (i, outcome) in outcomes.iter().enumerate() {
                    let entry = cluster_totals
                        .entry(self.query.assignment_units[i].as_ref())
                        .or_insert((self.query.realized_assignment[i], 0.0));
                    entry.1 += outcome;
                }
                let treated: Vec<f64> = cluster_totals
                    .values()
                    .filter_map(|(assigned, total)| assigned.then_some(*total))
                    .collect();
                let control: Vec<f64> = cluster_totals
                    .values()
                    .filter_map(|(assigned, total)| (!assigned).then_some(*total))
                    .collect();
                let clusters = cluster_totals.len();
                let scale = clusters as f64 / n as f64;
                let effect = scale
                    * (treated.iter().sum::<f64>() / treated.len() as f64
                        - control.iter().sum::<f64>() / control.len() as f64);
                let variance = scale.powi(2)
                    * (sample_variance(&treated) / treated.len() as f64
                        + sample_variance(&control) / control.len() as f64);
                (
                    effect,
                    variance,
                    clusters - treated_clusters,
                    *treated_clusters,
                    Arc::from([]),
                    Arc::<str>::from("cluster"),
                    Arc::<str>::from("cluster_neyman_variance_upper_bound_no_interval"),
                    "Complete cluster-randomization HT ITT over cluster outcome totals, with Neyman conservative variance; no confidence interval is reported",
                )
            }
            antecedent_core::RandomizationDesign::Stratified { blocks, treated_per_row } => {
                let mut grouped: std::collections::BTreeMap<&str, (Vec<f64>, Vec<f64>, usize)> =
                    std::collections::BTreeMap::new();
                for i in 0..n {
                    let group = grouped
                        .entry(&blocks[i])
                        .or_insert_with(|| (Vec::new(), Vec::new(), treated_per_row[i]));
                    if self.query.realized_assignment[i] {
                        group.0.push(outcomes[i]);
                    } else {
                        group.1.push(outcomes[i]);
                    }
                }
                let mut effect = 0.0;
                let mut variance = 0.0;
                let mut control_units = 0;
                let mut treatment_units = 0;
                for (_, (treated, control, _)) in &grouped {
                    let size = treated.len() + control.len();
                    let weight = size as f64 / n as f64;
                    effect += weight
                        * (treated.iter().sum::<f64>() / treated.len() as f64
                            - control.iter().sum::<f64>() / control.len() as f64);
                    variance += weight.powi(2)
                        * (sample_variance(treated) / treated.len() as f64
                            + sample_variance(control) / control.len() as f64);
                    treatment_units += treated.len();
                    control_units += control.len();
                }
                (
                    effect,
                    variance,
                    control_units,
                    treatment_units,
                    Arc::clone(blocks),
                    Arc::<str>::from("stratified"),
                    Arc::<str>::from("stratified_neyman_variance_upper_bound_no_interval"),
                    "Stratified difference in means weighted by block size, with blockwise Neyman conservative variance estimate; no confidence interval is reported",
                )
            }
        };
        let estimate = EffectEstimate::new(
            effect,
            variance.sqrt(),
            self.identification.required_assumptions.clone(),
            OverlapPolicy::ExplicitOverride,
        );
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
                estimator_id: randomized_estimator_id(&self.query),
                treatment: self.query.outcome,
                outcome: self.query.outcome,
                identify_cached: true,
                extra_diagnostics: vec![Diagnostic::new(
                    if matches!(self.query.design, antecedent_core::RandomizationDesign::Bernoulli)
                    {
                        "estimate.randomized.ht_itt"
                    } else {
                        "estimate.randomized.neyman_itt"
                    },
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
        result.randomized_effect = Some(crate::RandomizedEffectEstimate {
            effect,
            variance_upper_bound: variance,
            minimum_assignment_probability: self
                .query
                .assignment_probabilities
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min),
            assignment_design,
            blocks,
            control_units,
            treatment_units,
            uncertainty,
            assignment_units: Arc::clone(&self.query.assignment_units),
            outcome_units: Arc::clone(&self.query.outcome_units),
            treatment_arms: self.query.treatment_arms.clone(),
        });
        result.treatment = None;
        Ok(result)
    }
}

impl Study {
    pub(super) fn execute_policy_value(
        &self,
        data: &TabularData,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        CheckedPolicyValueOperation::checked(self, data, physical)?.execute(data, ctx)
    }

    pub(super) fn execute_randomized(
        &self,
        data: &TabularData,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        CheckedRandomizedOperation::checked(self, data, physical)?.execute(data, ctx)
    }
}

/// Frozen, held-out policy scoring operation. Prediction vectors and ownership
/// declarations are part of the retained query identity.
#[derive(Clone)]
pub(crate) struct CheckedPolicyValueOperation {
    query: antecedent_core::PolicyValueQuery,
    source_schema: antecedent_core::CausalSchema,
    source_rows: usize,
    identification: IdentificationResult,
    estimand: IdentifiedEstimand,
    physical: PhysicalExecutionPlan,
    result_context: IdentifiedResultContext,
}

impl std::fmt::Debug for CheckedPolicyValueOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CheckedPolicyValueOperation")
            .field("query", &self.query)
            .field("source_rows", &self.source_rows)
            .finish_non_exhaustive()
    }
}

impl CheckedPolicyValueOperation {
    pub(crate) fn checked(
        study: &Study,
        data: &TabularData,
        physical: &PhysicalExecutionPlan,
    ) -> Result<Self, CausalError> {
        let CausalQuery::PolicyValue(query) = &study.query else {
            return Err(CausalError::Compile {
                message: "policy-value execution requires PolicyValueQuery".into(),
            });
        };
        query.validate().map_err(|e| CausalError::Compile { message: e.to_string() })?;
        if query.assignment.len() != data.row_count()
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
        data.schema()
            .get(query.outcome)
            .map_err(|e| CausalError::Compile { message: e.to_string() })?;
        let n = data.row_count();
        let synthetic = antecedent_core::RandomizedEffectQuery::bernoulli_itt(
            query.outcome,
            query.assignment.clone(),
            (0..n)
                .map(|i| query.propensity[if query.propensity.len() == 1 { 0 } else { i }])
                .collect::<Vec<_>>(),
            query.evaluation_subject_ids.clone(),
            query.evaluation_subject_ids.clone(),
            ("control", "treated"),
        );
        let (mut identification, estimand) = randomized_identification(&synthetic);
        identification.query = CausalQuery::PolicyValue(query.clone());
        Ok(Self {
            query: query.clone(),
            source_schema: data.schema().clone(),
            source_rows: n,
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
        if data.schema() != &self.source_schema || data.row_count() != self.source_rows {
            return Err(CausalError::Unsupported {
                message: "policy value refresh requires the prepared schema and row order",
            });
        }
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled {
                stage: super::super::stage::STAGE_ESTIMATE_POINT,
            });
        }
        let y = match data.column(self.query.outcome).map_err(CausalError::from)? {
            antecedent_data::ColumnView::Float64(values) => values.values.as_slice(),
            _ => {
                return Err(CausalError::Unsupported {
                    message: "policy value outcome must be continuous",
                });
            }
        };
        let ipw = self.query.mu0.is_empty();
        let score = if ipw {
            antecedent_estimate::policy_value::evaluate_policy_value_ipw_scores(
                y, &self.query.assignment, &self.query.actions, &self.query.propensity,
                &self.query.reference, &self.query.costs, &self.query.reference_costs,
            )
        } else {
            antecedent_estimate::policy_value::evaluate_policy_value_scores(
                y, &self.query.assignment, &self.query.actions, &self.query.propensity,
                &self.query.mu0, &self.query.mu1, &self.query.reference,
                &self.query.costs, &self.query.reference_costs,
            )
        }
        .map_err(|message| CausalError::Unsupported { message })?;
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
                estimator_id: if ipw { EstimatorId::RandomizedIpwPolicy } else { EstimatorId::RandomizedDrPolicy },
                treatment: self.query.outcome,
                outcome: self.query.outcome,
                identify_cached: true,
                extra_diagnostics: vec![Diagnostic::new(
                    if ipw { "estimate.policy_value.ipw" } else { "estimate.policy_value.doubly_robust" },
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Info,
                    if ipw {
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
        result.policy_value = Some(crate::PolicyValueEstimate {
            policy_value: score.policy_value,
            reference_value: score.reference_value,
            incremental_value: score.incremental_value,
            relative_value_gap: score.relative_value_gap,
            treatment_rate: score.treatment_rate,
            total_cost: score.total_cost,
            policy_standard_error: score.policy_standard_error,
            reference_standard_error: score.reference_standard_error,
            incremental_standard_error: score.incremental_standard_error,
            prediction_ownership: Arc::from(if ipw {
                "no_outcome_nuisance_predictions"
            } else if self.query.disjoint_training_subjects {
                "declared_disjoint_training_subject_ids"
            } else {
                "caller_declared_cross_fitted_excluded_fold_ids"
            }),
            propensity_min: score.propensity_min,
            propensity_max: score.propensity_max,
            uncertainty: Arc::from(if ipw {
                "ipw_row_score_standard_error_independent_subjects"
            } else {
                "row_score_standard_error_independent_subjects"
            }),
        });
        Ok(result)
    }
}

pub(crate) fn randomized_identification(
    query: &antecedent_core::RandomizedEffectQuery,
) -> (IdentificationResult, IdentifiedEstimand) {
    let mut assumptions = antecedent_core::AssumptionSet::default();
    let mut standard = vec![Assumption::Consistency, Assumption::Positivity];
    if !matches!(query.design, antecedent_core::RandomizationDesign::Cluster { .. }) {
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
                antecedent_core::RandomizationDesign::Cluster { .. } =>
                    "Treatment assignment follows complete randomization of independent clusters; arbitrary dependence is allowed within clusters",
            }),
        },
        source: AssumptionSource::UserDeclared,
        scope: AssumptionScope::Identification,
        status: AssumptionStatus::Declared,
    });
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
    arena.set_derivation(
        functional,
        antecedent_expr::DerivationMeta::rule(
            "randomized.itt",
            Some(Arc::from("randomized assignment identifies the intention-to-treat contrast")),
        ),
    );
    let estimand = IdentifiedEstimand::new(
        "randomized.itt",
        Arc::from([]),
        Arc::from([]),
        Arc::from([]),
        functional,
        None,
    );
    let mut trace = DerivationTrace::default();
    trace.push(
        "randomized.itt",
        "the declared randomized assignment identifies the ITT without a causal graph",
    );
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

fn randomized_estimator_id(query: &antecedent_core::RandomizedEffectQuery) -> EstimatorId {
    match &query.design {
        antecedent_core::RandomizationDesign::Bernoulli => {
            if query.fixed_cuped.is_some() {
                EstimatorId::RandomizedFixedCupedHt
            } else {
                EstimatorId::RandomizedHt
            }
        },
        antecedent_core::RandomizationDesign::Complete { .. }
        | antecedent_core::RandomizationDesign::Stratified { .. }
        | antecedent_core::RandomizationDesign::Cluster { .. } => EstimatorId::RandomizedNeyman,
    }
}
