//! Retained graphless fuzzy-discontinuity and regression-kink execution.
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use antecedent_estimate::local_polynomial_ratio::fit_local_polynomial_ratio;

#[derive(Clone)]
pub(crate) struct CheckedLocalPolynomialRatioOperation {
    query: antecedent_core::LocalPolynomialRatioQuery,
    schema: antecedent_core::CausalSchema,
    rows: usize,
    identification: IdentificationResult,
    estimand: IdentifiedEstimand,
    physical: PhysicalExecutionPlan,
    result_context: IdentifiedResultContext,
}

impl std::fmt::Debug for CheckedLocalPolynomialRatioOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CheckedLocalPolynomialRatioOperation")
            .field("query", &self.query)
            .field("rows", &self.rows)
            .finish_non_exhaustive()
    }
}

impl CheckedLocalPolynomialRatioOperation {
    pub(crate) fn checked(
        study: &Study,
        data: &TabularData,
        physical: &PhysicalExecutionPlan,
    ) -> Result<Self, CausalError> {
        let CausalQuery::LocalPolynomialRatio(query) = &study.query else {
            return Err(CausalError::Compile {
                message: "local ratio operation requires LocalPolynomialRatioQuery".into(),
            });
        };
        query.validate().map_err(|error| CausalError::Compile { message: error.to_string() })?;
        if study.graph.class() != GraphClass::RandomizedTrial
            || study.structure_source != crate::support::StructureSource::RandomizedTrial
            || !matches!(study.inference, InferenceMode::Frequentist)
            || study.refute != RefuteSuite::None
            || study.bootstrap_replicates != 0
            || !study.custom_validators.is_empty()
        {
            return Err(CausalError::Unsupported {
                message: "local ratio requires a graphless, fixed-bandwidth frequentist design",
            });
        }
        for variable in [query.outcome, query.treatment, query.running] {
            data.schema()
                .get(variable)
                .map_err(|error| CausalError::Compile { message: error.to_string() })?;
            if !matches!(
                data.column(variable).map_err(CausalError::from)?,
                antecedent_data::ColumnView::Float64(_)
            ) {
                return Err(CausalError::Unsupported {
                    message: "local ratio requires continuous outcome, treatment, and running columns",
                });
            }
        }
        let (identification, estimand) = local_polynomial_ratio_identification(query);
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
                message: "local ratio refresh requires the prepared schema and row count",
            });
        }
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled {
                stage: super::super::stage::STAGE_ESTIMATE_POINT,
            });
        }
        let column = |id| -> Result<&[f64], CausalError> {
            match data.column(id).map_err(CausalError::from)? {
                antecedent_data::ColumnView::Float64(column) => Ok(column.values.as_slice()),
                _ => Err(CausalError::Unsupported {
                    message: "local ratio requires continuous columns",
                }),
            }
        };
        let fit = fit_local_polynomial_ratio(
            column(self.query.running)?,
            column(self.query.outcome)?,
            column(self.query.treatment)?,
            self.query.cutoff,
            self.query.bandwidth,
            self.query.kink,
        )
        .map_err(|message| CausalError::Compile { message })?;
        let interval_available = fit.standard_error.is_finite() && fit.standard_error > 0.0;
        let estimate = EffectEstimate::new(
            fit.estimate,
            if interval_available { fit.standard_error } else { f64::NAN },
            self.identification.required_assumptions.clone(),
            antecedent_estimate::OverlapPolicy::ExplicitOverride,
        );
        let mut result = finish_identified_execute_with_context(
            &self.result_context,
            Some(data),
            IdentifiedExecuteFinish {
                physical: &self.physical,
                identification: self.identification.clone(),
                estimand: self.estimand.clone(),
                estimate,
                identifier_id: IdentifierId::RandomizedDesign,
                estimator_id: EstimatorId::RandomizedHt,
                treatment: self.query.treatment,
                outcome: self.query.outcome,
                identify_cached: false,
                extra_diagnostics: vec![Diagnostic::new(
                    "estimate.quasi.local_ratio.support",
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Info,
                    format!(
                        "{} left and {} right local observations; first stage {}; fixed bandwidth {}; {} pilot and HC0 delta-method ratio SE; 95% normal interval calibrated on strong-first-stage known-truth fixtures{}",
                        fit.n_left,
                        fit.n_right,
                        fit.first_stage,
                        self.query.bandwidth,
                        if self.query.kink { "quartic" } else { "cubic" },
                        if interval_available {
                            ""
                        } else {
                            "; no positive finite SE, interval withheld"
                        }
                    ),
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
        result.local_polynomial_ratio = Some(crate::LocalPolynomialRatioEstimate {
            effect: fit.estimate,
            reduced_form: fit.reduced_form,
            first_stage: fit.first_stage,
            cutoff: self.query.cutoff,
            bandwidth: self.query.bandwidth,
            kink: self.query.kink,
            n_left: fit.n_left,
            n_right: fit.n_right,
            standard_error: fit.standard_error,
            ci_lower: interval_available.then_some(fit.ci_lower),
            ci_upper: interval_available.then_some(fit.ci_upper),
            reduced_form_standard_error: fit.reduced_form_standard_error,
            first_stage_standard_error: fit.first_stage_standard_error,
            uncertainty: Arc::from("rbc_hc0_delta_normal_fixed_bandwidth"),
        });
        // Graphless license handshake: the estimator's own weak-first-stage and
        // bandwidth-support screens already gate `interval_available`; the license
        // only confirms the exact family/design/method and the retained local
        // observation floor before publishing the pointwise ratio interval.
        if crate::support::license_if_graphless(
            crate::support::GraphlessSupportKey {
                family: "local_polynomial_ratio",
                design: if self.query.kink { "regression_kink" } else { "fuzzy_jump" },
                method: if self.query.kink {
                    "local_quartic_slope_rbc_hc0_delta"
                } else {
                    "local_quadratic_cubic_rbc_hc0_delta"
                },
                inference_claim: "pointwise_95_normal_interval",
            },
            crate::support::GraphlessAssignmentSupport {
                assignment_unit: "unit",
                treated: fit.n_right,
                control: fit.n_left,
                interval_95_published: interval_available,
                reported_intervals: usize::from(interval_available),
                ..Default::default()
            },
        )
        .is_some()
        {
            result.support_status = Some(crate::support::CellStatus::Licensed);
        }
        Ok(result)
    }
}

impl Study {
    pub(super) fn execute_local_polynomial_ratio(
        &self,
        data: &TabularData,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        CheckedLocalPolynomialRatioOperation::checked(self, data, physical)?.execute(data, ctx)
    }
}

pub(crate) fn local_polynomial_ratio_identification(
    query: &antecedent_core::LocalPolynomialRatioQuery,
) -> (IdentificationResult, IdentifiedEstimand) {
    let mut assumptions = antecedent_core::AssumptionSet::default();
    let local = if query.kink {
        "continuous untreated potential-outcome and treatment derivatives at the cutoff"
    } else {
        "continuous untreated potential outcomes at the cutoff"
    };
    for (id, description) in [
        ("local_continuity", local),
        (
            "no_precise_running_variable_manipulation",
            "units cannot precisely sort around the cutoff",
        ),
        (
            "exclusion_restriction",
            "the threshold affects the outcome through treatment receipt or dose",
        ),
        (
            "local_monotonicity",
            "the threshold does not move treatment in opposite directions across units",
        ),
        (
            "independent_local_observations",
            "the fitted local observations do not interfere with each other",
        ),
    ] {
        assumptions.push(antecedent_core::AssumptionRecord {
            assumption: antecedent_core::Assumption::Custom {
                id: Arc::from(id),
                description: Arc::from(description),
            },
            source: antecedent_core::AssumptionSource::UserDeclared,
            scope: antecedent_core::AssumptionScope::Identification,
            status: antecedent_core::AssumptionStatus::Declared,
        });
    }
    let rule =
        if query.kink { "regression_kink.local_slope_ratio" } else { "fuzzy_rd.local_jump_ratio" };
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
    arena.set_derivation(
        functional,
        antecedent_expr::DerivationMeta::rule(
            rule,
            Some(Arc::from("local treatment response at a declared cutoff and bandwidth")),
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
    derivation.push(rule, "the local reduced-form contrast divided by the local treatment first stage identifies the threshold complier response under the declared continuity, exclusion, and monotonicity assumptions");
    let identification = IdentificationResult::identified(
        CausalQuery::LocalPolynomialRatio(query.clone()),
        vec![estimand.clone()],
        arena,
        derivation,
        assumptions,
        IdentificationPerformanceRecord::default(),
    );
    (identification, estimand)
}
