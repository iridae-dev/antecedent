//! The independent-factor 2×2 reading of a four-cell multi-arm randomized design.
//!
//! Selected by declaring [`EstimatorId::RandomizedFactorialBernoulliHt`] on a
//! `RandomizedEffectQuery` whose `MultiArm` design has exactly four actions in cell order
//! `(a=0,b=0)`, `(a=0,b=1)`, `(a=1,b=0)`, `(a=1,b=1)`, each row's probability row stating the
//! known probability of every cell. The route calls
//! [`antecedent_estimate::factorial_bernoulli::estimate_factorial_2x2_cells`] with each row's
//! realized cell and that cell's probability: the Horvitz–Thompson cell means, the factor-A and
//! factor-B main effects, their interaction and the Young-inequality variance bounds. A design
//! whose probability rows are the products `qa · qb` of two independent Bernoulli factors gets
//! the numbers of `estimate_factorial_2x2` to the bit. The three pointwise normal intervals
//! `effect ± z · sqrt(bound)` are published once every cell holds
//! [`FACTORIAL_BERNOULLI_MIN_CELL_ROWS`] rows and every bound is positive.
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_io::graphless_support::{
    FACTORIAL_BERNOULLI_DESIGN, FACTORIAL_BERNOULLI_INTERVAL_UNCERTAINTY, FACTORIAL_BERNOULLI_KEY,
    FACTORIAL_BERNOULLI_MIN_CELL_ROWS, FACTORIAL_BERNOULLI_POINT_UNCERTAINTY,
};

use super::super::randomized_fit::assignment_probability_floor;
use super::{
    Arc, CausalError, CheckedRandomizedOperation, Diagnostic, DiagnosticKind, DiagnosticSeverity,
    EffectEstimate, EstimatorId, ExecutionContext, IdentifiedExecuteExtras,
    IdentifiedExecuteFinish, IdentifierId, Instant, OverlapPolicy, StudyResult, TableView,
    TabularData, finish_identified_execute_with_context,
};

/// Diagnostic code naming the estimator that ran.
const DIAGNOSTIC_ID: &str = "estimate.randomized.factorial_bernoulli_ht";

/// The estimand label of the primary (factor-A) contrast, shared with the fixed-cell factorial.
const ESTIMAND: &str = "factorial_primary_main_effect";

/// Whether the study declares the independent-factor 2×2 reading.
pub(super) fn selected(declared: Option<EstimatorId>) -> bool {
    declared == Some(EstimatorId::RandomizedFactorialBernoulliHt)
}

/// Refuse the independent-factor reading of anything but a complete four-cell multi-arm design.
pub(super) fn admit(
    query: &antecedent_core::RandomizedEffectQuery,
    missing: Option<&antecedent_data::MissingDeclaration>,
) -> Result<(), CausalError> {
    let antecedent_core::RandomizationDesign::MultiArm { arms, .. } = &query.design else {
        return Err(crate::unsupported_reason!(
            "option_not_applicable",
            "the randomized.factorial_bernoulli_ht estimator reads a multi-arm design whose four \
             actions are the 2x2 cells (a=0,b=0), (a=0,b=1), (a=1,b=0), (a=1,b=1); this design \
             is not multi-arm"
        ));
    };
    if arms.len() != 4 {
        return Err(crate::unsupported_reason!(
            "option_not_applicable",
            "the randomized.factorial_bernoulli_ht estimator reads exactly four actions, the 2x2 \
             cells (a=0,b=0), (a=0,b=1), (a=1,b=0), (a=1,b=1) in that order"
        ));
    }
    if let Some(declaration) = missing
        && declaration.policy != antecedent_data::MissingPolicy::ErrorOnMissing
    {
        return Err(CausalError::Estimate(
            antecedent_estimate::missing_policy::policy_not_applicable(
                declaration.policy,
                crate::analysis::missing_data::MissingRoute::Randomized.label(),
                "the independent-factor 2x2 contrasts read every assigned row's outcome and have \
                 no responder form",
                &[antecedent_data::MissingPolicy::ErrorOnMissing],
            ),
        ));
    }
    Ok(())
}

impl CheckedRandomizedOperation {
    /// Fit the independent-factor 2×2 contrasts and assemble the randomized result.
    #[expect(
        clippy::too_many_lines,
        reason = "one linear execution: read the rows, call the kernel, assemble and support the result"
    )]
    pub(super) fn execute_factorial_bernoulli(
        &self,
        data: &TabularData,
        _ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        admit(&self.query, self.extra.as_ref())?;
        let antecedent_core::RandomizationDesign::MultiArm { assignment, probabilities, arms } =
            &self.query.design
        else {
            unreachable!("admit refuses every other design")
        };
        if data.storage().weights().is_some() {
            return Err(CausalError::Estimate(
                antecedent_estimate::sampling_weights::unsupported_route(
                    "the independent-factor 2x2 contrasts read no sampling weights",
                ),
            ));
        }
        let antecedent_data::ColumnView::Float64(column) =
            data.column(self.query.outcome).map_err(CausalError::from)?
        else {
            return Err(CausalError::Unsupported {
                message: "randomized ITT outcome must be continuous",
            });
        };
        let outcomes = column.values.as_slice();
        let observed: Vec<bool> =
            (0..outcomes.len()).map(|i| column.validity.is_valid(i)).collect();
        if !observed.iter().all(|seen| *seen) {
            return Err(CausalError::Unsupported {
                message: "the independent-factor 2x2 contrasts need every assigned row's outcome",
            });
        }
        if outcomes.iter().any(|y| !y.is_finite()) {
            return Err(CausalError::Unsupported {
                message: "randomized ITT requires finite outcomes",
            });
        }
        let cell_probability: Vec<f64> =
            assignment.iter().enumerate().map(|(row, &cell)| probabilities[row][cell]).collect();
        let fit = antecedent_estimate::factorial_bernoulli::estimate_factorial_2x2_cells(
            outcomes,
            assignment,
            &cell_probability,
        )
        .map_err(|message| CausalError::Unsupported { message })?;
        let contrasts = &fit.contrasts;
        let cell_rows = &contrasts.cell_support;
        let control_units = cell_rows[0] + cell_rows[1];
        let treatment_units = cell_rows[2] + cell_rows[3];
        let bounds = [
            contrasts.factor_a_variance_bound,
            contrasts.factor_b_variance_bound,
            contrasts.interaction_variance_bound,
        ];
        let effects =
            [contrasts.factor_a_effect, contrasts.factor_b_effect, contrasts.interaction_effect];
        if effects.iter().chain(bounds.iter()).any(|value| !value.is_finite())
            || bounds.iter().any(|bound| *bound < 0.0)
        {
            return Err(CausalError::Unsupported {
                message: "the independent-factor 2x2 contrasts or their variance bounds are not \
                          finite under the declared cell probabilities",
            });
        }
        let publish = cell_rows.iter().all(|rows| *rows >= FACTORIAL_BERNOULLI_MIN_CELL_ROWS)
            && bounds.iter().all(|bound| *bound > 0.0);
        let interval = |effect: f64, bound: f64| {
            publish.then(|| {
                let span = antecedent_stats::NORMAL_Q975 * bound.sqrt();
                [effect - span, effect + span]
            })
        };
        let standard_error = publish.then(|| bounds[0].sqrt());
        let payload = crate::RandomizedEffectEstimate {
            effect: effects[0],
            estimand: Arc::from(ESTIMAND),
            intention_to_treat_effect: None,
            first_stage_effect: None,
            received_treatment: None,
            randomization_p_value: None,
            randomization_allocations: None,
            second_factor_effect: Some(effects[1]),
            factorial_interaction: Some(effects[2]),
            second_factor_variance: Some(bounds[1]),
            factorial_interaction_variance: Some(bounds[2]),
            multi_arm_values: arms
                .iter()
                .enumerate()
                .map(|(cell, label)| {
                    (
                        Arc::clone(label),
                        contrasts.cell_means[cell],
                        fit.cell_variances[cell],
                        cell_rows[cell],
                    )
                })
                .collect::<Vec<_>>()
                .into(),
            variance_upper_bound: bounds[0],
            standard_error,
            interval_95: interval(effects[0], bounds[0]),
            second_factor_interval_95: interval(effects[1], bounds[1]),
            factorial_interaction_interval_95: interval(effects[2], bounds[2]),
            multi_arm_intervals_95: Arc::from([]),
            minimum_assignment_probability: assignment_probability_floor(&self.query, true),
            assignment_design: Arc::from(FACTORIAL_BERNOULLI_DESIGN),
            blocks: Arc::from([]),
            periods: Arc::from([]),
            control_units,
            treatment_units,
            uncertainty: Arc::from(if publish {
                FACTORIAL_BERNOULLI_INTERVAL_UNCERTAINTY
            } else {
                FACTORIAL_BERNOULLI_POINT_UNCERTAINTY
            }),
            assignment_units: Arc::clone(&self.query.assignment_units),
            outcome_units: Arc::clone(&self.query.outcome_units),
            treatment_arms: self.query.treatment_arms.clone(),
        };
        let estimate = EffectEstimate::new(
            payload.effect,
            standard_error.unwrap_or(f64::NAN),
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
                estimator_id: EstimatorId::RandomizedFactorialBernoulliHt,
                treatment: self.query.outcome,
                outcome: self.query.outcome,
                identify_cached: true,
                extra_diagnostics: vec![Diagnostic::new(
                    DIAGNOSTIC_ID,
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Info,
                    if publish {
                        "Independent-factor 2x2 Horvitz-Thompson contrasts over the four cells of \
                         the multi-arm design: each cell mean is (1/n) sum of y/q over the rows \
                         realized in it, the factor-A and factor-B main effects and the \
                         interaction are fixed contrasts of the cell means, and each variance is \
                         the Young-inequality bound 4 sum w^2 V over the diagonal \
                         Horvitz-Thompson cell variances. The three 95% normal intervals are \
                         separate pointwise claims on that bound; their coverage is not measured"
                    } else {
                        "Independent-factor 2x2 Horvitz-Thompson contrasts over the four cells of \
                         the multi-arm design with Young-inequality variance bounds; no interval \
                         is published until every cell holds 30 rows and every bound is positive"
                    },
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
        result.payload = crate::result::ResultPayload::RandomizedEffect(payload);
        let min_probability = assignment_probability_floor(&self.query, true);
        if crate::support::support_if_graphless(
            FACTORIAL_BERNOULLI_KEY,
            crate::support::GraphlessAssignmentSupport {
                assignment_unit: "unit",
                treated: treatment_units,
                control: control_units,
                interval_95_published: publish,
                rows: self.rows,
                min_factorial_cell: cell_rows.iter().copied().min().unwrap_or(0),
                min_action_rows: cell_rows.iter().copied().min().unwrap_or(0),
                min_probability,
                reported_intervals: if publish { 3 } else { 0 },
                all_reported_intervals: publish,
                ..Default::default()
            },
        )
        .is_some()
        {
            result.support_status = Some(crate::support::CellStatus::Supported);
        }
        result.rebind_interval(false);
        result.treatment = None;
        result.extras.missing_data = Some(self.missing_report(data, &observed)?);
        Ok(result)
    }
}
