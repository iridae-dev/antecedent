//! Design-based prepare routes (panel, dose, survival, policy).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::Study;
use antecedent_core::CausalQuery;
use antecedent_data::TableView;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

use super::{PrepareRequest, PyPreparedAnalysis};
use crate::py_err;

#[pymethods]
impl PyPreparedAnalysis {
    /// Freeze a two-period panel or repeated-cross-section DiD with cluster uncertainty.
    #[staticmethod]
    #[pyo3(signature = (names, columns, outcome, treated, post, subjects, clusters, *, repeated_cross_section=false, accepted=false, seed=1, threads=None, options=None))]
    fn prepare_panel_did(
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        outcome: String,
        treated: Vec<bool>,
        post: Vec<bool>,
        subjects: Vec<String>,
        clusters: Vec<String>,
        repeated_cross_section: bool,
        accepted: bool,
        seed: u64,
        threads: Option<u32>,
        options: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let req = PrepareRequest::parse(names, options.as_ref(), seed, threads)?;
        req.refuse_design("panel difference-in-differences")?;
        let data = req.load_columns(py, columns)?;
        req.off_gil(py, move |req| {
            let outcome_id = crate::graph_build::schema_var_id(data.schema(), &outcome)?;
            let subject_arc: Vec<Arc<str>> =
                subjects.iter().map(|s| Arc::<str>::from(s.as_str())).collect();
            let cluster_arc: Vec<Arc<str>> =
                clusters.iter().map(|s| Arc::<str>::from(s.as_str())).collect();
            let query = if repeated_cross_section {
                antecedent_core::PanelDidQuery::repeated_cross_section(
                    outcome_id,
                    treated,
                    post,
                    subject_arc,
                    cluster_arc,
                )
            } else {
                antecedent_core::PanelDidQuery::new(
                    outcome_id,
                    treated,
                    post,
                    subject_arc,
                    cluster_arc,
                )
            };
            query.validate().map_err(|e| py_err(antecedent::CausalError::rejected(e)))?;
            let _ = accepted;
            let builder = Study::tabular(data).query(CausalQuery::PanelDid(query));
            req.finish_applied(builder, false)
        })
    }

    /// Freeze a one-row-per-subject augmented panel ATT with supplied nuisance columns.
    #[staticmethod]
    #[pyo3(signature = (names, columns, outcome_pre, outcome_post, propensity,
        untreated_change_prediction, treated, subjects, clusters, predictions_cross_fitted,
        *, accepted=false, seed=1, threads=None, options=None))]
    #[allow(clippy::too_many_arguments)]
    fn prepare_augmented_panel_did(
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        outcome_pre: String,
        outcome_post: String,
        propensity: String,
        untreated_change_prediction: String,
        treated: Vec<bool>,
        subjects: Vec<String>,
        clusters: Vec<String>,
        predictions_cross_fitted: bool,
        accepted: bool,
        seed: u64,
        threads: Option<u32>,
        options: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let req = PrepareRequest::parse(names, options.as_ref(), seed, threads)?;
        req.refuse_design("augmented panel difference-in-differences")?;
        let data = req.load_columns(py, columns)?;
        req.off_gil(py, move |req| {
            let variable = |name: &str| crate::graph_build::schema_var_id(data.schema(), name);
            let query = antecedent_core::PanelDidQuery::augmented_panel(
                variable(&outcome_post)?,
                variable(&outcome_pre)?,
                variable(&propensity)?,
                variable(&untreated_change_prediction)?,
                treated,
                subjects.iter().map(|s| Arc::<str>::from(s.as_str())).collect::<Vec<_>>(),
                clusters.iter().map(|s| Arc::<str>::from(s.as_str())).collect::<Vec<_>>(),
                predictions_cross_fitted,
            );
            query.validate().map_err(|e| py_err(antecedent::CausalError::rejected(e)))?;
            let _ = accepted;
            let builder = Study::tabular(data).query(CausalQuery::PanelDid(query));
            req.finish_applied(builder, false)
        })
    }

    /// Freeze one cohort-period comparison in a balanced staggered adoption panel.
    #[staticmethod]
    #[pyo3(signature = (names, columns, outcome, subjects, clusters, periods, cohorts,
        target_cohort, target_period, *, accepted=false, seed=1, threads=None, options=None))]
    #[allow(clippy::too_many_arguments)]
    fn prepare_staggered_group_time(
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        outcome: String,
        subjects: Vec<String>,
        clusters: Vec<String>,
        periods: Vec<i64>,
        cohorts: Vec<i64>,
        target_cohort: i64,
        target_period: i64,
        accepted: bool,
        seed: u64,
        threads: Option<u32>,
        options: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let req = PrepareRequest::parse(names, options.as_ref(), seed, threads)?;
        req.refuse_design("staggered difference-in-differences")?;
        let data = req.load_columns(py, columns)?;
        req.off_gil(py, move |req| {
            let outcome_id = crate::graph_build::schema_var_id(data.schema(), &outcome)?;
            let subject_arc: Vec<Arc<str>> =
                subjects.iter().map(|s| Arc::<str>::from(s.as_str())).collect();
            let cluster_arc: Vec<Arc<str>> =
                clusters.iter().map(|s| Arc::<str>::from(s.as_str())).collect();
            let query = antecedent_core::PanelDidQuery::staggered_group_time(
                outcome_id,
                subject_arc,
                cluster_arc,
                periods,
                cohorts,
                target_cohort,
                target_period,
            );
            query.validate().map_err(|e| py_err(antecedent::CausalError::rejected(e)))?;
            let _ = accepted;
            let builder = Study::tabular(data).query(CausalQuery::PanelDid(query));
            req.finish_applied(builder, false)
        })
    }

    /// Freeze all cohort-specific event-time comparisons in one retained Study.
    #[staticmethod]
    #[pyo3(signature = (names, columns, outcome, subjects, clusters, periods, cohorts,
        *, accepted=false, seed=1, threads=None, options=None))]
    #[allow(clippy::too_many_arguments)]
    fn prepare_staggered_event_study(
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        outcome: String,
        subjects: Vec<String>,
        clusters: Vec<String>,
        periods: Vec<i64>,
        cohorts: Vec<i64>,
        accepted: bool,
        seed: u64,
        threads: Option<u32>,
        options: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let req = PrepareRequest::parse(names, options.as_ref(), seed, threads)?;
        req.refuse_design("staggered event study")?;
        let data = req.load_columns(py, columns)?;
        req.off_gil(py, move |req| {
            let outcome_id = crate::graph_build::schema_var_id(data.schema(), &outcome)?;
            let subject_arc: Vec<Arc<str>> =
                subjects.iter().map(|s| Arc::<str>::from(s.as_str())).collect();
            let cluster_arc: Vec<Arc<str>> =
                clusters.iter().map(|s| Arc::<str>::from(s.as_str())).collect();
            let query = antecedent_core::PanelDidQuery::staggered_event_study(
                outcome_id,
                subject_arc,
                cluster_arc,
                periods,
                cohorts,
            );
            query.validate().map_err(|error| py_err(antecedent::CausalError::rejected(error)))?;
            let _ = accepted;
            let builder = Study::tabular(data).query(CausalQuery::PanelDid(query));
            req.finish_applied(builder, false)
        })
    }

    /// Freeze one treated unit and its balanced donor pool for native synthetic control.
    #[staticmethod]
    #[pyo3(signature = (names, columns, outcome, units, periods, treated_unit,
        intervention_period, *, difference_in_differences=false, uniform_unit_randomization=false, sharp_null_effect=None, augmentation_ridge=None,
        accepted=false, seed=1, threads=None, options=None))]
    #[allow(clippy::too_many_arguments)]
    fn prepare_synthetic_control(
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        outcome: String,
        units: Vec<String>,
        periods: Vec<i64>,
        treated_unit: String,
        intervention_period: i64,
        difference_in_differences: bool,
        uniform_unit_randomization: bool,
        sharp_null_effect: Option<f64>,
        augmentation_ridge: Option<f64>,
        accepted: bool,
        seed: u64,
        threads: Option<u32>,
        options: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let req = PrepareRequest::parse(names, options.as_ref(), seed, threads)?;
        req.refuse_design("synthetic control")?;
        let data = req.load_columns(py, columns)?;
        req.off_gil(py, move |req| {
            let outcome_id = crate::graph_build::schema_var_id(data.schema(), &outcome)?;
            let unit_arc: Vec<Arc<str>> =
                units.iter().map(|unit| Arc::<str>::from(unit.as_str())).collect();
            let query = antecedent_core::SyntheticControlQuery::new(
                outcome_id,
                unit_arc,
                periods,
                Arc::<str>::from(treated_unit),
                intervention_period,
            );
            let query =
                if difference_in_differences { query.difference_in_differences() } else { query };
            let query = if uniform_unit_randomization {
                query.with_uniform_unit_randomization()
            } else {
                query
            };
            let query = if let Some(effect) = sharp_null_effect {
                query.with_sharp_null_effect(effect)
            } else {
                query
            };
            let query = if let Some(ridge) = augmentation_ridge {
                query.with_augmentation(ridge)
            } else {
                query
            };
            query.validate().map_err(|error| py_err(antecedent::CausalError::rejected(error)))?;
            let _ = accepted;
            let builder = Study::tabular(data).query(CausalQuery::SyntheticControl(query));
            req.finish_applied(builder, false)
        })
    }

    /// Freeze a baseline-group continuous-dose response grid.
    #[staticmethod]
    #[pyo3(signature = (names, columns, outcome, dose, dose_density, baseline_groups,
        target_doses, bandwidth, density_provenance, *, min_local_support=3,
        policy_doses=None, reference_doses=None,
        accepted=false, seed=1, threads=None, options=None))]
    #[allow(clippy::too_many_arguments)]
    fn prepare_continuous_dose_response(
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        outcome: String,
        dose: String,
        dose_density: String,
        baseline_groups: Vec<String>,
        target_doses: Vec<f64>,
        bandwidth: f64,
        density_provenance: String,
        min_local_support: usize,
        policy_doses: Option<Vec<(String, f64)>>,
        reference_doses: Option<Vec<(String, f64)>>,
        accepted: bool,
        seed: u64,
        threads: Option<u32>,
        options: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let req = PrepareRequest::parse(names, options.as_ref(), seed, threads)?;
        req.refuse_design("continuous-dose response")?;
        let data = req.load_columns(py, columns)?;
        req.off_gil(py, move |req| {
            let fixed_policy = match (policy_doses, reference_doses) {
                (None, None) => None,
                (Some(policy), Some(reference)) => Some(antecedent_core::FixedGroupDosePolicy {
                    policy_doses: policy
                        .into_iter()
                        .map(|(group, dose)| (Arc::from(group), dose))
                        .collect::<Vec<_>>()
                        .into(),
                    reference_doses: reference
                        .into_iter()
                        .map(|(group, dose)| (Arc::from(group), dose))
                        .collect::<Vec<_>>()
                        .into(),
                }),
                _ => {
                    return Err(py_err(antecedent::CausalError::Compile {
                        message: "fixed dose policy requires both policy_doses and reference_doses"
                            .into(),
                    }));
                }
            };
            let query = antecedent_core::ContinuousDoseResponseQuery {
                outcome: crate::graph_build::schema_var_id(data.schema(), &outcome)?,
                dose: crate::graph_build::schema_var_id(data.schema(), &dose)?,
                dose_density: crate::graph_build::schema_var_id(data.schema(), &dose_density)?,
                baseline_groups: baseline_groups
                    .into_iter()
                    .map(Arc::<str>::from)
                    .collect::<Vec<_>>()
                    .into(),
                target_doses: target_doses.into(),
                bandwidth,
                min_local_support,
                density_provenance: Arc::from(density_provenance),
                fixed_policy,
            };
            query.validate().map_err(|error| py_err(antecedent::CausalError::rejected(error)))?;
            let _ = accepted;
            let builder = Study::tabular(data).query(CausalQuery::ContinuousDoseResponse(query));
            req.finish_applied(builder, false)
        })
    }

    /// Freeze a fixed-window fuzzy RD or regression-kink analysis.
    #[staticmethod]
    #[pyo3(signature = (names, columns, outcome, treatment, running, cutoff, bandwidth,
        *, kink=false, accepted=false, seed=1, threads=None, options=None))]
    #[allow(clippy::too_many_arguments)]
    fn prepare_local_polynomial_ratio(
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        outcome: String,
        treatment: String,
        running: String,
        cutoff: f64,
        bandwidth: f64,
        kink: bool,
        accepted: bool,
        seed: u64,
        threads: Option<u32>,
        options: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let req = PrepareRequest::parse(names, options.as_ref(), seed, threads)?;
        req.refuse_design("local polynomial ratio")?;
        let data = req.load_columns(py, columns)?;
        req.off_gil(py, move |req| {
            let query = antecedent_core::LocalPolynomialRatioQuery {
                outcome: crate::graph_build::schema_var_id(data.schema(), &outcome)?,
                treatment: crate::graph_build::schema_var_id(data.schema(), &treatment)?,
                running: crate::graph_build::schema_var_id(data.schema(), &running)?,
                cutoff,
                bandwidth,
                kink,
            };
            query.validate().map_err(|error| py_err(antecedent::CausalError::rejected(error)))?;
            let _ = accepted;
            let builder = Study::tabular(data).query(CausalQuery::LocalPolynomialRatio(query));
            req.finish_applied(builder, false)
        })
    }

    /// Freeze a known-randomization longitudinal regime value over whole subjects.
    #[staticmethod]
    #[pyo3(signature = (names, columns, outcome, periods, treatment_history, regime_actions,
        treatment_probabilities, censoring_probabilities, outcome_observed, subject_ids, fold_ids,
        excluded_fold_predictions, probabilities_known_by_design, minimum_probability,
        *, method="ipw", period_outcome_predictions=Vec::new(), known_fixed_outcome_predictions=false, stabilizing_numerator_probabilities=Vec::new(), q_predictions=Vec::new(), observation_history=Vec::new(), prediction_fold_ids=Vec::new(), rule_id=None, rule_version=None, rule_provenance=None, accepted=false, seed=1, threads=None, options=None))]
    #[allow(clippy::too_many_arguments)]
    fn prepare_longitudinal_regime(
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        outcome: String,
        periods: usize,
        treatment_history: Vec<bool>,
        regime_actions: Vec<bool>,
        treatment_probabilities: Vec<f64>,
        censoring_probabilities: Vec<f64>,
        outcome_observed: Vec<bool>,
        subject_ids: Vec<String>,
        fold_ids: Vec<u32>,
        excluded_fold_predictions: bool,
        probabilities_known_by_design: bool,
        minimum_probability: f64,
        method: &str,
        period_outcome_predictions: Vec<f64>,
        known_fixed_outcome_predictions: bool,
        stabilizing_numerator_probabilities: Vec<f64>,
        q_predictions: Vec<f64>,
        observation_history: Vec<bool>,
        prediction_fold_ids: Vec<u32>,
        rule_id: Option<String>,
        rule_version: Option<String>,
        rule_provenance: Option<String>,
        accepted: bool,
        seed: u64,
        threads: Option<u32>,
        options: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let req = PrepareRequest::parse(names, options.as_ref(), seed, threads)?;
        req.refuse_design("longitudinal regime")?;
        let data = req.load_columns(py, columns)?;
        req.off_gil(py, move |req| {
            let outcome_id = crate::graph_build::schema_var_id(data.schema(), &outcome)?;
            let method = match method {
                "ipw" => antecedent_core::LongitudinalRegimeMethod::Ipw,
                "g_formula" => antecedent_core::LongitudinalRegimeMethod::GFormula,
                "sequential_dr" => {
                    antecedent_core::LongitudinalRegimeMethod::SequentialDoublyRobust
                }
                "marginal_structural_model" => {
                    antecedent_core::LongitudinalRegimeMethod::MarginalStructuralModel
                }
                _ => {
                    return Err(py_err(antecedent::CausalError::Compile {
                        message: "unknown longitudinal regime method".into(),
                    }));
                }
            };
            let query = antecedent_core::LongitudinalRegimeQuery {
                outcome: outcome_id,
                method,
                period_outcome_predictions: if period_outcome_predictions.is_empty() {
                    None
                } else {
                    Some(period_outcome_predictions.into())
                },
                known_fixed_outcome_predictions,
                stabilizing_numerator_probabilities: if stabilizing_numerator_probabilities
                    .is_empty()
                {
                    None
                } else {
                    Some(stabilizing_numerator_probabilities.into())
                },
                q_predictions: if q_predictions.is_empty() {
                    None
                } else {
                    Some(q_predictions.into())
                },
                observation_history: if observation_history.is_empty() {
                    None
                } else {
                    Some(observation_history.into())
                },
                prediction_fold_ids: if prediction_fold_ids.is_empty() {
                    None
                } else {
                    Some(prediction_fold_ids.into())
                },
                periods,
                treatment_history: treatment_history.into(),
                regime_actions: regime_actions.into(),
                treatment_probabilities: treatment_probabilities.into(),
                censoring_probabilities: censoring_probabilities.into(),
                outcome_observed: outcome_observed.into(),
                subject_ids: subject_ids
                    .into_iter()
                    .map(Arc::<str>::from)
                    .collect::<Vec<_>>()
                    .into(),
                fold_ids: fold_ids.into(),
                excluded_fold_predictions,
                probabilities_known_by_design,
                minimum_probability,
                rule_id: rule_id.as_deref().map(Arc::<str>::from),
                rule_version: rule_version.as_deref().map(Arc::<str>::from),
                rule_provenance: rule_provenance.as_deref().map(Arc::<str>::from),
            };
            query.validate().map_err(|e| py_err(antecedent::CausalError::rejected(e)))?;
            let _ = accepted;
            let builder = Study::tabular(data).query(CausalQuery::LongitudinalRegime(query));
            req.finish_applied(builder, false)
        })
    }

    /// Freeze a graphless randomized survival or competing-risk analysis.
    #[staticmethod]
    #[pyo3(signature = (names, columns, duration, event, treatment, tau, target_cause=None, delayed_entry=None, *, independent_given=Vec::new(), censoring_times=Vec::new(), censoring_columns=Vec::new(), censoring_probability_floor=None, accepted=false, seed=1, threads=None, options=None))]
    #[allow(clippy::too_many_arguments)]
    fn prepare_survival(
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        duration: String,
        event: String,
        treatment: String,
        tau: f64,
        target_cause: Option<i64>,
        delayed_entry: Option<String>,
        independent_given: Vec<String>,
        censoring_times: Vec<f64>,
        censoring_columns: Vec<String>,
        censoring_probability_floor: Option<f64>,
        accepted: bool,
        seed: u64,
        threads: Option<u32>,
        options: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let req = PrepareRequest::parse(names, options.as_ref(), seed, threads)?;
        req.refuse_design("randomized survival")?;
        let data = req.load_columns(py, columns)?;
        req.off_gil(py, move |req| {
            let id = |name: &str| crate::graph_build::schema_var_id(data.schema(), name);
            let censoring_ids =
                censoring_columns.iter().map(|name| id(name)).collect::<Result<Vec<_>, _>>()?;
            let conditioning_ids =
                independent_given.iter().map(|name| id(name)).collect::<Result<Vec<_>, _>>()?;
            let query = antecedent_core::SurvivalQuery {
                duration: id(&duration)?,
                event: id(&event)?,
                treatment: id(&treatment)?,
                tau,
                delayed_entry: delayed_entry.as_deref().map(id).transpose()?,
                known_censoring: censoring_probability_floor.map(|minimum_probability| {
                    antecedent_core::KnownCensoringSurvival {
                        times: Arc::from(censoring_times),
                        columns: Arc::from(censoring_ids),
                        minimum_probability,
                    }
                }),
                observation_assumption: antecedent_core::ObservationAssumption::IndependentGiven(
                    Arc::from(conditioning_ids),
                ),
                functional: target_cause.map_or(
                    antecedent_core::SurvivalFunctional::SurvivalAndRmst,
                    |target_cause| antecedent_core::SurvivalFunctional::CumulativeIncidence {
                        target_cause,
                    },
                ),
            };
            query.validate().map_err(|e| py_err(antecedent::CausalError::rejected(e)))?;
            let _ = accepted;
            let builder = Study::tabular(data).query(CausalQuery::Survival(query));
            req.finish_applied(builder, false)
        })
    }

    /// Freeze a randomized held-out doubly robust policy-value study.
    #[staticmethod]
    #[pyo3(signature = (names, columns, outcome, assignment, propensity, actions, reference,
        mu0, mu1, costs, reference_costs, evaluation_subject_ids, disjoint_training_subjects,
        crossfit_fold_ownership_valid, *, uplift_bins=vec![], uplift_bin_count=0,
        uplift_training_subject_ids=vec![], global_constraints_present=false,
        regret_candidates=vec![], regret_selected_index=None, regret_training_subject_ids=vec![],
        accepted=false, seed=1, threads=None, options=None))]
    #[allow(clippy::too_many_arguments)]
    fn prepare_policy_value(
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        outcome: String,
        assignment: Vec<bool>,
        propensity: Vec<f64>,
        actions: Vec<bool>,
        reference: Vec<bool>,
        mu0: Vec<f64>,
        mu1: Vec<f64>,
        costs: Vec<f64>,
        reference_costs: Vec<f64>,
        evaluation_subject_ids: Vec<String>,
        disjoint_training_subjects: bool,
        crossfit_fold_ownership_valid: bool,
        uplift_bins: Vec<usize>,
        uplift_bin_count: usize,
        uplift_training_subject_ids: Vec<String>,
        global_constraints_present: bool,
        regret_candidates: Vec<Vec<bool>>,
        regret_selected_index: Option<usize>,
        regret_training_subject_ids: Vec<String>,
        accepted: bool,
        seed: u64,
        threads: Option<u32>,
        options: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let req = PrepareRequest::parse(names, options.as_ref(), seed, threads)?;
        req.refuse_design("held-out policy value")?;
        let data = req.load_columns(py, columns)?;
        req.off_gil(py, move |req| {
            let outcome_id = crate::graph_build::schema_var_id(data.schema(), &outcome)?;
            if regret_selected_index.is_some() == regret_candidates.is_empty()
                || (regret_selected_index.is_none() && !regret_training_subject_ids.is_empty())
            {
                return Err(py_err(antecedent::CausalError::Compile {
                    message: "finite-class regret candidates, selected index, and training subjects must be supplied together".into(),
                }));
            }
            let query = antecedent_core::PolicyValueQuery {
                outcome: outcome_id,
                assignment: assignment.into(),
                propensity: propensity.into(),
                actions: actions.into(),
                reference: reference.into(),
                mu0: mu0.into(),
                mu1: mu1.into(),
                costs: costs.into(),
                reference_costs: reference_costs.into(),
                evaluation_subject_ids: evaluation_subject_ids
                    .into_iter()
                    .map(Arc::<str>::from)
                    .collect::<Vec<_>>()
                    .into(),
                disjoint_training_subjects,
                crossfit_fold_ownership_valid,
                global_constraints_present,
                multi_action: None,
                uplift_bins: uplift_bins.into(),
                uplift_bin_count,
                uplift_training_subject_ids: uplift_training_subject_ids
                    .into_iter()
                    .map(Arc::<str>::from)
                    .collect::<Vec<_>>()
                    .into(),
                regret: regret_selected_index.map(|selected_index| {
                    antecedent_core::FixedCandidateRegretInputs {
                        candidates: regret_candidates
                            .into_iter()
                            .map(Arc::<[bool]>::from)
                            .collect::<Vec<_>>()
                            .into(),
                        selected_index,
                        training_subject_ids: regret_training_subject_ids
                            .into_iter()
                            .map(Arc::<str>::from)
                            .collect::<Vec<_>>()
                            .into(),
                    }
                }),
            };
            query
                .validate()
                .map_err(|e| py_err(antecedent::CausalError::rejected(e)))?;
            let _ = accepted;
            let builder = Study::tabular(data).query(CausalQuery::PolicyValue(query));
            req.finish_applied(builder, false)
        })
    }

    /// Freeze a fixed multi-action policy under known randomized probabilities.
    #[staticmethod]
    #[pyo3(signature = (names, columns, outcome, action_labels, assignment, propensities,
        actions, reference, costs, reference_costs, available, capacities,
        reference_capacities, budget, reference_budget, evaluation_subject_ids, *,
        cate_groups=vec![],
        accepted=false, seed=1, threads=None, options=None))]
    #[allow(clippy::too_many_arguments)]
    fn prepare_multi_action_policy_value(
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        outcome: String,
        action_labels: Vec<String>,
        assignment: Vec<usize>,
        propensities: Vec<f64>,
        actions: Vec<usize>,
        reference: Vec<usize>,
        costs: Vec<f64>,
        reference_costs: Vec<f64>,
        available: Vec<bool>,
        capacities: Vec<usize>,
        reference_capacities: Vec<usize>,
        budget: Option<f64>,
        reference_budget: Option<f64>,
        evaluation_subject_ids: Vec<String>,
        cate_groups: Vec<String>,
        accepted: bool,
        seed: u64,
        threads: Option<u32>,
        options: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let req = PrepareRequest::parse(names, options.as_ref(), seed, threads)?;
        req.refuse_design("multi-action policy value")?;
        let data = req.load_columns(py, columns)?;
        req.off_gil(py, move |req| {
            let outcome_id = crate::graph_build::schema_var_id(data.schema(), &outcome)?;
            let multi_action = antecedent_core::MultiActionPolicyInputs {
                action_labels: action_labels
                    .into_iter()
                    .map(Arc::<str>::from)
                    .collect::<Vec<_>>()
                    .into(),
                assignment: assignment.into(),
                actions: actions.into(),
                reference: reference.into(),
                propensities: propensities.into(),
                available: available.into(),
                costs: costs.into(),
                reference_costs: reference_costs.into(),
                capacities: capacities.into(),
                reference_capacities: reference_capacities.into(),
                budget,
                reference_budget,
                cate_groups: cate_groups
                    .into_iter()
                    .map(Arc::<str>::from)
                    .collect::<Vec<_>>()
                    .into(),
            };
            let query = antecedent_core::PolicyValueQuery {
                outcome: outcome_id,
                assignment: Vec::new().into(),
                propensity: Vec::new().into(),
                actions: Vec::new().into(),
                reference: Vec::new().into(),
                mu0: Vec::new().into(),
                mu1: Vec::new().into(),
                costs: Vec::new().into(),
                reference_costs: Vec::new().into(),
                evaluation_subject_ids: evaluation_subject_ids
                    .into_iter()
                    .map(Arc::<str>::from)
                    .collect::<Vec<_>>()
                    .into(),
                disjoint_training_subjects: false,
                crossfit_fold_ownership_valid: false,
                global_constraints_present: false,
                multi_action: Some(multi_action),
                uplift_bins: Arc::from([]),
                uplift_bin_count: 0,
                uplift_training_subject_ids: Arc::from([]),
                regret: None,
            };
            query.validate().map_err(|e| py_err(antecedent::CausalError::rejected(e)))?;
            let _ = accepted;
            let builder = Study::tabular(data).query(CausalQuery::PolicyValue(query));
            req.finish_applied(builder, false)
        })
    }
}
