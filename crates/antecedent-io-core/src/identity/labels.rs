//! Human-readable executed-functional labels.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use serde::Serialize;

use crate::query_wire::{
    CausalQueryWire, InterventionWire, OutcomeFunctionalWire, TargetPopulationWire, ValueWire,
};

/// Host/Python view of the executed functional: human-readable label pairs for
/// inspection reports.
///
/// The query kind and estimand tags are stable, and every value label has an explicit
/// rendering (comma-joined lists, `none` for an absent option, the wire's JSON for an
/// expression), never a derived `Debug`. Reals render with `to_string()`, so the labels
/// are display text. They feed no digest and no equality check; identity is carried by
/// [`digest_wire`](super::digest_wire) over the typed wire, which encodes reals by bit pattern. Do not
/// compare or hash these labels.
///
/// Every query kind reports through this owner so consume and prepare do not
/// re-decide science in the adapter.
#[must_use]
pub fn executed_functional_labels(query: &CausalQueryWire) -> Vec<(String, String)> {
    match query {
        CausalQueryWire::AverageEffect {
            treatment,
            outcome,
            control,
            active,
            target_population,
            outcome_functional,
            ..
        } => contrast_labels(ContrastLabels {
            kind: "average_effect",
            treatment: *treatment,
            outcome: *outcome,
            control,
            active,
            population: target_population,
            functional: Some(outcome_functional),
            temporal: "none".into(),
        }),
        CausalQueryWire::TemporalEffect {
            treatment,
            outcome,
            control,
            active,
            horizon_steps,
            target_population,
            ..
        } => contrast_labels(ContrastLabels {
            kind: "temporal_effect",
            treatment: *treatment,
            outcome: *outcome,
            control,
            active,
            population: target_population,
            functional: None,
            temporal: format!("horizon:{horizon_steps}"),
        }),
        CausalQueryWire::ConditionalEffect { inner } => {
            let mut labels = executed_functional_labels(inner);
            labels.retain(|(key, _)| key != "query_kind");
            let mut out = vec![("query_kind".into(), "conditional_effect".into())];
            out.extend(labels);
            out
        }
        CausalQueryWire::Mediation {
            treatment,
            outcome,
            control,
            active,
            target_population,
            horizons,
            ..
        } => contrast_labels(ContrastLabels {
            kind: "mediation",
            treatment: *treatment,
            outcome: *outcome,
            control,
            active,
            population: target_population,
            functional: None,
            temporal: horizons_label(horizons),
        }),
        CausalQueryWire::NestedCounterfactual {
            treatment,
            mediator,
            outcome,
            control_bits,
            active_bits,
        } => vec![
            ("query_kind".into(), "nested_counterfactual".into()),
            ("estimand".into(), "natural_direct_shared_exogenous".into()),
            ("treatment".into(), treatment.to_string()),
            ("mediator".into(), mediator.to_string()),
            ("outcome".into(), outcome.to_string()),
            ("control".into(), f64::from_bits(*control_bits).to_string()),
            ("active".into(), f64::from_bits(*active_bits).to_string()),
            ("temporal_coordinates".into(), "none".into()),
        ],
        CausalQueryWire::PathSpecific(path) => contrast_labels(ContrastLabels {
            kind: "path_specific",
            treatment: path.treatment,
            outcome: path.outcome,
            control: &path.control,
            active: &path.active,
            population: &path.target_population,
            functional: None,
            temporal: "none".into(),
        }),
        other => other_query_labels(other),
    }
}

struct ContrastLabels<'a> {
    kind: &'static str,
    treatment: u32,
    outcome: u32,
    control: &'a InterventionWire,
    active: &'a InterventionWire,
    population: &'a TargetPopulationWire,
    functional: Option<&'a OutcomeFunctionalWire>,
    temporal: String,
}

fn contrast_labels(contrast: ContrastLabels<'_>) -> Vec<(String, String)> {
    let mut out = vec![
        ("query_kind".into(), contrast.kind.into()),
        ("treatment".into(), contrast.treatment.to_string()),
        ("outcome".into(), contrast.outcome.to_string()),
        ("control".into(), intervention_label(contrast.control)),
        ("active".into(), intervention_label(contrast.active)),
        ("population".into(), population_label(contrast.population)),
        ("temporal_coordinates".into(), contrast.temporal),
    ];
    if let Some(functional) = contrast.functional {
        out.push(("outcome_functional".into(), outcome_functional_label(functional)));
    }
    out
}

// allow(too_many_lines): flat per-variant label mapping; splitting would obscure the query taxonomy
#[allow(
    clippy::too_many_lines,
    reason = "flat field-by-field wire check or conversion; the arms are independent and splitting would only scatter the list"
)]
fn other_query_labels(query: &CausalQueryWire) -> Vec<(String, String)> {
    match query {
        CausalQueryWire::Counterfactual { outcomes, interventions, .. } => vec![
            ("query_kind".into(), "counterfactual".into()),
            ("outcome".into(), join_ids(outcomes)),
            (
                "active".into(),
                interventions.iter().map(intervention_label).collect::<Vec<_>>().join(";"),
            ),
            ("temporal_coordinates".into(), "none".into()),
        ],
        CausalQueryWire::Distribution(distribution) => vec![
            ("query_kind".into(), "distribution".into()),
            ("outcome".into(), join_ids(&distribution.outcomes)),
            ("population".into(), population_label(&distribution.target_population)),
            ("temporal_coordinates".into(), "none".into()),
        ],
        CausalQueryWire::Response(response) => vec![
            ("query_kind".into(), "response".into()),
            ("population".into(), population_label(&response.target_population)),
            ("outcome_functional".into(), outcome_functional_label(&response.outcome_functional)),
            (
                "temporal_coordinates".into(),
                response
                    .temporal
                    .as_ref()
                    .map_or_else(|| "none".into(), |spec| horizons_label(&spec.horizons)),
            ),
        ],
        CausalQueryWire::Transport(transport) => vec![
            ("query_kind".into(), "transport".into()),
            (
                "population".into(),
                format!(
                    "source:{}->target:{}",
                    transport.source_population, transport.target_population
                ),
            ),
            ("temporal_coordinates".into(), "none".into()),
        ],
        CausalQueryWire::Interference(_) => vec![
            ("query_kind".into(), "interference".into()),
            ("temporal_coordinates".into(), "none".into()),
        ],
        CausalQueryWire::RandomizedEffect(query) => vec![
            ("query_kind".into(), "randomized_effect".into()),
            (
                "randomized_estimand".into(),
                match query.estimand {
                    crate::RandomizedEstimandWire::Itt => "itt".into(),
                    crate::RandomizedEstimandWire::CaceLate => "cace_late".into(),
                    crate::RandomizedEstimandWire::TreatmentOnTreated => {
                        "treatment_on_treated".into()
                    }
                },
            ),
            ("exact_randomization_test".into(), query.exact_randomization_test.to_string()),
            ("outcome".into(), query.outcome.to_string()),
            (
                "randomization_design".into(),
                match &query.design {
                    crate::RandomizationDesignWire::Bernoulli => "bernoulli".into(),
                    crate::RandomizationDesignWire::Switchback => "switchback".into(),
                    crate::RandomizationDesignWire::Complete { .. } => "complete".into(),
                    crate::RandomizationDesignWire::Cluster { .. } => "cluster".into(),
                    crate::RandomizationDesignWire::Stratified => "stratified".into(),
                    crate::RandomizationDesignWire::Factorial2x2 => "factorial_2x2".into(),
                    crate::RandomizationDesignWire::MultiArm => "multi_arm".into(),
                },
            ),
            (
                "treatment_arms".into(),
                format!("{}->{}", query.treatment_arms.0, query.treatment_arms.1),
            ),
            (
                "precision_adjustment".into(),
                if query.ancova_covariates.is_empty() {
                    query.fixed_cuped.map_or_else(
                        || "none".into(),
                        |(id, coefficient)| format!("fixed_cuped:{id}:{coefficient}"),
                    )
                } else {
                    format!("ancova:{}", join_ids(&query.ancova_covariates))
                },
            ),
            ("switchback_periods".into(), query.periods.join(",")),
            (
                "factorial_second_assignment".into(),
                query
                    .second_factor_assignment
                    .iter()
                    .map(|value| if *value { '1' } else { '0' })
                    .collect(),
            ),
            (
                "factorial_cell_counts".into(),
                query.factorial_cell_counts.map_or_else(
                    || "none".into(),
                    |counts| counts.iter().map(ToString::to_string).collect::<Vec<_>>().join(","),
                ),
            ),
            (
                "factorial_second_arms".into(),
                query
                    .second_factor_arms
                    .as_ref()
                    .map_or_else(|| "none".into(), |(first, second)| format!("{first}->{second}")),
            ),
            ("temporal_coordinates".into(), "none".into()),
        ],
        CausalQueryWire::PanelDid(query) => {
            // Same precedence as the wire decoder's design selection.
            let design = if let Some(rollout) = &query.rollout {
                rollout.design.as_str()
            } else if query.augmented.is_some() {
                "augmented_panel"
            } else if query.staggered_event_study {
                "staggered_event_study"
            } else if query.staggered_target.is_some() {
                "staggered_group_time"
            } else if query.repeated_cross_section {
                "repeated_cross_section"
            } else {
                "balanced_two_period_panel"
            };
            let mut dimensions = vec![
                ("query_kind".into(), "panel_did".into()),
                ("outcome".into(), query.outcome.to_string()),
                (
                    "augmented_nuisance".into(),
                    query.augmented.map_or_else(
                        || "none".into(),
                        |(first, second, third, flag)| format!("{first}:{second}:{third}:{flag}"),
                    ),
                ),
                ("design".into(), design.into()),
                ("temporal_coordinates".into(), "pre_post".into()),
            ];
            if let Some((cohort, period)) = query.staggered_target {
                dimensions
                    .push(("staggered_target".into(), format!("cohort={cohort},period={period}")));
            }
            if let Some(rollout) = &query.rollout {
                dimensions.push((
                    "rollout_estimator".into(),
                    format!(
                        "{}:{}:{}:{}:{}:anticipation={}:covariates={}",
                        rollout.estimator,
                        rollout.comparison,
                        rollout.nuisance,
                        rollout.base_period,
                        rollout.aggregation,
                        rollout.anticipation,
                        join_ids(&rollout.covariates)
                    ),
                ));
            }
            if let Some(resampling) = &query.resampling {
                dimensions.push((
                    "resampling".into(),
                    format!("{}:{}", resampling.scheme, resampling.replicates),
                ));
            }
            dimensions
        }
        CausalQueryWire::SyntheticControl(query) => {
            let mut dimensions = vec![
                (
                    "query_kind".into(),
                    if query.difference_in_differences {
                        "synthetic_did"
                    } else {
                        "synthetic_control"
                    }
                    .into(),
                ),
                ("outcome".into(), query.outcome.to_string()),
                ("treated_unit".into(), query.treated_unit.clone()),
                ("intervention_period".into(), query.intervention_period.to_string()),
                (
                    "donor_pool_size".into(),
                    query
                        .units
                        .iter()
                        .collect::<std::collections::BTreeSet<_>>()
                        .len()
                        .saturating_sub(1)
                        .to_string(),
                ),
            ];
            if query.uniform_unit_randomization {
                dimensions.push(("uniform_unit_randomization".into(), "true".into()));
            }
            if let Some(effect) = query.sharp_null_effect {
                dimensions.push(("sharp_null_effect".into(), effect.to_string()));
            }
            if let Some(ridge) = query.augmentation_ridge {
                dimensions.push(("augmentation_ridge".into(), ridge.to_string()));
            }
            if query.observed_window_support {
                dimensions.push(("observed_window_support".into(), "true".into()));
            }
            if let Some(rollout) = &query.rollout {
                dimensions.push((
                    "synthetic_interval".into(),
                    format!(
                        "{}:{}:adoptions={}",
                        rollout.interval,
                        rollout.replicates,
                        rollout
                            .adoptions
                            .iter()
                            .map(|(unit, period)| format!("{unit}={period}"))
                            .collect::<Vec<_>>()
                            .join(",")
                    ),
                ));
            }
            dimensions
        }
        CausalQueryWire::LocalPolynomialRatio(query) => {
            let mut dimensions: Vec<(String, String)> = vec![
                (
                    "query_kind".into(),
                    if query.kink { "regression_kink" } else { "fuzzy_rd" }.into(),
                ),
                ("outcome".into(), query.outcome.to_string()),
                ("treatment".into(), query.treatment.to_string()),
                ("running".into(), query.running.to_string()),
                ("cutoff".into(), query.cutoff.to_string()),
                ("bandwidth".into(), query.bandwidth.to_string()),
            ];
            if let Some(options) = &query.options {
                dimensions.push((
                    "local_ratio_options".into(),
                    format!(
                        "{}:temporal={}:density={}:balance={}",
                        options.bandwidth_rule,
                        options
                            .temporal_replicates
                            .map_or_else(|| "none".into(), |replicates| replicates.to_string()),
                        options.density_test,
                        join_ids(&options.balance_covariates)
                    ),
                ));
            }
            dimensions
        }
        CausalQueryWire::PolicyValue(query) => {
            let mut dimensions = vec![
                ("query_kind".into(), "policy_value".into()),
                ("outcome".into(), query.outcome.to_string()),
                (
                    "policy_actions".into(),
                    query
                        .multi_action
                        .as_ref()
                        .map_or(query.actions.len(), |multi| multi.actions.len())
                        .to_string(),
                ),
                ("temporal_coordinates".into(), "none".into()),
            ];
            if let Some(multi) = &query.multi_action {
                dimensions.push(("action_count".into(), multi.action_labels.len().to_string()));
            }
            dimensions
        }
        CausalQueryWire::ContinuousDoseResponse(query) => vec![
            ("query_kind".into(), "continuous_dose_response".into()),
            ("outcome".into(), query.outcome.to_string()),
            ("dose".into(), query.dose.to_string()),
            ("dose_density".into(), query.dose_density.to_string()),
            ("group_count".into(), query.baseline_groups.len().to_string()),
            (
                "target_doses".into(),
                query.target_doses.iter().map(ToString::to_string).collect::<Vec<_>>().join(","),
            ),
            ("bandwidth".into(), query.bandwidth.to_string()),
            ("density_provenance".into(), query.density_provenance.clone()),
        ],
        CausalQueryWire::Survival(query) => vec![
            ("query_kind".into(), "survival".into()),
            ("duration".into(), query.duration.to_string()),
            ("event".into(), query.event.to_string()),
            ("treatment".into(), query.treatment.to_string()),
            ("tau".into(), query.tau.to_string()),
            (
                "target_cause".into(),
                query.target_cause.map_or_else(|| "none".into(), |cause| cause.to_string()),
            ),
            (
                "delayed_entry".into(),
                query.delayed_entry.map_or_else(|| "none".into(), |entry| entry.to_string()),
            ),
            ("observation".into(), "independent_marginal".into()),
            ("temporal_coordinates".into(), "event_time".into()),
        ],
        CausalQueryWire::LongitudinalRegime(query) => vec![
            ("query_kind".into(), "longitudinal_regime".into()),
            ("method".into(), query.method.clone()),
            ("outcome".into(), query.outcome.to_string()),
            ("periods".into(), query.periods.to_string()),
            ("subjects".into(), query.subject_ids.len().to_string()),
            ("temporal_coordinates".into(), "subject_history".into()),
        ],
        CausalQueryWire::HistoryMatch(query) => vec![
            ("query_kind".into(), "history_match".into()),
            ("outcome".into(), query.outcome.to_string()),
            ("treated".into(), query.treated.to_string()),
            ("history".into(), join_ids(&query.history)),
            ("period".into(), query.period.map_or_else(|| "none".into(), |id| id.to_string())),
            ("weight".into(), query.weight.map_or_else(|| "none".into(), |id| id.to_string())),
            ("caliper".into(), query.caliper.map_or_else(|| "none".into(), |c| c.to_string())),
            ("temporal_coordinates".into(), "history_period".into()),
        ],
        CausalQueryWire::IvBounds(query) => vec![
            ("query_kind".into(), "iv_bounds".into()),
            ("outcome".into(), query.outcome.to_string()),
            ("treatment".into(), query.treatment.to_string()),
            ("instrument".into(), query.instrument.to_string()),
            ("weight".into(), query.weight.map_or_else(|| "none".into(), |id| id.to_string())),
            ("temporal_coordinates".into(), "cross_section".into()),
        ],
        CausalQueryWire::Association(query) => vec![
            ("query_kind".into(), "association".into()),
            ("outcome".into(), query.outcome.to_string()),
            ("predictors".into(), join_ids(&query.predictors)),
            ("conditioning".into(), join_ids(&query.conditioning)),
            (
                "functional".into(),
                match &query.functional {
                    crate::AssociationFunctionalWire::Auto => "auto".to_owned(),
                    crate::AssociationFunctionalWire::Conditional => "conditional".to_owned(),
                    crate::AssociationFunctionalWire::Coefficients => "coefficients".to_owned(),
                    crate::AssociationFunctionalWire::Measure { measure } => measure.clone(),
                    crate::AssociationFunctionalWire::Predictive { folds } => {
                        format!("predictive:{folds}")
                    }
                },
            ),
            (
                "outcome_scale".into(),
                query.outcome_scale.clone().unwrap_or_else(|| "schema".into()),
            ),
            ("weight".into(), query.weight.map_or_else(|| "none".into(), |id| id.to_string())),
            ("refused_after".into(), query.refused_after.clone().unwrap_or_else(|| "none".into())),
            ("temporal_coordinates".into(), "observed_law".into()),
        ],
        CausalQueryWire::SequentialContrast(query) => vec![
            ("query_kind".into(), "sequential_contrast".into()),
            ("outcome".into(), query.outcome.to_string()),
            ("first_treatment".into(), query.first_treatment.to_string()),
            ("second_treatment".into(), query.second_treatment.to_string()),
            ("baseline".into(), join_ids(&query.baseline)),
            ("intermediate".into(), join_ids(&query.intermediate)),
            ("active".into(), format!("{}{}", query.active[0], query.active[1])),
            ("control".into(), format!("{}{}", query.control[0], query.control[1])),
            ("subjects".into(), query.fold_of_row.len().to_string()),
            ("temporal_coordinates".into(), "two_decisions".into()),
        ],
        CausalQueryWire::AnomalyAttribution { targets, .. } => {
            named_outcomes("anomaly_attribution", targets)
        }
        CausalQueryWire::ChangeAttribution { outcome, .. } => {
            named_outcomes("change_attribution", &[*outcome])
        }
        CausalQueryWire::UnitChange { outcome, .. } => named_outcomes("unit_change", &[*outcome]),
        CausalQueryWire::MechanismChange { targets, .. } => {
            named_outcomes("mechanism_change", targets)
        }
        CausalQueryWire::Discover(query) => vec![
            ("query_kind".into(), "discover".into()),
            ("variables".into(), join_ids(&query.variables)),
            ("method".into(), json_label(&query.method)),
            (
                "temporal_coordinates".into(),
                if query.max_lag == 0 {
                    "none".into()
                } else {
                    format!("max_lag:{}", query.max_lag)
                },
            ),
        ],
        CausalQueryWire::Attribute(query) => {
            let (kind, outcomes) = match &query.target {
                crate::AttributeTargetWire::IntrinsicInfluence { target, .. } => {
                    ("intrinsic_influence", vec![*target])
                }
                crate::AttributeTargetWire::RootCauses { .. } => ("root_causes", Vec::new()),
                crate::AttributeTargetWire::DistributionChange { .. } => {
                    ("distribution_change", Vec::new())
                }
                crate::AttributeTargetWire::Paths { outcome, .. } => ("paths", vec![*outcome]),
                crate::AttributeTargetWire::FeatureRelevance { outcome, .. } => {
                    ("feature_relevance", vec![*outcome])
                }
                crate::AttributeTargetWire::Anomaly { .. } => ("anomaly", Vec::new()),
                crate::AttributeTargetWire::UnitChange { .. } => ("unit_change", Vec::new()),
                crate::AttributeTargetWire::MechanismChange { .. } => {
                    ("mechanism_change", Vec::new())
                }
                crate::AttributeTargetWire::StructureChange { .. } => {
                    ("structure_change", Vec::new())
                }
            };
            let mut dimensions = named_outcomes("attribute", &outcomes);
            dimensions.push(("attribution".into(), kind.into()));
            if let Some(embedded) = query.embedded() {
                dimensions.extend(
                    executed_functional_labels(embedded)
                        .into_iter()
                        .map(|(key, value)| (format!("embedded.{key}"), value)),
                );
            }
            dimensions
        }
        CausalQueryWire::Simulate(query) => {
            let (kind, outcomes) = match &query.target {
                crate::SimulationTargetWire::InterventionalSample { draws, .. } => {
                    (format!("interventional_sample:{draws}"), Vec::new())
                }
                crate::SimulationTargetWire::PolicyValue { outcome, draws, .. } => {
                    (format!("policy_value:{draws}"), vec![*outcome])
                }
                crate::SimulationTargetWire::UnitEffects { outcome, .. } => {
                    ("unit_effects".to_owned(), vec![*outcome])
                }
                crate::SimulationTargetWire::NoiseSample { draws, .. } => {
                    (format!("noise_sample:{draws}"), Vec::new())
                }
                crate::SimulationTargetWire::NoiseReplay { n_rows, .. } => {
                    (format!("noise_replay:{n_rows}"), Vec::new())
                }
                crate::SimulationTargetWire::Counterfactual { outcomes, .. } => {
                    ("counterfactual".to_owned(), outcomes.clone())
                }
                crate::SimulationTargetWire::TemporalRollout { n_rows, n_steps, .. } => {
                    (format!("temporal_rollout:{n_rows}x{n_steps}"), Vec::new())
                }
                crate::SimulationTargetWire::TemporalCounterfactual { n_rows, n_steps, .. } => {
                    (format!("temporal_counterfactual:{n_rows}x{n_steps}"), Vec::new())
                }
                crate::SimulationTargetWire::InProcess { target, digest, .. } => {
                    (format!("{target}:in_process:{digest}"), Vec::new())
                }
            };
            let mut dimensions = named_outcomes("simulate", &outcomes);
            dimensions.push(("simulation".into(), kind));
            dimensions
        }
        CausalQueryWire::TemporalDesign(query) => vec![
            ("query_kind".into(), "temporal_design".into()),
            ("outcome".into(), query.outcome.to_string()),
            ("design".into(), query.design_name().into()),
            ("lag".into(), query.lag.to_string()),
            (
                "functional".into(),
                query
                    .quantile_level
                    .map_or_else(|| "mean".into(), |level| format!("quantile:{level}")),
            ),
            ("rows".into(), query.units.len().to_string()),
            ("temporal_coordinates".into(), "unit_period".into()),
        ],
        // Contrast queries are labeled by `executed_functional_labels` before this
        // function is called. Matching them here keeps the match exhaustive, so a
        // new query kind is a compile error instead of a runtime panic.
        CausalQueryWire::AverageEffect { .. }
        | CausalQueryWire::TemporalEffect { .. }
        | CausalQueryWire::ConditionalEffect { .. }
        | CausalQueryWire::Mediation { .. }
        | CausalQueryWire::NestedCounterfactual { .. }
        | CausalQueryWire::PathSpecific(_) => Vec::new(),
    }
}

fn named_outcomes(kind: &'static str, outcomes: &[u32]) -> Vec<(String, String)> {
    vec![
        ("query_kind".into(), kind.into()),
        ("outcome".into(), join_ids(outcomes)),
        ("temporal_coordinates".into(), "none".into()),
    ]
}

/// The JSON rendering of a wire value, for labels of values with no shorter stable spelling
/// (an expression tree, a method with parameters). Field and variant names are the wire's.
fn json_label<T: Serialize>(value: &T) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "unrenderable".into())
}

fn join_ids(ids: &[u32]) -> String {
    ids.iter().map(ToString::to_string).collect::<Vec<_>>().join(",")
}

fn horizons_label(horizons: &[u32]) -> String {
    if horizons.is_empty() {
        "none".into()
    } else {
        format!(
            "horizons:{}",
            horizons.iter().map(ToString::to_string).collect::<Vec<_>>().join(",")
        )
    }
}

fn population_label(population: &TargetPopulationWire) -> String {
    match population {
        TargetPopulationWire::AllObserved => "all_observed".into(),
        TargetPopulationWire::Treated => "treated".into(),
        TargetPopulationWire::Untreated => "untreated".into(),
        TargetPopulationWire::Environment(id) => format!("environment:{id}"),
        TargetPopulationWire::PredicateNamed { name, .. } => format!("predicate:{name}"),
        TargetPopulationWire::PredicateRows(rows) => {
            format!("rows:{}", rows.iter().map(ToString::to_string).collect::<Vec<_>>().join(","))
        }
        TargetPopulationWire::CustomDistribution { handle, .. } => {
            format!("custom_distribution:{handle}")
        }
        TargetPopulationWire::RowWeights { .. } => "row_weights".into(),
        TargetPopulationWire::LocalAtCutoff { running, cutoff } => {
            format!("local_at_cutoff:{running}:{cutoff}")
        }
    }
}

fn outcome_functional_label(functional: &OutcomeFunctionalWire) -> String {
    match functional {
        OutcomeFunctionalWire::Mean => "mean".into(),
        OutcomeFunctionalWire::Exceedance(threshold) => format!("exceedance:{threshold}"),
        OutcomeFunctionalWire::ExceedanceGrid(grid) => format!(
            "exceedance_grid:{}",
            grid.iter().map(ToString::to_string).collect::<Vec<_>>().join(",")
        ),
        OutcomeFunctionalWire::Quantile(tau) => format!("quantile:{tau}"),
    }
}

pub(super) fn intervention_label(intervention: &InterventionWire) -> String {
    match intervention {
        InterventionWire::Set { variable, value } => {
            format!("set:{variable}={}", value_label(value))
        }
        InterventionWire::Shift { variable, delta } => {
            format!("shift:{variable}={}", value_label(delta))
        }
        InterventionWire::Stochastic { variable, .. } => format!("stochastic:{variable}"),
        InterventionWire::Soft { variable, .. } => format!("soft:{variable}"),
        // A program enters identity as its full tree and declared reads under its kind tag,
        // so two different programs (or a transform and an assignment of one expression)
        // never share a label. Released variants above are unchanged.
        InterventionWire::Transform { variable, expr, reads } => {
            format!("transform:{variable}:{}:{}", json_label(expr), json_label(reads))
        }
        InterventionWire::Assign { variable, expr, reads } => {
            format!("assign:{variable}:{}:{}", json_label(expr), json_label(reads))
        }
        InterventionWire::Sequence { steps } => format!(
            "sequence:{}",
            steps
                .iter()
                .map(|step| intervention_label(&step.intervention))
                .collect::<Vec<_>>()
                .join(",")
        ),
    }
}

fn value_label(value: &ValueWire) -> String {
    match value {
        ValueWire::Float64(value) => value.to_string(),
        ValueWire::Int64(value) => value.to_string(),
        ValueWire::Bool(value) => value.to_string(),
        ValueWire::Category(value) => format!("cat:{value}"),
        ValueWire::Label(value) => value.clone(),
    }
}
