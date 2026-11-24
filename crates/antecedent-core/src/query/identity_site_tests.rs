//! Identity sites of the query encodings (`formal/identity_sites.toml`, `core.query.*`),
//! checked with the category laws of [`crate::identity_site_laws`] over the fixtures of the
//! encoding tests. Each test names the site it checks.
//!
//! Variable-set fields that the inventory classes as `multiset` are sets in meaning whose
//! repeats are refused by the estimators that read them (repeated effect modifiers, mediators)
//! or read as given; the encoding keeps a repeated declaration apart from the deduplicated one,
//! so an identity never aliases a declaration a consumer refuses to one it accepts.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use super::*;
use crate::identity_site_laws::{digest, injective, multiset_law, ordered_law, set_law};

/// Three variables no fixture uses for anything else.
fn trio() -> [VariableId; 3] {
    [v(5), v(6), v(7)]
}

fn ids(items: &[VariableId]) -> Arc<[VariableId]> {
    Arc::from(items)
}

#[test]
fn site_average_effect_modifiers() {
    multiset_law(&trio(), |items| Some(digest(&average().with_effect_modifiers(items.to_vec()))));
}

#[test]
fn site_temporal_effect_modifiers() {
    multiset_law(&trio(), |items| Some(digest(&temporal().with_effect_modifiers(items.to_vec()))));
}

#[test]
fn site_mediation_mediators() {
    multiset_law(&trio(), |items| {
        let mut query = mediation();
        query.mediators = ids(items);
        Some(digest(&query))
    });
}

#[test]
fn site_distribution_conditioning_and_outcomes() {
    multiset_law(&trio(), |items| {
        let mut query = distribution();
        query.conditioning = ids(items);
        Some(digest(&query))
    });
    ordered_law(&trio(), |items| {
        let mut query = distribution();
        query.outcomes = ids(items);
        Some(digest(&query))
    });
}

#[test]
fn site_path_specific_path_nodes() {
    multiset_law(&trio(), |items| Some(digest(&path_specific().with_path_nodes(items.to_vec()))));
}

fn assumptions() -> [ObservationAssumption; 3] {
    [
        ObservationAssumption::Structural(Arc::from("mar")),
        ObservationAssumption::Structural(Arc::from("mnar")),
        ObservationAssumption::IndependentGiven(Arc::from([v(5)])),
    ]
}

#[test]
fn site_response_observation_assumptions() {
    multiset_law(&assumptions(), |items| {
        Some(digest(&response().with_observation(ObservationSpec::Complete, items.to_vec())))
    });
}

#[test]
fn site_observation_assumption_conditioning() {
    multiset_law(&trio(), |items| {
        Some(digest(&ObservationAssumption::IndependentGiven(ids(items))))
    });
    multiset_law(&trio(), |items| {
        Some(digest(&ObservationAssumption::OutcomeIndependentGiven(ids(items))))
    });
}

#[test]
fn site_response_functional_grid() {
    ordered_law(&[0.0, 0.5, 1.0], |values| Some(digest(&mean_curve(values))));
}

#[test]
fn site_transport_source_experiments() {
    multiset_law(&trio(), |items| {
        let mut query = transport();
        query.source_experiments = ids(items);
        Some(digest(&query))
    });
}

#[test]
fn site_transport_weighted_target_covariates() {
    multiset_law(&trio(), |items| {
        let mut target = weighted_target(1);
        target.covariates = ids(items);
        Some(digest(&target))
    });
}

#[test]
fn site_distribution_availability_marginals() {
    multiset_law(&trio(), |items| {
        Some(digest(&DistributionAvailability::SeparateMarginals { variables: ids(items) }))
    });
}

#[test]
fn site_sampling_selection_variables() {
    multiset_law(&trio(), |items| {
        Some(digest(&SamplingSelection::SelectedOn { variables: ids(items) }))
    });
}

#[test]
fn site_environment_variables_and_selection_targets() {
    let base = catalog(false).environments[0].clone();
    multiset_law(&trio(), |items| {
        let mut environment = base.clone();
        environment.selection_targets = ids(items);
        Some(digest(&environment))
    });
    let coordinates: Vec<VariableCoordinate> = trio()
        .iter()
        .map(|&variable| VariableCoordinate {
            variable,
            domain: VariableDomain::Binary,
            unit: None,
        })
        .collect();
    multiset_law(&coordinates, |items| {
        let mut environment = base.clone();
        environment.variables = Arc::from(items);
        Some(digest(&environment))
    });
}

#[test]
fn site_evidence_regime_variable_sets_and_values() {
    let base = catalog(false).regimes[0].clone();
    for field in 0..3 {
        multiset_law(&trio(), |items| {
            let mut regime = base.clone();
            match field {
                0 => regime.interventions = ids(items),
                1 => regime.measured = ids(items),
                _ => regime.conditioned_on = ids(items),
            }
            Some(digest(&regime))
        });
    }
    let values: Vec<InterventionAssignment> = trio()
        .iter()
        .map(|&variable| InterventionAssignment { variable, value: Value::f64(1.0) })
        .collect();
    multiset_law(&values, |items| {
        let mut regime = base.clone();
        regime.intervention_values = Arc::from(items);
        Some(digest(&regime))
    });
}

#[test]
fn site_evidence_catalog_encode() {
    let base = catalog(false);
    let environments: Vec<Environment> = base.environments.to_vec();
    multiset_law(&environments, |items| {
        let mut catalog = base.clone();
        catalog.environments = Arc::from(items);
        Some(digest(&catalog))
    });
    let regimes: Vec<EvidenceRegime> = base.regimes.to_vec();
    multiset_law(&regimes, |items| {
        let mut catalog = base.clone();
        catalog.regimes = Arc::from(items);
        Some(digest(&catalog))
    });
}

/// The fixture catalog with every variable its regimes name declared in both environments.
fn valid_catalog() -> EvidenceCatalog {
    let coordinates = |variables: &[u32]| -> Arc<[VariableCoordinate]> {
        variables
            .iter()
            .map(|&id| VariableCoordinate {
                variable: v(id),
                domain: VariableDomain::Continuous,
                unit: None,
            })
            .collect::<Vec<_>>()
            .into()
    };
    let mut catalog = catalog(false);
    catalog.environments = Arc::from([
        Environment {
            identity: Arc::from("source"),
            variables: coordinates(&[0, 1, 2, 3, 4, 5, 6]),
            selection_targets: Arc::from([v(0), v(1)]),
        },
        Environment {
            identity: Arc::from("target"),
            variables: coordinates(&[0]),
            selection_targets: Arc::from([]),
        },
    ]);
    catalog
}

/// `canonicalized` validates first: a repeated environment or regime is refused (the set
/// law's repeat is skipped), every reordering gives the same canonical catalog, and dropping
/// an entry another entry needs is refused.
#[test]
fn site_evidence_catalog_canonicalized() {
    let base = valid_catalog();
    let canonical = |catalog: &EvidenceCatalog| catalog.canonicalized().ok().map(|c| digest(&c));
    let environments: Vec<Environment> = base.environments.to_vec();
    set_law(&environments, |items| {
        let mut catalog = base.clone();
        catalog.environments = Arc::from(items);
        canonical(&catalog)
    });
    let regimes: Vec<EvidenceRegime> = base.regimes.to_vec();
    set_law(&regimes, |items| {
        let mut catalog = base.clone();
        catalog.regimes = Arc::from(items);
        canonical(&catalog)
    });
}

#[test]
fn site_catalog_distribution_identity() {
    let base = catalog(false).distributions()[0].clone();
    for field in 0..3 {
        multiset_law(&trio(), |items| {
            let mut distribution = base.clone();
            match field {
                0 => distribution.interventions = ids(items),
                1 => distribution.measured = ids(items),
                _ => distribution.conditioned_on = ids(items),
            }
            Some(distribution.canonical_identity())
        });
    }
    // Framing: text fields containing the descriptor's separators stay apart.
    let texts = ["a", "a|b", "a=b", "a,b", "a|study=b", ""];
    let inputs: Vec<(String, String)> = texts
        .iter()
        .flat_map(|study| {
            texts.iter().map(move |population| ((*study).to_owned(), (*population).to_owned()))
        })
        .collect();
    injective(&inputs, Clone::clone, |(study, population)| {
        let mut distribution = base.clone();
        distribution.study = if study.is_empty() { None } else { Some(Arc::from(study.as_str())) };
        distribution.population = Arc::from(population.as_str());
        Some(distribution.canonical_identity())
    });
}

#[test]
fn site_continuous_dose_group_doses() {
    let entries = [("g", 1.0), ("h", 0.5), ("k", 0.25)];
    multiset_law(&entries, |items| {
        let mut query = continuous_dose();
        query.fixed_policy = Some(FixedGroupDosePolicy {
            policy_doses: dose_rule(items),
            reference_doses: dose_rule(&[("g", 0.5), ("h", 0.5)]),
        });
        Some(digest(&query))
    });
}

#[test]
fn site_association_predictors_and_conditioning() {
    ordered_law(&trio(), |items| {
        let mut query = association();
        query.predictors = ids(items);
        Some(digest(&query))
    });
    ordered_law(&trio(), |items| {
        let mut query = association();
        query.conditioning = ids(items);
        Some(digest(&query))
    });
}

#[test]
fn site_attribute_target_sources_and_through() {
    let paths = |sources: &[VariableId], through: &[VariableId]| AttributeTarget::Paths {
        sources: ids(sources),
        outcome: v(2),
        max_paths: 64,
        max_len: 16,
        through: ids(through),
    };
    ordered_law(&trio(), |items| Some(digest(&paths(items, &[]))));
    ordered_law(&trio(), |items| Some(digest(&paths(&[v(0)], items))));
}

#[test]
fn site_anomaly_attribution_targets() {
    ordered_law(&trio(), |items| Some(digest(&AnomalyAttributionQuery::new(items.to_vec(), 10))));
}

#[test]
fn site_counterfactual_outcomes() {
    ordered_law(&trio(), |items| {
        let mut query = counterfactual();
        query.outcomes = ids(items);
        Some(digest(&query))
    });
}

#[test]
fn site_heterogeneity_robust_did_covariates() {
    ordered_law(&trio(), |items| {
        Some(digest(&HeterogeneityRobustDid {
            covariates: ids(items),
            ..HeterogeneityRobustDid::default()
        }))
    });
}

#[test]
fn site_discover_variables() {
    ordered_law(&trio(), |items| {
        let mut query = discover();
        query.variables = ids(items);
        Some(digest(&query))
    });
}

#[test]
fn site_discovery_method_focus() {
    ordered_law(&trio(), |items| {
        Some(digest(&DiscoveryMethod::ScreenedExactMarginals {
            focus: ids(items),
            max_variables: 8,
            in_probability: 0.5,
            out_probability: 0.1,
            prior: None,
        }))
    });
}

fn prior(edges: &[(VariableId, VariableId, f64)]) -> StructurePriorSpec {
    StructurePriorSpec {
        edge_inclusion: None,
        edges: Arc::from(edges),
        default_edge: None,
        soft_tiers: Arc::from([]),
        soft_tier_log_penalty: None,
    }
}

/// The per-edge map is keyed by `(from, to)` and validation refuses a repeated key, so a
/// repeated entry never validates; the encoding keeps it apart. Entries are sorted by the
/// whole entry, so even two entries for one key encode alike in either listed order.
#[test]
fn site_structure_prior_edges() {
    let edges = [(v(0), v(1), 0.5), (v(1), v(2), 0.25), (v(0), v(2), 0.75)];
    multiset_law(&edges, |items| Some(digest(&prior(items))));
    let one = prior(&[(v(0), v(1), 0.5), (v(0), v(1), 0.25)]);
    let other = prior(&[(v(0), v(1), 0.25), (v(0), v(1), 0.5)]);
    assert_eq!(digest(&one), digest(&other));
    let tiers = |items: &[VariableId]| {
        let mut spec = prior(&[]);
        spec.soft_tiers = Arc::from([ids(items)]);
        spec.soft_tier_log_penalty = Some(1.0);
        Some(digest(&spec))
    };
    ordered_law(&trio(), tiers);
}

#[test]
fn site_history_match_history() {
    ordered_law(&trio(), |items| {
        let mut query = history_match();
        query.history = ids(items);
        Some(digest(&query))
    });
}

#[test]
fn site_local_ratio_balance_covariates() {
    ordered_law(&trio(), |items| {
        Some(digest(&LocalRatioOptions {
            balance_covariates: ids(items),
            ..LocalRatioOptions::default()
        }))
    });
}

fn interventions() -> [Intervention; 3] {
    [set(5, 1.0), set(6, 0.0), set(7, 2.0)]
}

#[test]
fn site_do_sampler_interventions() {
    ordered_law(&interventions(), |items| {
        Some(digest(&DoSamplerTarget {
            method: DoSamplerMethod::Weighting { root_match_tolerance: None },
            outcome: v(1),
            interventions: Arc::from(items),
            draws: 10,
            seed: 1,
        }))
    });
}

#[test]
fn site_nested_counterfactual_target_lists() {
    let target = |outer: &[Intervention], inner: &[Intervention], frozen: &[VariableId]| {
        NestedCounterfactualTarget {
            outer: Arc::from(outer),
            inner: Arc::from(inner),
            frozen: ids(frozen),
            outcome: v(1),
        }
    };
    ordered_law(&trio(), |items| Some(digest(&target(&[set(0, 1.0)], &[set(0, 0.0)], items))));
    ordered_law(&interventions(), |items| Some(digest(&target(items, &[set(0, 0.0)], &[]))));
    ordered_law(&interventions(), |items| Some(digest(&target(&[set(0, 1.0)], items, &[]))));
}

#[test]
fn site_randomized_ancova_covariates() {
    ordered_law(&trio(), |items| {
        let mut query = randomized();
        query.ancova_covariates = ids(items);
        Some(digest(&query))
    });
}

#[test]
fn site_sequential_contrast_baseline_and_intermediate() {
    ordered_law(&trio(), |items| {
        let mut query = sequential();
        query.baseline = ids(items);
        Some(digest(&query))
    });
    ordered_law(&trio(), |items| {
        let mut query = sequential();
        query.intermediate = ids(items);
        Some(digest(&query))
    });
}

#[test]
fn site_simulation_target_interventions() {
    ordered_law(&interventions(), |items| {
        Some(digest(&SimulationTarget::InterventionalSample {
            interventions: Arc::from(items),
            draws: 100,
            seed: 3,
        }))
    });
}

#[test]
fn site_known_censoring_columns() {
    ordered_law(&trio(), |items| {
        Some(digest(&KnownCensoringSurvival {
            times: Arc::from([1.0, 2.0, 3.0]),
            columns: ids(items),
            minimum_probability: 0.01,
        }))
    });
}

#[test]
fn site_target_population_row_weights_depends_on() {
    ordered_law(&trio(), |items| {
        Some(digest(&TargetPopulation::RowWeights { weights: [1; 32], depends_on: ids(items) }))
    });
}

#[test]
fn site_temporal_assignment_confounders_and_history() {
    ordered_law(&trio(), |items| {
        Some(digest(&TemporalAssignment::RegimeSurvival {
            treatment: v(0),
            censored: v(1),
            confounders: ids(items),
            horizon: 2,
        }))
    });
    ordered_law(&trio(), |items| {
        Some(digest(&TemporalAssignment::ObservationalExcursion {
            treatment: v(0),
            history: ids(items),
        }))
    });
}

fn worlds() -> Vec<WorldSpec> {
    vec![WorldSpec::new([], []).unwrap(), WorldSpec::new([(v(0), 1.0)], []).unwrap()]
}

/// The event and the given atoms are sets: construction sorts and deduplicates them.
#[test]
fn site_counterfactual_event_atoms() {
    let world = WorldId::new;
    let atoms = [
        CounterfactualEvent::new(world(1), v(1), 1.0).unwrap(),
        CounterfactualEvent::new(world(0), v(0), 0.0).unwrap(),
        CounterfactualEvent::new(world(1), v(2), 1.0).unwrap(),
    ];
    let build = |event: &[CounterfactualEvent], given: &[CounterfactualEvent]| {
        CounterfactualEventQuery::new(
            worlds(),
            ExogenousCoupling::SharedLatentExogenous,
            event.to_vec(),
            given.to_vec(),
        )
        .ok()
    };
    set_law(&atoms, |items| build(items, &[]));
    set_law(&atoms, |items| build(&atoms[..1], items));
}

/// A world's interventions and routes are sets: construction sorts them and refuses a
/// repeated variable or edge, so a repeat is refused and skipped.
#[test]
fn site_world_spec_interventions() {
    let assignments = [(v(5), 1.0), (v(6), 0.0), (v(7), -0.0)];
    set_law(&assignments, |items| WorldSpec::new(items.to_vec(), []).ok());
    // -0.0 and 0.0 are one level.
    assert_eq!(WorldSpec::new([(v(5), 0.0)], []).ok(), WorldSpec::new([(v(5), -0.0)], []).ok());
}
