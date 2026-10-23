//! Canonical query encoding: every field reaches the bytes, unordered
//! collections do not depend on their listed order, and variants never collide.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use crate::InitialLaw;
use crate::canonical_encode::CanonicalEncode;
use crate::identity::CanonicalWriter;
use crate::ids::{ComponentId, RegimeId, VariableId};
use crate::intervention::Intervention;
use crate::value::Value;

use super::did::AugmentedPanelNuisance;
use super::*;

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn set(variable: u32, level: f64) -> Intervention {
    Intervention::set(v(variable), Value::f64(level))
}

fn labels(names: &[&str]) -> Arc<[Arc<str>]> {
    names.iter().map(|name| Arc::from(*name)).collect::<Vec<Arc<str>>>().into()
}

fn flags(values: &[bool]) -> Arc<[bool]> {
    Arc::from(values)
}

fn digest<T: CanonicalEncode>(value: &T) -> [u8; 32] {
    let mut writer = CanonicalWriter::new("t");
    value.encode(&mut writer);
    writer.finish()
}

fn assert_distinct(queries: &[CausalQuery]) {
    let digests = queries.iter().map(digest).collect::<Vec<_>>();
    for (i, a) in digests.iter().enumerate() {
        for (j, b) in digests.iter().enumerate().skip(i + 1) {
            assert_ne!(a, b, "queries {i} and {j} share an encoding");
        }
    }
}

/// Assert that the base query and each single-field edit of it encode
/// pairwise differently.
fn check<Q: Clone>(wrap: fn(Q) -> CausalQuery, base: &Q, edits: &[&dyn Fn(&mut Q)]) {
    let mut queries = vec![wrap(base.clone())];
    for edit in edits {
        let mut query = base.clone();
        edit(&mut query);
        queries.push(wrap(query));
    }
    assert_distinct(&queries);
}

fn same<Q>(wrap: fn(Q) -> CausalQuery, a: Q, b: Q) {
    assert_eq!(digest(&wrap(a)), digest(&wrap(b)));
}

fn average() -> AverageEffectQuery {
    AverageEffectQuery::binary_ate(v(0), v(1)).with_effect_modifiers([v(2), v(3)])
}

fn temporal() -> TemporalEffectQuery {
    TemporalEffectQuery::pulse(v(0), v(1), 1.0).with_effect_modifiers([v(2), v(3)])
}

fn counterfactual() -> CounterfactualQuery {
    CounterfactualQuery::new(v(1), vec![set(0, 1.0)])
}

fn nested() -> NestedCounterfactualQuery {
    NestedCounterfactualQuery::new(v(0), v(1), v(2)).unwrap()
}

fn anomaly() -> AnomalyAttributionQuery {
    AnomalyAttributionQuery::new(vec![v(0)], 10)
}

fn change() -> ChangeAttributionQuery {
    ChangeAttributionQuery::new(
        v(0),
        PopulationSelector::All,
        PopulationSelector::Rows(Arc::from([0_usize, 1])),
    )
}

fn mechanism() -> MechanismChangeQuery {
    MechanismChangeQuery::new(
        vec![v(0)],
        PopulationSelector::All,
        PopulationSelector::Rows(Arc::from([0_usize, 1])),
        0.05,
        4,
    )
}

fn unit_change() -> UnitChangeQuery {
    UnitChangeQuery::new(v(0), 10)
}

fn mediation() -> MediationQuery {
    MediationQuery::binary(v(0), v(1), [v(2), v(3)], MediationContrast::Total)
}

fn distribution() -> InterventionalDistributionQuery {
    InterventionalDistributionQuery::new(v(1), vec![set(0, 1.0)]).with_conditioning([v(2), v(3)])
}

fn path_specific() -> PathSpecificEffectQuery {
    PathSpecificEffectQuery::binary(v(0), v(1)).with_path_nodes([v(2), v(3)])
}

fn mean_curve(values: &[f64]) -> ResponseFunctional {
    ResponseFunctional::MeanCurve {
        outcome: v(1),
        treatment: ContinuousDomain::new(v(0), GridSpec::Values(Arc::from(values))),
    }
}

fn response() -> ResponseQuery {
    ResponseQuery::new(mean_curve(&[0.0, 1.0]))
}

fn feedback() -> FeedbackDeclaration {
    FeedbackDeclaration {
        delay_ticks: 1,
        level: 1.0,
        autoregression: 0.5,
        input_gain: 1.0,
        innovation_variance: 1.0,
        initial_law: InitialLaw::contemporaneous_observed(),
        initial_state: 0.0,
    }
}

fn temporal_spec(horizons: &[u32]) -> TemporalResponseSpec {
    TemporalResponseSpec::new(horizons.to_vec(), TemporalPolicy::pulse(-1), None).unwrap()
}

fn order<T>(mut items: Vec<T>, reversed: bool) -> Vec<T> {
    if reversed {
        items.reverse();
    }
    items
}

fn catalog(reversed: bool) -> EvidenceCatalog {
    let coordinate = |id: u32, domain: VariableDomain| VariableCoordinate {
        variable: v(id),
        domain,
        unit: None,
    };
    let environments = vec![
        Environment {
            identity: Arc::from("source"),
            variables: order(
                vec![
                    coordinate(0, VariableDomain::Binary),
                    coordinate(1, VariableDomain::Continuous),
                ],
                reversed,
            )
            .into(),
            selection_targets: order(vec![v(0), v(1)], reversed).into(),
        },
        Environment {
            identity: Arc::from("target"),
            variables: Arc::from([coordinate(0, VariableDomain::Binary)]),
            selection_targets: Arc::from([]),
        },
    ];
    let regime = |id: u32, variable: u32, label: &str| EvidenceRegime {
        id: RegimeId::from_raw(id),
        label: Some(Arc::from(label)),
        kind: RegimeKind::Experimental,
        evidence_kind: EvidenceKind::Available,
        interventions: order(vec![v(variable), v(variable + 2)], reversed).into(),
        intervention_values: order(
            vec![
                InterventionAssignment { variable: v(variable), value: Value::f64(1.0) },
                InterventionAssignment { variable: v(variable + 2), value: Value::f64(0.0) },
            ],
            reversed,
        )
        .into(),
        measured: order(vec![v(1), v(3)], reversed).into(),
        conditioned_on: order(vec![v(1), v(3)], reversed).into(),
        population: Arc::from("source"),
        distribution: DistributionAvailability::SeparateMarginals {
            variables: order(vec![v(1), v(3)], reversed).into(),
        },
        study: Some(Arc::from("study")),
        selection: SamplingSelection::SelectedOn {
            variables: order(vec![v(1), v(3)], reversed).into(),
        },
        origin: LawOrigin::Measured,
    };
    let binding = |regime: u32| RegimeBinding {
        dataset_identity: None,
        regime: RegimeId::from_raw(regime),
        snapshot_identity: Arc::from("snap"),
        schema_names: labels(&["x", "y"]),
        sampling: SamplingDesign::Independent,
        weights: Some(LicensedWeights { snapshot_identity: Arc::from("snap") }),
        dependence: DependenceGroup::IndependentStudies,
    };
    EvidenceCatalog {
        environments: order(environments, reversed).into(),
        regimes: order(vec![regime(1, 0, "a"), regime(2, 4, "b")], reversed).into(),
        bindings: order(vec![binding(1), binding(2)], reversed).into(),
        target_sampling: Some(TargetSampling::RepresentativeSample),
    }
}

fn transport() -> TransportQuery {
    TransportQuery::new(response(), "source", "target", vec![v(0), v(2)])
}

fn interference() -> InterferenceQuery {
    InterferenceQuery::new(
        AssignmentDesign::Bernoulli { probabilities: Arc::from([0.5, 0.4]) },
        ExposureMapping::OwnTreatment,
        InterferenceFunctional::ExposureContrast {
            outcome: v(0),
            from: ExposureLevel { own: 0.0, neighbors: 0.0 },
            to: ExposureLevel { own: 1.0, neighbors: 0.0 },
        },
    )
}

fn randomized() -> RandomizedEffectQuery {
    RandomizedEffectQuery::bernoulli_itt(
        v(1),
        vec![true, false],
        vec![0.5, 0.5],
        labels(&["a", "b"]),
        labels(&["a", "b"]),
        ("control", "treated"),
    )
}

fn multi_action() -> MultiActionPolicyInputs {
    MultiActionPolicyInputs {
        action_labels: labels(&["a", "b"]),
        assignment: Arc::from([0_usize, 1]),
        actions: Arc::from([1_usize, 1]),
        reference: Arc::from([0_usize, 0]),
        propensities: Arc::from([0.5, 0.5, 0.5, 0.5]),
        available: Arc::from([true, true, true, true]),
        costs: Arc::from([0.0, 1.0]),
        reference_costs: Arc::from([0.0, 1.0]),
        capacities: Arc::from([2_usize, 2]),
        reference_capacities: Arc::from([2_usize, 2]),
        budget: None,
        reference_budget: None,
        cate_groups: labels(&[]),
    }
}

fn policy_value() -> PolicyValueQuery {
    PolicyValueQuery {
        outcome: v(0),
        assignment: Arc::from([true, false]),
        propensity: Arc::from([0.5]),
        actions: Arc::from([true, false]),
        reference: Arc::from([false, false]),
        mu0: Arc::from([]),
        mu1: Arc::from([]),
        costs: Arc::from([0.0]),
        reference_costs: Arc::from([0.0]),
        evaluation_subject_ids: labels(&["a", "b"]),
        disjoint_training_subjects: true,
        crossfit_fold_ownership_valid: false,
        global_constraints_present: false,
        multi_action: None,
        uplift_bins: Arc::from([]),
        uplift_bin_count: 0,
        uplift_training_subject_ids: labels(&[]),
        regret: None,
    }
}

fn dose_rule(entries: &[(&str, f64)]) -> Arc<[(Arc<str>, f64)]> {
    entries.iter().map(|(group, dose)| (Arc::<str>::from(*group), *dose)).collect::<Vec<_>>().into()
}

fn continuous_dose() -> ContinuousDoseResponseQuery {
    ContinuousDoseResponseQuery {
        outcome: v(0),
        dose: v(1),
        dose_density: v(2),
        baseline_groups: labels(&["g", "h"]),
        target_doses: Arc::from([0.5, 1.0]),
        bandwidth: 0.25,
        min_local_support: 3,
        density_provenance: Arc::from("known"),
        fixed_policy: None,
    }
}

fn panel_did() -> PanelDidQuery {
    PanelDidQuery {
        design: DidSamplingDesign::BalancedPanel,
        outcome: v(0),
        treated: Arc::from([true, false]),
        post: Arc::from([false, true]),
        subjects: labels(&["a", "b"]),
        clusters: labels(&["c", "c"]),
        periods: Arc::from([0_i64, 1]),
        cohorts: Arc::from([1_i64, 0]),
        target: None,
        augmented: None,
        robust: None,
        resampling: None,
    }
}

fn synthetic_control() -> SyntheticControlQuery {
    SyntheticControlQuery {
        outcome: v(0),
        units: labels(&["a", "b"]),
        periods: Arc::from([0_i64, 1]),
        treated_unit: Arc::from("a"),
        intervention_period: 1,
        method: SyntheticPanelMethod::Control,
        uniform_unit_randomization: false,
        sharp_null_effect: None,
        augmentation_ridge: None,
        rollout: None,
    }
}

fn local_ratio() -> LocalPolynomialRatioQuery {
    LocalPolynomialRatioQuery {
        outcome: v(0),
        treatment: v(1),
        running: v(2),
        cutoff: 0.0,
        bandwidth: 1.0,
        kink: false,
        options: None,
    }
}

fn survival() -> SurvivalQuery {
    SurvivalQuery {
        duration: v(0),
        event: v(1),
        treatment: v(2),
        tau: 5.0,
        delayed_entry: None,
        known_censoring: None,
        observation_assumption: ObservationAssumption::IndependentGiven(Arc::from([])),
        functional: SurvivalFunctional::SurvivalAndRmst,
    }
}

fn longitudinal() -> LongitudinalRegimeQuery {
    LongitudinalRegimeQuery {
        outcome: v(0),
        method: LongitudinalRegimeMethod::Ipw,
        period_outcome_predictions: None,
        known_fixed_outcome_predictions: false,
        stabilizing_numerator_probabilities: None,
        q_predictions: None,
        observation_history: None,
        prediction_fold_ids: None,
        periods: 1,
        treatment_history: Arc::from([true, false]),
        regime_actions: Arc::from([true, true]),
        treatment_probabilities: Arc::from([0.5, 0.5]),
        censoring_probabilities: Arc::from([1.0, 1.0]),
        outcome_observed: Arc::from([true, true]),
        subject_ids: labels(&["a", "b"]),
        fold_ids: Arc::from([0_u32, 1]),
        excluded_fold_predictions: false,
        probabilities_known_by_design: true,
        minimum_probability: 0.01,
        rule_id: None,
        rule_version: None,
        rule_provenance: None,
    }
}

fn sequential() -> SequentialContrastQuery {
    SequentialContrastQuery {
        outcome: v(0),
        first_treatment: v(1),
        second_treatment: v(2),
        baseline: Arc::from([v(3)]),
        intermediate: Arc::from([v(4)]),
        active: [1, 1],
        control: [0, 0],
        fold_of_row: Arc::from([0_u32, 1]),
        fold_count: 2,
        min_probability: 0.05,
    }
}

fn history_match() -> HistoryMatchQuery {
    HistoryMatchQuery {
        outcome: v(0),
        treated: v(1),
        history: Arc::from([v(2), v(3)]),
        period: None,
        weight: None,
        caliper: None,
    }
}

#[test]
fn every_average_and_temporal_field_reaches_the_encoding() {
    check(
        CausalQuery::AverageEffect,
        &average(),
        &[
            &|q| q.treatment = v(5),
            &|q| q.outcome = v(6),
            &|q| q.effect_modifiers = Arc::from([v(2)]),
            &|q| q.control = set(0, 0.5),
            &|q| q.active = set(0, 2.0),
            &|q| q.target_population = TargetPopulation::Treated,
            &|q| q.outcome_functional = OutcomeFunctional::exceedance(1.0),
        ],
    );
    check(
        CausalQuery::ConditionalEffect,
        &ConditionalEffectQuery { inner: average() },
        &[&|q| q.inner.outcome = v(6), &|q| q.inner.effect_modifiers = Arc::from([v(2)])],
    );
    check(
        CausalQuery::TemporalEffect,
        &temporal(),
        &[
            &|q| q.treatment = v(5),
            &|q| q.outcome = v(6),
            &|q| q.policy = TemporalPolicy::pulse(1),
            &|q| q.policy = TemporalPolicy::sustained(0, 2),
            &|q| q.policy = TemporalPolicy::dynamic(crate::DynamicRuleId::from_raw(0), [0]),
            &|q| q.control = set(0, 0.5),
            &|q| q.active = set(0, 2.0),
            &|q| q.horizon_steps = 2,
            &|q| q.max_history_lag = Some(3),
            &|q| q.target_population = TargetPopulation::Untreated,
            &|q| q.effect_modifiers = Arc::from([v(2)]),
        ],
    );
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "one linear list of single-field mutations, each checked against the same base query"
)]
fn every_counterfactual_and_attribution_field_reaches_the_encoding() {
    check(
        CausalQuery::Counterfactual,
        &counterfactual(),
        &[
            &|q| q.outcomes = Arc::from([v(1), v(2)]),
            &|q| q.outcomes = Arc::from([v(2), v(1)]),
            &|q| q.interventions = Arc::from([set(0, 2.0)]),
            &|q| q.control = set(0, 0.5),
            &|q| q.allow_nested = true,
        ],
    );
    check(
        CausalQuery::NestedCounterfactual,
        &nested(),
        &[
            &|q| *q = NestedCounterfactualQuery::new(v(3), v(1), v(2)).unwrap(),
            &|q| *q = NestedCounterfactualQuery::new(v(0), v(3), v(2)).unwrap(),
            &|q| *q = NestedCounterfactualQuery::new(v(0), v(1), v(3)).unwrap(),
            &|q| *q = NestedCounterfactualQuery::with_levels(v(0), v(1), v(2), 0.5, 1.0).unwrap(),
            &|q| *q = NestedCounterfactualQuery::with_levels(v(0), v(1), v(2), 0.0, 2.0).unwrap(),
        ],
    );
    check(
        CausalQuery::AnomalyAttribution,
        &anomaly(),
        &[
            &|q| q.targets = Arc::from([v(0), v(1)]),
            &|q| q.unit_rows = Some(Arc::from([0_usize, 1])),
            &|q| q.unit_rows = Some(Arc::from([1_usize, 0])),
            &|q| q.unit_rows = Some(Arc::from([])),
            &|q| q.max_units = 11,
            &|q| q.reference = AnomalyReference::fixed(0.0, 1.0),
            &|q| q.reference = AnomalyReference::fixed(0.0, 2.0),
        ],
    );
    check(
        CausalQuery::ChangeAttribution,
        &change(),
        &[
            &|q| q.outcome = v(5),
            &|q| q.baseline = PopulationSelector::TimeRange { start: 0, end: 5 },
            &|q| q.baseline = PopulationSelector::Environment { env_index: 0 },
            &|q| std::mem::swap(&mut q.baseline, &mut q.comparison),
            &|q| q.comparison = PopulationSelector::Rows(Arc::from([1_usize, 0])),
            &|q| q.components = AttributionComponents::Inputs,
            &|q| q.components = AttributionComponents::All,
            &|q| {
                q.allocation = AllocationMethod::Sequential {
                    order: Arc::from([ComponentId::from_raw(1), ComponentId::from_raw(2)]),
                };
            },
            &|q| {
                q.allocation = AllocationMethod::Sequential {
                    order: Arc::from([ComponentId::from_raw(2), ComponentId::from_raw(1)]),
                };
            },
            &|q| q.allocation = AllocationMethod::PathBased,
            &|q| q.allocation = AllocationMethod::Shapley { approximation: ShapleyConfig::exact() },
            &|q| {
                q.allocation =
                    AllocationMethod::Shapley { approximation: ShapleyConfig::permutation(2_000) };
            },
            &|q| {
                q.allocation = AllocationMethod::Shapley {
                    approximation: ShapleyConfig::monte_carlo(2_000).with_seed(7),
                };
            },
            &|q| {
                q.allocation = AllocationMethod::Shapley {
                    approximation: ShapleyConfig::monte_carlo(2_000).with_max_exact_components(3),
                };
            },
            &|q| {
                q.allocation = AllocationMethod::Shapley {
                    approximation: ShapleyConfig::monte_carlo(2_000).with_exact_override(true),
                };
            },
            &|q| q.max_components = 65,
        ],
    );
    check(
        CausalQuery::MechanismChange,
        &mechanism(),
        &[
            &|q| q.targets = Arc::from([v(0), v(1)]),
            &|q| q.baseline = PopulationSelector::Rows(Arc::from([2_usize])),
            &|q| q.comparison = PopulationSelector::All,
            &|q| q.significance_level = OrderedFloatBits::from_f64(0.01),
            &|q| q.max_targets = 5,
        ],
    );
    check(
        CausalQuery::UnitChange,
        &unit_change(),
        &[
            &|q| q.outcome = v(5),
            &|q| q.unit_rows = Some(Arc::from([0_usize])),
            &|q| q.components = AttributionComponents::Mechanisms,
            &|q| q.allocation = AllocationMethod::PathBased,
            &|q| q.max_units = 11,
        ],
    );
}

#[test]
fn every_mediation_distribution_and_path_field_reaches_the_encoding() {
    check(
        CausalQuery::Mediation,
        &mediation(),
        &[
            &|q| q.treatment = v(5),
            &|q| q.outcome = v(6),
            &|q| q.mediators = Arc::from([v(2)]),
            &|q| q.contrast = MediationContrast::Direct,
            &|q| q.contrast = MediationContrast::Mediated,
            &|q| q.contrast = MediationContrast::NaturalDirect,
            &|q| q.contrast = MediationContrast::NaturalIndirect,
            &|q| q.control = set(0, 0.5),
            &|q| q.active = set(0, 2.0),
            &|q| q.target_population = TargetPopulation::Treated,
            &|q| q.horizons = Arc::from([1_u32, 2]),
        ],
    );
    check(
        CausalQuery::Distribution,
        &distribution(),
        &[
            &|q| q.outcomes = Arc::from([v(1), v(4)]),
            &|q| q.interventions = Arc::from([set(0, 2.0)]),
            &|q| q.interventions = Arc::from([set(0, 1.0), set(5, 1.0)]),
            &|q| q.conditioning = Arc::from([v(2)]),
            &|q| q.target_population = TargetPopulation::Treated,
        ],
    );
    check(
        CausalQuery::PathSpecific,
        &path_specific(),
        &[
            &|q| q.treatment = v(5),
            &|q| q.outcome = v(6),
            &|q| q.path_nodes = Arc::from([v(2)]),
            &|q| q.control = set(0, 0.5),
            &|q| q.active = set(0, 2.0),
            &|q| q.target_population = TargetPopulation::Treated,
            &|q| q.max_paths = 65,
            &|q| q.max_len = 17,
        ],
    );
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "one linear list of single-field mutations, each checked against the same base query"
)]
fn every_response_field_reaches_the_encoding() {
    let point = |at: f64, order: u8, scale: DerivativeScale| ResponseFunctional::PointDerivative {
        outcome: v(1),
        treatment: v(0),
        at,
        order,
        scale,
    };
    let directional = |direction: [f64; 2]| ResponseFunctional::DirectionalDerivative {
        outcomes: Arc::from([v(1)]),
        treatments: Arc::from([v(0), v(2)]),
        at: Arc::from([1.0, 2.0]),
        direction: Arc::from(direction),
    };
    let jacobian = |treatments: [VariableId; 2]| ResponseFunctional::Jacobian {
        outcomes: Arc::from([v(1)]),
        treatments: Arc::from(treatments),
        at: Arc::from([1.0, 2.0]),
        scale: DerivativeScale::Identity,
    };
    let derivative = |weighting: DerivativeWeighting| ResponseFunctional::AverageDerivative {
        outcome: v(1),
        treatment: v(0),
        weighting,
    };
    check(
        CausalQuery::Response,
        &response(),
        &[
            &|q| q.functional = mean_curve(&[0.0, 2.0]),
            &|q| q.functional = mean_curve(&[0.0, 1.0, 2.0]),
            &|q| {
                q.functional = ResponseFunctional::MeanCurve {
                    outcome: v(1),
                    treatment: ContinuousDomain::new(
                        v(0),
                        GridSpec::Linspace { start: 0.0, end: 1.0, points: 2 },
                    ),
                };
            },
            &|q| {
                q.functional = ResponseFunctional::MeanCurve {
                    outcome: v(1),
                    treatment: ContinuousDomain::new(
                        v(0),
                        GridSpec::Linspace { start: 0.0, end: 1.0, points: 3 },
                    ),
                };
            },
            &|q| q.functional = derivative(DerivativeWeighting::Observed),
            &|q| q.functional = derivative(DerivativeWeighting::Uniform { lower: 0.0, upper: 1.0 }),
            &|q| q.functional = derivative(DerivativeWeighting::Custom(Arc::from([1.0, 2.0]))),
            &|q| q.functional = derivative(DerivativeWeighting::Custom(Arc::from([2.0, 1.0]))),
            &|q| q.functional = point(1.0, 1, DerivativeScale::Identity),
            &|q| q.functional = point(2.0, 1, DerivativeScale::Identity),
            &|q| q.functional = point(1.0, 2, DerivativeScale::Identity),
            &|q| q.functional = point(1.0, 1, DerivativeScale::LogTreatment),
            &|q| q.functional = point(1.0, 1, DerivativeScale::LogOutcome),
            &|q| q.functional = point(1.0, 1, DerivativeScale::LogLog),
            &|q| q.functional = directional([1.0, 0.0]),
            &|q| q.functional = directional([0.0, 1.0]),
            &|q| q.functional = jacobian([v(0), v(2)]),
            &|q| q.functional = jacobian([v(2), v(0)]),
            &|q| {
                q.functional = ResponseFunctional::InterventionResponse {
                    outcome: v(1),
                    interventions: Arc::from([set(0, 1.0)]),
                };
            },
            &|q| q.target_population = TargetPopulation::Treated,
            &|q| {
                q.observation = ObservationSpec::RightCensored {
                    latent: v(1),
                    observed: v(4),
                    censoring: v(5),
                    event: v(6),
                };
            },
            &|q| {
                q.observation = ObservationSpec::LeftCensored {
                    latent: v(1),
                    observed: v(4),
                    censoring: v(5),
                    event: v(6),
                };
            },
            &|q| {
                q.observation =
                    ObservationSpec::IntervalCensored { latent: v(1), lower: v(4), upper: v(5) };
            },
            &|q| {
                q.observation = ObservationSpec::Truncated {
                    latent: v(1),
                    observed: v(4),
                    lower: Some(v(5)),
                    upper: None,
                };
            },
            &|q| {
                q.observation = ObservationSpec::Truncated {
                    latent: v(1),
                    observed: v(4),
                    lower: None,
                    upper: Some(v(5)),
                };
            },
            &|q| {
                q.observation =
                    ObservationSpec::Selected { latent: v(1), observed: v(4), indicator: v(5) };
            },
            &|q| {
                q.observation_assumptions =
                    Arc::from([ObservationAssumption::IndependentGiven(Arc::from([v(2)]))]);
            },
            &|q| {
                q.observation_assumptions =
                    Arc::from([ObservationAssumption::OutcomeIndependentGiven(Arc::from([v(2)]))]);
            },
            &|q| {
                q.observation_assumptions =
                    Arc::from([ObservationAssumption::Structural(Arc::from("mar"))]);
            },
            &|q| q.temporal = Some(temporal_spec(&[1, 2])),
            &|q| q.temporal = Some(temporal_spec(&[1, 3])),
            &|q| {
                q.temporal = Some(
                    TemporalResponseSpec::new(vec![1_u32, 2], TemporalPolicy::pulse(-2), None)
                        .unwrap(),
                );
            },
            &|q| {
                q.temporal = Some(
                    TemporalResponseSpec::new(vec![1_u32, 2], TemporalPolicy::pulse(-1), Some(4))
                        .unwrap(),
                );
            },
            &|q| q.temporal = Some(temporal_spec(&[1, 2]).with_feedback(feedback()).unwrap()),
            &|q| {
                let mut changed = feedback();
                changed.initial_law = InitialLaw::named("steady").unwrap();
                q.temporal = Some(temporal_spec(&[1, 2]).with_feedback(changed).unwrap());
            },
            &|q| {
                let mut changed = feedback();
                changed.delay_ticks = 2;
                q.temporal = Some(temporal_spec(&[1, 2]).with_feedback(changed).unwrap());
            },
            &|q| {
                let mut changed = feedback();
                changed.innovation_variance = 2.0;
                q.temporal = Some(temporal_spec(&[1, 2]).with_feedback(changed).unwrap());
            },
            &|q| {
                q.outcome_functional = OutcomeFunctional::Quantile(OrderedFloatBits::from_f64(0.5))
            },
        ],
    );
}

#[test]
fn every_transport_and_catalog_field_reaches_the_encoding() {
    check(
        CausalQuery::Transport,
        &transport(),
        &[
            &|q| q.response.functional = mean_curve(&[0.0, 2.0]),
            &|q| q.source_population = Arc::from("other"),
            &|q| q.target_population = Arc::from("other"),
            &|q| std::mem::swap(&mut q.source_population, &mut q.target_population),
            &|q| q.source_experiments = Arc::from([v(0)]),
            &|q| q.catalog = Some(EvidenceCatalog::empty()),
            &|q| q.catalog = Some(catalog(false)),
        ],
    );
    let with_catalog = |edit: &dyn Fn(&mut EvidenceCatalog)| {
        let mut changed = catalog(false);
        edit(&mut changed);
        let mut query = transport();
        query.catalog = Some(changed);
        query
    };
    let regime = |edit: &dyn Fn(&mut EvidenceRegime)| {
        with_catalog(&|c: &mut EvidenceCatalog| {
            let mut regimes = c.regimes.to_vec();
            edit(&mut regimes[0]);
            c.regimes = regimes.into();
        })
    };
    let queries = [
        with_catalog(&|_| {}),
        with_catalog(&|c| c.target_sampling = None),
        with_catalog(&|c| c.target_sampling = Some(TargetSampling::ConvenienceSample)),
        with_catalog(&|c| c.bindings = Arc::from([])),
        with_catalog(&|c| {
            let mut bindings = c.bindings.to_vec();
            bindings[0].schema_names = labels(&["y", "x"]);
            c.bindings = bindings.into();
        }),
        with_catalog(&|c| {
            let mut bindings = c.bindings.to_vec();
            bindings[0].weights = None;
            c.bindings = bindings.into();
        }),
        with_catalog(&|c| {
            let mut bindings = c.bindings.to_vec();
            bindings[0].dependence = DependenceGroup::LinkedUnits;
            c.bindings = bindings.into();
        }),
        with_catalog(&|c| {
            let mut bindings = c.bindings.to_vec();
            bindings[0].sampling = SamplingDesign::Clustered;
            c.bindings = bindings.into();
        }),
        with_catalog(&|c| {
            let mut bindings = c.bindings.to_vec();
            bindings[0].dataset_identity = Some(Arc::from("data"));
            c.bindings = bindings.into();
        }),
        with_catalog(&|c| {
            let mut environments = c.environments.to_vec();
            environments[1].identity = Arc::from("elsewhere");
            c.environments = environments.into();
        }),
        with_catalog(&|c| {
            let mut environments = c.environments.to_vec();
            environments[1].variables = Arc::from([VariableCoordinate {
                variable: v(0),
                domain: VariableDomain::Categorical { cardinality: 3 },
                unit: Some(Arc::from("mg")),
            }]);
            c.environments = environments.into();
        }),
        with_catalog(&|c| {
            let mut environments = c.environments.to_vec();
            environments[1].selection_targets = Arc::from([v(0)]);
            c.environments = environments.into();
        }),
        regime(&|r| r.id = RegimeId::from_raw(9)),
        regime(&|r| r.label = None),
        regime(&|r| r.kind = RegimeKind::Observational),
        regime(&|r| r.evidence_kind = EvidenceKind::Proposed),
        regime(&|r| r.interventions = Arc::from([v(0)])),
        regime(&|r| r.intervention_values = Arc::from([])),
        regime(&|r| r.measured = Arc::from([v(1)])),
        regime(&|r| r.conditioned_on = Arc::from([])),
        regime(&|r| r.population = Arc::from("target")),
        regime(&|r| r.distribution = DistributionAvailability::Joint),
        regime(&|r| r.study = None),
        regime(&|r| r.selection = SamplingSelection::Population),
        regime(&|r| r.origin = LawOrigin::ModelArtifact { artifact: Arc::from("posterior") }),
        regime(&|r| r.origin = LawOrigin::Recovered { derivation: Arc::from("posterior") }),
    ];
    assert_distinct(&queries.map(CausalQuery::Transport));
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "one linear list of single-field mutations, each checked against the same base query"
)]
fn every_design_query_field_reaches_the_encoding() {
    check(
        CausalQuery::Interference,
        &interference(),
        &[
            &|q| {
                q.assignment = AssignmentDesign::Bernoulli { probabilities: Arc::from([0.4, 0.5]) }
            },
            &|q| q.assignment = AssignmentDesign::CompleteRandomization { treated: 1 },
            &|q| {
                q.assignment = AssignmentDesign::ClusterRandomization {
                    clusters: Arc::from([0_u32, 1]),
                    treated_clusters: 1,
                };
            },
            &|q| {
                q.assignment = AssignmentDesign::TwoStageSaturation {
                    clusters: Arc::from([0_u32, 1]),
                    low_probability: 0.2,
                    high_probability: 0.8,
                    high_clusters: 1,
                    realized_saturation: Arc::from([0.2, 0.8]),
                };
            },
            &|q| {
                q.assignment = AssignmentDesign::ObservedExposure {
                    clusters: Arc::from([0_u32, 1]),
                    propensity_from: Arc::from([0.5, 0.5]),
                    propensity_to: Arc::from([0.5, 0.5]),
                    provenance: ExposurePropensityProvenance::Known,
                    assume_network_exchangeability: true,
                };
            },
            &|q| {
                q.assignment = AssignmentDesign::ObservedExposure {
                    clusters: Arc::from([0_u32, 1]),
                    propensity_from: Arc::from([0.5, 0.5]),
                    propensity_to: Arc::from([0.5, 0.5]),
                    provenance: ExposurePropensityProvenance::ExternallyEstimated,
                    assume_network_exchangeability: true,
                };
            },
            &|q| q.exposure = ExposureMapping::NeighborCount,
            &|q| q.exposure = ExposureMapping::NeighborFraction,
            &|q| q.exposure = ExposureMapping::WeightedNeighborExposure,
            &|q| q.exposure = ExposureMapping::Custom(Arc::from("k")),
            &|q| {
                q.functional = InterferenceFunctional::ExposureContrast {
                    outcome: v(0),
                    from: ExposureLevel { own: 1.0, neighbors: 0.0 },
                    to: ExposureLevel { own: 0.0, neighbors: 0.0 },
                };
            },
            &|q| {
                q.functional = InterferenceFunctional::ExposureContrast {
                    outcome: v(0),
                    from: ExposureLevel { own: 0.0, neighbors: 0.0 },
                    to: ExposureLevel { own: 1.0, neighbors: 0.5 },
                };
            },
            &|q| q.probability_draws = 5,
        ],
    );
    check(
        CausalQuery::RandomizedEffect,
        &randomized(),
        &[
            &|q| q.estimand = RandomizedEstimand::ComplierAverageCausalEffect,
            &|q| q.estimand = RandomizedEstimand::TreatmentOnTreated,
            &|q| q.design = RandomizationDesign::Complete { treated_units: 1 },
            &|q| q.design = RandomizationDesign::Cluster { treated_clusters: 1 },
            &|q| {
                q.design = RandomizationDesign::Stratified {
                    blocks: labels(&["s", "s"]),
                    treated_per_row: Arc::from([1_usize, 1]),
                };
            },
            &|q| {
                q.design = RandomizationDesign::Factorial2x2 {
                    second_factor_assignment: Arc::from([true, false]),
                    cell_counts: [1, 0, 0, 1],
                    second_factor_arms: (Arc::from("a"), Arc::from("b")),
                };
            },
            &|q| {
                q.design = RandomizationDesign::MultiArm {
                    assignment: Arc::from([0_usize, 1]),
                    probabilities: Arc::from(vec![vec![0.5, 0.5]; 2]),
                    arms: labels(&["control", "treated"]),
                };
            },
            &|q| q.design = RandomizationDesign::Switchback { periods: labels(&["p", "q"]) },
            &|q| q.outcome = v(5),
            &|q| q.realized_assignment = Arc::from([false, true]),
            &|q| q.assignment_probabilities = Arc::from([0.4, 0.5]),
            &|q| q.assignment_units = labels(&["b", "a"]),
            &|q| q.outcome_units = labels(&["b", "a"]),
            &|q| std::mem::swap(&mut q.treatment_arms.0, &mut q.treatment_arms.1),
            &|q| q.fixed_cuped = Some((v(2), 0.1)),
            &|q| q.fixed_cuped = Some((v(2), 0.2)),
            &|q| q.ancova_covariates = Arc::from([v(2)]),
            &|q| q.received_treatment = Some(Arc::from([true, false])),
            &|q| q.exact_randomization_test = true,
        ],
    );
    check(
        CausalQuery::PolicyValue,
        &policy_value(),
        &[
            &|q| q.outcome = v(5),
            &|q| q.assignment = Arc::from([false, true]),
            &|q| q.propensity = Arc::from([0.4]),
            &|q| q.actions = Arc::from([false, true]),
            &|q| q.reference = Arc::from([true, false]),
            &|q| q.mu0 = Arc::from([0.0, 0.0]),
            &|q| q.mu1 = Arc::from([0.0, 0.0]),
            &|q| q.costs = Arc::from([1.0]),
            &|q| q.reference_costs = Arc::from([1.0]),
            &|q| q.evaluation_subject_ids = labels(&["b", "a"]),
            &|q| q.disjoint_training_subjects = false,
            &|q| q.crossfit_fold_ownership_valid = true,
            &|q| q.global_constraints_present = true,
            &|q| q.multi_action = Some(multi_action()),
            &|q| {
                let mut inputs = multi_action();
                inputs.budget = Some(1.0);
                q.multi_action = Some(inputs);
            },
            &|q| {
                let mut inputs = multi_action();
                inputs.capacities = Arc::from([1_usize, 2]);
                q.multi_action = Some(inputs);
            },
            &|q| {
                let mut inputs = multi_action();
                inputs.cate_groups = labels(&["g", "g"]);
                q.multi_action = Some(inputs);
            },
            &|q| q.uplift_bins = Arc::from([0_usize, 1]),
            &|q| q.uplift_bin_count = 2,
            &|q| q.uplift_training_subject_ids = labels(&["c"]),
            &|q| {
                q.regret = Some(FixedCandidateRegretInputs {
                    candidates: Arc::from([flags(&[true, false]), flags(&[false, false])]),
                    selected_index: 0,
                    training_subject_ids: labels(&["c"]),
                });
            },
            &|q| {
                q.regret = Some(FixedCandidateRegretInputs {
                    candidates: Arc::from([flags(&[true, false]), flags(&[false, false])]),
                    selected_index: 1,
                    training_subject_ids: labels(&["c"]),
                });
            },
        ],
    );
    check(
        CausalQuery::ContinuousDoseResponse,
        &continuous_dose(),
        &[
            &|q| q.outcome = v(5),
            &|q| q.dose = v(5),
            &|q| q.dose_density = v(5),
            &|q| q.baseline_groups = labels(&["h", "g"]),
            &|q| q.target_doses = Arc::from([1.0, 0.5]),
            &|q| q.bandwidth = 0.5,
            &|q| q.min_local_support = 4,
            &|q| q.density_provenance = Arc::from("estimated"),
            &|q| {
                q.fixed_policy = Some(FixedGroupDosePolicy {
                    policy_doses: dose_rule(&[("g", 1.0), ("h", 0.5)]),
                    reference_doses: dose_rule(&[("g", 0.5), ("h", 0.5)]),
                });
            },
            &|q| {
                q.fixed_policy = Some(FixedGroupDosePolicy {
                    policy_doses: dose_rule(&[("g", 0.5), ("h", 0.5)]),
                    reference_doses: dose_rule(&[("g", 1.0), ("h", 0.5)]),
                });
            },
        ],
    );
    check(
        CausalQuery::PanelDid,
        &panel_did(),
        &[
            &|q| q.design = DidSamplingDesign::RepeatedCrossSection,
            &|q| q.design = DidSamplingDesign::StaggeredGroupTime,
            &|q| q.design = DidSamplingDesign::StaggeredEventStudy,
            &|q| q.design = DidSamplingDesign::AugmentedPanel,
            &|q| q.outcome = v(5),
            &|q| q.treated = Arc::from([false, true]),
            &|q| q.post = Arc::from([true, false]),
            &|q| q.subjects = labels(&["b", "a"]),
            &|q| q.clusters = labels(&["c", "d"]),
            &|q| q.periods = Arc::from([1_i64, 0]),
            &|q| q.cohorts = Arc::from([0_i64, 1]),
            &|q| q.target = Some((1, 1)),
            &|q| q.target = Some((1, 2)),
            &|q| {
                q.augmented = Some(AugmentedPanelNuisance {
                    outcome_pre: v(2),
                    propensity: v(3),
                    untreated_change_prediction: v(4),
                    predictions_cross_fitted: false,
                });
            },
            &|q| {
                q.augmented = Some(AugmentedPanelNuisance {
                    outcome_pre: v(2),
                    propensity: v(3),
                    untreated_change_prediction: v(4),
                    predictions_cross_fitted: true,
                });
            },
            &|q| q.design = DidSamplingDesign::StaggeredHeterogeneityRobust,
            &|q| q.design = DidSamplingDesign::CovariateAdjustedTwoPeriod,
            &|q| q.robust = Some(HeterogeneityRobustDid::default()),
            &|q| {
                q.robust = Some(HeterogeneityRobustDid {
                    estimator: RolloutDidEstimator::Imputation,
                    ..HeterogeneityRobustDid::default()
                });
            },
            &|q| {
                q.robust = Some(HeterogeneityRobustDid {
                    comparison: DidComparisonGroup::NotYetTreated,
                    ..HeterogeneityRobustDid::default()
                });
            },
            &|q| {
                q.robust = Some(HeterogeneityRobustDid {
                    nuisance: DidNuisanceMethod::OutcomeRegression,
                    ..HeterogeneityRobustDid::default()
                });
            },
            &|q| {
                q.robust = Some(HeterogeneityRobustDid {
                    base_period: DidBasePeriod::Universal,
                    ..HeterogeneityRobustDid::default()
                });
            },
            &|q| {
                q.robust = Some(HeterogeneityRobustDid {
                    aggregation: DidAggregation::Dynamic,
                    ..HeterogeneityRobustDid::default()
                });
            },
            &|q| {
                q.robust = Some(HeterogeneityRobustDid {
                    anticipation: 1,
                    ..HeterogeneityRobustDid::default()
                });
            },
            &|q| {
                q.robust = Some(HeterogeneityRobustDid {
                    covariates: Arc::from([v(7)]),
                    ..HeterogeneityRobustDid::default()
                });
            },
            &|q| {
                q.resampling = Some(DidResampling {
                    scheme: DidResamplingScheme::ClusterBootstrap,
                    replicates: 99,
                });
            },
            &|q| {
                q.resampling = Some(DidResampling {
                    scheme: DidResamplingScheme::ClusterMultiplier,
                    replicates: 99,
                });
            },
        ],
    );
    check(
        CausalQuery::SyntheticControl,
        &synthetic_control(),
        &[
            &|q| q.outcome = v(5),
            &|q| q.units = labels(&["b", "a"]),
            &|q| q.periods = Arc::from([1_i64, 0]),
            &|q| q.treated_unit = Arc::from("b"),
            &|q| q.intervention_period = -1,
            &|q| q.method = SyntheticPanelMethod::DifferenceInDifferences,
            &|q| q.uniform_unit_randomization = true,
            &|q| q.sharp_null_effect = Some(0.0),
            &|q| q.augmentation_ridge = Some(0.0),
            &|q| {
                q.rollout = Some(SyntheticRollout {
                    adoptions: Arc::from([(Arc::from("a"), 1_i64)]),
                    interval: SyntheticInterval::Conformal,
                    replicates: 0,
                });
            },
            &|q| {
                q.rollout = Some(SyntheticRollout {
                    adoptions: Arc::from([(Arc::from("a"), 1_i64)]),
                    interval: SyntheticInterval::PlaceboVariance,
                    replicates: 50,
                });
            },
        ],
    );
    check(
        CausalQuery::LocalPolynomialRatio,
        &local_ratio(),
        &[
            &|q| q.outcome = v(5),
            &|q| q.treatment = v(5),
            &|q| q.running = v(5),
            &|q| q.cutoff = -0.0,
            &|q| q.bandwidth = 2.0,
            &|q| q.kink = true,
            &|q| q.options = Some(LocalRatioOptions::default()),
            &|q| {
                q.options = Some(LocalRatioOptions {
                    bandwidth_rule: LocalBandwidthRule::MseOptimal,
                    ..LocalRatioOptions::default()
                });
            },
            &|q| {
                q.options = Some(LocalRatioOptions {
                    temporal_replicates: Some(99),
                    ..LocalRatioOptions::default()
                });
            },
            &|q| {
                q.options =
                    Some(LocalRatioOptions { density_test: true, ..LocalRatioOptions::default() });
            },
            &|q| {
                q.options = Some(LocalRatioOptions {
                    balance_covariates: Arc::from([v(7)]),
                    ..LocalRatioOptions::default()
                });
            },
        ],
    );
}

#[test]
fn every_longitudinal_and_survival_field_reaches_the_encoding() {
    check(
        CausalQuery::Survival,
        &survival(),
        &[
            &|q| q.duration = v(5),
            &|q| q.event = v(5),
            &|q| q.treatment = v(5),
            &|q| q.tau = 6.0,
            &|q| q.delayed_entry = Some(v(5)),
            &|q| {
                q.known_censoring = Some(KnownCensoringSurvival {
                    times: Arc::from([0.0, 5.0]),
                    columns: Arc::from([v(6), v(7)]),
                    minimum_probability: 0.01,
                });
            },
            &|q| {
                q.known_censoring = Some(KnownCensoringSurvival {
                    times: Arc::from([0.0, 5.0]),
                    columns: Arc::from([v(7), v(6)]),
                    minimum_probability: 0.01,
                });
            },
            &|q| {
                q.known_censoring = Some(KnownCensoringSurvival {
                    times: Arc::from([0.0, 5.0]),
                    columns: Arc::from([v(6), v(7)]),
                    minimum_probability: 0.02,
                });
            },
            &|q| {
                q.observation_assumption =
                    ObservationAssumption::IndependentGiven(Arc::from([v(3)]))
            },
            &|q| q.functional = SurvivalFunctional::CumulativeIncidence { target_cause: 1 },
            &|q| q.functional = SurvivalFunctional::CumulativeIncidence { target_cause: 2 },
        ],
    );
    check(
        CausalQuery::LongitudinalRegime,
        &longitudinal(),
        &[
            &|q| q.outcome = v(5),
            &|q| q.method = LongitudinalRegimeMethod::GFormula,
            &|q| q.method = LongitudinalRegimeMethod::SequentialDoublyRobust,
            &|q| q.method = LongitudinalRegimeMethod::MarginalStructuralModel,
            &|q| q.period_outcome_predictions = Some(Arc::from([1.0])),
            &|q| q.known_fixed_outcome_predictions = true,
            &|q| q.stabilizing_numerator_probabilities = Some(Arc::from([1.0])),
            &|q| q.q_predictions = Some(Arc::from([1.0])),
            &|q| q.observation_history = Some(Arc::from([true, true])),
            &|q| q.prediction_fold_ids = Some(Arc::from([0_u32, 1])),
            &|q| q.periods = 2,
            &|q| q.treatment_history = Arc::from([false, true]),
            &|q| q.regime_actions = Arc::from([false, true]),
            &|q| q.treatment_probabilities = Arc::from([0.4, 0.5]),
            &|q| q.censoring_probabilities = Arc::from([0.9, 1.0]),
            &|q| q.outcome_observed = Arc::from([false, true]),
            &|q| q.subject_ids = labels(&["b", "a"]),
            &|q| q.fold_ids = Arc::from([1_u32, 0]),
            &|q| q.excluded_fold_predictions = true,
            &|q| q.probabilities_known_by_design = false,
            &|q| q.minimum_probability = 0.02,
            &|q| q.rule_id = Some(Arc::from("rule")),
            &|q| q.rule_version = Some(Arc::from("rule")),
            &|q| q.rule_provenance = Some(Arc::from("rule")),
        ],
    );
    check(
        CausalQuery::SequentialContrast,
        &sequential(),
        &[
            &|q| q.outcome = v(5),
            &|q| q.first_treatment = v(5),
            &|q| q.second_treatment = v(5),
            &|q| q.baseline = Arc::from([v(3), v(5)]),
            &|q| std::mem::swap(&mut q.baseline, &mut q.intermediate),
            &|q| q.intermediate = Arc::from([]),
            &|q| q.active = [1, 0],
            &|q| q.control = [0, 1],
            &|q| q.fold_of_row = Arc::from([1_u32, 0]),
            &|q| q.fold_count = 3,
            &|q| q.min_probability = 0.1,
        ],
    );
    check(
        CausalQuery::HistoryMatch,
        &history_match(),
        &[
            &|q| q.outcome = v(5),
            &|q| q.treated = v(5),
            &|q| q.history = Arc::from([v(3), v(2)]),
            &|q| q.period = Some(v(5)),
            &|q| q.weight = Some(v(5)),
            &|q| q.caliper = Some(0.5),
        ],
    );
}

#[test]
fn reordering_an_unordered_collection_keeps_the_encoding() {
    same(CausalQuery::AverageEffect, average(), average().with_effect_modifiers([v(3), v(2)]));
    same(CausalQuery::TemporalEffect, temporal(), temporal().with_effect_modifiers([v(3), v(2)]));
    same(
        CausalQuery::ConditionalEffect,
        ConditionalEffectQuery { inner: average() },
        ConditionalEffectQuery { inner: average().with_effect_modifiers([v(3), v(2)]) },
    );
    same(CausalQuery::Mediation, mediation(), {
        let mut q = mediation();
        q.mediators = Arc::from([v(3), v(2)]);
        q
    });
    same(CausalQuery::Distribution, distribution(), distribution().with_conditioning([v(3), v(2)]));
    same(CausalQuery::PathSpecific, path_specific(), path_specific().with_path_nodes([v(3), v(2)]));
    let assumptions = |first: [VariableId; 2], reversed: bool| {
        let given = ObservationAssumption::IndependentGiven(Arc::from(first));
        let claim = ObservationAssumption::Structural(Arc::from("mar"));
        let list = if reversed { [claim, given] } else { [given, claim] };
        response().with_observation(ObservationSpec::Complete, list)
    };
    same(CausalQuery::Response, assumptions([v(2), v(3)], false), assumptions([v(3), v(2)], true));
    same(CausalQuery::Transport, transport(), {
        let mut q = transport();
        q.source_experiments = Arc::from([v(2), v(0)]);
        q
    });
    same(
        CausalQuery::Transport,
        {
            let mut q = transport();
            q.catalog = Some(catalog(false));
            q
        },
        {
            let mut q = transport();
            q.catalog = Some(catalog(true));
            q
        },
    );
    let with_rules = |policy: &[(&str, f64)]| {
        let mut q = continuous_dose();
        q.fixed_policy = Some(FixedGroupDosePolicy {
            policy_doses: dose_rule(policy),
            reference_doses: dose_rule(&[("g", 0.5), ("h", 0.5)]),
        });
        q
    };
    same(
        CausalQuery::ContinuousDoseResponse,
        with_rules(&[("g", 1.0), ("h", 0.5)]),
        with_rules(&[("h", 0.5), ("g", 1.0)]),
    );
}

#[test]
fn variants_with_similar_payloads_encode_differently() {
    let queries = [
        CausalQuery::AverageEffect(average()),
        CausalQuery::ConditionalEffect(ConditionalEffectQuery { inner: average() }),
        CausalQuery::TemporalEffect(temporal()),
        CausalQuery::Counterfactual(counterfactual()),
        CausalQuery::NestedCounterfactual(nested()),
        CausalQuery::AnomalyAttribution(anomaly()),
        CausalQuery::ChangeAttribution(change()),
        CausalQuery::MechanismChange(mechanism()),
        CausalQuery::UnitChange(unit_change()),
        CausalQuery::Mediation(mediation()),
        CausalQuery::Distribution(distribution()),
        CausalQuery::PathSpecific(path_specific()),
        CausalQuery::Response(response()),
        CausalQuery::Transport(transport()),
        CausalQuery::Interference(interference()),
        CausalQuery::RandomizedEffect(randomized()),
        CausalQuery::PolicyValue(policy_value()),
        CausalQuery::ContinuousDoseResponse(continuous_dose()),
        CausalQuery::PanelDid(panel_did()),
        CausalQuery::SyntheticControl(synthetic_control()),
        CausalQuery::LocalPolynomialRatio(local_ratio()),
        CausalQuery::Survival(survival()),
        CausalQuery::LongitudinalRegime(longitudinal()),
        CausalQuery::SequentialContrast(sequential()),
        CausalQuery::HistoryMatch(history_match()),
    ];
    assert_distinct(&queries);
    // A payload the transport query wraps is still a different query from the
    // bare response.
    assert_ne!(
        digest(&CausalQuery::Response(response())),
        digest(&CausalQuery::Transport(transport()))
    );
    // A path-specific and an average effect over the same pair stay distinct.
    assert_ne!(
        digest(&CausalQuery::AverageEffect(AverageEffectQuery::binary_ate(v(0), v(1)))),
        digest(&CausalQuery::PathSpecific(PathSpecificEffectQuery::binary(v(0), v(1))))
    );
}

fn discover() -> DiscoverQuery {
    DiscoverQuery::new(
        vec![v(0), v(1), v(2)],
        DiscoveryMethod::Pc,
        SearchBounds {
            alpha: 0.05,
            max_cond_size: 2,
            max_lag: 0,
            schedule: None,
            configuration: [7; 32],
        },
    )
}

fn attribute() -> AttributeQuery {
    AttributeQuery::new(AttributeTarget::Paths {
        sources: Arc::from([v(0)]),
        outcome: v(2),
        max_paths: 64,
        max_len: 16,
        through: Arc::from([]),
    })
}

fn simulate() -> SimulateQuery {
    SimulateQuery::new(SimulationTarget::InterventionalSample {
        interventions: Arc::from([set(0, 1.0)]),
        draws: 100,
        seed: 3,
    })
}

/// The tag a variant writes before its payload.
fn tagged<T: CanonicalEncode>(tag: u64, payload: &T) -> [u8; 32] {
    let mut writer = CanonicalWriter::new("t");
    writer.word(tag);
    payload.encode(&mut writer);
    writer.finish()
}

#[test]
fn the_structure_and_model_kinds_append_their_tags() {
    assert_eq!(digest(&CausalQuery::Discover(discover())), tagged(26, &discover()));
    assert_eq!(digest(&CausalQuery::Attribute(attribute())), tagged(27, &attribute()));
    assert_eq!(digest(&CausalQuery::Simulate(simulate())), tagged(28, &simulate()));
    assert_eq!(CausalQuery::Discover(discover()).kind(), CausalQueryKind::Discover);
    assert_eq!(CausalQuery::Attribute(attribute()).kind(), CausalQueryKind::Attribute);
    assert_eq!(CausalQuery::Simulate(simulate()).kind(), CausalQueryKind::Simulate);
    assert_distinct(&[
        CausalQuery::HistoryMatch(history_match()),
        CausalQuery::Discover(discover()),
        CausalQuery::Attribute(attribute()),
        CausalQuery::Simulate(simulate()),
        CausalQuery::AnomalyAttribution(anomaly()),
        CausalQuery::Attribute(AttributeQuery::new(AttributeTarget::Anomaly(anomaly()))),
    ]);
}

#[test]
fn every_discover_field_reaches_the_bytes() {
    check(
        CausalQuery::Discover,
        &discover(),
        &[
            &|q| q.variables = Arc::from([v(2), v(1), v(0)]),
            &|q| q.method = DiscoveryMethod::Fci,
            &|q| q.method = DiscoveryMethod::Ges { screen_pc: true, max_subset: None },
            &|q| q.method = DiscoveryMethod::Ges { screen_pc: false, max_subset: Some(4) },
            &|q| q.bounds.alpha = 0.01,
            &|q| q.bounds.max_cond_size = 3,
            &|q| q.bounds.max_lag = 1,
            &|q| {
                q.bounds.schedule =
                    Some(SearchSchedule { chains: 2, warmup: 10, draws: 20, thin: 1 });
            },
            &|q| q.bounds.configuration = [8; 32],
        ],
    );
}

#[test]
fn every_attribute_and_simulate_field_reaches_the_bytes() {
    check(
        CausalQuery::Attribute,
        &attribute(),
        &[
            &|q| {
                if let AttributeTarget::Paths { max_paths, .. } = &mut q.target {
                    *max_paths = 8;
                }
            },
            &|q| {
                if let AttributeTarget::Paths { through, .. } = &mut q.target {
                    *through = Arc::from([v(1)]);
                }
            },
            &|q| q.target = AttributeTarget::UnitChange(unit_change()),
            &|q| {
                q.target = AttributeTarget::MechanismChange {
                    query: mechanism(),
                    test: MechanismChangeTest::MeanDifference,
                };
            },
            &|q| {
                q.target = AttributeTarget::DistributionChange {
                    change: change(),
                    estimator: ChangeEstimator::Sampled {
                        measure: ChangeMeasure::MeanDifference,
                        n_samples: 100,
                        seed: 0,
                    },
                };
            },
            &|q| {
                q.target = AttributeTarget::RootCauses {
                    change: change(),
                    estimator: ChangeEstimator::Sampled {
                        measure: ChangeMeasure::MeanDifference,
                        n_samples: 100,
                        seed: 0,
                    },
                };
            },
        ],
    );
    check(
        CausalQuery::Simulate,
        &simulate(),
        &[
            &|q| {
                if let SimulationTarget::InterventionalSample { seed, .. } = &mut q.target {
                    *seed = 4;
                }
            },
            &|q| {
                if let SimulationTarget::InterventionalSample { draws, .. } = &mut q.target {
                    *draws = 101;
                }
            },
            &|q| {
                q.target = SimulationTarget::PolicyValue {
                    interventions: Arc::from([set(0, 1.0)]),
                    outcome: v(2),
                    draws: 100,
                };
            },
            &|q| {
                q.target = SimulationTarget::UnitEffects {
                    treatment: v(0),
                    outcome: v(2),
                    active: 1.0,
                    control: 0.0,
                };
            },
        ],
    );
}

#[test]
fn the_new_kinds_validate_their_declarations() {
    assert!(discover().validate().is_ok());
    let mut static_lag = discover();
    static_lag.bounds.max_lag = 1;
    assert!(static_lag.validate().is_err());
    let mut repeated = discover();
    repeated.variables = Arc::from([v(0), v(0)]);
    assert!(repeated.validate().is_err());
    let mut unscheduled = discover();
    unscheduled.method = DiscoveryMethod::StructureMcmc;
    assert!(unscheduled.validate().is_err());
    assert_eq!(DiscoveryMethod::StructureMcmc.inference(), DiscoveryInference::Bayesian);
    assert_eq!(DiscoveryMethod::PcmciPlus.output(), DiscoveredClass::TemporalCpdag);
    assert!(DiscoveredClass::Pag.is_equivalence_class());
    assert!(attribute().validate().is_ok());
    assert!(simulate().validate().is_ok());
    let same = SimulateQuery::new(SimulationTarget::UnitEffects {
        treatment: v(1),
        outcome: v(1),
        active: 1.0,
        control: 0.0,
    });
    assert!(same.validate().is_err());
}

fn temporal_design() -> TemporalDesignQuery {
    TemporalDesignQuery {
        outcome: v(1),
        units: labels(&["a", "a", "b", "b"]),
        periods: Arc::from([0_i64, 1, 0, 1]),
        lag: 1,
        assignment: TemporalAssignment::TwoArm {
            assignment: flags(&[true, false, false, true]),
            probability: Arc::from([0.5, 0.4, 0.5, 0.6]),
        },
        functional: TemporalDesignFunctional::Mean,
        min_probability: 0.05,
        outcome_misclassification: None,
    }
}

#[test]
fn the_temporal_design_kind_appends_its_tag_and_every_field_reaches_the_bytes() {
    assert_eq!(
        digest(&CausalQuery::TemporalDesign(temporal_design())),
        tagged(29, &temporal_design())
    );
    assert_eq!(
        CausalQuery::TemporalDesign(temporal_design()).kind(),
        CausalQueryKind::TemporalDesign
    );
    assert!(temporal_design().validate().is_ok());
    check(
        CausalQuery::TemporalDesign,
        &temporal_design(),
        &[
            &|q| q.outcome = v(5),
            &|q| q.units = labels(&["a", "a", "c", "c"]),
            &|q| q.periods = Arc::from([0_i64, 2, 0, 1]),
            &|q| q.lag = 0,
            &|q| q.functional = TemporalDesignFunctional::Quantile { level: 0.5 },
            &|q| q.min_probability = 0.1,
            &|q| {
                q.outcome_misclassification = Some(OutcomeMisclassification {
                    true_positive: 9,
                    false_negative: 1,
                    true_negative: 8,
                    false_positive: 2,
                })
            },
            &|q| {
                q.assignment = TemporalAssignment::Interference {
                    assignment: flags(&[true, false, false, true]),
                    probability: Arc::from([0.5, 0.4, 0.5, 0.6]),
                    clusters: labels(&["x", "x", "x", "x"]),
                }
            },
            &|q| {
                q.assignment = TemporalAssignment::Dose {
                    dose: Arc::from([0.1, 0.2, 0.3, 0.4]),
                    density: Arc::from([1.0, 1.0, 1.0, 1.0]),
                    bandwidth: 0.2,
                    target: TemporalDoseTarget::SmoothedContrast { active: 0.3, control: 0.1 },
                }
            },
            &|q| {
                q.assignment = TemporalAssignment::RegimeSurvival {
                    treatment: v(2),
                    censored: v(3),
                    confounders: Arc::from([v(4)]),
                    horizon: 2,
                }
            },
        ],
    );
    assert_distinct(&[
        CausalQuery::HistoryMatch(history_match()),
        CausalQuery::Simulate(simulate()),
        CausalQuery::TemporalDesign(temporal_design()),
    ]);
    let mut repeated = temporal_design();
    repeated.periods = Arc::from([0_i64, 0, 0, 1]);
    assert!(repeated.validate().is_err());
    let mut quantile_multi = temporal_design();
    quantile_multi.functional = TemporalDesignFunctional::Quantile { level: 1.5 };
    assert!(quantile_multi.validate().is_err());
}
