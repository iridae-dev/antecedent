//! Noisy-choice estimates of preference parameters: recovery of a known consistency and rates
//! from simulated logit choices, a hand-derived one-column fit, learnability and the measure,
//! clustered and model covariance, per-decider outcomes, determinism, and every refusal.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::reason_details;
use antecedent_design::choice_record::{
    BeliefSource, ChoiceOption, ChoiceRecord, ChoiceRecordError, ChoiceRecordSet, OptionBelief,
    Pooling,
};
use antecedent_design::preference_choice_model::{
    ASSUMPTION_LOGIT_CHOICE, ChoiceModelError, ChoiceModelEstimate, ChoiceModelOptions,
    ChoiceModelResult, CovarianceKind, EstimateOutcome, estimate_choice_model,
};
use antecedent_design::preference_decision::MeasureOrigin;
use antecedent_design::preference_model::{
    Attribute, LinearRelation, PreferenceExpr as E, PreferenceParameter, PreferenceSet,
    RelationOrigin, RelationSense, ValueModel,
};
use antecedent_io_core::distribution_artifact::DistributionCalibration;
use antecedent_stats::{ConditionalLogitDesign, ConditionalLogitOptions, fit_conditional_logit};

/// Small fixed-seed xorshift64* generator for simulating choices.
struct Xorshift(u64);

impl Xorshift {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in the open interval `(0, 1)`.
    #[allow(clippy::cast_precision_loss, reason = "53-bit integers are exact in f64")]
    fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / 9_007_199_254_740_992.0
    }

    fn gumbel(&mut self) -> f64 {
        -(-self.uniform().ln()).ln()
    }
}

#[allow(clippy::cast_precision_loss, reason = "test counts are far below 2^53")]
fn as_f64(n: usize) -> f64 {
    n as f64
}

fn close(a: f64, b: f64, tolerance: f64) -> bool {
    (a - b).abs() <= tolerance * a.abs().max(b.abs()).max(1.0)
}

fn attribute(id: &str, units: &str) -> Attribute {
    Attribute { id: id.to_owned(), units: units.to_owned() }
}

fn parameter(id: &str, lower: f64, upper: f64) -> PreferenceParameter {
    PreferenceParameter {
        id: id.to_owned(),
        units: "utility/unit".to_owned(),
        lower,
        upper,
        assumption_id: format!("declared_{id}"),
    }
}

#[allow(clippy::unnecessary_box_returns, reason = "expression variants hold boxed operands")]
fn b(e: E) -> Box<E> {
    Box::new(e)
}

fn mul(a: E, c: E) -> E {
    E::Mul(b(a), b(c))
}

fn add(a: E, c: E) -> E {
    E::Add(b(a), b(c))
}

fn model(attributes: Vec<Attribute>, parameters: Vec<PreferenceParameter>, value: E) -> ValueModel {
    ValueModel {
        attributes,
        preferences: PreferenceSet { parameters, relations: Vec::new() },
        value,
        utility_units: "usd".to_owned(),
    }
}

/// `per_benefit * benefit - cost`: the cost carries the unit of account.
fn one_rate_model(lower: f64, upper: f64) -> ValueModel {
    model(
        vec![attribute("benefit", "events"), attribute("cost", "usd")],
        vec![parameter("per_benefit", lower, upper)],
        E::Sub(b(mul(E::Parameter(0), E::Attribute(0))), b(E::Attribute(1))),
    )
}

/// `per_benefit * benefit + per_harm * harm - cost`.
fn two_rate_model() -> ValueModel {
    model(
        three_attributes(),
        vec![parameter("per_benefit", 0.0, 50.0), parameter("per_harm", -50.0, 0.0)],
        E::Sub(
            b(add(mul(E::Parameter(0), E::Attribute(0)), mul(E::Parameter(1), E::Attribute(1)))),
            b(E::Attribute(2)),
        ),
    )
}

/// `cost_weight * cost + per_benefit * benefit + per_harm * harm` with `cost_weight` fixed at -1.
fn fixed_weight_model() -> ValueModel {
    model(
        three_attributes(),
        vec![
            parameter("cost_weight", -1.0, -1.0),
            parameter("per_benefit", 0.0, 50.0),
            parameter("per_harm", -50.0, 0.0),
        ],
        add(
            add(mul(E::Parameter(0), E::Attribute(2)), mul(E::Parameter(1), E::Attribute(0))),
            mul(E::Parameter(2), E::Attribute(1)),
        ),
    )
}

fn three_attributes() -> Vec<Attribute> {
    vec![attribute("benefit", "events"), attribute("harm", "events"), attribute("cost", "usd")]
}

fn record(id: String, decider: String, options: Vec<Vec<f64>>, chosen: usize) -> ChoiceRecord {
    ChoiceRecord {
        observation_rows: vec![format!("row-{id}")],
        options: options
            .into_iter()
            .enumerate()
            .map(|(i, means)| ChoiceOption {
                id: format!("o{i}"),
                belief: OptionBelief::Means(means),
            })
            .collect(),
        chosen: format!("o{chosen}"),
        belief_source: BeliefSource::RecordedForecast { source_id: format!("forecast-{id}") },
        id,
        decider,
    }
}

fn record_set(model: &ValueModel, records: Vec<ChoiceRecord>, pooling: Pooling) -> ChoiceRecordSet {
    ChoiceRecordSet {
        attributes: model.attributes.clone(),
        population: "clinic patients".to_owned(),
        records,
        pooling,
    }
}

fn pooled() -> Pooling {
    Pooling::Pooled { assumption_id: "assumption.shared_preferences".to_owned() }
}

/// How simulated options are drawn and valued.
struct Simulation<'a> {
    /// Attribute `k` of each option is uniform on `(0, scales[k])`.
    scales: &'a [f64],
    /// The true value of an option from its attributes.
    value: &'a dyn Fn(&[f64]) -> f64,
    /// True choice consistency.
    lambda: f64,
    /// Options per record.
    n_options: usize,
    /// Attributes that take one common value across the options of a record.
    shared: &'a [usize],
}

/// Records whose choices follow the logit model: the chosen option maximises
/// `lambda * V + Gumbel noise`.
fn simulate(
    simulation: &Simulation<'_>,
    rng: &mut Xorshift,
    n: usize,
    prefix: &str,
    decider: &dyn Fn(usize) -> String,
) -> Vec<ChoiceRecord> {
    (0..n)
        .map(|r| {
            let common: Vec<f64> = simulation.scales.iter().map(|s| s * rng.uniform()).collect();
            let options: Vec<Vec<f64>> = (0..simulation.n_options)
                .map(|_| {
                    simulation
                        .scales
                        .iter()
                        .enumerate()
                        .map(|(k, s)| {
                            if simulation.shared.contains(&k) {
                                common[k]
                            } else {
                                s * rng.uniform()
                            }
                        })
                        .collect()
                })
                .collect();
            let chosen = options
                .iter()
                .map(|o| simulation.lambda * (simulation.value)(o.as_slice()) + rng.gumbel())
                .enumerate()
                .max_by(|a, c| a.1.total_cmp(&c.1))
                .map_or(0, |(i, _)| i);
            record(format!("{prefix}{r}"), decider(r), options, chosen)
        })
        .collect()
}

fn one_rate_simulation(value: &dyn Fn(&[f64]) -> f64) -> Simulation<'_> {
    Simulation { scales: &[2.0, 8.0], value, lambda: 1.0, n_options: 3, shared: &[] }
}

fn single(_: usize) -> String {
    "clinician".to_owned()
}

fn pooled_estimate(result: ChoiceModelResult) -> ChoiceModelEstimate {
    assert!(result.per_decider.is_empty());
    *result.pooled.expect("pooled estimate")
}

fn refusal(model: &ValueModel, records: &ChoiceRecordSet, options: ChoiceModelOptions) -> String {
    estimate_choice_model(model, records, options).expect_err("refused").detail().to_owned()
}

fn quick() -> ChoiceModelOptions {
    ChoiceModelOptions { n_draws: 200, ..ChoiceModelOptions::default() }
}

#[test]
fn two_rates_and_consistency_are_recovered_from_simulated_choices() {
    let truth = [4.0, -2.0];
    let value = |x: &[f64]| truth[0] * x[0] + truth[1] * x[1] - x[2];
    let simulation = Simulation {
        scales: &[2.0, 2.0, 8.0],
        value: &value,
        lambda: 1.0,
        n_options: 3,
        shared: &[],
    };
    let model = two_rate_model();
    let records = record_set(
        &model,
        simulate(&simulation, &mut Xorshift::new(11), 2000, "r", &single),
        pooled(),
    );
    let estimate = pooled_estimate(estimate_choice_model(&model, &records, quick()).unwrap());
    assert!((estimate.consistency - 1.0).abs() < 4.0 * estimate.consistency_standard_error);
    assert_eq!(estimate.parameters.len(), 2);
    for (found, want) in estimate.parameters.iter().zip(truth) {
        assert!((found.estimate - want).abs() < 4.0 * found.standard_error, "{found:?}");
        assert!(found.standard_error > 0.0 && found.standard_error < 0.5);
        assert!(found.inside_declared);
        assert!(found.interval.0 < found.estimate && found.estimate < found.interval.1);
    }
    assert_eq!(estimate.parameters[0].parameter_id, "per_benefit");
    assert_eq!(estimate.parameters[1].parameter_id, "per_harm");
    assert!(estimate.not_learnable.is_empty());
    assert_eq!(estimate.covariance_kind, CovarianceKind::Model);
    assert_eq!(estimate.interval_calibration, DistributionCalibration::NotMeasured);
    assert_eq!((estimate.n_records, estimate.n_deciders), (2000, 1));
    assert!(estimate.log_likelihood > estimate.null_log_likelihood);
    assert_eq!(estimate.model_identity, model.identity().unwrap());
    assert_eq!(estimate.records_digest, records.digest().unwrap());
    assert_eq!(estimate.decider, None);
    assert_eq!(
        estimate.measure.origin,
        MeasureOrigin::ChoiceEstimate { estimate_identity: estimate.identity.clone() }
    );
    assert!(estimate.measure.weights.is_none() && estimate.measure.unresolved.is_empty());
    assert!(estimate.measure.points.iter().all(|p| model.preferences.contains(p, 1e-9)));
    let kept = as_f64(estimate.measure.points.len()) / 200.0;
    assert!(close(kept + estimate.share_outside_declared, 1.0, 1e-12));
    let mut sorted = estimate.assumptions.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(estimate.assumptions, sorted);
    for id in [ASSUMPTION_LOGIT_CHOICE, "assumption.shared_preferences", "declared_per_harm"] {
        assert!(estimate.assumptions.iter().any(|a| a == id), "{id}");
    }
}

/// Engineering sanity check of the delta-method standard error and the normal interval over
/// repeated simulated samples; this is not a coverage measurement.
#[test]
fn standard_error_tracks_the_spread_of_estimates_over_replications() {
    let value = |x: &[f64]| 4.0 * x[0] - x[1];
    let simulation = one_rate_simulation(&value);
    let model = one_rate_model(0.0, 50.0);
    let options = ChoiceModelOptions { n_draws: 20, ..ChoiceModelOptions::default() };
    let mut rng = Xorshift::new(2024);
    let replications = 200;
    let mut estimates = Vec::with_capacity(replications);
    let mut errors = Vec::with_capacity(replications);
    let mut covered = 0;
    for _ in 0..replications {
        let records =
            record_set(&model, simulate(&simulation, &mut rng, 300, "r", &single), pooled());
        let estimate = pooled_estimate(estimate_choice_model(&model, &records, options).unwrap());
        let found = &estimate.parameters[0];
        if found.interval.0 <= 4.0 && 4.0 <= found.interval.1 {
            covered += 1;
        }
        estimates.push(found.estimate);
        errors.push(found.standard_error);
    }
    let n = as_f64(replications);
    let mean = estimates.iter().sum::<f64>() / n;
    let spread = (estimates.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
    let typical_error = errors.iter().sum::<f64>() / n;
    let ratio = typical_error / spread;
    assert!((0.7..=1.4).contains(&ratio), "standard error / spread = {ratio}");
    assert!(as_f64(covered) / n >= 0.85, "{covered} of {replications} intervals hold the truth");
}

/// Four records of one cost difference, the cheaper option chosen three times: the consistency
/// is `ln 3` and its information is `4 * (3/4) * (1/4) = 3/4`.
fn cost_only_records(model: &ValueModel, cheaper_chosen: usize) -> ChoiceRecordSet {
    let records = (0..4)
        .map(|r| {
            let benefit = f64::from(u8::try_from(r).unwrap());
            let chosen = usize::from(r >= cheaper_chosen);
            record(
                format!("r{r}"),
                "clinician".to_owned(),
                vec![vec![benefit, 0.0], vec![benefit, 1.0]],
                chosen,
            )
        })
        .collect();
    record_set(model, records, pooled())
}

#[test]
fn consistency_from_one_cost_column_matches_the_hand_derivation() {
    let model = one_rate_model(0.0, 50.0);
    let records = cost_only_records(&model, 3);
    let options = ChoiceModelOptions { n_draws: 4000, ..ChoiceModelOptions::default() };
    let estimate = pooled_estimate(estimate_choice_model(&model, &records, options).unwrap());
    assert!(close(estimate.consistency, 3.0_f64.ln(), 1e-9));
    assert!(close(estimate.consistency_standard_error, (4.0_f64 / 3.0).sqrt(), 1e-9));
    assert!(estimate.parameters.is_empty());
    assert_eq!(estimate.not_learnable, vec!["per_benefit".to_owned()]);
    assert_eq!(estimate.measure.unresolved, vec![0]);
    assert!(estimate.measure.points.iter().all(|p| p.len() == 1 && close(p[0], 25.0, 0.0)));
    // A draw is dropped only for a non-positive consistency: P(N(ln 3, 4/3) <= 0) ~ 0.171.
    assert!((0.14..0.20).contains(&estimate.share_outside_declared));
}

#[test]
fn a_term_that_never_differs_is_not_learnable_and_unresolved() {
    let value = |x: &[f64]| 4.0 * x[0] - 3.0 * x[1] - x[2];
    let simulation = Simulation {
        scales: &[2.0, 2.0, 8.0],
        value: &value,
        lambda: 1.0,
        n_options: 3,
        shared: &[1],
    };
    let mut model = fixed_weight_model();
    // Ignored: it touches the unresolved parameter, whose placeholder midpoint violates it.
    model.preferences.relations.push(LinearRelation {
        id: "net_benefit_non_negative".to_owned(),
        terms: vec![(1, 1.0), (2, 1.0)],
        sense: RelationSense::AtLeast,
        bound: 0.0,
        origin: RelationOrigin::Declared,
    });
    let records = record_set(
        &model,
        simulate(&simulation, &mut Xorshift::new(5), 600, "r", &single),
        pooled(),
    );
    let estimate = pooled_estimate(estimate_choice_model(&model, &records, quick()).unwrap());
    let ids: Vec<&str> = estimate.parameters.iter().map(|p| p.parameter_id.as_str()).collect();
    assert_eq!(ids, vec!["per_benefit"]);
    assert_eq!(estimate.not_learnable, vec!["per_harm".to_owned()]);
    assert_eq!(estimate.measure.unresolved, vec![2]);
    assert!(
        (estimate.parameters[0].estimate - 4.0).abs() < 4.0 * estimate.parameters[0].standard_error
    );
    assert!(!estimate.measure.points.is_empty());
    for point in &estimate.measure.points {
        assert_eq!(point.len(), 3);
        assert!(close(point[0], -1.0, 0.0) && close(point[2], -25.0, 0.0));
    }
    assert!(estimate.share_outside_declared < 0.05);
}

#[test]
fn relations_on_learnable_parameters_drop_draws() {
    let value = |x: &[f64]| 4.0 * x[0] - 2.0 * x[1] - x[2];
    let simulation = Simulation {
        scales: &[2.0, 2.0, 8.0],
        value: &value,
        lambda: 1.0,
        n_options: 3,
        shared: &[],
    };
    let mut model = two_rate_model();
    let records = record_set(
        &model,
        simulate(&simulation, &mut Xorshift::new(17), 800, "r", &single),
        pooled(),
    );
    let plain = pooled_estimate(estimate_choice_model(&model, &records, quick()).unwrap());
    let net = plain.parameters[0].estimate + plain.parameters[1].estimate;
    model.preferences.relations.push(LinearRelation {
        id: "net_at_least_estimate".to_owned(),
        terms: vec![(0, 1.0), (1, 1.0)],
        sense: RelationSense::AtLeast,
        bound: net,
        origin: RelationOrigin::Declared,
    });
    let cut = pooled_estimate(estimate_choice_model(&model, &records, quick()).unwrap());
    assert!((0.3..0.7).contains(&cut.share_outside_declared), "{}", cut.share_outside_declared);
    assert!(cut.measure.points.iter().all(|p| p[0] + p[1] >= net - 1e-9));
}

#[test]
fn delta_method_error_matches_a_finite_difference_gradient() {
    let value = |x: &[f64]| 4.0 * x[0] - 2.0 * x[1] - x[2];
    let simulation = Simulation {
        scales: &[2.0, 2.0, 8.0],
        value: &value,
        lambda: 1.0,
        n_options: 3,
        shared: &[],
    };
    let model = two_rate_model();
    let records = record_set(
        &model,
        simulate(&simulation, &mut Xorshift::new(3), 500, "r", &single),
        pooled(),
    );
    let estimate = pooled_estimate(estimate_choice_model(&model, &records, quick()).unwrap());
    let lambda = estimate.consistency;
    let mut beta = vec![lambda];
    beta.extend(estimate.parameters.iter().map(|p| lambda * p.estimate));
    let p = beta.len();
    let covariance = &estimate.coefficient_covariance;
    assert_eq!(covariance.len(), p * p);
    assert!(close(estimate.consistency_standard_error, covariance[0].sqrt(), 1e-12));
    let h = 1e-6;
    for (k, found) in estimate.parameters.iter().enumerate().map(|(c, f)| (c + 1, f)) {
        let gradient: Vec<f64> = (0..p)
            .map(|i| {
                let (mut up, mut down) = (beta.clone(), beta.clone());
                up[i] += h;
                down[i] -= h;
                (up[k] / up[0] - down[k] / down[0]) / (2.0 * h)
            })
            .collect();
        let variance: f64 = (0..p)
            .flat_map(|i| (0..p).map(move |j| (i, j)))
            .map(|(i, j)| gradient[i] * covariance[i * p + j] * gradient[j])
            .sum();
        assert!(close(found.standard_error, variance.sqrt(), 1e-6), "{found:?}");
        let z = (found.interval.1 - found.estimate) / found.standard_error;
        assert!(close(z, 1.959_963_984_540_054, 1e-9));
    }
}

fn round_robin(r: usize) -> String {
    format!("d{}", r % 10)
}

#[test]
fn clustered_covariance_is_used_for_several_deciders_and_matches_a_direct_fit() {
    let value = |x: &[f64]| 4.0 * x[0] - x[1];
    let simulation = one_rate_simulation(&value);
    let model = one_rate_model(0.0, 50.0);
    let simulated = simulate(&simulation, &mut Xorshift::new(99), 600, "r", &round_robin);
    let records = record_set(&model, simulated.clone(), pooled());
    let estimate = pooled_estimate(estimate_choice_model(&model, &records, quick()).unwrap());
    assert_eq!(estimate.covariance_kind, CovarianceKind::ClusteredByDecider);
    assert_eq!(estimate.n_deciders, 10);

    let mut features = Vec::new();
    let mut set_starts = vec![0];
    let mut chosen = Vec::new();
    let mut clusters = Vec::new();
    for record in &simulated {
        for option in &record.options {
            let OptionBelief::Means(x) = &option.belief else { unreachable!() };
            features.extend([-x[1], x[0]]);
        }
        set_starts.push(set_starts.last().unwrap() + record.options.len());
        chosen.push(record.options.iter().position(|o| o.id == record.chosen).unwrap());
        clusters.push(record.decider[1..].parse::<u32>().unwrap());
    }
    let design = ConditionalLogitDesign {
        features: &features,
        n_features: 2,
        set_starts: &set_starts,
        chosen: &chosen,
        clusters: Some(&clusters),
    };
    let fit = fit_conditional_logit(&design, ConditionalLogitOptions::default()).unwrap();
    assert!(close(estimate.consistency, fit.coefficients[0], 1e-12));
    let direct = fit.cluster_covariance.unwrap();
    for (found, want) in estimate.coefficient_covariance.iter().zip(&direct) {
        assert!(close(*found, *want, 1e-10));
    }

    let per_decider = record_set(&model, simulated, Pooling::PerDecider);
    let result = estimate_choice_model(&model, &per_decider, quick()).unwrap();
    assert!(result.pooled.is_none());
    assert_eq!(result.per_decider.len(), 10);
    for (decider, outcome) in &result.per_decider {
        let EstimateOutcome::Estimated(one) = outcome else { panic!("{decider}: {outcome:?}") };
        assert_eq!(one.covariance_kind, CovarianceKind::Model);
        assert_eq!(one.decider.as_deref(), Some(decider.as_str()));
        assert_eq!((one.n_records, one.n_deciders), (60, 1));
    }
}

/// Choices that always agree with `per_benefit = 5`: no finite maximum-likelihood estimate.
fn separated_records(decider: &str) -> Vec<ChoiceRecord> {
    [(3.0, 1.0, 1), (7.0, 1.0, 0), (1.0, 0.5, 1), (4.0, 0.5, 0)]
        .into_iter()
        .enumerate()
        .map(|(r, (cost, benefit, chosen))| {
            record(
                format!("{decider}-s{r}"),
                decider.to_owned(),
                vec![vec![0.0, 0.0], vec![benefit, cost]],
                chosen,
            )
        })
        .collect()
}

#[test]
fn per_decider_fits_refuse_on_their_own() {
    let value = |x: &[f64]| 4.0 * x[0] - x[1];
    let simulation = one_rate_simulation(&value);
    let model = one_rate_model(0.0, 50.0);
    let alpha = |_: usize| "alpha".to_owned();
    let mut records = simulate(&simulation, &mut Xorshift::new(8), 300, "a", &alpha);
    records.extend(separated_records("beta"));
    let set = record_set(&model, records, Pooling::PerDecider);
    let result = estimate_choice_model(&model, &set, quick()).unwrap();
    assert!(result.pooled.is_none());
    let names: Vec<&str> = result.per_decider.iter().map(|(d, _)| d.as_str()).collect();
    assert_eq!(names, vec!["alpha", "beta"]);
    let EstimateOutcome::Estimated(alpha) = &result.per_decider[0].1 else {
        panic!("alpha refused")
    };
    assert_eq!(alpha.n_records, 300);
    assert_eq!(alpha.records_digest, set.digest().unwrap());
    assert_eq!(
        result.per_decider[1].1,
        EstimateOutcome::Refused(ChoiceModelError::Separated { decider: Some("beta".to_owned()) })
    );
}

#[test]
fn a_fixed_seed_gives_the_same_estimate_and_record_order_does_not_matter() {
    let value = |x: &[f64]| 4.0 * x[0] - x[1];
    let simulation = one_rate_simulation(&value);
    let model = one_rate_model(0.0, 50.0);
    let simulated = simulate(&simulation, &mut Xorshift::new(21), 400, "r", &single);
    let records = record_set(&model, simulated.clone(), pooled());
    let first = estimate_choice_model(&model, &records, quick()).unwrap();
    assert_eq!(first, estimate_choice_model(&model, &records, quick()).unwrap());

    let other_seed = ChoiceModelOptions { seed: 7, ..quick() };
    let reseeded = pooled_estimate(estimate_choice_model(&model, &records, other_seed).unwrap());
    let first = pooled_estimate(first);
    assert_ne!(reseeded.identity, first.identity);
    assert_ne!(reseeded.measure.points, first.measure.points);
    assert_eq!(reseeded.parameters, first.parameters);

    let mut reversed = simulated;
    reversed.reverse();
    let reordered = record_set(&model, reversed, pooled());
    let again = pooled_estimate(estimate_choice_model(&model, &reordered, quick()).unwrap());
    assert_eq!(again.identity, first.identity);
    assert!(close(again.consistency, first.consistency, 1e-9));
    assert!(close(again.parameters[0].estimate, first.parameters[0].estimate, 1e-9));
    assert!(close(again.parameters[0].standard_error, first.parameters[0].standard_error, 1e-8));
    assert_eq!(again.measure.points.len(), first.measure.points.len());
}

#[test]
fn invalid_options_model_and_records_are_refused() {
    let model = one_rate_model(0.0, 50.0);
    let records = cost_only_records(&model, 3);
    for level in [0.0, 1.0, f64::NAN] {
        let options = ChoiceModelOptions { level, ..quick() };
        assert_eq!(
            refusal(&model, &records, options),
            reason_details::preference_choice_model::LEVEL_INVALID
        );
    }
    let options = ChoiceModelOptions { n_draws: 0, ..quick() };
    assert_eq!(
        refusal(&model, &records, options),
        reason_details::preference_choice_model::DRAWS_MISSING
    );
    let fit =
        ConditionalLogitOptions { gradient_tolerance: 0.0, ..ConditionalLogitOptions::default() };
    let options = ChoiceModelOptions { fit, ..quick() };
    assert_eq!(
        refusal(&model, &records, options),
        reason_details::preference_choice_model::FIT_OPTIONS_INVALID
    );

    let mut blank = model.clone();
    blank.utility_units = " ".to_owned();
    assert_eq!(
        refusal(&blank, &records, quick()),
        reason_details::preference_choice_model::MODEL_INVALID
    );

    let mut renamed = records;
    renamed.attributes[0].id = "gain".to_owned();
    let error = estimate_choice_model(&model, &renamed, quick()).unwrap_err();
    assert!(matches!(
        error,
        ChoiceModelError::Records(ChoiceRecordError::AttributesDiffer { index: 0, .. })
    ));
}

#[test]
fn missing_unit_of_account_and_nothing_to_learn_are_refused() {
    let free_cost = model(
        vec![attribute("benefit", "events"), attribute("cost", "usd")],
        vec![parameter("per_benefit", 0.0, 50.0), parameter("per_cost", -5.0, 0.0)],
        add(mul(E::Parameter(0), E::Attribute(0)), mul(E::Parameter(1), E::Attribute(1))),
    );
    let value = |x: &[f64]| 4.0 * x[0] - x[1];
    let simulation = one_rate_simulation(&value);
    let records = record_set(
        &free_cost,
        simulate(&simulation, &mut Xorshift::new(1), 50, "r", &single),
        pooled(),
    );
    assert_eq!(
        refusal(&free_cost, &records, quick()),
        reason_details::preference_choice_model::UNIT_OF_ACCOUNT_MISSING
    );

    let model = one_rate_model(0.0, 50.0);
    let identical = (0..3)
        .map(|r| record(format!("r{r}"), "clinician".to_owned(), vec![vec![1.0, 2.0]; 2], 0))
        .collect();
    let records = record_set(&model, identical, pooled());
    assert_eq!(
        refusal(&model, &records, quick()),
        reason_details::preference_choice_model::NOTHING_IDENTIFIED
    );
}

#[test]
fn rank_deficiency_names_the_parameters_involved() {
    let model = two_rate_model();
    let value = |x: &[f64]| 4.0 * x[0] - 2.0 * x[1] - x[2];
    let simulation = Simulation {
        scales: &[2.0, 2.0, 8.0],
        value: &value,
        lambda: 1.0,
        n_options: 3,
        shared: &[],
    };
    let simulated = simulate(&simulation, &mut Xorshift::new(4), 100, "r", &single);
    let rewrite = |harm: &dyn Fn(&[f64]) -> f64| -> ChoiceRecordSet {
        let mut records = simulated.clone();
        for record in &mut records {
            for option in &mut record.options {
                if let OptionBelief::Means(x) = &mut option.belief {
                    let rewritten = harm(x.as_slice());
                    x[1] = rewritten;
                }
            }
        }
        record_set(&model, records, pooled())
    };
    let twice_benefit = rewrite(&|x| 2.0 * x[0]);
    assert_eq!(
        estimate_choice_model(&model, &twice_benefit, quick()).unwrap_err(),
        ChoiceModelError::RankDeficient {
            decider: None,
            unit_of_account: false,
            parameters: vec!["per_benefit".to_owned(), "per_harm".to_owned()],
        }
    );
    let as_cost = rewrite(&|x| x[2]);
    assert_eq!(
        estimate_choice_model(&model, &as_cost, quick()).unwrap_err(),
        ChoiceModelError::RankDeficient {
            decider: None,
            unit_of_account: true,
            parameters: vec!["per_harm".to_owned()],
        }
    );
}

#[test]
fn separation_and_negative_consistency_are_refused() {
    let model = one_rate_model(0.0, 50.0);
    let separated = record_set(&model, separated_records("clinician"), pooled());
    let error = estimate_choice_model(&model, &separated, quick()).unwrap_err();
    assert_eq!(error, ChoiceModelError::Separated { decider: None });
    assert!(error.to_string().contains("compatible-set method"));

    let against = cost_only_records(&model, 1);
    match estimate_choice_model(&model, &against, quick()).unwrap_err() {
        ChoiceModelError::ConsistencyNotPositive { decider: None, consistency } => {
            assert!(close(consistency, -(3.0_f64.ln()), 1e-9));
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn fit_budget_covariance_and_declared_set_refusals() {
    let value = |x: &[f64]| 4.0 * x[0] - x[1];
    let simulation = one_rate_simulation(&value);
    let model = one_rate_model(0.0, 50.0);
    let simulated =
        simulate(&simulation, &mut Xorshift::new(31), 200, "r", &|r| format!("d{}", r % 2));

    let one = |r: &ChoiceRecord| ChoiceRecord { decider: "clinician".to_owned(), ..r.clone() };
    let single_decider = record_set(&model, simulated.iter().map(one).collect(), pooled());
    let fit = ConditionalLogitOptions { max_iterations: 0, ..ConditionalLogitOptions::default() };
    let options = ChoiceModelOptions { fit, ..quick() };
    assert_eq!(
        refusal(&model, &single_decider, options),
        reason_details::preference_choice_model::NOT_CONVERGED
    );

    // Two clusters' scores sum to the zero gradient, so the sandwich has rank one.
    let two_deciders = record_set(&model, simulated, pooled());
    assert_eq!(
        estimate_choice_model(&model, &two_deciders, quick()).unwrap_err(),
        ChoiceModelError::CovarianceNotPositiveDefinite {
            decider: None,
            kind: CovarianceKind::ClusteredByDecider,
        }
    );

    let narrow = one_rate_model(0.0, 1.0);
    let records = record_set(&narrow, single_decider.records, pooled());
    assert_eq!(
        estimate_choice_model(&narrow, &records, quick()).unwrap_err(),
        ChoiceModelError::NoDrawInsideSet { decider: None, drawn: 200 }
    );
}
