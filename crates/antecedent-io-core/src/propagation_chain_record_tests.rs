//! The retained propagation record: known-truth chains, shared evidence counted
//! once, a nonlinear utility against a closed form and a Monte Carlo reference,
//! changed assumptions reaching exactly their dependents, a future study that
//! updates the posterior before the decision is re-evaluated, and the consumer
//! replaying all of it from the stored bytes.

use antecedent_contract::propagation_chain::{POOL_WITH_JOINT_COVARIANCE, SUPPLY_POSTERIOR_DRAWS};
use antecedent_stats::normal_ppf;

use super::*;

const ROWS: usize = 1597;
const STRIDE: usize = 987;
const TABLE: &str = "joint-table";

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(17)
}

fn normal_grid(mean: f64, sd: f64) -> Vec<f64> {
    (0..ROWS).map(|i| mean + sd * normal_ppf((i as f64 + 0.5) / ROWS as f64)).collect()
}

/// A Fibonacci-lattice partner of [`normal_grid`]: draws of an independent normal.
fn independent_grid(mean: f64, sd: f64) -> Vec<f64> {
    let grid = normal_grid(mean, sd);
    (0..ROWS).map(|i| grid[(i * STRIDE) % ROWS]).collect()
}

fn near(got: f64, want: f64, tolerance: f64) {
    assert!((got - want).abs() <= tolerance, "{got} vs {want} (tolerance {tolerance})");
}

fn value(result: &ChainStageResultWire, name: &str) -> f64 {
    result.value(name).unwrap_or_else(|| panic!("no value `{name}` on {}", result.stage))
}

/// measurement mapping -> transport -> derived net benefit -> exponential
/// utility, over a posterior of the effect and a posterior of the cost, plus a
/// derived quantity of the cost alone.
fn release_draft() -> PropagationChainRecordWire {
    let sources = vec![
        ChainSourceWire::draws("effect", LawMeaning::Posterior, TABLE, &normal_grid(0.5, 0.6))
            .with_factor(SharedFactorKind::Row, "trial-rows"),
        ChainSourceWire::draws("cost", LawMeaning::Posterior, TABLE, &independent_grid(0.2, 0.3))
            .with_factor(SharedFactorKind::Prior, "cost-prior"),
    ];
    let stages = vec![
        ChainStageWire::new(
            "mapped",
            StageKind::MeasurementMapping,
            &["effect"],
            ChainOpWire::affine(ChainCoefficientWire::fixed(0.8), ChainCoefficientWire::fixed(0.1)),
        ),
        ChainStageWire::new(
            "transported",
            StageKind::Transport,
            &["mapped"],
            ChainOpWire::affine(ChainCoefficientWire::fixed(0.9), ChainCoefficientWire::fixed(0.0)),
        ),
        ChainStageWire::new(
            "net",
            StageKind::DerivedQuantity,
            &["transported", "cost"],
            ChainOpWire::binary("difference"),
        ),
        ChainStageWire::new(
            "utility",
            StageKind::DecisionUtility,
            &["net"],
            ChainOpWire::unary(ChainUnaryWire::exponential_utility(1.0)),
        )
        .with_decision_reference(0.0),
        ChainStageWire::new(
            "cost_doubled",
            StageKind::DerivedQuantity,
            &["cost"],
            ChainOpWire::affine(ChainCoefficientWire::fixed(2.0), ChainCoefficientWire::fixed(0.0)),
        ),
    ];
    PropagationChainRecordWire::draft(
        TableConstruction::JointDraws,
        ChainReading::Spread,
        "utility",
        sources,
        stages,
    )
}

fn release() -> PropagationChainRecordWire {
    PropagationChainRecordWire::seal(release_draft(), &ctx()).expect("seals")
}

fn stage<'a>(record: &'a PropagationChainRecordWire, id: &str) -> &'a ChainStageResultWire {
    record.results.iter().find(|r| r.stage == id).expect("stage result")
}

#[test]
fn a_four_stage_chain_matches_the_closed_form_and_not_the_utility_of_the_mean() {
    let record = release();
    assert_eq!(record.label, "uncalibrated_spread");
    // net = 0.9 (0.8 effect + 0.1) - cost ~ N(0.25, 0.72^2 0.36 + 0.09).
    let mean = 0.9 * (0.8 * 0.5 + 0.1) - 0.2;
    let variance = 0.72_f64.powi(2) * 0.36 + 0.09;
    let truth = 1.0 - (-mean + 0.5 * variance).exp();
    let utility = stage(&record, "utility");
    near(value(utility, "mean"), truth, 5e-3);
    let at_mean = 1.0 - (-mean).exp();
    assert!((at_mean - truth).abs() > 0.1, "the utility of the mean is a different number");
    let net = stage(&record, "net");
    near(value(net, "mean"), mean, 1e-3);
    near(value(net, "variance"), variance, 5e-3);
    // The decision is read against the Monte Carlo band: the gap is well beyond it.
    assert_eq!(utility.count("verdict_code"), Some(0));
}

#[test]
fn the_utility_agrees_with_an_independent_monte_carlo_reference() {
    let record = release();
    let effect = normal_grid(0.5, 0.6);
    let cost = independent_grid(0.2, 0.3);
    let reference: f64 = effect
        .iter()
        .zip(&cost)
        .map(|(e, c)| 1.0 - (-(0.9 * (0.8 * e + 0.1) - c)).exp())
        .sum::<f64>()
        / ROWS as f64;
    near(value(stage(&record, "utility"), "mean"), reference, 1e-12);
}

#[test]
fn the_consumer_replays_the_stored_bytes_to_identical_results() {
    let record = release();
    let bytes = record.export().expect("exports");
    let decoded = PropagationChainRecordWire::decode(&bytes).expect("decodes");
    assert_eq!(decoded, record);
    let consumed = decoded.consume(Some(&record.identity), &ctx()).expect("consumes");
    assert_eq!(consumed.results, record.results);
    assert_eq!(consumed.label, "uncalibrated_spread");
    assert_eq!(consumed.plan.dependents_of("cost"), vec!["net", "utility", "cost_doubled"]);
    assert!(consumed.answer().is_some());
    // The identity the consumer holds independently must match.
    let other = "0".repeat(64);
    assert_eq!(decoded.consume(Some(&other), &ctx()), Err(PropagationChainError::IdentityMismatch));
}

#[test]
fn edits_are_refused_resealed_or_not() {
    let record = release();
    // An edited draw changes the identity.
    let mut edited = record.clone();
    edited.sources[0].values[3] ^= 1;
    assert_eq!(edited.consume(None, &ctx()), Err(PropagationChainError::IdentityMismatch));
    // An edited result is not covered by the identity and does not replay.
    let mut result = record.clone();
    result.results[1].values[0].1 ^= 1;
    assert_eq!(result.consume(None, &ctx()), Err(PropagationChainError::ResultMismatch));
    // A resealed operator edit leaves results that no longer replay.
    let mut resealed = record.clone();
    resealed.stages[0].op =
        ChainOpWire::affine(ChainCoefficientWire::fixed(0.7), ChainCoefficientWire::fixed(0.1));
    resealed.identity = propagation_chain_identity(&resealed).expect("identity");
    assert_eq!(resealed.consume(None, &ctx()), Err(PropagationChainError::ResultMismatch));
    // A resealed label edit no longer matches the label the rule yields.
    let mut relabelled = record.clone();
    relabelled.label = "point_only".to_owned();
    relabelled.identity = propagation_chain_identity(&relabelled).expect("identity");
    assert_eq!(relabelled.consume(None, &ctx()), Err(PropagationChainError::IdentityMismatch));
    // Declared factors are part of the identity: dropping the shared row changes it.
    let mut dropped = record.clone();
    dropped.sources[0].factors.clear();
    assert_eq!(dropped.consume(None, &ctx()), Err(PropagationChainError::IdentityMismatch));
}

#[test]
fn the_identity_ignores_the_order_of_declared_factors_but_not_their_content() {
    let mut a = release_draft();
    a.sources[0] = a.sources[0]
        .clone()
        .with_factor(SharedFactorKind::Model, "m1")
        .with_factor(SharedFactorKind::Cluster, "c1");
    let mut b = a.clone();
    b.sources[0].factors.reverse();
    let sealed_a = PropagationChainRecordWire::seal(a.clone(), &ctx()).expect("seals");
    let sealed_b = PropagationChainRecordWire::seal(b, &ctx()).expect("seals");
    assert_eq!(sealed_a.identity, sealed_b.identity);
    a.sources[0].factors[1].id = "m2".to_owned();
    let sealed_c = PropagationChainRecordWire::seal(a, &ctx()).expect("seals");
    assert_ne!(sealed_a.identity, sealed_c.identity);
}

#[test]
fn another_version_or_feature_is_refused_before_the_payload_is_read() {
    let record = release();
    let mut future = record.clone();
    future.version = 2;
    let bytes = crate::to_cbor(&future).expect("encodes");
    assert_eq!(
        PropagationChainRecordWire::decode(&bytes),
        Err(PropagationChainError::UnsupportedVersion { version: 2 })
    );
    let mut foreign = record;
    foreign.required_features = vec!["something_else".to_owned()];
    let bytes = crate::to_cbor(&foreign).expect("encodes");
    assert_eq!(
        PropagationChainRecordWire::decode(&bytes),
        Err(PropagationChainError::UnsupportedSemantics("required features"))
    );
}

// --------------------------------------------------------- shared evidence

/// The 2^3 sign design: `T`, `e1` and `e2` are orthogonal columns with sample
/// variance 8/7. Estimate A is `10 + T + e1` and estimate B is `10 + T + e2`:
/// they share the evidence `T`.
fn overlapping_estimates() -> (Vec<f64>, Vec<f64>) {
    let mut a = Vec::new();
    let mut b = Vec::new();
    for t in [-1.0, 1.0] {
        for e1 in [-1.0, 1.0] {
            for e2 in [-1.0, 1.0] {
                a.push(10.0 + t + e1);
                b.push(10.0 + t + e2);
            }
        }
    }
    (a, b)
}

fn pool_draft(dependent: bool, declare_overlap: bool) -> PropagationChainRecordWire {
    let (a, b) = overlapping_estimates();
    let mut first = ChainSourceWire::draws("study_a", LawMeaning::Sampling, TABLE, &a);
    let mut second = ChainSourceWire::draws("study_b", LawMeaning::Sampling, TABLE, &b);
    if declare_overlap {
        first = first.with_factor(SharedFactorKind::Row, "trial-rows");
        second = second.with_factor(SharedFactorKind::Row, "trial-rows");
    } else {
        first = first.with_factor(SharedFactorKind::Row, "rows-a");
        second = second.with_factor(SharedFactorKind::Row, "rows-b");
    }
    PropagationChainRecordWire::draft(
        TableConstruction::JointDraws,
        ChainReading::Spread,
        "pooled",
        vec![first, second],
        vec![ChainStageWire::new(
            "pooled",
            StageKind::Pooling,
            &["study_a", "study_b"],
            ChainOpWire::pool(dependent),
        )],
    )
}

#[test]
fn an_independent_pool_of_overlapping_evidence_is_refused_with_its_next_attempt() {
    let refusal = PropagationChainRecordWire::seal(pool_draft(false, true), &ctx())
        .expect_err("refused")
        .to_refusal();
    assert_eq!(refusal.code(), "evidence_double_counted");
    assert_eq!(refusal.stage(), Some("pooled"));
    assert_eq!(refusal.next_attempt(), POOL_WITH_JOINT_COVARIANCE);
}

#[test]
fn a_dependent_pool_counts_the_shared_evidence_once() {
    let record = PropagationChainRecordWire::seal(pool_draft(true, true), &ctx()).expect("seals");
    let pooled = stage(&record, "pooled");
    near(value(pooled, "weight_0"), 0.5, 1e-12);
    near(value(pooled, "weight_1"), 0.5, 1e-12);
    // Var A = Var B = 16/7 and Cov = 8/7, so the pooled variance is (v + c) / 2.
    near(value(pooled, "joint_variance"), 12.0 / 7.0, 1e-12);
    near(value(pooled, "variance"), 12.0 / 7.0, 1e-12);
    // Counting T twice would claim 1 / (1/v + 1/v) = 8/7.
    near(value(pooled, "independent_variance"), 8.0 / 7.0, 1e-12);
    assert!(value(pooled, "independent_variance") < value(pooled, "variance"));
    record.consume(None, &ctx()).expect("replays");
}

#[test]
fn an_independent_pool_of_distinct_evidence_is_licensed() {
    let record = PropagationChainRecordWire::seal(pool_draft(false, false), &ctx());
    // The declared rows differ, so the pool is licensed as independent even though the
    // arithmetic of these particular draws is the overlapping design.
    assert!(record.is_ok());
}

// -------------------------------------------------- the joint-law counterexample

fn coupling(x: &[f64], y: &[f64]) -> PropagationChainRecordWire {
    PropagationChainRecordWire::draft(
        TableConstruction::JointDraws,
        ChainReading::Spread,
        "utility",
        vec![
            ChainSourceWire::draws("x", LawMeaning::Posterior, TABLE, x),
            ChainSourceWire::draws("y", LawMeaning::Posterior, TABLE, y),
        ],
        vec![
            ChainStageWire::new(
                "product",
                StageKind::DerivedQuantity,
                &["x", "y"],
                ChainOpWire::binary("product"),
            ),
            ChainStageWire::new(
                "shifted",
                StageKind::DerivedQuantity,
                &["product"],
                ChainOpWire::affine(
                    ChainCoefficientWire::fixed(1.0),
                    ChainCoefficientWire::fixed(-1.0),
                ),
            ),
            ChainStageWire::new(
                "utility",
                StageKind::DecisionUtility,
                &["shifted"],
                ChainOpWire::affine(
                    ChainCoefficientWire::fixed(1.0),
                    ChainCoefficientWire::fixed(0.0),
                ),
            ),
        ],
    )
}

#[test]
fn the_same_marginals_under_different_couplings_give_opposite_decisions() {
    // X and Y each take 0 and 2 with probability 1/2.
    let together = PropagationChainRecordWire::seal(
        coupling(&[0.0, 0.0, 2.0, 2.0], &[0.0, 0.0, 2.0, 2.0]),
        &ctx(),
    )
    .expect("seals");
    let apart = PropagationChainRecordWire::seal(
        coupling(&[0.0, 0.0, 2.0, 2.0], &[2.0, 2.0, 0.0, 0.0]),
        &ctx(),
    )
    .expect("seals");
    near(value(stage(&together, "utility"), "mean"), 1.0, 1e-12);
    near(value(stage(&apart, "utility"), "mean"), -1.0, 1e-12);
}

#[test]
fn draws_of_unrelated_tables_are_not_paired_by_index() {
    let mut draft = coupling(&[0.0, 1.0, 2.0], &[2.0, 1.0, 0.0]);
    draft.sources[1].alignment = Some("another-table".to_owned());
    let refusal = PropagationChainRecordWire::seal(draft, &ctx()).expect_err("refused");
    assert_eq!(refusal.to_refusal().code(), "joint_law_required");
}

// ------------------------------------------------------ changed assumptions

fn bit_values(result: &ChainStageResultWire) -> Vec<(String, u64)> {
    result.values.clone()
}

#[test]
fn withdrawing_a_source_invalidates_exactly_its_dependents() {
    let before = release();
    let after = before.with_change(ChainChangeWire::withdraw("cost"), &ctx()).expect("changes");
    for id in ["mapped", "transported"] {
        assert_eq!(stage(&after, id).status, STATUS_CURRENT);
        assert_eq!(bit_values(stage(&after, id)), bit_values(stage(&before, id)));
    }
    for id in ["net", "utility", "cost_doubled"] {
        let result = stage(&after, id);
        assert_eq!(result.status, STATUS_INVALIDATED);
        assert!(result.values.is_empty());
        assert_eq!(result.invalidated_by, vec!["cost".to_owned()]);
    }
    assert_ne!(after.identity, before.identity);
    after.consume(None, &ctx()).expect("the changed record replays");
    // The other source: the cost-only stage keeps its numbers bit for bit.
    let other = before.with_change(ChainChangeWire::withdraw("effect"), &ctx()).expect("changes");
    assert_eq!(stage(&other, "cost_doubled").status, STATUS_CURRENT);
    assert_eq!(
        bit_values(stage(&other, "cost_doubled")),
        bit_values(stage(&before, "cost_doubled"))
    );
    for id in ["mapped", "transported", "net", "utility"] {
        assert_eq!(stage(&other, id).status, STATUS_INVALIDATED, "{id}");
    }
    assert_eq!(
        before.dependents_of("effect").expect("plan"),
        vec!["mapped", "transported", "net", "utility"]
    );
}

#[test]
fn a_changed_prior_reweights_exactly_the_stages_that_depend_on_it() {
    let before = release();
    // The prior of the effect moves by an exponential tilt: for a normal posterior with
    // sd 0.6 the tilted law is normal with the mean shifted by lambda * variance.
    let lambda = 1.0;
    let log_weights: Vec<f64> = normal_grid(0.5, 0.6).iter().map(|e| lambda * e).collect();
    let after = before
        .with_change(ChainChangeWire::reweight("effect", &log_weights), &ctx())
        .expect("changes");
    let shifted = 0.5 + lambda * 0.36;
    let mapped = stage(&after, "mapped");
    assert_eq!(mapped.status, STATUS_REWEIGHTED);
    near(value(mapped, "mean"), 0.8 * shifted + 0.1, 1.5e-2);
    assert!(value(mapped, "effective_size") < ROWS as f64);
    for id in ["transported", "net", "utility"] {
        assert_eq!(stage(&after, id).status, STATUS_REWEIGHTED, "{id}");
    }
    // The stage that does not read the effect is untouched, bit for bit.
    assert_eq!(stage(&after, "cost_doubled").status, STATUS_CURRENT);
    assert_eq!(
        bit_values(stage(&after, "cost_doubled")),
        bit_values(stage(&before, "cost_doubled"))
    );
    // The utility moved with the new belief: a larger effect raises the net benefit.
    assert!(value(stage(&after, "utility"), "mean") > value(stage(&before, "utility"), "mean"));
    after.consume(None, &ctx()).expect("replays");
    // Changes append and keep their order.
    let twice = after
        .with_change(ChainChangeWire::reweight("cost", &vec![0.0; ROWS]), &ctx())
        .expect("changes");
    assert_eq!(twice.changes.len(), 2);
    assert_eq!(bit_values(stage(&twice, "mapped")), bit_values(stage(&after, "mapped")));
}

#[test]
fn a_change_that_names_no_source_or_fits_no_table_is_refused() {
    let record = release();
    assert!(matches!(
        record.with_change(ChainChangeWire::withdraw("nowhere"), &ctx()),
        Err(PropagationChainError::Malformed(_))
    ));
    assert!(matches!(
        record.with_change(ChainChangeWire::reweight("effect", &[0.0, 1.0]), &ctx()),
        Err(PropagationChainError::Malformed(_))
    ));
    let withdrawn =
        record.with_change(ChainChangeWire::withdraw("effect"), &ctx()).expect("changes");
    assert!(matches!(
        withdrawn.with_change(ChainChangeWire::reweight("effect", &vec![0.0; ROWS]), &ctx()),
        Err(PropagationChainError::Malformed(_))
    ));
}

// ------------------------------------------------------------- future study

fn update_draft(meaning: LawMeaning) -> PropagationChainRecordWire {
    let theta = normal_grid(0.5, 1.0);
    let noise = independent_grid(0.0, 1.0);
    PropagationChainRecordWire::draft(
        TableConstruction::JointDraws,
        ChainReading::Spread,
        "update",
        vec![ChainSourceWire::draws("theta", meaning, TABLE, &theta)],
        vec![
            ChainStageWire::new(
                "utility",
                StageKind::DecisionUtility,
                &["theta"],
                ChainOpWire::affine(
                    ChainCoefficientWire::fixed(1.0),
                    ChainCoefficientWire::fixed(0.0),
                ),
            )
            .with_decision_reference(0.0),
            ChainStageWire::new(
                "update",
                StageKind::PreposteriorUpdate,
                &["theta", "utility"],
                ChainOpWire::preposterior(12.0_f64.sqrt(), 4.0, 0.0, &noise),
            ),
        ],
    )
}

#[test]
fn a_future_study_updates_the_posterior_before_the_decision_is_re_evaluated() {
    let record = PropagationChainRecordWire::seal(update_draft(LawMeaning::Posterior), &ctx())
        .expect("seals");
    let update = stage(&record, "update");
    // theta ~ N(1/2, 1), n = 4 observations of variance 12, linear utility, reference 0:
    // the study's posterior mean has sd 1/2, so EVSI = 1/2 (phi(1) - Phi(-1)).
    let evsi = 0.5 * (antecedent_kernels::norm_pdf(1.0) - antecedent_kernels::norm_cdf(-1.0));
    near(value(update, "value_of_information"), evsi, 2e-2);
    near(value(update, "current_expected_utility"), 0.5, 1e-3);
    // The tower property: the mean of the updated expected utilities is the current one.
    near(value(update, "mean_updated_expected_utility"), 0.5, 2e-2);
    assert!(value(update, "probability_decision_changes") > 0.05);
    record.consume(None, &ctx()).expect("replays");
    // A changed prior on the parameter re-weights the update with the stages it feeds.
    let log_weights: Vec<f64> = normal_grid(0.5, 1.0).iter().map(|t| 0.5 * t).collect();
    let after = record
        .with_change(ChainChangeWire::reweight("theta", &log_weights), &ctx())
        .expect("changes");
    assert_eq!(stage(&after, "update").status, STATUS_REWEIGHTED);
    near(value(stage(&after, "update"), "current_expected_utility"), 1.0, 3e-2);
}

#[test]
fn a_sampling_law_is_not_updated_by_a_future_study() {
    let refusal = PropagationChainRecordWire::seal(update_draft(LawMeaning::Sampling), &ctx())
        .expect_err("refused")
        .to_refusal();
    assert_eq!(refusal.code(), "uncertainty_representation_insufficient");
    assert_eq!(refusal.next_attempt(), SUPPLY_POSTERIOR_DRAWS);
}

#[test]
fn a_study_too_informative_for_the_draws_is_refused() {
    let mut draft = update_draft(LawMeaning::Posterior);
    if let ChainOpWire::Preposterior { noise_sd, .. } = &mut draft.stages[1].op {
        *noise_sd = 1.0e-5_f64.to_bits();
    }
    let refusal =
        PropagationChainRecordWire::seal(draft, &ctx()).expect_err("refused").to_refusal();
    assert_eq!(refusal.code(), "posterior_draws_below_supported");
}

// ------------------------------------------------------------- enclosures

fn interval_draft(reading: ChainReading) -> PropagationChainRecordWire {
    PropagationChainRecordWire::draft(
        TableConstruction::JointDraws,
        reading,
        "utility",
        vec![
            ChainSourceWire::interval("effect", 0.0, 1.0),
            ChainSourceWire::interval("cost", 0.1, 0.4),
            ChainSourceWire::point("rate", 2.0),
        ],
        vec![
            ChainStageWire::new(
                "scaled",
                StageKind::Transport,
                &["effect", "rate"],
                ChainOpWire::binary("product"),
            ),
            ChainStageWire::new(
                "net",
                StageKind::DerivedQuantity,
                &["scaled", "cost"],
                ChainOpWire::binary("difference"),
            ),
            ChainStageWire::new(
                "utility",
                StageKind::DecisionUtility,
                &["net"],
                ChainOpWire::unary(ChainUnaryWire::exponential_utility(1.0)),
            ),
        ],
    )
}

#[test]
fn intervals_are_carried_as_an_enclosure_that_holds_under_every_coupling() {
    let record = PropagationChainRecordWire::seal(interval_draft(ChainReading::Enclosure), &ctx())
        .expect("seals");
    assert_eq!(record.label, "propagation_enclosure");
    // net = 2 effect - cost lies in [0 - 0.4, 2 - 0.1] = [-0.4, 1.9]; 1 - exp(-x) is increasing.
    let net = stage(&record, "net");
    near(value(net, "lower"), -0.4, 1e-12);
    near(value(net, "upper"), 1.9, 1e-12);
    let utility = stage(&record, "utility");
    near(value(utility, "lower"), 1.0 - 0.4_f64.exp(), 1e-12);
    near(value(utility, "upper"), 1.0 - (-1.9_f64).exp(), 1e-12);
    record.consume(None, &ctx()).expect("replays");
}

#[test]
fn an_interval_is_not_a_spread_a_posterior_or_a_reweightable_source() {
    for reading in [ChainReading::Spread, ChainReading::Posterior] {
        let refusal = PropagationChainRecordWire::seal(interval_draft(reading), &ctx())
            .expect_err("refused")
            .to_refusal();
        assert_eq!(refusal.code(), "uncertainty_representation_insufficient", "{reading:?}");
    }
    let refusal = PropagationChainRecordWire::seal(interval_draft(ChainReading::Interval), &ctx())
        .expect_err("refused")
        .to_refusal();
    assert_eq!(refusal.code(), "uncertainty_representation_insufficient");
    let record = PropagationChainRecordWire::seal(interval_draft(ChainReading::Enclosure), &ctx())
        .expect("seals");
    let refusal = record
        .with_change(ChainChangeWire::reweight("effect", &[0.0, 0.0]), &ctx())
        .expect_err("refused")
        .to_refusal();
    assert_eq!(refusal.next_attempt(), SUPPLY_DRAWS_FOR_INTERVAL);
    // Withdrawing an interval source still invalidates exactly its dependents.
    let after = record.with_change(ChainChangeWire::withdraw("cost"), &ctx()).expect("changes");
    assert_eq!(stage(&after, "scaled").status, STATUS_CURRENT);
    assert_eq!(stage(&after, "net").status, STATUS_INVALIDATED);
}

#[test]
fn a_point_only_chain_has_a_value_and_no_spread() {
    let draft = PropagationChainRecordWire::draft(
        TableConstruction::JointDraws,
        ChainReading::Point,
        "doubled",
        vec![ChainSourceWire::point("a", 3.0)],
        vec![ChainStageWire::new(
            "doubled",
            StageKind::DerivedQuantity,
            &["a"],
            ChainOpWire::affine(ChainCoefficientWire::fixed(2.0), ChainCoefficientWire::fixed(1.0)),
        )],
    );
    let record = PropagationChainRecordWire::seal(draft.clone(), &ctx()).expect("seals");
    assert_eq!(record.label, "point_only");
    near(value(stage(&record, "doubled"), "mean"), 7.0, 1e-12);
    let mut spread = draft;
    spread.reading = ChainReading::Spread.as_str().to_owned();
    let refusal =
        PropagationChainRecordWire::seal(spread, &ctx()).expect_err("refused").to_refusal();
    assert_eq!(refusal.code(), "uncertainty_representation_insufficient");
}

/// `(q + sp - 1) / (se + sp - 1)` over `[q, se, sp]`, guarded and range-checked.
fn correction_op(min_magnitude: f64) -> ChainOpWire {
    use ChainExprTermWire as Term;
    ChainOpWire::expression(vec![
        Term::input(0),
        Term::input(2),
        Term::Add,
        Term::constant(-1.0),
        Term::Add,
        Term::input(1),
        Term::input(2),
        Term::Add,
        Term::constant(-1.0),
        Term::Add,
        Term::divide(min_magnitude),
        Term::require_range(0.0, 1.0),
    ])
}

fn rate_columns() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    (normal_grid(0.31, 0.01), independent_grid(0.9, 0.01), independent_grid(0.8, 0.01))
}

fn correction_draft(min_magnitude: f64) -> PropagationChainRecordWire {
    let (q, se, sp) = rate_columns();
    PropagationChainRecordWire::draft(
        TableConstruction::JointDraws,
        ChainReading::Spread,
        "corrected",
        vec![
            ChainSourceWire::draws("q", LawMeaning::Posterior, TABLE, &q)
                .with_factor(SharedFactorKind::Row, "clinic-rows"),
            ChainSourceWire::draws("se", LawMeaning::Posterior, TABLE, &se)
                .with_factor(SharedFactorKind::Parameter, "validation-sample"),
            ChainSourceWire::draws("sp", LawMeaning::Posterior, TABLE, &sp)
                .with_factor(SharedFactorKind::Parameter, "validation-sample"),
        ],
        vec![ChainStageWire::new(
            "corrected",
            StageKind::MeasurementMapping,
            &["q", "se", "sp"],
            correction_op(min_magnitude),
        )],
    )
}

#[test]
fn a_ratio_expression_record_matches_the_formula_and_the_consumer_replays_it() {
    let record = PropagationChainRecordWire::seal(correction_draft(0.5), &ctx()).expect("seals");
    assert_eq!(record.label, "uncalibrated_spread");
    let (q, se, sp) = rate_columns();
    let corrected: Vec<f64> =
        (0..ROWS).map(|i| (q[i] + sp[i] - 1.0) / (se[i] + sp[i] - 1.0)).collect();
    let mean = corrected.iter().sum::<f64>() / ROWS as f64;
    near(value(stage(&record, "corrected"), "mean"), mean, 1e-12);
    let variance =
        corrected.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (ROWS as f64 - 1.0);
    near(value(stage(&record, "corrected"), "variance"), variance, 1e-12);
    // The ratio of the aligned draws is not the ratio of the means.
    near(mean, 11.0 / 70.0, 4e-3);
    // A stage that is an expression is a nonlinear operator in the plan.
    let plan = record.plan().expect("plan");
    assert_eq!(plan.stage("corrected").expect("stage").class, OperatorClass::Nonlinear);
    // The record encodes, decodes and replays to the stored bits.
    let bytes = record.export().expect("exports");
    let decoded = PropagationChainRecordWire::decode(&bytes).expect("decodes");
    assert_eq!(decoded, record);
    let consumed = decoded.consume(Some(&record.identity), &ctx()).expect("consumes");
    assert_eq!(consumed.results, record.results);
}

#[test]
fn an_expression_edit_is_refused_resealed_or_not_and_its_terms_are_part_of_the_identity() {
    let record = PropagationChainRecordWire::seal(correction_draft(0.5), &ctx()).expect("seals");
    // A changed bound is part of the identity even though it changes no number here.
    let looser = PropagationChainRecordWire::seal(correction_draft(0.25), &ctx()).expect("seals");
    assert_ne!(record.identity, looser.identity);
    assert_eq!(record.results, looser.results);
    // An unsealed edit of a constant no longer matches the identity.
    let mut edited = record.clone();
    edited.stages[0].op = ChainOpWire::expression(vec![ChainExprTermWire::input(0)]);
    assert_eq!(edited.consume(None, &ctx()), Err(PropagationChainError::IdentityMismatch));
    // A resealed edit leaves results that no longer replay.
    edited.identity = propagation_chain_identity(&edited).expect("identity");
    assert_eq!(edited.consume(None, &ctx()), Err(PropagationChainError::ResultMismatch));
    // More terms than the format admits are refused before anything is read.
    let mut long = record.clone();
    let too_many = antecedent_stats::MAX_EXPRESSION_TERMS + 1;
    long.stages[0].op = ChainOpWire::expression(vec![ChainExprTermWire::input(0); too_many]);
    assert_eq!(long.plan(), Err(PropagationChainError::LimitsExceeded("expression terms")));
}

#[test]
fn a_valid_expression_above_the_original_term_cap_seals_and_replays_its_numeric_result() {
    let mut draft = correction_draft(0.5);
    // One input and 32 (constant, add) pairs: 65 valid postfix terms, q + 4.
    // Exercise the supported range beyond the old cap with a numerical oracle,
    // rather than merely moving the overflow fixture to a larger allocation.
    let mut terms = vec![ChainExprTermWire::input(0)];
    for _ in 0..32 {
        terms.push(ChainExprTermWire::constant(0.125));
        terms.push(ChainExprTermWire::Add);
    }
    assert_eq!(terms.len(), 65);
    draft.stages[0].op = ChainOpWire::expression(terms);
    let record = PropagationChainRecordWire::seal(draft, &ctx()).expect("65 valid terms seal");
    let (q, _, _) = rate_columns();
    let mean = q.iter().sum::<f64>() / ROWS as f64;
    let variance = q.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / (ROWS - 1) as f64;
    let result = stage(&record, "corrected");
    near(value(result, "mean"), mean + 4.0, 1e-12);
    near(value(result, "variance"), variance, 1e-12);
    let decoded =
        PropagationChainRecordWire::decode(&record.export().expect("exports")).expect("decodes");
    assert_eq!(decoded, record);
    let consumed = decoded.consume(Some(&record.identity), &ctx()).expect("consumer replays");
    assert_eq!(consumed.results, record.results);
}

#[test]
fn a_denominator_that_reaches_zero_seals_nothing() {
    let mut draft = correction_draft(0.0);
    // The Youden index is negative on some rows.
    for (row, bits) in draft.sources[1].values.iter_mut().enumerate() {
        if row % 5 == 0 {
            *bits = 0.1_f64.to_bits();
        }
    }
    let refusal =
        PropagationChainRecordWire::seal(draft, &ctx()).expect_err("refused").to_refusal();
    assert_eq!(refusal.detail(), "propagation_chain.denominator_not_bounded");
    assert_eq!(refusal.code(), "invalid_argument");
    // A bound above the smallest Youden index refuses the same chain.
    let bounded = correction_draft(0.9);
    let refusal =
        PropagationChainRecordWire::seal(bounded, &ctx()).expect_err("refused").to_refusal();
    assert_eq!(refusal.detail(), "propagation_chain.denominator_not_bounded");
    // A corrected rate above one is refused, never clipped.
    let mut high = correction_draft(0.0);
    high.sources[0].values[10] = 0.99_f64.to_bits();
    let refusal = PropagationChainRecordWire::seal(high, &ctx()).expect_err("refused").to_refusal();
    assert_eq!(refusal.detail(), "propagation_chain.stage_left_domain");
}

#[test]
fn a_changed_assumption_about_a_rate_reweights_exactly_the_expression_that_reads_it() {
    let mut draft = correction_draft(0.0);
    draft.stages.push(ChainStageWire::new(
        "doubled_q",
        StageKind::DerivedQuantity,
        &["q"],
        ChainOpWire::affine(ChainCoefficientWire::fixed(2.0), ChainCoefficientWire::fixed(0.0)),
    ));
    let record = PropagationChainRecordWire::seal(draft, &ctx()).expect("seals");
    let log_weights: Vec<f64> = (0..ROWS).map(|i| 0.5 * (i as f64 / ROWS as f64)).collect();
    let changed = record
        .with_change(ChainChangeWire::reweight("se", &log_weights), &ctx())
        .expect("reweights");
    assert_eq!(stage(&changed, "corrected").status, STATUS_REWEIGHTED);
    assert_eq!(stage(&changed, "doubled_q").status, STATUS_CURRENT);
    assert_eq!(bit_values(stage(&changed, "doubled_q")), bit_values(stage(&record, "doubled_q")));
    assert_ne!(bit_values(stage(&changed, "corrected")), bit_values(stage(&record, "corrected")));
    let withdrawn = record.with_change(ChainChangeWire::withdraw("sp"), &ctx()).expect("withdraws");
    assert_eq!(stage(&withdrawn, "corrected").status, STATUS_INVALIDATED);
    assert_eq!(stage(&withdrawn, "doubled_q").status, STATUS_CURRENT);
}

/// The identity of a record built only from operators that existed before the
/// expression form is the stream the format documents, word for word.
#[test]
fn a_record_without_expressions_keeps_its_identity_encoding() {
    let values = [1.0, 2.0, 3.0, 4.0];
    let draft = PropagationChainRecordWire::draft(
        TableConstruction::JointDraws,
        ChainReading::Spread,
        "scaled",
        vec![ChainSourceWire::draws("x", LawMeaning::Posterior, TABLE, &values)],
        vec![ChainStageWire::new(
            "scaled",
            StageKind::Transport,
            &["x"],
            ChainOpWire::affine(ChainCoefficientWire::fixed(2.0), ChainCoefficientWire::fixed(1.0)),
        )],
    );
    let record = PropagationChainRecordWire::seal(draft, &ctx()).expect("seals");
    let mut writer = CanonicalWriter::new(PROPAGATION_CHAIN_TAG);
    writer.word(1); // format version
    writer.word(1).word(2).text("scaled"); // joint draws, spread, output
    writer.word(1); // one source
    writer.text("x").word(3).opt_word(Some(2)).opt_text(Some(TABLE)).word(1);
    writer.word(0); // no declared factors
    writer.word(4);
    for value in values {
        writer.word(value.to_bits());
    }
    writer.word(1); // one stage
    writer.text("scaled").word(2).word(1).texts(&["x"]); // transport, affine
    writer.word(0); // no own factors
    writer.word(1).word(1).real(2.0).word(1).real(1.0); // affine: fixed scale, fixed shift
    writer.opt_word(None);
    writer.word(0); // no changes
    writer.text("uncalibrated_spread");
    assert_eq!(record.identity, writer.finish_hex());
    // The new operator takes the next numbers: tag 7 for the operator, 5 for its class (nonlinear).
    assert_eq!(StageKind::MeasurementMapping.tag(), 1);
    assert_eq!(OperatorClass::Nonlinear.tag(), 5);
}
