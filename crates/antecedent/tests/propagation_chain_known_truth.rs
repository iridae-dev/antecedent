//! Known-truth checks of the compiled propagation chain. Every expected value
//! is a closed form of the declared inputs (moments of a normal law, interval
//! arithmetic, the moment-generating function of a normal), never a value
//! produced by this library: the chain is compiled, executed through its
//! retained program, and compared with the algebra.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::analysis::propagation_chain::{
    PropagationChain, compile_propagation_chain, consume_propagation_chain,
    independent_product_table,
};
use antecedent_contract::dependence::SharedFactorKind;
use antecedent_contract::propagation_chain::{
    ChainReading, LawMeaning, StageKind, TableConstruction,
};
use antecedent_core::ExecutionContext;
use antecedent_io::propagation_chain_record::{
    ChainCoefficientWire, ChainOpWire, ChainSourceWire, ChainStageWire, ChainUnaryWire,
    STATUS_CURRENT, STATUS_INVALIDATED,
};
use antecedent_io::quantity_wire::ScientificQuantityWire;
use antecedent_stats::normal_ppf;

const ROWS: usize = 1597;

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(5)
}

/// Midpoint quantiles of N(mean, sd^2): a symmetric deterministic sample of the law.
fn grid(mean: f64, sd: f64) -> Vec<f64> {
    (0..ROWS).map(|i| mean + sd * normal_ppf((i as f64 + 0.5) / ROWS as f64)).collect()
}

fn rel(value: f64, truth: f64, tolerance: f64) -> bool {
    (value - truth).abs() <= tolerance * truth.abs().max(1e-12)
}

#[test]
fn affine_stages_and_aligned_differences_match_the_normal_moments() {
    // x ~ N(0.5, 0.36); y = 2x + 1 is the same draw row by row.
    let x = grid(0.5, 0.6);
    let y: Vec<f64> = x.iter().map(|value| 2.0 * value + 1.0).collect();
    let chain = PropagationChain::seal(
        PropagationChain::draft(
            TableConstruction::JointDraws,
            ChainReading::Spread,
            "gap",
            vec![
                ChainSourceWire::draws("x", LawMeaning::Posterior, "table", &x)
                    .with_factor(SharedFactorKind::Row, "rows-x"),
                ChainSourceWire::draws("y", LawMeaning::Posterior, "table", &y)
                    .with_factor(SharedFactorKind::Prior, "prior-y"),
            ],
            vec![
                ChainStageWire::new(
                    "mapped",
                    StageKind::MeasurementMapping,
                    &["x"],
                    ChainOpWire::affine(
                        ChainCoefficientWire::fixed(0.8),
                        ChainCoefficientWire::fixed(0.1),
                    ),
                ),
                ChainStageWire::new(
                    "gap",
                    StageKind::DerivedQuantity,
                    &["x", "y"],
                    ChainOpWire::binary("difference"),
                ),
            ],
        ),
        &ctx(),
    )
    .expect("seals");
    // 0.8 x + 0.1: mean 0.5, variance 0.64 * 0.36.
    let mapped = chain.stage("mapped").expect("mapped");
    assert!((mapped.value("mean").unwrap() - 0.5).abs() < 1e-9);
    assert!(rel(mapped.value("variance").unwrap(), 0.64 * 0.36, 0.03));
    // x - y = -x - 1 on aligned rows: mean -1.5, variance 0.36. Paired
    // independently the variance would be 0.36 + 4 * 0.36 = 1.8.
    let gap = chain.stage("gap").expect("gap");
    assert!((gap.value("mean").unwrap() + 1.5).abs() < 1e-9);
    assert!(rel(gap.value("variance").unwrap(), 0.36, 0.03), "{:?}", gap.value("variance"));
}

#[test]
fn a_nonlinear_utility_is_the_normal_moment_generating_function_not_the_utility_of_the_mean() {
    let x = grid(0.5, 0.6);
    let chain = PropagationChain::seal(
        PropagationChain::draft(
            TableConstruction::JointDraws,
            ChainReading::Spread,
            "utility",
            vec![ChainSourceWire::draws("x", LawMeaning::Posterior, "table", &x)],
            vec![ChainStageWire::new(
                "utility",
                StageKind::DecisionUtility,
                &["x"],
                ChainOpWire::unary(ChainUnaryWire::exponential_utility(1.0)),
            )],
        ),
        &ctx(),
    )
    .expect("seals");
    // E[1 - exp(-X)] = 1 - exp(-m + s2 / 2) for X ~ N(m, s2).
    let exact = 1.0 - (-0.5_f64 + 0.5 * 0.36).exp();
    let at_the_mean = 1.0 - (-0.5_f64).exp();
    let mean = chain.stage("utility").unwrap().value("mean").unwrap();
    assert!((mean - exact).abs() < 0.01, "{mean} vs {exact}");
    assert!((mean - at_the_mean).abs() > 0.03, "Jensen gap lost: {mean} vs {at_the_mean}");
}

#[test]
fn independent_sources_join_by_product_and_expectations_multiply() {
    // X, Y independent two-point laws on {0, 2}: E[XY] = 1; pairing by index gives 2.
    let table = independent_product_table(&[vec![0.0, 2.0], vec![0.0, 2.0]], 16).expect("table");
    let chain = PropagationChain::seal(
        PropagationChain::draft(
            TableConstruction::IndependentProduct,
            ChainReading::Spread,
            "product",
            vec![
                ChainSourceWire::draws("x", LawMeaning::Posterior, "product-table", &table[0]),
                ChainSourceWire::draws("y", LawMeaning::Posterior, "product-table", &table[1]),
            ],
            vec![ChainStageWire::new(
                "product",
                StageKind::DerivedQuantity,
                &["x", "y"],
                ChainOpWire::binary("product"),
            )],
        ),
        &ctx(),
    )
    .expect("seals");
    let mean = chain.answer().expect("answer").value("mean").expect("mean");
    assert!((mean - 1.0).abs() < 1e-12);
}

#[test]
fn declared_intervals_propagate_to_their_interval_arithmetic_enclosure() {
    // net = 2 effect - cost over effect in [0, 1], cost in [0.1, 0.4] lies in [-0.4, 1.9].
    let chain = PropagationChain::seal(
        PropagationChain::draft(
            TableConstruction::JointDraws,
            ChainReading::Enclosure,
            "net",
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
            ],
        ),
        &ctx(),
    )
    .expect("seals");
    assert_eq!(chain.label(), "propagation_enclosure");
    let net = chain.stage("net").expect("net");
    assert!((net.value("lower").unwrap() + 0.4).abs() < 1e-12);
    assert!((net.value("upper").unwrap() - 1.9).abs() < 1e-12);
}

#[test]
fn an_assumption_envelope_is_the_sampling_mean_shifted_by_the_interval_ends() {
    // Sampling draws {1, 2, 3} plus a bias in [0, 1]: the row envelopes average to [2, 3].
    let draft = PropagationChain::draft(
        TableConstruction::JointDraws,
        ChainReading::AssumptionEnvelope,
        "answer",
        vec![
            ChainSourceWire::draws("estimate", LawMeaning::Sampling, "table", &[1.0, 2.0, 3.0]),
            ChainSourceWire::interval("bias", 0.0, 1.0),
        ],
        vec![ChainStageWire::new(
            "answer",
            StageKind::DerivedQuantity,
            &["estimate", "bias"],
            ChainOpWire::binary("sum"),
        )],
    )
    .with_output_quantity(ScientificQuantityWire {
        version: 1,
        variable_id: "study.estimate_plus_bias".into(),
        variable_name: "Estimate plus bias".into(),
        role: "outcome".into(),
        units: "points".into(),
        population_id: "target".into(),
        regime_id: "observed".into(),
        horizon: 1,
        functional_id: "mean".into(),
        conditioning: Vec::new(),
        transform_id: "identity".into(),
    });
    let plan = compile_propagation_chain(draft, &ctx()).expect("compiles");
    let chain = plan.execute(&ctx()).expect("executes");
    let (lower, upper, _) = chain.output_envelope().expect("envelope rows");
    let lower: Vec<f64> = lower.iter().map(|bits| f64::from_bits(*bits)).collect();
    let upper: Vec<f64> = upper.iter().map(|bits| f64::from_bits(*bits)).collect();
    assert_eq!(lower, vec![1.0, 2.0, 3.0]);
    assert_eq!(upper, vec![2.0, 3.0, 4.0]);
    let answer = chain.answer().expect("answer");
    assert!((answer.value("lower").unwrap() - 2.0).abs() < 1e-12);
    assert!((answer.value("upper").unwrap() - 3.0).abs() < 1e-12);
}

#[test]
fn withdrawing_a_source_invalidates_exactly_its_dependents_and_replay_is_exact() {
    let x = grid(0.5, 0.6);
    let z = grid(0.0, 1.0);
    let chain = PropagationChain::seal(
        PropagationChain::draft(
            TableConstruction::JointDraws,
            ChainReading::Spread,
            "total",
            vec![
                ChainSourceWire::draws("x", LawMeaning::Posterior, "table", &x)
                    .with_factor(SharedFactorKind::Row, "rows-x"),
                ChainSourceWire::draws("z", LawMeaning::Posterior, "table", &z)
                    .with_factor(SharedFactorKind::Prior, "prior-z"),
            ],
            vec![
                ChainStageWire::new(
                    "scaled",
                    StageKind::Transport,
                    &["x"],
                    ChainOpWire::affine(
                        ChainCoefficientWire::fixed(2.0),
                        ChainCoefficientWire::fixed(0.0),
                    ),
                ),
                ChainStageWire::new(
                    "total",
                    StageKind::DerivedQuantity,
                    &["scaled", "z"],
                    ChainOpWire::binary("sum"),
                ),
            ],
        ),
        &ctx(),
    )
    .expect("seals");
    assert_eq!(chain.dependents_of("z").expect("plan"), vec!["total".to_owned()]);
    let withdrawn = chain.withdraw("z", &ctx()).expect("changes");
    assert_eq!(withdrawn.stage("scaled"), chain.stage("scaled"));
    assert_eq!(withdrawn.stage("scaled").unwrap().status, STATUS_CURRENT);
    assert_eq!(withdrawn.stage("total").unwrap().status, STATUS_INVALIDATED);
    let bytes = chain.export().expect("exports");
    let consumed =
        consume_propagation_chain(&bytes, Some(chain.identity()), &ctx()).expect("consumes");
    assert_eq!(consumed.results, chain.results());
    // total = 2x + z: mean 1.0, variance 4 * 0.36 + 1 on independent-looking aligned draws is
    // not asserted; only the linear mean is a truth independent of the pairing.
    let total = chain.stage("total").unwrap().value("mean").unwrap();
    assert!((total - 1.0).abs() < 1e-9);
}
