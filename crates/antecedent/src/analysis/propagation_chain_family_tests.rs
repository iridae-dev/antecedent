//! Family adapters: estimates and the evidence behind them become chain
//! sources, the shared evidence is counted once through a pool and a nonlinear
//! decision utility, and a retired source takes out exactly its dependents.
//!
//! Every reference is closed form. Two equal-variance estimators with
//! correlation `r` pool to variance `v (1 + r) / 2`; the exponential utility
//! `1 - exp(-x)` of a normal pooled value has mean `1 - exp(-m + s2 / 2)`.

use antecedent_contract::evidence_ledger::{ObservationId, SourceDeclaration, SourceKind};

use super::*;

const TABLE_ROWS: usize = 20_000;
const SEED: u64 = 0xFA11;

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(3)
}

fn rows(range: std::ops::Range<u64>) -> impl Iterator<Item = ObservationId> {
    range.map(|index| ObservationId::from_key("family-rows", &index.to_le_bytes()))
}

struct Fixture {
    ledger: EvidenceLedger,
    a: SourceId,
    b: SourceId,
    c: SourceId,
}

/// Estimators `a` and `b` use rows 0..200 and 100..300: they share 100 rows.
/// Estimator `c` uses 1000..1100 and shares none.
fn fixture() -> Fixture {
    let mut ledger = EvidenceLedger::new();
    let mut add = |label: &str, range| {
        ledger
            .add_source(
                SourceDeclaration::new(label, SourceKind::Experiment)
                    .with_observations(rows(range)),
            )
            .expect("source")
    };
    let a = add("trial-a", 0..200);
    let b = add("trial-b", 100..300);
    let c = add("trial-c", 1000..1100);
    Fixture { ledger, a, b, c }
}

fn estimates(fixture: &Fixture) -> Vec<FamilyEstimate> {
    vec![
        FamilyEstimate::new("a", 0.4, 0.5, LawMeaning::Posterior).with_authority(fixture.a),
        FamilyEstimate::new("b", 0.6, 0.5, LawMeaning::Posterior).with_authority(fixture.b),
        FamilyEstimate::new("c", 0.2, 0.3, LawMeaning::Posterior).with_authority(fixture.c),
    ]
}

fn stages(dependent_pool: bool) -> Vec<ChainStageWire> {
    vec![
        ChainStageWire::new(
            "pooled",
            StageKind::Pooling,
            &["a", "b"],
            ChainOpWire::pool(dependent_pool),
        ),
        ChainStageWire::new(
            "utility",
            StageKind::DecisionUtility,
            &["pooled"],
            ChainOpWire::unary(ChainUnaryWire::exponential_utility(1.0)),
        ),
        ChainStageWire::new(
            "cost",
            StageKind::Transport,
            &["c"],
            ChainOpWire::affine(ChainCoefficientWire::fixed(1.0), ChainCoefficientWire::fixed(0.0)),
        ),
        ChainStageWire::new(
            "net",
            StageKind::DerivedQuantity,
            &["utility", "cost"],
            ChainOpWire::binary("difference"),
        ),
    ]
}

fn table(fixture: &Fixture) -> FamilyTable {
    family_table(
        "family",
        &estimates(fixture),
        &FamilyDependence::LedgerOverlap,
        Some(&fixture.ledger),
        TABLE_ROWS,
        SEED,
    )
    .expect("ledger overlap table")
}

#[test]
fn shared_rows_in_the_ledger_become_the_correlation_the_pool_and_utility_see() {
    let fixture = fixture();
    let family = table(&fixture);
    // 100 shared rows of 200 each: correlation 0.5, covariance 0.5 * 0.25.
    assert!((family.covariance[1] - 0.125).abs() < 1e-15);
    assert_eq!(family.covariance[2], 0.0);
    assert_eq!(family.covariance[5], 0.0);
    let chain =
        PropagationChain::seal(family.draft(ChainReading::Spread, "net", stages(true)), &ctx())
            .expect("a dependent pool of the shared evidence seals");
    // Dependent pool: weights (1/2, 1/2); variance 0.25 * (0.25 + 0.25 + 2 * 0.125).
    let pooled = chain.stage("pooled").expect("pooled");
    let variance = pooled.value("variance").expect("variance");
    assert!((pooled.value("mean").unwrap() - 0.5).abs() < 1e-9);
    assert!((variance - 0.1875).abs() < 1e-9, "{variance}");
    // Pooling the same evidence as if it were independent would halve the
    // variance (0.125): the shared rows are not counted twice.
    assert!(variance > 0.125 + 0.05);
    // The nonlinear utility is taken against the joint law, not its mean:
    // E[1 - exp(-X)] = 1 - exp(-m + s2 / 2) for X ~ N(m, s2).
    let exact = 1.0 - (-0.5_f64 + 0.5 * 0.1875).exp();
    let independent_claim = 1.0 - (-0.5_f64 + 0.5 * 0.125).exp();
    let utility = chain.stage("utility").expect("utility").value("mean").expect("mean");
    assert!((utility - exact).abs() < 0.01, "{utility} vs {exact}");
    assert!((utility - independent_claim).abs() > 0.01, "{utility} vs {independent_claim}");
    // Jensen: the expected utility is not the utility of the expected value.
    assert!((utility - (1.0 - (-0.5_f64).exp())).abs() > 0.01);
    // Cost is independent of the pooled evidence: the difference adds variances.
    let net = chain.stage("net").expect("net");
    let utility_variance = chain.stage("utility").unwrap().value("variance").unwrap();
    assert!(
        (net.value("variance").unwrap() - (utility_variance + 0.09)).abs()
            < 0.01 * (utility_variance + 0.09),
    );
    // The consumer re-derives every stage from the exported bytes.
    let bytes = chain.export().expect("export");
    consume_propagation_chain(&bytes, Some(chain.identity()), &ctx()).expect("replays");
}

#[test]
fn an_independent_pool_of_the_same_shared_rows_is_refused_by_the_chain_rule() {
    let fixture = fixture();
    let family = table(&fixture);
    let refusal =
        PropagationChain::seal(family.draft(ChainReading::Spread, "net", stages(false)), &ctx())
            .expect_err("independent pool of overlapping evidence");
    assert_eq!(refusal.code(), "evidence_double_counted");
    assert_eq!(refusal.next_attempt().name(), "pool_with_joint_covariance");
}

#[test]
fn a_retired_ledger_source_invalidates_exactly_its_dependents() {
    let mut fixture = fixture();
    let family = table(&fixture);
    let chain =
        PropagationChain::seal(family.draft(ChainReading::Spread, "net", stages(true)), &ctx())
            .expect("seals");
    // Nothing retired: applying the authority changes nothing.
    assert!(family.authority.retired(&fixture.ledger).is_empty());
    let unchanged = family.authority.apply(&chain, &fixture.ledger, &ctx()).expect("apply");
    assert_eq!(unchanged, chain);

    fixture.ledger.withdraw_source(fixture.b, "retracted by the investigators").expect("withdraw");
    let retired = family.authority.retired(&fixture.ledger);
    assert_eq!(retired.len(), 1);
    assert_eq!(retired[0].0, "b");
    assert!(retired[0].1.contains("retracted"));
    let updated = family.authority.apply(&chain, &fixture.ledger, &ctx()).expect("apply");
    for id in ["pooled", "utility", "net"] {
        assert_eq!(updated.stage(id).expect("stage").status, STATUS_INVALIDATED, "{id}");
        assert!(updated.require_current_value(id, "mean").is_err(), "{id}");
    }
    // The cost stage reads only the untouched source: its numbers are bit-identical.
    assert_eq!(updated.stage("cost").unwrap().status, STATUS_CURRENT);
    assert_eq!(updated.stage("cost"), chain.stage("cost"));
    assert!(updated.require_current_value("cost", "mean").is_ok());
    // Applying again does not withdraw twice.
    let again = family.authority.apply(&updated, &fixture.ledger, &ctx()).expect("idempotent");
    assert_eq!(again, updated);
    // A retired source cannot seed a new table.
    let refusal = family_table(
        "family",
        &estimates(&fixture),
        &FamilyDependence::LedgerOverlap,
        Some(&fixture.ledger),
        TABLE_ROWS,
        SEED,
    )
    .expect_err("retired source");
    assert_eq!(refusal.next_attempt().name(), "rerun_without_withdrawn_source");
    // The invalidated chain still replays independently.
    let bytes = updated.export().expect("export");
    consume_propagation_chain(&bytes, Some(updated.identity()), &ctx()).expect("replays");
}

#[test]
fn declared_dependence_must_agree_with_the_declared_evidence() {
    let fixture = fixture();
    let two = || {
        vec![
            FamilyEstimate::new("a", 0.4, 0.5, LawMeaning::Posterior)
                .with_factor(SharedFactorKind::Row, "pilot-rows"),
            FamilyEstimate::new("b", 0.6, 0.5, LawMeaning::Posterior)
                .with_factor(SharedFactorKind::Row, "pilot-rows"),
        ]
    };
    // Sharing rows and claiming independence is double counting.
    let refusal = family_table("t", &two(), &FamilyDependence::Independent, None, 500, 1)
        .expect_err("shared rows");
    assert_eq!(refusal.code(), "evidence_double_counted");
    // Unknown overlap is never promoted to independence.
    let unknown = vec![
        FamilyEstimate::new("a", 0.4, 0.5, LawMeaning::Posterior).with_unknown_overlap(),
        FamilyEstimate::new("b", 0.6, 0.5, LawMeaning::Posterior),
    ];
    let refusal = family_table("t", &unknown, &FamilyDependence::Independent, None, 500, 1)
        .expect_err("unknown overlap");
    assert_eq!(refusal.code(), "evidence_overlap_unknown");
    // A covariance whose diagonal is not the published standard errors.
    let wrong = FamilyDependence::Covariance(vec![0.3, 0.1, 0.1, 0.25]);
    assert!(family_table("t", &two(), &wrong, None, 500, 1).is_err());
    // A nonzero covariance with no declared shared evidence.
    let undeclared = vec![
        FamilyEstimate::new("a", 0.4, 0.5, LawMeaning::Posterior),
        FamilyEstimate::new("b", 0.6, 0.5, LawMeaning::Posterior),
    ];
    let coupled = FamilyDependence::Covariance(vec![0.25, 0.1, 0.1, 0.25]);
    let refusal = family_table("t", &undeclared, &coupled, None, 500, 1).expect_err("undeclared");
    assert_eq!(refusal.code(), "evidence_overlap_unknown");
    // Declared shared rows with a zero covariance.
    let zero = FamilyDependence::Covariance(vec![0.25, 0.0, 0.0, 0.25]);
    let refusal = family_table("t", &two(), &zero, None, 500, 1).expect_err("zero");
    assert_eq!(refusal.code(), "evidence_double_counted");
    // The consistent declaration builds, with the covariance exact in the draws.
    let consistent = FamilyDependence::Covariance(vec![0.25, 0.1, 0.1, 0.25]);
    let built = family_table("t", &two(), &consistent, None, 500, 1).expect("consistent");
    assert_eq!(built.sources.len(), 2);
    assert!(built.sources.iter().all(|source| source.factors.len() == 1));
    // A ledger source that declares no observations leaves the overlap unknown.
    let mut ledger = fixture.ledger.clone();
    let blank = ledger
        .add_source(SourceDeclaration::new("trial-undeclared", SourceKind::Experiment))
        .expect("source");
    let pair = vec![
        FamilyEstimate::new("a", 0.4, 0.5, LawMeaning::Posterior).with_authority(fixture.a),
        FamilyEstimate::new("u", 0.6, 0.5, LawMeaning::Posterior).with_authority(blank),
    ];
    let refusal = family_table("t", &pair, &FamilyDependence::LedgerOverlap, Some(&ledger), 500, 1)
        .expect_err("undeclared observations");
    assert_eq!(refusal.code(), "evidence_overlap_unknown");
}
