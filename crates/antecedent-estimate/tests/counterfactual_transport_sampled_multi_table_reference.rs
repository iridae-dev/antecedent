//! The sampled counterfactual transport estimate against a 50-digit multi-dataset delta reference.
//!
//! `oracle_data/sampled_transport_multi_table_reference.py` (frozen in the JSON beside it) writes
//! the identified functional of the two-source joint counterfactual `P*(Y_{x=1} = 1, X = 0)` as a
//! function of the cell weights of the THREE count tables it reads (source observational joint,
//! source `do(X = 1)` experiment, target marginal of `Z`) and forms the plug-in and the sum over
//! independent datasets of the multinomial delta variance `(1/N)[sum p g^2 - (sum p g)^2]`, with
//! gradients by 50-digit central differences. The single-table chain reference elsewhere checks
//! one dataset; this one checks the sum over independent datasets, tiny and very unbalanced table
//! sizes, 1-in-10^5 cells, and additive smoothing of a table with an unseen cell (the smoothed
//! table's shares, its own total in the variance). The same file also pins that sample size
//! enters only through the variance (identical shares at ten times the counts give the same
//! point and a standard error smaller by sqrt(10)).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#[path = "../../antecedent-identify/tests/support/counterfactual_transport_adapter.rs"]
mod counterfactual_transport_adapter;
#[path = "../../antecedent-identify/tests/support/counterfactual_transport_cases.rs"]
mod counterfactual_transport_cases;
#[path = "../../antecedent-identify/tests/support/finite_scm_oracle.rs"]
mod finite_scm_oracle;

use std::sync::Arc;

use antecedent_core::ExecutionContext;
use antecedent_estimate::counterfactual_transport_sampled::{
    OverlapDeclaration, SampledAxis, SampledCounterfactualTransportEstimate, SampledRegimeCounts,
    SampledRegimeCountsParts, SampledTransportLimits, SampledTransportRequest, SamplingDesign,
    UnseenCellRule, estimate_sampled_counterfactual_transport,
};
use antecedent_expr::ExactDiscreteLaw;
use antecedent_identify::{
    COUNTERFACTUAL_TRANSPORT_DEFAULT_LIMITS, CounterfactualTransportDecision,
    CounterfactualTransportDerivation, identify_counterfactual_transport,
};
use counterfactual_transport_adapter as adapter;
use counterfactual_transport_cases::all_cases;
use serde_json::Value;

fn table(law: &ExactDiscreteLaw, counts: Vec<u64>) -> SampledRegimeCounts {
    SampledRegimeCounts::try_new(SampledRegimeCountsParts {
        population: Arc::from(law.population()),
        regime: law.regime(),
        interventions: law
            .interventions()
            .iter()
            .map(|a| (a.variable, a.value.as_f64().expect("a finite level")))
            .collect(),
        axes: law
            .axes()
            .iter()
            .map(|a| SampledAxis {
                variable: a.variable,
                levels: a.values.iter().map(|v| v.as_f64().expect("a finite level")).collect(),
            })
            .collect(),
        counts,
        snapshot_identity: Arc::from(law.snapshot_identity()),
        design: SamplingDesign::IndependentUnits,
        structural_zeros: Vec::new(),
    })
    .expect("a table of an enumerated law is valid")
}

fn counts(v: &Value) -> Vec<u64> {
    v.as_array().unwrap().iter().map(|c| c.as_u64().unwrap()).collect()
}

struct Setup {
    derivation: CounterfactualTransportDerivation,
    laws: Vec<ExactDiscreteLaw>,
}

fn setup() -> Setup {
    let case = all_cases()
        .into_iter()
        .find(|c| c.name == "joint_counterfactual_two_source_regimes")
        .unwrap();
    let adapted = adapter::catalog(&case);
    let data = adapter::laws(&case, &adapted);
    let target = &case.targets[0];
    let problem = adapter::problem(&case);
    let sources = adapter::sources(&case, &adapter::evidence_controllable(&case));
    let query = adapter::query(&case, &target.query, sources);
    let decision = identify_counterfactual_transport(
        &problem,
        &query,
        &adapted.catalog,
        COUNTERFACTUAL_TRANSPORT_DEFAULT_LIMITS,
        &ExecutionContext::for_tests(1),
    )
    .unwrap();
    let CounterfactualTransportDecision::Identified(derivation) = decision else {
        panic!("the fixture identifies")
    };
    Setup { derivation: *derivation, laws: data.laws().to_vec() }
}

/// `do0` is the table of the `do(X = 0)` world of the same cited regime: the derivation reads only
/// the `do(X = 1)` world, so its counts must not matter.
fn tables_for(
    s: &Setup,
    obs: &[u64],
    do1: &[u64],
    do0: &[u64],
    tgt: &[u64],
) -> Vec<SampledRegimeCounts> {
    let mut out = Vec::new();
    for law in &s.laws {
        let cited = s
            .derivation
            .cited
            .iter()
            .any(|c| c.regime == law.regime() && c.population.as_ref() == law.population());
        if !cited {
            continue;
        }
        let which = match (law.population(), law.interventions().len(), law.probabilities().len()) {
            ("source", 0, 8) => obs,
            ("source", 1, 4) => {
                if law.interventions()[0].value.as_f64() == Some(1.0) {
                    do1
                } else {
                    do0
                }
            }
            ("target", 0, 2) => tgt,
            other => panic!("unexpected cited law {other:?}"),
        };
        out.push(table(law, which.to_vec()));
    }
    assert_eq!(out.len(), 4, "the cited regimes supply four tables, the functional reads three");
    out
}

fn estimate(
    s: &Setup,
    tables: Vec<SampledRegimeCounts>,
    rule: UnseenCellRule,
) -> SampledCounterfactualTransportEstimate {
    estimate_sampled_counterfactual_transport(
        &s.derivation,
        &SampledTransportRequest {
            tables,
            overlap: OverlapDeclaration::all_independent(),
            unseen_rule: rule,
            limits: SampledTransportLimits::default(),
        },
        &ExecutionContext::for_tests(7),
    )
    .unwrap_or_else(|e| panic!("{e}"))
}

#[test]
fn plug_in_and_summed_delta_standard_error_equal_the_multiprecision_reference() {
    let s = setup();
    let cases: Vec<Value> = serde_json::from_str(include_str!(
        "oracle_data/sampled_transport_multi_table_reference.json"
    ))
    .unwrap();
    for case in &cases {
        let name = case["name"].as_str().unwrap();
        let rule = case["smoothing"].as_f64().map_or(UnseenCellRule::Refuse, |a| {
            UnseenCellRule::AdditiveSmoothing { pseudo_count: a }
        });
        let do1 = counts(&case["do1"]);
        let do0: Vec<u64> = do1.iter().rev().map(|c| c + 3).collect();
        let tables = tables_for(&s, &counts(&case["obs"]), &do1, &do0, &counts(&case["target"]));
        let r = estimate(&s, tables, rule);
        let point = r.point.as_ref().expect("a point");
        let (want_point, want_se) = (case["point"].as_f64().unwrap(), case["se"].as_f64().unwrap());
        assert_eq!(
            (r.repairs.complements_clamped, r.repairs.probability_clamped),
            (0, false),
            "{name}: a repair would make the reference a different functional"
        );
        assert!(
            (point.probability - want_point).abs() <= 1e-12,
            "{name} point {} vs {want_point}",
            point.probability
        );
        assert_eq!(point.smoothed, case["smoothing"].as_f64().is_some(), "{name}");
        let se = r.inference.standard_error.as_ref().expect("a standard error");
        assert_eq!(
            se.datasets, 4,
            "{name}: the unread do(X=0) table is a fourth, zero-gradient dataset"
        );
        // central difference at relative step 1e-5: error O(h^2) of the gradient
        assert!(
            (se.value - want_se).abs() <= 1e-6 * want_se,
            "{name} se {} vs {want_se} (rel {:e})",
            se.value,
            (se.value - want_se).abs() / want_se
        );
    }
}

#[test]
fn ten_times_the_counts_keeps_the_point_and_shrinks_the_standard_error_by_root_ten() {
    let s = setup();
    let (obs, do1, tgt) = ([12u64, 9, 4, 15, 7, 6, 3, 5], [10u64, 6, 8, 6], [9u64, 11]);
    let scale = |v: &[u64]| v.iter().map(|c| c * 10).collect::<Vec<_>>();
    let small =
        estimate(&s, tables_for(&s, &obs, &do1, &[5, 6, 7, 8], &tgt), UnseenCellRule::Refuse);
    let large = estimate(
        &s,
        tables_for(&s, &scale(&obs), &scale(&do1), &[5, 6, 7, 8], &scale(&tgt)),
        UnseenCellRule::Refuse,
    );
    let (p1, p2) = (small.point.unwrap().probability, large.point.unwrap().probability);
    assert!((p1 - p2).abs() <= 1e-14, "{p1} vs {p2}");
    let ratio = small.inference.standard_error.unwrap().value
        / large.inference.standard_error.unwrap().value;
    assert!((ratio - 10f64.sqrt()).abs() <= 1e-6, "{ratio}");
}

#[test]
fn a_table_the_derivation_does_not_read_changes_nothing() {
    let s = setup();
    let (obs, do1, tgt) = ([12u64, 9, 4, 15, 7, 6, 3, 5], [10u64, 6, 8, 6], [9u64, 11]);
    let a = estimate(&s, tables_for(&s, &obs, &do1, &[5, 6, 7, 8], &tgt), UnseenCellRule::Refuse);
    let b =
        estimate(&s, tables_for(&s, &obs, &do1, &[900, 25, 70, 21], &tgt), UnseenCellRule::Refuse);
    assert_eq!(a.point.unwrap().probability.to_bits(), b.point.unwrap().probability.to_bits());
    assert_eq!(
        a.inference.standard_error.unwrap().value.to_bits(),
        b.inference.standard_error.unwrap().value.to_bits()
    );
}
