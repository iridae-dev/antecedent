//! 2.2B X8: recorded Y0 differential for counterfactual identification of the
//! effect of treatment on the treated.
//!
//! `conformance/identify/counterfactual_id_y0/expected.json` holds, per graph,
//! the verbatim output of y0 0.2.11's `id_star` on `{Y @ -X: -Y, X: +X}` and
//! of `identify_outcomes` on every interventional term it returned, produced
//! once in a deleted environment. This test parses that output, classifies it
//! and compares the class with Antecedent's decision on the same graph. Every
//! disagreement must be recorded with its reason, and every recorded
//! disagreement must still disagree.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::type_complexity,
    clippy::too_many_lines,
    clippy::needless_range_loop,
    clippy::result_large_err,
    reason = "test fixtures: small exact indices, levels and enumerated probabilities; exact bit comparisons are the assertion"
)]

#[path = "support/latent_scm.rs"]
mod latent_scm;

use std::collections::BTreeMap;

use antecedent_core::{CounterfactualEventQuery, ExecutionContext, VariableId};
use antecedent_identify::counterfactual_id::{
    COUNTERFACTUAL_ID_DEFAULT_LIMITS, COUNTERFACTUAL_ID_MEMORY_BYTES, CounterfactualIdProblem,
    decide_counterfactual_id,
};
use latent_scm::{LatentScm, Rng};
use serde_json::Value as Json;

const EXPECTED: &str =
    include_str!("../../../conformance/identify/counterfactual_id_y0/expected.json");

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Y0Class {
    /// `id_star` raised `ConflictUnidentifiable`.
    Conflict,
    /// `id_star` returned `Zero()`.
    Zero,
    /// An expression every interventional term of which `identify_outcomes` identified.
    IdentifiedFromObservational,
    /// An expression with a term `identify_outcomes` refused.
    ExperimentalOnly,
}

/// Classify one case from its verbatim output alone.
fn classify(case: &Json) -> Result<Y0Class, String> {
    let text = case["y0_id_star"].as_str().ok_or("no verbatim output")?;
    let class = if text.starts_with("ConflictUnidentifiable(") {
        Y0Class::Conflict
    } else if text == "Zero()" {
        Y0Class::Zero
    } else if text.starts_with("Sum[") || text.starts_with('P') || text.contains(" * ") {
        let terms = case["y0_terms"].as_array().ok_or("an expression without its terms")?;
        if terms.is_empty() {
            return Err(format!("an expression without terms: {text}"));
        }
        if terms.iter().all(|t| t["identify_outcomes"].as_str().is_some_and(|s| s != "None")) {
            Y0Class::IdentifiedFromObservational
        } else {
            Y0Class::ExperimentalOnly
        }
    } else {
        return Err(format!("unparsed y0 output: {text}"));
    };
    // The recorded status agrees with the verbatim output.
    let status = case["y0_status"].as_str().unwrap_or_default();
    let expected = match class {
        Y0Class::Conflict => "conflict",
        Y0Class::Zero => "zero",
        Y0Class::IdentifiedFromObservational | Y0Class::ExperimentalOnly => "expression",
    };
    if status != expected {
        return Err(format!("status {status} does not match the verbatim output {text}"));
    }
    Ok(class)
}

fn edges(case: &Json, key: &str) -> Vec<(u32, u32)> {
    case[key]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| {
            let pair = e.as_array().unwrap();
            (
                u32::try_from(pair[0].as_u64().unwrap()).unwrap(),
                u32::try_from(pair[1].as_u64().unwrap()).unwrap(),
            )
        })
        .collect()
}

/// Our decision: identified from `P(V)`, or not.
fn ours(case: &Json) -> bool {
    let n = usize::try_from(case["nodes"].as_u64().unwrap()).unwrap();
    let problem = CounterfactualIdProblem::new(
        vec![vec![0.0, 1.0]; n],
        &edges(case, "directed"),
        &edges(case, "bidirected"),
    )
    .unwrap();
    let x = VariableId::from_raw(u32::try_from(case["treatment"].as_u64().unwrap()).unwrap());
    let y = VariableId::from_raw(u32::try_from(case["outcome"].as_u64().unwrap()).unwrap());
    let query = CounterfactualEventQuery::effect_on_treated(x, 0.0, 1.0, y, 0.0).unwrap();
    match decide_counterfactual_id(
        &problem,
        &query,
        COUNTERFACTUAL_ID_DEFAULT_LIMITS,
        COUNTERFACTUAL_ID_MEMORY_BYTES,
        &ExecutionContext::for_tests(1),
    ) {
        Ok(_) => true,
        Err(refusal) => {
            assert_eq!(refusal.code, "cross_world_not_identified", "{}", refusal.message);
            false
        }
    }
}

/// Every disagreement, with the reason the record must give for it.
fn disagreements(cases: &[Json]) -> Result<BTreeMap<String, &'static str>, String> {
    let mut out = BTreeMap::new();
    for case in cases {
        let id = case["id"].as_str().unwrap().to_string();
        let class = classify(case)?;
        let identified = ours(case);
        let reason = match (class, identified) {
            (Y0Class::IdentifiedFromObservational, true) | (Y0Class::Conflict, false) => continue,
            (Y0Class::Zero, false) => "y0_zero_on_a_positive_probability_event",
            (Y0Class::Zero, true) => "y0_zero_where_identified",
            (Y0Class::Conflict, true) => "y0_conflict_where_identified",
            (Y0Class::IdentifiedFromObservational, false) => "y0_identified_where_refused",
            (Y0Class::ExperimentalOnly, _) => "y0_experimental_only",
        };
        out.insert(id, reason);
    }
    Ok(out)
}

fn fixture() -> Json {
    serde_json::from_str(EXPECTED).unwrap()
}

#[test]
fn y0_id_star_outputs_agree_with_our_decisions_and_every_disagreement_is_recorded() {
    let expected = fixture();
    assert_eq!(expected["oracle"]["reference"][0]["version"], "0.2.11");
    let cases = expected["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 257, "the recorded graph set");
    let found = disagreements(cases).unwrap();
    let recorded: BTreeMap<String, String> = expected["disagreements"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, v)| (k.clone(), v.as_str().unwrap().to_string()))
        .collect();
    let found_text: BTreeMap<String, String> =
        found.iter().map(|(k, v)| (k.clone(), (*v).to_string())).collect();
    assert_eq!(found_text, recorded, "recorded disagreements must equal the computed ones");
    // Agreement is the rule: both classes are well represented.
    let agree_identified = cases
        .iter()
        .filter(|c| classify(c).unwrap() == Y0Class::IdentifiedFromObservational && ours(c))
        .count();
    let agree_refused =
        cases.iter().filter(|c| classify(c).unwrap() == Y0Class::Conflict && !ours(c)).count();
    assert!(agree_identified >= 150, "{agree_identified}");
    assert!(agree_refused >= 50, "{agree_refused}");
    // Every recorded disagreement is a y0 zero on an event of positive probability:
    // a latent structural model on the same graph gives the event positive mass
    // (and our refusal is a checked conflict).
    let mut rng = Rng::new(31);
    for case in cases.iter().filter(|c| recorded.contains_key(c["id"].as_str().unwrap())) {
        assert_eq!(
            recorded[case["id"].as_str().unwrap()],
            "y0_zero_on_a_positive_probability_event"
        );
        let n = usize::try_from(case["nodes"].as_u64().unwrap()).unwrap();
        let to_usize = |e: Vec<(u32, u32)>| {
            e.into_iter().map(|(a, b)| (a as usize, b as usize)).collect::<Vec<_>>()
        };
        let scm = LatentScm::random(
            &mut rng,
            &vec![2; n],
            &to_usize(edges(case, "directed")),
            &to_usize(edges(case, "bidirected")),
            2,
        );
        let x = usize::try_from(case["treatment"].as_u64().unwrap()).unwrap();
        let y = usize::try_from(case["outcome"].as_u64().unwrap()).unwrap();
        assert!(scm.ett_numerator(x, 0, 1, y, 0) > 0.0, "{}", case["id"]);
    }
}

#[test]
fn the_verbatim_output_is_read_not_trusted() {
    let expected = fixture();
    let cases = expected["cases"].as_array().unwrap().clone();
    let baseline = disagreements(&cases).unwrap();
    // A flipped verdict is a new disagreement.
    let mut flipped = cases.clone();
    let bow = flipped.iter_mut().find(|c| c["id"] == "bow_arc").unwrap();
    bow["y0_id_star"] = Json::from("Sum[V1](P(V0) * P[V0](V1))");
    bow["y0_status"] = Json::from("expression");
    bow["y0_terms"] = serde_json::json!([{ "treatments": ["V0"], "outcomes": ["V1"], "identify_outcomes": "P(V1 | V0)" }]);
    assert_ne!(disagreements(&flipped).unwrap(), baseline);
    // A term y0 refused to identify changes the class.
    let mut refused = cases.clone();
    let front = refused.iter_mut().find(|c| c["id"] == "frontdoor").unwrap();
    front["y0_terms"][0]["identify_outcomes"] = Json::from("None");
    assert_eq!(
        disagreements(&refused).unwrap().get("frontdoor").copied(),
        Some("y0_experimental_only")
    );
    // A status edited away from the verbatim output is caught.
    let mut edited = cases;
    let mediation = edited.iter_mut().find(|c| c["id"] == "markovian_mediation").unwrap();
    mediation["y0_status"] = Json::from("conflict");
    assert!(disagreements(&edited).is_err());
}
