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
//! Beyond the verdict, y0's recorded expressions are evaluated: the verbatim
//! `id_star` string (a sum over named variables of a product of `P(..)` and
//! `P[..](..)` factors) and each term's `identify_outcomes` string (a product
//! of observational conditionals) are parsed and evaluated on the joint of a
//! random latent structural model, and compared with Antecedent's numerator on
//! the same joint and with the enumerated truth. y0's notation drops levels;
//! for this event they are fixed by role (the treatment in a subscript is `x`,
//! read as an outcome it is `x'`, the outcome is `y`, every other variable is
//! summed); a variable y0 prints without its sum in an observational factor
//! that no other factor reads is marginalized there, one it leaves free in a
//! term's conditioning set is read at level 0 on the positive joint (the
//! conditional does not depend on it), and the parser refuses anything else.
//! The enumerated truth checks these readings.
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

use antecedent_core::{CounterfactualEventQuery, ExecutionContext, RegimeId, Value, VariableId};
use antecedent_expr::{DiscreteAxis, ExactDiscreteLaw, LawTolerance};
use antecedent_identify::counterfactual_id::{
    COUNTERFACTUAL_ID_DEFAULT_LIMITS, COUNTERFACTUAL_ID_MEMORY_BYTES, CounterfactualIdDerivation,
    CounterfactualIdProblem, decide_counterfactual_id, evaluate_counterfactual_functional,
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

fn problem_of(case: &Json) -> CounterfactualIdProblem {
    let n = usize::try_from(case["nodes"].as_u64().unwrap()).unwrap();
    CounterfactualIdProblem::new(
        vec![vec![0.0, 1.0]; n],
        &edges(case, "directed"),
        &edges(case, "bidirected"),
    )
    .unwrap()
}

fn role(case: &Json, key: &str) -> u32 {
    u32::try_from(case[key].as_u64().unwrap()).unwrap()
}

/// Our derivation of y0's event `P(Y_{x = 0} = 0, X = 1)` (y0's `-X`, `+X`, `-Y`).
fn derive(case: &Json) -> Result<CounterfactualIdDerivation, String> {
    let x = VariableId::from_raw(role(case, "treatment"));
    let y = VariableId::from_raw(role(case, "outcome"));
    let query = CounterfactualEventQuery::effect_on_treated(x, 0.0, 1.0, y, 0.0).unwrap();
    decide_counterfactual_id(
        &problem_of(case),
        &query,
        COUNTERFACTUAL_ID_DEFAULT_LIMITS,
        COUNTERFACTUAL_ID_MEMORY_BYTES,
        &ExecutionContext::for_tests(1),
    )
    .map_err(|refusal| {
        // A refusal never claims non-identifiability.
        assert_eq!(refusal.code, "route_not_supported", "{}", refusal.message);
        refusal.detail.to_string()
    })
}

/// Our verdict.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Ours {
    /// Identified by ID* with ID on every district term.
    IdStar,
    /// Identified only through the binary-treatment consistency complement.
    Complement,
    /// Refused (ID* conflict or hedge, and no complement).
    Refused,
}

fn ours(case: &Json) -> Ours {
    match derive(case) {
        Ok(derivation)
            if derivation.numerators.iter().any(|(_, f)| f.uses_consistency_complement()) =>
        {
            Ours::Complement
        }
        Ok(_) => Ours::IdStar,
        Err(_) => Ours::Refused,
    }
}

/// Every disagreement, with the reason the record must give for it.
fn disagreements(cases: &[Json]) -> Result<BTreeMap<String, &'static str>, String> {
    let mut out = BTreeMap::new();
    for case in cases {
        let id = case["id"].as_str().unwrap().to_string();
        let class = classify(case)?;
        let reason = match (class, ours(case)) {
            (Y0Class::IdentifiedFromObservational, Ours::IdStar)
            | (Y0Class::Conflict, Ours::Refused) => continue,
            (Y0Class::Zero, Ours::Refused) => "y0_zero_on_a_positive_probability_event",
            (Y0Class::Zero, Ours::Complement) => "y0_zero_where_the_binary_complement_identifies",
            (Y0Class::Conflict, Ours::Complement) => {
                "y0_id_star_conflict_where_the_binary_complement_identifies"
            }
            (Y0Class::Zero, Ours::IdStar) => "y0_zero_where_identified",
            (Y0Class::Conflict, Ours::IdStar) => "y0_conflict_where_identified",
            (Y0Class::IdentifiedFromObservational, _) => "y0_identified_where_we_do_not_by_id_star",
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
        .filter(|c| {
            classify(c).unwrap() == Y0Class::IdentifiedFromObservational && ours(c) == Ours::IdStar
        })
        .count();
    let agree_refused = cases
        .iter()
        .filter(|c| classify(c).unwrap() == Y0Class::Conflict && ours(c) == Ours::Refused)
        .count();
    eprintln!("agree identified {agree_identified}, agree refused {agree_refused}");
    assert!(agree_identified >= 150, "{agree_identified}");
    assert!(agree_refused >= 50, "{agree_refused}");
    // Every recorded disagreement is either a y0 zero on an event a latent
    // structural model on the same graph gives positive mass (our refusal is a
    // checked ID* conflict), or a y0 id_star refusal (conflict or zero) where
    // our binary complement answers, and then our point equals the enumerated
    // truth.
    let mut rng = Rng::new(31);
    for case in cases.iter().filter(|c| recorded.contains_key(c["id"].as_str().unwrap())) {
        let scm = scm_of(case, &mut rng);
        let (x, y) = (role(case, "treatment") as usize, role(case, "outcome") as usize);
        match recorded[case["id"].as_str().unwrap()].as_str() {
            "y0_zero_on_a_positive_probability_event" => {
                assert!(scm.ett_numerator(x, 0, 1, y, 0) > 0.0, "{}", case["id"]);
            }
            "y0_zero_where_the_binary_complement_identifies"
            | "y0_id_star_conflict_where_the_binary_complement_identifies" => {
                let ours = our_numerator(case, &scm);
                let truth = scm.ett_numerator(x, 0, 1, y, 0);
                assert!(close(ours, truth), "{}: {ours} vs {truth}", case["id"]);
            }
            other => panic!("unexpected disagreement {other} on {}", case["id"]),
        }
    }
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-10 * (1.0 + a.abs().max(b.abs()))
}

fn scm_of(case: &Json, rng: &mut Rng) -> LatentScm {
    let n = usize::try_from(case["nodes"].as_u64().unwrap()).unwrap();
    let to_usize = |e: Vec<(u32, u32)>| {
        e.into_iter().map(|(a, b)| (a as usize, b as usize)).collect::<Vec<_>>()
    };
    LatentScm::random(
        rng,
        &vec![2; n],
        &to_usize(edges(case, "directed")),
        &to_usize(edges(case, "bidirected")),
        2,
    )
}

/// Our numerator `P(Y_{x = 0} = 0, X = 1)` on the model's joint.
fn our_numerator(case: &Json, scm: &LatentScm) -> f64 {
    let n = scm.cards.len();
    let axes: Vec<DiscreteAxis> = (0..n)
        .map(|i| DiscreteAxis {
            variable: VariableId::from_raw(u32::try_from(i).unwrap()),
            values: vec![Value::f64(0.0), Value::f64(1.0)].into(),
        })
        .collect();
    let law = ExactDiscreteLaw::try_new(
        "target",
        RegimeId::from_raw(0),
        [],
        axes,
        scm.observational(),
        "latent-scm",
        LawTolerance::default(),
    )
    .unwrap();
    let derivation = derive(case).unwrap();
    let point = evaluate_counterfactual_functional(
        &problem_of(case),
        &derivation,
        &law,
        &ExecutionContext::for_tests(1),
    )
    .unwrap();
    point.numerators[0].1
}

// ------------------------------------------------ y0 expressions, evaluated

/// A `P[subscript](outcomes)` factor (an empty subscript is `P(outcomes)`).
struct Factor {
    subscript: Vec<u32>,
    outcomes: Vec<u32>,
}

fn variable(token: &str) -> Result<u32, String> {
    token
        .trim()
        .strip_prefix('V')
        .and_then(|d| d.parse().ok())
        .ok_or_else(|| format!("not a variable: {token:?}"))
}

fn variables(list: &str) -> Result<Vec<u32>, String> {
    if list.trim().is_empty() {
        return Ok(Vec::new());
    }
    list.split(',').map(variable).collect()
}

fn factor(text: &str) -> Result<Factor, String> {
    if let Some(rest) = text.strip_prefix("P[") {
        let (subscript, outcomes) = rest.split_once("](").ok_or("an unclosed subscript")?;
        let outcomes = outcomes.strip_suffix(')').ok_or("an unclosed factor")?;
        return Ok(Factor { subscript: variables(subscript)?, outcomes: variables(outcomes)? });
    }
    let outcomes = text
        .strip_prefix("P(")
        .and_then(|r| r.strip_suffix(')'))
        .ok_or_else(|| format!("not a factor: {text:?}"))?;
    Ok(Factor { subscript: Vec::new(), outcomes: variables(outcomes)? })
}

/// `Sum[summed](f1 * f2 * ...)` or `f1 * f2 * ...`.
fn parse_id_star(text: &str) -> Result<(Vec<u32>, Vec<Factor>), String> {
    let (summed, body) = match text.strip_prefix("Sum[") {
        Some(rest) => {
            let (summed, body) = rest.split_once("](").ok_or("an unclosed sum")?;
            (variables(summed)?, body.strip_suffix(')').ok_or("an unclosed sum body")?)
        }
        None => (Vec::new(), text),
    };
    Ok((summed, body.split(" * ").map(factor).collect::<Result<_, _>>()?))
}

/// `observational` (`None`), or `P(a, .. | b, ..) * ...` as `(a.., b..)` pairs.
fn parse_identified(text: &str) -> Result<Option<Vec<(Vec<u32>, Vec<u32>)>>, String> {
    if text == "observational" {
        return Ok(None);
    }
    text.split(" * ")
        .map(|f| {
            let inner = f
                .strip_prefix("P(")
                .and_then(|r| r.strip_suffix(')'))
                .ok_or_else(|| format!("not a conditional: {f:?}"))?;
            let (a, b) = inner.split_once(" | ").unwrap_or((inner, ""));
            Ok((variables(a)?, variables(b)?))
        })
        .collect::<Result<Vec<_>, String>>()
        .map(Some)
}

/// Mass of the cells where every `(variable, level)` holds.
fn mass(joint: &[f64], n: usize, fixed: &[(u32, usize)]) -> f64 {
    joint
        .iter()
        .enumerate()
        .filter(|(flat, _)| fixed.iter().all(|&(v, l)| (flat >> (n - 1 - v as usize)) & 1 == l))
        .map(|(_, p)| p)
        .sum()
}

fn names(list: &Json) -> Result<Vec<u32>, String> {
    let mut out: Vec<u32> = list
        .as_array()
        .ok_or("a term without names")?
        .iter()
        .map(|v| variable(v.as_str().unwrap_or_default()))
        .collect::<Result<_, _>>()?;
    out.sort_unstable();
    Ok(out)
}

/// y0's numerator `P(Y_{-x} = -y, X = +x)` from its recorded expression and
/// terms, evaluated on a binary joint (`-v` is level 0, `+v` level 1).
fn y0_numerator(case: &Json, joint: &[f64]) -> Result<f64, String> {
    let n = usize::try_from(case["nodes"].as_u64().unwrap()).unwrap();
    let (x, y) = (role(case, "treatment"), role(case, "outcome"));
    let (summed, factors) = parse_id_star(case["y0_id_star"].as_str().ok_or("no expression")?)?;
    let terms = case["y0_terms"].as_array().ok_or("no terms")?;
    let mut total = 0.0;
    for assignment in 0..(1usize << summed.len()) {
        let summed_level =
            |v: u32| summed.iter().position(|&s| s == v).map(|i| (assignment >> i) & 1);
        let mut product = 1.0;
        for f in &factors {
            // Levels by role: x in a subscript, x' as an outcome, y, or summed.
            let mut binding = Vec::new();
            for &v in &f.subscript {
                let level = if v == x { Some(0) } else { summed_level(v) };
                binding.push((v, level.ok_or(format!("unbound subscript V{v}"))?));
            }
            let mut marginal = Vec::new();
            for &v in &f.outcomes {
                let level = match v {
                    _ if v == x => Some(1),
                    _ if v == y => Some(0),
                    _ => summed_level(v),
                };
                match level {
                    Some(level) => binding.push((v, level)),
                    // y0 prints a variable no other factor reads without its sum
                    // (`P(V0, V1, V2)` for the marginal over V0): marginalized
                    // within this factor; anything else is refused.
                    None if factors
                        .iter()
                        .filter(|g| g.outcomes.contains(&v) || g.subscript.contains(&v))
                        .count()
                        == 1 =>
                    {
                        marginal.push(v);
                    }
                    None => return Err(format!("unbound outcome V{v}")),
                }
            }
            let (mut sub, mut out) = (f.subscript.clone(), f.outcomes.clone());
            sub.sort_unstable();
            out.sort_unstable();
            let term = terms
                .iter()
                .find(|t| {
                    names(&t["treatments"]) == Ok(sub.clone())
                        && names(&t["outcomes"]) == Ok(out.clone())
                })
                .ok_or_else(|| format!("no recorded term for {sub:?} -> {out:?}"))?;
            let at = |vars: &[u32]| -> Result<Vec<(u32, usize)>, String> {
                vars.iter()
                    .map(|v| {
                        binding
                            .iter()
                            .find(|(b, _)| b == v)
                            .copied()
                            .ok_or_else(|| format!("V{v} is not bound by its term"))
                    })
                    .collect()
            };
            product *= match parse_identified(term["identify_outcomes"].as_str().unwrap_or("None"))?
            {
                None => {
                    if !f.subscript.is_empty() {
                        return Err("an observational term with a subscript".into());
                    }
                    let bound: Vec<u32> =
                        f.outcomes.iter().copied().filter(|v| !marginal.contains(v)).collect();
                    mass(joint, n, &at(&bound)?)
                }
                Some(conditionals) => {
                    let mut value = 1.0;
                    for (a, b) in conditionals {
                        // y0 may leave a variable free in a conditioning set (the
                        // conditional does not depend on it where defined, as for
                        // our own free variables): read at level 0 on this
                        // positive joint. The truth comparison checks the reading.
                        let a = at(&a)?;
                        let b: Vec<(u32, usize)> = b
                            .iter()
                            .map(|v| at(&[*v]).map(|l| l[0]).or(Ok::<_, String>((*v, 0))))
                            .collect::<Result<_, _>>()?;
                        let both: Vec<(u32, usize)> = a.iter().chain(&b).copied().collect();
                        value *= mass(joint, n, &both) / mass(joint, n, &b);
                    }
                    value
                }
            };
        }
        total += product;
    }
    Ok(total)
}

#[test]
fn y0_expressions_evaluate_to_our_numerators_on_enumerated_models() {
    let expected = fixture();
    let cases = expected["cases"].as_array().unwrap();
    let mut rng = Rng::new(57);
    let mut compared = 0;
    for case in cases.iter().filter(|c| {
        classify(c).unwrap() == Y0Class::IdentifiedFromObservational && ours(c) == Ours::IdStar
    }) {
        let scm = scm_of(case, &mut rng);
        let joint = scm.observational();
        let theirs = y0_numerator(case, &joint).unwrap_or_else(|e| panic!("{}: {e}", case["id"]));
        let ours = our_numerator(case, &scm);
        let (x, y) = (role(case, "treatment") as usize, role(case, "outcome") as usize);
        let truth = scm.ett_numerator(x, 0, 1, y, 0);
        assert!(close(theirs, ours), "{}: y0 {theirs} vs ours {ours}", case["id"]);
        assert!(close(ours, truth), "{}: ours {ours} vs truth {truth}", case["id"]);
        compared += 1;
    }
    assert!(compared >= 150, "{compared}");
    // The comparison reads the expression: another identified expression for
    // the front door (P(m) in place of P(m | x)) no longer matches.
    let mut edited = cases.iter().find(|c| c["id"] == "frontdoor").unwrap().clone();
    edited["y0_terms"][1]["identify_outcomes"] = Json::from("P(V1)");
    let scm = scm_of(&edited, &mut rng);
    let theirs = y0_numerator(&edited, &scm.observational()).unwrap();
    assert!(!close(theirs, our_numerator(&edited, &scm)), "an edited expression still matches");
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
