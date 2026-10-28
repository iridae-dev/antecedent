//! Finite-law conditioning, intervention, measured-indicator and threshold queries against an
//! exact rational enumeration.
//!
//! `oracle_data/finite_law_conditioning_reference.py` (frozen in the JSON beside it) enumerates
//! the joint of three random dyadic four-variable models (one with a structurally impossible
//! level), conditions by summing matching assignments, builds a mutilated model for each
//! intervention, adds the indicator of a measured construct as an extra variable and conditions
//! on its recorded value, and compares every signal level's conditional with the threshold
//! exactly. This test asks the product's `condition`, `intervene`, `condition_after_intervention`,
//! `condition_on_measured` and `threshold_crossing_observations` for the same quantities and
//! requires agreement to 1e-12, including the zero-evidence and unreachable refusals and a
//! threshold that equals an attained conditional.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;
use antecedent_model::compose::{
    ComposedFinite, ComposedModelSpec, Domain, MechanismContext, MechanismDescriptor,
    MechanismForm, MechanismOrigin, MechanismProvenance, ParentBinding, PosteriorDependence,
    TabularRow,
};
use antecedent_model::observation_query::{
    Direction, ObservationQueryError, condition, condition_after_intervention,
    condition_on_measured, intervene, threshold_crossing_observations,
};
use serde_json::Value;

const TOL: f64 = 1e-12;

fn strings(v: &Value) -> Vec<String> {
    v.as_array().unwrap().iter().map(|s| s.as_str().unwrap().to_owned()).collect()
}

fn pairs(v: &Value) -> Vec<(String, String)> {
    v.as_object()
        .unwrap()
        .iter()
        .map(|(k, l)| (k.clone(), l.as_str().unwrap().to_owned()))
        .collect()
}

fn borrowed(p: &[(String, String)]) -> Vec<(&str, &str)> {
    p.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect()
}

fn build(model: &Value) -> ComposedFinite {
    let variables = model["variables"].as_array().unwrap();
    let levels_of = |name: &str| -> Vec<String> {
        strings(&variables.iter().find(|v| v["name"] == name).unwrap()["levels"])
    };
    let mut mechanisms = Vec::new();
    let mut edges = Vec::new();
    for v in variables {
        let name = v["name"].as_str().unwrap();
        let levels = strings(&v["levels"]);
        let parents = strings(&v["parents"]);
        let rows = v["rows"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| TabularRow {
                parent_levels: parents.iter().cloned().zip(strings(&r["config"])).collect(),
                probabilities: levels
                    .iter()
                    .cloned()
                    .zip(r["probs"].as_array().unwrap().iter().map(|p| p.as_f64().unwrap()))
                    .collect(),
            })
            .collect();
        for p in &parents {
            edges.push((p.clone(), name.to_owned()));
        }
        mechanisms.push(MechanismDescriptor {
            variable: name.into(),
            form: MechanismForm::Tabular { levels: levels.clone(), rows },
            parents: parents
                .iter()
                .map(|p| ParentBinding {
                    variable: p.clone(),
                    units: "category".into(),
                    transform: "identity".into(),
                    domain: Domain::Levels(levels_of(p)),
                })
                .collect(),
            conditioning_set: parents.clone(),
            output_units: "category".into(),
            output_transform: "identity".into(),
            output_support: Domain::Levels(levels),
            context: MechanismContext { population: "pop".into(), regime: "observational".into() },
            provenance: MechanismProvenance {
                origin: MechanismOrigin::Declared,
                artifact_id: format!("declared:{name}"),
                evidence_id: format!("assumed:{name}"),
            },
            latents: Vec::new(),
            uncertainty: None,
        });
    }
    ComposedFinite::compile(&ComposedModelSpec {
        target: MechanismContext { population: "pop".into(), regime: "observational".into() },
        mechanisms,
        edges,
        invariances: Vec::new(),
        selection_diagrams: Vec::new(),
        dependence: PosteriorDependence::Unknown,
        admit_declared_mechanisms: true,
    })
    .unwrap()
}

/// The law orders levels by its own convention, so levels are matched by name.
fn compare(got: &[(String, f64)], want: &Value, what: &str) {
    let want = want.as_array().unwrap();
    assert_eq!(got.len(), want.len(), "{what}");
    for w in want {
        let name = w[0].as_str().unwrap();
        let (_, gp) =
            got.iter().find(|(l, _)| l == name).unwrap_or_else(|| panic!("{what}: {name}"));
        let wp = w[1].as_f64().unwrap();
        assert!((gp - wp).abs() <= TOL, "{what}: {name} {gp} vs {wp}");
    }
}

#[test]
fn queries_match_an_exact_rational_enumeration() {
    let models: Value =
        serde_json::from_str(include_str!("oracle_data/finite_law_conditioning_reference.json"))
            .unwrap();
    let ctx = ExecutionContext::for_tests(1);
    let mut counted = [0_usize; 6];
    for model in models.as_array().unwrap() {
        let finite = build(model);
        let law = finite.law(&[], &ctx).unwrap();
        for q in model["queries"].as_array().unwrap() {
            let what = q.to_string();
            let failed = q.get("error").and_then(Value::as_str);
            let outcome = q["outcome"].as_str().unwrap();
            match q["kind"].as_str().unwrap() {
                "condition" => {
                    let ev = pairs(&q["evidence"]);
                    let got = condition(&law, outcome, &borrowed(&ev), &ctx);
                    if failed == Some("zero_evidence") {
                        assert_eq!(
                            got.unwrap_err(),
                            ObservationQueryError::ZeroProbabilityEvidence
                        );
                        counted[5] += 1;
                        continue;
                    }
                    let got = got.unwrap();
                    assert!(
                        (got.evidence_probability - q["evidence_probability"].as_f64().unwrap())
                            .abs()
                            <= TOL,
                        "{what}"
                    );
                    compare(&got.levels, &q["levels"], &what);
                    counted[0] += 1;
                }
                "intervene" => {
                    let d = pairs(&q["do"]);
                    let got = intervene(&finite, outcome, &borrowed(&d), &ctx).unwrap();
                    compare(&got, &q["levels"], &what);
                    counted[1] += 1;
                }
                "after" => {
                    let d = pairs(&q["do"]);
                    let o = pairs(&q["observed"]);
                    let got = condition_after_intervention(
                        &finite,
                        outcome,
                        &borrowed(&d),
                        &borrowed(&o),
                        &ctx,
                    );
                    if failed == Some("zero_evidence") {
                        assert_eq!(
                            got.unwrap_err(),
                            ObservationQueryError::ZeroProbabilityEvidence
                        );
                        counted[5] += 1;
                        continue;
                    }
                    let got = got.unwrap();
                    assert!(
                        (got.evidence_probability - q["evidence_probability"].as_f64().unwrap())
                            .abs()
                            <= TOL,
                        "{what}"
                    );
                    compare(&got.levels, &q["levels"], &what);
                    counted[2] += 1;
                }
                "measured" => {
                    let o = pairs(&q["observed"]);
                    let lik: Vec<f64> = q["likelihood"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|x| x.as_f64().unwrap())
                        .collect();
                    let got = condition_on_measured(
                        &law,
                        outcome,
                        q["construct"].as_str().unwrap(),
                        &borrowed(&o),
                        &lik,
                        &ctx,
                    );
                    if failed == Some("zero_evidence") {
                        assert_eq!(
                            got.unwrap_err(),
                            ObservationQueryError::ZeroProbabilityEvidence
                        );
                        counted[5] += 1;
                        continue;
                    }
                    let got = got.unwrap();
                    assert!(
                        (got.evidence_probability - q["evidence_probability"].as_f64().unwrap())
                            .abs()
                            <= TOL,
                        "{what}"
                    );
                    compare(&got.outcome, &q["outcome_levels"], &what);
                    compare(&got.construct_posterior, &q["construct_levels"], &what);
                    counted[3] += 1;
                }
                "threshold" => {
                    let b = pairs(&q["background"]);
                    let direction =
                        if q["direction"] == "raise" { Direction::Raise } else { Direction::Lower };
                    let got = threshold_crossing_observations(
                        &law,
                        outcome,
                        q["outcome_level"].as_str().unwrap(),
                        q["signal"].as_str().unwrap(),
                        &borrowed(&b),
                        q["threshold"].as_f64().unwrap(),
                        direction,
                        &ctx,
                    );
                    if failed == Some("unreachable") {
                        match got.unwrap_err() {
                            ObservationQueryError::LevelUnreachable { extremum, .. } => assert!(
                                (extremum - q["extremum"].as_f64().unwrap()).abs() <= TOL,
                                "{what}"
                            ),
                            other => panic!("{what}: {other:?}"),
                        }
                        counted[5] += 1;
                        continue;
                    }
                    let got = got.unwrap();
                    let want = q["levels"].as_array().unwrap();
                    assert_eq!(got.levels.len(), want.len(), "{what}");
                    for w in want {
                        let g = got
                            .levels
                            .iter()
                            .find(|g| g.level == w["level"].as_str().unwrap())
                            .unwrap_or_else(|| panic!("{what}"));
                        assert!((g.probability - w["probability"].as_f64().unwrap()).abs() <= TOL);
                        assert!((g.conditional - w["conditional"].as_f64().unwrap()).abs() <= TOL);
                        assert_eq!(g.crosses, w["crosses"].as_bool().unwrap(), "{what}");
                    }
                    assert!(
                        (got.crossing_probability - q["crossing_probability"].as_f64().unwrap())
                            .abs()
                            <= TOL,
                        "{what}"
                    );
                    counted[4] += 1;
                }
                other => panic!("unknown kind {other}"),
            }
        }
    }
    // Every route and the refusals are exercised.
    assert!(counted.iter().all(|&n| n >= 5), "{counted:?}");
}
