//! Monte Carlo error, Wald and Wilson intervals, the best-candidate confidence set and the
//! selection-corrected interval against a definition-level reference
//! (`oracle_data/decision_reliability_reference.py`, exact rational arithmetic for the
//! Maritz-Jarrett weights, the Rockafellar-Uryasev maximum for the conditional value at risk,
//! direct enumeration for ranking probabilities).
//!
//! Regimes: quarter-grid draws with exact ties, a heavy-tailed sample, a five-draw sample,
//! an autocorrelated stream under three batch plans, extreme importance weights, shared and
//! independent coupling, simultaneous and marginal multiplicity.

use antecedent_core::ExecutionContext;
use antecedent_design::decision_reliability::{
    BatchPlan, CandidateCoupling, CandidateDraws, CandidateOrigin, ConfidenceSetBasis,
    DecisionDraws, DrawSegment, Functional, IntervalMultiplicity, IntervalRequest, PairCoupling,
    ProbabilityQuantity, PublishedRoute, RankingRequest, ReliabilityError, ReliabilityRequest,
    RiskTail, SourceParadigm, StreamId, StreamPurpose, ThresholdSide, WithinStreamDependence,
    derive_uncertainty, estimate, estimate_difference,
};
use serde_json::Value;

fn reference() -> Value {
    serde_json::from_str(include_str!("oracle_data/decision_reliability_reference.json"))
        .expect("frozen reference")
}

fn floats(value: &Value) -> Vec<f64> {
    value.as_array().expect("array").iter().map(|v| v.as_f64().expect("number")).collect()
}

fn functional(value: &Value) -> Functional {
    match value["kind"].as_str().expect("kind") {
        "expected" => Functional::ExpectedUtility,
        "threshold" => Functional::ThresholdProbability {
            threshold: value["threshold"].as_f64().expect("threshold"),
            side: if value["side"] == "above" {
                ThresholdSide::Above
            } else {
                ThresholdSide::AtOrBelow
            },
        },
        "quantile" => Functional::Quantile { p: value["p"].as_f64().expect("p") },
        "cvar" => Functional::Cvar {
            alpha: value["alpha"].as_f64().expect("alpha"),
            tail: if value["tail"] == "lower" { RiskTail::Lower } else { RiskTail::Upper },
        },
        other => panic!("unknown functional {other}"),
    }
}

fn dependence(value: &Value) -> WithinStreamDependence {
    if value == "independent" {
        WithinStreamDependence::Independent
    } else {
        WithinStreamDependence::Autocorrelated
    }
}

fn plan(value: &Value) -> BatchPlan {
    value.as_u64().map_or(BatchPlan::SquareRoot, |b| BatchPlan::Fixed(b as u32))
}

fn close(got: f64, want: f64, rel: f64, what: &str) {
    assert!(
        (got - want).abs() <= rel * want.abs().max(1e-3) + 1e-12,
        "{what}: got {got}, reference {want}"
    );
}

#[test]
fn estimates_match_the_definition_level_reference() {
    let reference = reference();
    let cases = reference["estimates"].as_array().expect("cases");
    assert_eq!(cases.len(), 63);
    for case in cases {
        let values = floats(&case["values"]);
        let weights = (!case["weights"].is_null()).then(|| floats(&case["weights"]));
        let f = functional(&case["functional"]);
        let got = estimate(
            &values,
            weights.as_deref(),
            dependence(&case["dependence"]),
            plan(&case["plan"]),
            &f,
        );
        let tag = format!("{} {}", case["name"], case["functional"]);
        let got = match got {
            Ok(g) => g.estimate,
            Err(e) => panic!("{tag}: {e:?}"),
        };
        let want = &case["expect"];
        close(got.value, want["value"].as_f64().unwrap(), 1e-12, &format!("{tag} value"));
        // Maritz-Jarrett drops index mass beyond twelve index standard deviations.
        close(got.standard_error, want["se"].as_f64().unwrap(), 1e-9, &format!("{tag} se"));
        close(
            got.effective_sample_size,
            want["ess"].as_f64().unwrap(),
            1e-9,
            &format!("{tag} ess"),
        );
        assert_eq!(got.n_draws, want["n"].as_u64().unwrap(), "{tag}");
    }
}

#[test]
fn differences_match_the_definition_level_reference() {
    let reference = reference();
    let cases = reference["differences"].as_array().expect("cases");
    assert!(cases.len() > 40);
    for case in cases {
        let (a, b) = (floats(&case["a"]), floats(&case["b"]));
        let weights = (!case["weights"].is_null()).then(|| floats(&case["weights"]));
        let coupling = if case["coupling"] == "shared" {
            PairCoupling::SharedStream
        } else {
            PairCoupling::IndependentStreams
        };
        let f = functional(&case["functional"]);
        let tag = format!("{} {} {}", case["name"], case["functional"], case["coupling"]);
        let got = estimate_difference(
            &a,
            &b,
            weights.as_deref(),
            dependence(&case["dependence"]),
            plan(&case["plan"]),
            &f,
            coupling,
        )
        .unwrap_or_else(|e| panic!("{tag}: {e:?}"));
        let want = &case["expect"];
        close(got.value, want["value"].as_f64().unwrap(), 1e-12, &format!("{tag} value"));
        close(got.standard_error, want["se"].as_f64().unwrap(), 1e-9, &format!("{tag} se"));
        close(got.standard_error_a, want["se_a"].as_f64().unwrap(), 1e-9, &format!("{tag} se_a"));
        close(got.standard_error_b, want["se_b"].as_f64().unwrap(), 1e-9, &format!("{tag} se_b"));
        match (got.effective_sample_size, want["ess"].as_f64()) {
            (Some(g), Some(w)) => close(g, w, 1e-9, &format!("{tag} ess")),
            (None, None) => {}
            other => panic!("{tag}: ess presence {other:?}"),
        }
    }
}

fn carrier(case: &Value) -> DecisionDraws {
    let candidates = case["candidates"].as_object().expect("candidates");
    let shared = case["shared"].as_bool().expect("shared");
    let n = floats(candidates.values().next().expect("column")).len();
    DecisionDraws {
        utility_unit: "utility".into(),
        producer: "reference".into(),
        source: SourceParadigm::KnownLaw,
        coupling: if shared {
            CandidateCoupling::SharedSimulation
        } else {
            CandidateCoupling::DeclaredIndependent
        },
        within_stream: dependence(&case["dependence"]),
        weights: (!case["weights"].is_null()).then(|| floats(&case["weights"])),
        repeated_evaluations: 0,
        origin: CandidateOrigin::Enumerated,
        search_record: None,
        candidates: candidates
            .iter()
            .map(|(id, column)| CandidateDraws {
                id: id.clone(),
                utilities: floats(column),
                segment: DrawSegment::whole(
                    StreamId::new(
                        777,
                        StreamPurpose::Selection,
                        0,
                        if shared { "shared" } else { id.as_str() },
                    ),
                    n,
                ),
            })
            .collect(),
    }
}

fn check_interval(
    got: &antecedent_design::decision_reliability::NumericalInterval,
    want: &Value,
    tag: &str,
) {
    close(got.estimate, want["estimate"].as_f64().unwrap(), 1e-12, &format!("{tag} estimate"));
    close(got.standard_error, want["se"].as_f64().unwrap(), 1e-9, &format!("{tag} se"));
    close(got.effective_sample_size, want["ess"].as_f64().unwrap(), 1e-9, &format!("{tag} ess"));
    close(got.lower, want["lower"].as_f64().unwrap(), 1e-8, &format!("{tag} lower"));
    close(got.upper, want["upper"].as_f64().unwrap(), 1e-8, &format!("{tag} upper"));
    close(got.level, want["level"].as_f64().unwrap(), 1e-12, &format!("{tag} level"));
}

#[test]
fn derived_uncertainty_matches_the_definition_level_reference() {
    let reference = reference();
    let ctx = ExecutionContext::for_tests(11);
    for case in reference["derive"].as_array().expect("scenarios") {
        let name = case["name"].as_str().unwrap();
        let draws = carrier(case);
        let ranking = case["ranking"].as_object().map(|r| RankingRequest {
            baseline: r["baseline"].as_str().map(str::to_owned),
            top_k: r["top_k"]
                .as_array()
                .unwrap()
                .iter()
                .map(|k| k.as_u64().unwrap() as u32)
                .collect(),
            batch_plan: plan(&case["plan"]),
        });
        let request = ReliabilityRequest {
            functional: functional(&case["functional"]),
            batch_plan: plan(&case["plan"]),
            ranking,
            ..ReliabilityRequest::expected_utility()
        };
        let interval = IntervalRequest {
            level: case["level"].as_f64().unwrap(),
            multiplicity: if case["multiplicity"] == "marginal" {
                IntervalMultiplicity::Marginal
            } else {
                IntervalMultiplicity::Simultaneous
            },
        };
        let got = derive_uncertainty(&draws, None, &request, &interval, &ctx)
            .unwrap_or_else(|e| panic!("{name}: {e:?}"));
        let want = &case["expect"];
        assert_eq!(got.decided_id, want["decided"].as_str().unwrap(), "{name}");
        assert_eq!(
            got.numerical.intervals_reported,
            want["intervals_reported"].as_u64().unwrap(),
            "{name}"
        );
        for utility in &got.numerical.utilities {
            check_interval(
                &utility.interval,
                &want["utilities"][&utility.id],
                &format!("{name} u {}", utility.id),
            );
        }
        assert_eq!(
            got.numerical.differences_vs_decision.len(),
            want["differences"].as_object().unwrap().len()
        );
        for difference in &got.numerical.differences_vs_decision {
            check_interval(
                &difference.interval,
                &want["differences"][&difference.id],
                &format!("{name} d {}", difference.id),
            );
        }
        let expected_probs = want["probabilities"].as_array().unwrap();
        assert_eq!(got.numerical.probabilities.len(), expected_probs.len(), "{name}");
        for (probability, expected) in got.numerical.probabilities.iter().zip(expected_probs) {
            assert_eq!(probability.id, expected["id"].as_str().unwrap());
            let label = match probability.quantity {
                ProbabilityQuantity::Best => "best".to_owned(),
                ProbabilityQuantity::TopK { k } => format!("top_{k}"),
                ProbabilityQuantity::BetterThanBaseline => "baseline".to_owned(),
            };
            assert_eq!(label, expected["quantity"].as_str().unwrap(), "{name}");
            check_interval(
                &probability.interval,
                expected,
                &format!("{name} p {} {label}", probability.id),
            );
            assert!(probability.interval.lower >= 0.0 && probability.interval.upper <= 1.0);
        }
        if interval.multiplicity == IntervalMultiplicity::Simultaneous {
            // A simultaneous request is a construction no coverage record measured.
            assert!(
                matches!(
                    got.publish(PublishedRoute::RankingProbability),
                    Err(ReliabilityError::CellNotLicensed { .. })
                ),
                "{name}"
            );
        }
        let set = &got.selection.confidence_set;
        let expected_set = &want["confidence_set"];
        assert_eq!(
            set.basis,
            if expected_set["basis"] == "paired_comparisons_with_best" {
                ConfidenceSetBasis::PairedComparisonsWithBest
            } else {
                ConfidenceSetBasis::MarginalBonferroni
            },
            "{name}"
        );
        let members: Vec<&str> = set.members.iter().map(String::as_str).collect();
        let expected_members: Vec<&str> = expected_set["members"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m.as_str().unwrap())
            .collect();
        assert_eq!(members, expected_members, "{name}");
        close(
            set.critical_value,
            expected_set["critical"].as_f64().unwrap(),
            1e-8,
            &format!("{name} set z"),
        );
        let corrected = got.selection.corrected.as_ref().expect("selected by rule");
        let expected = &want["corrected"];
        close(corrected.in_sample, expected["in_sample"].as_f64().unwrap(), 1e-12, name);
        close(corrected.standard_error, expected["se"].as_f64().unwrap(), 1e-9, name);
        close(corrected.numerical_half_width, expected["numerical"].as_f64().unwrap(), 1e-8, name);
        close(
            corrected.selection_extra_half_width,
            expected["extra"].as_f64().unwrap(),
            1e-7,
            name,
        );
        close(corrected.lower, expected["lower"].as_f64().unwrap(), 1e-8, name);
        close(corrected.upper, expected["upper"].as_f64().unwrap(), 1e-8, name);
        close(corrected.critical_value, expected["critical"].as_f64().unwrap(), 1e-8, name);
    }
}
