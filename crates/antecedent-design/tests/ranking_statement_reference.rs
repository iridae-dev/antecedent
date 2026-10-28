//! Tie-shared ranking statements and their Monte Carlo errors against exact enumeration.
//!
//! `oracle_data/ranking_statement_reference.py` (frozen in the JSON beside it) defines tie
//! sharing by averaging over every random linear order of the candidates, not by counting
//! strictly-greater and tied candidates, and computes the iid, self-normalised-weight and
//! batch-means errors from their definitions in exact rationals.

use antecedent_core::ExecutionContext;
use antecedent_design::decision_reliability::{
    BatchPlan, CandidateCoupling, CandidateDraws, CandidateOrigin, DecisionDraws, DrawSegment,
    RankingRequest, SourceParadigm, StreamId, StreamPurpose, WithinStreamDependence, rank,
};
use serde_json::Value;

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().expect("array").iter().map(|x| x.as_f64().expect("number")).collect()
}

fn close(got: f64, want: f64, ctx: &str) {
    assert!((got - want).abs() <= 1e-12 * (1.0 + want.abs()), "{ctx}: {got} vs {want}");
}

#[test]
fn ranking_probabilities_and_errors_match_exact_enumeration() {
    let cases: Value = serde_json::from_str(include_str!("oracle_data/ranking_statement_reference.json"))
        .expect("frozen reference parses");
    let ctx = ExecutionContext::for_tests(7);
    for case in cases.as_array().expect("cases") {
        let name = case["name"].as_str().expect("name");
        let columns: Vec<Vec<f64>> = case["utilities"].as_array().expect("u").iter().map(floats).collect();
        let n = columns[0].len();
        let shared = DrawSegment::whole(StreamId::new(20_240, StreamPurpose::Selection, 0, "ref"), n);
        let draws = DecisionDraws {
            utility_unit: "utility".into(),
            producer: "reference".into(),
            source: SourceParadigm::KnownLaw,
            coupling: CandidateCoupling::SharedSimulation,
            within_stream: if case["dependence"] == "autocorrelated" {
                WithinStreamDependence::Autocorrelated
            } else {
                WithinStreamDependence::Independent
            },
            weights: case["weights"].as_array().map(|w| w.iter().map(|x| x.as_f64().expect("w")).collect()),
            repeated_evaluations: 0,
            origin: CandidateOrigin::Enumerated,
            search_record: None,
            candidates: case["ids"]
                .as_array()
                .expect("ids")
                .iter()
                .zip(&columns)
                .map(|(id, u)| CandidateDraws {
                    id: id.as_str().expect("id").to_owned(),
                    utilities: u.clone(),
                    segment: shared.clone(),
                })
                .collect(),
        };
        let top_k: Vec<u32> = case["top_k"].as_array().expect("k").iter().map(|k| k.as_u64().expect("k") as u32).collect();
        let request = RankingRequest {
            baseline: case["baseline"].as_str().map(str::to_owned),
            top_k: top_k.clone(),
            batch_plan: case["batches"].as_u64().map_or(BatchPlan::SquareRoot, |b| BatchPlan::Fixed(b as u32)),
        };
        let report = rank(&draws, &request, &ctx).unwrap_or_else(|e| panic!("{name}: {e:?}"));
        for want in case["statements"].as_array().expect("statements") {
            let id = want["id"].as_str().expect("id");
            let got = report.statements.iter().find(|s| s.id == id).expect("candidate");
            let check = |g: &antecedent_design::decision_reliability::McEstimate, w: &Value, what: &str| {
                let ctxt = format!("{name}/{id}/{what}");
                close(g.value, w["value"].as_f64().expect("v"), &format!("{ctxt} value"));
                close(g.standard_error, w["se"].as_f64().expect("se"), &format!("{ctxt} se"));
                close(g.effective_sample_size, w["ess"].as_f64().expect("ess"), &format!("{ctxt} ess"));
            };
            check(&got.expected_utility, &want["expected_utility"], "utility");
            check(&got.probability_best, &want["best"], "best");
            for (g, w) in got.probability_top_k.iter().zip(want["top_k"].as_array().expect("tk")) {
                check(&g.estimate, w, &format!("top{}", g.k));
            }
            match (&got.probability_better_than_baseline, &want["baseline"]) {
                (Some(g), w) if !w.is_null() => check(g, w, "baseline"),
                (None, w) => assert!(w.is_null(), "{name}/{id}: missing baseline statement"),
                (Some(_), _) => panic!("{name}/{id}: unexpected baseline statement"),
            }
        }
        let total: f64 = report.statements.iter().map(|s| s.probability_best.value).sum();
        close(total, 1.0, &format!("{name} best probabilities sum"));
    }
}
