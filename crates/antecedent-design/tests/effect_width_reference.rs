//! `ReduceEffectPosteriorWidth` against 80-digit full-inverse evaluation of the treatment
//! standard error (`oracle_data/effect_width_reference.py`).
//!
//! The ranker's reduction is `se(baseline) - se(updated design)` with
//! `se = sqrt(sigma2 * [G^-1]_tt)`. The reference builds each updated Gram the way the candidate
//! describes it (rows scaled, an environment Gram added, columns bordered one at a time, an
//! intervention Gram supplied) and inverts it in full; the product uses one Cholesky solve. The
//! cases include a negative reduction (a noisier environment), a measurement plan that mixes
//! matched and unmatched variables, and a mis-shaped column, a repeated column and a singular
//! baseline that must be refused.

use std::sync::Arc;

use antecedent_core::{EnvironmentId, ExecutionContext, QueryId, VariableId};
use antecedent_design::candidate::{
    EnvironmentPlan, ExperimentPlan, MeasurementPlan, SamplingPlan,
};
use antecedent_design::{
    CandidateDesign, DesignCost, DesignError, DesignEvaluationContext, DesignObjective,
    DesignRanker, EffectWidthContext, EnvironmentGramSpec, InterventionDesignEffect,
    MeasureColumnSpec,
};
use antecedent_prob::{GraphIdentFlag, WeightedGraphSamples};
use serde_json::Value;

fn reference() -> Value {
    serde_json::from_str(include_str!("oracle_data/effect_width_reference.json"))
        .expect("frozen reference")
}

fn flat(value: &Value) -> Arc<[f64]> {
    value
        .as_array()
        .expect("rows")
        .iter()
        .flat_map(|row| row.as_array().expect("row").iter().map(|x| x.as_f64().expect("x")))
        .collect()
}

fn f(value: &Value) -> f64 {
    value.as_f64().expect("number")
}

fn context(r: &Value) -> EffectWidthContext {
    let specs: Vec<MeasureColumnSpec> = r["measure"]
        .as_object()
        .expect("measure")
        .iter()
        .map(|(var, spec)| MeasureColumnSpec {
            variable: VariableId::from_raw(var.parse().expect("var")),
            cross: spec["cross"]
                .as_array()
                .expect("cross")
                .iter()
                .map(f)
                .collect::<Vec<_>>()
                .into(),
            self_dot: f(&spec["self_dot"]),
            sigma2_after: spec["sigma2_after"].as_f64(),
        })
        .collect();
    EffectWidthContext {
        xtx: flat(&r["gram"]),
        sigma2: f(&r["sigma2"]),
        treatment_col: r["treatment_col"].as_u64().expect("t") as usize,
        n: r["n"].as_u64().expect("n"),
        measure_columns: Some(specs.into()),
        intervention_design: Some(InterventionDesignEffect {
            xtx: flat(&r["intervention_gram"]),
            sigma2: f(&r["intervention_sigma2"]),
            n: 40,
        }),
        environment_grams: Some(
            vec![EnvironmentGramSpec {
                environment: EnvironmentId::from_raw(1),
                xtx: flat(&r["env_gram"]),
                n: 18,
                sigma2: Some(f(&r["env_sigma2"])),
            }]
            .into(),
        ),
    }
}

fn score(ew: &EffectWidthContext, candidate: CandidateDesign) -> Result<f64, DesignError> {
    let graphs =
        WeightedGraphSamples::new([1.0], [GraphIdentFlag::Unidentified], [0]).expect("graphs");
    let eval: DesignEvaluationContext<'_, (), ()> = DesignEvaluationContext {
        graphs: &graphs,
        effect_width: Some(ew),
        model_loglik: None,
        decisions: None,
        query_id_unlock: None,
        env_id_unlock: None,
        identified_under_intervention: None,
        graph_features: None,
    };
    let ranking = DesignRanker::new().rank(
        &DesignObjective::ReduceEffectPosteriorWidth { query: QueryId::from_raw(0) },
        &[candidate],
        &eval,
        &ExecutionContext::for_tests(5),
    )?;
    assert_eq!(ranking.ranked.len(), 1);
    Ok(ranking.ranked[0].score)
}

fn sampling(n: u64) -> CandidateDesign {
    CandidateDesign::IncreaseSamplingRate(SamplingPlan {
        additional_samples: n,
        cost: DesignCost::zero(),
        tag: 0,
    })
}

fn measure(vars: &[u32]) -> CandidateDesign {
    CandidateDesign::Measure(MeasurementPlan {
        variables: vars.iter().copied().map(VariableId::from_raw).collect(),
        cost: DesignCost::zero(),
        tag: 0,
    })
}

fn environment(id: u32, rows: u64) -> CandidateDesign {
    CandidateDesign::ObserveEnvironment(EnvironmentPlan {
        environment: EnvironmentId::from_raw(id),
        additional_rows: rows,
        cost: DesignCost::zero(),
        tag: 0,
    })
}

fn close(got: f64, want: f64, tol: f64, what: &str) {
    assert!((got - want).abs() <= tol * (1.0 + want.abs()), "{what}: {got} vs {want}");
}

#[test]
fn every_candidate_kind_reduces_the_treatment_se_by_the_full_inverse_value() {
    let r = reference();
    let ew = context(&r);
    for case in r["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let candidate = match name {
            "sampling" | "sampling_zero" => sampling(case["param"].as_u64().unwrap()),
            "env_with_gram" => environment(1, 7),
            "env_rows_only" | "env_rows_zero" => environment(2, case["param"].as_u64().unwrap()),
            "intervene" => CandidateDesign::Intervene(ExperimentPlan {
                targets: Arc::from([VariableId::from_raw(1)]),
                cost: DesignCost::zero(),
                tag: 0,
            }),
            _ => {
                let vars: Vec<u32> = case["vars"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_u64().unwrap() as u32)
                    .collect();
                measure(&vars)
            }
        };
        close(score(&ew, candidate).unwrap(), f(&case["want"]), 1e-12, name);
    }
}

#[test]
fn a_noisier_environment_is_a_negative_reduction_and_the_gram_wins_over_declared_rows() {
    let r = reference();
    let ew = context(&r);
    let with_gram = score(&ew, environment(1, 7)).unwrap();
    assert!(with_gram < 0.0, "{with_gram}");
    // `additional_rows` is not consulted once the environment has a Gram: the Gram carries n.
    assert_eq!(with_gram, score(&ew, environment(1, 100_000)).unwrap());
}

#[test]
fn an_ill_conditioned_gram_keeps_the_reduction_to_the_full_inverse_value() {
    let r = reference();
    let ill = &r["ill"];
    let ew = EffectWidthContext {
        xtx: flat(&ill["gram"]),
        sigma2: 1.0,
        treatment_col: ill["treatment_col"].as_u64().unwrap() as usize,
        n: ill["n"].as_u64().unwrap(),
        measure_columns: None,
        intervention_design: None,
        environment_grams: None,
    };
    let got = score(&ew, sampling(ill["sampling_param"].as_u64().unwrap())).unwrap();
    // Condition number about 6e5 of the Gram: allow the corresponding loss of digits.
    close(got, f(&ill["sampling_want"]), 1e-8, "ill-conditioned sampling reduction");
}

#[test]
fn malformed_designs_are_refused_not_scored_as_zero() {
    let r = reference();
    let ew = context(&r);
    // Variable 8's column has four cross-products but the baseline design has three columns.
    assert!(matches!(score(&ew, measure(&[8])), Err(DesignError::Shape(_))));
    // The same column twice: after the first border the design has four columns, so the
    // three-entry cross-product no longer fits and the plan is refused.
    assert!(matches!(score(&ew, measure(&[7, 7])), Err(DesignError::Shape(_))));
    // An environment Gram of the wrong size.
    let mut bad = ew.clone();
    bad.environment_grams = Some(
        vec![EnvironmentGramSpec {
            environment: EnvironmentId::from_raw(1),
            xtx: Arc::from([1.0, 0.0, 0.0, 1.0]),
            n: 4,
            sigma2: None,
        }]
        .into(),
    );
    assert!(matches!(score(&bad, environment(1, 3)), Err(DesignError::Shape(_))));
    // A non-positive residual variance.
    let mut zero = ew.clone();
    zero.sigma2 = 0.0;
    assert!(matches!(score(&zero, sampling(5)), Err(DesignError::Config(_))));
    // A singular baseline Gram is an error, not "no information gain".
    let mut singular = ew.clone();
    singular.xtx = Arc::from([1.0, 1.0, 1.0, 1.0]);
    singular.treatment_col = 1;
    assert!(matches!(score(&singular, sampling(5)), Err(DesignError::Numerical(_))));
    // A treatment column outside the design.
    let mut outside = ew;
    outside.treatment_col = 3;
    assert!(matches!(score(&outside, sampling(5)), Err(DesignError::Shape(_))));
}
