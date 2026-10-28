//! Invalid declarations refused with a typed reason on the local-ratio, mean-curve and ADMG
//! routes.
//!
//! Each route first runs on a valid declaration and recovers its closed form: the fuzzy-jump
//! local ratio recovers the treatment effect 3, and the mean curve of `y = 3 a + z` rises by 3
//! per unit of the treatment. A non-positive or non-finite bandwidth, a non-finite cutoff, a
//! cutoff with no support on one side, an empty, non-finite or non-increasing response grid, and
//! an average effect on an ADMG with an unmeasured treatment-outcome confounder and no
//! instrument are each refused with the registered detail rather than estimated.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::{LocalPolynomialRatioQuery, RefuteSuite, Study};
use antecedent_core::{
    AverageEffectQuery, CausalQuery, ContinuousDomain, ExecutionContext, GridSpec,
    ResponseFunctional, ResponseIdentification, ResponseQuery, ResponseValue, VariableId,
};
use antecedent_data::TabularData;
use antecedent_graph::{Admg, Dag, DenseNodeId};

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn fuzzy_jump() -> TabularData {
    let (mut x, mut treatment, mut outcome) = (Vec::new(), Vec::new(), Vec::new());
    for step in 1..80 {
        for sign in [-1.0, 1.0] {
            let score = sign * f64::from(step) / 80.0;
            for replicate in 0..4 {
                let dose = if sign < 0.0 {
                    f64::from(u8::from(replicate == 0))
                } else {
                    f64::from(u8::from(replicate != 3))
                };
                x.push(score);
                treatment.push(dose);
                outcome.push(1.0 + 2.0 * score + 0.5 * score * score + 3.0 * dose + [-0.02, -0.01, 0.01, 0.02][replicate]);
            }
        }
    }
    TabularData::from_f64_columns([("x", x.as_slice()), ("t", treatment.as_slice()), ("y", outcome.as_slice())]).unwrap()
}

fn ratio(cutoff: f64, bandwidth: f64) -> CausalQuery {
    CausalQuery::LocalPolynomialRatio(LocalPolynomialRatioQuery {
        outcome: v(2), treatment: v(1), running: v(0), cutoff, bandwidth, kink: false, options: None,
    })
}

fn refusal(result: Result<antecedent::StudyResult, antecedent::CausalError>, what: &str) -> antecedent::CausalError {
    result.err().unwrap_or_else(|| panic!("{what}: published an estimate"))
}

fn ratio_run(cutoff: f64, bandwidth: f64) -> Result<antecedent::StudyResult, antecedent::CausalError> {
    Study::tabular(fuzzy_jump())
        .query(ratio(cutoff, bandwidth))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .and_then(|study| study.run(&ExecutionContext::for_tests(5)))
}

#[test]
fn the_local_ratio_recovers_its_effect_and_refuses_an_invalid_bandwidth_cutoff_or_support() {
    let effect = ratio_run(0.0, 1.0).unwrap().estimate().as_effect().unwrap().ate;
    assert!((effect - 3.0).abs() < 1e-8, "{effect}");
    for (label, cutoff, bandwidth) in [
        ("zero bandwidth", 0.0, 0.0),
        ("negative bandwidth", 0.0, -1.0),
        ("non-finite bandwidth", 0.0, f64::NAN),
        ("non-finite cutoff", f64::NAN, 1.0),
    ] {
        let error = refusal(ratio_run(cutoff, bandwidth), label);
        assert!(
            error.to_string().contains("cutoff must be finite and bandwidth positive"),
            "{label}: {error}"
        );
    }
    let error = refusal(ratio_run(5.0, 1.0), "a cutoff outside the data");
    assert!(error.to_string().contains("lacks full-rank support on the left side"), "{error}");
}

fn curve_data() -> (TabularData, Dag) {
    let n = 100;
    let z: Vec<f64> = (0..n).map(|i| (f64::from(i) * 0.37).sin()).collect();
    let a: Vec<f64> = (0..n as usize).map(|i| z[i] + (i as f64 * 0.71).cos()).collect();
    let y: Vec<f64> = (0..n as usize).map(|i| 3.0 * a[i] + z[i]).collect();
    let data = TabularData::from_f64_columns([
        ("a", a.as_slice()),
        ("y", y.as_slice()),
        ("z", z.as_slice()),
    ])
    .unwrap();
    let mut dag = Dag::with_variables(3);
    for (from, to) in [(2, 0), (2, 1), (0, 1)] {
        dag.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    (data, dag)
}

fn curve_run(grid: Vec<f64>) -> Result<antecedent::StudyResult, antecedent::CausalError> {
    let (data, dag) = curve_data();
    let query = CausalQuery::Response(ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: v(1),
        treatment: ContinuousDomain::new(v(0), GridSpec::Values(grid.into())),
    }));
    Study::tabular(data)
        .graph(dag)
        .query(query)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .and_then(|study| study.run(&ExecutionContext::for_tests(5)))
}

#[test]
fn the_mean_curve_rises_by_its_slope_and_refuses_an_invalid_grid() {
    let result = curve_run(vec![-0.5, 0.5]).unwrap();
    let ResponseIdentification::PointIdentified(ResponseValue::Surface { mean, .. }) =
        &result.response().unwrap().estimate
    else {
        panic!("a point-identified surface");
    };
    // y = 3 a + z with z adjusted: one unit of treatment moves the curve by 3.
    assert!((mean[1] - mean[0] - 3.0).abs() < 0.05, "{mean:?}");
    let error = refusal(curve_run(vec![]), "an empty grid");
    assert!(error.to_string().contains("a response grid requires at least two points"), "{error}");
    for (label, grid) in [("non-finite", vec![0.0, f64::NAN]), ("repeated", vec![0.5, 0.5])] {
        let error = refusal(curve_run(grid), label);
        assert!(
            error.to_string().contains("response-grid values must be finite and strictly increasing"),
            "{label}: {error}"
        );
    }
}

#[test]
fn an_admg_with_an_unmeasured_treatment_outcome_confounder_is_refused_as_not_identified() {
    let (data, _) = curve_data();
    let node = DenseNodeId::from_raw;
    let mut bow = Admg::with_variables(3);
    for (from, to) in [(2, 0), (2, 1), (0, 1)] {
        bow.insert_directed(node(from), node(to)).unwrap();
    }
    bow.insert_bidirected(node(0), node(1)).unwrap();
    let error = refusal(
        Study::tabular(data)
            .graph(bow)
            .query(CausalQuery::AverageEffect(AverageEffectQuery::binary_ate(v(0), v(1))))
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .and_then(|study| study.run(&ExecutionContext::for_tests(5))),
        "a bow-arc effect",
    );
    assert_eq!(error.reason_code(), Some("effect_not_identified"), "{error}");
    assert!(error.to_string().contains("identification status not_identified"), "{error}");
}
