//! Scenario evidence for the trial-to-target transport route and the ADMG average effect: the
//! declarations a user can make about weights, missing values, dependence and degenerate
//! designs are each honoured with a known answer or refused with a typed reason.
//!
//! The reference is the contrast counted by hand from the supplied probabilities:
//! `sum over trial rows of (+-1) (1 - s) / (s * arm probability) * y`, divided by the number
//! of target rows.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::{CausalError, RefuteSuite, Study, StudyBuilder, StudyResult, TransportTrialSpec};
use antecedent_core::{
    AverageEffectQuery, CausalQuery, ContinuousDomain, DeclaredSamplingDesign, ExecutionContext,
    GridSpec, ResponseFunctional, ResponseQuery, TransportQuery, VariableId,
};
use antecedent_data::{MissingDeclaration, MissingPolicy, TabularData};
use antecedent_estimate::{LinearAdjustmentAte, SamplingWeights};
use antecedent_graph::{Admg, Dag, DenseNodeId};

fn run_on_large_stack(run: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(run)
        .unwrap()
        .join()
        .expect("body panicked");
}

fn query() -> TransportQuery {
    let response = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(1),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(0),
            GridSpec::Values(Arc::from([0.0, 1.0])),
        ),
    });
    TransportQuery::new(response, "trial", "target", [VariableId::from_raw(0)])
}

fn spec() -> TransportTrialSpec {
    TransportTrialSpec {
        trial: VariableId::from_raw(2),
        selection_probability: VariableId::from_raw(3),
        treatment_probability: VariableId::from_raw(4),
    }
}

const ROWS: usize = 40;
const SELECTION: f64 = 0.5;
const PROPENSITY: f64 = 0.5;

/// Columns `a, y, trial, s, e`: arms alternate, the trial is rows 0 and 1 of every four, and
/// `y = 2 a + a small deterministic wobble`.
fn columns() -> [Vec<f64>; 5] {
    let a: Vec<f64> = (0..ROWS).map(|i| f64::from(u8::from(i % 2 == 0))).collect();
    let y: Vec<f64> = (0..ROWS).map(|i| 2.0 * a[i] + 0.1 * ((i * 7 % 5) as f64 - 2.0)).collect();
    let trial: Vec<f64> = (0..ROWS).map(|i| f64::from(u8::from(i % 4 < 2))).collect();
    [a, y, trial, vec![SELECTION; ROWS], vec![PROPENSITY; ROWS]]
}

fn table(columns: &[Vec<f64>; 5]) -> TabularData {
    TabularData::from_f64_columns([
        ("a", columns[0].as_slice()),
        ("y", columns[1].as_slice()),
        ("trial", columns[2].as_slice()),
        ("s", columns[3].as_slice()),
        ("e", columns[4].as_slice()),
    ])
    .unwrap()
}

/// The contrast counted by hand from the columns, independent of the product.
fn counted(columns: &[Vec<f64>; 5]) -> f64 {
    let targets = columns[2].iter().filter(|&&member| member == 0.0).count() as f64;
    let total: f64 = (0..ROWS)
        .filter(|&row| columns[2][row] == 1.0)
        .map(|row| {
            let (sign, arm) = if columns[0][row] == 1.0 {
                (1.0, PROPENSITY)
            } else {
                (-1.0, 1.0 - PROPENSITY)
            };
            sign * (1.0 - SELECTION) / (SELECTION * arm) * columns[1][row]
        })
        .sum();
    total / targets
}

fn study(
    data: TabularData,
    configure: impl FnOnce(StudyBuilder) -> StudyBuilder,
) -> Result<Study, CausalError> {
    let mut admg = Admg::with_variables(5);
    admg.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    configure(
        Study::tabular(data)
            .graph(admg)
            .query(CausalQuery::Transport(query()))
            .selection_targets(Arc::from([]))
            .transport_trial(spec())
            .refute(RefuteSuite::None),
    )
    .build()
}

fn go(
    data: TabularData,
    configure: impl FnOnce(StudyBuilder) -> StudyBuilder,
) -> Result<StudyResult, CausalError> {
    study(data, configure)?.run(&ExecutionContext::for_tests(5))
}

fn ate(result: &StudyResult) -> f64 {
    result.estimate().as_effect().unwrap().ate
}

fn refusal(result: Result<StudyResult, CausalError>, what: &str) -> CausalError {
    result.err().unwrap_or_else(|| panic!("{what}: an estimate was published"))
}

fn weights() -> Vec<f64> {
    (0..ROWS).map(|i| 1.0 + (i % 3) as f64).collect()
}

#[test]
fn cells_05_the_transport_baseline_equals_the_hand_counted_contrast() {
    run_on_large_stack(|| {
        let base = columns();
        let want = counted(&base);
        assert!((want - 2.0).abs() < 0.2, "{want}");
        let got = ate(&go(table(&base), |b| b).unwrap());
        assert!((got - want).abs() < 1e-12, "{got} vs {want}");
    });
}

#[test]
fn cells_05_transport_weights_are_refused_and_the_probabilities_are_its_design_weights() {
    run_on_large_stack(|| {
        // Observation weights on the table, constant or not, would describe the unweighted rows.
        for weights in [weights(), vec![1.0; ROWS]] {
            let data = table(&columns()).with_weights(Arc::from(weights)).unwrap();
            let error = refusal(go(data, |b| b), "table weights");
            assert_eq!(error.reason_code(), Some("option_not_applicable"), "{error}");
            assert!(error.to_string().contains("sampling_weights.route_unsupported"), "{error}");
        }
        // Weights declared on an estimator the transport route does not run change nothing.
        let declared = SamplingWeights::new(weights()).unwrap();
        let error = refusal(
            go(table(&columns()), |b| {
                b.estimator(LinearAdjustmentAte::new().with_sampling_weights(declared))
            }),
            "estimator weights",
        );
        assert_eq!(error.reason_code(), Some("option_not_applicable"), "{error}");
        assert!(error.to_string().contains("sampling_weights.route_unsupported"), "{error}");
        // The route's own weights are the supplied probabilities: halving the selection
        // probability moves the contrast to the hand count.
        let mut skewed = columns();
        skewed[3] = vec![0.25; ROWS];
        let (a, y, trial) = (&skewed[0], &skewed[1], &skewed[2]);
        let targets = trial.iter().filter(|&&member| member == 0.0).count() as f64;
        let by_hand: f64 = (0..ROWS)
            .filter(|&row| trial[row] == 1.0)
            .map(|row| {
                let sign = if a[row] == 1.0 { 1.0 } else { -1.0 };
                sign * (1.0 - 0.25) / (0.25 * 0.5) * y[row]
            })
            .sum::<f64>()
            / targets;
        let got = ate(&go(table(&skewed), |b| b).unwrap());
        assert!((got - by_hand).abs() < 1e-12, "{got} vs {by_hand}");
    });
}

#[test]
fn cells_05_transport_missing_values_are_refused_never_read_as_a_value() {
    run_on_large_stack(|| {
        let base = columns();
        let want = counted(&base);
        // A declared policy is refused: the route reads its rows in its own way.
        for policy in [MissingPolicy::CompleteCase, MissingPolicy::ErrorOnMissing] {
            let error = refusal(
                go(table(&base), |b| b.missing(MissingDeclaration::new(policy))),
                "missing declaration",
            );
            assert_eq!(error.reason_code(), Some("missing_policy_not_applicable"), "{error}");
        }
        // Row 5 is a control row of the trial (y = -0.2), row 4 a treated one, row 2 a target row.
        let mut hole = columns();
        hole[1][5] = f64::NAN;
        let error = refusal(go(table(&hole), |b| b), "a missing trial outcome");
        assert!(error.to_string().contains("trial outcomes must be finite"), "{error}");
        // The outcome of a target row is never read.
        let mut target_outcome = columns();
        target_outcome[1][2] = f64::NAN;
        let got = ate(&go(table(&target_outcome), |b| b).unwrap());
        assert!((got - want).abs() < 1e-12, "{got} vs {want}");
        // The treatment of a trial row and the trial indicator are read as yes/no, so a
        // missing value would silently become "treated" or "in the trial".
        let mut treatment = columns();
        treatment[0][5] = f64::NAN;
        let mut indicator = columns();
        indicator[2][2] = f64::NAN;
        for (what, data) in [("treatment of a trial row", treatment), ("trial indicator", indicator)]
        {
            let error = refusal(go(table(&data), |b| b), what);
            assert_eq!(error.reason_code(), Some("missing_policy_required"), "{what}: {error}");
        }
        // The treatment of a target row is never read.
        let mut target_treatment = columns();
        target_treatment[0][2] = f64::NAN;
        let got = ate(&go(table(&target_treatment), |b| b).unwrap());
        assert!((got - want).abs() < 1e-12, "{got} vs {want}");
        // A missing probability is outside (0, 1).
        for column in [3, 4] {
            let mut data = columns();
            data[column][4] = f64::NAN;
            let error = refusal(go(table(&data), |b| b), "a missing probability");
            assert!(error.to_string().contains("strictly inside (0,1)"), "{error}");
        }
    });
}

#[test]
fn cells_05_transport_declared_dependence_is_honoured_or_refused() {
    run_on_large_stack(|| {
        let base = columns();
        let want = counted(&base);
        // Independent subjects are what the delta-method interval and the row bootstrap cover.
        let independent = go(table(&base), |b| {
            b.sampling_design(DeclaredSamplingDesign::IndependentSubjectTrajectories)
        })
        .unwrap();
        assert!((ate(&independent) - want).abs() < 1e-12);
        // A clustered panel, or rows declared one dependent series on a non-temporal graph,
        // would be published with an interval that assumes independent rows.
        for design in
            [DeclaredSamplingDesign::ClusteredPanel, DeclaredSamplingDesign::DependentTimeSeries]
        {
            let error =
                refusal(go(table(&base), |b| b.sampling_design(design)), design.as_str());
            assert_eq!(error.reason_code(), Some("option_not_applicable"), "{design:?}: {error}");
            assert!(error.to_string().contains(design.as_str()), "{error}");
            assert!(error.to_string().contains("Next attempt"), "{error}");
        }
    });
}

#[test]
fn cells_05_transport_degenerate_designs_are_refused_or_flagged() {
    run_on_large_stack(|| {
        // The undegenerate design equals the hand count the refusals below depart from.
        let base = columns();
        let got = ate(&go(table(&base), |b| b).unwrap());
        assert!((got - counted(&base)).abs() < 1e-12, "{got}");
        // Probabilities of exactly 0 or 1 give an infinite weight.
        for (column, value) in [(3, 1.0), (3, 0.0), (4, 0.0), (4, 1.0)] {
            let mut data = columns();
            data[column] = vec![value; ROWS];
            let error = refusal(go(table(&data), |b| b), "a degenerate probability");
            assert!(error.to_string().contains("strictly inside (0,1)"), "{error}");
        }
        // No target rows, or no trial rows, leave nothing to transport from or to.
        for member in [0.0, 1.0] {
            let mut data = columns();
            data[2] = vec![member; ROWS];
            let error = refusal(go(table(&data), |b| b), "a one-sided trial indicator");
            assert!(
                error.to_string().contains("source-trial and target-population rows"),
                "{error}"
            );
        }
        // Every trial row on one arm contradicts the supplied arm probability: the contrast is
        // the design-weighted count, and the arm calibration check reports the contradiction.
        let mut one_arm = columns();
        one_arm[0] = vec![1.0; ROWS];
        let result = go(table(&one_arm), |b| b.refute(RefuteSuite::Cheap)).unwrap();
        let arm = result
            .refutations
            .iter()
            .find(|r| r.refuter.as_ref() == "transport.arm_calibration")
            .expect("the arm calibration check");
        assert!(!arm.passed, "{arm:?}");
        assert!((arm.original_ate - 20.0).abs() < 1e-12, "{arm:?}");
        assert!((arm.refuted_ate - 10.0).abs() < 1e-12, "{arm:?}");
    });
}

/// Treatment `t`, outcome `y`, confounder `z`; the counts give risks `.2/.8` when `z = 0` and
/// `.4/.6` when `z = 1` for `t = 0/1`, with `P(z = 1) = 1/2`.
fn projection_table() -> (TabularData, f64) {
    let (mut t, mut y, mut z) = (Vec::new(), Vec::new(), Vec::new());
    // (z, t, y, count)
    let cells: [(f64, f64, f64, usize); 8] = [
        (0.0, 0.0, 0.0, 80),
        (0.0, 0.0, 1.0, 20),
        (0.0, 1.0, 0.0, 20),
        (0.0, 1.0, 1.0, 80),
        (1.0, 0.0, 0.0, 60),
        (1.0, 0.0, 1.0, 40),
        (1.0, 1.0, 0.0, 40),
        (1.0, 1.0, 1.0, 60),
    ];
    for (zz, tt, yy, count) in cells {
        for _ in 0..count {
            t.push(tt);
            y.push(yy);
            z.push(zz);
        }
    }
    // Back-door adjustment on `z`, from the counts: sum_z P(z) (E[y | t=1, z] - E[y | t=0, z]).
    let total: usize = cells.iter().map(|c| c.3).sum();
    let mean = |zz: f64, tt: f64| {
        let rows: usize = cells.iter().filter(|c| c.0 == zz && c.1 == tt).map(|c| c.3).sum();
        let ones: usize =
            cells.iter().filter(|c| c.0 == zz && c.1 == tt && c.2 == 1.0).map(|c| c.3).sum();
        ones as f64 / rows as f64
    };
    let truth: f64 = [0.0, 1.0]
        .iter()
        .map(|&zz| {
            let weight: usize = cells.iter().filter(|c| c.0 == zz).map(|c| c.3).sum();
            weight as f64 / total as f64 * (mean(zz, 1.0) - mean(zz, 0.0))
        })
        .sum();
    (
        TabularData::from_f64_columns([
            ("t", t.as_slice()),
            ("y", y.as_slice()),
            ("z", z.as_slice()),
        ])
        .unwrap(),
        truth,
    )
}

#[test]
fn cells_05_the_average_effect_on_a_dag_equals_the_admg_projection_and_the_adjustment_formula() {
    let (data, truth) = projection_table();
    assert!((truth - 0.4).abs() < 1e-12, "{truth}");
    let node = DenseNodeId::from_raw;
    let edges = [(2_u32, 0_u32), (2, 1), (0, 1)];
    let effect = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let mut dag = Dag::with_variables(3);
    let mut admg = Admg::with_variables(3);
    for (from, to) in edges {
        dag.insert_directed(node(from), node(to)).unwrap();
        admg.insert_directed(node(from), node(to)).unwrap();
    }
    let ctx = ExecutionContext::for_tests(5);
    let on_dag = Study::tabular(data.clone())
        .graph(dag)
        .query(effect.clone())
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap();
    let on_admg = Study::tabular(data)
        .graph(admg)
        .query(effect)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap();
    assert!((ate(&on_dag) - truth).abs() < 1e-9, "{}", ate(&on_dag));
    assert!((ate(&on_admg) - truth).abs() < 1e-9, "{}", ate(&on_admg));
    assert!((ate(&on_dag) - ate(&on_admg)).abs() < 1e-12);
}
