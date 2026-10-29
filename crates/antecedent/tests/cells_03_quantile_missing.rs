//! Missing outcomes and held-out treatment levels on the quantile contrast, and missing outcomes
//! and graph classes on the sharp discontinuity.
//!
//! The quantile contrast is the average-effect route with a median outcome functional and
//! the cross-fitted AIPW estimator. Its missing-data policies are the route cell's: complete case
//! is exactly the fit on the table with the incomplete rows deleted, the error policy refuses a
//! missing outcome by name, inverse-probability weighting is refused (only randomized outcomes
//! have a responder-weighted estimator), and multiple imputation pools the contrast over completed
//! tables by Rubin's rules, written out here against the public imputer. A declared treatment
//! level that no row carries is refused, never extrapolated.
//!
//! The sharp discontinuity runs from a `Dag`, or an `Admg` with no latent confounding, and is
//! refused for latent confounding, an equivalence class or a temporal graph; its complete-case
//! policy is row deletion, and multiple imputation pools the discontinuity by Rubin's rules.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(clippy::float_cmp, clippy::cast_precision_loss, clippy::too_many_lines)]

use antecedent::{
    CausalError, EstimatorId, IdentifierId, IntoGraphInput, LocalPolynomialRatioQuery, RefuteSuite,
    Study, StudyResult,
};
use antecedent_core::{AverageEffectQuery, ExecutionContext, Lag, OutcomeFunctional, VariableId};
use antecedent_data::{ImputationSpec, MissingDeclaration, MissingPolicy, TabularData};
use antecedent_estimate::multiple_imputation::{ImputationColumn, ImputationUnits, Imputer};
use antecedent_estimate::{AipwAte, OtherLevels};
use antecedent_graph::{Admg, Cpdag, Dag, DenseNodeId, Pag, TemporalDag, ensure_lagged};

const ROWS: usize = 200;

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(0x0C31_0003)
}

fn dag3() -> Dag {
    let mut graph = Dag::with_variables(3);
    for (from, to) in [(2_u32, 0_u32), (2, 1), (0, 1)] {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    graph
}

// ---------------------------------------------------------------------------------------------
// The quantile contrast.
// ---------------------------------------------------------------------------------------------

/// `t`, `y`, `z` with a median shift of about one and the outcome missing on `gaps`.
fn quantile_columns(gaps: &[usize]) -> [Vec<f64>; 3] {
    let t: Vec<f64> = (0..ROWS).map(|i| (i % 2) as f64).collect();
    let z: Vec<f64> = (0..ROWS).map(|i| ((i / 3) % 2) as f64).collect();
    let mut y: Vec<f64> =
        (0..ROWS).map(|i| t[i] + 0.5 * z[i] + ((i * 7) % 11) as f64 / 11.0).collect();
    for &gap in gaps {
        y[gap] = f64::NAN;
    }
    [t, y, z]
}

fn table(columns: &[Vec<f64>; 3]) -> TabularData {
    TabularData::from_f64_columns([
        ("t", columns[0].as_slice()),
        ("y", columns[1].as_slice()),
        ("z", columns[2].as_slice()),
    ])
    .unwrap()
}

fn quantile(
    data: TabularData,
    levels: (f64, f64),
    estimator: AipwAte,
    missing: Option<MissingDeclaration>,
) -> Result<StudyResult, CausalError> {
    let query = AverageEffectQuery::with_levels(v(0), v(1), levels.0, levels.1)
        .with_outcome_functional(OutcomeFunctional::quantile(0.5));
    let mut builder = Study::tabular(data)
        .graph(dag3())
        .refute(RefuteSuite::None)
        .estimator(estimator)
        .query(query);
    if let Some(missing) = missing {
        builder = builder.missing(missing);
    }
    builder.build().and_then(|study| study.run(&ctx()))
}

fn plain() -> AipwAte {
    AipwAte::new().with_bootstrap_replicates(0)
}

fn effect_of(result: &StudyResult) -> (f64, f64) {
    let effect = result.estimate().as_effect().unwrap();
    (effect.ate, effect.uncertainty.analytic.se)
}

const GAPS: [usize; 4] = [5, 40, 77, 130];

#[test]
fn cells_03_quantile_complete_case_is_row_deletion_and_the_error_policy_names_the_gap() {
    let complete = quantile(table(&quantile_columns(&[])), (0.0, 1.0), plain(), None).unwrap();
    let (full, _) = effect_of(&complete);
    assert!((full - 1.0).abs() < 0.1, "{full}");

    let gapped = quantile_columns(&GAPS);
    let kept: Vec<usize> = (0..ROWS).filter(|row| !GAPS.contains(row)).collect();
    let deleted: [Vec<f64>; 3] =
        [0, 1, 2].map(|column| kept.iter().map(|&row| gapped[column][row]).collect());
    let reference = effect_of(&quantile(table(&deleted), (0.0, 1.0), plain(), None).unwrap());
    for missing in [None, Some(MissingDeclaration::new(MissingPolicy::CompleteCase))] {
        let declared = missing.is_some();
        let result = quantile(table(&gapped), (0.0, 1.0), plain(), missing).unwrap();
        assert_eq!(effect_of(&result), reference, "declared {declared}");
        let report = result.missing_data().as_ref().expect("a missing-data report");
        assert_eq!(report.policy().name(), "complete_case");
    }

    let error = quantile(
        table(&gapped),
        (0.0, 1.0),
        plain(),
        Some(MissingDeclaration::new(MissingPolicy::ErrorOnMissing)),
    )
    .err()
    .expect("a missing outcome was dropped under the error policy");
    assert_eq!(error.reason_code(), Some("missing_policy_required"), "{error}");
    assert!(error.to_string().contains("column `y` at row 5"), "{error}");

    let error = quantile(
        table(&gapped),
        (0.0, 1.0),
        plain(),
        Some(MissingDeclaration::new(MissingPolicy::InverseProbabilityOfObservation)),
    )
    .err()
    .expect("responder weighting was run on an observational quantile");
    assert_eq!(error.reason_code(), Some("missing_policy_not_applicable"), "{error}");
    assert!(error.to_string().contains("only randomized outcomes"), "{error}");
}

#[test]
fn cells_03_quantile_multiple_imputation_pools_to_the_hand_reference() {
    const IMPUTATIONS: u32 = 5;
    let spec = ImputationSpec::new().with_imputations(IMPUTATIONS).with_iterations(4).with_seed(9);
    let gaps: Vec<usize> = (0..ROWS).filter(|row| row % 7 == 3).collect();
    let columns = quantile_columns(&gaps);
    let data = table(&columns);
    let result = quantile(
        data.clone(),
        (0.0, 1.0),
        plain(),
        Some(MissingDeclaration::new(MissingPolicy::MultipleImputation(spec))),
    )
    .unwrap();
    let report = result.missing_data().as_ref().expect("a missing-data report");
    assert_eq!(report.policy().name(), "multiple_imputation");
    let report = report.imputation.as_ref().expect("an imputation report");
    assert_eq!(report.pooling.as_ref(), "rubin");
    assert_eq!(report.points.len(), IMPUTATIONS as usize);

    // The completed tables, from the public imputer on every column in table order.
    let imputer = Imputer::new(
        ["t", "y", "z"]
            .into_iter()
            .zip(&columns)
            .map(|(name, values)| ImputationColumn::new(name, values.clone()))
            .collect(),
        spec,
        ImputationUnits::Rows,
    )
    .unwrap();
    let (mut points, mut variances) = (Vec::new(), Vec::new());
    for index in 0..IMPUTATIONS {
        let completed = imputer.impute(index, &ctx()).unwrap();
        let values: [Vec<f64>; 3] = [0, 1, 2].map(|column| completed.columns()[column].to_vec());
        assert!(values[1].iter().all(|value| value.is_finite()));
        let (point, se) = effect_of(&quantile(table(&values), (0.0, 1.0), plain(), None).unwrap());
        points.push(point);
        variances.push(se * se);
    }
    let m = f64::from(IMPUTATIONS);
    let mean = points.iter().sum::<f64>() / m;
    let within = variances.iter().sum::<f64>() / m;
    let between = points.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / (m - 1.0);
    let total = within + (1.0 + 1.0 / m) * between;
    for (got, want) in report.points.iter().zip(&points) {
        assert!((got - want).abs() < 1e-12, "{got} vs {want}");
    }
    assert!(
        (report.within - within).abs() < 1e-12 * (1.0 + within),
        "{} vs {within}",
        report.within
    );
    assert!(
        (report.between - between).abs() < 1e-12 * (1.0 + between),
        "{} vs {between}",
        report.between
    );
    assert!((report.total - total).abs() < 1e-12 * (1.0 + total), "{} vs {total}", report.total);
    let (ate, se) = effect_of(&result);
    assert!((ate - mean).abs() < 1e-12, "{ate} vs {mean}");
    assert!((se - total.sqrt()).abs() < 1e-12, "{se} vs {}", total.sqrt());
    assert!(between > 0.0, "the imputations did not differ");
    assert!((ate - 1.0).abs() < 0.3, "{ate}");
}

#[test]
fn cells_03_quantile_refuses_a_treatment_level_no_row_carries() {
    let data = table(&quantile_columns(&[]));
    let ok = quantile(data.clone(), (0.0, 1.0), plain(), None).unwrap();
    assert!((effect_of(&ok).0 - 1.0).abs() < 0.1);
    for levels in [(0.0, 7.0), (5.0, 1.0), (0.0, 0.5)] {
        // Rows at the observed level 1 or 0 lie outside the declared pair: refused by name.
        let error = quantile(data.clone(), levels, plain(), None)
            .err()
            .unwrap_or_else(|| panic!("{levels:?}: a level no row carries was estimated"));
        assert_eq!(error.reason_code(), Some("treatment_level_undeclared"), "{levels:?}: {error}");
        assert!(
            error.to_string().contains("observed treatment levels: 0 (100 rows), 1 (100 rows)")
        );
        // Restricting to the declared pair leaves an arm with no rows, which the cross-fit refuses.
        let restricted = plain().with_other_levels(OtherLevels::ExcludeUndeclared);
        let error = quantile(data.clone(), levels, restricted, None)
            .err()
            .unwrap_or_else(|| panic!("{levels:?}: an empty arm was estimated"));
        assert!(error.to_string().contains("missing a treatment arm"), "{levels:?}: {error}");
    }
}

// ---------------------------------------------------------------------------------------------
// The sharp discontinuity.
// ---------------------------------------------------------------------------------------------

fn rd_columns(gap: Option<usize>) -> [Vec<f64>; 3] {
    let n = 600_usize;
    let r: Vec<f64> = (0..n).map(|i| -1.0 + 2.0 * (i as f64 + 0.5) / n as f64).collect();
    let t: Vec<f64> = r.iter().map(|&x| f64::from(u8::from(x >= 0.0))).collect();
    let mut y: Vec<f64> = (0..n)
        .map(|i| 3.0 * t[i] + 0.5 * r[i] + 0.05 * ((i * 37 % 17) as f64 / 17.0 - 0.5))
        .collect();
    if let Some(gap) = gap {
        y[gap] = f64::NAN;
    }
    [t, y, r]
}

fn rd(
    columns: &[Vec<f64>; 3],
    graph: impl IntoGraphInput,
    missing: Option<MissingDeclaration>,
) -> Result<StudyResult, CausalError> {
    let mut builder = Study::tabular(table(columns))
        .graph(graph)
        .query(AverageEffectQuery::binary_ate(v(0), v(1)))
        .identifier(IdentifierId::RdSharp)
        .estimator(EstimatorId::RdSharp)
        .rd_config(v(2), 0.0, 1.0)
        .bootstrap_replicates(0);
    if let Some(missing) = missing {
        builder = builder.missing(missing);
    }
    builder.build().and_then(|study| study.run(&ctx()))
}

#[test]
fn cells_03_sharp_rd_complete_case_is_row_deletion_and_other_policies_are_refused() {
    let full = effect_of(&rd(&rd_columns(None), dag3(), None).unwrap());
    assert!((full.0 - 3.0).abs() < 0.01, "{}", full.0);

    let gapped = rd_columns(Some(5));
    let kept: Vec<usize> = (0..600).filter(|&row| row != 5).collect();
    let deleted: [Vec<f64>; 3] =
        [0, 1, 2].map(|column| kept.iter().map(|&row| gapped[column][row]).collect());
    let reference = effect_of(&rd(&deleted, dag3(), None).unwrap());
    for missing in [None, Some(MissingDeclaration::new(MissingPolicy::CompleteCase))] {
        let result = rd(&gapped, dag3(), missing).unwrap();
        assert_eq!(effect_of(&result), reference);
        assert!((reference.0 - 3.0).abs() < 0.01, "{}", reference.0);
    }
    let error = rd(&gapped, dag3(), Some(MissingDeclaration::new(MissingPolicy::ErrorOnMissing)))
        .err()
        .expect("a missing outcome was dropped under the error policy");
    assert_eq!(error.reason_code(), Some("missing_policy_required"), "{error}");
    // Multiple imputation pools the discontinuity (its known-answer test is
    // build_quasi_experimental_synthetic_dependence.rs::discontinuity_imputation); responder
    // weighting has an estimator only for randomized outcomes.
    let pooled = rd(
        &gapped,
        dag3(),
        Some(MissingDeclaration::new(MissingPolicy::MultipleImputation(ImputationSpec::new()))),
    )
    .unwrap();
    assert!((effect_of(&pooled).0 - 3.0).abs() < 0.1, "{}", effect_of(&pooled).0);
    let error = rd(
        &gapped,
        dag3(),
        Some(MissingDeclaration::new(MissingPolicy::InverseProbabilityOfObservation)),
    )
    .err()
    .expect("a policy the discontinuity does not pool was ignored");
    assert_eq!(error.reason_code(), Some("missing_policy_not_applicable"), "{error}");
    assert!(error.to_string().contains("only randomized"), "{error}");
}

#[test]
fn cells_03_sharp_rd_runs_from_a_dag_or_confounding_free_admg_and_refuses_every_other_graph_class()
{
    let columns = rd_columns(None);
    let on_dag = effect_of(&rd(&columns, dag3(), None).unwrap());
    assert!((on_dag.0 - 3.0).abs() < 0.01, "{}", on_dag.0);

    let edges = [(2_u32, 0_u32), (0, 1), (2, 1)];
    let node = DenseNodeId::from_raw;
    let mut admg = Admg::with_variables(3);
    for (from, to) in edges {
        admg.insert_directed(node(from), node(to)).unwrap();
    }
    // An ADMG with no bidirected edge is a DAG: the same answer.
    assert_eq!(effect_of(&rd(&columns, admg.clone(), None).unwrap()), on_dag);
    // Latent confounding is not read by the discontinuity estimator: refused, not ignored.
    for (a, b) in [(0_u32, 1_u32), (2, 1), (2, 0)] {
        let mut latent = admg.clone();
        latent.insert_bidirected(node(a), node(b)).unwrap();
        let error = rd(&columns, latent, None)
            .err()
            .unwrap_or_else(|| panic!("{a}<->{b}: latent confounding was ignored"));
        assert!(error.to_string().contains("ADMG ATE requires estimator"), "{a}<->{b}: {error}");
        assert!(error.to_string().contains("rd.sharp"), "{a}<->{b}: {error}");
    }
    let mut cpdag = Cpdag::with_variables(3);
    let mut pag = Pag::with_variables(3);
    for (from, to) in edges {
        cpdag.insert_directed(node(from), node(to)).unwrap();
        pag.insert_directed(node(from), node(to)).unwrap();
    }
    let error = rd(&columns, cpdag, None).err().expect("a CPDAG was read as a DAG");
    assert!(error.to_string().contains("generalized.adjustment"), "{error}");
    assert!(error.to_string().contains("rd.sharp"), "{error}");
    let error = rd(&columns, pag, None).err().expect("a PAG was read as a DAG");
    assert!(error.to_string().contains("PAG"), "{error}");
    assert!(error.to_string().contains("rd.sharp"), "{error}");
}

#[test]
fn cells_03_discontinuity_refuses_a_temporal_graph_and_the_time_discontinuity_refuses_any_graph() {
    let columns = rd_columns(None);
    let mut temporal = TemporalDag::empty();
    let running = ensure_lagged(&mut temporal, v(2), Lag::CONTEMPORANEOUS).unwrap();
    let treatment = ensure_lagged(&mut temporal, v(0), Lag::CONTEMPORANEOUS).unwrap();
    let outcome = ensure_lagged(&mut temporal, v(1), Lag::CONTEMPORANEOUS).unwrap();
    temporal.insert_directed(running, treatment).unwrap();
    temporal.insert_directed(treatment, outcome).unwrap();
    temporal.insert_directed(running, outcome).unwrap();
    let error = rd(&columns, temporal.clone(), None).err().expect("a temporal graph was ignored");
    assert!(error.to_string().contains("unsupported data/graph/query combination"), "{error}");

    // The discontinuity in calendar time is a design-based query: it carries its own
    // identification contract and refuses every graph class instead of ignoring it.
    let time = |graph: Option<antecedent::AcceptedGraph>| {
        let n = 400_usize;
        let r: Vec<f64> = (0..n).map(|i| -1.0 + 2.0 * (i as f64 + 0.5) / n as f64).collect();
        let t: Vec<f64> = r.iter().map(|&x| f64::from(u8::from(x >= 0.0))).collect();
        let y: Vec<f64> = (0..n)
            .map(|i| 3.0 * t[i] + 0.5 * r[i] + 0.05 * ((i * 37 % 17) as f64 / 17.0 - 0.5))
            .collect();
        let data = TabularData::from_f64_columns([
            ("r", r.as_slice()),
            ("t", t.as_slice()),
            ("y", y.as_slice()),
        ])
        .unwrap();
        let query = LocalPolynomialRatioQuery {
            outcome: v(2),
            treatment: v(1),
            running: v(0),
            cutoff: 0.0,
            bandwidth: 1.0,
            kink: false,
            options: None,
        };
        let mut builder = Study::tabular(data).query(query);
        if let Some(graph) = graph {
            builder = builder.graph(graph);
        }
        builder.build().and_then(|study| study.run(&ctx()))
    };
    let ok = time(None).unwrap();
    assert!((ok.estimate().as_effect().unwrap().ate - 3.0).abs() < 0.01);
    let graphs: [antecedent::AcceptedGraph; 5] = [
        Dag::with_variables(3).into(),
        Admg::with_variables(3).into(),
        Cpdag::with_variables(3).into(),
        Pag::with_variables(3).into(),
        temporal.into(),
    ];
    for graph in graphs {
        let error = time(Some(graph)).err().expect("a graph was ignored by the design-based query");
        assert!(error.to_string().contains("must not be combined with graph="), "{error}");
    }
}
