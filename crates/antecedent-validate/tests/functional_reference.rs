//! Functional stability refuters (`refute_path`, `refute_distribution`,
//! `refute_window_contrast`) against exact finite-SCM evaluations of the identified
//! functionals (`oracle_data/functional_reference.py`).
//!
//! Each case is a small table of ternary / binary variables. The frozen file carries the data,
//! every perturbation the library RNG streams produce (Bernoulli row masks, contiguous-stretch
//! and rotation uniforms, the block-bootstrap standard errors of the two halves) and the
//! expected reports, computed by the script from the contingency table of the perturbed rows
//! with `fractions.Fraction`: the back-door effect and law, the path-specific mediation
//! contrast, the maximum conditional total variation, the p-value of the null under the
//! replicate spread and the split-half z test. The live test regenerates the perturbations from
//! the streams, asserts they equal the frozen ones, then runs the public refuters and compares
//! every report (value 1e-9, verdict exactly outside the 1e-7 band around the threshold). A
//! table the exact evaluation cannot read (an empty cell) must make the refuter refuse.

use std::sync::Arc;

use antecedent_core::{
    AssumptionSet, AverageEffectQuery, CausalQuery, CausalSchemaBuilder, ExecutionContext,
    Intervention, InterventionalDistributionQuery, MeasurementSpec, PathSpecificEffectQuery,
    RoleHint, SmallRoleSet, Value, ValueType, VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
};
use antecedent_estimate::{
    FunctionalDistribution, FunctionalDistributionWorkspace, FunctionalEffect,
    InterventionalDistributionEstimate, RowResampling, util::SamplingDeclaration,
};
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_identify::{
    IdIdentifier, IdentificationResult, IdentificationWorkspace, PathSpecificIdentifier,
};
use antecedent_validate::RefutationReport;
use antecedent_validate::functional::{refute_distribution, refute_path, refute_window_contrast};
use serde_json::{Value as Json, json};

// The refuters' historical streams (`streams.rs`); replicate r of slot s uses base + (s << 16) + r
// for the subsets and base + r for stretches and rotations.
const SUBSET_PATH: u64 = 0xF012_0000;
const SUBSET_DISTRIBUTION: u64 = 0xF013_0000;
const STRETCH_80: u64 = 0xF016_0000;
const STRETCH_50: u64 = 0xF017_0000;
const SHIFT: u64 = 0xF018_0000;
const SEED: u64 = 13;
const REPS: u64 = 20;
const SPLIT_HALF_REPLICATES: u32 = 40;

fn v(i: u32) -> VariableId {
    VariableId::from_raw(i)
}

fn reference() -> Json {
    serde_json::from_str(include_str!("oracle_data/functional_reference.json"))
        .expect("frozen reference")
}

fn floats(j: &Json) -> Vec<f64> {
    j.as_array().expect("array").iter().map(|x| x.as_f64().expect("f64")).collect()
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(SEED)
}

fn table(names: &[&str], columns: &[Vec<f64>]) -> TabularData {
    let n = columns[0].len();
    let mut builder = CausalSchemaBuilder::new();
    for name in names {
        builder
            .add_variable(
                *name,
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::Context),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
    }
    let cols = columns
        .iter()
        .enumerate()
        .map(|(i, values)| {
            OwnedColumn::Float64(
                Float64Column::new(
                    v(u32::try_from(i).unwrap()),
                    Arc::from(values.clone()),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            )
        })
        .collect();
    TabularData::new(
        OwnedColumnarStorage::try_new(builder.build().unwrap(), cols, None, None).unwrap(),
    )
}

fn case_table(case: &Json) -> TabularData {
    let columns: Vec<Vec<f64>> = case["columns"].as_array().unwrap().iter().map(floats).collect();
    let names: &[&str] =
        if case["kind"] == "path" { &["t", "m", "y", "c"] } else { &["t", "y", "c"] };
    table(names, &columns)
}

fn dag(edges: &[(u32, u32)], n: u32) -> Dag {
    let mut dag = Dag::with_variables(n);
    for &(a, b) in edges {
        dag.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    dag
}

// t = 0, y = 1, c = 2.
fn backdoor_dag() -> Dag {
    dag(&[(2, 0), (2, 1), (0, 1)], 3)
}

fn distribution_query(conditioned: bool) -> InterventionalDistributionQuery {
    let query =
        InterventionalDistributionQuery::new(v(1), [Intervention::set(v(0), Value::f64(1.0))]);
    if conditioned { query.with_conditioning([v(2)]) } else { query }
}

fn path_query() -> PathSpecificEffectQuery {
    PathSpecificEffectQuery::binary(v(0), v(2)).with_path_nodes([v(1)])
}

fn identify_effect() -> IdentificationResult {
    let id = IdIdentifier::new();
    let prep = id.prepare_dag(&backdoor_dag()).unwrap();
    id.identify_ate(
        &prep,
        &AverageEffectQuery::binary_ate(v(0), v(1)),
        &mut IdentificationWorkspace::default(),
    )
    .unwrap()
}

fn identify_distribution(query: &InterventionalDistributionQuery) -> IdentificationResult {
    let id = antecedent_identify::IdcIdentifier::new();
    let prep = id.prepare_dag(&backdoor_dag()).unwrap();
    id.identify(
        &prep,
        &CausalQuery::Distribution(query.clone()),
        &mut IdentificationWorkspace::default(),
    )
    .unwrap()
}

fn identify_path() -> IdentificationResult {
    // t = 0, m = 1, y = 2, c = 3: c -> t, m, y; t -> m, y; m -> y; the path runs through m.
    let id = PathSpecificIdentifier::new();
    let prep = id.prepare_dag(&dag(&[(3, 0), (3, 1), (3, 2), (0, 1), (0, 2), (1, 2)], 4)).unwrap();
    id.identify(
        &prep,
        &CausalQuery::PathSpecific(path_query()),
        &mut IdentificationWorkspace::default(),
    )
    .unwrap()
}

fn mask(stream: u64, n: usize, fraction: f64) -> Vec<bool> {
    let mut rng = ctx().rng.stream(stream);
    (0..n).map(|_| rng.next_f64() < fraction).collect()
}

fn first_draw(stream: u64) -> f64 {
    ctx().rng.stream(stream).next_f64()
}

/// Block-bootstrap standard errors of the two halves, computed on halves this test slices itself.
fn half_se(case: &Json) -> Vec<Json> {
    let columns: Vec<Vec<f64>> = case["columns"].as_array().unwrap().iter().map(floats).collect();
    let n = columns[0].len();
    let span = u32::try_from(case["span"].as_u64().unwrap()).unwrap();
    let identified = identify_effect();
    let estimator = FunctionalEffect::new()
        .with_bootstrap_replicates(SPLIT_HALF_REPLICATES)
        .with_row_resampling(RowResampling::WindowBlocks { span }, SamplingDeclaration::Undeclared);
    [(0, n / 2), (n / 2, n)]
        .into_iter()
        .map(|(a, b)| {
            let part: Vec<Vec<f64>> = columns.iter().map(|c| c[a..b].to_vec()).collect();
            let part = table(&["t", "y", "c"], &part);
            let prepared = estimator
                .prepare(
                    &part,
                    &identified.estimands[0],
                    &identified.arena,
                    identified.required_assumptions.clone(),
                    &[v(0), v(1), v(2)],
                )
                .unwrap();
            // `null` when the half gives no standard error (or no estimate at all).
            estimator
                .estimate(&prepared, &mut FunctionalDistributionWorkspace::default(), &ctx())
                .ok()
                .and_then(|estimate| estimate.se_bootstrap())
                .map_or(Json::Null, |se| json!(se))
        })
        .collect()
}

/// Circular block length the window bootstrap uses on each half (the library's own rule).
fn half_blocks(case: &Json) -> Vec<usize> {
    let n = case["columns"][0].as_array().unwrap().len();
    let span = u32::try_from(case["span"].as_u64().unwrap()).unwrap();
    [n / 2, n - n / 2]
        .into_iter()
        .map(|rows| RowResampling::WindowBlocks { span }.block_length(rows).unwrap())
        .collect()
}

fn perturbations(case: &Json) -> Json {
    let n = case["columns"][0].as_array().unwrap().len();
    let masks = |base: u64| {
        json!({
            "s0": (0..REPS).map(|r| mask(base + r, n, 0.8)).collect::<Vec<_>>(),
            "s1": (0..REPS).map(|r| mask(base + (1 << 16) + r, n, 0.5)).collect::<Vec<_>>(),
        })
    };
    match case["kind"].as_str().unwrap() {
        "path" => json!({ "masks": masks(SUBSET_PATH) }),
        "distribution" => json!({ "masks": masks(SUBSET_DISTRIBUTION) }),
        _ => json!({
            "stretch_u": {
                "80": (0..REPS).map(|r| first_draw(STRETCH_80 + r)).collect::<Vec<_>>(),
                "50": (0..REPS).map(|r| first_draw(STRETCH_50 + r)).collect::<Vec<_>>(),
            },
            "shift_u": (0..REPS).map(|r| first_draw(SHIFT + r)).collect::<Vec<_>>(),
            "half_se": half_se(case),
            "half_block": half_blocks(case),
        }),
    }
}

/// Writes the perturbations the reference script needs (the first stage of regeneration).
#[test]
#[ignore = "regeneration helper"]
fn functional_emit() {
    let data: Json = serde_json::from_str(
        &std::fs::read_to_string(std::env::var("FUNCTIONAL_REFERENCE_DATA").unwrap()).unwrap(),
    )
    .unwrap();
    let mut out = serde_json::Map::new();
    for case in data.as_array().unwrap() {
        out.insert(case["name"].as_str().unwrap().to_string(), perturbations(case));
    }
    std::fs::write(
        std::env::var("FUNCTIONAL_REFERENCE_OUT").unwrap(),
        serde_json::to_string(&Json::Object(out)).unwrap(),
    )
    .unwrap();
}

fn same(a: &Json, b: &Json, what: &str) {
    match (a, b) {
        (Json::Array(x), Json::Array(y)) => {
            assert_eq!(x.len(), y.len(), "{what}");
            for (i, (p, q)) in x.iter().zip(y).enumerate() {
                same(p, q, &format!("{what}[{i}]"));
            }
        }
        (Json::Object(x), Json::Object(y)) => {
            assert_eq!(x.len(), y.len(), "{what}");
            for (k, p) in x {
                same(p, &y[k], &format!("{what}.{k}"));
            }
        }
        (Json::Number(x), Json::Number(y)) => {
            let (x, y) = (x.as_f64().unwrap(), y.as_f64().unwrap());
            assert!((x - y).abs() <= 1e-12 * x.abs().max(1.0), "{what}: {x} vs {y}");
        }
        _ => assert_eq!(a, b, "{what}"),
    }
}

#[test]
fn functional_frozen_perturbations_are_the_library_streams() {
    for entry in reference().as_array().unwrap() {
        let name = entry["case"]["name"].as_str().unwrap();
        same(&entry["perturbations"], &perturbations(&entry["case"]), name);
    }
}

fn close(got: f64, want: f64, what: &str) {
    assert!((got - want).abs() <= 1e-9 * want.abs().max(1.0), "{what}: got {got} want {want}");
}

/// Compare the reports of one call with the expected ones; `strict` counts the verdicts checked.
fn compare(reports: &[RefutationReport], want: &Json, label: &str, checked: &mut usize) {
    let want = want.as_array().unwrap();
    assert_eq!(reports.len(), want.len(), "{label}: report count");
    for (got, want) in reports.iter().zip(want) {
        let id = want["id"].as_str().unwrap();
        let label = format!("{label} {id}");
        assert_eq!(&*got.refuter, id, "{label}: id");
        close(got.original_ate, want["original"].as_f64().unwrap(), &format!("{label} original"));
        close(got.refuted_ate, want["refuted"].as_f64().unwrap(), &format!("{label} refuted"));
        close(got.comparison, want["comparison"].as_f64().unwrap(), &format!("{label} comparison"));
        assert_eq!(got.informative, want["informative"].as_bool().unwrap(), "{label}: informative");
        assert_eq!(u64::from(got.replicates), want["replicates"].as_u64().unwrap(), "{label}");
        let near_threshold = if id.starts_with("distribution.") && id.ends_with("_tv") {
            (got.comparison - 0.1).abs() < 1e-7
        } else {
            (got.comparison - 0.05).abs() < 1e-7 && !id.ends_with("_tv")
        };
        if !near_threshold {
            assert_eq!(got.passed, want["passed"].as_bool().unwrap(), "{label}: verdict");
            *checked += 1;
        }
        assert_eq!(got.failure_condition.is_none(), got.passed, "{label}: failure text");
    }
}

fn expect(
    got: Result<Vec<RefutationReport>, antecedent_validate::ValidationError>,
    run: &Json,
    label: &str,
    checked: &mut usize,
    errors: &mut usize,
) {
    if run["reports"] == "error" {
        assert!(got.is_err(), "{label}: an unreadable cell must refuse, got {got:?}");
        *errors += 1;
    } else {
        compare(&got.unwrap_or_else(|e| panic!("{label}: {e:?}")), &run["reports"], label, checked);
    }
}

#[test]
fn functional_window_contrast_refuters_equal_the_exact_reference() {
    let identified = identify_effect();
    let (mut checked, mut errors, mut verdicts) = (0, 0, Vec::new());
    for entry in reference().as_array().unwrap() {
        if entry["case"]["kind"] != "window" {
            continue;
        }
        let name = entry["case"]["name"].as_str().unwrap();
        let data = case_table(&entry["case"]);
        let span = u32::try_from(entry["case"]["span"].as_u64().unwrap()).unwrap();
        for run in entry["runs"].as_array().unwrap() {
            let full = run["full"].as_bool().unwrap();
            let got = refute_window_contrast(
                &data,
                &identified,
                &identified.estimands[0],
                &[v(0), v(1), v(2)],
                &[v(0)],
                span,
                run["original"].as_f64().unwrap(),
                full,
                &ctx(),
            );
            if let Ok(reports) = &got {
                verdicts.extend(reports.iter().map(|r| (r.refuter.to_string(), r.passed)));
            }
            expect(got, run, &format!("{name} full={full}"), &mut checked, &mut errors);
        }
    }
    assert!(errors >= 1, "no case exercised the refusal of an unreadable stretch or rotation");
    for id in ["window.stretch_80", "window.placebo_treatment", "window.split_half"] {
        assert!(verdicts.iter().any(|(r, p)| r == id && *p) || id == "window.placebo_treatment");
    }
    assert!(verdicts.iter().any(|(_, p)| !p), "no refuter failed: {verdicts:?}");
    assert!(checked >= 20, "{checked}");
}

#[test]
fn functional_distribution_table_refuters_equal_the_exact_reference() {
    let (mut checked, mut errors) = (0, 0);
    let mut failed = false;
    for entry in reference().as_array().unwrap() {
        if entry["case"]["kind"] != "distribution" {
            continue;
        }
        let name = entry["case"]["name"].as_str().unwrap();
        let data = case_table(&entry["case"]);
        let query = distribution_query(entry["case"]["conditioned"].as_bool().unwrap());
        let identified = identify_distribution(&query);
        let estimator = FunctionalDistribution::new();
        let prepared = estimator
            .prepare(
                &data,
                &query,
                &identified.estimands[0],
                &identified.arena,
                identified.required_assumptions.clone(),
            )
            .unwrap();
        let original = estimator
            .estimate(&prepared, &[], &mut FunctionalDistributionWorkspace::default(), &ctx())
            .unwrap();
        // The unperturbed table is the exact law of the finite SCM's contingency table.
        let law = entry["law"].as_array().unwrap();
        assert_eq!(original.atoms.len(), law.len(), "{name}: atom count");
        for want in law {
            let group: Vec<f64> = floats(&want[0]);
            let y = want[1].as_f64().unwrap();
            let atom = original
                .atoms
                .iter()
                .find(|a| {
                    a.outcomes[0].1.as_f64() == Some(y)
                        && a.conditioning.iter().map(|c| c.1.as_f64().unwrap()).collect::<Vec<_>>()
                            == group
                })
                .unwrap_or_else(|| panic!("{name}: no atom for {group:?} y={y}"));
            close(atom.probability, want[2].as_f64().unwrap(), &format!("{name} law"));
        }
        for run in entry["runs"].as_array().unwrap() {
            let full = run["full"].as_bool().unwrap();
            let got = refute_distribution(
                &data,
                &query,
                &identified,
                &identified.estimands[0],
                &original,
                full,
                &ctx(),
            );
            if let Ok(reports) = &got {
                failed |= reports.iter().any(|r| !r.passed);
            }
            expect(got, run, &format!("{name} full={full}"), &mut checked, &mut errors);
        }
        // A normalization defect of the supplied table is reported, at the 1e-8 tolerance.
        for tamper in entry["tampered"].as_array().unwrap() {
            let scale = tamper["scale"].as_f64().unwrap();
            let atoms: Vec<_> = original
                .atoms
                .iter()
                .map(|a| antecedent_estimate::DistributionAtom {
                    probability: a.probability * scale,
                    ..a.clone()
                })
                .collect();
            let mut bad: InterventionalDistributionEstimate = original.clone();
            bad.atoms = atoms.into();
            let reports = refute_distribution(
                &data,
                &query,
                &identified,
                &identified.estimands[0],
                &bad,
                false,
                &ctx(),
            );
            let report = &reports.unwrap()[0];
            assert_eq!(&*report.refuter, "distribution.normalization");
            assert_eq!(
                report.passed,
                tamper["normalized"].as_bool().unwrap(),
                "{name} scale {scale}"
            );
            assert_eq!(report.failure_condition.is_none(), report.passed);
        }
        let mut empty = original.clone();
        empty.atoms = Arc::from(Vec::new());
        let reports = refute_distribution(
            &data,
            &query,
            &identified,
            &identified.estimands[0],
            &empty,
            false,
            &ctx(),
        );
        // No table to normalize is a failure, not a vacuous pass.
        let reports = reports.unwrap();
        assert!(!reports[0].passed, "{name}: empty table");
    }
    assert!(failed, "no case failed the 0.1 distance threshold");
    assert!(checked >= 10, "{checked}");
    let _ = errors;
}

#[test]
fn functional_path_subset_refuters_equal_the_exact_reference() {
    let identified = identify_path();
    let query = path_query();
    let (mut checked, mut errors) = (0, 0);
    let mut verdicts = Vec::new();
    for entry in reference().as_array().unwrap() {
        if entry["case"]["kind"] != "path" {
            continue;
        }
        let name = entry["case"]["name"].as_str().unwrap();
        let data = case_table(&entry["case"]);
        for run in entry["runs"].as_array().unwrap() {
            let full = run["full"].as_bool().unwrap();
            let original = run["original"].as_f64().unwrap();
            // The product's own full-table refit is the exact contrast.
            let estimator = FunctionalEffect::new();
            let prepared = estimator
                .prepare(
                    &data,
                    &identified.estimands[0],
                    &identified.arena,
                    AssumptionSet::new(),
                    &[v(0), v(2), v(1)],
                )
                .unwrap();
            let fit = estimator
                .estimate(&prepared, &mut FunctionalDistributionWorkspace::default(), &ctx())
                .unwrap();
            close(
                fit.ate,
                original - run["delta"].as_f64().unwrap(),
                &format!("{name} full-table contrast"),
            );
            let got = refute_path(
                &data,
                &query,
                &identified,
                &identified.estimands[0],
                original,
                full,
                &ctx(),
            );
            if let Ok(reports) = &got {
                verdicts.extend(reports.iter().map(|r| r.passed));
            }
            expect(got, run, &format!("{name} full={full}"), &mut checked, &mut errors);
        }
    }
    assert!(errors >= 1, "no case exercised the refusal of an unreadable subset");
    assert!(verdicts.iter().any(|&p| p) && verdicts.iter().any(|&p| !p), "{verdicts:?}");
    assert!(checked >= 4, "{checked}");
}
