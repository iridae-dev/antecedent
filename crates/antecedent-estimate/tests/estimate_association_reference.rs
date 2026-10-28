//! The descriptive association estimator against definition-level references.
//!
//! `oracle_data/estimate_association_reference.py` (frozen in the JSON beside it) recomputes, from
//! each case's data alone: the linear fit and the logistic and Poisson maximum likelihood fits
//! with their HC1 (rows) and Bartlett Newey-West (series) slope covariances, weighted group
//! means with their linearised errors, the Pearson error on the Kish count, the distance
//! correlation, the leave-fold-out predictions with their predictive and baseline laws, the
//! conjugate Normal-Inverse-Gamma marginals and, from frozen draws, the row and circular-block
//! bootstrap and the Bayesian bootstrap summaries. The draws are the library's own streams: this
//! test regenerates them and asserts equality with the frozen values, so a changed stream is a
//! visible failure rather than a silent change of the references. The selection rule is checked
//! against a decision table written from its documentation.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    AssociationFunctional, AssociationMeasure, AssociationOutcome, AssociationQuery,
    CategoryDomainId, CausalSchemaBuilder, ExecutionContext, MeasurementSpec, RoleHint,
    SmallRoleSet, StreamDomain, ValueType, VariableId, derive_stream,
};
use antecedent_data::{
    CategoricalColumn, CategoryCode, CategoryDomain, CategoryLevel, Float64Column, OwnedColumn,
    OwnedColumnarStorage, UnknownCategoryPolicy, ValidityBitmap, fill_circular_block_indexes,
};
use antecedent_estimate::{
    AssociationEstimate, AssociationEstimator, BayesianSettings, Modality, dependence_block_length,
    select_analysis,
};
use antecedent_kernels::{bayesian_bootstrap_weights, shuffled_fold_assignment, unbiased_index};
use serde_json::{Value, json};

const SEED: u64 = 17;
// The stream indices the estimator draws from (pinned here; a change shows as a stream mismatch).
const BOOTSTRAP_STREAM: u64 = 0xA550_C001;
const POSTERIOR_STREAM: u64 = 0xA550_C002;
const FOLD_STREAM: u64 = 0xA550_C003;

fn reference() -> Value {
    serde_json::from_str(include_str!("oracle_data/estimate_association_reference.json")).unwrap()
}

// ---------------------------------------------------------------------------------- data
enum Col {
    Num(Vec<f64>),
    Cat { codes: Vec<u32>, labels: Vec<String>, reference: Option<u32> },
}

fn parse_col(v: &Value) -> Col {
    if let Some(array) = v.as_array() {
        return Col::Num(array.iter().map(|x| x.as_f64().unwrap_or(f64::NAN)).collect());
    }
    Col::Cat {
        codes: v["codes"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect(),
        labels: v["labels"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_str().unwrap().to_owned())
            .collect(),
        reference: v["reference"].as_u64().map(|x| x as u32),
    }
}

/// Storage of the columns in order; the second field of each entry is the schema type.
fn storage(columns: &[(String, ValueType, Col)]) -> OwnedColumnarStorage {
    let mut builder = CausalSchemaBuilder::new();
    let mut built = Vec::new();
    let n = match &columns[0].2 {
        Col::Num(v) => v.len(),
        Col::Cat { codes, .. } => codes.len(),
    };
    for (i, (name, value_type, col)) in columns.iter().enumerate() {
        let id = VariableId::from_raw(u32::try_from(i).unwrap());
        let domain = matches!(col, Col::Cat { .. })
            .then(|| CategoryDomainId::from_raw(u32::try_from(i).unwrap() + 1));
        builder
            .add_variable(
                name.as_str(),
                value_type.clone(),
                SmallRoleSet::from_hint(RoleHint::Context),
                None,
                domain,
                MeasurementSpec::default(),
            )
            .unwrap();
        built.push(match col {
            Col::Num(values) => {
                let cleaned: Vec<f64> = values.clone();
                OwnedColumn::Float64(
                    Float64Column::new(id, Arc::from(cleaned), ValidityBitmap::all_valid(n))
                        .unwrap(),
                )
            }
            Col::Cat { codes, labels, reference } => {
                let domain = CategoryDomain::try_new(
                    CategoryDomainId::from_raw(u32::try_from(i).unwrap() + 1),
                    labels
                        .iter()
                        .map(|l| CategoryLevel { label: Arc::from(l.as_str()) })
                        .collect::<Vec<_>>(),
                    false,
                    reference.map(CategoryCode::from_raw),
                    UnknownCategoryPolicy::Fail,
                )
                .unwrap();
                OwnedColumn::Categorical(
                    CategoricalColumn::try_new(
                        id,
                        codes.iter().map(|c| CategoryCode::from_raw(*c)).collect::<Vec<_>>(),
                        ValidityBitmap::all_valid(n),
                        Arc::new(domain),
                    )
                    .unwrap(),
                )
            }
        });
    }
    OwnedColumnarStorage::try_new(builder.build().unwrap(), built, None, None).unwrap()
}

struct Built {
    storage: OwnedColumnarStorage,
    query: AssociationQuery,
    rows: usize,
    y: Vec<f64>,
}

fn build(case: &Value) -> Built {
    let object = case["columns"].as_object().unwrap();
    let names: Vec<&String> = object.keys().collect();
    let id_of = |name: &str| {
        VariableId::from_raw(
            u32::try_from(names.iter().position(|n| n.as_str() == name).unwrap()).unwrap(),
        )
    };
    let columns: Vec<(String, ValueType, Col)> = names
        .iter()
        .map(|name| {
            let col = parse_col(&object[name.as_str()]);
            let value_type = if matches!(col, Col::Cat { .. }) {
                ValueType::Categorical
            } else {
                ValueType::Continuous
            };
            ((*name).clone(), value_type, col)
        })
        .collect();
    let list = |key: &str| -> Vec<VariableId> {
        case[key].as_array().unwrap().iter().map(|v| id_of(v.as_str().unwrap())).collect()
    };
    let outcome = id_of(case["outcome"].as_str().unwrap());
    let functional = if case["folds"].is_u64() {
        AssociationFunctional::Predictive { folds: case["folds"].as_u64().unwrap() as u32 }
    } else {
        match case["kind"].as_str().unwrap() {
            "model" => AssociationFunctional::Coefficients,
            "strata" => AssociationFunctional::Conditional,
            "pearson" => AssociationFunctional::Measure(AssociationMeasure::Pearson),
            _ => AssociationFunctional::Measure(AssociationMeasure::Distance),
        }
    };
    let scale = match case["family"].as_str().unwrap() {
        "continuous" => AssociationOutcome::Continuous,
        "binary" => AssociationOutcome::Binary,
        _ => AssociationOutcome::Count,
    };
    let mut query = AssociationQuery::new(outcome, list("predictors"))
        .with_conditioning(list("conditioning"))
        .with_functional(functional)
        .with_outcome_scale(scale);
    if let Some(weight) = case["weight"].as_str() {
        query = query.with_weight(id_of(weight));
    }
    let y = match &columns[outcome.raw() as usize].2 {
        Col::Num(v) => v.clone(),
        Col::Cat { .. } => unreachable!(),
    };
    let rows_total = y.len();
    Built { storage: storage(&columns), query, rows: rows_total, y }
}

fn modality(case: &Value) -> Modality {
    if case["modality"] == "series" { Modality::Series } else { Modality::CrossSection }
}

fn estimator(case: &Value) -> AssociationEstimator {
    let mut estimator = AssociationEstimator::new(modality(case));
    if let Some(replicates) = case["boot"].as_u64() {
        estimator = estimator.with_bootstrap_replicates(u32::try_from(replicates).unwrap());
    }
    if case["bayes"].is_object() {
        estimator = estimator.with_bayesian(BayesianSettings {
            draws: case["bayes"]["draws"].as_u64().unwrap() as usize,
            prior_scale: case["bayes"]["prior_scale"].as_f64().unwrap(),
        });
    }
    estimator
}

// ---------------------------------------------------------------------------------- streams
fn is_conjugate(case: &Value) -> bool {
    case["family"] == "continuous"
        && case["modality"] != "series"
        && case["weight"].is_null()
        && matches!(case["kind"].as_str().unwrap(), "model" | "strata")
}

/// Block length the estimator forms from the centred outcome (span 1), through the public rule.
fn block_length(y: &[f64]) -> usize {
    let mean = y.iter().sum::<f64>() / y.len() as f64;
    let centred: Vec<f64> = y.iter().map(|v| v - mean).collect();
    dependence_block_length(1, y.len(), &[centred.as_slice()]).clamp(1, y.len())
}

/// The library's own draws for a resampling case, in the form frozen in the reference.
fn regenerate(case: &Value, rows: usize, y: &[f64]) -> Value {
    let ctx = ExecutionContext::for_tests(SEED);
    let series = case["modality"] == "series";
    let length = series.then(|| block_length(&y[..rows]));
    if let Some(replicates) = case["boot"].as_u64() {
        let mut indexes = Vec::new();
        for r in 0..replicates {
            let mut index = vec![0_usize; rows];
            if let Some(length) = length {
                let mut rng = ctx
                    .rng
                    .stream_for(StreamDomain::TemporalBlock, derive_stream(BOOTSTRAP_STREAM, r));
                fill_circular_block_indexes(rows, length, &mut rng, &mut index).unwrap();
            } else {
                let mut rng = ctx.rng.stream_for(StreamDomain::Estimate, BOOTSTRAP_STREAM ^ r);
                for slot in &mut index {
                    *slot = unbiased_index(&mut rng, rows);
                }
            }
            indexes.push(index);
        }
        return json!({"indexes": indexes, "block_length": length});
    }
    if case["bayes"].is_object() && !is_conjugate(case) {
        let draws = case["bayes"]["draws"].as_u64().unwrap();
        let mut weights = Vec::new();
        for d in 0..draws {
            let mut rng = ctx.rng.stream_for(StreamDomain::Estimate, POSTERIOR_STREAM ^ d);
            let count = length.map_or(rows, |l| rows.div_ceil(l));
            weights.push(bayesian_bootstrap_weights(&mut rng, count, false));
        }
        return json!({"weights": weights, "block_length": length});
    }
    if let Some(folds) = case["folds"].as_u64() {
        let folds = folds as usize;
        let assignment: Vec<usize> = if series {
            (0..rows).map(|r| r * folds / rows).collect()
        } else {
            let mut rng = ctx.rng.stream_for(StreamDomain::Estimate, FOLD_STREAM);
            shuffled_fold_assignment(&mut rng, rows, folds, None).unwrap()
        };
        return json!({ "assignment": assignment });
    }
    json!({})
}

/// Writes the library streams of every resampling case for the reference script to consume.
/// Set `ASSOCIATION_STREAMS_OUT` to the output path and run with `--ignored`.
#[test]
#[ignore = "regenerates the frozen streams; run by hand when the cases change"]
fn association_dump_library_streams() {
    let mut out = serde_json::Map::new();
    for case in reference()["resampling"].as_array().unwrap() {
        let built = build(case);
        // Complete rows only: the cases with missing cells carry no resampling plan.
        out.insert(
            case["name"].as_str().unwrap().to_owned(),
            regenerate(case, built.rows, &built.y),
        );
    }
    let path = std::env::var("ASSOCIATION_STREAMS_OUT").expect("ASSOCIATION_STREAMS_OUT");
    std::fs::write(path, serde_json::to_string(&Value::Object(out)).unwrap()).unwrap();
}

// ---------------------------------------------------------------------------------- compare
fn close(got: f64, want: f64, rel: f64, what: &str) {
    let scale = want.abs().max(1e-12);
    assert!(
        (got - want).abs() <= rel * scale,
        "{what}: got {got:e} want {want:e} (rel {:e})",
        (got - want).abs() / scale
    );
}

fn tolerance(case: &Value) -> (f64, f64) {
    let name = case["name"].as_str().unwrap();
    if name.contains("offsets") || name.contains("collinear") {
        return (1e-6, 1e-5);
    }
    if case["family"] != "continuous" {
        return (1e-6, 1e-5);
    }
    (1e-9, 1e-9)
}

fn check_plain(case: &Value, estimate: &AssociationEstimate) {
    let name = case["name"].as_str().unwrap();
    let (est_tol, se_tol) = tolerance(case);
    let want = &case["expect"];
    let points = want["estimate"].as_array().unwrap();
    let errors = want["se"].as_array().unwrap();
    assert_eq!(estimate.terms.len(), points.len(), "{name}: term count");
    for (k, term) in estimate.terms.iter().enumerate() {
        close(term.estimate, points[k].as_f64().unwrap(), est_tol, &format!("{name} estimate {k}"));
        match errors[k].as_f64() {
            Some(se) => {
                let got = term.standard_error.unwrap_or_else(|| panic!("{name}: se {k} missing"));
                close(got, se, se_tol, &format!("{name} se {k}"));
            }
            None => assert!(term.standard_error.is_none(), "{name}: se {k} should be absent"),
        }
    }
}

#[test]
fn estimates_and_analytic_errors_match_the_definition_references() {
    let reference = reference();
    for case in reference["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let built = build(case);
        let estimator = estimator(case);
        let prepared = estimator
            .prepare(&built.storage, &built.query)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let dropped = case["expect"]["rows_dropped"].as_u64().unwrap() as usize;
        assert_eq!(prepared.rows_dropped(), dropped, "{name}: rows dropped");
        assert_eq!(prepared.rows() + dropped, built.rows, "{name}: rows");
        let estimate = estimator
            .estimate(&prepared, &ExecutionContext::for_tests(SEED))
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        check_plain(case, &estimate);
        // The construction named for the error follows the modality and the analysis.
        let series = case["modality"] == "series";
        let method = match case["kind"].as_str().unwrap() {
            "model" if series => Some("sandwich_newey_west"),
            "model" => Some("sandwich_hc1"),
            "strata" if series => Some("newey_west_influence"),
            "strata" => Some("linearised_influence"),
            "pearson" if !series => Some("normal_theory_delta"),
            _ => None,
        };
        for term in &estimate.terms {
            assert_eq!(term.standard_error_method, term.standard_error.and(method), "{name}");
        }
    }
}

#[test]
fn resampling_summaries_and_posteriors_match_the_references_on_the_library_streams() {
    let reference = reference();
    for case in reference["resampling"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let built = build(case);
        if case["streams"].is_object() {
            assert_eq!(
                &regenerate(case, built.rows, &built.y),
                &case["streams"],
                "{name}: library streams drifted"
            );
        }
        let estimator = estimator(case);
        let prepared = estimator
            .prepare(&built.storage, &built.query)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let want = &case["expect"];
        let (est_tol, se_tol) = tolerance(case);
        let result = estimator.estimate(&prepared, &ExecutionContext::for_tests(SEED));
        if case["bayes"].is_object()
            && want["posterior"].as_array().is_some_and(|p| p.iter().any(Value::is_null))
        {
            let error = result.expect_err(name);
            assert_eq!(error.reason_code(), Some("posterior_draws_below_supported"), "{name}");
            continue;
        }
        let estimate = result.unwrap_or_else(|e| panic!("{name}: {e}"));
        let points = want["estimate"].as_array().unwrap();
        let analytic = want["se"].as_array().unwrap();
        for (k, term) in estimate.terms.iter().enumerate() {
            close(
                term.estimate,
                points[k].as_f64().unwrap(),
                est_tol,
                &format!("{name} estimate {k}"),
            );
            let boot = want["bootstrap"]["terms"].get(k);
            let boot_se = boot.and_then(|b| b["se"].as_f64());
            let expected_se = boot_se.or(analytic[k].as_f64());
            match expected_se {
                Some(se) => close(
                    term.standard_error.unwrap_or_else(|| panic!("{name}: se {k} missing")),
                    se,
                    se_tol,
                    &format!("{name} se {k}"),
                ),
                None => assert!(term.standard_error.is_none(), "{name}: se {k}"),
            }
            if boot.is_some() {
                let series = case["modality"] == "series";
                let method = if boot_se.is_some() {
                    Some(if series { "bootstrap_circular_block" } else { "bootstrap_row" })
                } else {
                    term.standard_error.and(Some(term.standard_error_method.unwrap()))
                };
                if boot_se.is_some() {
                    assert_eq!(term.standard_error_method, method, "{name}");
                } else {
                    assert!(
                        !term.standard_error_method.is_some_and(|m| m.starts_with("bootstrap")),
                        "{name}: a term with too few finite replicates keeps its analytic error"
                    );
                }
                match boot.unwrap()["spread"].as_array() {
                    Some(s) => {
                        let (lo, hi) = term.spread.unwrap_or_else(|| panic!("{name}: spread {k}"));
                        close(lo, s[0].as_f64().unwrap(), se_tol, &format!("{name} spread lo {k}"));
                        close(hi, s[1].as_f64().unwrap(), se_tol, &format!("{name} spread hi {k}"));
                    }
                    None => assert!(term.spread.is_none(), "{name}: spread {k}"),
                }
            }
            if let Some(posterior) = want["posterior"].get(k) {
                let got = term.posterior.unwrap_or_else(|| panic!("{name}: posterior {k}"));
                for (field, value) in
                    [("mean", got.mean), ("sd", got.sd), ("lower", got.lower), ("upper", got.upper)]
                {
                    close(
                        value,
                        posterior[field].as_f64().unwrap(),
                        se_tol,
                        &format!("{name} posterior {field} {k}"),
                    );
                }
            }
        }
        if case["boot"].is_u64() {
            let record = estimate.resampling.unwrap();
            assert_eq!(
                u64::from(record.replicates_ok),
                want["bootstrap"]["ok"].as_u64().unwrap(),
                "{name}"
            );
            assert_eq!(
                record.block_length.map(|l| l as u64),
                case["streams"]["block_length"].as_u64(),
                "{name}"
            );
        }
        if let Some(method) = want["posterior_method"].as_str() {
            let record = estimate.posterior.unwrap();
            assert_eq!(record.method, method, "{name}");
            let draws = if method == "bayesian_bootstrap" {
                case["bayes"]["draws"].as_u64().unwrap() as usize
            } else {
                0
            };
            assert_eq!(record.draws, draws, "{name}");
        }
        if let Some(cases) = want["prediction"].as_array() {
            let fit = estimate.prediction.unwrap();
            let series = case["modality"] == "series";
            assert_eq!(
                fit.fold_design,
                if series { "consecutive_blocks" } else { "random_assignment" },
                "{name}"
            );
            assert_eq!(fit.folds as u64, case["folds"].as_u64().unwrap(), "{name}");
            assert_eq!(fit.cases.len(), cases.len(), "{name}");
            for (i, (got, want)) in fit.cases.iter().zip(cases).enumerate() {
                for (field, value) in [
                    ("outcome", got.outcome),
                    ("weight", got.weight),
                    ("predicted", got.predicted),
                    ("predicted_sd", got.predicted_sd),
                    ("baseline", got.baseline),
                    ("baseline_sd", got.baseline_sd),
                ] {
                    close(
                        value,
                        want[field].as_f64().unwrap(),
                        est_tol.max(1e-8),
                        &format!("{name} case {i} {field}"),
                    );
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------- selection
#[test]
fn the_selection_rule_matches_its_documented_decision_table() {
    let reference = reference();
    for spec in reference["selection"].as_array().unwrap() {
        let name = spec["name"].as_str().unwrap();
        let mut columns: Vec<(String, ValueType, Col)> = Vec::new();
        let typed = |label: &str, c: &Value| -> (String, ValueType, Col) {
            let kind = c["type"].as_str().unwrap();
            let value_type = match kind {
                "continuous" => ValueType::Continuous,
                "binary" => ValueType::Binary,
                "count" => ValueType::Count,
                "unspecified" => ValueType::Unspecified,
                "categorical" => ValueType::Categorical,
                "ordinal" => ValueType::Ordinal,
                other => panic!("{other}"),
            };
            (label.to_owned(), value_type, parse_col(&c["values"]))
        };
        columns.push(typed("y", &spec["outcome"]));
        let mut predictors = Vec::new();
        let mut conditioning = Vec::new();
        for (i, c) in spec["predictors"].as_array().unwrap().iter().enumerate() {
            columns.push(typed(&format!("p{i}"), c));
            predictors.push(VariableId::from_raw(u32::try_from(columns.len() - 1).unwrap()));
        }
        for (i, c) in spec["conditioning"].as_array().unwrap().iter().enumerate() {
            columns.push(typed(&format!("c{i}"), c));
            conditioning.push(VariableId::from_raw(u32::try_from(columns.len() - 1).unwrap()));
        }
        let functional = match spec["functional"].as_str().unwrap() {
            "auto" => AssociationFunctional::Auto,
            "conditional" => AssociationFunctional::Conditional,
            "coefficients" => AssociationFunctional::Coefficients,
            "pearson" => AssociationFunctional::Measure(AssociationMeasure::Pearson),
            "distance" => AssociationFunctional::Measure(AssociationMeasure::Distance),
            _ => AssociationFunctional::Predictive { folds: 5 },
        };
        let scale = match spec["scale"].as_str().unwrap() {
            "schema" => AssociationOutcome::Schema,
            "continuous" => AssociationOutcome::Continuous,
            "binary" => AssociationOutcome::Binary,
            _ => AssociationOutcome::Count,
        };
        let query = AssociationQuery::new(VariableId::from_raw(0), predictors)
            .with_conditioning(conditioning)
            .with_functional(functional)
            .with_outcome_scale(scale);
        let storage = storage(&columns);
        let got = select_analysis(&storage, &query, Modality::CrossSection);
        let want = &spec["expect"];
        if want["refused"] == true {
            assert!(got.is_err(), "{name}: should be refused");
            continue;
        }
        let selection = got.unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(
            format!("{:?}", selection.analysis),
            want["analysis"].as_str().unwrap(),
            "{name}"
        );
        assert_eq!(
            format!("{:?}", selection.outcome_kind),
            want["kind"].as_str().unwrap(),
            "{name}"
        );
        assert_eq!(selection.outcome_source, want["source"].as_str().unwrap(), "{name}");
    }
}

// ---------------------------------------------------------------------------------- refusals
fn run(
    columns: &[(&str, Vec<f64>)],
    outcome: u32,
    predictors: &[u32],
    functional: AssociationFunctional,
    scale: AssociationOutcome,
    weight: Option<u32>,
    estimator: &AssociationEstimator,
) -> Result<AssociationEstimate, antecedent_estimate::EstimationError> {
    let cols: Vec<(String, ValueType, Col)> = columns
        .iter()
        .map(|(n, v)| ((*n).to_owned(), ValueType::Continuous, Col::Num(v.clone())))
        .collect();
    let mut query = AssociationQuery::new(
        VariableId::from_raw(outcome),
        predictors.iter().map(|p| VariableId::from_raw(*p)).collect::<Vec<_>>(),
    )
    .with_functional(functional)
    .with_outcome_scale(scale);
    if let Some(w) = weight {
        query = query.with_weight(VariableId::from_raw(w));
    }
    let storage = storage(&cols);
    let prepared = estimator.prepare(&storage, &query)?;
    estimator.estimate(&prepared, &ExecutionContext::for_tests(SEED))
}

fn code<T>(result: Result<T, antecedent_estimate::EstimationError>) -> &'static str {
    result.err().expect("should be refused").reason_code().unwrap_or("uncoded")
}

#[test]
fn declarations_the_estimator_cannot_honour_are_refused_with_their_reason() {
    let n = 30;
    let x: Vec<f64> = (0..n).map(|i| f64::from(i) / 4.0).collect();
    let y: Vec<f64> = (0..n).map(|i| f64::from(i % 7) + f64::from(i) / 5.0).collect();
    let flat = vec![1.0; n as usize];
    let two = |a: &[f64]| ("x", a.to_vec());
    let cols = [("y", y.clone()), two(&x)];
    let coef = AssociationFunctional::Coefficients;
    let auto = AssociationOutcome::Schema;
    let freq = AssociationEstimator::new(Modality::CrossSection);
    // Fewer than twenty draws form no posterior interval.
    let few = AssociationEstimator::new(Modality::CrossSection)
        .with_bayesian(BayesianSettings { draws: 19, prior_scale: 1.0 });
    let weighted =
        [("y", y.clone()), two(&x), ("w", (0..n).map(|i| f64::from(i % 3 + 1)).collect())];
    assert_eq!(
        code(run(&weighted, 0, &[1], coef, auto, Some(2), &few)),
        "posterior_draws_below_supported"
    );
    // A Bayesian reading takes draws, not replicates.
    let both = AssociationEstimator::new(Modality::CrossSection)
        .with_bayesian(BayesianSettings { draws: 30, prior_scale: 1.0 })
        .with_bootstrap_replicates(30);
    assert_eq!(code(run(&cols, 0, &[1], coef, auto, None, &both)), "option_not_applicable");
    // The distance correlation has no weighted form.
    let distance = AssociationFunctional::Measure(AssociationMeasure::Distance);
    assert_eq!(
        code(run(&weighted, 0, &[1], distance, auto, Some(2), &freq)),
        "route_not_supported"
    );
    let bayes = AssociationEstimator::new(Modality::CrossSection)
        .with_bayesian(BayesianSettings { draws: 30, prior_scale: 1.0 });
    assert_eq!(code(run(&cols, 0, &[1], distance, auto, None, &bayes)), "route_not_supported");
    // Weights: negative is invalid, all zero is invalid.
    let negative = [("y", y.clone()), two(&x), ("w", vec![-1.0; n as usize])];
    assert_eq!(
        code(run(&negative, 0, &[1], coef, auto, Some(2), &freq)),
        "sampling_weights_invalid"
    );
    let zero = [("y", y.clone()), two(&x), ("w", vec![0.0; n as usize])];
    assert_eq!(code(run(&zero, 0, &[1], coef, auto, Some(2), &freq)), "sampling_weights_invalid");
    // A model with no residual degree of freedom, a duplicated column, a constant correlation.
    let tiny = [("y", vec![1.0, 2.0]), ("x", vec![0.0, 1.0])];
    assert_eq!(code(run(&tiny, 0, &[1], coef, auto, None, &freq)), "invalid_argument");
    let twin = [("y", y.clone()), two(&x), ("z", x.clone())];
    assert_eq!(code(run(&twin, 0, &[1, 2], coef, auto, None, &freq)), "design_rank_deficient");
    let pearson = AssociationFunctional::Measure(AssociationMeasure::Pearson);
    let constant = [("y", y.clone()), ("x", flat.clone())];
    assert_eq!(code(run(&constant, 0, &[1], pearson, auto, None, &freq)), "invalid_argument");
    // The distance correlation of a constant column is as undefined as the Pearson one: refused,
    // never reported as zero.
    assert_eq!(code(run(&constant, 0, &[1], distance, auto, None, &freq)), "invalid_argument");
    // Scale violations: a 0/1 outcome holding 2, a count holding a fraction.
    let not_binary =
        [("y", vec![0.0, 1.0, 2.0, 1.0, 0.0, 1.0]), ("x", vec![1.0, 2.0, 3.0, 4.0, 5.0, 7.0])];
    assert_eq!(
        code(run(&not_binary, 0, &[1], coef, AssociationOutcome::Binary, None, &freq)),
        "invalid_argument"
    );
    assert_eq!(
        code(run(&cols, 0, &[1], coef, AssociationOutcome::Count, None, &freq)),
        "invalid_argument"
    );
    // A perfectly separated outcome has no finite logistic fit.
    let separated = [
        ("y", (0..n).map(|i| f64::from(i >= 15)).collect()),
        ("x", (0..n).map(f64::from).collect()),
    ];
    assert_eq!(code(run(&separated, 0, &[1], coef, auto, None, &freq)), "design_rank_deficient");
    // The declared sampling design must be the one the modality can honour.
    let wrong = AssociationEstimator::new(Modality::CrossSection).with_sampling_design(
        antecedent_estimate::util::SamplingDeclaration::Declared(
            antecedent_core::DeclaredSamplingDesign::DependentTimeSeries,
        ),
    );
    assert_eq!(code(run(&cols, 0, &[1], coef, auto, None, &wrong)), "route_not_supported");
}

#[test]
fn predictors_crossing_into_too_many_strata_are_refused() {
    let rows = 5000_usize;
    let level = |div: usize| (0..rows).map(|i| ((i / div) % 17) as f64).collect::<Vec<f64>>();
    let columns = [
        ("y", (0..rows).map(|i| (i % 5) as f64).collect()),
        ("a", level(1)),
        ("b", level(17)),
        ("c", level(289)),
    ];
    let result = run(
        &columns,
        0,
        &[1, 2, 3],
        AssociationFunctional::Conditional,
        AssociationOutcome::Schema,
        None,
        &AssociationEstimator::new(Modality::CrossSection),
    );
    assert_eq!(code(result), "invalid_argument");
}

#[test]
fn the_reference_covers_every_branch_it_names() {
    let reference = reference();
    assert!(reference["cases"].as_array().unwrap().len() >= 29);
    assert!(reference["resampling"].as_array().unwrap().len() >= 24);
    assert!(reference["selection"].as_array().unwrap().len() >= 33);
}
