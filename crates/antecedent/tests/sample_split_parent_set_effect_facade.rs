//! Discovery on a row split followed by an effect adjusted for the treatment's parent set,
//! through the facade, on sparse connected models of ten to twenty variables.
//!
//! Oracles: the truth of every study is computed from the structural coefficients alone (the
//! sum over directed paths from the treatment to the outcome), never from the simulated rows;
//! the reported effect is the least-squares coefficient of the treatment given the true parent
//! set on the analysis rows, solved by a routine written here; the unresolved edges the result
//! lists are exactly the nuisance-chain edges the equivalence class cannot orient; and an
//! undirected edge at the treatment is refused with that edge named, never completed.

#![allow(clippy::float_cmp, reason = "the declared level and replayed values are compared exactly")]
use antecedent::analysis::sample_split_analysis::{SplitRule, split_rows};
use antecedent::analysis::sample_split_parent_set_effect::{
    DISCOVERY_FRACTION, Knowledge, LINEAR_ESTIMATOR, LOG_LINEAR_ESTIMATOR, PARENT_SET_STRUCTURE,
    ParentSetEffect, ParentSetOptions, Transformation, estimate_parent_set_effect,
    estimate_parent_set_effect_with_knowledge,
};
use antecedent_core::{ExecutionContext, VariableId};
use antecedent_data::{TableView, TabularData};
use antecedent_io::measured_inference::split_parent_set_effect::ParentSetEffectSource;
use std::collections::BTreeSet;

#[path = "common/sparse_scm.rs"]
mod sparse_scm;

use sparse_scm::{MAX_DEGREE, Scm, Topology, Variant, invert};

const SEED: u64 = 4_242;
// This deterministic fixture admits both the historical half split and the version-two
// three-quarter discovery split; calibration keeps its separate, fixed seed schedule.
const SPLIT_SEED: u64 = 19;
const ROWS: usize = 4_000;
const WEIGHT_SEED: u64 = 0x5EED_0011;
const DATA_SEED: u64 = 0xDA7A_0022;

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(SEED)
}

fn v(index: usize) -> VariableId {
    VariableId::from_raw(u32::try_from(index).expect("small index"))
}

fn table(scm: &Scm, columns: &[Vec<f64>]) -> TabularData {
    TabularData::from_f64_columns(
        scm.names.iter().map(String::as_str).zip(columns.iter().map(Vec::as_slice)),
    )
    .expect("table")
}

struct Study {
    scm: Scm,
    data: TabularData,
}

fn study(topology: Topology, variables: usize, log: bool, rows: usize) -> Study {
    let scm = Scm::new(topology, variables, WEIGHT_SEED + variables as u64);
    // The fixture is a member of the published family: its strong-faithfulness margin holds.
    assert!(scm.margin() >= sparse_scm::MIN_MARGIN, "margin {}", scm.margin());
    let seed = DATA_SEED + variables as u64;
    let columns = if log { scm.simulate_log_linear(rows, seed) } else { scm.simulate(rows, seed) };
    let data = table(&scm, &columns);
    Study { scm, data }
}

fn estimate(study: &Study, options: ParentSetOptions) -> ParentSetEffect {
    let (x, y) = (study.scm.treatment, study.scm.outcome);
    estimate_parent_set_effect(&study.data, v(x), v(y), options, SPLIT_SEED, &ctx()).unwrap()
}

/// Least squares of `y` on an intercept, `x` and `adjustment`, by the normal equations:
/// `(coefficient of x, its standard error)`.
fn least_squares(table: &TabularData, x: usize, y: usize, adjustment: &[usize]) -> (f64, f64) {
    let column = |index: usize| table.float64_values(v(index)).unwrap();
    let (xs, ys) = (column(x), column(y));
    let covariates: Vec<Vec<f64>> = adjustment.iter().map(|&i| column(i)).collect();
    let n = ys.len();
    let k = 2 + covariates.len();
    let row = |i: usize| -> Vec<f64> {
        let mut r = vec![1.0, xs[i]];
        r.extend(covariates.iter().map(|c| c[i]));
        r
    };
    let mut gram = vec![vec![0.0; k]; k];
    let mut moment = vec![0.0; k];
    for i in 0..n {
        let r = row(i);
        for a in 0..k {
            moment[a] += r[a] * ys[i];
            for b in 0..k {
                gram[a][b] += r[a] * r[b];
            }
        }
    }
    let inverse = invert(&gram);
    let beta: Vec<f64> = (0..k).map(|a| (0..k).map(|b| inverse[a][b] * moment[b]).sum()).collect();
    let rss: f64 = (0..n)
        .map(|i| {
            let fitted: f64 = row(i).iter().zip(&beta).map(|(a, b)| a * b).sum();
            (ys[i] - fitted).powi(2)
        })
        .sum();
    (beta[1], (rss / (n - k) as f64 * inverse[1][1]).sqrt())
}

/// The analysis rows of the construction's split of the (declared-scale) table.
fn analysis_rows(study: &Study, log: bool) -> TabularData {
    let columns: Vec<Vec<f64>> = (0..study.scm.variables())
        .map(|i| {
            let raw = study.data.float64_values(v(i)).unwrap();
            if log { raw.iter().map(|value| value.ln()).collect() } else { raw }
        })
        .collect();
    let scaled = table(&study.scm, &columns);
    let variables: Vec<VariableId> = (0..study.scm.variables()).map(v).collect();
    split_rows(
        &scaled,
        &variables,
        SplitRule::SeededPermutation { seed: SPLIT_SEED, discovery_fraction: DISCOVERY_FRACTION },
    )
    .unwrap()
    .analysis
}

fn unordered(a: usize, b: usize) -> (u32, u32) {
    (u32::try_from(a.min(b)).unwrap(), u32::try_from(a.max(b)).unwrap())
}

/// The nuisance-chain edges the class cannot orient: those that hang off a node with no
/// incoming edge. The chain rooted at the mediator that follows the treatment is oriented away
/// from it.
fn undirected_chain_edges(scm: &Scm) -> BTreeSet<(u32, u32)> {
    let oriented_root = match scm.topology {
        Topology::HubConfoundedMultiPath | Topology::ConfoundedDirectAndMediated => scm.index("m"),
        Topology::LayeredMultiMediatorNuisance => scm.index("m2"),
    };
    let mut edges = BTreeSet::new();
    for (child, name) in scm.names.iter().enumerate() {
        if !name.starts_with('n') {
            continue;
        }
        let (parent, _) = scm.parents[child][0];
        let mut root = parent;
        while scm.names[root].starts_with('n') {
            root = scm.parents[root][0].0;
        }
        if root != oriented_root {
            edges.insert(unordered(parent, child));
        }
    }
    edges
}

/// The truth by independent intervention simulation (never the path product).
fn simulated_truth(scm: &Scm, log: bool, shift: f64) -> f64 {
    if log {
        scm.simulated_mean_ratio(shift, 4_000, 99)
    } else {
        scm.simulated_total_effect(4_000, 99)
    }
}

fn assert_known_truth(topology: Topology, variables: usize, log: bool) {
    let study = study(topology, variables, log, ROWS);
    let scm = &study.scm;
    assert!(scm.degrees().iter().all(|&d| d <= MAX_DEGREE));
    let options =
        if log { ParentSetOptions::log_linear(4, 1.0) } else { ParentSetOptions::linear(4) };
    let result = estimate(&study, options);
    let truth = simulated_truth(scm, log, 1.0);

    let mut parents: Vec<u32> =
        scm.treatment_parents().iter().map(|&p| u32::try_from(p).unwrap()).collect();
    parents.sort_unstable();
    assert_eq!(result.treatment_parents(), parents.as_slice());
    assert_eq!(result.discovery_rows() + result.analysis_rows(), ROWS as u64);

    let reported = result.reported();
    assert_eq!(
        result.transformation(),
        if log { Transformation::LogLinear } else { Transformation::Linear }
    );
    assert!(reported.lower < reported.point && reported.point < reported.upper);
    // The point lies within four standard errors of the independently computed truth.
    assert!(
        (reported.point - truth).abs() <= 4.0 * reported.standard_error,
        "{variables} variables, log={log}: estimate {} vs truth {truth} (se {})",
        reported.point,
        reported.standard_error
    );

    // Unrelated edges stay unresolved and are listed, none of them at the treatment.
    let listed: BTreeSet<(u32, u32)> =
        result.unresolved_edges().iter().map(|e| (e.a, e.b)).collect();
    let x = u32::try_from(scm.treatment).unwrap();
    assert!(listed.iter().all(|&(a, b)| a != x && b != x));
    assert_eq!(listed, undirected_chain_edges(scm), "the unresolved edges are the chain edges");
    assert!(!listed.is_empty());
}

#[test]
fn intervention_simulation_agrees_with_the_path_products_for_every_topology() {
    for topology in Topology::for_degree(MAX_DEGREE) {
        for variables in [10, 15, 20] {
            let scm = Scm::new(topology, variables, WEIGHT_SEED + variables as u64);
            assert!(scm.margin() >= sparse_scm::MIN_MARGIN);
            let linear = scm.simulated_total_effect(2_000, 5);
            assert!((linear - scm.total_effect()).abs() < 1e-9, "{topology:?} {variables}");
            let ratio = scm.simulated_mean_ratio(0.5, 2_000, 5);
            assert!((ratio / scm.mean_ratio(0.5) - 1.0).abs() < 1e-9, "{topology:?} {variables}");
        }
    }
}

#[test]
fn a_direct_edge_and_a_mediated_path_under_confounding_recover_the_total_effect() {
    let scm = Scm::new(Topology::ConfoundedDirectAndMediated, 15, WEIGHT_SEED);
    // The total effect is the direct coefficient plus the mediated path product.
    let (x, m, y) = (scm.index("x"), scm.index("m"), scm.index("y"));
    let direct = scm.parents[y].iter().find(|&&(a, _)| a == x).unwrap().1;
    let via = scm.parents[m][0].1 * scm.parents[y].iter().find(|&&(a, _)| a == m).unwrap().1;
    assert!((scm.total_effect() - (direct + via)).abs() < 1e-12);
    for log in [false, true] {
        assert_known_truth(Topology::ConfoundedDirectAndMediated, 15, log);
    }
}

#[test]
fn sparse_linear_study_of_ten_variables_recovers_the_total_effect() {
    assert_known_truth(Topology::HubConfoundedMultiPath, 10, false);
}

#[test]
fn sparse_linear_study_of_fifteen_variables_recovers_the_total_effect() {
    assert_known_truth(Topology::LayeredMultiMediatorNuisance, 15, false);
}

#[test]
fn sparse_linear_study_of_twenty_variables_recovers_the_total_effect() {
    assert_known_truth(Topology::HubConfoundedMultiPath, 20, false);
}

#[test]
fn sparse_log_linear_study_of_ten_variables_recovers_the_mean_ratio() {
    assert_known_truth(Topology::LayeredMultiMediatorNuisance, 10, true);
}

#[test]
fn sparse_log_linear_study_of_fifteen_variables_recovers_the_mean_ratio() {
    assert_known_truth(Topology::HubConfoundedMultiPath, 15, true);
}

#[test]
fn sparse_log_linear_study_of_twenty_variables_recovers_the_mean_ratio() {
    assert_known_truth(Topology::LayeredMultiMediatorNuisance, 20, true);
}

#[test]
fn the_total_effect_is_the_least_squares_coefficient_given_the_true_parents_on_the_analysis_rows() {
    let study = study(Topology::HubConfoundedMultiPath, 15, false, ROWS);
    let scm = &study.scm;
    let result = estimate(&study, ParentSetOptions::linear(4));
    let rows = analysis_rows(&study, false);
    assert_eq!(result.analysis_rows(), (ROWS / 4) as u64);
    assert_eq!(result.discovery_rows(), (3 * ROWS / 4) as u64);
    let (coefficient, se) =
        least_squares(&rows, scm.treatment, scm.outcome, &scm.treatment_parents());
    let reported = result.reported();
    assert!((reported.point - coefficient).abs() < 1e-9, "{} vs {coefficient}", reported.point);
    assert!((reported.standard_error - se).abs() < 1e-9);
    // A Student quantile on about 1000 degrees of freedom times the standard error.
    let half_width = (reported.upper - reported.lower) / 2.0;
    assert!((half_width / se - 1.961).abs() < 5e-3, "{}", half_width / se);
    // The split is a partition: the receipt's row sets share no row.
    let variables: Vec<VariableId> = (0..15).map(v).collect();
    let split = split_rows(
        &study.data,
        &variables,
        SplitRule::SeededPermutation { seed: SPLIT_SEED, discovery_fraction: DISCOVERY_FRACTION },
    )
    .unwrap();
    let discovery: BTreeSet<usize> = split.discovery_row_ids.iter().copied().collect();
    assert!(split.analysis_row_ids.iter().all(|row| !discovery.contains(row)));
}

#[test]
fn the_mean_ratio_is_the_exponential_of_the_log_scale_coefficient_and_its_interval() {
    let study = study(Topology::HubConfoundedMultiPath, 15, true, ROWS);
    let scm = &study.scm;
    let shift = 0.5;
    let result = estimate(&study, ParentSetOptions::log_linear(4, shift));
    let rows = analysis_rows(&study, true);
    let (coefficient, se) =
        least_squares(&rows, scm.treatment, scm.outcome, &scm.treatment_parents());
    let reported = result.reported();
    let ratio = (shift * coefficient).exp();
    assert!((reported.point - ratio).abs() < 1e-9 * ratio, "{} vs {ratio}", reported.point);
    // The interval is the exponential image of the Student interval on the log scale, and the
    // standard error is the delta-method image of the coefficient's.
    let quantile = (reported.upper.ln() - reported.lower.ln()) / (2.0 * shift * se);
    assert!((quantile - 1.961).abs() < 5e-3, "{quantile}");
    assert!(((reported.upper * reported.lower).sqrt() / ratio - 1.0).abs() < 1e-9);
    assert!((reported.standard_error - ratio * shift * se).abs() < 1e-9 * ratio);
    assert!(reported.lower > 0.0);
    assert_eq!(&*result.keyed_basis().estimator, LOG_LINEAR_ESTIMATOR);
}

#[test]
fn unambiguous_treatment_parents_are_accepted_with_unrelated_edges_unresolved() {
    let study = study(Topology::HubConfoundedMultiPath, 15, false, ROWS);
    let scm = &study.scm;
    let result = estimate(&study, ParentSetOptions::linear(4));
    let listed: BTreeSet<(u32, u32)> =
        result.unresolved_edges().iter().map(|e| (e.a, e.b)).collect();
    // Eight nuisance-chain edges hang off the roots h and p; the chain off the mediator is
    // oriented away from the treatment.
    assert!(listed.len() >= 3);
    assert_eq!(listed, undirected_chain_edges(scm));
    assert!(
        result.unresolved_edges().iter().all(|e| e.reason == "undirected_in_equivalence_class")
    );
    // The class is a partially directed graph, not a completed single graph, and the effect is
    // reported from it.
    assert!(result.reported().point.is_finite());
    assert_eq!(result.edge_count(), scm.edges().len());
    // Every listed edge is a true edge: nothing is invented.
    let truth: BTreeSet<(u32, u32)> = scm.edges().iter().map(|&(a, b)| unordered(a, b)).collect();
    assert!(listed.is_subset(&truth));
}

#[test]
fn an_undirected_edge_at_the_treatment_is_refused_and_named_not_completed() {
    let study = study(Topology::HubConfoundedMultiPath, 15, false, ROWS);
    let scm = &study.scm;
    // The hub has oriented edges into the treatment and the outcome, but its edge
    // to the first nuisance node is undirected: the hub's parent set is not the same in every
    // member of the class.
    let hub = scm.index("h");
    let error = estimate_parent_set_effect(
        &study.data,
        v(hub),
        v(scm.outcome),
        ParentSetOptions::linear(4),
        SPLIT_SEED,
        &ctx(),
    )
    .unwrap_err();
    let text = error.to_string();
    assert!(text.contains("cell_not_licensed"), "{text}");
    assert!(text.contains("parent_set_effect.treatment_parents_ambiguous"), "{text}");
    assert!(text.contains("h - n0"), "{text}");
    assert!(text.contains("no completion is chosen"), "{text}");
}

#[test]
fn log_linear_inputs_must_be_positive_and_the_refusal_names_the_column() {
    let log_study = study(Topology::HubConfoundedMultiPath, 15, true, ROWS);
    let scm = &log_study.scm;
    let mut columns: Vec<Vec<f64>> =
        (0..15).map(|i| log_study.data.float64_values(v(i)).unwrap()).collect();
    let target = scm.index("m");
    for value in columns[target].iter_mut().take(5) {
        *value = -*value;
    }
    let data = table(scm, &columns);
    let error = estimate_parent_set_effect(
        &data,
        v(scm.treatment),
        v(scm.outcome),
        ParentSetOptions::log_linear(4, 1.0),
        SPLIT_SEED,
        &ctx(),
    )
    .unwrap_err();
    let text = error.to_string();
    assert!(text.contains("invalid_argument"), "{text}");
    assert!(text.contains("parent_set_effect.log_linear_input_not_positive"), "{text}");
    assert!(text.contains("column `m` has 5 non-positive value(s)"), "{text}");
    // Positivity is a log-linear requirement only: the margin-checked linear fixture with the
    // same sign flips is a valid linear study (the log-normal columns above are not linear data).
    let linear = study(Topology::HubConfoundedMultiPath, 15, false, ROWS);
    let mut columns: Vec<Vec<f64>> =
        (0..15).map(|i| linear.data.float64_values(v(i)).unwrap()).collect();
    for value in columns[target].iter_mut().take(5) {
        *value = -value.abs();
    }
    let flipped = table(&linear.scm, &columns);
    estimate_parent_set_effect(
        &flipped,
        v(linear.scm.treatment),
        v(linear.scm.outcome),
        ParentSetOptions::linear(4),
        SPLIT_SEED,
        &ctx(),
    )
    .unwrap();
}

#[test]
fn the_transformation_must_be_declared_before_the_data_are_touched() {
    for name in ["undeclared", "", "log", "Linear"] {
        let error = Transformation::parse(name).unwrap_err();
        let text = error.to_string();
        assert!(text.contains("invalid_argument"), "{text}");
        assert!(text.contains("parent_set_effect.transformation_undeclared"), "{text}");
        assert!(text.contains("before any data are read"), "{text}");
    }
    assert_eq!(Transformation::parse("linear").unwrap(), Transformation::Linear);
    assert_eq!(Transformation::parse("log_linear").unwrap(), Transformation::LogLinear);
}

#[test]
fn a_log_linear_study_declares_its_shift_and_a_linear_study_declares_none() {
    let study = study(Topology::HubConfoundedMultiPath, 10, true, 400);
    let (x, y) = (v(study.scm.treatment), v(study.scm.outcome));
    for options in [
        ParentSetOptions {
            transformation: Transformation::LogLinear,
            max_degree: 4,
            log_shift: None,
        },
        ParentSetOptions::log_linear(4, 0.0),
        ParentSetOptions::log_linear(4, f64::NAN),
        ParentSetOptions {
            transformation: Transformation::Linear,
            max_degree: 4,
            log_shift: Some(1.0),
        },
    ] {
        let error =
            estimate_parent_set_effect(&study.data, x, y, options, SPLIT_SEED, &ctx()).unwrap_err();
        assert!(error.to_string().contains("parent_set_effect.source_invalid"), "{options:?}");
    }
}

#[test]
fn a_maximum_degree_above_four_is_refused_with_both_degrees_named() {
    let study = study(Topology::LayeredMultiMediatorNuisance, 15, false, 600);
    let (x, y) = (v(study.scm.treatment), v(study.scm.outcome));
    let error = estimate_parent_set_effect(
        &study.data,
        x,
        y,
        ParentSetOptions::linear(5),
        SPLIT_SEED,
        &ctx(),
    )
    .unwrap_err();
    let text = error.to_string();
    assert!(text.contains("cell_not_licensed"), "{text}");
    assert!(text.contains("parent_set_effect.degree_above_supported"), "{text}");
    assert!(text.contains("declared maximum graph degree is 5"), "{text}");
    assert!(text.contains("supported maximum is 4"), "{text}");
    let zero = estimate_parent_set_effect(
        &study.data,
        x,
        y,
        ParentSetOptions::linear(0),
        SPLIT_SEED,
        &ctx(),
    )
    .unwrap_err();
    assert!(zero.to_string().contains("parent_set_effect.source_invalid"));
}

#[test]
fn a_table_outside_ten_to_twenty_variables_is_refused() {
    let a: Vec<f64> = (0..60).map(|i| f64::from(i).sin()).collect();
    let b: Vec<f64> = (0..60).map(|i| f64::from(i).cos()).collect();
    let data = TabularData::from_f64_columns([("a", a.as_slice()), ("b", b.as_slice())]).unwrap();
    let error = estimate_parent_set_effect(
        &data,
        v(0),
        v(1),
        ParentSetOptions::linear(4),
        SPLIT_SEED,
        &ctx(),
    )
    .unwrap_err();
    assert!(error.to_string().contains("parent_set_effect.variable_count_not_measured"));
}

#[test]
fn the_same_seeds_give_the_same_split_graph_and_estimate_and_another_seed_another_split() {
    let study = study(Topology::LayeredMultiMediatorNuisance, 10, false, ROWS);
    let first = estimate(&study, ParentSetOptions::linear(4));
    let again = estimate(&study, ParentSetOptions::linear(4));
    assert_eq!(first, again);
    assert_eq!(first.source_artifact(), again.source_artifact());
    assert_eq!(first.structure_identity(), again.structure_identity());
    let (x, y) = (v(study.scm.treatment), v(study.scm.outcome));
    let other = estimate_parent_set_effect(
        &study.data,
        x,
        y,
        ParentSetOptions::linear(4),
        SPLIT_SEED + 1,
        &ctx(),
    )
    .unwrap();
    assert_ne!(first.receipt_identity(), other.receipt_identity());
}

#[test]
fn the_result_preserves_the_graph_status_and_the_attestation_reports_it() {
    let study = study(Topology::HubConfoundedMultiPath, 10, false, ROWS);
    let result = estimate(&study, ParentSetOptions::linear(4));
    assert_eq!(result.acceptance(), "treatment_parent_set_unambiguous");
    assert!(!result.unresolved_edges().is_empty());
    match result.measure(&ctx()) {
        Ok(measured) => {
            // A current attesting record covers this construction: the claim states the
            // acceptance reason and the unresolved edges it left open.
            let scope = &measured.report().validated_scope;
            assert_eq!(scope["acceptance"], "treatment_parent_set_unambiguous");
            let listed = scope["unresolved_edges_not_incident_to_treatment"].as_array().unwrap();
            assert_eq!(listed.len(), result.unresolved_edges().len());
            assert_eq!(scope["structure"], PARENT_SET_STRUCTURE);
        }
        Err(error) => {
            let text = error.to_string();
            assert!(text.contains("cell_not_licensed"), "{text}");
            assert!(text.contains("measured_inference.record_not_attesting"), "{text}");
        }
    }
}

#[test]
fn the_basis_is_keyed_under_its_own_structure_estimator_and_variable_count() {
    let study = study(Topology::HubConfoundedMultiPath, 10, false, ROWS);
    let basis = estimate(&study, ParentSetOptions::linear(4)).keyed_basis();
    assert_eq!(&*basis.structure, PARENT_SET_STRUCTURE);
    assert_eq!(&*basis.estimator, LINEAR_ESTIMATOR);
    assert_eq!(&*basis.dependence, "disjoint_rows");
    assert_eq!(
        &*basis.functional,
        "total_effect.variables_10.max_degree_4.discovery_fraction_0_75.topology_hub_confounded_multi_path"
    );
    // A table does not announce its topology: attestation needs every topology's record.
    let bases = estimate(&study, ParentSetOptions::linear(4)).keyed_bases();
    let functionals: Vec<&str> = bases.iter().map(|b| &*b.functional).collect();
    assert_eq!(
        functionals,
        [
            "total_effect.variables_10.max_degree_4.discovery_fraction_0_75.topology_hub_confounded_multi_path",
            "total_effect.variables_10.max_degree_4.discovery_fraction_0_75.topology_layered_multi_mediator_nuisance",
            "total_effect.variables_10.max_degree_4.discovery_fraction_0_75.topology_confounded_direct_and_mediated",
        ]
    );
    assert_eq!(basis.row_count, (ROWS / 4) as u64);
    // Never the keys of the fully directed construction or of a declared graph.
    assert_ne!(&*basis.structure, "discovered_sample_split");
    assert_ne!(&*basis.estimator, "sample_split.pc_parcorr.linear_adjustment");
    assert_ne!(&*basis.structure, "fixed");
}

#[test]
fn version_one_sources_replay_the_half_split_and_keep_their_original_basis() {
    let study = study(Topology::HubConfoundedMultiPath, 10, false, ROWS);
    let mut source = ParentSetEffectSource::new(
        study.scm.names.clone(),
        (0..10).map(|i| study.data.float64_values(v(i)).unwrap()).collect(),
        u32::try_from(study.scm.treatment).expect("small treatment index"),
        u32::try_from(study.scm.outcome).expect("small outcome index"),
        SPLIT_SEED,
        SEED,
        Transformation::Linear,
        4,
        None,
    );
    assert_eq!(source.version, 2);
    let current = antecedent_io::measured_inference::split_parent_set_effect::run(&source, &ctx())
        .expect("version-two execution");
    source.version = 1;
    let old = antecedent_io::measured_inference::split_parent_set_effect::run(&source, &ctx())
        .expect("version-one execution");
    assert_eq!(old.discovery_rows, (ROWS / 2) as u64);
    assert_eq!(old.analysis_rows, (ROWS / 2) as u64);
    assert_eq!(current.discovery_rows, (3 * ROWS / 4) as u64);
    assert_eq!(current.analysis_rows, (ROWS / 4) as u64);
    assert_eq!(
        &*old.keyed_basis().functional,
        "total_effect.variables_10.max_degree_4.topology_hub_confounded_multi_path"
    );
    assert_ne!(current.keyed_basis().functional, old.keyed_basis().functional);
    assert_ne!(current.receipt_identity, old.receipt_identity);
    let variables: Vec<VariableId> = (0..10).map(v).collect();
    let half = split_rows(
        &study.data,
        &variables,
        SplitRule::SeededPermutation { seed: SPLIT_SEED, discovery_fraction: 0.5 },
    )
    .unwrap();
    let original = antecedent_io::measured_inference::split_effect::adjusted_effect(
        &half.analysis,
        source.treatment,
        source.outcome,
        &old.neighbourhood.parents,
        old.coefficient.level,
    )
    .unwrap();
    assert_eq!(old.coefficient, original);
}

#[test]
fn an_unmeasured_variable_count_is_computed_and_never_attested() {
    // Twelve variables lie between measured counts: the estimate is computed, and `measure`
    // refuses it rather than borrowing the ten- or fifteen-variable record.
    let study = study(Topology::HubConfoundedMultiPath, 12, false, ROWS);
    let result = estimate(&study, ParentSetOptions::linear(4));
    assert!(result.reported().lower < result.reported().upper);
    let text = result.measure(&ctx()).unwrap_err().to_string();
    assert!(text.contains("cell_not_licensed"), "{text}");
    assert!(text.contains("measured_inference.record_not_attesting"), "{text}");
}

#[test]
fn a_source_that_states_another_format_or_roles_is_refused() {
    let study = study(Topology::LayeredMultiMediatorNuisance, 10, false, ROWS);
    let result = estimate(&study, ParentSetOptions::linear(4));
    let source = ParentSetEffectSource::from_bytes(result.source_artifact()).unwrap();
    assert_eq!(source.to_bytes().unwrap(), result.source_artifact());

    let mut other_version = source.clone();
    other_version.version = 3;
    let error = antecedent::analysis::sample_split_parent_set_effect::measure_source(
        &other_version.to_bytes().unwrap(),
        &ctx(),
    )
    .unwrap_err();
    assert!(error.to_string().contains("parent_set_effect.source_invalid"));

    let mut same_role = source;
    same_role.outcome = same_role.treatment;
    let error = antecedent::analysis::sample_split_parent_set_effect::measure_source(
        &same_role.to_bytes().unwrap(),
        &ctx(),
    )
    .unwrap_err();
    assert!(error.to_string().contains("parent_set_effect.source_invalid"));

    let mut padded = result.source_artifact().to_vec();
    padded.push(0);
    assert!(
        antecedent::analysis::sample_split_parent_set_effect::measure_source(&padded, &ctx())
            .is_err()
    );
}

#[test]
fn the_published_conditions_are_listed_for_each_family() {
    let guide = include_str!("../../../docs/selective-discovery.md");
    let section = guide
        .split("## Effect after discovery on a treatment parent set")
        .nth(1)
        .expect("the guide has the parent-set section");
    for family in ["Linear", "Log-linear"] {
        assert!(section.contains(&format!("### {family} family")), "{family}");
    }
    for condition in [
        "Signal strength",
        "Causal sufficiency",
        "Distribution",
        "Intervention support",
        "Faithfulness",
        "Maximum degree",
    ] {
        assert_eq!(
            section.matches(&format!("**{condition}.**")).count(),
            2,
            "{condition} is stated once for each family"
        );
    }
    assert!(section.contains(&format!("{}", sparse_scm::MIN_MARGIN)));
}

fn with_knowledge(study: &Study, knowledge: Option<Knowledge>, degree: u32) -> ParentSetEffect {
    let scm = &study.scm;
    estimate_parent_set_effect_with_knowledge(
        &study.data,
        v(scm.treatment),
        v(scm.outcome),
        ParentSetOptions::linear(degree),
        knowledge,
        SPLIT_SEED,
        &ctx(),
    )
    .unwrap()
}

fn links(pairs: Vec<(usize, usize)>) -> Vec<(u32, u32)> {
    pairs.into_iter().map(|(a, b)| (u32::try_from(a).unwrap(), u32::try_from(b).unwrap())).collect()
}

/// Every background-knowledge regime, validated end to end on the known-truth study: the
/// statements are true of the model, so the treatment's parents and the total effect are still
/// recovered, the statements enter the calibration key, and tiers (which orient every edge)
/// leave no edge unresolved.
#[test]
fn each_background_knowledge_regime_runs_end_to_end_on_the_known_truth_study() {
    let study = study(Topology::HubConfoundedMultiPath, 10, false, ROWS);
    let scm = &study.scm;
    let truth = simulated_truth(scm, false, 1.0);
    let mut parents: Vec<u32> =
        scm.treatment_parents().iter().map(|&p| u32::try_from(p).unwrap()).collect();
    parents.sort_unstable();
    let regimes: [(&str, Knowledge); 3] = [
        ("required", Knowledge { required: links(scm.required_links()), ..Knowledge::default() }),
        (
            "forbidden",
            Knowledge { forbidden: links(scm.forbidden_links()), ..Knowledge::default() },
        ),
        (
            "tiers",
            Knowledge {
                tiers: scm
                    .tiers()
                    .into_iter()
                    .map(|tier| tier.into_iter().map(|v| u32::try_from(v).unwrap()).collect())
                    .collect(),
                ..Knowledge::default()
            },
        ),
    ];
    for (label, knowledge) in regimes {
        let result = with_knowledge(&study, Some(knowledge), 4);
        assert_eq!(result.treatment_parents(), parents.as_slice(), "{label}");
        let reported = result.reported();
        assert!(
            (reported.point - truth).abs() <= 4.0 * reported.standard_error,
            "{label}: {} vs {truth}",
            reported.point
        );
        assert!(
            result.keyed_basis().functional.contains(&format!(".knowledge_{label}.")),
            "{label}: {}",
            result.keyed_basis().functional
        );
        if label == "tiers" {
            assert!(result.unresolved_edges().is_empty(), "tiers orient every edge");
        }
    }
    // Knowledge changes the source, so a measured cell cannot be borrowed from the knowledge-free one.
    let free = with_knowledge(&study, None, 4);
    assert_ne!(
        free.source_artifact(),
        with_knowledge(
            &study,
            Some(Knowledge { required: links(scm.required_links()), ..Knowledge::default() }),
            4
        )
        .source_artifact()
    );
}

#[test]
fn a_tight_declared_degree_runs_end_to_end_on_a_degree_three_study() {
    let scm =
        Scm::with_degree(Topology::HubConfoundedMultiPath, 10, WEIGHT_SEED, Variant::Licensed, 3);
    assert!(scm.degrees().iter().all(|&d| d <= 3));
    let columns = scm.simulate(ROWS, DATA_SEED);
    let study = Study { data: table(&scm, &columns), scm };
    let result = with_knowledge(&study, None, 3);
    assert!(result.keyed_basis().functional.contains(".max_degree_3."));
    let truth = simulated_truth(&study.scm, false, 1.0);
    let reported = result.reported();
    assert!((reported.point - truth).abs() <= 4.0 * reported.standard_error);
    // The denser third core has degree four and does not belong to a degree-three family.
    assert_eq!(result.keyed_bases().len(), 2);
}

#[test]
fn knowledge_naming_a_column_that_is_not_searched_is_refused() {
    let study = study(Topology::HubConfoundedMultiPath, 10, false, 600);
    let scm = &study.scm;
    let error = estimate_parent_set_effect_with_knowledge(
        &study.data,
        v(scm.treatment),
        v(scm.outcome),
        ParentSetOptions::linear(4),
        Some(Knowledge { required: vec![(0, 99)], ..Knowledge::default() }),
        SPLIT_SEED,
        &ctx(),
    )
    .unwrap_err();
    let text = error.to_string();
    assert!(text.contains("invalid_argument"), "{text}");
    assert!(text.contains("parent_set_effect.knowledge_invalid"), "{text}");
}
