//! conformance: propensity IPW, IV/2SLS, front-door two-stage.
//!
//! Fixtures under `conformance/estimate/*` are clean-room synthetic SCMs generated inline
//! (deterministic from a fixed seed) — independent of any `pinned baseline` install or CSV fixture. Each
//! test checks `|estimate.ate - expected.true_effect| < expected.tolerance`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::many_single_char_names, clippy::doc_markdown)]

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use antecedent::Study;
use antecedent::{EstimatorId, IdentifierId};
use antecedent_core::{
    AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, MeasurementSpec, RoleHint,
    SmallRoleSet, StreamDomain, TargetPopulation, ValueType, VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
};
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_kernels::standard_normal;
use serde_json::Value as JsonValue;

fn fixture_dir(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../conformance/estimate").join(name)
}

fn load_expected(name: &str) -> JsonValue {
    let raw = fs::read_to_string(fixture_dir(name).join("expected.json")).expect("expected.json");
    serde_json::from_str(&raw).expect("parse expected.json")
}

/// Build `TabularData` from `(name, role, column)` triples; variable ids follow slice order.
fn tabular_data(vars: &[(&str, RoleHint, Vec<f64>)]) -> TabularData {
    let n = vars[0].2.len();
    let mut b = CausalSchemaBuilder::new();
    for (name, role, _) in vars {
        b.add_variable(
            *name,
            ValueType::Continuous,
            SmallRoleSet::from_hint(*role),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
    }
    let schema = b.build().unwrap();
    let cols: Vec<OwnedColumn> = vars
        .iter()
        .enumerate()
        .map(|(i, (_, _, data))| {
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(u32::try_from(i).unwrap()),
                    Arc::from(data.clone()),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            )
        })
        .collect();
    let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
    TabularData::new(storage)
}

fn assert_recovers(result: &antecedent::StudyResult, expected: &JsonValue) {
    let true_effect = expected["true_effect"].as_f64().unwrap();
    let tolerance = expected["tolerance"].as_f64().unwrap();
    assert!(
        (result.estimate.ate - true_effect).abs() < tolerance,
        "ate={} expected true_effect={} tolerance={}",
        result.estimate.ate,
        true_effect,
        tolerance
    );
    assert_eq!(result.logical_plan.identifier.as_deref(), expected["identifier"].as_str());
    assert_eq!(result.logical_plan.estimator.as_deref(), expected["estimator"].as_str());
}

/// Largest accepted `|ln(SE_rust·√n_rust / SE_ref·√n_ref)|`: `ln 1.5`.
///
/// The recorded reference SE (`reference.outputs.se`) comes from the reference's own
/// `n = 800` draw of the same SCM, so the two SEs are compared after `√n`
/// rescaling, never as raw numbers. `ln 1.5` absorbs the sampling noise of
/// two SE estimates from different draws (the reference side is a bootstrap SE)
/// while still failing every constant-factor bug of 2 or more — variance
/// reported as SD, a dropped `√2`, a missing `n/(n−1)` squared, or an SE
/// computed on the wrong row count.
const SE_LOG_RATIO_TOLERANCE: f64 = 0.405_465_108_108_164_4;

/// Fixtures whose recorded reference SE is not a reference for the Rust estimator.
///
/// `aipw`'s reference block ran `backdoor.propensity_score_weighting`, a different
/// estimator (its point estimate is byte-identical to `propensity_ipw`'s).
/// `propensity_ipw`'s reference SE (0.273 at n = 800) is about five times the
/// Monte Carlo sampling SD of any IPW estimator on the SCM this test draws
/// from (≈0.054 Hajek with a fitted logistic propensity, ≈0.11 Horvitz–Thompson
/// with the true one), so it cannot calibrate this SCM. Comparing either
/// would test the reference's recording, not this crate; the Rust SEs for these two
/// estimators are covered by the `antecedent-estimate` coverage gates instead.
const SE_NOT_COMPARABLE: [&str; 2] = ["propensity_ipw", "aipw"];

/// Compare the reported SE against the fixture's recorded reference SE.
fn assert_reference_se(result: &antecedent::StudyResult, name: &str, n: usize) {
    assert!(!SE_NOT_COMPARABLE.contains(&name), "{name} has no comparable reference SE");
    let expected = load_expected(name);
    let outputs = &expected["reference"]["outputs"];
    let reference_se = outputs["se"].as_f64().expect("reference.outputs.se");
    let reference_n = outputs["n"].as_f64().expect("reference.outputs.n");
    let se = if result.estimate.se_analytic.is_finite() && result.estimate.se_analytic > 0.0 {
        result.estimate.se_analytic
    } else {
        result.estimate.se_bootstrap.expect("an analytic or bootstrap SE")
    };
    let log_ratio = (se * (n as f64).sqrt() / (reference_se * reference_n.sqrt())).ln();
    eprintln!(
        "se check {}: se={se} n={n} reference_se={reference_se} reference_n={reference_n} \
         log_ratio={log_ratio:.4}",
        expected["estimator"]
    );
    assert!(
        log_ratio.abs() <= SE_LOG_RATIO_TOLERANCE,
        "{}: SE {se} (n={n}) vs recorded reference {reference_se} (n={reference_n}): \
         |ln ratio| = {:.3} > ln 1.5 after sqrt(n) rescaling",
        expected["estimator"],
        log_ratio.abs()
    );
}

/// `Z ~ N(0,1)` confounder; `T ~ Bernoulli(sigmoid(-0.4 + 0.9 Z))`; `Y = 2T + Z + noise`.
/// True ATE = 2; a naive unadjusted contrast is biased by `Z`, exercising IPW.
fn propensity_ipw_scm(n: usize, seed: u64) -> (TabularData, Dag, AverageEffectQuery) {
    let mut rng = ExecutionContext::for_tests(seed).rng.stream_for(StreamDomain::Test, 0x5051_u64);
    let mut z = vec![0.0; n];
    let mut t = vec![0.0; n];
    let mut y = vec![0.0; n];
    for i in 0..n {
        let zi = standard_normal(&mut rng);
        let logit = -0.4 + 0.9 * zi;
        let p = 1.0 / (1.0 + (-logit).exp());
        let ti = if rng.next_f64() < p { 1.0 } else { 0.0 };
        let noise = standard_normal(&mut rng) * 0.4;
        z[i] = zi;
        t[i] = ti;
        y[i] = 2.0 * ti + zi + noise;
    }
    let data = tabular_data(&[
        ("t", RoleHint::TreatmentCandidate, t),
        ("y", RoleHint::OutcomeCandidate, y),
        ("z", RoleHint::Context, z),
    ]);
    let mut dag = Dag::with_variables(3);
    dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap(); // z -> t
    dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap(); // z -> y
    dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap(); // t -> y
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    (data, dag, query)
}

#[test]
fn estimate_propensity_ipw_recovers_ate() {
    let expected = load_expected("propensity_ipw");
    let (data, graph, query) = propensity_ipw_scm(1200, 3);
    let analysis = Study::tabular(data)
        .graph(graph)
        .query(query)
        .identifier(expected["identifier"].as_str().unwrap().parse::<IdentifierId>().unwrap())
        .estimator(expected["estimator"].as_str().unwrap().parse::<EstimatorId>().unwrap())
        .bootstrap_replicates(30)
        .build()
        .unwrap();
    let ctx = ExecutionContext::for_tests(9);
    let result = analysis.run(&ctx).unwrap();
    assert_recovers(&result, &expected);
    // No SE comparison here: see `SE_NOT_COMPARABLE`.
    assert!(result.estimate.overlap_report.is_some(), "propensity.weighting must report overlap");
}

/// Binary instrument `Z`; unobserved confounder `U` (absent from the graph) with
/// `T = 0.6 Z + U + noise`, `Y = 2T + U + noise`. True structural effect = 2.
fn iv_2sls_scm(n: usize, seed: u64) -> (TabularData, Dag, AverageEffectQuery) {
    let mut rng = ExecutionContext::for_tests(seed).rng.stream_for(StreamDomain::Test, 0x5052_u64);
    let mut z = vec![0.0; n];
    let mut t = vec![0.0; n];
    let mut y = vec![0.0; n];
    for i in 0..n {
        let zi = (i % 2) as f64;
        let u = standard_normal(&mut rng);
        let ti = 0.6 * zi + u + 0.1 * standard_normal(&mut rng);
        let yi = 2.0 * ti + u + 0.1 * standard_normal(&mut rng);
        z[i] = zi;
        t[i] = ti;
        y[i] = yi;
    }
    let data = tabular_data(&[
        ("t", RoleHint::TreatmentCandidate, t),
        ("y", RoleHint::OutcomeCandidate, y),
        ("z", RoleHint::Context, z),
    ]);
    let mut dag = Dag::with_variables(3);
    dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap(); // z -> t
    dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap(); // t -> y
    let query =
        AverageEffectQuery::with_levels(VariableId::from_raw(0), VariableId::from_raw(1), 0.0, 1.0);
    (data, dag, query)
}

#[test]
fn estimate_iv_2sls_recovers_structural_effect() {
    let expected = load_expected("iv_2sls");
    let (data, graph, query) = iv_2sls_scm(4000, 5);
    let analysis = Study::tabular(data)
        .graph(graph)
        .query(query)
        .identifier(expected["identifier"].as_str().unwrap().parse::<IdentifierId>().unwrap())
        .estimator(expected["estimator"].as_str().unwrap().parse::<EstimatorId>().unwrap())
        .bootstrap_replicates(30)
        .build()
        .unwrap();
    let ctx = ExecutionContext::for_tests(21);
    let result = analysis.run(&ctx).unwrap();
    assert_recovers(&result, &expected);
    assert_reference_se(&result, "iv_2sls", 4000);
}

/// `U -> T -> M -> Y` with `U -> Y` (no direct `T -> Y` edge; `U` unmeasured, absent from the
/// graph). `T = 3 + U + noise` (uncentred, so a dropped intercept shows), `M = 0.4T + noise`,
/// `Y = 5M + U + noise`. True mediated effect = `0.4 * 5 = 2`; neither path coefficient alone
/// (0.4, 5) nor the confounded `Y ~ T` slope (about 3) is near it.
fn frontdoor_scm(n: usize, seed: u64) -> (TabularData, Dag, AverageEffectQuery) {
    let mut rng = ExecutionContext::for_tests(seed).rng.stream_for(StreamDomain::Test, 0x5053_u64);
    let mut t = vec![0.0; n];
    let mut m = vec![0.0; n];
    let mut y = vec![0.0; n];
    for i in 0..n {
        let u = standard_normal(&mut rng);
        let ti = 3.0 + u + 0.1 * standard_normal(&mut rng);
        let mi = 0.4 * ti + 0.1 * standard_normal(&mut rng);
        let yi = 5.0 * mi + u + 0.1 * standard_normal(&mut rng);
        t[i] = ti;
        m[i] = mi;
        y[i] = yi;
    }
    let data = tabular_data(&[
        ("t", RoleHint::TreatmentCandidate, t),
        ("y", RoleHint::OutcomeCandidate, y),
        ("m", RoleHint::Context, m),
    ]);
    (data, frontdoor_dag(), frontdoor_query())
}

fn frontdoor_dag() -> Dag {
    let mut dag = Dag::with_variables(3);
    dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap(); // t -> m
    dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap(); // m -> y
    dag
}

fn frontdoor_query() -> AverageEffectQuery {
    AverageEffectQuery::with_levels(VariableId::from_raw(0), VariableId::from_raw(1), 0.0, 1.0)
}

fn records_assumption(result: &antecedent::StudyResult, id: &str) -> bool {
    result.estimate.assumptions.entries.iter().any(|r| match &r.assumption {
        antecedent_core::Assumption::ParametricRestriction(p) => p.id.as_ref() == id,
        antecedent_core::Assumption::Custom { id: custom, .. } => custom.as_ref() == id,
        _ => false,
    })
}

#[test]
fn estimate_frontdoor_two_stage_recovers_mediated_effect() {
    let expected = load_expected("frontdoor");
    let (data, graph, query) = frontdoor_scm(4000, 1);
    let analysis = Study::tabular(data)
        .graph(graph)
        .query(query)
        .identifier(expected["identifier"].as_str().unwrap().parse::<IdentifierId>().unwrap())
        .estimator(expected["estimator"].as_str().unwrap().parse::<EstimatorId>().unwrap())
        .bootstrap_replicates(30)
        .build()
        .unwrap();
    let ctx = ExecutionContext::for_tests(41);
    let result = analysis.run(&ctx).unwrap();
    assert_recovers(&result, &expected);
    assert!(records_assumption(&result, "frontdoor.linear_path_product"));
    // Path coefficients alone must miss the product truth (T→M = 0.4, M→Y = 5).
    let truth = expected["true_effect"].as_f64().unwrap();
    let tol = expected["tolerance"].as_f64().unwrap();
    assert!((0.4 - truth).abs() > tol, "T→M slope alone must fail the band");
    assert!((5.0 - truth).abs() > tol, "M→Y coefficient alone must fail the band");
    assert!(
        (result.estimate.ate - 5.0).abs() > tol && (result.estimate.ate - 0.4).abs() > tol,
        "estimate {} must not collapse to a single path coefficient",
        result.estimate.ate
    );

    // Large-sample SE of the product `ab` in this linear Gaussian SCM, by the delta method with
    // independent stages: n·Var(a) = σ²_M / Var(T) = 0.01 / 1.01 and
    // n·Var(b) = Var(Y | T, M) / Var(M | T) = (Var(U | T) + 0.01) / 0.01 with
    // Var(U | T) = 0.01 / 1.01, so n·Var(ab) = 25·0.0099 + 0.16·1.990 = 0.566.
    let var_u_given_t = 0.01 / 1.01;
    let n_var = 25.0 * (0.01 / 1.01) + 0.16 * (var_u_given_t + 0.01) / 0.01;
    let closed_form = (n_var / 4000.0_f64).sqrt();
    let log_ratio = (result.estimate.se_analytic / closed_form).ln();
    assert!(
        log_ratio.abs() < 0.1,
        "se_analytic={} closed form={closed_form}",
        result.estimate.se_analytic
    );
}

/// Exact 4000-row population table of `U ~ Bern(.5)`, `P(T=1|U) = .1 + .5U`,
/// `P(M=1|T) = .1 + .7T`, `P(Y=1|M,U) = .05 + .9MU` with `U` dropped. The latent modifies the
/// mediator's effect, so the enumerated effect `0.7 · 0.9 · E[U] = 0.315` is reached by the
/// front-door functional and missed by the product of coefficients (0.363).
fn frontdoor_interaction_table() -> TabularData {
    let (mut t, mut m, mut y) = (Vec::new(), Vec::new(), Vec::new());
    for u in [0.0_f64, 1.0] {
        for ti in [0.0_f64, 1.0] {
            let pt = if ti == 1.0 { 0.1 + 0.5 * u } else { 0.9 - 0.5 * u };
            for mi in [0.0_f64, 1.0] {
                let pm = if mi == 1.0 { 0.1 + 0.7 * ti } else { 0.9 - 0.7 * ti };
                for yi in [0.0_f64, 1.0] {
                    let py1 = 0.05 + 0.9 * mi * u;
                    let py = if yi == 1.0 { py1 } else { 1.0 - py1 };
                    let count = (0.5 * pt * pm * py * 4000.0).round() as usize;
                    t.extend(std::iter::repeat_n(ti, count));
                    m.extend(std::iter::repeat_n(mi, count));
                    y.extend(std::iter::repeat_n(yi, count));
                }
            }
        }
    }
    assert_eq!(t.len(), 4000);
    tabular_data(&[
        ("t", RoleHint::TreatmentCandidate, t),
        ("y", RoleHint::OutcomeCandidate, y),
        ("m", RoleHint::Context, m),
    ])
}

#[test]
fn estimate_frontdoor_functional_matches_enumerated_truth_where_path_product_does_not() {
    let expected = load_expected("frontdoor_functional");
    let truth = expected["true_effect"].as_f64().unwrap();
    let run = |estimator: &str| {
        Study::tabular(frontdoor_interaction_table())
            .graph(frontdoor_dag())
            .query(frontdoor_query())
            .identifier(expected["identifier"].as_str().unwrap().parse::<IdentifierId>().unwrap())
            .estimator(estimator.parse::<EstimatorId>().unwrap())
            .bootstrap_replicates(0)
            .build()
            .unwrap()
            .run(&ExecutionContext::for_tests(43))
            .unwrap()
    };
    let functional = run(expected["estimator"].as_str().unwrap());
    assert_recovers(&functional, &expected);
    assert!(records_assumption(&functional, "frontdoor.functional.saturated_cells"));
    assert!(!records_assumption(&functional, "frontdoor.linear_path_product"));
    assert!(functional.estimate.se_analytic > 0.0);

    let shortcut = run("frontdoor.linear_two_stage");
    let shortcut_limit = expected["linear_path_product_limit"].as_f64().unwrap();
    assert!((shortcut.estimate.ate - shortcut_limit).abs() < 1e-3, "{}", shortcut.estimate.ate);
    assert!((shortcut.estimate.ate - truth).abs() > 0.04);
    assert!(records_assumption(&shortcut, "frontdoor.linear_path_product"));
}

/// The claim a front-door result carries follows the estimator that produced the number.
/// The product of coefficients is a different functional of the observed law; it equals the
/// effect only when the latent confounder does not modify the mediator's effect, so that
/// restriction is part of the identification claim (as the Wald restriction is for IV). The
/// plug-in of the front-door functional keeps the nonparametric claim.
#[test]
fn frontdoor_claim_follows_the_executed_estimator() {
    use antecedent_core::{Assumption, AssumptionScope, IdentificationStatus};
    let path_product = |set: &antecedent_core::AssumptionSet| {
        set.entries
            .iter()
            .filter(|r| {
                matches!(&r.assumption, Assumption::ParametricRestriction(p)
                    if p.id.as_ref() == "frontdoor.linear_path_product")
            })
            .map(|r| r.scope.clone())
            .collect::<Vec<_>>()
    };
    for identifier in [IdentifierId::Frontdoor, IdentifierId::Auto] {
        let run = |estimator: EstimatorId| {
            Study::tabular(frontdoor_interaction_table())
                .graph(frontdoor_dag())
                .query(frontdoor_query())
                .identifier(identifier)
                .estimator(estimator)
                .bootstrap_replicates(0)
                .build()
                .unwrap()
                .run(&ExecutionContext::for_tests(47))
                .unwrap()
        };
        let linear = run(EstimatorId::FrontDoorTwoStage);
        assert_eq!(
            linear.identification.status,
            IdentificationStatus::IdentifiedUnderParametricRestrictions,
            "{identifier:?}"
        );
        assert_eq!(
            path_product(&linear.identification.required_assumptions),
            vec![AssumptionScope::Identification]
        );
        // Recorded once: the estimate inherits the claim's record, it does not add a second.
        assert_eq!(
            path_product(&linear.estimate.assumptions),
            vec![AssumptionScope::Identification]
        );

        let functional = run(EstimatorId::FrontDoorFunctional);
        assert_eq!(
            functional.identification.status,
            IdentificationStatus::NonparametricallyIdentified,
            "{identifier:?}"
        );
        assert!(path_product(&functional.identification.required_assumptions).is_empty());
        assert!((functional.estimate.ate - 0.315).abs() < 1e-12);
    }
}

/// Exact population table of a binary chain `T -> M -> Y`, `T -> Y`, `M ~ Bern(.3 + .4T)`,
/// `Y ~ Bern(.05 + .15M + .15T + .5MT)`: a treatment-mediator interaction on the outcome. No
/// latent confounder, so the pure natural indirect effect is nonparametrically identified by
/// the mediation formula and can be enumerated exactly:
/// `E[Y(0,M(1))] - E[Y(0,M(0))] = .155 - .095 = .06`. `mediation.linear` instead reports
/// `total - direct` as a product of population OLS slopes: the `T -> M` slope is `Cov(T,M) /
/// Var(T) = .10 / .25 = .4`, and the partial `M -> Y` slope controlling for `T` (equal to the
/// partial `T -> Y` slope by the symmetric design) solves
/// `[.25 .10; .10 .25] [c1;c2] = [.14;.14]`, giving `c2 = .4`; the reported indirect effect is
/// their product `.4 * .4 = .16`.
fn mediation_interaction_table() -> TabularData {
    let (mut t, mut m, mut y) = (Vec::new(), Vec::new(), Vec::new());
    for ti in [0.0_f64, 1.0] {
        let pm = if ti == 1.0 { 0.7 } else { 0.3 };
        for mi in [0.0_f64, 1.0] {
            let m_count = if mi == 1.0 { pm } else { 1.0 - pm };
            let py = 0.05 + 0.15 * mi + 0.15 * ti + 0.5 * mi * ti;
            for yi in [0.0_f64, 1.0] {
                let py_cell = if yi == 1.0 { py } else { 1.0 - py };
                let count = (0.5 * m_count * py_cell * 4000.0).round() as usize;
                t.extend(std::iter::repeat_n(ti, count));
                m.extend(std::iter::repeat_n(mi, count));
                y.extend(std::iter::repeat_n(yi, count));
            }
        }
    }
    assert_eq!(t.len(), 4000);
    tabular_data(&[
        ("t", RoleHint::TreatmentCandidate, t),
        ("y", RoleHint::OutcomeCandidate, y),
        ("m", RoleHint::Context, m),
    ])
}

/// Variable ids follow `mediation_interaction_table`'s column order: `t=0, y=1, m=2`.
fn mediation_interaction_dag() -> Dag {
    let mut dag = Dag::with_variables(3);
    for (s, t) in [(0, 1), (0, 2), (2, 1)] {
        dag.insert_directed(DenseNodeId::from_raw(s), DenseNodeId::from_raw(t)).unwrap();
    }
    dag
}

/// `mediation.linear`'s indirect estimate is `total - direct`, the total natural indirect
/// effect, while the path-specific identifier certifies the pure natural indirect effect; they
/// coincide only without treatment-mediator interaction. On the interaction fixture above the
/// two truths differ by 0.1, so the plain nonparametric claim the facade used to report was
/// false: the estimator does not evaluate the certified functional. The claim must be
/// downgraded to parametric with the restriction recorded, exactly as `frontdoor.linear_two_stage`
/// is downgraded when it substitutes a coefficient product for the front-door functional.
#[test]
fn mediation_linear_claim_is_restricted_under_interaction() {
    use antecedent_core::{
        Assumption, AssumptionScope, CausalQuery, IdentificationStatus, MediationContrast,
        MediationQuery,
    };
    let restriction = |set: &antecedent_core::AssumptionSet| {
        set.entries
            .iter()
            .filter(|r| {
                matches!(&r.assumption, Assumption::ParametricRestriction(p)
                    if p.id.as_ref() == "mediation.linear_no_interaction")
            })
            .map(|r| r.scope.clone())
            .collect::<Vec<_>>()
    };
    let query = MediationQuery::binary(
        VariableId::from_raw(0),
        VariableId::from_raw(1),
        Arc::from([VariableId::from_raw(2)]),
        MediationContrast::NaturalIndirect,
    );
    let result = Study::tabular(mediation_interaction_table())
        .graph(mediation_interaction_dag())
        .query(CausalQuery::Mediation(query))
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(53))
        .unwrap();

    let true_pure_nie = 0.06;
    let total_minus_direct = 0.16;
    assert!(
        (result.estimate.ate - total_minus_direct).abs() < 1e-6,
        "the estimator still reports total - direct: {}",
        result.estimate.ate
    );
    assert!((result.estimate.ate - true_pure_nie).abs() > 0.05, "{}", result.estimate.ate);

    assert_eq!(
        result.identification.status,
        IdentificationStatus::IdentifiedUnderParametricRestrictions
    );
    assert_eq!(
        restriction(&result.identification.required_assumptions),
        vec![AssumptionScope::Identification]
    );
    assert_eq!(restriction(&result.estimate.assumptions), vec![AssumptionScope::Identification]);
}

fn run_static(
    name: &str,
    data: TabularData,
    graph: Dag,
    query: AverageEffectQuery,
    seed: u64,
) -> antecedent::StudyResult {
    let expected = load_expected(name);
    let analysis = Study::tabular(data)
        .graph(graph)
        .query(query)
        .identifier(expected["identifier"].as_str().unwrap().parse::<IdentifierId>().unwrap())
        .estimator(expected["estimator"].as_str().unwrap().parse::<EstimatorId>().unwrap())
        .bootstrap_replicates(20)
        .build()
        .unwrap();
    let ctx = ExecutionContext::for_tests(seed);
    let result = analysis.run(&ctx).unwrap();
    assert_recovers(&result, &expected);
    result
}

#[test]
fn estimate_propensity_matching_recovers_att() {
    let (data, graph, mut query) = propensity_ipw_scm(1500, 11);
    query = query.with_target_population(antecedent_core::TargetPopulation::Treated);
    run_static("propensity_matching", data, graph, query, 12);
}

#[test]
fn estimate_propensity_stratification_recovers_ate() {
    let (data, graph, query) = propensity_ipw_scm(1500, 13);
    run_static("propensity_stratification", data, graph, query, 14);
}

#[test]
fn estimate_distance_matching_recovers_att() {
    let (data, graph, mut query) = propensity_ipw_scm(1500, 15);
    query = query.with_target_population(antecedent_core::TargetPopulation::Treated);
    run_static("distance_matching", data, graph, query, 16);
}

#[test]
fn estimate_aipw_recovers_ate() {
    let (data, graph, query) = propensity_ipw_scm(1500, 17);
    // No SE comparison here: see `SE_NOT_COMPARABLE`.
    run_static("aipw", data, graph, query, 18);
}

/// Heterogeneous effect `τ(Z) = 2 + Z` with `P(T=1|Z) = σ(0.8 Z)`, so
/// ATE = 2, ATT = 2 + E[Z|T=1] ≈ 2.350, ATC = 2 + E[Z|T=0] ≈ 1.650.
/// IPW must be scored against the estimand's own target, not against ATE for all three.
fn heterogeneous_effect_scm(n: usize, seed: u64) -> (TabularData, Dag) {
    let mut rng = ExecutionContext::for_tests(seed).rng.stream_for(StreamDomain::Test, 0x5057_u64);
    let mut z = vec![0.0; n];
    let mut t = vec![0.0; n];
    let mut y = vec![0.0; n];
    for i in 0..n {
        let zi = standard_normal(&mut rng);
        let p = 1.0 / (1.0 + (-0.8 * zi).exp());
        let ti = if rng.next_f64() < p { 1.0 } else { 0.0 };
        y[i] = 2.0 * ti + ti * zi + zi + 0.6 * standard_normal(&mut rng);
        z[i] = zi;
        t[i] = ti;
    }
    let data = tabular_data(&[
        ("t", RoleHint::TreatmentCandidate, t),
        ("y", RoleHint::OutcomeCandidate, y),
        ("z", RoleHint::Context, z),
    ]);
    let mut dag = Dag::with_variables(3);
    dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
    dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    (data, dag)
}

/// `E[Z | T = arm]` for `Z ~ N(0,1)`, `P(T=1|Z) = σ(0.8 Z)`, by quadrature.
fn heterogeneous_arm_mean_z(treated: bool) -> f64 {
    let (mut num, mut den) = (0.0, 0.0);
    for k in 0..18_000 {
        let z = -9.0 + f64::from(k) * 1e-3;
        let p = 1.0 / (1.0 + (-0.8 * z).exp());
        let w = (-0.5 * z * z).exp() * if treated { p } else { 1.0 - p };
        num += w * z;
        den += w;
    }
    num / den
}

#[test]
fn estimate_ipw_scores_heterogeneous_att_atc_against_own_targets() {
    let true_ate = 2.0;
    let true_att = 2.0 + heterogeneous_arm_mean_z(true);
    let true_atc = 2.0 + heterogeneous_arm_mean_z(false);
    // Gap must be large enough that scoring ATT/ATC against ATE fails.
    assert!((true_att - true_ate).abs() > 0.25);
    assert!((true_atc - true_ate).abs() > 0.25);
    assert!((true_att - true_atc).abs() > 0.5);

    let (data, graph) = heterogeneous_effect_scm(8_000, 71);
    let base = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let run = |population: TargetPopulation, seed: u64| {
        let query = base.clone().with_target_population(population);
        Study::tabular(data.clone())
            .graph(graph.clone())
            .query(query)
            .identifier(IdentifierId::BackdoorAdjustment)
            .estimator(EstimatorId::PropensityWeighting)
            .bootstrap_replicates(0)
            .build()
            .unwrap()
            .run(&ExecutionContext::for_tests(seed))
            .unwrap()
            .estimate
            .ate
    };

    let ate = run(TargetPopulation::AllObserved, 72);
    let att = run(TargetPopulation::Treated, 73);
    let atc = run(TargetPopulation::Untreated, 74);
    let tol = 0.20;
    assert!((ate - true_ate).abs() < tol, "ATE={ate} truth={true_ate}");
    assert!((att - true_att).abs() < tol, "ATT={att} truth={true_att}");
    assert!((atc - true_atc).abs() < tol, "ATC={atc} truth={true_atc}");
    // Scoring the ATT/ATC estimates against the ATE truth must fail.
    assert!((att - true_ate).abs() >= tol, "ATT must not pass an ATE-only band");
    assert!((atc - true_ate).abs() >= tol, "ATC must not pass an ATE-only band");
}

#[test]
fn estimate_efficient_backdoor_ipw_recovers_ate() {
    let (data, graph, query) = propensity_ipw_scm(1500, 19);
    run_static("efficient_backdoor", data, graph, query, 20);
}

#[test]
fn estimate_iv_wald_recovers_structural_effect() {
    let (data, graph, query) = iv_2sls_scm(4000, 21);
    let result = run_static("iv_wald", data, graph, query, 22);
    assert_reference_se(&result, "iv_wald", 4000);
}

/// On `z -> t -> y` the auto identifier lists back-door, IV, and general-ID estimands.
/// A Wald estimate built on the IV one must report the IV claim: parametric status and
/// the exclusion restriction, not the status or assumptions of the strategies it did not use.
#[test]
fn auto_with_wald_reports_the_iv_claim() {
    let (data, graph, query) = iv_2sls_scm(4000, 21);
    let analysis = Study::tabular(data)
        .graph(graph)
        .query(query)
        .identifier(IdentifierId::Auto)
        .estimator(EstimatorId::IvWald)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let result = analysis.run(&ExecutionContext::for_tests(22)).unwrap();
    assert!((result.estimate.ate - 2.0).abs() < 0.5, "ate={}", result.estimate.ate);
    assert_eq!(result.estimand.instruments.as_ref(), &[VariableId::from_raw(2)]);
    assert_eq!(
        result.identification.status,
        antecedent_core::IdentificationStatus::IdentifiedUnderParametricRestrictions
    );
    let exclusion =
        antecedent_core::Assumption::ExclusionRestriction { instrument: VariableId::from_raw(2) };
    assert!(
        result
            .identification
            .required_assumptions
            .entries
            .iter()
            .any(|r| r.assumption == exclusion)
    );
    assert!(result.estimate.assumptions.entries.iter().any(|r| r.assumption == exclusion));
}

/// Binary outcome logistic SCM: `Y ~ Bern(sigmoid(-0.5 + 1.2 T + 0.8 Z))` with confounded T.
/// Returns the naive treated-minus-control mean alongside the table (confounded by `Z`).
fn glm_binary_scm(n: usize, seed: u64) -> (TabularData, Dag, AverageEffectQuery, f64) {
    let mut rng = ExecutionContext::for_tests(seed).rng.stream_for(StreamDomain::Test, 0x5054_u64);
    let mut z = vec![0.0; n];
    let mut t = vec![0.0; n];
    let mut y = vec![0.0; n];
    let (mut sum1, mut n1, mut sum0, mut n0) = (0.0, 0.0, 0.0, 0.0);
    for i in 0..n {
        let zi = standard_normal(&mut rng);
        let logit_t = -0.3 + 0.8 * zi;
        let pt = 1.0 / (1.0 + (-logit_t).exp());
        let ti = if rng.next_f64() < pt { 1.0 } else { 0.0 };
        let logit_y = -0.5 + 1.2 * ti + 0.8 * zi;
        let py = 1.0 / (1.0 + (-logit_y).exp());
        let yi = if rng.next_f64() < py { 1.0 } else { 0.0 };
        z[i] = zi;
        t[i] = ti;
        y[i] = yi;
        if ti > 0.5 {
            sum1 += yi;
            n1 += 1.0;
        } else {
            sum0 += yi;
            n0 += 1.0;
        }
    }
    let unadjusted = sum1 / n1 - sum0 / n0;
    let data = tabular_data(&[
        ("t", RoleHint::TreatmentCandidate, t),
        ("y", RoleHint::OutcomeCandidate, y),
        ("z", RoleHint::Context, z),
    ]);
    let mut dag = Dag::with_variables(3);
    dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
    dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    (data, dag, query, unadjusted)
}

#[test]
fn estimate_glm_adjustment_recovers_positive_ate() {
    // Monte Carlo: logistic g-comp ATE is positive and typically ~0.2–0.3 under this SCM.
    let expected = load_expected("glm_adjustment");
    let truth = expected["true_effect"].as_f64().unwrap();
    let tol = expected["tolerance"].as_f64().unwrap();
    let (data, graph, query, unadjusted) = glm_binary_scm(2000, 23);
    assert!(
        (unadjusted - truth).abs() >= tol,
        "band must exclude the unadjusted contrast: unadjusted={unadjusted} truth={truth} tol={tol}"
    );
    let analysis = Study::tabular(data)
        .graph(graph)
        .query(query)
        .identifier(expected["identifier"].as_str().unwrap().parse::<IdentifierId>().unwrap())
        .estimator(expected["estimator"].as_str().unwrap().parse::<EstimatorId>().unwrap())
        .bootstrap_replicates(20)
        .build()
        .unwrap();
    let ctx = ExecutionContext::for_tests(24);
    let result = analysis.run(&ctx).unwrap();
    assert!(result.estimate.ate > 0.05, "ate={}", result.estimate.ate);
    assert!(
        (result.estimate.ate - truth).abs() < tol,
        "ate={} truth={truth} tol={tol}",
        result.estimate.ate
    );
}

/// The sharp design as a graph: `r -> t -> y` and `r -> y` (ids: t = 0, y = 1, r = 2).
/// The running variable is the treatment's only cause.
fn rd_dag() -> Dag {
    let mut dag = Dag::with_variables(3);
    dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
    dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    dag
}

fn rd_study(data: TabularData, graph: Dag, bandwidth: f64) -> Study {
    Study::tabular(data)
        .graph(graph)
        .query(AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1)))
        .identifier(IdentifierId::RdSharp)
        .estimator(EstimatorId::RdSharp)
        .rd_config(VariableId::from_raw(2), 0.0, bandwidth)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
}

/// `R` has density `2(r + 1)/9` on `[-1, 2]`, `T = 1{R >= 0}`, baseline
/// `1 + 0.5r + 0.8r^2 + r^3`, effect `tau(r) = 2 + 6r`. Closed forms: effect at the cutoff
/// `tau(0) = 2`; average over the `h = 0.4` window `2 + 6 h^2/3 = 2.32`; population
/// average `2 + 6 E[R] = 8`.
fn rd_heterogeneous_scm(n: usize, seed: u64) -> TabularData {
    let mut rng = ExecutionContext::for_tests(seed).rng.stream_for(StreamDomain::Test, 0x5056_u64);
    let (mut r, mut t, mut y) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        let ri = 3.0 * rng.next_f64().sqrt() - 1.0;
        let ti = if ri >= 0.0 { 1.0 } else { 0.0 };
        r[i] = ri;
        t[i] = ti;
        y[i] = 1.0
            + 0.5 * ri
            + 0.8 * ri * ri
            + ri * ri * ri
            + ti * (2.0 + 6.0 * ri)
            + 0.1 * standard_normal(&mut rng);
    }
    tabular_data(&[
        ("t", RoleHint::TreatmentCandidate, t),
        ("y", RoleHint::OutcomeCandidate, y),
        ("r", RoleHint::Context, r),
    ])
}

/// A population-wide `AverageEffect` run through `rd.sharp` is answered with the effect at
/// the cutoff and labelled as that: the three candidate targets are 2, 2.32 and 8 here, the
/// estimate is the first, and the result, its contract and its assumptions all say so.
#[test]
fn rd_sharp_reports_the_cutoff_effect_and_labels_it() {
    let study = rd_study(rd_heterogeneous_scm(60_000, 31), rd_dag(), 0.4);
    let result = study.run(&ExecutionContext::for_tests(32)).unwrap();
    // Smoothing bias of the cubic baseline term is about -0.4 h^3 = -0.026.
    assert!((result.estimate.ate - 2.0).abs() < 0.06, "jump={}", result.estimate.ate);
    assert!((result.estimate.ate - 2.32).abs() > 0.25, "jump={}", result.estimate.ate);
    assert!((result.estimate.ate - 8.0).abs() > 5.0, "jump={}", result.estimate.ate);

    let local = antecedent_core::TargetPopulation::local_at_cutoff(VariableId::from_raw(2), 0.0);
    let identified = result.identification.average_effect().expect("average effect");
    assert_eq!(identified.target_population, local);
    // The contract, read before any execution, already names the cutoff population, and
    // that label is what a coverage record has to match.
    assert_eq!(study.inspect().unwrap().functional.as_ref(), "local_at_cutoff.mean");

    assert!(
        result.diagnostics.iter().any(|d| d.code.as_ref() == "identify.rd.local_estimand"
            && d.message.contains("does not identify the population average effect")),
        "{:?}",
        result.diagnostics
    );
    let status = |id: &str| {
        result.estimate.assumptions.entries.iter().find_map(|r| match &r.assumption {
            antecedent_core::Assumption::Custom { id: i, .. } if i.as_ref() == id => Some(r.status),
            _ => None,
        })
    };
    assert_eq!(status("rd.continuity"), Some(antecedent_core::AssumptionStatus::Untestable));
    assert_eq!(status("rd.no_manipulation"), Some(antecedent_core::AssumptionStatus::Declared));
    // Checked against the treatment column on every row.
    assert_eq!(status("rd.sharp_assignment"), Some(antecedent_core::AssumptionStatus::Supported));
}

/// A graph that does not make the running variable the treatment's only cause contradicts
/// a sharp design, and a design is never assumed in its place.
#[test]
fn rd_sharp_refuses_a_graph_that_contradicts_the_design() {
    let (data, _) = rd_scm(500, 27);
    let err = rd_study(data, Dag::with_variables(3), 1.5)
        .run(&ExecutionContext::for_tests(28))
        .unwrap_err();
    assert_eq!(err.reason_code(), Some("effect_not_identified"), "{err}");
    assert!(err.to_string().contains("no edge from the running variable"), "{err}");
}

/// Imperfect compliance: `P(T=1 | R>=0) = 0.75`, `P(T=1 | R<0) = 0.25`, effect of T = 3.
/// The outcome jump (1.5) is an intent-to-treat contrast; the treatment column shows the
/// rule is not sharp, so nothing is reported under the treatment's name.
#[test]
fn rd_sharp_refuses_fuzzy_assignment() {
    let n = 4000;
    let mut rng = ExecutionContext::for_tests(29).rng.stream_for(StreamDomain::Test, 0x5057_u64);
    let (mut r, mut t, mut y) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        let ri = 2.0 * rng.next_f64() - 1.0;
        let p = if ri >= 0.0 { 0.75 } else { 0.25 };
        let ti = if rng.next_f64() < p { 1.0 } else { 0.0 };
        r[i] = ri;
        t[i] = ti;
        y[i] = 1.0 + 0.5 * ri + 3.0 * ti + 0.1 * standard_normal(&mut rng);
    }
    let data = tabular_data(&[
        ("t", RoleHint::TreatmentCandidate, t),
        ("y", RoleHint::OutcomeCandidate, y),
        ("r", RoleHint::Context, r),
    ]);
    let err = rd_study(data, rd_dag(), 1.0).run(&ExecutionContext::for_tests(30)).unwrap_err();
    assert_eq!(err.reason_code(), Some("rd_assignment_not_sharp"), "{err}");
}

/// Sharp RD: running variable R, T = 1{R >= 0}, Y = 3T + 0.5 R + noise.
fn rd_scm(n: usize, seed: u64) -> (TabularData, AverageEffectQuery) {
    let mut rng = ExecutionContext::for_tests(seed).rng.stream_for(StreamDomain::Test, 0x5055_u64);
    let mut r = vec![0.0; n];
    let mut t = vec![0.0; n];
    let mut y = vec![0.0; n];
    for i in 0..n {
        let ri = standard_normal(&mut rng);
        let ti = if ri >= 0.0 { 1.0 } else { 0.0 };
        let yi = 3.0 * ti + 0.5 * ri + 0.2 * standard_normal(&mut rng);
        r[i] = ri;
        t[i] = ti;
        y[i] = yi;
    }
    let data = tabular_data(&[
        ("t", RoleHint::TreatmentCandidate, t),
        ("y", RoleHint::OutcomeCandidate, y),
        ("r", RoleHint::Context, r),
    ]);
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    (data, query)
}

#[test]
fn estimate_rd_sharp_recovers_jump() {
    let expected = load_expected("rd_sharp");
    let (data, query) = rd_scm(3000, 25);
    let graph = rd_dag();
    let analysis = Study::tabular(data.clone())
        .graph(graph)
        .query(query)
        .identifier(IdentifierId::RdSharp)
        .estimator(EstimatorId::RdSharp)
        .rd_config(VariableId::from_raw(2), 0.0, 1.5)
        .bootstrap_replicates(20)
        .build()
        .unwrap();
    let ctx = ExecutionContext::for_tests(26);
    let result = analysis.run(&ctx).unwrap();
    assert_recovers(&result, &expected);

    // Sharp RD is the documented identify-per-click exception on the prepared
    // handle: prepare() stores no identification cache for it.
    let prepared = analysis.prepare(&ctx).unwrap();
    let click = prepared.estimate(&data, &ctx).unwrap();
    assert!(
        !click.diagnostics.iter().any(|d| d.code.as_ref() == "exec.identify.cached"),
        "sharp RD must not claim identification reuse"
    );
    assert_recovers(&click, &expected);

    // The default analytic SE is HC1; rd_design threads an explicit
    // homoskedastic opt-in, which changes the SE (not the jump) and swaps the
    // robust-SE assumption for the constant-variance one.
    let classical = Study::tabular(data)
        .graph(rd_dag())
        .query(rd_scm(3000, 25).1)
        .identifier(IdentifierId::RdSharp)
        .estimator(EstimatorId::RdSharp)
        .rd_design(
            antecedent::RdConfig::new(VariableId::from_raw(2), 0.0, 1.5)
                .with_se_kind(antecedent_estimate::AnalyticSeKind::Homoskedastic),
        )
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap();
    let declares = |r: &antecedent::StudyResult, id: &str| {
        r.estimate.assumptions.entries.iter().any(|a| {
            matches!(&a.assumption, antecedent_core::Assumption::ParametricRestriction(p)
                if p.id.as_ref() == id)
        })
    };
    assert_eq!(classical.estimate.ate.to_bits(), result.estimate.ate.to_bits());
    assert!(result.estimate.se_analytic.is_finite() && classical.estimate.se_analytic.is_finite());
    assert!((classical.estimate.se_analytic - result.estimate.se_analytic).abs() > 1e-9);
    assert!(declares(&result, "rd.sharp.conventional_robust_se"));
    assert!(!declares(&result, "rd.sharp.homoskedastic_se"));
    assert!(declares(&classical, "rd.sharp.homoskedastic_se"));
    assert!(!declares(&classical, "rd.sharp.conventional_robust_se"));
}

/// `conformance/estimate/rd_sharp/reference.py` computes the jump and both
/// analytic SEs of a frozen heteroskedastic design from the textbook formulas.
/// The default `rd.sharp` SE must equal its HC1 value and the explicit
/// homoskedastic opt-in its classical value.
#[test]
fn estimate_rd_sharp_analytic_se_matches_reference() {
    let block = load_expected("rd_sharp")["se_reference"].clone();
    let floats = |key: &str| -> Vec<f64> {
        block[key].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect()
    };
    let (running, outcome) = (floats("running"), floats("outcome"));
    let cutoff = block["cutoff"].as_f64().unwrap();
    // The frozen design lists the running variable and the outcome; under a sharp rule the
    // treatment column is determined by them.
    let treatment = running.iter().map(|&r| f64::from(r >= cutoff)).collect();
    let data = tabular_data(&[
        ("t", RoleHint::TreatmentCandidate, treatment),
        ("y", RoleHint::OutcomeCandidate, outcome),
        ("r", RoleHint::Context, running),
    ]);
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let bandwidth = block["bandwidth"].as_f64().unwrap();
    let expected = &block["expected"];
    let rel = block["relative_tolerance"].as_f64().unwrap();
    let close = |got: f64, key: &str| {
        let want = expected[key].as_f64().unwrap();
        assert!((got - want).abs() <= rel * want.abs(), "{key}: {got} vs reference {want}");
    };
    let ctx = ExecutionContext::for_tests(3);
    let run = |config: antecedent::RdConfig| {
        Study::tabular(data.clone())
            .graph(rd_dag())
            .query(query.clone())
            .identifier(IdentifierId::RdSharp)
            .estimator(EstimatorId::RdSharp)
            .rd_design(config)
            .bootstrap_replicates(0)
            .build()
            .unwrap()
            .run(&ctx)
            .unwrap()
    };
    let config = antecedent::RdConfig::new(VariableId::from_raw(2), cutoff, bandwidth);
    let default = run(config);
    close(default.estimate.ate, "jump");
    close(default.estimate.se_analytic, "se_hc1");
    let classical = run(config.with_se_kind(antecedent_estimate::AnalyticSeKind::Homoskedastic));
    close(classical.estimate.ate, "jump");
    close(classical.estimate.se_analytic, "se_homoskedastic");
}
