//! The compiled observation-recovery and exposure-measurement families against the direct
//! kernel calls they replace.
//!
//! Each compiled handle passes its unit's support gate and then makes the very call the direct
//! path makes, so every point, standard error and interval bound below agrees bit for bit, and
//! the calibration basis and coverage status the sampled recovery carries are the same. The
//! fixtures are the ones the kernels' own tests use: the enumerated binary missingness SCM of
//! the sampled recovery (`antecedent-estimate/tests/support/recovery_scm.rs`, included by path,
//! with the pattern law and seeded rows of `recovery_sampled.rs` and
//! `sampled_recovery_artifact.rs`), the frozen exact-rational oracles of
//! `recovery_chain_observation_reference.rs` and `measurement_exposure_reference.rs` (their JSON
//! read in place), and for the measured-exposure imputation a planted misclassified and a soft
//! exposure built like `tests/common/measured_exposure_dgp.rs`.
//!
//! Seeds: both paths take the caller's `ExecutionContext`, and every test hands both the same
//! `ExecutionContext::for_tests(seed)`; the bootstrap, imputation and latent-class seeds are
//! fields of the declared inputs, identical on both sides. So both paths draw the same numbers.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "the bits are compared exactly; fixture counts and indices are small"
)]

#[path = "../../antecedent-estimate/tests/support/recovery_scm.rs"]
mod recovery_scm;

use std::collections::BTreeMap;
use std::sync::Arc;

use antecedent::Study;
use antecedent::analysis::kernel_programs::chain_recovery::CompiledChainObservationRecovery;
use antecedent::analysis::kernel_programs::exposure_measurement::{
    DetectionColumns, RepeatCodings, SoftExposureColumns, SoftExposureMethod,
};
use antecedent::analysis::kernel_programs::measured_exposure_effect::ImputationColumns;
use antecedent::planner::PlanView;
use antecedent_core::{ExecutionContext, VariableId};
use antecedent_estimate::measurement::{
    DeclaredEstimate, ExposureErrorDeclaration, ExposureMeasurement, ImputationData, RatePair,
    SoftExposureCombination, correct_detection_dependent_prevalence,
    correct_exposure_misclassification, correct_soft_exposure_effect, fit_repeat_codings,
    impute_exposure, pool_measured_effect,
};
use antecedent_estimate::{
    ChainPatternLaw, ObservationPattern, ObservationRow, SampledObservationInput,
    SampledRecoveryConfig, SampledRecoveryResult, derive_sampled_recovery,
    estimate_sampled_recovery, evaluate_chain_recovery,
};
use antecedent_graph::{Admg, DenseNodeId, NodeRef};
use antecedent_identify::{
    ChainPartial, ChainRecoveryDecision, ChainRecoveryPlan, ChainRecoveryQuery,
    RecoveredEffectQuery, RecoveryLimits, decide_chain_recovery,
};
use antecedent_io::sampled_recovery_artifact::{
    SampledRecoveryArtifactInput, SampledRecoveryArtifactWire, SampledRecoveryConsumeLimits,
    SampledRecoveryExpectation,
};
use antecedent_stats::measurement_error::{
    BinaryErrorRates, DetectionAnchor, DetectionData, DetectionDeclaration, DetectionDirection,
    DetectionFunction, ExposureErrorStructure, ExposureOutcomeCounts, FalsePositiveRate,
    LatentClassConstraints, LatentClassData, LatentClassSettings, SoftExposureData,
    exposure_rates_from_validation,
};
use recovery_scm::{MModel, Rng, SNAPSHOT, v};
use serde_json::Value;

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(1)
}

fn same(direct: f64, compiled: f64, what: &str) {
    assert_eq!(direct.to_bits(), compiled.to_bits(), "{what}: {direct} vs {compiled}");
}

fn same_all(direct: &[f64], compiled: &[f64], what: &str) {
    assert_eq!(direct.len(), compiled.len(), "{what}: lengths");
    for (d, c) in direct.iter().zip(compiled) {
        same(*d, *c, what);
    }
}

fn assert_unit(view: &PlanView, unit: &str) {
    let support = view.unit_support().expect("the route is bound to its capability record");
    assert_eq!(support.unit(), unit);
    assert!(support.supported(), "{unit} must be supported");
    assert!(support.missing_evidence().is_empty());
}

// ---------------------------------------------------------------------------
// Sampled observation recovery (recovery_sampled.rs, sampled_recovery_artifact.rs)
// ---------------------------------------------------------------------------

// Nodes: X0 = 0, X1 = 1, O = 2, R0 = 3, R1 = 4 (proxies 5, 6).
const EDGES: [(u32, u32); 3] = [(0, 1), (2, 0), (2, 1)];
const P_O: f64 = 0.4;
const P_X0: [f64; 2] = [0.3, 0.7];
const P_X1: [f64; 4] = [0.2, 0.5, 0.6, 0.8];
const P_R0: [f64; 2] = [0.5, 0.8];
const P_R1: [f64; 2] = [0.6, 0.9];

fn bern(p1: f64, level: u8) -> f64 {
    if level == 1 { p1 } else { 1.0 - p1 }
}

fn model() -> MModel {
    let mut model = MModel::new(2, 1, &EDGES, &[vec![2], vec![0]], 5).unwrap();
    model.mechanisms.insert(2, vec![P_O]);
    model.mechanisms.insert(0, P_X0.to_vec());
    model.mechanisms.insert(1, P_X1.to_vec());
    model.mechanisms.insert(3, P_R0.to_vec());
    model.mechanisms.insert(4, P_R1.to_vec());
    model
}

fn pattern_probabilities() -> BTreeMap<ObservationPattern, f64> {
    let mut out = BTreeMap::new();
    for o in 0..2u8 {
        for x0 in 0..2u8 {
            for x1 in 0..2u8 {
                for r0 in 0..2u8 {
                    for r1 in 0..2u8 {
                        let p = bern(P_O, o)
                            * bern(P_X0[usize::from(o)], x0)
                            * bern(P_X1[usize::from(2 * x0 + o)], x1)
                            * bern(P_R0[usize::from(o)], r0)
                            * bern(P_R1[usize::from(x0)], r1);
                        let pattern = ObservationPattern {
                            responses: r0 | (r1 << 1),
                            proxies: (r0 & x0) | ((r1 & x1) << 1),
                            fully: o,
                        };
                        *out.entry(pattern).or_insert(0.0) += p;
                    }
                }
            }
        }
    }
    out
}

fn seeded_rows(n: usize, seed: u64) -> Vec<ObservationRow> {
    let table: Vec<(ObservationPattern, f64)> = pattern_probabilities().into_iter().collect();
    let mut rng = Rng::new(seed);
    (0..n)
        .map(|i| {
            let u = (rng.next() >> 11) as f64 / (1u64 << 53) as f64;
            let mut acc = 0.0;
            let mut chosen = table[table.len() - 1].0;
            for (pattern, p) in &table {
                acc += p;
                if u < acc {
                    chosen = *pattern;
                    break;
                }
            }
            ObservationRow { id: i as u64, pattern: chosen }
        })
        .collect()
}

fn effect_query(model: &MModel) -> RecoveredEffectQuery {
    let mut graph = Admg::empty();
    for n in 0..model.k + model.m {
        graph.add_node(NodeRef::Static(v(n))).unwrap();
    }
    for (a, b) in EDGES {
        graph.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    RecoveredEffectQuery { graph, outcomes: Arc::from([v(1)]), treatments: Arc::from([v(0)]) }
}

fn assert_same_sampled(direct: &SampledRecoveryResult, compiled: &SampledRecoveryResult) {
    same(direct.effect, compiled.effect, "effect");
    same(direct.effect_standard_error, compiled.effect_standard_error, "standard error");
    same(direct.interval.level, compiled.interval.level, "level");
    same(direct.interval.lower, compiled.interval.lower, "lower");
    same(direct.interval.upper, compiled.interval.upper, "upper");
    same_all(&direct.replicate_effects, &compiled.replicate_effects, "replicate effects");
    same_all(
        &direct.recovered_cell_covariance,
        &compiled.recovered_cell_covariance,
        "recovered covariance",
    );
    assert_eq!(direct.failed_replicates, compiled.failed_replicates);
    assert_eq!(direct.receipt, compiled.receipt);
    assert_eq!(direct.interval.calibration, compiled.interval.calibration);
    assert_eq!(direct.calibration_basis(), compiled.calibration_basis());
}

#[test]
fn compiled_sampled_recovery_equals_the_direct_estimate_and_its_measured_replay() {
    let model = model();
    let catalog = model.catalog();
    let effect = effect_query(&model);
    let derivation = derive_sampled_recovery(
        &model.graph,
        &model.query(),
        &catalog,
        &effect,
        RecoveryLimits::default(),
        &ctx(),
    )
    .unwrap();
    let data = SampledObservationInput {
        snapshot_id: SNAPSHOT.to_string(),
        rows: seeded_rows(2000, 8123),
    };
    let config = SampledRecoveryConfig::bca(7142);

    let direct = estimate_sampled_recovery(&derivation, &data, &config, &ctx()).unwrap();
    let compiled = Study::compile_sampled_observation_recovery(
        (*derivation).clone(),
        data.clone(),
        config,
        &ctx(),
    )
    .unwrap();
    assert_unit(&compiled.plan_view(), "recovery.sampled_observation");
    let twin = compiled.estimate(&ctx()).unwrap();
    assert_same_sampled(&direct, &twin);

    // The measured replay of the source the direct estimate exports, with the weighted row
    // binding of `sampled_recovery_artifact.rs`: the replay reports its interval with
    // calibration `not_measured`, independently of the current coverage records.
    let mut weighted = catalog.clone();
    Arc::make_mut(&mut weighted.bindings)[0].weights =
        Some(antecedent_core::SupportedWeights { snapshot_identity: Arc::from(SNAPSHOT) });
    let wire = SampledRecoveryArtifactWire::checked(&SampledRecoveryArtifactInput {
        graph: &model.graph,
        effect: &effect,
        derivation: &derivation,
        catalog: &weighted,
        input: &data,
        result: &direct,
        variable_names: &[],
    })
    .unwrap();
    let bytes = wire.export().unwrap();
    let expected = SampledRecoveryExpectation {
        premises_digest: Some(wire.premises_digest.clone()),
        data_digest: Some(wire.data_digest.clone()),
    };
    let measured = SampledRecoveryArtifactWire::measure(
        &bytes,
        &expected,
        SampledRecoveryConsumeLimits::default(),
        &ctx(),
    )
    .unwrap();
    let compiled = Study::compile_measured_sampled_observation_recovery(
        &bytes,
        expected,
        SampledRecoveryConsumeLimits::default(),
        &ctx(),
    )
    .unwrap();
    assert_unit(&compiled.plan_view(), "recovery.sampled_observation");
    let twin = compiled.measure(&ctx()).unwrap();
    assert_eq!(measured.report(), twin.report());
    assert_eq!(measured.export(), twin.export());
    assert_eq!(measured.report().calibration, twin.report().calibration);
    assert_eq!(twin.report().calibration, "not_measured");
    for (d, c) in measured.report().scalars.iter().zip(&twin.report().scalars) {
        same(d.point, c.point, "measured point");
        same(d.lower, c.lower, "measured lower");
        same(d.upper, c.upper, "measured upper");
        assert_eq!(d.calibration.status.as_str(), c.calibration.status.as_str());
    }
}

// ---------------------------------------------------------------------------
// Ordered-response recovery (recovery_chain_observation_reference.rs)
// ---------------------------------------------------------------------------

const IDS: [u32; 6] = [0, 1, 2, 3, 4, 5];

fn chain_reference() -> Value {
    serde_json::from_str(include_str!(
        "../../antecedent-estimate/tests/oracle_data/recovery_chain_observation_reference.json"
    ))
    .unwrap()
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn chain_graph(edges: &[(u32, u32)]) -> Admg {
    let mut graph = Admg::empty();
    let mut dense: BTreeMap<u32, DenseNodeId> = BTreeMap::new();
    for raw in IDS {
        dense.insert(raw, graph.add_node(NodeRef::Static(VariableId::from_raw(raw))).unwrap());
    }
    for (a, b) in edges {
        graph.insert_directed(dense[a], dense[b]).unwrap();
    }
    graph
}

fn chain_edges(head: usize, dependent: bool) -> Vec<(u32, u32)> {
    let [x1, x2, r1, r2, p1, p2] = IDS;
    let mut out = vec![(x1, x2), (x1, p1), (r1, p1), (x2, p2), (r2, p2)];
    let (rh, rt, xh) = if head == 0 { (r1, r2, x1) } else { (r2, r1, x2) };
    out.push((rh, rt));
    if dependent {
        out.push((xh, rt));
    }
    out
}

fn chain_plan(head: usize, dependent: bool) -> ChainRecoveryPlan {
    let v = VariableId::from_raw;
    let query = ChainRecoveryQuery {
        first: ChainPartial { variable: v(0), response: v(2), proxy: v(4) },
        second: ChainPartial { variable: v(1), response: v(3), proxy: v(5) },
    };
    match decide_chain_recovery(&chain_graph(&chain_edges(head, dependent)), &query, &ctx())
        .unwrap()
    {
        ChainRecoveryDecision::Recovered(plan) => *plan,
        ChainRecoveryDecision::NonRecoverable(_) => panic!("expected a recovery plan"),
    }
}

fn chain_law(case: &Value) -> ChainPatternLaw {
    let both = case["both"].as_array().unwrap();
    let (b0, b1) = (floats(&both[0]), floats(&both[1]));
    let (f, s) = (floats(&case["only_first"]), floats(&case["only_second"]));
    ChainPatternLaw::new(
        [[b0[0], b0[1]], [b1[0], b1[1]]],
        [f[0], f[1]],
        [s[0], s[1]],
        case["neither"].as_f64().unwrap(),
    )
    .unwrap()
}

#[test]
fn compiled_chain_recovery_equals_the_direct_evaluation_on_every_reference_case() {
    let data = chain_reference();
    let cases = data["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 24);
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let head = usize::try_from(case["head"].as_u64().unwrap()).unwrap();
        let plan = chain_plan(head, case["dependent"].as_bool().unwrap());
        let law = chain_law(case);
        let direct = evaluate_chain_recovery(&plan, &law).unwrap();
        let compiled: CompiledChainObservationRecovery =
            Study::compile_chain_observation_recovery(plan, law, &ctx()).unwrap();
        assert_unit(&compiled.plan_view(), "recovery.chain_observation");
        let twin = compiled.evaluate(&ctx()).unwrap();
        for (d, c) in direct.cells().iter().zip(twin.cells()) {
            same_all(d, c, name);
        }
        assert_eq!(direct, twin);
    }
}

// ---------------------------------------------------------------------------
// Exposure-error corrections (measurement_exposure_reference.rs)
// ---------------------------------------------------------------------------

fn measurement_reference() -> Value {
    serde_json::from_str(include_str!(
        "../../antecedent-estimate/tests/oracle_data/measurement_exposure_reference.json"
    ))
    .unwrap()
}

fn rates(se: f64, sp: f64, se_var: f64, sp_var: f64) -> BinaryErrorRates {
    BinaryErrorRates::new(se, sp, se_var, sp_var).unwrap()
}

fn assert_same_estimate(direct: &DeclaredEstimate, compiled: &DeclaredEstimate, what: &str) {
    same(direct.point, compiled.point, what);
    assert_eq!(direct.naive_point.map(f64::to_bits), compiled.naive_point.map(f64::to_bits));
    assert_eq!(
        direct.standard_error.map(f64::to_bits),
        compiled.standard_error.map(f64::to_bits),
        "{what}: standard error"
    );
    match (direct.wald_interval(0.95), compiled.wald_interval(0.95)) {
        (Ok((dl, du)), Ok((cl, cu))) => {
            same(dl, cl, what);
            same(du, cu, what);
        }
        (Err(d), Err(c)) => assert_eq!(d, c),
        _ => panic!("{what}: one path has an interval and the other does not"),
    }
    assert_eq!(direct, compiled);
}

#[test]
fn compiled_exposure_misclassification_equals_the_direct_correction() {
    for case in measurement_reference()["exposure"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let c: Vec<u64> =
            case["counts"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect();
        let counts = ExposureOutcomeCounts {
            exposed_cases: c[0],
            cases: c[1],
            exposed_controls: c[2],
            controls: c[3],
        };
        let r = floats(&case["rates"]);
        let v = floats(&case["rate_variances"]);
        let structure = if case["shared"].as_bool().unwrap() {
            ExposureErrorStructure::Shared(rates(r[0], r[1], v[0], v[1]))
        } else if case.get("validation_cases").is_some() {
            let tuple = |key: &str| {
                let a: Vec<u64> =
                    case[key].as_array().unwrap().iter().map(|x| x.as_u64().unwrap()).collect();
                (a[0], a[1], a[2], a[3])
            };
            exposure_rates_from_validation(tuple("validation_controls"), tuple("validation_cases"))
                .unwrap()
        } else {
            ExposureErrorStructure::ByOutcome {
                cases: rates(r[0], r[1], v[0], v[1]),
                controls: rates(r[2], r[3], v[2], v[3]),
            }
        };
        let direct = correct_exposure_misclassification(&counts, &structure, 0.1, &ctx()).unwrap();
        let compiled =
            Study::compile_exposure_misclassification(counts, structure, 0.1, &ctx()).unwrap();
        assert_unit(&compiled.plan_view(), "measurement.exposure_misclassification");
        let twin = compiled.correct(&ctx()).unwrap();
        same(direct.association.log_odds_ratio, twin.association.log_odds_ratio, name);
        same(direct.association.variance, twin.association.variance, name);
        assert_same_estimate(&direct.estimate, &twin.estimate, name);
        assert_eq!(direct, twin);
    }
    // A refusal is the direct path's refusal.
    let counts = ExposureOutcomeCounts {
        exposed_cases: 480,
        cases: 1000,
        exposed_controls: 180,
        controls: 1000,
    };
    let weak = ExposureErrorStructure::ByOutcome {
        cases: rates(0.9, 0.8, 0.0, 0.0),
        controls: rates(0.55, 0.5, 0.0, 0.0),
    };
    let direct = correct_exposure_misclassification(&counts, &weak, 0.1, &ctx()).unwrap_err();
    let twin = Study::compile_exposure_misclassification(counts, weak, 0.1, &ctx())
        .unwrap()
        .correct(&ctx())
        .unwrap_err();
    assert_eq!(direct, twin);
}

#[test]
fn compiled_soft_exposure_equals_the_direct_effect_for_every_combination() {
    let reference = measurement_reference();
    let soft = &reference["soft"];
    let columns = SoftExposureColumns {
        outcome: floats(&soft["outcome"]),
        probability: floats(&soft["probability"]),
        covariates: floats(&soft["covariate"]),
        covariate_count: 1,
    };
    let draws: Vec<f64> = soft["draws"].as_array().unwrap().iter().flat_map(floats).collect();
    let imputation = &reference["imputation"];
    let imputed_columns = SoftExposureColumns {
        outcome: floats(&imputation["outcome"]),
        probability: floats(&imputation["probability"]),
        covariates: floats(&imputation["covariate"]),
        covariate_count: 1,
    };
    let cases = [
        (columns.clone(), SoftExposureMethod::PosteriorMean),
        (columns.clone(), SoftExposureMethod::SuppliedDraws { draws, draw_count: 3 }),
        (
            imputed_columns,
            SoftExposureMethod::MultipleImputation {
                draw_count: imputation["draw_count"].as_u64().unwrap() as usize,
                seed: imputation["seed"].as_u64().unwrap(),
            },
        ),
    ];
    for (columns, method) in cases {
        let data = SoftExposureData {
            outcome: &columns.outcome,
            probability: &columns.probability,
            covariates: &columns.covariates,
            covariate_count: columns.covariate_count,
        };
        let combination = match &method {
            SoftExposureMethod::PosteriorMean => SoftExposureCombination::PosteriorMean,
            SoftExposureMethod::MultipleImputation { draw_count, seed } => {
                SoftExposureCombination::MultipleImputation { draw_count: *draw_count, seed: *seed }
            }
            SoftExposureMethod::SuppliedDraws { draws, draw_count } => {
                SoftExposureCombination::SuppliedDraws { draws, draw_count: *draw_count }
            }
            _ => unreachable!("the fixture states every method it uses"),
        };
        let direct = correct_soft_exposure_effect(&data, &combination, &ctx()).unwrap();
        let compiled =
            Study::compile_soft_exposure_effect(columns.clone(), method, &ctx()).unwrap();
        assert_unit(&compiled.plan_view(), "measurement.soft_exposure");
        let twin = compiled.correct(&ctx()).unwrap();
        same_all(&direct.coefficients, &twin.coefficients, "coefficients");
        same_all(&direct.covariance, &twin.covariance, "covariance");
        assert_same_estimate(&direct.estimate, &twin.estimate, "soft exposure");
        assert_eq!(direct, twin);
    }
}

fn detection_rows(per: usize, hits: &[u64]) -> (Vec<bool>, Vec<f64>) {
    let mut detected = Vec::new();
    let mut covariate = Vec::new();
    for (c, &h) in hits.iter().enumerate() {
        for k in 0..per {
            detected.push((k as u64) < h);
            covariate.push(c as f64);
        }
    }
    (detected, covariate)
}

#[test]
fn compiled_detection_prevalence_equals_the_direct_point_and_bounds() {
    let reference = measurement_reference();
    let logistic = &reference["logistic"];
    let bounds = &reference["bounds"];
    let hits = |case: &Value| -> Vec<u64> {
        case["hits"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect()
    };
    let point = (
        detection_rows(logistic["per"].as_u64().unwrap() as usize, &hits(logistic)),
        DetectionDeclaration::Full(DetectionFunction::Logistic {
            intercept: logistic["intercept"].as_f64().unwrap(),
            slope: logistic["slope"].as_f64().unwrap(),
            intercept_variance: logistic["intercept_variance"].as_f64().unwrap(),
            covariance: logistic["covariance"].as_f64().unwrap(),
            slope_variance: logistic["slope_variance"].as_f64().unwrap(),
        }),
        FalsePositiveRate {
            rate: logistic["false_positive"].as_f64().unwrap(),
            variance: logistic["false_positive_variance"].as_f64().unwrap(),
        },
        0.05,
    );
    let anchors = bounds["anchors"]
        .as_array()
        .unwrap()
        .iter()
        .map(|a| {
            let a = floats(a);
            DetectionAnchor { covariate: a[0], lower: a[1], upper: a[2] }
        })
        .collect();
    let envelope = (
        detection_rows(bounds["per"].as_u64().unwrap() as usize, &hits(bounds)),
        DetectionDeclaration::Partial { direction: DetectionDirection::Nondecreasing, anchors },
        FalsePositiveRate::NONE,
        0.1,
    );
    for ((detected, covariate), declaration, false_positive, floor) in [point, envelope] {
        let data = DetectionData { detected: &detected, covariate: &covariate };
        let direct = correct_detection_dependent_prevalence(
            &data,
            &declaration,
            &false_positive,
            floor,
            &ctx(),
        )
        .unwrap();
        let compiled = Study::compile_detection_dependent_prevalence(
            DetectionColumns { detected: detected.clone(), covariate: covariate.clone() },
            declaration,
            false_positive,
            floor,
            &ctx(),
        )
        .unwrap();
        assert_unit(&compiled.plan_view(), "measurement.detection_dependent_prevalence");
        let twin = compiled.correct(&ctx()).unwrap();
        match (&direct.estimate, &twin.estimate) {
            (Some(d), Some(c)) => assert_same_estimate(d, c, "detection"),
            (None, None) => {}
            _ => panic!("one path has a point estimate and the other does not"),
        }
        assert_eq!(direct, twin);
    }
}

#[test]
fn compiled_repeat_codings_equal_the_direct_latent_class_fit() {
    for case in measurement_reference()["latent_class"].as_array().unwrap() {
        let k = case["k"].as_u64().unwrap() as usize;
        let c = case["c"].as_u64().unwrap() as usize;
        let j = case["j"].as_u64().unwrap() as usize;
        let counts: Vec<u64> =
            case["counts"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect();
        let mut codings: Vec<u8> = Vec::new();
        let mut units = 0_usize;
        for (pattern, &count) in counts.iter().enumerate() {
            for _ in 0..count {
                for coding in 0..k {
                    let digit = (pattern / c.pow((k - 1 - coding) as u32)) % c;
                    codings.push(digit as u8);
                }
                units += 1;
            }
        }
        let settings =
            LatentClassSettings { tolerance: 1e-14, max_iterations: 50_000, ..Default::default() };
        let data = LatentClassData {
            codings: &codings,
            units,
            coding_count: k,
            categories: c,
            latent_levels: j,
        };
        let direct =
            fit_repeat_codings(&data, &LatentClassConstraints::default(), &settings, &ctx())
                .unwrap();
        let compiled = Study::compile_repeat_codings(
            RepeatCodings {
                codings: codings.clone(),
                units,
                coding_count: k,
                categories: c,
                latent_levels: j,
            },
            LatentClassConstraints::default(),
            settings,
            &ctx(),
        )
        .unwrap();
        assert_unit(&compiled.plan_view(), "measurement.repeat_codings_latent_class");
        let twin = compiled.fit(&ctx()).unwrap();
        same_all(&direct.prevalence, &twin.prevalence, "prevalence");
        same_all(&direct.response, &twin.response, "response");
        same(direct.log_likelihood, twin.log_likelihood, "log likelihood");
        match (&direct.standard_errors, &twin.standard_errors) {
            (Some(d), Some(c)) => {
                same_all(&d.prevalence, &c.prevalence, "prevalence standard errors");
                same_all(&d.response, &c.response, "response standard errors");
            }
            (None, None) => {}
            _ => panic!("one path has standard errors and the other does not"),
        }
        assert_eq!(direct, twin);
    }
}

// ---------------------------------------------------------------------------
// Measured-exposure imputation and pooling (measured_exposure_dgp.rs)
// ---------------------------------------------------------------------------

/// A planted exposure caused by a covariate and an outcome with a known exposure effect, as
/// `tests/common/measured_exposure_dgp.rs` plants them, from a fixed `SplitMix64` stream.
struct Planted {
    z: Vec<f64>,
    x: Vec<f64>,
    y: Vec<f64>,
    uniform: Vec<f64>,
}

fn planted(n: usize, seed: u64) -> Planted {
    let mut state = seed;
    let mut next = move || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    };
    let expit = |t: f64| 1.0 / (1.0 + (-t).exp());
    let z: Vec<f64> = (0..n).map(|_| 2.0 * next() - 1.0).collect();
    let x: Vec<f64> = z.iter().map(|&zi| f64::from(u8::from(next() < expit(0.3 * zi)))).collect();
    let y: Vec<f64> = (0..n).map(|i| 1.0 + 2.0 * x[i] + 0.5 * z[i] + (next() - 0.5)).collect();
    let uniform: Vec<f64> = (0..n).map(|_| next()).collect();
    Planted { z, x, y, uniform }
}

#[test]
fn compiled_imputation_and_pooling_equal_the_direct_draws_and_pooled_effect() {
    let p = planted(200, 0x4D45_0001);
    let (se, sp) = (0.85, 0.9);
    let recorded: Vec<bool> =
        p.x.iter()
            .zip(&p.uniform)
            .map(|(&x, &u)| if x == 1.0 { u < se } else { u >= sp })
            .collect();
    let probability: Vec<f64> =
        p.x.iter().zip(&p.uniform).map(|(&x, &u)| 0.1 + 0.6 * x + 0.2 * u).collect();
    let declarations = [
        ExposureMeasurement::Misclassified {
            recorded,
            error: ExposureErrorDeclaration::Shared(
                RatePair::declared(se, sp, 1e-4, 1e-4).unwrap(),
            ),
            min_youden: 0.1,
        },
        ExposureMeasurement::Soft { probability },
    ];
    for declaration in declarations {
        let data = ImputationData { outcome: &p.y, covariates: &p.z, covariate_count: 1 };
        let ctx = ExecutionContext::for_tests(0x4D45_0001);
        let direct = impute_exposure(&declaration, &data, 8, 11, &ctx).unwrap();
        let compiled = Study::compile_exposure_imputation(
            declaration.clone(),
            ImputationColumns { outcome: p.y.clone(), covariates: p.z.clone(), covariate_count: 1 },
            8,
            11,
            &ctx,
        )
        .unwrap();
        assert_unit(&compiled.plan_view(), "measurement.measured_exposure_effect");
        let twin = compiled.impute(&ctx).unwrap();
        same_all(&direct.draws, &twin.draws, "draws");
        same_all(&direct.probability, &twin.probability, "probability");
        assert_eq!(direct, twin);

        // Pool per-draw differences in means of the completed exposure, with their variances.
        let (points, variances): (Vec<f64>, Vec<f64>) = (0..direct.draw_count)
            .map(|k| {
                let column = &direct.draws[k * direct.rows..(k + 1) * direct.rows];
                let (mut sums, mut counts) = ([0.0; 2], [0.0; 2]);
                for (&exposed, &outcome) in column.iter().zip(&p.y) {
                    let arm = usize::from(exposed >= 0.5);
                    sums[arm] += outcome;
                    counts[arm] += 1.0;
                }
                let point = sums[1] / counts[1] - sums[0] / counts[0];
                (point, 1.0 / counts[1] + 1.0 / counts[0])
            })
            .unzip();
        let naive = Some(points[0] - 0.1);
        let direct = pool_measured_effect(&points, &variances, naive).unwrap();
        let compiled = Study::compile_measured_effect_pool(points, variances, naive, &ctx).unwrap();
        assert_unit(&compiled.plan_view(), "measurement.measured_exposure_effect");
        let twin = compiled.pool(&ctx).unwrap();
        same(direct.pooled.point, twin.pooled.point, "pooled point");
        same(direct.standard_error(), twin.standard_error(), "pooled standard error");
        let ((dl, du), (cl, cu)) = (direct.interval(0.95), twin.interval(0.95));
        same(dl, cl, "pooled lower");
        same(du, cu, "pooled upper");
        assert_eq!(direct, twin);
    }
}
