//! Producer of the committed cyclic family fixtures under `parity/cyclic/`.
//!
//! Each fixture directory holds the request, the exported report, the version-two
//! family record (`family.ant`), the separately retained expected identity and a
//! manifest. `cyclic_independent_consumer.rs` reads those committed bytes and never
//! calls this producer. Inputs are deterministic: simulated data come from a fixed
//! xorshift stream and every draw is seeded in the request.
//!
//! Analytic truths, with independent centred unit-variance disturbances:
//! `A = 0.4 B + U_A`, `B = 0.5 A + U_B` gives a B-on-A association of
//! 1.40625 / 1.8125 (about 0.776), a hard-clamp derivative of B in A of 0.5, and a
//! B response of 0.625 to a unit mean shift of `U_A`. `A = B + C + U_A`, `B = A + U_B`,
//! `C = A + U_C` solves uniquely, but `do(C = c)` leaves `0 = c + U_A + U_B`, so the
//! clamped world is failed mass.
//!
//! Generate or refresh the fixtures with
//! `cargo test -p antecedent --test cyclic_family_fixtures -- --ignored write_`
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::path::{Path, PathBuf};

use antecedent::analysis::Study;
use antecedent_core::ExecutionContext;
use antecedent_estimate::cyclic_dynamic_inference::{
    DynamicBootstrapMethod, DynamicQuantity, OuInferenceProblem,
};
use antecedent_estimate::cyclic_dynamics::OuParameter;
use antecedent_estimate::cyclic_likelihood::{CyclicBootstrapDesign, CyclicFitOptions};
use antecedent_io::cyclic_family_artifact::{
    CertificateDemandWire, CompiledCyclicFamilyProgram, CyclicComponentWire, CyclicFamilyRequest,
    CyclicInterventionWire, CyclicWorldWire, DynamicMethodWire, DynamicProblemWire,
    ExperimentRegimeWire, ExperimentalFitOptionsWire, LinearCyclicWire, RecoveryInferenceWire,
    RegimeClampWire, RegimeSampleWire, ResamplingRuleWire, SamplingDesignWire, UnitColumnWire,
    consume_cyclic_family,
};
use antecedent_model::continuous_cyclic::OrnsteinUhlenbeck;
use antecedent_model::nonlinear_equilibrium::RegimeRow;
use antecedent_test_support::read_fixture_bytes;
use serde_json::json;

const GENERATOR: &str =
    "cargo test -p antecedent --test cyclic_family_fixtures -- --ignored write_";

/// Committed fixture directory names. Each names what the fixture is.
const SPECIFIED_FEEDBACK_PAIR: &str = "specified-feedback-pair";
const OVERLAPPING_LOOPS: &str = "overlapping-loops";
const RECOVERED_ASYMPTOTIC_NORMAL: &str = "recovered-asymptotic-normal";
const RECOVERED_BOOTSTRAP: &str = "recovered-bootstrap";
const RECOVERED_UNRESOLVED_DIRECTION: &str = "recovered-unresolved-direction";
const DYNAMIC_INFERENCE_OU: &str = "dynamic-inference-ou";
const PREVIOUS_RECORD_FORMAT: &str = "previous-record-format";

/// Rows per simulated clamp-experiment regime.
const REGIME_ROWS: usize = 250;

fn fixture_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../parity/cyclic")
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(1)
}

/// Deterministic xorshift normal stream; no library random state.
struct Stream(u64);

impl Stream {
    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 11) as f64 + 0.5) / (1_u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let (u, v) = (self.uniform(), self.uniform());
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
}

fn set(variable: u32, value: f64) -> CyclicInterventionWire {
    CyclicInterventionWire::Set { variable, value }
}

fn world(id: u32, interventions: Vec<CyclicInterventionWire>) -> CyclicWorldWire {
    CyclicWorldWire { world: id, interventions, initial_states: Vec::new() }
}

fn specified(
    variables: Vec<u32>,
    coefficients: Vec<f64>,
    component_id: &str,
    worlds: Vec<CyclicWorldWire>,
    certificate: CertificateDemandWire,
) -> CyclicFamilyRequest {
    let n = variables.len();
    let mut covariance = vec![0.0; n * n];
    for i in 0..n {
        covariance[i * n + i] = 1.0;
    }
    CyclicFamilyRequest::LinearWorlds {
        variables,
        model: LinearCyclicWire {
            coefficients,
            intercept: vec![0.0; n],
            noise_mean: vec![0.0; n],
            noise_covariance: covariance,
        },
        component: CyclicComponentWire {
            component_id: component_id.into(),
            source_id: "specification".into(),
            certifying_evidence: vec!["independent-bench-test".into()],
        },
        worlds,
        certificate,
    }
}

/// `A = 0.4 B + U_A`, `B = 0.5 A + U_B` (A is 0, B is 1): as specified, `do(A = 0)`,
/// `do(A = 1)`, and a unit shift of `U_A`.
fn specified_feedback_pair() -> CyclicFamilyRequest {
    specified(
        vec![0, 1],
        vec![0.0, 0.4, 0.5, 0.0],
        "feedback-pair",
        vec![
            world(0, vec![]),
            world(1, vec![set(0, 0.0)]),
            world(2, vec![set(0, 1.0)]),
            world(3, vec![CyclicInterventionWire::Shift { variable: 0, delta: 1.0 }]),
        ],
        CertificateDemandWire::Required,
    )
}

/// `A = B + C + U_A`, `B = A + U_B`, `C = A + U_C` (A, B, C are 0, 1, 2): as specified
/// and under `do(C = 1)`, reported without a uniqueness certificate.
fn overlapping_loops() -> CyclicFamilyRequest {
    specified(
        vec![0, 1, 2],
        vec![0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        "overlapping-loops",
        vec![world(0, vec![]), world(1, vec![set(2, 1.0)])],
        CertificateDemandWire::ReportUncertified,
    )
}

/// One regime of the feedback pair: passive, A randomized around 2, or B around 3.
fn simulate(clamped: Option<u32>, stream: &mut Stream) -> Vec<Vec<f64>> {
    let mut a = Vec::with_capacity(REGIME_ROWS);
    let mut b = Vec::with_capacity(REGIME_ROWS);
    for _ in 0..REGIME_ROWS {
        let (ua, ub) = (stream.normal(), stream.normal());
        let (x, y) = match clamped {
            None => {
                let x = (ua + 0.4 * ub) / 0.8;
                (x, 0.5 * x + ub)
            }
            Some(0) => {
                let x = 2.0 + 1.5 * stream.normal();
                (x, 0.5 * x + ub)
            }
            Some(_) => {
                let y = 3.0 + 1.5 * stream.normal();
                (0.4 * y + ua, y)
            }
        };
        a.push(x);
        b.push(y);
    }
    vec![a, b]
}

/// Clamp experiments on the feedback pair, recovered under `inference`; worlds 1
/// (as recovered) and 2 (`do(A = 2)`).
fn recovery(designs: &[Option<u32>], inference: RecoveryInferenceWire) -> CyclicFamilyRequest {
    let mut stream = Stream(0x9e37_79b9_7f4a_7c15);
    let mut regimes = Vec::new();
    let mut samples = Vec::new();
    for (index, clamped) in designs.iter().enumerate() {
        let source = u64::try_from(index).unwrap() + 1;
        regimes.push(ExperimentRegimeWire {
            clamps: clamped
                .map(|unit| RegimeClampWire { unit, value: if unit == 0 { 2.0 } else { 3.0 } })
                .into_iter()
                .collect(),
            units: vec![0, 1],
            population: 0,
            dependent: false,
            source,
        });
        samples.push(RegimeSampleWire {
            source,
            columns: simulate(*clamped, &mut stream)
                .into_iter()
                .enumerate()
                .map(|(unit, values)| UnitColumnWire { unit: u32::try_from(unit).unwrap(), values })
                .collect(),
            sampling_units: None,
        });
    }
    CyclicFamilyRequest::ExperimentalRecovery {
        regimes,
        dropped: vec![],
        samples,
        options: ExperimentalFitOptionsWire::default(),
        inference,
        worlds: vec![world(1, vec![]), world(2, vec![set(0, 2.0)])],
        outcome: 1,
        source_id: "clamp-experiments".into(),
        certifying_evidence: vec!["held-out-regime".into()],
    }
}

const COMPLETE: &[Option<u32>] = &[None, Some(0), Some(1)];

fn recovered_asymptotic_normal() -> CyclicFamilyRequest {
    recovery(COMPLETE, RecoveryInferenceWire::AsymptoticNormal { draws: 64, seed: 7 })
}

fn recovered_bootstrap() -> CyclicFamilyRequest {
    recovery(
        COMPLETE,
        RecoveryInferenceWire::Bootstrap {
            replicates: 12,
            seed: 4,
            design: SamplingDesignWire::IndependentSubjectTrajectories,
            rule: ResamplingRuleWire::RowSubjectIid,
            block_length: None,
        },
    )
}

/// No regime clamps B, so the row into A is not covered.
fn recovered_unresolved_direction() -> CyclicFamilyRequest {
    recovery(&[None, Some(0)], RecoveryInferenceWire::AsymptoticNormal { draws: 16, seed: 9 })
}

/// Stationary OU fit of `dx = -(x - 0.5) dt + 2 dW`, with a seeded bootstrap sampling law
/// of its stationary mean.
fn dynamic_inference_ou() -> CyclicFamilyRequest {
    let mut stream = Stream(0x5eed_0f00);
    CyclicFamilyRequest::DynamicInference {
        problem: DynamicProblemWire::Ou(OuInferenceProblem {
            template: OrnsteinUhlenbeck {
                drift: vec![-1.0],
                center: vec![0.0],
                diffusion: vec![4.0],
            },
            parameters: vec![
                OuParameter::Drift { row: 0, column: 0 },
                OuParameter::Center { variable: 0 },
            ],
            rows: (0..300)
                .map(|_| RegimeRow { values: vec![0.5 + stream.normal()], clamps: vec![] })
                .collect(),
            options: CyclicFitOptions::default(),
        }),
        method: DynamicMethodWire::Bootstrap {
            kind: DynamicBootstrapMethod::Nonparametric,
            replicates: 20,
            design: SamplingDesignWire::IndependentSubjectTrajectories,
            resampling: ResamplingRuleWire::RowSubjectIid,
            stratification: CyclicBootstrapDesign::FixedAssignments,
            seed: 11,
        },
        quantities: vec![DynamicQuantity::OuStationaryMean { clamps: vec![], variable: 0 }],
    }
}

/// The request the version-one golden of `antecedent-io` was written from.
fn previous_record_request() -> CyclicFamilyRequest {
    specified(
        vec![10, 11],
        vec![0.0, 0.4, 0.5, 0.0],
        "feedback",
        vec![
            world(7, vec![]),
            world(8, vec![set(10, 2.0)]),
            world(9, vec![CyclicInterventionWire::Shift { variable: 10, delta: 1.0 }]),
        ],
        CertificateDemandWire::Required,
    )
}

fn previous_record_golden() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../antecedent-io/tests/fixtures/cyclic_family_v1.bin")
}

/// Every version-two fixture, with the analytic truths its manifest records.
fn cases() -> Vec<(&'static str, CyclicFamilyRequest, serde_json::Value)> {
    vec![
        (
            SPECIFIED_FEEDBACK_PAIR,
            specified_feedback_pair(),
            json!({
                "association_b_on_a": 1.40625 / 1.8125,
                "hard_clamp_derivative_b_in_a": 0.5,
                "mean_shift_response_b": 0.625,
            }),
        ),
        (OVERLAPPING_LOOPS, overlapping_loops(), json!({ "clamped_world_failed_mass": 1.0 })),
        (
            RECOVERED_ASYMPTOTIC_NORMAL,
            recovered_asymptotic_normal(),
            json!({ "simulated_clamp_response_b": 1.0, "law_meaning": "sampling_law" }),
        ),
        (
            RECOVERED_BOOTSTRAP,
            recovered_bootstrap(),
            json!({ "simulated_clamp_response_b": 1.0, "law_meaning": "sampling_law" }),
        ),
        (
            RECOVERED_UNRESOLVED_DIRECTION,
            recovered_unresolved_direction(),
            json!({ "unresolved_rows": [{ "target": 0, "missing_clamps": [1] }] }),
        ),
        (
            DYNAMIC_INFERENCE_OU,
            dynamic_inference_ou(),
            json!({ "simulated_stationary_mean": 0.5, "law_meaning": "sampling_law" }),
        ),
    ]
}

fn pretty<T: serde::Serialize>(value: &T) -> String {
    serde_json::to_string_pretty(value).unwrap() + "\n"
}

fn write(dir: &Path, name: &str, bytes: impl AsRef<[u8]>) {
    std::fs::write(dir.join(name), bytes).unwrap_or_else(|e| panic!("write {name}: {e}"));
}

#[test]
#[ignore = "writes parity/cyclic; run with `-- --ignored write_`"]
fn write_cyclic_family_fixtures() {
    let ctx = ctx();
    for (name, request, analytic) in cases() {
        let dir = fixture_root().join(name);
        std::fs::create_dir_all(&dir).expect("create the fixture directory");
        let prepared = Study::prepare_cyclic_family(&request, &ctx).expect("prepares");
        let (report, bytes) = prepared.export(&ctx).expect("exports");
        write(&dir, "request.json", pretty(&request));
        write(&dir, "report.json", pretty(&report));
        write(&dir, "family.ant", &bytes);
        write(&dir, "expected-identity.json", pretty(&json!({ "identity": report.identity })));
        write(
            &dir,
            "manifest.json",
            pretty(&json!({
                "family": request.family_name(),
                "generator": GENERATOR,
                "record_format": 2,
                "artifact_blake3": antecedent_core::blake3_hex(&bytes),
                "analytic": analytic,
                "fresh_process_cpu_replay": true,
                "calibration": "separate_unmeasured",
                "attestation": "separate_not_performed",
            })),
        );
    }

    // The version-one record keeps replaying after version two; its identity is
    // retained here, separately from the bytes.
    let bytes = std::fs::read(previous_record_golden()).expect("the antecedent-io golden");
    let identity = CompiledCyclicFamilyProgram::compile(&previous_record_request(), &ctx)
        .unwrap()
        .identity()
        .to_owned();
    let report = consume_cyclic_family(&bytes, &identity, &ctx).expect("the golden replays");
    let dir = fixture_root().join(PREVIOUS_RECORD_FORMAT);
    std::fs::create_dir_all(&dir).expect("create the fixture directory");
    write(&dir, "request.json", pretty(&previous_record_request()));
    write(&dir, "report.json", pretty(&report));
    write(&dir, "family.ant", &bytes);
    write(&dir, "expected-identity.json", pretty(&json!({ "identity": identity })));
    write(
        &dir,
        "manifest.json",
        pretty(&json!({
            "family": previous_record_request().family_name(),
            "generator": GENERATOR,
            "source": "crates/antecedent-io/tests/fixtures/cyclic_family_v1.bin",
            "record_format": 1,
            "artifact_blake3": antecedent_core::blake3_hex(&bytes),
            "fresh_process_cpu_replay": true,
            "calibration": "separate_unmeasured",
            "attestation": "separate_not_performed",
        })),
    );
}

/// The committed requests are the ones this producer writes; a changed generator
/// fails here until the fixtures are regenerated. Nothing is executed.
#[test]
fn committed_requests_match_their_generator() {
    let mut all = cases().into_iter().map(|(name, request, _)| (name, request)).collect::<Vec<_>>();
    all.push((PREVIOUS_RECORD_FORMAT, previous_record_request()));
    for (name, request) in all {
        let dir = fixture_root().join(name);
        let committed =
            read_fixture_bytes(&dir, "request.json", &format!("generate it with `{GENERATOR}`"));
        let committed: CyclicFamilyRequest = serde_json::from_slice(&committed).unwrap();
        assert_eq!(committed, request, "{name}: regenerate with `{GENERATOR}`");
    }
}
