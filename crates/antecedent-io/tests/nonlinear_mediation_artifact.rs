//! Nonlinear-mediation artifact: the exact-grid closed-form truth of the core tests through
//! the container, a re-estimating consumer, resealed-mutation refusals against a retained
//! identity, the premise and bootstrap records, unknown versions, the computation, export and
//! replay budgets of an estimator with no row ceiling, and the unmeasured calibration coordinate
//! with the closed interval status.
//!
//! SCM (identical to `antecedent-estimate/tests/nonlinear_mediation.rs`):
//! `M = alpha*A + gamma*X + e_M`; `Y = b0 + b1*A + c1*M + c2*M^2 + delta*A*M + bx*X` with no
//! outcome noise; `X` uniform on `{-1, 0, 1, 2, 3}` (`E[X] = 1`, `E[X^2] = 3`); the residual
//! levels are scaled so the degrees-of-freedom corrected residual variance equals `sigma2`.
//! Closed form: `NDE = b1 + delta*gamma*E[X]`,
//! `NIE = (c1 + delta)*alpha + c2*(alpha^2 + 2*alpha*gamma*E[X])`, `TE = NDE + NIE`.

use antecedent_core::ExecutionContext;
use antecedent_io::nonlinear_mediation_artifact::{
    MediationConfigWire, MediationDataWire, MediationPremisesWire,
    NONLINEAR_MEDIATION_ARTIFACT_VERSION, NonlinearMediationArtifact,
    NonlinearMediationArtifactError, NonlinearMediationRequestWire, decode_parts, encode_parts,
};

const XS: [f64; 5] = [-1.0, 0.0, 1.0, 2.0, 3.0];
const EX: f64 = 1.0;
const ALPHA: f64 = 0.5;
const GAMMA: f64 = 0.5;
const SIGMA2: f64 = 0.64;
const B0: f64 = 1.0;
const B1: f64 = 0.8;
const C1: f64 = 0.6;
const C2: f64 = 0.4;
const DELTA: f64 = 0.3;
const BX: f64 = 0.7;

fn data() -> MediationDataWire {
    let rows = 2.0 * 5.0 * 4.0;
    let v = SIGMA2 * (rows - 3.0) / rows;
    let lo = 0.4_f64;
    let hi = (2.0 * v - lo * lo).sqrt();
    let levels = [-hi, -lo, lo, hi];
    let mut d = MediationDataWire {
        treatment: vec![],
        mediator: vec![],
        outcome: vec![],
        covariate_names: vec!["x".into()],
        covariates: vec![vec![]],
    };
    for a in [0.0, 1.0] {
        for x in XS {
            for e in levels {
                let m = ALPHA * a + GAMMA * x + e;
                let y = B0 + B1 * a + C1 * m + C2 * m * m + DELTA * a * m + BX * x;
                d.treatment.push(a);
                d.mediator.push(m);
                d.outcome.push(y);
                d.covariates[0].push(x);
            }
        }
    }
    d
}

fn ignorable() -> MediationPremisesWire {
    MediationPremisesWire {
        unmeasured_treatment_outcome_confounding: false,
        unmeasured_treatment_mediator_confounding: false,
        unmeasured_mediator_outcome_confounding: false,
        treatment_induced_mediator_outcome_confounders: vec![],
        cross_world_independence: true,
    }
}

fn config(replicates: u32) -> MediationConfigWire {
    MediationConfigWire {
        estimand: "natural_effects".into(),
        outcome_degree: 2,
        quadrature_nodes: 8,
        integration_tolerance: 1e-6,
        min_arm_count: 10,
        max_support_violation: 0.25,
        bootstrap_replicates: replicates,
        seed: 7,
    }
}

fn request(replicates: u32) -> NonlinearMediationRequestWire {
    NonlinearMediationRequestWire {
        premises: ignorable(),
        config: config(replicates),
        data: data(),
    }
}

fn truth() -> (f64, f64, f64) {
    let nde = B1 + DELTA * GAMMA * EX;
    let nie = (C1 + DELTA) * ALPHA + C2 * (ALPHA * ALPHA + 2.0 * ALPHA * GAMMA * EX);
    (nde, nie, nde + nie)
}

#[path = "common/mod.rs"]
mod common;

fn close(actual: f64, expected: f64, what: &str) {
    common::close(actual, expected, 1e-8, what);
}

fn seal(replicates: u32) -> (NonlinearMediationArtifact, Vec<u8>) {
    let artifact = NonlinearMediationArtifact::seal(&request(replicates)).unwrap();
    let bytes = artifact.to_bytes("mediation-test").unwrap();
    (artifact, bytes)
}

fn refusal_detail(error: &NonlinearMediationArtifactError) -> String {
    error.refusal().expect("a registered refusal").1
}

#[test]
fn mediation_artifact_round_trips_the_closed_form_truth() {
    let (artifact, bytes) = seal(0);
    let consumed =
        NonlinearMediationArtifact::from_bytes(&bytes, Some(artifact.identity())).unwrap();
    let (nde, nie, te) = truth();
    let meta = consumed.meta();
    close(meta.result.natural_direct, nde, "NDE");
    close(meta.result.natural_indirect, nie, "NIE");
    close(meta.result.total, te, "TE");
    close(meta.result.natural_direct + meta.result.natural_indirect, meta.result.total, "identity");
    assert_eq!(meta.version, NONLINEAR_MEDIATION_ARTIFACT_VERSION);
    assert_eq!(meta.estimand, "natural_effects");
    assert_eq!(meta.calibration, "unmeasured");
    assert_eq!(meta.interval_status, "closed_calibration_unmeasured");
    assert_eq!(meta.inference_claim, "point_with_diagnostics");
    assert_eq!(meta.n_rows, 40);
    assert_eq!(consumed.meta(), artifact.meta());
    assert_eq!(consumed.estimate(), artifact.estimate());
    assert!(meta.result.natural_direct_se.is_none(), "no replicates were requested");
}

#[test]
fn mediation_artifact_records_premises_with_declared_and_checked_status() {
    let (artifact, _) = seal(0);
    let records = &artifact.meta().premise_records;
    let status = |name: &str| records.iter().find(|r| r.name == name).map(|r| r.status.as_str());
    assert_eq!(status("cross_world_independence"), Some("declared"));
    assert_eq!(status("no_unmeasured_mediator_outcome_confounding"), Some("declared"));
    assert_eq!(status("linear_gaussian_mediator"), Some("declared"));
    assert_eq!(status("arm_overlap"), Some("checked"));
    assert_eq!(status("integration_error_within_tolerance"), Some("checked"));
    assert!(records.iter().all(|r| r.holds), "a sealed artifact holds every premise");
    let specs = &artifact.meta().model_specs;
    assert_eq!(specs.outcome_degree, 2);
    assert_eq!(specs.quadrature_nodes, 8);
    assert_eq!(specs.covariates, vec!["x".to_string()]);
    assert!(specs.mediator.contains("linear_gaussian"));
}

#[test]
fn mediation_artifact_records_integration_error_and_overlap() {
    let (artifact, _) = seal(0);
    let result = &artifact.meta().result;
    assert_eq!(result.coarse_nodes, 8);
    assert_eq!(result.fine_nodes, 16);
    assert!(result.integration_error < 1e-10, "{}", result.integration_error);
    assert!(result.identity_residual < 1e-10);
    assert_eq!(result.overlap.treated_count, 20);
    assert_eq!(result.overlap.control_count, 20);
    assert!(result.overlap.treated_mediator_min < result.overlap.treated_mediator_max);
    assert!(result.overlap.mediator_support_violation <= 0.25);
    close(result.mediator.coefficients[1], ALPHA, "alpha");
    close(result.mediator.residual_variance, SIGMA2, "sigma2");
}

#[test]
fn mediation_artifact_bootstrap_record_replays_with_closed_interval_status() {
    let (artifact, bytes) = seal(12);
    let boot = &artifact.meta().result.bootstrap;
    assert_eq!(boot.seed, 7);
    assert_eq!(boot.replicates_requested, 12);
    assert_eq!(boot.replicate_ids, (0..12).collect::<Vec<u64>>());
    assert_eq!(u64::from(boot.replicates_succeeded) + boot.failed_replicate_ids.len() as u64, 12);
    assert_eq!(boot.interval_status, "closed_calibration_unmeasured");
    assert!(artifact.meta().result.natural_direct_se.is_some_and(|s| s > 0.0 && s.is_finite()));
    let replay = NonlinearMediationArtifact::from_bytes(&bytes, Some(artifact.identity())).unwrap();
    assert_eq!(replay.meta().result, artifact.meta().result, "the seeded bootstrap replays");
}

#[test]
fn mediation_artifact_refuses_changed_data_even_when_resealed() {
    let (artifact, bytes) = seal(0);
    let (meta, mut numbers) = decode_parts(&bytes).unwrap();
    // Outcome column starts after treatment and mediator: 2 * 40 values of 8 bytes.
    let offset = 2 * 40 * 8;
    let mut word = [0_u8; 8];
    word.copy_from_slice(&numbers[offset..offset + 8]);
    numbers[offset..offset + 8].copy_from_slice(&(f64::from_le_bytes(word) + 1.0).to_le_bytes());
    let tampered = encode_parts(&meta, &numbers, "tampered").unwrap();
    let error = NonlinearMediationArtifact::from_bytes(&tampered, None).unwrap_err();
    assert!(
        matches!(error, NonlinearMediationArtifactError::IdentityMismatch { field: "data" }),
        "{error:?}"
    );

    // Fully resealed: the producer re-seals changed data consistently. Only the identity the
    // consumer retained independently catches it.
    let mut changed = request(0);
    changed.data.outcome[0] += 1.0;
    let resealed = NonlinearMediationArtifact::seal(&changed).unwrap();
    let resealed_bytes = resealed.to_bytes("resealed").unwrap();
    assert!(NonlinearMediationArtifact::from_bytes(&resealed_bytes, None).is_ok());
    let error = NonlinearMediationArtifact::from_bytes(&resealed_bytes, Some(artifact.identity()))
        .unwrap_err();
    assert!(matches!(error, NonlinearMediationArtifactError::IdentityMismatch { .. }), "{error:?}");
    assert_eq!(refusal_detail(&error), "nonlinear_mediation.wrong_contract");
}

#[test]
fn mediation_artifact_refuses_a_changed_stored_result() {
    let (_, bytes) = seal(0);
    let (mut meta, numbers) = decode_parts(&bytes).unwrap();
    meta.result.natural_direct += 0.1;
    let tampered = encode_parts(&meta, &numbers, "tampered").unwrap();
    let error = NonlinearMediationArtifact::from_bytes(&tampered, None).unwrap_err();
    assert!(matches!(error, NonlinearMediationArtifactError::ResultMismatch(_)), "{error:?}");
    assert_eq!(refusal_detail(&error), "nonlinear_mediation.report_replay_mismatch");
}

#[test]
fn mediation_artifact_refuses_changed_premises_and_config() {
    let (_, bytes) = seal(0);
    let (mut meta, numbers) = decode_parts(&bytes).unwrap();
    meta.premises.unmeasured_mediator_outcome_confounding = true;
    let tampered = encode_parts(&meta, &numbers, "tampered").unwrap();
    let error = NonlinearMediationArtifact::from_bytes(&tampered, None).unwrap_err();
    assert_eq!(refusal_detail(&error), "nonlinear_mediation.confounding");

    let (mut meta, numbers) = decode_parts(&bytes).unwrap();
    meta.config.seed = 99;
    let tampered = encode_parts(&meta, &numbers, "tampered").unwrap();
    let error = NonlinearMediationArtifact::from_bytes(&tampered, None).unwrap_err();
    assert!(
        matches!(error, NonlinearMediationArtifactError::IdentityMismatch { field: "config" }),
        "{error:?}"
    );
}

#[test]
fn mediation_artifact_seal_refuses_confounding_and_interventional_estimands() {
    for flag in 0..3 {
        let mut r = request(0);
        match flag {
            0 => r.premises.unmeasured_treatment_outcome_confounding = true,
            1 => r.premises.unmeasured_treatment_mediator_confounding = true,
            _ => r.premises.unmeasured_mediator_outcome_confounding = true,
        }
        let error = NonlinearMediationArtifact::seal(&r).unwrap_err();
        assert_eq!(refusal_detail(&error), "nonlinear_mediation.confounding");
        assert_eq!(error.refusal().unwrap().0, "effect_not_identified");
    }
    let mut induced = request(0);
    induced.premises.treatment_induced_mediator_outcome_confounders = vec!["L".into()];
    let error = NonlinearMediationArtifact::seal(&induced).unwrap_err();
    assert_eq!(refusal_detail(&error), "nonlinear_mediation.treatment_induced_confounding");
    let mut interventional = request(0);
    interventional.config.estimand = "interventional_effects".into();
    let error = NonlinearMediationArtifact::seal(&interventional).unwrap_err();
    assert_eq!(refusal_detail(&error), "nonlinear_mediation.interventional_effects_closed");
}

#[test]
fn mediation_artifact_refuses_unknown_versions_features_and_corruption() {
    let (_, bytes) = seal(0);
    let (mut meta, numbers) = decode_parts(&bytes).unwrap();
    meta.version = NONLINEAR_MEDIATION_ARTIFACT_VERSION + 1;
    let future = encode_parts(&meta, &numbers, "future").unwrap();
    assert!(matches!(
        NonlinearMediationArtifact::from_bytes(&future, None),
        Err(NonlinearMediationArtifactError::UnsupportedVersion { .. })
    ));
    let (mut meta, numbers) = decode_parts(&bytes).unwrap();
    meta.feature = "something_else".into();
    let foreign = encode_parts(&meta, &numbers, "foreign").unwrap();
    assert!(matches!(
        NonlinearMediationArtifact::from_bytes(&foreign, None),
        Err(NonlinearMediationArtifactError::UnsupportedSemantics(_))
    ));
    let (mut meta, numbers) = decode_parts(&bytes).unwrap();
    meta.calibration = "measured".into();
    let claimed = encode_parts(&meta, &numbers, "claimed").unwrap();
    assert!(matches!(
        NonlinearMediationArtifact::from_bytes(&claimed, None),
        Err(NonlinearMediationArtifactError::UnsupportedSemantics("calibration"))
    ));
    let mut corrupt = bytes.clone();
    let middle = corrupt.len() / 2;
    corrupt[middle] ^= 0xFF;
    assert!(NonlinearMediationArtifact::from_bytes(&corrupt, None).is_err());
}

#[test]
fn mediation_artifact_caps_bootstrap_replicates() {
    let mut bootstrap = request(0);
    bootstrap.config.bootstrap_replicates = 1_000_000;
    assert!(matches!(
        NonlinearMediationArtifact::seal(&bootstrap),
        Err(NonlinearMediationArtifactError::LimitsExceeded("bootstrap replicates"))
    ));
}

/// The 40-row exact-grid dataset repeated `copies` times. Repetition leaves the fitted
/// coefficients, and so the closed-form effects, unchanged.
fn scaled_request(copies: usize) -> NonlinearMediationRequestWire {
    let base = data();
    let repeat = |column: &[f64]| -> Vec<f64> {
        column.iter().copied().cycle().take(column.len() * copies).collect()
    };
    NonlinearMediationRequestWire {
        premises: ignorable(),
        config: config(0),
        data: MediationDataWire {
            treatment: repeat(&base.treatment),
            mediator: repeat(&base.mediator),
            outcome: repeat(&base.outcome),
            covariate_names: base.covariate_names.clone(),
            covariates: base.covariates.iter().map(|c| repeat(c)).collect(),
        },
    }
}

fn assert_closed_form(artifact: &NonlinearMediationArtifact, rows: u64) {
    let (nde, nie, te) = truth();
    let result = &artifact.meta().result;
    assert_eq!(result.n_rows, rows);
    close(result.natural_direct, nde, "NDE");
    close(result.natural_indirect, nie, "NIE");
    close(result.total, te, "TE");
}

fn context_with_memory(hard_limit_bytes: u64) -> ExecutionContext {
    let mut ctx = ExecutionContext::production(0, 1);
    ctx.set_memory_limit(Some(hard_limit_bytes));
    ctx
}

/// Estimate, export under a budget, bind by digest and replay `copies * 40` rows.
fn exercise_budgets(copies: usize) {
    let request = scaled_request(copies);
    let rows = u64::try_from(40 * copies).unwrap();
    let artifact = NonlinearMediationArtifact::seal(&request).unwrap();
    assert_closed_form(&artifact, rows);
    exercise_export_and_replay(&request, &artifact, &|artifact| assert_closed_form(artifact, rows));
}

/// Export `artifact` under a budget, bind it to `request`'s dataset by digest and replay it;
/// `recovered` states what the estimate must still hold after each step.
fn exercise_export_and_replay(
    request: &NonlinearMediationRequestWire,
    artifact: &NonlinearMediationArtifact,
    recovered: &dyn Fn(&NonlinearMediationArtifact),
) {
    // Export: the dataset is embedded only inside the budget, and a refusal names the bytes.
    let embedded = artifact.to_bytes("scaled").unwrap();
    assert_eq!(artifact.to_bytes_within("scaled", embedded.len()).unwrap(), embedded);
    let error = artifact.to_bytes_within("scaled", embedded.len() - 1).unwrap_err();
    let (code, detail, text) = error.refusal().expect("a registered refusal");
    assert_eq!(
        (code, detail.as_str()),
        ("route_not_supported", "nonlinear_mediation.export_budget_exceeded")
    );
    assert!(text.contains(&format!("{} bytes", embedded.len())), "{text}");
    let error = artifact.to_bytes_within("scaled", 1_000).unwrap_err();
    assert_eq!(refusal_detail(&error), "nonlinear_mediation.export_budget_exceeded");
    // The refusal costs the caller nothing: the estimate is still complete in memory.
    recovered(artifact);

    // Replay: the digest-bound record carries no rows and needs the dataset it is bound to.
    let bound = artifact.to_bytes_digest_bound("bound").unwrap();
    assert!(decode_parts(&bound).unwrap().1.is_empty());
    assert!(!decode_parts(&embedded).unwrap().1.is_empty());
    let ctx = ExecutionContext::production(0, 1);
    let identity = artifact.identity();
    let error = NonlinearMediationArtifact::from_bytes_with(&bound, Some(identity), None, &ctx)
        .unwrap_err();
    assert_eq!(refusal_detail(&error), "nonlinear_mediation.data_required");
    let replay = NonlinearMediationArtifact::from_bytes_with(
        &bound,
        Some(identity),
        Some(&request.data),
        &ctx,
    )
    .unwrap();
    assert_eq!(replay.meta(), artifact.meta());
    let mut other = request.data.clone();
    other.outcome[0] += 1.0;
    assert!(matches!(
        NonlinearMediationArtifact::from_bytes_with(&bound, None, Some(&other), &ctx),
        Err(NonlinearMediationArtifactError::IdentityMismatch { field: "data" })
    ));
    // An artifact that embeds its dataset takes none, and replays on its own.
    assert!(matches!(
        NonlinearMediationArtifact::from_bytes_with(&embedded, None, Some(&request.data), &ctx),
        Err(NonlinearMediationArtifactError::UnsupportedSemantics(_))
    ));
    let replay = NonlinearMediationArtifact::from_bytes(&embedded, Some(identity)).unwrap();
    assert_eq!(replay.meta(), artifact.meta());
}

/// Deterministic draws: a `SplitMix64` stream with Box-Muller normals.
struct Population(u64);

impl Population {
    fn uniform(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1_u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let (u, v) = (1.0 - self.uniform(), self.uniform());
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
}

/// `rows` units of the SCM above with random draws: `A` a fair coin, `X` uniform on
/// `{-1, 0, 1, 2, 3}`, `M = alpha A + gamma X + e_M` with `e_M` normal of variance `sigma2`, and
/// the noiseless outcome.
fn population_request(rows: usize, seed: u64) -> NonlinearMediationRequestWire {
    let mut draws = Population(seed);
    let mut d = MediationDataWire {
        treatment: Vec::with_capacity(rows),
        mediator: Vec::with_capacity(rows),
        outcome: Vec::with_capacity(rows),
        covariate_names: vec!["x".into()],
        covariates: vec![Vec::with_capacity(rows)],
    };
    for _ in 0..rows {
        let a = f64::from(u8::from(draws.uniform() < 0.5));
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a uniform draw in [0, 1) scaled by the level count indexes a level"
        )]
        let x = XS[((draws.uniform() * 5.0) as usize).min(4)];
        let m = ALPHA * a + GAMMA * x + SIGMA2.sqrt() * draws.normal();
        d.treatment.push(a);
        d.mediator.push(m);
        d.outcome.push(B0 + B1 * a + C1 * m + C2 * m * m + DELTA * a * m + BX * x);
        d.covariates[0].push(x);
    }
    NonlinearMediationRequestWire { premises: ignorable(), config: config(0), data: d }
}

/// The closed-form effects, within the sampling error of the estimated mediator model.
fn assert_population_closed_form(artifact: &NonlinearMediationArtifact, rows: usize) {
    let (nde, nie, te) = truth();
    let result = &artifact.meta().result;
    assert_eq!(result.n_rows, u64::try_from(rows).unwrap());
    common::close(result.natural_direct, nde, 0.05, "NDE");
    common::close(result.natural_indirect, nie, 0.05, "NIE");
    common::close(result.total, te, 0.08, "TE");
}

/// The estimate completes on `rows` rows with its inferential standing exactly that of the
/// 40-row exact-grid fit, inside its runtime target, and exports and replays under explicit
/// budgets.
fn exercise_population(rows: usize, seed: u64, runtime_seconds: u64) {
    let request = population_request(rows, seed);
    let started = std::time::Instant::now();
    let artifact = NonlinearMediationArtifact::seal(&request).unwrap();
    assert_population_closed_form(&artifact, rows);
    assert!(started.elapsed().as_secs() < runtime_seconds, "{:?}", started.elapsed());

    let standing = artifact.meta();
    let reference = seal(0).0;
    let reference = reference.meta();
    assert_eq!(standing.calibration, "unmeasured");
    assert_eq!(
        (
            &standing.calibration,
            &standing.interval_status,
            &standing.inference_claim,
            &standing.estimand
        ),
        (
            &reference.calibration,
            &reference.interval_status,
            &reference.inference_claim,
            &reference.estimand
        ),
        "the declared inferential standing does not move with the number of rows"
    );
    assert!(standing.result.natural_direct_se.is_none(), "no standard error without replicates");

    // The computation budget refuses before any work and the export and replay budgets are
    // separate bounds on the same estimate.
    let error =
        NonlinearMediationArtifact::seal_with(&request, &context_with_memory(1_000)).unwrap_err();
    assert_eq!(refusal_detail(&error), "nonlinear_mediation.memory_budget_exceeded");
    exercise_export_and_replay(&request, &artifact, &|artifact| {
        assert_population_closed_form(artifact, rows);
    });
}

#[test]
fn mediation_has_no_row_ceiling_and_export_and_replay_are_budgeted() {
    // 30,000 rows: more than the former embedded-row ceiling.
    exercise_budgets(750);
}

#[test]
#[ignore = "estimates on 100,000 rows; run with --ignored"]
fn mediation_runs_at_one_hundred_thousand_rows() {
    exercise_budgets(2_500);
}

#[test]
fn nonlinear_mediation_fit_beyond_old_row_ceiling_50000() {
    exercise_population(50_000, 6_777_711, 900);
}

#[test]
#[ignore = "estimates on 100,000 rows; run with --ignored"]
fn nonlinear_mediation_fit_beyond_old_row_ceiling_100000() {
    exercise_population(100_000, 5_293_885, 900);
}

#[test]
fn mediation_computation_budget_refuses_before_work_and_honours_cancellation() {
    let request = request(0);
    // 40 rows need about 12 kB of working set.
    let error =
        NonlinearMediationArtifact::seal_with(&request, &context_with_memory(1_000)).unwrap_err();
    assert_eq!(
        error.refusal().map(|(code, detail, _)| (code, detail)),
        Some(("route_not_supported", "nonlinear_mediation.memory_budget_exceeded".into()))
    );
    assert!(
        NonlinearMediationArtifact::seal_with(&request, &context_with_memory(1_000_000)).is_ok()
    );
    let bytes = seal(0).1;
    let error = NonlinearMediationArtifact::from_bytes_with(
        &bytes,
        None,
        None,
        &context_with_memory(1_000),
    )
    .unwrap_err();
    assert_eq!(refusal_detail(&error), "nonlinear_mediation.memory_budget_exceeded");

    let cancelled = ExecutionContext::production(0, 1);
    cancelled.cancellation.cancel();
    let error = NonlinearMediationArtifact::seal_with(&request, &cancelled).unwrap_err();
    assert_eq!(
        error.refusal().map(|(code, detail, _)| (code, detail)),
        Some(("cancelled_no_claim", "nonlinear_mediation.cancelled".into()))
    );
}
