//! Cross-world selective execution through the facade: the checked capability
//! matrix, a sealed execution, a fresh-process replay and a licensed resume.
//!
//! Oracle (hand computed): mediation model with `X = U_X` (0.5), mediator
//! response types `qm = [0.3, 0.4, 0.1, 0.2]` of `X`, and `Y = M` when
//! `U_Y = 0` (0.75) and `Y = X` when `U_Y = 1` (0.25). Then
//! `E[Y_{x1,M_x0}] = 0.75 * 0.3 + 0.25 = 0.475`, `E[Y_{x0}] = 0.225`, so the
//! natural direct effect is `0.25`.

use antecedent::analysis::cross_world_execution::{
    CapabilityStatus, CrossWorldExecution, ExogenousInference, ExogenousVariable, FactualUnit,
    FiniteExecutionRequest, FiniteQuantity, FiniteScm, FiniteVariable, FiniteWorld, NoiseCoupling,
    StageStatus, WorldAtom, WorldRoute, capability_matrix,
};
use antecedent_contract::RequestedReading;
use antecedent_core::ExecutionContext;

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(21)
}

fn mediation() -> FiniteScm {
    FiniteScm {
        population: "trial-population".to_owned(),
        history: "baseline-regime".to_owned(),
        exogenous: vec![
            ExogenousVariable { name: "u_x".to_owned(), pmf: vec![0.5, 0.5] },
            ExogenousVariable { name: "u_m".to_owned(), pmf: vec![0.3, 0.4, 0.1, 0.2] },
            ExogenousVariable { name: "u_y".to_owned(), pmf: vec![0.75, 0.25] },
        ],
        variables: vec![
            FiniteVariable {
                name: "x".to_owned(),
                levels: 2,
                parents: vec![],
                exogenous: 0,
                table: vec![0, 1],
            },
            FiniteVariable {
                name: "m".to_owned(),
                levels: 2,
                parents: vec![0],
                exogenous: 1,
                table: vec![0, 0, 0, 1, 1, 0, 1, 1],
            },
            FiniteVariable {
                name: "y".to_owned(),
                levels: 2,
                parents: vec![0, 1],
                exogenous: 2,
                table: vec![0, 1, 0, 1, 0, 0, 1, 1],
            },
        ],
    }
}

fn request() -> FiniteExecutionRequest {
    let scm = mediation();
    let route = WorldRoute { parent: 1, child: 2, source: 1 };
    FiniteExecutionRequest {
        evidence_history: scm.history.clone(),
        scm,
        units: vec![FactualUnit { unit: "cohort-mean".to_owned(), observed: vec![] }],
        worlds: vec![
            FiniteWorld {
                name: "x1_with_m_of_x0".to_owned(),
                interventions: vec![(0, 1)],
                routes: vec![route],
            },
            FiniteWorld { name: "x0".to_owned(), interventions: vec![(0, 0)], routes: vec![] },
        ],
        quantity: FiniteQuantity::MeanContrast { plus: (0, 2), minus: (1, 2) },
        coupling: NoiseCoupling::SharedAbducedExogenous,
        exogenous: ExogenousInference::ExactEnumeration,
        target_population: None,
        reading: RequestedReading::Point,
    }
}

#[test]
fn the_capability_matrix_is_stated_and_every_closed_row_is_actionable() {
    let rows = capability_matrix();
    assert!(rows.iter().any(|r| r.status == CapabilityStatus::Executed));
    for row in rows.iter().filter(|r| r.status == CapabilityStatus::Closed) {
        assert!(row.refusal_code.is_some() && !row.remedy.is_empty(), "{}", row.id);
    }
}

#[test]
fn natural_direct_effect_seals_replays_and_resumes() {
    let sealed = CrossWorldExecution::seal(&request(), &ctx()).unwrap();
    assert!((sealed.result().value - 0.25).abs() < 1e-12, "{}", sealed.result().value);
    assert_eq!(sealed.result().label, "point_only");
    assert!(sealed.interval().is_err());
    assert_eq!(sealed.posterior().unwrap_err().code, "cell_not_licensed");

    let bytes = sealed.to_bytes("natural-direct").unwrap();
    let mut consumed =
        CrossWorldExecution::consume(&bytes, Some(sealed.identity()), &ctx()).unwrap();
    assert_eq!(consumed.result().value.to_bits(), sealed.result().value.to_bits());

    // Licensed resume: a new action definition (indirect effect) reuses the
    // verified abduction and equals a full rerun bit for bit.
    let mut next = request();
    next.worlds = vec![
        FiniteWorld {
            name: "x0_with_m_of_x1".to_owned(),
            interventions: vec![(0, 0)],
            routes: vec![WorldRoute { parent: 1, child: 2, source: 1 }],
        },
        FiniteWorld { name: "x1".to_owned(), interventions: vec![(0, 1)], routes: vec![] },
        FiniteWorld { name: "x0".to_owned(), interventions: vec![(0, 0)], routes: vec![] },
    ];
    // E[Y_{x0,M_x1}] - E[Y_{x0}] = 0.75 * 0.6 - 0.225 = 0.225.
    next.quantity = FiniteQuantity::MeanContrast { plus: (0, 2), minus: (2, 2) };
    let resumed = consumed.resume(&next, &ctx()).unwrap();
    assert_eq!(resumed.work.abductions, 0);
    assert_eq!(resumed.work.abductions_reused, 1);
    assert_eq!(resumed.stages[1].status, StageStatus::Reused);
    assert!((resumed.value - 0.225).abs() < 1e-12);
    let full = CrossWorldExecution::seal(&next, &ctx()).unwrap();
    assert_eq!(full.result().value.to_bits(), resumed.value.to_bits());
    assert_eq!(full.result().receipt, resumed.receipt);
}

#[test]
fn a_fresh_process_consumes_only_the_bytes_and_matches_the_oracle() {
    const PATH: &str = "ANTECEDENT_CROSS_WORLD_EXECUTION_REPLAY_PATH";
    if let Ok(path) = std::env::var(PATH) {
        let bytes = std::fs::read(path).unwrap();
        let consumed = CrossWorldExecution::consume(&bytes, None, &ctx()).unwrap();
        assert!((consumed.result().value - 0.25).abs() < 1e-12);
        assert_eq!(consumed.result().work.mechanism_fits, 0);
        return;
    }
    let sealed = CrossWorldExecution::seal(&request(), &ctx()).unwrap();
    let unique =
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let path = std::env::temp_dir()
        .join(format!("antecedent-cross-world-{}-{unique}.bin", std::process::id()));
    std::fs::write(&path, sealed.to_bytes("natural-direct").unwrap()).unwrap();
    // The child must select this very test: under a suite crate it is a module member, so its
    // libtest name carries the module path (a bare name would match nothing and "pass").
    let prefix =
        module_path!().split_once("::").map_or(String::new(), |(_, rest)| format!("{rest}::"));
    let test_name = format!("{prefix}a_fresh_process_consumes_only_the_bytes_and_matches_the_oracle");
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .arg("--exact")
        .arg(&test_name)
        .env(PATH, &path)
        .output()
        .unwrap();
    std::fs::remove_file(path).unwrap();
    assert!(
        String::from_utf8_lossy(&output.stdout).contains("running 1 test"),
        "the child selected no test: {}",
        String::from_utf8_lossy(&output.stdout)
    );
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
}

#[test]
fn independent_world_noise_is_never_a_counterfactual_joint_law() {
    let mut independent = request();
    independent.coupling = NoiseCoupling::IndependentWorldNoise;
    independent.quantity = FiniteQuantity::JointProbability(vec![
        WorldAtom { world: 0, variable: 2, level: 1 },
        WorldAtom { world: 1, variable: 2, level: 0 },
    ]);
    let error = CrossWorldExecution::seal(&independent, &ctx()).expect_err("refused");
    let refusal = error.refusal().unwrap();
    assert_eq!(refusal.code, "joint_law_required");
    assert!(refusal.remedy.is_some());
    // Coupled, the same event has the hand-computed mass 0.25 (only U_Y = 1
    // separates Y_{x1,M_x0} = 1 from Y_{x0} = 0).
    independent.coupling = NoiseCoupling::SharedAbducedExogenous;
    let coupled = CrossWorldExecution::seal(&independent, &ctx()).unwrap();
    assert!((coupled.result().value - 0.25).abs() < 1e-12);
}
