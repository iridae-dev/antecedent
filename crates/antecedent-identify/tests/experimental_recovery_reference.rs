//! Pair-coverage verdict of the clamp-experiment design judge against exact generic rank.
//!
//! `oracle_data/experimental_recovery_reference.py` (frozen in the JSON beside it) takes two
//! fixed dense rational systems `x = B x`, builds for every set of at most three clamp experiments
//! over three and four units the rows `b_tj + sum_u T[u][j] b_tu = T[t][j]` with
//! `T = (I - B_UU)^-1 B_UJ`, checks that `B` satisfies them, and reports the targets whose stacked
//! rows have full rank `n - 1` by exact elimination. The judge must call exactly those targets
//! covered, and must leave every other target's incoming row unresolved for lack of pair coverage.

use antecedent_core::VariableId;
use antecedent_identify::experimental_recovery::{
    ClampExogeneity, ClampExperimentSet, DeclaredClampExperiment, DeclaredNoiseLaw,
    DirectCoefficient, ExperimentDesign, ExperimentalRecoveryDecision, FiniteSampleConditioning,
    PhysicalDynamicalStability, RecoveryAttempt, RegimeAssumption, UniquenessRestriction,
    UnresolvedReason, WeakStability, decide_experimental_recovery,
};
use serde_json::Value as Json;

fn node(raw: u64) -> VariableId {
    VariableId::from_raw(u32::try_from(raw).unwrap())
}

fn design(n: u64, clamped: &[Vec<u64>]) -> ExperimentDesign {
    let experiments = clamped
        .iter()
        .map(|set| DeclaredClampExperiment {
            clamped: set.iter().map(|&u| node(u)).collect(),
            passive: (0..n).filter(|u| !set.contains(u)).map(node).collect(),
            exogeneity: ClampExogeneity::RandomizedIntervention,
        })
        .collect();
    ExperimentDesign {
        nodes: (0..n).map(node).collect(),
        attempt: RecoveryAttempt::ClampExperiments(ClampExperimentSet {
            experiments,
            noise: DeclaredNoiseLaw::IndependentGaussian,
            regimes: RegimeAssumption::Independent,
            weak_stability: WeakStability::InvertibilityDeclared,
            physical_dynamical_stability: PhysicalDynamicalStability::NotDeclared,
            finite_sample_conditioning: FiniteSampleConditioning::NotDeclared,
            uniqueness_restriction: UniquenessRestriction::NotOffered,
            equation_ranks: Vec::new(),
        }),
    }
}

#[test]
fn pair_coverage_equals_exact_generic_rank_on_every_small_experiment_set() {
    let cases: Json =
        serde_json::from_str(include_str!("oracle_data/experimental_recovery_reference.json"))
            .unwrap();
    let cases = cases["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 510);
    let mut closed = 0;
    for case in cases {
        let n = case["units"].as_u64().unwrap();
        let clamped: Vec<Vec<u64>> = case["clamped"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| e.as_array().unwrap().iter().map(|u| u.as_u64().unwrap()).collect())
            .collect();
        let determined: Vec<u64> =
            case["determined"].as_array().unwrap().iter().map(|u| u.as_u64().unwrap()).collect();
        let recovery = match decide_experimental_recovery(&design(n, &clamped)) {
            ExperimentalRecoveryDecision::Recovered(r)
            | ExperimentalRecoveryDecision::Partial(r) => r,
            other => panic!("{clamped:?}: unexpected decision {other:?}"),
        };
        let mut unresolved_targets = Vec::new();
        for direction in &recovery.unresolved {
            let UnresolvedReason::IncompletePairCoverage { target, .. } = &direction.reason else {
                panic!("{clamped:?}: unexpected reason {:?}", direction.reason);
            };
            unresolved_targets.push(*target);
        }
        for t in 0..n {
            let covered = determined.contains(&t);
            assert_eq!(
                !unresolved_targets.contains(&node(t)),
                covered,
                "n={n} experiments={clamped:?} target={t}"
            );
            let all_recoverable = (0..n).filter(|&s| s != t).all(|s| {
                recovery.recoverable.contains(&DirectCoefficient { into: node(t), from: node(s) })
            });
            assert_eq!(all_recoverable, covered, "recoverable set, target {t}, {clamped:?}");
        }
        if recovery.goal_closed() {
            closed += 1;
            assert_eq!(determined.len() as u64, n);
        }
    }
    assert!(closed > 0);
}
