//! Shared reading of the frozen procedure references (`oracle_data/procedure_*_reference.json`):
//! a declaration, a kernel and the exact terminal atoms written by `procedure_exact.py`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(dead_code, reason = "each test file uses the part of the support it needs")]

use antecedent_stats::adaptive_inference::{
    AssignmentRule, Attribution, BetaPrior, ExposureModel, FinalSelection, OutcomeKernel,
    OutcomeObservation, ProcedureDeclaration, ProcedureLaw, ProcedureSpec, Sidedness, StoppingRule,
};
use serde_json::Value;

pub fn never() -> bool {
    false
}

/// A number written as `n/d` or as a decimal string.
pub fn frac(value: &Value) -> f64 {
    let text = value.as_str().expect("a number written as text");
    match text.split_once('/') {
        Some((n, d)) => {
            n.parse::<f64>().expect("numerator") / d.parse::<f64>().expect("denominator")
        }
        None => text.parse().expect("a decimal"),
    }
}

pub fn fracs(value: &Value) -> Vec<f64> {
    value.as_array().expect("array").iter().map(frac).collect()
}

fn prior(value: &Value) -> BetaPrior {
    let pair = value.as_array().expect("a prior pair");
    BetaPrior { alpha: pair[0].as_f64().expect("alpha"), beta: pair[1].as_f64().expect("beta") }
}

pub fn declaration(d: &Value) -> ProcedureDeclaration {
    let arms = d["arms"].as_u64().expect("arms") as usize;
    let a = &d["assignment"];
    let assignment = match a["kind"].as_str().expect("kind") {
        "fixed" => AssignmentRule::Fixed { probabilities: fracs(&a["probabilities"]) },
        "epsilon_greedy" => AssignmentRule::EpsilonGreedy {
            epsilon: frac(&a["epsilon"]),
            prior: prior(&a["prior"]),
        },
        "thompson" => AssignmentRule::ThompsonBeta { prior: prior(&a["prior"]) },
        "thompson_floored" => AssignmentRule::ThompsonBetaFloored {
            prior: prior(&a["prior"]),
            min_probability: frac(&a["floor"]),
        },
        "ucb" => AssignmentRule::UpperConfidenceBound { exploration: frac(&a["exploration"]) },
        other => panic!("unknown assignment {other}"),
    };
    let s = &d["stopping"];
    let stopping = match s["kind"].as_str().expect("kind") {
        "none" => StoppingRule::NoStopping,
        "cap" => {
            StoppingRule::ResourceCap { max_units: s["max_units"].as_u64().expect("cap") as u32 }
        }
        "best" => StoppingRule::PosteriorProbabilityBest {
            threshold: frac(&s["threshold"]),
            prior: prior(&s["prior"]),
        },
        "loss" => StoppingRule::PosteriorExpectedLoss {
            loss_threshold: frac(&s["threshold"]),
            prior: prior(&s["prior"]),
        },
        "futility" => StoppingRule::PosteriorFutility {
            margin: frac(&s["margin"]),
            threshold: frac(&s["threshold"]),
            prior: prior(&s["prior"]),
        },
        "wald" => StoppingRule::WaldBoundaries {
            critical: fracs(&s["critical"]),
            sidedness: if s["sidedness"] == "upper" {
                Sidedness::OneSidedUpper
            } else {
                Sidedness::TwoSidedSymmetric
            },
        },
        other => panic!("unknown stopping {other}"),
    };
    let selection = match d["selection"]["kind"].as_str().expect("kind") {
        "none" => FinalSelection::NoSelection,
        "mean" => FinalSelection::HighestObservedMean,
        "posterior_mean" => FinalSelection::HighestPosteriorMean(prior(&d["selection"]["prior"])),
        other => panic!("unknown selection {other}"),
    };
    let retain = d.get("retain").map_or((1.0, 1.0), |r| (frac(&r[0]), frac(&r[1])));
    let observation = OutcomeObservation {
        lag_looks: d.get("lag").map_or(0, |l| l.as_u64().expect("lag") as usize),
        retain_success: retain.0,
        retain_failure: retain.1,
    };
    let exposure = d.get("exposure").map_or_else(
        || ExposureModel::none(arms),
        |e| ExposureModel {
            failure_probability: fracs(&e["failure"]),
            fallback_arm: e["fallback"].as_u64().expect("fallback") as usize,
            attribution: if e["attribution"] == "assigned" {
                Attribution::Assigned
            } else {
                Attribution::Delivered
            },
        },
    );
    ProcedureDeclaration {
        arms,
        context_probabilities: None,
        assignment: Some(assignment),
        enrolment: Some(
            d["enrolment"]
                .as_array()
                .expect("enrolment")
                .iter()
                .map(|n| n.as_u64().expect("units") as u32)
                .collect(),
        ),
        observation: Some(observation),
        exposure: Some(exposure),
        stopping: Some(stopping),
        selection: Some(selection),
    }
}

pub fn spec(d: &Value) -> ProcedureSpec {
    ProcedureSpec::new(declaration(d)).expect("a valid declaration")
}

pub fn kernel(k: &Value) -> OutcomeKernel {
    match k["kind"].as_str().expect("kind") {
        "fixed" => OutcomeKernel::Fixed { success_probabilities: fracs(&k["values"]) },
        _ => OutcomeKernel::PriorPredictive {
            priors: k["values"].as_array().expect("priors").iter().map(prior).collect(),
        },
    }
}

/// Every reference atom is in the law at its probability, and the law holds nothing else.
pub fn assert_atoms(law: &ProcedureLaw, atoms: &Value, tolerance: f64, what: &str) {
    let atoms = atoms.as_array().expect("atoms");
    let mut matched = vec![false; law.atoms.len()];
    for atom in atoms {
        let look = atom["look"].as_u64().expect("look") as usize;
        let stopped = atom["stopped"].as_bool().expect("stopped");
        let observed: Vec<u16> = atom["observed"]
            .as_array()
            .expect("observed")
            .iter()
            .map(|n| n.as_u64().expect("count") as u16)
            .collect();
        let decision = atom["decision"].as_u64().map(|n| n as usize);
        let want = atom["pf"].as_f64().expect("probability");
        let found = law.atoms.iter().position(|a| {
            a.look == look
                && a.rule_stopped == stopped
                && a.observed == observed
                && a.decision == decision
        });
        let index = found.unwrap_or_else(|| {
            panic!(
                "{what}: missing terminal state {look} {stopped} {observed:?} {decision:?} ({want})"
            )
        });
        matched[index] = true;
        let got = law.atoms[index].probability;
        assert!(
            (got - want).abs() <= tolerance,
            "{what}: state {look} {stopped} {observed:?} {decision:?}: {got} vs {want}"
        );
    }
    for (index, atom) in law.atoms.iter().enumerate() {
        assert!(
            matched[index] || atom.probability <= tolerance,
            "{what}: unexpected state {} {} {:?} {:?} with probability {}",
            atom.look,
            atom.rule_stopped,
            atom.observed,
            atom.decision,
            atom.probability
        );
    }
}
