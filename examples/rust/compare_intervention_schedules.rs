// SPDX-License-Identifier: MIT OR Apache-2.0
//! Replay a two-period schedule decision across structural and preference alternatives.
//! Pass a filename to save the bundle. Retain the printed identity separately and
//! replay with `--read FILE IDENTITY` in either the Rust or Python example.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "subject counts floor small nonnegative products of a fixed cohort and probabilities"
)]

use std::{fs, path::Path};

use antecedent::analysis::composition::BundleLimits;
use antecedent::design::{
    ScheduleBundleError, ScheduleDecisionDeclaration, ScheduleDecisionError, ScheduleSourceSubject,
    ScheduleStructure, compile_schedule_decision_bundle, consume_schedule_decision_bundle,
};
use antecedent_core::ExecutionContext;

/// Fit the synthetic cohort, replay the decision and exercise the missing-evidence boundary.
pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    write_example(None)
}

fn write_example(output: Option<&Path>) -> Result<(), Box<dyn std::error::Error>> {
    let endpoints = [[0.125, 0.25, 0.5, 0.875], [0.25, 0.375, 0.625, 1.0]];
    let early = [0.5, 0.75];
    let late = [[0.25, 0.75], [0.5, 0.25]];
    let retention = [[0.875, 0.75, 0.75, 0.875], [0.75, 0.875, 0.875, 0.75]];
    let mut source = Vec::new();
    for (baseline, outcomes) in endpoints.iter().enumerate() {
        for (schedule, &outcome) in outcomes.iter().enumerate() {
            let first = schedule / 2;
            let second = schedule % 2;
            let p = early[baseline];
            let q = late[baseline][first];
            let p_first = if first == 1 { p } else { 1.0 - p };
            let p_second = if second == 1 { q } else { 1.0 - q };
            let assigned = (8192.0 * p_first * p_second) as usize;
            let observed = (assigned as f64 * retention[baseline][schedule]) as usize;
            source.extend((0..assigned).map(|i| ScheduleSourceSubject {
                baseline: baseline == 1,
                actions: [first == 1, second == 1],
                endpoint: (i < observed).then_some(outcome),
            }));
        }
    }
    let target: Vec<_> = (0..1024).map(|i| i % 4 != 0).collect();
    let mut declaration = ScheduleDecisionDeclaration {
        calibration_design: None,
        schedules: vec![0, 1, 2, 3],
        structures: vec![ScheduleStructure::BothActionsCausal, ScheduleStructure::FirstActionOnly],
        preference_costs: vec![
            [0.0, 0.0625, 0.125, 0.1875],
            [0.0, 0.1875, 0.460_937_5, 0.460_937_5],
        ],
    };
    let ctx = ExecutionContext::production_default(0);
    let run = compile_schedule_decision_bundle(
        source.clone(),
        target.clone(),
        declaration.clone(),
        &ctx,
    )?;
    let bytes = run.bundle.to_bytes("two-period-schedules", &BundleLimits::default())?;
    read(&bytes, run.bundle.identity())?;
    if let Some(path) = output {
        fs::write(path, &bytes)?;
        println!("Retain this identity: {}", run.bundle.identity());
    }
    declaration.preference_costs.clear();
    match compile_schedule_decision_bundle(source, target, declaration, &ctx) {
        Err(ScheduleBundleError::Procedure(ScheduleDecisionError::MissingJointEvidence)) => {
            println!(
                "With preference evidence removed: {}",
                ScheduleDecisionError::MissingJointEvidence
            );
        }
        Err(error) => return Err(error.into()),
        Ok(_) => panic!("a schedule decision requires the joint preference costs"),
    }
    Ok(())
}

fn read(bytes: &[u8], identity: &str) -> Result<(), Box<dyn std::error::Error>> {
    let decision = consume_schedule_decision_bundle(bytes, identity)?;
    assert_eq!(decision.optimum_set, [0, 2, 3]);
    println!("Compatible schedules: {:?}", decision.optimum_set);
    println!("Coverage scope: {}", decision.coverage_scope);
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    match args.as_slice() {
        [] => run(),
        [path] => write_example(Some(Path::new(path))),
        [flag, path, digest] if flag == "--read" => {
            let identity = digest.to_str().ok_or("invalid identity")?;
            read(&fs::read(path)?, identity)
        }
        _ => Err("expected FILE or --read FILE IDENTITY".into()),
    }
}
