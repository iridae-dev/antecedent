//! Compare day-three cause-1 incidence in a randomized cohort with delayed entry.
//! Pass a filename to export the fitted result and print its identity. Retain that identity
//! separately, then pass `--read FILE IDENTITY` to this example or its Python counterpart.
use std::{error::Error, fs, path::Path, sync::Arc};

use antecedent::{Study, prelude::ExecutionContext, support::CellStatus};
use antecedent_core::{
    CausalQuery, ObservationAssumption, SurvivalFunctional, SurvivalQuery, VariableId,
};
use antecedent_data::TabularData;
use antecedent_io::{ReplayOptions, RetainedArtifactIdentity, load_analysis_result_expecting};

fn read(bytes: &[u8], identity: &RetainedArtifactIdentity) -> Result<(), Box<dyn Error>> {
    let loaded = load_analysis_result_expecting(bytes, identity)?;
    assert!(loaded.replay(&ReplayOptions::default()).replayed());
    let curve = loaded.body.survival.as_ref().ok_or("missing incidence curve")?;
    assert_eq!(curve.target_cause, Some(1));
    assert!((curve.tau - 3.0).abs() < f64::EPSILON);
    assert!(
        (curve.treated.last().ok_or("no treated estimate")?
            - curve.control.last().ok_or("no control estimate")?
            - 0.25)
            .abs()
            < 1e-12
    );
    assert_eq!(curve.graphless_support_status.as_deref(), Some("supported"));
    let [low, high] = curve.difference_at_tau_interval.ok_or("missing pointwise interval")?;
    assert!(low < 0.25 && 0.25 < high);
    println!("Replayed day-three interval [{low}, {high}]");
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if let [flag, path, digest] = args.as_slice() {
        if flag != "--read" {
            return Err("expected --read FILE IDENTITY".into());
        }
        let identity =
            RetainedArtifactIdentity::from_hex(digest.to_str().ok_or("invalid identity")?)?;
        return read(&fs::read(path)?, &identity);
    }
    if args.len() > 1 {
        return Err("expected FILE or --read FILE IDENTITY".into());
    }
    run(args.first().map(Path::new))
}

/// Fit both cohorts and optionally export the supported result for an independent consumer.
pub fn run(output: Option<&Path>) -> Result<(), Box<dyn Error>> {
    let ctx = ExecutionContext::production_default(297);
    for thin in [false, true] {
        let (mut duration, mut event, mut treatment, mut entry) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for arm in [0, 1] {
            for i in 0..160 {
                let slot = i % 8;
                let cause = if slot < 2 + 2 * arm {
                    1
                } else if slot == 7 {
                    2
                } else {
                    0
                };
                // Early censoring after the last event leaves twenty treated subjects at day three.
                let exit = if cause > 0 {
                    2.0
                } else if thin && arm == 1 && slot != 4 {
                    2.5
                } else {
                    3.0
                };
                duration.push(exit);
                event.push(f64::from(cause));
                treatment.push(f64::from(arm));
                entry.push(f64::from(i % 2));
            }
        }
        let data = TabularData::from_f64_columns([
            ("duration", duration.as_slice()),
            ("cause", &event),
            ("arm", &treatment),
            ("entry", &entry),
        ])?;
        let query = SurvivalQuery {
            duration: VariableId::from_raw(0),
            event: VariableId::from_raw(1),
            treatment: VariableId::from_raw(2),
            delayed_entry: Some(VariableId::from_raw(3)),
            tau: 3.0,
            known_censoring: None,
            stratified_censoring: None,
            observation_assumption: ObservationAssumption::IndependentGiven(Arc::from([])),
            functional: SurvivalFunctional::CumulativeIncidence { target_cause: 1 },
        };
        let plan = Study::tabular(data.clone())
            .query(CausalQuery::Survival(query))
            .bootstrap_replicates(299)
            .build()?
            .prepare(&ctx)?;
        let result = plan.estimate(&data, &ctx)?;
        let curve = result.survival().ok_or("missing incidence curve")?;
        assert!(curve.difference_at_tau_interval.is_some());
        assert_eq!(curve.minimum_horizon_risk_set, Some(if thin { 20 } else { 60 }));
        assert_eq!(result.support_status(), (!thin).then_some(CellStatus::Supported));
        println!(
            "At risk, support: {:?}, {:?}",
            curve.minimum_horizon_risk_set,
            result.support_status()
        );
        if !thin {
            let bytes = plan.encode_contracted_result_with_source(
                &result,
                &data,
                "delayed-entry-incidence",
                &ctx,
            )?;
            let identity = RetainedArtifactIdentity::of_bytes(&bytes);
            read(&bytes, &identity)?;
            if let Some(path) = output {
                fs::write(path, &bytes)?;
                println!("Retain this identity: {}", identity.to_hex());
            }
        }
    }
    Ok(())
}
