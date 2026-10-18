// Standalone prospective width-only supplement. Never computes or tests coverage.
// Frozen c30 DGP and two exact function excerpts are source-audited in plan.json.
#[path = "observational_effects.rs"]
mod dgp;
use antecedent_core::ExecutionContext;
use antecedent_estimate::observational_policy::{ObservationalPolicyValue, PolicyData, PolicyEvaluation, PolicyObjective, PolicyOrigin};
use antecedent_estimate::observational_survival::{CensoringAssumption, ObservationalSurvival, SurvivalData};
use dgp::{BinaryStudy, Censoring, HORIZON, binary_study, phi, survival_study, survival_truth};
use std::io::{Write, BufWriter};
const ATTEMPTS: u64 = 2048;
const ROWS: usize = 1000;
const Z95: f64 = 1.959_963_984_540_054;
const SEED_SALT: u64 = 0xF29ED7109891A5FD;
fn estimate_policy(
    data: &BinaryStudy,
    evaluation: PolicyEvaluation,
    seed: u64,
) -> Option<antecedent_estimate::observational_policy::PolicyValueEstimate> {
    ObservationalPolicyValue::new(vec![0.0, 1.0], PolicyObjective::Maximize, evaluation)
        .estimate(
            &PolicyData {
                outcome: &data.outcome,
                treatment: &data.treatment,
                covariates: &[&data.z],
            },
            &ExecutionContext::for_tests(seed),
        )
        .ok()
}

fn learned_rule_value(data: &BinaryStudy, rows: &[u32], actions: &[f64]) -> Option<f64> {
    let z = |i: usize| data.z[rows[i] as usize];
    let treated: Vec<f64> =
        (0..rows.len()).filter(|&i| (actions[i] - 1.0).abs() < f64::EPSILON).map(z).collect();
    let untreated: Vec<f64> = (0..rows.len()).filter(|&i| actions[i] == 0.0).map(z).collect();
    if treated.is_empty() || untreated.is_empty() {
        return None;
    }
    let fold =
        |values: &[f64], f: fn(f64, f64) -> f64, init: f64| values.iter().copied().fold(init, f);
    let mean = |values: &[f64]| values.iter().sum::<f64>() / values.len() as f64;
    if mean(&treated) > mean(&untreated) {
        // Treat above the threshold: it lies between the highest untreated and lowest treated z.
        let (low, high) = (
            fold(&untreated, f64::max, f64::NEG_INFINITY),
            fold(&treated, f64::min, f64::INFINITY),
        );
        (low <= high).then(|| 1.0 + 1.5 * phi(0.5 * (low + high)))
    } else {
        let (low, high) = (
            fold(&treated, f64::max, f64::NEG_INFINITY),
            fold(&untreated, f64::min, f64::INFINITY),
        );
        (low <= high).then(|| 1.0 - 1.5 * phi(0.5 * (low + high)))
    }
}

fn wald_width(estimate: f64, se: f64) -> Option<f64> {
    if !(estimate.is_finite() && se.is_finite() && se > 0.0) { return None; }
    let (lo, hi) = (estimate - Z95 * se, estimate + Z95 * se);
    (lo.is_finite() && hi.is_finite() && lo <= hi).then_some(hi - lo)
}
fn policy_width(seed: u64) -> Option<(f64, f64)> {
    let data = binary_study(ROWS, seed, 1.5, 0, 1.0);
    let estimate = estimate_policy(&data, PolicyEvaluation::HeldOut { evaluation_fraction: 0.4 }, seed)?;
    let truth = learned_rule_value(&data, &estimate.row_index, &estimate.policy_actions)?;
    // Width of exactly the source's Wald endpoints, with no coverage comparison.
    Some((wald_width(estimate.value, estimate.se)?, truth.abs()))
}
fn survival_width(seed: u64) -> Option<(f64, f64)> {
    let data = survival_study(ROWS, seed, Censoring::DependsOnCovariates(0.12), 1.0);
    let estimate = ObservationalSurvival::new(1.0, 0.0, HORIZON, CensoringAssumption::AtRandomGivenCovariates)
        .estimate(&SurvivalData { duration: &data.duration, event: &data.event,
            treatment: &data.treatment, covariates: &[&data.z1, &data.z2] },
            &ExecutionContext::for_tests(seed)).ok()?;
    let contrast = estimate.survival_difference;
    let truth = survival_truth(1.0, HORIZON).0 - survival_truth(0.0, HORIZON).0;
    Some((wald_width(contrast.estimate, contrast.se)?, truth.abs()))
}
fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let n = values.len();
    values[n / 2]
}
fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert!(args.len() == 2 && matches!(args[1].as_str(), "policy" | "survival"), "one fixed case argument required: policy or survival");
    let case = args[1].as_str();
    let base: u64 = if case == "policy" { 0x6b_0000 } else { 0x5e_0000 };
    let mut widths = Vec::new(); let mut truths = Vec::new(); let mut ratios = Vec::new();
    let mut out = BufWriter::new(std::io::stdout().lock());
    writeln!(out, "case,replicate,seed,status,width,abs_truth").unwrap();
    for replicate in 0..ATTEMPTS {
        let seed = (base + replicate) ^ SEED_SALT;
        let result = if case == "policy" { policy_width(seed) } else { survival_width(seed) };
        match result.filter(|(w,t)| w.is_finite() && *w >= 0.0 && t.is_finite() && *t > 0.0) {
            Some((width, truth)) => {
                writeln!(out, "{case},{replicate},{seed},reported,{width:.17e},{truth:.17e}").unwrap();
                widths.push(width);truths.push(truth);ratios.push(width/truth);
            },
            None => {writeln!(out, "{case},{replicate},{seed},refused,,").unwrap();}
        }
    }
    out.flush().unwrap();
    let reported = widths.len();
    let mw = if reported > 0 { median(&mut widths) } else { f64::NAN };
    let mt = if reported > 0 { median(&mut truths) } else { f64::NAN };
    let mr = if reported > 0 { median(&mut ratios) } else { f64::NAN };
    eprintln!("{}", serde_json::json!({"case":case,"attempts":ATTEMPTS,"reported":reported,
        "reporting_rate":reported as f64 / ATTEMPTS as f64,"median_width":mw,
        "median_abs_truth":mt,"median_width_over_median_abs_truth":mw/mt,
        "median_per_replicate_width_over_abs_truth":mr,"coverage_computed":false,
        "required_reported":2000,"max_median_width_over_effect":1.5}));
    assert!(reported >= 2000, "fewer than2000 finite reported intervals");
    assert!(mw / mt <= 1.5, "frozen median width target failed");
    assert!(mr <= 1.5, "per-replicate median width/truth target failed");
}
