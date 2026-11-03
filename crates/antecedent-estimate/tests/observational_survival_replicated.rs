//! Observational survival effects over independent replicate studies, against the exact
//! closed-form truth of the generating process, with a stated Monte-Carlo bound.
//!
//! The generating process has exponential event times with a proportional-hazards treatment
//! effect and two discrete confounders, so the survival probability and restricted mean survival
//! of each arm are finite sums of exponentials over the six confounder cells
//! (`common/observational_effects.rs::survival_truth`). Each of `R` independent studies of
//! `N` units estimates the survival-probability difference and the restricted-mean difference
//! at the horizon. The average of the `R` estimates has standard error `sqrt(mean(se^2) / R)`
//! (the studies are independent); the bias of the cross-fitted doubly robust score at this
//! size is far below the bound's floor of 0.004, so the replicate average must be within four
//! standard errors of the closed-form truth plus that floor. A single seeded draw is not used.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;
use antecedent_estimate::observational_survival::{
    CensoringAssumption, ObservationalSurvival, SurvivalData,
};

#[path = "common/observational_effects.rs"]
mod dgp;

use dgp::{Censoring, HORIZON, survival_study, survival_truth};

const REPLICATES: u64 = 20;
const UNITS: usize = 3000;

#[test]
fn replicate_average_survival_effects_match_the_closed_form_truth_within_a_stated_bound() {
    let (s1, r1) = survival_truth(1.0, HORIZON);
    let (s0, r0) = survival_truth(0.0, HORIZON);
    let (mut surv, mut surv_var, mut rmst, mut rmst_var) = (0.0, 0.0, 0.0, 0.0);
    for replicate in 0..REPLICATES {
        let data = survival_study(UNITS, 1000 + replicate, Censoring::Independent(0.15), 1.0);
        let estimate = ObservationalSurvival::new(
            1.0,
            0.0,
            HORIZON,
            CensoringAssumption::IndependentWithinArm,
        )
        .estimate(
            &SurvivalData {
                duration: &data.duration,
                event: &data.event,
                treatment: &data.treatment,
                covariates: &[&data.z1, &data.z2],
            },
            &ExecutionContext::for_tests(3 + replicate),
        )
        .unwrap();
        surv += estimate.survival_difference.estimate;
        surv_var += estimate.survival_difference.se * estimate.survival_difference.se;
        rmst += estimate.rmst_difference.estimate;
        rmst_var += estimate.rmst_difference.se * estimate.rmst_difference.se;
    }
    let r = REPLICATES as f64;
    let (surv, rmst) = (surv / r, rmst / r);
    let surv_bound = 4.0 * (surv_var / r / r).sqrt() + 0.004;
    let rmst_bound = 4.0 * (rmst_var / r / r).sqrt() + 0.004;
    assert!(
        (surv - (s1 - s0)).abs() <= surv_bound,
        "survival difference {surv} vs {} (bound {surv_bound})",
        s1 - s0
    );
    assert!(
        (rmst - (r1 - r0)).abs() <= rmst_bound,
        "rmst difference {rmst} vs {} (bound {rmst_bound})",
        r1 - r0
    );
    // The truth is not a null: the effect is far larger than the bound.
    assert!((s1 - s0).abs() > 5.0 * surv_bound && (r1 - r0).abs() > 5.0 * rmst_bound);
}
