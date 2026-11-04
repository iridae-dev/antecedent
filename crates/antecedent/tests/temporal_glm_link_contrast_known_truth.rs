//! Known-truth check of `temporal.glm_link_contrast` on every fixed `TemporalDag` panel
//! coordinate it can estimate: pulse and one-step sustained effects, explicit and accepted
//! structure, and every validation suite.
//!
//! The panel is a two-group design with a balanced +/-1 covariate `c` held at its observed lag: `c`
//! splits every group and every group's responders equally, so the logistic score equations are
//! solved by a zero `c` coefficient and the observed group proportions. The information matrix
//! is block diagonal and the contrast does not depend on `c`, so the standard error is the
//! saturated one. With 30 of 100 untreated and 60 of 100 treated subjects responding, the
//! probability-scale contrast is `0.6 - 0.3` and its Fisher delta-method standard error is
//! `sqrt(p1 (1 - p1) / n1 + p0 (1 - p0) / n0)`, both in closed form from the counts.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::{AcceptedGraph, EstimatorId, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, ExecutionContext, Lag, TemporalEffectQuery, TemporalPolicy, VariableId,
};
use antecedent_data::{PanelData, PanelUnit, TimeSeriesData};
use antecedent_graph::{TemporalDag, ensure_lagged};

const GROUP: usize = 100;
const UNTREATED_RESPONDERS: usize = 30;
const TREATED_RESPONDERS: usize = 60;
const STEPS: usize = 3;

/// Each unit: `x = [x0, 0, 0]`, `y = [0, y1, 0]`; the decision row is the second one.
fn panel() -> PanelData {
    let mut members = Vec::new();
    for (treated, responders) in [(0.0, UNTREATED_RESPONDERS), (1.0, TREATED_RESPONDERS)] {
        for index in 0..GROUP {
            let mut x = vec![0.0; STEPS];
            let mut c = vec![0.0; STEPS];
            let mut y = vec![0.0; STEPS];
            x[0] = treated;
            c[0] = if index % 2 == 0 { 1.0 } else { -1.0 };
            y[1] = f64::from(u8::from(index < responders));
            let series = TimeSeriesData::from_f64_columns(
                [("x", x.as_slice()), ("c", c.as_slice()), ("y", y.as_slice())],
                1,
            )
            .unwrap();
            members.push(PanelUnit { unit_id: u32::try_from(members.len()).unwrap(), series });
        }
    }
    PanelData::try_new(Arc::from(members)).unwrap()
}

/// `c_{t-1} -> x_{t-1}`, `c_{t-1} -> y_t`, `x_{t-1} -> y_t`.
fn graph() -> TemporalDag {
    let mut g = TemporalDag::empty();
    let c1 = ensure_lagged(&mut g, VariableId::from_raw(1), Lag::from_raw(1)).unwrap();
    let x1 = ensure_lagged(&mut g, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let y0 = ensure_lagged(&mut g, VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
    g.insert_directed(c1, x1).unwrap();
    g.insert_directed(c1, y0).unwrap();
    g.insert_directed(x1, y0).unwrap();
    g
}

fn query(sustained: bool) -> CausalQuery {
    let base = TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(2), 1.0)
        .with_horizon_steps(1)
        .with_max_history_lag(Some(1));
    CausalQuery::TemporalEffect(if sustained {
        base.with_policy(TemporalPolicy::sustained(-1, -1))
    } else {
        base.with_policy(TemporalPolicy::pulse(-1))
    })
}

#[test]
fn glm_link_contrast_returns_the_group_proportion_contrast_on_every_fixed_dag_coordinate() {
    #[allow(clippy::cast_precision_loss, reason = "small exact counts")]
    let (p0, p1, n) = (
        UNTREATED_RESPONDERS as f64 / GROUP as f64,
        TREATED_RESPONDERS as f64 / GROUP as f64,
        GROUP as f64,
    );
    let truth = p1 - p0;
    let truth_se = (p1 * (1.0 - p1) / n + p0 * (1.0 - p0) / n).sqrt();
    for accepted in [false, true] {
        for sustained in [false, true] {
            for suite in [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full] {
                let label = format!("accepted={accepted} sustained={sustained} {suite:?}");
                let base = Study::panel(panel());
                let base = if accepted {
                    base.graph(AcceptedGraph::temporal_dag(graph()))
                } else {
                    base.graph(graph())
                };
                let result = base
                    .query(query(sustained))
                    .estimator(EstimatorId::TemporalGlmLinkContrast)
                    .inference(InferenceMode::Frequentist)
                    .refute(suite)
                    .bootstrap_replicates(0)
                    .build()
                    .unwrap()
                    .run(&ExecutionContext::for_tests(4))
                    .unwrap();
                let estimate = result.estimate().as_effect().unwrap();
                assert!((estimate.ate - truth).abs() < 1e-8, "{label}: {}", estimate.ate);
                assert!(
                    (estimate.se_analytic() - truth_se).abs() < 1e-8,
                    "{label}: se {} vs {truth_se}",
                    estimate.se_analytic()
                );
                assert!(result.posterior.is_none(), "{label}: frequentist route");
            }
        }
    }
}
