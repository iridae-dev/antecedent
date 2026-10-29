// SPDX-License-Identifier: MIT OR Apache-2.0
//! Serial dependence within a unit enters the panel difference-in-differences only through the
//! unit's change score, and the cluster score sandwich reads it.
//!
//! Each unit's two waves are AR(1) correlated with a large positive (or negative) correlation
//! around a unit effect. The product's standard error equals, to rounding, the cluster score
//! sandwich of the subject change scores written here from the definition, and it follows
//! the closed form `sqrt(2 (1 - rho) (1/n1 + 1/n0))` that the correlation implies for the
//! change-score variance `2 (1 - rho)`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;
use std::sync::Arc;

use antecedent::prelude::ExecutionContext;
use antecedent::{CausalError, PanelDidQuery, Study};
use antecedent_core::{DeclaredSamplingDesign, StreamDomain, VariableId};
use antecedent_data::TabularData;
use antecedent_kernels::standard_normal;

struct Panel {
    outcome: Vec<f64>,
    treated: Vec<bool>,
    post: Vec<bool>,
    subjects: Vec<Arc<str>>,
    clusters: Vec<Arc<str>>,
}

/// `units_per_arm` units per arm; each unit is its own cluster. The two waves of a unit are
/// `a + e1` and `a + rho e1 + sqrt(1 - rho^2) e2`, so the lag-one correlation of the shocks is
/// `rho`; the effect on a treated unit is 1.5.
fn panel(units_per_arm: usize, rho: f64, seed: u64) -> Panel {
    let mut rng = ExecutionContext::for_tests(seed).rng.stream_for(StreamDomain::Test, 0x7103);
    let mut out = Panel {
        outcome: vec![],
        treated: vec![],
        post: vec![],
        subjects: vec![],
        clusters: vec![],
    };
    for unit in 0..2 * units_per_arm {
        let active = unit >= units_per_arm;
        let level = 2.0 * standard_normal(&mut rng);
        let e1 = standard_normal(&mut rng);
        let e2 = rho * e1 + (1.0 - rho * rho).sqrt() * standard_normal(&mut rng);
        for (after, shock) in [(false, e1), (true, e2)] {
            out.subjects.push(Arc::from(format!("s{unit}")));
            out.clusters.push(Arc::from(format!("s{unit}")));
            out.treated.push(active);
            out.post.push(after);
            out.outcome.push(
                level
                    + 0.3 * f64::from(u8::from(after))
                    + shock
                    + if active && after { 1.5 } else { 0.0 },
            );
        }
    }
    out
}

fn run(
    panel: &Panel,
    sampling: Option<DeclaredSamplingDesign>,
) -> Result<antecedent::StudyResult, CausalError> {
    let query = PanelDidQuery::new(
        VariableId::from_raw(0),
        panel.treated.clone(),
        panel.post.clone(),
        panel.subjects.clone(),
        panel.clusters.clone(),
    );
    let data = TabularData::from_f64_columns([("outcome", panel.outcome.as_slice())]).unwrap();
    let builder = Study::tabular(data.clone()).query(query);
    let builder = match sampling {
        Some(design) => builder.sampling_design(design),
        None => builder,
    };
    let prepared = builder.build()?.prepare(&ExecutionContext::for_tests(78))?;
    prepared.estimate(&data, &ExecutionContext::for_tests(78))
}

/// Difference of mean change scores and the cluster score sandwich, from the definition.
fn oracle(panel: &Panel) -> (f64, f64) {
    let subjects = panel.outcome.len() / 2;
    let change = |j: usize| panel.outcome[2 * j + 1] - panel.outcome[2 * j];
    let group = |treated: bool| -> Vec<usize> {
        (0..subjects).filter(|&j| panel.treated[2 * j] == treated).collect()
    };
    let mean =
        |members: &[usize]| members.iter().map(|&j| change(j)).sum::<f64>() / members.len() as f64;
    let (treated, control) = (group(true), group(false));
    let (m1, m0) = (mean(&treated), mean(&control));
    let mut scores = BTreeMap::<&str, f64>::new();
    for (members, m, sign) in [(&treated, m1, 1.0), (&control, m0, -1.0)] {
        for &j in members {
            *scores.entry(panel.clusters[2 * j].as_ref()).or_default() +=
                sign * (change(j) - m) / members.len() as f64;
        }
    }
    let g = scores.len() as f64;
    (m1 - m0, (g / (g - 1.0) * scores.values().map(|v| v * v).sum::<f64>()).sqrt())
}

#[test]
fn cells_00_panel_did_standard_error_reads_serial_dependence_through_the_change_score() {
    for (rho, seed) in [(0.9, 31_u64), (-0.8, 32)] {
        let panel = panel(60, rho, seed);
        let result = run(&panel, None).unwrap();
        let fit = result.panel_did().unwrap();
        let (effect, se) = oracle(&panel);
        assert!((fit.effect - effect).abs() <= 1e-10 * (1.0 + effect.abs()), "rho {rho}");
        assert!(
            (fit.standard_error - se).abs() <= 1e-10 * (1.0 + se),
            "rho {rho}: {} vs {se}",
            fit.standard_error
        );
        // The change score of a unit has variance 2 (1 - rho) around its arm mean, so the
        // standard error is the closed form below up to sampling noise of the 60 units per arm:
        // strong positive correlation shrinks it and strong negative correlation inflates it.
        let closed_form = (2.0 * (1.0 - rho) * (2.0 / 60.0)).sqrt();
        assert!(
            (fit.standard_error / closed_form - 1.0).abs() < 0.3,
            "rho {rho}: {} vs {closed_form}",
            fit.standard_error
        );
    }
}
