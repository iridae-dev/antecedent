//! Twin equality of the joint cluster-randomized decision kernels and their compiled route.
//!
//! `antecedent::decision_evidence::complete_cluster_joint_contrast` and
//! `antecedent::decision_evidence::centered_fixed_cuped_totals` are the two kernels the compiled
//! joint-design program (`JointDesign`, entered by `compile_and_evidence_from_cluster`) runs for
//! a complete cluster-randomized source: the centered fixed adjustment of a declared outcome
//! coordinate, then the joint per-row arm means and their covariance. The assertions are on
//! `to_bits()` of every arm mean and covariance entry, the numbers the evidence carries forward.
//!
//! Neither kernel draws randomness, so the compiled route's context seed does not reach them.
//! The direct kernels publish no interval of their own (a decision derives its interval from
//! the means and covariance), so there is no calibration key to compare.
// SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(deprecated, reason = "the direct facade doors are the paths under comparison")]
#![allow(clippy::cast_precision_loss, reason = "small deterministic fixtures")]

use antecedent::decision_evidence::{
    Dependence, centered_fixed_cuped_totals, compile_and_evidence_from_cluster,
    complete_cluster_joint_contrast,
};
use antecedent_estimate::joint_design::{
    cluster_decision::{FixedClusterAdjustment, JointClusterRecipe},
    excursion::ExcursionOutcome,
    experiment::ExperimentExposure,
};

const ROWS_PER_CLUSTER: usize = 12;
const CLUSTERS: usize = 80;
const COMPONENTS: [(&str, &str); 3] = [("frequency", "orders"), ("value", "usd"), ("cost", "usd")];

/// A cluster-level shock shared by every outcome of the cluster, so the outcomes are dependent.
fn shock(cluster: usize, salt: f64) -> f64 {
    ((cluster as f64 + 1.0) * 12.9898 + salt).sin() * 0.5
}

struct Fixture {
    totals: [Vec<f64>; 3],
    pretreatment: Vec<f64>,
    assigned: Vec<bool>,
    sizes: Vec<usize>,
}

fn fixture() -> Fixture {
    let assigned: Vec<bool> = (0..CLUSTERS).map(|c| c % 2 == 0).collect();
    let mut totals = [vec![], vec![], vec![]];
    for (c, treated) in assigned.iter().enumerate() {
        let u = shock(c, 0.0);
        let (mut f, mut v, mut k) = (0.0, 0.0, 0.0);
        for r in 0..ROWS_PER_CLUSTER {
            let row_noise = ((c * 31 + r * 17) as f64 * 0.7).sin() * 0.2;
            f += 2.0 + if *treated { 0.3 } else { 0.0 } + 0.6 * u + row_noise;
            v += 10.0 + if *treated { 1.0 } else { 0.0 } + 2.0 * u + 0.5 * row_noise;
            k += 6.0 + 0.1 * shock(c, 3.0);
        }
        totals[0].push(f);
        totals[1].push(v);
        totals[2].push(k);
    }
    let pretreatment =
        (0..CLUSTERS).map(|c| ROWS_PER_CLUSTER as f64 * (1.0 + shock(c, 0.0))).collect();
    Fixture { totals, pretreatment, assigned, sizes: vec![ROWS_PER_CLUSTER; CLUSTERS] }
}

fn recipe(fixture: &Fixture, adjustments: Vec<FixedClusterAdjustment>) -> JointClusterRecipe {
    JointClusterRecipe {
        population: "active_customers".into(),
        treated_regime_id: "free_shipping".into(),
        control_regime_id: "standard_shipping".into(),
        horizon: 4,
        cluster_ids: (0..CLUSTERS).map(|i| format!("cluster-{i}")).collect(),
        cluster_sizes: fixture.sizes.clone(),
        assignment_law: "complete-randomization".into(),
        treated_clusters: fixture.assigned.iter().filter(|a| **a).count(),
        consistency: "consistency".into(),
        assignments: fixture.assigned.clone(),
        outcomes: COMPONENTS
            .iter()
            .zip(&fixture.totals)
            .map(|((name, units), values)| ExcursionOutcome {
                component_name: (*name).into(),
                variable_id: format!("outcome.{name}"),
                units: (*units).into(),
                values: values.clone(),
            })
            .collect(),
        adjustments,
        exposure: ExperimentExposure::NoInterference {
            assumption: "assumption.no_cross_customer_spillover".into(),
        },
    }
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|value| value.to_bits()).collect()
}

/// The compiled route's arm means and covariances, bit for bit against the direct contrast.
fn assert_compiled_equals(
    source: JointClusterRecipe,
    direct: &antecedent::decision_evidence::JointClusterContrast,
) {
    let ctx = antecedent_core::ExecutionContext::for_tests(0);
    let bound = compile_and_evidence_from_cluster(source, "exp-shipping", &ctx).unwrap();
    let evidence = bound.numerical_evidence();
    assert_eq!(bits(&evidence.treated_means), bits(&direct.treated_means), "treated means");
    assert_eq!(bits(&evidence.control_means), bits(&direct.control_means), "control means");
    let Dependence::Joint { treated, control } = &evidence.dependence else {
        panic!("the compiled cluster evidence keeps the arm covariance")
    };
    assert_eq!(bits(treated), bits(&direct.treated_covariance), "treated covariance");
    assert_eq!(bits(control), bits(&direct.control_covariance), "control covariance");
}

#[test]
fn compiled_cluster_source_reproduces_the_direct_joint_contrast_bit_for_bit() {
    let fixture = fixture();
    let slices: Vec<&[f64]> = fixture.totals.iter().map(Vec::as_slice).collect();
    let direct =
        complete_cluster_joint_contrast(&slices, &fixture.assigned, CLUSTERS * ROWS_PER_CLUSTER)
            .unwrap();
    assert_compiled_equals(recipe(&fixture, Vec::new()), &direct);
}

#[test]
fn compiled_centered_adjustment_reproduces_the_direct_centered_totals_bit_for_bit() {
    let fixture = fixture();
    let coefficient = 0.4;
    let adjusted = centered_fixed_cuped_totals(
        &fixture.totals[0],
        &fixture.pretreatment,
        &fixture.sizes,
        coefficient,
    )
    .unwrap();
    let slices: Vec<&[f64]> =
        [adjusted.as_slice(), fixture.totals[1].as_slice(), fixture.totals[2].as_slice()].into();
    let direct =
        complete_cluster_joint_contrast(&slices, &fixture.assigned, CLUSTERS * ROWS_PER_CLUSTER)
            .unwrap();
    let adjustment = FixedClusterAdjustment {
        component_name: "frequency".into(),
        covariate: "pre_frequency".into(),
        coefficient,
        pretreatment_totals: fixture.pretreatment.clone(),
        centered: true,
        preassignment: "coefficient and covariate fixed before assignment".into(),
    };
    assert_compiled_equals(recipe(&fixture, vec![adjustment]), &direct);
    // The fixture's pretreatment totals vary by cluster, so the adjustment moves the treated
    // mean: the comparison above is not the unadjusted contrast compared with itself.
    let unadjusted = complete_cluster_joint_contrast(
        &fixture.totals.iter().map(Vec::as_slice).collect::<Vec<_>>(),
        &fixture.assigned,
        CLUSTERS * ROWS_PER_CLUSTER,
    )
    .unwrap();
    assert_ne!(direct.treated_means[0].to_bits(), unadjusted.treated_means[0].to_bits());
}

#[test]
fn centered_totals_restore_the_original_scale_at_the_population_pretreatment_mean() {
    // Independent analytic check: when every cluster's pretreatment total equals its size times
    // the population mean, the centered adjustment is the identity.
    let sizes = [3_usize, 5, 2, 6];
    let rows = sizes.iter().sum::<usize>() as f64;
    let per_row = 1.5;
    let pretreatment: Vec<f64> = sizes.iter().map(|n| *n as f64 * per_row).collect();
    assert_eq!((pretreatment.iter().sum::<f64>() / rows).to_bits(), per_row.to_bits());
    let outcome = [4.0, 7.5, 2.25, 9.0];
    let adjusted = centered_fixed_cuped_totals(&outcome, &pretreatment, &sizes, 0.75).unwrap();
    assert_eq!(bits(&adjusted), bits(&outcome));
}
