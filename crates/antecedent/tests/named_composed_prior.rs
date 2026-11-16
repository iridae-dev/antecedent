//! A composed external prior with named coordinates, through the study.
//!
//! The coordinates map by name onto the coefficient columns of the Bayesian g-computation: a
//! prior whose coordinates are shuffled and carry one the design lacks gives the posterior of the
//! positional prior in design order, which is the closed form `(X'X + V0^-1)^-1 (X'y + V0^-1 m)`
//! for the posterior mean of the coefficients. A prior none of whose names is a design column is
//! refused with the next attempt, and a power-prior discount selected from the data composes into
//! an ordinary prior whose provenance names the selection.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::{BayesianConfig, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, MeasurementSpec, RoleHint,
    SmallRoleSet, ValueType, VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
};
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_prob::{
    ExternalPriorSource, ExternalPriorWeight, GaussianCoefficientPrior, PriorSet, PriorSpec,
    compose_external_priors,
};

const ROWS: usize = 200;

fn names(items: &[&str]) -> Vec<Arc<str>> {
    items.iter().map(|name| Arc::from(*name)).collect()
}

fn data_and_graph() -> (TabularData, Dag, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut b = CausalSchemaBuilder::new();
    for (name, hint) in [
        ("t", RoleHint::TreatmentCandidate),
        ("y", RoleHint::OutcomeCandidate),
        ("z", RoleHint::Context),
    ] {
        b.add_variable(
            name,
            ValueType::Continuous,
            SmallRoleSet::from_hint(hint),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
    }
    let schema = b.build().unwrap();
    let t: Vec<f64> = (0..ROWS).map(|i| ((i * 7) % 3 == 0) as u8 as f64).collect();
    let z: Vec<f64> = (0..ROWS).map(|i| (i as f64 * 0.37).sin() + 0.01 * i as f64).collect();
    let y: Vec<f64> = (0..ROWS)
        .map(|i| 1.0 + 2.0 * t[i] + 0.8 * z[i] + 0.2 * ((i as f64 * 12.9898).sin()))
        .collect();
    let validity = ValidityBitmap::all_valid(ROWS);
    let column = |id: u32, values: &[f64]| {
        OwnedColumn::Float64(
            Float64Column::new(
                VariableId::from_raw(id),
                Arc::from(values.to_vec()),
                validity.clone(),
            )
            .unwrap(),
        )
    };
    let cols = vec![column(0, &t), column(1, &y), column(2, &z)];
    let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
    let mut dag = Dag::with_variables(3);
    for (a, c) in [(2, 0), (2, 1), (0, 1)] {
        dag.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(c)).unwrap();
    }
    (TabularData::new(storage), dag, t, y, z)
}

fn gaussian(mean: Vec<f64>, variance: Vec<f64>) -> PriorSet {
    let mut prior = PriorSet::new();
    prior.push(PriorSpec::GaussianCoefficients(GaussianCoefficientPrior {
        mean: Arc::from(mean),
        variance: Arc::from(variance),
    }));
    prior
}

fn source(id: &str, mean: Vec<f64>, variance: Vec<f64>, alpha: f64) -> ExternalPriorSource {
    ExternalPriorSource {
        id: Arc::from(id),
        prior: gaussian(mean, variance),
        weight: ExternalPriorWeight::power(alpha).unwrap(),
        ess: None,
    }
}

/// The posterior mean of the effect under the study with `sources` composed as the prior.
fn effect_mean(
    sources: Vec<ExternalPriorSource>,
    coordinate_names: Option<Vec<Arc<str>>>,
) -> Result<f64, antecedent::CausalError> {
    let (data, dag, ..) = data_and_graph();
    let n = sources[0].prior.gaussian_coefficients().unwrap().len();
    let baseline = gaussian(vec![0.0; n], vec![100.0; n]);
    let sources = Arc::<[ExternalPriorSource]>::from(sources);
    let mut composed = compose_external_priors(&sources, &baseline).unwrap();
    if let Some(names) = coordinate_names {
        composed = composed.with_coefficient_names(names).unwrap();
    }
    let result = Study::tabular(data)
        .graph(dag)
        .query(AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1)))
        .inference(InferenceMode::Bayesian(
            BayesianConfig::conjugate().n_draws(4000).prior_from_composed(sources, composed, None),
        ))
        .refute(RefuteSuite::None)
        .build()?
        .run(&ExecutionContext::for_tests(3))?;
    let post = result.posterior.as_ref().expect("posterior");
    Ok(post.summaries.mean[post.effect_column().unwrap()])
}

#[test]
fn a_named_prior_maps_by_name_onto_the_design_and_equals_the_positional_prior() {
    // Design order: intercept, coef_t, coef_z.
    let positional = effect_mean(
        vec![
            source("a", vec![0.8, 1.7, 0.9], vec![0.5, 0.4, 0.3], 1.0),
            source("b", vec![1.2, 2.3, 1.1], vec![0.6, 0.5, 0.4], 0.5),
        ],
        Some(names(&["intercept", "coef_t", "coef_z"])),
    )
    .unwrap();
    // The same coordinates shuffled, with one the design has no column for.
    let shuffled = effect_mean(
        vec![
            source("a", vec![0.9, 7.0, 0.8, 1.7], vec![0.3, 3.0, 0.5, 0.4], 1.0),
            source("b", vec![1.1, 7.0, 1.2, 2.3], vec![0.4, 3.0, 0.6, 0.5], 0.5),
        ],
        Some(names(&["coef_z", "extra", "intercept", "coef_t"])),
    )
    .unwrap();
    assert!((positional - shuffled).abs() <= 1e-9, "{positional} vs {shuffled}");
    // The composed moments are the precision sum; the effect is its ridge-type closed form.
    let (_, _, t, y, z) = data_and_graph();
    let (m, v0) = {
        let composed = compose_external_priors(
            &[
                source("a", vec![0.8, 1.7, 0.9], vec![0.5, 0.4, 0.3], 1.0),
                source("b", vec![1.2, 2.3, 1.1], vec![0.6, 0.5, 0.4], 0.5),
            ],
            &gaussian(vec![0.0; 3], vec![100.0; 3]),
        )
        .unwrap();
        let c = composed.prior.gaussian_coefficients().unwrap();
        (c.mean.to_vec(), c.variance.to_vec())
    };
    let closed = ridge_coefficient(&t, &z, &y, &m, &v0, 1);
    assert!((positional - closed).abs() < 0.02, "{positional} vs closed form {closed}");
}

/// Coefficient `which` of `(X'X + V0^-1)^-1 (X'y + V0^-1 m)` for `X = [1, t, z]`, by Gaussian
/// elimination.
fn ridge_coefficient(t: &[f64], z: &[f64], y: &[f64], m: &[f64], v0: &[f64], which: usize) -> f64 {
    let rows: Vec<[f64; 3]> = (0..t.len()).map(|i| [1.0, t[i], z[i]]).collect();
    let mut a = [[0.0; 4]; 3];
    for p in 0..3 {
        for q in 0..3 {
            a[p][q] = rows.iter().map(|r| r[p] * r[q]).sum::<f64>()
                + if p == q { 1.0 / v0[p] } else { 0.0 };
        }
        a[p][3] = rows.iter().zip(y).map(|(r, y)| r[p] * y).sum::<f64>() + m[p] / v0[p];
    }
    for col in 0..3 {
        let pivot = (col..3).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs())).unwrap();
        a.swap(col, pivot);
        for row in 0..3 {
            if row != col {
                let factor = a[row][col] / a[col][col];
                for k in col..4 {
                    a[row][k] -= factor * a[col][k];
                }
            }
        }
    }
    a[which][3] / a[which][which]
}

#[test]
fn a_named_prior_with_no_coordinate_of_the_design_is_refused_with_the_next_attempt() {
    let refused = effect_mean(
        vec![source("a", vec![0.0, 1.0, 2.0], vec![1.0, 1.0, 1.0], 1.0)],
        Some(names(&["a", "b", "c"])),
    )
    .unwrap_err();
    assert!(refused.to_string().contains("coefficient_names"), "{refused}");
}

#[test]
fn a_selected_power_prior_discount_is_part_of_the_composed_prior() {
    use antecedent_prob::{
        AlphaCriterion, AlphaSelectionOptions, LinearTarget, select_linear_prior,
    };
    let (_, _, t, y, z) = data_and_graph();
    let design: Vec<f64> = (0..ROWS).flat_map(|i| [1.0, t[i], z[i]]).collect();
    let sources = vec![
        source("a", vec![0.8, 1.7, 0.9], vec![0.5, 0.4, 0.3], 1.0),
        source("b", vec![1.2, 2.3, 1.1], vec![0.6, 0.5, 0.4], 1.0),
    ];
    let mut baseline = gaussian(vec![0.0; 3], vec![100.0; 3]);
    baseline.push(PriorSpec::KnownResidualVariance(0.05));
    let mut options = AlphaSelectionOptions::new(AlphaCriterion::EmpiricalBayes);
    options.grid = vec![0.0, 0.25, 0.5, 0.75, 1.0];
    options.discrepancy_bound = Some(0.5);
    let functional = [0.0, 1.0, 0.0];
    let target =
        LinearTarget { x: &design, y: &y, n_rows: ROWS, n_cols: 3, functional: &functional };
    let (selected, report) = select_linear_prior(&sources, &baseline, target, &options).unwrap();
    // The composition at the selected alpha is the ordinary composition at that alpha.
    let alphas = vec![report.selected_alpha; 2];
    let direct =
        antecedent_prob::compose_external_priors_with_alphas(&sources, &alphas, &alphas, &baseline)
            .unwrap();
    assert_eq!(selected.prior.gaussian_coefficients(), direct.prior.gaussian_coefficients());
    // The selection and its interval semantics are in the prior's provenance.
    let id = report.provenance_id();
    assert!(id.starts_with("power_prior_alpha_selected:criterion=empirical_bayes"), "{id}");
    assert!(id.contains("interval=bias_aware") && id.contains("discrepancy_bound=0.5"), "{id}");
    assert!(selected.prior.restrictions.iter().any(|r| &*r.id == id.as_str()));
    assert_eq!(report.interval_semantics(), "bias_aware");
}

#[test]
fn a_named_prior_equal_to_the_design_is_used_unchanged() {
    let composed = compose_external_priors(
        &[source("a", vec![0.8, 1.7, 0.9], vec![0.5, 0.4, 0.3], 1.0)],
        &gaussian(vec![0.0; 3], vec![100.0; 3]),
    )
    .unwrap();
    let design = names(&["intercept", "coef_t", "coef_z"]);
    let mapped = antecedent::inference::map_named_composed(
        &composed,
        &design,
        &design,
        &gaussian(vec![0.0; 3], vec![100.0; 3]),
    )
    .unwrap();
    assert_eq!(mapped, composed.prior);
}

#[test]
fn the_name_mapping_is_recorded_as_a_prior_restriction() {
    let composed = compose_external_priors(
        &[source("a", vec![0.8, 1.7, 0.9, 5.0], vec![0.5, 0.4, 0.3, 2.0], 1.0)],
        &gaussian(vec![0.0; 4], vec![100.0; 4]),
    )
    .unwrap();
    let source_names = names(&["intercept", "coef_t", "coef_z", "extra"]);
    let design = names(&["intercept", "coef_t", "coef_z", "coef_w"]);
    let mapped = antecedent::inference::map_named_composed(
        &composed,
        &source_names,
        &design,
        &gaussian(vec![0.0; 4], vec![9.0; 4]),
    )
    .unwrap();
    let restriction = mapped
        .restrictions
        .iter()
        .find(|r| &*r.id == "external_prior_name_mapping")
        .expect("the mapping is a restriction");
    assert!(restriction.description.contains("[coef_w]"), "{}", restriction.description);
    assert!(restriction.description.contains("[extra]"), "{}", restriction.description);
}

mod conflict {
    use antecedent::inference::{
        ExternalComposeSpec, declared_design_baseline, project_external_sources,
    };
    use antecedent::validate::{ConflictPolicy, ConflictSignals, apply_conflict_and_compose};

    use super::*;

    fn named_spec(conflict: Option<ConflictPolicy>) -> ExternalComposeSpec {
        let sources = Arc::<[ExternalPriorSource]>::from(vec![
            source("a", vec![0.9, 7.0, 0.8, 1.7], vec![0.3, 3.0, 0.5, 0.4], 1.0),
            source("b", vec![1.1, 7.0, 1.2, 2.3], vec![0.4, 3.0, 0.6, 0.5], 0.5),
        ]);
        let baseline = gaussian(vec![0.0; 4], vec![100.0; 4]);
        let composed = compose_external_priors(&sources, &baseline)
            .unwrap()
            .with_coefficient_names(names(&["coef_z", "extra", "intercept", "coef_t"]))
            .unwrap();
        ExternalComposeSpec {
            sources,
            composed,
            conflict_policy: conflict,
            baseline: Some(baseline),
        }
    }

    /// The conflict re-shrink is per source and per coordinate, so mapping each source onto the
    /// design by name first gives exactly the result of the positional sources in design order.
    #[test]
    fn the_conflict_shrink_of_a_reindexed_prior_equals_the_positional_result() {
        let policy = ConflictPolicy::try_new(0.05, 1.0).unwrap();
        let spec = named_spec(Some(policy));
        let design = names(&["intercept", "coef_t", "coef_z"]);
        let fill = declared_design_baseline(&BayesianConfig::conjugate(), 3);
        let (projected, baseline) = project_external_sources(
            &spec,
            spec.composed.coefficient_names.as_ref().unwrap(),
            &design,
            &fill,
        )
        .unwrap();
        // Hand-reordered positional sources: (intercept, coef_t, coef_z) from (z, extra, i, t).
        let manual = vec![
            source("a", vec![0.8, 1.7, 0.9], vec![0.5, 0.4, 0.3], 1.0),
            source("b", vec![1.2, 2.3, 1.1], vec![0.6, 0.5, 0.4], 0.5),
        ];
        for (p, m) in projected.iter().zip(&manual) {
            assert_eq!(p.prior.gaussian_coefficients(), m.prior.gaussian_coefficients());
        }
        let signals = [
            ConflictSignals { p_value: Some(0.001), kl: Some(2.0) },
            ConflictSignals { p_value: Some(0.5), kl: Some(0.0) },
        ];
        let manual_baseline = gaussian(vec![0.0; 3], vec![100.0; 3]);
        let (a, sa) = apply_conflict_and_compose(&projected, &baseline, &policy, &signals).unwrap();
        let (b, sb) =
            apply_conflict_and_compose(&manual, &manual_baseline, &policy, &signals).unwrap();
        assert_eq!(sa.alphas_applied, sb.alphas_applied);
        assert!(sa.alphas_applied[0] < sa.alphas_requested[0], "the conflicting source shrinks");
        assert_eq!(a.prior.gaussian_coefficients(), b.prior.gaussian_coefficients());
    }

    #[test]
    fn a_source_lacking_a_design_column_cannot_be_shrunk_over_the_design_and_says_so() {
        let spec = named_spec(None);
        let design = names(&["intercept", "coef_t", "coef_z", "coef_w"]);
        let fill = declared_design_baseline(&BayesianConfig::conjugate(), 4);
        let refused = project_external_sources(
            &spec,
            spec.composed.coefficient_names.as_ref().unwrap(),
            &design,
            &fill,
        )
        .unwrap_err();
        assert!(refused.to_string().contains("[coef_w]"), "{refused}");
    }
}
