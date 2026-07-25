#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::too_many_lines,
    clippy::manual_map,
    clippy::match_wildcard_for_single_variants,
    clippy::doc_markdown,
    clippy::map_unwrap_or
)]
//! Survey prior bank: catalog → compose → analyze target.
//!
//! Two fake survey posteriors tagged by product/context, ranked by
//! caller-supplied similarity, composed with power-prior weights, then
//! transferred into a new target survey.
//!
//! Run: `cargo run -p antecedent --example prior_bank_surveys`

use std::collections::BTreeMap;
use std::sync::Arc;

use antecedent::RefuteSuite;
use antecedent::io::encode_causal_posterior_bytes;
use antecedent::prelude::*;
use antecedent_core::{CausalRng, MeasurementSpec, RoleHint, SmallRoleSet, ValueType};
use antecedent_data::{Float64Column, OwnedColumn, OwnedColumnarStorage, ValidityBitmap};
use antecedent_graph::DenseNodeId;
use antecedent_io::{
    CompatibilityReport, DesignVariableRole, DesignVariableSummary, EstimandFingerprint,
    PriorCatalog, PriorSourceMeta, PriorSourceRef, TargetDesign,
};
use antecedent_prob::{
    ExternalPriorSource, ExternalPriorWeight, GaussianCoefficientPrior, PriorSet, PriorSpec,
};
use antecedent_validate::{ConflictPolicy, ConflictSignals, apply_conflict_and_compose};

fn survey(n: usize, seed: u64, ate: f64) -> (TabularData, Dag, AverageEffectQuery) {
    let mut rng = CausalRng::from_seed(seed);
    let mut z = vec![0.0; n];
    let mut t = vec![0.0; n];
    let mut y = vec![0.0; n];
    for i in 0..n {
        let u1 = rng.next_f64().max(1e-12);
        let u2 = rng.next_f64();
        let zi = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        let e1 = (-2.0 * rng.next_f64().max(1e-12).ln()).sqrt()
            * (2.0 * std::f64::consts::PI * rng.next_f64()).cos();
        let ti = if zi + e1 > 0.0 { 1.0 } else { 0.0 };
        let e2 = (-2.0 * rng.next_f64().max(1e-12).ln()).sqrt()
            * (2.0 * std::f64::consts::PI * rng.next_f64()).cos()
            * 0.35;
        z[i] = zi;
        t[i] = ti;
        y[i] = ate * ti + zi + e2;
    }

    let mut b = CausalSchemaBuilder::new();
    b.add_variable(
        "z",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::Context),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    b.add_variable(
        "t",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    b.add_variable(
        "y",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    let schema = b.build().unwrap();
    let cols = vec![
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(0), Arc::from(z), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(1), Arc::from(t), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(2), Arc::from(y), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
    ];
    let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
    let mut dag = Dag::with_variables(3);
    dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
    dag.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(1), VariableId::from_raw(2));
    (TabularData::new(storage), dag, query)
}

fn fit_artifact(
    data: TabularData,
    dag: Dag,
    query: AverageEffectQuery,
    seed: u64,
) -> Result<(Vec<u8>, f64), CausalError> {
    let result = CausalAnalysis::builder()
        .data(data)
        .graph(dag)
        .query(query)
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(96)))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()?
        .run(&ExecutionContext::for_tests(seed))?;
    let post = result.posterior.as_ref().expect("posterior");
    let mean = post.summaries.mean[post.effect_column().unwrap()];
    let bytes = encode_causal_posterior_bytes(post, "survey")?;
    Ok((bytes, mean))
}

fn main() -> Result<(), CausalError> {
    let (data_a, dag, query) = survey(160, 1, 2.0);
    let (data_b, _, _) = survey(160, 2, 1.5);
    let (data_t, _, _) = survey(180, 3, 1.8);

    let (art_a, mean_a) = fit_artifact(data_a, dag.clone(), query.clone(), 11)?;
    let (art_b, mean_b) = fit_artifact(data_b, dag.clone(), query.clone(), 12)?;

    let ate = EstimandFingerprint::new("ate", "t", "y");
    let design = vec![
        DesignVariableSummary::new("t", DesignVariableRole::Treatment),
        DesignVariableSummary::new("y", DesignVariableRole::Outcome),
        DesignVariableSummary::new("z", DesignVariableRole::Covariate),
    ];
    let tags_a = BTreeMap::from([
        ("product".into(), "widget".into()),
        ("context".into(), "launch".into()),
        ("population".into(), "field".into()),
    ]);
    let tags_b = BTreeMap::from([
        ("product".into(), "widget".into()),
        ("context".into(), "retention".into()),
        ("population".into(), "field".into()),
    ]);

    let sources = vec![
        PriorSourceRef::with_bytes(
            PriorSourceMeta::new("survey_launch", ate.clone(), "NonparametricallyIdentified")
                .with_tags(tags_a)
                .with_design(design.clone()),
            art_a,
        ),
        PriorSourceRef::with_bytes(
            PriorSourceMeta::new("survey_retention", ate.clone(), "NonparametricallyIdentified")
                .with_tags(tags_b)
                .with_design(design),
            art_b,
        ),
    ];
    let catalog = PriorCatalog::from_sources(sources);
    let target = TargetDesign::new(ate, ["z", "t", "y"]).with_tags(BTreeMap::from([
        ("product".into(), "widget".into()),
        ("population".into(), "field".into()),
    ]));
    let reports = catalog.filter_compatible(&target);
    let similarity = [("survey_launch".into(), 0.85), ("survey_retention".into(), 0.55)];
    let ranked = catalog.rank(&reports, &similarity);
    let accepted: Vec<&str> = ranked
        .iter()
        .filter_map(|r| match r {
            CompatibilityReport::Compatible { artifact_id }
            | CompatibilityReport::Partial { artifact_id, .. } => Some(artifact_id.as_str()),
            _ => None,
        })
        .collect();
    assert!(!accepted.is_empty(), "expected at least one compatible prior source");

    let w_launch = 0.85;
    let w_ret = 0.55;
    let w_sum = w_launch + w_ret;
    let mut src_a = PriorSet::new();
    src_a.push(PriorSpec::GaussianCoefficients(GaussianCoefficientPrior {
        mean: Arc::from(vec![0.0, mean_a, 0.0]),
        variance: Arc::from(vec![1.0, 0.25, 1.0]),
    }));
    let mut src_b = PriorSet::new();
    src_b.push(PriorSpec::GaussianCoefficients(GaussianCoefficientPrior {
        mean: Arc::from(vec![0.0, mean_b, 0.0]),
        variance: Arc::from(vec![1.0, 0.25, 1.0]),
    }));
    let ext_sources = Arc::<[ExternalPriorSource]>::from(vec![
        ExternalPriorSource {
            id: Arc::from("survey_launch"),
            prior: src_a,
            weight: ExternalPriorWeight::power_mixture(1.0, 0.6 * w_launch / w_sum)
                .expect("weight"),
        },
        ExternalPriorSource {
            id: Arc::from("survey_retention"),
            prior: src_b,
            weight: ExternalPriorWeight::power_mixture(1.0, 0.6 * w_ret / w_sum).expect("weight"),
        },
    ]);
    let baseline = PriorSet::weakly_informative(3);
    let policy = ConflictPolicy::try_new(0.05, 1.0).expect("policy");
    let (composed, _) = apply_conflict_and_compose(
        &ext_sources,
        &baseline,
        &policy,
        &[
            ConflictSignals { p_value: Some(0.4), kl: Some(0.05) },
            ConflictSignals { p_value: Some(0.3), kl: Some(0.1) },
        ],
    )
    .expect("compose");

    // Clear offline conflict for fit; α' already applied.
    let prior_for_fit = composed.clone();

    let target_result = CausalAnalysis::builder()
        .data(data_t)
        .graph(dag)
        .query(query)
        .inference(InferenceMode::Bayesian(
            BayesianConfig::conjugate().n_draws(96).prior_from_composed(
                Arc::clone(&ext_sources),
                prior_for_fit,
                None,
            ),
        ))
        .refute(RefuteSuite::Full)
        .bootstrap_replicates(0)
        .build()?
        .run(&ExecutionContext::for_tests(13))?;

    let post = target_result.posterior.as_ref().expect("target posterior");
    let mean = post.summaries.mean[post.effect_column().unwrap()];
    let sd = post.summaries.sd[post.effect_column().unwrap()];
    assert!(mean.is_finite());
    let sens = post.prior_sensitivity.as_ref().expect("prior sensitivity");
    assert!(!sens.alphas.is_empty());
    let ppc = target_result
        .predictive_checks
        .iter()
        .find(|c| matches!(c.kind, antecedent::validate::PredictiveCheckKind::Prior))
        .expect("prior ppc");

    println!("accepted sources: {accepted:?}");
    println!(
        "alphas_requested={:?} alphas_applied={:?}",
        composed.alphas_requested, composed.alphas_applied
    );
    println!("prior_ppc p={:.3} observed={:.3}", ppc.p_value, ppc.observed);
    println!("target effect_mean={mean:.4} sd={sd:.4}");
    println!("alpha_sensitivity alphas={:?} means={:?}", sens.alphas, sens.effect_means);
    Ok(())
}
