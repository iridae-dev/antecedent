use super::gcomp::{add_modifier_mean_uncertainty, random_intercept_gls_whiten};
use super::*;
use crate::gcomp::GcompContrast;
use crate::overlap::OverlapPolicy;
use crate::serial_dependence::SerialDependence;
use antecedent_core::IdentificationStatus;
use antecedent_core::{
    Assumption, AssumptionSet, AssumptionSource, AverageEffectQuery, ExecutionContext,
};
use antecedent_core::{
    CausalSchemaBuilder, MeasurementSpec, RoleHint, SmallRoleSet, ValueType, VariableId,
};
use antecedent_data::column::{Float64Column, ValidityBitmap};
use antecedent_data::{OwnedColumn, OwnedColumnarStorage, TabularData};
use antecedent_expr::{ExprId, IdentifiedEstimand};
use antecedent_prob::InferenceDiagnostics;
use antecedent_prob::{
    BayesLikelihood, GaussianCoefficientPrior, PosteriorQuantityKind, PriorSet, PriorSpec,
};
use antecedent_stats::{CompiledDesign, GlmFamily};

#[test]
fn gaussian_gcomp_contrast_is_treatment_coef_times_level_gap() {
    let beta = [1.0_f64, 2.5, -0.5];
    let matrix = [
        1.0, 1.0, 1.0, // intercept
        0.0, 1.0, 0.0, // treatment (overwritten by do())
        0.2, 0.4, 0.6, // covariate
    ];
    let ate = GcompContrast {
        family: GlmFamily::GaussianIdentity,
        x_colmajor: &matrix,
        nrows: 3,
        ncols: 3,
        t_col: 1,
        active: 1.0,
        control: 0.0,
    }
    .mean_contrast(&beta);
    assert!((ate - 2.5).abs() < 1e-15);
}

fn linear_scm_table(n: usize) -> (TabularData, VariableId, VariableId, VariableId) {
    linear_scm_table_with_noise(n, 0.0, 0)
}

fn linear_scm_table_shifted(
    n: usize,
    noise: f64,
) -> (TabularData, VariableId, VariableId, VariableId) {
    linear_scm_table_with_noise(n, noise, 17)
}

fn linear_scm_table_with_noise(
    n: usize,
    noise: f64,
    seed_off: usize,
) -> (TabularData, VariableId, VariableId, VariableId) {
    let mut b = CausalSchemaBuilder::new();
    b.add_variable(
        "Z",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::Context),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    b.add_variable(
        "T",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    b.add_variable(
        "Y",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    let schema = b.build().unwrap();
    let z = VariableId::from_raw(0);
    let t = VariableId::from_raw(1);
    let y = VariableId::from_raw(2);
    let mut zv = vec![0.0; n];
    let mut tv = vec![0.0; n];
    let mut yv = vec![0.0; n];
    for i in 0..n {
        zv[i] = (i as f64) * 0.1;
        tv[i] = if i % 2 == 0 { 1.0 } else { 0.0 };
        let e = if noise == 0.0 { 0.0 } else { (((i + seed_off) % 7) as f64 - 3.0) * noise };
        yv[i] = 2.0 * tv[i] + 0.5 * zv[i] + e;
    }
    let validity = ValidityBitmap::all_valid(n);
    let cols = vec![
        OwnedColumn::Float64(Float64Column::new(z, Arc::from(zv), validity.clone()).unwrap()),
        OwnedColumn::Float64(Float64Column::new(t, Arc::from(tv), validity.clone()).unwrap()),
        OwnedColumn::Float64(Float64Column::new(y, Arc::from(yv), validity).unwrap()),
    ];
    let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
    (TabularData::new(storage), t, y, z)
}

#[test]
fn gaussian_conditional_design_supports_multiple_declared_modifiers() {
    let n = 12;
    let mut schema_builder = CausalSchemaBuilder::new();
    for (name, role) in [
        ("W1", RoleHint::Context),
        ("W2", RoleHint::Context),
        ("T", RoleHint::TreatmentCandidate),
        ("Y", RoleHint::OutcomeCandidate),
    ] {
        schema_builder
            .add_variable(
                name,
                ValueType::Continuous,
                SmallRoleSet::from_hint(role),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
    }
    let schema = schema_builder.build().unwrap();
    let w1 = VariableId::from_raw(0);
    let w2 = VariableId::from_raw(1);
    let t = VariableId::from_raw(2);
    let y = VariableId::from_raw(3);
    let first: Vec<_> = (0..n).map(|i| i as f64).collect();
    let second: Vec<_> = (0..n).map(|i| (i as f64) * 2.0 - 3.0).collect();
    let treatment: Vec<_> = (0..n).map(|i| f64::from(i % 2 == 0)).collect();
    let outcome: Vec<_> = (0..n)
        .map(|i| {
            1.0 + treatment[i] * (2.0 + 0.5 * (first[i] - 5.5) - 0.25 * (second[i] - 8.0))
                + 0.2 * first[i]
                + 0.1 * second[i]
        })
        .collect();
    let validity = ValidityBitmap::all_valid(n);
    let cols = vec![
        OwnedColumn::Float64(
            Float64Column::new(w1, Arc::from(first.clone()), validity.clone()).unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(w2, Arc::from(second.clone()), validity.clone()).unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(t, Arc::from(treatment.clone()), validity.clone()).unwrap(),
        ),
        OwnedColumn::Float64(Float64Column::new(y, Arc::from(outcome), validity).unwrap()),
    ];
    let data = TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());
    let estimand = IdentifiedEstimand::backdoor(
        "backdoor.adjustment",
        Arc::from(vec![w1, w2]),
        ExprId::from_raw(0),
    );
    let mut query = AverageEffectQuery::binary_ate(t, y);
    query.effect_modifiers = Arc::from(vec![w1, w2]);
    let query = antecedent_core::ConditionalEffectQuery::try_new(query).unwrap();
    let prep =
        BayesianGComputationAte::new().prepare_conditional(&data, &estimand, &query).unwrap();

    assert_eq!(prep.design.ncols, 6);
    let nrows = prep.design.nrows;
    let at = |column: usize, row: usize| prep.design.matrix[column * nrows + row];
    assert!((at(1, 0) - treatment[0]).abs() < 1e-12);
    assert!((at(2, 0) - first[0]).abs() < 1e-12);
    assert!((at(3, 0) - second[0]).abs() < 1e-12);
    assert!((at(4, 0) - treatment[0] * (first[0] - 5.5)).abs() < 1e-12);
    assert!((at(5, 0) - treatment[0] * (second[0] - 8.0)).abs() < 1e-12);
    let names = prep.coef_names.unwrap();
    assert_eq!(names[4].as_ref(), "treatment_centered_modifier_0");
    assert_eq!(names[5].as_ref(), "treatment_centered_modifier_1");

    let mut duplicate = query.inner.clone();
    duplicate.effect_modifiers = Arc::from(vec![w1, w1]);
    let duplicate = antecedent_core::ConditionalEffectQuery::try_new(duplicate).unwrap();
    assert!(
        BayesianGComputationAte::new().prepare_conditional(&data, &estimand, &duplicate).is_err()
    );
}

#[test]
fn conditional_logit_contrast_uses_both_intervention_rows_on_outcome_scale() {
    let treatment = [0.0, 1.0, 0.0, 1.0];
    let modifier = [-2.0, -1.0, 1.0, 2.0];
    let interaction: Vec<f64> = treatment.iter().zip(modifier).map(|(t, w)| t * w).collect();
    let outcome = [0.0; 4];
    let id = VariableId::from_raw(2);
    let design = CompiledDesign::linear_adjustment(
        &treatment,
        &[(id, &modifier), (id, &interaction)],
        &outcome,
        &[],
    )
    .unwrap();
    let problem = PreparedBayesianProblem {
        design,
        method: Arc::from("backdoor.adjustment"),
        adjustment_set: Arc::from([]),
        active: 1.0,
        control: 0.0,
        overlap: OverlapPolicy::ExplicitOverride,
        coef_names: Some(Arc::from([
            Arc::from("intercept"),
            Arc::from("treatment_at_mean_modifier"),
            Arc::from("modifier_2"),
            Arc::from("treatment_centered_modifier_2"),
        ])),
        unit_ids: None,
        serial_dependence: SerialDependence::Iid,
        preparation: None,
    };
    let quantities: Vec<_> = (0..4)
        .map(|index| PosteriorQuantityKind::Coefficient { index, name: None })
        .chain([PosteriorQuantityKind::Effect { name: Arc::from("ate") }])
        .collect();
    let mut values = vec![0.3, 0.7, 0.2, 0.8, 0.0];
    add_modifier_mean_uncertainty(
        &problem,
        GlmFamily::BinomialLogit,
        &quantities,
        &mut values,
        1,
        4,
        17,
    );
    let mut rng = antecedent_core::CausalRng::from_seed(17 ^ 0x4D4F_4449_4649_4552);
    let weights = antecedent_kernels::bayesian_bootstrap_weights(&mut rng, 4, false);
    let expected = weights
        .iter()
        .zip(modifier)
        .map(|(weight, w)| {
            let baseline = 0.3 + 0.2 * w;
            weight
                * (GlmFamily::BinomialLogit.mean_from_eta(baseline + 0.7 + 0.8 * w)
                    - GlmFamily::BinomialLogit.mean_from_eta(baseline))
        })
        .sum::<f64>()
        / weights.iter().sum::<f64>();
    assert!((values[4] - expected).abs() < 1e-12);
}

fn linear_scm_pooled(
    n_each: usize,
    noise: f64,
) -> (TabularData, VariableId, VariableId, VariableId) {
    // Two batches concatenated (batch A seed_off=0, batch B seed_off=17).
    let n = n_each * 2;
    let mut b = CausalSchemaBuilder::new();
    b.add_variable(
        "Z",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::Context),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    b.add_variable(
        "T",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    b.add_variable(
        "Y",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    let schema = b.build().unwrap();
    let z = VariableId::from_raw(0);
    let t = VariableId::from_raw(1);
    let y = VariableId::from_raw(2);
    let mut zv = vec![0.0; n];
    let mut tv = vec![0.0; n];
    let mut yv = vec![0.0; n];
    for batch in 0..2 {
        let seed_off = if batch == 0 { 0 } else { 17 };
        for i in 0..n_each {
            let r = batch * n_each + i;
            zv[r] = (i as f64) * 0.1;
            tv[r] = if i % 2 == 0 { 1.0 } else { 0.0 };
            let e = (((i + seed_off) % 7) as f64 - 3.0) * noise;
            yv[r] = 2.0 * tv[r] + 0.5 * zv[r] + e;
        }
    }
    let validity = ValidityBitmap::all_valid(n);
    let cols = vec![
        OwnedColumn::Float64(Float64Column::new(z, Arc::from(zv), validity.clone()).unwrap()),
        OwnedColumn::Float64(Float64Column::new(t, Arc::from(tv), validity.clone()).unwrap()),
        OwnedColumn::Float64(Float64Column::new(y, Arc::from(yv), validity).unwrap()),
    ];
    let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
    (TabularData::new(storage), t, y, z)
}

#[test]
fn bayesian_and_frequentist_share_ate() {
    let n = 80;
    let (data, t, y, z) = linear_scm_table(n);
    let estimand = IdentifiedEstimand::backdoor(
        "backdoor.adjustment",
        Arc::from(vec![z]),
        ExprId::from_raw(0),
    );
    let query = AverageEffectQuery::binary_ate(t, y);

    let freq = crate::adjustment::LinearAdjustmentAte::new().with_bootstrap_replicates(0);
    let prep = freq.prepare(&data, &estimand, &query).unwrap();
    let mut ws = crate::adjustment::EstimationWorkspace::default();
    let freq_est =
        freq.fit(&prep, &mut ws, &ExecutionContext::for_tests(1), AssumptionSet::new()).unwrap();

    let bayes = BayesianGComputationAte {
        backend: BayesianBackendKind::ConjugateGaussian,
        n_draws: 400,
        seed: 5,
        prior_scale: 100.0,
        ..BayesianGComputationAte::new()
    };
    let bprep = bayes.prepare(&data, &estimand, &query).unwrap();
    let mut bws = BayesianGCompWorkspace::default();
    let post = bayes
        .fit(
            &bprep,
            IdentificationStatus::NonparametricallyIdentified,
            &mut bws,
            &ExecutionContext::for_tests(1),
        )
        .unwrap();
    let eq = post.effect_column().unwrap();
    let mean = post.summaries.mean[eq];
    assert!((freq_est.ate - 2.0).abs() < 1e-6, "frequentist ate={}", freq_est.ate);
    assert!((mean - freq_est.ate).abs() < 0.05, "bayes={mean} freq={}", freq_est.ate);
    assert_eq!(post.identification, IdentificationStatus::NonparametricallyIdentified);
    let coef_names: Vec<_> = post
        .draws
        .schema
        .quantities
        .iter()
        .filter_map(|q| match q {
            PosteriorQuantityKind::Coefficient { name, .. } => name.as_ref().map(AsRef::as_ref),
            _ => None,
        })
        .collect();
    assert!(coef_names.contains(&"intercept"), "{coef_names:?}");
    assert!(coef_names.iter().any(|n| n.starts_with("coef_")), "{coef_names:?}");
}

#[test]
fn posterior_records_the_preparation_receipt_of_the_admitted_columns() {
    let (data, t, y, z) = linear_scm_table(80);
    let estimand = IdentifiedEstimand::backdoor(
        "backdoor.adjustment",
        Arc::from(vec![z]),
        ExprId::from_raw(0),
    );
    let query = AverageEffectQuery::binary_ate(t, y);
    let estimator =
        BayesianGComputationAte { n_draws: 96, seed: 17, ..BayesianGComputationAte::new() };
    let prepared = estimator.prepare(&data, &estimand, &query).unwrap();
    let receipt = prepared.preparation.clone().expect("prepare records the receipt");
    assert_eq!(receipt.rows_admitted, Some(80));
    let mut workspace = BayesianGCompWorkspace::default();
    let posterior = estimator
        .fit(
            &prepared,
            IdentificationStatus::NonparametricallyIdentified,
            &mut workspace,
            &ExecutionContext::production(17, 1),
        )
        .unwrap();
    let recorded = posterior.preparation.as_ref().expect("posterior carries the receipt");
    assert_eq!(recorded.backend, receipt.backend);
    assert_eq!(recorded.rows_admitted, Some(80));
    assert_eq!(recorded.receipt_digest, receipt.receipt_digest);
}

#[test]
fn adaptive_laplace_unknown_variance_keeps_exact_nig_draws() {
    let (data, t, y, z) = linear_scm_table(80);
    let estimand = IdentifiedEstimand::backdoor(
        "backdoor.adjustment",
        Arc::from(vec![z]),
        ExprId::from_raw(0),
    );
    let query = AverageEffectQuery::binary_ate(t, y);
    let estimator =
        BayesianGComputationAte { n_draws: 96, seed: 17, ..BayesianGComputationAte::new() };
    let prepared = estimator.prepare(&data, &estimand, &query).unwrap();
    let mut workspace = BayesianGCompWorkspace::default();
    let posterior = estimator
        .fit(
            &prepared,
            IdentificationStatus::NonparametricallyIdentified,
            &mut workspace,
            &ExecutionContext::production(17, 1),
        )
        .unwrap();

    assert_eq!(posterior.diagnostics.backend_id.as_ref(), "conjugate_gaussian");
    assert_eq!(posterior.draws.n_draws, 96, "NIG draws must not be replaced by MVN redraws");
    assert!(!posterior.early_stopped, "exact NIG sampling materializes the requested draws");
}

#[test]
fn prior_does_not_create_identification() {
    let prior = PriorSet::weakly_informative(3);
    let post =
        nonidentified_with_prior(&prior, 0.0, 10.0, InferenceDiagnostics::analytic("none"), 64, 1)
            .unwrap();
    assert_eq!(post.identification, IdentificationStatus::NotIdentified);
    assert!(!post.assumptions.is_empty());
    assert!((post.unidentified_mass - 1.0).abs() < 1e-12);
    assert!(post.draws.n_draws > 0, "prior-predictive draws required");
}

#[test]
fn prior_predictive_effect_follows_the_explicit_effect_prior_not_the_intercept() {
    // The design-shaped prior's first coefficient (the intercept) is N(50, 0.1²);
    // the effect prior is N(-3, 0.5²). The draws must follow the effect prior.
    let mut prior = PriorSet::weakly_informative(2);
    if let Some(antecedent_prob::PriorSpec::GaussianCoefficients(g)) = prior.specs.first_mut() {
        *g = antecedent_prob::GaussianCoefficientPrior {
            mean: Arc::from(vec![50.0, 0.0]),
            variance: Arc::from(vec![0.01, 4.0]),
        };
    }
    let post = nonidentified_with_prior(
        &prior,
        -3.0,
        0.5,
        InferenceDiagnostics::analytic("none"),
        4000,
        9,
    )
    .unwrap();
    let draws = post.draws.column(0).unwrap();
    let mean = draws.iter().sum::<f64>() / draws.len() as f64;
    let sd =
        (draws.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (draws.len() - 1) as f64).sqrt();
    // Monte Carlo error of the mean is 0.5/sqrt(4000) = 0.008; of the sd about 0.006.
    assert!((mean + 3.0).abs() < 0.04, "mean={mean}");
    assert!((sd - 0.5).abs() < 0.03, "sd={sd}");
    for (mean, sd) in [(f64::NAN, 1.0), (0.0, 0.0), (0.0, f64::INFINITY), (0.0, -1.0)] {
        assert!(
            nonidentified_with_prior(
                &prior,
                mean,
                sd,
                InferenceDiagnostics::analytic("none"),
                8,
                1
            )
            .is_err()
        );
    }
}

#[test]
fn temporal_prepared_design_conjugate_recovers_pulse() {
    use antecedent_core::{
        CausalSchemaBuilder, Lag, MeasurementSpec, RoleHint, SmallRoleSet, TemporalEffectQuery,
        TemporalPolicy, ValueType,
    };
    use antecedent_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TimeIndex,
        TimeSeriesData, ValidityBitmap,
    };
    use antecedent_graph::{TemporalDag, ensure_lagged};
    use antecedent_identify::TemporalBackdoorIdentifier;

    use crate::temporal_adjustment::TemporalLinearAdjustment;

    let n = 300usize;
    let mut b = CausalSchemaBuilder::new();
    b.add_variable(
        "x",
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
    let mut x = vec![0.0; n];
    let mut y = vec![0.0; n];
    for t in 1..n {
        x[t] = ((t as f64) * 0.07).sin();
        y[t] = 0.8 * x[t - 1];
    }
    let cols = vec![
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(0), Arc::from(x), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(1), Arc::from(y), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
    ];
    let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
    let data = TimeSeriesData::try_new(
        storage,
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
    )
    .unwrap();
    let mut g = TemporalDag::empty();
    let x1 = ensure_lagged(&mut g, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let y0 = ensure_lagged(&mut g, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    g.insert_directed(x1, y0).unwrap();

    let q = TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
        .with_policy(TemporalPolicy::pulse(-1))
        .with_horizon_steps(1)
        .with_max_history_lag(Some(1));
    let id_res = TemporalBackdoorIdentifier::new().identify_temporal(&g, &q).unwrap();
    let estimand = id_res.result.estimands.first().unwrap();
    let temporal = TemporalLinearAdjustment::new();
    let prep = temporal
        .prepare(
            &data,
            estimand,
            &q,
            &id_res.indexer,
            None,
            &ExecutionContext::for_tests(1).kernel_policy,
        )
        .unwrap();
    let bayes = BayesianTemporalGcomp {
        inner: BayesianGComputationAte {
            backend: BayesianBackendKind::ConjugateGaussian,
            n_draws: 200,
            seed: 7,
            prior_scale: 100.0,
            ..BayesianGComputationAte::new()
        },
    };
    let bprep = BayesianTemporalGcomp::from_prepared_estimation(&prep);
    let mut ws = BayesianGCompWorkspace::default();
    let post = bayes
        .fit(
            &bprep,
            IdentificationStatus::NonparametricallyIdentified,
            &mut ws,
            &ExecutionContext::for_tests(1),
        )
        .unwrap();
    let eq = post.effect_column().unwrap();
    let mean = post.summaries.mean[eq];
    assert!((mean - 0.8).abs() < 0.05, "bayesian temporal pulse mean={mean}");
    assert!(post.probability_below(0.0).unwrap().is_finite());
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "fixture, two posterior fits and the comparing assertions of one scenario"
)]
fn hierarchical_unit_effects_widen_posterior_vs_stacked() {
    use crate::adjustment::LinearAdjustmentAte;
    use antecedent_core::AverageEffectQuery;

    let n_units = 40usize;
    let t_len = 8usize;
    let n = n_units * t_len;
    let mut t = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut z = Vec::with_capacity(n);
    let mut unit_ids = Vec::with_capacity(n);
    for u in 0..n_units {
        let a = if u % 2 == 0 { 1.0 } else { 0.0 };
        let unit_eff = 1.6 * ((u as f64) * 0.31).sin();
        for k in 0..t_len {
            t.push(a);
            z.push(0.0);
            y.push(2.0 * a + unit_eff + 0.15 * ((k + 1) as f64).sin());
            unit_ids.push(u as u32);
        }
    }
    let (data, treatment, outcome, _z) = {
        // Reuse column layout T,Y,Z via a local table.
        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "T",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "Y",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "Z",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        let schema = b.build().unwrap();
        let validity = ValidityBitmap::all_valid(n);
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(0), Arc::from(t), validity.clone())
                    .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(1), Arc::from(y), validity.clone())
                    .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(2), Arc::from(z), validity).unwrap(),
            ),
        ];
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        (
            TabularData::new(storage),
            VariableId::from_raw(0),
            VariableId::from_raw(1),
            VariableId::from_raw(2),
        )
    };
    let estimand = IdentifiedEstimand::backdoor(
        "backdoor.adjustment",
        Arc::from([] as [VariableId; 0]),
        ExprId::from_raw(0),
    );
    let query = AverageEffectQuery::binary_ate(treatment, outcome);
    let freq = LinearAdjustmentAte::new().with_bootstrap_replicates(0);
    let prep = freq.prepare(&data, &estimand, &query).unwrap();
    let stacked = BayesianGComputationAte::from_prepared_estimation(&prep);
    let mut hierarchical = stacked.clone();
    hierarchical.unit_ids = Some(unit_ids);
    let bayes = BayesianGComputationAte {
        backend: BayesianBackendKind::ConjugateGaussian,
        n_draws: 400,
        seed: 11,
        prior_scale: 10.0,
        ..BayesianGComputationAte::new()
    };
    let mut ws = BayesianGCompWorkspace::default();
    let ctx = ExecutionContext::for_tests(4);
    let post_s = bayes
        .fit(&stacked, IdentificationStatus::NonparametricallyIdentified, &mut ws, &ctx)
        .unwrap();
    let post_h = bayes
        .fit(&hierarchical, IdentificationStatus::NonparametricallyIdentified, &mut ws, &ctx)
        .unwrap();
    let eq = post_s.effect_column().unwrap();
    let sd_s = post_s.summaries.sd[eq];
    let sd_h = post_h.summaries.sd[eq];
    assert!(sd_s.is_finite() && sd_h.is_finite(), "sd stacked={sd_s} hierarchical={sd_h}");
    let pin: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../conformance/estimate/panel_hierarchical_vs_stacked/expected.json"
    ))
    .unwrap();
    let min_ratio = pin["min_posterior_sd_ratio_hierarchical_over_stacked"].as_f64().unwrap();
    assert!(
        sd_h > sd_s * min_ratio,
        "hierarchical posterior must be wider than stacked iid (stacked={sd_s}, hierarchical={sd_h}, min_ratio={min_ratio})"
    );
}

#[test]
fn unit_ids_length_mismatch_is_an_error() {
    use crate::adjustment::LinearAdjustmentAte;
    use antecedent_core::AverageEffectQuery;

    let n = 40usize;
    let t = [0.0, 1.0].repeat(n / 2);
    let y: Vec<f64> = t.iter().map(|&a| 2.0 * a).collect();
    let z = vec![0.0; n];
    let mut b = CausalSchemaBuilder::new();
    b.add_variable(
        "T",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    b.add_variable(
        "Y",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    b.add_variable(
        "Z",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::Context),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    let schema = b.build().unwrap();
    let validity = ValidityBitmap::all_valid(n);
    let cols = vec![
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(0), Arc::from(t), validity.clone()).unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(1), Arc::from(y), validity.clone()).unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(2), Arc::from(z), validity).unwrap(),
        ),
    ];
    let data = TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());
    let estimand = IdentifiedEstimand::backdoor(
        "backdoor.adjustment",
        Arc::from([] as [VariableId; 0]),
        ExprId::from_raw(0),
    );
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let freq = LinearAdjustmentAte::new().with_bootstrap_replicates(0);
    let prep = freq.prepare(&data, &estimand, &query).unwrap();
    let mut bprep = BayesianGComputationAte::from_prepared_estimation(&prep);
    bprep.unit_ids = Some(vec![0, 1]);
    let bayes = BayesianGComputationAte::conjugate();
    let mut ws = BayesianGCompWorkspace::default();
    let err = bayes
        .fit(
            &bprep,
            IdentificationStatus::NonparametricallyIdentified,
            &mut ws,
            &ExecutionContext::for_tests(1),
        )
        .unwrap_err();
    assert!(err.to_string().contains("unit_ids"), "length mismatch must fail closed, got {err}");
}

/// GLS whitening must not feed a
/// Poisson likelihood. With `unit_ids` present, the fit must consume the raw
/// count column (same posterior as stacked iid) and disclose the skip.
#[test]
#[allow(
    clippy::too_many_lines,
    reason = "fixture, fit and the assertions of one scenario, kept in a single test"
)]
fn poisson_random_intercept_does_not_consume_whitened_outcome() {
    use crate::adjustment::LinearAdjustmentAte;
    use antecedent_core::AverageEffectQuery;

    let n_units = 20usize;
    let t_len = 6usize;
    let n = n_units * t_len;
    let mut t = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut z = Vec::with_capacity(n);
    let mut unit_ids = Vec::with_capacity(n);
    for u in 0..n_units {
        let a = if u % 2 == 0 { 1.0 } else { 0.0 };
        // Unit intercepts vary within each treatment arm so MOM τ²>0 after OLS.
        let unit_base = 4.0 + (u % 5) as f64;
        for k in 0..t_len {
            t.push(a);
            z.push(0.0);
            // Within-unit count variation so residual σ²>0 (required by the whitener).
            let rate: f64 = unit_base * if a > 0.5 { 2.0 } else { 1.0 } + (k % 3) as f64;
            y.push(rate.round().max(0.0));
            unit_ids.push(u as u32);
        }
    }
    let mut b = CausalSchemaBuilder::new();
    b.add_variable(
        "T",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    b.add_variable(
        "Y",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    b.add_variable(
        "Z",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::Context),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    let schema = b.build().unwrap();
    let validity = ValidityBitmap::all_valid(n);
    let cols = vec![
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(0), Arc::from(t), validity.clone()).unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(1), Arc::from(y.clone()), validity.clone())
                .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(2), Arc::from(z), validity).unwrap(),
        ),
    ];
    let data = TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());
    let estimand = IdentifiedEstimand::backdoor(
        "backdoor.adjustment",
        Arc::from([] as [VariableId; 0]),
        ExprId::from_raw(0),
    );
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let freq = LinearAdjustmentAte::new().with_bootstrap_replicates(0);
    let prep = freq.prepare(&data, &estimand, &query).unwrap();

    // Whitening is applicable on this panel and would change the outcome column.
    let (x_star, y_star) = random_intercept_gls_whiten(
        &prep.design.matrix,
        &prep.design.outcome,
        prep.design.nrows,
        prep.design.ncols,
        &unit_ids,
    )
    .expect("panel with shared unit intercepts must admit GLS whitening");
    assert!(
        y_star.iter().zip(prep.design.outcome.iter()).any(|(a, b)| (a - b).abs() > 1e-9),
        "sanity: whitened outcome must differ from the raw counts"
    );
    assert_eq!(x_star.len(), prep.design.matrix.len());

    let stacked = BayesianGComputationAte::from_prepared_estimation(&prep);
    let mut hierarchical = stacked.clone();
    hierarchical.unit_ids = Some(unit_ids);

    let bayes = BayesianGComputationAte {
        backend: BayesianBackendKind::Laplace,
        likelihood: BayesLikelihood::PoissonLog,
        n_draws: 200,
        seed: 19,
        prior_scale: 10.0,
        ..BayesianGComputationAte::new()
    };
    let mut ws = BayesianGCompWorkspace::default();
    let ctx = ExecutionContext::for_tests(2);
    let post_plain = bayes
        .fit(&stacked, IdentificationStatus::NonparametricallyIdentified, &mut ws, &ctx)
        .unwrap();
    let post_ri = bayes
        .fit(&hierarchical, IdentificationStatus::NonparametricallyIdentified, &mut ws, &ctx)
        .unwrap();

    let eq = post_plain.effect_column().unwrap();
    let mean_plain = post_plain.summaries.mean[eq];
    let mean_ri = post_ri.summaries.mean[eq];
    assert!(
        (mean_plain - mean_ri).abs() < 1e-12,
        "Poisson + unit_ids must not consume the whitened column \
         (stacked={mean_plain}, with_unit_ids={mean_ri})"
    );
    assert!(
        post_ri.diagnostics.notes.iter().any(|n| n.as_ref() == RANDOM_INTERCEPT_WHITEN_SKIP_NOTE),
        "non-Gaussian random-intercept skip must be disclosed, got {:?}",
        post_ri.diagnostics.notes
    );
    assert!(
        !post_plain
            .diagnostics
            .notes
            .iter()
            .any(|n| n.as_ref() == RANDOM_INTERCEPT_WHITEN_SKIP_NOTE),
        "stacked fit must not emit the skip note"
    );

    // A Gaussian fit discloses when whitening could not apply and the rows were
    // fitted as independent (every row its own unit leaves no within-unit spread).
    let gaussian =
        BayesianGComputationAte { likelihood: BayesLikelihood::GaussianIdentity, ..bayes };
    let status = IdentificationStatus::NonparametricallyIdentified;
    let has_fallback_note = |post: &CausalPosterior| {
        post.diagnostics.notes.iter().any(|n| n.as_ref() == RANDOM_INTERCEPT_WHITEN_FALLBACK_NOTE)
    };
    let mut singleton_units = stacked.clone();
    singleton_units.unit_ids = Some((0..n as u32).collect());
    let fallback = gaussian.fit(&singleton_units, status, &mut ws, &ctx).unwrap();
    assert!(has_fallback_note(&fallback), "{:?}", fallback.diagnostics.notes);
    let whitened = gaussian.fit(&hierarchical, status, &mut ws, &ctx).unwrap();
    assert!(!has_fallback_note(&whitened), "{:?}", whitened.diagnostics.notes);
    let plain = gaussian.fit(&stacked, status, &mut ws, &ctx).unwrap();
    assert!(!has_fallback_note(&plain), "{:?}", plain.diagnostics.notes);
}

/// Seeded standard-normal draws for the calibration DGPs: Box-Muller over a `SplitMix64`
/// fixture stream whose state is the scrambled seed, so consecutive replicate seeds start
/// unrelated streams.
fn box_muller_fixture(seed: u64) -> impl FnMut() -> f64 {
    let mut rng = crate::splitmix::FixtureRng::new(crate::splitmix::seed_mix(seed));
    let mut next_unit = move || rng.uniform();
    move || {
        let u = next_unit();
        let v = next_unit();
        (-2.0 * u.max(1e-12).ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
}

fn noisy_lag1_pulse_series(n: usize, seed: u64) -> antecedent_data::TimeSeriesData {
    use antecedent_data::{SamplingRegularity, TimeIndex, TimeSeriesData};
    let mut b = CausalSchemaBuilder::new();
    b.add_variable(
        "x",
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
    let mut gauss = box_muller_fixture(seed);
    let mut x = vec![0.0; n];
    let mut y = vec![0.0; n];
    for t in 1..n {
        x[t] = 0.4 * gauss();
        y[t] = 0.8 * x[t - 1] + 0.35 * gauss();
    }
    let cols = vec![
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(0), Arc::from(x), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(1), Arc::from(y), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
    ];
    let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
    TimeSeriesData::try_new(
        storage,
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
    )
    .unwrap()
}

fn pulse_graph() -> antecedent_graph::TemporalDag {
    use antecedent_core::Lag;
    use antecedent_graph::{TemporalDag, ensure_lagged};
    let mut g = TemporalDag::empty();
    let x1 = ensure_lagged(&mut g, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let y0 = ensure_lagged(&mut g, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    g.insert_directed(x1, y0).unwrap();
    g
}

/// Default replicate count for the coverage gate (matches
/// `crates/antecedent/tests/common/calibration.rs`, which this crate cannot import).
const CALIBRATION_N_SIM: u32 = 400;

/// Replicate count, honoring `ANTECEDENT_CALIBRATION_NSIM` for local smoke runs.
fn calibration_n_sim() -> u32 {
    std::env::var("ANTECEDENT_CALIBRATION_NSIM")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(CALIBRATION_N_SIM)
}

/// Two-sided coverage tally: `level ± 3·MCSE`, `MCSE = sqrt(level(1-level)/n)`.
/// At 400 replicates and 90% that is `[0.855, 0.945]`, so under- and
/// over-coverage both fail.
struct CoverageGate {
    name: &'static str,
    level: f64,
    covered: u32,
    scored: u32,
    length_sum: f64,
}

impl CoverageGate {
    fn new(name: &'static str, level: f64) -> Self {
        Self { name, level, covered: 0, scored: 0, length_sum: 0.0 }
    }

    /// Score the equal-tailed posterior interval of the effect column.
    fn record(&mut self, post: &CausalPosterior, truth: f64) {
        self.scored += 1;
        let eq = post.effect_column().unwrap();
        let col = post.draws.column(eq).unwrap();
        let mut v: Vec<f64> = col.iter().copied().filter(|x| x.is_finite()).collect();
        if v.len() < 2 {
            return;
        }
        v.sort_by(f64::total_cmp);
        let lo_p = (1.0 - self.level) / 2.0;
        let last = (v.len() - 1) as f64;
        // `q` is a probability and `last >= 0`, so the rounded product is a valid index.
        #[allow(
            clippy::cast_sign_loss,
            reason = "the rounded index is clamped to [0, last] so it is non-negative and in range"
        )]
        let at = |q: f64| v[(last * q).round().clamp(0.0, last) as usize];
        let (lo, hi) = (at(lo_p), at(1.0 - lo_p));
        self.length_sum += hi - lo;
        if truth >= lo && truth <= hi {
            self.covered += 1;
        }
    }

    fn assert(&self) {
        let n = f64::from(self.scored);
        let rate = f64::from(self.covered) / n;
        let mcse = (self.level * (1.0 - self.level) / n).sqrt();
        let lo = (self.level - 3.0 * mcse).max(0.0);
        let hi = (self.level + 3.0 * mcse).min(1.0);
        eprintln!(
            "calibration {}: nominal={:.2} coverage={rate:.3} mcse={mcse:.4} band=[{lo:.3}, {hi:.3}] \
             mean_length={:.4} ({}/{} covered)",
            self.name,
            self.level,
            self.length_sum / n,
            self.covered,
            self.scored
        );
        assert!(
            rate >= lo && rate <= hi,
            "{} {:.0}% coverage={rate:.3} outside [{lo:.3}, {hi:.3}] ({}/{})",
            self.name,
            self.level * 100.0,
            self.covered,
            self.scored
        );
    }
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn bayesian_pulse_conjugate_nominal_90_coverage() {
    use crate::temporal_adjustment::TemporalLinearAdjustment;
    use antecedent_core::{TemporalEffectQuery, TemporalPolicy};
    use antecedent_graph::ensure_lagged;
    use antecedent_identify::TemporalBackdoorIdentifier;

    let n_sim = calibration_n_sim();
    let n = 160usize;
    let mut gate = CoverageGate::new("bayesian pulse (conjugate, direct estimator)", 0.9);
    let g = pulse_graph();
    let q = TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
        .with_policy(TemporalPolicy::pulse(-1))
        .with_horizon_steps(1)
        .with_max_history_lag(Some(1));
    let id_res = TemporalBackdoorIdentifier::new().identify_temporal(&g, &q).unwrap();
    let estimand = id_res.result.estimands.first().unwrap();
    let temporal = TemporalLinearAdjustment::new();
    for s in 0..n_sim {
        // Vary the sampler seed per replicate: a fixed seed reuses one set of
        // posterior draws, so its Monte-Carlo quantile error is shared by every
        // replicate instead of averaging out.
        let bayes = BayesianTemporalGcomp {
            inner: BayesianGComputationAte {
                backend: BayesianBackendKind::ConjugateGaussian,
                n_draws: 240,
                seed: 21 + u64::from(s),
                prior_scale: 8.0,
                ..BayesianGComputationAte::new()
            },
        };
        let data = noisy_lag1_pulse_series(n, 9_000 + u64::from(s));
        let prep = temporal
            .prepare(
                &data,
                estimand,
                &q,
                &id_res.indexer,
                None,
                &ExecutionContext::for_tests(1).kernel_policy,
            )
            .unwrap();
        let bprep = BayesianTemporalGcomp::from_prepared_estimation(&prep);
        let mut ws = BayesianGCompWorkspace::default();
        let post = bayes
            .fit(
                &bprep,
                IdentificationStatus::NonparametricallyIdentified,
                &mut ws,
                &ExecutionContext::for_tests(1),
            )
            .unwrap();
        gate.record(&post, 0.8);
    }
    gate.assert();
    let _ = ensure_lagged;
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn bayesian_sustained_single_step_conjugate_nominal_90_coverage() {
    use crate::temporal_adjustment::TemporalLinearAdjustment;
    use antecedent_core::{TemporalEffectQuery, TemporalPolicy};
    use antecedent_identify::TemporalBackdoorIdentifier;

    let n_sim = calibration_n_sim();
    let n = 160usize;
    let mut gate = CoverageGate::new("bayesian single-step sustained (conjugate)", 0.9);
    let g = pulse_graph();
    let q =
        TemporalEffectQuery::sustained(VariableId::from_raw(0), VariableId::from_raw(1), -1, 1.0)
            .with_policy(TemporalPolicy::sustained(-1, -1))
            .with_horizon_steps(1)
            .with_max_history_lag(Some(1));
    let id_res = TemporalBackdoorIdentifier::new().identify_temporal(&g, &q).unwrap();
    let estimand = id_res.result.estimands.first().unwrap();
    let temporal = TemporalLinearAdjustment::new();
    for s in 0..n_sim {
        // Vary the sampler seed per replicate: a fixed seed reuses one set of
        // posterior draws, so its Monte-Carlo quantile error is shared by every
        // replicate instead of averaging out.
        let bayes = BayesianTemporalGcomp {
            inner: BayesianGComputationAte {
                backend: BayesianBackendKind::ConjugateGaussian,
                n_draws: 240,
                seed: 22 + u64::from(s),
                prior_scale: 8.0,
                ..BayesianGComputationAte::new()
            },
        };
        let data = noisy_lag1_pulse_series(n, 11_000 + u64::from(s));
        let prep = temporal
            .prepare(
                &data,
                estimand,
                &q,
                &id_res.indexer,
                None,
                &ExecutionContext::for_tests(1).kernel_policy,
            )
            .unwrap();
        let bprep = BayesianTemporalGcomp::from_prepared_estimation(&prep);
        let mut ws = BayesianGCompWorkspace::default();
        let post = bayes
            .fit(
                &bprep,
                IdentificationStatus::NonparametricallyIdentified,
                &mut ws,
                &ExecutionContext::for_tests(1),
            )
            .unwrap();
        gate.record(&post, 0.8);
    }
    gate.assert();
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn bayesian_sustained_multi_step_conjugate_nominal_90_coverage() {
    use crate::temporal_sequential::estimate_sustained_window;
    use antecedent_core::{TemporalEffectQuery, TemporalPolicy};
    use antecedent_identify::TemporalBackdoorIdentifier;

    let n_sim = calibration_n_sim();
    let n = 160usize;
    let mut gate = CoverageGate::new("bayesian multi-step sustained (conjugate)", 0.9);
    let g = pulse_graph();
    let q =
        TemporalEffectQuery::sustained(VariableId::from_raw(0), VariableId::from_raw(1), -2, 1.0)
            .with_policy(TemporalPolicy::sustained(-2, -1))
            .with_horizon_steps(1)
            .with_max_history_lag(Some(2));
    let id_res = TemporalBackdoorIdentifier::new().identify_temporal(&g, &q).unwrap();
    let estimand = id_res.result.estimands.first().unwrap();
    for s in 0..n_sim {
        // Vary the sampler seed per replicate: a fixed seed reuses one set of
        // posterior draws, so its Monte-Carlo quantile error is shared by every
        // replicate instead of averaging out.
        let bayes = BayesianGComputationAte {
            backend: BayesianBackendKind::ConjugateGaussian,
            n_draws: 240,
            seed: 23 + u64::from(s),
            prior_scale: 8.0,
            ..BayesianGComputationAte::new()
        };
        let data = noisy_lag1_pulse_series(n, 13_000 + u64::from(s));
        let (_, posterior) = estimate_sustained_window(
            &crate::temporal_sequential::SequentialProblem {
                data: &data,
                graph: &g,
                indexer: &id_res.indexer,
                estimand,
            },
            &q,
            crate::temporal_sequential::SequentialInference {
                status: IdentificationStatus::NonparametricallyIdentified,
                assumptions: antecedent_core::AssumptionSet::default(),
                bootstrap_replicates: 0,
                bayesian: Some(&bayes),
            },
            &ExecutionContext::for_tests(1),
        )
        .unwrap();
        gate.record(posterior.as_ref().unwrap(), 0.8);
    }
    gate.assert();
}

fn noisy_lag1_pulse_series_with_unit(
    n: usize,
    seed: u64,
    unit_eff: f64,
    treat: f64,
) -> antecedent_data::TimeSeriesData {
    let mut gauss = box_muller_fixture(seed);
    let x = vec![treat; n];
    let mut y = vec![0.0; n];
    // Treatment is constant within unit. Stacked iid treats T repeats as
    // independent looks at the same contrast; random-intercept GLS must widen.
    for t in 1..n {
        y[t] = 0.8 * x[t - 1] + unit_eff + 0.35 * gauss();
    }
    let mut b = CausalSchemaBuilder::new();
    b.add_variable(
        "x",
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
            Float64Column::new(VariableId::from_raw(0), Arc::from(x), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(1), Arc::from(y), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
    ];
    let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
    antecedent_data::TimeSeriesData::try_new(
        storage,
        antecedent_data::TimeIndex {
            regularity: antecedent_data::SamplingRegularity::Regular { interval_ns: 1 },
            length: n,
        },
    )
    .unwrap()
}

fn unit_effect_lag1_panel(n_units: usize, t_len: usize, seed: u64) -> antecedent_data::PanelData {
    use antecedent_data::{PanelData, PanelUnit};
    // Random intercepts are redrawn per replicate: the hierarchical model's
    // interval is for a random-effects population. A fixed intercept pattern
    // (formerly `1.4·sin(0.37u)`) freezes the between-unit imbalance across
    // replicates, so the interval, which correctly prices intercept variance,
    // covered 100% at 400 replicates. SD ≈ 0.99 matches that pattern's spread.
    let mut intercepts = box_muller_fixture(seed ^ 0x0E7F_0E7F);
    let units: Vec<PanelUnit> = (0..n_units)
        .map(|u| {
            let unit_eff = 0.99 * intercepts();
            PanelUnit {
                unit_id: u as u32,
                series: noisy_lag1_pulse_series_with_unit(
                    t_len,
                    seed.wrapping_add(u as u64 * 19),
                    unit_eff,
                    if u % 2 == 0 { 1.0 } else { 0.0 },
                ),
            }
        })
        .collect();
    PanelData::try_new(Arc::from(units)).unwrap()
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn bayesian_panel_hierarchical_nominal_90_coverage() {
    use crate::temporal_adjustment::TemporalLinearAdjustment;
    use antecedent_core::{TemporalEffectQuery, TemporalPolicy};
    use antecedent_identify::TemporalBackdoorIdentifier;

    let n_sim = calibration_n_sim();
    let n_units = 24usize;
    let t_len = 40usize;
    let mut gate = CoverageGate::new("bayesian panel hierarchical", 0.9);
    let g = pulse_graph();
    let q = TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
        .with_policy(TemporalPolicy::pulse(-1))
        .with_horizon_steps(1)
        .with_max_history_lag(Some(1));
    let id_res = TemporalBackdoorIdentifier::new().identify_temporal(&g, &q).unwrap();
    let estimand = id_res.result.estimands.first().unwrap();
    let temporal = TemporalLinearAdjustment::new();
    for s in 0..n_sim {
        // Vary the sampler seed per replicate: a fixed seed reuses one set of
        // posterior draws, so its Monte-Carlo quantile error is shared by every
        // replicate instead of averaging out.
        let bayes = BayesianTemporalGcomp {
            inner: BayesianGComputationAte {
                backend: BayesianBackendKind::ConjugateGaussian,
                n_draws: 200,
                seed: 23 + u64::from(s),
                prior_scale: 8.0,
                ..BayesianGComputationAte::new()
            },
        };
        // Unit `u` draws from `seed + 19·u`; a stride of 1000 keeps every
        // replicate's units disjoint (a unit stride of 1 reused series
        // across replicates, so they were not independent datasets).
        let panel = unit_effect_lag1_panel(n_units, t_len, 13_000 + 1_000 * u64::from(s));
        let (prep, cluster_ids, _) = temporal
            .prepare_panel(
                &panel,
                estimand,
                &q,
                &id_res.indexer,
                None,
                &ExecutionContext::for_tests(1).kernel_policy,
            )
            .unwrap();
        let mut bprep = BayesianTemporalGcomp::from_prepared_estimation(&prep);
        bprep.unit_ids = Some(cluster_ids);
        let mut ws = BayesianGCompWorkspace::default();
        let post = bayes
            .fit(
                &bprep,
                IdentificationStatus::NonparametricallyIdentified,
                &mut ws,
                &ExecutionContext::for_tests(1),
            )
            .unwrap();
        gate.record(&post, 0.8);
    }
    gate.assert();
}

#[test]
fn hydrate_prior_from_posterior_and_refit() {
    let n = 60;
    let (data, t, y, z) = linear_scm_table(n);
    let estimand = IdentifiedEstimand::backdoor(
        "backdoor.adjustment",
        Arc::from(vec![z]),
        ExprId::from_raw(0),
    );
    let query = AverageEffectQuery::binary_ate(t, y);
    let bayes = BayesianGComputationAte {
        backend: BayesianBackendKind::ConjugateGaussian,
        n_draws: 200,
        seed: 3,
        prior_scale: 10.0,
        ..BayesianGComputationAte::new()
    };
    let prep = bayes.prepare(&data, &estimand, &query).unwrap();
    let mut ws = BayesianGCompWorkspace::default();
    let post = bayes
        .fit(
            &prep,
            IdentificationStatus::NonparametricallyIdentified,
            &mut ws,
            &ExecutionContext::for_tests(1),
        )
        .unwrap();
    let prior = hydrate_prior_from_posterior(&post, Some(prep.design.ncols)).unwrap();
    assert_eq!(prior.gaussian_coefficients().unwrap().len(), prep.design.ncols);
    assert!(hydrate_prior_from_posterior(&post, Some(prep.design.ncols + 1)).is_err());

    let sequential = BayesianGComputationAte { prior: Some(prior), ..bayes };
    let post2 = sequential
        .fit(
            &prep,
            IdentificationStatus::NonparametricallyIdentified,
            &mut ws,
            &ExecutionContext::for_tests(1),
        )
        .unwrap();
    assert!(post2.assumptions.entries.iter().any(|a| {
        matches!(a.source, AssumptionSource::Artifact)
            && matches!(&a.assumption, Assumption::PriorRestriction(pa) if pa.description.contains("sequential")
                && pa.description.contains("dense V0")
                && pa.description.contains("covariance"))
    }));
    let eq = post2.effect_column().unwrap();
    assert!(post2.summaries.mean[eq].is_finite());
}

/// Earns `estimate-bayes-transport-response-1`: absolute SD² must not be stored as V0.
#[test]
fn sequential_hydrate_v0_not_orders_narrower_than_pooled() {
    let n = 80;
    let noise = 0.4;
    let (data_a, t, y, z) = linear_scm_table_with_noise(n, noise, 0);
    let (data_b, _, _, _) = linear_scm_table_shifted(n, noise);
    let (pool, _, _, _) = linear_scm_pooled(n, noise);

    let estimand = IdentifiedEstimand::backdoor(
        "backdoor.adjustment",
        Arc::from(vec![z]),
        ExprId::from_raw(0),
    );
    let query = AverageEffectQuery::binary_ate(t, y);
    let bayes = BayesianGComputationAte {
        backend: BayesianBackendKind::ConjugateGaussian,
        n_draws: 400,
        seed: 9,
        prior_scale: 10.0,
        ..BayesianGComputationAte::new()
    };
    let mut ws = BayesianGCompWorkspace::default();

    let prep_a = bayes.prepare(&data_a, &estimand, &query).unwrap();
    let post_a = bayes
        .fit(
            &prep_a,
            IdentificationStatus::NonparametricallyIdentified,
            &mut ws,
            &ExecutionContext::for_tests(1),
        )
        .unwrap();
    assert!(
        post_a
            .draws
            .schema
            .quantities
            .iter()
            .any(|q| matches!(q, PosteriorQuantityKind::ResidualVariance)),
        "source posterior must retain residual_variance for V0 conversion"
    );
    let prior = hydrate_prior_from_posterior(&post_a, Some(prep_a.design.ncols)).unwrap();
    let coef = prior.gaussian_coefficients().unwrap();
    let sigma2 = post_a
        .draws
        .schema
        .quantities
        .iter()
        .position(|q| matches!(q, PosteriorQuantityKind::ResidualVariance))
        .map(|i| post_a.summaries.mean[i])
        .unwrap();
    for i in 0..coef.len() {
        let abs = post_a.summaries.sd[i] * post_a.summaries.sd[i];
        let expected_v0 = abs / sigma2;
        assert!(
            (coef.variance[i] - expected_v0).abs() / expected_v0.max(1e-18) < 1e-6,
            "coef {i}: V0 {} vs abs/σ² {expected_v0} (abs={abs}, σ²={sigma2})",
            coef.variance[i]
        );
    }

    let prep_b = bayes.prepare(&data_b, &estimand, &query).unwrap();
    let sequential = BayesianGComputationAte { prior: Some(prior), ..bayes.clone() };
    let post_seq = sequential
        .fit(
            &prep_b,
            IdentificationStatus::NonparametricallyIdentified,
            &mut ws,
            &ExecutionContext::for_tests(1),
        )
        .unwrap();
    let prep_pool = bayes.prepare(&pool, &estimand, &query).unwrap();
    let post_pool = bayes
        .fit(
            &prep_pool,
            IdentificationStatus::NonparametricallyIdentified,
            &mut ws,
            &ExecutionContext::for_tests(1),
        )
        .unwrap();
    let eq_s = post_seq.effect_column().unwrap();
    let eq_p = post_pool.effect_column().unwrap();
    let sd_seq = post_seq.summaries.sd[eq_s];
    let sd_pool = post_pool.summaries.sd[eq_p];
    assert!(sd_seq.is_finite() && sd_pool.is_finite() && sd_seq > 0.0 && sd_pool > 0.0);
    // Sequential conjugate updating reproduces the pooled posterior up to (a) the
    // Monte Carlo error of the draw summaries the prior is hydrated from and compared
    // on — relative SE 1/√(2D) per SD and SD/√D per mean at D = 400 draws — and (b)
    // the diagonal hydration, which drops the batch-A coefficient correlations (a
    // deterministic gap measured at 3.6% of the SD and 0.09 SD in the mean with
    // D = 40 000 on this fixture, whose treatment column is nearly orthogonal to the
    // others; a confounded design shows a larger gap). The bands are those two terms
    // with a 4σ Monte Carlo allowance, in
    // both directions: a V0 stored as the absolute SD² (orders narrower) or a lost
    // batch (√2 wider) both fall far outside.
    let draws = 400.0_f64;
    let sd_band = 0.036 + 4.0 * (1.0 / draws).sqrt();
    let ratio = sd_pool / sd_seq;
    assert!(
        (ratio - 1.0).abs() < sd_band,
        "sequential SD {sd_seq} vs pooled {sd_pool}: ratio {ratio} outside 1 ± {sd_band}"
    );
    let mean_gap = (post_seq.summaries.mean[eq_s] - post_pool.summaries.mean[eq_p]).abs();
    let mean_band = sd_pool * (0.09 + 4.0 * (2.0 / draws).sqrt());
    assert!(
        mean_gap < mean_band,
        "sequential mean {} vs pooled {}: gap {mean_gap} exceeds {mean_band}",
        post_seq.summaries.mean[eq_s],
        post_pool.summaries.mean[eq_p]
    );
}

/// Earns `tests-quality-7`: strong prior vs flat prior vs prior-dominated.
#[test]
fn strong_prior_moves_posterior_away_from_flat_and_data_moves_away_from_prior() {
    let n = 100;
    let (data, t, y, z) = linear_scm_table(n);
    let estimand = IdentifiedEstimand::backdoor(
        "backdoor.adjustment",
        Arc::from(vec![z]),
        ExprId::from_raw(0),
    );
    let query = AverageEffectQuery::binary_ate(t, y);
    let mut ws = BayesianGCompWorkspace::default();
    let flat = BayesianGComputationAte {
        backend: BayesianBackendKind::ConjugateGaussian,
        n_draws: 300,
        seed: 5,
        prior_scale: 1e4,
        ..BayesianGComputationAte::new()
    };
    let strong = BayesianGComputationAte { prior_scale: 0.05, ..flat.clone() };
    let prep = flat.prepare(&data, &estimand, &query).unwrap();
    let post_flat = flat
        .fit(
            &prep,
            IdentificationStatus::NonparametricallyIdentified,
            &mut ws,
            &ExecutionContext::for_tests(1),
        )
        .unwrap();
    let post_strong = strong
        .fit(
            &prep,
            IdentificationStatus::NonparametricallyIdentified,
            &mut ws,
            &ExecutionContext::for_tests(1),
        )
        .unwrap();
    let eq = post_flat.effect_column().unwrap();
    let m_flat = post_flat.summaries.mean[eq];
    let m_strong = post_strong.summaries.mean[eq];
    // True ATE ≈ 2; flat recovers it; strong isotropic prior at 0 pulls toward 0.
    assert!((m_flat - m_strong).abs() > 0.15, "prior ignored? flat={m_flat} strong={m_strong}");
    assert!(
        m_strong.abs() < m_flat.abs(),
        "strong prior should shrink toward 0: flat={m_flat} strong={m_strong}"
    );
    // Data not ignored: informative sample must move off the prior mean (0).
    assert!(m_strong.abs() > 0.05, "data ignored? strong posterior {m_strong} ≈ prior mean 0");
}

#[test]
fn hydrate_effect_functional_maps_treatment_coef() {
    let quantities = vec![
        PosteriorQuantityKind::Coefficient { index: 0, name: Some(Arc::from("intercept")) },
        PosteriorQuantityKind::Coefficient { index: 1, name: Some(Arc::from("coef_t")) },
        PosteriorQuantityKind::Effect { name: Arc::from("ate") },
        PosteriorQuantityKind::ResidualVariance,
    ];
    let mean = vec![0.1, 0.5, 2.0, 1.0];
    let sd = vec![1.0, 1.0, 0.4, 0.1];
    let names: Vec<Arc<str>> =
        vec![Arc::from("intercept"), Arc::from("coef_t"), Arc::from("coef_z")];
    let baseline = PriorSet::weakly_informative(3);
    let prior = hydrate_prior(
        &HydrateMapping::EffectFunctional { source_quantity: "ate".into() },
        &HydrateSource::summaries(&quantities, &mean, &sd).with_source_contrast(Some(0.5)),
        &HydrateTarget { baseline: &baseline, coef_names: &names, treatment_col: Some(1) },
    )
    .unwrap();
    let coef = prior.gaussian_coefficients().unwrap();
    // Identity ATE = Δ β_T, so β ~ N(2.0/0.5, (0.4/0.5)²).
    assert!((coef.mean[1] - 4.0).abs() < 1e-12);
    assert!((coef.variance[1] - 0.64).abs() < 1e-12);
    // Unmapped dims keep baseline (isotropic scale 10 → var 100).
    assert!((coef.mean[0] - 0.0).abs() < 1e-12);
    assert!((coef.variance[0] - 100.0).abs() < 1e-12);
    assert!((coef.variance[2] - 100.0).abs() < 1e-12);
    assert!(prior.restrictions.iter().any(|r| r.id.as_ref() == "external_effect_prior"));
}

#[test]
fn hydrate_mapping_hard_errors() {
    let quantities = vec![
        PosteriorQuantityKind::Coefficient { index: 0, name: Some(Arc::from("intercept")) },
        PosteriorQuantityKind::Coefficient { index: 1, name: Some(Arc::from("coef_t")) },
        PosteriorQuantityKind::Effect { name: Arc::from("ate") },
        PosteriorQuantityKind::ResidualVariance,
    ];
    let mean = vec![0.0, 1.0, 2.0, 1.0];
    let sd = vec![1.0, 1.0, 0.5, 0.1];
    let names2: Vec<Arc<str>> = vec![Arc::from("intercept"), Arc::from("coef_t")];
    let baseline2 = PriorSet::weakly_informative(2);
    // Identical with wrong expected dim via target names of different length than source coefs.
    let names3: Vec<Arc<str>> =
        vec![Arc::from("intercept"), Arc::from("coef_t"), Arc::from("coef_w")];
    let baseline3 = PriorSet::weakly_informative(3);
    assert!(
        hydrate_prior(
            &HydrateMapping::IdenticalCoefficientSubspace,
            &HydrateSource::summaries(&quantities, &mean, &sd),
            &HydrateTarget { baseline: &baseline3, coef_names: &names3, treatment_col: None },
        )
        .is_err()
    );

    assert!(
        hydrate_prior(
            &HydrateMapping::EffectFunctional { source_quantity: "missing".into() },
            &HydrateSource::summaries(&quantities, &mean, &sd).with_source_contrast(Some(1.0)),
            &HydrateTarget { baseline: &baseline2, coef_names: &names2, treatment_col: Some(1) },
        )
        .is_err()
    );

    assert!(
        hydrate_prior(
            &HydrateMapping::EffectFunctional { source_quantity: "ate".into() },
            &HydrateSource::summaries(&quantities, &mean, &sd),
            &HydrateTarget { baseline: &baseline2, coef_names: &names2, treatment_col: Some(1) },
        )
        .is_err()
    );
    assert!(
        hydrate_prior(
            &HydrateMapping::EffectFunctional { source_quantity: "ate".into() },
            &HydrateSource::summaries(&quantities, &mean, &sd).with_source_contrast(Some(0.0)),
            &HydrateTarget { baseline: &baseline2, coef_names: &names2, treatment_col: Some(1) },
        )
        .is_err()
    );

    assert!(
        hydrate_prior(
            &HydrateMapping::NamedParameters { pairs: vec![("ate".into(), "no_such_coef".into())] },
            &HydrateSource::summaries(&quantities, &mean, &sd),
            &HydrateTarget { baseline: &baseline2, coef_names: &names2, treatment_col: None },
        )
        .is_err()
    );

    assert!(
        hydrate_prior(
            &HydrateMapping::NamedParameters { pairs: vec![("no_src".into(), "coef_t".into())] },
            &HydrateSource::summaries(&quantities, &mean, &sd),
            &HydrateTarget { baseline: &baseline2, coef_names: &names2, treatment_col: None },
        )
        .is_err()
    );
}

#[test]
fn hydrate_effect_functional_lands_on_cate_coefficient() {
    let quantities = vec![
        PosteriorQuantityKind::Effect { name: Arc::from("ate") },
        PosteriorQuantityKind::ResidualVariance,
    ];
    let mean = vec![1.2, 1.0];
    let sd = vec![0.3, 0.1];
    let names: Vec<Arc<str>> = vec![
        Arc::from("intercept"),
        Arc::from("treatment_at_mean_modifier"),
        Arc::from("modifier"),
        Arc::from("treatment_centered_modifier"),
    ];
    let baseline = PriorSet::weakly_informative(4);
    let prior = hydrate_prior(
        &HydrateMapping::EffectFunctional { source_quantity: "ate".into() },
        &HydrateSource::summaries(&quantities, &mean, &sd).with_source_contrast(Some(0.6)),
        &HydrateTarget { baseline: &baseline, coef_names: &names, treatment_col: None },
    )
    .unwrap();
    let coef = prior.gaussian_coefficients().unwrap();
    assert!((coef.mean[1] - 2.0).abs() < 1e-12);
    assert!((coef.variance[1] - 0.25).abs() < 1e-12);
    assert!((coef.mean[3]).abs() < 1e-12);
}

#[test]
fn hydrate_named_parameters_overwrites_target() {
    let quantities = vec![
        PosteriorQuantityKind::Coefficient { index: 0, name: Some(Arc::from("intercept")) },
        PosteriorQuantityKind::Coefficient { index: 1, name: Some(Arc::from("coef_t")) },
        PosteriorQuantityKind::Effect { name: Arc::from("ate") },
        PosteriorQuantityKind::ResidualVariance,
    ];
    let mean = vec![0.0, 0.0, 1.5, 1.0];
    let sd = vec![1.0, 1.0, 0.2, 0.1];
    let names: Vec<Arc<str>> = vec![Arc::from("intercept"), Arc::from("coef_t")];
    let baseline = PriorSet::weakly_informative(2);
    let prior = hydrate_prior(
        &HydrateMapping::NamedParameters { pairs: vec![("ate".into(), "coef_t".into())] },
        &HydrateSource::summaries(&quantities, &mean, &sd),
        &HydrateTarget { baseline: &baseline, coef_names: &names, treatment_col: None },
    )
    .unwrap();
    let coef = prior.gaussian_coefficients().unwrap();
    assert!((coef.mean[1] - 1.5).abs() < 1e-12);
    assert!(prior.restrictions.iter().any(|r| r.id.as_ref() == "external_named_prior"));
}

#[test]
fn full_named_coefficient_map_permutates_dense_covariance() {
    let quantities = vec![
        PosteriorQuantityKind::Coefficient { index: 0, name: Some(Arc::from("beta_x")) },
        PosteriorQuantityKind::Coefficient { index: 1, name: Some(Arc::from("beta_y")) },
        PosteriorQuantityKind::ResidualVariance,
    ];
    let names = vec![Arc::from("target_y"), Arc::from("target_x")];
    let prior = hydrate_prior(
        &HydrateMapping::NamedParameters {
            pairs: vec![("beta_x".into(), "target_x".into()), ("beta_y".into(), "target_y".into())],
        },
        &HydrateSource::summaries(&quantities, &[1.0, 2.0, 1.0], &[2.0, 3.0, 0.0])
            .with_coefficient_covariance(Some(&[4.0, 1.0, 1.0, 9.0])),
        &HydrateTarget {
            baseline: &PriorSet::weakly_informative(2),
            coef_names: &names,
            treatment_col: None,
        },
    )
    .unwrap();
    let coefficients = prior.gaussian_coefficients().unwrap();
    assert_eq!(coefficients.mean.as_ref(), &[2.0, 1.0]);
    assert_eq!(coefficients.variance.as_ref(), &[9.0, 4.0]);
    let correlation = prior.coefficient_correlation().unwrap();
    assert!((correlation.matrix()[1] - 1.0 / 6.0).abs() < 1e-12);
    assert!(
        prior
            .restrictions
            .iter()
            .any(|restriction| { restriction.id.as_ref() == HYDRATED_COEFFICIENT_COVARIANCE_ID })
    );
    assert!(
        hydrate_prior(
            &HydrateMapping::NamedParameters {
                pairs: vec![
                    ("beta_x".into(), "target_x".into()),
                    ("beta_x".into(), "target_y".into()),
                ],
            },
            &HydrateSource::summaries(&quantities, &[1.0, 2.0, 1.0], &[2.0, 3.0, 0.0])
                .with_coefficient_covariance(Some(&[4.0, 1.0, 1.0, 9.0])),
            &HydrateTarget {
                baseline: &PriorSet::weakly_informative(2),
                coef_names: &names,
                treatment_col: None
            },
        )
        .is_err()
    );
    let mut ambiguous = quantities.clone();
    ambiguous[1] = PosteriorQuantityKind::Coefficient { index: 1, name: Some(Arc::from("beta_x")) };
    assert!(
        hydrate_prior(
            &HydrateMapping::NamedParameters { pairs: vec![("beta_x".into(), "target_x".into())] },
            &HydrateSource::summaries(&ambiguous, &[1.0, 2.0, 1.0], &[2.0, 3.0, 0.0])
                .with_coefficient_covariance(Some(&[4.0, 1.0, 1.0, 9.0])),
            &HydrateTarget {
                baseline: &PriorSet::weakly_informative(2),
                coef_names: &names,
                treatment_col: None
            },
        )
        .is_err()
    );
}

#[test]
fn hydrate_keeps_absolute_variances_for_source_without_residual_variance() {
    // A known-sigma^2 Gaussian, GLM, or draws-free summary source has no
    // residual_variance column: the absolute coefficient variance is kept and
    // marked for the target fit to convert with its own residual variance.
    let quantities = vec![
        PosteriorQuantityKind::Coefficient { index: 0, name: Some(Arc::from("intercept")) },
        PosteriorQuantityKind::Coefficient { index: 1, name: Some(Arc::from("coef_t")) },
        PosteriorQuantityKind::Effect { name: Arc::from("ate") },
    ];
    let mean = vec![0.0, 1.0, 2.0];
    let sd = vec![0.3, 0.6, 0.5];
    let names: Vec<Arc<str>> = vec![Arc::from("intercept"), Arc::from("coef_t")];
    let baseline = PriorSet::weakly_informative(2);

    let prior = hydrate_prior_from_quantity_summaries(&quantities, &mean, &sd, Some(2)).unwrap();
    let coef = prior.gaussian_coefficients().unwrap();
    assert_eq!(coef.variance.as_ref(), &[0.3 * 0.3, 0.6 * 0.6]);
    assert_eq!(prior.absolute_coefficient_scale(), Some(vec![0, 1]));

    let identical = hydrate_prior(
        &HydrateMapping::IdenticalCoefficientSubspace,
        &HydrateSource::summaries(&quantities, &mean, &sd),
        &HydrateTarget { baseline: &baseline, coef_names: &names, treatment_col: None },
    )
    .unwrap();
    assert_eq!(identical.absolute_coefficient_scale(), Some(vec![0, 1]));
    assert!(identical.residual_inv_gamma().is_some(), "baseline residual prior kept");

    // Effect functional with Δ = 0.5: absolute Var(β_T) = (0.5 / 0.5)² = 1.
    let effect = hydrate_prior(
        &HydrateMapping::EffectFunctional { source_quantity: "ate".into() },
        &HydrateSource::summaries(&quantities, &mean, &sd).with_source_contrast(Some(0.5)),
        &HydrateTarget { baseline: &baseline, coef_names: &names, treatment_col: Some(1) },
    )
    .unwrap();
    let coef = effect.gaussian_coefficients().unwrap();
    assert_eq!(coef.variance.as_ref(), &[100.0, 1.0]);
    assert_eq!(effect.absolute_coefficient_scale(), Some(vec![1]));
    assert!(effect.restrictions.iter().any(|r| r.id.as_ref() == "external_effect_prior"));

    let named = hydrate_prior(
        &HydrateMapping::NamedParameters { pairs: vec![("ate".into(), "coef_t".into())] },
        &HydrateSource::summaries(&quantities, &mean, &sd),
        &HydrateTarget { baseline: &baseline, coef_names: &names, treatment_col: None },
    )
    .unwrap();
    let coef = named.gaussian_coefficients().unwrap();
    assert_eq!(coef.variance.as_ref(), &[100.0, 0.25]);
    assert_eq!(named.absolute_coefficient_scale(), Some(vec![1]));

    // Resolving with a target σ̂² gives V0 = absolute / σ̂² on the mapped
    // coefficient only; the baseline V0 is untouched.
    let resolved = named.resolve_absolute_coefficient_scale(4.0).unwrap().unwrap();
    assert_eq!(resolved.gaussian_coefficients().unwrap().variance.as_ref(), &[100.0, 0.0625]);
    assert!(resolved.absolute_coefficient_scale().is_none());
}

#[test]
fn hydrate_with_residual_variance_is_unchanged() {
    // A source recording σ² converts at hydrate time exactly as before: V0 =
    // SD² / σ², no absolute-scale marker, and the same restriction records.
    let quantities = vec![
        PosteriorQuantityKind::Coefficient { index: 0, name: Some(Arc::from("intercept")) },
        PosteriorQuantityKind::Coefficient { index: 1, name: Some(Arc::from("coef_t")) },
        PosteriorQuantityKind::Effect { name: Arc::from("ate") },
        PosteriorQuantityKind::ResidualVariance,
    ];
    let mean = vec![0.0, 1.0, 2.0, 4.0];
    let sd = vec![0.3, 0.6, 0.5, 0.2];
    let names: Vec<Arc<str>> = vec![Arc::from("intercept"), Arc::from("coef_t")];
    let baseline = PriorSet::weakly_informative(2);

    let prior = hydrate_prior_from_quantity_summaries(&quantities, &mean, &sd, Some(2)).unwrap();
    let expected = GaussianCoefficientPrior::from_absolute_variance(
        Arc::from(vec![0.0, 1.0]),
        Arc::from(vec![0.3 * 0.3, 0.6 * 0.6]),
        4.0,
    )
    .unwrap();
    assert_eq!(prior.gaussian_coefficients().unwrap(), &expected);
    assert!(prior.restrictions.is_empty());

    let effect = hydrate_prior(
        &HydrateMapping::EffectFunctional { source_quantity: "ate".into() },
        &HydrateSource::summaries(&quantities, &mean, &sd).with_source_contrast(Some(0.5)),
        &HydrateTarget { baseline: &baseline, coef_names: &names, treatment_col: Some(1) },
    )
    .unwrap();
    assert_eq!(effect.gaussian_coefficients().unwrap().variance.as_ref(), &[100.0, 1.0 / 4.0]);
    assert!(effect.absolute_coefficient_scale().is_none());
    assert_eq!(effect.restrictions.len(), 1);
    assert!(
        effect.restrictions[0]
            .description
            .ends_with("absolute Var(β) converted to V0 with source σ²=4")
    );

    let named = hydrate_prior(
        &HydrateMapping::NamedParameters { pairs: vec![("ate".into(), "coef_t".into())] },
        &HydrateSource::summaries(&quantities, &mean, &sd),
        &HydrateTarget { baseline: &baseline, coef_names: &names, treatment_col: None },
    )
    .unwrap();
    assert_eq!(named.gaussian_coefficients().unwrap().variance.as_ref(), &[100.0, 0.25 / 4.0]);
    assert!(named.absolute_coefficient_scale().is_none());
    assert_eq!(
        named.restrictions[0].description.as_ref(),
        "external named-parameter prior (ate->coef_t); absolute posterior Var converted to \
         V0 with source σ²=4; diagonal only (off-diagonal covariance dropped)"
    );
}

/// Two batches with a treatment strongly confounded by `Z`, so the
/// treatment and `Z` coefficients are correlated in each batch's posterior.
fn confounded_table(
    batches: &[(usize, usize)],
) -> (TabularData, VariableId, VariableId, VariableId) {
    let mut b = CausalSchemaBuilder::new();
    for (name, hint) in [
        ("Z", RoleHint::Context),
        ("T", RoleHint::TreatmentCandidate),
        ("Y", RoleHint::OutcomeCandidate),
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
    let (z, t, y) = (VariableId::from_raw(0), VariableId::from_raw(1), VariableId::from_raw(2));
    let (mut zv, mut tv, mut yv) = (Vec::new(), Vec::new(), Vec::new());
    for &(n, shift) in batches {
        for i in 0..n {
            let k = i + shift;
            let zk = ((k * 37 % 101) as f64) / 101.0;
            let u = ((k * 53 % 97) as f64) / 97.0;
            let tk = if zk + 0.25 * u > 0.6 { 1.0 } else { 0.0 };
            let e = (((k * 29) % 11) as f64 - 5.0) * 0.12;
            zv.push(zk);
            tv.push(tk);
            yv.push(2.0 * tk + 1.5 * zk + e);
        }
    }
    let validity = ValidityBitmap::all_valid(zv.len());
    let cols = vec![
        OwnedColumn::Float64(Float64Column::new(z, Arc::from(zv), validity.clone()).unwrap()),
        OwnedColumn::Float64(Float64Column::new(t, Arc::from(tv), validity.clone()).unwrap()),
        OwnedColumn::Float64(Float64Column::new(y, Arc::from(yv), validity).unwrap()),
    ];
    let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
    (TabularData::new(storage), t, y, z)
}

/// Item: a known-σ² source records its σ² as a constant `ResidualVariance`
/// quantity, and hydration converts with it instead of the target's estimate.
#[test]
fn known_sigma2_source_records_its_residual_variance_for_hydration() {
    let (data, t, y, z) = confounded_table(&[(60, 0)]);
    let estimand = IdentifiedEstimand::backdoor(
        "backdoor.adjustment",
        Arc::from(vec![z]),
        ExprId::from_raw(0),
    );
    let query = AverageEffectQuery::binary_ate(t, y);
    let sigma2 = 0.3;
    let base = BayesianGComputationAte {
        backend: BayesianBackendKind::ConjugateGaussian,
        n_draws: 400,
        seed: 4,
        ..BayesianGComputationAte::new()
    };
    let prep = base.prepare(&data, &estimand, &query).unwrap();
    let mut prior = base.prior_in_force(prep.design.ncols);
    prior.specs.retain(|s| matches!(s, PriorSpec::GaussianCoefficients(_)));
    prior.push(PriorSpec::KnownResidualVariance(sigma2));
    let est = BayesianGComputationAte { prior: Some(prior), ..base };
    let post = est
        .fit(
            &prep,
            IdentificationStatus::NonparametricallyIdentified,
            &mut BayesianGCompWorkspace::default(),
            &ExecutionContext::for_tests(1),
        )
        .unwrap();
    let col = post
        .draws
        .schema
        .quantities
        .iter()
        .position(|q| matches!(q, PosteriorQuantityKind::ResidualVariance))
        .expect("a known-σ² posterior records its σ²");
    assert!(post.draws.column(col).unwrap().iter().all(|v| v.to_bits() == sigma2.to_bits()));
    let source_sigma2 = post.summaries.mean[col];
    assert!((source_sigma2 - sigma2).abs() < 1e-12);

    let hydrated = hydrate_prior_from_posterior(&post, Some(prep.design.ncols)).unwrap();
    assert!(
        hydrated.absolute_coefficient_scale().is_none(),
        "the recorded σ² converts at hydrate time; no target plug-in"
    );
    let coef = hydrated.gaussian_coefficients().unwrap();
    for i in 0..coef.len() {
        let expected = post.summaries.sd[i] * post.summaries.sd[i] / source_sigma2;
        assert!((coef.variance[i] - expected).abs() <= 1e-12 * expected.max(1.0));
    }
}

/// Item: hydrating the full coefficient covariance (dense `V0`) makes
/// sequential conjugate updating agree with pooling on a confounded design,
/// where the diagonal hydration is measurably overconfident.
#[test]
fn dense_hydration_sequential_matches_pooled_on_confounded_batches() {
    let (data_a, t, y, z) = confounded_table(&[(60, 0)]);
    let (data_b, _, _, _) = confounded_table(&[(60, 500)]);
    let (pool, _, _, _) = confounded_table(&[(60, 0), (60, 500)]);
    let estimand = IdentifiedEstimand::backdoor(
        "backdoor.adjustment",
        Arc::from(vec![z]),
        ExprId::from_raw(0),
    );
    let query = AverageEffectQuery::binary_ate(t, y);
    let sigma2 = 0.2;
    let draws = 40_000usize;
    let base = BayesianGComputationAte {
        backend: BayesianBackendKind::ConjugateGaussian,
        n_draws: draws,
        seed: 11,
        ..BayesianGComputationAte::new()
    };
    let ctx = ExecutionContext::for_tests(1);
    let mut ws = BayesianGCompWorkspace::default();
    let prep_a = base.prepare(&data_a, &estimand, &query).unwrap();
    let mut prior = base.prior_in_force(prep_a.design.ncols);
    prior.specs.retain(|s| matches!(s, PriorSpec::GaussianCoefficients(_)));
    prior.push(PriorSpec::KnownResidualVariance(sigma2));
    let known = BayesianGComputationAte { prior: Some(prior), ..base.clone() };
    let id = IdentificationStatus::NonparametricallyIdentified;
    let post_a = known.fit(&prep_a, id, &mut ws, &ctx).unwrap();

    let dense = hydrate_prior_from_posterior(&post_a, Some(prep_a.design.ncols)).unwrap();
    assert!(dense.coefficient_correlation().is_some(), "confounded source is correlated");
    assert!(dense.restrictions.iter().any(|r| r.id.as_ref() == HYDRATED_COEFFICIENT_COVARIANCE_ID));
    let mut diagonal = dense.clone();
    diagonal.specs.retain(|s| !matches!(s, PriorSpec::CoefficientCorrelation(_)));
    let with_known = |mut p: PriorSet| {
        p.push(PriorSpec::KnownResidualVariance(sigma2));
        BayesianGComputationAte { prior: Some(p), ..base.clone() }
    };

    let prep_b = base.prepare(&data_b, &estimand, &query).unwrap();
    let post_dense = with_known(dense).fit(&prep_b, id, &mut ws, &ctx).unwrap();
    let post_diag = with_known(diagonal).fit(&prep_b, id, &mut ws, &ctx).unwrap();
    let prep_pool = base.prepare(&pool, &estimand, &query).unwrap();
    let post_pool = known.fit(&prep_pool, id, &mut ws, &ctx).unwrap();

    let eff = |p: &CausalPosterior| {
        let e = p.effect_column().unwrap();
        (p.summaries.mean[e], p.summaries.sd[e])
    };
    let ((m_dense, sd_dense), (_, sd_diag), (m_pool, sd_pool)) =
        (eff(&post_dense), eff(&post_diag), eff(&post_pool));
    // Exact in the conjugate known-σ² algebra; what remains is the Monte Carlo
    // error of the hydrated moments and of the compared summaries (relative
    // SE ~ 1/√(2D) per SD); 6σ bands at D = 40 000.
    let band = 6.0 / (2.0 * draws as f64).sqrt();
    assert!(
        (sd_dense / sd_pool - 1.0).abs() < band,
        "dense sequential SD {sd_dense} vs pooled {sd_pool}"
    );
    assert!(
        (m_dense - m_pool).abs() < sd_pool * 6.0 * (2.0 / draws as f64).sqrt(),
        "dense sequential mean {m_dense} vs pooled {m_pool}"
    );
    assert!(
        sd_diag < sd_pool * (1.0 - 3.0 * band),
        "diagonal hydration should be overconfident here: diag {sd_diag} vs pooled {sd_pool}"
    );
}

/// Compatibility: summaries without draws (old or summary-only artifacts)
/// still hydrate the diagonal prior; draws too few to estimate a
/// positive-definite covariance give no covariance.
#[test]
fn summary_only_sources_keep_diagonal_hydration() {
    let quantities = vec![
        PosteriorQuantityKind::Coefficient { index: 0, name: None },
        PosteriorQuantityKind::Coefficient { index: 1, name: None },
        PosteriorQuantityKind::ResidualVariance,
    ];
    let prior = hydrate_prior_from_quantity_summaries(
        &quantities,
        &[0.0, 1.0, 2.0],
        &[1.0, 0.5, 0.1],
        None,
    )
    .unwrap();
    assert!(prior.coefficient_correlation().is_none());
    assert!(coefficient_covariance_from_draws(&quantities, &[], 0).unwrap().is_none());
    assert!(
        coefficient_covariance_from_draws(&quantities, &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], 2)
            .unwrap()
            .is_none()
    );
    assert!(coefficient_covariance_from_draws(&quantities, &[0.0; 5], 2).is_err());
    // A non-positive-definite covariance falls back to the diagonal, recorded.
    let singular = [1.0, 1.0, 1.0, 1.0];
    let prior = hydrate_prior_from_coefficient_moments(
        &quantities,
        &[0.0, 1.0, 2.0],
        &[1.0, 1.0, 0.1],
        Some(&singular),
        None,
    )
    .unwrap();
    assert!(prior.coefficient_correlation().is_none());
    assert!(prior.restrictions.iter().any(|r| r.id.as_ref() == HYDRATED_COVARIANCE_DROPPED_ID));
}

/// Binary treatment confounded by `z`; the effect `1 + 2 z` grows in `z`, so the effect on the
/// treated differs from the all-observed effect. A small deterministic wiggle keeps the
/// residual variance positive.
fn arm_effect_table(n: usize) -> (TabularData, f64, f64) {
    let z: Vec<f64> = (0..n).map(|i| (i as f64 / 7.0).sin() * 1.5).collect();
    let t: Vec<f64> =
        (0..n).map(|i| if z[i] + (i as f64 / 3.0).cos() > 0.0 { 1.0 } else { 0.0 }).collect();
    let y: Vec<f64> =
        (0..n).map(|i| t[i] * (1.0 + 2.0 * z[i]) + z[i] + (i as f64 / 5.0).sin() * 0.1).collect();
    let treated: Vec<usize> = (0..n).filter(|&i| t[i] > 0.5).collect();
    let untreated: Vec<usize> = (0..n).filter(|&i| t[i] < 0.5).collect();
    let att = treated.iter().map(|&i| 1.0 + 2.0 * z[i]).sum::<f64>() / treated.len() as f64;
    let atc = untreated.iter().map(|&i| 1.0 + 2.0 * z[i]).sum::<f64>() / untreated.len() as f64;
    let data = TabularData::from_f64_columns([
        ("t", t.as_slice()),
        ("y", y.as_slice()),
        ("z", z.as_slice()),
    ])
    .unwrap();
    (data, att, atc)
}

fn arm_effect_posterior_mean(population: antecedent_core::TargetPopulation) -> (f64, usize) {
    let (data, _, _) = arm_effect_table(400);
    let estimand = IdentifiedEstimand::backdoor(
        "backdoor.adjustment",
        Arc::from(vec![VariableId::from_raw(2)]),
        ExprId::from_raw(0),
    );
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
        .with_target_population(population);
    let bayes = BayesianGComputationAte {
        backend: BayesianBackendKind::ConjugateGaussian,
        n_draws: 2000,
        seed: 9,
        prior_scale: 100.0,
        ..BayesianGComputationAte::new()
    };
    let prep = bayes.prepare(&data, &estimand, &query).unwrap();
    let mut ws = BayesianGCompWorkspace::default();
    let post = bayes
        .fit(
            &prep,
            IdentificationStatus::NonparametricallyIdentified,
            &mut ws,
            &ExecutionContext::for_tests(3),
        )
        .unwrap();
    let eq = post.effect_column().unwrap();
    (post.summaries.mean[eq], prep.design.ncols)
}

#[test]
fn bayesian_gcomp_on_the_treated_and_untreated_recovers_each_arms_model_based_effect() {
    let (_, att, atc) = arm_effect_table(400);
    let (treated, treated_cols) =
        arm_effect_posterior_mean(antecedent_core::TargetPopulation::Treated);
    let (untreated, _) = arm_effect_posterior_mean(antecedent_core::TargetPopulation::Untreated);
    let (all, all_cols) = arm_effect_posterior_mean(antecedent_core::TargetPopulation::AllObserved);
    // Intercept, treatment, z, and one centred treatment-by-z interaction on an arm.
    assert_eq!((treated_cols, all_cols), (4, 3));
    assert!((treated - att).abs() < 0.05, "treated {treated} vs {att}");
    assert!((untreated - atc).abs() < 0.05, "untreated {untreated} vs {atc}");
    assert!((att - atc).abs() > 0.3, "the arms must differ for the check to bite");
    assert!(all.is_finite());
}

#[test]
fn bayesian_gcomp_arm_effect_centres_interactions_and_refuses_unsupported_targets() {
    let (data, _, _) = arm_effect_table(400);
    let estimand = IdentifiedEstimand::backdoor(
        "backdoor.adjustment",
        Arc::from(vec![VariableId::from_raw(2)]),
        ExprId::from_raw(0),
    );
    let base = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let on_treated =
        base.clone().with_target_population(antecedent_core::TargetPopulation::Treated);
    let prep = BayesianGComputationAte::conjugate().prepare(&data, &estimand, &on_treated).unwrap();
    let n = prep.design.nrows;
    let treatment = &prep.design.matrix[n..2 * n];
    let interaction = &prep.design.matrix[3 * n..4 * n];
    let treated_rows = treatment.iter().filter(|&&t| t > 0.5).count();
    let mean_over_treated: f64 =
        interaction.iter().zip(treatment).filter(|(_, t)| **t > 0.5).map(|(x, _)| *x).sum::<f64>()
            / treated_rows as f64;
    assert!(mean_over_treated.abs() < 1e-12, "{mean_over_treated}");
    let names = prep.coef_names.unwrap();
    assert_eq!(names[3].as_ref(), "treatment_x_centered_z");

    // A non-identity outcome model does not make the treatment coefficient the arm contrast.
    let logit = BayesianGComputationAte::new().with_likelihood(BayesLikelihood::BernoulliLogit);
    let error = logit.prepare(&data, &estimand, &on_treated).unwrap_err().to_string();
    assert!(error.contains("Gaussian identity"), "{error}");
    // A predicate population stays unsupported.
    let predicate = base.with_target_population(antecedent_core::TargetPopulation::Predicate(
        antecedent_core::PredicateExpr::rows(vec![0, 1, 2]),
    ));
    let error =
        BayesianGComputationAte::conjugate().prepare(&data, &estimand, &predicate).unwrap_err();
    assert!(error.to_string().contains("Treated or Untreated"), "{error}");
}
