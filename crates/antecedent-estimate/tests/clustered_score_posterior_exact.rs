//! The clustered double machine learning posterior mean, run from the cross-fitted score table
//! the cluster-DML route produces, against the normal-normal update written out by hand.
//!
//! The route is fitted on seeded clustered data with whole clusters owning folds; the expected
//! posterior mean is built here from the table's treated-minus-control contrast (mean `m`,
//! sample variance `s^2`) as `(n / s^2) / (n / s^2 + 1 / scale^2) * m`, at prior scales that
//! visibly shrink `m`, and the contrast mean is tied to the fitted effect.

use std::sync::Arc;

use antecedent_core::{
    AssumptionSet, AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, MeasurementSpec,
    RoleHint, SmallRoleSet, StreamDomain, ValueType, VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
};
use antecedent_estimate::posterior_mean::{
    PosteriorRequest, ScoreNamespace, score_contrast_posterior,
};
use antecedent_estimate::{
    AipwAte, AipwWorkspace, ClusterDml, EffectEstimate, EstimationError, ScoreTable,
};
use antecedent_expr::{ExprId, IdentifiedEstimand};
use antecedent_kernels::standard_normal;

/// Raw columns of one clustered data set; `cluster[i]` is the label of row `i`.
#[derive(Clone)]
struct Raw {
    t: Vec<f64>,
    y: Vec<f64>,
    z: Vec<Vec<f64>>,
    cluster: Vec<u32>,
}

/// `groups` clusters of `size` rows. Each cluster has a shared covariate shift, a shared
/// outcome shock and its own treatment effect `2 + u_g` (`u_g ~ N(0, 1)`), so rows of a cluster
/// are dependent. `Y = (2 + u_g) T + z0 + shock + noise`: the true average effect is 2.
fn draw(groups: usize, size: usize, seed: u64) -> Raw {
    let mut rng = ExecutionContext::for_tests(seed).rng.stream_for(StreamDomain::Estimate, 0xE4);
    let n = groups * size;
    let mut raw =
        Raw { t: vec![0.0; n], y: vec![0.0; n], z: vec![vec![0.0; n]; 2], cluster: vec![0; n] };
    for g in 0..groups {
        let shift = 0.7 * standard_normal(&mut rng);
        let shock = 1.5 * standard_normal(&mut rng);
        let effect = 2.0 + standard_normal(&mut rng);
        for k in 0..size {
            let i = g * size + k;
            let z0 = shift + standard_normal(&mut rng);
            let z1 = standard_normal(&mut rng);
            let eta = 0.6 * z0 - 0.4 * z1;
            let t = f64::from(u8::from(rng.next_f64() < 1.0 / (1.0 + (-eta).exp())));
            raw.t[i] = t;
            raw.z[0][i] = z0;
            raw.z[1][i] = z1;
            raw.y[i] = effect * t + z0 + 0.5 * z1 + shock + 0.5 * standard_normal(&mut rng);
            raw.cluster[i] = u32::try_from(g).unwrap();
        }
    }
    raw
}

fn build(raw: &Raw) -> (TabularData, IdentifiedEstimand, AverageEffectQuery) {
    let n = raw.t.len();
    let p = raw.z.len();
    let mut builder = CausalSchemaBuilder::new();
    for index in 0..p + 2 {
        let (name, hint) = match index {
            0 => ("t".to_string(), RoleHint::TreatmentCandidate),
            1 => ("y".to_string(), RoleHint::OutcomeCandidate),
            j => (format!("z{}", j - 2), RoleHint::Context),
        };
        builder
            .add_variable(
                name,
                ValueType::Continuous,
                SmallRoleSet::from_hint(hint),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
    }
    let schema = builder.build().unwrap();
    let mut values = vec![raw.t.clone(), raw.y.clone()];
    values.extend(raw.z.iter().cloned());
    let columns = values
        .into_iter()
        .enumerate()
        .map(|(i, v)| {
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(u32::try_from(i).unwrap()),
                    Arc::from(v),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            )
        })
        .collect();
    let storage = OwnedColumnarStorage::try_new(schema, columns, None, None).unwrap();
    let adjustment: Vec<VariableId> =
        (0..p).map(|j| VariableId::from_raw(u32::try_from(j + 2).unwrap())).collect();
    let estimand = IdentifiedEstimand::backdoor(
        "backdoor.adjustment",
        Arc::from(adjustment),
        ExprId::from_raw(0),
    );
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    (TabularData::new(storage), estimand, query)
}

fn declared(raw: &Raw, min_clusters: usize) -> AipwAte {
    AipwAte { cluster_dml: Some(ClusterDml::new(min_clusters).unwrap()), ..AipwAte::new() }
        .with_bootstrap_replicates(0)
        .with_cluster_ids(raw.cluster.clone())
}

fn run(est: &AipwAte, raw: &Raw, seed: u64) -> Result<EffectEstimate, EstimationError> {
    let (data, estimand, query) = build(raw);
    let problem = est.prepare(&data, &estimand, &query)?;
    est.fit(
        &problem,
        &mut AipwWorkspace::default(),
        &ExecutionContext::for_tests(seed),
        AssumptionSet::new(),
    )
}

fn table(estimate: &EffectEstimate) -> &ScoreTable {
    estimate.scores.table.as_ref().expect("the cluster-DML route keeps its score table")
}

#[test]
fn clustered_posterior_mean_equals_the_hand_update_of_the_route_score_table() {
    for (seed, scale) in [(5_u64, 0.5), (9, 2.0)] {
        let raw = draw(24, 8, seed);
        let fit = run(&declared(&raw, 10), &raw, seed).unwrap();
        let t = table(&fit);
        let d: Vec<f64> = t
            .column(0)
            .unwrap()
            .iter()
            .zip(t.column(1).unwrap())
            .map(|(control, treated)| treated - control)
            .collect();
        let n = d.len() as f64;
        let m = d.iter().sum::<f64>() / n;
        let s2 = d.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (n - 1.0);
        let precision = n / s2;
        let want = precision / (precision + 1.0 / (scale * scale)) * m;
        assert!((m - fit.ate).abs() < 1e-7, "{m} vs {}", fit.ate);
        let got = score_contrast_posterior(
            t.column(0).unwrap(),
            t.column(1).unwrap(),
            PosteriorRequest::IndependentNormal { scale },
            ScoreNamespace::Clustered,
        )
        .unwrap();
        assert!((got - want).abs() < 1e-12, "{got} vs {want}");
        assert!((got - m).abs() > 1e-4, "the prior must move the mean off the score mean");
    }
}
