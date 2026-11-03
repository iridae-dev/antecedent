//! Penalized-propensity AIPW and batch-retarget posterior means, run from the cross-fitted score
//! table the routes themselves produce, against the normal-normal update written out by hand.
//!
//! The score table comes from the AIPW fit on seeded data; the expected posterior mean is built
//! here from the table's treated-minus-control contrast (mean `m`, sample variance `s^2`):
//! `(n / s^2) / (n / s^2 + 1 / scale^2) * m`, at a prior scale that visibly shrinks `m`. The
//! table's contrast mean is also tied to the fitted effect and to the retargeted contrast.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::needless_range_loop,
    reason = "test fixtures index small literals and build dense designs element by element"
)]

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
    AipwAte, AipwWorkspace, EffectEstimate, EstimationError, ScoreTable, retarget,
};
use antecedent_expr::{ExprId, IdentifiedEstimand};
use antecedent_kernels::standard_normal;

/// Raw columns of one synthetic data set.
#[derive(Clone)]
struct Raw {
    t: Vec<f64>,
    y: Vec<f64>,
    z: Vec<Vec<f64>>,
}

/// `Y = 2 T + z0 + 0.5 z1 + noise`: the true average effect is 2.
fn draw(n: usize, p: usize, seed: u64) -> Raw {
    let mut rng = ExecutionContext::for_tests(seed).rng.stream_for(StreamDomain::Estimate, 0xE2);
    let mut z = vec![vec![0.0; n]; p];
    let mut t = vec![0.0; n];
    let mut y = vec![0.0; n];
    for i in 0..n {
        for column in &mut z {
            column[i] = standard_normal(&mut rng);
        }
        t[i] = {
            let eta = 0.8 * z[0][i] - 0.5 * z[1][i];
            f64::from(u8::from(rng.next_f64() < 1.0 / (1.0 + (-eta).exp())))
        };
        y[i] = 2.0 * t[i] + z[0][i] + 0.5 * z[1][i] + 0.5 * standard_normal(&mut rng);
    }
    Raw { t, y, z }
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

fn plain() -> AipwAte {
    AipwAte::new().with_bootstrap_replicates(0)
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
    estimate.scores.table.as_ref().expect("the penalized route keeps its score table")
}

/// Hand normal-normal update of the contrast `treated - control` of a score table.
fn hand_posterior(table: &ScoreTable, scale: f64) -> (f64, f64) {
    let d: Vec<f64> =
        table.column(0).unwrap().iter().zip(table.column(1).unwrap()).map(|(c, t)| t - c).collect();
    let n = d.len() as f64;
    let m = d.iter().sum::<f64>() / n;
    let s2 = d.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (n - 1.0);
    let precision = n / s2;
    (m, precision / (precision + 1.0 / (scale * scale)) * m)
}

#[test]
fn penalized_posterior_mean_equals_the_hand_update_of_the_route_score_table() {
    for (seed, scale) in [(7_u64, 0.8), (13, 2.5)] {
        let raw = draw(80, 2, seed);
        let estimate = run(&plain(), &raw, seed).unwrap();
        let t = table(&estimate);
        let (m, want) = hand_posterior(t, scale);
        // The table is the route's: its contrast mean is the fitted effect.
        assert!((m - estimate.ate).abs() < 1e-7, "{m} vs {}", estimate.ate);
        let got = score_contrast_posterior(
            t.column(0).unwrap(),
            t.column(1).unwrap(),
            PosteriorRequest::IndependentNormal { scale },
            ScoreNamespace::Penalized,
        )
        .unwrap();
        assert!((got - want).abs() < 1e-12, "{got} vs {want}");
        assert!((got - m).abs() > 1e-4, "the prior must move the mean off the score mean");
    }
}

#[test]
fn batch_retarget_posterior_mean_equals_the_hand_update_of_the_retargeted_score_table() {
    for (seed, scale) in [(11_u64, 0.8), (17, 2.5)] {
        let raw = draw(80, 2, seed);
        let estimate = run(&plain(), &raw, seed).unwrap();
        let t = table(&estimate);
        let ones = vec![1.0; t.n_rows];
        let (retargeted, _) = retarget(t, &ones, &[], None, None, None).unwrap();
        let (m, want) = hand_posterior(t, scale);
        // Unit weights: the retargeted contrast is the score-table contrast mean.
        assert!((retargeted.contrast.as_ref().unwrap().value - m).abs() < 1e-7);
        let got = score_contrast_posterior(
            t.column(0).unwrap(),
            t.column(1).unwrap(),
            PosteriorRequest::IndependentNormal { scale },
            ScoreNamespace::BatchRetarget,
        )
        .unwrap();
        assert!((got - want).abs() < 1e-12, "{got} vs {want}");
        assert!((got - m).abs() > 1e-4);
    }
}
