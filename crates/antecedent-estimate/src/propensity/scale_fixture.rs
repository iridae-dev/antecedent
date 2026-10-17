//! Seeded confounded populations with an exact treated count, for the matching scale tests.
//!
//! Covariates are standard normal (optionally snapped to a lattice, which makes the fitted
//! scores and the covariate distances tie, or copied between rows, which duplicates donors).
//! Exactly `treated` units are treated, drawn without replacement with selection weights
//! `σ(η)`, `η = −2 + c Σₖ zₖ`, so treatment is confounded by the covariates. The outcome is
//! `0.5·T + 0.5 Σₖ zₖ + N(0, 1)`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    AverageEffectQuery, ExecutionContext, StreamDomain, TargetPopulation, VariableId,
};
use antecedent_data::TabularData;
use antecedent_expr::{ExprId, IdentifiedEstimand};
use antecedent_kernels::{sigmoid, standard_normal};

/// Effect of treatment on the outcome.
pub(crate) const EFFECT: f64 = 0.5;

/// How a population is drawn.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Design {
    /// Rows (donors and treated together).
    pub(crate) rows: usize,
    /// Exactly this many rows are treated.
    pub(crate) treated: usize,
    /// Adjustment covariates.
    pub(crate) covariates: usize,
    /// Seed of the generator.
    pub(crate) seed: u64,
    /// Snap every covariate to multiples of this step.
    pub(crate) lattice: Option<f64>,
    /// Make the covariates of every row in the second half a copy of an earlier row's.
    pub(crate) duplicated: bool,
}

/// One drawn population.
pub(crate) struct Population {
    pub(crate) treatment: Vec<f64>,
    pub(crate) outcome: Vec<f64>,
    pub(crate) covariates: Vec<Vec<f64>>,
}

impl Design {
    pub(crate) fn population(&self) -> Population {
        let n = self.rows;
        let ctx = ExecutionContext::for_tests(self.seed);
        let mut rng = ctx.rng.stream_for(StreamDomain::Test, 0x6D53);
        let mut covariates: Vec<Vec<f64>> = (0..self.covariates)
            .map(|_| (0..n).map(|_| standard_normal(&mut rng)).collect())
            .collect();
        if let Some(step) = self.lattice {
            for column in &mut covariates {
                for value in column.iter_mut() {
                    *value = (*value / step).round() * step;
                }
            }
        }
        if self.duplicated {
            let half = n / 2;
            for row in half..n {
                let source = usize::try_from(rng.next_u64() % half as u64).unwrap_or(0);
                for column in &mut covariates {
                    column[row] = column[source];
                }
            }
        }
        let weight = 0.9 / (self.covariates as f64).sqrt();
        // Weighted sampling without replacement (Efraimidis–Spirakis): the `treated` rows with
        // the largest keys `ln(u) / w`.
        let keys: Vec<f64> = (0..n)
            .map(|row| {
                let score: f64 = covariates.iter().map(|column| column[row]).sum();
                let selection = sigmoid(-2.0 + weight * score);
                rng.next_f64().max(f64::MIN_POSITIVE).ln() / selection
            })
            .collect();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| keys[b].total_cmp(&keys[a]));
        let mut treatment = vec![0.0; n];
        for &row in order.iter().take(self.treated) {
            treatment[row] = 1.0;
        }
        let outcome = (0..n)
            .map(|row| {
                let score: f64 = covariates.iter().map(|column| column[row]).sum();
                EFFECT * treatment[row] + 0.5 * score + standard_normal(&mut rng)
            })
            .collect();
        Population { treatment, outcome, covariates }
    }
}

impl Population {
    /// The first `rows` rows (rows are exchangeable, so this is a random subsample).
    pub(crate) fn first(&self, rows: usize) -> Self {
        Self {
            treatment: self.treatment[..rows].to_vec(),
            outcome: self.outcome[..rows].to_vec(),
            covariates: self.covariates.iter().map(|column| column[..rows].to_vec()).collect(),
        }
    }

    /// The population as a table (`t`, `y`, `z0`, …) with the backdoor estimand adjusting for
    /// every covariate.
    pub(crate) fn table(&self) -> (TabularData, IdentifiedEstimand) {
        let names: Vec<String> = (0..self.covariates.len()).map(|k| format!("z{k}")).collect();
        let mut columns: Vec<(&str, &[f64])> =
            vec![("t", self.treatment.as_slice()), ("y", self.outcome.as_slice())];
        columns.extend(
            names
                .iter()
                .zip(&self.covariates)
                .map(|(name, column)| (name.as_str(), column.as_slice())),
        );
        let data = TabularData::from_f64_columns(columns).expect("the columns are equal length");
        let adjustment: Vec<VariableId> = (0..self.covariates.len())
            .map(|k| VariableId::from_raw(u32::try_from(k + 2).expect("few covariates")))
            .collect();
        let estimand = IdentifiedEstimand::backdoor(
            "backdoor.adjustment",
            Arc::from(adjustment),
            ExprId::from_raw(0),
        );
        (data, estimand)
    }
}

/// The effect on the treated.
pub(crate) fn att_query() -> AverageEffectQuery {
    AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
        .with_target_population(TargetPopulation::Treated)
}
