//! Row influence of the response-derivative functionals.
//!
//! Each column is the per-row score `φ_i` of one published coordinate, centred so that the
//! coordinate's estimate is `θ̂ + mean(φ)` to first order and `Var(θ̂) ≈ Σ φ_i² / (n (n − 1))`,
//! the convention [`crate::joint_influence_covariance`] reads. Columns of the several
//! completions of a graph class are placed on the same data rows by their `row_index`, so a
//! mixture over completions that share rows has a joint covariance and a sampling band even
//! when the completions' adjustment sets differ.

use antecedent_core::{DerivativeScale, DerivativeWeighting, ResponseFunctional, ResponseQuery};
use antecedent_data::TabularData;
use antecedent_stats::{gaussian_local_quadratic_bias_corrected_influence, silverman_bandwidth};

use crate::EstimationError;

use super::sample::{CompleteSample, row_ids};
use super::{
    ContinuousResponseEstimator, PseudoOutcome, ResponseInfluence, local_design_refusal,
    transform_point_derivative_gradient,
};

impl ContinuousResponseEstimator {
    /// The row influence of a derivative functional's published coordinates, one column per
    /// coordinate in the order the value lists them (a Jacobian row-major, a directional
    /// derivative one per outcome). `None` for a functional whose coordinates depend on the
    /// fitted level through a nonlinear scale that has no row influence here (a Jacobian on a
    /// log-outcome or log-log scale).
    ///
    /// The influence is computed from the same cross-fitted nuisances and local fits the
    /// point route publishes, so its standard error agrees with the route's own to first order.
    ///
    /// # Errors
    ///
    /// The refusals of the corresponding point route, or a functional that is not a
    /// derivative.
    pub fn derivative_influence(
        &self,
        data: &TabularData,
        query: &ResponseQuery,
    ) -> Result<Option<ResponseInfluence>, EstimationError> {
        match &query.functional {
            ResponseFunctional::PointDerivative { outcome, treatment, at, order, scale } => self
                .point_derivative_influence(data, *outcome, *treatment, *at, *order, *scale)
                .map(Some),
            ResponseFunctional::AverageDerivative { outcome, treatment, weighting } => {
                if !matches!(weighting, DerivativeWeighting::Observed) {
                    return Ok(None);
                }
                let sample =
                    CompleteSample::read(data, *outcome, &[*treatment], &self.adjustment_set)?;
                let scores = self.cross_fitted_ade_scores(&sample)?.scores;
                let mean = scores.iter().sum::<f64>() / scores.len() as f64;
                Ok(Some(ResponseInfluence {
                    columns: vec![scores.iter().map(|score| score - mean).collect()],
                    row_index: row_ids(&sample.keep)?,
                }))
            }
            ResponseFunctional::Jacobian { outcomes, treatments, at, scale } => {
                let factors = match scale {
                    DerivativeScale::Identity => vec![1.0; treatments.len()],
                    DerivativeScale::LogTreatment => at.to_vec(),
                    DerivativeScale::LogOutcome | DerivativeScale::LogLog => return Ok(None),
                };
                let run = self.plugin_gradient_run(data, outcomes, treatments, at, true)?;
                let mut columns = Vec::with_capacity(outcomes.len() * treatments.len());
                for influence in &run.influence {
                    for (column, factor) in influence.columns.iter().zip(&factors) {
                        columns.push(column.iter().map(|value| factor * value).collect());
                    }
                }
                Ok(Some(ResponseInfluence { columns, row_index: row_ids(&run.keep)? }))
            }
            ResponseFunctional::DirectionalDerivative { outcomes, treatments, at, direction } => {
                let run = self.plugin_gradient_run(data, outcomes, treatments, at, true)?;
                let mut columns = Vec::with_capacity(outcomes.len());
                for influence in &run.influence {
                    let mut combined = vec![0.0; run.n];
                    for (column, weight) in influence.columns.iter().zip(direction.iter()) {
                        for (slot, value) in combined.iter_mut().zip(column) {
                            *slot += weight * value;
                        }
                    }
                    columns.push(combined);
                }
                Ok(Some(ResponseInfluence { columns, row_index: row_ids(&run.keep)? }))
            }
            _ => Err(EstimationError::unsupported_coded(
                antecedent_core::reason_code!("route_not_supported"),
                "derivative influence serves the derivative functionals only",
            )),
        }
    }

    /// Row influence of the robust bias-corrected point derivative on its published scale:
    /// the delta-method gradient of the scale applied to the per-row influence of the
    /// corrected `(level, slope, curvature)`, times the row count.
    fn point_derivative_influence(
        &self,
        data: &TabularData,
        outcome: antecedent_core::VariableId,
        treatment: antecedent_core::VariableId,
        at: f64,
        order: u8,
        scale: DerivativeScale,
    ) -> Result<ResponseInfluence, EstimationError> {
        let sample = CompleteSample::read(data, outcome, &[treatment], &self.adjustment_set)?;
        let PseudoOutcome { values: pseudo, .. } = self.cross_fitted_pseudo_outcome(&sample)?;
        let bandwidth = self.options.bandwidth.unwrap_or(silverman_bandwidth(&sample.treatments)?);
        let (corrected, rows) = gaussian_local_quadratic_bias_corrected_influence(
            &sample.treatments,
            &pseudo,
            at,
            bandwidth,
            None,
        )
        .map_err(|err| local_design_refusal(err, &sample.treatments, at, bandwidth))?;
        let gradient = transform_point_derivative_gradient(
            corrected.value,
            corrected.first_derivative,
            corrected.second_derivative,
            at,
            order,
            scale,
        );
        let n = rows.len() as f64;
        let mut column: Vec<f64> = rows
            .iter()
            .map(|psi| n * (gradient[0] * psi[0] + gradient[1] * psi[1] + gradient[2] * psi[2]))
            .collect();
        let center = column.iter().sum::<f64>() / n;
        for value in &mut column {
            *value -= center;
        }
        Ok(ResponseInfluence { columns: vec![column], row_index: row_ids(&sample.keep)? })
    }
}
