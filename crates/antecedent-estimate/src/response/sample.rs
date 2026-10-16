//! Complete-case samples shared across response estimators.

use std::sync::Arc;

use antecedent_core::VariableId;
use antecedent_data::{TableView, TabularData};

use crate::EstimationError;
use crate::util::range;

/// Original row indices as the `u32` row ids influence scores carry; a row past the
/// `u32` range is an error rather than a saturated id that would alias another row.
pub(super) fn row_ids_u32(rows: &[usize]) -> Result<Arc<[u32]>, EstimationError> {
    rows.iter()
        .map(|&i| {
            u32::try_from(i)
                .map_err(|_| EstimationError::data_msg("row index exceeds the u32 row-id capacity"))
        })
        .collect()
}

pub(super) struct CompleteSample {
    /// Original dataframe row index of each retained complete row.
    pub(super) keep: Vec<usize>,
    pub(super) outcome: Vec<f64>,
    pub(super) treatments: Vec<f64>,
    pub(super) treatment_matrix: Vec<f64>,
    pub(super) adjustment: Vec<f64>,
    pub(super) treatment_cols: usize,
    pub(super) adjustment_cols: usize,
    pub(super) raw_cols: usize,
}

pub(super) fn read_shared_complete_samples(
    data: &TabularData,
    outcomes: &[VariableId],
    treatments: &[VariableId],
    adjustment: &[VariableId],
) -> Result<Vec<CompleteSample>, EstimationError> {
    let mut samples = Vec::with_capacity(outcomes.len());
    for &outcome in outcomes {
        samples.push(CompleteSample::read(data, outcome, treatments, adjustment)?);
    }
    if let Some(first) = samples.first() {
        if samples.iter().any(|sample| sample.keep != first.keep) {
            return Err(EstimationError::unsupported(
                "plug-in Jacobian and directional derivatives require a shared complete-case row set across outcomes",
            ));
        }
    }
    Ok(samples)
}

impl CompleteSample {
    pub(super) fn read(
        data: &TabularData,
        outcome: VariableId,
        treatments: &[VariableId],
        adjustment: &[VariableId],
    ) -> Result<Self, EstimationError> {
        let duplicate_treatment =
            treatments.iter().enumerate().any(|(i, value)| treatments[i + 1..].contains(value));
        let duplicate_adjustment =
            adjustment.iter().enumerate().any(|(i, value)| adjustment[i + 1..].contains(value));
        if treatments.is_empty()
            || duplicate_treatment
            || duplicate_adjustment
            || adjustment.iter().any(|v| treatments.contains(v) || *v == outcome)
        {
            return Err(EstimationError::unsupported(
                "treatments and adjustments must be unique and adjustments distinct from outcome",
            ));
        }
        let y = data.float64_values(outcome)?;
        let treatment_values: Vec<Vec<f64>> =
            treatments.iter().map(|&v| data.float64_values(v)).collect::<Result<_, _>>()?;
        let adjustment_values: Vec<Vec<f64>> =
            adjustment.iter().map(|&v| data.float64_values(v)).collect::<Result<_, _>>()?;
        let keep: Vec<usize> = (0..data.row_count())
            .filter(|&i| {
                y[i].is_finite()
                    && treatment_values.iter().all(|c| c[i].is_finite())
                    && adjustment_values.iter().all(|c| c[i].is_finite())
            })
            .collect();
        if keep.len() < 20 {
            return Err(EstimationError::unsupported(
                "continuous-response estimation requires at least 20 complete rows",
            ));
        }
        let outcome = keep.iter().map(|&i| y[i]).collect();
        let mut treatment_matrix = Vec::with_capacity(keep.len() * treatments.len());
        for column in &treatment_values {
            treatment_matrix.extend(keep.iter().map(|&i| column[i]));
        }
        let treatments = treatment_matrix[..keep.len()].to_vec();
        let mut adjustment_matrix = Vec::with_capacity(keep.len() * adjustment.len());
        for column in &adjustment_values {
            adjustment_matrix.extend(keep.iter().map(|&i| column[i]));
        }
        Ok(Self {
            keep,
            outcome,
            treatments,
            treatment_matrix,
            adjustment: adjustment_matrix,
            treatment_cols: treatment_values.len(),
            adjustment_cols: adjustment_values.len(),
            raw_cols: treatment_values.len() + adjustment_values.len(),
        })
    }

    pub(super) fn len(&self) -> usize {
        self.outcome.len()
    }

    #[cfg(test)]
    pub(super) fn raw_row(&self, row: usize) -> Vec<f64> {
        let mut out = vec![0.0; self.raw_cols];
        self.write_raw_row(row, &mut out);
        out
    }

    pub(super) fn write_raw_row(&self, row: usize, out: &mut [f64]) {
        debug_assert_eq!(out.len(), self.raw_cols);
        for (col, slot) in out.iter_mut().enumerate().take(self.treatment_cols) {
            *slot = self.treatment_matrix[col * self.len() + row];
        }
        for col in 0..self.adjustment_cols {
            out[self.treatment_cols + col] = self.adjustment[col * self.len() + row];
        }
    }

    pub(super) fn write_adjustment_row(&self, row: usize, out: &mut [f64]) {
        debug_assert_eq!(out.len(), self.adjustment_cols);
        for (col, slot) in out.iter_mut().enumerate().take(self.adjustment_cols) {
            *slot = self.adjustment[col * self.len() + row];
        }
    }

    #[cfg(test)]
    pub(super) fn adjustment_row(&self, row: usize) -> Vec<f64> {
        let mut out = vec![0.0; self.adjustment_cols];
        self.write_adjustment_row(row, &mut out);
        out
    }

    pub(super) fn raw_subset(&self, rows: &[usize]) -> Vec<f64> {
        let mut out = Vec::with_capacity(rows.len() * self.raw_cols);
        for col in 0..self.treatment_cols {
            out.extend(rows.iter().map(|&row| self.treatment_matrix[col * self.len() + row]));
        }
        for col in 0..self.adjustment_cols {
            out.extend(rows.iter().map(|&row| self.adjustment[col * self.len() + row]));
        }
        out
    }

    pub(super) fn adjustment_subset(&self, rows: &[usize]) -> Vec<f64> {
        let mut out = Vec::with_capacity(rows.len() * self.adjustment_cols);
        for col in 0..self.adjustment_cols {
            out.extend(rows.iter().map(|&row| self.adjustment[col * self.len() + row]));
        }
        out
    }

    pub(super) fn train_treatment_mean(&self, rows: &[usize]) -> f64 {
        rows.iter().map(|&i| self.treatments[i]).sum::<f64>() / rows.len() as f64
    }

    pub(super) fn treatment_column_range(&self, col: usize) -> (f64, f64) {
        range(&self.treatment_matrix[col * self.len()..(col + 1) * self.len()])
    }
}
