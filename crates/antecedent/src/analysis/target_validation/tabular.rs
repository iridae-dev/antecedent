//! Tabular perturbation and refitting.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    Arc, CausalQuery, CausalRng, ExecutionContext, Modal, OwnedColumn, PreparedRefit, TableView,
    TabularData, TargetPerturbation, TargetRefitError, VariableId, data_failed, mask_from, shuffle,
    unbiased_index, unsupported,
};

impl PreparedRefit<'_> {
    pub(super) fn tabular_refit(
        &self,
        data: &TabularData,
        perturbation: TargetPerturbation,
        rng: &mut CausalRng,
        ctx: &ExecutionContext,
    ) -> Result<Vec<f64>, TargetRefitError> {
        let storage = data.storage();
        let replace = |id: VariableId, values: Arc<[f64]>| {
            data.with_replaced_float(id, values).map_err(data_failed)
        };
        match perturbation {
            TargetPerturbation::PlaceboTreatment { .. } => {
                let values = Self::permuted(Self::column_of(storage, self.treatment)?, rng, false);
                self.reestimate(&Modal::Tabular(replace(self.treatment, values)?), ctx)
            }
            TargetPerturbation::PlaceboOutcome { .. } => {
                let values = Self::permuted(Self::column_of(storage, self.outcome)?, rng, false);
                self.reestimate(&Modal::Tabular(replace(self.outcome, values)?), ctx)
            }
            TargetPerturbation::RandomCommonCause { .. } => self.random_common_cause(rng, ctx),
            TargetPerturbation::Subset { fraction, .. } => {
                let n = data.row_count();
                if n < 2 {
                    return Err(unsupported("a subset needs at least two rows", "more data"));
                }
                let mut rows: Vec<usize> = (0..n).collect();
                shuffle(rng, &mut rows);
                #[allow(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "the subset fraction lies in (0, 1), and the saturating cast is clamped into [1, n]"
                )]
                let kept = ((fraction * n as f64).round() as usize).clamp(1, n);
                let mut keep = vec![false; n];
                for &row in &rows[..kept] {
                    keep[row] = true;
                }
                let masked = data.with_analysis_mask(mask_from(&keep)?).map_err(data_failed)?;
                self.reestimate(&Modal::Tabular(masked), ctx)
            }
            TargetPerturbation::Bootstrap { .. } => {
                let strata = self.change_populations(data.row_count());
                self.reestimate(
                    &Modal::Tabular(Self::bootstrap_within(data, rng, strata.as_deref())?),
                    ctx,
                )
            }
            TargetPerturbation::NegativeControlExposure { replicate } => {
                let control = self.control_exposure()?;
                let swapped =
                    replace(self.treatment, Arc::from(Self::column_of(storage, control)?))?;
                let data = if replicate == 0 { swapped } else { Self::bootstrap(&swapped, rng)? };
                self.reestimate(&Modal::Tabular(data), ctx)
            }
            TargetPerturbation::NegativeControlOutcome { replicate } => {
                let control = self.control_outcome()?;
                let swapped = replace(self.outcome, Arc::from(Self::column_of(storage, control)?))?;
                let data = if replicate == 0 { swapped } else { Self::bootstrap(&swapped, rng)? };
                self.reestimate(&Modal::Tabular(data), ctx)
            }
            TargetPerturbation::PreTrend { .. } | TargetPerturbation::Anticipation { .. } => {
                Err(unsupported(
                    "independent rows have no time order, so a lead or a pre-period does not exist",
                    "prepare the study on series or panel data",
                ))
            }
        }
    }

    /// Row bootstrap of a table whose columns are all complete float64 columns.
    fn bootstrap(data: &TabularData, rng: &mut CausalRng) -> Result<TabularData, TargetRefitError> {
        Self::bootstrap_within(data, rng, None)
    }

    /// The row positions of the baseline and the comparison population of a change attribution,
    /// when the study is one whose populations are row selections; `None` otherwise.
    fn change_populations(&self, rows: usize) -> Option<Vec<Vec<usize>>> {
        use antecedent_core::PopulationSelector;
        let CausalQuery::ChangeAttribution(query) = &self.study().query else {
            return None;
        };
        let positions = |selector: &PopulationSelector| -> Option<Vec<usize>> {
            match selector {
                PopulationSelector::All => Some((0..rows).collect()),
                PopulationSelector::Rows(rows_) => {
                    Some(rows_.iter().copied().filter(|&row| row < rows).collect())
                }
                PopulationSelector::TimeRange { start, end } => {
                    Some((*start..(*end).min(rows)).collect())
                }
                _ => None,
            }
        };
        Some(vec![positions(&query.baseline)?, positions(&query.comparison)?])
    }

    /// A row bootstrap that draws each position's source row from the stratum the position lies
    /// in (a population of a change attribution), so a resample never moves a row from one
    /// population into the other; positions outside every stratum keep their own row.
    fn bootstrap_within(
        data: &TabularData,
        rng: &mut CausalRng,
        strata: Option<&[Vec<usize>]>,
    ) -> Result<TabularData, TargetRefitError> {
        let storage = data.storage();
        if storage.analysis_mask().is_some() || storage.weights().is_some() {
            return Err(unsupported(
                "a row bootstrap of a masked or weighted table would change its population",
                "refute the unmasked, unweighted table, or use subset stability",
            ));
        }
        let n = data.row_count();
        let rows: Vec<usize> = match strata {
            None => (0..n).map(|_| unbiased_index(rng, n)).collect(),
            Some(strata) => {
                let mut rows: Vec<usize> = (0..n).collect();
                for stratum in strata.iter().filter(|stratum| !stratum.is_empty()) {
                    for &position in stratum {
                        rows[position] = stratum[unbiased_index(rng, stratum.len())];
                    }
                }
                rows
            }
        };
        let mut replacements = Vec::with_capacity(storage.columns().len());
        for column in storage.columns() {
            let OwnedColumn::Float64(column) = column else {
                return Err(unsupported(
                    "the row bootstrap gathers float64 columns only",
                    "encode every column as float64, or use subset stability",
                ));
            };
            if !column.validity.is_all_valid() {
                return Err(unsupported(
                    "the row bootstrap needs complete columns",
                    "resolve missing values first (a missing-data policy), or use subset \
                     stability",
                ));
            }
            let gathered: Arc<[f64]> = rows.iter().map(|&i| column.values[i]).collect();
            replacements.push((column.id, gathered));
        }
        data.with_replaced_floats(&replacements).map_err(data_failed)
    }
}
