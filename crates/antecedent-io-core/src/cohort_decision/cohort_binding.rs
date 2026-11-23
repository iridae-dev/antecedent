//! Cohort rows, weights and their canonical identity.
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;

use antecedent_core::CanonicalWriter;
use antecedent_kernels::numeric::CompensatedSum;
use serde::{Deserialize, Serialize};

use super::{COHORT_TAG, CohortIntakeLimits, CohortRefusal, WEIGHT_SUM_TOLERANCE, invalid};

/// One named feature column of a cohort.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CohortColumn {
    /// Finite numeric values, one per row.
    Numeric(Vec<f64>),
    /// Categorical levels, one per row.
    Categorical(Vec<String>),
}

impl CohortColumn {
    fn len(&self) -> usize {
        match self {
            Self::Numeric(values) => values.len(),
            Self::Categorical(values) => values.len(),
        }
    }
}

/// A declared cohort: strict named feature rows with identities and normalized weights.
///
/// The cohort digest binds every field, so a changed row, weight, identity, feature name
/// or population is a different cohort.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CohortBinding {
    pub(super) snapshot_id: String,
    pub(super) population_id: String,
    pub(super) row_ids: Vec<String>,
    pub(super) features: Vec<String>,
    pub(super) columns: Vec<CohortColumn>,
    pub(super) weights: Vec<f64>,
}

/// Sum with [`CompensatedSum`], so the result is nearly independent of row order.
pub(super) fn compensated_sum(terms: impl Iterator<Item = f64>) -> f64 {
    CompensatedSum::of(terms)
}

impl CohortBinding {
    /// Bind and validate a cohort.
    ///
    /// # Errors
    /// A blank snapshot or population, an empty or oversized cohort, duplicate or blank
    /// row identities or feature names, ragged or nonfinite columns, and weights that are
    /// negative, nonfinite or do not sum to one within [`WEIGHT_SUM_TOLERANCE`].
    pub fn new(
        snapshot_id: &str,
        population_id: &str,
        row_ids: Vec<String>,
        features: Vec<String>,
        columns: Vec<CohortColumn>,
        weights: Vec<f64>,
    ) -> Result<Self, CohortRefusal> {
        Self::new_within(
            snapshot_id,
            population_id,
            row_ids,
            features,
            columns,
            weights,
            &CohortIntakeLimits::default(),
        )
    }

    /// [`Self::new`] under the caller's intake bounds instead of [`super::MAX_COHORT_ROWS`].
    ///
    /// # Errors
    /// As [`Self::new`]; more rows than `limits.max_rows` is `cohort_too_large`.
    pub fn new_within(
        snapshot_id: &str,
        population_id: &str,
        row_ids: Vec<String>,
        features: Vec<String>,
        columns: Vec<CohortColumn>,
        weights: Vec<f64>,
        limits: &CohortIntakeLimits,
    ) -> Result<Self, CohortRefusal> {
        let cohort = Self {
            snapshot_id: snapshot_id.to_owned(),
            population_id: population_id.to_owned(),
            row_ids,
            features,
            columns,
            weights,
        };
        cohort.validate_within(limits)?;
        Ok(cohort)
    }

    /// Weights `raw / sum(raw)`: an explicit normalization the caller asks for.
    ///
    /// # Errors
    /// An empty vector, a nonfinite or negative weight, or a zero total.
    pub fn normalize_weights(raw: &[f64]) -> Result<Vec<f64>, CohortRefusal> {
        let remedy = "supply finite nonnegative weights with a positive total";
        if raw.is_empty() || raw.iter().any(|w| !w.is_finite() || *w < 0.0) {
            return Err(invalid(
                antecedent_core::reason_details::cohort_decision::WEIGHTS_INVALID,
                "weights",
                remedy,
            ));
        }
        let total = compensated_sum(raw.iter().copied());
        if !(total.is_finite() && total > 0.0) {
            return Err(invalid(
                antecedent_core::reason_details::cohort_decision::WEIGHTS_INVALID,
                "weights",
                remedy,
            ));
        }
        Ok(raw.iter().map(|w| w / total).collect())
    }

    /// Equal weights over `rows` rows.
    ///
    /// # Errors
    /// Zero rows.
    pub fn uniform_weights(rows: usize) -> Result<Vec<f64>, CohortRefusal> {
        if rows == 0 {
            return Err(invalid(
                antecedent_core::reason_details::cohort_decision::COHORT_EMPTY,
                "rows",
                "bind at least one cohort row",
            ));
        }
        #[allow(
            clippy::cast_precision_loss,
            reason = "the row count is bounded by MAX_COHORT_ROWS"
        )]
        let weight = 1.0 / rows as f64;
        Ok(vec![weight; rows])
    }

    /// Re-run the construction checks, for a cohort decoded from bytes.
    ///
    /// # Errors
    /// As [`Self::new`].
    pub fn validate(&self) -> Result<(), CohortRefusal> {
        self.validate_within(&CohortIntakeLimits::default())
    }

    /// [`Self::validate`] under the caller's intake bounds instead of [`super::MAX_COHORT_ROWS`].
    ///
    /// # Errors
    /// As [`Self::validate`].
    pub fn validate_within(&self, limits: &CohortIntakeLimits) -> Result<(), CohortRefusal> {
        let remedy = "declare a cohort with named finite feature rows and normalized weights";
        if self.snapshot_id.trim().is_empty() || self.population_id.trim().is_empty() {
            return Err(invalid(
                antecedent_core::reason_details::cohort_decision::COHORT_INVALID,
                "snapshot_or_population",
                remedy,
            ));
        }
        let rows = self.weights.len();
        if rows == 0 {
            return Err(invalid(
                antecedent_core::reason_details::cohort_decision::COHORT_EMPTY,
                "rows",
                remedy,
            ));
        }
        if rows > limits.max_rows {
            return Err(invalid(
                antecedent_core::reason_details::cohort_decision::COHORT_TOO_LARGE,
                rows.to_string(),
                remedy,
            ));
        }
        if self.features.is_empty()
            || self.features.iter().any(|name| name.trim().is_empty())
            || self.features.iter().collect::<BTreeSet<_>>().len() != self.features.len()
            || self.columns.len() != self.features.len()
        {
            return Err(invalid(
                antecedent_core::reason_details::cohort_decision::COHORT_INVALID,
                "features",
                remedy,
            ));
        }
        if self.row_ids.len() != rows
            || self.row_ids.iter().any(|id| id.trim().is_empty())
            || self.row_ids.iter().collect::<BTreeSet<_>>().len() != rows
        {
            return Err(invalid(
                antecedent_core::reason_details::cohort_decision::COHORT_INVALID,
                "row_ids",
                remedy,
            ));
        }
        for (name, column) in self.features.iter().zip(&self.columns) {
            let finite = match column {
                CohortColumn::Numeric(values) => values.iter().all(|v| v.is_finite()),
                CohortColumn::Categorical(values) => values.iter().all(|v| !v.trim().is_empty()),
            };
            if column.len() != rows || !finite {
                return Err(invalid(
                    antecedent_core::reason_details::cohort_decision::COHORT_INVALID,
                    name.clone(),
                    remedy,
                ));
            }
        }
        if self.weights.iter().any(|w| !w.is_finite() || *w < 0.0) {
            return Err(invalid(
                antecedent_core::reason_details::cohort_decision::WEIGHTS_INVALID,
                "weights",
                remedy,
            ));
        }
        let total = compensated_sum(self.weights.iter().copied());
        if (total - 1.0).abs() > WEIGHT_SUM_TOLERANCE {
            return Err(invalid(
                antecedent_core::reason_details::cohort_decision::WEIGHTS_NOT_NORMALIZED,
                format!("sum={total}"),
                "normalize the weights to sum to one (CohortBinding::normalize_weights)",
            ));
        }
        Ok(())
    }

    /// Caller-declared name of the cohort snapshot.
    #[must_use]
    pub fn snapshot_id(&self) -> &str {
        &self.snapshot_id
    }

    /// Population the rows are declared to come from.
    #[must_use]
    pub fn population_id(&self) -> &str {
        &self.population_id
    }

    /// Row identities, in row order.
    #[must_use]
    pub fn row_ids(&self) -> &[String] {
        &self.row_ids
    }

    /// Feature names, in the order supplied.
    #[must_use]
    pub fn features(&self) -> &[String] {
        &self.features
    }

    /// Feature columns, aligned with [`Self::features`].
    #[must_use]
    pub fn columns(&self) -> &[CohortColumn] {
        &self.columns
    }

    /// Normalized weights, in row order.
    #[must_use]
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Number of rows.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.weights.len()
    }

    /// Canonical digest of every field, as lowercase hex. This is the cohort snapshot
    /// identity a decision contract names in its input coordinate.
    #[must_use]
    pub fn digest(&self) -> String {
        let mut canon = CanonicalWriter::new(COHORT_TAG);
        canon.text(&self.snapshot_id).text(&self.population_id);
        canon.texts(&self.row_ids).texts(&self.features);
        canon.word(self.columns.len() as u64);
        for column in &self.columns {
            match column {
                CohortColumn::Numeric(values) => {
                    canon.text("numeric").reals(values);
                }
                CohortColumn::Categorical(values) => {
                    canon.text("categorical").texts(values);
                }
            }
        }
        canon.reals(&self.weights);
        canon.finish_hex()
    }

    pub(super) fn numeric(&self, name: &str) -> Option<&[f64]> {
        let index = self.features.iter().position(|f| f == name)?;
        match &self.columns[index] {
            CohortColumn::Numeric(values) => Some(values.as_slice()),
            CohortColumn::Categorical(_) => None,
        }
    }

    pub(super) fn level(&self, name: &str, row: usize) -> Option<&str> {
        let index = self.features.iter().position(|f| f == name)?;
        match &self.columns[index] {
            CohortColumn::Categorical(values) => values.get(row).map(String::as_str),
            CohortColumn::Numeric(_) => None,
        }
    }
}
