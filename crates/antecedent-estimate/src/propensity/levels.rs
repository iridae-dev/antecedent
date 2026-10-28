//! Declared treatment levels of a propensity problem.
//!
//! A propensity estimator contrasts two declared levels of the treatment column. Preparation
//! recodes the active level to `1.0` and the control level to `0.0`, so every downstream
//! step sees a 0/1 arm indicator whatever the column's own coding (`0/1`, `1/2`, reversed,
//! or arbitrary declared values). A row at neither declared level is outside the contrast:
//! it is either refused (the default) or excluded under an explicit policy that restricts the
//! target population to the rows at the two declared levels, and the exclusion is counted on
//! the result. Nothing is dropped silently.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::error::EstimationError;

/// What preparation does with a row whose treatment is at neither declared level.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum OtherLevels {
    /// Refuse the problem. The default: dropping those rows changes the estimand.
    #[default]
    Refuse,
    /// Exclude those rows and estimate on the rows at the two declared levels, the
    /// pairwise-subpopulation target. The count is reported on the result.
    ExcludeUndeclared,
}

/// The treatment levels a propensity problem was prepared for, and what that excluded.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DeclaredTreatmentLevels {
    /// Level coded `1.0` in the prepared treatment.
    pub active: f64,
    /// Level coded `0.0` in the prepared treatment.
    pub control: f64,
    /// Policy for rows at neither level that preparation applied.
    pub other_levels: OtherLevels,
    /// Complete-case rows at the active level.
    pub rows_at_active: u64,
    /// Complete-case rows at the control level.
    pub rows_at_control: u64,
    /// Complete-case rows at neither level that the policy excluded.
    pub excluded_rows: u64,
}

impl DeclaredTreatmentLevels {
    /// The population the contrast speaks for when rows were excluded.
    pub const RESTRICTED_POPULATION: &'static str = "rows at the two declared levels";

    /// Rows the estimate was computed on.
    #[must_use]
    pub const fn rows_used(&self) -> u64 {
        self.rows_at_active + self.rows_at_control
    }

    /// The restricted target population, present exactly when rows were excluded: the
    /// contrast then does not speak for the rows at other levels.
    #[must_use]
    pub const fn restricted_population(&self) -> Option<&'static str> {
        if self.excluded_rows > 0 { Some(Self::RESTRICTED_POPULATION) } else { None }
    }
}

/// Most distinct levels a refusal lists before it says there are more.
const LISTED_LEVELS: usize = 16;

/// The distinct treatment values of a column with their row counts, for a refusal that must
/// name what was observed: `observed treatment levels: 0 (700 rows), 1 (650 rows), 2 (650 rows)`.
///
/// Level comparison is exact, as in the classification itself. A column with more than
/// [`LISTED_LEVELS`] distinct values (a continuous treatment) lists the smallest ones and
/// says the list is incomplete.
#[allow(
    clippy::float_cmp,
    reason = "treatment levels are exact category labels, not computed values"
)]
pub(crate) fn describe_observed_levels(treatment: &[f64]) -> String {
    let mut seen: Vec<(f64, u64)> = Vec::new();
    let mut more = false;
    for &value in treatment {
        if let Some(entry) = seen.iter_mut().find(|(level, _)| *level == value) {
            entry.1 += 1;
        } else if seen.len() < LISTED_LEVELS {
            seen.push((value, 1));
        } else {
            more = true;
        }
    }
    seen.sort_by(|a, b| a.0.total_cmp(&b.0));
    let listed: Vec<String> =
        seen.iter().map(|(level, rows)| format!("{level} ({rows} rows)")).collect();
    let tail = if more { ", and further levels" } else { "" };
    format!("observed treatment levels: {}{tail}", listed.join(", "))
}

/// Classify the complete-case treatment values against the declared levels.
///
/// Returns, per input row, whether it is retained (at a declared level), and the record of
/// the classification. Level comparison is exact: declared levels are category labels.
///
/// # Errors
///
/// `treatment_level_undeclared` (detail `treatment_levels.undeclared_level`) when rows at
/// other levels exist and the policy is [`OtherLevels::Refuse`]. A declared level with no rows
/// is left to the estimators' own arm-population refusals.
#[allow(
    clippy::float_cmp,
    reason = "declared treatment levels are exact category labels, not computed values"
)]
pub(crate) fn classify_treatment_levels(
    treatment: &[f64],
    active: f64,
    control: f64,
    other_levels: OtherLevels,
) -> Result<(Vec<bool>, DeclaredTreatmentLevels), EstimationError> {
    let mut rows_at_active = 0u64;
    let mut rows_at_control = 0u64;
    let mut excluded_rows = 0u64;
    let mut first_other = None;
    let keep: Vec<bool> = treatment
        .iter()
        .map(|&t| {
            if t == active {
                rows_at_active += 1;
                true
            } else if t == control {
                rows_at_control += 1;
                true
            } else {
                excluded_rows += 1;
                first_other.get_or_insert(t);
                false
            }
        })
        .collect();
    if let (Some(value), OtherLevels::Refuse) = (first_other, other_levels) {
        return Err(crate::util::refuse(
            antecedent_core::reason_code!("treatment_level_undeclared"),
            "treatment_levels.undeclared_level",
            &format!(
                "{excluded_rows} complete-case rows have a treatment value at neither declared \
                 level (active {active}, control {control}); first found {value}; {}. Excluding \
                 them would restrict the target population to the rows at the two declared \
                 levels. Next attempt: declare that restriction with \
                 OtherLevels::ExcludeUndeclared, or use the full-population \
                 generalized-propensity estimator",
                describe_observed_levels(treatment)
            ),
        ));
    }
    Ok((
        keep,
        DeclaredTreatmentLevels {
            active,
            control,
            other_levels,
            rows_at_active,
            rows_at_control,
            excluded_rows,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classification_counts_each_level_and_the_exclusions() {
        let t = [2.0, 1.0, 3.0, 2.0, 1.0, 1.0, 4.0];
        let (keep, levels) =
            classify_treatment_levels(&t, 2.0, 1.0, OtherLevels::ExcludeUndeclared).unwrap();
        assert_eq!(keep, [true, true, false, true, true, true, false]);
        assert_eq!(
            (levels.rows_at_active, levels.rows_at_control, levels.excluded_rows),
            (2, 3, 2)
        );
        assert_eq!(levels.rows_used(), 5);
        assert_eq!(levels.restricted_population(), Some("rows at the two declared levels"));
    }

    #[test]
    fn undeclared_rows_are_refused_without_a_policy() {
        let err =
            classify_treatment_levels(&[0.0, 1.0, 2.0], 1.0, 0.0, OtherLevels::Refuse).unwrap_err();
        assert!(err.to_string().starts_with("reason=treatment_level_undeclared:"), "{err}");
    }

    #[test]
    fn a_fully_declared_column_reports_no_restriction() {
        let (keep, levels) =
            classify_treatment_levels(&[0.0, 1.0], 1.0, 0.0, OtherLevels::Refuse).unwrap();
        assert_eq!(keep, [true, true]);
        assert_eq!(levels.restricted_population(), None);
    }
}
