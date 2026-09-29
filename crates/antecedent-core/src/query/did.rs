//! Two-period and selected staggered group-time difference-in-differences queries.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::QueryError;
use crate::ids::VariableId;
use std::sync::Arc;

/// Sampling design for a difference-in-differences comparison.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum DidSamplingDesign {
    /// The same subjects contribute one observation in each period.
    #[default]
    BalancedPanel,
    /// Each subject contributes one observation in exactly one period.
    RepeatedCrossSection,
    /// A balanced staggered panel, compared with never-treated subjects at a selected time.
    StaggeredGroupTime,
    /// All cohort-specific event-time contrasts relative to adoption period minus one.
    StaggeredEventStudy,
    /// One row per subject with supplied propensity and untreated-change predictions.
    AugmentedPanel,
}

/// Nuisance columns for the augmented two-period panel ATT.
#[derive(Clone, Debug, PartialEq)]
pub struct AugmentedPanelNuisance {
    pub outcome_pre: VariableId,
    pub propensity: VariableId,
    pub untreated_change_prediction: VariableId,
    /// Caller declaration only; the engine does not certify model fitting.
    pub predictions_cross_fitted: bool,
}

/// Difference in outcome changes for a two-period or selected group-time design.
///
/// This estimates a point and cluster-robust standard error. Its causal
/// interpretation requires consistency, no anticipation, and parallel trends.
#[derive(Clone, Debug, PartialEq)]
pub struct PanelDidQuery {
    /// Whether subjects recur across periods or are sampled once.
    pub design: DidSamplingDesign,
    /// Outcome column.
    pub outcome: VariableId,
    /// Stable treatment group, aligned to outcome rows.
    pub treated: Arc<[bool]>,
    /// Pre/post period indicator, aligned to outcome rows.
    pub post: Arc<[bool]>,
    /// Subject identifier for each observation.
    pub subjects: Arc<[Arc<str>]>,
    /// Inference cluster for each observation, constant within subject.
    pub clusters: Arc<[Arc<str>]>,
    /// Calendar period of each row for a staggered comparison.
    pub periods: Arc<[i64]>,
    /// First treated period; zero denotes a never-treated subject.
    pub cohorts: Arc<[i64]>,
    /// Selected adoption cohort and post-adoption comparison period.
    pub target: Option<(i64, i64)>,
    /// Supplied nuisance columns for the one-row-per-subject augmented design.
    pub augmented: Option<AugmentedPanelNuisance>,
}

impl PanelDidQuery {
    /// Construct the row-aligned design contract.
    #[must_use]
    pub fn new(
        outcome: VariableId,
        treated: impl Into<Arc<[bool]>>,
        post: impl Into<Arc<[bool]>>,
        subjects: impl Into<Arc<[Arc<str>]>>,
        clusters: impl Into<Arc<[Arc<str>]>>,
    ) -> Self {
        Self {
            design: DidSamplingDesign::BalancedPanel,
            outcome,
            treated: treated.into(),
            post: post.into(),
            subjects: subjects.into(),
            clusters: clusters.into(),
            periods: Arc::from([]),
            cohorts: Arc::from([]),
            target: None,
            augmented: None,
        }
    }

    /// Select one group-time ATT from a staggered adoption panel.
    #[must_use]
    pub fn staggered_group_time(
        outcome: VariableId,
        subjects: impl Into<Arc<[Arc<str>]>>,
        clusters: impl Into<Arc<[Arc<str>]>>,
        periods: impl Into<Arc<[i64]>>,
        cohorts: impl Into<Arc<[i64]>>,
        target_cohort: i64,
        target_period: i64,
    ) -> Self {
        let subjects = subjects.into();
        let clusters = clusters.into();
        let periods = periods.into();
        let cohorts = cohorts.into();
        let treated = cohorts.iter().map(|value| *value == target_cohort).collect::<Vec<_>>();
        let post = periods.iter().map(|value| *value == target_period).collect::<Vec<_>>();
        Self {
            design: DidSamplingDesign::StaggeredGroupTime,
            outcome,
            treated: treated.into(),
            post: post.into(),
            subjects,
            clusters,
            periods,
            cohorts,
            target: Some((target_cohort, target_period)),
            augmented: None,
        }
    }

    /// Compare every adoption cohort with never-treated controls over the panel.
    #[must_use]
    pub fn staggered_event_study(
        outcome: VariableId,
        subjects: impl Into<Arc<[Arc<str>]>>,
        clusters: impl Into<Arc<[Arc<str>]>>,
        periods: impl Into<Arc<[i64]>>,
        cohorts: impl Into<Arc<[i64]>>,
    ) -> Self {
        let subjects = subjects.into();
        let cohorts = cohorts.into();
        let periods = periods.into();
        let treated = cohorts.iter().map(|value| *value > 0).collect::<Vec<_>>();
        let post = periods
            .iter()
            .zip(cohorts.iter())
            .map(|(period, cohort)| *cohort > 0 && *period >= *cohort)
            .collect::<Vec<_>>();
        Self {
            design: DidSamplingDesign::StaggeredEventStudy,
            outcome,
            treated: treated.into(),
            post: post.into(),
            subjects,
            clusters: clusters.into(),
            periods,
            cohorts,
            target: None,
            augmented: None,
        }
    }

    /// Construct an augmented panel ATT with supplied, row-aligned nuisance predictions.
    // The augmented panel binds four frozen columns plus assignment, subject,
    // cluster, and cross-fit metadata; each is a distinct caller input.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn augmented_panel(
        outcome_post: VariableId,
        outcome_pre: VariableId,
        propensity: VariableId,
        untreated_change_prediction: VariableId,
        treated: impl Into<Arc<[bool]>>,
        subjects: impl Into<Arc<[Arc<str>]>>,
        clusters: impl Into<Arc<[Arc<str>]>>,
        predictions_cross_fitted: bool,
    ) -> Self {
        let treated = treated.into();
        let post = vec![true; treated.len()];
        let mut query = Self::new(outcome_post, treated, post, subjects, clusters);
        query.design = DidSamplingDesign::AugmentedPanel;
        query.augmented = Some(AugmentedPanelNuisance {
            outcome_pre,
            propensity,
            untreated_change_prediction,
            predictions_cross_fitted,
        });
        query
    }

    /// Construct a repeated-cross-section design with one row per subject.
    #[must_use]
    pub fn repeated_cross_section(
        outcome: VariableId,
        treated: impl Into<Arc<[bool]>>,
        post: impl Into<Arc<[bool]>>,
        subjects: impl Into<Arc<[Arc<str>]>>,
        clusters: impl Into<Arc<[Arc<str>]>>,
    ) -> Self {
        let mut query = Self::new(outcome, treated, post, subjects, clusters);
        query.design = DidSamplingDesign::RepeatedCrossSection;
        query
    }

    /// Validate metadata dimensions and labels.
    ///
    /// # Errors
    ///
    /// Returns [`QueryError::InvalidRandomizedEffect`] when fewer than four rows
    /// are supplied or the metadata vectors are misaligned, subject or cluster
    /// labels are blank, an augmented panel is missing or has non-distinct
    /// columns or duplicate subjects, nuisance columns are attached without an
    /// augmented panel, or a staggered design has an invalid cohort/period
    /// selection or misaligned cohort/period vectors.
    pub fn validate(&self) -> Result<(), QueryError> {
        let n = self.treated.len();
        if n < 4 || self.post.len() != n || self.subjects.len() != n || self.clusters.len() != n {
            return Err(QueryError::InvalidRandomizedEffect(
                "DiD requires at least four aligned rows and metadata vectors".into(),
            ));
        }
        if self.subjects.iter().any(|x| x.trim().is_empty())
            || self.clusters.iter().any(|x| x.trim().is_empty())
        {
            return Err(QueryError::InvalidRandomizedEffect(
                "subject and cluster labels must be non-empty".into(),
            ));
        }
        if self.design == DidSamplingDesign::AugmentedPanel {
            let Some(nuisance) = &self.augmented else {
                return Err(QueryError::InvalidRandomizedEffect(
                    "augmented panel DiD requires nuisance columns".into(),
                ));
            };
            let variables = [
                self.outcome,
                nuisance.outcome_pre,
                nuisance.propensity,
                nuisance.untreated_change_prediction,
            ];
            if variables.iter().collect::<std::collections::BTreeSet<_>>().len() != variables.len()
                || self.subjects.iter().collect::<std::collections::BTreeSet<_>>().len() != n
                || self.post.iter().any(|post| !post)
                || !self.periods.is_empty()
                || !self.cohorts.is_empty()
                || self.target.is_some()
            {
                return Err(QueryError::InvalidRandomizedEffect(
                    "augmented panel DiD requires distinct columns and one unique subject per row"
                        .into(),
                ));
            }
        } else if self.augmented.is_some() {
            return Err(QueryError::InvalidRandomizedEffect(
                "nuisance columns require augmented panel DiD".into(),
            ));
        }
        if matches!(
            self.design,
            DidSamplingDesign::StaggeredGroupTime | DidSamplingDesign::StaggeredEventStudy
        ) {
            let (cohort, period) = if self.design == DidSamplingDesign::StaggeredGroupTime {
                let Some((cohort, period)) = self.target else {
                    return Err(QueryError::InvalidRandomizedEffect(
                        "staggered DiD requires a selected cohort and period".into(),
                    ));
                };
                (cohort, period)
            } else {
                (2, 2)
            };
            if (self.design == DidSamplingDesign::StaggeredGroupTime
                && (cohort <= 1 || period < cohort))
                || self.periods.len() != n
                || self.cohorts.len() != n
                || self.periods.iter().any(|p| *p <= 0)
                || self.cohorts.iter().any(|g| *g < 0)
            {
                return Err(QueryError::InvalidRandomizedEffect("staggered DiD requires aligned positive periods, nonnegative cohorts, and a post-adoption target with an observed pre-period".into()));
            }
        }
        Ok(())
    }
}
