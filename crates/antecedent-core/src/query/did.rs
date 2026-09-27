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
        let post = periods.iter().zip(cohorts.iter())
            .map(|(period, cohort)| *cohort > 0 && *period >= *cohort)
            .collect::<Vec<_>>();
        Self {
            design: DidSamplingDesign::StaggeredEventStudy,
            outcome, treated: treated.into(), post: post.into(), subjects,
            clusters: clusters.into(), periods, cohorts, target: None,
        }
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
        if matches!(self.design, DidSamplingDesign::StaggeredGroupTime | DidSamplingDesign::StaggeredEventStudy) {
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
            if (self.design == DidSamplingDesign::StaggeredGroupTime && (cohort <= 1 || period < cohort))
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
