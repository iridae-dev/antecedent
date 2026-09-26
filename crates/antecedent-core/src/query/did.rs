//! Two-period balanced-panel difference-in-differences query.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::QueryError;
use crate::ids::VariableId;
use std::sync::Arc;

/// Sampling design for a two-period difference-in-differences comparison.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum DidSamplingDesign {
    /// The same subjects contribute one observation in each period.
    #[default]
    BalancedPanel,
    /// Each subject contributes one observation in exactly one period.
    RepeatedCrossSection,
}

/// Difference in outcome changes for a two-period design.
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
        Ok(())
    }
}
