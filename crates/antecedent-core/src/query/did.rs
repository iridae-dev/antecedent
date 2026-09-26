//! Two-period balanced-panel difference-in-differences query.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::QueryError;
use crate::ids::VariableId;
use std::sync::Arc;

/// Difference in subject-level changes for a balanced two-period panel.
///
/// This estimates a point and cluster-robust standard error. Its causal
/// interpretation requires consistency, no anticipation, and parallel trends.
#[derive(Clone, Debug, PartialEq)]
pub struct PanelDidQuery {
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
            outcome,
            treated: treated.into(),
            post: post.into(),
            subjects: subjects.into(),
            clusters: clusters.into(),
        }
    }

    /// Validate metadata dimensions and labels.
    pub fn validate(&self) -> Result<(), QueryError> {
        let n = self.treated.len();
        if n < 4 || self.post.len() != n || self.subjects.len() != n || self.clusters.len() != n {
            return Err(QueryError::InvalidRandomizedEffect(
                "panel DiD requires at least four aligned rows and metadata vectors".into(),
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
