//! Query contract for two-arm randomized experiments.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::QueryError;
use crate::ids::VariableId;
use std::sync::Arc;

/// Supported assignment mechanisms for retained two-arm randomized ITT.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RandomizationDesign {
    /// Independent Bernoulli assignment with row-level inclusion probabilities.
    Bernoulli,
    /// Complete randomization of exactly `treated_units` of `n` units.
    Complete {
        /// Number of units assigned to treatment.
        treated_units: usize,
    },
    /// Complete randomization of assignment clusters, with outcome rows nested in them.
    Cluster {
        /// Number of clusters selected for treatment.
        treated_clusters: usize,
    },
    /// Complete randomization independently within named blocks.
    Stratified {
        /// Block membership in row order.
        blocks: Arc<[Arc<str>]>,
        /// Declared treated count for each row's block.
        treated_per_row: Arc<[usize]>,
    },
}

/// Intention-to-treat contrast for a two-arm randomized trial.
///
/// Assignment and unit identity are part of the query contract so a prepared
/// result retains the experiment that generated its outcomes. This query does
/// not require a causal graph: randomization supplies the identification.
#[derive(Clone, Debug, PartialEq)]
pub struct RandomizedEffectQuery {
    /// Assignment mechanism used by the trial.
    pub design: RandomizationDesign,
    /// Outcome variable.
    pub outcome: VariableId,
    /// Realized assignment, one value per outcome row.
    pub realized_assignment: Arc<[bool]>,
    /// Known marginal assignment probability, one value per row.
    pub assignment_probabilities: Arc<[f64]>,
    /// Assignment unit labels in row order.
    pub assignment_units: Arc<[Arc<str>]>,
    /// Outcome unit labels in row order.
    pub outcome_units: Arc<[Arc<str>]>,
    /// Control and treatment arm labels.
    pub treatment_arms: (Arc<str>, Arc<str>),
}

impl RandomizedEffectQuery {
    /// Construct a Bernoulli two-arm ITT query.
    #[must_use]
    pub fn bernoulli_itt(
        outcome: VariableId,
        realized_assignment: impl Into<Arc<[bool]>>,
        assignment_probabilities: impl Into<Arc<[f64]>>,
        assignment_units: impl Into<Arc<[Arc<str>]>>,
        outcome_units: impl Into<Arc<[Arc<str>]>>,
        treatment_arms: (impl Into<Arc<str>>, impl Into<Arc<str>>),
    ) -> Self {
        Self::with_design(
            RandomizationDesign::Bernoulli,
            outcome,
            realized_assignment,
            assignment_probabilities.into(),
            assignment_units.into(),
            outcome_units.into(),
            treatment_arms,
        )
    }

    /// Construct an ITT query for a declared assignment mechanism.
    #[must_use]
    pub fn with_design(
        design: RandomizationDesign,
        outcome: VariableId,
        realized_assignment: impl Into<Arc<[bool]>>,
        assignment_probabilities: impl Into<Arc<[f64]>>,
        assignment_units: impl Into<Arc<[Arc<str>]>>,
        outcome_units: impl Into<Arc<[Arc<str>]>>,
        treatment_arms: (impl Into<Arc<str>>, impl Into<Arc<str>>),
    ) -> Self {
        Self {
            design,
            outcome,
            realized_assignment: realized_assignment.into(),
            assignment_probabilities: assignment_probabilities.into(),
            assignment_units: assignment_units.into(),
            outcome_units: outcome_units.into(),
            treatment_arms: (treatment_arms.0.into(), treatment_arms.1.into()),
        }
    }

    /// Validate row alignment, allocations, and assignment probabilities.
    pub fn validate(&self) -> Result<(), QueryError> {
        let n = self.realized_assignment.len();
        if n < 2
            || self.assignment_probabilities.len() != n
            || self.assignment_units.len() != n
            || self.outcome_units.len() != n
        {
            return Err(QueryError::InvalidRandomizedEffect(
                "assignment, probabilities, and unit identities must have at least two aligned rows".into(),
            ));
        }
        if self.assignment_probabilities.iter().any(|p| !p.is_finite() || *p <= 0.0 || *p >= 1.0) {
            return Err(QueryError::InvalidRandomizedEffect(
                "assignment inclusion probabilities must be finite and strictly between zero and one".into(),
            ));
        }
        if self.assignment_units.iter().any(|unit| unit.trim().is_empty())
            || self.outcome_units.iter().any(|unit| unit.trim().is_empty())
        {
            return Err(QueryError::InvalidRandomizedEffect(
                "assignment and outcome unit labels must be non-empty".into(),
            ));
        }
        if self.outcome_units.iter().collect::<std::collections::HashSet<_>>().len() != n {
            return Err(QueryError::InvalidRandomizedEffect(
                "retained randomized ITT requires one outcome unit per row".into(),
            ));
        }
        if !matches!(self.design, RandomizationDesign::Cluster { .. })
            && self.assignment_units.iter().collect::<std::collections::HashSet<_>>().len() != n
        {
            return Err(QueryError::InvalidRandomizedEffect(
                "individual randomization requires one assignment unit per row".into(),
            ));
        }
        if self.treatment_arms.0.trim().is_empty()
            || self.treatment_arms.1.trim().is_empty()
            || self.treatment_arms.0 == self.treatment_arms.1
        {
            return Err(QueryError::InvalidRandomizedEffect(
                "treatment arm labels must be distinct and non-empty".into(),
            ));
        }
        if self.realized_assignment.iter().all(|x| *x)
            || self.realized_assignment.iter().all(|x| !*x)
        {
            return Err(QueryError::InvalidRandomizedEffect(
                "both assigned arms must be observed".into(),
            ));
        }
        match &self.design {
            RandomizationDesign::Bernoulli => {}
            RandomizationDesign::Complete { treated_units } => {
                if *treated_units < 2 || n.saturating_sub(*treated_units) < 2 {
                    return Err(QueryError::InvalidRandomizedEffect(
                        "complete-randomization variance requires at least two units in each arm"
                            .into(),
                    ));
                }
                if self.realized_assignment.iter().filter(|assigned| **assigned).count()
                    != *treated_units
                {
                    return Err(QueryError::InvalidRandomizedEffect(
                        "complete assignment must match the declared treated count".into(),
                    ));
                }
                let expected = *treated_units as f64 / n as f64;
                if self.assignment_probabilities.iter().any(|p| (*p - expected).abs() > 1e-12) {
                    return Err(QueryError::InvalidRandomizedEffect(
                        "complete assignment probabilities must equal treated_units / n".into(),
                    ));
                }
            }
            RandomizationDesign::Cluster { treated_clusters } => {
                let mut clusters = std::collections::BTreeMap::<&str, (bool, f64)>::new();
                for i in 0..n {
                    let assignment = self.realized_assignment[i];
                    let probability = self.assignment_probabilities[i];
                    if let Some((prior_assignment, prior_probability)) =
                        clusters.insert(&self.assignment_units[i], (assignment, probability))
                    {
                        if prior_assignment != assignment
                            || (prior_probability - probability).abs() > 1e-12
                        {
                            return Err(QueryError::InvalidRandomizedEffect(
                                "assignment and probability must be constant within each randomized cluster".into(),
                            ));
                        }
                    }
                }
                let total = clusters.len();
                if *treated_clusters < 2 || total.saturating_sub(*treated_clusters) < 2 {
                    return Err(QueryError::InvalidRandomizedEffect(
                        "cluster-randomization variance requires at least two clusters in each arm"
                            .into(),
                    ));
                }
                let observed = clusters.values().filter(|(treated, _)| *treated).count();
                let expected_probability = *treated_clusters as f64 / total as f64;
                if observed != *treated_clusters
                    || clusters.values().any(|(_, p)| (*p - expected_probability).abs() > 1e-12)
                {
                    return Err(QueryError::InvalidRandomizedEffect(
                        "cluster assignment must match declared treated count and inclusion probability".into(),
                    ));
                }
            }
            RandomizationDesign::Stratified { blocks, treated_per_row } => {
                if blocks.len() != n
                    || treated_per_row.len() != n
                    || blocks.iter().any(|b| b.trim().is_empty())
                {
                    return Err(QueryError::InvalidRandomizedEffect(
                        "stratified block labels and allocations must align with rows".into(),
                    ));
                }
                let mut by_block: std::collections::BTreeMap<
                    &str,
                    (usize, usize, Option<usize>, f64),
                > = std::collections::BTreeMap::new();
                for i in 0..n {
                    let entry = by_block.entry(&blocks[i]).or_insert((
                        0,
                        0,
                        Some(treated_per_row[i]),
                        self.assignment_probabilities[i],
                    ));
                    if entry.2 != Some(treated_per_row[i])
                        || (entry.3 - self.assignment_probabilities[i]).abs() > 1e-12
                    {
                        return Err(QueryError::InvalidRandomizedEffect(
                            "each block must have a consistent allocation and inclusion probability".into(),
                        ));
                    }
                    entry.0 += 1;
                    entry.1 += usize::from(self.realized_assignment[i]);
                }
                for (_, (total, treated_observed, treated_declared, probability)) in by_block {
                    let treated_declared = treated_declared.unwrap_or_default();
                    if treated_declared < 2 || total.saturating_sub(treated_declared) < 2 {
                        return Err(QueryError::InvalidRandomizedEffect(
                            "stratified variance requires at least two units per arm in every block".into(),
                        ));
                    }
                    if treated_observed != treated_declared
                        || (probability - treated_declared as f64 / total as f64).abs() > 1e-12
                    {
                        return Err(QueryError::InvalidRandomizedEffect(
                            "stratified assignment must match declared block counts and inclusion probabilities".into(),
                        ));
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn query() -> RandomizedEffectQuery {
        RandomizedEffectQuery::bernoulli_itt(
            VariableId::from_raw(0),
            [true, false, true, false],
            [0.5; 4],
            ["a", "b", "c", "d"].map(Arc::<str>::from),
            ["a", "b", "c", "d"].map(Arc::<str>::from),
            ("control", "treated"),
        )
    }

    #[test]
    fn validates_bernoulli_trial_metadata() {
        assert!(query().validate().is_ok());
        let mut invalid = query();
        invalid.assignment_probabilities = Arc::from([0.0, 0.5, 0.5, 0.5]);
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn rejects_repeated_units_and_one_observed_arm() {
        let mut repeated = query();
        repeated.assignment_units = Arc::from(["a", "a", "c", "d"].map(Arc::<str>::from));
        assert!(repeated.validate().is_err());
        let mut one_arm = query();
        one_arm.realized_assignment = Arc::from([true; 4]);
        assert!(one_arm.validate().is_err());
    }

    #[test]
    fn validates_complete_and_blocked_assignment_metadata() {
        let units: Vec<Arc<str>> = (0..8).map(|i| Arc::from(format!("u{i}"))).collect();
        let complete = RandomizedEffectQuery::with_design(
            RandomizationDesign::Complete { treated_units: 4 },
            VariableId::from_raw(0),
            [true, false, true, false, true, false, true, false],
            [0.5; 8],
            units.clone(),
            units.clone(),
            ("control", "treated"),
        );
        assert!(complete.validate().is_ok());

        let blocks: Vec<Arc<str>> =
            ["north", "north", "north", "north", "south", "south", "south", "south"]
                .map(Arc::from)
                .into();
        let stratified = RandomizedEffectQuery::with_design(
            RandomizationDesign::Stratified {
                blocks: blocks.into(),
                treated_per_row: Arc::from([2; 8]),
            },
            VariableId::from_raw(0),
            [true, false, true, false, true, false, true, false],
            [0.5; 8],
            units.clone(),
            units,
            ("control", "treated"),
        );
        assert!(stratified.validate().is_ok());

        let mut degenerate = stratified;
        degenerate.assignment_probabilities =
            Arc::from([0.25, 0.25, 0.25, 0.25, 0.5, 0.5, 0.5, 0.5]);
        assert!(degenerate.validate().is_err());
    }
}
