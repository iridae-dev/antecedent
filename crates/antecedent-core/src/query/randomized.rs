//! Query contract for two-arm randomized experiments.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::QueryError;
use crate::ids::VariableId;
use std::sync::Arc;

/// Supported assignment mechanisms for retained two-arm randomized ITT.
#[derive(Clone, Debug, PartialEq)]
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
    /// Jointly complete randomization to four fixed 2×2 factorial cells.
    Factorial2x2 {
        /// Assignment to the second factor in outcome-row order.
        second_factor_assignment: Arc<[bool]>,
        /// Declared counts for cells (A=0,B=0), (A=1,B=0), (A=0,B=1), (A=1,B=1).
        cell_counts: [usize; 4],
        /// Labels for the second factor's control and active levels.
        second_factor_arms: (Arc<str>, Arc<str>),
    },
    /// Independent assignment among three or more declared actions.
    MultiArm {
        /// Observed action indices aligned with outcome rows.
        assignment: Arc<[usize]>,
        /// Known action probabilities in declared label order for each row.
        probabilities: Arc<[Vec<f64>]>,
        /// Ordered action labels; the first is the reference arm.
        arms: Arc<[Arc<str>]>,
    },
    /// Unit-period assignment within independent switching sequences.
    Switchback {
        /// Period labels aligned with outcome rows; unique within each sequence.
        periods: Arc<[Arc<str>]>,
    },
}

/// Target of a retained randomized experiment query.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RandomizedEstimand {
    /// Effect of assignment, regardless of treatment receipt.
    IntentionToTreat,
    /// Wald complier effect under exclusion and monotonicity.
    ComplierAverageCausalEffect,
    /// Effect among recipients under one-sided noncompliance and exclusion.
    TreatmentOnTreated,
}

/// Intention-to-treat contrast for a two-arm randomized trial.
///
/// Assignment and unit identity are part of the query contract so a prepared
/// result retains the experiment that generated its outcomes. This query does
/// not require a causal graph: randomization supplies the identification.
#[derive(Clone, Debug, PartialEq)]
pub struct RandomizedEffectQuery {
    /// Distinguishes assignment ITT from a receipt-adjusted complier effect.
    pub estimand: RandomizedEstimand,
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
    /// Pre-assignment covariate and coefficient fixed before outcomes were observed.
    pub fixed_cuped: Option<(VariableId, f64)>,
    /// Pre-assignment covariates fitted jointly with assignment by ANCOVA.
    pub ancova_covariates: Arc<[VariableId]>,
    /// Observed treatment receipt in row order for CACE/LATE.
    pub received_treatment: Option<Arc<[bool]>>,
    /// Request exhaustive two-sided Fisher sharp-null randomization inference.
    pub exact_randomization_test: bool,
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
            estimand: RandomizedEstimand::IntentionToTreat,
            design,
            outcome,
            realized_assignment: realized_assignment.into(),
            assignment_probabilities: assignment_probabilities.into(),
            assignment_units: assignment_units.into(),
            outcome_units: outcome_units.into(),
            treatment_arms: (treatment_arms.0.into(), treatment_arms.1.into()),
            fixed_cuped: None,
            ancova_covariates: Arc::from([]),
            received_treatment: None,
            exact_randomization_test: false,
        }
    }

    /// Declare an externally fixed CUPED coefficient for a pre-assignment covariate.
    #[must_use]
    pub fn with_fixed_cuped(mut self, covariate: VariableId, coefficient: f64) -> Self {
        self.fixed_cuped = Some((covariate, coefficient));
        self
    }

    /// Fit a multi-covariate ANCOVA on independent Bernoulli trial rows.
    #[must_use]
    pub fn with_ancova(mut self, covariates: impl Into<Arc<[VariableId]>>) -> Self {
        self.ancova_covariates = covariates.into();
        self
    }

    /// Target the Wald complier effect from observed treatment receipt.
    #[must_use]
    pub fn with_received_treatment(mut self, received: impl Into<Arc<[bool]>>) -> Self {
        self.estimand = RandomizedEstimand::ComplierAverageCausalEffect;
        self.received_treatment = Some(received.into());
        self
    }

    /// Target recipients under declared one-sided noncompliance (no control-arm receipt).
    #[must_use]
    pub fn with_treatment_on_treated(mut self, received: impl Into<Arc<[bool]>>) -> Self {
        self.estimand = RandomizedEstimand::TreatmentOnTreated;
        self.received_treatment = Some(received.into());
        self
    }

    /// Enumerate every fixed-count assignment under the Fisher sharp null.
    #[must_use]
    pub fn with_exact_randomization_test(mut self) -> Self {
        self.exact_randomization_test = true;
        self
    }

    /// Validate row alignment, allocations, and assignment probabilities.
    pub fn validate(&self) -> Result<(), QueryError> {
        if let RandomizationDesign::MultiArm { assignment, probabilities, arms } = &self.design {
            let n = assignment.len();
            if self.estimand != RandomizedEstimand::IntentionToTreat || self.received_treatment.is_some()
                || self.fixed_cuped.is_some() || !self.ancova_covariates.is_empty()
                || self.exact_randomization_test
            {
                return Err(QueryError::InvalidRandomizedEffect("multi-arm contrasts do not combine with receipt adjustment, CUPED, ANCOVA, or exact Fisher inference".into()));
            }
            if n < 3 || arms.len() < 3 || probabilities.len() != n
                || self.realized_assignment.len() != n || self.assignment_probabilities.len() != n
                || self.assignment_units.len() != n || self.outcome_units.len() != n
                || arms.iter().any(|arm| arm.trim().is_empty())
                || arms.iter().collect::<std::collections::HashSet<_>>().len() != arms.len()
                || self.assignment_units.iter().any(|unit| unit.trim().is_empty())
                || self.outcome_units.iter().any(|unit| unit.trim().is_empty())
                || self.assignment_units.iter().collect::<std::collections::HashSet<_>>().len() != n
                || self.outcome_units.iter().collect::<std::collections::HashSet<_>>().len() != n
                || self.treatment_arms.0 != arms[0] || self.treatment_arms.1 != arms[1]
            {
                return Err(QueryError::InvalidRandomizedEffect("multi-arm labels, assignment, probabilities, and distinct unit identities must align".into()));
            }
            let mut observed = vec![0_usize; arms.len()];
            for i in 0..n {
                let arm = assignment[i];
                let row = &probabilities[i];
                if arm >= arms.len() || row.len() != arms.len()
                    || row.iter().any(|p| !p.is_finite() || *p <= 0.0 || *p >= 1.0)
                    || (row.iter().sum::<f64>() - 1.0).abs() > 1e-8
                    || self.realized_assignment[i] != (arm != 0)
                    || (self.assignment_probabilities[i] - row[arm]).abs() > 1e-12
                {
                    return Err(QueryError::InvalidRandomizedEffect("multi-arm observed actions and known probability rows must agree".into()));
                }
                observed[arm] += 1;
            }
            if observed.contains(&0) {
                return Err(QueryError::InvalidRandomizedEffect("multi-arm positivity failure: every declared arm needs observed support".into()));
            }
            return Ok(());
        }
        if self.exact_randomization_test
            && (!matches!(self.design, RandomizationDesign::Complete { .. })
                || self.received_treatment.is_some()
                || self.fixed_cuped.is_some()
                || !self.ancova_covariates.is_empty()
                || self.realized_assignment.len() > 20)
        {
            return Err(QueryError::InvalidRandomizedEffect(
                "exact randomization inference requires an unadjusted complete two-arm design with at most 20 units".into(),
            ));
        }
        if matches!(self.estimand, RandomizedEstimand::ComplierAverageCausalEffect | RandomizedEstimand::TreatmentOnTreated) {
            if !matches!(self.design, RandomizationDesign::Bernoulli)
                || self.fixed_cuped.is_some()
                || !self.ancova_covariates.is_empty()
                || self.received_treatment.as_ref().is_none_or(|receipt| receipt.len() != self.realized_assignment.len())
            {
                return Err(QueryError::InvalidRandomizedEffect(
                    "CACE/LATE requires Bernoulli assignment, row-aligned treatment receipt, and no CUPED adjustment".into(),
                ));
            }
            if self.estimand == RandomizedEstimand::TreatmentOnTreated
                && self.received_treatment.as_ref().is_some_and(|receipt|
                    receipt.iter().zip(self.realized_assignment.iter()).any(|(received, assigned)| *received && !*assigned))
            {
                return Err(QueryError::InvalidRandomizedEffect(
                    "treatment-on-treated requires observed one-sided noncompliance: no control-assigned unit received treatment".into(),
                ));
            }
        } else if self.received_treatment.is_some() {
            return Err(QueryError::InvalidRandomizedEffect(
                "treatment receipt is only valid for a CACE/LATE or treatment-on-treated query".into(),
            ));
        }
        if let Some((covariate, coefficient)) = self.fixed_cuped {
            if !matches!(self.design, RandomizationDesign::Bernoulli)
                || covariate == self.outcome
                || !coefficient.is_finite()
            {
                return Err(QueryError::InvalidRandomizedEffect(
                    "fixed CUPED requires Bernoulli assignment, a distinct pre-assignment covariate, and a finite externally fixed coefficient".into(),
                ));
            }
        }
        if !self.ancova_covariates.is_empty() {
            let unique = self.ancova_covariates.iter().copied().collect::<std::collections::HashSet<_>>();
            if !matches!(self.design, RandomizationDesign::Bernoulli)
                || self.fixed_cuped.is_some() || self.received_treatment.is_some()
                || self.ancova_covariates.iter().any(|id| *id == self.outcome)
                || unique.len() != self.ancova_covariates.len()
                || self.realized_assignment.len() <= self.ancova_covariates.len() + 2
                || self.assignment_probabilities.first().is_none_or(|first| self.assignment_probabilities.iter().any(|p| (*p - *first).abs() > 1e-12))
            {
                return Err(QueryError::InvalidRandomizedEffect(
                    "ANCOVA requires distinct pre-assignment covariates, independent unit-level Bernoulli assignment with a common probability, and residual degrees of freedom".into(),
                ));
            }
        }
        if matches!(self.design, RandomizationDesign::Factorial2x2 { .. })
            && (self.received_treatment.is_some() || self.fixed_cuped.is_some())
        {
            return Err(QueryError::InvalidRandomizedEffect(
                "factorial contrasts do not combine with receipt adjustment or CUPED on this route".into(),
            ));
        }
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
        if !matches!(self.design, RandomizationDesign::Cluster { .. } | RandomizationDesign::Switchback { .. })
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
            RandomizationDesign::MultiArm { .. } => unreachable!("validated above"),
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
            RandomizationDesign::Factorial2x2 { second_factor_assignment, cell_counts, second_factor_arms } => {
                if second_factor_assignment.len() != n || cell_counts.iter().any(|count| *count < 2)
                    || cell_counts.iter().sum::<usize>() != n
                    || second_factor_arms.0.trim().is_empty() || second_factor_arms.1.trim().is_empty()
                    || second_factor_arms.0 == second_factor_arms.1
                {
                    return Err(QueryError::InvalidRandomizedEffect(
                        "factorial design requires four cells with at least two units each, aligned second-factor assignment, and distinct labels".into(),
                    ));
                }
                let mut observed = [0_usize; 4];
                for (&a, &b) in self.realized_assignment.iter().zip(second_factor_assignment.iter()) {
                    observed[usize::from(a) + 2 * usize::from(b)] += 1;
                }
                let expected_primary = (cell_counts[1] + cell_counts[3]) as f64 / n as f64;
                if observed != *cell_counts || self.assignment_probabilities.iter().any(|p| (*p - expected_primary).abs() > 1e-12) {
                    return Err(QueryError::InvalidRandomizedEffect(
                        "factorial observed cell counts and primary-factor inclusion probabilities must match the fixed four-cell allocation".into(),
                    ));
                }
            }
            RandomizationDesign::Switchback { periods } => {
                if periods.len() != n || periods.iter().any(|period| period.trim().is_empty()) {
                    return Err(QueryError::InvalidRandomizedEffect(
                        "switchback period labels must be non-empty and aligned with rows".into(),
                    ));
                }
                let mut pairs = std::collections::HashSet::new();
                let mut sequences = std::collections::BTreeMap::<&str, [bool; 2]>::new();
                for i in 0..n {
                    if !pairs.insert((&*self.assignment_units[i], &*periods[i])) {
                        return Err(QueryError::InvalidRandomizedEffect(
                            "switchback periods must be unique within each sequence".into(),
                        ));
                    }
                    sequences.entry(&self.assignment_units[i])
                        .or_insert([false; 2])[usize::from(self.realized_assignment[i])] = true;
                }
                if sequences.len() < 2 || sequences.values().any(|arms| !arms[0] || !arms[1]) {
                    return Err(QueryError::InvalidRandomizedEffect(
                        "switchback variance requires at least two independent sequences with both observed arms in each".into(),
                    ));
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
    fn fixed_cuped_requires_distinct_covariate_and_finite_coefficient() {
        assert!(query().with_fixed_cuped(VariableId::from_raw(1), 2.0).validate().is_ok());
        assert!(query().with_fixed_cuped(VariableId::from_raw(0), 2.0).validate().is_err());
        assert!(query().with_fixed_cuped(VariableId::from_raw(1), f64::NAN).validate().is_err());
    }

    #[test]
    fn switchback_rejects_reused_period_and_single_sequence() {
        let mut query = RandomizedEffectQuery::with_design(
            RandomizationDesign::Switchback {
                periods: ["p0", "p1", "p0", "p1"].map(Arc::<str>::from).into(),
            },
            VariableId::from_raw(0), [true, false, true, false], [0.5; 4],
            ["s0", "s0", "s1", "s1"].map(Arc::<str>::from),
            ["r0", "r1", "r2", "r3"].map(Arc::<str>::from),
            ("off", "on"),
        );
        assert!(query.validate().is_ok());
        query.assignment_units = ["s0", "s0", "s0", "s0"].map(Arc::<str>::from).into();
        assert!(query.validate().is_err());
        query.assignment_units = ["s0", "s0", "s1", "s1"].map(Arc::<str>::from).into();
        query.design = RandomizationDesign::Switchback {
            periods: ["p0", "p0", "p0", "p1"].map(Arc::<str>::from).into(),
        };
        assert!(query.validate().is_err());
    }

    #[test]
    fn cace_requires_aligned_receipt_and_bernoulli_assignment() {
        let base = query();
        assert!(base.clone().with_received_treatment([true]).validate().is_err());
        let mut complete = base.with_received_treatment([true, false, true, false]);
        complete.design = RandomizationDesign::Complete { treated_units: 2 };
        assert!(complete.validate().is_err());
    }

    #[test]
    fn factorial_requires_four_supported_fixed_cells() {
        let base = RandomizedEffectQuery::with_design(
            RandomizationDesign::Factorial2x2 {
                second_factor_assignment: [false, false, false, false, true, true, true, true].into(),
                cell_counts: [2; 4],
                second_factor_arms: (Arc::from("off"), Arc::from("on")),
            },
            VariableId::from_raw(0),
            [false, false, true, true, false, false, true, true],
            [0.5; 8],
            (0..8).map(|i| Arc::<str>::from(format!("u{i}"))).collect::<Vec<_>>(),
            (0..8).map(|i| Arc::<str>::from(format!("y{i}"))).collect::<Vec<_>>(),
            ("control", "treated"),
        );
        assert!(base.validate().is_ok());
        let mut missing_cell = base.clone();
        missing_cell.design = RandomizationDesign::Factorial2x2 {
            second_factor_assignment: [false, false, false, false, true, true, true, true].into(),
            cell_counts: [3, 1, 2, 2],
            second_factor_arms: (Arc::from("off"), Arc::from("on")),
        };
        assert!(missing_cell.validate().is_err());
        assert!(base.with_exact_randomization_test().validate().is_err());
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
