//! Query contract for two-arm randomized experiments.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::probability::{probabilities_agree, row_totals_one};
use super::probability_rows::ProbabilityRows;
use super::{QueryError, RandomizedEffectFault};
use crate::canonical_encode::{
    CanonicalEncode, count, encode_flags, encode_usizes, encode_variables, flag,
};
use crate::identity::CanonicalWriter;
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
        /// Known action probabilities in declared label order for each row, in one flat
        /// table whose stride is the arm count.
        probabilities: ProbabilityRows,
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

/// The exact graphless support-key tuple `(family, design, method,
/// assignment_unit, inference_claim)` for a retained randomized query, shared by
/// the execute-side license handshake and the contract-side matrix coordinate so
/// the two cannot drift. The facets are independent structural properties the
/// query already fixed upstream (and `RandomizedEffectQuery::validate` guarantees
/// receipt/CUPED/ANCOVA only occur with Bernoulli assignment), so this is a pure
/// lookup, not the license decision itself.
#[must_use]
pub fn randomized_graphless_coordinate(
    design: &RandomizationDesign,
    estimand: RandomizedEstimand,
    received_treatment: bool,
    fixed_cuped: bool,
    has_ancova: bool,
) -> (&'static str, &'static str, &'static str, &'static str, &'static str) {
    match design {
        RandomizationDesign::Bernoulli if estimand == RandomizedEstimand::TreatmentOnTreated => (
            "complier_effect",
            "bernoulli_one_sided",
            "wald_ratio_influence",
            "unit",
            "pointwise_95_normal_interval",
        ),
        RandomizationDesign::Bernoulli if received_treatment => (
            "complier_effect",
            "bernoulli",
            "wald_ratio_influence",
            "unit",
            "pointwise_95_normal_interval",
        ),
        RandomizationDesign::Bernoulli if fixed_cuped => (
            "randomized_effect",
            "bernoulli",
            "fixed_cuped_ht_score",
            "unit",
            "pointwise_95_normal_interval",
        ),
        RandomizationDesign::Bernoulli if has_ancova => {
            ("randomized_effect", "bernoulli", "ancova_hc0", "unit", "pointwise_95_normal_interval")
        }
        RandomizationDesign::Bernoulli => (
            "randomized_effect",
            "bernoulli",
            "independent_action_ht_score",
            "unit",
            "pointwise_95_normal_interval",
        ),
        RandomizationDesign::Complete { .. } => (
            "randomized_effect",
            "complete",
            "neyman_difference_in_means",
            "unit",
            "pointwise_95_normal_interval",
        ),
        RandomizationDesign::Cluster { .. } => (
            "randomized_effect",
            "cluster",
            "neyman_unit_weighted_cluster_totals",
            "cluster",
            "pointwise_95_normal_interval",
        ),
        RandomizationDesign::Stratified { .. } => (
            "randomized_effect",
            "stratified",
            "blocked_neyman_difference_in_means",
            "unit",
            "pointwise_95_normal_interval",
        ),
        RandomizationDesign::Factorial2x2 { .. } => (
            "randomized_effect",
            "factorial_2x2",
            "fixed_cell_neyman_contrasts",
            "unit",
            "three_pointwise_95_normal_intervals",
        ),
        RandomizationDesign::MultiArm { .. } => (
            "randomized_effect",
            "multi_arm",
            "independent_action_ht_scores",
            "unit",
            "all_action_pointwise_95_normal_intervals",
        ),
        RandomizationDesign::Switchback { .. } => (
            "randomized_effect",
            "switchback",
            "independent_sequence_ht_score",
            "sequence",
            "pointwise_95_student_interval",
        ),
    }
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
    /// Request two-sided Fisher sharp-null randomization inference: exhaustive over the
    /// fixed-count assignments when they are few, a Monte Carlo permutation test otherwise.
    pub exact_randomization_test: bool,
}

/// Every array is row-, block- or arm-aligned and keeps its order; the arm
/// labels keep their positions.
impl CanonicalEncode for RandomizationDesign {
    fn encode(&self, writer: &mut CanonicalWriter) {
        match self {
            Self::Bernoulli => {
                writer.word(1);
            }
            Self::Complete { treated_units } => {
                writer.word(2).word(count(*treated_units));
            }
            Self::Cluster { treated_clusters } => {
                writer.word(3).word(count(*treated_clusters));
            }
            Self::Stratified { blocks, treated_per_row } => {
                writer.word(4).texts(blocks);
                encode_usizes(writer, treated_per_row);
            }
            Self::Factorial2x2 { second_factor_assignment, cell_counts, second_factor_arms } => {
                writer.word(5);
                encode_flags(writer, second_factor_assignment);
                encode_usizes(writer, cell_counts);
                writer.text(&second_factor_arms.0).text(&second_factor_arms.1);
            }
            Self::MultiArm { assignment, probabilities, arms } => {
                writer.word(6);
                encode_usizes(writer, assignment);
                writer.word(count(probabilities.len()));
                for row in probabilities.iter() {
                    writer.reals(row);
                }
                writer.texts(arms);
            }
            Self::Switchback { periods } => {
                writer.word(7).texts(periods);
            }
        }
    }
}

impl CanonicalEncode for RandomizedEstimand {
    fn encode(&self, writer: &mut CanonicalWriter) {
        let tag = match self {
            Self::IntentionToTreat => 1,
            Self::ComplierAverageCausalEffect => 2,
            Self::TreatmentOnTreated => 3,
        };
        writer.word(tag);
    }
}

/// Every field is written. Row-aligned arrays (assignment, probabilities, unit
/// labels, receipt) keep their order, and the arm pair keeps its roles.
/// `ancova_covariates` keep their order too: they are design-matrix columns,
/// and the conservative choice is never to merge two spellings.
impl CanonicalEncode for RandomizedEffectQuery {
    fn encode(&self, writer: &mut CanonicalWriter) {
        self.estimand.encode(writer);
        self.design.encode(writer);
        self.outcome.encode(writer);
        encode_flags(writer, &self.realized_assignment);
        writer
            .reals(&self.assignment_probabilities)
            .texts(&self.assignment_units)
            .texts(&self.outcome_units)
            .text(&self.treatment_arms.0)
            .text(&self.treatment_arms.1);
        match self.fixed_cuped {
            Some((covariate, coefficient)) => {
                writer.word(1);
                covariate.encode(writer);
                writer.real(coefficient);
            }
            None => {
                writer.word(0);
            }
        }
        encode_variables(writer, &self.ancova_covariates);
        match &self.received_treatment {
            Some(received) => {
                writer.word(1);
                encode_flags(writer, received);
            }
            None => {
                writer.word(0);
            }
        }
        writer.word(flag(self.exact_randomization_test));
    }
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

    /// Test the Fisher sharp null over the fixed-count assignments: every one is enumerated when
    /// they are few (an exact p-value), and a fixed-stream Monte Carlo permutation sample of them
    /// is drawn otherwise.
    #[must_use]
    pub fn with_exact_randomization_test(mut self) -> Self {
        self.exact_randomization_test = true;
        self
    }

    /// Validate row alignment, allocations, and assignment probabilities.
    ///
    /// # Errors
    ///
    /// Returns [`QueryError::RandomizedEffectRefused`] when any design's row
    /// alignment, arm allocations, block or factorial structure, adjustment
    /// covariates, or assignment probabilities are malformed or inconsistent
    /// with the declared estimand.
    // One match over every randomization design; each arm's alignment,
    // allocation, and probability checks are cohesive and share local state.
    #[allow(
        clippy::too_many_lines,
        reason = "one validation per query field, in declaration order"
    )]
    pub fn validate(&self) -> Result<(), QueryError> {
        if let RandomizationDesign::MultiArm { assignment, probabilities, arms } = &self.design {
            let n = assignment.len();
            if self.estimand != RandomizedEstimand::IntentionToTreat
                || self.received_treatment.is_some()
                || self.fixed_cuped.is_some()
                || !self.ancova_covariates.is_empty()
                || self.exact_randomization_test
            {
                return Err(QueryError::RandomizedEffectRefused(
                    RandomizedEffectFault::MultiArmCombination,
                ));
            }
            if n < 3
                || arms.len() < 3
                || probabilities.len() != n
                || self.realized_assignment.len() != n
                || self.assignment_probabilities.len() != n
                || self.assignment_units.len() != n
                || self.outcome_units.len() != n
                || arms.iter().any(|arm| arm.trim().is_empty())
                || arms.iter().collect::<std::collections::HashSet<_>>().len() != arms.len()
                || self.assignment_units.iter().any(|unit| unit.trim().is_empty())
                || self.outcome_units.iter().any(|unit| unit.trim().is_empty())
                || self.assignment_units.iter().collect::<std::collections::HashSet<_>>().len() != n
                || self.outcome_units.iter().collect::<std::collections::HashSet<_>>().len() != n
                || self.treatment_arms.0 != arms[0]
                || self.treatment_arms.1 != arms[1]
            {
                return Err(QueryError::RandomizedEffectRefused(
                    RandomizedEffectFault::MultiArmAlignment,
                ));
            }
            let mut observed = vec![0_usize; arms.len()];
            for i in 0..n {
                let arm = assignment[i];
                let row = &probabilities[i];
                if arm >= arms.len()
                    || row.len() != arms.len()
                    || row.iter().any(|p| !p.is_finite() || *p <= 0.0 || *p >= 1.0)
                    || !row_totals_one(row)
                    || self.realized_assignment[i] != (arm != 0)
                    || !probabilities_agree(self.assignment_probabilities[i], row[arm])
                {
                    return Err(QueryError::RandomizedEffectRefused(
                        RandomizedEffectFault::MultiArmProbabilityRows,
                    ));
                }
                observed[arm] += 1;
            }
            if observed.contains(&0) {
                return Err(QueryError::RandomizedEffectRefused(
                    RandomizedEffectFault::MultiArmPositivity,
                ));
            }
            return Ok(());
        }
        if self.exact_randomization_test
            && (!matches!(self.design, RandomizationDesign::Complete { .. })
                || self.received_treatment.is_some()
                || self.fixed_cuped.is_some()
                || !self.ancova_covariates.is_empty())
        {
            return Err(QueryError::RandomizedEffectRefused(
                RandomizedEffectFault::ExactTestDesign,
            ));
        }
        if matches!(
            self.estimand,
            RandomizedEstimand::ComplierAverageCausalEffect
                | RandomizedEstimand::TreatmentOnTreated
        ) {
            if !matches!(self.design, RandomizationDesign::Bernoulli)
                || self.fixed_cuped.is_some()
                || !self.ancova_covariates.is_empty()
                || self
                    .received_treatment
                    .as_ref()
                    .is_none_or(|receipt| receipt.len() != self.realized_assignment.len())
            {
                return Err(QueryError::RandomizedEffectRefused(
                    RandomizedEffectFault::ReceiptDesign,
                ));
            }
            if self.estimand == RandomizedEstimand::TreatmentOnTreated
                && self.received_treatment.as_ref().is_some_and(|receipt| {
                    receipt
                        .iter()
                        .zip(self.realized_assignment.iter())
                        .any(|(received, assigned)| *received && !*assigned)
                })
            {
                return Err(QueryError::RandomizedEffectRefused(
                    RandomizedEffectFault::TreatmentOnTreatedNoncompliance,
                ));
            }
        } else if self.received_treatment.is_some() {
            return Err(QueryError::RandomizedEffectRefused(
                RandomizedEffectFault::ReceiptWithoutComplier,
            ));
        }
        if let Some((covariate, coefficient)) = self.fixed_cuped {
            if !matches!(self.design, RandomizationDesign::Bernoulli)
                || covariate == self.outcome
                || !coefficient.is_finite()
            {
                return Err(QueryError::RandomizedEffectRefused(
                    RandomizedEffectFault::FixedCupedDesign,
                ));
            }
        }
        if !self.ancova_covariates.is_empty() {
            let unique =
                self.ancova_covariates.iter().copied().collect::<std::collections::HashSet<_>>();
            if !matches!(self.design, RandomizationDesign::Bernoulli)
                || self.fixed_cuped.is_some()
                || self.received_treatment.is_some()
                || self.ancova_covariates.iter().any(|id| *id == self.outcome)
                || unique.len() != self.ancova_covariates.len()
                || self.realized_assignment.len() <= self.ancova_covariates.len() + 2
                || self.assignment_probabilities.first().is_none_or(|first| {
                    self.assignment_probabilities.iter().any(|p| !probabilities_agree(*p, *first))
                })
            {
                return Err(QueryError::RandomizedEffectRefused(
                    RandomizedEffectFault::AncovaDesign,
                ));
            }
        }
        if matches!(self.design, RandomizationDesign::Factorial2x2 { .. })
            && (self.received_treatment.is_some() || self.fixed_cuped.is_some())
        {
            return Err(QueryError::RandomizedEffectRefused(
                RandomizedEffectFault::FactorialCombination,
            ));
        }
        let n = self.realized_assignment.len();
        if n < 2
            || self.assignment_probabilities.len() != n
            || self.assignment_units.len() != n
            || self.outcome_units.len() != n
        {
            return Err(QueryError::RandomizedEffectRefused(RandomizedEffectFault::RowAlignment));
        }
        if self.assignment_probabilities.iter().any(|p| !p.is_finite() || *p <= 0.0 || *p >= 1.0) {
            return Err(QueryError::RandomizedEffectRefused(
                RandomizedEffectFault::AssignmentProbabilityRange,
            ));
        }
        if self.assignment_units.iter().any(|unit| unit.trim().is_empty())
            || self.outcome_units.iter().any(|unit| unit.trim().is_empty())
        {
            return Err(QueryError::RandomizedEffectRefused(RandomizedEffectFault::EmptyUnitLabel));
        }
        if self.outcome_units.iter().collect::<std::collections::HashSet<_>>().len() != n {
            return Err(QueryError::RandomizedEffectRefused(
                RandomizedEffectFault::DuplicateOutcomeUnit,
            ));
        }
        if !matches!(
            self.design,
            RandomizationDesign::Cluster { .. } | RandomizationDesign::Switchback { .. }
        ) && self.assignment_units.iter().collect::<std::collections::HashSet<_>>().len() != n
        {
            return Err(QueryError::RandomizedEffectRefused(
                RandomizedEffectFault::DuplicateAssignmentUnit,
            ));
        }
        if self.treatment_arms.0.trim().is_empty()
            || self.treatment_arms.1.trim().is_empty()
            || self.treatment_arms.0 == self.treatment_arms.1
        {
            return Err(QueryError::RandomizedEffectRefused(RandomizedEffectFault::ArmLabels));
        }
        if self.realized_assignment.iter().all(|x| *x)
            || self.realized_assignment.iter().all(|x| !*x)
        {
            return Err(QueryError::RandomizedEffectRefused(RandomizedEffectFault::UnobservedArm));
        }
        match &self.design {
            RandomizationDesign::Bernoulli => {}
            RandomizationDesign::MultiArm { .. } => unreachable!("validated above"),
            RandomizationDesign::Complete { treated_units } => {
                if *treated_units < 2 || n.saturating_sub(*treated_units) < 2 {
                    return Err(QueryError::RandomizedEffectRefused(
                        RandomizedEffectFault::CompleteArmSize,
                    ));
                }
                if self.realized_assignment.iter().filter(|assigned| **assigned).count()
                    != *treated_units
                {
                    return Err(QueryError::RandomizedEffectRefused(
                        RandomizedEffectFault::CompleteTreatedCount,
                    ));
                }
                let expected = *treated_units as f64 / n as f64;
                if self.assignment_probabilities.iter().any(|p| !probabilities_agree(*p, expected))
                {
                    return Err(QueryError::RandomizedEffectRefused(
                        RandomizedEffectFault::CompleteProbability,
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
                            || !probabilities_agree(prior_probability, probability)
                        {
                            return Err(QueryError::RandomizedEffectRefused(
                                RandomizedEffectFault::ClusterConstancy,
                            ));
                        }
                    }
                }
                let total = clusters.len();
                if *treated_clusters < 2 || total.saturating_sub(*treated_clusters) < 2 {
                    return Err(QueryError::RandomizedEffectRefused(
                        RandomizedEffectFault::ClusterArmSize,
                    ));
                }
                let observed = clusters.values().filter(|(treated, _)| *treated).count();
                let expected_probability = *treated_clusters as f64 / total as f64;
                if observed != *treated_clusters
                    || clusters
                        .values()
                        .any(|(_, p)| !probabilities_agree(*p, expected_probability))
                {
                    return Err(QueryError::RandomizedEffectRefused(
                        RandomizedEffectFault::ClusterAllocation,
                    ));
                }
            }
            RandomizationDesign::Stratified { blocks, treated_per_row } => {
                if blocks.len() != n
                    || treated_per_row.len() != n
                    || blocks.iter().any(|b| b.trim().is_empty())
                {
                    return Err(QueryError::RandomizedEffectRefused(
                        RandomizedEffectFault::StratifiedAlignment,
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
                        || !probabilities_agree(entry.3, self.assignment_probabilities[i])
                    {
                        return Err(QueryError::RandomizedEffectRefused(
                            RandomizedEffectFault::StratifiedConsistency,
                        ));
                    }
                    entry.0 += 1;
                    entry.1 += usize::from(self.realized_assignment[i]);
                }
                for (_, (total, treated_observed, treated_declared, probability)) in by_block {
                    let treated_declared = treated_declared.unwrap_or_default();
                    if treated_declared < 2 || total.saturating_sub(treated_declared) < 2 {
                        return Err(QueryError::RandomizedEffectRefused(
                            RandomizedEffectFault::StratifiedArmSize,
                        ));
                    }
                    if treated_observed != treated_declared
                        || !probabilities_agree(probability, treated_declared as f64 / total as f64)
                    {
                        return Err(QueryError::RandomizedEffectRefused(
                            RandomizedEffectFault::StratifiedAllocation,
                        ));
                    }
                }
            }
            RandomizationDesign::Factorial2x2 {
                second_factor_assignment,
                cell_counts,
                second_factor_arms,
            } => {
                if second_factor_assignment.len() != n
                    || cell_counts.iter().any(|count| *count < 2)
                    || cell_counts.iter().sum::<usize>() != n
                    || second_factor_arms.0.trim().is_empty()
                    || second_factor_arms.1.trim().is_empty()
                    || second_factor_arms.0 == second_factor_arms.1
                {
                    return Err(QueryError::RandomizedEffectRefused(
                        RandomizedEffectFault::FactorialCells,
                    ));
                }
                let mut observed = [0_usize; 4];
                for (&a, &b) in self.realized_assignment.iter().zip(second_factor_assignment.iter())
                {
                    observed[usize::from(a) + 2 * usize::from(b)] += 1;
                }
                let expected_primary = (cell_counts[1] + cell_counts[3]) as f64 / n as f64;
                if observed != *cell_counts
                    || self
                        .assignment_probabilities
                        .iter()
                        .any(|p| !probabilities_agree(*p, expected_primary))
                {
                    return Err(QueryError::RandomizedEffectRefused(
                        RandomizedEffectFault::FactorialAllocation,
                    ));
                }
            }
            RandomizationDesign::Switchback { periods } => {
                if periods.len() != n || periods.iter().any(|period| period.trim().is_empty()) {
                    return Err(QueryError::RandomizedEffectRefused(
                        RandomizedEffectFault::SwitchbackLabels,
                    ));
                }
                let mut pairs = std::collections::HashSet::new();
                let mut sequences = std::collections::BTreeSet::<&str>::new();
                for i in 0..n {
                    if !pairs.insert((&*self.assignment_units[i], &*periods[i])) {
                        return Err(QueryError::RandomizedEffectRefused(
                            RandomizedEffectFault::SwitchbackDuplicatePeriod,
                        ));
                    }
                    sequences.insert(&self.assignment_units[i]);
                }
                if sequences.len() < 2 {
                    return Err(QueryError::RandomizedEffectRefused(
                        RandomizedEffectFault::SwitchbackSequences,
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
            VariableId::from_raw(0),
            [true, false, true, false],
            [0.5; 4],
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
        query.design = RandomizationDesign::Switchback {
            periods: ["p0", "p1", "p0", "p1"].map(Arc::<str>::from).into(),
        };
        query.realized_assignment = [true, true, false, false].into();
        assert!(
            query.validate().is_ok(),
            "global arm support suffices when one sequence happens to realize one arm"
        );
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
                second_factor_assignment: [false, false, false, false, true, true, true, true]
                    .into(),
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
