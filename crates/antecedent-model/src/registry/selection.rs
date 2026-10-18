//! Candidate constraints, validation-split declarations and selection provenance.
//!
//! The registry-wide family lists stay the default. A [`SelectionOptions`] narrows
//! or overrides them per node and declares how the held-out scores are cut when
//! rows are not exchangeable; the assignment then records, beside every candidate's
//! score, exactly which candidate set, split scheme and seed produced it.

use std::sync::Arc;

use antecedent_core::VariableId;

use crate::error::ModelError;

use super::{MechanismFamily, MechanismTyping};

/// Constraint on the candidate families of one node.
///
/// Resolution order: the node's list is [`Self::allowed`] when set, else the
/// registry's list for the node's type; [`Self::excluded`] families are then
/// removed; a [`Self::required`] family is selected whatever its rank (every
/// other remaining candidate is still scored on the same folds, so the choice
/// stays auditable).
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct NodeFamilyConstraint {
    /// When set, exactly these families are the node's candidates, replacing the
    /// registry list for the node's type (so a family outside the registry
    /// preset can be admitted for one node).
    pub allowed: Option<Arc<[MechanismFamily]>>,
    /// Families removed from the node's candidates.
    pub excluded: Arc<[MechanismFamily]>,
    /// Family selected for the node. Added to the candidate list when the
    /// registry list lacks it, unless [`Self::allowed`] is set and omits it
    /// (a contradiction, refused).
    pub required: Option<MechanismFamily>,
}

/// How rows are partitioned into training and held-out sets when candidates are scored.
///
/// Every candidate of a node is scored on the same partition, so scores stay
/// comparable. Exchangeable rows use the seeded shuffled folds; dependent rows
/// must declare their dependence, because a shuffled fold lets a held-out row's
/// neighbours train the model that predicts it and inflates flexible families.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum ValidationSplits {
    /// Rows are exchangeable: seeded, shuffled folds (stratified on the outcome
    /// for categorical families).
    #[default]
    Shuffled,
    /// Rows are in time order and serially dependent. Forward chaining over
    /// contiguous blocks: the rows are cut into one more block than there are
    /// folds, fold `k` holds out block `k` and trains on the rows before it, so no
    /// held-out row precedes a training row. The first block is only ever
    /// training. `purge` rows immediately before each held-out block are dropped
    /// from training so serial correlation across the boundary does not leak.
    TimeOrdered {
        /// Rows removed from the end of each training window.
        purge: usize,
    },
    /// Rows belong to units (the schema's `UnitId` column). Whole units are held
    /// out together, so a unit never trains the model that predicts it.
    Clustered,
}

impl ValidationSplits {
    /// Stable scheme id: `shuffled`, `time_ordered` or `clustered`.
    #[must_use]
    pub const fn id(self) -> &'static str {
        match self {
            Self::Shuffled => "shuffled",
            Self::TimeOrdered { .. } => "time_ordered",
            Self::Clustered => "clustered",
        }
    }
}

/// Per-call selection options. The default reproduces the registry-wide
/// behaviour exactly: no constraints and shuffled folds.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SelectionOptions {
    /// Constraints keyed by the node's variable. At most one per variable.
    pub constraints: Vec<(VariableId, NodeFamilyConstraint)>,
    /// How candidate scores are cut.
    pub splits: ValidationSplits,
}

impl SelectionOptions {
    /// Constraint declared for `variable`.
    #[must_use]
    pub fn constraint_for(&self, variable: VariableId) -> Option<&NodeFamilyConstraint> {
        self.constraints.iter().find(|(v, _)| *v == variable).map(|(_, c)| c)
    }
}

/// Which rule picked a node's family.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum SelectionRule {
    /// The highest-scoring candidate.
    BestScore,
    /// The call-wide [`super::SelectionPolicy::RequireFamily`].
    RequiredByPolicy,
    /// The node's own [`NodeFamilyConstraint::required`].
    RequiredByNodeConstraint,
}

impl SelectionRule {
    /// Stable rule id.
    #[must_use]
    pub const fn id(self) -> &'static str {
        match self {
            Self::BestScore => "best_score",
            Self::RequiredByPolicy => "required_by_policy",
            Self::RequiredByNodeConstraint => "required_by_node_constraint",
        }
    }
}

/// Selection provenance of one node: everything needed to reproduce the scores.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SelectionProvenance {
    /// Split scheme the held-out scores were cut with.
    pub splits: ValidationSplits,
    /// Held-out folds actually planned.
    pub folds: usize,
    /// Rows the partition covered.
    pub n_rows: usize,
    /// Seed of the shuffled partition; `None` when the scheme is deterministic
    /// by construction (time-ordered blocks).
    pub seed: Option<u64>,
    /// Families considered, in listing order (scored, failed and unfit alike).
    pub candidate_set: Arc<[MechanismFamily]>,
    /// The rule that picked the selected family.
    pub rule: SelectionRule,
    /// The per-node constraint that shaped the candidate set, when one applied.
    pub constraint: Option<NodeFamilyConstraint>,
}

/// Candidate families of one node after applying `constraint` to the registry's
/// list for the node's type.
///
/// # Errors
///
/// [`ModelError::InvalidSelection`] when the constraint is contradictory, requires
/// a family of the wrong outcome type, or leaves no candidate.
pub(super) fn resolve_candidates(
    registry_list: &[MechanismFamily],
    typing: MechanismTyping,
    constraint: Option<&NodeFamilyConstraint>,
    variable: VariableId,
) -> Result<Vec<MechanismFamily>, ModelError> {
    let Some(constraint) = constraint else {
        return Ok(registry_list.to_vec());
    };
    let invalid = |message: String| ModelError::InvalidSelection { message };
    if constraint.allowed.as_ref().is_some_and(|a| a.is_empty()) {
        return Err(invalid(format!("variable {variable}: the allowed family list is empty")));
    }
    let mut set: Vec<MechanismFamily> = Vec::new();
    let base: &[MechanismFamily] = constraint.allowed.as_deref().unwrap_or(registry_list);
    for &family in base {
        if !constraint.excluded.contains(&family) && !set.contains(&family) {
            set.push(family);
        }
    }
    if let Some(required) = constraint.required {
        if constraint.excluded.contains(&required) {
            return Err(invalid(format!(
                "variable {variable}: required family {} is also excluded",
                required.id()
            )));
        }
        if !set.contains(&required) {
            if constraint.allowed.is_some() {
                return Err(invalid(format!(
                    "variable {variable}: required family {} is not in the allowed list",
                    required.id()
                )));
            }
            set.push(required);
        }
    }
    let categorical_node = typing != MechanismTyping::Continuous;
    let fits_node = |family: &MechanismFamily| {
        if categorical_node { !family.is_continuous() } else { !family.is_categorical() }
    };
    if let Some(required) = constraint.required {
        if !fits_node(&required) {
            return Err(invalid(format!(
                "variable {variable} is typed {} but required family {} models a {} outcome",
                if categorical_node { "categorical" } else { "continuous" },
                required.id(),
                if categorical_node { "continuous" } else { "categorical" },
            )));
        }
    }
    // An outcome-agnostic candidate list (`allowed`) names families of both outcome
    // types; the ones that cannot model this node are not its candidates. The
    // provenance keeps the declared constraint and the resolved set, so nothing is
    // dropped unrecorded.
    set.retain(fits_node);
    if set.is_empty() {
        return Err(invalid(format!("variable {variable}: no candidate family remains")));
    }
    Ok(set)
}

#[cfg(test)]
mod tests {
    use super::super::cv::{split_plan, validation_folds};
    use super::*;

    fn variable(raw: u32) -> VariableId {
        VariableId::from_raw(raw)
    }

    fn ids(families: &[MechanismFamily]) -> Vec<&'static str> {
        families.iter().map(|f| f.id()).collect()
    }

    #[test]
    fn shuffled_folds_partition_the_rows_and_are_the_default() {
        assert_eq!(ValidationSplits::default(), ValidationSplits::Shuffled);
        let folds = validation_folds(100, ValidationSplits::Shuffled, None, None).unwrap();
        assert_eq!(folds.len(), 5);
        let mut held_out: Vec<usize> = folds.iter().flat_map(|f| f.test.clone()).collect();
        held_out.sort_unstable();
        assert_eq!(held_out, (0..100).collect::<Vec<_>>());
        for fold in &folds {
            assert_eq!(fold.train.len() + fold.test.len(), 100);
        }
        let again = validation_folds(100, ValidationSplits::Shuffled, None, None).unwrap();
        assert!(folds.iter().zip(&again).all(|(a, b)| a.test == b.test));
    }

    #[test]
    fn time_ordered_folds_never_train_on_a_row_at_or_after_a_held_out_row() {
        let n = 600;
        let purge = 7;
        let folds =
            validation_folds(n, ValidationSplits::TimeOrdered { purge }, None, None).unwrap();
        assert_eq!(folds.len(), 5);
        let mut previous_end = 0;
        for fold in &folds {
            let first_test = *fold.test.first().unwrap();
            assert_eq!(first_test, previous_end.max(n / 6), "held-out blocks are contiguous");
            previous_end = *fold.test.last().unwrap() + 1;
            let last_train = *fold.train.last().unwrap();
            assert!(last_train + purge < first_test, "no training row inside the purge window");
            assert!(fold.train.iter().all(|r| *r < first_test), "training precedes validation");
        }
        assert_eq!(previous_end, n);
        // Windows expand: each fold trains on more rows than the one before.
        assert!(folds.windows(2).all(|w| w[0].train.len() < w[1].train.len()));
    }

    #[test]
    fn time_ordered_folds_refuse_a_series_too_short_to_cut() {
        let error = validation_folds(20, ValidationSplits::TimeOrdered { purge: 0 }, None, None)
            .map(|_| ())
            .unwrap_err();
        assert!(matches!(error, ModelError::InvalidSelection { .. }), "{error}");
        assert!(split_plan(20, ValidationSplits::TimeOrdered { purge: 0 }, None).is_err());
    }

    #[test]
    fn clustered_folds_hold_whole_units_out_together() {
        let groups: Vec<u32> = (0..200_u32).map(|r| r % 23).collect();
        let folds =
            validation_folds(200, ValidationSplits::Clustered, Some(&groups), None).unwrap();
        assert_eq!(folds.len(), 5);
        for fold in &folds {
            let held: std::collections::HashSet<u32> =
                fold.test.iter().map(|&r| groups[r]).collect();
            assert!(fold.train.iter().all(|&r| !held.contains(&groups[r])));
            assert_eq!(fold.train.len() + fold.test.len(), 200);
        }
        let mut seen: Vec<usize> = folds.iter().flat_map(|f| f.test.clone()).collect();
        seen.sort_unstable();
        assert_eq!(seen, (0..200).collect::<Vec<_>>());
    }

    #[test]
    fn clustered_folds_need_unit_structure() {
        for groups in [None, Some(vec![0_u32; 50]), Some((0..50).collect::<Vec<u32>>())] {
            let error = validation_folds(50, ValidationSplits::Clustered, groups.as_deref(), None)
                .map(|_| ())
                .unwrap_err();
            assert!(matches!(error, ModelError::InvalidSelection { .. }), "{error}");
        }
    }

    #[test]
    fn constraints_resolve_to_the_documented_candidate_set() {
        use MechanismFamily as F;
        let registry = [F::LinearGaussian, F::LinearSpline, F::Constant];
        let node = variable(0);
        let resolve = |constraint: NodeFamilyConstraint, typing| {
            resolve_candidates(&registry, typing, Some(&constraint), node)
        };
        // No constraint: the registry list, untouched.
        assert_eq!(
            resolve_candidates(&registry, MechanismTyping::Continuous, None, node).unwrap(),
            registry
        );
        // `allowed` replaces the list, including with a family the preset lacks; families
        // of the other outcome type are not candidates of this node.
        let allowed = NodeFamilyConstraint {
            allowed: Some(Arc::from([F::TreeBoostedAdditive, F::Discrete, F::LinearGaussian])),
            ..NodeFamilyConstraint::default()
        };
        assert_eq!(
            ids(&resolve(allowed, MechanismTyping::Continuous).unwrap()),
            ["tree_boosted_additive", "linear_gaussian"]
        );
        // `excluded` removes; `required` is added when the list lacks it.
        let excluded_and_required = NodeFamilyConstraint {
            excluded: Arc::from([F::LinearSpline]),
            required: Some(F::TreeForestAdditive),
            ..NodeFamilyConstraint::default()
        };
        assert_eq!(
            ids(&resolve(excluded_and_required, MechanismTyping::Continuous).unwrap()),
            ["linear_gaussian", "constant", "tree_forest_additive"]
        );
    }

    #[test]
    fn contradictory_constraints_are_refused() {
        use MechanismFamily as F;
        let registry = [F::LinearGaussian, F::Constant];
        let refused = |constraint: NodeFamilyConstraint, typing| {
            matches!(
                resolve_candidates(&registry, typing, Some(&constraint), variable(0)),
                Err(ModelError::InvalidSelection { .. })
            )
        };
        let required = |family| NodeFamilyConstraint {
            required: Some(family),
            ..NodeFamilyConstraint::default()
        };
        // Required and excluded.
        assert!(refused(
            NodeFamilyConstraint {
                excluded: Arc::from([F::LinearGaussian]),
                required: Some(F::LinearGaussian),
                ..NodeFamilyConstraint::default()
            },
            MechanismTyping::Continuous
        ));
        // Required outside an explicit allowed list.
        assert!(refused(
            NodeFamilyConstraint {
                allowed: Some(Arc::from([F::LinearSpline])),
                required: Some(F::LinearGaussian),
                ..NodeFamilyConstraint::default()
            },
            MechanismTyping::Continuous
        ));
        // Required of the wrong outcome type.
        assert!(refused(required(F::Discrete), MechanismTyping::Continuous));
        assert!(refused(required(F::LinearGaussian), MechanismTyping::DeclaredDiscrete));
        // Empty allowed list; and an allowed list with nothing that fits the node.
        assert!(refused(
            NodeFamilyConstraint {
                allowed: Some(Arc::from([])),
                ..NodeFamilyConstraint::default()
            },
            MechanismTyping::Continuous
        ));
        assert!(refused(
            NodeFamilyConstraint {
                allowed: Some(Arc::from([F::Discrete])),
                ..NodeFamilyConstraint::default()
            },
            MechanismTyping::Continuous
        ));
    }
}
