//! Sample construction policies.
//!
//! Distinct from `antecedent_counterfactual::AbductionMissingPolicy` (abduction).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::VariableId;

/// How missing values (validity bits) affect row selection.
///
/// Each policy is a declaration about the missingness mechanism as well as a row rule; the
/// route that executes it states the assumption it needs (see the missing-data guide).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Default)]
#[non_exhaustive]
pub enum MissingPolicy {
    /// Drop any row where a requested column is invalid.
    ///
    /// Unbiased when the dropped rows are a random subset (missing completely at random) or,
    /// for a regression of the outcome on the retained covariates, when missingness depends
    /// only on those covariates. The dropped rows are always counted.
    #[default]
    CompleteCase,
    /// Fail if any requested column has an invalid value in a candidate row.
    ErrorOnMissing,
    /// Randomized outcomes missing at random given the arm and declared baseline covariates:
    /// each responder is weighted by the inverse of its estimated probability of being
    /// observed within its arm. The assignment of a non-responder is never dropped from
    /// its arm's denominator.
    InverseProbabilityOfObservation,
    /// Panel designs: drop every unit that has a missing value in any period, so the
    /// remaining panel is balanced. Whole units are dropped, never single unit-periods.
    BalancedPanel,
    /// Panel designs: each contrast uses the units observed in the periods that contrast
    /// needs. Unit and period identities are preserved; nothing is relabelled.
    AvailableCase,
    /// Covariance-based discovery: every pair of variables uses the rows observed on both,
    /// and the pairwise covariance is repaired to positive semidefinite before scoring.
    /// Valid when missingness is completely at random; the repair is reported.
    PairwiseAvailable,
    /// Multiple imputation with Rubin's-rules pooling (frequentist) or a stacked posterior
    /// (Bayesian). Every analysed row is kept: the missing cells are drawn `imputations`
    /// times from a proper imputation model, the route's estimator is fitted on every
    /// completed table, and the fits are pooled as one uncertainty path. Valid when
    /// missingness is at random given the variables of the imputation model and that model
    /// is correctly specified; the specification is part of the claim identity.
    MultipleImputation(ImputationSpec),
}

/// The imputation model of [`MissingPolicy::MultipleImputation`].
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Default)]
#[non_exhaustive]
pub enum ImputationModel {
    /// Chained equations (fully conditional specification): each incomplete variable is
    /// drawn from a Bayesian-bootstrap fit of its conditional model on every other variable
    /// of the imputation model, a normal linear regression for a continuous variable and a
    /// logistic regression for a binary (0/1) one, cycling until the declared iterations.
    #[default]
    ChainedEquations,
    /// Joint multivariate normal by data augmentation: the mean and covariance are drawn
    /// from their posterior given the completed data (Jeffreys prior, or a Bayesian
    /// bootstrap of whole sampling units under a clustered design) and the missing cells
    /// from their conditional normal law. Refuses an incomplete binary variable.
    JointNormal,
}

impl ImputationModel {
    /// Every model, in documentation order.
    pub const ALL: [Self; 2] = [Self::ChainedEquations, Self::JointNormal];

    /// Stable lower-case name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::ChainedEquations => "chained_equations",
            Self::JointNormal => "joint_normal",
        }
    }

    /// The model named by `name`, if any.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|model| model.name() == name)
    }
}

/// The missingness mechanism a caller declares for multiple imputation.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Default)]
#[non_exhaustive]
pub enum MissingnessAssumption {
    /// Missing at random given the variables of the imputation model.
    #[default]
    MissingAtRandom,
    /// Missing not at random: the probability of a missing value depends on the value
    /// itself. Imputation under a missing-at-random model is not valid; a declaration of
    /// it is refused and names a sensitivity analysis as the next attempt.
    MissingNotAtRandom,
}

impl MissingnessAssumption {
    /// Every assumption, in documentation order.
    pub const ALL: [Self; 2] = [Self::MissingAtRandom, Self::MissingNotAtRandom];

    /// Stable lower-case name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::MissingAtRandom => "missing_at_random",
            Self::MissingNotAtRandom => "missing_not_at_random",
        }
    }

    /// The assumption named by `name`, if any.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|assumption| assumption.name() == name)
    }
}

/// The declared inputs of a multiple-imputation analysis. Every field enters the claim
/// identity: two analyses that differ in any of them are different claims.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct ImputationSpec {
    /// Imputation model.
    pub model: ImputationModel,
    /// Number of completed tables (`m`), at least [`ImputationSpec::MIN_IMPUTATIONS`].
    pub imputations: u32,
    /// Gibbs / chained-equation sweeps before each completed table is taken.
    pub iterations: u32,
    /// Seed of the deterministic random streams; imputation `j` opens its own keyed stream.
    pub seed: u64,
    /// Declared missingness mechanism.
    pub assumption: MissingnessAssumption,
    /// Bootstrap replicates run inside every completed table by an estimator that publishes
    /// no analytic variance (a static mediation decomposition), as that table's
    /// within-imputation variance. Zero asks for none, and such an estimator is then refused
    /// for want of a variance to pool. Estimators with an analytic variance ignore it.
    pub bootstrap_replicates: u32,
}

impl ImputationSpec {
    /// Fewest imputations Rubin's rules accept (a between-imputation variance needs two).
    pub const MIN_IMPUTATIONS: u32 = 2;
    /// Default number of imputations.
    pub const DEFAULT_IMPUTATIONS: u32 = 20;
    /// Default number of sweeps.
    pub const DEFAULT_ITERATIONS: u32 = 10;

    /// Chained equations, 20 imputations, 10 sweeps, seed 0, missing at random.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            model: ImputationModel::ChainedEquations,
            imputations: Self::DEFAULT_IMPUTATIONS,
            iterations: Self::DEFAULT_ITERATIONS,
            seed: 0,
            assumption: MissingnessAssumption::MissingAtRandom,
            bootstrap_replicates: 0,
        }
    }

    /// With imputation model `model`.
    #[must_use]
    pub const fn with_model(mut self, model: ImputationModel) -> Self {
        self.model = model;
        self
    }

    /// With `imputations` completed tables.
    #[must_use]
    pub const fn with_imputations(mut self, imputations: u32) -> Self {
        self.imputations = imputations;
        self
    }

    /// With `iterations` sweeps per imputation.
    #[must_use]
    pub const fn with_iterations(mut self, iterations: u32) -> Self {
        self.iterations = iterations;
        self
    }

    /// With seed `seed`.
    #[must_use]
    pub const fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// With `replicates` bootstrap replicates inside each completed table, for an estimator
    /// that publishes no analytic variance.
    #[must_use]
    pub const fn with_bootstrap_replicates(mut self, replicates: u32) -> Self {
        self.bootstrap_replicates = replicates;
        self
    }

    /// With the declared missingness mechanism.
    #[must_use]
    pub const fn with_assumption(mut self, assumption: MissingnessAssumption) -> Self {
        self.assumption = assumption;
        self
    }

    /// Why the counts are out of range, if they are.
    #[must_use]
    pub fn range_problem(&self) -> Option<String> {
        // The counts have no upper bound of their own: how much work a declaration may cost
        // is the execution's budget and cancellation to decide, not a constant here.
        if self.imputations < Self::MIN_IMPUTATIONS {
            return Some(format!(
                "imputations must be at least {}, got {}",
                Self::MIN_IMPUTATIONS,
                self.imputations
            ));
        }
        if self.iterations == 0 {
            return Some(format!("iterations must be at least 1, got {}", self.iterations));
        }
        None
    }
}

impl Default for ImputationSpec {
    fn default() -> Self {
        Self::new()
    }
}

impl MissingPolicy {
    /// Stable lower-case name used in receipts, identities and the Python option.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::CompleteCase => "complete_case",
            Self::ErrorOnMissing => "error",
            Self::InverseProbabilityOfObservation => "inverse_probability",
            Self::BalancedPanel => "balanced_panel",
            Self::AvailableCase => "available_case",
            Self::PairwiseAvailable => "pairwise_available",
            Self::MultipleImputation(_) => "multiple_imputation",
        }
    }

    /// The imputation specification, for [`Self::MultipleImputation`].
    #[must_use]
    pub const fn imputation(self) -> Option<ImputationSpec> {
        match self {
            Self::MultipleImputation(spec) => Some(spec),
            _ => None,
        }
    }

    /// The policy named by `name`, if any. `multiple_imputation` names the default
    /// [`ImputationSpec`].
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        [
            Self::CompleteCase,
            Self::ErrorOnMissing,
            Self::InverseProbabilityOfObservation,
            Self::BalancedPanel,
            Self::AvailableCase,
            Self::PairwiseAvailable,
            Self::MultipleImputation(ImputationSpec::new()),
        ]
        .into_iter()
        .find(|policy| policy.name() == name)
    }

    /// Every policy name, in documentation order.
    pub const NAMES: [&'static str; 7] = [
        "error",
        "complete_case",
        "inverse_probability",
        "balanced_panel",
        "available_case",
        "pairwise_available",
        "multiple_imputation",
    ];
}

/// A caller's declaration of how missing values are handled: the policy and, for
/// [`MissingPolicy::InverseProbabilityOfObservation`], the covariates the observation
/// probability may depend on; for [`MissingPolicy::MultipleImputation`], the auxiliary
/// variables the imputation model reads beside the analysed ones.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct MissingDeclaration {
    /// Declared policy.
    pub policy: MissingPolicy,
    /// Covariates of the observation model; empty declares missing completely at random
    /// within arm.
    pub covariates: Arc<[VariableId]>,
}

impl MissingDeclaration {
    /// A declaration with no covariates.
    #[must_use]
    pub fn new(policy: MissingPolicy) -> Self {
        Self { policy, covariates: Arc::from([]) }
    }

    /// Declare the covariates the observation probability depends on.
    #[must_use]
    pub fn with_covariates(mut self, covariates: impl Into<Arc<[VariableId]>>) -> Self {
        self.covariates = covariates.into();
        self
    }
}

/// How the optional analysis mask is applied.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Default)]
#[non_exhaustive]
pub enum MaskPolicy {
    /// Intersect the analysis mask with complete-case / validity filtering.
    #[default]
    Honor,
    /// Ignore the analysis mask (still apply missingness policy).
    Ignore,
}
