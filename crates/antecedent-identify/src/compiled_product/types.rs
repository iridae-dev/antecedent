//! Outcomes, refusals, obligations and the construction parts of a compiled product.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

pub(super) const IDENTITY_TAG: &str = "antecedent.identify.compiled_product.v1";
pub(super) const IDENTITY_FORMAT: u64 = 1;
pub(super) const TARGET_BOUND_IDENTITY_FORMAT: u64 = 2;

/// Outcome of an identification or support decision.
///
/// The six outcomes are distinct claims. [`Self::NumericallyInfeasible`] is
/// not [`Self::Unidentified`], and [`Self::TimedOut`] is not a scientific
/// negative.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum IdentificationOutcome {
    /// A functional or set was identified under the recorded assumptions.
    Identified,
    /// The search finished and did not identify a functional or set.
    Unidentified,
    /// The question is outside what this strategy can decide.
    Unsupported,
    /// The search stopped because its work budget ran out.
    TimedOut,
    /// A numerical check failed. This is not an identification verdict.
    NumericallyInfeasible,
    /// The procedure has no calibration for the claim it would report.
    Uncalibrated,
}

/// What kind of statement an outcome is allowed to make.
///
/// [`Self::ModelConditional`] and [`Self::UnsupportedExtrapolation`] are not
/// [`Self::IdentifiedClaim`].
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum IdentificationClaimClass {
    /// Observationally identified under the recorded assumptions.
    IdentifiedClaim,
    /// Evaluated inside a model. Not an identified claim.
    ModelConditional,
    /// Extrapolated off the supported design. Not an identified claim.
    UnsupportedExtrapolation,
}

impl IdentificationClaimClass {
    /// Whether this class is an identified claim.
    #[must_use]
    pub const fn is_identified_claim(self) -> bool {
        matches!(self, Self::IdentifiedClaim)
    }
}

/// Why a product was refused: its outcome, claim class, and retained result disagree.
///
/// The refusals keep a stored number, a timeout, or a search failure from being retained
/// beside an identified functional.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Error)]
#[non_exhaustive]
pub enum ProductRefusal {
    /// The outcome is [`IdentificationOutcome::Identified`] but no estimand is retained.
    #[error("an identified product retains no estimand")]
    IdentifiedWithoutEstimand,
    /// The outcome is [`IdentificationOutcome::Identified`] but an assumption it relies on
    /// is contradicted.
    #[error("an identified product relies on contradicted assumption {index}")]
    IdentifiedWithContradictedAssumption {
        /// Position of the contradicted assumption in the retained set.
        index: usize,
    },
    /// The outcome is [`IdentificationOutcome::Identified`] but an unresolved obligation remains.
    ///
    /// A declared assumption is a premise, not an unresolved obligation, and does not raise this.
    #[error("an identified outcome carries unresolved obligation {index}")]
    IdentifiedWithUnresolvedObligation {
        /// Position of the unresolved obligation in the list that still carries it.
        index: usize,
    },
    /// [`IdentificationOutcome::Unidentified`] states the identified-claim class.
    #[error("an unidentified outcome cannot state the identified-claim class")]
    UnidentifiedWithIdentifiedClaim,
    /// A world obligation does not name its world.
    #[error("a world obligation does not name a world")]
    WorldObligationMissingWorld,
    /// The assembled functional names a world. A world obligation carries that name.
    #[error("the assembled functional names world {world:?}")]
    AssembledObligationNamesWorld {
        /// World named on the assembled functional.
        world: ScientificWorldId,
    },
    /// The outcome is [`IdentificationOutcome::Identified`] but the retained result's status
    /// is not an identified status.
    #[error("an identified product retains a result with status {status:?}")]
    IdentifiedFromUnidentifiedResult {
        /// Status of the retained result.
        status: IdentificationStatus,
    },
    /// A target world has no retained identification/support obligation.
    #[error("target world {world:?} has no retained obligation")]
    MissingWorldObligation {
        /// Required world identifier.
        world: ScientificWorldId,
    },
    /// One target world has multiple obligations, so its authority is ambiguous.
    #[error("world {world:?} has duplicate retained obligations")]
    DuplicateWorldObligation {
        /// Repeated world identifier.
        world: ScientificWorldId,
    },
    /// An obligation names a world outside the target.
    #[error("retained obligation names non-target world {world:?}")]
    UnexpectedWorldObligation {
        /// Foreign world identifier.
        world: ScientificWorldId,
    },
    /// A contrast or surface omits its assembled-functional check.
    #[error("the joint target has no assembled-functional obligation")]
    MissingAssembledObligation,
    /// An identified product or assembled functional relies on an unsupported world.
    #[error("an identified joint claim relies on non-identified world {world:?}")]
    IdentifiedWithoutWorldSupport {
        /// World whose outcome or claim class does not license identification.
        world: ScientificWorldId,
    },
    /// The product claims identification beyond its assembled-functional outcome.
    #[error("an identified product has no identified assembled functional")]
    IdentifiedWithoutAssembledSupport,
    /// Mediation has separate cross-world premises rather than a generic world collection.
    #[error("mediation requires theorem-scoped support validation")]
    MediationRequiresTheoremScopedSupport,
    /// A future target form requires its own support rules.
    #[error("the target requires specialized support validation")]
    TargetRequiresSpecializedSupport,
    /// Legacy retention did not bind this product to scientific target content.
    #[error("retained identification is not bound to a scientific target")]
    UnboundScientificTarget,
    /// The product was compiled for different scientific content, even if world labels match.
    #[error("retained identification belongs to scientific target {expected}, not {actual}")]
    TargetIdentityMismatch {
        /// Scientific target retained at compilation.
        expected: SemanticDigest,
        /// Target offered by the consumer.
        actual: SemanticDigest,
    },
    /// A search failure, timeout, numerical failure, or uncalibrated outcome retains an
    /// estimand.
    #[error("outcome {outcome:?} cannot retain an identified estimand")]
    EstimandBesideNonIdentification {
        /// The outcome that cannot carry an estimand.
        outcome: IdentificationOutcome,
    },
    /// A failure outcome states the identified-claim class.
    #[error("outcome {outcome:?} cannot state the identified-claim class")]
    IdentifiedClaimClassBesideFailure {
        /// The outcome that cannot state that class.
        outcome: IdentificationOutcome,
    },
}

/// Outcomes that are neither an identification nor a finished negative.
pub(super) const fn is_failure_outcome(outcome: IdentificationOutcome) -> bool {
    matches!(
        outcome,
        IdentificationOutcome::Unsupported
            | IdentificationOutcome::TimedOut
            | IdentificationOutcome::NumericallyInfeasible
            | IdentificationOutcome::Uncalibrated
    )
}

pub(super) const fn check_claim_class(
    outcome: IdentificationOutcome,
    claim_class: IdentificationClaimClass,
) -> Result<(), ProductRefusal> {
    if matches!(outcome, IdentificationOutcome::Unidentified) && claim_class.is_identified_claim() {
        return Err(ProductRefusal::UnidentifiedWithIdentifiedClaim);
    }
    if is_failure_outcome(outcome) && claim_class.is_identified_claim() {
        return Err(ProductRefusal::IdentifiedClaimClassBesideFailure { outcome });
    }
    Ok(())
}

/// Evidence factor the functional requires.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RequiredFactor {
    pub(super) name: Arc<str>,
    pub(super) variables: Arc<[VariableId]>,
}

impl RequiredFactor {
    /// Factor `name` over `variables`, in the caller's order.
    #[must_use]
    pub fn new(name: impl Into<Arc<str>>, variables: impl Into<Arc<[VariableId]>>) -> Self {
        Self { name: name.into(), variables: variables.into() }
    }

    /// Stable factor name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Variables the factor must supply.
    #[must_use]
    pub fn variables(&self) -> &[VariableId] {
        &self.variables
    }
}

/// Scientific gap the product still owes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UnresolvedObligation {
    pub(super) code: Arc<str>,
    pub(super) detail: Arc<str>,
}

impl UnresolvedObligation {
    /// Obligation `code` with human-readable `detail`.
    #[must_use]
    pub fn new(code: impl Into<Arc<str>>, detail: impl Into<Arc<str>>) -> Self {
        Self { code: code.into(), detail: detail.into() }
    }

    /// Stable code.
    #[must_use]
    pub fn code(&self) -> &str {
        &self.code
    }

    /// Human-readable detail.
    #[must_use]
    pub fn detail(&self) -> &str {
        &self.detail
    }
}

/// Identification and support obligation for one world or for the assembled functional.
///
/// [`Self::for_world`] names the intervention world. [`Self::assembled`] does not:
/// a contrast has one assembled obligation. Equality ignores [`Self::numerical_value`].
/// A finite value is feasibility, not an identified claim.
#[derive(Clone, Debug)]
pub struct IdentificationObligation {
    pub(super) world: Option<ScientificWorldId>,
    pub(super) outcome: IdentificationOutcome,
    pub(super) claim_class: IdentificationClaimClass,
    pub(super) unresolved: Vec<UnresolvedObligation>,
    pub(super) numerical_value: Option<f64>,
}

impl PartialEq for IdentificationObligation {
    fn eq(&self, other: &Self) -> bool {
        self.world == other.world
            && self.outcome == other.outcome
            && self.claim_class == other.claim_class
            && self.unresolved == other.unresolved
    }
}

impl Eq for IdentificationObligation {}

impl IdentificationObligation {
    /// Obligation for `world`.
    ///
    /// `numerical_value` is stored and does not select `outcome`.
    #[must_use]
    pub const fn for_world(
        world: ScientificWorldId,
        outcome: IdentificationOutcome,
        claim_class: IdentificationClaimClass,
        unresolved: Vec<UnresolvedObligation>,
        numerical_value: Option<f64>,
    ) -> Self {
        Self { world: Some(world), outcome, claim_class, unresolved, numerical_value }
    }

    /// Assembled-functional obligation. It names no world.
    ///
    /// `numerical_value` is stored and does not select `outcome`.
    #[must_use]
    pub const fn assembled(
        outcome: IdentificationOutcome,
        claim_class: IdentificationClaimClass,
        unresolved: Vec<UnresolvedObligation>,
        numerical_value: Option<f64>,
    ) -> Self {
        Self { world: None, outcome, claim_class, unresolved, numerical_value }
    }

    /// World this obligation names, or [`None`] for the assembled functional.
    #[must_use]
    pub const fn world(&self) -> Option<ScientificWorldId> {
        self.world
    }

    /// Identification outcome.
    #[must_use]
    pub const fn outcome(&self) -> IdentificationOutcome {
        self.outcome
    }

    /// Claim class.
    #[must_use]
    pub const fn claim_class(&self) -> IdentificationClaimClass {
        self.claim_class
    }

    /// Support gaps still open.
    #[must_use]
    pub fn unresolved(&self) -> &[UnresolvedObligation] {
        &self.unresolved
    }

    /// Stored numerical value, when a check produced one.
    #[must_use]
    pub const fn numerical_value(&self) -> Option<f64> {
        self.numerical_value
    }

    /// Whether `outcome` and `claim_class` are an identified claim.
    ///
    /// [`Self::numerical_value`] is not consulted.
    #[must_use]
    pub const fn is_identified_claim(&self) -> bool {
        states_identified_claim(self.outcome, self.claim_class)
    }

    /// Whether a finite value is stored and the outcome is not numerical infeasibility.
    #[must_use]
    pub const fn is_numerically_feasible(&self) -> bool {
        numerically_feasible(self.outcome, self.numerical_value)
    }
}

/// Fields the caller supplies on top of a lowered [`IdentificationResult`].
///
/// Factors and unresolved obligations are not on the result. The outcome and
/// claim class are explicit so a stored number cannot rewrite them.
#[derive(Clone, Debug)]
pub struct CompiledProductParts {
    /// Outcome of this product.
    pub outcome: IdentificationOutcome,
    /// Claim class of this product.
    pub claim_class: IdentificationClaimClass,
    /// Factors the functional requires.
    pub required_factors: Vec<RequiredFactor>,
    /// Obligations still open for the product as a whole.
    pub unresolved_obligations: Vec<UnresolvedObligation>,
    /// Numerical value stored beside the outcome. Not an identified claim.
    pub numerical_value: Option<f64>,
    /// Per-world identification and support obligations. Each one names its world.
    pub world_obligations: Vec<IdentificationObligation>,
    /// The one assembled-functional obligation, when the target is a contrast.
    pub assembled: Option<IdentificationObligation>,
}

impl CompiledProductParts {
    /// Outcome and claim class, with no factors, obligations, worlds, or number.
    #[must_use]
    pub const fn new(
        outcome: IdentificationOutcome,
        claim_class: IdentificationClaimClass,
    ) -> Self {
        Self {
            outcome,
            claim_class,
            required_factors: Vec::new(),
            unresolved_obligations: Vec::new(),
            numerical_value: None,
            world_obligations: Vec::new(),
            assembled: None,
        }
    }
}
