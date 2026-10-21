//! Identification product retained when a result is lowered.
//!
//! [`IdentificationResult`] is the strategy output. This record is what the
//! compiler keeps after that result is lowered: the functional or identified
//! set (estimands and the arena that owns their expressions), the derivation,
//! the assumptions, the required factors, the unresolved obligations, and the
//! bounded-search diagnostics. Search truncation stays an execution fact.
//! A finite number stored beside a result is numerical feasibility, not
//! identification.
//!
//! [`PartialEq`] and [`CompiledIdentificationProduct::identity`] cover those
//! scientific fields. They omit bounded-search diagnostics and stored
//! numerical values, so two runs that differ only by budget or by a feasible
//! number are the same identification claim.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;
use std::sync::Arc;

use thiserror::Error;

use antecedent_core::{
    Assumption, AssumptionRecord, AssumptionScope, AssumptionSet, AssumptionSource,
    AssumptionStatus, CanonicalWriter, Diagnostic, IdentificationStatus, ParametricAssumption,
    PriorAssumption, ScientificTarget, ScientificWorldId, SemanticDigest, TargetForm, Value,
    VariableId,
};
use antecedent_expr::{
    CausalExprArena, ContrastOp, DomainRef, ExprId, ExprListId, ExprNode, IdentifiedEstimand,
    InterventionSetId, PopulationKeyId, RdDesignParams, VarSetId,
};

use crate::result::IdentificationResult;

const IDENTITY_TAG: &str = "antecedent.identify.compiled_product.v1";
const IDENTITY_FORMAT: u64 = 1;
const TARGET_BOUND_IDENTITY_FORMAT: u64 = 2;

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
const fn is_failure_outcome(outcome: IdentificationOutcome) -> bool {
    matches!(
        outcome,
        IdentificationOutcome::Unsupported
            | IdentificationOutcome::TimedOut
            | IdentificationOutcome::NumericallyInfeasible
            | IdentificationOutcome::Uncalibrated
    )
}

const fn check_claim_class(
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
    name: Arc<str>,
    variables: Arc<[VariableId]>,
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
    code: Arc<str>,
    detail: Arc<str>,
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
    world: Option<ScientificWorldId>,
    outcome: IdentificationOutcome,
    claim_class: IdentificationClaimClass,
    unresolved: Vec<UnresolvedObligation>,
    numerical_value: Option<f64>,
}

/// Obligation for one intervention world.
///
/// This is [`IdentificationObligation`]. The name remains for existing imports.
pub type WorldObligation = IdentificationObligation;

/// Obligation for the assembled functional.
///
/// This is [`IdentificationObligation`]. The name remains for existing imports.
pub type AssembledFunctionalObligation = IdentificationObligation;

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

/// Compiled identification product.
///
/// Lowering calls [`Self::retain`] so the functional or identified set, the
/// derivation, the assumptions, and the bounded-search diagnostics are copied
/// out of the [`IdentificationResult`] before that result is dropped.
#[derive(Clone, Debug)]
#[must_use = "lowering must retain this identification product"]
#[non_exhaustive]
pub struct CompiledIdentificationProduct {
    estimands: Vec<IdentifiedEstimand>,
    arena: CausalExprArena,
    derivation: crate::result::DerivationTrace,
    assumptions: AssumptionSet,
    required_factors: Vec<RequiredFactor>,
    unresolved_obligations: Vec<UnresolvedObligation>,
    bounded_search: Vec<Diagnostic>,
    outcome: IdentificationOutcome,
    claim_class: IdentificationClaimClass,
    numerical_value: Option<f64>,
    world_obligations: Vec<IdentificationObligation>,
    assembled: Option<IdentificationObligation>,
    target_identity: Option<SemanticDigest>,
}

impl PartialEq for CompiledIdentificationProduct {
    fn eq(&self, other: &Self) -> bool {
        same_estimands(&self.estimands, &other.estimands)
            && self.arena == other.arena
            && self.derivation == other.derivation
            && self.assumptions == other.assumptions
            && self.required_factors == other.required_factors
            && self.unresolved_obligations == other.unresolved_obligations
            && self.outcome == other.outcome
            && self.claim_class == other.claim_class
            && self.world_obligations == other.world_obligations
            && self.assembled == other.assembled
            && self.target_identity == other.target_identity
    }
}

impl Eq for CompiledIdentificationProduct {}

impl CompiledIdentificationProduct {
    /// Build a product from retained pieces.
    ///
    /// `bounded_search` holds execution diagnostics only. `parts.numerical_value`
    /// does not change `parts.outcome`.
    ///
    /// The outcome, claim class, and retained pieces must agree. An
    /// [`IdentificationOutcome::Identified`] product retains at least one estimand, relies on
    /// no [`AssumptionStatus::Contradicted`] assumption, and carries no unresolved obligation.
    /// Declared assumptions are premises and may be carried. Unsupported, timed-out, numerically
    /// infeasible, uncalibrated, and unidentified products do not state
    /// [`IdentificationClaimClass::IdentifiedClaim`], and neither do their world or assembled
    /// obligations. A failure outcome retains no estimand. An identified world or assembled
    /// obligation likewise carries no unresolved obligation, and a world obligation names its
    /// world.
    ///
    /// # Errors
    ///
    /// A [`ProductRefusal`] when the pieces disagree.
    #[must_use = "a refused product is not a retained identification"]
    pub fn new(
        estimands: Vec<IdentifiedEstimand>,
        arena: CausalExprArena,
        derivation: crate::result::DerivationTrace,
        assumptions: AssumptionSet,
        bounded_search: Vec<Diagnostic>,
        parts: CompiledProductParts,
    ) -> Result<Self, ProductRefusal> {
        check_coherence(&estimands, &assumptions, &parts)?;
        Ok(Self {
            estimands,
            arena,
            derivation,
            assumptions,
            required_factors: parts.required_factors,
            unresolved_obligations: parts.unresolved_obligations,
            bounded_search,
            outcome: parts.outcome,
            claim_class: parts.claim_class,
            numerical_value: parts.numerical_value,
            world_obligations: parts.world_obligations,
            assembled: parts.assembled,
            target_identity: None,
        })
    }

    /// Copy the fields lowering must not drop, then attach `parts`.
    ///
    /// Estimands, the expression arena, the derivation, and the assumptions are
    /// cloned. Diagnostics that [`crate::envelope::search_truncated`] treats as
    /// a bounded search are copied into [`Self::bounded_search_diagnostics`];
    /// other diagnostics are not rewritten into a scientific outcome.
    ///
    /// Beyond the checks of [`Self::new`], an [`IdentificationOutcome::Identified`] product
    /// requires `result.status` to be an identified status.
    ///
    /// # Errors
    ///
    /// A [`ProductRefusal`] when `parts` disagree with `result`.
    #[must_use = "a refused product is not a retained identification"]
    pub fn retain(
        result: &IdentificationResult,
        parts: CompiledProductParts,
    ) -> Result<Self, ProductRefusal> {
        if parts.outcome == IdentificationOutcome::Identified
            && !matches!(
                result.status,
                IdentificationStatus::NonparametricallyIdentified
                    | IdentificationStatus::IdentifiedUnderParametricRestrictions
                    | IdentificationStatus::IdentifiedUnderPriorRestrictions
            )
        {
            return Err(ProductRefusal::IdentifiedFromUnidentifiedResult { status: result.status });
        }
        let bounded_search = result
            .diagnostics
            .iter()
            .filter(|diagnostic| is_bounded_search(diagnostic))
            .cloned()
            .collect();
        Self::new(
            result.estimands.clone(),
            result.arena.clone(),
            result.derivation.clone(),
            result.required_assumptions.clone(),
            bounded_search,
            parts,
        )
    }

    /// Identified functionals and sets. Empty when nothing was identified.
    #[must_use]
    pub fn identified(&self) -> &[IdentifiedEstimand] {
        &self.estimands
    }

    /// Arena that owns the estimand expressions.
    #[must_use]
    pub const fn arena(&self) -> &CausalExprArena {
        &self.arena
    }

    /// Derivation retained from the result.
    #[must_use]
    pub const fn derivation(&self) -> &crate::result::DerivationTrace {
        &self.derivation
    }

    /// Assumptions the functional relies on.
    ///
    /// Declared entries are premises. They are not unresolved obligations.
    #[must_use]
    pub const fn assumptions(&self) -> &AssumptionSet {
        &self.assumptions
    }

    /// Declared assumptions. These are premises of the product.
    #[must_use]
    pub fn premises(&self) -> impl Iterator<Item = &AssumptionRecord> {
        self.assumptions
            .entries
            .iter()
            .filter(|record| matches!(record.status, AssumptionStatus::Declared))
    }

    /// Factors the functional requires.
    #[must_use]
    pub fn required_factors(&self) -> &[RequiredFactor] {
        &self.required_factors
    }

    /// Obligations still open for the product as a whole.
    #[must_use]
    pub fn unresolved_obligations(&self) -> &[UnresolvedObligation] {
        &self.unresolved_obligations
    }

    /// Bounded-search diagnostics copied off the result.
    ///
    /// These record how far a search ran. They are not a scientific negative
    /// and they are not part of [`Self::identity`].
    #[must_use]
    pub fn bounded_search_diagnostics(&self) -> &[Diagnostic] {
        &self.bounded_search
    }

    /// Outcome supplied when the product was built.
    #[must_use]
    pub const fn outcome(&self) -> IdentificationOutcome {
        self.outcome
    }

    /// Claim class supplied when the product was built.
    #[must_use]
    pub const fn claim_class(&self) -> IdentificationClaimClass {
        self.claim_class
    }

    /// Numerical value stored beside the outcome.
    #[must_use]
    pub const fn numerical_value(&self) -> Option<f64> {
        self.numerical_value
    }

    /// Retain identification and bind its complete support structure to a scientific target.
    ///
    /// The compiler supplies the target corresponding to `result.query` and outcomes from
    /// its identification/support procedures. This method checks their retained structure;
    /// it does not infer support from caller declarations or numerical feasibility. Future
    /// consumers cannot reuse this product for different interventions, population, units,
    /// conditioning, horizon or assumptions merely by keeping the same world labels.
    ///
    /// # Errors
    ///
    /// A [`ProductRefusal`] if retention fails or the world/assembled checks do not cover
    /// `target`. Mediation requires its theorem-specific path.
    pub fn retain_for_target(
        result: &IdentificationResult,
        parts: CompiledProductParts,
        target: &ScientificTarget,
    ) -> Result<Self, ProductRefusal> {
        let mut product = Self::retain(result, parts)?;
        product.target_identity = Some(target.identity());
        product.validate_target(target)?;
        Ok(product)
    }

    /// Scientific target bound by [`Self::retain_for_target`].
    ///
    /// Legacy products retained without a target return `None`.
    #[must_use]
    pub const fn target_identity(&self) -> Option<SemanticDigest> {
        self.target_identity
    }

    /// Per-world obligations. Empty for a target that names no worlds.
    #[must_use]
    pub fn world_obligations(&self) -> &[IdentificationObligation] {
        &self.world_obligations
    }

    /// The assembled-functional obligation, when the target is a contrast.
    #[must_use]
    pub const fn assembled_obligation(&self) -> Option<&IdentificationObligation> {
        self.assembled.as_ref()
    }

    /// Validate the retained world and assembled obligations against a scientific target.
    ///
    /// Every target world must appear exactly once, without foreign obligations. Contrasts
    /// and dose-by-time surfaces require an assembled-functional obligation. Neither the
    /// product nor its assembled functional may claim identification beyond any required
    /// world. Numerical values never discharge these checks. Declared scientific assumptions
    /// remain explicit premises; they are not automatically treated as failed support.
    ///
    /// This validates the retained support structure, not the statistical evidence itself:
    /// the compiler must obtain the outcomes from its identification and support procedures.
    /// Mediation must use its own theorem-scoped validation rather than this generic check.
    ///
    /// # Errors
    ///
    /// A [`ProductRefusal`] for an unbound or different target, missing, duplicate or
    /// foreign worlds, omitted assembled obligations, unsupported identification claims,
    /// or a mediation target. Bind legacy products with [`Self::retain_for_target`].
    pub fn validate_target(&self, target: &ScientificTarget) -> Result<(), ProductRefusal> {
        let expected = self.target_identity.ok_or(ProductRefusal::UnboundScientificTarget)?;
        let actual = target.identity();
        if expected != actual {
            return Err(ProductRefusal::TargetIdentityMismatch { expected, actual });
        }
        let (worlds, needs_assembled) = match target.form() {
            TargetForm::World(world) => (std::slice::from_ref(world), false),
            TargetForm::Worlds(worlds) => (worlds.worlds(), false),
            TargetForm::Contrast(contrast) => (contrast.worlds().worlds(), true),
            TargetForm::DoseTime(surface) => (surface.worlds().worlds(), true),
            TargetForm::Mediation(_) => {
                return Err(ProductRefusal::MediationRequiresTheoremScopedSupport);
            }
            _ => return Err(ProductRefusal::TargetRequiresSpecializedSupport),
        };
        let target_worlds: BTreeSet<_> =
            worlds.iter().map(antecedent_core::InterventionWorld::id).collect();
        let mut retained_worlds = BTreeSet::new();
        for obligation in &self.world_obligations {
            let Some(world) = obligation.world else {
                return Err(ProductRefusal::WorldObligationMissingWorld);
            };
            if !target_worlds.contains(&world) {
                return Err(ProductRefusal::UnexpectedWorldObligation { world });
            }
            if !retained_worlds.insert(world) {
                return Err(ProductRefusal::DuplicateWorldObligation { world });
            }
        }
        if let Some(&world) = target_worlds.difference(&retained_worlds).next() {
            return Err(ProductRefusal::MissingWorldObligation { world });
        }
        if needs_assembled && self.assembled.is_none() {
            return Err(ProductRefusal::MissingAssembledObligation);
        }
        let identified_product = states_identified_claim(self.outcome, self.claim_class);
        let identified_assembled = self.assembled_is_identified_claim();
        if identified_product || identified_assembled {
            if let Some(world) =
                self.world_obligations.iter().find(|world| !world.is_identified_claim())
            {
                let Some(world_id) = world.world else {
                    return Err(ProductRefusal::WorldObligationMissingWorld);
                };
                return Err(ProductRefusal::IdentifiedWithoutWorldSupport { world: world_id });
            }
        }
        if identified_product && needs_assembled && !identified_assembled {
            return Err(ProductRefusal::IdentifiedWithoutAssembledSupport);
        }
        Ok(())
    }

    /// Whether this product states an identified claim.
    ///
    /// The product's numerical value is ignored. When any world obligation or
    /// an assembled obligation is recorded, every world and the assembled
    /// functional must themselves be identified claims: feasibility of the
    /// worlds does not identify the contrast.
    #[must_use]
    pub fn is_identified_claim(&self) -> bool {
        let own = states_identified_claim(self.outcome, self.claim_class);
        if self.world_obligations.is_empty() && self.assembled.is_none() {
            return own;
        }
        own && self.world_obligations.iter().all(IdentificationObligation::is_identified_claim)
            && self.assembled.as_ref().is_some_and(IdentificationObligation::is_identified_claim)
    }

    /// Whether the assembled functional is an identified claim.
    ///
    /// `false` when no assembled obligation was recorded. World-level numerical
    /// feasibility is not an input.
    #[must_use]
    pub const fn assembled_is_identified_claim(&self) -> bool {
        match &self.assembled {
            Some(obligation) => obligation.is_identified_claim(),
            None => false,
        }
    }

    /// Whether a finite value is stored and the outcome is not numerical infeasibility.
    #[must_use]
    pub const fn is_numerically_feasible(&self) -> bool {
        numerically_feasible(self.outcome, self.numerical_value)
    }

    /// Domain-separated digest of the scientific fields.
    ///
    /// The digest uses [`CanonicalWriter`]. Bounded-search diagnostics and
    /// numerical values are retained on the product and omitted here, matching
    /// [`PartialEq`].
    #[must_use]
    pub fn identity(&self) -> SemanticDigest {
        let mut writer = CanonicalWriter::new(IDENTITY_TAG);
        if let Some(target) = self.target_identity {
            writer.word(TARGET_BOUND_IDENTITY_FORMAT).bytes(target.as_bytes());
        } else {
            // Preserve the identity of legacy products retained without a target.
            writer.word(IDENTITY_FORMAT);
        }
        write_estimands(&mut writer, &self.estimands);
        write_arena(&mut writer, &self.arena);
        write_derivation(&mut writer, &self.derivation);
        write_assumptions(&mut writer, &self.assumptions);
        write_factors(&mut writer, &self.required_factors);
        write_obligations(&mut writer, &self.unresolved_obligations);
        writer.word(outcome_code(self.outcome));
        writer.word(claim_code(self.claim_class));
        write_worlds(&mut writer, &self.world_obligations);
        write_assembled(&mut writer, self.assembled.as_ref());
        SemanticDigest::from_bytes(writer.finish())
    }
}

fn check_coherence(
    estimands: &[IdentifiedEstimand],
    assumptions: &AssumptionSet,
    parts: &CompiledProductParts,
) -> Result<(), ProductRefusal> {
    check_claim_class(parts.outcome, parts.claim_class)?;
    for world in &parts.world_obligations {
        check_claim_class(world.outcome, world.claim_class)?;
    }
    if let Some(assembled) = &parts.assembled {
        check_claim_class(assembled.outcome, assembled.claim_class)?;
    }
    if is_failure_outcome(parts.outcome) && !estimands.is_empty() {
        return Err(ProductRefusal::EstimandBesideNonIdentification { outcome: parts.outcome });
    }
    check_obligation_scope(parts)?;
    if parts.outcome == IdentificationOutcome::Identified {
        if estimands.is_empty() {
            return Err(ProductRefusal::IdentifiedWithoutEstimand);
        }
        if let Some(index) = assumptions
            .entries
            .iter()
            .position(|record| matches!(record.status, AssumptionStatus::Contradicted))
        {
            return Err(ProductRefusal::IdentifiedWithContradictedAssumption { index });
        }
        if let Some(index) = first_unresolved(&parts.unresolved_obligations) {
            return Err(ProductRefusal::IdentifiedWithUnresolvedObligation { index });
        }
    }
    for world in &parts.world_obligations {
        if world.outcome == IdentificationOutcome::Identified {
            if let Some(index) = first_unresolved(&world.unresolved) {
                return Err(ProductRefusal::IdentifiedWithUnresolvedObligation { index });
            }
        }
    }
    if let Some(assembled) = &parts.assembled {
        if assembled.outcome == IdentificationOutcome::Identified {
            if let Some(index) = first_unresolved(&assembled.unresolved) {
                return Err(ProductRefusal::IdentifiedWithUnresolvedObligation { index });
            }
        }
    }
    Ok(())
}

fn first_unresolved(obligations: &[UnresolvedObligation]) -> Option<usize> {
    if obligations.is_empty() { None } else { Some(0) }
}

fn check_obligation_scope(parts: &CompiledProductParts) -> Result<(), ProductRefusal> {
    for obligation in &parts.world_obligations {
        if obligation.world.is_none() {
            return Err(ProductRefusal::WorldObligationMissingWorld);
        }
    }
    if let Some(assembled) = &parts.assembled {
        if let Some(world) = assembled.world {
            return Err(ProductRefusal::AssembledObligationNamesWorld { world });
        }
    }
    Ok(())
}

const fn states_identified_claim(
    outcome: IdentificationOutcome,
    class: IdentificationClaimClass,
) -> bool {
    matches!(outcome, IdentificationOutcome::Identified)
        && matches!(class, IdentificationClaimClass::IdentifiedClaim)
}

const fn numerically_feasible(outcome: IdentificationOutcome, value: Option<f64>) -> bool {
    if matches!(outcome, IdentificationOutcome::NumericallyInfeasible) {
        return false;
    }
    match value {
        Some(number) => number.is_finite(),
        None => false,
    }
}

const fn outcome_code(outcome: IdentificationOutcome) -> u64 {
    match outcome {
        IdentificationOutcome::Identified => 1,
        IdentificationOutcome::Unidentified => 2,
        IdentificationOutcome::Unsupported => 3,
        IdentificationOutcome::TimedOut => 4,
        IdentificationOutcome::NumericallyInfeasible => 5,
        IdentificationOutcome::Uncalibrated => 6,
    }
}

const fn claim_code(class: IdentificationClaimClass) -> u64 {
    match class {
        IdentificationClaimClass::IdentifiedClaim => 1,
        IdentificationClaimClass::ModelConditional => 2,
        IdentificationClaimClass::UnsupportedExtrapolation => 3,
    }
}

fn is_bounded_search(diagnostic: &Diagnostic) -> bool {
    let code = diagnostic.code.as_ref();
    code == crate::generalized::CAPPED_COMPLETION_DIAGNOSTIC_CODE
        || code.ends_with(".search_bounded")
        || code == crate::temporal_mag::HISTORY_CAPPED
}

fn count(len: usize) -> u64 {
    u64::try_from(len).unwrap_or(u64::MAX)
}

fn nth_raw(index: usize) -> u32 {
    u32::try_from(index).unwrap_or(u32::MAX)
}

fn same_estimands(left: &[IdentifiedEstimand], right: &[IdentifiedEstimand]) -> bool {
    left.len() == right.len() && left.iter().zip(right).all(|(a, b)| same_estimand(a, b))
}

fn same_estimand(left: &IdentifiedEstimand, right: &IdentifiedEstimand) -> bool {
    left.method == right.method
        && left.adjustment_set == right.adjustment_set
        && left.instruments == right.instruments
        && left.mediators == right.mediators
        && left.functional == right.functional
        && same_rd(left.rd_design, right.rd_design)
}

const fn same_rd(left: Option<RdDesignParams>, right: Option<RdDesignParams>) -> bool {
    match (left, right) {
        (None, None) => true,
        (Some(a), Some(b)) => {
            a.running_variable.raw() == b.running_variable.raw()
                && a.cutoff.to_bits() == b.cutoff.to_bits()
                && a.bandwidth.to_bits() == b.bandwidth.to_bits()
        }
        (None, Some(_)) | (Some(_), None) => false,
    }
}

fn write_estimands(writer: &mut CanonicalWriter, estimands: &[IdentifiedEstimand]) {
    writer.word(count(estimands.len()));
    for estimand in estimands {
        writer.text(&estimand.method);
        write_vars(writer, &estimand.adjustment_set);
        write_vars(writer, &estimand.instruments);
        write_vars(writer, &estimand.mediators);
        writer.word(u64::from(estimand.functional.raw()));
        match estimand.rd_design {
            None => {
                writer.word(0);
            }
            Some(design) => {
                writer.word(1);
                writer.word(u64::from(design.running_variable.raw()));
                writer.word(design.cutoff.to_bits());
                writer.word(design.bandwidth.to_bits());
            }
        }
    }
}

fn write_vars(writer: &mut CanonicalWriter, variables: &[VariableId]) {
    writer.word(count(variables.len()));
    for variable in variables {
        writer.word(u64::from(variable.raw()));
    }
}

fn write_arena(writer: &mut CanonicalWriter, arena: &CausalExprArena) {
    writer.word(count(arena.len()));
    for index in 0..arena.len() {
        write_node(writer, arena.node(ExprId::from_raw(nth_raw(index))));
    }
    writer.word(count(arena.var_set_count()));
    for index in 0..arena.var_set_count() {
        write_vars(writer, arena.var_set(VarSetId::from_raw(nth_raw(index))));
    }
    writer.word(count(arena.intervention_set_count()));
    for index in 0..arena.intervention_set_count() {
        let assignments =
            arena.intervention_assignments(InterventionSetId::from_raw(nth_raw(index)));
        writer.word(count(assignments.len()));
        for assignment in assignments {
            writer.word(u64::from(assignment.variable.raw()));
            write_value(writer, &assignment.value);
        }
    }
    writer.word(count(arena.list_count()));
    for index in 0..arena.list_count() {
        let list = arena.list(ExprListId::from_raw(nth_raw(index)));
        writer.word(count(list.len()));
        for expr in list {
            writer.word(u64::from(expr.raw()));
        }
    }
    writer.word(count(arena.population_count()));
    for index in 0..arena.population_count() {
        writer.text(arena.population(PopulationKeyId::from_raw(nth_raw(index))));
    }
}

fn write_node(writer: &mut CanonicalWriter, node: &ExprNode) {
    match node {
        ExprNode::Distribution {
            variables,
            conditioned_on,
            intervention,
            domain,
            population,
            regime,
        } => {
            writer.word(1);
            writer.word(u64::from(variables.raw()));
            writer.word(u64::from(conditioned_on.raw()));
            writer.word(u64::from(intervention.raw()));
            write_domain(writer, *domain);
            writer.word(u64::from(population.raw()));
            writer.opt_word(regime.map(|id| u64::from(id.raw())));
        }
        ExprNode::Kernel { body, bound, population, regime } => {
            writer.word(2);
            writer.word(u64::from(body.raw()));
            writer.word(u64::from(bound.raw()));
            writer.word(u64::from(population.raw()));
            writer.opt_word(regime.map(|id| u64::from(id.raw())));
        }
        ExprNode::Product(list) => {
            writer.word(3);
            writer.word(u64::from(list.raw()));
        }
        ExprNode::SumOut { variables, expr } => {
            writer.word(4);
            writer.word(u64::from(variables.raw()));
            writer.word(u64::from(expr.raw()));
        }
        ExprNode::IntegralOut { variables, expr } => {
            writer.word(5);
            writer.word(u64::from(variables.raw()));
            writer.word(u64::from(expr.raw()));
        }
        ExprNode::Ratio { numerator, denominator } => {
            writer.word(6);
            writer.word(u64::from(numerator.raw()));
            writer.word(u64::from(denominator.raw()));
        }
        ExprNode::Expectation { function, distribution } => {
            writer.word(7);
            writer.word(u64::from(function.variable().raw()));
            writer.word(u64::from(distribution.raw()));
        }
        ExprNode::Contrast { left, right, op } => {
            writer.word(8);
            writer.word(u64::from(left.raw()));
            writer.word(u64::from(right.raw()));
            write_contrast_op(writer, *op);
        }
    }
}

fn write_domain(writer: &mut CanonicalWriter, domain: DomainRef) {
    match domain {
        DomainRef::Observational => {
            writer.word(1);
        }
        DomainRef::Interventional => {
            writer.word(2);
        }
    }
}

fn write_contrast_op(writer: &mut CanonicalWriter, op: ContrastOp) {
    match op {
        ContrastOp::Difference => {
            writer.word(1);
        }
    }
}

fn write_value(writer: &mut CanonicalWriter, value: &Value) {
    match value {
        Value::Float64(number) => {
            writer.word(1);
            writer.real(*number);
        }
        Value::Int64(number) => {
            writer.word(2);
            writer.word(u64::from_le_bytes(number.to_le_bytes()));
        }
        Value::Bool(bit) => {
            writer.word(3);
            writer.word(u64::from(u8::from(*bit)));
        }
        Value::Category(code) => {
            writer.word(4);
            writer.word(u64::from(*code));
        }
        Value::Label(label) => {
            writer.word(5);
            writer.text(label);
        }
    }
}

fn write_derivation(writer: &mut CanonicalWriter, derivation: &crate::result::DerivationTrace) {
    writer.word(count(derivation.steps.len()));
    for step in &derivation.steps {
        writer.text(&step.rule);
        writer.text(&step.detail);
    }
}

fn write_assumptions(writer: &mut CanonicalWriter, assumptions: &AssumptionSet) {
    writer.word(count(assumptions.len()));
    for record in &assumptions.entries {
        write_assumption_record(writer, record);
    }
}

fn write_assumption_record(writer: &mut CanonicalWriter, record: &AssumptionRecord) {
    write_assumption(writer, &record.assumption);
    write_assumption_source(writer, &record.source);
    write_assumption_scope(writer, &record.scope);
    writer.word(assumption_status_code(record.status));
}

fn write_assumption(writer: &mut CanonicalWriter, assumption: &Assumption) {
    match assumption {
        Assumption::CausalMarkov => {
            writer.word(1);
        }
        Assumption::Faithfulness => {
            writer.word(2);
        }
        Assumption::CausalSufficiency => {
            writer.word(3);
        }
        Assumption::Consistency => {
            writer.word(4);
        }
        Assumption::Positivity => {
            writer.word(5);
        }
        Assumption::NoInterference => {
            writer.word(6);
        }
        Assumption::Stationarity => {
            writer.word(7);
        }
        Assumption::PiecewiseStationarity => {
            writer.word(8);
        }
        Assumption::NoSelectionBias => {
            writer.word(9);
        }
        Assumption::ExclusionRestriction { instrument } => {
            writer.word(10);
            writer.word(u64::from(instrument.raw()));
        }
        Assumption::Monotonicity => {
            writer.word(11);
        }
        Assumption::ParametricRestriction(restriction) => {
            writer.word(12);
            write_parametric(writer, restriction);
        }
        Assumption::PriorRestriction(restriction) => {
            writer.word(13);
            write_prior(writer, restriction);
        }
        Assumption::Custom { id, description } => {
            writer.word(14);
            writer.text(id);
            writer.text(description);
        }
    }
}

fn write_parametric(writer: &mut CanonicalWriter, restriction: &ParametricAssumption) {
    writer.text(&restriction.id);
    writer.text(&restriction.description);
}

fn write_prior(writer: &mut CanonicalWriter, restriction: &PriorAssumption) {
    writer.text(&restriction.id);
    writer.text(&restriction.description);
}

fn write_assumption_source(writer: &mut CanonicalWriter, source: &AssumptionSource) {
    match source {
        AssumptionSource::UserDeclared => {
            writer.word(1);
        }
        AssumptionSource::AlgorithmDefault { algorithm } => {
            writer.word(2);
            writer.text(algorithm);
        }
        AssumptionSource::Artifact => {
            writer.word(3);
        }
        AssumptionSource::Derived { from } => {
            writer.word(4);
            writer.text(from);
        }
    }
}

fn write_assumption_scope(writer: &mut CanonicalWriter, scope: &AssumptionScope) {
    match scope {
        AssumptionScope::Global => {
            writer.word(1);
        }
        AssumptionScope::Identification => {
            writer.word(2);
        }
        AssumptionScope::Estimation => {
            writer.word(3);
        }
        AssumptionScope::Discovery => {
            writer.word(4);
        }
        AssumptionScope::Variables { variables } => {
            writer.word(5);
            write_vars(writer, variables);
        }
    }
}

const fn assumption_status_code(status: AssumptionStatus) -> u64 {
    match status {
        AssumptionStatus::Declared => 1,
        AssumptionStatus::Supported => 2,
        AssumptionStatus::Contradicted => 3,
        AssumptionStatus::Untestable => 4,
    }
}

fn write_factors(writer: &mut CanonicalWriter, factors: &[RequiredFactor]) {
    writer.word(count(factors.len()));
    for factor in factors {
        writer.text(&factor.name);
        write_vars(writer, &factor.variables);
    }
}

fn write_obligations(writer: &mut CanonicalWriter, obligations: &[UnresolvedObligation]) {
    writer.word(count(obligations.len()));
    for obligation in obligations {
        writer.text(&obligation.code);
        writer.text(&obligation.detail);
    }
}

fn write_worlds(writer: &mut CanonicalWriter, worlds: &[IdentificationObligation]) {
    writer.word(count(worlds.len()));
    for world in worlds {
        let raw = world.world.map_or(0, |id| u64::from(id.raw()));
        writer.word(raw);
        writer.word(outcome_code(world.outcome));
        writer.word(claim_code(world.claim_class));
        write_obligations(writer, &world.unresolved);
    }
}

fn write_assembled(writer: &mut CanonicalWriter, assembled: Option<&IdentificationObligation>) {
    match assembled {
        None => {
            writer.word(0);
        }
        Some(obligation) => {
            writer.word(1);
            writer.word(outcome_code(obligation.outcome));
            writer.word(claim_code(obligation.claim_class));
            write_obligations(writer, &obligation.unresolved);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::IdentificationObligation;
    use antecedent_core::{
        Assumption, AssumptionRecord, AssumptionScope, AssumptionSet, AssumptionSource,
        AssumptionStatus, AverageEffectQuery, CausalQuery, Diagnostic, DiagnosticKind,
        DiagnosticSeverity, IdentificationStatus, ScientificWorldId, Value, VariableId,
    };
    use antecedent_expr::{CausalExprArena, DerivationMeta, ExprId, IdentifiedEstimand};

    use crate::result::{
        DerivationTrace, IdentificationPerformanceRecord, IdentificationResult,
        search_bounded_diagnostic,
    };

    use super::{
        AssembledFunctionalObligation, CompiledIdentificationProduct, CompiledProductParts,
        IdentificationClaimClass, IdentificationOutcome, ProductRefusal, RequiredFactor,
        UnresolvedObligation, WorldObligation, claim_code, outcome_code,
    };

    const OUTCOMES: [IdentificationOutcome; 6] = [
        IdentificationOutcome::Identified,
        IdentificationOutcome::Unidentified,
        IdentificationOutcome::Unsupported,
        IdentificationOutcome::TimedOut,
        IdentificationOutcome::NumericallyInfeasible,
        IdentificationOutcome::Uncalibrated,
    ];

    fn query() -> CausalQuery {
        CausalQuery::average_effect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
        ))
    }

    fn empty_product(
        outcome: IdentificationOutcome,
        claim_class: IdentificationClaimClass,
        numerical_value: Option<f64>,
    ) -> CompiledIdentificationProduct {
        let mut parts = CompiledProductParts::new(outcome, claim_class);
        parts.numerical_value = numerical_value;
        let estimands = if outcome == IdentificationOutcome::Identified {
            vec![estimand()]
        } else {
            Vec::new()
        };
        CompiledIdentificationProduct::new(
            estimands,
            CausalExprArena::new(),
            DerivationTrace::default(),
            AssumptionSet::new(),
            Vec::new(),
            parts,
        )
        .expect("coherent product")
    }

    fn estimand() -> IdentifiedEstimand {
        IdentifiedEstimand::backdoor(
            "backdoor.adjustment",
            Arc::from([VariableId::from_raw(2)]),
            ExprId::from_raw(0),
        )
    }

    fn interaction_target() -> antecedent_core::ScientificTarget {
        interaction_target_with(
            1.0,
            antecedent_core::TargetPopulation::AllObserved,
            VariableId::from_raw(2),
        )
    }

    fn interaction_target_with(
        active: f64,
        population: antecedent_core::TargetPopulation,
        outcome: VariableId,
    ) -> antecedent_core::ScientificTarget {
        use antecedent_core::{
            Conditioning, GraphFamily, InferenceSelection, Intervention, InterventionContrast,
            InterventionWorld, ModelAssumptions, ObservationUnit, OutcomeQuantity, QuantityUnits,
            ScientificTarget, TargetForm, TimeFrame, WorldAction, WorldBinding, WorldCollection,
        };
        let binding = WorldBinding::new(
            population,
            ObservationUnit::new("subject").unwrap(),
            OutcomeQuantity::mean(outcome, QuantityUnits::dimensionless()).unwrap(),
            Conditioning::unconditional(),
        )
        .unwrap();
        let levels = [(0.0, 0.0), (active, 0.0), (0.0, active), (active, active)];
        let worlds = WorldCollection::new(levels.into_iter().enumerate().map(|(index, (a, b))| {
            InterventionWorld::new(
                ScientificWorldId::from_raw(u32::try_from(index).unwrap()),
                WorldAction::at_decision([
                    Intervention::set(VariableId::from_raw(0), Value::f64(a)),
                    Intervention::set(VariableId::from_raw(1), Value::f64(b)),
                ])
                .unwrap(),
                index == 0,
                binding.clone(),
                TimeFrame::static_decision(),
            )
            .unwrap()
        }))
        .unwrap();
        let contrast = InterventionContrast::interaction(
            worlds,
            ScientificWorldId::from_raw(3),
            ScientificWorldId::from_raw(1),
            ScientificWorldId::from_raw(2),
            ScientificWorldId::from_raw(0),
        )
        .unwrap();
        ScientificTarget::new(
            TargetForm::Contrast(contrast),
            ModelAssumptions::new(GraphFamily::Dag, AssumptionSet::new()),
            InferenceSelection::Frequentist,
        )
    }

    fn supported_interaction_product() -> CompiledIdentificationProduct {
        let mut product = empty_product(
            IdentificationOutcome::Identified,
            IdentificationClaimClass::IdentifiedClaim,
            None,
        );
        product.world_obligations = (0..4)
            .map(|id| {
                IdentificationObligation::for_world(
                    ScientificWorldId::from_raw(id),
                    IdentificationOutcome::Identified,
                    IdentificationClaimClass::IdentifiedClaim,
                    Vec::new(),
                    None,
                )
            })
            .collect();
        product.assembled = Some(IdentificationObligation::assembled(
            IdentificationOutcome::Identified,
            IdentificationClaimClass::IdentifiedClaim,
            Vec::new(),
            None,
        ));
        product.target_identity = Some(interaction_target().identity());
        product
    }

    #[test]
    fn target_validation_requires_each_world_exactly_once_and_an_assembled_check() {
        let target = interaction_target();
        let product = supported_interaction_product();
        assert_eq!(product.validate_target(&target), Ok(()));
        let mut missing = product.clone();
        missing.world_obligations.pop();
        assert_eq!(
            missing.validate_target(&target),
            Err(ProductRefusal::MissingWorldObligation { world: ScientificWorldId::from_raw(3) })
        );
        let mut duplicate = product.clone();
        duplicate.world_obligations.push(product.world_obligations[0].clone());
        assert_eq!(
            duplicate.validate_target(&target),
            Err(ProductRefusal::DuplicateWorldObligation { world: ScientificWorldId::from_raw(0) })
        );
        let mut foreign = product.clone();
        foreign.world_obligations[0].world = Some(ScientificWorldId::from_raw(9));
        assert_eq!(
            foreign.validate_target(&target),
            Err(ProductRefusal::UnexpectedWorldObligation {
                world: ScientificWorldId::from_raw(9),
            })
        );
        let mut no_assembled = product;
        no_assembled.assembled = None;
        assert_eq!(
            no_assembled.validate_target(&target),
            Err(ProductRefusal::MissingAssembledObligation)
        );
    }

    #[test]
    fn target_validation_never_promotes_feasible_unsupported_or_conditional_worlds() {
        let target = interaction_target();
        for outcome in OUTCOMES {
            for class in [
                IdentificationClaimClass::ModelConditional,
                IdentificationClaimClass::UnsupportedExtrapolation,
            ] {
                let mut product = supported_interaction_product();
                let world = &mut product.world_obligations[2];
                world.outcome = outcome;
                world.claim_class = class;
                world.numerical_value = Some(4.0);
                assert_eq!(
                    product.validate_target(&target),
                    Err(ProductRefusal::IdentifiedWithoutWorldSupport {
                        world: ScientificWorldId::from_raw(2),
                    })
                );
                product.outcome = IdentificationOutcome::Unidentified;
                product.claim_class = IdentificationClaimClass::ModelConditional;
                assert!(product.assembled_is_identified_claim());
                assert!(product.validate_target(&target).is_err());
                product.assembled.as_mut().unwrap().claim_class =
                    IdentificationClaimClass::ModelConditional;
                assert_eq!(product.validate_target(&target), Ok(()));
                assert!(!product.is_identified_claim());
            }
        }
    }

    #[test]
    fn target_validation_requires_whole_functional_support_for_an_identified_product() {
        let target = interaction_target();
        let mut product = supported_interaction_product();
        let assembled = product.assembled.as_mut().unwrap();
        assembled.outcome = IdentificationOutcome::Unidentified;
        assembled.claim_class = IdentificationClaimClass::ModelConditional;
        assembled.numerical_value = Some(1.0);
        assert_eq!(
            product.validate_target(&target),
            Err(ProductRefusal::IdentifiedWithoutAssembledSupport)
        );
    }

    #[test]
    fn actual_joint_identification_is_bound_to_scientific_content_not_world_labels() {
        use antecedent_core::{
            GraphFamily, InferenceSelection, Intervention, ModelAssumptions, NodeRef,
            ResponseFunctional, ResponseQuery, ScientificTarget,
        };
        use antecedent_graph::Admg;

        let mut graph = Admg::empty();
        let a = graph.add_node(NodeRef::Static(VariableId::from_raw(0))).unwrap();
        let b = graph.add_node(NodeRef::Static(VariableId::from_raw(1))).unwrap();
        let y = graph.add_node(NodeRef::Static(VariableId::from_raw(2))).unwrap();
        graph.insert_directed(a, y).unwrap();
        graph.insert_directed(b, y).unwrap();
        let query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
            outcome: VariableId::from_raw(2),
            interventions: Arc::from([
                Intervention::set(VariableId::from_raw(0), Value::f64(1.0)),
                Intervention::set(VariableId::from_raw(1), Value::f64(1.0)),
            ]),
        });
        let result = crate::GeneralizedAdjustmentIdentifier::new()
            .identify_joint_admg_response(&graph, &query)
            .unwrap();
        assert_eq!(result.status, IdentificationStatus::NonparametricallyIdentified);
        let supported = supported_interaction_product();
        let mut parts = CompiledProductParts::new(supported.outcome, supported.claim_class);
        parts.world_obligations = supported.world_obligations;
        parts.assembled = supported.assembled;
        let target = interaction_target();
        let product =
            CompiledIdentificationProduct::retain_for_target(&result, parts.clone(), &target)
                .unwrap();
        assert_eq!(product.target_identity(), Some(target.identity()));
        assert_eq!(product.validate_target(&target), Ok(()));
        let bayesian = target.clone().with_inference(InferenceSelection::Bayesian);
        assert_eq!(product.validate_target(&bayesian), Ok(()));
        let changed = ScientificTarget::new(
            target.form().clone(),
            ModelAssumptions::new(GraphFamily::Admg, AssumptionSet::new()),
            InferenceSelection::Frequentist,
        );
        assert_eq!(
            product.validate_target(&changed),
            Err(ProductRefusal::TargetIdentityMismatch {
                expected: target.identity(),
                actual: changed.identity(),
            })
        );
        let rebound =
            CompiledIdentificationProduct::retain_for_target(&result, parts.clone(), &changed)
                .unwrap();
        assert_ne!(product, rebound);
        assert_ne!(product.identity(), rebound.identity());
        let legacy = CompiledIdentificationProduct::retain(&result, parts).unwrap();
        assert_eq!(legacy.target_identity(), None);
        assert_ne!(legacy.identity(), product.identity());
        assert_eq!(legacy.validate_target(&target), Err(ProductRefusal::UnboundScientificTarget));
        for changed in [
            interaction_target_with(
                2.0,
                antecedent_core::TargetPopulation::AllObserved,
                VariableId::from_raw(2),
            ),
            interaction_target_with(
                1.0,
                antecedent_core::TargetPopulation::Treated,
                VariableId::from_raw(2),
            ),
            interaction_target_with(
                1.0,
                antecedent_core::TargetPopulation::AllObserved,
                VariableId::from_raw(3),
            ),
        ] {
            assert_eq!(
                product.validate_target(&changed),
                Err(ProductRefusal::TargetIdentityMismatch {
                    expected: target.identity(),
                    actual: changed.identity(),
                })
            );
        }
    }

    #[test]
    fn temporal_surface_requires_every_scheduled_cell_and_its_whole_functional() {
        use antecedent_core::{
            DoseTimeSurface, GraphFamily, HistoryWindow, InferenceSelection, InitialLaw,
            Intervention, InterventionWorld, ModelAssumptions, ScientificTarget, TargetForm,
            TemporalPolicy, TimeFrame, WorldAction, WorldCollection,
        };
        let source = interaction_target();
        let antecedent_core::TargetForm::Contrast(contrast) = source.form() else {
            unreachable!("interaction fixture")
        };
        let binding = contrast.worlds().worlds()[0].binding().clone();
        let make_target = |history| {
            let worlds = WorldCollection::new((0..4).map(|id| {
                let level = if id < 2 { 0.0 } else { 1.0 };
                let horizon = id % 2 + 1;
                InterventionWorld::new(
                    ScientificWorldId::from_raw(id),
                    WorldAction::schedule(
                        Intervention::set(VariableId::from_raw(0), Value::f64(level)),
                        TemporalPolicy::pulse(0),
                    )
                    .unwrap(),
                    id == 0,
                    binding.clone(),
                    TimeFrame::temporal(
                        horizon,
                        history,
                        InitialLaw::named("observed_origin").unwrap(),
                    )
                    .unwrap(),
                )
                .unwrap()
            }))
            .unwrap();
            let surface = DoseTimeSurface::new(
                worlds,
                VariableId::from_raw(0),
                [0.0, 1.0],
                [1, 2],
                (0..4).map(ScientificWorldId::from_raw),
            )
            .unwrap();
            ScientificTarget::new(
                TargetForm::DoseTime(surface),
                ModelAssumptions::new(GraphFamily::TemporalDag, AssumptionSet::new()),
                InferenceSelection::Frequentist,
            )
        };
        let target = make_target(HistoryWindow::max_lag(2));
        let mut product = supported_interaction_product();
        product.target_identity = Some(target.identity());
        assert_eq!(product.validate_target(&target), Ok(()));
        product.assembled = None;
        assert_eq!(
            product.validate_target(&target),
            Err(ProductRefusal::MissingAssembledObligation)
        );
        let changed = make_target(HistoryWindow::max_lag(1));
        assert!(matches!(
            product.validate_target(&changed),
            Err(ProductRefusal::TargetIdentityMismatch { .. })
        ));
        product.world_obligations.pop();
        assert_eq!(
            product.validate_target(&target),
            Err(ProductRefusal::MissingWorldObligation { world: ScientificWorldId::from_raw(3) })
        );
    }

    #[test]
    fn mediation_cannot_acquire_authority_from_generic_world_validation() {
        use antecedent_core::{
            GraphFamily, InferenceSelection, Intervention, MediationArms, MediationComponent,
            MediationTarget, ModelAssumptions, ScientificTarget, TargetForm, TimeFrame,
        };
        let source = interaction_target();
        let TargetForm::Contrast(contrast) = source.form() else {
            unreachable!("interaction fixture")
        };
        let binding = contrast.worlds().worlds()[0].binding().clone();
        let arms = MediationArms::new(
            VariableId::from_raw(0),
            [VariableId::from_raw(1)],
            Intervention::set(VariableId::from_raw(0), Value::f64(1.0)),
            Intervention::set(VariableId::from_raw(0), Value::f64(0.0)),
            MediationComponent::Total,
        )
        .unwrap();
        let mediation =
            MediationTarget::new(binding, TimeFrame::static_decision(), None, arms, []).unwrap();
        let target = ScientificTarget::new(
            TargetForm::Mediation(mediation),
            ModelAssumptions::new(GraphFamily::Dag, AssumptionSet::new()),
            InferenceSelection::Frequentist,
        );
        let mut product = supported_interaction_product();
        product.target_identity = Some(target.identity());
        assert_eq!(
            product.validate_target(&target),
            Err(ProductRefusal::MediationRequiresTheoremScopedSupport)
        );
    }

    #[test]
    fn six_identification_outcomes_do_not_compare_equal() {
        assert_eq!(OUTCOMES.len(), 6);
        let mut codes = OUTCOMES.map(outcome_code);
        codes.sort_unstable();
        assert_eq!(codes, [1, 2, 3, 4, 5, 6]);
        for (index, left) in OUTCOMES.iter().enumerate() {
            for right in &OUTCOMES[index + 1..] {
                assert_ne!(left, right);
            }
        }
    }

    #[test]
    fn model_conditional_and_unsupported_extrapolation_are_not_identified_claims() {
        assert_ne!(
            IdentificationClaimClass::ModelConditional,
            IdentificationClaimClass::IdentifiedClaim
        );
        assert_ne!(
            IdentificationClaimClass::UnsupportedExtrapolation,
            IdentificationClaimClass::IdentifiedClaim
        );
        assert!(!IdentificationClaimClass::ModelConditional.is_identified_claim());
        assert!(!IdentificationClaimClass::UnsupportedExtrapolation.is_identified_claim());
        assert_ne!(
            claim_code(IdentificationClaimClass::ModelConditional),
            claim_code(IdentificationClaimClass::IdentifiedClaim)
        );
        assert_ne!(
            claim_code(IdentificationClaimClass::UnsupportedExtrapolation),
            claim_code(IdentificationClaimClass::IdentifiedClaim)
        );

        let conditional = empty_product(
            IdentificationOutcome::Identified,
            IdentificationClaimClass::ModelConditional,
            Some(1.0),
        );
        let extrapolated = empty_product(
            IdentificationOutcome::Identified,
            IdentificationClaimClass::UnsupportedExtrapolation,
            Some(2.0),
        );
        let identified = empty_product(
            IdentificationOutcome::Identified,
            IdentificationClaimClass::IdentifiedClaim,
            None,
        );
        assert!(!conditional.is_identified_claim());
        assert!(!extrapolated.is_identified_claim());
        assert!(identified.is_identified_claim());
        assert!(conditional.is_numerically_feasible());
    }

    #[test]
    fn numerically_feasible_value_beside_unidentified_is_not_an_identified_claim() {
        let feasible = empty_product(
            IdentificationOutcome::Unidentified,
            IdentificationClaimClass::ModelConditional,
            Some(1.25),
        );
        assert_eq!(feasible.numerical_value(), Some(1.25));
        assert_eq!(feasible.outcome(), IdentificationOutcome::Unidentified);
        assert!(feasible.is_numerically_feasible());
        assert!(!feasible.is_identified_claim());
        assert_ne!(feasible.outcome(), IdentificationOutcome::Identified);
        assert_ne!(feasible.outcome(), IdentificationOutcome::NumericallyInfeasible);

        let infeasible = empty_product(
            IdentificationOutcome::NumericallyInfeasible,
            IdentificationClaimClass::ModelConditional,
            Some(1.25),
        );
        assert!(!infeasible.is_numerically_feasible());
        assert!(!infeasible.is_identified_claim());
    }

    fn refused(
        estimands: Vec<IdentifiedEstimand>,
        assumptions: AssumptionSet,
        parts: CompiledProductParts,
    ) -> ProductRefusal {
        CompiledIdentificationProduct::new(
            estimands,
            CausalExprArena::new(),
            DerivationTrace::default(),
            assumptions,
            Vec::new(),
            parts,
        )
        .expect_err("incoherent product is refused")
    }

    #[test]
    fn an_outcome_that_disagrees_with_its_claim_or_estimands_is_refused() {
        // Identified with nothing retained.
        let identified_claim = || {
            CompiledProductParts::new(
                IdentificationOutcome::Identified,
                IdentificationClaimClass::IdentifiedClaim,
            )
        };
        assert_eq!(
            refused(Vec::new(), AssumptionSet::new(), identified_claim()),
            ProductRefusal::IdentifiedWithoutEstimand
        );

        // Identified while an assumption it relies on is contradicted.
        let mut assumptions = AssumptionSet::new();
        assumptions.push(AssumptionRecord {
            assumption: Assumption::Positivity,
            source: AssumptionSource::UserDeclared,
            scope: AssumptionScope::Identification,
            status: AssumptionStatus::Supported,
        });
        assumptions.push(AssumptionRecord {
            assumption: Assumption::Consistency,
            source: AssumptionSource::UserDeclared,
            scope: AssumptionScope::Identification,
            status: AssumptionStatus::Contradicted,
        });
        assert_eq!(
            refused(vec![estimand()], assumptions, identified_claim()),
            ProductRefusal::IdentifiedWithContradictedAssumption { index: 1 }
        );

        // An unidentified outcome may not state the identified-claim class.
        assert_eq!(
            refused(
                Vec::new(),
                AssumptionSet::new(),
                CompiledProductParts::new(
                    IdentificationOutcome::Unidentified,
                    IdentificationClaimClass::IdentifiedClaim,
                )
            ),
            ProductRefusal::UnidentifiedWithIdentifiedClaim
        );

        // A failure outcome may not state the identified-claim class or retain an estimand.
        for outcome in [
            IdentificationOutcome::Unsupported,
            IdentificationOutcome::TimedOut,
            IdentificationOutcome::NumericallyInfeasible,
            IdentificationOutcome::Uncalibrated,
        ] {
            assert_eq!(
                refused(
                    Vec::new(),
                    AssumptionSet::new(),
                    CompiledProductParts::new(outcome, IdentificationClaimClass::IdentifiedClaim)
                ),
                ProductRefusal::IdentifiedClaimClassBesideFailure { outcome }
            );
            assert_eq!(
                refused(
                    vec![estimand()],
                    AssumptionSet::new(),
                    CompiledProductParts::new(outcome, IdentificationClaimClass::ModelConditional)
                ),
                ProductRefusal::EstimandBesideNonIdentification { outcome }
            );
        }

        // The same rule covers world and assembled obligations.
        let mut with_world = CompiledProductParts::new(
            IdentificationOutcome::Unidentified,
            IdentificationClaimClass::ModelConditional,
        );
        with_world.world_obligations = vec![IdentificationObligation::for_world(
            ScientificWorldId::from_raw(1),
            IdentificationOutcome::TimedOut,
            IdentificationClaimClass::IdentifiedClaim,
            Vec::new(),
            None,
        )];
        assert_eq!(
            refused(Vec::new(), AssumptionSet::new(), with_world),
            ProductRefusal::IdentifiedClaimClassBesideFailure {
                outcome: IdentificationOutcome::TimedOut
            }
        );
    }

    #[test]
    fn assembled_contrast_is_not_identified_when_each_world_is_only_feasible() {
        let world = |raw: u32| {
            IdentificationObligation::for_world(
                ScientificWorldId::from_raw(raw),
                IdentificationOutcome::Unidentified,
                IdentificationClaimClass::ModelConditional,
                Vec::new(),
                Some(f64::from(raw)),
            )
        };
        let assembled = IdentificationObligation::assembled(
            IdentificationOutcome::Unidentified,
            IdentificationClaimClass::ModelConditional,
            Vec::new(),
            Some(4.0),
        );
        let mut parts = CompiledProductParts::new(
            IdentificationOutcome::Identified,
            IdentificationClaimClass::IdentifiedClaim,
        );
        parts.world_obligations = vec![world(1), world(2)];
        parts.assembled = Some(assembled);
        let product = CompiledIdentificationProduct::new(
            vec![estimand()],
            CausalExprArena::new(),
            DerivationTrace::default(),
            AssumptionSet::new(),
            Vec::new(),
            parts,
        )
        .unwrap();

        assert!(product.world_obligations().iter().all(WorldObligation::is_numerically_feasible));
        assert!(product.world_obligations().iter().all(|world| !world.is_identified_claim()));
        assert_eq!(
            product.assembled_obligation().map(AssembledFunctionalObligation::outcome),
            Some(IdentificationOutcome::Unidentified)
        );
        assert!(
            product
                .assembled_obligation()
                .is_some_and(AssembledFunctionalObligation::is_numerically_feasible,)
        );
        assert!(!product.assembled_is_identified_claim());
        assert!(!product.is_identified_claim());
    }

    #[test]
    fn retain_keeps_functional_derivation_assumptions_factors_obligations_and_bounded_search() {
        let mut assumptions = AssumptionSet::new();
        assumptions.push(AssumptionRecord {
            assumption: Assumption::Positivity,
            source: AssumptionSource::UserDeclared,
            scope: AssumptionScope::Identification,
            status: AssumptionStatus::Declared,
        });
        let mut derivation = DerivationTrace::default();
        derivation.push("backdoor", "adjustment set blocks back-door paths");
        let estimand = IdentifiedEstimand::backdoor(
            "backdoor.adjustment",
            Arc::from([VariableId::from_raw(2)]),
            ExprId::from_raw(0),
        );
        let mut result = IdentificationResult::identified(
            query(),
            vec![estimand],
            CausalExprArena::new(),
            derivation,
            assumptions,
            IdentificationPerformanceRecord::default(),
        );
        result
            .diagnostics
            .push(search_bounded_diagnostic("identify.example.search_bounded", "stopped"));
        result.diagnostics.push(Diagnostic::new(
            crate::generalized::CAPPED_COMPLETION_DIAGNOSTIC_CODE,
            DiagnosticKind::Execution,
            DiagnosticSeverity::Warning,
            "capped",
        ));
        result.diagnostics.push(Diagnostic::new(
            crate::temporal_mag::HISTORY_CAPPED,
            DiagnosticKind::Execution,
            DiagnosticSeverity::Warning,
            "history",
        ));
        result.diagnostics.push(Diagnostic::new(
            "identify.scientific.open_path",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Warning,
            "open",
        ));

        let mut parts = CompiledProductParts::new(
            IdentificationOutcome::Identified,
            IdentificationClaimClass::IdentifiedClaim,
        );
        parts.required_factors =
            vec![RequiredFactor::new("outcome_mean", Arc::from([VariableId::from_raw(1)]))];
        parts.unresolved_obligations =
            vec![UnresolvedObligation::new("support.overlap", "empty cell")];

        // A timeout cannot retain the identified functional of this result.
        let timed_out = CompiledProductParts::new(
            IdentificationOutcome::TimedOut,
            IdentificationClaimClass::ModelConditional,
        );
        assert_eq!(
            CompiledIdentificationProduct::retain(&result, timed_out).unwrap_err(),
            ProductRefusal::EstimandBesideNonIdentification {
                outcome: IdentificationOutcome::TimedOut
            }
        );
        // An identified outcome cannot be retained from a result that is not identified.
        let none = IdentificationResult::not_identified(
            query(),
            DerivationTrace::default(),
            AssumptionSet::new(),
            IdentificationPerformanceRecord::default(),
        );
        assert_eq!(
            CompiledIdentificationProduct::retain(&none, parts.clone()).unwrap_err(),
            ProductRefusal::IdentifiedFromUnidentifiedResult {
                status: IdentificationStatus::NotIdentified
            }
        );
        // An identified outcome may carry declared premises and must not carry an open obligation.
        assert_eq!(
            CompiledIdentificationProduct::retain(&result, parts.clone()).unwrap_err(),
            ProductRefusal::IdentifiedWithUnresolvedObligation { index: 0 }
        );
        let mut unidentified = parts.clone();
        unidentified.outcome = IdentificationOutcome::Unidentified;
        unidentified.claim_class = IdentificationClaimClass::ModelConditional;
        let carrying = CompiledIdentificationProduct::retain(&result, unidentified).unwrap();
        assert_eq!(carrying.unresolved_obligations().len(), 1);
        assert_eq!(carrying.unresolved_obligations()[0].code(), "support.overlap");
        assert!(!carrying.is_identified_claim());
        parts.unresolved_obligations.clear();
        let product = CompiledIdentificationProduct::retain(&result, parts).unwrap();

        assert_eq!(product.identified().len(), 1);
        assert_eq!(product.identified()[0].adjustment_set.as_ref(), &[VariableId::from_raw(2)]);
        assert_eq!(product.identified()[0].functional, ExprId::from_raw(0));
        assert!(product.arena().is_empty());
        assert_eq!(product.derivation().steps.len(), 1);
        assert_eq!(product.derivation().steps[0].rule.as_ref(), "backdoor");
        assert_eq!(product.assumptions().len(), 1);
        assert_eq!(product.premises().count(), 1);
        assert!(product.unresolved_obligations().is_empty());
        assert_eq!(product.required_factors().len(), 1);
        assert_eq!(product.required_factors()[0].name(), "outcome_mean");
        assert_eq!(product.bounded_search_diagnostics().len(), 3);
        assert!(product.bounded_search_diagnostics().iter().all(|diagnostic| {
            diagnostic.code.as_ref().ends_with(".search_bounded")
                || diagnostic.code.as_ref() == crate::generalized::CAPPED_COMPLETION_DIAGNOSTIC_CODE
                || diagnostic.code.as_ref() == crate::temporal_mag::HISTORY_CAPPED
        }));
        assert!(product.is_identified_claim());
    }

    #[test]
    fn scientific_identity_ignores_numerical_values_and_bounded_search_diagnostics() {
        let with_number = empty_product(
            IdentificationOutcome::Unidentified,
            IdentificationClaimClass::ModelConditional,
            Some(0.0),
        );
        let without_number = empty_product(
            IdentificationOutcome::Unidentified,
            IdentificationClaimClass::ModelConditional,
            None,
        );
        assert_eq!(with_number, without_number);
        assert_eq!(with_number.identity(), without_number.identity());
        assert_ne!(with_number.numerical_value(), without_number.numerical_value());

        let negative_zero = empty_product(
            IdentificationOutcome::Unidentified,
            IdentificationClaimClass::ModelConditional,
            Some(-0.0),
        );
        assert_eq!(with_number, negative_zero);
        assert_eq!(with_number.identity(), negative_zero.identity());

        let mut parts = CompiledProductParts::new(
            IdentificationOutcome::Unidentified,
            IdentificationClaimClass::ModelConditional,
        );
        parts.world_obligations = vec![IdentificationObligation::for_world(
            ScientificWorldId::from_raw(7),
            IdentificationOutcome::Unidentified,
            IdentificationClaimClass::ModelConditional,
            Vec::new(),
            Some(9.0),
        )];
        parts.assembled = Some(IdentificationObligation::assembled(
            IdentificationOutcome::TimedOut,
            IdentificationClaimClass::UnsupportedExtrapolation,
            Vec::new(),
            Some(8.0),
        ));
        let worlds = CompiledIdentificationProduct::new(
            Vec::new(),
            CausalExprArena::new(),
            DerivationTrace::default(),
            AssumptionSet::new(),
            Vec::new(),
            parts.clone(),
        )
        .unwrap();
        parts.world_obligations[0] = IdentificationObligation::for_world(
            ScientificWorldId::from_raw(7),
            IdentificationOutcome::Unidentified,
            IdentificationClaimClass::ModelConditional,
            Vec::new(),
            None,
        );
        if let Some(assembled) = parts.assembled.as_mut() {
            *assembled = IdentificationObligation::assembled(
                IdentificationOutcome::TimedOut,
                IdentificationClaimClass::UnsupportedExtrapolation,
                Vec::new(),
                None,
            );
        }
        let worlds_without_numbers = CompiledIdentificationProduct::new(
            Vec::new(),
            CausalExprArena::new(),
            DerivationTrace::default(),
            AssumptionSet::new(),
            Vec::new(),
            parts,
        )
        .unwrap();
        assert_eq!(worlds, worlds_without_numbers);
        assert_eq!(worlds.identity(), worlds_without_numbers.identity());
    }

    #[test]
    fn bounded_search_diagnostics_are_retained_and_omitted_from_identity() {
        let diagnostic = search_bounded_diagnostic("identify.example.search_bounded", "stopped");
        let with_search = CompiledIdentificationProduct::new(
            Vec::new(),
            CausalExprArena::new(),
            DerivationTrace::default(),
            AssumptionSet::new(),
            vec![diagnostic],
            CompiledProductParts::new(
                IdentificationOutcome::TimedOut,
                IdentificationClaimClass::ModelConditional,
            ),
        )
        .unwrap();
        let without_search = empty_product(
            IdentificationOutcome::TimedOut,
            IdentificationClaimClass::ModelConditional,
            None,
        );
        assert_eq!(with_search, without_search);
        assert_eq!(with_search.identity(), without_search.identity());
        assert_eq!(with_search.bounded_search_diagnostics().len(), 1);
        assert!(without_search.bounded_search_diagnostics().is_empty());

        let mut derivation = DerivationTrace::default();
        derivation.push("id", "hedge");
        let different_derivation = CompiledIdentificationProduct::new(
            Vec::new(),
            CausalExprArena::new(),
            derivation,
            AssumptionSet::new(),
            Vec::new(),
            CompiledProductParts::new(
                IdentificationOutcome::Unidentified,
                IdentificationClaimClass::ModelConditional,
            ),
        )
        .unwrap();
        let unidentified = empty_product(
            IdentificationOutcome::Unidentified,
            IdentificationClaimClass::ModelConditional,
            None,
        );
        assert_ne!(different_derivation, unidentified);
        assert_ne!(different_derivation.identity(), unidentified.identity());
    }

    #[test]
    fn functional_identity_ignores_expression_derivation_metadata() {
        let mut arena = CausalExprArena::new();
        let root = arena.backdoor_mean(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
            &[VariableId::from_raw(2)],
            Value::f64(1.0),
        );
        let parts = CompiledProductParts::new(
            IdentificationOutcome::Identified,
            IdentificationClaimClass::IdentifiedClaim,
        );
        let functional = CompiledIdentificationProduct::new(
            vec![IdentifiedEstimand::backdoor(
                "backdoor.adjustment",
                Arc::from([VariableId::from_raw(2)]),
                root,
            )],
            arena.clone(),
            DerivationTrace::default(),
            AssumptionSet::new(),
            Vec::new(),
            parts.clone(),
        )
        .unwrap();
        arena.set_derivation(root, DerivationMeta::rule("renamed", None));
        let same_functional = CompiledIdentificationProduct::new(
            vec![IdentifiedEstimand::backdoor(
                "backdoor.adjustment",
                Arc::from([VariableId::from_raw(2)]),
                root,
            )],
            arena,
            DerivationTrace::default(),
            AssumptionSet::new(),
            Vec::new(),
            parts,
        )
        .unwrap();
        let empty = empty_product(
            IdentificationOutcome::Unidentified,
            IdentificationClaimClass::ModelConditional,
            None,
        );
        assert_eq!(functional, same_functional);
        assert_eq!(functional.identity(), same_functional.identity());
        assert_ne!(functional.identity(), empty.identity());
        assert!(functional.is_identified_claim());
    }
}
