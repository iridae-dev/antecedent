//! The compiled identification product and its accessors.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

/// Compiled identification product.
///
/// Lowering calls [`Self::retain`] so the functional or identified set, the
/// derivation, the assumptions, and the bounded-search diagnostics are copied
/// out of the [`IdentificationResult`] before that result is dropped.
#[derive(Clone, Debug)]
#[must_use = "lowering must retain this identification product"]
#[non_exhaustive]
pub struct CompiledIdentificationProduct {
    pub(super) estimands: Vec<IdentifiedEstimand>,
    pub(super) arena: CausalExprArena,
    pub(super) derivation: crate::result::DerivationTrace,
    pub(super) assumptions: AssumptionSet,
    pub(super) required_factors: Vec<RequiredFactor>,
    pub(super) unresolved_obligations: Vec<UnresolvedObligation>,
    pub(super) bounded_search: Vec<Diagnostic>,
    pub(super) outcome: IdentificationOutcome,
    pub(super) claim_class: IdentificationClaimClass,
    pub(super) numerical_value: Option<f64>,
    pub(super) world_obligations: Vec<IdentificationObligation>,
    pub(super) assembled: Option<IdentificationObligation>,
    pub(super) target_identity: Option<SemanticDigest>,
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
        Self::retain(result, parts)?.bind_target(target)
    }

    /// Bind retained outcomes directly when an unsupported compilation has no source result.
    ///
    /// This uses the same world and assembled checks as [`Self::retain_for_target`]. It
    /// does not identify a functional. An existing binding cannot be reassigned to a
    /// different scientific question.
    ///
    /// # Errors
    /// Missing support structure or a changed scientific target.
    pub fn bind_target(mut self, target: &ScientificTarget) -> Result<Self, ProductRefusal> {
        if let Some(expected) = self.target_identity {
            let actual = target.identity();
            if expected != actual {
                return Err(ProductRefusal::TargetIdentityMismatch { expected, actual });
            }
        }
        self.target_identity = Some(target.identity());
        self.validate_target(target)?;
        Ok(self)
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
    /// an assembled obligation is recorded, every world and any required assembled
    /// functional must themselves be identified claims. Bound world collections do not
    /// require an assembled contrast. Legacy unbound collections remain conservative. Feasibility of the
    /// worlds does not identify the contrast.
    #[must_use]
    pub fn is_identified_claim(&self) -> bool {
        let own = states_identified_claim(self.outcome, self.claim_class);
        if self.world_obligations.is_empty() && self.assembled.is_none() {
            return own;
        }
        own && self.world_obligations.iter().all(IdentificationObligation::is_identified_claim)
            && self.assembled.as_ref().map_or(
                self.target_identity.is_some(),
                IdentificationObligation::is_identified_claim,
            )
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
