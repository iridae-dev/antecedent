//! Extracted stage of the parent estimator module.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

/// Receipt for a linear adjustment preparation checked against an identification result.
///
/// The retained arena and target make the proof context inspectable after the builder has
/// been discarded. `problem` is the physical linear design prepared from that same query.
#[derive(Clone, Debug)]
pub struct CheckedLinearAdjustmentAte {
    /// Selected causal target and its estimator roles.
    pub(super) target: IdentifiedEstimand,
    /// Selected source functional root.
    pub(super) source_functional: ExprId,
    /// One-arm observational adjustment functional derived from the source target.
    /// Evaluate with treatment bound to each declared arm, then subtract control from active.
    pub(super) executable_functional: ExprId,
    /// Checked role and population receipt for the source-to-executable lowering.
    pub(super) lowering: CheckedAdjustmentLowering,
    /// Structurally checked owner of the source and executable roots.
    pub(super) program: FunctionalProgram,
    /// Provider factors discovered while checking the executable root.
    pub(super) factor_requirements: Arc<[FactorRequirement]>,
    /// Query that was checked and used to bind the source target.
    pub(super) query: AverageEffectQuery,
    /// Assumptions attached to the selected identification claim.
    pub(super) required_assumptions: AssumptionSet,
    /// Checked and prepared linear adjustment design.
    pub(super) problem: PreparedEstimationProblem,
    pub(super) backend: antecedent_core::physical_choice::PhysicalAlternative,
}

impl CheckedLinearAdjustmentAte {
    /// Physical implementation that assembled this retained design.
    #[must_use]
    pub fn physical_backend(&self) -> antecedent_core::physical_choice::PhysicalAlternative {
        self.backend
    }
    /// Selected causal target.
    #[must_use]
    pub fn target(&self) -> &IdentifiedEstimand {
        &self.target
    }
    /// Checked program that owns the target and observational execution root.
    #[must_use]
    pub fn program(&self) -> &FunctionalProgram {
        &self.program
    }
    /// Checked source root.
    #[must_use]
    pub fn source_functional(&self) -> ExprId {
        self.source_functional
    }
    /// Checked executable arm root.
    #[must_use]
    pub fn executable_functional(&self) -> ExprId {
        self.executable_functional
    }
    /// Typed source-to-executable lowering.
    #[must_use]
    pub fn lowering(&self) -> &CheckedAdjustmentLowering {
        &self.lowering
    }
    /// Checked provider factor requirements.
    #[must_use]
    pub fn factor_requirements(&self) -> &[FactorRequirement] {
        &self.factor_requirements
    }
    /// Selected identification assumptions.
    #[must_use]
    pub fn required_assumptions(&self) -> &AssumptionSet {
        &self.required_assumptions
    }
    /// Prepared numerical design.
    #[must_use]
    pub fn problem(&self) -> &PreparedEstimationProblem {
        &self.problem
    }
}

/// Checked semantic correspondence for a static adjustment ATE lowering.
///
/// The executable root is an observational one-arm functional with treatment free as a
/// conditioning coordinate. Consumers evaluate it twice using the retained active and
/// control values and subtract control from active.
#[derive(Clone, Debug, PartialEq)]
pub struct CheckedAdjustmentLowering {
    /// Source interventional target root.
    pub source: ExprId,
    /// Executable observational root.
    pub executable: ExprId,
    /// Treatment role used by both arms.
    pub treatment: VariableId,
    /// Outcome role.
    pub outcome: VariableId,
    /// Adjustment roles in semantic order.
    pub adjustment: Arc<[VariableId]>,
    /// Active intervention value.
    pub active: f64,
    /// Control intervention value.
    pub control: f64,
    /// Population averaging semantics implemented by this checked lowering.
    pub population: TargetPopulation,
}
