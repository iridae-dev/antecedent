//! Execute the finest decomposition the retained design identifies, and name the next design.
//!
//! The sixteen reference worlds are checked one by one. A component step (one component
//! moved from its before to its after value, the other three held at a fixed context) is
//! identified exactly when both worlds it differences are identified. Those steps are executed
//! on the same aligned fitted law, the components with no identified step are listed, and every
//! missing premise is turned into the concrete design or analysis that would earn it. The
//! total-response contrast is executed on request. None of this promotes an unidentified
//! hybrid to a causal attribution.
// SPDX-License-Identifier: MIT OR Apache-2.0
use super::{
    ChangeComponent, CheckedReferenceDecomposition, ExecutionContext, JointPanelError,
    JointPanelExplanationAttempt, JointPanelExplanationRequest, JointPanelResult,
    MechanismPeriodMap, PanelDecompositionRequest, ReferenceObligation, engine,
};
use crate::analysis::joint_panel::JointPanelFit;
use engine::selection_uncertainty::SelectionDecomposition;

const COMPONENTS: [ChangeComponent; 4] = [
    ChangeComponent::Exposure,
    ChangeComponent::Saturation,
    ChangeComponent::Population,
    ChangeComponent::Mechanism,
];

/// One component moved from before to after in a context the design identifies.
#[derive(Clone, Debug, PartialEq)]
pub struct IdentifiedComponentStep {
    /// The component that changes.
    pub component: ChangeComponent,
    /// The other components, bit order exposure, saturation, population, mechanism; a set
    /// bit holds that component at its after value. The stepped component's own bit is clear.
    pub context_mask: u8,
    /// Hybrid response with the component at after minus at before, on the point fit.
    pub point: f64,
    /// The same contrast on every aligned law slot; failed slots stay `None`.
    pub draws: Vec<Option<f64>>,
}

/// The design or analysis that would earn one missing premise.
#[derive(Clone, Debug, PartialEq)]
pub struct NextDesignStep {
    /// The premise this step earns.
    pub obligation: ReferenceObligation,
    /// Components whose after value needed it, in decomposition order.
    pub components: Vec<ChangeComponent>,
    /// Hybrid masks that lack the premise.
    pub hybrids: Vec<u8>,
    /// What to run, in terms of the design that identifies it.
    pub design: String,
}

impl NextDesignStep {
    /// Keep the first-seen hybrid and component ordering for this missing premise.
    fn add_hybrid(&mut self, mask: u8) {
        self.hybrids.push(mask);
        let blamed = blamed_components(self.obligation, mask);
        for (bit, component) in COMPONENTS.iter().enumerate() {
            if blamed & (1 << bit) != 0 && !self.components.contains(component) {
                self.components.push(*component);
            }
        }
    }
}

/// Finer decomposition as far as the design supports, with what follows from the rest.
#[derive(Clone, Debug)]
pub struct DesignAdaptiveReference {
    /// The complete model-conditional decomposition with per-hybrid identification status.
    pub requested: CheckedReferenceDecomposition,
    /// Every component step whose two hybrids are both identified, executed on the law.
    pub identified_steps: Vec<IdentifiedComponentStep>,
    /// Components with no identified step: their attribution is model-conditional only.
    pub unidentified_components: Vec<ChangeComponent>,
    /// One entry per missing premise, each naming the design that would earn it.
    pub next_designs: Vec<NextDesignStep>,
    /// Change-point, smoothing and selection split of the decomposed law, for temporal fits.
    pub selection: Option<SelectionDecomposition>,
    /// The coarser total-response contrast, executed when the finer claim is incomplete.
    pub coarser: Option<JointPanelExplanationAttempt>,
}
impl DesignAdaptiveReference {
    /// Whether every component and context is identified, so nothing is left to attempt.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.requested.is_identified_reference()
    }
}

fn design_for(obligation: ReferenceObligation) -> &'static str {
    match obligation {
        ReferenceObligation::MissingIdentificationProof => {
            "run a design that identifies the base worlds (randomized factorial or multi-arm \
             assignment, a switchback, or a geo experiment with the exposure assigned) and \
             compile its proof; until then only the model-conditional decomposition exists"
        }
        ReferenceObligation::UnregisteredHybridWorld => {
            "add the assembled hybrid, with the exposure of one endpoint and the context of the \
             other, as an explicit intervention/context world of the request so the design is \
             checked against it"
        }
        ReferenceObligation::CrossPeriodMechanismTransport => {
            "identify the mechanism in each period separately: replicate the same programs in \
             both reference windows, or stagger or switch the exposure across the change window, \
             so a period-specific parameter intervention is identified; the reference-window \
             difference of program differences is the contrast executable meanwhile"
        }
        ReferenceObligation::ChangedPopulationTransport => {
            "observe the programs in both populations (stratified or geo-holdout assignment with \
             known population weights) or supply an explicit population transport proof with \
             overlap; reweighting fitted responses alone does not identify the target population"
        }
        ReferenceObligation::NonTerminalFunctional => {
            "retarget the compiled question to the requested time-weighted response functional \
             before attributing, so the proof covers the periods being weighted"
        }
    }
}

/// Components whose after value made this hybrid need the premise.
fn blamed_components(obligation: ReferenceObligation, mask: u8) -> u8 {
    let (touched, fallback) = match obligation {
        ReferenceObligation::ChangedPopulationTransport => (mask & 0b0100, 0b0100),
        ReferenceObligation::CrossPeriodMechanismTransport => (mask & 0b1010, 0b1010),
        ReferenceObligation::UnregisteredHybridWorld => (mask, 0b1111),
        ReferenceObligation::MissingIdentificationProof
        | ReferenceObligation::NonTerminalFunctional => (0b1111, 0b1111),
    };
    if touched == 0 { fallback } else { touched }
}

impl JointPanelResult {
    /// Split a temporal decomposition by the specification each aligned draw selected.
    /// A stationary fit has no specification to select and returns `None`.
    ///
    /// # Errors
    /// The decomposition does not come from this fitted law's aligned draws.
    pub fn decomposition_selection(
        &self,
        decomposition: &engine::decomposition::PanelDecomposition,
    ) -> Result<Option<SelectionDecomposition>, JointPanelError> {
        match &self.fit {
            JointPanelFit::Invariant(_) => Ok(None),
            JointPanelFit::Varying(fit) => {
                Ok(Some(engine::selection_uncertainty::decomposition_selection(
                    decomposition,
                    &fit.point,
                    &fit.draws,
                )?))
            }
        }
    }

    /// Execute the finest decomposition the retained design identifies.
    ///
    /// Component steps whose two hybrids are identified are computed on the aligned law;
    /// remaining components are listed with the design that would identify them. When the
    /// full decomposition is not identified and `fallback` is supplied, the coarser total
    /// contrast is executed on the same fit as well. Resource, cancellation and numerical
    /// failures propagate; they are never reported as unidentified.
    ///
    /// # Errors
    /// Invalid endpoints or fallback, cancellation, admission or numerical failure.
    pub fn attempt_design_adaptive_reference(
        &self,
        request: &PanelDecompositionRequest,
        periods: &MechanismPeriodMap,
        fallback: Option<JointPanelExplanationRequest>,
        ctx: &ExecutionContext,
    ) -> Result<DesignAdaptiveReference, JointPanelError> {
        let requested = self.decompose_reference_checked(request, periods, ctx)?;
        let identified = |mask: u8| requested.hybrids[usize::from(mask)].unresolved.is_empty();
        let mut identified_steps = Vec::new();
        for (bit, component) in COMPONENTS.iter().enumerate() {
            for context in 0_u8..16 {
                if context & (1 << bit) != 0 {
                    continue;
                }
                let stepped = context | (1 << bit);
                if !identified(context) || !identified(stepped) {
                    continue;
                }
                let contrast = |hybrids: &[f64; 16]| {
                    hybrids[usize::from(stepped)] - hybrids[usize::from(context)]
                };
                identified_steps.push(IdentifiedComponentStep {
                    component: *component,
                    context_mask: context,
                    point: contrast(&requested.decomposition.point.hybrids),
                    draws: requested
                        .decomposition
                        .draws
                        .iter()
                        .map(|draw| draw.as_ref().map(|draw| contrast(&draw.hybrids)))
                        .collect(),
                });
            }
        }
        let unidentified_components = COMPONENTS
            .iter()
            .copied()
            .filter(|component| !identified_steps.iter().any(|step| step.component == *component))
            .collect::<Vec<_>>();
        let mut next_designs: Vec<NextDesignStep> = Vec::new();
        for hybrid in &requested.hybrids {
            for obligation in &hybrid.obligations {
                if let Some(step) =
                    next_designs.iter_mut().find(|step| step.obligation == *obligation)
                {
                    step.add_hybrid(hybrid.mask);
                } else {
                    let mut step = NextDesignStep {
                        obligation: *obligation,
                        components: Vec::new(),
                        hybrids: Vec::new(),
                        design: design_for(*obligation).into(),
                    };
                    step.add_hybrid(hybrid.mask);
                    next_designs.push(step);
                }
            }
        }
        let selection = self.decomposition_selection(&requested.decomposition)?;
        let coarser = match fallback {
            Some(fallback) if !requested.is_identified_reference() => {
                Some(self.attempt_explanation(fallback, ctx)?)
            }
            _ => None,
        };
        Ok(DesignAdaptiveReference {
            requested,
            identified_steps,
            unidentified_components,
            next_designs,
            selection,
            coarser,
        })
    }
}
