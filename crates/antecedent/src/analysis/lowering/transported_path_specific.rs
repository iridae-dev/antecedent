//! Lower the transported static path-specific counterfactual into the common six-layer program.
//!
//! The question is the target-population path-specific edge-intervention contrast under a
//! supplied affine additive-noise structural model, finite source and target covariate laws and a
//! selection diagram. The answer is conditional on that declared model, so the identification
//! layer holds the model's digest; the source and target regime evidence are the program's two
//! evidence sources. The program's identity is the sealed request's canonical identity.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_contract::DataClassification;
use antecedent_core::{ByteEstimates, CanonicalWriter, PhysicalEffect, PhysicalOperator};
use antecedent_io::transported_counterfactual_artifact::{
    TransportedCounterfactualIdentity, TransportedCounterfactualRequestWire,
};

use crate::error::CausalError;
use crate::planner::lowered::{
    ClaimDemand, Composition, DesignShape, EvidenceDecisionProgram, EvidenceRole,
    EvidenceSourceRef, FunctionalKind, IdentificationLayer, LoweredProgram,
    NonGraphicalIdentification, PhysicalLayer, ProblemTarget, ProgramIdentityScope,
    ScientificProblem, TransportedPathSpecificLoweredParts, UncertaintyReading,
};
use crate::planner::statistical::{Realization, StatisticalProgram};

/// Plan id and evaluator of the route.
const PLAN_ID: &str = "counterfactual.transported_additive";
/// Domain tag of a digest of one identity component.
const COMPONENT_TAG: &str = "antecedent.transported_path_specific.component";

fn error(message: impl std::fmt::Display) -> CausalError {
    crate::compile_reason!("invalid_argument", "transported_path_specific.lowering: {message}")
}

/// A 32-byte digest of one hex identity component of the sealed request.
fn digest(component: &str, value: &str) -> [u8; 32] {
    let mut writer = CanonicalWriter::new(COMPONENT_TAG);
    writer.text(component).text(value);
    writer.finish()
}

/// Lower the sealed request whose canonical identity is `identity`.
///
/// # Errors
///
/// A request whose size overflows the read quote.
pub(crate) fn lower(
    request: &TransportedCounterfactualRequestWire,
    identity: &TransportedCounterfactualIdentity,
) -> Result<LoweredProgram, CausalError> {
    let query = digest("spec", &identity.spec_id);
    let scientific = ScientificProblem {
        runtime_query: None,
        identity_scope: ProgramIdentityScope::Semantic,
        target: ProblemTarget::QueryFunctional {
            kind: FunctionalKind::PathSpecific,
            outcome: None,
            query,
        },
        plan_id: Arc::from(PLAN_ID),
        data: DataClassification::MultiEnvironment,
        graph_class: "Dag",
        query_variables: Arc::from(Vec::new()),
        source_digest: Some(digest("evidence", &identity.evidence_digest)),
        support_coordinate: None,
        supported: false,
        support_status: None,
    };
    let identification = IdentificationLayer::NonGraphical(Arc::new(
        NonGraphicalIdentification::SpecifiedModelConditional {
            model: digest("model", &identity.model_digest),
        },
    ));
    let design = DesignShape::not_design_based(
        "supplied affine additive-noise model; finite source and target covariate laws",
    );
    let statistical = StatisticalProgram {
        identifier: None,
        estimator: None,
        realization: Realization::ExactPoint { evaluator: Arc::from(PLAN_ID) },
        models: Arc::from(Vec::new()),
        folds: None,
        resampling: None,
        posterior: None,
        design,
        operators: Arc::from(Vec::new()),
    };
    let evidence = EvidenceDecisionProgram {
        sources: Arc::from(vec![
            EvidenceSourceRef {
                role: EvidenceRole::TransportSource,
                digest: Some(digest("source_law", &identity.source_law_digest)),
            },
            EvidenceSourceRef {
                role: EvidenceRole::Observations,
                digest: Some(digest("target_law", &identity.target_law_digest)),
            },
        ]),
        composition: Composition::Single,
        decision: None,
        planning: None,
        chain: None,
    };
    let points = request.source_law.points.len() + request.target_law.points.len();
    let output = (points as u64)
        .checked_mul((request.model.mechanisms.len() as u64).saturating_add(1))
        .and_then(|cells| cells.checked_mul(8))
        .ok_or_else(|| error("request size overflow"))?;
    let read = PhysicalOperator::declare(PhysicalEffect::Read)
        .estimates(ByteEstimates::absent().with_output(output))
        .build()
        .map_err(error)?;
    let physical = PhysicalLayer { operators: Arc::from(vec![read]), design };
    let claim = ClaimDemand { uncertainty: UncertaintyReading::None, kind: None };
    Ok(TransportedPathSpecificLoweredParts {
        scientific,
        identification,
        statistical,
        evidence,
        physical,
        claim,
    }
    .finish())
}

#[cfg(test)]
mod site_tests {
    use super::digest;
    use crate::identity_laws::{ADVERSARIAL_TEXT, all_distinct, injective};

    #[test]
    fn site_transported_path_specific_component_digest_frames_role_and_value() {
        injective(&ADVERSARIAL_TEXT, |pair| *pair, |pair| Some(digest(pair.0, pair.1)));
        // One sealed hex value read in two roles is two different digests.
        let value = "00ff";
        all_distinct(&[
            digest("spec", value),
            digest("evidence", value),
            digest("model", value),
            digest("source_law", value),
            digest("target_law", value),
        ]);
    }
}
