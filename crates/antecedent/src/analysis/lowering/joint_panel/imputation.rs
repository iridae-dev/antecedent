//! Lower a multiple imputation over a joint panel source: completed panels drawn under the
//! declared imputation model, each fitted as an observed panel, and their responses pooled.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_contract::{ClaimKind, DataClassification};
use antecedent_core::{
    ByteEstimates, CanonicalWriter, PhysicalEffect, PhysicalOperator, SemanticDigest,
};
use antecedent_data::{ImputationDomain, ImputationSpec};
use antecedent_estimate::joint_panel::PanelInference;

use super::design::{read_and_fit, refuse};
use crate::analysis::joint_panel::JointPanelRequest;
use crate::analysis::joint_panel::compiled::support_status;
use crate::error::CausalError;
use crate::planner::lowered::{
    ClaimDemand, Composition, DesignShape, EvidenceDecisionProgram, EvidenceRole,
    EvidenceSourceRef, IdentificationLayer, JointPanelLoweredParts, JointPanelRoute,
    LoweredProgram, NonGraphicalIdentification, PhysicalLayer, ProblemTarget, ProgramIdentityScope,
    ScientificProblem, UncertaintyReading,
};
use crate::planner::statistical::{ModelRole, ModelSlot, Realization, StatisticalProgram};
use crate::support::CellStatus;

/// What an imputation program fits instead of one quoted design matrix.
const DESIGN_REASON: &str = "chained-equation imputation and one panel fit per completed table";

/// Lower the program of a prepared joint-panel multiple imputation (`JointPanelImputation`).
///
/// The program wraps one child `JointPanelResponse` program per completed panel; each child
/// lowers and executes inside this program's dispatch. The six layers:
///
/// 1. the request's scientific target over panel data;
/// 2. identification conditional on the specified panel model and the declared imputation
///    model;
/// 3. the pooled realization: Rubin's rules over frequentist fits, the stacked posterior of
///    Bayesian fits, over `spec.imputations` completed panels;
/// 4. the incomplete panel source as observations;
/// 5. the read of `source_bytes`, the imputation draws, one fit per completed panel and the
///    pooling;
/// 6. the response claim.
///
/// `source` is the identity of the incomplete panel source. `spec` and `domains` are the
/// declared imputation model: tables, sweeps, seed, missingness assumption and the explicit
/// variable domains.
///
/// # Errors
/// A program layer the planner refuses to declare.
pub(crate) fn lower(
    request: &JointPanelRequest,
    spec: &ImputationSpec,
    domains: &[(Arc<str>, ImputationDomain)],
    source: SemanticDigest,
    source_bytes: u64,
) -> Result<LoweredProgram, CausalError> {
    let model = declaration_digest(request, spec, domains, source)?;
    let status = support_status(JointPanelRoute::Imputation);
    let bayesian = matches!(request.options.inference, PanelInference::Bayesian { .. });
    let scientific = ScientificProblem {
        runtime_query: None,
        identity_scope: ProgramIdentityScope::Semantic,
        target: ProblemTarget::Target(request.target.clone()),
        plan_id: Arc::from("joint_panel_imputation"),
        data: DataClassification::Panel,
        graph_class: "joint_panel_model",
        query_variables: Arc::from([]),
        source_digest: Some(*source.as_bytes()),
        support_coordinate: None,
        supported: status == CellStatus::Supported,
        support_status: Some(status),
    };
    let identification = IdentificationLayer::NonGraphical(Arc::new(
        NonGraphicalIdentification::SpecifiedModelConditional { model },
    ));
    let shape = DesignShape::not_design_based(DESIGN_REASON);
    let realization = if bayesian {
        Realization::DeclaredPosterior { evaluator: Arc::from("joint_panel_imputation_stacked_v1") }
    } else {
        Realization::ExactPoint { evaluator: Arc::from("joint_panel_imputation_rubin_v1") }
    };
    let statistical = StatisticalProgram {
        identifier: None,
        estimator: None,
        realization,
        models: Arc::from([
            ModelSlot {
                role: ModelRole::Nuisance,
                family: Arc::from("native.chained_equations"),
                inputs: Arc::from([]),
                output: None,
            },
            ModelSlot {
                role: ModelRole::Outcome,
                family: Arc::from("joint_panel.response"),
                inputs: Arc::from([]),
                output: None,
            },
        ]),
        folds: None,
        resampling: None,
        posterior: None,
        design: shape,
        operators: Arc::from(read_and_fit()?),
    };
    let evidence = EvidenceDecisionProgram {
        sources: Arc::from([EvidenceSourceRef {
            role: EvidenceRole::Observations,
            digest: Some(*source.as_bytes()),
        }]),
        composition: Composition::Chained { stages: 2 },
        decision: None,
        planning: None,
        chain: None,
    };
    let read = PhysicalOperator::declare(PhysicalEffect::Read)
        .estimates(ByteEstimates::absent().with_output(source_bytes))
        .build()
        .map_err(refuse)?;
    let fit = PhysicalOperator::declare(PhysicalEffect::Fit)
        .estimates(
            ByteEstimates::absent().with_scratch(source_bytes.saturating_mul(2)).with_retained(
                source_bytes.saturating_mul(u64::from(spec.imputations).saturating_add(1)),
            ),
        )
        .build()
        .map_err(refuse)?;
    let physical = PhysicalLayer { operators: Arc::from([read, fit]), design: shape };
    let claim = ClaimDemand {
        uncertainty: if bayesian {
            UncertaintyReading::Posterior
        } else {
            UncertaintyReading::Sampling
        },
        kind: Some(ClaimKind::Response),
    };
    Ok(JointPanelLoweredParts {
        route: JointPanelRoute::Imputation,
        scientific,
        identification,
        statistical,
        evidence,
        physical,
        claim,
    }
    .finish())
}

/// Digest of everything the pooled response is conditional on: the incomplete source, the
/// panel request and the declared imputation model.
fn declaration_digest(
    request: &JointPanelRequest,
    spec: &ImputationSpec,
    domains: &[(Arc<str>, ImputationDomain)],
    source: SemanticDigest,
) -> Result<[u8; 32], CausalError> {
    let mut writer = CanonicalWriter::new("antecedent.lowering.joint_panel.imputation.v1");
    writer
        .bytes(source.as_bytes())
        .text(spec.model.name())
        .word(u64::from(spec.imputations))
        .word(u64::from(spec.iterations))
        .word(spec.seed)
        .text(spec.assumption.name())
        .word(u64::from(spec.bootstrap_replicates))
        .word(domains.len() as u64);
    for (name, domain) in domains {
        writer.text(name).text(domain.name());
    }
    let columns = &request.columns;
    writer.word(columns.channels.len() as u64);
    for channel in &columns.channels {
        writer.word(u64::from(channel.raw()));
    }
    writer
        .word(u64::from(columns.outcome.raw()))
        .word(u64::from(columns.confounder.raw()))
        .word(u64::from(columns.season.raw()))
        .word(request.seasons as u64)
        .word(request.lag_order as u64)
        .bytes(request.target.identity().as_bytes())
        .word(request.worlds.len() as u64);
    for world in &request.worlds {
        writer.text(&world.program.id).word(world.horizon() as u64);
    }
    for encoded in [
        antecedent_io::convert::to_cbor(&request.options).map_err(refuse)?,
        antecedent_io::convert::to_cbor(&request.family).map_err(refuse)?,
        antecedent_io::convert::to_cbor(&request.interactions).map_err(refuse)?,
    ] {
        writer.bytes(&encoded);
    }
    Ok(writer.finish())
}
