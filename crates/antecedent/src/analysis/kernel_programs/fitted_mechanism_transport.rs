//! Fitted linear-Gaussian mechanism transport as a compiled program family.
//!
//! `transport_linear_gaussian` takes no execution context. Compiling binds the request and uses
//! the context only for the support gate's program and for cancellation. Executing passes the
//! `transport.fitted_linear_gaussian_mechanism` gate and calls `transport_linear_gaussian` with
//! the same request.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::DataClassification;
use antecedent_core::{CanonicalWriter, ExecutionContext, NotSupported};
use antecedent_model::compose::{
    ComposedModelSpec, CompositionError, InvarianceDeclaration, MechanismDescriptor,
};
use antecedent_model::mechanism_transport::{
    MechanismTransportRequest, TransportedLinearGaussian, transport_linear_gaussian,
};

use super::discretized_bridge::{
    Multiset, encode_context, encode_dependence, encode_descriptor, encode_diagram,
    encode_invariance,
};
use super::{KernelProgram, KernelProgramShape, KernelRealization};
use crate::analysis::Study;
use crate::analysis::supplied_identity::{Canonical, Declaration, encoded};
use crate::error::CausalError;
use crate::planner::PlanView;
use crate::planner::lowered::{
    EvidenceRole, FunctionalKind, LoweredProgram, ProgramCell, UncertaintyReading,
};

const TAG: &str = "antecedent.kernel_program.fitted_mechanism_transport";

const SHAPE: KernelProgramShape = KernelProgramShape {
    cell: ProgramCell::FittedMechanismTransport,
    plan_id: "transport.fitted_linear_gaussian_mechanism",
    functional: FunctionalKind::InterventionalDistribution,
    evaluator: "fitted_linear_gaussian_mechanism_transport",
    realization: KernelRealization::Exact,
    uncertainty: UncertaintyReading::None,
    role: EvidenceRole::TransportSource,
    graph_class: "selection_diagram",
    data: DataClassification::MultiEnvironment,
    design: "carry fitted linear-Gaussian mechanisms under a declared selection diagram",
};

struct Descriptor<'a>(&'a MechanismDescriptor);

impl Canonical for Descriptor<'_> {
    fn encode(&self, out: &mut CanonicalWriter) {
        encode_descriptor(&self.0.canonical(), out);
    }
}

struct Invariance<'a>(&'a InvarianceDeclaration);

impl Canonical for Invariance<'_> {
    fn encode(&self, out: &mut CanonicalWriter) {
        encode_invariance(self.0, out);
    }
}

/// The request. Source and target mechanisms are maps keyed by variable (a duplicate variable
/// is a finding, so a repeated descriptor stays). Edges, invariances and the diagram's nodes
/// are the multisets [`ComposedModelSpec::canonical`] sorts without dropping a duplicate.
struct Request<'a>(&'a MechanismTransportRequest);

impl Canonical for Request<'_> {
    fn encode(&self, out: &mut CanonicalWriter) {
        let request = self.0;
        encode_context(&request.target, out);
        // The composition canonicalizer sorts the diagram and the dependence; the kernel's
        // identity reads that order and keeps duplicates.
        let canonical = ComposedModelSpec {
            target: request.target.clone(),
            mechanisms: Vec::new(),
            edges: Vec::new(),
            invariances: Vec::new(),
            selection_diagrams: vec![request.diagram.clone()],
            dependence: request.dependence.clone(),
            admit_declared_mechanisms: false,
        }
        .canonical();
        encode_diagram(&canonical.selection_diagrams[0], out);
        let source: Vec<Descriptor<'_>> =
            request.source_mechanisms.iter().map(Descriptor).collect();
        let target: Vec<Descriptor<'_>> =
            request.target_mechanisms.iter().map(Descriptor).collect();
        Multiset(&source).encode(out);
        Multiset(&target).encode(out);
        Multiset(&request.edges).encode(out);
        let invariances: Vec<Invariance<'_>> = request.invariances.iter().map(Invariance).collect();
        Multiset(&invariances).encode(out);
        encode_dependence(&canonical.dependence, out);
    }
}

fn request_bytes(request: &MechanismTransportRequest) -> Result<[u8; 32], CausalError> {
    encoded(SHAPE.plan_id, || {
        let mut declaration = Declaration::new(TAG);
        declaration.field("request", &Request(request));
        declaration.finish()
    })
}

fn support(refusal: &NotSupported) -> CompositionError {
    CompositionError::Unsupported { message: refusal.message().to_owned() }
}

/// A compiled fitted-mechanism transport.
#[derive(Clone, Debug)]
pub struct CompiledFittedMechanismTransport {
    program: KernelProgram<MechanismTransportRequest>,
}

impl CompiledFittedMechanismTransport {
    /// The compiled program's plan view, including the support decision of its unit.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        self.program.view()
    }

    /// Every layer of the compiled program.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram {
        self.program.program_layers()
    }

    /// The request the program binds.
    #[must_use]
    pub fn request(&self) -> &MechanismTransportRequest {
        self.program.input()
    }

    /// Pass the support gate, then call `transport_linear_gaussian` with the bound request.
    /// The kernel takes no context.
    ///
    /// # Errors
    /// The support refusal, and every refusal of `transport_linear_gaussian`.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<TransportedLinearGaussian, CompositionError> {
        self.program.gate(ctx).map_err(|refusal| support(&refusal))?;
        transport_linear_gaussian(self.program.input())
    }
}

impl Study {
    /// Compile `transport_linear_gaussian` for this request. The context is the gate's; the
    /// kernel is called later with the request alone. Compiling executes nothing.
    ///
    /// # Errors
    /// Cancellation, or compiling from inside a lowered dispatch.
    pub fn compile_fitted_mechanism_transport(
        request: &MechanismTransportRequest,
        ctx: &ExecutionContext,
    ) -> Result<CompiledFittedMechanismTransport, CausalError> {
        let bytes = request_bytes(request)?;
        let program = KernelProgram::compile(&SHAPE, request.clone(), &bytes, ctx)?;
        Ok(CompiledFittedMechanismTransport { program })
    }
}

#[cfg(test)]
mod site_tests {
    use super::*;
    use crate::identity_laws::{bit_exact, multiset_laws, ordered_laws};
    use antecedent_model::compose::{
        ContextCoordinate, Domain, MechanismContext, MechanismForm, MechanismOrigin,
        MechanismProvenance, ParameterId, ParameterTerm, PosteriorDependence,
        SelectionDiagramDeclaration, UncertaintyMeaning,
    };

    fn context(population: &str) -> MechanismContext {
        MechanismContext { population: population.to_owned(), regime: "observational".to_owned() }
    }

    fn mechanism(variable: &str, intercept: f64) -> MechanismDescriptor {
        MechanismDescriptor {
            variable: variable.to_owned(),
            form: MechanismForm::LinearGaussian { intercept, sigma: 1.0, coefficients: Vec::new() },
            parents: Vec::new(),
            conditioning_set: Vec::new(),
            output_units: "u".to_owned(),
            output_transform: "identity".to_owned(),
            output_support: Domain::Real,
            context: context("source"),
            provenance: MechanismProvenance {
                origin: MechanismOrigin::Fitted,
                artifact_id: format!("fit:{variable}"),
                evidence_id: format!("data:{variable}"),
            },
            latents: Vec::new(),
            uncertainty: None,
        }
    }

    fn request(
        edges: &[(String, String)],
        intercept: f64,
        dependence: PosteriorDependence,
    ) -> MechanismTransportRequest {
        MechanismTransportRequest {
            target: context("target"),
            diagram: SelectionDiagramDeclaration {
                source_population: "source".to_owned(),
                target_population: "target".to_owned(),
                selection_nodes: Vec::new(),
            },
            source_mechanisms: vec![mechanism("X", intercept), mechanism("Y", 0.0)],
            target_mechanisms: Vec::new(),
            edges: edges.to_vec(),
            invariances: vec![InvarianceDeclaration {
                variable: "Y".to_owned(),
                coordinate: ContextCoordinate::Population,
                source: "source".to_owned(),
                target: "target".to_owned(),
                assumption_id: "carry-y".to_owned(),
            }],
            dependence,
        }
    }

    /// `request_bytes`: edges are a multiset (canonical sort keeps a duplicate); a joint
    /// covariance is the ordered matrix in canonical parameter order; an intercept enters by
    /// its bits.
    #[test]
    fn site_fitted_mechanism_transport_edges_are_a_multiset_and_covariance_is_ordered() {
        let digest = |edges: &[(String, String)]| {
            request_bytes(&request(edges, 0.0, PosteriorDependence::Unknown)).ok()
        };
        multiset_laws(
            &[("X".to_owned(), "Y".to_owned()), ("Y".to_owned(), "Z".to_owned())],
            digest,
        );
        bit_exact(|intercept| {
            request_bytes(&request(&[], intercept, PosteriorDependence::Unknown)).ok()
        });
        let covariance_of = |covariance: &[f64]| {
            let dependence = PosteriorDependence::Joint {
                meaning: UncertaintyMeaning::Posterior,
                parameters: vec![
                    ParameterId { mechanism: "X".to_owned(), term: ParameterTerm::Intercept },
                    ParameterId {
                        mechanism: "Y".to_owned(),
                        term: ParameterTerm::Coefficient("X".to_owned()),
                    },
                ],
                covariance: covariance.to_vec(),
            };
            request_bytes(&request(&[], 0.0, dependence)).ok()
        };
        ordered_laws(&[1.0, 0.25, 0.25, 4.0], covariance_of);
    }
}
