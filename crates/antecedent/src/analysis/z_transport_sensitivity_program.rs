//! The compiled outcome-kernel contamination range of a checked z-transport formula.
//!
//! [`compile_z_transport_mechanism_sensitivity`] compiles the one-factor sensitivity analysis of
//! the registered surrogate z-transport formula `sum_w P(Y | w, X, do(Z)) P(w | do(Z))`: the
//! selection diagram, the catalog-bound functional and the exact transport data a prepared
//! z-transport stage holds, the contamination bound and the decision threshold. It is a program of
//! the `ZTransportMechanismSensitivity` family on the supplied-terminal lowering, with a route,
//! the support unit `sensitivity.joint_mechanism` read from `parity/compiled_units.toml`, a program
//! identity over every declared input and the support gate at execution. Execution calls
//! `z_transport_mechanism_sensitivity`, the kernel the direct path calls, on the same inputs, so
//! every number is the direct path's; a kernel refusal is returned unchanged as
//! [`TerminalError::Kernel`].
//!
//! The identity reads the diagram through io-core's transport identity wire (canonical graph,
//! sorted selection targets), the functional through its derivation record, its query wire, its
//! provider-bound expression arena wire and root, its canonical catalog wire and its cited regimes
//! in order, and the data through the content identity of each law in order and the support
//! bound. Nothing is rendered through `Debug`.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::ClaimKind;
use antecedent_core::{CanonicalWriter, ExecutionContext};
use antecedent_expr::ExactTransportData;
use antecedent_graph::SelectionDiagram;
use antecedent_identify::BoundZTransportFunctional;
use antecedent_io_core::expr_wire::expr_arena_to_wire;
use antecedent_io_core::transport_catalog_wire::EvidenceCatalogWire;
use antecedent_io_core::transport_query_wire::ZTransportQueryWire;
use antecedent_validate::{
    ZTransportMechanismSensitivityResult, ZTransportSensitivityError,
    z_transport_mechanism_sensitivity,
};

use crate::analysis::supplied_identity::{Canonical, Declaration, Wire, variant};
pub use crate::analysis::supplied_terminal::TerminalError;
use crate::analysis::supplied_terminal::{
    Reading, TerminalSpec, admit_compile, compiled_terminal, lower,
};
use crate::error::CausalError;
use crate::planner::lowered::{Composition, EvidenceRole, FunctionalKind, ProgramCell};

compiled_terminal! {
    /// A compiled outcome-kernel contamination range of a checked z-transport formula.
    #[derive(Clone, Debug)]
    CompiledZTransportMechanismSensitivity {
        diagram: SelectionDiagram,
        functional: BoundZTransportFunctional,
        data: ExactTransportData,
        max_fraction: f64,
        decision_threshold: Option<f64>,
    }
}

/// Compile the contamination range of the checked z-transport `functional` on `data`, the
/// outcome kernel contaminated up to `max_fraction`, with the tipping fraction of
/// `decision_threshold` when one is declared.
///
/// Nothing is evaluated at compile; the program's identity covers every declared input.
///
/// # Errors
/// `public_builder_reentry` inside a lowered dispatch.
pub fn compile_z_transport_mechanism_sensitivity(
    diagram: SelectionDiagram,
    functional: BoundZTransportFunctional,
    data: ExactTransportData,
    max_fraction: f64,
    decision_threshold: Option<f64>,
    ctx: &ExecutionContext,
) -> Result<CompiledZTransportMechanismSensitivity, CausalError> {
    admit_compile(ctx)?;
    let question = Declaration::new("antecedent.z_transport_mechanism_sensitivity.v2")
        .field("diagram", &DiagramField(&diagram))
        .field("functional", &FunctionalField(&functional))
        .field("data", &DataField(&data))
        .field("max_fraction", &max_fraction)
        .field("decision_threshold", &decision_threshold)
        .finish();
    let lowered = lower(TerminalSpec {
        cell: ProgramCell::ZTransportMechanismSensitivity,
        plan_id: "supplied.z_transport_mechanism_sensitivity.v1",
        functional: FunctionalKind::IdentifiedSet,
        graph_class: "selection_diagram",
        question,
        evaluator: "mechanism_sensitivity::z_transport_mechanism_sensitivity",
        reading: Reading::Envelope,
        role: EvidenceRole::TransportSource,
        composition: Composition::Transported,
        decision: None,
        claim: Some(ClaimKind::Bounds),
    })?;
    Ok(CompiledZTransportMechanismSensitivity {
        lowered,
        diagram,
        functional,
        data,
        max_fraction,
        decision_threshold,
    })
}

impl CompiledZTransportMechanismSensitivity {
    /// The declared contamination bound.
    #[must_use]
    pub const fn max_fraction(&self) -> f64 {
        self.max_fraction
    }

    /// The declared decision threshold.
    #[must_use]
    pub const fn decision_threshold(&self) -> Option<f64> {
        self.decision_threshold
    }

    /// The baseline contrast, the exact contamination range, the tipping fraction and the
    /// optimization receipt.
    ///
    /// # Errors
    /// The kernel's own refusals (a formula other than the registered surrogate, an outcome
    /// domain that is not numeric, an incomplete kernel or parent marginal, an invalid fraction or
    /// threshold, cancellation), or the compiler's.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<ZTransportMechanismSensitivityResult, TerminalError<ZTransportSensitivityError>>
    {
        let scoped = self.enter(ctx)?;
        z_transport_mechanism_sensitivity(
            &self.diagram,
            &self.functional,
            &self.data,
            self.max_fraction,
            self.decision_threshold,
            &scoped,
        )
        .map_err(TerminalError::Kernel)
    }
}

// --- canonical identity of the declared inputs -----------------------------------------------

/// A selection diagram by io-core's transport identity wire.
struct DiagramField<'a>(&'a SelectionDiagram);

impl Canonical for DiagramField<'_> {
    fn encode(&self, out: &mut CanonicalWriter) {
        match antecedent_io_core::transport_identity(Some(self.0), None) {
            Ok(wire) => Wire(&wire).encode(out),
            Err(error) => {
                out.text("unencodable").text(&error.to_string());
            }
        }
        // The wire indexes dense nodes; each dense node's label names its variable.
        self.0.causal_graph().nodes().encode(out);
    }
}

/// A catalog-bound z functional: its derivation record and query wire, its provider-bound arena
/// and root, its catalog (canonical when it canonicalizes) and its cited regimes in order.
struct FunctionalField<'a>(&'a BoundZTransportFunctional);

impl Canonical for FunctionalField<'_> {
    fn encode(&self, out: &mut CanonicalWriter) {
        let functional = self.0;
        Wire(&functional.derivation().to_record()).encode(out);
        Wire(&ZTransportQueryWire::from_query(functional.derivation().query())).encode(out);
        match expr_arena_to_wire(functional.arena()) {
            Ok(wire) => Wire(&wire).encode(out),
            Err(error) => {
                out.text("unencodable").text(&error.to_string());
            }
        }
        out.word(u64::from(functional.root().raw()));
        if let Ok(canonical) = functional.catalog().canonicalized() {
            variant(out, "canonical");
            Wire(&EvidenceCatalogWire::from_catalog(&canonical)).encode(out);
        } else {
            variant(out, "declared");
            Wire(&EvidenceCatalogWire::from_catalog(functional.catalog())).encode(out);
        }
        let cited: Vec<u64> =
            functional.cited_regimes().iter().map(|regime| u64::from(regime.raw())).collect();
        cited.encode(out);
    }
}

/// Exact transport data: the content identity of every law in order, and the support bound.
struct DataField<'a>(&'a ExactTransportData);

impl Canonical for DataField<'_> {
    fn encode(&self, out: &mut CanonicalWriter) {
        let identities = self.0.law_identities();
        out.word(identities.len() as u64);
        for identity in identities {
            out.bytes(identity.as_bytes());
        }
        out.word(self.0.max_support_rows() as u64);
    }
}
