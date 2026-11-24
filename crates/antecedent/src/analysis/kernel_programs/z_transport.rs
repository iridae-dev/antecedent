//! Exact and empirical single-source z-transport as a compiled program family.
//!
//! A checked z-transport formula (a functional bound to the evidence catalog by
//! `antecedent_identify::bind_z_transport_catalog`), the source laws it cites, the treatment
//! request and the evaluation limits are everything `StudyBuilder::z_transport` and
//! `StudyBuilder::z_transport_empirical` take. Compiling prepares them exactly as those entries
//! do and binds the digest of the checked derivation, the catalog, every law's content identity,
//! the request and the limits into a six-layer program under the `ZTransportEstimate` route.
//! Every estimate then calls the very `PreparedZTransport` method the direct path calls (the
//! exact point, the percentile or posterior interval of an empirical table, and the two
//! combinations of separately prepared components), so the distribution, the interval and its
//! reason are the direct path's bit for bit.
//!
//! The route's unit (`transport.single_source_z_transport`) has not yet earned its support, so
//! the estimates pass its support gate only once `GATE_ON_UNIT_SUPPORT` is turned on; until
//! then no valid use refuses.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::DataClassification;
use antecedent_core::{CanonicalEncode, CanonicalWriter, ExecutionContext};
use antecedent_estimate::BayesianTransportLawProvider;
use antecedent_expr::{Assignment, ExactEvaluationLimits, ExactTransportData};
use antecedent_graph::SelectionDiagram;
use antecedent_identify::BoundZTransportFunctional;
use antecedent_io::IoError;
use antecedent_io::expr_arena_to_wire;
use antecedent_io::transport_query_wire::ZTransportQueryWire;

use super::{KernelProgram, KernelProgramShape, KernelRealization, io_refusal};
use crate::analysis::supplied_identity::{Canonical, SetOf, Wire};
use crate::analysis::{PreparedZTransport, Study, StudyBuilder, ZTransportResult};
use crate::error::CausalError;
use crate::planner::PlanView;
use crate::planner::lowered::{
    EvidenceRole, FunctionalKind, LoweredProgram, ProgramCell, UncertaintyReading,
};

/// Whether the estimates pass the support gate of the unit the route is bound to
/// (`parity/compiled_units.toml`). It stays off until that unit's record has earned its support,
/// so no valid use refuses before then; turning it on makes a release build refuse the route
/// while the unit is unsupported and a development build record what it would refuse.
const GATE_ON_UNIT_SUPPORT: bool = false;

/// Exact laws: the identified target law, point only.
const EXACT: KernelProgramShape = KernelProgramShape {
    cell: ProgramCell::ZTransportEstimate,
    plan_id: "transport.z_transport.exact",
    functional: FunctionalKind::InterventionalDistribution,
    evaluator: "checked_z_transport_formula_exact_evaluation",
    realization: KernelRealization::Exact,
    uncertainty: UncertaintyReading::None,
    role: EvidenceRole::TransportSource,
    graph_class: "selection_admg",
    data: DataClassification::MultiEnvironment,
    design: "checked single-source z-transport formula evaluated over the source laws it cites",
};

/// Empirical tables: the plug-in point with a nominal resampling or posterior interval whose
/// coverage is not measured.
const EMPIRICAL: KernelProgramShape = KernelProgramShape {
    cell: ProgramCell::ZTransportEstimate,
    plan_id: "transport.z_transport.empirical_plugin",
    functional: FunctionalKind::InterventionalDistribution,
    evaluator: "checked_z_transport_formula_empirical_plugin",
    realization: KernelRealization::Exact,
    uncertainty: UncertaintyReading::Sampling,
    role: EvidenceRole::TransportSource,
    graph_class: "selection_admg",
    data: DataClassification::MultiEnvironment,
    design: "checked single-source z-transport formula evaluated over counted source tables",
};

/// The prepared formula with the request and limits it was prepared under, so a refresh
/// compiles again under the same identity inputs.
#[derive(Clone, Debug)]
struct ZTransportInput {
    prepared: PreparedZTransport,
    request: Assignment,
    limits: ExactEvaluationLimits,
    empirical: bool,
}

/// A compiled single-source z-transport: the prepared checked formula and the program that
/// binds it.
#[derive(Clone, Debug)]
pub struct CompiledZTransport {
    program: KernelProgram<ZTransportInput>,
}

/// The digest of the checked selection diagram, the checked derivation (its record, its query
/// and its expression arena), the provider-bound formula (its arena and root), the catalog the
/// formula is bound to, the cited regimes, every law's content identity, the request and the
/// limits. The graph enters by its canonical identity (sorted edge lists) and the selection
/// targets as a set; the catalog by its own canonical encoding, whose environments, regimes and
/// bindings are sets; the derivation's record, query and arenas by their wire forms, whose
/// arenas list nodes in their interned order; the request in the order the assignment keeps it
/// (sorted by variable), each value by its canonical encoding. A wire that does not encode
/// writes its error under its own tag.
fn canonical(
    prepared: &PreparedZTransport,
    request: &Assignment,
    limits: ExactEvaluationLimits,
    empirical: bool,
) -> [u8; 32] {
    let functional = prepared.functional();
    let derivation = functional.derivation();
    let mut writer = CanonicalWriter::new("antecedent.z_transport.compiled_input.v2");
    writer.text(if empirical { "empirical" } else { "exact" });
    let diagram = prepared.diagram();
    Canonical::encode(diagram.causal_graph(), &mut writer);
    // A set: identification reads the selection targets as a node set.
    Canonical::encode(&SetOf(diagram.selection_targets()), &mut writer);
    Canonical::encode(&Wire(&derivation.to_record()), &mut writer);
    Canonical::encode(&Wire(&ZTransportQueryWire::from_query(derivation.query())), &mut writer);
    write_arena(&mut writer, derivation.arena());
    write_arena(&mut writer, functional.arena());
    writer.word(u64::from(functional.root().raw()));
    CanonicalEncode::encode(functional.catalog(), &mut writer);
    writer
        .words(&functional.cited_regimes().iter().map(|r| u64::from(r.raw())).collect::<Vec<_>>());
    let data = prepared.data();
    writer.word(data.laws().len() as u64);
    for law in data.laws() {
        writer.bytes(law.content_identity().as_bytes());
    }
    writer.word(request.entries().len() as u64);
    for (variable, value) in request.entries() {
        CanonicalEncode::encode(variable, &mut writer);
        CanonicalEncode::encode(value, &mut writer);
    }
    writer.word(limits.operations as u64).word(limits.depth as u64);
    writer.finish()
}

/// An expression arena by its wire form, or its conversion error under its own tag.
fn write_arena(writer: &mut CanonicalWriter, arena: &antecedent_expr::CausalExprArena) {
    match expr_arena_to_wire(arena) {
        Ok(wire) => Canonical::encode(&Wire(&wire), writer),
        Err(error) => {
            writer.text("unencodable").text(&error.to_string());
        }
    }
}

fn compile_refusal(error: &CausalError) -> IoError {
    IoError::Convert(format!("z_transport.compile: {error}"))
}

fn compile(
    prepared: PreparedZTransport,
    request: Assignment,
    limits: ExactEvaluationLimits,
    empirical: bool,
    ctx: &ExecutionContext,
) -> Result<CompiledZTransport, IoError> {
    antecedent_estimate::refuse_cancelled(ctx, "z-transport compile").map_err(IoError::from)?;
    let bytes = canonical(&prepared, &request, limits, empirical);
    let shape = if empirical { &EMPIRICAL } else { &EXACT };
    let input = ZTransportInput { prepared, request, limits, empirical };
    let program = KernelProgram::compile(shape, input, &bytes, ctx)
        .map_err(|error| compile_refusal(&error))?;
    Ok(CompiledZTransport { program })
}

impl Study {
    /// Prepare a checked z-transport formula over exact laws or empirical tables, as
    /// `StudyBuilder::z_transport` does, and compile it under the `ZTransportEstimate` route.
    /// Compiling evaluates nothing.
    ///
    /// # Errors
    ///
    /// The refusals of `StudyBuilder::z_transport`, compiling from inside a lowered dispatch,
    /// or cancellation.
    pub fn compile_z_transport(
        diagram: SelectionDiagram,
        functional: BoundZTransportFunctional,
        data: ExactTransportData,
        request: Assignment,
        limits: ExactEvaluationLimits,
        ctx: &ExecutionContext,
    ) -> Result<CompiledZTransport, IoError> {
        let prepared =
            StudyBuilder::z_transport(diagram, functional, data, request.clone(), limits, ctx)?;
        compile(prepared, request, limits, false, ctx)
    }

    /// Prepare an empirical plug-in evaluation whose every law carries counts, as
    /// `StudyBuilder::z_transport_empirical` does, and compile it under the `ZTransportEstimate`
    /// route. Compiling evaluates nothing.
    ///
    /// # Errors
    ///
    /// The refusals of `StudyBuilder::z_transport_empirical`, compiling from inside a lowered
    /// dispatch, or cancellation.
    pub fn compile_z_transport_empirical(
        diagram: SelectionDiagram,
        functional: BoundZTransportFunctional,
        data: ExactTransportData,
        request: Assignment,
        limits: ExactEvaluationLimits,
        ctx: &ExecutionContext,
    ) -> Result<CompiledZTransport, IoError> {
        let prepared = StudyBuilder::z_transport_empirical(
            diagram,
            functional,
            data,
            request.clone(),
            limits,
            ctx,
        )?;
        compile(prepared, request, limits, true, ctx)
    }
}

impl CompiledZTransport {
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

    /// The prepared checked formula the program binds, for export, sensitivity and inspection.
    #[must_use]
    pub fn prepared(&self) -> &PreparedZTransport {
        &self.program.input().prepared
    }

    fn gate(&self, ctx: &ExecutionContext) -> Result<(), IoError> {
        if GATE_ON_UNIT_SUPPORT {
            self.program.gate(ctx).map_err(|refusal| io_refusal(&refusal))?;
        }
        Ok(())
    }

    /// Pass the support gate and evaluate the frozen formula, as `PreparedZTransport::estimate`
    /// does: exact laws stay point-only, and counted tables attach the nominal percentile
    /// bootstrap.
    ///
    /// # Errors
    ///
    /// The support refusal (`not_supported`) once the gate is on, in a release build, while the
    /// unit is not supported, and the refusals of the evaluation.
    pub fn estimate(&self, ctx: &ExecutionContext) -> Result<ZTransportResult, IoError> {
        self.gate(ctx)?;
        self.prepared().estimate_with(None, 0, ctx)
    }

    /// Pass the support gate and attach a finite-discrete posterior interval instead of the
    /// percentile bootstrap, as `PreparedZTransport::estimate_bayesian` does.
    ///
    /// # Errors
    ///
    /// As [`Self::estimate`].
    pub fn estimate_bayesian(
        &self,
        provider: BayesianTransportLawProvider,
        draws: u32,
        ctx: &ExecutionContext,
    ) -> Result<ZTransportResult, IoError> {
        self.gate(ctx)?;
        self.prepared().estimate_with(Some(provider), draws, ctx)
    }

    /// Pass the support gate of both programs and combine two separately compiled formulas
    /// whose outcome and treatment sets lie in disconnected components of the checked graph, as
    /// `PreparedZTransport::estimate_independent_components` does.
    ///
    /// # Errors
    ///
    /// As [`Self::estimate`], and the refusals of the combination.
    pub fn estimate_independent_components(
        left: &Self,
        right: &Self,
        max_atoms: usize,
        ctx: &ExecutionContext,
    ) -> Result<ZTransportResult, IoError> {
        left.gate(ctx)?;
        right.gate(ctx)?;
        PreparedZTransport::combine_independent_components(
            left.prepared(),
            right.prepared(),
            max_atoms,
            ctx,
        )
    }

    /// Pass the support gate of both programs and combine two separately compiled factors whose
    /// outcomes m-separate given the union of their treatments, as
    /// `PreparedZTransport::estimate_intervention_factorized_components` does.
    ///
    /// # Errors
    ///
    /// As [`Self::estimate`], and the refusals of the combination.
    pub fn estimate_intervention_factorized_components(
        left: &Self,
        right: &Self,
        max_atoms: usize,
        ctx: &ExecutionContext,
    ) -> Result<ZTransportResult, IoError> {
        left.gate(ctx)?;
        right.gate(ctx)?;
        PreparedZTransport::combine_factorized_components(
            left.prepared(),
            right.prepared(),
            max_atoms,
            ctx,
        )
    }

    /// Rebind fresh laws under the retained checked program, as `PreparedZTransport::refresh`
    /// does, and compile the refreshed formula under the same route.
    ///
    /// # Errors
    ///
    /// The refusals of `PreparedZTransport::refresh`, or of compiling.
    pub fn refresh(
        &self,
        data: ExactTransportData,
        ctx: &ExecutionContext,
    ) -> Result<Self, IoError> {
        let input = self.program.input();
        let prepared = input.prepared.refresh(data, ctx)?;
        compile(prepared, input.request.clone(), input.limits, input.empirical, ctx)
    }
}

#[cfg(test)]
mod identity_tests {
    use antecedent_core::{Value, VariableId};

    use super::*;
    use crate::analysis::z_transport::tests::fixture;

    fn prepared() -> (PreparedZTransport, Assignment) {
        let ctx = ExecutionContext::for_tests(0);
        let (diagram, functional, data, request) = fixture(false);
        let limits = ExactEvaluationLimits::default();
        let prepared =
            StudyBuilder::z_transport(diagram, functional, data, request.clone(), limits, &ctx)
                .unwrap();
        (prepared, request)
    }

    #[test]
    fn the_z_transport_input_identity_is_canonical() {
        let limits = ExactEvaluationLimits::default();
        let (first, request) = prepared();
        let (second, _) = prepared();
        let base = canonical(&first, &request, limits, false);
        assert_eq!(base, canonical(&second, &request, limits, false), "repeatable");
        assert_ne!(base, canonical(&first, &request, limits, true), "a changed provider kind");
        let tighter = ExactEvaluationLimits { operations: limits.operations - 1, ..limits };
        assert_ne!(base, canonical(&first, &request, tighter, false), "changed limits");
        let level = |value: f64| {
            let request = Assignment::from_pairs([(VariableId::from_raw(2), Value::f64(value))]);
            canonical(&first, &request, limits, false)
        };
        assert_ne!(level(0.0), level(-0.0), "reals by bits");
        assert_ne!(base, level(0.0), "a changed request value");
    }
}
