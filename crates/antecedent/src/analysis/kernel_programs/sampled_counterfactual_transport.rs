//! The sampled counterfactual transport estimate and its measured interval as compiled program
//! families.
//!
//! A checked counterfactual transport derivation and a sampled request (count tables, the
//! declared overlap of their units, the unseen-cell rule and the limits) are everything the
//! sampled producer reads. Compiling them binds their digest into a six-layer program under the
//! `SampledCounterfactualTransport` route; estimating passes the support gate of that route's
//! unit and then calls `sampled_counterfactual_transport_artifact`, the call the direct path
//! makes, so the plug-in estimate, its standard error and the sealed artifact are the direct
//! path's.
//!
//! A sealed sampled artifact is everything its interval reads. Compiling its bytes binds them
//! under the `SampledCounterfactualTransportInterval` route; measuring passes the support gate
//! of that route's unit and then calls `SampledCounterfactualTransportArtifactWire::measure`:
//! the replay, the delta Wald interval and the binding of the interval to its coverage record
//! are the direct path's.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_contract::DataClassification;
use antecedent_core::{CanonicalWriter, ExecutionContext};
use antecedent_estimate::counterfactual_transport_sampled::{
    SampledCounterfactualTransportEstimate, SampledRegimeCounts, SampledTransportRequest,
};
use antecedent_identify::CounterfactualTransportDerivation;
use antecedent_io::IoError;
use antecedent_io::measured_inference::MeasuredInference;
use antecedent_io::sampled_counterfactual_transport_artifact::{
    SampledCountTableWire, SampledCounterfactualTransportArtifactError,
    SampledCounterfactualTransportArtifactWire, SampledCounterfactualTransportConsumeLimits,
    SampledCounterfactualTransportExpectedIdentity, SampledLimitsWire, SampledOverlapWire,
    SampledUnseenRuleWire, sampled_counterfactual_transport_artifact,
};
use antecedent_io_core::counterfactual_transport_wire::{
    CounterfactualTransportDerivationWire, CounterfactualTransportProblemWire,
    CounterfactualTransportQueryWire, CounterfactualTransportSearchLimitsWire,
};
use antecedent_io_core::transport_catalog_wire::EvidenceCatalogWire;
use serde::Serialize;

use super::{KernelProgram, KernelProgramShape, KernelRealization, canonical_cbor, io_refusal};
use crate::analysis::Study;
use crate::error::CausalError;
use crate::planner::PlanView;
use crate::planner::lowered::{
    EvidenceRole, FunctionalKind, LoweredProgram, ProgramCell, UncertaintyReading,
};

const SAMPLED: KernelProgramShape = KernelProgramShape {
    cell: ProgramCell::SampledCounterfactualTransport,
    plan_id: "counterfactual.transport_sampled",
    functional: FunctionalKind::Counterfactual,
    evaluator: "sampled_counterfactual_transport_plug_in",
    realization: KernelRealization::Score,
    uncertainty: UncertaintyReading::Sampling,
    role: EvidenceRole::TransportSource,
    graph_class: "admg_selection_diagram",
    data: DataClassification::MultiEnvironment,
    design: "sampled count tables of the cited regimes under a declared unit overlap",
};

const INTERVAL: KernelProgramShape = KernelProgramShape {
    cell: ProgramCell::SampledCounterfactualTransportInterval,
    plan_id: "counterfactual.transport_sampled_interval",
    functional: FunctionalKind::Counterfactual,
    evaluator: "sampled_counterfactual_transport_delta_wald",
    realization: KernelRealization::Score,
    uncertainty: UncertaintyReading::Sampling,
    role: EvidenceRole::TransportSource,
    graph_class: "admg_selection_diagram",
    data: DataClassification::MultiEnvironment,
    design: "replay of a sealed sampled estimate from independently drawn units",
};

/// The derivation and the sampled request of a compiled estimate.
#[derive(Clone, Debug)]
struct SampledInput {
    derivation: CounterfactualTransportDerivation,
    request: SampledTransportRequest,
    variable_names: Vec<String>,
}

/// The canonical identity of a sampled estimate's declaration, in the artifact's wire forms.
#[derive(Serialize)]
struct Canonical<'a> {
    problem: CounterfactualTransportProblemWire,
    query: CounterfactualTransportQueryWire,
    variable_names: &'a [String],
    search: CounterfactualTransportSearchLimitsWire,
    catalog: EvidenceCatalogWire,
    derivation: CounterfactualTransportDerivationWire,
    tables: Vec<SampledCountTableWire>,
    overlap: SampledOverlapWire,
    unseen_rule: SampledUnseenRuleWire,
    limits: SampledLimitsWire,
}

/// The sealed artifact bytes and the consumer's identity and bounds of a compiled interval.
#[derive(Clone, Debug)]
struct IntervalInput {
    artifact: Arc<[u8]>,
    expected: SampledCounterfactualTransportExpectedIdentity,
    limits: SampledCounterfactualTransportConsumeLimits,
}

/// The count tables in the kernel's own order: sorted by table key (population, regime and
/// intervention levels by their bits), then by content identity. The kernel sorts them the same
/// way before it estimates and refuses two tables of one key, so they are a set.
fn canonical_tables(tables: &[SampledRegimeCounts]) -> Vec<SampledCountTableWire> {
    let mut sorted: Vec<&SampledRegimeCounts> = tables.iter().collect();
    sorted.sort_by_cached_key(|table| (table.key(), table.content_identity()));
    sorted.into_iter().map(SampledCountTableWire::from_table).collect()
}

/// Canonical bytes of a compiled interval: the sealed artifact bytes, each expected digest the
/// replay checks (a presence word, then the text) and every consumer bound the replay enforces.
/// The replay refuses on a different expectation or a tighter bound, so both are part of the
/// question, as they are for the measured sampled-recovery replay.
fn interval_bytes(input: &IntervalInput) -> [u8; 32] {
    let mut writer =
        CanonicalWriter::new("antecedent.kernel_program.sampled_counterfactual_transport_interval");
    writer.bytes(&input.artifact);
    writer
        .opt_text(input.expected.premises_digest.as_deref())
        .opt_text(input.expected.data_digest.as_deref());
    let limits = &input.limits;
    writer
        .word(limits.search.operations as u64)
        .word(limits.search.depth as u64)
        .word(limits.search_memory_bytes)
        .word(limits.evaluation.operations as u64)
        .word(limits.evaluation.depth as u64)
        .word(limits.max_tables as u64)
        .word(limits.max_table_cells as u64);
    writer.finish()
}

/// A compiled sampled counterfactual transport estimate: the derivation, the request and the
/// program that binds them.
#[derive(Clone, Debug)]
pub struct CompiledSampledCounterfactualTransport {
    program: KernelProgram<SampledInput>,
}

/// A compiled interval of a sealed sampled counterfactual transport artifact: the bytes, the
/// consumer's expected identity and bounds, and the program that binds them.
#[derive(Clone, Debug)]
pub struct CompiledSampledCounterfactualTransportInterval {
    program: KernelProgram<IntervalInput>,
}

impl Study {
    /// Compile the sampled estimate of `derivation` on `request` under the
    /// `SampledCounterfactualTransport` route. `variable_names` are the names the artifact
    /// records, in ascending id order, or empty. Compiling executes nothing.
    ///
    /// # Errors
    ///
    /// Compiling from inside a lowered dispatch, cancellation, or a declaration that does not
    /// encode.
    pub fn compile_sampled_counterfactual_transport(
        derivation: CounterfactualTransportDerivation,
        request: SampledTransportRequest,
        variable_names: Vec<String>,
        ctx: &ExecutionContext,
    ) -> Result<CompiledSampledCounterfactualTransport, CausalError> {
        let canonical = canonical_cbor(&Canonical {
            problem: CounterfactualTransportProblemWire::from_problem(&derivation.problem),
            query: CounterfactualTransportQueryWire::from_query(&derivation.query),
            variable_names: &variable_names,
            search: CounterfactualTransportSearchLimitsWire {
                operations: derivation.search.operations_limit,
                depth: derivation.search.depth_limit,
                memory_limit_bytes: derivation.search.memory_limit_bytes,
            },
            catalog: EvidenceCatalogWire::from_catalog(&derivation.catalog),
            derivation: CounterfactualTransportDerivationWire::from_derivation(&derivation),
            tables: canonical_tables(&request.tables),
            overlap: SampledOverlapWire::from_declaration(&request.overlap),
            unseen_rule: SampledUnseenRuleWire::from_rule(request.unseen_rule),
            limits: SampledLimitsWire::from_limits(&request.limits),
        })?;
        let program = KernelProgram::compile(
            &SAMPLED,
            SampledInput { derivation, request, variable_names },
            &canonical,
            ctx,
        )?;
        Ok(CompiledSampledCounterfactualTransport { program })
    }

    /// Compile the interval of the sealed sampled artifact `artifact` under the
    /// `SampledCounterfactualTransportInterval` route; `expected` and `limits` are the
    /// consumer's identity and bounds for its replay. Compiling executes nothing.
    ///
    /// # Errors
    ///
    /// Compiling from inside a lowered dispatch, or cancellation.
    pub fn compile_sampled_counterfactual_transport_interval(
        artifact: &[u8],
        expected: SampledCounterfactualTransportExpectedIdentity,
        limits: SampledCounterfactualTransportConsumeLimits,
        ctx: &ExecutionContext,
    ) -> Result<CompiledSampledCounterfactualTransportInterval, CausalError> {
        let input = IntervalInput { artifact: Arc::from(artifact), expected, limits };
        let canonical = interval_bytes(&input);
        let program = KernelProgram::compile(&INTERVAL, input, &canonical, ctx)?;
        Ok(CompiledSampledCounterfactualTransportInterval { program })
    }
}

impl CompiledSampledCounterfactualTransport {
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

    /// Pass the support gate, estimate from the counts and seal the artifact, as
    /// `sampled_counterfactual_transport_artifact` does.
    ///
    /// # Errors
    ///
    /// The support refusal (reason `not_supported`) in a release build when the unit is not
    /// supported, and the refusals of the estimator and the encoder.
    pub fn estimate(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<
        (SampledCounterfactualTransportArtifactWire, SampledCounterfactualTransportEstimate),
        SampledCounterfactualTransportArtifactError,
    > {
        self.program.gate(ctx).map_err(|refusal| {
            SampledCounterfactualTransportArtifactError::Replay {
                code: refusal.code(),
                message: refusal.message().to_owned(),
            }
        })?;
        let input = self.program.input();
        sampled_counterfactual_transport_artifact(
            &input.derivation,
            &input.request,
            &input.variable_names,
            ctx,
        )
    }
}

impl CompiledSampledCounterfactualTransportInterval {
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

    /// Pass the support gate, replay the artifact and build the delta Wald interval annotated
    /// with the calibration status of exactly this construction, as
    /// `SampledCounterfactualTransportArtifactWire::measure` does.
    ///
    /// # Errors
    ///
    /// The support refusal (`not_supported`) in a release build when the unit is not supported,
    /// any replay refusal, or a design whose variance the delta method does not give
    /// (`route_not_supported`).
    pub fn measure(&self, ctx: &ExecutionContext) -> Result<MeasuredInference, IoError> {
        self.program.gate(ctx).map_err(|refusal| io_refusal(&refusal))?;
        let input = self.program.input();
        SampledCounterfactualTransportArtifactWire::measure(
            &input.artifact,
            &input.expected,
            input.limits,
            ctx,
        )
    }
}

#[cfg(test)]
mod site_tests {
    use antecedent_core::{RegimeId, SemanticDigest, VariableId};
    use antecedent_estimate::counterfactual_transport_sampled::{
        SampledAxis, SampledRegimeCountsParts, SamplingDesign,
    };

    use super::*;
    use crate::identity_laws::{all_distinct, set_laws};

    fn table(regime: u32, first: u64) -> SampledRegimeCounts {
        SampledRegimeCounts::try_new(SampledRegimeCountsParts {
            population: Arc::from("p"),
            regime: RegimeId::from_raw(regime),
            interventions: vec![],
            axes: vec![SampledAxis { variable: VariableId::from_raw(0), levels: vec![0.0, 1.0] }],
            counts: vec![first, 4],
            snapshot_identity: Arc::from("snapshot"),
            design: SamplingDesign::IndependentUnits,
            structural_zeros: vec![],
        })
        .unwrap()
    }

    /// The tables of `regimes`, or `None` for a repeated regime, which the kernel refuses (two
    /// tables of one key).
    fn tables(regimes: &[u32]) -> Option<Vec<u8>> {
        let mut seen = regimes.to_vec();
        seen.sort_unstable();
        seen.dedup();
        if seen.len() != regimes.len() {
            return None;
        }
        let tables: Vec<SampledRegimeCounts> = regimes.iter().map(|&r| table(r, 3)).collect();
        Some(canonical_cbor(&canonical_tables(&tables)).unwrap())
    }

    /// `compile_sampled_counterfactual_transport` (its `canonical_tables`): the count tables are
    /// a set, in the kernel's own (key, content identity) order; a changed count is another table.
    #[test]
    fn site_sampled_counterfactual_transport_tables_are_a_set() {
        set_laws(&[0_u32, 1, 2], tables);
        all_distinct(&[
            canonical_cbor(&canonical_tables(&[table(0, 3)])).unwrap(),
            canonical_cbor(&canonical_tables(&[table(0, 5)])).unwrap(),
        ]);
    }

    fn interval(
        artifact: &[u8],
        premises: Option<&str>,
        data: Option<&str>,
        max_tables: usize,
    ) -> SemanticDigest {
        let ctx = ExecutionContext::for_tests(0);
        let expected = SampledCounterfactualTransportExpectedIdentity {
            premises_digest: premises.map(str::to_owned),
            data_digest: data.map(str::to_owned),
        };
        let limits = SampledCounterfactualTransportConsumeLimits {
            max_tables,
            ..SampledCounterfactualTransportConsumeLimits::default()
        };
        Study::compile_sampled_counterfactual_transport_interval(artifact, expected, limits, &ctx)
            .unwrap()
            .lowered()
            .identity()
    }

    /// `compile_sampled_counterfactual_transport_interval` (its `interval_bytes`): the sealed
    /// bytes are framed, each expected digest the replay checks carries a presence word in its
    /// own position, and the consumer bounds are read.
    #[test]
    fn site_sampled_counterfactual_transport_interval_binds_the_expectation() {
        all_distinct(&[
            interval(b"artifact", None, None, 8),
            interval(b"artifact", Some("d"), None, 8),
            interval(b"artifact", None, Some("d"), 8),
            interval(b"artifact", Some(""), None, 8),
            interval(b"artifact", None, None, 9),
            interval(b"artifac", None, None, 8),
        ]);
    }
}
