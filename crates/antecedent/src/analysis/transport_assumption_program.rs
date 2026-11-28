//! Compiled assumption range of a transported point.
//!
//! [`compile_transport_assumption_range`] compiles the range a caller states over a
//! [`TransportSensitivitySpec`] into a `TransportAssumptionRange` program and executes
//! [`transport_assumption_range`] on that spec. The kernel takes no execution context: the
//! context is used to compile and to pass the support gate.
//!
//! A refusal of the kernel is [`TerminalError::Kernel`]; the compiler's own refusals are
//! [`TerminalError::Refused`].
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::ClaimKind;
use antecedent_core::{CanonicalWriter, ExecutionContext};
use antecedent_design::transport_sensitivity::{
    TransportAssumptionRange, TransportSensitivityRefusal, TransportSensitivitySpec,
    ViolationFamily, transport_assumption_range,
};

use crate::analysis::supplied_identity::{Canonical, Declaration, encoded, unencodable, variant};
use crate::analysis::supplied_terminal::{
    Reading, TerminalError, TerminalSpec, admit_compile, compiled_terminal, lower,
};
use crate::error::CausalError;
use crate::planner::lowered::{Composition, EvidenceRole, FunctionalKind, ProgramCell};

/// Violation family by its tag and the bound's bits. An unknown variant refuses the compile.
struct Family(ViolationFamily);

impl Canonical for Family {
    fn encode(&self, out: &mut CanonicalWriter) {
        match self.0 {
            ViolationFamily::AdditiveShift { bound } => {
                variant(out, "additive_shift").real(bound);
            }
            ViolationFamily::ProportionalShift { bound } => {
                variant(out, "proportional_shift").real(bound);
            }
            _ => unencodable(out, "ViolationFamily"),
        }
    }
}

compiled_terminal! {
    /// A compiled assumption range of one transported point.
    #[derive(Clone, Debug)]
    CompiledTransportAssumptionRange {
        spec: TransportSensitivitySpec,
    }
}

/// Compile the assumption range of `spec`.
///
/// The route enters by [`TransportRoute::as_str`](antecedent_design::transport_sensitivity::TransportRoute::as_str),
/// the name the kernel writes. The quantity, including its conditioning, enters in declaration
/// order: the kernel copies that order into the result and does not sort it. The point value
/// and the family's bound enter by bits. There is no set-valued input.
///
/// # Errors
/// `public_builder_reentry` inside a lowered dispatch, or a violation family this encoding
/// does not name.
pub fn compile_transport_assumption_range(
    spec: TransportSensitivitySpec,
    ctx: &ExecutionContext,
) -> Result<CompiledTransportAssumptionRange, CausalError> {
    admit_compile(ctx)?;
    let spec_for_identity = &spec;
    let question = encoded("supplied.transport_assumption_range.v1", || {
        Declaration::new("antecedent.transport_assumption_range.v1")
            .field("route", spec_for_identity.route.as_str())
            .field("quantity", &spec_for_identity.quantity)
            .field("point_value", &spec_for_identity.point_value)
            .field("family", &Family(spec_for_identity.family))
            .field("grid_points", &spec_for_identity.grid_points)
            .field("source_population", spec_for_identity.source_population.as_str())
            .field("target_population", spec_for_identity.target_population.as_str())
            .field("query_binding", spec_for_identity.query_binding.as_str())
            .field("provider_snapshot", spec_for_identity.provider_snapshot.as_str())
            .field("causal_contract_id", spec_for_identity.causal_contract_id.as_str())
            .finish()
    })?;
    let lowered = lower(TerminalSpec {
        cell: ProgramCell::TransportAssumptionRange,
        plan_id: "supplied.transport_assumption_range.v1",
        functional: FunctionalKind::IdentifiedSet,
        graph_class: "transported_point_value",
        question,
        evaluator: "transport_sensitivity::transport_assumption_range",
        reading: Reading::Exact,
        role: EvidenceRole::ChainSource,
        composition: Composition::Single,
        decision: None,
        claim: Some(ClaimKind::Bounds),
    })?;
    Ok(CompiledTransportAssumptionRange { lowered, spec })
}

impl CompiledTransportAssumptionRange {
    /// The assumption range. The kernel takes the spec and no context; the gate has already run.
    ///
    /// # Errors
    /// The kernel's own refusals, or the compiler's.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<TransportAssumptionRange, TerminalError<TransportSensitivityRefusal>> {
        let (_scoped, publication) = self.enter(ctx)?;
        publication.finish(transport_assumption_range(&self.spec).map_err(TerminalError::Kernel))
    }
}

#[cfg(test)]
mod site_tests {
    use antecedent_contract::{QuantityCondition, QuantityRole, ScientificQuantity};
    use antecedent_design::transport_sensitivity::TransportRoute;

    use super::*;
    use crate::identity_laws::{bit_exact, ordered_laws};

    fn quantity(conditioning: &[QuantityCondition]) -> ScientificQuantity {
        ScientificQuantity {
            variable_id: "effect".into(),
            variable_name: "effect".into(),
            role: QuantityRole::Outcome,
            units: "mmHg".into(),
            population_id: "target".into(),
            regime_id: "do(a=1)".into(),
            horizon: 0,
            functional_id: "mean".into(),
            conditioning: conditioning.to_vec(),
            transform_id: "identity".into(),
        }
    }

    fn spec(
        point_value: f64,
        bound: f64,
        conditioning: &[QuantityCondition],
    ) -> TransportSensitivitySpec {
        TransportSensitivitySpec {
            route: TransportRoute::CohortDecision,
            quantity: quantity(conditioning),
            point_value,
            family: ViolationFamily::AdditiveShift { bound },
            grid_points: 3,
            source_population: "source".into(),
            target_population: "target".into(),
            query_binding: "query".into(),
            provider_snapshot: "snapshot".into(),
            causal_contract_id: "contract".into(),
        }
    }

    fn identity_of(spec: TransportSensitivitySpec) -> antecedent_core::SemanticDigest {
        let ctx = ExecutionContext::for_tests(7);
        compile_transport_assumption_range(spec, &ctx).expect("compiles").identity()
    }

    #[test]
    fn site_transport_assumption_program_conditioning_is_ordered_and_values_are_bit_exact() {
        let conditions = [
            QuantityCondition { variable_id: "age".into(), value_id: "adult".into() },
            QuantityCondition { variable_id: "site".into(), value_id: "clinic".into() },
        ];
        ordered_laws(&conditions, |conditioning| Some(identity_of(spec(3.0, 2.0, conditioning))));
        bit_exact(|point_value| Some(identity_of(spec(point_value, 2.0, &[]))));
        bit_exact(|bound| Some(identity_of(spec(3.0, bound, &[]))));
    }
}
