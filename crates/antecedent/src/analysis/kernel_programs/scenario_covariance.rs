//! The shared-row covariance of declared scenario estimates as a compiled program family.
//!
//! A covariance specification (`scenario_covariance_artifact::CovarianceSpec`) carries the
//! shared row table, the scenario declarations in matrix order and the method (a seeded
//! whole-row bootstrap or exact enumeration of the resampling distribution). Compiling one
//! binds the digest of its canonical bytes into a six-layer program under the
//! `SharedScenarioCovariance` route. Computing passes the support gate of the route's unit and
//! then calls `CovarianceSpec::compute` / `compute_within`, the calls the direct path makes:
//! the bootstrap draws come from the declared seed, so the matrix, the means, the replicate
//! ids and every digest are the direct path's.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::DataClassification;
use antecedent_core::ExecutionContext;
use antecedent_estimate::scenario_covariance::ScenarioCovariance;
use antecedent_io::IoError;
use antecedent_io::scenario_covariance_artifact::{
    CovarianceSpec, FunctionalWire, ScenarioCovarianceConsumeLimits,
};

use super::{KernelProgram, KernelProgramShape, KernelRealization, canonical_cbor, io_refusal};
use crate::analysis::Study;
use crate::error::CausalError;
use crate::planner::PlanView;
use crate::planner::lowered::{
    EvidenceRole, FunctionalKind, LoweredProgram, ProgramCell, UncertaintyReading,
};

const SCENARIO_COVARIANCE: KernelProgramShape = KernelProgramShape {
    cell: ProgramCell::SharedScenarioCovariance,
    plan_id: "transport.shared_scenario_covariance",
    functional: FunctionalKind::InterventionalDistribution,
    evaluator: "shared_row_resampling_covariance",
    realization: KernelRealization::Score,
    uncertainty: UncertaintyReading::Sampling,
    role: EvidenceRole::JointScenarios,
    graph_class: "cpdag_completions",
    data: DataClassification::Tabular,
    design: "one replicate identity across scenarios over the shared rows",
};

/// The specification as the kernel reads it. A score term's cell pattern is a conjunction of
/// column codes (a row matches when every pair holds), so it is a set: sorted, each pair once.
/// The scenarios (matrix order), the terms (summed in order), the rows and every other list keep
/// their declared order.
fn canonical_spec(spec: &CovarianceSpec) -> CovarianceSpec {
    let mut canonical = spec.clone();
    for scenario in &mut canonical.scenarios {
        if let FunctionalWire::LinearScore { terms } = &mut scenario.functional {
            for term in terms {
                term.pattern.sort_unstable();
                term.pattern.dedup();
            }
        }
    }
    canonical
}

/// A compiled scenario covariance: the specification and the program that binds it.
#[derive(Clone, Debug)]
pub struct CompiledScenarioCovariance {
    program: KernelProgram<CovarianceSpec>,
}

impl Study {
    /// Compile the shared-row covariance of the scenarios `spec` declares under the
    /// `SharedScenarioCovariance` route. Compiling executes nothing.
    ///
    /// # Errors
    ///
    /// Compiling from inside a lowered dispatch, cancellation, or a specification that does not
    /// encode.
    pub fn compile_scenario_covariance(
        spec: CovarianceSpec,
        ctx: &ExecutionContext,
    ) -> Result<CompiledScenarioCovariance, CausalError> {
        let canonical = canonical_cbor(&canonical_spec(&spec))?;
        let program = KernelProgram::compile(&SCENARIO_COVARIANCE, spec, &canonical, ctx)?;
        Ok(CompiledScenarioCovariance { program })
    }
}

impl CompiledScenarioCovariance {
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

    /// The specification the program binds.
    #[must_use]
    pub const fn spec(&self) -> &CovarianceSpec {
        self.program.input()
    }

    /// Pass the support gate and compute the joint covariance, as `CovarianceSpec::compute`
    /// does.
    ///
    /// # Errors
    ///
    /// The support refusal (`not_supported`) in a release build when the unit is not supported,
    /// and the refusals of the estimators and the method.
    pub fn covariance(&self, ctx: &ExecutionContext) -> Result<ScenarioCovariance, IoError> {
        self.program.gate(ctx).map_err(|refusal| io_refusal(&refusal))?;
        self.program.input().compute(ctx)
    }

    /// [`Self::covariance`] under the caller's bounds, as `CovarianceSpec::compute_within` does.
    ///
    /// # Errors
    ///
    /// As [`Self::covariance`]; more rows than `limits.max_rows` is `execution_resource_limit`.
    pub fn covariance_within(
        &self,
        ctx: &ExecutionContext,
        limits: &ScenarioCovarianceConsumeLimits,
    ) -> Result<ScenarioCovariance, IoError> {
        self.program.gate(ctx).map_err(|refusal| io_refusal(&refusal))?;
        self.program.input().compute_within(ctx, limits)
    }
}

#[cfg(test)]
mod site_tests {
    use antecedent_core::SemanticDigest;
    use antecedent_io::scenario_covariance_artifact::{
        CovarianceMethodWire, CovarianceScenarioWire, RowTableWire, ScoreTermWire,
    };

    use super::*;
    use crate::identity_laws::{ordered_laws, set_laws};

    fn term(coefficient: f64, pattern: &[(&str, i64)]) -> ScoreTermWire {
        ScoreTermWire {
            coefficient,
            pattern: pattern.iter().map(|&(column, code)| (column.to_owned(), code)).collect(),
        }
    }

    fn scenario(id: &str, terms: Vec<ScoreTermWire>) -> CovarianceScenarioWire {
        CovarianceScenarioWire {
            id: id.to_owned(),
            snapshot: "snapshot:rows".into(),
            dependence: "shared_rows".into(),
            unit_ids: None,
            functional: FunctionalWire::LinearScore { terms },
        }
    }

    fn spec(scenarios: Vec<CovarianceScenarioWire>) -> CovarianceSpec {
        CovarianceSpec {
            table: RowTableWire {
                columns: vec!["z".into(), "x".into(), "y".into()],
                unit_ids: vec!["u0".into(), "u1".into()],
                rows: vec![vec![0, 1, 1], vec![1, 0, 1]],
            },
            scenarios,
            method: CovarianceMethodWire::ExactEnumeration {
                max_compositions: 100,
                max_failure_mass: 0.0,
            },
        }
    }

    fn identity(spec: CovarianceSpec) -> SemanticDigest {
        let ctx = ExecutionContext::for_tests(0);
        Study::compile_scenario_covariance(spec, &ctx).unwrap().lowered().identity()
    }

    /// `compile_scenario_covariance`: a score term's cell pattern is a set (a conjunction of
    /// column codes); the scenarios (matrix order) and the terms (summed in order) are ordered.
    #[test]
    fn site_scenario_covariance_patterns_are_sets_and_scenarios_ordered() {
        set_laws(&[("x", 1), ("y", 1), ("z", 0)], |pattern| {
            Some(identity(spec(vec![scenario("a", vec![term(1.0, pattern)])])))
        });
        ordered_laws(&["a", "b", "c"], |ids| {
            let scenarios =
                ids.iter().map(|id| scenario(id, vec![term(1.0, &[("x", 1)])])).collect();
            Some(identity(spec(scenarios)))
        });
        ordered_laws(&[1.0, 2.0, 3.0], |coefficients| {
            let terms = coefficients.iter().map(|&c| term(c, &[("x", 1)])).collect();
            Some(identity(spec(vec![scenario("a", terms)])))
        });
    }
}
