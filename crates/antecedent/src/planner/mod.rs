//! Logical / physical analysis planning.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod binding;
mod logical;
mod plan;

pub(crate) use binding::{PlanBinding, RouteDefaults};
pub(crate) use logical::{
    StaticAteCompileInput, StaticCpdagAteCompileInput, StaticCpdagResponseCompileInput,
    StaticDistributionCompileInput, StaticPagAteCompileInput, StaticPagResponseCompileInput,
    StaticPathSpecificCompileInput, StaticResponseCompileInput, TemporalClassEffectCompileInput,
    compile_logical_codetermined_joint, compile_logical_distribution,
    compile_logical_path_specific, compile_logical_static_ate, compile_logical_static_cpdag_ate,
    compile_logical_static_cpdag_response, compile_logical_static_pag_ate,
    compile_logical_static_pag_response, compile_logical_static_response,
    compile_logical_temporal_class_effect, compile_logical_temporal_effect,
    compile_logical_temporal_effect_classified, compile_logical_temporal_response,
};
pub use plan::{LogicalAnalysisPlan, PhysicalExecutionPlan, reject_dag_only_on_pag};

#[cfg(test)]
mod tests;
