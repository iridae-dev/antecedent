//! Logical / physical analysis planning.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod binding;
pub(crate) mod fact;
mod logical;
pub mod lowered;
mod plan;
pub mod plan_view;
pub mod statistical;

pub(crate) use binding::{PlanBinding, RouteDefaults};
pub(crate) use logical::{
    StaticAdmgPathSpecificCompileInput, StaticAteCompileInput, StaticCpdagAteCompileInput,
    StaticCpdagPathSpecificCompileInput, StaticCpdagResponseCompileInput,
    StaticDistributionCompileInput, StaticPagAteCompileInput, StaticPagPathSpecificCompileInput,
    StaticPagResponseCompileInput, StaticPathSpecificCompileInput, StaticResponseCompileInput,
    TemporalClassEffectCompileInput, compile_logical_codetermined_joint,
    compile_logical_distribution, compile_logical_path_specific,
    compile_logical_path_specific_admg, compile_logical_path_specific_cpdag,
    compile_logical_path_specific_pag, compile_logical_static_ate,
    compile_logical_static_cpdag_ate, compile_logical_static_cpdag_response,
    compile_logical_static_pag_ate, compile_logical_static_pag_response,
    compile_logical_static_response, compile_logical_temporal_class_effect,
    compile_logical_temporal_effect, compile_logical_temporal_effect_classified,
    compile_logical_temporal_response,
};
pub(crate) use lowered::{
    linear_gaussian_physical_operators, linear_gaussian_program_operators,
    linear_gaussian_scientific_problem,
};
pub use plan::{LogicalAnalysisPlan, PhysicalExecutionPlan, reject_dag_only_on_pag};
pub use plan_view::{PlanRoute, PlanStage, PlanView, RouteFamily};

#[cfg(test)]
mod tests;
