//! Compiler-native robust decisions over declared finite joint scenarios.
//! Missing joint dependence, a posterior request or a scalar fit is not a
//! substitute for this source. A stopped or unidentified set stays incomplete.
pub use super::execute::CompiledRobustDecision;
pub use antecedent_design::decision_robust_artifact::{
    JointScenarioCarrier, MAX_COMPILED_ROBUST_ITEMS, MAX_SCENARIO_CARRIER_BYTES,
};

pub use antecedent_design::decision_robust_artifact::ScenarioDecision as CompiledRobustResult;
