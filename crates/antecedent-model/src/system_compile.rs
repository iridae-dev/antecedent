// SPDX-License-Identifier: MIT OR Apache-2.0
//! Semantic execution blocks shared by DAG, equilibrium, dynamic and equation
//! model compilers. Scheduling supplies no existence/uniqueness/stability or
//! identification certificate; those belong to the relevant executing operator.
use crate::{
    compile::CompiledCausalModel,
    continuous_cyclic::{OdeSystem, OrnsteinUhlenbeck},
    equation_system::PolynomialSystem,
    nonlinear_equilibrium::{Clamp, EquilibriumError, EquilibriumModel},
};
use serde::{Deserialize, Serialize};

/// Mathematical interpretation of the compiled state/equation program.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SystemSemantics {
    /// Topologically evaluated structural assignments.
    AcyclicScm,
    /// Simultaneous fixed-disturbance structural equilibrium.
    EquilibriumScm,
    /// Deterministic finite-horizon initial-value dynamics.
    ContinuousTime,
    /// Stochastic dynamics with innovations over time, not fixed-noise equilibrium.
    LinearStochasticDynamics,
    /// Named residual equations over a declared domain.
    EquationConstraints,
}
/// Physical execution unit; solver iterations never become causal graph edges.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "block", rename_all = "snake_case", deny_unknown_fields)]
pub enum ExecutionBlock {
    /// A single mechanism whose external parents have already been evaluated.
    Mechanism {
        /// Endogenous output coordinate.
        variable: usize,
        /// Stable mechanism identity.
        mechanism: String,
    },
    /// Coupled residual equations, solved jointly under an explicit numerical policy.
    SimultaneousEquations {
        /// Joint endogenous coordinates.
        variables: Vec<usize>,
        /// Stable equation identities, independent of solver order.
        equations: Vec<String>,
    },
    /// Joint finite-horizon deterministic or stochastic state evolution.
    DynamicEvolution {
        /// State coordinates evolved together.
        variables: Vec<usize>,
        /// Stable drift-equation identities.
        mechanisms: Vec<String>,
    },
}
impl ExecutionBlock {
    /// Coordinates produced by this block, in declared dense order.
    #[must_use]
    pub fn variables(&self) -> &[usize] {
        match self {
            Self::Mechanism { variable, .. } => std::slice::from_ref(variable),
            Self::SimultaneousEquations { variables, .. }
            | Self::DynamicEvolution { variables, .. } => variables,
        }
    }
}
/// Immutable semantic execution schedule, distinct from a scientific certificate.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompiledSystemPlan {
    /// Declared mathematical semantics; never inferred from a numerical root.
    pub semantics: SystemSemantics,
    /// Sources before sinks, with joint blocks retained explicitly.
    pub blocks: Vec<ExecutionBlock>,
}
/// Compile equilibrium mechanisms with the existing post-clamp SCC scheduler.
pub fn compile_equilibrium(
    model: &EquilibriumModel,
    clamps: &[Clamp],
) -> Result<CompiledSystemPlan, EquilibriumError> {
    let blocks = model
        .blocks(clamps)?
        .into_iter()
        .map(|variables| {
            if variables.len() == 1
                && !model.equations[variables[0]]
                    .reads()
                    .unwrap_or_default()
                    .contains(&variables[0])
            {
                let variable = variables[0];
                ExecutionBlock::Mechanism {
                    variable,
                    mechanism: model.equations[variable].id.clone(),
                }
            } else {
                let equations = variables.iter().map(|&i| model.equations[i].id.clone()).collect();
                ExecutionBlock::SimultaneousEquations { variables, equations }
            }
        })
        .collect();
    Ok(CompiledSystemPlan { semantics: SystemSemantics::EquilibriumScm, blocks })
}
/// Compile a deterministic continuous-time initial-value block.
pub fn compile_ode(model: &OdeSystem) -> Result<CompiledSystemPlan, EquilibriumError> {
    model.validate()?;
    Ok(CompiledSystemPlan {
        semantics: SystemSemantics::ContinuousTime,
        blocks: vec![ExecutionBlock::DynamicEvolution {
            variables: (0..model.equations.len()).collect(),
            mechanisms: model.equations.iter().map(|e| e.id.clone()).collect(),
        }],
    })
}
/// Compile linear stochastic dynamics without claiming stationary stability.
pub fn compile_ou(model: &OrnsteinUhlenbeck) -> Result<CompiledSystemPlan, EquilibriumError> {
    let n = model.validate()?;
    Ok(CompiledSystemPlan {
        semantics: SystemSemantics::LinearStochasticDynamics,
        blocks: vec![ExecutionBlock::DynamicEvolution {
            variables: (0..n).collect(),
            mechanisms: (0..n).map(|i| format!("ou_drift:{i}")).collect(),
        }],
    })
}
/// Compile arbitrary polynomial residual topology into a joint constraint block.
pub fn compile_constraints(
    model: &PolynomialSystem,
) -> Result<CompiledSystemPlan, EquilibriumError> {
    model.validate()?;
    Ok(CompiledSystemPlan {
        semantics: SystemSemantics::EquationConstraints,
        blocks: vec![ExecutionBlock::SimultaneousEquations {
            variables: (0..model.variables.len()).collect(),
            equations: model.equations.iter().map(|e| e.id.clone()).collect(),
        }],
    })
}
impl CompiledCausalModel {
    /// View the existing DAG compiler as single-mechanism execution blocks. This
    /// retains its fitted store/topological fast path and callback portability.
    #[must_use]
    pub fn execution_plan(&self) -> CompiledSystemPlan {
        CompiledSystemPlan {
            semantics: SystemSemantics::AcyclicScm,
            blocks: self
                .node_order
                .iter()
                .map(|id| ExecutionBlock::Mechanism {
                    variable: id.as_usize(),
                    mechanism: format!(
                        "variable:{}",
                        self.output_layout.variables[id.as_usize()].raw()
                    ),
                })
                .collect(),
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::nonlinear_equilibrium::{Activation, Equation};
    #[test]
    fn compiled_scc_blocks_preserve_semantics_equation_ids_and_intervention_schedule() {
        let model = EquilibriumModel {
            equations: vec![
                Equation {
                    id: "a".into(),
                    intercept: 0.0,
                    weights: vec![0.0, 0.4],
                    activation: Activation::Tanh,
                    expression: None,
                    parameters: vec![],
                },
                Equation {
                    id: "b".into(),
                    intercept: 0.0,
                    weights: vec![0.5, 0.0],
                    activation: Activation::Tanh,
                    expression: None,
                    parameters: vec![],
                },
            ],
            noise_covariance: vec![1.0, 0.0, 0.0, 1.0],
        };
        let plan = compile_equilibrium(&model, &[]).unwrap();
        assert_eq!(plan.semantics, SystemSemantics::EquilibriumScm);
        assert!(
            matches!(&plan.blocks[0],ExecutionBlock::SimultaneousEquations{equations,..} if equations==&["a","b"])
        );
        let clamped = compile_equilibrium(&model, &[Clamp { variable: 0, value: 1.0 }]).unwrap();
        assert!(
            matches!(&clamped.blocks[0],ExecutionBlock::Mechanism{variable:1,mechanism} if mechanism=="b")
        );
        let ode =
            compile_ode(&OdeSystem { equations: model.equations, decay: vec![1.0, 1.0] }).unwrap();
        assert_eq!(ode.semantics, SystemSemantics::ContinuousTime);
        assert!(matches!(ode.blocks[0], ExecutionBlock::DynamicEvolution { .. }));
    }
}
