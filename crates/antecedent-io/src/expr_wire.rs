//! Causal expression arena wire types.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{RegimeId, Value, VariableId};
use antecedent_expr::{
    CausalExprArena, ContrastOp, DomainRef, ExprId, ExprListId, ExprNode, FunctionalProgram,
    InterventionAssignment, InterventionSetId, OutcomeExprId, ProgramLimits, ProgramSchema,
    ProgramVariable, VarSetId,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::error::IoError;
use crate::query_wire::ValueWire;

/// Expr arena wire (tables only; hash indexes rebuilt on load).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ExprArenaWire {
    /// Derivation records keyed by expression id; absent on legacy artifacts.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub derivations: Vec<(u32, DerivationWire)>,
    /// Variable sets: list of variable raw ids.
    pub var_sets: Vec<Vec<u32>>,
    /// Intervention assignment sets.
    pub interventions: Vec<Vec<InterventionAssignmentWire>>,
    /// Expression lists (raw `ExprIds`).
    pub lists: Vec<Vec<u32>>,
    /// Expression nodes in id order.
    pub nodes: Vec<ExprNodeWire>,
}

/// Complete checked functional program wire record.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct FunctionalProgramWire {
    /// Owned expression arena.
    pub arena: ExprArenaWire,
    /// Source root.
    pub source: u32,
    /// Executable root.
    pub executable: u32,
    /// Semantic variable names keyed by stable variable id.
    #[serde(default)]
    pub variables: Vec<(u32, String)>,
}

/// Encode a functional program while preserving its distinct roots and schema.
pub fn functional_program_to_wire(
    program: &FunctionalProgram,
) -> Result<FunctionalProgramWire, IoError> {
    Ok(FunctionalProgramWire {
        arena: expr_arena_to_wire(program.arena())?,
        source: program.mapping().source.raw(),
        executable: program.mapping().executable.raw(),
        variables: program
            .schema()
            .variables()
            .map(|(id, value)| (id.raw(), value.name.to_string()))
            .collect(),
    })
}

/// Decode, structurally verify, and bound a functional program.
pub fn functional_program_from_wire(
    wire: &FunctionalProgramWire,
    limits: ProgramLimits,
) -> Result<FunctionalProgram, IoError> {
    if wire.arena.nodes.len() > limits.max_nodes {
        return Err(IoError::Convert("invalid functional program: node limit exceeded".into()));
    }
    if wire.arena.derivations.len() > limits.max_nodes {
        return Err(IoError::Convert(
            "invalid functional program: derivation limit exceeded".into(),
        ));
    }
    let table_entries = wire
        .arena
        .var_sets
        .iter()
        .map(Vec::len)
        .chain(wire.arena.interventions.iter().map(Vec::len))
        .chain(wire.arena.lists.iter().map(Vec::len))
        .fold(0usize, usize::saturating_add)
        .saturating_add(wire.arena.var_sets.len())
        .saturating_add(wire.arena.interventions.len())
        .saturating_add(wire.arena.lists.len());
    let populations = wire
        .arena
        .nodes
        .iter()
        .filter_map(|node| match node {
            ExprNodeWire::Distribution { population, .. }
            | ExprNodeWire::Kernel { population, .. } => Some(population.as_str()),
            _ => None,
        })
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    if table_entries.saturating_add(populations).saturating_add(wire.variables.len())
        > limits.max_table_entries
    {
        return Err(IoError::Convert("invalid functional program: table limit exceeded".into()));
    }
    let arena = expr_arena_from_wire(&wire.arena)?;
    let schema = ProgramSchema::new(wire.variables.iter().map(|(id, name)| {
        (VariableId::from_raw(*id), ProgramVariable { name: Arc::from(name.as_str()) })
    }));
    FunctionalProgram::new(
        arena,
        schema,
        ExprId::from_raw(wire.source),
        ExprId::from_raw(wire.executable),
        limits,
    )
    .map_err(|e| IoError::Convert(format!("invalid functional program: {e}")))
}

/// Durable derivation step, separate from expression algebraic identity.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DerivationWire {
    /// Named inference rule.
    pub rule: String,
    /// Human-readable projection.
    pub note: Option<String>,
    /// Input expression, when present.
    pub input: Option<u32>,
    /// Output expression.
    pub output: Option<u32>,
    /// Graph operation checked by this step.
    pub graph_operation: Option<String>,
    /// Checked premises.
    pub premises: Vec<String>,
    /// Supplied evidence regimes.
    pub evidence: Vec<u32>,
    /// Parent proof steps.
    pub parents: Vec<u32>,
}

/// Intervention assignment wire.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct InterventionAssignmentWire {
    /// Symbolic placeholder rather than a concrete floating-point value.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub symbolic: bool,
    /// Variable.
    pub variable: u32,
    /// Value.
    pub value: ValueWire,
}

/// Expression node wire.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ExprNodeWire {
    /// Distribution factor.
    Distribution {
        /// Variables.
        variables: u32,
        /// Conditioning set.
        conditioned_on: u32,
        /// Intervention set.
        intervention: u32,
        /// Domain tag.
        domain: String,
        /// Population key. Absent / empty is the default single-study label.
        #[serde(default, skip_serializing_if = "String::is_empty")]
        population: String,
        /// Catalog regime raw id.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        regime: Option<u32>,
    },
    /// Intermediate kernel (nested subexpression, not a supplied law).
    Kernel {
        /// Body expression.
        body: u32,
        /// Kernel parameter coordinates (free until explicitly marginalized).
        bound: u32,
        /// Population key.
        #[serde(default, skip_serializing_if = "String::is_empty")]
        population: String,
        /// Catalog regime raw id.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        regime: Option<u32>,
    },
    /// Product.
    Product(u32),
    /// Sum-out.
    SumOut {
        /// Variables.
        variables: u32,
        /// Body.
        expr: u32,
    },
    /// Integral-out.
    IntegralOut {
        /// Variables.
        variables: u32,
        /// Body.
        expr: u32,
    },
    /// Ratio.
    Ratio {
        /// Numerator.
        numerator: u32,
        /// Denominator.
        denominator: u32,
    },
    /// Expectation.
    Expectation {
        /// Outcome variable.
        function: u32,
        /// Distribution.
        distribution: u32,
    },
    /// Contrast.
    Contrast {
        /// Left.
        left: u32,
        /// Right.
        right: u32,
        /// Op.
        op: String,
    },
}

/// Encode arena.
///
/// # Errors
///
/// Arena indexes that do not fit in `u32`.
pub fn expr_arena_to_wire(arena: &CausalExprArena) -> Result<ExprArenaWire, IoError> {
    let mut var_sets = Vec::with_capacity(arena.var_set_count());
    for i in 0..arena.var_set_count() {
        let id = VarSetId::from_raw(u32::try_from(i).map_err(|_| IoError::TooLarge)?);
        var_sets.push(arena.var_set(id).iter().map(|v| v.raw()).collect());
    }
    let mut interventions = Vec::with_capacity(arena.intervention_set_count());
    for i in 0..arena.intervention_set_count() {
        let id = InterventionSetId::from_raw(u32::try_from(i).map_err(|_| IoError::TooLarge)?);
        interventions.push(
            arena
                .intervention_assignments(id)
                .iter()
                .map(|a| InterventionAssignmentWire {
                    variable: a.variable.raw(),
                    symbolic: a.is_symbolic(),
                    value: if a.is_symbolic() {
                        ValueWire::Float64(0.0)
                    } else {
                        ValueWire::from_value(&a.value)
                    },
                })
                .collect(),
        );
    }
    let mut lists = Vec::with_capacity(arena.list_count());
    for i in 0..arena.list_count() {
        let id = ExprListId::from_raw(u32::try_from(i).map_err(|_| IoError::TooLarge)?);
        lists.push(arena.list(id).iter().map(|e| e.raw()).collect());
    }
    let mut nodes = Vec::with_capacity(arena.len());
    let mut derivations = Vec::new();
    for i in 0..arena.len() {
        let id = ExprId::from_raw(u32::try_from(i).map_err(|_| IoError::TooLarge)?);
        nodes.push(node_to_wire(arena.node(id), arena));
        if let Some(d) = arena.derivation(id) {
            derivations.push((
                id.raw(),
                DerivationWire {
                    rule: d.rule.to_string(),
                    note: d.note.as_ref().map(ToString::to_string),
                    input: d.input.map(ExprId::raw),
                    output: d.output.map(ExprId::raw),
                    graph_operation: d.graph_operation.as_ref().map(ToString::to_string),
                    premises: d.premises.iter().map(ToString::to_string).collect(),
                    evidence: d.evidence.iter().map(|v| v.raw()).collect(),
                    parents: d.parents.iter().map(|v| v.raw()).collect(),
                },
            ));
        }
    }
    Ok(ExprArenaWire { derivations, var_sets, interventions, lists, nodes })
}

/// Decode arena by re-interning in order.
///
/// # Errors
///
/// Unknown tags or malformed indexes.
pub fn expr_arena_from_wire(w: &ExprArenaWire) -> Result<CausalExprArena, IoError> {
    let mut arena = CausalExprArena::new();
    for (index, vs) in w.var_sets.iter().enumerate() {
        if vs.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(IoError::Convert("noncanonical expression variable set".into()));
        }
        let id = arena.intern_var_set(vs.iter().copied().map(VariableId::from_raw));
        if id.raw() as usize != index {
            return Err(IoError::Convert("duplicate variable-set table entry".into()));
        }
    }
    for (index, iv) in w.interventions.iter().enumerate() {
        if iv.windows(2).any(|pair| pair[0].variable >= pair[1].variable) {
            return Err(IoError::Convert("duplicate or unordered intervention assignments".into()));
        }
        let id = arena.intern_intervention_assignments(iv.iter().map(|a| InterventionAssignment {
            variable: VariableId::from_raw(a.variable),
            value: if a.symbolic { Value::symbolic_intervention() } else { a.value.to_value() },
        }));
        if id.raw() as usize != index {
            return Err(IoError::Convert("duplicate intervention table entry".into()));
        }
    }
    for (index, list) in w.lists.iter().enumerate() {
        if list.iter().any(|entry| *entry as usize >= w.nodes.len()) {
            return Err(IoError::Convert("expression list names a node outside the arena".into()));
        }
        let id = arena.intern_list(list.iter().copied().map(ExprId::from_raw));
        if id.raw() as usize != index {
            return Err(IoError::Convert("duplicate expression-list table entry".into()));
        }
    }
    for (index, node) in w.nodes.iter().enumerate() {
        validate_node_indexes(node, index, w)?;
        let id = intern_node_from_wire(&mut arena, node)?;
        if id.raw() as usize != index {
            return Err(IoError::Convert("duplicate expression-node table entry".into()));
        }
    }
    let mut seen = std::collections::BTreeSet::new();
    for (id, d) in &w.derivations {
        if *id as usize >= arena.len()
            || !seen.insert(*id)
            || d.output.is_some_and(|out| out != *id)
            || d.input.iter().chain(d.parents.iter()).any(|p| *p >= *id)
        {
            return Err(IoError::Convert("invalid or cyclic expression derivation".into()));
        }
        arena.set_derivation(
            ExprId::from_raw(*id),
            antecedent_expr::DerivationMeta {
                rule: Arc::from(d.rule.as_str()),
                note: d.note.as_deref().map(Arc::from),
                input: d.input.map(ExprId::from_raw),
                output: d.output.map(ExprId::from_raw),
                graph_operation: d.graph_operation.as_deref().map(Arc::from),
                premises: d.premises.iter().map(|s| Arc::from(s.as_str())).collect(),
                evidence: d.evidence.iter().copied().map(RegimeId::from_raw).collect(),
                parents: d.parents.iter().copied().map(ExprId::from_raw).collect(),
            },
        );
    }
    Ok(arena)
}

fn validate_node_indexes(
    node: &ExprNodeWire,
    index: usize,
    arena: &ExprArenaWire,
) -> Result<(), IoError> {
    let varset = |id: u32| (id as usize) < arena.var_sets.len();
    let child = |id: u32| (id as usize) < index;
    let valid = match node {
        ExprNodeWire::Distribution { variables, conditioned_on, intervention, .. } => {
            varset(*variables)
                && varset(*conditioned_on)
                && (*intervention as usize) < arena.interventions.len()
        }
        ExprNodeWire::Kernel { body, bound, .. } => child(*body) && varset(*bound),
        ExprNodeWire::Product(list) => {
            arena.lists.get(*list as usize).is_some_and(|l| l.iter().all(|id| child(*id)))
        }
        ExprNodeWire::SumOut { variables, expr }
        | ExprNodeWire::IntegralOut { variables, expr } => varset(*variables) && child(*expr),
        ExprNodeWire::Ratio { numerator, denominator } => child(*numerator) && child(*denominator),
        ExprNodeWire::Expectation { distribution, .. } => child(*distribution),
        ExprNodeWire::Contrast { left, right, .. } => child(*left) && child(*right),
    };
    if valid {
        Ok(())
    } else {
        Err(IoError::Convert("invalid or cyclic expression-table reference".into()))
    }
}

fn node_to_wire(n: &ExprNode, arena: &CausalExprArena) -> ExprNodeWire {
    match n {
        ExprNode::Distribution {
            variables,
            conditioned_on,
            intervention,
            domain,
            population,
            regime,
        } => ExprNodeWire::Distribution {
            variables: variables.raw(),
            conditioned_on: conditioned_on.raw(),
            intervention: intervention.raw(),
            domain: match domain {
                DomainRef::Observational => "observational".into(),
                DomainRef::Interventional => "interventional".into(),
            },
            population: arena.population(*population).to_owned(),
            regime: regime.map(RegimeId::raw),
        },
        ExprNode::Kernel { body, bound, population, regime } => ExprNodeWire::Kernel {
            body: body.raw(),
            bound: bound.raw(),
            population: arena.population(*population).to_owned(),
            regime: regime.map(RegimeId::raw),
        },
        ExprNode::Product(list) => ExprNodeWire::Product(list.raw()),
        ExprNode::SumOut { variables, expr } => {
            ExprNodeWire::SumOut { variables: variables.raw(), expr: expr.raw() }
        }
        ExprNode::IntegralOut { variables, expr } => {
            ExprNodeWire::IntegralOut { variables: variables.raw(), expr: expr.raw() }
        }
        ExprNode::Ratio { numerator, denominator } => {
            ExprNodeWire::Ratio { numerator: numerator.raw(), denominator: denominator.raw() }
        }
        ExprNode::Expectation { function, distribution } => ExprNodeWire::Expectation {
            function: function.variable().raw(),
            distribution: distribution.raw(),
        },
        ExprNode::Contrast { left, right, op } => ExprNodeWire::Contrast {
            left: left.raw(),
            right: right.raw(),
            op: match op {
                ContrastOp::Difference => "difference".into(),
            },
        },
    }
}

fn intern_node_from_wire(arena: &mut CausalExprArena, n: &ExprNodeWire) -> Result<ExprId, IoError> {
    match n {
        ExprNodeWire::Distribution {
            variables,
            conditioned_on,
            intervention,
            domain,
            population,
            regime,
        } => {
            let domain = parse_domain(domain)?;
            arena
                .intern_distribution_tagged(
                    VarSetId::from_raw(*variables),
                    VarSetId::from_raw(*conditioned_on),
                    InterventionSetId::from_raw(*intervention),
                    domain,
                    population.as_str(),
                    regime.map(RegimeId::from_raw),
                    None,
                )
                .map_err(|e| IoError::Convert(e.to_string()))
        }
        ExprNodeWire::Kernel { body, bound, population, regime } => Ok(arena.intern_kernel(
            ExprId::from_raw(*body),
            VarSetId::from_raw(*bound),
            population.as_str(),
            regime.map(RegimeId::from_raw),
        )),
        other => Ok(arena.intern(node_from_wire(other)?)),
    }
}

fn parse_domain(domain: &str) -> Result<DomainRef, IoError> {
    match domain {
        "observational" => Ok(DomainRef::Observational),
        "interventional" => Ok(DomainRef::Interventional),
        other => Err(IoError::Convert(format!("unknown DomainRef `{other}`"))),
    }
}

fn node_from_wire(n: &ExprNodeWire) -> Result<ExprNode, IoError> {
    Ok(match n {
        ExprNodeWire::Distribution { .. } | ExprNodeWire::Kernel { .. } => {
            return Err(IoError::Convert("tagged leaves intern through the arena".into()));
        }
        ExprNodeWire::Product(list) => ExprNode::Product(ExprListId::from_raw(*list)),
        ExprNodeWire::SumOut { variables, expr } => ExprNode::SumOut {
            variables: VarSetId::from_raw(*variables),
            expr: ExprId::from_raw(*expr),
        },
        ExprNodeWire::IntegralOut { variables, expr } => ExprNode::IntegralOut {
            variables: VarSetId::from_raw(*variables),
            expr: ExprId::from_raw(*expr),
        },
        ExprNodeWire::Ratio { numerator, denominator } => ExprNode::Ratio {
            numerator: ExprId::from_raw(*numerator),
            denominator: ExprId::from_raw(*denominator),
        },
        ExprNodeWire::Expectation { function, distribution } => ExprNode::Expectation {
            function: OutcomeExprId::identity(VariableId::from_raw(*function)),
            distribution: ExprId::from_raw(*distribution),
        },
        ExprNodeWire::Contrast { left, right, op } => ExprNode::Contrast {
            left: ExprId::from_raw(*left),
            right: ExprId::from_raw(*right),
            op: match op.as_str() {
                "difference" => ContrastOp::Difference,
                other => return Err(IoError::Convert(format!("unknown ContrastOp `{other}`"))),
            },
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::RegimeId;
    use antecedent_expr::ExprNode;

    #[test]
    fn checked_program_wire_round_trip_revalidates_structure() {
        use antecedent_expr::{FunctionalProgram, ProgramLimits, ProgramSchema, ProgramVariable};
        let mut arena = CausalExprArena::new();
        let y = VariableId::from_raw(1);
        let vars = arena.intern_var_set([y]);
        let conditioning = arena.empty_var_set();
        let intervention = arena.empty_intervention_set();
        let root =
            arena.intern_distribution(vars, conditioning, intervention, DomainRef::Observational);
        let schema = ProgramSchema::new([(y, ProgramVariable { name: Arc::from("outcome") })]);
        let program =
            FunctionalProgram::new(arena, schema, root, root, ProgramLimits::default()).unwrap();
        let wire = functional_program_to_wire(&program).unwrap();
        let loaded = functional_program_from_wire(&wire, ProgramLimits::default()).unwrap();
        assert_eq!(loaded.mapping(), program.mapping());
        assert_eq!(loaded.schema().variable(y).unwrap().name.as_ref(), "outcome");
        assert_eq!(loaded.factor_requirements(), program.factor_requirements());
        assert!(
            functional_program_from_wire(
                &wire,
                ProgramLimits { max_nodes: 0, max_table_entries: 10, max_depth: 10 },
            )
            .is_err()
        );
        assert!(
            functional_program_from_wire(
                &wire,
                ProgramLimits { max_nodes: 10, max_table_entries: 1, max_depth: 10 },
            )
            .is_err()
        );
        let mut invalid = wire;
        invalid.executable = u32::MAX;
        assert!(functional_program_from_wire(&invalid, ProgramLimits::default()).is_err());
    }

    #[test]
    fn nested_kernel_round_trip_preserves_numerical_value() {
        use antecedent_core::Value;
        use antecedent_expr::{Assignment, EmpiricalTableProvider, EvalContext, FactorSpec};
        let mut arena = CausalExprArena::new();
        let x = VariableId::from_raw(0);
        let y = VariableId::from_raw(1);
        let xs = arena.intern_var_set([x]);
        let ys = arena.intern_var_set([y]);
        let empty = arena.empty_var_set();
        let intervention = arena.empty_intervention_set();
        let conditional = arena
            .intern_distribution_tagged(
                ys,
                xs,
                intervention,
                DomainRef::Observational,
                "trial",
                Some(RegimeId::from_raw(4)),
                None,
            )
            .unwrap();
        let kernel = arena.intern_kernel(conditional, xs, "derived", None);
        let marginal = arena
            .intern_distribution_tagged(
                xs,
                empty,
                intervention,
                DomainRef::Observational,
                "target",
                Some(RegimeId::from_raw(5)),
                None,
            )
            .unwrap();
        let list = arena.intern_list([kernel, marginal]);
        let product = arena.intern(ExprNode::Product(list));
        let root = arena.intern(ExprNode::SumOut { variables: xs, expr: product });
        let mut provider = EmpiricalTableProvider::new();
        provider.set_domain(x, [Value::f64(0.0), Value::f64(1.0)]);
        for (level, mass, conditional_mass) in [(0.0, 0.75, 0.2), (1.0, 0.25, 0.8)] {
            let env = Assignment::from_pairs([(x, Value::f64(level)), (y, Value::f64(1.0))]);
            let spec = FactorSpec {
                variables: &[x],
                conditioned_on: &[],
                intervention: &[],
                domain: DomainRef::Observational,
                population: "target",
                regime: Some(RegimeId::from_raw(5)),
            };
            provider.insert_probability(&spec, &env, mass).unwrap();
            let spec = FactorSpec {
                variables: &[y],
                conditioned_on: &[x],
                intervention: &[],
                domain: DomainRef::Observational,
                population: "trial",
                regime: Some(RegimeId::from_raw(4)),
            };
            provider.insert_probability(&spec, &env, conditional_mass).unwrap();
        }
        let wire = expr_arena_to_wire(&arena).unwrap();
        let mut loaded = expr_arena_from_wire(&wire).unwrap();
        let simplified = loaded.simplify(root).unwrap();
        let env = Assignment::from_pairs([(y, Value::f64(1.0))]);
        for (arena, root) in [(&arena, root), (&loaded, root), (&loaded, simplified)] {
            let value = arena
                .compile(root)
                .unwrap()
                .evaluate_with(arena, &provider, &EvalContext::default(), &env)
                .unwrap();
            assert!((value - 0.35).abs() < 1e-12);
        }
        assert_eq!(loaded.leaf_bindings(root).len(), 2);
    }

    #[test]
    fn symbolic_interventions_and_derivations_survive_json() {
        use antecedent_identify::{
            PopulationFactor, TransportCertificate, TransportFormula, bind_transport_derivation,
            lower_transport_formula,
        };
        let mut arena = CausalExprArena::new();
        let formula = TransportFormula::Direct(PopulationFactor {
            population: Arc::from("trial"),
            regime: Some(RegimeId::from_raw(7)),
            variables: Arc::from([VariableId::from_raw(1)]),
            conditioned_on: Arc::from([]),
            interventions: Arc::from([VariableId::from_raw(0)]),
        });
        let root = lower_transport_formula(&mut arena, &formula);
        bind_transport_derivation(
            &mut arena,
            root,
            &TransportCertificate {
                rule: Arc::from("transport.sid.direct"),
                selection_targets: Arc::from([]),
                premises: Arc::from([Arc::from("invariant outcome mechanism")]),
            },
            "test certificate",
        );
        let wire = expr_arena_to_wire(&arena).unwrap();
        assert!(
            wire.interventions.iter().flatten().any(|a| a.symbolic),
            "symbolic intervention must set the wire bit"
        );
        assert!(
            wire.interventions
                .iter()
                .flatten()
                .filter(|a| a.symbolic)
                .all(|a| !matches!(a.value, ValueWire::Float64(v) if v.is_nan())),
            "wire value for symbolic must not be NaN"
        );
        let bytes = serde_json::to_vec(&wire).unwrap();
        let decoded: ExprArenaWire = serde_json::from_slice(&bytes).unwrap();
        let loaded = expr_arena_from_wire(&decoded).unwrap();
        assert_eq!(loaded.pretty(root), arena.pretty(root));
        assert_eq!(loaded.derivation(root), arena.derivation(root));
        assert_eq!(loaded.leaf_bindings(root), arena.leaf_bindings(root));
        assert_eq!(loaded.derivation(root).unwrap().evidence.as_ref(), &[RegimeId::from_raw(7)]);
        assert!(
            (0..loaded.intervention_set_count()).any(|i| {
                loaded
                    .intervention_assignments(InterventionSetId::from_raw(
                        u32::try_from(i).unwrap(),
                    ))
                    .iter()
                    .any(InterventionAssignment::is_symbolic)
            }),
            "decode of symbolic:true must rebuild the marker, not NaN"
        );
        let mut invalid = decoded;
        invalid.nodes.push(ExprNodeWire::Kernel {
            body: u32::MAX,
            bound: 0,
            population: String::new(),
            regime: None,
        });
        assert!(expr_arena_from_wire(&invalid).is_err());
    }

    #[test]
    fn unreferenced_expression_list_entries_are_bounds_checked() {
        let wire = ExprArenaWire {
            derivations: Vec::new(),
            var_sets: Vec::new(),
            interventions: Vec::new(),
            lists: vec![vec![3]],
            nodes: Vec::new(),
        };
        let error = expr_arena_from_wire(&wire).unwrap_err().to_string();
        assert!(error.contains("outside the arena"), "{error}");
    }

    #[test]
    fn concrete_nan_intervention_does_not_set_wire_symbolic_bit() {
        use antecedent_core::Value;
        let mut arena = CausalExprArena::new();
        let t = VariableId::from_raw(0);
        let id = arena.intern_intervention_assignments([InterventionAssignment {
            variable: t,
            value: Value::f64(f64::NAN),
        }]);
        assert!(!arena.intervention_assignments(id)[0].is_symbolic());
        let wire = expr_arena_to_wire(&arena).unwrap();
        let entry = wire.interventions.iter().flatten().find(|a| a.variable == t.raw()).unwrap();
        assert!(!entry.symbolic, "concrete NaN must not set symbolic");
        assert!(matches!(entry.value, ValueWire::Float64(v) if v.is_nan()));
        let loaded = expr_arena_from_wire(&wire).unwrap();
        let restored = loaded.intervention_assignments(id);
        assert_eq!(restored.len(), 1);
        assert!(!restored[0].is_symbolic());
        assert!(matches!(restored[0].value, Value::Float64(v) if v.is_nan()));
    }

    #[test]
    fn old_distribution_json_defaults_empty_population() {
        let json = r#"{"distribution":{"variables":0,"conditioned_on":0,"intervention":0,"domain":"observational"}}"#;
        let node: ExprNodeWire = serde_json::from_str(json).unwrap();
        let ExprNodeWire::Distribution { population, regime, .. } = node else {
            panic!("expected distribution");
        };
        assert!(population.is_empty());
        assert!(regime.is_none());
    }

    #[test]
    fn kernel_round_trip_preserves_population_and_regime() {
        let mut arena = CausalExprArena::new();
        let y = arena.intern_var_set([VariableId::from_raw(1)]);
        let z = arena.intern_var_set([VariableId::from_raw(2)]);
        let empty_i = arena.empty_intervention_set();
        let body = arena
            .intern_distribution_tagged(
                y,
                z,
                empty_i,
                DomainRef::Observational,
                "source",
                Some(RegimeId::from_raw(3)),
                Some(DomainRef::Observational),
            )
            .unwrap();
        let kernel = arena.intern_kernel(body, z, "source", Some(RegimeId::from_raw(3)));
        let product = {
            let list = arena.intern_list([kernel]);
            arena.intern(ExprNode::Product(list))
        };
        let wire = expr_arena_to_wire(&arena).unwrap();
        assert!(wire.nodes.iter().any(|n| matches!(n, ExprNodeWire::Kernel { .. })));
        let loaded = expr_arena_from_wire(&wire).unwrap();
        assert_eq!(loaded.pretty(kernel), arena.pretty(kernel));
        assert_eq!(loaded.leaf_bindings(product), arena.leaf_bindings(product));
    }
}
