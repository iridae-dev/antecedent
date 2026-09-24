//! Immutable, structurally checked expression programs.
//!
//! Structural checking here does not establish normalization or causal identification.

use crate::{
    Assignment, CausalExprArena, CompiledEvaluator, DistributionProvider, DomainRef, EvalContext,
    EvalError, ExprId, ExprNode, InterventionSetId, PopulationKeyId, VarSetId,
};
use antecedent_core::{RegimeId, VariableId};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    sync::Arc,
};

/// Semantic declaration for one expression variable.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProgramVariable {
    /// Stable semantic name.
    pub name: Arc<str>,
}

/// Variable semantics attached to a functional program.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ProgramSchema {
    variables: BTreeMap<VariableId, ProgramVariable>,
    duplicate_ids: bool,
}
impl ProgramSchema {
    /// Construct a schema from stable variable ids and names.
    pub fn new(variables: impl IntoIterator<Item = (VariableId, ProgramVariable)>) -> Self {
        let mut mapped = BTreeMap::new();
        let mut duplicate_ids = false;
        for (id, value) in variables {
            duplicate_ids |= mapped.insert(id, value).is_some();
        }
        Self { variables: mapped, duplicate_ids }
    }
    /// Read a variable declaration.
    pub fn variable(&self, id: VariableId) -> Option<&ProgramVariable> {
        self.variables.get(&id)
    }
    /// Iterate declared variables.
    pub fn variables(&self) -> impl Iterator<Item = (VariableId, &ProgramVariable)> {
        self.variables.iter().map(|(id, v)| (*id, v))
    }
}

/// Explicit correspondence between user source and executable expression.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProgramMapping {
    /// Source expression root.
    pub source: ExprId,
    /// Executable expression root.
    pub executable: ExprId,
}

/// Resource bounds checked when accepting a program.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProgramLimits {
    /// Maximum nodes in the arena.
    pub max_nodes: usize,
    /// Maximum total entries across interned sets, interventions, expression lists,
    /// populations, and schema variables.
    pub max_table_entries: usize,
    /// Maximum expression dependency depth, including product children.
    pub max_depth: usize,
}
impl Default for ProgramLimits {
    fn default() -> Self {
        Self { max_nodes: 100_000, max_table_entries: 1_000_000, max_depth: 512 }
    }
}

/// A provider factor that the executable expression will request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FactorRequirement {
    /// Variables in factor.
    pub variables: Arc<[VariableId]>,
    /// Conditioning variables.
    pub conditioned_on: Arc<[VariableId]>,
    /// Intervention assignments.
    pub intervention: Arc<[crate::InterventionAssignment]>,
    /// Factor domain.
    pub domain: DomainRef,
    /// Population key.
    pub population: Arc<str>,
    /// Evidence regime, if any.
    pub regime: Option<RegimeId>,
}

/// Errors from checked program construction.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ProgramError {
    /// Program exceeds configured limits.
    Limit(&'static str),
    /// A root or table reference is invalid.
    InvalidReference(&'static str),
    /// Source-to-executable mapping does not match supplied roots.
    MappingMismatch,
    /// Schema repeated one variable id.
    DuplicateSchemaVariable,
    /// An expression variable is absent from the semantic schema.
    MissingSchemaVariable(VariableId),
}
impl fmt::Display for ProgramError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for ProgramError {}

#[derive(Clone, Debug)]
struct ProgramInner {
    arena: CausalExprArena,
    schema: ProgramSchema,
    mapping: ProgramMapping,
    free_variables: Arc<[VariableId]>,
    factor_requirements: Arc<[FactorRequirement]>,
}

/// Immutable owner of an arena, schema, source/executable roots, and checked analyses.
#[derive(Clone, Debug)]
pub struct FunctionalProgram(Arc<ProgramInner>);
impl FunctionalProgram {
    /// Check and own an expression program. This establishes structural validity only.
    pub fn new(
        arena: CausalExprArena,
        schema: ProgramSchema,
        source: ExprId,
        executable: ExprId,
        limits: ProgramLimits,
    ) -> Result<Self, ProgramError> {
        if arena.len() > limits.max_nodes {
            return Err(ProgramError::Limit("nodes"));
        }
        if arena.table_entry_count().saturating_add(schema.variables.len())
            > limits.max_table_entries
        {
            return Err(ProgramError::Limit("tables"));
        }
        if source.raw() as usize >= arena.len() || executable.raw() as usize >= arena.len() {
            return Err(ProgramError::InvalidReference("root"));
        }
        if schema.duplicate_ids {
            return Err(ProgramError::DuplicateSchemaVariable);
        }
        validate_arena(&arena, &schema, limits.max_depth)?;
        let mapping = ProgramMapping { source, executable };
        let compiled =
            arena.compile(executable).map_err(|_| ProgramError::InvalidReference("executable"))?;
        let free_variables: Arc<[VariableId]> = Arc::from(compiled.free_vars());
        let factor_requirements = collect_requirements(&arena, executable);
        Ok(Self(Arc::new(ProgramInner {
            arena,
            schema,
            mapping,
            free_variables,
            factor_requirements,
        })))
    }
    /// Owned expression arena, exposed read-only.
    pub fn arena(&self) -> &CausalExprArena {
        &self.0.arena
    }
    /// Semantic schema.
    pub fn schema(&self) -> &ProgramSchema {
        &self.0.schema
    }
    /// Source and executable root correspondence.
    pub fn mapping(&self) -> ProgramMapping {
        self.0.mapping
    }
    /// Cached free variables of the executable expression.
    pub fn free_variables(&self) -> &[VariableId] {
        &self.0.free_variables
    }
    /// Cached provider factor requirements.
    pub fn factor_requirements(&self) -> &[FactorRequirement] {
        &self.0.factor_requirements
    }
    /// Compile an evaluator bound to this program's arena and executable root.
    pub fn compile(&self) -> Result<ProgramEvaluator, EvalError> {
        let evaluator = self.0.arena.compile(self.0.mapping.executable)?;
        Ok(ProgramEvaluator { program: self.clone(), evaluator })
    }
}

/// Compiled evaluator which cannot be supplied an unrelated arena.
#[derive(Clone, Debug)]
pub struct ProgramEvaluator {
    program: FunctionalProgram,
    evaluator: CompiledEvaluator,
}
impl ProgramEvaluator {
    /// Evaluate the executable root.
    pub fn evaluate(
        &self,
        provider: &dyn DistributionProvider,
        ctx: &EvalContext,
    ) -> Result<f64, EvalError> {
        self.evaluate_with(provider, ctx, &Assignment::new())
    }
    /// Evaluate with explicit semantic variable bindings.
    pub fn evaluate_with(
        &self,
        provider: &dyn DistributionProvider,
        ctx: &EvalContext,
        env: &Assignment,
    ) -> Result<f64, EvalError> {
        self.evaluator.evaluate_program(&self.program.0.arena, provider, ctx, env)
    }
}

fn validate_arena(
    a: &CausalExprArena,
    schema: &ProgramSchema,
    max_depth: usize,
) -> Result<(), ProgramError> {
    let mut depths = vec![0usize; a.len()];
    for i in 0..a.len() {
        let n = a.node(ExprId::from_raw(i as u32));
        let valid_expr = |id: ExprId| (id.raw() as usize) < i;
        let mut child_depth = 0usize;
        match n {
            ExprNode::Distribution {
                variables, conditioned_on, intervention, population, ..
            } => {
                check_tables(a, *variables, *conditioned_on, *intervention, *population)?;
                check_variables(a.var_set(*variables).iter().copied(), schema)?;
                check_variables(a.var_set(*conditioned_on).iter().copied(), schema)?;
                check_variables(
                    a.intervention_assignments(*intervention).iter().map(|v| v.variable),
                    schema,
                )?;
            }
            ExprNode::Kernel { body, bound, population, .. } => {
                if !valid_expr(*body) {
                    return Err(ProgramError::InvalidReference("kernel body"));
                }
                child_depth = depths[body.raw() as usize];
                if bound.raw() as usize >= a.var_set_count()
                    || population.raw() as usize >= a.population_count()
                {
                    return Err(ProgramError::InvalidReference("kernel table"));
                }
                check_variables(a.var_set(*bound).iter().copied(), schema)?;
            }
            ExprNode::Product(list) => {
                if list.raw() as usize >= a.list_count()
                    || a.list(*list).iter().any(|id| !valid_expr(*id))
                {
                    return Err(ProgramError::InvalidReference("product list"));
                }
                if a.list(*list).iter().any(|id| id.raw() as usize >= i) {
                    return Err(ProgramError::InvalidReference("forward product child"));
                }
                child_depth =
                    a.list(*list).iter().map(|id| depths[id.raw() as usize]).max().unwrap_or(0);
            }
            ExprNode::SumOut { variables, expr } | ExprNode::IntegralOut { variables, expr } => {
                if !valid_expr(*expr) || variables.raw() as usize >= a.var_set_count() {
                    return Err(ProgramError::InvalidReference("marginalization"));
                }
                child_depth = depths[expr.raw() as usize];
                check_variables(a.var_set(*variables).iter().copied(), schema)?;
            }
            ExprNode::Ratio { numerator, denominator } => {
                if !valid_expr(*numerator) || !valid_expr(*denominator) {
                    return Err(ProgramError::InvalidReference("ratio"));
                }
                child_depth =
                    depths[numerator.raw() as usize].max(depths[denominator.raw() as usize]);
            }
            ExprNode::Expectation { function, distribution } => {
                if !valid_expr(*distribution) {
                    return Err(ProgramError::InvalidReference("expectation"));
                }
                child_depth = depths[distribution.raw() as usize];
                check_variables([function.variable()], schema)?;
            }
            ExprNode::Contrast { left, right, .. } => {
                if !valid_expr(*left) || !valid_expr(*right) {
                    return Err(ProgramError::InvalidReference("contrast"));
                }
                child_depth = depths[left.raw() as usize].max(depths[right.raw() as usize]);
            }
        }
        let depth = child_depth.saturating_add(1);
        if depth > max_depth {
            return Err(ProgramError::Limit("depth"));
        }
        depths[i] = depth;
    }
    Ok(())
}

fn check_variables(
    variables: impl IntoIterator<Item = VariableId>,
    schema: &ProgramSchema,
) -> Result<(), ProgramError> {
    for variable in variables {
        if schema.variable(variable).is_none() {
            return Err(ProgramError::MissingSchemaVariable(variable));
        }
    }
    Ok(())
}
fn check_tables(
    a: &CausalExprArena,
    v: VarSetId,
    c: VarSetId,
    i: InterventionSetId,
    p: PopulationKeyId,
) -> Result<(), ProgramError> {
    if v.raw() as usize >= a.var_set_count()
        || c.raw() as usize >= a.var_set_count()
        || i.raw() as usize >= a.intervention_set_count()
        || p.raw() as usize >= a.population_count()
    {
        Err(ProgramError::InvalidReference("factor table"))
    } else {
        Ok(())
    }
}
fn collect_requirements(a: &CausalExprArena, root: ExprId) -> Arc<[FactorRequirement]> {
    let mut seen = BTreeSet::new();
    let mut stack = vec![root];
    let mut out = Vec::new();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        match a.node(id) {
            ExprNode::Distribution {
                variables,
                conditioned_on,
                intervention,
                domain,
                population,
                regime,
            } => out.push(FactorRequirement {
                variables: Arc::from(a.var_set(*variables)),
                conditioned_on: Arc::from(a.var_set(*conditioned_on)),
                intervention: Arc::from(a.intervention_assignments(*intervention)),
                domain: *domain,
                population: Arc::from(a.population(*population)),
                regime: *regime,
            }),
            ExprNode::Kernel { body, .. }
            | ExprNode::SumOut { expr: body, .. }
            | ExprNode::IntegralOut { expr: body, .. } => stack.push(*body),
            ExprNode::Product(list) => stack.extend(a.list(*list)),
            ExprNode::Ratio { numerator, denominator } => stack.extend([*numerator, *denominator]),
            ExprNode::Expectation { distribution, .. } => stack.push(*distribution),
            ExprNode::Contrast { left, right, .. } => stack.extend([*left, *right]),
        }
    }
    let mut unique = Vec::new();
    for item in out {
        if !unique.contains(&item) {
            unique.push(item);
        }
    }
    Arc::from(unique)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DomainRef, ExprNode};
    use antecedent_core::VariableId;

    fn distribution_arena() -> (CausalExprArena, ExprId) {
        let mut arena = CausalExprArena::new();
        let y = VariableId::from_raw(1);
        let vars = arena.intern_var_set([y]);
        let cond = arena.empty_var_set();
        let intervention = arena.empty_intervention_set();
        let root = arena.intern_distribution(vars, cond, intervention, DomainRef::Observational);
        (arena, root)
    }

    #[test]
    fn checked_program_owns_roots_and_requirements() {
        let (mut arena, source) = distribution_arena();
        let x = VariableId::from_raw(0);
        let executable_variables = arena.intern_var_set([x]);
        let empty = arena.empty_var_set();
        let no_intervention = arena.empty_intervention_set();
        let executable = arena.intern_distribution(
            executable_variables,
            empty,
            no_intervention,
            DomainRef::Interventional,
        );
        let schema = ProgramSchema::new([
            (VariableId::from_raw(1), ProgramVariable { name: Arc::from("outcome") }),
            (x, ProgramVariable { name: Arc::from("treatment") }),
        ]);
        let program =
            FunctionalProgram::new(arena, schema, source, executable, ProgramLimits::default())
                .unwrap();
        assert_eq!(program.mapping(), ProgramMapping { source, executable });
        assert_eq!(program.factor_requirements().len(), 1);
        assert!(program.free_variables().contains(&x));
        assert!(!program.free_variables().contains(&VariableId::from_raw(1)));
        assert_eq!(program.factor_requirements()[0].domain, DomainRef::Interventional);
        assert!(program.compile().is_ok());
    }

    #[test]
    fn rejects_invalid_table_references() {
        let mut arena = CausalExprArena::new();
        let root = arena.intern(ExprNode::SumOut {
            variables: VarSetId::from_raw(99),
            expr: ExprId::from_raw(0),
        });
        let error = FunctionalProgram::new(
            arena,
            ProgramSchema::default(),
            root,
            root,
            ProgramLimits::default(),
        )
        .unwrap_err();
        assert_eq!(error, ProgramError::InvalidReference("marginalization"));
    }

    #[test]
    fn rejects_missing_schema_variables_and_forward_references() {
        let (arena, root) = distribution_arena();
        assert_eq!(
            FunctionalProgram::new(
                arena.clone(),
                ProgramSchema::default(),
                root,
                root,
                ProgramLimits::default()
            )
            .unwrap_err(),
            ProgramError::MissingSchemaVariable(VariableId::from_raw(1))
        );

        let mut cyclic = arena;
        let next = ExprId::from_raw(cyclic.len() as u32 + 1);
        let parent = cyclic.intern(crate::ExprNode::Kernel {
            body: next,
            bound: VarSetId::from_raw(0),
            population: PopulationKeyId::from_raw(0),
            regime: None,
        });
        let schema = ProgramSchema::new([(
            VariableId::from_raw(1),
            ProgramVariable { name: Arc::from("outcome") },
        )]);
        assert!(matches!(
            FunctionalProgram::new(cyclic, schema, parent, parent, ProgramLimits::default()),
            Err(ProgramError::InvalidReference("kernel body"))
        ));
    }

    #[test]
    fn rejects_duplicate_schema_and_resource_limit_overruns() {
        let (arena, root) = distribution_arena();
        let duplicate = ProgramSchema::new([
            (VariableId::from_raw(1), ProgramVariable { name: Arc::from("first") }),
            (VariableId::from_raw(1), ProgramVariable { name: Arc::from("second") }),
        ]);
        assert_eq!(
            FunctionalProgram::new(arena.clone(), duplicate, root, root, ProgramLimits::default())
                .unwrap_err(),
            ProgramError::DuplicateSchemaVariable
        );
        let schema = ProgramSchema::new([(
            VariableId::from_raw(1),
            ProgramVariable { name: Arc::from("outcome") },
        )]);
        assert_eq!(
            FunctionalProgram::new(
                arena.clone(),
                schema.clone(),
                root,
                root,
                ProgramLimits { max_nodes: 10, max_table_entries: 1, max_depth: 10 },
            )
            .unwrap_err(),
            ProgramError::Limit("tables")
        );
        assert_eq!(
            FunctionalProgram::new(
                arena,
                schema,
                root,
                root,
                ProgramLimits { max_nodes: 10, max_table_entries: 10, max_depth: 0 },
            )
            .unwrap_err(),
            ProgramError::Limit("depth")
        );
    }

    #[test]
    fn bounds_dependency_depth_before_recursive_compilation() {
        let (mut arena, mut root) = distribution_arena();
        let empty = arena.empty_var_set();
        for _ in 0..4 {
            root = arena.intern_kernel(root, empty, "", None);
        }
        let schema = ProgramSchema::new([(
            VariableId::from_raw(1),
            ProgramVariable { name: Arc::from("outcome") },
        )]);
        assert_eq!(
            FunctionalProgram::new(
                arena,
                schema,
                root,
                root,
                ProgramLimits { max_nodes: 10, max_table_entries: 20, max_depth: 3 },
            )
            .unwrap_err(),
            ProgramError::Limit("depth")
        );
    }
}
