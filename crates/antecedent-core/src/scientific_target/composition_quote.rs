//! Checked, iterative quotation of the target forms used by point composition.
use super::{InterventionWorld, ScientificTarget, TargetForm};
use crate::{
    ExecutionContext,
    assumption::{Assumption, AssumptionScope, AssumptionSource},
    intervention::{Intervention, StochasticPolicy, TemporalPolicy},
    query::{OutcomeFunctional, PredicateExpr, TargetPopulation},
    value::Value,
    value_expr::{MAX_EXPR_DEPTH, ValueExpr},
};

/// Caller work, payload and encoder nesting budgets for composition target quotation.
#[derive(Clone, Copy, Debug)]
pub struct CompositionQuoteLimits {
    /// Maximum visited nodes, list elements, and bytes of text combined.
    pub max_work: usize,
    /// Maximum target payload bytes charged by the quote.
    pub max_payload_bytes: usize,
    /// Maximum nested expression or intervention sequence depth. The existing
    /// recursive encoder's validated expression limit is also enforced.
    pub max_depth: usize,
}
/// Checked work and structural payload totals, before identity encoding.
#[derive(Clone, Copy, Debug, Default)]
pub struct CompositionTargetQuote {
    /// Checked logical work charged to traversal and contents.
    pub work: usize,
    /// Checked structural payload bytes, excluding admitted execution workspace.
    pub payload_bytes: usize,
}

struct Quote<'a> {
    limits: &'a CompositionQuoteLimits,
    ctx: &'a ExecutionContext,
    out: CompositionTargetQuote,
}
impl Quote<'_> {
    fn charge(&mut self, work: usize, bytes: usize) -> Result<(), &'static str> {
        if self.ctx.cancellation.is_cancelled() {
            return Err("cancelled");
        }
        self.out.work = self.out.work.checked_add(work).ok_or("quote overflow")?;
        self.out.payload_bytes =
            self.out.payload_bytes.checked_add(bytes).ok_or("quote overflow")?;
        if self.out.work > self.limits.max_work
            || self.out.payload_bytes > self.limits.max_payload_bytes
        {
            return Err("target quote limit");
        }
        Ok(())
    }
    fn list(&mut self, n: usize, width: usize) -> Result<(), &'static str> {
        self.charge(n, n.checked_mul(width).ok_or("quote overflow")?)
    }
    fn text(&mut self, s: &str) -> Result<(), &'static str> {
        self.charge(s.len(), s.len())
    }
    fn assumption(
        &mut self,
        record: &crate::assumption::AssumptionRecord,
    ) -> Result<(), &'static str> {
        self.charge(1, std::mem::size_of_val(record))?;
        match &record.assumption {
            Assumption::Custom { id, description } => {
                self.text(id)?;
                self.text(description)?;
            }
            Assumption::ParametricRestriction(p) => {
                self.text(&p.id)?;
                self.text(&p.description)?;
            }
            Assumption::PriorRestriction(p) => {
                self.text(&p.id)?;
                self.text(&p.description)?;
            }
            Assumption::CausalMarkov
            | Assumption::Faithfulness
            | Assumption::CausalSufficiency
            | Assumption::Consistency
            | Assumption::Positivity
            | Assumption::NoInterference
            | Assumption::Stationarity
            | Assumption::PiecewiseStationarity
            | Assumption::NoSelectionBias
            | Assumption::ExclusionRestriction { .. }
            | Assumption::Monotonicity => {}
        }
        if let AssumptionScope::Variables { variables } = &record.scope {
            self.list(variables.len(), 8)?;
            // Canonical scope encoding sorts a copied vector.
            let n = variables.len();
            let log =
                if n <= 1 { 0 } else { usize::BITS as usize - (n - 1).leading_zeros() as usize };
            self.charge(n.checked_mul(log).ok_or("quote overflow")?, 0)?;
        }
        match &record.source {
            AssumptionSource::AlgorithmDefault { algorithm } => self.text(algorithm)?,
            AssumptionSource::Derived { from } => self.text(from)?,
            AssumptionSource::UserDeclared | AssumptionSource::Artifact => {}
        }
        Ok(())
    }
    fn policy(&mut self, policy: &TemporalPolicy) -> Result<(), &'static str> {
        self.charge(1, std::mem::size_of_val(policy))?;
        match policy {
            TemporalPolicy::Dynamic { active_at, .. } => self.list(active_at.len(), 8)?,
            TemporalPolicy::Pulse { .. } | TemporalPolicy::Sustained { .. } => {}
        }
        Ok(())
    }
    fn value(&mut self, value: &Value) -> Result<(), &'static str> {
        self.charge(1, std::mem::size_of_val(value))?;
        match value {
            Value::Label(s) => self.text(s)?,
            Value::Float64(_) | Value::Int64(_) | Value::Bool(_) | Value::Category(_) => {}
        }
        Ok(())
    }
    fn world(&mut self, world: &InterventionWorld) -> Result<(), &'static str> {
        self.charge(1, std::mem::size_of_val(world))?;
        let b = world.binding();
        self.text(b.unit().name())?;
        self.text(b.outcome().units().name())?;
        self.list(b.conditioning().variables().len(), 8)?;
        match b.population() {
            TargetPopulation::Predicate(PredicateExpr::Named(s)) => self.text(s)?,
            TargetPopulation::Predicate(PredicateExpr::Rows(rows)) => self.list(rows.len(), 8)?,
            TargetPopulation::RowWeights { depends_on, .. } => self.list(depends_on.len(), 8)?,
            TargetPopulation::AllObserved
            | TargetPopulation::Treated
            | TargetPopulation::Untreated
            | TargetPopulation::Environment(_)
            | TargetPopulation::CustomDistribution(_)
            | TargetPopulation::LocalAtCutoff { .. } => {}
        }
        match b.outcome().functional() {
            OutcomeFunctional::ExceedanceGrid(grid) => self.list(grid.len(), 8)?,
            OutcomeFunctional::Mean
            | OutcomeFunctional::Exceedance(_)
            | OutcomeFunctional::Quantile(_) => {}
        }
        if let Some(name) = world.time().initial_law().and_then(|l| l.name()) {
            self.text(name)?;
        }
        if let Some(name) = world.time().continuous_initial_law().and_then(|l| l.name()) {
            self.text(name)?;
        }
        let action = world.action();
        if let Some(actions) = action.structural_actions() {
            self.list(actions.len(), std::mem::size_of::<super::StructuralAction>())?;
            for action in actions {
                if let Some((family, bits)) = action.mechanism_bits() {
                    self.text(family)?;
                    self.list(bits.len(), 8)?;
                }
                if let Some((law, _)) = action.initial_state_change() {
                    self.text(law)?;
                }
            }
        }
        if let Some((intervention, policy)) = action.schedule_parts() {
            self.policy(policy)?;
            self.interventions(std::iter::once(intervention))?;
        }
        if let Some(items) = action.at_decision_interventions() {
            self.list(items.len(), std::mem::size_of::<Intervention>())?;
            self.interventions(items.iter())?;
        }
        Ok(())
    }
    fn interventions<'a>(
        &mut self,
        roots: impl Iterator<Item = &'a Intervention>,
    ) -> Result<(), &'static str> {
        // Explicit stack. Every push is bounded before reserving its storage.
        let mut stack: Vec<(&Intervention, usize)> = Vec::new();
        for root in roots {
            self.charge(1, 32)?;
            stack.try_reserve(1).map_err(|_| "allocation refused")?;
            stack.push((root, 1));
        }
        while let Some((item, depth)) = stack.pop() {
            if depth > self.limits.max_depth || depth > MAX_EXPR_DEPTH {
                return Err("target nesting limit");
            }
            self.charge(1, std::mem::size_of_val(item))?;
            match item {
                Intervention::Set { value, .. } => self.value(value)?,
                Intervention::Shift { delta, .. } => self.value(delta)?,
                Intervention::Stochastic { policy, .. } => {
                    if let StochasticPolicy::Categorical { probs } = policy {
                        self.list(probs.len(), 8)?;
                    }
                }
                Intervention::Soft { mechanism, .. } => {
                    self.text(&mechanism.family_id)?;
                    self.list(mechanism.parameters.len(), 8)?;
                }
                Intervention::Transform { expr, reads, .. }
                | Intervention::Assign { expr, reads, .. } => {
                    self.list(reads.current.len(), 8)?;
                    self.list(reads.history.len(), 16)?;
                    self.expression(expr, depth.checked_add(1).ok_or("quote overflow")?)?;
                }
                Intervention::Sequence(sequence) => {
                    self.list(
                        sequence.steps.len(),
                        std::mem::size_of::<crate::intervention::SequencedIntervention>(),
                    )?;
                    for step in sequence.steps.iter() {
                        self.policy(&step.temporal)?;
                        self.charge(1, 32)?;
                        stack.try_reserve(1).map_err(|_| "allocation refused")?;
                        stack.push((&step.intervention, depth + 1));
                    }
                }
            }
        }
        Ok(())
    }
    fn expression(&mut self, root: &ValueExpr, initial_depth: usize) -> Result<(), &'static str> {
        let mut stack = Vec::new();
        stack.try_reserve(1).map_err(|_| "allocation refused")?;
        stack.push((root, initial_depth));
        while let Some((item, depth)) = stack.pop() {
            if depth > self.limits.max_depth || depth > MAX_EXPR_DEPTH {
                return Err("target nesting limit");
            }
            self.charge(1, std::mem::size_of_val(item).checked_add(32).ok_or("quote overflow")?)?;
            fn push<'a>(
                stack: &mut Vec<(&'a ValueExpr, usize)>,
                child: &'a ValueExpr,
                depth: usize,
                limit: usize,
            ) -> Result<(), &'static str> {
                if stack.len() >= limit {
                    return Err("target quote limit");
                }
                stack.try_reserve(1).map_err(|_| "allocation refused")?;
                stack.push((child, depth + 1));
                Ok(())
            }
            match item {
                ValueExpr::Neg(x) | ValueExpr::Not(x) | ValueExpr::Indicator(x) => {
                    push(&mut stack, x, depth, self.limits.max_work)?
                }
                ValueExpr::Arith { lhs, rhs, .. }
                | ValueExpr::Compare { lhs, rhs, .. }
                | ValueExpr::Logic { lhs, rhs, .. } => {
                    push(&mut stack, lhs, depth, self.limits.max_work)?;
                    push(&mut stack, rhs, depth, self.limits.max_work)?;
                }
                ValueExpr::Clip { value, lower, upper } => {
                    push(&mut stack, value, depth, self.limits.max_work)?;
                    push(&mut stack, lower, depth, self.limits.max_work)?;
                    push(&mut stack, upper, depth, self.limits.max_work)?;
                }
                ValueExpr::If { condition, then_branch, else_branch } => {
                    push(&mut stack, condition, depth, self.limits.max_work)?;
                    push(&mut stack, then_branch, depth, self.limits.max_work)?;
                    push(&mut stack, else_branch, depth, self.limits.max_work)?;
                }
                // Scientific worlds refuse callback interventions at construction.
                ValueExpr::Callback(_) => return Err("callback target action is not portable"),
                ValueExpr::Const(_)
                | ValueExpr::Generated
                | ValueExpr::Var(_)
                | ValueExpr::Lag { .. } => {}
            }
        }
        Ok(())
    }
}

/// Quote world and contrast targets before hashing, cloning, or encoding them.
/// The caller must already have admitted the visitor's stack and later workspace.
/// History-indexed laws and other target forms are outside point composition.
/// # Errors
/// Returns a reason on cancellation, overflow, excessive nesting, or caller limits.
pub fn quote_composition_target(
    target: &ScientificTarget,
    limits: &CompositionQuoteLimits,
    ctx: &ExecutionContext,
) -> Result<CompositionTargetQuote, &'static str> {
    if target.history_indexed_law().is_some() {
        return Err("history-indexed target law unsupported");
    }
    let mut q = Quote { limits, ctx, out: CompositionTargetQuote::default() };
    q.charge(1, std::mem::size_of_val(target))?;
    q.list(target.assumptions().records().len(), 64)?;
    // ModelAssumptions canonicalization builds an ordered set of record digests.
    let n = target.assumptions().records().len();
    let log = if n <= 1 { 0 } else { usize::BITS as usize - (n - 1).leading_zeros() as usize };
    q.charge(n.checked_mul(log).and_then(|v| v.checked_mul(16)).ok_or("quote overflow")?, 0)?;
    for record in &target.assumptions().records().entries {
        q.assumption(record)?;
    }
    match target.form() {
        TargetForm::World(world) => q.world(world)?,
        TargetForm::Contrast(c) => {
            q.list(c.worlds().worlds().len(), 256)?;
            for world in c.worlds().worlds() {
                q.world(world)?;
            }
        }
        _ => return Err("target form unsupported by point composition"),
    }
    Ok(q.out)
}

/// Quote a borrowed result's assumption records before comparing premises.
/// # Errors
/// Cancellation, overflow, or work/payload budget refusal.
pub fn quote_composition_assumptions(
    records: &crate::assumption::AssumptionSet,
    limits: &CompositionQuoteLimits,
    ctx: &ExecutionContext,
) -> Result<CompositionTargetQuote, &'static str> {
    let mut q = Quote { limits, ctx, out: CompositionTargetQuote::default() };
    q.list(records.len(), 64)?;
    for record in &records.entries {
        q.assumption(record)?;
    }
    Ok(q.out)
}

/// Hash the exact ordered borrowed assumption premises after their quotation and admission.
/// Source and status are included because result-premise equality includes them.
#[must_use]
pub fn composition_assumptions_identity(
    records: &crate::assumption::AssumptionSet,
) -> crate::identity::SemanticDigest {
    use crate::canonical_encode::CanonicalEncode;
    use crate::identity::{CanonicalWriter, SemanticDigest};
    let mut writer = CanonicalWriter::new("antecedent.composition.assumption_premises.v1");
    writer.word(records.len() as u64);
    for record in &records.entries {
        record.assumption.encode(&mut writer);
        match &record.scope {
            AssumptionScope::Global => {
                writer.word(1);
            }
            AssumptionScope::Identification => {
                writer.word(2);
            }
            AssumptionScope::Estimation => {
                writer.word(3);
            }
            AssumptionScope::Discovery => {
                writer.word(4);
            }
            AssumptionScope::Variables { variables } => {
                writer.word(5).word(variables.len() as u64);
                for variable in variables.iter() {
                    writer.word(u64::from(variable.raw()));
                }
            }
        }
        match &record.source {
            AssumptionSource::UserDeclared => {
                writer.word(1);
            }
            AssumptionSource::AlgorithmDefault { algorithm } => {
                writer.word(2).text(algorithm);
            }
            AssumptionSource::Artifact => {
                writer.word(3);
            }
            AssumptionSource::Derived { from } => {
                writer.word(4).text(from);
            }
        }
        writer.text(record.status.as_str());
    }
    SemanticDigest::from_bytes(writer.finish())
}
