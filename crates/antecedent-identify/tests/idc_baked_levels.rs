//! Known-truth check of IDC at a fixed intervention level against exactly enumerated binary SCMs.
//!
//! `identify_conditional_at_levels` bakes the hard-set level of the intervened variable into
//! the emitted factors. The two arms of a conditional contrast are then two different
//! functionals of one observed law, and a conditioning variable moved into the intervention
//! set by rule 2 stays a free variable evaluated at each of its observed values. The oracle
//! enumerates the exact interventional law in the SCM and never consults the identifier.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    reason = "test scaffolding compares exact constants and indexes with small literals"
)]

use std::sync::Arc;

use antecedent_core::{CausalRng, Intervention, Value, VariableId};
use antecedent_expr::{Assignment, DistributionProvider, EvalContext, EvalError, FactorSpec};
use antecedent_graph::{Admg, DenseNodeId};
use antecedent_identify::{IdcIdentifier, IdentificationStatus, IdentificationWorkspace};

fn v(i: usize) -> VariableId {
    VariableId::from_raw(i as u32)
}
fn d(i: usize) -> DenseNodeId {
    DenseNodeId::from_raw(i as u32)
}
fn level(value: usize) -> Value {
    Value::f64(value as f64)
}
/// Bit `index` of `word`: the value of node `index` in a joint-table cell.
const fn bit(word: usize, index: usize) -> usize {
    (word >> index) & 1
}

/// Binary SCM over observed nodes `0..n` (a topological order) with one independent binary
/// latent per bidirected edge.
struct Scm {
    n: usize,
    directed: Vec<(usize, usize)>,
    bidirected: Vec<(usize, usize)>,
    latent_p: Vec<f64>,
    /// `tables[i][k] = P(V_i = 1 | inputs = k)`; input bits are the observed parents
    /// (ascending) followed by the incident latents (ascending).
    tables: Vec<Vec<f64>>,
}

impl Scm {
    fn inputs(&self, node: usize) -> (Vec<usize>, Vec<usize>) {
        let parents = self.directed.iter().filter(|e| e.1 == node).map(|e| e.0).collect();
        let latents = self
            .bidirected
            .iter()
            .enumerate()
            .filter(|(_, e)| e.0 == node || e.1 == node)
            .map(|(k, _)| k)
            .collect();
        (parents, latents)
    }

    /// Seeded generic mechanisms: every conditional probability is drawn in `(0.12, 0.88)`.
    fn random(
        n: usize,
        directed: Vec<(usize, usize)>,
        bidirected: Vec<(usize, usize)>,
        rng: &mut CausalRng,
    ) -> Self {
        let mut draw = || 0.12 + 0.76 * (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        let latent_p = bidirected.iter().map(|_| draw()).collect();
        let mut scm = Self { n, directed, bidirected, latent_p, tables: Vec::new() };
        for node in 0..n {
            let (parents, latents) = scm.inputs(node);
            let table = (0..1usize << (parents.len() + latents.len())).map(|_| draw()).collect();
            scm.tables.push(table);
        }
        scm
    }

    fn admg(&self) -> Admg {
        let mut g = Admg::with_variables(self.n as u32);
        for &(a, b) in &self.directed {
            g.insert_directed(d(a), d(b)).unwrap();
        }
        for &(a, b) in &self.bidirected {
            g.insert_bidirected(d(a), d(b)).unwrap();
        }
        g
    }

    /// Exact joint over the observed nodes (bit `i` of the index is `V_i`), under
    /// `do(node = value)` for every entry of `interventions`.
    fn joint(&self, interventions: &[(usize, usize)]) -> Vec<f64> {
        let inputs: Vec<_> = (0..self.n).map(|i| self.inputs(i)).collect();
        let mut out = vec![0.0; 1 << self.n];
        for u in 0..1usize << self.bidirected.len() {
            let mut latent_mass = 1.0;
            for (k, p) in self.latent_p.iter().enumerate() {
                latent_mass *= if bit(u, k) == 1 { *p } else { 1.0 - *p };
            }
            for (obs, cell) in out.iter_mut().enumerate() {
                let mut mass = latent_mass;
                for (node, (parents, latents)) in inputs.iter().enumerate() {
                    let value = bit(obs, node);
                    if let Some(&(_, forced)) = interventions.iter().find(|iv| iv.0 == node) {
                        if value != forced {
                            mass = 0.0;
                        }
                        continue;
                    }
                    let mut index = 0;
                    for (slot, &p) in parents.iter().enumerate() {
                        index |= bit(obs, p) << slot;
                    }
                    for (slot, &k) in latents.iter().enumerate() {
                        index |= bit(u, k) << (parents.len() + slot);
                    }
                    let p1 = self.tables[node][index];
                    mass *= if value == 1 { p1 } else { 1.0 - p1 };
                }
                *cell += mass;
            }
        }
        out
    }
}

/// Mass of the cells of `joint` matching every `(node, value)` constraint.
fn mass(joint: &[f64], constraints: &[(usize, usize)]) -> f64 {
    joint
        .iter()
        .enumerate()
        .filter(|(cell, _)| constraints.iter().all(|&(node, value)| bit(*cell, node) == value))
        .map(|(_, p)| *p)
        .sum()
}

/// Answers every observational factor `P(A | B)` from one exact joint table.
struct JointProvider {
    joint: Vec<f64>,
}

impl JointProvider {
    fn constraints(
        vars: &[VariableId],
        assignment: &Assignment,
    ) -> Result<Vec<(usize, usize)>, EvalError> {
        vars.iter()
            .map(|var| {
                let value = assignment
                    .get(*var)
                    .and_then(Value::as_f64)
                    .ok_or(EvalError::MissingBinding(*var))?;
                Ok((var.raw() as usize, usize::from(value != 0.0)))
            })
            .collect()
    }
}

impl DistributionProvider for JointProvider {
    fn probability(
        &self,
        spec: &FactorSpec<'_>,
        assignment: &Assignment,
        _: &EvalContext,
    ) -> Result<f64, EvalError> {
        let mut constraints = Self::constraints(spec.conditioned_on, assignment)?;
        let denominator = mass(&self.joint, &constraints);
        constraints.extend(Self::constraints(spec.variables, assignment)?);
        if denominator == 0.0 {
            return Err(EvalError::DivisionByZero);
        }
        Ok(mass(&self.joint, &constraints) / denominator)
    }

    fn support(
        &self,
        vars: &[VariableId],
        _: &EvalContext,
    ) -> Result<Arc<[Arc<[Value]>]>, EvalError> {
        Ok((0..1usize << vars.len())
            .map(|row| (0..vars.len()).map(|i| level(bit(row, i))).collect())
            .collect())
    }

    fn outcome(
        &self,
        var: VariableId,
        assignment: &Assignment,
        _: &EvalContext,
    ) -> Result<f64, EvalError> {
        assignment.get(var).and_then(Value::as_f64).ok_or(EvalError::MissingBinding(var))
    }

    fn n_draws(&self) -> Option<usize> {
        None
    }
}

/// `P(y = 1 | do(x = level), z = z_value)` for both values of `z`, read off the exact SCM.
fn exact_law(scm: &Scm, x: usize, y: usize, z: usize, lv: usize) -> [f64; 2] {
    let law = scm.joint(&[(x, lv)]);
    let at = |z_value: usize| mass(&law, &[(z, z_value), (y, 1)]) / mass(&law, &[(z, z_value)]);
    [at(0), at(1)]
}

/// Every free variable of an identified conditional law at a baked level takes both values;
/// the law there must equal the exact `P(y | do(x = level), z)`.
#[test]
fn baked_level_idc_matches_exact_conditional_intervention_on_every_three_node_admg() {
    let idc = IdcIdentifier::new();
    let mut ws = IdentificationWorkspace::default();
    let mut rng = CausalRng::from_seed(0x1DC5);
    let pairs = [(0usize, 1usize), (0, 2), (1, 2)];
    let pick = |mask: usize| -> Vec<(usize, usize)> {
        pairs.iter().enumerate().filter(|(k, _)| bit(mask, *k) == 1).map(|(_, e)| *e).collect()
    };
    let (mut identified, mut refused, mut moved_free, mut distinct_arms) = (0usize, 0, 0, 0);
    for directed_mask in 0..8usize {
        for bidirected_mask in 0..8usize {
            let scm = Scm::random(3, pick(directed_mask), pick(bidirected_mask), &mut rng);
            let observed = JointProvider { joint: scm.joint(&[]) };
            let prep = idc.prepare(&scm.admg()).unwrap();
            for (x, z, y) in [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)] {
                let mut arms = Vec::new();
                for lv in 0..2usize {
                    let context = format!(
                        "directed={:?} bidirected={:?} P({y} | do({x}={lv}), {z})",
                        scm.directed, scm.bidirected
                    );
                    let intervention = [Intervention::set(v(x), level(lv))];
                    let res = idc
                        .identify_conditional_at_levels(
                            &prep,
                            &[v(y)],
                            &intervention,
                            &[v(z)],
                            &mut ws,
                        )
                        .unwrap_or_else(|e| panic!("{context}: IDC returned Err({e})"));
                    let symbolic = idc
                        .identify_conditional(&prep, &[v(y)], &intervention, &[v(z)], &mut ws)
                        .unwrap();
                    assert_eq!(res.status, symbolic.status, "{context}: same verdict");
                    if res.status == IdentificationStatus::NotIdentified {
                        refused += 1;
                        assert!(res.hedge.is_some(), "{context}");
                        continue;
                    }
                    assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);
                    identified += 1;
                    let functional = res.estimands[0].functional;
                    let free = res.arena.free_variables(functional);
                    assert!(
                        !free.contains(&v(x)),
                        "{context}: a baked level must not leave the intervened variable free"
                    );
                    moved_free += usize::from(free.contains(&v(z)));
                    let plan = res.arena.compile(functional).unwrap();
                    let law = scm.joint(&[(x, lv)]);
                    for row in 0..1usize << free.len() {
                        let mut bits = 0usize;
                        let mut env = Assignment::new();
                        for (i, var) in free.iter().enumerate() {
                            let value = bit(row, i);
                            bits |= value << var.raw();
                            env.set(*var, level(value));
                        }
                        let actual = plan
                            .evaluate_with(&res.arena, &observed, &EvalContext::default(), &env)
                            .unwrap();
                        let z_value = (z, bit(bits, z));
                        let expected =
                            mass(&law, &[z_value, (y, bit(bits, y))]) / mass(&law, &[z_value]);
                        assert!(
                            (actual - expected).abs() < 1e-9,
                            "{context}: functional {} = {actual}, truth {expected} at bits {bits:b}",
                            res.arena.pretty(functional),
                        );
                    }
                    arms.push(res.arena.pretty(functional));
                }
                // Two identified arms are two functionals whenever the exact laws differ.
                if let [zero, one] = arms.as_slice() {
                    let (a, b) = (exact_law(&scm, x, y, z, 0), exact_law(&scm, x, y, z, 1));
                    if (a[0] - b[0]).abs() > 1e-6 || (a[1] - b[1]).abs() > 1e-6 {
                        assert_ne!(zero, one, "levels 0 and 1 emitted one functional");
                        distinct_arms += 1;
                    }
                }
            }
        }
    }
    assert!(identified > 100 && refused > 20, "{identified} identified, {refused} refused");
    assert!(moved_free > 20, "{moved_free} identified laws left a conditioning variable free");
    assert!(distinct_arms > 20, "{distinct_arms} cases with distinct arms");
}
