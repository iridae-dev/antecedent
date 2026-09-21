//! Known-truth checks of Shpitser–Pearl ID against exactly enumerated binary SCMs.
//!
//! Every test builds a fully specified latent-variable SCM (one binary latent
//! per bidirected edge), enumerates the exact observational joint and the exact
//! interventional law by truncated factorization *in the SCM*, and compares the
//! identified functional — evaluated on the observational joint only — with
//! that truth. The oracle never consults the identifier.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation, clippy::too_many_lines)]

use std::sync::Arc;

use antecedent_core::{
    AverageEffectQuery, CausalQuery, CausalRng, Intervention, InterventionalDistributionQuery,
    Value, VariableId,
};
use antecedent_expr::{Assignment, DistributionProvider, EvalContext, EvalError, FactorSpec};
use antecedent_graph::{Admg, DenseNodeId};
use antecedent_identify::{
    IdIdentifier, IdcIdentifier, IdentificationResult, IdentificationStatus,
    IdentificationWorkspace,
};

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

/// Binary SCM over observed nodes `0..n` (a topological order) with one
/// independent binary latent per bidirected edge.
struct Scm {
    n: usize,
    directed: Vec<(usize, usize)>,
    bidirected: Vec<(usize, usize)>,
    latent_p: Vec<f64>,
    /// `tables[i][k] = P(V_i = 1 | inputs = k)`; input bits are the observed
    /// parents (ascending) followed by the incident latents (ascending).
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

    /// Seeded generic mechanisms: every conditional probability is drawn in
    /// `(0.12, 0.88)`, so the law is strictly positive and no parent is inert.
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

    /// Exact joint over the observed nodes (bit `i` of the index is `V_i`),
    /// under `do(node = value)` for every entry of `interventions`.
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
/// Intervention labels on a factor only bind values (the evaluator has already
/// written them into the assignment); the law consulted is always this one.
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

fn distribution_query(outcomes: &[usize], treatments: &[usize]) -> CausalQuery {
    let interventions: Vec<_> =
        treatments.iter().map(|&t| Intervention::set(v(t), level(1))).collect();
    CausalQuery::Distribution(
        InterventionalDistributionQuery::new(v(outcomes[0]), interventions)
            .with_outcomes(outcomes.iter().map(|&o| v(o)).collect::<Vec<_>>()),
    )
}

/// Evaluate the identified law at every assignment of its free variables and
/// compare with `truth(assignment bits)`. Free variables other than the
/// outcomes and treatments (line-3 interventions such as the napkin's `z`)
/// are therefore checked to leave the value unchanged.
fn assert_matches_truth(
    res: &IdentificationResult,
    provider: &JointProvider,
    tolerance: f64,
    truth: impl Fn(usize) -> f64,
    context: &str,
) {
    let functional = res.estimands[0].functional;
    let arena = res.arena.clone();
    let free = arena.free_variables(functional);
    let plan = res.arena.compile(functional).unwrap();
    for row in 0..1usize << free.len() {
        let mut bits = 0usize;
        let mut env = Assignment::new();
        for (i, var) in free.iter().enumerate() {
            let value = bit(row, i);
            bits |= value << var.raw();
            env.set(*var, level(value));
        }
        let actual =
            plan.evaluate_with(&res.arena, provider, &EvalContext::default(), &env).unwrap();
        let expected = truth(bits);
        assert!(
            (actual - expected).abs() < tolerance,
            "{context}: functional {} = {actual}, truth {expected} at free bits {bits:b}",
            res.arena.pretty(functional),
        );
    }
}

/// `W -> Z -> X -> Y`, `W <-> X`, `W <-> Y` with nodes `W=0, Z=1, X=2, Y=3`.
fn napkin() -> Scm {
    let mut scm = Scm {
        n: 4,
        directed: vec![(0, 1), (1, 2), (2, 3)],
        bidirected: vec![(0, 2), (0, 3)],
        latent_p: vec![0.35, 0.7],
        tables: Vec::new(),
    };
    let b = |index: usize, slot: usize| bit(index, slot) as f64;
    // W | u1, u2 ; Z | w ; X | z, u1 ; Y | x, u2
    scm.tables.push((0..4).map(|k| 0.2 + 0.3 * b(k, 0) + 0.25 * b(k, 1)).collect());
    scm.tables.push((0..2).map(|k| 0.3 + 0.45 * b(k, 0)).collect());
    scm.tables.push((0..4).map(|k| 0.15 + 0.5 * b(k, 0) + 0.2 * b(k, 1)).collect());
    scm.tables.push((0..4).map(|k| 0.1 + 0.35 * b(k, 0) + 0.4 * b(k, 1)).collect());
    scm
}

#[test]
fn napkin_distribution_matches_exact_intervention_for_every_z() {
    let scm = napkin();
    let id = IdIdentifier::new();
    let prep = id.prepare(&scm.admg()).unwrap();
    let mut ws = IdentificationWorkspace::default();
    let res = id.identify(&prep, &distribution_query(&[3], &[2]), &mut ws).unwrap();
    assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);
    assert!(
        res.derivation.steps.iter().any(|s| s.rule.as_ref() == "general.id.line7"),
        "the napkin is identified only through line 7"
    );

    let provider = JointProvider { joint: scm.joint(&[]) };
    let do_x = [scm.joint(&[(2, 0)]), scm.joint(&[(2, 1)])];
    // Independent closed form: sum_w P(x,y|z,w)P(w) / sum_w P(x|z,w)P(w), any z.
    for (x, law) in do_x.iter().enumerate() {
        for y in 0..2 {
            let truth = mass(law, &[(3, y)]);
            for z in 0..2 {
                let (mut num, mut den) = (0.0, 0.0);
                for w in 0..2 {
                    let pw = mass(&provider.joint, &[(0, w)]);
                    let pzw = mass(&provider.joint, &[(0, w), (1, z)]);
                    num += mass(&provider.joint, &[(0, w), (1, z), (2, x), (3, y)]) / pzw * pw;
                    den += mass(&provider.joint, &[(0, w), (1, z), (2, x)]) / pzw * pw;
                }
                assert!((num / den - truth).abs() < 1e-12, "textbook napkin formula");
            }
        }
    }
    assert_matches_truth(
        &res,
        &provider,
        1e-10,
        |bits| mass(&do_x[bit(bits, 2)], &[(3, bit(bits, 3))]),
        "napkin",
    );
    // The functional must actually range over z for the independence check to bite.
    let arena = res.arena.clone();
    assert!(arena.free_variables(res.estimands[0].functional).contains(&v(1)));
}

#[test]
fn napkin_ate_contrast_matches_exact_intervention() {
    let scm = napkin();
    let id = IdIdentifier::new();
    let prep = id.prepare(&scm.admg()).unwrap();
    let mut ws = IdentificationWorkspace::default();
    let res =
        id.identify_ate(&prep, &AverageEffectQuery::with_levels(v(2), v(3), 0.0, 1.0), &mut ws);
    let res = res.unwrap();
    assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);
    let provider = JointProvider { joint: scm.joint(&[]) };
    let ate = mass(&scm.joint(&[(2, 1)]), &[(3, 1)]) - mass(&scm.joint(&[(2, 0)]), &[(3, 1)]);
    assert_matches_truth(&res, &provider, 1e-10, |_| ate, "napkin ATE");
}

/// Bounded exhaustive evidence that ID is complete and sound.
///
/// Domain: every ADMG on 2, 3 and 4 nodes whose directed edges respect the
/// fixed order `0 < 1 < …` (all `2^m` directed × `2^m` bidirected subsets of
/// the `m` node pairs: 4 + 64 + 4096 graphs), crossed with every ordered
/// (treatment, outcome) pair — so treatments with observed parents, outcomes
/// upstream of the treatment and bow arcs are all covered: 50 072 queries.
///
/// Each query must end in exactly one of two ways; an `Err` is a failure.
///  (i) identified: the functional, evaluated on the observational joint of
///      `PARAMETERIZATIONS` independent random positive binary SCMs, equals
///      the SCM's own interventional law for every value of every free
///      variable;
///  (ii) not identified: the certificate is a hedge for the original query.
///
/// Caps: 4 nodes, binary variables, single treatment × single outcome, two
/// parameterizations per graph.
#[test]
fn every_small_admg_query_is_identified_correctly_or_refuted_by_a_verified_hedge() {
    const PARAMETERIZATIONS: usize = 2;
    let id = IdIdentifier::new();
    let mut ws = IdentificationWorkspace::default();
    let mut rng = CausalRng::from_seed(0x001D_C0DE);
    let (mut identified, mut hedged, mut ratio_forms) = (0usize, 0usize, 0usize);
    for n in 2..=4usize {
        let pairs: Vec<(usize, usize)> =
            (0..n).flat_map(|a| (a + 1..n).map(move |b| (a, b))).collect();
        let pick = |mask: usize| -> Vec<(usize, usize)> {
            pairs.iter().enumerate().filter(|(k, _)| bit(mask, *k) == 1).map(|(_, e)| *e).collect()
        };
        for directed_mask in 0..1usize << pairs.len() {
            for bidirected_mask in 0..1usize << pairs.len() {
                let scms: Vec<Scm> = (0..PARAMETERIZATIONS)
                    .map(|_| Scm::random(n, pick(directed_mask), pick(bidirected_mask), &mut rng))
                    .collect();
                let providers: Vec<JointProvider> =
                    scms.iter().map(|scm| JointProvider { joint: scm.joint(&[]) }).collect();
                let prep = id.prepare(&scms[0].admg()).unwrap();
                for x in 0..n {
                    let mut do_x: Option<Vec<[Vec<f64>; 2]>> = None;
                    for y in (0..n).filter(|&y| y != x) {
                        let context = format!(
                            "n={n} directed={:?} bidirected={:?} do({x}) -> {y}",
                            scms[0].directed, scms[0].bidirected
                        );
                        let res = id
                            .identify(&prep, &distribution_query(&[y], &[x]), &mut ws)
                            .unwrap_or_else(|e| panic!("{context}: ID returned Err({e})"));
                        match res.status {
                            IdentificationStatus::NonparametricallyIdentified => {
                                assert!(res.hedge.is_none(), "{context}");
                                identified += 1;
                                ratio_forms += usize::from(has_ratio(&res));
                                let truth = do_x.get_or_insert_with(|| {
                                    scms.iter()
                                        .map(|scm| [scm.joint(&[(x, 0)]), scm.joint(&[(x, 1)])])
                                        .collect()
                                });
                                for (provider, truth) in providers.iter().zip(truth.iter()) {
                                    assert_matches_truth(
                                        &res,
                                        provider,
                                        1e-9,
                                        |bits| mass(&truth[bit(bits, x)], &[(y, bit(bits, y))]),
                                        &context,
                                    );
                                }
                                // The contrast route bakes do-levels into the
                                // factors, which must not pin a summation
                                // variable. Checked on every ≤3-node query and
                                // on every ratio-form derivation.
                                if n <= 3 || has_ratio(&res) {
                                    let ate = id
                                        .identify_ate(
                                            &prep,
                                            &AverageEffectQuery::with_levels(v(x), v(y), 0.0, 1.0),
                                            &mut ws,
                                        )
                                        .unwrap_or_else(|e| panic!("{context}: ATE Err({e})"));
                                    for (provider, truth) in providers.iter().zip(truth.iter()) {
                                        let expected =
                                            mass(&truth[1], &[(y, 1)]) - mass(&truth[0], &[(y, 1)]);
                                        assert_matches_truth(
                                            &ate,
                                            provider,
                                            1e-9,
                                            |_| expected,
                                            &context,
                                        );
                                    }
                                }
                            }
                            IdentificationStatus::NotIdentified => {
                                hedged += 1;
                                assert!(res.estimands.is_empty(), "{context}");
                                let hedge = res.hedge.as_ref().unwrap_or_else(|| {
                                    panic!("{context}: not identified without a hedge")
                                });
                                hedge
                                    .verify(&prep, &[v(x)], &[v(y)])
                                    .unwrap_or_else(|e| panic!("{context}: {e}; {hedge:?}"));
                            }
                            other => panic!("{context}: unexpected status {other:?}"),
                        }
                    }
                }
            }
        }
    }
    assert_eq!(identified + hedged, 2 * 4 + 6 * 64 + 12 * 4096);
    // Both outcomes and the ratio-of-marginals derivations must actually occur.
    assert!(identified > 1_000 && hedged > 1_000, "identified={identified} hedged={hedged}");
    assert!(ratio_forms > 0, "no query exercised a conditional of a marginalized C-factor");
    println!("identified={identified} hedged={hedged} ratio_forms={ratio_forms}");
}

fn has_ratio(res: &IdentificationResult) -> bool {
    fn walk(arena: &antecedent_expr::CausalExprArena, id: antecedent_expr::ExprId) -> bool {
        match arena.node(id) {
            antecedent_expr::ExprNode::Ratio { .. } => true,
            antecedent_expr::ExprNode::Product(list) => {
                arena.list(*list).iter().any(|&e| walk(arena, e))
            }
            antecedent_expr::ExprNode::SumOut { expr, .. } => walk(arena, *expr),
            _ => false,
        }
    }
    walk(&res.arena, res.estimands[0].functional)
}

fn ratio_depth(res: &IdentificationResult) -> usize {
    fn walk(arena: &antecedent_expr::CausalExprArena, id: antecedent_expr::ExprId) -> usize {
        match arena.node(id) {
            antecedent_expr::ExprNode::Ratio { numerator, denominator } => {
                1 + walk(arena, *numerator).max(walk(arena, *denominator))
            }
            antecedent_expr::ExprNode::Product(list) => {
                arena.list(*list).iter().map(|&e| walk(arena, e)).max().unwrap_or(0)
            }
            antecedent_expr::ExprNode::SumOut { expr, .. } => walk(arena, *expr),
            _ => 0,
        }
    }
    walk(&res.arena, res.estimands[0].functional)
}

/// Seeded sample beyond the exhaustive domain: 5- and 6-node ADMGs with joint
/// treatments and joint outcomes (sets of size 1–2), same two-way contract.
/// The sample must reach the ratio-of-marginals derivations; the doubly nested
/// form is too rare to sample and has its own fixture below.
#[test]
fn sampled_larger_admgs_with_joint_queries_keep_the_two_way_contract() {
    let id = IdIdentifier::new();
    let mut ws = IdentificationWorkspace::default();
    let mut rng = CausalRng::from_seed(0x5EED);
    let (mut identified, mut hedged, mut deepest) = (0usize, 0usize, 0usize);
    for case in 0..1500usize {
        let n = 5 + case % 2;
        let pairs: Vec<(usize, usize)> =
            (0..n).flat_map(|a| (a + 1..n).map(move |b| (a, b))).collect();
        // Sparse-ish graphs: dense bidirected structure is almost always a hedge.
        let directed: Vec<_> = pairs.iter().copied().filter(|_| rng.next_u64() % 5 < 2).collect();
        let bidirected: Vec<_> = pairs.iter().copied().filter(|_| rng.next_u64() % 4 < 1).collect();
        let scm = Scm::random(n, directed, bidirected, &mut rng);
        let mut nodes: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            nodes.swap(i, (rng.next_u64() % (i as u64 + 1)) as usize);
        }
        let x_len = 1 + (rng.next_u64() % 2) as usize;
        let y_len = 1 + (rng.next_u64() % 2) as usize;
        let xs = &nodes[..x_len];
        let ys = &nodes[x_len..x_len + y_len];
        let context = format!(
            "case {case}: directed={:?} bidirected={:?} do({xs:?}) -> {ys:?}",
            scm.directed, scm.bidirected
        );
        let prep = id.prepare(&scm.admg()).unwrap();
        let res = id
            .identify(&prep, &distribution_query(ys, xs), &mut ws)
            .unwrap_or_else(|e| panic!("{context}: ID returned Err({e})"));
        let x_vars: Vec<_> = xs.iter().map(|&x| v(x)).collect();
        let y_vars: Vec<_> = ys.iter().map(|&y| v(y)).collect();
        if res.status == IdentificationStatus::NotIdentified {
            hedged += 1;
            let hedge = res.hedge.as_ref().expect("hedge");
            hedge
                .verify(&prep, &x_vars, &y_vars)
                .unwrap_or_else(|e| panic!("{context}: {e}; {hedge:?}"));
            continue;
        }
        assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified, "{context}");
        identified += 1;
        deepest = deepest.max(ratio_depth(&res));
        let provider = JointProvider { joint: scm.joint(&[]) };
        let truth: Vec<Vec<f64>> = (0..1usize << x_len)
            .map(|row| {
                let forced: Vec<_> =
                    xs.iter().enumerate().map(|(i, &x)| (x, bit(row, i))).collect();
                scm.joint(&forced)
            })
            .collect();
        assert_matches_truth(
            &res,
            &provider,
            1e-9,
            |bits| {
                let row = xs.iter().enumerate().fold(0, |acc, (i, &x)| acc | (bit(bits, x) << i));
                let wanted: Vec<_> = ys.iter().map(|&y| (y, bit(bits, y))).collect();
                mass(&truth[row], &wanted)
            },
            &context,
        );
        if x_len == 1 && y_len == 1 && ratio_depth(&res) > 0 {
            let ate = id
                .identify_ate(
                    &prep,
                    &AverageEffectQuery::with_levels(v(xs[0]), v(ys[0]), 0.0, 1.0),
                    &mut ws,
                )
                .unwrap_or_else(|e| panic!("{context}: ATE Err({e})"));
            let expected = mass(&truth[1], &[(ys[0], 1)]) - mass(&truth[0], &[(ys[0], 1)]);
            assert_matches_truth(&ate, &provider, 1e-9, |_| expected, &context);
        }
    }
    println!("identified={identified} hedged={hedged} deepest_ratio_nesting={deepest}");
    assert!(identified > 300 && hedged > 100, "identified={identified} hedged={hedged}");
    assert!(deepest >= 1, "sample never formed a conditional of a marginalized C-factor");
}

/// A napkin whose inner graph is again a napkin, so line 7 → line 2 happens
/// twice and the second C-factor is built from conditionals of an already
/// marginalized law (a ratio inside a ratio). Nodes
/// `W1=0, Z1=1, W2=2, Z2=3, X=4, Y=5`; `W1` ties `Z2` into the outer district
/// `{W1, W2, Z2, X, Y}` and is dropped by the first line 2, after which
/// `G[{W2, Z2, X, Y}]` is the ordinary napkin.
#[test]
fn nested_napkin_needs_a_conditional_of_a_conditional() {
    let mut rng = CausalRng::from_seed(0x000A_9C1A);
    let scm = Scm::random(
        6,
        vec![(0, 1), (1, 4), (2, 3), (3, 4), (4, 5)],
        vec![(0, 2), (0, 3), (2, 4), (2, 5)],
        &mut rng,
    );
    let id = IdIdentifier::new();
    let prep = id.prepare(&scm.admg()).unwrap();
    let mut ws = IdentificationWorkspace::default();
    let res = id.identify(&prep, &distribution_query(&[5], &[4]), &mut ws).unwrap();
    assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);
    let line7 = res.derivation.steps.iter().filter(|s| s.rule.as_ref() == "general.id.line7");
    assert_eq!(line7.count(), 2);
    assert_eq!(ratio_depth(&res), 2);

    let provider = JointProvider { joint: scm.joint(&[]) };
    let do_x = [scm.joint(&[(4, 0)]), scm.joint(&[(4, 1)])];
    assert_matches_truth(
        &res,
        &provider,
        1e-10,
        |bits| mass(&do_x[bit(bits, 4)], &[(5, bit(bits, 5))]),
        "nested napkin",
    );
    // Contrast route: X's own conditional is part of the second C-factor, so
    // its do-label must be dropped wherever X is a summation variable.
    let ate = id
        .identify_ate(&prep, &AverageEffectQuery::with_levels(v(4), v(5), 0.0, 1.0), &mut ws)
        .unwrap();
    let expected = mass(&do_x[1], &[(5, 1)]) - mass(&do_x[0], &[(5, 1)]);
    assert_matches_truth(&ate, &provider, 1e-10, |_| expected, "nested napkin ATE");
}

/// Front-door `X -> M -> Y`, `X <-> Y` is identified; adding `M <-> Y` joins
/// `{X, M, Y}` into one district, which is a hedge for `P(y | do(x))`.
#[test]
fn frontdoor_with_confounded_mediator_is_a_verified_hedge() {
    let mut g = Admg::with_variables(3);
    g.insert_directed(d(0), d(1)).unwrap();
    g.insert_directed(d(1), d(2)).unwrap();
    g.insert_bidirected(d(0), d(2)).unwrap();
    let id = IdIdentifier::new();
    let mut ws = IdentificationWorkspace::default();
    let query = distribution_query(&[2], &[0]);
    let frontdoor = id.identify(&id.prepare(&g).unwrap(), &query, &mut ws).unwrap();
    assert_eq!(frontdoor.status, IdentificationStatus::NonparametricallyIdentified);

    g.insert_bidirected(d(1), d(2)).unwrap();
    let prep = id.prepare(&g).unwrap();
    let res = id.identify(&prep, &query, &mut ws).unwrap();
    assert_eq!(res.status, IdentificationStatus::NotIdentified);
    assert!(res.estimands.is_empty());
    let hedge = res.hedge.expect("hedge certificate");
    assert_eq!(hedge.f.as_ref(), [v(0), v(1), v(2)]);
    assert_eq!(hedge.f_prime.as_ref(), [v(1), v(2)]);
    hedge.verify(&prep, &[v(0)], &[v(2)]).unwrap();

    // The verifier is a real check, not a rubber stamp.
    assert!(hedge.verify(&prep, &[v(1)], &[v(2)]).is_err(), "F' meets the treatment");
    assert!(hedge.verify(&prep, &[v(0)], &[v(0)]).is_err(), "roots are not ancestors of Y");
    let mut shrunk = hedge.clone();
    shrunk.f = shrunk.f_prime.clone();
    shrunk.f_dense = shrunk.f_prime_dense.clone();
    assert!(shrunk.verify(&prep, &[v(0)], &[v(2)]).is_err(), "F misses the treatment");
    assert!(
        hedge.verify(&id.prepare(&Admg::with_variables(3)).unwrap(), &[v(0)], &[v(2)]).is_err(),
        "no bidirected structure, no C-forest"
    );
}

/// IDC on every 3-node ADMG and every role assignment `(x, z, y)`, so the
/// conditioning variable is post-treatment, pre-treatment or unrelated:
/// identified ⇒ equals the SCM's `P(y | do(x), z) = P_x(y, z) / P_x(z)`;
/// otherwise not identified with a hedge. Never `Err`.
#[test]
fn idc_matches_exact_conditional_intervention_on_every_three_node_admg() {
    let idc = IdcIdentifier::new();
    let mut ws = IdentificationWorkspace::default();
    let mut rng = CausalRng::from_seed(0x1DC);
    let pairs = [(0usize, 1usize), (0, 2), (1, 2)];
    let pick = |mask: usize| -> Vec<(usize, usize)> {
        pairs.iter().enumerate().filter(|(k, _)| bit(mask, *k) == 1).map(|(_, e)| *e).collect()
    };
    let (mut identified, mut refused, mut post_treatment) = (0usize, 0usize, 0usize);
    for directed_mask in 0..8usize {
        for bidirected_mask in 0..8usize {
            let scm = Scm::random(3, pick(directed_mask), pick(bidirected_mask), &mut rng);
            let provider = JointProvider { joint: scm.joint(&[]) };
            let prep = idc.prepare(&scm.admg()).unwrap();
            for (x, z, y) in [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)] {
                let context = format!(
                    "directed={:?} bidirected={:?} P({y} | do({x}), {z})",
                    scm.directed, scm.bidirected
                );
                let query = CausalQuery::Distribution(
                    InterventionalDistributionQuery::new(v(y), [Intervention::set(v(x), level(1))])
                        .with_conditioning([v(z)]),
                );
                let res = idc
                    .identify(&prep, &query, &mut ws)
                    .unwrap_or_else(|e| panic!("{context}: IDC returned Err({e})"));
                if res.status == IdentificationStatus::NotIdentified {
                    refused += 1;
                    assert!(res.hedge.is_some(), "{context}");
                    continue;
                }
                assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);
                identified += 1;
                post_treatment += usize::from(scm.directed.contains(&(x, z)));
                let do_x = [scm.joint(&[(x, 0)]), scm.joint(&[(x, 1)])];
                assert_matches_truth(
                    &res,
                    &provider,
                    1e-9,
                    |bits| {
                        let law = &do_x[bit(bits, x)];
                        let z_value = (z, bit(bits, z));
                        mass(law, &[z_value, (y, bit(bits, y))]) / mass(law, &[z_value])
                    },
                    &context,
                );
            }
        }
    }
    assert_eq!(identified + refused, 6 * 64);
    assert!(post_treatment > 20 && refused > 20, "{identified} {refused} {post_treatment}");
}

/// A post-treatment conditioning variable that rule 2 cannot move into the
/// intervention: `X -> Z -> Y`, `X -> Y`, `Z <-> Y`. IDC must return
/// `P_x(y, z) / P_x(z)` with the latent `Z <-> Y` dependence intact.
#[test]
fn idc_post_treatment_conditioning_with_confounded_mediator() {
    let mut rng = CausalRng::from_seed(0x1DC2);
    let scm = Scm::random(3, vec![(0, 1), (0, 2), (1, 2)], vec![(1, 2)], &mut rng);
    let idc = IdcIdentifier::new();
    let prep = idc.prepare(&scm.admg()).unwrap();
    let mut ws = IdentificationWorkspace::default();
    let query = CausalQuery::Distribution(
        InterventionalDistributionQuery::new(v(2), [Intervention::set(v(0), level(1))])
            .with_conditioning([v(1)]),
    );
    let res = idc.identify(&prep, &query, &mut ws).unwrap();
    assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);
    assert!(
        !res.derivation.steps.iter().any(|s| s.rule.as_ref() == "general.idc.line1"),
        "Z <-> Y blocks rule 2, so Z must stay a conditioning variable"
    );
    let provider = JointProvider { joint: scm.joint(&[]) };
    let do_x = [scm.joint(&[(0, 0)]), scm.joint(&[(0, 1)])];
    assert_matches_truth(
        &res,
        &provider,
        1e-10,
        |bits| {
            let law = &do_x[bits & 1];
            mass(law, &[(1, bit(bits, 1)), (2, bit(bits, 2))]) / mass(law, &[(1, bit(bits, 1))])
        },
        "IDC post-treatment Z",
    );
}

/// `P(A | do(A))` is a point mass, not the observational `P(A)`; a query whose
/// outcomes overlap its interventions (or its conditioning set) is malformed
/// and must be refused before the recursion can emit anything.
#[test]
fn overlapping_outcome_and_intervention_is_refused() {
    let mut g = Admg::with_variables(2);
    g.insert_directed(d(0), d(1)).unwrap();
    let id = IdIdentifier::new();
    let idc = IdcIdentifier::new();
    let prep = id.prepare(&g).unwrap();
    let mut ws = IdentificationWorkspace::default();
    for outcomes in [vec![0usize], vec![0, 1]] {
        let query = distribution_query(&outcomes, &[0]);
        assert!(id.identify(&prep, &query, &mut ws).is_err(), "ID accepted Y ∩ X ≠ ∅");
        assert!(idc.identify(&prep, &query, &mut ws).is_err(), "IDC accepted Y ∩ X ≠ ∅");
    }
    let conditioning_is_outcome = CausalQuery::Distribution(
        InterventionalDistributionQuery::new(v(1), [Intervention::set(v(0), level(1))])
            .with_conditioning([v(1)]),
    );
    assert!(idc.identify(&prep, &conditioning_is_outcome, &mut ws).is_err());
    let no_outcomes = CausalQuery::Distribution(
        InterventionalDistributionQuery::new(v(1), [Intervention::set(v(0), level(1))])
            .with_outcomes(Vec::new()),
    );
    assert!(id.identify(&prep, &no_outcomes, &mut ws).is_err(), "ID accepted Y = ∅");
    // The schedule-contrast entry has no query object to validate.
    let schedule = id.identify_schedule_contrast(
        &prep,
        v(0),
        &[v(0)],
        &level(1),
        &level(0),
        distribution_query(&[0], &[0]),
        &mut ws,
    );
    assert!(schedule.is_err(), "schedule contrast accepted an intervened outcome");
}

/// Frozen worked examples: status, the published lines the derivation takes,
/// the hedge node sets, and — for identified cases — agreement with an
/// enumerated SCM on the same graph.
#[test]
fn frozen_id_cases_keep_status_lines_certificate_and_truth() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/identify/id_completeness/expected.json"
    ))
    .unwrap();
    let id = IdIdentifier::new();
    let mut ws = IdentificationWorkspace::default();
    let mut rng = CausalRng::from_seed(0xF1C5);
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 6);
    for case in cases {
        let name = case["id"].as_str().unwrap();
        let strings = |key: &str| -> Vec<&str> {
            case[key].as_array().unwrap().iter().map(|s| s.as_str().unwrap()).collect()
        };
        let nodes = strings("nodes");
        let index = |label: &str| nodes.iter().position(|n| *n == label).expect("declared node");
        let (mut directed, mut bidirected) = (Vec::new(), Vec::new());
        for edge in strings("edges") {
            if let Some((a, b)) = edge.split_once("<->") {
                bidirected.push((index(a), index(b)));
            } else {
                let (a, b) = edge.split_once("->").expect("edge syntax");
                directed.push((index(a), index(b)));
            }
        }
        let xs: Vec<usize> = strings("treatments").into_iter().map(index).collect();
        let ys: Vec<usize> = strings("outcomes").into_iter().map(index).collect();
        let scm = Scm::random(nodes.len(), directed, bidirected, &mut rng);
        let prep = id.prepare(&scm.admg()).unwrap();
        let res = id.identify(&prep, &distribution_query(&ys, &xs), &mut ws).unwrap();
        if case["status"] == "not_identified" {
            assert_eq!(res.status, IdentificationStatus::NotIdentified, "{name}");
            let hedge = res.hedge.as_ref().expect("hedge");
            let named = |vars: &[VariableId]| -> Vec<&str> {
                vars.iter().map(|var| nodes[var.raw() as usize]).collect()
            };
            let expected = |key: &str| -> Vec<&str> {
                case["hedge"][key].as_array().unwrap().iter().map(|s| s.as_str().unwrap()).collect()
            };
            assert_eq!(named(&hedge.f), expected("f"), "{name}");
            assert_eq!(named(&hedge.f_prime), expected("f_prime"), "{name}");
            let x_vars: Vec<_> = xs.iter().map(|&x| v(x)).collect();
            let y_vars: Vec<_> = ys.iter().map(|&y| v(y)).collect();
            hedge.verify(&prep, &x_vars, &y_vars).unwrap();
            continue;
        }
        assert_eq!(case["status"], "identified", "{name}");
        assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified, "{name}");
        let lines: Vec<&str> = res
            .derivation
            .steps
            .iter()
            .filter_map(|step| step.rule.strip_prefix("general.id."))
            .collect();
        assert_eq!(lines, strings("lines"), "{name}");
        let provider = JointProvider { joint: scm.joint(&[]) };
        let x = xs[0];
        let do_x = [scm.joint(&[(x, 0)]), scm.joint(&[(x, 1)])];
        assert_matches_truth(
            &res,
            &provider,
            1e-10,
            |bits| mass(&do_x[bit(bits, x)], &[(ys[0], bit(bits, ys[0]))]),
            name,
        );
    }
}
