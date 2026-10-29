//! The whole-program joint intervention on an unrolled panel graph, against exactly enumerated
//! binary SCMs.
//!
//! Each case declares a panel structure (periods, channels, context, carryover reach, outcome and
//! channel persistence, history feedback, latent confounding), builds a fully specified binary SCM
//! with one latent per bidirected edge and strictly positive random mechanisms, and compares the
//! general-ID functional of `do(every channel at every period)` -- evaluated on the observed joint
//! only -- with the interventional law enumerated in the SCM. A non-identifiable structure (a bow
//! between a channel and the outcome it drives) must stay refused with a hedge that verifies
//! against the unrolled graph. The oracle never consults the identifier.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    reason = "test scaffolding compares exact constants and indexes with small literals"
)]

use std::sync::Arc;

use antecedent_core::{CausalRng, Value, VariableId};
use antecedent_expr::{Assignment, DistributionProvider, EvalContext, EvalError, FactorSpec};
use antecedent_identify::IdentificationResult;
use antecedent_identify::panel_unrolled::{
    PanelNode, UnrolledPanelGraph, UnrolledPanelIdentification, UnrolledPanelSpec,
    identify_joint_panel_intervention,
};

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

fn spec(periods: usize, channels: usize) -> UnrolledPanelSpec {
    UnrolledPanelSpec { periods, channels, ..UnrolledPanelSpec::default() }
}

/// The directed and bidirected edges of a spec, derived here from the structural definition.
fn edges(graph: &UnrolledPanelGraph) -> (Vec<(usize, usize)>, Vec<(usize, usize)>) {
    let s = graph.spec();
    let ix = |n: PanelNode| graph.node_index(n).expect("node");
    let mut directed = Vec::new();
    for z in 0..s.contexts {
        for t in 0..s.periods {
            for c in 0..s.channels {
                directed.push((
                    ix(PanelNode::Context(z)),
                    ix(PanelNode::Channel { channel: c, period: t }),
                ));
            }
            directed.push((ix(PanelNode::Context(z)), ix(PanelNode::Outcome(t))));
        }
    }
    for t in 0..s.periods {
        for c in 0..s.channels {
            for u in t..=(t + s.carryover_reach).min(s.periods - 1) {
                directed.push((
                    ix(PanelNode::Channel { channel: c, period: t }),
                    ix(PanelNode::Outcome(u)),
                ));
            }
            if t > 0 && s.channel_persistence {
                directed.push((
                    ix(PanelNode::Channel { channel: c, period: t - 1 }),
                    ix(PanelNode::Channel { channel: c, period: t }),
                ));
            }
            if t > 0 && s.history_feedback {
                directed.push((
                    ix(PanelNode::Outcome(t - 1)),
                    ix(PanelNode::Channel { channel: c, period: t }),
                ));
            }
        }
        if t > 0 && s.outcome_persistence {
            directed.push((ix(PanelNode::Outcome(t - 1)), ix(PanelNode::Outcome(t))));
        }
    }
    directed.sort_unstable();
    directed.dedup();
    let bidirected = s.latent_confounding.iter().map(|&(a, b)| (ix(a), ix(b))).collect();
    (directed, bidirected)
}

fn scm_for(graph: &UnrolledPanelGraph, rng: &mut CausalRng) -> Scm {
    let n = graph.node_index(PanelNode::Outcome(graph.spec().periods - 1)).unwrap() + 1;
    let (directed, bidirected) = edges(graph);
    Scm::random(n, directed, bidirected, rng)
}

/// Check the identified functional of every outcome-period set against the exact SCM law.
fn check_identified(label: &str, panel: UnrolledPanelSpec, outcome_sets: &[Vec<usize>]) {
    let graph = UnrolledPanelGraph::build(panel).unwrap();
    let treatments: Vec<usize> = graph.program_nodes().iter().map(|t| t.raw() as usize).collect();
    for outcome_periods in outcome_sets {
        let UnrolledPanelIdentification::Identified(result) =
            identify_joint_panel_intervention(&graph, outcome_periods).unwrap()
        else {
            panic!("{label} {outcome_periods:?}: expected an identified functional");
        };
        let outcomes: Vec<usize> = outcome_periods
            .iter()
            .map(|&p| graph.node_index(PanelNode::Outcome(p)).unwrap())
            .collect();
        for seed in 0..3u64 {
            let mut rng = CausalRng::from_seed(0x9A4E_1000 + seed);
            let scm = scm_for(&graph, &mut rng);
            let provider = JointProvider { joint: scm.joint(&[]) };
            let laws: Vec<Vec<f64>> = (0..1usize << treatments.len())
                .map(|mask| {
                    let fixed: Vec<(usize, usize)> =
                        treatments.iter().enumerate().map(|(k, &t)| (t, bit(mask, k))).collect();
                    scm.joint(&fixed)
                })
                .collect();
            assert_matches_truth(
                &result,
                &provider,
                1e-9,
                |bits| {
                    let mask = treatments
                        .iter()
                        .enumerate()
                        .map(|(k, &t)| bit(bits, t) << k)
                        .sum::<usize>();
                    let constraints: Vec<(usize, usize)> =
                        outcomes.iter().map(|&o| (o, bit(bits, o))).collect();
                    mass(&laws[mask], &constraints)
                },
                &format!("{label} outcomes {outcome_periods:?} seed {seed}"),
            );
        }
    }
}

/// A static program (one period): two channels with a context, with and without a latent
/// common cause of the two channels.
#[test]
fn static_two_channel_program_matches_the_exact_scm_law() {
    check_identified("static", UnrolledPanelSpec { contexts: 1, ..spec(1, 2) }, &[vec![0]]);
    check_identified(
        "static channels share an unmeasured cause",
        UnrolledPanelSpec {
            contexts: 1,
            latent_confounding: vec![(
                PanelNode::Channel { channel: 0, period: 0 },
                PanelNode::Channel { channel: 1, period: 0 },
            )],
            ..spec(1, 2)
        },
        &[vec![0]],
    );
}

/// A three-period program (lag two) with carryover, serial dependence, channel persistence and
/// history feedback: every observed-variable DAG is identified, and the joint law of every
/// outcome subset is the SCM's.
#[test]
fn three_period_program_with_feedback_matches_the_exact_scm_law() {
    check_identified(
        "L=2 one channel",
        UnrolledPanelSpec {
            contexts: 1,
            carryover_reach: 2,
            outcome_persistence: true,
            channel_persistence: true,
            history_feedback: true,
            ..spec(3, 1)
        },
        &[vec![2], vec![0, 1, 2], vec![1]],
    );
    check_identified(
        "L=2 two channels, reach 1",
        UnrolledPanelSpec {
            carryover_reach: 1,
            outcome_persistence: true,
            history_feedback: true,
            ..spec(3, 2)
        },
        &[vec![2]],
    );
}

/// Latent confounding that general ID still resolves: the first channel is confounded with the
/// final outcome only, and acts on it through the intermediate outcome (a front-door), with
/// feedback of that outcome onto the second period's channel.
#[test]
fn temporal_front_door_through_the_intermediate_outcome_matches_the_exact_scm_law() {
    check_identified(
        "temporal front-door",
        UnrolledPanelSpec {
            outcome_persistence: true,
            history_feedback: true,
            latent_confounding: vec![(
                PanelNode::Channel { channel: 0, period: 0 },
                PanelNode::Outcome(1),
            )],
            ..spec(2, 1)
        },
        &[vec![1]],
    );
}

/// A bow between a channel and the outcome it drives directly is not identified, in a static
/// program and in a temporal one; the hedge verifies against the unrolled graph.
#[test]
fn a_bow_between_a_channel_and_its_outcome_stays_refused_with_the_hedge() {
    for (label, panel, outcome) in [
        (
            "static bow",
            UnrolledPanelSpec {
                latent_confounding: vec![(
                    PanelNode::Channel { channel: 0, period: 0 },
                    PanelNode::Outcome(0),
                )],
                ..spec(1, 1)
            },
            0,
        ),
        (
            "temporal bow at the later period",
            UnrolledPanelSpec {
                outcome_persistence: true,
                history_feedback: true,
                latent_confounding: vec![(
                    PanelNode::Channel { channel: 0, period: 1 },
                    PanelNode::Outcome(1),
                )],
                ..spec(2, 1)
            },
            1,
        ),
        (
            "carryover bow",
            UnrolledPanelSpec {
                carryover_reach: 1,
                latent_confounding: vec![(
                    PanelNode::Channel { channel: 0, period: 0 },
                    PanelNode::Outcome(1),
                )],
                ..spec(2, 1)
            },
            1,
        ),
    ] {
        let graph = UnrolledPanelGraph::build(panel).unwrap();
        match identify_joint_panel_intervention(&graph, &[outcome]).unwrap() {
            UnrolledPanelIdentification::NotIdentified { hedge, result } => {
                assert!(result.estimands.is_empty(), "{label}");
                hedge.verify_carried().unwrap_or_else(|e| panic!("{label}: {e}"));
            }
            UnrolledPanelIdentification::Identified(_) => {
                panic!("{label}: a bow must not be identified")
            }
        }
    }
}

/// Degenerate programs, nodes outside the program and an empty outcome set are typed errors.
#[test]
fn a_declaration_outside_the_program_or_degenerate_is_refused() {
    assert!(UnrolledPanelGraph::build(spec(0, 1)).is_err());
    assert!(UnrolledPanelGraph::build(spec(1, 0)).is_err());
    let outside = UnrolledPanelSpec {
        latent_confounding: vec![(
            PanelNode::Channel { channel: 3, period: 0 },
            PanelNode::Outcome(0),
        )],
        ..spec(1, 1)
    };
    assert!(UnrolledPanelGraph::build(outside).is_err());
    let selfie = UnrolledPanelSpec {
        latent_confounding: vec![(PanelNode::Outcome(0), PanelNode::Outcome(0))],
        ..spec(1, 1)
    };
    assert!(UnrolledPanelGraph::build(selfie).is_err());
    let graph = UnrolledPanelGraph::build(spec(2, 1)).unwrap();
    assert!(identify_joint_panel_intervention(&graph, &[]).is_err());
    assert!(identify_joint_panel_intervention(&graph, &[2]).is_err());
}

/// The functional is a function of the graph: the same structure without the bow identifies and
/// with it refuses, and an identified result retains no hedge.
#[test]
fn removing_the_bow_restores_identification() {
    let bow = UnrolledPanelSpec {
        latent_confounding: vec![(
            PanelNode::Channel { channel: 0, period: 0 },
            PanelNode::Outcome(0),
        )],
        ..spec(1, 1)
    };
    let without = spec(1, 1);
    let with = UnrolledPanelGraph::build(bow).unwrap();
    let clean = UnrolledPanelGraph::build(without).unwrap();
    assert!(matches!(
        identify_joint_panel_intervention(&with, &[0]).unwrap(),
        UnrolledPanelIdentification::NotIdentified { .. }
    ));
    let UnrolledPanelIdentification::Identified(result) =
        identify_joint_panel_intervention(&clean, &[0]).unwrap()
    else {
        panic!("identified");
    };
    assert!(result.hedge.is_none());
}
