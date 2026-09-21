//! R13 `tests-quality-5`: numeric identification fixtures the transport sweeps
//! and status-only visibility pins omit.
//!
//! Each case builds a hand-derived binary SCM, identifies from the graph alone,
//! and checks the identified expression or its numeric value against truncated
//! factorization — never against `identify() == Ok` alone.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation, clippy::too_many_lines)]

use std::sync::Arc;

use antecedent_core::{
    AverageEffectQuery, CausalQuery, IdentificationStatus, Intervention, ResponseFunctional,
    ResponseQuery, Value, VariableId,
};
use antecedent_expr::{Assignment, DistributionProvider, EvalContext, EvalError, FactorSpec};
use antecedent_graph::{Admg, DenseNodeId, MarkedEdge, Pag};
use antecedent_identify::{
    IdIdentifier, IdentificationResult, IdentificationWorkspace, identify_pag_response_general,
};
use serde_json::Value as Json;

fn v(i: usize) -> VariableId {
    VariableId::from_raw(i as u32)
}
fn d(i: usize) -> DenseNodeId {
    DenseNodeId::from_raw(i as u32)
}
fn level(value: usize) -> Value {
    Value::f64(value as f64)
}
const fn bit(word: usize, index: usize) -> usize {
    (word >> index) & 1
}

fn fixture() -> Json {
    serde_json::from_str(include_str!("../../../conformance/identify/r13_quality5/expected.json"))
        .unwrap()
}

fn case<'a>(fixture: &'a Json, id: &str) -> &'a Json {
    fixture["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["id"] == id)
        .unwrap_or_else(|| panic!("missing fixture case {id}"))
}

fn mass(joint: &[f64], constraints: &[(usize, usize)]) -> f64 {
    joint
        .iter()
        .enumerate()
        .filter(|(cell, _)| constraints.iter().all(|&(node, value)| bit(*cell, node) == value))
        .map(|(_, p)| *p)
        .sum()
}

struct JointProvider {
    joint: Vec<f64>,
    /// When true, do-targets outside `variables` are conditioned on (MAG ID).
    condition_on_interventions: bool,
}

impl JointProvider {
    fn observational(joint: Vec<f64>) -> Self {
        Self { joint, condition_on_interventions: false }
    }

    fn mag(joint: Vec<f64>) -> Self {
        Self { joint, condition_on_interventions: true }
    }

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
        if self.condition_on_interventions {
            // MAG adjustment leaves do-targets out of `conditioned_on`.
            for assignment_do in spec.intervention {
                if spec.variables.contains(&assignment_do.variable) {
                    continue;
                }
                let value = assignment
                    .get(assignment_do.variable)
                    .and_then(Value::as_f64)
                    .ok_or(EvalError::MissingBinding(assignment_do.variable))?;
                constraints
                    .push((assignment_do.variable.raw() as usize, usize::from(value != 0.0)));
            }
        }
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

/// Evaluate a scalar functional, requiring every free-variable assignment to
/// agree (napkin-style free pre-treatment variables; ATE contrasts).
fn evaluate_constant(
    res: &IdentificationResult,
    provider: &JointProvider,
    truth: f64,
    context: &str,
) {
    let functional = res.estimands[0].functional;
    let arena = res.arena.clone();
    let free = arena.free_variables(functional);
    let plan = res.arena.compile(functional).unwrap();
    for row in 0..1usize << free.len().max(0) {
        let mut env = Assignment::new();
        for (i, var) in free.iter().enumerate() {
            env.set(*var, level(bit(row, i)));
        }
        let actual =
            plan.evaluate_with(&res.arena, provider, &EvalContext::default(), &env).unwrap();
        assert!(
            (actual - truth).abs() < 1e-12,
            "{context}: {} = {actual}, truth {truth} at free row {row}",
            res.arena.pretty(functional),
        );
    }
}

fn evaluate_free(
    res: &IdentificationResult,
    provider: &JointProvider,
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
            (actual - expected).abs() < 1e-12,
            "{context}: {} = {actual}, truth {expected} at free bits {bits:b}",
            res.arena.pretty(functional),
        );
    }
}

fn parent_free_joint() -> Vec<f64> {
    // X=0, Y=1; P(X=1)=0.4, P(Y=1|X)=0.15+0.5 X. Cell bit i is V_i.
    let mut joint = vec![0.0; 4];
    for x in 0..2 {
        let px = if x == 1 { 0.4 } else { 0.6 };
        for y in 0..2 {
            let py = 0.15 + 0.5 * x as f64;
            joint[x | (y << 1)] += px * if y == 1 { py } else { 1.0 - py };
        }
    }
    joint
}

fn with_parent_joint() -> Vec<f64> {
    // Z=0, X=1, Y=2; P(Z=1)=0.35, P(X=1|Z)=0.2+0.45 Z, P(Y=1|X,Z)=0.1+0.4 X+0.25 Z
    let mut joint = vec![0.0; 8];
    for z in 0..2 {
        let pz = if z == 1 { 0.35 } else { 0.65 };
        for x in 0..2 {
            let px = 0.2 + 0.45 * z as f64;
            for y in 0..2 {
                let py = 0.1 + 0.4 * x as f64 + 0.25 * z as f64;
                joint[z | (x << 1) | (y << 2)] +=
                    pz * if x == 1 { px } else { 1.0 - px } * if y == 1 { py } else { 1.0 - py };
            }
        }
    }
    joint
}

#[allow(clippy::needless_range_loop)] // the index is a node/row id shared by several parallel tables
fn napkin_joint(do_x: Option<usize>) -> Vec<f64> {
    let latent_p = [0.35, 0.7];
    let mut out = vec![0.0; 16];
    for u in 0..4 {
        let (u1, u2) = (bit(u, 0), bit(u, 1));
        let mut latent_mass = 1.0;
        for (k, p) in latent_p.iter().enumerate() {
            latent_mass *= if bit(u, k) == 1 { *p } else { 1.0 - *p };
        }
        for obs in 0..16 {
            let (w, z, x, y) = (bit(obs, 0), bit(obs, 1), bit(obs, 2), bit(obs, 3));
            let mut mass = latent_mass;
            let pw = 0.2 + 0.3 * u1 as f64 + 0.25 * u2 as f64;
            mass *= if w == 1 { pw } else { 1.0 - pw };
            let pz = 0.3 + 0.45 * w as f64;
            mass *= if z == 1 { pz } else { 1.0 - pz };
            if let Some(forced) = do_x {
                if x != forced {
                    continue;
                }
            } else {
                let px = 0.15 + 0.5 * z as f64 + 0.2 * u1 as f64;
                mass *= if x == 1 { px } else { 1.0 - px };
            }
            let py = 0.1 + 0.35 * x as f64 + 0.4 * u2 as f64;
            mass *= if y == 1 { py } else { 1.0 - py };
            out[obs] += mass;
        }
    }
    out
}

#[allow(clippy::needless_range_loop)] // the index is a node/row id shared by several parallel tables
fn discriminating_joint(do_t: Option<usize>) -> Vec<f64> {
    // Observed A=0,Q=1,C=2,T=3,Y=4; latents L1,L2 for Q<->C and C<->T.
    let mut out = vec![0.0; 1 << 5];
    for l1 in 0..2 {
        for l2 in 0..2 {
            let latent_mass =
                (if l1 == 1 { 0.4 } else { 0.6 }) * (if l2 == 1 { 0.55 } else { 0.45 });
            for obs in 0..1 << 5 {
                let (a, q, c, t, y) =
                    (bit(obs, 0), bit(obs, 1), bit(obs, 2), bit(obs, 3), bit(obs, 4));
                let mut mass = latent_mass;
                mass *= if a == 1 { 0.4 } else { 0.6 };
                let pq = 0.15 + 0.35 * a as f64 + 0.25 * f64::from(l1);
                mass *= if q == 1 { pq } else { 1.0 - pq };
                let pc = 0.2 + 0.3 * f64::from(l1) + 0.25 * f64::from(l2);
                mass *= if c == 1 { pc } else { 1.0 - pc };
                if let Some(forced) = do_t {
                    if t != forced {
                        continue;
                    }
                } else {
                    let pt = 0.25 + 0.4 * f64::from(l2);
                    mass *= if t == 1 { pt } else { 1.0 - pt };
                }
                let py = 0.05 + 0.2 * q as f64 + 0.25 * c as f64 + 0.35 * t as f64;
                mass *= if y == 1 { py } else { 1.0 - py };
                out[obs] += mass;
            }
        }
    }
    out
}

fn insert_admg_edges(graph: &mut Admg, edges: &[&str], index: impl Fn(&str) -> usize) {
    for edge in edges {
        if let Some((a, b)) = edge.split_once("<->") {
            graph.insert_bidirected(d(index(a)), d(index(b))).unwrap();
        } else {
            let (a, b) = edge.split_once("->").expect("edge syntax");
            graph.insert_directed(d(index(a)), d(index(b))).unwrap();
        }
    }
}

fn insert_pag_edges(pag: &mut Pag, edges: &[&str], index: impl Fn(&str) -> usize) {
    for edge in edges {
        if let Some((a, b)) = edge.split_once("<->") {
            pag.insert_marked(MarkedEdge::bidirected(d(index(a)), d(index(b)))).unwrap();
        } else {
            let (a, b) = edge.split_once("->").expect("edge syntax");
            pag.insert_directed(d(index(a)), d(index(b))).unwrap();
        }
    }
}

/// Treatment with an observed parent: the identifying functional is the
/// covariate-adjusted mean, not the parent-free empty-backdoor `P(y|x)`, and
/// the numeric ATE matches truncated factorization on a small parameter sweep.
#[test]
fn treatment_with_observed_parent_changes_functional_and_matches_scm_ate() {
    let fixture = fixture();
    let pin = case(&fixture, "treatment_with_observed_parent");
    let id = IdIdentifier::new();
    let mut ws = IdentificationWorkspace::default();

    let mut parent_free = Admg::with_variables(2);
    let parent_free_edges: Vec<&str> =
        pin["parent_free_edges"].as_array().unwrap().iter().map(|e| e.as_str().unwrap()).collect();
    insert_admg_edges(&mut parent_free, &parent_free_edges, |name| match name {
        "X" => 0,
        "Y" => 1,
        other => panic!("parent-free node {other}"),
    });
    let prep_free = id.prepare(&parent_free).unwrap();
    let free = id
        .identify(
            &prep_free,
            &CausalQuery::AverageEffect(AverageEffectQuery::with_levels(v(0), v(1), 0.0, 1.0)),
            &mut ws,
        )
        .unwrap();
    assert_eq!(free.status, IdentificationStatus::NonparametricallyIdentified);
    let free_pretty = free.arena.pretty(free.estimands[0].functional);
    assert!(
        !free_pretty.contains('Σ'),
        "parent-free identification must be empty-backdoor, got {free_pretty}"
    );
    evaluate_constant(
        &free,
        &JointProvider::observational(parent_free_joint()),
        pin["parent_free_expected_ate"].as_f64().unwrap(),
        "parent-free ATE",
    );

    let nodes: Vec<&str> =
        pin["nodes"].as_array().unwrap().iter().map(|n| n.as_str().unwrap()).collect();
    let index = |name: &str| nodes.iter().position(|n| *n == name).unwrap();
    let mut with_parent = Admg::with_variables(nodes.len() as u32);
    let edges: Vec<&str> =
        pin["edges"].as_array().unwrap().iter().map(|e| e.as_str().unwrap()).collect();
    insert_admg_edges(&mut with_parent, &edges, index);
    let prep = id.prepare(&with_parent).unwrap();
    let query = CausalQuery::AverageEffect(AverageEffectQuery::with_levels(
        v(index("X")),
        v(index("Y")),
        0.0,
        1.0,
    ));

    // Small numeric sweep over the directed coefficient of X while Z remains a parent.
    for (shift, expected_ate) in [(0.4, 0.4), (0.25, 0.25), (0.55, 0.55)] {
        let mut joint = vec![0.0; 8];
        for z in 0..2 {
            let pz = if z == 1 { 0.35 } else { 0.65 };
            for x in 0..2 {
                let px = 0.2 + 0.45 * z as f64;
                for y in 0..2 {
                    let py = 0.1 + shift * x as f64 + 0.25 * z as f64;
                    joint[z | (x << 1) | (y << 2)] += pz
                        * if x == 1 { px } else { 1.0 - px }
                        * if y == 1 { py } else { 1.0 - py };
                }
            }
        }
        let res = id.identify(&prep, &query, &mut ws).unwrap();
        assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);
        let pretty = res.arena.pretty(res.estimands[0].functional);
        assert_ne!(pretty, free_pretty, "observed parent must change the identifying functional");
        assert!(
            pretty.contains('Σ') && pretty.contains("V0"),
            "expected adjustment over Z (=V0), got {pretty}"
        );
        // g-formula on the joint: Σ_z [P(Y=1|X=1,Z=z) - P(Y=1|X=0,Z=z)] P(Z=z)
        let mut g_ate = 0.0;
        for z in 0..2 {
            let pz = mass(&joint, &[(0, z)]);
            let ey1 = mass(&joint, &[(0, z), (1, 1), (2, 1)]) / mass(&joint, &[(0, z), (1, 1)]);
            let ey0 = mass(&joint, &[(0, z), (1, 0), (2, 1)]) / mass(&joint, &[(0, z), (1, 0)]);
            g_ate += (ey1 - ey0) * pz;
        }
        assert!((g_ate - expected_ate).abs() < 1e-12, "g-formula shift {shift}");
        evaluate_constant(
            &res,
            &JointProvider::observational(joint),
            expected_ate,
            &format!("with-parent shift {shift}"),
        );
    }

    let baseline = id.identify(&prep, &query, &mut ws).unwrap();
    evaluate_constant(
        &baseline,
        &JointProvider::observational(with_parent_joint()),
        pin["expected_ate"].as_f64().unwrap(),
        "with-parent baseline ATE",
    );
}

/// Napkin graph: identified expression is the free-`z` ratio, and its ATE equals
/// the truncated-factorization interventional contrast on the pinned SCM.
#[test]
fn napkin_identified_expression_matches_enumerated_interventional_ate() {
    let fixture = fixture();
    let pin = case(&fixture, "napkin");
    let nodes: Vec<&str> =
        pin["nodes"].as_array().unwrap().iter().map(|n| n.as_str().unwrap()).collect();
    let index = |name: &str| nodes.iter().position(|n| *n == name).unwrap();
    let mut graph = Admg::with_variables(nodes.len() as u32);
    insert_admg_edges(
        &mut graph,
        &pin["edges"].as_array().unwrap().iter().map(|e| e.as_str().unwrap()).collect::<Vec<_>>(),
        index,
    );

    let id = IdIdentifier::new();
    let prep = id.prepare(&graph).unwrap();
    let mut ws = IdentificationWorkspace::default();
    let dist = id
        .identify(
            &prep,
            &CausalQuery::Distribution(
                antecedent_core::InterventionalDistributionQuery::new(
                    v(index("Y")),
                    [Intervention::set(v(index("X")), level(1))],
                )
                .with_outcomes([v(index("Y"))]),
            ),
            &mut ws,
        )
        .unwrap();
    assert_eq!(dist.status, IdentificationStatus::NonparametricallyIdentified);
    let lines: Vec<&str> = dist
        .derivation
        .steps
        .iter()
        .filter_map(|step| step.rule.strip_prefix("general.id."))
        .collect();
    assert_eq!(
        lines,
        pin["lines"].as_array().unwrap().iter().map(|s| s.as_str().unwrap()).collect::<Vec<_>>()
    );
    let pretty = dist.arena.pretty(dist.estimands[0].functional);
    assert!(
        pretty.contains('/') || pretty.contains("Σ"),
        "napkin must surface the ratio-of-marginals form, got {pretty}"
    );
    let arena = dist.arena.clone();
    assert!(
        arena.free_variables(dist.estimands[0].functional).contains(&v(index("Z"))),
        "napkin keeps z free: {pretty}"
    );

    let provider = JointProvider::observational(napkin_joint(None));
    let do_x = [napkin_joint(Some(0)), napkin_joint(Some(1))];
    let expected_do: Vec<f64> =
        pin["expected_do_y1"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
    for (x, law) in do_x.iter().enumerate() {
        let truth = mass(law, &[(index("Y"), 1)]);
        assert!((truth - expected_do[x]).abs() < 1e-12, "pinned do mean x={x}");
    }
    evaluate_free(
        &dist,
        &provider,
        |bits| mass(&do_x[bit(bits, index("X"))], &[(index("Y"), bit(bits, index("Y")))]),
        "napkin distribution",
    );

    let ate = id
        .identify_ate(
            &prep,
            &AverageEffectQuery::with_levels(v(index("X")), v(index("Y")), 0.0, 1.0),
            &mut ws,
        )
        .unwrap();
    assert_eq!(ate.status, IdentificationStatus::NonparametricallyIdentified);
    evaluate_constant(&ate, &provider, pin["expected_ate"].as_f64().unwrap(), "napkin ATE");
}

/// Discriminating-path visibility: `T→Y` is invisible without the path-start
/// witness `A`, and with it the adjustment functional recovers the SCM ATE.
#[allow(clippy::float_cmp)] // exact constants: the values compared are representable results, not measurements
#[test]
fn discriminating_path_visibility_matches_enumerated_effect() {
    let fixture = fixture();
    let pin = case(&fixture, "discriminating_path_visibility");
    let nodes: Vec<&str> =
        pin["nodes"].as_array().unwrap().iter().map(|n| n.as_str().unwrap()).collect();
    let index = |name: &str| nodes.iter().position(|n| *n == name).unwrap();

    let mut mag = Pag::with_variables(nodes.len() as u32);
    insert_pag_edges(
        &mut mag,
        &pin["edges"].as_array().unwrap().iter().map(|e| e.as_str().unwrap()).collect::<Vec<_>>(),
        index,
    );
    let query = |t_level: f64| {
        ResponseQuery::new(ResponseFunctional::InterventionResponse {
            outcome: v(index("Y")),
            interventions: Arc::from([Intervention::set(v(index("T")), Value::f64(t_level))]),
        })
    };
    let env1 = identify_pag_response_general(&mag, &query(1.0)).unwrap();
    assert_eq!(env1.status, IdentificationStatus::NonparametricallyIdentified);
    assert_eq!(env1.identified_weight.0, pin["identified_weight"].as_f64().unwrap());
    assert_eq!(env1.unidentified_weight.0, pin["unidentified_weight"].as_f64().unwrap());
    let res1 = &env1.cases[0].result;
    assert_eq!(res1.estimands[0].method.as_ref(), pin["method"].as_str().unwrap());
    let pretty = res1.arena.pretty(res1.estimands[0].functional);
    assert!(
        pretty.contains('Σ') && pretty.contains("V1") && pretty.contains("V2"),
        "discriminating-path visibility adjusts over Q,C; got {pretty}"
    );

    let provider = JointProvider::mag(discriminating_joint(None));
    let expected_do: Vec<f64> =
        pin["expected_do_y1"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
    for (t, &expected) in expected_do.iter().enumerate() {
        let env = identify_pag_response_general(&mag, &query(t as f64)).unwrap();
        let res = &env.cases[0].result;
        let truth = mass(&discriminating_joint(Some(t)), &[(index("Y"), 1)]);
        assert!((truth - expected).abs() < 1e-12, "SCM do mean t={t}");
        evaluate_constant(res, &provider, expected, &format!("discriminating-path E[Y|do(T={t})]"));
    }
    evaluate_constant(
        &identify_pag_response_general(&mag, &query(1.0)).unwrap().cases[0].result,
        &provider,
        expected_do[1],
        "discriminating-path treated mean",
    );
    evaluate_constant(
        &identify_pag_response_general(&mag, &query(0.0)).unwrap().cases[0].result,
        &provider,
        expected_do[0],
        "discriminating-path control mean",
    );
    assert!(
        (expected_do[1] - expected_do[0] - pin["expected_ate"].as_f64().unwrap()).abs() < 1e-12
    );

    // Without A the collider spine never leaves a vertex non-adjacent to Y.
    let without_nodes: Vec<&str> = pin["without_witness_nodes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| n.as_str().unwrap())
        .collect();
    let without_index =
        |name: &str| without_nodes.iter().position(|n| *n == name).expect("without-witness node");
    let mut invisible = Pag::with_variables(without_nodes.len() as u32);
    insert_pag_edges(
        &mut invisible,
        &pin["without_witness_edges"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| e.as_str().unwrap())
            .collect::<Vec<_>>(),
        without_index,
    );
    let refused = identify_pag_response_general(
        &invisible,
        &ResponseQuery::new(ResponseFunctional::InterventionResponse {
            outcome: v(without_index("Y")),
            interventions: Arc::from([Intervention::set(v(without_index("T")), Value::f64(1.0))]),
        }),
    )
    .unwrap();
    assert_eq!(refused.status, IdentificationStatus::NotIdentified);
    assert_eq!(refused.identified_weight.0, 0.0);
}
