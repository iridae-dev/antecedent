use super::super::subsets;
use super::*;
use antecedent_core::DistributionAvailability;
use antecedent_core::{
    DependenceGroup, EvidenceKind, EvidenceRegime, RegimeBinding, RegimeKind, SamplingDesign,
};

use crate::test_support::v;

fn v_of(i: usize) -> VariableId {
    VariableId::from_raw(u32::try_from(i).unwrap())
}

fn admg(n: u32, directed: &[(u32, u32)], bidirected: &[(u32, u32)]) -> Admg {
    let mut graph = Admg::with_variables(n);
    for (a, b) in directed {
        graph.insert_directed(DenseNodeId::from_raw(*a), DenseNodeId::from_raw(*b)).unwrap();
    }
    for (a, b) in bidirected {
        graph.insert_bidirected(DenseNodeId::from_raw(*a), DenseNodeId::from_raw(*b)).unwrap();
    }
    graph
}

fn catalog(regimes: &[(u32, &[u32])]) -> EvidenceCatalog {
    let regimes = regimes
        .iter()
        .map(|(id, measured)| {
            EvidenceRegime::try_new(
                RegimeId::from_raw(*id),
                RegimeKind::Observational,
                EvidenceKind::Available,
                [],
                [],
                measured.iter().copied().map(v).collect::<Vec<_>>(),
                "target",
                DistributionAvailability::Joint,
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let bindings = regimes
        .iter()
        .map(|r| RegimeBinding {
            dataset_identity: None,
            regime: r.id,
            snapshot_identity: Arc::from(format!("snapshot-{}", r.id.raw())),
            schema_names: Arc::from([]),
            sampling: SamplingDesign::Independent,
            weights: None,
            dependence: DependenceGroup::IndependentStudies,
        })
        .collect::<Vec<_>>();
    EvidenceCatalog::try_new([], regimes, bindings, None).unwrap()
}

fn query() -> MixedSourceQuery {
    MixedSourceQuery {
        outcomes: Arc::from([v(1)]),
        treatments: Arc::from([v(0)]),
        target: Arc::from("target"),
        sources: Arc::from([]),
    }
}

/// The search's oracle (the graph crate's m-separation on a mutilated copy)
/// and the checker's independent augmented-graph criterion agree on every
/// sampled ADMG over four nodes, mutilation and conditioning set.
#[test]
fn oracle_and_independent_checker_agree_on_mutilated_separation() {
    let n = 4usize;
    let pairs: Vec<(u32, u32)> = (0..4).flat_map(|a| (a + 1..4).map(move |b| (a, b))).collect();
    let singles = std::iter::once(0).chain((0..n).map(bit)).collect::<Vec<Mask>>();
    let mut checked = 0usize;
    for code in (0..3usize.pow(6)).step_by(11) {
        let (mut directed, mut bidirected) = (Vec::new(), Vec::new());
        let mut rest = code;
        for &(a, b) in &pairs {
            match rest % 3 {
                1 => directed.push((a, b)),
                2 => bidirected.push((a, b)),
                _ => {}
            }
            rest /= 3;
        }
        let graph = admg(4, &directed, &bidirected);
        let checker = Checker::new(&graph).unwrap();
        let variables = (0..4).map(v).collect::<Vec<_>>();
        let mut engine = Engine::new(&graph, variables, Vec::new(), &{
            let mut q = query();
            q.outcomes = Arc::from([v(1)]);
            q
        })
        .unwrap();
        let set = |mask: Mask| members(mask).collect::<Set>();
        for &over in &singles {
            for &under in &singles {
                for a in (0..n).map(bit) {
                    for b in (0..n).map(bit).filter(|b| *b != a) {
                        for cond in 0..16 as Mask {
                            if cond & (a | b) != 0 {
                                continue;
                            }
                            let search = engine.separated(over, under, a, b, cond);
                            let check = checker.separated(
                                &set(over),
                                &set(under),
                                &set(a),
                                &set(b),
                                &set(cond),
                            );
                            assert_eq!(
                                search, check,
                                "{directed:?} {bidirected:?} over {over:#b} under {under:#b} \
                                 {a:#b} vs {b:#b} | {cond:#b}"
                            );
                            checked += 1;
                        }
                    }
                }
            }
        }
    }
    assert!(checked > 50_000, "{checked}");
}

/// The checker decides side conditions itself: a hand-built proof that
/// exchanges an observation for an action across an unblocked confounding
/// path is rejected even though its expressions are well formed.
#[test]
fn the_checker_rejects_an_unsound_exchange_the_search_never_proposes() {
    let bow = admg(2, &[(0, 1)], &[(0, 1)]);
    let c = catalog(&[(1, &[0, 1])]);
    let q = query();
    let leaf = MixedSourceLeaf {
        regime: RegimeId::from_raw(1),
        study: Arc::from("target"),
        population: Arc::from("target"),
        snapshot: Some(Arc::from("snapshot-1")),
        identity: c.distribution(RegimeId::from_raw(1)).unwrap().canonical_identity(),
    };
    let step =
        |rule, premises: Vec<usize>, params: Vec<u32>, y: Vec<u32>, d: Vec<u32>, z: Vec<u32>| {
            MixedStep {
                rule,
                premises,
                params: params.into_iter().map(v).collect(),
                y: y.into_iter().map(v).collect(),
                intervened: d.into_iter().map(v).collect(),
                conditioned: z.into_iter().map(v).collect(),
                source: (rule == MixedRule::Input).then(|| leaf.clone()),
                relaxed: false,
                expression: ExprId::from_raw(0),
            }
        };
    let mut steps = vec![
        step(MixedRule::Input, vec![], vec![], vec![0, 1], vec![], vec![]),
        step(MixedRule::Condition, vec![0], vec![0], vec![1], vec![], vec![0]),
        step(MixedRule::Rule2ToDo, vec![1], vec![0], vec![1], vec![0], vec![]),
    ];
    let mut arena = CausalExprArena::new();
    for i in 0..steps.len() {
        let premises = steps[i].premises.iter().map(|p| steps[*p].expression).collect::<Vec<_>>();
        steps[i].expression = build_expression(&mut arena, &steps[i], &premises, "target").unwrap();
    }
    assert_eq!(
        check_steps(&bow, &q, &c, &steps, Some(&arena), false).unwrap_err(),
        invalid_derivation()
    );
    // Without the confounder the same steps are a valid derivation.
    let clean = admg(2, &[(0, 1)], &[]);
    check_steps(&clean, &q, &c, &steps, Some(&arena), false).unwrap();
    // ... and the search finds exactly that on the clean graph, and nothing on the bow.
    let ctx = ExecutionContext::for_tests(1);
    let found = |graph: &Admg| {
        matches!(
            decide_mixed_source(graph, &q, &c, MIXED_SOURCE_DEFAULT_LIMITS, &ctx).unwrap(),
            MixedSourceDecision::NamedRoute { .. } | MixedSourceDecision::Identified { .. }
        )
    };
    assert!(found(&clean) && !found(&bow));
}

/// A binary structural model consistent with an ADMG: every node is a random
/// function of its parents and the latent bits of its bidirected edges, flipped
/// by independent noise so every configuration has positive mass.
type Mechanism = Box<dyn Fn(&[u8], &[u8]) -> u8>;

struct Model {
    n: usize,
    exo_p: Vec<f64>,
    f: Vec<Mechanism>,
}

impl Model {
    fn random(
        state: &mut u64,
        n: usize,
        directed: &[(usize, usize)],
        bidirected: &[(usize, usize)],
    ) -> Self {
        let mut next = || {
            *state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (*state >> 33) as u32
        };
        let mut exo_p =
            (0..n).map(|_| 0.15 + 0.2 * f64::from(next() % 1000) / 1000.0).collect::<Vec<_>>();
        exo_p.extend(bidirected.iter().map(|_| 0.3 + 0.4 * f64::from(next() % 1000) / 1000.0));
        let f = (0..n)
            .map(|i| {
                let parents = directed
                    .iter()
                    .filter(|(_, to)| *to == i)
                    .map(|(from, _)| *from)
                    .collect::<Vec<_>>();
                let latents = bidirected
                    .iter()
                    .enumerate()
                    .filter(|(_, (a, b))| *a == i || *b == i)
                    .map(|(k, _)| n + k)
                    .collect::<Vec<_>>();
                let width = parents.len() + latents.len();
                let table =
                    (0..1usize << width).map(|_| u8::from(next() % 2 == 1)).collect::<Vec<_>>();
                Box::new(move |values: &[u8], exo: &[u8]| {
                    let key = parents
                        .iter()
                        .map(|p| values[*p])
                        .chain(latents.iter().map(|l| exo[*l]))
                        .fold(0usize, |acc, bit| (acc << 1) | usize::from(bit));
                    table[key] ^ exo[i]
                }) as Mechanism
            })
            .collect();
        Self { n, exo_p, f }
    }

    /// Exact joint law over `measured` (first variable most significant) under `do_`.
    fn law(&self, do_: &[(usize, u8)], measured: &[usize]) -> Vec<f64> {
        let mut out = vec![0.0; 1 << measured.len()];
        let m = self.exo_p.len();
        for mask in 0..(1usize << m) {
            let exo = (0..m).map(|bit| u8::from((mask >> bit) & 1 == 1)).collect::<Vec<_>>();
            let mut weight = 1.0;
            for (bit, p) in self.exo_p.iter().enumerate() {
                weight *= if exo[bit] == 1 { *p } else { 1.0 - *p };
            }
            let mut values = vec![0u8; self.n];
            for i in 0..self.n {
                values[i] = match do_.iter().find(|(v, _)| *v == i) {
                    Some((_, level)) => *level,
                    None => (self.f[i])(&values, &exo),
                };
            }
            let index = measured.iter().fold(0usize, |acc, v| (acc << 1) | usize::from(values[*v]));
            out[index] += weight;
        }
        out
    }
}

/// Every quantity the frozen rule set derives, deletions included, computes
/// exactly the enumerated model's `P(y | z, do(d))`: each rule is applied to
/// every quantity of the closure over an observational joint and a trial, and
/// the resulting expression is evaluated by the exact provider and compared
/// with the truth. This covers rules the identified-query sweeps rarely reach.
#[test]
#[allow(clippy::too_many_lines, reason = "One fixture, one closure, one comparison per quantity")]
fn every_derived_quantity_matches_the_enumerated_model() {
    use antecedent_expr::{
        Assignment, DiscreteAxis, ExactDiscreteLaw, ExactEvaluationLimits, ExactEvaluationPlan,
        ExactEvaluationSpec, ExactTransportData, InterventionAssignment as ExprAssignment,
        LawTolerance,
    };
    let ctx = ExecutionContext::for_tests(1);
    let mut state = 0x5EED_u64;
    let (mut checked, mut deletions) = (0usize, 0usize);
    for case in 0..24 {
        let n = 4;
        let (mut directed, mut bidirected) = (Vec::new(), Vec::new());
        for a in 0..n {
            for b in a + 1..n {
                state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                match (state >> 40) % 5 {
                    0 | 1 => directed.push((a, b)),
                    2 => bidirected.push((a, b)),
                    _ => {}
                }
            }
        }
        let model = Model::random(&mut state, n, &directed, &bidirected);
        let graph = admg(
            4,
            &directed
                .iter()
                .map(|(a, b)| (u32::try_from(*a).unwrap(), u32::try_from(*b).unwrap()))
                .collect::<Vec<_>>(),
            &bidirected
                .iter()
                .map(|(a, b)| (u32::try_from(*a).unwrap(), u32::try_from(*b).unwrap()))
                .collect::<Vec<_>>(),
        );
        // Regime 1: observational joint of every variable; regime 2: a trial of one
        // variable measuring the rest.
        let trial_on = case % n;
        let rest = (0..n).filter(|v| *v != trial_on).collect::<Vec<_>>();
        let studies = if case % 2 == 0 {
            vec![(1u32, Vec::<usize>::new(), (0..n).collect::<Vec<_>>()), (2, vec![trial_on], rest)]
        } else {
            let partial = rest.iter().copied().take(2).collect::<Vec<_>>();
            vec![
                (1u32, Vec::<usize>::new(), vec![0, 1, 2]),
                (2, Vec::new(), vec![1, 2, 3]),
                (3, vec![trial_on], partial),
            ]
        };
        let mut regimes = Vec::new();
        let mut laws = Vec::new();
        for (id, on, measured) in &studies {
            let measured: &Vec<usize> = measured;
            regimes.push(
                EvidenceRegime::try_new(
                    RegimeId::from_raw(*id),
                    if on.is_empty() {
                        RegimeKind::Observational
                    } else {
                        RegimeKind::Experimental
                    },
                    EvidenceKind::Available,
                    on.iter().map(|v| v_of(*v)).collect::<Vec<_>>(),
                    [],
                    measured.iter().map(|v| v_of(*v)).collect::<Vec<_>>(),
                    "target",
                    DistributionAvailability::Joint,
                )
                .unwrap(),
            );
            let axes = measured
                .iter()
                .map(|i| DiscreteAxis {
                    variable: v_of(*i),
                    values: Arc::from([
                        antecedent_core::Value::Bool(false),
                        antecedent_core::Value::Bool(true),
                    ]),
                })
                .collect::<Vec<_>>();
            for levels in 0..(1usize << on.len()) {
                let assignments = on
                    .iter()
                    .enumerate()
                    .map(|(b, v)| (*v, u8::from((levels >> b) & 1 == 1)))
                    .collect::<Vec<_>>();
                laws.push(
                    ExactDiscreteLaw::try_new(
                        "target",
                        RegimeId::from_raw(*id),
                        assignments
                            .iter()
                            .map(|(v, l)| {
                                ExprAssignment::concrete(
                                    v_of(*v),
                                    antecedent_core::Value::Bool(*l == 1),
                                )
                            })
                            .collect::<Vec<_>>(),
                        axes.clone(),
                        model.law(&assignments, measured),
                        format!("snapshot-{id}"),
                        LawTolerance::default(),
                    )
                    .unwrap(),
                );
            }
        }
        // Every study is cited: each leaf selects its law by world among them.
        let cited = regimes.iter().map(|regime| regime.id).collect::<Vec<_>>();
        let data = ExactTransportData::try_new(laws, 4096).unwrap().with_world_bound_leaves(cited);
        let catalog = EvidenceCatalog::try_new(
            [],
            regimes.clone(),
            regimes
                .iter()
                .map(|r| RegimeBinding {
                    dataset_identity: None,
                    regime: r.id,
                    snapshot_identity: Arc::from(format!("snapshot-{}", r.id.raw())),
                    schema_names: Arc::from([]),
                    sampling: SamplingDesign::Independent,
                    weights: None,
                    dependence: DependenceGroup::IndependentStudies,
                })
                .collect::<Vec<_>>(),
            None,
        )
        .unwrap();
        let q = MixedSourceQuery {
            outcomes: Arc::from([v_of(1)]),
            treatments: Arc::from([v_of(0)]),
            target: Arc::from("target"),
            sources: Arc::from([]),
        };
        let validated = validate_mixed_source_query(&graph, &q, &catalog).unwrap();
        let variables = (0..n).map(v_of).collect::<Vec<_>>();
        let mut engine =
            Engine::new(&graph, variables, validated.inputs.clone(), &validated.query).unwrap();
        // Close the rule set: no target, a generous budget.
        engine.target = (0, 0, 0);
        engine.found = None;
        let budget =
            SearchBudget::new(SearchLimits { operations: 4_000_000, depth: 64 }, &ctx).unwrap();
        let mut search = SharedSearch::new(budget);
        assert!(matches!(engine.run(&mut search), RunEnd::Fixpoint));
        let mut verify = |steps: &[MixedStep], arena: &CausalExprArena, label: &str| {
            let step = &steps[steps.len() - 1];
            let (y, d, z) = (step.y.clone(), step.intervened.clone(), step.conditioned.clone());
            let sorted = |vs: &[VariableId]| vs.iter().map(|v| v.as_usize()).collect::<Vec<_>>();
            let bound = z.iter().chain(d.iter()).copied().collect::<Vec<_>>();
            // Every assignment of the conditioning values and the intervention levels.
            for levels in 0..(1usize << bound.len()) {
                let assign = bound
                    .iter()
                    .enumerate()
                    .map(|(b, v)| (*v, u8::from((levels >> b) & 1 == 1)))
                    .collect::<Vec<_>>();
                let request = Assignment::from_pairs(
                    assign.iter().map(|(v, l)| (*v, antecedent_core::Value::Bool(*l == 1))),
                );
                let plan = ExactEvaluationPlan::compile(
                    ExactEvaluationSpec {
                        arena,
                        root: step.expression,
                        outcomes: y.clone().into(),
                        request,
                        limits: ExactEvaluationLimits::default(),
                        tolerance: LawTolerance::default(),
                    },
                    data.clone(),
                    &ctx,
                )
                .unwrap_or_else(|e| panic!("{directed:?} {bidirected:?} {label}: {e:?}"));
                let got = plan.evaluate(&ctx).unwrap_or_else(|e| {
                    panic!("{directed:?} {bidirected:?} {label} {y:?}|do{d:?},{z:?}: {e:?}")
                });
                let levels_of = |vs: &[VariableId]| {
                    vs.iter()
                        .map(|v| assign.iter().find(|(w, _)| w == v).map(|(_, l)| *l).unwrap())
                        .collect::<Vec<_>>()
                };
                let do_ = sorted(&d).into_iter().zip(levels_of(&d)).collect::<Vec<_>>();
                let measured = sorted(&y).into_iter().chain(sorted(&z)).collect::<Vec<_>>();
                let joint = model.law(&do_, &measured);
                let z_levels = levels_of(&z);
                for (atom, p) in got.atoms.iter().zip(got.probabilities.iter()) {
                    let y_levels = atom
                        .iter()
                        .map(|value| u8::from(*value == antecedent_core::Value::Bool(true)))
                        .collect::<Vec<_>>();
                    let key = |ys: &[u8]| {
                        ys.iter()
                            .chain(z_levels.iter())
                            .fold(0usize, |acc, bit| (acc << 1) | usize::from(*bit))
                    };
                    let denominator: f64 = (0..1usize << y.len())
                        .map(|combo| {
                            let ys = (0..y.len())
                                .map(|b| u8::from((combo >> (y.len() - 1 - b)) & 1 == 1))
                                .collect::<Vec<_>>();
                            joint[key(&ys)]
                        })
                        .sum();
                    let truth = joint[key(&y_levels)] / denominator;
                    assert!(
                        (p - truth).abs() < 1e-9,
                        "{directed:?} {bidirected:?} {label}: P({y:?} | do{d:?}, {z:?}) = {p}, truth {truth}"
                    );
                    checked += 1;
                }
            }
        };
        for index in 0..engine.steps.len() {
            let last = engine.steps[index].clone();
            let (steps, arena) = engine.extract(&last, &catalog, "target").unwrap();
            verify(&steps, &arena, &format!("step {index} {:?}", last.rule));
        }
        // Deletions never derive a new quantity here, so their candidates are checked
        // directly: whenever the independent checker accepts one, its number is the truth.
        for index in 0..engine.steps.len() {
            let base = engine.steps[index].clone();
            for (rule, pool) in [(MixedRule::Rule1Delete, base.z), (MixedRule::Rule3Delete, base.d)]
            {
                for w in subsets(pool, MIXED_SOURCE_MAX_MOVE) {
                    let mut candidate = base.clone();
                    candidate.rule = rule;
                    candidate.premises = vec![index];
                    candidate.params = w;
                    candidate.input = None;
                    candidate.relaxed = false;
                    if rule == MixedRule::Rule1Delete {
                        candidate.z = base.z & !w;
                    } else {
                        candidate.d = base.d & !w;
                    }
                    let (steps, arena) = engine.extract(&candidate, &catalog, "target").unwrap();
                    if check_steps(&graph, &q, &catalog, &steps, Some(&arena), false).is_ok() {
                        deletions += 1;
                        verify(&steps, &arena, &format!("{rule:?} of step {index}"));
                    }
                }
            }
        }
    }
    assert!(checked > 10_000, "{checked}");
    assert!(deletions > 20, "{deletions}");
    eprintln!("checked {checked} cells, {deletions} valid deletion applications");
}

type LawCache = HashMap<(Vec<(usize, u8)>, Vec<usize>), Vec<f64>>;
type Edges = Vec<(usize, usize)>;

fn u32_edges(edges: &[(usize, usize)]) -> Vec<(u32, u32)> {
    edges.iter().map(|(a, b)| (u32::try_from(*a).unwrap(), u32::try_from(*b).unwrap())).collect()
}

/// `P(y | do(d), z)` of the enumerated model at the given levels, over the
/// assignments of `y` (first variable most significant).
fn truth(
    model: &Model,
    cache: &mut LawCache,
    do_: &[(usize, u8)],
    y: &[usize],
    z: &[(usize, u8)],
) -> Vec<f64> {
    let measured = y.iter().copied().chain(z.iter().map(|(v, _)| *v)).collect::<Vec<_>>();
    let joint = cache
        .entry((do_.to_vec(), measured.clone()))
        .or_insert_with(|| model.law(do_, &measured))
        .clone();
    let key = z.iter().fold(0usize, |acc, (_, level)| (acc << 1) | usize::from(*level));
    let cells = (0..1usize << y.len()).map(|ys| joint[(ys << z.len()) | key]).collect::<Vec<_>>();
    let total = cells.iter().sum::<f64>();
    cells.iter().map(|cell| cell / total).collect()
}

/// The unary do-calculus rules the search attempts (deletions are checked but
/// not searched), with the quantity each moves a premise to.
const SEARCHED_UNARY: [MixedRule; 4] = [
    MixedRule::Rule1Insert,
    MixedRule::Rule2ToDo,
    MixedRule::Rule2ToObservation,
    MixedRule::Rule3Insert,
];

fn conclusion_of(rule: MixedRule, (y, d, z): (Mask, Mask, Mask), w: Mask) -> (Mask, Mask, Mask) {
    match rule {
        MixedRule::Rule1Insert => (y, d, z | w),
        MixedRule::Rule1Delete => (y, d, z & !w),
        MixedRule::Rule2ToDo => (y, d | w, z & !w),
        MixedRule::Rule2ToObservation => (y, d & !w, z | w),
        MixedRule::Rule3Insert => (y, d | w, z),
        MixedRule::Rule3Delete => (y, d & !w, z),
        _ => unreachable!("only the one-premise do-calculus rules"),
    }
}

/// Every do-calculus rule instance the independent checker accepts asserts an
/// equality; the enumerated structural model must satisfy it. Both the search's
/// oracle and the checker share the side-condition definitions, so a rule that
/// is unsound in both (say rule 3 without Pearl's `W(Z)` exception) is caught
/// here and nowhere else: this is the semantic test of each rule's side
/// condition. For every random ADMG, random model and every premise quantity
/// `P(y | do(d), z)` the test also requires the search's accepted applications
/// to equal the checker's for the searched rules.
#[test]
#[allow(clippy::too_many_lines, reason = "One enumeration of graphs, premises, rules and levels")]
fn every_accepted_rule_instance_holds_in_the_enumerated_model() {
    let ctx = ExecutionContext::for_tests(1);
    let mut state = 0x0BAD_5EED_u64;
    let mut next = move || {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        (state >> 40) as usize
    };
    let mut graphs: Vec<(usize, Edges, Edges)> = Vec::new();
    // Every three-node ADMG (each pair: none, directed, bidirected, both).
    for code in 0..4usize.pow(3) {
        let (mut directed, mut bidirected) = (Vec::new(), Vec::new());
        for (k, (a, b)) in [(0, 1), (0, 2), (1, 2)].into_iter().enumerate() {
            let kind = (code >> (2 * k)) & 3;
            if kind & 1 == 1 {
                directed.push((a, b));
            }
            if kind & 2 == 2 {
                bidirected.push((a, b));
            }
        }
        graphs.push((3, directed, bidirected));
    }
    // Random four- and five-node ADMGs.
    for size in [4usize, 4, 5] {
        for _ in 0..if size == 4 { 30 } else { 8 } {
            let (mut directed, mut bidirected) = (Vec::new(), Vec::new());
            for a in 0..size {
                for b in a + 1..size {
                    match next() % 6 {
                        0 | 1 => directed.push((a, b)),
                        2 => bidirected.push((a, b)),
                        3 => {
                            directed.push((a, b));
                            bidirected.push((a, b));
                        }
                        _ => {}
                    }
                }
            }
            bidirected.truncate(if size == 4 { 4 } else { 3 });
            graphs.push((size, directed, bidirected));
        }
    }
    let mut accepted = BTreeMap::<&str, usize>::new();
    let mut multi_w_with_exception = 0usize;
    let mut exception_only_insert = 0usize;
    let mut ancestral_and_non_ancestral = 0usize;
    let mut compared = 0usize;
    for (index, (n, directed, bidirected)) in graphs.iter().enumerate() {
        let (n, cap) = (*n, Mask::try_from(1usize << *n).unwrap());
        let graph = admg(u32::try_from(n).unwrap(), &u32_edges(directed), &u32_edges(bidirected));
        let mut model_state = 0x1000 + index as u64;
        let model = Model::random(&mut model_state, n, directed, bidirected);
        let mut cache = HashMap::new();
        let checker = Checker::new(&graph).unwrap();
        let universe = (0..n).collect::<Set>();
        let variables = (0..n).map(v_of).collect::<Vec<_>>();
        let stride = if n == 5 { 3 } else { 1 };
        let set = |mask: Mask| members(mask).collect::<Set>();
        for code in (0..4usize.pow(u32::try_from(n).unwrap())).step_by(stride) {
            let role = |i: usize| (code >> (2 * i)) & 3;
            let mask_of = |r: usize| (0..n).filter(|i| role(*i) == r).fold(0, |m, i| m | bit(i));
            let (y, d, z) = (mask_of(0), mask_of(1), mask_of(2));
            if y == 0 || size(d) > MIXED_SOURCE_MAX_DO {
                continue;
            }
            let mut engine = Engine::new(&graph, variables.clone(), Vec::new(), &query()).unwrap();
            engine.add(Raw {
                rule: MixedRule::Input,
                premises: Vec::new(),
                params: 0,
                y,
                d,
                z,
                free: y | d,
                input: None,
                relaxed: false,
            });
            let budget =
                SearchBudget::new(SearchLimits { operations: 4_000_000, depth: 64 }, &ctx).unwrap();
            let mut search = SharedSearch::new(budget);
            engine.expand(&mut search, 0, 1).unwrap();
            let found = engine
                .steps
                .iter()
                .filter(|s| SEARCHED_UNARY.contains(&s.rule) && s.premises == [0])
                .map(|s| (s.rule.as_str(), s.params))
                .collect::<BTreeSet<_>>();
            let mut by_checker = BTreeSet::new();
            for w in 1..cap {
                if size(w) > MIXED_SOURCE_MAX_MOVE {
                    continue;
                }
                for rule in [
                    MixedRule::Rule1Insert,
                    MixedRule::Rule1Delete,
                    MixedRule::Rule2ToDo,
                    MixedRule::Rule2ToObservation,
                    MixedRule::Rule3Insert,
                    MixedRule::Rule3Delete,
                ] {
                    let (cy, cd, cz) = conclusion_of(rule, (y, d, z), w);
                    if size(cd) > MIXED_SOURCE_MAX_DO {
                        continue;
                    }
                    let holds = unary_rule_holds(
                        &checker,
                        &universe,
                        rule,
                        (&set(y), &set(d), &set(z)),
                        (&set(cy), &set(cd), &set(cz)),
                        &set(w),
                        &set(y),
                    );
                    if !holds {
                        continue;
                    }
                    if SEARCHED_UNARY.contains(&rule) {
                        by_checker.insert((rule.as_str(), w));
                    }
                    *accepted.entry(rule.as_str()).or_default() += 1;
                    if rule == MixedRule::Rule3Insert {
                        let past = checker.ancestors(&set(z), &set(d));
                        let inside = set(w).intersection(&past).count();
                        if inside > 0 && inside < size(w) {
                            ancestral_and_non_ancestral += 1;
                        }
                        if size(w) == 2 && inside > 0 {
                            multi_w_with_exception += 1;
                        }
                        if inside == 0 {
                            exception_only_insert += 1;
                        }
                    }
                    // The equality the rule asserts, at every level.
                    let bound = set(d | z | w | cd | cz).into_iter().collect::<Vec<_>>();
                    let ys = members(y).collect::<Vec<_>>();
                    for levels in 0..1usize << bound.len() {
                        let level = |v: usize| {
                            let position = bound.iter().position(|b| *b == v).unwrap();
                            u8::from((levels >> position) & 1 == 1)
                        };
                        let at =
                            |mask: Mask| members(mask).map(|v| (v, level(v))).collect::<Vec<_>>();
                        let before = truth(&model, &mut cache, &at(d), &ys, &at(z));
                        let after = truth(&model, &mut cache, &at(cd), &ys, &at(cz));
                        for (a, b) in before.iter().zip(&after) {
                            assert!(
                                (a - b).abs() < 1e-9,
                                "{rule:?} through {w:#b} is unsound: {directed:?} \
                                 {bidirected:?} P({ys:?} | do({d:#b}), {z:#b}) = {a}, but \
                                 the conclusion (do {cd:#b}, given {cz:#b}) is {b}"
                            );
                        }
                    }
                }
            }
            compared += 1;
            assert_eq!(
                found, by_checker,
                "search and checker disagree on {directed:?} {bidirected:?} \
                 y {y:#b} do {d:#b} z {z:#b}"
            );
        }
    }
    eprintln!(
        "{compared} premises, accepted {accepted:?}, rule 3 insertions: \
         {multi_w_with_exception} multi-W with an ancestral member, \
         {exception_only_insert} with no ancestral member, \
         {ancestral_and_non_ancestral} mixing both"
    );
    for rule in ["rule1_insert", "rule1_delete", "rule2_to_do", "rule2_to_observation"] {
        assert!(accepted.get(rule).copied().unwrap_or(0) > 100, "{rule}: {accepted:?}");
    }
    for rule in ["rule3_insert", "rule3_delete"] {
        assert!(accepted.get(rule).copied().unwrap_or(0) > 100, "{rule}: {accepted:?}");
    }
    assert!(multi_w_with_exception > 20, "{multi_w_with_exception}");
    assert!(exception_only_insert > 20, "{exception_only_insert}");
    assert!(ancestral_and_non_ancestral > 20, "{ancestral_and_non_ancestral}");
}

/// The counterexample rule 3 needs its `W(Z)` exception for: `W -> Z` and
/// `W <-> Y`. Deleting `do(W)` from `P(y | do(w), z)` (or inserting it into
/// `P(y | z)`) is invalid because `W` is an ancestor of the observed `Z`, so
/// conditioning on `Z` re-opens the confounded path; the enumerated model
/// proves the two quantities differ. The same rules are valid once `W` is not
/// an ancestor of `Z`.
#[test]
fn rule_three_refuses_the_ancestral_counterexample() {
    let (w, zed, y) = (0usize, 1usize, 2usize);
    let one = |i: usize| Set::from([i]);
    let both = |a: usize, b: usize| Set::from([a, b]);
    let none = Set::new();
    let counterexample = admg(3, &[(0, 1)], &[(0, 2)]);
    let checker = Checker::new(&counterexample).unwrap();
    let universe = Set::from([0, 1, 2]);
    // P(y | do(w), z) -> P(y | z): refused.
    assert!(!unary_rule_holds(
        &checker,
        &universe,
        MixedRule::Rule3Delete,
        (&one(y), &one(w), &one(zed)),
        (&one(y), &none, &one(zed)),
        &one(w),
        &none,
    ));
    // P(y | z) -> P(y | do(w), z): refused.
    assert!(!unary_rule_holds(
        &checker,
        &universe,
        MixedRule::Rule3Insert,
        (&one(y), &none, &one(zed)),
        (&one(y), &one(w), &one(zed)),
        &one(w),
        &none,
    ));
    // ... and the model agrees that they differ.
    let differs = (0..20u64).any(|seed| {
        let mut state = 0xC0DE + seed;
        let model = Model::random(&mut state, 3, &[(0, 1)], &[(0, 2)]);
        let mut cache = HashMap::new();
        [0u8, 1].into_iter().any(|z_level| {
            [0u8, 1].into_iter().any(|w_level| {
                let with = truth(&model, &mut cache, &[(w, w_level)], &[y], &[(zed, z_level)]);
                let without = truth(&model, &mut cache, &[], &[y], &[(zed, z_level)]);
                (with[1] - without[1]).abs() > 1e-3
            })
        })
    });
    assert!(differs, "the counterexample must separate the two quantities");
    // Without the confounding edge both rules hold.
    let plain = Checker::new(&admg(3, &[(0, 1)], &[])).unwrap();
    assert!(unary_rule_holds(
        &plain,
        &universe,
        MixedRule::Rule3Delete,
        (&one(y), &one(w), &one(zed)),
        (&one(y), &none, &one(zed)),
        &one(w),
        &none,
    ));
    // With `Z -> W` instead (`W` is not an ancestor of `Z`) the exception cuts the
    // confounding edge and both rules hold even though `W <-> Y` remains.
    let reversed = Checker::new(&admg(3, &[(1, 0)], &[(0, 2)])).unwrap();
    for (rule, premise, conclusion) in [
        (MixedRule::Rule3Delete, (one(w), one(zed)), (none.clone(), one(zed))),
        (MixedRule::Rule3Insert, (none.clone(), one(zed)), (one(w), one(zed))),
    ] {
        assert!(unary_rule_holds(
            &reversed,
            &universe,
            rule,
            (&one(y), &premise.0, &premise.1),
            (&one(y), &conclusion.0, &conclusion.1),
            &one(w),
            &none,
        ));
    }
    // Two inserted actions, one ancestral (`W1 -> Z`) and one not (`Z -> W2`): only
    // the ancestral one keeps its confounding edge, so inserting both is refused
    // while inserting the non-ancestral one alone is allowed.
    let mixed = admg(4, &[(0, 1), (1, 2)], &[(0, 3), (2, 3)]);
    let checker = Checker::new(&mixed).unwrap();
    let universe = Set::from([0, 1, 2, 3]);
    let holds = |inserted: Set| {
        unary_rule_holds(
            &checker,
            &universe,
            MixedRule::Rule3Insert,
            (&one(3), &none, &one(1)),
            (&one(3), &inserted, &one(1)),
            &inserted,
            &none,
        )
    };
    assert!(!holds(one(0)), "an ancestral action keeps its edges");
    assert!(!holds(both(0, 2)));
    assert!(holds(one(2)), "a non-ancestral action is cut from its edges");
}

/// The deletion rules are accepted by the checker but never attempted by the
/// search, because they contribute nothing in this rule set: over random
/// ADMGs and catalogs (single and double trials), every deletion the checker
/// accepts on a quantity of the closure (its expression still free of the
/// deleted variable) lands on a quantity the other rules already derived, and
/// the closure never holds a quantity first derived by a deletion.
#[test]
fn deletions_are_checked_but_add_nothing_to_the_closure() {
    let ctx = ExecutionContext::for_tests(1);
    let mut state = 0xFACE_u64;
    let mut next = move || {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        (state >> 40) as usize
    };
    let (mut closures, mut accepted, mut quantities) = (0usize, 0usize, 0usize);
    for _ in 0..250 {
        let n = 4 + next() % 2;
        let (mut directed, mut bidirected) = (Vec::new(), Vec::new());
        for a in 0..n {
            for b in a + 1..n {
                match next() % 6 {
                    0 | 1 => directed.push((a, b)),
                    2 => bidirected.push((a, b)),
                    3 => {
                        directed.push((a, b));
                        bidirected.push((a, b));
                    }
                    _ => {}
                }
            }
        }
        let graph = admg(u32::try_from(n).unwrap(), &u32_edges(&directed), &u32_edges(&bidirected));
        let mut inputs = Vec::new();
        for k in 0..2 + next() % 3 {
            let mut measured = (0..n).collect::<Vec<_>>();
            while measured.len() > 1 + next() % 3 {
                measured.remove(next() % measured.len());
            }
            let mut on = Vec::new();
            let outside = (0..n).filter(|v| !measured.contains(v)).collect::<Vec<_>>();
            if next() % 2 == 0 && !outside.is_empty() {
                on.push(outside[next() % outside.len()]);
                let more = outside.iter().copied().filter(|v| !on.contains(v)).collect::<Vec<_>>();
                if next() % 3 == 0 && !more.is_empty() {
                    on.push(more[next() % more.len()]);
                }
            }
            inputs.push(MixedInput {
                regime: RegimeId::from_raw(u32::try_from(k).unwrap() + 1),
                study: Arc::from(format!("s{k}")),
                measured: measured.iter().map(|i| v_of(*i)).collect(),
                intervened: on.iter().map(|i| v_of(*i)).collect(),
                identity: format!("i{k}"),
                snapshot: None,
                marginals: Vec::new(),
            });
        }
        let variables = (0..n).map(v_of).collect::<Vec<_>>();
        let mut engine = Engine::new(&graph, variables, inputs, &query()).unwrap();
        engine.target = (0, 0, 0);
        let budget =
            SearchBudget::new(SearchLimits { operations: 40_000_000, depth: 64 }, &ctx).unwrap();
        let mut search = SharedSearch::new(budget);
        if !matches!(engine.run(&mut search), RunEnd::Fixpoint) {
            continue;
        }
        closures += 1;
        quantities += engine.steps.len();
        assert!(
            !engine.rule_counts.contains_key("rule1_delete")
                && !engine.rule_counts.contains_key("rule3_delete")
        );
        let checker = Checker::new(&graph).unwrap();
        let universe = (0..n).collect::<Set>();
        let set = |mask: Mask| members(mask).collect::<Set>();
        for step in &engine.steps {
            for (rule, pool) in [(MixedRule::Rule1Delete, step.z), (MixedRule::Rule3Delete, step.d)]
            {
                for w in subsets(pool, MIXED_SOURCE_MAX_MOVE) {
                    let (cy, cd, cz) = conclusion_of(rule, (step.y, step.d, step.z), w);
                    if unary_rule_holds(
                        &checker,
                        &universe,
                        rule,
                        (&set(step.y), &set(step.d), &set(step.z)),
                        (&set(cy), &set(cd), &set(cz)),
                        &set(w),
                        &set(step.free),
                    ) {
                        accepted += 1;
                        assert!(
                            engine.known.contains_key(&(cy, cd, cz)),
                            "{directed:?} {bidirected:?}: a deletion derives a new quantity"
                        );
                    }
                }
            }
        }
    }
    eprintln!("{closures} closures, {quantities} quantities, {accepted} accepted deletions");
    assert!(closures > 200 && accepted > 20, "{closures} {accepted}");
}

/// Each field of a checked derivation is load-bearing.
#[test]
fn every_field_of_a_checked_derivation_is_load_bearing() {
    let graph = admg(3, &[(0, 1), (1, 2)], &[]);
    let c = catalog(&[(1, &[0, 1]), (2, &[1, 2])]);
    let q = MixedSourceQuery { outcomes: Arc::from([v(2)]), ..query() };
    let ctx = ExecutionContext::for_tests(1);
    let MixedSourceDecision::Identified { derivation, .. } =
        decide_mixed_source(&graph, &q, &c, MIXED_SOURCE_DEFAULT_LIMITS, &ctx).unwrap()
    else {
        panic!("identified");
    };
    verify_mixed_source_derivation(&graph, &derivation, &c).unwrap();
    let bad = invalid_derivation();
    let mutate = |edit: &dyn Fn(&mut MixedSourceDerivation)| {
        let mut copy = (*derivation).clone();
        edit(&mut copy);
        verify_mixed_source_derivation(&graph, &copy, &c).unwrap_err()
    };
    let last = derivation.steps().len() - 1;
    assert_eq!(mutate(&|d| d.steps[last].premises = vec![0]), bad);
    assert_eq!(mutate(&|d| d.steps[last].params.clear()), bad);
    assert_eq!(mutate(&|d| d.steps[0].source.as_mut().unwrap().identity.push('x')), bad);
    assert_eq!(
        mutate(&|d| d.steps[0].source.as_mut().unwrap().regime = RegimeId::from_raw(2)),
        bad
    );
    assert_eq!(mutate(&|d| d.steps[0].relaxed = true), bad);
    assert_eq!(mutate(&|d| d.steps[last].intervened.clear()), bad);
    assert_eq!(mutate(&|d| d.steps[last].expression = ExprId::from_raw(0)), bad);
    assert_eq!(mutate(&|d| d.arena = CausalExprArena::new()), bad);
    assert_eq!(mutate(&|d| d.steps.truncate(last)), bad);
    assert_eq!(mutate(&|d| d.query.target = Arc::from("other")), bad);
    assert_eq!(mutate(&|d| d.graph_signature.push('x')), bad);
    // A rule the step's quantity does not follow from.
    assert_eq!(mutate(&|d| d.steps[last].rule = MixedRule::Rule3Delete), bad);
}

#[test]
fn the_shared_expression_layer_compiles_the_root_and_binds_its_leaves() {
    let graph = admg(3, &[(0, 1), (1, 2)], &[]);
    let c = catalog(&[(1, &[0, 1]), (2, &[1, 2])]);
    let q = MixedSourceQuery { outcomes: Arc::from([v(2)]), ..query() };
    let ctx = ExecutionContext::for_tests(1);
    let MixedSourceDecision::Identified { derivation, .. } =
        decide_mixed_source(&graph, &q, &c, MIXED_SOURCE_DEFAULT_LIMITS, &ctx).unwrap()
    else {
        panic!("identified");
    };
    let expressions = derivation.steps().iter().map(|step| step.expression).collect::<Vec<_>>();
    check_with_shared_expression_layer(&derivation.arena, &derivation.steps, &expressions).unwrap();
    // A proof that cites a leaf the root expression does not read, or drops
    // one it reads, no longer binds.
    let mut extra = derivation.steps.clone();
    let input = extra.iter().position(|s| s.source.is_some()).unwrap();
    extra[input].source.as_mut().unwrap().regime = RegimeId::from_raw(2);
    assert_eq!(
        check_with_shared_expression_layer(&derivation.arena, &extra, &expressions).unwrap_err(),
        invalid_derivation()
    );
    // A root outside the arena is refused by the shared compiler.
    let mut outside = expressions.clone();
    *outside.last_mut().unwrap() = ExprId::from_raw(u32::MAX);
    assert_eq!(
        check_with_shared_expression_layer(&derivation.arena, &derivation.steps, &outside)
            .unwrap_err(),
        invalid_derivation()
    );
}

/// Completeness against the complete ID algorithm (single observational study).
mod reference {
    use super::*;
    use crate::{IdIdentifier, IdentificationWorkspace, result::IdentificationStatus};
    use antecedent_core::{
        CausalQuery, Intervention, InterventionalDistributionQuery, SearchBudget, Value,
    };
    use antecedent_expr::ExprNode;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(super) enum Verdict {
        Identified,
        NotCertified,
        Exhausted,
    }

    /// Whether ID's formula contains a ratio (a conditional of a marginalised
    /// C-factor: the napkin family).
    fn has_ratio(result: &crate::IdentificationResult) -> bool {
        fn walk(arena: &antecedent_expr::CausalExprArena, id: ExprId) -> bool {
            match arena.node(id) {
                ExprNode::Ratio { .. } => true,
                ExprNode::Product(list) => arena.list(*list).iter().any(|&e| walk(arena, e)),
                ExprNode::SumOut { expr, .. } => walk(arena, *expr),
                _ => false,
            }
        }
        result.estimands.first().is_some_and(|e| walk(&result.arena, e.functional))
    }

    /// Shpitser-Pearl ID on `P(outcomes | do(treatments))`: `None` when not
    /// identifiable, else whether the formula needs a ratio.
    pub(super) fn id_identifiable(
        graph: &Admg,
        outcomes: &[u32],
        treatments: &[u32],
    ) -> Option<bool> {
        let id = IdIdentifier::new();
        let prepared = id.prepare(graph).unwrap();
        let query = CausalQuery::Distribution(
            InterventionalDistributionQuery::new(
                v(outcomes[0]),
                treatments
                    .iter()
                    .map(|t| Intervention::set(v(*t), Value::f64(1.0)))
                    .collect::<Vec<_>>(),
            )
            .with_outcomes(outcomes.iter().map(|o| v(*o)).collect::<Vec<_>>()),
        );
        let result =
            id.identify(&prepared, &query, &mut IdentificationWorkspace::default()).unwrap();
        match result.status {
            IdentificationStatus::NonparametricallyIdentified => Some(has_ratio(&result)),
            IdentificationStatus::NotIdentified => None,
            other => panic!("unexpected ID status {other:?}"),
        }
    }

    /// The generic rule search alone over one observational study of every
    /// variable (the named routes are not run).
    pub(super) fn search_only(graph: &Admg, outcomes: &[u32], treatments: &[u32]) -> Verdict {
        let n = u32::try_from(graph.node_count()).unwrap();
        let all = (0..n).collect::<Vec<_>>();
        let evidence = catalog(&[(1, &all)]);
        let q = MixedSourceQuery {
            outcomes: outcomes.iter().map(|o| v(*o)).collect(),
            treatments: treatments.iter().map(|t| v(*t)).collect(),
            target: Arc::from("target"),
            sources: Arc::from([]),
        };
        let validated = validate_mixed_source_query(graph, &q, &evidence).unwrap();
        let ctx = ExecutionContext::for_tests(1);
        let budget = SearchBudget::new(MIXED_SOURCE_DEFAULT_LIMITS, &ctx).unwrap();
        match rule_search(graph, &validated, &evidence, &mut SharedSearch::new(budget), Vec::new())
            .unwrap()
        {
            MixedSourceDecision::Identified { .. } => Verdict::Identified,
            MixedSourceDecision::NotCertified(_) => Verdict::NotCertified,
            MixedSourceDecision::Exhausted(_) => Verdict::Exhausted,
            other => panic!("unexpected {other:?}"),
        }
    }

    type Edges = Vec<(u32, u32)>;

    /// Outcome counts of a sweep, by query shape `(|Y|, |X|)`.
    #[derive(Default, Debug)]
    pub(super) struct Tally {
        pub(super) queries: usize,
        /// ID-identifiable queries whose ID formula is a ratio, and how many the search identified.
        pub(super) ratio: [usize; 2],
        /// Per shape: (ID-identifiable, identified, not certified, exhausted, non-identifiable).
        pub(super) shapes: BTreeMap<(usize, usize), [usize; 5]>,
        /// Search identified a query ID refutes (must stay empty).
        pub(super) unsound: Vec<String>,
        /// ID-identifiable queries the search did not identify: (edge count, |X|, description).
        pub(super) gaps: Vec<(usize, usize, String)>,
    }

    impl Tally {
        pub(super) fn totals(&self) -> [usize; 5] {
            let mut t = [0; 5];
            for c in self.shapes.values() {
                for (a, b) in t.iter_mut().zip(c) {
                    *a += b;
                }
            }
            t
        }
    }

    /// Run both procedures on one query and record the outcome.
    pub(super) fn record(
        tally: &mut Tally,
        graph: &Admg,
        d: &[(u32, u32)],
        b: &[(u32, u32)],
        ys: &[u32],
        xs: &[u32],
    ) {
        let ratio = id_identifiable(graph, ys, xs);
        let verdict = search_only(graph, ys, xs);
        let cell = tally.shapes.entry((ys.len(), xs.len())).or_default();
        tally.queries += 1;
        let text = format!("directed={d:?} bidirected={b:?} y={ys:?} do={xs:?}");
        let Some(ratio) = ratio else {
            cell[4] += 1;
            if verdict == Verdict::Identified {
                tally.unsound.push(text);
            }
            return;
        };
        cell[0] += 1;
        if ratio {
            tally.ratio[0] += 1;
            tally.ratio[1] += usize::from(verdict == Verdict::Identified);
        }
        match verdict {
            Verdict::Identified => cell[1] += 1,
            Verdict::NotCertified => cell[2] += 1,
            Verdict::Exhausted => cell[3] += 1,
        }
        if verdict != Verdict::Identified {
            tally.gaps.push((d.len() + b.len(), xs.len(), format!("{verdict:?} {text}")));
        }
    }

    /// Every non-empty disjoint (outcomes, treatments) pair over `n` nodes.
    pub(super) fn all_queries(n: usize) -> Vec<(Vec<u32>, Vec<u32>)> {
        let mut out = Vec::new();
        for code in 0..3usize.pow(u32::try_from(n).unwrap()) {
            let (mut ys, mut xs) = (Vec::new(), Vec::new());
            let mut rest = code;
            for i in 0..n {
                match rest % 3 {
                    1 => ys.push(u32::try_from(i).unwrap()),
                    2 => xs.push(u32::try_from(i).unwrap()),
                    _ => {}
                }
                rest /= 3;
            }
            if !ys.is_empty() && !xs.is_empty() {
                out.push((ys, xs));
            }
        }
        out
    }

    /// The ADMG of `code` over the `n(n-1)/2` ordered pairs: two bits per pair
    /// (bit 0 directed `a -> b`, bit 1 bidirected `a <-> b`), so `code`
    /// ranges over every ADMG up to relabelling.
    pub(super) fn graph_of(n: usize, code: usize) -> (Admg, Edges, Edges) {
        let (mut d, mut b) = (Vec::new(), Vec::new());
        let mut k = 0;
        for x in 0..u32::try_from(n).unwrap() {
            for y in x + 1..u32::try_from(n).unwrap() {
                if (code >> (2 * k)) & 1 == 1 {
                    d.push((x, y));
                }
                if (code >> (2 * k)) & 2 == 2 {
                    b.push((x, y));
                }
                k += 1;
            }
        }
        (admg(u32::try_from(n).unwrap(), &d, &b), d, b)
    }

    pub(super) fn pairs(n: usize) -> u32 {
        u32::try_from(n * (n - 1) / 2).unwrap()
    }

    pub(super) fn report(name: &str, tally: &Tally) {
        let [idn, found, nc, ex, non] = tally.totals();
        eprintln!(
            "{name}: queries={} id_identifiable={idn} identified={found} not_certified={nc} exhausted={ex} id_not_identifiable={non} ratio_form={:?} unsound={}",
            tally.queries,
            tally.ratio,
            tally.unsound.len()
        );
        for ((y, x), c) in &tally.shapes {
            eprintln!(
                "  |Y|={y} |X|={x}: id={} found={} nc={} ex={} non={}",
                c[0], c[1], c[2], c[3], c[4]
            );
        }
    }

    /// Seeded generator of the sampled sweeps (a 64-bit LCG; deterministic).
    pub(super) struct Lcg(pub(super) u64);

    impl Lcg {
        pub(super) fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            self.0 >> 33
        }
    }

    /// `graphs` random ADMGs over `n` nodes (edge density varied per graph so the
    /// sample holds sparse and dense graphs), `per_graph` random queries each.
    pub(super) fn sample(n: usize, graphs: usize, per_graph: usize, seed: u64) -> Tally {
        let mut rng = Lcg(seed);
        let mut tally = Tally::default();
        let queries = all_queries(n);
        let nodes = u32::try_from(n).unwrap();
        for _ in 0..graphs {
            let density = 1 + rng.next() % 3;
            let (mut d, mut b) = (Vec::new(), Vec::new());
            for x in 0..nodes {
                for y in x + 1..nodes {
                    if rng.next() % 6 < density + 1 {
                        d.push((x, y));
                    }
                    if rng.next() % 6 < density {
                        b.push((x, y));
                    }
                }
            }
            let g = admg(nodes, &d, &b);
            for _ in 0..per_graph {
                let (ys, xs) = &queries[usize::try_from(rng.next()).unwrap() % queries.len()];
                record(&mut tally, &g, &d, &b, ys, xs);
            }
        }
        tally
    }

    /// Figures behind the documented completeness rates. Run with
    /// `N=<nodes> CASES=<graphs> cargo test -p antecedent-identify --lib --release
    /// measure -- --ignored --nocapture`; `CASES=0` is the exhaustive sweep over every
    /// ADMG (n=3: 768 queries; n=4: 204,800 queries, about 80 s in release).
    #[test]
    #[ignore = "measurement: minutes in release"]
    fn measure() {
        let var = |name: &str, default: usize| {
            std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
        };
        let (n, cases) = (var("N", 3), var("CASES", 0));
        let start = std::time::Instant::now();
        let tally = if cases == 0 {
            let mut tally = Tally::default();
            let queries = all_queries(n);
            for code in 0..4usize.pow(pairs(n)) {
                let (g, d, b) = graph_of(n, code);
                for (ys, xs) in &queries {
                    record(&mut tally, &g, &d, &b, ys, xs);
                }
            }
            tally
        } else {
            sample(n, cases, 3, 0xC0FF_EE00 + n as u64)
        };
        report(&format!("n={n} cases={cases} t={:?}", start.elapsed()), &tally);
        let mut gaps = tally.gaps.clone();
        gaps.sort();
        for (_, _, g) in gaps.iter().filter(|g| g.1 <= MIXED_SOURCE_MAX_DO).take(40) {
            eprintln!("  GAP {g}");
        }
        for u in &tally.unsound {
            eprintln!("  UNSOUND {u}");
        }
    }

    /// Every ADMG on three nodes, every query: the rule search identifies
    /// exactly what ID identifies (612 of 768 queries), and nothing ID refutes.
    #[test]
    fn three_node_sweep_matches_id_exactly() {
        let mut tally = Tally::default();
        let queries = all_queries(3);
        for code in 0..4usize.pow(pairs(3)) {
            let (g, d, b) = graph_of(3, code);
            for (ys, xs) in &queries {
                record(&mut tally, &g, &d, &b, ys, xs);
            }
        }
        report("three-node sweep", &tally);
        assert!(tally.unsound.is_empty(), "{:?}", tally.unsound);
        assert_eq!(tally.queries, 64 * 12);
        // [ID-identifiable, identified, not certified, exhausted, not identifiable]
        assert_eq!(tally.totals(), [612, 612, 0, 0, 156]);
    }

    /// Four- and five-node samples: never identified where ID refutes; every
    /// ID-identifiable query with at most `MIXED_SOURCE_MAX_DO` treatments is
    /// identified except the pinned napkin-family gaps; a larger treatment set is
    /// identified by the deeper passes or stops as a budget receipt. The rate floors are the measured rates
    /// (exhaustive four-node sweep 142,826 of 142,827; five-node sample 97.3%)
    /// rounded down.
    #[test]
    fn four_and_five_node_samples_are_sound_and_meet_the_completeness_floor() {
        let four = {
            // A stride coprime to 4^6 visits 57 graphs spread over the whole code space.
            let mut tally = Tally::default();
            let queries = all_queries(4);
            for code in (0..4usize.pow(pairs(4))).step_by(73) {
                let (g, d, b) = graph_of(4, code);
                for (ys, xs) in &queries {
                    record(&mut tally, &g, &d, &b, ys, xs);
                }
            }
            tally
        };
        let five = sample(5, 100, 3, 0x5EED_0005);
        for (name, tally) in [("four-node stride sample", &four), ("five-node sample", &five)] {
            report(name, tally);
            assert!(tally.unsound.is_empty(), "{name}: {:?}", tally.unsound);
            let [identifiable, identified, ..] = tally.totals();
            let within_do = tally.gaps.iter().filter(|g| g.1 <= MIXED_SOURCE_MAX_DO).count();
            let beyond_do = tally.gaps.iter().filter(|g| g.1 > MIXED_SOURCE_MAX_DO).count();
            assert_eq!(identifiable - identified, within_do + beyond_do);
            // Gaps within the first-pass intervention bound are the napkin family: rare.
            assert!(within_do <= 1, "{name}: {:?}", tally.gaps);
        }
        let [identifiable, identified, ..] = four.totals();
        assert!(identifiable > 1_500, "{identifiable}");
        assert!(identified * 1000 >= identifiable * 999, "{identified}/{identifiable}");
        let [identifiable, identified, ..] = five.totals();
        assert!(identifiable > 150, "{identifiable}");
        assert!(identified * 100 >= identifiable * 95, "{identified}/{identifiable}");
    }

    /// One known-incomplete query: ID identifies it, the rule search does not.
    struct Gap {
        name: &'static str,
        nodes: u32,
        directed: &'static [(u32, u32)],
        bidirected: &'static [(u32, u32)],
        outcomes: &'static [u32],
        treatments: &'static [u32],
        /// Verdicts the search may reach: the first-pass verdict, or what the deeper
        /// passes make of it (identified, or a budget receipt).
        allowed: &'static [Verdict],
    }

    /// The known gaps of the rule search against ID, one minimal example per class.
    /// A future improvement flips an entry deliberately: update the entry, the
    /// class list in the module docs and the record's `inference_notes` together.
    const KNOWN_GAPS: &[Gap] = &[
        // Class 1, the napkin: `W -> Z -> X -> Y` with `W <-> X` and `W <-> Y`.
        // ID identifies it through a ratio of a marginalised C-factor
        // (`P(y | do(x)) = sum_w P(y, x | z, w) P(w) / sum_w P(x | z, w) P(w)`),
        // a form the frozen rules reach from the joint only by a chain the
        // search's two-variable moves and forward closure do not find.
        Gap {
            name: "napkin",
            nodes: 4,
            directed: &[(0, 1), (1, 2), (2, 3)],
            bidirected: &[(0, 2), (0, 3)],
            outcomes: &[3],
            treatments: &[2],
            allowed: &[Verdict::NotCertified, Verdict::Identified, Verdict::Exhausted],
        },
        // Class 1 in a five-node graph (the only such gap in the seeded sample).
        Gap {
            name: "napkin_family_five_nodes",
            nodes: 5,
            directed: &[(0, 1), (0, 3), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4)],
            bidirected: &[(0, 2), (0, 4)],
            outcomes: &[4],
            treatments: &[2, 3],
            allowed: &[Verdict::NotCertified, Verdict::Identified, Verdict::Exhausted],
        },
        // Class 1 with two outcomes: the napkin `0 -> 1 -> 2 -> 4`, `0 <-> 2`, `0 <-> 4`
        // plus a second outcome `3` confounded with `4`.
        Gap {
            name: "napkin_family_two_outcomes",
            nodes: 5,
            directed: &[(0, 1), (1, 2), (2, 4), (3, 4)],
            bidirected: &[(0, 2), (0, 4), (3, 4)],
            outcomes: &[3, 4],
            treatments: &[2],
            allowed: &[Verdict::NotCertified, Verdict::Identified, Verdict::Exhausted],
        },
        // Class 2, the intervention bound: the first pass carries at most
        // `MIXED_SOURCE_MAX_DO` = 3 actions, so a fourth treatment is reached only by
        // a deeper pass (identified) or ends as a budget receipt, never `NotCertified`.
        Gap {
            name: "four_treatments_edgeless",
            nodes: 5,
            directed: &[],
            bidirected: &[],
            outcomes: &[0],
            treatments: &[1, 2, 3, 4],
            allowed: &[Verdict::Identified, Verdict::Exhausted],
        },
        Gap {
            name: "four_treatments_confounded_chain",
            nodes: 5,
            directed: &[(0, 1), (1, 2)],
            bidirected: &[(1, 3)],
            outcomes: &[4],
            treatments: &[0, 1, 2, 3],
            allowed: &[Verdict::Identified, Verdict::Exhausted],
        },
        // Class 3, the budget: with six nodes the default 20,000 operations run out
        // before the closure does, even for a trivial effect. A receipt, not a verdict.
        Gap {
            name: "six_nodes_exhaust_the_default_budget",
            nodes: 6,
            directed: &[(2, 3)],
            bidirected: &[],
            outcomes: &[5],
            treatments: &[0, 1, 2],
            allowed: &[Verdict::Exhausted],
        },
    ];

    #[test]
    fn known_incomplete_queries_stay_not_certified_until_deliberately_fixed() {
        for gap in KNOWN_GAPS {
            let g = admg(gap.nodes, gap.directed, gap.bidirected);
            assert!(
                id_identifiable(&g, gap.outcomes, gap.treatments).is_some(),
                "{}: ID must identify the pinned gap",
                gap.name
            );
            let verdict = search_only(&g, gap.outcomes, gap.treatments);
            assert!(
                gap.allowed.contains(&verdict),
                "{}: the rule search's verdict {verdict:?} on a known gap left {:?}",
                gap.name,
                gap.allowed
            );
        }
        // The one-treatment-fewer twins are identified by the first pass.
        let three = admg(4, &[], &[]);
        assert_eq!(search_only(&three, &[0], &[1, 2, 3]), Verdict::Identified);
        let chain = admg(5, &[(0, 1), (1, 2)], &[(1, 3)]);
        assert_eq!(search_only(&chain, &[4], &[0, 1, 2]), Verdict::Identified);
    }
}
