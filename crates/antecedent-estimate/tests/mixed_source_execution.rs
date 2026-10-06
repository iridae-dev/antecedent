//! Exact execution of checked mixed-source formulas against enumerated structural
//! models.
//!
//! Every law a fixture supplies is enumerated exactly from a binary structural
//! model, so a formula is checked against the model's own interventional truth.
//! The soundness sweep draws random ADMGs and models, random study catalogs and
//! random queries, and requires every identified query's number to equal the
//! enumerated truth.

mod common;
#[path = "common/mixed_fixture.rs"]
mod mixed_fixture;

use std::sync::Arc;

use antecedent_core::{ExecutionContext, Value};
use antecedent_estimate::{evaluate_exact_mixed_source, prepare_exact_mixed_source};
use antecedent_expr::{Assignment, ExactEvaluationLimits, ExactTransportData};
use antecedent_identify::{
    BoundMixedSourceFunctional, MIXED_SOURCE_DEFAULT_LIMITS, MixedSourceDecision, MixedSourceQuery,
    bind_mixed_source_catalog, decide_mixed_source,
};
use common::z_scm::{risk_of, vid};
use mixed_fixture::{
    Rng, X, Y, Z, build, chain_scm, edges32, frontdoor_scm, graph, query, query_many, random_scm,
    request, request_many, study,
};

fn bound(
    g: &antecedent_graph::Admg,
    q: &MixedSourceQuery,
    catalog: &antecedent_core::EvidenceCatalog,
) -> BoundMixedSourceFunctional {
    let ctx = ExecutionContext::for_tests(3);
    let MixedSourceDecision::Identified { derivation, .. } =
        decide_mixed_source(g, q, catalog, MIXED_SOURCE_DEFAULT_LIMITS, &ctx).unwrap()
    else {
        panic!("the catalog identifies the query");
    };
    bind_mixed_source_catalog(g, &derivation, catalog).unwrap()
}

fn point(
    functional: &BoundMixedSourceFunctional,
    data: &ExactTransportData,
    treatment: usize,
    level: bool,
) -> f64 {
    risk_of(
        &evaluate_exact_mixed_source(
            functional,
            data.clone(),
            request(treatment, level),
            ExactEvaluationLimits::default(),
            &ExecutionContext::for_tests(3),
        )
        .unwrap(),
    )
}

#[test]
fn complementary_studies_formula_matches_the_target_interventional_truth() {
    let scm = chain_scm();
    let g = graph(3, &[(0, 1), (1, 2)], &[]);
    let studies = [study("study-1", &[], &[X, Z]), study("study-2", &[], &[Z, Y])];
    let (catalog, data) = build(&scm, &studies);
    let functional = bound(&g, &query(Y, X), &catalog);
    for x in [0u8, 1] {
        let p = point(&functional, &data, X, x == 1);
        let truth = scm.risk(&[(X, x)], Y);
        assert!((p - truth).abs() < 1e-12, "do(X={x}): formula {p}, truth {truth}");
    }
    // The two studies' margins differ from the interventional truth of the joint's
    // other conditionals: the formula is not a passive relabelling.
    assert!((scm.risk(&[(X, 1)], Y) - scm.risk(&[(X, 0)], Y)).abs() > 1e-3);
}

#[test]
fn a_surrogate_trial_corrects_the_confounded_observational_conditional() {
    let scm = frontdoor_scm();
    let g = graph(3, &[(0, 1), (1, 2)], &[(0, 2)]);
    let studies = [study("observational", &[], &[X, Z]), study("trial", &[Z], &[X, Y])];
    let (catalog, data) = build(&scm, &studies);
    let functional = bound(&g, &query(Y, X), &catalog);
    for x in [0u8, 1] {
        let p = point(&functional, &data, X, x == 1);
        let truth = scm.risk(&[(X, x)], Y);
        assert!((p - truth).abs() < 1e-12, "do(X={x}): formula {p}, truth {truth}");
    }
    // The fixture is not trivial: the observational conditional is confounded.
    let observational = scm.law(&[], &[X, Y]);
    let naive = observational[3] / (observational[2] + observational[3]);
    assert!((naive - scm.risk(&[(X, 1)], Y)).abs() > 1e-2, "naive {naive}");
}

#[test]
fn the_plan_is_compiled_once_and_evaluated_without_searching() {
    let scm = chain_scm();
    let g = graph(3, &[(0, 1), (1, 2)], &[]);
    let (catalog, data) =
        build(&scm, &[study("study-1", &[], &[X, Z]), study("study-2", &[], &[Z, Y])]);
    let functional = bound(&g, &query(Y, X), &catalog);
    let ctx = ExecutionContext::for_tests(3);
    let plan = prepare_exact_mixed_source(
        &functional,
        data,
        request(X, true),
        ExactEvaluationLimits::default(),
        &ctx,
    )
    .unwrap();
    let first = plan.evaluate(&ctx).unwrap();
    let second = plan.evaluate(&ctx).unwrap();
    assert_eq!(
        first.probabilities.iter().map(|p| p.to_bits()).collect::<Vec<_>>(),
        second.probabilities.iter().map(|p| p.to_bits()).collect::<Vec<_>>()
    );
    assert!((risk_of(&first) - scm.risk(&[(X, 1)], Y)).abs() < 1e-12);
}

#[test]
fn every_alternative_derivation_computes_the_same_number() {
    let scm = chain_scm();
    let g = graph(3, &[(0, 1), (1, 2)], &[]);
    // Two trials of do(X) measuring {Y} both supply the target itself.
    let (catalog, data) =
        build(&scm, &[study("trial-a", &[X], &[Y]), study("trial-b", &[X], &[Y])]);
    let ctx = ExecutionContext::for_tests(3);
    let MixedSourceDecision::Identified { derivation, alternatives, .. } =
        decide_mixed_source(&g, &query(Y, X), &catalog, MIXED_SOURCE_DEFAULT_LIMITS, &ctx).unwrap()
    else {
        panic!("identified");
    };
    assert_eq!(alternatives.len(), 1);
    for candidate in std::iter::once(&*derivation).chain(&alternatives) {
        let functional = bind_mixed_source_catalog(&g, candidate, &catalog).unwrap();
        for x in [false, true] {
            let p = point(&functional, &data, X, x);
            assert!((p - scm.risk(&[(X, u8::from(x))], Y)).abs() < 1e-12);
        }
    }
}

#[test]
fn execution_reads_only_the_regimes_the_proof_cites_and_refuses_bad_requests() {
    let scm = chain_scm();
    let g = graph(3, &[(0, 1), (1, 2)], &[]);
    let studies = [study("study-1", &[], &[X, Z]), study("study-2", &[], &[Z, Y])];
    let (catalog, data) = build(&scm, &studies);
    let functional = bound(&g, &query(Y, X), &catalog);
    let ctx = ExecutionContext::for_tests(3);
    let limits = ExactEvaluationLimits::default();
    let evaluate = |data: ExactTransportData, request: Assignment| {
        evaluate_exact_mixed_source(&functional, data, request, limits, &ctx)
    };
    // A request must bind exactly the certified treatments.
    assert!(evaluate(data.clone(), Assignment::from_pairs([])).is_err());
    assert!(
        evaluate(
            data.clone(),
            Assignment::from_pairs([(vid(X), Value::Bool(true)), (vid(Z), Value::Bool(true))])
        )
        .is_err()
    );
    // A law of a cited regime is required.
    let without_second = ExactTransportData::try_new(
        data.laws().iter().filter(|law| law.regime().raw() != 2).cloned().collect::<Vec<_>>(),
        4096,
    )
    .unwrap();
    assert!(evaluate(without_second, request(X, true)).is_err());
    // A law that disagrees with its regime is refused before evaluation.
    let mut other = chain_scm();
    other.exo_p[0] = 0.9;
    let (_, shifted) = build(&other, &studies);
    let (_, mismatched) =
        build(&scm, &[study("study-1", &[], &[X]), study("study-2", &[], &[Z, Y])]);
    assert!(evaluate(mismatched, request(X, true)).is_err());
    // A different population's numbers evaluate (same snapshot names) to a different point:
    // the frozen catalog binds identity, not values.
    let moved = point(&functional, &shifted, X, true);
    assert!(
        (moved - scm.risk(&[(X, 1)], Y)).abs() < 1e-12,
        "the formula reads the chain's own margins"
    );
}

/// The truth check of one identified query: every candidate proof, at every level
/// of the treatments, equals the enumerated model's interventional truth.
fn assert_matches_truth(
    (g, scm, catalog, data): (
        &antecedent_graph::Admg,
        &common::z_scm::Scm,
        &antecedent_core::EvidenceCatalog,
        &ExactTransportData,
    ),
    (outcome, treatments): (usize, &[usize]),
    derivations: &[&antecedent_identify::MixedSourceDerivation],
    label: &str,
) {
    for candidate in derivations {
        let functional = bind_mixed_source_catalog(g, candidate, catalog).unwrap();
        for levels in 0..1usize << treatments.len() {
            let p = risk_of(
                &evaluate_exact_mixed_source(
                    &functional,
                    data.clone(),
                    request_many(treatments, levels),
                    ExactEvaluationLimits::default(),
                    &ExecutionContext::for_tests(3),
                )
                .unwrap(),
            );
            let do_ = treatments
                .iter()
                .enumerate()
                .map(|(k, t)| (*t, u8::try_from((levels >> k) & 1).unwrap()))
                .collect::<Vec<_>>();
            let truth = scm.risk(&do_, outcome);
            assert!(
                (p - truth).abs() < 1e-9,
                "{label}: do{do_:?} -> v{outcome}: formula {p}, truth {truth}\n{}",
                candidate.proof_graph(&[]).join("\n")
            );
        }
    }
}

type Edges = Vec<(usize, usize)>;

fn random_graph(rng: &mut Rng, n: usize) -> (Edges, Edges) {
    let (mut directed, mut bidirected) = (Vec::new(), Vec::new());
    for a in 0..n {
        for b in a + 1..n {
            match rng.below(7) {
                0 | 1 => directed.push((a, b)),
                2 => bidirected.push((a, b)),
                // A directed edge with a confounder on the same pair (a bow).
                3 => {
                    directed.push((a, b));
                    bidirected.push((a, b));
                }
                _ => {}
            }
        }
    }
    (directed, bidirected)
}

/// Random ADMGs, models, study catalogs (up to two experiments in one study) and
/// queries with one or several treatments: every identified query's number, at
/// every level of the treatments, equals the enumerated truth. The graphs draw
/// bows and confounded parents, so the sweep reaches rule 3 with inserted actions
/// that are and are not ancestors of the observed set.
#[test]
fn every_identified_query_matches_the_enumerated_truth() {
    let mut rng = Rng(0x0009_5EA2_C4E1);
    let ctx = ExecutionContext::for_tests(3);
    let (mut identified, mut with_experiment, mut named, mut trials) =
        (0usize, 0usize, 0usize, 0usize);
    let (mut multi, mut two_experiments) = (0usize, 0usize);
    let mut rules_used = std::collections::BTreeMap::<&str, usize>::new();
    for trial in 0..900 {
        let n = [3, 4, 4, 5][trial % 4];
        let (directed, bidirected) = random_graph(&mut rng, n);
        let g = graph(u32::try_from(n).unwrap(), &edges32(&directed), &edges32(&bidirected));
        let scm = random_scm(&mut rng, n, &directed, &bidirected);
        let mut studies = Vec::new();
        for s in 0..2 + rng.below(3) {
            let mut measured = (0..n).collect::<Vec<_>>();
            while measured.len() > 1 + rng.below(3) {
                measured.remove(rng.below(measured.len()));
            }
            let mut on = Vec::new();
            if rng.below(2) == 0 {
                let mut candidates = (0..n).filter(|v| !measured.contains(v)).collect::<Vec<_>>();
                while !candidates.is_empty() && on.len() < 1 + rng.below(2) {
                    on.push(candidates.remove(rng.below(candidates.len())));
                }
            }
            studies.push(study(&format!("study-{s}"), &on, &measured));
        }
        let (catalog, data) = build(&scm, &studies);
        let outcome = rng.below(n);
        let mut treatments = Vec::new();
        let wanted = if n >= 4 && rng.below(2) == 0 { 2 } else { 1 };
        while treatments.len() < wanted {
            let candidate = rng.below(n);
            if candidate != outcome && !treatments.contains(&candidate) {
                treatments.push(candidate);
            }
        }
        trials += 1;
        let q = query_many(outcome, &treatments);
        match decide_mixed_source(&g, &q, &catalog, MIXED_SOURCE_DEFAULT_LIMITS, &ctx).unwrap() {
            MixedSourceDecision::Identified { derivation, alternatives, .. } => {
                identified += 1;
                multi += usize::from(treatments.len() > 1);
                for step in derivation.steps() {
                    *rules_used.entry(step.rule.as_str()).or_default() += 1;
                }
                with_experiment += usize::from(studies.iter().any(|s| !s.on.is_empty()));
                two_experiments += usize::from(studies.iter().any(|s| s.on.len() > 1));
                let all = std::iter::once(&*derivation).chain(&alternatives).collect::<Vec<_>>();
                let label = format!(
                    "graph {directed:?} {bidirected:?} studies {:?}",
                    studies.iter().map(|s| (s.on.clone(), s.measured.clone())).collect::<Vec<_>>()
                );
                assert_matches_truth(
                    (&g, &scm, &catalog, &data),
                    (outcome, &treatments),
                    &all,
                    &label,
                );
            }
            MixedSourceDecision::NamedRoute { .. } => named += 1,
            _ => {}
        }
    }
    eprintln!(
        "{identified} identified ({multi} multi-treatment), {named} named, {trials} trials; \
         rules {rules_used:?}"
    );
    assert!(
        identified >= 40,
        "only {identified} of {trials} random queries identified ({named} named)"
    );
    assert!(with_experiment >= 10, "only {with_experiment} identified queries used an experiment");
    assert!(multi >= 10, "only {multi} identified queries had several treatments");
    assert!(two_experiments >= 1, "no identified query used a two-variable experiment");
    for rule in ["rule2_to_do", "rule3_insert", "marginalize", "product"] {
        assert!(
            rules_used.get(rule).copied().unwrap_or(0) > 0,
            "{rule} never used: {rules_used:?}"
        );
    }
}

/// Every three-node ADMG (each pair: none, directed, bidirected, both) with every
/// pair of studies from a menu (observational and single-experiment studies over
/// every measured subset) and every single-outcome query with any treatment set:
/// each identified formula equals the enumerated truth. This includes the
/// counterexample `W -> Z`, `W <-> Y` that rule 3's `W(Z)` exception exists for.
#[test]
fn every_identified_query_on_every_three_node_admg_matches_the_enumerated_truth() {
    let ctx = ExecutionContext::for_tests(3);
    let mut rng = Rng(0x3A3A_0001);
    let mut menu = Vec::new();
    for measured_mask in 1..8usize {
        let measured = (0..3).filter(|v| (measured_mask >> v) & 1 == 1).collect::<Vec<_>>();
        menu.push((Vec::new(), measured.clone()));
        for on in (0..3).filter(|v| !measured.contains(v)) {
            menu.push((vec![on], measured.clone()));
        }
    }
    let mut queries = Vec::new();
    for outcome in 0..3usize {
        let rest = (0..3).filter(|v| *v != outcome).collect::<Vec<_>>();
        for mask in 1..4usize {
            queries.push((
                outcome,
                (0..2).filter(|k| (mask >> k) & 1 == 1).map(|k| rest[k]).collect::<Vec<_>>(),
            ));
        }
    }
    let (mut identified, mut multi, mut checked) = (0usize, 0usize, 0usize);
    let mut rules_used = std::collections::BTreeMap::<&str, usize>::new();
    for code in 0..64usize {
        let (mut directed, mut bidirected) = (Vec::new(), Vec::new());
        for (k, (a, b)) in [(0usize, 1usize), (0, 2), (1, 2)].into_iter().enumerate() {
            let kind = (code >> (2 * k)) & 3;
            if kind & 1 == 1 {
                directed.push((a, b));
            }
            if kind & 2 == 2 {
                bidirected.push((a, b));
            }
        }
        let g = graph(3, &edges32(&directed), &edges32(&bidirected));
        let scm = random_scm(&mut rng, 3, &directed, &bidirected);
        for a in 0..menu.len() {
            for b in a + 1..menu.len() {
                // Every pair is a distinct catalog; the stride keeps the sweep bounded
                // while still visiting every pair against some graph.
                if (a * 31 + b * 7 + code) % 4 != 0 {
                    continue;
                }
                let studies = [
                    study("first", &menu[a].0, &menu[a].1),
                    study("second", &menu[b].0, &menu[b].1),
                ];
                let (catalog, data) = build(&scm, &studies);
                for (outcome, treatments) in &queries {
                    let q = query_many(*outcome, treatments);
                    let Ok(MixedSourceDecision::Identified { derivation, alternatives, .. }) =
                        decide_mixed_source(&g, &q, &catalog, MIXED_SOURCE_DEFAULT_LIMITS, &ctx)
                    else {
                        continue;
                    };
                    identified += 1;
                    multi += usize::from(treatments.len() > 1);
                    for step in derivation.steps() {
                        *rules_used.entry(step.rule.as_str()).or_default() += 1;
                    }
                    let all =
                        std::iter::once(&*derivation).chain(&alternatives).collect::<Vec<_>>();
                    checked += all.len();
                    let label = format!(
                        "graph {directed:?} {bidirected:?} studies {:?}",
                        (&menu[a], &menu[b])
                    );
                    assert_matches_truth(
                        (&g, &scm, &catalog, &data),
                        (*outcome, treatments),
                        &all,
                        &label,
                    );
                }
            }
        }
    }
    eprintln!("{identified} identified ({multi} multi), {checked} proofs; rules {rules_used:?}");
    assert!(identified > 200 && multi > 20, "{identified} {multi}");
    assert!(rules_used.get("rule3_insert").copied().unwrap_or(0) > 0, "{rules_used:?}");
}

/// The counterexample rule 3's `W(Z)` exception exists for: `W -> Z -> Y` and
/// `W <-> Y`, with the observational `P(Z, Y)` and a trial `P(Z | do(W))`. The
/// tempting formula `sum_z P(z | do(w)) P(y | z)` differs from the enumerated
/// truth, so the search must not return it: it returns nothing.
#[test]
fn the_ancestral_action_counterexample_is_never_identified_and_its_wrong_formula_differs() {
    let (w, zed, y) = (0usize, 1usize, 2usize);
    let directed = [(w, zed), (zed, y)];
    let bidirected = [(w, y)];
    let g = graph(3, &edges32(&directed), &edges32(&bidirected));
    let ctx = ExecutionContext::for_tests(3);
    let mut differing = 0usize;
    for seed in 0..20u64 {
        let mut rng = Rng(0xC0DE + seed);
        let scm = random_scm(&mut rng, 3, &directed, &bidirected);
        let (catalog, _) = build(
            &scm,
            &[study("observational", &[], &[zed, y]), study("trial-on-w", &[w], &[zed])],
        );
        let decision = decide_mixed_source(
            &g,
            &query_many(y, &[w]),
            &catalog,
            MIXED_SOURCE_DEFAULT_LIMITS,
            &ctx,
        )
        .unwrap();
        assert!(matches!(decision, MixedSourceDecision::NotCertified(_)), "{decision:?}");
        // The formula the unsound insertion would give, against the truth.
        for level in [0u8, 1] {
            let z_law = scm.law(&[(w, level)], &[zed]);
            let observational = scm.law(&[], &[zed, y]);
            let wrong = (0..2usize)
                .map(|z| {
                    let p_y_given_z = observational[(z << 1) | 1]
                        / (observational[z << 1] + observational[(z << 1) | 1]);
                    z_law[z] * p_y_given_z
                })
                .sum::<f64>();
            differing += usize::from((wrong - scm.risk(&[(w, level)], y)).abs() > 1e-3);
        }
    }
    assert!(differing > 3, "the counterexample must actually separate the formulas: {differing}");
}

/// A trial of `do(X = true)` only supplies one level. The symbolic search states its
/// identification for every level, so it never cites such a regime; a proof made from
/// the unrestricted family refuses to bind against a catalog that restricts the
/// regime, and the plan reads no level the family did not supply.
#[test]
fn a_value_restricted_trial_is_neither_searched_nor_bound_nor_executed() {
    use antecedent_core::{EvidenceCatalog, InterventionAssignment};
    use antecedent_expr::ExactTransportData as Data;
    let scm = chain_scm();
    let g = graph(3, &[(0, 1), (1, 2)], &[]);
    let (catalog, data) = build(&scm, &[study("trial", &[X], &[Y])]);
    let ctx = ExecutionContext::for_tests(3);
    let MixedSourceDecision::Identified { derivation, .. } =
        decide_mixed_source(&g, &query(Y, X), &catalog, MIXED_SOURCE_DEFAULT_LIMITS, &ctx).unwrap()
    else {
        panic!("the family trial supplies the target");
    };
    let functional = bind_mixed_source_catalog(&g, &derivation, &catalog).unwrap();
    // The same trial restricted to the level X = true.
    let mut regimes = catalog.regimes.to_vec();
    regimes[0].intervention_values =
        Arc::from([InterventionAssignment { variable: vid(X), value: Value::Bool(true) }]);
    let restricted = EvidenceCatalog::try_new(
        catalog.environments.to_vec(),
        regimes,
        catalog.bindings.to_vec(),
        None,
    )
    .unwrap();
    let decision =
        decide_mixed_source(&g, &query(Y, X), &restricted, MIXED_SOURCE_DEFAULT_LIMITS, &ctx)
            .unwrap();
    let MixedSourceDecision::NotCertified(inspection) = decision else {
        panic!("a restricted trial identifies nothing symbolically: {decision:?}");
    };
    assert_eq!(inspection.exclusions[0].reason, "restricted_levels");
    // A proof from the family refuses to bind to the restricted catalog.
    assert!(bind_mixed_source_catalog(&g, &derivation, &restricted).is_err());
    // Executing the family's plan on the restricted level's law alone evaluates that
    // level and refuses the level that was never measured.
    let only_true = Data::try_new(
        data.laws()
            .iter()
            .filter(|law| {
                law.interventions().iter().all(|a| a.value == antecedent_core::Value::Bool(true))
            })
            .cloned()
            .collect::<Vec<_>>(),
        4096,
    )
    .unwrap();
    let evaluate = |level: bool| {
        evaluate_exact_mixed_source(
            &functional,
            only_true.clone(),
            request(X, level),
            ExactEvaluationLimits::default(),
            &ctx,
        )
    };
    assert!(evaluate(true).is_ok());
    assert!(evaluate(false).is_err());
}

/// The Ananke surrogate-experiment specimen (GID and AID): `X1 -> W -> Y`, `X2 -> Y`,
/// `X1 <-> X2`, `X1 <-> W`, `X2 <-> Y`, with experiments on each treatment. The searched
/// formula matches the enumerated truth of `P(Y | do(X1, X2))`.
#[test]
fn the_ananke_gid_and_aid_specimens_match_the_enumerated_truth() {
    let (x1, x2, w, y) = (0usize, 1usize, 2usize, 3usize);
    let directed = [(x1, w), (w, y), (x2, y)];
    let bidirected = [(x1, x2), (x1, w), (x2, y)];
    let mut rng = Rng(0xA4A4);
    let scm = random_scm(&mut rng, 4, &directed, &bidirected);
    let g = graph(4, &[(0, 2), (2, 3), (1, 3)], &[(0, 1), (0, 2), (1, 3)]);
    let q = MixedSourceQuery {
        outcomes: Arc::from([vid(y)]),
        treatments: Arc::from([vid(x1), vid(x2)]),
        target: Arc::from("target"),
        sources: Arc::from([]),
    };
    let ctx = ExecutionContext::for_tests(3);
    let cases = [
        vec![study("exp-x1", &[x1], &[w, x2, y]), study("exp-x2", &[x2], &[w, x1, y])],
        vec![study("exp-x1-ancestral", &[x1], &[w]), study("exp-x2", &[x2], &[w, x1, y])],
    ];
    for studies in cases {
        let (catalog, data) = build(&scm, &studies);
        let MixedSourceDecision::Identified { derivation, .. } =
            decide_mixed_source(&g, &q, &catalog, MIXED_SOURCE_DEFAULT_LIMITS, &ctx).unwrap()
        else {
            panic!("identified");
        };
        let functional = bind_mixed_source_catalog(&g, &derivation, &catalog).unwrap();
        for (a, b) in [(0u8, 0u8), (0, 1), (1, 0), (1, 1)] {
            let request = Assignment::from_pairs([
                (vid(x1), Value::Bool(a == 1)),
                (vid(x2), Value::Bool(b == 1)),
            ]);
            let p = risk_of(
                &evaluate_exact_mixed_source(
                    &functional,
                    data.clone(),
                    request,
                    ExactEvaluationLimits::default(),
                    &ctx,
                )
                .unwrap(),
            );
            let truth = scm.risk(&[(x1, a), (x2, b)], y);
            assert!((p - truth).abs() < 1e-12, "do({a},{b}): {p} vs {truth}");
        }
    }
}
