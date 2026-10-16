//! Self-tests of the finite-SCM truth oracle and its counterfactual-transport
//! fixture library. Every asserted value is a hand calculation (documented in
//! the test) or a closed-form transport formula evaluated on enumerated
//! evidence tables. Nothing here touches the identification algorithm.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::float_cmp,
    clippy::needless_range_loop,
    dead_code,
    missing_docs,
    reason = "test oracle: dyadic probabilities make exact float equality the assertion"
)]

#[path = "support/finite_scm_oracle.rs"]
mod finite_scm_oracle;

#[path = "support/counterfactual_transport_cases.rs"]
mod counterfactual_transport_cases;

use counterfactual_transport_cases::{Case, ExpectedClass, all_cases};
use finite_scm_oracle::{
    Atom, CfVar, EvidenceSpec, FiniteScm, Input, Mechanism, Population, Setting, TargetQuery,
    all_regimes, bern, build_evidence, evidence_discrepancy, selection_discrepancy,
    verify_non_identifiability_witness,
};

const EVIDENCE_TOL: f64 = 1e-15;
const GAP_TOL: f64 = 1e-3;
const FORMULA_TOL: f64 = 1e-14;

/// `X = Ux ~ Bern(1/2)`, `Y = X xor Uy`, `Uy ~ Bern(1/4)`.
fn xor_model() -> FiniteScm {
    FiniteScm::new()
        .exo("Ux", bern(0.5))
        .exo("Uy", bern(0.25))
        .endo("X", 2, Mechanism::new(|_, u| u[0]))
        .endo("Y", 2, Mechanism::new(|v, u| v[0] ^ u[1]))
}

fn y_under(x: usize) -> CfVar {
    CfVar::under("Y", &[("X", x)])
}

#[test]
fn marginal_interventional_probabilities_match_hand_values() {
    // Y_{X=1} = 1 xor Uy is 1 iff Uy = 0 (3/4); Y_{X=0} = Uy is 1 w.p. 1/4.
    let m = xor_model();
    assert_eq!(m.probability(&[Atom::new(y_under(1), 1)]), 0.75);
    assert_eq!(m.probability(&[Atom::new(y_under(0), 1)]), 0.25);
    // Observational P(Y=1) = 1/2 * 3/4 + 1/2 * 1/4.
    assert_eq!(m.observational().probability(&[("Y", 1)]), 0.5);
    // The interventional law table agrees with the counterfactual marginal.
    let law = m.interventional(&[("X", 1)]);
    assert_eq!(law.probability(&[("X", 1), ("Y", 1)]), 0.75);
    assert_eq!(law.probability(&[("X", 1), ("Y", 0)]), 0.25);
    assert_eq!(law.probability(&[("X", 0)]), 0.0);
}

#[test]
fn counterfactual_joint_differs_from_product_of_marginals() {
    // (Y_1, Y_0) = (1 xor Uy, Uy): never (1,1) or (0,0); (1,0) iff Uy = 0.
    let m = xor_model();
    let t = m.counterfactual_table(&[y_under(1), y_under(0)]);
    assert_eq!(t.variables, vec!["Y_{X=1}", "Y_{X=0}"]);
    assert_eq!(t.probability(&[("Y_{X=1}", 1), ("Y_{X=0}", 1)]), 0.0);
    assert_eq!(t.probability(&[("Y_{X=1}", 1), ("Y_{X=0}", 0)]), 0.75);
    assert_eq!(t.probability(&[("Y_{X=1}", 0), ("Y_{X=0}", 1)]), 0.25);
    assert_eq!(t.probability(&[("Y_{X=1}", 0), ("Y_{X=0}", 0)]), 0.0);
    // The product of marginals would put 3/4 * 1/4 = 3/16 on (1,1).
    let product = 0.75 * 0.25;
    assert_eq!(product, 0.1875);
    assert_ne!(t.probability(&[("Y_{X=1}", 1), ("Y_{X=0}", 1)]), product);
}

#[test]
fn laws_are_normalized() {
    let m = xor_model();
    assert_eq!(m.observational().total(), 1.0);
    assert_eq!(m.interventional(&[("X", 0)]).total(), 1.0);
    assert_eq!(m.counterfactual_table(&[y_under(1), y_under(0)]).total(), 1.0);
    for case in all_cases() {
        for pop in &case.populations {
            assert_eq!(pop.scm.observational().total(), 1.0, "{}", case.name);
        }
        for table in build_evidence(&case.populations, &case.evidence) {
            assert_eq!(table.table.total(), 1.0, "{}", case.name);
        }
    }
}

#[test]
fn consistency_holds() {
    // Y_{X=x} = Y whenever X = x: P(Y_{X=1}=1, X=1) = P(Y=1, X=1) = 1/2 * 3/4.
    let m = xor_model();
    let cf = m.probability(&[Atom::new(y_under(1), 1), Atom::factual("X", 1)]);
    let fact = m.probability(&[Atom::factual("Y", 1), Atom::factual("X", 1)]);
    assert_eq!(cf, 0.375);
    assert_eq!(cf, fact);
    let cond = m.conditional(&[Atom::new(y_under(1), 1)], &[Atom::factual("X", 1)]);
    assert_eq!(cond.value(), Some(0.75));
    // Fully pointwise: the joint of (Y_{X=1}, Y, X) puts no mass where X=1 and
    // Y_{X=1} differs from Y.
    let t = m.counterfactual_table(&[y_under(1), CfVar::factual("Y"), CfVar::factual("X")]);
    for c in 0..t.probs.len() {
        let d = t.decode(c);
        if d[2] == 1 && d[0] != d[1] {
            assert_eq!(t.probs[c], 0.0);
        }
    }
}

#[test]
fn composition_holds() {
    // X -> W -> Y with W = X xor Uw (Uw ~ Bern(1/4)), Y = W.
    // Composition: Y_{X=1, W=W_{X=1}} = Y_{X=1}.
    let m = FiniteScm::new()
        .exo("Ux", bern(0.5))
        .exo("Uw", bern(0.25))
        .endo("X", 2, Mechanism::new(|_, u| u[0]))
        .endo("W", 2, Mechanism::new(|v, u| v[0] ^ u[1]))
        .endo("Y", 2, Mechanism::new(|v, _| v[1]));
    let w1 = CfVar::under("W", &[("X", 1)]);
    let composed =
        CfVar::nested("Y", vec![("X", Setting::Level(1)), ("W", Setting::cf(w1.clone()))]);
    let plain = CfVar::under("Y", &[("X", 1)]);
    let a = m.counterfactual_table(&[composed, w1.clone()]);
    let b = m.counterfactual_table(&[plain, w1]);
    assert_eq!(a.probs, b.probs);
    // Y_{X=1} = 1 xor Uw is 1 w.p. 3/4.
    assert_eq!(b.marginal(&["Y_{X=1}"]).probs, vec![0.25, 0.75]);
}

#[test]
fn nested_natural_effects_match_hand_values() {
    // X = Ux, M = X xor Um (Um ~ Bern(1/4)), Y = M xor X.
    // Y_{1, M_0} = M_0 xor 1 = Um xor 1  -> P = 3/4.
    // Y_{0, M_0} = Um                    -> P = 1/4.
    // Y_{0, M_1} = M_1 = 1 xor Um        -> P = 3/4.
    let m = FiniteScm::new()
        .exo("Ux", bern(0.5))
        .exo("Um", bern(0.25))
        .endo("X", 2, Mechanism::new(|_, u| u[0]))
        .endo("M", 2, Mechanism::new(|v, u| v[0] ^ u[1]))
        .endo("Y", 2, Mechanism::new(|v, _| v[1] ^ v[0]));
    let nested = |x: usize, xp: usize| {
        Atom::new(
            CfVar::nested(
                "Y",
                vec![("X", Setting::Level(x)), ("M", Setting::cf(CfVar::under("M", &[("X", xp)])))],
            ),
            1,
        )
    };
    assert_eq!(m.probability(&[nested(1, 0)]), 0.75);
    assert_eq!(m.probability(&[nested(0, 0)]), 0.25);
    assert_eq!(m.probability(&[nested(0, 1)]), 0.75);
    // NDE = 3/4 - 1/4 and NIE = 3/4 - 1/4.
    assert_eq!(m.probability(&[nested(1, 0)]) - m.probability(&[nested(0, 0)]), 0.5);
    assert_eq!(m.probability(&[nested(0, 1)]) - m.probability(&[nested(0, 0)]), 0.5);
}

#[test]
fn conditional_counterfactual_and_zero_support() {
    // X = U, Y = U with U ~ Bern(1/2): ETT P(Y_{X=0}=1 | X=1) = P(U=1 | U=1) = 1,
    // while P(Y_{X=0}=1) = P(U=1) = 1/2.
    let m = FiniteScm::new().exo("U", bern(0.5)).endo("X", 2, Mechanism::new(|_, u| u[0])).endo(
        "Y",
        2,
        Mechanism::new(|_, u| u[0]),
    );
    let event = [Atom::new(y_under(0), 1)];
    assert_eq!(m.conditional(&event, &[Atom::factual("X", 1)]).value(), Some(1.0));
    assert_eq!(m.probability(&event), 0.5);
    // A null conditioning event is undefined, not zero.
    let degenerate = m.with_exo_law("U", bern(0.0));
    let cond = degenerate.conditional(&event, &[Atom::factual("X", 1)]);
    assert_eq!(cond.evidence, 0.0);
    assert_eq!(cond.value(), None);
}

#[test]
fn lookup_mechanism_matches_closure() {
    let table =
        Mechanism::lookup(vec![Input::Endo(0), Input::Exo(1)], vec![2, 2], vec![0, 1, 1, 0]);
    let looked = FiniteScm::new()
        .exo("Ux", bern(0.5))
        .exo("Uy", bern(0.25))
        .endo("X", 2, Mechanism::new(|_, u| u[0]))
        .endo("Y", 2, table);
    assert_eq!(looked.observational(), xor_model().observational());
}

#[test]
fn induced_structure_and_selection_discrepancy_are_read_off_the_mechanisms() {
    let m = xor_model();
    let s = m.induced_structure();
    assert_eq!(s.directed, vec![("X".to_owned(), "Y".to_owned())]);
    assert!(s.bidirected.is_empty());
    // Changing the law of the Y-private noise flags exactly Y.
    let shifted = m.with_exo_law("Uy", bern(0.125));
    let d = selection_discrepancy(&m, &shifted);
    assert_eq!(d.exo_laws, vec!["Uy".to_owned()]);
    assert!(d.mechanism_vars.is_empty());
    assert_eq!(d.flagged, vec!["Y".to_owned()]);
    // Replacing a mechanism flags that variable.
    let swapped = m.with_mechanism("X", Mechanism::new(|_, _| 0));
    assert_eq!(selection_discrepancy(&m, &swapped).flagged, vec!["X".to_owned()]);
    // Identical populations differ nowhere.
    assert!(selection_discrepancy(&m, &m.clone()).flagged.is_empty());
}

#[test]
fn two_model_witness_helper_accepts_and_rejects() {
    // A: X = U, Y = X.  B: Y reads the natural X, so do(X) has no direct reach.
    let a = FiniteScm::new().exo("U", bern(0.5)).endo("X", 2, Mechanism::new(|_, u| u[0])).endo(
        "Y",
        2,
        Mechanism::new(|v, _| v[0]),
    );
    let b = a.blind_to_natural("Y", "X");
    let fa = vec![Population::new("p", a.clone())];
    let fb = vec![Population::new("p", b)];
    let specs = vec![EvidenceSpec::new("p", &[], &["X", "Y"])];
    let q = TargetQuery::new("p", vec![Atom::new(y_under(1), 1)], vec![]);
    let report = verify_non_identifiability_witness(&fa, &fb, &specs, &q, EVIDENCE_TOL, GAP_TOL)
        .expect("a valid witness");
    assert_eq!(report.max_evidence_discrepancy, 0.0);
    assert_eq!(report.value_a, 1.0);
    assert_eq!(report.value_b, 0.5);
    assert_eq!(report.gap, 0.5);
    // Supplying the do(X=1) table breaks the agreement.
    let with_do = vec![
        EvidenceSpec::new("p", &[], &["X", "Y"]),
        EvidenceSpec::new("p", &[("X", 1)], &["X", "Y"]),
    ];
    assert!(
        verify_non_identifiability_witness(&fa, &fb, &with_do, &q, EVIDENCE_TOL, GAP_TOL)
            .unwrap_err()
            .contains("disagree on supplied evidence")
    );
    // A model compared with itself agrees on the target: not a witness.
    assert!(
        verify_non_identifiability_witness(&fa, &fa, &specs, &q, EVIDENCE_TOL, GAP_TOL)
            .unwrap_err()
            .contains("agree on the target")
    );
}

#[test]
fn all_regimes_enumerates_every_subset_and_level_combination() {
    // Binary X, Y: 1 (empty) + 2 + 2 + 4 regimes.
    assert_eq!(all_regimes(&xor_model()).len(), 9);
}

// ---------------------------------------------------------------------------
// fixture library
// ---------------------------------------------------------------------------

fn case(name: &str) -> Case {
    all_cases().into_iter().find(|c| c.name == name).unwrap_or_else(|| panic!("no case {name}"))
}

fn sorted(mut v: Vec<(String, String)>) -> Vec<(String, String)> {
    v.sort();
    v
}

#[test]
fn declared_graphs_match_the_structure_induced_by_the_models() {
    for c in all_cases() {
        let declared_directed: Vec<(String, String)> = sorted(
            c.graph.directed.iter().map(|(a, b)| ((*a).to_owned(), (*b).to_owned())).collect(),
        );
        let declared_bidirected: Vec<(String, String)> = sorted(
            c.graph
                .bidirected
                .iter()
                .map(|(a, b)| {
                    let (a, b) = ((*a).to_owned(), (*b).to_owned());
                    if a <= b { (a, b) } else { (b, a) }
                })
                .collect(),
        );
        let families = std::iter::once(&c.populations).chain(c.alternative.iter());
        for (n, family) in families.enumerate() {
            for pop in family {
                let ind = pop.scm.induced_structure();
                for e in &ind.directed {
                    assert!(
                        declared_directed.contains(e),
                        "{} {}: extra edge {e:?}",
                        c.name,
                        pop.name
                    );
                }
                for e in &ind.bidirected {
                    assert!(
                        declared_bidirected.contains(e),
                        "{} {}: extra bidirected edge {e:?}",
                        c.name,
                        pop.name
                    );
                }
                // The primary family realizes the declared graph exactly.
                if n == 0 {
                    assert_eq!(ind.directed, declared_directed, "{} {}", c.name, pop.name);
                    assert_eq!(ind.bidirected, declared_bidirected, "{} {}", c.name, pop.name);
                }
            }
        }
    }
}

#[test]
fn populations_differ_only_at_declared_selection_nodes() {
    for c in all_cases() {
        let flagged_ok: Vec<&str> = c.graph.selection.iter().map(|(_, v)| *v).collect();
        let mut families = vec![&c.populations];
        if let Some(alt) = &c.alternative {
            families.push(alt);
        }
        for family in families {
            if family.len() < 2 {
                continue;
            }
            let src = &family[0].scm;
            for other in &family[1..] {
                let d = selection_discrepancy(src, &other.scm);
                for v in &d.flagged {
                    assert!(
                        flagged_ok.contains(&v.as_str()),
                        "{}: {} differs at undeclared node {v}",
                        c.name,
                        other.name
                    );
                }
            }
        }
    }
}

#[test]
fn fixture_values_class_and_witness_flags_are_consistent() {
    let cases = all_cases();
    assert_eq!(cases.len(), 8);
    for c in &cases {
        assert!(!c.derivation.is_empty());
        for t in &c.targets {
            if t.class == ExpectedClass::UndefinedZeroSupport {
                assert!(t.expected_value.is_none());
            } else {
                let v = t.expected_value.expect("defined target");
                assert!((0.0..=1.0).contains(&v), "{} {}", c.name, t.label);
            }
        }
        let obstructed = c.targets[0].class != ExpectedClass::Identified
            && c.targets[0].class != ExpectedClass::UndefinedZeroSupport;
        assert_eq!(c.alternative.is_some(), obstructed, "{}", c.name);
    }
}

#[test]
fn nonlinear_mediation_values_and_mediation_formula() {
    let c = case("nonlinear_mediation_covariate_shift");
    let expected = [0.5, 23.0 / 64.0, 19.0 / 32.0, 5.0 / 16.0];
    let (xs, xps) = ([1, 0, 1, 0], [0, 0, 1, 1]);
    for (k, t) in c.targets.iter().enumerate() {
        assert_eq!(t.expected_value, Some(expected[k]), "{}", t.label);
    }
    // Mediation formula on the supplied tables: source full joint, target Z marginal.
    let ev = build_evidence(&c.populations, &c.evidence);
    let (src, tz) = (&ev[0].table, &ev[1].table);
    for k in 0..4 {
        let mut f = 0.0;
        for z in 0..2 {
            for m in 0..3 {
                let pm =
                    src.conditional(&[("M", m)], &[("X", xps[k]), ("Z", z)]).expect("positivity");
                let py = src
                    .conditional(&[("Y", 1)], &[("X", xs[k]), ("M", m), ("Z", z)])
                    .expect("positivity");
                f += tz.probability(&[("Z", z)]) * pm * py;
            }
        }
        assert!((f - expected[k]).abs() <= FORMULA_TOL, "formula {k}: {f}");
    }
    // NDE and NIE of the nonlinear mechanism.
    let nde = expected[0] - expected[1];
    let nie = expected[3] - expected[1];
    assert_eq!(nde, 0.5 - 23.0 / 64.0);
    assert_eq!(nie, 5.0 / 16.0 - 23.0 / 64.0);
}

#[test]
fn two_regime_joint_counterfactual_and_ett_match_transport_formulas() {
    let joint = case("joint_counterfactual_two_source_regimes");
    let ett = case("effect_of_treatment_on_treated_with_selection");
    assert_eq!(joint.targets[0].expected_value, Some(47.0 / 128.0));
    assert_eq!(ett.targets[0].expected_value, Some(47.0 / 64.0));
    let ev = build_evidence(&joint.populations, &joint.evidence);
    let (obs, do1, tz) = (&ev[0].table, &ev[2].table, &ev[3].table);
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for z in 0..2 {
        let pz = tz.probability(&[("Z", z)]);
        let exp = do1.conditional(&[("Y", 1)], &[("Z", z)]).unwrap();
        let both = obs.conditional(&[("Y", 1), ("X", 1)], &[("Z", z)]).unwrap();
        numerator += pz * (exp - both);
        denominator += pz * obs.conditional(&[("X", 0)], &[("Z", z)]).unwrap();
    }
    assert!((numerator - 47.0 / 128.0).abs() <= FORMULA_TOL);
    assert!((numerator / denominator - 47.0 / 64.0).abs() <= FORMULA_TOL);
    // Marginal X=0 in the target is exactly 1/2 here, so the ETT is twice the joint.
    assert!((denominator - 0.5).abs() <= FORMULA_TOL);
}

#[test]
fn latent_district_front_door_formula_matches_enumeration() {
    let c = case("transportable_latent_district");
    assert_eq!(c.targets[0].expected_value, Some(0.5));
    let ev = build_evidence(&c.populations, &c.evidence);
    let (src, tx) = (&ev[0].table, &ev[1].table);
    let mut f = 0.0;
    for m in 0..2 {
        let pm = tx.conditional(&[("M", m)], &[("X", 1)]).unwrap();
        let inner: f64 = (0..2)
            .map(|x| {
                src.conditional(&[("Y", 1)], &[("X", x), ("M", m)]).unwrap()
                    * tx.probability(&[("X", x)])
            })
            .sum();
        f += pm * inner;
    }
    assert!((f - 0.5).abs() <= FORMULA_TOL, "{f}");
}

#[test]
fn recanting_witness_models_agree_on_every_regime_and_differ_on_the_target() {
    let c = case("recanting_witness_without_selection");
    let alt = c.alternative.as_ref().unwrap();
    // 1 + 4 + 4 + 4 + 2 combos per subset size: all subsets of 4 binary variables = 3^4 regimes.
    assert_eq!(c.evidence.len(), 81);
    assert_eq!(evidence_discrepancy(&c.populations, alt, &c.evidence), 0.0);
    let report = verify_non_identifiability_witness(
        &c.populations,
        alt,
        &c.evidence,
        &c.targets[0].query,
        EVIDENCE_TOL,
        GAP_TOL,
    )
    .expect("recanting witness");
    assert_eq!(report.value_a, 0.75);
    assert_eq!(report.value_b, 0.25);
    // Cross-world joint of (W_{X=1}, W_{X=0}): differs although both marginals match.
    let w = [CfVar::under("W", &[("X", 1)]), CfVar::under("W", &[("X", 0)])];
    let ta = c.populations[0].scm.counterfactual_table(&w);
    let tb = alt[0].scm.counterfactual_table(&w);
    assert_eq!(ta.marginal(&["W_{X=1}"]).probs, tb.marginal(&["W_{X=1}"]).probs);
    assert_eq!(ta.marginal(&["W_{X=0}"]).probs, tb.marginal(&["W_{X=0}"]).probs);
    assert_ne!(ta.probs, tb.probs);
}

#[test]
fn outcome_selection_witness_is_a_valid_obstruction() {
    let c = case("selection_on_outcome_mechanism");
    let alt = c.alternative.as_ref().unwrap();
    let report = verify_non_identifiability_witness(
        &c.populations,
        alt,
        &c.evidence,
        &c.targets[0].query,
        EVIDENCE_TOL,
        GAP_TOL,
    )
    .expect("selection-on-outcome witness");
    assert_eq!(report.value_a, 7.0 / 8.0);
    assert_eq!(report.value_b, 7.0 / 16.0);
    assert_eq!(c.targets[0].expected_value, Some(7.0 / 8.0));
}

#[test]
fn zero_support_target_is_undefined_but_source_is_defined() {
    let c = case("zero_support_conditioning_event");
    assert_eq!(c.targets[0].expected_value, Some(0.25));
    assert_eq!(c.targets[0].class, ExpectedClass::Identified);
    assert_eq!(c.targets[1].expected_value, None);
    assert_eq!(c.targets[1].class, ExpectedClass::UndefinedZeroSupport);
    // The null event is exactly zero mass in the supplied target table.
    let ev = build_evidence(&c.populations, &c.evidence);
    assert_eq!(ev[2].table.probability(&[("X", 1)]), 0.0);
}

#[test]
fn missing_experiment_is_witnessed_and_resolved_by_supplying_it() {
    let c = case("missing_interventional_evidence");
    let alt = c.alternative.as_ref().unwrap();
    let report = verify_non_identifiability_witness(
        &c.populations,
        alt,
        &c.evidence,
        &c.targets[0].query,
        EVIDENCE_TOL,
        GAP_TOL,
    )
    .expect("missing-experiment witness");
    assert_eq!(report.value_a, 19.0 / 32.0);
    assert_eq!(report.value_b, 23.0 / 64.0);
    // Supplying the source do(X=1) table separates the two models, so the
    // witness no longer applies: the same target is identified from it.
    let mut with_experiment = c.evidence.clone();
    with_experiment.push(EvidenceSpec::new("source", &[("X", 1)], &["Z", "X", "Y"]));
    assert!(evidence_discrepancy(&c.populations, alt, &with_experiment) > GAP_TOL);
    let c2 = case("joint_counterfactual_two_source_regimes");
    assert!(c2.evidence.iter().any(|e| e.population == "source" && !e.regime.is_empty()));
}
