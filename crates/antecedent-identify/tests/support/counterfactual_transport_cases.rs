//! Named finite-SCM fixtures for general counterfactual transport.
//!
//! Each [`Case`] fixes a selection diagram, a finite SCM per population, the
//! evidence regimes supplied, one or more counterfactual targets with their
//! exact values (computed by the independent enumerator in
//! `finite_scm_oracle`), and the expected identifiability class from
//! Correa-Lee-Bareinboim style counterfactual transport theory, derived by hand
//! in each case's `derivation` text. Non-identifiable cases carry an explicit
//! second model family (`alternative`) that agrees on every supplied evidence
//! table and differs on the target.
//!
//! This module reaches the oracle as `crate::finite_scm_oracle`; declare both
//! modules at the test crate root.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::float_cmp,
    dead_code,
    missing_docs,
    reason = "test fixtures: dyadic probabilities, exact enumerated values, shared across test binaries"
)]

use crate::finite_scm_oracle::{
    Atom, CfVar, EvidenceSpec, FiniteScm, Mechanism, Population, Setting, TargetQuery, all_regimes,
    bern,
};

/// What the theory says about a target given the declared graph and evidence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpectedClass {
    /// Identified from the supplied evidence by a transport formula.
    Identified,
    /// Not identifiable in any population of the declared model family from
    /// any evidence (a rung-three obstruction such as a recanting witness),
    /// even without a selection node.
    NotIdentifiable,
    /// Identifiable in the source, but a selection node on the outcome (or an
    /// unblockable path to it) prevents transporting it to the target.
    NotTransportable,
    /// Identifiable if the right regime were supplied, but the supplied
    /// regimes do not determine it (typed "need more evidence" refusal).
    EvidenceInsufficient,
    /// The target conditions on a probability-zero event; the quantity is
    /// undefined, not merely unidentified.
    UndefinedZeroSupport,
}

/// The declared graph and selection diagram of a case.
#[derive(Clone, Debug)]
pub struct CaseGraph {
    pub observed: Vec<&'static str>,
    pub directed: Vec<(&'static str, &'static str)>,
    pub bidirected: Vec<(&'static str, &'static str)>,
    /// `(selection node, variable whose mechanism it flags)`.
    pub selection: Vec<(&'static str, &'static str)>,
}

/// One counterfactual target with its exact value and expected class.
#[derive(Clone, Debug)]
pub struct Target {
    pub label: &'static str,
    pub query: TargetQuery,
    /// Exact value in the stated population; `None` if conditioning evidence
    /// has probability zero.
    pub expected_value: Option<f64>,
    pub class: ExpectedClass,
}

/// A transport fixture.
#[derive(Clone, Debug)]
pub struct Case {
    pub name: &'static str,
    pub derivation: &'static str,
    pub graph: CaseGraph,
    /// The true model family (a source population per entry plus the target).
    pub populations: Vec<Population>,
    pub target_population: &'static str,
    /// Evidence regimes supplied to the identifier.
    pub evidence: Vec<EvidenceSpec>,
    pub targets: Vec<Target>,
    /// A second model family agreeing on every evidence table but differing on
    /// the (first) target; present exactly for non-identified/transportable
    /// obstruction cases.
    pub alternative: Option<Vec<Population>>,
}

fn target(
    label: &'static str,
    populations: &[Population],
    query: TargetQuery,
    class: ExpectedClass,
) -> Target {
    let expected_value = query.value(populations);
    Target {
        label,
        query,
        expected_value,
        class,
    }
}

fn pops(source: FiniteScm, target: FiniteScm) -> Vec<Population> {
    vec![
        Population::new("source", source),
        Population::new("target", target),
    ]
}

fn y_is_one(var: CfVar) -> Vec<Atom> {
    vec![Atom::new(var, 1)]
}

fn nested(var: &str, x: usize, m: &str, xp: usize) -> CfVar {
    CfVar::nested(
        var,
        vec![
            ("X", Setting::Level(x)),
            (m, Setting::cf(CfVar::under(m, &[("X", xp)]))),
        ],
    )
}

// ---------------------------------------------------------------------------
// (1) nonlinear mediation, natural direct / indirect effect, covariate shift
// ---------------------------------------------------------------------------

fn mediation_population(uz: f64) -> FiniteScm {
    // Z = Uz; X = Z xor Ux; M = (X + Z + Um) mod 3 (ternary); Y = [M + X*Z >= 2] xor Uy.
    FiniteScm::new()
        .exo("Uz", bern(uz))
        .exo("Ux", bern(0.25))
        .exo("Um", vec![0.25, 0.25, 0.5])
        .exo("Uy", bern(0.125))
        .endo("Z", 2, Mechanism::new(|_, u| u[0]))
        .endo("X", 2, Mechanism::new(|v, u| v[0] ^ u[1]))
        .endo("M", 3, Mechanism::new(|v, u| (v[1] + v[0] + u[2]) % 3))
        .endo(
            "Y",
            2,
            Mechanism::new(|v, u| usize::from(v[2] + v[1] * v[0] >= 2) ^ u[3]),
        )
}

/// Natural direct/indirect effect transport. Only the covariate law `P(Z)`
/// differs (selection node on `Z`).
///
/// Derivation. The model is Markovian, `Z -> X, Z -> M, Z -> Y, X -> M,
/// X -> Y, M -> Y`. With no recanting witness the nested counterfactual
/// `Y_{x, M_{x'}}` is identified by the mediation formula
/// `sum_{z,m} P(y | x, m, z) P(m | x', z) P(z)`. The selection node flags only
/// `Z`, so `P(y | x, m, z)` and `P(m | x', z)` are invariant (their variables
/// are d-separated from `S` given `Z`); only `P*(z)`, available as a target
/// marginal, is substituted. Positivity holds (every `(x, m, z)` cell has mass).
/// NDE = `Y_{1,M_0} - Y_{0,M_0}`, NIE = `Y_{0,M_1} - Y_{0,M_0}`.
pub fn nonlinear_mediation_covariate_shift() -> Case {
    let source = mediation_population(0.25);
    let tgt = mediation_population(0.75);
    let populations = pops(source, tgt);
    let q = |x, xp| TargetQuery::new("target", y_is_one(nested("Y", x, "M", xp)), vec![]);
    Case {
        name: "nonlinear_mediation_covariate_shift",
        derivation: "Markovian mediation with ternary mediator and threshold outcome; selection \
                     on Z only; Y_{x,M_{x'}} = sum_{z,m} P(y|x,m,z)P(m|x',z)P*(z)",
        graph: CaseGraph {
            observed: vec!["Z", "X", "M", "Y"],
            directed: vec![
                ("Z", "X"),
                ("Z", "M"),
                ("Z", "Y"),
                ("X", "M"),
                ("X", "Y"),
                ("M", "Y"),
            ],
            bidirected: vec![],
            selection: vec![("S_Z", "Z")],
        },
        target_population: "target",
        evidence: vec![
            EvidenceSpec::new("source", &[], &["Z", "X", "M", "Y"]),
            EvidenceSpec::new("target", &[], &["Z"]),
        ],
        targets: vec![
            target("Y_{x1,M_{x0}}", &populations, q(1, 0), ExpectedClass::Identified),
            target("Y_{x0,M_{x0}}", &populations, q(0, 0), ExpectedClass::Identified),
            target("Y_{x1,M_{x1}}", &populations, q(1, 1), ExpectedClass::Identified),
            target("Y_{x0,M_{x1}}", &populations, q(0, 1), ExpectedClass::Identified),
        ],
        populations,
        alternative: None,
    }
}

// ---------------------------------------------------------------------------
// confounded covariate model shared by (2), (3) and (8)
// ---------------------------------------------------------------------------

fn confounded_covariate_population(uz: f64) -> FiniteScm {
    // Z = Uz; X = U xor (Z and Ux); Y = X xor (Z and U) xor Uy. U confounds X, Y.
    FiniteScm::new()
        .exo("Uz", bern(uz))
        .exo("U", bern(0.5))
        .exo("Ux", bern(0.25))
        .exo("Uy", bern(0.125))
        .endo("Z", 2, Mechanism::new(|_, u| u[0]))
        .endo("X", 2, Mechanism::new(|v, u| u[1] ^ (v[0] & u[2])))
        .endo(
            "Y",
            2,
            Mechanism::new(|v, u| v[1] ^ (v[0] & u[1]) ^ u[3]),
        )
}

fn confounded_graph() -> CaseGraph {
    CaseGraph {
        observed: vec!["Z", "X", "Y"],
        directed: vec![("Z", "X"), ("Z", "Y"), ("X", "Y")],
        bidirected: vec![("X", "Y")],
        selection: vec![("S_Z", "Z")],
    }
}

fn source_and_target_z() -> (FiniteScm, FiniteScm) {
    (
        confounded_covariate_population(0.25),
        confounded_covariate_population(0.75),
    )
}

/// Joint counterfactual `P*(Y_{x=1}=1, X=0)` needing the source observational
/// law and the source `do(X)` experiment jointly, plus the target marginal of Z.
///
/// Derivation. Selection flags `Z` only, so `(X, Y_x)` is independent of `S`
/// given `Z`. Binary `X`: `P(Y_1=1, X=0 | z) = P(Y_1=1 | z) - P(Y=1, X=1 | z)`
/// by consistency, so
/// `P*(Y_1=1, X=0) = sum_z P*(z) [P(y=1 | do(x=1), z) - P(y=1, x=1 | z)]`.
/// `X <-> Y` makes `P(y | do(x), z)` unreachable from observations (needs the
/// experiment), and the second term is source-observational only.
pub fn joint_counterfactual_two_source_regimes() -> Case {
    let (s, t) = source_and_target_z();
    let populations = pops(s, t);
    let query = TargetQuery::new(
        "target",
        vec![
            Atom::new(CfVar::under("Y", &[("X", 1)]), 1),
            Atom::factual("X", 0),
        ],
        vec![],
    );
    Case {
        name: "joint_counterfactual_two_source_regimes",
        derivation: "P*(Y_1=1,X=0)=sum_z P*(z)[P(y=1|do(x=1),z)-P(y=1,x=1|z)]; needs source obs \
                     and source do(X) together with target marginal of Z",
        graph: confounded_graph(),
        target_population: "target",
        evidence: vec![
            EvidenceSpec::new("source", &[], &["Z", "X", "Y"]),
            EvidenceSpec::new("source", &[("X", 0)], &["Z", "X", "Y"]),
            EvidenceSpec::new("source", &[("X", 1)], &["Z", "X", "Y"]),
            EvidenceSpec::new("target", &[], &["Z"]),
        ],
        targets: vec![target(
            "Y_{x1}=1,X=0",
            &populations,
            query,
            ExpectedClass::Identified,
        )],
        populations,
        alternative: None,
    }
}

/// Effect of treatment on the treated, `P*(Y_{x=1}=1 | X=0)`, with factual
/// evidence and a selection node on the covariate.
///
/// Derivation. Same model and numerator as the joint counterfactual case; the
/// denominator is `P*(X=0) = sum_z P*(z) P(X=0 | z)` (X's mechanism given Z is
/// invariant), which needs the source observational law. ETT is the ratio.
pub fn effect_of_treatment_on_treated_with_selection() -> Case {
    let (s, t) = source_and_target_z();
    let populations = pops(s, t);
    let query = TargetQuery::new(
        "target",
        vec![Atom::new(CfVar::under("Y", &[("X", 1)]), 1)],
        vec![Atom::factual("X", 0)],
    );
    Case {
        name: "effect_of_treatment_on_treated_with_selection",
        derivation: "ETT = sum_z P*(z)[P(y|do(x1),z)-P(y,x1|z)] / sum_z P*(z)P(x0|z); needs \
                     source obs + source do(X) + target marginal of Z",
        graph: confounded_graph(),
        target_population: "target",
        evidence: vec![
            EvidenceSpec::new("source", &[], &["Z", "X", "Y"]),
            EvidenceSpec::new("source", &[("X", 0)], &["Z", "X", "Y"]),
            EvidenceSpec::new("source", &[("X", 1)], &["Z", "X", "Y"]),
            EvidenceSpec::new("target", &[], &["Z"]),
        ],
        targets: vec![target(
            "P(Y_{x1}=1 | X=0)",
            &populations,
            query,
            ExpectedClass::Identified,
        )],
        populations,
        alternative: None,
    }
}

// ---------------------------------------------------------------------------
// (4) transportable latent district
// ---------------------------------------------------------------------------

fn front_door_population(um: f64) -> FiniteScm {
    // U confounds X, Y. X = U xor Ux; M = X xor Um; Y = M xor U xor Uy.
    FiniteScm::new()
        .exo("U", bern(0.5))
        .exo("Ux", bern(0.125))
        .exo("Um", bern(um))
        .exo("Uy", bern(0.125))
        .endo("X", 2, Mechanism::new(|_, u| u[0] ^ u[1]))
        .endo("M", 2, Mechanism::new(|v, u| v[0] ^ u[2]))
        .endo("Y", 2, Mechanism::new(|v, u| v[1] ^ u[0] ^ u[3]))
}

/// A bidirected district `{X, Y}` with the selection node on `M`, outside it.
///
/// Derivation. Front-door graph `X -> M -> Y`, `X <-> Y`, selection flags `M`.
/// The district `{X, Y}` has `Q[{X,Y}]` fully source-identified (`P(y | m, x)`,
/// `P(x)` are invariant since `{M, X}` blocks `S -> M <- X <-> Y` at `X`).
/// `P*(y | do(x)) = sum_m P*(m | x) sum_{x'} P(y | m, x') P*(x')`. The target
/// need only observe `{X, M}`; `Y` is never measured there.
pub fn transportable_latent_district() -> Case {
    let populations = pops(front_door_population(0.25), front_door_population(0.5));
    let query = TargetQuery::new(
        "target",
        vec![Atom::new(CfVar::under("Y", &[("X", 1)]), 1)],
        vec![],
    );
    Case {
        name: "transportable_latent_district",
        derivation: "front-door with latent X<->Y district; selection on M outside the district; \
                     P*(y|do(x)) = sum_m P*(m|x) sum_x' P(y|m,x')P*(x')",
        graph: CaseGraph {
            observed: vec!["X", "M", "Y"],
            directed: vec![("X", "M"), ("M", "Y")],
            bidirected: vec![("X", "Y")],
            selection: vec![("S_M", "M")],
        },
        target_population: "target",
        evidence: vec![
            EvidenceSpec::new("source", &[], &["X", "M", "Y"]),
            EvidenceSpec::new("target", &[], &["X", "M"]),
        ],
        targets: vec![target(
            "Y_{x1}=1",
            &populations,
            query,
            ExpectedClass::Identified,
        )],
        populations,
        alternative: None,
    }
}

// ---------------------------------------------------------------------------
// (5) recanting witness
// ---------------------------------------------------------------------------

fn recanting_witness_population(blind_w: bool) -> FiniteScm {
    // X = Ux; W = X xor U (model A) or W = U (model B); Z = W; Y = W xor Z xor Uy.
    let w = if blind_w {
        Mechanism::new(|_, u| u[0])
    } else {
        Mechanism::new(|v, u| v[0] ^ u[0])
    };
    FiniteScm::new()
        .exo("U", bern(0.5))
        .exo("Ux", bern(0.5))
        .exo("Uy", bern(0.25))
        .endo("X", 2, Mechanism::new(|_, u| u[1]))
        .endo("W", 2, w)
        .endo("Z", 2, Mechanism::new(|v, _| v[1]))
        .endo("Y", 2, Mechanism::new(|v, u| v[1] ^ v[2] ^ u[2]))
}

/// Recanting witness (Avin, Shpitser, Pearl 2005, "Identifiability of
/// path-specific effects"): not identified even with no selection node and with
/// every observational and interventional law supplied.
///
/// Derivation. Graph `X -> W -> Z -> Y`, `W -> Y`. The path-specific quantity
/// `Y_{x=1, Z_{x=0}}` has `W` on a path that is active in the `x=0` world
/// (`W -> Z -> Y`) and on a path active in the `x=1` world (`W -> Y`): `W` is a
/// recanting witness, so the cross-world joint `(W_{x=1}, W_{x=0})` is needed
/// and no do-law contains it. Witness: model A has `W = X xor U` (so
/// `W_1 xor W_0 = 1`), model B has `W = U` (so `W_1 = W_0`). Both give `W`
/// uniform and independent of everything it is allowed to depend on under every
/// do-regime, hence identical observational and all interventional joints, but
/// `P(Y_{x=1, Z_{x=0}} = 1)` is 3/4 in A and 1/4 in B.
pub fn recanting_witness_without_selection() -> Case {
    let a = recanting_witness_population(false);
    let b = recanting_witness_population(true);
    let populations = vec![Population::new("pop", a.clone())];
    let alternative = vec![Population::new("pop", b)];
    let evidence: Vec<EvidenceSpec> = all_regimes(&a)
        .into_iter()
        .map(|regime| {
            let r: Vec<(&str, usize)> = regime.iter().map(|(n, l)| (n.as_str(), *l)).collect();
            EvidenceSpec::new("pop", &r, &["X", "W", "Z", "Y"])
        })
        .collect();
    let query = TargetQuery::new(
        "pop",
        vec![Atom::new(
            CfVar::nested(
                "Y",
                vec![
                    ("X", Setting::Level(1)),
                    ("Z", Setting::cf(CfVar::under("Z", &[("X", 0)]))),
                ],
            ),
            1,
        )],
        vec![],
    );
    Case {
        name: "recanting_witness_without_selection",
        derivation: "Avin-Shpitser-Pearl 2005 recanting witness W for Y_{x1,Z_{x0}}; two models \
                     with identical observational and all interventional laws differ on it",
        graph: CaseGraph {
            observed: vec!["X", "W", "Z", "Y"],
            directed: vec![("X", "W"), ("W", "Z"), ("W", "Y"), ("Z", "Y")],
            bidirected: vec![],
            selection: vec![],
        },
        target_population: "pop",
        evidence,
        targets: vec![target(
            "Y_{x1,Z_{x0}}",
            &populations,
            query,
            ExpectedClass::NotIdentifiable,
        )],
        populations,
        alternative: Some(alternative),
    }
}

// ---------------------------------------------------------------------------
// (6) selection directly on the outcome mechanism
// ---------------------------------------------------------------------------

fn outcome_selection_population(uy: f64) -> FiniteScm {
    // X = U xor Ux; Y = X xor (U and Uy). U confounds X, Y. Uy is Y-private.
    FiniteScm::new()
        .exo("U", bern(0.5))
        .exo("Ux", bern(0.25))
        .exo("Uy", bern(uy))
        .endo("X", 2, Mechanism::new(|_, u| u[0] ^ u[1]))
        .endo("Y", 2, Mechanism::new(|v, u| v[0] ^ (u[0] & u[2])))
}

/// Selection on the outcome mechanism with a latent confounder: not
/// transportable. Source observational and experimental laws and the target
/// observational law are supplied, no target experiment.
///
/// Derivation. Graph `X -> Y`, `X <-> Y`, `S -> Y`. `P*(y | do(x))` is not
/// identified from target observations (confounding), and the source
/// experiment cannot be borrowed because `S` is a parent of `Y`
/// (`Y` is not d-separated from `S` in `G_{\bar X}` given `X`). Witness: both
/// families share the source model; in the target, model A has
/// `Y = X xor (U and Uy)` (`Uy` law 1/4), model B has the same equation with `X`
/// replaced by its natural value so `do(X)` has a different reach. The target
/// observational law is identical; `P*(Y_{x=1} = 1)` is 7/8 in A and 7/16 in B.
pub fn selection_on_outcome_mechanism() -> Case {
    let source = outcome_selection_population(0.5);
    let tgt_a = outcome_selection_population(0.25);
    let tgt_b = tgt_a.blind_to_natural("Y", "X");
    let populations = pops(source.clone(), tgt_a);
    let alternative = pops(source, tgt_b);
    let query = TargetQuery::new(
        "target",
        vec![Atom::new(CfVar::under("Y", &[("X", 1)]), 1)],
        vec![],
    );
    Case {
        name: "selection_on_outcome_mechanism",
        derivation: "X->Y, X<->Y, S->Y; target do-law not identified from target observations and \
                     the source experiment cannot transport; two-model witness",
        graph: CaseGraph {
            observed: vec!["X", "Y"],
            directed: vec![("X", "Y")],
            bidirected: vec![("X", "Y")],
            selection: vec![("S_Y", "Y")],
        },
        target_population: "target",
        evidence: vec![
            EvidenceSpec::new("source", &[], &["X", "Y"]),
            EvidenceSpec::new("source", &[("X", 0)], &["X", "Y"]),
            EvidenceSpec::new("source", &[("X", 1)], &["X", "Y"]),
            EvidenceSpec::new("target", &[], &["X", "Y"]),
        ],
        targets: vec![target(
            "Y_{x1}=1",
            &populations,
            query,
            ExpectedClass::NotTransportable,
        )],
        populations,
        alternative: Some(alternative),
    }
}

// ---------------------------------------------------------------------------
// (7) zero-support conditioning event
// ---------------------------------------------------------------------------

fn zero_support_population(px: f64) -> FiniteScm {
    // X = Ux (selected); Y = X xor Uy with Uy ~ Bern(1/4).
    FiniteScm::new()
        .exo("Ux", bern(px))
        .exo("Uy", bern(0.25))
        .endo("X", 2, Mechanism::new(|_, u| u[0]))
        .endo("Y", 2, Mechanism::new(|v, u| v[0] ^ u[1]))
}

/// Conditioning evidence that has probability zero in the target.
///
/// Derivation. `X -> Y`, `S -> X`. In the source `P(X=1) = 1/2` and
/// `P(Y_{x=0}=1 | X=1) = P(Uy=1) = 1/4`. In the target the selection node
/// drives `P*(X=1) = 0`, so `P*(Y_{x=0}=1 | X=1)` conditions on a null event: it
/// is undefined, and the (positivity-dependent) transport formula
/// `P*(y_x | x') = P(y_x | x')` would silently read an unsupported cell. The
/// correct output is a typed refusal, not a number.
pub fn zero_support_conditioning_event() -> Case {
    let populations = pops(zero_support_population(0.5), zero_support_population(0.0));
    let ett = |pop: &str| {
        TargetQuery::new(
            pop,
            vec![Atom::new(CfVar::under("Y", &[("X", 0)]), 1)],
            vec![Atom::factual("X", 1)],
        )
    };
    Case {
        name: "zero_support_conditioning_event",
        derivation: "ETT conditioned on X=1 where the target law gives X=1 probability 0; \
                     defined in the source (1/4), undefined in the target",
        graph: CaseGraph {
            observed: vec!["X", "Y"],
            directed: vec![("X", "Y")],
            bidirected: vec![],
            selection: vec![("S_X", "X")],
        },
        target_population: "target",
        evidence: vec![
            EvidenceSpec::new("source", &[], &["X", "Y"]),
            EvidenceSpec::new("source", &[("X", 0)], &["X", "Y"]),
            EvidenceSpec::new("target", &[], &["X"]),
        ],
        targets: vec![
            target(
                "ETT in source (defined)",
                &populations,
                ett("source"),
                ExpectedClass::Identified,
            ),
            target(
                "ETT in target (null evidence)",
                &populations,
                ett("target"),
                ExpectedClass::UndefinedZeroSupport,
            ),
        ],
        populations,
        alternative: None,
    }
}

// ---------------------------------------------------------------------------
// (8) missing interventional evidence
// ---------------------------------------------------------------------------

/// The covariate-shift model of the two-regime cases, but with the source
/// experiment withheld.
///
/// Derivation. With `X <-> Y` and no `do(X)` evidence, `P*(Y_{x=1}=1)` is
/// determined by neither observational law. Witness: model A is the true model;
/// model B makes `Y` read the natural value of `X`, so `do(X)` has no direct
/// reach on `Y`. They share every observational law (source and target) hence
/// every supplied table, yet `P*(Y_{x=1}=1)` is 19/32 in A and 23/64 in B.
/// With the source `do(X)` table supplied the same target is identified (see
/// `joint_counterfactual_two_source_regimes`).
pub fn missing_interventional_evidence() -> Case {
    let (s, t) = source_and_target_z();
    let alternative = pops(
        s.blind_to_natural("Y", "X"),
        t.blind_to_natural("Y", "X"),
    );
    let populations = pops(s, t);
    let query = TargetQuery::new(
        "target",
        vec![Atom::new(CfVar::under("Y", &[("X", 1)]), 1)],
        vec![],
    );
    Case {
        name: "missing_interventional_evidence",
        derivation: "X<->Y with only observational evidence in both populations; two models \
                     with equal observational laws differ on P*(y|do(x))",
        graph: confounded_graph(),
        target_population: "target",
        evidence: vec![
            EvidenceSpec::new("source", &[], &["Z", "X", "Y"]),
            EvidenceSpec::new("target", &[], &["Z"]),
        ],
        targets: vec![target(
            "Y_{x1}=1",
            &populations,
            query,
            ExpectedClass::EvidenceInsufficient,
        )],
        populations,
        alternative: Some(alternative),
    }
}

/// Every named case, in a fixed order.
pub fn all_cases() -> Vec<Case> {
    vec![
        nonlinear_mediation_covariate_shift(),
        joint_counterfactual_two_source_regimes(),
        effect_of_treatment_on_treated_with_selection(),
        transportable_latent_district(),
        recanting_witness_without_selection(),
        selection_on_outcome_mechanism(),
        zero_support_conditioning_event(),
        missing_interventional_evidence(),
    ]
}
