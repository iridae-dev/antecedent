//! Observation-conditioned queries, one test per entry point.
//!
//! Confounded finite model (all variables binary, levels "0" / "1"):
//!   Z ~ Bernoulli(0.5)
//!   P(S = 1 | Z = 0) = 0.9, P(S = 1 | Z = 1) = 0.1
//!   P(Y = 1 | S, Z): (0,0) 0.1, (0,1) 0.5, (1,0) 0.3, (1,1) 0.9   (keyed (S, Z))
//! so that, by hand,
//!   P(S = 1) = 0.5
//!   P(Y = 1, S = 1) = 0.5*0.9*0.3 + 0.5*0.1*0.9 = 0.18 -> P(Y=1 | S=1) = 0.36
//!   P(Y = 1, S = 0) = 0.5*0.1*0.1 + 0.5*0.9*0.5 = 0.23 -> P(Y=1 | S=0) = 0.46
//!   P(Y = 1 | S = 1, Z = 0) = 0.3, P(S = 1, Z = 0) = 0.45
//!   P(Y = 1 | do(S=1)) = 0.5*0.3 + 0.5*0.9 = 0.6
//!   P(Y = 1 | do(S=0)) = 0.5*0.1 + 0.5*0.5 = 0.3
//!   P(Y = 1 | do(Z=1)) = 0.1*0.9 + 0.9*0.5 = 0.54

use antecedent_core::ExecutionContext;
use antecedent_model::compose::{
    ComposedFinite, ComposedModelSpec, Domain, MechanismContext, MechanismDescriptor,
    MechanismForm, MechanismOrigin, MechanismProvenance, ParentBinding, PosteriorDependence,
    TabularRow,
};
use antecedent_model::observation_query::{
    ObservationQueryError, condition, conditioning_and_intervention, intervene,
};

const TOL: f64 = 1e-12;

fn near(a: f64, b: f64) {
    assert!((a - b).abs() <= TOL, "{a} vs {b}");
}

fn level(levels: &[(String, f64)], name: &str) -> f64 {
    levels.iter().find(|(l, _)| l == name).map(|(_, p)| *p).unwrap()
}

fn table(
    var: &str,
    parents: &[(&str, &[&str])],
    levels: &[&str],
    rows: &[(&[&str], &[f64])],
) -> MechanismDescriptor {
    let level_names: Vec<String> = levels.iter().map(|s| (*s).to_owned()).collect();
    MechanismDescriptor {
        variable: var.into(),
        form: MechanismForm::Tabular {
            levels: level_names.clone(),
            rows: rows
                .iter()
                .map(|(config, probs)| TabularRow {
                    parent_levels: parents
                        .iter()
                        .zip(config.iter())
                        .map(|((p, _), l)| ((*p).to_owned(), (*l).to_owned()))
                        .collect(),
                    probabilities: level_names.iter().cloned().zip(probs.iter().copied()).collect(),
                })
                .collect(),
        },
        parents: parents
            .iter()
            .map(|(p, lv)| ParentBinding {
                variable: (*p).to_owned(),
                units: "category".into(),
                transform: "identity".into(),
                domain: Domain::Levels(lv.iter().map(|s| (*s).to_owned()).collect()),
            })
            .collect(),
        conditioning_set: parents.iter().map(|(p, _)| (*p).to_owned()).collect(),
        output_units: "category".into(),
        output_transform: "identity".into(),
        output_support: Domain::Levels(level_names),
        context: MechanismContext { population: "pop".into(), regime: "observational".into() },
        provenance: MechanismProvenance {
            origin: MechanismOrigin::Declared,
            artifact_id: format!("declared:{var}"),
            evidence_id: format!("assumed:{var}"),
        },
        latents: Vec::new(),
        uncertainty: None,
    }
}

fn spec_of(mechanisms: Vec<MechanismDescriptor>) -> ComposedModelSpec {
    let mut edges = Vec::new();
    for m in &mechanisms {
        for p in &m.parents {
            edges.push((p.variable.clone(), m.variable.clone()));
        }
    }
    ComposedModelSpec {
        target: MechanismContext { population: "pop".into(), regime: "observational".into() },
        mechanisms,
        edges,
        invariances: Vec::new(),
        selection_diagrams: Vec::new(),
        dependence: PosteriorDependence::Unknown,
        admit_declared_mechanisms: true,
    }
}

const B: &[&str] = &["0", "1"];

fn confounded() -> ComposedFinite {
    ComposedFinite::compile(&spec_of(vec![
        table("Z", &[], B, &[(&[], &[0.5, 0.5])]),
        table("S", &[("Z", B)], B, &[(&["0"], &[0.1, 0.9]), (&["1"], &[0.9, 0.1])]),
        table(
            "Y",
            &[("S", B), ("Z", B)],
            B,
            &[
                (&["0", "0"], &[0.9, 0.1]),
                (&["0", "1"], &[0.5, 0.5]),
                (&["1", "0"], &[0.7, 0.3]),
                (&["1", "1"], &[0.1, 0.9]),
            ],
        ),
    ]))
    .unwrap()
}

#[test]
fn conditioning_reads_the_observational_law_given_the_recorded_values() {
    let ctx = ExecutionContext::for_tests(1);
    let law = confounded().law(&[], &ctx).unwrap();
    let on_one = condition(&law, "Y", &[("S", "1")], &ctx).unwrap();
    near(on_one.evidence_probability, 0.5);
    near(level(&on_one.levels, "1"), 0.36);
    near(level(&on_one.levels, "0"), 0.64);
    let on_zero = condition(&law, "Y", &[("S", "0")], &ctx).unwrap();
    near(level(&on_zero.levels, "1"), 0.46);
    // Two recorded values read the table cell directly.
    let cell = condition(&law, "Y", &[("S", "1"), ("Z", "0")], &ctx).unwrap();
    near(cell.evidence_probability, 0.45);
    near(level(&cell.levels, "1"), 0.3);
    // The outcome cannot be its own evidence, and evidence must name the model.
    assert_eq!(
        condition(&law, "Y", &[("Y", "1")], &ctx).unwrap_err(),
        ObservationQueryError::OutcomeObserved("Y".into())
    );
    assert!(matches!(
        condition(&law, "Y", &[("Q", "1")], &ctx).unwrap_err(),
        ObservationQueryError::UnknownVariable(_)
    ));
}

#[test]
fn intervention_reads_the_truncated_factorization_not_the_conditional() {
    let ctx = ExecutionContext::for_tests(1);
    let model = confounded();
    near(level(&intervene(&model, "Y", &[("S", "1")], &ctx).unwrap(), "1"), 0.6);
    near(level(&intervene(&model, "Y", &[("S", "0")], &ctx).unwrap(), "1"), 0.3);
    // Z has no parent, so setting it equals observing it.
    near(level(&intervene(&model, "Y", &[("Z", "1")], &ctx).unwrap(), "1"), 0.54);
    let law = model.law(&[], &ctx).unwrap();
    near(level(&condition(&law, "Y", &[("Z", "1")], &ctx).unwrap().levels, "1"), 0.54);
    // The conditional of S = 1 is a different number from the intervention.
    assert!(
        (level(&condition(&law, "Y", &[("S", "1")], &ctx).unwrap().levels, "1") - 0.6).abs() > 0.2
    );
    assert!(matches!(
        intervene(&model, "Y", &[("Q", "1")], &ctx).unwrap_err(),
        ObservationQueryError::UnknownVariable(_)
    ));
    assert!(matches!(
        intervene(&model, "Y", &[("S", "7")], &ctx).unwrap_err(),
        ObservationQueryError::UnknownLevel { .. }
    ));
}

#[test]
fn the_paired_query_returns_both_laws_of_one_model_and_their_largest_gap() {
    let ctx = ExecutionContext::for_tests(1);
    let model = confounded();
    let one = conditioning_and_intervention(&model, "Y", &[("S", "1")], &ctx).unwrap();
    near(level(&one.conditioning.levels, "1"), 0.36);
    near(level(&one.intervention, "1"), 0.6);
    near(one.max_abs_difference, 0.24);
    let zero = conditioning_and_intervention(&model, "Y", &[("S", "0")], &ctx).unwrap();
    near(level(&zero.conditioning.levels, "1"), 0.46);
    near(level(&zero.intervention, "1"), 0.3);
    near(zero.max_abs_difference, 0.16);
    // Each half equals the stand-alone entry point on the same model.
    let alone = intervene(&model, "Y", &[("S", "1")], &ctx).unwrap();
    assert_eq!(one.intervention, alone);
    // An unknown name fails the pair as it fails each half.
    assert!(matches!(
        conditioning_and_intervention(&model, "Y", &[("Q", "1")], &ctx).unwrap_err(),
        ObservationQueryError::UnknownVariable(_)
    ));
}
