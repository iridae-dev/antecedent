//! Independent replay of the mechanism-composition artifact, and the semantic
//! mutations a consumer must refuse.
//!
//! Oracle (hand computed; see the model-crate suite for the derivation):
//!   X = 1 + 2 `e_X` ; M = 0.5 + 2 X + `e_M` ; Y = -1 + 0.5 X + 3 M + 0.5 `e_Y`
//!   do(X=2): E[Y] = 13.5, Var[Y] = 9.25
//!   finite chain, do(X=1): P(Y=1) = 0.72

use std::collections::BTreeMap;

use antecedent_contract::{
    AdmissionBasis, AdmissionCoordinate, DependenceDeclaration, MechanismCompositionDeclaration,
    MechanismDeclaration, MechanismOriginKind, QuantityRole, ScientificQuantity,
    TransportAdmission, check_mechanism_composition,
};
use antecedent_core::ExecutionContext;
use antecedent_io_core::{
    CompositionQueryWire, CompositionResultWire, MechanismCompositionArtifactError,
    MechanismCompositionArtifactWire, MechanismNumbers, lower_plan, mechanism_composition_identity,
};
use antecedent_model::compose::{
    ComposedModelSpec, Domain, MechanismContext, MechanismDescriptor, MechanismFamily,
    MechanismForm, MechanismOrigin, MechanismProvenance, MechanismUncertainty, ParameterTerm,
    ParentBinding, PosteriorDependence, TabularRow, UncertaintyMeaning,
};

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(11)
}

fn lg(var: &str, intercept: f64, sigma: f64, parents: &[(&str, f64)]) -> MechanismDescriptor {
    MechanismDescriptor {
        variable: var.into(),
        form: MechanismForm::LinearGaussian {
            intercept,
            sigma,
            coefficients: parents.iter().map(|(p, c)| ((*p).to_owned(), *c)).collect(),
        },
        parents: parents
            .iter()
            .map(|(p, _)| ParentBinding {
                variable: (*p).to_owned(),
                units: "u".into(),
                transform: "identity".into(),
                domain: Domain::Real,
            })
            .collect(),
        conditioning_set: parents.iter().map(|(p, _)| (*p).to_owned()).collect(),
        output_units: "u".into(),
        output_transform: "identity".into(),
        output_support: Domain::Real,
        context: MechanismContext { population: "pop".into(), regime: "observational".into() },
        provenance: MechanismProvenance {
            origin: MechanismOrigin::Fitted,
            artifact_id: format!("fit:{var}"),
            evidence_id: format!("data:{var}"),
        },
        latents: Vec::new(),
        uncertainty: None,
    }
}

fn spec(mechanisms: Vec<MechanismDescriptor>) -> ComposedModelSpec {
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
        admit_declared_mechanisms: false,
    }
}

fn chain() -> Vec<MechanismDescriptor> {
    vec![
        lg("X", 1.0, 2.0, &[]),
        lg("M", 0.5, 1.0, &[("X", 2.0)]),
        lg("Y", -1.0, 0.5, &[("X", 0.5), ("M", 3.0)]),
    ]
}

fn law_query() -> CompositionQueryWire {
    CompositionQueryWire::GaussianInterventionLaw {
        interventions: vec![("X".into(), 2.0)],
        outcome: "Y".into(),
    }
}

fn reseal(wire: &mut MechanismCompositionArtifactWire) {
    wire.premises_digest =
        mechanism_composition_identity(&wire.premises, &wire.query).expect("digest");
}

#[test]
fn a_sealed_artifact_replays_to_the_closed_form_and_round_trips() {
    let wire = MechanismCompositionArtifactWire::seal(&spec(chain()), law_query(), &ctx()).unwrap();
    let CompositionResultWire::GaussianLaw { mean_bits, variance_bits } = &wire.result else {
        panic!("wrong result kind");
    };
    assert!((f64::from_bits(*mean_bits) - 13.5).abs() < 1e-12);
    assert!((f64::from_bits(*variance_bits) - 9.25).abs() < 1e-12);

    let bytes = wire.export().unwrap();
    let decoded = MechanismCompositionArtifactWire::decode(&bytes).unwrap();
    assert_eq!(decoded, wire);
    let consumed = decoded.consume(Some(&wire.premises_digest), &ctx()).unwrap();
    assert_eq!(consumed.family, MechanismFamily::LinearGaussian);
    assert_eq!(consumed.result, wire.result);
    assert_eq!(consumed.premises_digest, wire.premises_digest);
}

#[test]
fn listing_order_does_not_change_the_identity() {
    let forward =
        MechanismCompositionArtifactWire::seal(&spec(chain()), law_query(), &ctx()).unwrap();
    let mut shuffled = chain();
    shuffled.reverse();
    for m in &mut shuffled {
        m.parents.reverse();
        m.conditioning_set.reverse();
        if let MechanismForm::LinearGaussian { coefficients, .. } = &mut m.form {
            coefficients.reverse();
        }
    }
    let mut shuffled_spec = spec(shuffled);
    shuffled_spec.edges.reverse();
    let backward =
        MechanismCompositionArtifactWire::seal(&shuffled_spec, law_query(), &ctx()).unwrap();
    assert_eq!(forward.premises_digest, backward.premises_digest);
    assert_eq!(forward.export().unwrap(), backward.export().unwrap());
}

#[test]
fn semantic_mutations_are_refused() {
    let wire = MechanismCompositionArtifactWire::seal(&spec(chain()), law_query(), &ctx()).unwrap();

    // An edit that is not resealed.
    let mut edited = wire.clone();
    if let antecedent_io_core::mechanism_composition_artifact::MechanismFormWire::LinearGaussian {
        slot: antecedent_io_core::MechanismSlotWire::LinearGaussian { intercept, .. },
    } = &mut edited.premises.mechanisms[0].form
    {
        *intercept += 1.0;
    }
    assert_eq!(
        edited.consume(None, &ctx()).unwrap_err(),
        MechanismCompositionArtifactError::PremisesMismatch
    );

    // A resealed edit that changes the answer: the stored result no longer replays.
    reseal(&mut edited);
    assert_eq!(
        edited.consume(None, &ctx()).unwrap_err(),
        MechanismCompositionArtifactError::ResultMismatch
    );

    // A forged result under honest premises.
    let mut forged = wire.clone();
    forged.result = CompositionResultWire::GaussianLaw {
        mean_bits: 13.5_f64.to_bits(),
        variance_bits: 9.0_f64.to_bits(),
    };
    assert_eq!(
        forged.consume(None, &ctx()).unwrap_err(),
        MechanismCompositionArtifactError::ResultMismatch
    );

    // A caller that holds a different identity.
    assert_eq!(
        wire.consume(Some("0000"), &ctx()).unwrap_err(),
        MechanismCompositionArtifactError::PremisesMismatch
    );

    // A resealed unit edit makes the stored mechanisms incoherent: a structured
    // refusal, not a number.
    let mut unit = wire.clone();
    for m in &mut unit.premises.mechanisms {
        if m.variable == "Y" {
            for p in &mut m.parents {
                if p.variable == "M" {
                    p.units = "kPa".into();
                }
            }
        }
    }
    reseal(&mut unit);
    match unit.consume(None, &ctx()).unwrap_err() {
        MechanismCompositionArtifactError::Incoherent { refusals } => {
            assert_eq!(refusals.len(), 1);
            assert_eq!(refusals[0].code, "quantity_semantics_mismatch");
            assert_eq!(refusals[0].detail, "mechanism_composition.units_mismatch");
            assert_eq!(refusals[0].offending.as_deref(), Some("M->Y"));
            refusals[0].validate().unwrap();
        }
        other => panic!("{other:?}"),
    }

    // A resealed removal of a mechanism leaves a parent without a mechanism.
    let mut dropped = wire.clone();
    dropped.premises.mechanisms.retain(|m| m.variable != "M");
    reseal(&mut dropped);
    match dropped.consume(None, &ctx()).unwrap_err() {
        MechanismCompositionArtifactError::Incoherent { refusals } => {
            assert!(
                refusals
                    .iter()
                    .any(|r| r.detail == "mechanism_composition.identity_parent_without_mechanism")
            );
        }
        other => panic!("{other:?}"),
    }

    // Another version is refused before the payload is read.
    let mut future = wire.clone();
    future.version = 2;
    let bytes = antecedent_io_core::to_cbor(&future).unwrap();
    assert_eq!(
        MechanismCompositionArtifactWire::decode(&bytes).unwrap_err(),
        MechanismCompositionArtifactError::UnsupportedVersion { version: 2 }
    );
    let mut foreign = wire;
    foreign.required_features = vec!["other_feature_v1".into()];
    let bytes = antecedent_io_core::to_cbor(&foreign).unwrap();
    assert_eq!(
        MechanismCompositionArtifactWire::decode(&bytes).unwrap_err(),
        MechanismCompositionArtifactError::UnsupportedSemantics("required features")
    );
}

fn uncertain_chain(dependence: PosteriorDependence) -> ComposedModelSpec {
    let mut mechanisms = chain();
    let unc = |terms: Vec<ParameterTerm>, cov: Vec<f64>| MechanismUncertainty {
        meaning: UncertaintyMeaning::SamplingDistribution,
        parameters: terms,
        covariance: cov,
    };
    mechanisms[1].uncertainty = Some(unc(
        vec![ParameterTerm::Intercept, ParameterTerm::Coefficient("X".into())],
        vec![0.04, -0.01, -0.01, 0.01],
    ));
    mechanisms[0].uncertainty = Some(unc(vec![ParameterTerm::Intercept], vec![0.25]));
    mechanisms[2].uncertainty = Some(unc(
        vec![
            ParameterTerm::Intercept,
            ParameterTerm::Coefficient("M".into()),
            ParameterTerm::Coefficient("X".into()),
        ],
        vec![0.09, 0.0, 0.0, 0.0, 0.0025, 0.0, 0.0, 0.0, 0.01],
    ));
    let mut s = spec(mechanisms);
    s.dependence = dependence;
    s
}

fn contrast_query() -> CompositionQueryWire {
    CompositionQueryWire::GaussianMeanContrast {
        outcome: "Y".into(),
        arms: vec![(1.0, vec![("X".into(), 2.0)])],
    }
}

#[test]
fn uncertainty_artifact_needs_a_declared_dependence_and_unknown_is_a_joint_law_refusal() {
    // Independent by declaration: variance 0.540625 (hand computed).
    let independent = uncertain_chain(PosteriorDependence::IndependentByDesign {
        justification: "disjoint trials".into(),
    });
    let wire =
        MechanismCompositionArtifactWire::seal(&independent, contrast_query(), &ctx()).unwrap();
    let CompositionResultWire::GaussianFunctional { point_bits, variance_bits, dependence, method } =
        &wire.result
    else {
        panic!("wrong result kind");
    };
    assert!((f64::from_bits(*point_bits) - 13.5).abs() < 1e-12);
    assert!((f64::from_bits(*variance_bits) - 0.540_625).abs() < 1e-12);
    assert_eq!(dependence, "declared_independent_by_design");
    assert_eq!(method, "first_order_delta_method");
    assert!(wire.consume(None, &ctx()).is_ok());

    // Flipping the declared dependence is a different premise and a different identity.
    let mut flipped = wire.clone();
    flipped.premises.dependence =
        antecedent_io_core::mechanism_composition_artifact::DependenceWire::Unknown;
    reseal(&mut flipped);
    assert_ne!(flipped.premises_digest, wire.premises_digest);
    match flipped.consume(None, &ctx()).unwrap_err() {
        MechanismCompositionArtifactError::Incoherent { refusals } => {
            assert_eq!(refusals.len(), 1);
            assert_eq!(refusals[0].code, "joint_law_required");
            assert_eq!(refusals[0].detail, "mechanism_composition.posterior_dependence_unknown");
            assert!(refusals[0].remedy.is_some());
        }
        other => panic!("{other:?}"),
    }

    // Unknown dependence cannot even be sealed into an uncertainty artifact,
    // yet the point query on the same composition seals.
    let unknown = uncertain_chain(PosteriorDependence::Unknown);
    assert!(matches!(
        MechanismCompositionArtifactWire::seal(&unknown, contrast_query(), &ctx()),
        Err(MechanismCompositionArtifactError::Incoherent { .. })
    ));
    assert!(MechanismCompositionArtifactWire::seal(&unknown, law_query(), &ctx()).is_ok());
}

fn table(var: &str, parents: &[(&str, &[&str])], rows: &[(&[&str], f64)]) -> MechanismDescriptor {
    // Binary variable; `rows` gives (parent configuration, P(level "1")).
    let levels = vec!["0".to_owned(), "1".to_owned()];
    MechanismDescriptor {
        variable: var.into(),
        form: MechanismForm::Tabular {
            levels: levels.clone(),
            rows: rows
                .iter()
                .map(|(config, p1)| TabularRow {
                    parent_levels: parents
                        .iter()
                        .zip(config.iter())
                        .map(|((p, _), l)| ((*p).to_owned(), (*l).to_owned()))
                        .collect(),
                    probabilities: vec![("0".into(), 1.0 - p1), ("1".into(), *p1)],
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
        output_support: Domain::Levels(levels),
        context: MechanismContext { population: "pop".into(), regime: "observational".into() },
        provenance: MechanismProvenance {
            origin: MechanismOrigin::Fitted,
            artifact_id: format!("fit:{var}"),
            evidence_id: format!("data:{var}"),
        },
        latents: Vec::new(),
        uncertainty: None,
    }
}

#[test]
fn finite_composition_artifact_matches_the_enumerated_truth() {
    let b: &[&str] = &["0", "1"];
    let mechanisms = vec![
        table("X", &[], &[(&[], 0.4)]),
        table("M", &[("X", b)], &[(&["0"], 0.2), (&["1"], 0.7)]),
        // parents (M, X): P(Y=1) = 0.1, 0.5 (M=1,X=0), 0.3 (M=0,X=1), 0.9
        table(
            "Y",
            &[("M", b), ("X", b)],
            &[(&["0", "0"], 0.1), (&["1", "0"], 0.5), (&["0", "1"], 0.3), (&["1", "1"], 0.9)],
        ),
    ];
    let query = CompositionQueryWire::FiniteInterventionDistribution {
        interventions: vec![("X".into(), "1".into())],
        outcome: "Y".into(),
    };
    let wire = MechanismCompositionArtifactWire::seal(&spec(mechanisms), query, &ctx()).unwrap();
    let CompositionResultWire::FiniteDistribution { probabilities } = &wire.result else {
        panic!("wrong result kind");
    };
    // do(X=1): P(Y=1) = 0.3*0.3 + 0.7*0.9 = 0.72
    assert_eq!(probabilities[0].0, "0");
    assert!((f64::from_bits(probabilities[0].1) - 0.28).abs() < 1e-12);
    assert!((f64::from_bits(probabilities[1].1) - 0.72).abs() < 1e-12);
    let consumed = wire.consume(None, &ctx()).unwrap();
    assert_eq!(consumed.family, MechanismFamily::Tabular);
    // The same numbers decode under a fresh consumer from bytes.
    let again = MechanismCompositionArtifactWire::decode(&wire.export().unwrap()).unwrap();
    assert!(again.consume(None, &ctx()).is_ok());
}

fn quantity(variable: &str, role: QuantityRole, population: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: variable.into(),
        variable_name: variable.into(),
        role,
        units: "u".into(),
        population_id: population.into(),
        regime_id: "observational".into(),
        horizon: 0,
        functional_id: "realization".into(),
        conditioning: Vec::new(),
        transform_id: "identity".into(),
    }
}

fn declaration(parent_population_for_m: &str) -> MechanismCompositionDeclaration {
    let m = |v: &str, role: QuantityRole, pop: &str, parents: Vec<ScientificQuantity>| {
        MechanismDeclaration {
            conditioning_variable_ids: parents.iter().map(|p| p.variable_id.clone()).collect(),
            output: quantity(v, role, pop),
            parents,
            origin: MechanismOriginKind::Fitted,
            artifact_id: format!("fit:{v}"),
            evidence_id: format!("data:{v}"),
            latent_causes: Vec::new(),
        }
    };
    MechanismCompositionDeclaration {
        target_population_id: "pop".into(),
        target_regime_id: "observational".into(),
        mechanisms: vec![
            m("X", QuantityRole::Treatment, "pop", vec![]),
            m(
                "M",
                QuantityRole::Mediator,
                parent_population_for_m,
                vec![quantity("X", QuantityRole::Treatment, parent_population_for_m)],
            ),
            m(
                "Y",
                QuantityRole::Outcome,
                "pop",
                vec![
                    quantity("X", QuantityRole::Treatment, "pop"),
                    quantity("M", QuantityRole::Mediator, "pop"),
                ],
            ),
        ],
        edges: vec![("X".into(), "M".into()), ("X".into(), "Y".into()), ("M".into(), "Y".into())],
        admissions: Vec::new(),
        dependence: DependenceDeclaration::Unknown,
        admit_declared_mechanisms: false,
    }
}

fn numbers() -> BTreeMap<String, MechanismNumbers> {
    let mut out = BTreeMap::new();
    for d in chain() {
        out.insert(
            d.variable.clone(),
            MechanismNumbers {
                form: d.form.clone(),
                output_support: Domain::Real,
                parent_domains: d
                    .parents
                    .iter()
                    .map(|p| (p.variable.clone(), Domain::Real))
                    .collect(),
                uncertainty: None,
                latent_loadings: Vec::new(),
            },
        );
    }
    out
}

#[test]
fn a_semantic_plan_binds_to_numbers_and_a_foreign_population_needs_its_admission() {
    // Coherent semantic declaration -> plan -> numbers -> the same answer as the
    // hand-built specification.
    let plan = check_mechanism_composition(&declaration("pop")).unwrap();
    let lowered = lower_plan(&plan, &numbers(), PosteriorDependence::Unknown).unwrap();
    let from_plan = MechanismCompositionArtifactWire::seal(&lowered, law_query(), &ctx()).unwrap();
    let direct =
        MechanismCompositionArtifactWire::seal(&spec(chain()), law_query(), &ctx()).unwrap();
    assert_eq!(from_plan.result, direct.result);
    assert_eq!(from_plan.premises_digest, direct.premises_digest);

    // The dependence law must be of the kind the plan declares.
    let err = lower_plan(
        &plan,
        &numbers(),
        PosteriorDependence::IndependentByDesign { justification: "x".into() },
    )
    .unwrap_err();
    assert!(matches!(err, MechanismCompositionArtifactError::Malformed(_)));

    // Numbers must exist for every planned mechanism.
    let mut missing = numbers();
    missing.remove("M");
    assert!(lower_plan(&plan, &missing, PosteriorDependence::Unknown).is_err());

    // M fitted in another population: refused without a declaration ...
    let refusals = check_mechanism_composition(&declaration("source")).unwrap_err();
    assert_eq!(refusals.len(), 1);
    assert_eq!(refusals[0].code, "quantity_semantics_mismatch");
    assert_eq!(refusals[0].detail, "mechanism_composition.population_undeclared_transport");
    // ... and admitted by a selection diagram that does not point into M.
    let mut admitted = declaration("source");
    admitted.admissions.push(TransportAdmission {
        variable_id: "M".into(),
        coordinate: AdmissionCoordinate::Population,
        source_id: "source".into(),
        target_id: "pop".into(),
        basis: AdmissionBasis::SelectionDiagram { targets: vec!["X".into()] },
    });
    let plan = check_mechanism_composition(&admitted).unwrap();
    let lowered = lower_plan(&plan, &numbers(), PosteriorDependence::Unknown).unwrap();
    assert_eq!(lowered.selection_diagrams.len(), 1);
    let sealed = MechanismCompositionArtifactWire::seal(&lowered, law_query(), &ctx()).unwrap();
    // The admission is part of the premises: removing it, resealed, refuses.
    let mut stripped = sealed.clone();
    stripped.premises.selection_diagrams.clear();
    reseal(&mut stripped);
    match stripped.consume(None, &ctx()).unwrap_err() {
        MechanismCompositionArtifactError::Incoherent { refusals } => {
            assert!(
                refusals
                    .iter()
                    .any(|r| r.detail == "mechanism_composition.population_undeclared_transport")
            );
        }
        other => panic!("{other:?}"),
    }
}
