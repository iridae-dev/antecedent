//! F9 evidence obligations: construction from real failed factor contracts, the
//! joint-regime rule, order-independent identity, and the rule that a study
//! never establishes an assumption.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::assumption::{AssumptionSource, AssumptionStatus};
use antecedent_core::{
    DistributionAvailability, Environment, EvidenceCatalog, EvidenceKind, EvidenceObligation,
    EvidenceObligationKind, EvidenceObligationSpec, EvidenceOffer, EvidenceRegime, FactorNeed,
    ObligationKind, ObligationProvenance, ObligationRecord, ObligationRegime, ObligationScope,
    RegimeId, RegimeKind, VariableCoordinate, VariableDomain, VariableId,
    unresolved_assumption_obligations,
};

const X: u32 = 0;
const Y: u32 = 1;
const Z: u32 = 2;

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn vars(raw: &[u32]) -> Vec<VariableId> {
    raw.iter().copied().map(v).collect()
}

fn environment(name: &str) -> Environment {
    let coordinate =
        |raw| VariableCoordinate { variable: v(raw), domain: VariableDomain::Binary, unit: None };
    Environment::try_new(name, [coordinate(X), coordinate(Y), coordinate(Z)], []).unwrap()
}

fn regime(
    id: u32,
    kind: RegimeKind,
    interventions: &[u32],
    measured: &[u32],
    population: &str,
    distribution: DistributionAvailability,
) -> EvidenceRegime {
    EvidenceRegime::try_new(
        RegimeId::from_raw(id),
        kind,
        EvidenceKind::Available,
        vars(interventions),
        [],
        vars(measured),
        population,
        distribution,
    )
    .unwrap()
}

fn catalog(regimes: Vec<EvidenceRegime>) -> EvidenceCatalog {
    EvidenceCatalog::try_new([environment("source"), environment("target")], regimes, [], None)
        .unwrap()
}

fn provenance(step: &str) -> ObligationProvenance {
    ObligationProvenance {
        family: Arc::from("transport"),
        source: Arc::from("contract:test"),
        proof_step: Some(Arc::from(step)),
    }
}

/// The need must really be unmet by the catalog, then it becomes an obligation.
fn obligation_of(catalog: &EvidenceCatalog, need: &FactorNeed<'_>) -> EvidenceObligation {
    let unmet = catalog.unmet_factor_dependencies(&[(Arc::from("f"), *need)]);
    assert_eq!(unmet.len(), 1, "the need must be a real failure of the catalog");
    EvidenceObligation::from_unmet_factor(catalog, "f", need, provenance("leaf:3")).unwrap()
}

fn offer(
    population: &str,
    interventions: &[u32],
    measured: &[u32],
    joint: bool,
    samples: Option<u64>,
) -> EvidenceOffer {
    EvidenceOffer {
        quantities: std::collections::BTreeMap::new(),
        population: Arc::from(population),
        interventions: vars(interventions).into(),
        conditioned_on: Arc::from([]),
        measured: vars(measured).into(),
        joint,
        additional_samples: samples,
    }
}

fn spec(kind: EvidenceObligationKind) -> EvidenceObligationSpec {
    EvidenceObligationSpec {
        quantities: std::collections::BTreeMap::new(),
        kind,
        scope: ObligationScope::Factor,
        variables: vars(&[Y, Z]).into(),
        population: Some(Arc::from("source")),
        regime: ObligationRegime {
            interventions: vars(&[X]).into(),
            conditioned_on: Arc::from([]),
            joint: true,
        },
        reason: Arc::from("owed by contract"),
        required_slots: Arc::from([Arc::from("factor:f")]),
        min_additional_samples: None,
        provenance: provenance("leaf:3"),
    }
}

#[test]
fn f9_unmet_factors_become_kind_specific_obligations() {
    let marginals = DistributionAvailability::SeparateMarginals { variables: vars(&[Y, Z]).into() };
    let joint = DistributionAvailability::Joint;

    // No available experiment on do(X) in the source: an experiment is owed.
    let empty = catalog(vec![]);
    let need_y = FactorNeed {
        population: "source",
        variables: &vars(&[Y]),
        conditioned_on: &[],
        interventions: &vars(&[X]),
    };
    let intervene = obligation_of(&empty, &need_y);
    assert_eq!(intervene.kind, EvidenceObligationKind::Intervene);
    assert_eq!(intervene.population.as_deref(), Some("source"));
    assert_eq!(intervene.provenance.proof_step.as_deref(), Some("leaf:3"));
    assert_eq!(intervene.required_slots.as_ref(), [Arc::<str>::from("factor:f")]);
    assert!(!intervene.regime.joint);

    // No target law at all: the population must be observed.
    let need_target = FactorNeed {
        population: "target",
        variables: &vars(&[Z]),
        conditioned_on: &[],
        interventions: &[],
    };
    let observe = obligation_of(&empty, &need_target);
    assert_eq!(observe.kind, EvidenceObligationKind::ObservePopulation);

    // do(X) exists in the source but only measured Y: Z must be measured.
    let y_only =
        catalog(vec![regime(1, RegimeKind::Experimental, &[X], &[Y], "source", joint.clone())]);
    let need_z = FactorNeed {
        population: "source",
        variables: &vars(&[Z]),
        conditioned_on: &[],
        interventions: &vars(&[X]),
    };
    assert_eq!(obligation_of(&y_only, &need_z).kind, EvidenceObligationKind::Measure);

    // Two variables are read: only separate marginals exist, so a joint is owed.
    let separate =
        catalog(vec![regime(1, RegimeKind::Experimental, &[X], &[Y, Z], "source", marginals)]);
    let need_yz = FactorNeed {
        population: "source",
        variables: &vars(&[Y, Z]),
        conditioned_on: &[],
        interventions: &vars(&[X]),
    };
    let joint_ob = obligation_of(&separate, &need_yz);
    assert_eq!(joint_ob.kind, EvidenceObligationKind::ProvideJointLaw);
    assert!(joint_ob.regime.joint);

    // A conditional law P(Y | Z) under do(X) that the do(X) regime lacks.
    let need_cond = FactorNeed {
        population: "source",
        variables: &vars(&[Y]),
        conditioned_on: &vars(&[Z]),
        interventions: &vars(&[X]),
    };
    let cond = obligation_of(&y_only, &need_cond);
    assert_eq!(cond.kind, EvidenceObligationKind::ProvideConditionalLaw);
    assert_eq!(cond.regime.conditioned_on.as_ref(), [v(Z)]);
}

#[test]
fn f9_every_kind_constructs_with_its_stable_name() {
    for kind in EvidenceObligationKind::ALL {
        let mut s = spec(kind);
        match kind {
            EvidenceObligationKind::ObservePopulation
            | EvidenceObligationKind::ObserveEnvironment => {
                s.regime.interventions = Arc::from([]);
            }
            EvidenceObligationKind::ProvideConditionalLaw => {
                s.regime.conditioned_on = vars(&[Z]).into();
                s.variables = vars(&[Y]).into();
            }
            EvidenceObligationKind::IncreaseSample => s.min_additional_samples = Some(10),
            EvidenceObligationKind::EstablishAssumption => {
                s.variables = Arc::from([]);
                s.population = None;
                s.regime = ObligationRegime::observational(false);
            }
            _ => {}
        }
        let built = EvidenceObligation::try_new(s).unwrap();
        assert_eq!(built.kind, kind);
        assert!(built.id.starts_with(&format!("eo1:{}:", kind.as_str())));
        assert_eq!(EvidenceObligationKind::from_name(kind.as_str()), Some(kind));
    }
    assert_eq!(EvidenceObligationKind::ALL.len(), 9);
}

#[test]
fn f9_separate_studies_never_address_one_joint_regime_factor() {
    let obligation =
        EvidenceObligation::try_new(spec(EvidenceObligationKind::ProvideJointLaw)).unwrap();
    // Two separate studies, each a joint law over only one of the two variables.
    assert!(!obligation.addressed_by(&offer("source", &[X], &[Y], true, Some(100))));
    assert!(!obligation.addressed_by(&offer("source", &[X], &[Z], true, Some(100))));
    // One study measuring both but only as separate marginals.
    assert!(!obligation.addressed_by(&offer("source", &[X], &[Y, Z], false, Some(100))));
    // Right variables and joint law, wrong population or wrong regime.
    assert!(!obligation.addressed_by(&offer("target", &[X], &[Y, Z], true, Some(100))));
    assert!(!obligation.addressed_by(&offer("source", &[], &[Y, Z], true, Some(100))));
    assert!(!obligation.addressed_by(&offer("source", &[X, Y], &[Z], true, Some(100))));
    // The matching joint regime in the matching population does.
    assert!(obligation.addressed_by(&offer("source", &[X], &[Y, Z], true, Some(100))));
}

#[test]
fn f9_establish_assumption_is_never_addressed_by_a_study() {
    let mut s = spec(EvidenceObligationKind::EstablishAssumption);
    s.variables = Arc::from([]);
    s.population = None;
    s.regime = ObligationRegime::observational(false);
    let assumption = EvidenceObligation::try_new(s).unwrap();
    assert!(!assumption.satisfiable_by_study());
    for joint in [true, false] {
        assert!(!assumption.addressed_by(&offer("source", &[X], &[Y, Z], joint, Some(1_000_000))));
    }
    for kind in EvidenceObligationKind::ALL {
        assert_eq!(
            kind.satisfiable_by_study(),
            kind != EvidenceObligationKind::EstablishAssumption
        );
    }
    // An assumption naming a population is refused as a malformed obligation.
    let mut named = spec(EvidenceObligationKind::EstablishAssumption);
    named.variables = Arc::from([]);
    assert_eq!(
        EvidenceObligation::try_new(named).unwrap_err().detail,
        "evidence_obligations.invalid_obligation"
    );
}

#[test]
fn f9_unresolved_assumption_records_become_assumption_obligations() {
    let unresolved = ObligationRecord::new(
        "assume:no_unmeasured_confounding",
        ObligationScope::Program,
        AssumptionSource::UserDeclared,
        ObligationKind::CheckNotRun,
        AssumptionStatus::Declared,
        "no unmeasured confounding",
    )
    .with_required_check("check:overlap");
    let resolved = ObligationRecord::new(
        "implied:graph",
        ObligationScope::Atom,
        AssumptionSource::UserDeclared,
        ObligationKind::GraphicalImplication,
        AssumptionStatus::Supported,
        "graph implication",
    );
    let obligations = unresolved_assumption_obligations(&[unresolved, resolved]);
    assert_eq!(obligations.len(), 1);
    assert_eq!(obligations[0].kind, EvidenceObligationKind::EstablishAssumption);
    assert_eq!(obligations[0].provenance.family.as_ref(), "assumption_record");
    assert_eq!(obligations[0].provenance.proof_step.as_deref(), Some("check:overlap"));
    assert!(!obligations[0].satisfiable_by_study());
}

#[test]
fn f9_identity_is_order_independent_and_provenance_sensitive() {
    let a = EvidenceObligation::try_new(spec(EvidenceObligationKind::ProvideJointLaw)).unwrap();
    let mut reordered = spec(EvidenceObligationKind::ProvideJointLaw);
    reordered.variables = vars(&[Z, Y]).into();
    let b = EvidenceObligation::try_new(reordered).unwrap();
    assert_eq!(a.id, b.id);
    assert_eq!(a.canonical(), b.canonical());
    let mut other_step = spec(EvidenceObligationKind::ProvideJointLaw);
    other_step.provenance = provenance("leaf:4");
    assert_ne!(a.id, EvidenceObligation::try_new(other_step).unwrap().id);
}

#[test]
fn f9_malformed_obligations_are_refused_with_registered_codes() {
    let mut one_variable = spec(EvidenceObligationKind::ProvideJointLaw);
    one_variable.variables = vars(&[Y]).into();
    let error = EvidenceObligation::try_new(one_variable).unwrap_err();
    assert_eq!(
        (error.code, error.detail),
        ("invalid_argument", "evidence_obligations.invalid_obligation")
    );

    let mut not_joint = spec(EvidenceObligationKind::ProvideJointLaw);
    not_joint.regime.joint = false;
    let error = EvidenceObligation::try_new(not_joint).unwrap_err();
    assert_eq!(
        (error.code, error.detail),
        ("transport_missing_evidence", "evidence_obligations.wrong_contract")
    );

    let mut blank = spec(EvidenceObligationKind::Measure);
    blank.reason = Arc::from("  ");
    assert!(EvidenceObligation::try_new(blank).is_err());

    let mut duplicate = spec(EvidenceObligationKind::Measure);
    duplicate.variables = vars(&[Y, Y]).into();
    assert!(EvidenceObligation::try_new(duplicate).is_err());

    let no_rows = spec(EvidenceObligationKind::IncreaseSample);
    assert!(EvidenceObligation::try_new(no_rows).is_err());

    let mut no_population = spec(EvidenceObligationKind::Measure);
    no_population.population = None;
    assert!(EvidenceObligation::try_new(no_population).is_err());
}

#[test]
fn f9_increase_sample_is_addressed_only_by_enough_rows() {
    let mut s = spec(EvidenceObligationKind::IncreaseSample);
    s.variables = vars(&[Y]).into();
    s.regime.joint = false;
    s.min_additional_samples = Some(500);
    let obligation = EvidenceObligation::try_new(s).unwrap();
    assert!(!obligation.addressed_by(&offer("source", &[X], &[Y], true, Some(499))));
    assert!(!obligation.addressed_by(&offer("source", &[X], &[Y], true, None)));
    assert!(obligation.addressed_by(&offer("source", &[X], &[Y], true, Some(500))));
}

fn scientific_coordinates()
-> std::collections::BTreeMap<VariableId, antecedent_core::ScientificQuantity> {
    [Y, Z]
        .into_iter()
        .map(|id| {
            (
                VariableId::from_raw(id),
                antecedent_core::ScientificQuantity {
                    variable_id: format!("schema:{id}"),
                    variable_name: format!("measurement {id}"),
                    role: antecedent_core::QuantityRole::Outcome,
                    units: "kg".into(),
                    population_id: "source".into(),
                    regime_id: "observational".into(),
                    horizon: 3,
                    functional_id: "law".into(),
                    conditioning: vec![],
                    transform_id: "identity".into(),
                },
            )
        })
        .collect()
}

#[test]
fn scientific_obligations_bind_every_semantic_dimension_and_ignore_display_labels() {
    let base = EvidenceObligation::try_new(spec(EvidenceObligationKind::ProvideJointLaw))
        .unwrap()
        .with_quantities(scientific_coordinates())
        .unwrap();
    let mut renamed = scientific_coordinates();
    renamed.get_mut(&VariableId::from_raw(Y)).unwrap().variable_name = "renamed".into();
    assert_eq!(base.clone().with_quantities(renamed).unwrap().id, base.id);
    for dimension in [
        "variable",
        "role",
        "units",
        "regime",
        "horizon",
        "functional",
        "conditioning",
        "transform",
    ] {
        let mut coordinates = scientific_coordinates();
        let q = coordinates.get_mut(&VariableId::from_raw(Y)).unwrap();
        match dimension {
            "variable" => q.variable_id = "different".into(),
            "role" => q.role = antecedent_core::QuantityRole::Mediator,
            "units" => q.units = "lb".into(),
            "regime" => q.regime_id = "do(t=1)".into(),
            "horizon" => q.horizon = 4,
            "functional" => q.functional_id = "mean".into(),
            "conditioning" => {
                q.conditioning = vec![antecedent_core::QuantityCondition {
                    variable_id: "schema:c".into(),
                    value_id: "high".into(),
                }]
            }
            _ => q.transform_id = "log".into(),
        }
        assert_ne!(base.clone().with_quantities(coordinates).unwrap().id, base.id, "{dimension}");
    }
    let mut wrong_population = scientific_coordinates();
    wrong_population.get_mut(&VariableId::from_raw(Y)).unwrap().population_id = "target".into();
    assert!(base.clone().with_quantities(wrong_population).is_err());
    let mut incomplete = scientific_coordinates();
    incomplete.remove(&VariableId::from_raw(Y));
    assert!(base.with_quantities(incomplete).is_err());
}

#[test]
fn scientific_obligation_screen_refuses_missing_or_incompatible_measurement_semantics() {
    let obligation = EvidenceObligation::try_new(spec(EvidenceObligationKind::ProvideJointLaw))
        .unwrap()
        .with_quantities(scientific_coordinates())
        .unwrap();
    let mut supplied = offer("source", &[X], &[Y, Z], true, None);
    assert!(!obligation.addressed_by(&supplied));
    supplied.quantities = scientific_coordinates();
    assert!(obligation.addressed_by(&supplied));
    supplied.quantities.get_mut(&VariableId::from_raw(Y)).unwrap().units = "lb".into();
    assert!(!obligation.addressed_by(&supplied));
}
