//! The chain rule: what each stage may carry, what is refused, and which
//! attempt the refusal names.

use super::*;

fn factor(kind: SharedFactorKind, id: &str) -> FactorDecl {
    FactorDecl { kind, id: id.to_owned() }
}

fn draws(id: &str, meaning: LawMeaning, factors: Vec<FactorDecl>) -> ChainSourceDecl {
    ChainSourceDecl {
        id: id.to_owned(),
        representation: Representation::Draws,
        meaning: Some(meaning),
        alignment: Some("table-1".to_owned()),
        complete_disclosure: true,
        factors,
    }
}

fn fixed(id: &str, representation: Representation) -> ChainSourceDecl {
    ChainSourceDecl {
        id: id.to_owned(),
        representation,
        meaning: None,
        alignment: None,
        complete_disclosure: true,
        factors: Vec::new(),
    }
}

fn stage(id: &str, kind: StageKind, class: OperatorClass, inputs: &[&str]) -> ChainStageDecl {
    ChainStageDecl {
        id: id.to_owned(),
        kind,
        class,
        inputs: inputs.iter().map(|s| (*s).to_owned()).collect(),
        own_factors: Vec::new(),
    }
}

fn chain(
    sources: Vec<ChainSourceDecl>,
    stages: Vec<ChainStageDecl>,
    reading: ChainReading,
) -> ChainDecl {
    let output = stages.last().map(|s| s.id.clone()).unwrap_or_default();
    ChainDecl { construction: TableConstruction::JointDraws, sources, stages, output, reading }
}

/// measurement -> transport -> derived -> utility over a posterior of the
/// effect, a posterior of the mapping slope and a posterior of a cost.
fn release_chain(reading: ChainReading) -> ChainDecl {
    chain(
        vec![
            draws(
                "effect",
                LawMeaning::Posterior,
                vec![factor(SharedFactorKind::Row, "trial-rows")],
            ),
            draws(
                "slope",
                LawMeaning::Posterior,
                vec![factor(SharedFactorKind::Row, "calibration-rows")],
            ),
            draws(
                "cost",
                LawMeaning::Posterior,
                vec![factor(SharedFactorKind::Prior, "cost-prior")],
            ),
        ],
        vec![
            stage(
                "mapped",
                StageKind::MeasurementMapping,
                OperatorClass::Affine,
                &["effect", "slope"],
            ),
            stage("transported", StageKind::Transport, OperatorClass::Affine, &["mapped"]),
            stage(
                "net",
                StageKind::DerivedQuantity,
                OperatorClass::Nonlinear,
                &["transported", "cost"],
            ),
            stage("utility", StageKind::DecisionUtility, OperatorClass::Nonlinear, &["net"]),
        ],
        reading,
    )
}

#[test]
fn a_chain_of_four_stages_is_licensed_with_exact_dependents() {
    let plan = check_propagation_chain(&release_chain(ChainReading::Spread)).expect("licensed");
    assert_eq!(plan.label, PROPAGATION_SPREAD_LABEL);
    assert_eq!(plan.alignment.as_deref(), Some("table-1"));
    assert_eq!(plan.stage("mapped").unwrap().sources, vec!["effect", "slope"]);
    // The cost enters at the third stage: stages before it do not depend on it.
    assert_eq!(plan.dependents_of("cost"), vec!["net", "utility"]);
    assert_eq!(plan.dependents_of("slope"), vec!["mapped", "transported", "net", "utility"]);
    assert_eq!(plan.dependents_of("effect"), vec!["mapped", "transported", "net", "utility"]);
    assert!(plan.dependents_of("no-such-source").is_empty());
}

#[test]
fn an_independent_pool_of_overlapping_evidence_is_refused_and_names_the_covariance_route() {
    let sources = vec![
        draws("a", LawMeaning::Sampling, vec![factor(SharedFactorKind::Row, "rows-1")]),
        draws("b", LawMeaning::Sampling, vec![factor(SharedFactorKind::Row, "rows-1")]),
    ];
    let pooled = |class| {
        chain(
            sources.clone(),
            vec![stage("pool", StageKind::Pooling, class, &["a", "b"])],
            ChainReading::Spread,
        )
    };
    let refusal =
        check_propagation_chain(&pooled(OperatorClass::IndependentPool)).expect_err("refused");
    assert_eq!(refusal.code(), "evidence_double_counted");
    assert_eq!(refusal.detail(), "propagation_chain.pool_shares_evidence");
    assert_eq!(refusal.next_attempt(), POOL_WITH_JOINT_COVARIANCE);
    assert!(refusal.to_string().contains("next attempt"));
    // The dependent pool is the licensed route and records what it shares.
    let plan = check_propagation_chain(&pooled(OperatorClass::DependentPool)).expect("licensed");
    let dependence = plan.stage("pool").unwrap().dependence.clone().expect("dependence");
    assert_eq!(dependence.status, DependenceStatus::Dependent);
    assert_eq!(dependence.shared, vec!["row:rows-1"]);
}

#[test]
fn two_inputs_built_from_one_source_share_it_without_a_declared_factor() {
    let sources = vec![draws("a", LawMeaning::Sampling, Vec::new())];
    let stages = vec![
        stage("double", StageKind::DerivedQuantity, OperatorClass::Affine, &["a"]),
        stage("shifted", StageKind::DerivedQuantity, OperatorClass::Affine, &["a"]),
        stage("pool", StageKind::Pooling, OperatorClass::IndependentPool, &["double", "shifted"]),
    ];
    let refusal = check_propagation_chain(&chain(sources, stages, ChainReading::Spread))
        .expect_err("refused");
    assert_eq!(refusal.code(), "evidence_double_counted");
    assert_eq!(refusal.stage(), Some("pool"));
}

#[test]
fn an_independent_pool_of_distinct_evidence_is_licensed() {
    let sources = vec![
        draws("a", LawMeaning::Sampling, vec![factor(SharedFactorKind::Row, "rows-1")]),
        draws("b", LawMeaning::Sampling, vec![factor(SharedFactorKind::Row, "rows-2")]),
    ];
    let stages =
        vec![stage("pool", StageKind::Pooling, OperatorClass::IndependentPool, &["a", "b"])];
    let plan =
        check_propagation_chain(&chain(sources, stages, ChainReading::Spread)).expect("licensed");
    assert_eq!(
        plan.stage("pool").unwrap().dependence.as_ref().unwrap().status,
        DependenceStatus::Independent
    );
}

#[test]
fn a_shared_prior_or_model_is_shared_evidence() {
    for kind in [SharedFactorKind::Prior, SharedFactorKind::Model, SharedFactorKind::Disturbance] {
        let sources = vec![
            draws("a", LawMeaning::Posterior, vec![factor(kind, "shared")]),
            draws("b", LawMeaning::Posterior, vec![factor(kind, "shared")]),
        ];
        let stages =
            vec![stage("pool", StageKind::Pooling, OperatorClass::IndependentPool, &["a", "b"])];
        let refusal = check_propagation_chain(&chain(sources, stages, ChainReading::Spread))
            .expect_err("refused");
        assert_eq!(refusal.code(), "evidence_double_counted", "{kind:?}");
    }
}

#[test]
fn a_stage_bringing_a_shared_factor_makes_its_dependents_overlap() {
    // Two measurement mappings calibrated on the same sample share it through the stage.
    let sources = vec![
        draws("a", LawMeaning::Sampling, vec![factor(SharedFactorKind::Row, "rows-1")]),
        draws("b", LawMeaning::Sampling, vec![factor(SharedFactorKind::Row, "rows-2")]),
    ];
    let mut first = stage("map-a", StageKind::MeasurementMapping, OperatorClass::Affine, &["a"]);
    first.own_factors.push(factor(SharedFactorKind::Row, "calibration"));
    let mut second = stage("map-b", StageKind::MeasurementMapping, OperatorClass::Affine, &["b"]);
    second.own_factors.push(factor(SharedFactorKind::Row, "calibration"));
    let stages = vec![
        first,
        second,
        stage("pool", StageKind::Pooling, OperatorClass::IndependentPool, &["map-a", "map-b"]),
    ];
    let refusal = check_propagation_chain(&chain(sources, stages, ChainReading::Spread))
        .expect_err("refused");
    assert_eq!(refusal.detail(), "propagation_chain.pool_shares_evidence");
}

#[test]
fn an_undeclared_overlap_blocks_an_independent_pool() {
    let mut partial = draws("a", LawMeaning::Sampling, Vec::new());
    partial.complete_disclosure = false;
    let sources = vec![partial, draws("b", LawMeaning::Sampling, Vec::new())];
    let stages =
        vec![stage("pool", StageKind::Pooling, OperatorClass::IndependentPool, &["a", "b"])];
    let refusal = check_propagation_chain(&chain(sources, stages, ChainReading::Spread))
        .expect_err("refused");
    assert_eq!(refusal.code(), "evidence_overlap_unknown");
    assert_eq!(refusal.next_attempt(), DECLARE_SHARED_FACTORS);
}

#[test]
fn unaligned_draws_are_not_paired_by_index() {
    let mut other = draws("b", LawMeaning::Posterior, Vec::new());
    other.alignment = Some("table-2".to_owned());
    let stages =
        vec![stage("sum", StageKind::DerivedQuantity, OperatorClass::Nonlinear, &["a", "b"])];
    // Distinct evidence: the next attempt is the independent product table.
    let independent = chain(
        vec![draws("a", LawMeaning::Posterior, Vec::new()), other.clone()],
        stages.clone(),
        ChainReading::Spread,
    );
    let refusal = check_propagation_chain(&independent).expect_err("refused");
    assert_eq!(refusal.code(), "joint_law_required");
    assert_eq!(refusal.next_attempt(), BUILD_INDEPENDENT_PRODUCT_TABLE);
    // Shared evidence: only aligned joint draws can carry it.
    let shared_factor = vec![factor(SharedFactorKind::Model, "m")];
    let mut dependent_b = other;
    dependent_b.factors = shared_factor.clone();
    let dependent = chain(
        vec![draws("a", LawMeaning::Posterior, shared_factor), dependent_b],
        stages,
        ChainReading::Spread,
    );
    let refusal = check_propagation_chain(&dependent).expect_err("refused");
    assert_eq!(refusal.next_attempt(), SUPPLY_ALIGNED_DRAWS);
}

#[test]
fn an_independent_product_table_cannot_carry_a_shared_factor() {
    let mut decl = chain(
        vec![
            draws("a", LawMeaning::Posterior, vec![factor(SharedFactorKind::Row, "r")]),
            draws("b", LawMeaning::Posterior, vec![factor(SharedFactorKind::Row, "r")]),
        ],
        vec![stage("sum", StageKind::DerivedQuantity, OperatorClass::Nonlinear, &["a", "b"])],
        ChainReading::Spread,
    );
    decl.construction = TableConstruction::IndependentProduct;
    let refusal = check_propagation_chain(&decl).expect_err("refused");
    assert_eq!(refusal.code(), "evidence_double_counted");
    assert_eq!(refusal.next_attempt(), SUPPLY_JOINT_DRAWS);
    // Distinct evidence is licensed as an independent product.
    decl.sources[1].factors = vec![factor(SharedFactorKind::Row, "other")];
    check_propagation_chain(&decl).expect("licensed");
}

#[test]
fn a_point_carries_no_spread_and_an_interval_no_posterior() {
    let point_chain = |reading| {
        chain(
            vec![fixed("p", Representation::Point), fixed("q", Representation::Point)],
            vec![stage("net", StageKind::DerivedQuantity, OperatorClass::Nonlinear, &["p", "q"])],
            reading,
        )
    };
    check_propagation_chain(&point_chain(ChainReading::Point)).expect("a value");
    let refusal = check_propagation_chain(&point_chain(ChainReading::Spread)).expect_err("refused");
    assert_eq!(refusal.code(), "representation_mapping_required");
    assert_eq!(refusal.next_attempt(), SUPPLY_DRAWS_FOR_FIXED_INPUT);

    let interval_chain = |reading| {
        chain(
            vec![fixed("lo", Representation::Interval), fixed("p", Representation::Point)],
            vec![
                stage("scaled", StageKind::MeasurementMapping, OperatorClass::Affine, &["lo"]),
                stage(
                    "net",
                    StageKind::DerivedQuantity,
                    OperatorClass::Nonlinear,
                    &["scaled", "p"],
                ),
            ],
            reading,
        )
    };
    let plan =
        check_propagation_chain(&interval_chain(ChainReading::Enclosure)).expect("enclosure");
    assert_eq!(plan.label, PROPAGATION_ENCLOSURE_LABEL);
    let refusal =
        check_propagation_chain(&interval_chain(ChainReading::Posterior)).expect_err("refused");
    assert_eq!(refusal.detail(), "propagation_chain.interval_is_not_a_posterior");
    assert_eq!(refusal.next_attempt(), SUPPLY_POSTERIOR_DRAWS);
    let refusal =
        check_propagation_chain(&interval_chain(ChainReading::Spread)).expect_err("refused");
    assert_eq!(refusal.next_attempt(), USE_ENCLOSURE_READING);
}

#[test]
fn a_sampling_law_is_not_a_posterior() {
    let decl = chain(
        vec![draws("a", LawMeaning::Sampling, Vec::new())],
        vec![stage("u", StageKind::DecisionUtility, OperatorClass::Nonlinear, &["a"])],
        ChainReading::Posterior,
    );
    let refusal = check_propagation_chain(&decl).expect_err("refused");
    assert_eq!(refusal.code(), "representation_mapping_required");
    assert_eq!(refusal.detail(), "propagation_chain.sampling_law_is_not_a_posterior");
    // Mixing a sampling law with a posterior is a meaning mismatch.
    let mixed = chain(
        vec![
            draws("a", LawMeaning::Sampling, Vec::new()),
            draws("b", LawMeaning::Posterior, Vec::new()),
        ],
        vec![stage("sum", StageKind::DerivedQuantity, OperatorClass::Nonlinear, &["a", "b"])],
        ChainReading::Spread,
    );
    let refusal = check_propagation_chain(&mixed).expect_err("refused");
    assert_eq!(refusal.code(), "distribution_meaning_mismatch");
    assert_eq!(refusal.next_attempt(), MATCH_LAW_MEANING);
}

#[test]
fn intervals_cannot_be_pooled_or_mixed_with_draws_or_updated() {
    let interval_pool = chain(
        vec![fixed("a", Representation::Interval), fixed("b", Representation::Interval)],
        vec![stage("pool", StageKind::Pooling, OperatorClass::DependentPool, &["a", "b"])],
        ChainReading::Enclosure,
    );
    assert_eq!(
        check_propagation_chain(&interval_pool).expect_err("refused").detail(),
        "propagation_chain.interval_cannot_be_pooled"
    );
    let mixed = chain(
        vec![fixed("a", Representation::Interval), draws("b", LawMeaning::Posterior, Vec::new())],
        vec![stage("sum", StageKind::DerivedQuantity, OperatorClass::Nonlinear, &["a", "b"])],
        ChainReading::Spread,
    );
    assert_eq!(
        check_propagation_chain(&mixed).expect_err("refused").next_attempt(),
        SUPPLY_DRAWS_FOR_INTERVAL
    );
    let interval_update = chain(
        vec![fixed("a", Representation::Interval), fixed("b", Representation::Interval)],
        vec![stage("update", StageKind::PreposteriorUpdate, OperatorClass::Update, &["a", "b"])],
        ChainReading::Spread,
    );
    assert_eq!(
        check_propagation_chain(&interval_update).expect_err("refused").next_attempt(),
        SUPPLY_POSTERIOR_DRAWS
    );
}

#[test]
fn a_preposterior_update_needs_posterior_draws_and_is_terminal() {
    let sources = vec![draws("theta", LawMeaning::Posterior, Vec::new())];
    let base = vec![
        stage("utility", StageKind::DecisionUtility, OperatorClass::Nonlinear, &["theta"]),
        stage(
            "update",
            StageKind::PreposteriorUpdate,
            OperatorClass::Update,
            &["theta", "utility"],
        ),
    ];
    let plan = check_propagation_chain(&chain(sources, base.clone(), ChainReading::Spread))
        .expect("licensed");
    assert_eq!(plan.stage("update").unwrap().meaning, Some(LawMeaning::Posterior));
    // A sampling law is not updated by a likelihood.
    let sampling = vec![draws("theta", LawMeaning::Sampling, Vec::new())];
    let refusal = check_propagation_chain(&chain(sampling, base.clone(), ChainReading::Spread))
        .expect_err("refused");
    assert_eq!(refusal.detail(), "propagation_chain.sampling_law_is_not_a_posterior");
    // Nothing may consume the update.
    let mut after = base;
    after.push(stage("after", StageKind::DerivedQuantity, OperatorClass::Nonlinear, &["update"]));
    let refusal = check_propagation_chain(&chain(
        vec![draws("theta", LawMeaning::Posterior, Vec::new())],
        after,
        ChainReading::Spread,
    ))
    .expect_err("refused");
    assert_eq!(refusal.code(), "invalid_argument");
}

#[test]
fn an_interval_reading_of_a_composed_quantity_needs_a_measured_record() {
    let refusal =
        check_propagation_chain(&release_chain(ChainReading::Interval)).expect_err("refused");
    assert_eq!(refusal.code(), "interval_level_not_measured");
    assert_eq!(refusal.next_attempt(), MEASURE_COMPOSED_COVERAGE);
}

#[test]
fn an_operator_must_fit_its_stage_and_the_declaration_must_be_well_formed() {
    let sources = vec![draws("a", LawMeaning::Posterior, Vec::new())];
    let wrong = chain(
        sources.clone(),
        vec![stage("pool", StageKind::Pooling, OperatorClass::Affine, &["a"])],
        ChainReading::Spread,
    );
    assert_eq!(check_propagation_chain(&wrong).expect_err("refused").code(), "route_not_supported");
    let forward = chain(
        sources.clone(),
        vec![stage("x", StageKind::DerivedQuantity, OperatorClass::Nonlinear, &["later"])],
        ChainReading::Spread,
    );
    assert_eq!(check_propagation_chain(&forward).expect_err("refused").code(), "invalid_argument");
    let repeated = chain(
        sources,
        vec![stage("a", StageKind::DerivedQuantity, OperatorClass::Nonlinear, &["a"])],
        ChainReading::Spread,
    );
    assert_eq!(check_propagation_chain(&repeated).expect_err("refused").code(), "invalid_argument");
}

#[test]
fn a_point_source_inside_a_draw_chain_is_listed_as_fixed() {
    let decl = chain(
        vec![draws("a", LawMeaning::Posterior, Vec::new()), fixed("rate", Representation::Point)],
        vec![stage("net", StageKind::DerivedQuantity, OperatorClass::Nonlinear, &["a", "rate"])],
        ChainReading::Spread,
    );
    let plan = check_propagation_chain(&decl).expect("licensed");
    assert_eq!(plan.fixed_sources, vec!["rate"]);
}

#[test]
fn every_named_next_attempt_is_catalogued_and_distinct() {
    let names: BTreeSet<&str> = NEXT_ATTEMPTS.iter().map(|a| a.name()).collect();
    assert_eq!(names.len(), NEXT_ATTEMPTS.len());
    for attempt in NEXT_ATTEMPTS {
        assert_eq!(NextAttempt::from_name(&NEXT_ATTEMPTS, attempt.name()), Some(attempt));
    }
}

#[test]
fn canonical_tags_are_distinct_and_names_round_trip() {
    let kinds: BTreeSet<u64> = StageKind::ALL.iter().map(|k| k.tag()).collect();
    assert_eq!(kinds.len(), StageKind::ALL.len());
    let classes: BTreeSet<u64> = OperatorClass::ALL.iter().map(|k| k.tag()).collect();
    assert_eq!(classes.len(), OperatorClass::ALL.len());
    for reading in ChainReading::ALL {
        assert_eq!(ChainReading::from_name(reading.as_str()), Some(reading));
    }
}
