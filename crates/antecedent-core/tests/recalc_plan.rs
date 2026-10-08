//! 2.3 C2 stage model: data-driven mutation tables for `RecalcPlan`.
//!
//! Each mutation asserts the complete per-stage status table (stage label and status), that
//! every reused stage keeps an identical effective identity, and that every in-process
//! recomputed stage's effective identity really changed. The fresh-process cases assert that
//! no stage whose artifact is missing is ever reported as reused.

use std::collections::BTreeMap;

use antecedent_core::reason_code::is_registered;
use antecedent_core::recalc::{
    Boundary, Branch, ChangedDependency, MissingDependency, RecalcCapabilities, RecalcPlan,
    RefusalReason, RequestSupport, ResumeContext, RetargetSupport, Stage, StageIdentities,
    StageIdentity, StageStatus,
};

type Versions = BTreeMap<Stage, String>;
type Table = Vec<(String, String)>;

fn b(i: u8) -> Branch {
    Branch::new(i).unwrap()
}

fn native_versions() -> Versions {
    let mut v = BTreeMap::new();
    for stage in [
        Stage::Graph,
        Stage::Query,
        Stage::Regime,
        Stage::Evidence,
        Stage::SourcePopulation,
        Stage::TargetPopulation,
        Stage::DataSnapshot,
        Stage::RowDesign,
        Stage::TreatmentGrid,
        Stage::LearnerFoldsRng,
        Stage::Utility,
        Stage::Identification,
        Stage::ScoreArtifact,
        Stage::Law,
        Stage::Decision,
    ] {
        v.insert(stage, "v1".to_string());
    }
    v
}

/// Native branch plus two independent external studies, each with a provider request and prior.
fn full_versions() -> Versions {
    let mut v = native_versions();
    for i in 0..2 {
        for stage in [Stage::ExternalStudy(b(i)), Stage::ProviderRequest(b(i)), Stage::Prior(b(i))]
        {
            v.insert(stage, "v1".to_string());
        }
    }
    v
}

fn identities(v: &Versions) -> StageIdentities {
    let mut ids = StageIdentities::new();
    for (stage, version) in v {
        ids.set(*stage, StageIdentity::of(&stage.label(), &[version.as_bytes()]));
    }
    ids
}

fn bump(v: &Versions, stage: Stage) -> Versions {
    let mut out = v.clone();
    out.insert(stage, "v2".to_string());
    out
}

fn licensed() -> RecalcCapabilities {
    RecalcCapabilities::in_process(RetargetSupport::Licensed)
}

fn run(prev: &Versions, req: &Versions, caps: &RecalcCapabilities) -> RecalcPlan {
    RecalcPlan::plan(&identities(prev), &identities(req), caps)
}

fn table(plan: &RecalcPlan) -> Table {
    plan.entries().iter().map(|e| (e.stage.label(), e.status.to_string())).collect()
}

fn reused_string(stage: Stage) -> String {
    let dep = stage.dependencies(RetargetSupport::Licensed).first().copied().unwrap_or(stage);
    format!("reused({dep})")
}

/// The whole expected table: every stage of `versions` reused unless overridden.
fn expected(versions: &Versions, overrides: &[(Stage, &str)]) -> Table {
    versions
        .keys()
        .map(|stage| {
            let status = overrides
                .iter()
                .find(|(s, _)| s == stage)
                .map_or_else(|| reused_string(*stage), |(_, text)| (*text).to_string());
            (stage.label(), status)
        })
        .collect()
}

fn assert_identity_consistency(
    prev: &Versions,
    req: &Versions,
    caps: &RecalcCapabilities,
    plan: &RecalcPlan,
) {
    let before = identities(prev).effective(caps.retarget);
    let after = identities(req).effective(caps.retarget);
    for entry in plan.entries() {
        assert_eq!(Some(&entry.identity), after.get(&entry.stage), "{}", entry.stage);
        let same = before.get(&entry.stage) == after.get(&entry.stage);
        match entry.status {
            StageStatus::Reused { .. } => {
                assert!(same, "{} reused but identity changed", entry.stage)
            }
            StageStatus::Recomputed { because } if because != ChangedDependency::FreshProcess => {
                assert!(!same, "{} recomputed but identity unchanged", entry.stage);
            }
            _ => {}
        }
    }
}

fn assert_reused_keep_identity(base: &RecalcPlan, plan: &RecalcPlan) {
    for stage in plan.reused() {
        assert_eq!(base.identity(stage), plan.identity(stage), "{stage}");
    }
}

#[test]
fn c2_recalc_unchanged_request_reuses_every_stage() {
    let v = full_versions();
    let plan = run(&v, &v, &licensed());
    assert_eq!(table(&plan), expected(&v, &[]));
    assert!(plan.recomputed().is_empty() && plan.refused().is_empty());
    assert_identity_consistency(&v, &v, &licensed(), &plan);
}

#[test]
fn c2_recalc_first_request_has_nothing_to_reuse() {
    let v = native_versions();
    let plan = RecalcPlan::plan(&StageIdentities::new(), &identities(&v), &licensed());
    assert!(plan.reused().is_empty());
    assert_eq!(plan.recomputed().len(), v.len());
    assert_eq!(
        plan.status(Stage::Decision),
        Some(&StageStatus::Recomputed {
            because: ChangedDependency::Own {
                stage: Stage::Decision,
                change: antecedent_core::recalc::ChangeKind::Added,
            }
        })
    );
}

/// `(name, mutated stage, capabilities, overrides)` for in-process single-input mutations.
type MutationCase = (&'static str, Stage, RecalcCapabilities, Vec<(Stage, &'static str)>);

/// Cases that mutate the target population and the utility.
fn target_cases() -> Vec<MutationCase> {
    vec![
        (
            "utility only recomputes the decision",
            Stage::Utility,
            licensed(),
            vec![
                (Stage::Utility, "recomputed(own:utility:modified)"),
                (Stage::Decision, "recomputed(upstream:utility<-utility:modified)"),
            ],
        ),
        (
            "licensed target-weight change reuses frozen scores",
            Stage::TargetPopulation,
            licensed(),
            vec![
                (Stage::TargetPopulation, "recomputed(own:target_population:modified)"),
                (Stage::Law, "recomputed(upstream:target_population<-target_population:modified)"),
                (Stage::Decision, "recomputed(upstream:law<-target_population:modified)"),
            ],
        ),
        (
            "undeclared retarget refits the score artifact on a target change",
            Stage::TargetPopulation,
            RecalcCapabilities::in_process(RetargetSupport::NotDeclared),
            vec![
                (Stage::TargetPopulation, "recomputed(own:target_population:modified)"),
                (
                    Stage::ScoreArtifact,
                    "recomputed(upstream:target_population<-target_population:modified)",
                ),
                (Stage::Law, "recomputed(upstream:score_artifact<-target_population:modified)"),
                (Stage::Decision, "recomputed(upstream:law<-target_population:modified)"),
            ],
        ),
        (
            "incompatible target weights are refused at the law",
            Stage::TargetPopulation,
            RecalcCapabilities::in_process(RetargetSupport::Incompatible),
            vec![
                (Stage::TargetPopulation, "recomputed(own:target_population:modified)"),
                (Stage::Law, "refused(recalc.retarget_incompatible)"),
                (Stage::Decision, "refused(recalc.blocked_by_refused_dependency[law])"),
            ],
        ),
    ]
}

/// Cases that mutate external studies, data, folds and the identified request.
fn input_cases() -> Vec<MutationCase> {
    let ext0 = Stage::ExternalStudy(b(0));
    vec![
        (
            "external study 0 invalidates only its prior, provider and the decision",
            ext0,
            licensed(),
            vec![
                (ext0, "recomputed(own:external_study.0:modified)"),
                (
                    Stage::ProviderRequest(b(0)),
                    "recomputed(upstream:external_study.0<-external_study.0:modified)",
                ),
                (
                    Stage::Prior(b(0)),
                    "recomputed(upstream:external_study.0<-external_study.0:modified)",
                ),
                (Stage::Decision, "recomputed(upstream:prior.0<-external_study.0:modified)"),
            ],
        ),
        (
            "new outcomes (data snapshot) refit",
            Stage::DataSnapshot,
            licensed(),
            vec![
                (Stage::DataSnapshot, "recomputed(own:data_snapshot:modified)"),
                (
                    Stage::ScoreArtifact,
                    "recomputed(upstream:data_snapshot<-data_snapshot:modified)",
                ),
                (Stage::Law, "recomputed(upstream:score_artifact<-data_snapshot:modified)"),
                (Stage::Decision, "recomputed(upstream:law<-data_snapshot:modified)"),
            ],
        ),
        (
            "changed folds or RNG refit",
            Stage::LearnerFoldsRng,
            licensed(),
            vec![
                (Stage::LearnerFoldsRng, "recomputed(own:learner_folds_rng:modified)"),
                (
                    Stage::ScoreArtifact,
                    "recomputed(upstream:learner_folds_rng<-learner_folds_rng:modified)",
                ),
                (Stage::Law, "recomputed(upstream:score_artifact<-learner_folds_rng:modified)"),
                (Stage::Decision, "recomputed(upstream:law<-learner_folds_rng:modified)"),
            ],
        ),
        (
            "graph change re-identifies",
            Stage::Graph,
            licensed(),
            reidentify_overrides(Stage::Graph, "graph"),
        ),
        (
            "regime change re-identifies",
            Stage::Regime,
            licensed(),
            reidentify_overrides(Stage::Regime, "regime"),
        ),
        (
            "query change re-identifies",
            Stage::Query,
            licensed(),
            reidentify_overrides(Stage::Query, "query"),
        ),
    ]
}

#[test]
fn c2_recalc_mutation_tables_are_exact() {
    let mut cases = target_cases();
    cases.extend(input_cases());
    for (name, stage, caps, overrides) in cases {
        let base = if matches!(
            stage,
            Stage::ExternalStudy(_) | Stage::Graph | Stage::Query | Stage::Regime
        ) {
            full_versions()
        } else {
            native_versions()
        };
        let base_plan = run(&base, &base, &licensed());
        let changed = bump(&base, stage);
        let plan = run(&base, &changed, &caps);
        assert_eq!(table(&plan), expected(&base, &overrides), "{name}");
        assert_reused_keep_identity(&base_plan, &plan);
        if caps.retarget != RetargetSupport::Incompatible {
            assert_identity_consistency(&base, &changed, &caps, &plan);
        }
    }
}

fn reidentify_overrides(stage: Stage, label: &'static str) -> Vec<(Stage, &'static str)> {
    // Leak-free static strings are required by the table, so spell each text out per label.
    let own: &'static str = match label {
        "graph" => "recomputed(own:graph:modified)",
        "regime" => "recomputed(own:regime:modified)",
        _ => "recomputed(own:query:modified)",
    };
    let ident: &'static str = match label {
        "graph" => "recomputed(upstream:graph<-graph:modified)",
        "regime" => "recomputed(upstream:regime<-regime:modified)",
        _ => "recomputed(upstream:query<-query:modified)",
    };
    let scores: &'static str = match label {
        "graph" => "recomputed(upstream:identification<-graph:modified)",
        "regime" => "recomputed(upstream:identification<-regime:modified)",
        _ => "recomputed(upstream:identification<-query:modified)",
    };
    let law: &'static str = match label {
        "graph" => "recomputed(upstream:score_artifact<-graph:modified)",
        "regime" => "recomputed(upstream:score_artifact<-regime:modified)",
        _ => "recomputed(upstream:score_artifact<-query:modified)",
    };
    let decision: &'static str = match label {
        "graph" => "recomputed(upstream:law<-graph:modified)",
        "regime" => "recomputed(upstream:law<-regime:modified)",
        _ => "recomputed(upstream:law<-query:modified)",
    };
    let mut overrides = vec![
        (stage, own),
        (Stage::Identification, ident),
        (Stage::ScoreArtifact, scores),
        (Stage::Law, law),
        (Stage::Decision, decision),
    ];
    // Both external branches are bound to the same causal request. Their data remain
    // unchanged, but their request and prior interpretation must be checked again.
    for branch in [b(0), b(1)] {
        overrides.push((Stage::ProviderRequest(branch), ident));
        overrides.push((Stage::Prior(branch), ident));
    }
    overrides
}

#[test]
fn c2_recalc_evidence_change_reidentifies_and_rebuilds_every_prior() {
    let base = full_versions();
    let changed = bump(&base, Stage::Evidence);
    let plan = run(&base, &changed, &licensed());
    let overrides = [
        (Stage::Evidence, "recomputed(own:evidence:modified)"),
        (Stage::ProviderRequest(b(0)), "recomputed(upstream:evidence<-evidence:modified)"),
        (Stage::ProviderRequest(b(1)), "recomputed(upstream:evidence<-evidence:modified)"),
        (Stage::Prior(b(0)), "recomputed(upstream:evidence<-evidence:modified)"),
        (Stage::Prior(b(1)), "recomputed(upstream:evidence<-evidence:modified)"),
        (Stage::Identification, "recomputed(upstream:evidence<-evidence:modified)"),
        (Stage::ScoreArtifact, "recomputed(upstream:identification<-evidence:modified)"),
        (Stage::Law, "recomputed(upstream:score_artifact<-evidence:modified)"),
        (Stage::Decision, "recomputed(upstream:law<-evidence:modified)"),
    ];
    assert_eq!(table(&plan), expected(&base, &overrides));
    assert_identity_consistency(&base, &changed, &licensed(), &plan);
}

#[test]
fn c2_recalc_external_requests_and_priors_bind_the_full_causal_request() {
    let base = full_versions();
    let before = run(&base, &base, &licensed());
    for input in [
        Stage::Graph,
        Stage::Query,
        Stage::Regime,
        Stage::Evidence,
        Stage::SourcePopulation,
        Stage::TargetPopulation,
        Stage::TreatmentGrid,
    ] {
        let changed = bump(&base, input);
        let plan = run(&base, &changed, &licensed());
        for branch in [b(0), b(1)] {
            assert!(matches!(
                plan.status(Stage::ExternalStudy(branch)),
                Some(StageStatus::Reused { .. })
            ));
            for stage in [Stage::ProviderRequest(branch), Stage::Prior(branch)] {
                assert!(
                    matches!(plan.status(stage), Some(StageStatus::Recomputed { .. })),
                    "{input}: {stage}"
                );
                assert_ne!(before.identity(stage), plan.identity(stage), "{input}: {stage}");
            }
        }
        assert_identity_consistency(&base, &changed, &licensed(), &plan);
    }
}

#[test]
fn c2_recalc_bound_score_snapshot_reuses_scores_without_claiming_supplied_data() {
    let base = native_versions();
    let resume = ResumeContext {
        portable_scores: true,
        scores_snapshot_bound: true,
        ..ResumeContext::default()
    };
    assert!(!resume.supplied_data);
    let caps = RecalcCapabilities::fresh_process(RetargetSupport::Licensed, resume);
    let plan = run(&base, &base, &caps);
    assert!(plan.is_executable());
    assert!(matches!(plan.status(Stage::DataSnapshot), Some(StageStatus::Reused { .. })));
    assert!(matches!(plan.status(Stage::ScoreArtifact), Some(StageStatus::Reused { .. })));
    assert_eq!(
        plan.recomputed_computations(),
        vec![Stage::Identification, Stage::Law, Stage::Decision]
    );
    // A bound digest only licenses those exact scores, not data for a refit.
    for stage in [Stage::DataSnapshot, Stage::Query, Stage::LearnerFoldsRng] {
        let changed = run(&base, &bump(&base, stage), &caps);
        assert!(!changed.is_executable(), "{stage}");
        assert!(matches!(changed.status(Stage::ScoreArtifact), Some(StageStatus::Refused { .. })));
    }
}

#[test]
fn c2_recalc_external_study_change_leaves_unrelated_branches_identical() {
    let base = full_versions();
    let base_plan = run(&base, &base, &licensed());
    let changed = bump(&base, Stage::ExternalStudy(b(0)));
    let plan = run(&base, &changed, &licensed());
    for stage in [
        Stage::ExternalStudy(b(1)),
        Stage::ProviderRequest(b(1)),
        Stage::Prior(b(1)),
        Stage::Identification,
        Stage::ScoreArtifact,
        Stage::Law,
    ] {
        assert!(matches!(plan.status(stage), Some(StageStatus::Reused { .. })), "{stage}");
        assert_eq!(plan.identity(stage), base_plan.identity(stage), "{stage}");
    }
    for stage in [
        Stage::ExternalStudy(b(0)),
        Stage::ProviderRequest(b(0)),
        Stage::Prior(b(0)),
        Stage::Decision,
    ] {
        assert_ne!(plan.identity(stage), base_plan.identity(stage), "{stage}");
    }
    // Changing the other study changes the mirror-image set.
    let other = run(&base, &bump(&base, Stage::ExternalStudy(b(1))), &licensed());
    assert_eq!(
        other.recomputed(),
        vec![
            Stage::ExternalStudy(b(1)),
            Stage::ProviderRequest(b(1)),
            Stage::Prior(b(1)),
            Stage::Decision
        ]
    );
}

#[test]
fn c2_recalc_added_and_removed_external_studies_reach_only_the_decision() {
    let base = full_versions();
    let mut without = base.clone();
    for stage in [Stage::ExternalStudy(b(1)), Stage::ProviderRequest(b(1)), Stage::Prior(b(1))] {
        without.remove(&stage);
    }
    let removed = run(&base, &without, &licensed());
    let overrides = [(Stage::Decision, "recomputed(upstream:prior.1<-prior.1:removed)")];
    assert_eq!(table(&removed), expected(&without, &overrides));

    let added = run(&without, &base, &licensed());
    let overrides = [
        (Stage::ExternalStudy(b(1)), "recomputed(own:external_study.1:added)"),
        (Stage::ProviderRequest(b(1)), "recomputed(own:provider_request.1:added)"),
        (Stage::Prior(b(1)), "recomputed(own:prior.1:added)"),
        (Stage::Decision, "recomputed(upstream:prior.1<-prior.1:added)"),
    ];
    assert_eq!(table(&added), expected(&base, &overrides));
}

#[test]
fn c2_recalc_off_grid_and_unsupported_requests_refuse_or_name_a_licensed_route() {
    let base = full_versions();
    let cases = [
        (
            RequestSupport::OffGrid { licensed_route: None },
            "refused(recalc.off_grid_request)",
            "recalc.off_grid_request",
            "route_not_supported",
        ),
        (
            RequestSupport::OffGrid { licensed_route: Some("dose.smoothed_response") },
            "refused(recalc.off_grid_request[dose.smoothed_response])",
            "recalc.off_grid_request",
            "route_not_supported",
        ),
        (
            RequestSupport::Unsupported { licensed_route: None },
            "refused(recalc.unsupported_request)",
            "recalc.unsupported_request",
            "route_not_supported",
        ),
    ];
    for (request, grid_text, detail, code) in cases {
        let caps = RecalcCapabilities { request, ..licensed() };
        let plan = run(&base, &base, &caps);
        let overrides = [
            (Stage::TreatmentGrid, grid_text),
            (
                Stage::ProviderRequest(b(0)),
                "refused(recalc.blocked_by_refused_dependency[treatment_grid])",
            ),
            (
                Stage::ProviderRequest(b(1)),
                "refused(recalc.blocked_by_refused_dependency[treatment_grid])",
            ),
            (Stage::Prior(b(0)), "refused(recalc.blocked_by_refused_dependency[treatment_grid])"),
            (Stage::Prior(b(1)), "refused(recalc.blocked_by_refused_dependency[treatment_grid])"),
            (Stage::ScoreArtifact, "refused(recalc.blocked_by_refused_dependency[treatment_grid])"),
            (Stage::Law, "refused(recalc.blocked_by_refused_dependency[score_artifact])"),
            (Stage::Decision, "refused(recalc.blocked_by_refused_dependency[law])"),
        ];
        assert_eq!(table(&plan), expected(&base, &overrides));
        let (stage, reason) = plan.first_refusal().unwrap();
        assert_eq!(stage, Stage::TreatmentGrid);
        assert_eq!(reason.detail(), detail);
        assert_eq!(reason.reason_code(), code);
        assert!(!plan.is_executable());
    }
}

#[test]
fn c2_recalc_every_refusal_detail_is_a_namespaced_literal_with_a_registered_code() {
    let reasons = [
        RefusalReason::Unavailable { missing: MissingDependency::Fit },
        RefusalReason::Unavailable { missing: MissingDependency::Data },
        RefusalReason::Unavailable { missing: MissingDependency::Provider },
        RefusalReason::OffGrid { licensed_route: None },
        RefusalReason::Unsupported { licensed_route: None },
        RefusalReason::RetargetIncompatible,
        RefusalReason::Blocked { by: Stage::Law },
    ];
    for reason in reasons {
        let detail = reason.detail();
        let (namespace, name) = detail.split_once('.').unwrap();
        assert_eq!(namespace, "recalc");
        assert!(name.chars().all(|c| c.is_ascii_lowercase() || c == '_'), "{detail}");
        assert!(is_registered(reason.reason_code()), "{}", reason.reason_code());
    }
}

fn fresh(retarget: RetargetSupport, resume: ResumeContext) -> RecalcCapabilities {
    RecalcCapabilities::fresh_process(retarget, resume)
}

#[test]
fn c2_recalc_an_ordinary_loaded_result_never_claims_reuse_of_a_derived_stage() {
    let v = full_versions();
    let caps = fresh(RetargetSupport::Licensed, ResumeContext::default());
    let plan = run(&v, &v, &caps);
    let overrides = [
        (Stage::DataSnapshot, "refused(recalc.unavailable_data)"),
        (Stage::ProviderRequest(b(0)), "refused(recalc.unavailable_provider)"),
        (Stage::ProviderRequest(b(1)), "refused(recalc.unavailable_provider)"),
        (Stage::Prior(b(0)), "recomputed(fresh_process)"),
        (Stage::Prior(b(1)), "recomputed(fresh_process)"),
        (Stage::Identification, "recomputed(fresh_process)"),
        (Stage::ScoreArtifact, "refused(recalc.unavailable_fit)"),
        (Stage::Law, "refused(recalc.blocked_by_refused_dependency[score_artifact])"),
        (Stage::Decision, "refused(recalc.blocked_by_refused_dependency[law])"),
    ];
    assert_eq!(table(&plan), expected(&v, &overrides));
    for entry in plan.entries() {
        if !entry.stage.is_input() {
            assert!(
                !matches!(entry.status, StageStatus::Reused { .. }),
                "{} claimed reuse in a fresh process",
                entry.stage
            );
        }
    }
    let (_, reason) = plan.first_refusal().unwrap();
    assert_eq!(reason, RefusalReason::Unavailable { missing: MissingDependency::Data });
}

#[test]
fn c2_recalc_supplied_data_resumes_by_recomputing_and_a_missing_callback_blocks_the_decision() {
    let v = full_versions();
    let resume = ResumeContext { supplied_data: true, ..ResumeContext::default() };
    let plan = run(&v, &v, &fresh(RetargetSupport::Licensed, resume));
    let overrides = [
        (Stage::ProviderRequest(b(0)), "refused(recalc.unavailable_provider)"),
        (Stage::ProviderRequest(b(1)), "refused(recalc.unavailable_provider)"),
        (Stage::Prior(b(0)), "recomputed(fresh_process)"),
        (Stage::Prior(b(1)), "recomputed(fresh_process)"),
        (Stage::Identification, "recomputed(fresh_process)"),
        (Stage::ScoreArtifact, "recomputed(fresh_process)"),
        (Stage::Law, "recomputed(fresh_process)"),
        // The first refused edge of the decision in declared order is the missing callback.
        (Stage::Decision, "refused(recalc.blocked_by_refused_dependency[provider_request.0])"),
    ];
    assert_eq!(table(&plan), expected(&v, &overrides));

    let resume =
        ResumeContext { supplied_data: true, supplied_provider: true, ..ResumeContext::default() };
    let plan = run(&v, &v, &fresh(RetargetSupport::Licensed, resume));
    for entry in plan.entries() {
        let derived = !entry.stage.is_input();
        assert_eq!(
            matches!(entry.status, StageStatus::Recomputed { .. }),
            derived,
            "{}",
            entry.stage
        );
    }
    assert!(plan.is_executable());
}

#[test]
fn c2_recalc_portable_scores_resume_without_data_but_never_reuse_the_law_or_decision() {
    let v = native_versions();
    let resume = ResumeContext { portable_scores: true, ..ResumeContext::default() };
    let plan = run(&v, &v, &fresh(RetargetSupport::Licensed, resume));
    let overrides = [
        (Stage::DataSnapshot, "refused(recalc.unavailable_data)"),
        (Stage::Identification, "recomputed(fresh_process)"),
        (Stage::Law, "recomputed(fresh_process)"),
        (Stage::Decision, "recomputed(fresh_process)"),
    ];
    assert_eq!(table(&plan), expected(&v, &overrides));

    // Portable scores do not survive changed folds without data to refit on.
    let changed = bump(&v, Stage::LearnerFoldsRng);
    let plan = run(&v, &changed, &fresh(RetargetSupport::Licensed, resume));
    assert_eq!(
        plan.status(Stage::ScoreArtifact),
        Some(&StageStatus::Refused {
            reason: RefusalReason::Unavailable { missing: MissingDependency::Data }
        })
    );

    // A compatible target change still reweights portable scores.
    let retargeted = bump(&v, Stage::TargetPopulation);
    let plan = run(&v, &retargeted, &fresh(RetargetSupport::Licensed, resume));
    assert_eq!(
        plan.status(Stage::ScoreArtifact),
        Some(&StageStatus::Reused { dependency: Stage::Identification })
    );
    assert_eq!(
        plan.status(Stage::Law),
        Some(&StageStatus::Recomputed {
            because: ChangedDependency::Upstream {
                via: Stage::TargetPopulation,
                origin: Stage::TargetPopulation,
                change: antecedent_core::recalc::ChangeKind::Modified,
            }
        })
    );
}

#[test]
fn c2_recalc_missing_fit_and_snapshot_name_the_specific_dependency() {
    let v = native_versions();
    let cases = [
        (ResumeContext::default(), MissingDependency::Fit),
        (ResumeContext { portable_fit: true, ..ResumeContext::default() }, MissingDependency::Data),
    ];
    for (resume, missing) in cases {
        let plan = run(&v, &v, &fresh(RetargetSupport::Licensed, resume));
        assert_eq!(
            plan.status(Stage::ScoreArtifact),
            Some(&StageStatus::Refused { reason: RefusalReason::Unavailable { missing } })
        );
        assert_eq!(
            plan.status(Stage::DataSnapshot),
            Some(&StageStatus::Refused {
                reason: RefusalReason::Unavailable { missing: MissingDependency::Data }
            })
        );
    }
    // A fit with supplied data is rebuilt, not reused.
    let resume =
        ResumeContext { portable_fit: true, supplied_data: true, ..ResumeContext::default() };
    let plan = run(&v, &v, &fresh(RetargetSupport::Licensed, resume));
    assert_eq!(
        plan.status(Stage::ScoreArtifact),
        Some(&StageStatus::Recomputed { because: ChangedDependency::FreshProcess })
    );
}

#[test]
fn c2_recalc_identity_is_canonical_and_order_independent() {
    assert_ne!(StageIdentity::of("x", &[b"ab", b"c"]), StageIdentity::of("x", &[b"a", b"bc"]));
    assert_ne!(StageIdentity::of("x", &[b"a"]), StageIdentity::of("y", &[b"a"]));
    assert!(!StageIdentity::of("x", &[]).is_absent());
    assert!(StageIdentity::ABSENT.is_absent());
    assert_eq!(StageIdentity::of("x", &[b"a"]).to_hex().len(), 64);

    let v = full_versions();
    let forward = identities(&v);
    let mut backward = StageIdentities::new();
    for (stage, version) in v.iter().rev() {
        backward.set(*stage, StageIdentity::of(&stage.label(), &[version.as_bytes()]));
    }
    assert_eq!(forward, backward);
    let a = RecalcPlan::plan(&forward, &forward, &licensed());
    let c = RecalcPlan::plan(&backward, &backward, &licensed());
    assert_eq!(a.canonical_identity(), c.canonical_identity());
    let changed = run(&v, &bump(&v, Stage::Utility), &licensed());
    assert_ne!(a.canonical_identity(), changed.canonical_identity());
    // The retarget license is part of an effective identity: it changes which stages a target
    // change reaches.
    let licensed_ids = forward.effective(RetargetSupport::Licensed);
    let undeclared_ids = forward.effective(RetargetSupport::NotDeclared);
    assert_eq!(licensed_ids.get(&Stage::Graph), undeclared_ids.get(&Stage::Graph));
    assert_ne!(licensed_ids.get(&Stage::ScoreArtifact), undeclared_ids.get(&Stage::ScoreArtifact));
}

#[test]
fn c2_recalc_branch_indices_are_bounded() {
    assert!(Branch::new(0).is_some());
    assert!(Branch::new(antecedent_core::recalc::MAX_EXTERNAL_BRANCHES).is_none());
    assert_eq!(Branch::new(3).unwrap().index(), 3);
    let boundary = Boundary::InProcess;
    assert_eq!(licensed().boundary, boundary);
}
