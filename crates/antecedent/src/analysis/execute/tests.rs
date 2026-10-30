// SPDX-License-Identifier: MIT OR Apache-2.0
//! Unit tests of the execute module's shared helpers and envelope mixing.

use super::*;

mod class_mixture_mass_tests {
    //! A class mixture's published masses are the envelope's own split.
    use antecedent_identify::{GraphIdentificationCase, IdentificationEnvelope, ProbabilityMass};

    use super::*;

    fn case(status: IdentificationStatus, weight: f64) -> GraphIdentificationCase<u32> {
        GraphIdentificationCase {
            graph: 0,
            result: IdentificationResult::from_parts(
                antecedent_identify::IdentificationResultParts {
                    status,
                    query: CausalQuery::AverageEffect(AverageEffectQuery::binary_ate(
                        VariableId::from_raw(0),
                        VariableId::from_raw(1),
                    )),
                    estimands: Vec::new(),
                    arena: antecedent_expr::CausalExprArena::new(),
                    derivation: DerivationTrace::default(),
                    required_assumptions: antecedent_core::AssumptionSet::default(),
                    diagnostics: Vec::new(),
                    performance: IdentificationPerformanceRecord::default(),
                    hedge: None,
                },
            ),
            weight: ProbabilityMass(weight),
        }
    }

    /// A completion identified only under prior restrictions carries identified
    /// mass — `IdentificationEnvelope::from_cases` counts it in
    /// `identified_weight` — but no frequentist arm estimates it, so it is
    /// unevaluable, not unidentified.
    ///
    /// The temporal class arm used to ask "may this case be estimated?" here
    /// while the envelope diagnostic published beside it asked "does this case
    /// carry identified mass?", so a temporal class result with a
    /// prior-restricted completion reported `unidentified_mass = 0.25` next to
    /// an `identify.temporal_*.envelope` diagnostic saying
    /// `unidentified_mass = 0`. Both now read
    /// [`antecedent_identify::carries_identified_mass`].
    #[test]
    fn prior_restricted_mass_is_unevaluable_not_unidentified() {
        let envelope = IdentificationEnvelope::from_cases(vec![
            case(IdentificationStatus::NonparametricallyIdentified, 0.5),
            case(IdentificationStatus::IdentifiedUnderPriorRestrictions, 0.25),
            case(IdentificationStatus::NotIdentified, 0.25),
        ]);
        // Only the first completion is estimable, so only it has a value.
        let outcomes = [
            ClassAtomOutcome::Evaluated(2.0),
            ClassAtomOutcome::NotEvaluated,
            ClassAtomOutcome::NotEvaluated,
        ];
        let (mixture, facts) = class_structural_mixture(
            &envelope,
            crate::result::StructuralWeightBasis::CompletionEnumeration,
            |index, _| u64::try_from(index).unwrap_or(u64::MAX),
            None,
            &outcomes,
        );
        assert!((facts.total_weight - 1.0).abs() < 1e-12);
        assert!(
            (mixture.unevaluable_mass - 0.25).abs() < 1e-12,
            "prior-restricted mass is identified but unevaluable, got {}",
            mixture.unevaluable_mass
        );
        // The invariant the two predicates used to break: the mixture's split
        // is the envelope's split, so the published mass and the envelope
        // diagnostic on the same result cannot contradict each other.
        assert!(
            (mixture.unidentified_mass - envelope.unidentified_weight.0 / facts.total_weight).abs()
                < 1e-12,
            "mixture unidentified_mass {} contradicts the envelope's {}",
            mixture.unidentified_mass,
            envelope.unidentified_weight.0
        );
        assert!(
            (mixture.identified_mass + mixture.unevaluable_mass
                - envelope.identified_weight.0 / facts.total_weight)
                .abs()
                < 1e-12,
            "identified-and-evaluated plus unevaluable must be the envelope's identified mass"
        );
        let total = mixture.identified_mass
            + mixture.unidentified_mass
            + mixture.unevaluable_mass
            + mixture.subsampled_out_mass;
        assert!((total - 1.0).abs() < 1e-12, "the four masses must conserve, got {total}");
    }
}

mod envelope_se_tests {
    use super::*;

    #[test]
    fn multi_atom_se_requires_joint_sampling_covariance() {
        let se = mix_weighted_analytic_se([(0.5, 2.0), (0.5, 0.0)]);
        assert!(se.is_nan());
        assert!((mix_weighted_analytic_se([(1.0, 2.0)]) - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn mix_weighted_analytic_se_is_nan_if_any_atom_is_nonfinite() {
        assert!(mix_weighted_analytic_se([(0.5, 0.1), (0.5, f64::NAN)]).is_nan());
        assert!(mix_weighted_analytic_se([(1.0, f64::INFINITY)]).is_nan());
    }
}

mod envelope_refuter_target_tests {
    //! Envelope refuters compare each atom with its own estimate.
    use antecedent_discovery::{SamplingSummary, set_edge};
    use antecedent_prob::InferenceDiagnostics;

    use super::*;

    fn known_truth_data(n: usize) -> TabularData {
        let (mut t, mut y, mut z) = (Vec::new(), Vec::new(), Vec::new());
        for _ in 0..(n / 16) {
            for (zv, tv, count) in [(0.0, 0.0, 6), (0.0, 1.0, 2), (1.0, 0.0, 2), (1.0, 1.0, 6)] {
                for row in 0..count {
                    let epsilon = if row % 2 == 0 { -0.2 } else { 0.2 };
                    t.push(tv);
                    z.push(zv);
                    y.push(2.0 * tv + 2.0 * zv + epsilon);
                }
            }
        }
        TabularData::from_f64_columns([
            ("t", t.as_slice()),
            ("y", y.as_slice()),
            ("z", z.as_slice()),
        ])
        .unwrap()
    }

    /// Two identified atoms fit on the same rows: unadjusted (effect 3) and
    /// Z-adjusted (effect 2), at weights 0.5 / 0.3.
    fn fitted_atoms(data: &TabularData, ctx: &ExecutionContext) -> Vec<EnvelopeRefuteAtom> {
        let direct = set_edge(0u64, 3, 0, 1, true);
        let adjusted =
            set_edge(set_edge(set_edge(0u64, 3, 0, 1, true), 3, 2, 0, true), 3, 2, 1, true);
        let gp = GraphPosterior::from_atoms(
            3,
            vec![0.5, 0.3],
            vec![direct, adjusted],
            vec![0.0; 9],
            vec![0.0; 9],
            SamplingSummary {
                ess: 1.0,
                diagnostics: InferenceDiagnostics::analytic("r3_refuter_targets"),
                rejected_invalid: 0,
            },
        )
        .unwrap();
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let identified =
            crate::analysis::prepared::build_graph_posterior_identification_cache(&gp, &query, ctx)
                .unwrap();
        let key_weights = IdentifiedKeyWeights::new(&identified.graphs);
        identified
            .atoms
            .iter()
            .map(|atom| {
                let original = crate::strategy_table::estimate_static_effect(
                    &crate::estimator_spec::EstimatorSpec::Default(
                        EstimatorId::LinearAdjustmentAte,
                    ),
                    StaticEstimateInput {
                        data,
                        estimand: &atom.estimand,
                        query: &query,
                        bootstrap_replicates: 0,
                        overlap_policy: None,
                        population_registry: None,
                        sampling_design: None,
                    },
                    atom.identification.required_assumptions.clone(),
                    ctx,
                    &mut StaticEstimateWorkspaces::default(),
                )
                .unwrap();
                EnvelopeRefuteAtom {
                    key: atom.key,
                    weight: key_weights.get(atom.key),
                    estimand: atom.estimand.clone(),
                    indexer: None,
                    original,
                }
            })
            .collect()
    }

    /// `bootstrap.ci_coverage`'s report for `atoms` against `data`.
    ///
    /// Picked (over `data.subset`) because it is the one refuter in the mixed
    /// suite whose verdict directly compares the *passed-in* `original.ate`
    /// against a CI the refuter builds independently from the atom's own
    /// `estimand`+data (`problem.original.ate >= lo && <= hi` in
    /// `bootstrap_refute::coverage_report`). `data.subset` (and RCC) instead
    /// centre on their own full-sample refit of that same `estimand`
    /// (`fix: make validate refuters, sensitivity and stability checks say
    /// only what they test`), so tampering with `original.ate` alone can no
    /// longer move their verdict — they would pass here regardless of which
    /// target the envelope wiring routed.
    fn bootstrap_report(
        data: &TabularData,
        atoms: &[EnvelopeRefuteAtom],
        ctx: &ExecutionContext,
    ) -> antecedent_validate::RefutationReport {
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let (reports, _) = EnvelopeRefutation {
            data,
            query: &query,
            suite: RefuteSuite::Full,
            estimator: "linear.adjustment.ate",
            custom: &[],
            temporal: None,
        }
        .run(atoms, &mut EstimationWorkspace::default(), ctx)
        .unwrap();
        reports
            .into_iter()
            .find(|r| r.refuter.as_ref() == "bootstrap.ci_coverage")
            .expect("bootstrap.ci_coverage")
    }

    #[test]
    fn envelope_refuters_compare_each_atom_with_its_own_estimate() {
        let ctx = ExecutionContext::for_tests(5);
        let data = known_truth_data(320);
        let atoms = fitted_atoms(&data, &ctx);
        assert_eq!(atoms.len(), 2);
        let effects: Vec<f64> = atoms.iter().map(|a| a.original.ate).collect();
        assert!((effects[0] - effects[1]).abs() > 0.9, "atoms must disagree: {effects:?}");
        let pooled = atoms.iter().map(|a| a.weight * a.original.ate).sum::<f64>()
            / atoms.iter().map(|a| a.weight).sum::<f64>();
        assert!((pooled - 2.625).abs() < 1e-8);

        // Stable atoms with different effects pass against their own estimates;
        // the mixed original_ate is the mass-weighted mean of what was compared.
        let stable = bootstrap_report(&data, &atoms, &ctx);
        assert!(stable.passed, "stable heterogeneous atoms must pass: {stable:?}");
        assert!((stable.original_ate - pooled).abs() < 1e-8);

        // The coverage check is centred on each atom's own full-sample refit, so the
        // estimate an atom reports does not move its verdict: routing every atom to
        // the pooled mixture changes only the published original_ate, and a reported
        // estimate that no refit reproduces is not a coverage failure either.
        let mut pooled_target = fitted_atoms(&data, &ctx);
        for atom in &mut pooled_target {
            atom.original.ate = pooled;
        }
        let pooled_report = bootstrap_report(&data, &pooled_target, &ctx);
        assert!(pooled_report.passed, "{pooled_report:?}");
        assert!((pooled_report.original_ate - pooled).abs() < 1e-8);

        let mut shifted = fitted_atoms(&data, &ctx);
        shifted[1].original.ate += 0.5;
        let shifted_report = bootstrap_report(&data, &shifted, &ctx);
        assert!(shifted_report.passed, "{shifted_report:?}");
        let alone = bootstrap_report(&data, &shifted[..1], &ctx);
        assert!(alone.passed, "{alone:?}");
    }

    #[test]
    fn single_atom_graph_posterior_keeps_its_se_without_omission_diagnostic() {
        // D-4: one contributing atom omits nothing, so the between-atom
        // omission diagnostic must not fire.
        let direct = set_edge(0u64, 3, 0, 1, true);
        let gp = GraphPosterior::from_atoms(
            3,
            vec![1.0],
            vec![direct],
            vec![0.0; 9],
            vec![0.0; 9],
            SamplingSummary {
                ess: 1.0,
                diagnostics: InferenceDiagnostics::analytic("single_atom"),
                rejected_invalid: 0,
            },
        )
        .unwrap();
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let result = Study::tabular(known_truth_data(160))
            .graph_posterior(gp)
            .query(query)
            .refute(RefuteSuite::None)
            .inference(InferenceMode::Frequentist)
            .build()
            .unwrap()
            .run(&ExecutionContext::for_tests(2))
            .unwrap();
        let effect = result.estimate.as_effect().unwrap();
        assert!(effect.se_analytic().is_finite());
        // The only atom is T -> Y with no adjustment; the table's unadjusted contrast is exactly
        // 2 + 2 * (3/4 - 1/4) = 3 (the paired residuals cancel in every cell).
        assert!((effect.ate - 3.0).abs() < 1e-9, "{}", effect.ate);
        assert!(result.diagnostics.iter().all(|d| {
            d.code.as_ref() != "estimate.envelope.se_omits_between_atom_variance"
                && d.code.as_ref() != "estimate.graph_posterior.joint_if_se"
        }));
    }
}

mod support_tests {
    use super::*;

    #[test]
    fn parametric_scm_identification_functional_is_resolvable() {
        let treatment = VariableId::from_raw(0);
        let outcome = VariableId::from_raw(1);
        let anomaly_query = antecedent_core::AnomalyAttributionQuery::new([outcome], 10);
        let query = CausalQuery::AnomalyAttribution(anomaly_query);

        let (identification, estimand) = parametric_scm_identification(query, treatment, outcome);

        // Arena actually has content (the bug was a nil ExprId(0) into an *empty* arena, which
        // ignored `treatment`/`outcome` entirely).
        assert!(!identification.arena.is_empty());

        // `functional` resolves to a real node, not a dangling/out-of-range id.
        let _ = identification.arena.node(estimand.functional);

        // And carries derivation metadata naming the real treatment/outcome.
        let derivation = identification
            .arena
            .derivation(estimand.functional)
            .expect("functional should have derivation metadata");
        assert_eq!(derivation.rule.as_ref(), "gcm.parametric");
        let note = derivation.note.as_deref().unwrap_or_default();
        assert!(note.contains(&format!("{treatment:?}")));
        assert!(note.contains(&format!("{outcome:?}")));

        // The adjustment set stays deliberately empty (GCM doesn't identify via backdoor
        // covariates); this is unchanged behavior, asserted here as a scope guard.
        assert!(estimand.adjustment_set.is_empty());
        assert!(identification.required_assumptions.entries.iter().any(|record| {
            matches!(
                &record.assumption,
                antecedent_core::Assumption::ParametricRestriction(restriction)
                    if restriction.id.as_ref() == "gcm.supplied_structural_mechanisms"
            ) && record.scope == antecedent_core::AssumptionScope::Identification
        }));
    }

    #[test]
    fn graph_envelope_unions_assumptions_from_every_identified_case() {
        fn case_result(
            query: &AverageEffectQuery,
            status: IdentificationStatus,
            assumption_id: &'static str,
        ) -> IdentificationResult {
            let mut assumptions = antecedent_core::AssumptionSet::new();
            assumptions.push(antecedent_core::AssumptionRecord {
                assumption: antecedent_core::Assumption::Custom {
                    id: Arc::from(assumption_id),
                    description: Arc::from("case-specific restriction"),
                },
                source: antecedent_core::AssumptionSource::AlgorithmDefault {
                    algorithm: Arc::from("test.case"),
                },
                scope: antecedent_core::AssumptionScope::Identification,
                status: antecedent_core::AssumptionStatus::Declared,
            });
            IdentificationResult::from_parts(antecedent_identify::IdentificationResultParts {
                status,
                query: CausalQuery::AverageEffect(query.clone()),
                estimands: Vec::new(),
                arena: CausalExprArena::new(),
                derivation: DerivationTrace::default(),
                required_assumptions: assumptions,
                diagnostics: Vec::new(),
                performance: IdentificationPerformanceRecord::default(),
                hedge: None,
            })
        }

        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let envelope = IdentificationEnvelope::from_cases(vec![
            antecedent_identify::GraphIdentificationCase {
                graph: Pag::with_variables(2),
                result: case_result(
                    &query,
                    IdentificationStatus::IdentifiedUnderParametricRestrictions,
                    "case.zero",
                ),
                weight: antecedent_identify::ProbabilityMass(0.5),
            },
            antecedent_identify::GraphIdentificationCase {
                graph: Pag::with_variables(2),
                result: case_result(
                    &query,
                    IdentificationStatus::IdentifiedUnderPriorRestrictions,
                    "case.one",
                ),
                weight: antecedent_identify::ProbabilityMass(0.5),
            },
        ]);
        let result = envelope_to_identification_result(&envelope, &query);
        assert_eq!(result.status, IdentificationStatus::IdentifiedUnderPriorRestrictions);
        for expected in ["case.zero", "case.one"] {
            assert!(result.required_assumptions.entries.iter().any(|record| {
                matches!(&record.assumption, antecedent_core::Assumption::Custom { id, .. } if id.as_ref() == expected)
            }));
        }
    }
}

mod identify_only_tests {
    use antecedent_core::{
        AverageEffectQuery, CausalQuery, Intervention, InterventionalDistributionQuery, Value,
        VariableId,
    };
    use antecedent_data::TabularData;
    use antecedent_discovery::SamplingSummary;
    use antecedent_discovery::{mask_is_dag, set_edge};
    use antecedent_graph::{Admg, DenseNodeId, Pag};
    use antecedent_prob::InferenceDiagnostics;

    use super::*;
    use crate::support::SupportRefusal;

    fn toy_data() -> TabularData {
        TabularData::from_f64_columns([("t", &[0.0_f64, 1.0][..]), ("y", &[0.0_f64, 1.0][..])])
            .unwrap()
    }

    fn ate() -> AverageEffectQuery {
        AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
    }

    fn known_truth_graph_mixture_data(n: usize) -> TabularData {
        // Sixteen rows per block give the exact joint distribution
        // P(T=1 | Z=0)=1/4 and P(T=1 | Z=1)=3/4.  Hence the unadjusted
        // contrast is 2 + 2*(3/4 - 1/4) = 3, while adjustment for Z
        // recovers the structural coefficient 2.  Pairing +/- residuals
        // within every cell keeps both population regressions exact.
        assert_eq!(n % 16, 0);
        let mut treatment = Vec::with_capacity(n);
        let mut outcome = Vec::with_capacity(n);
        let mut confounder = Vec::with_capacity(n);
        for _ in 0..(n / 16) {
            for (z, t, count) in [(0.0, 0.0, 6), (0.0, 1.0, 2), (1.0, 0.0, 2), (1.0, 1.0, 6)] {
                for row in 0..count {
                    let epsilon = if row % 2 == 0 { -0.2 } else { 0.2 };
                    treatment.push(t);
                    confounder.push(z);
                    outcome.push(2.0 * t + 2.0 * z + epsilon);
                }
            }
        }
        TabularData::from_f64_columns([
            ("t", treatment.as_slice()),
            ("y", outcome.as_slice()),
            ("z", confounder.as_slice()),
        ])
        .unwrap()
    }

    /// Records progress labels so the test can prove identification is computed
    /// on fresh runs and prepare only, never on a prepared estimate/refresh click.
    #[derive(Default)]
    struct RecordingProgress(std::sync::Mutex<Vec<String>>);

    impl antecedent_core::ProgressSink for RecordingProgress {
        fn report(&self, _fraction: f64, stage: &str) {
            self.0.lock().unwrap().push(stage.to_owned());
        }
    }

    struct CancelOnEnvelopeIdentify {
        token: antecedent_core::CancellationToken,
    }

    impl antecedent_core::ProgressSink for CancelOnEnvelopeIdentify {
        fn report(&self, _fraction: f64, stage: &str) {
            if stage == "envelope.identify" {
                self.token.cancel();
            }
        }
    }

    fn recording_ctx(seed: u64) -> (ExecutionContext, std::sync::Arc<RecordingProgress>) {
        let sink = std::sync::Arc::new(RecordingProgress::default());
        let mut ctx = ExecutionContext::for_tests(seed);
        ctx.progress =
            Some(std::sync::Arc::clone(&sink) as std::sync::Arc<dyn antecedent_core::ProgressSink>);
        (ctx, sink)
    }

    fn identify_computations(sink: &RecordingProgress) -> usize {
        sink.0
            .lock()
            .unwrap()
            .iter()
            .filter(|stage| stage.as_str() == crate::analysis::stage::PROGRESS_IDENTIFY_COMPUTE)
            .count()
    }

    fn cancel_on_envelope_identify_ctx(seed: u64) -> ExecutionContext {
        let mut ctx = ExecutionContext::for_tests(seed);
        let token = ctx.cancellation.clone();
        ctx.progress = Some(std::sync::Arc::new(CancelOnEnvelopeIdentify { token }));
        ctx
    }

    fn cached_count(result: &StudyResult) -> usize {
        result.diagnostics.iter().filter(|d| d.code.as_ref() == "exec.identify.cached").count()
    }

    fn assert_identify_only_refused(err: &CausalError, message: &'static str) {
        assert!(
            matches!(
                err,
                CausalError::Support { id: SupportRefusal::Refused, message: m } if *m == message
            ),
            "{err:?}"
        );
        assert_eq!(err.to_string(), format!("refused: {message}"));
    }

    #[test]
    fn identify_only_refuses_graph_posterior() {
        // graph_posterior x Frequentist is now closed at `build()`, so reach
        // identify_only's own guard through the still-open Bayesian cell.
        let gp = GraphPosterior::from_atoms(
            2,
            vec![1.0],
            vec![0u64],
            vec![0.0; 4],
            vec![0.0; 4],
            SamplingSummary {
                ess: 1.0,
                diagnostics: InferenceDiagnostics::analytic("test"),
                rejected_invalid: 0,
            },
        )
        .unwrap();
        let err = Study::tabular(toy_data())
            .graph_posterior(gp)
            .query(ate())
            .refute(RefuteSuite::None)
            .inference(InferenceMode::Bayesian(BayesianConfig::conjugate()))
            .build()
            .unwrap()
            .identify_only()
            .unwrap_err();
        assert_identify_only_refused(
            &err,
            "identify_only is not a graph-posterior cell; identification \
                          runs per-graph inside execute.",
        );
    }

    #[test]
    #[allow(
        clippy::too_many_lines,
        reason = "fixture of two disagreeing atoms, one Bayesian run and the pinned scalar, SE, mixture and identified-set assertions"
    )]
    fn graph_posterior_ate_known_truth_mixture() {
        // Analytic fixture: atom 0 estimates the unadjusted effect 3, atom 1
        // adjusts for Z and estimates 2, and atom 2 is the reverse-causal
        // Y -> T DAG with no admissible adjustment (weight 0.2). Disagreeing
        // estimands withhold the scalar; posterior BMA and structural_response
        // still publish P(tau | identified) and the identified set [2, 3].
        let expected: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../../conformance/bayesian/known_truth_mixtures/expected.json"
        ))
        .unwrap();
        let pin = &expected["static_average_effect"];
        let n = usize::try_from(pin["n"].as_u64().unwrap()).unwrap();
        let weights: Vec<f64> = pin["posterior_weights"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let atom_effects: Vec<f64> = pin["identified_atom_effects"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let identified_mass = pin["identified_mass"].as_f64().unwrap();
        let weighted_sum = pin["identified_weighted_sum"].as_f64().unwrap();
        let mixture_truth = pin["expected_effect_given_identified"].as_f64().unwrap();
        let unidentified_truth = pin["expected_unidentified_mass"].as_f64().unwrap();
        let tolerance = pin["effect_abs_tolerance"].as_f64().unwrap();
        assert!(
            (weights[0] * atom_effects[0] + weights[1] * atom_effects[1] - weighted_sum).abs()
                < 1e-12
        );
        assert!((weighted_sum / identified_mass - mixture_truth).abs() < 1e-12);
        assert!((weights[2] - unidentified_truth).abs() < 1e-12);

        let direct = set_edge(0u64, 3, 0, 1, true); // T -> Y; no adjustment.
        let adjusted =
            set_edge(set_edge(set_edge(0u64, 3, 0, 1, true), 3, 2, 0, true), 3, 2, 1, true); // Z -> T, Z -> Y, T -> Y.
        let unidentified = set_edge(0u64, 3, 1, 0, true); // Y -> T; no admissible adjustment.
        assert!(mask_is_dag(direct, 3));
        assert!(mask_is_dag(adjusted, 3));
        assert!(mask_is_dag(unidentified, 3));

        // Keep marginals consistent with the three frozen atoms even though
        // effect execution consumes their masks and weights directly.
        let mut marginals = vec![0.0; 9];
        marginals[1] = weights[0] + weights[1]; // T -> Y in both identified atoms.
        marginals[3] = weights[2]; // Y -> T only in the unidentified atom.
        marginals[6] = weights[1]; // Z -> T only in the adjusted atom.
        marginals[7] = weights[1]; // Z -> Y only in the adjusted atom.

        // Cover each licensed validation coordinate so cheap/full support is
        // demonstrated by the same known-truth staged mixture, not inferred
        // from a validation-none run.
        let mut report_counts = Vec::new();
        for suite in [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full] {
            let gp = GraphPosterior::from_atoms(
                3,
                weights.clone(),
                vec![direct.clone(), adjusted.clone(), unidentified.clone()],
                marginals.clone(),
                marginals.clone(),
                SamplingSummary {
                    ess: 1.0 / weights.iter().map(|w| w * w).sum::<f64>(),
                    diagnostics: InferenceDiagnostics::analytic("known_truth_mixtures"),
                    rejected_invalid: 0,
                },
            )
            .unwrap()
            .with_algorithm("known_truth_fixture");
            let data = known_truth_graph_mixture_data(n);
            let (ctx, sink) = recording_ctx(1);
            let study = Study::tabular(data.clone())
                .graph_posterior(gp)
                .query(ate())
                .refute(suite)
                .inference(InferenceMode::Bayesian(
                    BayesianConfig::conjugate().n_draws(256).prior_scale(1_000_000.0),
                ))
                .build()
                .unwrap();
            let fresh = study.clone().run(&ctx).unwrap();
            assert_eq!(identify_computations(&sink), 1, "a fresh run identifies its atoms once");
            let mut prepared = study.prepare(&ctx).unwrap();
            assert_eq!(identify_computations(&sink), 2, "prepare identifies the atoms once");
            let click = prepared.estimate(&data, &ctx).unwrap();
            let refreshed =
                crate::analysis::prepared::rebind_estimate(&mut prepared, data.clone(), &ctx)
                    .unwrap();
            assert_eq!(
                identify_computations(&sink),
                2,
                "prepared estimate and refresh clicks must not re-identify"
            );
            assert!(
                !click.estimate.as_effect().unwrap().ate.is_finite(),
                "{suite:?} scalar ate must be withheld when estimands disagree"
            );
            assert!(!fresh.estimate.as_effect().unwrap().ate.is_finite());
            assert!(!refreshed.estimate.as_effect().unwrap().ate.is_finite());
            assert_eq!(click.support_status.unwrap().as_str(), "licensed");
            let click_post = click.posterior.as_ref().expect("prepared graph mixture posterior");
            let eq = click_post.effect_column().expect("posterior effect column");
            assert!(
                (click_post.summaries.mean[eq] - mixture_truth).abs() < tolerance,
                "BMA mean={} fixture aggregate={mixture_truth}",
                click_post.summaries.mean[eq]
            );
            let structural = click.structural_response.as_ref().expect("structural mixture");
            assert!(structural.conditional_on_identified.is_none());
            let identified_set = structural.identified_set.as_ref().expect("identified set");
            assert!(
                (identified_set.lower[0] - atom_effects[1]).abs() < tolerance
                    && (identified_set.upper[0] - atom_effects[0]).abs() < tolerance,
                "identified set [{}, {}] truth [{}, {}]",
                identified_set.lower[0],
                identified_set.upper[0],
                atom_effects[1],
                atom_effects[0]
            );
            assert!(
                click.diagnostics.iter().any(|d| {
                    d.code.as_ref() == "estimate.graph_posterior.structural_aggregation"
                        && d.message.contains("graph_dependent_atoms")
                }),
                "{suite:?} must disclose GraphDependentAtoms"
            );
            let fresh_post = fresh.posterior.as_ref().expect("fresh graph mixture posterior");
            let refreshed_post =
                refreshed.posterior.as_ref().expect("refreshed graph mixture posterior");
            assert_eq!(click_post.identification, IdentificationStatus::GraphDependent);
            assert!((click_post.unidentified_mass - unidentified_truth).abs() < 1e-12);
            assert!((fresh_post.unidentified_mass - unidentified_truth).abs() < 1e-12);
            assert!((refreshed_post.unidentified_mass - unidentified_truth).abs() < 1e-12);
            assert_eq!(
                cached_count(&fresh),
                0,
                "fresh graph-posterior execution must identify its atoms"
            );
            assert_eq!(
                cached_count(&click),
                1,
                "prepared graph-posterior execution must consume its cache exactly once"
            );
            assert_eq!(cached_count(&refreshed), 1, "same-schema refresh must reuse the cache");
            assert_eq!(click.refutations.len(), fresh.refutations.len());
            assert_eq!(click.predictive_checks.len(), fresh.predictive_checks.len());
            if matches!(suite, RefuteSuite::Full) {
                assert!(
                    click.diagnostics.iter().any(|d| {
                        d.code.as_ref() == "refute.bayesian.ppc.envelope"
                            && d.message.contains("prior-sensitivity")
                    }),
                    "full validation must run per-atom prior sensitivity"
                );
                assert!(
                    click_post.prior_sensitivity.is_none(),
                    "GraphDependentAtoms withhold a mixed prior-sensitivity object"
                );
            } else {
                assert!(click_post.prior_sensitivity.is_none());
            }
            match suite {
                RefuteSuite::None => {
                    assert!(click.predictive_checks.is_empty(), "validation none must not run PPC");
                }
                RefuteSuite::Cheap | RefuteSuite::PlaceboAndRcc | RefuteSuite::Full => {
                    assert!(
                        click
                            .predictive_checks
                            .iter()
                            .any(|c| { c.kind == antecedent_validate::PredictiveCheckKind::Prior }),
                        "graph-posterior {suite:?} must attach mixture-weighted prior PPC"
                    );
                    assert!(
                        click.predictive_checks.iter().any(|c| {
                            c.kind == antecedent_validate::PredictiveCheckKind::Posterior
                        }),
                        "graph-posterior {suite:?} must attach mixture-weighted posterior PPC"
                    );
                    assert!(
                        click.diagnostics.iter().any(|d| {
                            d.code.as_ref() == "refute.envelope.graph_dependent_atoms"
                        }),
                        "graph-posterior {suite:?} must retain per-atom refuters when scalar is withheld"
                    );
                    assert!(
                        click
                            .diagnostics
                            .iter()
                            .all(|d| { d.code.as_ref() != "refute.envelope.effect_mixture" }),
                        "outer scalar mix must not run under GraphDependentAtoms"
                    );
                }
            }
            report_counts.push(click.refutations.len());
        }
        assert_eq!(report_counts[0], 0, "validation none must emit no reports");
        assert!(report_counts[1] > 0, "cheap validation must execute a refuter");
        assert!(
            report_counts[2] > report_counts[1],
            "full validation must add prior-sensitivity reports beyond the cheap suite"
        );
    }

    #[test]
    #[allow(
        clippy::too_many_lines,
        reason = "the frequentist twin of the Bayesian mixture pin: same disagreeing atoms, one run and its pinned assertions"
    )]
    fn graph_posterior_ate_known_truth_mixture_frequentist() {
        // Same disagreeing atoms as the Bayesian pin: scalar ate and SE are
        // withheld under GraphDependentAtoms; structural_response publishes
        // the identified set and per-atom values.
        let expected: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../../conformance/bayesian/known_truth_mixtures/expected.json"
        ))
        .unwrap();
        let pin = &expected["static_average_effect"];
        let n = usize::try_from(pin["n"].as_u64().unwrap()).unwrap();
        let weights: Vec<f64> = pin["posterior_weights"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let unidentified_truth = pin["expected_unidentified_mass"].as_f64().unwrap();

        let direct = set_edge(0u64, 3, 0, 1, true);
        let adjusted =
            set_edge(set_edge(set_edge(0u64, 3, 0, 1, true), 3, 2, 0, true), 3, 2, 1, true);
        let unidentified = set_edge(0u64, 3, 1, 0, true);
        let mut marginals = vec![0.0; 9];
        marginals[1] = weights[0] + weights[1];
        marginals[3] = weights[2];
        marginals[6] = weights[1];
        marginals[7] = weights[1];

        let mut report_counts = Vec::new();
        for suite in [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full] {
            let gp = GraphPosterior::from_atoms(
                3,
                weights.clone(),
                vec![direct.clone(), adjusted.clone(), unidentified.clone()],
                marginals.clone(),
                marginals.clone(),
                SamplingSummary {
                    ess: 1.0 / weights.iter().map(|w| w * w).sum::<f64>(),
                    diagnostics: InferenceDiagnostics::analytic("known_truth_mixtures"),
                    rejected_invalid: 0,
                },
            )
            .unwrap()
            .with_algorithm("known_truth_fixture");
            let data = known_truth_graph_mixture_data(n);
            let (ctx, sink) = recording_ctx(1);
            let study = Study::tabular(data.clone())
                .graph_posterior(gp)
                .query(ate())
                .refute(suite)
                .inference(InferenceMode::Frequentist)
                .build()
                .unwrap();
            let fresh = study.clone().run(&ctx).unwrap();
            assert_eq!(identify_computations(&sink), 1, "a fresh run identifies its atoms once");
            let mut prepared = study.prepare(&ctx).unwrap();
            assert_eq!(identify_computations(&sink), 2, "prepare identifies the atoms once");
            let click = prepared.estimate(&data, &ctx).unwrap();
            let refreshed =
                crate::analysis::prepared::rebind_estimate(&mut prepared, data.clone(), &ctx)
                    .unwrap();
            assert_eq!(
                identify_computations(&sink),
                2,
                "prepared estimate and refresh clicks must not re-identify"
            );
            assert!(
                !click.estimate.as_effect().unwrap().ate.is_finite(),
                "{suite:?} scalar ate must be withheld when estimands disagree"
            );
            assert!(!click.estimate.as_effect().unwrap().se_analytic().is_finite());
            assert!(!fresh.estimate.as_effect().unwrap().ate.is_finite());
            assert!(!refreshed.estimate.as_effect().unwrap().ate.is_finite());
            assert_eq!(click.support_status.unwrap().as_str(), "licensed");
            assert!(click.posterior.is_none(), "Frequentist mixture must not attach a posterior");
            assert_eq!(click.identification.status, IdentificationStatus::GraphDependent);
            assert_eq!(fresh.identification.status, IdentificationStatus::GraphDependent);
            let structural = click.structural_response.as_ref().expect("structural mixture");
            assert!(structural.conditional_on_identified.is_none());
            let identified_set = structural.identified_set.as_ref().expect("identified set");
            assert!(
                (identified_set.lower[0] - 2.0).abs() < 1e-8
                    && (identified_set.upper[0] - 3.0).abs() < 1e-8,
                "identified set [{}, {}]",
                identified_set.lower[0],
                identified_set.upper[0]
            );
            assert!(
                click.diagnostics.iter().any(|d| {
                    d.code.as_ref() == "estimate.graph_posterior.structural_aggregation"
                        && d.message.contains("graph_dependent_atoms")
                }),
                "{suite:?} must disclose GraphDependentAtoms"
            );
            assert!(
                fresh.diagnostics.iter().any(|d| {
                    d.code.as_ref() == "estimate.graph_posterior.envelope"
                        && d.message.contains(&format!("unidentified_mass={unidentified_truth}"))
                }),
                "fresh Frequentist mixture must retain unidentified mass"
            );
            assert!(
                fresh
                    .diagnostics
                    .iter()
                    .all(|d| d.code.as_ref() != "estimate.graph_posterior.joint_if_se"),
                "joint-IF SE is only published under SameEstimandWeightedMean"
            );
            // The one-shot facade now prepares the sealed graph-posterior plan,
            // so its per-atom identification is reused during execution.
            assert_eq!(cached_count(&fresh), 1);
            assert_eq!(cached_count(&click), 1);
            assert_eq!(cached_count(&refreshed), 1);
            assert_eq!(click.refutations.len(), fresh.refutations.len());
            match suite {
                RefuteSuite::None => {
                    assert!(click.refutations.is_empty(), "validation none must emit no reports");
                }
                RefuteSuite::Cheap | RefuteSuite::PlaceboAndRcc | RefuteSuite::Full => {
                    assert!(!click.refutations.is_empty(), "{suite:?} must execute a refuter");
                    assert!(
                        click
                            .diagnostics
                            .iter()
                            .any(|d| { d.code.as_ref() == "refute.envelope.class_posterior" }),
                        "Frequentist {suite:?} must retain per-atom refuters when scalar is withheld"
                    );
                    assert!(
                        click
                            .diagnostics
                            .iter()
                            .all(|d| { d.code.as_ref() != "refute.envelope.effect_mixture" }),
                        "outer scalar mix must not run under GraphDependentAtoms"
                    );
                }
            }
            report_counts.push(click.refutations.len());
        }
        assert_eq!(report_counts[0], 0, "validation none must emit no reports");
        assert!(report_counts[1] > 0, "cheap validation must execute a refuter");
        assert!(
            report_counts[2] >= report_counts[1],
            "full validation must not drop the cheap refuters"
        );
    }

    fn cheap_overlap_comparison(gp: GraphPosterior, n: usize, frequentist: bool) -> f64 {
        let mut builder = Study::tabular(known_truth_graph_mixture_data(n))
            .graph_posterior(gp)
            .query(ate())
            .refute(RefuteSuite::Cheap);
        builder = if frequentist {
            builder.inference(InferenceMode::Frequentist)
        } else {
            builder.inference(InferenceMode::Bayesian(
                BayesianConfig::conjugate().n_draws(256).prior_scale(1_000_000.0),
            ))
        };
        let result = builder.build().unwrap().run(&ExecutionContext::for_tests(3)).unwrap();
        result
            .refutations
            .iter()
            .find(|r| r.refuter.as_ref() == "overlap.assessment")
            .expect("cheap suite must emit overlap.assessment")
            .comparison
    }

    #[test]
    fn graph_posterior_overlap_withheld_when_estimands_disagree_bayesian() {
        // Distinct adjustment sets are GraphDependentAtoms. Cheap overlap is
        // the first contributing atom, Bayesian and Frequentist alike.
        let n = 64;
        let direct = set_edge(0u64, 3, 0, 1, true);
        let adjusted =
            set_edge(set_edge(set_edge(0u64, 3, 0, 1, true), 3, 2, 0, true), 3, 2, 1, true);
        let unidentified = set_edge(0u64, 3, 1, 0, true);
        let mix = GraphPosterior::from_atoms(
            3,
            vec![0.5, 0.3, 0.2],
            vec![direct.clone(), adjusted.clone(), unidentified.clone()],
            vec![0.0; 9],
            vec![0.0; 9],
            SamplingSummary {
                ess: 1.0,
                diagnostics: InferenceDiagnostics::analytic("overlap_mixture"),
                rejected_invalid: 0,
            },
        )
        .unwrap();
        let first_only = GraphPosterior::from_atoms(
            3,
            vec![1.0],
            vec![direct],
            vec![0.0; 9],
            vec![0.0; 9],
            SamplingSummary {
                ess: 1.0,
                diagnostics: InferenceDiagnostics::analytic("overlap_first_atom"),
                rejected_invalid: 0,
            },
        )
        .unwrap();
        let mixed = cheap_overlap_comparison(mix, n, false);
        let first = cheap_overlap_comparison(first_only, n, false);
        assert!(
            (mixed - first).abs() < 1e-9,
            "GraphDependentAtoms overlap comparison={mixed} must match first atom={first}"
        );
    }

    #[test]
    fn frequentist_graph_posterior_overlap_withheld_when_estimands_disagree() {
        // Under GraphDependentAtoms the cheap overlap report comes from the
        // first contributing atom only, not a mass-weighted mix.
        let n = 64;
        let direct = set_edge(0u64, 3, 0, 1, true);
        let adjusted =
            set_edge(set_edge(set_edge(0u64, 3, 0, 1, true), 3, 2, 0, true), 3, 2, 1, true);
        let unidentified = set_edge(0u64, 3, 1, 0, true);
        let mix = GraphPosterior::from_atoms(
            3,
            vec![0.5, 0.3, 0.2],
            vec![direct.clone(), adjusted.clone(), unidentified.clone()],
            vec![0.0; 9],
            vec![0.0; 9],
            SamplingSummary {
                ess: 1.0,
                diagnostics: InferenceDiagnostics::analytic("overlap_mixture"),
                rejected_invalid: 0,
            },
        )
        .unwrap();
        let first_only = GraphPosterior::from_atoms(
            3,
            vec![1.0],
            vec![direct],
            vec![0.0; 9],
            vec![0.0; 9],
            SamplingSummary {
                ess: 1.0,
                diagnostics: InferenceDiagnostics::analytic("overlap_first_atom"),
                rejected_invalid: 0,
            },
        )
        .unwrap();
        let mixed = cheap_overlap_comparison(mix, n, true);
        let first = cheap_overlap_comparison(first_only, n, true);
        assert!(
            (mixed - first).abs() < 1e-9,
            "GraphDependentAtoms overlap comparison={mixed} must match first atom={first}"
        );
    }

    #[test]
    fn fresh_graph_posterior_cancelled_during_identification_returns_typed_error() {
        let direct = set_edge(0u64, 3, 0, 1, true);
        let adjusted =
            set_edge(set_edge(set_edge(0u64, 3, 0, 1, true), 3, 2, 0, true), 3, 2, 1, true);
        let gp = GraphPosterior::from_atoms(
            3,
            vec![0.5, 0.5],
            vec![direct, adjusted],
            vec![0.0; 9],
            vec![0.0; 9],
            SamplingSummary {
                ess: 2.0,
                diagnostics: InferenceDiagnostics::analytic("cancel_during_identification"),
                rejected_invalid: 0,
            },
        )
        .unwrap();
        let ctx = cancel_on_envelope_identify_ctx(17);
        let error = Study::tabular(known_truth_graph_mixture_data(16))
            .graph_posterior(gp)
            .query(ate())
            .refute(RefuteSuite::None)
            .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(16)))
            .build()
            .unwrap()
            .run(&ctx)
            .expect_err("identification cancellation must not become unidentified mass");

        assert!(
            matches!(error, CausalError::Cancelled { stage } if stage == crate::analysis::stage::STAGE_IDENTIFY),
            "expected identify-stage cancellation, got {error:?}"
        );
    }

    #[test]
    fn bidirected_admg_distribution_cheap_builds_as_a_licensed_functional_refuter_state() {
        // The mixed-graph distribution and its cheap refuter state are licensed.
        let mut admg = Admg::with_variables(2);
        admg.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        admg.insert_bidirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let query = InterventionalDistributionQuery::new(
            VariableId::from_raw(1),
            [Intervention::set(VariableId::from_raw(0), Value::f64(1.0))],
        );
        let study = Study::tabular(toy_data())
            .graph(admg)
            .query(CausalQuery::Distribution(query))
            .refute(RefuteSuite::Cheap)
            .build()
            .unwrap();
        assert_eq!(study.support_status, Some(crate::support::CellStatus::Licensed));
    }

    #[test]
    fn identify_only_refuses_non_dag_admg_graph() {
        let mut pag = Pag::with_variables(2);
        pag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let err = Study::tabular(toy_data())
            .graph(pag)
            .query(ate())
            .refute(RefuteSuite::None)
            .build()
            .unwrap()
            .identify_only()
            .unwrap_err();
        assert_identify_only_refused(
            &err,
            "identify_only supports static DAG and ADMG graphs only.",
        );
    }
}

mod envelope_validation_diagnostic_tests {
    use super::*;

    #[test]
    fn prior_sensitivity_is_claimed_only_when_it_mixed() {
        let mixed = envelope_validation_diagnostics("a,b", Some(true));
        assert_eq!(mixed.len(), 1);
        assert!(mixed[0].message.contains("prior-sensitivity evaluated"));
        let unmixed = envelope_validation_diagnostics("a,b", Some(false));
        assert!(!unmixed[0].message.contains("prior-sensitivity evaluated"), "{unmixed:?}");
        let warning = unmixed
            .iter()
            .find(|d| d.code.as_ref() == "refute.bayesian.prior_sensitivity.not_mixed")
            .expect("not-mixed warning");
        assert_eq!(warning.severity, DiagnosticSeverity::Warning);
        let cheap = envelope_validation_diagnostics("a,b", None);
        assert_eq!(cheap.len(), 1);
        assert!(!cheap[0].message.contains("prior-sensitivity"));
    }
}

mod identify_cached_diagnostic_tests {
    //! The cached-identification diagnostics share one code and differ only in wording.
    use super::*;

    #[test]
    fn both_wordings_share_the_cached_code() {
        let prepared = identify_cached_diagnostic();
        let checked = identify_cached_checked_operation_diagnostic();
        assert_eq!(prepared.code.as_ref(), IDENTIFY_CACHED_CODE);
        assert_eq!(checked.code.as_ref(), IDENTIFY_CACHED_CODE);
        assert_eq!(prepared.severity, DiagnosticSeverity::Info);
        assert_eq!(checked.severity, DiagnosticSeverity::Info);
        assert!(prepared.message.contains("prepare-time cache"));
        assert!(checked.message.contains("checked prepared operation"));
    }
}
