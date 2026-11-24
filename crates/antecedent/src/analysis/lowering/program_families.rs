//! The compiled program families that lower outside the route matrix, as one table.
//!
//! One row per family: its name, the [`RouteFamily`](crate::planner::RouteFamily) that
//! lowers it, and the support unit key its support status is read under (an empty
//! string until the family is keyed to a unit). `program_families!` hands every row to
//! a generator macro, and the generators build from it the `PlanRoute` variants with
//! their stable names and families (`registry`), the `ProgramCell` variants with their
//! stable names, plan routes and every exhaustive match over them (`planner::lowered`).
//! Adding a family is one row here.
//!
//! A row's name is its identity: `ProgramCell::as_str` and `PlanRoute::as_str` write it
//! into claim ids, contract seals and evidence manifests. A row is never renamed, and
//! rows keep their order.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

/// Expand `$generator! { [$context] rows }` over every program-family row.
///
/// A row is `doc name => RouteFamily variant, unit "support unit key";`, the doc being
/// the plan route's public doc. `$context` is passed through first, in brackets, so a
/// generator can join the rows to a table of its own.
macro_rules! program_families {
    ($generator:ident $([$($context:tt)*])?) => {
        $generator! {
            [$($($context)*)?]
            /// Complete observational policy learning and scoring.
            ObservationalPolicy => DesignBased, unit "estimate.observational_policy_value";
            /// Joint measured-signal observation of a composed finite model.
            JointMeasuredObservation => Structural, unit "observation_query.finite_law_conditioning";
            /// Dependent-error measurement of one finite construct.
            DependentMeasuredObservation => Structural, unit "observation_query.finite_law_conditioning";
            /// Fitted dependent-error measurement with posterior draws.
            FittedDependentMeasuredObservation => Structural, unit "observation_query.finite_law_conditioning";
            /// Model-derived paired intervention worlds and a decision.
            CyclicActionWorldDecision => Cyclic, unit "cyclic.action_world_decisions";
            /// Joint fitted-mechanism posterior predictive action worlds.
            CyclicPosteriorActionDecision => Cyclic, unit "cyclic.action_world_decisions";
            /// Target-population feedback standardization over a cyclic fit and a decision.
            CyclicTargetFeedbackDecision => Cyclic, unit "cyclic.action_world_decisions";
            /// Checked bridge transport and predictive utility.
            BridgeTransportDecision => Structural, unit "transport.bridge_decision";
            /// Premise assessment of a checked bridge transport.
            BridgeTransportPremiseAssessment => Structural, unit "transport.bridge_premise_assessment";
            /// Matched bridge validation and conditional predictive utility.
            AssessedBridgeTransportDecision => Structural, unit "transport.bridge_decision";
            /// Fitted source-target effect posterior and utility.
            JointTransportEffectDecision => Structural, unit "transport.joint_transport_decision";
            /// Trial inverse-probability-weighted transport and a decision.
            TrialIpwTransportDecision => Structural, unit "transport.trial_ipw_decision";
            /// Trial inverse-probability-weighted predictive transport and a decision.
            TrialIpwPredictiveDecision => Structural, unit "transport.trial_ipw_decision";
            /// Joint source-target predictive transport and a decision.
            JointTransportPredictiveDecision => Structural, unit "transport.joint_transport_decision";
            /// Target-population time-ordered path-specific transport.
            TemporalPathTransport => Temporal, unit "transport.temporal_path_specific";
            /// Whole-chain resampling union around an assumption envelope.
            SamplingAssumptionUnion => PropagationChain, unit "composition.sampling_assumption_union";
            /// Whole-chain resampling union with independent and identically distributed units.
            IidSamplingAssumptionUnion => PropagationChain, unit "composition.sampling_assumption_union";
            /// Compiler-generated whole-cluster bootstrap assumption union.
            ClusterSamplingAssumptionUnion => PropagationChain, unit "composition.sampling_assumption_union";
            /// Whole-chain resampling union with moving-block resampling.
            MovingBlockSamplingAssumptionUnion => PropagationChain, unit "composition.sampling_assumption_union";
            /// An observational policy value under m-out-of-n full-refit subsampling.
            PolicySubsampling => DesignBased, unit "estimate.observational_policy_value";
            /// Sequential policy evaluation with a reliability diagnostic.
            SequentialPolicyReliability => DesignBased, unit "decision_reliability.sampled_decision";
            /// Replay a bundle with a compiler-executed terminal.
            BundleRecomputation => PropagationChain, unit "composition.bundle";
            /// Bundle replay with a live scientific bridge terminal.
            BridgeBundleRecomputation => PropagationChain, unit "composition.bundle";
            /// Bundle replay with a refitted cyclic posterior producer.
            CyclicPosteriorBundleRecomputation => PropagationChain, unit "composition.bundle";
            /// Bundle replay with an embedded cyclic posterior fit node.
            CyclicPosteriorNodeBundleRecomputation => PropagationChain, unit "composition.bundle";
            /// Bundle replay with an embedded trial-IPW fit node.
            TrialIpwNodeBundleRecomputation => PropagationChain, unit "composition.bundle";
            /// Bundle replay with a graph-imputation transport fit.
            GraphImputationTransportNodeBundleRecomputation => PropagationChain, unit "composition.bundle";
            /// Bundle replay with a joint source-target effect fit.
            JointTransportEffectNodeBundleRecomputation => PropagationChain, unit "composition.bundle";
            /// Bundle replay with an embedded joint predictive transport fit and decision.
            JointTransportPredictiveNodeDecisionBundleRecomputation => PropagationChain, unit "composition.bundle";
            /// Bundle replay with an embedded trial-IPW predictive fit and decision.
            TrialIpwPredictiveNodeDecisionBundleRecomputation => PropagationChain, unit "composition.bundle";
            /// Fit the cyclic posterior, standardize over the target frame and score the decision.
            ///
            /// # Errors
            /// A changed execution seed, a source or model refusal, or an invalid target frame.
            CyclicTargetFeedbackNodeDecisionBundleRecomputation => PropagationChain, unit "composition.bundle";
            /// Bundle replay with an embedded graph-imputation predictive fit and decision.
            GraphImputationPredictiveNodeDecisionBundleRecomputation => PropagationChain, unit "composition.bundle";
            /// Bundle replay with an embedded assessed-bridge posterior fit and decision.
            AssessedBridgePosteriorNodeDecisionBundleRecomputation => PropagationChain, unit "composition.bundle";
            /// Bundle replay with an embedded sequential policy reliability fit.
            SequentialPolicyReliabilityNodeBundleRecomputation => PropagationChain, unit "composition.bundle";
            /// Checked connected components and a joint posterior decision.
            JointComponentPosteriorDecision => PropagationChain, unit "cyclic.component_posterior_decisions";
            /// Checked connected components with a bounded-component posterior decision.
            BoundedComponentPosteriorDecision => PropagationChain, unit "cyclic.component_posterior_decisions";
            /// Bundle replay with a parameter-posterior decision terminal.
            PosteriorDecisionBundleRecomputation => PropagationChain, unit "composition.bundle";
            /// A graph-selection imputation composite.
            GraphImputation => GraphImputation, unit "graph_imputation.compiled_candidate";
            /// Graph-imputation effect posterior and downstream utility.
            GraphImputationEffectDecision => GraphImputation, unit "graph_imputation.compiled_decision";
            /// Graph-imputation Bernoulli predictive-world draws and a decision.
            GraphImputationBernoulliPredictiveWorldDecision => GraphImputation, unit "graph_imputation.compiled_decision";
            /// Graph-imputation Gaussian target predictive-world draws and a decision.
            GraphImputationGaussianTargetPredictiveWorldDecision => GraphImputation, unit "graph_imputation.compiled_decision";
            /// Graph-imputation Bernoulli predictive decision with a whole-cluster bootstrap.
            GraphImputationBernoulliPredictiveClusterBootstrapDecision => GraphImputation, unit "graph_imputation.compiled_decision";
            /// Graph-imputation Gaussian predictive decision with a whole-cluster bootstrap.
            GraphImputationGaussianPredictiveClusterBootstrapDecision => GraphImputation, unit "graph_imputation.compiled_decision";
            /// Graph-imputation target standardization and downstream utility.
            GraphImputationTransportDecision => GraphImputation, unit "graph_imputation.compiled_decision";
            /// Graph-imputation transport with a whole-cluster bootstrap.
            GraphImputationTransportClusterBootstrapDecision => GraphImputation, unit "graph_imputation.compiled_decision";
            /// Graph-imputation transport with a confounded target.
            GraphImputationConfoundedTransportDecision => GraphImputation, unit "graph_imputation.compiled_decision";
            /// Graph-imputation confounded transport with a whole-cluster bootstrap.
            GraphImputationConfoundedTransportClusterBootstrapDecision => GraphImputation, unit "graph_imputation.compiled_decision";
            /// Graph-imputation transport over mixed target covariates.
            GraphImputationKernelTransportDecision => GraphImputation, unit "graph_imputation.compiled_decision";
            /// Graph-imputation mixed-covariate kernel transport with a whole-cluster bootstrap.
            GraphImputationKernelTransportClusterBootstrapDecision => GraphImputation, unit "graph_imputation.compiled_decision";
            /// Graph-imputation confounded kernel transport with a whole-cluster bootstrap.
            GraphImputationConfoundedKernelTransportClusterBootstrapDecision => GraphImputation, unit "graph_imputation.compiled_decision";
            /// Graph-imputation transport under a missing-at-random target.
            GraphImputationMarTransportDecision => GraphImputation, unit "graph_imputation.compiled_decision";
            /// A scientifically closed categorical posterior candidate.
            CategoricalPosteriorCandidate => CategoricalPosteriorCandidate, unit "categorical_treatment.compiled_candidate";
            /// A bounded fixed-shift MNAR operation.
            MnarFixedShift => DesignBased, unit "mnar_fixed_shift.compiled_candidate";
            /// Latent additive soft-intervention discovery and prediction.
            LatentSoftIntervention => Structural, unit "estimate.latent_soft_intervention";
            /// Forecasts of timestamped events and their composed world responses.
            EventProcess => Structural, unit "forecast.event_process";
            /// A fitted joint multi-channel panel response over its declared program worlds.
            JointPanelResponse => JointPanel, unit "joint_response.interval_readings";
            /// A fitted joint panel response under a piecewise or smooth time-varying mechanism.
            JointPanelTimeVarying => JointPanel, unit "joint_response.interval_readings";
            /// A joint panel response pooled over multiply imputed completed panels.
            JointPanelImputation => JointPanel, unit "joint_response.interval_readings";
            /// A search for the best program over a fitted joint panel response.
            JointPanelProgramSearch => JointPanel, unit "joint_response.interval_readings";
            /// Held-out selection of a program from randomized or instrumented candidates.
            JointPanelProgramSelection => JointPanel, unit "joint_response.interval_readings";
            /// An independent experiment conditioning a joint panel or design law, and the
            /// decision and next-study value read from the conditioned law.
            JointPanelExperiment => JointPanel, unit "joint_response.interval_readings";
            /// A design-native joint functional with its retained design proof.
            JointDesign => JointPanel, unit "joint_response.interval_readings";
            /// A validation check of a compiled temporal structure search on the study's own data.
            DiscoveryValidation => Discovery, unit "validate.discovery_stability";
            /// Permutation falsification of a study's declared DAG against its retained table.
            GraphFalsification => Structural, unit "gcm.model.graph_falsification";
            /// The transported static path-specific counterfactual of a supplied affine
            /// additive-noise model.
            TransportedPathSpecific => Structural, unit "counterfactual.transported_additive";
            /// An edge-strength or do-contrast operation of a prepared study on its own fitted
            /// model.
            FittedModelAttribution => Structural, unit "gcm.attribute.mechanism_refit";
            /// A do-sampler, evaluation or nested counterfactual of a prepared study on its own
            /// fitted model.
            FittedModelSimulation => Structural, unit "gcm.simulate.mechanism_refit";
            /// A process-local feedback system of declared callback mechanisms, solved by
            /// Newton's method.
            LocalCyclicCallback => Cyclic, unit "cyclic.general_systems";
            /// A structural decision over supplied point, graph-dependent, weighted-atom or scenario claims.
            SuppliedClaimDecision => PropagationChain, unit "decision.structural_robustness";
            /// A decision over per-action identified sets of utility, supplied or enclosed from input sets.
            IdentifiedSetDecision => PropagationChain, unit "decision.structural_robustness";
            /// A decision contract evaluated on a declared aligned law or a grid of means.
            ContractLawDecision => PropagationChain, unit "decision.contract";
            /// A functional of one action's utility under a declared aligned law.
            DecisionFunctionalValue => PropagationChain, unit "decision.functionals";
            /// A decision or functional over several supported inputs, with support decided per action.
            SupportedInputDecision => PropagationChain, unit "composition.boundary";
            /// A decision over the surface of an assumption-sensitivity artifact.
            SensitivityDecision => PropagationChain, unit "decision.sensitivity_composition";
            /// An inverse decision query over forward evidence, or its finite-enumeration baseline.
            InverseQueryDecision => PropagationChain, unit "decision.functional_inverse";
            /// Ranking statements over aligned decision draws.
            DecisionDrawsRanking => PropagationChain, unit "decision_reliability.ranking_statement";
            /// Catalogued posterior sources adapted to a future candidate signal.
            PriorSignalAdaptation => Planning, unit "design.prior_signal_planning";
            /// The evidence obligations a failed identification owes.
            EvidenceObligations => Planning, unit "design.evidence_obligations";
            /// The Bayes action and expected regret of a finite decision problem under state draws.
            FiniteActionDecision => PropagationChain, unit "design.evsi";
            /// A bounded search over adaptive policy families against a response source.
            AdaptivePolicySearch => Planning, unit "design.adaptive_policy_search";
            /// A search for the declared studies that repair a failed identification contract.
            IdentificationRepair => Planning, unit "design.identification_repair";
            /// The cheapest declared studies that make a failed multi-source decision identify.
            StudyAdditionPlanning => Planning, unit "design.study_planning";
            /// A frozen z-transport failure and the declared evidence that would repair it.
            ZTransportEvidencePlanning => Planning, unit "design.identification_repair";
            /// A ranking of candidate designs under an objective over a graph posterior.
            DesignRanking => Planning, unit "design.ranking_artifact";
            /// Further choice studies ranked by expected value of sample information.
            ChoiceStudyRanking => Planning, unit "design.evsi";
            /// Identification flags of graphs and graph posteriors, with and without interventions.
            StructureIdentification => Planning, unit "design.evidence_plan";
            /// Classical sID, the evidence-catalog search and binding, and meta-transport identification.
            ClassicalTransportIdentification => DesignBased, unit "transport.classical_identification";
            /// Single-source z-transport, its catalog binding and the two-source combination.
            ZTransportIdentification => DesignBased, unit "transport.z_identification";
            /// Multi-source mz-transport identification and its catalog binding.
            MzTransportIdentification => DesignBased, unit "transport.multi_source_limited_experiment";
            /// Mixed-source proof search and its catalog binding.
            MixedSourceIdentification => DesignBased, unit "identify.mixed_source_proof_search";
            /// Conditional transport over a latent-confounded selection ADMG.
            AdmgConditionalTransportIdentification => DesignBased, unit "transport.admg_conditional";
            /// General finite counterfactual transport identification.
            CounterfactualTransportIdentification => DesignBased, unit "counterfactual.transport_general";
            /// Recovery of a law from one partially observed pattern law.
            ObservationRecoveryIdentification => DesignBased, unit "recovery.binary_observation";
            /// Recovery of an ordered response chain.
            ChainRecoveryIdentification => DesignBased, unit "recovery.chain_observation";
            /// The raw contrast of two arms, its comparison with an adjusted estimate, and a
            /// mean-outcome pair on declared reporting scales.
            DescriptiveComparison => DesignBased, unit "estimate.descriptive_comparison";
            /// The conditional odds ratio of matched case-control sets.
            MatchedCaseControl => DesignBased, unit "estimate.matched_case_control_odds_ratio";
            /// A conditional prediction from a lagged linear regression of one series.
            TemporalLinearPrediction => Associational, unit "estimate.temporal_linear_prediction";
            /// The measured sample-split effect replayed from its canonical source.
            SampleSplitEffect => Structural, unit "discovery.sample_split_effect";
            /// The measured sample-split parent-set effect replayed from its canonical source.
            SampleSplitParentSetEffect => Structural, unit "discovery.sample_split_parent_set_effect";
            /// A binary nested-Markov Fisher interval over declared regime counts.
            NestedMarkovFisherInterval => Structural, unit "identify.binary_nested_markov_fisher";
            /// A binary nested-Markov posterior over declared regime counts.
            NestedMarkovPosterior => Structural, unit "identify.binary_nested_markov";
            /// A learned polynomial-basis joint source-target transport posterior.
            LearnedJointTransport => Structural, unit "transport.learned_joint";
            /// A Gaussian joint source-target transport posterior and its measured target effect.
            JointBayesianTransport => Structural, unit "transport.joint_bayesian";
            /// The shared-row covariance of declared scenario estimates.
            SharedScenarioCovariance => Structural, unit "transport.shared_scenario_covariance";
            /// Equilibria of a feedback system with registered external mechanisms.
            ExternalFeedbackEquilibrium => Cyclic, unit "cyclic.external_mechanisms";
            /// A sampled counterfactual transport estimate from count tables.
            SampledCounterfactualTransport => Structural, unit "counterfactual.transport_sampled";
            /// The measured interval of a sealed sampled counterfactual transport estimate.
            SampledCounterfactualTransportInterval => Structural, unit "counterfactual.transport_sampled_interval";
            /// Sampled observation recovery with its whole-row bootstrap.
            SampledObservationRecovery => Structural, unit "recovery.sampled_observation";
            /// Measured replay of a sampled observation-recovery source.
            MeasuredSampledObservationRecovery => Structural, unit "recovery.sampled_observation";
            /// Ordered-response observation recovery on an exact observed law.
            ChainObservationRecovery => Structural, unit "recovery.chain_observation";
            /// Misclassification-corrected exposure association from case-control counts.
            ExposureMisclassificationCorrection => Associational, unit "measurement.exposure_misclassification";
            /// Linear outcome model on per-unit exposure probabilities.
            SoftExposureCorrection => Associational, unit "measurement.soft_exposure";
            /// Prevalence under a declared covariate-dependent detection function.
            DetectionDependentPrevalenceCorrection => Associational, unit "measurement.detection_dependent_prevalence";
            /// Latent-class fit of repeat codings of one latent facet.
            RepeatCodingsLatentClass => Associational, unit "measurement.repeat_codings_latent_class";
            /// Imputation of a measured exposure from its declaration.
            MeasuredExposureImputation => PropagationChain, unit "measurement.measured_exposure_effect";
            /// Rubin pooling of per-draw measured-exposure estimates.
            MeasuredExposurePooling => PropagationChain, unit "measurement.measured_exposure_effect";
            /// Power-prior discount of a Beta-Binomial source chosen from the target trials.
            PowerPriorBetaSelection => Associational, unit "association.conjugate_count_posterior";
            /// Power-prior discount of a Gamma-Poisson source chosen from the target counts.
            PowerPriorGammaSelection => Associational, unit "association.conjugate_count_posterior";
            /// Power-prior discount of external Gaussian sources chosen from a linear target.
            PowerPriorLinearSelection => Associational, unit "prior.composed_named_coordinates";
            /// Exact or empirical evaluation of a checked single-source z-transport formula.
            ZTransportEstimate => DesignBased, unit "transport.single_source_z_transport";
            /// Declared external prior sources composed with a baseline at their discounts.
            ExternalPriorComposition => Associational, unit "prior.external_composition";
            /// Declared external prior sources moved toward a target population.
            ExternalPriorTransport => Associational, unit "prior.external_composition";
            /// Augmented transported response grid on caller-supplied nuisances.
            TransportResponseGrid => Structural, unit "transport.augmented_response_grid";
            /// Finite graph/selection transport scenario set.
            TransportScenarioSet => Structural, unit "transport.finite_scenario_envelope";
            /// Measurement-error model carried to another population under a declared premise.
            MeasurementModelTransport => Structural, unit "transport.measurement_model";
            /// Differential measurement error pooled across validated populations.
            PooledDifferentialTransport => Structural, unit "measurement.model_contract";
            /// Retained fitted effect transported to a cohort of another population.
            TransportedCohortEffect => Structural, unit "transport.covariate_shift_cohort_effect";
            /// Bias-aware intervals and band of a randomized-dose response grid.
            DoseGridBiasAware => DesignBased, unit "estimate.dose_grid_bias_aware";
            /// Smoothed dose-response transport fitted with caller-supplied learners.
            SmoothedDoseLearnerTransport => DesignBased, unit "transport.smoothed_dose_caller_learners";
            /// Likelihood fit of a feedback system with registered external mechanisms.
            ExternalFeedbackFit => Cyclic, unit "cyclic.external_mechanisms";
            /// Measured interval of one named result of an adaptive experiment.
            AdaptiveResultInterval => DesignBased, unit "adaptive_inference.unsupported_guarantees";
            /// Measured credible interval of one named result of a choice study.
            ChoiceResultInterval => DesignBased, unit "choice_inference.interval_candidates";
            /// Measured interval of a declared cohort's mean conditional effect.
            CohortMeanInterval => Structural, unit "cohort_decision.interval";
            /// Measured posterior interval of a count-fitted observation probability.
            CountFittedInterval => Structural, unit "observation_query.calibrated_probability";
            /// Measured whole-unit interval of a finite two-step temporal response.
            TemporalDependentInterval => Temporal, unit "temporal.dependent_interval";
            /// Rosenbaum hidden-bias bounds of supplied matched pairs, pair vectors or matched
            /// sets.
            MatchedDesignSensitivity => DesignBased, unit "sensitivity.matched_hidden_bias";
            /// E-values of an effect estimate, a vector of estimates or posterior draws a caller
            /// supplies.
            SuppliedEstimateEValue => PropagationChain, unit "sensitivity.supplied_estimate_evalue";
            /// The outcome-kernel contamination range of a checked z-transport formula.
            ZTransportMechanismSensitivity => Structural, unit "sensitivity.joint_mechanism";
        }
    };
}

pub(crate) use program_families;

#[cfg(test)]
#[path = "program_families_tests.rs"]
mod tests;
