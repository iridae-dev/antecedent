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
            FittedDependentMeasuredObservation => Structural, unit "";
            /// Model-derived paired intervention worlds and a decision.
            CyclicActionWorldDecision => Cyclic, unit "";
            /// Joint fitted-mechanism posterior predictive action worlds.
            CyclicPosteriorActionDecision => Cyclic, unit "";
            /// Target-population feedback standardization over a cyclic fit and a decision.
            CyclicTargetFeedbackDecision => Cyclic, unit "";
            /// Checked bridge transport and predictive utility.
            BridgeTransportDecision => Structural, unit "";
            /// Premise assessment of a checked bridge transport.
            BridgeTransportPremiseAssessment => Structural, unit "";
            /// Matched bridge validation and conditional predictive utility.
            AssessedBridgeTransportDecision => Structural, unit "";
            /// Fitted source-target effect posterior and utility.
            JointTransportEffectDecision => Structural, unit "";
            /// Trial inverse-probability-weighted transport and a decision.
            TrialIpwTransportDecision => Structural, unit "";
            /// Trial inverse-probability-weighted predictive transport and a decision.
            TrialIpwPredictiveDecision => Structural, unit "";
            /// Joint source-target predictive transport and a decision.
            JointTransportPredictiveDecision => Structural, unit "";
            /// Target-population time-ordered path-specific transport.
            TemporalPathTransport => Temporal, unit "";
            /// Whole-chain resampling union around an assumption envelope.
            SamplingAssumptionUnion => PropagationChain, unit "";
            /// Whole-chain resampling union with independent and identically distributed units.
            IidSamplingAssumptionUnion => PropagationChain, unit "";
            /// Compiler-generated whole-cluster bootstrap assumption union.
            ClusterSamplingAssumptionUnion => PropagationChain, unit "";
            /// Whole-chain resampling union with moving-block resampling.
            MovingBlockSamplingAssumptionUnion => PropagationChain, unit "";
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
            JointComponentPosteriorDecision => PropagationChain, unit "";
            /// Checked connected components with a bounded-component posterior decision.
            BoundedComponentPosteriorDecision => PropagationChain, unit "";
            /// Bundle replay with a parameter-posterior decision terminal.
            PosteriorDecisionBundleRecomputation => PropagationChain, unit "composition.bundle";
            /// A graph-selection imputation composite.
            GraphImputation => GraphImputation, unit "graph_imputation.compiled_candidate";
            /// Graph-imputation effect posterior and downstream utility.
            GraphImputationEffectDecision => GraphImputation, unit "";
            /// Graph-imputation Bernoulli predictive-world draws and a decision.
            GraphImputationBernoulliPredictiveWorldDecision => GraphImputation, unit "";
            /// Graph-imputation Gaussian target predictive-world draws and a decision.
            GraphImputationGaussianTargetPredictiveWorldDecision => GraphImputation, unit "";
            /// Graph-imputation Bernoulli predictive decision with a whole-cluster bootstrap.
            GraphImputationBernoulliPredictiveClusterBootstrapDecision => GraphImputation, unit "";
            /// Graph-imputation Gaussian predictive decision with a whole-cluster bootstrap.
            GraphImputationGaussianPredictiveClusterBootstrapDecision => GraphImputation, unit "";
            /// Graph-imputation target standardization and downstream utility.
            GraphImputationTransportDecision => GraphImputation, unit "";
            /// Graph-imputation transport with a whole-cluster bootstrap.
            GraphImputationTransportClusterBootstrapDecision => GraphImputation, unit "";
            /// Graph-imputation transport with a confounded target.
            GraphImputationConfoundedTransportDecision => GraphImputation, unit "";
            /// Graph-imputation confounded transport with a whole-cluster bootstrap.
            GraphImputationConfoundedTransportClusterBootstrapDecision => GraphImputation, unit "";
            /// Graph-imputation transport over mixed target covariates.
            GraphImputationKernelTransportDecision => GraphImputation, unit "";
            /// Graph-imputation mixed-covariate kernel transport with a whole-cluster bootstrap.
            GraphImputationKernelTransportClusterBootstrapDecision => GraphImputation, unit "";
            /// Graph-imputation confounded kernel transport with a whole-cluster bootstrap.
            GraphImputationConfoundedKernelTransportClusterBootstrapDecision => GraphImputation, unit "";
            /// Graph-imputation transport under a missing-at-random target.
            GraphImputationMarTransportDecision => GraphImputation, unit "";
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
        }
    };
}

pub(crate) use program_families;

#[cfg(test)]
#[path = "program_families_tests.rs"]
mod tests;
