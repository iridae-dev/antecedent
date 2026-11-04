//! Joint experimental conditioning: several mapped functionals of one retained panel posterior,
//! updated together under their declared sampling covariance.
//!
//! Each component keeps its own supported [`BridgeContract`], response functional and paired
//! per-draw affine mapping, and passes the same validation as a single-component update. What
//! this adds is the likelihood: the experiment's estimators are not independent, so the update
//! conditions the shared draws on their joint Gaussian law instead of a product of marginals.
//! Nothing here supplies a missing covariance; evidence without one cannot construct an update.
// SPDX-License-Identifier: MIT OR Apache-2.0
use super::{
    Arc, DecisionProblem, DistributionArtifact, DistributionCalibration, DistributionTrust,
    EvidenceLedger, ExecutionContext, ExperimentRankingReceipt, JointPanelError,
    JointPanelResponseFunctional, JointPanelResult, PanelExperimentDecisionSource,
    PanelExperimentDecisionState, PanelExperimentEvidence, PanelExperimentMapping,
    PanelExperimentModularPolicy, PanelExperimentUpdate, PanelFutureSample, PanelFutureSampleValue,
    SemanticDigest, SourceId, admit, contrast_weights, law::MAX_LAW_EXPERIMENT_BYTES,
    rank_law_next_samples, source, validate_bridge_projection, validate_mapping_moments,
    validate_model_dependency, validate_model_origin, validate_update,
};
use antecedent_contract::bridge::BridgeContract;
use antecedent_design::decision_contract::UtilityExpr;

/// Largest number of components whose assumed-bias box is enumerated at its vertices.
const MAX_BIAS_COMPONENTS: usize = 12;
/// The `format` discriminator of a joint experiment artifact.
const JOINT_EXPERIMENT_FORMAT: &str = "joint_panel_joint_experiment_v1";

/// One experimental component: the supported translation of one measured quantity to one
/// functional of the retained response, with its paired per-draw affine realization.
#[derive(Clone, Debug)]
pub struct PanelJointExperimentComponent {
    /// Supported translation. Its source is this component's experimental quantity.
    pub bridge: BridgeContract,
    /// Time functional of the retained worlds that the bridge's target names exactly.
    pub functional: JointPanelResponseFunctional,
    /// One affine realization per retained draw slot, paired with the draws.
    pub mapping: PanelExperimentMapping,
}

/// A joint Gaussian sample-mean likelihood over two or more mapped components.
#[derive(Clone, Debug)]
pub struct PanelJointExperimentUpdate {
    /// Components in the order of `observed_means` and `sampling_covariance`.
    pub components: Vec<PanelJointExperimentComponent>,
    /// Observed experimental means, in each component's source units.
    pub observed_means: Vec<f64>,
    /// Known `K x K` row-major covariance of the mean estimators (not of one observation).
    pub sampling_covariance: Vec<f64>,
    /// Independent experimental units behind the estimates.
    pub sample_size: u64,
    /// Explicit modular policy, as for a single-component update.
    pub modular_policy: PanelExperimentModularPolicy,
    /// Declared origin and overlap authority shared by every component.
    pub evidence: PanelExperimentEvidence,
    /// Declared per-component absolute bias bounds on the measured contrasts, or `None`.
    ///
    /// When present the evidence is an assumption range, not a likelihood: it supports only
    /// `bias_sensitivity`, and no posterior, law source or ranking.
    pub contrast_bias_bounds: Option<Vec<f64>>,
}

fn experiment_id(c: &PanelJointExperimentComponent) -> Option<&String> {
    c.bridge.estimand_bridge.as_ref().map(|t| &t.experiment_id)
}

impl PanelJointExperimentUpdate {
    /// Each component as the single-component update that already owns its validation.
    ///
    /// The per-observation variance handed to that owner is only the finiteness-checked
    /// carrier of the component's marginal; the joint likelihood never uses it.
    fn scalar_updates(&self) -> Result<Vec<PanelExperimentUpdate>, JointPanelError> {
        let k = self.components.len();
        let bad_bounds = self.contrast_bias_bounds.as_ref().is_some_and(|bounds| {
            bounds.len() != k
                || k > MAX_BIAS_COMPONENTS
                || bounds.iter().any(|b| !b.is_finite() || *b < 0.0)
        });
        if bad_bounds {
            return Err(JointPanelError::compile(
                "assumed contrast bias bounds must be finite, nonnegative and one per component",
            ));
        }
        if k < 2 {
            return Err(JointPanelError::Unsupported(
                "a joint experimental update needs at least two components. Remedy: use `condition_experiment` for one measured quantity".into(),
            ));
        }
        if self.observed_means.len() != k
            || self.sampling_covariance.len() != k * k
            || self.sample_size == 0
        {
            return Err(JointPanelError::compile(
                "observed means, sampling covariance and unit count must align with the components",
            ));
        }
        for (i, a) in self.components.iter().enumerate() {
            for b in &self.components[..i] {
                if a.bridge.target.quantity == b.bridge.target.quantity
                    || a.bridge.source.quantity == b.bridge.source.quantity
                {
                    return Err(JointPanelError::Unsupported(
                        "two components name the same measured or target quantity; counting it twice would overstate the evidence. Remedy: declare each quantity once".into(),
                    ));
                }
            }
            if experiment_id(a).is_none() || experiment_id(a) != experiment_id(&self.components[0])
            {
                return Err(JointPanelError::Unsupported(
                    "components must be estimated by the one experiment whose covariance is declared. Remedy: bridge each component to that experiment, or update each independent experiment separately".into(),
                ));
            }
        }
        self.components
            .iter()
            .enumerate()
            .map(|(i, c)| {
                Ok(PanelExperimentUpdate {
                    bridge: c.bridge.clone(),
                    functional: c.functional.clone(),
                    observed_mean: self.observed_means[i],
                    observation_variance: self.sampling_covariance[i * k + i]
                        * self.sample_size as f64,
                    sample_size: self.sample_size,
                    modular_policy: self.modular_policy,
                    mapping: c.mapping.clone(),
                    evidence: self.evidence.clone(),
                })
            })
            .collect()
    }
}

/// Prior and posterior moments of one derived quantity of the updated components.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PanelJointCombination {
    /// Mean under the retained posterior.
    pub prior_mean: f64,
    /// Mean after the joint update.
    pub posterior_mean: f64,
    /// Variance under the retained posterior.
    pub prior_variance: f64,
    /// Variance after the joint update.
    pub posterior_variance: f64,
}

/// Posterior means of a derived quantity over assumed biases of the observed contrasts.
///
/// The range is over the enumerated nominal and vertex biases only. It is an assumption
/// sensitivity, not a statistical interval and not a guaranteed enclosure.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PanelJointSensitivity {
    /// Smallest posterior mean among the enumerated biases.
    pub enumerated_minimum: f64,
    /// Largest posterior mean among the enumerated biases.
    pub enumerated_maximum: f64,
    /// Posterior mean with no assumed bias.
    pub nominal: f64,
    /// Number of enumerated bias settings.
    pub settings: usize,
}

/// A joint conditional update retaining one aligned weight vector over the original draws.
#[derive(Clone, Debug)]
pub struct PanelJointExperimentResult {
    original: JointPanelResult,
    update: Arc<PanelJointExperimentUpdate>,
    scalars: Vec<Arc<PanelExperimentUpdate>>,
    laws: Vec<DistributionArtifact>,
    source_values: Vec<Vec<f64>>,
    prior_weights: Vec<f64>,
    weights: Vec<f64>,
    contract_id: String,
    dependency: SemanticDigest,
    retention: Arc<antecedent_core::execution::AdmissionLease>,
}

/// Conditional weights of `prior` under the joint likelihood of `observed`.
fn joint_weights(
    update: &PanelJointExperimentUpdate,
    observed: &[f64],
    source_values: &[Vec<f64>],
    prior: &[f64],
) -> Result<Vec<f64>, JointPanelError> {
    let intercepts: Vec<Vec<f64>> =
        update.components.iter().map(|c| c.mapping.intercepts.clone()).collect();
    let slopes: Vec<Vec<f64>> =
        update.components.iter().map(|c| c.mapping.slopes.clone()).collect();
    antecedent_prob::finite_belief::JointAffineGaussianExperiment {
        observed_means: observed,
        sampling_covariance: &update.sampling_covariance,
        target_values: source_values,
        intercepts: &intercepts,
        slopes: &slopes,
    }
    .condition(prior)
    .map_err(JointPanelError::compile)
}

fn moments(values: &[f64], weights: &[f64]) -> (f64, f64) {
    let mass = weights.iter().sum::<f64>();
    let mean = values.iter().zip(weights).map(|(v, w)| v * w).sum::<f64>() / mass;
    let variance =
        values.iter().zip(weights).map(|(v, w)| (v - mean).powi(2) * w).sum::<f64>() / mass;
    (mean, variance)
}

impl JointPanelResult {
    /// Condition this retained posterior jointly on several mapped experimental components.
    ///
    /// Every component passes the single-component bridge, origin, overlap, window and mapping
    /// checks. Only the quantities the bridges name are informed; no other functional of the
    /// response, program or population is promoted by the update.
    ///
    /// # Errors
    /// A sampling distribution, an unsupported or duplicated component, a covariance that is
    /// not symmetric positive definite, stale or overlapping evidence, or a bridge that does
    /// not name its exact response functional.
    pub fn condition_joint_experiment(
        &self,
        update: PanelJointExperimentUpdate,
        ledger: &EvidenceLedger,
        ctx: &ExecutionContext,
    ) -> Result<PanelJointExperimentResult, JointPanelError> {
        let scalars = update.scalar_updates()?;
        for scalar in &scalars {
            validate_update(scalar, ledger)?;
        }
        source::check_cancelled(ctx)?;
        let coefficients = contrast_weights(self.request())?;
        let mut priors = Vec::with_capacity(scalars.len());
        for scalar in &scalars {
            validate_model_origin(self, scalar, ledger)?;
            validate_bridge_projection(self, scalar)?;
            let law = self.numerical_project_law_source(&scalar.functional)?;
            if !matches!(
                law.semantic(),
                antecedent_io::quantity_wire::DistributionMeaningWire::CausalFunctionalPosterior
                    | antecedent_io::quantity_wire::DistributionMeaningWire::ParameterPosterior
            ) {
                return Err(JointPanelError::Unsupported(
                    "experimental likelihood updates require a posterior, never estimator sampling draws".into(),
                ));
            }
            priors.push(law);
        }
        let count = priors[0].n_draws();
        let prior_weights =
            priors[0].metadata().weights.clone().unwrap_or_else(|| vec![1.0 / count as f64; count]);
        if priors.iter().any(|law| {
            law.n_draws() != count
                || law.metadata().weights.as_ref().is_some_and(|w| *w != prior_weights)
        }) || update
            .components
            .iter()
            .any(|c| c.mapping.intercepts.len() != count || c.mapping.slopes.len() != count)
        {
            return Err(JointPanelError::compile(
                "components, mappings and retained draws must share the same aligned slots; implicit independent pairing is forbidden",
            ));
        }
        let width = coefficients.len();
        let bytes = count
            .checked_mul(scalars.len() * (width + 8) * 2 + 8)
            .and_then(|n| n.checked_mul(16))
            .and_then(|n| u64::try_from(n).ok())
            .ok_or_else(|| JointPanelError::compile("joint experiment quote overflow"))?;
        let retention = Arc::new(admit(ctx, bytes)?);
        let mut source_values = Vec::with_capacity(scalars.len());
        for (scalar, law) in scalars.iter().zip(&priors) {
            let values =
                antecedent_prob::finite_belief::linear_functional_rows(law.draws(), &coefficients)
                    .map_err(JointPanelError::compile)?;
            validate_mapping_moments(scalar, &values)?;
            source_values.push(values);
        }
        let weights =
            joint_weights(&update, &update.observed_means, &source_values, &prior_weights)?;
        let contract_id = priors[0].metadata().identity.causal_contract_id.clone();
        let dependency =
            joint_dependency(self.dependency_identity(), &contract_id, &update, &weights);
        let laws = priors
            .iter()
            .map(|law| {
                let mut metadata = law.metadata().clone();
                metadata.weights = Some(weights.clone());
                metadata.identity.causal_contract_id = dependency.to_hex();
                metadata.identity.semantic =
                    antecedent_io::quantity_wire::DistributionMeaningWire::CausalFunctionalPosterior;
                metadata.identity.provider_id =
                    "antecedent.compiled_joint_panel.joint_conditional_experiment".into();
                metadata.calibration = DistributionCalibration::NotMeasured;
                metadata.trust = DistributionTrust::Unverified;
                DistributionArtifact::new(metadata, law.draws().to_vec())
                    .map_err(JointPanelError::compile)
            })
            .collect::<Result<Vec<_>, _>>()?;
        source::check_cancelled(ctx)?;
        Ok(PanelJointExperimentResult {
            original: self.clone(),
            scalars: scalars.into_iter().map(Arc::new).collect(),
            update: Arc::new(update),
            laws,
            source_values,
            prior_weights,
            weights,
            contract_id,
            dependency,
            retention,
        })
    }
}

/// Everything the joint law depends on, but not the decision utility or candidate programs.
fn joint_dependency(
    source_dependency: SemanticDigest,
    contract_id: &str,
    update: &PanelJointExperimentUpdate,
    weights: &[f64],
) -> SemanticDigest {
    let mut joint = antecedent_core::CanonicalWriter::new("antecedent.joint_panel.joint_update.v1");
    joint.word(update.components.len() as u64).word(update.sample_size);
    for (i, component) in update.components.iter().enumerate() {
        joint
            .text(&component.bridge.identity().to_hex())
            .text(&component.bridge.target.quantity.functional_id)
            .real(update.observed_means[i]);
        for (a, b) in component.mapping.intercepts.iter().zip(&component.mapping.slopes) {
            joint.real(*a).real(*b);
        }
    }
    for value in &update.sampling_covariance {
        joint.real(*value);
    }
    if let Some(bounds) = &update.contrast_bias_bounds {
        joint.word(1);
        for bound in bounds {
            joint.real(*bound);
        }
    } else {
        joint.word(0);
    }
    let evidence = &update.evidence;
    let mut origins =
        vec![evidence.model.to_hex(), evidence.experiment.to_hex(), evidence.mapping.to_hex()];
    origins.extend(evidence.prior_origins.iter().map(SourceId::to_hex));
    SemanticDigest::from_bytes(
        antecedent_io::joint_panel_experiment_artifact::experiment_update_identity(
            contract_id,
            source_dependency,
            SemanticDigest::from_bytes(joint.finish()),
            update.observed_means[0],
            update.sampling_covariance[0],
            update.sample_size,
            &update.components[0].mapping.intercepts,
            &update.components[0].mapping.slopes,
            weights,
            &origins,
        ),
    )
}

impl PanelJointExperimentResult {
    /// The original compiled result and its retained identification obligations.
    #[must_use]
    pub fn original(&self) -> &JointPanelResult {
        &self.original
    }
    /// This update is conditional on the declared joint likelihood and paired mappings.
    #[must_use]
    pub fn scope(&self) -> &'static str {
        if self.update.contrast_bias_bounds.is_some() {
            "assumption_bounded_joint_experimental_update;no_likelihood_claim;calibration_not_measured"
        } else {
            "model_conditional_joint_experimental_update;calibration_not_measured"
        }
    }
    /// Exact joint update dependency, independent of any decision utility.
    #[must_use]
    pub fn dependency_identity(&self) -> SemanticDigest {
        self.dependency
    }
    /// The quantities the experiment informs; every other prediction remains model-conditional.
    #[must_use]
    pub fn informed_quantities(&self) -> Vec<antecedent_contract::ScientificQuantity> {
        self.update.components.iter().map(|c| c.bridge.target.quantity.clone()).collect()
    }
    /// Posterior weights over the retained draws, aligned across every component.
    ///
    /// # Errors
    /// Withdrawal, supersession, changed origins, invalid bridge authority, or assumption-bounded
    /// evidence, which has no posterior.
    pub fn posterior_weights(&self, ledger: &EvidenceLedger) -> Result<&[f64], JointPanelError> {
        self.validate_decision_use(ledger)?;
        Ok(&self.weights)
    }
    /// Moments of an expression over the updated components under the aligned joint weights.
    ///
    /// `Input(k)` is component `k` (the bridged response contrast). The derived quantity's
    /// uncertainty keeps the components' dependence; it is not a combination of marginals.
    /// A nonlinear expression is evaluated per retained draw, never on the means.
    ///
    /// # Errors
    /// Current authority, an input outside the components or a nonfinite value.
    pub fn derived_quantity(
        &self,
        expression: &UtilityExpr,
        ledger: &EvidenceLedger,
    ) -> Result<PanelJointCombination, JointPanelError> {
        self.validate_decision_use(ledger)?;
        let values = self.derived_values(expression)?;
        let (prior_mean, prior_variance) = moments(&values, &self.prior_weights);
        let (posterior_mean, posterior_variance) = moments(&values, &self.weights);
        Ok(PanelJointCombination { prior_mean, posterior_mean, prior_variance, posterior_variance })
    }
    /// Moments of `sum_k coefficients[k] * component_k`; see [`Self::derived_quantity`].
    ///
    /// # Errors
    /// As [`Self::derived_quantity`], or a coefficient count other than the component count.
    pub fn combination(
        &self,
        coefficients: &[f64],
        ledger: &EvidenceLedger,
    ) -> Result<PanelJointCombination, JointPanelError> {
        if coefficients.len() != self.source_values.len()
            || coefficients.iter().any(|c| !c.is_finite())
        {
            return Err(JointPanelError::compile(
                "one finite coefficient per updated component is required",
            ));
        }
        let expression = coefficients.iter().enumerate().fold(
            UtilityExpr::Const(0.0),
            |sum, (index, coefficient)| {
                UtilityExpr::sum(
                    sum,
                    UtilityExpr::product(
                        UtilityExpr::Const(*coefficient),
                        UtilityExpr::Input(index),
                    ),
                )
            },
        );
        self.derived_quantity(&expression, ledger)
    }
    fn derived_values(&self, expression: &UtilityExpr) -> Result<Vec<f64>, JointPanelError> {
        let mut inputs = vec![0.0; self.source_values.len()];
        (0..self.weights.len())
            .map(|slot| {
                for (input, values) in inputs.iter_mut().zip(&self.source_values) {
                    *input = values[slot];
                }
                expression.evaluate(&inputs).map_err(JointPanelError::compile).and_then(|v| {
                    v.is_finite().then_some(v).ok_or_else(|| {
                        JointPanelError::compile("derived quantity is not finite in a draw")
                    })
                })
            })
            .collect()
    }
    /// Posterior mean of a derived quantity over the assumed biases of assumption-bounded
    /// evidence, enumerated at the nominal setting and every vertex of the bias box.
    ///
    /// The range is an assumption sensitivity kept apart from any statistical interval.
    ///
    /// # Errors
    /// Current authority, or evidence that declared no assumed bias bounds.
    pub fn bias_sensitivity(
        &self,
        expression: &UtilityExpr,
        ledger: &EvidenceLedger,
    ) -> Result<PanelJointSensitivity, JointPanelError> {
        self.validate_current(ledger)?;
        let Some(bounds) = &self.update.contrast_bias_bounds else {
            return Err(JointPanelError::Unsupported(
                "no assumed bias bounds were declared. Remedy: use `derived_quantity` for evidence with a likelihood".into(),
            ));
        };
        let values = self.derived_values(expression)?;
        let k = bounds.len();
        let mean_at = |shift: &dyn Fn(usize) -> f64| -> Result<f64, JointPanelError> {
            let observed: Vec<f64> =
                self.update.observed_means.iter().enumerate().map(|(i, o)| o + shift(i)).collect();
            let weights =
                joint_weights(&self.update, &observed, &self.source_values, &self.prior_weights)?;
            Ok(moments(&values, &weights).0)
        };
        let nominal = mean_at(&|_| 0.0)?;
        let (mut low, mut high) = (nominal, nominal);
        for mask in 0..(1_usize << k) {
            let mean = mean_at(&|i| if (mask >> i) & 1 == 1 { bounds[i] } else { -bounds[i] })?;
            low = low.min(mean);
            high = high.max(mean);
        }
        Ok(PanelJointSensitivity {
            enumerated_minimum: low,
            enumerated_maximum: high,
            nominal,
            settings: (1_usize << k) + 1,
        })
    }
    /// The updated law of one component's response functional, with its conditional standing.
    ///
    /// Authorization re-checks every component's evidence, not only this one's, because the
    /// shared weights depend on all of them.
    ///
    /// # Errors
    /// Current evidence or bridge authority, assumption-bounded evidence (no likelihood
    /// exists), an out-of-range component, or projection binding.
    pub fn law_source(
        &self,
        component: usize,
        ledger: &EvidenceLedger,
    ) -> Result<PanelExperimentDecisionSource, JointPanelError> {
        self.validate_decision_use(ledger)?;
        let (Some(law), Some(scalar)) = (self.laws.get(component), self.scalars.get(component))
        else {
            return Err(JointPanelError::compile(
                "component is not one this experiment updated; its prediction stays model-conditional",
            ));
        };
        let weights = self.original.projection(&scalar.functional)?.1;
        let source = self.original.projected_law_source_with_dependency(
            law,
            Some(weights),
            self.dependency,
        )?;
        Ok(PanelExperimentDecisionSource {
            source,
            update: Arc::clone(scalar),
            original_dependency: self.original.dependency_identity(),
            retention: Arc::clone(&self.retention),
            companions: self
                .scalars
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != component)
                .map(|(_, s)| Arc::clone(s))
                .collect(),
        })
    }
    /// Evaluate an existing decision contract on one component's updated law.
    ///
    /// # Errors
    /// Existing decision semantics and current authority are enforced without relaxation.
    pub fn decide(
        &self,
        component: usize,
        contract: &antecedent_design::decision_contract::DecisionContract,
        ledger: &EvidenceLedger,
    ) -> Result<antecedent_design::scientific_source::ScopedDecisionResult, JointPanelError> {
        self.law_source(component, ledger)?.evaluate(
            contract,
            antecedent_design::scientific_source::ScientificConsumptionScope::ModelConditional,
            ledger,
        )
    }
    /// Rank independent future samples of one component's mapped estimand.
    ///
    /// The future sample is a new, independent study valued against the jointly updated
    /// weights and the caller's decision; it does not rank other estimands.
    ///
    /// # Errors
    /// As [`PanelExperimentResult::rank_next_samples`](super::PanelExperimentResult::rank_next_samples),
    /// and an out-of-range component.
    pub fn rank_next_samples<A: serde::Serialize>(
        &self,
        component: usize,
        problem: &DecisionProblem<A, PanelExperimentDecisionState>,
        candidates: &[PanelFutureSample],
        replicates: usize,
        ledger: &EvidenceLedger,
        ctx: &ExecutionContext,
    ) -> Result<Vec<PanelFutureSampleValue>, JointPanelError> {
        let scientific = self.law_source(component, ledger)?;
        let coefficients = contrast_weights(self.original.request())?;
        rank_law_next_samples(
            &ExperimentRankingReceipt::from(&scientific),
            &self.laws[component],
            &self.scalars[component],
            &coefficients,
            problem,
            candidates,
            replicates,
            ctx,
        )
    }
    fn validate_current(&self, ledger: &EvidenceLedger) -> Result<(), JointPanelError> {
        self.retention.observe_current();
        for scalar in &self.scalars {
            validate_update(scalar, ledger)?;
            validate_model_origin(&self.original, scalar, ledger)?;
        }
        Ok(())
    }
    fn validate_decision_use(&self, ledger: &EvidenceLedger) -> Result<(), JointPanelError> {
        self.validate_current(ledger)?;
        if self.update.contrast_bias_bounds.is_some() {
            return Err(JointPanelError::Unsupported(
                "the evidence is assumption-bounded, so it supports no likelihood, decision or ranking. Remedy: use `bias_sensitivity`, or supply evidence without an interference bound".into(),
            ));
        }
        Ok(())
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct JointExperimentWire {
    format: String,
    version: u16,
    source_dependency: [u8; 32],
    contract_id: String,
    bridges: Vec<antecedent_io::bridge_artifact::FullBridgeContractWire>,
    ledger: antecedent_io::evidence_ledger_artifact::EvidenceLedgerMeta,
    origins: antecedent_io::joint_panel_experiment_artifact::PanelExperimentOrigins,
    observed: Vec<u64>,
    covariance: Vec<u64>,
    sample_size: u64,
    bias_bounds: Option<Vec<u64>>,
    intercepts: Vec<Vec<u64>>,
    slopes: Vec<Vec<u64>>,
    source_values: Vec<Vec<u64>>,
    prior_weights: Vec<u64>,
    weights: Vec<u64>,
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}
fn floats(values: &[u64]) -> Vec<f64> {
    values.iter().map(|v| f64::from_bits(*v)).collect()
}

/// Independently replayed numerical component, without current authority.
#[derive(Clone, Debug)]
pub struct HistoricalJointExperiment {
    weights: Vec<f64>,
    dependency: SemanticDigest,
    retention: Arc<antecedent_core::execution::AdmissionLease>,
}
impl HistoricalJointExperiment {
    /// Replayed posterior weights; historical data, not permission to consume withdrawn evidence.
    #[must_use]
    pub fn weights(&self) -> &[f64] {
        self.retention.observe_current();
        &self.weights
    }
    /// Replayed update dependency.
    #[must_use]
    pub fn dependency_identity(&self) -> SemanticDigest {
        self.dependency
    }
    /// A historical replay never grants caller-current scientific authority.
    #[must_use]
    pub const fn grants_current_authority(&self) -> bool {
        false
    }
}

impl PanelJointExperimentResult {
    /// Retain the update's declared inputs and the weights it claims, with the ledger history.
    ///
    /// # Errors
    /// Current authority, serialization, cancellation or resource admission fails.
    pub fn export(
        &self,
        ledger: &EvidenceLedger,
        ctx: &ExecutionContext,
    ) -> Result<Vec<u8>, JointPanelError> {
        self.validate_current(ledger)?;
        source::check_cancelled(ctx)?;
        let cells = self
            .weights
            .len()
            .checked_mul(self.source_values.len() * 4 + 2)
            .and_then(|n| n.checked_mul(32))
            .and_then(|n| u64::try_from(n).ok())
            .ok_or_else(|| JointPanelError::compile("joint experiment export quote overflow"))?;
        let _lease = admit(ctx, cells)?;
        let (_, history) =
            antecedent_io::evidence_ledger_artifact::EvidenceLedgerArtifact::seal(ledger, &[])
                .map_err(JointPanelError::compile)?;
        let evidence = &self.update.evidence;
        let wire = JointExperimentWire {
            format: JOINT_EXPERIMENT_FORMAT.into(),
            version: 1,
            source_dependency: *self.original.dependency_identity().as_bytes(),
            contract_id: self.contract_id.clone(),
            bridges: self
                .update
                .components
                .iter()
                .map(|c| {
                    antecedent_io::bridge_artifact::FullBridgeContractWire::from_domain(&c.bridge)
                        .map_err(JointPanelError::compile)
                })
                .collect::<Result<_, _>>()?,
            ledger: history,
            origins: antecedent_io::joint_panel_experiment_artifact::PanelExperimentOrigins {
                model: evidence.model.to_hex(),
                experiment: evidence.experiment.to_hex(),
                mapping: evidence.mapping.to_hex(),
                priors: evidence.prior_origins.iter().map(SourceId::to_hex).collect(),
            },
            observed: bits(&self.update.observed_means),
            covariance: bits(&self.update.sampling_covariance),
            sample_size: self.update.sample_size,
            bias_bounds: self.update.contrast_bias_bounds.as_deref().map(bits),
            intercepts: self
                .update
                .components
                .iter()
                .map(|c| bits(&c.mapping.intercepts))
                .collect(),
            slopes: self.update.components.iter().map(|c| bits(&c.mapping.slopes)).collect(),
            source_values: self.source_values.iter().map(|v| bits(v)).collect(),
            prior_weights: bits(&self.prior_weights),
            weights: bits(&self.weights),
        };
        let bytes = serde_json::to_vec(&wire).map_err(JointPanelError::compile)?;
        if bytes.len() > MAX_LAW_EXPERIMENT_BYTES {
            return Err(JointPanelError::compile(
                "joint experiment artifact exceeds the byte limit",
            ));
        }
        source::check_cancelled(ctx)?;
        Ok(bytes)
    }

    /// Replay an exported update from its retained inputs alone: re-validate its bridges,
    /// ledger history and mappings, recompute the weights and check them against the claim.
    ///
    /// # Errors
    /// Unknown format, altered inputs, invalid historical evidence or weights that do not
    /// follow from the retained likelihood.
    pub fn replay(
        bytes: &[u8],
        ctx: &ExecutionContext,
    ) -> Result<HistoricalJointExperiment, JointPanelError> {
        source::check_cancelled(ctx)?;
        if bytes.is_empty() || bytes.len() > MAX_LAW_EXPERIMENT_BYTES {
            return Err(JointPanelError::compile(
                "joint experiment artifact is empty or exceeds the byte limit",
            ));
        }
        let quote =
            bytes.len().checked_mul(32).and_then(|n| u64::try_from(n).ok()).ok_or_else(|| {
                JointPanelError::compile("joint experiment artifact quote overflow")
            })?;
        let retention = Arc::new(admit(ctx, quote)?);
        let wire: JointExperimentWire =
            serde_json::from_slice(bytes).map_err(JointPanelError::compile)?;
        if wire.format != JOINT_EXPERIMENT_FORMAT || wire.version != 1 {
            return Err(JointPanelError::compile("unknown joint experiment artifact format"));
        }
        let history = antecedent_io::evidence_ledger_artifact::EvidenceLedgerArtifact::from_meta(
            &wire.ledger,
            None,
        )
        .map_err(JointPanelError::compile)?;
        let source_dependency = SemanticDigest::from_bytes(wire.source_dependency);
        let id = |text: &str| {
            SourceId::from_hex(text)
                .ok_or_else(|| JointPanelError::compile("invalid evidence origin digest"))
        };
        let update = PanelJointExperimentUpdate {
            components: wire
                .bridges
                .iter()
                .zip(wire.intercepts.iter().zip(&wire.slopes))
                .map(|(bridge, (intercepts, slopes))| {
                    Ok(PanelJointExperimentComponent {
                        bridge: bridge.to_domain().map_err(JointPanelError::compile)?,
                        // Window binding needs the original panel; consuming it re-checks that.
                        functional: JointPanelResponseFunctional::TerminalMean,
                        mapping: PanelExperimentMapping {
                            dependency: source_dependency,
                            intercepts: floats(intercepts),
                            slopes: floats(slopes),
                        },
                    })
                })
                .collect::<Result<_, JointPanelError>>()?,
            observed_means: floats(&wire.observed),
            sampling_covariance: floats(&wire.covariance),
            sample_size: wire.sample_size,
            modular_policy: PanelExperimentModularPolicy::FrozenJointImportance,
            evidence: PanelExperimentEvidence {
                model: id(&wire.origins.model)?,
                prior_origins: wire
                    .origins
                    .priors
                    .iter()
                    .map(|p| id(p))
                    .collect::<Result<_, _>>()?,
                experiment: id(&wire.origins.experiment)?,
                mapping: id(&wire.origins.mapping)?,
            },
            contrast_bias_bounds: wire.bias_bounds.as_deref().map(floats),
        };
        let scalars = update.scalar_updates()?;
        let source_values: Vec<Vec<f64>> = wire.source_values.iter().map(|v| floats(v)).collect();
        if source_values.len() != scalars.len() {
            return Err(JointPanelError::compile("retained source values do not match components"));
        }
        for (scalar, values) in scalars.iter().zip(&source_values) {
            validate_update(scalar, history.ledger())?;
            validate_model_dependency(source_dependency, scalar, history.ledger())?;
            validate_mapping_moments(scalar, values)?;
        }
        let weights = joint_weights(
            &update,
            &update.observed_means,
            &source_values,
            &floats(&wire.prior_weights),
        )?;
        if bits(&weights) != wire.weights {
            return Err(JointPanelError::compile(
                "retained weights do not follow from the retained joint likelihood",
            ));
        }
        source::check_cancelled(ctx)?;
        let dependency = joint_dependency(source_dependency, &wire.contract_id, &update, &weights);
        Ok(HistoricalJointExperiment { weights, dependency, retention })
    }

    /// Bind a replayed artifact to this actual result and caller-current evidence.
    ///
    /// # Errors
    /// Withdrawn or changed evidence, or an artifact that is not this update.
    pub fn consume_current(
        &self,
        bytes: &[u8],
        component: usize,
        ledger: &EvidenceLedger,
        ctx: &ExecutionContext,
    ) -> Result<PanelExperimentDecisionSource, JointPanelError> {
        let current = self.law_source(component, ledger)?;
        let replay = Self::replay(bytes, ctx)?;
        if replay.dependency != self.dependency || bits(&replay.weights) != bits(&self.weights) {
            return Err(JointPanelError::compile(
                "artifact differs from this current joint experiment update",
            ));
        }
        Ok(current)
    }
}
