//! Future studies in regimes other than the one already observed.
//!
//! A candidate study may run in another population, window, dose, duration, exposure
//! regime, recency or spend than the experiment that produced the current update. It then
//! measures a different functional of the same retained joint response, through its own
//! licensed bridge (with the transport premise that regime needs) and its own paired
//! per-draw affine mapping. The retained draws, their current posterior weights and the
//! terminal responses stay exactly those of the update; only what each draw predicts the
//! study will observe changes. The ranking therefore values the study through the actual
//! update it would cause, the response it would move and the decision that response drives.
// SPDX-License-Identifier: MIT OR Apache-2.0
use super::{
    ExecutionContext, ExperimentRankingReceipt, JointPanelError, JointPanelResponseFunctional,
    PanelExperimentDecisionState, PanelExperimentMapping, PanelExperimentResult, PanelFutureSample,
    PanelFutureSampleValue, SemanticDigest, contrast_weights, rank_law_next_studies,
    validate_bridge_window,
};
use antecedent_contract::bridge::{BridgeContract, SourceMappingDependence};
use antecedent_contract::evidence_ledger::{EvidenceLedger, Overlap, SourceId, SourceStatus};
use antecedent_design::decision::DecisionProblem;

/// The setting a future study would run in and how it connects to the retained response.
#[derive(Clone, Debug)]
pub struct PanelFutureRegime {
    /// The time functional of the retained worlds that the study's estimand corresponds to.
    pub functional: JointPanelResponseFunctional,
    /// Licensed translation from that study's population, window, dose, duration, exposure,
    /// recency and spend to the retained contrast, with its transport premise and the
    /// mapping uncertainty and source dependence it carries.
    pub bridge: BridgeContract,
    /// One affine realization per retained draw slot, paired with the draws.
    pub mapping: PanelExperimentMapping,
    /// Current ledger origin of this regime's mapping evidence.
    pub mapping_evidence: SourceId,
}

/// A candidate study: sampling model and cost, and optionally its own regime.
#[derive(Clone, Debug)]
pub struct PanelFutureStudy {
    /// Sample size, observation variance and cost in the decision's utility unit.
    pub sample: PanelFutureSample,
    /// `None` is another sample of the already observed estimand.
    pub regime: Option<PanelFutureRegime>,
}

/// Value of one candidate study together with the regime that produced it.
#[derive(Clone, Debug)]
pub struct PanelFutureStudyValue {
    /// Gross and net value, standard error and the question and action bindings.
    pub value: PanelFutureSampleValue,
    /// Identity of the bridge, functional and paired mapping used; `None` for the observed estimand.
    pub regime_identity: Option<SemanticDigest>,
}

/// The actual conditional update a candidate study would cause for one hypothetical outcome.
#[derive(Clone, Debug)]
pub struct PanelFutureStudyUpdate {
    /// Candidate identity.
    pub study_id: String,
    /// The hypothetical sample mean conditioned on.
    pub observed_mean: f64,
    /// Posterior mass per retained draw slot after this outcome.
    pub posterior_weights: Vec<f64>,
    /// Response means before this outcome, in projected world order.
    pub prior_response_means: Vec<f64>,
    /// Response means after this outcome, in the same order.
    pub response_means: Vec<f64>,
    /// Regime identity, as in the ranking.
    pub regime_identity: Option<SemanticDigest>,
}

struct PreparedRegime {
    source_values: Vec<f64>,
    intercepts: Vec<f64>,
    slopes: Vec<f64>,
    means: Vec<f64>,
    identity: Option<SemanticDigest>,
}

impl PanelExperimentResult {
    fn prepare_regime(
        &self,
        regime: Option<&PanelFutureRegime>,
        ledger: &EvidenceLedger,
    ) -> Result<PreparedRegime, JointPanelError> {
        let coefficients = contrast_weights(self.original.request())?;
        let width = coefficients.len();
        let Some(regime) = regime else {
            let source_values = antecedent_prob::finite_belief::linear_functional_rows(
                self.law.draws(),
                &coefficients,
            )
            .map_err(JointPanelError::compile)?;
            let (intercepts, slopes) =
                (self.update.mapping.intercepts.clone(), self.update.mapping.slopes.clone());
            let means = mapped_means(&source_values, &intercepts, &slopes)?;
            return Ok(PreparedRegime { source_values, intercepts, slopes, means, identity: None });
        };
        regime
            .bridge
            .check_licensing()
            .map_err(|error| JointPanelError::compile(format!("{error:?}")))?;
        if regime.bridge.estimand_bridge.is_none() || regime.bridge.source_dependence.is_none() {
            return Err(JointPanelError::compile(
                "a future study in another regime needs the full population/window/dose/duration/exposure/recency/spend translation and its source dependence",
            ));
        }
        validate_bridge_window(&self.original, &regime.functional, &regime.bridge)?;
        if regime.mapping.dependency != self.original.dependency_identity() {
            return Err(JointPanelError::Unsupported(
                "future-study mapping refers to another fitted source or numerical dependency"
                    .into(),
            ));
        }
        if !ledger
            .source(regime.mapping_evidence)
            .is_some_and(|source| source.status == SourceStatus::Active)
        {
            return Err(JointPanelError::Unsupported(
                "future-study mapping evidence is missing, withdrawn or superseded".into(),
            ));
        }
        if matches!(
            regime.bridge.source_dependence,
            Some(SourceMappingDependence::Independent { .. })
        ) && !matches!(
            ledger
                .overlap(self.update.evidence.model, regime.mapping_evidence)
                .map_err(JointPanelError::compile)?,
            Overlap::Independent
        ) {
            return Err(JointPanelError::Unsupported(
                "future-study mapping independence contradicts current ledger origins".into(),
            ));
        }
        let count = self.law.n_draws();
        if regime.mapping.intercepts.len() != count
            || regime.mapping.slopes.len() != count
            || regime.mapping.intercepts.iter().any(|value| !value.is_finite())
            || regime.mapping.slopes.iter().any(|value| !value.is_finite() || *value == 0.0)
        {
            return Err(JointPanelError::compile(
                "future-study mapping needs one finite nonzero-slope realization per retained draw slot; implicit pairing is forbidden",
            ));
        }
        // The regime's functional is read from the same retained draws, so slot i is the
        // same joint realization the update weighted.
        let projected = self.original.numerical_project_law_source(&regime.functional)?;
        if projected.n_draws() != count || projected.shape()[1] != width {
            return Err(JointPanelError::compile(
                "future-study functional does not align with the updated draw slots",
            ));
        }
        let source_values = antecedent_prob::finite_belief::linear_functional_rows(
            projected.draws(),
            &coefficients,
        )
        .map_err(JointPanelError::compile)?;
        regime
            .bridge
            .validate_finite_affine_mapping(
                &regime.mapping.intercepts,
                &regime.mapping.slopes,
                &source_values,
            )
            .map_err(|error| JointPanelError::compile(format!("{error:?}")))?;
        let means =
            mapped_means(&source_values, &regime.mapping.intercepts, &regime.mapping.slopes)?;
        let weights = self.original.projection(&regime.functional)?.1;
        let mut writer = antecedent_core::CanonicalWriter::new("antecedent.panel.future_regime.v1");
        writer
            .bytes(regime.bridge.identity().as_bytes())
            .bytes(self.dependency.as_bytes())
            .text(&regime.mapping_evidence.to_hex())
            .word(weights.iter().map(Vec::len).sum::<usize>() as u64);
        for value in weights.iter().flatten() {
            writer.word(value.to_bits());
        }
        for value in regime.mapping.intercepts.iter().chain(&regime.mapping.slopes) {
            writer.word(value.to_bits());
        }
        Ok(PreparedRegime {
            source_values,
            intercepts: regime.mapping.intercepts.clone(),
            slopes: regime.mapping.slopes.clone(),
            means,
            identity: Some(SemanticDigest::from_bytes(writer.finish())),
        })
    }

    /// Rank candidate studies, each in its own regime, by the value of the update it would cause.
    ///
    /// Every study is sampled through its own licensed bridge and paired mapping. The current
    /// finite posterior is reweighted by that study's likelihood, the caller's terminal actions
    /// and utility are evaluated on every whole response row, and the gain over deciding now,
    /// minus the declared cost, is the net value. A regime whose bridge, transport premise,
    /// mapping pairing or evidence is not licensed refuses the whole ranking; it is never
    /// replaced by the observed estimand.
    ///
    /// # Errors
    /// Invalid catalog, replicate bounds or costs, an unlicensed or unaligned regime, stale
    /// current authority, resources, cancellation or callback failures.
    pub fn rank_next_studies<A: serde::Serialize>(
        &self,
        problem: &DecisionProblem<A, PanelExperimentDecisionState>,
        studies: &[PanelFutureStudy],
        replicates: usize,
        ledger: &EvidenceLedger,
        ctx: &ExecutionContext,
    ) -> Result<Vec<PanelFutureStudyValue>, JointPanelError> {
        self.retention.observe_current();
        let scientific = self.law_source(ledger)?;
        let coefficients = contrast_weights(self.original.request())?;
        let mut samples = Vec::with_capacity(studies.len());
        let mut means = Vec::with_capacity(studies.len());
        let mut identities = std::collections::BTreeMap::new();
        for study in studies {
            if study.regime.is_some() {
                let prepared = self.prepare_regime(study.regime.as_ref(), ledger)?;
                means.push(Some(prepared.means));
                identities.insert(study.sample.id.clone(), prepared.identity);
            } else {
                means.push(None);
                identities.insert(study.sample.id.clone(), None);
            }
            samples.push(study.sample.clone());
        }
        let values = rank_law_next_studies(
            &ExperimentRankingReceipt::from(&scientific),
            &self.law,
            &self.update,
            &coefficients,
            problem,
            &samples,
            &means,
            replicates,
            ctx,
        )?;
        Ok(values
            .into_iter()
            .map(|value| {
                let regime_identity = identities.get(&value.id).copied().flatten();
                PanelFutureStudyValue { value, regime_identity }
            })
            .collect())
    }

    /// Execute the update a candidate study would cause if it observed `observed_mean`.
    ///
    /// The result is the actual conditioned posterior over the same joint draws and the moved
    /// response means; deciding on those means is what the ranking's value averages over.
    ///
    /// # Errors
    /// An unlicensed regime, stale authority, a nonfinite outcome or an invalid likelihood.
    pub fn update_for_future_study(
        &self,
        study: &PanelFutureStudy,
        observed_mean: f64,
        ledger: &EvidenceLedger,
    ) -> Result<PanelFutureStudyUpdate, JointPanelError> {
        self.validate_current(ledger)?;
        if !observed_mean.is_finite()
            || study.sample.sample_size == 0
            || !study.sample.observation_variance.is_finite()
            || study.sample.observation_variance <= 0.0
        {
            return Err(JointPanelError::compile(
                "a hypothetical outcome needs a finite mean and a valid Gaussian sample model",
            ));
        }
        let prepared = self.prepare_regime(study.regime.as_ref(), ledger)?;
        let prior = self
            .law
            .metadata()
            .weights
            .clone()
            .ok_or_else(|| JointPanelError::compile("conditional law must retain weights"))?;
        let posterior = antecedent_prob::finite_belief::AffineGaussianExperiment {
            observed_mean,
            observation_variance: study.sample.observation_variance,
            sample_size: study.sample.sample_size,
            target_values: &prepared.source_values,
            intercepts: &prepared.intercepts,
            slopes: &prepared.slopes,
        }
        .condition(&prior)
        .map_err(JointPanelError::compile)?;
        let width = self.prior_means.len();
        let mut response_means = vec![0.0; width];
        for (row, weight) in self.law.draws().chunks_exact(width).zip(&posterior) {
            for (mean, value) in response_means.iter_mut().zip(row) {
                *mean += weight * value;
            }
        }
        if response_means.iter().any(|mean| !mean.is_finite()) {
            return Err(JointPanelError::compile("future-study response mean overflow"));
        }
        let mut prior_response_means = vec![0.0; width];
        let mass = prior.iter().sum::<f64>();
        for (row, weight) in self.law.draws().chunks_exact(width).zip(&prior) {
            for (mean, value) in prior_response_means.iter_mut().zip(row) {
                *mean += weight / mass * value;
            }
        }
        Ok(PanelFutureStudyUpdate {
            study_id: study.sample.id.clone(),
            observed_mean,
            posterior_weights: posterior,
            prior_response_means,
            response_means,
            regime_identity: prepared.identity,
        })
    }
}

fn mapped_means(
    source_values: &[f64],
    intercepts: &[f64],
    slopes: &[f64],
) -> Result<Vec<f64>, JointPanelError> {
    let means = source_values
        .iter()
        .zip(intercepts)
        .zip(slopes)
        .map(|((value, intercept), slope)| (value - intercept) / slope)
        .collect::<Vec<_>>();
    if means.iter().any(|mean| !mean.is_finite()) {
        return Err(JointPanelError::compile("future experiment mapped mean overflow"));
    }
    Ok(means)
}
