//! Consumers of original F18 effects, without a new constancy or transfer guarantee.
//!
//! An independently consumed F18 artifact is required. Transport contrasts are diagnostic,
//! source-bank preferences are explicitly data dependent, and policy comparisons are affine
//! point evaluations. Neither non-rejection nor agreement licenses transport or pooling.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;

use antecedent_core::ScientificQuantity;
use antecedent_design::decision_contract::{DecisionContract, UtilityExpr};
use antecedent_design::decision_eval::{
    DecisionResult, MeanSource, Verdict, evaluate_contract_on_means,
};
use antecedent_estimate::effect_constancy::{ContrastReport, PartitionReport, PartitionSupport};
use antecedent_io::prior_bank::{CompatibilityReport, PriorCatalog, TargetDesign};
use serde::{Deserialize, Serialize};

use super::effect_constancy::{EffectConstancy, EffectConstancyIdentity};

/// Original evidence identity and unchanged inferential standing on every consumer result.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ConstancyEvidence {
    /// Identity retained independently and verified by the original artifact consumer.
    pub identity: EffectConstancyIdentity,
    /// Point-only inference; this is not a measured heterogeneity guarantee.
    pub inference_claim: String,
    /// Original F18 calibration status, currently unmeasured.
    pub calibration: String,
    /// Original non-rejection and power caveats.
    pub caveats: Vec<String>,
}

/// Original covariance-aware, family-adjusted contrast for a transport review.
#[derive(Clone, Debug, PartialEq)]
pub struct TransportConstancyDiagnostic {
    /// Original F18 evidence and unchanged limitations.
    pub evidence: ConstancyEvidence,
    /// Original declared contrast, with its original orientation and Holm family.
    pub contrast: ContrastReport,
    /// Always true: this diagnostic never establishes a transport theorem.
    pub separate_transport_identification_required: bool,
}

/// Explicit association of a prior-bank entry with an original F18 partition.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PriorPartitionBinding {
    /// Exact original catalog artifact identity.
    pub artifact_id: String,
    /// Original F18 partition label, rather than a reconstructed effect.
    pub partition: String,
}

/// Original compatibility filtering and advisory effect-similarity ranking.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ConstancyPriorRanking {
    /// Original F18 evidence and unchanged limitations.
    pub evidence: ConstancyEvidence,
    /// Original compatibility reports, in original catalog order.
    pub compatibility: Vec<CompatibilityReport>,
    /// Negative absolute effect differences; higher is closer to the declared target.
    pub scores: Vec<(String, f64)>,
    /// Original bank ranking; incompatible entries remain excluded.
    pub ranked: Vec<CompatibilityReport>,
    /// Explicitly true: selecting a prior with target estimates is data dependent.
    pub data_dependent_selection: bool,
    /// Always false; ranking supplies no automatic pooling or posterior transfer license.
    pub posterior_transfer_licensed: bool,
}

/// One original supported effect and the actual input supplied to the policy engine.
#[derive(Clone, Debug, PartialEq)]
pub struct ConstancyPolicyPartition {
    /// Original partition label.
    pub label: String,
    /// Original partition coordinate, not inferred from its label.
    pub coordinate: String,
    /// Actual point source used by the original engine.
    pub source: MeanSource,
    /// Original affine policy evaluation.
    pub result: DecisionResult,
}

/// Point policy results over every original fully supported partition.
#[derive(Clone, Debug, PartialEq)]
pub struct ConstancyPolicyReview {
    /// Original F18 evidence and unchanged limitations.
    pub evidence: ConstancyEvidence,
    /// Each original partition and its original affine decision evaluation.
    pub partitions: Vec<ConstancyPolicyPartition>,
    /// All exactly tied admissible leaders common to every partition; may be empty.
    pub common_leaders: Vec<String>,
    /// Always false: agreement among point estimates is not an optimality guarantee.
    pub generalization_guarantee: bool,
}

/// Original-artifact or downstream declaration failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConstancyConsumerError {
    /// Stable original reason code.
    pub code: &'static str,
    /// Stable consumer detail.
    pub detail: &'static str,
    /// Retained original consumer diagnostic.
    pub message: String,
}
impl std::fmt::Display for ConstancyConsumerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}: {}", self.code, self.detail, self.message)
    }
}
impl std::error::Error for ConstancyConsumerError {}

fn invalid(detail: &'static str, message: impl Into<String>) -> ConstancyConsumerError {
    ConstancyConsumerError {
        code: antecedent_core::reason_code!("invalid_argument"),
        detail,
        message: message.into(),
    }
}

fn bounded_policy(contract: &DecisionContract) -> Result<(), ConstancyConsumerError> {
    let refuse =
        || invalid("effect_constancy_consumer.limits_exceeded", "bounded affine policy required");
    if contract.actions.len() > 64 || !contract.constraints.is_empty() {
        return Err(refuse());
    }
    let mut nodes = 0usize;
    for action in &contract.actions {
        if action.inputs.len() > 16 {
            return Err(refuse());
        }
        let mut pending = vec![(&action.utility, 0usize)];
        while let Some((expr, depth)) = pending.pop() {
            nodes += 1;
            if nodes > 2048 || depth > 64 {
                return Err(refuse());
            }
            match expr {
                UtilityExpr::Const(_) | UtilityExpr::Input(_) => {}
                UtilityExpr::Neg(child) => pending.push((child, depth + 1)),
                UtilityExpr::Add(a, b)
                | UtilityExpr::Sub(a, b)
                | UtilityExpr::Mul(a, b)
                | UtilityExpr::Min(a, b)
                | UtilityExpr::Max(a, b) => {
                    pending.push((a, depth + 1));
                    pending.push((b, depth + 1));
                }
            }
        }
    }
    Ok(())
}

/// A freshly verified original F18 artifact used by real downstream engines.
#[derive(Clone, Debug)]
pub struct ConstancyConsumers {
    original: EffectConstancy,
}
impl ConstancyConsumers {
    /// Verify full original artifact evidence against a separately retained identity.
    ///
    /// # Errors
    /// Original artifact corruption, changed evidence identity or non-replaying values.
    pub fn consume(
        bytes: &[u8],
        expected: &EffectConstancyIdentity,
    ) -> Result<Self, ConstancyConsumerError> {
        let original = EffectConstancy::consume(bytes, Some(expected)).map_err(|error| {
            invalid("effect_constancy_consumer.artifact_invalid", error.to_string())
        })?;
        Ok(Self { original })
    }

    fn evidence(&self) -> ConstancyEvidence {
        let report = self.original.report();
        ConstancyEvidence {
            identity: self.original.identity().clone(),
            inference_claim: report.inference_claim,
            calibration: report.calibration,
            caveats: report.caveats,
        }
    }

    fn partition(&self, label: &str) -> Result<&PartitionReport, ConstancyConsumerError> {
        let partition = self
            .original
            .result()
            .partitions
            .iter()
            .find(|p| p.label == label)
            .ok_or_else(|| invalid("effect_constancy_consumer.partition_missing", label))?;
        if partition.support != PartitionSupport::Supported {
            return Err(ConstancyConsumerError {
                code: antecedent_core::reason_code!("route_not_supported"),
                detail: "effect_constancy_consumer.support_unsupported",
                message: label.into(),
            });
        }
        Ok(partition)
    }

    /// Read a selected original contrast for an explicitly separate transport review.
    ///
    /// # Errors
    /// Unknown/partial partition or a contrast absent from the original declared family.
    pub fn transport_diagnostic(
        &self,
        left: &str,
        right: &str,
    ) -> Result<TransportConstancyDiagnostic, ConstancyConsumerError> {
        self.partition(left)?;
        self.partition(right)?;
        let contrast = self
            .original
            .result()
            .contrasts
            .iter()
            .find(|c| c.left == left && c.right == right)
            .ok_or_else(|| {
                invalid(
                    "effect_constancy_consumer.contrast_missing",
                    "the original oriented contrast is not in the declared family",
                )
            })?;
        Ok(TransportConstancyDiagnostic {
            evidence: self.evidence(),
            contrast: contrast.clone(),
            separate_transport_identification_required: true,
        })
    }

    /// Filter the original prior catalog, then rank usable entries by declared effect proximity.
    ///
    /// Does not synthesize a posterior from standard errors, pool studies, or feed target
    /// evidence back as an independent prior. Actual prior transfer still uses its original
    /// artifact, overlap and causal compatibility checks.
    ///
    /// # Errors
    /// Unbounded catalog, duplicate/missing source bindings or unsupported partitions.
    pub fn rank_prior_sources(
        &self,
        catalog: &PriorCatalog,
        target: &TargetDesign,
        target_partition: &str,
        bindings: &[PriorPartitionBinding],
    ) -> Result<ConstancyPriorRanking, ConstancyConsumerError> {
        if catalog.sources.len() > 1024 || bindings.len() != catalog.sources.len() {
            return Err(invalid(
                "effect_constancy_consumer.catalog_binding_invalid",
                "one explicit partition binding per bounded catalog entry is required",
            ));
        }
        let target_effect = self.partition(target_partition)?.effect;
        let mut seen = BTreeSet::new();
        let mut scores = Vec::with_capacity(bindings.len());
        for source in &catalog.sources {
            if !seen.insert(source.meta.artifact_id.as_str()) {
                return Err(invalid(
                    "effect_constancy_consumer.catalog_binding_invalid",
                    "duplicate catalog identity",
                ));
            }
            let mut matching = bindings.iter().filter(|b| b.artifact_id == source.meta.artifact_id);
            let binding = matching.next().ok_or_else(|| {
                invalid(
                    "effect_constancy_consumer.catalog_binding_invalid",
                    "missing source partition",
                )
            })?;
            if matching.next().is_some() {
                return Err(invalid(
                    "effect_constancy_consumer.catalog_binding_invalid",
                    "duplicate source binding",
                ));
            }
            let score = -(self.partition(&binding.partition)?.effect - target_effect).abs();
            if !score.is_finite() {
                return Err(invalid(
                    "effect_constancy_consumer.catalog_binding_invalid",
                    "nonfinite effect difference",
                ));
            }
            scores.push((source.meta.artifact_id.clone(), score));
        }
        let compatibility = catalog.filter_compatible(target);
        let ranked = catalog.rank(&compatibility, &scores);
        Ok(ConstancyPriorRanking {
            evidence: self.evidence(),
            compatibility,
            scores,
            ranked,
            data_dependent_selection: true,
            posterior_transfer_licensed: false,
        })
    }

    /// Run the original affine mean decision on each original supported partition effect.
    ///
    /// The explicitly supplied effect coordinate must preserve the original common estimand,
    /// units, population and regime. All tied leaders use the original exact comparison.
    ///
    /// # Errors
    /// Semantic mismatch, partial support, invalid/non-affine contract or unavailable law.
    pub fn policy_review(
        &self,
        contract: &DecisionContract,
        effect: &ScientificQuantity,
    ) -> Result<ConstancyPolicyReview, ConstancyConsumerError> {
        bounded_policy(contract)?;
        let estimand = &self.original.result().estimand;
        if effect.validate().is_err()
            || effect.functional_id != estimand.estimand
            || effect.units != estimand.units
            || effect.population_id != estimand.population
            || effect.regime_id != estimand.regime
            || effect.transform_id != "identity"
            || !effect.conditioning.is_empty()
        {
            return Err(invalid(
                "effect_constancy_consumer.quantity_mismatch",
                "explicit effect coordinate must preserve the original estimand",
            ));
        }
        let mut common: Option<BTreeSet<String>> = None;
        let mut partitions = Vec::with_capacity(self.original.result().partitions.len());
        for original in &self.original.result().partitions {
            let partition = self.partition(&original.label)?;
            let source = MeanSource {
                coordinates: vec![effect.clone()],
                means: vec![partition.effect],
                provider_id: "effect_constancy_original_partition".into(),
                snapshot_id: format!(
                    "{}:partition:{}:{}:{}",
                    self.original.identity().evidence_id,
                    partition.label.len(),
                    partition.label,
                    partition.coordinate
                ),
                causal_contract_id: self.original.identity().estimand_id.clone(),
                rng_id: "no_draws_point_effect".into(),
            };
            let result = evaluate_contract_on_means(contract, &source).map_err(|error| {
                invalid("effect_constancy_consumer.decision_unsupported", format!("{error:?}"))
            })?;
            let leaders: BTreeSet<String> = match &result.verdict {
                Verdict::UniquelyOptimal(id) => [id.clone()].into_iter().collect(),
                Verdict::Indistinguishable(ids) => ids.iter().cloned().collect(),
                Verdict::NoAdmissibleAction => BTreeSet::new(),
            };
            common = Some(match common {
                None => leaders,
                Some(old) => old.intersection(&leaders).cloned().collect(),
            });
            partitions.push(ConstancyPolicyPartition {
                label: partition.label.clone(),
                coordinate: partition.coordinate.clone(),
                source,
                result,
            });
        }
        Ok(ConstancyPolicyReview {
            evidence: self.evidence(),
            partitions,
            common_leaders: common.unwrap_or_default().into_iter().collect(),
            generalization_guarantee: false,
        })
    }
}
