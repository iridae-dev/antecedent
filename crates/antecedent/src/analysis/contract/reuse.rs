//! Prepared result reuse identities and snapshot-bound retargeting.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{ContractIdentities, ExecutionContext};
use antecedent_io::{
    ScoreReuseIdentityWire, TargetWeightsIdentityWire, TargetWeightsSectionWire,
    score_reuse_digest, target_weights_digest,
};

use crate::error::CausalError;
use crate::result::{RowWeightsBinding, StudyResult};

use super::{CausalContract, PreparedStudy, ReuseSection, io_err};

impl PreparedStudy {
    /// Score-table reuse key. Stricter than identification: folds, rows,
    /// nuisance provenance, the inference binding (estimator options, overlap, backend)
    /// and the snapshot are part of the digest.
    ///
    /// # Errors
    ///
    /// Canonical-encoding failures.
    pub fn score_reuse_identity(
        &self,
    ) -> Result<Option<antecedent_core::SemanticDigest>, CausalError> {
        if self.score_table().is_none() {
            return Ok(None);
        }
        let identities = self.contract()?.identities;
        self.score_reuse_wire(&identities)
            .map(|wire| score_reuse_digest(&wire).map_err(|e| io_err(&e)))
            .transpose()
    }

    /// Reuse key of a compatible coded adjusted mean model, separate from score-table reuse.
    /// It binds the frozen rows, design, roles and estimator-specific uncertainty configuration.
    /// A key alone supplies no fitted state to a fresh process.
    ///
    /// # Errors
    /// Canonical encoding or contract construction failure.
    pub fn adjusted_fit_reuse_identity(
        &self,
    ) -> Result<Option<antecedent_core::SemanticDigest>, CausalError> {
        let identities = self.contract()?.identities;
        self.adjusted_fit_reuse_wire(&identities)
            .as_ref()
            .map(antecedent_io::adjusted_fit_reuse_digest)
            .transpose()
            .map_err(|e| io_err(&e))
    }

    fn retarget_reuse_digest(
        &self,
        identities: &ContractIdentities,
    ) -> Result<Option<antecedent_core::SemanticDigest>, CausalError> {
        if let Some(score) = self.score_reuse_wire(identities) {
            return score_reuse_digest(&score).map(Some).map_err(|e| io_err(&e));
        }
        self.adjusted_fit_reuse_wire(identities)
            .as_ref()
            .map(antecedent_io::adjusted_fit_reuse_digest)
            .transpose()
            .map_err(|e| io_err(&e))
    }

    /// Score-reuse payload for this handle's score table under `identities`.
    fn score_reuse_wire(&self, identities: &ContractIdentities) -> Option<ScoreReuseIdentityWire> {
        let table = self.score_table()?;
        let row_ids: Vec<u64> = table.row_index.iter().map(|id| id.get()).collect();
        Some(ScoreReuseIdentityWire::score_table(&antecedent_io::ScoreTableKey {
            identification: identities.identification,
            data_snapshot: identities.data_snapshot,
            row_index: &row_ids,
            fold_ids: &table.fold_ids,
            n_folds: table.n_folds,
            adjustment_set: &table.adjustment_set,
            nuisance_provenance: table.nuisance_provenance.as_ref(),
            treatment: table.treatment,
            intervened: &table.intervened,
            bootstrap_replicates: self.study().settings.bootstrap_replicates,
            inference_binding: identities.inference_binding,
        }))
    }

    /// The reuse layers this handle exports beside a result, advertised on the
    /// contract so the seal and the claim id cover them.
    pub(super) fn bind_reuse(
        &self,
        contract: &mut CausalContract,
        result: &StudyResult,
    ) -> Result<ReuseSection, CausalError> {
        let mut reuse = ReuseSection::default();
        if let Some(score) = self.score_reuse_wire(&contract.identities) {
            contract.score_reuse = Some(score_reuse_digest(&score).map_err(|err| io_err(&err))?);
            reuse.score_reuse = Some(score);
        } else if let Some(fit) = self.adjusted_fit_reuse_wire(&contract.identities) {
            contract.score_reuse =
                Some(antecedent_io::adjusted_fit_reuse_digest(&fit).map_err(|e| io_err(&e))?);
            reuse.adjusted_fit_reuse = Some(fit);
        }
        if let Some(binding) = &result.extras.row_weights {
            contract.target_weights = Some(binding.target_weights);
            reuse.target_weights = Some(TargetWeightsSectionWire {
                identity: binding.identity.clone(),
                values: binding.weights.to_vec(),
            });
        }
        if let Some(checked) = self.execution().checked_aipw() {
            // The cross-fitted procedure publishes its complete-case rows on the
            // score table; the trimmed common-support procedure has no table and
            // binds the rows its preparation retained.
            let rows = match (
                result.estimate.as_effect().and_then(|effect| effect.scores.table.as_ref()),
                checked.lowering().procedure,
            ) {
                (Some(table), _) => table.row_index.to_vec(),
                (None, antecedent_estimate::CheckedAipwProcedure::TrimmedLogisticOls) => {
                    checked.lowering().rows.to_vec()
                }
                (None, antecedent_estimate::CheckedAipwProcedure::CrossFittedLogisticOls) => {
                    return Err(CausalError::Compile {
                        message:
                            "checked AIPW execution did not retain its complete-case score rows"
                                .into(),
                    });
                }
            };
            let rows = antecedent_io::CheckedAipwRowsWire {
                format: 1,
                rows: rows.iter().map(|id| id.get()).collect(),
                data_snapshot: *contract.identities.data_snapshot.as_bytes(),
            };
            contract.checked_aipw_rows =
                Some(antecedent_io::checked_aipw_rows_digest(&rows).map_err(|err| io_err(&err))?);
            reuse.checked_aipw_rows = Some(rows);
        }
        Ok(reuse)
    }

    /// Bind row weights to this handle's data snapshot and score table.
    ///
    /// # Errors
    ///
    /// No score table, or canonical-encoding failures.
    pub(crate) fn row_weights_binding(
        &self,
        weights: &[f64],
        depends_on: &[antecedent_core::VariableId],
    ) -> Result<RowWeightsBinding, CausalError> {
        let identity = self.row_weights_identity(weights, depends_on)?;
        let target_weights = target_weights_digest(&identity).map_err(|err| io_err(&err))?;
        Ok(RowWeightsBinding {
            weights: Arc::from(weights),
            depends_on: Arc::from(depends_on),
            identity,
            target_weights,
        })
    }

    fn row_weights_identity(
        &self,
        weights: &[f64],
        depends_on: &[antecedent_core::VariableId],
    ) -> Result<TargetWeightsIdentityWire, CausalError> {
        let identities = self.contract()?.identities;
        let score=self.retarget_reuse_digest(&identities)?.ok_or(CausalError::Unsupported {
            message:"row-weight target requires a prepared score table (frozen scores) or a compatible retained adjusted fit",
        })?;
        Ok(TargetWeightsIdentityWire::new(weights, identities.data_snapshot, score, depends_on))
    }

    /// Re-execute the row-weight retarget an exported contract carries.
    ///
    /// Row weights index the rows of one data snapshot and one score table:
    /// this handle must hold that snapshot and that table.
    ///
    /// # Errors
    ///
    /// `reason=row_weights_bound_to_snapshot` when this handle holds a
    /// different snapshot or score table; otherwise the refusals of
    /// [`Self::retarget`].
    pub fn reexecute_retarget(
        &self,
        section: &TargetWeightsSectionWire,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: "reexecute_retarget" });
        }
        // Both representations are live while the expected identity is compared:
        // native VariableIds and the raw-u32 dependencies in the identity payload.
        // The artifact's weights are borrowed rather than copied for this check.
        let _dependencies = ctx
            .acquire_admission_product(
                0,
                u64::try_from(section.identity.depends_on.len()).map_err(|_| {
                    CausalError::Resource { message: "retarget dependency count overflow".into() }
                })?,
                u64::try_from(size_of::<antecedent_core::VariableId>() + size_of::<u32>())
                    .map_err(|_| CausalError::Resource {
                        message: "retarget dependency width overflow".into(),
                    })?,
            )
            .map_err(|error| match error {
                antecedent_core::execution::AdmissionError::Cancelled => {
                    CausalError::Cancelled { stage: "reexecute_retarget" }
                }
                error => CausalError::Resource { message: error.to_string() },
            })?;
        let depends_on: Vec<antecedent_core::VariableId> = section
            .identity
            .depends_on
            .iter()
            .copied()
            .map(antecedent_core::VariableId::from_raw)
            .collect();
        let identity = self.row_weights_identity(&section.values, &depends_on)?;
        if identity != section.identity {
            return Err(row_weights_bound_to_snapshot!());
        }
        self.retarget(&section.values, &depends_on, ctx)
    }
}
