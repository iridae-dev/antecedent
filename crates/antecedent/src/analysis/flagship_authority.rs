//! Current source authority for a flagship decision, separate from numerical replay.
use antecedent_contract::evidence_ledger::{
    CombinationRequest, EvidenceLedger, LedgerError, ResultId, SourceDeclaration, SourceId,
    SynthesisOperation,
};
use serde::{Deserialize, Serialize};

/// Source revisions and the ledger result authorizing a specific numerical payload.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionAuthorityBinding {
    /// Ledger result identity, including the payload identity in its label.
    pub result: String,
    /// Exact content-addressed source revisions, not mutable source names.
    pub sources: Vec<String>,
}

/// Caller-held current authority. An artifact's historical ledger snapshot cannot replace it.
#[derive(Debug)]
pub struct DecisionAuthority {
    ledger: EvidenceLedger,
    sources: [SourceId; 3],
    result: ResultId,
}

impl DecisionAuthority {
    pub(super) fn new(
        declarations: [SourceDeclaration; 3],
        payload: &str,
    ) -> Result<Self, LedgerError> {
        let mut ledger = EvidenceLedger::new();
        let mut sources = Vec::new();
        for declaration in declarations {
            sources.push(ledger.add_source(declaration)?);
        }
        let sources = [sources[0], sources[1], sources[2]];
        let result = register(&mut ledger, sources, payload, None)?;
        Ok(Self { ledger, sources, result })
    }

    pub(super) fn require_valid(&self) -> Result<(), LedgerError> {
        self.ledger.require_valid(self.result).map(|_| ())
    }

    pub(super) fn rebind(
        &mut self,
        declarations: [SourceDeclaration; 3],
        payload: &str,
    ) -> Result<(), LedgerError> {
        self.require_valid()?;
        let mut ledger = self.ledger.clone();
        let mut sources = self.sources;
        for (coordinate, mut replacement) in declarations.into_iter().enumerate() {
            if let antecedent_contract::evidence_ledger::ObservationBasis::Declared(rows) =
                &mut replacement.observations
            {
                rows.sort_unstable();
                rows.dedup();
            }
            if ledger
                .source(sources[coordinate])
                .is_some_and(|source| source.declaration != replacement)
            {
                let applied = ledger.replace_source(sources[coordinate], replacement)?;
                sources[coordinate] =
                    applied.effect.created_source.expect("replacement records source");
            }
        }
        let result = register(&mut ledger, sources, payload, Some(self.result))?;
        self.ledger = ledger;
        self.sources = sources;
        self.result = result;
        Ok(())
    }

    pub(super) fn binding(&self) -> Result<DecisionAuthorityBinding, LedgerError> {
        self.require_valid()?;
        Ok(DecisionAuthorityBinding {
            result: self.result.to_hex(),
            sources: self.sources.iter().map(SourceId::to_hex).collect(),
        })
    }

    /// Read the ledger for lineage and downstream classification.
    #[must_use]
    pub fn ledger(&self) -> &EvidenceLedger {
        &self.ledger
    }

    /// Withdraw a contributing source; trust loss uses the same irreversible retirement.
    ///
    /// # Errors
    /// An unknown coordinate or an invalid ledger operation.
    pub fn withdraw_source(&mut self, coordinate: usize, reason: &str) -> Result<(), LedgerError> {
        if let Some(source) = self.sources.get(coordinate) {
            self.ledger.withdraw_source(*source, reason)?;
            return Ok(());
        }
        Err(invalid_authority("source coordinate outside the flagship source set"))
    }

    /// Supersede a source with a new content revision. Existing decisions require recomputation.
    ///
    /// # Errors
    /// Unknown/retired source or malformed replacement.
    pub fn supersede_source(
        &mut self,
        coordinate: usize,
        replacement: SourceDeclaration,
    ) -> Result<(), LedgerError> {
        let source = self.sources.get(coordinate).copied().ok_or_else(|| {
            invalid_authority("source coordinate outside the flagship source set")
        })?;
        self.ledger.supersede_source(source, replacement)?;
        Ok(())
    }

    pub(super) fn validate(
        &self,
        binding: &DecisionAuthorityBinding,
        payload: &str,
    ) -> Result<(), LedgerError> {
        let id = ResultId::from_hex(&binding.result)
            .ok_or_else(|| invalid_authority("malformed result identity"))?;
        let result = self.ledger.require_valid(id)?;
        if result.label != label(payload)
            || self
                .ledger
                .result_sources(id)
                .iter()
                .map(SourceId::to_hex)
                .collect::<std::collections::BTreeSet<_>>()
                != binding.sources.iter().cloned().collect()
        {
            return Err(LedgerError {
                kind: antecedent_contract::evidence_ledger::RefusalKind::InvalidArgument,
                code: antecedent_core::reason_code!("invalid_argument"),
                detail: "flagship_decision.authority_mismatch",
                message: "decision payload or source revisions differ from current authority"
                    .into(),
                sources: vec![],
            });
        }
        Ok(())
    }
}

fn invalid_authority(message: &str) -> LedgerError {
    LedgerError {
        kind: antecedent_contract::evidence_ledger::RefusalKind::InvalidArgument,
        code: antecedent_core::reason_code!("invalid_argument"),
        detail: "flagship_decision.authority_mismatch",
        message: message.into(),
        sources: vec![],
    }
}

fn label(payload: &str) -> String {
    format!("flagship_decision:{payload}")
}

fn register(
    ledger: &mut EvidenceLedger,
    sources: [SourceId; 3],
    payload: &str,
    replaces: Option<ResultId>,
) -> Result<ResultId, LedgerError> {
    ledger.combine(CombinationRequest {
        label: label(payload),
        operation: SynthesisOperation::RepeatedObservationUse {
            sources: sources.to_vec(),
            stage: "flagship_decision".into(),
        },
        replaces,
    })
}
