//! Layer four of a propagation-chain program: the declared chain, its dependence
//! contract and the record identity the execution seals.
//!
//! The chain's stages are the statistical and evidence program; the dependence
//! contract (which rows, clusters, models, priors, disturbances and periods
//! each stage inherits, and what every combining stage shares) is the
//! identification of overlap. The program carries the draft record, so an
//! execution seals the record the program was lowered from and no other: the
//! record's identity (declaration, draws, future-study draws and changes) is
//! part of the lowered identity.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_contract::propagation_chain::ChainPlan;
use antecedent_core::CanonicalWriter;
use antecedent_io::propagation_chain_record::{ChainDeferredTableWire, PropagationChainRecordWire};

/// The chain a program was lowered from.
#[derive(Clone, Debug, PartialEq)]
pub struct ChainProgram {
    draft: Arc<PropagationChainRecordWire>,
    /// The family table a deferred draft binds at execution.
    deferred: Option<Arc<ChainDeferredTableWire>>,
    plan: ChainPlan,
    declaration_identity: Arc<str>,
}

impl ChainProgram {
    pub(crate) fn new(
        draft: PropagationChainRecordWire,
        deferred: Option<ChainDeferredTableWire>,
        plan: ChainPlan,
        declaration_identity: String,
    ) -> Self {
        Self {
            draft: Arc::new(draft),
            deferred: deferred.map(Arc::new),
            plan,
            declaration_identity: Arc::from(declaration_identity),
        }
    }

    /// The declared chain: sources, stages, future-study draws and changes, with
    /// no results.
    #[must_use]
    pub fn draft(&self) -> &PropagationChainRecordWire {
        &self.draft
    }

    /// The family table the draft binds at execution, when its table is deferred.
    #[must_use]
    pub fn deferred(&self) -> Option<&ChainDeferredTableWire> {
        self.deferred.as_deref()
    }

    /// What the chain rule decided for every stage.
    #[must_use]
    pub const fn plan(&self) -> &ChainPlan {
        &self.plan
    }

    /// The identity the sealed record carries: it covers everything the program
    /// was lowered from and none of the results.
    #[must_use]
    pub fn declaration_identity(&self) -> &str {
        &self.declaration_identity
    }

    /// Stages that depend on `source`: the ones a change to it reaches.
    #[must_use]
    pub fn dependents_of(&self, source: &str) -> Vec<&str> {
        self.plan.dependents_of(source)
    }

    pub(crate) fn write(&self, writer: &mut CanonicalWriter) {
        writer.text(&self.declaration_identity).text(self.plan.label);
        writer.word(self.plan.stages.len() as u64);
        for stage in &self.plan.stages {
            writer.text(&stage.id).text(stage.kind.as_str()).text(stage.representation.as_str());
            writer.texts(&stage.sources);
            match &stage.dependence {
                Some(dependence) => {
                    // The status by its fixed identity word, the variant name the released
                    // identity hashed through `Debug`: the bytes are unchanged, and a renamed
                    // or added variant can no longer change them silently.
                    writer
                        .word(1)
                        .text(dependence.status.identity_word())
                        .texts(&dependence.shared);
                }
                None => {
                    writer.word(0);
                }
            }
        }
    }
}
