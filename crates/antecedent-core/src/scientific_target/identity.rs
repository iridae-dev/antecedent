//! The target form, the packaged [`ScientificTarget`], and its semantic identity.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::canonical_encode::CanonicalEncode;
use crate::identity::{CanonicalWriter, SemanticDigest};
use std::hash::{Hash, Hasher};

use super::{
    DoseTimeSurface, HistoryIndexedTargetLaw, InferenceSelection, InterventionContrast,
    InterventionWorld, MediationTarget, ModelAssumptions, TimeFrame, WorldCollection,
};
use super::{SCIENTIFIC_TARGET_FORMAT, SCIENTIFIC_TARGET_IDENTITY_TAG};

/// Which scientific object a [`ScientificTarget`] is.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum TargetForm {
    /// One world, `E[Y(a)]` under that world's action.
    World(InterventionWorld),
    /// Aligned worlds before a contrast or surface is chosen.
    Worlds(WorldCollection),
    /// Difference or reference interaction.
    Contrast(InterventionContrast),
    /// Declared dose × time grid.
    DoseTime(DoseTimeSurface),
    /// Mediation, including its cross-world premises.
    Mediation(MediationTarget),
}

/// Scientific target. Equality and [`Self::identity`] ignore [`InferenceSelection`]
/// and include a history-indexed target law when one is attached.
#[derive(Clone, Debug)]
pub struct ScientificTarget {
    form: TargetForm,
    assumptions: ModelAssumptions,
    inference: InferenceSelection,
    history_indexed_law: Option<HistoryIndexedTargetLaw>,
}

impl PartialEq for ScientificTarget {
    fn eq(&self, other: &Self) -> bool {
        self.form == other.form
            && self.assumptions == other.assumptions
            && self.history_indexed_law == other.history_indexed_law
    }
}

impl Eq for ScientificTarget {}

impl Hash for ScientificTarget {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.form.hash(state);
        self.assumptions.hash(state);
        self.history_indexed_law.hash(state);
    }
}

impl ScientificTarget {
    /// Package a validated form. This does not identify or estimate it.
    #[must_use]
    pub const fn new(
        form: TargetForm,
        assumptions: ModelAssumptions,
        inference: InferenceSelection,
    ) -> Self {
        Self { form, assumptions, inference, history_indexed_law: None }
    }

    /// Scientific object.
    #[must_use]
    pub const fn form(&self) -> &TargetForm {
        &self.form
    }

    /// Graph and model assumptions.
    #[must_use]
    pub const fn assumptions(&self) -> &ModelAssumptions {
        &self.assumptions
    }

    /// Realization label. Excluded from [`Self::eq`] and [`Self::identity`].
    #[must_use]
    pub const fn inference(&self) -> InferenceSelection {
        self.inference
    }

    /// Same science under a different realization label.
    #[must_use]
    pub const fn with_inference(mut self, inference: InferenceSelection) -> Self {
        self.inference = inference;
        self
    }

    /// Attach the observational law over history-indexed experiments.
    ///
    /// The law is not an [`InitialLaw`] and is not filled in from unit histories.
    #[must_use = "the history-indexed law is part of the target"]
    pub fn with_history_indexed_law(mut self, law: HistoryIndexedTargetLaw) -> Self {
        self.history_indexed_law = Some(law);
        self
    }

    /// Law over history-indexed experiments, when the target declares one.
    #[must_use = "absence means unit histories have not supplied a target law"]
    pub const fn history_indexed_law(&self) -> Option<&HistoryIndexedTargetLaw> {
        self.history_indexed_law.as_ref()
    }

    /// Horizon shared by every world, when every frame is temporal and they agree.
    ///
    /// A static frame, an empty target, or two different horizons yield [`None`].
    #[must_use = "the horizon route admits only a shared horizon of two"]
    pub fn shared_temporal_horizon(&self) -> Option<u32> {
        let mut shared = None;
        for frame in time_frames(&self.form) {
            let horizon = frame.horizon_steps()?;
            match shared {
                None => shared = Some(horizon),
                Some(existing) if existing != horizon => return None,
                Some(_) => {}
            }
        }
        shared
    }

    /// BLAKE3 digest of the scientific fields via [`CanonicalWriter`].
    ///
    /// Population, worlds, horizon, history, initial law, quantity, conditioning,
    /// graph, and the set of assumptions and scopes enter the digest. A world, a
    /// world collection, a dose × time surface, and a contrast are hashed by
    /// scientific content. A caller-chosen world id or a reference flag does not.
    /// A world collection is that set of contents. A dose × time surface hashes
    /// each cell's content in grid order. Inference selection, assumption source
    /// and status, and record order do not. A history-indexed target law is
    /// appended only when one is attached, so a target without one keeps the
    /// format-3 digest it had before that law existed.
    #[must_use]
    pub fn identity(&self) -> SemanticDigest {
        let mut writer = CanonicalWriter::new(SCIENTIFIC_TARGET_IDENTITY_TAG);
        writer.word(u64::from(SCIENTIFIC_TARGET_FORMAT));
        self.form.encode(&mut writer);
        self.assumptions.encode(&mut writer);
        if let Some(law) = &self.history_indexed_law {
            law.encode(&mut writer);
        }
        SemanticDigest::from_bytes(writer.finish())
    }
}

fn time_frames(form: &TargetForm) -> Vec<&TimeFrame> {
    match form {
        TargetForm::World(world) => vec![world.time()],
        TargetForm::Worlds(worlds) => worlds.worlds().iter().map(InterventionWorld::time).collect(),
        TargetForm::Contrast(contrast) => {
            contrast.worlds().worlds().iter().map(InterventionWorld::time).collect()
        }
        TargetForm::DoseTime(surface) => {
            surface.worlds().worlds().iter().map(InterventionWorld::time).collect()
        }
        TargetForm::Mediation(mediation) => vec![mediation.time()],
    }
}

impl CanonicalEncode for TargetForm {
    fn encode(&self, writer: &mut CanonicalWriter) {
        match self {
            Self::World(world) => {
                writer.word(1);
                world.encode(writer);
            }
            Self::Worlds(worlds) => {
                writer.word(2);
                worlds.encode(writer);
            }
            Self::Contrast(contrast) => {
                writer.word(3);
                contrast.encode(writer);
            }
            Self::DoseTime(surface) => {
                writer.word(4);
                surface.encode(writer);
            }
            Self::Mediation(mediation) => {
                writer.word(5);
                mediation.encode(writer);
            }
        }
    }
}
