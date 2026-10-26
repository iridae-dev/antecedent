//! The one identification product a prepared [`Study`] carries.
//!
//! Preparation identifies a study once and installs exactly one product: the
//! static contrast, a PAG or CPDAG envelope, the temporal backdoor by horizon, a
//! temporal class envelope, a transport formula, or the per-atom products of a
//! graph posterior. The variants are mutually exclusive by construction, so the
//! study holds a single [`IdentificationCache`] rather than one optional field
//! per kind with an implicit priority between them.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_identify::IdentificationResult;

use super::Study;
use crate::analysis::prepared::{
    CachedCpdagIdentification, CachedDbnPosteriorIdentification,
    CachedGraphPosteriorIdentification, CachedPagIdentification, CachedStaticIdentification,
    CachedTemporalClassIdentification, CachedTemporalClassPosteriorIdentification,
    CachedTemporalIdentification, IdentificationCache,
};

/// Typed accessors for each variant, on the product and on the study that holds it.
macro_rules! slot_accessors {
    ($($variant:ident, $method:ident, $ty:ty, $doc:literal;)*) => {
        impl IdentificationCache {
            $(
                #[doc = $doc]
                pub(crate) fn $method(&self) -> Option<&Arc<$ty>> {
                    if let Self::$variant(cache) = self { Some(cache) } else { None }
                }
            )*
        }

        impl Study {
            $(
                #[doc = $doc]
                pub(crate) fn $method(&self) -> Option<&Arc<$ty>> {
                    self.identification.as_ref().and_then(IdentificationCache::$method)
                }
            )*
        }
    };
}

slot_accessors! {
    Static, static_cache, CachedStaticIdentification,
        "The prepared static contrast identification, when that is the installed product.";
    Pag, pag_cache, CachedPagIdentification,
        "The prepared PAG envelope, when that is the installed product.";
    Cpdag, cpdag_cache, CachedCpdagIdentification,
        "The prepared CPDAG envelope, when that is the installed product.";
    Temporal, temporal_cache, CachedTemporalIdentification,
        "The prepared temporal identification by horizon, when that is the installed product.";
    TemporalClass, temporal_class_cache, CachedTemporalClassIdentification,
        "The prepared temporal class envelope, when that is the installed product.";
    Transport, transport_cache, antecedent_identify::TransportIdentification,
        "The prepared transport formula, when that is the installed product.";
    GraphPosterior, graph_posterior_cache, CachedGraphPosteriorIdentification,
        "The prepared per-atom static graph-posterior identification, when installed.";
    DbnPosterior, dbn_posterior_cache, CachedDbnPosteriorIdentification,
        "The prepared per-atom DBN-posterior identification, when installed.";
    TemporalClassPosterior, temporal_class_posterior_cache,
        CachedTemporalClassPosteriorIdentification,
        "The prepared per-atom temporal class-posterior envelopes, when installed.";
}

impl IdentificationCache {
    /// The identification result that stands for the whole product, when it has one.
    ///
    /// A graph or DBN posterior answers with its first atom; a class envelope and a
    /// transport formula carry no single per-graph result.
    pub(crate) fn primary(&self) -> Option<&IdentificationResult> {
        match self {
            Self::Static(cache) => Some(&cache.identification),
            Self::Pag(cache) => Some(&cache.identification),
            Self::Cpdag(cache) => Some(&cache.identification),
            Self::Temporal(cache) => cache.by_horizon.first().map(|entry| &entry.identification),
            Self::DbnPosterior(cache) => cache.atoms.first().map(|atom| &atom.identification),
            Self::GraphPosterior(cache) => cache.atoms.first().map(|atom| &atom.identification),
            Self::TemporalClass(_) | Self::TemporalClassPosterior(_) | Self::Transport(_) => None,
        }
    }
}

impl Study {
    /// Install `cache` as the study's identification product, replacing any other.
    pub(crate) fn install_identification(&mut self, cache: IdentificationCache) {
        self.identification = Some(cache);
    }

    /// Drop the installed identification product, whatever it is.
    pub(crate) fn clear_identification(&mut self) {
        self.identification = None;
    }

    /// Drop the installed product when `drop_it` says it is of a kind the caller rebinds,
    /// leaving any other kind in place.
    pub(crate) fn clear_identification_where(
        &mut self,
        drop_it: impl FnOnce(&IdentificationCache) -> bool,
    ) {
        if self.identification.as_ref().is_some_and(drop_it) {
            self.identification = None;
        }
    }
}
