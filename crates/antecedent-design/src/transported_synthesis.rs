//! Transport-then-synthesize: a transported claim enters evidence synthesis
//! with its transport identity and its extra uncertainty counted once.
//!
//! The runtime evidence ledger is the only overlap model. Before any number is
//! combined, the combination is registered with the ledger, which refuses a
//! source that would count as native target data and as a transported claim,
//! overlapping prior and likelihood, and unknown overlap. The numerics are
//! `antecedent_stats::synthesis_transport`: each transport's variance enters
//! once per transport, a transport serving several claims shares one error,
//! and a transport declared with no uncertainty is refused rather than read as
//! zero.
//!
//! Three routes, each with its own ledger operation:
//!
//! - **pooling**: generalized least squares over native and transported
//!   summaries ([`SynthesisOperation::StatisticalPooling`]);
//! - **borrowing**: hierarchical normal borrowing across exchangeable native
//!   and transported studies ([`SynthesisOperation::BayesianBorrowing`] with no
//!   prior source);
//! - **transported prior**: a transported claim, as a source of kind prior, is
//!   the prior of a native likelihood
//!   ([`SynthesisOperation::BayesianBorrowing`] with that prior).
//!
//! The route is atomic: the combination is registered on a copy of the ledger
//! and committed only when the numbers were computed. The result is a point
//! and a model variance; no interval, posterior coverage or calibration is
//! claimed, and the dependence between source estimates and native estimates
//! is the caller's declaration.
//!
//! # Ownership
//!
//! The ledger is borrowed mutably for one call and nothing is retained. The
//! contributions are borrowed slices; the result owns its data.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;

use antecedent_contract::evidence_ledger::{
    ClaimClass, CombinationRequest, EvidenceLedger, LedgerError, MassAccount, Overlap,
    PoolingDependence, ResultId, SourceId, SynthesisOperation,
};
use antecedent_core::SemanticDigest;
use antecedent_core::identity::CanonicalWriter;
use antecedent_stats::synthesis::{MeanPrior, TauPrior};
use antecedent_stats::synthesis_transport::{
    SummaryOrigin, TransportAwareSummary, TransportedSynthesisError, borrow_transported,
    pool_transported, transport_aware_covariance, transported_prior_update,
};

/// Version of the transported-synthesis result semantics.
pub const TRANSPORTED_SYNTHESIS_RESULT_VERSION: u32 = 1;

const TAG_COVARIANCE: &str = "antecedent.design.transported_synthesis.covariance";

/// A claim from the evidence ledger, with the number it contributes.
#[derive(Clone, Debug, PartialEq)]
pub struct TransportedContribution {
    /// The ledger source the number comes from.
    pub source: SourceId,
    /// Point estimate of the common target quantity.
    pub estimate: f64,
    /// Sampling variance of the estimate.
    pub sampling_variance: f64,
    /// Variance of the transport error. Required exactly when the source
    /// carries a transport identity; refused on a native source.
    pub transport_variance: Option<f64>,
}

/// How the contributions are combined.
#[derive(Clone, Debug, PartialEq)]
pub enum TransportedSynthesisMethod {
    /// Generalized-least-squares pooling.
    Pool {
        /// `k x k` covariance of the sampling errors, in the order of the
        /// contributions sorted by source identity; independent when absent.
        sampling_covariance: Option<Vec<f64>>,
    },
    /// Hierarchical normal borrowing across exchangeable studies.
    Borrow {
        /// Prior on the common mean.
        mean_prior: MeanPrior,
        /// Prior on the between-study standard deviation.
        tau_prior: TauPrior,
    },
    /// One transported claim is the prior of the native contributions.
    TransportedPrior {
        /// The prior's ledger source (kind prior, carrying a transport
        /// identity), one of the contributions.
        prior: SourceId,
    },
}

impl TransportedSynthesisMethod {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Pool { .. } => "pool",
            Self::Borrow { .. } => "borrow",
            Self::TransportedPrior { .. } => "transported_prior",
        }
    }
}

/// A transported synthesis that was registered in the ledger and computed.
#[derive(Clone, Debug, PartialEq)]
pub struct TransportedSynthesis {
    /// The ledger result the combination was registered as.
    pub result: ResultId,
    /// The route taken.
    pub method: &'static str,
    /// Point value of the common quantity.
    pub estimate: f64,
    /// Model variance under the declared covariance (not a coverage claim).
    pub variance: f64,
    /// Variance the transports added over the same synthesis with every
    /// transport error zero, when the route computes it (pooling).
    pub transport_variance_added: Option<f64>,
    /// Native contributing sources, sorted.
    pub native_sources: Vec<SourceId>,
    /// Transported contributing sources, sorted.
    pub transported_sources: Vec<SourceId>,
    /// Transport identities that contributed, sorted and distinct.
    pub transport_identities: Vec<SemanticDigest>,
    /// Observation mass the ledger counted, each observation once.
    pub mass: MassAccount,
    /// What the ledger says the result licenses.
    pub claim_class: ClaimClass,
}

/// Why a transported synthesis was refused.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum TransportedSynthesisRefusal {
    /// The evidence ledger refused the combination (double counting, unknown
    /// overlap, a retired source, a malformed declaration).
    Ledger(LedgerError),
    /// The numerical routine refused its input.
    Numeric(TransportedSynthesisError),
    /// A contribution is inconsistent with the ledger's declaration of its
    /// source.
    Declaration {
        /// Namespaced `transported_synthesis.*` detail.
        detail: &'static str,
        /// The source at fault, when one is.
        source: Option<SourceId>,
        /// What would license the request.
        remedy: &'static str,
    },
}

impl TransportedSynthesisRefusal {
    /// Registered reason code.
    #[must_use]
    pub fn code(&self) -> &'static str {
        match self {
            Self::Ledger(error) => error.code,
            Self::Numeric(TransportedSynthesisError::SharedTransportNeedsCovariance { .. }) => {
                antecedent_core::reason_code!("joint_law_required")
            }
            Self::Numeric(_) => antecedent_core::reason_code!("invalid_argument"),
            Self::Declaration { detail, .. } => {
                if *detail == DETAIL_TRANSPORT_VARIANCE_UNDECLARED {
                    antecedent_core::reason_code!("transport_missing_evidence")
                } else {
                    antecedent_core::reason_code!("invalid_argument")
                }
            }
        }
    }

    /// Namespaced detail of the refusal.
    #[must_use]
    pub fn detail(&self) -> &'static str {
        match self {
            Self::Ledger(error) => error.detail,
            Self::Numeric(TransportedSynthesisError::SharedTransportNeedsCovariance { .. }) => {
                DETAIL_SHARED_TRANSPORT
            }
            Self::Numeric(_) => DETAIL_NUMERIC,
            Self::Declaration { detail, .. } => detail,
        }
    }
}

/// A transported claim was contributed without the variance of its transport.
pub const DETAIL_TRANSPORT_VARIANCE_UNDECLARED: &str =
    "transported_synthesis.transport_variance_undeclared";
/// A native source was contributed with a transport variance.
pub const DETAIL_TRANSPORT_VARIANCE_ON_NATIVE: &str =
    "transported_synthesis.transport_variance_on_native_source";
/// A source was contributed more than once.
pub const DETAIL_SOURCE_LISTED_TWICE: &str = "transported_synthesis.source_listed_twice";
/// A contribution names a source that is not in the ledger.
pub const DETAIL_UNKNOWN_SOURCE: &str = "transported_synthesis.unknown_source";
/// No contribution was supplied.
pub const DETAIL_NO_CONTRIBUTIONS: &str = "transported_synthesis.no_contributions";
/// The transported-prior route was not given a transported prior among its
/// contributions, or the likelihood was not native.
pub const DETAIL_PRIOR_ROLE: &str = "transported_synthesis.prior_role_mismatch";
/// One transport serves several claims on a route that cannot share its error.
pub const DETAIL_SHARED_TRANSPORT: &str = "transported_synthesis.shared_transport_needs_covariance";
/// Shared transport error was declared, but the sources' observations are
/// shared or unknown and no sampling covariance models that.
pub const DETAIL_OVERLAP_NOT_MODELLED: &str = "transported_synthesis.overlap_not_modelled";
/// The numerical routine refused its input.
pub const DETAIL_NUMERIC: &str = "transported_synthesis.numeric_refused";

fn declaration(
    detail: &'static str,
    source: Option<SourceId>,
    remedy: &'static str,
) -> TransportedSynthesisRefusal {
    TransportedSynthesisRefusal::Declaration { detail, source, remedy }
}

struct Classified {
    source: SourceId,
    summary: TransportAwareSummary,
    transport_digest: Option<SemanticDigest>,
}

fn classify(
    ledger: &EvidenceLedger,
    contributions: &[TransportedContribution],
) -> Result<Vec<Classified>, TransportedSynthesisRefusal> {
    if contributions.is_empty() {
        return Err(declaration(
            DETAIL_NO_CONTRIBUTIONS,
            None,
            "contribute at least one claim from the evidence ledger",
        ));
    }
    let mut seen: BTreeSet<SourceId> = BTreeSet::new();
    let mut out = Vec::with_capacity(contributions.len());
    for c in contributions {
        if !seen.insert(c.source) {
            return Err(declaration(
                DETAIL_SOURCE_LISTED_TWICE,
                Some(c.source),
                "contribute each source once; one source counted twice is double counting",
            ));
        }
        let Some(source) = ledger.source(c.source) else {
            return Err(declaration(
                DETAIL_UNKNOWN_SOURCE,
                Some(c.source),
                "add the source to the evidence ledger before contributing it",
            ));
        };
        let (origin, transport_digest) = match (&source.declaration.transport, c.transport_variance)
        {
            (Some(transport), Some(variance)) => {
                let digest = transport.digest();
                (
                    SummaryOrigin::Transported {
                        transport_key: digest.to_hex(),
                        transport_variance: variance,
                    },
                    Some(digest),
                )
            }
            (Some(_), None) => {
                return Err(declaration(
                    DETAIL_TRANSPORT_VARIANCE_UNDECLARED,
                    Some(c.source),
                    "declare the variance of the transport error; a transport with no declared uncertainty is not a zero-variance transport",
                ));
            }
            (None, Some(_)) => {
                return Err(declaration(
                    DETAIL_TRANSPORT_VARIANCE_ON_NATIVE,
                    Some(c.source),
                    "a native source has no transport error; drop the transport variance, or declare the source's transport identity in the ledger",
                ));
            }
            (None, None) => (SummaryOrigin::Native, None),
        };
        out.push(Classified {
            source: c.source,
            summary: TransportAwareSummary {
                estimate: c.estimate,
                sampling_variance: c.sampling_variance,
                origin,
            },
            transport_digest,
        });
    }
    out.sort_by_key(|c| c.source);
    Ok(out)
}

fn covariance_digest(covariance: &[f64]) -> SemanticDigest {
    let mut w = CanonicalWriter::new(TAG_COVARIANCE);
    w.word(covariance.len() as u64).reals(covariance);
    SemanticDigest::from_bytes(w.finish())
}

/// Register a transported synthesis in the evidence ledger and compute it.
///
/// Contributions are sorted by source identity, so their order is not part of
/// the result. The ledger is changed only when the whole synthesis succeeded.
///
/// # Errors
/// [`TransportedSynthesisRefusal::Declaration`] for a contribution that
/// contradicts its source's declaration; [`TransportedSynthesisRefusal::Ledger`]
/// for double counting (a source both native and transported, overlapping
/// prior and likelihood), unknown overlap or a retired source;
/// [`TransportedSynthesisRefusal::Numeric`] for a refused numerical input.
pub fn synthesize_transported(
    ledger: &mut EvidenceLedger,
    label: &str,
    contributions: &[TransportedContribution],
    method: &TransportedSynthesisMethod,
) -> Result<TransportedSynthesis, TransportedSynthesisRefusal> {
    let classified = classify(ledger, contributions)?;
    let ids: Vec<SourceId> = classified.iter().map(|c| c.source).collect();
    let summaries: Vec<TransportAwareSummary> =
        classified.iter().map(|c| c.summary.clone()).collect();

    // The ledger decides the overlap before any number is combined.
    let (operation, prior_index) = match method {
        TransportedSynthesisMethod::Pool { sampling_covariance } => {
            let covariance = transport_aware_covariance(&summaries, sampling_covariance.as_deref())
                .map_err(TransportedSynthesisRefusal::Numeric)?;
            let k = summaries.len();
            let has_cross_covariance = (0..k).any(|i| {
                (0..k).any(|j| i != j && covariance[i * k + j].to_bits() != 0_f64.to_bits())
            });
            if has_cross_covariance && sampling_covariance.is_none() {
                // The cross terms are the shared transport error only. They
                // say nothing about shared data, so the data must be disjoint.
                for (i, a) in ids.iter().enumerate() {
                    for b in ids.iter().skip(i + 1) {
                        let overlap =
                            ledger.overlap(*a, *b).map_err(TransportedSynthesisRefusal::Ledger)?;
                        if !matches!(overlap, Overlap::Independent) {
                            return Err(declaration(
                                DETAIL_OVERLAP_NOT_MODELLED,
                                Some(*a),
                                "supply a sampling covariance that models the shared or unknown observations, or contribute sources with disjoint, declared observations",
                            ));
                        }
                    }
                }
            }
            let dependence = if has_cross_covariance {
                PoolingDependence::DeclaredCovariance {
                    covariance: covariance_digest(&covariance),
                    has_cross_covariance,
                }
            } else {
                PoolingDependence::Independent
            };
            (SynthesisOperation::StatisticalPooling { sources: ids.clone(), dependence }, None)
        }
        TransportedSynthesisMethod::Borrow { .. } => (
            SynthesisOperation::BayesianBorrowing { priors: Vec::new(), likelihood: ids.clone() },
            None,
        ),
        TransportedSynthesisMethod::TransportedPrior { prior } => {
            let Some(index) = ids.iter().position(|id| id == prior) else {
                return Err(declaration(
                    DETAIL_PRIOR_ROLE,
                    Some(*prior),
                    "the prior is one of the contributions",
                ));
            };
            if classified[index].transport_digest.is_none() {
                return Err(declaration(
                    DETAIL_PRIOR_ROLE,
                    Some(*prior),
                    "the prior of this route is a transported claim; a native prior is a native source",
                ));
            }
            if let Some(other) = classified
                .iter()
                .enumerate()
                .find(|(i, c)| *i != index && c.transport_digest.is_some())
            {
                return Err(declaration(
                    DETAIL_PRIOR_ROLE,
                    Some(other.1.source),
                    "the likelihood of the prior route is native target evidence; pool or borrow to combine several transported claims",
                ));
            }
            let likelihood: Vec<SourceId> =
                ids.iter().enumerate().filter(|(i, _)| *i != index).map(|(_, id)| *id).collect();
            (
                SynthesisOperation::BayesianBorrowing { priors: vec![*prior], likelihood },
                Some(index),
            )
        }
    };
    let request = CombinationRequest { label: label.to_owned(), operation, replaces: None };
    let mut trial = ledger.clone();
    let result = trial.combine(request).map_err(TransportedSynthesisRefusal::Ledger)?;

    let (estimate, variance, transport_variance_added) = match method {
        TransportedSynthesisMethod::Pool { sampling_covariance } => {
            let pooled = pool_transported(&summaries, sampling_covariance.as_deref())
                .map_err(TransportedSynthesisRefusal::Numeric)?;
            (pooled.pooled.estimate, pooled.pooled.variance, Some(pooled.transport_variance_added))
        }
        TransportedSynthesisMethod::Borrow { mean_prior, tau_prior } => {
            let borrowed = borrow_transported(&summaries, *mean_prior, tau_prior)
                .map_err(TransportedSynthesisRefusal::Numeric)?;
            (borrowed.posterior.mean, borrowed.posterior.variance, None)
        }
        TransportedSynthesisMethod::TransportedPrior { .. } => {
            let index = prior_index.unwrap_or(0);
            let likelihood: Vec<TransportAwareSummary> = summaries
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != index)
                .map(|(_, s)| s.clone())
                .collect();
            let update = transported_prior_update(&summaries[index], &likelihood)
                .map_err(TransportedSynthesisRefusal::Numeric)?;
            (update.mean, update.variance, None)
        }
    };

    let record = trial.result(result).cloned().ok_or_else(|| {
        declaration(
            DETAIL_UNKNOWN_SOURCE,
            None,
            "the ledger did not record the combination it accepted",
        )
    })?;
    let mut native_sources = Vec::new();
    let mut transported_sources = Vec::new();
    let mut identities: BTreeSet<SemanticDigest> = BTreeSet::new();
    for c in &classified {
        match c.transport_digest {
            Some(digest) => {
                transported_sources.push(c.source);
                identities.insert(digest);
            }
            None => native_sources.push(c.source),
        }
    }
    *ledger = trial;
    Ok(TransportedSynthesis {
        result,
        method: method.as_str(),
        estimate,
        variance,
        transport_variance_added,
        native_sources,
        transported_sources,
        transport_identities: identities.into_iter().collect(),
        mass: record.mass,
        claim_class: record.claim_class,
    })
}
