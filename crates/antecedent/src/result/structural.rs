use super::{
    CausalPosterior, CausalResponse, IdentificationStatus, ResponseEnvelope, ResponseValue,
};

/// Meaning of structural atom weights retained on a response result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum StructuralWeightBasis {
    /// Probabilities supplied by a graph posterior.
    PosteriorProbability,
    /// Enumeration weights over CPDAG/PAG completions; not posterior probabilities.
    CompletionEnumeration,
    /// Caller-declared mass over class members. Not a graph posterior and not
    /// completion enumeration.
    CallerSuppliedClassPrior,
}

impl StructuralWeightBasis {
    /// Snake-case name used by the bindings and the result artifact wire format.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PosteriorProbability => "posterior_probability",
            Self::CompletionEnumeration => "completion_enumeration",
            Self::CallerSuppliedClassPrior => "caller_supplied_class_prior",
        }
    }
}

impl From<StructuralWeightBasis> for antecedent_io::StructuralWeightBasisWire {
    fn from(basis: StructuralWeightBasis) -> Self {
        match basis {
            StructuralWeightBasis::PosteriorProbability => Self::PosteriorProbability,
            StructuralWeightBasis::CompletionEnumeration => Self::CompletionEnumeration,
            StructuralWeightBasis::CallerSuppliedClassPrior => Self::CallerSuppliedClassPrior,
        }
    }
}

/// How identified structural atoms may be combined.
///
/// Mass-weighting is supported only when every contributing atom names the same
/// estimand. Completion enumeration is a [`StructuralWeightBasis`], not a
/// reason to treat class members as posterior probability.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum StructuralAggregationPolicy {
    /// Identified atoms share one estimand identity; a probability-weighted
    /// mean of their values is the same functional.
    SameEstimandWeightedMean,
    /// Identified atoms share one partially identified estimand; publish the
    /// identified set, not a fake scalar mixture.
    IdentifiedSetEnvelope,
    /// Identified atoms name different estimands. No scalar mixture.
    GraphDependentAtoms,
}

impl StructuralAggregationPolicy {
    /// Snake-case name used in diagnostics.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SameEstimandWeightedMean => "same_estimand_weighted_mean",
            Self::IdentifiedSetEnvelope => "identified_set_envelope",
            Self::GraphDependentAtoms => "graph_dependent_atoms",
        }
    }
}

/// One structural atom in a graph-dependent response.
#[derive(Clone, Debug)]
pub struct StructuralResponseAtom {
    /// Stable key within this result.
    pub graph_key: u64,
    /// Raw atom weight in the declared basis.
    pub weight: f64,
    /// Structural identification/evaluation status.
    pub status: IdentificationStatus,
    /// Numerical response when identified and evaluable.
    pub value: Option<ResponseValue>,
    /// Completion-conditional posterior, without assigning structural probabilities.
    pub posterior: Option<CausalPosterior>,
    /// Full function-valued atom, including its conditional sampling uncertainty.
    pub response: Option<CausalResponse>,
}

/// Structural uncertainty retained separately from sampling uncertainty.
#[derive(Clone, Debug)]
pub struct StructuralResponseMixture {
    /// Interpretation of [`StructuralResponseAtom::weight`].
    pub weight_basis: StructuralWeightBasis,
    /// Every examined structural atom, including nonidentified/unevaluable atoms.
    pub atoms: Vec<StructuralResponseAtom>,
    /// Fraction of total weight with an evaluable identified response.
    pub identified_mass: f64,
    /// Fraction proved or conservatively treated as structurally unidentified.
    pub unidentified_mass: f64,
    /// Fraction identified in theory but not evaluable by the selected estimator.
    pub unevaluable_mass: f64,
    /// Fraction on identified atoms that were never evaluated because the
    /// Interactive latency tier's graph budget left them out of the stratified
    /// subsample. Neither unidentified nor a failed estimate; the four masses
    /// sum to one. Zero outside the Interactive tier.
    pub subsampled_out_mass: f64,
    /// Pointwise range over identified atom point responses.
    pub identified_set: Option<ResponseEnvelope>,
    /// Interval for a scalar [`Self::identified_set`] that adds sampling
    /// uncertainty to the point bounds. Published on class-aware
    /// temporal Pulse / Sustained effects. Frequentist: an Imbens–Manski interval
    /// with per-completion endpoints from the shared circular-block replicates,
    /// covering the true effect with asymptotic probability at least the stated
    /// level whenever it is one retained identified completion's effect.
    /// Bayesian: product-posterior envelope quantiles from independently seeded
    /// per-completion posterior draws (every retained completion's posterior puts
    /// at most `1 − Φ(c)` of its mass outside each endpoint). When the completion
    /// enumeration was capped the interval is still published, flagged
    /// `truncated`, with a warning diagnostic.
    pub identified_set_interval: Option<antecedent_estimate::IdentifiedSetInterval>,
    /// Probability-weighted summary when a scalar `SameEstimandWeightedMean` mix is
    /// published (`E[τ | identified]`). Retained unidentified mass does not
    /// suppress this field; consumers should read `identification.status` and
    /// `unidentified_mass` alongside it.
    pub conditional_on_identified: Option<ResponseValue>,
    /// Whether reported mass covers the full class rather than a capped subset.
    pub full_mass_scope: bool,
    /// Atoms whose identification search was capped before a determination.
    pub truncated_atoms: usize,
    /// How the identified atoms were combined, for routes that aggregate across
    /// graphs. `None` for routes whose atoms are not graph alternatives to be
    /// combined under a policy.
    pub aggregation_policy: Option<StructuralAggregationPolicy>,
}
