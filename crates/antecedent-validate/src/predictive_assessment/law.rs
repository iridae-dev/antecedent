//! Predictive laws, the quantity they are about, and realized observations.
//!
//! A [`PredictorLaws`] is one named predictor: for every case it supplies one
//! [`ForecastLaw`], declares the [`PredictiveTarget`] the laws are about and
//! where they came from ([`LawSource`]), and declares what it was fitted on
//! ([`FitDeclaration`](super::FitDeclaration)).
//!
//! The law sources are native finite and closed-form families and laws supplied
//! by an external provider (declared through the provider contract's object
//! identity). A law that exists only as a callback is refused by name.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::overlap::FitDeclaration;
use super::refusal::{PredictiveAssessmentError, detail, invalid, refuse, unsupported};
use antecedent_core::reason_code;

/// Sanity bound on cases; the working limit is [`super::MAX_ASSESSMENT_WORK`].
pub const MAX_CASES: usize = 1 << 24;
/// Sanity bound on predictors; the working limit is [`super::MAX_ASSESSMENT_WORK`].
pub const MAX_PREDICTORS: usize = 1024;
/// Sanity bound on atoms, draws or categories of one case's law; the working limit is
/// [`super::MAX_ASSESSMENT_WORK`].
pub const MAX_LAW_ATOMS: usize = 1 << 20;
/// Sanity bound on the dimension of a joint law; the working limit is
/// [`super::MAX_ASSESSMENT_WORK`].
pub const MAX_JOINT_DIMENSION: usize = 1024;
/// Tolerance on a probability vector summing to one.
pub const PROBABILITY_SUM_TOLERANCE: f64 = 1e-9;

/// How far ahead of the information time the target lies.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Horizon {
    /// Not a forecast: held-out units observed at the information time.
    Contemporaneous,
    /// A forecast this many steps ahead of the forecast origin.
    Steps(u32),
}

impl Horizon {
    /// Stable spelling for messages.
    #[must_use]
    pub fn describe(self) -> String {
        match self {
            Self::Contemporaneous => "contemporaneous".to_owned(),
            Self::Steps(steps) => format!("{steps} steps ahead"),
        }
    }
}

/// What a predictive law is a law of: the quantity, horizon, population and units.
///
/// The observations carry the same declaration; a law whose target differs from
/// the observations' is refused rather than scored.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct PredictiveTarget {
    /// The predicted quantity (a stable variable or functional name).
    pub quantity: String,
    /// The horizon of the prediction.
    pub horizon: Horizon,
    /// The population the quantity is defined for.
    pub population: String,
    /// The units or scale of the quantity.
    pub units: String,
}

impl PredictiveTarget {
    /// A target declaration.
    #[must_use]
    pub fn new(
        quantity: impl Into<String>,
        horizon: Horizon,
        population: impl Into<String>,
        units: impl Into<String>,
    ) -> Self {
        Self {
            quantity: quantity.into(),
            horizon,
            population: population.into(),
            units: units.into(),
        }
    }

    pub(crate) fn validate(&self, whose: &str) -> Result<(), PredictiveAssessmentError> {
        for (field, value) in
            [("quantity", &self.quantity), ("population", &self.population), ("units", &self.units)]
        {
            if value.trim().is_empty() {
                return Err(refuse(
                    reason_code!("quantity_semantics_mismatch"),
                    detail::TARGET_MISMATCH,
                    format!(
                        "{whose} declares a blank target {field}; a predictive target names its \
                         quantity, horizon, population and units"
                    ),
                ));
            }
        }
        Ok(())
    }

    /// Refuse unless `observed` is the same quantity, population, units and horizon.
    pub(crate) fn check_matches(
        &self,
        observed: &Self,
        predictor: &str,
    ) -> Result<(), PredictiveAssessmentError> {
        let remedy = "supply observations of the quantity the law predicts, or restate the law for \
                      the observed target; no conversion is inferred";
        for (field, law_value, observed_value) in [
            ("quantity", &self.quantity, &observed.quantity),
            ("population", &self.population, &observed.population),
            ("units", &self.units, &observed.units),
        ] {
            if law_value != observed_value {
                return Err(refuse(
                    reason_code!("quantity_semantics_mismatch"),
                    detail::TARGET_MISMATCH,
                    format!(
                        "predictor `{predictor}` predicts {field} `{law_value}` but the observations \
                         record {field} `{observed_value}`; {remedy}"
                    ),
                ));
            }
        }
        if self.horizon != observed.horizon {
            return Err(refuse(
                reason_code!("quantity_semantics_mismatch"),
                detail::HORIZON_MISMATCH,
                format!(
                    "predictor `{predictor}` predicts {} but the observations are {}; {remedy}",
                    self.horizon.describe(),
                    observed.horizon.describe(),
                ),
            ));
        }
        Ok(())
    }
}

/// The identity of one observation: a 32-byte digest.
///
/// The same key in two places is the same observation. The facade maps an
/// evidence-ledger observation identity onto this key without change
/// ([`ObservationKey::from_bytes`]). A key minted here from a caller-chosen
/// name ([`ObservationKey::from_key`]) lives in a namespace disjoint from the
/// ledger: it compares equal only to keys minted the same way, so an overlap
/// test against ledger-derived keys cannot see shared rows. Such keys are
/// remembered as `caller_minted` and an assessment never reports a predictor
/// that touches one as verified held out.
///
/// Equality, ordering and hashing are by digest only.
#[derive(Clone, Copy, Debug)]
pub struct ObservationKey {
    digest: [u8; 32],
    caller_minted: bool,
}

impl PartialEq for ObservationKey {
    fn eq(&self, other: &Self) -> bool {
        self.digest == other.digest
    }
}
impl Eq for ObservationKey {}
impl std::hash::Hash for ObservationKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.digest.hash(state);
    }
}
impl PartialOrd for ObservationKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for ObservationKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.digest.cmp(&other.digest)
    }
}

impl ObservationKey {
    /// Wrap a 32-byte digest, the identity of an evidence-ledger observation.
    #[must_use]
    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self { digest: bytes, caller_minted: false }
    }

    /// Restore a digest whose origin is caller-minted or provenance-unknown.
    ///
    /// This is conservative: it cannot grant ledger-backed held-out status.
    #[must_use]
    pub const fn from_caller_minted_bytes(bytes: [u8; 32]) -> Self {
        Self { digest: bytes, caller_minted: true }
    }

    /// The digest bytes.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.digest
    }

    /// The key of an observation named by a stable key within a domain, for
    /// example a data snapshot digest and a row index.
    ///
    /// The key is in a namespace disjoint from evidence-ledger observation
    /// identities and is marked caller-minted: it can show that two
    /// caller-minted keys coincide (in-sample), but a predictor whose fit set
    /// or whose assessed set holds such a key is labelled `unverified`, never
    /// `held_out_verified`. Map ledger observation ids through
    /// [`ObservationKey::from_bytes`] to verify held-out status.
    #[must_use]
    pub fn from_key(domain: &str, key: &[u8]) -> Self {
        let mut writer =
            antecedent_core::CanonicalWriter::new("antecedent.predictive_assessment.observation");
        writer.text(domain).bytes(key);
        Self { digest: writer.finish(), caller_minted: true }
    }

    /// Whether the key was minted from a caller-chosen name rather than mapped
    /// from an evidence-ledger observation identity.
    #[must_use]
    pub const fn is_caller_minted(&self) -> bool {
        self.caller_minted
    }

    /// Lowercase hex spelling (64 characters).
    #[must_use]
    pub fn to_hex(&self) -> String {
        antecedent_core::to_hex(&self.digest)
    }
}

/// One realized outcome.
#[derive(Clone, Debug, PartialEq)]
pub enum Outcome {
    /// A real-valued outcome.
    Real(f64),
    /// A category index (`0` or `1` for a Bernoulli law).
    Category(u32),
    /// A vector outcome of a joint law.
    Vector(Vec<f64>),
}

impl Outcome {
    pub(crate) const fn family(&self) -> OutcomeFamily {
        match self {
            Self::Real(_) => OutcomeFamily::Real,
            Self::Category(_) => OutcomeFamily::Category,
            Self::Vector(_) => OutcomeFamily::Vector,
        }
    }
}

/// The kind of outcome a law predicts.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum OutcomeFamily {
    Real,
    Category,
    Vector,
}

impl OutcomeFamily {
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Real => "real",
            Self::Category => "category",
            Self::Vector => "vector",
        }
    }
}

/// The realized observations an assessment scores against.
#[derive(Clone, Debug, PartialEq)]
pub struct RealizedObservations {
    /// The quantity, horizon, population and units the observations record.
    pub target: PredictiveTarget,
    /// One identity per case; distinct.
    pub ids: Vec<ObservationKey>,
    /// One outcome per case, in the order of `ids`.
    pub outcomes: Vec<Outcome>,
}

impl RealizedObservations {
    pub(crate) fn validate(&self) -> Result<(), PredictiveAssessmentError> {
        self.target.validate("the observations")?;
        let n = self.ids.len();
        if n == 0 || n > MAX_CASES {
            return Err(super::refusal::bounds(format!(
                "{n} cases supplied; 1 to {MAX_CASES} are admitted"
            )));
        }
        if self.outcomes.len() != n {
            return Err(invalid(
                detail::INVALID_OBSERVATIONS,
                format!("{} outcomes for {n} observation identities", self.outcomes.len()),
            ));
        }
        let mut sorted = self.ids.clone();
        sorted.sort_unstable();
        if sorted.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(invalid(
                detail::INVALID_OBSERVATIONS,
                "observation identities repeat; each realized observation is scored once, so \
                 repeated use must be declared as distinct identities",
            ));
        }
        let family = self.outcomes[0].family();
        for outcome in &self.outcomes {
            if outcome.family() != family {
                return Err(invalid(
                    detail::OUTCOME_KIND,
                    "the realized outcomes mix real, category and vector outcomes",
                ));
            }
            match outcome {
                Outcome::Real(value) if !value.is_finite() => {
                    return Err(invalid(
                        detail::INVALID_OBSERVATIONS,
                        "a realized outcome is not finite",
                    ));
                }
                Outcome::Vector(values)
                    if values.is_empty() || values.iter().any(|value| !value.is_finite()) =>
                {
                    return Err(invalid(
                        detail::INVALID_OBSERVATIONS,
                        "a realized vector outcome is empty or not finite",
                    ));
                }
                _ => {}
            }
        }
        Ok(())
    }
}

/// One predictive law for one case.
#[derive(Clone, Debug, PartialEq)]
pub enum ForecastLaw {
    /// A normal law `N(mean, sd^2)` of a real outcome.
    Gaussian {
        /// Mean.
        mean: f64,
        /// Standard deviation, positive.
        sd: f64,
    },
    /// A Bernoulli law of a `0/1` category outcome.
    Bernoulli {
        /// `P(Y = 1)`.
        probability: f64,
    },
    /// A categorical law over categories `0..K` (unordered).
    Categorical {
        /// Category probabilities summing to one.
        probabilities: Vec<f64>,
    },
    /// An exact finite law of a real outcome.
    FiniteReal {
        /// Support points, strictly increasing.
        support: Vec<f64>,
        /// Probabilities summing to one.
        probabilities: Vec<f64>,
    },
    /// An equally weighted sample (ensemble) of a real outcome. The sample is
    /// scored as the law it is: the empirical distribution of its draws.
    EmpiricalSample {
        /// The draws.
        draws: Vec<f64>,
    },
    /// A provider-evaluated law: only its log density and cumulative
    /// probabilities at the realized outcome are available.
    PointEvaluated {
        /// The log density (or log mass) at the outcome, when the provider supplies it.
        log_density: Option<f64>,
        /// `F(y-)`, the probability strictly below the outcome.
        cdf_below: f64,
        /// `F(y)`, the probability at or below the outcome.
        cdf_at: f64,
    },
    /// An exact finite joint law of a vector outcome.
    JointFinite {
        /// Dimension of the outcome.
        dimension: usize,
        /// Atoms, row-major (`atoms * dimension` reals).
        atoms: Vec<f64>,
        /// Atom probabilities summing to one.
        probabilities: Vec<f64>,
    },
    /// An equally weighted joint sample of a vector outcome.
    JointSample {
        /// Dimension of the outcome.
        dimension: usize,
        /// Draws, row-major (`draws * dimension` reals).
        draws: Vec<f64>,
    },
}

/// The family of a [`ForecastLaw`].
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum LawKind {
    /// [`ForecastLaw::Gaussian`].
    Gaussian,
    /// [`ForecastLaw::Bernoulli`].
    Bernoulli,
    /// [`ForecastLaw::Categorical`].
    Categorical,
    /// [`ForecastLaw::FiniteReal`].
    FiniteReal,
    /// [`ForecastLaw::EmpiricalSample`].
    EmpiricalSample,
    /// [`ForecastLaw::PointEvaluated`].
    PointEvaluated,
    /// [`ForecastLaw::JointFinite`].
    JointFinite,
    /// [`ForecastLaw::JointSample`].
    JointSample,
}

impl LawKind {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Gaussian => "gaussian",
            Self::Bernoulli => "bernoulli",
            Self::Categorical => "categorical",
            Self::FiniteReal => "finite_real",
            Self::EmpiricalSample => "empirical_sample",
            Self::PointEvaluated => "point_evaluated",
            Self::JointFinite => "joint_finite",
            Self::JointSample => "joint_sample",
        }
    }
}

fn check_probabilities(
    probabilities: &[f64],
    what: &str,
    minimum: usize,
) -> Result<(), PredictiveAssessmentError> {
    if probabilities.len() < minimum || probabilities.len() > MAX_LAW_ATOMS {
        return Err(invalid(
            detail::INVALID_LAW,
            format!(
                "{what} has {} probabilities; {minimum} to {MAX_LAW_ATOMS} are admitted",
                probabilities.len()
            ),
        ));
    }
    if probabilities.iter().any(|p| !p.is_finite() || *p < 0.0) {
        return Err(invalid(
            detail::INVALID_LAW,
            format!("{what} has a probability that is negative or not finite"),
        ));
    }
    let total: f64 = probabilities.iter().sum();
    if (total - 1.0).abs() > PROBABILITY_SUM_TOLERANCE {
        return Err(invalid(
            detail::INVALID_LAW,
            format!("{what} probabilities sum to {total}, not 1"),
        ));
    }
    Ok(())
}

impl ForecastLaw {
    /// A Gaussian law.
    #[must_use]
    pub const fn gaussian(mean: f64, sd: f64) -> Self {
        Self::Gaussian { mean, sd }
    }

    /// A Bernoulli law with `P(Y = 1) = probability`.
    #[must_use]
    pub const fn bernoulli(probability: f64) -> Self {
        Self::Bernoulli { probability }
    }

    /// The family of this law.
    #[must_use]
    pub const fn kind(&self) -> LawKind {
        match self {
            Self::Gaussian { .. } => LawKind::Gaussian,
            Self::Bernoulli { .. } => LawKind::Bernoulli,
            Self::Categorical { .. } => LawKind::Categorical,
            Self::FiniteReal { .. } => LawKind::FiniteReal,
            Self::EmpiricalSample { .. } => LawKind::EmpiricalSample,
            Self::PointEvaluated { .. } => LawKind::PointEvaluated,
            Self::JointFinite { .. } => LawKind::JointFinite,
            Self::JointSample { .. } => LawKind::JointSample,
        }
    }

    pub(crate) const fn outcome_family(&self) -> OutcomeFamily {
        match self {
            Self::Gaussian { .. }
            | Self::FiniteReal { .. }
            | Self::EmpiricalSample { .. }
            | Self::PointEvaluated { .. } => OutcomeFamily::Real,
            Self::Bernoulli { .. } | Self::Categorical { .. } => OutcomeFamily::Category,
            Self::JointFinite { .. } | Self::JointSample { .. } => OutcomeFamily::Vector,
        }
    }

    /// Check the law is a well formed law.
    ///
    /// # Errors
    /// `invalid_argument` with detail `predictive_assessment.invalid_law`.
    pub fn validate(&self) -> Result<(), PredictiveAssessmentError> {
        match self {
            Self::Gaussian { mean, sd } => {
                if !mean.is_finite() || !sd.is_finite() || *sd <= 0.0 {
                    return Err(invalid(
                        detail::INVALID_LAW,
                        format!(
                            "a Gaussian law needs a finite mean and a positive sd; got ({mean}, {sd})"
                        ),
                    ));
                }
            }
            Self::Bernoulli { probability } => {
                if !probability.is_finite() || !(0.0..=1.0).contains(probability) {
                    return Err(invalid(
                        detail::INVALID_LAW,
                        format!("a Bernoulli probability {probability} is outside [0, 1]"),
                    ));
                }
            }
            Self::Categorical { probabilities } => {
                check_probabilities(probabilities, "a categorical law", 2)?;
            }
            Self::FiniteReal { support, probabilities } => {
                if support.len() != probabilities.len() {
                    return Err(invalid(
                        detail::INVALID_LAW,
                        "a finite law needs one probability per support point",
                    ));
                }
                check_probabilities(probabilities, "a finite law", 1)?;
                if support.iter().any(|x| !x.is_finite())
                    || support.windows(2).any(|w| w[0] >= w[1])
                {
                    return Err(invalid(
                        detail::INVALID_LAW,
                        "a finite law's support must be finite and strictly increasing",
                    ));
                }
            }
            Self::EmpiricalSample { draws } => {
                if draws.is_empty() || draws.len() > MAX_LAW_ATOMS {
                    return Err(invalid(
                        detail::INVALID_LAW,
                        format!(
                            "a sample law has {} draws; 1 to {MAX_LAW_ATOMS} are admitted",
                            draws.len()
                        ),
                    ));
                }
                if draws.iter().any(|x| !x.is_finite()) {
                    return Err(invalid(detail::INVALID_LAW, "a sample draw is not finite"));
                }
            }
            Self::PointEvaluated { log_density, cdf_below, cdf_at } => {
                let ordered = cdf_below.is_finite()
                    && cdf_at.is_finite()
                    && *cdf_below >= 0.0
                    && cdf_below <= cdf_at
                    && *cdf_at <= 1.0;
                if !ordered || log_density.is_some_and(|value| !value.is_finite()) {
                    return Err(invalid(
                        detail::INVALID_LAW,
                        "a provider-evaluated law needs 0 <= F(y-) <= F(y) <= 1 and a finite \
                         log density when one is supplied",
                    ));
                }
            }
            Self::JointFinite { dimension, atoms, probabilities } => {
                check_dimension(*dimension)?;
                check_probabilities(probabilities, "a joint finite law", 1)?;
                if atoms.len() != probabilities.len() * dimension
                    || atoms.iter().any(|x| !x.is_finite())
                {
                    return Err(invalid(
                        detail::INVALID_LAW,
                        "a joint finite law needs atoms * dimension finite reals",
                    ));
                }
            }
            Self::JointSample { dimension, draws } => {
                check_dimension(*dimension)?;
                if draws.is_empty()
                    || draws.len() % dimension != 0
                    || draws.len() / dimension > MAX_LAW_ATOMS
                    || draws.iter().any(|x| !x.is_finite())
                {
                    return Err(invalid(
                        detail::INVALID_LAW,
                        "a joint sample needs between one and the maximum number of finite \
                         draws of the declared dimension",
                    ));
                }
            }
        }
        Ok(())
    }

    /// Work units scoring this law costs (a bound used for the work budget).
    pub(crate) fn work(&self) -> u64 {
        let size = |n: usize| u64::try_from(n).unwrap_or(u64::MAX);
        match self {
            Self::Gaussian { .. } | Self::Bernoulli { .. } | Self::PointEvaluated { .. } => 8,
            Self::Categorical { probabilities } => size(probabilities.len()) + 8,
            Self::FiniteReal { support, .. } => 4 * size(support.len()) + 8,
            Self::EmpiricalSample { draws } => {
                let m = size(draws.len());
                m.saturating_mul(u64::from(64 - m.leading_zeros()) + 2) + 8
            }
            Self::JointFinite { dimension, probabilities, .. } => {
                let n = size(probabilities.len());
                n.saturating_mul(n).saturating_mul(size(*dimension)) + 8
            }
            Self::JointSample { dimension, draws } => {
                let m = size(draws.len() / dimension.max(&1));
                m.saturating_mul(m).saturating_mul(size(*dimension)) + 8
            }
        }
    }
}

fn check_dimension(dimension: usize) -> Result<(), PredictiveAssessmentError> {
    if dimension == 0 || dimension > MAX_JOINT_DIMENSION {
        return Err(invalid(
            detail::INVALID_LAW,
            format!(
                "a joint law has dimension {dimension}; 1 to {MAX_JOINT_DIMENSION} are admitted"
            ),
        ));
    }
    Ok(())
}

/// The five identity fields of an external provider's object, as the provider
/// contract declares them.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct ProviderIdentity {
    /// Provider service or model identity.
    pub provider_id: String,
    /// Object identity within the provider.
    pub object_id: String,
    /// Provider implementation or model version.
    pub version_id: String,
    /// Input data or model snapshot.
    pub snapshot_id: String,
    /// Exact request fingerprint.
    pub request_id: String,
}

impl ProviderIdentity {
    fn is_complete(&self) -> bool {
        [&self.provider_id, &self.object_id, &self.version_id, &self.snapshot_id, &self.request_id]
            .into_iter()
            .all(|field| !field.trim().is_empty())
    }
}

/// How far a provider's law is trusted.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ProviderTrustLabel {
    /// No provider verification claim.
    Unverified,
    /// Supplier assertion only.
    ExternallyAttested,
    /// The exact provider contract passed object-level verification.
    ExactRequestVerified,
}

impl ProviderTrustLabel {
    /// Stable `snake_case` name, equal to the provider-trust wire spelling.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Unverified => "unverified",
            Self::ExternallyAttested => "externally_attested",
            Self::ExactRequestVerified => "verified_extension",
        }
    }
}

/// Where a predictive law came from.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum LawSource {
    /// A native model of this library, named by its label.
    Native {
        /// Model label.
        model: String,
    },
    /// A law supplied by an external provider and declared through the
    /// provider contract. Its scores are as trustworthy as the supplier.
    Provider {
        /// The provider object identity.
        identity: ProviderIdentity,
        /// The retained trust level.
        trust: ProviderTrustLabel,
    },
    /// A law that exists only as a callback. Not built: always refused.
    Callback {
        /// Description of the callback, for the refusal message.
        description: String,
    },
}

impl LawSource {
    pub(crate) fn validate(&self, predictor: &str) -> Result<(), PredictiveAssessmentError> {
        match self {
            Self::Native { model } if model.trim().is_empty() => Err(invalid(
                detail::INVALID_LAW,
                format!("predictor `{predictor}` declares a blank native model label"),
            )),
            Self::Provider { identity, .. } if !identity.is_complete() => Err(invalid(
                detail::PROVIDER_IDENTITY_INCOMPLETE,
                format!(
                    "predictor `{predictor}` is supplied by a provider whose object identity is \
                     incomplete; declare provider, object, version, snapshot and request \
                     identities through the provider contract"
                ),
            )),
            Self::Native { .. } | Self::Provider { .. } => Ok(()),
            Self::Callback { description } => Err(unsupported(
                detail::LAW_CALLBACK_NOT_BUILT,
                format!(
                    "predictor `{predictor}` is a law callback ({description}); callback laws \
                     are not built. Supply the law as a Gaussian, Bernoulli, categorical, finite \
                     or sample law, or as provider-evaluated density and cumulative values \
                     declared through the provider contract"
                ),
            )),
        }
    }

    /// Stable `snake_case` name of the source class.
    #[must_use]
    pub const fn class(&self) -> &'static str {
        match self {
            Self::Native { .. } => "native",
            Self::Provider { .. } => "provider",
            Self::Callback { .. } => "callback",
        }
    }
}

/// One named predictor: its law for every case, where it came from and what it
/// was fitted on.
#[derive(Clone, Debug, PartialEq)]
pub struct PredictorLaws {
    /// Unique name.
    pub name: String,
    /// Where the laws came from.
    pub source: LawSource,
    /// What the laws are about.
    pub target: PredictiveTarget,
    /// One law per case, in the order of the observations.
    pub laws: Vec<ForecastLaw>,
    /// What the laws were fitted or tuned on.
    pub fit: FitDeclaration,
}

impl PredictorLaws {
    pub(crate) fn validate(&self, cases: usize) -> Result<(), PredictiveAssessmentError> {
        if self.name.trim().is_empty() {
            return Err(invalid(detail::INVALID_LAW, "a predictor needs a non-blank name"));
        }
        self.source.validate(&self.name)?;
        self.target.validate(&format!("predictor `{}`", self.name))?;
        if self.laws.len() != cases {
            return Err(invalid(
                detail::INVALID_LAW,
                format!(
                    "predictor `{}` supplies {} laws for {cases} observations; one law per case \
                     is required",
                    self.name,
                    self.laws.len()
                ),
            ));
        }
        for law in &self.laws {
            law.validate()?;
        }
        Ok(())
    }
}
