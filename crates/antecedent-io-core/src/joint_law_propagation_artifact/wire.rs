//! Versioned joint-law propagation wire forms and legacy result decoding.

use antecedent_stats::{DecisionVerdict, Marginal, ValueBounds};
use serde::{Deserialize, Serialize};

use super::{Error, floats, math_error, verdict_code};

// ----------------------------------------------------------------- the wire

/// One input of the composition.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PropagationInputWire {
    /// Stable input identity.
    pub input_id: String,
    /// Identity of the evidence the input rests on.
    pub evidence_id: String,
    /// The input's role (`estimate`, `transported_interval`, ...).
    pub role: String,
    /// `aligned`, `independent_marginals` or `moments_only`.
    pub alignment: String,
    /// The alignment identity, present exactly when `alignment` is `aligned`.
    pub alignment_id: Option<String>,
}

/// The propagation rule's declaration.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PropagationRuleWire {
    /// The operation's stable name.
    pub operation: String,
    /// `unknown`, `declared_independent`, `covariance`, `aligned_joint_law`,
    /// `known_weights`, `correlation_range` or `marginals_only`.
    pub dependence: String,
    /// The covariance declaration identity, for `covariance`.
    pub covariance_id: Option<String>,
    /// Lower correlation bits, for `correlation_range`.
    pub rho_lo: Option<u64>,
    /// Upper correlation bits, for `correlation_range`.
    pub rho_hi: Option<u64>,
    /// The reading (`point`, `spread` or `envelope`).
    pub reading: String,
}

/// A finite marginal, as bit patterns.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MarginalWire {
    /// Atom values.
    pub values: Vec<u64>,
    /// Atom masses.
    pub masses: Vec<u64>,
}

impl MarginalWire {
    /// A marginal of equal-mass draws.
    #[must_use]
    pub fn from_draws(draws: &[f64]) -> Self {
        let mass = 1.0 / draws.len().max(1) as f64;
        Self {
            values: draws.iter().map(|v| v.to_bits()).collect(),
            masses: vec![mass.to_bits(); draws.len()],
        }
    }

    pub(super) fn marginal(&self) -> Result<Marginal, Error> {
        Marginal::from_weighted(&floats(&self.values), &floats(&self.masses)).map_err(math_error)
    }
}

/// What an envelope over every coupling is asked for.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "snake_case")]
pub enum EnvelopeQueryWire {
    /// `P(X + Y <= t)`.
    SumCdf {
        /// Threshold bits.
        threshold: u64,
    },
    /// `P(X - Y <= t)`.
    DifferenceCdf {
        /// Threshold bits.
        threshold: u64,
    },
    /// `E[X Y]`.
    ProductMoment,
    /// `Cov(X, Y)`.
    Covariance,
    /// `Var(a X + b Y)`.
    LinearVariance {
        /// Coefficient of `X`, bits.
        a: u64,
        /// Coefficient of `Y`, bits.
        b: u64,
    },
    /// `E[g(X, Y)]` for a closed value functional.
    Functional {
        /// The functional's stable name.
        functional: String,
    },
}

/// The numbers a computation consumed.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "snake_case")]
pub enum PropagationRequestWire {
    /// `sum a_i X_i` from means and a covariance (delta method).
    LinearCombination {
        /// Coefficient bits.
        coefficients: Vec<u64>,
        /// Mean bits.
        means: Vec<u64>,
        /// Row-major covariance bits.
        covariance: Vec<u64>,
    },
    /// `a X + b Y` from aligned draws.
    AlignedLinearCombination {
        /// Coefficient of `X`, bits.
        a: u64,
        /// Coefficient of `Y`, bits.
        b: u64,
        /// Draws of `X`, bits.
        x: Vec<u64>,
        /// Draws of `Y`, bits, row-aligned with `x`.
        y: Vec<u64>,
    },
    /// `E[g(X, Y)]` from aligned draws.
    AlignedFunctional {
        /// The functional's stable name.
        functional: String,
        /// Draws of `X`, bits.
        x: Vec<u64>,
        /// Draws of `Y`, bits, row-aligned with `x`.
        y: Vec<u64>,
    },
    /// The spread of the rows `g(x_i, y_i)` of aligned draws (spread reading).
    AlignedFunctionalSpread {
        /// The functional's stable name.
        functional: String,
        /// Draws of `X`, bits.
        x: Vec<u64>,
        /// Draws of `Y`, bits, row-aligned with `x`.
        y: Vec<u64>,
    },
    /// `P(g(X, Y) > threshold)` from aligned draws with its Monte Carlo band
    /// and a verdict against a probability cutoff (spread reading).
    AlignedExceedance {
        /// The functional's stable name.
        functional: String,
        /// Draws of `X`, bits.
        x: Vec<u64>,
        /// Draws of `Y`, bits, row-aligned with `x`.
        y: Vec<u64>,
        /// Threshold of the composed value, bits.
        threshold: u64,
        /// Probability cutoff of the verdict, bits.
        cutoff: u64,
        /// Level of the Monte Carlo band, bits.
        band_level: u64,
    },
    /// `P(X > threshold)` of a finite mixture of draw sets with known weights
    /// (spread reading).
    MixtureExceedance {
        /// Weight bits.
        weights: Vec<u64>,
        /// Draws of each component, bits.
        components: Vec<Vec<u64>>,
        /// Threshold, bits.
        threshold: u64,
    },
    /// A finite mixture with known weights.
    Mixture {
        /// Weight bits.
        weights: Vec<u64>,
        /// Component mean bits.
        means: Vec<u64>,
        /// Component variance bits.
        variances: Vec<u64>,
    },
    /// A Fréchet-Hoeffding envelope of two marginals.
    FrechetEnvelope {
        /// The query.
        query: EnvelopeQueryWire,
        /// Marginal of `X`.
        x: MarginalWire,
        /// Marginal of `Y`.
        y: MarginalWire,
    },
    /// Ranges over the declared correlation range (in the rule).
    CorrelationRange {
        /// Mean of `X`, bits.
        x_mean: u64,
        /// Standard deviation of `X`, bits.
        x_sd: u64,
        /// Mean of `Y`, bits.
        y_mean: u64,
        /// Standard deviation of `Y`, bits.
        y_sd: u64,
        /// Coefficient of `X`, bits.
        a: u64,
        /// Coefficient of `Y`, bits.
        b: u64,
    },
    /// The pooled estimate and variance over the declared correlation range.
    PooledCorrelationRange {
        /// Study estimate bits.
        estimates: Vec<u64>,
        /// Study standard error bits.
        standard_errors: Vec<u64>,
    },
}

/// The versioned artifact.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JointLawPropagationArtifactWire {
    /// Format version.
    pub version: u32,
    /// Required feature marker.
    pub required_features: Vec<String>,
    /// The rule that supported the computation.
    pub rule: PropagationRuleWire,
    /// The inputs.
    pub inputs: Vec<PropagationInputWire>,
    /// The numbers consumed.
    pub request: PropagationRequestWire,
    /// The route the rule yielded.
    pub route: String,
    /// The label of the result.
    pub label: String,
    /// The alignment identity the rows share, for an aligned route.
    pub alignment_id: Option<String>,
    /// The covariance declaration identity, for a covariance route.
    pub covariance_id: Option<String>,
    /// Evidence identities shared by two or more inputs.
    pub shared_evidence: Vec<String>,
    /// The stored result (floats as bit patterns, counts as integers). A
    /// version 1 artifact stored a list of named entries; it is mapped into
    /// this body on decode.
    #[serde(deserialize_with = "deserialize_result")]
    pub result: PropagationResultWire,
    /// Digest of everything above except the result.
    pub identity: String,
}

/// A consumed artifact.
#[derive(Clone, Debug, PartialEq)]
pub struct ConsumedJointLawPropagation {
    /// The identity that was recomputed and verified.
    pub identity: String,
    /// The operation.
    pub operation: String,
    /// The route.
    pub route: String,
    /// The label of the result.
    pub label: String,
    /// The alignment identity, for an aligned route.
    pub alignment_id: Option<String>,
    /// The replayed result, bit-identical to the stored one.
    pub result: PropagationResultWire,
}

impl ConsumedJointLawPropagation {
    /// A named float value.
    #[must_use]
    pub fn value(&self, name: &str) -> Option<f64> {
        self.result.value(name)
    }

    /// Every named float value in stored order.
    #[must_use]
    pub fn values(&self) -> Vec<(&'static str, f64)> {
        self.result.float_entries().into_iter().map(|(n, b)| (n, f64::from_bits(b))).collect()
    }

    /// The number of aligned rows, for an aligned route.
    #[must_use]
    pub fn n_rows(&self) -> Option<u64> {
        self.result.n_rows()
    }

    /// The decision verdict of a composed exceedance decision.
    #[must_use]
    pub fn verdict(&self) -> Option<DecisionVerdict> {
        self.result.verdict()
    }

    /// Whether the result is an envelope (a structural envelope or an
    /// assumption range) and so is not a point.
    #[must_use]
    pub fn is_envelope(&self) -> bool {
        self.label != antecedent_contract::POINT_ONLY_LABEL
            && self.label != antecedent_contract::UNCALIBRATED_SPREAD_LABEL
    }
}

// ----------------------------------------------------------- result

/// A lower and an upper bound, as bit patterns.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoundsWire {
    /// Lower bound bits.
    pub lower: u64,
    /// Upper bound bits.
    pub upper: u64,
}

impl From<ValueBounds> for BoundsWire {
    fn from(bounds: ValueBounds) -> Self {
        Self { lower: bounds.lower.to_bits(), upper: bounds.upper.to_bits() }
    }
}

/// The quantity a single envelope bounds.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EnvelopeQuantityWire {
    /// `P(X + Y <= t)`.
    SumCdf,
    /// `P(X - Y <= t)`.
    DifferenceCdf,
    /// `E[X Y]`.
    ProductMoment,
    /// `Cov(X, Y)`.
    Covariance,
    /// `Var(a X + b Y)`.
    LinearVariance,
    /// `E[g(X, Y)]` for a closed value functional.
    Functional,
}

impl EnvelopeQuantityWire {
    const ALL: [Self; 6] = [
        Self::SumCdf,
        Self::DifferenceCdf,
        Self::ProductMoment,
        Self::Covariance,
        Self::LinearVariance,
        Self::Functional,
    ];

    /// The stable name of the lower bound.
    #[must_use]
    pub const fn lower_name(self) -> &'static str {
        match self {
            Self::SumCdf => "sum_cdf_lower",
            Self::DifferenceCdf => "difference_cdf_lower",
            Self::ProductMoment => "product_moment_lower",
            Self::Covariance => "covariance_lower",
            Self::LinearVariance => "linear_variance_lower",
            Self::Functional => "functional_lower",
        }
    }

    /// The stable name of the upper bound.
    #[must_use]
    pub const fn upper_name(self) -> &'static str {
        match self {
            Self::SumCdf => "sum_cdf_upper",
            Self::DifferenceCdf => "difference_cdf_upper",
            Self::ProductMoment => "product_moment_upper",
            Self::Covariance => "covariance_upper",
            Self::LinearVariance => "linear_variance_upper",
            Self::Functional => "functional_upper",
        }
    }
}

/// The stored verdict of a composed exceedance decision.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerdictWire {
    /// Act.
    Act,
    /// Withhold.
    Withhold,
    /// Unresolved.
    Unresolved,
}

impl From<DecisionVerdict> for VerdictWire {
    fn from(verdict: DecisionVerdict) -> Self {
        match verdict {
            DecisionVerdict::Act => Self::Act,
            DecisionVerdict::Withhold => Self::Withhold,
            DecisionVerdict::Unresolved => Self::Unresolved,
        }
    }
}

impl From<VerdictWire> for DecisionVerdict {
    fn from(verdict: VerdictWire) -> Self {
        match verdict {
            VerdictWire::Act => Self::Act,
            VerdictWire::Withhold => Self::Withhold,
            VerdictWire::Unresolved => Self::Unresolved,
        }
    }
}

/// The stored result of a propagation: floats as IEEE bit patterns, counts as
/// integers, the verdict as its own enum.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "snake_case")]
pub enum PropagationResultWire {
    /// Mean and variance of a linear combination under a covariance.
    Moments {
        /// Mean bits.
        mean: u64,
        /// Variance bits.
        variance: u64,
    },
    /// Moments of a linear combination of aligned draws.
    AlignedMoments {
        /// Mean bits.
        mean: u64,
        /// Variance bits.
        variance: u64,
        /// Covariance bits.
        covariance: u64,
        /// Aligned row count.
        n_rows: u64,
    },
    /// Mean of a functional over aligned draws.
    AlignedMean {
        /// Mean bits.
        mean: u64,
        /// Aligned row count.
        n_rows: u64,
    },
    /// Spread of a functional over aligned draws.
    AlignedSpread {
        /// Mean bits.
        mean: u64,
        /// Variance bits.
        variance: u64,
        /// Monte Carlo standard error of the mean, bits.
        mean_standard_error: u64,
        /// Aligned row count.
        n_rows: u64,
    },
    /// An exceedance probability over aligned draws with its verdict.
    AlignedExceedance {
        /// Probability bits.
        probability: u64,
        /// Standard error bits.
        standard_error: u64,
        /// Monte Carlo band bits.
        band: u64,
        /// The verdict against the cutoff.
        verdict: VerdictWire,
        /// Aligned row count.
        n_rows: u64,
    },
    /// An exceedance probability of a finite mixture.
    MixtureExceedance {
        /// Probability bits.
        probability: u64,
        /// Standard error bits.
        standard_error: u64,
    },
    /// Moments of a finite mixture.
    Mixture {
        /// Mean bits.
        mean: u64,
        /// Variance bits.
        variance: u64,
        /// Within-component variance bits.
        within: u64,
        /// Between-component variance bits.
        between: u64,
    },
    /// A Fréchet-Hoeffding envelope of one quantity.
    Envelope {
        /// The bounded quantity.
        quantity: EnvelopeQuantityWire,
        /// Its bounds.
        bounds: BoundsWire,
    },
    /// Ranges over a declared correlation range.
    CorrelationRange {
        /// Bounds of `E[X Y]`.
        product_moment: BoundsWire,
        /// Bounds of `Cov(X, Y)`.
        covariance: BoundsWire,
        /// Bounds of `Var(a X + b Y)`.
        linear_variance: BoundsWire,
    },
    /// The pooled estimate and variance over a declared correlation range.
    PooledRange {
        /// Bounds of the pooled estimate.
        estimate: BoundsWire,
        /// Bounds of the pooled variance.
        variance: BoundsWire,
    },
}

impl PropagationResultWire {
    /// Every float entry by its stable name, in stored order.
    #[must_use]
    pub fn float_entries(&self) -> Vec<(&'static str, u64)> {
        match *self {
            Self::Moments { mean, variance } => vec![("mean", mean), ("variance", variance)],
            Self::AlignedMoments { mean, variance, covariance, .. } => {
                vec![("mean", mean), ("variance", variance), ("covariance", covariance)]
            }
            Self::AlignedMean { mean, .. } => vec![("mean", mean)],
            Self::AlignedSpread { mean, variance, mean_standard_error, .. } => {
                vec![
                    ("mean", mean),
                    ("variance", variance),
                    ("mean_standard_error", mean_standard_error),
                ]
            }
            Self::AlignedExceedance { probability, standard_error, band, .. } => {
                vec![
                    ("probability", probability),
                    ("standard_error", standard_error),
                    ("band", band),
                ]
            }
            Self::MixtureExceedance { probability, standard_error } => {
                vec![("probability", probability), ("standard_error", standard_error)]
            }
            Self::Mixture { mean, variance, within, between } => vec![
                ("mean", mean),
                ("variance", variance),
                ("within", within),
                ("between", between),
            ],
            Self::Envelope { quantity, bounds } => {
                vec![(quantity.lower_name(), bounds.lower), (quantity.upper_name(), bounds.upper)]
            }
            Self::CorrelationRange { product_moment, covariance, linear_variance } => {
                let mut out = Vec::with_capacity(6);
                for (quantity, bounds) in [
                    (EnvelopeQuantityWire::ProductMoment, product_moment),
                    (EnvelopeQuantityWire::Covariance, covariance),
                    (EnvelopeQuantityWire::LinearVariance, linear_variance),
                ] {
                    out.push((quantity.lower_name(), bounds.lower));
                    out.push((quantity.upper_name(), bounds.upper));
                }
                out
            }
            Self::PooledRange { estimate, variance } => vec![
                ("pooled_estimate_lower", estimate.lower),
                ("pooled_estimate_upper", estimate.upper),
                ("pooled_variance_lower", variance.lower),
                ("pooled_variance_upper", variance.upper),
            ],
        }
    }

    /// A named float value.
    #[must_use]
    pub fn value(&self, name: &str) -> Option<f64> {
        self.float_entries().into_iter().find(|(n, _)| *n == name).map(|(_, b)| f64::from_bits(b))
    }

    /// The aligned row count, for a result over aligned draws.
    #[must_use]
    pub fn n_rows(&self) -> Option<u64> {
        match *self {
            Self::AlignedMoments { n_rows, .. }
            | Self::AlignedMean { n_rows, .. }
            | Self::AlignedSpread { n_rows, .. }
            | Self::AlignedExceedance { n_rows, .. } => Some(n_rows),
            _ => None,
        }
    }

    /// The decision verdict, for an exceedance decision.
    #[must_use]
    pub fn verdict(&self) -> Option<DecisionVerdict> {
        match self {
            Self::AlignedExceedance { verdict, .. } => Some((*verdict).into()),
            _ => None,
        }
    }

    /// The entries in the earlier named-list layout: floats as bit patterns,
    /// `n_rows` and `verdict_code` as integers.
    #[must_use]
    pub fn named_entries(&self) -> Vec<(String, u64)> {
        let mut out: Vec<(String, u64)> =
            self.float_entries().into_iter().map(|(n, b)| (n.to_owned(), b)).collect();
        if let Some(rows) = self.n_rows() {
            out.push(("n_rows".to_owned(), rows));
        }
        if let Some(verdict) = self.verdict() {
            // The earlier layout placed the verdict code before the row count.
            let at = out.len() - usize::from(self.n_rows().is_some());
            out.insert(at, ("verdict_code".to_owned(), verdict_code(verdict)));
        }
        out
    }

    /// Map the earlier named-list layout into the typed result. The entry
    /// names and their order identify the shape exactly as the earlier writer
    /// produced them.
    fn from_named_entries(entries: &[(String, u64)]) -> Result<Self, String> {
        let names: Vec<&str> = entries.iter().map(|(n, _)| n.as_str()).collect();
        let bits: Vec<u64> = entries.iter().map(|(_, b)| *b).collect();
        let bounds = |at: usize| BoundsWire { lower: bits[at], upper: bits[at + 1] };
        let shape = |expected: &[&str]| names == expected;
        if shape(&["mean", "variance"]) {
            return Ok(Self::Moments { mean: bits[0], variance: bits[1] });
        }
        if shape(&["mean", "variance", "covariance", "n_rows"]) {
            return Ok(Self::AlignedMoments {
                mean: bits[0],
                variance: bits[1],
                covariance: bits[2],
                n_rows: bits[3],
            });
        }
        if shape(&["mean", "n_rows"]) {
            return Ok(Self::AlignedMean { mean: bits[0], n_rows: bits[1] });
        }
        if shape(&["mean", "variance", "mean_standard_error", "n_rows"]) {
            return Ok(Self::AlignedSpread {
                mean: bits[0],
                variance: bits[1],
                mean_standard_error: bits[2],
                n_rows: bits[3],
            });
        }
        if shape(&["probability", "standard_error", "band", "verdict_code", "n_rows"]) {
            let verdict =
                [DecisionVerdict::Act, DecisionVerdict::Withhold, DecisionVerdict::Unresolved]
                    .into_iter()
                    .find(|verdict| verdict_code(*verdict) == bits[3])
                    .ok_or_else(|| "unknown stored verdict code".to_owned())?;
            return Ok(Self::AlignedExceedance {
                probability: bits[0],
                standard_error: bits[1],
                band: bits[2],
                verdict: verdict.into(),
                n_rows: bits[4],
            });
        }
        if shape(&["probability", "standard_error"]) {
            return Ok(Self::MixtureExceedance { probability: bits[0], standard_error: bits[1] });
        }
        if shape(&["mean", "variance", "within", "between"]) {
            return Ok(Self::Mixture {
                mean: bits[0],
                variance: bits[1],
                within: bits[2],
                between: bits[3],
            });
        }
        if let Some(quantity) =
            EnvelopeQuantityWire::ALL.into_iter().find(|q| shape(&[q.lower_name(), q.upper_name()]))
        {
            return Ok(Self::Envelope { quantity, bounds: bounds(0) });
        }
        if shape(&[
            "product_moment_lower",
            "product_moment_upper",
            "covariance_lower",
            "covariance_upper",
            "linear_variance_lower",
            "linear_variance_upper",
        ]) {
            return Ok(Self::CorrelationRange {
                product_moment: bounds(0),
                covariance: bounds(2),
                linear_variance: bounds(4),
            });
        }
        if shape(&[
            "pooled_estimate_lower",
            "pooled_estimate_upper",
            "pooled_variance_lower",
            "pooled_variance_upper",
        ]) {
            return Ok(Self::PooledRange { estimate: bounds(0), variance: bounds(2) });
        }
        Err("stored result entries do not form a known result".to_owned())
    }
}

/// Read the result of either layout: the typed body, or the earlier list of
/// named entries mapped into it.
fn deserialize_result<'de, D>(deserializer: D) -> Result<PropagationResultWire, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Stored {
        Typed(PropagationResultWire),
        Named(Vec<(String, u64)>),
    }
    match Stored::deserialize(deserializer)? {
        Stored::Typed(result) => Ok(result),
        Stored::Named(entries) => {
            if entries.len() > MAX_NAMED_RESULT_ENTRIES {
                return Err(serde::de::Error::custom("too many stored result entries"));
            }
            PropagationResultWire::from_named_entries(&entries).map_err(serde::de::Error::custom)
        }
    }
}

/// Most entries the earlier named-list layout may carry.
const MAX_NAMED_RESULT_ENTRIES: usize = 16;
