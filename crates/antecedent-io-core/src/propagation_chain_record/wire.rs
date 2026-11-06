//! Public wire forms and constructors for propagation-chain records.

use antecedent_contract::dependence::SharedFactorKind;
use antecedent_contract::propagation_chain::{ChainPlan, LawMeaning, Representation, StageKind};
use serde::{Deserialize, Serialize};

use super::{bits, word_of};

// ------------------------------------------------------------------ the wire

/// One declared shared factor.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChainFactorWire {
    /// `row`, `cluster`, `model`, `parameter`, `prior`, `disturbance` or `period`.
    pub kind: String,
    /// The factor's identity.
    pub id: String,
}

/// One source of the chain.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChainSourceWire {
    /// The source's identity.
    pub id: String,
    /// `point`, `interval` or `draws`.
    pub representation: String,
    /// `sampling` or `posterior`, for draws.
    pub meaning: Option<String>,
    /// The alignment identity of the draw table, for draws.
    pub alignment: Option<String>,
    /// Whether the factor list is complete.
    pub complete_disclosure: bool,
    /// The factors the source uses.
    pub factors: Vec<ChainFactorWire>,
    /// Bit patterns: the draws of a column, one value for a point, the two
    /// ends of an interval.
    pub values: Vec<u64>,
}

/// A coefficient of an affine stage.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ChainCoefficientWire {
    /// A constant, as bits.
    Fixed {
        /// Value bits.
        value: u64,
    },
    /// The stage's input at this position after the first.
    Input {
        /// Position among the stage's inputs.
        position: u64,
    },
}

/// A unary transformation.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ChainUnaryWire {
    /// `exp(x)`.
    Exp,
    /// `ln(x)`.
    Log,
    /// `x^2`.
    Square,
    /// `1 / x`.
    Reciprocal,
    /// The logistic function.
    Logistic,
    /// `1 - exp(-r x)`.
    ExponentialUtility {
        /// Risk aversion bits.
        risk_aversion: u64,
    },
    /// The kinked linear utility.
    PiecewiseLinear {
        /// Kink bits.
        kink: u64,
        /// Slope below the kink, bits.
        below_slope: u64,
        /// Slope at and above the kink, bits.
        above_slope: u64,
    },
    /// The step utility.
    Threshold {
        /// Cutoff bits.
        cutoff: u64,
        /// Value at and below the cutoff, bits.
        below: u64,
        /// Value above the cutoff, bits.
        above: u64,
    },
}

/// One term of an expression stage, in postfix order.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "term", rename_all = "snake_case", deny_unknown_fields)]
pub enum ChainExprTermWire {
    /// Push the stage's input at this position.
    Input {
        /// Position among the stage's inputs.
        position: u64,
    },
    /// Push a constant, as bits.
    Constant {
        /// Value bits.
        value: u64,
    },
    /// Pop `b`, `a`; push `a + b`.
    Add,
    /// Pop `b`, `a`; push `a - b`.
    Subtract,
    /// Pop `b`, `a`; push `a * b`.
    Multiply,
    /// Pop `b`, `a`; push `a / b`, refused when the draws of `b` reach zero or
    /// some draw has magnitude below the bound.
    Divide {
        /// Smallest magnitude a denominator draw may have, bits.
        min_magnitude: u64,
    },
    /// Check the top column lies in the range; refused outside it, never clipped.
    RequireRange {
        /// Smallest admissible value, bits.
        low: u64,
        /// Largest admissible value, bits.
        high: u64,
    },
}

/// The operator of one stage.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum ChainOpWire {
    /// `input_0 * scale + shift`.
    Affine {
        /// Slope.
        scale: ChainCoefficientWire,
        /// Intercept.
        shift: ChainCoefficientWire,
    },
    /// A fixed-weight combination.
    Reweight {
        /// Weight bits, one per input.
        weights: Vec<u64>,
    },
    /// A pool.
    Pool {
        /// Whether the weights come from the joint covariance of the draws.
        dependent: bool,
    },
    /// A binary transformation.
    Binary {
        /// `sum`, `difference`, `product` or `ratio`.
        combine: String,
    },
    /// A unary transformation.
    Unary {
        /// The transformation.
        function: ChainUnaryWire,
    },
    /// A preposterior update by a future study.
    Preposterior {
        /// Per-observation noise scale bits.
        noise_sd: u64,
        /// Observation count bits.
        sample_size: u64,
        /// Reference utility bits.
        reference_utility: u64,
        /// Standard-normal draws of the simulated studies, one per row, bits.
        noise: Vec<u64>,
    },
    /// A formula over the inputs in postfix order: a measurement correction as a
    /// ratio of aligned draws, a predictive line with an independent residual.
    Expression {
        /// The terms, in postfix order.
        terms: Vec<ChainExprTermWire>,
    },
}

/// One stage of the chain.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChainStageWire {
    /// The stage's identity.
    pub id: String,
    /// The stage kind's stable name.
    pub kind: String,
    /// Identities of the nodes it reads, in operator order.
    pub inputs: Vec<String>,
    /// Factors the stage brings itself.
    pub own_factors: Vec<ChainFactorWire>,
    /// The operator.
    pub op: ChainOpWire,
    /// For a utility stage, the utility of the reference action the expected
    /// utility is compared with (bits).
    pub decision_reference: Option<u64>,
}

/// A change appended to a record.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "change", rename_all = "snake_case", deny_unknown_fields)]
pub enum ChainChangeWire {
    /// The source is withdrawn: every stage that depends on it is invalidated.
    Withdraw {
        /// The withdrawn source.
        source: String,
    },
    /// An assumption about the source changed (a prior, a model, a selection
    /// weight): its rows are re-weighted, and every stage that depends on it
    /// is summarized under the new weights.
    Reweight {
        /// The re-weighted source.
        source: String,
        /// Log importance weight of each row, bits.
        log_weights: Vec<u64>,
    },
}

/// The stored result of one stage.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChainStageResultWire {
    /// The stage's identity.
    pub stage: String,
    /// `current`, `reweighted` or `invalidated`.
    pub status: String,
    /// Withdrawn sources the stage depends on, for an invalidated stage.
    pub invalidated_by: Vec<String>,
    /// Named values: floats as bit patterns, counts as integers.
    pub values: Vec<(String, u64)>,
}

/// The retained propagation record.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PropagationChainRecordWire {
    /// Format version.
    pub version: u32,
    /// Required feature marker.
    pub required_features: Vec<String>,
    /// `joint_draws` or `independent_product`.
    pub construction: String,
    /// `point`, `spread`, `posterior`, `enclosure` or `interval`.
    pub reading: String,
    /// The stage whose output is the answer.
    pub output: String,
    /// The sources.
    pub sources: Vec<ChainSourceWire>,
    /// The stages.
    pub stages: Vec<ChainStageWire>,
    /// The changes, in the order they were appended.
    pub changes: Vec<ChainChangeWire>,
    /// The label of the answer.
    pub label: String,
    /// The stored result of every stage, in stage order.
    pub results: Vec<ChainStageResultWire>,
    /// Digest of everything above except the results.
    pub identity: String,
}

/// A consumed record.
#[derive(Clone, Debug, PartialEq)]
pub struct ConsumedPropagationChain {
    /// The identity that was recomputed and verified.
    pub identity: String,
    /// The label of the answer.
    pub label: String,
    /// The stage whose output is the answer.
    pub output: String,
    /// The replayed results, bit-identical to the stored ones.
    pub results: Vec<ChainStageResultWire>,
    /// The plan the chain rule yielded.
    pub plan: ChainPlan,
}

/// Whether a stored value name holds an integer count or code.
fn is_count(name: &str) -> bool {
    matches!(name, "n_rows" | "verdict_code")
}

impl ChainStageResultWire {
    /// A named float value, if the stage carries it.
    #[must_use]
    pub fn value(&self, name: &str) -> Option<f64> {
        if is_count(name) {
            return None;
        }
        self.values.iter().find(|(n, _)| n == name).map(|(_, bits)| f64::from_bits(*bits))
    }

    /// A named count, if the stage carries it.
    #[must_use]
    pub fn count(&self, name: &str) -> Option<u64> {
        if !is_count(name) {
            return None;
        }
        self.values.iter().find(|(n, _)| n == name).map(|(_, v)| *v)
    }
}

impl ConsumedPropagationChain {
    /// The stored result of a stage.
    #[must_use]
    pub fn stage(&self, id: &str) -> Option<&ChainStageResultWire> {
        self.results.iter().find(|result| result.stage == id)
    }

    /// The result of the stage whose output is the answer.
    #[must_use]
    pub fn answer(&self) -> Option<&ChainStageResultWire> {
        self.stage(&self.output)
    }
}

// -------------------------------------------------------------- constructors

impl ChainFactorWire {
    /// A factor of `kind` with identity `id`.
    #[must_use]
    pub fn new(kind: SharedFactorKind, id: &str) -> Self {
        Self { kind: kind.as_str().to_owned(), id: id.to_owned() }
    }
}

impl ChainSourceWire {
    /// One column of an aligned draw table.
    #[must_use]
    pub fn draws(id: &str, meaning: LawMeaning, alignment: &str, values: &[f64]) -> Self {
        Self {
            id: id.to_owned(),
            representation: Representation::Draws.as_str().to_owned(),
            meaning: Some(meaning.as_str().to_owned()),
            alignment: Some(alignment.to_owned()),
            complete_disclosure: true,
            factors: Vec::new(),
            values: bits(values),
        }
    }

    /// A value fixed by declaration.
    #[must_use]
    pub fn point(id: &str, value: f64) -> Self {
        Self {
            id: id.to_owned(),
            representation: Representation::Point.as_str().to_owned(),
            meaning: None,
            alignment: None,
            complete_disclosure: true,
            factors: Vec::new(),
            values: bits(&[value]),
        }
    }

    /// A declared interval.
    #[must_use]
    pub fn interval(id: &str, lower: f64, upper: f64) -> Self {
        Self {
            id: id.to_owned(),
            representation: Representation::Interval.as_str().to_owned(),
            meaning: None,
            alignment: None,
            complete_disclosure: true,
            factors: Vec::new(),
            values: bits(&[lower, upper]),
        }
    }

    /// Declare that the source uses a factor.
    #[must_use]
    pub fn with_factor(mut self, kind: SharedFactorKind, id: &str) -> Self {
        self.factors.push(ChainFactorWire::new(kind, id));
        self
    }

    /// Declare that the factor list may be partial: the overlap with other
    /// sources is unknown.
    #[must_use]
    pub fn with_unknown_overlap(mut self) -> Self {
        self.complete_disclosure = false;
        self
    }
}

impl ChainCoefficientWire {
    /// A constant coefficient.
    #[must_use]
    pub fn fixed(value: f64) -> Self {
        Self::Fixed { value: value.to_bits() }
    }

    /// The stage's input at `position` (after the first) as the coefficient.
    #[must_use]
    pub fn input(position: usize) -> Self {
        Self::Input { position: word_of(position) }
    }
}

impl ChainUnaryWire {
    /// The exponential utility `1 - exp(-r x)`.
    #[must_use]
    pub fn exponential_utility(risk_aversion: f64) -> Self {
        Self::ExponentialUtility { risk_aversion: risk_aversion.to_bits() }
    }

    /// The kinked linear utility.
    #[must_use]
    pub fn piecewise_linear(kink: f64, below_slope: f64, above_slope: f64) -> Self {
        Self::PiecewiseLinear {
            kink: kink.to_bits(),
            below_slope: below_slope.to_bits(),
            above_slope: above_slope.to_bits(),
        }
    }

    /// The step utility.
    #[must_use]
    pub fn threshold(cutoff: f64, below: f64, above: f64) -> Self {
        Self::Threshold { cutoff: cutoff.to_bits(), below: below.to_bits(), above: above.to_bits() }
    }
}

impl ChainExprTermWire {
    /// Push the stage's input at `position`.
    #[must_use]
    pub fn input(position: usize) -> Self {
        Self::Input { position: word_of(position) }
    }

    /// Push a constant.
    #[must_use]
    pub fn constant(value: f64) -> Self {
        Self::Constant { value: value.to_bits() }
    }

    /// A division whose denominator draws must stay at least `min_magnitude`
    /// away from zero (and never reach it).
    #[must_use]
    pub fn divide(min_magnitude: f64) -> Self {
        Self::Divide { min_magnitude: min_magnitude.to_bits() }
    }

    /// A check that the top column lies in `[low, high]`.
    #[must_use]
    pub fn require_range(low: f64, high: f64) -> Self {
        Self::RequireRange { low: low.to_bits(), high: high.to_bits() }
    }
}

impl ChainOpWire {
    /// A formula over the inputs in postfix order.
    #[must_use]
    pub fn expression(terms: Vec<ChainExprTermWire>) -> Self {
        Self::Expression { terms }
    }

    /// `input_0 * scale + shift`.
    #[must_use]
    pub fn affine(scale: ChainCoefficientWire, shift: ChainCoefficientWire) -> Self {
        Self::Affine { scale, shift }
    }

    /// A fixed-weight combination of the inputs.
    #[must_use]
    pub fn reweight(weights: &[f64]) -> Self {
        Self::Reweight { weights: bits(weights) }
    }

    /// A pool of the inputs.
    #[must_use]
    pub fn pool(dependent: bool) -> Self {
        Self::Pool { dependent }
    }

    /// A binary transformation (`sum`, `difference`, `product`, `ratio`).
    #[must_use]
    pub fn binary(combine: &str) -> Self {
        Self::Binary { combine: combine.to_owned() }
    }

    /// A unary transformation.
    #[must_use]
    pub fn unary(function: ChainUnaryWire) -> Self {
        Self::Unary { function }
    }

    /// A preposterior update by a future study with the given standard-normal
    /// draws, one per row.
    #[must_use]
    pub fn preposterior(
        noise_sd: f64,
        sample_size: f64,
        reference_utility: f64,
        noise: &[f64],
    ) -> Self {
        Self::Preposterior {
            noise_sd: noise_sd.to_bits(),
            sample_size: sample_size.to_bits(),
            reference_utility: reference_utility.to_bits(),
            noise: bits(noise),
        }
    }
}

impl ChainStageWire {
    /// A stage reading `inputs`.
    #[must_use]
    pub fn new(id: &str, kind: StageKind, inputs: &[&str], op: ChainOpWire) -> Self {
        Self {
            id: id.to_owned(),
            kind: kind.as_str().to_owned(),
            inputs: inputs.iter().map(|input| (*input).to_owned()).collect(),
            own_factors: Vec::new(),
            op,
            decision_reference: None,
        }
    }

    /// Declare a factor the stage brings itself (a calibration sample behind a
    /// measurement mapping, a selection model behind a transport).
    #[must_use]
    pub fn with_own_factor(mut self, kind: SharedFactorKind, id: &str) -> Self {
        self.own_factors.push(ChainFactorWire::new(kind, id));
        self
    }

    /// Compare the stage's expected utility with the utility of a reference
    /// action, and report the verdict.
    #[must_use]
    pub fn with_decision_reference(mut self, reference_utility: f64) -> Self {
        self.decision_reference = Some(reference_utility.to_bits());
        self
    }
}

impl ChainChangeWire {
    /// Withdraw `source`.
    #[must_use]
    pub fn withdraw(source: &str) -> Self {
        Self::Withdraw { source: source.to_owned() }
    }

    /// Re-weight the rows of `source` by the given log importance weights.
    #[must_use]
    pub fn reweight(source: &str, log_weights: &[f64]) -> Self {
        Self::Reweight { source: source.to_owned(), log_weights: bits(log_weights) }
    }
}

// --------------------------------------------------------------- conversions
