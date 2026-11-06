//! Propagation of structural enclosures through chain stages.

use super::{
    ChainError, Coefficient, Combine, ExprTerm, StageOp, StageSpec, UnaryFn, validate_spec,
};

// ------------------------------------------------------------------ enclosures

/// A closed interval of values.
///
/// An enclosure is the image of a box of inputs under a chain. It holds under
/// every coupling of the inputs and is not an interval of uncertainty, a
/// spread or a posterior.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Enclosure {
    /// Lower end.
    pub lower: f64,
    /// Upper end.
    pub upper: f64,
}

impl Enclosure {
    /// An enclosure from its ends.
    ///
    /// # Errors
    /// An end that is not finite, or ends out of order.
    pub fn new(lower: f64, upper: f64) -> Result<Self, ChainError> {
        if !(lower.is_finite() && upper.is_finite()) {
            return Err(ChainError::NonFinite("enclosure end"));
        }
        if lower > upper {
            return Err(ChainError::Shape("an enclosure's lower end exceeds its upper end"));
        }
        Ok(Self { lower, upper })
    }

    fn point(value: f64) -> Self {
        Self { lower: value, upper: value }
    }

    fn hull(values: &[f64]) -> Self {
        let lower = values.iter().copied().fold(f64::INFINITY, f64::min);
        let upper = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        Self { lower, upper }
    }

    fn contains_zero(self) -> bool {
        self.lower <= 0.0 && 0.0 <= self.upper
    }

    fn add(self, other: Self) -> Self {
        Self { lower: self.lower + other.lower, upper: self.upper + other.upper }
    }

    fn negate(self) -> Self {
        Self { lower: -self.upper, upper: -self.lower }
    }

    fn scale(self, factor: f64) -> Self {
        Self::hull(&[self.lower * factor, self.upper * factor])
    }

    fn mul(self, other: Self) -> Self {
        Self::hull(&[
            self.lower * other.lower,
            self.lower * other.upper,
            self.upper * other.lower,
            self.upper * other.upper,
        ])
    }

    fn reciprocal(self) -> Result<Self, ChainError> {
        if self.contains_zero() {
            return Err(ChainError::EnclosureUnbounded("a divisor interval contains zero"));
        }
        Ok(Self::hull(&[1.0 / self.lower, 1.0 / self.upper]))
    }

    fn map(self, function: UnaryFn) -> Result<Self, ChainError> {
        match function {
            UnaryFn::Log if self.lower <= 0.0 => {
                return Err(ChainError::EnclosureUnbounded(
                    "a logarithm of an interval reaching zero",
                ));
            }
            UnaryFn::Reciprocal => return self.reciprocal(),
            _ => {}
        }
        let mut values = vec![function.apply(self.lower), function.apply(self.upper)];
        if let Some(turn) = function.turning_point() {
            if self.lower < turn && turn < self.upper {
                values.push(function.apply(turn));
            }
        }
        Ok(Self::hull(&values))
    }
}

fn check_enclosure(enclosure: Enclosure) -> Result<Enclosure, ChainError> {
    if enclosure.lower.is_finite() && enclosure.upper.is_finite() {
        Ok(enclosure)
    } else {
        Err(ChainError::Domain("an enclosure left the finite numbers"))
    }
}

/// Interval arithmetic of an expression. An input that appears twice is two
/// independent boxes, so the result holds under every coupling and may be wider
/// than the image of the box.
fn enclose_expression(terms: &[ExprTerm], inputs: &[Enclosure]) -> Result<Enclosure, ChainError> {
    let mut stack: Vec<Enclosure> = Vec::with_capacity(terms.len());
    let underflow = ChainError::Shape("an expression operation reads two columns");
    for term in terms {
        match term {
            ExprTerm::Input(slot) => stack.push(inputs[*slot]),
            ExprTerm::Constant(value) => stack.push(Enclosure::point(*value)),
            ExprTerm::Add | ExprTerm::Subtract | ExprTerm::Multiply => {
                let (Some(right), Some(left)) = (stack.pop(), stack.pop()) else {
                    return Err(underflow);
                };
                stack.push(match term {
                    ExprTerm::Add => left.add(right),
                    ExprTerm::Subtract => left.add(right.negate()),
                    _ => left.mul(right),
                });
            }
            ExprTerm::Divide { min_magnitude } => {
                let (Some(right), Some(left)) = (stack.pop(), stack.pop()) else {
                    return Err(underflow);
                };
                let nearest = if right.lower > 0.0 { right.lower } else { -right.upper };
                if right.contains_zero() || nearest < *min_magnitude {
                    return Err(ChainError::EnclosureUnbounded(
                        "a divisor interval reaches zero or the declared bound",
                    ));
                }
                stack.push(left.mul(right.reciprocal()?));
            }
            ExprTerm::RequireRange { low, high } => {
                let Some(top) = stack.last() else {
                    return Err(underflow);
                };
                if top.lower < *low || top.upper > *high {
                    return Err(ChainError::Domain(
                        "an expression enclosure leaves its required range",
                    ));
                }
            }
        }
    }
    match (stack.pop(), stack.is_empty()) {
        (Some(result), true) => Ok(result),
        _ => Err(ChainError::Shape("an expression leaves exactly one column")),
    }
}

/// Carry source intervals through a chain by interval arithmetic.
///
/// A point is an interval with equal ends. The result holds under every
/// coupling of the sources. A pool and a preposterior update are defined on
/// draws, not on intervals, and are refused: an interval is not a law.
///
/// # Errors
/// A malformed chain, a pool or update stage, or a divisor or logarithm
/// interval that cannot be bounded.
pub fn evaluate_enclosures(
    sources: &[Enclosure],
    stages: &[StageSpec],
) -> Result<Vec<Enclosure>, ChainError> {
    validate_spec(sources.len(), stages)?;
    let mut out: Vec<Enclosure> = Vec::with_capacity(stages.len());
    let node = |nodes: &[Enclosure], index: usize| -> Enclosure {
        if index < sources.len() { sources[index] } else { nodes[index - sources.len()] }
    };
    for stage in stages {
        let inputs: Vec<Enclosure> = stage.inputs.iter().map(|i| node(&out, *i)).collect();
        let result = match &stage.op {
            StageOp::Affine { scale, shift } => {
                let pick = |coefficient: &Coefficient| match coefficient {
                    Coefficient::Fixed(value) => Enclosure::point(*value),
                    Coefficient::Input(slot) => inputs[*slot],
                };
                inputs[0].mul(pick(scale)).add(pick(shift))
            }
            StageOp::Reweight { weights } => {
                let mut total = Enclosure::point(0.0);
                for (weight, input) in weights.iter().zip(&inputs) {
                    total = total.add(input.scale(*weight));
                }
                total
            }
            StageOp::Binary(combine) => match combine {
                Combine::Sum => inputs[0].add(inputs[1]),
                Combine::Difference => inputs[0].add(inputs[1].negate()),
                Combine::Product => inputs[0].mul(inputs[1]),
                Combine::Ratio => inputs[0].mul(inputs[1].reciprocal()?),
            },
            StageOp::Unary(function) => inputs[0].map(*function)?,
            StageOp::Expression(terms) => enclose_expression(terms, &inputs)?,
            StageOp::Pool { .. } => {
                return Err(ChainError::Operator(
                    "an interval is not a law: pooling needs aligned draws",
                ));
            }
            StageOp::Preposterior(_) => {
                return Err(ChainError::Operator(
                    "an interval is not a posterior: a future-study update needs posterior draws",
                ));
            }
        };
        out.push(check_enclosure(result)?);
    }
    Ok(out)
}
