//! Propagation of structural enclosures through chain stages.

use super::{
    CANCEL_STRIDE, ChainError, Coefficient, Combine, ExprTerm, StageOp, StageSpec,
    UnaryFn, check_cancel, validate_spec,
};
use antecedent_core::CancellationToken;

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
        if let Some(turn) = function.turning_point()
            && self.lower < turn
            && turn < self.upper
        {
            values.push(function.apply(turn));
        }
        Ok(Self::hull(&values))
    }
}

/// One source of a draw-indexed assumption envelope. A fixed enclosure is an
/// assumption without probabilities; a draw column carries its own law meaning
/// in the compiler contract, outside this numerical evaluator.
#[derive(Clone, Copy, Debug)]
pub enum MixedSource<'a> {
    /// The same interval assumption on every draw row.
    Fixed(Enclosure),
    /// An aligned column of sampling or posterior draws.
    Draws(&'a [f64]),
}

/// Evaluate each draw row with its draw values fixed and its interval
/// assumptions left as boxes. The result is one enclosure per stage per row.
/// Averaging lower and upper endpoints later encloses the expected value under
/// every fixed admissible assumption. It grants no sampling coverage and puts
/// no probability on the interval assumptions.
///
/// # Errors
/// Missing or mismatched draw rows, a table budget exceeded, cancellation, or
/// an interval operator whose image is not finite.
pub fn evaluate_mixed_enclosures(
    sources: &[MixedSource<'_>],
    stages: &[StageSpec],
    cancel: &CancellationToken,
) -> Result<Vec<Vec<Enclosure>>, ChainError> {
    validate_spec(sources.len(), stages)?;
    let Some(rows) = sources.iter().find_map(|source| match source {
        MixedSource::Draws(values) => Some(values.len()),
        MixedSource::Fixed(_) => None,
    }) else {
        return Err(ChainError::Shape("a mixed enclosure needs draw rows"));
    };
    if rows == 0 {
        return Err(ChainError::Shape("a mixed enclosure needs draw rows"));
    }
    let Some(cells) = rows.checked_mul(sources.len() + stages.len()) else {
        return Err(super::chain_search(crate::execution_quote::unrepresentable_quote(
            "chain table cells",
        )));
    };
    let Ok(quoted) = u64::try_from(cells) else {
        return Err(super::chain_search(crate::execution_quote::unrepresentable_quote(
            "chain table cells",
        )));
    };
    let _lease = super::admit_chain_cells(
        quoted,
        &crate::execution_quote::open_execution(),
        "chain table cells",
    )?;
    for source in sources {
        if let MixedSource::Draws(values) = source
            && (values.len() != rows || values.iter().any(|value| !value.is_finite()))
        {
            return Err(ChainError::Shape("mixed draw columns must be aligned and finite"));
        }
    }
    let mut columns: Vec<Vec<Enclosure>> =
        (0..stages.len()).map(|_| Vec::with_capacity(rows)).collect();
    for row in 0..rows {
        if row % CANCEL_STRIDE == 0 {
            check_cancel(cancel)?;
        }
        let boxes: Vec<Enclosure> = sources
            .iter()
            .map(|source| match source {
                MixedSource::Fixed(value) => Ok(*value),
                MixedSource::Draws(values) => Enclosure::new(values[row], values[row]),
            })
            .collect::<Result<_, _>>()?;
        let evaluated = evaluate_enclosures(&boxes, stages)?;
        for (column, value) in columns.iter_mut().zip(evaluated) {
            column.push(value);
        }
    }
    Ok(columns)
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
