//! A coherent finite joint from a chain of categorical conditional models.
//!
//! Category and coordinate ordering are explicit inputs, never inferred from
//! observed frequencies. Each categorical conditional uses ordered binary hazards.

use crate::{
    DesignView, FittedPredictor, LearnError, LearnerSpec, PredictionTask, RowSelection, TargetView,
    resolve_for,
};
use antecedent_core::ExecutionContext;

/// Fitted finite joint with retained empirical support counts.
///
/// The law is estimated on the *observed* support: a category level that never occurs
/// in the sample has an at-risk hazard of exactly `0` in every context, so every cell
/// containing it has probability exactly `0` — a statement about the sample, not a
/// structural zero of the population. [`Self::counts`] reports the zero cells so a caller
/// can tell them apart; a level that should be possible despite being unobserved needs
/// external smoothing before the joint is used as a population law.
pub struct FiniteJoint {
    cardinalities: Vec<usize>,
    conditionals: Vec<Vec<Conditional>>,
    counts: Vec<u64>,
}
enum Conditional {
    Constant(f64),
    Learned(Box<dyn FittedPredictor>),
}

impl FiniteJoint {
    /// Fit one coherent law on categorical codes aligned by physical row.
    /// # Errors
    /// Invalid domains/codes, empty samples, provider or resource failures.
    #[allow(clippy::float_cmp)] // exact constants: the values compared are representable results, not measurements
    pub fn fit(
        columns: &[Vec<usize>],
        cardinalities: &[usize],
        learner: LearnerSpec,
        max_cells: usize,
        ctx: &ExecutionContext,
    ) -> Result<Self, LearnError> {
        learner.validate()?;
        let size = joint_size(cardinalities, max_cells)?;
        let n = columns.first().map_or(0, Vec::len);
        if columns.len() != cardinalities.len()
            || n == 0
            || n > u32::MAX as usize
            || columns
                .iter()
                .zip(cardinalities)
                .any(|(col, k)| col.len() != n || col.iter().any(|v| v >= k))
        {
            return Err(LearnError::Shape { message: "invalid finite categorical sample" });
        }
        workspace(n, cardinalities, size, ctx)?;
        let mut counts = vec![0u64; size];
        for row in 0..n {
            let index = columns.iter().zip(cardinalities).fold(0, |i, (c, k)| i * k + c[row]);
            counts[index] += 1;
        }
        let mut conditionals = Vec::new();
        let factory = resolve_for(learner, PredictionTask::BinaryProbability)?;
        for axis in 0..columns.len() {
            if ctx.cancellation.is_cancelled() {
                return Err(LearnError::Unsupported { message: "categorical fit cancelled" });
            }
            let width = 1 + cardinalities[..axis].iter().map(|k| k - 1).sum::<usize>();
            let mut design = vec![
                1.0;
                n.checked_mul(width).ok_or(LearnError::Shape {
                    message: "categorical design overflow"
                })?
            ];
            let mut col = 1;
            for previous in 0..axis {
                for level in 0..cardinalities[previous] - 1 {
                    for row in 0..n {
                        design[col * n + row] = f64::from(columns[previous][row] == level);
                    }
                    col += 1;
                }
            }
            let x = DesignView::from_column_major(&design, n, width)?;
            let mut hazards = Vec::new();
            for level in 0..cardinalities[axis] - 1 {
                #[allow(clippy::cast_possible_truncation)]
                let rows: Vec<u32> =
                    (0..n).filter(|r| columns[axis][*r] >= level).map(|r| r as u32).collect();
                let y: Vec<f64> = columns[axis].iter().map(|v| f64::from(*v == level)).collect();
                let positives = rows.iter().filter(|r| y[**r as usize] == 1.0).count();
                hazards.push(if rows.is_empty() || positives == 0 {
                    Conditional::Constant(0.0)
                } else if positives == rows.len() {
                    Conditional::Constant(1.0)
                } else {
                    Conditional::Learned(factory.fit(
                        x.with_rows(RowSelection::new(&rows))?,
                        TargetView::new(&y),
                        None,
                        ctx,
                    )?)
                });
            }
            conditionals.push(hazards);
        }
        Ok(Self { cardinalities: cardinalities.to_vec(), conditionals, counts })
    }

    /// Empirical counts, in the same Cartesian order as the predicted law.
    #[must_use]
    pub fn counts(&self) -> &[u64] {
        &self.counts
    }

    /// Materialize a normalized joint, bounded by the fit-time Cartesian limit.
    /// # Errors
    /// Invalid provider probabilities, cancellation, or provider prediction failure.
    #[allow(clippy::needless_range_loop)] // the index is a node/row id shared by several parallel tables
    pub fn probabilities(&self, ctx: &ExecutionContext) -> Result<Vec<f64>, LearnError> {
        let size = self.counts.len();
        workspace(size, &self.cardinalities, size, ctx)?;
        let mut result = vec![1.0; size];
        let mut assignments = vec![vec![0usize; size]; self.cardinalities.len()];
        for i in 0..size {
            let mut value = i;
            for axis in (0..self.cardinalities.len()).rev() {
                assignments[axis][i] = value % self.cardinalities[axis];
                value /= self.cardinalities[axis];
            }
        }
        for axis in 0..self.cardinalities.len() {
            let width = 1 + self.cardinalities[..axis].iter().map(|k| k - 1).sum::<usize>();
            let mut design = vec![1.0; size * width];
            let mut col = 1;
            for previous in 0..axis {
                for level in 0..self.cardinalities[previous] - 1 {
                    for row in 0..size {
                        design[col * size + row] = f64::from(assignments[previous][row] == level);
                    }
                    col += 1;
                }
            }
            let x = DesignView::from_column_major(&design, size, width)?;
            for (level, model) in self.conditionals[axis].iter().enumerate() {
                let mut probabilities = vec![0.0; size];
                match model {
                    Conditional::Constant(p) => probabilities.fill(*p),
                    Conditional::Learned(model) => model.predict(x, &mut probabilities, ctx)?,
                }
                for row in 0..size {
                    let p = probabilities[row];
                    if !p.is_finite() || !(0.0..=1.0).contains(&p) {
                        return Err(LearnError::Shape {
                            message: "invalid categorical probability",
                        });
                    }
                    match assignments[axis][row].cmp(&level) {
                        std::cmp::Ordering::Equal => result[row] *= p,
                        std::cmp::Ordering::Greater => result[row] *= 1.0 - p,
                        std::cmp::Ordering::Less => {}
                    }
                }
            }
            if ctx.cancellation.is_cancelled() {
                return Err(LearnError::Unsupported {
                    message: "categorical prediction cancelled",
                });
            }
        }
        Ok(result)
    }
}

fn workspace(
    rows: usize,
    cardinalities: &[usize],
    cells: usize,
    ctx: &ExecutionContext,
) -> Result<(), LearnError> {
    let width = cardinalities
        .iter()
        .try_fold(1usize, |n, k| n.checked_add(k - 1))
        .ok_or(LearnError::Shape { message: "categorical design overflow" })?;
    let bytes = rows
        .checked_mul(width + cardinalities.len() + 8)
        .and_then(|n| n.checked_add(cells))
        .and_then(|n| n.checked_mul(8))
        .ok_or(LearnError::Shape { message: "categorical workspace overflow" })?;
    if ctx.cancellation.is_cancelled()
        || ctx.memory.hard_limit_bytes.is_some_and(|limit| bytes as u64 > limit)
    {
        return Err(LearnError::Unsupported {
            message: "categorical workspace budget/cancellation",
        });
    }
    Ok(())
}

fn joint_size(cardinalities: &[usize], max_cells: usize) -> Result<usize, LearnError> {
    if cardinalities.is_empty() {
        return Err(LearnError::Shape { message: "finite joint needs a coordinate" });
    }
    cardinalities.iter().try_fold(1usize, |size, k| {
        size.checked_mul(*k).filter(|n| *n > 0 && *n <= max_cells).ok_or(LearnError::Shape {
            message: "finite joint exceeds declared cardinality budget",
        })
    })
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    #[test]
    fn misspecified_additive_hazards_remain_a_plugin_not_an_exact_law() {
        let mut columns = vec![vec![], vec![], vec![]];
        for a in 0..2 {
            for b in 0..2 {
                for i in 0..10 {
                    columns[0].push(a);
                    columns[1].push(b);
                    columns[2].push(usize::from(i < if a == b { 1 } else { 9 }));
                }
            }
        }
        let ctx = ExecutionContext::for_tests(2);
        let model = FiniteJoint::fit(
            &columns,
            &[2, 2, 2],
            LearnerSpec::parse("logistic").unwrap(),
            8,
            &ctx,
        )
        .unwrap();
        let probabilities = model.probabilities(&ctx).unwrap();
        assert!((probabilities.iter().sum::<f64>() - 1.).abs() < 1e-10);
        // True P(0,0,1)=.025, but an additive logistic chain cannot represent XOR.
        assert!((probabilities[1] - 0.025).abs() > 0.08);
        assert_eq!(model.counts()[1], 1);
    }
    #[test]
    fn categorical_joint_is_normalized_and_retains_sampling_zeros() {
        let columns = vec![vec![0, 0, 0, 1, 1, 1, 2, 2, 2], vec![0, 0, 1, 0, 1, 1, 0, 0, 1]];
        let model = FiniteJoint::fit(
            &columns,
            &[3, 2],
            LearnerSpec::parse("logistic").unwrap(),
            6,
            &ExecutionContext::for_tests(3),
        )
        .unwrap();
        let p = model.probabilities(&ExecutionContext::for_tests(3)).unwrap();
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-10);
        assert_eq!(model.counts(), &[2, 1, 1, 2, 2, 1]);
        let sparse = FiniteJoint::fit(
            &[vec![0, 0, 1, 1]],
            &[3],
            LearnerSpec::parse("logistic").unwrap(),
            3,
            &ExecutionContext::for_tests(3),
        )
        .unwrap();
        assert_eq!(sparse.counts(), &[2, 2, 0]);
        // The never-observed level gets probability exactly zero, and the observed mass is
        // the empirical split: the joint is a statement about the observed support.
        let sparse_p = sparse.probabilities(&ExecutionContext::for_tests(3)).unwrap();
        assert!((sparse_p[0] - 0.5).abs() < 1e-9 && (sparse_p[1] - 0.5).abs() < 1e-9);
        assert_eq!(sparse_p[2], 0.0);
        assert!(p.iter().all(|v| v.is_finite() && *v >= 0.0));
        assert!(
            FiniteJoint::fit(
                &columns,
                &[3, 2],
                LearnerSpec::Auto,
                5,
                &ExecutionContext::for_tests(3)
            )
            .is_err()
        );
    }
}
