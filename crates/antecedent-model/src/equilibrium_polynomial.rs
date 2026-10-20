// SPDX-License-Identifier: MIT OR Apache-2.0
//! Exact eligible-mechanism lowering into the shared bounded equation solver.
//! Addition retains separate monomials. Products use an error-free FMA split
//! so a rounded coefficient cannot silently certify roots of another equation.
use crate::{
    equation_system::{Interval, PolynomialEquation, PolynomialSystem, Monomial},
    mechanism_expr::MechanismExpression as E,
    nonlinear_equilibrium::{Activation, Clamp, EquilibriumError, EquilibriumModel},
};
use antecedent_core::ExecutionContext;
fn invalid(s: &str) -> EquilibriumError {
    EquilibriumError::Invalid(s.into())
}
fn constant(value: f64, n: usize) -> Monomial {
    Monomial { coefficient: value, powers: vec![0; n] }
}
fn exact_product(a: f64, b: f64) -> Result<Vec<f64>, EquilibriumError> {
    if a == 0.0 || b == 0.0 {
        return Ok(vec![0.0]);
    }
    let high = a * b;
    // With a normal product above 2^-969 the exact 106-bit product's tail
    // is representable as a binary64 value. More extreme products are refused.
    if !high.is_finite() || high.abs() < f64::MIN_POSITIVE * 9_007_199_254_740_992.0 {
        return Err(invalid("polynomial coefficient product outside error-free lowering range"));
    }
    let low = a.mul_add(b, -high);
    Ok(if low == 0.0 { vec![high] } else { vec![high, low] })
}
fn expand(
    expression: &E,
    parameters: &[f64],
    n: usize,
    operations: &mut usize,
    ctx: &ExecutionContext,
) -> Result<Vec<Monomial>, EquilibriumError> {
    if ctx.cancellation.is_cancelled() {
        return Err(EquilibriumError::Resource("cancelled".into()));
    }
    let result = match expression {
        E::Constant { value } => vec![constant(*value, n)],
        E::Parameter { parameter } => vec![constant(
            *parameters.get(*parameter).ok_or_else(|| invalid("polynomial parameter"))?,
            n,
        )],
        E::Variable { variable } => {
            let mut term = constant(1.0, n);
            *term.powers.get_mut(*variable).ok_or_else(|| invalid("polynomial variable"))? = 1;
            vec![term]
        }
        E::Add { terms } => {
            let mut result = Vec::new();
            for term in terms {
                result.extend(expand(term, parameters, n, operations, ctx)?);
                if result.len() > 256 {
                    return Err(invalid("polynomial expansion term limit"));
                }
            }
            result
        }
        E::Multiply { factors } => {
            let mut result = vec![constant(1.0, n)];
            for factor in factors {
                let other = expand(factor, parameters, n, operations, ctx)?;
                let mut product = Vec::new();
                for a in &result {
                    for b in &other {
                        *operations += 1;
                        if *operations > 1_000_000 {
                            return Err(EquilibriumError::Resource(
                                "polynomial lowering operation budget".into(),
                            ));
                        }
                        let powers = a
                            .powers
                            .iter()
                            .zip(&b.powers)
                            .map(|(a, b)| {
                                a.checked_add(*b)
                                    .filter(|&p| p <= 8)
                                    .ok_or_else(|| invalid("polynomial per-variable degree limit"))
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        for coefficient in exact_product(a.coefficient, b.coefficient)? {
                            product.push(Monomial { coefficient, powers: powers.clone() });
                            if product.len() > 256 {
                                return Err(invalid("polynomial expansion term limit"));
                            }
                        }
                    }
                }
                result = product;
            }
            result
        }
        E::Tanh { .. } | E::Sin { .. } | E::Softplus { .. } => {
            return Err(invalid("mechanism is not eligible for exact polynomial root isolation"));
        }
    };
    Ok(result)
}
/// Lower one explicit joint-noise realization and hard intervention to named
/// polynomial residuals over a finite box. Eligible polynomial compositions and
/// whole-domain affine `ReLU` branches retain exact represented coefficients.
/// Other mechanisms, excessive expansion and coefficient underflow are refused.
pub fn compile_polynomial_equilibrium(
    model: &EquilibriumModel,
    domain: &[Interval],
    noise: &[f64],
    clamps: &[Clamp],
    ctx: &ExecutionContext,
) -> Result<PolynomialSystem, EquilibriumError> {
    model.validate()?;
    let n = model.order();
    if n > 8
        || domain.len() != n
        || noise.len() != n
        || noise.iter().any(|v| !v.is_finite())
        || domain.iter().any(|b| !b.lower.is_finite() || !b.upper.is_finite() || b.lower >= b.upper)
    {
        return Err(invalid("equilibrium polynomial domain/noise (order<=8)"));
    }
    let mut fixed = vec![None; n];
    for c in clamps {
        if c.variable >= n || !c.value.is_finite() || fixed[c.variable].is_some() {
            return Err(invalid("equilibrium polynomial clamp"));
        }
        fixed[c.variable] = Some(c.value);
    }
    let _lease = ctx
        .acquire_admission(0, 1_048_576)
        .map_err(|e| EquilibriumError::Resource(e.to_string()))?;
    let mut equations = Vec::new();
    let mut operations = 0;
    for (i, equation) in model.equations.iter().enumerate() {
        if ctx.cancellation.is_cancelled() {
            return Err(EquilibriumError::Resource("cancelled".into()));
        }
        let mut variable = constant(1.0, n);
        variable.powers[i] = 1;
        let mut terms = vec![variable];
        if let Some(value) = fixed[i] {
            terms.push(constant(-value, n));
        } else {
            terms.push(constant(-equation.intercept, n));
            terms.push(constant(-noise[i], n));
            let mechanism = if let Some(expression) = &equation.expression {
                expand(expression, &equation.parameters, n, &mut operations, ctx)?
            } else {
                let mut weights = equation.weights.clone();
                match equation.activation {
                    Activation::Identity => {}
                    Activation::Relu => {
                        let range = domain
                            .iter()
                            .zip(&weights)
                            .fold(Interval { lower: 0.0, upper: 0.0 }, |sum, (bound, &weight)| {
                                sum.add_outward(bound.scale_outward(weight))
                            });
                        if range.upper <= 0.0 {
                            weights.fill(0.0);
                        } else if range.lower < 0.0 {
                            return Err(invalid(
                                "ReLU branch crosses a kink in the declared isolation domain",
                            ));
                        }
                    }
                    Activation::Tanh | Activation::Sin => {
                        if weights.iter().any(|&w| w != 0.0) {
                            return Err(invalid(
                                "non-polynomial activation is not eligible for polynomial root isolation",
                            ));
                        }
                    }
                }
                weights
                    .iter()
                    .enumerate()
                    .map(|(j, &weight)| {
                        let mut term = constant(weight, n);
                        term.powers[j] = 1;
                        term
                    })
                    .collect()
            };
            terms.extend(mechanism.into_iter().map(|mut term| {
                term.coefficient = -term.coefficient;
                term
            }));
        }
        if terms.len() > 256 {
            return Err(invalid("polynomial residual term limit"));
        }
        equations.push(PolynomialEquation { id: equation.id.clone(), terms });
    }
    let system = PolynomialSystem {
        variables: (0..n).map(|i| format!("variable:{i}")).collect(),
        equations,
        domain: domain.to_vec(),
    };
    system.validate()?;
    Ok(system)
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::nonlinear_equilibrium::Equation;
    #[test]
    fn polynomial_mechanism_interoperability_preserves_multiple_roots_and_clamps() {
        let model = EquilibriumModel {
            equations: vec![Equation {
                id: "cubic".into(),
                intercept: 0.0,
                weights: vec![0.0],
                activation: Activation::Identity,
                expression: Some(E::Add {
                    terms: vec![
                        E::Multiply {
                            factors: vec![E::Constant { value: 2.0 }, E::Variable { variable: 0 }],
                        },
                        E::Multiply {
                            factors: vec![
                                E::Constant { value: -1.0 },
                                E::Variable { variable: 0 },
                                E::Variable { variable: 0 },
                                E::Variable { variable: 0 },
                            ],
                        },
                    ],
                }),
                parameters: vec![],
            }],
            noise_covariance: vec![1.0],
        };
        let ctx = ExecutionContext::for_tests(0);
        let domain = [Interval { lower: -2.0, upper: 2.0 }];
        let system = compile_polynomial_equilibrium(&model, &domain, &[0.0], &[], &ctx).unwrap();
        let roots =
            system.isolate(crate::equation_system::IsolationPolicy::default(), &ctx).unwrap();
        assert_eq!(roots.roots.len(), 3);
        assert!(roots.unresolved.is_empty());
        let clamped = compile_polynomial_equilibrium(
            &model,
            &domain,
            &[0.7],
            &[Clamp { variable: 0, value: 0.5 }],
            &ctx,
        )
        .unwrap()
        .isolate(crate::equation_system::IsolationPolicy::default(), &ctx)
        .unwrap();
        assert_eq!(clamped.roots.len(), 1);
        assert!(clamped.roots[0][0].lower <= 0.5 && clamped.roots[0][0].upper >= 0.5);
        let coefficients = exact_product(0.3, 0.7).unwrap();
        assert_eq!(coefficients.len(), 2);
    }
}
