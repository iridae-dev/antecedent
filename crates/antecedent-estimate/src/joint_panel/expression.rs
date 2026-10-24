//! Bind a fitted response to scheduled controls without replacing its mechanism.
// SPDX-License-Identifier: MIT OR Apache-2.0
use super::{PanelData, PanelDraw, PanelError, PanelProgram, check_cancelled, invalid, likelihood};
use antecedent_core::{ArithOp, ExecutionContext, ValueExpr, VariableId};
use std::sync::Arc;

const MAX_NODES: usize = 65_536;

/// Binding of an actual channel/period to a distinct scalar expression variable.
#[derive(Clone, Debug)]
pub struct PanelExpressionControl {
    /// Stable channel ID from the fitted source.
    pub channel: String,
    /// Exact zero-based program period.
    pub period: usize,
    /// Variable used by the bounded numerical optimizer.
    pub variable: VariableId,
}

/// A scalar response expression derived from the actual fitted mechanism.
#[derive(Clone, Debug)]
pub struct CompiledPanelExpression {
    /// Population-averaged response with the declared time weights.
    pub expression: ValueExpr,
    retention: Arc<antecedent_core::execution::AdmissionLease>,
}
impl CompiledPanelExpression {
    /// Reserved expression storage, including one downstream optimizer clone.
    #[must_use]
    pub fn admitted_bytes(&self) -> u64 {
        self.retention.remainder().bytes()
    }
}

struct Builder {
    remaining: usize,
}
impl Builder {
    fn node(&mut self, value: ValueExpr) -> Result<ValueExpr, PanelError> {
        self.remaining = self.remaining.checked_sub(1).ok_or_else(|| {
            PanelError::Resource(
                "panel objective exceeds 65536 expression nodes; use bounded best-found search"
                    .into(),
            )
        })?;
        Ok(value)
    }
    fn constant(&mut self, value: f64) -> Result<ValueExpr, PanelError> {
        if !value.is_finite() {
            return Err(invalid("nonfinite compiled response constant"));
        }
        self.node(ValueExpr::constant(value))
    }
    fn binary(
        &mut self,
        op: ArithOp,
        lhs: ValueExpr,
        rhs: ValueExpr,
    ) -> Result<ValueExpr, PanelError> {
        self.node(ValueExpr::Arith { op, lhs: Box::new(lhs), rhs: Box::new(rhs) })
    }
    fn copy(&mut self, expression: &ValueExpr) -> Result<ValueExpr, PanelError> {
        match expression {
            ValueExpr::Const(n) => self.constant(n.get()),
            ValueExpr::Var(v) => self.node(ValueExpr::var(*v)),
            ValueExpr::Neg(value) => {
                let value = self.copy(value)?;
                self.node(ValueExpr::Neg(Box::new(value)))
            }
            ValueExpr::Arith { op, lhs, rhs } => {
                let lhs = self.copy(lhs)?;
                let rhs = self.copy(rhs)?;
                self.binary(*op, lhs, rhs)
            }
            _ => Err(invalid(
                "response feature has no certified scalar expression; use bounded best-found search",
            )),
        }
    }
    fn substitute(
        &mut self,
        expression: &ValueExpr,
        channels: &[String],
        saturated: &[ValueExpr],
        depth: usize,
    ) -> Result<ValueExpr, PanelError> {
        if depth > 64 {
            return Err(invalid("response expression depth exceeds compiler limit"));
        }
        match expression {
            ValueExpr::Var(v) => {
                let index = channels
                    .iter()
                    .position(|id| *id == v.raw().to_string())
                    .ok_or_else(|| invalid("unbound response feature variable"))?;
                self.copy(&saturated[index])
            }
            ValueExpr::Const(n) => self.constant(n.get()),
            ValueExpr::Neg(value) => {
                let value = self.substitute(value, channels, saturated, depth + 1)?;
                self.node(ValueExpr::Neg(Box::new(value)))
            }
            ValueExpr::Arith { op, lhs, rhs } => {
                let lhs = self.substitute(lhs, channels, saturated, depth + 1)?;
                let rhs = self.substitute(rhs, channels, saturated, depth + 1)?;
                self.binary(*op, lhs, rhs)
            }
            _ => Err(invalid(
                "response feature has no certified scalar expression; use bounded best-found search",
            )),
        }
    }
    /// `expression^n` for a positive integer `n` by repeated multiplication.
    fn power(&mut self, expression: &ValueExpr, n: i32) -> Result<ValueExpr, PanelError> {
        let mut result = self.copy(expression)?;
        for _ in 1..n {
            let factor = self.copy(expression)?;
            result = self.binary(ArithOp::Mul, result, factor)?;
        }
        Ok(result)
    }
    fn sum(&mut self, mut values: Vec<ValueExpr>) -> Result<ValueExpr, PanelError> {
        if values.is_empty() {
            return self.constant(0.0);
        }
        while values.len() > 1 {
            let mut next = Vec::with_capacity(values.len().div_ceil(2));
            let mut items = values.into_iter();
            while let Some(lhs) = items.next() {
                next.push(if let Some(rhs) = items.next() {
                    self.binary(ArithOp::Add, lhs, rhs)?
                } else {
                    lhs
                });
            }
            values = next;
        }
        values.pop().ok_or_else(|| invalid("missing response expression"))
    }
}

/// The expression language has no power operator, so only an integer Hill shape has a
/// certified scalar expression; any other fitted shape remains a typed refusal.
fn integer_shape(shape: f64) -> Result<i32, PanelError> {
    (1..=8).find(|n| (shape - f64::from(*n)).abs() <= 1e-12).ok_or_else(|| {
        invalid(
            "a non-integer or large Hill shape has no certified scalar expression; use bounded best-found search",
        )
    })
}

/// Compile a real fitted mean response into a scalar expression over exact
/// scheduled controls. Uncontrolled periods, initial histories, population,
/// covariates, interactions and all fitted parameters retain their actual values.
/// A temporal path contains one mechanism per response period; a stationary path
/// contains one mechanism. This grants no causal or statistical promotion.
///
/// # Errors
/// Invalid programs/parameters/bindings, unsupported expressions, resources or cancellation.
#[expect(
    clippy::too_many_lines,
    reason = "binds every response component to the same expression and admission"
)]
pub fn compile_panel_expression(
    data: &PanelData,
    path: &[PanelDraw],
    program: &PanelProgram,
    controls: &[PanelExpressionControl],
    time_weights: &[f64],
    ctx: &ExecutionContext,
) -> Result<CompiledPanelExpression, PanelError> {
    check_cancelled(ctx)?;
    data.validate()?;
    super::validate_panel_programs(data, std::slice::from_ref(program))?;
    if time_weights.len() != program.horizon()
        || time_weights.iter().any(|w| !w.is_finite())
        || (path.len() != 1 && path.len() != program.horizon())
    {
        return Err(invalid("compiled expression time/path axes differ from the actual program"));
    }
    let mut variables = std::collections::HashSet::new();
    let mut schedules = std::collections::HashSet::new();
    for control in controls {
        if !variables.insert(control.variable)
            || !schedules.insert((&control.channel, control.period))
            || control.period >= program.horizon()
            || !data.channels.iter().any(|c| c.id == control.channel)
        {
            return Err(invalid("invalid or duplicate compiled scheduled control"));
        }
    }
    // Bound construction before allocating expanded feature trees. The lease also
    // covers a downstream numerical optimizer's retained expression clone.
    let quote = (MAX_NODES * std::mem::size_of::<ValueExpr>() * 3) as u64;
    super::check_memory(ctx, quote)?;
    let retention =
        Arc::new(ctx.acquire_admission(0, quote).map_err(|e| PanelError::Resource(e.to_string()))?);
    let compiled = likelihood::CompiledPanel::new(data)?;
    for draw in path {
        likelihood::validate_parameters(&compiled, &draw.parameters)?;
        if draw.group_intercepts.len() != data.groups.len()
            || draw.group_intercepts.iter().any(|v| !v.is_finite())
        {
            return Err(invalid("compiled expression group-effect axis differs"));
        }
    }
    let mut builder = Builder { remaining: MAX_NODES };
    let mut response_terms = Vec::new();
    let channel_ids = data.channels.iter().map(|c| c.id.clone()).collect::<Vec<_>>();
    for (group_index, group) in data.groups.iter().enumerate() {
        let scenario = program
            .scenarios
            .iter()
            .find(|s| s.group == group.id)
            .ok_or_else(|| invalid("compiled response group absent"))?;
        for (time, weight) in time_weights.iter().enumerate().filter(|(_, w)| **w != 0.0) {
            check_cancelled(ctx)?;
            let draw = &path[if path.len() == 1 { 0 } else { time }];
            let p = &draw.parameters;
            let mut saturated = Vec::with_capacity(data.channels.len());
            let half_saturation = p.half_saturation_of(data.family.class_of(group_index));
            for (channel_index, channel) in p.channels.iter().enumerate() {
                let values = &program
                    .channels
                    .iter()
                    .find(|c| c.id == channel.id)
                    .ok_or_else(|| invalid("compiled response channel absent"))?
                    .values;
                let history = scenario
                    .history
                    .iter()
                    .find(|h| h.id == channel.id)
                    .ok_or_else(|| invalid("compiled response history absent"))?
                    .values;
                let mut exposure_terms = Vec::new();
                for (lag, lag_weight) in channel.lag.iter().enumerate().filter(|(_, w)| **w != 0.0)
                {
                    let value = if let Some(period) = time.checked_sub(lag) {
                        if let Some(control) =
                            controls.iter().find(|c| c.channel == channel.id && c.period == period)
                        {
                            builder.node(ValueExpr::var(control.variable))?
                        } else {
                            builder.constant(values[period])?
                        }
                    } else {
                        builder.constant(history[2 + time - lag])?
                    };
                    let coefficient = builder.constant(*lag_weight)?;
                    exposure_terms.push(builder.binary(ArithOp::Mul, coefficient, value)?);
                }
                let exposure = builder.sum(exposure_terms)?;
                let power = match p.family.hill_shape.get(channel_index) {
                    None => 1,
                    Some(shape) => integer_shape(*shape)?,
                };
                // x^s / (x^s + k^s); s = 1 is exactly the Michaelis-Menten form.
                let numerator =
                    if power == 1 { exposure } else { builder.power(&exposure, power)? };
                let copy = builder.copy(&numerator)?;
                let k = builder.constant(half_saturation.powi(power))?;
                let denominator = builder.binary(ArithOp::Add, copy, k)?;
                saturated.push(builder.binary(ArithOp::Div, numerator, denominator)?);
            }
            let mut terms = vec![builder.constant(
                p.intercept
                    + draw.group_intercepts[group_index]
                    + p.confounder * scenario.confounder[time]
                    + p.season_effects[scenario.season[time]],
            )?];
            for (channel, expression) in p.channels.iter().zip(&saturated) {
                let coefficient = builder.constant(channel.coefficient)?;
                let expression = builder.copy(expression)?;
                terms.push(builder.binary(ArithOp::Mul, coefficient, expression)?);
            }
            for (feature, coefficient) in data.interactions.iter().zip(&p.interactions) {
                let expression = builder.substitute(
                    &feature.expression.to_domain(),
                    &channel_ids,
                    &saturated,
                    0,
                )?;
                let coefficient = builder.constant(coefficient.coefficient)?;
                terms.push(builder.binary(ArithOp::Mul, coefficient, expression)?);
            }
            let mean = builder.sum(terms)?;
            let coefficient = builder.constant(weight / data.groups.len() as f64)?;
            response_terms.push(builder.binary(ArithOp::Mul, coefficient, mean)?);
        }
    }
    let expression = builder.sum(response_terms)?;
    Ok(CompiledPanelExpression { expression, retention })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn value(expression: &ValueExpr, x: f64) -> f64 {
        match expression {
            ValueExpr::Const(n) => n.get(),
            ValueExpr::Var(_) => x,
            ValueExpr::Neg(v) => -value(v, x),
            ValueExpr::Arith { op, lhs, rhs } => op.apply(value(lhs, x), value(rhs, x)),
            _ => panic!("unsupported test expression"),
        }
    }
    #[test]
    fn derived_expression_matches_the_real_frozen_response_across_scheduled_controls() {
        let (data, draw, _) = super::super::tests::fixture();
        let mut program = super::super::tests::programs(&data).remove(3);
        let ctx = ExecutionContext::for_tests(31);
        let mut weights = vec![0.0; 24];
        weights[2] = 1.0;
        let compiled = compile_panel_expression(
            &data,
            std::slice::from_ref(&draw),
            &program,
            &[PanelExpressionControl {
                channel: "0".into(),
                period: 1,
                variable: VariableId::from_raw(99),
            }],
            &weights,
            &ctx,
        )
        .unwrap();
        for i in 0..32 {
            let x = f64::from(i) / 8.0;
            program.channels[0].values[1] = x;
            let real = super::super::evaluate_panel_programs(
                &data,
                &draw,
                std::slice::from_ref(&program),
                &ctx,
            )
            .unwrap();
            assert!((value(&compiled.expression, x) - real[0].means[2]).abs() < 1e-10);
        }
    }
}
