//! Versioned numerical certificates are replayed, never trusted as assertions.
// SPDX-License-Identifier: MIT OR Apache-2.0
use super::{
    ArithOp, CertificateOptions, CertificateStop, CertifiedControl, CertifiedRegion,
    CertifiedRestriction, ConcaveDualBound, Deserialize, ExecutionContext,
    GlobalProgramCertificate, MonotoneReduction, ObjectiveInterval, PreparedCertifiedSearch,
    ProgramSearchError, SemanticDigest, SeparableBound, Serialize, ValueExpr, VariableId,
    cancelled, invalid,
};
use antecedent_io_core::convert::{from_cbor, to_cbor};

const MAX_BYTES: usize = 32 * 1024 * 1024;
#[derive(Serialize, Deserialize)]
enum Instruction {
    Constant(f64),
    Variable(u32),
    Negate,
    Binary(u8),
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Document {
    version: u32,
    expression: Vec<Instruction>,
    controls: Vec<(u32, ObjectiveInterval)>,
    restrictions: Vec<CertifiedRestriction>,
    initial: Vec<f64>,
    options: CertificateOptions,
    proof: Proof,
}
#[derive(Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct Proof {
    identity: String,
    controls: Vec<f64>,
    incumbent: ObjectiveInterval,
    global_upper: f64,
    gap: f64,
    stop: CertificateStop,
    frontier: Vec<CertifiedRegion>,
    splits: usize,
    infeasible_regions: usize,
    dominated_regions: usize,
    monotone_reductions: Vec<MonotoneReduction>,
    dual_bound: Option<ConcaveDualBound>,
    separable_bound: Option<SeparableBound>,
}
impl From<&GlobalProgramCertificate> for Proof {
    fn from(c: &GlobalProgramCertificate) -> Self {
        Self {
            identity: c.identity.to_hex(),
            controls: c.controls.clone(),
            incumbent: c.incumbent,
            global_upper: c.global_upper,
            gap: c.gap,
            stop: c.stop,
            frontier: c.frontier.clone(),
            splits: c.splits,
            infeasible_regions: c.infeasible_regions,
            dominated_regions: c.dominated_regions,
            monotone_reductions: c.monotone_reductions.clone(),
            dual_bound: c.dual_bound.clone(),
            separable_bound: c.separable_bound.clone(),
        }
    }
}
fn encode(e: &ValueExpr, out: &mut Vec<Instruction>) -> Result<(), ProgramSearchError> {
    match e {
        ValueExpr::Const(n) => out.push(Instruction::Constant(n.get())),
        ValueExpr::Var(v) => out.push(Instruction::Variable(v.raw())),
        ValueExpr::Neg(v) => {
            encode(v, out)?;
            out.push(Instruction::Negate);
        }
        ValueExpr::Arith { op, lhs, rhs } => {
            encode(lhs, out)?;
            encode(rhs, out)?;
            let code = match op {
                ArithOp::Add => 0,
                ArithOp::Sub => 1,
                ArithOp::Mul => 2,
                ArithOp::Div => 3,
                ArithOp::Min => 4,
                ArithOp::Max => 5,
                _ => return Err(invalid("unsupported certificate opcode")),
            };
            out.push(Instruction::Binary(code));
        }
        _ => return Err(invalid("unsupported certificate expression")),
    }
    Ok(())
}
fn decode(instructions: Vec<Instruction>) -> Result<ValueExpr, ProgramSearchError> {
    if instructions.len() > 65_536 {
        return Err(invalid("certificate expression node limit"));
    }
    let mut stack: Vec<(ValueExpr, usize)> = Vec::new();
    for instruction in instructions {
        let (expr, depth) = match instruction {
            Instruction::Constant(x) if x.is_finite() => (ValueExpr::constant(x), 0),
            Instruction::Variable(v) => (ValueExpr::var(VariableId::from_raw(v)), 0),
            Instruction::Negate => {
                let (v, depth) =
                    stack.pop().ok_or_else(|| invalid("certificate stack underflow"))?;
                (ValueExpr::Neg(Box::new(v)), depth + 1)
            }
            Instruction::Binary(code) => {
                let (rhs, rd) =
                    stack.pop().ok_or_else(|| invalid("certificate stack underflow"))?;
                let (lhs, ld) =
                    stack.pop().ok_or_else(|| invalid("certificate stack underflow"))?;
                let op = match code {
                    0 => ArithOp::Add,
                    1 => ArithOp::Sub,
                    2 => ArithOp::Mul,
                    3 => ArithOp::Div,
                    4 => ArithOp::Min,
                    5 => ArithOp::Max,
                    _ => return Err(invalid("unknown certificate opcode")),
                };
                (ValueExpr::Arith { op, lhs: Box::new(lhs), rhs: Box::new(rhs) }, ld.max(rd) + 1)
            }
            Instruction::Constant(_) => return Err(invalid("nonfinite certificate constant")),
        };
        if depth > 64 {
            return Err(invalid("certificate expression depth limit"));
        }
        stack.push((expr, depth));
    }
    if stack.len() != 1 {
        return Err(invalid("incomplete certificate expression"));
    }
    Ok(stack.pop().ok_or_else(|| invalid("missing certificate expression"))?.0)
}
impl PreparedCertifiedSearch {
    /// Exact objective/domain identity, without granting scientific permission.
    #[must_use]
    pub fn identity(&self) -> SemanticDigest {
        self.identity
    }

    /// Verify that an independent compiler produced this same objective, retaining
    /// the artifact's declared controls and operational restrictions.
    /// # Errors
    /// Unsupported expression or any changed numerical objective.
    pub fn verify_expression(&self, expression: ValueExpr) -> Result<(), ProgramSearchError> {
        let derived = Self::compile(expression, self.controls.clone(), self.restrictions.clone())?;
        if derived.identity != self.identity {
            return Err(invalid("certificate compiler expression changed"));
        }
        Ok(())
    }

    /// Execute and export the complete deterministic numerical proof. Re-execution
    /// prevents pairing another domain's certificate with this expression.
    ///
    /// # Errors
    /// Execution, admission, size or encoding failure.
    pub fn export_replay(
        &self,
        initial: Vec<f64>,
        options: CertificateOptions,
        ctx: &ExecutionContext,
    ) -> Result<Vec<u8>, ProgramSearchError> {
        let certificate = self.execute(initial.clone(), options, ctx)?;
        let mut expression = Vec::new();
        encode(&self.expression, &mut expression)?;
        let document = Document {
            version: 1,
            expression,
            controls: self.controls.iter().map(|c| (c.variable.raw(), c.bounds)).collect(),
            restrictions: self.restrictions.clone(),
            initial,
            options,
            proof: Proof::from(&certificate),
        };
        let bytes = to_cbor(&document).map_err(|e| invalid(e.to_string()))?;
        if bytes.len() > MAX_BYTES {
            return Err(invalid("certificate artifact size limit"));
        }
        Ok(bytes)
    }
    /// Independently reconstruct the objective and repeat all region/shape/dual
    /// calculations. Current consumption supplies an expected domain identity;
    /// historical consumption may omit it, retaining only numerical authority.
    ///
    /// # Errors
    /// Changed domain, forged bounds, malformed document, cancellation or resources.
    pub fn replay(
        bytes: &[u8],
        expected: Option<SemanticDigest>,
        ctx: &ExecutionContext,
    ) -> Result<(Self, GlobalProgramCertificate), ProgramSearchError> {
        cancelled(ctx)?;
        if bytes.len() > MAX_BYTES {
            return Err(invalid("certificate artifact size limit"));
        }
        let _input = ctx
            .acquire_admission(0, (bytes.len() as u64).saturating_mul(16))
            .map_err(|e| ProgramSearchError::Resource(e.to_string()))?;
        let d: Document = from_cbor(bytes).map_err(|e| invalid(e.to_string()))?;
        if d.version != 1 {
            return Err(invalid("unsupported certificate artifact version"));
        }
        let plan = Self::compile(
            decode(d.expression)?,
            d.controls
                .into_iter()
                .map(|(v, bounds)| CertifiedControl { variable: VariableId::from_raw(v), bounds })
                .collect(),
            d.restrictions,
        )?;
        if expected.is_some_and(|identity| identity != plan.identity) {
            return Err(invalid("certificate objective/domain changed"));
        }
        let result = plan.execute(d.initial, d.options, ctx)?;
        if Proof::from(&result) != d.proof {
            return Err(invalid("certificate replay proof mismatch"));
        }
        Ok((plan, result))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn replay_recomputes_bounds_and_refuses_tampering_and_changed_domains() {
        let x = ValueExpr::var(VariableId::from_raw(7));
        let p = PreparedCertifiedSearch::compile(
            x.clone().mul(ValueExpr::constant(1.0).sub(x)),
            vec![CertifiedControl {
                variable: VariableId::from_raw(7),
                bounds: ObjectiveInterval { lower: 0.0, upper: 1.0 },
            }],
            vec![],
        )
        .unwrap();
        let ctx = ExecutionContext::for_tests(1);
        let bytes = p
            .export_replay(
                vec![0.0],
                CertificateOptions { absolute_gap: 1e-3, max_splits: 50 },
                &ctx,
            )
            .unwrap();
        let (_, c) = PreparedCertifiedSearch::replay(&bytes, Some(p.identity()), &ctx).unwrap();
        assert!(c.global_upper >= 0.25);
        assert!(
            PreparedCertifiedSearch::replay(
                &bytes,
                Some(SemanticDigest::from_bytes([9; 32])),
                &ctx
            )
            .is_err()
        );
        let mut forged: Document = from_cbor(&bytes).unwrap();
        forged.proof.global_upper = 0.1;
        assert!(PreparedCertifiedSearch::replay(&to_cbor(&forged).unwrap(), None, &ctx).is_err());
        let mut trailing = bytes;
        trailing.push(0);
        assert!(PreparedCertifiedSearch::replay(&trailing, None, &ctx).is_err());
    }
}
