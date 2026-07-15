//! Fuzz expression arena intern / estimand method parsing.
#![no_main]

use std::str::FromStr;

use causal_core::VariableId;
use causal_expr::{CausalExprArena, DomainRef, EstimandMethod, ExprNode};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let mut arena = CausalExprArena::new();
    for chunk in data.chunks(4).take(64) {
        if chunk.is_empty() {
            continue;
        }
        let a = VariableId::from_raw(u32::from(chunk.get(0).copied().unwrap_or(0) % 8));
        let b = VariableId::from_raw(u32::from(chunk.get(1).copied().unwrap_or(1) % 8));
        let vars = arena.intern_var_set([a, b]);
        let empty = arena.empty_var_set();
        let empty_do = arena.empty_intervention_set();
        let domain = if chunk.get(2).copied().unwrap_or(0) & 1 == 0 {
            DomainRef::Observational
        } else {
            DomainRef::Interventional
        };
        let dist = arena.intern(ExprNode::Distribution {
            variables: vars,
            conditioned_on: empty,
            intervention: empty_do,
            domain,
        });
        let tag = chunk.get(3).copied().unwrap_or(0) % 4;
        match tag {
            0 => {
                let _ = arena.intern(ExprNode::SumOut { variables: empty, expr: dist });
            }
            1 => {
                let _ = arena.intern(ExprNode::IntegralOut { variables: empty, expr: dist });
            }
            2 => {
                let vars_a = arena.intern_var_set([a]);
                let other = arena.intern(ExprNode::Distribution {
                    variables: vars_a,
                    conditioned_on: empty,
                    intervention: empty_do,
                    domain: DomainRef::Observational,
                });
                let _ = arena.intern(ExprNode::Ratio {
                    numerator: dist,
                    denominator: other,
                });
            }
            _ => {
                let list = arena.intern_list([dist]);
                let _ = arena.intern(ExprNode::Product(list));
            }
        }
    }
    if let Ok(s) = std::str::from_utf8(data) {
        let slice = if s.len() > 256 { &s[..256] } else { s };
        let _ = EstimandMethod::from_str(slice.trim());
    }
});
