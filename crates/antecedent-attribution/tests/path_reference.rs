//! Directed-path products of linear structural models against exact rational references.
//!
//! `oracle_data/path_reference.py` enumerates every directed path and multiplies its edge
//! coefficients in rational arithmetic, and separately inverts `I - B` exactly; the two agree
//! before a case is written.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_attribution::path_decompose;
use antecedent_core::{ExecutionContext, VariableId};
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_model::{CompiledCausalModel, CompiledMechanismStore, MechanismSlot};
use serde::Deserialize;

/// Directed-path products against exact rational path enumeration and matrix inversion,
/// with edges inserted in an order unrelated to node order (the coefficient vector of a
/// child must follow that child's parent gather, not insertion order).
#[test]
fn path_products_match_exact_total_effects() {
    #[derive(Deserialize)]
    struct Fixture {
        cases: Vec<Case>,
    }
    #[derive(Deserialize)]
    struct Case {
        name: String,
        n: u32,
        edges: Vec<(u32, u32, f64)>,
        source: u32,
        outcome: u32,
        paths: Vec<Expected>,
        total: f64,
    }
    #[derive(Deserialize)]
    struct Expected {
        path: Vec<u32>,
        product: f64,
    }
    let fixture: Fixture =
        serde_json::from_str(include_str!("oracle_data/path_reference.json")).unwrap();
    for case in &fixture.cases {
        let mut g = Dag::with_variables(case.n);
        for &(p, c, _) in &case.edges {
            g.insert_directed(DenseNodeId::from_raw(p), DenseNodeId::from_raw(c)).unwrap();
        }
        let compiled = CompiledCausalModel::compile(g).unwrap();
        let mut slots =
            vec![MechanismSlot::Constant { value: 0.0 }; case.n as usize];
        for gather in compiled.parent_gathers.iter() {
            let coeffs: Vec<f64> = gather
                .parents
                .iter()
                .map(|p| {
                    case.edges
                        .iter()
                        .find(|e| e.0 == p.as_usize() as u32 && e.1 == gather.child.as_usize() as u32)
                        .map(|e| e.2)
                        .unwrap()
                })
                .collect();
            slots[gather.child.as_usize()] = MechanismSlot::LinearGaussian {
                intercept: 0.0,
                coeffs: Arc::from(coeffs),
                sigma: 1.0,
            };
        }
        let model = compiled.with_mechanisms(CompiledMechanismStore { slots: Arc::from(slots) });
        let result = path_decompose(
            &model,
            &[VariableId::from_raw(case.source)],
            VariableId::from_raw(case.outcome),
            1000,
            16,
            &ExecutionContext::for_tests(1),
        )
        .unwrap();
        assert_eq!(result.path_breakdown.len(), case.paths.len(), "{}", case.name);
        for want in &case.paths {
            let got = result
                .path_breakdown
                .iter()
                .find(|p| {
                    let vars: Vec<u32> = p.path.iter().map(|v| v.raw()).collect();
                    vars == want.path
                })
                .unwrap_or_else(|| panic!("{}: missing path {:?}", case.name, want.path));
            assert!(
                (got.contribution - want.product).abs() <= 1e-12 * want.product.abs().max(1.0),
                "{} {:?}: {} vs {}",
                case.name,
                want.path,
                got.contribution,
                want.product
            );
        }
        // Signed shares keep cancelling branches: the diamond's total is exactly zero.
        assert!(
            (result.total_change - case.total).abs() <= 1e-12 * case.total.abs().max(1.0),
            "{}: total {} vs {}",
            case.name,
            result.total_change,
            case.total
        );
    }
}
