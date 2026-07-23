//! Fuzz DAG insert / cycle rejection / ancestry queries.
#![no_main]

use antecedent_graph::{DSeparationWorkspace, Dag, DenseNodeId};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }
    let n = (data[0] % 12).max(1);
    let mut dag = Dag::with_variables(u32::from(n));
    for w in data[1..].chunks(2) {
        if w.len() < 2 {
            break;
        }
        let from = DenseNodeId::from_raw(u32::from(w[0] % n));
        let to = DenseNodeId::from_raw(u32::from(w[1] % n));
        let _ = dag.insert_directed(from, to);
    }
    if n >= 2 {
        let mut ws = DSeparationWorkspace::default();
        let _ = dag.is_d_separated(
            DenseNodeId::from_raw(0),
            DenseNodeId::from_raw(1),
            &[],
            &mut ws,
        );
    }
});
