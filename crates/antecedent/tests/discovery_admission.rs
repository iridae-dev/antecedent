//! Admitting a finished discovery is the same graph `accept_discovered` returns.
//!
//! A second admission reads the execution that already ran. It does not compile
//! another search. The downstream program identity is the one that graph carries,
//! and a different table is refused without being stored on the graph.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::discovery::StaticDiscoverParams;
use antecedent::discovery_defaults::resolve_ci;
use antecedent::{CompiledExecution, DiscoverySpec, StructureSource, Study, accept_discovered};
use antecedent_core::{
    AttributeQuery, AttributeTarget, CausalQuery, CausalRng, ExecutionContext, SemanticDigest,
    VariableId,
};
use antecedent_data::TabularData;
use antecedent_discovery::BackgroundKnowledge;

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(0x5EED_0017)
}

fn on_large_stack(name: &str, body: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .name(name.into())
        .stack_size(32 * 1024 * 1024)
        .spawn(body)
        .unwrap()
        .join()
        .unwrap();
}

/// `z -> t -> y`, `z -> y` with uniform noise, so LiNGAM has a DAG to accept.
fn chain_table(n: usize, seed: u64) -> TabularData {
    let mut rng = CausalRng::from_seed(seed);
    let mut uniform = || 2.0 * rng.next_f64() - 1.0;
    let (mut z, mut t, mut y) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..n {
        let zi = uniform();
        let ti = 0.8 * zi + uniform();
        let yi = 1.5 * ti + 0.6 * zi + uniform();
        z.push(zi);
        t.push(ti);
        y.push(yi);
    }
    TabularData::from_f64_columns([("z", z.as_slice()), ("t", t.as_slice()), ("y", y.as_slice())])
        .unwrap()
}

fn static_params() -> StaticDiscoverParams {
    StaticDiscoverParams {
        alpha: 0.05,
        max_cond_size: 2,
        fdr: None,
        ci: resolve_ci("parcorr", None).unwrap(),
        screen_pc: false,
        max_subset: None,
        knowledge: BackgroundKnowledge::default(),
    }
}

fn paths() -> CausalQuery {
    CausalQuery::Attribute(AttributeQuery::new(AttributeTarget::Paths {
        sources: Arc::from([v(1)]),
        outcome: v(2),
        max_paths: 16,
        max_len: 4,
        through: Arc::from([]),
    }))
}

fn program(study: &Study, ctx: &ExecutionContext) -> SemanticDigest {
    study.prepare(ctx).unwrap().contract().unwrap().identities.program.expect("a prepared program")
}

#[test]
fn a_finished_discovery_admits_as_accept_discovered() {
    on_large_stack("discovery-admission", || {
        let data = chain_table(200, 4);
        let search = Study::tabular(data.clone())
            .discovery(
                DiscoverySpec::lingam(static_params(), 0.1, "parcorr"),
                vec![v(0), v(1), v(2)],
            )
            .build()
            .unwrap();
        let execution = CompiledExecution::run(&search, &ctx()).unwrap();
        let identity = execution.lowered_identity();
        let (discovery_contract, claim) = execution.contract_and_claim(&ctx()).unwrap();
        let accepted = accept_discovered(execution.result(), &claim, true).unwrap();
        let provenance = accepted.provenance().expect("the review gate records provenance").clone();
        assert_eq!(
            provenance.data_snapshot,
            *discovery_contract.identities.data_snapshot.as_bytes(),
            "provenance keeps the discovery snapshot and not a copy of the table"
        );

        let query = paths();
        let admitted = Study::tabular(data.clone())
            .query(query.clone())
            .admit_finished_discovery(&execution, true, &ctx())
            .unwrap()
            .build()
            .unwrap();
        assert_eq!(execution.lowered_identity(), identity, "admission does not search again");
        let again = Study::tabular(data.clone())
            .query(query.clone())
            .admit_finished_discovery(&execution, true, &ctx())
            .unwrap()
            .build()
            .unwrap();
        assert_eq!(execution.lowered_identity(), identity);

        let bare = accepted.as_dag().expect("LiNGAM accepts a DAG").clone();
        let on_accepted =
            Study::tabular(data.clone()).graph(accepted).query(query.clone()).build().unwrap();
        let on_bare = Study::tabular(data).graph(bare).query(query).build().unwrap();

        let admitted_contract = admitted.prepare(&ctx()).unwrap().contract().unwrap();
        let accepted_contract = on_accepted.prepare(&ctx()).unwrap().contract().unwrap();
        assert_eq!(admitted_contract.structure_source, StructureSource::Accepted);
        assert_eq!(admitted_contract.structure_provenance.as_deref(), Some(&provenance));
        assert_eq!(admitted_contract.discovery_algorithm.as_deref(), Some("direct_lingam"));
        assert_eq!(admitted_contract.identities.program, accepted_contract.identities.program);
        assert_eq!(admitted_contract.identities.program, Some(program(&again, &ctx())));
        assert_ne!(admitted_contract.identities.program, Some(program(&on_bare, &ctx())));

        let other = chain_table(200, 9);
        let refused = Study::tabular(other).query(paths()).admit_finished_discovery(
            &execution,
            true,
            &ctx(),
        );
        let message = refused.expect_err("a different table is not this discovery").to_string();
        assert!(message.contains("different table"), "{message}");
        assert_eq!(execution.lowered_identity(), identity);
    });
}
