// SPDX-License-Identifier: MIT OR Apache-2.0
//! Fact projections for compiled plan layers.

use super::{
    Composition, DesignShape, EvidenceDecisionProgram, Fact, PosteriorPlan, ProblemTarget,
    ResamplingPlan, ScientificProblem, StatisticalProgram, WidthSource, fact, hex,
    realization_name,
};

pub(super) fn design_json(design: DesignShape) -> Fact {
    match design.source() {
        WidthSource::Quoted => fact!({
            "source": "quoted",
            "rows": design.rows(),
            "columns": design.columns(),
            "direct": design.direct(),
        }),
        WidthSource::NotDesignBased(reason) => fact!({
            "source": "not_design_based",
            "reason": reason,
        }),
    }
}

pub(super) fn composition_json(composition: Composition) -> Fact {
    match composition {
        Composition::Single => fact!({"kind": "single"}),
        Composition::AtomMixture { atoms } => fact!({"kind": "atom_mixture", "atoms": atoms}),
        Composition::ScenarioSet { scenarios } => {
            fact!({"kind": "scenario_set", "scenarios": scenarios})
        }
        Composition::Transported => fact!({"kind": "transported"}),
        Composition::Pooled { sources } => fact!({"kind": "pooled", "sources": sources}),
        Composition::Chained { stages } => fact!({"kind": "chained", "stages": stages}),
    }
}

pub(super) fn resampling_json(plan: ResamplingPlan) -> Fact {
    fact!({
        "method": plan.method().as_str(),
        "replicates": plan.replicates(),
        "unit": plan.unit().as_str(),
    })
}

pub(super) fn posterior_json(plan: &PosteriorPlan) -> Fact {
    fact!({
        "algorithm": plan.algorithm().as_str(),
        "draws": plan.draws(),
        "prior": plan.prior().name(),
    })
}

pub(super) fn scientific_json(problem: &ScientificProblem) -> Fact {
    let target = match problem.target() {
        ProblemTarget::Target(target) => {
            fact!({"form": "target", "identity": target.identity().to_hex()})
        }
        ProblemTarget::Targets(targets) => fact!({
            "form": "targets",
            "identities": targets.iter().map(|target| target.identity().to_hex()).collect::<Vec<_>>(),
        }),
        ProblemTarget::QueryFunctional { kind, outcome, query } => fact!({
            "form": "query_functional",
            "kind": kind.as_str(),
            "outcome": outcome.map(Fact::Variable),
            "query": hex(query),
        }),
    };
    fact!({
        "target": target,
        "plan_id": problem.plan_id(),
        "data": crate::planner::lowered::data_classification_name(problem.data()),
        "graph_class": problem.graph_class(),
        "query_variables": Fact::variables(problem.query_variables()),
        "source_digest": problem.source_digest().as_ref().map(hex),
    })
}

pub(super) fn statistical_json(program: &StatisticalProgram) -> Fact {
    fact!({
        "identity": program.identity().to_hex(),
        "identifier": program.identifier().as_ref().map(crate::IdentifierId::as_str),
        "estimator": program.estimator().as_ref().map(crate::EstimatorId::as_str),
        "realization": realization_name(program.realization()),
        "models": program
            .models()
            .iter()
            .map(|model| fact!({
                "role": model.role().as_str(),
                "family": model.family(),
                "inputs": Fact::variables(model.inputs()),
                "output": model.output().map(Fact::Variable),
            }))
            .collect::<Vec<_>>(),
        "folds": program.folds().map(|folds| fact!({
            "folds": folds.folds(),
            "unit": folds.unit().as_str(),
            "embargo": folds.embargo(),
        })),
        "resampling": program.resampling().map(resampling_json),
        "posterior": program.posterior().map(posterior_json),
        "design": design_json(program.design()),
        "operators": program.operators().len(),
    })
}

pub(super) fn evidence_json(program: &EvidenceDecisionProgram) -> Fact {
    let decision = program.decision().map(|decision| {
        let evsi = decision.information_value().map(|evsi| {
            let tasks = evsi.monte_carlo();
            fact!({
                "candidates": evsi.candidates().iter().map(|candidate| &**candidate).collect::<Vec<_>>(),
                "prior_observations": evsi.prior_observation_ids().len(),
                "rng_seed": evsi.rng_seed(),
                "max_candidates": evsi.max_candidates(),
                "draws_per_candidate": evsi.rank().draws_per_candidate(),
                "monte_carlo_tasks": tasks.tasks(),
                "monte_carlo_bytes": tasks.total_bytes(),
                "cost_map": evsi.cost_map().is_some(),
            })
        });
        fact!({
            "carrier": decision.carrier().as_str(),
            "affine_utility": decision.utility().is_affine(),
            "actions": decision.actions(),
            "constraints": decision.constraints(),
            "chance_threshold": decision.chance_threshold(),
            "utility_unit": decision.utility_unit(),
            "contract_identity": decision.contract_identity(),
            "scenario_source_identity": decision.scenario_source_identity(),
            "information_value": evsi,
        })
    });
    let planning = program.planning().map(|planning| {
        fact!({
            "request_identity": planning.request_identity().to_hex(),
            "simulation": planning.simulation().map(|quote| fact!({
                "replicates": quote.replicates(),
                "record_bytes": quote.record_bytes(),
                "scratch_bytes": quote.scratch_bytes(),
                "workers": quote.workers(),
                "total_bytes": quote.total_bytes(),
            })),
        })
    });
    let chain = program.chain().map(|chain| {
        fact!({
            "label": chain.plan().label,
            "output": chain.plan().output,
            "declaration_identity": chain.declaration_identity(),
            "stages": chain
                .plan()
                .stages
                .iter()
                .map(|stage| fact!({
                    "id": stage.id,
                    "kind": stage.kind.as_str(),
                    "representation": stage.representation.as_str(),
                    "sources": stage.sources,
                }))
                .collect::<Vec<_>>(),
        })
    });
    fact!({
        "identity": program.identity().to_hex(),
        "sources": program
            .sources()
            .iter()
            .map(|source| fact!({
                "role": source.role().as_str(),
                "digest": source.digest().as_ref().map(hex),
            }))
            .collect::<Vec<_>>(),
        "composition": composition_json(program.composition()),
        "decision": decision,
        "planning": planning,
        "chain": chain,
    })
}
