# antecedent-design

Experiment, measurement and decision design: candidate ranking by identification mass and
standard-error reduction, a heuristic graph-channel entropy score, decision utility with the
preposterior value of information, study and transport planning, and batched Monte Carlo
evaluation.

`RankedCandidate.implemented_functional` names the mathematics actually scored, and
`RankedCandidate.evaluation` says whether a score is that functional's exact value or a Monte
Carlo estimate. `ReduceDecisionRegret` is the expected value of sample information (Raiffa and
Schlaifer's preposterior analysis): the expected reduction in decision regret from a
candidate's sample, given a decision problem, a prior (`DecisionPrior`) and a sampling model
(`DecisionSignal`). It is exact for finite-support signals and for the conjugate normal model
with an affine utility. The other objectives are heuristics, not expected information gain
under a likelihood `p(y | G, design)`; `DesignObjective::is_exact_information_functional` is
true only for `ReduceDecisionRegret`.

Most users want the [`antecedent`](https://crates.io/crates/antecedent) facade, which binds ranking to a prepared study's contract and re-exports the types under `antecedent::design`. Depend on this crate directly only to rank designs outside a study.

Main types: `DesignRanker`, `DesignObjective`, `RankedCandidate`, `ScoredDesignRanking`. API reference: <https://docs.rs/antecedent-design>.

## License

MIT OR Apache-2.0.
