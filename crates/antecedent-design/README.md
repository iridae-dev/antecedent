# antecedent-design

Experiment and measurement design primitives: heuristic graph-channel entropy
(historically named EIG), unlock-list identification mass, OLS Gram SE reduction,
decision utility with preposterior value of information, and batched Monte Carlo
candidate ranking.

`RankedCandidate.implemented_functional` names the mathematics actually scored, and
`RankedCandidate.evaluation` says whether a score is that functional's exact value or
a Monte Carlo estimate. `ReduceDecisionRegret` is the expected value of sample
information (Raiffa & Schlaifer's preposterior analysis): the expected reduction in
decision regret from a candidate's sample, given a decision problem, a prior
(`DecisionPrior`) and a sampling model (`DecisionSignal`). It is exact for
finite-support signals and for the conjugate normal model with an affine utility.
The other objectives are heuristics, not expected information gain under a
likelihood `p(y | G, design)`; `DesignObjective::is_exact_information_functional`
is true only for `ReduceDecisionRegret`.
