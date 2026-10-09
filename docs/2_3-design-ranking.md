# Design ranking: which study to run next (2.3)

`antecedent.design` is one package with one entry point, `rank_designs`, and one
result, `DesignRankingResult`. The result's `basis` says what its order means, and
different objectives answer different questions and are never mixed in one result.

```python
design.rank_designs(
    candidates, *, objective=None, decision=None, prior=None,
    signal=None, cost_map=None, ...
)
```

| `basis` | You supplied | Candidates are | Ranked by |
| --- | --- | --- | --- |
| `"identification"` | `prior=StructurePrior(...)`, no `decision` | plans: `Measurement`, `Experiment`, `Environment`, `Sampling` | the gain in the probability that a query becomes identified |
| `"evsi"` | `decision=DesignDecision(...)`, no cost map | `Candidate` studies | expected value of sample information, costs reported separately |
| `"net_value"` | `decision=` and `cost_map=` | `Candidate` studies | EVSI net of study cost under the cost map |
| `"structural_sufficiency_cost"` | `StructuralCandidate` declarations | structural declarations | caller-declared sufficiency, cost units, budget, id |
| `"graph_entropy"` | `objective=GraphEntropy()`, `prior=` | plans | original heuristic graph-channel entropy reduction |
| `"effect_width"` | `objective=EffectWidth(...)` | plans | signed linear-model treatment standard-error reduction |
| `"model_distinction"` | `objective=ModelDistinction(...)` | plans | original reliability-scaled pairwise log-likelihood-gap heuristic |
| `"decision_regret"` | `objective=DecisionRegret(...)` | plans | preposterior EVSI using a callable utility |

Claim labels: every value is **point-only**. A Monte Carlo value (when a signal has
no exact integral) is a `monte_carlo_estimate` with no coverage claim. The Monte
Carlo error, `rank_uncertain` and the whole ranking's `calibration == "unmeasured"`
are diagnostics, not licensed intervals. A ranking recommends an action; it does
not run it.

The former `design_ranking` module is gone; there is no second ranking surface.
`rank_structural` remains a compatibility view of `rank_designs` for structural
declarations (see the end of this page).

## Ranking plans by identification

With no decision declared, plans are ranked by how much they raise the probability
that a query is identified under a `StructurePrior`, the weights over candidate
causal structures with each flagged identified or not. Snippet from
`examples/python/rank_designs.py`:

```python
from antecedent import design

ranking = design.rank_designs(
    [
        design.Measurement([3], tag=1),
        design.Environment(7, additional_rows=50),
        design.Sampling(10),
        design.Experiment([0]),
    ],
    prior=design.StructurePrior(
        weights=(0.5, 0.3, 0.2), identified=(True, False, False), keys=(10, 20, 30)
    ),
    query_id=0,
    variable_unlocks={0: [3]},         # query id -> variable ids whose measurement identifies it
    environment_unlocks={0: [7]},      # query id -> environment ids whose observation does
    monte_carlo=design.MonteCarlo(
        min_batches=2, max_batches=4, batch_size=4, rank_uncertainty_threshold=1.0
    ),
    rng_seed=3,
)
ranking.basis                     # "identification"
ranking.best                      # the top IdentificationCandidate
print(ranking.explain())
```

`StructurePrior.uniform([True, False])` gives equal weight on each structure.
`max_cost=` and `max_sample_budget=` exclude plans, reported as `ranking.violations`.
The score is relative to the prior's current identified mass (0.75 moves 25 percent
to 100 percent). `candidate.probability` is the absolute identified mass after the
plan, including the prior baseline; `candidate.score` is its gain. The
`min_identification` gate also uses the absolute mass. An identification ranking
has **no artifact**: `export`,
`identity` and `expectation` raise `CausalValueError` naming the basis.

## Ranking studies by value of information

Declare the decision the information is for, a signal request, and the candidate
studies.

```python
from antecedent import design
from antecedent.joint_distribution import ScientificQuantity

def quantity(name):
    return ScientificQuantity(
        variable_id=f"schema:{name}", variable_name=name, role="outcome",
        units="dimensionless", population_id="target", regime_id="observational",
        horizon=0, functional_id="state",
    )

# Bet on a coin whose bias is 1/4 or 3/4 with equal prior weight: the bet pays theta - 1/2.
decision = design.DesignDecision(
    contract="contract-bet",
    actions=(design.ActionUtility("abstain", 0.0, 0.0), design.ActionUtility("bet", -0.5, 1.0)),
    prior=design.StatePrior.draws([0.25, 0.75]),
    utility_units="utility",
)
signal = design.SignalSpec(
    prior_id="prior-1",
    state=quantity("state"),
    observation=quantity("flips"),
    evidence_lineage=("snapshot:example",),
    rng_seed=3,
)
studies = [
    design.Candidate("two-flips", 2, design.BinomialSignal(), cost=0.02),
    design.Candidate("four-flips", 4, design.BinomialSignal(), cost=0.10),
]
cost_map = design.CostMap("utility", "utility", 1.0)

ranking = design.rank_designs(studies, decision=decision, signal=signal, cost_map=cost_map)
ranking.basis                       # "net_value"
ranking.candidate("two-flips").evsi <= ranking.evpi
```

(`examples/python/rank_designs_evsi.py` runs this fixture. Its oracle: with
`n = 2`, `EVSI = 1/16` and `EVPI = 1/8`.)

The declaration types:

- `DesignDecision(contract, actions, prior, utility_units)`: the terminal action
  set is the same before and after the information; an action-set change is a
  different problem and refuses. `actions` are `ActionUtility(id, intercept, slope)`
  with utility `intercept + slope * state`.
- `StatePrior`: `StatePrior.draws([...])` (equally weighted draws, any signal
  family) or `StatePrior.normal(mean, variance)` (conjugate, needs a
  `GaussianMeanSignal`, no constraints). Not to be confused with `StructurePrior`.
- `SignalSpec`: the exact scientific request a signal answers: the state and
  observation `ScientificQuantity` coordinates, the evidence lineage and the
  conditional-independence assumption.
- `Candidate(id, sample_size, provider, cost=, cost_unit=, signal=, reused_observations=, plan=)`.
  `provider` is `BinomialSignal()`, `GaussianMeanSignal(...)` or an `ExternalSignal`.
- `CostMap(cost_unit, utility_unit, utility_per_cost)`: a positive linear map from
  study cost to the decision's utility unit. Currency without a cost-to-utility
  map does not compare with EVSI.

### From a contract, and from a belief

You rarely write `ActionUtility` by hand. `DesignDecision.from_contract` extracts
the affine action utilities from a `decision.Contract`, and
`StatePrior.from_distribution` reads one coordinate of a joint distribution that is
a *belief* about the state:

```python
declared = design.DesignDecision.from_contract(
    contract, prior=design.StatePrior.draws([0.0, 1.0])
)
declared.actions          # (ActionUtility("guess0", 1.0, -1.0), ActionUtility("guess1", 0.0, 1.0))
declared.contract is contract

prior = design.StatePrior.from_distribution(belief, 0)   # a coordinate index or a ScientificQuantity
```

`from_contract` refuses by name an action whose utility is not affine in the state
(`'square' utility is not affine`), a second state quantity (pass `state=`), a
criterion other than expected utility, and hard constraints. `from_distribution`
accepts only `parameter_posterior`, `causal_functional_posterior`,
`posterior_predictive` or `interventional_predictive` distributions: an estimator's
sampling distribution, a bootstrap or an empirical outcome is not a belief and
refuses, as do unequal weights and draws marked unsupported.

### External signals

An `ExternalSignal` carries a foreign provider's observation law and update. Trust
is always `externally_attested`, never native, and its value is never natively
replayed.

```python
law = design.ExternalLaw.posterior(
    states=[0.0, 1.0], statistics=[0.0, 1.0], predictive=[0.5, 0.5],
    posterior=[[0.75, 0.25], [0.25, 0.75]],
)
external = design.ExternalSignal("lab", "signal-object", "v1", "snap", "lab-qa", law)
candidate = design.Candidate("study", 1, external, cost=0.0)
```

`ExternalLaw.likelihood(states, statistics, probabilities)`, `.posterior(...)` and
`.decision_values(branch_probabilities, action_ids, values)` are the three update
modes (`native_update`, `external_posterior`, `external_decision_values`). A
likelihood is updated natively; a posterior is checked for coherence (it must
average back to the prior); decision values are only checked and combined. The
candidate's `assumptions` include
`update_computed_externally_not_verified_by_antecedent` for the second and third.
Attested candidate, prior or sample-size identities that do not match the request
refuse (`signal_provider.candidate_mismatch`, `.prior_mismatch`,
`.sample_size_mismatch`); an incoherent posterior or law refuses
(`signal_provider.posterior_incoherent`, `.law_incoherent`).

### The identification gate

With both `decision=` and a structure `prior=`, each candidate must carry a
`plan`. Its identification probability is reported as `ranking.gate` and candidates
below `min_identification` are not valued:

```python
informative = design.Candidate("cand-1", 1, external, plan=design.Measurement([3]))
ranked = design.rank_designs(
    [informative], decision=declared, signal=signal, cost_map=cost_map,
    prior=design.StructurePrior(weights=(0.5, 0.5), identified=(True, False), keys=(1, 2)),
    variable_unlocks={0: [3]}, min_identification=0.5,
)
ranked.gate.rejected      # ids that failed the gate and were left out of the value ranking
```

A candidate without a plan under `prior=` raises `CausalValueError` ("needs a
plan"). Identifiability is a gate, not a score: it never mixes into the value.

## Reading a value ranking

Each `CandidateValue` has `evsi`, `evpi` (the upper bound), `net_value` (when a
cost map was used), `study_cost_utility`, `rank`, `rank_uncertain`, `integration`
(method `exact`, `monte_carlo` or `externally_computed`, with error and replicates),
`provider_trust`, `update_mode`, `natively_replayed`, `trust_limit` and
`source_overlap`. The frozen F14 guess decision (binary state, prior 1/2) is the
oracle: signals of accuracy 3/4 and 5/8 have EVSI 1/4 and 1/8 and `EVPI = 1/2`;
with a utility-unit cost of 1/10 the net values are 3/20 and 1/40. `design.evsi(decision,
candidate, ...)` values one candidate and is `rank_designs([candidate], ...)` for it.

The ranking depends only on the candidate set, never on the order supplied, and its
`identity` is invariant to that order. A search bounded by `max_candidates`
(default 1024) says so in `ranking.search` (`truncated`, `evaluated`,
`unevaluated_ids`). Ties are reported in `ranking.ties`. `ranking.explain()` states
the basis and, with a gate, which candidates "failed, not valued".

## Exporting and consuming

A value ranking exports a durable artifact, which an independent consumer replays
by recomputation:

```python
ranking = design.rank_designs(studies, decision=decision, signal=signal, cost_map=cost_map)
consumed = design.consume(ranking.export(), expected_identity=ranking.expectation())
consumed.identity == ranking.identity
all(entry.natively_replayed for entry in consumed.entries)
```

`consume(data, *, expected_identity=, skip_expectation_check=)`:

- `expected_identity` is a `design.Expectation` of identities retained
  independently of the bytes (`ranking.expectation()` builds one; construct
  `Expectation(cost_map=..., decision_contract_identity=..., ...)` by hand for a
  stricter or partial check). A changed signal, update mode, source digest, cost
  mapping or contract refuses even when the artifact was resealed.
- Without it the bytes are only checked against themselves, so a resealed artifact
  would be accepted. Pass `skip_expectation_check=True` to say that is intended.
  Supplying neither, or both, raises `CausalValueError`.
- What is replayed: a native exact-integration signal is rebuilt and its EVSI,
  EVPI and net value recomputed (`natively_replayed`); an external likelihood or
  posterior has only the arithmetic recomputed from the retained attested table; an
  external decision value is checked for coherence and combined; a Monte Carlo
  value is bound but not re-simulated. Corruption, truncation and unknown versions
  raise `CausalSerializationError`.

`ranking.lineage`, `consumed.lineage` and `ranking.stages_behind()` name every stage
behind the ranking, including the foreign provider. The ranking is bound to a
joint law only by its source digest (`source_digests=`).

## Refusals

`design.DesignRankingRefusal` and its subtypes carry `reason_code`, `remedy`,
`stage`, `detail`, `offending`, `expected` and `supplied`:

| Type | Code and detail | Cause |
| --- | --- | --- |
| `SignalProviderRefusal` | `design_signal_invalid`, `signal_provider.*` | wrong candidate, prior or sample size; incoherent law or posterior |
| `CostUnitsRefusal` | `design_cost_units_mismatch`, `evsi.cost_units_mismatch`, `evsi.cost_map_required`, `design_ranking.cost_units_mismatch` | currency without a mapping, a mapping to another utility unit, `require_net_value=True` with no map, `consume` under a different cost map |
| `SourceOverlapRefusal` | `evsi.source_overlap` (offending: the shared observation) | a study reuses observations the prior already summarizes (`prior_observations=`, `reused_observations=`) |
| `DesignRankingRefusal` | `evsi.*`, `design_ranking.*` | a changed terminal action set, bound violations, incompatible artifact |

Catch `CausalUnsupportedError` to handle all of them; see
[refusals](refusal-and-partial-knowledge.md#structured-refusals-and-their-remedies).

## Structural declarations through the shared ranker

When neither a structure prior nor a decision exists, pass `StructuralCandidate`
values directly to `design.rank_designs`. It returns `DesignRankingResult` with
`basis == "structural_sufficiency_cost"`, ordered by the caller-declared verified
sufficiency flag, then cost units, sample budget and semantic id. The flag is not
an identification proof produced by the ranker. Entries have no numeric `score`
or `probability`; no identification probability or value of information is inferred.
The structural ordering identity is invariant to input order. `rank_structural`
delegates to the same implementation and returns the existing `StructuralRanking`
compatibility view; composite structural consumers continue using that view.

```python
ranking = design.rank_designs([
    design.StructuralCandidate("repair", True, 3, 100),
    design.StructuralCandidate("cheap", True, 1, 100),
])
ranking.basis                 # "structural_sufficiency_cost"
ranking.best.id               # "cheap"
```

## Other native objectives through the shared ranker

Use typed objective declarations instead of calling `_native.rank_designs` with
raw dictionaries. The candidates are the same four design plan types.

```python
ranking = design.rank_designs(
    [design.Sampling(20), design.Sampling(100)],
    objective=design.EffectWidth(
        xtx=(4.0, 0.0, 0.0, 9.0), sigma2=4.0, treatment_col=1, n=20,
    ),
)
ranking.basis                 # "effect_width"
ranking.best.implemented_functional  # "ols_gram_se_reduction"
```

`GraphEntropy()` uses `prior=StructurePrior(...)` and its optional graph features.
It exposes the existing soft-observation heuristic, not likelihood-based expected
information gain. `EffectWidth` uses declared linear-model Gram information;
`MeasurementColumn`, `DesignInformation` and `EnvironmentInformation` supply
prospective covariate, intervention and environment information. Sampling and
unspecified environments use the native isotropic sample-size scaling. The score
is `se_before - se_after`, not a calibrated posterior or confidence interval width.

`ModelDistinction(model_ids, log_likelihoods)` takes one aligned draw row per model
and retains the native heuristic reliability-scaled log-score contrast. It is not
a posterior model probability or a Bayes factor.

For a callable utility, use the same ranker with a `DecisionRegret` objective:

```python
import numpy as np

ranking = design.rank_designs(
    [design.Sampling(1), design.Sampling(10)],
    objective=design.DecisionRegret(
        actions=(0.0, 1.0),
        utility=lambda actions, outcomes: np.outer(actions, outcomes**2 - 0.25).ravel(),
        prior=design.StatePrior.draws((0.25, 0.75)),
        signal=design.BinomialSignal(),
    ),
)
```

The utility returns a contiguous flat `float64` action-by-outcome array and must be
deterministic. Native callback refusals retain the original exception as their
cause. Only native-licensed plans are scored; unsupported candidates appear in
`violations`. `DecisionRegret` scores EVSI before cost, without subtracting or
converting study costs. For affine utilities and portable signal-provider study
artifacts, continue using `decision=DesignDecision(...)` and `cost_map=`.

These native objective results expose `ObjectiveCandidate` entries, each retaining
`implemented_functional`, `evaluation`, `stderr` and `rank_uncertain`. Their original
native scoring semantics are unchanged. None has an identification probability or
portable artifact. Bounds use `max_cost` and `max_sample_budget`; unrelated value
or gate arguments refuse rather than being silently applied. Calibration remains
`unmeasured` for every objective.

Runnable examples: [`rank_designs.py`](../examples/python/rank_designs.py)
(identification basis), [`rank_designs_evsi.py`](../examples/python/rank_designs_evsi.py)
(net value) and [`decision_lifecycle.py`](../examples/python/decision_lifecycle.py)
(contract to ranking).

## Where it fits

Study candidates also come from [identification repair](2_3-decisions-breadth.md#repair),
which names the evidence a failed contract owes; the ranking values the studies
that would supply it. A ranking exported here is a `study_ranking` bundle node (see
[composition](2_3-composition.md)), and the full path from analysis to ranking is the
[2.3 lifecycle](2_3-lifecycle.md). The route-by-route scope table is in
[decisions, design, repair and estimator breadth](2_3-decisions-breadth.md#signals-evsi-and-design-ranking).
