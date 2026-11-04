# Categorical-choice evidence to decision

Repeated categorical choices (which of K options was chosen, many times) run through the same compiler-owned layers as the other adaptive-inference families: a declared contract, a procedure-matrix cell that supports each route, an estimator, a sealed artifact that a consumer replays, and a decision value over the result. This page states what each stage keeps and refuses.

Rust: `antecedent::analysis::choice_inference` (contract `antecedent_contract::adaptive_design::ChoiceDesignContract`, estimators `antecedent_estimate::choice_inference`, artifact `antecedent_io::choice_inference_result_artifact`, ranking `antecedent_design::choice_evsi`). Python: `antecedent.choice_inference`.

## Stages

| Stage | Object | What it says | What it never says |
| --- | --- | --- | --- |
| Declare | `ChoiceStudy` / `ChoiceDesignDeclaration` | ordered options, what a choice means, the population, every respondent's rows, retained sources, looks and stopping rule; nothing is defaulted and a missing element is its own typed refusal | that a stated preference identifies a behavioural or causal outcome |
| Qualify prior | retained sources | each source becomes a Dirichlet prior at a power-prior discount, only when its ordered options, meaning and population match the target, ; cross-population transfer is refused until a supported source-to-target method exists | that a distribution transports across populations; the discount is a declaration, not an estimated transfer |
| Infer | `Posterior`, `Stopping`, `Hierarchical`, `HierarchicalStopping` | the pooled Dirichlet posterior of one choice per respondent, the hierarchical Dirichlet-multinomial posterior of repeated choices (two or three options), or the contract's stopping decision on either; each authorized by the procedure matrix | a calibrated interval; every result is point-only with `calibration: unmeasured` |
| Operating characteristics | `operating_characteristics` | the exact law of the declared sequential procedure (selection, stop at each look, horizon, expected enrolment) under a fixed truth or a Dirichlet prior predictive | a bound over other truths, an error rate for any other truth, a coverage statement |
| Rank | `rank_studies` | expected value of sample information over the pooled posterior at a look (exact enumeration or Monte Carlo), or over a hierarchical posterior for one new respondent making one or two choices; net of cost | that the application should field the study |
| Export / consume | `export`, `consume_choice_inference_result_artifact` | `choice_inference_result_v1`, bound to its contract; the consumer replays the contract through the evidence ledger and recomputes every value | trust in the stored numbers |

## Dependence and conflict

The same observation row is never counted twice, between a source and the respondents or between sources: the evidence ledger judges overlap and a contract that repeats a row is refused (`choice_inference.source_overlaps_respondents`, `choice_inference.sources_overlap`). Pooled sources require one choice per distinct respondent, identified by a bare respondent ID or `ID#0`; these spellings are normalized for overlap checks. Pooled counts of respondents who chose repeatedly, and pooled sources under a hierarchical outcome, are refused rather than treated as independent. Heterogeneity is screened per source with a prior-predictive log evidence ratio against a vague prior versus the remaining sources and current look; above `ln 10` the source is flagged, and the posterior reports the flag, while stopping decisions and study rankings refuse until it is resolved (`choice_inference.source_conflict`). For fractional power-prior counts this is a generalized score rather than a calibrated Bayes factor. The threshold is a screen with unmeasured calibration.

## Intervals

`interval()` refuses (`quantity_semantics_mismatch`): an interval belongs to one named result. `measured_interval(request)` builds the interval of one result and reports it with its calibration status. The measured choice cells (the routes below) resolve against the current coverage records; a study outside them still reports its interval, annotated `not_measured`, and until a record exists every study is `not_measured`. Only a stopping decision has no interval (`route_not_supported`). The measured cells are Dirichlet share intervals of the pooled posterior (final-only, and looks under an efficacy or decision-theoretic rule) and the hierarchical population-share interval (final-only and looks under a decision-theoretic rule).

## Verification

- The Dirichlet family is checked against an independent mpmath reference (`dirichlet_choice_reference.py`).
- The hierarchical posterior is checked against two independent references (`hierarchical_choice_reference.py`): mpmath tanh-sinh for two options, a trapezoid rule in the coordinates `ln(kappa pi_k)` and scipy adaptive cubature for three.
- The procedure law is checked against an exact rational reference built from whole-batch multinomial and Dirichlet-multinomial probabilities (`choice_procedure_reference.py`).
- The pooled EVSI enumeration is checked against the closed-form beta-binomial sum for two options and against Monte Carlo within its standard error. Hierarchical study value is checked against an independent two-dimensional numerical quadrature; its result reports posterior convergence and predictive-mass diagnostics, not a certified ranking order.

Provenance records: `adaptive_inference.dirichlet_choice`, `adaptive_inference.hierarchical_choice`, `adaptive_inference.choice_procedure_law` and `choice_inference.contract`. The references were written by the implementing agent, not by an independent reviewer; the records say so.

## Calibration (wired, not run)

Calibration is tied to commit hashes and is a later phase. The choice routes are wired for it and measured nothing:

- `crates/antecedent/tests/choice_inference_calibration.rs` holds `#[ignore]`d prior-predictive expectation gates (probability-best calibration, correctness of the selected option at the stopping threshold, EVSI against realized gain, the procedure law against simulation, and the conflict screen's false-flag rate against its Markov bound of one in ten).
- `crates/antecedent-estimate/tests/choice_interval_calibration.rs` holds four record-emitting coverage designs (pooled share interval, hierarchical population-share interval, with and without looks).
- Both are registered in `scripts/gate_calibration.sh` and `parity/v3_test_registry.toml`.

Owed: run the gates, refresh the frozen family roster after review, and let the collector write the rows. Until then every output carries `calibration: not_measured`. A record stays current while its dependency fingerprint is unchanged; otherwise a probe decides whether it still applies.

## What is not claimed

- **Calibration.** No coverage of any interval, no frequentist error rate of a stopping rule for any truth but the declared one, no calibrated conflict screen.
- **Behaviour.** Shares of stated choices do not identify behavioural or causal outcomes; the frame carries its meaning and a mismatch is refused.
- **Shapes below one.** The quadrature needs bounded densities; priors with a shape below one are refused.
- **Other preference models.** No conjoint, MaxDiff or arbitrary random-coefficient likelihood, and no externally fitted model promoted to native authority; a new family needs its own independent numerical evidence and calibration.
- **Transport.** A source from another population or meaning is refused; a rationale and discount do not establish transport.

## Boundary

The application owns candidate designs, acquiring observations, fielding, monitoring, orchestration and authorizing fieldwork. Antecedent validates the contract, computes the posterior, the decision criterion, the procedure's exact law and the study values.

A pooled native posterior may be reused through `RetainedSource.from_result` when the source
matches the target frame and base prior and has an independently
retained artifact identity. The source likelihood is discounted once; the base prior is not
counted twice. Its artifact and identity travel into the new result and are replayed on
consumption. Nested pooled posterior ancestry and its compounded discounts are replayed, subject to an eight-level and 16 MiB bound. Hierarchical posterior source reuse remains unsupported.

A prepared session owns current source standing in its evidence ledger. Withdrawing a retained
source blocks further analysis, study ranking and `consume_current` for that session. Independent
artifact replay remains historical; it does not restore a withdrawn session's authority. When
source standing changes outside the session, the application must propagate that withdrawal to
each affected session.

Hierarchical study ranking considers one new independent respondent making one or two choices,
with within-respondent dependence retained. It refits the native hierarchical posterior for
each possible count vector. It reports numerical diagnostics and marks pairwise ordering as
uncertain; it does not establish a calibrated uncertainty claim.
