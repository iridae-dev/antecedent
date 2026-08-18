# Antecedent and other causal libraries

DoWhy, EconML, Tigramite, and causal-learn are good libraries. Antecedent's
conformance suite pins two of them — DoWhy and Tigramite — as black-box
reference baselines: we run the pinned releases, record their outputs, and
test Antecedent's independent implementations against those outputs (see
[ADR 0009](https://github.com/iridae-dev/antecedent/blob/main/adr/0009-parity-baselines.md)).
Antecedent's code is not derived from either project, but choosing them as
the yardstick should tell you how seriously we take them. This page describes where
each tool's center of gravity is, so you can pick the right one — sometimes
that is not Antecedent, and the last section lists things Antecedent
deliberately does not do.

The one-line orientation: Antecedent is an identification-first engine for
carrying a causal analysis from structure through estimation, interventions,
and counterfactuals while preserving uncertainty about the structure itself.
Each library below is better than Antecedent at something.

## Antecedent vs. DoWhy

[DoWhy](https://github.com/py-why/dowhy) established the four-step causal
workflow (model, identify, estimate, refute) that most of the Python ecosystem
now assumes, and its `gcm` module covers graphical causal models, attribution,
and root-cause analysis. It is the most widely used general causal inference
library in Python, with the tutorials, examples, and community answers that
come with that.

The overlap with Antecedent is large by design: end-to-end workflow,
identification step, refutation emphasis, GCM-based attribution. The
differences are mostly about enforcement and uncertainty:

- **Enforcement.** DoWhy computes an identification result and then lets you
  proceed; the workflow is advisory. Antecedent's pipeline is gating:
  estimation does not run on an unidentified query, an unreviewed partial
  graph raises an error rather than being completed, and priors cannot upgrade
  a nonparametric identification status.
- **Structural uncertainty.** DoWhy takes the graph as given. Antecedent
  treats discovered structure as evidence — CPDAGs and PAGs keep their
  unoriented marks, Bayesian graph posteriors propagate into effect posteriors,
  and posterior mass on structures where the effect is unidentified is
  reported rather than renormalized away.
- **Surface.** DoWhy is Python with string-named estimands and scikit-learn
  interoperability. Antecedent is a Rust engine with typed queries and typed
  graphs, exposed to Python; there is no scikit-learn integration.

Choose DoWhy when you want the mature ecosystem: the widest set of worked
examples, direct integration with EconML's estimators for heterogeneous
effects, and a community that has probably already answered your question.

## Antecedent vs. EconML

[EconML](https://github.com/py-why/econml) is about estimating heterogeneous
treatment effects with machine learning: double machine learning, causal
forests, doubly robust learners, meta-learners, and ML-augmented instrumental
variables, with valid confidence intervals and policy-learning utilities on
top. It largely assumes the identification question is settled (typically
unconfoundedness or a known instrument) and puts its depth into flexible
estimation.

Antecedent is close to the opposite emphasis. Its identification machinery is
broad (backdoor, front-door, IV, ID/IDC on ADMGs, temporal strategies,
partial-graph adjustment), while its conditional-effect estimation is linear —
there are no ML-based CATE estimators and none are planned short-term.

The honest division: if your question is "which customers respond most, and
how confident are we?", use EconML. If your question is "is this effect
identified at all, from which structure, and what happens to it when the
structure is uncertain?", use Antecedent. If you need both, run Antecedent for
structure and identification and EconML for the CATE — there is no built-in
integration, but the identification result tells you which adjustment set to
hand EconML.

## Antecedent vs. Tigramite

[Tigramite](https://github.com/jakobrunge/tigramite) is the reference
implementation of the PCMCI family of time-series causal discovery algorithms,
from the group that develops them. It has a rich library of conditional
independence tests, mature plotting for lagged causal graphs, and a causal
effects module for estimation on time-series graphs.

Antecedent independently implements the PCMCI family (PCMCI, PCMCI+, LPCMCI,
J-PCMCI+, and regime-specific RPCMCI workflows) and validates them black-box
against outputs recorded from a pinned Tigramite release — Tigramite remains
the reference implementation; Antecedent checks its own answers against it
without sharing any code. What it adds is everything after discovery in the same
engine: temporal identification through unfolded backdoor and temporal
mediation, pulse and sustained intervention queries, temporal counterfactual
trajectories, and incremental `CausalState` for online analysis. What it lacks
relative to Tigramite: visualization (Antecedent has graph interchange, not
plotting), some CI test variants, and same-day access to new methods — new
PCMCI research lands in Tigramite first, by definition.

Choose Tigramite for time-series discovery research, exploratory workflows
where plotting matters, or when you want the newest algorithms from the
source. Choose Antecedent when temporal discovery is step one of an effect
or intervention analysis and you want one engine — and one set of
guarantees — from the lagged graph to the estimate.

## Antecedent vs. causal-learn

[causal-learn](https://github.com/py-why/causal-learn) is the broadest
collection of causal discovery algorithms in Python: constraint-based (PC, FCI
and variants), score-based (GES, GRaSP, exact search), functional causal
models (the LiNGAM family, additive noise models, PNL), Granger causality, and
more, with reference implementations from the groups that authored many of the
methods. It stops, deliberately, at the graph.

Antecedent implements a smaller discovery set (PC, FCI, RFCI, GES,
DirectLiNGAM, NOTEARS, plus the temporal and Bayesian-posterior methods) and
is opinionated about what happens next: the output is typed as a CPDAG or PAG
that keeps its uncertainty, and it feeds identification and estimation rather
than being handed back as an adjacency matrix.

Choose causal-learn when discovery itself is the research question, or when
you need an algorithm Antecedent doesn't implement. Choose Antecedent when
discovery is the first step of an effect analysis and you want the equivalence
class, not a guessed DAG, to be what flows downstream. They also compose: a
graph found with causal-learn can be reviewed and passed to Antecedent's
`analyze()` as an edge list or via NetworkX interchange.

## Continuous responses, transport, and interference

From 0.5, Antecedent treats continuous causal responses as first-class
objects (`ResponseCurve`, derivatives, elasticities, Jacobians), not as a
sequence of unrelated binary contrasts. Identification, empirical support, and
uncertainty kind stay separate axes. Structural transport and randomized
interference live in stage modules because they change what identifies the
estimand.

If your question is a flexible ML dose–response under assumed unconfoundedness,
Kennedy-style tooling or EconML may still be a better fit for the *estimator*.
If your question is whether a demand curve, elasticity, or transported response
is identified from a graph — and what happens when support is weak — Antecedent
is the engine that keeps those judgments from collapsing into one number.
Incomplete-graph (PAG) response cells are not licensed; see the
[support matrix](support-matrix.md).

## When to use each

- **You want the most worked examples, tutorials, and community support for a
  general causal analysis** — DoWhy.
- **You need heterogeneous treatment effects estimated with ML, with
  confidence intervals or a policy on top** — EconML.
- **You are doing time-series discovery research, want plots, or need the
  newest PCMCI-family methods** — Tigramite.
- **You need a discovery algorithm outside Antecedent's set, or discovery is
  the research question** — causal-learn.
- **You want one engine from structure to estimate with identification
  enforced, structural uncertainty preserved, temporal and online analysis,
  or a Rust-native core** — Antecedent.

These are not exclusive. A reasonable pipeline discovers with causal-learn or
Tigramite, reviews the graph, and estimates with Antecedent or DoWhy + EconML.

## What Antecedent deliberately does not support

Some of these are scope decisions, some are sequencing; all are current and
stated rather than discovered the hard way.

- **ML-based heterogeneous effect estimation.** No double machine learning,
  causal forests, or meta-learners. `ConditionalEffect` is linear. Use EconML.
- **Silent handling of structural uncertainty.** No auto-orientation of
  CPDAG/PAG marks, no silent completion of partial graphs, no dropping of
  unidentified graph-posterior mass. These raise errors or are reported —
  that is the point of the library, but it means more friction than tools
  that guess for you.
- **Full PAG-native ID/IDC.** Identification over PAGs uses generalized
  adjustment plus explicit envelopes or enumerated completions with reported
  unidentified mass. Complete PAG-native identification is not claimed.
- **Automatic estimator choice.** `AutoIdentifier` reports which strategies
  apply; it does not pick one for you.
- **Priors that rescue identification.** Bayesian machinery quantifies
  uncertainty for identified (or partially identified) queries; it cannot turn
  an unidentified query into an identified one.
- **Unsupervised regime discovery.** RPCMCI workflows require the regime
  structure to be specified; discovering regimes is out of scope.
- **Exact DAG posteriors beyond 6 nodes.** The exact posterior is a hard
  n ≤ 6; larger graphs use order/structure MCMC or CI-screened posteriors.
- **Visualization.** No plotting module. Graphs interchange with NetworkX,
  DOT, JSON, and GML; bring your own renderer.
- **A string query language.** Queries are typed objects, not do-calculus
  strings. This is a design choice, not a gap we intend to close.
- **Bindings beyond Python and Rust.** No R, Julia, or JS bindings through
  1.0. The engine is CPU-native Rust (`faer`). A post-1.0 WebAssembly /
  TypeScript facade is a runtime track, not an R/Julia port.
- **Complete general sID.** 0.5 ships a sound transport subset with
  `NotCertified` outside it; definitive non-transportability certificates and
  multi-node c-component recursion are not claimed.

If one of these is load-bearing for you today, one of the libraries above is
the better choice, and we would rather you find that out here than after an
afternoon of integration.
