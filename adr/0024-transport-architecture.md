# ADR 0024 — Transport architecture and licensing

- Status: Accepted
- Date: 2026-09-20
- Extends: [0022](0022-causal-compiler-contract.md), [0020](0020-support-matrix-and-prepared-workflow.md), [0023](0023-learner-substrate.md)

## Context

1.10 licensed one transport cell: `analyze` + `TransportQuery` + trial-to-target
IPW on an explicit ADMG. Identification is the implemented sID subset
(`Direct` or S-admissible standardize). `NotCertified` means no implemented
sound rule applied. That refusal also absorbed "the required source experiment
is missing," so a caller could not tell a catalog gap from a verified
impossibility witness or from an unimplemented multi-node branch.

2.0 transport (T0–T10) is structural population transport, not statistical
prior/evidence transport (`antecedent.priors`). It must not grow a second
workflow engine or a new `analyze` matrix shape. The licensed trial-IPW cell
stays. New work lives in `antecedent.transport` and in core catalog / outcome
records that identify and later estimate consume.

Classical sID completeness is scoped to the experimental-information family of
Bareinboim & Pearl, “A General Algorithm for Deciding Transportability of Experimental Results” (arXiv:1312.7485v1). The API can
represent finite catalogs that the theorem does not treat as its input family.
Completeness must not be claimed for those catalogs, nor for later
z-/limited-experiment contracts.

## Decision

### Four layers

Keep these distinct. A type or function belongs to exactly one:

| Layer | Owns | Must not own |
| --- | --- | --- |
| Theoretical evidence availability | What the named theorem family treats as given (observed vars, allowed experiments, distribution family) | A particular supplied table, learner, or thread budget |
| Concrete supplied catalog | Environments, regimes, bindings, target sampling that the caller actually has | Completeness claims; implicit `do(A,B)` from `do(A)` and `do(B)` |
| Statistical provider | How a certified factor is estimated (tables in T4; learners in T6) | Identification search; physical execution policy |
| Physical execution | `ExecutionContext` budget, cancellation, kernels | Scientific meaning of a formula or catalog |

`antecedent-learn` (ADR 0023) is a statistical provider when T6 consumes it.
It is not a transport identifier.

### Theorem families

`TheoremScope` names the family, pinned reference/version, graph assumptions,
observed variables, allowed experiments, distribution family, query scope,
outcome guarantees, and implemented computation limits.

- **Classical sID** — completeness applies only to its stated mathematical
  input family. Implemented computation in T0–T1 is the existing sound subset
  plus target-only observational truncated factorization; multi-node c-component recursion is T3.
- **Finite-catalog search** — sound search over a supplied `EvidenceCatalog`.
  Missing a factor in one derivation is not a proof that no alternative
  catalog-supported formula exists. Not a completeness theorem.
- **Limited-experiment / z-contracts** — later families. Named so they cannot
  inherit sID completeness by accident.

### Typed outcomes

`TransportOutcome` is the caller-visible result. Callers match variants and
reason codes; they must not parse prose.

| Variant | Meaning |
| --- | --- |
| `Identified` | A sound implemented rule produced a formula |
| `ProvenNonTransportable` | A checked impossibility witness for the named theorem family |
| `NotCertified` | No implemented sound rule applies; historical meaning preserved |
| `InvalidInput` | Query, diagram, or catalog failed validation |
| `MissingEvidence` | A required available regime or provider input is absent |
| `MissingProvider` | A statistical provider required by a certified factor is absent |
| `UnsupportedEvaluator` | The formula is identified but this evaluator does not run it |
| `SupportFailure` | Stage-specific support coordinate failed |
| `NumericalFailure` | Evaluation failed numerically after a certified formula |
| `BudgetCancel` | Step, memory, recursion, or cancellation budget exhausted |

`NotCertified` is not renamed. Proven non-transportable is a new variant, not
a reinterpretation of old refusals. A missing experiment is `MissingEvidence`,
not `NotCertified`. Budget exhaustion never claims impossibility.

Each outcome carries a stable reason id and an optional factor/graph location.

### Support coordinates

Transport support is stage-specific. Do not invent an `analyze` capability to
fit the 1.10 matrix shape.

- **Identify** — graph class × evidence setting × target functional ×
  observation contract.
- **Evaluate** — evaluator × evidence setting × target functional.
- **Uncertainty** — uncertainty method × evaluator × observation contract.

The licensed trial-IPW `analyze` cell remains the one public analyze route
for transport. New identify/catalog surfaces are not analyze cells.

### Extension points

Stay on the ADR 0020 prepared handle and ADR 0022 claim/identity model.

- The optional catalog is encoded on `TransportQueryWire`, so environment
  coordinates, separate regime identities, intervention values, measurements,
  projections, bindings, weights, and target sampling participate in the query
  and identification identity. Flattened `source_experiments` is only a
  compatibility view, never a replacement for supplied catalog semantics.
- Python identify and prepared transport both forward the full catalog. Legacy
  queries without a catalog retain their existing wire representation.
- No durable wire-format break. New optional fields default to absent. A new
  `TransportIdentification` / wire variant (`MissingEvidence`) is additive:
  existing `Transportable` / `NotCertified` artifacts still decode. The
  licensed trial-IPW path requires available evidence before estimation.

### Existing licensed cell

`analyze` + explicit `Admg` + Frequentist + validation `none` + trial column
bindings is unchanged. Recursive factorization and `NotCertified` still refuse
estimation. `MissingEvidence` also refuses estimation. Do not route new
transport work through `analyze` except this cell.

## T2 addendum — identification-product identity

T2 lowers `TransportFormula` into the existing `antecedent-expr` arena. The
licensed trial-IPW cell's identification product no longer stores a stub
`E[Y | do(T)]` (`inspectable_do_expectation`). Inspect and the identification
digest now name the lowered Direct / Standardize / Recursive formula. That is
an intentional identity change for the licensed cell's identification product,
not a silent rewrite of the stub. Trial-IPW **estimation** still gates on
Direct / Standardize and does not evaluate the arena.

Old expression artifacts default `population=""`, `regime=None`, and an empty
optional derivation table. New symbolic interventions use an explicit wire
marker; they are never JSON NaN. New expression digests include their durable
derivation records. Legacy artifacts retain their absent metadata on reload. A
population or regime swap is a different `ExprId` and fails certificate bind.

## Consequences

- `antecedent-core` owns `TheoremScope`, `TransportOutcome`, stage-specific
  support coordinates, and the T1 catalog records (`Environment`,
  `EvidenceRegime`, `EvidenceCatalog`, bindings, target sampling).
- `TransportIdentifier` tries target-only identification before demanding a
  source experiment. A causally sufficient target graph permits its observational
  truncated factorization with an empty source catalog. An invariant graph with
  latent confounding does not manufacture a target experimental distribution.
  Missing required evidence is `MissingEvidence`.
- Reason codes for the new outcome kinds are registered in
  `parity/reason_codes.toml`. They are the closed vocabulary for later runtime
  refusals; identify still uses dotted certificate ids on the outcome record.
- Python `antecedent.transport` grows catalog dataclasses. Existing
  `TransportQuery(source_experiments=..., trial=...)` stays byte-compatible.
- T3–T10 (classical sID completeness, table evaluators,
  learner-backed uncertainty, multi-source search) remain out of this decision.
