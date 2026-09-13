# Identified multi-completion PAG envelope

**Suite path:** `conformance/estimate/pag_ate_envelope_identified`

Positive evidence for the static `Pag` × `AverageEffect` and `ConditionalEffect`
cells: a PAG whose MAG completions identify the effect by generalized
adjustment with visible edges out of `t`, at two materially different values.
The older `pag_ate_envelope` fixture stays as negative evidence (every
completion there has an invisible causal edge and is refused).

## Graph

`v o-o t o-o z o-o m`, `t -> y`, `m -> y`, `x -> y`. The completion sampler
keeps seven MAGs (64 endpoint assignments audited exhaustively):

| Orientation | Visible witness for `t`'s edges | Adjustment | Effect |
| --- | --- | --- | --- |
| `m -> z -> t -> v` | `z` (not adjacent to `y` or `v`) | `{z}` | direct |
| `z -> t -> v`, `z -> m` | `z` | `{z}` | direct |
| `z <-> t -> v`, `z -> m` | `z` | `{z}` | direct |
| `z -> t -> v`, `z <-> m` | `z` | `{z}` | direct |
| `v -> t -> z -> m` | `v` (not adjacent to `y` or `z`) | `{}` | total (via `z -> m -> y`) |
| `v <-> t -> z -> m` | `v` | `{}` | total |
| `v <- t -> z -> m` | none — no arrowhead into `t` | not identified | — |

Identified mass is 6 of 7 completions; the unidentified mass (1) is reported
and never renormalized into the effect. The Frequentist envelope is the
equal-weight mixture over the six identified completions: four estimate the
direct effect (≈0.350) and two the total effect (≈0.422), so the reported
number (0.3744) is a genuine mixture of disagreeing completions.

## Data

The contingency table is the whole frozen input. It is `round(2000 · P(cell))`
under the binary law in `expected.json` (`law`), which `reference.py`
regenerates and checks. The modifier `x` affects only `y`, so it is
pre-treatment in every completion and the ConditionalEffect envelope has the
same six identified completions.

## Pins and reference

`reference.py` is an independent numpy implementation (written from the
formulas, not from the Rust code) of the per-completion OLS fits, their
Frisch–Waugh / delta-method influence functions, and the joint-IF SE of the
frozen-weight mixture on the shared rows. It pins:

- Frequentist mass-weighted ATE `0.37443451148801365`, joint-IF SE
  `0.021664212100486056` (`linear.adjustment.ate`);
- Frequentist ConditionalEffect envelope `0.3743035897503231`, joint-IF SE
  `0.02138134948402955` (`conditional.linear.adjustment`; the influence
  includes the sampling variation of the modifier mean).

The Bayesian values are seeded output pins (conjugate, 64 draws, prior scale
10, seed 1): the envelope mean and SD of the draw-level mixture of
per-completion posteriors. That SD includes between-completion spread, so it
exceeds the Frequentist SE of the mixture functional.

Consumer: `crates/antecedent/tests/pag_identified_envelope_numeric_pins.rs`
(explicit and accepted PAG × `none`/`cheap`/`full`, Frequentist and Bayesian,
fresh and prepared paths, plus a check that the envelope's identified
completions and adjustment sets match `identified_completions`).

## Expected summary

Top-level keys: `bayesian, case, columns, conditional, contingency_table, frequentist, graph, identification, identified_completions, law, query, schema_version` (12 fields).
