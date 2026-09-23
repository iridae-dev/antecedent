# Identify: ID completeness cases

Hand-derived Shpitser–Pearl ID worked examples on ADMGs, consumed by
`crates/antecedent-identify/tests/id_completeness.rs`
(`frozen_id_cases_keep_status_lines_certificate_and_truth`).

`expected.json` records, per case, the graph, the query, the status, the
sequence of published ID lines the derivation takes, and for non-identifiable
queries the hedge node sets `(F, F')`. It stores no numbers: the consuming test
builds a positive binary latent-variable SCM on each graph, enumerates the
observational joint and the interventional law exactly, evaluates the
identified functional on the observational joint alone, and requires agreement
to `1e-10` for every value of every free variable.

| Case | Why it is here |
| --- | --- |
| `napkin` | Smallest graph whose derivation is line 7 → line 2 → line 6: the C-factor carried out of line 7 is marginalized and then factorized again, so the answer is a ratio of marginals of that C-factor. Free in `z`. |
| `nested_napkin` | A napkin whose inner graph is again a napkin: the second C-factor is built from conditionals of an already marginalized law (a ratio inside a ratio). |
| `frontdoor_admg` | Line 7 → line 2 → line 1, where no ratio is needed. |
| `frontdoor_confounded_mediator` | Front-door plus `M <-> Y`: one district, a hedge. |
| `bow_arc` | The minimal hedge. |
| `joint_treatment_confounded_pair` | Joint `do(x1, x2)` whose hedge sits on a strict subset of the treatments. |

## Sweep (same test file)

The completeness evidence is the contract checked over whole graph families,
not these six cases:

- `every_small_admg_query_is_identified_correctly_or_refuted_by_a_verified_hedge`:
  every ADMG on 2, 3 and 4 nodes whose directed edges respect a fixed order
  (4 + 64 + 4096 graphs) × every ordered (treatment, outcome) pair = 49 544
  queries, two random positive binary parameterizations per graph. Each query
  is either identified and equal to the enumerated `P(y | do(x))` to `1e-9`
  (40 381 queries, contrast route re-checked on all ≤3-node queries and on every
  ratio-form derivation), or not identified with a certificate that
  `HedgeCertificate::verify` accepts for the original query (9 163 queries).
  An `Err` fails the test.
- `sampled_larger_admgs_with_joint_queries_keep_the_two_way_contract`: 1 500
  seeded 5–6 node ADMGs with treatment and outcome sets of size 1–2.
- `idc_matches_exact_conditional_intervention_on_every_three_node_admg`: IDC
  over all 64 three-node ADMGs × 6 role assignments against
  `P_x(y, z) / P_x(z)`.

Caps: binary variables, ≤4 nodes exhaustively (≤6 sampled), single treatment ×
single outcome exhaustively (sets of ≤2 sampled). The hedge check is
existential over edge subsets because the certificate names node sets only.
