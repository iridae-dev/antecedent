# External prior bank

Transfer previously fit posteriors into a new analysis as power / mixture
priors — without pretending priors create nonparametric identification.

Surveys are the usual motivating domain (product / context tags), but the
library speaks in **sources**, **targets**, and **designs**.

## Workflow

```text
catalog.filter(target) → rank(similarity)
  → map(effect|named params) → power-prior / mixture (α_k, w_k)
  → apply transport policy → PriorSet + assumptions
  → analyze(..., inference=Bayesian(prior_from=…))
  → prior PPC + prior sensitivity (α grid when banked)
```

1. Wrap posterior artifacts with `PriorSourceMeta` / `PriorSource`.
2. `PriorCatalog.compatible_with(...)` → accept / partial / reject reasons.
3. Callers supply **similarity scores**; the library ranks but does not invent
   domain similarity.
4. `compose_external_priors(...)` builds a `ComposedPrior` (optional
   `ConflictPolicy`, `TransportPolicy`).
5. Pass the composed prior as `Bayesian(prior_from=...)`.
6. With `refute="full"`, the ATE path attaches an **α-multiplier** sensitivity
   grid (not isotropic scales) when an external compose is present.

Python example: [`python/examples/prior_bank_surveys.py`](../python/examples/prior_bank_surveys.py).

## Invariants

- Priors are recorded as `PriorRestriction` assumptions; they **never** upgrade
  `IdentificationStatus`.
- Heterogeneous designs transfer at the **effect-functional** (or explicitly
  mapped parameter) level by default — not silent `coef_i → coef_i`.
  When `Bayesian(prior_from=artifact)` leaves `mapping` unset, hydrate chooses
  identical subspace for matching designs and `EffectFunctional` when layouts
  differ and an effect quantity exists; otherwise an explicit mapping is required.
- Unmapped / incompatible mass goes to a weakly informative baseline; never
  silently renormalized.
- Conflict can only **shrink** external weight (`α → 0`), never invent
  identification or increase α.
- Population / environment shifts require an explicit `TransportPolicy`; missing
  policy → structured `transport_policy_required`. Pass `prior_sources=` (or
  `source_populations=`) with `target_population=` so compose reads
  `tags["population"]` via `populations_from_prior_sources` — callers need not
  thread population tags manually when catalog meta is available.

## What callers must supply

| Concern | Owner |
|---------|--------|
| Similarity / ranking scores | Caller |
| Product / context / taxonomy tags | Caller conventions on `tags` |
| Max-trust / governance of the bank | Caller |
| Compatibility, mapping, α/weights, transport assumptions, diagnostics | Library |

Exact tag keys on `compatible_with(tags=...)` are hard filters. Soft similarity
belongs in `catalog.rank(scores=...)`.

## Supported transport policies

Documented in parity `bayes.prior_bank.transport`:

- `InvariantConditionalOutcome` — `P(Y|do(T),X)` stable across populations
- `InvariantEffectModifiers`
- `InvariantPropensity` (without transport weights → α forced to 0)

Unsupported environment / unidentified shifts are **not claimed**.

## Gates

```bash
bash scripts/gate_bayesian.sh
```

Runs `conformance/bayesian/prior_bank_*` plus `python/tests/test_prior_bank.py`.
