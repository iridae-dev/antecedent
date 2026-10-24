# Causal analysis report

## Summary

- **Reading:** an executed claim
- **Query:** AverageEffect
- **Inference:** bayesian
- **Identifier:** none
- **Estimator:** bayesian.gcomp
- **Route:** StaticAteDag
- **Claim kind:** point
- **Value:** 2.0322712926
- **Support status:** licensed

- The claim publishes a value of kind point; the sections below state what it rests on and where it stops.
- The route is a licensed support-matrix cell.
- Calibration status is unavailable: the reported uncertainty carries no measured-coverage record.
- Identification status is nonparametrically\_identified.

## Claim

- **Claim id:** 15136bc597e8134567177018b9481071a0447d18c3e196bbd6865b88efbe2d10
- **Kind:** point
- **Value:** 2.0322712926
- **Outcome units:** none
- **Identification domain:** identified
- **Support domain:** unknown
- **Evaluated domain:** evaluated
- **Execution:** 8803f9ddaa94d51aa42daff15db1617ee0414860d03e9825dd54714d229958f6

## Identification

- **Graph class:** Dag
- **Structure source:** explicit
- **Identifier:** none
- **Estimator:** bayesian.gcomp
- **Resolved estimator:** bayesian.gcomp
- **Discovery algorithm:** none

- **Status:** nonparametrically\_identified
- **Identified mass:** 1
- **Unidentified mass:** 0
- **Unevaluable mass:** 0
- **Incomplete-search mass:** 0
- **Mass covers the full class:** yes
- **Weight basis:** none
- **Search capped:** no

## Assumptions

1 obligations are recorded; 1 remain unresolved.

| Id | Scope | Kind | Status | Source | Required check | Description |
| --- | --- | --- | --- | --- | --- | --- |
| assumption.0 | program | graphical\_implication | declared | algorithm\_default:backdoor | none | causal\_markov |

### Assumptions the identification relies on

| Assumption | Status |
| --- | --- |
| causal\_markov | declared |

## Diagnostics and evidence

### Evidence the claim references

- snapshot:439cb7aac1ca19021c183e342c677bda79e74a23a234acb023f0347ad1007c5d
- identification:90c1dcd8dfb83272abc040fe94c2dca6453ecf272596a99488135c410a9c0f8b
- program:a69cdf07862250de813155a83fa0ab68813df14b3592e0a1f56773b20e749acc

## Uncertainty

| Source | Reported target | Omitted |
| --- | --- | --- |
| sampling | analytic\_se | no |
| parameter | posterior | no |

### Uncertainty the program must publish

- **claim\_kind:** point
- **posterior.algorithm:** conjugate
- **posterior.draws:** 96
- **posterior.prior:** isotropic\_normal(scale=30)
- **reading:** posterior
- **realization:** bayesian

### Calibration

- **Calibration status:** unavailable
- **Governing record:** none
- **Reason:** estimator\_grid\_not\_measured
- **Measured rows (smallest):** none
- **Measured rows (largest):** none
- **Record dependence:** none
- **Measured at:** none
- **Nominal level:** none
- **Observed coverage:** none
- **Basis query:** AverageEffect
- **Basis graph class:** Dag
- **Basis structure:** fixed
- **Basis modality:** tabular
- **Basis inference:** Bayesian
- **Basis estimator:** bayesian.gcomp
- **Basis interval method:** posterior\_quantile
- **Basis standard error:** 
- **Basis dependence:** iid
- **Basis posterior:** conjugate\_gaussian.gaussian\_identity.prior\_scale=30
- **Basis functional:** all\_observed.mean
- **Basis level:** 0.95
- **Basis identification:** point
- **Basis rows:** 512
- **Basis successful replicates:** none

## Support and licence

- **Support status:** licensed
- **Modality:** tabular
- **Inference:** bayesian
- **Query kind:** AverageEffect
- **Rows:** 512
- **Functional:** all\_observed.mean
- **Posterior construction:** conjugate\_gaussian.gaussian\_identity.prior\_scale=30
- **Posterior draws requested:** 96

### Support slot

- **Matrix status:** licensed
- **Matrix coordinate:** AverageEffect:Dag:explicit:Bayesian:none
- **Empirical support:** unavailable:not\_evaluated

### Support cell of the program

- **Coordinate:** AverageEffect:Dag:explicit:Bayesian:none
- **Licensed:** yes
- **Status:** licensed

## Refusals and limits

- **Operation:** execute
- **Applicability:** licensed
- **Readiness:** executable
- **Coordinate:** AverageEffect:Dag:explicit:Bayesian:none

### Limits

- Calibration is unavailable (estimator\_grid\_not\_measured).

### Blockers and refusals

The operation report records no blocker.

### Unmet obligations

| Id | Scope | Kind | Status | Source | Required check | Description |
| --- | --- | --- | --- | --- | --- | --- |
| assumption.0 | program | graphical\_implication | declared | algorithm\_default:backdoor | none | causal\_markov |

- **Layers the premises preserve:** target, identification, program, support
- **Layers the premises invalidate:** none

This report restates the contract. It computes no estimate, grants no licence, and a status recorded here is the status the records carry.

## Program plan

- **Route:** StaticAteDag
- **Family:** static\_effect
- **Stage:** prepared
- **Depth:** complete
- **Program identity:** 29dc309662efe1f5ea644d45c073d1b691c26cc890864bb23a510eea40d5a06a
- **Sealed operation:** bayesian\_gcomp

### Scientific problem

- **data:** tabular
- **graph\_class:** Dag
- **plan\_id:** static\_ate
- **query\_variables:** t, y
- **target.form:** target
- **target.identity:** c1f19633ed84499d5f80449f4aba6903d5ea55effd9ee985bae72097b6894379

### Identification product

- **Carried:** retained
- **Status:** nonparametrically\_identified
- **Installed at prepare:** yes

| Retained product | Status | Estimands | Completions | Truncated | Atoms | Identified mass | Unidentified mass |
| --- | --- | --- | --- | --- | --- | --- | --- |
| static | nonparametrically\_identified | backdoor.adjustment \[z\] | 0 | 0 | 0 | none | none |

### Statistical program

- **design.columns:** 3
- **design.direct:** yes
- **design.rows:** 512
- **design.source:** quoted
- **estimator:** bayesian.gcomp
- **identifier:** backdoor.adjustment
- **identity:** c60e345cc2025733e74355b99dff0a2c6222b32eec6f3f3916da86b5f488f632
- **models\[0\].family:** ols.faer
- **models\[0\].inputs:** t, z
- **models\[0\].output:** y
- **models\[0\].role:** outcome
- **operators:** 3
- **posterior.algorithm:** conjugate
- **posterior.draws:** 96
- **posterior.prior:** isotropic\_normal(scale=30)
- **realization:** bayesian

### Evidence and decision program

- **composition.kind:** single
- **identity:** 99dec0d4a3c47302aa2f4f28dc40b3ccfcab003e3737b2601341a639ef0ef027
- **sources\[0\].role:** observations

### Physical plan

- **backend.quotes\[0\].alternative:** direct\_prepared\_numeric\_views
- **backend.quotes\[0\].cost:** 4608
- **backend.quotes\[0\].resident\_bytes:** 12288
- **backend.quotes\[0\].work\_cells:** 3072
- **backend.quotes\[1\].alternative:** streamed\_reduction
- **backend.quotes\[1\].cost:** 4617
- **backend.quotes\[1\].resident\_bytes:** 12360
- **backend.quotes\[1\].work\_cells:** 3072
- **backend.quotes\[2\].alternative:** materialized\_design
- **backend.quotes\[2\].cost:** 7680
- **backend.quotes\[2\].resident\_bytes:** 24576
- **backend.quotes\[2\].work\_cells:** 4608
- **backend.quotes\[3\].alternative:** optional\_lazy\_preparation
- **backend.quotes\[3\].cost:** 4608
- **backend.quotes\[3\].resident\_bytes:** 12288
- **backend.quotes\[3\].work\_cells:** 3072
- **backend.rejected\[0\].alternative:** direct\_prepared\_numeric\_views
- **backend.rejected\[0\].reason:** missing operator capability prepared\_numeric\_view
- **backend.rejected\[1\].alternative:** streamed\_reduction
- **backend.rejected\[1\].reason:** missing operator capability streamed\_reduction
- **backend.rejected\[2\].alternative:** optional\_lazy\_preparation
- **backend.rejected\[2\].reason:** missing operator capability lazy\_preparation
- **backend.selected:** materialized\_design
- **backend.selected\_cost:** 7680
- **design.columns:** 3
- **design.direct:** yes
- **design.rows:** 512
- **design.source:** quoted
- **memory.components\[0\].bytes:** 12288
- **memory.components\[0\].component:** encoded\_width
- **memory.components\[1\].bytes:** 192
- **memory.components\[1\].component:** masks
- **memory.components\[2\].bytes:** 0
- **memory.components\[2\].component:** design\_expansion
- **memory.components\[3\].bytes:** 0
- **memory.components\[3\].component:** learner\_fold\_copies
- **memory.components\[4\].bytes:** 16480
- **memory.components\[4\].component:** solver\_scratch
- **memory.components\[5\].bytes:** 0
- **memory.components\[5\].component:** retained\_scores\_draws
- **memory.components\[6\].bytes:** 4120
- **memory.components\[6\].component:** output
- **memory.components\[7\].bytes:** 0
- **memory.components\[7\].component:** concurrent\_workers
- **memory.learner\_fold\_copies:** 0
- **memory.total:** 33080
- **memory.workers:** 1
- **operators\[0\].capability:** cpu
- **operators\[0\].designs:** 1
- **operators\[0\].device\_work:** no
- **operators\[0\].effect:** read
- **operators\[0\].output\_bytes:** 12288
- **operators\[0\].retained\_bytes:** 0
- **operators\[0\].scratch\_bytes:** 0
- **operators\[0\].sources:** 1
- **operators\[1\].capability:** cpu
- **operators\[1\].designs:** 1
- **operators\[1\].device\_work:** no
- **operators\[1\].effect:** fit
- **operators\[1\].output\_bytes:** 8
- **operators\[1\].retained\_bytes:** 12288
- **operators\[1\].scratch\_bytes:** 16480
- **operators\[1\].sources:** 1
- **operators\[2\].capability:** cpu
- **operators\[2\].designs:** 1
- **operators\[2\].device\_work:** no
- **operators\[2\].effect:** intervene
- **operators\[2\].output\_bytes:** 0
- **operators\[2\].retained\_bytes:** 0
- **operators\[2\].scratch\_bytes:** 0
- **operators\[2\].sources:** 1
- **operators\[3\].capability:** cpu
- **operators\[3\].designs:** 0
- **operators\[3\].device\_work:** no
- **operators\[3\].effect:** select
- **operators\[3\].output\_bytes:** 0
- **operators\[3\].retained\_bytes:** 0
- **operators\[3\].scratch\_bytes:** 12288
- **operators\[3\].sources:** 1
- **operators\[4\].capability:** cpu
- **operators\[4\].designs:** 1
- **operators\[4\].device\_work:** no
- **operators\[4\].effect:** scan
- **operators\[4\].output\_bytes:** 0
- **operators\[4\].retained\_bytes:** 0
- **operators\[4\].scratch\_bytes:** 36864
- **operators\[4\].sources:** 1
- **output\_bytes:** 12296
- **retained\_bytes:** 12288
- **scratch\_bytes:** 65632

## Identities

- **Target:** 0e8cc7ffd6ab51ac2162a8525997413b0b2885693570d5b847b4bcaefe2cab28
- **Identification:** 90c1dcd8dfb83272abc040fe94c2dca6453ecf272596a99488135c410a9c0f8b
- **Identification product:** 7f3a931cd7364b9acaefd685b977f177b064987a2153b15ba9e4ba49434f4d79
- **Program:** a69cdf07862250de813155a83fa0ab68813df14b3592e0a1f56773b20e749acc
- **Inference binding:** 4aa359cfcc5437f528593e35dd4d50c367102d8e2a94e421c9b85051287a157e
- **Observation:** 55d73a79374fd7a00313b8db3294dd7d0beb9db0a71fcb319640b2bf15e67131
- **Data snapshot:** 439cb7aac1ca19021c183e342c677bda79e74a23a234acb023f0347ad1007c5d
- **Score reuse:** none
- **Target weights:** none

