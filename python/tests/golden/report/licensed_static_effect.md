# Causal analysis report

## Summary

- **Reading:** an executed claim
- **Query:** AverageEffect
- **Inference:** frequentist
- **Identifier:** none
- **Estimator:** linear.adjustment.ate
- **Route:** StaticAteDag
- **Claim kind:** point
- **Value:** 2
- **Support status:** licensed

- The claim publishes a value of kind point; the sections below state what it rests on and where it stops.
- The route is a licensed support-matrix cell.
- Calibration status is scope\_not\_assessed: the reported uncertainty carries no measured-coverage record.
- Identification status is nonparametrically\_identified.

## Claim

- **Claim id:** 0e5e3219fe545d5a178953dcd13ae093376a8c54c2ca950e39c0cef863c7f33b
- **Kind:** point
- **Value:** 2
- **Outcome units:** none
- **Identification domain:** identified
- **Support domain:** unknown
- **Evaluated domain:** evaluated
- **Execution:** 431eb95b6b88f79a166060ef435b0f059a70bd3cc265bf3100ca01d57aa4b171

## Identification

- **Graph class:** Dag
- **Structure source:** explicit
- **Identifier:** none
- **Estimator:** none
- **Resolved estimator:** linear.adjustment.ate
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

- snapshot:7acd4e323d010cf6c9a4fb5b578a6461f2d751ce3591de0d5f593f57dea1e9d5
- identification:96fba9fff73599a0e3cee2cba940842671350661995c3151c25535bafcb0ecbf
- program:20ffe89f49a215eea89f516d10461d1109c90405b844ee1328b656c92ccb46b1

## Uncertainty

| Source | Reported target | Omitted |
| --- | --- | --- |
| sampling | analytic\_se | no |

### Uncertainty the program must publish

- **claim\_kind:** point
- **reading:** sampling
- **realization:** frequentist

### Calibration

- **Calibration status:** scope\_not\_assessed
- **Governing record:** cov.average\_effect.dag.frequentist.analytic\_se.l95.linear\_adjustment\_analytic\_ci\_coverage
- **Reason:** coverage\_record\_not\_attesting
- **Measured rows (smallest):** 150
- **Measured rows (largest):** 600
- **Record dependence:** iid
- **Measured at:** c30e10304f4bd85a01ad81dd133a360ddc95c2f1
- **Nominal level:** 0.95
- **Observed coverage:** 0.9325
- **Basis query:** AverageEffect
- **Basis graph class:** Dag
- **Basis structure:** fixed
- **Basis modality:** tabular
- **Basis inference:** Frequentist
- **Basis estimator:** linear.adjustment.ate
- **Basis interval method:** analytic\_se
- **Basis standard error:** homoskedastic
- **Basis dependence:** iid
- **Basis posterior:** 
- **Basis functional:** all\_observed.mean
- **Basis level:** 0.95
- **Basis identification:** point
- **Basis rows:** 240
- **Basis successful replicates:** none

## Support and licence

- **Support status:** licensed
- **Modality:** tabular
- **Inference:** frequentist
- **Query kind:** AverageEffect
- **Rows:** 240
- **Functional:** all\_observed.mean

### Support slot

- **Matrix status:** licensed
- **Matrix coordinate:** AverageEffect:Dag:explicit:Frequentist:none
- **Empirical support:** unavailable:not\_evaluated

### Support cell of the program

- **Coordinate:** AverageEffect:Dag:explicit:Frequentist:none
- **Licensed:** yes
- **Status:** licensed

## Refusals and limits

- **Operation:** execute
- **Applicability:** licensed
- **Readiness:** executable
- **Coordinate:** AverageEffect:Dag:explicit:Frequentist:none

### Limits

- Calibration is scope\_not\_assessed (coverage\_record\_not\_attesting).

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
- **Program identity:** 5f95679de9c0fbdf41f81eb7a173ac770b7693f20d4aa8b2ac0e95502b601b5a
- **Sealed operation:** linear\_adjustment

### Scientific problem

- **data:** tabular
- **graph\_class:** Dag
- **plan\_id:** static\_ate
- **query\_variables:** t, y
- **source\_digest:** e3bdd7be223b5dd4ce9ba6bbe2db8b128dc18eaecbba941a41039e817e7c13c8
- **target.form:** target
- **target.identity:** c1f19633ed84499d5f80449f4aba6903d5ea55effd9ee985bae72097b6894379

### Identification product

- **Carried:** retained
- **Status:** nonparametrically\_identified
- **Installed at prepare:** yes
- **Product identity:** e76bc64db428c844176dea8584ef4a25db7b7aa1a7a564cbf338632aeb5113cb
- **Outcome:** identified
- **Claim class:** identified\_claim

| Estimand | Adjustment set | Instruments | Mediators |
| --- | --- | --- | --- |
| backdoor.adjustment | z | none | none |

| Required factor | Variables |
| --- | --- |
| factor.0 | z |
| factor.1 | y |

| Retained product | Status | Estimands | Completions | Truncated | Atoms | Identified mass | Unidentified mass |
| --- | --- | --- | --- | --- | --- | --- | --- |
| static | nonparametrically\_identified | backdoor.adjustment \[z\] | 0 | 0 | 0 | none | none |

### Statistical program

- **design.columns:** 3
- **design.direct:** yes
- **design.rows:** 240
- **design.source:** quoted
- **estimator:** linear.adjustment.ate
- **identifier:** backdoor.adjustment
- **identity:** 759d8f09f71e1e62b85564ebaa320c8b8428a1520e954075920373b89bfe6251
- **models\[0\].family:** ols.faer
- **models\[0\].inputs:** t, z
- **models\[0\].output:** y
- **models\[0\].role:** outcome
- **operators:** 3
- **realization:** frequentist

### Evidence and decision program

- **composition.kind:** single
- **identity:** 7428bd06bdc75756cfab8d0fb1ff32fa179774e737ece2f3e9ab14ea66136159
- **sources\[0\].digest:** e3bdd7be223b5dd4ce9ba6bbe2db8b128dc18eaecbba941a41039e817e7c13c8
- **sources\[0\].role:** observations

### Physical plan

- **backend.quotes\[0\].alternative:** direct\_prepared\_numeric\_views
- **backend.quotes\[0\].cost:** 2160
- **backend.quotes\[0\].resident\_bytes:** 5760
- **backend.quotes\[0\].work\_cells:** 1440
- **backend.quotes\[1\].alternative:** streamed\_reduction
- **backend.quotes\[1\].cost:** 2169
- **backend.quotes\[1\].resident\_bytes:** 5832
- **backend.quotes\[1\].work\_cells:** 1440
- **backend.quotes\[2\].alternative:** materialized\_design
- **backend.quotes\[2\].cost:** 3600
- **backend.quotes\[2\].resident\_bytes:** 11520
- **backend.quotes\[2\].work\_cells:** 2160
- **backend.quotes\[3\].alternative:** optional\_lazy\_preparation
- **backend.quotes\[3\].cost:** 2160
- **backend.quotes\[3\].resident\_bytes:** 5760
- **backend.quotes\[3\].work\_cells:** 1440
- **backend.rejected\[0\].alternative:** streamed\_reduction
- **backend.rejected\[0\].reason:** missing operator capability streamed\_reduction
- **backend.rejected\[1\].alternative:** materialized\_design
- **backend.rejected\[1\].reason:** cost 3600 is not lower than selected cost 2160
- **backend.rejected\[2\].alternative:** optional\_lazy\_preparation
- **backend.rejected\[2\].reason:** missing operator capability lazy\_preparation
- **backend.selected:** direct\_prepared\_numeric\_views
- **backend.selected\_cost:** 2160
- **design.columns:** 3
- **design.direct:** yes
- **design.rows:** 240
- **design.source:** quoted
- **memory.components\[0\].bytes:** 5760
- **memory.components\[0\].component:** encoded\_width
- **memory.components\[1\].bytes:** 90
- **memory.components\[1\].component:** masks
- **memory.components\[2\].bytes:** 0
- **memory.components\[2\].component:** design\_expansion
- **memory.components\[3\].bytes:** 0
- **memory.components\[3\].component:** learner\_fold\_copies
- **memory.components\[4\].bytes:** 7776
- **memory.components\[4\].component:** solver\_scratch
- **memory.components\[5\].bytes:** 0
- **memory.components\[5\].component:** retained\_scores\_draws
- **memory.components\[6\].bytes:** 1944
- **memory.components\[6\].component:** output
- **memory.components\[7\].bytes:** 69984
- **memory.components\[7\].component:** concurrent\_workers
- **memory.learner\_fold\_copies:** 0
- **memory.total:** 85554
- **memory.workers:** 10
- **operators\[0\].capability:** cpu
- **operators\[0\].designs:** 1
- **operators\[0\].device\_work:** no
- **operators\[0\].effect:** read
- **operators\[0\].output\_bytes:** 16000
- **operators\[0\].retained\_bytes:** 0
- **operators\[0\].scratch\_bytes:** 0
- **operators\[0\].sources:** 1
- **operators\[1\].capability:** cpu
- **operators\[1\].designs:** 1
- **operators\[1\].device\_work:** no
- **operators\[1\].effect:** fit
- **operators\[1\].output\_bytes:** 8
- **operators\[1\].retained\_bytes:** 16000
- **operators\[1\].scratch\_bytes:** 286976
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
- **operators\[3\].scratch\_bytes:** 5760
- **operators\[3\].sources:** 1
- **operators\[4\].capability:** cpu
- **operators\[4\].designs:** 1
- **operators\[4\].device\_work:** no
- **operators\[4\].effect:** scan
- **operators\[4\].output\_bytes:** 0
- **operators\[4\].retained\_bytes:** 0
- **operators\[4\].scratch\_bytes:** 17280
- **operators\[4\].sources:** 1
- **output\_bytes:** 16008
- **retained\_bytes:** 16000
- **scratch\_bytes:** 310016

## Identities

- **Target:** 716927442f91b16cbcc7381cce285105e7f5c5dd9a867660215fb6cc0d85083a
- **Identification:** 96fba9fff73599a0e3cee2cba940842671350661995c3151c25535bafcb0ecbf
- **Identification product:** 7f3a931cd7364b9acaefd685b977f177b064987a2153b15ba9e4ba49434f4d79
- **Program:** 20ffe89f49a215eea89f516d10461d1109c90405b844ee1328b656c92ccb46b1
- **Inference binding:** 122227db05904a9fdcabbcab561919a16691db0b8a445cffef22541a91240ba4
- **Observation:** ce6fe76d79b8e0d0cc58ebbbeb1a4dcac0b651b8c0d124f0a590f82102d46be8
- **Data snapshot:** 7acd4e323d010cf6c9a4fb5b578a6461f2d751ce3591de0d5f593f57dea1e9d5
- **Score reuse:** none
- **Target weights:** none

