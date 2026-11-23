# Causal analysis report

## Summary

- **Reading:** an executed claim
- **Query:** AverageEffect
- **Inference:** frequentist
- **Identifier:** none
- **Estimator:** linear.adjustment.ate
- **Route:** StaticAteDag
- **Claim kind:** point
- **Value:** 2.10294276347
- **Support status:** supported

- The claim publishes a value of kind point; the sections below state what it rests on and where it stops.
- The route is a supported support-matrix cell.
- Calibration status is outside\_measured\_scope: the reported uncertainty carries no measured-coverage record.
- Identification status is nonparametrically\_identified.

## Claim

- **Claim id:** d092090a8595d4748bee2b394a60b9341f1dc01ee52acaea314af6b00a8c15a7
- **Kind:** point
- **Value:** 2.10294276347
- **Outcome units:** none
- **Identification domain:** identified
- **Support domain:** unknown
- **Evaluated domain:** evaluated
- **Execution:** f665209785a90a37b045a311e565178ed10b6d06dfb133286c10aadcd22e5e6c

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

- snapshot:e167b3c755118a57186bc568da11f2540f81234b1a96682827b3726587ecb212
- identification:c6d909e9ff3b13ffb992906dede96d440f43ca82ee8b1deec9a23c7069e928c6
- program:c33f620a69b419fe73af38578aea1b5a6f2a2da4d45240deab66d3f779f792db

## Uncertainty

| Source | Reported target | Omitted |
| --- | --- | --- |
| sampling | analytic\_se | no |

### Uncertainty the program must publish

- **claim\_kind:** point
- **reading:** sampling
- **realization:** frequentist

### Calibration

- **Calibration status:** outside\_measured\_scope
- **Governing record:** cov.average\_effect.dag.frequentist.analytic\_se.l95.practitioner\_linear\_adjustment\_analytic\_interval.v2\_validation\_none
- **Reason:** sample\_size\_outside\_measured\_range
- **Measured rows (smallest):** 1000
- **Measured rows (largest):** 100000
- **Record dependence:** iid
- **Measured at:** 434787cc9d8b641f42d9970eb052eb69cc8cf772
- **Nominal level:** 0.95
- **Observed coverage:** 0.945
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
- **Basis rows:** 128
- **Basis successful replicates:** none

## Support

- **Support status:** supported
- **Modality:** tabular
- **Inference:** frequentist
- **Query kind:** AverageEffect
- **Rows:** 128
- **Functional:** all\_observed.mean

### Support slot

- **Matrix status:** supported
- **Matrix coordinate:** AverageEffect:Dag:explicit:Frequentist:none
- **Empirical support:** unavailable:not\_evaluated

### Support cell of the program

- **Coordinate:** AverageEffect:Dag:explicit:Frequentist:none
- **Supported:** yes
- **Status:** supported

## Refusals and limits

- **Operation:** execute
- **Applicability:** supported
- **Readiness:** executable
- **Coordinate:** AverageEffect:Dag:explicit:Frequentist:none

### Limits

- Calibration is outside\_measured\_scope (sample\_size\_outside\_measured\_range).

### Blockers and refusals

The operation report records no blocker.

### Unmet obligations

| Id | Scope | Kind | Status | Source | Required check | Description |
| --- | --- | --- | --- | --- | --- | --- |
| assumption.0 | program | graphical\_implication | declared | algorithm\_default:backdoor | none | causal\_markov |

- **Layers the premises preserve:** target, identification, program, support
- **Layers the premises invalidate:** none

This report restates the contract. It computes no estimate, grants no support, and a status recorded here is the status the records carry.

## Program plan

- **Route:** StaticAteDag
- **Family:** static\_effect
- **Stage:** prepared
- **Depth:** complete
- **Program identity:** 6ac8dd815e4024dfb820b5e1f111075749ffb9adc6590eb1415e5a47848df1de
- **Sealed operation:** linear\_adjustment

### Scientific problem

- **data:** tabular
- **graph\_class:** Dag
- **plan\_id:** static\_ate
- **query\_variables:** t, y
- **source\_digest:** a509218488dd0fe7c3510e7968827ea2d4fa054fe28f494c511c571522eb380c
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
- **design.rows:** 128
- **design.source:** quoted
- **estimator:** linear.adjustment.ate
- **identifier:** backdoor.adjustment
- **identity:** d16c7aff1c83dc4c178095ea67494564b4e0bf2d5c0cae9c1f0d3807c80e7734
- **models\[0\].family:** ols.faer
- **models\[0\].inputs:** t, z
- **models\[0\].output:** y
- **models\[0\].role:** outcome
- **operators:** 3
- **realization:** frequentist

### Evidence and decision program

- **composition.kind:** single
- **identity:** d04523ce371c1bf7c21d12c226308f2bdbb419646dddd9d090c9548bb2ef9ff6
- **sources\[0\].digest:** a509218488dd0fe7c3510e7968827ea2d4fa054fe28f494c511c571522eb380c
- **sources\[0\].role:** observations

### Physical plan

- **backend.quotes\[0\].alternative:** direct\_prepared\_numeric\_views
- **backend.quotes\[0\].cost:** 1152
- **backend.quotes\[0\].resident\_bytes:** 3072
- **backend.quotes\[0\].work\_cells:** 768
- **backend.quotes\[1\].alternative:** streamed\_reduction
- **backend.quotes\[1\].cost:** 1161
- **backend.quotes\[1\].resident\_bytes:** 3144
- **backend.quotes\[1\].work\_cells:** 768
- **backend.quotes\[2\].alternative:** materialized\_design
- **backend.quotes\[2\].cost:** 1920
- **backend.quotes\[2\].resident\_bytes:** 6144
- **backend.quotes\[2\].work\_cells:** 1152
- **backend.quotes\[3\].alternative:** optional\_lazy\_preparation
- **backend.quotes\[3\].cost:** 1152
- **backend.quotes\[3\].resident\_bytes:** 3072
- **backend.quotes\[3\].work\_cells:** 768
- **backend.rejected\[0\].alternative:** streamed\_reduction
- **backend.rejected\[0\].reason:** missing operator capability streamed\_reduction
- **backend.rejected\[1\].alternative:** materialized\_design
- **backend.rejected\[1\].reason:** cost 1920 is not lower than selected cost 1152
- **backend.rejected\[2\].alternative:** optional\_lazy\_preparation
- **backend.rejected\[2\].reason:** missing operator capability lazy\_preparation
- **backend.selected:** direct\_prepared\_numeric\_views
- **backend.selected\_cost:** 1152
- **design.columns:** 3
- **design.direct:** yes
- **design.rows:** 128
- **design.source:** quoted
- **memory.components\[0\].bytes:** 3072
- **memory.components\[0\].component:** encoded\_width
- **memory.components\[1\].bytes:** 48
- **memory.components\[1\].component:** masks
- **memory.components\[2\].bytes:** 0
- **memory.components\[2\].component:** design\_expansion
- **memory.components\[3\].bytes:** 0
- **memory.components\[3\].component:** learner\_fold\_copies
- **memory.components\[4\].bytes:** 4192
- **memory.components\[4\].component:** solver\_scratch
- **memory.components\[5\].bytes:** 0
- **memory.components\[5\].component:** retained\_scores\_draws
- **memory.components\[6\].bytes:** 1048
- **memory.components\[6\].component:** output
- **memory.components\[7\].bytes:** 0
- **memory.components\[7\].component:** concurrent\_workers
- **memory.learner\_fold\_copies:** 0
- **memory.total:** 8360
- **memory.workers:** 1
- **operators\[0\].capability:** cpu
- **operators\[0\].designs:** 1
- **operators\[0\].device\_work:** no
- **operators\[0\].effect:** read
- **operators\[0\].output\_bytes:** 10624
- **operators\[0\].retained\_bytes:** 0
- **operators\[0\].scratch\_bytes:** 0
- **operators\[0\].sources:** 1
- **operators\[1\].capability:** cpu
- **operators\[1\].designs:** 1
- **operators\[1\].device\_work:** no
- **operators\[1\].effect:** fit
- **operators\[1\].output\_bytes:** 8
- **operators\[1\].retained\_bytes:** 10624
- **operators\[1\].scratch\_bytes:** 215296
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
- **operators\[3\].scratch\_bytes:** 3072
- **operators\[3\].sources:** 1
- **operators\[4\].capability:** cpu
- **operators\[4\].designs:** 1
- **operators\[4\].device\_work:** no
- **operators\[4\].effect:** scan
- **operators\[4\].output\_bytes:** 0
- **operators\[4\].retained\_bytes:** 0
- **operators\[4\].scratch\_bytes:** 9216
- **operators\[4\].sources:** 1
- **output\_bytes:** 10632
- **retained\_bytes:** 10624
- **scratch\_bytes:** 227584

## Identities

- **Target:** 4191b15a90102d684a46e203b41d22015a7f8dbbcf3217bae22f46c607ca315a
- **Identification:** c6d909e9ff3b13ffb992906dede96d440f43ca82ee8b1deec9a23c7069e928c6
- **Identification product:** 7f3a931cd7364b9acaefd685b977f177b064987a2153b15ba9e4ba49434f4d79
- **Program:** c33f620a69b419fe73af38578aea1b5a6f2a2da4d45240deab66d3f779f792db
- **Inference binding:** 122227db05904a9fdcabbcab561919a16691db0b8a445cffef22541a91240ba4
- **Observation:** 3b529d1e0fdc0de92a0de9213d1320e0b655fc151538998bdc3ae60e70be5e18
- **Data snapshot:** e167b3c755118a57186bc568da11f2540f81234b1a96682827b3726587ecb212
- **Score reuse:** 5b4e2a401f0a3152f2e078d81cffe867c296724e87932453261823ff7166b020
- **Target weights:** none

