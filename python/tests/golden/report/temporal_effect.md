# Causal analysis report

## Summary

- **Reading:** an executed claim
- **Query:** PulseEffect
- **Inference:** frequentist
- **Identifier:** none
- **Estimator:** temporal.linear.adjustment
- **Route:** TemporalEffectDag
- **Claim kind:** point
- **Value:** 0.8976448135
- **Support status:** licensed

- The claim publishes a value of kind point; the sections below state what it rests on and where it stops.
- The route is a licensed support-matrix cell.
- Calibration status is scope\_not\_assessed: the reported uncertainty carries no measured-coverage record.
- Identification status is nonparametrically\_identified.

## Claim

- **Claim id:** 81537b4bbc2e6baac55e52292a0081f7fcf8239605604374a09ea44f126eadba
- **Kind:** point
- **Value:** 0.8976448135
- **Outcome units:** none
- **Identification domain:** identified
- **Support domain:** unknown
- **Evaluated domain:** evaluated
- **Execution:** 2d1dc74d9ee93138dc130eb6aa8f4182205eb7b729c50b3cad98e184408d76a0

## Identification

- **Graph class:** TemporalDag
- **Structure source:** explicit
- **Identifier:** none
- **Estimator:** none
- **Resolved estimator:** temporal.linear.adjustment
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

2 obligations are recorded; 2 remain unresolved.

| Id | Scope | Kind | Status | Source | Required check | Description |
| --- | --- | --- | --- | --- | --- | --- |
| assumption.0 | program | graphical\_implication | declared | algorithm\_default:backdoor | none | causal\_markov |
| assumption.1 | program | graphical\_implication | declared | algorithm\_default:temporal.backdoor.unfolded | none | stationarity |

### Assumptions the identification relies on

| Assumption | Status |
| --- | --- |
| causal\_markov | declared |
| stationarity | declared |

## Diagnostics and evidence

### Evidence the claim references

- snapshot:d9896080f5dcee6c44004e70185b5034c2836f6aadb73bc6f0fae14fd1debd58
- identification:32ee7fe95e290cfb06cc6e453763e356a1509c373d4adaf58ff734c18524b1b1
- program:903d5c382ecf74a8b732fc58ff38a4f8861d0dc061f0f72c5d47d3283774ff97

## Uncertainty

| Source | Reported target | Omitted |
| --- | --- | --- |
| sampling | bootstrap\_se | no |

### Uncertainty the program must publish

- **claim\_kind:** point
- **reading:** sampling
- **realization:** frequentist
- **resampling.method:** block\_bootstrap
- **resampling.replicates:** 199
- **resampling.unit:** calendar\_block

### Calibration

- **Calibration status:** scope\_not\_assessed
- **Governing record:** cov.pulse\_effect.temporal\_dag.frequentist.circular\_block\_se.l95.practitioner\_temporal\_pulse\_h1\_interval
- **Reason:** coverage\_record\_not\_attesting
- **Measured rows (smallest):** 319
- **Measured rows (largest):** 4159
- **Record dependence:** circular\_block:single\_window
- **Measured at:** 70a71dd33bc55cbc2398b0f91c4a2c797afad618
- **Nominal level:** 0.95
- **Observed coverage:** 0.938
- **Basis query:** PulseEffect
- **Basis graph class:** TemporalDag
- **Basis structure:** fixed
- **Basis modality:** series
- **Basis inference:** Frequentist
- **Basis estimator:** temporal.linear.adjustment
- **Basis interval method:** circular\_block\_se
- **Basis standard error:** 
- **Basis dependence:** circular\_block:single\_window
- **Basis posterior:** 
- **Basis functional:** pulse.h1.all\_observed
- **Basis level:** 0.95
- **Basis identification:** point
- **Basis rows:** 399
- **Basis successful replicates:** 199

## Support and licence

- **Support status:** licensed
- **Modality:** series
- **Inference:** frequentist
- **Query kind:** PulseEffect
- **Rows:** 400
- **Functional:** pulse.h1.all\_observed

### Support slot

- **Matrix status:** licensed
- **Matrix coordinate:** PulseEffect:TemporalDag:explicit:Frequentist:none
- **Empirical support:** unavailable:not\_evaluated

### Support cell of the program

- **Coordinate:** PulseEffect:TemporalDag:explicit:Frequentist:none
- **Licensed:** yes
- **Status:** licensed

## Refusals and limits

- **Operation:** execute
- **Applicability:** licensed
- **Readiness:** executable
- **Coordinate:** PulseEffect:TemporalDag:explicit:Frequentist:none

### Limits

- Calibration is scope\_not\_assessed (coverage\_record\_not\_attesting).

### Blockers and refusals

The operation report records no blocker.

### Unmet obligations

| Id | Scope | Kind | Status | Source | Required check | Description |
| --- | --- | --- | --- | --- | --- | --- |
| assumption.0 | program | graphical\_implication | declared | algorithm\_default:backdoor | none | causal\_markov |
| assumption.1 | program | graphical\_implication | declared | algorithm\_default:temporal.backdoor.unfolded | none | stationarity |

- **Layers the premises preserve:** target, identification, program, support
- **Layers the premises invalidate:** none

This report restates the contract. It computes no estimate, grants no licence, and a status recorded here is the status the records carry.

## Program plan

- **Route:** TemporalEffectDag
- **Family:** temporal
- **Stage:** prepared
- **Depth:** complete
- **Program identity:** b72447818dbaf6e76771af2a790494a7ef0b45b13872f0422ce528d8ea3c1d90
- **Sealed operation:** temporal\_dag\_effect

### Scientific problem

- **data:** temporal
- **graph\_class:** TemporalDag
- **plan\_id:** temporal\_effect
- **query\_variables:** pressure, defect
- **target.form:** target
- **target.identity:** 202838a6681237bf6a9adebb778f4bdaf8ddbcc1479710c764b707c3872f2ba6

### Identification product

- **Carried:** retained
- **Status:** nonparametrically\_identified
- **Installed at prepare:** yes

| Retained product | Status | Estimands | Completions | Truncated | Atoms | Identified mass | Unidentified mass |
| --- | --- | --- | --- | --- | --- | --- | --- |
| temporal | nonparametrically\_identified | temporal.backdoor.unfolded \[none\] | 0 | 0 | 0 | none | none |

### Statistical program

- **design.reason:** the lagged adjustment design is assembled from the unfolded sample at execution
- **design.source:** not\_design\_based
- **estimator:** temporal.linear.adjustment
- **identifier:** temporal.backdoor.unfolded
- **identity:** a6768793b0dd74f93230471aac150813dc924022c4b6221ef6b42ea8e6f6c849
- **models\[0\].family:** temporal.linear.adjustment
- **models\[0\].inputs:** pressure
- **models\[0\].output:** defect
- **models\[0\].role:** outcome
- **operators:** 9
- **realization:** frequentist
- **resampling.method:** block\_bootstrap
- **resampling.replicates:** 199
- **resampling.unit:** calendar\_block

### Evidence and decision program

- **composition.kind:** single
- **identity:** 99dec0d4a3c47302aa2f4f28dc40b3ccfcab003e3737b2601341a639ef0ef027
- **sources\[0\].role:** observations

### Physical plan

- **backend.quotes\[0\].alternative:** direct\_prepared\_numeric\_views
- **backend.quotes\[0\].cost:** 161600
- **backend.quotes\[0\].resident\_bytes:** 6400
- **backend.quotes\[0\].work\_cells:** 160800
- **backend.quotes\[1\].alternative:** streamed\_reduction
- **backend.quotes\[1\].cost:** 320804
- **backend.quotes\[1\].resident\_bytes:** 6432
- **backend.quotes\[1\].work\_cells:** 320000
- **backend.quotes\[2\].alternative:** materialized\_design
- **backend.quotes\[2\].cost:** 163200
- **backend.quotes\[2\].resident\_bytes:** 12800
- **backend.quotes\[2\].work\_cells:** 161600
- **backend.quotes\[3\].alternative:** optional\_lazy\_preparation
- **backend.quotes\[3\].cost:** 161600
- **backend.quotes\[3\].resident\_bytes:** 6400
- **backend.quotes\[3\].work\_cells:** 160800
- **backend.rejected\[0\].alternative:** direct\_prepared\_numeric\_views
- **backend.rejected\[0\].reason:** missing operator capability prepared\_numeric\_view
- **backend.rejected\[1\].alternative:** streamed\_reduction
- **backend.rejected\[1\].reason:** missing operator capability streamed\_reduction
- **backend.rejected\[2\].alternative:** optional\_lazy\_preparation
- **backend.rejected\[2\].reason:** missing operator capability lazy\_preparation
- **backend.selected:** materialized\_design
- **backend.selected\_cost:** 163200
- **design.reason:** the lagged adjustment design is assembled from the unfolded sample at execution
- **design.source:** not\_design\_based
- **memory.components\[0\].bytes:** 6400
- **memory.components\[0\].component:** encoded\_width
- **memory.components\[1\].bytes:** 100
- **memory.components\[1\].component:** masks
- **memory.components\[2\].bytes:** 0
- **memory.components\[2\].component:** design\_expansion
- **memory.components\[3\].bytes:** 0
- **memory.components\[3\].component:** learner\_fold\_copies
- **memory.components\[4\].bytes:** 9648
- **memory.components\[4\].component:** solver\_scratch
- **memory.components\[5\].bytes:** 1592
- **memory.components\[5\].component:** retained\_scores\_draws
- **memory.components\[6\].bytes:** 4808
- **memory.components\[6\].component:** output
- **memory.components\[7\].bytes:** 86832
- **memory.components\[7\].component:** concurrent\_workers
- **memory.learner\_fold\_copies:** 0
- **memory.total:** 109380
- **memory.workers:** 10
- **operators\[0\].capability:** cpu
- **operators\[0\].designs:** 1
- **operators\[0\].device\_work:** no
- **operators\[0\].effect:** read
- **operators\[0\].output\_bytes:** 6400
- **operators\[0\].retained\_bytes:** 0
- **operators\[0\].scratch\_bytes:** 0
- **operators\[0\].sources:** 1
- **operators\[1\].capability:** cpu
- **operators\[1\].designs:** 1
- **operators\[1\].device\_work:** no
- **operators\[1\].effect:** fit
- **operators\[1\].output\_bytes:** 8
- **operators\[1\].retained\_bytes:** 6400
- **operators\[1\].scratch\_bytes:** 96480
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
- **operators\[3\].scratch\_bytes:** 6400
- **operators\[3\].sources:** 1
- **operators\[4\].capability:** cpu
- **operators\[4\].designs:** 1
- **operators\[4\].device\_work:** no
- **operators\[4\].effect:** scan
- **operators\[4\].output\_bytes:** 0
- **operators\[4\].retained\_bytes:** 0
- **operators\[4\].scratch\_bytes:** 19200
- **operators\[4\].sources:** 1
- **output\_bytes:** 6408
- **retained\_bytes:** 6400
- **scratch\_bytes:** 122080

## Identities

- **Target:** 2d59f60479809144a79d268cc4f9badb5db40c24552686d65dbbf9edb3640b96
- **Identification:** 32ee7fe95e290cfb06cc6e453763e356a1509c373d4adaf58ff734c18524b1b1
- **Identification product:** 0b50b6a391a63ad2c167b75e36ecf0f1203cdc61c8a3d822d510a3d6a64faccf
- **Program:** 903d5c382ecf74a8b732fc58ff38a4f8861d0dc061f0f72c5d47d3283774ff97
- **Inference binding:** c0b56df81bae1a8835154001766ba818ecda7b1aa1c180ef71640e6b9db4d029
- **Observation:** f27aa98772837e28c8c83c5a99a36ed0443730553037d310c7d2d063f044050f
- **Data snapshot:** d9896080f5dcee6c44004e70185b5034c2836f6aadb73bc6f0fae14fd1debd58
- **Score reuse:** none
- **Target weights:** none

