# Causal analysis report

## Summary

- **Reading:** an executed claim
- **Query:** Unknown
- **Inference:** frequentist
- **Identifier:** none
- **Estimator:** sequential.crossfit\_contrast.wald
- **Route:** SequentialContrast
- **Claim kind:** point
- **Support status:** off\_axis

- The claim publishes a value of kind point; the sections below state what it rests on and where it stops.
- The query is off the support-matrix axis; no matrix licence applies to it.
- Calibration status is unavailable: the reported uncertainty carries no measured-coverage record.
- Identification status is nonparametrically\_identified.

## Claim

- **Claim id:** 2a97d0a2f5e9285199caf07945112ecf8b03d25ff8e2170af27a4f67bfe87a6c
- **Kind:** point
- **Value:** none
- **Outcome units:** none
- **Identification domain:** identified
- **Support domain:** unknown
- **Evaluated domain:** evaluated
- **Execution:** 8803f9ddaa94d51aa42daff15db1617ee0414860d03e9825dd54714d229958f6

## Identification

- **Graph class:** RandomizedTrial
- **Structure source:** randomized\_trial
- **Identifier:** none
- **Estimator:** none
- **Resolved estimator:** sequential.crossfit\_contrast.wald
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

0 obligations are recorded; 0 remain unresolved.

### Assumptions the identification relies on

| Assumption | Status |
| --- | --- |
| consistency | declared |
| no\_interference | declared |
| positivity | declared |
| custom:sequential\_conditional\_exchangeability | declared |
| custom:no\_unmeasured\_second\_decision\_confounder | declared |
| custom:nuisance\_product\_rate | declared |

## Diagnostics and evidence

### Evidence the claim references

- snapshot:70480c58ee5f1a795d90ac591624509b73ae22077b2cbaa91a40999a05a4e578
- identification:b50bd7459618af7272fc7218a6716baf35e481d61770a70bb902f5e355d17f28
- program:71523fc7a0d1380a0c873a9c99a92baccae1d388dd99e9b87e5467a0b80fe247

## Uncertainty

The uncertainty slot is unavailable: omitted.

### Uncertainty the program must publish

- **claim\_kind:** point
- **reading:** sampling
- **realization:** frequentist

### Calibration

- **Calibration status:** unavailable
- **Governing record:** none
- **Reason:** calibration\_basis\_missing
- **Measured rows (smallest):** none
- **Measured rows (largest):** none
- **Record dependence:** none
- **Measured at:** none
- **Nominal level:** none
- **Observed coverage:** none

## Support and licence

- **Support status:** off\_axis
- **Modality:** tabular
- **Inference:** frequentist
- **Query kind:** Unknown
- **Rows:** 80
- **Functional:** 

### Support slot

- **Matrix status:** off\_axis
- **Matrix coordinate:** none
- **Empirical support:** unavailable:not\_evaluated

### Support cell of the program

- **Coordinate:** none
- **Licensed:** no
- **Status:** none

## Refusals and limits

- **Operation:** inspect
- **Applicability:** unknown
- **Readiness:** unknown\_support
- **Coordinate:** none

### Limits

- The claim states no scalar value.
- Calibration is unavailable (calibration\_basis\_missing).

### Blockers and refusals

| Id | Scientific impossibility | Reason |
| --- | --- | --- |
| support.off\_axis | yes | query is not on the public support-matrix axis |

### Next actions

Next actions are suggestions for the caller; none is taken automatically.

| Action | Description |
| --- | --- |
| no\_licensed\_neighbor | no justified licensed neighbor; do not relabel the graph class |

- **Layers the premises preserve:** target
- **Layers the premises invalidate:** support, identification, results

This report restates the contract. It computes no estimate, grants no licence, and a status recorded here is the status the records carry.

## Program plan

- **Route:** SequentialContrast
- **Family:** design\_based
- **Stage:** prepared
- **Depth:** complete
- **Program identity:** 9146dc10af9d4475f1856a884f6f29899ed3cb2324ac12fed042ad6d4a120dc6
- **Sealed operation:** sequential\_contrast

### Scientific problem

- **data:** tabular
- **graph\_class:** RandomizedTrial
- **plan\_id:** sequential.crossfit\_contrast.wald
- **query\_variables:** y, a0, a1
- **target.form:** target
- **target.identity:** 707f28ed0e51ad3ccbd68caba93e90a5af9a064af883003d2930537f65d2f800

### Identification product

- **Carried:** retained
- **Status:** none
- **Installed at prepare:** no
- **Product identity:** 57a4e6325cc37036fa8ece0286e8c39d78dc50e3275f92f56501a400c757a9eb
- **Outcome:** identified
- **Claim class:** identified\_claim

| Estimand | Adjustment set | Instruments | Mediators |
| --- | --- | --- | --- |
| sequential.crossfit\_contrast | none | none | none |

### Statistical program

- **design.reason:** augmented sequential scores average per-subject nuisance fits
- **design.source:** not\_design\_based
- **estimator:** sequential.crossfit\_contrast.wald
- **folds.folds:** 2
- **folds.unit:** subject
- **identifier:** randomized.design
- **identity:** d4b9da9d0dfb1c788a84efb45b8d9078fb7cdcae3de6c82ecb55b0014694793c
- **models\[0\].family:** crossfit\_first\_decision\_propensity
- **models\[0\].inputs:** l0
- **models\[0\].output:** a0
- **models\[0\].role:** propensity
- **models\[1\].family:** crossfit\_second\_decision\_propensity
- **models\[1\].inputs:** l0, a0, l1
- **models\[1\].output:** a1
- **models\[1\].role:** propensity
- **models\[2\].family:** crossfit\_sequential\_outcome\_regression
- **models\[2\].inputs:** l0, a0, l1, a1
- **models\[2\].output:** y
- **models\[2\].role:** outcome
- **operators:** 0
- **realization:** frequentist

### Evidence and decision program

- **composition.kind:** single
- **identity:** 99dec0d4a3c47302aa2f4f28dc40b3ccfcab003e3737b2601341a639ef0ef027
- **sources\[0\].role:** observations

### Physical plan

- **backend.quotes\[0\].alternative:** direct\_prepared\_numeric\_views
- **backend.quotes\[0\].cost:** 2000
- **backend.quotes\[0\].resident\_bytes:** 3200
- **backend.quotes\[0\].work\_cells:** 1600
- **backend.quotes\[1\].alternative:** streamed\_reduction
- **backend.quotes\[1\].cost:** 2825
- **backend.quotes\[1\].resident\_bytes:** 3400
- **backend.quotes\[1\].work\_cells:** 2400
- **backend.quotes\[2\].alternative:** materialized\_design
- **backend.quotes\[2\].cost:** 2800
- **backend.quotes\[2\].resident\_bytes:** 6400
- **backend.quotes\[2\].work\_cells:** 2000
- **backend.quotes\[3\].alternative:** optional\_lazy\_preparation
- **backend.quotes\[3\].cost:** 2000
- **backend.quotes\[3\].resident\_bytes:** 3200
- **backend.quotes\[3\].work\_cells:** 1600
- **backend.rejected\[0\].alternative:** direct\_prepared\_numeric\_views
- **backend.rejected\[0\].reason:** missing operator capability prepared\_numeric\_view
- **backend.rejected\[1\].alternative:** streamed\_reduction
- **backend.rejected\[1\].reason:** missing operator capability streamed\_reduction
- **backend.rejected\[2\].alternative:** optional\_lazy\_preparation
- **backend.rejected\[2\].reason:** missing operator capability lazy\_preparation
- **backend.selected:** materialized\_design
- **backend.selected\_cost:** 2800
- **design.reason:** augmented sequential scores average per-subject nuisance fits
- **design.source:** not\_design\_based
- **memory.components\[0\].bytes:** 3200
- **memory.components\[0\].component:** encoded\_width
- **memory.components\[1\].bytes:** 50
- **memory.components\[1\].component:** masks
- **memory.components\[2\].bytes:** 0
- **memory.components\[2\].component:** design\_expansion
- **memory.components\[3\].bytes:** 3200
- **memory.components\[3\].component:** learner\_fold\_copies
- **memory.components\[4\].bytes:** 4080
- **memory.components\[4\].component:** solver\_scratch
- **memory.components\[5\].bytes:** 1920
- **memory.components\[5\].component:** retained\_scores\_draws
- **memory.components\[6\].bytes:** 680
- **memory.components\[6\].component:** output
- **memory.components\[7\].bytes:** 0
- **memory.components\[7\].component:** concurrent\_workers
- **memory.learner\_fold\_copies:** 1
- **memory.total:** 13130
- **memory.workers:** 1
- **operators\[0\].capability:** cpu
- **operators\[0\].designs:** 0
- **operators\[0\].device\_work:** no
- **operators\[0\].effect:** select
- **operators\[0\].output\_bytes:** 0
- **operators\[0\].retained\_bytes:** 0
- **operators\[0\].scratch\_bytes:** 1920
- **operators\[0\].sources:** 1
- **operators\[1\].capability:** cpu
- **operators\[1\].designs:** 1
- **operators\[1\].device\_work:** no
- **operators\[1\].effect:** scan
- **operators\[1\].output\_bytes:** 0
- **operators\[1\].retained\_bytes:** 0
- **operators\[1\].scratch\_bytes:** 5760
- **operators\[1\].sources:** 1
- **output\_bytes:** 0
- **retained\_bytes:** 0
- **scratch\_bytes:** 7680

## Identities

- **Target:** ecbbe1071b706a90817dd8577bd5cfa984215b1cab05c0bceb32cabd45b21731
- **Identification:** b50bd7459618af7272fc7218a6716baf35e481d61770a70bb902f5e355d17f28
- **Identification product:** none
- **Program:** 71523fc7a0d1380a0c873a9c99a92baccae1d388dd99e9b87e5467a0b80fe247
- **Inference binding:** 122227db05904a9fdcabbcab561919a16691db0b8a445cffef22541a91240ba4
- **Observation:** 9656ec48033e1c382f79800434aa8a3f9496560dbd8a369cbbf1d830f00ef5bd
- **Data snapshot:** 70480c58ee5f1a795d90ac591624509b73ae22077b2cbaa91a40999a05a4e578
- **Score reuse:** none
- **Target weights:** none

