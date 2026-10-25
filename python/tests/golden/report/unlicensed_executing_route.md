# Causal analysis report

## Summary

- **Reading:** an executed claim
- **Query:** Unknown
- **Inference:** frequentist
- **Identifier:** none
- **Estimator:** policy.triangular\_kernel\_inverse\_density
- **Route:** ContinuousDoseResponse
- **Claim kind:** point
- **Support status:** off\_axis

- The claim publishes a value of kind point; the sections below state what it rests on and where it stops.
- The query is off the support-matrix axis; no matrix licence applies to it.
- Calibration status is unavailable: the reported uncertainty carries no measured-coverage record.
- Identification status is nonparametrically\_identified.

## Claim

- **Claim id:** 2e411540fd5d3ae25bc18333c0d48cf2ec90e0c61928e16617eafd931efd2e4a
- **Kind:** point
- **Value:** none
- **Outcome units:** none
- **Identification domain:** identified
- **Support domain:** unknown
- **Evaluated domain:** evaluated
- **Execution:** be545e43ed54352a0a8593bca6162f8b8d38c7ca26e7a23095dfe3b1f059b00d

## Identification

- **Graph class:** RandomizedTrial
- **Structure source:** randomized\_trial
- **Identifier:** none
- **Estimator:** none
- **Resolved estimator:** policy.triangular\_kernel\_inverse\_density
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
| custom:conditional\_dose\_exchangeability | declared |
| custom:correct\_observed\_dose\_density | declared |
| custom:continuous\_dose\_positivity | declared |
| custom:consistency\_no\_interference | declared |

## Diagnostics and evidence

### Evidence the claim references

- snapshot:50d6acb07bbd29d1c109ef80f066ce47665d609e472b6ed0074cd84a527626a8
- identification:5e87f6212366eeb0df8fb5292c0261d79d64df2681499b0f8738ba8cc323f110
- program:176791b0861cb7e3b0d3b9b438bb3e047253fe69a626033bb54da103d39de4ed

## Uncertainty

| Source | Reported target | Omitted |
| --- | --- | --- |
| sampling | continuous\_dose\_point\_only\_no\_interval | no |

### Uncertainty the program must publish

- **claim\_kind:** response
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
- **Rows:** 10
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

- **Route:** ContinuousDoseResponse
- **Family:** design\_based
- **Stage:** prepared
- **Depth:** complete
- **Program identity:** 00b817e75c72fa52a0fe223e18151a52abc4f45aa7f6b43732ce132a7120f4db
- **Sealed operation:** continuous\_dose\_response

### Scientific problem

- **data:** tabular
- **graph\_class:** RandomizedTrial
- **plan\_id:** policy.continuous\_dose\_response
- **query\_variables:** y, dose, density
- **target.form:** target
- **target.identity:** 592566bfbd3c52bfe8763f8fa2fb3e4d9da4dee6a9f053c6b1534f124481f984

### Identification product

- **Carried:** retained
- **Status:** none
- **Installed at prepare:** no
- **Product identity:** 5edbc2d67a838c54fbdb631250a41bc9c8191b600ed0e59223dc8acd09ce1017
- **Outcome:** identified
- **Claim class:** identified\_claim

| Estimand | Adjustment set | Instruments | Mediators |
| --- | --- | --- | --- |
| policy.conditional\_dose\_response | none | none | none |

### Statistical program

- **design.reason:** a local kernel average per baseline group fits no design matrix
- **design.source:** not\_design\_based
- **estimator:** policy.triangular\_kernel\_inverse\_density
- **identifier:** policy.continuous\_dose\_exchangeability
- **identity:** 6f52d2838b00dd5f4db7a21eb128527b4ac8957d145ddd8235bd7c6d93b92707
- **models\[0\].family:** supplied\_conditional\_dose\_density
- **models\[0\].inputs:** dose
- **models\[0\].output:** density
- **models\[0\].role:** propensity
- **models\[1\].family:** epanechnikov\_kernel
- **models\[1\].inputs:** dose
- **models\[1\].output:** y
- **models\[1\].role:** outcome
- **operators:** 0
- **realization:** frequentist

### Evidence and decision program

- **composition.kind:** single
- **identity:** 99dec0d4a3c47302aa2f4f28dc40b3ccfcab003e3737b2601341a639ef0ef027
- **sources\[0\].role:** observations

### Physical plan

- **backend.quotes\[0\].alternative:** direct\_prepared\_numeric\_views
- **backend.quotes\[0\].cost:** 90
- **backend.quotes\[0\].resident\_bytes:** 240
- **backend.quotes\[0\].work\_cells:** 60
- **backend.quotes\[1\].alternative:** streamed\_reduction
- **backend.quotes\[1\].cost:** 99
- **backend.quotes\[1\].resident\_bytes:** 312
- **backend.quotes\[1\].work\_cells:** 60
- **backend.quotes\[2\].alternative:** materialized\_design
- **backend.quotes\[2\].cost:** 150
- **backend.quotes\[2\].resident\_bytes:** 480
- **backend.quotes\[2\].work\_cells:** 90
- **backend.quotes\[3\].alternative:** optional\_lazy\_preparation
- **backend.quotes\[3\].cost:** 90
- **backend.quotes\[3\].resident\_bytes:** 240
- **backend.quotes\[3\].work\_cells:** 60
- **backend.rejected\[0\].alternative:** direct\_prepared\_numeric\_views
- **backend.rejected\[0\].reason:** missing operator capability prepared\_numeric\_view
- **backend.rejected\[1\].alternative:** streamed\_reduction
- **backend.rejected\[1\].reason:** missing operator capability streamed\_reduction
- **backend.rejected\[2\].alternative:** optional\_lazy\_preparation
- **backend.rejected\[2\].reason:** missing operator capability lazy\_preparation
- **backend.selected:** materialized\_design
- **backend.selected\_cost:** 150
- **design.reason:** a local kernel average per baseline group fits no design matrix
- **design.source:** not\_design\_based
- **memory.components\[0\].bytes:** 240
- **memory.components\[0\].component:** encoded\_width
- **memory.components\[1\].bytes:** 6
- **memory.components\[1\].component:** masks
- **memory.components\[2\].bytes:** 0
- **memory.components\[2\].component:** design\_expansion
- **memory.components\[3\].bytes:** 0
- **memory.components\[3\].component:** learner\_fold\_copies
- **memory.components\[4\].bytes:** 416
- **memory.components\[4\].component:** solver\_scratch
- **memory.components\[5\].bytes:** 0
- **memory.components\[5\].component:** retained\_scores\_draws
- **memory.components\[6\].bytes:** 104
- **memory.components\[6\].component:** output
- **memory.components\[7\].bytes:** 3744
- **memory.components\[7\].component:** concurrent\_workers
- **memory.learner\_fold\_copies:** 0
- **memory.total:** 4510
- **memory.workers:** 10
- **operators\[0\].capability:** cpu
- **operators\[0\].designs:** 0
- **operators\[0\].device\_work:** no
- **operators\[0\].effect:** select
- **operators\[0\].output\_bytes:** 0
- **operators\[0\].retained\_bytes:** 0
- **operators\[0\].scratch\_bytes:** 240
- **operators\[0\].sources:** 1
- **operators\[1\].capability:** cpu
- **operators\[1\].designs:** 1
- **operators\[1\].device\_work:** no
- **operators\[1\].effect:** scan
- **operators\[1\].output\_bytes:** 0
- **operators\[1\].retained\_bytes:** 0
- **operators\[1\].scratch\_bytes:** 720
- **operators\[1\].sources:** 1
- **output\_bytes:** 0
- **retained\_bytes:** 0
- **scratch\_bytes:** 960

## Identities

- **Target:** 973d16e7e4724adacac8e3444436474054c545c98e28f887e09bcf0117a3c4ea
- **Identification:** 5e87f6212366eeb0df8fb5292c0261d79d64df2681499b0f8738ba8cc323f110
- **Identification product:** none
- **Program:** 176791b0861cb7e3b0d3b9b438bb3e047253fe69a626033bb54da103d39de4ed
- **Inference binding:** 122227db05904a9fdcabbcab561919a16691db0b8a445cffef22541a91240ba4
- **Observation:** 8ff262ed42133fc63a4e0806aaa95ae7360c1b3d8d9b28522acbb40fd516d6e6
- **Data snapshot:** 50d6acb07bbd29d1c109ef80f066ce47665d609e472b6ed0074cd84a527626a8
- **Score reuse:** none
- **Target weights:** none

