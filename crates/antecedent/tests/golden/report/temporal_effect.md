# Causal analysis report

## Summary

- **Reading:** an executed claim
- **Query:** PulseEffect
- **Inference:** frequentist
- **Identifier:** none
- **Estimator:** temporal.linear.adjustment
- **Route:** TemporalEffectDag
- **Claim kind:** point
- **Value:** 1.99994085174
- **Support status:** licensed

- The claim publishes a value of kind point; the sections below state what it rests on and where it stops.
- The route is a licensed support-matrix cell.
- Calibration status is unavailable: the reported uncertainty carries no measured-coverage record.
- Identification status is nonparametrically\_identified.

## Claim

- **Claim id:** 89b7c0892f3e5af8e70cff87fb9180e7eb935d80a2577dc0d5a5b863727ce133
- **Kind:** point
- **Value:** 1.99994085174
- **Outcome units:** none
- **Identification domain:** identified
- **Support domain:** unknown
- **Evaluated domain:** evaluated
- **Execution:** 8803f9ddaa94d51aa42daff15db1617ee0414860d03e9825dd54714d229958f6

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

- snapshot:d6e0ced2509fd188ab1e02403a213a2c5a0e68843b85a3355998250a2a5152e5
- identification:0ddb7fd84bec5325370c98d2616c93065203c8e521701334743508194fd3b324
- program:d8ff19a3a1ff2502d6e049f73836a7f7b27eaf10915feddf616349b8ca5fd6f3

## Uncertainty

The uncertainty slot is unavailable: omitted.

### Uncertainty the program must publish

- **claim\_kind:** point
- **reading:** sampling
- **realization:** frequentist

### Calibration

- **Calibration status:** unavailable
- **Governing record:** none
- **Reason:** no\_interval\_reported
- **Measured rows (smallest):** none
- **Measured rows (largest):** none
- **Record dependence:** none
- **Measured at:** none
- **Nominal level:** none
- **Observed coverage:** none
- **Basis query:** PulseEffect
- **Basis graph class:** TemporalDag
- **Basis structure:** fixed
- **Basis modality:** series
- **Basis inference:** Frequentist
- **Basis estimator:** temporal.linear.adjustment
- **Basis interval method:** none
- **Basis standard error:** 
- **Basis dependence:** iid
- **Basis posterior:** 
- **Basis functional:** pulse.h1.all\_observed
- **Basis level:** 0
- **Basis identification:** point
- **Basis rows:** 319
- **Basis successful replicates:** none

## Support and licence

- **Support status:** licensed
- **Modality:** series
- **Inference:** frequentist
- **Query kind:** PulseEffect
- **Rows:** 320
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

- Calibration is unavailable (no\_interval\_reported).

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
- **Program identity:** a8f8dc9728b29da8de6e7a82ccb0fae5317df40b69db7525acb1190b3b54ac8b
- **Sealed operation:** temporal\_dag\_effect

### Scientific problem

- **data:** temporal
- **graph\_class:** TemporalDag
- **plan\_id:** temporal\_effect
- **query\_variables:** t, y
- **target.form:** target
- **target.identity:** 2c1802e8cc1a3a40d3601b550f9ec629ceb3fd8562b9de07b61d2696434cab8e

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
- **identity:** 781ea7fc6a52ba7ce53d5d50d7f0d9a374d0927b2cc3021b1f074ffc55ff70ea
- **models\[0\].family:** temporal.linear.adjustment
- **models\[0\].inputs:** t
- **models\[0\].output:** y
- **models\[0\].role:** outcome
- **operators:** 3
- **realization:** frequentist

### Evidence and decision program

- **composition.kind:** single
- **identity:** 99dec0d4a3c47302aa2f4f28dc40b3ccfcab003e3737b2601341a639ef0ef027
- **sources\[0\].role:** observations

### Physical plan

- **design.reason:** the lagged adjustment design is assembled from the unfolded sample at execution
- **design.source:** not\_design\_based
- **operators\[0\].capability:** cpu
- **operators\[0\].designs:** 1
- **operators\[0\].device\_work:** no
- **operators\[0\].effect:** read
- **operators\[0\].output\_bytes:** 5120
- **operators\[0\].retained\_bytes:** 0
- **operators\[0\].scratch\_bytes:** 0
- **operators\[0\].sources:** 1
- **operators\[1\].capability:** cpu
- **operators\[1\].designs:** 1
- **operators\[1\].device\_work:** no
- **operators\[1\].effect:** fit
- **operators\[1\].output\_bytes:** 8
- **operators\[1\].retained\_bytes:** 5120
- **operators\[1\].scratch\_bytes:** 10240
- **operators\[1\].sources:** 1
- **operators\[2\].capability:** cpu
- **operators\[2\].designs:** 1
- **operators\[2\].device\_work:** no
- **operators\[2\].effect:** intervene
- **operators\[2\].output\_bytes:** 0
- **operators\[2\].retained\_bytes:** 0
- **operators\[2\].scratch\_bytes:** 0
- **operators\[2\].sources:** 1
- **output\_bytes:** 5128
- **retained\_bytes:** 5120
- **scratch\_bytes:** 10240

## Identities

- **Target:** 25cd29dacec93515dc7f4444ce62dac101733331cef1f8247dfbed301773bd62
- **Identification:** 0ddb7fd84bec5325370c98d2616c93065203c8e521701334743508194fd3b324
- **Identification product:** 16c5967b5bdfb668cc1e0fd3d30148ac9aea15943417b05691ec113fa572b0e0
- **Program:** d8ff19a3a1ff2502d6e049f73836a7f7b27eaf10915feddf616349b8ca5fd6f3
- **Inference binding:** 122227db05904a9fdcabbcab561919a16691db0b8a445cffef22541a91240ba4
- **Observation:** 6a1da6b2a79f2950955ca198b631d44dc37ec38365f718d01a5992c7c599b23b
- **Data snapshot:** d6e0ced2509fd188ab1e02403a213a2c5a0e68843b85a3355998250a2a5152e5
- **Score reuse:** none
- **Target weights:** none

