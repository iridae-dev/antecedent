# Causal analysis report

## Summary

- **Reading:** an executed claim
- **Query:** AverageEffect
- **Inference:** frequentist
- **Identifier:** none
- **Estimator:** linear.adjustment.ate
- **Route:** StaticAteDag
- **Claim kind:** point
- **Value:** 1.90592295108
- **Support status:** licensed

- The claim publishes a value of kind point; the sections below state what it rests on and where it stops.
- The route is a licensed support-matrix cell.
- Calibration status is scope\_not\_assessed: the reported uncertainty carries no measured-coverage record.
- Identification status is nonparametrically\_identified.

## Claim

- **Claim id:** 66a99ead62f5a273458b60344a5bd7cd1be6b0319551d0c269f3faae7747539f
- **Kind:** point
- **Value:** 1.90592295108
- **Outcome units:** none
- **Identification domain:** identified
- **Support domain:** unknown
- **Evaluated domain:** evaluated
- **Execution:** 8803f9ddaa94d51aa42daff15db1617ee0414860d03e9825dd54714d229958f6

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

- snapshot:4517e9819225a4a546dfa1ee31cbbacd7bdbd6e1f8230afb5cf34074ea92f177
- identification:90c1dcd8dfb83272abc040fe94c2dca6453ecf272596a99488135c410a9c0f8b
- program:9ca21929bce36cca6f950bc30202f03470e22bd49d0934ce1fc89e788da70b5d

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
- **Reason:** sample\_size\_outside\_measured\_range
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
- **Basis rows:** 128
- **Basis successful replicates:** none

## Support and licence

- **Support status:** licensed
- **Modality:** tabular
- **Inference:** frequentist
- **Query kind:** AverageEffect
- **Rows:** 128
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

- Calibration is scope\_not\_assessed (sample\_size\_outside\_measured\_range).

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
- **Program identity:** 35039554206619be5ecfe5ff435474047673cf7ef6b879200052c620b84299ab
- **Sealed operation:** linear\_adjustment

### Scientific problem

- **data:** tabular
- **graph\_class:** Dag
- **plan\_id:** static\_ate
- **query\_variables:** t, y
- **source\_digest:** 5eb9e3cc187391923b6d47b7df03782e4b345eb9b202c6fee4b05e7341f97ace
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
- **identity:** 759d8f09f71e1e62b85564ebaa320c8b8428a1520e954075920373b89bfe6251
- **models\[0\].family:** ols.faer
- **models\[0\].inputs:** t, z
- **models\[0\].output:** y
- **models\[0\].role:** outcome
- **operators:** 3
- **realization:** frequentist

### Evidence and decision program

- **composition.kind:** single
- **identity:** 24cb30af80bb68a8bfcc381552a8fd5b0cdddda50b3df126fd3100116e4c7de2
- **sources\[0\].digest:** 5eb9e3cc187391923b6d47b7df03782e4b345eb9b202c6fee4b05e7341f97ace
- **sources\[0\].role:** observations

### Physical plan

- **design.columns:** 3
- **design.direct:** yes
- **design.rows:** 128
- **design.source:** quoted
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
- **output\_bytes:** 10632
- **retained\_bytes:** 10624
- **scratch\_bytes:** 215296

## Identities

- **Target:** 0e8cc7ffd6ab51ac2162a8525997413b0b2885693570d5b847b4bcaefe2cab28
- **Identification:** 90c1dcd8dfb83272abc040fe94c2dca6453ecf272596a99488135c410a9c0f8b
- **Identification product:** 7f3a931cd7364b9acaefd685b977f177b064987a2153b15ba9e4ba49434f4d79
- **Program:** 9ca21929bce36cca6f950bc30202f03470e22bd49d0934ce1fc89e788da70b5d
- **Inference binding:** 122227db05904a9fdcabbcab561919a16691db0b8a445cffef22541a91240ba4
- **Observation:** 55d73a79374fd7a00313b8db3294dd7d0beb9db0a71fcb319640b2bf15e67131
- **Data snapshot:** 4517e9819225a4a546dfa1ee31cbbacd7bdbd6e1f8230afb5cf34074ea92f177
- **Score reuse:** none
- **Target weights:** none

