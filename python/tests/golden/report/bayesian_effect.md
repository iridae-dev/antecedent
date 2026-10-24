# Causal analysis report

## Summary

- **Reading:** an executed claim
- **Query:** AverageEffect
- **Inference:** bayesian
- **Identifier:** none
- **Estimator:** bayesian.gcomp
- **Route:** StaticAteDag
- **Claim kind:** point
- **Value:** 1.99992834668
- **Support status:** licensed

- The claim publishes a value of kind point; the sections below state what it rests on and where it stops.
- The route is a licensed support-matrix cell.
- Calibration status is unavailable: the reported uncertainty carries no measured-coverage record.
- Identification status is nonparametrically\_identified.

## Claim

- **Claim id:** 15357c9de7b52ec1d6fb601b7df298a799a799b992e1c7bc492c9cdc34ed4da3
- **Kind:** point
- **Value:** 1.99992834668
- **Outcome units:** none
- **Identification domain:** identified
- **Support domain:** unknown
- **Evaluated domain:** evaluated
- **Execution:** 431eb95b6b88f79a166060ef435b0f059a70bd3cc265bf3100ca01d57aa4b171

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

- snapshot:4227e9268ff5722d293afe79bb6aad7465835dccce4b0821d02d515b3ee89e54
- identification:96fba9fff73599a0e3cee2cba940842671350661995c3151c25535bafcb0ecbf
- program:4960d48b0ac07dcec33f995c87575fa7361da1a8fd576d266ce0555609c15f24

## Uncertainty

| Source | Reported target | Omitted |
| --- | --- | --- |
| sampling | analytic\_se | no |
| parameter | posterior | no |

### Uncertainty the program must publish

- **claim\_kind:** point
- **posterior.algorithm:** conjugate
- **posterior.draws:** 64
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
- **Basis rows:** 240
- **Basis successful replicates:** none

## Support and licence

- **Support status:** licensed
- **Modality:** tabular
- **Inference:** bayesian
- **Query kind:** AverageEffect
- **Rows:** 240
- **Functional:** all\_observed.mean
- **Posterior construction:** conjugate\_gaussian.gaussian\_identity.prior\_scale=30
- **Posterior draws requested:** 64

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
- **Program identity:** 3d3f18a67c4fdc70eeaba3f94126bca6432492dc5d9c253062f2b6b5b209df4b
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

- **design.reason:** the estimator fits its propensity, outcome, instrument or learner models on their own bases, sized at execution; no single adjustment design is quoted
- **design.source:** not\_design\_based
- **estimator:** bayesian.gcomp
- **identifier:** backdoor.adjustment
- **identity:** 60947a87a66a02d78b9b73fbd8694c6c317fff953fafa1c8f39f1055163754ef
- **models\[0\].family:** ols.faer
- **models\[0\].inputs:** t, z
- **models\[0\].output:** y
- **models\[0\].role:** outcome
- **operators:** 3
- **posterior.algorithm:** conjugate
- **posterior.draws:** 64
- **posterior.prior:** isotropic\_normal(scale=30)
- **realization:** bayesian

### Evidence and decision program

- **composition.kind:** single
- **identity:** 99dec0d4a3c47302aa2f4f28dc40b3ccfcab003e3737b2601341a639ef0ef027
- **sources\[0\].role:** observations

### Physical plan

- **design.reason:** the estimator fits its propensity, outcome, instrument or learner models on their own bases, sized at execution; no single adjustment design is quoted
- **design.source:** not\_design\_based
- **operators\[0\].capability:** cpu
- **operators\[0\].designs:** 1
- **operators\[0\].device\_work:** no
- **operators\[0\].effect:** read
- **operators\[0\].output\_bytes:** 5760
- **operators\[0\].retained\_bytes:** 0
- **operators\[0\].scratch\_bytes:** 0
- **operators\[0\].sources:** 1
- **operators\[1\].capability:** cpu
- **operators\[1\].designs:** 1
- **operators\[1\].device\_work:** no
- **operators\[1\].effect:** fit
- **operators\[1\].output\_bytes:** 8
- **operators\[1\].retained\_bytes:** 5760
- **operators\[1\].scratch\_bytes:** 11520
- **operators\[1\].sources:** 1
- **operators\[2\].capability:** cpu
- **operators\[2\].designs:** 1
- **operators\[2\].device\_work:** no
- **operators\[2\].effect:** intervene
- **operators\[2\].output\_bytes:** 0
- **operators\[2\].retained\_bytes:** 0
- **operators\[2\].scratch\_bytes:** 0
- **operators\[2\].sources:** 1
- **output\_bytes:** 5768
- **retained\_bytes:** 5760
- **scratch\_bytes:** 11520

## Identities

- **Target:** 716927442f91b16cbcc7381cce285105e7f5c5dd9a867660215fb6cc0d85083a
- **Identification:** 96fba9fff73599a0e3cee2cba940842671350661995c3151c25535bafcb0ecbf
- **Identification product:** 7f3a931cd7364b9acaefd685b977f177b064987a2153b15ba9e4ba49434f4d79
- **Program:** 4960d48b0ac07dcec33f995c87575fa7361da1a8fd576d266ce0555609c15f24
- **Inference binding:** 95c7307af95d924d90ecc3096a26533797f24289716e662681f0cae4b8946621
- **Observation:** ce6fe76d79b8e0d0cc58ebbbeb1a4dcac0b651b8c0d124f0a590f82102d46be8
- **Data snapshot:** 4227e9268ff5722d293afe79bb6aad7465835dccce4b0821d02d515b3ee89e54
- **Score reuse:** none
- **Target weights:** none

