# Measurement models

A measured column is not the scientific quantity it is recorded for. A
measurement model states how they relate, for which population, regime and
arm, and what may be concluded. It is separate from the schema's
`MeasurementSpec` flag, so no schema digest changes.

## Kinds

Six kinds are distinct in type and in refusal: observed, latent scientific,
noisy or misclassified, probabilistic feature assignment, learned vector
representation, derived. A model relates one non-latent *indicator* to one
latent scientific *construct*. Offering a proxy, a learned representation or a
derived column as the construct is refused (`proxy_offered_as_construct`), as
is naming the measured column as the construct.

## Binding and structure

A model is bound to a population, a regime and an arm. Error that differs by
arm is an explicit per-arm declaration; error that differs by population or
time has no bounded route and is refused, never pooled with the
non-differential route (`differential_error_not_pooled`).

## Representations

A representation has a coordinate system and a version. A change needs an
explicit mapping; a mapping carries values only and carries intervention
meaning only when it declares the evidence for it. Coordinates of a learned
vector representation are not intervenable. (`representation_mapping_required`)

## Routes

| Route | Correction | Refused |
|---|---|---|
| classical continuous error | corrected normal equations (= regression calibration for a linear model), known or validated error variance | nonlinear outcome models |
| repeated measurements | same, error variance and ICC from the sample's own replicates | normal-theory variance intervals |
| binary misclassification | Rogan-Gladen prevalence and risk difference, Youden index above the declared minimum | exposure misclassification |
| probabilistic categorical | confusion-matrix inversion, category proportions and outcome means | per-unit soft assignments |
| external latent posterior | posterior-mean substitution, or Rubin's rules for outcome-conditioned draws | any trust inferred from metadata |

Each route's assumptions, allowed operations and refused operations are in
`antecedent_contract::measurement::route_card`.

## Uncertainty

Variances are first-order delta-method variances that include the uncertainty
of the estimated measurement parameters (error variance, sensitivity and
specificity, confusion columns, between-imputation variance). The record says
whether parameter uncertainty was propagated, declared exact, or not
representable (`ParameterUncertainty`). The standard error is an uncalibrated
estimate: no interval is produced and an interval request is refused
(`cell_not_licensed`) until the whole measurement-plus-estimation method has a
measured coverage record.

## Artifact

`measurement_model_v1` carries the model, its measurement, binding,
representation and evidence identities and an optional corrected functional
with its sufficient inputs (moments, counts, class summaries or
per-imputation fits). A consumer holding the bytes and the retained model
identity recomputes the point and standard error with `antecedent-stats` only
and refuses any stored number that does not recompute.

## References

Carroll, Ruppert, Stefanski and Crainiceanu (2006), *Measurement Error in
Nonlinear Models*, doi:10.1201/9781420010138. Rogan and Gladen (1978), Am J
Epidemiol 107:71-76, doi:10.1093/oxfordjournals.aje.a112510. Rubin (1987),
*Multiple Imputation for Nonresponse in Surveys*, doi:10.1002/9780470316696.
