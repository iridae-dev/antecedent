# Observational network exposure utility

`interference.estimate_observational_network_exposure` accepts a fixed network,
an observed binary treatment assignment, two requested own-treatment/neighbor
exposure levels, and caller-supplied probabilities for those exposures. The
probabilities may be declared `known` or `externally_estimated`; the utility
does not fit the exposure model.

An explicit `PartialInterference` cluster map is required. The native kernel
checks the network edge list stays within that partition, requires positive
exposure probabilities and observed rows at both requested levels, and reports
unit and cluster support. It returns Horvitz–Thompson and Hájek point contrasts
and a cluster sandwich variance diagnostic for the HT contrast. The direct
utility remains point-only. On the retained `analyze` path, known fixed
propensities yield a pointwise 95% cluster t interval for the HT contrast when
there are at least 30 independent clusters, eight observed clusters at each
requested exposure, and positive between-cluster score variation. A 400-study
known-truth fixture covered the effect in 381 studies. Externally estimated
propensities and thinner support keep the point and an explicit interval
refusal. The interval holds supplied probabilities fixed and does not include
exposure-model estimation uncertainty.

This utility does not establish observational identification. Users must defend
no unmeasured network confounding conditional on covariates used to obtain the
probabilities, independent clusters, correct network and exposure mapping,
consistency, positivity, and partial interference. This route remains outside
the graphless support-matrix licenses.
