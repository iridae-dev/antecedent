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
and a cluster sandwich variance diagnostic for the HT contrast. The variance
treats supplied probabilities as fixed and does not account for exposure-model
estimation; no calibrated interval or coverage claim is made.

This utility does not establish observational identification. Users must defend
no unmeasured network confounding conditional on covariates used to obtain the
probabilities, correct network and exposure mapping, consistency, positivity,
and partial interference. It does not extend the licensed randomized
`InterferenceQuery` path or support-matrix entries.
