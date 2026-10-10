# Selected binary nested-Markov Fisher candidate

Oracle kind: `closed_form`. In the selected Verma SCM, X1 and X2 are independent fair bits, P(X3=0|X2)=(0.75,0.25), and P(X4=0|X3)=(0.8,0.4). The 16 multinomial cells at N16000 are positive integers; the causal contrast is0.2.

Expected covariance for `[mean0,mean1,contrast]`, multiplied byN, is
`[[0.34,0.2,-0.14],[0.2,67/150,37/150],[-0.14,37/150,29/75]]`.
Independent stratified multinomial algebra gives var(c0)=var(c1)=0.375/N, var(q40)=(32/75)/N and var(q41)=0.64/N; score orthogonality and the causal gradients yield the full matrix above. The implementation instead inverts the eleven-parameter expected Fisher information from analytic cell derivatives. Agreement verifies dependence propagation and size scaling at this truth, without measuring repeated-sampling coverage.

Run `cargo test -p antecedent-estimate --features calibration-internal --test nested_markov_uncertainty --offline` and `cargo test -p antecedent-io --test nested_markov_artifact --offline`. The IO test's dev-only feature exercises the original source replay and semantic mutations. The normal measured adapter separately licenses named 95% mean0, mean1 and contrast endpoints at its attested n1000..4000 IID interior-model scope. Exact IID multinomial sampling and correct interior model specification are declared premises, not authenticated by cell counts. Sparse cells, fractional pseudo-counts and singular information refuse. This covariance oracle itself does not measure coverage; the original source artifact retains its unmeasured standing inside a separately verified measured envelope.
