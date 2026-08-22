# Hypotheses and falsification criteria

## Scope and notation

For target token (i), let (z_i^{(0)}=h_i) be the causal Mamba-3 stream
output and let (z_i^{(r+1)}=R_\theta(z_i^{(r)},H_{\leq i})) be the shared
latent cell over frozen causal context. Define

\[
\ell_i^{(r)}=-\log p_\theta(y_i\mid z_i^{(r)}),\qquad
\Delta_{i,r}=\ell_i^{(r)}-\ell_i^{(r+1)}.
\]

The deployable conditional value is

\[
g_{i,r}=\mathbb E[\Delta_{i,r}\mid z_i^{(r)},H_{\leq i},d_{\rm rem},c_r],
\]

where (d_{\rm rem}) is remaining trusted depth and (c_r) is incremental
cost. A later controller may continue when
(\hat g_{i,r}/c_r>\lambda), subject to a hard maximum. Future loss is allowed
only to train or evaluate the predictor, never as an inference input.

For finite depths (mathcal D), define operational compute demand

\[
d^*_\epsilon(i)=\min\{d\in\mathcal D:\ell_i^{(d)}\leq
\min_{j\in\mathcal D}\ell_i^{(j)}+\epsilon\}.
\]

The marginal curve is (M_i(d)=\ell_i^{(d-1)}-\ell_i^{(d)}). These are
empirical quantities, not circuit-complexity classes or claims about formal
computational power.

## Primary hypotheses

### H0 — no useful heterogeneity

Extra recurrence has no useful heterogeneous marginal value. The null predicts
low inter-token variance in (M_i(d)), no improvement of a state-conditioned
value predictor over a mean-only predictor, and no adaptive-routing gain at
matched compute.

### H1 — heterogeneous marginal value

Marginal value varies materially across tokens and states. Report bootstrap
intervals for inter-token variance, a non-trivial helped subset, a non-trivial
harmed/neutral subset, and a spread in (d^*_{\epsilon}) not explained by
frequency alone.

### H2 — state-conditioned prediction

A predictor (f_\phi(z_i^{(r)},H_{\leq i},d_{\rm rem},c_r)) estimates realized
next-step loss reduction better than these pre-registered baselines:

1. global mean by depth;
2. token frequency and token identity;
3. entropy, margin, and current-loss proxies;
4. position, context length, lexical flags, and representation drift;
5. an early/initial-depth predictor that does not observe the online state.

Use held-out-token MSE, rank correlation, calibration, and top-k allocation
regret. Lower MSE alone is insufficient.

### H3 — Pareto improvement

At matched measured work, the marginal-value controller has a better
accuracy/latency frontier than fixed depth, early depth prediction,
ACT/PonderNet halting, an ANIRA-style online policy, and representation-drift
or KL early exit. “Better” means a paired bootstrap improvement at two
non-extreme budgets without exceeding the trusted maximum.

This is an extension claim about the Mamba-3/frozen-context/token-skipping
setting, not a general novelty claim.

### H4 — computational rather than superficial allocation

Allocation responds to generator-controlled sequential structure rather than
only sequence length, frequency, entropy, position, input length, or lexical
features. Balance these features, use mixed depths with equal length, and test
held-out depths. A policy that uses a visible depth marker or length but fails
after marker removal does not support H4.

### H5 — partial depth extrapolation

Allocation or prediction partially extrapolates to unseen sequential depths.
Train on (D\in\{1,2,4,8\}) and test on (D\in\{12,16,24,32\}), including a
fixed-length task. Success requires positive correlation with realized value
and non-catastrophic performance relative to the best fixed-depth baseline;
failure to extrapolate is a useful negative result.

## Phase-1 prerequisite and falsification

Phase 1 does not train a controller. It tests the fixed checkpoint at
(T=0,1,2,4,8). GO requires finite, resumable two-pass training; no
catastrophic reasoner degradation; some tokens that improve and some that do
not or worsen; measurable (d^*_{\epsilon}) spread; and reproducible target-
GPU latency. If all tokens prefer (T=8), all gains are numerical noise, or
recurrence is unstable, the controller is not a justified next step.
