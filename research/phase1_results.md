# Phase 1 results — halted partial run

This file is intentionally not a GO/NO-GO report. Phase 1 was halted on user
request while the corrected reasoner retraining was in progress.

## Preserved work

- The initial two-pass reasoner and matched baseline completed, and their
  validation artifacts exist under `results/phase1/`.
- A reference test then found that the initial equation-level Mamba-3 fallback
  was not invariant to chunk partitioning. The cause was K scaling being
  omitted from intra-chunk attention. The implementation was corrected to
  match the official kernel ordering, and the new causal/chunk/backward tests
  pass.
- Because the initial checkpoints were trained with the flawed reference, they
  are retained only as preliminary/debug artifacts and must not be used as the
  final Phase 1 result.
- Corrected reasoner retraining was halted at approximately step 5,400 of
  14,146. A checkpoint from the latest periodic save is preserved locally but
  excluded from Git because of its size. The corrected baseline was not
  retrained after the fix.

## Current conclusion

No final Phase 1 GO/NO-GO conclusion is made. Resume from the corrected run
after validating the saved checkpoint, complete the corrected baseline, then
rerun depth evaluation, synchronized profiling, and the final report.
