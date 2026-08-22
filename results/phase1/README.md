# Phase 1 artifacts

The validation summaries and examples currently in this directory were
generated from the preliminary pre-fix checkpoints. They are retained for
debugging only: a later chunk-partition test found and fixed K-scaling in the
portable Mamba-3 reference, so these numbers are not final Phase 1 results.

The corrected retraining was halted before evaluation. Large `.npz` token
artifacts, local profiler traces, and checkpoints are intentionally excluded
from Git; rerun the documented scripts after resuming corrected training.
