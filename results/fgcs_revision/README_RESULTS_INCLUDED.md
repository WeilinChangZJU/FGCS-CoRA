# Included Result Evidence

This package keeps compact final evidence rather than full per-job JSON directories.

Included:

- Final seed-level CSV: `results/fgcs_revision/summary_after_sensitivity/revision_per_run.csv`
- Final aggregate summaries and paired tests under `results/fgcs_revision/summary_after_sensitivity/`
- Final LaTeX tables under `results/fgcs_revision/tables_after_sensitivity/`
- Executed P0/P1 priority manifests under `results/fgcs_revision/priority/`

Excluded:

- `results/fgcs_revision/main/`, `stress/`, `failure/`, `backbone/`, `ablation/`, and `sensitivity/` raw metrics JSON directories.
- `results/fgcs_revision/smoke_*`
- Generated figures and plotting scripts.
- Execution logs.

The retained CSV summaries are sufficient to audit the manuscript-level numerical claims, seed counts, paired tests, and final tables without carrying bulky intermediate files.
