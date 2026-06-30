# FGCS-CoRA

This repository package contains the key CoRA implementation and final experiment evidence used for the FGCS major-revision empirical study.

Repository target: https://github.com/WeilinChangZJU/FGCS-CoRA

## Contents

- `tools/`: core experiment runner, backbone definitions, data I/O, manifest creation, coverage checking, and summary/table generation scripts.
- `results/fgcs_revision/summary_after_sensitivity/`: final seed-level and aggregate CSV summaries.
- `results/fgcs_revision/tables_after_sensitivity/`: final LaTeX table snippets generated from the final summaries.
- `results/fgcs_revision/priority/`: priority manifests and coverage/validation notes for the executed P0/P1 suites.

## Intentionally Excluded

- Generated figures and all plotting/drawing programs.
- Logs, smoke outputs, intermediate advisor packages, zip archives, checkpoints, and raw per-job JSON directories.
- Raw/preprocessed dataset folders (`data`, `data_stress`, and `dataset`) because they are large and environment-specific.
- P2/CSDI artifacts, because CSDI is not used in the final retained experiment package.

The seed-level experimental evidence needed for checking reported values is preserved in `revision_per_run.csv`; the full raw metrics JSON directories are omitted to keep the public repository compact.

## Minimal Environment

Install the Python dependencies in `requirements.txt` or `tools/requirements_revision.txt`.

```powershell
pip install -r requirements.txt
```

## Key Result Files

- `revision_per_run.csv`: seed-level final records used to build summaries.
- `revision_summary_mean_std.csv`: aggregate mean/std/n values.
- `revision_paired_tests.csv`: paired Random-vs-CoRA tests with confidence intervals and adjusted p-values.
- `revision_failure_summary.csv`: selected-client non-return quality/system metrics.
- `revision_backbone_summary.csv`: SAITS backbone check summaries.

No smoke outputs are included or used as manuscript evidence.
