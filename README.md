# FGCS-CoRA

This repository package contains the key CoRA implementation and final experiment evidence used for the FGCS major-revision empirical study.

## Contents

- `tools/`: core experiment runner, backbone definitions, data I/O, manifest creation, coverage checking, and summary/table generation scripts.
- `results/fgcs_revision/summary_after_sensitivity/`: final seed-level and aggregate CSV summaries.
- `results/fgcs_revision/tables_after_sensitivity/`: final LaTeX table snippets generated from the final summaries.
- `results/fgcs_revision/priority/`: priority manifests and coverage/validation notes for the executed P0/P1 suites.

## Minimal Environment

Install the Python dependencies in `requirements.txt` or `tools/requirements_revision.txt`.

```powershell
pip install -r requirements.txt
```
