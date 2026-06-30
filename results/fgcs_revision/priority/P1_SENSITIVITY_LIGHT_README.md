# P1 Sensitivity Light README

Generated: `2026-06-30 07:58:21 +0800`

## Scope

- Suite name: `P1_sensitivity_light`.
- Manifest suite label: `sensitivity`.
- Purpose: hyperparameter sensitivity, appendix-oriented supporting evidence.
- This is not mechanism ablation. P1-ablation remains mechanism ablation.
- P2/CSDI is not included.

## Fixed Design

- Dataset: `ETTm1`
- Setting: `hetero_h4`
- Backbone: `gru`
- Failure mode: `none`
- Seeds: `0, 1, 2, 3, 4, 5, 6, 7, 8, 9`

## Default Parameters

- `beta_hardness` = `0.7`
- `rho` = `0.6`
- `T_part` = `15`
- `T_refresh` = `5`
- `K_min` = `5`
- `score_floor` = `1e-9`
- `local_steps` = `20`
- `stepalloc_min_steps` = `15`
- `stepalloc_max_steps` = `25`
- `stepalloc_power` = `1.0`
- `ena_alpha` = `1.0`
- `ena_reference_mode` = `mean_effective`
- `aggregation_rule` = `ena`

## One-factor-at-a-time Grids

- CoRA-Core `beta_hardness`: `0.0, 0.3, 0.5, 0.7, 0.9`
- CoRA-Core `rho`: `0.5, 0.6, 0.7, 0.8, 1.0`
- CoRA-Core `T_part`: `0, 5, 10, 15, 20`
- CoRA-Core `T_refresh`: `0, 2, 5, 10`
- CoRA-StepAlloc `ena_alpha`: `0.5, 1.0, 1.5, 2.0`
- CoRA-StepAlloc `step_bounds`: `(10,30), (12,28), (15,25), (18,22), (20,20)`

## Notes

- `rho` changes the partial-round quota and therefore the operating budget/schedule. Treat this as operating-setting sensitivity, not same-budget performance comparison.
- `T_refresh=0` uses the existing runner convention for no periodic refresh.
- `(20,20)` is the uniform-step reference inside StepAlloc.
- Duplicate default-valued configurations are intentionally not de-duplicated because each belongs to a different one-factor sweep. This preserves the requested `28 configurations x 10 seeds = 280 jobs` design.

## Counts

- Expected configurations: `28`
- Expected jobs: `280`
- Generated jobs: `280`

| Sensitivity group | Jobs |
|---|---:|
| `core_T_part` | 50 |
| `core_T_refresh` | 40 |
| `core_beta` | 50 |
| `core_rho` | 50 |
| `stepalloc_alpha` | 40 |
| `stepalloc_step_bounds` | 50 |

## Metadata Columns

- `sensitivity_group`
- `sensitivity_param`
- `sensitivity_value`
- `sensitivity_method`
- `default_param_value`
- `is_default_value`
