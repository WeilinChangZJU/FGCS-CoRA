# Priority Coverage Report

Metrics scanned: 1990

## P0 Coverage Gate

Minimum threshold: at least 10 complete paired seeds for Random, CoRA-Core, and CoRA-StepAlloc in every required P0 scenario. For SAITS, FedAvg is preferred as a schedule-level reference and reported separately.

| Suite | Backbone | Dataset | Variant | Failure mode | Failure rate | Complete paired seeds | Status | Missing methods | Missing seeds by method |
|---|---|---|---|---|---:|---:|---|---|---|
| main | gru | ETTm1 | hetero | none | 0.00 | 20 | PASS | [] | {} |
| main | gru | ETTm1 | mcar_p10 | none | 0.00 | 20 | PASS | [] | {} |
| main | gru | Beijing_AQI | hetero | none | 0.00 | 20 | PASS | [] | {} |
| main | gru | Beijing_AQI | mcar_p10 | none | 0.00 | 20 | PASS | [] | {} |
| main | gru | PhysioNet2012 | hetero | none | 0.00 | 20 | PASS | [] | {} |
| main | gru | PhysioNet2012 | mcar_p10 | none | 0.00 | 20 | PASS | [] | {} |
| stress | gru | ETTm1 | hetero_h1 | none | 0.00 | 20 | PASS | [] | {} |
| stress | gru | ETTm1 | hetero_h2 | none | 0.00 | 20 | PASS | [] | {} |
| stress | gru | ETTm1 | hetero_h3 | none | 0.00 | 20 | PASS | [] | {} |
| stress | gru | ETTm1 | hetero_h4 | none | 0.00 | 20 | PASS | [] | {} |
| stress | gru | ETTm1 | mcar_p10 | none | 0.00 | 20 | PASS | [] | {} |
| stress | gru | ETTm1 | mcar_p20 | none | 0.00 | 20 | PASS | [] | {} |
| stress | gru | ETTm1 | mcar_p30 | none | 0.00 | 20 | PASS | [] | {} |
| stress | gru | ETTm1 | mcar_p40 | none | 0.00 | 20 | PASS | [] | {} |
| stress | gru | ETTm1 | mcar_p50 | none | 0.00 | 20 | PASS | [] | {} |
| stress | gru | ETTm1 | mcar_p60 | none | 0.00 | 20 | PASS | [] | {} |
| stress | gru | ETTm1 | mcar_p70 | none | 0.00 | 20 | PASS | [] | {} |
| stress | gru | ETTm1 | mcar_p80 | none | 0.00 | 20 | PASS | [] | {} |
| failure | gru | ETTm1 | hetero_h4 | uniform | 0.05 | 10 | PASS | [] | {} |
| failure | gru | ETTm1 | hetero_h4 | uniform | 0.10 | 10 | PASS | [] | {} |
| failure | gru | ETTm1 | hetero_h4 | uniform | 0.20 | 10 | PASS | [] | {} |
| failure | gru | ETTm1 | hetero_h4 | score_correlated | 0.05 | 10 | PASS | [] | {} |
| failure | gru | ETTm1 | hetero_h4 | score_correlated | 0.10 | 10 | PASS | [] | {} |
| failure | gru | ETTm1 | hetero_h4 | score_correlated | 0.20 | 10 | PASS | [] | {} |
| backbone | saits | ETTm1 | hetero_h4 | none | 0.00 | 10 | PASS | [] | {}; FedAvg seeds=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] |

Minimum threshold satisfied: YES
Final Section 5 can be drafted: YES

## Manifest Completion

| Manifest | Jobs | Missing expected JSONs |
|---|---:|---:|
| `P0A_main_central_paired.csv` | 180 | 0 |
| `P0B_stress_core_10seeds.csv` | 360 | 0 |
| `P0D_failure_h4_10seeds.csv` | 180 | 0 |
| `P0C_saits_h4_10seeds.csv` | 40 | 0 |

## Failure-Return Non-Return Check

Failure-return metrics include selected-client non-return cases: YES
- `results\fgcs_revision\failure\gru\seed_0\ETTm1\hetero_h4\score_correlated_r0p05\metrics_CoRA-Core_score_correlated_r0p05_ETTm1_hetero_h4.json`: valid_return_rate=0.9523195876288659, wasted_assigned_step_ratio=0.04768041237113402
- `results\fgcs_revision\failure\gru\seed_0\ETTm1\hetero_h4\score_correlated_r0p05\metrics_CoRA-StepAlloc_score_correlated_r0p05_ETTm1_hetero_h4.json`: valid_return_rate=0.9536082474226805, wasted_assigned_step_ratio=0.047036082474226804
- `results\fgcs_revision\failure\gru\seed_0\ETTm1\hetero_h4\score_correlated_r0p05\metrics_Random_score_correlated_r0p05_ETTm1_hetero_h4.json`: valid_return_rate=0.9510309278350515, wasted_assigned_step_ratio=0.04896907216494845
- `results\fgcs_revision\failure\gru\seed_0\ETTm1\hetero_h4\score_correlated_r0p10\metrics_CoRA-Core_score_correlated_r0p10_ETTm1_hetero_h4.json`: valid_return_rate=0.9162371134020618, wasted_assigned_step_ratio=0.08376288659793814
- `results\fgcs_revision\failure\gru\seed_0\ETTm1\hetero_h4\score_correlated_r0p10\metrics_CoRA-StepAlloc_score_correlated_r0p10_ETTm1_hetero_h4.json`: valid_return_rate=0.9175257731958762, wasted_assigned_step_ratio=0.0847938144329897
- `results\fgcs_revision\failure\gru\seed_0\ETTm1\hetero_h4\score_correlated_r0p10\metrics_Random_score_correlated_r0p10_ETTm1_hetero_h4.json`: valid_return_rate=0.913659793814433, wasted_assigned_step_ratio=0.08634020618556701
- `results\fgcs_revision\failure\gru\seed_0\ETTm1\hetero_h4\score_correlated_r0p20\metrics_CoRA-Core_score_correlated_r0p20_ETTm1_hetero_h4.json`: valid_return_rate=0.8208762886597938, wasted_assigned_step_ratio=0.1791237113402062
- `results\fgcs_revision\failure\gru\seed_0\ETTm1\hetero_h4\score_correlated_r0p20\metrics_CoRA-StepAlloc_score_correlated_r0p20_ETTm1_hetero_h4.json`: valid_return_rate=0.8234536082474226, wasted_assigned_step_ratio=0.1788659793814433
- `results\fgcs_revision\failure\gru\seed_0\ETTm1\hetero_h4\score_correlated_r0p20\metrics_Random_score_correlated_r0p20_ETTm1_hetero_h4.json`: valid_return_rate=0.8208762886597938, wasted_assigned_step_ratio=0.1791237113402062
- `results\fgcs_revision\failure\gru\seed_0\ETTm1\hetero_h4\uniform_r0p05\metrics_CoRA-Core_uniform_r0p05_ETTm1_hetero_h4.json`: valid_return_rate=0.9420103092783505, wasted_assigned_step_ratio=0.05798969072164949
- `results\fgcs_revision\failure\gru\seed_0\ETTm1\hetero_h4\uniform_r0p05\metrics_CoRA-StepAlloc_uniform_r0p05_ETTm1_hetero_h4.json`: valid_return_rate=0.9407216494845361, wasted_assigned_step_ratio=0.059342783505154637
- `results\fgcs_revision\failure\gru\seed_0\ETTm1\hetero_h4\uniform_r0p05\metrics_Random_uniform_r0p05_ETTm1_hetero_h4.json`: valid_return_rate=0.9407216494845361, wasted_assigned_step_ratio=0.059278350515463915
- `results\fgcs_revision\failure\gru\seed_0\ETTm1\hetero_h4\uniform_r0p10\metrics_CoRA-Core_uniform_r0p10_ETTm1_hetero_h4.json`: valid_return_rate=0.9085051546391752, wasted_assigned_step_ratio=0.09149484536082474
- `results\fgcs_revision\failure\gru\seed_0\ETTm1\hetero_h4\uniform_r0p10\metrics_CoRA-StepAlloc_uniform_r0p10_ETTm1_hetero_h4.json`: valid_return_rate=0.9085051546391752, wasted_assigned_step_ratio=0.09104381443298969
- `results\fgcs_revision\failure\gru\seed_0\ETTm1\hetero_h4\uniform_r0p10\metrics_Random_uniform_r0p10_ETTm1_hetero_h4.json`: valid_return_rate=0.9007731958762887, wasted_assigned_step_ratio=0.09922680412371133
- `results\fgcs_revision\failure\gru\seed_0\ETTm1\hetero_h4\uniform_r0p20\metrics_CoRA-Core_uniform_r0p20_ETTm1_hetero_h4.json`: valid_return_rate=0.7925257731958762, wasted_assigned_step_ratio=0.20747422680412372
- `results\fgcs_revision\failure\gru\seed_0\ETTm1\hetero_h4\uniform_r0p20\metrics_CoRA-StepAlloc_uniform_r0p20_ETTm1_hetero_h4.json`: valid_return_rate=0.7925257731958762, wasted_assigned_step_ratio=0.20650773195876287
- `results\fgcs_revision\failure\gru\seed_0\ETTm1\hetero_h4\uniform_r0p20\metrics_Random_uniform_r0p20_ETTm1_hetero_h4.json`: valid_return_rate=0.7899484536082474, wasted_assigned_step_ratio=0.21005154639175258
- `results\fgcs_revision\failure\gru\seed_0\ETTm1\mcar_p10\score_correlated_r0p05\metrics_CoRA-Core_score_correlated_r0p05_ETTm1_mcar_p10.json`: valid_return_rate=0.9587628865979382, wasted_assigned_step_ratio=0.041237113402061855
- `results\fgcs_revision\failure\gru\seed_0\ETTm1\mcar_p10\score_correlated_r0p05\metrics_CoRA-StepAlloc_score_correlated_r0p05_ETTm1_mcar_p10.json`: valid_return_rate=0.9587628865979382, wasted_assigned_step_ratio=0.04188144329896907

## Missing Jobs

No missing expected JSONs for P0 manifests.
