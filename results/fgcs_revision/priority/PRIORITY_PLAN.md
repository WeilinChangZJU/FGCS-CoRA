# Priority Revision Execution Plan

P0 is the minimum reviewer-compliant package for result-grounded Section 5 drafting. P1/P2 are completeness and robustness additions and should not be run until all P0 manifests finish and coverage passes.

## Manifest Counts and Runtime Estimates

| Manifest | Jobs | Estimate |
|---|---:|---:|
| `P0A_main_central_paired.csv` | 180 | 6.12 h |
| `P0B_stress_core_10seeds.csv` | 360 | 12.23 h |
| `P0C_saits_h4_10seeds.csv` | 40 | 3.30 h |
| `P0D_failure_h4_10seeds.csv` | 180 | 4.55 h |
| `P1_main_full_refs_10seeds.csv` | 240 | 8.15 h |
| `P1_stress_fedavg_10seeds.csv` | 120 | 4.08 h |
| `P1_failure_mcar10_10seeds.csv` | 180 | 4.55 h |
| `P1_saits_mcar10_10seeds.csv` | 40 | 3.30 h |
| `P2_csdi_h4_5seeds.csv` | 20 | 3.33 h |

Runtime estimates use observed smoke-job timings: GRU non-failure 122.3 s/job, GRU failure-return 91.0 s/job, SAITS 297.3 s/job, and placeholder CSDI 600.0 s/job.

## Exact Filters

- P0-A: `manifest_main.csv`; GRU; ETTm1, Beijing_AQI, PhysioNet2012; hetero and mcar_p10; Random, CoRA-Core, CoRA-StepAlloc; seeds 0:9.
- P0-B: `manifest_stress.csv`; GRU; ETTm1; H1-H4 and MCAR 10%-80%; Random, CoRA-Core, CoRA-StepAlloc; seeds 0:9.
- P0-C: `manifest_backbone.csv`; SAITS; ETTm1 hetero_h4; FedAvg, Random, CoRA-Core, CoRA-StepAlloc; seeds 0:9.
- P0-D: `manifest_failure.csv`; GRU; ETTm1 hetero_h4; uniform and score_correlated; 0.05, 0.10, 0.20; Random, CoRA-Core, CoRA-StepAlloc; seeds 0:9.
- P1/P2: generated for optional completeness only; not part of the minimum P0 drafting gate.

## Execution Order

1. `P0A_main_central_paired.csv` (180 jobs, estimated 6.12 h)
2. `P0B_stress_core_10seeds.csv` (360 jobs, estimated 12.23 h)
3. `P0D_failure_h4_10seeds.csv` (180 jobs, estimated 4.55 h)
4. `P0C_saits_h4_10seeds.csv` (40 jobs, estimated 3.30 h)

## Optional Manifests

- `P1_main_full_refs_10seeds.csv` (240 jobs, estimated 8.15 h)
- `P1_stress_fedavg_10seeds.csv` (120 jobs, estimated 4.08 h)
- `P1_failure_mcar10_10seeds.csv` (180 jobs, estimated 4.55 h)
- `P1_saits_mcar10_10seeds.csv` (40 jobs, estimated 3.30 h)
- `P2_csdi_h4_5seeds.csv` (20 jobs, estimated 3.33 h)

## Drafting Gate

Final Section 5 must not be drafted until P0 coverage reports at least 10 complete paired seeds for all central Random/Core/StepAlloc scenarios and regenerated final summaries, tables, and figures are available.
