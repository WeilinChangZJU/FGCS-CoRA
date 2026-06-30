# Priority Manifest README

Generated from existing manifest CSV files. Filters were applied to exact values inferred from each manifest rather than hard-coded assumptions.

## Source `main`

- Path: `D:\CoRA\results\fgcs_revision\manifest_main.csv`
- Rows: 840
- Header: `suite, backbone, seed, dataset, variant, method_label, method_canonical, failure_mode, failure_rate, out_dir, expected_json, argv_json, command`
- Detected columns: backbone=backbone, dataset=dataset, failure_mode=failure_mode, failure_rate=failure_rate, method=method_label, seed=seed, suite=suite, variant=variant

Available values:
- `dataset`: Beijing_AQI, ETTm1, PhysioNet2012
- `variant`: hetero, mcar_p10
- `method_label`: CoRA-Core, CoRA-StepAlloc, FedAvg, FedProx, LocalOnly, Random, qFedAvg
- `seed`: 0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 3, 4, 5, 6, 7, 8, 9
- `failure_mode`: none
- `failure_rate`: 0.0

## Source `stress`

- Path: `D:\CoRA\results\fgcs_revision\manifest_stress.csv`
- Rows: 1200
- Header: `suite, backbone, seed, dataset, variant, method_label, method_canonical, failure_mode, failure_rate, out_dir, expected_json, argv_json, command`
- Detected columns: backbone=backbone, dataset=dataset, failure_mode=failure_mode, failure_rate=failure_rate, method=method_label, seed=seed, suite=suite, variant=variant

Available values:
- `dataset`: ETTm1
- `variant`: hetero_h1, hetero_h2, hetero_h3, hetero_h4, mcar_p10, mcar_p20, mcar_p30, mcar_p40, mcar_p50, mcar_p60, mcar_p70, mcar_p80
- `method_label`: CoRA-Core, CoRA-StepAlloc, FedAvg, Random, Top-m
- `seed`: 0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 3, 4, 5, 6, 7, 8, 9
- `failure_mode`: none
- `failure_rate`: 0.0

## Source `failure`

- Path: `D:\CoRA\results\fgcs_revision\manifest_failure.csv`
- Rows: 360
- Header: `suite, backbone, seed, dataset, variant, method_label, method_canonical, failure_mode, failure_rate, out_dir, expected_json, argv_json, command`
- Detected columns: backbone=backbone, dataset=dataset, failure_mode=failure_mode, failure_rate=failure_rate, method=method_label, seed=seed, suite=suite, variant=variant

Available values:
- `dataset`: ETTm1
- `variant`: hetero_h4, mcar_p10
- `method_label`: CoRA-Core, CoRA-StepAlloc, Random
- `seed`: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
- `failure_mode`: score_correlated, uniform
- `failure_rate`: 0.05, 0.1, 0.2

## Source `backbone`

- Path: `D:\CoRA\results\fgcs_revision\manifest_backbone.csv`
- Rows: 100
- Header: `suite, backbone, seed, dataset, variant, method_label, method_canonical, failure_mode, failure_rate, out_dir, expected_json, argv_json, command`
- Detected columns: backbone=backbone, dataset=dataset, failure_mode=failure_mode, failure_rate=failure_rate, method=method_label, seed=seed, suite=suite, variant=variant

Available values:
- `dataset`: ETTm1
- `variant`: hetero_h4, mcar_p10
- `method_label`: CoRA-Core, CoRA-StepAlloc, FedAvg, Random
- `seed`: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
- `failure_mode`: none
- `failure_rate`: 0.0

## `P0A_main_central_paired.csv`

- Source: `main`
- Rows: 180
- Seeds requested: `0, 1, 2, 3, 4, 5, 6, 7, 8, 9`
- Backbones: {'gru': 'gru'}
- Datasets: {'ETTm1': 'ETTm1', 'Beijing_AQI': 'Beijing_AQI', 'PhysioNet2012': 'PhysioNet2012'}
- Variants/settings: {'hetero': 'hetero', 'mcar_p10': 'mcar_p10'}
- Methods: {'Random': 'Random', 'CoRA-Core': 'CoRA-Core', 'CoRA-StepAlloc': 'CoRA-StepAlloc'}
- Scenario-method row counts:
  - `Beijing_AQI` / `hetero` / `CoRA-Core`: 10
  - `Beijing_AQI` / `hetero` / `CoRA-StepAlloc`: 10
  - `Beijing_AQI` / `hetero` / `Random`: 10
  - `Beijing_AQI` / `mcar_p10` / `CoRA-Core`: 10
  - `Beijing_AQI` / `mcar_p10` / `CoRA-StepAlloc`: 10
  - `Beijing_AQI` / `mcar_p10` / `Random`: 10
  - `ETTm1` / `hetero` / `CoRA-Core`: 10
  - `ETTm1` / `hetero` / `CoRA-StepAlloc`: 10
  - `ETTm1` / `hetero` / `Random`: 10
  - `ETTm1` / `mcar_p10` / `CoRA-Core`: 10
  - `ETTm1` / `mcar_p10` / `CoRA-StepAlloc`: 10
  - `ETTm1` / `mcar_p10` / `Random`: 10
  - `PhysioNet2012` / `hetero` / `CoRA-Core`: 10
  - `PhysioNet2012` / `hetero` / `CoRA-StepAlloc`: 10
  - `PhysioNet2012` / `hetero` / `Random`: 10
  - `PhysioNet2012` / `mcar_p10` / `CoRA-Core`: 10
  - `PhysioNet2012` / `mcar_p10` / `CoRA-StepAlloc`: 10
  - `PhysioNet2012` / `mcar_p10` / `Random`: 10

## `P0B_stress_core_10seeds.csv`

- Source: `stress`
- Rows: 360
- Seeds requested: `0, 1, 2, 3, 4, 5, 6, 7, 8, 9`
- Backbones: {'gru': 'gru'}
- Datasets: {'ETTm1': 'ETTm1'}
- Variants/settings: {'hetero_h1': 'hetero_h1', 'hetero_h2': 'hetero_h2', 'hetero_h3': 'hetero_h3', 'hetero_h4': 'hetero_h4', 'mcar_p10': 'mcar_p10', 'mcar_p20': 'mcar_p20', 'mcar_p30': 'mcar_p30', 'mcar_p40': 'mcar_p40', 'mcar_p50': 'mcar_p50', 'mcar_p60': 'mcar_p60', 'mcar_p70': 'mcar_p70', 'mcar_p80': 'mcar_p80'}
- Methods: {'Random': 'Random', 'CoRA-Core': 'CoRA-Core', 'CoRA-StepAlloc': 'CoRA-StepAlloc'}
- Scenario-method row counts:
  - `ETTm1` / `hetero_h1` / `CoRA-Core`: 10
  - `ETTm1` / `hetero_h1` / `CoRA-StepAlloc`: 10
  - `ETTm1` / `hetero_h1` / `Random`: 10
  - `ETTm1` / `hetero_h2` / `CoRA-Core`: 10
  - `ETTm1` / `hetero_h2` / `CoRA-StepAlloc`: 10
  - `ETTm1` / `hetero_h2` / `Random`: 10
  - `ETTm1` / `hetero_h3` / `CoRA-Core`: 10
  - `ETTm1` / `hetero_h3` / `CoRA-StepAlloc`: 10
  - `ETTm1` / `hetero_h3` / `Random`: 10
  - `ETTm1` / `hetero_h4` / `CoRA-Core`: 10
  - `ETTm1` / `hetero_h4` / `CoRA-StepAlloc`: 10
  - `ETTm1` / `hetero_h4` / `Random`: 10
  - `ETTm1` / `mcar_p10` / `CoRA-Core`: 10
  - `ETTm1` / `mcar_p10` / `CoRA-StepAlloc`: 10
  - `ETTm1` / `mcar_p10` / `Random`: 10
  - `ETTm1` / `mcar_p20` / `CoRA-Core`: 10
  - `ETTm1` / `mcar_p20` / `CoRA-StepAlloc`: 10
  - `ETTm1` / `mcar_p20` / `Random`: 10
  - `ETTm1` / `mcar_p30` / `CoRA-Core`: 10
  - `ETTm1` / `mcar_p30` / `CoRA-StepAlloc`: 10
  - `ETTm1` / `mcar_p30` / `Random`: 10
  - `ETTm1` / `mcar_p40` / `CoRA-Core`: 10
  - `ETTm1` / `mcar_p40` / `CoRA-StepAlloc`: 10
  - `ETTm1` / `mcar_p40` / `Random`: 10
  - `ETTm1` / `mcar_p50` / `CoRA-Core`: 10
  - `ETTm1` / `mcar_p50` / `CoRA-StepAlloc`: 10
  - `ETTm1` / `mcar_p50` / `Random`: 10
  - `ETTm1` / `mcar_p60` / `CoRA-Core`: 10
  - `ETTm1` / `mcar_p60` / `CoRA-StepAlloc`: 10
  - `ETTm1` / `mcar_p60` / `Random`: 10
  - `ETTm1` / `mcar_p70` / `CoRA-Core`: 10
  - `ETTm1` / `mcar_p70` / `CoRA-StepAlloc`: 10
  - `ETTm1` / `mcar_p70` / `Random`: 10
  - `ETTm1` / `mcar_p80` / `CoRA-Core`: 10
  - `ETTm1` / `mcar_p80` / `CoRA-StepAlloc`: 10
  - `ETTm1` / `mcar_p80` / `Random`: 10

## `P0C_saits_h4_10seeds.csv`

- Source: `backbone`
- Rows: 40
- Seeds requested: `0, 1, 2, 3, 4, 5, 6, 7, 8, 9`
- Backbones: {'saits': 'saits'}
- Datasets: {'ETTm1': 'ETTm1'}
- Variants/settings: {'hetero_h4': 'hetero_h4'}
- Methods: {'FedAvg': 'FedAvg', 'Random': 'Random', 'CoRA-Core': 'CoRA-Core', 'CoRA-StepAlloc': 'CoRA-StepAlloc'}
- Scenario-method row counts:
  - `ETTm1` / `hetero_h4` / `CoRA-Core`: 10
  - `ETTm1` / `hetero_h4` / `CoRA-StepAlloc`: 10
  - `ETTm1` / `hetero_h4` / `FedAvg`: 10
  - `ETTm1` / `hetero_h4` / `Random`: 10

## `P0D_failure_h4_10seeds.csv`

- Source: `failure`
- Rows: 180
- Seeds requested: `0, 1, 2, 3, 4, 5, 6, 7, 8, 9`
- Backbones: {'gru': 'gru'}
- Datasets: {'ETTm1': 'ETTm1'}
- Variants/settings: {'hetero_h4': 'hetero_h4'}
- Methods: {'Random': 'Random', 'CoRA-Core': 'CoRA-Core', 'CoRA-StepAlloc': 'CoRA-StepAlloc'}
- Failure modes: {'uniform': 'uniform', 'score_correlated': 'score_correlated'}
- Failure rates: 0.05, 0.1, 0.2
- Scenario-method row counts:
  - `ETTm1` / `hetero_h4` / `CoRA-Core`: 60
  - `ETTm1` / `hetero_h4` / `CoRA-StepAlloc`: 60
  - `ETTm1` / `hetero_h4` / `Random`: 60

## `P1_main_full_refs_10seeds.csv`

- Source: `main`
- Rows: 240
- Seeds requested: `0, 1, 2, 3, 4, 5, 6, 7, 8, 9`
- Backbones: {'gru': 'gru'}
- Datasets: {'ETTm1': 'ETTm1', 'Beijing_AQI': 'Beijing_AQI', 'PhysioNet2012': 'PhysioNet2012'}
- Variants/settings: {'hetero': 'hetero', 'mcar_p10': 'mcar_p10'}
- Methods: {'FedAvg': 'FedAvg', 'FedProx': 'FedProx', 'qFedAvg': 'qFedAvg', 'LocalOnly': 'LocalOnly'}
- Scenario-method row counts:
  - `Beijing_AQI` / `hetero` / `FedAvg`: 10
  - `Beijing_AQI` / `hetero` / `FedProx`: 10
  - `Beijing_AQI` / `hetero` / `LocalOnly`: 10
  - `Beijing_AQI` / `hetero` / `qFedAvg`: 10
  - `Beijing_AQI` / `mcar_p10` / `FedAvg`: 10
  - `Beijing_AQI` / `mcar_p10` / `FedProx`: 10
  - `Beijing_AQI` / `mcar_p10` / `LocalOnly`: 10
  - `Beijing_AQI` / `mcar_p10` / `qFedAvg`: 10
  - `ETTm1` / `hetero` / `FedAvg`: 10
  - `ETTm1` / `hetero` / `FedProx`: 10
  - `ETTm1` / `hetero` / `LocalOnly`: 10
  - `ETTm1` / `hetero` / `qFedAvg`: 10
  - `ETTm1` / `mcar_p10` / `FedAvg`: 10
  - `ETTm1` / `mcar_p10` / `FedProx`: 10
  - `ETTm1` / `mcar_p10` / `LocalOnly`: 10
  - `ETTm1` / `mcar_p10` / `qFedAvg`: 10
  - `PhysioNet2012` / `hetero` / `FedAvg`: 10
  - `PhysioNet2012` / `hetero` / `FedProx`: 10
  - `PhysioNet2012` / `hetero` / `LocalOnly`: 10
  - `PhysioNet2012` / `hetero` / `qFedAvg`: 10
  - `PhysioNet2012` / `mcar_p10` / `FedAvg`: 10
  - `PhysioNet2012` / `mcar_p10` / `FedProx`: 10
  - `PhysioNet2012` / `mcar_p10` / `LocalOnly`: 10
  - `PhysioNet2012` / `mcar_p10` / `qFedAvg`: 10

## `P1_stress_fedavg_10seeds.csv`

- Source: `stress`
- Rows: 120
- Seeds requested: `0, 1, 2, 3, 4, 5, 6, 7, 8, 9`
- Backbones: {'gru': 'gru'}
- Datasets: {'ETTm1': 'ETTm1'}
- Variants/settings: {'hetero_h1': 'hetero_h1', 'hetero_h2': 'hetero_h2', 'hetero_h3': 'hetero_h3', 'hetero_h4': 'hetero_h4', 'mcar_p10': 'mcar_p10', 'mcar_p20': 'mcar_p20', 'mcar_p30': 'mcar_p30', 'mcar_p40': 'mcar_p40', 'mcar_p50': 'mcar_p50', 'mcar_p60': 'mcar_p60', 'mcar_p70': 'mcar_p70', 'mcar_p80': 'mcar_p80'}
- Methods: {'FedAvg': 'FedAvg'}
- Scenario-method row counts:
  - `ETTm1` / `hetero_h1` / `FedAvg`: 10
  - `ETTm1` / `hetero_h2` / `FedAvg`: 10
  - `ETTm1` / `hetero_h3` / `FedAvg`: 10
  - `ETTm1` / `hetero_h4` / `FedAvg`: 10
  - `ETTm1` / `mcar_p10` / `FedAvg`: 10
  - `ETTm1` / `mcar_p20` / `FedAvg`: 10
  - `ETTm1` / `mcar_p30` / `FedAvg`: 10
  - `ETTm1` / `mcar_p40` / `FedAvg`: 10
  - `ETTm1` / `mcar_p50` / `FedAvg`: 10
  - `ETTm1` / `mcar_p60` / `FedAvg`: 10
  - `ETTm1` / `mcar_p70` / `FedAvg`: 10
  - `ETTm1` / `mcar_p80` / `FedAvg`: 10

## `P1_failure_mcar10_10seeds.csv`

- Source: `failure`
- Rows: 180
- Seeds requested: `0, 1, 2, 3, 4, 5, 6, 7, 8, 9`
- Backbones: {'gru': 'gru'}
- Datasets: {'ETTm1': 'ETTm1'}
- Variants/settings: {'mcar_p10': 'mcar_p10'}
- Methods: {'Random': 'Random', 'CoRA-Core': 'CoRA-Core', 'CoRA-StepAlloc': 'CoRA-StepAlloc'}
- Failure modes: {'uniform': 'uniform', 'score_correlated': 'score_correlated'}
- Failure rates: 0.05, 0.1, 0.2
- Scenario-method row counts:
  - `ETTm1` / `mcar_p10` / `CoRA-Core`: 60
  - `ETTm1` / `mcar_p10` / `CoRA-StepAlloc`: 60
  - `ETTm1` / `mcar_p10` / `Random`: 60

## `P1_saits_mcar10_10seeds.csv`

- Source: `backbone`
- Rows: 40
- Seeds requested: `0, 1, 2, 3, 4, 5, 6, 7, 8, 9`
- Backbones: {'saits': 'saits'}
- Datasets: {'ETTm1': 'ETTm1'}
- Variants/settings: {'mcar_p10': 'mcar_p10'}
- Methods: {'FedAvg': 'FedAvg', 'Random': 'Random', 'CoRA-Core': 'CoRA-Core', 'CoRA-StepAlloc': 'CoRA-StepAlloc'}
- Scenario-method row counts:
  - `ETTm1` / `mcar_p10` / `CoRA-Core`: 10
  - `ETTm1` / `mcar_p10` / `CoRA-StepAlloc`: 10
  - `ETTm1` / `mcar_p10` / `FedAvg`: 10
  - `ETTm1` / `mcar_p10` / `Random`: 10

## `P2_csdi_h4_5seeds.csv`

- Source: `backbone`
- Rows: 20
- Seeds requested: `0, 1, 2, 3, 4`
- Backbones: {'csdi': 'csdi'}
- Datasets: {'ETTm1': 'ETTm1'}
- Variants/settings: {'hetero_h4': 'hetero_h4'}
- Methods: {'FedAvg': 'FedAvg', 'Random': 'Random', 'CoRA-Core': 'CoRA-Core', 'CoRA-StepAlloc': 'CoRA-StepAlloc'}
- Scenario-method row counts:
  - `ETTm1` / `hetero_h4` / `CoRA-Core`: 5
  - `ETTm1` / `hetero_h4` / `CoRA-StepAlloc`: 5
  - `ETTm1` / `hetero_h4` / `FedAvg`: 5
  - `ETTm1` / `hetero_h4` / `Random`: 5

## Notes

- P0-A uses 10 paired seeds (0:9) rather than 20 because the previous full-run timing showed the 20-seed package is prohibitive on the current single GPU.
