# P0-A+ / P0-B+ Manifest README

Purpose: extend completed P0-A/P0-B GRU central comparisons from paired seeds 0-9 to paired seeds 0-19 by adding seeds 10-19 only.

No algorithmic implementation, dataset, mask, model protocol, or summarizer changes are made by this script.

Seeds: `10, 11, 12, 13, 14, 15, 16, 17, 18, 19`

## `P0A_plus_main_central_paired_seeds10_19.csv`

- Source: `D:\CoRA\results\fgcs_revision\manifest_main.csv`
- Output rows: 180
- Expected rows: 180
- Header: `suite, backbone, seed, dataset, variant, method_label, method_canonical, failure_mode, failure_rate, out_dir, expected_json, argv_json, command`
- Detected columns: backbone=backbone, dataset=dataset, failure_mode=failure_mode, failure_rate=failure_rate, method=method_label, seed=seed, variant=variant
- Backbone filter: {'gru': 'gru'}
- Dataset filters: {'ETTm1': 'ETTm1', 'Beijing_AQI': 'Beijing_AQI', 'PhysioNet2012': 'PhysioNet2012'}
- Variant filters: {'hetero': 'hetero', 'mcar_p10': 'mcar_p10'}
- Method filters: {'Random': 'Random', 'CoRA-Core': 'CoRA-Core', 'CoRA-StepAlloc': 'CoRA-StepAlloc'}
- Failure filter: `failure_mode=none`, `failure_rate=0.0`

Scenario-method row counts:
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

## `P0B_plus_stress_core_seeds10_19.csv`

- Source: `D:\CoRA\results\fgcs_revision\manifest_stress.csv`
- Output rows: 360
- Expected rows: 360
- Header: `suite, backbone, seed, dataset, variant, method_label, method_canonical, failure_mode, failure_rate, out_dir, expected_json, argv_json, command`
- Detected columns: backbone=backbone, dataset=dataset, failure_mode=failure_mode, failure_rate=failure_rate, method=method_label, seed=seed, variant=variant
- Backbone filter: {'gru': 'gru'}
- Dataset filters: {'ETTm1': 'ETTm1'}
- Variant filters: {'hetero_h1': 'hetero_h1', 'hetero_h2': 'hetero_h2', 'hetero_h3': 'hetero_h3', 'hetero_h4': 'hetero_h4', 'mcar_p10': 'mcar_p10', 'mcar_p20': 'mcar_p20', 'mcar_p30': 'mcar_p30', 'mcar_p40': 'mcar_p40', 'mcar_p50': 'mcar_p50', 'mcar_p60': 'mcar_p60', 'mcar_p70': 'mcar_p70', 'mcar_p80': 'mcar_p80'}
- Method filters: {'Random': 'Random', 'CoRA-Core': 'CoRA-Core', 'CoRA-StepAlloc': 'CoRA-StepAlloc'}
- Failure filter: `failure_mode=none`, `failure_rate=0.0`

Scenario-method row counts:
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
