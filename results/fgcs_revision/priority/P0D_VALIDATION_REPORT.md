# P0-D Validation Report

Generated: `2026-06-29 04:10:36 +0800`

## Coverage

| Mode | Rate | Complete paired seeds | Valid return mean | Wasted steps mean | Uniform expected valid |
|---|---:|---:|---:|---:|---:|
| uniform | 0.05 | 10 | 0.9454 | 0.0546 | 0.9500 |
| uniform | 0.10 | 10 | 0.8972 | 0.1027 | 0.9000 |
| uniform | 0.20 | 10 | 0.7913 | 0.2088 | 0.8000 |
| score_correlated | 0.05 | 10 | 0.9480 | 0.0522 | 0.9500 |
| score_correlated | 0.10 | 10 | 0.9009 | 0.0995 | 0.9000 |
| score_correlated | 0.20 | 10 | 0.8025 | 0.1984 | 0.8000 |

## Score-Correlated Failure Diagnostic

- Trace selected-client records: `69840`.
- Correlation between `omega_before` and assigned non-return probability: `0.0438`.
- Bottom quartile mean non-return probability / observed fail rate: `0.0916` / `0.0829`.
- Top quartile mean non-return probability / observed fail rate: `0.1238` / `0.1239`.
