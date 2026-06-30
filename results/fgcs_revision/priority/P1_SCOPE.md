# P1 Scope Decision Record

Generated: `2026-06-28 19:49:10 +0800`

- P1-core includes main full references, stress FedAvg, failure mcar10, and SAITS mcar10.
- P1-core does not include P2/CSDI.
- Advisor decision: ablation, mechanism verification, and sensitivity analysis are retained.
- `P1_ablation_10seeds.csv` is mechanism ablation and is enabled after P1-core.
- `P1_sensitivity_light_10seeds.csv` is hyperparameter sensitivity, not ablation. It starts only after P1-core and P1-ablation packaging complete, or after P1-ablation is explicitly marked skipped/failed with a written reason.
- P2/CSDI remains disabled unless explicitly instructed later.
