from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "fgcs_revision"
PRIORITY = RESULTS / "priority"
SOURCE_MANIFEST = PRIORITY / "P0B_stress_core_10seeds.csv"
OUT_MANIFEST = PRIORITY / "P1_sensitivity_light_10seeds.csv"
README = PRIORITY / "P1_SENSITIVITY_LIGHT_README.md"

DATASET = "ETTm1"
VARIANT = "hetero_h4"
BACKBONE = "gru"
SEEDS = list(range(10))
EXPECTED_ROWS = 280

DEFAULTS = {
    "beta_hardness": "0.7",
    "rho": "0.6",
    "T_part": "15",
    "T_refresh": "5",
    "K_min": "5",
    "score_floor": "1e-9",
    "local_steps": "20",
    "stepalloc_min_steps": "15",
    "stepalloc_max_steps": "25",
    "stepalloc_power": "1.0",
    "ena_alpha": "1.0",
    "ena_reference_mode": "mean_effective",
    "aggregation_rule": "ena",
}


def norm(x: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(x).strip().lower())


def slug(x: Any) -> str:
    s = str(x).strip()
    s = s.replace("e-09", "em9").replace("e-9", "em9")
    s = s.replace(".", "p").replace("-", "m").replace("+", "")
    return re.sub(r"[^A-Za-z0-9_]+", "_", s).strip("_")


def read_manifest(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise RuntimeError(f"Manifest has no header: {path}")
        return list(reader.fieldnames), list(reader)


def parse_argv(row: dict[str, str]) -> list[str]:
    raw = row.get("argv_json", "")
    argv = json.loads(raw)
    if not isinstance(argv, list):
        raise RuntimeError("argv_json is not a list")
    return [str(x) for x in argv]


def set_arg(argv: list[str], key: str, value: Any) -> list[str]:
    value = str(value)
    out = list(argv)
    if key in out:
        idx = out.index(key)
        if idx == len(out) - 1:
            raise RuntimeError(f"Argument {key} has no value")
        out[idx + 1] = value
    else:
        out.extend([key, value])
    return out


def remove_arg(argv: list[str], key: str) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(argv):
        if argv[i] == key:
            i += 2
            continue
        out.append(argv[i])
        i += 1
    return out


def shell_command(argv: Sequence[str]) -> str:
    return subprocess.list2cmdline(list(argv))


def find_template(rows: Sequence[dict[str, str]], method_label: str) -> dict[str, str]:
    for row in rows:
        if (
            norm(row.get("suite", "")) == "stress"
            and norm(row.get("backbone", "")) == norm(BACKBONE)
            and norm(row.get("dataset", "")) == norm(DATASET)
            and norm(row.get("variant", "")) == norm(VARIANT)
            and str(row.get("seed", "")) in {"0", "0.0"}
            and norm(row.get("method_label", "")) == norm(method_label)
        ):
            return dict(row)
    raise RuntimeError(f"Could not find sensitivity template row for {method_label} in {SOURCE_MANIFEST}")


def base_argv(template: dict[str, str], method_canonical: str) -> list[str]:
    argv = parse_argv(template)
    argv = set_arg(argv, "--data_root", "data_stress")
    argv = set_arg(argv, "--dataset", DATASET)
    argv = set_arg(argv, "--variant", VARIANT)
    argv = set_arg(argv, "--backbone", BACKBONE)
    argv = set_arg(argv, "--rounds", "50")
    argv = set_arg(argv, "--local_steps", DEFAULTS["local_steps"])
    argv = set_arg(argv, "--batch_size", "64")
    argv = set_arg(argv, "--eval_every", "5")
    argv = set_arg(argv, "--eval_split", "val")
    argv = set_arg(argv, "--checkpoint_selection", "best_val_rmse_avg")
    argv = set_arg(argv, "--h0", "1.0")
    argv = set_arg(argv, "--K_min", DEFAULTS["K_min"])
    argv = set_arg(argv, "--score_floor", DEFAULTS["score_floor"])
    argv = set_arg(argv, "--stepalloc_power", DEFAULTS["stepalloc_power"])
    argv = set_arg(argv, "--method", method_canonical)
    for key in [
        "--failure_mode",
        "--failure_rate",
        "--failure_qmax",
        "--failure_apply_full_rounds",
        "--failure_execute_training",
    ]:
        argv = remove_arg(argv, key)
    return argv


def sensitivity_configs() -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for value in ["0.0", "0.3", "0.5", "0.7", "0.9"]:
        configs.append(
            {
                "group": "core_beta",
                "param": "beta_hardness",
                "value": value,
                "method_label_base": "CoRA-Core",
                "method_canonical": "cora_core",
                "default": DEFAULTS["beta_hardness"],
                "overrides": {"--beta_hardness": value},
            }
        )
    for value in ["0.5", "0.6", "0.7", "0.8", "1.0"]:
        configs.append(
            {
                "group": "core_rho",
                "param": "rho",
                "value": value,
                "method_label_base": "CoRA-Core",
                "method_canonical": "cora_core",
                "default": DEFAULTS["rho"],
                "overrides": {"--rho": value},
                "note": "rho changes the partial-round quota and is operating-setting sensitivity, not same-budget comparison.",
            }
        )
    for value in ["0", "5", "10", "15", "20"]:
        configs.append(
            {
                "group": "core_T_part",
                "param": "T_part",
                "value": value,
                "method_label_base": "CoRA-Core",
                "method_canonical": "cora_core",
                "default": DEFAULTS["T_part"],
                "overrides": {"--T_part": value},
            }
        )
    for value in ["0", "2", "5", "10"]:
        configs.append(
            {
                "group": "core_T_refresh",
                "param": "T_refresh",
                "value": value,
                "method_label_base": "CoRA-Core",
                "method_canonical": "cora_core",
                "default": DEFAULTS["T_refresh"],
                "overrides": {"--T_refresh": value},
                "note": "T_refresh=0 uses the runner's existing no-periodic-refresh convention.",
            }
        )
    for value in ["0.5", "1.0", "1.5", "2.0"]:
        configs.append(
            {
                "group": "stepalloc_alpha",
                "param": "ena_alpha",
                "value": value,
                "method_label_base": "CoRA-StepAlloc",
                "method_canonical": "cora_stepalloc",
                "default": DEFAULTS["ena_alpha"],
                "overrides": {"--ena_alpha": value},
            }
        )
    for lo, hi in [(10, 30), (12, 28), (15, 25), (18, 22), (20, 20)]:
        value = f"{lo}_{hi}"
        configs.append(
            {
                "group": "stepalloc_step_bounds",
                "param": "step_bounds",
                "value": value,
                "method_label_base": "CoRA-StepAlloc",
                "method_canonical": "cora_stepalloc",
                "default": "15_25",
                "overrides": {"--stepalloc_min_steps": str(lo), "--stepalloc_max_steps": str(hi)},
                "note": "(20,20) is the uniform-step reference within StepAlloc.",
            }
        )
    return configs


def full_overrides(config: dict[str, Any]) -> dict[str, str]:
    overrides = {
        "--beta_hardness": DEFAULTS["beta_hardness"],
        "--rho": DEFAULTS["rho"],
        "--T_part": DEFAULTS["T_part"],
        "--T_refresh": DEFAULTS["T_refresh"],
        "--K_min": DEFAULTS["K_min"],
        "--score_floor": DEFAULTS["score_floor"],
        "--stepalloc_min_steps": DEFAULTS["stepalloc_min_steps"],
        "--stepalloc_max_steps": DEFAULTS["stepalloc_max_steps"],
        "--stepalloc_power": DEFAULTS["stepalloc_power"],
    }
    if str(config["method_canonical"]) == "cora_stepalloc":
        overrides.update(
            {
                "--aggregation_rule": DEFAULTS["aggregation_rule"],
                "--ena_reference_mode": DEFAULTS["ena_reference_mode"],
                "--ena_alpha": DEFAULTS["ena_alpha"],
            }
        )
    overrides.update({str(k): str(v) for k, v in dict(config["overrides"]).items()})
    return overrides


def build_rows(header: Sequence[str], templates: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen_expected: set[str] = set()
    configs = sensitivity_configs()
    for config in configs:
        base_method = str(config["method_label_base"])
        method_canonical = str(config["method_canonical"])
        template = templates[base_method]
        for seed in SEEDS:
            method_label = f"{base_method}_{config['param']}_{slug(config['value'])}"
            tag = f"sens_{method_label}"
            out_dir = (
                RESULTS
                / "sensitivity"
                / BACKBONE
                / f"seed_{seed}"
                / DATASET
                / VARIANT
                / str(config["group"])
                / f"{config['param']}_{slug(config['value'])}"
            )
            expected = out_dir / f"metrics_{tag}_{DATASET}_{VARIANT}.json"
            rel_out = str(out_dir.relative_to(ROOT))
            rel_expected = str(expected.relative_to(ROOT))
            if rel_expected in seen_expected:
                raise RuntimeError(f"Expected JSON collision: {rel_expected}")
            seen_expected.add(rel_expected)

            argv = base_argv(template, method_canonical)
            argv = set_arg(argv, "--seed", seed)
            argv = set_arg(argv, "--out_dir", rel_out)
            argv = set_arg(argv, "--tag", tag)
            for key, value in full_overrides(config).items():
                argv = set_arg(argv, key, value)

            row = {h: template.get(h, "") for h in header}
            row.update(
                {
                    "suite": "sensitivity",
                    "backbone": BACKBONE,
                    "seed": str(seed),
                    "dataset": DATASET,
                    "variant": VARIANT,
                    "method_label": method_label,
                    "method_canonical": method_canonical,
                    "failure_mode": "none",
                    "failure_rate": "0.0",
                    "out_dir": rel_out,
                    "expected_json": rel_expected,
                    "argv_json": json.dumps(argv),
                    "command": shell_command(argv),
                    "sensitivity_group": str(config["group"]),
                    "sensitivity_param": str(config["param"]),
                    "sensitivity_value": str(config["value"]),
                    "sensitivity_method": base_method,
                    "default_param_value": str(config["default"]),
                    "is_default_value": "1" if str(config["value"]) == str(config["default"]) else "0",
                }
            )
            rows.append(row)
    return rows


def write_manifest(path: Path, header: Sequence[str], rows: Sequence[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(header))
        writer.writeheader()
        writer.writerows(rows)


def write_readme(rows: Sequence[dict[str, str]], configs: Sequence[dict[str, Any]]) -> None:
    group_counts: dict[str, int] = {}
    for row in rows:
        group_counts[row["sensitivity_group"]] = group_counts.get(row["sensitivity_group"], 0) + 1
    lines = [
        "# P1 Sensitivity Light README",
        "",
        f"Generated: `{time.strftime('%Y-%m-%d %H:%M:%S %z')}`",
        "",
        "## Scope",
        "",
        "- Suite name: `P1_sensitivity_light`.",
        "- Manifest suite label: `sensitivity`.",
        "- Purpose: hyperparameter sensitivity, appendix-oriented supporting evidence.",
        "- This is not mechanism ablation. P1-ablation remains mechanism ablation.",
        "- P2/CSDI is not included.",
        "",
        "## Fixed Design",
        "",
        f"- Dataset: `{DATASET}`",
        f"- Setting: `{VARIANT}`",
        f"- Backbone: `{BACKBONE}`",
        "- Failure mode: `none`",
        f"- Seeds: `{', '.join(str(s) for s in SEEDS)}`",
        "",
        "## Default Parameters",
        "",
        *[f"- `{k}` = `{v}`" for k, v in DEFAULTS.items()],
        "",
        "## One-factor-at-a-time Grids",
        "",
        "- CoRA-Core `beta_hardness`: `0.0, 0.3, 0.5, 0.7, 0.9`",
        "- CoRA-Core `rho`: `0.5, 0.6, 0.7, 0.8, 1.0`",
        "- CoRA-Core `T_part`: `0, 5, 10, 15, 20`",
        "- CoRA-Core `T_refresh`: `0, 2, 5, 10`",
        "- CoRA-StepAlloc `ena_alpha`: `0.5, 1.0, 1.5, 2.0`",
        "- CoRA-StepAlloc `step_bounds`: `(10,30), (12,28), (15,25), (18,22), (20,20)`",
        "",
        "## Notes",
        "",
        "- `rho` changes the partial-round quota and therefore the operating budget/schedule. Treat this as operating-setting sensitivity, not same-budget performance comparison.",
        "- `T_refresh=0` uses the existing runner convention for no periodic refresh.",
        "- `(20,20)` is the uniform-step reference inside StepAlloc.",
        "- Duplicate default-valued configurations are intentionally not de-duplicated because each belongs to a different one-factor sweep. This preserves the requested `28 configurations x 10 seeds = 280 jobs` design.",
        "",
        "## Counts",
        "",
        f"- Expected configurations: `28`",
        f"- Expected jobs: `{EXPECTED_ROWS}`",
        f"- Generated jobs: `{len(rows)}`",
        "",
        "| Sensitivity group | Jobs |",
        "|---|---:|",
    ]
    for group, count in sorted(group_counts.items()):
        lines.append(f"| `{group}` | {count} |")
    lines += [
        "",
        "## Metadata Columns",
        "",
        "- `sensitivity_group`",
        "- `sensitivity_param`",
        "- `sensitivity_value`",
        "- `sensitivity_method`",
        "- `default_param_value`",
        "- `is_default_value`",
        "",
    ]
    README.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if not SOURCE_MANIFEST.exists():
        raise FileNotFoundError(SOURCE_MANIFEST)
    header, source_rows = read_manifest(SOURCE_MANIFEST)
    extra = [
        "sensitivity_group",
        "sensitivity_param",
        "sensitivity_value",
        "sensitivity_method",
        "default_param_value",
        "is_default_value",
    ]
    out_header = list(header) + [c for c in extra if c not in header]
    templates = {
        "CoRA-Core": find_template(source_rows, "CoRA-Core"),
        "CoRA-StepAlloc": find_template(source_rows, "CoRA-StepAlloc"),
    }
    configs = sensitivity_configs()
    if len(configs) != 28:
        raise SystemExit(f"Generated config count {len(configs)} != 28")
    rows = build_rows(out_header, templates)
    if len(rows) != EXPECTED_ROWS:
        raise SystemExit(f"Generated row count {len(rows)} != {EXPECTED_ROWS}")
    write_manifest(OUT_MANIFEST, out_header, rows)
    write_readme(rows, configs)
    print(f"[OK] wrote {OUT_MANIFEST}")
    print(f"[OK] rows={len(rows)} configs={len(configs)} seeds={len(SEEDS)}")
    print(f"[OK] readme={README}")


if __name__ == "__main__":
    main()
