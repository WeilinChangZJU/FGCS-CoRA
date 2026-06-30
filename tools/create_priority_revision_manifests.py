from __future__ import annotations

import argparse
import csv
import os
import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(ROOT, "results", "fgcs_revision")
PRIORITY_DIR = os.path.join(RESULTS_DIR, "priority")


COL_ALIASES = {
    "suite": ["suite", "experiment_suite"],
    "backbone": ["backbone", "model_backbone"],
    "seed": ["seed", "random_seed"],
    "dataset": ["dataset", "data", "dataset_name"],
    "variant": ["variant", "setting", "missingness", "scenario"],
    "method": ["method_label", "method", "method_name", "method_canonical"],
    "failure_mode": ["failure_mode", "nonreturn_mode"],
    "failure_rate": ["failure_rate", "nonreturn_rate"],
}


METHOD_ALIASES = {
    "fedavg": ["fedavg", "fed avg"],
    "fedprox": ["fedprox", "fed prox"],
    "qfedavg": ["qfedavg", "q fed avg"],
    "localonly": ["localonly", "local only"],
    "random": ["random"],
    "cora-core": ["cora-core", "cora core", "core", "coracore"],
    "cora-stepalloc": ["cora-stepalloc", "cora stepalloc", "stepalloc", "cora step alloc", "corastepalloc"],
    "top-m": ["top-m", "topm", "top m"],
}


def norm(x: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(x).strip().lower())


def parse_seed_spec(spec: str) -> List[int]:
    out: List[int] = []
    for part in str(spec).replace(",", " ").split():
        part = part.strip()
        if not part:
            continue
        if ":" in part or "-" in part:
            sep = ":" if ":" in part else "-"
            a, b = part.split(sep, 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    seen = set()
    seeds = []
    for s in out:
        if s not in seen:
            seen.add(s)
            seeds.append(s)
    return seeds


def detect_columns(header: Sequence[str]) -> Dict[str, str]:
    lower = {h.strip().lower(): h for h in header}
    detected: Dict[str, str] = {}
    for logical, aliases in COL_ALIASES.items():
        for alias in aliases:
            if alias.lower() in lower:
                detected[logical] = lower[alias.lower()]
                break
    required = ["backbone", "seed", "dataset", "variant", "method"]
    missing = [c for c in required if c not in detected]
    if missing:
        raise RuntimeError(f"Cannot detect required manifest columns {missing}; header={list(header)}")
    return detected


def read_manifest(path: str) -> Tuple[List[str], Dict[str, str], List[Dict[str, str]]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise RuntimeError(f"Manifest has no header: {path}")
        header = list(reader.fieldnames)
        cols = detect_columns(header)
        rows = list(reader)
    return header, cols, rows


def exact_values(rows: Sequence[Dict[str, str]], col: str) -> List[str]:
    return sorted({str(r.get(col, "")).strip() for r in rows if str(r.get(col, "")).strip()})


def infer_exact(
    rows: Sequence[Dict[str, str]],
    col: str,
    desired: str,
    *,
    aliases: Optional[Dict[str, List[str]]] = None,
) -> Optional[str]:
    values = exact_values(rows, col)
    desired_norms = {norm(desired)}
    if aliases:
        desired_norms.update(norm(x) for x in aliases.get(desired.lower(), []))
    for value in values:
        if norm(value) in desired_norms:
            return value
    for value in values:
        vn = norm(value)
        if any(dn and (dn in vn or vn in dn) for dn in desired_norms):
            return value
    return None


def infer_many(
    rows: Sequence[Dict[str, str]],
    col: str,
    desired_values: Sequence[str],
    *,
    aliases: Optional[Dict[str, List[str]]] = None,
) -> Tuple[List[str], Dict[str, str], List[str]]:
    mapped: List[str] = []
    mapping: Dict[str, str] = {}
    missing: List[str] = []
    for desired in desired_values:
        exact = infer_exact(rows, col, desired, aliases=aliases)
        if exact is None:
            missing.append(desired)
        else:
            mapped.append(exact)
            mapping[desired] = exact
    return mapped, mapping, missing


def row_seed(row: Dict[str, str], col: str) -> Optional[int]:
    try:
        return int(float(str(row.get(col, "")).strip()))
    except Exception:
        return None


def row_rate(row: Dict[str, str], col: Optional[str]) -> Optional[float]:
    if not col:
        return None
    try:
        return round(float(str(row.get(col, "")).strip()), 6)
    except Exception:
        return None


def filter_rows(
    rows: Sequence[Dict[str, str]],
    cols: Dict[str, str],
    *,
    backbones: Sequence[str],
    datasets: Sequence[str],
    variants: Sequence[str],
    methods: Sequence[str],
    seeds: Sequence[int],
    failure_modes: Optional[Sequence[str]] = None,
    failure_rates: Optional[Sequence[float]] = None,
) -> List[Dict[str, str]]:
    seed_set = set(int(s) for s in seeds)
    backbone_set = {norm(x) for x in backbones}
    dataset_set = {norm(x) for x in datasets}
    variant_set = {norm(x) for x in variants}
    method_set = {norm(x) for x in methods}
    failure_mode_set = None if failure_modes is None else {norm(x) for x in failure_modes}
    failure_rate_set = None if failure_rates is None else {round(float(x), 6) for x in failure_rates}
    fmode_col = cols.get("failure_mode")
    frate_col = cols.get("failure_rate")
    out: List[Dict[str, str]] = []
    for row in rows:
        seed = row_seed(row, cols["seed"])
        if seed not in seed_set:
            continue
        if norm(row.get(cols["backbone"], "")) not in backbone_set:
            continue
        if norm(row.get(cols["dataset"], "")) not in dataset_set:
            continue
        if norm(row.get(cols["variant"], "")) not in variant_set:
            continue
        if norm(row.get(cols["method"], "")) not in method_set:
            continue
        if failure_mode_set is not None:
            if fmode_col is None or norm(row.get(fmode_col, "")) not in failure_mode_set:
                continue
        if failure_rate_set is not None:
            rate = row_rate(row, frate_col)
            if rate not in failure_rate_set:
                continue
        out.append(dict(row))
    return out


def write_manifest(path: str, header: Sequence[str], rows: Sequence[Dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(header))
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows: Sequence[Dict[str, str]], cols: Dict[str, str]) -> str:
    keys = [cols["dataset"], cols["variant"], cols["method"], cols["seed"]]
    if "failure_mode" in cols:
        keys.append(cols["failure_mode"])
    if "failure_rate" in cols:
        keys.append(cols["failure_rate"])
    summary = []
    for key in keys:
        vals = exact_values(rows, key)
        label = key
        if len(vals) <= 20:
            summary.append(f"- `{label}`: {', '.join(vals)}")
        else:
            summary.append(f"- `{label}`: {len(vals)} values")
    return "\n".join(summary)


def estimate_seconds(name: str, count: int, sec: Dict[str, float]) -> float:
    if name.startswith("P0D") or name.startswith("P1_failure"):
        return count * sec["gru_failure"]
    if "saits" in name.lower():
        return count * sec["saits"]
    if "csdi" in name.lower():
        return count * sec["csdi"]
    return count * sec["gru"]


def fmt_hours(seconds: float) -> str:
    return f"{seconds / 3600.0:.2f} h"


def main() -> None:
    ap = argparse.ArgumentParser("Create priority-compressed CoRA FGCS revision manifests")
    ap.add_argument("--results_dir", type=str, default=RESULTS_DIR)
    ap.add_argument("--priority_dir", type=str, default=PRIORITY_DIR)
    ap.add_argument("--main_seeds", type=str, default="0:9")
    ap.add_argument("--p0_seeds", type=str, default="0:9")
    ap.add_argument("--csdi_seeds", type=str, default="0:4")
    ap.add_argument("--gru_seconds", type=float, default=122.3)
    ap.add_argument("--gru_failure_seconds", type=float, default=91.0)
    ap.add_argument("--saits_seconds", type=float, default=297.3)
    ap.add_argument("--csdi_seconds", type=float, default=600.0)
    args = ap.parse_args()

    os.makedirs(args.priority_dir, exist_ok=True)
    paths = {
        "main": os.path.join(args.results_dir, "manifest_main.csv"),
        "stress": os.path.join(args.results_dir, "manifest_stress.csv"),
        "failure": os.path.join(args.results_dir, "manifest_failure.csv"),
        "backbone": os.path.join(args.results_dir, "manifest_backbone.csv"),
    }
    loaded: Dict[str, Tuple[List[str], Dict[str, str], List[Dict[str, str]]]] = {}
    for key, path in paths.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        loaded[key] = read_manifest(path)

    main_seeds = parse_seed_spec(args.main_seeds)
    p0_seeds = parse_seed_spec(args.p0_seeds)
    csdi_seeds = parse_seed_spec(args.csdi_seeds)
    datasets_main = ["ETTm1", "Beijing_AQI", "PhysioNet2012"]
    variants_main = ["hetero", "mcar_p10"]
    stress_variants = ["hetero_h1", "hetero_h2", "hetero_h3", "hetero_h4"] + [f"mcar_p{x}" for x in [10, 20, 30, 40, 50, 60, 70, 80]]
    central_methods = ["Random", "CoRA-Core", "CoRA-StepAlloc"]
    ref_methods = ["FedAvg", "FedProx", "qFedAvg", "LocalOnly"]
    backbone_methods = ["FedAvg", "Random", "CoRA-Core", "CoRA-StepAlloc"]
    failure_modes = ["uniform", "score_correlated"]
    failure_rates = [0.05, 0.10, 0.20]

    manifest_specs = [
        ("P0A_main_central_paired.csv", "main", ["gru"], datasets_main, variants_main, central_methods, main_seeds, None, None),
        ("P0B_stress_core_10seeds.csv", "stress", ["gru"], ["ETTm1"], stress_variants, central_methods, p0_seeds, None, None),
        ("P0C_saits_h4_10seeds.csv", "backbone", ["saits"], ["ETTm1"], ["hetero_h4"], backbone_methods, p0_seeds, None, None),
        ("P0D_failure_h4_10seeds.csv", "failure", ["gru"], ["ETTm1"], ["hetero_h4"], central_methods, p0_seeds, failure_modes, failure_rates),
        ("P1_main_full_refs_10seeds.csv", "main", ["gru"], datasets_main, variants_main, ref_methods, p0_seeds, None, None),
        ("P1_stress_fedavg_10seeds.csv", "stress", ["gru"], ["ETTm1"], stress_variants, ["FedAvg"], p0_seeds, None, None),
        ("P1_failure_mcar10_10seeds.csv", "failure", ["gru"], ["ETTm1"], ["mcar_p10"], central_methods, p0_seeds, failure_modes, failure_rates),
        ("P1_saits_mcar10_10seeds.csv", "backbone", ["saits"], ["ETTm1"], ["mcar_p10"], backbone_methods, p0_seeds, None, None),
        ("P2_csdi_h4_5seeds.csv", "backbone", ["csdi"], ["ETTm1"], ["hetero_h4"], backbone_methods, csdi_seeds, None, None),
    ]

    all_notes: List[str] = []
    plan_rows: List[Tuple[str, int, float]] = []
    readme_lines = [
        "# Priority Manifest README",
        "",
        "Generated from existing manifest CSV files. Filters were applied to exact values inferred from each manifest rather than hard-coded assumptions.",
        "",
    ]

    for source_name, (header, cols, rows) in loaded.items():
        readme_lines += [
            f"## Source `{source_name}`",
            "",
            f"- Path: `{paths[source_name]}`",
            f"- Rows: {len(rows)}",
            f"- Header: `{', '.join(header)}`",
            f"- Detected columns: {', '.join(f'{k}={v}' for k, v in sorted(cols.items()))}",
            "",
            "Available values:",
            summarize_rows(rows, cols),
            "",
        ]

    for filename, source, desired_backbones, desired_datasets, desired_variants, desired_methods, seeds, desired_fmodes, desired_frates in manifest_specs:
        header, cols, rows = loaded[source]
        backbones, bmap, bmiss = infer_many(rows, cols["backbone"], desired_backbones)
        datasets, dmap, dmiss = infer_many(rows, cols["dataset"], desired_datasets)
        variants, vmap, vmiss = infer_many(rows, cols["variant"], desired_variants)
        methods, mmap, mmiss = infer_many(rows, cols["method"], desired_methods, aliases=METHOD_ALIASES)
        fmodes: Optional[List[str]] = None
        fmode_map: Dict[str, str] = {}
        fmiss: List[str] = []
        if desired_fmodes is not None:
            if "failure_mode" not in cols:
                fmiss = list(desired_fmodes)
            else:
                fmodes, fmode_map, fmiss = infer_many(rows, cols["failure_mode"], desired_fmodes)
        missing = bmiss + dmiss + vmiss + mmiss + fmiss
        if missing:
            all_notes.append(f"{filename}: missing desired values {missing}")
        selected = filter_rows(
            rows,
            cols,
            backbones=backbones,
            datasets=datasets,
            variants=variants,
            methods=methods,
            seeds=seeds,
            failure_modes=fmodes,
            failure_rates=desired_frates,
        )
        out_path = os.path.join(args.priority_dir, filename)
        write_manifest(out_path, header, selected)
        seconds = estimate_seconds(filename, len(selected), {
            "gru": args.gru_seconds,
            "gru_failure": args.gru_failure_seconds,
            "saits": args.saits_seconds,
            "csdi": args.csdi_seconds,
        })
        plan_rows.append((filename, len(selected), seconds))
        readme_lines += [
            f"## `{filename}`",
            "",
            f"- Source: `{source}`",
            f"- Rows: {len(selected)}",
            f"- Seeds requested: `{', '.join(str(s) for s in seeds)}`",
            f"- Backbones: {bmap}",
            f"- Datasets: {dmap}",
            f"- Variants/settings: {vmap}",
            f"- Methods: {mmap}",
        ]
        if desired_fmodes is not None:
            readme_lines += [
                f"- Failure modes: {fmode_map}",
                f"- Failure rates: {', '.join(str(x) for x in desired_frates or [])}",
            ]
        counts = Counter((r.get(cols["dataset"], ""), r.get(cols["variant"], ""), r.get(cols["method"], "")) for r in selected)
        readme_lines += [
            "- Scenario-method row counts:",
            *[f"  - `{k[0]}` / `{k[1]}` / `{k[2]}`: {v}" for k, v in sorted(counts.items())],
            "",
        ]

    if args.main_seeds == "0:9":
        all_notes.append("P0-A uses 10 paired seeds (0:9) rather than 20 because the previous full-run timing showed the 20-seed package is prohibitive on the current single GPU.")

    readme_lines += [
        "## Notes",
        "",
        *(f"- {note}" for note in all_notes),
        "",
    ]
    with open(os.path.join(args.priority_dir, "PRIORITY_MANIFEST_README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(readme_lines))

    order = [
        "P0A_main_central_paired.csv",
        "P0B_stress_core_10seeds.csv",
        "P0D_failure_h4_10seeds.csv",
        "P0C_saits_h4_10seeds.csv",
    ]
    p1p2 = [
        "P1_main_full_refs_10seeds.csv",
        "P1_stress_fedavg_10seeds.csv",
        "P1_failure_mcar10_10seeds.csv",
        "P1_saits_mcar10_10seeds.csv",
        "P2_csdi_h4_5seeds.csv",
    ]
    plan_map = {name: (count, seconds) for name, count, seconds in plan_rows}
    plan_lines = [
        "# Priority Revision Execution Plan",
        "",
        "P0 is the minimum reviewer-compliant package for result-grounded Section 5 drafting. P1/P2 are completeness and robustness additions and should not be run until all P0 manifests finish and coverage passes.",
        "",
        "## Manifest Counts and Runtime Estimates",
        "",
        "| Manifest | Jobs | Estimate |",
        "|---|---:|---:|",
    ]
    for name, count, seconds in plan_rows:
        plan_lines.append(f"| `{name}` | {count} | {fmt_hours(seconds)} |")
    plan_lines += [
        "",
        "Runtime estimates use observed smoke-job timings: GRU non-failure 122.3 s/job, GRU failure-return 91.0 s/job, SAITS 297.3 s/job, and placeholder CSDI 600.0 s/job.",
        "",
        "## Exact Filters",
        "",
        "- P0-A: `manifest_main.csv`; GRU; ETTm1, Beijing_AQI, PhysioNet2012; hetero and mcar_p10; Random, CoRA-Core, CoRA-StepAlloc; seeds 0:9.",
        "- P0-B: `manifest_stress.csv`; GRU; ETTm1; H1-H4 and MCAR 10%-80%; Random, CoRA-Core, CoRA-StepAlloc; seeds 0:9.",
        "- P0-C: `manifest_backbone.csv`; SAITS; ETTm1 hetero_h4; FedAvg, Random, CoRA-Core, CoRA-StepAlloc; seeds 0:9.",
        "- P0-D: `manifest_failure.csv`; GRU; ETTm1 hetero_h4; uniform and score_correlated; 0.05, 0.10, 0.20; Random, CoRA-Core, CoRA-StepAlloc; seeds 0:9.",
        "- P1/P2: generated for optional completeness only; not part of the minimum P0 drafting gate.",
        "",
        "## Execution Order",
        "",
    ]
    for i, name in enumerate(order, start=1):
        count, seconds = plan_map.get(name, (0, 0.0))
        plan_lines.append(f"{i}. `{name}` ({count} jobs, estimated {fmt_hours(seconds)})")
    plan_lines += [
        "",
        "## Optional Manifests",
        "",
        *(f"- `{name}` ({plan_map.get(name, (0, 0.0))[0]} jobs, estimated {fmt_hours(plan_map.get(name, (0, 0.0))[1])})" for name in p1p2),
        "",
        "## Drafting Gate",
        "",
        "Final Section 5 must not be drafted until P0 coverage reports at least 10 complete paired seeds for all central Random/Core/StepAlloc scenarios and regenerated final summaries, tables, and figures are available.",
        "",
    ]
    with open(os.path.join(args.priority_dir, "PRIORITY_PLAN.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(plan_lines))

    print(f"[OK] wrote priority manifests under {args.priority_dir}")
    for name, count, seconds in plan_rows:
        print(f"{name}: jobs={count} estimate={fmt_hours(seconds)}")


if __name__ == "__main__":
    main()
