from __future__ import annotations

import csv
import os
import re
import sys
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(ROOT, "results", "fgcs_revision")
PRIORITY_DIR = os.path.join(RESULTS_DIR, "priority")

COL_ALIASES = {
    "backbone": ["backbone", "model_backbone"],
    "seed": ["seed", "random_seed"],
    "dataset": ["dataset", "data", "dataset_name"],
    "variant": ["variant", "setting", "missingness", "scenario"],
    "method": ["method_label", "method", "method_name", "method_canonical"],
    "failure_mode": ["failure_mode", "nonreturn_mode"],
    "failure_rate": ["failure_rate", "nonreturn_rate"],
}

METHOD_ALIASES = {
    "random": ["random"],
    "cora-core": ["cora-core", "cora core", "core", "coracore"],
    "cora-stepalloc": ["cora-stepalloc", "cora stepalloc", "stepalloc", "cora step alloc", "corastepalloc"],
}


def norm(x: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(x).strip().lower())


def detect_columns(header: Sequence[str]) -> Dict[str, str]:
    lower = {h.strip().lower(): h for h in header}
    out: Dict[str, str] = {}
    for logical, aliases in COL_ALIASES.items():
        for alias in aliases:
            if alias.lower() in lower:
                out[logical] = lower[alias.lower()]
                break
    missing = [c for c in ["backbone", "seed", "dataset", "variant", "method"] if c not in out]
    if missing:
        raise RuntimeError(f"Cannot detect required columns {missing}; header={list(header)}")
    return out


def read_manifest(path: str) -> Tuple[List[str], Dict[str, str], List[Dict[str, str]]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise RuntimeError(f"Manifest has no header: {path}")
        header = list(reader.fieldnames)
        return header, detect_columns(header), list(reader)


def values(rows: Sequence[Dict[str, str]], col: str) -> List[str]:
    return sorted({str(r.get(col, "")).strip() for r in rows if str(r.get(col, "")).strip()})


def infer_value(rows: Sequence[Dict[str, str]], col: str, desired: str, aliases: Optional[Dict[str, List[str]]] = None) -> str:
    desired_norms = {norm(desired)}
    if aliases:
        desired_norms.update(norm(x) for x in aliases.get(desired.lower(), []))
    vals = values(rows, col)
    for val in vals:
        if norm(val) in desired_norms:
            return val
    for val in vals:
        vn = norm(val)
        if any(dn and (dn in vn or vn in dn) for dn in desired_norms):
            return val
    raise RuntimeError(f"Cannot infer value for {desired!r} in column {col}; available={vals}")


def infer_many(rows: Sequence[Dict[str, str]], col: str, desired: Sequence[str], aliases: Optional[Dict[str, List[str]]] = None) -> Dict[str, str]:
    return {x: infer_value(rows, col, x, aliases) for x in desired}


def row_seed(row: Dict[str, str], col: str) -> Optional[int]:
    try:
        return int(float(str(row.get(col, "")).strip()))
    except Exception:
        return None


def rate_is_none(row: Dict[str, str], cols: Dict[str, str]) -> bool:
    mode_col = cols.get("failure_mode")
    rate_col = cols.get("failure_rate")
    mode = norm(row.get(mode_col, "none")) if mode_col else "none"
    try:
        rate = float(row.get(rate_col, 0.0) or 0.0) if rate_col else 0.0
    except Exception:
        rate = 0.0
    return mode in {"", "none"} and abs(rate) < 1e-12


def filter_manifest(
    rows: Sequence[Dict[str, str]],
    cols: Dict[str, str],
    *,
    backbones: Sequence[str],
    datasets: Sequence[str],
    variants: Sequence[str],
    methods: Sequence[str],
    seeds: Sequence[int],
) -> List[Dict[str, str]]:
    bset = {norm(x) for x in backbones}
    dset = {norm(x) for x in datasets}
    vset = {norm(x) for x in variants}
    mset = {norm(x) for x in methods}
    sset = set(int(s) for s in seeds)
    out: List[Dict[str, str]] = []
    for row in rows:
        if row_seed(row, cols["seed"]) not in sset:
            continue
        if norm(row.get(cols["backbone"], "")) not in bset:
            continue
        if norm(row.get(cols["dataset"], "")) not in dset:
            continue
        if norm(row.get(cols["variant"], "")) not in vset:
            continue
        if norm(row.get(cols["method"], "")) not in mset:
            continue
        if not rate_is_none(row, cols):
            continue
        out.append(dict(row))
    return out


def write_csv(path: str, header: Sequence[str], rows: Sequence[Dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(header))
        writer.writeheader()
        writer.writerows(rows)


def summarize_counts(rows: Sequence[Dict[str, str]], cols: Dict[str, str]) -> List[str]:
    counter = Counter((r[cols["dataset"]], r[cols["variant"]], r[cols["method"]]) for r in rows)
    return [f"- `{d}` / `{v}` / `{m}`: {n}" for (d, v, m), n in sorted(counter.items())]


def main() -> int:
    os.makedirs(PRIORITY_DIR, exist_ok=True)
    seeds = list(range(10, 20))
    methods_wanted = ["Random", "CoRA-Core", "CoRA-StepAlloc"]
    datasets_main_wanted = ["ETTm1", "Beijing_AQI", "PhysioNet2012"]
    variants_main_wanted = ["hetero", "mcar_p10"]
    stress_variants_wanted = ["hetero_h1", "hetero_h2", "hetero_h3", "hetero_h4"] + [f"mcar_p{x}" for x in [10, 20, 30, 40, 50, 60, 70, 80]]

    specs = [
        {
            "name": "P0A_plus_main_central_paired_seeds10_19.csv",
            "source": os.path.join(RESULTS_DIR, "manifest_main.csv"),
            "expected": 180,
            "datasets": datasets_main_wanted,
            "variants": variants_main_wanted,
        },
        {
            "name": "P0B_plus_stress_core_seeds10_19.csv",
            "source": os.path.join(RESULTS_DIR, "manifest_stress.csv"),
            "expected": 360,
            "datasets": ["ETTm1"],
            "variants": stress_variants_wanted,
        },
    ]

    readme: List[str] = [
        "# P0-A+ / P0-B+ Manifest README",
        "",
        "Purpose: extend completed P0-A/P0-B GRU central comparisons from paired seeds 0-9 to paired seeds 0-19 by adding seeds 10-19 only.",
        "",
        "No algorithmic implementation, dataset, mask, model protocol, or summarizer changes are made by this script.",
        "",
        f"Seeds: `{', '.join(str(s) for s in seeds)}`",
        "",
    ]
    failures: List[str] = []
    for spec in specs:
        header, cols, rows = read_manifest(spec["source"])
        exact_backbones = infer_many(rows, cols["backbone"], ["gru"])
        exact_datasets = infer_many(rows, cols["dataset"], spec["datasets"])
        exact_variants = infer_many(rows, cols["variant"], spec["variants"])
        exact_methods = infer_many(rows, cols["method"], methods_wanted, METHOD_ALIASES)
        selected = filter_manifest(
            rows,
            cols,
            backbones=list(exact_backbones.values()),
            datasets=list(exact_datasets.values()),
            variants=list(exact_variants.values()),
            methods=list(exact_methods.values()),
            seeds=seeds,
        )
        out_path = os.path.join(PRIORITY_DIR, spec["name"])
        write_csv(out_path, header, selected)
        if len(selected) != int(spec["expected"]):
            failures.append(f"{spec['name']}: expected {spec['expected']} rows, got {len(selected)}")
        readme.extend(
            [
                f"## `{spec['name']}`",
                "",
                f"- Source: `{spec['source']}`",
                f"- Output rows: {len(selected)}",
                f"- Expected rows: {spec['expected']}",
                f"- Header: `{', '.join(header)}`",
                f"- Detected columns: {', '.join(f'{k}={v}' for k, v in sorted(cols.items()))}",
                f"- Backbone filter: {exact_backbones}",
                f"- Dataset filters: {exact_datasets}",
                f"- Variant filters: {exact_variants}",
                f"- Method filters: {exact_methods}",
                "- Failure filter: `failure_mode=none`, `failure_rate=0.0`",
                "",
                "Scenario-method row counts:",
                *summarize_counts(selected, cols),
                "",
            ]
        )
    readme_path = os.path.join(PRIORITY_DIR, "P0AB_PLUS_README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("\n".join(readme))
    if failures:
        for failure in failures:
            print(f"[ERROR] {failure}", file=sys.stderr)
        print(f"[INFO] wrote {readme_path}", file=sys.stderr)
        return 2
    print(f"[OK] wrote P0-A+/P0-B+ manifests under {PRIORITY_DIR}")
    print(f"[OK] wrote {readme_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
