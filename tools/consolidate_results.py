from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import numpy as np

ALLOWED_DATASETS = {"ETTm1", "Beijing_AQI", "PhysioNet2012"}


def _as_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if not np.isfinite(v):
            return None
        return v
    except Exception:
        return None


def _safe_mean_std(vals: List[float]) -> Tuple[Optional[float], Optional[float], int]:
    clean = [float(v) for v in vals if v is not None and np.isfinite(v)]
    if not clean:
        return None, None, 0
    arr = np.asarray(clean, dtype=np.float64)
    return float(arr.mean()), float(arr.std()), int(arr.size)


def _parse_variant_from_path(root: str) -> str:
    parts = os.path.normpath(root).split(os.sep)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].startswith("seed_") and i + 1 < len(parts):
            return parts[i + 1]
    return "unknown"


def _parse_seed_from_path(root: str) -> Optional[int]:
    parts = os.path.normpath(root).split(os.sep)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].startswith("seed_"):
            try:
                return int(parts[i].split("seed_")[-1])
            except Exception:
                return None
    return None


def _parse_method_dataset_variant_from_filename(fname: str) -> Tuple[str, str, str]:
    core = fname.replace("metrics_", "").replace(".json", "")
    for d in sorted(ALLOWED_DATASETS, key=len, reverse=True):
        needle = f"_{d}_"
        if needle in core:
            left, right = core.split(needle, 1)
            return left, d, right
    return core, "Unknown", "unknown"


def _extract_metadata(fp: str, data: Dict[str, Any]) -> Dict[str, Any]:
    root = os.path.dirname(fp)
    fname = os.path.basename(fp)
    tag_fallback, ds_fallback, var_fallback = _parse_method_dataset_variant_from_filename(fname)
    run_meta = data.get("run_meta", {}) if isinstance(data.get("run_meta"), dict) else {}
    seed = run_meta.get("seed", _parse_seed_from_path(root))
    dataset = str(run_meta.get("dataset", ds_fallback))
    variant = str(run_meta.get("variant", _parse_variant_from_path(root) if var_fallback == "unknown" else var_fallback))
    method = str(run_meta.get("method", tag_fallback))
    tag = str(run_meta.get("tag", method))
    return {
        "file": fp,
        "seed": seed,
        "dataset": dataset,
        "variant": variant,
        "method": method,
        "tag": tag,
    }


def _extract_global_summary(data: Dict[str, Any]) -> Dict[str, Optional[float]]:
    # federated methods
    audit = data.get("audit_global_test")
    if isinstance(audit, dict):
        return {
            "rmse_avg": _as_float(audit.get("rmse_avg", audit.get("rmse_macro"))),
            "mae_avg": _as_float(audit.get("mae_avg", audit.get("mae_macro"))),
            "rmse_worst": _as_float(audit.get("rmse_worst")),
            "mae_worst": _as_float(audit.get("mae_worst")),
            "rmse_std": _as_float(audit.get("rmse_std")),
            "mae_std": _as_float(audit.get("mae_std")),
            "rmse_micro": _as_float(audit.get("rmse_micro")),
            "mae_micro": _as_float(audit.get("mae_micro")),
            "n_clients": _as_float(audit.get("n_clients")),
        }
    # localonly
    ts = data.get("test_summary")
    if isinstance(ts, dict):
        return {
            "rmse_avg": _as_float(ts.get("rmse_avg", ts.get("rmse_macro"))),
            "mae_avg": _as_float(ts.get("mae_avg", ts.get("mae_macro"))),
            "rmse_worst": _as_float(ts.get("rmse_worst")),
            "mae_worst": _as_float(ts.get("mae_worst")),
            "rmse_std": _as_float(ts.get("rmse_std")),
            "mae_std": _as_float(ts.get("mae_std")),
            "rmse_micro": _as_float(ts.get("rmse_micro")),
            "mae_micro": _as_float(ts.get("mae_micro")),
            "n_clients": _as_float(ts.get("n_clients")),
        }
    # fallback to last curve entry
    curve = data.get("global_curve")
    if isinstance(curve, list) and len(curve) > 0 and isinstance(curve[-1], dict):
        e = curve[-1]
        return {
            "rmse_avg": _as_float(e.get("rmse_avg", e.get("rmse_macro"))),
            "mae_avg": _as_float(e.get("mae_avg", e.get("mae_macro"))),
            "rmse_worst": _as_float(e.get("rmse_worst")),
            "mae_worst": _as_float(e.get("mae_worst")),
            "rmse_std": _as_float(e.get("rmse_std")),
            "mae_std": _as_float(e.get("mae_std")),
            "rmse_micro": _as_float(e.get("rmse_micro")),
            "mae_micro": _as_float(e.get("mae_micro")),
            "n_clients": None,
        }
    return {k: None for k in ["rmse_avg","mae_avg","rmse_worst","mae_worst","rmse_std","mae_std","rmse_micro","mae_micro","n_clients"]}


def _extract_comm_runtime(data: Dict[str, Any]) -> Dict[str, Optional[float]]:
    comm = data.get("comm_stats") if isinstance(data.get("comm_stats"), dict) else {}
    return {
        "uplink_mb": _as_float(comm.get("uplink_mb", comm.get("comm_total_up_mb"))),
        "runtime_sec_total": _as_float(data.get("runtime_sec_total")),
    }


def _extract_curve_end(data: Dict[str, Any]) -> Dict[str, Optional[float]]:
    curve = data.get("global_curve")
    if not isinstance(curve, list) or len(curve) == 0 or not isinstance(curve[-1], dict):
        return {
            "curve_last_round": None,
            "curve_last_rmse_avg": None,
            "curve_last_mae_avg": None,
            "curve_last_rmse_worst": None,
            "curve_last_mae_worst": None,
            "curve_points": 0,
        }
    e = curve[-1]
    return {
        "curve_last_round": _as_float(e.get("round")),
        "curve_last_rmse_avg": _as_float(e.get("rmse_avg", e.get("rmse_macro"))),
        "curve_last_mae_avg": _as_float(e.get("mae_avg", e.get("mae_macro"))),
        "curve_last_rmse_worst": _as_float(e.get("rmse_worst")),
        "curve_last_mae_worst": _as_float(e.get("mae_worst")),
        "curve_points": len(curve),
    }


def _dominates(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
    ax, ay = a
    bx, by = b
    return (ax <= bx and ay <= by) and (ax < bx or ay < by)


def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys: List[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    parser = argparse.ArgumentParser("Consolidate CoRA metrics_*.json files")
    parser.add_argument("--results_dir", type=str, default="results/final_paper")
    parser.add_argument("--out_dir", type=str, default="results/final_paper")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    per_run_rows: List[Dict[str, Any]] = []
    aggregated: DefaultDict[str, DefaultDict[str, DefaultDict[str, DefaultDict[str, List[float]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )

    processed = 0
    read_errors = 0
    schema_issues: List[str] = []

    for root, _dirs, files in os.walk(args.results_dir):
        for fname in files:
            if not (fname.startswith("metrics_") and fname.endswith(".json")):
                continue
            fp = os.path.join(root, fname)
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                read_errors += 1
                schema_issues.append(f"[READ_ERROR] {fp} :: {e}")
                continue

            meta = _extract_metadata(fp, data)
            if meta["dataset"] not in ALLOWED_DATASETS:
                continue
            g = _extract_global_summary(data)
            cr = _extract_comm_runtime(data)
            ce = _extract_curve_end(data)

            if g["rmse_avg"] is None:
                schema_issues.append(f"[MISSING_GLOBAL_SUMMARY] {fp}")

            row = {**meta, **g, **cr, **ce}
            per_run_rows.append(row)
            processed += 1

            for metric_name in [
                "rmse_avg", "mae_avg", "rmse_worst", "mae_worst",
                "rmse_std", "mae_std", "rmse_micro", "mae_micro",
                "uplink_mb", "runtime_sec_total",
            ]:
                v = row.get(metric_name)
                if v is not None and np.isfinite(v):
                    aggregated[str(meta["variant"])][str(meta["dataset"])][str(meta["method"])][metric_name].append(float(v))

    summary_rows: List[Dict[str, Any]] = []
    pareto_rows: List[Dict[str, Any]] = []
    for variant, ds_map in aggregated.items():
        for dataset, meth_map in ds_map.items():
            mean_points: Dict[str, Tuple[float, float]] = {}
            for method, metric_map in meth_map.items():
                row: Dict[str, Any] = {"variant": variant, "dataset": dataset, "method": method}
                for metric_name, vals in metric_map.items():
                    mu, sd, n = _safe_mean_std(vals)
                    row[f"{metric_name}_mean"] = mu
                    row[f"{metric_name}_std"] = sd
                    row[f"{metric_name}_n"] = n
                summary_rows.append(row)
                rmse_mu = row.get("rmse_avg_mean")
                up_mu = row.get("uplink_mb_mean")
                if rmse_mu is not None and up_mu is not None:
                    mean_points[method] = (float(rmse_mu), float(up_mu))
            for method, xy in mean_points.items():
                nondom = True
                for other, xyo in mean_points.items():
                    if other == method:
                        continue
                    if _dominates(xyo, xy):
                        nondom = False
                        break
                pareto_rows.append({
                    "variant": variant,
                    "dataset": dataset,
                    "method": method,
                    "rmse_avg_mean": xy[0],
                    "uplink_mb_mean": xy[1],
                    "pareto_nondominated": int(nondom),
                })

    _write_csv(os.path.join(args.out_dir, "FINAL_RESULTS_PER_RUN.csv"), per_run_rows)
    _write_csv(os.path.join(args.out_dir, "FINAL_RESULTS_SUMMARY.csv"), summary_rows)
    _write_csv(os.path.join(args.out_dir, "PARETO_FRONTIERS.csv"), pareto_rows)
    with open(os.path.join(args.out_dir, "CONSOLIDATE_REPORT.txt"), "w", encoding="utf-8") as f:
        f.write(f"processed={processed}\n")
        f.write(f"read_errors={read_errors}\n")
        for line in schema_issues:
            f.write(line + "\n")

    print(f"[OK] processed={processed} read_errors={read_errors}")
    print(f"[OK] wrote: {os.path.join(args.out_dir, 'FINAL_RESULTS_PER_RUN.csv')}")
    print(f"[OK] wrote: {os.path.join(args.out_dir, 'FINAL_RESULTS_SUMMARY.csv')}")
    print(f"[OK] wrote: {os.path.join(args.out_dir, 'PARETO_FRONTIERS.csv')}")
    print(f"[OK] wrote: {os.path.join(args.out_dir, 'CONSOLIDATE_REPORT.txt')}")


if __name__ == "__main__":
    main()
