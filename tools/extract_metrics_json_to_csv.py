from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ALLOWED_DATASETS = {"ETTm1", "Beijing_AQI", "PhysioNet2012"}


def as_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if not np.isfinite(v):
            return None
        return v
    except Exception:
        return None


def safe_json_dumps(x: Any) -> str:
    try:
        return json.dumps(x, ensure_ascii=False)
    except Exception:
        return str(x)


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
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


def parse_variant_from_path(root: str) -> str:
    parts = os.path.normpath(root).split(os.sep)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].startswith("seed_") and i + 1 < len(parts):
            return parts[i + 1]
    return "unknown"


def parse_seed_from_path(root: str) -> Optional[int]:
    parts = os.path.normpath(root).split(os.sep)
    for p in parts:
        if p.startswith("seed_"):
            try:
                return int(p.split("seed_")[-1])
            except Exception:
                return None
    return None


def parse_method_dataset_variant_from_filename(fname: str) -> Tuple[str, str, str]:
    core = fname.replace("metrics_", "").replace(".json", "")
    for d in sorted(ALLOWED_DATASETS, key=len, reverse=True):
        needle = f"_{d}_"
        if needle in core:
            left, right = core.split(needle, 1)
            return left, d, right
    return core, "Unknown", "unknown"


def extract_meta(fp: str, data: Dict[str, Any]) -> Dict[str, Any]:
    root = os.path.dirname(fp)
    fname = os.path.basename(fp)
    tag_fallback, ds_fallback, var_fallback = parse_method_dataset_variant_from_filename(fname)
    run_meta = data.get("run_meta", {}) if isinstance(data.get("run_meta"), dict) else {}
    return {
        "file": fp,
        "seed": run_meta.get("seed", parse_seed_from_path(root)),
        "variant": str(run_meta.get("variant", parse_variant_from_path(root) if var_fallback == "unknown" else var_fallback)),
        "dataset": str(run_meta.get("dataset", ds_fallback)),
        "method": str(run_meta.get("method", tag_fallback)),
        "tag": str(run_meta.get("tag", run_meta.get("method", tag_fallback))),
        "eval_split": str(run_meta.get("eval_split", "")),
        "method_canonical": str(run_meta.get("method_canonical", "")),
        "participation_policy": str(run_meta.get("participation_policy", "")),
        "stepalloc_policy": str(run_meta.get("stepalloc_policy", "")),
        "aggregation_rule_resolved": str(run_meta.get("aggregation_rule_resolved", "")),
        "aggregation_rule_cfg": str(run_meta.get("aggregation_rule", "")),
        "ena_reference_mode_cfg": str(run_meta.get("ena_reference_mode", "")),
        "ena_alpha_cfg": as_float(run_meta.get("ena_alpha")),
        "ena_clip_min_cfg": as_float(run_meta.get("ena_clip_min")),
        "ena_clip_max_cfg": as_float(run_meta.get("ena_clip_max")),
        "stepalloc_min_steps_cfg": as_float(run_meta.get("stepalloc_min_steps")),
        "stepalloc_max_steps_cfg": as_float(run_meta.get("stepalloc_max_steps")),
        "stepalloc_power_cfg": as_float(run_meta.get("stepalloc_power")),
        "local_steps_cfg": as_float(run_meta.get("local_steps")),
        "beta_hardness_cfg": as_float(run_meta.get("beta_hardness")),
        "rho_cfg": as_float(run_meta.get("rho")),
        "T_part_cfg": as_float(run_meta.get("T_part")),
        "T_refresh_cfg": as_float(run_meta.get("T_refresh")),
        "K_min_cfg": as_float(run_meta.get("K_min")),
        "score_floor_cfg": as_float(run_meta.get("score_floor")),
        "failure_mode_cfg": str(run_meta.get("failure_mode", "none")),
        "failure_rate_cfg": as_float(run_meta.get("failure_rate")),
        "failure_qmax_cfg": as_float(run_meta.get("failure_qmax")),
        "checkpoint_selection_cfg": str(run_meta.get("checkpoint_selection", "")),
    }


def extract_global_summary(data: Dict[str, Any]) -> Dict[str, Optional[float]]:
    audit = data.get("audit_global_test")
    if isinstance(audit, dict):
        return {
            "rmse_avg": as_float(audit.get("rmse_avg", audit.get("rmse_macro"))),
            "mae_avg": as_float(audit.get("mae_avg", audit.get("mae_macro"))),
            "rmse_worst": as_float(audit.get("rmse_worst")),
            "mae_worst": as_float(audit.get("mae_worst")),
            "rmse_std": as_float(audit.get("rmse_std")),
            "mae_std": as_float(audit.get("mae_std")),
            "rmse_micro": as_float(audit.get("rmse_micro")),
            "mae_micro": as_float(audit.get("mae_micro")),
            "n_clients": as_float(audit.get("n_clients")),
            "source": "audit_global_test",
        }
    ts = data.get("test_summary")
    if isinstance(ts, dict):
        return {
            "rmse_avg": as_float(ts.get("rmse_avg", ts.get("rmse_macro"))),
            "mae_avg": as_float(ts.get("mae_avg", ts.get("mae_macro"))),
            "rmse_worst": as_float(ts.get("rmse_worst")),
            "mae_worst": as_float(ts.get("mae_worst")),
            "rmse_std": as_float(ts.get("rmse_std")),
            "mae_std": as_float(ts.get("mae_std")),
            "rmse_micro": as_float(ts.get("rmse_micro")),
            "mae_micro": as_float(ts.get("mae_micro")),
            "n_clients": as_float(ts.get("n_clients")),
            "source": "test_summary",
        }
    return {
        "rmse_avg": None, "mae_avg": None, "rmse_worst": None, "mae_worst": None,
        "rmse_std": None, "mae_std": None, "rmse_micro": None, "mae_micro": None,
        "n_clients": None, "source": "missing",
    }


def main() -> None:
    ap = argparse.ArgumentParser("Flatten CoRA metrics_*.json to CSVs")
    ap.add_argument("--results_dir", type=str, default="results/final_paper")
    ap.add_argument("--out_dir", type=str, default="results/final_paper/extracted")
    args = ap.parse_args()

    per_run: List[Dict[str, Any]] = []
    global_curve_rows: List[Dict[str, Any]] = []
    round_stats_rows: List[Dict[str, Any]] = []
    per_client_rows: List[Dict[str, Any]] = []
    participation_rows: List[Dict[str, Any]] = []
    allocation_rows: List[Dict[str, Any]] = []
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
                schema_issues.append(f"[READ_ERROR] {fp} :: {e}")
                continue

            meta = extract_meta(fp, data)
            if meta["dataset"] not in ALLOWED_DATASETS:
                continue

            gs = extract_global_summary(data)
            comm = data.get("comm_stats") if isinstance(data.get("comm_stats"), dict) else {}
            row_run = {
                **meta,
                "runtime_sec_total": as_float(data.get("runtime_sec_total")),
                "uplink_mb": as_float(comm.get("uplink_mb", comm.get("comm_total_up_mb"))),
                "global_test_rmse_avg": gs["rmse_avg"],
                "global_test_mae_avg": gs["mae_avg"],
                "global_test_rmse_worst": gs["rmse_worst"],
                "global_test_mae_worst": gs["mae_worst"],
                "global_test_rmse_std": gs["rmse_std"],
                "global_test_mae_std": gs["mae_std"],
                "global_test_rmse_micro": gs["rmse_micro"],
                "global_test_mae_micro": gs["mae_micro"],
                "global_test_n_clients": gs["n_clients"],
                "global_test_source": gs["source"],
            }
            fail = data.get("failure_stats") if isinstance(data.get("failure_stats"), dict) else {}
            ckpt = data.get("selected_checkpoint") if isinstance(data.get("selected_checkpoint"), dict) else {}
            row_run.update({
                "checkpoint_selection": ckpt.get("selection"),
                "checkpoint_round": ckpt.get("round"),
                "failure_mode": fail.get("failure_mode"),
                "failure_rate": as_float(fail.get("failure_rate")),
                "valid_return_rate": as_float(fail.get("valid_return_rate")),
                "failed_return_rate": as_float(fail.get("failed_return_rate")),
                "wasted_assigned_step_ratio": as_float(fail.get("wasted_assigned_step_ratio")),
                "returned_assigned_step_ratio": as_float(fail.get("returned_assigned_step_ratio")),
                "empty_return_rounds": as_float(fail.get("empty_return_rounds")),
            })
            curve = data.get("global_curve")
            if isinstance(curve, list) and curve and isinstance(curve[-1], dict):
                e = curve[-1]
                row_run["curve_end_round"] = e.get("round")
                row_run["curve_end_rmse_avg"] = as_float(e.get("rmse_avg", e.get("rmse_macro")))
                row_run["curve_end_mae_avg"] = as_float(e.get("mae_avg", e.get("mae_macro")))
                row_run["curve_end_rmse_worst"] = as_float(e.get("rmse_worst"))
                row_run["curve_end_mae_worst"] = as_float(e.get("mae_worst"))
                row_run["curve_points"] = len(curve)
            per_run.append(row_run)

            if isinstance(curve, list):
                for e in curve:
                    if not isinstance(e, dict):
                        continue
                    r = dict(meta)
                    for k, v in e.items():
                        r[k] = safe_json_dumps(v) if isinstance(v, (list, dict)) else v
                    global_curve_rows.append(r)
            else:
                schema_issues.append(f"[NO_GLOBAL_CURVE] {fp}")

            rs_list = data.get("round_stats")
            if isinstance(rs_list, list):
                for e in rs_list:
                    if not isinstance(e, dict):
                        continue
                    r = dict(meta)
                    for k, v in e.items():
                        r[k] = safe_json_dumps(v) if isinstance(v, (list, dict)) else v
                    round_stats_rows.append(r)
            else:
                schema_issues.append(f"[NO_ROUND_STATS] {fp}")

            per_client = data.get("per_client_test")
            if isinstance(per_client, dict):
                for cid, metrics in per_client.items():
                    if not isinstance(metrics, dict):
                        continue
                    per_client_rows.append({
                        **meta,
                        "client_id": int(cid) if str(cid).isdigit() else str(cid),
                        "rmse": as_float(metrics.get("rmse")),
                        "mae": as_float(metrics.get("mae")),
                        "denom": as_float(metrics.get("denom")),
                    })
            else:
                schema_issues.append(f"[NO_PER_CLIENT_TEST] {fp}")

            part = data.get("participation")
            if isinstance(part, dict):
                for cid, info in part.items():
                    if not isinstance(info, dict):
                        continue
                    participation_rows.append({
                        **meta,
                        "client_id": int(cid) if str(cid).isdigit() else str(cid),
                        "selected_rounds": info.get("selected_rounds"),
                        "trained_rounds": info.get("trained_rounds"),
                    })
            else:
                schema_issues.append(f"[NO_PARTICIPATION] {fp}")

            atr = data.get("allocation_trace")
            if isinstance(atr, dict):
                for rnd, client_map in atr.items():
                    if not isinstance(client_map, dict):
                        continue
                    for cid, info in client_map.items():
                        if not isinstance(info, dict):
                            continue
                        allocation_rows.append({
                            **meta,
                            "round": int(rnd) if str(rnd).isdigit() else rnd,
                            "client_id": int(cid) if str(cid).isdigit() else str(cid),
                            # legacy columns kept for backward compatibility
                            "coverage_steps": info.get("coverage_steps"),
                            "adaptive_steps": info.get("adaptive_steps"),
                            "lambda": info.get("lambda"),
                            "regime_step_counts": safe_json_dumps(info.get("regime_step_counts")),
                            # current runner fields
                            "selected": info.get("selected"),
                            "valid_return": info.get("valid_return"),
                            "failed_to_return": info.get("failed_to_return"),
                            "failure_mode": info.get("failure_mode"),
                            "failure_probability": as_float(info.get("failure_probability")),
                            "failure_uniform_draw": as_float(info.get("failure_uniform_draw")),
                            "assigned_local_steps": info.get("assigned_local_steps"),
                            "effective_steps": info.get("effective_steps"),
                            "avg_loss": as_float(info.get("avg_loss")),
                            "loss_history": as_float(info.get("loss_history")),
                            "omega_before": as_float(info.get("omega_before")),
                            "omega_after": as_float(info.get("omega_after")),
                            "aggregation_rule": info.get("aggregation_rule"),
                            "ena_reference_mode": info.get("ena_reference_mode"),
                            "aggregation_ref_steps": as_float(info.get("aggregation_ref_steps")),
                            "ena_alpha": as_float(info.get("ena_alpha")),
                            "ena_clip_min": as_float(info.get("ena_clip_min")),
                            "ena_clip_max": as_float(info.get("ena_clip_max")),
                            "ena_ratio_raw": as_float(info.get("ena_ratio_raw")),
                            "ena_scale_powered": as_float(info.get("ena_scale_powered")),
                            "ena_scale": as_float(info.get("ena_scale")),
                            "aggregation_base_weight": as_float(info.get("aggregation_base_weight")),
                            "aggregation_effective_weight": as_float(info.get("aggregation_effective_weight")),
                        })
            else:
                schema_issues.append(f"[NO_ALLOCATION_TRACE] {fp}")

    write_csv(os.path.join(args.out_dir, "EXTRACTED_PER_RUN.csv"), per_run)
    write_csv(os.path.join(args.out_dir, "EXTRACTED_GLOBAL_CURVE.csv"), global_curve_rows)
    write_csv(os.path.join(args.out_dir, "EXTRACTED_ROUND_STATS.csv"), round_stats_rows)
    write_csv(os.path.join(args.out_dir, "EXTRACTED_PER_CLIENT_TEST.csv"), per_client_rows)
    write_csv(os.path.join(args.out_dir, "EXTRACTED_PARTICIPATION.csv"), participation_rows)
    write_csv(os.path.join(args.out_dir, "EXTRACTED_ALLOCATION_TRACE.csv"), allocation_rows)

    report_path = os.path.join(args.out_dir, "SCHEMA_REPORT.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        for line in schema_issues:
            f.write(line + "\n")

    print(f"[OK] Exported to: {args.out_dir}")
    print("[OK] Files: EXTRACTED_PER_RUN.csv, EXTRACTED_GLOBAL_CURVE.csv, EXTRACTED_ROUND_STATS.csv, EXTRACTED_PER_CLIENT_TEST.csv, EXTRACTED_PARTICIPATION.csv, EXTRACTED_ALLOCATION_TRACE.csv")
    if schema_issues:
        print(f"[WARN] Schema issues found. See: {report_path}")


if __name__ == "__main__":
    main()
