from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    from scipy import stats as scipy_stats  # type: ignore
except Exception:  # pragma: no cover
    scipy_stats = None

DATASETS = {"ETTm1", "Beijing_AQI", "PhysioNet2012"}
METHOD_LABELS = {
    "fedavg": "FedAvg",
    "fedprox": "FedProx",
    "qfedavg": "qFedAvg",
    "localonly": "LocalOnly",
    "random": "Random",
    "cora_core": "CoRA-Core",
    "cora_stepalloc": "CoRA-StepAlloc",
    "cora_stepalloc_ena": "CoRA-StepAlloc",
    "cora_topk": "Top-m",
    "cora_noema": "NoEMA",
    "cora_nowarmup": "NoWarmup",
    "cora_norefresh": "NoRefresh",
}
KNOWN_LABELS = set(METHOD_LABELS.values()) | {"NoEMA", "NoWarmup", "NoRefresh", "Top-m"}


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


def parse_suite_from_path(path: str) -> str:
    parts = os.path.normpath(path).split(os.sep)
    if "fgcs_revision" in parts:
        i = parts.index("fgcs_revision")
        if i + 1 < len(parts):
            return parts[i + 1]
    for x in ["main", "stress", "ablation", "failure", "backbone"]:
        if x in parts:
            return x
    return "unknown"


def _clean_tag(tag: str) -> str:
    out = str(tag).strip()
    for mode in ["_uniform_r", "_score_correlated_r"]:
        if mode in out:
            out = out.split(mode)[0]
    for prefix in ["gru_", "saits_", "csdi_"]:
        if out.lower().startswith(prefix):
            out = out[len(prefix):]
    return out.replace("_", " ").strip()


def safe_method_label(data: Dict[str, Any], fname: str) -> str:
    meta = data.get("run_meta", {}) if isinstance(data.get("run_meta"), dict) else {}
    tag_raw = str(meta.get("tag", "")).strip()
    if tag_raw:
        tag = _clean_tag(tag_raw)
        tag_compact = tag.replace(" ", "_")
        alias = {
            "FedAvg": "FedAvg",
            "FedProx": "FedProx",
            "qFedAvg": "qFedAvg",
            "LocalOnly": "LocalOnly",
            "Random": "Random",
            "Top-m": "Top-m",
            "Top m": "Top-m",
            "CoRA-Core": "CoRA-Core",
            "CoRA Core": "CoRA-Core",
            "CoRA-StepAlloc": "CoRA-StepAlloc",
            "CoRA StepAlloc": "CoRA-StepAlloc",
            "NoEMA": "NoEMA",
            "NoWarmup": "NoWarmup",
            "NoRefresh": "NoRefresh",
        }
        if tag in alias:
            return alias[tag]
        if tag_compact in alias:
            return alias[tag_compact]
    canon = str(meta.get("method_canonical", meta.get("method", ""))).strip().lower()
    if canon in METHOD_LABELS:
        # For ablation aliases, the raw requested method is more informative.
        raw = str(meta.get("method", "")).strip().lower()
        if raw in METHOD_LABELS and raw in {"cora_noema", "cora_nowarmup", "cora_norefresh"}:
            return METHOD_LABELS[raw]
        return METHOD_LABELS[canon]
    core = fname.replace("metrics_", "").replace(".json", "")
    for d in sorted(DATASETS, key=len, reverse=True):
        needle = f"_{d}_"
        if needle in core:
            core = core.split(needle, 1)[0]
            break
    return _clean_tag(core)


def iter_metric_json(results_dir: str) -> Iterable[Tuple[str, Dict[str, Any]]]:
    for root, _dirs, files in os.walk(results_dir):
        for fname in files:
            if fname.startswith("metrics_") and fname.endswith(".json"):
                fp = os.path.join(root, fname)
                try:
                    with open(fp, "r", encoding="utf-8") as f:
                        yield fp, json.load(f)
                except Exception as e:
                    print(f"[WARN] cannot read {fp}: {e}")


def extract_row(fp: str, data: Dict[str, Any]) -> Dict[str, Any]:
    meta = data.get("run_meta", {}) if isinstance(data.get("run_meta"), dict) else {}
    audit = data.get("audit_global_test", {}) if isinstance(data.get("audit_global_test"), dict) else {}
    final = data.get("final_round_global_test", {}) if isinstance(data.get("final_round_global_test"), dict) else {}
    comm = data.get("comm_stats", {}) if isinstance(data.get("comm_stats"), dict) else {}
    fail = data.get("failure_stats", {}) if isinstance(data.get("failure_stats"), dict) else {}
    ckpt = data.get("selected_checkpoint", {}) if isinstance(data.get("selected_checkpoint"), dict) else {}
    bmeta = meta.get("backbone_meta", {}) if isinstance(meta.get("backbone_meta"), dict) else {}
    backbone = str(meta.get("backbone", bmeta.get("backbone_name", "gru")) or "gru").strip().lower()
    fname = os.path.basename(fp)
    return {
        "file": fp,
        "suite": parse_suite_from_path(fp),
        "backbone": backbone,
        "seed": int(meta.get("seed", -1)),
        "dataset": str(meta.get("dataset", "")),
        "variant": str(meta.get("variant", "")),
        "method_label": safe_method_label(data, fname),
        "method_canonical": str(meta.get("method_canonical", meta.get("method", ""))),
        "checkpoint_selection": str(ckpt.get("selection", meta.get("checkpoint_selection", ""))),
        "checkpoint_round": ckpt.get("round"),
        "rmse_avg": as_float(audit.get("rmse_avg")),
        "rmse_worst": as_float(audit.get("rmse_worst")),
        "mae_avg": as_float(audit.get("mae_avg")),
        "mae_worst": as_float(audit.get("mae_worst")),
        "rmse_micro": as_float(audit.get("rmse_micro")),
        "final_rmse_avg": as_float(final.get("rmse_avg")),
        "final_rmse_worst": as_float(final.get("rmse_worst")),
        "uplink_mb": as_float(comm.get("uplink_mb")),
        "runtime_sec_total": as_float(data.get("runtime_sec_total")),
        "failure_mode": str(fail.get("failure_mode", meta.get("failure_mode", "none"))),
        "failure_rate": float(fail.get("failure_rate", meta.get("failure_rate", 0.0)) or 0.0),
        "valid_return_rate": as_float(fail.get("valid_return_rate")),
        "failed_return_rate": as_float(fail.get("failed_return_rate")),
        "wasted_assigned_step_ratio": as_float(fail.get("wasted_assigned_step_ratio")),
        "returned_assigned_step_ratio": as_float(fail.get("returned_assigned_step_ratio")),
        "empty_return_rounds": as_float(fail.get("empty_return_rounds")),
        "num_parameters": as_float(bmeta.get("num_parameters")),
        "trainable_parameters": as_float(bmeta.get("trainable_parameters")),
    }


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys: List[str] = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys if keys else ["empty"])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def mean_std_n(vals: List[Optional[float]]) -> Tuple[Optional[float], Optional[float], int]:
    arr = np.asarray([v for v in vals if v is not None and np.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return None, None, 0
    return float(arr.mean()), float(arr.std(ddof=1)) if arr.size > 1 else 0.0, int(arr.size)


def paired_stats(a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
    # diff = method - baseline; negative favors method for error/time/budget metrics.
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    diff = diff[np.isfinite(diff)]
    n = int(diff.size)
    if n == 0:
        return {"n_pairs": 0}
    mean = float(diff.mean())
    sd = float(diff.std(ddof=1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else 0.0
    if scipy_stats is not None and n > 1:
        tcrit = float(scipy_stats.t.ppf(0.975, df=n - 1))
        t_p = float(scipy_stats.ttest_1samp(diff, popmean=0.0, nan_policy="omit").pvalue)
        try:
            w_p = float(scipy_stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided", mode="auto").pvalue)
        except Exception:
            w_p = None
    else:
        tcrit = 1.96
        t_p = None
        w_p = None
    ci_low = mean - tcrit * se
    ci_high = mean + tcrit * se
    dz = float(mean / sd) if sd > 0 else (float("inf") if mean != 0 else 0.0)
    wins = int(np.sum(diff < 0.0))
    losses = int(np.sum(diff > 0.0))
    ties = int(np.sum(diff == 0.0))
    return {
        "n_pairs": n,
        "mean_diff": mean,
        "std_diff": sd,
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "paired_t_p": t_p,
        "wilcoxon_p": w_p,
        "cohen_dz": dz,
        "wins": wins,
        "losses": losses,
        "ties": ties,
    }


def holm_adjust(rows: List[Dict[str, Any]], pkey: str, outkey: str) -> None:
    idx_p = [(i, rows[i].get(pkey)) for i in range(len(rows)) if rows[i].get(pkey) is not None]
    idx_p = [(i, float(p)) for i, p in idx_p if np.isfinite(float(p))]
    m = len(idx_p)
    if m == 0:
        return
    idx_p.sort(key=lambda x: x[1])
    prev = 0.0
    for rank, (i, p) in enumerate(idx_p, start=1):
        adj = min(1.0, max(prev, (m - rank + 1) * p))
        rows[i][outkey] = adj
        prev = adj


def main() -> None:
    ap = argparse.ArgumentParser("Summarize FGCS revision experiments and run paired tests")
    ap.add_argument("--results_dir", type=str, default="results/fgcs_revision")
    ap.add_argument("--out_dir", type=str, default="results/fgcs_revision/summary")
    ap.add_argument("--baseline", type=str, default="Random")
    args = ap.parse_args()

    rows = [extract_row(fp, data) for fp, data in iter_metric_json(args.results_dir)]
    rows = [r for r in rows if r["dataset"] in DATASETS]
    os.makedirs(args.out_dir, exist_ok=True)
    write_csv(os.path.join(args.out_dir, "revision_per_run.csv"), rows)

    group_keys = ["suite", "backbone", "dataset", "variant", "method_label", "failure_mode", "failure_rate"]
    grouped: DefaultDict[Tuple[Any, ...], Dict[str, List[Optional[float]]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = tuple(r[k] for k in group_keys)
        for metric in [
            "rmse_avg",
            "rmse_worst",
            "mae_avg",
            "mae_worst",
            "uplink_mb",
            "runtime_sec_total",
            "valid_return_rate",
            "wasted_assigned_step_ratio",
            "returned_assigned_step_ratio",
            "empty_return_rounds",
            "num_parameters",
        ]:
            grouped[key][metric].append(r.get(metric))
    summary_rows: List[Dict[str, Any]] = []
    for key, mmap in grouped.items():
        out = {k: v for k, v in zip(group_keys, key)}
        for metric, vals in mmap.items():
            mu, sd, n = mean_std_n(vals)
            out[f"{metric}_mean"] = mu
            out[f"{metric}_std"] = sd
            out[f"{metric}_n"] = n
        summary_rows.append(out)
    write_csv(os.path.join(args.out_dir, "revision_summary_mean_std.csv"), summary_rows)

    scenario_keys = ["suite", "backbone", "dataset", "variant", "failure_mode", "failure_rate"]
    by_scenario_method_seed: DefaultDict[Tuple[Any, ...], Dict[str, Dict[int, Dict[str, Any]]]] = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        skey = tuple(r[k] for k in scenario_keys)
        by_scenario_method_seed[skey][str(r["method_label"])][int(r["seed"])] = r

    paired_rows: List[Dict[str, Any]] = []
    for skey, method_map in by_scenario_method_seed.items():
        if args.baseline not in method_map:
            continue
        base_map = method_map[args.baseline]
        for method, mmap in method_map.items():
            if method == args.baseline:
                continue
            common = sorted(set(base_map) & set(mmap))
            if not common:
                continue
            for metric in ["rmse_worst", "rmse_avg", "uplink_mb", "runtime_sec_total", "wasted_assigned_step_ratio"]:
                a = np.asarray([mmap[s].get(metric) for s in common], dtype=np.float64)
                b = np.asarray([base_map[s].get(metric) for s in common], dtype=np.float64)
                finite = np.isfinite(a) & np.isfinite(b)
                if not finite.any():
                    continue
                stat = paired_stats(a[finite], b[finite])
                paired_rows.append(
                    {
                        **{k: v for k, v in zip(scenario_keys, skey)},
                        "baseline": args.baseline,
                        "method": method,
                        "metric": metric,
                        **stat,
                    }
                )

    families: DefaultDict[Tuple[Any, ...], List[int]] = defaultdict(list)
    for i, r in enumerate(paired_rows):
        fam = tuple(r[k] for k in scenario_keys + ["metric"])
        families[fam].append(i)
    for _fam, indices in families.items():
        sub = [paired_rows[i] for i in indices]
        holm_adjust(sub, "paired_t_p", "paired_t_p_holm")
        holm_adjust(sub, "wilcoxon_p", "wilcoxon_p_holm")
        for loc, i in enumerate(indices):
            paired_rows[i].update({k: sub[loc].get(k) for k in ["paired_t_p_holm", "wilcoxon_p_holm"]})
    write_csv(os.path.join(args.out_dir, "revision_paired_tests.csv"), paired_rows)

    failure_rows = [r for r in summary_rows if str(r.get("suite")) == "failure"]
    write_csv(os.path.join(args.out_dir, "revision_failure_summary.csv"), failure_rows)
    backbone_rows = [r for r in summary_rows if str(r.get("suite")) == "backbone"]
    write_csv(os.path.join(args.out_dir, "revision_backbone_summary.csv"), backbone_rows)

    report = os.path.join(args.out_dir, "SUMMARY_REPORT.txt")
    with open(report, "w", encoding="utf-8") as f:
        f.write(f"per_run_rows={len(rows)}\n")
        f.write(f"summary_rows={len(summary_rows)}\n")
        f.write(f"paired_rows={len(paired_rows)}\n")
        f.write(f"scipy_available={scipy_stats is not None}\n")
        f.write("paired_difference = method - baseline; negative favors the method for RMSE/MAE/runtime/wasted-budget metrics.\n")
        f.write("scenario pairing includes suite, backbone, dataset, variant, failure_mode, and failure_rate.\n")
    print(f"[OK] per_run={len(rows)} summary={len(summary_rows)} paired={len(paired_rows)}")
    print(f"[OK] wrote summary directory: {args.out_dir}")


if __name__ == "__main__":
    main()
