from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Set, Tuple


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(ROOT, "results", "fgcs_revision")
PRIORITY_DIR = os.path.join(RESULTS_DIR, "priority")


METHOD_ALIASES = {
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


def norm_method(data: Dict[str, Any], fname: str) -> str:
    meta = data.get("run_meta", {}) if isinstance(data.get("run_meta"), dict) else {}
    tag = str(meta.get("tag", "")).strip()
    for suffix in ["_uniform_r", "_score_correlated_r"]:
        if suffix in tag:
            tag = tag.split(suffix, 1)[0]
    for prefix in ["saits_", "csdi_", "gru_"]:
        if tag.lower().startswith(prefix):
            tag = tag[len(prefix):]
    tag_alias = {
        "FedAvg": "FedAvg",
        "FedProx": "FedProx",
        "qFedAvg": "qFedAvg",
        "LocalOnly": "LocalOnly",
        "Random": "Random",
        "CoRA-Core": "CoRA-Core",
        "CoRA-StepAlloc": "CoRA-StepAlloc",
        "Top-m": "Top-m",
    }
    if tag in tag_alias:
        return tag_alias[tag]
    canon = str(meta.get("method_canonical", meta.get("method", ""))).strip().lower()
    return METHOD_ALIASES.get(canon, tag or canon or fname)


def suite_from_path(path: str) -> str:
    parts = os.path.normpath(path).split(os.sep)
    if "fgcs_revision" in parts:
        i = parts.index("fgcs_revision")
        if i + 1 < len(parts):
            return parts[i + 1]
    return "unknown"


def iter_metrics(results_dir: str) -> Iterable[Dict[str, Any]]:
    skip_parts = {"smoke_summary", "smoke_figures", "smoke_tables", "priority_summary"}
    for root, _dirs, files in os.walk(results_dir):
        parts = set(os.path.normpath(root).split(os.sep))
        if parts & skip_parts:
            continue
        for fname in files:
            if not (fname.startswith("metrics_") and fname.endswith(".json")):
                continue
            fp = os.path.join(root, fname)
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue
            meta = data.get("run_meta", {}) if isinstance(data.get("run_meta"), dict) else {}
            fail = data.get("failure_stats", {}) if isinstance(data.get("failure_stats"), dict) else {}
            row = {
                "file": fp,
                "suite": suite_from_path(fp),
                "backbone": str(meta.get("backbone", "gru")).strip().lower(),
                "seed": int(meta.get("seed", -1)),
                "dataset": str(meta.get("dataset", "")),
                "variant": str(meta.get("variant", "")),
                "method": norm_method(data, fname),
                "failure_mode": str(fail.get("failure_mode", meta.get("failure_mode", "none"))),
                "failure_rate": round(float(fail.get("failure_rate", meta.get("failure_rate", 0.0)) or 0.0), 6),
                "valid_return_rate": fail.get("valid_return_rate"),
                "wasted_assigned_step_ratio": fail.get("wasted_assigned_step_ratio"),
                "returned_assigned_step_ratio": fail.get("returned_assigned_step_ratio"),
                "empty_return_rounds": fail.get("empty_return_rounds"),
            }
            yield row


def read_manifest(path: str) -> List[Dict[str, str]]:
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def expected_missing_jobs(manifest_path: str) -> Tuple[int, List[str]]:
    rows = read_manifest(manifest_path)
    missing = []
    for row in rows:
        expected = row.get("expected_json", "")
        if expected and not os.path.isabs(expected):
            expected = os.path.join(ROOT, expected)
        if expected and not os.path.isfile(expected):
            missing.append(row.get("command") or expected)
    return len(rows), missing


def scenario_key(row: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        row["suite"],
        row["backbone"],
        row["dataset"],
        row["variant"],
        row["failure_mode"],
        row["failure_rate"],
    )


def coverage_for(
    rows: Sequence[Dict[str, Any]],
    *,
    suite: str,
    backbone: str,
    datasets: Sequence[str],
    variants: Sequence[str],
    failure_modes: Sequence[str],
    failure_rates: Sequence[float],
    methods: Sequence[str],
    threshold: int,
    fedavg_preferred: bool = False,
) -> Tuple[List[str], bool]:
    by_scenario_method: DefaultDict[Tuple[Any, ...], DefaultDict[str, Set[int]]] = defaultdict(lambda: defaultdict(set))
    for row in rows:
        if row["suite"] != suite or row["backbone"] != backbone:
            continue
        if row["dataset"] not in datasets or row["variant"] not in variants:
            continue
        if row["failure_mode"] not in failure_modes:
            continue
        if round(float(row["failure_rate"]), 6) not in {round(float(x), 6) for x in failure_rates}:
            continue
        by_scenario_method[scenario_key(row)][row["method"]].add(int(row["seed"]))

    lines: List[str] = []
    ok_all = True
    for dataset in datasets:
        for variant in variants:
            for fmode in failure_modes:
                for frate in failure_rates:
                    key = (suite, backbone, dataset, variant, fmode, round(float(frate), 6))
                    mmap = by_scenario_method.get(key, defaultdict(set))
                    required_seed_sets = [mmap.get(m, set()) for m in methods]
                    common = set.intersection(*required_seed_sets) if required_seed_sets else set()
                    missing_methods = [m for m in methods if not mmap.get(m)]
                    missing_by_method = {m: sorted(set(range(10)) - mmap.get(m, set())) for m in methods if len(mmap.get(m, set())) < threshold}
                    status = "PASS" if len(common) >= threshold and not missing_methods else "FAIL"
                    if status != "PASS":
                        ok_all = False
                    extra = ""
                    if fedavg_preferred:
                        extra = f"; FedAvg seeds={sorted(mmap.get('FedAvg', set()))}"
                    lines.append(
                        f"| {suite} | {backbone} | {dataset} | {variant} | {fmode} | {frate:.2f} | {len(common)} | {status} | {missing_methods} | {missing_by_method}{extra} |"
                    )
    return lines, ok_all


def failure_nonreturn_check(rows: Sequence[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    lines = []
    ok = False
    for row in rows:
        if row["suite"] == "failure" and row["failure_mode"] in {"uniform", "score_correlated"}:
            try:
                wasted = float(row.get("wasted_assigned_step_ratio") or 0.0)
                valid = float(row.get("valid_return_rate") or 1.0)
            except Exception:
                continue
            if wasted > 0.0 or valid < 1.0:
                ok = True
                lines.append(f"- `{row['file']}`: valid_return_rate={valid}, wasted_assigned_step_ratio={wasted}")
    return ok, lines


def main() -> None:
    ap = argparse.ArgumentParser("Check priority paired-seed coverage for CoRA FGCS revision")
    ap.add_argument("--results_dir", type=str, default=RESULTS_DIR)
    ap.add_argument("--priority_dir", type=str, default=PRIORITY_DIR)
    ap.add_argument("--out", type=str, default=os.path.join(PRIORITY_DIR, "PRIORITY_COVERAGE_REPORT.md"))
    args = ap.parse_args()

    rows = list(iter_metrics(args.results_dir))
    main_datasets = ["ETTm1", "Beijing_AQI", "PhysioNet2012"]
    main_variants = ["hetero", "mcar_p10"]
    stress_variants = ["hetero_h1", "hetero_h2", "hetero_h3", "hetero_h4"] + [f"mcar_p{x}" for x in [10, 20, 30, 40, 50, 60, 70, 80]]
    required_methods = ["Random", "CoRA-Core", "CoRA-StepAlloc"]

    table_lines = [
        "| Suite | Backbone | Dataset | Variant | Failure mode | Failure rate | Complete paired seeds | Status | Missing methods | Missing seeds by method |",
        "|---|---|---|---|---|---:|---:|---|---|---|",
    ]
    ok_flags = []
    lines, ok = coverage_for(
        rows,
        suite="main",
        backbone="gru",
        datasets=main_datasets,
        variants=main_variants,
        failure_modes=["none"],
        failure_rates=[0.0],
        methods=required_methods,
        threshold=10,
    )
    table_lines.extend(lines)
    ok_flags.append(ok)
    lines, ok = coverage_for(
        rows,
        suite="stress",
        backbone="gru",
        datasets=["ETTm1"],
        variants=stress_variants,
        failure_modes=["none"],
        failure_rates=[0.0],
        methods=required_methods,
        threshold=10,
    )
    table_lines.extend(lines)
    ok_flags.append(ok)
    lines, ok = coverage_for(
        rows,
        suite="failure",
        backbone="gru",
        datasets=["ETTm1"],
        variants=["hetero_h4"],
        failure_modes=["uniform", "score_correlated"],
        failure_rates=[0.05, 0.10, 0.20],
        methods=required_methods,
        threshold=10,
    )
    table_lines.extend(lines)
    ok_flags.append(ok)
    lines, ok = coverage_for(
        rows,
        suite="backbone",
        backbone="saits",
        datasets=["ETTm1"],
        variants=["hetero_h4"],
        failure_modes=["none"],
        failure_rates=[0.0],
        methods=required_methods,
        threshold=10,
        fedavg_preferred=True,
    )
    table_lines.extend(lines)
    ok_flags.append(ok)

    manifest_names = [
        "P0A_main_central_paired.csv",
        "P0B_stress_core_10seeds.csv",
        "P0D_failure_h4_10seeds.csv",
        "P0C_saits_h4_10seeds.csv",
    ]
    manifest_lines = ["| Manifest | Jobs | Missing expected JSONs |", "|---|---:|---:|"]
    missing_detail: List[str] = []
    for name in manifest_names:
        total, missing = expected_missing_jobs(os.path.join(args.priority_dir, name))
        manifest_lines.append(f"| `{name}` | {total} | {len(missing)} |")
        if missing:
            missing_detail.append(f"### Missing jobs for `{name}`")
            missing_detail.extend(f"- `{x}`" for x in missing[:200])
            if len(missing) > 200:
                missing_detail.append(f"- ... {len(missing) - 200} additional missing jobs omitted")

    nonreturn_ok, nonreturn_lines = failure_nonreturn_check(rows)
    p0_ok = all(ok_flags) and nonreturn_ok
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    report = [
        "# Priority Coverage Report",
        "",
        f"Metrics scanned: {len(rows)}",
        "",
        "## P0 Coverage Gate",
        "",
        "Minimum threshold: at least 10 complete paired seeds for Random, CoRA-Core, and CoRA-StepAlloc in every required P0 scenario. For SAITS, FedAvg is preferred as a schedule-level reference and reported separately.",
        "",
        *table_lines,
        "",
        f"Minimum threshold satisfied: {'YES' if p0_ok else 'NO'}",
        f"Final Section 5 can be drafted: {'YES' if p0_ok else 'NO'}",
        "",
        "## Manifest Completion",
        "",
        *manifest_lines,
        "",
        "## Failure-Return Non-Return Check",
        "",
        f"Failure-return metrics include selected-client non-return cases: {'YES' if nonreturn_ok else 'NO'}",
        *(nonreturn_lines[:20] if nonreturn_lines else ["- No non-return evidence found yet."]),
        "",
        "## Missing Jobs",
        "",
        *(missing_detail if missing_detail else ["No missing expected JSONs for P0 manifests."]),
        "",
    ]
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print(f"[OK] wrote {args.out}")
    print(f"p0_complete={p0_ok} metrics={len(rows)}")


if __name__ == "__main__":
    main()
