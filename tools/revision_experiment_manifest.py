from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import sys
from typing import Dict, List, Tuple


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_seeds(spec: str) -> List[int]:
    out: List[int] = []
    for part in str(spec).replace(",", " ").split():
        part = part.strip()
        if not part:
            continue
        if ":" in part or "-" in part:
            sep = ":" if ":" in part else "-"
            a, b = part.split(sep, 1)
            out.extend(list(range(int(a), int(b) + 1)))
        else:
            out.append(int(part))
    seen = set()
    uniq = []
    for s in out:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


def parse_csv_list(spec: str) -> List[str]:
    return [x.strip() for x in str(spec).replace(";", ",").split(",") if x.strip()]


def method_args(label: str) -> Tuple[str, List[str]]:
    key = label.lower()
    if key == "fedavg":
        return "fedavg", ["--method", "fedavg"]
    if key == "fedprox":
        return "fedprox", ["--method", "fedprox", "--mu", "0.01"]
    if key == "qfedavg":
        return "qfedavg", ["--method", "qfedavg", "--q_param", "0.2"]
    if key == "localonly":
        return "localonly", ["--method", "localonly"]
    if key == "random":
        return "random", ["--method", "random"]
    if key in {"top-m", "topm", "cora-topm"}:
        return "cora_topk", ["--method", "cora_topk"]
    if key in {"cora-core", "core"}:
        return "cora_core", ["--method", "cora_core"]
    if key in {"cora-stepalloc", "stepalloc"}:
        return "cora_stepalloc", ["--method", "cora_stepalloc", "--aggregation_rule", "ena", "--ena_reference_mode", "mean_effective", "--ena_alpha", "1.0"]
    if key == "noema":
        return "cora_noema", ["--method", "cora_noema"]
    if key == "nowarmup":
        return "cora_nowarmup", ["--method", "cora_nowarmup"]
    if key == "norefresh":
        return "cora_norefresh", ["--method", "cora_norefresh"]
    raise ValueError(f"unknown method label: {label}")


def safe_rate_tag(rate: float) -> str:
    return (f"{rate:.2f}").replace(".", "p")


def safe_label(x: str) -> str:
    return str(x).replace(" ", "_").replace("/", "_").replace("\\", "_")


def add_job(
    rows: List[Dict[str, str]],
    *,
    suite: str,
    seed: int,
    dataset: str,
    variant: str,
    method_label: str,
    backbone: str,
    results_dir: str,
    data_root: str,
    device: str,
    runner_path: str,
    common: List[str],
    extra: List[str],
) -> None:
    canonical, margs = method_args(method_label)
    backbone = str(backbone).strip().lower()
    tag_core = safe_label(method_label)
    tag = tag_core if backbone == "gru" else f"{backbone}_{tag_core}"
    failure_mode = "none"
    failure_rate = "0.0"
    if "--failure_mode" in extra:
        idx = extra.index("--failure_mode")
        failure_mode = extra[idx + 1]
    if "--failure_rate" in extra:
        idx = extra.index("--failure_rate")
        failure_rate = extra[idx + 1]
        tag = f"{tag}_{failure_mode}_r{safe_rate_tag(float(failure_rate))}"

    out_dir = os.path.join(results_dir, suite, backbone, f"seed_{seed}", dataset, variant)
    if failure_mode != "none" and float(failure_rate) > 0:
        out_dir = os.path.join(out_dir, f"{failure_mode}_r{safe_rate_tag(float(failure_rate))}")
    expected_json = os.path.join(out_dir, f"metrics_{tag}_{dataset}_{variant}.json")
    argv = [
        sys.executable,
        "-u",
        runner_path,
        "--data_root",
        data_root,
        "--dataset",
        dataset,
        "--variant",
        variant,
        "--seed",
        str(seed),
        "--device",
        device,
        "--out_dir",
        out_dir,
        "--tag",
        tag,
        "--backbone",
        backbone,
    ] + common + margs + extra
    rows.append(
        {
            "suite": suite,
            "backbone": backbone,
            "seed": str(seed),
            "dataset": dataset,
            "variant": variant,
            "method_label": method_label,
            "method_canonical": canonical,
            "failure_mode": failure_mode,
            "failure_rate": failure_rate,
            "out_dir": out_dir,
            "expected_json": expected_json,
            "argv_json": json.dumps(argv, ensure_ascii=False),
            "command": " ".join(shlex.quote(x) for x in argv),
        }
    )


def main() -> None:
    ap = argparse.ArgumentParser("Generate command manifest for FGCS CoRA revision experiments")
    ap.add_argument("--out_csv", type=str, default="results/fgcs_revision/REVISION_MANIFEST.csv")
    ap.add_argument("--results_dir", type=str, default="results/fgcs_revision")
    ap.add_argument("--data_root", type=str, default="data")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--runner_path", type=str, default=os.path.join(SCRIPT_DIR, "main_experiment_runner.py"))
    ap.add_argument("--seeds", type=str, default="0:19", help="main/stress/ablation GRU seeds; e.g. '0:19' or '0 1 2'")
    ap.add_argument("--failure_seeds", type=str, default="0:9")
    ap.add_argument("--backbone_seeds", type=str, default="0:9", help="SAITS backbone-check seeds")
    ap.add_argument("--csdi_seeds", type=str, default="0:4", help="CSDI backbone-check seeds; CSDI is much slower")
    ap.add_argument("--suite", choices=["all", "main", "stress", "ablation", "failure", "backbone"], default="all")
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--local_steps", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--saits_variants", type=str, default="hetero_h4,mcar_p10")
    ap.add_argument("--csdi_variants", type=str, default="hetero_h4")
    ap.add_argument("--include_csdi", type=int, default=1)
    ap.add_argument("--csdi_batch_size", type=int, default=16)
    ap.add_argument("--csdi_eval_every", type=int, default=10)
    args = ap.parse_args()

    seeds = parse_seeds(args.seeds)
    failure_seeds = parse_seeds(args.failure_seeds)
    backbone_seeds = parse_seeds(args.backbone_seeds)
    csdi_seeds = parse_seeds(args.csdi_seeds)
    common = [
        "--rounds",
        str(args.rounds),
        "--local_steps",
        str(args.local_steps),
        "--batch_size",
        str(args.batch_size),
        "--eval_every",
        str(args.eval_every),
        "--eval_split",
        "val",
        "--checkpoint_selection",
        "best_val_rmse_avg",
        "--beta_hardness",
        "0.7",
        "--h0",
        "1.0",
        "--rho",
        "0.6",
        "--T_part",
        "15",
        "--T_refresh",
        "5",
        "--K_min",
        "5",
        "--score_floor",
        "1e-9",
        "--stepalloc_min_steps",
        "15",
        "--stepalloc_max_steps",
        "25",
        "--stepalloc_power",
        "1.0",
    ]
    rows: List[Dict[str, str]] = []

    if args.suite in {"all", "main"}:
        for seed in seeds:
            for dataset in ["ETTm1", "Beijing_AQI", "PhysioNet2012"]:
                for variant in ["hetero", "mcar_p10"]:
                    for method in ["FedAvg", "FedProx", "qFedAvg", "LocalOnly", "Random", "CoRA-Core", "CoRA-StepAlloc"]:
                        add_job(
                            rows,
                            suite="main",
                            seed=seed,
                            dataset=dataset,
                            variant=variant,
                            method_label=method,
                            backbone="gru",
                            results_dir=args.results_dir,
                            data_root=args.data_root,
                            device=args.device,
                            runner_path=args.runner_path,
                            common=common,
                            extra=[],
                        )

    if args.suite in {"all", "stress"}:
        stress_variants = [f"mcar_p{x}" for x in [10, 20, 30, 40, 50, 60, 70, 80]] + [
            "hetero_h1",
            "hetero_h2",
            "hetero_h3",
            "hetero_h4",
        ]
        for seed in seeds:
            for variant in stress_variants:
                for method in ["FedAvg", "Random", "Top-m", "CoRA-Core", "CoRA-StepAlloc"]:
                    add_job(
                        rows,
                        suite="stress",
                        seed=seed,
                        dataset="ETTm1",
                        variant=variant,
                        method_label=method,
                        backbone="gru",
                        results_dir=args.results_dir,
                        data_root=args.data_root,
                        device=args.device,
                        runner_path=args.runner_path,
                        common=common,
                        extra=[],
                    )

    if args.suite in {"all", "ablation"}:
        for seed in seeds:
            for method in ["Random", "Top-m", "CoRA-Core", "NoEMA", "NoWarmup", "NoRefresh", "CoRA-StepAlloc"]:
                add_job(
                    rows,
                    suite="ablation",
                    seed=seed,
                    dataset="ETTm1",
                    variant="hetero_h4",
                    method_label=method,
                    backbone="gru",
                    results_dir=args.results_dir,
                    data_root=args.data_root,
                    device=args.device,
                    runner_path=args.runner_path,
                    common=common,
                    extra=[],
                )
            for method in ["Random", "Top-m", "CoRA-Core", "CoRA-StepAlloc"]:
                add_job(
                    rows,
                    suite="ablation",
                    seed=seed,
                    dataset="ETTm1",
                    variant="mcar_p70",
                    method_label=method,
                    backbone="gru",
                    results_dir=args.results_dir,
                    data_root=args.data_root,
                    device=args.device,
                    runner_path=args.runner_path,
                    common=common,
                    extra=[],
                )

    if args.suite in {"all", "failure"}:
        for seed in failure_seeds:
            for variant in ["hetero_h4", "mcar_p10"]:
                for mode in ["uniform", "score_correlated"]:
                    for rate in [0.05, 0.10, 0.20]:
                        extra = [
                            "--failure_mode",
                            mode,
                            "--failure_rate",
                            str(rate),
                            "--failure_qmax",
                            "0.8",
                            "--failure_apply_full_rounds",
                            "1",
                        ]
                        for method in ["Random", "CoRA-Core", "CoRA-StepAlloc"]:
                            add_job(
                                rows,
                                suite="failure",
                                seed=seed,
                                dataset="ETTm1",
                                variant=variant,
                                method_label=method,
                                backbone="gru",
                                results_dir=args.results_dir,
                                data_root=args.data_root,
                                device=args.device,
                                runner_path=args.runner_path,
                                common=common,
                                extra=extra,
                            )

    if args.suite in {"all", "backbone"}:
        # Reviewer-requested additional backbone check. SAITS is the primary
        # added attention-based backbone; CSDI is optional and intentionally
        # smaller by default because diffusion sampling is substantially slower.
        for seed in backbone_seeds:
            for variant in parse_csv_list(args.saits_variants):
                for method in ["FedAvg", "Random", "CoRA-Core", "CoRA-StepAlloc"]:
                    add_job(
                        rows,
                        suite="backbone",
                        seed=seed,
                        dataset="ETTm1",
                        variant=variant,
                        method_label=method,
                        backbone="saits",
                        results_dir=args.results_dir,
                        data_root=args.data_root,
                        device=args.device,
                        runner_path=args.runner_path,
                        common=common,
                        extra=[],
                    )
        if int(args.include_csdi):
            csdi_common = []
            for i, x in enumerate(common):
                # Override batch_size and eval_every for CSDI to keep the check tractable.
                if i > 0 and common[i - 1] in {"--batch_size", "--eval_every"}:
                    continue
                if x in {"--batch_size", "--eval_every"}:
                    continue
                csdi_common.append(x)
            csdi_common += ["--batch_size", str(args.csdi_batch_size), "--eval_every", str(args.csdi_eval_every)]
            for seed in csdi_seeds:
                for variant in parse_csv_list(args.csdi_variants):
                    for method in ["FedAvg", "Random", "CoRA-Core", "CoRA-StepAlloc"]:
                        add_job(
                            rows,
                            suite="backbone",
                            seed=seed,
                            dataset="ETTm1",
                            variant=variant,
                            method_label=method,
                            backbone="csdi",
                            results_dir=args.results_dir,
                            data_root=args.data_root,
                            device=args.device,
                            runner_path=args.runner_path,
                            common=csdi_common,
                            extra=["--csdi_num_steps", "20", "--csdi_eval_samples", "1"],
                        )

    if not rows:
        raise RuntimeError("No jobs generated; check --suite and seed options")
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OK] wrote {len(rows)} jobs -> {args.out_csv}")
    print("[OK] first command:")
    print(rows[0]["command"])


if __name__ == "__main__":
    main()
