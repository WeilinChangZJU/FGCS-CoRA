from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from typing import Dict, List


def read_manifest(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser("Run jobs listed in REVISION_MANIFEST.csv")
    ap.add_argument("--manifest", type=str, default="results/fgcs_revision/REVISION_MANIFEST.csv")
    ap.add_argument("--log_dir", type=str, default="logs/fgcs_revision")
    ap.add_argument("--suite", type=str, default="", help="optional suite filter")
    ap.add_argument("--dataset", type=str, default="", help="optional dataset filter")
    ap.add_argument("--variant", type=str, default="", help="optional variant filter")
    ap.add_argument("--method", type=str, default="", help="optional method_label filter")
    ap.add_argument("--backbone", type=str, default="", help="optional backbone filter: gru, saits, csdi")
    ap.add_argument("--skip_existing", type=int, default=1)
    ap.add_argument("--dry_run", type=int, default=0)
    ap.add_argument("--max_jobs", type=int, default=0)
    ap.add_argument("--cwd", type=str, default=".")
    args = ap.parse_args()

    rows = read_manifest(args.manifest)
    filters = {"suite": args.suite, "dataset": args.dataset, "variant": args.variant, "method_label": args.method, "backbone": args.backbone}
    filt_rows = []
    for r in rows:
        ok = True
        for k, v in filters.items():
            if v and str(r.get(k, "")) != v:
                ok = False
                break
        if ok:
            filt_rows.append(r)
    if args.max_jobs and args.max_jobs > 0:
        filt_rows = filt_rows[: int(args.max_jobs)]

    os.makedirs(args.log_dir, exist_ok=True)
    print(f"[INFO] loaded={len(rows)} selected={len(filt_rows)} manifest={args.manifest}")
    done = 0
    skipped = 0
    failed = 0
    for i, row in enumerate(filt_rows, start=1):
        expected = row.get("expected_json", "")
        if args.skip_existing and expected and os.path.isfile(expected):
            skipped += 1
            print(f"[{i}/{len(filt_rows)}] SKIP existing {expected}")
            continue
        argv = json.loads(row["argv_json"])
        log_name = f"{row.get('suite','suite')}_{row.get('backbone','gru')}_s{row.get('seed')}_{row.get('dataset')}_{row.get('variant')}_{row.get('method_label')}_{row.get('failure_mode','none')}_{row.get('failure_rate','0')}.log"
        log_name = log_name.replace(os.sep, "_").replace(" ", "_")
        log_path = os.path.join(args.log_dir, log_name)
        print(f"[{i}/{len(filt_rows)}] RUN {row.get('suite')} backbone={row.get('backbone','gru')} seed={row.get('seed')} {row.get('dataset')} {row.get('variant')} {row.get('method_label')} failure={row.get('failure_mode')}:{row.get('failure_rate')}")
        print(" ".join(argv))
        if args.dry_run:
            continue
        os.makedirs(os.path.dirname(expected), exist_ok=True)
        t0 = time.time()
        with open(log_path, "w", encoding="utf-8") as logf:
            proc = subprocess.run(argv, cwd=args.cwd, stdout=logf, stderr=subprocess.STDOUT, text=True)
        dt = time.time() - t0
        if proc.returncode != 0:
            failed += 1
            print(f"[FAIL] exit={proc.returncode} time={dt:.1f}s log={log_path}")
            sys.exit(proc.returncode)
        done += 1
        print(f"[OK] time={dt:.1f}s log={log_path}")
    print(f"[DONE] completed={done} skipped={skipped} failed={failed}")


if __name__ == "__main__":
    main()
