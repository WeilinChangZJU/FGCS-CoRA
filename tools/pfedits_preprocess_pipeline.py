from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import random
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split  # type: ignore

# =============================================================================
# Utilities
# =============================================================================


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _read_json(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def stable_uint32(*parts: object) -> int:
    s = "||".join(str(p) for p in parts).encode("utf-8")
    h = hashlib.sha256(s).digest()
    return int.from_bytes(h[:4], byteorder="little", signed=False)


# =============================================================================
# Masks / statistics
# =============================================================================


def build_missing_mask(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    return np.isfinite(arr).astype(np.float32)


def make_holdout_mask(
    M: np.ndarray,
    ratio: float,
    rng: np.random.Generator,
    mode: str = "global",
    min_per_col: int = 0,
) -> np.ndarray:
    M = np.asarray(M, dtype=np.float32)
    if M.ndim != 2:
        raise ValueError(f"M must be [T,D], got {M.shape}")
    H = np.zeros_like(M, dtype=np.float32)
    ratio = float(ratio)
    if ratio <= 0:
        return H
    Mbin = (M > 0.5)

    if mode == "global":
        obs = np.flatnonzero(Mbin.reshape(-1))
        if obs.size == 0:
            return H
        k = int(np.round(obs.size * ratio))
        k = max(k, 1)
        k = min(k, obs.size)
        sel = rng.choice(obs, size=k, replace=False)
        H.reshape(-1)[sel] = 1.0
    elif mode == "per_col":
        _, D = M.shape
        for d in range(D):
            obs_d = np.where(Mbin[:, d])[0]
            if obs_d.size == 0:
                continue
            k = int(np.round(obs_d.size * ratio))
            if min_per_col > 0:
                k = max(k, int(min_per_col))
            k = min(k, obs_d.size)
            if k <= 0:
                continue
            sel = rng.choice(obs_d, size=k, replace=False)
            H[sel, d] = 1.0
    else:
        raise ValueError(f"Unknown holdout mode: {mode}")

    if np.any((H > 0.5) & (~Mbin)):
        raise RuntimeError("Holdout mask must be a subset of M")
    return H.astype(np.float32)


def describe_mask(M: np.ndarray) -> Dict[str, float]:
    M = np.asarray(M, dtype=np.float32)
    if M.size == 0:
        return {"missing_rate": float("nan")}
    miss = 1.0 - M.reshape(-1)
    miss = miss[np.isfinite(miss)]
    if miss.size == 0:
        return {"missing_rate": float("nan")}
    q = np.quantile(miss, [0.0, 0.5, 0.9, 1.0])
    return {
        "missing_rate": float(np.mean(miss)),
        "q0": float(q[0]),
        "q50": float(q[1]),
        "q90": float(q[2]),
        "q100": float(q[3]),
    }


def describe_holdout_rate(M: np.ndarray, H: np.ndarray) -> Dict[str, float]:
    Mb = (np.asarray(M) > 0.5)
    Hb = (np.asarray(H) > 0.5)
    obs = int(Mb.sum())
    hol = int((Hb & Mb).sum())
    return {
        "obs_cnt": float(obs),
        "holdout_cnt": float(hol),
        "holdout_rate": float(hol / obs) if obs > 0 else float("nan"),
    }


# =============================================================================
# Global scaler
# =============================================================================


@dataclass
class GlobalMaskedStandardScaler:
    mean_: np.ndarray
    std_: np.ndarray
    eps_: float = 1e-6

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"X must be [T,D], got {X.shape}")
        out = X.copy()
        fin = np.isfinite(out)
        if np.any(fin):
            idx_d = np.where(fin)[1]
            out[fin] = (out[fin] - self.mean_[idx_d]) / self.std_[idx_d]
        out[~fin] = 0.0
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def save(self, out_root: str, meta: Dict[str, object]) -> None:
        ensure_dir(out_root)
        payload = {
            "type": "GlobalMaskedStandardScaler",
            "created_at": utc_now_iso(),
            "mean": np.asarray(self.mean_, dtype=np.float32),
            "std": np.asarray(self.std_, dtype=np.float32),
            "eps": float(self.eps_),
            "meta": meta,
        }
        with open(os.path.join(out_root, "scaler.pkl"), "wb") as f:
            pickle.dump(payload, f)


def fit_global_scaler_from_train(train_arrays: Iterable[np.ndarray], eps: float = 1e-6) -> GlobalMaskedStandardScaler:
    sums: Optional[np.ndarray] = None
    sumsq: Optional[np.ndarray] = None
    cnts: Optional[np.ndarray] = None
    for X in train_arrays:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"Train array must be [T,D], got {X.shape}")
        fin = np.isfinite(X)
        D = X.shape[1]
        if sums is None:
            sums = np.zeros((D,), dtype=np.float64)
            sumsq = np.zeros((D,), dtype=np.float64)
            cnts = np.zeros((D,), dtype=np.int64)
        if D != sums.shape[0]:
            raise ValueError("Feature dimension mismatch across clients")
        for d in range(D):
            vals = X[fin[:, d], d]
            if vals.size == 0:
                continue
            sums[d] += float(np.sum(vals, dtype=np.float64))
            sumsq[d] += float(np.sum(vals.astype(np.float64) ** 2))
            cnts[d] += int(vals.size)
    if sums is None or sumsq is None or cnts is None:
        raise RuntimeError("No train data available for scaler fitting")

    D = int(sums.shape[0])
    mean = np.zeros((D,), dtype=np.float32)
    std = np.ones((D,), dtype=np.float32)
    for d in range(D):
        if cnts[d] <= 0:
            continue
        mu = sums[d] / cnts[d]
        var = max((sumsq[d] / cnts[d]) - mu * mu, 0.0)
        sd = math.sqrt(var)
        mean[d] = float(mu)
        std[d] = float(sd if sd > eps else 1.0)
    return GlobalMaskedStandardScaler(mean, std, eps)


# =============================================================================
# Disk layout
# =============================================================================


def save_client_split(
    root: str,
    client_name: str,
    split: str,
    X: np.ndarray,
    M: np.ndarray,
    H: Optional[np.ndarray] = None,
    M_nat: Optional[np.ndarray] = None,
) -> None:
    X = np.asarray(X, dtype=np.float32)
    M = np.asarray(M, dtype=np.float32)
    if X.shape != M.shape:
        raise ValueError(f"X/M shape mismatch: X{X.shape} vs M{M.shape}")
    if H is None:
        H = np.zeros_like(M, dtype=np.float32)
    else:
        H = np.asarray(H, dtype=np.float32)
        if H.shape != M.shape:
            raise ValueError(f"H/M shape mismatch: H{H.shape} vs M{M.shape}")
    Mbin = (M > 0.5).astype(np.float32)
    Hbin = (H > 0.5).astype(np.float32)
    if np.any((Hbin > 0.5) & (Mbin <= 0.5)):
        raise RuntimeError("H must be subset of M")
    M_in = (Mbin * (1.0 - Hbin)).astype(np.float32)

    sp = os.path.join(root, "clients", client_name, split)
    ensure_dir(sp)
    np.save(os.path.join(sp, "X.npy"), X.astype(np.float32))
    np.save(os.path.join(sp, "M.npy"), Mbin.astype(np.float32))
    np.save(os.path.join(sp, "H.npy"), Hbin.astype(np.float32))
    np.save(os.path.join(sp, "I_eval.npy"), Hbin.astype(np.float32))
    np.save(os.path.join(sp, "M_in.npy"), M_in.astype(np.float32))
    if M_nat is not None:
        np.save(os.path.join(sp, "M_nat.npy"), (np.asarray(M_nat) > 0.5).astype(np.float32))


def save_root_manifest(root: str, manifest: Dict[str, object]) -> None:
    ensure_dir(root)
    with open(os.path.join(root, "_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def save_root_meta(root: str, meta: Dict[str, object]) -> None:
    ensure_dir(root)
    with open(os.path.join(root, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


@dataclass
class ClientSplitsRaw:
    name: str
    id: int
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray


@dataclass
class ClientSplitsBase:
    name: str
    id: int
    X_train: np.ndarray
    Mnat_train: np.ndarray
    X_val: np.ndarray
    Mnat_val: np.ndarray
    X_test: np.ndarray
    Mnat_test: np.ndarray


# =============================================================================
# Splitting helpers
# =============================================================================


def split_time_into_clients(T: int, num_clients: int) -> List[Tuple[int, int]]:
    base = T // num_clients
    extra = T % num_clients
    bounds: List[Tuple[int, int]] = []
    s = 0
    for k in range(num_clients):
        length = base + (1 if k < extra else 0)
        e = s + length
        bounds.append((s, e))
        s = e
    return bounds


def tvt_split_indices(n: int, train_ratio: float, val_ratio: float) -> Tuple[int, int]:
    if n < 3:
        raise ValueError(f"Sequence too short to split: n={n}")
    tr = max(int(np.floor(n * float(train_ratio))), 1)
    va = max(int(np.floor(n * float(val_ratio))), 1)
    if tr + va >= n:
        va = max(1, n - tr - 1)
    tr_end = tr
    va_end = min(tr + va, n - 1)
    return tr_end, va_end


def split_df_by_months(
    df: pd.DataFrame,
    split_policy: str,
    val_months: int,
    test_months: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, List[str]]]:
    months = df.index.to_period("M")
    uniq = months.unique().sort_values()
    need = val_months + test_months + 1
    if uniq.size < need:
        raise ValueError(f"Need at least {need} months, got {uniq.size}")
    if split_policy == "latest_test":
        test_m = uniq[-test_months:]
        val_m = uniq[-(test_months + val_months):-test_months]
        train_m = uniq[:-(test_months + val_months)]
    elif split_policy == "earliest_test":
        test_m = uniq[:test_months]
        val_m = uniq[test_months:test_months + val_months]
        train_m = uniq[test_months + val_months:]
    else:
        raise ValueError(f"Unknown split_policy={split_policy}")
    te = df[months.isin(test_m)]
    va = df[months.isin(val_m)]
    tr = df[months.isin(train_m)]
    if len(tr) == 0 or len(va) == 0 or len(te) == 0:
        raise RuntimeError("Empty split after month-based partitioning")
    info = {
        "train_months": [str(m) for m in train_m],
        "val_months": [str(m) for m in val_m],
        "test_months": [str(m) for m in test_m],
    }
    return tr, va, te, info


# =============================================================================
# Dataset loaders
# =============================================================================


def load_ettm1_csv(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    if "date" not in df.columns:
        raise ValueError("ETTm1.csv must contain date column")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"], keep="first")
    df = df.set_index("date")
    df = df.select_dtypes(include=[np.number])
    if df.shape[1] == 0:
        raise ValueError("ETTm1 contains no numeric columns")
    return df[~df.index.duplicated(keep="first")].sort_index()


def load_and_clean_beijing_station(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    req = ["year", "month", "day", "hour"]
    if not all(c in df.columns for c in req):
        raise ValueError(f"{csv_path} missing one of {req}")
    df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime").drop_duplicates(subset=["datetime"], keep="first")
    df = df.set_index("datetime")
    drop_cols = ["year", "month", "day", "hour", "wd", "No", "station"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    df = df.select_dtypes(include=[np.number])
    df = df[~df.index.duplicated(keep="first")].sort_index()
    full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq="H")
    return df.reindex(full_idx)


OUTCOME_FILES = ["Outcomes-a.txt", "Outcomes-b.txt", "Outcomes-c.txt"]
RAW_SETS = ["set-a", "set-b", "set-c"]


def load_physio_outcomes(root_dir: str) -> pd.Series:
    frames = []
    for fn in OUTCOME_FILES:
        p = os.path.join(root_dir, fn)
        if not os.path.isfile(p):
            continue
        df = pd.read_csv(p, sep=",")
        if not {"RecordID", "In-hospital_death"}.issubset(df.columns):
            raise ValueError(f"{p} missing RecordID/In-hospital_death")
        df = df[["RecordID", "In-hospital_death"]].copy()
        df["RecordID"] = df["RecordID"].astype(int)
        frames.append(df.set_index("RecordID"))
    if not frames:
        raise FileNotFoundError("No Outcomes-*.txt found")
    out = pd.concat(frames, axis=0)
    out = out[~out.index.duplicated(keep="first")]
    return out["In-hospital_death"].astype(int)


def load_physio_patients_48h(raw_root: str) -> Tuple[Dict[int, pd.DataFrame], List[str]]:
    patients: Dict[int, pd.DataFrame] = {}
    all_params: set[str] = set()
    for sub in RAW_SETS:
        dirp = os.path.join(raw_root, sub)
        if not os.path.isdir(dirp):
            continue
        for fn in os.listdir(dirp):
            if not fn.endswith(".txt"):
                continue
            rid = int(fn.replace(".txt", ""))
            df = pd.read_csv(os.path.join(dirp, fn))
            if not {"Time", "Parameter", "Value"}.issubset(df.columns):
                continue
            def to_hour(x: object) -> Optional[int]:
                try:
                    return int(str(x).split(":")[0])
                except Exception:
                    return None
            df["Hour"] = df["Time"].apply(to_hour)
            df = df.dropna(subset=["Hour"])
            df["Hour"] = df["Hour"].astype(int)
            df = df[(df["Hour"] >= 0) & (df["Hour"] <= 47)]
            if len(df) == 0:
                continue
            piv = df.pivot_table(values="Value", index="Hour", columns="Parameter", aggfunc="mean")
            piv = piv.reindex(range(48))
            patients[rid] = piv
            all_params |= set(piv.columns.astype(str).tolist())
    if not patients:
        raise RuntimeError("No PhysioNet patient records parsed")
    return patients, sorted(list(all_params))


def physio_build_matrix(
    patients: Dict[int, pd.DataFrame],
    record_ids: List[int],
    all_params: List[str],
    drop_params: List[str],
) -> np.ndarray:
    keep = [p for p in all_params if p not in set(drop_params)]
    D = len(keep)
    X3 = np.full((len(record_ids), 48, D), np.nan, dtype=np.float32)
    for i, rid in enumerate(record_ids):
        df = patients[rid].reindex(columns=keep)
        X3[i] = df.to_numpy(dtype=np.float32)
    return X3


def stratified_round_robin_assign(ids: List[int], y: np.ndarray, num_clients: int, seed: int) -> Dict[int, List[int]]:
    rng = random.Random(int(seed))
    ids0 = [rid for rid, lab in zip(ids, y) if int(lab) == 0]
    ids1 = [rid for rid, lab in zip(ids, y) if int(lab) == 1]
    rng.shuffle(ids0)
    rng.shuffle(ids1)
    buckets = {cid: [] for cid in range(num_clients)}
    k = 0
    for rid in ids0:
        buckets[k % num_clients].append(rid)
        k += 1
    k = 0
    for rid in ids1:
        buckets[k % num_clients].append(rid)
        k += 1
    for cid in range(num_clients):
        rng.shuffle(buckets[cid])
    return buckets


# =============================================================================
# Base dataset builders
# =============================================================================


def build_base_ettm1(
    raw_csv: str,
    out_base_root: str,
    num_clients: int,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[List[ClientSplitsBase], List[str], Dict[str, object]]:
    df = load_ettm1_csv(raw_csv)
    feats = list(df.columns)
    X_full = df.to_numpy(dtype=np.float32)
    bounds = split_time_into_clients(X_full.shape[0], num_clients)
    raw_clients: List[ClientSplitsRaw] = []
    for cid, (s, e) in enumerate(bounds):
        Xc = X_full[s:e]
        tr_end, va_end = tvt_split_indices(len(Xc), train_ratio, val_ratio)
        raw_clients.append(ClientSplitsRaw(f"ettm1_client_{cid}", cid, Xc[:tr_end], Xc[tr_end:va_end], Xc[va_end:]))
    scaler = fit_global_scaler_from_train([c.X_train for c in raw_clients])
    scaler.save(out_base_root, {"dataset": "ETTm1", "num_clients": num_clients})
    clients: List[ClientSplitsBase] = []
    for c in raw_clients:
        clients.append(
            ClientSplitsBase(
                name=c.name,
                id=c.id,
                X_train=scaler.transform(c.X_train),
                Mnat_train=build_missing_mask(c.X_train),
                X_val=scaler.transform(c.X_val),
                Mnat_val=build_missing_mask(c.X_val),
                X_test=scaler.transform(c.X_test),
                Mnat_test=build_missing_mask(c.X_test),
            )
        )
    meta = {
        "dataset": "ETTm1",
        "created_at": utc_now_iso(),
        "seed": int(seed),
        "num_clients": int(num_clients),
        "client_definition": "time-contiguous segment of the full timeline",
        "split_policy": {"type": "within-client-chronological", "train_ratio": float(train_ratio), "val_ratio": float(val_ratio)},
        "num_features": int(X_full.shape[1]),
        "features": feats,
    }
    return clients, feats, meta


def build_base_beijing_aqi(
    src_dir: str,
    out_base_root: str,
    split_policy: str,
    val_months: int,
    test_months: int,
    seed: int,
) -> Tuple[List[ClientSplitsBase], List[str], Dict[str, object]]:
    files = sorted([os.path.join(src_dir, x) for x in os.listdir(src_dir) if x.endswith(".csv")])
    if len(files) == 0:
        raise FileNotFoundError(f"No Beijing station csv files in {src_dir}")

    station_dfs: Dict[str, pd.DataFrame] = {}
    common_cols: Optional[set[str]] = None
    month_info_any: Optional[Dict[str, List[str]]] = None
    for fp in files:
        name = os.path.splitext(os.path.basename(fp))[0]
        df = load_and_clean_beijing_station(fp)
        station_dfs[name] = df
        cols = set(df.columns.astype(str).tolist())
        common_cols = cols if common_cols is None else (common_cols & cols)
    if common_cols is None or len(common_cols) == 0:
        raise RuntimeError("No common numeric feature across Beijing stations")
    feats = sorted(list(common_cols))

    raw_clients: List[ClientSplitsRaw] = []
    for cid, name in enumerate(sorted(station_dfs.keys())):
        df = station_dfs[name][feats].copy().sort_index()
        tr, va, te, info = split_df_by_months(df, split_policy=split_policy, val_months=val_months, test_months=test_months)
        if month_info_any is None:
            month_info_any = info
        raw_clients.append(ClientSplitsRaw(name, cid, tr.to_numpy(dtype=np.float32), va.to_numpy(dtype=np.float32), te.to_numpy(dtype=np.float32)))

    scaler = fit_global_scaler_from_train([c.X_train for c in raw_clients])
    scaler.save(out_base_root, {"dataset": "Beijing_AQI", "num_clients": len(raw_clients)})
    clients: List[ClientSplitsBase] = []
    for c in raw_clients:
        clients.append(
            ClientSplitsBase(
                name=c.name,
                id=c.id,
                X_train=scaler.transform(c.X_train),
                Mnat_train=build_missing_mask(c.X_train),
                X_val=scaler.transform(c.X_val),
                Mnat_val=build_missing_mask(c.X_val),
                X_test=scaler.transform(c.X_test),
                Mnat_test=build_missing_mask(c.X_test),
            )
        )
    meta = {
        "dataset": "Beijing_AQI",
        "created_at": utc_now_iso(),
        "seed": int(seed),
        "num_clients": int(len(raw_clients)),
        "client_definition": "each client = one monitoring station",
        "split_policy": {"type": "month-based", "policy": split_policy, "val_months": int(val_months), "test_months": int(test_months), "months": month_info_any},
        "num_features": int(len(feats)),
        "features": feats,
    }
    return clients, feats, meta


def build_base_physionet2012(
    physio_root: str,
    out_base_root: str,
    num_clients: int,
    train_frac: float,
    val_frac_in_train: float,
    seed: int,
    stratify_outcome: bool = True,
) -> Tuple[List[ClientSplitsBase], List[str], Dict[str, object]]:
    outcomes = load_physio_outcomes(physio_root)
    patients, all_params = load_physio_patients_48h(physio_root)
    drop_params = ["Age", "Gender", "ICUType", "Height"]
    keep = [p for p in all_params if p not in set(drop_params)]

    common_ids = sorted([rid for rid in patients.keys() if rid in outcomes.index])
    if len(common_ids) == 0:
        raise RuntimeError("No patient has both time-series and outcome")
    labels = outcomes.loc[common_ids].to_numpy(dtype=np.int64)

    if stratify_outcome:
        buckets = stratified_round_robin_assign(common_ids, labels, num_clients, seed)
    else:
        rng = random.Random(int(seed))
        ids = list(common_ids)
        rng.shuffle(ids)
        buckets = {cid: [] for cid in range(num_clients)}
        for i, rid in enumerate(ids):
            buckets[i % num_clients].append(rid)

    raw_clients: List[ClientSplitsRaw] = []
    for cid in range(num_clients):
        rids = list(buckets[cid])
        if len(rids) < 3:
            raise RuntimeError(f"Client {cid} too small: {len(rids)} episodes")
        y = outcomes.loc[rids].to_numpy(dtype=np.int64)
        idx = np.arange(len(rids))
        train_idx, test_idx = train_test_split(
            idx,
            test_size=max(1, int(round(len(idx) * (1.0 - float(train_frac))))),
            random_state=int(seed),
            shuffle=True,
            stratify=y if (len(np.unique(y)) > 1 and len(idx) >= 4) else None,
        )
        y_train = y[train_idx]
        val_size = max(1, int(round(len(train_idx) * float(val_frac_in_train))))
        if val_size >= len(train_idx):
            val_size = max(1, len(train_idx) - 1)
        tr2_idx, val_idx = train_test_split(
            train_idx,
            test_size=val_size,
            random_state=int(seed),
            shuffle=True,
            stratify=y_train if (len(np.unique(y_train)) > 1 and len(train_idx) >= 4) else None,
        )
        train_ids = [rids[i] for i in tr2_idx]
        val_ids = [rids[i] for i in val_idx]
        test_ids = [rids[i] for i in test_idx]
        X_train = physio_build_matrix(patients, train_ids, all_params, drop_params).reshape(len(train_ids) * 48, len(keep))
        X_val = physio_build_matrix(patients, val_ids, all_params, drop_params).reshape(len(val_ids) * 48, len(keep))
        X_test = physio_build_matrix(patients, test_ids, all_params, drop_params).reshape(len(test_ids) * 48, len(keep))
        raw_clients.append(ClientSplitsRaw(f"physio_client_{cid}", cid, X_train, X_val, X_test))

    scaler = fit_global_scaler_from_train([c.X_train for c in raw_clients])
    scaler.save(out_base_root, {"dataset": "PhysioNet2012", "num_clients": num_clients})
    clients: List[ClientSplitsBase] = []
    for c in raw_clients:
        clients.append(
            ClientSplitsBase(
                name=c.name,
                id=c.id,
                X_train=scaler.transform(c.X_train),
                Mnat_train=build_missing_mask(c.X_train),
                X_val=scaler.transform(c.X_val),
                Mnat_val=build_missing_mask(c.X_val),
                X_test=scaler.transform(c.X_test),
                Mnat_test=build_missing_mask(c.X_test),
            )
        )
    meta = {
        "dataset": "PhysioNet2012",
        "created_at": utc_now_iso(),
        "seed": int(seed),
        "num_clients": int(num_clients),
        "client_definition": "stratified round-robin assignment of patient episodes",
        "split_policy": {"type": "within-client-patient-level", "train_frac": float(train_frac), "val_frac_in_train": float(val_frac_in_train), "stratify_outcome": bool(stratify_outcome)},
        "num_features": int(len(keep)),
        "features": keep,
    }
    return clients, keep, meta


def write_base_dataset(root_base: str, dataset_meta: Dict[str, object], clients: List[ClientSplitsBase]) -> None:
    if os.path.isdir(root_base):
        shutil.rmtree(root_base)
    ensure_dir(root_base)
    for c in clients:
        save_client_split(root_base, c.name, "train", c.X_train, c.Mnat_train, H=np.zeros_like(c.Mnat_train), M_nat=c.Mnat_train)
        save_client_split(root_base, c.name, "val", c.X_val, c.Mnat_val, H=np.zeros_like(c.Mnat_val), M_nat=c.Mnat_val)
        save_client_split(root_base, c.name, "test", c.X_test, c.Mnat_test, H=np.zeros_like(c.Mnat_test), M_nat=c.Mnat_test)
    save_root_meta(root_base, dataset_meta)
    save_root_manifest(
        root_base,
        {
            "dataset": dataset_meta["dataset"],
            "created_at": utc_now_iso(),
            "layout": "federated_base_v2",
            "clients": [{"id": int(c.id), "name": c.name} for c in clients],
        },
    )


# =============================================================================
# Structural missingness generation
# =============================================================================


def gen_structural_mask_mcar(M_nat: np.ndarray, miss_rate: float, rng: np.random.Generator) -> np.ndarray:
    M_nat = (np.asarray(M_nat, dtype=np.float32) > 0.5)
    obs = np.flatnonzero(M_nat.reshape(-1))
    out = M_nat.copy().reshape(-1)
    if obs.size == 0 or miss_rate <= 0:
        return out.reshape(M_nat.shape).astype(np.float32)
    k = int(round(obs.size * float(miss_rate)))
    k = min(max(k, 0), int(obs.size))
    if k > 0:
        drop = rng.choice(obs, size=k, replace=False)
        out[drop] = False
    return out.reshape(M_nat.shape).astype(np.float32)


def window_starts(T: int, window: int, stride: int) -> np.ndarray:
    if T < window:
        return np.zeros((0,), dtype=np.int64)
    return np.arange(0, T - window + 1, stride, dtype=np.int64)


def _count_low_obs_windows(M: np.ndarray, window: int, stride: int, threshold: int) -> Tuple[int, List[int]]:
    starts = window_starts(M.shape[0], window, stride)
    bad: List[int] = []
    for i, st in enumerate(starts):
        obs = int(M[st:st + window].sum())
        if obs < threshold:
            bad.append(i)
    return len(bad), bad


def _enforce_window_trainable(M: np.ndarray, M_nat: np.ndarray, window: int, stride: int, min_obs_per_window: int) -> np.ndarray:
    M = (np.asarray(M) > 0.5)
    M_nat = (np.asarray(M_nat) > 0.5)
    starts = window_starts(M.shape[0], window, stride)
    if starts.size == 0 or min_obs_per_window <= 0:
        return M.astype(np.float32)
    for st in starts:
        view = M[st:st + window]
        if int(view.sum()) >= min_obs_per_window:
            continue
        nat_view = M_nat[st:st + window]
        cand = np.flatnonzero((nat_view.reshape(-1)) & (~view.reshape(-1)))
        if cand.size == 0:
            continue
        need = min_obs_per_window - int(view.sum())
        take = cand[:need]
        flat = view.reshape(-1)
        flat[take] = True
        M[st:st + window] = flat.reshape(view.shape)
    return M.astype(np.float32)


def gen_structural_mask_hetero_v2(
    M_nat: np.ndarray,
    *,
    window: int,
    stride: int,
    rng: np.random.Generator,
    target_missing_rate_p: float,
    p_min: float,
    p_max: float,
    block_mean_len: float = 8.0,
    block_max_len_mult: float = 4.0,
    min_obs_per_window_d1: int = 4,
    min_obs_per_window_multi: int = 8,
) -> Tuple[np.ndarray, Dict[str, float]]:
    M_nat = (np.asarray(M_nat, dtype=np.float32) > 0.5)
    T, D = M_nat.shape
    target = float(np.clip(target_missing_rate_p, p_min, p_max))
    M = M_nat.copy()

    obs_total = int(M_nat.sum())
    if obs_total == 0:
        return M.astype(np.float32), {"target_missing_rate": target, "achieved_missing_rate": 0.0, "num_blocks": 0}

    max_block_len = max(1, int(round(block_mean_len * block_max_len_mult)))
    num_blocks = 0
    attempts = 0
    while attempts < 5000:
        current_missing = 1.0 - float(M[M_nat].mean())
        if current_missing >= target:
            break
        attempts += 1
        d = int(rng.integers(0, D))
        start = int(rng.integers(0, max(1, T)))
        length = int(min(max_block_len, max(1, rng.poisson(block_mean_len) + 1)))
        end = min(T, start + length)
        M[start:end, d] = False
        num_blocks += 1

    min_obs = int(min_obs_per_window_d1 if D == 1 else min_obs_per_window_multi)
    M = _enforce_window_trainable(M.astype(np.float32), M_nat.astype(np.float32), window, stride, min_obs)
    achieved = 1.0 - float(M[M_nat].mean()) if int(M_nat.sum()) > 0 else 0.0
    meta = {
        "target_missing_rate": float(target),
        "achieved_missing_rate": float(achieved),
        "num_blocks": float(num_blocks),
    }
    return M.astype(np.float32), meta


# =============================================================================
# Variant generation
# =============================================================================


def build_variant_from_base(
    base_root: str,
    variant_root: str,
    dataset_tag: str,
    variant: str,
    holdout_ratio: float,
    holdout_mode: str,
    seed: int,
    window: int,
    stride: int,
    hetero_p_min: float,
    hetero_p_max: float,
    hetero_p_alpha: float,
    hetero_p_beta: float,
    hetero_block_mean_len: float,
    hetero_block_max_len_mult: float,
    hetero_min_obs_per_window_d1: int,
    hetero_min_obs_per_window_multi: int,
) -> Dict[str, object]:
    if os.path.isdir(variant_root):
        shutil.rmtree(variant_root)
    ensure_dir(variant_root)

    base_meta = _read_json(os.path.join(base_root, "meta.json"))
    manifest = _read_json(os.path.join(base_root, "_manifest.json"))
    clients = manifest["clients"]

    variant_spec = resolve_variant_spec(
        variant,
        hetero_p_min=hetero_p_min,
        hetero_p_max=hetero_p_max,
        hetero_p_alpha=hetero_p_alpha,
        hetero_p_beta=hetero_p_beta,
        hetero_block_mean_len=hetero_block_mean_len,
        hetero_block_max_len_mult=hetero_block_max_len_mult,
        hetero_min_obs_per_window_d1=hetero_min_obs_per_window_d1,
        hetero_min_obs_per_window_multi=hetero_min_obs_per_window_multi,
    )

    overlay_meta_by_client: Dict[str, Dict[str, object]] = {}
    client_summaries: List[Dict[str, object]] = []
    hetero_p_by_client: Dict[str, float] = {}

    if variant_spec["kind"] == "hetero":
        if str(variant_spec.get("mode", "")) == "fixed_schedule":
            targets = build_fixed_hetero_targets(
                num_clients=len(clients),
                low=float(variant_spec["low"]),
                high=float(variant_spec["high"]),
                dataset_tag=dataset_tag,
                variant=str(variant_spec["name"]),
                seed=int(seed),
                p_min=float(variant_spec["p_min"]),
                p_max=float(variant_spec["p_max"]),
            )
            for c, p_client in zip(clients, targets):
                hetero_p_by_client[str(c["name"])] = float(p_client)
        else:
            rng_p = np.random.default_rng(stable_uint32(dataset_tag, variant_spec["name"], seed, "hetero_p"))
            for c in clients:
                hetero_p_by_client[str(c["name"])] = float(
                    np.clip(
                        rng_p.beta(float(variant_spec["alpha"]), float(variant_spec["beta"])),
                        float(variant_spec["p_min"]),
                        float(variant_spec["p_max"]),
                    )
                )

    for c in clients:
        cname = str(c["name"])
        cpath = os.path.join(base_root, "clients", cname)
        holdout_stats: Dict[str, Dict[str, float]] = {}
        missing_struct_train = None
        for split in ("train", "val", "test"):
            sp = os.path.join(cpath, split)
            X = np.load(os.path.join(sp, "X.npy")).astype(np.float32)
            M_nat = np.load(os.path.join(sp, "M_nat.npy")).astype(np.float32)

            if variant_spec["kind"] == "natural":
                M = M_nat.copy()
                overlay_meta = {"type": "natural"}

            elif variant_spec["kind"] == "mcar":
                miss_rate = float(variant_spec["miss_rate"])
                rng_m = np.random.default_rng(stable_uint32(dataset_tag, variant_spec["name"], cname, split, seed))
                M = gen_structural_mask_mcar(M_nat, miss_rate=miss_rate, rng=rng_m)
                overlay_meta = {
                    "type": "mcar",
                    "target_missing_rate": float(miss_rate),
                }

            elif variant_spec["kind"] == "hetero":
                p_client = hetero_p_by_client[cname]
                rng_h = np.random.default_rng(stable_uint32(dataset_tag, variant_spec["name"], cname, split, seed))
                M, overlay_meta = gen_structural_mask_hetero_v2(
                    M_nat,
                    window=int(window),
                    stride=int(stride),
                    rng=rng_h,
                    target_missing_rate_p=p_client,
                    p_min=float(variant_spec["p_min"]),
                    p_max=float(variant_spec["p_max"]),
                    block_mean_len=float(variant_spec["block_mean_len"]),
                    block_max_len_mult=float(variant_spec["block_max_len_mult"]),
                    min_obs_per_window_d1=int(variant_spec["min_obs_per_window_d1"]),
                    min_obs_per_window_multi=int(variant_spec["min_obs_per_window_multi"]),
                )
                if str(variant_spec.get("mode", "")) == "fixed_schedule":
                    overlay_meta = {
                        **overlay_meta,
                        "preset": str(variant_spec["preset"]),
                        "mode": "fixed_schedule",
                        "low": float(variant_spec["low"]),
                        "high": float(variant_spec["high"]),
                    }
                else:
                    overlay_meta = {
                        **overlay_meta,
                        "preset": str(variant_spec["preset"]),
                        "mode": "beta_sampling",
                        "alpha": float(variant_spec["alpha"]),
                        "beta": float(variant_spec["beta"]),
                    }

            else:
                raise ValueError(f"Unknown variant spec: {variant_spec}")

            overlay_meta_by_client[cname] = overlay_meta
            if split == "train":
                H = np.zeros_like(M, dtype=np.float32)
                missing_struct_train = describe_mask(M)
            else:
                rng_hold = np.random.default_rng(stable_uint32("holdout", dataset_tag, variant, cname, split, seed))
                H = make_holdout_mask(M, ratio=float(holdout_ratio), rng=rng_hold, mode=holdout_mode, min_per_col=0)
            holdout_stats[split] = describe_holdout_rate(M, H)
            save_client_split(variant_root, cname, split, X=X, M=M, H=H, M_nat=M_nat)

        client_summaries.append(
            {
                "id": int(c["id"]),
                "name": cname,
                "hetero_target_p": None if cname not in hetero_p_by_client else float(hetero_p_by_client[cname]),
                "missing_nat_train": describe_mask(np.load(os.path.join(cpath, "train", "M_nat.npy")).astype(np.float32)),
                "missing_struct_train": missing_struct_train,
                "holdout": holdout_stats,
            }
        )

    hetero_p_stats = None
    if hetero_p_by_client:
        ps = np.asarray(list(hetero_p_by_client.values()), dtype=np.float32)
        hetero_p_stats = {
            "min": float(np.min(ps)),
            "max": float(np.max(ps)),
            "mean": float(np.mean(ps)),
            "std": float(np.std(ps)),
            "q10": float(np.quantile(ps, 0.10)),
            "q50": float(np.quantile(ps, 0.50)),
            "q90": float(np.quantile(ps, 0.90)),
        }

    meta = {
        "dataset": dataset_tag,
        "created_at": utc_now_iso(),
        "seed": int(seed),
        "variant": variant,
        "window": int(window),
        "stride": int(stride),
        "holdout_ratio": float(holdout_ratio),
        "holdout_mode": str(holdout_mode),
        "overlay": {
            "type": "natural" if variant_spec["kind"] == "natural" else (
                "mcar" if variant_spec["kind"] == "mcar" else "hetero_v2"),
            "variant_spec": variant_spec,
            "hetero_p_stats": hetero_p_stats,
        },
        "mask_convention": {"M": "observed mask (1=observed)", "H": "holdout mask (1=eval point)", "M_in": "M*(1-H)"},
        "notes": "Variant data with evaluation holdout on val/test and canonical input-visible mask M_in.",
    }
    manifest = {
        "dataset": dataset_tag,
        "created_at": utc_now_iso(),
        "layout": "federated_variant_v2",
        "variant": variant,
        "seed": int(seed),
        "window": int(window),
        "stride": int(stride),
        "holdout_ratio": float(holdout_ratio),
        "holdout_mode": str(holdout_mode),
        "clients": client_summaries,
        "overlay_meta_by_client": overlay_meta_by_client,
    }
    save_root_meta(variant_root, meta)
    save_root_manifest(variant_root, manifest)
    return meta


# =============================================================================
# Regime descriptors / fed masks
# =============================================================================


def longest_zero_run_1d(x01: np.ndarray) -> int:
    x01 = np.asarray(x01, dtype=np.float32).reshape(-1)
    best = 0
    cur = 0
    for v in x01:
        if float(v) <= 0.5:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def compute_s_g_for_window(Mw: np.ndarray, q_g: float = 0.8) -> Tuple[float, float]:
    Mw = np.asarray(Mw, dtype=np.float32)
    L, D = Mw.shape
    s = 1.0 - float(np.mean(Mw))
    gaps = np.zeros((D,), dtype=np.float32)
    for d in range(D):
        gaps[d] = float(longest_zero_run_1d(Mw[:, d]))
    g = float(np.quantile(gaps, q_g)) if D > 0 else 0.0
    return float(s), float(g)


def discretize_bins(values: np.ndarray, edges: List[float]) -> np.ndarray:
    v = np.asarray(values, dtype=np.float32)
    bins = np.zeros_like(v, dtype=np.int64)
    for thr in edges:
        bins += (v > float(thr)).astype(np.int64)
    return bins


def gen_fed_masks_and_regimes(
    variant_root: str,
    out_fed_masks_root: str,
    dataset_tag: str,
    variant: str,
    window: int,
    stride: int,
    q_g: float,
    R_s: int,
    R_g: int,
    seed: int,
) -> Dict[str, object]:
    if os.path.isdir(out_fed_masks_root):
        shutil.rmtree(out_fed_masks_root)
    ensure_dir(out_fed_masks_root)

    clients_dir = os.path.join(variant_root, "clients")
    if not os.path.isdir(clients_dir):
        raise FileNotFoundError(f"Missing clients dir: {clients_dir}")

    s_all: List[float] = []
    gnorm_all: List[float] = []
    for cname in sorted(os.listdir(clients_dir)):
        Mp = os.path.join(clients_dir, cname, "train", "M.npy")
        M = np.load(Mp).astype(np.float32)
        starts = window_starts(M.shape[0], window, stride)
        for st in starts:
            Mw = M[int(st):int(st + window)]
            s, g = compute_s_g_for_window(Mw, q_g=q_g)
            s_all.append(s)
            gnorm_all.append(g / float(window))

    if len(s_all) == 0:
        raise RuntimeError("No train windows found for regime generation")
    s_all_np = np.asarray(s_all, dtype=np.float32)
    gnorm_all_np = np.asarray(gnorm_all, dtype=np.float32)
    s_edges = [float(np.quantile(s_all_np, i / R_s)) for i in range(1, R_s)]
    g_edges = [float(np.quantile(gnorm_all_np, i / R_g)) for i in range(1, R_g)]

    for cname in sorted(os.listdir(clients_dir)):
        c_out = os.path.join(out_fed_masks_root, cname)
        ensure_dir(c_out)
        for split in ("train", "val", "test"):
            sp = os.path.join(clients_dir, cname, split)
            M = np.load(os.path.join(sp, "M.npy")).astype(np.float32)
            H = np.load(os.path.join(sp, "H.npy")).astype(np.float32) if os.path.isfile(os.path.join(sp, "H.npy")) else np.zeros_like(M, dtype=np.float32)
            starts = window_starts(M.shape[0], window, stride)
            W = int(starts.shape[0])
            if W == 0:
                np.save(os.path.join(c_out, f"{split}_starts.npy"), np.zeros((0,), dtype=np.int64))
                np.save(os.path.join(c_out, f"{split}_s.npy"), np.zeros((0,), dtype=np.float32))
                np.save(os.path.join(c_out, f"{split}_g.npy"), np.zeros((0,), dtype=np.float32))
                np.save(os.path.join(c_out, f"{split}_regime.npy"), np.zeros((0,), dtype=np.int64))
                continue
            s_arr = np.zeros((W,), dtype=np.float32)
            g_arr = np.zeros((W,), dtype=np.float32)
            for i, st in enumerate(starts):
                Mw = M[int(st):int(st + window)]
                s, g = compute_s_g_for_window(Mw, q_g=q_g)
                s_arr[i] = s
                g_arr[i] = g
            s_bin = discretize_bins(s_arr, s_edges)
            g_bin = discretize_bins(g_arr / float(window), g_edges)
            regime = (s_bin * int(R_g) + g_bin).astype(np.int64)
            np.save(os.path.join(c_out, f"{split}_starts.npy"), starts.astype(np.int64))
            np.save(os.path.join(c_out, f"{split}_s.npy"), s_arr.astype(np.float32))
            np.save(os.path.join(c_out, f"{split}_g.npy"), g_arr.astype(np.float32))
            np.save(os.path.join(c_out, f"{split}_regime.npy"), regime.astype(np.int64))
            np.save(os.path.join(c_out, f"{split}_M_eff.npy"), M.astype(np.float32))
            np.save(os.path.join(c_out, f"{split}_H_eff.npy"), H.astype(np.float32))

    meta = {
        "dataset": dataset_tag,
        "variant": variant,
        "created_at": utc_now_iso(),
        "seed": int(seed),
        "window": int(window),
        "stride": int(stride),
        "q_g": float(q_g),
        "R_s": int(R_s),
        "R_g": int(R_g),
        "R": int(R_s * R_g),
        "binning": {"s_edges": s_edges, "g_edges_normed_by_L": g_edges, "method": "quantiles on TRAIN windows"},
        "notes": "Regime metadata for CoRA: window starts, s/g descriptors, and regime ids.",
    }
    with open(os.path.join(out_fed_masks_root, "_regime_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


# =============================================================================
# Main orchestration
# =============================================================================

DEFAULT_WS = {
    "ETTm1": (96, 48),
    "PhysioNet2012": (48, 48),
    "Beijing_AQI": (48, 24),
}


HETERO_PRESETS: Dict[str, Dict[str, float | str]] = {
    "hetero_h1": {"mode": "fixed_schedule", "low": 0.20, "high": 0.37, "label": "weak"},
    "hetero_h2": {"mode": "fixed_schedule", "low": 0.15, "high": 0.42, "label": "moderate"},
    "hetero_h3": {"mode": "fixed_schedule", "low": 0.10, "high": 0.47, "label": "medium_default"},
    "hetero_h4": {"mode": "fixed_schedule", "low": 0.05, "high": 0.52, "label": "strong"},
}

"""
HETERO_PRESETS: Dict[str, Dict[str, float | str]] = {
    "hetero_h1": {"alpha": 8.0,  "beta": 20.0, "label": "weak"},
    "hetero_h2": {"alpha": 4.0,  "beta": 10.0, "label": "moderate"},
    "hetero_h3": {"alpha": 2.0,  "beta": 5.0,  "label": "medium_default"},
    "hetero_h4": {"alpha": 1.43, "beta": 3.57, "label": "strong"},
}
"""

# 新增
def build_fixed_hetero_targets(
    num_clients: int,
    low: float,
    high: float,
    *,
    dataset_tag: str,
    variant: str,
    seed: int,
    p_min: float,
    p_max: float,
) -> List[float]:
    if num_clients <= 0:
        return []
    vals = np.linspace(float(low), float(high), int(num_clients), dtype=np.float32)
    vals = np.clip(vals, float(p_min), float(p_max))
    rng = np.random.default_rng(stable_uint32(dataset_tag, variant, seed, "hetero_fixed_shuffle"))
    perm = rng.permutation(int(num_clients))
    vals = vals[perm]
    return [float(x) for x in vals]


def resolve_variant_spec(
    variant: str,
    *,
    hetero_p_min: float,
    hetero_p_max: float,
    hetero_p_alpha: float,
    hetero_p_beta: float,
    hetero_block_mean_len: float,
    hetero_block_max_len_mult: float,
    hetero_min_obs_per_window_d1: int,
    hetero_min_obs_per_window_multi: int,
) -> Dict[str, object]:
    v = str(variant)

    if v in ("natural", "orig", "base"):
        return {"kind": "natural", "name": v}

    if v.startswith("mcar_p"):
        suffix = v[len("mcar_p"):]
        try:
            miss_rate = float(suffix) / 100.0
        except Exception as e:
            raise ValueError(f"Invalid MCAR variant name: {v}") from e
        if not (0.0 <= miss_rate < 1.0):
            raise ValueError(f"MCAR miss rate must be in [0,1): {v}")
        return {
            "kind": "mcar",
            "name": v,
            "miss_rate": float(miss_rate),
        }

    if v == "hetero":
        return {
            "kind": "hetero",
            "name": v,
            "preset": "cli_default",
            "alpha": float(hetero_p_alpha),
            "beta": float(hetero_p_beta),
            "p_min": float(hetero_p_min),
            "p_max": float(hetero_p_max),
            "block_mean_len": float(hetero_block_mean_len),
            "block_max_len_mult": float(hetero_block_max_len_mult),
            "min_obs_per_window_d1": int(hetero_min_obs_per_window_d1),
            "min_obs_per_window_multi": int(hetero_min_obs_per_window_multi),
        }

    if v in HETERO_PRESETS:
        preset = HETERO_PRESETS[v]
        return {
            "kind": "hetero",
            "name": v,
            "preset": str(preset["label"]),
            "mode": str(preset["mode"]),
            "low": float(preset["low"]),
            "high": float(preset["high"]),
            "p_min": float(hetero_p_min),
            "p_max": float(hetero_p_max),
            "block_mean_len": float(hetero_block_mean_len),
            "block_max_len_mult": float(hetero_block_max_len_mult),
            "min_obs_per_window_d1": int(hetero_min_obs_per_window_d1),
            "min_obs_per_window_multi": int(hetero_min_obs_per_window_multi),
        }

    raise ValueError(
        f"Unknown variant: {v}. "
        f"Supported: natural/orig/base, mcar_pXX, hetero, {list(HETERO_PRESETS.keys())}"
    )

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser("CoRA preprocess pipeline")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--dataset_root", type=str, default="dataset")
    parser.add_argument("--datasets", nargs="+", default=["ettm1", "beijing", "physio"], choices=["ettm1", "beijing", "physio"])
    parser.add_argument("--variants", nargs="+", default=["natural", "hetero", "mcar_p10"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout_ratio", type=float, default=0.10)
    parser.add_argument("--holdout_mode", type=str, default="global", choices=["global", "per_col"])

    parser.add_argument("--ettm1_csv", type=str, default=os.path.join("dataset", "electricity_transformer_temperature", "ETTm1.csv"))
    parser.add_argument("--ettm1_clients", type=int, default=20)
    parser.add_argument("--ettm1_train_ratio", type=float, default=16.0 / 24.0)
    parser.add_argument("--ettm1_val_ratio", type=float, default=4.0 / 24.0)

    parser.add_argument("--beijing_dir", type=str, default=os.path.join("dataset", "beijing_multisite_air_quality", "PRSA_Data_20130301-20170228"))
    parser.add_argument("--beijing_split_policy", type=str, default="latest_test", choices=["latest_test", "earliest_test"])
    parser.add_argument("--beijing_val_months", type=int, default=10)
    parser.add_argument("--beijing_test_months", type=int, default=10)

    parser.add_argument("--physio_root", type=str, default=os.path.join("dataset", "physionet_2012"))
    parser.add_argument("--physio_clients", type=int, default=20)
    parser.add_argument("--physio_train_frac", type=float, default=0.8)
    parser.add_argument("--physio_val_frac_in_train", type=float, default=0.2)
    parser.add_argument("--physio_stratify_outcome", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--q_g", type=float, default=0.8)
    parser.add_argument("--R_s", type=int, default=4)
    parser.add_argument("--R_g", type=int, default=3)

    parser.add_argument("--hetero_p_min", type=float, default=0.05)
    parser.add_argument("--hetero_p_max", type=float, default=0.70)
    parser.add_argument("--hetero_p_alpha", type=float, default=2.0)
    parser.add_argument("--hetero_p_beta", type=float, default=5.0)
    parser.add_argument("--hetero_block_mean_len", type=float, default=8.0)
    parser.add_argument("--hetero_block_max_len_mult", type=float, default=4.0)
    parser.add_argument("--hetero_min_obs_per_window_d1", type=int, default=4)
    parser.add_argument("--hetero_min_obs_per_window_multi", type=int, default=8)

    args = parser.parse_args(argv)
    set_seed(int(args.seed))

    for ds_key in args.datasets:
        if ds_key == "ettm1":
            dataset_tag = "ETTm1"
            base_root = os.path.join(args.data_root, f"{dataset_tag}_PFed_base")
            clients, feats, meta = build_base_ettm1(args.ettm1_csv, base_root, args.ettm1_clients, args.ettm1_train_ratio, args.ettm1_val_ratio, args.seed)
            write_base_dataset(base_root, meta, clients)
        elif ds_key == "beijing":
            dataset_tag = "Beijing_AQI"
            base_root = os.path.join(args.data_root, f"{dataset_tag}_PFed_base")
            clients, feats, meta = build_base_beijing_aqi(args.beijing_dir, base_root, args.beijing_split_policy, args.beijing_val_months, args.beijing_test_months, args.seed)
            write_base_dataset(base_root, meta, clients)
        elif ds_key == "physio":
            dataset_tag = "PhysioNet2012"
            base_root = os.path.join(args.data_root, f"{dataset_tag}_PFed_base")
            clients, feats, meta = build_base_physionet2012(args.physio_root, base_root, args.physio_clients, args.physio_train_frac, args.physio_val_frac_in_train, args.seed, args.physio_stratify_outcome)
            write_base_dataset(base_root, meta, clients)
        else:
            raise ValueError(ds_key)

        window, stride = DEFAULT_WS[dataset_tag]
        for variant in args.variants:
            variant_root = os.path.join(args.data_root, f"{dataset_tag}_PFed_{variant}")
            fed_root = os.path.join(args.data_root, f"{dataset_tag}_PFed_fed_masks_{variant}")
            build_variant_from_base(
                base_root=base_root,
                variant_root=variant_root,
                dataset_tag=dataset_tag,
                variant=variant,
                holdout_ratio=args.holdout_ratio,
                holdout_mode=args.holdout_mode,
                seed=args.seed,
                window=window,
                stride=stride,
                hetero_p_min=args.hetero_p_min,
                hetero_p_max=args.hetero_p_max,
                hetero_p_alpha=args.hetero_p_alpha,
                hetero_p_beta=args.hetero_p_beta,
                hetero_block_mean_len=args.hetero_block_mean_len,
                hetero_block_max_len_mult=args.hetero_block_max_len_mult,
                hetero_min_obs_per_window_d1=args.hetero_min_obs_per_window_d1,
                hetero_min_obs_per_window_multi=args.hetero_min_obs_per_window_multi,
            )
            gen_fed_masks_and_regimes(
                variant_root=variant_root,
                out_fed_masks_root=fed_root,
                dataset_tag=dataset_tag,
                variant=variant,
                window=window,
                stride=stride,
                q_g=args.q_g,
                R_s=args.R_s,
                R_g=args.R_g,
                seed=args.seed,
            )
            print(f"[OK] dataset={dataset_tag} variant={variant} -> {variant_root} | {fed_root}")


if __name__ == "__main__":
    main()
