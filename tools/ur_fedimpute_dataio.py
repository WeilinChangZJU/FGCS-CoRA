from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# =============================================================================
# Dataset / layout conventions
# =============================================================================

DATASET_CANONICAL = {
    "ettm1": "ETTm1",
    "physio": "PhysioNet2012",
    "physionet2012": "PhysioNet2012",
    "beijing": "Beijing_AQI",
    "beijing_aqi": "Beijing_AQI",
}

DEFAULT_WINDOW_STRIDE = {
    "ETTm1": (96, 48),
    "PhysioNet2012": (48, 48),
    "Beijing_AQI": (48, 24),
}


# =============================================================================
# Helpers
# =============================================================================


def _canonical_dataset_name(name: str) -> str:
    raw = str(name).strip()
    if raw in DEFAULT_WINDOW_STRIDE:
        return raw
    key = raw.lower()
    if key in DATASET_CANONICAL:
        return DATASET_CANONICAL[key]
    raise ValueError(f"Unknown dataset name: {name}")


def _read_json(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _pick_existing(root: str, candidates: List[str]) -> str:
    for fn in candidates:
        p = os.path.join(root, fn)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"No candidate file found under {root}: {candidates}")


def resolve_variant_root(data_root: str, dataset: str, variant: str) -> str:
    dataset = _canonical_dataset_name(dataset)
    variant = str(variant).strip()
    if variant == "":
        raise ValueError("variant must not be empty")
    cand = os.path.join(data_root, f"{dataset}_PFed_{variant}")
    if not os.path.isdir(cand):
        raise FileNotFoundError(f"Variant root not found: {cand}")
    _ = _pick_existing(cand, ["meta.json", "_meta.json"])
    mani = os.path.join(cand, "_manifest.json")
    if not os.path.isfile(mani):
        raise FileNotFoundError(f"Missing _manifest.json: {mani}")
    return cand


def resolve_fed_masks_root(data_root: str, dataset: str, variant: str) -> str:
    dataset = _canonical_dataset_name(dataset)
    variant = str(variant).strip()
    if variant == "":
        raise ValueError("variant must not be empty")
    cand = os.path.join(data_root, f"{dataset}_PFed_fed_masks_{variant}")
    if not os.path.isdir(cand):
        raise FileNotFoundError(f"Fed-mask root not found: {cand}")
    meta = os.path.join(cand, "_regime_meta.json")
    if not os.path.isfile(meta):
        raise FileNotFoundError(f"Missing _regime_meta.json: {meta}")
    return cand


@dataclass(frozen=True)
class ClientInfo:
    client_id: int
    client_name: str


@dataclass(frozen=True)
class VariantMeta:
    dataset: str
    variant_name: str
    window: int
    stride: int
    holdout_ratio: float


@dataclass(frozen=True)
class FedMaskIndex:
    starts: np.ndarray
    s: np.ndarray
    g: np.ndarray
    regime_id: np.ndarray


# =============================================================================
# Metadata loading
# =============================================================================


def load_variant_meta(variant_root: str) -> VariantMeta:
    meta_path = _pick_existing(variant_root, ["meta.json", "_meta.json"])
    meta = _read_json(meta_path)
    dataset = _canonical_dataset_name(str(meta.get("dataset", "")))
    variant_name = str(meta.get("variant", meta.get("variant_name", ""))).strip()
    if variant_name == "":
        variant_name = os.path.basename(variant_root).split("_PFed_")[-1]
    window, stride = DEFAULT_WINDOW_STRIDE[dataset]
    window = int(meta.get("window", window))
    stride = int(meta.get("stride", stride))
    holdout_ratio = float(meta.get("holdout_ratio", 0.10))
    return VariantMeta(dataset, variant_name, window, stride, holdout_ratio)


def load_clients_from_manifest(variant_root: str) -> List[ClientInfo]:
    mani = _read_json(os.path.join(variant_root, "_manifest.json"))
    clients = mani.get("clients", None)
    if not isinstance(clients, list) or len(clients) == 0:
        croot = os.path.join(variant_root, "clients")
        if not os.path.isdir(croot):
            raise RuntimeError(f"clients/ missing: {variant_root}")
        names = sorted([x for x in os.listdir(croot) if os.path.isdir(os.path.join(croot, x))])
        return [ClientInfo(i, name) for i, name in enumerate(names)]

    out: List[ClientInfo] = []
    for i, c in enumerate(clients):
        name = str(c.get("name", "")).strip()
        if name == "":
            continue
        cid = c.get("id", i)
        out.append(ClientInfo(int(cid), name))

    if len(out) == 0:
        raise RuntimeError(f"No valid clients parsed from manifest: {variant_root}")

    ids = [c.client_id for c in out]
    names = [c.client_name for c in out]
    if len(set(ids)) != len(ids):
        raise RuntimeError(f"Duplicate client ids detected in {variant_root}")
    if len(set(names)) != len(names):
        raise RuntimeError(f"Duplicate client names detected in {variant_root}")

    out.sort(key=lambda x: x.client_id)
    return out


def _load_index_for_split(client_dir: str, split: str) -> FedMaskIndex:
    starts_p = os.path.join(client_dir, f"{split}_starts.npy")
    s_p = os.path.join(client_dir, f"{split}_s.npy")
    g_p = os.path.join(client_dir, f"{split}_g.npy")
    r_p = os.path.join(client_dir, f"{split}_regime.npy")

    if not os.path.isfile(starts_p):
        return FedMaskIndex(
            starts=np.zeros((0,), dtype=np.int64),
            s=np.zeros((0,), dtype=np.float32),
            g=np.zeros((0,), dtype=np.float32),
            regime_id=np.zeros((0,), dtype=np.int64),
        )

    starts = np.load(starts_p).astype(np.int64)
    W = int(starts.shape[0])
    s = np.load(s_p).astype(np.float32) if os.path.isfile(s_p) else np.zeros((W,), dtype=np.float32)
    g = np.load(g_p).astype(np.float32) if os.path.isfile(g_p) else np.zeros((W,), dtype=np.float32)
    regime_id = np.load(r_p).astype(np.int64) if os.path.isfile(r_p) else np.zeros((W,), dtype=np.int64)
    if not (starts.shape[0] == s.shape[0] == g.shape[0] == regime_id.shape[0]):
        raise RuntimeError(f"Inconsistent fed-mask index length: {client_dir} split={split}")
    return FedMaskIndex(starts, s, g, regime_id)


def load_fed_masks_indices(
    fed_masks_root: str,
    clients: List[ClientInfo],
) -> Tuple[Dict[str, Dict[str, FedMaskIndex]], Dict[str, Any]]:
    meta = _read_json(os.path.join(fed_masks_root, "_regime_meta.json"))
    out: Dict[str, Dict[str, FedMaskIndex]] = {}
    for ci in clients:
        cdir = os.path.join(fed_masks_root, ci.client_name)
        if not os.path.isdir(cdir):
            raise RuntimeError(f"Fed-mask client directory missing: {cdir}")
        out[ci.client_name] = {
            "train": _load_index_for_split(cdir, "train"),
            "val": _load_index_for_split(cdir, "val"),
            "test": _load_index_for_split(cdir, "test"),
        }
    return out, meta


# =============================================================================
# Deterministic I_train sampler
# =============================================================================


def _splitmix64(x: int) -> int:
    z = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
    z = (z ^ (z >> 31)) & 0xFFFFFFFFFFFFFFFF
    return int(z)


def _seed_for_itrain(global_seed: int, client_id: int, win_id: int) -> int:
    x = (int(global_seed) & 0xFFFFFFFFFFFFFFFF) ^ ((int(client_id) & 0xFFFFFFFF) << 32) ^ (int(win_id) & 0xFFFFFFFF)
    return _splitmix64(x)


def _apply_min_per_col_np(mask: np.ndarray, M_obs: np.ndarray, rng: np.random.Generator, min_per_col: int) -> None:
    if min_per_col <= 0:
        return
    _, D = M_obs.shape
    for d in range(D):
        obs = np.where(M_obs[:, d])[0]
        if obs.size == 0:
            continue
        cur = int(mask[:, d].sum())
        if cur >= int(min_per_col):
            continue
        cand = obs[~mask[obs, d]]
        need = int(min_per_col) - cur
        if cand.size == 0:
            continue
        pick = cand if cand.size <= need else rng.choice(cand, size=need, replace=False)
        mask[pick, d] = True


def _hard_guards_np(mask: np.ndarray, M_obs: np.ndarray, rng: np.random.Generator) -> None:
    obs_flat = np.flatnonzero(M_obs.reshape(-1))
    m = int(obs_flat.size)
    if m <= 0:
        return
    nnz = int(mask.sum())
    if nnz == 0:
        pick = int(rng.choice(obs_flat, size=1, replace=False)[0])
        mask.reshape(-1)[pick] = True
        nnz = 1
    if m > 1 and nnz >= m:
        ones = np.flatnonzero(mask.reshape(-1))
        drop = int(rng.choice(ones, size=1, replace=False)[0])
        mask.reshape(-1)[drop] = False


def itrain_mask_np(
    M_obs_2d: np.ndarray,
    *,
    ratio: float,
    seed: int,
    client_id: int,
    win_id: int,
    min_per_col: int = 0,
) -> np.ndarray:
    if M_obs_2d.ndim != 2:
        raise ValueError(f"itrain_mask_np expects [L,D], got {M_obs_2d.shape}")
    ratio = float(np.clip(ratio, 0.0, 0.999999))
    M_obs = np.asarray(M_obs_2d, dtype=bool, order="C")
    s = _seed_for_itrain(seed, client_id, win_id)
    rng = np.random.default_rng(s)
    base = (rng.random(size=M_obs.shape) < ratio) & M_obs
    _apply_min_per_col_np(base, M_obs, rng, min_per_col)
    _hard_guards_np(base, M_obs, rng)
    return base


def itrain_nnz_np(
    M_obs_2d: np.ndarray,
    *,
    ratio: float,
    seed: int,
    client_id: int,
    win_id: int,
    min_per_col: int = 0,
) -> int:
    return int(
        itrain_mask_np(
            M_obs_2d,
            ratio=ratio,
            seed=seed,
            client_id=client_id,
            win_id=win_id,
            min_per_col=min_per_col,
        ).sum()
    )


# =============================================================================
# Window dataset
# =============================================================================


def _pad_window_2d(arr2d: np.ndarray, target_dim: int) -> np.ndarray:
    if arr2d.ndim != 2:
        raise ValueError(f"Expected 2D array, got {arr2d.shape}")
    L, D = arr2d.shape
    if D == target_dim:
        return arr2d
    if D > target_dim:
        raise ValueError(f"Cannot crop dimensions: D={D} > target_dim={target_dim}")
    out = np.zeros((L, target_dim), dtype=arr2d.dtype)
    out[:, :D] = arr2d
    return out


def _validate_starts(starts: np.ndarray, T: int, window: int, client_name: str, split: str) -> None:
    if starts.ndim != 1:
        raise RuntimeError(f"starts must be 1D, got {starts.shape} | client={client_name} split={split}")
    if starts.size == 0:
        return
    if int(starts.min()) < 0:
        raise RuntimeError(f"Negative start detected | client={client_name} split={split}")
    max_ok = int(T - window)
    if int(starts.max()) > max_ok:
        raise RuntimeError(f"Start out of range: max={int(starts.max())} > {max_ok} | client={client_name} split={split}")


class FederatedWindowDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        M: np.ndarray,
        M_nat: np.ndarray,
        H: Optional[np.ndarray],
        M_in: Optional[np.ndarray],
        client_id: int,
        client_name: str,
        split: Literal["train", "val", "test"],
        window: int,
        stride: int,
        starts: Optional[np.ndarray] = None,
        s: Optional[np.ndarray] = None,
        g: Optional[np.ndarray] = None,
        regime_id: Optional[np.ndarray] = None,
        pad_to_dim: Optional[int] = None,
        strict_check: bool = True,
    ):
        if X.shape != M.shape or X.shape != M_nat.shape:
            raise ValueError(f"X/M/M_nat shape mismatch: X{X.shape} M{M.shape} M_nat{M_nat.shape}")
        if H is not None and H.shape != X.shape:
            raise ValueError(f"H shape mismatch: H{H.shape} X{X.shape}")
        if M_in is not None and M_in.shape != X.shape:
            raise ValueError(f"M_in shape mismatch: M_in{M_in.shape} X{X.shape}")

        self.client_id = int(client_id)
        self.client_name = str(client_name)
        self.split = split
        self.window = int(window)
        self.stride = int(stride)
        self.pad_to_dim = None if pad_to_dim is None else int(pad_to_dim)
        self.strict_check = bool(strict_check)

        self.X = np.asarray(X)
        self.M = np.asarray(M)
        self.M_nat = np.asarray(M_nat)
        self.H = None if H is None else np.asarray(H)
        self.M_in = None if M_in is None else np.asarray(M_in)

        if self.H is not None and self.strict_check:
            if np.any((self.H > 0.5) & (self.M <= 0.5)):
                raise RuntimeError(f"H must be subset of M | client={self.client_name} split={self.split}")
        if self.M_in is not None and self.strict_check:
            ref = self.M if self.H is None else self.M * (1.0 - (self.H > 0.5).astype(np.float32))
            if float(np.max(np.abs(self.M_in - ref))) > 1e-6:
                raise RuntimeError(f"M_in != canonical visible mask | client={self.client_name} split={self.split}")

        T = int(self.X.shape[0])
        if T < self.window:
            self.starts = np.zeros((0,), dtype=np.int64)
            self.s = np.zeros((0,), dtype=np.float32)
            self.g = np.zeros((0,), dtype=np.float32)
            self.regime_id = np.zeros((0,), dtype=np.int64)
            return

        self.starts = np.asarray(starts, dtype=np.int64) if starts is not None else np.arange(0, T - self.window + 1, self.stride, dtype=np.int64)
        if self.strict_check:
            _validate_starts(self.starts, T, self.window, self.client_name, self.split)
        W = int(self.starts.shape[0])

        def _ensure_meta(arr: Optional[np.ndarray], dtype, fill):
            if arr is None:
                return np.full((W,), fill, dtype=dtype)
            arr = np.asarray(arr)
            if arr.shape[0] != W:
                raise ValueError(f"Window metadata length mismatch: W={W} meta={arr.shape}")
            return arr.astype(dtype)

        self.s = _ensure_meta(s, np.float32, 0.0)
        self.g = _ensure_meta(g, np.float32, 0.0)
        self.regime_id = _ensure_meta(regime_id, np.int64, 0)

    def __len__(self) -> int:
        return int(self.starts.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        st = int(self.starts[idx])
        ed = st + self.window
        Xw = np.array(self.X[st:ed], dtype=np.float32, copy=True)
        Mw = np.array(self.M[st:ed], dtype=np.float32, copy=True)
        Mn = np.array(self.M_nat[st:ed], dtype=np.float32, copy=True)
        Hw = None if self.H is None else np.array(self.H[st:ed], dtype=np.float32, copy=True)
        if self.M_in is None:
            Min = Mw if Hw is None else Mw * (1.0 - (Hw > 0.5).astype(np.float32))
        else:
            Min = np.array(self.M_in[st:ed], dtype=np.float32, copy=True)

        if self.pad_to_dim is not None:
            Xw = _pad_window_2d(Xw, self.pad_to_dim)
            Mw = _pad_window_2d(Mw, self.pad_to_dim)
            Mn = _pad_window_2d(Mn, self.pad_to_dim)
            Min = _pad_window_2d(Min, self.pad_to_dim)
            if Hw is not None:
                Hw = _pad_window_2d(Hw, self.pad_to_dim)

        out: Dict[str, torch.Tensor] = {
            "X": torch.from_numpy(Xw),
            "M": torch.from_numpy(Mw),
            "M_nat": torch.from_numpy(Mn),
            "M_in": torch.from_numpy(Min),
            "M_eff": torch.from_numpy(Min),
            "client_id": torch.tensor(self.client_id, dtype=torch.long),
            "win_id": torch.tensor(int(idx), dtype=torch.long),
            "t_start": torch.tensor(st, dtype=torch.long),
            "s": torch.tensor(float(self.s[idx]), dtype=torch.float32),
            "g": torch.tensor(float(self.g[idx]), dtype=torch.float32),
            "regime_id": torch.tensor(int(self.regime_id[idx]), dtype=torch.long),
            "win_length": torch.tensor(self.window, dtype=torch.long),
        }
        if Hw is not None and self.split in ("val", "test"):
            ht = torch.from_numpy(Hw)
            out["H"] = ht
            out["I_eval"] = ht
        return out


# =============================================================================
# Array loading and dataloader construction
# =============================================================================


def _load_split_arrays(client_split_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    Xp = os.path.join(client_split_dir, "X.npy")
    Mp = os.path.join(client_split_dir, "M.npy")
    Mnp = os.path.join(client_split_dir, "M_nat.npy")
    if not os.path.isfile(Xp) or not os.path.isfile(Mp) or not os.path.isfile(Mnp):
        raise FileNotFoundError(f"Missing required split arrays under {client_split_dir}")
    X = np.load(Xp, mmap_mode="r")
    M = np.load(Mp, mmap_mode="r")
    M_nat = np.load(Mnp, mmap_mode="r")
    Hp = os.path.join(client_split_dir, "H.npy")
    Ip = os.path.join(client_split_dir, "I_eval.npy")
    H = np.load(Hp, mmap_mode="r") if os.path.isfile(Hp) else (np.load(Ip, mmap_mode="r") if os.path.isfile(Ip) else None)
    Minp = os.path.join(client_split_dir, "M_in.npy")
    M_in = np.load(Minp, mmap_mode="r") if os.path.isfile(Minp) else None
    return X, M, M_nat, H, M_in


def _infer_global_D_max(variant_root: str, clients: List[ClientInfo]) -> int:
    D_max = 0
    for ci in clients:
        p = os.path.join(variant_root, "clients", ci.client_name, "train", "X.npy")
        if not os.path.isfile(p):
            raise FileNotFoundError(p)
        X = np.load(p, mmap_mode="r")
        if X.ndim != 2:
            raise RuntimeError(f"X.npy must be [T,D], got {X.shape} @ {p}")
        D_max = max(D_max, int(X.shape[1]))
    if D_max <= 0:
        raise RuntimeError("Unable to infer global D_max")
    return D_max


def make_federated_dataloaders(
    data_root: str,
    dataset_name: str,
    variant: str,
    batch_size: int,
    shuffle_train: bool = True,
    use_fed_masks: bool = True,
    pad_to_global_max: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    strict_check: bool = True,
) -> Tuple[Dict[int, Dict[str, DataLoader]], List[ClientInfo], Dict[str, Any]]:
    dataset = _canonical_dataset_name(dataset_name)
    variant_root = resolve_variant_root(data_root, dataset, variant)
    vmeta = load_variant_meta(variant_root)
    clients = load_clients_from_manifest(variant_root)

    fed_indices: Optional[Dict[str, Dict[str, FedMaskIndex]]] = None
    fed_meta: Optional[Dict[str, Any]] = None
    if use_fed_masks:
        fed_root = resolve_fed_masks_root(data_root, dataset, variant)
        fed_indices, fed_meta = load_fed_masks_indices(fed_root, clients)

    D_max = _infer_global_D_max(variant_root, clients) if pad_to_global_max else 0
    pad_dim = D_max if pad_to_global_max else None

    loaders: Dict[int, Dict[str, DataLoader]] = {}
    for ci in clients:
        loaders[ci.client_id] = {}
        for split in ("train", "val", "test"):
            split_dir = os.path.join(variant_root, "clients", ci.client_name, split)
            X, M, M_nat, H, M_in = _load_split_arrays(split_dir)
            idx = None if fed_indices is None else fed_indices[ci.client_name][split]
            starts = None if idx is None else idx.starts
            s = None if idx is None else idx.s
            g = None if idx is None else idx.g
            regime_id = None if idx is None else idx.regime_id
            ds = FederatedWindowDataset(
                X=X,
                M=M,
                M_nat=M_nat,
                H=H,
                M_in=M_in,
                client_id=ci.client_id,
                client_name=ci.client_name,
                split=split,  # type: ignore[arg-type]
                window=vmeta.window,
                stride=vmeta.stride,
                starts=starts,
                s=s,
                g=g,
                regime_id=regime_id,
                pad_to_dim=pad_dim,
                strict_check=strict_check,
            )
            loaders[ci.client_id][split] = DataLoader(
                ds,
                batch_size=int(batch_size),
                shuffle=bool(shuffle_train and split == "train"),
                drop_last=False,
                num_workers=int(num_workers),
                pin_memory=bool(pin_memory),
            )

    meta: Dict[str, Any] = {
        "dataset": dataset,
        "variant_root": variant_root,
        "effective_window": int(vmeta.window),
        "effective_stride": int(vmeta.stride),
        "holdout_ratio": float(vmeta.holdout_ratio),
        "pad_to_global_max": bool(pad_to_global_max),
        "D_max": int(D_max),
        "use_fed_masks": bool(use_fed_masks),
        "strict_check": bool(strict_check),
    }
    if fed_meta is not None:
        meta["fed_masks_meta"] = fed_meta
    return loaders, clients, meta


# =============================================================================
# Mask protocol
# =============================================================================

TrainLossMaskMode = Literal["I_train", "missing", "M"]
EvalMaskMode = Literal["H", "I_eval", "missing", "M", "auto"]


@dataclass(frozen=True)
class MaskProtocolConfig:
    seed: int = 42
    train_holdout_ratio: float = 0.15
    train_min_per_col: int = 0
    force_x_obs: bool = True
    train_loss_mask_mode: TrainLossMaskMode = "I_train"
    eval_mask_mode: EvalMaskMode = "auto"
    prevent_eval_leakage: bool = True
    strict_no_leakage: bool = True


def to_bool_mask(m: torch.Tensor) -> torch.Tensor:
    return m if m.dtype == torch.bool else (m > 0.5)


def _missing_from(M_nat: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    return to_bool_mask(M_nat) & (~to_bool_mask(M))


def assert_no_leakage(input_mask: torch.Tensor, target_mask: torch.Tensor, strict: bool = True) -> None:
    inter = to_bool_mask(input_mask) & to_bool_mask(target_mask)
    leaked = int(inter.sum().item())
    if leaked == 0:
        return
    msg = f"Mask leakage detected: |input ∩ target| = {leaked} > 0"
    if strict:
        raise RuntimeError(msg)
    print(f"[WARN] {msg}")


def _make_itrain_from_M(
    M_bool: torch.Tensor,
    client_id: torch.Tensor,
    win_id: torch.Tensor,
    ratio: float,
    seed: int,
    min_per_col: int = 0,
) -> torch.Tensor:
    if M_bool.ndim != 3:
        raise ValueError(f"M_bool must be [B,L,D], got {tuple(M_bool.shape)}")
    B, L, D = M_bool.shape
    M_cpu = M_bool.detach().cpu().numpy().astype(bool)
    cid = client_id.detach().cpu().numpy().astype(np.int64)
    wid = win_id.detach().cpu().numpy().astype(np.int64)
    out = np.zeros((B, L, D), dtype=bool)
    for i in range(B):
        out[i] = itrain_mask_np(
            M_cpu[i],
            ratio=float(ratio),
            seed=int(seed),
            client_id=int(cid[i]),
            win_id=int(wid[i]),
            min_per_col=int(min_per_col),
        )
    return torch.from_numpy(out).to(device=M_bool.device)


def prepare_train_batch(
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    cfg: MaskProtocolConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if "X" not in batch:
        raise KeyError("batch missing X")
    X = batch["X"].to(device)
    M_base = batch["M_in"].to(device) if "M_in" in batch else (batch["M"].to(device) if "M" in batch else batch["M_eff"].to(device))
    M_base_bool = to_bool_mask(M_base)
    M = (batch["M"] if "M" in batch else batch["M_eff"]).to(device)

    if cfg.train_loss_mask_mode == "I_train":
        if "client_id" not in batch or "win_id" not in batch:
            raise KeyError("I_train mode requires client_id and win_id")
        I_train = _make_itrain_from_M(
            M_base_bool,
            batch["client_id"],
            batch["win_id"],
            float(cfg.train_holdout_ratio),
            int(cfg.seed),
            int(cfg.train_min_per_col),
        )
        loss_mask = I_train
        input_mask_bool = M_base_bool & (~I_train)
        assert_no_leakage(input_mask_bool, loss_mask, strict=cfg.strict_no_leakage)
    elif cfg.train_loss_mask_mode == "missing":
        if "M_nat" not in batch:
            raise KeyError("missing mode requires M_nat")
        loss_mask = _missing_from(batch["M_nat"].to(device), M)
        input_mask_bool = M_base_bool
    elif cfg.train_loss_mask_mode == "M":
        loss_mask = M_base_bool
        input_mask_bool = M_base_bool
    else:
        raise ValueError(f"Unknown train_loss_mask_mode: {cfg.train_loss_mask_mode}")

    M_in_f = input_mask_bool.to(dtype=X.dtype)
    X_in = X * M_in_f if cfg.force_x_obs else X
    return X_in, M_in_f, loss_mask.to(dtype=X.dtype)


@torch.no_grad()
def prepare_eval_batch(
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    split: Literal["val", "test"],
    cfg: MaskProtocolConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if "X" not in batch:
        raise KeyError("batch missing X")
    X = batch["X"].to(device)
    M = (batch["M"] if "M" in batch else batch["M_eff"]).to(device)
    M_bool = to_bool_mask(M)

    mode = cfg.eval_mask_mode
    if mode == "auto":
        mode = "H" if ("H" in batch or "I_eval" in batch) else "missing"

    if mode in ("H", "I_eval"):
        H = batch["H"].to(device) if "H" in batch else batch["I_eval"].to(device)
        eval_mask = to_bool_mask(H) & M_bool
    elif mode == "missing":
        if "M_nat" not in batch:
            raise KeyError("missing eval mode requires M_nat")
        eval_mask = _missing_from(batch["M_nat"].to(device), M)
    elif mode == "M":
        eval_mask = M_bool
    else:
        raise ValueError(f"Unknown eval_mask_mode: {cfg.eval_mask_mode}")

    if mode in ("H", "I_eval") and cfg.prevent_eval_leakage:
        if "M_in" in batch:
            input_mask_bool = to_bool_mask(batch["M_in"].to(device))
        else:
            input_mask_bool = M_bool & (~eval_mask)
        assert_no_leakage(input_mask_bool, eval_mask, strict=cfg.strict_no_leakage)
    else:
        input_mask_bool = M_bool

    M_in_f = input_mask_bool.to(dtype=X.dtype)
    X_in = X * M_in_f if cfg.force_x_obs else X
    return X_in, M_in_f, eval_mask.to(dtype=X.dtype)


# =============================================================================
# CLI preflight
# =============================================================================


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("CoRA data interface preflight")
    p.add_argument("--data_root", type=str, default="data")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--variant", type=str, default="hetero")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--no_fed_masks", action="store_true")
    p.add_argument("--no_pad", action="store_true")
    p.add_argument("--strict_check", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    loaders, clients, meta = make_federated_dataloaders(
        data_root=args.data_root,
        dataset_name=args.dataset,
        variant=args.variant,
        batch_size=int(args.batch_size),
        shuffle_train=True,
        use_fed_masks=(not bool(args.no_fed_masks)),
        pad_to_global_max=(not bool(args.no_pad)),
        strict_check=bool(args.strict_check),
    )
    print("=" * 80)
    print("[DATAIO] loaded")
    print(f"dataset={_canonical_dataset_name(args.dataset)} variant={args.variant}")
    print(f"variant_root={meta['variant_root']}")
    print(f"num_clients={len(clients)} window={meta['effective_window']} stride={meta['effective_stride']}")
    print(f"D_max={meta['D_max']} pad={meta['pad_to_global_max']} fed_masks={meta['use_fed_masks']}")
    print("=" * 80)

    cfg = MaskProtocolConfig(seed=int(args.seed))
    for ci in clients[: min(3, len(clients))]:
        batch = next(iter(loaders[ci.client_id]["train"]))
        X_in, M_in, loss_mask = prepare_train_batch(batch, device, cfg)
        print(f"client={ci.client_name} train batch -> X_in={tuple(X_in.shape)} M_in={tuple(M_in.shape)} loss={tuple(loss_mask.shape)}")
        batch = next(iter(loaders[ci.client_id]["val"]))
        X_in, M_in, eval_mask = prepare_eval_batch(batch, device, split="val", cfg=cfg)
        print(f"client={ci.client_name} val batch -> X_in={tuple(X_in.shape)} M_in={tuple(M_in.shape)} eval={tuple(eval_mask.shape)}")


if __name__ == "__main__":
    main()
