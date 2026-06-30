
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# --- path injection ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from tools.ur_fedimpute_dataio import (
        MaskProtocolConfig,
        make_federated_dataloaders,
        prepare_eval_batch,
        prepare_train_batch,
    )
    from tools.backbone_models import build_imputer, describe_backbone, model_impute, model_training_loss
except ImportError:
    from ur_fedimpute_dataio import (
        MaskProtocolConfig,
        make_federated_dataloaders,
        prepare_eval_batch,
        prepare_train_batch,
    )
    from backbone_models import build_imputer, describe_backbone, model_impute, model_training_loss

# =============================================================================
# Configuration
# =============================================================================

MethodName = Literal[
    "fedavg",
    "fedprox",
    "qfedavg",
    "localonly",
    "random",
    "fedopt",
    "scaffold",
    "cora",
    "cora_core",
    "cora_participation_only",
    "cora_stepalloc",
    "cora_stepalloc_ena",
    "cora_topk",
    "cora_noema",
    "cora_norefresh",
    "cora_nowarmup",
]

_REMOVED_METHODS = {
    "cora_local",
    "cora_local_asm",
    "cora_no_coverage",
    "cora_no_asm",
    "cora_no_sgp",
}

_ALIAS_TO_CANONICAL = {
    "cora": "cora_core",
    "cora_core": "cora_core",
    "cora_participation_only": "cora_core",
    "cora_stepalloc": "cora_stepalloc",
    "cora_stepalloc_ena": "cora_stepalloc_ena",
    "cora_topk": "cora_topk",
    # convenience ablation aliases
    "cora_noema": "cora_core",
    "cora_norefresh": "cora_core",
    "cora_nowarmup": "cora_core",
}


@dataclass
class ExperimentConfig:
    tag: str = ""
    seed: int = 42
    device: str = "cuda"
    out_dir: str = "results"

    data_root: str = "data"
    dataset: str = "ETTm1"
    variant: str = "hetero"
    batch_size: int = 64

    method: str = "fedavg"
    rounds: int = 50
    local_steps: int = 20

    # backbone. Use backbone=gru for the original CoRA experiments.
    # SAITS and CSDI are adapter-based checks requested in the FGCS revision.
    backbone: Literal["gru", "saits", "csdi"] = "gru"
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.0
    lr: float = 1e-3

    # SAITS adapter parameters. Defaults are intentionally lightweight relative
    # to the official best-tuned centralized configs so the federated check is
    # feasible under the same strict-budget protocol.
    saits_d_model: int = 64
    saits_d_inner: int = 128
    saits_n_groups: int = 2
    saits_n_group_inner_layers: int = 1
    saits_n_head: int = 4
    saits_d_k: int = 16
    saits_d_v: int = 16
    saits_dropout: float = 0.1
    saits_input_with_mask: int = 1
    saits_diagonal_attention_mask: int = 1
    saits_param_sharing_strategy: str = "between_group"
    saits_mit: int = 1

    # CSDI adapter parameters. CSDI is computationally heavier than GRU/SAITS;
    # keep eval_samples small for revision-scale backbone checks.
    csdi_layers: int = 2
    csdi_channels: int = 32
    csdi_nheads: int = 4
    csdi_diffusion_embedding_dim: int = 64
    csdi_beta_start: float = 1e-4
    csdi_beta_end: float = 0.5
    csdi_num_steps: int = 20
    csdi_schedule: str = "quad"
    csdi_timeemb: int = 64
    csdi_featureemb: int = 16
    csdi_eval_samples: int = 1
    csdi_is_linear: int = 0

    # federated baselines
    mu: float = 0.01
    q_param: float = 0.2

    # evaluation
    eval_every: int = 5
    eval_split: Literal["val", "test"] = "val"

    # score-guided participation core
    beta_hardness: float = 0.7
    h0: float = 1.0
    rho: float = 0.6
    T_part: int = 15
    T_refresh: int = 5
    K_min: int = 5

    # adaptive local-step plug-in (budget-preserving)
    score_floor: float = 1e-9

    stepalloc_min_steps: int = 15
    stepalloc_max_steps: int = 25
    stepalloc_power: float = 1.0

    # selected-client non-return injection for revised FGCS experiments
    # failure_mode: none, uniform, or score_correlated. Failure is sampled after
    # active-set selection and step assignment. Failed selected clients do not
    # return scalar losses or model deltas and therefore are excluded from score
    # updates and aggregation. Assigned steps remain counted as committed budget.
    failure_mode: Literal["none", "uniform", "score_correlated"] = "none"
    failure_rate: float = 0.0
    failure_qmax: float = 0.8
    failure_apply_full_rounds: bool = True
    failure_execute_training: bool = False

    # checkpoint selection. The revised paper reports test metrics at the
    # validation-selected checkpoint, not necessarily the last round.
    checkpoint_selection: Literal["best_val_rmse_avg", "last"] = "best_val_rmse_avg"

    # aggregation runtime knobs
    aggregation_rule: str = ""
    ena_reference_mode: str = "mean_effective"
    ena_eps: float = 1e-12
    ena_alpha: float = 1.0
    ena_clip_min: Optional[float] = None
    ena_clip_max: Optional[float] = None

    # FedOpt
    fedopt_type: Literal["adam", "yogi", "adagrad"] = "adam"
    fedopt_beta1: float = 0.9
    fedopt_beta2: float = 0.999
    fedopt_server_lr: float = 1e-1
    fedopt_tau: float = 1e-3

    # SCAFFOLD
    scaffold_global_lr: float = 1.0

    # masking protocol
    train_holdout_ratio: float = 0.15
    train_min_per_col: int = 0

    # deprecated knobs kept only for CLI compatibility / archived commands
    lambda_warmup_rounds: int = 15
    lambda_min: float = 0.2
    gamma: float = 0.8
    p_asm: float = 0.1
    tau_vis: int = 1
    psi: Literal["identity", "power"] = "identity"
    psi_power: float = 1.0
    fixed_coverage_ratio: float = -1.0
    coverage_floor_steps: int = 1


@dataclass
class MethodPlan:
    participation_policy: str
    stepalloc_policy: str
    aggregation_rule: str
    local_train_method: str
    update_score: bool


@dataclass
class ClientUpdatePackage:
    cid: int
    state_dict: Dict[str, torch.Tensor]
    avg_loss: float
    eff_steps: int
    assigned_steps: int
    omega: float


# =============================================================================
# Backbone model
# =============================================================================


class GRUImputer(nn.Module):
    def __init__(self, d_in: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.d_in = int(d_in)
        self.gru = nn.GRU(
            input_size=self.d_in * 2,
            hidden_size=int(hidden_size),
            num_layers=int(num_layers),
            batch_first=True,
            dropout=float(dropout) if int(num_layers) > 1 else 0.0,
        )
        self.head = nn.Linear(int(hidden_size), self.d_in)

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        self.gru.flatten_parameters()
        inp = torch.cat([x, m], dim=-1)
        h, _ = self.gru(inp)
        return self.head(h)


# =============================================================================
# State helpers
# =============================================================================


def _stable_int_seed(*parts: Any) -> int:
    s = "|".join(str(p) for p in parts).encode("utf-8")
    return int(hashlib.md5(s).hexdigest()[:8], 16)


@torch.no_grad()
def _clone_state_dict_cpu(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


@torch.no_grad()
def _zeros_like_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: torch.zeros_like(v.detach().cpu()) for k, v in model.state_dict().items()}


def _weighted_sum_state_dicts(
    dicts: List[Dict[str, torch.Tensor]],
    weights: np.ndarray,
) -> Dict[str, torch.Tensor]:
    if len(dicts) == 0:
        raise ValueError("No state dicts provided")
    if len(dicts) != int(weights.shape[0]):
        raise ValueError("weights length does not match number of state dicts")
    keys = list(dicts[0].keys())
    out: Dict[str, torch.Tensor] = {}
    for k in keys:
        stacked = torch.stack(
            [dicts[i][k].detach().cpu() * float(weights[i]) for i in range(len(dicts))],
            dim=0,
        )
        out[k] = stacked.sum(dim=0)
    return out


def _canonicalize_method(requested_method: str) -> str:
    m = str(requested_method).strip().lower()
    if m in _REMOVED_METHODS:
        raise ValueError(
            f"Method '{m}' has been intentionally removed from the core CoRA runner because it was empirically falsified. "
            f"Use the archived script if you need to reproduce those deprecated variants."
        )
    if m in _ALIAS_TO_CANONICAL:
        return _ALIAS_TO_CANONICAL[m]
    if m in {
        "fedavg",
        "fedprox",
        "qfedavg",
        "localonly",
        "random",
        "fedopt",
        "scaffold",
    }:
        return m
    raise ValueError(f"Unsupported method: {requested_method}")


def _apply_method_preset(cfg: ExperimentConfig, requested_method: str) -> ExperimentConfig:
    """
    Convenience aliases for ablation:
      - cora_noema      -> cora_core with beta_hardness = 0.0
      - cora_norefresh  -> cora_core with T_refresh = 0
      - cora_nowarmup   -> cora_core with T_part = 0
    """
    m = str(requested_method).strip().lower()
    if m == "cora_noema":
        cfg.beta_hardness = 0.0
    elif m == "cora_norefresh":
        cfg.T_refresh = 0
    elif m == "cora_nowarmup":
        cfg.T_part = 0
    return cfg


def resolve_method_plan(canonical_method: str) -> MethodPlan:
    if canonical_method == "fedavg":
        return MethodPlan("all", "fixed", "model_avg", "fedavg", False)
    if canonical_method == "fedprox":
        return MethodPlan("all", "fixed", "model_avg", "fedprox", False)
    if canonical_method == "qfedavg":
        return MethodPlan("all", "fixed", "qfedavg", "fedavg", False)
    if canonical_method == "localonly":
        return MethodPlan("all", "fixed", "localonly", "fedavg", False)
    if canonical_method == "random":
        return MethodPlan("random", "fixed", "model_avg", "fedavg", False)
    if canonical_method == "fedopt":
        return MethodPlan("all", "fixed", "fedopt", "fedavg", False)
    if canonical_method == "scaffold":
        return MethodPlan("all", "fixed", "scaffold", "scaffold", False)
    if canonical_method == "cora_core":
        return MethodPlan("cora_score", "fixed", "model_avg", "fedavg", True)
    if canonical_method == "cora_stepalloc":
        return MethodPlan("cora_score", "cora_budgeted", "ena", "fedavg", True)
    if canonical_method == "cora_stepalloc_ena":
        return MethodPlan("cora_score", "cora_budgeted", "ena", "fedavg", True)
    if canonical_method == "cora_topk":
        return MethodPlan("cora_topk", "fixed", "model_avg", "fedavg", True)
    raise ValueError(f"Unsupported canonical method for plan resolution: {canonical_method}")


def _apply_method_runtime_preset(cfg: ExperimentConfig, canonical_method: str) -> ExperimentConfig:
    plan = resolve_method_plan(canonical_method)
    agg = str(getattr(cfg, "aggregation_rule", "") or "").strip().lower()
    if agg == "":
        cfg.aggregation_rule = str(plan.aggregation_rule)
    if canonical_method in {"cora_stepalloc", "cora_stepalloc_ena"} and str(cfg.aggregation_rule).strip().lower() == "model_avg":
        cfg.aggregation_rule = "ena"
    if not str(getattr(cfg, "ena_reference_mode", "") or "").strip():
        cfg.ena_reference_mode = "mean_effective"
    return cfg


# =============================================================================
# Client-side routing state
# =============================================================================


class ClientState:
    def __init__(self, cid: int, cfg: ExperimentConfig):
        self.cid = int(cid)
        self.beta = float(cfg.beta_hardness)
        self.omega: float = float(cfg.h0)
        self.loss_history: float = float(cfg.h0)

    def update_omega(self, loss_val: float) -> None:
        lv = float(loss_val)
        self.omega = self.beta * self.omega + (1.0 - self.beta) * lv
        self.loss_history = lv


# =============================================================================
# Server-side method state
# =============================================================================


class FedOptState:
    def __init__(self, model: nn.Module, cfg: ExperimentConfig):
        self.opt_type = str(cfg.fedopt_type)
        self.beta1 = float(cfg.fedopt_beta1)
        self.beta2 = float(cfg.fedopt_beta2)
        self.server_lr = float(cfg.fedopt_server_lr)
        self.tau = float(cfg.fedopt_tau)
        self.m = _zeros_like_state_dict(model)
        self.v = _zeros_like_state_dict(model)

    @torch.no_grad()
    def step(
        self,
        global_model: nn.Module,
        deltas: List[Dict[str, torch.Tensor]],
        weights: np.ndarray,
    ) -> None:
        if len(deltas) == 0:
            return
        agg = _weighted_sum_state_dicts(deltas, weights)
        current = _clone_state_dict_cpu(global_model)
        new_state: Dict[str, torch.Tensor] = {}

        for k in current.keys():
            g = agg[k]
            self.m[k] = self.beta1 * self.m[k] + (1.0 - self.beta1) * g
            if self.opt_type == "adagrad":
                self.v[k] = self.v[k] + g**2
            elif self.opt_type == "yogi":
                g2 = g**2
                self.v[k] = self.v[k] - (1.0 - self.beta2) * g2 * torch.sign(self.v[k] - g2)
            else:  # adam
                self.v[k] = self.beta2 * self.v[k] + (1.0 - self.beta2) * (g**2)
            new_state[k] = current[k] + self.server_lr * (self.m[k] / (torch.sqrt(self.v[k]) + self.tau))

        global_model.load_state_dict(new_state)


class ScaffoldState:
    def __init__(self, model: nn.Module, clients: List[int], cfg: ExperimentConfig):
        self.global_lr = float(cfg.scaffold_global_lr)
        self.c_global = _zeros_like_state_dict(model)
        self.c_local = {int(cid): _zeros_like_state_dict(model) for cid in clients}


# =============================================================================
# Evaluation
# =============================================================================


@torch.no_grad()
def eval_model_single(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mask_cfg: MaskProtocolConfig,
    split: Literal["val", "test"],
) -> Dict[str, float]:
    model.eval()
    sum_sq = 0.0
    sum_abs = 0.0
    sum_m = 0.0

    for batch in loader:
        X_in, M_in, M_eval = prepare_eval_batch(batch, device, split=split, cfg=mask_cfg)
        X_true = batch["X"].to(device)
        rec = model_impute(model, X_in, M_in)
        diff = rec - X_true
        M_eval_f = M_eval.to(dtype=X_true.dtype)

        sum_sq += float(torch.sum((diff**2) * M_eval_f).item())
        sum_abs += float(torch.sum(torch.abs(diff) * M_eval_f).item())
        sum_m += float(torch.sum(M_eval_f).item())

    if sum_m <= 0:
        return {
            "rmse": float("inf"),
            "mae": float("inf"),
            "sum_sq": float("inf"),
            "sum_abs": float("inf"),
            "denom": 0.0,
        }

    return {
        "rmse": float(np.sqrt(sum_sq / (sum_m + 1e-5))),
        "mae": float(sum_abs / (sum_m + 1e-5)),
        "sum_sq": float(sum_sq),
        "sum_abs": float(sum_abs),
        "denom": float(sum_m),
    }


@torch.no_grad()
def eval_over_clients(
    model: nn.Module,
    loaders: Dict[int, Dict[str, DataLoader]],
    clients: List[int],
    device: torch.device,
    mask_cfg: MaskProtocolConfig,
    split: Literal["val", "test"],
) -> Tuple[Dict[str, float], Dict[int, Dict[str, float]]]:
    per: Dict[int, Dict[str, float]] = {}
    sum_sq_total = 0.0
    sum_abs_total = 0.0
    sum_m_total = 0.0

    for cid in clients:
        m = eval_model_single(model, loaders[cid][split], device, mask_cfg, split)
        per[int(cid)] = {
            "rmse": float(m["rmse"]),
            "mae": float(m["mae"]),
            "denom": float(m["denom"]),
        }
        if np.isfinite(m["sum_sq"]) and float(m["denom"]) > 1e-9:
            sum_sq_total += float(m["sum_sq"])
            sum_abs_total += float(m["sum_abs"])
            sum_m_total += float(m["denom"])

    if len(per) == 0:
        inf = float("inf")
        return {
            "rmse_macro": inf,
            "mae_macro": inf,
            "rmse_worst": inf,
            "mae_worst": inf,
            "rmse_std": inf,
            "mae_std": inf,
            "rmse_micro": inf,
            "mae_micro": inf,
            "n_clients": 0.0,
        }, per

    rmses = [x["rmse"] for x in per.values()]
    maes = [x["mae"] for x in per.values()]

    out = {
        "rmse_macro": float(np.mean(rmses)),
        "mae_macro": float(np.mean(maes)),
        "rmse_worst": float(np.max(rmses)),
        "mae_worst": float(np.max(maes)),
        "rmse_std": float(np.std(rmses)),
        "mae_std": float(np.std(maes)),
        "rmse_micro": float(np.sqrt(sum_sq_total / (sum_m_total + 1e-5))) if sum_m_total > 1e-9 else float("inf"),
        "mae_micro": float(sum_abs_total / (sum_m_total + 1e-5)) if sum_m_total > 1e-9 else float("inf"),
        "n_clients": float(len(per)),
    }
    return out, per


# =============================================================================
# Local training loops
# =============================================================================


def _masked_mse_loss(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    mask_cfg: MaskProtocolConfig,
) -> Tuple[torch.Tensor, float]:
    X_true = batch["X"].to(device)
    X_in, M_in, loss_mask = prepare_train_batch(batch, device, mask_cfg)
    loss = model_training_loss(model, X_true, X_in, M_in, loss_mask)
    return loss, float(loss.detach().item())


def train_local_baseline(
    model: nn.Module,
    loader: DataLoader,
    state: ClientState,
    cfg: ExperimentConfig,
    device: torch.device,
    mask_cfg: MaskProtocolConfig,
    method: str,
    local_steps_override: Optional[int] = None,
    update_score: bool = False,
) -> Tuple[Dict[str, torch.Tensor], float, int, Dict[str, Any]]:
    local_steps = int(local_steps_override if local_steps_override is not None else cfg.local_steps)

    if len(loader.dataset) == 0 or local_steps <= 0:  # type: ignore[arg-type]
        return _clone_state_dict_cpu(model), float(state.loss_history), 0, {}

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.lr))

    global_ref_params = None
    if method == "fedprox":
        global_ref_params = [p.detach().clone() for p in model.parameters()]

    total_loss = 0.0
    steps = 0
    data_iterator = iter(loader)

    for _ in range(local_steps):
        try:
            batch = next(data_iterator)
        except StopIteration:
            data_iterator = iter(loader)
            batch = next(data_iterator)

        optimizer.zero_grad(set_to_none=True)
        loss, task_loss_val = _masked_mse_loss(model, batch, device, mask_cfg)

        if method == "fedprox" and global_ref_params is not None:
            prox = 0.0
            for w, w_ref in zip(model.parameters(), global_ref_params):
                prox += (w - w_ref).norm(2) ** 2
            loss = loss + (float(cfg.mu) / 2.0) * prox

        loss.backward()
        optimizer.step()

        total_loss += task_loss_val
        steps += 1

    avg_loss = total_loss / max(1, steps)
    state.loss_history = float(avg_loss)
    if update_score:
        state.update_omega(float(avg_loss))

    aux = {
        "assigned_local_steps": int(local_steps),
        "effective_steps": int(steps),
        "omega_after": float(state.omega),
        "loss_history": float(state.loss_history),
        "avg_loss": float(avg_loss),
    }
    return _clone_state_dict_cpu(model), float(avg_loss), int(steps), aux


def train_local_scaffold(
    model: nn.Module,
    loader: DataLoader,
    state: ClientState,
    cfg: ExperimentConfig,
    device: torch.device,
    mask_cfg: MaskProtocolConfig,
    scaffold_state: ScaffoldState,
    local_steps_override: Optional[int] = None,
) -> Tuple[Dict[str, torch.Tensor], float, int, Dict[str, Any], Dict[str, Dict[str, torch.Tensor]]]:
    local_steps = int(local_steps_override if local_steps_override is not None else cfg.local_steps)

    if len(loader.dataset) == 0 or local_steps <= 0:  # type: ignore[arg-type]
        return _clone_state_dict_cpu(model), float(state.loss_history), 0, {}, {}

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.lr))
    global_params = _clone_state_dict_cpu(model)
    c_global = scaffold_state.c_global
    c_local = scaffold_state.c_local[int(state.cid)]

    total_loss = 0.0
    steps = 0
    data_iterator = iter(loader)

    for _ in range(local_steps):
        try:
            batch = next(data_iterator)
        except StopIteration:
            data_iterator = iter(loader)
            batch = next(data_iterator)

        optimizer.zero_grad(set_to_none=True)
        loss, task_loss_val = _masked_mse_loss(model, batch, device, mask_cfg)
        loss.backward()

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is None:
                    continue
                correction = c_global[name].to(device) - c_local[name].to(device)
                param.grad.add_(correction)

        optimizer.step()
        total_loss += task_loss_val
        steps += 1

    avg_loss = total_loss / max(1, steps)
    state.loss_history = float(avg_loss)

    local_params = _clone_state_dict_cpu(model)
    y_delta: Dict[str, torch.Tensor] = {}
    c_delta: Dict[str, torch.Tensor] = {}
    coef = 1.0 / (float(steps) * float(cfg.lr)) if steps > 0 else 0.0

    new_c_local: Dict[str, torch.Tensor] = {}
    for name in global_params.keys():
        y_delta[name] = local_params[name] - global_params[name]
        if coef > 0:
            c_plus = c_local[name] - c_global[name] - coef * y_delta[name]
        else:
            c_plus = c_local[name].clone()
        c_delta[name] = c_plus - c_local[name]
        new_c_local[name] = c_plus

    scaffold_state.c_local[int(state.cid)] = new_c_local

    aux = {
        "assigned_local_steps": int(local_steps),
        "effective_steps": int(steps),
        "omega_after": float(state.omega),
        "loss_history": float(state.loss_history),
        "avg_loss": float(avg_loss),
    }
    package = {"y_delta": y_delta, "c_delta": c_delta}
    return local_params, float(avg_loss), int(steps), aux, package


# =============================================================================
# Aggregation and participation
# =============================================================================


def aggregate_client_packages(
    global_model: nn.Module,
    client_packages: List[ClientUpdatePackage],
    aggregation_rule: str,
    q_param: float,
    ena_reference_mode: str,
    configured_local_steps: int,
    ena_eps: float = 1e-12,
    ena_alpha: float = 1.0,
    ena_clip_min: Optional[float] = None,
    ena_clip_max: Optional[float] = None,
) -> Dict[str, Any]:
    aggregation_rule = str(aggregation_rule).strip().lower()
    n = len(client_packages)
    if n == 0:
        return {
            "aggregation_rule": aggregation_rule,
            "ref_steps": None,
            "client_scales": {},
            "client_base_weights": {},
            "client_effective_weights": {},
        }

    global_state = _clone_state_dict_cpu(global_model)
    deltas = [{k: pkg.state_dict[k] - global_state[k] for k in global_state.keys()} for pkg in client_packages]

    base_weights = np.ones((n,), dtype=np.float64) / max(1, n)
    scales = np.ones((n,), dtype=np.float64)
    raw_ratios = np.ones((n,), dtype=np.float64)
    powered_scales = np.ones((n,), dtype=np.float64)
    ref_steps: Optional[float] = None

    if aggregation_rule == "qfedavg":
        raw = np.asarray([(max(1e-10, float(pkg.avg_loss)) ** float(q_param)) for pkg in client_packages], dtype=np.float64)
        if float(raw.sum()) > 0 and np.isfinite(raw).all():
            base_weights = raw / raw.sum()
    elif aggregation_rule == "ena":
        mode = str(ena_reference_mode).strip().lower()
        if mode == "mean_assigned":
            ref_steps = float(np.mean([max(1, int(pkg.assigned_steps)) for pkg in client_packages]))
        elif mode == "configured_local_steps":
            ref_steps = float(max(1, int(configured_local_steps)))
        else:
            ref_steps = float(np.mean([max(1, int(pkg.eff_steps)) for pkg in client_packages]))
            mode = "mean_effective"
        alpha = float(ena_alpha)
        if (not np.isfinite(alpha)) or alpha < 0.0:
            raise ValueError("ena_alpha must be finite and >= 0")
        raw_ratios = np.asarray(
            [float(ref_steps) / max(float(ena_eps), float(max(1, int(pkg.eff_steps)))) for pkg in client_packages],
            dtype=np.float64,
        )
        powered_scales = np.power(raw_ratios, alpha)
        scales = powered_scales.copy()
        clip_min = None if ena_clip_min is None else float(ena_clip_min)
        clip_max = None if ena_clip_max is None else float(ena_clip_max)
        if clip_min is not None and clip_max is not None and clip_min > clip_max:
            raise ValueError("ena_clip_min cannot exceed ena_clip_max")
        if clip_min is not None:
            scales = np.maximum(scales, float(clip_min))
        if clip_max is not None:
            scales = np.minimum(scales, float(clip_max))
        ena_reference_mode = mode
    elif aggregation_rule != "model_avg":
        raise ValueError(f"Unsupported aggregation_rule: {aggregation_rule}")

    agg_weights = base_weights * scales
    agg_delta = _weighted_sum_state_dicts(deltas, agg_weights)
    new_state = {k: global_state[k] + agg_delta[k] for k in global_state.keys()}
    global_model.load_state_dict(new_state)

    return {
        "aggregation_rule": aggregation_rule,
        "ena_reference_mode": str(ena_reference_mode).strip().lower() if aggregation_rule == "ena" else "",
        "ena_alpha": float(ena_alpha) if aggregation_rule == "ena" else None,
        "ena_clip_min": None if aggregation_rule != "ena" else (None if ena_clip_min is None else float(ena_clip_min)),
        "ena_clip_max": None if aggregation_rule != "ena" else (None if ena_clip_max is None else float(ena_clip_max)),
        "ref_steps": float(ref_steps) if ref_steps is not None else None,
        "client_ratio_raw": {int(pkg.cid): float(raw_ratios[i]) for i, pkg in enumerate(client_packages)},
        "client_scale_powered": {int(pkg.cid): float(powered_scales[i]) for i, pkg in enumerate(client_packages)},
        "client_scales": {int(pkg.cid): float(scales[i]) for i, pkg in enumerate(client_packages)},
        "client_base_weights": {int(pkg.cid): float(base_weights[i]) for i, pkg in enumerate(client_packages)},
        "client_effective_weights": {int(pkg.cid): float(agg_weights[i]) for i, pkg in enumerate(client_packages)},
    }


def aggregate_scaffold(
    global_model: nn.Module,
    scaffold_state: ScaffoldState,
    packages: List[Dict[str, Dict[str, torch.Tensor]]],
    total_client_num: int,
) -> None:
    if len(packages) == 0:
        return

    n = len(packages)
    weights = np.ones((n,), dtype=np.float64) / n
    y_list = [pkg["y_delta"] for pkg in packages]
    c_list = [pkg["c_delta"] for pkg in packages]

    agg_y = _weighted_sum_state_dicts(y_list, weights)
    current = _clone_state_dict_cpu(global_model)
    new_state: Dict[str, torch.Tensor] = {}
    for k in current.keys():
        new_state[k] = current[k] + scaffold_state.global_lr * agg_y[k]
    global_model.load_state_dict(new_state)

    for k in scaffold_state.c_global.keys():
        stacked = torch.stack([pkg[k] for pkg in c_list], dim=0)
        scaffold_state.c_global[k] = scaffold_state.c_global[k] + stacked.sum(dim=0) / float(total_client_num)


def _select_topk_by_score(clients: List[int], client_states: Dict[int, ClientState], m: int) -> List[int]:
    ranked = sorted(
        clients,
        key=lambda c: (-float(client_states[c].omega), int(c)),
    )
    return sorted(ranked[:m])


def select_active_clients(
    clients: List[int],
    client_states: Dict[int, ClientState],
    cfg: ExperimentConfig,
    round_idx: int,
    participation_policy: str,
) -> Tuple[List[int], Dict[str, Any]]:
    policy = str(participation_policy).strip().lower()

    if policy == "all":
        return list(clients), {"mode": "all", "threshold": -float("inf"), "full": True}

    if policy == "random":
        if round_idx <= int(cfg.T_part):
            return list(clients), {"mode": "warmup_all", "threshold": -float("inf"), "full": True}
        if int(cfg.T_refresh) > 0 and round_idx % int(cfg.T_refresh) == 0:
            return list(clients), {"mode": "refresh_all", "threshold": -float("inf"), "full": True}

        N = len(clients)
        m = max(int(cfg.K_min), int(np.ceil(float(cfg.rho) * N)))
        m = min(max(1, m), N)

        rng = np.random.default_rng(
            _stable_int_seed(cfg.seed, cfg.dataset, cfg.variant, policy, round_idx, "sel")
        )
        active = sorted(rng.choice(np.asarray(clients, dtype=np.int64), size=m, replace=False).astype(int).tolist())
        return active, {"mode": "random_partial", "threshold": -float("inf"), "full": False}

    if policy in ("cora_score", "cora_topk"):
        if round_idx <= int(cfg.T_part):
            return list(clients), {"mode": "warmup_all", "threshold": -float("inf"), "full": True}
        if int(cfg.T_refresh) > 0 and round_idx % int(cfg.T_refresh) == 0:
            return list(clients), {"mode": "refresh_all", "threshold": -float("inf"), "full": True}

        N = len(clients)
        m = max(int(cfg.K_min), int(np.ceil(float(cfg.rho) * N)))
        m = min(max(1, m), N)

        raw_scores = np.array([max(float(cfg.score_floor), float(client_states[c].omega)) for c in clients], dtype=np.float64)

        if policy == "cora_topk":
            active = _select_topk_by_score(clients, client_states, m)
            threshold = float(min(client_states[c].omega for c in active)) if len(active) > 0 else -float("inf")
            return active, {"mode": "score_topk", "threshold": threshold, "full": False}

        probs = raw_scores / float(raw_scores.sum())
        rng = np.random.default_rng(
            _stable_int_seed(cfg.seed, cfg.dataset, cfg.variant, policy, round_idx, "sel")
        )
        active_np = rng.choice(np.asarray(clients, dtype=np.int64), size=m, replace=False, p=probs)
        active = sorted(active_np.astype(int).tolist())
        threshold = float(min(client_states[c].omega for c in active)) if len(active) > 0 else -float("inf")
        return active, {"mode": "score_guided", "threshold": threshold, "full": False}

    raise ValueError(f"Unsupported participation_policy: {participation_policy}")


def _bounded_integer_step_allocation(
    active_cids: List[int],
    weights: np.ndarray,
    total_budget: int,
    min_steps: int,
    max_steps: int,
) -> Dict[int, int]:
    n = len(active_cids)
    if n == 0:
        return {}
    if min_steps > max_steps:
        raise ValueError("stepalloc_min_steps cannot exceed stepalloc_max_steps")
    if total_budget < n * min_steps:
        raise ValueError(
            f"Infeasible step-allocation budget: total_budget={total_budget} < n_clients*min_steps={n * min_steps}."
        )
    if total_budget > n * max_steps:
        raise ValueError(
            f"Infeasible step-allocation budget: total_budget={total_budget} > n_clients*max_steps={n * max_steps}."
        )

    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 1 or w.shape[0] != n or (not np.isfinite(w).all()) or float(w.sum()) <= 0:
        w = np.ones((n,), dtype=np.float64)
    w = w / float(w.sum())

    alloc = np.full((n,), int(min_steps), dtype=np.int64)
    remaining = int(total_budget - int(np.sum(alloc)))

    while remaining > 0:
        available = np.where(alloc < int(max_steps))[0]
        if available.size == 0:
            raise RuntimeError("No available client can receive additional local steps, but budget still remains.")

        avail_w = w[available]
        if float(avail_w.sum()) <= 0 or (not np.isfinite(avail_w).all()):
            avail_w = np.ones((available.size,), dtype=np.float64)
        avail_w = avail_w / float(avail_w.sum())

        raw = float(remaining) * avail_w
        adds = np.floor(raw).astype(np.int64)

        progress = 0
        for j, idx in enumerate(available.tolist()):
            cap = int(max_steps) - int(alloc[idx])
            add = min(int(adds[j]), cap)
            if add > 0:
                alloc[idx] += add
                remaining -= add
                progress += add

        if remaining <= 0:
            break

        order = available[np.argsort(-avail_w)]
        for idx in order.tolist():
            if remaining <= 0:
                break
            if int(alloc[idx]) < int(max_steps):
                alloc[idx] += 1
                remaining -= 1
                progress += 1

        if progress <= 0:
            raise RuntimeError("Step allocation made no progress; check bounds and weights.")

    return {int(active_cids[i]): int(alloc[i]) for i in range(n)}


def allocate_local_steps(
    active_cids: List[int],
    client_states: Dict[int, ClientState],
    cfg: ExperimentConfig,
    stepalloc_policy: str,
    part_info: Dict[str, Any],
) -> Tuple[Dict[int, int], Dict[str, Any]]:
    default_alloc = {int(cid): int(cfg.local_steps) for cid in active_cids}
    meta = {
        "allocation_mode": "fixed",
        "total_budget": int(len(active_cids) * int(cfg.local_steps)),
        "min_steps": int(cfg.local_steps),
        "max_steps": int(cfg.local_steps),
        "score_power": 1.0,
    }

    if str(stepalloc_policy).strip().lower() != "cora_budgeted":
        return default_alloc, meta
    if part_info.get("mode") != "score_guided":
        return default_alloc, meta
    if len(active_cids) == 0:
        return default_alloc, meta

    total_budget = int(len(active_cids) * int(cfg.local_steps))
    min_steps = int(cfg.stepalloc_min_steps)
    max_steps = int(cfg.stepalloc_max_steps)

    power = float(cfg.stepalloc_power)
    if (not np.isfinite(power)) or power <= 0:
        raise ValueError("stepalloc_power must be > 0")

    raw = np.asarray([max(float(cfg.score_floor), float(client_states[cid].omega)) for cid in active_cids], dtype=np.float64)
    weights = np.power(raw, power)

    alloc = _bounded_integer_step_allocation(active_cids, weights, total_budget, min_steps, max_steps)
    meta = {
        "allocation_mode": "adaptive_budgeted",
        "total_budget": int(total_budget),
        "min_steps": int(min_steps),
        "max_steps": int(max_steps),
        "score_power": float(power),
    }
    return alloc, meta


# =============================================================================
# Selected-client non-return injection
# =============================================================================


def _bool_cfg(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(int(x))
    return str(x).strip().lower() in {"1", "true", "yes", "y", "on"}


def _solve_score_correlated_multiplier(base: np.ndarray, target_rate: float, qmax: float) -> float:
    """Find c such that mean(min(qmax, c * base)) approximately equals target_rate."""
    b = np.asarray(base, dtype=np.float64)
    b = np.where(np.isfinite(b) & (b > 0), b, 1.0)
    target = float(np.clip(target_rate, 0.0, 1.0))
    qmax = float(np.clip(qmax, 0.0, 1.0))
    if target <= 0.0 or qmax <= 0.0:
        return 0.0
    if target >= qmax:
        return float("inf")
    lo, hi = 0.0, max(target / max(float(np.mean(b)), 1e-12), 1e-12)
    for _ in range(64):
        val = float(np.mean(np.minimum(qmax, hi * b)))
        if val >= target:
            break
        hi *= 2.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        val = float(np.mean(np.minimum(qmax, mid * b)))
        if val < target:
            lo = mid
        else:
            hi = mid
    return hi


def sample_nonreturn_clients(
    active_cids: List[int],
    client_states: Dict[int, ClientState],
    cfg: ExperimentConfig,
    round_idx: int,
    part_info: Dict[str, Any],
) -> Tuple[set[int], Dict[str, Any]]:
    """Sample selected clients that fail to return on this round.

    Failure is injected after active-set selection and step assignment. Failed
    clients remain part of the assigned budget but do not enter the valid-return
    set used for score updates and aggregation.
    """
    active = [int(c) for c in active_cids]
    mode = str(getattr(cfg, "failure_mode", "none") or "none").strip().lower()
    rate = float(np.clip(float(getattr(cfg, "failure_rate", 0.0) or 0.0), 0.0, 1.0))
    qmax = float(np.clip(float(getattr(cfg, "failure_qmax", 0.8) or 0.8), 0.0, 1.0))
    apply_full = _bool_cfg(getattr(cfg, "failure_apply_full_rounds", True))

    if len(active) == 0 or mode in {"", "none", "no", "false"} or rate <= 0.0:
        return set(), {
            "failure_mode": "none" if mode in {"", "none", "no", "false"} else mode,
            "failure_rate": 0.0,
            "failure_qmax": qmax,
            "applied": False,
            "prob_by_client": {int(c): 0.0 for c in active},
            "failed_clients": [],
        }

    if bool(part_info.get("full", False)) and not apply_full:
        return set(), {
            "failure_mode": mode,
            "failure_rate": rate,
            "failure_qmax": qmax,
            "applied": False,
            "prob_by_client": {int(c): 0.0 for c in active},
            "failed_clients": [],
        }

    scores = np.asarray([max(float(cfg.score_floor), float(client_states[c].omega)) for c in active], dtype=np.float64)
    if mode == "uniform":
        probs = np.full((len(active),), rate, dtype=np.float64)
    elif mode in {"score_correlated", "score-correlated", "scorecorr"}:
        mean_score = float(np.mean(scores)) if np.isfinite(scores).all() and float(np.mean(scores)) > 0 else 1.0
        base = scores / mean_score
        c = _solve_score_correlated_multiplier(base, target_rate=rate, qmax=qmax)
        if np.isinf(c):
            probs = np.full((len(active),), qmax, dtype=np.float64)
        else:
            probs = np.minimum(qmax, c * base)
        mode = "score_correlated"
    else:
        raise ValueError(f"Unsupported failure_mode: {mode}")

    failed: set[int] = set()
    prob_by_client: Dict[int, float] = {}
    u_by_client: Dict[int, float] = {}
    for cid, q in zip(active, probs.tolist()):
        q = float(np.clip(q, 0.0, 1.0))
        prob_by_client[int(cid)] = q
        # Use a client/round indexed random variate so common selected clients
        # face the same failure draw across methods under the same seed.
        rng = np.random.default_rng(
            _stable_int_seed(cfg.seed, cfg.dataset, cfg.variant, mode, float(rate), int(round_idx), int(cid), "nonreturn")
        )
        u = float(rng.random())
        u_by_client[int(cid)] = u
        if u < q:
            failed.add(int(cid))

    return failed, {
        "failure_mode": mode,
        "failure_rate": rate,
        "failure_qmax": qmax,
        "applied": True,
        "prob_by_client": prob_by_client,
        "uniform_by_client": u_by_client,
        "failed_clients": sorted(failed),
    }


# =============================================================================
# LocalOnly baseline
# =============================================================================


def run_localonly_suite(
    loaders: Dict[int, Dict[str, DataLoader]],
    clients: List[int],
    cfg: ExperimentConfig,
    device: torch.device,
    mask_cfg: MaskProtocolConfig,
    D: int,
    window_length: int,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"per_client_test": {}}
    total_steps = int(cfg.rounds) * int(cfg.local_steps)
    t0 = time.perf_counter()

    for cid in clients:
        model = build_imputer(cfg, D, int(window_length), device=device).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.lr))

        data_iterator = iter(loaders[cid]["train"])
        steps_done = 0
        model.train()

        if len(loaders[cid]["train"].dataset) > 0:  # type: ignore[arg-type]
            while steps_done < total_steps:
                try:
                    batch = next(data_iterator)
                except StopIteration:
                    data_iterator = iter(loaders[cid]["train"])
                    batch = next(data_iterator)

                optimizer.zero_grad(set_to_none=True)
                loss, _ = _masked_mse_loss(model, batch, device, mask_cfg)
                loss.backward()
                optimizer.step()
                steps_done += 1

        metrics = eval_model_single(model, loaders[cid]["test"], device, mask_cfg, split="test")
        out["per_client_test"][str(cid)] = {
            "rmse": float(metrics["rmse"]),
            "mae": float(metrics["mae"]),
        }

    rmses = [v["rmse"] for v in out["per_client_test"].values()]
    maes = [v["mae"] for v in out["per_client_test"].values()]
    out["test_summary"] = {
        "rmse_avg": float(np.mean(rmses)),
        "mae_avg": float(np.mean(maes)),
        "rmse_worst": float(np.max(rmses)),
        "mae_worst": float(np.max(maes)),
        "rmse_std": float(np.std(rmses)),
        "mae_std": float(np.std(maes)),
        "n_clients": int(len(rmses)),
    }
    out["comm_stats"] = {"uplink_mb": 0.0}
    out["backbone_meta"] = {
        "backbone_name": str(getattr(cfg, "backbone", "gru")),
        "note": "LocalOnly trains one independent model per client; parameter count is per-client.",
    }
    out["runtime_sec_total"] = float(time.perf_counter() - t0)
    return out


# =============================================================================
# Main experiment runner
# =============================================================================


def run_experiment(cfg: ExperimentConfig) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)

    requested_method = str(cfg.method).strip().lower()
    cfg = _apply_method_preset(cfg, requested_method)
    canonical_method = _canonicalize_method(requested_method)
    cfg = _apply_method_runtime_preset(cfg, canonical_method)
    plan = resolve_method_plan(canonical_method)

    random.seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    torch.manual_seed(int(cfg.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg.seed))

    device = torch.device(cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu")

    loaders, client_infos, data_meta = make_federated_dataloaders(
        data_root=cfg.data_root,
        dataset_name=cfg.dataset,
        variant=cfg.variant,
        batch_size=cfg.batch_size,
        shuffle_train=True,
        use_fed_masks=True,
        pad_to_global_max=True,
        strict_check=True,
    )

    clients = [ci.client_id for ci in client_infos]
    D = int(data_meta["D_max"])

    mask_cfg = MaskProtocolConfig(
        seed=int(cfg.seed),
        train_holdout_ratio=float(cfg.train_holdout_ratio),
        train_min_per_col=int(cfg.train_min_per_col),
        force_x_obs=True,
        train_loss_mask_mode="I_train",
        eval_mask_mode="auto",
        prevent_eval_leakage=True,
        strict_no_leakage=True,
    )

    if canonical_method == "localonly":
        metrics = run_localonly_suite(loaders, clients, cfg, device, mask_cfg, D, int(data_meta["effective_window"]))
        metrics["run_meta"] = {
            **asdict(cfg),
            "device_resolved": str(device),
            "D": D,
            "method": requested_method,
            "method_canonical": canonical_method,
            "participation_policy": plan.participation_policy,
            "stepalloc_policy": plan.stepalloc_policy,
            "aggregation_rule_resolved": plan.aggregation_rule,
            "local_train_method": plan.local_train_method,
            "backbone_meta": metrics.get("backbone_meta", {}),
        }
        save_tag = cfg.tag.strip() if cfg.tag.strip() else requested_method
        out_path = os.path.join(cfg.out_dir, f"metrics_{save_tag}_{cfg.dataset}_{cfg.variant}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved: {out_path}", flush=True)
        return

    window_length = int(data_meta["effective_window"])
    global_model = build_imputer(cfg, D, window_length, device=device).to(device)
    local_model_buffer = build_imputer(cfg, D, window_length, device=device).to(device)

    client_states = {cid: ClientState(cid, cfg) for cid in clients}
    fedopt_state = FedOptState(global_model, cfg) if plan.aggregation_rule == "fedopt" else None
    scaffold_state = ScaffoldState(global_model, clients, cfg) if plan.aggregation_rule == "scaffold" else None
    model_mb = sum(p.numel() * p.element_size() for p in global_model.parameters()) / (1024**2)

    metrics: Dict[str, Any] = {
        "run_meta": {
            **asdict(cfg),
            "device_resolved": str(device),
            "D": D,
            "method": requested_method,
            "method_canonical": canonical_method,
            "participation_policy": plan.participation_policy,
            "stepalloc_policy": plan.stepalloc_policy,
            "aggregation_rule_resolved": plan.aggregation_rule,
            "local_train_method": plan.local_train_method,
            "backbone_meta": describe_backbone(global_model),
        },
        "global_curve": [],
        "round_stats": [],
        "audit_global_test": {},
        "per_client_test": {},
        "comm_stats": {"uplink_mb": 0.0},
        "allocation_trace": {},
        "participation": {},
    }

    t0_total = time.perf_counter()
    selected_rounds = {cid: 0 for cid in clients}
    trained_rounds = {cid: 0 for cid in clients}  # valid-return rounds
    failed_rounds = {cid: 0 for cid in clients}
    failure_totals = {
        "selected_clients": 0,
        "valid_returns": 0,
        "failed_returns": 0,
        "assigned_steps": 0,
        "returned_assigned_steps": 0,
        "wasted_assigned_steps": 0,
        "empty_return_rounds": 0,
    }

    best_checkpoint_state: Optional[Dict[str, torch.Tensor]] = None
    best_checkpoint_round: Optional[int] = None
    best_checkpoint_val: Optional[Dict[str, float]] = None
    best_checkpoint_score = float("inf")

    eval_every = int(max(1, cfg.eval_every))
    eval_split = str(cfg.eval_split).strip().lower()
    if eval_split not in ("val", "test"):
        eval_split = "val"

    for rnd in range(1, int(cfg.rounds) + 1):
        active_cids, part_info = select_active_clients(clients, client_states, cfg, rnd, plan.participation_policy)
        for cid in active_cids:
            selected_rounds[cid] += 1

        local_step_alloc, alloc_meta = allocate_local_steps(active_cids, client_states, cfg, plan.stepalloc_policy, part_info)
        failed_cids, failure_info = sample_nonreturn_clients(active_cids, client_states, cfg, rnd, part_info)

        client_packages: List[ClientUpdatePackage] = []
        scaffold_packages: List[Dict[str, Dict[str, torch.Tensor]]] = []
        round_traces: Dict[str, Any] = {}

        for cid in active_cids:
            local_model_buffer.load_state_dict(global_model.state_dict())
            assigned_steps = int(local_step_alloc.get(cid, int(cfg.local_steps)))
            omega_before = float(client_states[cid].omega)
            fail_prob = float(failure_info.get("prob_by_client", {}).get(int(cid), 0.0))
            failed_to_return = int(cid) in failed_cids

            if failed_to_return:
                failed_rounds[cid] += 1
                aux_failed: Dict[str, Any] = {
                    "selected": True,
                    "valid_return": False,
                    "failed_to_return": True,
                    "failure_mode": str(failure_info.get("failure_mode", "none")),
                    "failure_probability": float(fail_prob),
                    "failure_uniform_draw": failure_info.get("uniform_by_client", {}).get(int(cid)),
                    "omega_before": float(omega_before),
                    "omega_after": float(client_states[cid].omega),
                    "assigned_local_steps": int(assigned_steps),
                    "effective_steps": 0,
                    "avg_loss": None,
                    "loss_history": float(client_states[cid].loss_history),
                }
                # Optional runtime accounting: execute the local computation and
                # discard it. The default is False to keep failure sweeps tractable;
                # assigned/wasted budget is still recorded exactly.
                if _bool_cfg(getattr(cfg, "failure_execute_training", False)):
                    if plan.local_train_method in ("fedavg", "fedprox"):
                        _, discarded_loss, discarded_steps, _discard_aux = train_local_baseline(
                            local_model_buffer,
                            loaders[cid]["train"],
                            client_states[cid],
                            cfg,
                            device,
                            mask_cfg,
                            method=plan.local_train_method,
                            local_steps_override=assigned_steps,
                            update_score=False,
                        )
                        aux_failed["discarded_local_training_executed"] = True
                        aux_failed["discarded_avg_loss"] = float(discarded_loss)
                        aux_failed["discarded_effective_steps"] = int(discarded_steps)
                    else:
                        aux_failed["discarded_local_training_executed"] = False
                round_traces[str(cid)] = aux_failed
                continue

            if plan.local_train_method in ("fedavg", "fedprox"):
                update_sd, avg_loss, eff_steps, aux = train_local_baseline(
                    local_model_buffer,
                    loaders[cid]["train"],
                    client_states[cid],
                    cfg,
                    device,
                    mask_cfg,
                    method=plan.local_train_method,
                    local_steps_override=assigned_steps,
                    update_score=bool(plan.update_score),
                )
            elif plan.local_train_method == "scaffold":
                update_sd, avg_loss, eff_steps, aux, scaffold_pkg = train_local_scaffold(
                    local_model_buffer,
                    loaders[cid]["train"],
                    client_states[cid],
                    cfg,
                    device,
                    mask_cfg,
                    scaffold_state=scaffold_state,  # type: ignore[arg-type]
                    local_steps_override=assigned_steps,
                )
                if eff_steps > 0:
                    scaffold_packages.append(scaffold_pkg)
            else:
                raise RuntimeError(f"Unexpected local_train_method: {plan.local_train_method}")

            aux = dict(aux)
            aux["selected"] = True
            aux["valid_return"] = True
            aux["failed_to_return"] = False
            aux["failure_mode"] = str(failure_info.get("failure_mode", "none"))
            aux["failure_probability"] = float(fail_prob)
            aux["failure_uniform_draw"] = failure_info.get("uniform_by_client", {}).get(int(cid))
            aux["omega_before"] = float(omega_before)
            aux["omega_after"] = float(client_states[cid].omega)
            aux["assigned_local_steps"] = int(assigned_steps)
            aux["avg_loss"] = float(avg_loss)
            round_traces[str(cid)] = aux

            if eff_steps <= 0:
                continue

            client_packages.append(
                ClientUpdatePackage(
                    cid=int(cid),
                    state_dict=update_sd,
                    avg_loss=float(avg_loss),
                    eff_steps=int(eff_steps),
                    assigned_steps=int(assigned_steps),
                    omega=float(client_states[cid].omega),
                )
            )
            trained_rounds[cid] += 1

        aggregation_info: Dict[str, Any] = {
            "aggregation_rule": str(cfg.aggregation_rule).strip().lower(),
            "ref_steps": None,
            "client_ratio_raw": {},
            "client_scale_powered": {},
            "client_scales": {},
            "client_base_weights": {},
            "client_effective_weights": {},
            "ena_reference_mode": "",
            "ena_alpha": None,
            "ena_clip_min": None,
            "ena_clip_max": None,
        }

        if plan.aggregation_rule == "fedopt":
            if len(client_packages) > 0 and fedopt_state is not None:
                global_state = _clone_state_dict_cpu(global_model)
                deltas = [{k: pkg.state_dict[k] - global_state[k] for k in global_state.keys()} for pkg in client_packages]
                weights = np.ones((len(deltas),), dtype=np.float64) / max(1, len(deltas))
                fedopt_state.step(global_model, deltas, weights)
                aggregation_info["client_scales"] = {int(pkg.cid): 1.0 for pkg in client_packages}
                aggregation_info["client_base_weights"] = {int(pkg.cid): float(1.0 / max(1, len(client_packages))) for pkg in client_packages}
                aggregation_info["client_effective_weights"] = dict(aggregation_info["client_base_weights"])
        elif plan.aggregation_rule == "scaffold":
            if scaffold_state is not None:
                aggregate_scaffold(global_model, scaffold_state, scaffold_packages, total_client_num=len(clients))
                aggregation_info["client_scales"] = {int(pkg.cid): 1.0 for pkg in client_packages}
                aggregation_info["client_base_weights"] = {int(pkg.cid): float(1.0 / max(1, len(client_packages))) for pkg in client_packages}
                aggregation_info["client_effective_weights"] = dict(aggregation_info["client_base_weights"])
        else:
            aggregation_info = aggregate_client_packages(
                global_model=global_model,
                client_packages=client_packages,
                aggregation_rule=str(cfg.aggregation_rule),
                q_param=float(cfg.q_param),
                ena_reference_mode=str(cfg.ena_reference_mode),
                configured_local_steps=int(cfg.local_steps),
                ena_eps=float(cfg.ena_eps),
                ena_alpha=float(cfg.ena_alpha),
                ena_clip_min=cfg.ena_clip_min,
                ena_clip_max=cfg.ena_clip_max,
            )

        for pkg in client_packages:
            cid_key = str(pkg.cid)
            if cid_key not in round_traces:
                round_traces[cid_key] = {"selected": True}
            round_traces[cid_key]["aggregation_rule"] = str(aggregation_info.get("aggregation_rule", cfg.aggregation_rule))
            round_traces[cid_key]["ena_reference_mode"] = str(aggregation_info.get("ena_reference_mode", ""))
            round_traces[cid_key]["aggregation_ref_steps"] = aggregation_info.get("ref_steps")
            round_traces[cid_key]["ena_alpha"] = aggregation_info.get("ena_alpha")
            round_traces[cid_key]["ena_clip_min"] = aggregation_info.get("ena_clip_min")
            round_traces[cid_key]["ena_clip_max"] = aggregation_info.get("ena_clip_max")
            round_traces[cid_key]["ena_ratio_raw"] = float(aggregation_info.get("client_ratio_raw", {}).get(pkg.cid, 1.0))
            round_traces[cid_key]["ena_scale_powered"] = float(aggregation_info.get("client_scale_powered", {}).get(pkg.cid, 1.0))
            round_traces[cid_key]["ena_scale"] = float(aggregation_info.get("client_scales", {}).get(pkg.cid, 1.0))
            round_traces[cid_key]["aggregation_base_weight"] = float(
                aggregation_info.get("client_base_weights", {}).get(pkg.cid, 0.0)
            )
            round_traces[cid_key]["aggregation_effective_weight"] = float(
                aggregation_info.get("client_effective_weights", {}).get(pkg.cid, 0.0)
            )

        metrics["comm_stats"]["uplink_mb"] += float(len(client_packages)) * float(model_mb)

        assigned_sum = int(sum(int(v) for v in local_step_alloc.values()))
        effective_sum = int(sum(int(pkg.eff_steps) for pkg in client_packages))
        failed_assigned_sum = int(sum(int(local_step_alloc.get(cid, 0)) for cid in failed_cids))
        returned_assigned_sum = int(sum(int(pkg.assigned_steps) for pkg in client_packages))
        valid_return_rate_round = float(len(client_packages) / max(1, len(active_cids)))
        wasted_step_ratio_round = float(failed_assigned_sum / max(1, assigned_sum))

        failure_totals["selected_clients"] += int(len(active_cids))
        failure_totals["valid_returns"] += int(len(client_packages))
        failure_totals["failed_returns"] += int(len(failed_cids))
        failure_totals["assigned_steps"] += int(assigned_sum)
        failure_totals["returned_assigned_steps"] += int(returned_assigned_sum)
        failure_totals["wasted_assigned_steps"] += int(failed_assigned_sum)
        if len(client_packages) == 0 and len(active_cids) > 0:
            failure_totals["empty_return_rounds"] += 1

        round_stat = {
            "round": int(rnd),
            "selected": int(len(active_cids)),
            "trained": int(len(client_packages)),
            "valid_returns": int(len(client_packages)),
            "failed_returns": int(len(failed_cids)),
            "valid_return_rate": float(valid_return_rate_round),
            "failed_assigned_steps_sum": int(failed_assigned_sum),
            "returned_assigned_steps_sum": int(returned_assigned_sum),
            "wasted_assigned_step_ratio": float(wasted_step_ratio_round),
            "failure_mode": str(failure_info.get("failure_mode", "none")),
            "failure_rate": float(failure_info.get("failure_rate", 0.0)),
            "participation_mode": str(part_info.get("mode", "all")),
            "participation_threshold": float(part_info.get("threshold", -float("inf"))),
            "full_participation": int(bool(part_info.get("full", False))),
            "assigned_local_steps_sum": int(assigned_sum),
            "effective_local_steps_sum": int(effective_sum),
            "allocation_mode": str(alloc_meta.get("allocation_mode", "fixed")),
            "stepalloc_total_budget": int(alloc_meta.get("total_budget", len(active_cids) * int(cfg.local_steps))),
            "stepalloc_min_steps": int(alloc_meta.get("min_steps", cfg.local_steps)),
            "stepalloc_max_steps": int(alloc_meta.get("max_steps", cfg.local_steps)),
            "stepalloc_score_power": float(alloc_meta.get("score_power", 1.0)),
            "aggregation_rule": str(aggregation_info.get("aggregation_rule", cfg.aggregation_rule)),
            "ena_reference_mode": str(aggregation_info.get("ena_reference_mode", "")),
            "ena_alpha": aggregation_info.get("ena_alpha"),
            "ena_clip_min": aggregation_info.get("ena_clip_min"),
            "ena_clip_max": aggregation_info.get("ena_clip_max"),
            "aggregation_ref_steps": aggregation_info.get("ref_steps"),
        }
        metrics["round_stats"].append(round_stat)
        metrics["allocation_trace"][str(rnd)] = round_traces

        if (rnd % eval_every == 0) or (rnd == int(cfg.rounds)):
            summary, _ = eval_over_clients(global_model, loaders, clients, device, mask_cfg, split=eval_split)  # type: ignore[arg-type]
            elapsed = float(time.perf_counter() - t0_total)
            entry = {
                "round": int(rnd),
                "split": eval_split,
                "rmse_avg": float(summary["rmse_macro"]),
                "mae_avg": float(summary["mae_macro"]),
                "rmse_worst": float(summary["rmse_worst"]),
                "mae_worst": float(summary["mae_worst"]),
                "rmse_std": float(summary["rmse_std"]),
                "mae_std": float(summary["mae_std"]),
                "rmse_micro": float(summary["rmse_micro"]),
                "mae_micro": float(summary["mae_micro"]),
                "uplink_mb": float(metrics["comm_stats"]["uplink_mb"]),
                "runtime_sec": elapsed,
                "selected": int(len(active_cids)),
                "trained": int(len(client_packages)),
                "valid_returns": int(len(client_packages)),
                "failed_returns": int(len(failed_cids)),
                "valid_return_rate": float(valid_return_rate_round),
                "failed_assigned_steps_sum": int(failed_assigned_sum),
                "returned_assigned_steps_sum": int(returned_assigned_sum),
                "wasted_assigned_step_ratio": float(wasted_step_ratio_round),
                "failure_mode": str(failure_info.get("failure_mode", "none")),
                "failure_rate": float(failure_info.get("failure_rate", 0.0)),
                "participation_mode": str(part_info.get("mode", "all")),
                "allocation_mode": str(alloc_meta.get("allocation_mode", "fixed")),
                "aggregation_rule": str(aggregation_info.get("aggregation_rule", cfg.aggregation_rule)),
                "ena_reference_mode": str(aggregation_info.get("ena_reference_mode", "")),
                "ena_alpha": aggregation_info.get("ena_alpha"),
                "ena_clip_min": aggregation_info.get("ena_clip_min"),
                "ena_clip_max": aggregation_info.get("ena_clip_max"),
                "aggregation_ref_steps": aggregation_info.get("ref_steps"),
                "assigned_local_steps_sum": int(assigned_sum),
                "effective_local_steps_sum": int(effective_sum),
            }
            metrics["global_curve"].append(entry)

            if str(cfg.checkpoint_selection).strip().lower() == "best_val_rmse_avg":
                if eval_split == "val":
                    val_summary = summary
                else:
                    val_summary, _ = eval_over_clients(global_model, loaders, clients, device, mask_cfg, split="val")
                val_score = float(val_summary["rmse_macro"])
                if np.isfinite(val_score) and val_score < best_checkpoint_score:
                    best_checkpoint_score = val_score
                    best_checkpoint_round = int(rnd)
                    best_checkpoint_state = _clone_state_dict_cpu(global_model)
                    best_checkpoint_val = {
                        "rmse_avg": float(val_summary["rmse_macro"]),
                        "mae_avg": float(val_summary["mae_macro"]),
                        "rmse_worst": float(val_summary["rmse_worst"]),
                        "mae_worst": float(val_summary["mae_worst"]),
                        "rmse_std": float(val_summary["rmse_std"]),
                        "mae_std": float(val_summary["mae_std"]),
                        "rmse_micro": float(val_summary["rmse_micro"]),
                        "mae_micro": float(val_summary["mae_micro"]),
                    }

            print(
                f"[R{rnd:02d}] {eval_split.upper()} RMSE_avg={entry['rmse_avg']:.4f} "
                f"MAE_avg={entry['mae_avg']:.4f} RMSE_worst={entry['rmse_worst']:.4f} | "
                f"active={len(client_packages)} sel={len(active_cids)} "
                f"assigned={entry['assigned_local_steps_sum']} effective={entry['effective_local_steps_sum']} "
                f"agg={entry['aggregation_rule']}",
                flush=True,
            )

    final_summary_test, final_per_test = eval_over_clients(global_model, loaders, clients, device, mask_cfg, split="test")
    metrics["final_round_global_test"] = {
        "rmse_avg": float(final_summary_test["rmse_macro"]),
        "mae_avg": float(final_summary_test["mae_macro"]),
        "rmse_worst": float(final_summary_test["rmse_worst"]),
        "mae_worst": float(final_summary_test["mae_worst"]),
        "rmse_std": float(final_summary_test["rmse_std"]),
        "mae_std": float(final_summary_test["mae_std"]),
        "rmse_micro": float(final_summary_test["rmse_micro"]),
        "mae_micro": float(final_summary_test["mae_micro"]),
        "n_clients": int(final_summary_test["n_clients"]),
    }

    selected_checkpoint_mode = str(cfg.checkpoint_selection).strip().lower()
    if selected_checkpoint_mode == "best_val_rmse_avg" and best_checkpoint_state is not None:
        global_model.load_state_dict(best_checkpoint_state)
        selected_checkpoint = {
            "selection": "best_val_rmse_avg",
            "round": int(best_checkpoint_round) if best_checkpoint_round is not None else None,
            "validation": best_checkpoint_val,
        }
    else:
        selected_checkpoint = {
            "selection": "last",
            "round": int(cfg.rounds),
            "validation": None,
        }

    summary_test, per_test = eval_over_clients(global_model, loaders, clients, device, mask_cfg, split="test")
    metrics["selected_checkpoint"] = selected_checkpoint
    metrics["audit_global_test"] = {
        "rmse_avg": float(summary_test["rmse_macro"]),
        "mae_avg": float(summary_test["mae_macro"]),
        "rmse_worst": float(summary_test["rmse_worst"]),
        "mae_worst": float(summary_test["mae_worst"]),
        "rmse_std": float(summary_test["rmse_std"]),
        "mae_std": float(summary_test["mae_std"]),
        "rmse_micro": float(summary_test["rmse_micro"]),
        "mae_micro": float(summary_test["mae_micro"]),
        "n_clients": int(summary_test["n_clients"]),
    }
    metrics["per_client_test"] = per_test
    metrics["participation"] = {
        str(cid): {
            "selected_rounds": int(selected_rounds[cid]),
            "valid_return_rounds": int(trained_rounds[cid]),
            "trained_rounds": int(trained_rounds[cid]),
            "failed_return_rounds": int(failed_rounds[cid]),
        }
        for cid in clients
    }
    metrics["failure_stats"] = {
        **failure_totals,
        "failure_mode": str(getattr(cfg, "failure_mode", "none")),
        "failure_rate": float(getattr(cfg, "failure_rate", 0.0) or 0.0),
        "failure_qmax": float(getattr(cfg, "failure_qmax", 0.8) or 0.8),
        "valid_return_rate": float(failure_totals["valid_returns"] / max(1, failure_totals["selected_clients"])),
        "failed_return_rate": float(failure_totals["failed_returns"] / max(1, failure_totals["selected_clients"])),
        "returned_assigned_step_ratio": float(failure_totals["returned_assigned_steps"] / max(1, failure_totals["assigned_steps"])),
        "wasted_assigned_step_ratio": float(failure_totals["wasted_assigned_steps"] / max(1, failure_totals["assigned_steps"])),
    }
    metrics["runtime_sec_total"] = float(time.perf_counter() - t0_total)

    save_tag = cfg.tag.strip() if cfg.tag.strip() else requested_method
    out_path = os.path.join(cfg.out_dir, f"metrics_{save_tag}_{cfg.dataset}_{cfg.variant}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved: {out_path}", flush=True)


# =============================================================================
# CLI
# =============================================================================


def _build_cfg_from_args(args: argparse.Namespace) -> ExperimentConfig:
    cfg = ExperimentConfig()
    for k, v in vars(args).items():
        if v is None:
            continue
        if hasattr(cfg, k):
            if k in {"failure_apply_full_rounds", "failure_execute_training"}:
                setattr(cfg, k, _bool_cfg(v))
            else:
                setattr(cfg, k, v)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser("CoRA core experiment runner")

    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)

    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)

    parser.add_argument("--method", type=str, default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--local_steps", type=int, default=None)

    parser.add_argument("--backbone", type=str, default=None, choices=["gru", "saits", "csdi"])
    parser.add_argument("--hidden_size", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)

    parser.add_argument("--saits_d_model", type=int, default=None)
    parser.add_argument("--saits_d_inner", type=int, default=None)
    parser.add_argument("--saits_n_groups", type=int, default=None)
    parser.add_argument("--saits_n_group_inner_layers", type=int, default=None)
    parser.add_argument("--saits_n_head", type=int, default=None)
    parser.add_argument("--saits_d_k", type=int, default=None)
    parser.add_argument("--saits_d_v", type=int, default=None)
    parser.add_argument("--saits_dropout", type=float, default=None)
    parser.add_argument("--saits_input_with_mask", type=int, default=None)
    parser.add_argument("--saits_diagonal_attention_mask", type=int, default=None)
    parser.add_argument("--saits_param_sharing_strategy", type=str, default=None)
    parser.add_argument("--saits_mit", type=int, default=None)

    parser.add_argument("--csdi_layers", type=int, default=None)
    parser.add_argument("--csdi_channels", type=int, default=None)
    parser.add_argument("--csdi_nheads", type=int, default=None)
    parser.add_argument("--csdi_diffusion_embedding_dim", type=int, default=None)
    parser.add_argument("--csdi_beta_start", type=float, default=None)
    parser.add_argument("--csdi_beta_end", type=float, default=None)
    parser.add_argument("--csdi_num_steps", type=int, default=None)
    parser.add_argument("--csdi_schedule", type=str, default=None)
    parser.add_argument("--csdi_timeemb", type=int, default=None)
    parser.add_argument("--csdi_featureemb", type=int, default=None)
    parser.add_argument("--csdi_eval_samples", type=int, default=None)
    parser.add_argument("--csdi_is_linear", type=int, default=None)

    parser.add_argument("--mu", type=float, default=None)
    parser.add_argument("--q_param", type=float, default=None)

    parser.add_argument("--eval_every", type=int, default=None)
    parser.add_argument("--eval_split", type=str, default=None)

    parser.add_argument("--beta_hardness", type=float, default=None)
    parser.add_argument("--h0", type=float, default=None)
    parser.add_argument("--score_floor", type=float, default=None)
    parser.add_argument("--rho", type=float, default=None)
    parser.add_argument("--T_part", type=int, default=None)
    parser.add_argument("--T_refresh", type=int, default=None)
    parser.add_argument("--K_min", type=int, default=None)

    parser.add_argument("--stepalloc_min_steps", type=int, default=None)
    parser.add_argument("--stepalloc_max_steps", type=int, default=None)
    parser.add_argument("--stepalloc_power", type=float, default=None)

    parser.add_argument("--failure_mode", type=str, default=None)
    parser.add_argument("--failure_rate", type=float, default=None)
    parser.add_argument("--failure_qmax", type=float, default=None)
    parser.add_argument("--failure_apply_full_rounds", type=int, default=None)
    parser.add_argument("--failure_execute_training", type=int, default=None)
    parser.add_argument("--checkpoint_selection", type=str, default=None)

    parser.add_argument("--aggregation_rule", type=str, default=None)
    parser.add_argument("--ena_reference_mode", type=str, default=None)
    parser.add_argument("--ena_eps", type=float, default=None)
    parser.add_argument("--ena_alpha", type=float, default=None)
    parser.add_argument("--ena_clip_min", type=float, default=None)
    parser.add_argument("--ena_clip_max", type=float, default=None)

    parser.add_argument("--fedopt_type", type=str, default=None)
    parser.add_argument("--fedopt_beta1", type=float, default=None)
    parser.add_argument("--fedopt_beta2", type=float, default=None)
    parser.add_argument("--fedopt_server_lr", type=float, default=None)
    parser.add_argument("--fedopt_tau", type=float, default=None)

    parser.add_argument("--scaffold_global_lr", type=float, default=None)

    parser.add_argument("--train_holdout_ratio", type=float, default=None)
    parser.add_argument("--train_min_per_col", type=int, default=None)

    # deprecated / ignored knobs kept for archival CLI compatibility
    parser.add_argument("--lambda_warmup_rounds", type=int, default=None)
    parser.add_argument("--lambda_min", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--p_asm", type=float, default=None)
    parser.add_argument("--tau_vis", type=int, default=None)
    parser.add_argument("--psi", type=str, default=None)
    parser.add_argument("--psi_power", type=float, default=None)
    parser.add_argument("--fixed_coverage_ratio", type=float, default=None)
    parser.add_argument("--coverage_floor_steps", type=int, default=None)

    cfg = _build_cfg_from_args(parser.parse_args())
    run_experiment(cfg)


if __name__ == "__main__":
    main()
