from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from main_experiment_runner import (  # noqa: E402
    ExperimentConfig,
    _bounded_integer_step_allocation,
    resolve_method_plan,
    sample_nonreturn_clients,
    ClientState,
)
from backbone_models import build_imputer, model_impute, model_training_loss  # noqa: E402


def check_defaults() -> None:
    cfg = ExperimentConfig()
    assert cfg.rho == 0.6
    assert cfg.T_part == 15
    assert cfg.T_refresh == 5
    assert cfg.K_min == 5
    assert cfg.score_floor == 1e-9
    assert cfg.stepalloc_min_steps == 15
    assert cfg.stepalloc_max_steps == 25
    assert cfg.backbone == "gru"
    assert resolve_method_plan("cora_stepalloc").aggregation_rule == "ena"


def check_stepalloc_budget() -> None:
    active = list(range(6))
    weights = np.array([1, 2, 3, 4, 5, 6], dtype=float)
    alloc = _bounded_integer_step_allocation(active, weights, total_budget=120, min_steps=15, max_steps=25)
    assert sum(alloc.values()) == 120
    assert min(alloc.values()) >= 15
    assert max(alloc.values()) <= 25


def check_score_correlated_failure() -> None:
    cfg = ExperimentConfig(seed=7, failure_mode="score_correlated", failure_rate=0.2, failure_qmax=0.8)
    clients = list(range(5))
    states = {c: ClientState(c, cfg) for c in clients}
    for c in clients:
        states[c].omega = 1.0 + c
    failed, meta = sample_nonreturn_clients(clients, states, cfg, round_idx=3, part_info={"full": False})
    probs = meta["prob_by_client"]
    assert probs[4] >= probs[0]
    assert set(failed).issubset(set(clients))


def _dummy_cfg(backbone: str) -> SimpleNamespace:
    base = ExperimentConfig(device="cpu", backbone=backbone)
    # Reduce CSDI for smoke tests only.
    base.csdi_num_steps = 3
    base.csdi_layers = 1
    base.csdi_channels = 8
    base.csdi_nheads = 2
    base.csdi_diffusion_embedding_dim = 16
    base.csdi_timeemb = 8
    base.csdi_featureemb = 4
    base.csdi_eval_samples = 1
    base.saits_d_model = 16
    base.saits_d_inner = 32
    base.saits_n_head = 2
    base.saits_d_k = 8
    base.saits_d_v = 8
    base.saits_n_groups = 1
    return base


def check_backbone_adapters() -> None:
    torch.manual_seed(0)
    B, L, D = 2, 8, 3
    x_true = torch.randn(B, L, D)
    m_in = (torch.rand(B, L, D) > 0.4).float()
    loss_mask = ((torch.rand(B, L, D) > 0.7).float() * (1.0 - m_in)).float()
    # Ensure at least one target position.
    if float(loss_mask.sum()) <= 0:
        loss_mask[0, 0, 0] = 1.0
        m_in[0, 0, 0] = 0.0
    x_in = x_true * m_in
    for backbone in ["gru", "saits", "csdi"]:
        cfg = _dummy_cfg(backbone)
        model = build_imputer(cfg, D, L, device=torch.device("cpu"))
        loss = model_training_loss(model, x_true, x_in, m_in, loss_mask)
        assert torch.isfinite(loss).item(), backbone
        loss.backward()
        rec = model_impute(model, x_in, m_in)
        assert tuple(rec.shape) == (B, L, D), backbone
        assert torch.isfinite(rec).all().item(), backbone


def main() -> None:
    check_defaults()
    check_stepalloc_budget()
    check_score_correlated_failure()
    check_backbone_adapters()
    print("[OK] revision code validation passed")


if __name__ == "__main__":
    main()
