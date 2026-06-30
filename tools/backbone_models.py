from __future__ import annotations

"""Backbone adapters for the FGCS CoRA revision experiments.

The CoRA server-side control logic only requires each selected client to return
one scalar training loss and one model delta.  This module provides a common
adapter interface for the GRU backbone used in the original experiments and for
SAITS/CSDI backbone checks requested during major revision.

The SAITS and CSDI implementations are vendored under ``tools/third_party`` and
kept close to their official open-source code.  The wrappers below adapt their
input/output conventions to the leakage-free CoRA masked-reconstruction data
protocol used by ``ur_fedimpute_dataio.py``.
"""

import copy
import os
import sys
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SAITS_ROOT = os.path.join(CURRENT_DIR, "third_party", "saits_official")
CSDI_ROOT = os.path.join(CURRENT_DIR, "third_party", "csdi_official")
for _p in [SAITS_ROOT, CSDI_ROOT]:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)


class GRUImputer(nn.Module):
    """GRU imputation backbone used by the original CoRA experiments."""

    backbone_name = "gru"

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


class SAITSImputer(nn.Module):
    """Adapter around the official SAITS architecture.

    SAITS expects a dictionary with keys ``X`` and ``missing_mask``.  In CoRA,
    ``X`` is the leakage-free input tensor ``X_in`` and ``missing_mask`` is the
    visibility mask ``M_in``.  The returned imputation tensor has shape
    ``[batch, time, feature]`` and can be evaluated under the same masks as the
    GRU backbone.
    """

    backbone_name = "saits"

    def __init__(
        self,
        d_in: int,
        window_length: int,
        d_model: int = 64,
        d_inner: int = 128,
        n_groups: int = 2,
        n_group_inner_layers: int = 1,
        n_head: int = 4,
        d_k: int = 16,
        d_v: int = 16,
        dropout: float = 0.1,
        input_with_mask: bool = True,
        diagonal_attention_mask: bool = True,
        param_sharing_strategy: str = "between_group",
        mit: bool = True,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        try:
            from modeling.saits import SAITS  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "Cannot import the vendored SAITS implementation. Expected files under "
                f"{SAITS_ROOT!r}. Original error: {e}"
            ) from e

        self.d_in = int(d_in)
        self.window_length = int(window_length)
        self.model = SAITS(
            n_groups=int(n_groups),
            n_group_inner_layers=int(n_group_inner_layers),
            d_time=int(window_length),
            d_feature=int(d_in),
            d_model=int(d_model),
            d_inner=int(d_inner),
            n_head=int(n_head),
            d_k=int(d_k),
            d_v=int(d_v),
            dropout=float(dropout),
            input_with_mask=bool(input_with_mask),
            diagonal_attention_mask=bool(diagonal_attention_mask),
            param_sharing_strategy=str(param_sharing_strategy),
            MIT=bool(mit),
            device=device if device is not None else torch.device("cpu"),
        )

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        # Official SAITS modules store a device used for the diagonal attention
        # mask.  Keep it synchronized with the actual input device.
        if hasattr(self.model, "device"):
            self.model.device = x.device
        for stack_name in ["layer_stack_for_first_block", "layer_stack_for_second_block"]:
            stack = getattr(self.model, stack_name, None)
            if stack is not None:
                for layer in stack:
                    if hasattr(layer, "device"):
                        layer.device = x.device
        inputs = {"X": x, "missing_mask": m}
        imputed, _parts = self.model.impute(inputs)
        return imputed


class CSDIImputer(nn.Module):
    """Adapter around the official CSDI diffusion network.

    The official CSDI training objective predicts diffusion noise on held-out
    observed targets.  The adapter uses the same loss while taking the CoRA
    leakage-free masks as input: ``M_in`` is the conditional mask and
    ``loss_mask`` is the diffusion target mask.  At evaluation time it returns
    the mean of ``eval_samples`` reverse-diffusion samples.  By default the
    configuration is intentionally lightweight; all key parameters are exposed
    via CLI flags in ``main_experiment_runner.py``.
    """

    backbone_name = "csdi"

    def __init__(
        self,
        d_in: int,
        window_length: int,
        layers: int = 2,
        channels: int = 32,
        nheads: int = 4,
        diffusion_embedding_dim: int = 64,
        beta_start: float = 1e-4,
        beta_end: float = 0.5,
        num_steps: int = 20,
        schedule: str = "quad",
        timeemb: int = 64,
        featureemb: int = 16,
        eval_samples: int = 1,
        is_linear: bool = False,
    ):
        super().__init__()
        try:
            from diff_models import diff_CSDI  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "Cannot import the vendored CSDI implementation. Expected files under "
                f"{CSDI_ROOT!r}. Original error: {e}"
            ) from e

        self.target_dim = int(d_in)
        self.window_length = int(window_length)
        self.emb_time_dim = int(timeemb)
        self.emb_feature_dim = int(featureemb)
        self.eval_samples = int(max(1, eval_samples))
        self.num_steps = int(max(1, num_steps))
        self.is_unconditional = False
        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim + 1
        self.embed_layer = nn.Embedding(num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim)

        config_diff = {
            "layers": int(layers),
            "channels": int(channels),
            "nheads": int(nheads),
            "diffusion_embedding_dim": int(diffusion_embedding_dim),
            "beta_start": float(beta_start),
            "beta_end": float(beta_end),
            "num_steps": int(self.num_steps),
            "schedule": str(schedule),
            "is_linear": bool(is_linear),
            "side_dim": int(self.emb_total_dim),
        }
        self.diffmodel = diff_CSDI(config_diff, inputdim=2)

        if str(schedule) == "quad":
            beta = np.linspace(float(beta_start) ** 0.5, float(beta_end) ** 0.5, self.num_steps) ** 2
        elif str(schedule) == "linear":
            beta = np.linspace(float(beta_start), float(beta_end), self.num_steps)
        else:
            raise ValueError(f"Unsupported CSDI schedule: {schedule}")
        alpha_hat = 1.0 - beta
        alpha = np.cumprod(alpha_hat)
        self.register_buffer("beta", torch.tensor(beta, dtype=torch.float32), persistent=False)
        self.register_buffer("alpha_hat", torch.tensor(alpha_hat, dtype=torch.float32), persistent=False)
        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32), persistent=False)
        self.register_buffer("alpha_torch", torch.tensor(alpha, dtype=torch.float32).view(-1, 1, 1), persistent=False)

    def _time_embedding(self, pos: torch.Tensor, d_model: int) -> torch.Tensor:
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model, device=pos.device, dtype=torch.float32)
        position = pos.unsqueeze(2).float()
        div_term = 1 / torch.pow(
            torch.tensor(10000.0, device=pos.device),
            torch.arange(0, d_model, 2, device=pos.device).float() / float(d_model),
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def _observed_tp(self, batch_size: int, length: int, device: torch.device) -> torch.Tensor:
        # Official CSDI uses explicit timepoints.  Our windowed benchmark has a
        # regular grid, so the local index is the appropriate shared time axis.
        return torch.arange(length, device=device, dtype=torch.float32).unsqueeze(0).expand(batch_size, -1)

    def _side_info(self, observed_tp: torch.Tensor, cond_mask: torch.Tensor) -> torch.Tensor:
        # cond_mask: [B, K, L]
        B, K, L = cond_mask.shape
        time_embed = self._time_embedding(observed_tp, self.emb_time_dim)  # [B,L,emb]
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)
        feature_embed = self.embed_layer(torch.arange(K, device=cond_mask.device))
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        side_info = torch.cat([time_embed, feature_embed], dim=-1).permute(0, 3, 2, 1)
        side_mask = cond_mask.unsqueeze(1)
        return torch.cat([side_info, side_mask], dim=1)

    def _set_input_to_diffmodel(self, noisy_data: torch.Tensor, observed_data: torch.Tensor, cond_mask: torch.Tensor) -> torch.Tensor:
        cond_obs = (cond_mask * observed_data).unsqueeze(1)
        noisy_target = ((1.0 - cond_mask) * noisy_data).unsqueeze(1)
        return torch.cat([cond_obs, noisy_target], dim=1)

    def training_loss(
        self,
        x_true: torch.Tensor,
        x_in: torch.Tensor,
        m_in: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Convert [B,L,D] to official CSDI convention [B,K,D_time].
        observed_data = x_true.permute(0, 2, 1).float()
        cond_mask = m_in.permute(0, 2, 1).float()
        target_mask = loss_mask.permute(0, 2, 1).float()
        observed_mask = torch.clamp(cond_mask + target_mask, 0.0, 1.0)
        B, K, L = observed_data.shape
        device = observed_data.device
        t = torch.randint(0, self.num_steps, [B], device=device)
        current_alpha = self.alpha_torch.to(device)[t]
        noise = torch.randn_like(observed_data)
        noisy_data = (current_alpha ** 0.5) * observed_data + ((1.0 - current_alpha) ** 0.5) * noise
        total_input = self._set_input_to_diffmodel(noisy_data, observed_data, cond_mask)
        side_info = self._side_info(self._observed_tp(B, L, device), cond_mask)
        predicted = self.diffmodel(total_input, side_info, t)
        # Use the externally supplied leakage-free target mask.  It is a subset
        # of observed_mask by construction in ur_fedimpute_dataio.
        target_mask = target_mask * observed_mask
        residual = (noise - predicted) * target_mask
        denom = torch.clamp(torch.sum(target_mask), min=1.0)
        return torch.sum(residual ** 2) / denom

    @torch.no_grad()
    def _reverse_sample(self, observed_data: torch.Tensor, cond_mask: torch.Tensor, n_samples: int) -> torch.Tensor:
        B, K, L = observed_data.shape
        device = observed_data.device
        side_info = self._side_info(self._observed_tp(B, L, device), cond_mask)
        imputed_samples = torch.zeros(B, int(n_samples), K, L, device=device)
        alpha_hat = self.alpha_hat.to(device)
        alpha = self.alpha.to(device)
        beta = self.beta.to(device)
        for i in range(int(n_samples)):
            current_sample = torch.randn_like(observed_data)
            for t_int in range(self.num_steps - 1, -1, -1):
                cond_obs = (cond_mask * observed_data).unsqueeze(1)
                noisy_target = ((1.0 - cond_mask) * current_sample).unsqueeze(1)
                diff_input = torch.cat([cond_obs, noisy_target], dim=1)
                t = torch.tensor([t_int], device=device, dtype=torch.long)
                predicted = self.diffmodel(diff_input, side_info, t)
                coeff1 = 1.0 / (alpha_hat[t_int] ** 0.5)
                coeff2 = (1.0 - alpha_hat[t_int]) / ((1.0 - alpha[t_int]) ** 0.5)
                current_sample = coeff1 * (current_sample - coeff2 * predicted)
                if t_int > 0:
                    sigma = (((1.0 - alpha[t_int - 1]) / (1.0 - alpha[t_int])) * beta[t_int]) ** 0.5
                    current_sample = current_sample + sigma * torch.randn_like(current_sample)
            imputed_samples[:, i] = current_sample.detach()
        return imputed_samples

    def impute_values(self, x_in: torch.Tensor, m_in: torch.Tensor, n_samples: Optional[int] = None) -> torch.Tensor:
        n = int(n_samples if n_samples is not None else self.eval_samples)
        observed_data = x_in.permute(0, 2, 1).float()
        cond_mask = m_in.permute(0, 2, 1).float()
        samples = self._reverse_sample(observed_data, cond_mask, n_samples=max(1, n))
        sample_mean = samples.mean(dim=1)
        completed = cond_mask * observed_data + (1.0 - cond_mask) * sample_mean
        return completed.permute(0, 2, 1)

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        return self.impute_values(x, m, n_samples=self.eval_samples)


def build_imputer(cfg: Any, d_in: int, window_length: int, device: Optional[torch.device] = None) -> nn.Module:
    backbone = str(getattr(cfg, "backbone", "gru") or "gru").strip().lower()
    if backbone == "gru":
        return GRUImputer(d_in=d_in, hidden_size=int(cfg.hidden_size), num_layers=int(cfg.num_layers), dropout=float(cfg.dropout))
    if backbone == "saits":
        return SAITSImputer(
            d_in=d_in,
            window_length=window_length,
            d_model=int(getattr(cfg, "saits_d_model", 64)),
            d_inner=int(getattr(cfg, "saits_d_inner", 128)),
            n_groups=int(getattr(cfg, "saits_n_groups", 2)),
            n_group_inner_layers=int(getattr(cfg, "saits_n_group_inner_layers", 1)),
            n_head=int(getattr(cfg, "saits_n_head", 4)),
            d_k=int(getattr(cfg, "saits_d_k", 16)),
            d_v=int(getattr(cfg, "saits_d_v", 16)),
            dropout=float(getattr(cfg, "saits_dropout", getattr(cfg, "dropout", 0.1))),
            input_with_mask=bool(int(getattr(cfg, "saits_input_with_mask", 1))),
            diagonal_attention_mask=bool(int(getattr(cfg, "saits_diagonal_attention_mask", 1))),
            param_sharing_strategy=str(getattr(cfg, "saits_param_sharing_strategy", "between_group")),
            mit=bool(int(getattr(cfg, "saits_mit", 1))),
            device=device,
        )
    if backbone == "csdi":
        return CSDIImputer(
            d_in=d_in,
            window_length=window_length,
            layers=int(getattr(cfg, "csdi_layers", 2)),
            channels=int(getattr(cfg, "csdi_channels", 32)),
            nheads=int(getattr(cfg, "csdi_nheads", 4)),
            diffusion_embedding_dim=int(getattr(cfg, "csdi_diffusion_embedding_dim", 64)),
            beta_start=float(getattr(cfg, "csdi_beta_start", 1e-4)),
            beta_end=float(getattr(cfg, "csdi_beta_end", 0.5)),
            num_steps=int(getattr(cfg, "csdi_num_steps", 20)),
            schedule=str(getattr(cfg, "csdi_schedule", "quad")),
            timeemb=int(getattr(cfg, "csdi_timeemb", 64)),
            featureemb=int(getattr(cfg, "csdi_featureemb", 16)),
            eval_samples=int(getattr(cfg, "csdi_eval_samples", 1)),
            is_linear=bool(int(getattr(cfg, "csdi_is_linear", 0))),
        )
    raise ValueError(f"Unsupported backbone: {backbone}")


def model_training_loss(
    model: nn.Module,
    x_true: torch.Tensor,
    x_in: torch.Tensor,
    m_in: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    if hasattr(model, "training_loss") and callable(getattr(model, "training_loss")):
        return model.training_loss(x_true, x_in, m_in, loss_mask)  # type: ignore[misc]
    rec = model(x_in, m_in)
    return torch.sum((rec - x_true) ** 2 * loss_mask) / (torch.sum(loss_mask) + 1e-5)


@torch.no_grad()
def model_impute(model: nn.Module, x_in: torch.Tensor, m_in: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "impute_values") and callable(getattr(model, "impute_values")):
        return model.impute_values(x_in, m_in)  # type: ignore[misc]
    return model(x_in, m_in)


def describe_backbone(model: nn.Module) -> Dict[str, Any]:
    name = getattr(model, "backbone_name", model.__class__.__name__)
    return {
        "backbone_name": str(name),
        "class_name": model.__class__.__name__,
        "num_parameters": int(sum(p.numel() for p in model.parameters())),
        "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
    }
