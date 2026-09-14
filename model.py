from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


HIDDEN_DIM = 64
TOTAL_MAMBA_BLOCKS = 3
EARLY_EXIT_BLOCKS = 1
DROPOUT = 0.05
GATE_HIDDEN_DIM = 32

EARLY_LOSS_WEIGHT = 0.30
GATE_LOSS_WEIGHT = 0.35
COMPUTE_PENALTY_WEIGHT = 0.01
BENEFIT_MARGIN = 0.07

ENTROPY_EPS = 1e-8
SCAN_EPS = 1e-4
CONV_KERNEL = 5


class MiniMambaBlock(nn.Module):
    """Compact unidirectional Mamba-style block used in the manuscript."""

    def __init__(
        self,
        hidden_dim: int = HIDDEN_DIM,
        dropout: float = DROPOUT,
        conv_kernel: int = CONV_KERNEL,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim

        self.norm = nn.LayerNorm(hidden_dim)
        self.in_proj = nn.Linear(hidden_dim, hidden_dim * 2)

        self.depthwise_conv = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=conv_kernel,
            padding=conv_kernel // 2,
            groups=hidden_dim,
            bias=True,
        )

        self.dt_proj = nn.Linear(hidden_dim, hidden_dim)
        self.b_proj = nn.Linear(hidden_dim, hidden_dim)
        self.c_proj = nn.Linear(hidden_dim, hidden_dim)

        self.A_log = nn.Parameter(torch.zeros(hidden_dim))
        self.D = nn.Parameter(torch.ones(hidden_dim))

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def selective_scan(self, v: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = v.shape

        delta = F.softplus(self.dt_proj(v)) + SCAN_EPS
        b_t = torch.tanh(self.b_proj(v))
        c_t = torch.tanh(self.c_proj(v))

        a = torch.exp(self.A_log).view(1, hidden_dim)
        d = self.D.view(1, hidden_dim)

        state = torch.zeros(
            batch_size,
            hidden_dim,
            device=v.device,
            dtype=v.dtype,
        )

        outputs = []
        for t in range(seq_len):
            delta_t = delta[:, t, :]
            v_t = v[:, t, :]
            b_cur = b_t[:, t, :]
            c_cur = c_t[:, t, :]

            a_bar = torch.exp(-delta_t * a)
            b_bar = (1.0 - a_bar) * b_cur

            state = a_bar * state + b_bar * v_t
            s_t = c_cur * state + d * v_t
            outputs.append(s_t)

        return torch.stack(outputs, dim=1)

    def forward(self, z_prev: torch.Tensor) -> torch.Tensor:
        residual = z_prev

        z_tilde = self.norm(z_prev)
        ug = self.in_proj(z_tilde)
        u, g = ug.chunk(2, dim=-1)

        v = self.depthwise_conv(u.transpose(1, 2)).transpose(1, 2)
        v = F.silu(v)

        s = self.selective_scan(v)
        y = s * F.silu(g)
        y = self.out_proj(y)
        y = self.dropout(y)

        return residual + y


class BenefitGatedSharedMambaHAR(nn.Module):
    """Benefit-gated shared MiniMamba HAR model.

    Expected input:
        x_normalized: (batch, channels, time)

    Manuscript default:
        H=64, 3 MiniMamba blocks, early exit after block 1,
        depthwise kernel=5, dropout=0.05.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden_dim: int = HIDDEN_DIM,
        total_blocks: int = TOTAL_MAMBA_BLOCKS,
        early_exit_blocks: int = EARLY_EXIT_BLOCKS,
        dropout: float = DROPOUT,
        gate_hidden_dim: int = GATE_HIDDEN_DIM,
    ) -> None:
        super().__init__()

        if not 1 <= early_exit_blocks < total_blocks:
            raise ValueError(
                "early_exit_blocks must satisfy "
                "1 <= early_exit_blocks < total_blocks"
            )
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2.")

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.total_blocks = total_blocks
        self.early_exit_blocks = early_exit_blocks

        self.input_proj = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
        )

        self.blocks = nn.ModuleList(
            [
                MiniMambaBlock(
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    conv_kernel=CONV_KERNEL,
                )
                for _ in range(total_blocks)
            ]
        )

        self.readout_norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        gate_input_dim = hidden_dim + num_classes + 6
        self.benefit_gate = nn.Sequential(
            nn.LayerNorm(gate_input_dim),
            nn.Linear(gate_input_dim, gate_hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden_dim, 1),
        )

    def pool_feature(self, z: torch.Tensor) -> torch.Tensor:
        return self.readout_norm(z).mean(dim=1)

    def classify_from_feature(self, h: torch.Tensor) -> torch.Tensor:
        return self.classifier(h)

    def build_gate_input(
        self,
        x_normalized: torch.Tensor,
        h_early: torch.Tensor,
        early_logits: torch.Tensor,
    ) -> torch.Tensor:
        """g_in = [h_e, o_e, c_e, m_e, u_e, r_t, r_s, r_a].

        All gate inputs are detached from the shared representation graph,
        as stated in the manuscript.
        """
        x_gate = x_normalized.detach()
        h_gate = h_early.detach()
        logits_gate = early_logits.detach()

        prob = F.softmax(logits_gate, dim=1)
        top2 = torch.topk(prob, k=2, dim=1).values

        confidence = top2[:, 0:1]
        margin = top2[:, 0:1] - top2[:, 1:2]

        entropy = -(prob * torch.log(prob + ENTROPY_EPS)).sum(
            dim=1, keepdim=True
        )
        entropy = entropy / math.log(self.num_classes)

        temporal_difference = (
            (x_gate[:, :, 1:] - x_gate[:, :, :-1])
            .pow(2)
            .mean(dim=(1, 2))
            .unsqueeze(1)
        )
        signal_energy = (
            x_gate.pow(2)
            .mean(dim=(1, 2))
            .unsqueeze(1)
        )
        absolute_magnitude = (
            x_gate.abs()
            .mean(dim=(1, 2))
            .unsqueeze(1)
        )

        return torch.cat(
            [
                h_gate,
                logits_gate,
                confidence,
                margin,
                entropy,
                temporal_difference,
                signal_energy,
                absolute_magnitude,
            ],
            dim=1,
        )

    def forward_all(
        self,
        x_normalized: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Training-time paired early/full evaluation."""
        z = self.input_proj(x_normalized).transpose(1, 2)

        h_early = None
        early_logits = None

        for i, block in enumerate(self.blocks):
            z = block(z)

            if i + 1 == self.early_exit_blocks:
                h_early = self.pool_feature(z)
                early_logits = self.classify_from_feature(h_early)

        if h_early is None or early_logits is None:
            raise RuntimeError("Early route was not produced.")

        gate_input = self.build_gate_input(
            x_normalized, h_early, early_logits
        )
        gate_logit = self.benefit_gate(gate_input).squeeze(1)
        benefit_score = torch.sigmoid(gate_logit)

        h_final = self.pool_feature(z)
        final_logits = self.classify_from_feature(h_final)

        return early_logits, final_logits, gate_logit, benefit_score

    def forward(
        self,
        x_normalized: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.forward_all(x_normalized)

    @torch.no_grad()
    def forward_dynamic(
        self,
        x_normalized: torch.Tensor,
        benefit_tau: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Conditional inference using b >= tau for the full route."""
        z = self.input_proj(x_normalized).transpose(1, 2)

        for i in range(self.early_exit_blocks):
            z = self.blocks[i](z)

        h_early = self.pool_feature(z)
        early_logits = self.classify_from_feature(h_early)

        gate_input = self.build_gate_input(
            x_normalized, h_early, early_logits
        )
        gate_logit = self.benefit_gate(gate_input).squeeze(1)
        benefit_score = torch.sigmoid(gate_logit)

        full_mask = benefit_score >= benefit_tau

        dynamic_logits = early_logits.clone()
        route = torch.zeros(
            x_normalized.size(0),
            dtype=torch.long,
            device=x_normalized.device,
        )

        if full_mask.any():
            z_full = z[full_mask]

            for i in range(self.early_exit_blocks, self.total_blocks):
                z_full = self.blocks[i](z_full)

            h_final = self.pool_feature(z_full)
            final_logits = self.classify_from_feature(h_final)

            dynamic_logits[full_mask] = final_logits
            route[full_mask] = 1

        return dynamic_logits, route, benefit_score


@torch.no_grad()
def make_benefit_target(
    early_logits: torch.Tensor,
    final_logits: torch.Tensor,
    y: torch.Tensor,
    benefit_margin: float = BENEFIT_MARGIN,
) -> torch.Tensor:
    """Construct the manuscript benefit target y_gate."""
    early_ce = F.cross_entropy(early_logits, y, reduction="none")
    final_ce = F.cross_entropy(final_logits, y, reduction="none")

    early_pred = early_logits.argmax(dim=1)
    final_pred = final_logits.argmax(dim=1)

    corrected_by_full = (early_pred != y) & (final_pred == y)
    full_harms_early = (early_pred == y) & (final_pred != y)
    meaningful_loss_gain = (early_ce - final_ce) > benefit_margin

    return torch.where(
        corrected_by_full,
        torch.ones_like(early_ce),
        torch.where(
            full_harms_early,
            torch.zeros_like(early_ce),
            meaningful_loss_gain.float(),
        ),
    )


def compute_training_objective(
    early_logits: torch.Tensor,
    final_logits: torch.Tensor,
    gate_logit: torch.Tensor,
    benefit_score: torch.Tensor,
    y: torch.Tensor,
    early_loss_weight: float = EARLY_LOSS_WEIGHT,
    gate_loss_weight: float = GATE_LOSS_WEIGHT,
    compute_penalty_weight: float = COMPUTE_PENALTY_WEIGHT,
    benefit_margin: float = BENEFIT_MARGIN,
) -> Dict[str, torch.Tensor]:
    """L = L_f + lambda_e L_e + lambda_g L_g + lambda_c L_c."""
    final_loss = F.cross_entropy(final_logits, y)
    early_loss = F.cross_entropy(early_logits, y)

    benefit_target = make_benefit_target(
        early_logits,
        final_logits,
        y,
        benefit_margin=benefit_margin,
    )

    gate_loss = F.binary_cross_entropy_with_logits(
        gate_logit,
        benefit_target,
    )
    compute_penalty = benefit_score.mean()

    total_loss = (
        final_loss
        + early_loss_weight * early_loss
        + gate_loss_weight * gate_loss
        + compute_penalty_weight * compute_penalty
    )

    return {
        "loss": total_loss,
        "final_loss": final_loss,
        "early_loss": early_loss,
        "gate_loss": gate_loss,
        "compute_penalty": compute_penalty,
        "benefit_target": benefit_target,
    }


__all__ = [
    "MiniMambaBlock",
    "BenefitGatedSharedMambaHAR",
    "make_benefit_target",
    "compute_training_objective",
]
