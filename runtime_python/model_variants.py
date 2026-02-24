import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from run import ChampionNet, GatedFFN


class ExpandedClassifierHead(nn.Module):
    """
    Larger classifier head that keeps base checkpoint compatibility:
    - retains `weight` and `bias` keys used by the original final linear
    - adds an extra adapter branch (256 -> expansion_dim -> 10)
    - starts near original behavior with alpha initialized to 0
    """

    def __init__(self, in_dim: int = 256, out_dim: int = 10, expansion_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.adapter_up = nn.Linear(in_dim, expansion_dim, bias=False)
        self.adapter_down = nn.Linear(expansion_dim, out_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        nn.init.normal_(self.adapter_up.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.adapter_down.weight)

    def forward(self, x):
        base_logits = F.linear(x, self.weight, self.bias)
        adapter_logits = self.adapter_down(self.dropout(F.silu(self.adapter_up(x))))
        return base_logits + self.alpha * adapter_logits


class ExpandedClassifierHeadXL(nn.Module):
    """
    Wider classifier head for extra capacity and richer routing:
    - keeps base-compatible keys (`weight`, `bias`, `adapter_up/down`, `alpha`)
    - adds a second wider adapter branch + learned token-wise router
    - remains warm-start compatible from base/large checkpoints
    """

    def __init__(
        self,
        in_dim: int = 256,
        out_dim: int = 10,
        expansion_dim: int = 768,
        extra_expansion_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))

        # Reuse large-head key names so xlarge can warm-start from large checkpoints.
        self.adapter_up = nn.Linear(in_dim, expansion_dim, bias=False)
        self.adapter_down = nn.Linear(expansion_dim, out_dim, bias=False)
        self.alpha = nn.Parameter(torch.tensor(0.0))

        self.adapter_up_b = nn.Linear(in_dim, extra_expansion_dim, bias=False)
        self.adapter_down_b = nn.Linear(extra_expansion_dim, out_dim, bias=False)
        self.router = nn.Linear(in_dim, 2, bias=True)
        self.beta = nn.Parameter(torch.tensor(0.0))

        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        nn.init.normal_(self.adapter_up.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.adapter_down.weight)
        nn.init.normal_(self.adapter_up_b.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.adapter_down_b.weight)
        nn.init.zeros_(self.router.weight)
        nn.init.zeros_(self.router.bias)

    def forward(self, x):
        base_logits = F.linear(x, self.weight, self.bias)

        a1 = self.adapter_down(self.dropout(F.silu(self.adapter_up(x))))
        a2 = self.adapter_down_b(self.dropout(F.gelu(self.adapter_up_b(x))))

        route = torch.softmax(self.router(x), dim=-1)
        mix = route[..., :1] * a1 + route[..., 1:2] * a2

        return base_logits + self.alpha * a1 + self.beta * mix


class ExpandedClassifierHeadXXL(nn.Module):
    """
    Maximum-capacity classifier head with tri-branch routing:
    - retains all xlarge-compatible keys for warm-starting
    - adds a third branch and second-stage router for richer fusion
    - remains near-base behavior at init via alpha/beta/gamma initialized to 0
    """

    def __init__(
        self,
        in_dim: int = 256,
        out_dim: int = 10,
        expansion_dim: int = 1024,
        extra_expansion_dim: int = 2048,
        third_expansion_dim: int = 3072,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))

        # Keep xlarge key names to preserve compatibility.
        self.adapter_up = nn.Linear(in_dim, expansion_dim, bias=False)
        self.adapter_down = nn.Linear(expansion_dim, out_dim, bias=False)
        self.alpha = nn.Parameter(torch.tensor(0.0))

        self.adapter_up_b = nn.Linear(in_dim, extra_expansion_dim, bias=False)
        self.adapter_down_b = nn.Linear(extra_expansion_dim, out_dim, bias=False)
        self.router = nn.Linear(in_dim, 2, bias=True)
        self.beta = nn.Parameter(torch.tensor(0.0))

        # New xxl-only branch.
        self.adapter_up_c = nn.Linear(in_dim, third_expansion_dim, bias=False)
        self.adapter_down_c = nn.Linear(third_expansion_dim, out_dim, bias=False)
        self.router2 = nn.Linear(in_dim, 3, bias=True)
        self.gamma = nn.Parameter(torch.tensor(0.0))

        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        nn.init.normal_(self.adapter_up.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.adapter_down.weight)
        nn.init.normal_(self.adapter_up_b.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.adapter_down_b.weight)
        nn.init.zeros_(self.router.weight)
        nn.init.zeros_(self.router.bias)

        nn.init.normal_(self.adapter_up_c.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.adapter_down_c.weight)
        nn.init.zeros_(self.router2.weight)
        nn.init.zeros_(self.router2.bias)

    def forward(self, x):
        base_logits = F.linear(x, self.weight, self.bias)

        a1 = self.adapter_down(self.dropout(F.silu(self.adapter_up(x))))
        a2 = self.adapter_down_b(self.dropout(F.gelu(self.adapter_up_b(x))))
        a3 = self.adapter_down_c(self.dropout(F.mish(self.adapter_up_c(x))))

        route_ab = torch.softmax(self.router(x), dim=-1)
        mix_ab = route_ab[..., :1] * a1 + route_ab[..., 1:2] * a2

        route_all = torch.softmax(self.router2(x), dim=-1)
        mix_all = route_all[..., :1] * a1 + route_all[..., 1:2] * a2 + route_all[..., 2:3] * a3

        return base_logits + self.alpha * a1 + self.beta * mix_ab + self.gamma * mix_all


class ExpandedClassifierHeadXXXL(nn.Module):
    """
    Highest-capacity classifier head with quad-branch routing:
    - preserves xxlarge-compatible keys for warm-starting from xxlarge checkpoints
    - adds a fourth branch plus a third routing stage
    - initializes new scaling params at 0 to remain stable at startup
    """

    def __init__(
        self,
        in_dim: int = 256,
        out_dim: int = 10,
        expansion_dim: int = 1024,
        extra_expansion_dim: int = 2048,
        third_expansion_dim: int = 3072,
        fourth_expansion_dim: int = 4096,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))

        # Keep xxlarge key names to preserve compatibility.
        self.adapter_up = nn.Linear(in_dim, expansion_dim, bias=False)
        self.adapter_down = nn.Linear(expansion_dim, out_dim, bias=False)
        self.alpha = nn.Parameter(torch.tensor(0.0))

        self.adapter_up_b = nn.Linear(in_dim, extra_expansion_dim, bias=False)
        self.adapter_down_b = nn.Linear(extra_expansion_dim, out_dim, bias=False)
        self.router = nn.Linear(in_dim, 2, bias=True)
        self.beta = nn.Parameter(torch.tensor(0.0))

        self.adapter_up_c = nn.Linear(in_dim, third_expansion_dim, bias=False)
        self.adapter_down_c = nn.Linear(third_expansion_dim, out_dim, bias=False)
        self.router2 = nn.Linear(in_dim, 3, bias=True)
        self.gamma = nn.Parameter(torch.tensor(0.0))

        # New xxxlarge-only branch.
        self.adapter_up_d = nn.Linear(in_dim, fourth_expansion_dim, bias=False)
        self.adapter_down_d = nn.Linear(fourth_expansion_dim, out_dim, bias=False)
        self.router3 = nn.Linear(in_dim, 4, bias=True)
        self.delta = nn.Parameter(torch.tensor(0.0))

        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        nn.init.normal_(self.adapter_up.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.adapter_down.weight)
        nn.init.normal_(self.adapter_up_b.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.adapter_down_b.weight)
        nn.init.zeros_(self.router.weight)
        nn.init.zeros_(self.router.bias)

        nn.init.normal_(self.adapter_up_c.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.adapter_down_c.weight)
        nn.init.zeros_(self.router2.weight)
        nn.init.zeros_(self.router2.bias)

        nn.init.normal_(self.adapter_up_d.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.adapter_down_d.weight)
        nn.init.zeros_(self.router3.weight)
        nn.init.zeros_(self.router3.bias)

    def forward(self, x):
        base_logits = F.linear(x, self.weight, self.bias)

        a1 = self.adapter_down(self.dropout(F.silu(self.adapter_up(x))))
        a2 = self.adapter_down_b(self.dropout(F.gelu(self.adapter_up_b(x))))
        a3 = self.adapter_down_c(self.dropout(F.mish(self.adapter_up_c(x))))
        a4 = self.adapter_down_d(self.dropout(F.relu(self.adapter_up_d(x))))

        route_ab = torch.softmax(self.router(x), dim=-1)
        mix_ab = route_ab[..., :1] * a1 + route_ab[..., 1:2] * a2

        route_abc = torch.softmax(self.router2(x), dim=-1)
        mix_abc = route_abc[..., :1] * a1 + route_abc[..., 1:2] * a2 + route_abc[..., 2:3] * a3

        route_abcd = torch.softmax(self.router3(x), dim=-1)
        mix_abcd = (
            route_abcd[..., :1] * a1
            + route_abcd[..., 1:2] * a2
            + route_abcd[..., 2:3] * a3
            + route_abcd[..., 3:4] * a4
        )

        return base_logits + self.alpha * a1 + self.beta * mix_ab + self.gamma * mix_abc + self.delta * mix_abcd


class ExpandedClassifierHeadUltra(nn.Module):
    """
    Maximum-capacity classifier head with five routed branches:
    - keeps xxxlarge-compatible keys for warm-starting
    - adds a fifth branch + fourth routing stage
    - adds domain-expert calibration (science/code/writing/math friendly) for smarter adaptation
    - initializes new scale parameters at 0 for stable adaptation
    """

    def __init__(
        self,
        in_dim: int = 256,
        out_dim: int = 10,
        expansion_dim: int = 1024,
        extra_expansion_dim: int = 2048,
        third_expansion_dim: int = 3072,
        fourth_expansion_dim: int = 4096,
        fifth_expansion_dim: int = 6144,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))

        # Keep xxxlarge keys for compatibility.
        self.adapter_up = nn.Linear(in_dim, expansion_dim, bias=False)
        self.adapter_down = nn.Linear(expansion_dim, out_dim, bias=False)
        self.alpha = nn.Parameter(torch.tensor(0.0))

        self.adapter_up_b = nn.Linear(in_dim, extra_expansion_dim, bias=False)
        self.adapter_down_b = nn.Linear(extra_expansion_dim, out_dim, bias=False)
        self.router = nn.Linear(in_dim, 2, bias=True)
        self.beta = nn.Parameter(torch.tensor(0.0))

        self.adapter_up_c = nn.Linear(in_dim, third_expansion_dim, bias=False)
        self.adapter_down_c = nn.Linear(third_expansion_dim, out_dim, bias=False)
        self.router2 = nn.Linear(in_dim, 3, bias=True)
        self.gamma = nn.Parameter(torch.tensor(0.0))

        self.adapter_up_d = nn.Linear(in_dim, fourth_expansion_dim, bias=False)
        self.adapter_down_d = nn.Linear(fourth_expansion_dim, out_dim, bias=False)
        self.router3 = nn.Linear(in_dim, 4, bias=True)
        self.delta = nn.Parameter(torch.tensor(0.0))

        # New ultralarge-only branch.
        self.adapter_up_e = nn.Linear(in_dim, fifth_expansion_dim, bias=False)
        self.adapter_down_e = nn.Linear(fifth_expansion_dim, out_dim, bias=False)
        self.router4 = nn.Linear(in_dim, 5, bias=True)
        self.epsilon = nn.Parameter(torch.tensor(0.0))

        # Architecture upgrade: low-cost domain-expert correction + bounded calibration.
        self.pre_norm = nn.LayerNorm(in_dim)
        self.domain_router = nn.Linear(in_dim, 4, bias=True)
        self.domain_experts = nn.Linear(in_dim, out_dim * 4, bias=False)
        self.calib_gate = nn.Linear(in_dim, out_dim, bias=True)
        self.zeta = nn.Parameter(torch.tensor(0.0))
        self.theta = nn.Parameter(torch.tensor(0.0))

        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        nn.init.normal_(self.adapter_up.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.adapter_down.weight)
        nn.init.normal_(self.adapter_up_b.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.adapter_down_b.weight)
        nn.init.zeros_(self.router.weight)
        nn.init.zeros_(self.router.bias)

        nn.init.normal_(self.adapter_up_c.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.adapter_down_c.weight)
        nn.init.zeros_(self.router2.weight)
        nn.init.zeros_(self.router2.bias)

        nn.init.normal_(self.adapter_up_d.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.adapter_down_d.weight)
        nn.init.zeros_(self.router3.weight)
        nn.init.zeros_(self.router3.bias)

        nn.init.normal_(self.adapter_up_e.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.adapter_down_e.weight)
        nn.init.zeros_(self.router4.weight)
        nn.init.zeros_(self.router4.bias)

        nn.init.ones_(self.pre_norm.weight)
        nn.init.zeros_(self.pre_norm.bias)
        nn.init.zeros_(self.domain_router.weight)
        nn.init.zeros_(self.domain_router.bias)
        nn.init.normal_(self.domain_experts.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.calib_gate.weight)
        nn.init.zeros_(self.calib_gate.bias)

    def forward(self, x):
        base_logits = F.linear(x, self.weight, self.bias)

        a1 = self.adapter_down(self.dropout(F.silu(self.adapter_up(x))))
        a2 = self.adapter_down_b(self.dropout(F.gelu(self.adapter_up_b(x))))
        a3 = self.adapter_down_c(self.dropout(F.mish(self.adapter_up_c(x))))
        a4 = self.adapter_down_d(self.dropout(F.relu(self.adapter_up_d(x))))
        a5 = self.adapter_down_e(self.dropout(F.selu(self.adapter_up_e(x))))

        route_ab = torch.softmax(self.router(x), dim=-1)
        mix_ab = route_ab[..., :1] * a1 + route_ab[..., 1:2] * a2

        route_abc = torch.softmax(self.router2(x), dim=-1)
        mix_abc = route_abc[..., :1] * a1 + route_abc[..., 1:2] * a2 + route_abc[..., 2:3] * a3

        route_abcd = torch.softmax(self.router3(x), dim=-1)
        mix_abcd = (
            route_abcd[..., :1] * a1
            + route_abcd[..., 1:2] * a2
            + route_abcd[..., 2:3] * a3
            + route_abcd[..., 3:4] * a4
        )

        route_abcde = torch.softmax(self.router4(x), dim=-1)
        mix_abcde = (
            route_abcde[..., :1] * a1
            + route_abcde[..., 1:2] * a2
            + route_abcde[..., 2:3] * a3
            + route_abcde[..., 3:4] * a4
            + route_abcde[..., 4:5] * a5
        )

        # Domain-expert calibration branch improves robustness across diverse domains.
        h = self.pre_norm(x)
        dom_logits = self.domain_experts(self.dropout(F.silu(h)))
        dom_logits = dom_logits.view(*dom_logits.shape[:-1], 4, base_logits.shape[-1])
        dom_w = torch.softmax(self.domain_router(h), dim=-1).unsqueeze(-1)
        dom_mix = (dom_w * dom_logits).sum(dim=-2)
        calib = torch.tanh(self.calib_gate(h))

        return (
            base_logits
            + self.alpha * a1
            + self.beta * mix_ab
            + self.gamma * mix_abc
            + self.delta * mix_abcd
            + self.epsilon * mix_abcde
            + self.zeta * dom_mix
            + self.theta * calib
        )


class CrossAttentionFusion(nn.Module):
    """
    Lightweight cross-attention module that lets adapter branches attend to each other.
    Input: (B, T, N_branches, out_dim) tensor of stacked branch outputs.
    Output: (B, T, out_dim) fused representation.
    """
    def __init__(self, out_dim: int = 10, n_branches: int = 6, n_heads: int = 2):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = max(1, out_dim // n_heads)
        inner = self.n_heads * self.head_dim
        self.q_proj = nn.Linear(out_dim, inner, bias=False)
        self.k_proj = nn.Linear(out_dim, inner, bias=False)
        self.v_proj = nn.Linear(out_dim, inner, bias=False)
        self.out_proj = nn.Linear(inner, out_dim, bias=False)
        self.scale = self.head_dim ** -0.5
        self.fusion_weight = nn.Linear(out_dim, 1, bias=True)

    def forward(self, branch_stack):
        # branch_stack: (B, T, N, D)
        B, T, N, D = branch_stack.shape
        flat = branch_stack.view(B * T, N, D)
        q = self.q_proj(flat).view(B * T, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(flat).view(B * T, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(flat).view(B * T, N, self.n_heads, self.head_dim).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # (B*T, heads, N, head_dim)
        out = out.transpose(1, 2).contiguous().view(B * T, N, self.n_heads * self.head_dim)
        out = self.out_proj(out)  # (B*T, N, D)
        # Weighted sum over branches
        w = torch.softmax(self.fusion_weight(out), dim=1)  # (B*T, N, 1)
        fused = (w * out).sum(dim=1)  # (B*T, D)
        return fused.view(B, T, D)


class ExpandedClassifierHeadMega(nn.Module):
    """
    Maximum-capacity classifier head with six routed branches + cross-attention fusion:
    - keeps ultralarge-compatible keys for warm-starting
    - adds a sixth branch (Mish activation) + fifth routing stage
    - adds CrossAttentionFusion for inter-branch communication
    - adds a reasoning_gate that amplifies logit differences for sharper routing
    - initializes new scale parameters at 0 for stable adaptation
    """

    def __init__(
        self,
        in_dim: int = 256,
        out_dim: int = 10,
        expansion_dim: int = 1024,
        extra_expansion_dim: int = 2048,
        third_expansion_dim: int = 3072,
        fourth_expansion_dim: int = 4096,
        fifth_expansion_dim: int = 6144,
        sixth_expansion_dim: int = 8192,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))

        # Keep ultralarge keys for compatibility.
        self.adapter_up = nn.Linear(in_dim, expansion_dim, bias=False)
        self.adapter_down = nn.Linear(expansion_dim, out_dim, bias=False)
        self.alpha = nn.Parameter(torch.tensor(0.0))

        self.adapter_up_b = nn.Linear(in_dim, extra_expansion_dim, bias=False)
        self.adapter_down_b = nn.Linear(extra_expansion_dim, out_dim, bias=False)
        self.router = nn.Linear(in_dim, 2, bias=True)
        self.beta = nn.Parameter(torch.tensor(0.0))

        self.adapter_up_c = nn.Linear(in_dim, third_expansion_dim, bias=False)
        self.adapter_down_c = nn.Linear(third_expansion_dim, out_dim, bias=False)
        self.router2 = nn.Linear(in_dim, 3, bias=True)
        self.gamma = nn.Parameter(torch.tensor(0.0))

        self.adapter_up_d = nn.Linear(in_dim, fourth_expansion_dim, bias=False)
        self.adapter_down_d = nn.Linear(fourth_expansion_dim, out_dim, bias=False)
        self.router3 = nn.Linear(in_dim, 4, bias=True)
        self.delta = nn.Parameter(torch.tensor(0.0))

        self.adapter_up_e = nn.Linear(in_dim, fifth_expansion_dim, bias=False)
        self.adapter_down_e = nn.Linear(fifth_expansion_dim, out_dim, bias=False)
        self.router4 = nn.Linear(in_dim, 5, bias=True)
        self.epsilon = nn.Parameter(torch.tensor(0.0))

        # Domain-expert calibration (from ultralarge).
        self.pre_norm = nn.LayerNorm(in_dim)
        self.domain_router = nn.Linear(in_dim, 4, bias=True)
        self.domain_experts = nn.Linear(in_dim, out_dim * 4, bias=False)
        self.calib_gate = nn.Linear(in_dim, out_dim, bias=True)
        self.zeta = nn.Parameter(torch.tensor(0.0))
        self.theta = nn.Parameter(torch.tensor(0.0))

        # --- Megalarge-only additions ---
        # Sixth branch with Mish activation.
        self.adapter_up_f = nn.Linear(in_dim, sixth_expansion_dim, bias=False)
        self.adapter_down_f = nn.Linear(sixth_expansion_dim, out_dim, bias=False)
        self.router5 = nn.Linear(in_dim, 6, bias=True)
        self.iota = nn.Parameter(torch.tensor(0.0))

        # Cross-attention fusion lets branches attend to each other.
        self.cross_attn_fusion = CrossAttentionFusion(out_dim=out_dim, n_branches=6, n_heads=2)
        self.kappa = nn.Parameter(torch.tensor(0.0))

        # Reasoning gate amplifies logit differences for sharper routing.
        self.reasoning_gate = nn.Sequential(
            nn.Linear(in_dim, in_dim // 4, bias=False),
            nn.SiLU(),
            nn.Linear(in_dim // 4, out_dim, bias=True),
        )
        self.lambda_ = nn.Parameter(torch.tensor(0.0))

        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        for name in ['adapter_up', 'adapter_up_b', 'adapter_up_c', 'adapter_up_d', 'adapter_up_e', 'adapter_up_f']:
            nn.init.normal_(getattr(self, name).weight, mean=0.0, std=0.02)
        for name in ['adapter_down', 'adapter_down_b', 'adapter_down_c', 'adapter_down_d', 'adapter_down_e', 'adapter_down_f']:
            nn.init.zeros_(getattr(self, name).weight)
        for name in ['router', 'router2', 'router3', 'router4', 'router5', 'domain_router']:
            nn.init.zeros_(getattr(self, name).weight)
            nn.init.zeros_(getattr(self, name).bias)
        nn.init.ones_(self.pre_norm.weight)
        nn.init.zeros_(self.pre_norm.bias)
        nn.init.normal_(self.domain_experts.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.calib_gate.weight)
        nn.init.zeros_(self.calib_gate.bias)
        # Reasoning gate init
        for m in self.reasoning_gate:
            if hasattr(m, 'weight'):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        base_logits = F.linear(x, self.weight, self.bias)

        a1 = self.adapter_down(self.dropout(F.silu(self.adapter_up(x))))
        a2 = self.adapter_down_b(self.dropout(F.gelu(self.adapter_up_b(x))))
        a3 = self.adapter_down_c(self.dropout(F.mish(self.adapter_up_c(x))))
        a4 = self.adapter_down_d(self.dropout(F.relu(self.adapter_up_d(x))))
        a5 = self.adapter_down_e(self.dropout(F.selu(self.adapter_up_e(x))))
        a6 = self.adapter_down_f(self.dropout(F.mish(self.adapter_up_f(x))))

        route_ab = torch.softmax(self.router(x), dim=-1)
        mix_ab = route_ab[..., :1] * a1 + route_ab[..., 1:2] * a2

        route_abc = torch.softmax(self.router2(x), dim=-1)
        mix_abc = route_abc[..., :1] * a1 + route_abc[..., 1:2] * a2 + route_abc[..., 2:3] * a3

        route_abcd = torch.softmax(self.router3(x), dim=-1)
        mix_abcd = (
            route_abcd[..., :1] * a1
            + route_abcd[..., 1:2] * a2
            + route_abcd[..., 2:3] * a3
            + route_abcd[..., 3:4] * a4
        )

        route_abcde = torch.softmax(self.router4(x), dim=-1)
        mix_abcde = (
            route_abcde[..., :1] * a1
            + route_abcde[..., 1:2] * a2
            + route_abcde[..., 2:3] * a3
            + route_abcde[..., 3:4] * a4
            + route_abcde[..., 4:5] * a5
        )

        route_all = torch.softmax(self.router5(x), dim=-1)
        mix_all = (
            route_all[..., :1] * a1
            + route_all[..., 1:2] * a2
            + route_all[..., 2:3] * a3
            + route_all[..., 3:4] * a4
            + route_all[..., 4:5] * a5
            + route_all[..., 5:6] * a6
        )

        # Domain-expert calibration branch.
        h = self.pre_norm(x)
        dom_logits = self.domain_experts(self.dropout(F.silu(h)))
        dom_logits = dom_logits.view(*dom_logits.shape[:-1], 4, base_logits.shape[-1])
        dom_w = torch.softmax(self.domain_router(h), dim=-1).unsqueeze(-1)
        dom_mix = (dom_w * dom_logits).sum(dim=-2)
        calib = torch.tanh(self.calib_gate(h))

        # Cross-attention fusion: stack all 6 branches and let them attend.
        branch_stack = torch.stack([a1, a2, a3, a4, a5, a6], dim=-2)  # (B,T,6,out_dim)
        cross_fused = self.cross_attn_fusion(branch_stack)

        # Reasoning gate: amplify logit differences for sharper decisions.
        reason = torch.tanh(self.reasoning_gate(x))

        return (
            base_logits
            + self.alpha * a1
            + self.beta * mix_ab
            + self.gamma * mix_abc
            + self.delta * mix_abcd
            + self.epsilon * mix_abcde
            + self.iota * mix_all
            + self.zeta * dom_mix
            + self.theta * calib
            + self.kappa * cross_fused
            + self.lambda_ * reason
        )


class ChampionNetLarge(nn.Module):
    """
    Backbone-compatible larger model:
    - keeps layers 0..9 and layer 11 from ChampionNet
    - replaces layer 10 with ExpandedClassifierHead
    """

    def __init__(self, expansion_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        base = ChampionNet()
        layers = [base.layers[i] for i in range(10)]
        layers.append(ExpandedClassifierHead(256, 10, expansion_dim=expansion_dim, dropout=dropout))
        layers.append(base.layers[11])
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ChampionNetXL(nn.Module):
    """
    Backbone-compatible xlarge model:
    - keeps layers 0..9 and layer 11 from ChampionNet
    - replaces layer 10 with ExpandedClassifierHeadXL
    """

    def __init__(self, expansion_dim: int = 768, extra_expansion_dim: int = 1024, dropout: float = 0.1):
        super().__init__()
        base = ChampionNet()
        layers = [base.layers[i] for i in range(10)]
        layers.append(
            ExpandedClassifierHeadXL(
                256,
                10,
                expansion_dim=expansion_dim,
                extra_expansion_dim=extra_expansion_dim,
                dropout=dropout,
            )
        )
        layers.append(base.layers[11])
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ChampionNetXXL(nn.Module):
    """
    Backbone-compatible xxlarge model:
    - keeps layers 0..9 and layer 11 from ChampionNet
    - replaces layer 10 with ExpandedClassifierHeadXXL
    """

    def __init__(
        self,
        expansion_dim: int = 1024,
        extra_expansion_dim: int = 2048,
        third_expansion_dim: int = 3072,
        dropout: float = 0.1,
    ):
        super().__init__()
        base = ChampionNet()
        layers = [base.layers[i] for i in range(10)]
        layers.append(
            ExpandedClassifierHeadXXL(
                256,
                10,
                expansion_dim=expansion_dim,
                extra_expansion_dim=extra_expansion_dim,
                third_expansion_dim=third_expansion_dim,
                dropout=dropout,
            )
        )
        layers.append(base.layers[11])
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ChampionNetXXXL(nn.Module):
    """
    Backbone-compatible xxxlarge model:
    - keeps layers 0..9 and layer 11 from ChampionNet
    - replaces layer 10 with ExpandedClassifierHeadXXXL
    """

    def __init__(
        self,
        expansion_dim: int = 1024,
        extra_expansion_dim: int = 2048,
        third_expansion_dim: int = 3072,
        fourth_expansion_dim: int = 4096,
        dropout: float = 0.1,
    ):
        super().__init__()
        base = ChampionNet()
        layers = [base.layers[i] for i in range(10)]
        layers.append(
            ExpandedClassifierHeadXXXL(
                256,
                10,
                expansion_dim=expansion_dim,
                extra_expansion_dim=extra_expansion_dim,
                third_expansion_dim=third_expansion_dim,
                fourth_expansion_dim=fourth_expansion_dim,
                dropout=dropout,
            )
        )
        layers.append(base.layers[11])
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ChampionNetUltra(nn.Module):
    """
    Backbone-compatible ultralarge model:
    - keeps layers 0..9 and layer 11 from ChampionNet
    - replaces layer 10 with ExpandedClassifierHeadUltra
    """

    def __init__(
        self,
        expansion_dim: int = 1024,
        extra_expansion_dim: int = 2048,
        third_expansion_dim: int = 3072,
        fourth_expansion_dim: int = 4096,
        fifth_expansion_dim: int = 6144,
        dropout: float = 0.1,
    ):
        super().__init__()
        base = ChampionNet()
        layers = [base.layers[i] for i in range(10)]
        layers.append(
            ExpandedClassifierHeadUltra(
                256,
                10,
                expansion_dim=expansion_dim,
                extra_expansion_dim=extra_expansion_dim,
                third_expansion_dim=third_expansion_dim,
                fourth_expansion_dim=fourth_expansion_dim,
                fifth_expansion_dim=fifth_expansion_dim,
                dropout=dropout,
            )
        )
        layers.append(base.layers[11])
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ChampionNetMega(nn.Module):
    """
    Backbone-compatible megalarge model:
    - keeps layers 0..9 and layer 11 from ChampionNet
    - replaces layer 10 with ExpandedClassifierHeadMega
    - adds GatedFFN after BitNetLinear for extra backbone capacity
    """

    def __init__(
        self,
        expansion_dim: int = 1024,
        extra_expansion_dim: int = 2048,
        third_expansion_dim: int = 3072,
        fourth_expansion_dim: int = 4096,
        fifth_expansion_dim: int = 6144,
        sixth_expansion_dim: int = 8192,
        dropout: float = 0.1,
    ):
        super().__init__()
        base = ChampionNet()
        layers = [base.layers[i] for i in range(10)]
        # Insert GatedFFN for extra backbone capacity (between BitNet and classifier).
        layers.append(GatedFFN(d_model=256, d_inner=512))
        layers.append(
            ExpandedClassifierHeadMega(
                256,
                10,
                expansion_dim=expansion_dim,
                extra_expansion_dim=extra_expansion_dim,
                third_expansion_dim=third_expansion_dim,
                fourth_expansion_dim=fourth_expansion_dim,
                fifth_expansion_dim=fifth_expansion_dim,
                sixth_expansion_dim=sixth_expansion_dim,
                dropout=dropout,
            )
        )
        layers.append(base.layers[11])
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def build_model(
    model_size: str = "base",
    expansion_dim: int = 512,
    dropout: float = 0.1,
    extra_expansion_dim: Optional[int] = None,
    third_expansion_dim: Optional[int] = None,
    fourth_expansion_dim: Optional[int] = None,
    fifth_expansion_dim: Optional[int] = None,
    sixth_expansion_dim: Optional[int] = None,
) -> nn.Module:
    if model_size == "base":
        return ChampionNet()
    if model_size == "large":
        return ChampionNetLarge(expansion_dim=expansion_dim, dropout=dropout)
    if model_size == "xlarge":
        extra_dim = int(extra_expansion_dim) if extra_expansion_dim is not None else int(max(1024, expansion_dim * 2))
        return ChampionNetXL(expansion_dim=expansion_dim, extra_expansion_dim=extra_dim, dropout=dropout)
    if model_size == "xxlarge":
        extra_dim = int(extra_expansion_dim) if extra_expansion_dim is not None else int(max(2048, expansion_dim * 2))
        third_dim = int(third_expansion_dim) if third_expansion_dim is not None else int(max(3072, extra_dim + expansion_dim))
        return ChampionNetXXL(
            expansion_dim=expansion_dim,
            extra_expansion_dim=extra_dim,
            third_expansion_dim=third_dim,
            dropout=dropout,
        )
    if model_size == "xxxlarge":
        extra_dim = int(extra_expansion_dim) if extra_expansion_dim is not None else int(max(2048, expansion_dim * 2))
        third_dim = int(third_expansion_dim) if third_expansion_dim is not None else int(max(3072, extra_dim + expansion_dim))
        fourth_dim = (
            int(fourth_expansion_dim)
            if fourth_expansion_dim is not None
            else int(max(4096, third_dim + expansion_dim))
        )
        return ChampionNetXXXL(
            expansion_dim=expansion_dim,
            extra_expansion_dim=extra_dim,
            third_expansion_dim=third_dim,
            fourth_expansion_dim=fourth_dim,
            dropout=dropout,
        )
    if model_size == "ultralarge":
        extra_dim = int(extra_expansion_dim) if extra_expansion_dim is not None else int(max(2048, expansion_dim * 2))
        third_dim = int(third_expansion_dim) if third_expansion_dim is not None else int(max(3072, extra_dim + expansion_dim))
        fourth_dim = (
            int(fourth_expansion_dim)
            if fourth_expansion_dim is not None
            else int(max(4096, third_dim + expansion_dim))
        )
        fifth_dim = (
            int(fifth_expansion_dim)
            if fifth_expansion_dim is not None
            else int(max(6144, fourth_dim + expansion_dim))
        )
        return ChampionNetUltra(
            expansion_dim=expansion_dim,
            extra_expansion_dim=extra_dim,
            third_expansion_dim=third_dim,
            fourth_expansion_dim=fourth_dim,
            fifth_expansion_dim=fifth_dim,
            dropout=dropout,
        )
    if model_size == "megalarge":
        extra_dim = int(extra_expansion_dim) if extra_expansion_dim is not None else int(max(2048, expansion_dim * 2))
        third_dim = int(third_expansion_dim) if third_expansion_dim is not None else int(max(3072, extra_dim + expansion_dim))
        fourth_dim = (
            int(fourth_expansion_dim)
            if fourth_expansion_dim is not None
            else int(max(4096, third_dim + expansion_dim))
        )
        fifth_dim = (
            int(fifth_expansion_dim)
            if fifth_expansion_dim is not None
            else int(max(6144, fourth_dim + expansion_dim))
        )
        sixth_dim = (
            int(sixth_expansion_dim)
            if sixth_expansion_dim is not None
            else int(max(8192, fifth_dim + expansion_dim))
        )
        return ChampionNetMega(
            expansion_dim=expansion_dim,
            extra_expansion_dim=extra_dim,
            third_expansion_dim=third_dim,
            fourth_expansion_dim=fourth_dim,
            fifth_expansion_dim=fifth_dim,
            sixth_expansion_dim=sixth_dim,
            dropout=dropout,
        )
    raise ValueError(
        f"Unknown model_size={model_size!r}. Use 'base', 'large', 'xlarge', 'xxlarge', 'xxxlarge', 'ultralarge', or 'megalarge'."
    )


def load_weights_for_model(model: nn.Module, state_dict: dict, model_size: str) -> Tuple[List[str], List[str]]:
    if model_size == "base":
        incompatible = model.load_state_dict(state_dict, strict=False)
        return list(incompatible.missing_keys), list(incompatible.unexpected_keys)

    if model_size == "large":
        try:
            incompatible = model.load_state_dict(state_dict, strict=False)
        except RuntimeError as exc:
            raise RuntimeError(
                "Failed to load large checkpoint: likely head-dimension mismatch. "
                "Use matching --expansion_dim or auto-detect from checkpoint."
            ) from exc
        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys)

        # Expected when loading base checkpoint into large model.
        allowed_missing = {
            "layers.10.adapter_up.weight",
            "layers.10.adapter_down.weight",
            "layers.10.alpha",
        }
        missing_filtered = [k for k in missing if k and k not in allowed_missing]
        unexpected_filtered = [k for k in unexpected if k]
        return missing_filtered, unexpected_filtered

    if model_size == "xlarge":
        try:
            incompatible = model.load_state_dict(state_dict, strict=False)
        except RuntimeError as exc:
            raise RuntimeError(
                "Failed to load xlarge checkpoint: likely expansion/aux dimension mismatch. "
                "Use matching --expansion_dim/--extra_expansion_dim or auto-detect from checkpoint."
            ) from exc
        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys)

        # Expected when warm-starting xlarge from base/large checkpoints.
        allowed_missing = {
            "layers.10.adapter_up.weight",
            "layers.10.adapter_down.weight",
            "layers.10.alpha",
            "layers.10.adapter_up_b.weight",
            "layers.10.adapter_down_b.weight",
            "layers.10.router.weight",
            "layers.10.router.bias",
            "layers.10.beta",
        }
        missing_filtered = [k for k in missing if k and k not in allowed_missing]
        unexpected_filtered = [k for k in unexpected if k]
        return missing_filtered, unexpected_filtered

    if model_size == "xxlarge":
        try:
            incompatible = model.load_state_dict(state_dict, strict=False)
        except RuntimeError as exc:
            raise RuntimeError(
                "Failed to load xxlarge checkpoint: likely expansion/aux/third dimension mismatch. "
                "Use matching --expansion_dim/--extra_expansion_dim/--third_expansion_dim or auto-detect from checkpoint."
            ) from exc
        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys)

        # Expected when warm-starting xxlarge from base/large/xlarge checkpoints.
        allowed_missing = {
            "layers.10.adapter_up.weight",
            "layers.10.adapter_down.weight",
            "layers.10.alpha",
            "layers.10.adapter_up_b.weight",
            "layers.10.adapter_down_b.weight",
            "layers.10.router.weight",
            "layers.10.router.bias",
            "layers.10.beta",
            "layers.10.adapter_up_c.weight",
            "layers.10.adapter_down_c.weight",
            "layers.10.router2.weight",
            "layers.10.router2.bias",
            "layers.10.gamma",
        }
        missing_filtered = [k for k in missing if k and k not in allowed_missing]
        unexpected_filtered = [k for k in unexpected if k]
        return missing_filtered, unexpected_filtered

    if model_size == "xxxlarge":
        try:
            incompatible = model.load_state_dict(state_dict, strict=False)
        except RuntimeError as exc:
            raise RuntimeError(
                "Failed to load xxxlarge checkpoint: likely expansion/aux/third/fourth dimension mismatch. "
                "Use matching dims or auto-detect from checkpoint."
            ) from exc
        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys)

        # Expected when warm-starting xxxlarge from base/large/xlarge/xxlarge checkpoints.
        allowed_missing = {
            "layers.10.adapter_up.weight",
            "layers.10.adapter_down.weight",
            "layers.10.alpha",
            "layers.10.adapter_up_b.weight",
            "layers.10.adapter_down_b.weight",
            "layers.10.router.weight",
            "layers.10.router.bias",
            "layers.10.beta",
            "layers.10.adapter_up_c.weight",
            "layers.10.adapter_down_c.weight",
            "layers.10.router2.weight",
            "layers.10.router2.bias",
            "layers.10.gamma",
            "layers.10.adapter_up_d.weight",
            "layers.10.adapter_down_d.weight",
            "layers.10.router3.weight",
            "layers.10.router3.bias",
            "layers.10.delta",
        }
        missing_filtered = [k for k in missing if k and k not in allowed_missing]
        unexpected_filtered = [k for k in unexpected if k]
        return missing_filtered, unexpected_filtered

    if model_size == "ultralarge":
        try:
            incompatible = model.load_state_dict(state_dict, strict=False)
        except RuntimeError as exc:
            raise RuntimeError(
                "Failed to load ultralarge checkpoint: likely expansion/aux/third/fourth/fifth dimension mismatch. "
                "Use matching dims or auto-detect from checkpoint."
            ) from exc
        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys)

        # Expected when warm-starting ultralarge from smaller checkpoints.
        allowed_missing = {
            "layers.10.adapter_up.weight",
            "layers.10.adapter_down.weight",
            "layers.10.alpha",
            "layers.10.adapter_up_b.weight",
            "layers.10.adapter_down_b.weight",
            "layers.10.router.weight",
            "layers.10.router.bias",
            "layers.10.beta",
            "layers.10.adapter_up_c.weight",
            "layers.10.adapter_down_c.weight",
            "layers.10.router2.weight",
            "layers.10.router2.bias",
            "layers.10.gamma",
            "layers.10.adapter_up_d.weight",
            "layers.10.adapter_down_d.weight",
            "layers.10.router3.weight",
            "layers.10.router3.bias",
            "layers.10.delta",
            "layers.10.adapter_up_e.weight",
            "layers.10.adapter_down_e.weight",
            "layers.10.router4.weight",
            "layers.10.router4.bias",
            "layers.10.epsilon",
            "layers.10.pre_norm.weight",
            "layers.10.pre_norm.bias",
            "layers.10.domain_router.weight",
            "layers.10.domain_router.bias",
            "layers.10.domain_experts.weight",
            "layers.10.calib_gate.weight",
            "layers.10.calib_gate.bias",
            "layers.10.zeta",
            "layers.10.theta",
        }
        missing_filtered = [k for k in missing if k and k not in allowed_missing]
        unexpected_filtered = [k for k in unexpected if k]
        return missing_filtered, unexpected_filtered

    if model_size == "megalarge":
        try:
            incompatible = model.load_state_dict(state_dict, strict=False)
        except RuntimeError as exc:
            raise RuntimeError(
                "Failed to load megalarge checkpoint: likely expansion dimension mismatch. "
                "Use matching dims or auto-detect from checkpoint."
            ) from exc
        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys)

        # Expected when warm-starting megalarge from smaller checkpoints.
        allowed_missing = {
            "layers.10.adapter_up.weight",
            "layers.10.adapter_down.weight",
            "layers.10.alpha",
            "layers.10.adapter_up_b.weight",
            "layers.10.adapter_down_b.weight",
            "layers.10.router.weight",
            "layers.10.router.bias",
            "layers.10.beta",
            "layers.10.adapter_up_c.weight",
            "layers.10.adapter_down_c.weight",
            "layers.10.router2.weight",
            "layers.10.router2.bias",
            "layers.10.gamma",
            "layers.10.adapter_up_d.weight",
            "layers.10.adapter_down_d.weight",
            "layers.10.router3.weight",
            "layers.10.router3.bias",
            "layers.10.delta",
            "layers.10.adapter_up_e.weight",
            "layers.10.adapter_down_e.weight",
            "layers.10.router4.weight",
            "layers.10.router4.bias",
            "layers.10.epsilon",
            "layers.10.pre_norm.weight",
            "layers.10.pre_norm.bias",
            "layers.10.domain_router.weight",
            "layers.10.domain_router.bias",
            "layers.10.domain_experts.weight",
            "layers.10.calib_gate.weight",
            "layers.10.calib_gate.bias",
            "layers.10.zeta",
            "layers.10.theta",
            # Megalarge-only keys
            "layers.10.adapter_up_f.weight",
            "layers.10.adapter_down_f.weight",
            "layers.10.router5.weight",
            "layers.10.router5.bias",
            "layers.10.iota",
            "layers.10.cross_attn_fusion.q_proj.weight",
            "layers.10.cross_attn_fusion.k_proj.weight",
            "layers.10.cross_attn_fusion.v_proj.weight",
            "layers.10.cross_attn_fusion.out_proj.weight",
            "layers.10.cross_attn_fusion.fusion_weight.weight",
            "layers.10.cross_attn_fusion.fusion_weight.bias",
            "layers.10.kappa",
            "layers.10.reasoning_gate.0.weight",
            "layers.10.reasoning_gate.2.weight",
            "layers.10.reasoning_gate.2.bias",
            "layers.10.lambda_",
            # GatedFFN backbone layer (layers.10 in mega = GatedFFN)
            "layers.10.norm.weight",
            "layers.10.up_proj.weight",
            "layers.10.down_proj.weight",
            "layers.10.layer_scale.gamma",
        }
        # Megalarge uses layers.10=GatedFFN, layers.11=Head, layers.12=norm
        # So head keys are at layers.11.* not layers.10.*
        mega_allowed = set()
        for k in allowed_missing:
            if k.startswith("layers.10."):
                mega_key = "layers.11." + k[len("layers.10."):]
                mega_allowed.add(mega_key)
        mega_allowed.update(allowed_missing)
        # Also allow GatedFFN keys at layers.10
        mega_allowed.update({
            "layers.10.norm.weight",
            "layers.10.up_proj.weight",
            "layers.10.down_proj.weight",
            "layers.10.layer_scale.gamma",
        })
        missing_filtered = [k for k in missing if k and k not in mega_allowed]
        unexpected_filtered = [k for k in unexpected if k]
        return missing_filtered, unexpected_filtered

    raise ValueError(f"Unsupported model_size={model_size!r}")


def detect_model_size_from_state_dict(state_dict: dict) -> str:
    # Megalarge: has GatedFFN at layers.10 + head at layers.11
    if "layers.11.adapter_up_f.weight" in state_dict or "layers.11.router5.weight" in state_dict:
        return "megalarge"
    if "layers.10.adapter_up_e.weight" in state_dict or "layers.10.router4.weight" in state_dict:
        return "ultralarge"
    if "layers.10.adapter_up_d.weight" in state_dict or "layers.10.router3.weight" in state_dict:
        return "xxxlarge"
    if "layers.10.adapter_up_c.weight" in state_dict or "layers.10.router2.weight" in state_dict:
        return "xxlarge"
    if "layers.10.adapter_up_b.weight" in state_dict or "layers.10.router.weight" in state_dict:
        return "xlarge"
    if "layers.10.adapter_up.weight" in state_dict or "layers.10.adapter_down.weight" in state_dict:
        return "large"
    return "base"


def detect_large_head_expansion_dim(state_dict: dict, default: int = 512) -> int:
    key = "layers.10.adapter_up.weight"
    weight = state_dict.get(key)
    if isinstance(weight, torch.Tensor) and weight.ndim == 2:
        return int(weight.shape[0])
    return int(default)


def detect_xlarge_aux_expansion_dim(state_dict: dict, default: int = 1024) -> int:
    key = "layers.10.adapter_up_b.weight"
    weight = state_dict.get(key)
    if isinstance(weight, torch.Tensor) and weight.ndim == 2:
        return int(weight.shape[0])
    return int(default)


def detect_xxlarge_third_expansion_dim(state_dict: dict, default: int = 3072) -> int:
    key = "layers.10.adapter_up_c.weight"
    weight = state_dict.get(key)
    if isinstance(weight, torch.Tensor) and weight.ndim == 2:
        return int(weight.shape[0])
    return int(default)


def detect_xxxlarge_fourth_expansion_dim(state_dict: dict, default: int = 4096) -> int:
    key = "layers.10.adapter_up_d.weight"
    weight = state_dict.get(key)
    if isinstance(weight, torch.Tensor) and weight.ndim == 2:
        return int(weight.shape[0])
    return int(default)


def detect_ultralarge_fifth_expansion_dim(state_dict: dict, default: int = 6144) -> int:
    key = "layers.10.adapter_up_e.weight"
    weight = state_dict.get(key)
    if isinstance(weight, torch.Tensor) and weight.ndim == 2:
        return int(weight.shape[0])
    return int(default)


def detect_megalarge_sixth_expansion_dim(state_dict: dict, default: int = 8192) -> int:
    key = "layers.11.adapter_up_f.weight"
    weight = state_dict.get(key)
    if isinstance(weight, torch.Tensor) and weight.ndim == 2:
        return int(weight.shape[0])
    return int(default)


def list_trainable_parameter_names(model: nn.Module) -> Sequence[str]:
    return [name for name, p in model.named_parameters() if p.requires_grad]
