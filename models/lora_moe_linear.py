"""LoRA-MoE adapter with single-layer gating heads."""

import torch
import torch.nn as nn

__all__ = ['LoRAMoEAdapterLinear']


class LoRAMoEAdapterLinear(nn.Module):
    """Gated mixture of ``num_lora`` LoRA experts.

    Args:
        in_features, out_features: shape of the projection being adapted.
        r, lora_alpha: LoRA rank and scaling (effective scale ``lora_alpha / r``).
        num_lora: number of experts.
        depth: number of blocks in the backbone.
        s, b: shape of the depth schedule ``beta(l) = sigmoid(s(1 - l/depth) - b)``.
        gate_proj_dim: width of the global gating projection, whose softmax is
            the target of the gate-diversity loss. Unused at inference.
    """

    def __init__(self, in_features, out_features, r=16, lora_alpha=16, num_lora=4,
                 depth=12, s=4.0, b=6.0, gate_proj_dim=100):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.num_lora = num_lora
        self.depth = depth
        self.s = s
        self.b = b
        self.scaling = lora_alpha / r

        self.lora_As = nn.ModuleList(
            [nn.Linear(in_features, r, bias=False) for _ in range(num_lora)])
        self.lora_Bs = nn.ModuleList(
            [nn.Linear(r, out_features, bias=False) for _ in range(num_lora)])
        self.lora_dropout = nn.Dropout(0.1)

        for A in self.lora_As:
            nn.init.normal_(A.weight, std=0.01)
        for B in self.lora_Bs:
            nn.init.zeros_(B.weight)

        self.global_gate = nn.Linear(in_features, num_lora)
        self.local_gate = nn.Linear(in_features, num_lora)
        self.global_proj = nn.Linear(num_lora, gate_proj_dim) if gate_proj_dim else None

    def compute_beta(self, block_index):
        return torch.sigmoid(torch.tensor(self.s * (1 - block_index / self.depth) - self.b))

    def forward(self, x, block_index=-1):
        """Returns ``(result, aux)``; ``aux`` holds the gate distributions."""
        global_gate_logits = self.global_gate(x.mean(dim=1))
        local_gate_logits = self.local_gate(x)

        global_gate_scores = torch.softmax(
            global_gate_logits, dim=-1, dtype=torch.float32).unsqueeze(1)
        local_gate_scores = torch.softmax(
            local_gate_logits, dim=-1, dtype=torch.float32)

        beta = self.compute_beta(block_index).to(x.device)
        gate_scores = beta * global_gate_scores + (1 - beta) * local_gate_scores

        if self.global_proj is not None:
            diversity_global = torch.softmax(
                self.global_proj(global_gate_logits), dim=-1, dtype=torch.float32)
        else:
            diversity_global = global_gate_scores.squeeze(1)

        lora_outputs = torch.stack(
            [self.lora_Bs[k](self.lora_As[k](self.lora_dropout(x))) * self.scaling
             for k in range(self.num_lora)], dim=-1)
        result = torch.sum(lora_outputs * gate_scores.unsqueeze(2), dim=-1)

        aux = {
            'global_gate_scores': diversity_global,
            'local_gate_scores': local_gate_scores,
            'gate_scores': gate_scores,
        }
        return result, aux
