"""Slot Attention (Locatello et al., 2020 -- arXiv:2006.15055).

The iterative attention module used by ``SlotAttentionEncoder`` to bind a
feature map into a fixed set of permutation-invariant slot vectors. Also
provides the soft positional embedding the feature map is augmented with
before binding: slots localise purely through attention weights, so the
spatial position of each feature has to be encoded into the features
themselves.
"""

import torch
from torch import nn


def build_grid(resolution):
    """Normalised coordinate grid of shape ``[1, H, W, 4]``.

    The last axis stacks ``(x, y, 1 - x, 1 - y)`` so the linear projection in
    ``SoftPositionEmbed`` can express any axis-aligned position without a bias
    term. ``resolution`` is an ``(H, W)`` pair.
    """
    ranges = [torch.linspace(0.0, 1.0, steps=r) for r in resolution]
    grid_y, grid_x = torch.meshgrid(*ranges, indexing="ij")
    grid = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]
    grid = grid.unsqueeze(0)  # [1, H, W, 2]
    return torch.cat([grid, 1.0 - grid], dim=-1)  # [1, H, W, 4]


class SoftPositionEmbed(nn.Module):
    """Add a learned linear projection of a coordinate grid to a feature map."""

    def __init__(self, hidden_size, resolution):
        super().__init__()
        self.proj = nn.Linear(4, hidden_size)
        # Deterministic grid -- kept out of the state dict (non-persistent).
        self.register_buffer("grid", build_grid(resolution), persistent=False)

    def forward(self, x):
        # x: [B, H, W, C]
        return x + self.proj(self.grid.to(x.dtype))


class SlotAttention(nn.Module):
    """Iterative slot attention binding ``N`` input tokens into ``K`` slots.

    Each iteration: slots emit queries, the input tokens are keys/values, and
    attention is normalised *over the slot axis* so slots compete to explain
    each token. The per-slot weighted mean of values updates the slot through a
    GRU cell followed by a residual MLP.
    """

    def __init__(self, num_slots, dim, iters=3, eps=1e-8, hidden_dim=128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim**-0.5

        # Slots are sampled each forward pass from a learned Gaussian shared
        # across slots -- this is what keeps them permutation-invariant.
        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.xavier_uniform_(self.slots_logsigma)

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim),
        )

        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, inputs, num_slots=None, return_attn=False):
        # inputs: [B, N, D]
        b, n, d = inputs.shape
        n_s = num_slots if num_slots is not None else self.num_slots

        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_logsigma.exp().expand(b, n_s, -1)
        slots = mu + sigma * torch.randn(
            mu.shape, device=inputs.device, dtype=inputs.dtype
        )

        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)

        last_attn = None
        for _ in range(self.iters):
            slots_prev = slots
            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            # Attention logits, softmax over the slot axis: slots compete.
            dots = torch.einsum("bid,bjd->bij", q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            # The pre-renormalisation softmax is the per-token slot
            # competition -- i.e. the soft segmentation mask.
            last_attn = attn
            # Re-normalise over tokens -> per-slot weighted mean of values.
            attn = attn / attn.sum(dim=-1, keepdim=True)
            updates = torch.einsum("bjd,bij->bid", v, attn)

            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d),
            )
            slots = slots.reshape(b, n_s, d)
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        if return_attn:
            # [B, K, N] competition weights from the final iteration.
            return slots, last_attn
        return slots
