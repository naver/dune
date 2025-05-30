# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
from math import sqrt

import torch.nn.functional as F
from torch import Tensor, nn, softmax


logger = logging.getLogger("dinov2")


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        qk_norm: bool = False,
        norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.q_norm = norm_layer(head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(head_dim) if qk_norm else nn.Identity()

    def forward(self, x: Tensor, return_attention=False) -> Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )

        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = self.q_norm(q), self.k_norm(k)

        if return_attention:
            scale_factor = 1 / sqrt(q.size(-1))
            attn_weight = q @ k.transpose(-2, -1) * scale_factor
            return softmax(attn_weight, dim=-1)

        x = (
            F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.attn_drop, is_causal=False
            )
            .transpose(1, 2)
            .reshape(B, N, C)
        )

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None, return_attention=False) -> Tensor:
        if attn_bias is not None:
            raise AssertionError("xFormers is required for using nested tensors")
        return super().forward(x, return_attention=return_attention)
