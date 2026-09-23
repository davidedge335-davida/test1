# -*- coding: utf-8 -*-
"""
E(n)-Block 与 Lift 模块 —— 对应附录 B.2 Table 1 与 Lemma A.1 (残差分解)。

E(n)-Block:  2 × 几何感知注意力层 + FFN
    - 附录 Table 1: "Each E(n)-Block contains 2 geometry-aware attention
      layers (Algorithm 1) followed by a feedforward network"
    - FFN: MLP[d, 4d, d], GELU, Dropout 0.1 (三个阶段一致, 隐层为 4 倍宽)

Lift (ι_{n→n+1} 的架构实现):  f_{n+1}(Z) = ψ(f_n(π(Z))) + δ_{n+1}(Z)
    - 附录 Eq.(5): ψ 是逐点线性嵌入, δ_{n+1} 是可学习残差
    - 附录 B.2: "their final linear layers are zero-initialized so that
      δ_{n+1} ≈ 0 at the start of training" —— 保证初始化时上一阶段学到的
      E(n) 对称性被完整继承 (Lemma A.1 Part 2), 渐进扩张不破坏低阶特征
    - 实现说明: 论文中 δ_{n+1}(Z) 只需是 E(n+1)-等变函数即可 (证明第 8 行
      引用 Proposition A.2)。这里取其最简单的实例——作用在不变特征通道上的
      逐点 MLP (不变特征天然满足平凡表示下的等变性), 坐标流原样传递;
      这与 Algorithm 1 第 6 行 "Lift representation via ι 并加入阶段特定
      变化 δ_n" 的语义一致 (提升的是表示/特征, 坐标由注意力层负责演化)。
"""

import torch
import torch.nn as nn

from .attention import GeometryAwareAttention, knn_graph


class FeedForward(nn.Module):
    """FFN: MLP[d, 4d, d] + GELU + Dropout(0.1) + 残差 + LayerNorm (附录 Table 1)。"""

    def __init__(self, dim: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, expansion * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expansion * dim, dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.norm(h + self.net(h))


class EnBlock(nn.Module):
    """一个阶段的 E(n)-Block: [Attn ×2] → FFN。

    两层注意力共享同一张 kNN 图 (以 Block 输入坐标构建):
    - 单层内的坐标更新幅度受 coord_scale 控制, 图结构在 Block 粒度上刷新
      即可兼顾效率与几何一致性; 附录 Remark 1 保证任意时刻构图都等变。
    """

    def __init__(self, dim: int, num_heads: int, cover_dim: int,
                 num_layers: int = 2, k: int = 20,
                 rbf_d_min: float = 0.1, rbf_d_max: float = 2.0,
                 rbf_gamma: float = 0.5, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            GeometryAwareAttention(
                dim=dim, num_heads=num_heads, cover_dim=cover_dim, k=k,
                rbf_d_min=rbf_d_min, rbf_d_max=rbf_d_max, rbf_gamma=rbf_gamma)
            for _ in range(num_layers)
        ])
        self.ffn = FeedForward(dim, dropout=dropout)
        self.k = k

    def forward(self, h, z, cover_desc=None, z_dist=None):
        knn_idx = knn_graph(z, self.k)          # Block 级共享 kNN 图 (3D, 附录 Remark 1)
        for layer in self.layers:
            h, z = layer(h, z, cover_desc=cover_desc, knn_idx=knn_idx, z_dist=z_dist)
        h = self.ffn(h)                          # FFN 只作用于不变流
        return h, z


class Lift(nn.Module):
    """阶段间提升 ι_{n→n+1}: h ↦ ψ(h) + δ(h), δ 末层零初始化。

    对应附录 Eq.(5) 与 B.2 的零初始化说明:
        ψ:  Linear(d_in → d_out)                 —— 逐点线性嵌入
        δ:  MLP(d_in → d_out), 末层 weight/bias 置零
    ⇒ 初始时 Lift(h) = ψ(h): 纯线性重嵌入, E(n) 对称性无损继承 (Lemma A.1);
      训练中 δ 逐步注入阶段特定的新几何变化。
    """

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.psi = nn.Linear(d_in, d_out)
        self.delta = nn.Sequential(
            nn.Linear(d_in, d_out),
            nn.SiLU(),
            nn.Linear(d_out, d_out),
        )
        # ZeroInit(附录 B.2): δ_{n+1} ≈ 0 at start of training
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.psi(h) + self.delta(h)
