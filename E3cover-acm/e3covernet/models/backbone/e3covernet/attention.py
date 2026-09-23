# -*- coding: utf-8 -*-
"""
几何感知注意力层 (Geometry-Aware Attention Layer) —— 主论文 §3.3, Eqs.(3)-(10)。

双路径同步更新 (论文 Figure 1(c) "dual-helix"):
    不变特征路径 (稳定性):  Eqs.(5)-(7)
    等变坐标路径 (敏感性):  Eqs.(8)-(10)

与基座 vgsatorras/egnn 的 E_GCL 的关系 (步骤 2 差异分析确认的 delta):
    1. sigmoid 门控         → softmax 多头注意力 (Eq.6)          [本文件实现]
    2. 原始 d^2 输入        → 16 维 RBF 编码 φ(d)                [rbf.py]
    3. 单头坐标更新         → 逐头标量门 w^h_ij (Eqs.8-9)        [本文件实现]
    4. 全连接图             → kNN(k=20) 稀疏图                    [本文件实现;
       附录 Remark 1: kNN 由 E(n)-不变的欧氏距离定义, 不破坏等变性]
    5. 保留 EGNN 的数值稳定技巧: 坐标更新使用 r_ij/(d_ij+1) 归一化
       (原仓库 models/egnn_clean/egnn_clean.py 中 coord2radial 的 norm 处理),
       防止近邻极近时梯度爆炸; 该缩放是不变标量乘子, 不影响等变性证明。

等变性 (附录 Proposition A.2):
    Layer(H, g·Z) = (H', g·Z')  ∀ g=(t,R)∈E(n)
    - 消息/注意力/门控只吃不变量 (H, φ(d), 覆盖描述子) ⇒ H' 不变 (Step 2)
    - Δz_i = Σ_h Σ_j α^h w^h r_ij 是相对位移的不变标量组合 ⇒ 等变 (Step 3)
    tests/test_equivariance.py 对该性质做了数值验证。
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .rbf import RadialBasisEncoding


# --------------------------------------------------------------------------
# kNN 图构建与邻居收集 (附录 Remark 1: E(n)-不变)
# --------------------------------------------------------------------------
def knn_graph(z: torch.Tensor, k: int) -> torch.Tensor:
    """基于欧氏距离的 kNN 索引 (排除自身)。

    Args:  z [B,N,3]
    Returns: idx [B,N,k]
    """
    with torch.no_grad():
        dist = torch.cdist(z, z)                               # [B,N,N]
        n = z.shape[1]
        eye = torch.eye(n, device=z.device, dtype=torch.bool).unsqueeze(0)
        dist = dist.masked_fill(eye, float("inf"))             # 排除 i=j (r_ii=0 无意义)
        k_eff = min(k, n - 1)
        idx = dist.topk(k_eff, dim=-1, largest=False).indices  # [B,N,k]
    return idx


def gather_neighbors(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """按邻居索引收集特征。 x [B,N,C], idx [B,N,k] → [B,N,k,C]"""
    b, n, c = x.shape
    k = idx.shape[-1]
    idx_flat = idx.reshape(b, n * k, 1).expand(-1, -1, c)      # [B,N*k,C]
    return torch.gather(x, 1, idx_flat).reshape(b, n, k, c)


def _mlp(dims, act=nn.SiLU):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


class GeometryAwareAttention(nn.Module):
    """单层几何感知注意力 (Algorithm 1 第 8-15 行的计算单元)。

    Args:
        dim:        特征维度 d_h (阶段相关: 64/128/256, 附录 Table 1)
        num_heads:  注意力头数 H_a (4/4/8)
        cover_dim:  本阶段覆盖集描述子的维度 K_n (3/8/10); 0 表示不用覆盖
        k:          kNN 邻居数 (论文 k=20)
        rbf_*:      本阶段 RBF 超参 (附录 Table 1)
        coord_scale: 坐标更新的整体缩放, 缓解三阶段累计漂移; 不变标量,
                     不影响等变性 (工程超参, 论文未规定)
    """

    def __init__(self, dim: int, num_heads: int, cover_dim: int = 0,
                 k: int = 20, rbf_d_min: float = 0.1, rbf_d_max: float = 2.0,
                 rbf_gamma: float = 0.5, cover_embed_dim: int = 16,
                 coord_scale: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0, "dim 必须能被 num_heads 整除"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.k = k
        self.coord_scale = coord_scale

        # φ(d): RBF 距离编码 (Eq.4 之后; 附录 B.2)
        self.rbf = RadialBasisEncoding(16, rbf_d_min, rbf_d_max, rbf_gamma)

        # 覆盖描述子嵌入: 把 K_n 维不变描述子映射到 cover_embed_dim,
        # 作为"群离散化位置编码"拼入消息输入 (步骤 2 确认的实现决策)
        self.cover_dim = cover_dim
        cov_e = 0
        if cover_dim > 0:
            self.cover_embed = nn.Sequential(
                nn.Linear(cover_dim, cover_embed_dim), nn.SiLU())
            cov_e = cover_embed_dim

        rbf_dim = self.rbf.out_dim
        # Eq.(5): m_ij = MLP_m(h_j, φ(d_ij) [, cov_j])
        self.mlp_m = _mlp([dim + rbf_dim + cov_e, dim, dim])
        # Eq.(6): α_ij = softmax_j( MLP_attn(h_i, h_j, φ(d_ij)) ), 逐头 logits
        self.mlp_attn = _mlp([2 * dim + rbf_dim, dim, num_heads])
        # Eq.(8): w^h_ij = MLP^h_w(h_j, φ(d_ij)), 逐头标量门
        self.mlp_w = _mlp([dim + rbf_dim, dim, num_heads])
        # Eq.(7) 中的 Linear(Σ_j α m)
        self.out_proj = nn.Linear(dim, dim)
        # post-norm (附录 Table 1: 各 Block 含 LayerNorm; 只作用于不变特征流,
        # 与等变性无关)
        self.norm = nn.LayerNorm(dim)

    def forward(self, h: torch.Tensor, z: torch.Tensor,
                cover_desc: Optional[torch.Tensor] = None,
                knn_idx: Optional[torch.Tensor] = None,
                z_dist: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: [B,N,D]  不变特征 H
            z: [B,N,3]  等变坐标 Z
            cover_desc: [B,N,K_n] 覆盖集不变描述子 (covering.py 输出), 可为 None
            knn_idx: 预计算的 kNN 索引 (同一 Block 内两层可复用同一张图)
            z_dist: [B,N,d_n] 用于计算 φ(d_ij) 的阶段投影坐标 (附录 B.1:
                    E(1) 阶段 φ 作用于 1D 投影 z^(1), 坐标更新仍用 3D r_ij);
                    None 时用 3D 坐标 z 计算距离
        Returns:
            h': [B,N,D] (不变),  z': [B,N,3] (等变)
        """
        b, n, d = h.shape
        if knn_idx is None:
            knn_idx = knn_graph(z, self.k)                     # [B,N,k]
        k = knn_idx.shape[-1]

        # ---------- 几何量 (Eqs.3-4) ----------
        z_j = gather_neighbors(z, knn_idx)                     # [B,N,k,3]
        r_ij = z_j - z.unsqueeze(2)                            # Eq.(3) 相对位移 (3D)
        d3_ij = r_ij.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        if z_dist is None:
            d_ij = d3_ij                                       # Eq.(4) 3D 距离
        else:                                                  # 阶段投影距离 |z^(n)_i - z^(n)_j|
            zd_j = gather_neighbors(z_dist, knn_idx)
            d_ij = (zd_j - z_dist.unsqueeze(2)).norm(dim=-1, keepdim=True).clamp(min=1e-8)
        phi = self.rbf(d_ij)                                   # [B,N,k,16] φ(d)

        h_j = gather_neighbors(h, knn_idx)                     # [B,N,k,D]
        h_i = h.unsqueeze(2).expand(-1, -1, k, -1)             # [B,N,k,D]

        # ---------- 不变特征路径 (Eqs.5-7) ----------
        if self.cover_dim > 0 and cover_desc is not None:
            cov_j = gather_neighbors(cover_desc, knn_idx)      # [B,N,k,K_n]
            cov_j = self.cover_embed(cov_j)                    # [B,N,k,cov_e]
            msg_in = torch.cat([h_j, phi, cov_j], dim=-1)
        else:
            msg_in = torch.cat([h_j, phi], dim=-1)
        m_ij = self.mlp_m(msg_in)                              # Eq.(5)  [B,N,k,D]

        logits = self.mlp_attn(torch.cat([h_i, h_j, phi], dim=-1))  # [B,N,k,H]
        alpha = torch.softmax(logits, dim=2)                   # Eq.(6) 对邻居 j 归一化

        # Eq.(7): H' = H + Linear( Σ_j α_ij · m_ij ), 多头按通道拆分实现
        m_heads = m_ij.view(b, n, k, self.num_heads, self.head_dim)
        agg = (alpha.unsqueeze(-1) * m_heads).sum(dim=2)       # [B,N,H,D/H]
        h_out = h + self.out_proj(agg.reshape(b, n, d))
        h_out = self.norm(h_out)                               # 不变流上的 LN

        # ---------- 等变坐标路径 (Eqs.8-10) ----------
        w = self.mlp_w(torch.cat([h_j, phi], dim=-1))          # Eq.(8)  [B,N,k,H]
        gate = (alpha * w).sum(dim=-1, keepdim=True)           # Σ_h α^h w^h  [B,N,k,1]
        # EGNN 数值稳定技巧: r_ij/(d_ij+1) —— 不变标量缩放, 不影响等变性
        r_hat = r_ij / (d3_ij + 1.0)
        delta_z = (gate * r_hat).sum(dim=2)                    # Eq.(9)  [B,N,3]
        z_out = z + self.coord_scale * delta_z                 # Eq.(10) Z' = Z + ΔZ

        return h_out, z_out
