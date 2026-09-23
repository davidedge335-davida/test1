# -*- coding: utf-8 -*-
"""
E3CoverNet 骨干 —— 主论文 Algorithm 1 的三阶段前向递归 + 附录 B.2 Table 1 规格。

阶段配置 (附录 Table 1):
    Init:    Point Embedding  MLP[3(+C), 64, 64] + LayerNorm + ReLU
    Stage 1: E(1)-Block(×2 attn)  dim 64,  4 heads, K1=3,  RBF γ=0.5, [0.1, 2.0]
    Stage 2: Lift 64→128 (ZeroInit δ); E(2)-Block  dim 128, 4 heads, K2=8
    Stage 3: Lift 128→256;             E(3)-Block  dim 256, 8 heads, K3=10,
             RBF γ=0.2, [0.1, 2.5] (coarse-to-fine 扩大感受野)
    Output:  Mean Pool + Linear[256, D]

Algorithm 1 逐行对应 (见 forward 内注释):
    1:  Z(0) ← P;  H(0) ← Embed(P)
    2:  for stage n = 1,2,3
    4:      C_n = 覆盖集 (covering.py)
    5-7:    n>1 时 Lift + δ_n (blocks.Lift, 零初始化残差)
    9-15:   几何感知注意力 (attention.py 双路径)
    17: return H(3)

接口契约 (与 e3covernet 基座对齐):
    输入  [B, N, 3+C]  (xyz + 可选颜色/法向等逐点特征)
    输出  [B, D]       (对齐 single_object_encoder 的输出维度)
    另提供 return_points=True 返回逐点特征 (检索任务备用 / 可视化)。
"""

from typing import Optional

import torch
import torch.nn as nn

from .blocks import EnBlock, Lift
from .covering import build_covering, pca_frame


# 附录 Table 1 的阶段规格, 单点集中管理便于消融 (论文 Table 4/5 的
# E(1)-only / E(1)⊂E(2) / no-covering 变体只需截断或置 cover_dim=0)
STAGE_SPECS = [
    #  dim, heads, K_n, rbf_gamma, rbf_d_max
    dict(dim=64,  heads=4, num_cover=3,  rbf_gamma=0.5, rbf_d_max=2.0),  # E(1)
    dict(dim=128, heads=4, num_cover=8,  rbf_gamma=0.5, rbf_d_max=2.0),  # E(2)
    dict(dim=256, heads=8, num_cover=10, rbf_gamma=0.2, rbf_d_max=2.5),  # E(3)
]


class E3CoverNet(nn.Module):
    """渐进式 E(1)⊂E(2)⊂E(3) 几何感知 3D 骨干。

    Args:
        in_feat_dim: 除 xyz 外的逐点输入特征维度 C (E3CoverNet 常用 rgb ⇒ 3;
                     纯坐标输入则为 0)
        out_dim:     全局特征输出维度 D (对齐下游融合模块/检索嵌入维度)
        k:           kNN 邻居数 (论文 20)
        num_stages:  使用的阶段数, 默认 3; 设 1/2 可复现 Table 4 的
                     "E(1) only" / "E(1)⊂E(2)" 消融
        use_covering: False 时复现 Table 4 的 "Vanilla (No covering)" 消融
    """

    def __init__(self, in_feat_dim: int = 3, out_dim: int = 256,
                 k: int = 20, num_stages: int = 3, use_covering: bool = True,
                 dropout: float = 0.1, embed_mode: str = "invariant",
                 stage_projected_distance: bool = True):
        """
        embed_mode:
            "invariant": Embed 输入 = [||p-p̄||, x̃] (协变 PCA 参考系坐标), 严格
                         E(3)-不变 ⇒ 整网满足 Prop. A.2 的前提;
            "raw":       Embed 输入 = 原始 xyz (Algorithm 1 第 1 行的字面实现,
                         不严格不变, 对应论文 "Robust ≠ Invariant" 的口径,
                         可保留手性信息)。
        stage_projected_distance:
            True 时 stage n 的 φ(d_ij) 用 n 维投影距离 (附录 B.1: E(1) 阶段
            φ 作用于 z^(1), E(2) 阶段作用于平面坐标), 坐标更新仍用 3D r_ij;
            False 时三阶段均用 3D 距离。
        """
        super().__init__()
        assert 1 <= num_stages <= 3
        assert embed_mode in ("invariant", "raw")
        self.num_stages = num_stages
        self.use_covering = use_covering
        self.embed_mode = embed_mode
        self.stage_projected_distance = stage_projected_distance

        # ---- Algorithm 1 第 1 行: H(0) ← Embed(P) ----
        # 附录 Table 1: MLP[3, 64, 64], LayerNorm, ReLU (有额外特征时 +C)。
        # ★ 实现决策: Proposition A.2 的前提是 H 为不变特征, 若 Embed 直接
        #   消费原始 xyz, H(0) 会随 g∈E(3) 变化, 整条不变性链条 (附录 A.2
        #   Step 2) 从第 0 层就被破坏 (tests/test_equivariance.py 可复现)。
        #   因此 Embed 的几何输入取 *不变量*:
        #     [ ||p_i - p̄||,  x̃_i (协变 PCA 参考系内坐标, 3 维) ]
        #   x̃ 与原坐标信息等价 (可逆线性变换), 但对任意 g∈E(3) 严格不变
        #   (covering.pca_frame 的协变符号规则保证, 含反射)。表中 "MLP[3,64,64]"
        #   的 3 维几何输入在此实现为参考系坐标, 额外拼 1 维半径。
        d0 = STAGE_SPECS[0]["dim"]
        geo_in = 4 if embed_mode == "invariant" else 3
        self.embed = nn.Sequential(
            nn.Linear(geo_in + in_feat_dim, d0), nn.ReLU(),
            nn.Linear(d0, d0), nn.LayerNorm(d0), nn.ReLU(),
        )

        self.coverings = nn.ModuleList()
        self.lifts = nn.ModuleList()   # lifts[n] 连接 stage n → n+1
        self.blocks = nn.ModuleList()
        prev_dim = d0
        for n in range(num_stages):
            spec = STAGE_SPECS[n]
            # Algorithm 1 第 4 行: 覆盖集 C_n
            self.coverings.append(
                build_covering(n + 1, spec["num_cover"]) if use_covering else nn.Identity())
            # Algorithm 1 第 5-7 行: 阶段间 Lift (stage 1 不需要)
            if n > 0:
                self.lifts.append(Lift(prev_dim, spec["dim"]))
            self.blocks.append(EnBlock(
                dim=spec["dim"], num_heads=spec["heads"],
                cover_dim=spec["num_cover"] if use_covering else 0,
                num_layers=2, k=k,
                rbf_d_min=0.1, rbf_d_max=spec["rbf_d_max"],
                rbf_gamma=spec["rbf_gamma"], dropout=dropout))
            prev_dim = spec["dim"]

        # ---- Output: Mean Pool + Linear[256, D] (附录 Table 1 末行) ----
        self.out_proj = nn.Linear(prev_dim, out_dim)
        self.out_dim = out_dim

    def forward(self, pc: torch.Tensor, return_points: bool = False):
        """
        Args:
            pc: [B, N, 3+C] 点云 (前 3 维必须是 xyz; 附录 B.3: 输入已
                center + unit-sphere 归一化, 该步骤属于数据接口, 本模块不重复做)
        Returns:
            global_feat: [B, D];  若 return_points 另返回 (H(3) [B,N,d3], Z(3))
        """
        z = pc[..., :3].contiguous()          # Algorithm 1 第 1 行: Z(0) ← P
        feats = pc[..., 3:]                                       # 非几何通道 (rgb 等)
        if self.embed_mode == "invariant":
            # H(0) ← Embed(不变量): [半径, 协变参考系坐标 x̃]
            mean, evecs = pca_frame(z)
            x_frame = torch.einsum("bnd,bdk->bnk", z - mean, evecs)
            radius = (z - mean).norm(dim=-1, keepdim=True)
            h = self.embed(torch.cat([radius, x_frame, feats], dim=-1))
        else:
            h = self.embed(pc)                # H(0) ← Embed(P), 字面实现

        for n in range(self.num_stages):      # Algorithm 1 第 2 行: n = 1,2,3
            if n > 0:
                h = self.lifts[n - 1](h)      # 第 5-7 行: ι_{n-1→n} + δ_n
            # 第 4 行: 覆盖描述子 (由当前坐标 Z(n-1) 计算, E(3)-不变)
            cov = self.coverings[n](z) if self.use_covering else None
            # 阶段投影坐标 z^(n) ∈ R^n (附录 B.1), 用于 φ(d_ij); n=3 退化为 3D
            z_dist = None
            if self.stage_projected_distance and n < 2:
                mean, evecs = pca_frame(z)
                z_dist = torch.einsum("bnd,bdk->bnk", z - mean, evecs[..., :n + 1])
            # 第 9-15 行: 双路径几何感知注意力
            h, z = self.blocks[n](h, z, cover_desc=cov, z_dist=z_dist)

        pooled = h.mean(dim=1)                # Mean Pool (不变量的均值仍不变)
        global_feat = self.out_proj(pooled)   # Linear[256, D]
        if return_points:
            return global_feat, h, z          # 第 17 行: H(3) 供下游使用
        return global_feat
