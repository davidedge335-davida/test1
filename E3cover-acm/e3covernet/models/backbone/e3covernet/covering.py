# -*- coding: utf-8 -*-
"""
可学习覆盖集 C_n ⊂ E(n) —— 对应主论文 §3.2 与附录 B.1。

论文设定 (附录 B.1):
    E(1): K1=3  个平移锚点, 初始化在 PCA 主轴投影坐标的 {Q25, Q50, Q75} 分位数;
          O(1) 的反射 {+1,-1} 被 RBF 的对称性吸收 (φ(|z_i-z_j|)=φ(|z_j-z_i|))。
    E(2): K2=8  个 SO(2) 旋转角, 均匀初始化 θ_k = 2πk/8; O(2) 反射由注意力层的
          距离不变性处理。
    E(3): K3=10 个 O(3) 覆盖胞, 旋转轴用 Fibonacci 格点初始化 (附录 Eq.17):
              z_i = 1 - 2i/(K3+1),  φ_i = 2πi/Φ,  Φ=(1+√5)/2
          每个覆盖元素 = 固定轴 v_i + 一个可学习旋转角。

╔══════════════════════════════════════════════════════════════════════════╗
║ 实现决策 (步骤 2 已确认): 论文主文对"覆盖元素如何进入注意力计算"是抽象的  ║
║ ("organizes geometric priors")。为了既忠实又严格保持附录 A.2 的等变性,    ║
║ 我们让每个覆盖元素 g_k 产生一个 *不变* 几何描述子:                        ║
║   - 先构造数据依赖的等变参考系 (质心 + PCA 特征向量), 该参考系随输入点云  ║
║     一起旋转/平移 (协变), 因此"在参考系内计算的量"对全局 E(3) 变换不变;   ║
║   - E(1): c_ik = |proj_v1(z_i) - t_k|          (到平移锚点的 1D 距离)     ║
║   - E(2): c_ik = ||u_i - R(θ_k) u_i||          (SO(2) 覆盖旋转引起的弦位移)║
║   - E(3): c_ik = ||x̃_i - R(v_k, α_k) x̃_i||    (SO(3) 覆盖旋转引起的位移)  ║
║ 这些 K_n 维描述子作为"群离散化位置编码"拼进消息 MLP (Eq.5), 由于全部是    ║
║ 范数/绝对值, 全局 g∈E(3) 作用后数值不变 ⇒ 不破坏 Proposition A.2。        ║
║ PCA 参考系做 detach 处理: 视作输入的确定性函数 (类似数据预处理), 避免     ║
║ eigh 反传的数值不稳定; 锚点/角度本身仍是可学习参数, 与论文"learnable      ║
║ covering elements refined during training"一致。                          ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import math
import torch
import torch.nn as nn


# --------------------------------------------------------------------------
# 等变参考系: 质心 + PCA 特征向量 (协变于输入 ⇒ 系内坐标不变)
# --------------------------------------------------------------------------
@torch.no_grad()
def pca_frame(z: torch.Tensor):
    """计算每个点云的质心与 PCA 参考系。

    Args:
        z: [B, N, 3] 点坐标
    Returns:
        mean:  [B, 1, 3] 质心
        evecs: [B, 3, 3] 列向量为特征向量, 按特征值从大到小排列

    注: 整体 no_grad + detach —— 参考系是输入的确定性函数, 等变性由
        "frame 随 g 协变 ⇒ frame 内坐标不变" 保证, 不需要梯度流过 eigh。
        符号歧义 (PCA 特征向量 ±v 均合法) 用"最大绝对值分量取正"规则固定;
        反射引起的残余歧义被下游的范数/绝对值运算吸收 (附录 B.1 对 O(1)/O(2)
        反射的处理方式与此一致)。
    """
    mean = z.mean(dim=1, keepdim=True)                      # [B,1,3]
    zc = z - mean
    cov = torch.einsum("bni,bnj->bij", zc, zc) / z.shape[1]  # [B,3,3]
    # eigh 返回特征值升序; flip 后按主轴、次轴、第三轴降序排列
    _, evecs = torch.linalg.eigh(cov)
    evecs = evecs.flip(-1)                                   # [B,3,3]
    # 固定符号 —— 必须用 *协变* 规则 (随数据一起变换), 否则参考系不等变:
    # 取每个轴上投影分布的三阶矩 (偏度) 为正:
    #     sign_j = sign( Σ_i ⟨z_i - z̄, v_j⟩³ )
    # 投影值 ⟨z_i - z̄, v_j⟩ 在全局 g=(t,R) 下不变 (v_j 随数据协变旋转),
    # 数据被反射时投影同号翻转 ⇒ 符号选择跟随反射, 参考系严格协变,
    # 系内坐标因此对任意 g∈E(3) 不变 (含 det R = -1 的反射)。
    proj = torch.einsum("bnd,bdk->bnk", zc, evecs)           # [B,N,3]
    skew = (proj ** 3).sum(dim=1, keepdim=True)              # [B,1,3]
    sign = torch.where(skew.abs() > 1e-6, skew.sign(),
                       torch.ones_like(skew))                # 近对称时退化取 +1
    evecs = evecs * sign
    return mean, evecs


# --------------------------------------------------------------------------
# E(1) 覆盖: K1=3 平移锚点 (附录 B.1 "E(1) Covering")
# --------------------------------------------------------------------------
class CoveringE1(nn.Module):
    def __init__(self, num_anchors: int = 3):
        super().__init__()
        self.num_anchors = num_anchors
        # 可学习偏移, 初始化为 0 ⇒ 训练起点严格等于论文的分位数初始化
        self.anchor_offset = nn.Parameter(torch.zeros(num_anchors))
        # 分位点 {0.25, 0.5, 0.75} (对 num_anchors=3)
        q = torch.arange(1, num_anchors + 1, dtype=torch.float32) / (num_anchors + 1)
        self.register_buffer("quantiles", q)

    @property
    def out_dim(self) -> int:
        return self.num_anchors

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:  z [B,N,3]
        Returns: c [B,N,K1] —— 点到各平移锚点的 1D 距离 (E(3)-不变)
        """
        mean, evecs = pca_frame(z)
        primary_axis = evecs[..., 0]                          # [B,3] 主轴
        proj = torch.einsum("bnd,bd->bn", z - mean, primary_axis)  # [B,N] 1D 投影
        # 每个点云各自的分位数锚点 (detach: 锚点基座由数据决定, 论文的
        # "initialized at quartile positions"; 可学习部分在 offset 上)
        base = torch.quantile(proj.detach(), self.quantiles.to(proj.dtype), dim=1)  # [K1,B]
        base = base.transpose(0, 1)                          # [B,K1]
        anchors = base + self.anchor_offset                  # [B,K1]
        # c_ik = |z^(1)_i - t_k| ; O(1) 反射 (proj → -proj) 引起的整体翻转
        # 会同步翻转分位数锚点, |·| 保证描述子不变
        return (proj.unsqueeze(-1) - anchors.unsqueeze(1)).abs()   # [B,N,K1]


# --------------------------------------------------------------------------
# E(2) 覆盖: K2=8 个 SO(2) 旋转角 (附录 B.1 "E(2) Covering")
# --------------------------------------------------------------------------
class CoveringE2(nn.Module):
    def __init__(self, num_angles: int = 8):
        super().__init__()
        self.num_angles = num_angles
        # θ_k = 2πk/8 均匀初始化, 训练中可学习 (论文: "all covering elements
        # are subsequently refined as learnable parameters")
        init = 2.0 * math.pi * torch.arange(num_angles, dtype=torch.float32) / num_angles
        self.angles = nn.Parameter(init)

    @property
    def out_dim(self) -> int:
        return self.num_angles

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:  z [B,N,3]
        Returns: c [B,N,K2] —— 覆盖旋转 R(θ_k) 作用在主轴平面坐标上
                              引起的弦位移范数 ||u_i - R(θ_k)u_i|| (不变)
        """
        mean, evecs = pca_frame(z)
        u = torch.einsum("bnd,bdk->bnk", z - mean, evecs[..., :2])  # [B,N,2] 平面坐标
        ux, uy = u[..., 0:1], u[..., 1:2]                    # [B,N,1]
        cos = torch.cos(self.angles).view(1, 1, -1)          # [1,1,K2]
        sin = torch.sin(self.angles).view(1, 1, -1)
        # R(θ)u = (ux cosθ - uy sinθ, ux sinθ + uy cosθ)
        rx = ux * cos - uy * sin
        ry = ux * sin + uy * cos
        dx, dy = ux - rx, uy - ry
        return torch.sqrt(dx * dx + dy * dy + 1e-12)         # [B,N,K2]


# --------------------------------------------------------------------------
# E(3) 覆盖: K3=10 个 Fibonacci 轴 + 可学习角 (附录 B.1, Eq.17)
# --------------------------------------------------------------------------
class CoveringE3(nn.Module):
    def __init__(self, num_cells: int = 10):
        super().__init__()
        self.num_cells = num_cells
        # ---- 附录 Eq.(17): Fibonacci 格点采样 S^2 上的旋转轴 ----
        #   z_i = 1 - 2i/(K3+1),  φ_i = 2πi/Φ,  Φ 黄金比
        i = torch.arange(1, num_cells + 1, dtype=torch.float32)
        zc = 1.0 - 2.0 * i / (num_cells + 1)
        phi = 2.0 * math.pi * i / ((1.0 + math.sqrt(5.0)) / 2.0)
        r = torch.sqrt(torch.clamp(1.0 - zc * zc, min=0.0))
        axes = torch.stack([r * torch.cos(phi), r * torch.sin(phi), zc], dim=-1)  # [K3,3]
        # 轴方向可学习 (forward 中重新归一化保持在 S^2 上)
        self.axes = nn.Parameter(axes)
        # 每个覆盖胞附带一个可学习旋转角, 初始化 π/2 (论文只说 "an associated
        # learnable angle", 未给初值; π/2 让初始位移在 0 与最大值之间)
        self.cell_angles = nn.Parameter(torch.full((num_cells,), math.pi / 2.0))

    @property
    def out_dim(self) -> int:
        return self.num_cells

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:  z [B,N,3]
        Returns: c [B,N,K3] —— Rodrigues 旋转 R(v_k, α_k) 作用于参考系坐标
                              引起的位移范数 (E(3)-不变)
        """
        mean, evecs = pca_frame(z)
        x = torch.einsum("bnd,bdk->bnk", z - mean, evecs)    # [B,N,3] 参考系坐标 x̃
        v = torch.nn.functional.normalize(self.axes, dim=-1)  # [K3,3] 单位轴
        v = v.view(1, 1, self.num_cells, 3)                  # [1,1,K3,3]
        a = self.cell_angles.view(1, 1, self.num_cells, 1)   # [1,1,K3,1]
        xe = x.unsqueeze(2)                                  # [B,N,1,3]
        # Rodrigues: R x = x cosα + (v × x) sinα + v (v·x)(1-cosα)
        dot = (v * xe).sum(dim=-1, keepdim=True)             # [B,N,K3,1]
        cross = torch.cross(v.expand_as(xe.expand(-1, -1, self.num_cells, -1)),
                            xe.expand(-1, -1, self.num_cells, -1), dim=-1)
        rot = xe * torch.cos(a) + cross * torch.sin(a) + v * dot * (1.0 - torch.cos(a))
        return (xe - rot).norm(dim=-1)                       # [B,N,K3]


def build_covering(stage: int, num_elements: int) -> nn.Module:
    """按阶段号构造覆盖集 (Algorithm 1 第 4 行: obtain learned covering set C_n)。"""
    if stage == 1:
        return CoveringE1(num_elements)
    if stage == 2:
        return CoveringE2(num_elements)
    if stage == 3:
        return CoveringE3(num_elements)
    raise ValueError(f"stage must be 1/2/3, got {stage}")
