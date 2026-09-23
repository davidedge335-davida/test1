# -*- coding: utf-8 -*-
"""
RBF 距离编码 φ(d) —— 对应主论文 Eq.(4) 之后的描述与附录 B.2。

论文规定:
    φ(d) = [exp(-γ (d - c_k)^2)]_{k=1..16}
其中 16 个中心按对数间隔排布:
    c_k = d_min * (d_max / d_min)^{(k-1)/15}
带宽 γ 分阶段: Stage1/2 用 γ=0.5, d∈[0.1, 2.0]; Stage3 用 γ=0.2, d∈[0.1, 2.5]
(coarse-to-fine: 末阶段扩大几何感受野, 见附录 B.2 表 1 脚注)。

等变性说明 (附录 A.2, Step 1):
    d_ij = ||z_i - z_j|| 在任意 g=(t,R)∈E(n) 作用下不变 (R∈O(n) 保范数,
    含反射 det R = ±1), 因此 φ(d) 是严格 E(n)-不变量, 可以安全地喂给
    任何 MLP 而不破坏整层的等变性。
"""

import torch
import torch.nn as nn


class RadialBasisEncoding(nn.Module):
    """把标量距离 d 编码为 16 维平滑不变特征。

    Args:
        num_centers: RBF 中心数量 K (论文固定 16)
        d_min, d_max: 中心覆盖的距离范围 (点云已归一化到单位球, 见附录 B.3
                      "centered and rescaled to unit sphere", 故 2.0/2.5 足够
                      覆盖球内最大点对距离)
        gamma: 高斯核带宽 γ
        learnable: 是否让中心/带宽可学习。论文未明确说明, 默认冻结
                   (buffer), 与 SchNet [30] 的常规做法一致。
    """

    def __init__(self, num_centers: int = 16, d_min: float = 0.1,
                 d_max: float = 2.0, gamma: float = 0.5,
                 learnable: bool = False):
        super().__init__()
        # 对数间隔中心: c_k = d_min * (d_max/d_min)^{(k-1)/(K-1)}  (附录 B.2)
        exponents = torch.arange(num_centers, dtype=torch.float32) / (num_centers - 1)
        centers = d_min * (d_max / d_min) ** exponents            # [K]
        gamma_t = torch.tensor(float(gamma))
        if learnable:
            self.centers = nn.Parameter(centers)
            self.gamma = nn.Parameter(gamma_t)
        else:
            self.register_buffer("centers", centers)
            self.register_buffer("gamma", gamma_t)
        self.num_centers = num_centers

    @property
    def out_dim(self) -> int:
        return self.num_centers

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        """
        Args:
            d: [..., 1] 成对欧氏距离 (最后一维为 1)
        Returns:
            [..., K] RBF 编码 φ(d)
        """
        # φ(d)_k = exp(-γ (d - c_k)^2)
        return torch.exp(-self.gamma * (d - self.centers) ** 2)
