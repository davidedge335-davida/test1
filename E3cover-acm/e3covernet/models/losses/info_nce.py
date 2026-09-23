# -*- coding: utf-8 -*-
"""
对称 InfoNCE 对比损失 —— 主论文 §3.4 "Text-to-Shape Retrieval" 的 L_cont
(Eq.(12)) + 附录 B.3 的可学习温度 τ (初始化 0.07)。

    L = 1/2 [ CE(S/τ, y) + CE(Sᵀ/τ, y) ]
其中 S_ij = <t_i, s_j> 为 L2 归一化后的文本-形状余弦相似度矩阵,
y_i = i (batch 内一一配对), 双向 (text→shape 与 shape→text) 取平均,
即 CLIP [26] 风格的对称对比目标; 温度按 log 参数化保证 τ>0。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SymmetricInfoNCE(nn.Module):
    def __init__(self, init_temperature: float = 0.07,
                 max_logit_scale: float = 100.0):
        super().__init__()
        # 学习 log(1/τ), 与 CLIP 一致; τ=0.07 ⇒ logit_scale ≈ 14.28
        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(1.0 / init_temperature)))
        self.max_logit_scale = max_logit_scale

    @property
    def temperature(self) -> torch.Tensor:
        return 1.0 / self.logit_scale.exp()

    def forward(self, text_emb: torch.Tensor, shape_emb: torch.Tensor):
        """
        Args:
            text_emb:  [B, D] 文本嵌入 T
            shape_emb: [B, D] 形状嵌入 S (对角线为匹配对)
        Returns:
            loss (标量), logits [B, B] (供 R@K 评测复用)
        """
        t = F.normalize(text_emb, dim=-1)
        s = F.normalize(shape_emb, dim=-1)
        # 数值安全: 限制 1/τ 上界 (CLIP 同款技巧)
        scale = self.logit_scale.exp().clamp(max=self.max_logit_scale)
        logits = scale * t @ s.t()                        # S/τ, [B,B]
        target = torch.arange(t.shape[0], device=t.device)
        # 对称: text→shape 按行, shape→text 按列 (Eq.12 的双向求和)
        loss = 0.5 * (F.cross_entropy(logits, target) +
                      F.cross_entropy(logits.t(), target))
        return loss, logits
