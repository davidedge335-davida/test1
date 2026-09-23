# -*- coding: utf-8 -*-
"""
跨注意力融合模块 —— 主论文 §3.4 "3D Visual Grounding":
    "The set {F_3D^(k)} is fused with language tokens through a cross-attention
     module, where text queries attend over geometry-sensitive 3D features
     for referred-object prediction."

这是论文 backbone-level 统一管线 (Table 3) 中"固定不变"的融合模块。e3covernet
官方仓库的 E3CoverNet baseline 使用的是 DGCNN 物体图融合, 与论文管线不同 (PointNet++
在官方管线上 SR3D ≈ 40%, 在论文管线上 48.5%), 因此这里按论文描述实现一个独立
的跨注意力融合, 供 grounding/e3cover_grounding_net.py 使用。

结构 (每层):
    text  ← text  + CrossAttn(Q=text, K/V=objects)      # 文本查询关注 3D 特征
    text  ← text  + FFN(text)
    objs  ← objs  + CrossAttn(Q=objects, K/V=text)      # 对象回读文本, 便于逐对象打分
    objs  ← objs  + FFN(objs)
最后由文本池化向量 q 与每个对象特征做点积得到 referred-object logits。
论文未给出层数/头数, 取 2 层 / 8 头 (MVT/ViL3DRel 等同类管线的常用配置)。
"""

import torch
import torch.nn as nn


class _CrossBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.t2o = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.o2t = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn_t = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(),
                                   nn.Dropout(dropout), nn.Linear(4 * dim, dim))
        self.ffn_o = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(),
                                   nn.Dropout(dropout), nn.Linear(4 * dim, dim))
        self.n1, self.n2, self.n3, self.n4 = (nn.LayerNorm(dim) for _ in range(4))
        self.drop = nn.Dropout(dropout)

    def forward(self, text, objs, text_pad_mask, obj_pad_mask):
        # 文本查询关注几何敏感 3D 特征 (论文原句)
        a, _ = self.t2o(text, objs, objs, key_padding_mask=obj_pad_mask)
        text = self.n1(text + self.drop(a))
        text = self.n2(text + self.drop(self.ffn_t(text)))
        # 对象回读文本 (使每个对象特征携带查询语义, 用于逐对象打分)
        a, _ = self.o2t(objs, text, text, key_padding_mask=text_pad_mask)
        objs = self.n3(objs + self.drop(a))
        objs = self.n4(objs + self.drop(self.ffn_o(objs)))
        return text, objs


class CrossAttentionFusion(nn.Module):
    """
    Args:
        dim: 融合维度 D (3D 特征与文本投影后的公共维度)
        num_layers / num_heads / dropout: 融合层超参 (论文未规定)
    forward:
        text_tokens: [B,L,D]  text_pad_mask: [B,L] True=padding
        obj_feats:   [B,K,D]  obj_pad_mask:  [B,K] True=padding (无效 proposal)
    returns:
        logits [B,K]  (padding 位置为 -inf), fused_objs [B,K,D]
    """

    def __init__(self, dim: int = 256, num_layers: int = 2, num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [_CrossBlock(dim, num_heads, dropout) for _ in range(num_layers)])
        self.query_proj = nn.Linear(dim, dim)
        self.obj_proj = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, text_tokens, obj_feats, text_pad_mask=None, obj_pad_mask=None):
        text, objs = text_tokens, obj_feats
        for layer in self.layers:
            text, objs = layer(text, objs, text_pad_mask, obj_pad_mask)
        # 文本 masked-mean 池化 → 查询向量 q; 逐对象打分 s_k = <W_q q, W_o F_k>
        if text_pad_mask is not None:
            keep = (~text_pad_mask).float().unsqueeze(-1)
            q = (text * keep).sum(1) / keep.sum(1).clamp(min=1.0)
        else:
            q = text.mean(1)
        logits = torch.einsum("bd,bkd->bk", self.query_proj(q), self.obj_proj(objs)) * self.scale
        if obj_pad_mask is not None:
            logits = logits.masked_fill(obj_pad_mask, float("-inf"))
        return logits, objs
