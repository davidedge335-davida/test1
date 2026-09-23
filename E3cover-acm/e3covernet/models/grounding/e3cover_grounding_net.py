# -*- coding: utf-8 -*-
"""
3D 视觉定位网络 —— 论文 backbone-level 统一管线 (§3.4 下支路, Table 3 设置):
    per-object 3D 编码器 (E3CoverNet, 可换 baseline) → {F_3D^(k)}
    冻结 BERT-base → 文本 token 特征 → 线性投影到 D
    跨注意力融合 (fusion/cross_attention_fusion.py) → referred-object logits
    损失: 对 target 索引的交叉熵 (基座 e3covernet 的主损失口径一致)

附录 B.3: "only the 3D encoder and the cross-attention fusion module are
trained" ⇒ BERT 冻结, 只有 encoder / text_proj / fusion 参与优化。

张量接口 (与 e3covernet 官方 dataloader 的 batch 字段对齐, 见 in_out/
pt_datasets/listening_dataset.py 的 __getitem__):
    objects:     [B, K, N, 3+C]   K 个候选物体点云 (xyz + rgb, C=3)
    obj_mask:    [B, K]           True = 有效候选 (基座用 context_size 表达,
                                  可由 arange(K) < context_size 得到)
    input_ids / attention_mask:  BERT tokenizer 输出 [B, L]
    target_pos:  [B]              目标物体在 K 中的下标
数据加载/预处理按约束不在此实现。
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbone.e3covernet import E3CoverNet
from ..fusion.cross_attention_fusion import CrossAttentionFusion


class E3CoverGroundingNet(nn.Module):
    def __init__(self, fusion_dim: int = 256, in_feat_dim: int = 3,
                 bert_name: str = "bert-base-uncased", use_bert: bool = True,
                 object_encoder: Optional[nn.Module] = None,
                 encoder_kwargs: Optional[dict] = None,
                 fusion_layers: int = 2, fusion_heads: int = 8, dropout: float = 0.1,
                 objects_chunk: int = 64):
        super().__init__()
        # ---- 3D 编码器: 默认 E3CoverNet; 传入 object_encoder 可做 backbone-swap ----
        if object_encoder is None:
            object_encoder = E3CoverNet(in_feat_dim=in_feat_dim, out_dim=fusion_dim,
                                        **(encoder_kwargs or {}))
        self.object_encoder = object_encoder
        self.objects_chunk = objects_chunk    # 分块编码 B*K 个物体, 控制显存

        # ---- 文本: 冻结 BERT-base (附录 Table 2) + 可训练投影 ----
        self.use_bert = use_bert
        if use_bert:
            from transformers import BertModel
            self.text_encoder = BertModel.from_pretrained(bert_name)
            for p in self.text_encoder.parameters():
                p.requires_grad_(False)
            text_dim = self.text_encoder.config.hidden_size
        else:
            self.text_encoder = None
            text_dim = 768                    # 数据接口直接提供 token 特征 [B,L,768]
        self.text_proj = nn.Linear(text_dim, fusion_dim)

        # ---- 跨注意力融合 (论文管线中固定的模块, 参与训练) ----
        self.fusion = CrossAttentionFusion(fusion_dim, fusion_layers, fusion_heads, dropout)

    def encode_objects(self, objects: torch.Tensor) -> torch.Tensor:
        """[B,K,N,3+C] → [B,K,D]  (等价于基座 utils.get_siamese_features)"""
        b, k = objects.shape[:2]
        flat = objects.reshape(b * k, *objects.shape[2:])
        feats = [self.object_encoder(flat[i:i + self.objects_chunk])
                 for i in range(0, b * k, self.objects_chunk)]
        return torch.cat(feats, 0).reshape(b, k, -1)

    def encode_text(self, input_ids=None, attention_mask=None, text_feats=None):
        if self.use_bert:
            out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            tok = out.last_hidden_state                       # [B,L,768]
        else:
            tok = text_feats
        return self.text_proj(tok)

    def forward(self, objects, obj_mask=None, input_ids=None, attention_mask=None,
                text_feats=None, target_pos=None):
        obj_feats = self.encode_objects(objects)              # {F_3D^(k)}
        text_tokens = self.encode_text(input_ids, attention_mask, text_feats)
        text_pad = (attention_mask == 0) if attention_mask is not None else None
        obj_pad = (~obj_mask) if obj_mask is not None else None
        logits, _ = self.fusion(text_tokens, obj_feats, text_pad, obj_pad)  # [B,K]
        out = {"logits": logits}
        if target_pos is not None:
            out["loss"] = F.cross_entropy(logits, target_pos)
            out["acc"] = (logits.argmax(-1) == target_pos).float().mean()
        return out
