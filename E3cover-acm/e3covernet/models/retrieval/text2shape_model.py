# -*- coding: utf-8 -*-
"""
Text2Shape 检索双塔模型 —— 主论文 §3.4 上支路 + 附录 B.3。

结构:
    形状塔:  E3CoverNet 骨干 → mean pool → Linear → 形状嵌入 S      (论文 Fig.1(d))
    文本塔:  BERT-base (冻结, 附录 Table 2 "Language encoder: frozen")
             → [CLS] 池化 → Linear 投影 → 文本嵌入 T
    对齐:    对称 InfoNCE (losses/info_nce.py), τ init 0.07

附录 B.3: "The retrieval setting is bimodal: the model uses only text and
3D shape inputs" —— 不使用图像/多视图监督, 因此没有第三塔。

依赖: pip install transformers (仅文本塔; 若你的数据接口直接给出文本
嵌入张量, 可用 text_encoder=None 跳过 BERT, 直接投影)。
"""

from typing import Optional

import torch
import torch.nn as nn

from ..backbone.e3covernet import E3CoverNet
from ..losses.info_nce import SymmetricInfoNCE


class Text2ShapeE3CoverNet(nn.Module):
    def __init__(self, embed_dim: int = 256, in_feat_dim: int = 0,
                 bert_name: str = "bert-base-uncased",
                 use_bert: bool = True):
        super().__init__()
        # ---- 形状塔: 论文骨干, 输出维度即联合嵌入维度 D ----
        self.shape_encoder = E3CoverNet(in_feat_dim=in_feat_dim,
                                        out_dim=embed_dim)

        # ---- 文本塔: 冻结 BERT-base + 可训练投影头 ----
        self.use_bert = use_bert
        if use_bert:
            from transformers import BertModel      # 延迟导入, 便于无网测试
            self.text_encoder = BertModel.from_pretrained(bert_name)
            for p in self.text_encoder.parameters():
                p.requires_grad_(False)             # 附录 Table 2: frozen
            text_dim = self.text_encoder.config.hidden_size  # 768
        else:
            self.text_encoder = None
            text_dim = 768                          # 数据接口直接提供 BERT 特征
        self.text_proj = nn.Linear(text_dim, embed_dim)

        # ---- 对齐目标: L_cont (Eq.12), τ init 0.07 ----
        self.criterion = SymmetricInfoNCE(init_temperature=0.07)

    def encode_shape(self, points: torch.Tensor) -> torch.Tensor:
        """points [B, N, 3+C] → 形状嵌入 [B, D]"""
        return self.shape_encoder(points)

    def encode_text(self, input_ids: Optional[torch.Tensor] = None,
                    attention_mask: Optional[torch.Tensor] = None,
                    text_feats: Optional[torch.Tensor] = None) -> torch.Tensor:
        """token ids ([B,L]) 或现成 BERT 特征 ([B,768]) → 文本嵌入 [B,D]"""
        if self.use_bert:
            out = self.text_encoder(input_ids=input_ids,
                                    attention_mask=attention_mask)
            cls = out.last_hidden_state[:, 0]       # [CLS] 池化
        else:
            assert text_feats is not None
            cls = text_feats
        return self.text_proj(cls)

    def forward(self, points, input_ids=None, attention_mask=None,
                text_feats=None):
        """返回 (loss, logits, shape_emb, text_emb); logits 直接用于 R@K。"""
        s = self.encode_shape(points)
        t = self.encode_text(input_ids, attention_mask, text_feats)
        loss, logits = self.criterion(t, s)
        return loss, logits, s, t
