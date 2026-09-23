# -*- coding: utf-8 -*-
"""
Text2Shape 检索训练脚本 (极简版) —— 主论文 §3.4 上支路 + 附录 Table 2 检索列。

按约束: **不包含任何 Dataset/DataLoader 实现**。数据接口由外部提供,
只需满足下面 `get_batch_iterator` 的张量协议:

    points:         [B, 2048, 3]  单位球归一化点云 (附录 B.3; N=2048)
    input_ids:      [B, L]        BERT tokenizer 输出
    attention_mask: [B, L]

超参 (附录 Table 2, Retrieval 列):
    AdamW, lr 3e-4, weight decay 1e-2, cosine + 5 epoch warmup,
    120 epochs, batch 64, τ init 0.07, BERT-base frozen。

用法:
    python train_text2shape.py                 # 用随机张量跑通冒烟测试
    实际训练时把 get_batch_iterator 换成你的数据接口即可, 其余零改动。
"""

import argparse
import math
import sys
from pathlib import Path

import torch

# 让脚本可以独立运行 (仓库内运行时可去掉这两行, 走正常包导入)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from e3covernet.models.retrieval.text2shape_model import Text2ShapeE3CoverNet  # noqa: E402


# ---------------------------------------------------------------------------
# 数据接口占位: 替换为你现成的数据管道 (约束: 本仓库不实现数据处理)
# ---------------------------------------------------------------------------
def get_batch_iterator(num_batches: int, batch_size: int, num_points: int,
                       seq_len: int, device: torch.device):
    """Yield (points, input_ids, attention_mask) —— 这里用随机张量占位。"""
    for _ in range(num_batches):
        points = torch.randn(batch_size, num_points, 3, device=device)
        # 单位球归一化 (附录 B.3): 数据接口应已完成, 占位数据在此模拟
        points = points - points.mean(dim=1, keepdim=True)
        points = points / points.norm(dim=-1).amax(dim=1)[:, None, None]
        input_ids = torch.randint(0, 30000, (batch_size, seq_len), device=device)
        attention_mask = torch.ones_like(input_ids)
        yield points, input_ids, attention_mask


def build_scheduler(optimizer, warmup_epochs: int, total_epochs: int):
    """cosine + warmup (附录 Table 2)。"""
    def fn(ep):
        if ep < warmup_epochs:
            return (ep + 1) / warmup_epochs
        t = (ep - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * t))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


@torch.no_grad()
def recall_at_k(logits: torch.Tensor, ks=(1, 5, 10)):
    """batch 内 R@K (text→shape): 论文 Table 3/4 的评测口径 (全库检索时
    把 logits 换成全量相似度矩阵即可, 逻辑相同)。"""
    target = torch.arange(logits.shape[0], device=logits.device)
    ranks = logits.argsort(dim=-1, descending=True)
    out = {}
    for k in ks:
        out[f"R@{k}"] = (ranks[:, :k] == target[:, None]).any(-1).float().mean().item()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=120)      # 附录 Table 2
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-points", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--batches-per-epoch", type=int, default=2,
                        help="占位数据的每 epoch 批数; 真实训练由数据接口决定")
    parser.add_argument("--no-bert", action="store_true",
                        help="无网环境冒烟测试: 跳过 BERT 下载, 文本塔吃现成特征")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Text2ShapeE3CoverNet(embed_dim=args.embed_dim, in_feat_dim=0,
                                 use_bert=not args.no_bert).to(device)

    # 附录 Table 2 (Retrieval): AdamW lr 3e-4, wd 1e-2 —— 只优化可训练参数
    # (BERT 冻结; criterion 的可学习温度 τ 也在 model.parameters() 内)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, args.warmup_epochs, args.epochs)

    for epoch in range(args.epochs):
        model.train()
        for points, input_ids, attention_mask in get_batch_iterator(
                args.batches_per_epoch, args.batch_size,
                args.num_points, 24, device):
            if args.no_bert:
                text_feats = torch.randn(points.shape[0], 768, device=device)
                loss, logits, _, _ = model(points, text_feats=text_feats)
            else:
                loss, logits, _, _ = model(points, input_ids, attention_mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # 等变坐标流的梯度可能偶发尖峰, 裁剪保平稳 (工程惯例)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        scheduler.step()
        metrics = recall_at_k(logits)
        print(f"epoch {epoch:03d} | loss {loss.item():.4f} | "
              f"tau {model.criterion.temperature.item():.4f} | {metrics}")


if __name__ == "__main__":
    main()
