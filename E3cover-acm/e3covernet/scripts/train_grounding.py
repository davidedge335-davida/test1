# -*- coding: utf-8 -*-
"""
3D 视觉定位训练脚本 (极简版) —— 论文统一管线, 附录 Table 2 grounding 列:
    AdamW, lr 1e-4, wd 1e-2, cosine + 5 epoch warmup, 80 epochs,
    batch 32, N=1024, BERT-base 冻结。

按约束不含 Dataset/DataLoader。`get_batch_iterator` 用随机张量占位, 真实
训练时替换为 e3covernet 官方 dataloader (in_out/pt_datasets/listening_dataset.py)
产出的 batch, 字段映射:
    batch['objects']       → objects   [B,K,N,6]
    batch['context_size']  → obj_mask  = arange(K) < context_size[:,None]
    batch['target_pos']    → target_pos
    文本: 用 BertTokenizer 对 batch['utterance'] 编码得到 input_ids/attention_mask
    (基座自带的 tokens 字段是 LSTM 词表下标, 不能直接喂 BERT)
"""

import argparse
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from e3covernet.models.grounding.e3cover_grounding_net import E3CoverGroundingNet  # noqa: E402


def get_batch_iterator(num_batches, batch_size, num_objects, num_points, seq_len, device):
    for _ in range(num_batches):
        objects = torch.randn(batch_size, num_objects, num_points, 6, device=device)
        objects[..., :3] -= objects[..., :3].mean(2, keepdim=True)
        objects[..., :3] /= objects[..., :3].norm(dim=-1).amax(2)[..., None, None]
        ctx = torch.randint(num_objects // 2, num_objects + 1, (batch_size,), device=device)
        obj_mask = torch.arange(num_objects, device=device)[None] < ctx[:, None]
        target_pos = (torch.rand(batch_size, device=device) * ctx).long()
        input_ids = torch.randint(0, 30000, (batch_size, seq_len), device=device)
        attention_mask = torch.ones_like(input_ids)
        yield objects, obj_mask, input_ids, attention_mask, target_pos


def build_scheduler(optimizer, warmup, total):
    def fn(ep):
        if ep < warmup:
            return (ep + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (ep - warmup) / max(1, total - warmup)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-points", type=int, default=1024)
    p.add_argument("--max-objects", type=int, default=52)      # e3covernet 默认 max_context_size
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--fusion-dim", type=int, default=256)
    p.add_argument("--batches-per-epoch", type=int, default=2)
    p.add_argument("--no-bert", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = E3CoverGroundingNet(fusion_dim=args.fusion_dim, in_feat_dim=3,
                                use_bert=not args.no_bert).to(device)
    optimizer = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad],
                                  lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, args.warmup_epochs, args.epochs)

    for epoch in range(args.epochs):
        model.train()
        for objects, obj_mask, ids, am, tgt in get_batch_iterator(
                args.batches_per_epoch, args.batch_size, args.max_objects,
                args.num_points, 24, device):
            if args.no_bert:
                out = model(objects, obj_mask, target_pos=tgt,
                            text_feats=torch.randn(objects.shape[0], 24, 768, device=device),
                            attention_mask=am)
            else:
                out = model(objects, obj_mask, ids, am, target_pos=tgt)
            optimizer.zero_grad(set_to_none=True)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        scheduler.step()
        print(f"epoch {epoch:03d} | loss {out['loss'].item():.4f} | acc {out['acc'].item():.3f}")


if __name__ == "__main__":
    main()
