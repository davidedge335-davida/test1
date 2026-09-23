# -*- coding: utf-8 -*-
"""
数值验证附录 A.2 (Proposition A.2) 与整体接口:

  1. 单层等变性:  Layer(H, g·Z) = (H', g·Z')   —— 特征不变 + 坐标等变
  2. 骨干不变性:  全局特征 (仅由不变流池化得到) 对 g∈E(3) 不变
     (含反射 det R = -1, 论文强调 E(3) ⊃ SE(3) 包含反射)
  3. 接口形状:    [B, N, 3+C] → [B, D], 与 e3covernet 编码器契约一致
  4. 反传冒烟:    损失可反传, 无 NaN

运行:  python tests/test_equivariance.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from e3covernet.models.backbone.e3covernet import E3CoverNet, GeometryAwareAttention  # noqa: E402
from e3covernet.models.losses.info_nce import SymmetricInfoNCE  # noqa: E402

torch.manual_seed(0)


def random_e3(reflection: bool = False):
    """随机 g=(t,R)∈E(3); reflection=True 时 det R = -1 (O(3)\\SO(3))。"""
    q, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.det(q) < 0:
        q = q * torch.tensor([-1.0, 1.0, 1.0])
    if reflection:
        q = q * torch.tensor([-1.0, 1.0, 1.0])     # 翻一个轴 ⇒ det=-1
    t = torch.randn(3) * 2.0
    return q, t


def test_layer_equivariance():
    layer = GeometryAwareAttention(dim=64, num_heads=4, cover_dim=0, k=8).eval()
    h = torch.randn(2, 32, 64)
    z = torch.randn(2, 32, 3)
    for refl in (False, True):
        R, t = random_e3(refl)
        with torch.no_grad():
            h1, z1 = layer(h, z)
            h2, z2 = layer(h, z @ R.t() + t)
        feat_err = (h1 - h2).abs().max().item()
        coord_err = (z1 @ R.t() + t - z2).abs().max().item()
        tag = "reflection" if refl else "rotation"
        assert feat_err < 1e-4, f"[layer/{tag}] 特征不变性破坏: {feat_err}"
        assert coord_err < 1e-4, f"[layer/{tag}] 坐标等变性破坏: {coord_err}"
        print(f"[PASS] layer equivariance ({tag}): "
              f"feat_err={feat_err:.2e}, coord_err={coord_err:.2e}")


def test_backbone_invariance():
    net = E3CoverNet(in_feat_dim=0, out_dim=128).eval()
    pc = torch.randn(2, 128, 3)
    pc = pc - pc.mean(1, keepdim=True)
    pc = pc / pc.norm(dim=-1).amax(1)[:, None, None]   # 单位球归一化 (附录 B.3)
    for refl in (False, True):
        R, t = random_e3(refl)
        with torch.no_grad():
            f1 = net(pc)
            f2 = net(pc @ R.t() + t)
        err = (f1 - f2).abs().max().item()
        rel = err / f1.abs().max().item()
        tag = "reflection" if refl else "rotation"
        # 注: PCA 参考系的符号规则在一般位置下随 g 协变, 严格不变;
        #     数值误差主要来自 eigh/quantile 的浮点扰动
        assert rel < 1e-3, f"[backbone/{tag}] 全局特征不变性破坏: rel={rel}"
        print(f"[PASS] backbone invariance ({tag}): abs={err:.2e}, rel={rel:.2e}")


def _symmetric_chair(n_side=64):
    """左右/前后均对称的椅子状点云 (偏度≈0, 对称轴上符号消歧不可靠的极端情形)。"""
    torch.manual_seed(1)
    seat = torch.rand(n_side, 3) * torch.tensor([1.0, 1.0, 0.05]) - torch.tensor([0.5, 0.5, 0.0])
    back = torch.rand(n_side, 3) * torch.tensor([1.0, 0.05, 1.0]) - torch.tensor([0.5, 0.5, 0.0])
    legs = torch.cat([torch.rand(n_side // 4, 3) * torch.tensor([0.05, 0.05, 0.5])
                      + torch.tensor([sx, sy, -0.5])
                      for sx in (-0.5, 0.45) for sy in (-0.5, 0.45)])
    pc = torch.cat([seat, back, legs])
    pc = torch.cat([pc, pc * torch.tensor([-1.0, 1.0, 1.0])])   # 强制精确左右镜像对称
    pc = pc - pc.mean(0)
    return (pc / pc.norm(dim=-1).max()).unsqueeze(0)


def test_symmetric_object_invariance():
    """回归测试: 对称形状 (Text2Shape 的椅子/桌子) 上, 参考系轴符号不确定但
    该翻转恰是点云的对称置换, 不变性必须仍然成立。"""
    pc = _symmetric_chair()
    for spd in (True, False):
        net = E3CoverNet(in_feat_dim=0, out_dim=128, embed_mode="invariant",
                         stage_projected_distance=spd).eval()
        with torch.no_grad():
            f1 = net(pc)
            worst = 0.0
            for refl in (False, True):
                for _ in range(3):
                    R, t = random_e3(refl)
                    f2 = net(pc @ R.t() + t)
                    worst = max(worst, ((f1 - f2).abs().max() / f1.abs().max()).item())
        assert worst < 1e-3, f"[symmetric/spd={spd}] 不变性退化: rel={worst}"
        print(f"[PASS] symmetric-object invariance (stage_projected_distance={spd}): rel={worst:.2e}")


def test_raw_embed_mode_runs():
    net = E3CoverNet(in_feat_dim=3, out_dim=64, embed_mode="raw")
    out = net(torch.randn(2, 64, 6))
    assert out.shape == (2, 64)
    print("[PASS] embed_mode='raw' (Algorithm 1 字面实现) runs")


def test_grounding_pipeline():
    from e3covernet.models.grounding.e3cover_grounding_net import E3CoverGroundingNet
    net = E3CoverGroundingNet(fusion_dim=64, in_feat_dim=3, use_bert=False,
                              encoder_kwargs=dict(num_stages=1))
    objects = torch.randn(2, 5, 64, 6)
    obj_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]).bool()
    am = torch.ones(2, 7, dtype=torch.long); am[0, 5:] = 0
    out = net(objects, obj_mask, attention_mask=am, text_feats=torch.randn(2, 7, 768),
              target_pos=torch.tensor([0, 4]))
    assert out["logits"].shape == (2, 5) and torch.isinf(out["logits"][0, 3:]).all()
    out["loss"].backward()
    print(f"[PASS] grounding pipeline (BERT-frozen + cross-attn): loss={out['loss'].item():.3f}")


def test_interface_and_backward():
    # e3covernet 契约: [B*K, N, 3+C] → [B*K, D]  (C=3: rgb)
    net = E3CoverNet(in_feat_dim=3, out_dim=256)
    pc = torch.randn(4, 256, 6)
    feat = net(pc)
    assert feat.shape == (4, 256), f"接口形状不符: {feat.shape}"
    # 检索 loss 反传冒烟
    crit = SymmetricInfoNCE()
    text = torch.randn(4, 256)
    loss, logits = crit(text, feat)
    loss.backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert len(grads) > 0 and all(torch.isfinite(g).all() for g in grads)
    print(f"[PASS] interface [4,256,6]->[4,256]; loss={loss.item():.4f}, "
          f"grad params={len(grads)}, all finite")


def test_ablation_variants():
    # Table 4 消融变体可构造性: no-covering / E(1) only / E(1)⊂E(2)
    for kwargs in (dict(use_covering=False), dict(num_stages=1),
                   dict(num_stages=2)):
        net = E3CoverNet(in_feat_dim=0, out_dim=64, **kwargs).eval()
        out = net(torch.randn(2, 64, 3))
        assert out.shape == (2, 64)
    print("[PASS] ablation variants (Table 4) constructible")


if __name__ == "__main__":
    test_layer_equivariance()
    test_backbone_invariance()
    test_symmetric_object_invariance()
    test_raw_embed_mode_runs()
    test_grounding_pipeline()
    test_interface_and_backward()
    test_ablation_variants()
    print("\nALL TESTS PASSED")
