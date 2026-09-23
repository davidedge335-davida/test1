# -*- coding: utf-8 -*-
"""E3CoverNet: Progressive E(3)-Aware 3D Backbone (论文核心模块包)。

用法 (对齐 e3covernet 基座的 single_object_encoder 接口):
    from e3covernet.models.backbone.e3covernet import E3CoverNet
    encoder = E3CoverNet(in_feat_dim=3, out_dim=args.object_latent_dim)
    feat = encoder(points)   # [B*K, N, 6] -> [B*K, D]
"""

from .backbone import E3CoverNet
from .attention import GeometryAwareAttention
from .blocks import EnBlock, Lift
from .covering import CoveringE1, CoveringE2, CoveringE3, build_covering
from .rbf import RadialBasisEncoding

__all__ = [
    "E3CoverNet", "GeometryAwareAttention", "EnBlock", "Lift",
    "CoveringE1", "CoveringE2", "CoveringE3", "build_covering",
    "RadialBasisEncoding",
]
