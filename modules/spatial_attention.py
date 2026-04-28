"""
================================================================================
空间注意力模块 (Spatial Attention Module, SAM)
================================================================================
作者: DU
日期: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List, Any

ACTIVATIONS = {
    'leaky_relu': lambda neg_slope=0.2: nn.LeakyReLU(neg_slope),
    'elu': lambda alpha=1.0: nn.ELU(alpha),
    'gelu': lambda: nn.GELU(),
    'relu': lambda: nn.ReLU(),
    'tanh': lambda: nn.Tanh(),
    'silu': lambda: nn.SiLU(),
}


class HeteroMultiHeadAttention(nn.Module):
    """
    异构多头图注意力机制
    """

    def __init__(self, embed_dim: int, num_heads: int, head_dim: Optional[int] = None,
                 is_cross_type: bool = False, dropout: float = 0.1,
                 attn_activation: str = 'leaky_relu', negative_slope: float = 0.2,
                 use_bias: bool = False, init_method: str = 'xavier_uniform'):
        """
        Args:
            embed_dim: 输入输出嵌入维度
            num_heads: 注意力头数
            head_dim: 每头维度，None则自动计算为embed_dim//num_heads
            is_cross_type: 是否跨类型注意力(MS/SM)，True则src和trg使用独立投影矩阵
            dropout: Dropout概率
            attn_activation: 注意力分数激活函数
            negative_slope: LeakyReLU负斜率
            use_bias: 投影是否使用偏置
            init_method: 参数初始化方法
        """
        super().__init__()

        self.embed_dim, self.num_heads = embed_dim, num_heads
        self.head_dim = head_dim or embed_dim // num_heads
        self.inner_dim = self.num_heads * self.head_dim  # 实际投影维度
        self.scale = self.head_dim ** -0.5  # 可选的缩放因子(本实现不使用)
        self.is_cross_type = is_cross_type

        self.W_src = nn.Linear(embed_dim, self.inner_dim, bias=use_bias)
        self.W_trg = nn.Linear(embed_dim, self.inner_dim, bias=use_bias) if is_cross_type else self.W_src

        self.a_src = nn.Parameter(torch.empty(1, 1, num_heads, self.head_dim))
        self.a_trg = nn.Parameter(torch.empty(1, 1, num_heads, self.head_dim))

        self.W_out = nn.Linear(self.inner_dim, embed_dim, bias=use_bias)

        if attn_activation == 'leaky_relu':
            self.attn_act = nn.LeakyReLU(negative_slope)
        else:
            self.attn_act = ACTIVATIONS.get(attn_activation, lambda: nn.LeakyReLU(0.2))()

        self.feat_dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)

        self._init_weights(init_method)

    def _init_weights(self, method: str):
        """参数初始化"""
        init_fn = getattr(nn.init, method + '_', nn.init.xavier_uniform_)
        for module in [self.W_src, self.W_out] + ([self.W_trg] if self.is_cross_type else []):
            init_fn(module.weight)
            if module.bias is not None: nn.init.zeros_(module.bias)
        init_fn(self.a_src)
        init_fn(self.a_trg)

    def forward(self, src: torch.Tensor, trg: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None, trg_mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        """
        B, Ns, _ = src.shape
        Nt = trg.shape[1]
        NH, Dh = self.num_heads, self.head_dim

        # 应用特征Dropout后投影，reshape为多头格式: [B, N, NH, Dh]
        src_proj = self.W_src(self.feat_dropout(src)).view(B, Ns, NH, Dh)
        trg_proj = self.W_trg(self.feat_dropout(trg)).view(B, Nt, NH, Dh)

        # GAT加性注意力分解: e_ij = a_src·Wh_i + a_trg·Wh_j
        # score_src: [B, Ns, NH] = Σ_d (src_proj * a_src)
        # score_trg: [B, Nt, NH] = Σ_d (trg_proj * a_trg)
        score_src = (src_proj * self.a_src).sum(dim=-1)  # [B, Ns, NH]
        score_trg = (trg_proj * self.a_trg).sum(dim=-1)  # [B, Nt, NH]

        # 广播相加: [B, Ns, NH, 1] + [B, 1, NH, Nt] → [B, Ns, NH, Nt]
        attn_scores = self.attn_act(score_src.unsqueeze(-1) + score_trg.unsqueeze(1).transpose(-1, -2))

        if src_mask is not None and trg_mask is not None:
            # 边掩码: [B, Ns, 1, 1] & [B, 1, 1, Nt] → [B, Ns, 1, Nt] → 广播到 [B, Ns, NH, Nt]
            edge_mask = src_mask.view(B, Ns, 1, 1) & trg_mask.view(B, 1, 1, Nt)
            attn_scores = attn_scores.masked_fill(~edge_mask, float('-inf'))

        # Softmax归一化(沿目标节点维度)，确保每个源节点对所有有效目标的权重和为1
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = attn_weights.nan_to_num(0.0)
        attn_weights = self.attn_dropout(attn_weights)

        # einsum: [B,Ns,NH,Nt] × [B,Nt,NH,Dh] → [B,Ns,NH,Dh]
        aggregated = torch.einsum('bsht,bthd->bshd', attn_weights, trg_proj)
        output = self.W_out(aggregated.reshape(B, Ns, -1))

        return output, attn_weights.permute(0, 2, 1, 3)



class HeteroGATLayer(nn.Module):
    """
    异构图注意力层 - 处理MM/MS/SM/SS四种边类型
    """

    def __init__(self, embed_dim: int, num_heads: int, head_dim: Optional[int] = None,
                 dropout: float = 0.1, attn_activation: str = 'leaky_relu', negative_slope: float = 0.2,
                 fusion_type: str = 'concat', fusion_activation: str = 'gelu',
                 use_residual: bool = True, use_layer_norm: bool = True, layer_norm_eps: float = 1e-6,
                 use_bias: bool = False, init_method: str = 'xavier_uniform'):
        """
        Args:
            embed_dim: 嵌入维度
            num_heads: 注意力头数
            head_dim: 每头维度
            dropout: Dropout概率
            attn_activation: 注意力激活函数
            negative_slope: LeakyReLU负斜率
            fusion_type: 融合方式 ('concat', 'gate', 'sum')
            fusion_activation: 融合层激活函数
            use_residual: 是否使用残差连接
            use_layer_norm: 是否使用LayerNorm
            layer_norm_eps: LayerNorm epsilon
            use_bias: 投影是否使用偏置
            init_method: 参数初始化方法
        """
        super().__init__()
        self.embed_dim, self.use_residual = embed_dim, use_residual

        # 公共参数字典(简化初始化)
        attn_kwargs = dict(embed_dim=embed_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                           attn_activation=attn_activation, negative_slope=negative_slope,
                           use_bias=use_bias, init_method=init_method)

        # 四种边类型的注意力模块
        self.attn_mm = HeteroMultiHeadAttention(**attn_kwargs, is_cross_type=False)  # 量测→量测(共享投影)
        self.attn_ms = HeteroMultiHeadAttention(**attn_kwargs, is_cross_type=True)   # 量测→目标(独立投影)
        self.attn_sm = HeteroMultiHeadAttention(**attn_kwargs, is_cross_type=True)   # 目标→量测(独立投影)
        self.attn_ss = HeteroMultiHeadAttention(**attn_kwargs, is_cross_type=False)  # 目标→目标(共享投影)

        # 特征融合层
        self.fusion_type = fusion_type
        fusion_act = ACTIVATIONS.get(fusion_activation, lambda: nn.GELU())()

        if fusion_type == 'concat':
            # 拼接融合: [2*D] → D
            self.meas_fusion = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.LayerNorm(embed_dim), fusion_act, nn.Dropout(dropout))
            self.target_fusion = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.LayerNorm(embed_dim), fusion_act, nn.Dropout(dropout))
        elif fusion_type == 'gate':
            # 门控融合: gate * agg1 + (1-gate) * agg2
            self.meas_gate = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.Sigmoid())
            self.target_gate = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.Sigmoid())
        # fusion_type == 'sum': 无需额外参数

        # 输出LayerNorm (Pre-LN: LN在残差连接后)
        self.meas_norm = nn.LayerNorm(embed_dim, eps=layer_norm_eps) if use_layer_norm else nn.Identity()
        self.target_norm = nn.LayerNorm(embed_dim, eps=layer_norm_eps) if use_layer_norm else nn.Identity()

        # 融合层参数初始化
        self._init_fusion(init_method)

    def _init_fusion(self, method: str):
        """初始化融合层参数"""
        init_fn = getattr(nn.init, method + '_', nn.init.xavier_uniform_)
        for module in self.modules():
            if isinstance(module, nn.Linear) and module not in [self.attn_mm.W_src, self.attn_mm.W_out,
                                                                 self.attn_ms.W_src, self.attn_ms.W_trg, self.attn_ms.W_out,
                                                                 self.attn_sm.W_src, self.attn_sm.W_trg, self.attn_sm.W_out,
                                                                 self.attn_ss.W_src, self.attn_ss.W_out]:
                init_fn(module.weight)
                if module.bias is not None: nn.init.zeros_(module.bias)

    def _fuse(self, agg1: torch.Tensor, agg2: torch.Tensor, fusion_module, gate_module=None) -> torch.Tensor:
        """特征融合"""
        if self.fusion_type == 'concat':
            return fusion_module(torch.cat([agg1, agg2], dim=-1))
        elif self.fusion_type == 'gate':
            gate = gate_module(torch.cat([agg1, agg2], dim=-1))
            return gate * agg1 + (1 - gate) * agg2
        else:  # sum
            return agg1 + agg2

    def forward(self, meas_feat: torch.Tensor, target_feat: torch.Tensor,
                meas_mask: Optional[torch.Tensor] = None, target_mask: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        """
        # 保存残差
        meas_res, target_res = meas_feat, target_feat

        # ==================== 四种边注意力并行计算 ====================
        agg_mm, attn_mm = self.attn_mm(meas_feat, meas_feat, meas_mask, meas_mask)      # 量测←量测
        agg_ms, attn_ms = self.attn_ms(meas_feat, target_feat, meas_mask, target_mask)  # 量测←目标
        agg_sm, attn_sm = self.attn_sm(target_feat, meas_feat, target_mask, meas_mask)  # 目标←量测
        agg_ss, attn_ss = self.attn_ss(target_feat, target_feat, target_mask, target_mask)  # 目标←目标

        # ==================== 特征融合 ====================
        meas_fused = self._fuse(agg_mm, agg_ms, getattr(self, 'meas_fusion', None), getattr(self, 'meas_gate', None))
        target_fused = self._fuse(agg_ss, agg_sm, getattr(self, 'target_fusion', None), getattr(self, 'target_gate', None))

        # ==================== 残差连接 + LayerNorm ====================
        meas_out = self.meas_norm(meas_fused + meas_res if self.use_residual else meas_fused)
        target_out = self.target_norm(target_fused + target_res if self.use_residual else target_fused)

        return {'meas_feat': meas_out, 'target_feat': target_out,
                'attn_mm': attn_mm, 'attn_ms': attn_ms, 'attn_sm': attn_sm, 'attn_ss': attn_ss}


class SpatialAttentionModule(nn.Module):
    """
    空间注意力模块 (Spatial Attention Module)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化SAM模块
        """
        super().__init__()

        # ==================== 提取配置 ====================
        self.embed_dim = config.get('embed_dim', 256)
        self.num_heads = config.get('num_heads', 8)
        self.num_layers = config.get('num_layers', 2)
        self.head_dim = config.get('head_dim', self.embed_dim // self.num_heads)

        # Dropout (支持细粒度配置)
        base_dropout = config.get('dropout', 0.1)
        self.attn_dropout = config.get('attn_dropout', base_dropout)
        self.feat_dropout = config.get('feat_dropout', base_dropout)

        # 注意力配置
        self.attn_activation = config.get('attn_activation', 'leaky_relu')
        self.negative_slope = config.get('negative_slope', 0.2)

        # 融合配置
        self.fusion_type = config.get('fusion_type', 'concat')
        self.fusion_activation = config.get('fusion_activation', 'gelu')

        # 结构配置
        self.use_residual = config.get('use_residual', True)
        self.use_layer_norm = config.get('use_layer_norm', True)
        self.layer_norm_eps = config.get('layer_norm_eps', 1e-6)
        self.use_bias = config.get('use_bias', False)

        # 初始化方法
        self.init_method = config.get('init_method', 'xavier_uniform')

        layer_kwargs = dict(
            embed_dim=self.embed_dim, num_heads=self.num_heads, head_dim=self.head_dim,
            dropout=base_dropout, attn_activation=self.attn_activation, negative_slope=self.negative_slope,
            fusion_type=self.fusion_type, fusion_activation=self.fusion_activation,
            use_residual=self.use_residual, use_layer_norm=self.use_layer_norm,
            layer_norm_eps=self.layer_norm_eps, use_bias=self.use_bias, init_method=self.init_method
        )
        self.layers = nn.ModuleList([HeteroGATLayer(**layer_kwargs) for _ in range(self.num_layers)])

    def forward(self, meas_encoded: torch.Tensor, target_encoded: torch.Tensor,
                meas_mask: Optional[torch.Tensor] = None, target_mask: Optional[torch.Tensor] = None,
                return_attention: bool = True) -> Dict[str, Any]:
        """
        前向传播
        """
        B, M, D = meas_encoded.shape
        S = target_encoded.shape[1]
        device = meas_encoded.device

        # 默认掩码: 全部有效
        if meas_mask is None: meas_mask = torch.ones(B, M, dtype=torch.bool, device=device)
        if target_mask is None: target_mask = torch.ones(B, S, dtype=torch.bool, device=device)

        # 初始化特征
        meas_feat, target_feat = meas_encoded, target_encoded
        attention_weights = []

        # 逐层计算
        for layer_idx, layer in enumerate(self.layers):
            layer_out = layer(meas_feat, target_feat, meas_mask, target_mask)
            meas_feat, target_feat = layer_out['meas_feat'], layer_out['target_feat']

            if return_attention:
                attention_weights.append({
                    'layer': layer_idx,
                    'mm': layer_out['attn_mm'],  # [B, NH, M, M]
                    'ms': layer_out['attn_ms'],  # [B, NH, M, S]
                    'sm': layer_out['attn_sm'],  # [B, NH, S, M]
                    'ss': layer_out['attn_ss']   # [B, NH, S, S]
                })

        # 构造输出
        output = {'meas_embeddings': meas_feat, 'target_embeddings': target_feat}
        if return_attention: output['attention_weights'] = attention_weights

        return output