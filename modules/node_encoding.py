"""
================================================================================
节点编码器模块 (Node Encoding Module)
================================================================================
作者: DU
日期: 2025
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Any, Tuple

# 激活函数映射表
ACTIVATIONS = {'gelu': nn.GELU, 'silu': nn.SiLU, 'tanh': nn.Tanh}


class BaseNodeEncoder(nn.Module):
    """
    节点编码器基类: input_dim → embed_dim
    """

    def __init__(self, input_dim: int, embed_dim: int, hidden_dim: int, dropout: float = 0.1,
                 activation: str = 'gelu', output_activation: str = 'none', use_residual: bool = True,
                 norm_type: str = 'layer', init_method: str = 'xavier_uniform'):
        """
        初始化编码器
        """
        super().__init__()
        self.use_residual, self.embed_dim = use_residual, embed_dim

        # 主干网络: fc1 → norm1 → act → drop → fc2 → norm2
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.norm1 = nn.LayerNorm(hidden_dim) if norm_type == 'layer' else nn.BatchNorm1d(hidden_dim)
        self.norm2 = nn.LayerNorm(embed_dim) if norm_type == 'layer' else nn.BatchNorm1d(embed_dim)
        self.activation = ACTIVATIONS.get(activation, nn.GELU)()
        self.dropout = nn.Dropout(dropout)

        # 残差投影（仅当维度不匹配时）
        self.residual_proj = nn.Linear(input_dim, embed_dim, bias=False) if use_residual and input_dim != embed_dim else None

        # 输出激活（可选，用于约束输出范围）
        self.output_activation = ACTIVATIONS.get(output_activation, lambda: nn.Identity())() if output_activation != 'none' else nn.Identity()

        # 参数初始化
        self._init_weights(init_method)

    def _init_weights(self, method: str):
        init_fn = {'xavier_uniform': nn.init.xavier_uniform_, 'xavier_normal': nn.init.xavier_normal_,
                   'kaiming_uniform': nn.init.kaiming_uniform_, 'kaiming_normal': nn.init.kaiming_normal_}.get(method,
                                                                                                               nn.init.xavier_uniform_)
        for m in [self.fc1, self.fc2] + ([self.residual_proj] if self.residual_proj else []):
            init_fn(m.weight)
            if hasattr(m, 'bias') and m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, num_nodes, input_dim] → [batch, num_nodes, embed_dim]"""
        h = self.dropout(self.activation(self.norm1(self.fc1(x))))
        h = self.norm2(self.fc2(h))
        if self.use_residual: h = h + (self.residual_proj(x) if self.residual_proj else x)
        return self.output_activation(self.activation(h))


class NodeEncodingModule(nn.Module):
    """
    统一节点编码模块
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        # 全局配置
        embed_dim = config.get('embed_dim', 256)
        norm_type = config.get('norm_type', 'layer')
        init_method = config.get('init_method', 'xavier_uniform')

        # 量测编码器配置
        meas_cfg = {
            'input_dim': config.get('meas_dim', 2),
            'embed_dim': embed_dim,
            'hidden_dim': config.get('meas_hidden_dim', embed_dim // 2),
            'dropout': config.get('meas_dropout', 0.1),
            'activation': config.get('meas_activation', 'gelu'),
            'output_activation': config.get('meas_output_activation', 'none'),
            'use_residual': config.get('meas_use_residual', True),
            'norm_type': norm_type,
            'init_method': init_method
        }

        # 目标编码器配置
        target_cfg = {
            'input_dim': config.get('target_dim', 5),
            'embed_dim': embed_dim,
            'hidden_dim': config.get('target_hidden_dim', embed_dim // 2),
            'dropout': config.get('target_dropout', 0.1),
            'activation': config.get('target_activation', 'gelu'),
            'output_activation': config.get('target_output_activation', 'none'),
            'use_residual': config.get('target_use_residual', True),
            'norm_type': norm_type,
            'init_method': init_method
        }

        # 独立的编码器实例
        self.meas_encoder = BaseNodeEncoder(**meas_cfg)
        self.target_encoder = BaseNodeEncoder(**target_cfg)

        # 保存配置供外部访问
        self.embed_dim = embed_dim
        self.meas_dim = meas_cfg['input_dim']
        self.target_dim = target_cfg['input_dim']

    def forward(self, meas_norm: torch.Tensor, states_norm: torch.Tensor,
                meas_mask: Optional[torch.Tensor] = None, target_mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        并行编码量测和目标节点
        """
        return self.meas_encoder(meas_norm), self.target_encoder(states_norm)