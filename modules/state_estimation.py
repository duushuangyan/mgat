"""
================================================================================
状态估计模块 (State Estimation Module)
================================================================================
作者: DU
日期: 2025
"""

import torch
import torch.nn as nn
from typing import Dict, Any
import logging


class StateEstimationModule(nn.Module):
    """
    状态估计模块
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化状态估计模块
        """
        super().__init__()

        self.embed_dim = config.get('embed_dim', 256)
        self.output_dim = config.get('output_dim', 5)
        self.hidden_dim = config.get('hidden_dim', 128)
        self.num_layers = config.get('num_layers', 2)
        self.dropout = config.get('dropout', 0.1)

        # 日志
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"StateEstimation初始化: embed_dim={self.embed_dim}, "
                         f"output_dim={self.output_dim}, hidden_dim={self.hidden_dim}")

        layers = []
        in_dim = self.embed_dim

        # 隐藏层
        for _ in range(self.num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.GELU(),
                nn.Dropout(self.dropout)
            ])
            in_dim = self.hidden_dim

        layers.append(nn.Linear(in_dim, self.output_dim))
        self.mlp = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        """Xavier初始化权重"""
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
            self,
            fused_embeddings: torch.Tensor,
            target_mask: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播：状态预测
        """
        # MLP预测: [B, N, D] → [B, N, 5]
        predicted_states = self.mlp(fused_embeddings)

        # 应用掩码：无效目标的状态置0
        if target_mask is not None:
            predicted_states = predicted_states * target_mask.unsqueeze(-1).float()

        return {'predicted_states': predicted_states}