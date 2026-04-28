"""
================================================================================
时间位置编码模块 (Time Position Encoding)
================================================================================
作者: DU
日期: 2025
"""

import torch
import torch.nn as nn
import math
from typing import Dict, Any


class LearnablePositionEncoding(nn.Module):
    """
    可学习的时间位置编码
    """

    def __init__(self, encoding_len: int, embed_dim: int, dropout: float = 0.1):
        """
        Args:
            max_len: 最大序列长度（时间窗口大小T）
            embed_dim: 嵌入维度D
            dropout: Dropout概率
        """
        super().__init__()
        # 可学习位置嵌入 [max_len, embed_dim]，使用截断正态分布初始化
        self.pe = nn.Parameter(torch.zeros(encoding_len, embed_dim))
        nn.init.trunc_normal_(self.pe, std=0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：将位置编码加到输入张量上
        """
        # x: [B, N, T, D], pe[:T]: [T, D] → 广播到 [1, 1, T, D]
        T = x.shape[2]
        return self.dropout(x + self.pe[:T].unsqueeze(0).unsqueeze(0))


class SinusoidalPositionEncoding(nn.Module):
    """
    固定的正余弦时间位置编码
    """

    def __init__(self, encoding_len: int, embed_dim: int, dropout: float = 0.1):
        """
        Args:
            max_len: 最大序列长度
            embed_dim: 嵌入维度
            dropout: Dropout概率
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # 预计算位置编码并注册为buffer（不参与梯度计算，但会随模型保存/加载）
        # position: [max_len, 1], div_term: [embed_dim//2]
        position = torch.arange(encoding_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))

        # pe: [max_len, embed_dim]
        pe = torch.zeros(encoding_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度用sin
        pe[:, 1::2] = torch.cos(position * div_term[:embed_dim // 2]) if embed_dim % 2 == 0 else torch.cos(position * div_term[:-1])  # 奇数维度用cos

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：将位置编码加到输入张量上
        """
        # x: [B, N, T, D], pe[:T]: [T, D] → 广播到 [1, 1, T, D]
        T = x.shape[2]
        return self.dropout(x + self.pe[:T].unsqueeze(0).unsqueeze(0))


def create_position_encoding(config: Dict[str, Any]) -> nn.Module:
    """
    根据配置创建位置编码模块
    """
    encoding_type = config.get('type', 'learnable')
    encoding_len = config.get('encoding_len', 10)
    embed_dim = config.get('embed_dim', 256)
    dropout = config.get('dropout', 0.1)

    if encoding_type == 'learnable':
        return LearnablePositionEncoding(encoding_len, embed_dim, dropout)
    elif encoding_type == 'sinusoidal':
        return SinusoidalPositionEncoding(encoding_len, embed_dim, dropout)
    else:
        raise ValueError(f"不支持的位置编码类型: {encoding_type}, 可选: 'learnable', 'sinusoidal'")