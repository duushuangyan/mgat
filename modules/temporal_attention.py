"""
================================================================================
时序注意力模块 (Temporal Attention Module, TAM)
================================================================================
作者: DU
日期: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Any
import logging

from .time_encoding import create_position_encoding


class TemporalAttentionModule(nn.Module):
    """
    时序注意力模块 (TAM)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化TAM模块
        """
        super().__init__()

        self.embed_dim = config.get('embed_dim', 256)
        self.num_heads = config.get('num_heads', 8)
        self.num_layers = config.get('num_layers', 2)
        self.ffn_dim = config.get('ffn_dim', 512)
        self.dropout = config.get('dropout', 0.1)
        self.time_window = config.get('time_window', 10)

        assert self.embed_dim % self.num_heads == 0, \
            f"embed_dim({self.embed_dim})必须能被num_heads({self.num_heads})整除"
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = math.sqrt(self.head_dim)

        # 日志
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"TAM初始化: embed_dim={self.embed_dim}, heads={self.num_heads}, "
                        f"layers={self.num_layers}, window={self.time_window}")

        pe_config = config.get('position_encoding', {})
        pe_config['encoding_len'] = pe_config.get('encoding_len', self.time_window)
        pe_config['embed_dim'] = self.embed_dim
        pe_config['dropout'] = self.dropout
        self.position_encoding = create_position_encoding(pe_config)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ffn_dim,
            dropout=self.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers,
            enable_nested_tensor=False  # 显式禁用以避免警告
        )

        # 最终层归一化
        self.final_norm = nn.LayerNorm(self.embed_dim)

    def _prepare_batch_data(
        self,
        alive_target_ids: List[List[int]],
        history_embeddings: List[Dict[int, Dict[int, torch.Tensor]]],
        current_timestep: int,
        device: torch.device
    ) -> tuple:
        """
        将memory输出的变长字典数据整理成固定形状张量
        """
        B = len(alive_target_ids)
        T = self.time_window
        D = self.embed_dim

        # 计算最大目标数N_max
        N_max = max(len(ids) for ids in alive_target_ids) if alive_target_ids else 1
        N_max = max(N_max, 1)

        # 初始化输出张量
        sequences = torch.zeros(B, N_max, T, D, device=device)
        seq_mask = torch.zeros(B, N_max, T, dtype=torch.bool, device=device)
        target_ids_tensor = torch.full((B, N_max), -1, dtype=torch.long, device=device)
        target_mask = torch.zeros(B, N_max, dtype=torch.bool, device=device)

        # 时间窗口范围
        t_start = max(0, current_timestep - T + 1)
        t_end = current_timestep + 1

        # 遍历每个batch
        for b in range(B):
            batch_alive_ids = alive_target_ids[b]
            batch_history = history_embeddings[b]

            for n, target_id in enumerate(batch_alive_ids):
                target_ids_tensor[b, n] = target_id
                target_mask[b, n] = True

                if target_id not in batch_history:
                    # 这样Transformer不会因为全mask而产生NaN
                    seq_mask[b, n, T - 1] = True
                    # sequences[b, n, T-1] 保持为零向量
                    continue

                emb_dict = batch_history[target_id]
                has_any_valid = False

                for t_abs in range(t_start, t_end):
                    if t_abs in emb_dict:
                        t_rel = T - 1 - (current_timestep - t_abs)
                        if 0 <= t_rel < T:
                            sequences[b, n, t_rel] = emb_dict[t_abs]
                            seq_mask[b, n, t_rel] = True
                            has_any_valid = True

                if not has_any_valid:
                    seq_mask[b, n, T - 1] = True

        return sequences, seq_mask, target_ids_tensor, target_mask

    def forward(
        self,
        alive_target_ids: List[List[int]],
        history_embeddings: List[Dict[int, Dict[int, torch.Tensor]]],
        current_timestep: int,
        device: torch.device,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播：时序注意力聚合
        """
        sequences, seq_mask, target_ids, target_mask = self._prepare_batch_data(
            alive_target_ids, history_embeddings, current_timestep, device
        )

        B, N, T, D = sequences.shape
        sequences = self.position_encoding(sequences)

        # 合并batch和目标维度: [B, N, T, D] → [B*N, T, D]
        sequences_flat = sequences.view(B * N, T, D)
        padding_mask = ~seq_mask.view(B * N, T)

        all_masked = padding_mask.all(dim=1)
        if all_masked.any():
            # 将全mask行的最后一个位置设为False（有效）
            padding_mask[all_masked, -1] = False

        if not return_attention:
            encoded = self.transformer_encoder(sequences_flat, src_key_padding_mask=padding_mask)
        else:
            tam_attn_weights = {}
            x = sequences_flat                          # [B*N, T, D]

            for layer_idx, layer in enumerate(self.transformer_encoder.layers):
                # ----- Self-Attention Block (带注意力权重提取) -----
                x_normed = layer.norm1(x)
                sa_output, attn_w = layer.self_attn(
                    x_normed, x_normed, x_normed,
                    key_padding_mask=padding_mask,
                    need_weights=True,
                    average_attn_weights=False          # 保留多头维度 [B*N, NH, T, T]
                )
                x = x + layer.dropout1(sa_output)

                # ----- Feed-Forward Block (原样调用) -----
                x = x + layer._ff_block(layer.norm2(x))

                # 存储注意力权重 (detach防止参与反向传播)
                tam_attn_weights[layer_idx] = attn_w.detach()

            # TransformerEncoder 末尾的可选 norm
            if self.transformer_encoder.norm is not None:
                x = self.transformer_encoder.norm(x)

            encoded = x

        # NaN检测和处理
        if torch.isnan(encoded).any():
            self.logger.warning("TAM Transformer输出包含NaN，使用零向量替换")
            encoded = torch.nan_to_num(encoded, nan=0.0)

        # 最终层归一化
        encoded = self.final_norm(encoded)

        # 再次NaN检测
        if torch.isnan(encoded).any():
            self.logger.warning("TAM final_norm后仍有NaN，使用零向量替换")
            encoded = torch.nan_to_num(encoded, nan=0.0)

        fused_flat = encoded[:, -1, :]
        fused_embeddings = fused_flat.view(B, N, D)

        fused_embeddings = fused_embeddings * target_mask.unsqueeze(-1).float()

        output = {
            'fused_embeddings': fused_embeddings,
            'target_ids': target_ids,
            'target_mask': target_mask
        }

        if return_attention:
            output['tam_attn_weights'] = tam_attn_weights   # {layer_idx: [B*N, NH, T, T]}

        return output