"""
================================================================================
MGAT: 多扩展目标跟踪图注意力网络
================================================================================
作者: DU
日期: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import logging

# 导入子模块
from .node_encoding import NodeEncodingModule
from .spatial_attention import SpatialAttentionModule
from .association import AssociationModule
from .memory import MemoryModule
from .temporal_attention import TemporalAttentionModule
from .state_estimation import StateEstimationModule

class MGAT(nn.Module):
    """
    MGAT: 多扩展目标跟踪图注意力网络

    模块组成:
    - NodeEncodingModule: 节点特征编码 (meas: 2D→D, target: 5D→D)
    - SpatialAttentionModule: 异构图注意力 (MM/MS/SM/SS边)
    - AssociationModule: 数据关联 (量测分类、关联、新生/消亡检测)
    - MemoryModule: 记忆模块 (历史嵌入存储与管理)
    - TemporalAttentionModule: 时序注意力 [TODO]
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化MGAT模型
        """
        super().__init__()

        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        self.embed_dim = config.get('embed_dim', 256)

        encoder_config = dict(config.get('encoder', {}))
        encoder_config['embed_dim'] = self.embed_dim  # 确保一致性
        self.node_encoder = NodeEncodingModule(encoder_config)
        self.logger.info(f"NodeEncodingModule初始化完成: meas_dim={self.node_encoder.meas_dim}, "
                        f"target_dim={self.node_encoder.target_dim}, embed_dim={self.embed_dim}")

        sam_config = dict(config.get('sam', {}))
        sam_config['embed_dim'] = self.embed_dim
        self.sam = SpatialAttentionModule(sam_config)
        self.logger.info(f"SpatialAttentionModule初始化完成: num_heads={sam_config.get('num_heads', 8)}, "
                        f"num_layers={sam_config.get('num_layers', 2)}")

        assoc_config = dict(config.get('association', {}))
        assoc_config['embed_dim'] = self.embed_dim
        self.association = AssociationModule(assoc_config)
        self.logger.info(f"AssociationModule初始化完成: hidden_dim={assoc_config.get('hidden_dim', 128)}")

        memory_config = dict(config.get('memory', {}))
        memory_config['embed_dim'] = self.embed_dim  # 确保一致性
        self.memory = MemoryModule(memory_config)
        self.logger.info(f"MemoryModule初始化完成: max_targets={memory_config.get('max_targets', 50)}, "
                        f"max_history={memory_config.get('max_history_length', 100)}")

        tam_config = dict(config.get('tam', {}))
        tam_config['embed_dim'] = self.embed_dim
        self.tam = TemporalAttentionModule(tam_config)
        self.time_window = tam_config.get('time_window', 10)
        self.logger.info(f"TemporalAttentionModule初始化完成: time_window={self.time_window}")

        state_config = dict(config.get('state_estimation', {}))
        state_config['embed_dim'] = self.embed_dim
        state_config['output_dim'] = tam_config.get('output_dim', 5)
        self.state_estimator = StateEstimationModule(state_config)
        self.logger.info(f"StateEstimationModule初始化完成")

        self.current_timestep = 0
        self.is_initialized = False

    def reset(self, batch_size: int):
        """
        重置模型状态（新场景/新epoch开始时调用）
        """
        self.current_timestep = 0
        self.is_initialized = False
        # 重置记忆模块
        self.memory.reset(batch_size)
        self.logger.debug(f"MGAT模型状态已重置, batch_size={batch_size}")

    def forward(
        self,
        meas_norm: torch.Tensor,
        states_norm: torch.Tensor,
        target_ids: torch.Tensor,
        meas_mask: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
        meas_ids: Optional[torch.Tensor] = None,
        is_initial: bool = False,
        return_attention: bool = True,
        update_memory: bool = True
    ) -> Dict[str, Any]:
        """
        完整前向传播
        """
        B, M, _ = meas_norm.shape
        S = states_norm.shape[1]
        device = meas_norm.device

        # ==================== Step1: 节点特征编码 ====================
        # meas_norm: [B, M, 2] → meas_encoded: [B, M, D]
        # states_norm: [B, S, 5] → target_encoded: [B, S, D]
        meas_encoded, target_encoded = self.node_encoder(meas_norm, states_norm, meas_mask, target_mask)

        # ==================== Step2: 空间注意力 ====================
        sam_output = self.sam(meas_encoded=meas_encoded, target_encoded=target_encoded, meas_mask=meas_mask, target_mask=target_mask, return_attention=return_attention)

        meas_embeddings = sam_output['meas_embeddings']      # [B, M, D]
        target_embeddings = sam_output['target_embeddings']  # [B, S, D]

        # 提取最后一层的MS注意力权重用于关联
        attention_ms = None
        if return_attention and 'attention_weights' in sam_output:
            attention_ms = sam_output['attention_weights'][-1]['ms']

        # ==================== Step3: 数据关联 ====================
        # 获取当前 Memory 中的全局最大ID。考虑到 target_ids 输入可能包含 Memory 中尚未记录的 ID (如 Teacher Forcing 的第一帧?)
        current_max_ids = self.memory.get_max_ids(device)
        if target_ids.numel() > 0:
            input_max = target_ids.max(dim=1).values
            current_max_ids = torch.max(current_max_ids, input_max)
        assoc_output = self.association(meas_embeddings=meas_embeddings, target_embeddings=target_embeddings, target_ids=target_ids, attention_ms=attention_ms, meas_mask=meas_mask, target_mask=target_mask, base_target_ids=current_max_ids)

        # ==================== Step4: 记忆更新 ====================
        if update_memory:
            # 更新记忆：存储当前时刻的目标嵌入，memory.current_timestep应在外部通过reset()设置，或在此处同步
            # self.memory.current_timestep = self.current_timestep
            self.memory.update(assoc_output)

        # ==================== Step5: TAM时序聚合 ====================
        # 从记忆中提取历史嵌入供TAM使用
        history_output = self.memory.extract_for_tam(assoc_output)

        # TAM时序注意力聚合
        # 输入: 变长字典结构的历史嵌入
        # 输出: fused_embeddings [B, N_max, D], target_ids [B, N_max], target_mask [B, N_max]
        tam_output = self.tam(
            alive_target_ids=history_output['alive_target_ids'],
            history_embeddings=history_output['history_embeddings'],
            current_timestep=self.current_timestep,
            device=device,
            return_attention=return_attention
        )

        # ==================== Step6: 状态预测 ====================
        # 输入: fused_embeddings [B, N_max, D]
        # 输出: predicted_states [B, N_max, 5]
        state_output = self.state_estimator(
            fused_embeddings=tam_output['fused_embeddings'],
            target_mask=tam_output['target_mask']
        )

        # ==================== 更新时间步 ====================
        if update_memory:
            self.current_timestep += 1
            self.memory.advance_timestep()

        # ==================== 构造输出 ====================
        # 注意: 必须包含 predicted_states, target_ids, target_mask 供 preprocessor 使用
        output = {
            # ========== 供preprocessor.target_node_preprocess使用的必要输出 ==========
            # 使用TAM输出的target_ids/mask，因为它们反映了当前时刻存活的目标
            'predicted_states': state_output['predicted_states'],  # [B, N_max, 5]
            'target_ids': tam_output['target_ids'],  # [B, N_max]
            'target_mask': tam_output['target_mask'],  # [B, N_max]

            # ========== TAM输出 ==========
            'fused_embeddings': tam_output['fused_embeddings'],  # [B, N_max, D]

            # ========== 编码特征 ==========
            'meas_encoded': meas_encoded,  # [B, M, D]
            'target_encoded': target_encoded,  # [B, S, D]

            # ========== SAM输出 ==========
            'meas_embeddings': meas_embeddings,  # [B, M, D]
            'target_embeddings': target_embeddings,  # [B, S, D]

            # ========== 关联结果 ==========
            'association_probs': assoc_output['association_probs'],  # [B, M, S+2]
            'target_probs': assoc_output['target_probs'],  # [B, M, S]
            'clutter_probs': assoc_output['clutter_probs'],  # [B, M]
            'new_target_probs': assoc_output['new_target_probs'],  # [B, M]

            'pred_meas_labels': assoc_output['pred_meas_labels'],  # [B, M]
            'pred_meas_target_ids': assoc_output['pred_meas_target_ids'],  # [B, M]
            'is_clutter': assoc_output['is_clutter'],  # [B, M]
            'is_real_meas': assoc_output['is_real_meas'],  # [B, M]
            'is_new_target_meas': assoc_output['is_new_target_meas'],  # [B, M]

            'has_new_target': assoc_output['has_new_target'],  # [B]
            'new_target_embeddings': assoc_output['new_target_embeddings'],  # [B, 1, D]
            'new_target_ids': assoc_output['new_target_ids'],  # [B, 1]
            'new_target_meas_count': assoc_output['new_target_meas_count'],
            'dead_target_mask': assoc_output['dead_target_mask'],  # [B, S]
            'target_meas_counts': assoc_output['target_meas_counts'],  # [B, S]
            # 关联置信度
            'new_target_confidence': assoc_output['new_target_confidence'],  # [B]
            'death_confidence': assoc_output['death_confidence'],  # [B, S]
            # 关联logit（pre-sigmoid，供损失函数直接使用BCE_with_logits）
            'birth_logit': assoc_output['birth_logit'],  # [B]
            'death_logit': assoc_output['death_logit'],  # [B, S]
            # 更新后的目标张量 (含新生目标)
            'updated_target_embeddings': assoc_output['updated_target_embeddings'],  # [B, S+1, D]
            'updated_target_ids': assoc_output['updated_target_ids'],  # [B, S+1]
            'updated_target_mask': assoc_output['updated_target_mask'],  # [B, S+1]

            # 历史嵌入信息
            'alive_target_ids_list': history_output['alive_target_ids'],  # List[List[int]]
            'history_embeddings_dict': history_output['history_embeddings'],  # List[Dict]

            # 输入掩码和ID (透传)
            'meas_mask': meas_mask,
            'meas_ids': meas_ids,
            'input_target_ids': target_ids,  # 输入的target_ids（区别于输出的）
            'input_target_mask': target_mask,  # 输入的target_mask
        }

        # 可选：注意力权重
        if return_attention and 'attention_weights' in sam_output:
            output['attention_weights'] = sam_output['attention_weights']
        if return_attention and 'tam_attn_weights' in tam_output:
            output['tam_attn_weights'] = tam_output['tam_attn_weights']

        return output

    def dump_memory(self) -> Dict[str, Any]:
        """输出记忆内容（在epoch最后一个时间步调用）"""
        return self.memory.dump()