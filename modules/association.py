"""
作者: DU
日期: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any
import math
import logging

# 特殊ID标记
CLUTTER_ID = -1  # 杂波量测的目标ID
class AssociationModule(nn.Module):
    '''
    数据关联模块
    '''
    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        # 基本配置
        self.embed_dim = config.get('embed_dim', 256)
        self.hidden_dim = config.get('hidden_dim', 128)
        self.head_hidden_dim = config.get('head_hidden_dim', 64)
        self.dropout = config.get('dropout', 0.1)

        # 关联策略
        self.use_attention_prior = config.get('use_attention_prior', True)

        # 成对特征维度
        self.pair_feature_type = config.get('pair_feature_type', 'full')
        pair_feat_dim = 4 * self.embed_dim if self.pair_feature_type == 'full' else 3 * self.embed_dim

        # 初始化方法
        init_method = config.get('init_method', 'xavier_uniform')

        # 目标关联MLP
        self.score_mlp = nn.Sequential(
            nn.Linear(pair_feat_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, 1)
        )
        # 基于量测特征的杂波判别头
        self.clutter_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.head_hidden_dim),
            nn.LayerNorm(self.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.head_hidden_dim, 1)
        )
        # 基于量测特征的新生目标判别头
        self.new_target_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.head_hidden_dim),
            nn.LayerNorm(self.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.head_hidden_dim, 1)
        )

        # 保留可学习偏置作为全局基准，用于杂波/新生目标分数
        self.clutter_bias = nn.Parameter(torch.zeros(1))
        self.new_target_bias = nn.Parameter(torch.zeros(1))

        # 注意力融合参数
        if self.use_attention_prior:
            # 融合强度（控制注意力调整项的影响程度）
            self.attn_fusion_weight = nn.Parameter(torch.tensor(config.get('attn_fusion_weight_init', 0.0)))
            # 注意力调整方法选择
            self.attn_fusion_method = config.get('attn_fusion_method', 'log_odds')

        # 新生目标嵌入生成器，将聚合的量测嵌入投影到目标嵌入空间
        self.new_target_generator = nn.Sequential(
            nn.Linear(self.embed_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim)
        )

        self.birth_count_threshold = nn.Parameter(torch.tensor(config.get('birth_count_threshold_init', 5.0)))
        self.birth_threshold_sharpness = nn.Parameter(torch.tensor(config.get('birth_threshold_sharpness_init', 1.0)))
        self.birth_target_confidence = config.get('birth_target_confidence', 0.5)
        self.death_count_threshold = nn.Parameter(torch.tensor(config.get('death_count_threshold_init', 0.5)))
        self.death_threshold_sharpness = nn.Parameter(torch.tensor(config.get('death_threshold_sharpness_init', 2.0)))
        self.death_target_confidence = config.get('death_target_confidence', 0.5)
        # 参数初始化
        self._init_weights(init_method)
        self.logger = logging.getLogger(self.__class__.__name__)

    def _init_weights(self, method: str):
        """参数初始化"""
        init_fn = getattr(nn.init, method + '_', nn.init.xavier_uniform_)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init_fn(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def _compute_pair_features(self, meas_emb: torch.Tensor, target_emb: torch.Tensor) -> torch.Tensor:
        """
        计算量测-目标成对特征
        """
        B, M, D = meas_emb.shape
        S = target_emb.shape[1]

        # 扩展维度以便广播: meas[B,M,1,D], target[B,1,S,D] → [B,M,S,D]
        meas_exp = meas_emb.unsqueeze(2).expand(-1, -1, S, -1)
        target_exp = target_emb.unsqueeze(1).expand(-1, M, -1, -1)

        # 构造关联特征
        if self.pair_feature_type == 'full':
            # full: [concat, diff, element-wise product] → 4D
            return torch.cat([meas_exp, target_exp, meas_exp - target_exp, meas_exp * target_exp], dim=-1)
        else:
            # simple: [concat, diff] → 3D
            return torch.cat([meas_exp, target_exp, meas_exp - target_exp], dim=-1)

    def _compute_attention_adjustment(
            self,
            attn_prior: torch.Tensor,  # [B, M, S]
            method: str = 'log_odds'
    ) -> torch.Tensor:
        """
        将注意力概率转换为logits空间的调整项
        """
        S = attn_prior.shape[-1]
        log_attn = torch.log(attn_prior.clamp(min=1e-10))
        if method == 'log_odds':
            uniform_log = math.log(1.0 / S)
            adjustment = log_attn - uniform_log
        elif method == 'standardized':
            # 对每个量测独立标准化其注意力分布
            mu = log_attn.mean(dim=-1, keepdim=True)  # [B, M, 1]
            sigma = log_attn.std(dim=-1, keepdim=True).clamp(min=1e-6)  # [B, M, 1]
            adjustment = (log_attn - mu) / sigma
        else:
            raise ValueError(f"Unknown attention fusion method: {method}")
        return adjustment

    def _compute_soft_counts(
            self,
            association_probs: torch.Tensor,  # [B, M, S+2]
            meas_mask: torch.Tensor,  # [B, M]
            target_mask: torch.Tensor,  # [B, S]
            S: int
    ) -> Dict[str, torch.Tensor]:
        """
        计算软计数（可微）
        """
        B, M, _ = association_probs.shape
        # 将掩码扩展为float用于乘法
        meas_mask_float = meas_mask.float()  # [B, M]

        # ==================== 新生目标软计数 ====================
        new_target_probs = association_probs[:, :, S]  # [B, M]
        soft_new_target_count = (new_target_probs * meas_mask_float).sum(dim=1)  # [B]

        # ==================== 每个目标的软关联计数 ====================
        if S > 0:
            target_probs = association_probs[:, :, :S]  # [B, M, S]
            # 应用量测掩码: [B, M, S] * [B, M, 1] -> [B, M, S]
            target_probs_masked = target_probs * meas_mask_float.unsqueeze(-1)
            soft_target_counts = target_probs_masked.sum(dim=1)  # [B, S]
            soft_target_counts = soft_target_counts * target_mask.float()  # [B, S]
        else:
            soft_target_counts = torch.zeros(B, 0, device=association_probs.device)

        # ==================== 杂波软计数 ====================
        clutter_probs = association_probs[:, :, S + 1]  # [B, M]
        soft_clutter_count = (clutter_probs * meas_mask_float).sum(dim=1)  # [B]

        if torch.isnan(soft_new_target_count).any():
            self.logger.warning("NaN in soft_new_target_count")
            soft_new_target_count = soft_new_target_count.nan_to_num(0.0)

        return {
            'soft_new_target_count': soft_new_target_count,  # [B]
            'soft_target_counts': soft_target_counts,  # [B, S]
            'soft_clutter_count': soft_clutter_count  # [B]
        }

    def _compute_soft_confidences(
            self,
            soft_counts: Dict[str, torch.Tensor],
            target_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        基于软计数计算软置信度（可微）
        """
        # ==================== 新生目标 ====================
        # logit = k * (count - τ)，count > τ 时 logit > 0 → confidence > 0.5
        birth_logit = self.birth_threshold_sharpness * (
            soft_counts['soft_new_target_count'] - self.birth_count_threshold)  # [B]
        soft_birth_confidence = torch.sigmoid(birth_logit)  # [B]

        # ==================== 消亡目标 ====================
        # logit = k * (τ - count)，count < τ 时 logit > 0 → confidence > 0.5
        death_logit = self.death_threshold_sharpness * (
            self.death_count_threshold - soft_counts['soft_target_counts'])  # [B, S]
        soft_death_confidence = torch.sigmoid(death_logit) * target_mask.float()  # [B, S]

        death_logit = death_logit * target_mask.float()  # [B, S]

        return {
            'soft_birth_confidence': soft_birth_confidence,  # [B]
            'soft_death_confidence': soft_death_confidence,  # [B, S]
            'birth_logit': birth_logit,                      # [B]  ← 新增
            'death_logit': death_logit,                      # [B, S] ← 新增
        }

    def _compute_hard_assignments(
            self,
            association_probs: torch.Tensor,  # [B, M, S+2]
            meas_mask: torch.Tensor,  # [B, M]
            target_mask: torch.Tensor,  # [B, S]
            target_ids: torch.Tensor,  # [B, S]
            meas_embeddings: torch.Tensor,  # [B, M, D]
            S: int,
            device: torch.device,
            base_target_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        计算硬分配结果（用于推理和Memory更新）
        """
        B, M, _ = association_probs.shape

        # ==================== 硬分配（不可微，但不需要梯度）====================
        pred_meas_labels = association_probs.argmax(dim=-1)  # [B, M]
        # 需要处理S=0的边界情况
        if S == 0:
            # 没有现有目标，标签只能是 0(新生) 或 1(杂波)
            is_clutter = (pred_meas_labels == 1) & meas_mask
            is_new_target_meas = (pred_meas_labels == 0) & meas_mask
            is_real_meas = meas_mask & ~is_clutter
            is_existing_target_meas = torch.zeros_like(meas_mask)

            pred_meas_target_ids = torch.full((B, M), -1, dtype=torch.long, device=device)
            hard_target_meas_counts = torch.zeros(B, 0, dtype=torch.long, device=device)

            # 消亡掩码为空（无现有目标）
            hard_death_confidence = torch.zeros(B, 0, device=device)
            dead_target_mask = torch.zeros(B, 0, dtype=torch.bool, device=device)
        else:
            # 生成各类量测的掩码
            is_clutter = (pred_meas_labels == S + 1) & meas_mask
            is_new_target_meas = (pred_meas_labels == S) & meas_mask
            is_real_meas = meas_mask & ~is_clutter
            is_existing_target_meas = is_real_meas & ~is_new_target_meas

            # ==================== 目标ID分配 ====================
            pred_meas_target_ids = torch.full((B, M), -1, dtype=torch.long, device=device)
            # 分配现有目标ID
            valid_target_labels = pred_meas_labels.clamp(0, S - 1)
            gathered_ids = torch.gather(target_ids, dim=1, index=valid_target_labels)
            pred_meas_target_ids = torch.where(is_existing_target_meas, gathered_ids, pred_meas_target_ids)

            # ==================== 消亡目标处理 ====================
            labels_clamped = pred_meas_labels.clamp(0, S - 1)
            one_hot = F.one_hot(labels_clamped, num_classes=S).float()
            valid_mask_for_count = is_existing_target_meas.unsqueeze(-1).float()
            one_hot_masked = one_hot * valid_mask_for_count
            hard_target_meas_counts = one_hot_masked.sum(dim=1).long()  # [B, S]

            hard_death_confidence = torch.sigmoid(
                self.death_threshold_sharpness * (self.death_count_threshold - hard_target_meas_counts.float()))
            dead_target_mask = target_mask & (hard_death_confidence > self.death_target_confidence)

        hard_new_target_count = is_new_target_meas.sum(dim=1)  # [B]
        hard_birth_confidence = torch.sigmoid(self.birth_threshold_sharpness * (hard_new_target_count.float() - self.birth_count_threshold))
        has_new_target = hard_birth_confidence > self.birth_target_confidence  # [B]

        no_birth_mask = ~has_new_target  # [B]，标记没有新生目标的batch
        reclassify_to_clutter = is_new_target_meas & no_birth_mask.unsqueeze(1)  # [B, M]
        is_clutter = is_clutter | reclassify_to_clutter           # 将这些量测归入杂波
        is_new_target_meas = is_new_target_meas & (~reclassify_to_clutter)  # 清除无效的新生量测标记
        is_real_meas = meas_mask & ~is_clutter                    # 重新计算真实量测掩码
        is_existing_target_meas = is_real_meas & ~is_new_target_meas  # 重新计算现有目标量测掩码

        # 生成新目标ID
        if base_target_ids is not None:
            new_target_ids = (base_target_ids + 1).long().unsqueeze(1)
        else:
            if S > 0:
                masked_target_ids = target_ids.float().masked_fill(~target_mask, float('-inf'))
                max_existing_ids = masked_target_ids.max(dim=1).values
                max_existing_ids = torch.where(max_existing_ids.isinf(), torch.zeros_like(max_existing_ids), max_existing_ids)
            else:   # S=0的边界情况
                max_existing_ids = torch.zeros(B, device=device)
            new_target_ids = (max_existing_ids + 1).long().unsqueeze(1)

        # 生成新目标嵌入
        new_target_embeddings = torch.zeros(B, 1, self.embed_dim, device=device)
        for b in range(B):
            if has_new_target[b]:
                new_meas_indices = is_new_target_meas[b].nonzero(as_tuple=True)[0]
                if len(new_meas_indices) > 0:
                    new_meas_emb = meas_embeddings[b, new_meas_indices]
                    aggregated_emb = new_meas_emb.mean(dim=0)
                    new_target_embeddings[b, 0] = self.new_target_generator(aggregated_emb)

        # 只有当 has_new_target 为 True 时才更新新生目标量测的ID
        new_target_update_mask = is_new_target_meas & has_new_target.unsqueeze(1)
        pred_meas_target_ids = torch.where(new_target_update_mask, new_target_ids.expand(-1, M), pred_meas_target_ids)

        # 更新后的目标状态ID和掩码，对应的张量new_target_embeddings在子函数外补充更新
        updated_target_ids = torch.cat([target_ids, new_target_ids], dim=1)
        updated_target_mask = torch.cat([target_mask, has_new_target.unsqueeze(1)], dim=1)

        return {
            # 量测分配结果
            'pred_meas_labels': pred_meas_labels,
            'pred_meas_target_ids': pred_meas_target_ids,
            'is_clutter': is_clutter,
            'is_real_meas': is_real_meas,
            'is_new_target_meas': is_new_target_meas,
            'is_existing_target_meas': is_existing_target_meas,

            # 新生目标
            'has_new_target': has_new_target,
            'new_target_embeddings': new_target_embeddings,
            'new_target_ids': new_target_ids,
            'hard_new_target_count': hard_new_target_count,

            # 消亡目标
            'dead_target_mask': dead_target_mask,
            'hard_target_meas_counts': hard_target_meas_counts,

            # 更新后的目标张量
            'updated_target_ids': updated_target_ids,
            'updated_target_mask': updated_target_mask,
        }

    def forward(
            self,
            meas_embeddings: torch.Tensor,
            target_embeddings: torch.Tensor,
            target_ids: torch.Tensor,
            attention_ms: Optional[torch.Tensor] = None,
            meas_mask: Optional[torch.Tensor] = None,
            target_mask: Optional[torch.Tensor] = None,
            base_target_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:

        B, M, D = meas_embeddings.shape
        S = target_embeddings.shape[1]
        device = meas_embeddings.device

        if meas_mask is None: meas_mask = torch.ones(B, M, dtype=torch.bool, device=device)
        if target_mask is None: target_mask = torch.ones(B, S, dtype=torch.bool, device=device)

        # 成对特征: [B, M, S, pair_feat_dim]
        pair_features = self._compute_pair_features(meas_embeddings, target_embeddings)
        # MLP计算分数: [B, M, S, 1] → [B, M, S]
        target_scores = self.score_mlp(pair_features).squeeze(-1)

        if self.use_attention_prior and attention_ms is not None:
            attn_prior = attention_ms.mean(dim=1)  # [B, NH, M, S] → [B, M, S]
            attn_adjustment = self._compute_attention_adjustment(attn_prior, method=self.attn_fusion_method)  # [B, M, S]
            alpha = torch.sigmoid(self.attn_fusion_weight)  # (0, 1)
            target_scores = target_scores + alpha * attn_adjustment
        clutter_scores = self.clutter_bias + self.clutter_head(meas_embeddings)  # [B, M, 1]
        new_target_scores = self.new_target_bias + self.new_target_head(meas_embeddings)  # [B, M, 1]
        # 合并所有分数: [B, M, S+2]
        all_scores = torch.cat([target_scores, new_target_scores, clutter_scores], dim=-1)

        # target_mask_exp: [B, 1, S] → 广播到 [B, M, S]
        target_mask_exp = target_mask.unsqueeze(1)
        full_mask = torch.cat([target_mask_exp.expand(-1, M, -1), torch.ones(B, M, 2, dtype=torch.bool, device=device)], dim=-1)
        all_scores = all_scores.masked_fill(~full_mask, float('-inf'))
        association_probs = F.softmax(all_scores, dim=-1).nan_to_num(0.0)  # [B, M, S+2]

        soft_counts = self._compute_soft_counts(association_probs, meas_mask, target_mask, S)
        soft_confidences = self._compute_soft_confidences(soft_counts, target_mask)

        hard_assignments = self._compute_hard_assignments(association_probs, meas_mask, target_mask, target_ids, meas_embeddings, S, device, base_target_ids)
        # 补充 target_embeddings 到 updated
        hard_assignments['updated_target_embeddings'] = torch.cat([target_embeddings, hard_assignments['new_target_embeddings']], dim=1)
        return {
            # ===== 关联概率 =====
            'association_probs': association_probs,  # [B, M, S+2]
            'target_probs': association_probs[:, :, :S],  # [B, M, S]
            'new_target_probs': association_probs[:, :, S],  # [B, M]
            'clutter_probs': association_probs[:, :, S + 1],  # [B, M]

            # ===== 软计数（可微，用于损失计算）=====
            'soft_new_target_count': soft_counts['soft_new_target_count'],  # [B]
            'soft_target_counts': soft_counts['soft_target_counts'],  # [B, S]
            'soft_clutter_count': soft_counts['soft_clutter_count'],  # [B]

            # ===== 软置信度（可微，用于损失计算）=====
            'soft_birth_confidence': soft_confidences['soft_birth_confidence'],  # [B]
            'soft_death_confidence': soft_confidences['soft_death_confidence'],  # [B, S]

            # ===== 硬分配结果（确定性，用于Memory更新）=====
            'pred_meas_labels': hard_assignments['pred_meas_labels'],  # [B, M]
            'pred_meas_target_ids': hard_assignments['pred_meas_target_ids'],  # [B, M]
            'is_clutter': hard_assignments['is_clutter'],  # [B, M]
            'is_real_meas': hard_assignments['is_real_meas'],  # [B, M]
            'is_new_target_meas': hard_assignments['is_new_target_meas'],  # [B, M]
            'is_existing_target_meas': hard_assignments['is_existing_target_meas'],  # [B, M]

            # ===== 新生目标 =====
            'has_new_target': hard_assignments['has_new_target'],  # [B]
            'new_target_embeddings': hard_assignments['new_target_embeddings'],  # [B, 1, D]
            'new_target_ids': hard_assignments['new_target_ids'],  # [B, 1]
            'new_target_meas_count': hard_assignments['hard_new_target_count'],  # [B] (硬计数，用于调试)

            # ===== 消亡目标 =====
            'dead_target_mask': hard_assignments['dead_target_mask'],  # [B, S]
            'target_meas_counts': hard_assignments['hard_target_meas_counts'],  # [B, S] (硬计数)

            # ===== 用于损失计算的软置信度（命名保持兼容）=====
            'new_target_confidence': soft_confidences['soft_birth_confidence'],  # [B] (软！)
            'death_confidence': soft_confidences['soft_death_confidence'],  # [B, S] (软！)

            # ===== 用于损失计算的pre-sigmoid logit（直接传给BCE_with_logits）=====
            'birth_logit': soft_confidences['birth_logit'],  # [B]
            'death_logit': soft_confidences['death_logit'],  # [B, S]

            # ===== 更新后的目标张量 =====
            'updated_target_embeddings': hard_assignments['updated_target_embeddings'],
            'updated_target_ids': hard_assignments['updated_target_ids'],
            'updated_target_mask': hard_assignments['updated_target_mask'],
        }