"""
作者: DU
日期: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Set, Any, Optional, Tuple
import logging

from .metrics import (
    ellipse_to_covariance,
    point_set_ellipse_gw,
    ellipse_gw,
    norm_state_to_ellipse_params,
    weighted_mean_and_cov,
    batched_weighted_mean_and_cov,
    EPS_GW,
    EPS_COV
)

# 常量定义
CLUTTER_ID = -1      # 杂波量测的ID
PADDING_ID = -2      # 填充量测的ID
EPS = 1e-7           # 数值稳定性常数
LOGIT_CLAMP = 15.0   # logits 的 clamp 范围

# 均匀分布协方差修正因子 (椭圆内均匀采样: Var = a²/4)
UNIFORM_COV_SCALE = 0.25


class MGATLoss(nn.Module):
    """
    MGAT多任务损失函数
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化损失函数模块
        """
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

        # ==================== 主损失分量权重 ====================
        self.w_meas = config.get('meas_association_weight', 1.0)
        self.w_birth = config.get('birth_detection_weight', 1.0)
        self.w_death = config.get('death_detection_weight', 1.0)
        self.w_state = config.get('state_prediction_weight', 0.1)

        # ==================== GW辅助损失权重 (新增) ====================
        # λ₁: 已知状态目标GW损失的权重，控制几何约束强度
        self.w_gw_known = config.get('gw_known_weight', 0.1)
        # λ₂: 新生目标GW损失的权重
        self.w_gw_birth = config.get('gw_birth_weight', 0.1)
        # 最小有效权重阈值: 若目标j的总关联权重 Σ_i w_{i,j} < 此阈值，
        # 则认为该目标的加权统计量不可靠，跳过其GW损失计算
        self.gw_min_weight = config.get('gw_min_weight', 0.5)
        # GW计算中的协方差正则化系数 ε_cov
        self.gw_eps = config.get('gw_eps', EPS_COV)
        # 均匀分布协方差修正因子
        self.uniform_cov_scale = config.get('uniform_cov_scale', UNIFORM_COV_SCALE)

        # ==================== 状态GW损失权重 ====================
        self.w_gw_state = config.get('gw_state_weight', 1.0)

        # ==================== 类别权重（处理不平衡）====================
        self.use_class_weight = config.get('use_class_weight', False)
        self.clutter_weight = config.get('clutter_class_weight', 1.0)
        self.birth_weight = config.get('birth_class_weight', 2.0)

        # ==================== Focal Loss参数 ====================
        self.use_focal_loss = config.get('use_focal_loss', False)
        self.focal_gamma = config.get('focal_gamma', 2.0)

        # ==================== 损失 clamp 配置 ====================
        self.max_loss_value = config.get('max_loss_value', 50.0)

        self.logger.info(
            f"MGATLoss v6初始化: "
            f"w_meas={self.w_meas}, w_birth={self.w_birth}, "
            f"w_death={self.w_death}, w_state={self.w_state}, "
            f"w_gw_known={self.w_gw_known}, w_gw_birth={self.w_gw_birth}, "
            f"w_gw_state={self.w_gw_state}"
        )

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        batch_data: Dict[str, Any],
        preprocessor: Any,
        timestep: int = 0
    ) -> Dict[str, torch.Tensor]:
        """
        计算总损失
        """
        device = outputs['association_probs'].device
        is_initial = (timestep == 0)

        ce_loss = self._compute_ce_loss(
            association_probs=outputs['association_probs'],
            meas_ids=batch_data['meas_ids'],
            meas_mask=batch_data['meas_mask'],
            birth_ids=batch_data['birth_ids'],
            target_ids=batch_data['input_target_ids'],
            target_mask=batch_data['input_target_mask'],
            device=device,
            is_initial=is_initial
        )

        gw_known_loss = self._compute_gw_known_loss(
            association_probs=outputs['association_probs'],
            meas_norm=batch_data['meas_norm_for_gw'],
            meas_mask=batch_data['meas_mask'],
            states_norm=batch_data['states_norm_for_gw'],
            target_mask=batch_data['input_target_mask'],
            death_labels=batch_data['death_labels'],
            preprocessor=preprocessor,
            device=device
        )

        gw_birth_loss = self._compute_gw_birth_loss(
            association_probs=outputs['association_probs'],
            meas_norm=batch_data['meas_norm_for_gw'],
            meas_mask=batch_data['meas_mask'],
            device=device
        )

        gw_known_loss = self._safe_loss(gw_known_loss, 'gw_known', device)
        gw_birth_loss = self._safe_loss(gw_birth_loss, 'gw_birth', device)
        ce_loss = self._safe_loss(ce_loss, 'ce', device)

        # L_meas = L_CE + λ₁ · L_GW^(known) + λ₂ · L_GW^(new)
        meas_loss = ce_loss + self.w_gw_known * gw_known_loss + self.w_gw_birth * gw_birth_loss

        B = outputs['association_probs'].shape[0]
        birth_loss = self._compute_birth_detection_loss(
            birth_logit=outputs['birth_logit'],
            birth_ids=[set() for _ in range(B)] if is_initial else batch_data['birth_ids'],
            device=device
        )

        death_loss = self._compute_death_detection_loss(
            death_logit=outputs['death_logit'],
            death_labels=batch_data['death_labels'],
            target_mask=batch_data['input_target_mask'],
            device=device
        )

        state_result = self._compute_state_prediction_loss(
            pred_states=outputs['predicted_states'],
            pred_ids=outputs['target_ids'],
            pred_mask=outputs['target_mask'],
            next_gt=batch_data.get('next_gt', None),
            preprocessor=preprocessor,
            device=device
        )
        state_loss = state_result['total']
        state_gw_loss = state_result['gw']

        meas_loss = self._safe_loss(meas_loss, 'meas', device)
        birth_loss = self._safe_loss(birth_loss, 'birth', device)
        death_loss = self._safe_loss(death_loss, 'death', device)
        state_loss = self._safe_loss(state_loss, 'state', device)
        state_gw_loss = self._safe_loss(state_gw_loss, 'state_gw', device)

        association_loss = self.w_meas * meas_loss + self.w_birth * birth_loss + self.w_death * death_loss
        total_loss = association_loss + self.w_state * state_loss

        total_loss = self._safe_loss(total_loss, 'total', device)

        return {
            'total': total_loss,
            'association': association_loss,
            'meas': meas_loss,
            'ce': ce_loss,
            'gw_known': gw_known_loss,
            'gw_birth': gw_birth_loss,
            'birth': birth_loss,
            'death': death_loss,
            'state': state_loss,
            'gw_state': state_gw_loss
        }

    def _safe_loss(self, loss: torch.Tensor, name: str, device: torch.device) -> torch.Tensor:
        """
        安全损失处理：替换 NaN/Inf，clamp 过大值。
        """
        if not loss.requires_grad:
            return loss

        if torch.isnan(loss).any() or torch.isinf(loss).any():
            self.logger.warning(f"{name}_loss 出现NaN/Inf，零化处理（保持计算图）")
            # loss * 0.0 保持计算图连接，梯度为零但不断开计算图拓扑
            return loss * 0.0

        if loss.item() > self.max_loss_value:
            return loss.clamp(max=self.max_loss_value)

        return loss

    def _prob_to_logit(self, prob: torch.Tensor) -> torch.Tensor:
        """
        将概率转换为 logits（数值稳定版）
        """
        prob_safe = torch.nan_to_num(prob, nan=0.5, posinf=1.0-EPS, neginf=EPS)
        prob_safe = prob_safe.clamp(EPS, 1.0 - EPS)
        logits = torch.log(prob_safe / (1.0 - prob_safe))
        return logits.clamp(-LOGIT_CLAMP, LOGIT_CLAMP)

    def _compute_ce_loss(
        self,
        association_probs: torch.Tensor,
        meas_ids: torch.Tensor,
        meas_mask: torch.Tensor,
        birth_ids: List[Set[int]],
        target_ids: torch.Tensor,
        target_mask: torch.Tensor,
        device: torch.device,
        is_initial: bool = False
    ) -> torch.Tensor:
        """
        多分类交叉熵损失
        L_CE = -(1/N) Σ_i Σ_c y_{i,c} log w_{i,c}
        """
        B, M, num_classes = association_probs.shape
        S = num_classes - 2

        labels = self._build_meas_labels(
            meas_ids, meas_mask, birth_ids, target_ids, target_mask,
            S, device, is_initial
        )

        probs_safe = torch.nan_to_num(association_probs, nan=1.0/num_classes, posinf=1.0-EPS, neginf=EPS)
        probs_safe = probs_safe.clamp(min=EPS, max=1.0 - EPS)
        log_probs = torch.log(probs_safe)

        log_probs_flat = log_probs.view(-1, num_classes)
        labels_flat = labels.view(-1)
        mask_flat = meas_mask.view(-1).float()

        labels_clamped = labels_flat.clamp(0, num_classes - 1)
        nll_per_sample = -torch.gather(log_probs_flat, dim=1, index=labels_clamped.unsqueeze(1)).squeeze(1)

        if self.use_focal_loss:
            probs_flat = probs_safe.view(-1, num_classes)
            pt = torch.gather(probs_flat, dim=1, index=labels_clamped.unsqueeze(1)).squeeze(1)
            focal_weight = (1 - pt).pow(self.focal_gamma)
            nll_per_sample = focal_weight * nll_per_sample

        if self.use_class_weight:
            class_weights = torch.ones(num_classes, device=device)
            class_weights[S] = self.birth_weight
            class_weights[S + 1] = self.clutter_weight
            sample_weights = class_weights[labels_clamped]
            nll_per_sample = nll_per_sample * sample_weights

        valid_count = mask_flat.sum().clamp(min=1.0)
        return (nll_per_sample * mask_flat).sum() / valid_count

    def _build_meas_labels(
        self,
        meas_ids: torch.Tensor,
        meas_mask: torch.Tensor,
        birth_ids: List[Set[int]],
        target_ids: torch.Tensor,
        target_mask: torch.Tensor,
        S: int,
        device: torch.device,
        is_initial: bool = False
    ) -> torch.Tensor:
        """
        构造量测关联的真实标签
        """
        B, M = meas_ids.shape
        labels = torch.full((B, M), S + 1, dtype=torch.long, device=device)  # 默认杂波

        # 新生目标量测（仅当 is_initial=False）
        birth_mask = torch.zeros(B, M, dtype=torch.bool, device=device)
        if not is_initial and birth_ids:
            for b in range(B):
                if b < len(birth_ids) and birth_ids[b]:
                    for bid in birth_ids[b]:
                        birth_mask[b] |= (meas_ids[b] == bid)
            labels[birth_mask & meas_mask] = S

        if S > 0:
            if self.logger.isEnabledFor(logging.DEBUG):
                for b in range(B):
                    valid_ids = target_ids[b][target_mask[b]]
                    if valid_ids.numel() > 0 and valid_ids.unique().shape[0] != valid_ids.shape[0]:
                        self.logger.debug(
                            f"_build_meas_labels: 样本{b}中target_ids存在重复有效ID，"
                            f"请排查preprocessor。unique={valid_ids.unique().tolist()}, "
                            f"all={valid_ids.tolist()}"
                        )

            meas_exp = meas_ids.unsqueeze(-1)
            tgt_exp = target_ids.unsqueeze(1)
            match_matrix = (meas_exp == tgt_exp) & target_mask.unsqueeze(1)
            has_match = match_matrix.any(dim=-1)
            match_idx = match_matrix.float().argmax(dim=-1)

            if is_initial:
                existing_mask = has_match & meas_mask & (meas_ids >= 0)
            else:
                existing_mask = has_match & meas_mask & (meas_ids >= 0) & ~birth_mask

            labels[existing_mask] = match_idx[existing_mask]

        return labels

    def _compute_gw_known_loss(
        self,
        association_probs: torch.Tensor,
        meas_norm: torch.Tensor,
        meas_mask: torch.Tensor,
        states_norm: torch.Tensor,
        target_mask: torch.Tensor,
        death_labels: torch.Tensor,
        preprocessor: Any,
        device: torch.device
    ) -> torch.Tensor:
        """
        已知状态目标的GW辅助损失（在归一化量测坐标空间计算）
        """
        B, M, num_classes = association_probs.shape
        S = num_classes - 2

        if S == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # 形状一致性断言: association_probs中的S维度必须与states_norm的目标数一致
        assert states_norm.shape[1] >= S, (
            f"states_norm目标数({states_norm.shape[1]}) < association中的S({S})，"
            f"请检查模型输入和association_probs的维度是否一致"
        )

        target_probs = association_probs[:, :, :S]                      # [B, M, S]
        w = target_probs * meas_mask.float().unsqueeze(-1)              # [B, M, S]

        mean_j, cov_j, total_w = batched_weighted_mean_and_cov(
            meas_norm, w, eps=self.gw_eps
        )
        # mean_j: [B, S, 2],  cov_j: [B, S, 2, 2],  total_w: [B, S]
        mu_j, Sigma_j = self._states_norm_to_ellipse_in_pos_space(
            states_norm[:, :S], preprocessor, device,
            cov_scale=self.uniform_cov_scale
        )
        # mu_j: [B, S, 2],  Sigma_j: [B, S, 2, 2]
        gw_per_target = point_set_ellipse_gw(
            mean_j, cov_j, mu_j, Sigma_j, eps=self.gw_eps
        )                                                               # [B, S]
        gw_per_target = torch.nan_to_num(gw_per_target, nan=0.0, posinf=0.0, neginf=0.0)
        gw_per_target = gw_per_target.clamp(max=10.0)

        alive_mask = target_mask.float() * (1.0 - death_labels)         # [B, S]
        weight_ok  = (total_w > self.gw_min_weight).float()             # [B, S]
        valid_mask = alive_mask * weight_ok                             # [B, S]

        n_valid = valid_mask.sum().item()
        if n_valid > 0:
            gw_valid_values = gw_per_target[valid_mask > 0]
            self.logger.debug(
                f"GW_known诊断: valid_targets={n_valid:.0f}, "
                f"gw_mean={gw_valid_values.mean().item():.2e}, "
                f"gw_max={gw_valid_values.max().item():.2e}, "
                f"total_w_range=[{total_w[valid_mask > 0].min().item():.1f}, "
                f"{total_w[valid_mask > 0].max().item():.1f}]"
            )
        else:
            self.logger.debug(
                f"GW_known诊断: valid_targets=0, "
                f"alive={alive_mask.sum().item():.0f}, "
                f"weight_ok={weight_ok.sum().item():.0f}"
            )

        numerator   = (gw_per_target * valid_mask).sum()
        denominator = valid_mask.sum().clamp(min=1.0)

        return numerator / denominator

    def _states_norm_to_ellipse_in_pos_space(
        self,
        states_norm: torch.Tensor,
        preprocessor: Any,
        device: torch.device,
        cov_scale: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将归一化5D目标状态转换为归一化量测坐标空间中的椭圆高斯参数。
        """
        return norm_state_to_ellipse_params(
            states_norm,
            angle_scale=preprocessor.angle_scale,
            major_range=preprocessor.major_range,
            major_center=preprocessor.major_center,
            minor_range=preprocessor.minor_range,
            minor_center=preprocessor.minor_center,
            pos_scale=preprocessor.pos_scale,
            cov_scale=cov_scale
        )

    def _compute_gw_birth_loss(
        self,
        association_probs: torch.Tensor,
        meas_norm: torch.Tensor,
        meas_mask: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        """
        新生目标GW辅助损失——随机分裂子集交叉验证法（在归一化坐标空间计算）
        """
        B, M, num_classes = association_probs.shape
        S = num_classes - 2

        w_new_all = association_probs[:, :, S] * meas_mask.float()      # [B, M]
        losses = []

        for b in range(B):
            w_new = w_new_all[b]                                         # [M]
            total_w = w_new.sum()
            # 全局权重不足时跳过
            if total_w < self.gw_min_weight:
                continue

            valid_indices = meas_mask[b].nonzero(as_tuple=True)[0]       # 有效量测索引
            n_valid = len(valid_indices)

            # 每个子集至少需要2个量测点才能计算有意义的协方差
            if n_valid < 4:
                continue

            # 随机排列有效量测索引，等分为两个子集
            perm = torch.randperm(n_valid, device=device)
            half = n_valid // 2
            idx_A = valid_indices[perm[:half]]                           # 子集A索引
            idx_B = valid_indices[perm[half:]]                           # 子集B索引

            z_A = meas_norm[b, idx_A]                                    # [n_A, 2]
            z_B = meas_norm[b, idx_B]                                    # [n_B, 2]
            w_A = w_new[idx_A]                                           # [n_A]
            w_B = w_new[idx_B]                                           # [n_B]

            w_A_sum = w_A.sum()
            w_B_sum = w_B.sum()

            # 每个子集的权重都需要足够（否则统计量不可靠）
            half_min_weight = self.gw_min_weight * 0.3  # 子集阈值适当降低
            if w_A_sum < half_min_weight or w_B_sum < half_min_weight:
                continue
            mu_A, Sigma_A, _ = weighted_mean_and_cov(z_A, w_A, eps=self.gw_eps)
            mu_B, Sigma_B, _ = weighted_mean_and_cov(z_B, w_B, eps=self.gw_eps)

            gw_fwd = point_set_ellipse_gw(
                mu_B, Sigma_B,                                           # 带梯度
                mu_A.detach(), Sigma_A.detach(),                         # 锚点（无梯度）
                eps=self.gw_eps
            )
            gw_rev = point_set_ellipse_gw(
                mu_A, Sigma_A,                                           # 带梯度
                mu_B.detach(), Sigma_B.detach(),                         # 锚点（无梯度）
                eps=self.gw_eps
            )

            gw_sym = (gw_fwd + gw_rev) * 0.5
            if not (torch.isnan(gw_sym) or torch.isinf(gw_sym)):
                losses.append(gw_sym)
        if not losses:
            return torch.tensor(0.0, device=device, requires_grad=True)

        return torch.stack(losses).mean()

    def _compute_birth_detection_loss(
        self,
        birth_logit: torch.Tensor,
        birth_ids: List[Set[int]],
        device: torch.device
    ) -> torch.Tensor:
        """
        计算新生目标检测损失（所有时间步均调用，t=0时标签应为全零以提供负样本监督）
        """
        B = birth_logit.shape[0]

        if torch.isnan(birth_logit).any() or torch.isinf(birth_logit).any():
            self.logger.warning("birth_logit contains NaN/Inf")
            return birth_logit.sum() * 0.0  # 保持计算图连接

        labels = torch.zeros(B, dtype=torch.float32, device=device)
        if birth_ids:
            for b in range(B):
                if b < len(birth_ids) and birth_ids[b] and len(birth_ids[b]) > 0:
                    labels[b] = 1.0

        return F.binary_cross_entropy_with_logits(birth_logit, labels, reduction='mean')

    def _compute_death_detection_loss(
            self,
            death_logit: torch.Tensor,
            death_labels: torch.Tensor,
            target_mask: torch.Tensor,
            device: torch.device
    ) -> torch.Tensor:
        """
        消亡目标检测损失
        """
        valid_count = target_mask.sum()
        if valid_count == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        if torch.isnan(death_logit).any() or torch.isinf(death_logit).any():
            self.logger.warning("death_logit contains NaN/Inf")
            return death_logit.sum() * 0.0  # 保持计算图连接

        logits_flat = death_logit.view(-1)
        labels_flat = death_labels.view(-1).float()
        mask_flat = target_mask.view(-1).float()

        bce = F.binary_cross_entropy_with_logits(logits_flat, labels_flat, reduction='none')
        valid_count_clamped = mask_flat.sum().clamp(min=1.0)

        return (bce * mask_flat).sum() / valid_count_clamped

    def _compute_state_prediction_loss(
        self,
        pred_states: torch.Tensor,
        pred_ids: torch.Tensor,
        pred_mask: torch.Tensor,
        next_gt: Optional[List[torch.Tensor]],
        preprocessor: Any,
        device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """
        计算状态预测损失（纯椭圆GW距离，5D归一化空间）
        """
        zero = torch.tensor(0.0, device=device, requires_grad=True)
        if next_gt is None:
            return {'total': zero, 'gw': zero}

        B, N, _ = pred_states.shape
        pred_list, gt_list = [], []

        for b in range(B):
            if next_gt[b] is None or next_gt[b].numel() == 0:
                continue

            gt_raw = next_gt[b][:, :7].to(device)                               # [N_b, 7] 物理坐标
            gt_ids = next_gt[b][:, 7].long().to(device)

            # GT: 7D物理 → 7D归一化 → 5D提取
            gt_5d = preprocessor.norm_gt_to_5d(gt_raw)                           # [N_b, 5]

            valid_pred_mask = pred_mask[b] & (pred_ids[b] >= 0)
            match = (pred_ids[b].unsqueeze(-1) == gt_ids.unsqueeze(0)) & valid_pred_mask.unsqueeze(-1)
            matched_pred_idx, matched_gt_idx = match.nonzero(as_tuple=True)

            if len(matched_pred_idx) > 0:
                pred_list.append(pred_states[b, matched_pred_idx])               # [K_b, 5]
                gt_list.append(gt_5d[matched_gt_idx])                            # [K_b, 5]

        if not pred_list:
            return {'total': zero, 'gw': zero}

        pred_all = torch.cat(pred_list, dim=0)                                   # [K, 5]
        gt_all = torch.cat(gt_list, dim=0)                                       # [K, 5]

        pred_all = torch.nan_to_num(pred_all, nan=0.0, posinf=0.5, neginf=-0.5)
        gt_all = torch.nan_to_num(gt_all, nan=0.0, posinf=0.5, neginf=-0.5)

        # ==================== 椭圆-椭圆GW距离损失 ====================
        gw_loss = self._compute_state_gw_loss(pred_all, gt_all, preprocessor, device)

        return {
            'total': self.w_gw_state * gw_loss,
            'gw': gw_loss
        }

    def _compute_state_gw_loss(
        self,
        pred_norm: torch.Tensor,
        gt_norm: torch.Tensor,
        preprocessor: Any,
        device: torch.device
    ) -> torch.Tensor:
        """
        计算预测与真实状态之间的椭圆-椭圆GW距离损失（5D归一化空间）。
        """
        # ---- 预测椭圆: 5D归一化 → (μ_pred, Σ_pred) ----
        mu_pred, Sigma_pred = norm_state_to_ellipse_params(
            pred_norm,
            angle_scale=preprocessor.angle_scale,
            major_range=preprocessor.major_range,
            major_center=preprocessor.major_center,
            minor_range=preprocessor.minor_range,
            minor_center=preprocessor.minor_center,
            pos_scale=preprocessor.pos_scale
        )

        mu_gt, Sigma_gt = norm_state_to_ellipse_params(
            gt_norm.detach(),
            angle_scale=preprocessor.angle_scale,
            major_range=preprocessor.major_range,
            major_center=preprocessor.major_center,
            minor_range=preprocessor.minor_range,
            minor_center=preprocessor.minor_center,
            pos_scale=preprocessor.pos_scale
        )

        gw_per_target = ellipse_gw(
            mu_pred, Sigma_pred,
            mu_gt, Sigma_gt,
            eps=self.gw_eps
        )                                                                        # [K]

        gw_per_target = torch.nan_to_num(gw_per_target, nan=0.0, posinf=0.0, neginf=0.0)
        gw_per_target = gw_per_target.clamp(max=10.0)

        return gw_per_target.mean()