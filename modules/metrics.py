"""
作者: DU
日期: 2025
"""

import torch
from typing import Tuple

# ================================================================================
# 点集-椭圆高斯瓦瑟斯坦距离 (Point-Set to Ellipse Gaussian Wasserstein Distance)
# ================================================================================

# ── 数值稳定性常数 ──────────────────────────────────────────────────────
EPS_COV = 1e-8
_SQRT_EPS = 1e-30
EPS_GW = EPS_COV


# 基础工具函数
def rotation_matrix_2d(theta: torch.Tensor) -> torch.Tensor:
    """
    构造2D旋转矩阵。
    """
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    row0 = torch.stack([cos_t, -sin_t], dim=-1)   # [..., 2]
    row1 = torch.stack([sin_t,  cos_t], dim=-1)    # [..., 2]
    R = torch.stack([row0, row1], dim=-2)           # [..., 2, 2]
    return R


def ellipse_to_covariance(
    theta: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor
) -> torch.Tensor:
    """
    将椭圆参数(θ, a, b)转换为对应高斯分布的协方差矩阵。
    """
    R = rotation_matrix_2d(theta)                     # [..., 2, 2]
    # 构造对角矩阵 diag(a², b²)
    D = torch.zeros(*theta.shape, 2, 2, device=theta.device, dtype=theta.dtype)
    D[..., 0, 0] = a ** 2
    D[..., 1, 1] = b ** 2
    # Σ = R D R^T
    Sigma = R @ D @ R.transpose(-2, -1)               # [..., 2, 2]
    return Sigma


def symmetrize(M: torch.Tensor) -> torch.Tensor:
    """
    强制对称化: M ← (M + M^T) / 2
    """
    return (M + M.transpose(-2, -1)) * 0.5


def matrix_sqrt_2x2(M: torch.Tensor, eps: float = EPS_GW) -> torch.Tensor:
    """
    2×2对称正定矩阵的解析平方根
    """
    # det(M) = M00*M11 - M01*M10
    det_M = M[..., 0, 0] * M[..., 1, 1] - M[..., 0, 1] * M[..., 1, 0]
    det_M = det_M.clamp(min=0.0)
    s = torch.sqrt(det_M + _SQRT_EPS)

    # tr(M) = M00 + M11
    tr_M = M[..., 0, 0] + M[..., 1, 1]
    t = torch.sqrt((tr_M + 2.0 * s).clamp(min=_SQRT_EPS))

    # M^{1/2} = (M + s·I) / t
    I = torch.eye(2, device=M.device, dtype=M.dtype)
    # 扩展 I 到与 M 相同的批量维度
    for _ in range(M.dim() - 2):
        I = I.unsqueeze(0)
    I = I.expand_as(M)

    M_sqrt = (M + s[..., None, None] * I) / t[..., None, None]
    return M_sqrt

def point_set_ellipse_gw(
    mean_P: torch.Tensor,
    cov_P: torch.Tensor,
    mu_E: torch.Tensor,
    Sigma_E: torch.Tensor,
    eps: float = EPS_GW
) -> torch.Tensor:
    """
    计算点集高斯近似 N(mean_P, cov_P) 与椭圆高斯分布 N(mu_E, Sigma_E)
    """
    diff = mean_P - mu_E                                         # [..., 2]
    position_term = (diff * diff).sum(dim=-1)                    # [...]

    cov_P_sym = symmetrize(cov_P)
    S_P_half = matrix_sqrt_2x2(cov_P_sym, eps)                  # [..., 2, 2]

    G = S_P_half @ Sigma_E @ S_P_half                            # [..., 2, 2]
    G = symmetrize(G)                                             # 强制对称化

    G_half = matrix_sqrt_2x2(G, eps)                             # [..., 2, 2]

    tr_SP    = cov_P_sym[..., 0, 0] + cov_P_sym[..., 1, 1]
    tr_Sigma = Sigma_E[..., 0, 0]   + Sigma_E[..., 1, 1]
    tr_G_half = G_half[..., 0, 0]   + G_half[..., 1, 1]

    shape_term = tr_SP + tr_Sigma - 2.0 * tr_G_half              # [...]
    shape_term = shape_term.clamp(min=0.0)

    W2_sq = position_term + shape_term

    return W2_sq

def weighted_mean_and_cov(
    points: torch.Tensor,
    weights: torch.Tensor,
    eps: float = EPS_GW
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    计算单个目标的加权均值与加权协方差矩阵
    """
    total_w = weights.sum().clamp(min=eps)

    mean = (weights.unsqueeze(-1) * points).sum(dim=0) / total_w      # [2]

    diff = points - mean.unsqueeze(0)                                  # [N, 2]
    weighted_diff = weights.unsqueeze(-1) * diff                       # [N, 2]
    cov = weighted_diff.t() @ diff / total_w                           # [2, 2]

    I2 = torch.eye(2, device=points.device, dtype=points.dtype)
    cov = cov + eps * I2
    cov = symmetrize(cov)

    return mean, cov, total_w

def batched_weighted_mean_and_cov(
    points: torch.Tensor,
    weights: torch.Tensor,
    eps: float = EPS_GW
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    向量化批量计算: 对 B 个样本中每个样本的 S 个目标，
    """
    # total_w: [B, S]
    total_w = weights.sum(dim=1).clamp(min=eps)

    # 加权均值: [B, S, 2]
    # points: [B, M, 1, 2],  weights: [B, M, S, 1]
    mean = (weights.unsqueeze(-1) * points.unsqueeze(2)).sum(dim=1) / total_w.unsqueeze(-1)

    # 加权协方差: [B, S, 2, 2]
    # diff: [B, M, S, 2] = points[:, :, None, :] - mean[:, None, :, :]
    diff = points.unsqueeze(2) - mean.unsqueeze(1)                     # [B, M, S, 2]
    # 外积: [B, M, S, 2, 1] * [B, M, S, 1, 2] -> [B, M, S, 2, 2]
    outer = diff.unsqueeze(-1) * diff.unsqueeze(-2)                    # [B, M, S, 2, 2]
    # 加权求和: weights: [B, M, S] -> [B, M, S, 1, 1]
    weighted_outer = weights.unsqueeze(-1).unsqueeze(-1) * outer       # [B, M, S, 2, 2]
    cov = weighted_outer.sum(dim=1) / total_w.unsqueeze(-1).unsqueeze(-1)  # [B, S, 2, 2]

    I2 = torch.eye(2, device=points.device, dtype=points.dtype)
    I2 = I2.reshape(1, 1, 2, 2).expand_as(cov)
    cov = cov + eps * I2
    cov = symmetrize(cov)

    return mean, cov, total_w

def ellipse_gw(
    mu_1: torch.Tensor,
    Sigma_1: torch.Tensor,
    mu_2: torch.Tensor,
    Sigma_2: torch.Tensor,
    eps: float = EPS_GW
) -> torch.Tensor:
    """
    计算两个椭圆高斯分布 N(mu_1, Sigma_1) 与 N(mu_2, Sigma_2) 之间的
    平方 2-Wasserstein 距离 W_2^2，与最优传输理论中的标准定义严格一致。
    """
    diff = mu_1 - mu_2                                               # [..., 2]
    position_term = (diff * diff).sum(dim=-1)                        # [...]

    Sigma_1_sym = symmetrize(Sigma_1)
    S1_half = matrix_sqrt_2x2(Sigma_1_sym, eps)                     # [..., 2, 2]

    G = S1_half @ Sigma_2 @ S1_half                                 # [..., 2, 2]
    G = symmetrize(G)                                                # 消除浮点误差

    G_half = matrix_sqrt_2x2(G, eps)                                # [..., 2, 2]

    tr_S1     = Sigma_1_sym[..., 0, 0] + Sigma_1_sym[..., 1, 1]
    tr_S2     = Sigma_2[..., 0, 0]     + Sigma_2[..., 1, 1]
    tr_G_half = G_half[..., 0, 0]      + G_half[..., 1, 1]

    shape_term = (tr_S1 + tr_S2 - 2.0 * tr_G_half).clamp(min=0.0)  # [...]

    W2_sq = (position_term + shape_term).clamp(min=0.0)              # [...]

    return W2_sq


def norm_state_to_ellipse_params(
    states_norm: torch.Tensor,
    angle_scale: float,
    major_range: float,
    major_center: float,
    minor_range: float,
    minor_center: float,
    pos_scale: float,
    min_axis: float = 1e-4,
    cov_scale: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将归一化5D状态向量转换为归一化位置空间中的椭圆高斯参数 (μ, Σ)。
    """
    mu = states_norm[..., :2]                                            # [..., 2]

    theta_raw = states_norm[..., 2] * angle_scale                        # [...]

    a_raw = states_norm[..., 3] * major_range + major_center
    b_raw = states_norm[..., 4] * minor_range + minor_center
    a_pos = (a_raw / pos_scale).clamp(min=min_axis)                      # [...]
    b_pos = (b_raw / pos_scale).clamp(min=min_axis)                      # [...]

    Sigma = ellipse_to_covariance(theta_raw, a_pos, b_pos)               # [..., 2, 2]
    if cov_scale != 1.0:
        Sigma = Sigma * cov_scale

    return mu, Sigma