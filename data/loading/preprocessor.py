"""
雷达数据预处理器 - 归一化、关联矩阵构建、消亡目标注入
作者: DU | 日期: 2025
"""
import torch                                                                    # PyTorch深度学习框架
import numpy as np                                                              # NumPy数值计算库
from typing import List, Tuple, Dict, Any, Optional, Set                        # 类型注解
import logging, math                                                            # 日志和数学库


class RadarDataPreprocessor:
    """雷达数据预处理器：量测/状态归一化、关联矩阵、消亡目标注入"""

    # 7D完整状态 → 5D模型状态的索引映射
    # 7D: [x(0), y(1), vx(2), vy(3), θ(4), a(5), b(6)]
    # 5D: [x(0), y(1), θ(2), a(3), b(4)]
    _7D_TO_5D = [0, 1, 4, 5, 6]

    def __init__(self, config: Dict[str, Any]):
        """初始化预处理器参数"""
        self.config = config
        # 量测归一化参数
        self.norm_method = config.normalization_method                          # 归一化方法
        self.norm_scale = config.normalization_scale                            # 归一化尺度
        self.to_cartesian = config.convert_to_cartesian                         # 是否转笛卡尔
        # 状态归一化参数
        self.pos_scale = config.position_scale                                  # 位置尺度L
        self.vel_scale = 2.0 * config.max_speed                                 # 速度尺度2*v_max
        self.angle_scale = 2.0 * math.pi                                        # 方向角尺度2π
        self.major_center = (config.major_axis_min + config.major_axis_max) / 2  # 长轴中心
        self.major_range = config.major_axis_max - config.major_axis_min        # 长轴范围
        self.minor_center = (config.minor_axis_min + config.minor_axis_max) / 2  # 短轴中心
        self.minor_range = config.minor_axis_max - config.minor_axis_min        # 短轴范围
        # 填充值
        self.tgt_pad_id, self.meas_pad_id = -1, -2                              # 目标和量测填充ID
        # 噪声参数
        noise_cfg = config.target_node_init_noise
        self.noise_enabled = noise_cfg.enabled                                  # 噪声开关
        self.noise_pos, self.noise_vel = noise_cfg.position_std, noise_cfg.velocity_std
        self.noise_angle, self.noise_shape = noise_cfg.angle_std, noise_cfg.shape_std
        # 时间步长
        self.delta_t = getattr(config, 'delta_t', None) or 1.0
        # 日志
        self.logger = logging.getLogger(self.__class__.__name__)

    def meas_preprocess(self, meas: torch.Tensor, gt: List[torch.Tensor], meas_ids: torch.Tensor,
                       mode: str = 'train') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """量测数据预处理：归一化+关联矩阵"""
        B, M, _ = meas.shape
        norm_meas = self._norm_meas(meas)                                       # 归一化量测
        max_tgt = max((g.shape[0] for g in gt), default=1)
        if mode == 'train':
            association_matrix = self._build_assoc_matrix(meas_ids, gt, B, M, max_tgt, meas.device)  # 构建关联矩阵
        else:
            association_matrix = torch.zeros(B, M, max_tgt, dtype=torch.float32, device=meas.device)
        return norm_meas, association_matrix

    def _norm_meas(self, meas: torch.Tensor) -> torch.Tensor:
        """归一化量测数据"""
        if self.to_cartesian:                                                   # 极坐标转笛卡尔
            r, theta = meas[..., 0], meas[..., 1]
            meas = torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=-1)
        if self.norm_method == 'centered_sigmoid':
            return meas / self.norm_scale - 0.5                                 # x/L - 0.5
        elif self.norm_method == 'centered_tanh':
            return 0.5 * torch.tanh(meas / self.norm_scale)
        elif self.norm_method == 'minmax_sigmoid':
            return torch.sigmoid(meas / self.norm_scale)
        return meas

    def meas_denorm(self, norm_meas: torch.Tensor) -> torch.Tensor:
        """量测反归一化"""
        if self.norm_method == 'centered_sigmoid':
            return (norm_meas + 0.5) * self.norm_scale
        elif self.norm_method == 'centered_tanh':
            return torch.atanh(norm_meas / 0.5) * self.norm_scale
        elif self.norm_method == 'minmax_sigmoid':
            return torch.log(norm_meas / (1 - norm_meas + 1e-8)) * self.norm_scale
        return norm_meas

    def _build_assoc_matrix(self, meas_ids: torch.Tensor, gt: List[torch.Tensor],
                           B: int, M: int, max_tgt: int, device) -> torch.Tensor:
        """向量化构建关联矩阵"""
        assoc = torch.zeros(B, M, max_tgt, dtype=torch.float32, device=device)
        for b in range(B):
            if gt[b].shape[0] == 0: continue
            tgt_ids = gt[b][:, 7].long()                                        # 目标ID列表
            unique_ids = tgt_ids.unique()                                       # 唯一ID
            id_map = {tid.item(): i for i, tid in enumerate(unique_ids)}        # ID到索引映射
            valid = (meas_ids[b] != self.meas_pad_id) & (meas_ids[b] >= 0)      # 有效量测掩码
            for m_idx in torch.where(valid)[0]:                                 # 遍历有效量测
                m_tid = meas_ids[b, m_idx].item()
                if m_tid in id_map and id_map[m_tid] < max_tgt:
                    assoc[b, m_idx, id_map[m_tid]] = 1.0
        return assoc

    def _norm_state(self, states: torch.Tensor) -> torch.Tensor:
        """状态向量归一化 [N, 7] -> [N, 7]（内部使用，保留完整7D）"""
        n = states.clone()
        n[:, 0] = states[:, 0] / self.pos_scale - 0.5                           # x归一化
        n[:, 1] = states[:, 1] / self.pos_scale - 0.5                           # y归一化
        n[:, 2] = states[:, 2] / self.vel_scale                                 # vx归一化
        n[:, 3] = states[:, 3] / self.vel_scale                                 # vy归一化
        n[:, 4] = states[:, 4] / self.angle_scale                               # θ归一化
        n[:, 5] = (states[:, 5] - self.major_center) / self.major_range         # 长轴归一化
        n[:, 6] = (states[:, 6] - self.minor_center) / self.minor_range         # 短轴归一化
        return n

    def _to_5d(self, states_7d_norm: torch.Tensor) -> torch.Tensor:
        """从7D归一化状态提取5D模型状态 [..., 7] -> [..., 5]"""
        return states_7d_norm[..., self._7D_TO_5D]

    def norm_gt_to_5d(self, gt_raw_7d: torch.Tensor) -> torch.Tensor:
        """将7D物理GT状态归一化并提取为5D [N, 7] -> [N, 5]"""
        return self._to_5d(self._norm_state(gt_raw_7d))

    def state_denorm(self, norm_states: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """7D状态反归一化（内部使用）"""
        d = norm_states.clone()
        d[..., 0] = (norm_states[..., 0] + 0.5) * self.pos_scale
        d[..., 1] = (norm_states[..., 1] + 0.5) * self.pos_scale
        d[..., 2] = norm_states[..., 2] * self.vel_scale
        d[..., 3] = norm_states[..., 3] * self.vel_scale
        d[..., 4] = norm_states[..., 4] * self.angle_scale
        d[..., 5] = norm_states[..., 5] * self.major_range + self.major_center
        d[..., 6] = norm_states[..., 6] * self.minor_range + self.minor_center
        return d * mask.unsqueeze(-1).float() if mask is not None else d

    def state_denorm_5d(self, norm_5d: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """5D模型状态反归一化 [..., 5] -> [..., 5] (物理坐标)
        5D布局: [x, y, θ, a, b]
        """
        d = norm_5d.clone()
        d[..., 0] = (norm_5d[..., 0] + 0.5) * self.pos_scale                   # x
        d[..., 1] = (norm_5d[..., 1] + 0.5) * self.pos_scale                   # y
        d[..., 2] = norm_5d[..., 2] * self.angle_scale                          # θ
        d[..., 3] = norm_5d[..., 3] * self.major_range + self.major_center      # a
        d[..., 4] = norm_5d[..., 4] * self.minor_range + self.minor_center      # b
        return d * mask.unsqueeze(-1).float() if mask is not None else d

    def compute_velocity(self, pos_current_norm: torch.Tensor,
                         pos_next_norm: torch.Tensor) -> torch.Tensor:
        """从相邻时刻归一化位置计算物理速度 [m/s]
        Args:
            pos_current_norm: [..., 2] 当前时刻归一化位置 (来自模型输入)
            pos_next_norm:    [..., 2] 下一时刻归一化位置 (来自模型预测)
        Returns:
            velocity: [..., 2] 物理速度 [vx, vy] (m/s)
        """
        # 归一化空间位移 → 物理位移 → 除以时间步长
        delta_pos_phys = (pos_next_norm - pos_current_norm) * self.pos_scale
        return delta_pos_phys / self.delta_t

    def _add_noise(self, states_5d: torch.Tensor) -> torch.Tensor:
        """添加初始噪声（5D模型状态: [x, y, θ, a, b]）"""
        n = states_5d.shape[0]
        dev = states_5d.device
        noisy = states_5d.clone()
        noisy[:, 0:2] += torch.randn(n, 2, device=dev) * self.noise_pos         # 位置噪声 [x, y]
        noisy[:, 2] += torch.randn(n, device=dev) * self.noise_angle             # 角度噪声 [θ]
        noisy[:, 3:5] += torch.randn(n, 2, device=dev) * self.noise_shape        # 形状噪声 [a, b]
        return torch.clamp(noisy, -0.5, 0.5)

    def init_target_node_preprocess(self, gt: List[torch.Tensor], add_noise: bool = True
                                   ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """初始时刻(t=0)目标节点预处理，输出5D模型状态"""
        B = len(gt)
        dev = gt[0].device if B > 0 else 'cpu'
        max_tgt = max((g.shape[0] if g.numel() > 0 else 0 for g in gt), default=1)
        states = torch.zeros(B, max_tgt, 5, dtype=torch.float32, device=dev)     # 5D
        ids = torch.full((B, max_tgt), self.tgt_pad_id, dtype=torch.long, device=dev)
        mask = torch.zeros(B, max_tgt, dtype=torch.bool, device=dev)
        for b in range(B):
            if gt[b].numel() == 0: continue
            n = gt[b].shape[0]
            norm_5d = self._to_5d(self._norm_state(gt[b][:, :7]))                # 7D→归一化→5D
            if add_noise and self.noise_enabled: norm_5d = self._add_noise(norm_5d)
            states[b, :n], ids[b, :n], mask[b, :n] = norm_5d, gt[b][:, 7].long(), True
        return states, ids, mask

    def target_node_preprocess_for_training(self, gt: List[torch.Tensor], birth_ids: List[Set[int]],
                                           death_ids: List[Set[int]], death_states: List[List[np.ndarray]],
                                           add_noise: bool = True
                                          ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Teacher Forcing模式：过滤新生目标、注入消亡目标、生成消亡标签
        """
        B = len(gt)
        dev = gt[0].device if B > 0 else 'cpu'
        filtered = []
        for b in range(B):
            g = gt[b]
            if g.numel() == 0: filtered.append(g); continue
            births = birth_ids[b] if birth_ids else set()
            if not births: filtered.append(g); continue
            keep = torch.tensor([int(g[i, 7].item()) not in births for i in range(g.shape[0])], dtype=torch.bool, device=dev)
            filtered.append(g[keep] if keep.any() else torch.zeros((0, 8), dtype=g.dtype, device=dev))
        merged = []
        death_in_gt_indices = []
        injected_death_counts = []
        for b in range(B):
            g = filtered[b]
            ds = death_states[b] if death_states else []
            deaths = death_ids[b] if death_ids else set()

            if not deaths:
                # 无消亡目标
                merged.append(g)
                death_in_gt_indices.append([])
                injected_death_counts.append(0)
                continue

            if g.numel() > 0:
                gt_ids_set = {int(g[i, 7].item()) for i in range(g.shape[0])}
            else:
                gt_ids_set = set()

            in_gt_indices = []  # 在filtered[b]中的索引
            to_inject = []      # 需要额外注入的death_states

            for did in deaths:
                if did in gt_ids_set:
                    for i in range(g.shape[0]):
                        if int(g[i, 7].item()) == did:
                            in_gt_indices.append(i)
                            break
                else:
                    for s in ds:
                        if int(s[7]) == did:
                            to_inject.append(s)
                            break

            if to_inject:
                pred = [np.append(self._cv_predict(s[:7]), s[7]) for s in to_inject]
                pred_t = torch.from_numpy(np.stack(pred)).float().to(dev)
                merged.append(torch.cat([g, pred_t], 0) if g.numel() > 0 else pred_t)
            else:
                merged.append(g)

            death_in_gt_indices.append(in_gt_indices)
            injected_death_counts.append(len(to_inject))

        max_tgt = max((m.shape[0] if m.numel() > 0 else 0 for m in merged), default=1)
        states = torch.zeros(B, max_tgt, 5, dtype=torch.float32, device=dev)     # 5D
        ids = torch.full((B, max_tgt), self.tgt_pad_id, dtype=torch.long, device=dev)
        mask = torch.zeros(B, max_tgt, dtype=torch.bool, device=dev)
        death_labels = torch.zeros(B, max_tgt, dtype=torch.float32, device=dev)
        for b in range(B):
            m = merged[b]
            if m.numel() == 0: continue
            n = m.shape[0]
            norm_5d = self._to_5d(self._norm_state(m[:, :7]))                    # 7D→归一化→5D
            if add_noise and self.noise_enabled: norm_5d = self._add_noise(norm_5d)
            states[b, :n], ids[b, :n], mask[b, :n] = norm_5d, m[:, 7].long(), True
            # 消亡标签——两类消亡目标都需要标记
            # (a) 在gt中已存在的消亡目标：按其在filtered gt中的索引标记
            for idx in death_in_gt_indices[b]:
                death_labels[b, idx] = 1.0
            # (b) 额外注入的消亡目标：在末尾位置标记
            n_injected = injected_death_counts[b]
            if n_injected > 0:
                death_labels[b, n - n_injected:n] = 1.0
        return states, ids, mask, death_labels

    def target_node_preprocess_scheduled_sampling(
        self,
        gt: List[torch.Tensor],
        birth_ids: List[Set[int]],
        death_ids: List[Set[int]],
        death_states: List[List[np.ndarray]],
        prev_outputs: Optional[Dict[str, torch.Tensor]],
        tf_ratio: float,
        add_noise: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Scheduled Sampling 目标节点预处理
        """
        import random as _random
        B = len(gt)
        dev = gt[0].device if B > 0 else 'cpu'

        # ---- 逐样本决定使用GT或模型预测 ----
        use_gt = [True] * B
        if prev_outputs is not None and tf_ratio < 1.0:
            use_gt = [_random.random() < tf_ratio for _ in range(B)]

        # ---- 逐样本处理 ----
        per_sample_states = []   # List of [n_b, 5] tensors (可能带梯度)
        per_sample_ids = []      # List of List[int]
        per_sample_deaths = []   # List of List[float]

        for b in range(B):
            births_b = birth_ids[b] if birth_ids else set()
            deaths_b = death_ids[b] if death_ids else set()
            dstates_b = death_states[b] if death_states else []

            if use_gt[b]:
                s, i, d = self._ss_prepare_gt(gt[b], births_b, deaths_b, dstates_b, add_noise, dev)
            else:
                s, i, d = self._ss_prepare_model(prev_outputs, b, deaths_b, dstates_b, dev)

            per_sample_states.append(s)
            per_sample_ids.append(i)
            per_sample_deaths.append(d)

        # ---- 填充到统一维度并构建批次张量 ----
        max_tgt = max((s.shape[0] for s in per_sample_states), default=0)
        max_tgt = max(max_tgt, 1)

        padded_states = []
        ids = torch.full((B, max_tgt), self.tgt_pad_id, dtype=torch.long, device=dev)
        mask = torch.zeros(B, max_tgt, dtype=torch.bool, device=dev)
        death_labels = torch.zeros(B, max_tgt, dtype=torch.float32, device=dev)

        for b in range(B):
            s = per_sample_states[b]                                             # [n_b, 5]
            n = s.shape[0]
            if n < max_tgt:
                pad = torch.zeros(max_tgt - n, 5, dtype=s.dtype, device=dev)
                s = torch.cat([s, pad], dim=0)                                   # [max_tgt, 5] 梯度安全
            padded_states.append(s)
            if n > 0:
                ids[b, :n] = torch.tensor(per_sample_ids[b], dtype=torch.long, device=dev)
                mask[b, :n] = True
                death_labels[b, :n] = torch.tensor(per_sample_deaths[b], dtype=torch.float32, device=dev)

        states = torch.stack(padded_states, dim=0)                               # [B, max_tgt, 5]
        return states, ids, mask, death_labels

    def _ss_prepare_gt(
        self,
        gt_b: torch.Tensor,
        births_b: Set[int],
        deaths_b: Set[int],
        death_states_b: List[np.ndarray],
        add_noise: bool,
        dev
    ) -> Tuple[torch.Tensor, List[int], List[float]]:
        """
        Scheduled Sampling: 单样本GT路径
        """
        g = gt_b
        # Step 1: 过滤新生目标
        if g.numel() > 0 and births_b:
            keep = torch.tensor(
                [int(g[i, 7].item()) not in births_b for i in range(g.shape[0])],
                dtype=torch.bool, device=dev
            )
            g = g[keep] if keep.any() else torch.zeros((0, 8), dtype=g.dtype, device=dev)

        gt_ids_set = {int(g[i, 7].item()) for i in range(g.shape[0])} if g.numel() > 0 else set()

        states_list, ids_list, deaths_list = [], [], []

        # Step 2: 处理GT中已存在的目标
        if g.numel() > 0:
            norm_5d = self._to_5d(self._norm_state(g[:, :7]))                    # [n, 5]
            if add_noise and self.noise_enabled:
                norm_5d = self._add_noise(norm_5d)
            for i in range(g.shape[0]):
                tid = int(g[i, 7].item())
                states_list.append(norm_5d[i])                                   # 无梯度(来自数据)
                ids_list.append(tid)
                deaths_list.append(1.0 if tid in deaths_b else 0.0)

        # Step 3: 注入不在GT中的消亡目标
        if deaths_b:
            for did in deaths_b:
                if did not in gt_ids_set:
                    for s in death_states_b:
                        if int(s[7]) == did:
                            pred = self._cv_predict(s[:7])
                            pred_t = torch.from_numpy(pred.reshape(1, 7)).float().to(dev)
                            norm_5d_d = self._to_5d(self._norm_state(pred_t))[0]
                            states_list.append(norm_5d_d)
                            ids_list.append(did)
                            deaths_list.append(1.0)
                            break

        if not states_list:
            return torch.zeros(0, 5, device=dev), [], []
        return torch.stack(states_list), ids_list, deaths_list

    def _ss_prepare_model(
        self,
        prev_outputs: Dict[str, torch.Tensor],
        b: int,
        deaths_b: Set[int],
        death_states_b: List[np.ndarray],
        dev
    ) -> Tuple[torch.Tensor, List[int], List[float]]:
        """
        Scheduled Sampling: 单样本模型预测路径
        """
        pred_states = prev_outputs['predicted_states'][b]                        # [N, 5] 有梯度
        pred_ids = prev_outputs['target_ids'][b]                                 # [N]
        pred_mask = prev_outputs['target_mask'][b]                               # [N]

        states_list, ids_list, deaths_list = [], [], []
        pred_ids_set = set()

        # 收集模型预测的有效目标
        for i in range(pred_ids.shape[0]):
            if pred_mask[i].item() and pred_ids[i].item() >= 0:
                tid = pred_ids[i].item()
                pred_ids_set.add(tid)
                states_list.append(pred_states[i])                               # 保留梯度!
                ids_list.append(tid)
                deaths_list.append(1.0 if tid in deaths_b else 0.0)

        # 注入模型未跟踪的消亡目标
        if deaths_b:
            for did in deaths_b:
                if did not in pred_ids_set:
                    for s in death_states_b:
                        if int(s[7]) == did:
                            pred = self._cv_predict(s[:7])
                            pred_t = torch.from_numpy(pred.reshape(1, 7)).float().to(dev)
                            norm_5d_d = self._to_5d(self._norm_state(pred_t))[0]
                            states_list.append(norm_5d_d.detach())               # 无梯度(注入)
                            ids_list.append(did)
                            deaths_list.append(1.0)
                            break

        if not states_list:
            return torch.zeros(0, 5, device=dev), [], []
        return torch.stack(states_list), ids_list, deaths_list

    def target_node_preprocess(self, model_outputs: Dict[str, torch.Tensor]
                              ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """非初始时刻预处理：直接使用模型输出"""
        return model_outputs['predicted_states'], model_outputs['target_ids'], model_outputs['target_mask']

    def _cv_predict(self, state: np.ndarray) -> np.ndarray:
        """CV模型预测：x'=x+vx*dt, y'=y+vy*dt, 速度/形状不变"""
        s = np.asarray(state, dtype=np.float64)
        x, y, vx, vy = s[0], s[1], s[2], s[3]
        theta = s[4] if len(s) >= 5 else (np.arctan2(vy, vx) if vx or vy else 0.0)
        a, b = (s[5], s[6]) if len(s) >= 7 else (110.0, 65.0)
        dt = float(self.delta_t)
        new_theta = np.arctan2(vy, vx) if (vx or vy) else theta
        return np.array([x + vx*dt, y + vy*dt, vx, vy, new_theta, a, b], dtype=np.float64)

    def __call__(self, meas: torch.Tensor, gt: List[torch.Tensor], meas_ids: torch.Tensor,
                mode: str = 'train') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """兼容性调用接口"""
        return self.meas_preprocess(meas, gt, meas_ids, mode)

    def inverse_normalize(self, norm_meas: torch.Tensor) -> torch.Tensor:
        """兼容性反归一化方法"""
        return self.meas_denorm(norm_meas)
