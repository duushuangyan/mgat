"""
多目标数据生成模块 - 管理多个扩展目标及其量测数据
作者: DU | 日期: 2025
"""
import numpy as np
import itertools
from typing import Dict, List, Tuple, Any, Optional, Set
from numpy.random import default_rng
from data.single_target import ExtendedTarget
from data.kinematics_models import generate_random_kinematics_model, initialize_kinematics_state
from utils.config import dotdict


class MultiTargetGenerator:
    """多扩展目标数据生成器：自动模式下随机生成目标并管理生命周期"""

    __slots__ = ('rng', 'batch_idx', 'target_gen_seed', 'delta_t', 'num_average_false_measurements',
                 'field_of_view_lb', 'field_of_view_ub', 'min_targets', 'max_targets', 'min_lifecycle',
                 'position_margin', 'direction_random_offset', 'no_birth_last_steps', 'fov_center_x',
                 'fov_center_y', 'p_birth', 'p_death', 'avg_targets', 'targets_config', 't', 'targets',
                 'trajectories', 'measurements', 'measurements_unique_ids', 'unique_id_counter',
                 'target_lifecycles', 'target_rng', 'birth_tags', 'death_tags', '_total_time_steps')

    def __init__(self, args: Any, rng: np.random.Generator, batch_idx: int):
        """初始化多目标生成器，解析配置参数"""
        self.rng, self.batch_idx = rng, batch_idx                               # 随机数生成器和批次索引
        self.target_gen_seed = rng.integers(0, 2**32)                           # 生成目标专用种子
        self.delta_t = args.simulation.delta_t                                  # 时间步长
        self.num_average_false_measurements = args.measurement.false_alarm_rate # 平均杂波数
        sim = args.simulation                                                   # 仿真配置简写
        self.field_of_view_lb, self.field_of_view_ub = sim.field_of_view_lb, sim.field_of_view_ub  # 视场边界
        self.min_targets = getattr(sim, 'min_targets', 2)                       # 最小目标数
        self.max_targets = getattr(sim, 'max_targets', 10)                      # 最大目标数
        self.min_lifecycle = getattr(sim, 'min_lifecycle', 10)                  # 最小生命周期
        self.position_margin = getattr(sim, 'position_margin', 400)             # 位置边界裕度
        self.direction_random_offset = getattr(sim, 'direction_random_offset', 0.6)  # 方向随机偏移
        self.no_birth_last_steps = getattr(sim, 'no_birth_last_steps', 10)      # 最后N步禁止新生
        self.fov_center_x = self.fov_center_y = (self.field_of_view_lb + self.field_of_view_ub) / 2  # 视场中心
        auto = getattr(args, 'auto_targets', None)                              # 自动目标配置
        self.p_birth = getattr(auto, 'p_birth', 0.1) if auto else 0.1           # 出生概率
        self.p_death = getattr(auto, 'p_death', 0.01) if auto else 0.01         # 消亡概率
        self.avg_targets = getattr(auto, 'avg_targets', 3) if auto else 3       # 平均目标数
        self.targets_config = args                                              # 保存完整配置
        self._total_time_steps = sim.time_steps                                 # 总时间步数缓存
        self.reset()                                                            # 执行初始化重置

    def reset(self):
        """重置生成器状态，初始化目标和轨迹"""
        self.target_rng = default_rng(self.target_gen_seed + self.batch_idx * 1000)  # 确定性RNG
        self.t, self.targets = 0, []                                            # 时间步和目标列表
        self.trajectories, self.measurements, self.measurements_unique_ids = {}, {}, {}  # 数据字典
        self.unique_id_counter = itertools.count(0)                             # ID计数器
        self.target_lifecycles = {}                                             # 生命周期记录
        self.birth_tags, self.death_tags = {}, {}                               # 出生/消亡标签
        num_init = np.clip(self.target_rng.poisson(self.avg_targets), self.min_targets, self.max_targets)  # 初始目标数
        self._ensure_config()                                                   # 确保配置完整
        self.targets = [self._create_target(next(self.unique_id_counter),       # 批量创建初始目标
                        default_rng(self.target_rng.integers(0, 2**32))) for _ in range(num_init)]
        for tgt in self.targets:                                                # 初始化轨迹和生命周期
            self.trajectories[tgt.id] = np.concatenate([tgt.kinematics_state, tgt.shape_state, [0]]).reshape(1, -1)
            self.target_lifecycles[tgt.id] = 0
        self.birth_tags[0] = {tgt.id for tgt in self.targets}                   # t=0所有目标标记为新生
        self.death_tags[0] = set()                                              # t=0无消亡
        self.generate_measurements()                                            # 生成初始量测

    def _ensure_config(self):
        """确保自动模式配置完整"""
        if not hasattr(self.targets_config, 'auto_targets'):                    # 缺少auto_targets
            self.targets_config.auto_targets = dotdict({})                      # 创建空配置
        if not hasattr(self.targets_config.auto_targets, 'initial_shape_state'):  # 缺少形状配置
            self.targets_config.auto_targets.initial_shape_state = dotdict({    # 使用默认值
                'major_axis': dotdict({'min': 80.0, 'max': 100.0}),
                'minor_axis': dotdict({'min': 50.0, 'max': 60.0})})

    def _create_target(self, tid: int, trng: np.random.Generator) -> ExtendedTarget:
        """创建单个随机目标：位置在视场内部，方向朝向中心"""
        margin = self.position_margin                                           # 边界裕度
        pos = trng.uniform([self.field_of_view_lb + margin] * 2,                # 在有效范围内随机位置
                          [self.field_of_view_ub - margin] * 2)
        dx, dy = self.fov_center_x - pos[0], self.fov_center_y - pos[1]         # 指向中心的向量
        base_dir = np.arctan2(dy, dx)                                           # 基础方向角
        direction = np.arctan2(np.sin(base_dir + trng.choice([-1, 1]) * trng.uniform(0, self.direction_random_offset)),
                              np.cos(base_dir + trng.choice([-1, 1]) * trng.uniform(0, self.direction_random_offset)))  # 加偏移并规范化
        kin_cfg = getattr(self.targets_config.auto_targets, 'kinematics_model', None)  # 运动模型配置
        model_type = getattr(kin_cfg, 'model_type', 'CV_only') if kin_cfg else 'CV_only'
        m_type, m_params = generate_random_kinematics_model(model_type, trng, self.targets_config, direction)
        kin_state = initialize_kinematics_state(m_type, m_params, pos)          # 初始化运动状态
        shape_cfg = self.targets_config.auto_targets.initial_shape_state        # 形状配置
        major = trng.uniform(shape_cfg.major_axis.min, shape_cfg.major_axis.max)  # 随机长轴
        minor = trng.uniform(shape_cfg.minor_axis.min, shape_cfg.minor_axis.max)  # 随机短轴
        orientation = np.arctan2(kin_state[3], kin_state[2])                    # 方向角跟随速度方向
        return ExtendedTarget(kin_state, [orientation, major, minor], self.delta_t, tid,
                             self.targets_config, trng, direction)              # 创建目标实例

    def step(self, add_new_objects: bool = True) -> Dict[str, int]:
        """执行一个时间步：更新状态、处理边界和生命周期、生成量测"""
        self.t += self.delta_t                                                  # 时间步进
        t_int = int(self.t)                                                     # 整数时间步
        ids_before = {tgt.id for tgt in self.targets}                           # step前的目标ID集合
        changes = {'removed_outside': 0, 'removed_death': 0, 'added_birth': 0, 'current_count': 0}
        # 更新状态并处理边界
        new_states = [tgt.update_state(t_int) for tgt in self.targets]          # 批量计算新状态
        keep_mask = np.array([(self.field_of_view_lb <= s[0] <= self.field_of_view_ub and  # 视场内判断
                               self.field_of_view_lb <= s[1] <= self.field_of_view_ub) for s in new_states])
        removed_ids = []                                                        # 记录移除的ID
        new_targets = []                                                        # 保留的目标列表
        for i, (tgt, state, keep) in enumerate(zip(self.targets, new_states, keep_mask)):
            if keep:                                                            # 在视场内
                tgt.kinematics_state, tgt.shape_state = state[:4].copy(), state[4:7].copy()  # 更新目标状态
                tgt.state_history = np.vstack([tgt.state_history, state])       # 追加历史
                self.trajectories[tgt.id] = np.vstack([self.trajectories.get(tgt.id, state.reshape(1,-1)), state])
                new_targets.append(tgt)                                         # 保留目标
            else:                                                               # 离开视场
                removed_ids.append(tgt.id)                                      # 记录移除ID
                if tgt.id not in self.trajectories: self.trajectories[tgt.id] = tgt.state_history
                self.target_lifecycles.pop(tgt.id, None)                        # 清理生命周期
        self.targets = new_targets                                              # 更新目标列表
        changes['removed_outside'] = len(removed_ids)                           # 统计越界移除
        # 处理消亡
        n_cur = len(self.targets)                                               # 当前目标数
        if n_cur > self.min_targets:                                            # 允许消亡
            changes['removed_death'] = self._process_deaths(t_int)              # 处理随机消亡
        # 处理出生
        if add_new_objects and self.t < (self._total_time_steps - self.no_birth_last_steps):
            changes['added_birth'] = self._process_births(t_int)                # 处理随机出生
        # 更新生命周期
        for tgt in self.targets:                                                # 存活目标生命周期+1
            self.target_lifecycles[tgt.id] = self.target_lifecycles.get(tgt.id, 0) + 1
        changes['current_count'] = len(self.targets)                            # 当前目标数
        self.t = t_int                                                          # 规范化时间
        ids_after = {tgt.id for tgt in self.targets}                            # step后的目标ID集合
        self.birth_tags[t_int] = ids_after - ids_before                         # 新生目标
        self.death_tags[t_int] = ids_before - ids_after                         # 消亡目标
        self.generate_measurements()                                            # 生成量测
        return changes

    def _process_deaths(self, t_int: int) -> int:
        """向量化处理目标消亡"""
        if not self.targets: return 0                                           # 无目标直接返回
        max_rm = len(self.targets) - self.min_targets                           # 最大可移除数
        if max_rm <= 0: return 0                                                # 不能再移除
        seed = self.target_gen_seed + self.batch_idx * 10000 + t_int * 100      # 确定性种子
        rng = default_rng(seed)                                                 # 创建单个RNG
        # 向量化判断消亡：生命周期>=min_lifecycle且随机数<p_death
        lifecycles = np.array([self.target_lifecycles.get(tgt.id, 0) for tgt in self.targets])
        eligible = lifecycles >= self.min_lifecycle                             # 符合消亡条件的目标
        rand_vals = rng.random(len(self.targets))                               # 批量生成随机数（一次性）
        death_mask = eligible & (rand_vals < self.p_death)                      # 消亡掩码
        n_death = death_mask.sum()                                              # 消亡数量
        if n_death > max_rm:                                                    # 超过最大移除数
            death_idx = np.where(death_mask)[0]                                 # 获取消亡索引
            keep_idx = rng.choice(death_idx, size=n_death - max_rm, replace=False)  # 随机保留部分
            death_mask[keep_idx] = False                                        # 取消消亡标记
        # 执行移除
        new_targets, removed = [], 0                                            # 新目标列表和移除计数
        for i, tgt in enumerate(self.targets):
            if death_mask[i]:                                                   # 需要移除
                if tgt.id not in self.trajectories: self.trajectories[tgt.id] = tgt.state_history
                self.target_lifecycles.pop(tgt.id, None)                        # 清理生命周期
                removed += 1
            else: new_targets.append(tgt)                                       # 保留
        self.targets = new_targets
        return removed

    def _process_births(self, t_int: int) -> int:
        """处理目标出生"""
        if len(self.targets) >= self.max_targets: return 0                      # 已达上限
        seed = self.target_gen_seed + self.batch_idx * 10000 + t_int * 100 + 50000
        rng = default_rng(seed)
        if rng.random() >= self.p_birth: return 0                               # 未触发出生
        tid = next(self.unique_id_counter)                                      # 新目标ID
        trng = default_rng(rng.integers(0, 2**32))                              # 目标专用RNG
        new_tgt = self._create_target(tid, trng)                                # 创建目标
        self.targets.append(new_tgt)                                            # 添加到列表
        init_state = np.concatenate([new_tgt.kinematics_state, new_tgt.shape_state, [t_int]])
        self.trajectories[tid] = init_state.reshape(1, -1)                      # 初始化轨迹
        self.target_lifecycles[tid] = 0                                         # 初始化生命周期
        return 1

    def generate_measurements(self):
        """生成当前时间步的量测数据（真实量测+杂波）"""
        t_int = int(self.t)                                                     # 整数时间步
        seed = self.target_gen_seed + self.batch_idx * 10000 + t_int * 100      # 确定性种子
        rng = default_rng(seed)
        # 生成真实量测
        true_meas = [tgt.update_measurements(t_int) for tgt in self.targets]    # 所有目标量测
        true_meas = np.vstack([m for m in true_meas if len(m) > 0]) if any(len(m) > 0 for m in true_meas) else np.zeros((0, 3))
        # 生成杂波
        n_false = rng.poisson(self.num_average_false_measurements)              # 杂波数量
        if n_false > 0:
            xy = rng.uniform([self.field_of_view_lb]*2, [self.field_of_view_ub]*2, (n_false, 2))  # 向量化生成位置
            valid = (xy[:, 0] >= 0) & (xy[:, 1] >= 0)                           # 有效性过滤
            xy = xy[valid]
            if len(xy) > 0:
                r, a = np.hypot(xy[:, 0], xy[:, 1]), np.arctan2(xy[:, 1], xy[:, 0])  # 转极坐标
                false_meas = np.column_stack([r, a, np.full(len(r), -1)])       # 杂波ID=-1
            else: false_meas = np.zeros((0, 3))
        else: false_meas = np.zeros((0, 3))
        # 合并并打乱
        if len(true_meas) == 0 and len(false_meas) == 0:
            self.measurements[self.t] = np.zeros((0, 2))
            self.measurements_unique_ids[self.t] = np.zeros(0)
            return
        all_meas = np.vstack([m for m in [true_meas, false_meas] if len(m) > 0])
        perm = rng.permutation(len(all_meas))                                   # 随机打乱索引
        all_meas = all_meas[perm]
        self.measurements[self.t] = all_meas[:, :2]                             # 存储[range, angle]
        self.measurements_unique_ids[self.t] = all_meas[:, 2]                   # 存储target_id

    def finish(self):
        """完成仿真，确保所有轨迹记录完整"""
        for tgt in self.targets:
            if tgt.id not in self.trajectories or len(tgt.state_history) > len(self.trajectories[tgt.id]):
                self.trajectories[tgt.id] = tgt.state_history

    def get_state_at_timestep(self, target_id: int, timestep: int) -> Optional[np.ndarray]:
        """获取指定目标在指定时间步的状态[7维]"""
        traj = self.trajectories.get(target_id)
        if traj is None: return None
        mask = traj[:, -1] == timestep                                          # 时间戳匹配
        return traj[mask][0, :-1] if mask.any() else None                       # 返回状态（不含时间戳）

    def __repr__(self) -> str:
        return f'MultiTargetGenerator(batch={self.batch_idx}, t={self.t}, n={len(self.targets) if self.targets else 0})'