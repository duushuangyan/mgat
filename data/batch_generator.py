"""
多目标跟踪数据集生成模块 - 并行生成多场景数据
作者: DU | 日期: 2025
"""
import numpy as np
from numpy.random import SeedSequence, default_rng
import concurrent.futures
import os, random
from typing import List, Tuple, Dict, Any, Set
from dataclasses import dataclass
from data.multi_target import MultiTargetGenerator


@dataclass
class SimulationResult:
    """仿真结果数据类"""
    batch_idx: int                                                              # 批次索引
    measurements: Dict[int, np.ndarray]                                         # 量测数据
    ground_truth: Dict[int, np.ndarray]                                         # 标签数据
    unique_measurement_ids: Dict[int, np.ndarray]                               # 量测ID
    trajectories: Dict[int, np.ndarray]                                         # 轨迹数据
    new_rng: np.random.Generator                                                # 更新后的RNG
    birth_tags: Dict[int, Set[int]]                                             # 新生标签
    death_tags: Dict[int, Set[int]]                                             # 消亡标签


class ParallelDataGenerator:
    """并行数据生成器：确保可复现的多场景数据生成"""

    def __init__(self, params: Any):
        """初始化并行数据生成器"""
        self.params = params
        self._set_seeds(params.random_seed)                                     # 设置所有随机种子
        self.device = params.device                                             # 计算设备
        self.n_timesteps = params.simulation.time_steps                         # 时间步数
        self.batch_size = params.dataset.batch_size                             # 批次大小
        # 创建确定性RNG序列
        master = SeedSequence(params.random_seed)                               # 主序列
        self.rngs = [default_rng(master.spawn(1)[0].generate_state(1)[0] + i) for i in range(self.batch_size)]
        # 创建数据生成器
        self.datagens = [MultiTargetGenerator(params, rng, i) for i, rng in enumerate(self.rngs)]
        # 线程池
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.batch_size, thread_name_prefix="DataGen")

    def _set_seeds(self, seed: int):
        """设置所有随机数源种子"""
        random.seed(seed)                                                       # Python内置random
        np.random.seed(seed)                                                    # NumPy全局
        os.environ['PYTHONHASHSEED'] = str(seed)                                # Python哈希种子

    def _process_single(self, args: Tuple[int, MultiTargetGenerator, int]) -> SimulationResult:
        """处理单个仿真任务"""
        idx, gen, n_steps = args
        gen.reset()                                                             # 重置生成器
        for _ in range(n_steps - 1): gen.step()                                 # 执行仿真步骤
        gen.finish()                                                            # 完成仿真
        # 构建ground_truth：向量化处理轨迹数据
        gt = self._build_ground_truth(gen.trajectories, n_steps)
        return SimulationResult(idx, gen.measurements.copy(), gt, gen.measurements_unique_ids.copy(),
                               gen.trajectories.copy(), gen.rng, gen.birth_tags.copy(), gen.death_tags.copy())

    def _build_ground_truth(self, trajectories: Dict[int, np.ndarray], n_steps: int) -> Dict[int, np.ndarray]:
        """向量化构建ground_truth字典"""
        if not trajectories: return {t: np.zeros((0, 8)) for t in range(n_steps)}
        # 将所有轨迹合并并按时间戳分组
        all_data = []                                                           # 收集所有轨迹数据
        for tid, traj in sorted(trajectories.items()):                          # 按ID排序保证确定性
            if traj.size == 0: continue
            # traj: [N, 8] -> 最后一列是时间戳，前7列是状态
            tid_col = np.full((len(traj), 1), tid)                              # 添加目标ID列
            all_data.append(np.hstack([traj[:, :-1], tid_col, traj[:, -1:]]))   # [状态7维, ID, 时间戳]
        if not all_data: return {t: np.zeros((0, 8)) for t in range(n_steps)}
        combined = np.vstack(all_data)                                          # 合并所有数据 [N, 9]
        gt = {}
        for t in range(n_steps):
            mask = combined[:, -1] == t                                         # 该时间步的数据
            gt[t] = combined[mask, :-1].astype(np.float64) if mask.any() else np.zeros((0, 8))  # 去掉时间戳列
        return gt

    def get_batch(self) -> Tuple[List[List[np.ndarray]], List[List[np.ndarray]], List[List[np.ndarray]],
                                 List[Dict[int, np.ndarray]], List[Dict[int, Set[int]]], List[Dict[int, Set[int]]]]:
        """并行生成一个批次的训练数据"""
        args = [(i, gen, self.n_timesteps) for i, gen in enumerate(self.datagens)]
        results = sorted(self.executor.map(self._process_single, args), key=lambda x: x.batch_idx)
        # 更新RNG状态
        for gen, res in zip(self.datagens, results): gen.rng = res.new_rng
        # 整理输出
        measurements, ground_truth, unique_ids, trajectories, birth_tags, death_tags = [], [], [], [], [], []
        for res in results:
            measurements.append([res.measurements.get(t, np.zeros((0, 2))) for t in range(self.n_timesteps)])
            ground_truth.append([res.ground_truth.get(t, np.zeros((0, 8))) for t in range(self.n_timesteps)])
            unique_ids.append([res.unique_measurement_ids.get(t, np.array([])) for t in range(self.n_timesteps)])
            trajectories.append(res.trajectories)
            birth_tags.append(res.birth_tags)
            death_tags.append(res.death_tags)
        return measurements, ground_truth, unique_ids, trajectories, birth_tags, death_tags

    def __del__(self):
        """析构：关闭线程池"""
        if hasattr(self, 'executor'): self.executor.shutdown(wait=True)