"""
雷达数据加载器 - 支持批处理的并行训练
作者: DU | 日期: 2025
"""
import numpy as np
import torch
from torch import Tensor
from typing import List, Tuple, Dict, Optional, Set
import logging, warnings, json
from pathlib import Path
from datetime import datetime
from data.batch_generator import ParallelDataGenerator
from utils.config import load_config


class RadarDataLoader:
    """雷达数据加载器：批次生成、时间步访问、birth/death标签提取"""

    def __init__(self, config_path: str, mode: str = 'train', save_generated_data: bool = False,
                 save_dir: Optional[str] = None, verbose: bool = True):
        """初始化数据加载器"""
        self.verbose, self.mode = verbose, mode
        self.logger = self._setup_logging() if verbose else None                # 设置日志
        if verbose: self.logger.info(f"初始化RadarDataLoader (mode={mode})...")
        self.config_path = config_path
        self.params = load_config(config_path)                                  # 加载配置
        self.batch_size = self.params.dataset.batch_size                        # 批次大小
        self.total_timesteps = self.params.simulation.time_steps                # 总时间步
        self.field_of_view_ub = self.params.simulation.field_of_view_ub         # 视场上界
        self.field_of_view_lb = self.params.simulation.field_of_view_lb         # 视场下界
        device_cfg = getattr(self.params, 'device', 'auto')                     # 设备配置
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu') if device_cfg == 'auto' else \
                     ('cuda' if device_cfg == 'cuda' and torch.cuda.is_available() else 'cpu')
        self.save_generated_data = save_generated_data
        self.save_dir = Path(save_dir) if save_dir else Path(f"radar_data_{mode}_{datetime.now():%Y%m%d_%H%M%S}")
        if save_generated_data: self.save_dir.mkdir(parents=True, exist_ok=True)
        self.current_seed, self.current_epoch = self.params.random_seed, 0      # 种子和epoch
        self.data_generator = None                                              # 数据生成器
        self.current_batch_data, self.current_batch_idx, self.current_timestep = None, 0, 0
        if verbose: self.logger.info("RadarDataLoader初始化完成")

    def _setup_logging(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger('RadarDataLoader')
        logger.setLevel(logging.INFO)
        logger.propagate = False
        if not logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            logger.addHandler(h)
        return logger

    def set_epoch(self, epoch: int):
        """设置epoch，更新随机种子"""
        self.current_epoch = epoch
        self.current_seed = self.params.random_seed + epoch
        self.params.random_seed = self.current_seed
        self.data_generator = ParallelDataGenerator(self.params)                # 重建生成器
        self.current_batch_idx, self.current_timestep, self.current_batch_data = 0, 0, None
        if self.verbose: self.logger.info(f"Epoch {epoch} 开始, 种子: {self.current_seed}")

    def batch_data(self) -> Tuple[Tensor, List, Tensor, List[Dict], List[Dict], List[Dict]]:
        """获取一个新批次的完整数据"""
        raw_meas, raw_gt, raw_ids, trajs, births, deaths = self._generate_batch()
        meas, gt, ids = self._format_batch(raw_meas, raw_gt, raw_ids)           # 格式化数据
        self.current_batch_data = {'measurements': meas, 'ground_truth': gt, 'unique_ids': ids,
                                   'trajectories': trajs, 'birth_tags': births, 'death_tags': deaths}
        self.current_timestep = 0                                               # 重置时间步
        self.current_batch_idx += 1                                             # 批次计数
        # 转换为张量
        meas_t = torch.from_numpy(meas).float()
        ids_t = torch.from_numpy(ids).long()
        gt_t = [[torch.from_numpy(gt[b][t]).float().to(self.device) if self.device != 'cpu'
                else torch.from_numpy(gt[b][t]).float() for t in range(len(gt[b]))] for b in range(len(gt))]
        if self.device != 'cpu': meas_t, ids_t = meas_t.to(self.device), ids_t.to(self.device)
        return meas_t, gt_t, ids_t, trajs, births, deaths

    def _generate_batch(self) -> Tuple:
        """生成原始批次数据"""
        if self.data_generator is None:
            self.params.random_seed = self.current_seed
            self.data_generator = ParallelDataGenerator(self.params)
            if self.verbose: self.logger.info(f"创建数据生成器 (种子: {self.current_seed})")
        raw_meas, raw_gt, raw_ids, trajs, births, deaths = self.data_generator.get_batch()
        if self.save_generated_data: self._save_batch((raw_meas, raw_gt, raw_ids, trajs), self.current_batch_idx)
        return raw_meas, raw_gt, raw_ids, trajs, births, deaths

    def _format_batch(self, raw_meas: List, raw_gt: List, raw_ids: List) -> Tuple[np.ndarray, List, np.ndarray]:
        """格式化批次数据：填充measurements和unique_ids，ground_truth保持变长"""
        B, T = len(raw_meas), self.total_timesteps
        # 计算最大量测数
        max_meas = max((raw_meas[b][t].shape[0] for b in range(B) for t in range(T)
                       if t < len(raw_meas[b]) and len(raw_meas[b][t]) > 0), default=1)
        max_meas = max(max_meas, 1)
        if self.verbose and self.current_batch_idx == 0: self.logger.info(f"批次最大量测数: {max_meas}")
        # 预分配数组
        meas = np.zeros((B, T, max_meas, 2), dtype=np.float32)
        ids = np.full((B, T, max_meas), -2, dtype=np.int32)
        gt = []
        for b in range(B):
            batch_gt = []
            for t in range(T):
                if t < len(raw_meas[b]) and raw_meas[b][t].shape[0] > 0:
                    n = raw_meas[b][t].shape[0]
                    meas[b, t, :n] = raw_meas[b][t]
                if t < len(raw_ids[b]) and len(raw_ids[b][t]) > 0:
                    n = len(raw_ids[b][t])
                    ids[b, t, :n] = raw_ids[b][t]
                batch_gt.append(raw_gt[b][t] if t < len(raw_gt[b]) else np.zeros((0, 8), dtype=np.float32))
            gt.append(batch_gt)
        return meas, gt, ids

    def _extract_timestep(self, t: int, update_counter: bool = True) -> Tuple[Tensor, List[Tensor], Tensor, Tensor, List[Set], List[Set], List[List[np.ndarray]]]:
        """提取指定时间步数据的内部方法（消除代码重复）"""
        if self.current_batch_data is None: raise RuntimeError("请先调用batch_data()")
        data = self.current_batch_data
        meas = data['measurements'][:, t]                                       # [B, max_meas, 2]
        uids = data['unique_ids'][:, t]                                         # [B, max_meas]
        # ground_truth转换
        gt = [torch.from_numpy(data['ground_truth'][b][t]).float().to(self.device) if t < len(data['ground_truth'][b]) and self.device != 'cpu'
             else torch.from_numpy(data['ground_truth'][b][t]).float() if t < len(data['ground_truth'][b])
             else torch.zeros((0, 8), dtype=torch.float32, device=self.device) for b in range(self.batch_size)]
        # birth/death标签
        births = [data['birth_tags'][b].get(t, set()) for b in range(self.batch_size)]
        deaths = [data['death_tags'][b].get(t, set()) for b in range(self.batch_size)]
        # 消亡目标历史状态
        death_states = [self._get_death_states(b, t, deaths[b]) for b in range(self.batch_size)]
        if update_counter: self.current_timestep += 1                           # 更新时间步计数
        # 转换为张量
        meas_t = torch.from_numpy(meas).float()
        uids_t = torch.from_numpy(uids).long()
        if self.device != 'cpu': meas_t, uids_t = meas_t.to(self.device), uids_t.to(self.device)
        return meas_t, gt, uids_t, uids_t != -2, births, deaths, death_states

    def timestep_data(self) -> Tuple[Tensor, List[Tensor], Tensor, Tensor, List[Set], List[Set], List[List[np.ndarray]]]:
        """获取当前时间步数据并移动指针"""
        if self.current_timestep >= self.total_timesteps: raise RuntimeError(f"已处理完所有时间步 ({self.total_timesteps})")
        return self._extract_timestep(self.current_timestep, update_counter=True)

    def next_timestep_data(self) -> Optional[Tuple[Tensor, List[Tensor], Tensor, Tensor, List[Set], List[Set], List[List[np.ndarray]]]]:
        """获取下一时间步数据（不移动指针）"""
        if self.current_timestep >= self.total_timesteps: return None
        return self._extract_timestep(self.current_timestep, update_counter=False)

    def get_timestep_data(self, t: int) -> Optional[Tuple[Tensor, List[Tensor], Tensor, Tensor]]:
        """获取指定时间步数据（随机访问）"""
        if self.current_batch_data is None: raise RuntimeError("请先调用batch_data()")
        if t < 0 or t >= self.total_timesteps: return None
        data = self.current_batch_data
        meas = torch.from_numpy(data['measurements'][:, t]).float()
        uids = torch.from_numpy(data['unique_ids'][:, t]).long()
        gt = [torch.from_numpy(data['ground_truth'][b][t]).float() if t < len(data['ground_truth'][b])
             else torch.zeros((0, 8), dtype=torch.float32) for b in range(self.batch_size)]
        if self.device != 'cpu':
            meas, uids = meas.to(self.device), uids.to(self.device)
            gt = [g.to(self.device) for g in gt]
        return meas, gt, uids, uids != -2

    def _get_death_states(self, batch_idx: int, t: int, death_ids: Set[int]) -> List[np.ndarray]:
        """获取消亡目标在t-1时刻的状态"""
        if not death_ids or t <= 0 or self.current_batch_data is None: return []
        trajs = self.current_batch_data.get('trajectories', [])
        if batch_idx >= len(trajs): return []
        batch_trajs = trajs[batch_idx]
        states = []
        for tid in death_ids:
            traj = batch_trajs.get(tid)
            if traj is None or len(traj) == 0: continue
            mask = traj[:, -1] == (t - 1)                                       # 查找t-1时刻
            if mask.any(): states.append(np.append(traj[mask][0, :-1], tid))    # [状态7维, ID]
        return states

    def get_death_target_states(self, batch_idx: int, timestep: int, death_ids: Set[int]) -> List[np.ndarray]:
        """公共接口：获取消亡目标状态"""
        return self._get_death_states(batch_idx, timestep, death_ids)

    def _save_batch(self, batch_data: Tuple, batch_idx: int):
        """保存批次数据"""
        epoch_dir = self.save_dir / f"epoch_{self.current_epoch}"
        epoch_dir.mkdir(exist_ok=True)
        raw_meas, raw_gt, raw_ids, trajs = batch_data
        save_dict = {f'meas_{i}_{t}': m for i, seq in enumerate(raw_meas) for t, m in enumerate(seq)}
        save_dict.update({f'gt_{i}_{t}': g for i, seq in enumerate(raw_gt) for t, g in enumerate(seq)})
        save_dict.update({f'ids_{i}_{t}': d for i, seq in enumerate(raw_ids) for t, d in enumerate(seq)})
        np.savez(epoch_dir / f"batch_{batch_idx:04d}.npz", **save_dict)
        with open(epoch_dir / f"batch_{batch_idx:04d}_traj.json", 'w') as f:
            json.dump([{str(k): v.tolist() for k, v in td.items()} for td in trajs], f)
        if self.verbose and batch_idx == 0: self.logger.info(f"保存数据到: {epoch_dir}")