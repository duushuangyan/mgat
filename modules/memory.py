"""
作者: DU
日期: 2025
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
import logging


class MemoryModule(nn.Module):
    """
    记忆模块 - 动态目标嵌入存储
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化记忆模块
        """
        super().__init__()

        self.embed_dim = config.get('embed_dim', 256)
        self.max_targets = config.get('max_targets', 50)
        self.max_history_length = config.get('max_history_length', 100)
        self.max_timesteps = config.get('max_timesteps', 100)

        self.current_timestep: int = 0
        self.batch_size: int = 0
        # 格式: batch_memories[b][target_id] = {'embeddings': {t: emb}, 'birth_time': t, ...}
        self.batch_memories: List[Dict[int, Dict[str, Any]]] = []

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"MemoryModule初始化: embed_dim={self.embed_dim}, "
                        f"max_targets={self.max_targets}, max_history={self.max_history_length}")

    def reset(self, batch_size: Optional[int] = None):
        """
        重置记忆模块（每个epoch开始时调用）
        """
        if batch_size is not None:
            self.batch_size = batch_size

        self.current_timestep = 0
        self.batch_memories = [{} for _ in range(self.batch_size)]
        self.max_target_ids = torch.zeros(self.batch_size, dtype=torch.long)
        self.logger.debug(f"记忆模块重置: batch_size={self.batch_size}, timestep=0")

    def get_max_ids(self, device: torch.device) -> torch.Tensor:
        """获取当前每个batch的最大ID"""
        if self.max_target_ids.device != device:
            self.max_target_ids = self.max_target_ids.to(device)
        return self.max_target_ids

    def update(self, assoc_output: Dict[str, Any], timestep: Optional[int] = None):
        """
        根据关联模块输出更新记忆
        """
        t = timestep if timestep is not None else self.current_timestep

        updated_embeddings = assoc_output['updated_target_embeddings']  # [B, S', D]
        updated_ids = assoc_output['updated_target_ids']                # [B, S']
        updated_mask = assoc_output['updated_target_mask']              # [B, S']
        # dead_target_mask 仅覆盖原有目标 [B, S]，不包含新生目标位置
        dead_mask = assoc_output['dead_target_mask']                    # [B, S]

        if updated_ids.numel() > 0 and updated_mask.any():
            masked_ids = updated_ids.masked_fill(~updated_mask, -1)  # [B, S']
            current_batch_max = masked_ids.max(dim=1).values  # [B]
            current_batch_max = torch.clamp(current_batch_max, min=0)
            self.max_target_ids = torch.max(self.max_target_ids.to(updated_ids.device), current_batch_max)

        B, S_prime, D = updated_embeddings.shape
        S = dead_mask.shape[1]  # 原有目标数（S' = S + 1）

        # 检查batch_size一致性
        if B != self.batch_size:
            self.logger.warning(f"batch_size不匹配: {B} vs {self.batch_size}，重新初始化")
            self.reset(B)

        for b in range(B):
            batch_mem = self.batch_memories[b]

            for s in range(S_prime):
                if not updated_mask[b, s].item():
                    continue
                target_id = int(updated_ids[b, s].item())
                if target_id < 0:
                    continue
                embedding = updated_embeddings[b, s]  # [D]

                if target_id not in batch_mem:
                    batch_mem[target_id] = {
                        'embeddings': {t: embedding},   # {timestep: embedding}
                        'birth_time': t,                # 新生时间步
                        'death_time': -1,               # -1表示存活
                        'is_active': True               # 存活标志
                    }
                    self.logger.debug(f"Batch {b}: 注册新目标 ID={target_id} at t={t}")
                else:
                    target_mem = batch_mem[target_id]
                    if target_mem['is_active']:
                        target_mem['embeddings'][t] = embedding

            for s in range(S):
                if dead_mask[b, s].item():
                    # 获取原有目标的ID（updated_ids的前S个位置对应原有目标）
                    target_id = int(updated_ids[b, s].item())
                    if target_id >= 0 and target_id in batch_mem:
                        target_mem = batch_mem[target_id]
                        if target_mem['is_active']:
                            target_mem['is_active'] = False
                            target_mem['death_time'] = t
                            self.logger.debug(f"Batch {b}: 目标消亡 ID={target_id} at t={t}")

    def advance_timestep(self):
        """时间步前进（在每个时间步处理完成后调用）"""
        self.current_timestep += 1

    def extract_for_tam(self, assoc_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        从记忆中提取当前存活目标的历史嵌入（供TAM使用）
        """
        updated_ids = assoc_output['updated_target_ids']      # [B, S']
        updated_mask = assoc_output['updated_target_mask']    # [B, S']
        dead_mask = assoc_output['dead_target_mask']          # [B, S]

        B, S_prime = updated_ids.shape
        S = dead_mask.shape[1]  # 原有目标数，S' = S + 1

        alive_mask = updated_mask.clone()  # [B, S']
        alive_mask[:, :S] = alive_mask[:, :S] & ~dead_mask  # 前S个位置: 有效且未消亡

        alive_target_ids = []      # List[List[int]]: alive_target_ids[b] = [id1, id2, ...]
        history_embeddings = []    # List[Dict[int, Dict[int, Tensor]]]: history[b][tid][t] = emb

        for b in range(B):
            batch_alive_ids = []   # 该batch存活的目标ID列表
            batch_history = {}     # 该batch存活目标的历史嵌入 {target_id: {t: emb}}

            # 遍历所有目标位置，筛选存活目标
            for s in range(S_prime):
                # 跳过非存活目标
                if not alive_mask[b, s].item():
                    continue

                target_id = int(updated_ids[b, s].item())
                # 跳过无效ID（ID应为0-N的整数）或不在记忆中的目标（理论上不应发生）
                if target_id < 0 or target_id not in self.batch_memories[b]:
                    continue

                batch_alive_ids.append(target_id)
                batch_history[target_id] = self.batch_memories[b][target_id]['embeddings']

            alive_target_ids.append(batch_alive_ids)
            history_embeddings.append(batch_history)

        return {
            'alive_target_ids': alive_target_ids,       # List[List[int]]
            'history_embeddings': history_embeddings    # List[Dict[int, Dict[int, Tensor]]]
        }

    def dump(self) -> Dict[str, Any]:
        """
        输出记忆模块数据
        """
        # 总时间步数 = max_timesteps（配置值，默认100）
        total_timesteps = self.max_timesteps

        result = {
            'batch_size': self.batch_size,
            'total_timesteps': total_timesteps,
            'batch_data': []
        }

        for b in range(self.batch_size):
            batch_mem = self.batch_memories[b]

            targets_data = {}
            for target_id, target_mem in batch_mem.items():
                targets_data[target_id] = {
                    'birth_time': target_mem['birth_time'],
                    'death_time': target_mem['death_time'],
                    'is_active': target_mem['is_active'],
                    # 嵌入用1代替，只保留时间步信息
                    'embeddings': {t: 1 for t in target_mem['embeddings'].keys()}
                }

            timestep_stats = {}
            for t in range(total_timesteps):
                alive_ids = [
                    tid for tid, tmem in batch_mem.items()
                    if tmem['birth_time'] <= t and (tmem['death_time'] == -1 or tmem['death_time'] > t)
                ]
                timestep_stats[t] = {
                    'num_alive': len(alive_ids),
                    'alive_ids': sorted(alive_ids)
                }

            result['batch_data'].append({
                'total_targets': len(batch_mem),
                'targets': targets_data,
                'timestep_stats': timestep_stats
            })

        return result

    def detach_all(self):
        """
        分离所有存储嵌入的梯度
        """
        for b in range(len(self.batch_memories)):
            for target_id, target_mem in self.batch_memories[b].items():
                target_mem['embeddings'] = {t: emb.detach() for t, emb in target_mem['embeddings'].items()}
        self.logger.debug("已分离所有存储嵌入的梯度")