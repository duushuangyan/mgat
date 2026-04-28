import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any, Set
from contextlib import contextmanager
import warnings
# ==================== 检测可选依赖 ====================
try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import scienceplots

    HAS_SCIENCEPLOTS = True
except ImportError:
    HAS_SCIENCEPLOTS = False
    warnings.warn("scienceplots未安装，使用备用样式。安装: pip install SciencePlots", stacklevel=2)


class DataVisualizer:
    """
    统一管理全局绑图配置
    """

    FIGSIZE_SINGLE = (3.5, 2.8)  # IEEE单栏尺寸
    FIGSIZE_DOUBLE = (7.0, 3.5)  # IEEE双栏尺寸
    FIGSIZE_SQUARE = (3.5, 3.5)  # 正方形（场景图）
    COLORS = sns.color_palette("colorblind", 12)  # 色盲友好配色

    def __init__(self, save_dir: str = "./figures", style: str = "ieee", fmt: str = "pdf", dpi: int = 300,
                 font_family: str = "Times New Roman", font_size: int = 8, use_latex: bool = False, transparent: bool = True, verbose: bool = True):
        """
        初始化可视化器
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.style, self.fmt, self.dpi = style, fmt, dpi
        self.transparent, self.verbose = transparent, verbose

        # 样式映射
        self._styles = {
            'ieee': ['science', 'ieee', 'no-latex'] if HAS_SCIENCEPLOTS else ['default'],
            'nature': ['science', 'nature', 'no-latex'] if HAS_SCIENCEPLOTS else ['default'],
            'presentation': ['science', 'notebook', 'no-latex'] if HAS_SCIENCEPLOTS else ['default'],
            'default': ['default']
        }

        # 自定义rcParams（确保即使没有scienceplots也能获得学术风格）
        self._rc = {
            'font.family': 'serif',
            'font.serif': [font_family, 'DejaVu Serif', 'Computer Modern Roman'],
            'font.size': font_size,
            'axes.labelsize': font_size,
            'axes.titlesize': font_size + 1,
            'xtick.labelsize': font_size - 1,
            'ytick.labelsize': font_size - 1,
            'legend.fontsize': font_size - 1,
            'axes.grid': False,
            'axes.linewidth': 0.8,
            'axes.spines.top': True,
            'axes.spines.right': True,
            'xtick.direction': 'in',
            'ytick.direction': 'in',
            'xtick.major.width': 0.6,
            'ytick.major.width': 0.6,
            'xtick.major.size': 3,
            'ytick.major.size': 3,
            'lines.linewidth': 1.0,
            'lines.markersize': 4,
            'figure.dpi': dpi,
            'savefig.dpi': dpi,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.02,
            'text.usetex': use_latex,
            'mathtext.fontset': 'cm',
        }

        if verbose:
            print(f"[BaseVisualizer] 初始化 | 目录: {self.save_dir} | 风格: {style} | 格式: {fmt}")

    # ======================== 上下文管理器 ========================
    @contextmanager
    def canvas(self, filename: str, figsize: Tuple[float, float] = None,
               subfolder: str = "", nrows: int = 1, ncols: int = 1, **fig_kw):
        """
        画布上下文管理器
        """
        figsize = figsize or self.FIGSIZE_SINGLE
        with plt.style.context(self._styles.get(self.style, ['default'])):
            plt.rcParams.update(self._rc)
            fig, axs = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True, **fig_kw)
            try:
                yield (fig, axs)
            finally:
                out_dir = self.save_dir / subfolder if subfolder else self.save_dir
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{filename}.{self.fmt}"
                fig.savefig(out_path, transparent=self.transparent, format=self.fmt)
                plt.close(fig)
                if self.verbose:
                    print(f"[Saved] {out_path}")
    # ======================== 工具方法 ========================
    @staticmethod
    def to_np(data: Any) -> np.ndarray:
        """通用数据转NumPy（支持Tensor/List/ndarray）"""
        if HAS_TORCH and isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        return np.asarray(data) if not isinstance(data, np.ndarray) else data

    def color(self, idx: int) -> tuple:
        """获取循环配色（基于目标ID）"""
        return self.COLORS[idx % len(self.COLORS)]

    # ==========仿真场景可视化==========
    def batch_scenario_plot(
            self,
            measurements: List,
            unique_ids: List,
            trajectories: List[Dict[int, np.ndarray]],
            epoch: int,
            single_idx: int = 0,
            multi_indices: List[int] = None,
            fov: Tuple[float, float] = (0, 10000),
            subfolder: str = "scenarios"
    ):
        """
            1. 单场景详图: 带完整图例（每个目标ID + Measurements标签），带坐标轴标签
            2. 多场景网格图: 2×3紧凑布局，每个子图仅显示统计信息（目标数、量测数）
        """
        COLOR_CLUTTER = '#808080'  # 灰色: 杂波/噪声
        multi_indices = multi_indices or list(range(min(6, len(trajectories))))

        with self.canvas(f"scenario_e{epoch}_s{single_idx}", figsize=self.FIGSIZE_SINGLE, subfolder=subfolder) as (
        fig, ax):
            meas_b, ids_b = measurements[single_idx], unique_ids[single_idx]
            # 合并所有时间步的量测（支持单时刻和多时刻两种格式）
            if isinstance(meas_b, list):
                all_meas = np.vstack([self.to_np(m) for m in meas_b if len(m) > 0]) if meas_b else np.zeros((0, 2))
                all_ids = np.concatenate([self.to_np(i) for i in ids_b if len(i) > 0]) if ids_b else np.array([])
            else:
                all_meas, all_ids = self.to_np(meas_b), self.to_np(ids_b)
            traj_dict = trajectories[single_idx]
            legend_handles = []
            valid_mask = (all_ids != -2)
            if valid_mask.any() and len(all_meas) > 0:
                valid_meas = all_meas[valid_mask]
                r, theta = valid_meas[:, 0], valid_meas[:, 1]  # 极坐标
                x_meas, y_meas = r * np.cos(theta), r * np.sin(theta)  # 转笛卡尔
                ax.scatter(x_meas, y_meas, c=COLOR_CLUTTER, s=2, alpha=0.9, marker='.', zorder=1,
                           edgecolors='none')
                legend_handles.append(
                    Line2D([0], [0], marker='o', color='w', label='Meas', markerfacecolor=COLOR_CLUTTER,
                           markersize=4, linestyle='None'))
            for target_id, track_data in traj_dict.items():
                track = self.to_np(track_data)
                if len(track) == 0:
                    continue
                color = self.color(target_id)  # 获取该目标的颜色
                xs, ys, vxs, vys = track[:, 0], track[:, 1], track[:, 2], track[:, 3]
                thetas, majors, minors = track[:, 4], track[:, 5] * 2, track[:, 6] * 2  # 半轴→全轴
                for j in range(len(xs)):
                    ax.add_patch(patches.Ellipse(xy=(xs[j], ys[j]), width=majors[j], height=minors[j],
                                                 angle=np.degrees(thetas[j]), edgecolor=color, facecolor='none',
                                                 linewidth=0.6, alpha=0.85, zorder=2))
                ax.scatter(xs, ys, c=[color], s=3, marker='o', zorder=3, edgecolors='none')  # 质心点
                step = max(1, len(xs) // 8)
                idx = np.arange(0, len(xs), step)
                if len(idx) > 0:
                    ax.quiver(xs[idx], ys[idx], vxs[idx] * 0.4, vys[idx] * 0.4, color=color, angles='xy',
                              scale_units='xy', scale=1, width=0.004, headwidth=3, headlength=3.5, zorder=4, alpha=0.9)
                legend_handles.append(Line2D([], [], color=color, lw=1.2, label=f'ID {target_id}'))
            ax.set_xlim(*fov)
            ax.set_ylim(*fov)
            ax.set_aspect('equal', adjustable='box')
            ax.set_title(f"Scenario (Epoch {epoch}, Sample {single_idx})", fontsize=9)
            ax.set_xlabel("X Position (m)")
            ax.set_ylabel("Y Position (m)")
            if legend_handles:
                ax.legend(handles=legend_handles, loc='upper right', fontsize=6, frameon=True, framealpha=0.85,
                          edgecolor='gray', fancybox=False, handlelength=0.8, handletextpad=0.3, borderpad=0.3,
                          labelspacing=0.2)

        nrows, ncols = 2, 3
        figsize = (ncols * 3.0, nrows * 2.8)

        with self.canvas(f"scenarios_e{epoch}_grid", figsize=figsize, subfolder=subfolder, nrows=nrows,
                         ncols=ncols) as (fig, axs):
            axs_flat = axs.flatten() if isinstance(axs, np.ndarray) else [axs]

            for i, b_idx in enumerate(multi_indices[:nrows * ncols]):
                ax = axs_flat[i]

                if b_idx >= len(trajectories):
                    ax.text(0.5, 0.5, "No Data", ha='center', va='center', transform=ax.transAxes, fontsize=10,
                            color='gray')
                    ax.axis('off')
                    continue
                meas_b, ids_b = measurements[b_idx], unique_ids[b_idx]
                if isinstance(meas_b, list):
                    all_meas = np.vstack([self.to_np(m) for m in meas_b if len(m) > 0]) if meas_b else np.zeros((0, 2))
                    all_ids = np.concatenate([self.to_np(i) for i in ids_b if len(i) > 0]) if ids_b else np.array([])
                else:
                    all_meas, all_ids = self.to_np(meas_b), self.to_np(ids_b)
                traj_dict = trajectories[b_idx]
                valid_mask = (all_ids != -2)
                if valid_mask.any() and len(all_meas) > 0:
                    valid_meas = all_meas[valid_mask]
                    r, theta = valid_meas[:, 0], valid_meas[:, 1]
                    x_meas, y_meas = r * np.cos(theta), r * np.sin(theta)
                    ax.scatter(x_meas, y_meas, c=COLOR_CLUTTER, s=2, alpha=0.9, marker='.', zorder=1,
                               edgecolors='none')
                for target_id, track_data in traj_dict.items():
                    track = self.to_np(track_data)
                    if len(track) == 0:
                        continue
                    color = self.color(target_id)
                    xs, ys, vxs, vys = track[:, 0], track[:, 1], track[:, 2], track[:, 3]
                    thetas, majors, minors = track[:, 4], track[:, 5] * 2, track[:, 6] * 2
                    for j in range(len(xs)):
                        ax.add_patch(patches.Ellipse(xy=(xs[j], ys[j]), width=majors[j], height=minors[j],
                                                     angle=np.degrees(thetas[j]), edgecolor=color, facecolor='none',
                                                     linewidth=0.6, alpha=0.85, zorder=2))
                    ax.scatter(xs, ys, c=[color], s=3, marker='o', zorder=3, edgecolors='none')
                    step = max(1, len(xs) // 8)
                    idx = np.arange(0, len(xs), step)
                    if len(idx) > 0:
                        ax.quiver(xs[idx], ys[idx], vxs[idx] * 0.4, vys[idx] * 0.4, color=color, angles='xy',
                                  scale_units='xy', scale=1, width=0.004, headwidth=3, headlength=3.5, zorder=4,
                                  alpha=0.9)
                n_targets = len(traj_dict)
                n_meas = np.sum(all_ids != -2) if len(all_ids) > 0 else 0
                ax.text(0.03, 0.97, f"T:{n_targets} M:{n_meas}", transform=ax.transAxes, fontsize=7, va='top',
                        ha='left', bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7, ec='gray', lw=0.5))
                ax.set_xlim(*fov)
                ax.set_ylim(*fov)
                ax.set_aspect('equal', adjustable='box')
                ax.set_title(f"E{epoch}-S{b_idx}", fontsize=8, pad=2)
            for ax in axs_flat[len(multi_indices):]:
                ax.axis('off')

    def step_output_plot(
            self,
            meas_t: 'torch.Tensor',  # [B, M, 2] 极坐标量测 (range, angle)
            meas_ids: 'torch.Tensor',  # [B, M] 量测真实归属ID (-1=杂波, -2=填充, >=0=目标)
            meas_mask: 'torch.Tensor',  # [B, M] 量测有效掩码 (True=有效)
            gt_t: 'List[torch.Tensor]',  # 当前时刻GT状态, gt_t[b]: [N_b, 8], 列=[x,y,vx,vy,θ,a,b,id]
            next_gt: 'Any',  # 下一时刻GT状态, List[Tensor] [N_b, 8] 或 None
            pred_7d: 'List[np.ndarray]',  # 预测7D物理状态, pred_7d[b]: [N, 7] 列=[x,y,vx,vy,θ,a,b]
            pred_ids: 'torch.Tensor',  # [B, N] 预测目标ID
            pred_mask: 'torch.Tensor',  # [B, N] 预测目标掩码
            outputs: 'Dict[str, Any]',  # MGAT模型输出字典（用于关联结果）
            birth_ids: 'List[Set[int]]',  # 当前时间步新生目标ID集合列表
            death_ids: 'List[Set[int]]',  # 当前时间步消亡目标ID集合列表
            epoch: int,  # 当前epoch
            timestep: int,  # 当前时间步
            batch_idx: int = 0,  # 选择可视化的样本索引
            subfolder: str = "step_outputs"  # 保存子文件夹
    ):
        """
        单时间步模型输出可视化 - 三子图布局
        """
        b = batch_idx
        meas_np, mask_np, ids_np = self.to_np(meas_t[b]), self.to_np(meas_mask[b]).astype(bool), self.to_np(meas_ids[b])
        meas_xy = np.column_stack(
            [meas_np[:, 0] * np.cos(meas_np[:, 1]), meas_np[:, 0] * np.sin(meas_np[:, 1])])  # [M, 2]

        gt_np = self.to_np(gt_t[b]) if gt_t[b].numel() > 0 else np.zeros((0, 8))

        if next_gt is not None and next_gt[b] is not None and next_gt[b].numel() > 0:
            next_gt_np = self.to_np(next_gt[b])  # [N', 8]
        else:
            next_gt_np = np.zeros((0, 8))

        pred_np = pred_7d[b]                                             # [N, 7]
        pred_ids_np = self.to_np(pred_ids[b])                            # [N]
        pred_mask_np = self.to_np(pred_mask[b]).astype(bool)             # [N]

        assoc_meas_ids = self.to_np(outputs['pred_meas_target_ids'][b])  # [M]
        is_clutter = self.to_np(outputs['is_clutter'][b]).astype(bool)
        is_new_meas = self.to_np(outputs['is_new_target_meas'][b]).astype(bool)
        new_target_id = int(self.to_np(outputs['new_target_ids'][b, 0])) if outputs['has_new_target'][b] else -1

        input_target_ids = self.to_np(outputs['input_target_ids'][b])    # [S]
        input_target_mask = self.to_np(outputs['input_target_mask'][b]).astype(bool)  # [S]

        birth_set = birth_ids[b] if birth_ids else set()
        death_set = death_ids[b] if death_ids else set()

        figsize = (11.5, 3.8)
        with self.canvas(f"step_e{epoch}_t{timestep}_b{b}", figsize=figsize, subfolder=subfolder, nrows=1, ncols=3) as (
        fig, axs):

            ax1 = axs[0]
            handles_1 = []
            for i in range(gt_np.shape[0]):
                x, y, vx, vy, theta, a, b_ax, tid = gt_np[i, 0], gt_np[i, 1], gt_np[i, 2], gt_np[i, 3], gt_np[i, 4], \
                gt_np[i, 5], gt_np[i, 6], int(gt_np[i, 7])
                color = self.color(tid)
                ax1.add_patch(
                    patches.Ellipse(xy=(x, y), width=2 * a, height=2 * b_ax, angle=np.degrees(theta), edgecolor=color,
                                    facecolor='none', linewidth=0.5, zorder=1))
                ax1.scatter([x], [y], c=[color], s=10, marker='o', zorder=2, edgecolors='none')
                ax1.quiver(x, y, vx * 1.5, vy * 1.5, color=color, angles='xy', scale_units='xy', scale=1, width=0.002,
                           headwidth=3.5, headlength=4, zorder=3, alpha=0.9)
                handles_1.append(Line2D([], [], color=color, lw=0.8, marker='o', markersize=3, label=f'T{tid}'))
            if mask_np.any():
                ax1.scatter(meas_xy[mask_np, 0], meas_xy[mask_np, 1], c='#808080', s=3, alpha=0.7, marker='.', zorder=5,
                            edgecolors='none')
                handles_1.insert(0, Line2D([0], [0], marker='o', color='w', markerfacecolor='#808080', markersize=4,
                                           linestyle='None', label='Meas'))
            ax1.set_aspect('equal', adjustable='datalim')
            ax1.set_title(f'Ground Truth (t={timestep})', fontsize=9, pad=3)
            ax1.set_xlabel('X (m)', fontsize=8)
            ax1.set_ylabel('Y (m)', fontsize=8)
            if handles_1: ax1.legend(handles=handles_1, loc='upper right', fontsize=5, framealpha=0.85,
                                     edgecolor='gray', handlelength=1.2, labelspacing=0.3)

            ax2 = axs[1]
            handles_2 = []
            existing_meas_mask = mask_np & (~is_clutter) & (~is_new_meas)
            unique_assoc_ids = np.unique(
                assoc_meas_ids[existing_meas_mask & (assoc_meas_ids >= 0)]) if existing_meas_mask.any() else np.array([])
            clutter_mask = mask_np & is_clutter
            if clutter_mask.any():
                ax2.scatter(meas_xy[clutter_mask, 0], meas_xy[clutter_mask, 1], c='#808080', s=4, alpha=0.7, marker='.',
                            zorder=1, edgecolors='none')
                handles_2.append(
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='#808080', markersize=4, linestyle='None',
                           label='Clutter'))
            new_meas_mask = mask_np & is_new_meas
            if new_meas_mask.any():
                ax2.scatter(meas_xy[new_meas_mask, 0], meas_xy[new_meas_mask, 1], c='purple', s=8, alpha=0.85,
                            marker='*', zorder=2, edgecolors='none')
                handles_2.append(
                    Line2D([0], [0], marker='*', color='w', markerfacecolor='purple', markersize=5, linestyle='None',
                           label=f'Birth(ID={new_target_id})'))
            for tid in unique_assoc_ids:
                tid_mask = existing_meas_mask & (assoc_meas_ids == tid)
                if tid_mask.any():
                    color = self.color(int(tid))
                    ax2.scatter(meas_xy[tid_mask, 0], meas_xy[tid_mask, 1], c=[color], s=5, alpha=0.85, marker='o',
                                zorder=2, edgecolors='none')
                    handles_2.append(
                        Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=4, linestyle='None',
                               label=f'T{int(tid)}'))
            for i in range(gt_np.shape[0]):
                tid = int(gt_np[i, 7])
                if tid < 0:
                    continue
                if not (input_target_mask & (input_target_ids == tid)).any():
                    continue
                x, y, theta, a, b_ax = gt_np[i, 0], gt_np[i, 1], gt_np[i, 4], gt_np[i, 5], gt_np[i, 6]
                ax2.add_patch(
                    patches.Ellipse(xy=(x, y), width=2 * a, height=2 * b_ax, angle=np.degrees(theta), edgecolor='black',
                                    facecolor='none', linewidth=0.4, linestyle='--', zorder=3))
            handles_2.append(Line2D([], [], color='black', lw=0.6, linestyle='--', label='Input_T'))
            ax2.set_aspect('equal', adjustable='datalim')
            ax2.set_title(f'Association Result (t={timestep})', fontsize=9, pad=3)
            ax2.set_xlabel('X (m)', fontsize=8)
            ax2.set_ylabel('Y (m)', fontsize=8)
            if handles_2: ax2.legend(handles=handles_2, loc='upper right', fontsize=5, framealpha=0.85,
                                     edgecolor='gray', handlelength=1.2, labelspacing=0.3)

            ax3 = axs[2]
            handles_3 = []

            all_tids = set()
            gt_t_by_id = {}
            for i in range(gt_np.shape[0]):
                tid = int(gt_np[i, 7])
                if tid >= 0:
                    gt_t_by_id[tid] = gt_np[i, :7]
                    all_tids.add(tid)
            gt_next_by_id = {}
            for i in range(next_gt_np.shape[0]):
                tid = int(next_gt_np[i, 7])
                if tid >= 0:
                    gt_next_by_id[tid] = next_gt_np[i, :7]
                    all_tids.add(tid)
            pred_by_id = {}
            for i in range(pred_np.shape[0]):
                if pred_mask_np[i]:
                    tid = int(pred_ids_np[i])
                    if tid >= 0:
                        pred_by_id[tid] = pred_np[i, :7]
                        all_tids.add(tid)

            for tid in sorted(all_tids):
                color = self.color(tid)
                has_gt_t = tid in gt_t_by_id
                has_gt_next = tid in gt_next_by_id
                has_pred = tid in pred_by_id
                added_legend = False
                vel_scale = 0.4

                if has_gt_t:
                    x, y, vx, vy, theta, a, b_ax = gt_t_by_id[tid]
                    ax3.add_patch(
                        patches.Ellipse(xy=(x, y), width=2 * a, height=2 * b_ax, angle=np.degrees(theta),
                                        edgecolor=color, facecolor='none', linewidth=0.5, zorder=1))
                    ax3.scatter([x], [y], c=[color], s=6, marker='o', zorder=2, edgecolors='none')
                    ax3.quiver(x, y, vx * vel_scale, vy * vel_scale, color=color, angles='xy', scale_units='xy', scale=1,
                               width=0.001, headwidth=3.5, headlength=4, zorder=3, alpha=1)
                    handles_3.append(Line2D([], [], color=color, lw=0.8, marker='o', markersize=3, label=f'GT_T{tid}'))
                    added_legend = True

                if has_gt_next:
                    x, y = gt_next_by_id[tid][0], gt_next_by_id[tid][1]
                    theta, a, b_ax = gt_next_by_id[tid][4], gt_next_by_id[tid][5], gt_next_by_id[tid][6]
                    ax3.add_patch(
                        patches.Ellipse(xy=(x, y), width=2 * a, height=2 * b_ax, angle=np.degrees(theta),
                                        edgecolor=color, facecolor='none', linewidth=0.5, zorder=4))
                    ax3.scatter([x], [y], c=[color], s=6, marker='o', zorder=5, edgecolors='none')

                if has_pred:
                    x, y, vx, vy, theta, a, b_ax = pred_by_id[tid]
                    ax3.add_patch(
                        patches.Ellipse(xy=(x, y), width=2 * a, height=2 * b_ax, angle=np.degrees(theta),
                                        edgecolor=color, facecolor='none', linewidth=0.5, linestyle='--', zorder=6))
                    ax3.scatter([x], [y], c='none', s=6, marker='o', zorder=7, edgecolors=color, linewidths=0.5)
                    ax3.quiver(x, y, vx * vel_scale, vy * vel_scale, color=color, angles='xy', scale_units='xy', scale=1,
                               width=0.001, headwidth=3.5, headlength=4, zorder=3, alpha=1)
                    handles_3.append(
                        Line2D([], [], color=color, lw=0.8, linestyle='--', marker='o', markersize=3, fillstyle='none',
                               label=f'Pred_T{tid}'))

            ax3.set_aspect('equal', adjustable='datalim')
            ax3.set_title(f'Prediction (t→t+1, t={timestep})', fontsize=9, pad=3)
            ax3.set_xlabel('X (m)', fontsize=8)
            ax3.set_ylabel('Y (m)', fontsize=8)
            if handles_3: ax3.legend(handles=handles_3, loc='upper right', fontsize=5, framealpha=0.85,
                                     edgecolor='gray', handlelength=1.2, labelspacing=0.3, ncol=2)