import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
import time
from datetime import datetime
import yaml
import sys
import os

# 导入项目模块
from modules.mgat import MGAT
from modules.loss import MGATLoss
from data.loading.radardata_loader import RadarDataLoader
from data.loading.preprocessor import RadarDataPreprocessor
from utils.config import load_config, save_config, setup_logging
from utils.data_visualization import DataVisualizer


# ----------------------------
# 内存监控（仅用于打印RSS）
# ----------------------------
import psutil
proc = psutil.Process(os.getpid())
def mem_mb() -> float:
    return proc.memory_info().rss / 1024 / 1024


def compute_tf_ratio(epoch: int, ss_config) -> float:
    """
    计算当前epoch的Teacher Forcing比率

    调度策略:
      epoch < warmup:             tf_ratio = initial  (纯TF预热)
      warmup <= epoch < w+decay:  tf_ratio 线性衰减
      epoch >= warmup + decay:    tf_ratio = final

    Args:
        epoch: 当前epoch编号
        ss_config: scheduled_sampling配置字典

    Returns:
        tf_ratio: [0, 1]，1.0=纯TF，0.0=完全使用模型预测
    """
    initial = ss_config.get('initial_tf_ratio', 1.0)
    final = ss_config.get('final_tf_ratio', 0.0)
    warmup = ss_config.get('warmup_epochs', 1000)
    decay = ss_config.get('decay_epochs', 5000)

    if epoch < warmup:
        return initial
    progress = min(1.0, (epoch - warmup) / max(1, decay))
    return initial + (final - initial) * progress


def main():
    parser = argparse.ArgumentParser(description='MGAT多扩展目标跟踪模型训练')
    parser.add_argument('--data_config', type=str, default='configs/data/data_config.yaml', help='数据配置文件路径')
    parser.add_argument('--train_config', type=str, default='configs/experiment/train_config.yaml', help='训练配置文件路径')
    parser.add_argument('--model_config', type=str, default='configs/model/model_config.yaml', help='模型配置文件路径')
    parser.add_argument('--pre_config', type=str, default='configs/experiment/preproces_config.yaml', help='数据预处理配置文件路径')
    parser.add_argument('--resume_training', type=str, default=None, help='恢复训练的检查点路径')
    parser.add_argument('--output_dir', type=str, default='outputs', help='输出目录')
    parser.add_argument('--device', type=str, default='auto', help='计算设备 (cpu, cuda, auto)')
    parser.add_argument('--training_mode', type=str, default='teacher_forcing', help='训练模式: teacher_forcing,评估验证：evaluating')
    args = parser.parse_args()

    # 创建输出目录和日志
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(args.output_dir) / f"mgat_train_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    # 创建TensorBoard、checkpoints子目录
    tensorboard_dir, checkpoints_dir = save_dir / 'tensorboard', save_dir / 'checkpoints'
    [d.mkdir(exist_ok=True) for d in (tensorboard_dir, checkpoints_dir)]

    # 设置日志
    logger = setup_logging(save_dir)
    logger.info("=" * 60)
    logger.info("MGAT多扩展目标跟踪模型训练")
    logger.info("=" * 60)
    logger.info(f"训练模式: {args.training_mode}")

    # 设置计算设备，当args.device == 'auto' 且 CUDA 可用 -> 使用cuda；否则如果用户指定cpu/cuda -> 直接使用；如果auto但无cuda -> cpu
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available()
                          else args.device if args.device != 'auto' else 'cpu')
    logger.info(f"使用计算设备: {device}")

    data_params = load_config(args.data_config)
    training_params = load_config(args.train_config)
    model_params = load_config(args.model_config)
    preprocess_params = load_config(args.pre_config)
    save_config(data_params, save_dir / 'data_config.yaml')
    save_config(training_params, save_dir / 'train_config.yaml')
    save_config(model_params, save_dir / 'model_config.yaml')
    save_config(preprocess_params, save_dir / 'preprocess_config.yaml')

    # 创建数据加载器
    logger.info("创建数据加载器...")
    data_loader = RadarDataLoader(config_path=args.data_config, mode='train', save_generated_data=False, verbose=True)
    batch_size = data_loader.batch_size
    time_steps = data_loader.total_timesteps
    logger.info(f"  批次大小: {batch_size}, 时间步数: {time_steps}")

    # 创建预处理器
    logger.info("创建预处理器...")
    preprocessor = RadarDataPreprocessor(preprocess_params)

    # 创建MGAT模型
    logger.info("创建MGAT模型...")
    model = MGAT(model_params).to(device)

    # 创建损失函数
    logger.info("创建损失计算模块...")
    mgatloss = MGATLoss(model_params.loss).to(device)

    # 创建优化器和学习率调度器
    logger.info("创建优化器和调度器...")
    optimizer = optim.AdamW(
        model.parameters(),
        lr=training_params.learning_rate,
        weight_decay=training_params.weight_decay,
        betas=(0.9, 0.999)
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=training_params.lr_reduce_patience,
        factor=training_params.lr_reduce_factor,
        min_lr=training_params.lr_limit
    )

    # 检查点恢复逻辑
    start_epoch = 0
    if args.resume_training is not None and Path(args.resume_training).exists():
        checkpoint = torch.load(args.resume_training, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f"成功恢复检查点: {args.resume_training}，从 Epoch {start_epoch} 继续训练")

    # 创建TensorBoard记录器
    writer = SummaryWriter(log_dir=str(tensorboard_dir))

    # 创建数据可视化模块
    visualizer = DataVisualizer(save_dir=save_dir / 'demo')
    plot_epoch_interval = 1000
    plot_timestep_interval = 10

    # ==================== 训练循环 ====================
    logger.info("=" * 60)
    logger.info(f"开始训练...epoch={training_params.num_epochs}")
    logger.info("=" * 60)
    # 读取BPTT步长参数truncation_steps
    truncation_steps = training_params.get('truncation_steps', -1)
    if truncation_steps > 0:
        logger.info(f"启用截断BPTT (TBPTT)，截断步长: {truncation_steps}")
    else:
        logger.info("使用全序列BPTT训练")

    # 读取Scheduled Sampling配置
    ss_config = training_params.get('scheduled_sampling', {})
    ss_enabled = ss_config.get('enabled', False)
    if ss_enabled:
        logger.info(f"启用Scheduled Sampling: warmup={ss_config.get('warmup_epochs', 50)}, "
                     f"decay={ss_config.get('decay_epochs', 1000)}, "
                     f"tf_ratio: {ss_config.get('initial_tf_ratio', 1.0)} → {ss_config.get('final_tf_ratio', 0.0)}")
    else:
        logger.info("使用纯Teacher Forcing训练（Scheduled Sampling未启用）")

    for epoch in range(start_epoch, training_params.num_epochs):
        epoch_start_time = time.time()  # 记录推理时间
        logger.info(f"[MEM] epoch {epoch} start RSS={mem_mb():.1f} MB")     # 监控显存
        model.train()

        # 设置数据加载器的epoch（更新随机种子以获取不同数据）
        data_loader.set_epoch(epoch)

        # 获取批次数据
        measurements, ground_truth, unique_ids, trajectories, birth_tags, death_tags = data_loader.batch_data()
        # 场景可视化（完整时间步数据）
        if (epoch == 0 or epoch % plot_epoch_interval == 0) and visualizer is not None:
            visualizer.batch_scenario_plot(measurements, unique_ids, trajectories, epoch, single_idx=0, multi_indices=[2, 5, 8, 12, 15, 18])

        # 重置模型状态
        model.reset(batch_size)
        # 梯度清零
        optimizer.zero_grad()

        # 标量损失（用于日志与TensorBoard）
        epoch_losses = {'total': 0.0, 'association':0.0, 'meas': 0.0, 'ce': 0.0, 'gw_known': 0.0, 'gw_birth': 0.0, 'birth': 0.0, 'death': 0.0, 'state': 0.0, 'gw_state': 0.0}
        # 累积损失：这是完整BPTT的关键,不在每个时间步立即backward，而是累积loss张量与计算图保持连接，直到整个序列结束后一次性backward(也可以实现截断BPTT)
        accumulated_loss = torch.tensor(0.0, device=device, requires_grad=True)
        # 记录当前块包含的时间步数
        steps_in_chunk = 0

        # 预测轨迹存储: pred_trajectories[b][target_id] = list of [x,y,vx,vy,θ,a,b,t] (物理坐标)
        # 结构与 data_loader.batch_data() 返回的 trajectories 类似
        pred_trajectories = [{} for _ in range(batch_size)]

        # Scheduled Sampling: 计算当前epoch的TF比率
        if ss_enabled:
            tf_ratio = compute_tf_ratio(epoch, ss_config)
            if epoch % 50 == 0:
                logger.info(f"Epoch {epoch}: tf_ratio = {tf_ratio:.4f}")
        else:
            tf_ratio = 1.0  # 纯TF

        # 上一时间步的模型输出（供Scheduled Sampling使用）
        prev_outputs = None

        # 逐时间步处理
        for t in range(time_steps):
            # 获取当前时间步的数据
            meas_t, gt_t, meas_ids, meas_mask, birth_ids, death_ids, death_states = data_loader.timestep_data()

            # 下一时刻数据，用于状态损失计算
            next_data = data_loader.next_timestep_data()
            if next_data is not None:
                next_meas, next_gt, next_meas_ids, next_meas_mask, next_birth, next_death, next_death_states = next_data
            else:
                next_gt = None

            # 量测预处理
            meas_norm, association_matrix = preprocessor.meas_preprocess(meas=meas_t, gt=gt_t, meas_ids=meas_ids, mode='train')

            # Teacher Forcing训练策略: 所有时间步都使用处理后的真实目标状态（5D: [x,y,θ,a,b]）
            if t == 0:
                # 初始时刻：使用所有真实目标状态（无新生无消亡）
                states_norm, target_ids, target_mask = preprocessor.init_target_node_preprocess(gt=gt_t, add_noise=True)
                death_labels = torch.zeros_like(target_mask, dtype=torch.float, device=device)
            else:
                if args.training_mode=='teacher_forcing':
                    if ss_enabled and tf_ratio < 1.0:
                        # Scheduled Sampling模式：以概率tf_ratio使用GT，以概率1-tf_ratio使用模型预测
                        states_norm, target_ids, target_mask, death_labels = preprocessor.target_node_preprocess_scheduled_sampling(
                            gt=gt_t,
                            birth_ids=birth_ids,
                            death_ids=death_ids,
                            death_states=death_states,
                            prev_outputs=prev_outputs,
                            tf_ratio=tf_ratio,
                            add_noise=True
                        )
                    else:
                        # 纯Teacher Forcing模式（Scheduled Sampling未启用或处于预热阶段）
                        states_norm, target_ids, target_mask, death_labels = preprocessor.target_node_preprocess_for_training(
                            gt=gt_t,
                            birth_ids=birth_ids,
                            death_ids=death_ids,
                            death_states=death_states,
                            add_noise=True
                        )
                if args.training_mode=='evaluating':
                    # 模型验证时在非初始时刻使用模型上一时间步的预测作为当前时间步的输入（暂时还未完成等待后续完成）
                    states_norm, target_ids, target_mask, death_labels = preprocessor.target_node_preprocess(outputs)


            # 前向传播
            outputs = model( meas_norm=meas_norm, states_norm=states_norm, target_ids=target_ids,
                             meas_mask=meas_mask, target_mask=target_mask, meas_ids=meas_ids,
                             is_initial=(t == 0), return_attention=False, update_memory=True)

            # 存储当前输出供下一时间步Scheduled Sampling使用
            prev_outputs = outputs

            # ==================== 计算7D预测状态（含隐式速度）====================
            # 模型输出5D预测状态: [B, N, 5] (归一化: x,y,θ,a,b)
            # 通过与当前输入位置的差分计算速度，组成7D物理状态
            pred_5d_norm = outputs['predicted_states']                    # [B, N, 5] 归一化
            pred_ids_out = outputs['target_ids']                         # [B, N]
            pred_mask_out = outputs['target_mask']                       # [B, N]

            # 反归一化预测5D到物理坐标
            with torch.no_grad():
                pred_5d_phys = preprocessor.state_denorm_5d(pred_5d_norm, pred_mask_out)  # [B, N, 5]
                # 反归一化当前输入5D到物理坐标
                input_5d_phys = preprocessor.state_denorm_5d(states_norm, target_mask)    # [B, S, 5]

            # 逐样本计算速度并构建7D状态、存储预测轨迹
            pred_7d_list = []  # 存储每个样本的预测7D状态 (numpy, 物理坐标)
            for b_idx in range(batch_size):
                B_pred_ids = pred_ids_out[b_idx].detach().cpu()          # [N]
                B_pred_mask = pred_mask_out[b_idx].detach().cpu().numpy().astype(bool)
                B_pred_5d = pred_5d_phys[b_idx].detach().cpu().numpy()   # [N, 5]
                B_input_ids = target_ids[b_idx].detach().cpu()           # [S]
                B_input_mask = target_mask[b_idx].detach().cpu().numpy().astype(bool)
                B_input_5d = input_5d_phys[b_idx].detach().cpu().numpy() # [S, 5]

                N = B_pred_ids.shape[0]
                pred_7d = np.zeros((N, 7), dtype=np.float64)             # [x,y,vx,vy,θ,a,b]
                for i in range(N):
                    if not B_pred_mask[i]:
                        continue
                    pid = B_pred_ids[i].item()
                    # 填入5D: [x,y,θ,a,b] → 7D对应位置
                    pred_7d[i, 0] = B_pred_5d[i, 0]                     # x
                    pred_7d[i, 1] = B_pred_5d[i, 1]                     # y
                    pred_7d[i, 4] = B_pred_5d[i, 2]                     # θ
                    pred_7d[i, 5] = B_pred_5d[i, 3]                     # a
                    pred_7d[i, 6] = B_pred_5d[i, 4]                     # b
                    # 在输入中查找同ID目标，计算速度 v = (pos_pred - pos_input) / Δt
                    matched = (B_input_ids == pid) & torch.tensor(B_input_mask)
                    if matched.any():
                        j = matched.nonzero(as_tuple=True)[0][0].item()
                        pred_7d[i, 2] = (B_pred_5d[i, 0] - B_input_5d[j, 0]) / preprocessor.delta_t  # vx
                        pred_7d[i, 3] = (B_pred_5d[i, 1] - B_input_5d[j, 1]) / preprocessor.delta_t  # vy

                    # 存储到预测轨迹 (从t=1开始，因为预测的是下一时刻状态)
                    if pid >= 0:
                        state_row = np.append(pred_7d[i], t + 1)         # [x,y,vx,vy,θ,a,b,t+1]
                        if pid not in pred_trajectories[b_idx]:
                            pred_trajectories[b_idx][pid] = [state_row]
                        else:
                            pred_trajectories[b_idx][pid].append(state_row)

                pred_7d_list.append(pred_7d)

            # ==================== 可视化当前时间步模型预测输出 ====================
            if (epoch % plot_epoch_interval == 0) and visualizer is not None:
                visualizer.step_output_plot(
                    meas_t=meas_t,                # 当前时间步量测 [B, M, 2]
                    meas_ids=meas_ids,            # 量测真实ID [B, M]
                    meas_mask=meas_mask,          # 量测掩码 [B, M]
                    gt_t=gt_t,                    # 当前时刻GT状态 List[Tensor], [N_b, 8]
                    next_gt=next_gt,              # 下一时刻GT状态 List[Tensor], [N_b, 8] 或 None
                    pred_7d=pred_7d_list,         # 预测7D物理状态 List[ndarray], [N, 7]
                    pred_ids=pred_ids_out,        # 预测目标ID [B, N]
                    pred_mask=pred_mask_out,       # 预测目标掩码 [B, N]
                    outputs=outputs,              # MGAT模型输出字典（用于关联结果）
                    birth_ids=birth_ids,          # 新生目标ID List[Set]
                    death_ids=death_ids,          # 消亡目标ID List[Set]
                    epoch=epoch,
                    timestep=t,
                    batch_idx=0,                  # 选择第0个样本可视化
                    subfolder="step_outputs"
                )

            # 准备损失计算所需数据
            losses_calculate_data = {
                'meas_ids': meas_ids,              # [B, M]，每个量测的真实归属目标ID：杂波=-1，填充=-2，真实目标>=0
                'meas_mask': meas_mask,            # [B, M]，量测有效掩码：True=有效量测，False=填充位置
                'birth_ids': birth_ids,            # 长度B的列表，birth_ids[b]为第b个样本在当前时间步新生的目标ID集合
                'death_labels': death_labels,      # 消亡目标标签：1.0=该目标在当前时刻消亡，0.0=存活
                'input_target_ids': target_ids,    # 模型输入的目标节点ID：填充位置=-1
                'input_target_mask': target_mask,  # 模型输入目标有效掩码：True=有效目标
                'next_gt': next_gt,                # 下一时刻真实状态：next_gt[b]形状[N_b, 8]，8列=[x,y,vx,vy,θ,a,b,id]（损失内部会提取5D）
                'meas_norm_for_gw': meas_norm,     # [B, M, 2]，归一化量测坐标，用于GW损失中的加权统计量计算
                'states_norm_for_gw': states_norm.detach(),  # [B, S, 5]，归一化目标状态（detach, 作为GW参考常量）
                'gt_t': gt_t,                      # List[Tensor]，当前时刻GT状态（物理坐标）
            }

            # 损失计算
            losses = mgatloss(outputs, batch_data=losses_calculate_data, preprocessor=preprocessor, timestep=t)

            # 累积损失（保持计算图连接）
            accumulated_loss = accumulated_loss + losses['total']

            steps_in_chunk += 1

            # TensorBoard记录可视化：每个时间步的各项损失
            if epoch == 0 or epoch % 20 == 0:
                # 记录可视化每一个时间步的各项损失。add_scalars 语法：(主标题, {线名称: Y值}, X值)
                writer.add_scalars('Step_Dynamics/Total_Loss', {f'Epoch_{epoch}': losses['total'].item()}, t)
                writer.add_scalars('Step_Dynamics/Assoc_Loss', {f'Epoch_{epoch}': losses['association'].item()}, t)
                writer.add_scalars('Step_Dynamics/State_Loss', {f'Epoch_{epoch}': losses['state'].item()}, t)
                writer.add_scalars('Step_Dynamics/Meas_Loss', {f'Epoch_{epoch}': losses['meas'].item()}, t)
                writer.add_scalars('Step_Dynamics/CE_Loss', {f'Epoch_{epoch}': losses['ce'].item()}, t)
                writer.add_scalars('Step_Dynamics/GW_Known_Loss', {f'Epoch_{epoch}': losses['gw_known'].item()}, t)
                writer.add_scalars('Step_Dynamics/GW_Birth_Loss', {f'Epoch_{epoch}': losses['gw_birth'].item()}, t)
                writer.add_scalars('Step_Dynamics/Birth_Loss', {f'Epoch_{epoch}': losses['birth'].item()}, t)
                writer.add_scalars('Step_Dynamics/Death_Loss', {f'Epoch_{epoch}': losses['death'].item()}, t)
                writer.add_scalars('Step_Dynamics/GW_State_Loss', {f'Epoch_{epoch}': losses['gw_state'].item()}, t)

            # 累积标量损失（用于epoch级统计）
            for key in epoch_losses:
                epoch_losses[key] += losses[key].item() if torch.is_tensor(losses[key]) else losses[key]

            # 打印进度及当前损失
            if t % 25 == 0:
                logger.info(
                    f"Epoch {epoch}, Step {t}/{time_steps} | "
                    f"Total: {losses['total'].item():.4f} | "
                    f"Meas: {losses['meas'].item():.4f} "
                    f"(CE: {losses['ce'].item():.8f}, "
                    f"GW_k: {losses['gw_known'].item():.8f}, "
                    f"GW_b: {losses['gw_birth'].item():.8f}) | "
                    f"Birth: {losses['birth'].item():.8f} | "
                    f"Death: {losses['death'].item():.8f} | "
                    f"State(GW): {losses['state'].item():.8f}"
                )

            # ==================== TBPTT 截断反向传播逻辑 ====================
            # 判断是否到达截断点或序列的最后一个时间步
            is_truncation_point = (truncation_steps > 0) and ((t + 1) % truncation_steps == 0)
            is_last_step = (t + 1) == time_steps
            if is_truncation_point or is_last_step:
                # 1. 归一化损失 (除以当前块的步数，而不是总步数，将累积的损失除以时间步数，保持梯度幅度稳定)
                accumulated_loss = accumulated_loss / steps_in_chunk
                # 2. BPTT backward（对累积损失一次性反向传播）
                accumulated_loss.backward()
                # 3. 梯度裁剪、返回原始梯度范数并记录
                grad_norm_before = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()
                writer.add_scalar('Training/Gradient_Norm_Before_Clip', grad_norm_before, t)
                # 4. 参数更新
                optimizer.step()
                optimizer.zero_grad()  # 立即清空梯度
                # 5. 分离记忆模块梯度 (Detach Memory)，使得之前时间步存储在 Memory 中的 Embedding 变为常量
                # 下一个块的前向传播可以读取它们，但反向传播会在截断点停止
                model.memory.detach_all()
                # 6. 分离上一时间步模型输出的梯度（TBPTT边界：防止跨块梯度泄漏）
                # 在Scheduled Sampling模式下，prev_outputs中的predicted_states可能被用作
                # 下一时间步的输入。backward()后计算图被释放，必须detach以防止后续使用时报错
                if prev_outputs is not None:
                    prev_outputs = {
                        k: v.detach() if isinstance(v, torch.Tensor) else v
                        for k, v in prev_outputs.items()
                    }
                # 7. 重置块累积变量
                accumulated_loss = torch.tensor(0.0, device=device, requires_grad=True)
                steps_in_chunk = 0
                # 打印 TBPTT 更新日志
                logger.debug(f"Step {t}: TBPTT Update performed. Grad Norm: {grad_norm_before:.4f}")

        # ==================== Epoch结束处理 ====================
        # 学习率调度
        scheduler.step(epoch_losses['total'])
        current_lr = optimizer.param_groups[0]['lr']
        # 计算epoch耗时
        epoch_time = time.time() - epoch_start_time
        # 监控学习率
        writer.add_scalar('Training/Learning_Rate', current_lr, epoch)
        # 监控Scheduled Sampling比率
        if ss_enabled:
            writer.add_scalar('Training/TF_Ratio', tf_ratio, epoch)

        # 计算时间步平均损失
        ave_losses = {key: val / time_steps for key, val in epoch_losses.items()}
        # 平均损失日志和TensorBoard
        logger.info(f"Epoch {epoch}, Average loss: "
                    f"|ave_total: {ave_losses['total']:.4f}"
                    f"|ave_association: {ave_losses['association']:.4f}"
                    f"|ave_state: {ave_losses['state']:.4f}")
        writer.add_scalar('Epoch/AVE_Total', ave_losses['total'], epoch)
        writer.add_scalar('Epoch/AVE_Assoc', ave_losses['association'], epoch)
        writer.add_scalar('Epoch/AVE_State', ave_losses['state'], epoch)
        writer.add_scalar('Epoch/Sum_Total', epoch_losses['total'], epoch)
        writer.add_scalar('Epoch/Sum_Association', epoch_losses['association'], epoch)
        writer.add_scalar('Epoch/Sum_State', epoch_losses['state'], epoch)

        # 打印epoch信息
        tf_info = f" | TF_Ratio: {tf_ratio:.4f}" if ss_enabled else ""
        logger.info(
            f"Epoch {epoch}/{training_params.num_epochs} | "
            f"Time: {epoch_time:.1f}s | "
            f"LR: {current_lr:.6f} | "
            f"Loss: {epoch_losses['total']:.4f}"
            f"(-Assoc: {epoch_losses['association']:.4f}-State: {epoch_losses['state']:.4f})"
            f"{tf_info} | ")

        # ==================== 保存检查点 ====================
        # 保存检查点
        if (epoch + 1) % training_params.save_interval == 0:
            ckpt_path = checkpoints_dir / f'checkpoint_epoch_{epoch + 1}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'avg_losses': ave_losses
            }, ckpt_path)
            logger.info(f"保存检查点: {ckpt_path}")

        # epoch结束处（进入下一个epoch前），监控CPU泄漏
        del measurements, ground_truth, unique_ids, trajectories, birth_tags, death_tags, pred_trajectories
        data_loader.current_batch_data = None  # 让上一批次的numpy缓存可回收
        import gc
        gc.collect()
        logger.info(f"[MEM] epoch {epoch} end   RSS={mem_mb():.1f} MB")

    # ==================== 训练结束 ====================
    # 关闭TensorBoard写入器
    writer.close()
    # 保存最终模型
    final_path = checkpoints_dir / 'final_model.pt'
    torch.save(model.state_dict(), final_path)
    logger.info(f"训练完成，最终模型已保存至: {final_path}")


if __name__ == "__main__":
    main()