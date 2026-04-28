"""
单目标数据生成模块 - 椭圆扩展目标的创建和管理
作者: DU | 日期: 2025
"""
import numpy as np
from typing import Union, Optional, Any
from data.kinematics_models import create_kinematics_model


class ExtendedTarget:
    """椭圆扩展目标类：运动状态[x,y,vx,vy] + 形状状态[orientation,major,minor]"""

    __slots__ = ('kinematics_state', 'shape_state', 'delta_t', 'id', 'target_config',  # 使用__slots__减少内存占用
                 'rng', '_preset_direction', 'initial_state', 'measurements_history',
                 'marked_for_removal', 'kinematics_model_type', 'kinematics_model_params',
                 'speed', 'direction', 'kinematics_model', 'orientation_mode',
                 'orientation_angle', 'state_history', 'noise_std')

    def __init__(self, movements: Union[list, np.ndarray], shapes: Union[list, np.ndarray],
                 delta_t: float, id: int, target_config: Any = None,
                 rng: Optional[np.random.Generator] = None, preset_direction: Optional[float] = None):
        self.kinematics_state = np.asarray(movements, dtype=np.float64)         # 运动状态向量[x,y,vx,vy]，使用float64保证精度
        self.shape_state = (np.array(shapes) if len(shapes) == 3                # 形状状态：3元素直接使用，2元素补0方向角
                           else np.array([0.0, shapes[0], shapes[1]]))
        self.delta_t, self.id, self.target_config, self.rng = delta_t, id, target_config, rng  # 基本属性一行赋值
        vx, vy = movements[2], movements[3]                                     # 提取速度分量用于方向计算
        self._preset_direction = (preset_direction if preset_direction is not None  # 预设方向优先，否则从速度计算
                                  else (np.arctan2(vy, vx) if vx or vy else None))
        self.initial_state = [self.kinematics_state.copy(), self.shape_state.copy(), 0]  # 保存初始状态副本
        self.measurements_history, self.marked_for_removal = {}, False          # 量测历史和移除标记
        self._init_kinematics()                                                 # 初始化运动学参数
        self.state_history = np.concatenate([self.kinematics_state, self.shape_state, [0]]).reshape(1, -1)  # 状态历史[8列]
        self._init_noise()                                                      # 初始化过程噪声参数

    def _init_noise(self):
        """初始化7维过程噪声标准差向量[x,y,vx,vy,θ,a,b]"""
        self.noise_std = np.zeros(7)                                            # 默认全零噪声
        cfg = getattr(self.target_config, 'data_generation', None)              # 安全获取data_generation配置
        if cfg and hasattr(cfg, 'process_noise'):                               # 存在process_noise配置
            pn = cfg.process_noise                                              # 获取噪声配置对象
            if hasattr(pn, 'kinematic'): self.noise_std[:4] = pn.kinematic      # 运动学噪声[x,y,vx,vy]
            if hasattr(pn, 'orientation'): self.noise_std[4] = pn.orientation   # 方向角噪声
            if hasattr(pn, 'shape'): self.noise_std[5:7] = pn.shape             # 形状噪声[major,minor]

    def _init_kinematics(self):
        """初始化运动模型参数，支持CV/CA/CT三种模型"""
        auto_cfg = getattr(self.target_config, 'auto_targets', None)            # 获取auto_targets配置
        kin_cfg = getattr(auto_cfg, 'kinematics_model', None) if auto_cfg else None  # 获取运动模型配置
        model_type = getattr(kin_cfg, 'model_type', 'CV_only') if kin_cfg else 'CV_only'  # 模型类型，默认CV_only
        direction = self._preset_direction if self._preset_direction is not None else self.rng.uniform(-np.pi, np.pi)  # 运动方向
        model_cfg = {'CV_only': 'cv_model', 'mixed': self.rng.choice(['cv_model', 'ca_model', 'ct_model'])}  # 模型选择映射
        chosen = model_cfg.get(model_type, 'cv_model') if model_type == 'CV_only' else model_cfg['mixed']  # 选择具体模型
        cfg_map = {'cv_model': self._cfg_cv, 'ca_model': self._cfg_ca, 'ct_model': self._cfg_ct}  # 配置函数映射
        cfg_map.get(chosen, self._cfg_cv)(kin_cfg, direction)                   # 调用对应配置函数
        self.kinematics_model = create_kinematics_model(self.kinematics_model_type, self.kinematics_model_params)  # 创建模型实例
        self._init_heading(auto_cfg)                                            # 初始化方向角控制模式

    def _cfg_cv(self, kin_cfg, direction):
        """配置CV匀速直线模型"""
        cv = getattr(kin_cfg, 'cv_model', None) if kin_cfg else None            # 获取CV模型配置
        speed = self.rng.uniform(getattr(cv, 'min_speed', 5.0), getattr(cv, 'max_speed', 15.0)) if cv else 10.0  # 随机速度
        self.kinematics_model_type, self.kinematics_model_params = 'CV', {'speed': speed, 'direction': direction}  # 模型类型和参数
        cos_d, sin_d = np.cos(direction), np.sin(direction)                     # 预计算三角函数避免重复计算
        self.kinematics_state[2:4] = speed * cos_d, speed * sin_d               # 设置速度分量vx,vy
        self.speed, self.direction = speed, direction                           # 保存速度和方向

    def _cfg_ca(self, kin_cfg, direction):
        """配置CA匀加速模型"""
        ca = getattr(kin_cfg, 'ca_model', None) if kin_cfg else None            # 获取CA模型配置
        init_spd = self.rng.uniform(getattr(ca, 'min_initial_speed', 2.0), getattr(ca, 'max_initial_speed', 10.0)) if ca else 5.0
        accel = self.rng.uniform(getattr(ca, 'min_acceleration', 0.5), getattr(ca, 'max_acceleration', 3.0)) if ca else 1.0
        self.kinematics_model_type = 'CA'                                       # 模型类型CA
        self.kinematics_model_params = {'initial_speed': init_spd, 'acceleration': accel, 'direction': direction}
        cos_d, sin_d = np.cos(direction), np.sin(direction)                     # 预计算三角函数
        self.kinematics_state[2:4] = init_spd * cos_d, init_spd * sin_d         # 设置初始速度分量
        self.speed, self.direction = init_spd, direction                        # 保存速度和方向

    def _cfg_ct(self, kin_cfg, direction):
        """配置CT匀速转弯模型"""
        ct = getattr(kin_cfg, 'ct_model', None) if kin_cfg else None            # 获取CT模型配置
        speed = self.rng.uniform(getattr(ct, 'min_speed', 5.0), getattr(ct, 'max_speed', 15.0)) if ct else 10.0
        turn_rate = self.rng.uniform(getattr(ct, 'min_turn_rate', -0.2), getattr(ct, 'max_turn_rate', 0.2)) if ct else 0.1
        self.kinematics_model_type = 'CT'                                       # 模型类型CT
        self.kinematics_model_params = {'speed': speed, 'turn_rate': turn_rate, 'initial_direction': direction}
        cos_d, sin_d = np.cos(direction), np.sin(direction)                     # 预计算三角函数
        self.kinematics_state[2:4] = speed * cos_d, speed * sin_d               # 设置速度分量
        self.speed, self.direction = speed, direction                           # 保存速度和方向

    def _init_heading(self, auto_cfg):
        """初始化方向角控制模式：auto跟随运动方向，manual使用固定值"""
        heading_cfg = getattr(auto_cfg, 'heading', None) if auto_cfg else None  # 获取heading配置
        self.orientation_mode = getattr(heading_cfg, 'mode', 'auto') if heading_cfg else 'auto'  # 方向模式
        if self.orientation_mode == 'manual' and hasattr(heading_cfg, 'angle'):  # 手动模式设置固定方向角
            self.orientation_angle = heading_cfg.angle                          # 保存手动方向角
            self.shape_state[0] = heading_cfg.angle                             # 设置形状方向角
        elif self.orientation_mode == 'auto':                                   # 自动模式跟随运动方向
            self.shape_state[0] = self.direction                                # 方向角=运动方向

    def update_state(self, t: int) -> np.ndarray:
        """更新目标状态，返回8维向量[x,y,vx,vy,θ,a,b,t]"""
        new_kin = self.kinematics_model.update(self.kinematics_state.copy(), self.delta_t)  # 运动模型更新
        self.speed = np.hypot(new_kin[2], new_kin[3])                            # 更新速度大小，hypot比sqrt(x²+y²)更快更准
        self.direction = np.arctan2(new_kin[3], new_kin[2])                     # 更新运动方向
        new_shape = self.shape_state.copy()                                     # 复制形状状态
        new_shape[0] = (self.direction if self.orientation_mode == 'auto'       # 自动模式：方向角=运动方向
                       else getattr(self, 'orientation_angle', self.shape_state[0]))  # 手动模式：使用固定值
        if self.rng is not None:                                                # 存在RNG则注入过程噪声
            noise = self.rng.normal(0, self.noise_std)                          # 生成7维高斯噪声
            new_kin += noise[:4]                                                # 运动状态加噪声
            new_shape += noise[4:]                                              # 形状状态加噪声
            new_shape[1:3] = np.maximum(0.1, new_shape[1:3])                    # 确保长短轴为正，向量化max
        return np.concatenate([new_kin, new_shape, [int(t)]])                   # 返回完整8维状态向量

    def update_measurements(self, t: int) -> np.ndarray:
        """生成当前时间步的雷达量测点[N,3]，每行为[range,angle,target_id]"""
        meas_cfg = self.target_config.measurement                               # 获取量测配置
        num_pts = max(meas_cfg.min_points, self.rng.poisson(meas_cfg.avg_points))  # 泊松采样点数，下限min_points
        center, orientation, axes = self.kinematics_state[:2], self.shape_state[0], self.shape_state[1:3]  # 提取椭圆参数
        points = self._gen_ellipse_pts_vectorized(center, orientation, axes, num_pts)  # 向量化生成椭圆内部点
        ranges = np.hypot(points[:, 0], points[:, 1])                           # 批量计算距离，hypot更高效
        angles = np.arctan2(points[:, 1], points[:, 0])                         # 批量计算角度
        ranges += self.rng.normal(0, meas_cfg.range_noise, num_pts)             # 添加距离噪声
        angles += self.rng.normal(0, meas_cfg.angle_noise, num_pts)             # 添加角度噪声
        x_coords, y_coords = ranges * np.cos(angles), ranges * np.sin(angles)   # 转换回笛卡尔坐标检查边界
        fov_ub = self.target_config.simulation.field_of_view_ub                 # 视场上界
        valid = (x_coords >= 0) & (y_coords >= 0) & (x_coords <= fov_ub) & (y_coords <= fov_ub) & (ranges > 0)  # 有效性掩码
        ranges, angles = ranges[valid], angles[valid]                           # 过滤无效点
        result = np.column_stack([ranges, angles, np.full(len(ranges), self.id)]) if len(ranges) > 0 else np.zeros((0, 3))  # 构建结果
        self.measurements_history[t] = result                                   # 保存到历史
        return result

    def _gen_ellipse_pts_vectorized(self, center: np.ndarray, orientation: float,
                                    axes: np.ndarray, n: int) -> np.ndarray:
        """向量化生成椭圆内部均匀分布点（优化版：避免拒绝采样循环）"""
        # 使用极坐标均匀采样：r=sqrt(U), θ=2πV，其中U,V~Uniform(0,1)
        # 这样点在单位圆内均匀分布，无需拒绝采样
        r = np.sqrt(self.rng.uniform(0, 1, n))                                  # 径向采样，sqrt保证均匀分布
        theta = self.rng.uniform(0, 2*np.pi, n)                                 # 角度均匀采样
        unit_pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])      # 单位圆内的点[n,2]
        scaled = unit_pts * axes                                                # 缩放到椭圆[n,2]，广播axes
        cos_o, sin_o = np.cos(orientation), np.sin(orientation)                 # 预计算旋转矩阵元素
        rotated = np.column_stack([scaled[:, 0]*cos_o - scaled[:, 1]*sin_o,     # 旋转x分量
                                   scaled[:, 0]*sin_o + scaled[:, 1]*cos_o])    # 旋转y分量
        return rotated + center                                                 # 平移到椭圆中心

    def __repr__(self) -> str:
        return f'ExtendedTarget(ID={self.id}, pos=[{self.kinematics_state[0]:.1f},{self.kinematics_state[1]:.1f}])'