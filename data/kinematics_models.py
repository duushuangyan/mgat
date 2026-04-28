"""
运动模型模块 - CV/CA/CT三种基本运动模型
作者: DU | 日期: 2025
"""
import numpy as np
from typing import Dict, Any, Tuple, Optional


class CVModel:
    """匀速直线运动模型：x'=x+vx*dt, y'=y+vy*dt, 速度不变"""
    __slots__ = ('params',)                                                     # 使用__slots__减少内存
    def __init__(self, params: Dict[str, Any]): self.params = params            # 保存参数
    def update(self, state: np.ndarray, dt: float) -> np.ndarray:
        new = state.copy()                                                      # 复制状态避免修改原数组
        new[0] += state[2] * dt                                                 # x = x + vx * dt
        new[1] += state[3] * dt                                                 # y = y + vy * dt
        return new                                                              # 速度保持不变


class CAModel:
    """匀加速直线运动模型：位置和速度都随时间变化"""
    __slots__ = ('params', 'ax', 'ay')
    def __init__(self, params: Dict[str, Any]):
        self.params = params
        cos_d, sin_d = np.cos(params['direction']), np.sin(params['direction'])  # 预计算三角函数
        self.ax, self.ay = params['acceleration'] * cos_d, params['acceleration'] * sin_d  # 加速度分量
    def update(self, state: np.ndarray, dt: float) -> np.ndarray:
        new = state.copy()
        dt2 = 0.5 * dt * dt                                                     # 预计算dt²/2
        new[0] += state[2] * dt + self.ax * dt2                                 # x = x + vx*dt + ax*dt²/2
        new[1] += state[3] * dt + self.ay * dt2                                 # y = y + vy*dt + ay*dt²/2
        new[2] += self.ax * dt                                                  # vx = vx + ax*dt
        new[3] += self.ay * dt                                                  # vy = vy + ay*dt
        return new


class CTModel:
    """匀速转弯运动模型：速度大小不变，方向随时间变化形成圆弧"""
    __slots__ = ('params', 'turn_rate', 'speed')
    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.turn_rate, self.speed = params['turn_rate'], params['speed']       # 转弯率和速度
    def update(self, state: np.ndarray, dt: float) -> np.ndarray:
        new = state.copy()
        v = np.hypot(state[2], state[3])                                        # 当前速度大小
        heading = np.arctan2(state[3], state[2])                                # 当前航向角
        if v < 1e-6: v, heading = self.speed, self.params.get('initial_direction', 0.0)  # 速度为0时使用初始值
        new_heading = heading + self.turn_rate * dt                             # 更新航向角
        cos_h, sin_h = np.cos(new_heading), np.sin(new_heading)                 # 预计算新航向三角函数
        new[2], new[3] = v * cos_h, v * sin_h                                   # 更新速度分量
        new[0] += (state[2] + new[2]) * 0.5 * dt                                # 使用平均速度更新位置x
        new[1] += (state[3] + new[3]) * 0.5 * dt                                # 使用平均速度更新位置y
        return new


# 模型类型到类的映射字典
_MODEL_MAP = {'CV': CVModel, 'CA': CAModel, 'CT': CTModel}


def create_kinematics_model(model_type: str, params: Dict[str, Any]):
    """工厂函数：根据类型创建运动模型实例"""
    if model_type not in _MODEL_MAP: raise ValueError(f"未知模型类型: {model_type}")
    return _MODEL_MAP[model_type](params)


def generate_random_kinematics_model(model_type: str, rng: np.random.Generator,
                                    config: Any, direction: Optional[float] = None) -> Tuple[str, Dict[str, Any]]:
    """生成随机运动模型及参数，支持预设方向"""
    d = direction if direction is not None else rng.uniform(-np.pi, np.pi)      # 使用预设方向或随机生成
    auto = getattr(config, 'auto_targets', None)                                # 获取auto_targets配置
    kin_cfg = getattr(auto, 'kinematics_model', None) if auto else None         # 获取运动模型配置
    if kin_cfg is None: return 'CV', {'type': 'CV', 'speed': 10.0, 'direction': d}  # 无配置使用默认

    # 根据模型类型选择并生成参数
    m_type = getattr(kin_cfg, 'model_type', 'CV_only')                          # 获取模型类型
    chosen = 'CV' if m_type == 'CV_only' else rng.choice(['CV', 'CA', 'CT'])    # CV_only或mixed随机选择

    if chosen == 'CV':
        cv = getattr(kin_cfg, 'cv_model', None)                                 # CV配置
        spd = rng.uniform(getattr(cv, 'min_speed', 5.0), getattr(cv, 'max_speed', 15.0)) if cv else 10.0
        return 'CV', {'type': 'CV', 'speed': spd, 'direction': d}
    elif chosen == 'CA':
        ca = getattr(kin_cfg, 'ca_model', None)                                 # CA配置
        init_spd = rng.uniform(getattr(ca, 'min_initial_speed', 2.0), getattr(ca, 'max_initial_speed', 10.0)) if ca else 5.0
        accel = rng.uniform(getattr(ca, 'min_acceleration', 0.5), getattr(ca, 'max_acceleration', 3.0)) if ca else 1.0
        return 'CA', {'type': 'CA', 'initial_speed': init_spd, 'acceleration': accel, 'direction': d}
    else:  # CT
        ct = getattr(kin_cfg, 'ct_model', None)                                 # CT配置
        spd = rng.uniform(getattr(ct, 'min_speed', 5.0), getattr(ct, 'max_speed', 15.0)) if ct else 10.0
        min_tr, max_tr = getattr(ct, 'min_turn_rate', -0.2), getattr(ct, 'max_turn_rate', 0.2)
        turn_rate = rng.choice([-1, 1]) * rng.uniform(abs(min_tr), abs(max_tr)) if min_tr >= 0 else rng.uniform(min_tr, max_tr)
        return 'CT', {'type': 'CT', 'speed': spd, 'turn_rate': turn_rate, 'initial_direction': d}


def initialize_kinematics_state(model_type: str, params: Dict[str, Any], initial_position: np.ndarray) -> np.ndarray:
    """根据模型类型和参数初始化运动状态[x,y,vx,vy]"""
    x, y = initial_position                                                     # 提取初始位置
    if model_type == 'CV':
        d = params['direction']                                                 # 运动方向
        vx, vy = params['speed'] * np.cos(d), params['speed'] * np.sin(d)       # 速度分量
    elif model_type == 'CA':
        d = params['direction']
        vx, vy = params['initial_speed'] * np.cos(d), params['initial_speed'] * np.sin(d)
    elif model_type == 'CT':
        d = params.get('initial_direction', 0.0)
        vx, vy = params['speed'] * np.cos(d), params['speed'] * np.sin(d)
    else: raise ValueError(f"未知模型类型: {model_type}")
    return np.array([x, y, vx, vy], dtype=np.float64)                           # 返回状态向量